# E64/T1536/topk8 t256/w2x4 persistent optimization

## Target workload

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

All performance measurements were collected on `a07-3` in the `hyg_fyd1`
container after confirming that the GPU was idle.  The implementation worktree
was:

```text
/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
```

## Adjacent t192 reference

The reference used the same code and environment, with only the exact tuned CSV
row changed back to `tile_m=tile_m2=192`.

| Metric | Five samples (us) | Median (us) |
|---|---|---:|
| GEMM1 | 73.779, 76.079, 74.762, 73.026, 74.970 | 74.762 |
| GEMM2 | 53.806, 54.527, 54.270, 50.158, 50.038 | 53.806 |
| Fused MoE | 182.37, 191.28, 192.90, 181.59, 182.15 | 182.37 |

Log:

```text
/tmp/current_t192_r5_late.log
```

## Initial t256 implementation

The first correct t256 implementation executed one N task per workgroup and did
not yet reuse a persistent workgroup across N tiles.

| Metric | Three samples (us) | Median (us) |
|---|---|---:|
| GEMM1 | 98.580, 98.977, 100.746 | 98.977 |
| GEMM2 | 92.227, 89.514, 93.596 | 92.227 |
| Fused MoE | 248.27, 247.03, 251.32 | 248.27 |

Log:

```text
/tmp/t256_single_task_const0_r3.log
```

## Retained optimizations

### 1. Extend both persistent launchers to t256/w2x4/b4

Both launchers in
`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py`
now accept the external `t256x256x256/w2x4/b4` launch geometry.  The exact tuned
CSV row routes GEMM1 and GEMM2 to this tile.

GEMM1 keeps four persistent N tasks per workgroup.  GEMM2 keeps seven persistent
N tasks per workgroup.  This amortizes block prologue work, expert lookup, and
the repeated launch-side setup across all N tiles for one expert tile.

### 2. Fix the t256 persistent drain carry

The constexpr-unrolled t256 drain loses the carried K128 register state across
later drain iterations.  For drain iterations after the first, the kernel now
reloads K128-0 from LDS instead of consuming an invalid carried value.  This is
required for random-input correctness when the persistent outer loop is active.

### 3. Write fused ScaleA in the row32 A-preshuffle layout

The initial t256 fused GEMM1 implementation wrote ScaleA with the ordinary
WMMA-interleaved address formula.  GEMM2's A-preshuffle path consumes ScaleA in
the row32-interleaved layout:

```text
row_tile32 = grouped_row // 32
row_in_tile32 = grouped_row % 32
dst_dword = row_tile32 * (scale_dwords_per_row * 32)
          + scale_dword * 32
          + row_in_tile32
```

With 192 valid rows in a 256-row tile, the old formula made GEMM2 read zero or
unrelated padding scales for expert-local rows 128 through 191.  The corrected
direct store uses the row32 formula above.  After the fix, GEMM2 valid rows are
bitwise identical to the standalone producer path.

### 4. Use a 192-row compute geometry inside the balanced t256 launch tile

For this exact balanced workload, every expert owns 192 valid rows while the
external expert stride remains 256 rows.  The retained specialization therefore
keeps:

```text
launch tile_m = 256
expert stride = 256
compute tile_m = 192
```

This avoids allocating accumulators and issuing WMMA/DS work for the 64 padding
rows.  Internally, the two `wave_m` groups each process 96 rows, matching the
balanced t192 compute geometry, while global addressing and the GEMM grid retain
the t256 expert layout.

The specialization is selected only when all of the following hold:

- `tile_m == 256`
- `n_experts == 64`
- `contiguous_m == 28672`, which identifies this E64/T1536/topk8 route arena
- `AITER_MOE_EXPERT_BALANCE=true`

Other inputs retain the generic t256 geometry.  The balanced flag is passed as a
compile-time argument to the persistent launchers, so the kernel module does not
read test configuration directly.

### 5. Preserve generic fallbacks

- E64/T2048/topk8, where every expert has 256 valid rows, uses the full t256
  compute geometry and passed byte/bitwise validation.
- With `AITER_MOE_EXPERT_BALANCE=false`, the t256/w2x4 persistent specialization
  is not selected; the existing generic fallback passed the normal random MoE
  correctness test.
- The t192 path retains its original compile-time geometry and behavior.

## Final performance

The final three-round const0 result is:

| Metric | Samples (us) | Median (us) | Versus initial t256 | Versus adjacent t192 |
|---|---|---:|---:|---:|
| GEMM1 | 71.452, 76.252, 73.360 | 73.360 | 25.88% faster | 1.88% faster |
| GEMM2 | 47.462, 51.938, 47.563 | 47.563 | 48.43% faster | 11.60% faster |
| Fused MoE | 175.46, 188.44, 178.94 | 178.94 | 27.93% faster | 1.88% faster |

Log:

```text
/tmp/t256_final_const0_r3.log
```

The longer five-round run showed normal machine variance but remained close to
the t192 reference:

| Metric | Five samples (us) | Median (us) |
|---|---|---:|
| GEMM1 | 77.406, 74.842, 76.658, 74.296, 73.967 | 74.842 |
| GEMM2 | 51.980, 50.283, 53.062, 48.236, 51.904 | 51.904 |
| Fused MoE | 186.89, 182.96, 185.71, 180.90, 184.33 | 184.33 |

Log:

```text
/tmp/t256_vm192_m28672_const0_r5.log
```

## Correctness validation

Random target-shape validation:

```text
pass=True
logits_diff=3.38491e-06
rel_l2=0.00260189
```

GEMM2 reference and output hashes were identical after restoring the required
t256 drain reload behavior.

Full-256 balanced fallback (`tokens=2048`) also passed:

```text
payload semantic mismatch = 0
ScaleA mismatch = 0
GEMM2 valid-row bitwise mismatch = 0
pass=True
logits_diff=3.3833e-06
rel_l2=0.0026013
```

The non-balanced fallback passed the standard MoE test:

```text
pass=True
logits_diff=3.47335e-06
rel_l2=0.00263566
```

## Rejected experiments

The following changes were removed because they did not provide stable benefit
or did not preserve exact output behavior:

- forcing full A/ScaleA loads and full output stores for partial-M tiles;
- zero-initializing the complete producer/output arenas;
- changing t256 `WMMA_REUSE` from 3 to 1;
- changing GEMM2 to `b3` with two cached A stages;
- runtime `96+96` row redistribution while retaining eight accumulator rows;
- conditionally skipping A LDS loads for inactive accumulator rows;
- staging t256 ScaleA through LDS/TDM instead of the retained direct row32 store;
- shrinking the balanced grouped arena below the normal worst-case size.

## Nine-sample t192 versus t256 retest

The shorter three-round comparison above was sensitive to machine-to-machine
and run-to-run variation.  A later retest executed the following command three
separate times for each tile, producing nine raw samples per version:

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

The median below is taken over all nine raw values directly, rather than over
the three per-command medians.

### t192 raw samples

```text
GEMM1: 70.264, 72.674, 74.641, 73.302, 74.496, 73.083, 76.809, 73.705, 71.552
GEMM2: 50.310, 48.978, 49.005, 51.103, 53.764, 52.113, 55.944, 51.017, 50.085
MoE:   181.54, 179.50, 187.96, 185.79, 188.37, 185.54, 190.61, 183.11, 179.13
```

### t256 raw samples

```text
GEMM1: 74.032, 76.653, 73.425, 74.917, 77.985, 76.990, 76.010, 76.341, 76.623
GEMM2: 50.293, 54.257, 49.231, 52.341, 55.010, 53.384, 48.353, 51.908, 51.963
MoE:   181.92, 188.23, 180.82, 189.00, 193.09, 188.82, 183.15, 189.60, 188.02
```

### Nine-sample comparison

| Metric | t192 median (us) | t256 median (us) | t256 versus t192 |
|---|---:|---:|---:|
| GEMM1 | 73.302 | 76.341 | 4.15% slower |
| GEMM2 | 51.017 | 51.963 | 1.85% slower |
| Fused MoE | 185.54 | 188.23 | 1.45% slower |

Logs:

```text
/tmp/compare_t192_run1.log
/tmp/compare_t192_run2.log
/tmp/compare_t192_run3.log
/tmp/compare_t256_run1.log
/tmp/compare_t256_run2.log
/tmp/compare_t256_run3.log
```

This nine-sample result supersedes the earlier three-sample claim that t256 was
faster.  The t256 persistent implementation is substantially faster than the
initial t256 implementation, but it has not yet matched t192 on this workload.
