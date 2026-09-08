# Optimization results

## Implementation

The optimized assembly keeps the production launch contract:

```text
grid=(2880,4,1)
cluster=(4,4,1)
block=(128,1,1)
kernarg=184 bytes
```

It changes only the isolated copy under this directory:

- restores the eight original 4x4 TDM multicast descriptors;
- retains `s_wait_tensorcnt 0x0` after the first output
  `tensor_store_from_lds` before LDS reuse;
- swaps each accumulator group from `G0,U0,G1,U1` to contiguous gate/up pairs;
- uses `v_pk_mul_f32`, `v_dual_min_num_f32`, and `v_dual_add_f32`;
- double-buffers Trans32 temporaries and prepares the next eight outputs while
  the current eight outputs execute `RCP/EXP`.

Static epilogue comparison:

| Metric | Safe baseline | Optimized |
|---|---:|---:|
| Output-tail instructions | 2399 | 1903 |
| `v_exp_f32` | 256 | 256 |
| `v_rcp_f32` | 256 | 256 |
| scalar `v_mul_f32` occurrences | 770 | 2 |
| `v_pk_mul_f32` | 0 | 384 |
| `v_dual_*` operations | 0 | 512 |
| `v_swap_b32` | 0 | 128 |

## Correctness

Machine: `b8-3` (`heliosr-2b805-b8-3.aus-b200.dcgpu`).

Random standalone was run three times. Every run reported:

```text
checkAllclose passed
gemm_a4w4 err = 0
output hash = dda7e24e10f23cac0d1c02e448bec71f
rel_l2 = 0.0009367674611323901
```

A 50-launch CUDA-event run also passed with the same output hash:

```text
iterations = 50
device_time_avg = 666.461 us
gemm_a4w4 err = 0
```

Additional random seeds also passed:

```text
seed=1: err=0, rel_l2=0.00093677390526811
seed=2: err=0, rel_l2=0.0009367715837032677
```

Const0 also reported exact equality:

```text
gemm_a4w4 err = 0
max_err_info = (0.0, 0.0, 0.0)
rel_l2 = 0.0
```

## Performance

The user explicitly requested testing on `b8-3` without excluding other GPU
processes, so these numbers can contain system-load noise.

| Data | Timing method | Time |
|---|---|---:|
| random | 20-iteration torch profiler | `657.980 us` |
| const0 | 20-iteration torch profiler | `515.603 us` |
| const0 | single CUDA-event sample | `526.023 us` |

The isolated safe baseline was measured with the same copied runner on the
same `b8-3` checkout:

| Data | Safe baseline | Optimized | Reduction |
|---|---:|---:|---:|
| random | `914.002 us` | `657.980 us` | `256.022 us` (`28.01%`) |
| const0 | `696.739 us` | `515.603 us` | `181.136 us` (`26.00%`) |

The original raw-output kernel was also measured on `b8-3` with const0:

```text
moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps = 538.402 us
```

Despite performing `Silu(gate) * up`, the optimized kernel's const0 result
(`515.603 us`) is approximately `22.799 us` faster than that raw-output
reference in this run. The raw kernel writes 6144 BF16 columns, while the
activated kernel writes 3072 BF16 columns.

Reference measurements from the controlled decomposition on `b8-2`:

| Variant | const0 CUDA-event time |
|---|---:|
| Original raw 4x1 kernel | `565.250 us` |
| Safe act1, independent loads | `709.842 us` |
| Multicast act1, old scalar SiLU | `576.877 us` |
| Multicast act1, no activation | `501.592 us` |

The optimized packed/pipelined epilogue therefore removes most of the SiLU
overhead while retaining the required output TensorCnt wait.

The existing repository files were intentionally not replaced. The older
`b8-3` checkout does not contain the complete current quant/injection stack, so
this isolated optimization was validated with the standalone 184-byte ABI
runner rather than by modifying its e2e Python modules.

## Commands

```bash
python my_code/moe_gemm1_act1_optimized/build_optimized.py

bash my_code/moe_gemm1_act1_optimized/test_optimized.sh quick-random
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-random
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-const0
```
