# `persistent_overlap_pad8_prefetch_stage0` optimization performance history

This file is the running performance record for candidates derived from
`persistent_overlap_pad8_prefetch_stage0.s`. All accepted measurements use the
assembly GEMM1 kernel inside the full MoE e2e pipeline. A candidate is retained
only when random-data e2e validation passes; const0 equality alone is not a
sufficient correctness result.

## Measurement protocol

- Machine: `d01-3`
- Workload: E96, T16384, top-k 6, K7168, intermediate dimension 3072, a4w4,
  SiLU, no bias.
- A/ScaleA producer: `three_kernel` unless a row says otherwise.
- Timing source: GEMM1 profiler time from the full MoE e2e run.
- Comparison command:

```bash
ROUNDS=3 RUN_VERIFY=1 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

The GPU/KFD ownership check must be run immediately before and after every
remote GPU test. Machine performance can change after a reboot or reconnect,
so every new candidate run must include `persistent_overlap_pad8_prefetch_stage0`
in the same interleaved benchmark invocation.

## Starting point: 2026-09-13 d01-3

These measurements were supplied from a fresh three-round d01-3 run and define
the optimization starting point for subsequent work.

| data | case | grid | GEMM1 samples (us) | GEMM1 median us | GEMM1 vs first case | MoE e2e samples (us) | MoE e2e median us | MoE e2e vs first case | pass | logits_diff | rel_l2 | MoE output hash128 | ref output hash128 |
|---|---|---|---|---:|---:|---|---:|---:|:---:|---:|---:|---|---|
| const0 | baseline | standard | 764.821, 763.077, 765.354 | 764.821 | +0.00% | 1643.33, 1642.22, 1642.61 | 1642.61 | +0.00% | True | 0.0000e+00 | 0.0000e+00 | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| const0 | optimized_v1 | standard | 589.946, 590.887, 582.744 | 589.946 | +22.86% | 1471.52, 1474.78, 1465.33 | 1471.52 | +10.42% | True | 0.0000e+00 | 0.0000e+00 | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| const0 | double_lds | standard | 563.325, 566.047, 559.836 | 563.325 | +26.35% | 1447.76, 1451.29, 1443.87 | 1447.76 | +11.86% | True | 0.0000e+00 | 0.0000e+00 | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| const0 | persistent | 16x16 | 571.675, 568.866, 568.904 | 568.904 | +25.62% | 1453.73, 1452.87, 1451.95 | 1452.87 | +11.55% | True | 0.0000e+00 | 0.0000e+00 | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| const0 | persistent_overlap | 16x16 | 553.938, 558.759, 559.170 | 558.759 | +26.94% | 1439.74, 1441.81, 1443.03 | 1441.81 | +12.22% | True | 0.0000e+00 | 0.0000e+00 | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| const0 | persistent_overlap_pad8 | 16x16 | 553.257, 549.408, 552.872 | 552.872 | +27.71% | 1436.76, 1432.00, 1435.40 | 1435.40 | +12.61% | True | 0.0000e+00 | 0.0000e+00 | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| const0 | persistent_overlap_pad8_prefetch_stage0 | 16x16 | 528.899, 530.670, 530.756 | 530.670 | +30.62% | 1410.94, 1411.81, 1410.65 | 1410.94 | +14.10% | True | 0.0000e+00 | 0.0000e+00 | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |

The direct optimization baseline is therefore:

```text
persistent_overlap_pad8_prefetch_stage0 GEMM1 median = 530.670 us
persistent_overlap_pad8_prefetch_stage0 MoE median   = 1410.94 us
```

## Candidate results

New runs are appended here with the exact command, machine state, correctness
result, hashes, samples, medians, and change relative to the same-run stage-0
baseline.

### `I0,O0,O1,I1` delayed stage-1 prefetch

Kernel:

```text
persistent_overlap_pad8_prefetch_stage0_delayed_stage1.s
SHA256=f81d3203e41e891fdd759f35ec9c783329f8a6e91e8510258bb0c21089f05746
```

The candidate issues next-task stage 0 before the SiLU epilogue, preserves both
current-task output transfers ahead of stage 1, waits for stage 0 locally, and
then issues next-task stage 1. The resulting per-wave order is
`I0,O0,O1,I1`, with at most three TDM operations outstanding.

Random e2e smoke comparison on d01-3:

| data | case | GEMM1 us | MoE e2e us | pass | logits_diff | rel_l2 | MoE output hash128 | ref output hash128 |
|---|---|---:|---:|:---:|---:|---:|---|---|
| random | `persistent_overlap_pad8_prefetch_stage0` | 641.451 | 1679.63 | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |
| random | delayed stage 1 | 643.481 | 1687.69 | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |

Three-round interleaved const0 comparison:

| data | case | GEMM1 samples (us) | GEMM1 median us | change vs same-run stage-0 | MoE e2e samples (us) | MoE e2e median us | pass | MoE output hash128 | ref output hash128 |
|---|---|---|---:|---:|---|---:|:---:|---|---|
| const0 | `persistent_overlap_pad8_prefetch_stage0` | 533.947, 529.007, 532.654 | 532.654 | baseline | 1416.32, 1412.91, 1414.42 | 1414.42 | True | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| const0 | delayed stage 1 | 532.372, 536.787, 532.964 | 532.964 | -0.06% | 1415.50, 1416.79, 1418.31 | 1416.79 | True | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260913T132337Z_e2e-const0/
```

Decision: correctness passed, but the same-run GEMM1 median regressed by
`0.310 us` (`0.06%`). Keep the file as an analyzed experiment and retain
`persistent_overlap_pad8_prefetch_stage0.s` as the optimization baseline.

### B64 accumulator clear plus full SQC instruction prefetch

Kernel:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full.s
SHA256=6d3b387137b1f5d3de0e95fa8dddb8067e0dae9847ecf773fdc50b5d5ba28bb3
```

Changes relative to the stage-0 starting point:

- Replace 2,560 scalar `v_mov_b32_e32` accumulator clears with 1,280
  `v_mov_b64_e32` clears.
- Let logical wave 0 issue 12 `s_prefetch_inst` operations covering the
  approximately 48 KiB shader footprint.
- Keep all input/output TDM operations, barriers, waits, WMMA operations,
  DS operations, ABI, LDS allocation, and the 16x16 persistent grid unchanged.

Random MoE e2e validation repeatedly passed with:

```text
logits_diff = 3.3980e-06
rel_l2      = 2.6069e-03
MoE hash    = 1556fc617347e2dabc9cff19dbfd822b
ref hash    = 1a5d22911ba167160b4f2c12092a5193
```

Three-round paired const0 comparison:

| case | GEMM1 samples (us) | GEMM1 median us | change vs stage-0 | MoE e2e median us | pass |
|---|---|---:|---:|---:|:---:|
| `persistent_overlap_pad8_prefetch_stage0` | 529.840, 527.828, 534.381 | 529.840 | baseline | 1417.08 | True |
| B64 clear + full instruction prefetch | 511.108, 514.393, 510.328 | 511.108 | +3.54% | 1395.23 | True |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260913T175617Z_e2e-const0/
```

Decision: accepted as the current stable winner. Relative to the original
user-supplied `530.670 us` starting point, the median is lower by `19.562 us`
or `3.69%`.

### ATT progression through the accepted winner

ATT result directory:

```text
history_runs/heliosr-1b114-d01-3_20260913T173018Z_att/
```

| case | steady task cycles | dispatch cycles | full-wave cycles |
|---|---:|---:|---:|
| stage-0 starting point | 26,716.3 | 1,086,017 | 1,078,699.8 |
| B64 clear | 26,621.9 | 1,043,300 | 1,033,135.5 |
| B64 clear + full instruction prefetch | 26,239.7 | 961,587 | 953,050.5 |

For the accepted winner, the remaining measured waits were approximately:

| group | cycles/task | task share |
|---|---:|---:|
| barrier wait | 3,613.5 | 13.77% |
| `DScnt` wait | 1,883.8 | 7.18% |
| `TENSORcnt` wait | 1,204.0 | 4.59% |

The dominant cluster-barrier site remained about `2,042 cycles/task`, so the
next large optimization target is owner-wave arrival skew rather than the
already shortened accumulator clear.

### Cooperative instruction-prefetch owner sweep

The 48 KiB SQC prefetch was distributed across two or three waves while all
compute and memory operations remained unchanged.

Random e2e smoke results:

| case | GEMM1 us | MoE e2e us | pass | logits_diff | rel_l2 |
|---|---:|---:|:---:|---:|---:|
| one-wave full prefetch | 619.342 | 1666.26 | True | 3.3980e-06 | 2.6069e-03 |
| two-wave distributed prefetch | 621.573 | 1675.85 | True | 3.3980e-06 | 2.6069e-03 |
| one-wave full prefetch, second run | 621.594 | 1672.14 | True | 3.3980e-06 | 2.6069e-03 |
| three-wave distributed prefetch | 621.214 | 1667.44 | True | 3.3980e-06 | 2.6069e-03 |

Three-round const0 comparison for the best screening candidate:

| case | GEMM1 samples (us) | GEMM1 median us | change | MoE samples (us) | MoE median us | change |
|---|---|---:|---:|---|---:|---:|
| one-wave full prefetch | 514.037, 512.424, 513.265 | 513.265 | baseline | 1402.63, 1399.26, 1398.01 | 1399.26 | baseline |
| three-wave distributed prefetch | 517.902, 512.321, 513.063 | 513.063 | +0.04% | 1401.93, 1398.12, 1399.90 | 1399.90 | -0.05% |

Result directories:

```text
history_runs/heliosr-1b114-d01-3_20260913T180618Z_e2e-random/
history_runs/heliosr-1b114-d01-3_20260913T180700Z_e2e-random/
history_runs/heliosr-1b114-d01-3_20260913T180909Z_e2e-const0/
```

Decision: reject both cooperative-prefetch variants. The three-wave result is
within run noise and slightly worsens MoE e2e time.

### B-owner launch-position sweep on the accepted winner

The B-owner stage-0/stage-1 TDM issue was moved across the independent B64
accumulator-clear block. All candidates produced exact standalone const0
output. Two-round standalone screening results were:

| B64 clears before B launch | median us |
|---:|---:|
| source schedule | 491.024 |
| 0 | 492.458 |
| 32 | 494.445 |
| 64 | 496.556 |
| 96 | 493.720 |
| 128 | 492.578 |
| 160 | 498.327 |
| 192 | 495.775 |

Decision: reject the sweep. None beat the source schedule in the same
standalone run, so no e2e promotion was justified.

### WPT2 FlyDSL reference injection

The generated WPT2 reference was renamed only for injection; its code and
184-byte ABI were otherwise unchanged. It passed random MoE e2e correctness,
but was substantially slower than the persistent assembly winner:

| case | grid | GEMM1 us | MoE e2e us | pass |
|---|---|---:|---:|:---:|
| B64 clear + full instruction prefetch | 16x16 | 621.438 | 1669.27 | True |
| FlyDSL WPT2 reference | standard | 700.557 | 1710.95 | True |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260913T181941Z_e2e-random/
```

Decision: do not replace the current hotloop with the full WPT2 schedule. Its
balanced TDM ownership remains useful as a reference for a minimal port.

### All-SIMD owner-wave trace of the accepted winner

Trace directory:

```text
history_runs/winner_b64_full_allsimd_20260913/
```

The trace used `--all-simd --ana-att`. Logical owners were identified from the
first full-setup path in each captured wave. The physical SIMD selection did
not equal the logical wave number for every owner.

| logical owner | ATT SIMD selection | task mean cycles | barrier wait | `DScnt` wait | `TENSORcnt` wait |
|---|---|---:|---:|---:|---:|
| A | SIMD0 | 26,780.7 | 3,668.1 | 1,873.3 | 1,260.4 |
| B | SIMD3 | 26,619.8 | 3,940.6 | 1,478.0 | 996.8 |
| ScaleA | SIMD2 | 26,511.0 | 4,490.8 | 1,547.8 | 558.2 |
| ScaleB | SIMD1 | 25,971.9 | 4,119.7 | 1,178.1 | 546.4 |

The scale owners spend substantially more time in barriers and much less time
in TDM waits. This supports a minimal A/B payload split, but it does not justify
copying the slower WPT2 compute schedule.

### Persistent 16-cluster grid-shape experiment

MI400 Shader Programming Guide section 3.5.5.1 defines `TTMP7[15:0]` as
Cluster ID Y and `TTMP9` as Cluster ID X. The correct flattened physical
cluster ID is therefore:

```text
cluster_x + num_cluster_x * cluster_y
```

An initial experimental generator used the opposite multiplier. Const0 could
not detect the resulting missing/duplicated tasks and reported invalid low
times; random e2e failed with `logits_diff=8.8240e-01` and
`rel_l2=9.6826e-01`. Those timings are excluded.

After correcting the flattening formula, every shape passed random e2e with the
standard hashes and error metrics:

| grid | GEMM1 us | MoE e2e us | pass |
|---|---:|---:|:---:|
| 64x4 | 616.610 | 1659.43 | True |
| 32x8 | 620.748 | 1669.79 | True |
| 8x32 | 618.364 | 1664.39 | True |
| 4x64 | 621.179 | 1670.11 | True |

The best random screening shape, 64x4, was compared against 16x16 for three
const0 rounds:

| grid | GEMM1 samples (us) | GEMM1 median us | change | MoE samples (us) | MoE median us | change |
|---|---|---:|---:|---|---:|---:|
| 16x16 | 514.309, 514.944, 511.475 | 514.309 | baseline | 1395.20, 1396.86, 1393.96 | 1395.20 | baseline |
| 64x4 | 513.409, 513.400, 512.943 | 513.400 | +0.18% | 1394.10, 1392.63, 1395.35 | 1394.10 | +0.08% |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260914T023102Z_e2e-const0/
```

Decision: keep 16x16 as the stable default. The 64x4 change is correct but its
`0.18%` GEMM1 difference is too small to distinguish from machine noise.

### First minimal owner-balance prototype

Prototype:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_owner_balance.s
SHA256=81f379e34061305e286c0a833413c25e1850775dc0a3fbbad1bacfc37bd9ef76
```

The prototype preserved the DS/WMMA layout, halved the original A/B TDM outer
extent, and issued the second A/B half from the former ScaleA/ScaleB owner with
a second descriptor in `s80:s91`. Static assembly and invariant checks passed.
The first random e2e launch produced a GPU page fault before correctness could
be evaluated. The candidate is rejected and must not be benchmarked. Follow-up
diagnosis is split into A-only and B-only variants before any further combined
test.

### a07-3 performance starting point

The machine had not rebooted, but the baseline was measured once after moving
the experiment from d01-3 so that later a07-3 candidates have a local reference.
Both kernels passed random MoE e2e:

| case | GEMM1 us | MoE e2e us | pass | logits_diff | rel_l2 |
|---|---:|---:|:---:|---:|---:|
| stage-0 starting point | 625.160 | 1602.49 | True | 3.3980e-06 | 2.6069e-03 |
| B64 clear + full instruction prefetch | 603.635 | 1580.10 | True | 3.3980e-06 | 2.6069e-03 |

Three-round const0 comparison:

| case | GEMM1 samples (us) | GEMM1 median us | change | MoE samples (us) | MoE median us | change |
|---|---|---:|---:|---|---:|---:|
| stage-0 starting point | 519.294, 511.077, 517.730 | 517.730 | baseline | 1353.09, 1344.27, 1348.83 | 1348.83 | baseline |
| B64 clear + full instruction prefetch | 482.485, 482.551, 488.452 | 482.551 | +6.79% | 1316.05, 1316.11, 1320.63 | 1316.11 | +2.43% |

Result directories:

```text
history_runs/heliosr-1b114-a07-3_20260914T025916Z_e2e-random/
history_runs/heliosr-1b114-a07-3_20260914T030123Z_e2e-const0/
```

The a07-3 starting point for subsequent comparisons is therefore
`482.551 us` for the current stable winner.

### Owner-balance descriptor diagnosis

The first six-descriptor prototype was split into A-only and B-only variants.
The descriptors were changed to the unbounded WPT2 shape while retaining the
same half-payload source and LDS offsets.

The A-only variant passed full random MoE e2e:

| case | GEMM1 us | MoE e2e us | pass | logits_diff | rel_l2 |
|---|---:|---:|:---:|---:|---:|
| current winner | 604.331 | 1587.04 | True | 3.3980e-06 | 2.6069e-03 |
| A-only owner balance | 672.865 | 1638.47 | True | 3.3980e-06 | 2.6069e-03 |

Result directory:

```text
history_runs/heliosr-1b114-a07-3_20260914T030810Z_e2e-random/
```

The A descriptor and half-payload offsets are therefore correct, but adding an
extra A TDM to the old ScaleA owner with the WPT1 wait schedule regresses GEMM1
by `11.34%`.

The B-only variant did not complete. Its process remained in uninterruptible
`D` state after `Ctrl+C` and `SIGTERM`, holding two KFD queues and approximately
8.6 GiB VRAM. No timing from that run is valid. This result indicates that the
six-descriptor cross-owner ordering is unsafe for the B multicast path.

A replacement candidate now follows the exact production WPT2 owner order:

```text
wave 0/1: A half, then ScaleA half
wave 2/3: B half, then ScaleB half

persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_wpt2_owner.s
SHA256=026946add17eb4f29d4f107afebdaa1d8b1f8bc1fa28b624d4e6121f94a13af6
```

Its static audit passes with 120 `tensor_load_to_lds` sites, unchanged
WMMA/DS/barrier/wait counts, unchanged 184-byte ABI, 1024 VGPRs, 104 numbered
SGPRs, and 320 KiB LDS. GPU validation is pending until the stuck a07-3 process
is cleared without resetting the GPU from this workflow.

## Invalidated d01-3 pre-reboot measurement (2026-09-14 04:54 UTC)

This section is retained only as an audit record. The machine rebooted after
these measurements, and the user explicitly invalidated them. Do not use the
`575.477 us` result below as a performance starting point or compare later
candidates against it.

The d01-3 machine performance changed again, so the stage-0 baseline and the
current stable winner were remeasured in the same run before continuing kernel
optimization. GPU/KFD was idle before and after the measurements. The reported
clocks at the beginning of the const0 run were:

```text
FCLK   = 1950 MHz
MCLK   = 1900 MHz
GFXCLK = 2355 MHz
SOCCLK = 1350 MHz
```

Selected cases:

```text
persistent_overlap_pad8_prefetch_stage0.s
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full.s
```

The random e2e correctness command was:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full \
  AITER_HISTORY_E2E_ROUNDS=1 RUN_VERIFY=1 RUN_ATT=0 \
  bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

| case | GEMM1 us | MoE e2e us | pass | logits_diff | rel_l2 | MoE output hash128 | ref output hash128 |
|---|---:|---:|:---:|---:|---:|---|---|
| `persistent_overlap_pad8_prefetch_stage0` | 708.572 | 1803.84 | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |
| `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full` | 690.574 | 1792.04 | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |

The three-round const0 command was:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full \
  AITER_HISTORY_E2E_ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
  bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

| case | GEMM1 samples (us) | GEMM1 median us | change vs stage-0 | MoE e2e samples (us) | MoE e2e median us | change vs stage-0 | pass | MoE output hash128 | ref output hash128 |
|---|---|---:|---:|---|---:|---:|:---:|---|---|
| `persistent_overlap_pad8_prefetch_stage0` | 597.816, 590.569, 591.756 | 591.756 | baseline | 1564.79, 1559.35, 1557.27 | 1559.35 | baseline | True | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full` | 575.477, 576.734, 571.688 | 575.477 | +2.75% | 1541.67, 1541.93, 1535.44 | 1541.67 | +1.13% | True | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |

Result directories on d01-3:

```text
history_runs/heliosr-1b114-d01-3_20260914T045447Z_e2e-random/
history_runs/heliosr-1b114-d01-3_20260914T045525Z_e2e-const0/
```

This pre-reboot starting point is invalid.

## d01-3 post-reboot performance starting point (2026-09-14 06:16 UTC)

The baseline was measured again after the d01-3 reboot. GPU/KFD was idle before
and after both benchmark commands. The reported clocks at the beginning of the
const0 run were:

```text
FCLK   = 1950 MHz
MCLK   = 1900 MHz
GFXCLK = 2354 MHz
SOCCLK = 1350 MHz
```

The random e2e correctness command was:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full \
  AITER_HISTORY_E2E_ROUNDS=1 RUN_VERIFY=1 RUN_ATT=0 \
  bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

| case | GEMM1 us | MoE e2e us | change vs stage-0 | pass | logits_diff | rel_l2 | MoE output hash128 | ref output hash128 |
|---|---:|---:|---:|:---:|---:|---:|---|---|
| `persistent_overlap_pad8_prefetch_stage0` | 645.808 | 1692.17 | baseline | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |
| `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full` | 620.108 | 1668.96 | +3.98% GEMM1, +1.37% MoE | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |

The three-round const0 command was:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full \
  AITER_HISTORY_E2E_ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
  bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

| case | GEMM1 samples (us) | GEMM1 median us | change vs stage-0 | MoE e2e samples (us) | MoE e2e median us | change vs stage-0 | pass | MoE output hash128 | ref output hash128 |
|---|---|---:|---:|---|---:|---:|:---:|---|---|
| `persistent_overlap_pad8_prefetch_stage0` | 535.402, 530.649, 534.497 | 534.497 | baseline | 1420.05, 1412.88, 1418.17 | 1418.17 | baseline | True | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |
| `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full` | 512.404, 510.664, 511.361 | 511.361 | +4.33% | 1396.82, 1392.55, 1392.69 | 1392.69 | +1.80% | True | 21291d9023c8af8a6324fe20f346a967 | 21291d9023c8af8a6324fe20f346a967 |

Result directories on d01-3:

```text
history_runs/heliosr-1b114-d01-3_20260914T061621Z_e2e-random/
history_runs/heliosr-1b114-d01-3_20260914T061833Z_e2e-const0/
```

For subsequent d01-3 experiments after this reboot, use `511.361 us` as the
stable-winner GEMM1 const0 starting point. Each candidate must still be compared
against both stage-0 and the stable winner in the same benchmark run.

## Post-reboot ATT baseline

Three independent SIMD3-select ATT captures were collected for the stable
winner after the reboot:

```text
history_runs/heliosr-1b114-d01-3_20260914T062011Z_att/
history_runs/heliosr-1b114-d01-3_20260914T062118Z_att/
history_runs/heliosr-1b114-d01-3_20260914T062230Z_att/
```

| capture | steady task mean cycles | p50 | p90 | dispatch cycles |
|---|---:|---:|---:|---:|
| `20260914T062011Z` | 26,471.6 | 25,861.5 | 28,205.5 | 972,404 |
| `20260914T062118Z` | 26,851.8 | 26,162.5 | 29,385.8 | 977,948 |
| `20260914T062230Z` | 26,571.7 | 26,325.0 | 27,396.3 | 966,629 |

Median-of-capture aggregate:

| wait group | cycles/task | task share |
|---|---:|---:|
| `s_barrier_wait` | 3,706.0 | 13.95% |
| `s_wait_dscnt` | 1,433.8 | 5.40% |
| `s_wait_tensorcnt` | 1,268.0 | 4.77% |
| `s_wait_idle` | 14.3 | 0.05% |

The dominant sites are `0x998c s_barrier_wait 0xfffd` at 2,049.5 cycles/task,
`0x9408 s_wait_dscnt 0x8` at 491.8 cycles/task, and
`0x2f94 s_wait_tensorcnt 0x1` at 438.0 cycles/task.

An all-SIMD capture was also collected at:

```text
history_runs/post_reboot_winner_allsimd_20260914/
```

| logical owner | SIMD | task mean cycles | all waits | non-wait cycles | barrier | DScnt | TENSORcnt |
|---|---|---:|---:|---:|---:|---:|---:|
| A | SIMD0 | 26,654.1 | 6,792.1 | 19,862.0 | 3,721.1 | 1,888.3 | 1,068.8 |
| B | SIMD3 | 26,851.7 | 6,726.1 | 20,125.7 | 3,621.3 | 1,625.4 | 1,465.0 |
| ScaleA | SIMD2 | 27,470.8 | 7,660.7 | 19,810.1 | 5,237.4 | 2,000.8 | 412.8 |
| ScaleB | SIMD1 | 26,580.5 | 6,456.0 | 20,124.5 | 4,373.6 | 1,521.7 | 545.7 |

The scale owners spend more time waiting at barriers, while B has the largest
TDM wait and B/ScaleB have the largest non-wait work. This makes B-side latency
the useful target; reducing a scale-owner barrier wait by delaying that owner
would not shorten the cluster critical path.

Machine-readable outputs:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_thread_trace_metrics_post_reboot_20260914.json
post_reboot_winner_allsimd_owner_metrics_20260914.json
```

## Post-reboot scheduling and cache-hint experiments

### Dominant DScnt scheduling

The `0xA5C8` source wait maps to post-build PC `0x9408` and costs about
`491.8 cycles/task`. Several schedules were tested:

| candidate | random correctness | const0 result | decision |
|---|---|---|---|
| `dscnt8_late` | failed: `rel_l2=1.1896e-02`, hash mismatch | not run | rejected; the crossed WMMA has a real LDS dependency |
| `dscnt8_after_wmma1` | failed: `rel_l2=1.0361e-02`, hash mismatch | not run | rejected |
| `dscnt12_then8` | exact standard random result/hash | `513.243 us` vs `508.159 us` (`-1.00%`) | rejected; the extra wait costs more than the overlap saves |

The rejected `dscnt12_then8` result directory is:

```text
history_runs/heliosr-1b114-d01-3_20260914T065740Z_e2e-const0/
```

### Input TDM temporal hints

The CDNA5 load-hint definitions used here are `NT_RT` (near-cache
non-temporal, far-cache regular) and `NT_HT` (near-cache non-temporal,
far-cache high-priority temporal).

| candidate | validation | same-run const0 result | decision |
|---|---|---|---|
| B/ScaleB `HT` | random exact | 3-round `515.844 us` vs `514.783 us` (`-0.21%`) | rejected |
| B/ScaleB `NT_HT` | random exact | 1-round `511.008 us` vs `513.466 us` (`+0.48%`) | below promotion threshold |
| A/ScaleA `NT_RT` | random exact | 1-round `515.205 us` vs `517.164 us` (`+0.38%`) | below promotion threshold |
| output store `WB` | random exact | 1-round `519.319 us` vs `512.829 us` (`-1.27%`) | rejected |
| output store `HT` | random exact | 1-round `513.380 us` vs `512.640 us` (`-0.14%`) | rejected |

Combining the near/far policy across all 60 input TDM instructions produced a
repeatable improvement:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s
SHA256=8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169
```

| case | GEMM1 samples (us) | GEMM1 median us | change | MoE samples (us) | MoE median us | change | pass |
|---|---|---:|---:|---|---:|---:|:---:|
| stable winner | 519.518, 514.685, 515.427 | 515.427 | baseline | 1411.21, 1405.92, 1409.90 | 1409.90 | baseline | True |
| all input TDM `NT_RT` | 509.435, 511.989, 510.413 | 510.413 | +0.97% | 1401.78, 1404.60, 1402.83 | 1402.83 | +0.50% | True |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260914T075219Z_e2e-const0/
```

The mixed A/ScaleA=`NT_RT`, B/ScaleB=`NT_HT` version was also correct, but its
three-round result was slightly weaker: `511.161 us` versus same-run
`515.759 us` (`+0.89%`). The all-`NT_RT` version is therefore the current
candidate winner.

### Early steady TDM issue

Moving the four B/ScaleB steady TDM instructions immediately after their
workgroup barriers passed random e2e. Moving all eight A/ScaleA/B/ScaleB TDM
instructions caused a GPU hang and is rejected. The B/ScaleB-only change has
completed an idle three-round comparison after the next reconnect:

| case | GEMM1 samples (us) | GEMM1 median us | change | MoE samples (us) | MoE median us | change | pass |
|---|---|---:|---:|---|---:|---:|:---:|
| all input TDM `NT_RT` | 502.749, 505.033, 505.024 | 505.024 | baseline | 1390.27, 1389.17, 1391.29 | 1390.27 | baseline | True |
| all `NT_RT` + early B/ScaleB TDM | 506.016, 508.255, 504.965 | 506.016 | -0.20% | 1394.80, 1392.93, 1394.06 | 1394.06 | -0.27% | True |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260914T084746Z_e2e-const0/
```

Decision: reject the early B/ScaleB issue combination. It remains correct but
does not improve either the GEMM1 or fused-MoE median.

## d01-3 reconnect performance starting point (2026-09-14 08:42 UTC)

After `d01-3` became reachable again, all three comparison kernels were
remeasured in the same process and machine state. The GPU was idle before the
performance run.

Random e2e validation:

| case | GEMM1 us | MoE e2e us | pass | logits_diff | rel_l2 | MoE output hash128 | ref output hash128 |
|---|---:|---:|:---:|---:|---:|---|---|
| stage-0 | 640.844 | 1683.02 | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |
| B64 clear + full instruction prefetch | 620.398 | 1668.79 | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |
| all input TDM `NT_RT` | 621.184 | 1669.22 | True | 3.3980e-06 | 2.6069e-03 | 1556fc617347e2dabc9cff19dbfd822b | 1a5d22911ba167160b4f2c12092a5193 |

Three-round const0 comparison:

| case | GEMM1 samples (us) | GEMM1 median us | change vs stage-0 | MoE samples (us) | MoE median us | change vs stage-0 | pass |
|---|---|---:|---:|---|---:|---:|:---:|
| stage-0 | 527.657, 533.439, 531.622 | 531.622 | baseline | 1407.50, 1419.12, 1415.73 | 1415.73 | baseline | True |
| B64 clear + full instruction prefetch | 509.346, 510.901, 511.161 | 510.901 | +3.90% | 1393.71, 1395.47, 1396.14 | 1395.47 | +1.43% | True |
| all input TDM `NT_RT` | 506.203, 506.124, 506.347 | 506.203 | +4.78% | 1391.09, 1392.73, 1390.72 | 1391.09 | +1.74% | True |

Result directories:

```text
history_runs/heliosr-1b114-d01-3_20260914T084225Z_e2e-random/
history_runs/heliosr-1b114-d01-3_20260914T084345Z_e2e-const0/
```

The current same-machine starting point is therefore `506.203 us` for
`persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s`.

## all-`NT_RT` ATT refresh (2026-09-14 08:50 UTC)

Three independent SIMD3-select traces were captured for the new winner:

```text
history_runs/heliosr-1b114-d01-3_20260914T085039Z_att/
history_runs/heliosr-1b114-d01-3_20260914T085212Z_att/
history_runs/heliosr-1b114-d01-3_20260914T085339Z_att/
```

| capture | steady task mean cycles | p50 | p90 | dispatch cycles |
|---|---:|---:|---:|---:|
| `20260914T085039Z` | 26,522.3 | 26,100.5 | 27,868.9 | 969,627 |
| `20260914T085212Z` | 26,536.9 | 26,154.5 | 27,830.0 | 955,718 |
| `20260914T085339Z` | 26,896.7 | 26,667.0 | 28,543.9 | 970,406 |

Median-of-capture wait breakdown:

| wait group | cycles/task | task share |
|---|---:|---:|
| `s_barrier_wait` | 3,362.5 | 12.67% |
| `s_wait_dscnt` | 1,523.2 | 5.74% |
| `s_wait_tensorcnt` | 1,410.6 | 5.32% |
| `s_wait_idle` | 155.4 | 0.59% |

The dominant static sites are the six K-ring `s_barrier_wait 0xfffd`
occurrences at `1,959.5 cycles/task`, the dominant `s_wait_dscnt 0x8` at
`531.4 cycles/task`, and the persistent-boundary `s_wait_tensorcnt 0x1` at
`486.7 cycles/task`.

An all-SIMD capture was collected at:

```text
history_runs/all_nt_rt_allsimd_20260914/winner_all_nt_rt_allsimd/
```

| logical owner | task mean cycles | barrier | DScnt | TENSORcnt | `s_wait_idle` |
|---|---:|---:|---:|---:|---:|
| A | 26,360.5 | 3,590.5 | 1,790.6 | 1,056.3 | 85.4 |
| B | 27,888.0 | 3,962.4 | 1,605.2 | 1,612.0 | 432.1 |
| ScaleA | 26,762.5 | 4,381.1 | 1,663.6 | 855.9 | 9.0 |
| ScaleB | 27,306.6 | 4,714.6 | 1,878.6 | 548.8 | 15.1 |

B is the measured critical owner. Its unusually large final `s_wait_idle`
exposure motivated the next experiment: move independent output descriptor,
address, and workgroup-barrier setup ahead of the all-counter drain while
keeping the drain before `global_prefetch_b8` and next-task TDM issue.

Machine-readable results:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_thread_trace_metrics_20260914.json
all_nt_rt_allsimd_owner_metrics_20260914.json
```

## `s_wait_idle` output-setup overlap experiment

Generated candidates:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_wait_idle_after_output_setup.s
SHA256=08278bb157999ec27dce4a644415d22da7756bc01b893523b581a59c34369408

persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_wait_idle_b_deferred.s
SHA256=b0e9be3c14aef6747e21b5f5497856097130ffdcf202277c9f79af3bbaff9c8a
```

Both candidates assemble for gfx1250 and pass random MoE e2e with the standard
hashes and error metrics. Single-round screening was:

| case | GEMM1 us | MoE e2e us | change | pass |
|---|---:|---:|---:|:---:|
| all input TDM `NT_RT` | 629.687 | 1686.73 | baseline | True |
| wait after common output setup | 624.800 | 1674.82 | +0.78% GEMM1 | True |
| all input TDM `NT_RT`, second run | 623.689 | 1675.20 | baseline | True |
| B-owner wait deferred through next descriptor | 623.187 | 1674.89 | +0.08% GEMM1 | True |

The common output-setup overlap candidate advanced to a three-round const0
comparison. The run did not reach the candidate: its first, unmodified
all-`NT_RT` baseline launch remained inside the GPU call for more than 11
minutes after printing the normal pipeline preflight line. No timing from that
run is valid. The process was left untouched because this workflow does not
kill GPU processes or reset/reboot the machine without user authorization.

Two follow-up old-output guard candidates were generated and assembled, but
were not launched while that process still owned the GPU:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_b_output_guard2.s
SHA256=560834f406b0ffa47941bf356d9f1833da117a8d82b8a4148b77d86a2f0729db

persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_all_output_guard2.s
SHA256=1832dd2a1da75827061c51edbe20805b2cc6dd40809e55fb689d25acd34d1f9b
```

These change selected prefetched-task old-output guards from
`s_wait_tensorcnt 0x1` to `s_wait_tensorcnt 0x2`. They require random MoE e2e
validation before any performance result can be accepted because the relaxed
guard may permit an old output TDM to overlap stage-2 input LDS reuse.

## Four-owner A+B quarter / split-scale experiment (2026-09-15)

The experiment distributes the stage payload work as follows, using the
validated physical owner order `q0,q2,q1,q3`:

```text
owner 0: A q0 + B q0 + ScaleA half 0
owner 1: A q2 + B q2 + ScaleA half 1
owner 2: A q1 + B q1 + ScaleB half 0
owner 3: A q3 + B q3 + ScaleB half 1
```

Every owner issues its requests in `A -> B -> Scale` order.  The resident
implementation keeps the primary payload descriptor in `s32:s43`, the scale
descriptor in `s80:s91`, and the opposite payload descriptor in `s44:s55`.
The opposite global address is rebased from the current primary address using
a task-local 64-bit delta in `s104:s105`.  The LDS address is derived from the
current primary ring with the exact A/B ring deltas (`0x30000` for rings 0/1
and `0x2e000` for rings 2/3).

Several correctness bugs were isolated before timing:

- Saving next-task coordinates in `s92:s93` corrupted the current output LDS
  address held in `s93`.
- Using `s100` as tail descriptor-mask scratch corrupted current output state.
- Building a B-ring LDS address as `base + A_RING[ring]` overran ring 2/3 by
  `0x2000`; one path reached `0x50000`, exactly beyond the 320 KiB LDS limit.
- Advancing the resident opposite pointer by `0x800` during a prefetched-task
  setup and then rebasing it from an already advanced primary pointer skipped
  one K stage.

The conservative, correct version disables input TDM prefetch between
persistent tasks, advances through full task setup, and retimes the WPT1
wait thresholds for three input requests per stage:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_ab4_scale_half_tdm_full_setup_loop_wait6.s
SHA256=02dffd3a7f6ca015a25d52e8a2273c6cd49deed57bde4f0f758c1a97c7c70c06
```

Three consecutive random MoE e2e runs passed.  The observed error varied only
within the normal reference range:

```text
run 1: logits_diff=3.3980e-06, rel_l2=2.6069e-03
run 2: logits_diff=5.4192e-06, rel_l2=3.2922e-03
run 3: logits_diff=4.0503e-06, rel_l2=2.8461e-03
```

An idle three-round const0 comparison produced:

| case | GEMM1 samples (us) | GEMM1 median us | change | MoE samples (us) | MoE median us | change | pass |
|---|---|---:|---:|---|---:|---:|:---:|
| all input TDM `NT_RT` | 506.609, 507.269, 506.795 | 506.795 | baseline | 1393.08, 1396.86, 1395.67 | 1395.67 | baseline | True |
| resident A+B quarter / split Scale, full setup, wait6 | 533.038, 536.743, 568.717 | 536.743 | -5.91% | 1422.35, 1422.34, 1457.18 | 1422.35 | -1.91% | True |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260915T095128Z_e2e-const0/
```

For comparison, a correct implementation that reconstructs the complete
opposite descriptor at every issue measured `998.773 us` versus a same-run
`507.305 us` baseline.  It was rejected because descriptor copies and eight
`v_readfirstlane_b32` operations in the hot loop nearly doubled GEMM1 time.

The resident 2+3 transformation therefore does not beat the current winner in
its verified form.  Early next-task input TDM variants remain diagnostic only:
their TDM writes can overlap the current output LDS lifetime and have produced
NaNs or a stalled launch.  No timing from those failing runs is valid.

### Scalar-only persistent transition

To separate next-task address preparation from unsafe early LDS reuse, the
following candidate advances the next logical task and all expert-local base
pointers during the current epilogue, but does not issue next-task input TDMs:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_ab4_scale_half_tdm_state_only_tail_wait6.s
SHA256=900f945b45572975079df28149d132c9478a2829aba1e1c3e9d96731c3bffd02
```

It passed three consecutive random MoE e2e runs:

```text
run 1: logits_diff=1.3821e-05, rel_l2=5.2575e-03
run 2: logits_diff=3.5693e-05, rel_l2=8.4490e-03
run 3: logits_diff=1.5025e-05, rel_l2=5.4818e-03
```

An idle three-round const0 comparison produced:

| case | GEMM1 samples (us) | GEMM1 median us | change | MoE samples (us) | MoE median us | change | pass |
|---|---|---:|---:|---|---:|---:|:---:|
| all input TDM `NT_RT` | 504.560, 505.964, 507.155 | 505.964 | baseline | 1388.83, 1391.76, 1391.86 | 1391.76 | baseline | True |
| resident 2+3, scalar-only tail, wait6 | 537.906, 536.639, 537.166 | 537.166 | -6.17% | 1425.23, 1426.14, 1424.94 | 1425.23 | -2.40% | True |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260915T104150Z_e2e-const0/
```

The scalar-only transition removes repeated full task setup, but does not
recover the cost of issuing three TDM operations per owner per K stage.  This
confirms that the remaining regression is in the 2+3 hotloop itself rather
than persistent-loop bookkeeping.

### Low-overhead alternatives on the current winner

The output-boundary and scalar-zero candidates were then compared for nine
interleaved const0 rounds on an idle `d01-3`:

| case | GEMM1 samples (us) | GEMM1 median us | change | MoE median us | change | pass |
|---|---|---:|---:|---:|---:|:---:|
| all input TDM `NT_RT` | 504.344, 506.522, 505.671, 506.668, 505.970, 505.740, 506.984, 507.218, 504.690 | 505.970 | baseline | 1392.37 | baseline | True |
| wait after common output setup | 505.871, 505.837, 505.272, 505.395, 507.081, 505.449, 503.838, 507.488, 507.315 | 505.837 | +0.03% | 1393.32 | -0.07% | True |

The `0.03%` GEMM1 difference is measurement noise, so the wait relocation is
not promoted.  A separate nine-round run also found no measurable benefit from
output-descriptor hoisting or packed SGPR-zero initialization:

| case | GEMM1 median us | change | MoE median us | change | pass |
|---|---:|---:|---:|---:|:---:|
| all input TDM `NT_RT` | 505.799 | baseline | 1392.13 | baseline | True |
| output descriptor hoist | 505.924 | -0.02% | 1391.85 | +0.02% | True |
| packed SGPR zero initialization | 505.837 | -0.01% | 1391.70 | +0.03% | True |

Result directories:

```text
history_runs/heliosr-1b114-d01-3_20260915T105126Z_e2e-const0/
history_runs/heliosr-1b114-d01-3_20260915T110115Z_e2e-const0/
```

The B-only and all-owner `s_wait_tensorcnt 0x1 -> 0x2` variants passed random
e2e but did not improve the three-round const0 median.  The WMMA inline-zero
accumulator experiment failed random e2e (`logits_diff=4.7469e-01`) and is
rejected.  Applying production-style `6/4` waits to the WPT2 owner kernel also
produced NaNs, so those relaxed waits are not used.

## Complete resident 2+3 next-task tail (2026-09-15)

The complete-tail experiment first exposed two distinct synchronization
requirements. Multicast completion is local to each requesting wave: after
every wave waits for its own request, a workgroup barrier is sufficient to
publish the local LDS data to the four consumer waves. A second cluster barrier
is not required at the next-task stage-0 consumer. The existing cluster
barriers remain at persistent task and four-phase transitions.

The current output allocation can, however, occupy the stage-0 A/B rings for
this K-ring phase. An A-only early-tail diagnostic failed random e2e, while a
Scale-only diagnostic passed. Therefore the accepted implementation does not
write A/B stage 0 before the current output TDM has finished reading LDS.

The final schedule is:

```text
current task descriptor/address preparation
  -> build next-task A, B, and Scale descriptors
  -> preserve primary A/B descriptor in s56:s67
  -> preserve opposite A/B descriptor in s68:s79
  -> issue the disjoint Scale tail
  -> current SiLU and output O0/O1
  -> s_wait_tensorcnt 0
  -> workgroup barrier
  -> issue next-task A -> B
  -> existing persistent cluster boundary
next task
  -> s_wait_tensorcnt 0
  -> workgroup barrier
  -> consume stage 0
```

No new cluster barrier was added. The descriptor tail words temporarily reused
as next-task coordinates are saved through `v250:v251` and restored before the
opposite descriptor is copied to `s68:s79`. Scale remains early because its LDS
range is disjoint from the current output allocation. Moving Scale behind the
output drain produced an invalid descriptor lifetime and was rejected.

Final files:

```text
build_ab_quarter_scale_half_tdm_variant.py
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_ab4_scale_half_tdm_full_tail_wg_sync.s
```

Final SHA256 values:

```text
build_ab_quarter_scale_half_tdm_variant.py:
2e256605eb296aa7d98fd990fc842a82f9217e00413ffd34d0335494635f6f28

persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_ab4_scale_half_tdm_full_tail_wg_sync.s:
a3ba3e41dbfd6c0aa7f42c3fb3622e085bbbf2fe67798fdc0a0aed0a6e27e92d
```

Random MoE e2e validation passed for three seeds:

```text
seed 0: logits_diff=5.3662e-06, rel_l2=3.2760e-03
seed 1: logits_diff=4.2335e-05, rel_l2=9.2015e-03
seed 2: logits_diff=5.9792e-06, rel_l2=3.4581e-03
```

The idle d01-3 three-round const0 comparison was:

| case | GEMM1 samples (us) | GEMM1 median us | MoE samples (us) | MoE median us | pass |
|---|---|---:|---|---:|:---:|
| all input TDM `NT_RT` | 509.522, 506.817, 508.080 | 508.080 | 1395.72, 1391.72, 1392.94 | 1392.94 | True |
| resident 2+3, full setup, wait6 | 559.642, 530.967, 530.330 | 530.967 | 1445.74, 1419.21, 1417.38 | 1419.21 | True |
| resident 2+3, complete tail, WG-local retirement | 619.078, 616.826, 617.261 | 617.261 | 1506.07, 1502.51, 1502.28 | 1502.51 | True |

Result directory:

```text
history_runs/heliosr-1b114-d01-3_20260915T145523Z_e2e-const0/
```

The complete tail is now correct, but it is not a performance winner. Its
`617.261 us` median is 16.25% slower than the same-run `530.967 us` 2+3
full-setup version. Retiring O0/O1 before issuing the overlapping A/B payload
serializes the output-to-input transition, and the saved-descriptor copies add
scalar work. A faster complete tail requires rotating the prefetched K stage
into a ring that is disjoint from the current output allocation, rather than
writing the fixed stage-0 A/B ring before output retirement.
