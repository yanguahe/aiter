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

## Post-reconnect relative baseline (2026-09-09)

The b8-3 host was restarted and its absolute performance changed.  All numbers
in this section were therefore remeasured after the reconnect.  GPU utilization
was already 100% because another user's workload was present, as explicitly
allowed for this task; only same-run relative comparisons are used.

The reproducible comparison entry point is:

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-const0
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-random
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

It always compares this stable chain:

| Label | ISA SHA256 |
|---|---|
| safe baseline | `3f40709bc179544c163b2fd80e8cffc253c261646caf659340ecaad5cfb0d681` |
| optimized v1 | `c90bbe6221b03ae6e05d1d5570b16be25217133e4105860eac901043f316feae` |
| double-output-LDS candidate | `33034bb76ada72f9fbc895c1edbf56ee339d0ed6eb28d496196b68323ceed331` |

An additional candidate can be appended to the same interleaved run with
`AITER_HISTORY_CANDIDATE=/path/to/candidate.s`.

### Standalone comparison

| Data | Safe baseline | Optimized v1 | Double output LDS | v2 vs v1 |
|---|---:|---:|---:|---:|
| random | `1017.893 us` | `886.096 us` | `876.202 us` | `-1.12%` |
| const0 | `910.636 us` | `811.222 us` | `797.913 us` | `-1.64%` |

The double-output-LDS candidate removes the wait immediately after the first
output `tensor_store_from_lds`.  Banks 2 and 3 stage into the adjacent 8 KiB
wave-local LDS slice, so their writes cannot overwrite bytes still consumed by
the first store.  Ten consecutive random validations were bit-identical to the
optimized-v1 output.

### Full MoE e2e comparison

| Data | Version | GEMM1 | Fused MoE | Correctness |
|---|---|---:|---:|---|
| random | safe baseline | `1018.402 us` | `2148.37 us` | `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03`, pass |
| random | optimized v1 | `883.667 us` | `2023.62 us` | `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03`, pass |
| random | double output LDS | `870.683 us` | `2007.91 us` | `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03`, pass |
| const0 | optimized v1 | `825.712 us` | `1923.98 us` | `logits_diff=0`, `rel_l2=0`, pass |
| const0 | double output LDS | `812.408 us` | `1888.64 us` | `logits_diff=0`, `rel_l2=0`, pass |

### ATT cycle comparison

Captured by `benchmark_att_history.sh` after the reconnect:

| Version | ATT maximum GFXCLK cycles | ATT wall time | Mean GFXCLK |
|---|---:|---:|---:|
| safe baseline | `2,062,938` | `959.040 us` | approximately `2.151 GHz` |
| optimized v1 | `1,830,700` | `852.500 us` | approximately `2.147 GHz` |
| double output LDS | `1,783,185` | `833.160 us` | approximately `2.139 GHz` |

The double-output-LDS change reduces the measured dispatch span by `47,515`
GFXCLK cycles (`2.60%`) relative to optimized v1 and by `279,753` cycles
(`13.56%`) relative to the safe baseline.  It still measures `3.291x` the
requested `541,900.8`-cycle target.

The corresponding logs are under:

```text
my_code/moe_gemm1_act1_optimized/history_runs/
my_code/moe_gemm1_act1_optimized/att_history/20260909_101629/
```

## a07-3 reproduction of the historical 515.603 us version

The historical `515.603 us` result was measured on b8-3 with
`moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s`, SHA256
`c90bbe6221b03ae6e05d1d5570b16be25217133e4105860eac901043f316feae`.

After confirming that a07-3 had no GPU/KFD users, the updated history script
first reproduced the exact command:

```bash
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-const0
```

Result:

```text
device_time_avg = 522.5756 us
```

The same invocation then ran the three-version, same-input interleaved CUDA
event comparison:

| Version | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|
| independent-load safe baseline | `704.784 us` | `704.389 us` | `698.839 us` | `708.237 us` |
| historical 515.603 us reference / optimized v1 | `528.216 us` | `528.033 us` | `524.802 us` | `531.268 us` |
| double-output-LDS candidate | `514.138 us` | `513.700 us` | `507.641 us` | `519.362 us` |

The historical optimized-v1 kernel is therefore reproduced within about
`1.35%` of its earlier b8-3 result when using the exact original command. In
the interleaved comparison, double-output-LDS improves on optimized v1 by
`14.078 us` (`2.67%`). The GPU/KFD check was clean both immediately before and
after the benchmark.

Full log:

```text
my_code/moe_gemm1_act1_optimized/history_runs/20260909_125250_perf-const0.log
```

## Canonical a07-3 e2e const0 benchmark

The acceptance metric is now explicitly the ASM GEMM1 profiler row from:

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

The GPU/KFD checks immediately before and after this run were both idle.

| Version | GEMM1 | Fused MoE | Correctness |
|---|---:|---:|---|
| independent-load safe baseline | `724.460 us` | `1700.75 us` | `logits_diff=0`, `rel_l2=0`, pass |
| historical 515.603 us reference / optimized v1 | `558.548 us` | `1542.56 us` | `logits_diff=0`, `rel_l2=0`, pass |
| double-output-LDS candidate | **`551.248 us`** | **`1530.19 us`** | `logits_diff=0`, `rel_l2=0`, pass |

On this canonical measurement, double-output-LDS improves GEMM1 over optimized
v1 by `7.300 us` (`1.31%`) and improves the complete fused MoE operation by
`12.37 us` (`0.80%`).  The earlier `515.603 us` value remains a valid b8-3
standalone measurement, but it is not the final e2e performance metric.

Full log:

```text
my_code/moe_gemm1_act1_optimized/history_runs/20260909_134433_a07_e2e-const0.log
```

## d01-3 post-reboot canonical e2e const0 benchmark

After d01-3 rebooted, the user requested exactly one invocation of the
canonical command and no separate remote checks or additional measurements:

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

The values below replace the earlier d01-3 measurements.

| Version | GEMM1 | Fused MoE | Correctness |
|---|---:|---:|---|
| independent-load safe baseline | `765.878 us` | `2029.95 us` | `logits_diff=0`, `rel_l2=0`, pass |
| historical 515.603 us reference / optimized v1 | `580.872 us` | `1819.18 us` | `logits_diff=0`, `rel_l2=0`, pass |
| double-output-LDS candidate | **`554.139 us`** | **`1798.40 us`** | `logits_diff=0`, `rel_l2=0`, pass |

Double-output-LDS improves GEMM1 over optimized v1 by `26.733 us` (`4.602%`)
and improves complete fused MoE time by `20.78 us` (`1.142%`).  It improves
GEMM1 over the safe baseline by `211.739 us` (`27.647%`).  The script reported
an initial `sclk` of `2358 MHz`.  This section records the requested single
post-reboot run; no repeat or reverse-order benchmark was performed.

Saved performance summary:

```text
my_code/moe_gemm1_act1_optimized/history_runs/20260909_153855_d01_reboot_e2e-const0_summary.log
```

The benchmark script also saved its raw remote log at:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/moe_gemm1_act1_optimized/history_runs/20260909_153855_e2e-const0.log
```
