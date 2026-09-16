# `resident 2+3` no-Scale-TDM wait4 experiment and thread-trace analysis

## Experiment

The source kernel is:

```text
my_code/moe_gemm1_act1_optimized/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_
  ab4_scale_half_tdm_full_setup_loop_wait6.s
SHA256=02dffd3a7f6ca015a25d52e8a2273c6cd49deed57bde4f0f758c1a97c7c70c06
```

It was copied to this diagnostic variant:

```text
my_code/moe_gemm1_act1_optimized/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_
  ab4_no_scale_tdm_full_setup_loop_wait4.s
SHA256=bcf2675b551551462f9073e2936a25187da031321a9a649a8a88fb5234709393
```

Only these executable changes were made:

- delete all `56` static
  `tensor_load_to_lds s[80:83], s[84:91] th:TH_LOAD_NT_RT` instructions used
  to load ScaleA or ScaleB;
- change all `20` `s_wait_tensorcnt 0x6` instructions to
  `s_wait_tensorcnt 0x4`;
- change all `4` `s_wait_tensorcnt 0x3` instructions to
  `s_wait_tensorcnt 0x2`.

The wait conversion preserves the intended number of queued K stages: the
original pipeline issues three TDM requests per stage (A, B, Scale), while the
diagnostic issues two (A, B). No descriptor setup, LDS read, WMMA, epilogue,
barrier, or resource metadata was changed.

This kernel is a performance diagnostic, not a correct general implementation.
The requested const0 workload happens to produce the same zero output and hash,
but that result does not validate missing ScaleA/ScaleB data for random input.

## Const0 performance

The GPU was idle before the measurement and no test process remained after it.
The parent and diagnostic kernels were measured in one interleaved run on
`d01-3`:

```bash
AITER_HISTORY_CASE_LIST=ab4_scale_half_tdm_full_setup_wait6,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_ab4_no_scale_tdm_full_setup_loop_wait4.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=3 RUN_VERIFY=0 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

| kernel | GEMM1 samples | median | fused MoE samples | median |
|---|---:|---:|---:|---:|
| resident 2+3 full-setup wait6 | 565.024, 528.863, 561.687 us | 561.687 us | 1450.85, 1418.56, 1447.37 us | 1447.37 us |
| no-Scale-TDM wait4 | 517.674, 516.008, 517.387 us | 517.387 us | 1406.13, 1403.33, 1404.15 us | 1404.15 us |

The diagnostic lowers GEMM1 by `44.300 us`, or `7.89%`, and lowers the fused
MoE median by `43.22 us`, or `2.99%`. This is the reliable performance result;
ATT-instrumented profiler timestamps are not used as timing measurements.

The same diagnostic now has the selectable case name
`ab4_no_scale_tdm_wait4`, so it can be rerun without candidate-path arguments:

```bash
AITER_HISTORY_CASE_LIST=ab4_scale_half_tdm_full_setup_wait6,ab4_no_scale_tdm_wait4 \
ROUNDS=3 RUN_VERIFY=0 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

## ATT capture

The fixed `benchmark_history.sh` ATT path used the MoE e2e launcher, selected
the eighth GEMM1 invocation directly, and captured the four SIMD selections
sequentially:

```bash
AITER_HISTORY_CASE_LIST=candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_ab4_no_scale_tdm_full_setup_loop_wait4.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
RUN_VERIFY=0 \
AITER_ATT_E2E_ITERS=2 \
AITER_ATT_SIMD_LIST=0,1,2,3 \
AITER_ATT_TIMEOUT_SECONDS=300 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att
```

The trace is on `d01-3` at:

```text
my_code/moe_gemm1_act1_optimized/history_runs/
  heliosr-1b114-d01-3_20260916T043543Z_att/att/candidate/
```

Every SIMD directory contains one non-empty `.att`, one decoded `code.json`,
and one decoded wave JSON. The capture configuration was:

```yaml
kernel_iteration_range: "[8]"
att_target_cu: 1
att_shader_engine_mask: "0x1"
att_simd_select: "0"  # repeated sequentially for 1, 2, and 3
att_buffer_size: "0x10000000"
```

The analysis command was:

```bash
python3 my_code/moe_gemm1_act1_optimized/analyze_all_simd_owner_trace.py \
  my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260916T043543Z_att/att/candidate \
  --output \
  my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260916T043543Z_att/att/candidate/no_scale_tdm_wait4_detailed_metrics.json
```

`analyze_all_simd_owner_trace.py` imports and uses:

```text
my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py
SHA256=6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

Persistent tasks are delimited by consecutive dynamic occurrences of:

```asm
s_add_co_u32 s28, s28, 16
s_cmp_lt_u32 s28, 0x240
s_cbranch_scc0 5
```

The first interval per wave is dropped.

## Sampling coverage

| owner | SIMD | steady tasks | mean cycles/task | median | p90 | full traced wave cycles |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 3 | 34,651.0 | 36,440 | 38,672.0 | 1,230,985 |
| B | 3 | 17 | 30,523.1 | 29,432 | 33,997.2 | 1,187,180 |
| ScaleA | 2 | 1 | 26,549.0 | 26,549 | 26,549.0 | 1,126,548 |
| ScaleB | 1 | 17 | 33,058.6 | 28,462 | 43,777.4 | 1,203,919 |

The cross-owner median of the mean task costs is `31,790.9 cycles`. The owner
sample counts are uneven, and ScaleA has only one steady interval. Therefore
the task-cycle aggregate is used to rank bottlenecks inside this capture, not
as a replacement for the three-round profiler result. The median full-wave
span is `1,195,549.5 cycles`, only `0.28%` below the parent capture's
`1,198,936 cycles`; ATT instrumentation and different sampled waves obscure the
`7.89%` profiler improvement.

## Explicit wait stall

| owner | task cycles | barrier | TENSORcnt | DScnt | KMCnt + idle | all explicit waits |
|---|---:|---:|---:|---:|---:|---:|
| A | 34,651.0 | 4,871.7 (14.06%) | 1,723.7 (4.97%) | 790.0 (2.28%) | 809.0 (2.33%) | 8,194.3 (23.65%) |
| B | 30,523.1 | 3,145.2 (10.30%) | 2,213.2 (7.25%) | 705.8 (2.31%) | 59.5 (0.19%) | 6,123.6 (20.06%) |
| ScaleA | 26,549.0 | 985.0 (3.71%) | 2,336.0 (8.80%) | 896.0 (3.37%) | 43.0 (0.16%) | 4,260.0 (16.05%) |
| ScaleB | 33,058.6 | 2,891.6 (8.75%) | 2,517.8 (7.62%) | 731.8 (2.21%) | 170.5 (0.52%) | 6,311.7 (19.09%) |
| cross-owner median | 31,790.9 | 3,018.4 (9.49%) | 2,274.6 (7.15%) | 760.9 (2.39%) | 115.0 (0.36%) | 6,217.6 (19.56%) |

Deleting Scale TDM decreases barrier stall but increases exposed TENSORcnt
stall:

| representative metric | parent wait6 | no-Scale wait4 | change |
|---|---:|---:|---:|
| K-ring cluster barrier | 1,588.0 | 1,193.5 cycles/task | -394.5 (-24.8%) |
| all `s_barrier_wait` | 3,404.5 | 3,018.4 cycles/task | -386.1 (-11.3%) |
| all `s_wait_tensorcnt` | 1,653.5 | 2,274.6 cycles/task | +621.1 (+37.6%) |
| all `s_wait_dscnt` | 613.1 | 760.9 cycles/task | +147.8 (+24.1%) |
| all explicit waits | 5,807.9 | 6,217.6 cycles/task | +409.8 (+7.1%) |

This is not contradictory to the profiler speedup. Removing one request from
every A+B+Scale trio reduces total TDM traffic and instruction count. Changing
`wait6` to `wait4`, however, makes the wave drain to a lower outstanding count.
The deleted scale issue also removes useful instruction distance between A/B
issue and the next wait. The remaining A/B completion latency is consequently
more visible at `s_wait_tensorcnt`, even though the complete kernel is faster.

## Dominant K-ring cluster barrier

Deleting Scale TDM shifts the PCs but preserves the six logical K-ring waits:

```text
A / ScaleA: 0xb374  s_barrier_wait 0xfffd
B / ScaleB: 0xd37c  s_barrier_wait 0xfffd
```

| owner | hits/task | stall cycles/task | task share |
|---|---:|---:|---:|
| A | 6 | 985.3 | 2.84% |
| B | 6 | 1,621.9 | 5.31% |
| ScaleA | 6 | 257.0 | 0.97% |
| ScaleB | 6 | 1,401.6 | 4.24% |
| cross-owner median | 6 | 1,193.5 | 3.75% |

The cluster barrier remains the largest repeated synchronization site, but its
representative cost falls by about `25%`. The owner ordering changes: ScaleA is
now the late arrival and waits least, while B waits longest. This is evidence
that the removed scale transfer materially affected ring arrival timing. It
also shows that the remaining imbalance has moved to the A/B payload paths;
the cluster protocol itself is not the sole source of the delay.

The task-boundary cluster wait at `0x10360` costs `90-631 cycles/task` across
owners and remains secondary.

## Remaining TDM bottleneck

The largest waits are now the retimed `s_wait_tensorcnt 0x4` instructions:

| owner | PC | role in pipeline | stall cycles/task | task share |
|---|---:|---|---:|---:|
| ScaleA | `0x4d44` | full-setup A/B completion | 1,761.0 | 6.63% |
| B | `0x3d78` | full-setup A/B completion | 1,215.5 | 3.98% |
| ScaleB | `0x5d3c` | full-setup A/B completion | 1,180.4 | 3.57% |
| A | `0x9ef4` | repeated K-stage A/B completion | 1,288.3 | 3.72% |
| B | `0xbf08` | repeated K-stage A/B completion | 506.4 | 1.66% |

The full-setup sites have this decoded form:

```asm
s_barrier_signal -1
s_barrier_wait 0xffff
s_wait_tensorcnt 0x4
s_barrier_signal -1
s_barrier_wait 0xffff
ds_load_...
```

The performance bottleneck has therefore moved from a mixed scale/payload TDM
and cluster-arrival problem toward completion of the remaining A/B payload
TDM. Simply lowering the wait threshold in proportion to request count does
not preserve the same amount of latency hiding.

The most useful follow-up diagnostic is a wait-threshold sweep on this exact
no-Scale binary shape, for example `wait4`, `wait5`, and `wait6`, while keeping
the 56 deleted scale loads unchanged. This distinguishes excess draining at
`wait4` from unavoidable A/B completion latency. It is only a diagnostic until
a correct scale delivery path is restored.

## Other long-latency instructions

The issue timeline remains dominated by useful matrix and data movement work:

| group | representative cycles/task | representative share |
|---|---:|---:|
| WMMA issue | 9,284.7 | 29.21% |
| SALU/control or scheduling gap | 5,080.2 | 15.98% |
| LDS read issue or completion gap | 3,206.8 | 10.09% |
| barrier wait timeline | 3,089.7 | 9.72% |
| packed SiLU VALU | 2,703.2 | 8.50% |
| TENSORcnt wait timeline | 2,305.1 | 7.25% |
| EXP/RCP issue | 1,639.9 | 5.16% |
| other VALU | 1,712.9 | 5.39% |
| DScnt wait timeline | 1,163.5 | 3.66% |

Individual WMMA instructions normally complete in `9-17 cycles`. The longest
non-wait completion event is a `ds_load_b128` at ScaleB PC `0x5e14`, with
`109.4` mean and `415` maximum cycles. B has a corresponding `ds_load_b128` at
`0x3e50`, with `71.0` mean and `262` maximum cycles. These long LDS returns are
partly overlapped, while the following `s_wait_dscnt` exposes only about `2.4%`
at the representative level.

The repeated `s_nop 0` after an indirect `s_set_pc_i64` has approximately
`25-27` cycles of observed latency and about `180-193 cycles/task` of issue
gap. It is much smaller than the TENSORcnt and barrier losses. Several SiLU
instructions have large next-issue gaps but architectural trace latency of only
`1-2 cycles`; these gaps represent scheduling or shared-pipeline delay and are
not evidence that an individual `v_rcp_f32`, `v_exp_f32`, or packed multiply
takes hundreds of serialized cycles.

## Resource usage

The diagnostic retains the parent's metadata:

```text
LDS       = 327680 bytes = 320 KiB/workgroup
SGPR      = 106/wave
VGPR      = 1024/wave
wave size = 32
block     = 128 threads = 4 waves
scratch   = 0
kernarg   = 184 bytes
```

These values reach the documented gfx1250 limits for LDS, normal SGPRs, and
VGPRs. The relevant local hardware references are section 3.3.1.1
(`106` normal SGPRs), section 3.3.2.1 (`1024` VGPRs), and section 3.3.4
(`320 KiB` LDS) in:

```text
mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/
  MI400_Shader_Programming#65.txt
```

No occupancy or extra-buffer solution is available without first reducing one
of those allocations.

## Bottleneck conclusion

For the no-Scale-TDM diagnostic, the current bottleneck order is:

1. remaining A/B TDM completion exposed at `s_wait_tensorcnt 0x4` (`7.15%`);
2. workgroup and cluster arrival imbalance (`9.49%` total barrier stall), with
   the six K-ring cluster waits alone at `3.75%`;
3. LDS-read completion (`2.39%` explicit DScnt stall, with isolated
   `ds_load_b128` latencies up to `415 cycles`);
4. useful WMMA and SiLU work, which dominates issue time but is not a wait
   bottleneck.

The `7.89%` profiler gain confirms that ScaleA/ScaleB TDM traffic and its
associated scheduling are expensive. The trace also shows that a correct
replacement must avoid turning the saved scale traffic into a stricter A/B
drain point. A practical implementation should retain or regenerate scale
values through a cheaper path, then tune the remaining tensor wait threshold
and owner arrival schedule together.
