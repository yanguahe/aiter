# tensor_wait_after_wmma: fresh a07-3 thread-trace analysis

Captured on **2026-09-17, 15:19-15:20 UTC**, on `a07-3`, inside `hyg_fyd1`.
This report analyzes a new capture of the retained kernel at commit `3b0ad10a`.
It does not reuse the earlier `20260917T070446Z` capture as the measurement.

## Findings

The largest exposed costs are **cluster/workgroup synchronization and LDS
completion**, followed by TDM completion. Across 16 complete instruction-level
wave traces, explicit wait stall accounts for **38.47%** of the observed wave
lifetimes: barrier `19.72%`, DScnt `13.16%`, TENSORcnt `5.14%`, and other waits
`0.45%`. The two role-specific PCs implementing the six K-ring cluster waits
per task contribute **7.62%** by themselves.

The observed physical WGPs finish very evenly within each captured SE. The
completion-imbalance metric has mean **0.022984%**, median approximately
**0.0131%**, and maximum **0.093461%**. This provides no evidence of a large
within-SE WGP-distribution tail in this capture. It does not rule out internal
producer/consumer phase imbalance: cluster barriers can make WGPs finish
together while propagating the cost of a late arrival to their peers.

The current evidence supports prioritizing the K-ring readiness path and the
repeated `s_wait_dscnt 0x8` sites. It does not establish cache-miss rates,
physical TDM saturation, or a specific recoverable percentage of wall time;
those would need additional counters or controlled implementation experiments.

## Kernel, workload, and resources

```text
case: tensor_wait_after_wmma
ISA: persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s
ISA SHA256: 02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d
branch: hyg_gfx1250_gemm_a4w4
commit: 3b0ad10a Optimize persistent MoE GEMM1 scheduling

experts=96, tokens=16384, topk=6
model_dim=7168, inter_dim=3072, a4w4, SiLU, no bias
grid=(16,16,1), cluster=(4,4,1), block=(128,1,1)
16 physical clusters, 576 logical cluster tasks, 36 tasks per resident WG
```

The retained change executes four independent WMMAs before each of eight
static K-ring `s_wait_tensorcnt 0x2` / `s_barrier_signal -1` pairs. It preserves
`s_wait_dscnt 0x8`, counter thresholds, input/output LDS lifetimes, and the
workgroup barrier preceding the next TDM refill.

| Resource | Verified value | Evidence |
|---|---:|---|
| LDS/workgroup | 327,680 B = 320 KiB | compiled code-object note |
| VGPR/wave | 1,024 | compiled code-object note |
| SGPR metadata count | 106 | compiled `.sgpr_count` |
| Numbered SGPR declaration | 104 | source `.amdhsa_next_free_sgpr` |
| Scratch/private segment | 0 B | compiled code-object note |
| Kernarg | 184 B | compiled code-object note |
| Wave size | 32 | metadata and runtime |

The SGPR fields above are reported separately; their difference is not a claim
that additional working registers are available. The GPU reports 8 XCCs,
16 shader banks/SEs, 256 WGPs, and 1,024 SIMD32s. The LDS and VGPR allocations
are consistent with the observed one resident wave slot per physical SIMD.

Metadata evidence: [code_object_notes.log](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/code_object_notes.log)
and [out_agent_info.csv](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/out_agent_info.csv).

## Correctness and timing outside ATT

Fresh random MoE e2e passed before profiling:

```text
logits_diff=3.39799e-06
rel_l2=0.00260689
pass=True
moe_output_hash128=1556fc617347e2dabc9cff19dbfd822b
ref_output_hash128=1a5d22911ba167160b4f2c12092a5193
```

The random hashes differ because the production numerical comparison is not
bit-exact against the reference. They match the retained kernel's established
random outputs. Const0 produced identical output and reference hashes:
`21291d9023c8af8a6324fe20f346a967`.

A separate, untraced three-round MoE e2e const0 run used 20 iterations per round:

| Metric | Samples, us | Median, us |
|---|---|---:|
| GEMM1 profiler | 511.608, 506.412, 511.966 | **511.608** |
| Fused MoE e2e | 1346.41, 1343.10, 1346.05 | **1346.05** |

This is the current timing reference, not a new parent/candidate improvement
claim. ATT timing is kept separate because tracing changes execution.

The source ISA and benchmark SHA256 matched between local and remote trees;
the fixed snapshot verified at commit
`23c2caaafa5f1c6e6d5d9f756980fe004af4202c`, with 1,854 payload files.
The default `three_kernel` A/ScaleA producer was used.

GPU/KFD checks were clear before ATT. A later post-capture sample briefly
reported 2% utilization and about 757 MiB VRAM without a visible KFD owner;
it returned to 0% and the original approximately 165 MiB idle allocation before
the untraced performance run began. The performance post-check was clear.
No test/profiler process remained after the work.

Timing log: [performance.log](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/performance.log).

## Capture configuration and actual coverage

```text
launcher: normal MoE e2e through benchmark_history.sh
kernel regex: ^moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1$
kernel_iteration_range: [8]
att_target_cu: 1
att_shader_engine_mask: 0xf
att_simd_select: 0, 1, 2, 3 in separate sequential captures
att_buffer_size: 0x10000000
capture timeout: 300 seconds
decoder: /data/yanguahe/code/wk_sp1/decoder_new/librocprof-trace-decoder.so
```

The normal untraced preflight built/loaded the extension before profiling.
Clang was isolated from rocprof injection. All four captures completed with
return code zero and had nonempty `.att`, one `code.json`, and decoded wave
JSON files. The benchmark also successfully ran `my_code/analyze_att_capture.py`.

Coverage must be interpreted at two different levels:

- **Instruction traces:** four complete waves per SIMD-select capture, one
  for each of SE0-SE3 at the selected CU/WGP; 16 wave traces across four
  independent captures.
- **Occupancy:** 16 physical WGPs per captured SE, 64 per capture, with all
  four physical SIMDs represented. These 64 WGPs are observed repeatedly
  across the four captures, not 256 distinct WGPs.
- The device exposes 16 SEs / 256 WGPs. Mask `0xf` covers **four SEs / 64 WGPs**,
  so this is not an all-256-WGP completion-balance claim. It satisfies the
  existing analyzer's SE0-SE3 input contract without changing that tool.

Each instruction wave contains exactly:

```text
36 logical tasks
64,512 v_wmma_scale_f32_32x16x128_f4 = 36 x 1,792
9,216 v_exp_f32_e32                 = 36 x 256
9,216 v_rcp_f32_e32                 = 36 x 256
```

The full GEMM/SiLU computation is therefore present in this const0 capture.

## Cycle accounting and the complete task ledger

The canonical `trace_segment_cycles.py` is the file referenced by
`flydsl-align-reference-kernel.mdc:104-111`. Its SHA256 is:

```text
6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

The remote tracked copy had SHA256 `7f724112...`; this was verified to be the
same text with LF instead of CRLF. The canonical bytes were copied into this
run's `analysis_tools/` directory to satisfy the existing checksum gate,
without overwriting the remote tracked file.

The task boundary was taken from each fresh trace's statistics CSV and checked
against `code.json`, using these three consecutive instructions:

```asm
; trace virtual addresses: 0xcea4, 0xcea8, 0xceb0
s_add_co_u32 s28, s28, 16
s_cmp_lt_u32 s28, 0x240
s_cbranch_scc0 5
```

With two identical sample points, the canonical tool alternates start/end and
selects pairs `(0,1), (2,3), ...`: 18 pairs per wave, 72 per capture. Its normal
and `--specific-part-representative-trace` modes were both run for all four
captures. The supplemental analyzer validates these canonical pairs and then
accounts for **every adjacent boundary**, all 35 body tasks per wave, so the
whole-kernel budget does not omit alternate tasks or the first task.

| Complete decoded-wave region | Mean cycles/wave | Share |
|---|---:|---:|
| Initial setup + first task | 35,760.500 | 2.9635% |
| All following 35 task intervals | 1,170,106.750 | 96.9661% |
| Final boundary/drain | 849.813 | 0.0704% |
| **Entire decoded wave** | **1,206,717.063** | **100%** |

All 560 consecutive body-task intervals have mean `33,431.621`, median
`31,617`, p90 `40,419.2`, minimum `24,112`, and maximum `95,377` cycles.
The mean and median are different because some intervals have long tails.

The denominator for the following whole-kernel percentages is the sum of the
16 complete decoded-wave spans, **19,307,473 cycles**. Equivalently, mean
stall per wave is divided by the mean span `1,206,717.063`. These are measured
per-wave exposures over the entire kernel, not a sum of simultaneous GPU
cycles and not a percentage of an unrelated maximum occupancy span.

For clock context, `analyze_att_capture.py` reported maximum observed occupancy
span `1,304,257` shader cycles at SIMD2-select/SE2, with matching capture
REALTIME wall span `586.240 us`. Mean GFXCLK across 16 capture x SE series was
`2231.155 MHz`. Independent SIMD captures are not simultaneous hardware state;
cross-SE raw shader timestamps must not be directly subtracted.

## Long waits over the entire kernel

| Wait family | Mean stall cycles/wave | Entire-wave share |
|---|---:|---:|
| `s_barrier_wait` | 237,984.250 | **19.7216%** |
| `s_wait_dscnt` | 158,805.500 | **13.1601%** |
| `s_wait_tensorcnt` | 62,014.000 | **5.1391%** |
| `s_wait_idle` | 3,969.688 | 0.3290% |
| `s_wait_kmcnt` | 1,437.000 | 0.1191% |
| **All explicit waits** | **464,210.438** | **38.4689%** |

Independent aggregation of `Stall` from the four `stats_ui_*.csv` files exactly
matches the decoded-wave total for every wait family. The key instruction
forms are:

| Instruction | Mean stall cycles/wave | Entire-wave share |
|---|---:|---:|
| `s_wait_dscnt 0x8` | 127,783.188 | **10.5893%** |
| `s_barrier_wait 0xfffd` (all cluster sites) | 123,242.938 | **10.2131%** |
| `s_barrier_wait 0xffff` (workgroup sites) | 114,741.313 | **9.5086%** |
| `s_wait_tensorcnt 0x2` | 60,253.063 | **4.9931%** |
| `s_wait_dscnt 0x14` | 22,194.688 | 1.8393% |
| `s_wait_dscnt 0x0` | 6,230.563 | 0.5163% |
| `s_wait_dscnt 0xa` | 1,758.688 | 0.1457% |
| `s_wait_tensorcnt 0x1` | 1,757.938 | 0.1457% |
| `s_wait_dscnt 0x4` | 838.375 | 0.0695% |

Thus approximately 80% of DScnt stall is at `wait8`, while nearly all TENSORcnt
stall is at `wait2`. The old-output `wait1` exposure is already small.

### Dominant PCs

PCs are fresh `code.json` virtual addresses. Contributions are averaged over
all 16 waves, including waves that do not execute that role-specific PC.

| PC | Instruction / context | Mean stall cycles/wave | Entire-wave share | Maximum observed latency |
|---|---|---:|---:|---:|
| `0x998c` | `s_barrier_wait 0xfffd`, B/ScaleB K-ring | 47,174.063 | 3.9093% | 14,049 |
| `0x7c84` | `s_barrier_wait 0xfffd`, A/ScaleA K-ring | 44,721.500 | 3.7060% | 8,341 |
| `0x9408` | `s_wait_dscnt 0x8`, B-family path | 29,430.688 | 2.4389% | 15,038 |
| `0x76f4` | `s_wait_dscnt 0x8`, A-family path | 28,676.125 | 2.3764% | 9,216 |
| `0xcec0` | persistent-task cluster boundary | 16,877.813 | 1.3987% | 5,382 |
| `0x7904` | `s_barrier_wait 0xffff` | 16,722.625 | 1.3858% | 9,356 |
| `0x6308` | `s_wait_dscnt 0x8` | 13,972.125 | 1.1579% | 10,260 |
| `0x801c` | `s_wait_dscnt 0x8` | 11,144.750 | 0.9236% | 8,648 |
| `0x960c` | `s_wait_tensorcnt 0x2` | 9,404.750 | 0.7794% | 12,128 |
| `0x8ed4` | `s_wait_tensorcnt 0x2` | 8,901.188 | 0.7376% | 12,812 |

The two dominant K-ring PCs together contribute `91,895.563 cycles/wave`,
**7.6153%** of the entire decoded wave. Each applicable wave executes its
K-ring site 216 times, six per task. This is approximately `2,552.655` exposed
cycles per logical task when averaged over all 36 tasks. It is a subset of
the cluster/barrier totals above, not an additional category.

The two large DScnt sites at `0x9408` and `0x76f4` together contribute **4.8153%**.
This supports investigating LDS producer-to-consumer distance and the shared
LDS/TDM path. It does not justify moving a wait past a dependent WMMA or
relaxing its count without a new liveness/queue proof.

## Other long-latency instructions and issue attribution

An instruction's ATT `latency` and the interval until the next instruction
issues are different measurements. Memory completion can overlap independent
work. Latencies must not be summed and added to the explicit waits.

The issue-gap partition assigns `[issue_i, issue_(i+1))` once, to instruction
`i`, and closes exactly to each decoded-wave span. It is an attribution of the
selected wave's timeline, not functional-unit busy time or an instruction's
architectural execution latency.

| Non-wait PC | Instruction | Mean / maximum ATT latency | Whole-wave issue-gap share |
|---|---|---:|---:|
| `0x5f98` | `s_nop 0` | 32.96 / 12,652 | 0.3442% |
| `0x9868` | `ds_load_b128 v[44:47], v72 offset:4608` | 12.01 / 12,425 | 0.1254% |
| `0xcdb4` | `ds_store_b64 v91, v[106:107] offset:48` | 17.05 / 7,503 | 0.0479% |
| `0xcda4` | `ds_store_b64 v91, v[102:103] offset:16` | 71.18 / 6,746 | 0.2094% |
| `0xb738` | `ds_store_b64 v91, v[102:103] offset:16` | 59.12 / 5,051 | 0.1734% |
| `0xac80` | `ds_store_b64 v91, v[128:129] offset:6944` | 8.06 / 1,267 | 0.3252% |
| `0x7168` | scaled FP4 WMMA | 16.07 / 17 | 0.1768% |
| `0xcd8c` | `v_cvt_pk_bf16_f32 v131, v224, v225` | 1 / 1 | 0.1641% |

The `v_cvt_pk_bf16_f32` row is a useful counterexample: it has latency one,
but nontrivial following issue gaps. Those gaps cannot be called BF16
conversion execution time. Likewise, an isolated 12k-cycle observation at
`s_nop 0` is not its architectural NOP cost. The trace alone does not isolate
the scheduler, dependency, or downstream-resource contribution to such gaps.

| Disjoint issue-timeline category | Share |
|---|---:|
| WMMA-attributed gaps | 28.1865% |
| Barrier-wait intervals | 20.0004% |
| DScnt-wait intervals | 14.4670% |
| LDS-read-attributed gaps | 10.8623% |
| Other VALU-attributed gaps | 9.4183% |
| SALU/control-attributed gaps | 5.5265% |
| TENSORcnt-wait intervals | 5.2315% |
| EXP/RCP-attributed gaps | 3.2661% |
| LDS-write-attributed gaps | 1.4761% |
| Remaining categories | 1.5653% |

These category percentages form one partition. They must not be added to the
stall-family table, which measures overlapping aspects of the same timeline.

## Owner observations

Owner identity was inferred from the executed role-marker instructions in each
wave, rather than assuming SIMD-select IDs equal logical wave IDs.

| Owner | Mean body task cycles | Barrier stall share | DScnt stall share | TENSORcnt stall share |
|---|---:|---:|---:|---:|
| A | 33,274.89 | 17.96% | 14.27% | 7.27% |
| B | 32,714.52 | 16.02% | 11.91% | 9.04% |
| ScaleA | 34,588.70 | 22.44% | 13.19% | 1.15% |
| ScaleB | 33,148.37 | 22.31% | 13.26% | 3.31% |

Wait percentages use each owner's complete-wave denominator. The A/B owners
expose more TDM wait, while scale owners expose more barrier wait. This is
consistent with phase/readiness skew between payload and scale paths.
Because SIMD-select captures were collected sequentially and shader counters
are not synchronized across SIMDs, this table is not a simultaneous four-wave
arrival-time measurement and cannot prove which wave was last at every barrier.

## Physical-WGP load balance from analyze_att_capture.py

The analyzer keys physical WGPs by `(capture, SE, packed-SA-WGP, kernel-PC label)`.
SA is decoded as `(packed >> 7) & 1`; WGP is `packed & 0x7f`. The occupancy
kernel-PC label is `1`; it must not be confused with UI dispatch ID `16501`.

Each independent capture contains 256 complete occupancy wave lifetimes,
64 physical WGPs, one lifetime per physical SIMD/slot key, and maximum
concurrent slot count one. Every WGP has one aggregate active episode with
zero inter-episode idle gap. This is consistent with one resident persistent
workgroup per WGP rather than a sequence of separately launched workgroups.

The tool's completion metric, computed within one capture x SE x kernel label,
is:

```text
WGP envelope = last wave end - first wave start
final-end span = max(WGP final end) - min(WGP final end)
completion imbalance = final-end span / median(WGP envelope)
```

| SIMD-select capture | WGPs observed | End-span range across its four SEs, cycles | Mean imbalance | Maximum imbalance |
|---|---:|---:|---:|---:|
| 0 | 64 | 108-172 | 0.011945% | 0.014608% |
| 1 | 64 | 156-1,094 | 0.041400% | 0.093461% |
| 2 | 64 | 102-514 | 0.018112% | 0.039424% |
| 3 | 64 | 96-478 | 0.020478% | 0.040208% |

The largest observation is SIMD1-select/SE2: 16 WGPs, median envelope
`1,170,535.5 cycles`, and end-span `1,094 cycles`. All 16 capture x SE rows
remain below **0.1%** by this metric.

Interpretation and limits:

- The observed WGPs have nearly identical completion envelopes, with no
  substantial physical-WGP straggler in the sampled SEs.
- Uniform final completion is compatible with large cluster-barrier waits;
  synchronization may equalize finish times while preserving internal stalls.
- Occupancy has no logical task/WG identifier. It does not directly measure
  FLOPs, bytes, or each WGP's per-task duration. The 36-task count is established
  separately by instruction traces and the persistent mapping.
- WGP envelopes merge raw profiler timestamps across that WGP's physical
  SIMDs. They are the analyzer's diagnostic metric, not REALTIME-aligned wall
  durations. Different SE/capture clocks must not be combined into a global
  final-end span.
- The other 12 SEs / 192 WGPs were not captured by mask `0xf`.

Full analyzer output: [analyze_att_capture.log](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/analyze_att_capture.log).

## Bottleneck assessment

1. **K-ring and workgroup readiness are the first exposed bottleneck.**
   Barrier stall is `19.72%`, including `7.62%` at the repeated K-ring sites.
   The observed WGP-completion balance does not support a large grid-straggler
   explanation. Internal producer/consumer readiness and phase alignment are
   the more useful investigation targets.
2. **LDS completion is the next priority.** `wait8` alone is `10.59%` of the
   entire wave. CDNA5 documents that LDS completion makes read data available
   in VGPRs; these waits cannot be removed on the assumption that TDM or a
   cluster barrier also satisfies the LDS-to-VGPR dependency.
3. **TDM completion remains material but smaller.** It accounts for `5.14%`,
   almost entirely `wait2`. The accepted four-WMMA overlap has not eliminated
   this latency. Further schedule changes need site-specific queue/liveness
   accounting and a fresh correctness/performance comparison.
4. **Isolated long non-wait latencies are not the dominant aggregate cost.**
   Large maxima exist at LDS instructions and NOPs, but their individual
   issue shares are much smaller than the recurring barrier/DScnt groups.
   Shortening source instruction count alone need not shorten the critical path.
5. **Occupancy is resource constrained.** The code already allocates 320 KiB
   LDS and 1,024 VGPRs/wave. Additional resident waves or a complete additional
   input stage require a resource/layout redesign, not a launch-parameter-only
   change. This report makes no performance claim for such a redesign.

No ISA, counter threshold, launch topology, or production package file was
modified for this analysis.

## Hardware references

Documented facts were checked against the local gfx1250 corpus, rather than
applying the older MI308X examples in the generic profiling rule:

- `MI450/amd-instinct-cdna5-instruction-set-architecture.txt`, section
  **3.4.11 Time**, lines 2073-2095: shader counters are not synchronized
  across SIMDs; fixed-frequency REALTIME is intended for cross-wave timing.
- The same ISA, **5.6.6 Cluster Barriers**, lines 3319-3330: member count is
  workgroups; recommended workgroup synchronization followed by one signaling
  wave per workgroup, with all waves waiting.
- The same ISA, **5.7 Data Dependency Resolution**, lines 3411-3423, and
  **5.7.1.4 LDS**, lines 3583-3586: per-wave outstanding counters and LDS
  completion semantics.
- The same ISA, **10.11.1 Tensor Instructions**, lines 10147-10156:
  TENSORcnt counts instructions, Tensor load/store completion is ordered within
  one wave and unordered with other waves and other memory-instruction types.
- `architecture/subsystem/SH/MI400_Shader_Programming#65.txt`, **1.5 Hardware
  Internals**, lines 1482-1497: four SIMD32s per WGP, shared LDS/WGP cache,
  and one TDM per SIMD pair. Lines 358-359 document up to 320 KiB LDS;
  **3.3.2.1 VGPR Allocation**, lines 2151-2171, documents 1,024 VGPRs per
  wave and the MI450 per-SIMD pool.

The performance attribution above is measured or explicitly identified as an
inference. The hardware references establish dependency/resource constraints;
they do not by themselves prove a particular cache or contention cause.

## Reproduction and artifacts

Run from `/data/yanguahe/code/wk_sp1/aiter` inside `hyg_fyd1`, after checking all
GPUs/KFD are idle on the host for performance/ATT commands:

```bash
AITER_HISTORY_CASE_LIST=tensor_wait_after_wmma \
AITER_ATT_SHADER_ENGINE_MASK=0xf \
AITER_ATT_SIMD_LIST=0,1,2,3 \
AITER_ATT_TIMEOUT_SECONDS=300 \
RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att
```

Exact raw capture root retained on a07-3:

```text
my_code/moe_gemm1_act1_optimized/history_runs/
heliosr-1b114-a07-3_20260917T151911Z_att/att/tensor_wait_after_wmma/
```

Analysis artifacts, copied locally as well as retained remotely:

```text
my_code/moe_gemm1_act1_optimized/history_runs/
a07_tensor_wait_after_wmma_fresh_20260917T1517Z/
```

```bash
CASE_ROOT=my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-a07-3_20260917T151911Z_att/att/tensor_wait_after_wmma
TASK_ROOT=my_code/moe_gemm1_act1_optimized/history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z

python3 my_code/analyze_att_capture.py --dir "$CASE_ROOT" --no-plot

python3 "$TASK_ROOT/analysis_tools/analyze_tensor_wait_after_wmma_trace.py" \
  "$CASE_ROOT" \
  --trace-tool "$TASK_ROOT/analysis_tools/trace_segment_cycles.py" \
  --capture-tool my_code/analyze_att_capture.py \
  --output-dir "$TASK_ROOT/analysis"

# These two canonical-tool modes were executed for SIMD0, SIMD1, SIMD2, SIMD3.
python3 "$TASK_ROOT/analysis_tools/trace_segment_cycles.py" \
  "$TASK_ROOT/analysis/seg_asm_simd0.json"
python3 "$TASK_ROOT/analysis_tools/trace_segment_cycles.py" \
  "$TASK_ROOT/analysis/seg_asm_simd0_representative.json" \
  --specific-part-representative-trace
```

The supplemental analyzer is also retained as
[analyze_tensor_wait_after_wmma_trace.py](analyze_tensor_wait_after_wmma_trace.py).
It verifies exact instruction counts, owner markers, trace-CSV anchors,
canonical sampled pairs, every adjacent task boundary, and additive timeline
closure, and imports the requested occupancy analyzer for machine-readable
WGP results.

Key evidence:

- [Machine-readable metrics](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/analysis/metrics.json)
- [Canonical segment summary, SIMD0](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/analysis/seg_asm_simd0.log)
- [Representative instruction drill-down, SIMD0](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/analysis/seg_asm_simd0_representative.log)
- [Independent CSV wait totals](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/csv_wait_validation.json)
- [Capture execution log](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/capture.log)
- [Analyzer execution log](history_runs/a07_tensor_wait_after_wmma_fresh_20260917T1517Z/analysis.log)

```text
supplemental analyzer SHA256:
90cebb6cbc23d4beccfcdd95853864b777c4ed12b35355ef81bafe8155c1c451

my_code/analyze_att_capture.py SHA256:
b7269e54fb4850673f627cbb3e6e734cbd84329fdcce01bc5143c707ec6acf2b

metrics.json SHA256:
600c4f11386d7de60b52b00805062d0909fd469bbdc534db3fd56347ef56631f
```
