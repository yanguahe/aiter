# `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt` thread-trace analysis and optimization plan

## Scope and result

This report analyzes:

```text
my_code/moe_gemm1_act1_optimized/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s
SHA256=8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169
```

The four-SIMD ATT capture was collected on `d01-3` while GPU/KFD was idle. The
const0 MoE preflight passed with identical output and reference hashes:

```text
moe_output_hash128=21291d9023c8af8a6324fe20f346a967
ref_output_hash128=21291d9023c8af8a6324fe20f346a967
```

The current steady-task median is `27,951 GFXCLK cycles`. Explicit waits account
for approximately `6,751 cycles/task`, or `24.15%`:

1. barrier stall: `4,083.5 cycles/task`, `14.61%`;
2. LDS `DScnt` stall: `1,908 cycles/task`, `6.83%`;
3. input/output TDM `TENSORcnt` stall: `858 cycles/task`, `3.07%`;
4. `s_wait_idle`: `13.5 cycles/task`, `0.05%`.

The dominant repeated site is the six-hit K-ring cluster barrier, with a
cross-owner median of `2,044 cycles/task`, or `7.31%`. The second major
bottleneck is the recurring `s_wait_dscnt 0x8` after LDS-read bursts. Together,
cluster synchronization and LDS completion explain most of the removable
latency.

## Capture and analysis method

The capture directory on `d01-3` is:

```text
my_code/moe_gemm1_act1_optimized/history_runs/
  heliosr-1b114-d01-3_20260916T082046Z_att/att/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt/
```

The capture command was:

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt \
RUN_VERIFY=0 \
AITER_ATT_E2E_ITERS=2 \
AITER_ATT_SIMD_LIST=0,1,2,3 \
AITER_ATT_TIMEOUT_SECONDS=300 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att
```

ATT selected the known eighth GEMM1 invocation directly:

```yaml
kernel_iteration_range: "[8]"
att_target_cu: 1
att_shader_engine_mask: "0x1"
att_simd_select: "0"  # repeated sequentially for 1, 2, and 3
```

Every SIMD capture contains one non-empty `.att`, one `code.json`, and one wave
JSON. No benchmark or ATT process remained after capture.

The analysis uses the workflow-specified parser:

```text
my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py
SHA256=6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

`analyze_all_simd_owner_trace.py` imports this parser and delimits persistent
tasks with consecutive dynamic occurrences of:

```asm
s_add_co_u32 s28, s28, 16
s_cmp_lt_u32 s28, 0x240
s_cbranch_scc0 5
```

The first interval per traced wave is dropped. Because SIMD1 and SIMD2 contain
only three steady intervals and each has one large tail, this report uses the
per-owner median as its primary statistic. Means and p90 values remain useful
for identifying tail behavior but are not used as the representative cycle
cost.

## Capture coverage

| logical owner | SIMD | steady tasks | median cycles/task | mean | p90 | maximum |
|---|---:|---:|---:|---:|---:|---:|
| A | 0 | 17 | 27,971 | 31,494.4 | 42,400.8 | 52,010 |
| B | 3 | 17 | 28,536 | 33,012.0 | 43,994.0 | 76,842 |
| ScaleA | 2 | 3 | 27,931 | 44,098.7 | 69,307.8 | 79,652 |
| ScaleB | 1 | 3 | 25,789 | 39,927.7 | 59,996.2 | 68,548 |
| cross-owner median | â€” | â€” | 27,951 | â€” | â€” | â€” |

The ScaleA/ScaleB means are dominated by one long interval each. Their medians
remain close to A/B and to the previous all-`NT_RT` all-SIMD capture, whose
cross-owner task midpoint was about `27.0k cycles`. The previous trace also
reported approximately `4.17k` barrier, `1.73k` DScnt, and `0.96k` TENSORcnt
cycles/task. That agreement supports the current robust ranking.

## Phase breakdown

The medians across the four owner captures are:

| phase | cycles/task | share of representative task |
|---|---:|---:|
| task boundary, next-task setup, and K hotloop | 23,484.3 | 84.02% |
| output descriptor/address setup and next stage-0 prefetch | 563.8 | 2.02% |
| SiLU banks 0-1 and first output launch | 1,747.0 | 6.25% |
| SiLU banks 2-3 | 1,896.5 | 6.79% |
| second output launch and task boundary | 150.5 | 0.54% |

The K hotloop and its synchronization dominate. The two SiLU/output-compute
regions are approximately `13.0%` together. Further epilogue tuning has a much
smaller ceiling than improving the K-ring barrier and LDS-read schedule.

## Explicit wait breakdown

The following values are per-owner medians across steady tasks:

| owner | task cycles | barrier | DScnt | TENSORcnt | idle | sum of wait-group medians |
|---|---:|---:|---:|---:|---:|---:|
| A | 27,971 | 4,091 (14.63%) | 1,957 (7.00%) | 1,047 (3.74%) | 9 (0.03%) | 7,104 (25.40%) |
| B | 28,536 | 4,076 (14.28%) | 2,286 (8.01%) | 915 (3.21%) | 151 (0.53%) | 7,428 (26.03%) |
| ScaleA | 27,931 | 5,167 (18.50%) | 772 (2.76%) | 450 (1.61%) | 9 (0.03%) | 6,398 (22.91%) |
| ScaleB | 25,789 | 3,555 (13.78%) | 1,859 (7.21%) | 801 (3.11%) | 18 (0.07%) | 6,233 (24.17%) |
| cross-owner median | 27,951 | 4,083.5 (14.61%) | 1,908 (6.83%) | 858 (3.07%) | 13.5 (0.05%) | 6,751 (24.15%) |

The explicit-wait ceiling is therefore about one quarter of a steady task.
This is not an achievable direct speedup because the waits enforce real
dependencies, but it shows where useful overlap must be created.

## Dominant K-ring cluster barrier

The six repeated K-ring waits appear at two owner-path PCs:

```text
A / ScaleA: 0x7c84  s_barrier_wait 0xfffd
B / ScaleB: 0x998c  s_barrier_wait 0xfffd
```

They follow the last four WMMA instructions of a K-ring phase:

```asm
v_wmma_scale_f32_32x16x128_f4 ...
v_wmma_scale_f32_32x16x128_f4 ...
s_cbranch_scc0 ...
s_barrier_wait 0xfffd
```

| owner | hits/task | median stall cycles/task | task share |
|---|---:|---:|---:|
| A | 6 | 1,981 | 7.08% |
| B | 6 | 2,107 | 7.38% |
| ScaleA | 6 | 2,567 | 9.19% |
| ScaleB | 6 | 1,478 | 5.73% |
| cross-owner median | 6 | 2,044 | 7.31% |

The hardware guide defines barrier ID `-3` as the user cluster barrier and
states that it completes only after all workgroups in the cluster signal. The
large owner spread means that the barrier is exposing different arrival times,
not slow execution of the barrier instruction itself. ScaleB waits least in
this capture and is the likely late local owner; ScaleA waits most. Historical
captures sometimes placed B closer to the critical arrival path, so the
optimization should target the B/ScaleB side rather than hard-code one owner
from a single trace.

The task-boundary cluster barrier at `0xcf50` is much smaller in the median
case. Reducing the six K-ring waits has the larger payoff.

## LDS completion stalls

`s_wait_dscnt` is the second bottleneck at `1,908 cycles/task`, or `6.83%`.
The dominant sites are repeated `s_wait_dscnt 0x8` instructions after the last
WMMA group for one operand bank and before new LDS values are consumed:

```text
A      0x76f4: median 882 cycles/task
B      0x9408: median 729 cycles/task
ScaleB 0x8cd0: median 979 cycles/task
```

ScaleA has only three intervals. Its DScnt median is `772 cycles/task`, while
one interval contains several multi-thousand-cycle waits and raises the mean to
`6,639.7`. This is a tail event, not the normal steady-task cost.

The decoded sequence around the hot waits is:

```asm
v_wmma_scale_f32_32x16x128_f4 ...
v_wmma_scale_f32_32x16x128_f4 ...
s_wait_dscnt 0x8
s_set_vgpr_msb ...
v_wmma_scale_f32_32x16x128_f4 ...
ds_load_b32 ...
```

This indicates insufficient issue distance between a preceding LDS load burst
and the first dependent WMMA group. The wait is not purely an LDS-bandwidth
limit; it also reflects the current placement of DS requests and VGPR reuse.

## TDM completion stalls

TENSORcnt contributes `858 cycles/task`, or `3.07%`. The largest one-time sites
are the full-setup/stage-0 waits:

```text
A      0x22f4  s_wait_tensorcnt 0x1: median 525 cycles
B      0x2f94  s_wait_tensorcnt 0x1: median 566 cycles
ScaleA 0x3818  s_wait_tensorcnt 0x1: median 345 cycles
ScaleB 0x40b8  s_wait_tensorcnt 0x1: median 791 cycles
```

The repeated `s_wait_tensorcnt 0x2` sites contribute smaller amounts. Stage-0
prefetch is already hiding most TDM latency, so TENSORcnt should not be the
first optimization target.

## Other long-latency instructions

The issue-timeline medians are:

| category | cycles/task | representative share |
|---|---:|---:|
| WMMA issue | 9,661.5 | 34.57% |
| barrier-wait timeline | 4,159.5 | 14.88% |
| LDS-read issue/completion gap | 3,316.0 | 11.86% |
| DScnt-wait timeline | 2,317.0 | 8.29% |
| SALU/control or scheduler gap | 1,983.0 | 7.09% |
| packed SiLU VALU | 1,751.0 | 6.26% |
| other VALU | 1,270.0 | 4.54% |
| EXP/RCP issue | 1,000.0 | 3.58% |
| TENSORcnt-wait timeline | 889.0 | 3.18% |
| explicit NOP | 243.0 | 0.87% |

The rows are independent cross-owner medians and are used for ranking; they do
not form an exact additive decomposition.

Some `ds_load_b128` and `ds_store_b64` events show large completion-latency
outliers:

- B `ds_load_b128` at `0x30dc`: mean `112.4`, maximum `1,895 cycles`;
- B `ds_store_b64` at `0xace8`: mean `192.3`, maximum `1,318 cycles`;
- ScaleA `ds_store_b64` at `0xb808`: mean `336.3`, maximum `1,005 cycles`.

These operations are asynchronous and can overlap later instructions. Their
completion latency must not be summed as serialized kernel time. The exposed
portion is represented by the measured DScnt waits.

The repeated `s_nop 0` after `s_set_pc_i64` has owner-average latency of roughly
`27-62 cycles` and an issue-timeline cost near `243 cycles/task` (`0.87%`). The
ISA documents `s_nop 0` as a one-cycle NOP, so the larger trace interval is
primarily indirect-control-flow and scheduling exposure around the instruction,
not the architectural execution latency of the NOP itself.

## Resource usage

The assembly metadata is:

```text
.amdhsa_group_segment_fixed_size 327680
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 184
.amdhsa_wavefront_size32 1
.amdhsa_next_free_vgpr 1024
.amdhsa_next_free_sgpr 104
```

| resource | kernel use | documented gfx1250 capacity | implication |
|---|---:|---:|---|
| LDS | 327,680 bytes = 320 KiB/WG | 320 KiB/WG | no additional LDS allocation is available |
| VGPR | 1,024/wave | 1,024/wave and 1,024/SIMD | one wave consumes the SIMD VGPR pool |
| SGPR | 104 named SGPRs | 106 normal SGPRs | only `s104:s105` remain available |
| scratch | 0 | â€” | no spill traffic |
| block | 128 threads | four wave32 | one logical owner wave per SIMD |

The local hardware source is:

```text
mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/
  MI400_Shader_Programming#65.txt
```

Relevant sections are 3.3.1.1 for 106 normal SGPRs, 3.3.2.1 for the 1,024-VGPR
limit, 3.3.4 for the 320 KiB LDS limit, 4.3.6.6 for cluster barriers, and the
data-dependency table for `S_WAIT_DSCNT`/`S_WAIT_TENSORCNT` semantics.

The kernel cannot hide stalls by adding another resident wave or workgroup:
VGPR and LDS allocations are already at their limits. Adding a conventional
extra input/output buffer is also impossible without reclaiming existing LDS
or VGPR state.

## Optimization plan

### Phase 1: precompute next-ring scalar state in the two free SGPRs

Goal: reduce the six K-ring cluster waits without changing the barrier protocol,
TDM order, LDS layout, or VGPR allocation.

1. Use currently unused `s104:s105` to stage the next ring selector and one
   next-ring descriptor/base value.
2. Place this scalar work between the final independent WMMA instructions of
   the current phase, after its inputs are known but before
   `s_barrier_wait 0xfffd`.
3. Consume the staged state immediately after the barrier instead of executing
   the current selector/address sequence on the next phase's critical path.
4. Implement separate B-path and ScaleB-path variants first. These are the
   likely late-arrival side in the current and historical captures.
5. Preserve every existing `s_barrier_signal`, `s_barrier_wait`, TDM issue, and
   `s_wait_tensorcnt` instruction in the first experiment.

This uses the only two free SGPRs but no extra LDS/VGPRs. The target is to lower
the K-ring barrier median from about `2,044` to below `1,600 cycles/task`, worth
up to roughly `1.5-2%` kernel time after accounting for partial overlap.

This is distinct from the previously tested output-descriptor hoist, which
changed the epilogue setup and produced no measurable gain. Phase 1 moves only
next-K-ring scalar state into otherwise exposed WMMA latency slots.

### Phase 2: increase LDS-load issue distance without relaxing dependencies

Goal: reduce the `s_wait_dscnt 0x8` median while retaining the known-correct
dependency point.

1. Build a static liveness map for each `ds_load_b32`/`ds_load_b128` destination
   VGPR block and its first consuming WMMA.
2. Move one independent LDS-load group at a time to the earliest point after
   the old value's last consumer.
3. Interleave those loads with WMMA groups that consume different VGPR banks.
4. Keep the final `s_wait_dscnt 0x8` at its original location initially. The
   first variants change issue distance only, not the wait threshold.
5. Start with the B and ScaleB paths at `0x9408` and `0x8cd0`, then apply the
   proven schedule to A.

The target is a `20-30%` reduction in DScnt stall, corresponding to roughly
`380-570 cycles/task` or `1.4-2.0%` of total cycles.

Previous experiments establish two hard constraints:

- moving `s_wait_dscnt 0x8` past a dependent WMMA caused random-data errors;
- inserting an extra `s_wait_dscnt 0xc` before the existing wait regressed
  performance by about `1%`.

Moving B/ScaleB TDM issue earlier also regressed the same-machine median, so the
first LDS experiment must leave TDM issue order unchanged.

The new implementation must therefore move only proven-independent load issues
and must not move the final dependency wait across a consumer.

### Phase 3: replace indirect K-ring dispatch only after Phases 1-2

The repeated `s_set_pc_i64`/`s_nop` transition costs about `243 cycles/task` in
the issue timeline. Test a compact direct conditional-branch dispatcher that
selects the four ring bodies without duplicating them. Do not simply delete the
NOP; its observed interval includes indirect-control-flow scheduling, and the
ISA permits required wait-state padding around control/register hazards.

This phase has a ceiling below `1%`, but it may also reduce instruction-fetch
variance. Code size must remain compatible with the existing 12-entry SQC
prefetch footprint.

### Phase 4: evaluate temporal LDS reuse for a deeper K ring

This is the high-risk, higher-ceiling option. The output LDS is not consumed
until the epilogue. Audit whether any output-only ranges can temporarily hold
additional K stages during the hotloop, then be fully retired and reused for
output. If a safe six- or eight-stage ring fits within the existing 320 KiB,
the number of K-ring cluster barriers could be reduced.

The required protocol is:

1. prove the extra input-ring and live output ranges never overlap;
2. complete all input TDM and cluster synchronization before switching that LDS
   region to output use;
3. retain the current output TDM retirement protocol before the next task;
4. avoid increasing VGPR count beyond 1,024 or SGPR count beyond 106.

Halving the six K-ring waits has an ideal ceiling near `1,000 cycles/task`
(`3.5-4%`). This phase should be attempted only after the scheduling-only
variants because an LDS lifetime mistake can corrupt data or hang the cluster.

## Validation sequence

For every candidate:

1. assemble and audit ABI/resource metadata;
2. run one const0 launch to detect hangs;
3. run random MoE e2e and require the established accuracy gate and hashes;
4. when correct, run a same-machine interleaved const0 benchmark against this
   all-`NT_RT` source;
5. promote only a repeatable improvement;
6. capture four-SIMD ATT for every implemented candidate, including rejected
   candidates, and compare the K-ring, DScnt, TENSORcnt, and complete-duration
   medians against this report.

The first implementation should be Phase 1, B/ScaleB next-ring scalar-state
precompute. It directly attacks the largest measured single stall, fits in the
two available SGPRs, and does not require changing LDS or TDM correctness
protocols.

## Optimization experiments

### Phase 1 feasibility audit: next-ring scalar-state precompute

Date: 2026-09-17

The source ISA was re-audited before consuming `s104:s105`. The premise of the
original Phase 1 proposal does not match the final all-`NT_RT` instruction
schedule:

1. the next input descriptor pointer, bounds, validity, and `s58` K offset are
   already updated before the final WMMA group of each ring body;
2. `s_cmp_lt_i32 s58, s59` is already separated from the dependent
   `s_cbranch_scc0` by four independent WMMAs;
3. the repeated `s_barrier_wait 0xfffd` is followed only by a direct branch to
   the next pre-unrolled compute body;
4. there is no remaining descriptor or ring-selector SALU sequence after the
   barrier that can be staged in `s104:s105` and then consumed after the wait.

Adding a duplicate selector or pointer in `s104:s105` would add instructions
without removing any instruction from the repeated critical path. No ISA was
generated for this invalid transformation. The two free SGPRs remain available
for a later transformation that has a real consumer.

### Experiment 1: skip steady direct-branch NOPs

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_steady_direct_skip_nop.s
SHA256=294bee488fd412c7317173ace7127cbc288cedb6e41652d829db752e4af9af9b
```

The candidate was copied from the accepted all-`NT_RT` source. It preserved the
initial indirect `s_set_pc_i64` targets and their required `s_nop 0`, but added
two labels after those NOPs. The two repeated direct K-ring branches were
retargeted to the new labels:

```text
0x8e48: branch to the instruction after the 0x7158 NOP
0xab50: branch to the instruction after the 0x8e6c NOP
```

No barrier, TDM, LDS operation, wait threshold, or resource metadata changed.
The hypothesis was that direct branches did not require the padding used by the
initial indirect entry and that removing six dynamic NOPs would recover part of
the measured `~243 cycles/task` NOP timeline.

Static assembly and link succeeded. Random and const0 MoE e2e correctness both
passed:

```text
random: logits_diff=3.3980e-06, rel_l2=2.6069e-03, pass=True
const0: logits_diff=0, rel_l2=0, hashes identical
```

Correctness result directories on `a07-3`:

```text
history_runs/heliosr-1b114-a07-3_20260916T175019Z_e2e-random/
history_runs/heliosr-1b114-a07-3_20260916T175121Z_e2e-const0/
```

An idle three-round alternating-order const0 comparison produced:

| case | GEMM1 samples (us) | GEMM1 median | change | MoE samples (us) | MoE median | change |
|---|---|---:|---:|---|---:|---:|
| all-`NT_RT` source | 504.789, 505.908, 504.713 | **504.789** | baseline | 1329.85, 1333.12, 1331.42 | **1331.42** | baseline |
| steady direct skip-NOP | 502.891, 505.948, 505.090 | **505.090** | **-0.06%** | 1330.92, 1331.63, 1331.87 | **1331.63** | **-0.02%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T175324Z_e2e-const0/
```

The four-SIMD ATT capture was collected immediately afterward while the GPU was
idle:

```text
history_runs/heliosr-1b114-a07-3_20260916T175518Z_att/
```

The trace comparison uses all consecutive persistent-task boundary intervals,
drops the first interval, and takes the median within each capture followed by
the cross-owner median:

| metric | source | candidate | change |
|---|---:|---:|---:|
| selected-SIMD duration | 1,203,994.5 | 1,216,886.5 | **+12,892.0 (+1.07%)** |
| steady task | 31,616.75 | 32,361.5 | **+744.75 (+2.36%)** |
| dynamic instructions/task | 7,341.5 | 7,335.5 | -6.0 |
| dynamic `s_nop 0` hits/task | 28.0 | 22.0 | **-6.0** |
| NOP issue timeline | 242.75 | 88.0 | **-154.75** |
| barrier stall | 6,069.75 | 6,621.25 | **+551.5** |
| `TENSORcnt` stall | 813.0 | 982.5 | **+169.5** |
| `DScnt` stall | 4,508.25 | 4,402.0 | -106.25 |

What matched the hypothesis:

- all six steady direct transitions skipped their NOP;
- the NOP timeline fell by `154.75 cycles/task`;
- random and const0 correctness proved that direct-branch execution does not
  require those two padding instructions.

What did not match the hypothesis:

- the saved issue distance exposed the next synchronization dependency instead
  of shortening the critical path;
- barrier stall increased by `551.5 cycles/task` and `TENSORcnt` stall increased
  by `169.5 cycles/task`;
- the selected-SIMD duration and same-run profiler both regressed.

The NOPs therefore provide useful latency cover even though they are not needed
for direct-branch correctness. Removing them makes the wave reach later waits
earlier, but does not make the cluster or TDM data ready earlier. This candidate
is rejected, is not added to `benchmark_history.sh`, and its ISA is deleted
after recording the result. Future control-flow work must move independent work
into this interval rather than merely remove the interval.

### Experiment 2: replace K-ring NOPs with bound-index precompute

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_ring_kbound_precompute.s
SHA256=98085a9bd6e59b532ce04b194ef3f69e3ae1fc44c1255323ef17db1d97b9e89d
```

This candidate started again from the accepted all-`NT_RT` source rather than
from the rejected skip-NOP candidate. It implemented a safe form of the Phase 1
idea with `s104`:

1. replace each of the eight static K-ring entry `s_nop 0` instructions with
   `s_add_co_u32 s104, s58, 0x500`;
2. keep the result live across the ring body;
3. remove the corresponding late `s_add_co_u32 s24, s58, 0x500`;
4. change the two descriptor-bound checks in each body from `s24` to `s104`;
5. preserve all TDM operations, barrier operations, waits, LDS addresses, and
   output-overlap control.

The transformation removed one dynamic instruction per K stage while keeping a
real instruction in the indirect-entry padding slot. Static audit found eight
NOP replacements, eight late-add removals, and sixteen compare rewrites. The
resource contract remained 320 KiB LDS, 1,024 VGPRs, and 104 SGPRs.

Assembly/link succeeded. Random and const0 MoE e2e both passed without changing
the established numerical behavior:

```text
random: logits_diff=3.3980e-06, rel_l2=2.6069e-03, pass=True
const0: logits_diff=0, rel_l2=0, hashes identical
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T180717Z_e2e-random/
```

An idle three-round alternating-order const0 comparison produced:

| case | GEMM1 samples (us) | GEMM1 median | change | MoE samples (us) | MoE median | change |
|---|---|---:|---:|---|---:|---:|
| all-`NT_RT` source | 507.896, 508.219, 507.199 | **507.896** | baseline | 1330.85, 1336.80, 1334.47 | **1334.47** | baseline |
| ring bound precompute | 508.206, 509.758, 506.566 | **508.206** | **-0.06%** | 1334.45, 1340.40, 1333.92 | **1334.45** | +0.00% |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T180827Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260916T181014Z_att/
```

| metric | source | candidate | change |
|---|---:|---:|---:|
| selected-SIMD duration | 1,197,172.5 | 1,195,389.5 | -1,783.0 (-0.15%) |
| selected-SIMD event span | 1,195,134.5 | 1,194,780.0 | -354.5 (-0.03%) |
| steady task | 32,063.25 | 32,061.25 | **-2.0 (-0.006%)** |
| dynamic instructions/task | 7,341.5 | 7,313.5 | **-28.0** |
| dynamic `s_nop 0` hits/task | 28.0 | 0.0 | **-28.0** |
| NOP issue timeline | 242.0 | 0.0 | **-242.0** |
| `s104` precompute hits/task | 0.0 | 28.0 | +28.0 |
| SALU/control timeline | 1,961.5 | 2,247.75 | **+286.25** |
| barrier stall | 6,065.5 | 5,878.5 | -187.0 |
| `DScnt` stall | 4,303.75 | 4,131.5 | -172.25 |
| `TENSORcnt` stall | 1,200.0 | 1,094.25 | -105.75 |

What matched the hypothesis:

- all 28 dynamic hotloop NOPs became useful precompute operations;
- 28 late bound-index adds were removed, reducing dynamic instructions by 28;
- barrier, DScnt, and TENSORcnt stall medians all moved in the favorable
  direction;
- correctness and every synchronization protocol remained intact.

What did not match the hypothesis:

- the cross-owner steady-task median improved by only two cycles;
- the issue-gap attribution moved from the removed NOP/late-add positions into
  other SALU instructions, increasing the aggregate SALU/control timeline by
  `286.25 cycles/task`;
- the ATT duration improvement was only `0.15%`, while the canonical profiler
  median regressed by `0.06%`. Both differences are below the machine's
  repeatable-noise threshold.

The original late add contributed about `161.75 cycles/task` of issue timeline,
while the source NOPs contributed about `242 cycles/task`. In the candidate the
early `s104` adds cost only 28 cycles/task, but the `s104` bounds compares and
other nearby scalar instructions inherited most of the exposed scheduling gap.
The optimization changes attribution and instruction count without shortening
the measured task critical path.

This candidate is rejected as no measurable improvement, is not added to
`benchmark_history.sh`, and its ISA and temporary generator are deleted after
recording the result. Future Phase 1 work should not extend a scalar value across
an entire ring body unless it also removes a real dependency from the critical
path.

### Experiment 3: single-sided B/ScaleB tail-load advance

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_dscnt9408_tail4_early.s
SHA256=a15e7b261086a1a02128bdcf5a977c92c4351a932471353882c5eb191c5c74e0
```

This candidate moved one group of four `ds_load_b128` instructions for
`v56:v71` ahead of two independent WMMAs in the static body feeding the decoded
B/ScaleB hotspot at `0x9408`. The moved destination VGPRs are not operands of the
crossed WMMAs. All wait positions and thresholds remained unchanged.

Assembly/link succeeded. Random and const0 MoE e2e passed:

```text
random: logits_diff=3.3980e-06, rel_l2=2.6069e-03, pass=True
const0: logits_diff=0, rel_l2=0, hashes identical
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T181947Z_e2e-random/
```

The first idle three-round comparison was too small to classify:

```text
source GEMM1 median    = 507.022 us
candidate GEMM1 median = 506.558 us
apparent change        = +0.09%
```

A nine-round alternating-order run reduced the difference further:

| case | GEMM1 samples (us) | GEMM1 median | change | MoE median | change |
|---|---|---:|---:|---:|---:|
| all-`NT_RT` source | 506.795, 505.669, 507.388, 505.092, 505.844, 505.828, 508.719, 507.565, 506.682 | **506.682** | baseline | **1333.61** | baseline |
| B-side tail4-early | 508.895, 508.828, 506.281, 509.062, 506.521, 505.566, 504.664, 505.022, 506.728 | **506.521** | **+0.03%** | **1333.29** | **+0.02%** |

Performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260916T182103Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260916T182848Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260916T182444Z_att/
```

| metric | source | candidate | change |
|---|---:|---:|---:|
| selected-SIMD duration | 1,203,333.5 | 1,192,217.0 | **-11,116.5 (-0.92%)** |
| steady task | 32,179.25 | 31,730.25 | **-449.0 (-1.40%)** |
| barrier stall | 6,184.25 | 6,204.0 | +19.75 |
| total `DScnt` stall | 3,954.75 | 4,121.0 | **+166.25** |
| `TENSORcnt` stall | 1,226.5 | 864.5 | -362.0 |
| LDS-read issue timeline | 3,441.0 | 3,352.25 | -88.75 |

The targeted B-side site improved:

```text
ScaleB 0x9408: 1,560.0 -> 1,314.0 cycles/task  (-246.0, -15.8%)
B      0x9408: 1,356.5 -> 1,080.5 cycles/task  (-276.0, -20.3%)
```

The corresponding A-side site regressed:

```text
A 0x76f4: 1,480.0 -> 2,102.0 cycles/task  (+622.0, +42.0%)
```

What matched the hypothesis:

- the crossed WMMA instructions were independent and correctness remained
  unchanged;
- the intended `0x9408` wait was reduced substantially;
- ATT task and selected-SIMD duration moved in the favorable direction.

What did not match the hypothesis:

- moving only one side of the shared schedule transferred DS pressure to the A
  owner, increasing total cross-owner DScnt;
- the profiler improvement fell from `0.09%` in three rounds to `0.03%` in nine
  rounds, which is measurement noise and not a promotable gain;
- the lower ATT median therefore did not translate into repeatable wall time.

The next experiment starts from this ISA and applies the corresponding A-side
tail-load move as well. Once that derived candidate exists, this single-sided
ISA and its generator are deleted. The acceptance condition for the balanced
version is lower total DScnt without moving the stall to either owner.

### Experiment 4: balanced A+B tail-load advance

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_dscnt_ab_tail4_early.s
SHA256=594b7455a19189bb92d93dad80c17dccaa2dd975e49aa1b6a9528a0aa91ad9e8
```

This candidate was derived from Experiment 3 by applying the matching four-load
advance to the A-side schedule. Both selected load groups therefore move ahead
of two independent WMMAs, while all waits and thresholds remain unchanged.

Assembly/link succeeded. Random and const0 MoE e2e passed:

```text
random: logits_diff=3.3980e-06, rel_l2=2.6069e-03, pass=True
const0: logits_diff=0, rel_l2=0, hashes identical
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T183616Z_e2e-random/
```

The first three-round run started with a transient non-zero GPU-use/VRAM sample
and is not accepted as performance evidence. A subsequent idle nine-round
alternating-order run produced:

| case | GEMM1 samples (us) | GEMM1 median | change | MoE median | change |
|---|---|---:|---:|---:|---:|
| all-`NT_RT` source | 506.395, 507.458, 506.569, 505.641, 506.577, 507.104, 507.422, 506.048, 506.908 | **506.577** | baseline | **1332.95** | baseline |
| balanced A+B tail4-early | 507.283, 509.113, 506.391, 506.942, 503.547, 506.233, 506.722, 508.049, 506.638 | **506.722** | **-0.03%** | **1333.86** | **-0.07%** |

Accepted performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T183917Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260916T184216Z_att/
```

| metric | source | candidate | change |
|---|---:|---:|---:|
| selected-SIMD duration | 1,218,709.5 | 1,224,370.0 | **+5,660.5 (+0.46%)** |
| selected-SIMD event span | 1,217,939.0 | 1,222,578.0 | **+4,639.0 (+0.38%)** |
| steady task | 32,388.75 | 32,346.0 | -42.75 (-0.13%) |
| barrier stall | 6,349.25 | 5,967.5 | -381.75 |
| total `DScnt` stall | 4,213.25 | 4,085.25 | -128.0 |
| `TENSORcnt` stall | 900.5 | 1,271.5 | **+371.0** |

The two intended owner groups mostly improved:

```text
A      0x76f4: 1,967.0 -> 1,676.5  (-290.5)
ScaleA 0x76f4: 1,435.0 -> 1,294.0  (-141.0)
B      0x9408: 1,472.5 -> 1,321.5  (-151.0)
ScaleB 0x9408: 1,361.5 -> 1,535.5  (+174.0)
```

What matched the hypothesis:

- the A hotspot that regressed in Experiment 3 was recovered;
- three of four owner-specific target waits fell;
- cross-owner total DScnt and barrier stall both decreased;
- correctness remained unchanged.

What did not match the hypothesis:

- ScaleB's target DScnt increased by 174 cycles/task;
- earlier DS traffic increased `TENSORcnt` stall by 371 cycles/task, consistent
  with additional contention on the shared LDS/WGP/TDM path;
- steady-task improvement was only 43 cycles and complete selected-SIMD duration
  regressed by `0.46%`;
- the nine-round canonical profiler median regressed by `0.03%`.

Moving a four-request burst earlier is too coarse: it improves completion time
for selected owners but creates a new burst and competes with TDM progress. The
next derived experiment keeps both sides symmetric but interleaves two loads,
one WMMA, two loads, and the second WMMA instead of moving all four loads as one
burst. This candidate is rejected, is not added to `benchmark_history.sh`, and
is deleted after the interleaved descendant is generated.

### Experiment 5: interleave the balanced A+B tail loads around the WMMAs

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_dscnt_ab_tail4_interleave.s
SHA256=a02eb7d98eaf30b1d21d0f8f41f0601f2d3970c9563063cee749a6f52dc973b5
```

This candidate was copied from the rejected balanced A+B tail-load candidate.
It retained the same four-load movement on both A and B sides but split each
four-request burst around the two independent WMMAs:

```text
balanced tail4-early: load0, load1, load2, load3, WMMA0, WMMA1
interleaved:          load0, load1, WMMA0, load2, load3, WMMA1
```

The destination VGPRs remain independent of the crossed WMMAs. No wait threshold,
barrier, TDM command, LDS address, kernarg, grid, output conversion, or numerical
operation was changed. This preserved the kernel's function and precision.

Assembly/link succeeded. Random and const0 MoE e2e both passed:

```text
random: logits_diff=3.3980e-06, rel_l2=2.6069e-03, pass=True
const0: logits_diff=0, rel_l2=0, hashes identical
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T185036Z_e2e-random/
```

The idle three-round comparison was a repeatable regression:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| all-`NT_RT` source | **505.794 us** | baseline | **1332.87 us** | baseline |
| interleaved A+B tail loads | **506.886 us** | **-0.22%** | **1335.37 us** | **-0.19%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T185239Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260916T185457Z_att/
```

| metric | source | candidate | change |
|---|---:|---:|---:|
| selected-SIMD duration | 1,202,784.5 | 1,211,767.0 | **+8,982.5 (+0.75%)** |
| selected-SIMD event span | 1,201,972.5 | 1,211,294.5 | **+9,322.0 (+0.78%)** |
| steady task | 32,277.25 | 32,131.5 | -145.75 (-0.45%) |
| barrier stall | 5,835.0 | 6,628.0 | **+793.0** |
| total `DScnt` stall | 4,315.5 | 4,102.5 | **-213.0** |
| `TENSORcnt` stall | 1,152.0 | 1,237.5 | +85.5 |

Owner-specific exact waits were redistributed as follows:

```text
A      0x76f4: 1,640.5 -> 1,573.5  (-67.0)
ScaleA 0x76f4: 1,058.0 -> 1,125.5  (+67.5)
B      0x9408: 1,893.0 -> 1,834.0  (-59.0)
ScaleB 0x9408: 1,347.0 -> 1,005.5  (-341.5)
```

What matched the hypothesis:

- spreading the four-load burst around useful WMMA work reduced aggregate
  cross-owner `DScnt` by `213 cycles/task`;
- the ScaleB hotspot fell by `341.5 cycles/task`;
- the steady-task segment improved by `0.45%`;
- correctness, output hashes, ABI, and numerical precision were unchanged.

What did not match the hypothesis:

- the revised issue order increased the dominant synchronization/barrier stall
  by `793 cycles/task`, more than the entire DScnt reduction;
- `TENSORcnt` also increased by `85.5 cycles/task`;
- selected-SIMD duration regressed by `0.75%`, and the canonical profiler
  independently measured a `0.22%` GEMM1 regression.

The interleaving again transfers exposed LDS latency to the shared cluster/TDM
synchronization path. It does not shorten the end-to-end task critical path.
This candidate is rejected, is not added to `benchmark_history.sh`, and its ISA
and temporary generator are deleted after recording the result.

### Phase 4 feasibility audit: a deeper K ring does not fit this kernel

No ISA candidate was generated for this phase because the required LDS-capacity
proof fails before implementation. Generating a kernel that aliases live input
or output data would violate the mandatory function and precision constraint and
could hang the 4x4 cluster.

The current four-stage input ring occupies:

```text
A payload       32 KiB/stage
B payload       32 KiB/stage
ScaleA           2 KiB/stage
ScaleB           2 KiB/stage
--------------------------------
one K256 stage  68 KiB
four stages    272 KiB
```

The concrete LDS map leaves only `[0x24000,0x30000)`, or `48 KiB`, unused.
The hardware and kernel limits are:

```text
allocated LDS                 = 320 KiB
current four-stage input ring = 272 KiB
free address space            =  48 KiB
five-stage input ring         = 340 KiB  (20 KiB over limit)
six-stage input ring          = 408 KiB  (88 KiB over limit)
eight-stage input ring        = 544 KiB (224 KiB over limit)
```

The output LDS does not provide an additional hotloop allocation. Its four live
wave regions for the observed final ring selection occupy `72 KiB` inside the
stage-2/stage-3 A/B payload ranges. Those addresses become output storage only
after their final input consumers retire. Reusing them earlier would overwrite
live matrix operands; reusing them later cannot remove any preceding steady
K-ring barrier because the hotloop has already ended.

The persistent-boundary capacity proof is similarly negative. Keeping the
current output staging live while holding a complete next-task four-stage input
ring would require:

```text
272 KiB next-task input + 72 KiB current output = 344 KiB
```

This exceeds the documented `320 KiB` LDS limit by `24 KiB`. The accepted
stage-0 prefetch works because stage 0 does not overlap the live output ranges;
it does not create capacity for a complete deeper ring.

Reclaiming all four existing ScaleA/ScaleB LDS sets would free only `16 KiB`.
Together with the `48 KiB` hole this is `64 KiB`, still `4 KiB` short of one
complete `68 KiB` K256 stage. More importantly, the scales are live WMMA inputs.
Replacing their TDM/LDS path with direct loads requires a new load schedule and
new live register state. The kernel already declares `1024 VGPRs/wave` and uses
the full `1024 VGPR/SIMD` hardware allocation; it also names `104` of the `106`
normal SGPRs. There is no register allocation available for a second live scale
representation or another operand tile while all accumulators are live.

A partial fifth stage in the `48 KiB` hole also cannot remove a ring barrier. At
least `20 KiB` of that stage would still have to wait for an old ring slot to be
released, so the synchronization dependency remains. Experiments 3-5 already
show that issuing a partial LDS-read group earlier only redistributes exposed
latency among DScnt, TENSORcnt, and the cluster barrier without reducing the
complete task critical path.

The hardware facts used by this audit are documented in
`MI400_Shader_Programming#65.txt`: section 1.1.1.1 and section 3.3.4 specify the
`320 KiB` LDS limit, section 3.3.1.1 specifies `106` normal SGPRs, and section
3.3.2.1 specifies up to `1024` VGPRs per wave and `1024` VGPRs per MI450 SIMD.
The ISA metadata independently confirms `327680` LDS bytes, `1024` VGPRs, and
`104` named SGPRs.

Phase 4 is therefore rejected as an incremental optimization. A real deeper
ring requires a different kernel family that first changes at least one of the
following structural choices:

- reduce the M/N tile so fewer accumulator VGPRs and less input LDS are live;
- replace the scale data path and prove an exact-equivalent direct-load schedule;
- replace or shrink the output-LDS staging protocol;
- reduce ring-stage payload size and rebuild the WMMA schedule around it.

Each option changes the core LDS/register/dataflow design rather than the local
schedule of this ISA. It needs a separate implementation and validation plan.

## Previous stopping point and resumed search

The accepted all-`NT_RT` source remains the best version in this optimization
series:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s
SHA256=8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169
```

No tested candidate produced a repeatable profiler improvement while preserving
random and const0 MoE e2e correctness. The observed local transformations have
reached a common limit: barrier, DScnt, and TENSORcnt expose different parts of
the same shared LDS/WGP/TDM dependency chain, so reducing one local wait moves
the latency to another owner or synchronization point.

At this point the original four phases had no remaining implementation with a
positive repeatable result. The user subsequently requested that the search
continue. The resumed experiments below keep the same non-negotiable rule:
kernel function, ABI, numerical operation order, and output precision must not
change. A candidate that fails random MoE e2e is never timed as a valid result.

### Experiment 6: dual-issue epilogue constant initialization

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_epilogue_dual_const_init.s
SHA256=b2e451c581527deca5a848b2245c8f183a3a69e7a76b4c419cad3c8de45f8edc
```

Each of the four accumulator-bank epilogues initialized two replicated vector
constants with four scalar `v_mov_b32_e32` instructions. The candidate copied
the accepted all-`NT_RT` ISA and replaced each group with two legal VOPD pairs:

```asm
v_dual_mov_b32 v96, 0xbfb8aa3b :: v_dual_mov_b32 v97, 0xbfb8aa3b
v_dual_mov_b32 v98, s102       :: v_dual_mov_b32 v99, s102
```

The destinations satisfy the VOPD even/odd-bank rule. The bit patterns written
to `v96:v99` are unchanged, and no arithmetic, conversion, TDM, LDS address,
wait, barrier, metadata, ABI, or launch parameter changed. Static dynamic
instruction count falls by eight instructions per task; encoded byte size is
unchanged because one VOPD pair occupies the same bytes as two VOP1 moves.

Assembly/link and random MoE e2e passed:

```text
random: logits_diff=3.39799e-06, rel_l2=2.60689e-03, pass=True
MoE output hash128: 1556fc617347e2dabc9cff19dbfd822b
ref output hash128: 1a5d22911ba167160b4f2c12092a5193
```

Const0 also remained bitwise identical:

```text
logits_diff=0
rel_l2=0
MoE output hash128=21291d9023c8af8a6324fe20f346a967
ref output hash128=21291d9023c8af8a6324fe20f346a967
```

Correctness result directories:

```text
history_runs/heliosr-1b114-a07-3_20260916T195025Z_e2e-random/
history_runs/heliosr-1b114-a07-3_20260916T195105Z_e2e-const0/
```

Two independent idle nine-round alternating-order comparisons produced:

| run | case | GEMM1 median | change | MoE median | change |
|---|---|---:|---:|---:|---:|
| 1 | all-`NT_RT` source | 507.413 us | baseline | 1334.38 us | baseline |
| 1 | dual constant init | 506.202 us | +0.24% | 1333.19 us | +0.09% |
| 2 | all-`NT_RT` source | 504.998 us | baseline | 1332.06 us | baseline |
| 2 | dual constant init | 504.676 us | +0.06% | 1331.14 us | +0.07% |

Performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260916T195612Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260916T200332Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260916T195932Z_att/
```

Cross-owner medians from that single source/candidate capture were:

| metric | source | candidate | change |
|---|---:|---:|---:|
| task mean | 34,577.79 | 33,917.94 | -659.85 (-1.91%) |
| task median | 33,182.0 | 32,241.5 | -940.5 (-2.83%) |
| full-wave mean | 1,222,582.5 | 1,197,363.0 | -25,219.5 (-2.06%) |
| barrier stall | 6,973.74 | 6,631.88 | -341.85 |
| total `DScnt` stall | 5,249.88 | 4,325.94 | -923.94 |
| `TENSORcnt` stall | 1,712.00 | 2,780.38 | +1,068.38 |
| packed SiLU issue timeline | 1,751.00 | 1,744.97 | -6.03 |

What matched the hypothesis:

- all four banks received exactly the same constants and both random and const0
  correctness remained unchanged;
- the packed-SiLU issue timeline fell by about six cycles/task, consistent with
  eliminating eight dynamic instructions;
- both profiler repetitions moved in the favorable direction.

What did not match the hypothesis:

- the profiler gains shrank from `0.24%` to `0.06%`, below a repeatable
  promotion threshold;
- the single ATT pair shows changes of hundreds of cycles in unrelated DScnt,
  TENSORcnt, and barrier categories, far larger than an eight-instruction edit;
  these changes are capture variance rather than a causal effect of the VOPD
  replacement;
- the encoded code size did not shrink, so the change does not improve the SQC
  instruction-prefetch footprint.

The candidate is rejected as a sub-noise instruction-count optimization. It is
not added to `benchmark_history.sh`; its ISA is deleted after this result is
recorded. The next experiment targets output LDS request count, where one ISA
change can remove 32 DS instructions per task while preserving every output byte.

### Experiment 7: pair output LDS stores with `ds_store_2addr_b64`

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_output_ds_2addr.s
SHA256=52448eecd1c281b46f811ef9ae64ae473eaf07cf3cd8153ca04f9ccfd4877b64
```

The source writes each accumulator bank to output LDS with sixteen
`ds_store_b64` instructions. Each adjacent pair writes two disjoint eight-byte
values separated by 16 bytes. CDNA5 `ds_store_2addr_b64` expresses exactly this
operation because its two offsets are independently encoded in eight-byte
units.

For each bank, the candidate precomputed three additional row-group addresses
in dead `v92:v94` and replaced the sixteen stores with eight two-address stores:

```asm
v_add_nc_u32_e32 v92, 0x900,  v91
v_add_nc_u32_e32 v93, 0x1200, v91
v_add_nc_u32_e32 v94, 0x1b00, v91

ds_store_2addr_b64 v91, v[100:101], v[102:103] offset1:2
ds_store_2addr_b64 v91, v[104:105], v[106:107] offset0:4 offset1:6
```

The same pattern is used at bases `v92`, `v93`, and `v94`. The effective LDS
addresses and source BF16 words are byte-for-byte identical to the original
stores. Across four banks, the edit changes 64 single-address stores into 32
two-address stores and adds 12 early address calculations, for a net reduction
of 20 dynamic instructions per task. Arithmetic, conversion, output TDM,
barriers, ABI, grid, and numerical precision are unchanged.

Assembly/link and random MoE e2e passed:

```text
random: logits_diff=3.39799e-06, rel_l2=2.60689e-03, pass=True
MoE output hash128: 1556fc617347e2dabc9cff19dbfd822b
ref output hash128: 1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T200847Z_e2e-random/
```

Two independent idle nine-round alternating-order comparisons disagreed:

| run | case | GEMM1 median | change | MoE median | change |
|---|---|---:|---:|---:|---:|
| 1 | all-`NT_RT` source | 505.950 us | baseline | 1331.20 us | baseline |
| 1 | output DS 2addr | 505.330 us | +0.12% | 1331.14 us | +0.00% |
| 2 | all-`NT_RT` source | 505.354 us | baseline | 1332.61 us | baseline |
| 2 | output DS 2addr | 506.202 us | **-0.17%** | 1332.18 us | +0.03% |

Performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260916T200939Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260916T201634Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260916T201233Z_att/
```

Cross-owner medians from the ATT pair were:

| metric | source | candidate | change |
|---|---:|---:|---:|
| task mean | 33,419.21 | 33,915.41 | **+496.21 (+1.49%)** |
| task median | 32,414.0 | 31,764.5 | -649.5 (-2.00%) |
| full-wave mean | 1,208,881.5 | 1,218,991.5 | **+10,110.0 (+0.84%)** |
| barrier stall | 6,933.38 | 6,421.68 | -511.71 |
| total `DScnt` stall | 4,122.76 | 5,623.82 | **+1,501.06** |
| `TENSORcnt` stall | 1,740.85 | 1,705.41 | -35.44 |
| LDS-write issue timeline | 252.41 | 177.41 | **-75.00 (-29.71%)** |

The median output-drain `s_wait_dscnt 0x0` cost increased from approximately
`156.6` to `294.9 cycles/task`. This is consistent with each two-address DS
instruction doing more work before retiring, despite the lower issue count.

What matched the hypothesis:

- the candidate preserved every output byte and passed random and const0 MoE
  e2e correctness;
- output LDS store instruction count was halved;
- the LDS-write issue timeline fell by about `29.7%`.

What did not match the hypothesis:

- output-drain latency increased substantially;
- aggregate DScnt stall increased by about `1,501 cycles/task` in ATT;
- the full-wave ATT duration regressed by `0.84%`;
- the two profiler repetitions changed sign (`+0.12%`, then `-0.17%`), proving
  that the apparent first-run gain was noise.

The output path is completion-limited rather than instruction-issue-limited.
Combining two stores into one request reduces issue traffic but worsens the
retirement critical path. The candidate is rejected, is not added to
`benchmark_history.sh`, and its ISA is deleted after recording the result.

### Experiment 8: incrementally advance the next-task A/ScaleA pointers

Candidate before deriving the follow-up:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_incremental_a_sa_ptr.s
SHA256=0573dfadc6d536a7d34bad8799ec22f1b1318fc7c5eb0ec43a1eaa8a2d19801b
```

For the persistent `+16` task stride, local M tile `s55` is unchanged. Therefore
the next-task A and ScaleA tensor pointers change only when the expert changes,
and their pointer deltas are exactly the same as their already-computed expert
base deltas:

```text
A expert/base delta      = s29 * 0x380000
ScaleA expert/base delta = s29 * 0x038000
s29                      = 0 or 4
```

The candidate copied the accepted all-`NT_RT` ISA, applied each delta to both
the expert base and the corresponding existing tensor pointer, and deleted the
two redundant five-instruction 64-bit pointer rematerializations. It adds two
`s_add_nc_u64` operations and removes ten multiply/add operations, for a net
reduction of eight SALU instructions on every non-final persistent transition.

The identity is exact:

```text
old next A pointer = (old A base + expert_delta) + M * strideA
                   = old A pointer + expert_delta
```

ScaleA follows the same equation. No pointer approximation, data-layout change,
arithmetic change, wait relaxation, or ABI change is involved.

Assembly/link and random MoE e2e passed:

```text
random: logits_diff=3.39799e-06, rel_l2=2.60689e-03, pass=True
MoE output hash128: 1556fc617347e2dabc9cff19dbfd822b
ref output hash128: 1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T202309Z_e2e-random/
```

The idle nine-round alternating-order comparison produced:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| all-`NT_RT` source | 507.052 us | baseline | 1332.75 us | baseline |
| incremental A/ScaleA pointers | 507.383 us | **-0.07%** | 1333.61 us | **-0.06%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260916T202356Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T025749Z_att/
```

Cross-owner ATT medians were:

| metric | source | candidate | change |
|---|---:|---:|---:|
| task mean | 33,901.00 | 33,224.18 | -676.82 (-2.00%) |
| task median | 32,336.0 | 32,308.5 | -27.5 (-0.09%) |
| full-wave mean | 1,192,355.0 | 1,199,004.0 | **+6,649.0 (+0.56%)** |
| barrier stall | 6,878.26 | 6,455.15 | -423.12 |
| total `DScnt` stall | 4,673.35 | 4,346.85 | -326.50 |
| `TENSORcnt` stall | 1,677.59 | 1,598.35 | -79.24 |
| SALU/control timeline | 1,946.50 | 1,931.65 | -14.85 |

What matched the hypothesis:

- pointer values and all random/const0 results were unchanged;
- the SALU/control timeline fell by `14.85 cycles/task`;
- the intended A/ScaleA pointer rematerialization disappeared from the dynamic
  transition path.

What did not match the hypothesis:

- the canonical profiler regressed by `0.07%`;
- complete full-wave ATT duration regressed by `0.56%` even though several
  per-task wait categories moved favorably;
- reducing only the A/ScaleA half leaves both B and ScaleB pointer
  rematerializations on the same prefetch critical path, so stage-0 TDM issue is
  not advanced enough to change the end-to-end dependency.

This partial candidate is not promotable. It is retained only long enough to
derive a full four-pointer incremental version, then deleted. The follow-up uses
the kernel's existing fixed `s14=0xe00` and `s16=0xe0` stride contract to update
B and ScaleB with exact `+16`/wrap deltas as well.

### Experiment 9: incrementally advance all four next-task tensor pointers

Accepted candidate:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_incremental_all_ptr.s
SHA256=5cb009b909427b231f281c014cfafc578e8e9d21c90049413437970234dea72a
```

This candidate was copied from Experiment 8 and extends the exact incremental
update to B and ScaleB. The kernel itself fixes:

```text
s14 = 0xe00  # B stride
s16 = 0xe0   # ScaleB stride
```

For the two possible persistent transitions, the complete byte deltas are:

| transition | B delta | ScaleB delta |
|---|---:|---:|
| same expert, `N += 16` | `0xe00000` | `0xe0000` |
| expert `+= 4`, `N -= 8` | `0x4d00000` | `0x4d0000` |

Because `s29` is exactly `0` or `4`, both cases are represented without a
branch or second literal operand:

```asm
s_mul_i32    s24, s29, 0xfc0000
s_add_co_u32 s24, s24, 0xe00000
s_add_nc_u64 s[74:75], s[74:75], s[24:25]

s_mul_i32    s24, s29, 0xfc000
s_add_co_u32 s24, s24, 0xe0000
s_add_nc_u64 s[78:79], s[78:79], s[24:25]
```

For `s29=0`, these yield the same-expert deltas. For `s29=4`, they yield the
expert-wrap deltas exactly. Together with the A/ScaleA changes, all four
five-instruction pointer rematerializations are removed. The replacement adds
eight pointer-update instructions, for a net reduction of twelve SALU
instructions on every non-final persistent transition.

No global address, LDS address, descriptor field, input value, arithmetic
operation, output conversion, wait, barrier, ABI field, resource declaration,
or launch parameter changes.

Assembly/link and random MoE e2e passed:

```text
random: logits_diff=3.39799e-06, rel_l2=2.60689e-03, pass=True
MoE output hash128: 1556fc617347e2dabc9cff19dbfd822b
ref output hash128: 1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T030643Z_e2e-random/
```

Two independent idle nine-round alternating-order comparisons both improved:

| run | case | GEMM1 median | change | MoE median | change |
|---|---|---:|---:|---:|---:|
| 1 | all-`NT_RT` source | 508.674 us | baseline | 1334.67 us | baseline |
| 1 | incremental all pointers | 507.523 us | **+0.23%** | 1333.71 us | **+0.07%** |
| 2 | all-`NT_RT` source | 508.735 us | baseline | 1334.14 us | baseline |
| 2 | incremental all pointers | 507.325 us | **+0.28%** | 1333.09 us | **+0.08%** |

Performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260917T030748Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260917T031124Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T031438Z_att/
```

Cross-owner ATT medians were:

| metric | source | candidate | change |
|---|---:|---:|---:|
| task mean | 32,874.74 | 32,798.06 | -76.68 (-0.23%) |
| task median | 31,903.5 | 31,194.0 | -709.5 (-2.22%) |
| full-wave mean | 1,204,764.0 | 1,179,336.0 | **-25,428.0 (-2.11%)** |
| barrier stall | 6,583.65 | 6,271.53 | -312.12 |
| total `DScnt` stall | 4,563.71 | 4,110.26 | -453.44 |
| `TENSORcnt` stall | 1,128.29 | 1,377.35 | +249.06 |
| SALU/control timeline | 1,949.24 | 1,920.53 | **-28.71** |
| TDM issue timeline | 102.82 | 102.82 | 0.00 |

What matched the hypothesis:

- random and const0 results, hashes, ABI, and numerical precision remained
  unchanged;
- both independent profiler comparisons improved in the same direction;
- SALU/control issue time decreased by `28.71 cycles/task`;
- complete full-wave ATT duration decreased by `2.11%`;
- TDM issue time was unchanged, confirming that the optimization advances the
  same descriptor rather than changing the memory operation.

What did not fully match the hypothesis:

- `TENSORcnt` increased by `249.06 cycles/task`, so the earlier descriptor issue
  exposes more of the same TDM completion latency at the later wait;
- only part of the static twelve-instruction saving appears directly as lower
  steady-task mean because synchronization variance remains dominant.

The profiler improvement is small but repeatable, and the ATT direction is
consistent at complete-wave scope. This candidate is accepted, retained, and
added to `benchmark_history.sh` as `incremental_all_ptr`. A checksum-guarded
reproducer is retained as `build_incremental_all_ptr_variant.py`.

### Experiment 10: stack dual constant initialization on the pointer winner

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_incremental_all_ptr_dual_const.s
SHA256=86ff093445d27b90c1812933e8dff806ca8ce882e6ed67a6a0fb4cccf2c151f8
```

This candidate was copied from the accepted `incremental_all_ptr` ISA and added
the exact `v_dual_mov_b32` constant initialization from Experiment 6. The edit
is orthogonal to the pointer update and preserves the same constant bits,
arithmetic order, ABI, LDS layout, TDM operations, waits, and barriers.

Random MoE e2e passed with the established result:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T032303Z_e2e-random/
```

The idle nine-round comparison against the pointer winner produced:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `incremental_all_ptr` | 506.009 us | baseline | 1331.86 us | baseline |
| pointer winner + dual constants | 505.645 us | +0.07% | 1332.42 us | **-0.04%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T032406Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T032722Z_att/
```

| metric | pointer winner | stacked candidate | change |
|---|---:|---:|---:|
| task mean | 34,188.97 | 34,732.47 | **+543.50 (+1.59%)** |
| task median | 32,979.0 | 32,743.0 | -236.0 (-0.72%) |
| full-wave mean | 1,217,443.0 | 1,229,390.5 | **+11,947.5 (+0.98%)** |
| barrier stall | 6,544.12 | 7,417.53 | **+873.41** |
| total `DScnt` stall | 5,037.59 | 4,882.09 | -155.50 |
| `TENSORcnt` stall | 1,840.35 | 1,895.21 | +54.85 |
| SALU/control timeline | 1,921.12 | 1,914.18 | -6.94 |
| packed SiLU timeline | 1,751.00 | 1,745.00 | -6.00 |

What matched the hypothesis:

- numerical behavior remained unchanged;
- the packed-SiLU timeline again fell by about six cycles/task;
- GEMM1 moved slightly in the favorable direction.

What did not match the hypothesis:

- the `0.07%` GEMM1 change is below the repeatable promotion threshold;
- MoE e2e regressed slightly;
- barrier stall increased by `873.41 cycles/task`, and complete full-wave ATT
  duration regressed by `0.98%`.

The small VOPD instruction saving does not shorten the pointer winner's critical
path. This stacked candidate is rejected, is not added to `benchmark_history.sh`,
and its ISA is deleted after recording the result.
### Experiment 11: skip redundant tensor-pointer rematerialization

Candidate before deriving the follow-up:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_incremental_all_ptr_skip_recompute.s
SHA256=68ded5b15ec284fdc5fd5be89cb3865db93c8bc1092155e0edb962f5cacd9c9a
```

The accepted pointer winner updates `s72:s79` during the current task epilogue,
before issuing next-task stage-0 TDM. The prefetched next-task entry nevertheless
recomputed all four pointers from the expert bases and tile coordinates. This
candidate retained that calculation for the first/full-setup task and skipped
it only when `s101=1` identifies the already-prefetched persistent path:

```asm
s_cmp_eq_u32 s101, 0
s_cbranch_scc0 .Lmoe_prefetched_tensor_bases_ready
    # original four pointer materializations, full-setup path only
.Lmoe_prefetched_tensor_bases_ready:
```

The prefetched path skips sixteen SALU operations and executes two branch-control
operations instead. Pointer values are those already proven in Experiment 9;
the first task still uses the complete original setup.

Assembly/link and random MoE e2e passed with unchanged accuracy and hashes:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T033503Z_e2e-random/
```

The idle nine-round comparison against the pointer winner produced:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `incremental_all_ptr` | 504.474 us | baseline | 1330.23 us | baseline |
| skip prefetched pointer recompute | 504.687 us | **-0.04%** | 1330.31 us | **-0.01%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T033611Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T033930Z_att/
```

| metric | pointer winner | skip-recompute | change |
|---|---:|---:|---:|
| task mean | 33,648.97 | 33,437.50 | -211.47 (-0.63%) |
| task median | 31,799.0 | 31,689.0 | -110.0 (-0.35%) |
| full-wave mean | 1,193,200.5 | 1,195,441.5 | **+2,241.0 (+0.19%)** |
| barrier stall | 7,101.47 | 5,983.00 | -1,118.47 |
| total `DScnt` stall | 3,925.09 | 5,249.71 | **+1,324.62** |
| `TENSORcnt` stall | 2,235.71 | 1,401.68 | -834.03 |
| SALU/control timeline | 1,920.47 | 1,906.29 | -14.18 |

What matched the hypothesis:

- the prefetched-path SALU/control timeline decreased by `14.18 cycles/task`;
- random and const0 correctness remained unchanged;
- barrier and TENSORcnt exposure moved in the favorable direction.

What did not match the hypothesis:

- DScnt increased by `1,324.62 cycles/task`, absorbing the saved setup time;
- full-wave duration regressed by `0.19%`;
- the canonical profiler regressed slightly.

Skipping the common pointer block makes the following LDS request sequence arrive
earlier without balancing the per-wave transition work. This partial candidate
is not promoted. It is retained only long enough to derive a role-specialized
state-update version, then deleted.
### Experiment 12: role-specialized persistent input state

Candidate before deriving the follow-up:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_role_state.s
SHA256=c1898ec61417e4446c9b681f93bd425474e8ba23eb3483dc0895b14dbda5c9b8
```

This candidate was copied from the rejected skip-recompute descendant of the
pointer winner. Each wave owns exactly one input tensor role throughout the
persistent loop: A, B, ScaleA, or ScaleB. The prior code nevertheless advanced
all four expert-local bases and all four tensor pointers independently in every
wave before branching to the role-specific descriptor builder.

The candidate keeps the output base update common, moves each input base and
pointer update into its existing role branch, and leaves the other three private
SGPR states stale because that wave never selects or consumes them. The complete
4x4 workgroup still issues the same four TDM descriptors to the same LDS ranges.
No memory address selected by an owner wave changes.

Random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T034653Z_e2e-random/
```

The idle nine-round comparison against the pointer winner was neutral:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `incremental_all_ptr` | 505.465 us | baseline | 1333.71 us | baseline |
| role-specialized state | 505.466 us | -0.00% | 1333.23 us | +0.04% |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T034843Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T035159Z_att/
```

| metric | pointer winner | role-specialized | change |
|---|---:|---:|---:|
| task mean | 34,521.15 | 33,095.71 | -1,425.44 (-4.13%) |
| task median | 32,322.5 | 31,735.5 | -587.0 (-1.82%) |
| full-wave mean | 1,210,179.5 | 1,188,539.5 | **-21,640.0 (-1.79%)** |
| barrier stall | 7,349.09 | 6,633.03 | -716.06 |
| total `DScnt` stall | 4,811.88 | 4,656.85 | -155.03 |
| `TENSORcnt` stall | 2,739.09 | 1,009.21 | **-1,729.88** |
| SALU/control timeline | 1,920.65 | 1,879.76 | **-40.88** |

What matched the hypothesis:

- random/const0 results remained unchanged;
- owner-private state removed unnecessary cross-role scalar updates;
- SALU/control, TENSORcnt, barrier, and full-wave ATT medians all improved.

What did not match the hypothesis:

- the canonical GEMM1 median was unchanged to `0.001 us`;
- the ATT reduction did not translate to wall time, indicating that the skipped
  transition work was hidden behind a different device-wide critical path.

The candidate is not promoted by itself. It is retained only long enough to
derive a version that also preserves and incrementally advances the output
pointer, removing the remaining common output-address rematerialization.
### Experiment 13: role-specialized input state plus incremental output pointer

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_role_state_c_ptr.s
SHA256=3f431696cd2e4f02b67feaa1369a672cf3c0d508db34afb69641475cb077f256
```

This candidate was copied from Experiment 12. In addition to maintaining only
the input base and pointer owned by each wave, it incrementally advances the
per-wave output pointer `s44:s45`. For a `+16` persistent task transition, the
exact output-pointer deltas are:

```text
same expert, N += 16       : 0x1000
expert += 4, N -= 8       : 0x17ff800
delta formula              : 0x1000 + s29 * 0x5ffa00
```

The prefetched entry preserves this pointer and skips both the four input-pointer
rematerializations and the output-offset reconstruction. The original code still
runs on the first/full-setup task. The transformation changes only address
calculation; all resulting addresses, descriptors, memory operations, arithmetic,
and output conversion remain identical.

Random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T035925Z_e2e-random/
```

The idle nine-round comparison against the pointer winner produced:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `incremental_all_ptr` | 504.536 us | baseline | 1330.98 us | baseline |
| role state + output pointer | 504.618 us | **-0.02%** | 1331.51 us | **-0.04%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T040056Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T040357Z_att/
```

| metric | pointer winner | candidate | change |
|---|---:|---:|---:|
| task mean | 33,133.62 | 34,276.35 | **+1,142.74 (+3.45%)** |
| task median | 31,864.5 | 32,277.5 | **+413.0 (+1.30%)** |
| full-wave mean | 1,225,001.0 | 1,202,492.5 | -22,508.5 (-1.84%) |
| barrier stall | 6,752.85 | 6,705.03 | -47.82 |
| total `DScnt` stall | 4,402.03 | 5,354.65 | **+952.62** |
| `TENSORcnt` stall | 970.53 | 1,911.85 | **+941.32** |
| SALU/control timeline | 1,920.94 | 1,873.15 | **-47.79** |

What matched the hypothesis:

- the complete address transformation preserved random and const0 results;
- SALU/control decreased by `47.79 cycles/task`;
- the selected full-wave span moved in the favorable direction.

What did not match the hypothesis:

- steady-task mean and median both regressed;
- DScnt and TENSORcnt together increased by approximately `1,894 cycles/task`;
- the profiler showed no GEMM1 or MoE improvement.

Removing scalar work again advances memory operations into a busier part of the
shared LDS/TDM schedule. The saved scalar instructions are hidden, while the
earlier requests expose more memory completion latency. This candidate is
rejected, is not added to `benchmark_history.sh`, and its ISA is deleted after
recording the result.
### Experiment 14: carry `task_id mod 24` in `s104`

Candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_incremental_all_ptr_remainder_state.s
SHA256=b7e9305fe6b777a6b1b9d6d6c7ec7c4cd0fb4809622d0624768b7cb826f8f513
```

The persistent transition previously recomputed `s28 mod 24` with a reciprocal
multiply, shift, multiply, and subtract before selecting the `N += 16` or
`N -= 8` path. This candidate stores the initial remainder in the otherwise free
`s104`, then advances it with the same two candidates used for `s54`.

The metadata changes `.amdhsa_next_free_sgpr` from `104` to `105`, still below
the documented `106` normal-SGPR limit. Task order, grid mapping, tensor
addresses, arithmetic, and output precision are unchanged.

Assembly/link and random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T041103Z_e2e-random/
```

The idle nine-round comparison against the pointer winner produced:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `incremental_all_ptr` | 505.906 us | baseline | 1333.19 us | baseline |
| carried remainder state | 506.145 us | **-0.05%** | 1332.77 us | +0.03% |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T041321Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T041744Z_att/
```

| metric | pointer winner | remainder state | change |
|---|---:|---:|---:|
| task mean | 33,111.38 | 33,894.79 | **+783.41 (+2.37%)** |
| task median | 31,666.0 | 32,908.5 | **+1,242.5 (+3.92%)** |
| full-wave mean | 1,189,709.5 | 1,201,156.0 | **+11,446.5 (+0.96%)** |
| barrier stall | 6,096.85 | 6,877.50 | **+780.65** |
| total `DScnt` stall | 5,126.09 | 5,031.53 | -94.56 |
| `TENSORcnt` stall | 1,538.62 | 1,922.24 | +383.62 |
| SALU/control timeline | 1,919.24 | 1,914.18 | -5.06 |

What matched the hypothesis:

- the state recurrence produced exactly the same task mapping and output;
- one reciprocal-division sequence disappeared from each transition;
- SALU/control decreased slightly.

What did not match the hypothesis:

- maintaining the extra state saved only about five measured SALU cycles/task;
- barrier and TENSORcnt exposure increased by more than the scalar saving;
- GEMM1 regressed by `0.05%` and full-wave ATT duration regressed by `0.96%`.

This candidate is rejected, is not added to `benchmark_history.sh`, and its ISA
is deleted after recording the result. The extra `s104` lifetime also consumes
the final practical normal-SGPR slot without delivering a critical-path gain.
### Experiment 15: hoist immutable per-wave state out of the persistent loop

Accepted candidate:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist.s
SHA256=283f2b5a1a0d4e82134211b3724816549450ba5abe6806554bfe67f1a9e6f332
```

The prefetched-task entry previously repeated state that is invariant for the
entire lifetime of a persistent wave:

- fixed dimensions and strides in `s12:s21`;
- logical wave ID and wave-mode setup;
- lane-ID normalization;
- fixed scheduling constants `s50:s52` and `s61`.

The candidate was copied from the accepted `incremental_all_ptr` ISA and moved
`.Lmoe_persistent_state_ready` below that one-time setup. Full-setup tasks still
execute the original initialization. Prefetched tasks enter after it and retain
the already-live values. Per-task state such as `s44:s45`, `s54:s55`,
`s58:s60`, and `s68:s71` is still reconstructed exactly as before.

No memory operation, numerical instruction, output layout, descriptor value,
barrier, wait threshold, ABI field, or launch geometry changes. Random MoE e2e
passed with the established accuracy and hashes:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T042556Z_e2e-random/
```

The first nine-round run contained severe transient outliers, including a
`640.242 us` candidate GEMM1 sample and a `3894.82 us` candidate MoE sample. It
is retained as diagnostic evidence but is not used for the promotion decision:

```text
history_runs/heliosr-1b114-a07-3_20260917T042704Z_e2e-const0/
```

Two subsequent idle nine-round comparisons were stable and both favored the
candidate:

| run | case | GEMM1 median | change | MoE median | change |
|---|---|---:|---:|---:|---:|
| 2 | `incremental_all_ptr` | 507.315 us | baseline | 1332.26 us | baseline |
| 2 | static-state hoist | 504.597 us | **+0.54%** | 1330.76 us | **+0.11%** |
| 3 | `incremental_all_ptr` | 507.005 us | baseline | 1354.27 us | noisy baseline |
| 3 | static-state hoist | 504.751 us | **+0.44%** | 1330.95 us | candidate stable |

Stable performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260917T043443Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260917T043920Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T043035Z_att/
```

| metric | pointer winner | static-state hoist | change |
|---|---:|---:|---:|
| task mean | 34,101.15 | 32,894.76 | **-1,206.38 (-3.54%)** |
| task median | 31,935.5 | 31,900.5 | -35.0 (-0.11%) |
| full-wave mean | 1,213,897.5 | 1,186,638.5 | **-27,259.0 (-2.25%)** |
| barrier stall | 7,214.12 | 6,229.24 | -984.88 |
| total `DScnt` stall | 4,559.53 | 4,999.56 | +440.03 |
| `TENSORcnt` stall | 1,386.29 | 1,989.06 | +602.76 |
| SALU/control timeline | 1,919.00 | 1,880.71 | **-38.29** |

What matched the hypothesis:

- all persistent-live values remained valid and correctness was unchanged;
- SALU/control and full-wave duration decreased;
- two clean profiler reruns reproduced a `0.44-0.54%` GEMM1 improvement.

What did not fully match the hypothesis:

- DScnt and TENSORcnt increased, so part of the earlier arrival is paid at the
  subsequent memory waits;
- the first performance run had severe machine-level outliers and could not be
  used as evidence.

This candidate is accepted, retained, and added to `benchmark_history.sh` as
`static_state_hoist`. Its checksum-guarded reproducer is
`build_static_state_hoist_variant.py`.

### Experiment 16: compact next-task descriptor zero initialization

Accepted candidate:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64.s
SHA256=44e6768485b99328561e22c754873bf13ea92cd561ed1c5817fdfbf64e228cae
```

This candidate was copied from the accepted `static_state_hoist` ISA. In each
of the four next-task owner paths it removes the two `s33/s34` zero writes that
are overwritten before use, and replaces eight scalar zero writes to
`s36:s43` with four `s_mov_b64` instructions. The descriptor values presented
to `tensor_load_to_lds`, all waits, barriers, memory operations, arithmetic,
ABI fields, and launch geometry are unchanged.

Random MoE e2e passed with the established accuracy:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Three independent idle nine-round comparisons consistently favored the
candidate:

| run | `static_state_hoist` | descriptor-b64 | GEMM1 change |
|---|---:|---:|---:|
| 1 | 504.023 us | 503.382 us | **+0.13%** |
| 2 | 504.889 us | 504.063 us | **+0.16%** |
| 3, after the a07-3 system-state change | 514.767 us | 513.919 us | **+0.16%** |

The third comparison was part of a four-version interleaved rerun after a07-3
temporarily lost its initialized GPU driver. All versions were remeasured after
the driver and container recovered, and the GPU was idle before and after the
run:

```text
history_runs/heliosr-1b114-a07-3_20260917T053606Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T054056Z_att/
```

| metric | `static_state_hoist` | descriptor-b64 | change |
|---|---:|---:|---:|
| task mean | 34,651.62 | 33,766.88 | **-884.74 (-2.55%)** |
| task median | 32,157.75 | 32,181.00 | +23.25 (+0.07%) |
| full-wave mean | 1,220,334.75 | 1,192,071.25 | **-28,263.50 (-2.32%)** |
| barrier stall | 7,180.02 | 6,785.31 | -394.71 (-5.50%) |
| total `DScnt` stall | 5,804.19 | 4,851.87 | -952.32 (-16.41%) |
| `TENSORcnt` stall | 1,507.13 | 1,754.43 | +247.29 (+16.41%) |
| `s_wait_idle` stall | 60.28 | 46.38 | -13.90 (-23.05%) |
| SALU/control timeline | 1,866.76 | 1,861.84 | -4.93 (-0.26%) |

What matched the hypothesis:

- random and const0 results remained unchanged;
- all three profiler comparisons moved in the favorable direction;
- complete-wave ATT duration decreased by `2.32%`;
- barrier and DScnt exposure both decreased.

What did not fully match the hypothesis:

- the visible SALU/control reduction was only `4.93 cycles/task`, so most of
  the removed scalar instructions were already overlapped;
- `TENSORcnt` exposure increased by `247.29 cycles/task`;
- steady-task median was statistically flat even though full-wave duration and
  profiler medians improved.

The candidate is accepted, retained, and added to `benchmark_history.sh` as
`descriptor_b64`. Its checksum-guarded reproducer is
`build_descriptor_b64_variant.py`.

### Experiment 17: algebraically compact next-task descriptors

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_compact.s
SHA256=2b5e07b0fd2996cd193aca9ccc6589f0df90145db3aa8d61f47768dd6b512c09
```

This candidate was copied from Experiment 16. It constructed the descriptor
pointer with `s_mov_b64`, removed mask/OR operations whose inputs were known
zero, and directly materialized constant descriptor fields. Across the four
owner paths this removed another 60 source instructions, approximately 15
instructions from the active owner path of each persistent transition. The
descriptor bit fields were exhaustively compared against the Experiment 16
form before GPU testing.

Random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T054935Z_e2e-random/
```

The idle nine-round comparison was below the resolution needed for promotion:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `descriptor_b64` | 509.874 us | baseline | 1346.43 us | baseline |
| descriptor compact | 509.702 us | +0.03% | 1344.30 us | +0.16% |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T055409Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T055716Z_att/
```

| metric | `descriptor_b64` | descriptor compact | change |
|---|---:|---:|---:|
| task mean | 33,766.88 | 33,980.32 | +213.44 (+0.63%) |
| task median | 32,181.00 | 32,225.50 | +44.50 (+0.14%) |
| full-wave mean | 1,192,071.25 | 1,217,639.75 | **+25,568.50 (+2.14%)** |
| barrier stall | 6,785.31 | 7,134.99 | +349.68 (+5.15%) |
| total `DScnt` stall | 4,851.87 | 4,127.19 | -724.68 (-14.94%) |
| `TENSORcnt` stall | 1,754.43 | 1,897.99 | +143.56 (+8.18%) |
| `s_wait_idle` stall | 46.38 | 215.96 | +169.57 (+365.60%) |
| SALU/control timeline | 1,861.84 | 1,839.94 | -21.90 (-1.18%) |

What matched the hypothesis:

- the algebraically simplified descriptors preserved random and const0 output;
- SALU/control decreased by `21.90 cycles/task`;
- DScnt exposure decreased by `724.68 cycles/task`.

What did not match the hypothesis:

- the profiler gain was only `0.03%`, below the observed same-machine noise;
- full-wave duration regressed by `2.14%` and task median regressed by `0.14%`;
- barrier, TENSORcnt, and `s_wait_idle` exposure all increased, outweighing the
  visible scalar saving.

This reproduces the recurring result that aggressively shortening scalar setup
can advance memory traffic into a more contended schedule without reducing the
critical path. The candidate is rejected, is not added to
`benchmark_history.sh`, and its ISA and temporary generator are deleted after
recording the result.

### Experiment 18: use B64 zeroing in every remaining descriptor block

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_all_zero.s
SHA256=fbfeb4295ec2aa5681f7324138c47e77d279ac6e0c664a7aa22a61c2981a2498
```

This candidate was copied from the accepted `descriptor_b64` ISA. The source
still contained eight descriptor blocks that cleared `s36:s43` with eight
`s_mov_b32` instructions. They were changed to four `s_mov_b64` instructions
per block. No descriptor field, memory instruction, wait, barrier, arithmetic,
ABI field, or launch geometry changed.

Random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T060618Z_e2e-random/
```

Two idle nine-round comparisons both showed a small regression:

| run | `descriptor_b64` | all-descriptor B64 zero | GEMM1 change | MoE change |
|---|---:|---:|---:|---:|
| 1 | 508.061 us | 508.592 us | **-0.10%** | -0.03% |
| 2 | 512.088 us | 512.645 us | **-0.11%** | -0.09% |

Performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260917T060654Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260917T061245Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T060937Z_att/
```

| metric | `descriptor_b64` | all-descriptor B64 zero | change |
|---|---:|---:|---:|
| task mean | 33,766.88 | 32,806.18 | -960.71 (-2.85%) |
| task median | 32,181.00 | 31,670.75 | -510.25 (-1.59%) |
| full-wave mean | 1,192,071.25 | 1,177,113.00 | -14,958.25 (-1.25%) |
| barrier stall | 6,785.31 | 6,617.06 | -168.25 (-2.48%) |
| total `DScnt` stall | 4,851.87 | 4,428.77 | -423.10 (-8.72%) |
| `TENSORcnt` stall | 1,754.43 | 1,235.43 | -519.00 (-29.58%) |
| `s_wait_idle` stall | 46.38 | 204.78 | **+158.40 (+341.50%)** |
| SALU/control timeline | 1,861.84 | 1,862.31 | +0.47 (+0.03%) |

What matched the hypothesis:

- random and const0 results remained unchanged;
- task and full-wave ATT spans decreased;
- DScnt and TENSORcnt exposure decreased substantially.

What did not match the hypothesis:

- neither of two profiler reruns improved; both regressed by approximately
  `0.10%`;
- the removed B32 instructions were not visible as a SALU/control reduction;
- `s_wait_idle` exposure increased and the favorable ATT sample did not
  reproduce as end-to-end device time.

The profiler result is the promotion criterion. This candidate is rejected, is
not added to `benchmark_history.sh`, and its ISA and temporary generator are
deleted after recording the result.

### Experiment 19: remove overwritten pointer zeros from remaining descriptors

Candidate under evaluation:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_dead_init_all.s
SHA256=32c39c42c401215a9409190efc46caef19f04986715fe7183665f0aa22b8df2b
```

This candidate was copied from the accepted `descriptor_b64` ISA. It removes
the remaining eight pairs of `s_mov_b32 s33, 0` and `s_mov_b32 s34, 0` whose
values are overwritten before either register is read. It does not change the
subsequent pointer writes, descriptor values, memory instructions, wait or
barrier protocol, arithmetic, ABI, or launch geometry.

Random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T061701Z_e2e-random/
```

An unrelated process appeared on a07-3 immediately after the correctness run,
so no performance data was collected until the machine became idle again. The
subsequent idle nine-round comparison was:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `descriptor_b64` | 510.427 us | baseline | 1343.93 us | baseline |
| remove all remaining dead pointer zeros | 512.009 us | **-0.31%** | 1346.68 us | **-0.20%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T062441Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T062803Z_att/
```

| metric | `descriptor_b64` | dead-init removal | change |
|---|---:|---:|---:|
| task mean | 33,766.88 | 33,769.01 | +2.13 (+0.01%) |
| task median | 32,181.00 | 31,935.75 | -245.25 (-0.76%) |
| full-wave mean | 1,192,071.25 | 1,187,256.75 | -4,814.50 (-0.40%) |
| barrier stall | 6,785.31 | 7,312.49 | **+527.18 (+7.77%)** |
| total `DScnt` stall | 4,851.87 | 4,709.59 | -142.28 (-2.93%) |
| `TENSORcnt` stall | 1,754.43 | 1,552.62 | -201.81 (-11.50%) |
| `s_wait_idle` stall | 46.38 | 60.22 | +13.84 (+29.84%) |
| SALU/control timeline | 1,861.84 | 1,860.29 | -1.54 (-0.08%) |

What matched the hypothesis:

- all removed writes were semantically dead and random/const0 results were
  unchanged;
- task median and full-wave ATT duration decreased slightly;
- DScnt and TENSORcnt exposure decreased.

What did not match the hypothesis:

- the removed instructions produced only `1.54 cycles/task` of visible
  SALU/control reduction;
- barrier exposure increased by `527.18 cycles/task`;
- the canonical GEMM1 profiler median regressed by `0.31%`, with a matching
  `0.20%` MoE regression.

The shorter descriptor stream again shifts owner arrival times without
reducing the synchronized critical path. This candidate is rejected, is not
added to `benchmark_history.sh`, and its ISA and temporary generator are
deleted after recording the result.

### Experiment 20: pair persistent-entry state copies with `s_mov_b64`

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_state_copy_b64.s
SHA256=2070bae085528f5fd4571177b83e543de0b7edfa529a0c093cf61415467e4abc
```

This candidate was copied from the accepted `descriptor_b64` ISA and changed
only the four common persistent-entry copies:

```asm
s_mov_b32 s44, s2
s_mov_b32 s45, s3
s_mov_b32 s68, s54
s_mov_b32 s69, s55
```

to two pair copies:

```asm
s_mov_b64 s[44:45], s[2:3]
s_mov_b64 s[68:69], s[54:55]
```

The CDNA5 ISA documents `S_MOV_B64` as a scalar move operating on an SGPR pair.
The source and destination pairs are naturally aligned, and the values and all
subsequent consumers are unchanged.

Random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T063644Z_e2e-random/
```

The idle nine-round comparison was:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `descriptor_b64` | 509.422 us | baseline | 1343.83 us | baseline |
| paired state copies | 510.728 us | **-0.26%** | 1346.16 us | **-0.17%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T063757Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T064210Z_att/
```

| metric | `descriptor_b64` | paired state copies | change |
|---|---:|---:|---:|
| task mean | 33,766.88 | 34,618.29 | **+851.41 (+2.52%)** |
| task median | 32,181.00 | 32,815.50 | **+634.50 (+1.97%)** |
| full-wave mean | 1,192,071.25 | 1,216,722.75 | **+24,651.50 (+2.07%)** |
| barrier stall | 6,785.31 | 7,339.37 | +554.06 (+8.17%) |
| total `DScnt` stall | 4,851.87 | 5,231.25 | +379.38 (+7.82%) |
| `TENSORcnt` stall | 1,754.43 | 1,761.74 | +7.31 (+0.42%) |
| SALU/control timeline | 1,861.84 | 1,864.00 | +2.16 (+0.12%) |

What matched the hypothesis:

- the pair moves assembled and preserved all random and const0 results;
- the static instruction count decreased by two instructions per task.

What did not match the hypothesis:

- the shorter source did not reduce the measured SALU/control timeline;
- barrier and DScnt exposure both increased;
- profiler and full-wave ATT both showed a clear regression.

This candidate is rejected, is not added to `benchmark_history.sh`, and its ISA
and temporary generator are deleted after recording the result.

### Experiment 21: overlap each K-ring TENSORcnt wait with four independent WMMAs

Accepted candidate:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s
SHA256=02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d
```

This candidate was copied from the accepted `descriptor_b64` ISA. At each of
the eight static K-ring boundaries, the source order was:

```asm
s_wait_dscnt 0x8
s_wait_tensorcnt 0x2
s_barrier_signal -1
four independent v_wmma_scale_f32_32x16x128_f4 instructions
s_barrier_wait 0xffff
```

The candidate preserves the `s_wait_dscnt 0x8` dependency, executes the four
WMMAs whose operands are already resident in VGPRs, and only then performs
`s_wait_tensorcnt 0x2` followed by `s_barrier_signal -1`. The workgroup signal
still occurs after both waits, so no wave advertises LDS safety before its own
TDM completion. The change only overlaps TDM progress with independent WMMA
work; it does not relax a counter threshold or move a wait across a dependent
LDS consumer.

This ordering follows the CDNA5 split-barrier model documented in the MI400
Shader Programming Guide: a wave may execute independent work between
`S_BARRIER_SIGNAL` and `S_BARRIER_WAIT`, while barrier completion requires all
member waves to signal. Here the signal is deliberately delayed until the TDM
dependency is satisfied, preserving the original safety condition.

Random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T065155Z_e2e-random/
```

Two idle nine-round comparisons reproduced the gain:

| run | `descriptor_b64` | wait-after-WMMA | GEMM1 change | MoE change |
|---|---:|---:|---:|---:|
| 1 | 511.358 us | 509.334 us | **+0.40%** | +0.11% |
| 2 | 511.898 us | 509.884 us | **+0.39%** | +0.22% |

The first attempted rerun overlapped an unrelated GPU workload and was
discarded before interpretation. The second row above is the clean rerun.

Valid performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260917T065407Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260917T070052Z_e2e-const0/
```

A later named-case integration run had nearly equal independent medians
(`509.855 us` parent and `510.172 us` candidate), but the candidate was faster
in 6 of 9 matched rounds with a `-1.410 us` paired median. Across all three
clean nine-round comparisons, the pooled parent/candidate medians were
`510.962/509.985 us`; the candidate won 20 of 27 matched rounds and the paired
median was `-1.741 us` (`+0.34%`). This is consistent with a small real gain
under substantial machine-level tail noise.

The clean four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T070446Z_att/
```

| metric | `descriptor_b64` | wait-after-WMMA | change |
|---|---:|---:|---:|
| task mean | 33,766.88 | 33,412.84 | -354.04 (-1.05%) |
| task median | 32,181.00 | 31,375.00 | **-806.00 (-2.50%)** |
| full-wave mean | 1,192,071.25 | 1,185,739.00 | -6,332.25 (-0.53%) |
| barrier stall | 6,785.31 | 6,691.85 | -93.46 (-1.38%) |
| total `DScnt` stall | 4,851.87 | 4,963.46 | +111.59 (+2.30%) |
| `TENSORcnt` stall | 1,754.43 | 1,444.09 | **-310.34 (-17.69%)** |
| `s_wait_idle` stall | 46.38 | 72.68 | +26.29 (+56.69%) |
| SALU/control timeline | 1,861.84 | 1,746.50 | -115.34 (-6.20%) |

What matched the hypothesis:

- random and const0 results remained unchanged;
- `TENSORcnt` exposure fell by `310.34 cycles/task`;
- task median, task mean, and full-wave duration all decreased;
- the workgroup barrier did not absorb the saved TDM latency; its measured
  stall also decreased slightly;
- two independent profiler runs reproduced a `0.39-0.40%` GEMM1 improvement.

What did not fully match the hypothesis:

- DScnt exposure increased by `111.59 cycles/task`;
- final `s_wait_idle` exposure increased by `26.29 cycles/task`, though both
  increases are smaller than the TENSORcnt reduction.

This candidate is accepted, retained, and added to `benchmark_history.sh` as
`tensor_wait_after_wmma`. Its checksum-guarded reproducer is
`build_tensor_wait_after_wmma_variant.py`.

### Experiment 22: overlap the workgroup barrier with a fifth WMMA

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma_barrier_wmma5.s
SHA256=c90ea092603a8b3ea7ea320ac72d6b4fdf6f52e8569bee7e1b8511750ea27871
```

This candidate was copied from the accepted `tensor_wait_after_wmma` ISA. At
each of the eight static K-ring boundaries, the first WMMA following
`s_barrier_wait 0xffff` was moved immediately before that wait. Its inputs were
already in VGPRs after `s_wait_dscnt 0x8`, so the workgroup barrier continued to
guard only the following `tensor_load_to_lds` that reuses LDS. The intent was to
cover more of the split-barrier interval with independent matrix work while
issuing the next TDM at the same earliest legal point.

Random MoE e2e passed with the established error and hashes:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T071156Z_e2e-random/
```

One performance attempt overlapped another user's GPU workload and was
discarded. Two valid idle nine-round comparisons did not establish a stable
gain:

| run | `tensor_wait_after_wmma` | barrier-WMMA5 | GEMM1 change | MoE change |
|---|---:|---:|---:|---:|
| 1 | 511.188 us | 510.250 us | +0.18% | +0.03% |
| 2 | 511.907 us | 512.044 us | **-0.03%** | -0.01% |

Valid performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260917T072125Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260917T072901Z_e2e-const0/
```

The clean four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T072348Z_att/
```

| metric | `tensor_wait_after_wmma` | barrier-WMMA5 | change |
|---|---:|---:|---:|
| task mean | 33,412.84 | 33,325.40 | -87.44 (-0.26%) |
| task median | 31,375.00 | 31,845.00 | **+470.00 (+1.50%)** |
| full-wave mean | 1,185,739.00 | 1,196,789.00 | **+11,050.00 (+0.93%)** |
| barrier stall | 6,691.85 | 6,754.57 | +62.72 (+0.94%) |
| total `DScnt` stall | 4,963.46 | 4,388.81 | -574.65 (-11.58%) |
| `TENSORcnt` stall | 1,444.09 | 1,895.43 | **+451.34 (+31.25%)** |
| SALU/control timeline | 1,746.50 | 1,746.90 | +0.40 (+0.02%) |
| WMMA timeline | 9,381.35 | 9,533.69 | +152.34 (+1.62%) |

What matched the hypothesis:

- random and const0 results remained unchanged;
- DScnt exposure decreased because the moved WMMA changed the downstream LDS
  issue spacing;
- one profiler run showed a small positive result.

What did not match the hypothesis:

- the second clean profiler run was flat to slightly slower;
- task median and full-wave ATT duration regressed;
- moving the fifth WMMA exposed another `451.34 cycles/task` at TENSORcnt and
  increased the measured WMMA timeline.

The fifth WMMA is therefore better left after the workgroup barrier. This
candidate is rejected, is not added to `benchmark_history.sh`, and its ISA and
temporary generator are deleted after recording the result.

### Experiment 23: place the K-ring TENSORcnt wait after three WMMAs

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma3.s
SHA256=c80c8516b75f92e70d356b129df5d56d97a24fa60b3926bd1ccbb7c3afbb5898
```

This candidate was copied from the accepted `tensor_wait_after_wmma` ISA. It
moved each K-ring `s_wait_tensorcnt 0x2` and the following
`s_barrier_signal -1` one WMMA earlier, leaving three independent WMMAs before
the wait and one after the signal. The experiment tests whether splitting the
available independent work between TDM overlap and split-barrier overlap is
better than placing all four WMMAs before the TDM wait.

Random MoE e2e passed with unchanged accuracy and hashes:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T073330Z_e2e-random/
```

The idle nine-round comparison was statistically flat:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `tensor_wait_after_wmma` | 513.068 us | baseline | 1348.70 us | baseline |
| wait after three WMMAs | 513.041 us | +0.01% | 1348.69 us | +0.00% |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T073418Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T073726Z_att/
```

| metric | wait after four WMMAs | wait after three WMMAs | change |
|---|---:|---:|---:|
| task mean | 33,412.84 | 33,398.12 | -14.72 (-0.04%) |
| task median | 31,375.00 | 31,755.75 | **+380.75 (+1.21%)** |
| full-wave mean | 1,185,739.00 | 1,200,974.00 | **+15,235.00 (+1.28%)** |
| barrier stall | 6,691.85 | 6,518.34 | -173.52 (-2.59%) |
| total `DScnt` stall | 4,963.46 | 5,088.44 | +124.99 (+2.52%) |
| `TENSORcnt` stall | 1,444.09 | 1,574.40 | +130.31 (+9.02%) |
| SALU/control timeline | 1,746.50 | 1,763.76 | +17.26 (+0.99%) |

What matched the hypothesis:

- random and const0 results remained unchanged;
- moving one WMMA after the signal reduced measured barrier exposure.

What did not match the hypothesis:

- the shorter TDM-overlap window increased TENSORcnt and DScnt exposure;
- full-wave duration regressed by `1.28%`;
- the profiler result was indistinguishable from zero.

The four-WMMA placement is the better point in this schedule. The three-WMMA
candidate is rejected, is not added to `benchmark_history.sh`, and its ISA and
temporary generator are deleted after recording the result.

### Experiment 24: place the K-ring TENSORcnt wait after five WMMAs

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma5.s
SHA256=922181178e5304af78d6e19f4d80b84ab21b4f3bba03264530069d34ed2e9004
```

This candidate was copied from the accepted `tensor_wait_after_wmma` ISA. It
moved the first WMMA after `s_barrier_wait 0xffff` ahead of
`s_wait_tensorcnt 0x2`, giving TDM completion five independent WMMAs of overlap.
The wait still precedes `s_barrier_signal -1`, and the workgroup barrier still
precedes the following `tensor_load_to_lds`.

Random MoE e2e passed with unchanged accuracy and hashes:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T074652Z_e2e-random/
```

One performance attempt overlapped another user's GPU workload and was
discarded. The valid idle nine-round comparison was:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `tensor_wait_after_wmma` | 508.498 us | baseline | 1345.23 us | baseline |
| wait after five WMMAs | 510.075 us | **-0.31%** | 1346.12 us | **-0.07%** |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T075111Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T075411Z_att/
```

| metric | wait after four WMMAs | wait after five WMMAs | change |
|---|---:|---:|---:|
| task mean | 33,412.84 | 32,930.37 | -482.47 (-1.44%) |
| task median | 31,375.00 | 31,800.50 | **+425.50 (+1.36%)** |
| full-wave mean | 1,185,739.00 | 1,177,264.50 | -8,474.50 (-0.71%) |
| barrier stall | 6,691.85 | 6,921.91 | +230.06 (+3.44%) |
| total `DScnt` stall | 4,963.46 | 4,334.87 | -628.59 (-12.66%) |
| `TENSORcnt` stall | 1,444.09 | 1,417.87 | -26.22 (-1.82%) |
| SALU/control timeline | 1,746.50 | 1,746.87 | +0.37 (+0.02%) |

The additional WMMA reduced mean DScnt exposure, but it increased the robust
task median and barrier exposure. The profiler also regressed. The moved WMMA
has a shorter reuse distance from an earlier update of the same accumulator,
so its scoreboard delay is not useful extra TDM overlap. This candidate is
rejected, is not added to `benchmark_history.sh`, and its ISA and temporary
generator are deleted after recording the result.

### Experiment 25: prepare the next TDM offset before TENSORcnt wait

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma_desc_ready.s
SHA256=97a3a34eea0b3319fddfa8fa51c21a09740410ebcd625ab71ec7ca15152fc8e6
```

This candidate was copied from the accepted `tensor_wait_after_wmma` ISA. It
moved the independent per-stage `s_mov_b32 s33, s9x` descriptor offset before
`s_wait_tensorcnt 0x2`. All four WMMAs remain before the tensor wait, and the
wait and signal remain adjacent and in the original dependency order.

Random MoE e2e passed with unchanged accuracy and hashes:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T075944Z_e2e-random/
```

Two idle nine-round comparisons were inconsistent and did not establish a
gain:

| run | `tensor_wait_after_wmma` | descriptor-before-wait | GEMM1 change | MoE change |
|---|---:|---:|---:|---:|
| 1 | 510.456 us | 508.439 us | +0.40% | +0.13% |
| 2 | 509.944 us | 510.077 us | **-0.03%** | -0.07% |

Performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260917T080034Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260917T080434Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T080701Z_att/
```

| metric | `tensor_wait_after_wmma` | descriptor-before-wait | change |
|---|---:|---:|---:|
| task mean | 33,412.84 | 34,273.26 | **+860.43 (+2.57%)** |
| task median | 31,375.00 | 32,264.25 | **+889.25 (+2.83%)** |
| full-wave mean | 1,185,739.00 | 1,206,716.75 | **+20,977.75 (+1.77%)** |
| barrier stall | 6,691.85 | 6,955.25 | +263.40 (+3.94%) |
| total `DScnt` stall | 4,963.46 | 5,189.12 | +225.66 (+4.55%) |
| `TENSORcnt` stall | 1,444.09 | 1,933.93 | **+489.84 (+33.92%)** |
| SALU/control timeline | 1,746.50 | 1,722.60 | -23.90 (-1.37%) |

Although the independent move reduced the SALU/control timeline, its new issue
position consistently changed the TDM/barrier arrival pattern in the wrong
direction. The second profiler run and the ATT both reject it. The candidate is
not added to `benchmark_history.sh`, and its ISA and temporary generator are
deleted after recording the result.

### Experiment 26: precompute the balanced B/ScaleB column mask

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma_static_column_mask.s
SHA256=2559fde554ad7351d3406ada7f744b4e0d7f3ccd24ea5de036f5cf2877a4abac
```

This candidate was copied from the accepted `tensor_wait_after_wmma` ISA. The
pinned production FlyDSL source uses `column_mask = 0x1111` for the fixed
balanced E96/T16384 configuration. The candidate computes
`0x1111 << local_n` once into the otherwise unused `s104`, changes
`.amdhsa_next_free_sgpr` from 104 to 105, and replaces six dynamic B/ScaleB
column-mask loops with `s_mov_b32 s53, s104`. The resulting descriptor bits are
identical for the supported workload; all memory instructions and numerical
operations remain unchanged.

Random MoE e2e passed:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T081832Z_e2e-random/
```

The idle nine-round comparison was:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `tensor_wait_after_wmma` | 507.791 us | baseline | 1342.31 us | baseline |
| static column mask | 509.052 us | **-0.25%** | 1342.55 us | -0.02% |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T082017Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T082303Z_att/
```

| metric | `tensor_wait_after_wmma` | static column mask | change |
|---|---:|---:|---:|
| task mean | 33,412.84 | 32,254.43 | -1,158.41 (-3.47%) |
| task median | 31,375.00 | 31,525.50 | +150.50 (+0.48%) |
| full-wave mean | 1,185,739.00 | 1,178,601.25 | -7,137.75 (-0.60%) |
| barrier stall | 6,691.85 | 6,225.22 | -466.63 (-6.97%) |
| total `DScnt` stall | 4,963.46 | 4,605.65 | -357.81 (-7.21%) |
| `TENSORcnt` stall | 1,444.09 | 1,456.12 | +12.03 (+0.83%) |
| SALU/control timeline | 1,746.50 | 1,700.15 | -46.35 (-2.65%) |

The implementation removed the intended loops and improved several mean ATT
metrics, but the robust task median increased and the canonical profiler
regressed by `0.25%`. The shortened B/ScaleB paths changed owner arrival balance
without reducing the synchronized critical path. This all-owner mask reuse is
rejected and removed; a B-only variant is the remaining targeted follow-up.

### Experiment 27: precompute the balanced B-owner column mask only

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma_static_b_column_mask.s
SHA256=71971859924af5e62ebc83bca33a55e4642b12799f12ca23f94b3bd8dadd1267
```

This candidate was copied from the accepted `tensor_wait_after_wmma` ISA. It
kept the ScaleB builders unchanged and reused the persistent balanced mask only
in the three B-owner descriptor builders. The goal was to shorten the measured
late B owner without advancing ScaleB by the same amount.

Random MoE e2e passed with unchanged accuracy and hashes:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T083038Z_e2e-random/
```

Two idle nine-round comparisons did not reproduce one another:

| run | `tensor_wait_after_wmma` | B-only static mask | GEMM1 change | MoE change |
|---|---:|---:|---:|---:|
| 1 | 511.276 us | 509.225 us | +0.40% | +0.20% |
| 2 | 511.034 us | 512.040 us | **-0.20%** | +0.07% |

Performance result directories:

```text
history_runs/heliosr-1b114-a07-3_20260917T083126Z_e2e-const0/
history_runs/heliosr-1b114-a07-3_20260917T083345Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T083654Z_att/
```

| metric | `tensor_wait_after_wmma` | B-only static mask | change |
|---|---:|---:|---:|
| task mean | 33,412.84 | 32,768.65 | -644.19 (-1.93%) |
| task median | 31,375.00 | 31,635.50 | +260.50 (+0.83%) |
| full-wave mean | 1,185,739.00 | 1,180,592.75 | -5,146.25 (-0.43%) |
| barrier stall | 6,691.85 | 6,668.99 | -22.87 (-0.34%) |
| total `DScnt` stall | 4,963.46 | 4,907.04 | -56.41 (-1.14%) |
| `TENSORcnt` stall | 1,444.09 | 1,271.40 | -172.69 (-11.96%) |
| SALU/control timeline | 1,746.50 | 1,728.28 | -18.22 (-1.04%) |

The targeted B path became shorter in the trace, but the cross-owner robust
task median increased and profiler results changed sign between runs. This is
another owner-arrival redistribution rather than a repeatable kernel-level
gain. The candidate is rejected, is not added to `benchmark_history.sh`, and
its ISA and temporary generator are deleted after recording the result.

### Experiment 28: keep wait-after-WMMA only for the A/ScaleA family

Rejected candidate before cleanup:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_a_family.s
SHA256=0a1465b6c522adbc675132da34ea4cbe46a03799f810a6c67caa80c9bdc64d82
```

This candidate was copied from the accepted `tensor_wait_after_wmma` ISA. The
first four static K-ring boundaries, shared by A/ScaleA, kept the accepted
wait-after-four-WMMA order. The four B/ScaleB boundaries restored the source
order with `s_wait_tensorcnt 0x2` before those WMMAs. This directly tested the
ATT observation that B was the latest owner while ScaleB had timing headroom.

Random MoE e2e passed with unchanged accuracy and hashes:

```text
logits_diff=3.39799e-06
rel_l2=2.60689e-03
pass=True
MoE output hash128=1556fc617347e2dabc9cff19dbfd822b
ref output hash128=1a5d22911ba167160b4f2c12092a5193
```

Random correctness directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T084113Z_e2e-random/
```

The idle nine-round comparison was:

| case | GEMM1 median | change | MoE median | change |
|---|---:|---:|---:|---:|
| `tensor_wait_after_wmma` | 508.829 us | baseline | 1343.57 us | baseline |
| A/ScaleA wait-after-WMMA only | 509.767 us | **-0.18%** | 1343.91 us | -0.03% |

Performance result directory:

```text
history_runs/heliosr-1b114-a07-3_20260917T084158Z_e2e-const0/
```

The four-SIMD ATT capture is:

```text
history_runs/heliosr-1b114-a07-3_20260917T124840Z_att/
```

| metric | `tensor_wait_after_wmma` | A-family-only schedule | change |
|---|---:|---:|---:|
| task mean | 33,412.84 | 32,725.19 | -687.65 (-2.06%) |
| task median | 31,375.00 | 31,495.00 | +120.00 (+0.38%) |
| full-wave mean | 1,185,739.00 | 1,180,455.50 | -5,283.50 (-0.45%) |
| barrier stall | 6,691.85 | 6,242.28 | -449.57 (-6.72%) |
| total `DScnt` stall | 4,963.46 | 4,384.47 | -578.99 (-11.67%) |
| `TENSORcnt` stall | 1,444.09 | 1,856.51 | **+412.43 (+28.56%)** |
| SALU/control timeline | 1,746.50 | 1,778.66 | +32.16 (+1.84%) |

The B/ScaleB source order exchanged DScnt and barrier exposure for substantially
more TENSORcnt latency. The profiler and robust task median both reject it. The
candidate is not added to `benchmark_history.sh`, and its ISA and temporary
generator are deleted after recording the result.

## Current stopping point

The retained endpoint is:

```text
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_static_state_hoist_descriptor_b64_tensor_wait_after_wmma.s
SHA256=02320b67ae67a8349f7942a31f89e3f2f3de6efe130e937d04d1d4bdb9273b2d
benchmark case=tensor_wait_after_wmma
```

It preserves the production ABI, numerical instruction sequence, output
layout, persistent overlap protocol, TDM thresholds, LDS allocation, and
cluster geometry. Relative to `descriptor_b64`, two clean nine-round runs
reproduced a `0.39-0.40%` GEMM1 improvement, and random/const0 MoE e2e passed.

Further incremental edits no longer have a convincing expected benefit:

1. the safe tensor-wait placement sweep found four independent WMMAs to be the
   only repeatable point; three and five WMMAs were flat or slower;
2. moving more work across the workgroup barrier increases TENSORcnt or WMMA
   scoreboard exposure;
3. descriptor and scalar reductions are largely hidden and repeatedly convert
   into owner-arrival imbalance at the cluster barrier;
4. B/ScaleB fixed-mask and owner-specific schedules improved selected means but
   did not improve the robust profiler median;
5. the remaining dominant stalls are dependency-bearing `s_wait_dscnt` sites
   and the K-ring cluster barrier. Earlier experiments showed that moving the
   final DS wait across a consumer changes random results, while another full
   input ring does not fit alongside the 320 KiB LDS allocation;
6. the kernel already consumes 1024 VGPRs, so deeper operand buffering requires
   a structural register/LDS redesign rather than another local instruction
   move.

Optimization is paused here. A credible next step would require redesigning the
K-ring ownership and operand staging together, with a new LDS/register layout
and a fresh barrier proof. That is a separate structural kernel version, not a
safe incremental change to this correctness-qualified endpoint.
