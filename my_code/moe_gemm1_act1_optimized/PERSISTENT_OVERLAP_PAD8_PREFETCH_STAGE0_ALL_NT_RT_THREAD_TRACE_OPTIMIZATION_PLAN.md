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
| cross-owner median | — | — | 27,951 | — | — | — |

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
| scratch | 0 | — | no spill traffic |
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
6. capture four-SIMD ATT only for a promoted scheduling change and compare the
   K-ring and DScnt medians against this report.

The first implementation should be Phase 1, B/ScaleB next-ring scalar-state
precompute. It directly attacks the largest measured single stall, fits in the
two available SGPRs, and does not require changing LDS or TDM correctness
protocols.
