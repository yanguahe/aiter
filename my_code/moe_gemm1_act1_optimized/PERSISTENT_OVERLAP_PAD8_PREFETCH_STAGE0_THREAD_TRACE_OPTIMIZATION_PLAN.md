# `persistent_overlap_pad8_prefetch_stage0` thread-trace analysis and optimization plan

## Scope and result

This report analyzes the pure-assembly kernel:

```text
my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage0.s
```

The const0 trace was captured three times on `d01-3` with an idle GPU. The
median steady-task cost is `27,298.1 GFXCLK cycles`. Explicit wait stall accounts for
`7,109.0 cycles/task`, or `26.04%` of the task. The largest remaining bottleneck
is cluster/workgroup synchronization, followed by LDS-read completion and input
TDM completion.

The next implementation should first overlap next-task stage-1 setup/transfer
with the current task tail, then move next-ring descriptor work before the six
dominant cluster waits, and finally reschedule the LDS-read bursts around the
largest `s_wait_dscnt 0x8` sites. A new occupancy level is not available without
a much larger tile redesign because the current kernel already consumes the
full per-WGP LDS capacity and the full per-SIMD VGPR capacity.

## Capture and analysis method

The three ATT runs are:

```text
my_code/moe_gemm1_act1_optimized/history_runs/
  heliosr-1b114-d01-3_20260913T091935Z_att/
  heliosr-1b114-d01-3_20260913T092015Z_att/
  heliosr-1b114-d01-3_20260913T092053Z_att/
```

Each run captured one `SIMD3-select` wave from each of four SEs. The analysis
uses the rule-provided parser copied into this experiment directory:

```text
trace_segment_cycles.py
SHA256=6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

`analyze_stage0_thread_trace.py` imports that parser and uses consecutive
dynamic occurrences of this sequence as task boundaries:

```asm
s_add_co_u32 s28, s28, 16
s_cmp_lt_u32 s28, 0x240
s_cbranch_scc0 5
```

This sequence runs once after every completed persistent task, independent of
whether the next task enters through the full-setup or prefetched path. Each
capture has `72` complete boundary-to-boundary intervals. The first interval of
each of the four traced waves is removed, leaving `68` steady intervals per
capture.

The analysis command is:

```bash
python3 my_code/moe_gemm1_act1_optimized/analyze_stage0_thread_trace.py \
  20260913T091935Z \
  20260913T092015Z \
  20260913T092053Z \
  --output \
  my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage0_thread_trace_metrics.json
```

The resulting compact machine-readable data is stored in
`persistent_overlap_pad8_prefetch_stage0_thread_trace_metrics.json`.

`trace_segment_cycles.py` measures an interval as `end_ts - start_ts`, so the
task cycle includes wait stalls. For the mutually exclusive issue-timeline
breakdown, each dynamic instruction owns the timestamp gap to the following
instruction. A non-wait instruction's reported latency can overlap later
instructions; therefore long non-wait latency is used as a diagnostic and is
not summed as serialized kernel time.

## Capture stability

| capture | steady tasks | mean cycles/task | p50 | p90 | dispatch max cycles |
|---|---:|---:|---:|---:|---:|
| `20260913T091935Z` | 68 | 26,023.8 | 25,804.0 | 27,435.2 | 1,085,831 |
| `20260913T092015Z` | 68 | 27,298.1 | 25,985.5 | 34,780.5 | 1,078,342 |
| `20260913T092053Z` | 68 | 27,946.2 | 26,604.0 | 34,727.7 | 1,087,027 |

The dispatch spans are within `0.81%` of one another. Some individual tasks
have long synchronization tails, which raises the second and third capture
means and p90 values. The remainder of this report takes the median across the
three per-capture means for every metric.

For historical context, the earlier `persistent_overlap_pad8` traces reported
`28,664.3 cycles/task` and `1,113,445` dispatch cycles. The stage-0 prefetch
version is lower by `4.77%` per steady task and `2.48%` at dispatch level. These
are separate three-capture sets, so they establish direction rather than a
cycle-exact paired comparison.

## Phase breakdown

| phase | cycles/task | share |
|---|---:|---:|
| boundary + next-task setup + K hotloop | 22,585.9 | 82.74% |
| output descriptor/address setup + next-stage0 prefetch | 506.2 | 1.85% |
| SiLU banks 0-1 + first output launch | 1,787.5 | 6.55% |
| SiLU banks 2-3 | 1,846.0 | 6.76% |
| second output setup/launch + task boundary | 147.0 | 0.54% |

The K loop and its synchronization dominate. The two SiLU/output phases are
about `13.31%` together. Further epilogue work can still help, but it cannot
match the immediate opportunity in the K-loop waits.

## Wait-stall breakdown

| instruction group | stall cycles/task | task share |
|---|---:|---:|
| `s_barrier_wait` | 4,034.7 | 14.78% |
| `s_wait_dscnt` | 1,670.0 | 6.12% |
| `s_wait_tensorcnt` | 1,390.1 | 5.09% |
| `s_wait_idle` | 14.2 | 0.05% |
| total | 7,109.0 | 26.04% |

The exact wait-immediate totals are:

| instruction | stall cycles/task | task share | dynamic count/task |
|---|---:|---:|---:|
| `s_barrier_wait 0xfffd` | 2,417.9 | 8.86% | 9 |
| `s_barrier_wait 0xffff` | 1,361.9 | 4.99% | 37 |
| `s_wait_dscnt 0x8` | 1,219.0 | 4.47% | 84 |
| `s_wait_tensorcnt 0x2` | 955.1 | 3.50% | 30 |
| `s_wait_tensorcnt 0x1` | 439.8 | 1.61% | 1 |
| `s_wait_dscnt 0x0` | 148.9 | 0.55% | 3 |
| `s_wait_dscnt 0x14` | 146.5 | 0.54% | 56 |
| `s_wait_dscnt 0x4` | 21.8 | 0.08% | 28 |
| `s_wait_idle` | 14.2 | 0.05% | 1 |

The prior output-pad8 trace had `3,457.1 cycles/task` in
`s_wait_tensorcnt`; stage-0 prefetch reduces that to `1,390.1`, a directional
reduction of about `59.8%`. The old full-drain boundary wait is absent from the
steady path. Its replacement at ATT PC `0x371c`,
`s_wait_tensorcnt 0x1`, costs `439.8 cycles/task`.

The optimization has therefore worked as intended. The bottleneck has shifted:
barrier stall now exceeds TDM wait stall by almost `3x`.

## Dominant static wait sites

| ATT PC | instruction | stall cycles/task | task share | count/task | interpretation |
|---|---|---:|---:|---:|---|
| `0xad14` | `s_barrier_wait 0xfffd` | 1,972.2 | 7.23% | 6 | dominant K-ring cluster barrier; peers reach the reuse point at different times |
| `0xa790` | `s_wait_dscnt 0x8` | 546.0 | 2.00% | 7 | repeated hotloop operand LDS-read drain |
| `0x376c` | `s_barrier_wait 0xffff` | 521.0 | 1.91% | 1 | first prefetched-task LDS-ready workgroup barrier |
| `0x371c` | `s_wait_tensorcnt 0x1` | 439.8 | 1.61% | 1 | waits for enough previous output TDM progress before issuing the next input stage |
| `0x93a8` | `s_wait_tensorcnt 0x2` | 246.6 | 0.90% | 7 | repeated input TDM readiness point |
| `0xa950` | `s_wait_tensorcnt 0x2` | 187.6 | 0.69% | 7 | repeated input TDM readiness point |
| `0x9920` | `s_wait_dscnt 0x8` | 187.4 | 0.69% | 7 | second large repeated LDS-read drain |
| `0xe2d8` | `s_barrier_wait 0xfffd` | 180.6 | 0.66% | 1 | persistent task-boundary cluster barrier |
| `0x9ae0` | `s_wait_tensorcnt 0x2` | 166.5 | 0.61% | 7 | repeated input TDM readiness point |
| `0xa9a0` | `s_barrier_wait 0xffff` | 158.3 | 0.58% | 7 | repeated workgroup synchronization after operand readiness |

The `0xad14` site alone accounts for `81.6%` of all
`s_barrier_wait 0xfffd` stall. Its signal occurs hundreds of static
instructions earlier and substantial WMMA work already lies between signal and
wait. This means the remaining delay is primarily peer progress skew. Moving
the existing signal a few instructions earlier is unlikely to solve it.

The ideal bound from deleting the `0xad14` stall alone is `1,972 cycles/task`,
or `7.23%` of current cycles. Since the reduced task would take fewer cycles,
the corresponding ideal throughput speedup is `7.79%`. This is an upper bound;
the barrier itself is required for safe multicast LDS reuse.

## Issue-timeline breakdown

| category | cycles/task | task share |
|---|---:|---:|
| WMMA issue | 9,618.3 | 35.23% |
| barrier wait | 4,107.4 | 15.05% |
| LDS read issue | 3,465.3 | 12.69% |
| DScnt wait | 2,139.5 | 7.84% |
| SALU/control issue | 1,959.9 | 7.18% |
| packed SiLU VALU issue | 1,751.0 | 6.41% |
| TENSORcnt wait | 1,421.1 | 5.21% |
| other VALU issue | 1,271.0 | 4.66% |
| EXP/RCP issue | 1,000.0 | 3.66% |
| explicit NOP | 237.8 | 0.87% |
| LDS write issue | 182.0 | 0.67% |
| TDM issue | 102.8 | 0.38% |

WMMA is necessary work. The actionable exposed time is barrier, DScnt, and
TENSORcnt wait. These three issue-timeline categories total about `28.1%`.

## Other long-latency instructions

Long non-wait events are almost entirely `ds_load_b128` plus one
`ds_load_b32`. The largest representative sites are:

| ATT PC | instruction | issue-gap cycles/task | task share | median per-capture max latency | max observed latency |
|---|---|---:|---:|---:|---:|
| `0x9608` | `ds_load_b128 v[28:31], v73 offset:2560` | 36.9 | 0.135% | 493 | 9,107 |
| `0x9594` | `ds_load_b32 v82, v80 offset:2048` | 29.9 | 0.110% | 703 | 764 |
| `0xac40` | `ds_load_b128 v[68:71], v72 offset:7680` | 28.4 | 0.104% | 334 | 570 |
| `0xabb0` | `ds_load_b128 v[28:31], v72 offset:2560` | 28.1 | 0.103% | 402 | 490 |
| `0xabf0` | `ds_load_b128 v[44:47], v72 offset:4608` | 27.6 | 0.101% | 521 | 772 |

The isolated latency outliers are large, but their mutually exclusive issue-gap
shares are each below `0.14%`. They are not independent serialized costs. Their
practical impact is already exposed at the later `s_wait_dscnt` sites, so the
correct optimization target is the DS-load schedule and bank mapping rather
than deleting an individual load.

## Resource usage and occupancy consequence

The compiled code object reports:

```text
cluster_dims:                (4, 4, 1)
max_flat_workgroup_size:     128
wavefront_size:              32
group_segment_fixed_size:    327680 bytes = 320 KiB
private_segment_fixed_size:  0
vgpr_count:                  1024
sgpr_count:                  106
```

The assembly directive reports `.amdhsa_next_free_sgpr 104`, so numbered shader
state reaches `s103`; `s104:s105` are the only unused normal SGPR numbers. The
code-object `sgpr_count=106` includes the architecture/runtime allocation. The
kernel has no private scratch allocation.

The hardware constraints are decisive:

- MI400 Shader Programming Guide §2.2 and §3.3.4: one workgroup may allocate up
  to `320 KiB` LDS, and LDS is shared by all four SIMD32s in a WGP. This kernel
  allocates all `320 KiB`, so a WGP cannot host a second workgroup.
- MI400 Shader Programming Guide §3.3.1.1: a wave has `106` normal SGPRs. SGPR
  capacity is effectively full at metadata level, although dead ranges can be
  reused inside the hand-written schedule.
- MI400 Shader Programming Guide §3.3.2.1/§3.3.2.3: MI450 supports at most
  `1024` VGPRs per wave and has `1024` physical VGPRs per SIMD. This kernel uses
  all `1024`, so each SIMD can host only one of the workgroup's four wave32s.
- MI455X whitepaper, page 10: a WGP has four SIMD32s, `320 KiB` usable LDS, a
  separate TDM unit, and a `64 KiB` instruction cache.

The resulting execution shape is one 128-thread workgroup per WGP and one wave
per SIMD. There is no second resident wave that can hide a stalled wave's
barrier, TDM, or LDS latency. Reducing only VGPR or only LDS usage is insufficient
to increase occupancy; both would have to fall to roughly half their current
footprint for two such workgroups to coexist.

## Hardware ordering constraints

The plan below preserves these documented requirements:

- MI400 Shader Programming Guide §4.10.1: TDM operations from one wave complete
  in issue order; TDM can execute in parallel with shader instructions.
- MI400 Shader Programming Guide §4.3.6.6: the cluster barrier counts
  workgroups; one wave per workgroup signals it, and all waves in the cluster
  must wait before reusing peer LDS.
- CDNA5 ISA §5.7.1: `DScnt` tracks LDS completion and `TENSORcnt` tracks TDM
  completion.
- CDNA5 ISA §5.7.2 and §7.12.1: WMMA/TRANS operations can co-execute with
  independent instructions, but their documented RAW/WAR/WAW spacing must be
  retained.

## Implementation plan

### Phase 1: delayed stage-1 cross-task prefetch

This is the first candidate because it targets the measured `0x371c`
`s_wait_tensorcnt 0x1` and the following `0x376c` workgroup barrier without
adding LDS or VGPR state.

Keep the existing early stage-0 order and change the task tail from:

```text
I0(next), O0(current), O1(current)
  -> next task waits
  -> I1(next)
```

to:

```text
I0(next), O0(current), O1(current)
  -> s_wait_tensorcnt 0x2 if required to cap the queue
  -> I1(next)
  -> persistent boundary
```

Implementation details:

1. Reuse the existing next-task state already computed for stage 0.
2. After `O1` has consumed the output descriptor SGPRs, rebuild the stage-1
   A/B/ScaleA/ScaleB descriptor in the same `s32:s43` range.
3. Issue stage 1 only after the final hotloop cluster synchronization, when no
   peer can still read the current task's stage-1 LDS. Stage 1 does not overlap
   the output staging regions.
4. Treat `s101` as a bit mask: bit 0 means stage 0 is prefetched and bit 1 means
   stage 1 is prefetched. This needs no new SGPR.
5. Preserve per-wave TDM order. A pre-issue `s_wait_tensorcnt 0x2` keeps the
   documented in-flight window at three or fewer operations.
6. At the next task, skip the stage-1 issue and place its readiness wait as late
   as possible, after stage-0 WMMA and descriptor work.

Ideal removable time is the measured `439.8 cycles/task` at `0x371c` plus some
of the `521.0 cycles/task` at `0x376c`. A realistic first target is
`300-700 cycles/task` (`1.1-2.6%`). This differs from the rejected stage-0+1
variant, which used `I0,I1,O0,O1` and delayed both output transfers. The new
order leaves `O0/O1` ahead of `I1`.

### Phase 2: precompute the next K-ring descriptor before `0xad14`

The barrier at `0xad14` is mandatory, but scalar work performed after it can be
moved before it:

1. For each of the six dynamic ring transitions, identify the next descriptor's
   pointer increments, bounds, LDS base, multicast mask, and branch selector.
2. Move operations that do not touch LDS and do not modify live WMMA operands
   above `s_barrier_wait 0xfffd`.
3. Reuse dead descriptor temporaries in `s24:s43` and the existing dead ranges;
   do not raise `.amdhsa_next_free_sgpr`.
4. Keep `s_barrier_signal -3`, `s_barrier_wait 0xfffd`, and TDM issue order
   unchanged. The first LDS-writing TDM remains after the wait.
5. After the wait, issue the prepared TDM immediately and enter the existing
   DS/WMMA schedule.

The `0xad14` wait contributes `1,972.2 cycles/task`, but most of that is peer
arrival skew. Descriptor hoisting can hide only the independent post-wait SALU
work. A reasonable target is `100-200 cycles` per ring transition, or
`600-1,200 cycles/task` (`2.2-4.4%`) if enough setup exists and code motion does
not lengthen another path.

### Phase 3: reschedule the two dominant LDS-read drains

Focus first on `0xa790` and `0x9920`:

1. Split each large `ds_load_b128` burst into an early subset and a late subset.
2. Consume already-ready operands with independent WMMA/SALU instructions
   before issuing the remaining loads.
3. Move `s_wait_dscnt 0x8` to the last legal point before the first dependent
   WMMA, preserving all CDNA5 WMMA hazard spacing.
4. Test thresholds `0x14`, `0x10`, `0xc`, and `0x8` only where the exact number
   of live DS operations is statically known.
5. Compare ATT `DScnt wait`, not only profiler microseconds; reject a change that
   moves the stall to another DS wait without reducing total task cycles.

The total `s_wait_dscnt 0x8` budget is `1,219.0 cycles/task`. A practical target
is a `20-35%` reduction, or roughly `240-430 cycles/task` (`0.9-1.6%`).

### Phase 4: selective input-LDS skew within the existing 48 KiB hole

If scheduling does not reduce `DScnt`, use the TDM padding capability described
in MI400 Shader Programming Guide §4.10 to alter the hotloop input layout:

1. Generate A-only, B-only, and A+B skew variants.
2. Add a small per-row or per-slab LDS pad and update every matching DS address.
3. Keep the four-stage ring and the output regions unchanged.
4. Limit added space to the existing `[0x24000,0x30000)` 48 KiB hole, retaining
   `group_segment_fixed_size=327680`.
5. Use ATT to require reductions at `0xa790`, `0x9920`, and total
   `s_wait_dscnt`; individual `ds_load` max-latency outliers are not sufficient
   evidence.

This phase has a lower confidence than pure scheduling because the current
layout and TDM descriptors must be changed together.

### Phase 5: targeted ownership balancing only if barriers remain dominant

The WPT1 layout assigns about `32 KiB`, `32 KiB`, `2 KiB`, and `2 KiB` of TDM
traffic to the four owner waves for each K256 stage. This is a plausible source
of the local and cluster arrival skew. A prior full WPT2/reference-schedule port
regressed, so it should not be reused wholesale.

If phases 1-4 plateau, build a minimal ownership-only experiment that preserves
the current hand-written WMMA/DS schedule:

```text
wave 0/1: half A payload + half ScaleA
wave 2/3: half B payload + half ScaleB
```

Each wave would move about `17 KiB` per stage. Descriptor SGPRs must be reused
serially, and the TDM issue schedule must keep the per-wave outstanding count at
three or fewer. The acceptance criterion is a reduction in both
`s_barrier_wait 0xffff` and `s_barrier_wait 0xfffd`; a producer that merely
adds descriptors without reducing barrier stall must be rejected.

### Phase 6: occupancy redesign is a separate kernel family

Two resident workgroups per WGP would require approximately:

```text
LDS  <= 160 KiB/workgroup
VGPR <= 512/wave
```

The current values are `320 KiB` and `1024`. Achieving both thresholds requires
a smaller output tile, fewer live accumulator banks, and a one- or two-stage K
pipeline. That increases logical task count and synchronization overhead and is
not an incremental modification of the current kernel. It should be explored
only as a separately named kernel after the wait-overlap work above.

Dynamic VGPR deallocation is not useful here: the architecture does not allow a
wave to reallocate VGPRs after deallocation, while this persistent wave must
continue into subsequent tasks.

## Candidate acceptance sequence

Every implementation phase should create a new named assembly file and retain
`persistent_overlap_pad8_prefetch_stage0.s` as the comparison baseline.

1. Run the static ABI/resource/TDM/barrier audit.
2. Run standalone random validation with several seeds and repeated launches.
3. Run full random MoE e2e validation and require the same output hash and
   `rel_l2 < 0.01`.
4. Run a three-round alternating-order `e2e-const0` comparison on an idle
   `d01-3`.
5. Capture at least three ATT runs and require lower median dispatch cycles and
   lower steady-task cycles.
6. Confirm no regression in the three-kernel A/ScaleA producer pipeline.

Representative commands:

```bash
bash /data/yanguahe/code/gpu_users.sh

AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/<candidate>.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=1 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random

AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/<candidate>.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0

AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/<candidate>.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att
```

GPU/KFD ownership must be checked before and after every GPU run. No GPU reset,
process termination, or machine restart is part of this workflow.
