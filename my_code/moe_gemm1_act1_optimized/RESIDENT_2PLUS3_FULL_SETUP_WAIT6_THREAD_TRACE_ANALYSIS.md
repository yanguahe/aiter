# `resident 2+3, full setup, wait6` thread-trace analysis

## Scope and conclusion

This report analyzes the pure-assembly kernel:

```text
my_code/moe_gemm1_act1_optimized/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_
  ab4_scale_half_tdm_full_setup_loop_wait6.s
SHA256=02dffd3a7f6ca015a25d52e8a2273c6cd49deed57bde4f0f758c1a97c7c70c06
```

The trace was captured on `d01-3` through the MoE e2e launcher. The profiler
run immediately before capture reported:

```text
GEMM1      = 531.689 us
fused MoE  = 1415.31 us
logits_diff= 0
rel_l2     = 0
```

The main result is that this version is synchronization-bound after its useful
WMMA work. The representative steady task is `29,724.4 GFXCLK cycles`. Explicit
wait stall has a cross-owner median of `5,807.9 cycles/task`, or `19.54%`:

1. `s_barrier_wait`: `3,404.5 cycles/task`, `11.45%`.
2. `s_wait_tensorcnt`: `1,653.5 cycles/task`, `5.56%`.
3. `s_wait_dscnt`: `613.1 cycles/task`, `2.06%`.

The dominant individual site is the K-ring cluster barrier. It executes six
times per task and costs a cross-owner median of `1,588.0 cycles/task`, or
`5.34%`. The owner spread is very large: A waits only `353.0 cycles/task`, while
B, ScaleA, and ScaleB wait `1,568.3`, `1,925.0`, and `1,607.6 cycles/task`.
This makes owner arrival imbalance, followed by incomplete hiding of input TDM,
the highest-value optimization target.

## Capture method

The decoded traces are stored on `d01-3` under:

```text
my_code/moe_gemm1_act1_optimized/history_runs/full_setup_wait6_att_20260916/
```

The four captures selected one wave from each SIMD and mapped to the logical
owners as follows:

| ATT selection | logical owner |
|---|---|
| `SIMD0` | A |
| `SIMD1` | ScaleB |
| `SIMD2` | ScaleA |
| `SIMD3` | B |

The capture used the normal MoE e2e launcher, not the standalone ATT launcher:

```bash
python3 my_code/moe_gemm1_act1_optimized/run_e2e_candidate.py \
  --isa my_code/moe_gemm1_act1_optimized/persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_ab4_scale_half_tdm_full_setup_loop_wait6.s \
  --grid-x 16 --grid-y 16 -- \
  --scenario bench --data-format a4w4 \
  --experts 96 --tokens 16384 --topk 6 --iters 2 \
  --model-dim 7168 --inter-dim 3072 --act silu \
  --no-bias --no-check-aot-cache --const-init 0
```

Each SIMD was captured separately with this ATT selection:

```yaml
kernel_include_regex: '^moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1$'
kernel_iteration_range: "[8]"
advanced_thread_trace: true
att_target_cu: 1
att_shader_engine_mask: "0x1"
att_simd_select: "0"  # repeated for 1, 2, and 3
att_buffer_size: "0x10000000"
```

The direct eighth-invocation selection is deliberate. A preliminary
`--kernel-trace --stats` pass was not used: it is unnecessary for this fixed
e2e launch sequence and previously made the capture path less reliable.

The analysis uses the checked-in parser required by the local workflow:

```text
my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py
SHA256=6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

Consecutive occurrences of the following sequence delimit one persistent
task:

```asm
s_add_co_u32 s28, s28, 16
s_cmp_lt_u32 s28, 0x240
s_cbranch_scc0 5
```

The first complete interval in each captured wave is discarded. This avoids
mixing the warm entry path with the steady persistent loop.

## Sampling coverage

| owner | SIMD | matched tasks | steady tasks | mean cycles/task | median | p90 | full traced wave cycles |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 2 | 1 | 26,330.0 | 26,330 | 26,330.0 | 1,192,871 |
| B | 3 | 14 | 13 | 29,804.2 | 29,091 | 33,021.4 | 1,205,001 |
| ScaleA | 2 | 3 | 2 | 32,916.0 | 32,916 | 35,305.6 | 1,289,220 |
| ScaleB | 1 | 18 | 17 | 29,644.6 | 28,315 | 31,811.4 | 1,179,387 |

The representative task value in this report is the median of the four owner
means, `29,724.4 cycles`. It is intentionally not weighted by the number of
intervals, because the four SIMD captures cover unequal portions of the
persistent work queue. A and ScaleA have few steady intervals, so their exact
percentages are directional rather than population estimates. The same owner
ordering nevertheless appears at repeated barrier and TDM sites, making the
critical-path conclusion useful.

## How cycle attribution is interpreted

`trace_segment_cycles.py` reads each dynamic instruction as:

```text
[timestamp, type, stall, latency, code_idx]
```

Two different quantities are used below:

- **wait stall** is the trace `stall` field on `s_*wait_*` and
  `s_barrier_wait`. It is the direct blocked time attributed to the wait.
- **issue timeline** assigns the timestamp gap from one dynamic instruction to
  the next instruction to the first instruction. This partition sums to the
  task interval, but a large gap can include scheduler delay and contention.
- **completion latency** is the trace `latency` field. Async TDM and LDS work can
  overlap later instructions, so completion latency must not be added as if it
  were serialized kernel time.

This distinction matters for `tensor_load_to_lds`: the issue itself is cheap,
while its unresolved completion is exposed later by `s_wait_tensorcnt`.

## Explicit wait stall by owner

| owner | task cycles | `s_barrier_wait` | `s_wait_tensorcnt` | `s_wait_dscnt` | other waits | all explicit waits |
|---|---:|---:|---:|---:|---:|---:|
| A | 26,330.0 | 2,986.0 (11.34%) | 496.0 (1.88%) | 646.0 (2.45%) | 43.0 (0.16%) | 4,171.0 (15.84%) |
| B | 29,804.2 | 3,823.1 (12.83%) | 1,506.5 (5.05%) | 801.3 (2.69%) | 83.7 (0.28%) | 6,214.6 (20.85%) |
| ScaleA | 32,916.0 | 5,204.0 (15.81%) | 1,951.0 (5.93%) | 555.0 (1.69%) | 43.0 (0.13%) | 7,753.0 (23.55%) |
| ScaleB | 29,644.6 | 2,894.5 (9.76%) | 1,800.4 (6.07%) | 580.2 (1.96%) | 126.0 (0.43%) | 5,401.2 (18.22%) |
| cross-owner median | 29,724.4 | 3,404.5 (11.45%) | 1,653.5 (5.56%) | 613.1 (2.06%) | 63.3 (0.21%) | 5,807.9 (19.54%) |

The theoretical ceiling from deleting every explicit wait stall is therefore
about `19.5%` at the representative-task level. This is not an achievable
speedup: the waits enforce real data and synchronization dependencies, and
moving one wait generally moves part of the latency to another site.

## Dominant K-ring cluster barrier

The six repeated K-ring waits use the same logical cluster barrier, with a
different PC on the two owner-path families:

```text
A / ScaleA: 0xb5e4  s_barrier_wait 0xfffd
B / ScaleB: 0xd61c  s_barrier_wait 0xfffd
```

The surrounding decoded sequence is:

```asm
v_wmma_scale_f32_32x16x128_f4 ...
v_wmma_scale_f32_32x16x128_f4 ...
s_cbranch_scc0 ...
s_barrier_wait 0xfffd
```

Per-task cost:

| owner | hits/task | stall cycles/task | task share |
|---|---:|---:|---:|
| A | 6 | 353.0 | 1.34% |
| B | 6 | 1,568.3 | 5.26% |
| ScaleA | 6 | 1,925.0 | 5.85% |
| ScaleB | 6 | 1,607.6 | 5.42% |
| cross-owner median | 6 | 1,588.0 | 5.34% |

The hardware guide states that `-3` is the user cluster barrier and that the
barrier completes only after all workgroups in the cluster have signaled. It
also recommends synchronizing the waves in each workgroup before one wave per
workgroup signals the cluster barrier. Thus the measured stall contains the
combined arrival spread within each WG and across the 4x4 cluster.

A reaches these waits much later than the other owners and consequently waits
far less. ScaleA is the earliest arrival and waits the longest. This strongly
indicates that the A-owner path is close to the local critical arrival path.
The trace alone cannot separate that intra-WG skew from cross-WG scheduling
skew, but adding work to A before this barrier would move in the wrong
direction. The useful choices are to shorten A's path or move independent A
setup to an earlier part of the ring while giving the early owners useful work
to cover their otherwise idle interval.

The task-boundary cluster wait at `0x10600` is smaller:

| owner | stall cycles/task | task share |
|---|---:|---:|
| A | 119.0 | 0.45% |
| B | 385.7 | 1.29% |
| ScaleA | 353.0 | 1.07% |
| ScaleB | 313.3 | 1.06% |

It is secondary to the six K-ring barriers.

## Input TDM waits

The large one-hit `s_wait_tensorcnt 0x6` sites occur after the initial/full
setup path and before LDS consumption:

| owner | PC | stall cycles/task | task share |
|---|---:|---:|---:|
| B | `0x3dcc` | 995.2 | 3.34% |
| ScaleA | `0x4dc8` | 932.5 | 2.83% |
| ScaleB | `0x5df0` | 959.2 | 3.24% |

Their decoded shape is:

```asm
s_barrier_signal -1
s_barrier_wait 0xffff
s_wait_tensorcnt 0x6
s_barrier_signal -1
s_barrier_wait 0xffff
ds_load_...
```

The next-stage steady-loop waits add more exposed TDM latency, especially on
the scale owners:

```text
ScaleA 0x993c: 746.0 cycles/task, 7 hits/task, 2.27%
ScaleB 0xc184: 216.4 cycles/task, 7 hits/task, 0.73%
ScaleB 0xc988: 115.2 cycles/task, 7 hits/task, 0.39%
```

Overall `s_wait_tensorcnt` is only `1.88%` on A but `5.05%` to `6.07%` on B,
ScaleA, and ScaleB. The 2+3 distribution therefore reduces TDM ownership
concentration but does not fully cover the input completion latency. The
follow-up experiment that removes ScaleA/ScaleB TDM loads is a useful diagnostic:
it will show how much of these waits and the K-ring arrival spread is caused by
scale traffic rather than A/B payload traffic.

## Workgroup barriers and LDS waits

Total workgroup-barrier (`s_barrier_wait 0xffff`) stall varies strongly by
owner:

```text
A       2,227.0 cycles/task, 8.46%
B       1,578.8 cycles/task, 5.30%
ScaleA  2,633.5 cycles/task, 8.00%
ScaleB    713.3 cycles/task, 2.41%
```

Large one-time setup waits include:

```text
A       0x2dd4  923.0 cycles/task
A       0x2798  521.0 cycles/task
B       0x3690  780.2 cycles/task
ScaleA  0xb198  767.0 cycles/task
ScaleA  0xa994  664.0 cycles/task
```

These waits align four owner waves before consuming shared LDS state or before
issuing the next owner-specific transfer. Their imbalance is the second part of
the synchronization problem; it is not a raw barrier-instruction throughput
limit.

`s_wait_dscnt` costs `1.66%` to `2.68%` per owner. The recurring hot sites are
`s_wait_dscnt 0x8` immediately after a burst of `ds_load_b128` and WMMA work,
for example `0xaddc`, `0xb614`, `0xc61c`, and `0xcfcc`. This is real LDS-read
completion pressure, but it is materially smaller than the cluster and TDM
waits.

## Non-wait instruction latency and issue timeline

The representative issue-timeline split is shown below. Each row is the median
of the four owner values. Independently taking medians means the rows are for
ranking and do not form an exact additive decomposition.

| group | cycles/task | representative share | observed owner range |
|---|---:|---:|---:|
| WMMA issue | 9,256.9 | 31.14% | 9,196.0-9,433.6 |
| SALU/control issue or scheduling gap | 5,421.4 | 18.24% | 4,781.0-5,576.0 |
| barrier wait timeline | 3,475.5 | 11.69% | 2,970.2-5,279.5 |
| LDS read issue or completion gap | 3,279.1 | 11.03% | 3,081.0-3,387.5 |
| packed SiLU VALU | 1,924.3 | 6.47% | 1,750.0-2,647.0 |
| TENSORcnt wait timeline | 1,684.0 | 5.67% | 526.0-1,982.0 |
| other VALU | 1,348.0 | 4.53% | 1,320.0-1,492.8 |
| EXP/RCP issue | 1,110.2 | 3.73% | 1,000.0-1,808.5 |
| DScnt wait timeline | 981.1 | 3.30% | 832.0-1,242.2 |
| TDM issue | 351.0 | 1.18% | 277.0-485.4 |

WMMA is the largest category, but it is productive compute rather than a wait.
Individual WMMA completion latency is typically `11-17 cycles`; the roughly
`9.3k cycles/task` total comes from the large number of matrix instructions.

The longest non-wait completion events are `ds_load_b128`:

| owner | example PC | mean latency | max latency | issue gap/task |
|---|---:|---:|---:|---:|
| B | `0x3ea4` | 27.1 | 258 | 27.1 |
| ScaleA | `0x4ea0` | 121.5 | 242 | 121.5 |
| ScaleB | `0x5ec8` | 51.4 | 194 | 51.4 |

These isolated long returns agree with the `s_wait_dscnt` evidence, but their
completion latency overlaps other instructions. No non-wait PC contributes a
stable serialized cost comparable with the `~1.59k-cycle` K-ring barrier or
the `~0.95k-cycle` full-setup TDM waits.

Several one-cycle SALU branches show `100-315` cycle next-issue gaps, and the
ScaleA epilogue has similar gaps assigned to `v_exp_f32`, `v_rcp_f32`, and
packed operations. Their architectural latency remains only `1-2 cycles` in
the trace. They should be interpreted as wave scheduling or shared-pipeline
gaps at that timestamp, not as the latency of the named instruction. This is
why the bottleneck conclusion uses explicit wait stall and repeated owner
patterns rather than attributing all following idle time to a branch or VALU.

## Resource usage and occupancy consequence

The code object metadata is:

```text
.amdhsa_group_segment_fixed_size 327680
.amdhsa_private_segment_fixed_size 0
.amdhsa_kernarg_size 184
.amdhsa_wavefront_size32 1
.amdhsa_next_free_vgpr 1024
.amdhsa_next_free_sgpr 106
block = (128, 1, 1) = 4 wave32
```

| resource | kernel use | documented gfx1250 limit | consequence |
|---|---:|---:|---|
| LDS | 327,680 bytes = 320 KiB/WG | 320 KiB/WG | full allocation; no room for another LDS buffer |
| VGPR | 1,024/wave | 1,024/wave and 1,024/SIMD | one such wave consumes the full SIMD VGPR pool |
| SGPR | 106/wave | 106 normal SGPRs/wave | all normal SGPR names are in use |
| scratch | 0 | n/a | no spill traffic |

The local hardware guide records these limits in:

```text
mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/
  MI400_Shader_Programming#65.txt

SGPR allocation: section 3.3.1.1, lines 2074-2084
VGPR allocation: section 3.3.2.1, lines 2151-2171
LDS allocation:  section 3.3.4,   lines 2444-2452
cluster barrier: section 4.3.6.6, lines 5364-5412
```

Because the block has four waves, the 1,024-VGPR allocation naturally places
one wave on each of the four SIMD32s and prevents a second resident wave on a
SIMD. The 320 KiB LDS allocation also occupies the documented maximum. There
is therefore no occupancy headroom to hide a blocked owner with another wave
or WG. Every exposed barrier, TENSORcnt, and DScnt wait appears directly in
elapsed cycles.

## Bottleneck ranking and next actions

1. **K-ring arrival imbalance is the largest single removable loss.** The six
   cluster waits consume a representative `5.34%`; A is the late owner. Shorten
   the A path before each ring transition, or move independent setup from that
   path earlier. Do not add more pre-barrier work to A.
2. **Input TDM completion is the next bottleneck.** B/Scale owners expose
   `5-6%` total TENSORcnt stall. Earlier issue, fewer scale transfers, or useful
   work between TDM issue and `s_wait_tensorcnt 0x6` should be tested.
3. **Workgroup arrival skew remains significant.** The large `0xffff` waits
   show that the four specialized owner paths are not balanced before LDS
   consumption. Reassigning owner-only descriptor/control work can help if it
   shortens the latest path.
4. **LDS completion is smaller but measurable.** Reorder independent WMMA or
   descriptor work between the last `ds_load_*` and `s_wait_dscnt 0x8`; the
   ceiling is roughly `2%` unless the schedule change also improves barriers.
5. **Adding another buffer is not a local option.** LDS, VGPR, and SGPR usage
   already reaches the documented limits. Any extra buffering first requires
   reclaiming existing LDS or registers.

The planned ScaleA/ScaleB-TDM removal experiment directly tests item 2. If it
substantially lowers `s_wait_tensorcnt` but leaves the six K-ring waits intact,
the ring is primarily owner-compute/scheduling limited. If both fall together,
scale TDM completion is delaying the cluster arrival path and should be
redesigned rather than merely rescheduled.
