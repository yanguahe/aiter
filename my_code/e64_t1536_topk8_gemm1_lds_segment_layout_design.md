# E64/T1536/topk8 GEMM1 LDS segment-aware layout

## Scope

This design applies only to the retained GEMM1 shape and kernel geometry:

```text
experts          = 64
tokens           = 1536
topk             = 8
model_dim        = 7168
inter_dim        = 2048
tile             = 192x256x256
workgroup        = w2x4 (8 waves)
buffers          = 4
cluster          = 4x1
persistent tasks = 4
B TDM hint       = 6 (NT_HT)
WMMA reuse       = reuseB / reuse3
```

The implementation does not change the external A, B, ScaleA, ScaleB, or
output layouts. It changes only the LDS placement and the mapping between the
two resident wave slots and logical `wave_m`.

## Hardware basis

The design follows the local MI400 hardware documentation rather than treating
all long `s_wait_dscnt` latency as a generic bank conflict.

### LDS banks and segments

`MI400_Shader_Programming#65.txt`, section 4.7.1 (page 169), states that LDS:

- shares a 384 KiB physical memory with WGP$;
- may allocate up to 320 KiB to one workgroup;
- has 64 banks, each 4 bytes wide;
- is divided into 64 KiB segments;
- decodes an LDS address as
  `{Segment[2:0], SRAM_address[7:0], Bank[5:0], ByteInBank[1:0]}`.

The dual-port architecture notes in `architecture.txt`, section 2.12 (pages
41-43), state that the two LDS/TCP ports conflict when they access the same
segment. When any segment conflicts across the ports, only the priority port
is sent to RAM for that arbitration opportunity.

### SIMD issue and intrinsic LDS latency

`MI400_Shader_Programming#65.txt`, sections 1.5 and 5.3.6, state that:

- each SIMD pair shares a command/data bus to LDS and TA;
- each SIMD may issue at most one VMEM instruction per cycle;
- the two SIMDs in a pair compete for one LDS/TA bus.

Therefore, two resident waves on the same SIMD do not independently use the two
LDS ports in the same cycle. The direct dual-port hazard is between the two
SIMD-pair ports. Separating the two resident slots is still useful for keeping
their phases from repeatedly selecting the same segment, but the layout must
first separate the two ports.

Section 5.7.6 gives `DS_LOAD_B128` an ideal independent repeat rate of 2 cycles
and an approximate dependent latency of 58 cycles. The existing lane mapping
already distributes each 16-lane half over all 64 banks. A small row padding
cannot reduce the fundamental two-cycle transfer cost; the avoidable part is
cross-port segment serialization.

## Baseline trace evidence

The retained pre-change kernel was:

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ps4pf2hm_earlynext_o1w_xor_wait1
```

Its eight valid compute waves had a combined span of `994,047 cycles`.

| wait class | latency cycles | share of wave span |
|---|---:|---:|
| `s_wait_dscnt` | 222,278 | 22.3609% |
| `s_barrier_wait` | 185,600 | 18.6711% |
| `s_wait_tensorcnt` | 15,224 | 1.5315% |

`s_wait_dscnt 0x0` and `s_wait_dscnt 0xa` accounted for 91.91% of all DScnt
latency. The baseline four-buffer pitch was `0xee00`, so most A/B/scale reads
for a given stage selected the same segment from both SIMD-pair ports.

## Final LDS layout

For this shape:

```text
STAGE_A  = 0x6000 bytes
STAGE_B  = 0x8000 bytes
STAGE_SA = 0x0600 bytes
STAGE_SB = 0x0800 bytes

A_HALF   = 0x3000 bytes
B_HALF   = 0x4000 bytes
SA_HALF  = 0x0300 bytes
SB_HALF  = 0x0400 bytes
```

The final layout allocates five 64 KiB segments, or 320 KiB total:

| segment | contents | bytes used |
|---:|---|---:|
| 0 | four A halves for logical `wave_m=0`, four ScaleA halves for `wave_m=0`, four ScaleB halves for port 0 | 56,320 |
| 1 | four A halves for logical `wave_m=1`, four ScaleA halves for `wave_m=1`, four ScaleB halves for port 1 | 56,320 |
| 2 | four B halves consumed by port 0 | 65,536 |
| 3 | four B halves consumed by port 1 | 65,536 |
| 4 | fused SiLU output LDS | 52,224 |

Within segments 0 and 1:

```text
A stage s      = s * 0x3000
ScaleA stage s = 0xc000 + s * 0x0300
ScaleB stage s = 0xcc00 + s * 0x0400
```

Within segments 2 and 3:

```text
B stage s = s * 0x4000
```

No input region crosses a segment boundary. The output occupies segment 4 and
does not overlap next-task stage 0/1 prefetches.

The GEMM1 path keeps `A_CACHE_STAGES=0`, as it did before this change. The
two-stage A cache belongs to the GEMM2 persistent schedule, so this layout does
not add A reloads or change the number of input TDM operations.

## Resident-slot remap

The logical M ownership is changed from:

```text
wave_m = wave // 4
```

to:

```text
physical_wave_slot = wave // 4
port               = wave_n // 2
wave_m             = physical_wave_slot ^ port
```

Every logical `(wave_m, wave_n)` tile still appears exactly once. For the same
resident-slot phase, the two SIMD-pair ports now select different A/ScaleA
segments, while B and ScaleB are already split by port.

| physical port | resident slot | logical `wave_m` | A/ScaleA segment | B segment | ScaleB segment |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 2 | 0 |
| 0 | 1 | 1 | 1 | 2 | 0 |
| 1 | 0 | 1 | 1 | 3 | 1 |
| 1 | 1 | 0 | 0 | 3 | 1 |

The output TDM keeps the original predicate
`(store_phase ^ wave_m) == target_phase`. With the remap, this predicate
selects one physical resident slot per output phase and still covers all four
48-row output slices. Replacing `wave_m` with `physical_wave_slot` was tested
and was incorrect because it left the middle output rows unwritten.

## Correctness and performance

The final kernel symbol is:

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ldsseg5_xorm_ps4pf2hm_earlynext_o1w_xor_wait1
```

Random validation preserves the existing accuracy:

```text
logits_diff = 3.38491e-06
rel_l2      = 0.00260189
pass        = True
```

The adjacent idle-GPU const0 comparison on a07-3 was:

| version | GEMM1 samples (us) | median (us) | change |
|---|---|---:|---:|
| retained pre-change linear layout | 75.835, 75.903, 74.374 | 75.835 | baseline |
| five-segment split + `wave_m` remap | 73.098, 74.648, 73.922 | 73.922 | 2.52% faster |

Logs:

```text
baseline:  my_code/moe_prefill_switch_ab_runs/20261003T090157Z
candidate: my_code/moe_prefill_switch_ab_runs/20261003T092205Z
```

## Trace result after the redesign

The trace was captured from the functionally identical pre-cleanup symbol
ending in `_ldsseg5_split_xorm_v8`. It is stored on a07-3 at:

```text
my_code/thread_trace_runs/e64_t1536_topk8_gemm1_ldsseg5_split_xorm_a07_3_20261003
```

The comparison against the pre-change trace is:

| metric | pre-change | segment-aware | change |
|---|---:|---:|---:|
| average full-wave interval | 120,636.4 | 113,773.1 cycles | -5.69% |
| combined valid-wave span | 994,047 | 940,935 cycles | -5.34% |
| `s_wait_dscnt` latency | 222,278 (22.3609%) | 89,946 (9.5592%) | -59.53% cycles |
| `s_wait_dscnt` exposed stall | 215,534 (21.6825%) | 83,250 (8.8476%) | -61.37% cycles |
| `s_barrier_wait` latency | 185,600 (18.6711%) | 284,239 (30.2081%) | +53.15% cycles |
| `s_wait_tensorcnt` latency | 15,224 (1.5315%) | 28,310 (3.0087%) | +85.96% cycles |
| DScnt + barrier latency | 407,878 (41.0321%) | 374,185 (39.7674%) | -8.26% cycles |

The redesign therefore removes most of the avoidable DScnt delay, but much of
that gain reappears as barrier waiting. The slower resident slot now determines
progress while the faster slot waits at each workgroup barrier. The higher
`s_wait_tensorcnt` reflects the changed issue/arrival timing; the GEMM1 input
TDM count is unchanged.

## Comparison with the refactor v123 cyclic layout

The `hyg/moe_a4w4_pr_refactor` worktree uses a four-segment cyclic layout for
its fused-quant GEMM1. Each ring stage owns one 64 KiB segment. The first
A/ScaleA owner lives at the front of stage `s`, while the second owner lives at
the tail of stage `(s + 1) % 4`. B and ScaleB remain in the middle of stage
`s`. This places the two resident `wave_m` values on different segments while
reducing LDS to 256 KiB.

That method was ported to the current BF16-output GEMM1 and tested in two forms:

| cyclic variant | GEMM1 samples (us) | median (us) | versus five-segment |
|---|---|---:|---:|
| cyclic + 32-byte padding per 1 KiB | 74.924, 74.243, 74.783 | 74.783 | 1.16% slower |
| cyclic without padding | 73.604, 75.236, 74.947 | 74.947 | 1.39% slower |

Both variants passed random validation after correcting the relocated B base
from `STAGE_A` to `B_OFF`. Neither improves on the five-segment median of
73.922 us, so the cyclic changes were not retained in the active source. The
likely reason is that this kernel still has a 52,224-byte BF16 output staging
tile and a different epilogue/barrier balance from the refactor worktree's
compact fused-quant output. The 64 KiB reduction does not compensate for the
remaining B/ScaleB same-segment traffic and cyclic address work here.

Logs:

```text
cyclic + p32: my_code/moe_prefill_switch_ab_runs/20261003T142356Z
cyclic:       my_code/moe_prefill_switch_ab_runs/20261003T143049Z
```

The remaining `s_wait_dscnt` is not evidence that all remaining cycles are
bank conflicts. It includes the documented intrinsic `DS_LOAD_B128` latency,
same-SIMD issue serialization, and true producer/consumer dependencies. The
next optimization target should be wave-arrival balance or scheduling around
the barriers, rather than more padding or another wholesale LDS relocation.

## Reproduction

Performance and correctness:

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048

ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

Trace analysis:

```bash
python3 my_code/analyze_gemm_wait_cycles.py "$UI_DIR" --top 30

python3 /data/yanguahe/code/wk_sp1/cursor_rules/fmha_flydsl_new_api_opt/.cursor/rules/trace_segment_cycles.py \
  "$UI_DIR" \
  --wv 0 \
  --interval-start 'global_prefetch_b8 v0, s[0:1] scope:SCOPE_SE' \
  --interval-end 's_endpgm' \
  --rank-by sum \
  -k 40 \
  --top-events \
  --hide-occurrence-details
```

## Exact-symbol recapture on the retained kernel

The cleaned, retained `_ldsseg5_xorm` symbol was recaptured on a07-3 after the
v123 cyclic-layout experiment was removed. The source SHA256 was:

```text
feac6e778189df20ad56b9cada9ee5c0822db3e78d935a5e30fa4f115bd2cb6d
```

The exact captured symbol was:

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ldsseg5_xorm_ps4pf2hm_earlynext_o1w_xor_wait1
```

The capture is stored on a07-3 at:

```text
my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_ldsseg5_xorm_exact_a07_3_20261003
```

To minimize capture count, the run selected SIMD3 and collected its two slots
on all four shader engines. Eight traces containing `v_wmma*` were treated as
valid compute waves. Eighteen early-exit traces with no `v_wmma*` were excluded
from the hot-path denominator. The eight full `wave.begin -> wave.end` spans
sum to `937,378 cycles`; this is the denominator used below. The first
`global_prefetch_b8` to `s_endpgm` interval averages `115,551.8 cycles`.

### Aggregate waits

| wait class | dynamic count | decoder latency cycles | share of full wave span | exposed stall cycles | exposed stall share |
|---|---:|---:|---:|---:|---:|
| `s_wait_tensorcnt` | 936 | 34,281 | **3.657116%** | 33,345 | **3.557263%** |
| `s_wait_dscnt` | 6,696 | 94,503 | **10.081632%** | 87,807 | **9.367299%** |
| `s_barrier_wait` | 960 | 284,920 | **30.395422%** | 283,960 | **30.293009%** |
| `s_wait_kmcnt` | 72 | 23,838 | 2.543051% | 23,766 | 2.535370% |

`s_wait_tensorcnt + s_wait_dscnt` contribute `128,784 cycles`, or
`13.738748%` of the complete compute-wave span. Their exposed stall is
`121,152 cycles`, or `12.924562%`.

All observed `s_wait_tensorcnt` forms are:

| instruction | count | latency cycles | latency/span | exposed stall/span | maximum latency |
|---|---:|---:|---:|---:|---:|
| `s_wait_tensorcnt 0x2` | 808 | 18,649 | 1.989486% | 1.903288% | 755 |
| `s_wait_tensorcnt 0x3` | 8 | 12,890 | 1.375112% | 1.374259% | 2,844 |
| `s_wait_tensorcnt 0x0` | 64 | 2,448 | 0.261154% | 0.254326% | 603 |
| `s_wait_tensorcnt 0x1` | 56 | 294 | 0.031364% | 0.025390% | 191 |

All observed `s_wait_dscnt` forms are:

| instruction | count | latency cycles | latency/span | exposed stall/span | maximum latency |
|---|---:|---:|---:|---:|---:|
| `s_wait_dscnt 0x0` | 1,040 | 62,908 | 6.711060% | 6.600112% | 1,458 |
| `s_wait_dscnt 0xa` | 992 | 21,460 | 2.289365% | 2.183537% | 1,405 |
| `s_wait_dscnt 0x1d` | 864 | 3,456 | 0.368688% | 0.276516% | 4 |
| `s_wait_dscnt 0x6` | 64 | 1,615 | 0.172289% | 0.165462% | 82 |
| `s_wait_dscnt 0x8` | 928 | 1,075 | 0.114682% | 0.015682% | 19 |
| `s_wait_dscnt 0x4` | 56 | 1,043 | 0.111268% | 0.105294% | 72 |
| `s_wait_dscnt 0x1b` | 864 | 864 | 0.092172% | 0.000000% | 1 |
| `s_wait_dscnt 0x19` | 864 | 864 | 0.092172% | 0.000000% | 1 |
| `s_wait_dscnt 0x17` | 864 | 864 | 0.092172% | 0.000000% | 1 |
| `s_wait_dscnt 0xf` | 32 | 128 | 0.013655% | 0.010241% | 4 |
| `s_wait_dscnt 0x13` | 32 | 128 | 0.013655% | 0.010241% | 4 |
| `s_wait_dscnt 0x2` | 32 | 34 | 0.003627% | 0.000213% | 3 |
| `s_wait_dscnt 0xd` | 32 | 32 | 0.003414% | 0.000000% | 1 |
| `s_wait_dscnt 0x11` | 32 | 32 | 0.003414% | 0.000000% | 1 |

### Current bottleneck

The primary exposed bottleneck is now workgroup barrier arrival skew, rather
than LDS completion itself. Two repeated hot-loop stage-handoff barriers at
PC `0x37f4` and `0x410c` account for `227,027 cycles`, or `24.219365%` of the
full span. They follow this dependency chain:

```text
s_wait_tensorcnt 0x2
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
tensor_load_to_lds ...
```

Across both branch-specialized copies of that chain, the TENSORcnt waits take
`1.957481%`, the DScnt drains take `5.571285%`, and the barriers take
`24.219365%`; the complete stage-handoff chain therefore accounts for
`31.748131%` of the measured span.

The two decoded resident slots are strongly asymmetric:

| decoded slot | span cycles | `s_wait_tensorcnt` | `s_wait_dscnt` | `s_barrier_wait` |
|---|---:|---:|---:|---:|
| slot 0 | 467,534 | 3.953937% | 8.625469% | **39.957308%** |
| slot 1 | 469,844 | 3.361754% | 11.530636% | **20.880548%** |

This indicates that one resident slot commonly reaches the stage barrier much
earlier, while the partner slot carries more DS completion delay. The WGP-level
completion distribution is otherwise balanced: `analyze_att_capture.py`
reports mean completion imbalance `1.4689%`, median `0.9937%`, and maximum
`3.2162%`. The remaining problem is therefore mostly intra-workgroup progress
skew around the repeated input-ring handoff, not uneven dispatch across WGPs.

The next largest useful instruction class is
`v_wmma_scale_f32_32x16x128_f4`: `35.273390%` decoder latency and
`14.626863%` exposed stall. The remaining non-compute ordering costs are
`s_wait_dscnt` at `10.081632%`, `s_wait_tensorcnt` at `3.657116%`, and
prologue/scalar `s_wait_kmcnt` at `2.543051%`.

The hardware interpretation follows the MI400 Shader Programming Guide:

- Section 4.7, pages 169-170: LDS is organized as 64 banks and 64 KiB
  segments; indexed LDS latency varies with bank conflicts, and completion is
  tracked by DScnt.
- Section 4.3.7.2.4, page 99: an LDS read decrements DScnt when its result is
  available in VGPRs. A long `s_wait_dscnt` can therefore include bank/segment
  contention, issue serialization, intrinsic DS latency, and a true consumer
  dependency; ATT alone does not separate those components.
- Section 4.10.1, pages 206-207: TDM operations are tracked by TENSORcnt and
  complete in order within one wave, but are unordered across waves.
- Section 5.2.2, page 222: equal-priority scheduling is intended to help
  cooperating waves reach barriers together. The measured slot asymmetry shows
  that the kernel still has enough path or resource imbalance to leave roughly
  30% of its span attributed to barrier waits despite that scheduling policy.

## All-SIMD capture: logical-wave placement and TDM ownership

The preceding exact-symbol result selected only physical SIMD3. A second ATT
run captured SIMD selectors 0, 1, 2, and 3 independently to determine where all
eight logical workgroup waves execute:

```text
my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_ldsseg5_xorm_allsimd_a07_3_20261003
```

For every capture, `occupancy.json` was grouped by physical SA/WGP and split
into eight-wave allocation episodes. All `128/128` episodes in each of the four
captures had the same launch order:

```text
(SIMD0,slot0), (SIMD3,slot0), (SIMD2,slot0), (SIMD1,slot0),
(SIMD0,slot1), (SIMD3,slot1), (SIMD2,slot1), (SIMD1,slot1)
```

The SPI resource-allocation documentation states that a threadgroup is broken
into individual waves and emitted sequentially. Matching that allocation order
to the kernel's `wave = thread_idx.x // 32` gives:

| logical wave | physical SIMD | resident slot | input TDM owner |
|---:|---:|---:|---|
| 0 | 0 | 0 | A half 0 |
| 1 | 3 | 0 | A half 1 |
| 2 | 2 | 0 | B half 0 |
| 3 | 1 | 0 | B half 1 |
| 4 | 0 | 1 | ScaleA half 0 |
| 5 | 3 | 1 | ScaleA half 1 |
| 6 | 2 | 1 | ScaleB half 0 |
| 7 | 1 | 1 | ScaleB half 1 |

The branch-specific first TDM PCs independently confirm the owner groups:

```text
0x203c -> A owners      (waves 0/1)
0x2090 -> B owners      (waves 2/3, TH_LOAD_NT_HT)
0x2118 -> ScaleA owners (waves 4/5)
0x2180 -> ScaleB owners (waves 6/7)
```

All four shader engines reproduced the same pairing:

```text
physical SIMD0: A + ScaleA
physical SIMD3: A + ScaleA
physical SIMD2: B + ScaleB
physical SIMD1: B + ScaleB
```

Each logical wave had four valid compute-wave samples, one per shader engine.
Their average full-wave spans and wait shares were:

| wave | owner | average span | TENSORcnt latency | DScnt latency | barrier latency |
|---:|---|---:|---:|---:|---:|
| 0 | A half 0 | 119,468.8 | 4.286% | 9.719% | 41.300% |
| 1 | A half 1 | 113,986.0 | 4.187% | 9.334% | 38.009% |
| 2 | B half 0 | 119,707.2 | **26.257%** | 8.544% | 16.263% |
| 3 | B half 1 | 115,774.8 | **28.758%** | 6.157% | 15.775% |
| 4 | ScaleA half 0 | 120,839.5 | 3.382% | **12.551%** | 22.611% |
| 5 | ScaleA half 1 | 114,562.5 | 3.568% | **11.981%** | 18.546% |
| 6 | ScaleB half 0 | 121,103.2 | 8.858% | **12.331%** | 21.233% |
| 7 | ScaleB half 1 | 116,332.2 | 8.234% | 8.572% | 23.554% |

Waves 4-7 are therefore not uniformly faster merely because their scale TDM
payloads are smaller. All eight waves execute a full `96x64` WMMA subtile. The
ScaleA/ScaleB owner waves also encounter different LDS consumer paths and the
second output-store phase. In these captures, waves 4 and 6 have the two largest
average spans, while the B-owner waves have by far the largest TENSORcnt wait.

Across all 32 valid compute-wave traces, the whole-workgroup wait picture is:

| wait class | latency/span | exposed stall/span |
|---|---:|---:|
| `s_barrier_wait` | 24.642981% | **24.541046%** |
| `s_wait_tensorcnt` | 10.947289% | **10.847902%** |
| `s_wait_dscnt` | 9.917929% | **9.206930%** |
| `s_wait_kmcnt` | 1.704177% | **1.696532%** |

This all-SIMD result refines the SIMD3-only result. Barrier waiting remains the
largest aggregate exposed wait, but the B-owner waves themselves spend
`26-29%` of their time in TENSORcnt waits. Much of the barrier time recorded by
the A and scale-owner waves is therefore a consequence of waiting for the
B-owner path to finish its larger 16 KiB-per-stage TDM and associated compute
path.
