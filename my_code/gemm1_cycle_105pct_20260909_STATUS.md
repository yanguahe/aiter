# gfx1250 MoE GEMM1 cycle optimization

## Current direction: baseline routeks layout by default

The active implementation now keeps the baseline
`moe_fused_quant_preshuffle_routeks_fd7168_r8_fp4_pk8_srctk6_noKS`
producer from commit `93665e8417afe1f07cb9bbe1c4902c38da8e3fa3` by default.  The
A-preshuffled `_apre` producer/consumer path is available as an opt-in
experiment through `AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1`; the default remains
off to preserve the existing production selection.  Its optimized producer
quantizes each source token once, builds a grouped-row-to-token map, and uses
an LDS transpose to emit coalesced A/ScaleA preshuffle stores.
The balanced-row shortcut, cluster-sync skipping, static geometry, and static
swizzle remain removed, so GEMM1 still uses dynamic `m_tile_map` lookup and
supports non-balanced routing.

The exact GEMM1 implementation from that commit is retained in-tree as
`mxfp4_preshuffle_gfx1250_tdm_93665e.py`.  The baseline case uses the
`my_code/run_gemm1_baseline_93665e.py` entry point to select it inside the
Python process.
This allows baseline and current kernels to be benchmarked without changing
Git state or relying on a baseline-selection environment variable.

Older timing results in the historical sections that contain an `_apre`,
`nocs`, or static-swizzle kernel must not be mixed with the current
baseline-routeks, dynamic-routing measurements.

Target workload:

```text
E=96, M/expert=1024, N=6144, K=7168, A4W4, fused SiLU, BF16 output
```

Target requested by the user:

```text
1.05 x 516,096 = 541,900.8 GFXCLK cycles
```

The fused kernel has an independent lower bound of at least 552,960 cycles
from the serialized WMMA and Trans32 work, before ordinary VALU, LDS/TDM,
barrier, launch, and pipeline-drain costs.  Experiments nevertheless measure
against both the requested raw-WMMA target and the fused-kernel lower bound.

## Reproducible same-machine comparison

Run the comparison script inside the existing `hyg_fyd1` container:

```bash
cd /data/yanguahe/code/wk_sp1/aiter
ROUNDS=2 RUN_VERIFY=1 RUN_ATT=0 \
  bash my_code/reproduce_compare.sh
```

It records the container hostname, UTC time, branch, HEAD, source SHA256 values,
per-kernel timings, and fused MOE end-to-end timing. It verifies all versions on
random input and benchmarks them with the exact target shape and `--const-init 0`:

1. `baseline_93665e`: exact GEMM1 kernel from commit `93665e...`.
2. `sync_mg4_fc8`: renamed current-branch baseline, synchronized `mg4/fc8`.
3. `sync_mg2_fc12`: synchronized `mg2/fc12` schedule candidate.
4. `sync_mg4_fc28`: synchronized `mg4/fc28` schedule candidate.
5. `sync_mg4_fc28_apre`: `sync_mg4_fc28` plus GEMM1 A preshuffle and the
   matching A/ScaleA LDS layout, TDM descriptors, and LDS load mapping.
6. `sync_mg4_fc28_hard`: synchronized `mg4/fc28` with hard SiLU.
7. `sync_mg4_fc28_relu`: synchronized `mg4/fc28` with ReLU gate.

Odd rounds run baseline-to-current; even rounds reverse the order.  The final
summary reports every sample, median/min/max, and improvement relative to the
same-run baseline.  Set `RUN_ATT=1` when paired cycle captures are required.

### Optimized A-preshuffle producer result on d01-3

#### Functional-equivalence conclusion

The first three-kernel prototype used:

```text
moe_invert_route_rows_tk6
dynamic_per_group_scaled_quant_kernel
moe_scatter_preshuffled_a_fd7168_r32_lds
```

It was byte-identical to the original direct `_apre` producer on the tested
inputs.  However, `dynamic_per_group_scaled_quant_kernel` is a separate HIP/C++
quantization implementation, so this prototype alone was not enough to claim
universal bit equivalence for special floating-point inputs.

The final implementation therefore uses:

```text
moe_quant_token_fd7168_fp4_pk8
moe_invert_route_rows_tk6
moe_scatter_preshuffled_a_fd7168_r32_lds
```

`moe_quant_token_fd7168_fp4_pk8` reuses exactly the same
`emit_mx_e8m0_scale`, `_ROUND_MODE`,
`v_cvt_scalef32_pk8_fp4_bf16`, four-lane MX32 reduction, and pack mapping as
the original routeks producer.  The inverse-map kernel only changes the
direction of the existing bijective route map, and the LDS scatter kernel only
copies and permutes those bytes into the requested A/ScaleA layouts.  Therefore
the valid routed rows are a byte-exact transformation, not a numerical
approximation.

The two-round run on `heliosr-1b114-d01-3` completed at
`20260910T153812Z`:

| case | GEMM1 samples (us) | GEMM1 median us | GEMM1 vs 93665e | MOE e2e samples (us) | MOE e2e median us | MOE e2e vs 93665e | random pass | hash |
|---|---|---:|---:|---|---:|---:|:---:|---|
| baseline_93665e | 696.024, 697.422 | 696.723 | +0.00% | 1673.15, 1671.24 | 1672.20 | +0.00% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28 | 609.458, 614.717 | 612.087 | +12.15% | 1588.37, 1593.85 | 1591.11 | +4.85% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28_apre | 560.762, 566.543 | 563.652 | +19.10% | 1444.34, 1451.25 | 1447.80 | +13.42% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |

The optimized GEMM1-input producer has the following two-round profile:

| component | samples (us) | median (us) |
|---|---|---:|
| FlyDSL token FP4 quant | 26.1, 26.6 | 26.35 |
| grouped-row inverse map | 5.5, 5.5 | 5.50 |
| LDS-transpose A/ScaleA preshuffle scatter | 75.1, 75.5 | 75.30 |
| explicit producer total | 106.7, 107.6 | 107.15 |

The previous direct `_apre` route-scatter producer measured about `450.5 us`.
The replacement reduces explicit producer GPU time by `76.22%` (`4.20x`) and
is below the `192 us` target.  Compared directly with `sync_mg4_fc28`, the new
candidate improves GEMM1 by `7.91%` and MOE end-to-end by `9.01%`.

Functional equivalence was checked at the producer outputs, not only at the
final MoE output.  The token quant kernel reuses the original routeks
producer's `emit_mx_e8m0_scale` and
`v_cvt_scalef32_pk8_fp4_bf16` path. Against the original direct `_apre`
producer:

```text
balanced_target:  payload_bad=0 scale_bad=0
unbalanced_random: payload_bad=0 scale_bad=0
```

The expanded byte-equivalence sweep also passed for `K=512` (fallback),
`1024`, `2048`, `3072`, `4096`, `7168`, and `8192`, with `topk=2/4/6/8`,
balanced and random routing, and finite BF16 edge values including signed zero
and FP4 rounding boundaries.  The fast three-kernel path is selected only when
its geometry is valid (`K % 1024 == 0`, grouped rows divisible by 32, non-EP
stage1 with `topk > 1`, and all temporary/output byte offsets below 2 GiB);
other shapes retain the original producer.

Padding rows are intentionally unwritten by both implementations and are never
consumed by GEMM.  The byte-comparison tests zero-initialized both output
buffers, so the full buffers, including padding, could also be compared.

#### Why three kernels are faster than the original fused producer

The original `_apre` producer assigns one wave to every route.  For the target
shape there are `16,384` tokens, `topk=6`, and therefore `98,304` routes.  The
same activation row is consequently loaded and quantized six times.  Its
preshuffled payload stores are also poorly coalesced: a wave writes eight
16-byte segments separated by 256 bytes.

The replacement changes the work decomposition without changing the result:

1. `moe_quant_token_*` quantizes each of the `16,384` source tokens once.  This
   reduces quantization work and logical BF16 input reads to one sixth of the
   route-per-wave implementation.
2. `moe_invert_route_rows_tk6` converts the existing
   `route -> grouped_row` mapping into `grouped_row -> token`.  It uses the
   dynamic route result and makes no balanced-routing assumption.
3. `moe_scatter_preshuffled_a_*_lds` assigns one workgroup to 32 adjacent
   grouped rows.  It reads each compact token payload as contiguous 128-byte
   chunks, stages a 32-row tile through LDS with a 33-dword pitch, then emits
   two contiguous 256-byte A segments per wave.  ScaleA is emitted as a
   contiguous 128-byte row-tile store.

The two additional launches cost only a few microseconds.  Eliminating sixfold
quantization and replacing scattered global stores with LDS-transposed,
coalesced stores saves substantially more time.

Artifacts:

```text
my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-d01-3_20260910T153812Z
```

### Previous baseline-compatible result

The two-round run on `heliosr-1b114-a07-3` completed at
`20260910T122422Z`:

| case | GEMM1 samples (us) | GEMM1 median us | GEMM1 vs 93665e | MOE e2e samples (us) | MOE e2e median us | MOE e2e vs 93665e | random pass | hash |
|---|---|---:|---:|---|---:|---:|:---:|---|
| baseline_93665e | 713.628, 705.837 | 709.733 | +0.00% | 1643.83, 1629.68 | 1636.76 | +0.00% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc8 | 642.171, 643.124 | 642.648 | +9.45% | 1574.17, 1581.34 | 1577.76 | +3.60% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg2_fc12 | 637.062, 650.056 | 643.559 | +9.32% | 1564.85, 1586.58 | 1575.71 | +3.73% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28 | 635.348, 633.735 | 634.542 | +10.59% | 1562.77, 1566.71 | 1564.74 | +4.40% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28_hard | 600.207, 614.597 | 607.402 | +14.42% | 1527.25, 1551.06 | 1539.15 | +5.96% | True | `91f3c3c87e7e17e854bcc5c3dbb7032f0f5a039a033a205a1cba04206799b5ca` |
| sync_mg4_fc28_relu | 586.739, 588.063 | 587.401 | +17.24% | 1516.62, 1515.64 | 1516.13 | +7.37% | True | `ca4a57024a6c0ee78991a0a2dcfd227852945fd356657fe64df596f62e98bdd4` |

Current conclusions:

1. `sync_mg4_fc28` is the best exact-SiLU version: GEMM1 improves by
   `10.59%` and MOE end-to-end improves by `4.40%`, with the same output
   hash as `baseline_93665e`.
2. `sync_mg4_fc28_hard` improves GEMM1 by `14.42%` and MOE end-to-end by
   `5.96%`, but uses the approximate hard-SiLU path and changes the output
   hash.
3. `sync_mg4_fc28_relu` is the fastest measured version: GEMM1 improves by
   `17.24%` and MOE end-to-end by `7.37%`, with the largest activation
   semantic change.
4. Every version passed the random-input production accuracy gate.

Artifacts:

```text
my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260910T122422Z
```

### Historical results

The first d01-3 smoke run (`ROUNDS=1`) produced:

| Version | GEMM1 us | vs same-run baseline | Random verification |
|---|---:|---:|---|
| baseline | 554.430 | 0.00% | PASS, hash match |
| nocs_mg4_fc8 | 534.229 | +3.64% | PASS, hash match |
| nocs_mg2_fc12 | 536.190 | +3.29% | PASS, hash match |
| nocs_mg4_fc28 | 527.944 | +4.78% | PASS, hash match |

Artifacts are under:

```text
my_code/gemm1_cycle_105pct_20260909/runs/<host>_<UTC timestamp>/
```

## Experiment 1: 256x512 WG, eight waves, three buffers

The candidate kept each wave's 128x128 output tile unchanged and enlarged the
WG to 256x512 with a 2x4 wave grid.  Two waves per SIMD were intended to hide
LDS/TDM and Trans32 latency.  A4W4 correctness passed exactly on a07-3, but the
20-iteration result was catastrophically slower:

```text
gemm1 = 14,925.218 us
logits_diff = 0
rel_l2 = 0
```

The candidate is rejected.  It is kept only as an opt-in geometry while the
experiment is being diagnosed and is not selected by the requested command.

## Experiment 2: native v_tanh_f32 SiLU identity

The candidate replaced `sigmoid(x) = 1 / (1 + exp(-x))` with
`0.5 * (tanh(0.5*x) + 1)` to remove one Trans32 instruction per output.
Random-input verification passed at the existing tolerance, but the a07-3
const0 benchmark regressed:

```text
baseline gemm1 = 585.504 us
native tanh    = 610.061 us
logits_diff    = 0
```

The native-tanh candidate is rejected and has been removed from the kernel.

## Experiment 3: two LDS buffers, epilogue batch four

This candidate reduced the ring from four 68 KiB stages to two and reduced the
epilogue batch width from eight to four.  Exact const0 correctness passed, but
the a07-3 20-iteration result regressed substantially:

```text
gemm1 = 786.917 us
logits_diff = 0
```

The two-buffer K256 pipeline does not hide TDM latency sufficiently.  It is not
a default-path candidate.

## Experiment 4: one output TDM per wave

The candidate replaced the workgroup-wide output TDM with four independent
per-wave stores to remove the final WG barrier and reduce each descriptor's
payload.  Random-input validation produced NaNs and failed, so the descriptor
mapping/protocol is not valid as implemented.  The change was removed before
any performance result was accepted.

## Experiment 5: skip steady-state cluster wrap synchronization

The initial cluster synchronization is retained so peer TDM requests begin in
phase.  The six steady-state ring-wrap cluster barriers are skipped while the
existing WG READY/REUSE fences and TDM `early_timeout` behavior remain active.

Random-input verification passed.  The first a07-3 const0 benchmark improved
from the same-session baseline `585.504 us` to:

```text
gemm1 = 566.131 us
logits_diff = 0
```

The paired ATT capture confirmed that this path is a real cycle reduction:

```text
baseline max cycles = 1,149,739
NOCS max cycles     = 1,070,624
cycle reduction     =    79,115 (6.88%)

NOCS ATT wall time  = 536.600 us
NOCS mean GFXCLK    = 1997.260 MHz
```

The steady-state cluster synchronization had therefore been a measurable
serialization cost, not merely wall-time noise.  The initial cluster sync is
still required and remains enabled.

## Experiment 6: reshape the compiler schedule groups

The NOCS trace still spent about 2,616 cycles per active wave stalled at
`s_wait_dscnt`.  The existing schedule divided each k128 body into groups of
four WMMA instructions and retained eight WMMA instructions at the end to cover
the next-stage LDS fence.  `AITER_FLYDSL_GEMM1_MMA_GROUP` and
`AITER_FLYDSL_GEMM1_FENCE_COVER_MMA` were added to sweep this split without
changing numerical semantics.

Representative 20-iteration const-zero results on a07-3 are:

| MMA group | fence-cover WMMA | GEMM1 us |
|---:|---:|---:|
| 2 | 4 | 545.454 |
| 2 | 8 | 539.742 |
| 2 | 12 | 536.516 |
| 2 | 16 | 534.817 |
| 2 | 20 | 535.618 |
| 2 | 24 | 531.355 |
| 2 | 26 | 541.534 |
| 2 | 28 | 530.560 |
| 2 | 30 | 533.430 |
| 1 | 28 | 537.336 |
| 4 | 20 | 543.354 |
| 4 | 24 | 528.169 (one run; another run was 565.383) |
| 4 | 28 | 528.409 |
| 8 | 24 | 541.066 |

Dynamic clock variation makes the sub-microsecond ordering of the best wall-time
points unreliable.  `mg=4, fc=28` was therefore captured with ATT.  The paired
cycle result is:

```text
NOCS mg4/fc28 max cycles = 1,052,184
ATT wall time             =   531.040 us
mean GFXCLK               =  1996.304 MHz

vs baseline: -97,555 cycles (-8.49%)
vs NOCS:     -18,440 cycles (-1.72%)
```

The random-input full-shape verification is not a zero-input-only result.  The
optimized and baseline kernels produced the same output hash:

```text
logits_diff   = 3.39799e-06
rel_l2        = 0.00260689
pass          = True
optimized SHA = aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
baseline SHA  = aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
```

The mg4/fc28 trace moves most of the core delay away from explicit DS waits,
but does not remove it entirely:

| Phase / stall (cycles per active wave) | NOCS mg4/fc8 | NOCS mg4/fc28 |
|---|---:|---:|
| prologue span | 4,750.750 | 4,154.111 |
| core span | 19,054.604 | 18,902.847 |
| epilogue span | 5,273.021 | 5,515.438 |
| core `s_wait_dscnt` stall | 2,616.188 | 609.035 |
| core `ds_load_b128` stall | 0.924 | 640.778 |
| core WMMA stall | 1,543.750 | 1,937.035 |
| core `s_wait_tensorcnt` stall | 75.021 | 207.632 |

Thus the more aggressive grouping successfully overlaps the explicit DS wait,
but part of that latency reappears on the dependent LDS loads and WMMA stream.
It is a useful incremental win rather than the factor-of-two change required by
the requested raw-WMMA target.

## Current limiting resource and next experiment

The mg4/fc28 code object reports:

```text
.vgpr_count                  = 804 per wave
.group_segment_fixed_size   = 278,528 B (272 KiB) per workgroup
ATT simultaneous slots/SIMD = 1
```

The MI400 Shader Programming Guide states that MI450 has 1,024 physical VGPRs
per SIMD and allocates Wave32 VGPRs in blocks of 16 (section 3.3.2.1, lines
2153-2171 in the local text extraction).  It also states that a workgroup may
allocate up to 320 KiB LDS and that LDS is shared by all four SIMDs in a WGP
(section 3.3.4, lines 2444-2452).  Therefore the current 804-VGPR wave and
272-KiB workgroup necessarily run with a single active wave slot per SIMD and a
single resident workgroup per WGP.

The next candidate enables the existing `w4x2` eight-wave geometry for the full
FP4 prefill fast path.  It keeps the 256x256 workgroup tile and 4-way cluster
multicast, but halves each wave's M accumulator tile from 128x128 to 64x128.
The intended effect is to reduce VGPR usage below 512 so two waves from the same
workgroup can reside on each SIMD and overlap the long prologue, LDS, and
epilogue latencies.  This experiment must pass the same random-input hash check
before any performance result is accepted.

## 2026-09-10 continuation: static geometry and activation experiments

The balanced target has fixed geometry: `N=6144`, 96 experts, four 256-row
M-tiles per expert, and six N-clusters per expert.  An opt-in static path now
removes the generic runtime tail-group division and uses expert-major task
ordering.  Random verification is bit-identical to the exact historical path:

```text
logits_diff = 3.39799e-06
rel_l2      = 0.00260689
SHA256      = aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
```

A paired a07-3 ATT capture before the subsequent machine reset measured:

```text
nocs mg4/fc28          = 1,104,372 cycles
static expert-major    = 1,082,411 cycles
reduction              =    21,961 cycles (1.99%)
```

The static path reduced the mean per-wave prologue from `4,774.847` to
`4,505.125 cycles` and the mean epilogue from `5,611.236` to `5,452.236
cycles`.  It removed 165 dynamic prologue instructions per wave while leaving
the WMMA count unchanged.

After an SSH/system reset, a fresh one-round six-version comparison produced:

| Case | GEMM1 us | vs baseline | Random correctness |
|---|---:|---:|---|
| baseline | 600.873 | +0.00% | PASS |
| nocs mg4/fc8 | 573.193 | +4.61% | PASS |
| nocs mg2/fc12 | 574.307 | +4.42% | PASS |
| nocs mg4/fc28 | 565.196 | +5.94% | PASS |
| static expert-major | 559.921 | +6.82% | PASS, exact hash |
| static expert-major + hard SiLU | 554.935 | +7.65% | PASS |

The hard-SiLU experiment replaces sigmoid with a clipped linear approximation.
It is not bit-identical, but passes the test's production correctness gate:

```text
logits_diff = 3.26392e-05  (< 0.01)
rel_l2      = 0.00807999
```

Its paired exact-SiLU ATT capture was interrupted by a later SSH reset, so the
wall-time signal is not yet accepted as a cycle result.  Under the resulting
slow machine state, hard-SiLU alone measured `1,636,845 cycles`.  Splitting the
next-k128 LDS register prefetch around the FRONT/BACK WMMA groups measured
`1,641,564 cycles`, a `0.29%` regression, and was removed.

Additional rejected experiments in this continuation:

- normal `w4x2` eight-wave geometry: random-correct, `620.067 us`;
- `tile_k=128`, four buffers: random-correct, `647.463 us`;
- `tile_n=128`, three buffers: incorrect, `logits_diff=0.0810313`;
- per-wave 128x64 output TDM: NaN/Inf output beginning near row 131;
- one-TRANS rational SiLU: random-correct but `613.801 us` at batch eight;
- degree-five polynomial SiLU: random-correct but `564.816 us`;
- delayed accumulator zeroing: paired ATT `1,086,574 cycles`, worse than
  static expert-major `1,082,411 cycles`.

## User-run two-round wall-time comparison (2026-09-10 06:49 UTC)

The user ran the focused reproduction set on a07-3 with two rounds in opposite
orders.  The complete run directory was copied back locally with matching
SHA256 checksums for `summary.md`, `bench.tsv`, and `verify.tsv`:

```text
my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260910T064949Z/
```

| Case | GEMM1 samples (us) | Median (us) | vs baseline | Random verification |
|---|---:|---:|---:|---|
| baseline | 572.600, 577.623 | 575.111 | +0.00% | PASS |
| nocs mg4/fc28 | 544.058, 542.872 | 543.465 | +5.50% | PASS, exact hash |
| static expert-major | 522.423, 515.309 | 518.866 | +9.78% | PASS, exact hash |
| static expert-major + hard SiLU | 513.489, 517.592 | 515.541 | +10.36% | PASS, approximate |
| static expert-major + ReLU gate | 491.186, 488.842 | 490.014 | +14.80% | PASS, approximate |

Incremental GEMM1 improvements from the medians are:

```text
static exact vs nocs mg4/fc28 = 4.53%
hard SiLU vs static exact     = 0.64%
ReLU gate vs static exact     = 5.56%
ReLU gate vs hard SiLU        = 4.95%
```

The exact paths retain the reference output hash:

```text
aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
```

Approximate activation results are:

| Activation path | logits_diff | rel_l2 | Gate | Output SHA256 |
|---|---:|---:|:---:|---|
| hard SiLU | 3.2639e-05 | 8.0800e-03 | PASS | `91f3c3c87e7e17e854bcc5c3dbb7032f0f5a039a033a205a1cba04206799b5ca` |
| ReLU gate | 6.4241e-05 | 1.1336e-02 | PASS | `ca4a57024a6c0ee78991a0a2dcfd227852945fd356657fe64df596f62e98bdd4` |

The production gate is `logits_diff < 0.01`, so both approximations pass this
test by a wide margin.  They are nevertheless semantic approximations rather
than bit-identical SiLU and must remain clearly distinguished from the exact
static version.

Current wall-time conclusions:

1. `static_swz1` is the best exact version and is the safe default candidate.
2. `static_swz1_hard` gives a small additional GEMM1 improvement while keeping
   a relatively close SiLU approximation.
3. `static_swz1_relu` is the fastest measured candidate (`490.014 us` median,
   `488.842 us` best sample), but has the largest semantic deviation and should
   only be selected when the production logits-diff gate is the accepted
   correctness contract.
4. These are wall-time results only.  Per the user's request, no new ATT capture
   was collected for this comparison.
