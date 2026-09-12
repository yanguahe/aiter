# Persistent-overlap output-pad8 results

Only the retained `persistent_overlap_output_pad8.s` kernel remains in the
directory.  Rejected XDL-arbitration, two-address-store, WPT1, WPT2, and copied
reference ISA files were removed after the comparisons below were completed.

## d01-3: `persistent_overlap_xdl0`

Date: 2026-09-12

Candidate SHA256:

```text
90816d1261b703e49f9fb362b90db54709f4c2d7fb1381c6c697d42b632dc358
```

Random MoE e2e, one alternating-order round:

| case | GEMM1 | fused MoE | correctness |
|---|---:|---:|---|
| `persistent_overlap` | `666.970 us` | `2100.57 us` | `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03`, hash matched |
| `persistent_overlap_xdl0` | `666.672 us` | `2094.26 us` | `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03`, hash matched |

Const0 MoE e2e, three alternating-order rounds:

| case | samples | median | fused MoE median | result |
|---|---|---:|---:|---|
| `persistent_overlap` | `543.989, 541.318, 543.621 us` | `543.621 us` | `1782.93 us` | reference |
| `persistent_overlap_xdl0` | `543.395, 547.213, 548.339 us` | `547.213 us` | `1785.50 us` | `0.66%` slower |

Conclusion: leave `SCHED_MODE.bit[2]=1` in the retained hand-written assembly.
The FlyDSL result for `xdl0` does not transfer to this instruction schedule.

Raw result directories on d01-3:

```text
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T143921Z_e2e-random
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T144037Z_e2e-const0
```

## d01-3: retained `persistent_overlap_output_pad8`

Candidate SHA256:

```text
039ed787b1f136ee402b76e3bd0b7c9bf439148a0156d0fcca856ed6ab25ccad
```

This candidate changes only the activated-output LDS layout:

```text
row pitch       128 B -> 144 B
half size       0x2000 -> 0x2400
TDM tile_dim0   128 B -> 144 B
global bound    128 B (unchanged)
```

The final 16 bytes of every LDS row are outside the 128-byte global bound and
are discarded by output TDM OOB handling.  All eight double-buffered output
regions are disjoint; the maximum LDS end is `0x4c800`, below `0x50000`.

Random MoE e2e, three alternating-order rounds:

| case | GEMM1 samples | median | fused MoE median | correctness |
|---|---|---:|---:|---|
| `persistent_overlap` | `673.716, 665.139, 671.912 us` | `671.912 us` | `2095.48 us` | `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03`, hash matched |
| `persistent_overlap_output_pad8` | `657.741, 663.908, 662.399 us` | `662.399 us` | `2088.09 us` | `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03`, hash matched |

Random GEMM1 improved by `1.42%`.  Every round produced:

```text
output_sha256=aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
```

Const0 MoE e2e, five alternating-order rounds:

| case | GEMM1 samples | median | fused MoE median | result |
|---|---|---:|---:|---|
| `persistent_overlap` | `544.575, 540.821, 543.749, 544.576, 539.414 us` | `543.749 us` | `1785.69 us` | reference |
| `persistent_overlap_output_pad8` | `538.236, 535.033, 537.259, 535.153, 539.176 us` | `537.259 us` | `1777.31 us` | `1.19%` faster GEMM1 |

### ATT evidence

Three single-SIMD-select ATT pairs were collected in both test orders.  Full
kernel span is noisy, so the retained comparison uses the median of three
captures and separately checks the directly affected output waits.

| capture | first case | baseline cycles | pad8 cycles |
|---|---|---:|---:|
| `20260912T154121Z` | baseline | `1,138,614` | `1,152,616` |
| `20260912T154441Z` | pad8 | `1,142,702` | `1,107,196` |
| `20260912T154643Z` | baseline | `1,111,173` | `1,113,445` |
| median | mixed | `1,138,614` | `1,113,445` |

The median full-kernel span fell by `25,169 cycles`, or `2.21%`.

The first two output `s_wait_dscnt 0` instructions are much more stable and
show the intended mechanism directly:

| capture | baseline stall cycles | pad8 stall cycles | reduction |
|---|---:|---:|---:|
| `20260912T154121Z` | `81,855` | `27,509` | `66.39%` |
| `20260912T154441Z` | `80,491` | `27,422` | `65.93%` |
| `20260912T154643Z` | `80,932` | `25,985` | `67.89%` |
| median | `80,932` | `27,422` | `66.12%` |

This confirms that the 144-byte row pitch removes most of the output LDS drain
stall.  Total span still varies with input `s_wait_tensorcnt`, cluster barrier,
and WMMA idle time, which is why a single ATT capture is insufficient.

After adding the stable case name to `benchmark_history.sh`, a final named-case
random e2e smoke test also passed:

| case | GEMM1 | fused MoE | correctness |
|---|---:|---:|---|
| `persistent_overlap` | `667.391 us` | `2090.42 us` | passed, reference hash |
| `persistent_overlap_pad8` | `661.628 us` | `2090.45 us` | passed, matching hash |

## DS two-address variants

`DS_STORE_2ADDR_B64` uses offsets in units of 8 bytes.  The correct pairs are:

```text
offset1:2
offset0:4 offset1:6
```

Using byte-like values `4/8/12` was incorrect and produced NaNs; all retained
generated files use the corrected encoding.

The compact-layout two-address candidate was random-correct but slightly slower:

```text
baseline  = 666.900 us
candidate = 667.851 us
regression = 0.14%
```

The combined pad8 plus two-address candidate was also correct:

```text
SHA256=7d4b0c5f0feb61ce61f704e40d6c18ca2a2730fa6aaec7f80621ae56171bb599
```

| case | GEMM1 samples | median | fused MoE median |
|---|---|---:|---:|
| `persistent_overlap` | `542.001, 544.303, 540.957, 544.181, 547.013 us` | `544.181 us` | `1782.62 us` |
| `output_pad8_2addr` | `537.445, 536.015, 539.817, 533.199, 538.296 us` | `537.445 us` | `1779.41 us` |

Its `1.24%` gain over that round's baseline is effectively the same as plain
pad8, while the assembly transformation is more complex.  It is therefore not
the retained benchmark winner.

## Full FlyDSL exactopt and WPT experiments

The d01-3 reference ISA came from repository HEAD:

```text
1f4370313e29ba0f00ed25648ee6eef6ad5dcb66
```

Reference hashes:

```text
WPT2 a895c9b97307629ffb34510671890dfc9207cd07a388fd959c8fd548698d8f4a
WPT1 f7abfe2891d73af3a190af8d437545512d065b9d5d441715e9c300ff8cec5ea1
```

All tested reference-derived kernels passed random MoE e2e correctness, but
none beat the hand-written persistent kernel:

| candidate | observed GEMM1 | comparison |
|---|---:|---|
| WPT2 exactopt, standard grid | `725.643 us` | large regression |
| WPT2 exactopt, persistent full drain | `726.269 us` | large regression |
| WPT1 exactopt, persistent full drain | `553.365 us` const0 | `2.24%` slower than same-round `541.249 us` baseline |
| WPT1 exactopt, persistent overlap | `705.561 us` random median | `5.98%` slower than same-round `665.764 us` baseline |

The WPT1 overlap result used three alternating rounds and kept the same random
hash and precision metrics.  Its regression rules out task-boundary drain as
the reason the reference schedule is slower.

## Final decision

Retain `persistent_overlap_output_pad8.s` as `persistent_overlap_pad8` in
`benchmark_history.sh`.  Keep the original `persistent_overlap.s` unchanged as
the same-machine baseline.  Do not port the following mechanisms into the
retained hand-written kernel:

- `DISABLE_XDL_ARB_STALL=0`;
- compact output `DS_STORE_2ADDR_B64` by itself;
- full WPT2 FlyDSL exactopt instruction schedule;
- WPT1 FlyDSL exactopt schedule.

The remaining `mg4/fc28` and reuse effects are already entangled with the slow
reference schedule, so no additional hand-written hotloop rewrite is justified
without a new instruction-level experiment.

Raw result directories on d01-3:

```text
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T150603Z_e2e-random
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T150717Z_e2e-const0
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T152548Z_e2e-random
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T152657Z_e2e-const0
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T153921Z_e2e-random
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T154121Z_att
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T154441Z_att
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T154643Z_att
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T154857Z_e2e-random
my_code/moe_gemm1_act1_optimized/history_runs/heliosr-1b114-d01-3_20260912T155931Z_e2e-random
```
