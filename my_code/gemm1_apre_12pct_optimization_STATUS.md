# gfx1250 MoE GEMM1 A-preshuffle optimization

## Scope and acceptance criteria

This follow-up starts from `sync_mg4_fc28_apre` at commit
`2df2cf70a82fc72cb233356c3bc4938acf5fc9f5` and targets the E96/T16384/K7168
stage-1 A4W4 MoE GEMM on a07-3.

The optimization is restricted to GEMM1. It must preserve:

- the existing GEMM1 call interface and exact-SiLU numerical behavior;
- random-input correctness and the baseline output hash;
- support for non-balanced token distributions;
- the runtime binary search over `m_tile_map` for every expert M boundary;
- the interfaces and behavior of every other MoE kernel.

The target is at least a 12% GEMM1 latency reduction relative to the active
post-reboot `sync_mg4_fc28_apre` median below. The current acceptance threshold
is therefore `552.924 * 0.88 = 486.573 us` on a07-3. The older pre-reboot result
is retained only as historical context.

## Active post-reboot a07-3 starting point

The machine rebooted before this run, changing the performance state. All three
starting cases were therefore measured again from the restored `629acef` source
semantics. The three task source files on a07-3 matched the local SHA256 values
before the run, and GPU/KFD was idle before and after it.

Run directory:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T023748Z
```

Command:

```bash
CASE_LIST=baseline_93665e,sync_mg4_fc28,sync_mg4_fc28_apre \
ROUNDS=5 RUN_VERIFY=1 RUN_ATT=0 bash my_code/reproduce_compare.sh
```

| case | GEMM1 samples (us) | GEMM1 median us | GEMM1 vs 93665e | MOE e2e samples (us) | MOE e2e median us | MOE e2e vs 93665e | random pass | hash |
|---|---|---:|---:|---|---:|---:|:---:|---|
| baseline_93665e | 719.743, 703.510, 704.567, 698.765, 705.656 | 704.567 | +0.00% | 1651.71, 1642.01, 1646.63, 1631.05, 1647.35 | 1646.63 | +0.00% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28 | 631.459, 639.485, 629.824, 672.968, 639.682 | 639.485 | +9.24% | 1566.62, 1576.66, 1561.91, 1603.25, 1575.40 | 1575.40 | +4.33% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28_apre | 549.184, 552.924, 558.746, 553.397, 550.779 | 552.924 | +21.52% | 1388.31, 1393.47, 1391.53, 1392.34, 1389.50 | 1391.53 | +15.49% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |

The active 12% GEMM1 target is `486.573 us`.

## Pre-reboot a07-3 starting point (superseded)

Run directory:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260910T163535Z
```

Command:

```bash
CASE_LIST=baseline_93665e,sync_mg4_fc28,sync_mg4_fc28_apre \
ROUNDS=5 RUN_VERIFY=1 RUN_ATT=0 bash my_code/reproduce_compare.sh
```

| case | GEMM1 samples (us) | GEMM1 median us | GEMM1 vs 93665e | MOE e2e samples (us) | MOE e2e median us | MOE e2e vs 93665e | random pass | hash |
|---|---|---:|---:|---|---:|---:|:---:|---|
| baseline_93665e | 790.891, 798.579, 801.832, 794.301, 796.247 | 796.247 | +0.00% | 1811.06, 1815.16, 1817.51, 1808.37, 1812.97 | 1812.97 | +0.00% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28 | 712.224, 705.248, 702.319, 694.790, 711.259 | 705.248 | +11.43% | 1732.76, 1712.02, 1708.75, 1704.21, 1723.87 | 1712.02 | +5.57% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28_apre | 608.750, 607.508, 603.726, 606.779, 609.037 | 607.508 | +23.70% | 1525.99, 1524.33, 1518.43, 1523.19, 1526.04 | 1524.33 | +15.92% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |

All three random verification runs passed with identical output hashes. GPU/KFD
was idle before and after the run.

## Hardware basis

The local MI455X/CDNA5 documentation identifies the relevant optimization
resources: native Wave32 execution, asynchronous descriptor-driven TDM transfers
between global memory and LDS, WGP multicast, increased LDS bandwidth, and the
`V_WMMA_SCALE_F32_32X16X128_F4` instruction. Candidate changes below are only
accepted when supported by cycle/ISA evidence and same-machine measurements.

## Candidate results

### Post-reboot candidates

| candidate | GEMM1 samples (us) | median (us) | vs active start | random/hash | status |
|---|---|---:|---:|---|---|
| `xdl0_reuse_wpt2` | 539.812, 537.445, 538.443 | 538.443 | +2.62% | pass, baseline hash | Exact, but `wpt2` currently also changes GEMM2 and the gain is far below 12%. |
| `output_store_w2` | 557.825, 555.511, 551.226 | 555.511 | -0.47% | pass, baseline hash | Regression; two larger output TDM descriptors did not reduce the final drain. |
| `xdl0_reuse_wpt2_eb4` | 539.715, 539.857, 539.663 | 539.715 | +2.39% | pass, baseline hash | Batch 4 is slower than the existing exact-SiLU batch 8 path. |
| `xdl0_reuse_wpt2_delayzero` | 538.645, 537.169, 538.691 | 538.645 | +2.58% | pass, baseline hash | Delaying accumulator zeroing is within noise and does not beat `xdl0_reuse_wpt2`. |
| `xdl0_reuse_wpt2_fc24` | 540.845, 539.928, 540.063 | 539.928 | +2.35% | pass, baseline hash | A wider TDM/DS scheduling window is slower than `fc28`. |
| `b2_midcarry` | 913.484, 918.426, 921.640 | 918.426 | -66.11% | pass, baseline hash | Two-buffer residency cannot compensate for the loss of pipeline depth. |
| `exactopt_bisect7` | 538.846, 541.846, 544.233 | 541.846 | +2.00% | pass, baseline hash | Seven iterations are logically sufficient for E96, but the generated schedule is slower than the original eight-step loop. |
| `exactopt_ostore2p` | 533.634, 533.058, 532.563 | 533.058 | +3.59% | pass, baseline hash | Overlaps the first half-output TDM with the second half of exact-SiLU epilogue work; current best. |

The best combination was then isolated to GEMM1 with a dedicated
`AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM` override and added to
`reproduce_compare.sh` as `sync_mg4_fc28_apre_exactopt`. GEMM2 retained its
original `wpt1`, XDL arbitration, and WMMA reuse settings.

Run directory:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T032118Z
```

| case | GEMM1 samples (us) | GEMM1 median us | MOE e2e samples (us) | MOE e2e median us | random/hash |
|---|---|---:|---|---:|---|
| `sync_mg4_fc28_apre` | 555.941, 553.397, 551.624, 554.459, 552.794 | 553.397 | 1386.71, 1385.78, 1383.65, 1387.31, 1388.44 | 1386.71 | pass, baseline hash |
| `sync_mg4_fc28_apre_exactopt` | 538.629, 537.354, 538.319, 538.237, 537.376 | 538.237 | 1370.87, 1372.13, 1368.61, 1372.48, 1370.42 | 1370.87 | pass, baseline hash |

This first isolated exactopt measurement improved GEMM1 by `2.74%` and MOE e2e
by `1.14%`. A later revision overlaps the first half of the output TDM store
with the second half of the exact-SiLU epilogue; its final same-run result is
recorded below.

Non-balanced validation used random gating with
`AITER_MOE_EXPERT_BALANCE=false` at the full E96/T16384/topk6/K7168 shape. It
passed with `logits_diff=3.4878e-06` and `rel_l2=2.6411e-03`. The optimized
launcher still executes the original fixed-step binary search over
`m_tile_map`; no balanced-row shortcut or static expert geometry was added.

`xdl0_reuse_wpt2` run directory:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T024359Z_postreboot_xdl0_reuse_wpt2
```

Its MOE e2e samples were `1391.69, 1393.48, 1390.98 us` with a median of
`1391.69 us`. This comparison is diagnostic only because the global `wpt2`
override also changed GEMM2.

### Pre-reboot exact candidates

These measurements use the superseded `607.508 us` start and are retained to
avoid repeating low-value searches after the reboot.

| candidate | GEMM1 samples (us) | median (us) | vs 607.508 us | result |
|---|---|---:|---:|---|
| WMMA A+B reuse | 599.139, 604.432 | 601.785 | +0.94% | Correct; small gain. |
| WMMA A-only reuse | 601.267, 604.261 | 602.764 | +0.78% | Correct; small gain. |
| WMMA B-only reuse | 603.306, 600.329 | 601.817 | +0.94% | Correct; small gain. |
| `DISABLE_XDL_ARB_STALL=0` | 596.568, 595.240 | 595.904 | +1.91% | Correct; small gain. |
| `wpt2` | 592.474, 596.826 | 594.650 | +2.12% | Correct, but the experimental override also changed GEMM2. |
| xdl0 + reuse + wpt2 | 644.886, 589.211, 586.993 | 589.211 | +3.01% | Correct; first sample was anomalous and total gain remained far below 12%. |
| `cluster_m=1` | 602.708, 604.220 | 603.464 | +0.67% | Correct; extra B traffic offsets most barrier savings. |
| `cluster_m=2` | 597.682, 599.923 | 598.803 | +1.43% | Correct; extra B traffic offsets most barrier savings. |
| lazy TDM descriptor construction | 605.371, 599.697 | 602.534 | +0.82% | Correct; small gain. |
| transitive cluster sync | 605.624, 615.255, 607.035 | 607.035 | +0.08% | Correct; no useful gain. |
| scalar `m_tile_map` view | 606.959, 606.967 | 606.963 | +0.09% | Correct; no useful gain. |
| `post-misched=0` | 612.742, 610.604 | 611.673 | -0.69% | Regression. |
| `early_timeout=0` | 622.455, 628.694 | 625.575 | -2.97% | Regression. |
| dynamic M-major swizzle | 621.197, 618.483 | 619.840 | -2.03% | Regression. |
| expert scheduling mode off | 713.710, 711.420 | 712.565 | -17.29% | Large regression. |
| `tile_k=512,b2` | 678.417, 676.779 | 677.598 | -11.54% | Regression. |
| `tile_n=128,b4` | 775.316, 775.173 | 775.245 | -27.61% | Regression. |
| direct global scales | 1077.765, 1079.352 | 1078.559 | -77.54% | Correct but unusably slow. |

### Structural investigations

- ATT on the pre-reboot baseline reported `.vgpr_count = 804`, `278,528 B`
  LDS, one active wave slot per SIMD, and one WG per WGP. A complete active wave
  spent about `6,237` cycles in the prologue, `23,932` cycles in the WMMA core,
  and `5,008` cycles in the epilogue.
- The main measured stalls were about `3,522 cycles/wave` in
  `s_barrier_wait`, `1,945` in WMMA, `1,935` in `ds_load_b128`, and `947` in
  `s_wait_dscnt`. The final output TDM wait contributed about `1,699 cycles/wave`.
- `tile_m=128` originally produced NaN/Inf in the latter half of every tile.
  The cause was overlapping short TDM transfers: A and ScaleA required guard
  space, and the output store required fewer participating waves. With those
  corrections and four pipeline buffers it became exact, but GEMM1 regressed to
  `977.788, 982.005 us` because it doubled the number of workgroups without
  reducing effective LDS residency.
- Three-buffer next-stage carry produced a repeatable `logits_diff=0.148409` on
  both `tile_m=128` and `tile_m=256`. Disabling carry restored exactness, but the
  resulting `tile_m=256,b3` GEMM1 was `774.353, 774.720 us`.
- A correct double-buffer carry experiment passed random verification but ran at
  roughly `1.22 ms`; the two-buffer pipeline could not overlap enough TDM work
  to benefit from its lower LDS allocation.
- `tile_m=128` variants without the TDM guard corrections, `wpt4`, and several
  alternate wave mappings produced NaN. Two split-phase cluster-barrier
  experiments deadlocked and were abandoned after restoring an idle GPU.

No candidate reaches either the initial post-reboot `486.573 us` threshold or
the later same-run paired threshold reported below.

## Final retained candidate

The retained `sync_mg4_fc28_apre_exactopt` case combines:

- GEMM1-only `wpt2` TDM ownership;
- `DISABLE_XDL_ARB_STALL=0`;
- exact WMMA A/B operand reuse;
- two-phase output TDM stores, with the first half overlapped with the second
  half of the exact-SiLU epilogue.

Run directory:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T041850Z
```

| case | GEMM1 samples (us) | median (us) | MOE e2e samples (us) | median (us) | random/hash |
|---|---|---:|---|---:|---|
| `sync_mg4_fc28_apre` | 564.321, 566.851, 570.492, 552.020, 557.066 | 564.321 | 1402.32, 1403.51, 1409.57, 1382.08, 1396.28 | 1402.32 | pass, baseline hash |
| `sync_mg4_fc28_apre_exactopt` | 546.078, 542.391, 552.638, 547.004, 534.109 | 546.078 | 1383.52, 1379.51, 1389.87, 1384.42, 1372.06 | 1383.52 | pass, baseline hash |

The final same-run improvements are `3.23%` for GEMM1 and `1.34%` for MOE e2e.
Because the post-reboot machine state drifted by roughly 2% between measurement
windows, this paired run is the authoritative relative comparison. Its same-run
12% threshold is `564.321 * 0.88 = 496.603 us`; the retained kernel remains
`49.475 us` above it.

## Post-reboot ATT before output-store overlap

ATT directory:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T034924Z/att/heliosr-1b114-a07-3_sync_mg4_fc28_apre_exactopt
```

The captured xdl0/reuse/wpt2 dispatch had a maximum span of `1,269,223` shader cycles
at an ATT-derived mean GFXCLK of about `2.087 GHz`. Every physical SIMD still
had one simultaneously active slot. All 256 observed physical SIMD keys ran 45
sequential wave lifetimes, and the maximum cross-WGP completion imbalance was
only `0.0108%`, so tail imbalance is not the remaining bottleneck.

For 144 complete active waves, the mean phase split was:

| phase | cycles/wave |
|---|---:|
| prologue | 6,501 |
| WMMA core | 22,372 |
| epilogue | 5,459 |
| total | 34,333 |

The largest explicit waits were:

| wait | cycles/wave | interpretation |
|---|---:|---|
| final `s_wait_tensorcnt 0x0` | 2,059 | output TDM drain immediately before `s_endpgm` |
| workgroup `s_barrier_wait 0xffff` | 2,464 | startup, pipeline reuse, and epilogue synchronization |
| cluster `s_barrier_wait 0xfffd` | 1,129 | required cross-WG ring-buffer synchronization |
| `s_wait_dscnt` | 347 | remaining LDS dependency waits |

Reaching `486.573 us` from the current exact candidate requires roughly another
`9.6%`. That is comparable to eliminating nearly all of the final output-TDM
drain plus cluster-barrier wait, both of which are correctness-critical. This is
why the remaining work requires a pipeline redesign rather than another launch
or scheduling knob.

## Follow-up output pipeline experiments

A four-phase output-store variant was tested after the two-phase implementation.
It remained exact, including the baseline random-output hash, but was slower:

| output pipeline | GEMM1 samples (us) | median (us) | MOE e2e samples (us) | median (us) | random/hash |
|---|---|---:|---|---:|---|
| four-phase | 535.680, 536.162, 536.357 | 536.162 | 1374.15, 1373.60, 1370.01 | 1373.60 | pass, baseline hash |
| restored two-phase | 533.664, 532.964, 534.090 | 533.664 | 1371.43, 1370.84, 1376.17 | 1371.43 | pass, baseline hash |

The four-phase split regressed GEMM1 by `0.47%`, so it was removed and the
two-phase implementation remains retained. Raw run directories:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T051344Z
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T052027Z
```

These one-case invocations completed verification and all benchmark rounds, but
the final summary helper exited with `StatisticsError` because
`baseline_93665e` was intentionally absent from `CASE_LIST`; the values above
come directly from the generated logs and `bench.tsv`.

## ATT after two-phase output-store overlap

ATT directory:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/a07-3_20260911T0526_exactopt_att/a07-3_exactopt_ostore2p
```

The retained two-phase kernel had a maximum dispatch span of `1,190,856` shader
cycles at an ATT-derived mean GFXCLK of about `2.072 GHz`. Occupancy remained
one simultaneously active slot per physical SIMD, with 45 sequential wave
lifetimes per slot and only `0.0121%` maximum WGP completion imbalance.

For 144 complete active SIMD3 waves, the mean phase split was:

| phase | cycles/wave |
|---|---:|
| prologue | 6,130 |
| WMMA core | 21,702 |
| epilogue | 3,499 |
| total | 31,331 |

The output overlap reduced the final `s_wait_tensorcnt 0x0` to only about
`26 cycles/wave`; output TDM drain is therefore no longer the main remaining
bottleneck. The largest waits are now:

| wait site | cycles/wave | interpretation |
|---|---:|---|
| initial LDS-ready `s_barrier_wait 0xffff` | 1,072 | imbalance between the two TDM owner groups after the first-stage transfers |
| steady cluster `s_barrier_wait 0xfffd` | 797 | required four-M-workgroup ring synchronization |
| steady LDS-ready `s_barrier_wait 0xffff` | 731 | imbalance between owner groups at stage reuse |
| second-half epilogue `s_barrier_wait 0xffff` | 360 | output producer/consumer synchronization |
| all `s_wait_dscnt` sites | 586 | LDS dependencies and descriptor progress |

This evidence motivates the next controlled experiment: cross the ScaleA and
ScaleB TDM owner groups so each group constructs one row-bounded and one
unbounded descriptor. The experiment does not alter tensor contents, the
runtime `m_tile_map` binary search, WMMA arithmetic, or any external interface.

## ATT-guided follow-up candidates

| candidate | GEMM1 samples (us) | median (us) | MOE e2e samples (us) | median (us) | random/hash | result |
|---|---|---:|---|---:|---|---|
| cross ScaleA/ScaleB owner groups | 537.954, 536.058, 535.032 | 536.058 | 1377.24, 1373.48, 1369.80 | 1373.48 | pass, baseline hash | `0.45%` slower than the adjacent retained two-phase median; removed. |
| column-major WMMA snake | 536.436 | 536.436 | 1374.70 | 1374.70 | pass, baseline hash | No predicted gain appeared; `0.52%` slower than the adjacent retained median, so the experiment was stopped after one round and removed. |

The column-major experiment was motivated by the SIMD3 ATT attribution: the
row-major kernel executes about 1,344 `matrix_a_reuse` WMMA instructions per
wave at `5.844` attributed cycles on average, versus 392 `matrix_b_reuse`
instructions at `1.778` attributed cycles. Reordering the independent C-tile
updates preserved each accumulator's K-order and produced the exact baseline
hash, but wall-clock performance did not follow the per-instruction attribution.
The ATT duration field therefore cannot be used as a direct throughput model for
this instruction reordering.

Run directories:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T054851Z
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T060020Z
```

### Three-buffer dual-residency investigation

`tile_m=128, num_buffers=3` reduces the statically allocated LDS arena from
`278,528 B` (`272 KiB`) to `156,672 B` (`153 KiB`) per workgroup, which is
small enough for two workgroups to fit in the documented `320 KiB` LDS capacity
of one WGP. This was the only remaining occupancy-oriented configuration with a
credible path to a double-resident kernel.

The previously observed three-buffer corruption was isolated to the unrolled
drain carry from LDS buffer 2 back to buffer 0. Four-buffer drains carry through
buffers 1/2/3 and never obscure the literal base-0 pointer. Preserving the
literal LDS base for the three-buffer wrap restored exactness:

- `logits_diff=3.39799e-06`
- `rel_l2=2.60689e-03`
- output hash matched the baseline

However, its first valid performance sample was:

| candidate | GEMM1 (us) | MOE e2e (us) | result |
|---|---:|---:|---|
| `tile_m=128,b3`, corrected carry wrap | 683.349 | 1515.95 | `28.0%` slower than the adjacent retained `533.664 us`; removed. |

Run directories for the failed strict-wait diagnosis and the corrected carry:

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T062451Z
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260911T063423Z
```

The strict `tensorcnt=0` experiment retained the same incorrect
`logits_diff=0.148409`, proving that transfer completion order was not the
cause. The address correction fixed the function, but the extra workgroups and
shallower three-stage pipeline outweighed the potential second-WG residency.
Both experimental changes were removed from the retained source.
