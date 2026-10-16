# E96/T16384/topk6 MoE optimization record

## Target workload

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072
```

All performance measurements below were collected on `a07-3` in the
`hyg_fyd1` container after confirming that the whole GPU was idle.

## Optimized-launcher baseline

The three-round `e2e-const0` baseline uses
`launch_gemm_a8w4_tdm_optimized` with the tuned
`t256x256x256/w2x2/b4` GEMM1 and GEMM2 configurations.

| Kernel | Tile / waves / buffers | Cluster `(m,n)` | Samples (us) | Median (us) |
|---|---|---:|---:|---:|
| GEMM1 | `t256x256x256/w2x2/b4` | `(4,4)` | 580.801, 596.227, 595.109 | 595.109 |
| GEMM2 | `t256x256x256/w2x2/b4` | `(1,4)` | 361.724, 359.727, 359.866 | 359.866 |
| Fused MoE | — | — | 1137.02, 1148.64, 1149.41 | 1148.64 |

Log: `/tmp/t256_baseline_e96_r3_new.log`

## GEMM2: t256 persistent N-task loop

The retained GEMM2 changes are restricted to the exact target shape and tuned
`t256x256x256/w2x2/b4` configuration. Its launch cluster is
`cluster_m=1, cluster_n=4`.

- One workgroup processes all seven N tasks belonging to the same M tile and
  local N position, amortizing task setup and expert lookup.
- The next task's stage 0 and stage 1 input TDMs overlap the current task's
  output TDM. Input buffers 0/1 and output buffers 2/3 occupy disjoint LDS
  regions during this overlap.
- The first A stage is cached once and reused by all seven persistent N tasks.
  This uses about 32 KiB of the otherwise free LDS tail and keeps total LDS use
  below the 320 KiB limit.
- `wmma_reuse=3` enables B-operand reuse for the t256 specialization.
- FlyDSL vector-SSA promotion does not preserve the three K128 rmem carries
  between the four constexpr-unrolled drain calls for this geometry. Drain
  tiles 1 through 3 therefore reload K128-0 from LDS. This is required for
  random-input correctness.

Three-round `e2e-const0` result:

| Kernel | Samples (us) | Median (us) | Change vs baseline |
|---|---:|---:|---:|
| GEMM2 persistent | 353.939, 347.309, 348.380 | 348.380 | +3.19% |
| Fused MoE | 1138.75, 1129.65, 1130.68 | 1130.68 | +1.56% |

Log: `/tmp/t256_g1_bth0_r3.log`

Random-input validation used the same shape with `e2e-random`:

```text
pass=True
logits_diff=3.39799e-06
rel_l2=0.00260689
GEMM2 reference/output hash128: cbce5dc196dde71ff8271c3ac69af990
```

Log: `/tmp/t256_g2_only_random.log`

## Rejected GEMM1 t256 persistent N-task loop

A six-task fused persistent loop was brought up for GEMM1 with
`t256x256x256/w2x2/b4`. The performance result below is the three-round
`e2e-const0` measurement with GEMM1's original B multicast geometry,
`cluster_m=4, cluster_n=4`. GEMM2 continued to use
`cluster_m=1, cluster_n=4`.

| Kernel | Tile / waves / buffers | Cluster `(m,n)` | Samples (us) | Median (us) | Change vs baseline |
|---|---|---:|---:|---:|---:|
| GEMM1 persistent | `t256x256x256/w2x2/b4` | `(4,4)` | 779.341, 782.472, 780.570 | 780.570 | -31.16% |
| GEMM2 persistent | `t256x256x256/w2x2/b4` | `(1,4)` | 357.073, 353.781, 348.257 | 353.781 | +1.69% |
| Fused MoE | — | — | 1327.94, 1327.82, 1320.76 | 1327.82 | -15.60% |

Log: `/tmp/t256_w2x2_g1p6_cm4_const_r3.log`

The `cluster_m=1, cluster_n=4` GEMM1 variant was also functionally correct on
random input, but was rejected during the initial run because it was even
slower than the optimized launcher:

| Data | GEMM1 persistent (us) | Optimized GEMM1 in the same run (us) | Result |
|---|---:|---:|---|
| random | 967.772 | 740.044 | rejected, 30.9% slower |

Correctness for the complete MoE pipeline remained within the existing gate:

```text
pass=True
logits_diff=3.39799e-06
rel_l2=0.00260689
```

Log: `/tmp/t256_both_persistent_random3.log`

Both cluster variants regressed, so the GEMM1 persistent-loop code was
removed.

## Rejected t256/w2x4 persistent experiment

The requested `t256x256x256/w2x4/b4` persistent geometry was evaluated with
both `cluster_m=4, cluster_n=4` and `cluster_m=1, cluster_n=4`.

The 4x4 cluster version was functionally correct, but a one-round random run
measured GEMM1 at `2320.246 us` and GEMM2 at `1613.999 us`. Synchronizing 16
workgroups at the K-stage cluster barriers dominated the multicast benefit.

The 1x4 cluster version was also correct. Its comparable three-round const0
result was:

| Kernel | Tile / waves / buffers | Cluster `(m,n)` | Samples (us) | Median (us) | Change vs baseline |
|---|---|---:|---:|---:|---:|
| GEMM1 persistent | `t256x256x256/w2x4/b4` | `(1,4)` | 640.052, 639.325, 640.658 | 640.052 | -7.55% |
| GEMM2 persistent | `t256x256x256/w2x4/b4` | `(1,4)` | 418.676, 417.728, 417.333 | 417.728 | -16.08% |
| Fused MoE | — | — | 1250.14, 1248.93, 1248.90 | 1248.93 | -8.73% |

Log: `/tmp/t256_w2x4_cm1_persist_const_r3.log`

Because both kernels regressed, all t256/w2x4 code and CSV changes were
removed and tuning returned to `t256x256x256/w2x2/b4`.
