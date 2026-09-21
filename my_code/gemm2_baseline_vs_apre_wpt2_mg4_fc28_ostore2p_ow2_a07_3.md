# a07-3 MoE baseline 与 GEMM1/GEMM2 apre 优化版性能对比

<!-- markdownlint-disable MD013 -->

## 1. 测试目的

本文比较以下两套完整 MoE 路径：

| 对比项 | GEMM1 路径 | GEMM2 路径 |
|---|---|---|
| baseline | `baseline_93665e` | `baseline` |
| optimized | `sync_mg4_fc28_apre_exactopt` | `apre_wpt2_mg4_fc28_ostore2p_ow2` |

重点比较：

- GEMM1 producer；
- GEMM1 kernel；
- GEMM2 producer；
- GEMM2 kernel；
- 完整 MoE e2e；
- GEMM1、GEMM2 的 executed TFLOP/s。

## 2. 测试环境与命令

测试环境：

```text
host      = heliosr-1b114-a07-3
container = hyg_fyd1
branch    = hyg_gfx1250_gemm_a4w4
HEAD      = e853468239014ae0643768357295847ce0f1132b
date      = 2026-09-21
```

测试前、两组性能测试之间以及测试结束后均通过
`/data/yanguahe/code/gpu_users.sh` 检查，a07-3 GPU 均为空闲状态，测试结束后没有残留 GPU 进程。

### 2.1 `e2e-const0`

```bash
AITER_REPRO_GEMM1_CASE=baseline_93665e \
CASE_LIST=baseline \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh e2e-const0 --gemm2

CASE_LIST=apre_wpt2_mg4_fc28_ostore2p_ow2 \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh e2e-const0 --gemm2
```

### 2.2 `e2e-random`

```bash
AITER_REPRO_GEMM1_CASE=baseline_93665e \
CASE_LIST=baseline \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh e2e-random --gemm2

CASE_LIST=apre_wpt2_mg4_fc28_ostore2p_ow2 \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh e2e-random --gemm2
```

公共 workload：

```text
data_format = a4w4
act         = silu
experts     = 96
tokens      = 16384
topk        = 6
model_dim   = 7168
inter_dim   = 3072
e2e iters   = 20
rounds      = 3
```

## 3. producer 统计口径

baseline 的两个 producer 各自是一个 fused kernel：

```text
GEMM1 producer:
  moe_fused_quant_preshuffle_routeks_fd7168_r8_fp4_pk8_srctk6_noKS

GEMM2 producer:
  moe_fused_quant_preshuffle_routeks_fd3072_r8_fp4_pk8_srcrow_noKS
```

optimized 的 GEMM1 producer 由三个 kernel 组成：

```text
moe_quant_token_fd7168_fp4_pk8_hidtdm7
moe_invert_route_rows_tk6
moe_scatter_preshuffled_a_fd7168_r32_lds_pe7_slds_skipempty
```

optimized 的 GEMM2 producer 是：

```text
moe_quant_preshuffled_a_fd3072_rpw2_pf2_direct_hidtdm6_otdmw2
```

文中的 optimized GEMM1 producer 样本是每一轮上述三个 kernel 的
`device_time_avg` 之和，再对三轮总和取中位数。Profiler 表格只保留到 `0.1 us`，
因此 producer 合计也按该精度记录。`moe_route`、`moe_contiguous_psum_remap`、
`moe_gather_reduce_*` 和初始化用的 `aten::fill_` 不计入 producer 合计，但均包含在
MoE e2e 时间内。

## 4. `e2e-const0` 对比

### 4.1 latency

| 部分 | baseline samples (us) | baseline median (us) | optimized samples (us) | optimized median (us) | 减少 (us) | latency 降幅 | speedup |
|---|---|---:|---|---:|---:|---:|---:|
| GEMM1 producer | `151.7, 149.6, 140.5` | 149.600 | `69.0, 66.8, 66.1` | 66.800 | 82.800 | 55.35% | 2.240x |
| GEMM1 | `692.006, 683.551, 669.393` | 683.551 | `523.360, 523.368, 523.516` | 523.368 | 160.183 | 23.43% | 1.306x |
| GEMM2 producer | `72.1, 72.2, 72.6` | 72.200 | `52.5, 53.1, 52.5` | 52.500 | 19.700 | 27.29% | 1.375x |
| GEMM2 | `446.879, 443.453, 443.115` | 443.453 | `335.889, 334.580, 334.696` | 334.696 | 108.757 | 24.53% | 1.325x |
| MoE e2e | `1630.83, 1614.20, 1590.27` | 1614.20 | `1247.39, 1241.76, 1240.39` | 1241.76 | 372.44 | 23.07% | 1.300x |

optimized GEMM1 producer 的三段明细如下：

| kernel | samples (us) | median (us) |
|---|---|---:|
| `moe_quant_token_fd7168_fp4_pk8_hidtdm7` | `22.6, 22.8, 22.6` | 22.6 |
| `moe_invert_route_rows_tk6` | `5.0, 1.9, 2.4` | 2.4 |
| `moe_scatter_preshuffled_a_fd7168_r32_lds_pe7_slds_skipempty` | `41.4, 42.1, 41.1` | 41.4 |
| 每轮三段合计 | `69.0, 66.8, 66.1` | 66.8 |

### 4.2 GEMM 算力与等效带宽

| kernel | baseline (TFLOP/s) | optimized (TFLOP/s) | 增加 (TFLOP/s) | TFLOP/s 增幅 | baseline effective R+W (TB/s) | optimized effective R+W (TB/s) | 增加 (TB/s) | TB/s 增幅 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GEMM1 | 12,667.2 | 16,544.1 | 3,876.9 | 30.61% | 4.290 | 5.603 | 1.313 | 30.61% |
| GEMM2 | 9,762.8 | 12,935.1 | 3,172.3 | 32.49% | 5.523 | 7.317 | 1.794 | 32.49% |

TFLOP/s 和 effective R+W TB/s 均使用与中位 latency 对应的轮次数据。执行 FLOP 数和
等效读写字节数分别为：

```text
GEMM1 = 8,658,654,068,736 FLOP
GEMM2 = 4,329,327,034,368 FLOP
GEMM1 = 3,224,371,200 B effective R+W
GEMM2 = 2,692,743,168 B effective R+W
```

### 4.3 正确性

| case | pass | logits_diff | rel_l2 | MoE output hash128 | ref output hash128 |
|---|:---:|---:|---:|---|---|
| baseline | True | 0 | 0 | `21291d9023c8af8a6324fe20f346a967` | `21291d9023c8af8a6324fe20f346a967` |
| optimized | True | 0 | 0 | `21291d9023c8af8a6324fe20f346a967` | `21291d9023c8af8a6324fe20f346a967` |

## 5. `e2e-random` 对比

### 5.1 latency

| 部分 | baseline samples (us) | baseline median (us) | optimized samples (us) | optimized median (us) | 减少 (us) | latency 降幅 | speedup |
|---|---|---:|---|---:|---:|---:|---:|
| GEMM1 producer | `153.4, 159.9, 150.8` | 153.400 | `70.7, 72.9, 69.4` | 70.700 | 82.700 | 53.91% | 2.170x |
| GEMM1 | `854.789, 857.169, 838.101` | 854.789 | `663.913, 663.650, 661.900` | 663.650 | 191.139 | 22.36% | 1.288x |
| GEMM2 producer | `75.0, 73.5, 74.6` | 74.600 | `57.0, 56.9, 56.3` | 56.900 | 17.700 | 23.73% | 1.311x |
| GEMM2 | `517.507, 516.779, 504.645` | 516.779 | `413.117, 411.029, 410.378` | 411.029 | 105.750 | 20.46% | 1.257x |
| MoE e2e | `1883.93, 1893.10, 1849.66` | 1883.93 | `1490.45, 1491.95, 1481.54` | 1490.45 | 393.48 | 20.89% | 1.264x |

optimized GEMM1 producer 的三段明细如下：

| kernel | samples (us) | median (us) |
|---|---|---:|
| `moe_quant_token_fd7168_fp4_pk8_hidtdm7` | `24.1, 25.6, 23.9` | 24.1 |
| `moe_invert_route_rows_tk6` | `2.6, 2.2, 2.0` | 2.2 |
| `moe_scatter_preshuffled_a_fd7168_r32_lds_pe7_slds_skipempty` | `44.0, 45.1, 43.5` | 44.0 |
| 每轮三段合计 | `70.7, 72.9, 69.4` | 70.7 |

### 5.2 GEMM 算力与等效带宽

| kernel | baseline (TFLOP/s) | optimized (TFLOP/s) | 增加 (TFLOP/s) | TFLOP/s 增幅 | baseline effective R+W (TB/s) | optimized effective R+W (TB/s) | 增加 (TB/s) | TB/s 增幅 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GEMM1 | 10,129.6 | 13,047.0 | 2,917.4 | 28.80% | 3.431 | 4.419 | 0.988 | 28.80% |
| GEMM2 | 8,377.5 | 10,532.9 | 2,155.4 | 25.73% | 4.739 | 5.958 | 1.219 | 25.73% |

### 5.3 正确性

| case | pass | logits_diff | rel_l2 | MoE output hash128 | ref output hash128 |
|---|:---:|---:|---:|---|---|
| baseline | True | `3.39799e-06` | `0.00260689` | `1556fc617347e2dabc9cff19dbfd822b` | `1a5d22911ba167160b4f2c12092a5193` |
| optimized | True | `3.39799e-06` | `0.00260689` | `1556fc617347e2dabc9cff19dbfd822b` | `1a5d22911ba167160b4f2c12092a5193` |

optimized 与 baseline 的 MoE output hash128 完全相同，说明本次对比中的 producer/GEMM
优化没有引入相对 baseline 的输出变化。random 输入下 MoE 输出与高精度 reference 并非
bit-exact，但两套被测实现具有相同的 `logits_diff`、`rel_l2` 和 hash，且均通过脚本阈值。

## 6. 结论

在 a07-3 当前空闲状态下，optimized 完整路径相对 baseline 的收益为：

| 模式 | kernel | producer 降幅 | kernel 降幅 | baseline TFLOP/s | optimized TFLOP/s | TFLOP/s 增幅 | baseline TB/s | optimized TB/s | MoE e2e 降幅 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `e2e-const0` | GEMM1 | 55.35% | 23.43% | 12,667.2 | 16,544.1 | 30.61% | 4.290 | 5.603 | 23.07% |
| `e2e-const0` | GEMM2 | 27.29% | 24.53% | 9,762.8 | 12,935.1 | 32.49% | 5.523 | 7.317 | 23.07% |
| `e2e-random` | GEMM1 | 53.91% | 22.36% | 10,129.6 | 13,047.0 | 28.80% | 3.431 | 4.419 | 20.89% |
| `e2e-random` | GEMM2 | 23.73% | 20.46% | 8,377.5 | 10,532.9 | 25.73% | 4.739 | 5.958 | 20.89% |

表中的 TB/s 是脚本按有效输入读取量与输出写入量之和计算的 `effective R+W`，不是硬件计数器测得的物理 HBM 流量。由于同一 GEMM 的 FLOP 数和有效读写字节数固定，TFLOP/s 与 effective R+W TB/s 的相对增幅都由 kernel latency 的下降决定；表内显示值经过小数位舍入。

主要结论：

1. const0 和 random 两种输入下，四个目标部分均获得稳定收益，没有出现局部优化拖累整体结果的情况。
2. const0 下四个目标部分的中位 latency 合计减少约 `371.44 us`，MoE e2e 减少 `372.44 us`；两者基本吻合。
3. random 下四个目标部分的中位 latency 合计减少约 `397.29 us`，MoE e2e 减少 `393.48 us`；约 `3.81 us` 的差异可由其余 kernel 和 profiler 轮间波动解释。
4. GEMM1 executed TFLOP/s 提升 `28.80%~30.61%`，effective R+W 从 `3.431~4.290 TB/s` 提升到 `4.419~5.603 TB/s`。
5. GEMM2 executed TFLOP/s 提升 `25.73%~32.49%`，effective R+W 从 `4.739~5.523 TB/s` 提升到 `5.958~7.317 TB/s`。
6. const0 与 random 的 optimized 输出都与各自 baseline 逐 hash 相同，正确性测试全部通过。

## 7. 原始结果目录

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260921T112300Z_gemm2_e2e-const0
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260921T112540Z_gemm2_e2e-const0
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260921T112752Z_gemm2_e2e-random
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260921T112922Z_gemm2_e2e-random
```
