# Baseline vs. a4w4_prefill_v2_yadai: E64/T1536/topk8

## 测试范围

- 机器：`a07-3`
- 容器：`hyg_fyd_e2e`
- 规模：`experts=64, tokens=1536, topk=8, model_dim=7168, inter_dim=2048`
- 数据：const0
- 每个 case：`num_warmup=5, num_iters=20`
- 外层轮数：3
- Baseline label：`04cc526b8f06f1e54f836718964e8f2e2c844fa8`
- Yadai commit：`8eebd41402b3bf323b2290b1ea98cdbf221bb51e`
- Run ID：`20261001T120752Z`

本轮日志保存在：

```text
/app/aiter/my_code/moe_prefill_yadai_compare_runs/20261001T120752Z
```

执行命令：

```bash
cd /app/aiter
ROUNDS=3 bash my_code/run_moe_prefill_yadai_compare.sh
```

脚本在 baseline 前后以及每个 yadai kernel/e2e case 前后检查了 GPU/KFD、GPU
utilization 和 VRAM。首次检查遇到一个已经退出的瞬态进程和 92% residual utilization，
脚本等待 2 秒并确认全部归零后才开始正式测试。每个 case 结束后也等待 residual
utilization 回落到 0，因此本报告中的性能数据有效。

Baseline 阶段不执行真实 Git 操作，直接测试 `/app/aiter` 当前 working tree。日志中的
commit 是脚本提供的 label；如果 `/app/aiter` 存在未提交修改，则 baseline 表示该
working tree，而不是严格的纯 commit snapshot。

Yadai 阶段在容器内先执行：

```bash
source /data/yanguahe/code/git_env
```

然后 fetch 最新 `ROCm/dev/a4w4_prefill_v2_yadai`，确认 tracked 文件 clean，并显式
设置 `PYTHONPATH`。本轮 import 路径为：

```text
/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai/aiter/__init__.py
```

## 计时口径

两边 MoE e2e 使用相同计时方式：

```text
testGraph=False
use_cuda_event=False
num_warmup=5
num_iters=20
backend=torch.profiler
metric=get_trace_perf(...).device_time_sum
```

`AITER_LOG_MORE=1` 会额外打印 CUDA-event 时间，但最终 e2e 时间使用 torch profiler
的 GPU `device_time_sum`，不是 CUDA-event 输出。

## 总体性能中值

“其他 kernel 合计”先对每一轮执行：

```text
MoE e2e - GEMM1 in e2e - GEMM2 in e2e
```

再对三轮结果取中值。

| 项目 | Baseline samples | Baseline median | Yadai samples | Yadai median | Yadai - Baseline |
|---|---|---:|---|---:|---:|
| GEMM1 in e2e | 82.989, 80.405, 78.842 us | 80.405 us | 125.147, 118.716, 122.453 us | 122.453 us | +42.047 us (+52.29%) |
| GEMM2 in e2e | 60.368, 61.484, 58.889 us | 60.368 us | 91.289, 89.642, 90.258 us | 90.258 us | +29.889 us (+49.51%) |
| 其他 kernel 合计 | 78.602, 73.681, 70.438 us | 73.681 us | 61.499, 63.237, 65.878 us | 63.237 us | -10.443 us (-14.17%) |
| **MoE e2e** | **221.960, 215.570, 208.170 us** | **215.570 us** | **277.936, 271.595, 278.589 us** | **277.936 us** | **+62.366 us (+28.93%)** |

Yadai 的非 GEMM 部分减少约 `10.44 us`，但两个 GEMM 在 e2e 中合计增加约
`71.94 us`，因此最终 MoE e2e 增加约 `62.37 us`。

## Yadai isolated GEMM 性能

Yadai 的 `scenario kernel` 是固定输入和固定 buffer 下连续执行单个 kernel 的 isolated
测量，不等同于 e2e 中的 GEMM 时间。

| Kernel | Samples | Median |
|---|---|---:|
| GEMM1 isolated | 82.850, 81.940, 82.100 us | 82.100 us |
| GEMM2 isolated | 59.830, 60.410, 60.860 us | 60.410 us |

与 Yadai e2e 中值相比：

```text
GEMM1: 82.100 us isolated -> 122.453 us in e2e
GEMM2: 60.410 us isolated ->  90.258 us in e2e
```

## 非 GEMM 阶段中值对比

下面按语义阶段聚合。每个阶段先在每一轮中求和，再对三轮取中值，因此不同阶段的中值
相加不一定严格等于“其他 kernel 合计”的中值。

| 非 GEMM 阶段 | Baseline median | Yadai median | 差值 |
|---|---:|---:|---:|
| 两个 PyTorch `fill` 合计 | 3.637 us | 4.837 us | +1.200 us (+33.00%) |
| Route | 6.453 us | 6.384 us | -0.068 us (-1.06%) |
| Contiguous psum/remap | 4.684 us | 4.905 us | +0.221 us (+4.72%) |
| GEMM1 A/ScaleA 量化与 preshuffle | 26.532 us | 28.658 us | +2.126 us (+8.01%) |
| GEMM1→GEMM2 独立量化 | 11.363 us | 0 us | -11.363 us (-100.00%) |
| Gather/reduce | 19.542 us | 18.332 us | -1.211 us (-6.19%) |

### 非 GEMM 原始 symbol 中值

`fill` 每轮出现两次，因此使用 `device_time_sum / 19` 作为单轮总贡献。其他 symbol 每轮
出现一次，其贡献等于对应的 `device_time_avg`。

| Kernel symbol | Baseline median | Yadai median | 说明 |
|---|---:|---:|---|
| PyTorch vectorized fill，两个 launch 合计 | 3.637 us | 4.837 us | 公共辅助初始化 |
| `moe_route` | 6.453 us | 6.384 us | 基本相同 |
| `moe_contiguous_psum_remap` | 4.684 us | 4.905 us | 基本相同 |
| `moe_quant_token_fd7168_fp4_pk8_hidtdm7` | 11.147 us | — | Baseline GEMM1 A quant |
| `moe_invert_route_rows_tk8` | 3.158 us | — | Yadai 删除 |
| `moe_scatter_preshuffled_a_fd7168_r32_lds_pe7_slds_skipempty` | 12.626 us | — | Yadai 删除 |
| `moe_token_multidest_quant_fusepre_k8_fd7168_r8_fp4_pk8_hidtdm4_quant` | — | 23.142 us | Yadai multidest payload quant/scatter |
| `moe_scatter_preshuffle_scale_b224_r8_k8_g` | — | 5.574 us | Yadai ScaleA scatter/preshuffle |
| `moe_quant_preshuffled_a_fd2048_rpw2_pf2_direct_hidtdm4_otdmw2` | 11.363 us | — | Yadai 融入 GEMM1 epilogue |
| `moe_gather_reduce_bf16_d7168_tk8_sk1_v4_wbf16_frlds` | 19.542 us | 18.332 us | Yadai 快约 1.21 us |

## Kernel pipeline 差异

Profiler 的展示表按耗时排序，不代表 dispatch 顺序。根据实际 symbol 和 pipeline 源码，
核心顺序如下。

### Baseline

```text
moe_route
-> moe_contiguous_psum_remap
-> moe_quant_token_fd7168_fp4_pk8_hidtdm7
-> moe_invert_route_rows_tk8
-> moe_scatter_preshuffled_a_fd7168_r32_lds_pe7_slds_skipempty
-> GEMM1 t192
-> moe_quant_preshuffled_a_fd2048_rpw2_pf2_direct_hidtdm4_otdmw2
-> GEMM2 t192
-> moe_gather_reduce_bf16_d7168_tk8_sk1_v4_wbf16_frlds
```

### Yadai

```text
moe_route
-> moe_contiguous_psum_remap
-> moe_token_multidest_quant_fusepre_k8_fd7168_r8_fp4_pk8_hidtdm4_quant
-> moe_scatter_preshuffle_scale_b224_r8_k8_g
-> GEMM1 t256 + SiLU + GEMM2 A quant
-> GEMM2 t256
-> moe_gather_reduce_bf16_d7168_tk8_sk1_v4_wbf16_frlds
```

Yadai 将核心自定义 kernel 数从 9 个减少到 7 个：

- GEMM1 A 准备从 quant + invert + payload scatter 三个 kernel 改为 multidest quant +
  scale scatter 两个 kernel。
- 删除 GEMM1 后的独立 GEMM2 A quant，将其融合进 GEMM1 `q1r8` epilogue。

本轮数据表明，kernel fusion 确实让非 GEMM 部分更快，但 Yadai 的 t256 GEMM1/GEMM2
在 e2e 上下文中的增加量更大，最终导致 MoE e2e 回退。

## 正确性

Baseline 三轮：

```text
logits_diff = 0
rel_l2      = 0
pass        = True
```

Yadai 三轮：

```text
logits_diff = 0
rel_l2      = 0
pass        = True
```

Baseline GEMM1、GEMM2 和最终 MoE output hash 均与 reference 一致：

```text
GEMM1: c281c06c980fd4ca89d84615b26083b9
GEMM2: 8435f663d2aae0fe93d109c485cd9265
MoE:   6bebf6409ef198fe1a0255681f4f784f
```
