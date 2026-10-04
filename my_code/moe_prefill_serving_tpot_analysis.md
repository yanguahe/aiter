# MoE Prefill 优化对 Serving TPOT 的影响分析

## 目录

- [1. 背景与目标](#1-背景与目标)
- [2. 对比版本与测试条件](#2-对比版本与测试条件)
- [3. Serving Benchmark 结果](#3-serving-benchmark-结果)
- [4. 第一轮定位：Decode fallback 不完整](#4-第一轮定位decode-fallback-不完整)
- [5. 第二轮定位：共享 quant emitter](#5-第二轮定位共享-quant-emitter)
- [6. Decode CUDAGraph Trace 对比](#6-decode-cudagraph-trace-对比)
- [7. 最终判断](#7-最终判断)
- [8. 后续验证建议](#8-后续验证建议)
- [9. 实验产物与当前状态](#9-实验产物与当前状态)

## 1. 背景与目标

`hyg/moe_a4w4_pr` 分支的最终目标是优化 LLM 推理中 MoE prefill 阶段的性能。

目标 shape 为：

```text
data_format = a4w4
experts     = 96（DSv4 单测）/ 257（DSR1 serving，含 shared expert）
tokens      = 16384
topk        = 6（DSv4）/ 9（DSR1 serving）
model_dim   = 7168
inter_dim   = 3072（DSv4）/ 2048（DSR1）
activation  = SiLU
```

主要优化包括：

- GEMM1 A/ScaleA preshuffle；
- GEMM2 A/ScaleA preshuffle；
- GEMM1、GEMM2 的专用 LDS layout、descriptor 和 load mapping；
- GEMM1/GEMM2 TDM、WMMA scheduling 和 output-store 优化；
- 仅在 tuned `t256x256x256/w2x2/b4` tile 上启用 A-preshuffle；
- 未命中 tuned tile 时恢复原 row-major A producer。

Serving benchmark 中观察到：

- TTFT 明显改善，说明 prefill 优化生效；
- TPOT 一度增加约 `1%~2%`；
- 需要判断 TPOT 差异来自真实的 MoE decode kernel 回退，还是 serving scheduler、batch composition 或机器运行状态波动。

## 2. 对比版本与测试条件

### 2.1 Baseline

```text
commit: 314ab7d48a077f3ccc055e7d353040c10a5f7fe5
```

Baseline server 启动方式：

```bash
FAKE_EPLB=1 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
bash /app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh \
  /data/models/DeepSeek-R1-0528-MXFP4
```

### 2.2 Optimized

最终分析使用的 optimized commit：

```text
881a01aca0eb3c0362faea15165746ca3335def2
```

相关修正提交：

```text
70b178e428f129ead6563d54ab161201d5188672
Restore fused quantization for untuned GEMM2 tiles

881a01aca0eb3c0362faea15165746ca3335def2
Restore baseline quant emitter outside A-preshuffle
```

Optimized server 保留 A-preshuffle 专用参数，但不再全局设置：

```text
AITER_FLYDSL_MXFP4_CLUSTER_N
AITER_TDM_NEXT_STAGE_PREFETCH
```

`T=16384` tuned CSV 已包含：

```text
cluster_n=4
next_stage_prefetch=1
```

因此：

- prefill `t256...` 仍使用 tuned cluster/prefetch；
- decode `t64...` 恢复默认 `cluster_n=1` 和 `next_stage_prefetch=0`。

### 2.3 Serving Benchmark

```bash
cd /app/ATOM/scripts/performance
bash /app/scripts/dsr1/bench_dsr1fp4.sh \
  /data/models/DeepSeek-R1-0528-MXFP4
```

主要参数：

```text
ISL             = 1024
OSL             = 1024
num_prompts     = 256
max_concurrency = 64
request_rate    = inf
ignore_eos      = true
```

## 3. Serving Benchmark 结果

### 3.1 Baseline 与各 optimized 版本

| 版本 | Mean TTFT | Mean TPOT | Mean E2EL | Total throughput |
|---|---:|---:|---:|---:|
| baseline `314ab7d` | 1894.54 ms | 31.56 ms | 34182.35 ms | 3833.98 tok/s |
| `2da6980d` | 1724.19 ms | 32.30 ms | 34769.11 ms | 3769.28 tok/s |
| `70b178e4` run 1 | 1710.98 ms | 31.98 ms | 34424.15 ms | 3807.04 tok/s |
| `70b178e4` run 2 | 1728.84 ms | 32.28 ms | 34748.31 ms | 3771.50 tok/s |
| `881a01ac` | 1720.27 ms | 32.19 ms | 34655.37 ms | 3781.62 tok/s |

### 3.2 直接观察

相对 baseline，`881a01ac`：

```text
TTFT:       1894.54 -> 1720.27 ms，改善约 9.20%
TPOT:         31.56 ->   32.19 ms，增加约 2.00%
E2EL:      34182.35 -> 34655.37 ms，增加约 1.38%
Throughput: 3833.98 -> 3781.62 tok/s，下降约 1.37%
```

TTFT 改善说明 prefill 优化真实生效。

但 optimized 多轮 TPOT 为：

```text
31.98 ms
32.28 ms
32.19 ms
```

自身波动达到 `0.30 ms`，已经接近相对 baseline 的部分差值，不能仅凭单轮 e2e TPOT 判定某个 decode kernel 回退。

## 4. 第一轮定位：Decode fallback 不完整

### 4.1 原始问题

Baseline 对 A4W4、无 bias 的路径使用 GEMM1 fused quant：

```python
_fuse_quant = _b1 is None
```

路径为：

```text
GEMM1 + SiLU + FP4 quant epilogue
    -> GEMM2
```

早期 optimized 版本改成：

```python
_fuse_quant = (not _is_fp4) and (_b1 is None)
```

这使所有 A4W4 调用都关闭 fused quant，即使 decode tile 没有命中 A-preshuffle，也会走：

```text
GEMM1 写 BF16 y
    -> 独立 quant_a2 kernel
    -> GEMM2
```

### 4.2 修正

在 commit `70b178e4` 中改为：

```python
_fuse_quant = (_b1 is None) and not (_is_fp4 and _gemm2_a_preshuffle)
```

结果：

- prefill `t256...` 且 GEMM2 A-preshuffle 启用：保留 BF16 `y` 和独立 A-preshuffle producer；
- decode `t64...` 未命中：恢复 baseline GEMM1 fused quant epilogue。

同时 optimized server 不再全局设置：

```text
AITER_FLYDSL_MXFP4_CLUSTER_N=4
AITER_TDM_NEXT_STAGE_PREFETCH=1
```

### 4.3 结果

TPOT 从 `32.30 ms` 一度恢复到 `31.98 ms`，但重复运行得到 `32.28 ms`，说明仍存在较明显的运行波动。

## 5. 第二轮定位：共享 quant emitter

### 5.1 假设

当前分支曾修改普通 producer 与 A-preshuffle producer 共用的 `_emit_quant_block_loop`：

```python
fx.max(...)
emit_mx_e8m0_scale(...)
```

被替换为：

```python
maximumf(...)
_emit_mx_e8m0_scale_apre(...)
```

虽然功能等价，但可能产生不同的 MLIR/ISA 或寄存器调度。

### 5.2 修正

在 commit `881a01ac` 中：

- 非 A-preshuffle `_emit_quant_block_loop` 完整恢复 baseline 实现；
- A-preshuffle 继续使用独立 `_emit_quant_block_loop_apre`；
- 对比 `314ab7d` 后，baseline 原有 emitter 区域不再存在算法差异。

### 5.3 正确性

```text
T=64 decode fallback: pass=True
T=16384 prefill Apre: pass=True
```

`T=16384` 输出 hash 与修改前一致。

### 5.4 性能

修正后 TPOT 为：

```text
32.19 ms
```

仍位于修正前 `31.98~32.28 ms` 的波动范围内。

因此，共享 quant emitter 不是剩余 TPOT 差异的主要原因。

## 6. Decode CUDAGraph Trace 对比

### 6.1 Trace workload

为避免完整 `1024×1024` trace 过大，使用固定 decode workload：

```text
num_prompts     = 64
max_concurrency = 64
input_len       = 1
output_len      = 32
ignore_eos      = true
```

主 trace 覆盖一个很小的 prefill 和约 31 个固定 `batch=64` decode steps。

Trace 文件：

```text
/app/traces/tpot_base_314_20260922/rank_0/
/app/traces/tpot_opt_881a_20260922/rank_0/
```

### 6.2 Serving 指标

| 版本 | Mean TTFT | Mean TPOT | Mean E2EL |
|---|---:|---:|---:|
| baseline | 381.30 ms | 32.30 ms | 1382.75 ms |
| optimized | 441.82 ms | 31.69 ms | 1424.13 ms |

在固定 decode batch 下，optimized TPOT 反而比 baseline 快约 `1.89%`。

### 6.3 Kernel 集合与调用次数

两份 trace：

```text
trace events: 145,924 vs 145,924
kernel events: 49,693 vs 49,693
逐 kernel name 调用次数差异: 0
```

严格事件顺序在随机采样/utility kernel 附近存在少量相对顺序差异，但不是 MoE graph 拓扑变化；kernel multiset 和所有 kernel 调用次数完全一致。

两版都记录：

```text
decode[bs=64 tok=64 d=64] × 32
```

说明固定 workload 下 scheduler 形成的 decode batch 完全相同。

### 6.4 每个 decode step

| 指标 | baseline | optimized | 变化 |
|---|---:|---:|---:|
| decode annotation 平均时间 | 31.633 ms | 31.405 ms | optimized 快 0.72% |
| GPU kernel 总时间/step | 28.475 ms | 28.359 ms | optimized 快 0.41% |
| kernel 数/step | 1452 | 1452 | 相同 |

### 6.5 MoE kernel 总时间

按 kernel name 过滤 MoE、route、quant、gather 和 grouped GEMM：

```text
baseline:  673796.721 us
optimized: 671430.124 us
```

optimized 的 MoE kernel 总时间快约 `0.35%`。

主要 kernel 对比：

```text
GEMM2 K2048:
  baseline  221007.028 us / 1972 calls
  optimized 218444.915 us / 1972 calls
  optimized 总计快约 2562 us

GEMM1 K7168:
  baseline  422997.473 us / 1972 calls
  optimized 422472.273 us / 1972 calls
  optimized 总计快约 525 us

quant_a1:
  baseline  11440.621 us / 1972 calls
  optimized 11090.071 us / 1972 calls
  optimized 总计快约 351 us
```

`moe_route` 和 `moe_contiguous_psum_remap` 在 optimized trace 中有微小增加，但整个 MoE 合计仍然更快。

### 6.6 全部 GPU kernel

```text
baseline:  968895.565 us
optimized: 964005.128 us
```

optimized 的全部 GPU kernel 总时间快约 `0.50%`。

因此可以排除：

- decode 意外启用了 A-preshuffle；
- decode 多出或缺少 kernel；
- decode batch size 不同；
- GEMM1/GEMM2 decode kernel 本身回退；
- MoE decode kernel 总时间回退。

## 7. 最终判断

现有证据不支持“optimized 分支导致 MoE decode kernel 变慢”。

完整 `ISL=1024/OSL=1024/concurrency=64` benchmark 的 TPOT 差异更可能来自以下因素。

### 7.1 Prefill 改变请求进入 decode 的时间分布

Serving 的最大 batch token 数为：

```text
max_num_batched_tokens = 16384
```

每个请求输入 1024 tokens，因此单个 prefill batch 最多容纳：

```text
16384 / 1024 = 16 requests
```

64 个请求约分为 4 轮 prefill。

optimized prefill 更快，会让多组请求更集中地进入 decode；baseline prefill 较慢，请求进入 decode 的时间更分散。这会改变：

- 同时处于 decode 的 sequence 数量；
- decode batch occupancy；
- 不同请求之间的 scheduler 等待；
- 请求完成和 batch shrink 的时间分布。

Serving 输出的 TPOT 不是单个 MoE kernel 时间，而是包含 scheduler、batch composition、graph replay、attention 和请求间等待的服务指标。

### 7.2 机器运行状态与长时间测试波动

optimized 正式 benchmark 的 TPOT 样本为：

```text
31.98 ms
32.28 ms
32.19 ms
```

单版本自身波动约 `0.30 ms`。

而固定 batch trace 中 optimized GPU kernel 和 MoE kernel 均没有回退。这表明完整测试中的 `0.4~0.7 ms` 差值包含明显的机器运行状态、GPU 工作频率、温度、内存带宽或 host scheduling 波动。

### 7.3 长 KV attention 占比

固定 trace 的 KV 长度为 1–32，而完整测试 decode 的 KV 长度约为 1024–2048。长 KV 下 attention 和 KV-cache 访问占比更高。

分支只修改 MoE 相关源码，没有修改 attention kernel。如果长上下文阶段出现差异，更可能来自机器带宽/频率和 serving 调度，而不是当前 MoE kernel 代码。

尝试采集 `input=1024/output=32/concurrency=64` 的完整 profiler trace 时，主机出现：

```text
system load > 130
swap 使用约 47%
```

为避免机器再次 hang，该 trace 被终止，server 也被提前关闭。

## 8. 后续验证建议

后续不再依赖大规模 server trace，使用单测继续定位。

### 8.1 MoE decode 单测矩阵

分别在 baseline 与 optimized 上测试：

```text
tokens = 64, 128, 256, 512
experts = 257
topk = 9
model_dim = 7168
inter_dim = 2048
data_format = a4w4
```

记录：

- quant_a1；
- GEMM1；
- GEMM2；
- route/psum/gather；
- MoE e2e。

每个 shape 至少多轮交替 A/B，避免机器漂移。

### 8.2 评价指标拆分

Prefill 优化应主要关注：

- TTFT；
- prefill-only MoE e2e；
- `T=16384` GEMM1/GEMM2 与 producer 时间。

Decode 等价性应通过：

- 固定 token 数的 MoE 单测；
- 固定 `decode[bs=...]` 的 CUDAGraph trace；
- kernel 序列与逐 kernel 时间。

不要仅把完整 serving TPOT 当作某个 MoE decode kernel 的耗时。

### 8.3 更严格的 serving A/B

若后续重新获得独占机器时间，建议：

1. 锁定或至少记录 GPU clock、温度和功耗；
2. baseline/optimized 交替执行，而不是先跑完一版再跑另一版；
3. 每版至少 3 轮；
4. 保存详细 per-request TTFT/ITL；
5. 同时记录 scheduler 每步的 decode batch size；
6. 对 TTFT、TPOT、throughput 分别取 median。

## 9. 实验产物与当前状态

### 9.1 Trace

```text
/app/traces/tpot_base_314_20260922/
/app/traces/tpot_opt_881a_20260922/
```

### 9.2 Serving 结果

```text
/app/ATOM/scripts/performance/result_baseline_314ab7d48.txt
/app/ATOM/scripts/performance/result_optimized_2da6980d.txt
/app/ATOM/scripts/performance/result_optimized_70b178e4.txt
/app/ATOM/scripts/performance/result_optimized_881a01ac.txt
```

### 9.3 当前代码状态

```text
branch: hyg/moe_a4w4_pr
HEAD:   881a01aca0eb3c0362faea15165746ca3335def2
```

### 9.4 Server 状态

```text
ATOM server 已关闭
GPU 无残留进程
关闭时间：2026-09-22 06:07 UTC
```

## 总结

当前 optimized 分支的 prefill 优化有效，TTFT 改善约 `9%`。

固定 decode batch 的 trace 证明：

- decode kernel 集合和调用次数与 baseline 相同；
- optimized 的 MoE kernel 总时间没有回退；
- optimized 的全部 GPU kernel 总时间也没有回退。

因此，完整 serving benchmark 中观察到的 TPOT 增加不能归因于 MoE decode kernel。最合理的解释是 prefill 加速改变了 serving 调度轨迹，再叠加机器运行状态和长 KV attention 的性能波动。
