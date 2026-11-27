# FlyDSL F7 — `aten::sum` → `at::native::reduce_kernel`（reduce 模式的 topk 归约）

来源：PyTorch 内置 `torch.sum`（非 aiter kernel）；由 `flydsl_moe_stage2` 在 reduce 模式收尾调用（`moe_kernels.py:1187-1188`）
trace 名：CPU `aten::sum`，GPU `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, ...>>`

## 1. 概述

当 stage2 GEMM（[F6](flydsl_06_moe2_gemm.md)）走 **reduce 模式**时，它把每个 `(token, slot)` 的 down 投影结果写到
暂存张量 `out_staging[token*topk + slot, model_dim]`（已乘路由权重），**不做 topk 累加**。
随后 host 调用 `torch.sum` 把同一 token 的 `topk=9` 个 slot 沿 dim=1 求和，得到最终 `out[token, model_dim]`：

```python
# moe_kernels.py:1187-1188
torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)
```

这一步在 GPU 上由 PyTorch 的通用归约 kernel `at::native::reduce_kernel` 执行。

## 2. 何时被选中（M 条件）

仅当 [F6](flydsl_06_moe2_gemm.md) 的 `kernelName2` 为 **reduce** 模式（即 `accumulate=False`）时执行。
被测里出现于 **M=4096 / 8192 / 16384**（这些 M 的 `kernelName2` 含 `_reduce`）。
小/中 M（≤256）用 atomic 模式，**不执行此步**。

## 3. 输入/输出 shape

| 张量 | shape | dtype | 说明 |
|---|---|---|---|
| 输入（staging）| `[token_num, topk=9, 7168]` | bf16 | F6 写出的每 (token,slot) 行（已乘 `sorted_weights`）|
| 输出 `out` | `[token_num, 7168]` | bf16 | 沿 topk 维求和的最终结果 |

## 4. 为什么大 M 用 reduce 而非 atomic

- atomic 模式（小/中 M）：topk slot 用 `atomic fadd` 直接累加到同一 token 行，省一次显存往返；但
  大 M 时同一 token 的多个 slot 高频原子争用，且写入分散，吞吐下降。
- reduce 模式（大 M）：F6 顺序写 `[tok*topk, H]` 暂存（无原子争用），再用一次规整的 `torch.sum` 批量归约，
  显存访问连续、带宽利用率高。代价是多一个 `[tok*topk, H]` 暂存缓冲与一次额外 kernel。

## 5. 说明

- 这是 **PyTorch 内置算子**，不在 aiter 源码内；其实现为 `at::native::reduce_kernel`（模板参数 `<vt, ...>`）。
- 在 `tt.log` 中体现为 CPU 侧 `aten::sum` + GPU 侧 `at::native::reduce_kernel`（如 M=8192 约 204µs，M=16384 约 413µs），
  是 FlyDSL 大 M 路径相对 MXFP4 路径较慢的因素之一（MXFP4 用专用 [scatter_reduce](mxfp4_07_scatter_reduce.md) 完成等价归约）。
