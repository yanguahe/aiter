# MXFP4 M9/M10 — `moe_scatter_reduce`（topk 加权求和）

源码：`aiter/csrc/kernels/mxfp4_moe/moe_aux/moe_scatter_reduce.cuh`（bf16 在 `:17`，mxfp4 在 `:98`）/ `mxfp4_moe_aux.cu:310/349`
host op：`aiter.mxfp4_moe_scatter_reduce(...)`（`fused_moe.py:1294`）/ `aiter.mxfp4_moe_scatter_reduce_q(...)`（`:1261`）
trace 名：`scatter_reduce_kernel_impl<7168, 9, 8, NT>`（M9，bf16）/ `scatter_reduce_mxfp4_kernel<7168, 9, 8, NT>`（M10，mxfp4）

## 1. 概述

MXFP4 **nonatomic** 路径（gemm2 写出每个 sorted 行而不累加）的 topk 归约 kernel。把同一 token 的
`topk=9` 个 slot 的 down 投影结果，按路由权重 `sorted_weights` 加权求和成最终 `out[token, 7168]`。
两个变体：
- **M9 `scatter_reduce_kernel_impl`**：gemm2 写的是 **bf16** `flat_out`，直接读取 + fma。
- **M10 `scatter_reduce_mxfp4_kernel`**：gemm2 写的是 **fp4 `flat_out_q` + e8m0 `flat_out_scale`**（mxfp4out 模式），
  先解包再 fma（读取量更小）。

## 2. 何时被选中（M 条件）

- 仅 MXFP4 **nonatomic**（BM=128）路径，即 gemm2 epilog 非 atomic 时执行（`fused_moe.py:1289` 之后）。
- **M9**：M=4096（`gemm2` nonatomic、非 mxfp4out）。
- **M10**：M=8192 / 16384（`gemm2` mxfp4out）。
- 小/中 M（≤256）走 atomic，**不执行任何 scatter_reduce**。

## 3. Launch 参数与 shape

`grid = dim3(ceil(7168/(COLS_PER_THREAD*THREADS)), M)`，按 (列块, token) 二维铺开。
`COLS_PER_THREAD=8`；M9 `THREADS=128`，M10 `THREADS=128`（`aux.cu:30-33`）。
`NT_HINTS = (BM>=128)`（大 M DRAM-bound，用 non-temporal load/store；`aux.cu:326/364`）。

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `flat_out`(M9) | `[max_sorted, 7168]` | bf16 | gemm2 每 sorted 行的结果 |
| `flat_out_q`(M10) | `[max_sorted, 7168/2]` | uint8 | fp4 暂存 |
| `flat_out_scale`(M10) | `[max_sorted, 7168/32]` | uint8 | e8m0 暂存 |
| `reverse_sorted` | `[M*9]` | i32 | `(token,slot) → sorted 位置` |
| `sorted_weights` | `[max_sorted]` | f32 | 路由权重 |
| `out`(out) | `[M, 7168]` | bf16 | 最终输出 |

## 4. 计算逻辑

每个线程负责一个 token 的 `COLS_PER_THREAD=8` 列：

**M9（`:17`）**：
```
acc[8] = 0
for i in 0..TOPK:
    sorted_pos = reverse_sorted[token*TOPK + i]
    w = sorted_weights[sorted_pos]
    读 flat_out[sorted_pos, col..col+8]（int4 向量，NT 可选）
    acc[c] = fma(bf16→f32, w, acc[c])
out[token, col..] = bf16(acc)        # int4 向量写出
```

**M10（`:98`）**：与 M9 相同，但每次读 `flat_out_q` 的 1 个 u32（8 个 fp4）+ 对应 `flat_out_scale` 的 e8m0 字节，
`fp4_to_fp32_packed_x8(pack, scale)` 解包成 8 个 f32 后再 fma（`:131-138`）。锁定 `COLS_PER_THREAD=8`
（一个 u32 = 一个 32 列 block 内的 8 个 fp4，latency/MLP 最优）。

## 5. 关键优化点

| 技术 | 目的 |
|---|---|
| `reverse_sorted` 直接定位 topk 行 | 无需扫描，O(topk) 读取每 token |
| `int4`(16B) 向量化读写（M9）| 一次处理 8 个 bf16 |
| mxfp4 暂存（M10）| gemm2→reduce 之间的中间数据量 ↓ ~3.8×，缓解大 M 的 DRAM 带宽瓶颈 |
| `NT_HINTS`（BM≥128）| 大 M 数据不复用，non-temporal 绕过 L2 |
| 列块 × token 二维 grid | 列方向并行 + token 方向并行，铺满 CU |
