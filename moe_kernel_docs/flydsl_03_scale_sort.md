# FlyDSL F3 — `mxfp4_moe_sort_kernel`（大 M：e8m0 scale 排序 + shuffle）

源码：`aiter/csrc/kernels/quant_kernels.cu`（kernel）/ host `mxfp4_moe_sort_hip`（`aiter/aiter/ops/quant.py:759`）
派发：`fused_dynamic_mx_quant_moe_sort` 的 split 分支（`quant.py:986`），紧跟在 [F2](flydsl_02_dynamic_quant.md) 之后
trace 名：`void aiter::mxfp4_moe_sort_kernel<256, 32, 32, 32>` / `<256, 64, 4, 32>`（模板为 block/tile 参数）

## 1. 概述

承接 [F2](flydsl_02_dynamic_quant.md)：F2 产出的是「按原行序、未 shuffle」的 per-token e8m0 scale。
本 kernel 按 `sorted_ids` 把这些 scale 字节 gather 到 sorted 行序，并 shuffle 成 FlyDSL gemm 所需的
**tile 布局**（与 [mxfp4_04 sort_scales](mxfp4_04_sort_scales.md) 在 MXFP4 路径里的角色相同，但这是 FlyDSL 路径用的字节搬运版本）。
它是 **dtype 无关的纯字节 shuffle**（fp4/fp8 的 e8m0 字节布局相同，只是字节值不同）。

## 2. 何时被选中（M 条件）

仅在 [F2](flydsl_02_dynamic_quant.md) 的 split 路径后执行（即 M≥256 的 stage1 输入；以及 M≥4096 的 stage2 中间量）。
- 被测里出现于 **M=256**（1 次：stage1 输入）与 **M=4096/8192/16384**（2 次：stage1 输入 + stage2 中间量）。
- trace 中两条不同模板实参（如 `<256,64,4,32>` 与 `<256,32,32,32>`）对应 stage1（block_m=128 排序）与 stage2（sort_block_m=32）不同的块布局。

## 3. Launch 参数与 shape（host `mxfp4_moe_sort_fwd`，`quant.py:773`）

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `scale`(=F2 输出) | `[token_rows, cols/32]` | e8m0(uint8) | 未 shuffle 的 per-token scale |
| `sorted_ids` | `[sorted_size]` | i32 | fused id（取低 24 位还原 token）|
| `num_valid_ids` | `[..]` | i32 | 有效行数 |
| `out_scale`(out) | `[pad32(sorted_size), pad8(cols/32)]` | e8m0(uint8) | sorted-tile shuffle 布局 |

`scaleN_pad = pad8(ceil(cols/32))`（`quant.py:782`），与 gemm 操作数的 `mx_scale_shuffle_idx` 对齐。

## 4. 计算逻辑

对每个目标 sorted-tile 位置：由 `sorted_ids` 还原 token → 从未 shuffle 的 `scale[token, k]` 读对应 e8m0 字节 →
按 `mx_scale_shuffle_idx`/tile 索引写入 `out_scale`。padding/越界行写合法占位字节。
（即 [F1](flydsl_01_fused_quant_sort.md) 第 5 步的 scale 搬运部分被单独成 kernel。）

## 5. 关键优化点

| 技术 | 目的 |
|---|---|
| 纯字节 shuffle，dtype 无关 | 同一 kernel 处理 fp4/fp8 的 e8m0 搬运 |
| 与 F2 解耦 | F2 专注高带宽量化、F3 专注轻量 scale 重排 |
| 直接写 gemm tile 布局 | gemm 无需 GEMM 内 reshuffle |
