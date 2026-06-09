# MXFP4 M6 — `moe_sort_scales::sort_scales_kernel_impl`（a_scale 排序 + shuffle）

源码：`aiter/csrc/kernels/mxfp4_moe/moe_aux/moe_sort_scales.cuh:14`（kernel）/ `mxfp4_moe_aux.cu:263`（host）
host op：`aiter.mxfp4_moe_sort_scales(...)`（`fused_moe.py:1163`）
trace 名：`void aiter::mxfp4_moe::moe_sort_scales::sort_scales_kernel_impl<{32|128}, 385, 512, 7168, 256, 512, 1024>`

## 1. 概述

[mxfp4_03 量化](mxfp4_03_quant.md) 产出的 `a_scale[M, 224]` 是**按原 token 顺序、未 shuffle** 的 e8m0 字节。
本 kernel 把它按 `sorted_token_ids` 的顺序 gather，并重排成 **gemm1 MFMA scale-tile 所需的 shuffle 布局**
`a_scale_sorted_shuffled`，使 gemm1 能用 `buffer_load_lds`/`buffer_load_b128` 直接以 tile 布局取 scale。
（小 M inline_quant 路径不跑此 kernel——scale 由 gemm1 内部即时产出 sorted-tile 布局。）

## 2. 何时被选中（M 条件）

- 仅 MXFP4 **threestage** 路径（`prologue != "inline_quant"`）执行（`fused_moe.py:1158-1170`）。
- 被测 M：**256（BM=32）、4096 / 8192 / 16384（BM=128）**。
- 约束：要求 `BM≥32`（MN_PACK=2 布局），host 在 BM=16 时 clamp 到 32（`aux.cu:281`）。

## 3. Launch 参数与 shape

模板：`<BM, NE=385, D_INTER=512, D_HIDDEN=7168, BK=256, N_CTAS=512, THREADS=1024>`，`<<<512,1024>>>`。

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `a_scale` | `[M, 7168/32]=[M,224]` | uint8 | 未排序 per-token e8m0（来自 M5）|
| `sorted_token_ids` | `[max_sorted]` | i32 | fused id（取低 24 位还原 token）|
| `cumsum_tensor` | `[1]` | i32 | 实际有效行数 |
| `a_scale_sorted_shuffled`(out) | `[pad32(max_sorted)*224*2]` | uint8 | gemm1 tile 布局 scale（host 分配，`fused_moe.py:1159-1162`）|

## 4. 计算逻辑（`:14-78`）

把输出看成 `n_chunks = MAX_SORTED/BM` 个 chunk，每 chunk `DWORDS_PER_CHUNK = C_M1*C_K1*K_LANE*N_LANE`
个 4 字节单元（`C_M1=BM/32`、`C_K1=(7168/32)/8=28`、`K_LANE=4`、`N_LANE=16`、`MN_PACK=2`、`K_PACK=BK/128=2`）。
512×1024 线程以 grid-stride 处理全部 `n_chunks*DWORDS_PER_CHUNK` 个 work_id：

1. 由 work_id 反解出 `(chunk, mi, ku, k_lane, n_lane)`（`:43-48`）。
2. 若 `chunk < actual_n_chunks`：对 `MN_PACK=2` 个子行，从 `sorted_token_ids[chunk*BM + (mi*2+im)*16 + n_lane]`
   取低 24 位得到 `token`（越界 token 取 0），再从 `a_scale[token*224 + k_idx]` 读 `K_PACK*MN_PACK=4` 个 e8m0
   字节（`k_idx = ku*8 + ikxdl*4 + k_lane`，`:64-71`）。
3. 把 4 个字节打包成 1 个 u32 顺序写入 `a_scale_sorted_shuffled[work_id*4]`（`:74-76`）。

这套 `(MN_PACK, K_PACK, C_M1, C_K1, K_LANE=4, N_LANE=16)` 的索引拆解，正是 gemm1 的 `mfma_scale_*`
对 A-scale 操作数的 lane×byte 排布，gemm1 因此可零成本对齐取用。

## 5. 关键优化点

| 技术 | 目的 |
|---|---|
| gather + tile-shuffle 合一 | 排序重排与 MFMA scale 布局在一遍内完成 |
| 4 字节打包 u32 写出 | 单次写满一个 dword，写带宽友好 |
| 越界 token 取 0 | padding 行的 scale 取合法字节，避免越界读 |
| 与 gemm1 操作数布局完全对齐 | gemm1 用 `buffer_load_lds` 直取，无需 GEMM 内重排 |
