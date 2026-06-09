# MXFP4 M5 — `moe_sort_quant::quant_kernel_impl`（激活量化 bf16→mxfp4）

源码：`aiter/csrc/kernels/mxfp4_moe/moe_aux/moe_sort_quant.cuh:294`（kernel）/ `mxfp4_moe_aux.cu:219`（host）
host op：`aiter.mxfp4_moe_quant(...)`（`fused_moe.py:1140`）
trace 名：`void aiter::mxfp4_moe::moe_sort_quant::quant_kernel_impl<385, 9, {32|128}, 7168, 512, 1024>`

## 1. 概述

在 MXFP4 **threestage** 路径（M≥256）里，激活量化是一个独立 kernel：把输入
`hidden_states[M, 7168] (bf16)` 按 **每 32 元素一组** 量化成 `fp4 e2m1`（`a_quant[M, 3584] uint8`，
2 个 fp4/字节）+ 每组一个 `e8m0` 缩放字节（`a_scale[M, 224] uint8`，`7168/32=224`）。
（小 M 的 inline_quant 路径不跑这个 kernel——量化融进 gemm1。）

注意：本 kernel 产出的是**未排序**的 per-token scale；GEMM 需要的是按 sorted 顺序重排+shuffle 后的
scale，由 [mxfp4_04 `sort_scales`](mxfp4_04_sort_scales.md) 完成。

## 2. 何时被选中（M 条件）

- MXFP4 threestage 路径（`kernelName1` 不含 `INLINEQUANT`）→ `_mxfp4_moe_run` 在三段排序后调
  `aiter.mxfp4_moe_quant`（`fused_moe.py:1140`）。
- 被测 M：**256（MB=32）、4096 / 8192 / 16384（MB=128）**。

## 3. Launch 参数与 shape

模板：`<NE=385, TOPK=9, MB, D_HIDDEN=7168, N_CTAS=512, THREADS=1024>`，`<<<512,1024>>>`（`aux.cu:26-27` `kNCtasSort/kThreadsSort`）。

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `a_input` (= hidden_states) | `[M, 7168]` | bf16 | 输入激活 |
| `a_quant`(out) | `[M, 7168/2]=[M,3584]` | uint8 | 打包 fp4（2/字节）|
| `a_scale`(out) | `[M, 7168/32]=[M,224]` | uint8 | 每 32 列一个 e8m0 |
| `bf16_zero_out` | nullptr / `[M,7168]` | bf16 | 本配置为 nullptr（不清零）|

## 4. Grid / Block 映射

- 把 `M*224` 个 32-列 block 看作工作单元，512 个 CTA × 1024 线程 grid-stride 处理（`quant_impl`，`:218`）。
- 每个 wave（64 lane）= `BLOCKS_PER_WAVE=16` 个 32-列 block × `LANES_PER_BLOCK=4`；每 4 个 lane 协作量化一个 32-列组。

## 5. 计算逻辑（`quant_impl`，`:218-285`）

对每个 `(token, 32-列组)`：
1. 每 lane 用 `int4` 读 8 个 bf16（`lane_in_block*8`），取局部 `|bf16|` 最大值（屏蔽符号位 `&0x7FFF`）。
2. **4-lane 内 DPP 归约**（`mov_dpp 0xB1/0x4E`，`:262-265`）得到该 32-列组的 `absmax`。
3. 由 `absmax` 算 e8m0 指数：`bexp = ((f32bits+0x200000)>>23)&0xFF`（含 1.5× 处舍入），
   `scale = clamp(bexp-2, 0, 254)`（`:268-270`），量化除数 `qs = 2^(scale)`。
4. 用 4 次 `__builtin_amdgcn_cvt_scalef32_pk_fp4_bf16`（每次 2 个 bf16→2 个 fp4）打包成 1 个 u32 写入 `a_quant`。
5. `lane_in_block==0` 的 lane 写 `a_scale[block]=scale`。

`scale=bexp-2`（而非 `-1`）：fp4 e2m1 最大值为 6，把缩放对齐到 2 的幂以保证反量化精确（`value=fp4*2^(scale)`）。

## 6. 关键优化点

| 技术 | 目的 |
|---|---|
| `int4`（16B）向量化加载 | 一条指令读 8 个 bf16 |
| 4-lane DPP 组内 amax 归约 | 32 元素的 absmax 完全在寄存器内完成，无 LDS |
| `cvt_scalef32_pk_fp4_bf16` 硬件打包 | gfx950 原生 bf16→fp4 缩放转换指令 |
| e8m0 指数 = 2 的幂 | 反量化无尾数误差，量化/反量化缩放自洽 |
| 512 CTA grid-stride | 与排序解耦后，量化可独立铺满全部 CU |
