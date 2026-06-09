# FlyDSL F1 — `fused_mx_quant_moe_sort_kernel`（小 M：融合激活量化 + scale 排序）

源码：`aiter/csrc/kernels/quant_kernels.cu:1685`（kernel）/ host `fused_dynamic_mx_quant_moe_sort_hip`（`:1856`）
派发：`fused_dynamic_mx_quant_moe_sort`（`aiter/aiter/ops/quant.py:878`，`use_fused` 分支 `:958`）
被 `fused_moe_2stages` 在 stage1 输入量化处调用（`fused_moe.py:2125`）
trace 名：`void aiter::fused_mx_quant_moe_sort_kernel<std::bfloat16_t, opus::fp4_t, 256, 32>`（`<DTYPE_I, DTYPE_O, BLOCK_SIZE, THREAD_DATA>`）

## 1. 概述

FlyDSL 路径在**小 M** 时把「激活 bf16→mxfp4 量化」与「把 e8m0 scale 写成 sorted-tile 布局」融合进
**一个 kernel**：每个 sorted 行从 `sorted_ids` 还原出对应 token，加载该 token 的输入行，按 32 元素一组量化为
fp4（写 `out`，token 顺序），并把该组 e8m0 写到 `scale` 的 **shuffle 后 sorted tile 位置**（供 FlyDSL gemm 直接取用）。
省去「先量化全部、再单独排序 scale」的两次 launch（即把 [F2](flydsl_02_dynamic_quant.md)+[F3](flydsl_03_scale_sort.md) 合一）。

## 2. 何时被选中（M 条件）

`fused_dynamic_mx_quant_moe_sort`（`quant.py:953`）的 `use_fused` 判定（topk=9）：
- **stage1 输入**：`M ≤ 8*256/topk ≈ 227` → 融合（即 M=4/8/16/32/64/128）。
- **stage2 输入（中间量）**：`M_rows = token*topk ≤ 8*1024 = 8192` 时融合；本基准 stage1 多数已用 `_fp4q` 融合中间量化，故 F1 主要出现在小 M 的 stage1 输入量化。
- `M ≥ 256`（stage1）改走 [F2](flydsl_02_dynamic_quant.md)+[F3](flydsl_03_scale_sort.md) 拆分路径。

## 3. Launch 参数与 shape

模板 `<bf16, fp4_t, BLOCK_SIZE=256, THREAD_DATA=32>`；`grid = num_cu*blocks_per_cu`（persistent）或 `num_blocks`（`:1808-1812`）。

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `input` | `[token_num, cols]`（stage1 cols=7168）| bf16 | 激活 |
| `sorted_ids` | `[sorted_size]` | i32 | fused id（低 24 位 = token，高 8 位 = slot）|
| `num_valid_ids` | `[..]` | i32 | 有效行数 |
| `out`(out) | `[M, cols/2]` | fp4x2 | 量化结果（**按原 token 顺序**写）|
| `scale`(out) | `[pad32(sorted_size), pad8(cols/32)]` | e8m0(uint8) | **sorted-tile shuffle 布局** |

## 4. 计算逻辑（`:1735-1800`）

按 `(block_m, sub_block_m)` 网格遍历 sorted 行块：
1. 由 `sorted_ids[strided_idx]` 取 `token = id & 0xFFFFFF`、`slot = id >> 24`；越界则 `token=num_tokens`（跳过）。
2. `offset_base = token`（stage1，topk==1）或 `token*topk + slot`（stage2）。加载该行 `cols` 个 bf16。
3. 每 `num_thread_per_group = 32/THREAD_DATA` 个线程协作一组 32 元素：`absMax = multithread_reduce(max|·|)`。
4. `row_scale = f32_to_e8m0_scale(absMax) * 0.25`（fp4 max=6→floor_pow2=4，故 ×1/4），取指数字节 `e8m0 = (bits>>23)&0xFF`。
5. **scale 写入 shuffle 位置**：`addr = mx_scale_shuffle_idx(scaleN_pad, sorted_row, scale_k)`，`scale[addr]=e8m0`（`:1790`）。
6. `scaled_quant_vgpr_impl` 用 `cvt_scalef32_pk_fp4_*` 把该组量化为 fp4 写 `out[offset_base*cols + ...]`（token 顺序）。

要点：**fp4 数据按 token 原序写**（gemm 用 sorted_ids gather 读），**scale 按 sorted-tile 序写**（gemm 直接 tile 取）。

## 5. 关键优化点

| 技术 | 目的 |
|---|---|
| 量化 + scale 排序融合 | 小 M 省一次 launch（vs F2+F3）|
| `mx_scale_shuffle_idx` 直写 tile 布局 | gemm 无需再 reshuffle scale |
| persistent grid（`num_cu*blocks_per_cu`）| 小 M 下用持久块铺满 CU |
| e8m0 = 2 的幂 + fp4 ×1/4 | 量化/反量化缩放自洽，无尾数误差 |
| `multithread_reduce` 组内 amax | 32 元素 absmax 在寄存器/wave 内完成 |
