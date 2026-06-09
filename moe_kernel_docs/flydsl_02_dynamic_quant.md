# FlyDSL F2 — `dynamic_per_group_scaled_quant_kernel`（大 M：纯激活量化）

源码：`aiter/csrc/kernels/quant_kernels.cu:29`（kernel）/ host `per_1x32_mx_quant_hip`（经 `dynamic_per_group_scaled_quant`，`:837`）
派发：`fused_dynamic_mx_quant_moe_sort` 的 split 分支（`quant.py:970-985`）
trace 名：`void aiter::dynamic_per_group_scaled_quant_kernel<std::bfloat16_t, opus::fp4_t, 32, 32, ...>`

## 1. 概述

FlyDSL 路径在**大 M** 时把「量化」与「scale 排序」拆开：本 kernel 只做**纯 per-group（每 32 元素）量化**，
把输入 `[M, cols]` bf16 量化为 `[M, cols/2]` fp4，并产出**未 shuffle、按原行序**的 per-token e8m0 scale。
随后由 [F3 `mxfp4_moe_sort_kernel`](flydsl_03_scale_sort.md) 把 scale 排序+shuffle 成 tile 布局。
优势：每个输入行只读一次（而 F1 融合 kernel 会按 topk 重复读同一 token 行）。

## 2. 何时被选中（M 条件）

`fused_dynamic_mx_quant_moe_sort`（`quant.py`）的 split 分支（与 [F1](flydsl_01_fused_quant_sort.md) 互斥）：
- **stage1 输入**：M ≥ 256（> 8*256/topk≈227）。
- **stage2 输入（中间量）**：仅当 stage1 kernel 名**不含** `_fp4`（中间量化未被 gemm1 融合）时执行，
  且 `token*topk > 8192`——即本基准的 **M=4096 / 8192 / 16384**（此时 stage1 与 stage2 各跑一次 F2，故 trace 里 cnt≈2×）。

## 3. Launch 参数与 shape

模板 `<bf16, fp4_t, thread_data_size=32, group_size=32, shuffle_scale=false, emit_e8m0=...>`，`<<<grid, block_size>>>`。

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `input` | `[ori_rows, ori_cols]` | bf16 | 激活（stage1 cols=7168 / stage2 cols=512）|
| `out`(out) | `[ori_rows, cols/2]` | fp4x2 | 量化结果（原行序）|
| `scale`(out) | `[ori_rows, cols/32]` | e8m0(uint8) | **未 shuffle** 的 per-token scale |

split 路径调用时 `shuffle=False`（`quant.py:982`），故此处 scale 不做 tile shuffle。

## 4. 计算逻辑（`:55-154`）

每个线程组（`num_thread_per_group = 32/thread_data_size`）处理一个 `(row, 32-列组)`：
1. `groupId → (x=row, y=列组)`；越界返回。
2. 向量读 `thread_data` 个 bf16，组内 `absMax = multithread_reduce(max|·|)`。
3. `inverted_scale = f32_to_e8m0_scale(absMax) * 0.25`（fp4）。组首线程写 `scale`（这里 `shuffle_scale=false` → 直接 `scale[groupId]=e8m0`）。
4. `store_vector` 用 `cvt_scalef32_pk_fp4_*` 把该组量化为 fp4 写 `out`。

与 [mxfp4_03 `quant_kernel_impl`](mxfp4_03_quant.md) 算法等价，差别在：本 kernel 是通用 quant 路径
（FlyDSL 用），mxfp4_03 是 MXFP4 专用路径（DPP 组内归约 + 专用布局）。

## 5. 关键优化点

| 技术 | 目的 |
|---|---|
| 每行只读一次 | 大 M 下比融合 kernel 省 topk× 的输入读 |
| 量化与 scale-shuffle 解耦 | 各自铺满 CU；scale-shuffle 交给 F3 的轻量字节搬运 |
| `multithread_reduce` 组内 amax | 32 元素 absmax 在寄存器/wave 内 |
| e8m0 = 2 的幂 + fp4 ×1/4 | 量化/反量化缩放自洽 |
