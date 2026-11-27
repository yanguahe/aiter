# FlyDSL F4 — `opus_moe_sorting_entry<...>`（opus 路由排序，single / multi-phase）

源码：`aiter/csrc/include/moe_sorting_opus.h`（kernel entry `:107`，dispatch `:3497/3519`）
host op：`aiter.moe_sorting_opus_fwd(...)`（`fused_moe.py:110`，经 `_moe_sorting_impl:70`）
trace 名：`opus_moe_sorting_entry<MoeSortingKernel<...>>`、`<MoeSortingMultiPhaseKernel_P0_v1/_P0_v2/_P1/_P23<...>>`、`<MoeSortingClearWorkspaceKernel<...>>`

## 1. 概述

FlyDSL 路径的路由排序（等价于 MXFP4 路径的 [sort_quant](mxfp4_01_sort_quant.md)/[3stage_sort](mxfp4_02_3stage_sort.md)）。
把 `topk_ids[M,9]` 重排成按专家分组、padding 到 `block_size` 倍数的 `sorted_ids`（fused id `slot<<24|token`），
并产出 `sorted_weights / sorted_expert_ids / num_valid_ids / moe_buf`。
opus 用 **LDS mesh + DPP wave-cumsum** 实现计数与前缀和，按 token 量自动在 **单 kernel** 与 **多相（multi-phase）** 间切换。

## 2. 何时被选中（随 M 变体）

`moe_sorting_get_workspace_size`（`:1378`）按 `moe_sorting_is_oneshot`（`:1348`）决定：
`is_oneshot = tokens ≤ sub_token`，`sub_token` 由 LDS 容量推出（NE=385 时 mesh 每行 386 ints）。

| M | 路径 | 实际 kernel（按序）| trace 依据 |
|--:|---|---|---|
| ≤ ~32 | **single** | `MoeSortingKernel`（1 个，oneshot）| M=4..32 |
| 64 ~ 1024 (`tokens<2048`) | **mp-small** | `MoeSortingMultiPhaseKernel_P0_v2` + `_P23`（2 个）| M=64/128/256 |
| ≥ 2048 (`tokens≥2048`) | **mp-full** | `ClearWorkspaceKernel` + `_P0_v1` + `_P1` + `_P23`（4 个）| M=4096/8192/16384 |

派发逻辑：`moe_sorting_opus`（`:3497`）→ oneshot 走 single；否则 `moe_sorting_opus_mp`（`:3519`）按
`tokens<2048` 选 `MP_DISPATCH_SMALL_`(P0_v2+P23) 或 `MP_DISPATCH_`(ClearWS+P0_v1+P1+P23)。

## 3. Launch 参数与 shape（`_moe_sorting_impl`，`fused_moe.py:85-125`）

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `topk_ids` | `[M, 9]` | i32 | 路由专家 id |
| `topk_weights` | `[M, 9]` | f32 | 路由权重 |
| `sorted_ids`(out) | `[max_num_tokens_padded]` | i32 | fused id |
| `sorted_weights`(out) | 同上 | f32 | 排序后权重 |
| `sorted_expert_ids`(out) | `[max_num_m_blocks]` | i32 | 每 block 专家 |
| `num_valid_ids`(out) | `[2]` | i32 | [0]=padding 后行数, [1]=实际行数 |
| `moe_buf`(out) | `[M, model_dim]` | bf16 | 输出缓冲（预分配）|
| `workspace` | mp 时分配 | uint8 | multi-phase 跨相暂存（single 时为 None）|

`max_num_tokens_padded = M*topk + NE*block_size - topk`（`fused_moe.py:85`）。

## 4. 各 kernel 职责

- **`MoeSortingKernel`（single，`:448`）**：在单 kernel 内用 LDS `smem_tokens[sub_token][NE]` mesh 标记每 token 命中的专家，
  分批（`i_token += sub_tokens`）做 `wave_cumsum`（DPP 16-row scan，`:548`）累加位置，直接写 `sorted_ids/sorted_weights/cumsum`。
- **`MoeSortingClearWorkspaceKernel`（`:1407`）**：mp-full 前清零跨相 workspace（mesh/cumsum/sem）。
- **`MoeSortingMultiPhaseKernel_P0_v1/_P0_v2`（`:1543/1665`）**：相 0，分块写 mesh 计数（v1 用于大 M、v2 用于小 M）。
- **`_P1`（`:1857`）**：相 1，跨块归约 + 专家 cumsum。
- **`_P23`（`:2640`）**：相 2+3 合并，放置 token 到 `sorted_ids` + padding + 写 `sorted_expert_ids`（需 LDS，`GetSmemSize`）。

## 5. 关键优化点

| 技术 | 目的 |
|---|---|
| oneshot 单 kernel（小 M）| 全程在 LDS + 寄存器内完成，无跨 kernel workspace |
| multi-phase（大 M）| 计数/归约/放置拆相 + 多 CTA，token 量大时吞吐可扩展 |
| DPP `wave_cumsum`（16-row）| 前缀和在 wave 内完成，省 LDS 往返 |
| `ds_bpermute` 跨 lane 汇总 | 组内位置传播无需 atomics |
| LDS mesh `[sub_token][NE]` | 用 LDS 容量自适应 sub_token，决定 oneshot 阈值 |
