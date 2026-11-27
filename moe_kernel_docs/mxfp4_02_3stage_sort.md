# MXFP4 M2/M3/M4 — `moe_3stage_sort`（三段并行排序）

源码：`aiter/csrc/kernels/mxfp4_moe/moe_aux/moe_3stage_sort.cuh`（`launch` 在 `:158`）/ `mxfp4_moe_aux.cu:133`（host）
host op：`aiter.mxfp4_moe_sort(..., prologue=1)`（`fused_moe.py:1128`）
trace 名：`sort_count_kernel_impl<385,9,16,1024>` + `sort_cumsum_kernel_impl<385,16,{32|128},1024>` + `sort_place_pad_kernel_impl<385,9,16,{32|128},1024>`

## 1. 概述

MXFP4 路径在**中/大 M（BM=32/128，threestage）**时用的路由排序，把单 block 串行排序拆成
**三个 kernel、多 CTA 并行**，以提升大 token 量下的排序吞吐：

1. **M2 `sort_count`**：`N_SORT_CTAS=16` 个 CTA 各自分块统计本段内每专家的 token 数，
   写出 `block_offsets[NE][16]`（每专家×每 CTA 的局部计数）。
2. **M3 `sort_cumsum`**：单 CTA 把 16 路局部计数求和成 `total_count[e]`，padding 到 `MB` 倍数得
   `cumsum`，把 `block_offsets` 原地改成「每 (expert,CTA) 的写入起始偏移」，并写 `sorted_expert_ids`、
   `masked_m`、`cumsum_tensor`。
3. **M4 `sort_place_pad`**：16 个 CTA 各自把本段 `(token,slot)` 用局部偏移 `atomicAdd` 放进
   `sorted_token_ids`，并填 padding 槽位。

逻辑与小 M 的 [mxfp4_01](mxfp4_01_sort_quant.md) 等价，但用「计数→前缀和→放置」三段拆分 + 多 CTA 分段，
避免单 block 在大 `M*topk` 上成为瓶颈。**它只排序，不做量化**（量化是独立的 [mxfp4_03](mxfp4_03_quant.md)）。

## 2. 何时被选中（M 条件）

- MXFP4 路径，`kernelName1` **不含** `INLINEQUANT`（即 `BM32_NT` 或 `BM128`）→ `prologue_name="threestage"`
  → `_mxfp4_moe_run` 调 `aiter.mxfp4_moe_sort(prologue=1)`（`fused_moe.py:1128`）。
- 被测 M：**256（MB=32）、4096 / 8192 / 16384（MB=128）**。

## 3. Launch 参数与 shape

`launch`（`:158`）顺序发射 3 个 kernel，全部 `THREADS=1024`，`N_SORT_CTAS=16`：

| kernel | grid | 关键模板参数 | 额外 scratch |
|---|---|---|---|
| `sort_count` | `<<<16,1024>>>` | `<NE=385,TOPK=9,16,1024>` | `block_offsets[NE*16]`、`real_counts[NE]`（host 临时分配，`aux.cu:135`）|
| `sort_cumsum` | `<<<1,1024>>>` | `<NE,16,MB,1024>` | — |
| `sort_place_pad` | `<<<16,1024>>>` | `<NE,TOPK,16,MB,1024>` | — |

输入/输出张量与 [mxfp4_01](mxfp4_01_sort_quant.md) §3 相同（`topk_ids/topk_weight` → `sorted_token_ids/
sorted_expert_ids/sorted_weights/reverse_sorted/m_indices/masked_m/cumsum_tensor`），但**不涉及** `a_quant/a_scale`。

## 4. 计算逻辑

### M2 `sort_count_kernel_impl`（`:10`）
- 每 CTA 负责 `total_pairs = M*topk` 的一个连续分段 `[cta*per_cta, end)`。
- LDS `local_count[NE]` 清零 → 段内 `atomicAdd(&local_count[topk_ids[i]],1)` → 写出
  `block_offsets[e*16 + cta] = local_count[e]`（列优先：专家在外、CTA 在内）。

### M3 `sort_cumsum_kernel_impl`（`:41`）
- `total_count[e] = Σ_c block_offsets[e*16+c]`；`padded_count[e]=round_up(total,MB)`；`masked_m[e]=padded`。
- 单线程串行前缀和得 `expert_starts[e]`、`cumsum_tensor[0]=Σpadded`。
- 把 `block_offsets[e*16+c]` 原地改写成「专家 e、第 c 段」的全局写入起点（`expert_starts[e]` 累加各段计数）。
- 写 `sorted_expert_ids[b]=e`（`b∈[expert_starts[e]/MB, expert_starts[e+1]/MB)`）。

### M4 `sort_place_pad_kernel_impl`（`:96`）
- 每 CTA 读回本段每专家的局部写起点 `local_offsets[e]=block_offsets[e*16+cta]` 与 `row_starts[e]`。
- 段内每条 `(token,slot)`：`sp=atomicAdd(&local_offsets[eid],1)` → 写
  `sorted_token_ids[sp]=(token&0xFFFFFF)|((slot&0xFF)<<24)`、`sorted_weights`、`m_indices=token`、`reverse_sorted[i]=sp`。
- padding：每专家 `[row_starts[e]+real_counts[e], row_starts[e+1])` 填 `token_id=M`、`weight=0`。

## 5. 与小 M 排序的差异

| | M1 `sort_quant`（≤128）| M2/3/4 `3stage`（≥256）|
|---|---|---|
| kernel 数 | 1 | 3 |
| 排序并行度 | 单 CTA | 16 CTA 分段 |
| 前缀和 | DPP wave-scan（block 内）| 单 CTA 串行 + 多段偏移 |
| 是否兼做量化/清零 | 是（清零 atomic 输出）| 否（量化独立，输出非 atomic）|

## 6. 关键优化点

| 技术 | 目的 |
|---|---|
| 计数/前缀和/放置三段拆分 | 大 `M*topk` 下用多 CTA 并行计数与放置，吞吐随 token 扩展 |
| `block_offsets` 列优先 + 段内偏移 | 每段 (expert,CTA) 独立偏移，放置阶段免全局原子争用 |
| padding 用越界 token id (`=M`) | 与 gemm1 的 buffer descriptor 越界丢弃配合，免有效行分支 |
| `MB` 倍数对齐 | 每个 M-block 恰好属于一个专家，GEMM 内无需混合专家 |
