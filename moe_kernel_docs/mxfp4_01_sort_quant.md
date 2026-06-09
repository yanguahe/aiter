# MXFP4 M1 — `moe_sort_quant::sort_quant_kernel_impl`（小 M 单 kernel 排序）

源码：`aiter/csrc/kernels/mxfp4_moe/moe_aux/moe_sort_quant.cuh:308`（kernel）/ `mxfp4_moe_aux.cu:97`（host 派发）
host op：`aiter.mxfp4_moe_sort(..., prologue=0)`（`fused_moe.py:1146`）
trace 名：`void aiter::mxfp4_moe::moe_sort_quant::sort_quant_kernel_impl<385, 9, 16, 7168, 128, 1024, ...>`

## 1. 概述

MXFP4 路径在**小 M（BM=16，inline_quant）**时用来做路由排序的单一 kernel。它把
`topk_ids` 重排成「按专家分组、每组 padding 到 `BM` 倍数」的 `sorted_token_ids`，并产出
`sorted_expert_ids / sorted_weights / reverse_sorted / m_indices / masked_m / cumsum`。
由于走的是 `kSkipQuant=true` 的实例，**它不做激活量化**（量化被融合进 gemm1，见
[mxfp4_05](mxfp4_05_gemm1.md)）；同时它顺便把 gemm2 的 atomic 输出缓冲清零。

该模板 `sort_quant_kernel_impl` 是「排序子核 + 量化子核 + 清零子核」的合体，靠 `blockIdx.x` 与
两个编译期开关 `kSkipQuant/kSkipSort` 选择实际执行的部分。本 kernel（M1）= 排序 + 清零。

## 2. 何时被选中（M 条件）

- MXFP4 路径，`kernelName1` 含 `INLINEQUANT`（即 `BM16_INLINEQUANT`）→ `prologue_name="inline_quant"` →
  `_mxfp4_moe_run` 调 `aiter.mxfp4_moe_sort(prologue=0)`（`fused_moe.py:1146`）。
- 因为 gemm2 是 `ATOMIC`，`bf16_zero_out = atomic_output_buf`（非空），host 派发到
  `launch_sort_only_with_zero_init`（`mxfp4_moe_aux.cu:168-187`）。
- 被测 M：**4 / 8 / 16 / 32 / 64 / 128** 全部走此 kernel（3-kernel 流水线之一）。

## 3. Launch 参数与 shape

模板实参：`<NE=385, TOPK=9, MB=16, D_HIDDEN=7168, N_CTAS=128, THREADS=1024, kSkipQuant=true>`。
`<<<128, 1024>>>`（128 个 CTA，每 CTA 1024 线程）。

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `topk_ids` | `[M, 9]` | i32 | 每 token 的 topk 专家 id |
| `topk_weight` | `[M, 9]` | f32 | 对应路由权重 |
| `sorted_token_ids`(out) | `[max_sorted]` | i32 | fused id `(slot<<24)|token` |
| `sorted_expert_ids`(out) | `[max_sorted/16]` | i32 | 每个 M-block 的专家 id |
| `sorted_weights`(out) | `[max_sorted]` | f32 | 排序后的路由权重 |
| `reverse_sorted`(out) | `[M*9]` | i32 | `原 (token,slot) → sorted 位置` 的逆映射（供 scatter_reduce）|
| `m_indices`(out) | `[max_sorted]` | i32 | 排序后每行的 token_id（padding 处填 `M`）|
| `masked_m`(out) | `[NE]` | i32 | 每专家 padding 后的行数 |
| `cumsum_tensor`(out) | `[1]` | i32 | 总有效行数（padding 后）|
| `bf16_zero_out`(out) | `[M, 7168]` | bf16 | gemm2 atomic 输出缓冲，被清零 |

`max_sorted = round_up(M*topk + active*(BM-1), BM)`，`active = min(NE, M*topk)`（`fused_moe.py:1096-1098`）。

## 4. Grid / Block 映射

- **CTA 0**：执行 `sort_subkernel`（整个排序在单 block 内完成，`moe_sort_quant.cuh:320-326`）。
- **CTA 1..127**：只参与 `zero_init_bf16_out_impl`（清零 atomic 输出缓冲，`:330-332`）。
- 所有 128 个 CTA 都参与清零（`bf16_zero_out != nullptr` 时尾部统一清零）。

排序选单 CTA 是因为 NE=385、M 很小时 `M*topk` 也小（≤128×9=1152），单 block + LDS 计数足够快，
且能让 `count/cumsum/place` 共享 LDS、避免跨 block 同步。

## 5. 计算逻辑（`sort_subkernel`，`:192-216`）

1. **`count_tokens_per_expert`**（`:68`）：1024 线程把 `topk_ids` 的 `M*topk` 个条目按 4 路 `int4`
   向量化读入，`atomicAdd` 到共享内存 `count[NE]`。
2. **`parallel_cumsum`**（`:96`）：对 `round_up(count[e], BM)` 做 **DPP wave 内 inclusive scan**
   （`dpp_inclusive_scan_wave`，`:51`，用 `mov_dpp` 6 步移位完成 64-lane 扫描）+ 跨 wave 再扫一次，
   得到每专家在 `sorted` 中的起始偏移 `cumsum[e]`（已 padding 到 BM 倍数），并清零 `counter[e]`。
3. **`place_tokens`**（`:133`）：每条 `(token,slot)` 用 `atomicAdd(&counter[eid],1)` 拿到组内位置，
   写 `sorted_token_ids[sp] = (token & 0xFFFFFF)|((slot&0xFF)<<24)`、`m_indices[sp]=token`、
   `sorted_weights[sp]=topk_weight`、`reverse_sorted[i]=sp`。
4. **`fill_padding_gaps`**（`:167`）：每专家 `[start+cnt, end)` 的 padding 槽位写
   `sorted_token_ids = m_indices = M`（越界 token id，让 gemm1 的 `buffer_load` 被硬件丢弃）、
   `sorted_weights=0`；并把每个 M-block 的 `sorted_expert_ids[b]=e`。
5. 写 `masked_m[e]=cumsum[e+1]-cumsum[e]`、`cumsum_tensor[0]=cumsum[NE]`。

清零子核 `zero_init_bf16_out_impl`（`:13`）：把 `[M,7168]` bf16 缓冲按 `int4`（16B）向量化 grid-stride 清零。

## 6. 关键优化点

| 技术 | 目的 |
|---|---|
| 三件事（排序/量化/清零）合一模板 | 用 `blockIdx.x` + `constexpr` 开关复用代码，减少 kernel 数量 |
| DPP wave-scan 前缀和 | 385 专家的 cumsum 在寄存器内完成，无需多轮 LDS 往返 |
| `int4` 向量化计数/清零 | 单指令处理 4 个 topk id / 8 个 bf16，带宽友好 |
| padding 用越界 token id (`=M`) | gemm1 直接靠 buffer descriptor 越界丢弃，无需分支判断有效行 |
| 清零与排序在同一次 launch 重叠 | CTA0 排序时，其余 127 CTA 并行清零 atomic 输出缓冲 |
