# FlyDSL F6 — `mfma_moe2_afp4_wfp4_bf16_cshuffle_*`（GEMM2：down 投影 + topk 归约策略）

源码：`aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`（`compile_mixed_moe_gemm2:2752`，kernel `moe_gemm2:3028`，名构造 `:2995-2999`）
host：`flydsl_moe_stage2`（`aiter/aiter/ops/flydsl/moe_kernels.py:1029`）→ `_flydsl_stage2_wrapper`（`fused_moe.py:909`）
配置名：`flydsl_moe2_afp4_wfp4_bf16_t{M}x{N}x{K}_{atomic|reduce}[_persist][_bnt{n}][_sbm{N}][_xcd{n}]`
trace（编译符号）名：`mfma_moe2_afp4_wfp4_bf16_cshuffle_t{M}x{N}x{K}_vscale_fix3[_persist_cu256|_pm{n}][_sbm{N}][_xcd{n}]`

> `atomic`/`reduce` 是 `accumulate` 标志，**不进编译符号名**（仅进 Python lru-cache key）；故符号名里看不到 reduce/atomic。

## 1. 概述

FlyDSL 路径的 stage2 GEMM（DSL 生成）。对每个「(token,slot) → 专家」计算 fp4×fp4 的 down 投影
（`K=inter_dim=512`，`N=model_dim=7168`），用 **cshuffle** LDS 重排 epilogue，并按 `sorted_weights`
（编译符号里的 **`vscale`**）在 epilogue 内乘上路由权重。topk 归约由 `accumulate` 决定：
- **atomic**（小/中 M）：`atomic fadd` 累加到 `out[token]`，topk slot 自然汇总。
- **reduce**（大 M）：写每个 `out[token*topk+slot]`，再由 host `torch.sum`（[F7](flydsl_07_topk_reduce.md)）归约。

## 2. 何时被选中（M 与后缀）

由 tuned CSV 给出 `kernelName2`。被测观察：

| M | 配置名（节选）| 模式 | 归约 |
|--:|---|---|---|
| 4 | `...t32x256x256_atomic` | atomic | epilogue 原子加 |
| 8/16 | `...t32x128x256_atomic_bnt2` | atomic | |
| 64/128 | `...t32x256x256_atomic_bnt2_persist` | atomic+persist | |
| 256 | `...t32x256x256_atomic_bnt2_sbm64` | atomic, sort_block_m=64 | |
| 4096 | `...t64x256x256_reduce_persist_sbm128` | **reduce** | → torch.sum [F7] |
| 8192 | `...t64x256x256_reduce_sbm128` | reduce | → [F7] |
| 16384 | `...t64x256x256_reduce_xcd4` | reduce + XCD | → [F7] |

`reduce` 模式选择见 host `accumulate = mode != "reduce"`（`moe_kernels.py:1074`）；`persist` 在 `m_blocks>256` 时自动开（`:1105-1110`）。

## 3. Launch 参数与 shape（`flydsl_moe_stage2`，`moe_kernels.py:1029-1190`）

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `inter_states` | `[token_num, topk, 512/2]` | fp4x2 | 中间量（gemm1 `_fp4q` 或 [F2]+[F3]）|
| `a2_scale` | `[sorted_size, 512/32]` | e8m0 | sorted-tile |
| `w2` | `[E, 7168, 512]` 预 shuffle | fp4x2 | down 权重 |
| `w2_scale` | `[E*7168, 512/32]` | e8m0 | |
| `sorted_token_ids/sorted_expert_ids/num_valid_ids` | — | i32 | 排序结果 |
| `sorted_weights` | `[sorted_size]` | f32 | 路由权重（epilogue `vscale` 乘）|
| `out` | atomic: `torch.zeros[tok,7168]` / reduce: `torch.empty[tok*topk,7168]` | bf16 | 输出/暂存 |

## 4. Grid / Block / 计算逻辑

- block：256 线程（`:2920`）。grid：`by`=N-tile（model_dim），`gx=ceil(model_dim/tile_n)`；`gy=cu_num`（persistent）或 `ceil(size_expert_ids/persist_m)`（`:4535-4549`）。
- **persistent（`_persist_cu256`）**：`grid_y=cu_num=256`，每 CTA 轮转连续 `tiles_per_block` 个 M-tile（相邻 tile 同专家 → B 权重 L2 复用，`:3270-3273`）。
- MFMA：同 fp4 `mfma_scale_f32_16x16x128_f8f6f4`（`cbsz=4`/`blgp=4`，`:2851`），单累加器（down 投影无 gate/up）；K 只 2 个 tile（512/256），LDS ping-pong 双缓冲。
- **cshuffle epilogue**（`mfma_epilogues.py:85`）：把分散的 MFMA per-lane 片段经 `lds_out` 重排成连续 N 向量再写出。
- **`vscale`**：在 `write_row_to_lds` 内对每行乘 `sorted_weights`（`v=v*tw`，`:4359`），由 `doweight_stage2`（= `sorted_weights is not None`）开启。
- **归约**：
  - atomic：`store_pair` 用 `AtomicRMWOp fadd`（agent scope）累加到 `out[token]` 行（`:4417`）。
  - reduce：plain non-temporal store 到 `out[token*topk+slot]`（`:4402`），host 再 `torch.sum(view(tok,topk,H),dim=1)`（`:1187`）。
- **`_sbm{N}`（sort_block_m）**：允许 stage2 用比排序 block_m 更小的 tile_m（`sort_blk = bx_m >> log2(sort_block_m)`，`:3312`）。

## 5. 后缀解码（符号名，`mixed:2995-2999`）

| token | 含义 |
|---|---|
| `cshuffle` | LDS C-shuffle epilogue（f16/bf16 恒开）|
| `vscale` | epilogue 内乘 `sorted_weights`（路由权重）|
| `fix3` | 内部版本/迭代标记（字面量）|
| `_persist_cu{cu}` | 持久网格，grid_y=cu_num（MI355X=256）|
| `_pm{n}` | 旧式持久（每 WG 的 M-tile 数）|
| `_sbm{N}` | sort_block_m≠tile_m |
| `_xcd{n}` | XCD swizzle |
| *(无)* | atomic vs reduce 不在符号名内 |

## 6. 关键优化点

| 技术 | 目的 |
|---|---|
| atomic 归约（小/中 M）| topk 归约靠 `atomic fadd` 完成，省独立 reduce |
| reduce + torch.sum（大 M）| 大 M 下 atomic 争用大，改写每 slot + 批量 sum（[F7](flydsl_07_topk_reduce.md)）|
| 持久网格 + 连续 M-tile | 相邻 tile 同专家、复用 B 的 L2 |
| cshuffle LDS epilogue | 把分散 MFMA 片段重排成连续写，写带宽友好 |
| `vscale` 融合权重 | 路由权重在 epilogue 乘，省一遍读写 |
| `_sbm` 解耦 tile_m / 排序块 | stage2 用更小 tile_m 提占用，同时复用大 block_m 排序 |
| XCD swizzle（大 M）| 跨 8 XCD 负载均衡 |
