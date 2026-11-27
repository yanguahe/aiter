# FlyDSL F5 — `mfma_moe1_silu_mul_afp4_wfp4_*`（GEMM1：gate/up + SiLU·mul，可融合中间量化）

源码：`aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`（`compile_mixed_moe_gemm1:92`，kernel `moe_gemm1:437`，名构造 `:266-269`）
host：`flydsl_moe_stage1`（`aiter/aiter/ops/flydsl/moe_kernels.py:679`）→ `_flydsl_stage1_wrapper`（`fused_moe.py:845`）
配置名：`flydsl_moe1_afp4_wfp4_bf16_t{M}x{N}x{K}[_w{wpe}][_kb{kb}][_bnt{n}][_fp4][_xcd{n}]`
trace（编译符号）名：`mfma_moe1_silu_mul_afp4_wfp4_{out}_t{M}x{N}x{K}_pm{pm}[_fp4q][_sort][_async][_xcd{n}]_v32`

> 注意「配置名」（CSV/选择用）与「编译符号名」（trace 显示）是两套命名，一一对应；例：
> 配置 `flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w3_fp4` → 符号 `mfma_moe1_silu_mul_afp4_wfp4_fp4_t32x128x256_pm1_fp4q_sort_async_v32`。

## 1. 概述

FlyDSL 路径的 stage1 GEMM（DSL 生成）。对每个「M-block × N-tile × 专家」计算 fp4×fp4 的
gate 与 up 两路（`SEPARATED` 模式两个 B 流、两个 f32 累加器），epilogue 做 `y = SiLU(gate)·up`；
当符号含 **`_fp4q`** 时，epilogue 进一步把中间量化为 fp4 并把 e8m0 写成 stage2 的 sorted-tile 布局
（融合了 [F2](flydsl_02_dynamic_quant.md)+[F3](flydsl_03_scale_sort.md) 对中间量的处理），返回 `(out_packed, out_scale_sorted)`。

## 2. 何时被选中（M 与后缀）

由 tuned CSV 给出 `kernelName1`（`fused_moe.py:1666`，`_flydsl_stage1_wrapper`）。被测观察：

| M | 配置名（节选）| 含 `_fp4`？ | 含义 |
|--:|---|---|---|
| 4 | `...t32x128x256_w3_fp4` | 是 | tile 32×128×256，fuse 中间 fp4 量化 |
| 8 | `...t32x64x256_w4_fp4` | 是 | |
| 64/128 | `...t32x128x256_w2_fp4` / `...t32x64x256_w2_fp4` | 是 | |
| 256 | `...t64x128x256_w3_fp4` | 是 | tile_m=64 |
| 4096 | `...t128x128x256_w4` | **否** | 不 fuse → 中间量化交给独立 [F2](flydsl_02_dynamic_quant.md)+[F3](flydsl_03_scale_sort.md) |
| 8192 | `...t128x128x256_w2_xcd4` | 否 | XCD swizzle |
| 16384 | `...t64x128x256_w4_bnt0_xcd4` | 否 | tile_m=64, XCD |

`_fp4`（配置）↔ `_fp4q_sort`（符号）= 是否把中间量化融进 gemm1（`fuse_quant="fp4"`，`fused_moe.py:1673/2192`）。

## 3. Launch 参数与 shape（`flydsl_moe_stage1`，`moe_kernels.py:679-1028`）

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `a` | `[token_num, model_dim/2]`（fp4 时 model_dim 翻倍还原）| fp4x2 | 激活（来自 [F1](flydsl_01_fused_quant_sort.md)/[F2](flydsl_02_dynamic_quant.md)）|
| `a1_scale` | `[sorted_size, K/32]` | e8m0 | sorted-tile 激活 scale |
| `w1` | `[E, 2*512, 7168]` 预 shuffle | fp4x2 | gate||up 权重 |
| `w1_scale` | `[E*1024, 7168/32]` | e8m0 | |
| `sorted_token_ids/sorted_expert_ids/num_valid_ids` | — | i32 | 排序结果（[F4](flydsl_04_opus_sorting.md)）|
| `out` | bf16 `[tok,topk,512]` 或 fp4 `[tok,topk,256]`+`out_scale` | — | 中间量（`_fp4q` 时返回 (packed, scale)）|

## 4. Grid / Block / 计算逻辑

- block：`num_waves = min(4, tile_n/32)`，`threads = num_waves*64`（`mixed:152`）。
- grid：`by`=N-tile（inter_dim）、`bx_persist`=M-block、`bz`=k_batch（split-K）；separated 模式 `gx=ceil(inter/tile_n/2)`（一个 WG 同时算 gate 与 up）（`:2710-2746`）。`persist_m` 默认 1（stage1，`:125-127`）。
- MFMA：`rocdl.mfma_scale_f32_16x16x128_f8f6f4`（`cbsz=4`/`blgp=4` 选 fp4 子格式，`:1354`），按 e8m0 在 MFMA 内反量化；gate/up 各自链入 SrcC。
- K-loop：LDS ping-pong 双缓冲（`lds_x_ping/pong`，`:1483-1523`），`sched_barrier(0)` 分隔阶段，B-major 调度（B 驻寄存器、A 在 LDS 复用）。
- `_async`：`raw_ptr_buffer_load_lds` 直接 global→LDS DMA（绕过 VGPR，`:1142`）。
- epilogue（cshuffle）：LDS C-shuffle 重排 → `_act_vec4` 做 SiLU·mul（`_silu_mul_vec4:1942`）；`_fp4q` 时在 `store_pair`（`:2257`）内对每 32 元素 XOR-shuffle 求 amax → e8m0 → `cvt` 量化 fp4，并按 `(d0..d5)` 6 维 swizzle 写 `out_scale_sorted`（`:2416`，stage2-ready）。
- `_kb{k_batch}`（split-K）：K 切到 `block.z`，gate/up partials `atomic fadd` 进零缓冲，再单独 `silu_and_mul` 融合归约（`:2447`，host `:774`）。

## 5. 后缀解码（符号名，`mixed:266-269`）

| token | 含义 |
|---|---|
| `silu_mul` | gate/up 激活·乘（字面量，swiglu 时也叫此名）|
| `t{M}x{N}x{K}` | tile_m × tile_n × tile_k |
| `pm{n}` | persist_m（每 WG 的 M-tile 数；stage1 默认 1）|
| `_fp4q`/`_fp8q` | 融合 fp4/fp8 中间量化（返回 (packed, scale)）|
| `_sort` | 写 sorted-tile e8m0 scale buffer |
| `_async` | global→LDS 直接 DMA |
| `_sk{n}` | split-K（atomic partial reduce；配置名作 `_kb{n}`）|
| `_gui`/`_go` | gate-up-interleave / mock-gate-only |
| `_xcd{n}` | XCD swizzle 组宽 |
| `_v32` | 内部版本号（字面量）|

## 6. 关键优化点

| 技术 | 目的 |
|---|---|
| `_fp4q` 融合中间量化（小/中 M）| 省去独立 [F2]+[F3] 对中间量的两次 launch |
| 输出 sorted-tile fp4+e8m0 | stage2 直接 tile 取，无需 reshuffle |
| LDS ping-pong + B-major 调度 | 隐藏 VMEM 延迟、MFMA 背靠背 |
| `_async` 直 DMA 到 LDS | 绕过 VGPR，省寄存器/拷贝 |
| `mfma_scale_f32_16x16x128_f8f6f4` | gfx950 原生 fp4×fp4 缩放 MFMA |
| XCD swizzle（大 M）| 跨 8 XCD 负载均衡 |
| split-K（`_kb`）| 小 token、大 K 时提高 CU 占用 |
