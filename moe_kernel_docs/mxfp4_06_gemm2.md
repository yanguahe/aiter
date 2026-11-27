# MXFP4 M8 — `gemm2::kernel`（GEMM2：down 投影 + topk 归约策略）

源码：`aiter/csrc/kernels/mxfp4_moe/gemm_a4w4/gemm2_a4w4.cuh:51`（kernel）/ epilogue `common/mxfp4_epilogs.hpp:149/218/245`
host op：`aiter.mxfp4_moe_gemm2_a4w4(...)` 或 `..._mxfp4out(...)`（`fused_moe.py:1275 / 1252`）
trace 名：`void aiter::mxfp4_moe::gemm2::kernel<655360, 385, 512, 7168, 9, {16|32|128}, (EpilogPolicy)...>`
（模板 = `<MAX_M, NUM_EXPERTS, K=512, N_OUT=7168, TOPK=9, BM, kEpilog, kUseNT, kXcdSwizzle, kMxfp4Out>`）

## 1. 概述

MXFP4 路径的 stage2 GEMM：对每个「(token,slot) → 专家」计算 `inter_q × w2ᵀ`（`K=inter_dim=512`，
`N_OUT=model_dim=7168`），得到该 (token,slot) 的 down 投影。**topk 归约策略由 epilogue 决定**：

- **Atomic**（`EpilogPolicy::Atomic`，BM∈{16,32,64}）：epilogue 先乘 `sorted_weights`，再用
  `global_atomic_fadd_v2bf16` **原子加到 `out[token]`**（topk 个 slot 自然累加）→ 无需 scatter_reduce。
- **Nonatomic**（`EpilogPolicy::Nonatomic`，BM=128）：写出每个 sorted 行到 `flat_out`（不乘权重），
  之后由 [scatter_reduce](mxfp4_07_scatter_reduce.md) 做 topk 加权求和。
  - 普通 nonatomic：`flat_out` 为 bf16；
  - **`kMxfp4Out`**：`flat_out` 压成 fp4（`flat_out_q`）+ e8m0（`flat_out_scale`），scatter_reduce_q 再解包
    （读取量 ↓ ~3.8×，仅 M≥8192 启用）。

## 2. 何时被选中（M 条件）

由 `kernelName2` 解析（`_parse_mxfp4_g2_kname`）：

| M | BM | epilog | kMxfp4Out | 后续 |
|--:|--:|---|---|---|
| 4 / 8 / 16 / 32 / 64 / 128 | 16 | **Atomic** | — | 无 scatter_reduce |
| 256 | 32 | Atomic | — | 无 scatter_reduce |
| 4096 | 128 | Nonatomic | false | → `scatter_reduce`(bf16) [M9] |
| 8192 / 16384 | 128 | Nonatomic | true | → `scatter_reduce_q`(mxfp4) [M10] |

`__launch_bounds__(256, Nonatomic?1:(BM==16?4:2))`。Nonatomic ⇒ `kUseAGPR && kPersistent`（持久网格）。

## 3. Launch 参数与 shape

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `A_q`(=inter_sorted_quant) | `[max_sorted, 512/2]` | fp4x2 | gemm1 输出的中间量 |
| `A_scale` | tile 布局 | e8m0 | 中间量 scale |
| `B_q`(=w2) | `[E, 7168, 512/2]` | fp4x2 | 预 shuffle 的 down 权重 |
| `B_scale`(=w2_scale) | `[E*7168, 512/32]` | e8m0 | 权重 scale |
| `sorted_token_ids` | `[max_sorted]` | i32 | fused id（atomic 时还原 token 做累加地址）|
| `sorted_weights` | `[max_sorted]` | f32 | 路由权重（atomic epilog 内乘）|
| `out_bf16`(atomic) / `flat_out`(nonatomic) | `[M,7168]` / `[max_sorted,7168]` | bf16 | 输出 |
| `flat_out_scale`(mxfp4out) | `[max_sorted,7168/32]` | uint8 | mxfp4out 模式的 e8m0 |

`K=512 → K_TILES_TOTAL=512/256=2`（K 很短，epilogue 占比相对高）。

## 4. Grid / Block 映射

- `pid → (m_block_idx, n_block_idx)`，`num_n_blocks = 7168/256 = 28`；Nonatomic 用持久网格（grid_y=cu_num）轮转 M-tile。
- LDS 布局按 atomic/nonatomic 分两套（`LDSLayout`，`:34/42`）：nonatomic 多一块 `s_Ascale`，并用 AGPR 累加。

## 5. 计算逻辑

主循环同 gemm1（fp4×fp4 `mfma_scale_f32_16x16x128_f8f6f4`，A 在 LDS、B 寄存器，双缓冲），但 K 仅 2 个 tile。
区别全在 **epilogue**（`mxfp4_epilogs.hpp`）：

- **`apply_atomic_bf16_epilog`（`:149`）**：对每个 sorted 行取 `weight=sorted_weights[sorted_pos]`，把 `acc*weight`
  转 bf16，用 `atomic_pk_add_bf16`（`global_atomic_fadd_v2bf16`）累加到 `out[token]`（输出缓冲已被
  [mxfp4_01](mxfp4_01_sort_quant.md) 预清零）。topk 个 slot 因 token 相同自动汇总。
- **`apply_bf16_flat_epilog_bm128`（`:218`）**：直接把 `acc` 转 bf16 写 `flat_out[sorted_pos]`（不乘权重）。
- **`apply_mxfp4_flat_epilog_bm128`（`:245`）**：对每 32 个结果求 amax→e8m0，`cvt_scalef32_pk_fp4_f32` 压成 fp4
  写 `flat_out_q` + `flat_out_scale`。

## 6. 关键优化点

| 技术 | 目的 |
|---|---|
| Atomic epilogue（小/中 M）| topk 归约直接靠 `atomic_fadd_v2bf16` 完成，省掉一个 scatter_reduce kernel |
| Nonatomic + 持久网格（大 M）| 大 M 下 atomic 争用严重，改为写 flat + 独立归约；持久网格让相邻 M-tile 同专家、复用 B 的 L2 |
| `kMxfp4Out`（M≥8192）| 中间结果压 fp4 暂存，scatter_reduce 读取量 ↓ ~3.8×（lossy，但只在大 M 收益>损失时启用）|
| AGPR 累加（nonatomic）| 释放 VGPR 压力 |
| K=512 短 + epilogue 重 | down 投影 K 小，优化重点在归约/写出而非 K-loop |

## 7. 与 FlyDSL gemm2（[F6](flydsl_06_moe2_gemm.md)）的差异对比

两者算的是**同一份数学**：按「(token,slot) → 专家」做 fp4×fp4 缩放 MFMA 的 down 投影
（`K=inter_dim=512`、`N=model_dim=7168`），cshuffle epilogue 重排后按 `sorted_weights` 乘路由权重，
再做 topk 归约。但在**接口、功能、实现逻辑**三层上差异显著。下面 M8 指本 kernel（`gemm2_a4w4.cuh`），
F6 指 FlyDSL kernel（`mixed_moe_gemm_2stage.py`）。

### 7.1 接口差异

| 维度 | mxfp4 HIP gemm2（M8）| FlyDSL gemm2（F6）|
|---|---|---|
| 实现形态 | 单个 C++ `__global__` 模板（`gemm2_a4w4.cuh:51`），hipcc 预编译 | Python 运行时构建 MLIR IR、JIT 编译并缓存（`compile_mixed_moe_gemm2:2882`，`~/.flydsl/cache`）|
| 配置方式 | `kernelName` 解析成模板参数 `<MAX_M,NE,K,N_OUT,TOPK,BM,kEpilog,kUseNT,kXcdSwizzle,kMxfp4Out>` | Python kwargs（`tile_m/n/k`、`mode`、`flat_output`、`flat_mxfp4`、`sort_block_m`、`persist`…）|
| host 调用约定 | **void**，结果写预分配 buffer；分两个 op：`mxfp4_moe_gemm2_a4w4`（atomic/bf16-flat，`mxfp4_moe.py:160`）与 `..._mxfp4out`（fp4-flat，`:128`）| `flydsl_moe_stage2(...)`，**有返回值** `out`；`out=None` 内部分配（`moe_kernels.py:1050`）|
| 中间量（A=inter）布局 | `inter_sorted_quant` `[max_sorted, 512/2]`，**sorted-major、A 读 sorted-direct**（行=排序位）| `inter_states` `[token_num, topk, 512/2]`，**token-major、经 `sorted_token_ids` gather**（排序位→(token,slot)）|
| 归约模式入口 | 模板 `kEpilog`（`Atomic`/`Nonatomic`）+ `kMxfp4Out` | Python `mode`（`atomic`/`reduce` → `accumulate`）+ `flat_output`/`flat_mxfp4` |
| 大 M 归约衔接 | flat → 同仓 HIP [scatter_reduce](mxfp4_07_scatter_reduce.md)(_q)（[M9]/[M10]）| 原生 `reduce` → host `torch.sum`（[F7](flydsl_07_topk_reduce.md)）；亦可 `flat_mxfp4` 接 HIP scatter_reduce_q |
| weight 入参 | `sorted_weights`（atomic epilogue 内乘）| `sorted_weights`（编译符号 `vscale`，cshuffle 内乘）|

> 关键契约差异：**stage1↔stage2 的中间量布局不同**——M8 的 inter 是 **sorted-major**（gemm1 直接写
> sorted 行、gemm2 sorted-direct 读）；F6 的 inter 是 **token-major** `[tok,topk]`（gemm2 用 sorted id
> 间接 gather）。这是两条流水线无法直接互换 stage 的根因。

### 7.2 功能差异

| 功能 | mxfp4 HIP gemm2（M8）| FlyDSL gemm2（F6）|
|---|---|---|
| topk 归约策略 | `Atomic`（BM∈{16,32,64}）+ `Nonatomic`-flat（BM=128）| `atomic` + `reduce`（写 `[tok,topk]` 后 host `torch.sum`）|
| 大 M flat 输出 | bf16-flat（→ [M9] scatter_reduce）/ `kMxfp4Out` fp4-flat（→ [M10] scatter_reduce_q，M≥8192）| `flat_output`(bf16) / `flat_mxfp4`(fp4+`out_scale`)——**后加**，专为接 HIP scatter_reduce(_q) |
| 持久网格 | `Nonatomic` ⇒ **恒** `kPersistent`（grid_y=cu_num）| `persist` 可选（`m_blocks>256` 自动开）|
| AGPR 累加 | `Nonatomic` ⇒ `kUseAGPR` | 无显式 AGPR 控制（交 MLIR 后端）|
| sort_block_m 解耦 | BM 语义固定（atomic{16,32,64}/nonatomic{128}）| `_sbm{N}`：tile_m 可小于排序 block_m（`sort_block_m`）|
| bias | 无 | 可选 `bias`（GEMM 后加）|

### 7.3 实现逻辑差异

| 环节 | mxfp4 HIP gemm2（M8）| FlyDSL gemm2（F6）|
|---|---|---|
| A→LDS | sorted-direct `buffer_load_lds`（行=排序位）| 经 `sorted_token_ids` 把 `inter_states[token,slot]` gather 入 LDS（async DMA / 寄存器）|
| K-loop 缓冲 | K=512→2 tile，B 软件流水 `kStages=2`，atomic/nonatomic 两套 LDS 布局（`gemm2_a4w4.cuh:34/42`）| K=2 tile，LDS ping-pong 双缓冲，单累加器 |
| MFMA scale op_sel | `mfma_f4f4_*` 封装 `mfma_scale_f32_16x16x128_f8f6f4`，A-scale op_sel = `ikxdl*2+i`（固定 m-half=2）| `rocdl.mfma_scale_f32_16x16x128_f8f6f4`（cbsz=4/blgp=4，`mixed:2851`），op_sel = `ikxdl*pack_M+imxdl` + `_rearrange` |
| atomic 归约 | `atomic_pk_add_bf16`(`global_atomic_fadd_v2bf16`) → `out[token]`（`mxfp4_epilogs.hpp:149`）| `AtomicRMWOp fadd`（agent scope）→ `out[token]`（`mixed:4417`）|
| reduce/flat 归约 | 写 `flat_out[sorted_pos]` → 外部 scatter_reduce（`:218`/`:245`）| plain NT store → `out[token*topk+slot]` → host `torch.sum`（`mixed:4402`，host `moe_kernels.py:1187`）|
| weight 应用位置 | atomic epilogue（`acc*weight` 后再 atomic）| cshuffle `write_row_to_lds` 内（`v=v*tw`，`mixed:4359`）|
| fp4-flat 量化指令 | **硬件** `cvt_scalef32_pk_fp4_f32`（`apply_mxfp4_flat_epilog_bm128`，`mxfp4_epilogs.hpp:245`）| **软件** `_f32_to_e2m1`（`flat_mxfp4`，`mixed:4527`）——硬件 intrinsic 在 rocdl 未暴露 |
| epilogue 重排 | cshuffle（atomic/nonatomic 两套 LDS 布局 + AGPR）| cshuffle（`mfma_epilogues.py:85`）|

### 7.4 小结

- **相同处**：down-proj 数学、MFMA 指令（`mfma_scale_f32_16x16x128_f8f6f4`）、K=2-tile 双缓冲、
  cshuffle 重排、`sorted_weights` 在 epilogue 内乘、以及「小 M atomic / 大 M flat+外部归约」的整体策略，两者一致。
- **不同处**：
  1. **中间量布局**：M8 sorted-major（gemm2 sorted-direct 读）；F6 token-major `[tok,topk]`（gather）——
     stage1↔stage2 契约的根本差异。
  2. **接口契约**：M8 = void + 预分配 + 两个独立 op（含 `_mxfp4out`）+ 字符串模板；F6 = 返回张量 + kwargs +
     `mode`/`flat_*` 开关。
  3. **大 M 归约**：M8 nonatomic-flat → 同仓 HIP scatter_reduce(_q)；F6 原生 `reduce` → host `torch.sum`
     （但 `flat_mxfp4` 也能改接 HIP scatter_reduce_q）。
  4. **fp4-flat 量化**：M8 硬件 `cvt_scalef32_pk_fp4`；F6 软件 `_f32_to_e2m1`。

> 若要把 FlyDSL gemm2 做成 M8 的 **drop-in**（吃 sorted-major `inter_sorted_quant`、void 副作用、与 HIP
> 一致的 scale op_sel / atomic 行为），需另写一份新 kernel——即
> `aiter/aiter/ops/flydsl/kernels/mxfp4_a4w4_gemm.py::compile_mxfp4_gemm2_a4w4`（已验证 BM=16 cos 1.0 /
> BM=32 cos 0.9999，与本节描述的 F6 不是同一实现）。
