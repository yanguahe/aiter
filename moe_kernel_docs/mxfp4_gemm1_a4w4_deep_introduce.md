# MXFP4 GEMM1 (`gemm1::kernel`) 深度介绍 — gate/up + inline-quant + SiLU·mul + 中间 fp4 再量化

> 本文是 `mx_fn`（HIP/mxfp4 路径）中 MoE **stage1 GEMM** 的深度剖析，遵循
> `.cursor/rules/kernel-deep-introduction-doc.mdc` 方法论：每个事实都可追溯到源码 + ISA，
> 指令语义取自 CDNA4 ISA 文档。

## 文档元信息（benchmark / trait / 构建配置）

- **目标 kernel**：`aiter::mxfp4_moe::gemm1::kernel<655360, 385, 7168, 1024, 16, true, true, 0>`
  = `<MAX_M, NUM_EXPERTS=385, K=D_HIDDEN=7168, N_OUT=2*D_INTER=1024, BM=16, kUseNT=true, kInlineQuant=true, kXcdSwizzle=0>`
- **关键维度符号**（模板参数 + 内部 tile 常量；源码 `gemm1_a4w4.cuh:22-62`）：
  | 符号 | 值 | 含义 |
  |---|---|---|
  | `MAX_M` | 655360 | M 维上界，仅用于 buffer-resource 容量上界计算 |
  | `NUM_EXPERTS` | 385 | 专家数 |
  | `K` (=D_HIDDEN) | 7168 | GEMM 收缩维（= 隐藏维），主循环沿此方向 |
  | `N_OUT` (=2·D_INTER) | 1024 | 输出列数 = gate 512 + up 512 |
  | **`BM`** | 16 | **block-M：一个 workgroup（M-block）处理的「排序后行(sorted rows)」数量，也即 MoE 排序的 block_size**。本 GEMM 支持 `BM∈{16,32,128}`；`BM=16` 对应最廉价的单 CTA 排序 + inline-quant。是模板参数（`<…,1024,16,…>` 中的 `16`）|
  | **`BN`** | 256 | **block-N：一个 workgroup 计算的输出列 tile 宽度**（256 = gate 128 + up 128，由 4 个 wave 切分）。内部 `constexpr`（`gemm1_a4w4.cuh:51`）|
  | **`BK`** | 256 | **block-K：K 方向每个 K-tile 的宽度**。内部 `constexpr`（`gemm1_a4w4.cuh:52`），故 `K_TILES_TOTAL = K/BK = 7168/256 = 28`。注意 ≠ 单条 MFMA 的 K=128（每个 K-tile 含 `BK/128 = 2` 个 MFMA-K 子步）|
  | `kUseNT` / `kInlineQuant` / `kXcdSwizzle` | true/true/0 | non-temporal 加载 / 内联量化 / XCD swizzle 开关 |
- **源码**：
  - 模板 kernel：`aiter/csrc/kernels/mxfp4_moe/gemm_a4w4/gemm1_a4w4.cuh:28`
  - epilogue：`aiter/csrc/kernels/mxfp4_moe/gemm_a4w4/common/mxfp4_epilogs.hpp::apply_cshuffle_quant_epilog:21`
  - MFMA 封装：`common/mfma_f4f4.hpp::mfma_f4f4_vgpr:22`
  - 实例化 blob：`aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT.cu`
  - 分发：`gemm_a4w4/mxfp4_moe_gemm.cu::mxfp4_moe_gemm1_a4w4_kernel`（codegen lookup table）
  - host op：`aiter.mxfp4_moe_gemm1_a4w4`（`aiter/aiter/ops/mxfp4_moe.py:145`）
  - 权重/scale preshuffle：`aiter/aiter/ops/shuffle.py::shuffle_weight_a16w4:53` / `shuffle_scale_a16w4:141`
- **ISA dump**（gfx950 / CDNA4，编译 flag 同 build.ninja + `-S --cuda-device-only --offload-arch=gfx950`）：
  - ASM：`aiter/mxfp4_moe_isa/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT.gfx950.s`（3656 行）
  - LLVM IR：`aiter/mxfp4_moe_isa/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM16_INLINEQUANT.gfx950.ll`（4438 行）
  - kernel 符号：`_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEE...`
- **ISA 资源元数据（已核对 .s）**：`.amdhsa_next_free_vgpr 139`、`.amdhsa_next_free_sgpr 35`、
  `.amdhsa_group_segment_fixed_size 16384`、`.amdhsa_accum_offset 140`、`vgpr_spill_count 0`。
- **指令语义参考**：`.cursor/rules/amd-instinct-cdna4-instruction-set-architecture.txt`、
  `.cursor/rules/amd-cdna-4-architecture-whitepaper.txt`。
- **benchmark 形状（KIMI）**：NE=385、K=D_HIDDEN=7168、N_OUT=2*D_INTER=1024、D_INTER=512、TOPK=9，
  M=16（小 M，走 BM=16 inline-quant），gfx950 MI355X。a4w4：A=fp4（由 bf16 hidden 内联量化）、
  W=fp4、e8m0 per-32 微缩放；输出中间量 = fp4 + e8m0，已按 gemm2 的 sorted-tile 布局排好。

## 目录

- [1. Kernel 概览与软件流水](#1-kernel-概览与软件流水)
  - [1.1 Buffering：A / A_scale / B / B_scale / inter 在 GM/LDS/VGPR 的分布](#11-buffering)
  - [1.2 融合数学拆成子阶段：inline-quant → GEMM → SiLU·mul → fp4 再量化](#12-融合数学子阶段)
  - [1.3 Wave / 调度结构](#13-wave--调度结构)
  - [1.4 精化伪代码](#14-精化伪代码)
  - [1.5 每操作主指令与计数](#15-每操作主指令与计数)
  - [1.6 调度原语](#16-调度原语)
- [2. 完整 GM/LDS/VGPR 布局图](#2-完整-gmldsvgpr-布局图)
  - [2.1 A：inline-quant（bf16 GM → fp4 LDS → VGPR）](#21-a-inline-quant)
  - [2.2 B：a16w4 权重（GM → VGPR）](#22-b-a16w4)
  - [2.3 A_scale（inline 产出）/ B_scale（make_preshuffle）](#23-scale)
  - [2.4 累加器与 cshuffle（VGPR → LDS）](#24-累加器与-cshuffle)
  - [2.5 输出：inter fp4 + e8m0 sorted-tile（LDS → GM）](#25-输出)
  - [2.6 张量驻留汇总](#26-驻留汇总)
  - [2.7 底层调用链与 issue 计数](#27-调用链与计数)

---

## 1. Kernel 概览与软件流水

本 kernel 是 MoE 第一层 GEMM。对每个「M-block × N-block × 专家 e」计算 `inter = act(a_q × w1ᵀ)`：
`w1` 含 **gate(前 512 列)** 与 **up(后 512 列)**（`N_OUT = 2*D_INTER = 1024`），主循环沿 K=7168
做 **28 个 K-tile**（`K_TILES_TOTAL = K/BK = 7168/256 = 28`，K-tile 块大小 `BK=256`）的 fp4×fp4 缩放
MFMA 累加，epilogue 内 `y = SiLU(gate)·up` 并**就地再量化为 mxfp4**，产出 `inter_sorted_quant`（fp4）+
`inter_sorted_shuffled_scale`（e8m0，已是 gemm2 的 tile 布局）。

> 注意区分两个「K」：**K-tile 块大小 = `BK=256`**（故 7168/256 = **28** 个 tile），而单条 MFMA 指令
> `v_mfma_scale_f32_16x16x128_f8f6f4` 的 **K=128**。每个 256 宽的 K-tile 内部含 `256/128 = 2` 个 MFMA-K 子步
> （即 `E_K=2`，源码里的 `a[i][0]/b[..][0]` 与 `a[i][1]/b[..][1]` 两段）。所以「MFMA-K=128 的子步总数」=
> `28 × 2 = 56 = 7168/128`——你算的 56 正是这个子步数，而非 K-tile 数。

本实例是 **M=16 的 INLINEQUANT 路径**（`kInlineQuant=true, BM=16`）：A 不预量化，kernel 内直接读
bf16 `hidden_states`，**线程本地**完成 bf16→fp4 量化后写入 LDS（省掉独立的量化/排序 kernel），
从而能配合最廉价的 BM=16 单 CTA 排序。

一个 workgroup = 256 线程 = 4 wave，处理一个 `BM×256` 输出 tile（256 列 = gate 的 128 + up 的 128，
由 `wave_n` 切分）。grid = `total_m_blocks * (N_OUT/256 = 4)`，`total_m_blocks = cumsum/BM`
（`gemm1_a4w4.cuh:575`）。`__launch_bounds__(256, 3)`（BM=16 → 3 waves/EU）。

<a id="11-buffering"></a>

### 1.1 Buffering：A / A_scale / B / B_scale / inter 在 GM/LDS/VGPR 的分布

LDS 由一个 `union LDSPool` 描述（`gemm1_a4w4.cuh:99-109`）：

```cpp
union alignas(16) LDSPool {
    struct {
        alignas(16) __hip_fp4x2_storage_t s_Aq    [kAStages][BM][BK / 2];   // 3 * 16 * 128 = 6144 B
        alignas(16) uint8_t               s_Ascale[kSubBlocks * K_TILES_TOTAL * 256]; // 1*28*256 = 7168 B
    } kloop;                                                                 // 合计 13312 B
    float lds_acc[BM * BN];                                                  // 16*256*4 = 16384 B
};
```

- `kAStages=3`（BM<128 时 A 用 3 个 LDS 槽 → 无 WAR 冒险）、`kStages=2`（B 软件流水深度）、
  `kSubBlocks=1`（BM<32）、`kMChunks=BM/16=1`、`BK=256`、`BK/2=128`（fp4x2 打包）。
- `union` 让「主循环用的 `s_Aq`+`s_Ascale`(13312 B)」与「epilogue 用的 `lds_acc`(16384 B)」**复用同一块 LDS**
  （主循环结束后才进 epilogue，二者不同时存活）→ 总 LDS = `max(13312, 16384) = 16384 B`，
  与 `.amdhsa_group_segment_fixed_size 16384` 完全吻合。

| 张量 | Global memory | LDS | VGPR |
|---|---|---|---|
| A（hidden）| `hidden_states[token, 7168]` bf16；按 `m_indices[sorted_row]` gather token | inline-quant 量化后的 fp4 写入 `s_Aq[3][16][128]`（XOR16 swizzle）| `a[kMChunks=1][2]` = `i32x4`×2（一个 lane 持 256 个 fp4 = 一行的两个 128-K 段）→ 喂 MFMA |
| A_scale | （inline，不读 GM）| e8m0 攒进 `s_Ascale[1*28*256]`（按 MFMA 消费的 (lane,kgroup) 布局）| `a_scale_aiter[kSubBlocks=1]`（一个 dword，含 4 个 e8m0 字节）|
| B = w1（gate‖up）| `B_ps_q[E, 1024, 7168/2]` fp4x2（`shuffle_weight_a16w4(w1,16,True)`）| 不过 LDS（寄存器驻留）| `b[kStages=2][4][2]` = `i32x4`（双缓冲 × 4 个 N-子块 × 2 个 128-K 段）|
| B_scale = w1_scale | `B_ps_scale[E*1024, 7168/32]` e8m0 | 不过 LDS | `b_scale_v[kStages=2][2]`（dword）|
| inter_out | `A_q_out[max_sorted, 512/2]` uint8（**输出**，fp4）| 经 `lds_acc` cshuffle 重排 | `accm[kMChunks=1][4]` = `f32x4`×4（gate/up 累加器，`v_o` 等价）|
| inter_scale_out | `A_scale_out`（**输出**，e8m0，gemm2 tile 布局）| — | epilogue 内 `scales_per_mr[M_REPS=1]` uint8 |

VGPR 驻留核对（`.s` / `.ll`）：每 lane 主要寄存器 = `a`(2×i32x4=8 VGPR) + `b`(2×4×2×i32x4=64 VGPR)
+ `accm`(1×4×f32x4=16 VGPR) + scale/地址/临时；`.amdhsa_next_free_vgpr 139`、`.amdhsa_accum_offset 140`
说明 accumulator/常规 VGPR 共占 ~139（本配置 `kUseAGPR=false`，BM=128 才用 AGPR）。

> 关键点：**B 全程寄存器驻留**（`b[kStages][4][2]`，双缓冲软件流水），**A 走 LDS**（inline-quant 把 bf16→fp4
> 写进 `s_Aq`，再 `ds_read` 回 VGPR 喂 MFMA）。这与 FlyDSL 版「A 也走 LDS、B 走 make_preshuffle」一致，
> 但 A 的来源不同（HIP 在 kernel 内量化，FlyDSL 由外部 quant kernel 产出）。

<a id="12-融合数学子阶段"></a>

### 1.2 融合数学拆成子阶段：inline-quant → GEMM → SiLU·mul → fp4 再量化

gemm1 把「激活量化 + GEMM + 激活函数 + 中间量化」四件事融合进一个 kernel，分布在流水的不同阶段：

1. **inline-quant（主循环每个 K-tile，prologue+steady-state）**：读 bf16 hidden → 线程本地求每 32-K 组
   的 amax → 编码 e8m0 → `cvt_scalef32_pk_fp4_bf16` 量化成 fp4 写 `s_Aq`，e8m0 写 `s_Ascale`。
   （`inline_quant_kt` / `inline_quant_finish_kt`，`gemm1_a4w4.cuh:209/268`）
2. **GEMM（主循环 28 K-tile）**：`ds_read` 取 `s_Aq` → `a`，B 从寄存器，`mfma_f4f4_vgpr`
   按 e8m0 在 MFMA 内反量化做 fp4×fp4，累加进 `accm[1][4]`（gate/up 各占 J 的偶/奇）。
3. **SiLU·mul（epilogue）**：`accm` 经 LDS cshuffle 重排成连续 N 向量，取 `gate_v[e]`/`up_v[e]`，
   `result[e] = silu_mul_fast(gate, up)`（`mxfp4_epilogs.hpp:88`，SiLU 用快速 exp2/rcp）。
4. **输出 fp4 再量化（epilogue）**：对每 32 个 result 经 DPP-quad 求 amax → e8m0 →
   `cvt_scalef32_pk_fp4_f32` 压成 fp4，写 `aq_out` + `a_scale_out`（gemm2 tile 布局）。

> e8m0 舍入常数：epilogue 用 `quant_scale = uint_as_float(amax + 0x200000) * 0.25`
> （`mxfp4_epilogs.hpp:104`），即 mxfp4 精确的 `+0x200000`（0.25 ULP）；inline-quant 端的 e8m0 由
> `inline_quant_encode_e8m0` 产出，二者一致。

<a id="13-wave--调度结构"></a>

### 1.3 Wave / 调度结构

- 256 线程 = 4 wave；`wave = tid/64`、`wave_n = wave`、`lane = tid%64`（`gemm1_a4w4.cuh:77-79`）。
  4 个 wave 各算 256 列输出 tile 的一个 64 列段（`wave_n*(BN/4)`），gate/up 在列上交织。
- **K 主循环软件流水**：B 用 `kStages=2` 双缓冲（`b[K_C%2]`），A 用 `kAStages=3` 三槽
  （`read_slot/write_slot = OFFSET%3 / K_C%3`，`gemm1_a4w4.cuh:468-469`），每次 `__syncthreads()`
  后先 `ds_read` 当前槽 A、再为未来槽做 inline-quant 写 LDS、再发 4 个 N-子块的 MFMA。
- **`s_setprio` 围栏**：BM≠128 时每个 MFMA 簇用 `s_setprio(1)`…`s_setprio(0)` 包裹
  （`gemm1_a4w4.cuh:500-502/511-513/533-538`），把共享 MFMA 发射端口在计算 wave 之间错峰交接；
  配合 `__builtin_amdgcn_sched_barrier(0)` 作为编译期硬调度栅栏（见 §1.6）。

<a id="14-精化伪代码"></a>

### 1.4 精化伪代码

下面对应 `run_one`（`gemm1_a4w4.cuh:377-571`）的 BM=16 INLINEQUANT 路径，每行一条语句。

```cpp
// ── grid 入口（gemm1_a4w4.cuh:573-592）──
total_m_blocks = cumsum_tensor[0] / 16
total_tiles    = total_m_blocks * (N_OUT/256 = 4)
if pid >= total_tiles: return
// BM=16 走默认映射（kInlineQuant&&BM==32 才用 remap_xcd）
m_block_idx = pid / 4
n_block_idx = pid % 4
e           = sorted_expert_ids[m_block_idx]

// ── run_one(m_block_idx, n_block_idx, e) ──
m_row = m_block_idx * 16
// 缓存本 wave 负责的 sorted-row → token 映射（inline: kCachedInline=1）
rcls               = wave*4 + lane/16
cached_row_inline0 = m_indices[m_row + rcls]
// B / B_scale 的 per-expert/N base
for j in 0..3:
    b_load_s_base[j] = (e*N_OUT + n_block_idx*256 + wave_n*64 + j*16) * (K/2)
for mw in 0..1:
    b_scale_s_base[mw]    = (e*kBS_per_expert_dw + (mni_base+mw)*kBS_stride_n0_dw) * 4
    b_scale_s_base_hi[mw] = b_scale_s_base[mw] + 16*(kBS_stride_k0_dw*4)

// ── Prologue：填 kStages=2 个 K-tile（K_C=0,1）──
for K_C in 0..1:
    scale_accum = 0
    inline_quant_kt<B128_IDX=0, SUB=0, kPackScale=true>(K_C, K_C, cached_row_inline0, &scale_accum)
    issue_b_load_j<K_C>(b[K_C], 0)
    issue_b_load_j<K_C>(b[K_C], 1)
    inline_quant_kt<1, 0, true>(K_C, K_C, cached_row_inline0, &scale_accum)
    issue_b_load_j<K_C>(b[K_C], 2)
    issue_b_load_j<K_C>(b[K_C], 3)
    inline_quant_pack_write(K_C, scale_accum)     // 把 4 个 e8m0 打成 1 dword 写 s_Ascale
    issue_b_scale_load<K_C>(b_scale_v[K_C])

// ── Steady-state：kUnroll = K_TILES_TOTAL - kStages = 26 次 ──
for OFFSET in 0..25:
    K_C        = 2 + OFFSET
    read_slot  = OFFSET % 3
    write_slot = K_C    % 3
    slot_b     = OFFSET % 2
    __syncthreads()
    issue_a_ds_read(read_slot)                     // ds_read s_Aq → a[0][0..1]
    issue_a_scale_ds_read(K_C - 2)                 // ds_read s_Ascale → a_scale_aiter[0]
    h_v0 = inline_quant_load_kt<0>(K_C, cached_row_inline0)   // raw_buffer_load_b128 bf16
    h_v1 = inline_quant_load_kt<1>(K_C, cached_row_inline0)
    sched_barrier(0)
    for J in 0..3:
        sched_barrier(0)
        s_setprio(1)
        issue_mfma_cluster<J, kInit=(OFFSET==0)>(slot_b)       // 2× mfma_f4f4_vgpr
        s_setprio(0)
        sched_barrier(0)
        issue_b_load_j<K_C>(b[slot_b], J)                      // 2× buffer_load_b128
        sched_barrier(0)
    issue_b_scale_load<K_C>(b_scale_v[slot_b])                 // 2× buffer_load_b32
    scale_accum = 0
    inline_quant_finish_kt<0, 0, true>(write_slot, K_C, h_v0, &scale_accum)  // 量化→s_Aq
    inline_quant_finish_kt<1, 0, true>(write_slot, K_C, h_v1, &scale_accum)
    inline_quant_pack_write(K_C, scale_accum)

// ── Drain：最后 kStages=2 个 K-tile，只算不预取 ──
for S in 0..1:
    kt          = K_TILES_TOTAL - 2 + S        // 26, 27
    read_slot_a = kt % 3
    slot_b_drain= kt % 2
    __syncthreads()
    issue_a_ds_read(read_slot_a)
    issue_a_scale_ds_read(kt)
    for J in 0..3:
        issue_mfma_cluster<J>(slot_b_drain)

// ── Epilogue ──
__syncthreads()
apply_cshuffle_quant_epilog<N_OUT, BM>(accm, A_q_out, A_scale_out,
    m_block_idx, m_row, n_block_idx, wave, wave_n, lane, tid, lds_acc)
```

epilogue 内部（`mxfp4_epilogs.hpp:21-145`，每行一条语句）：

```cpp
// 1) accm → lds_acc（cshuffle：gate 写前 128 列，up 写 +128 列）
for i in 0..BM/16-1:                       // BM=16 → 1
    row_base = i*16 + (lane/16)*4
    for J in 0..3:
        is_up     = (J%2==1)
        col_local = wave_n*32 + (J/2)*16 + (lane%16)
        lds_col   = is_up ? 128+col_local : col_local
        for v in 0..3:
            lds_acc[(row_base+v)*256 + lds_col] = accm[i][J][v]
__syncthreads()
// 2) 读回 → SiLU·mul → fp4 再量化（M_REPS = BM/16 = 1）
for mr in 0..M_REPS-1:
    row_local = mr*16 + (tid/16)
    for e in 0..7:
        gate_v[e] = lds_acc[row_local*256 + gate_col(e)]
        up_v[e]   = lds_acc[row_local*256 + 128 + gate_col(e)]
    for e in 0..7:
        result[e] = silu_mul_fast(gate_v[e], up_v[e])
    local_max = max(|result[0..7]|)
    local_max = fmax(local_max, dpp(local_max, 0xB1))   // cross-lane quad
    local_max = fmax(local_max, dpp(local_max, 0x4E))
    quant_scale     = uint_as_float(float_as_uint(local_max) + 0x200000) * 0.25
    scales_per_mr[mr] = min(float_as_uint(quant_scale) >> 23, 254)
    packed = 0
    packed = cvt_scalef32_pk_fp4_f32(packed, result[0], result[1], quant_scale, 0)
    packed = cvt_scalef32_pk_fp4_f32(packed, result[2], result[3], quant_scale, 1)
    packed = cvt_scalef32_pk_fp4_f32(packed, result[4], result[5], quant_scale, 2)
    packed = cvt_scalef32_pk_fp4_f32(packed, result[6], result[7], quant_scale, 3)
    nontemporal_store(packed, &aq_out[out_row*K_G2_HALF + byte_pos])
// 3) e8m0 写 a_scale_out（kk==0 的 lane 负责，BM=16 写低字节）
if kk == 0:
    dword_off = m_block_idx*kAS_per_chunk_dw + (n_block_idx>>1)*64 + wave_grp*16 + m_lane
    a_scale_out[dword_off*4 + (n_block_idx&1)*2] = scales_per_mr[0]
```

<a id="15-每操作主指令与计数"></a>

### 1.5 每操作主指令与计数

下表为「**一个 warp 处理整个 kernel（28 K-tile + epilogue）**」的指令数，已用 `.s`（剥离 `;` 注释、
锚定指令字段起始）核对。tiling 常量：`K_TILES_TOTAL=28`、每 tile 4 个 N-子块(J)、BM=16 → `kMChunks=1`、
`kSubBlocks=1`、epilogue `M_REPS=1`、`EVec=8`。

| # | 操作 | 主指令 | 每 K-tile | ×28 K-tile | 推导 |
|---|---|---|---|---|---|
| 1 | GEMM MFMA | `v_mfma_scale_f32_16x16x128_f8f6f4`（`mfma_f4f4_vgpr`）| 8 | **224** | `issue_mfma_cluster` BM=16 = 2 MFMA/J × 4 J = 8；×28 = 224（与 .s `v_mfma`=224 一致）|
| 2 | inline-quant 输入 fp4 转换 | `v_cvt_scalef32_pk_fp4_bf16` | 8 | 224 | 每 tile 2 次 `inline_quant_kt/finish_kt`(B128_IDX 0,1)，各 4 个 cvt；8×28=224 |
| 2e| 输出 fp4 再量化（epilogue）| `v_cvt_scalef32_pk_fp4_f32` | — | 4 | `M_REPS=1` × 4 个 cvt（result 8 元素打 4 次）|
| — | **fp4 转换合计** | — | — | **228** | 224 + 4（与 .s `cvt_scalef32_pk_fp4`=228 一致）|
| 3 | B(a16w4) 载入 | `buffer_load_dwordx4`（`issue_b_load_j`，每 J 两次 @ +0/+1024B）| 8 | 224 | 4 J × 2 = 8；×28 = 224 |
| 4 | inline A bf16 载入 | `buffer_load_dwordx4`（`inline_quant_load_kt`，B128_IDX 0,1）| 2 | 56 | 2×28 = 56 |
| 5 | B_scale 载入 | `buffer_load_dword`（`issue_b_scale_load`，2 mw）| 2 | 56 | 2×28 = 56 |
| — | **buffer_load 合计** | — | — | **336** | 224+56+56（与 .s `buffer_load`=336 一致）|
| 6 | A fp4 写 LDS | `ds_write_b32`（inline_quant 把 fp4 packed 写 `s_Aq`）| ~3 | — | 与 e8m0/读回共同构成 .s `ds_write`=92 |
| 7 | A fp4 读 LDS | `ds_read_b128`（`issue_a_ds_read`）| ~3 | — | 与 A_scale 读、cshuffle 读共同构成 .s `ds_read`=92 |
| 8 | SiLU·mul | `silu_mul_fast`（`v_exp_f32`/`v_rcp_f32`/`v_mul`/`v_fma`）| — | — | epilogue 每元素一组（fast exp2/rcp）|
| 9 | epilogue amax | `v_max_f32` + `v_mov_b32_dpp`(0xB1, 0x4E) | — | — | 8 元素串行 max + 2 次 cross-lane quad DPP |

> 说明：counts 3-5 的 `buffer_load` 三项相加 = 336，counts 1 的 MFMA = 224，counts 2/2e 的 fp4 转换 = 228，
> 三者均与 ISA dump 精确吻合，构成本 kernel 计数可信度的锚点。`ds_read`/`ds_write` 各 92 由「A 的 inline 写 +
> 主循环读 + epilogue cshuffle 写/读」共同组成（细分见 §2.1/§2.4）。无 `global_atomic`（gemm1 不做 topk 归约）。

<a id="16-调度原语"></a>

### 1.6 调度原语

主循环里穿插两类调度原语（`gemm1_a4w4.cuh:499-543`）：

| 原语 | 是否产出真实 ISA | 语义（取自 CDNA4 ISA 文档）|
|---|---|---|
| `__builtin_amdgcn_sched_barrier(0)` | **否**（编译期栅栏）| 降为 AMDGPU `SCHED_BARRIER` 伪指令；`mask=0`=「不允许任何类别跨越」，使 MFMA/VALU/VMEM/DS 都不能越过此点重排。仅约束 LLVM 调度器，不发射机器码。本 kernel 用它把每个 MFMA 与其后的 `issue_b_load_j` 锁定顺序（B 预取紧跟在对应 MFMA 后）。|
| `__builtin_amdgcn_s_setprio(1/0)` | **是**（`s_setprio` 真实指令）| ISA 文档 `S_SETPRIO SIMM16`（p.12-49）：设置用户 wave 优先级，`SIMM16[1:0]` 0–3，0 最低、3 最高（整体优先级 = `{SPIPrio, UserPrio, WaveAge}`）。这里 `s_setprio(1)`…MFMA…`s_setprio(0)` 把 MFMA 簇围起来：进入 MFMA 前抬优先级抢占共享发射端口，算完降回，让另一计算 wave 错峰接手。仅 BM≠128 启用。|

> 与 FlyDSL 版一致点：两者都用 `sched_barrier(0)` 作硬调度栅栏、`s_setprio` 做 wave 错峰。差别在 HIP
> 用 C++ `__builtin_amdgcn_*`，FlyDSL 用 `rocdl.sched_barrier` / `rocdl.asm_s_setprio` 的 MLIR intrinsic。

---

## 2. 完整 GM/LDS/VGPR 布局图

记号：`lane = tid%64`、`wave = tid/64`、`wave_n = wave`。fp4x2 表示 1 字节含 2 个 fp4
（沿 K 打包，故 K 字节数 = K/2）。

<a id="21-a-inline-quant"></a>

### 2.1 A：inline-quant（bf16 GM → fp4 LDS → VGPR）

A 不预量化：kernel 内每个 K-tile 直接读 bf16 `hidden_states`，线程本地量化成 fp4 写 LDS。

**GM 读地址**（`inline_quant_load_kt` / `inline_quant_kt`，`gemm1_a4w4.cuh:197-221`）：

```cpp
// row_token = m_indices[m_row + wave*4 + lane/16]  （sorted-row → 原 token）
v_voff = row_token * (K * 2)                 // 每 token 行 = K=7168 个 bf16 = 7168*2 字节；*2 因 K*2
       + ((lane >> 2) & 3) * 64              // 同一 32-K 组内 4 个 lane 各偏 64 字节
       + (lane & 3) * 16                     // (lane&3) 选 4 个 16 字节段 → 拼成 32 个 bf16=一个量化组
s_soff = kt * (BK * 2) + B128_IDX * 256      // K-tile 基址 + 半 tile（B128_IDX∈{0,1}）
h_v    = raw_buffer_load_b128(hidden_rsrc, v_voff, s_soff)   // 16 字节 = 8 个 bf16
```

**线程本地量化**：`hm[j]=h_dw[j]&0x7FFF7FFF`（清符号取 |x|）→ `inline_quant_pkmax_u16` 两两取大
→ 本 lane 8 个 bf16 的 amax → `inline_quant_dpp_quad_amax`（DPP `0xB1`/`0x4E` 跨 4 lane）得整 32-K 组 amax
→ `inline_quant_encode_e8m0` → `qs = uint_as_float(e8m0<<23)` →
`cvt_scalef32_pk_fp4_bf16(pk, bf16x2, qs, op_sel=0..3)` 四次填满 1 dword（8 个 fp4）。

**LDS 写地址**（`s_Aq`，含 XOR16 swizzle，`gemm1_a4w4.cuh:241-248`）：

```cpp
r        = SUB*16 + (wave*4 + lane/16)        // 行（BM=16 → SUB=0, r=wave*4+lane/16 ∈ 0..15）
kb_in_kt = B128_IDX*4 + ((lane>>2)&3)         // tile 内 K 子块
mask_r   = lds_swizzle_mask<BK/2=128>(r)      // 行相关的 XOR 掩码（消 bank conflict）
b_off    = (lane&3)*4
s_Aq[slot][r][ ((kb_in_kt*16) ^ mask_r) + b_off ] = pk    // 1 dword = 8 fp4
```

**LDS 读回 → VGPR**：`issue_a_ds_read` 用 `ds_read_b128` 把 `s_Aq[read_slot]` 取进 `a[0][0..1]`
（一个 lane 持 `i32x4`×2 = 256 个 fp4 = 一行的两个 128-K 段）。

**可视化（lane → 32-K 量化组）**，固定一行 `r`、一个 `kt`、`B128_IDX=0`：

| lane | (lane>>2)&3 | (lane&3) | 负责的 bf16（该行该 tile 内）| 产出 |
|---:|:---:|:---:|---|---|
| 0 | 0 | 0 | K 段[0:8] | 4×cvt → 1 dword fp4 |
| 1 | 0 | 1 | K 段[8:16] | 同组（共享 amax）|
| 2 | 0 | 2 | K 段[16:24] | 同组 |
| 3 | 0 | 3 | K 段[24:32] | 同组（4 lane 经 DPP-quad 共享一个 e8m0）|
| 4 | 1 | 0 | K 段[64:72] | 下一 32-K 组 |
| ... | ... | ... | ... | ... |

即 **每 4 个连续 lane（`lane&3`）拼成一个 32-K 量化组**，通过 DPP-quad 在组内共享 amax → 一个 e8m0；
这正是「线程本地、无需跨 wave 归约」的廉价量化（mxfp4 文档称之为 inline-quant）。

**e8m0 写 `s_Ascale`**（`inline_quant_pack_write`，`gemm1_a4w4.cuh:261-266`）：

```cpp
pack_byte  = B128_IDX*2 + SUB                  // 4 个 e8m0 打进 scale_accum 的 4 个字节
lane_tgt   = ((lane>>2)&3)*16 + (wave*4 + lane/16)
s_Ascale[ kt*256 + lane_tgt*4 ] = scale_accum  // 1 dword = 4 个 e8m0
```

<a id="22-b-a16w4"></a>

### 2.2 B：a16w4 权重（GM → VGPR，不过 LDS）

B（`w1` = gate‖up）由 `shuffle_weight_a16w4(w1,16,True)` 预排成 a16w4 7D 布局，gemm1 用寄存器驻留方式
直接从 GM 读进 `b[kStages][4][2]`，不经 LDS。

**GM 读地址**（`issue_b_load_j`，`gemm1_a4w4.cuh:312-320`）：

```cpp
// per-(expert,N-block,wave,j) 基址（readfirstlane，标量）
b_load_s_base[j] = (e*N_OUT + n_block_idx*256 + wave_n*64 + j*16) * (K/2)    // 单位：fp4x2 字节
K_BYTE = K_C * 2048                                                          // K-tile 偏移
v_voff = (lane/16)*256 + (lane%16)*16 + K_BYTE                               // per-lane 16B 向量
// 每个 j 发两次 b128：@ +0 和 @ +1024 字节（两个 128-K 段）
buffer_load_b128_imm<0,    aux=kUseNT?2:0>(b[..][j][0], B_ps_q_rsrc, v_voff, b_load_s_base[j])
buffer_load_b128_imm<1024, aux>          (b[..][j][1], B_ps_q_rsrc, v_voff, b_load_s_base[j])
```

- `j∈0..3` 对应 256 列 N-tile 内 4 个 16 列子块；`(lane/16)` 选 4 个 256B 行、`(lane%16)*16` 选 16B 段，
  正好覆盖 a16w4 的 `[KLane=4, NLane=16, KPack=16]` 内层（与 `shuffle_weight_a16w4` 的
  `permute(0,2,1,4,5,3,6)` 对应，见 `shuffle.py:27-32`）。
- `kUseNT=true` → `buffer_load` 带 non-temporal hint（aux=2），权重只读一次不污染 cache。

**可视化（lane → B 的 16B 向量，固定 j、K_C=0）**：

| lane | lane/16 | lane%16 | v_voff（字节）| 取 a16w4 的 |
|---:|:---:|:---:|---:|---|
| 0 | 0 | 0 | 0 | (KLane=0, NLane=0) 的 16B=32 fp4 |
| 1 | 0 | 1 | 16 | (KLane=0, NLane=1) |
| ... | ... | ... | ... | ... |
| 15 | 0 | 15 | 240 | (KLane=0, NLane=15) |
| 16 | 1 | 0 | 256 | (KLane=1, NLane=0) |
| ... | ... | ... | ... | ... |
| 63 | 3 | 15 | 1008 | (KLane=3, NLane=15) |

一个 lane 的 `b[..][j][0..1]` = `i32x4`×2 = 256 个 fp4，恰为 MFMA 的 B 操作数（K=128 × 2 段）。

<a id="23-scale"></a>

### 2.3 A_scale（inline 产出）/ B_scale（make_preshuffle）

**A_scale**：由 §2.1 的 inline-quant 直接产出并写 `s_Ascale`（无 GM 读）。主循环用
`issue_a_scale_ds_read(kt)` 把对应 K-tile 的 e8m0 dword 取进 `a_scale_aiter[0]`，喂给 MFMA。

**B_scale**（`w1_scale`，make_preshuffle tile 布局）：`issue_b_scale_load`（`gemm1_a4w4.cuh:322-331`）：

```cpp
// 常量（gemm1_a4w4.cuh:64-72）
kBS_c_n1          = N_OUT/16/2          = 1024/16/2 = 32
kBS_c_k1          = (K/32)/4/2          = (7168/32)/4/2 = 28      // = kAS_c_k1, static_assert==28
kBS_stride_k0_dw  = 64
kBS_stride_n0_dw  = kBS_c_k1 * 64       = 1792
kBS_per_expert_dw = kBS_c_n1 * kBS_stride_n0_dw = 57344
// 读地址
v_voff = ((lane/16)*16 + (lane%16)) * 4
K_C_HI = K_C / 16
IMM    = (K_C - K_C_HI*16) * (kBS_stride_k0_dw*4)
s_off  = (K_C_HI==0) ? b_scale_s_base[mw] : b_scale_s_base_hi[mw]
bs_sub[mw] = buffer_load_b32_imm<IMM>(B_ps_scale_rsrc, v_voff, s_off)   // 1 dword=4 e8m0
```

**MFMA 如何消费 scale**（CDNA4 ISA 文档 §7.2.1，line 4321-4362）：`v_mfma_scale_f32_16x16x128_f8f6f4`
是 4-dword 指令，把「Load-Scale + MFMA」合一。K 维 block size=32，128/32 = **每行 4 个 e8m0**，M=N=16 →
共 `16*4=64` 个 8-bit scale = 1/4 个 VGPR 跨 64 lane。`{OP_SEL_HI[0],OP_SEL[0]}` 选 A scale 的 2-bit 字节码
（`00`→Src[7:0]、`01`→[15:8]、`10`→[23:16]、`11`→[31:24]），`{OP_SEL_HI[1],OP_SEL[1]}` 选 B scale。
本 kernel `mfma_f4f4_vgpr<AB,BB>` 把 `op_sel_a=AB`、`op_sel_b=BB` 透传（`mfma_f4f4.hpp:22-29`，
`cbsz=4`/`blgp=4` 选 fp4 子格式）。scale 在点积后、写回累加前施加。

<a id="24-累加器与-cshuffle"></a>

### 2.4 累加器与 cshuffle（VGPR → LDS → VGPR）

GEMM 累加器 `accm[kMChunks=1][4]`（`f32x4`×4，每 lane 64 个 f32）。J 的偶/奇分别是 gate/up
（`in_b = J%2`）。epilogue 第 1 步把 `accm` 经 `lds_acc[BM*BN=16*256]` 重排成「行连续 N」布局
（`mxfp4_epilogs.hpp:39-53`）：

```cpp
row_base  = i*16 + (lane/16)*4               // i∈0..BM/16-1（BM=16→0）；lane/16∈0..3 → 行 0..15
col_local = wave_n*32 + (J/2)*16 + (lane%16)
lds_col   = (J%2==1) ? 128+col_local : col_local      // gate 写 [0:128)，up 写 [128:256)
for v in 0..3: lds_acc[(row_base+v)*256 + lds_col] = accm[i][J][v]
```

`__syncthreads()` 后第 2 步按 `(m_lane=tid/16, n_lane=tid%16)` 读回，`wave_grp=n_lane/4`、`kk=n_lane%4`：

```cpp
row_local = mr*16 + m_lane                   // mr∈0..M_REPS-1（BM=16→0）
gate_col  = wave_grp*32 + 8*kk + e           // e∈0..7
gate_v[e] = lds_acc[row_local*256 + gate_col]
up_v[e]   = lds_acc[row_local*256 + 128 + gate_col]
```

这样 gate/up 同一逻辑列被同一 lane 取到，`silu_mul_fast(gate, up)` 后即得一行 8 个连续 inter 元素。

**ASCII：lds_acc 一行（256 f32 = 128 gate + 128 up）**

```
lds_acc row r (r = 0..15):
  col[0  : 128)  ← gate（wave_n*32 + (J/2)*16 + lane%16，4 wave 拼满 128）
  col[128: 256)  ← up  （同列 +128）
读回：lane (m_lane=r) 取 gate_col = wave_grp*32 + 8*kk + e 与其 +128 的 up，共 8 对
```

<a id="25-输出"></a>

### 2.5 输出：inter fp4 + e8m0 sorted-tile（LDS → GM）

**fp4 数据**（`aq_out` = `inter_sorted_quant[max_sorted, 512/2]`，`mxfp4_epilogs.hpp:108-122`）：

```cpp
quant_scale = uint_as_float(float_as_uint(local_max) + 0x200000) * 0.25   // mxfp4 精确舍入
packed = cvt_scalef32_pk_fp4_f32(packed, result[0],result[1], quant_scale, 0)  // ×4 填 1 dword
byte_pos = n_block_idx*(BN_INT/2) + wave_grp*16 + kk*4         // BN_INT=128 → /2=64
out_row  = m_row + row_local                                   // sorted 行（非 token！）
K_G2_HALF= N_INTER/2 = 256                                     // 每行 256 字节 = 512 fp4 = D_INTER
nontemporal_store(packed, &aq_out[out_row*256 + byte_pos])
```

输出是 **sorted 序**（`out_row = m_row + row_local`），直接作为 gemm2 的 A（gemm2 sorted-direct 读）。

**e8m0**（`a_scale_out`，gemm2 tile 布局，`mxfp4_epilogs.hpp:124-144`）：

```cpp
ku    = n_block_idx >> 1
ikxdl = n_block_idx & 1
// BM=16：每 dword 只写低字节，高字节是 pad
dword_off = m_block_idx*kAS_per_chunk_dw + ku*64 + wave_grp*16 + m_lane   // kAS_per_chunk_dw=1*28*64... 见下
a_scale_out[dword_off*4 + ikxdl*2] = scales_per_mr[0]
```

其中 epilogue 内 `kAS_c_k1=(N_INTER/32)/4/2`、`kAS_per_chunk_dw=1*kAS_c_k1*64`（N_INTER=512）。
`(ku, ikxdl)=(n_block>>1, n_block&1)` 把 4 个 N-block 映射到 e8m0 dword 的 (kgroup, byte)，与 gemm2
读 a2_scale 的 `op_sel = ikxdl*2` 取字节方式精确配对。

<a id="26-驻留汇总"></a>

### 2.6 张量驻留汇总

| 张量 | GM | LDS | VGPR（每 lane）| 备注 |
|---|---|---|---|---|
| A（bf16 hidden）| `[token,7168]` | 量化后 fp4 → `s_Aq[3][16][128]` | `a[1][2]`=i32x4×2 (8 VGPR) | inline-quant，gather via m_indices |
| A_scale | — | `s_Ascale[1*28*256]` | `a_scale_aiter[1]` (1 dword) | inline 产出 |
| B（w1 a16w4）| `[385,1024,3584]` | — | `b[2][4][2]`=i32x4 (64 VGPR) | 寄存器驻留双缓冲 |
| B_scale | `[385*1024,224]` | — | `b_scale_v[2][2]` | make_preshuffle tile |
| accm（gate/up）| — | 经 `lds_acc[16*256]` cshuffle | `accm[1][4]`=f32x4×4 (16 VGPR) | union 复用 LDS |
| inter_out | `[max_sorted,256]` uint8 | — | epilogue 临时 `packed` | fp4，sorted 序 |
| inter_scale_out | tile 布局 | — | `scales_per_mr[1]` | e8m0，gemm2-ready |

资源：VGPR 139、SGPR 35、LDS 16384 B、spill 0（与 .s 一致）。

<a id="27-调用链与计数"></a>

### 2.7 底层调用链与 issue 计数

```
mxfp4_moe_gemm1_a4w4 (host, mxfp4_moe.py:145)
└─ mxfp4_moe_gemm1_a4w4_kernel (mxfp4_moe_gemm.cu)         // 按 kernelName 查 g1_cshuffle_lookup
   └─ gemm1::launch<655360,385,7168,1024,16,true,true,0>   // blob 实例化
      └─ gemm1::kernel<...>                                 // gemm1_a4w4.cuh:28
         ├─ run_one(m_block_idx, n_block_idx, e)            // :377
         │  ├─ inline_quant_load_kt / inline_quant_kt / inline_quant_finish_kt  // bf16→fp4 + e8m0
         │  ├─ issue_b_load_j<K_C>                          // a16w4 B-load (buffer_load_b128)
         │  ├─ issue_b_scale_load<K_C> / issue_a_scale_load // scale (buffer_load_b32)
         │  ├─ issue_a_ds_read / issue_a_scale_ds_read      // ds_read s_Aq / s_Ascale
         │  ├─ issue_mfma_cluster<J> → mfma_f4f4_vgpr       // v_mfma_scale_f32_16x16x128_f8f6f4
         │  └─ apply_cshuffle_quant_epilog                  // mxfp4_epilogs.hpp:21
         │     ├─ silu_mul_fast                             // SiLU·mul (exp2/rcp)
         │     └─ cvt_scalef32_pk_fp4_f32 + nontemporal_store / a_scale_out 写
```

**whole-kernel issue 计数（warp，已核对 .s，剥离 `;` 注释）**：

| 指令 | 计数 | 推导 |
|---|---:|---|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 224 | 8/tile × 28 |
| `v_cvt_scalef32_pk_fp4`（bf16+f32）| 228 | 输入 224 + 输出 4 |
| `buffer_load`（dwordx4 + dword）| 336 | B 224 + inline-A 56 + B_scale 56 |
| `ds_read` | 92 | A 读 + A_scale 读 + cshuffle 读 |
| `ds_write` | 92 | inline-A fp4 写 + e8m0 写 + cshuffle 写 |
| `global_atomic` | 0 | gemm1 不做 topk 归约 |
| VGPR / SGPR / LDS / spill | 139 / 35 / 16384 / 0 | `.amdhsa_*` |

> 三个锚点（MFMA 224、fp4 转换 228、buffer_load 336）均与 ISA dump 精确吻合，确保上述 tiling 推导可信。
> 本 kernel 与 FlyDSL gemm1 的接口/布局/精度差异见 `mxfp4_05_gemm1.md §7`。

