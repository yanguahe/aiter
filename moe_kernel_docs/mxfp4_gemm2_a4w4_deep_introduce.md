# MXFP4 GEMM2 (`gemm2::kernel`) 深度介绍 — down 投影 + atomic topk 归约

> 本文是 `mx_fn`（HIP/mxfp4 路径）中 MoE **stage2 GEMM** 的深度剖析，遵循
> `.cursor/rules/kernel-deep-introduction-doc.mdc`：每个事实可追溯到源码 + ISA，指令语义取自 CDNA4 ISA 文档。

## 文档元信息（benchmark / trait / 构建配置）

- **目标 kernel**：`aiter::mxfp4_moe::gemm2::kernel<655360, 385, 512, 7168, 9, 16, (EpilogPolicy)0=Atomic, true, 0, false>`
  = `<MAX_M, NUM_EXPERTS=385, K=D_INTER=512, N_OUT=D_HIDDEN=7168, TOPK=9, BM=16, kEpilog=Atomic, kUseNT=true, kXcdSwizzle=0, kMxfp4Out=false>`
- **源码**：
  - 模板 kernel：`aiter/csrc/kernels/mxfp4_moe/gemm_a4w4/gemm2_a4w4.cuh:51`
  - atomic epilogue：`common/mxfp4_epilogs.hpp::apply_atomic_bf16_epilog:149`
    （另有 `apply_bf16_flat_epilog_bm128:218`、`apply_mxfp4_flat_epilog_bm128:245` 供大 M nonatomic 路径，本实例不用）
  - `atomic_pk_add_bf16`：`mxfp4_epilogs.hpp:15`（封装 `__builtin_amdgcn_global_atomic_fadd_v2bf16`）
  - MFMA 封装：`common/mfma_f4f4.hpp::mfma_f4f4_vgpr:22`
  - 实例化 blob：`aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT.cu`
  - 分发：`gemm_a4w4/mxfp4_moe_gemm.cu::mxfp4_moe_gemm2_a4w4_kernel`
  - host op：`aiter.mxfp4_moe_gemm2_a4w4`（`aiter/aiter/ops/mxfp4_moe.py:160`）
  - 权重/scale preshuffle：`aiter/aiter/ops/shuffle.py::shuffle_weight_a16w4:53` / `shuffle_scale_a16w4:141`
- **ISA dump**（gfx950 / CDNA4）：
  - ASM：`aiter/mxfp4_moe_isa/mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT.gfx950.s`（480 行）
  - LLVM IR：`...gfx950.ll`（461 行）
  - kernel 符号：`_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEE...`
- **ISA 资源元数据（已核对 .s）**：`.amdhsa_next_free_vgpr 85`、`.amdhsa_next_free_sgpr 35`、
  `.amdhsa_group_segment_fixed_size 16384`、`.amdhsa_accum_offset 88`、`vgpr_spill_count 0`。
- **指令语义参考**：`.cursor/rules/amd-instinct-cdna4-instruction-set-architecture.txt`、
  `.cursor/rules/amd-cdna-4-architecture-whitepaper.txt`。
- **benchmark 形状（KIMI）**：NE=385、K=D_INTER=512、N_OUT=D_HIDDEN=7168、TOPK=9，M=16（BM=16 atomic），
  gfx950 MI355X。a4w4：A=fp4（gemm1 产出的中间量，sorted 序）、W=fp4、e8m0 per-32；输出 bf16，atomic 累加到
  `out[token]`。

## 目录

- [1. Kernel 概览与软件流水](#1-kernel-概览与软件流水)
  - [1.1 Buffering：A(inter) / A_scale / B(w2) / B_scale / out 的分布](#11-buffering)
  - [1.2 epilogue 归约拆成子阶段：cshuffle → ×weight → atomic](#12-归约子阶段)
  - [1.3 Wave / 调度结构](#13-wave--调度结构)
  - [1.4 精化伪代码](#14-精化伪代码)
  - [1.5 每操作主指令与计数](#15-每操作主指令与计数)
  - [1.6 调度原语](#16-调度原语)
- [2. 完整 GM/LDS/VGPR 布局图](#2-完整-gmldsvgpr-布局图)
  - [2.1 A：inter（sorted-direct，GM → LDS → VGPR）](#21-a-inter)
  - [2.2 B：a16w4 权重 w2（GM → VGPR）](#22-b-a16w4)
  - [2.3 A_scale / B_scale（make_preshuffle tile）](#23-scale)
  - [2.4 累加器与 cshuffle（VGPR → LDS）](#24-累加器与-cshuffle)
  - [2.5 输出：atomic 累加到 out[token]（LDS → GM）](#25-输出)
  - [2.6 张量驻留汇总](#26-驻留汇总)
  - [2.7 底层调用链与 issue 计数](#27-调用链与计数)

---

## 1. Kernel 概览与软件流水

本 kernel 是 MoE 第二层（down 投影）GEMM。对每个「(token,slot) → 专家 e」计算 `out += w2 · inter`：
`K = inter_dim = 512`（仅 **2 个 K-tile**，K 很短，epilogue 占比相对高），`N = model_dim = 7168`
（`num_n_blocks = 7168/256 = 28`）。fp4×fp4 缩放 MFMA 累加进 `accm`，**topk 归约在 atomic epilogue 完成**：
对每个 sorted 行取 `weight = sorted_weights[sorted_pos]`，把 `acc*weight` 转 bf16 用
`global_atomic_fadd_v2bf16` **原子加到 `out[token]`**（topk 个 slot 因 token 相同自动汇总），故无需独立的
scatter_reduce kernel。

一个 workgroup = 256 线程 = 4 wave，处理一个 `BM×256` 的输出 tile。grid 按 `(m_block_idx, n_block_idx)`
展开，`total_m_blocks = cumsum/BM`，`total_tiles = total_m_blocks*28`。`__launch_bounds__(256, 4)`
（BM=16 atomic → 4 waves/EU）。本实例 `kAtomic=true`（`kUseAGPR=kPersistent=false`，BM=128 才用）。

A（中间量 `inter_sorted_quant`）是 **sorted 序**，gemm2 **sorted-direct 读**（行 = 排序位，不经 m_indices
gather）；`sorted_token_ids` 仅在 epilogue 用来把 sorted 行还原成 `token` 作 atomic 写地址。

<a id="11-buffering"></a>

### 1.1 Buffering：A(inter) / A_scale / B(w2) / B_scale / out 的分布

LDS 由 `LDSLayout<kAtomic, BM, BK, BN, kStages, kAS_LDS_slot_bytes>` 描述（`gemm2_a4w4.cuh:127`）。
atomic 路径下 `s_Aq[kStages][BM][BK/2]` 与 epilogue 的 `lds_acc[BM*BN]` 复用同一块 LDS：
`lds_acc = 16*256*4 = 16384 B`，与 `.amdhsa_group_segment_fixed_size 16384` 吻合。

| 张量 | Global memory | LDS | VGPR（每 lane）|
|---|---|---|---|
| A = inter | `A_q[max_sorted, 512/2]` fp4x2（gemm1 输出，sorted 序）| sorted-direct → `s_Aq[kStages][16][128]`（XOR16 swizzle）| `a[kMChunks=1][2]`=i32x4×2 |
| A_scale | tile 布局 e8m0 | 不过 LDS（atomic 路径直接寄存器）| `a_scale_v[1][2]`、`a_scale_aiter[1]` |
| B = w2 | `B_q[385, 7168, 512/2]` fp4x2（`shuffle_weight_a16w4(w2,16,False)`）| 不过 LDS（寄存器驻留）| `b[kStages=2][4][2]`=i32x4 |
| B_scale = w2_scale | `B_scale[385*7168, 512/32]` e8m0 | 不过 LDS | `b_scale_v[2][2]` |
| out | `out[M, 7168]` bf16（被 sort_quant 预清零）| 经 `lds_acc[16*256]` cshuffle | `accm[kMChunks=1][4]`=f32x4×4 |

VGPR 仅 85（远少于 gemm1 的 139）：K 短（2 tile）、无 inline-quant、单累加器（down 投影无 gate/up 两路）。

<a id="12-归约子阶段"></a>

### 1.2 epilogue 归约拆成子阶段：cshuffle → ×weight → atomic

down 投影本身平凡（2 K-tile MFMA），**核心在 atomic epilogue**（`apply_atomic_bf16_epilog`），分三子阶段：

1. **cshuffle**：`accm[1][4]` 经 `lds_acc` 重排成「行连续 N」（`col = wave_n*64 + J*16 + lane%16`，
   单一 down-proj 列，无 gate/up 拆分），`__syncthreads()`。
2. **取行 + ×weight**：按 `(m_lane=tid/32, n_lane=tid%32)`，`M_REPS = BM/8 = 2`，每个
   `sorted_pos = m_row + mr*8 + m_lane` → `token_id = sorted_token_ids[sorted_pos] & 0x00FFFFFF`；
   **padding guard**：`if token_id >= M: continue`；`weight = sorted_weights[sorted_pos]`。
3. **atomic 累加**：每行 `kNAtomic=4` 次、每次 2 列（`kColsPerStride=2`），`acc*weight` 转 bf16 →
   `atomic_pk_add_bf16(out[token*7168 + n_block*256 + col_start + s*64], {v0*w, v1*w})`。

topk 个 (token,slot) 因 `token_id` 相同，原子加自然汇总成最终 `out[token]`。

<a id="13-wave--调度结构"></a>

### 1.3 Wave / 调度结构

- 256 线程 = 4 wave；`wave=tid/64`、`wave_n=wave`、`lane=tid%64`。4 wave 各算 256 列 tile 的 64 列段。
- **K 很短（2 tile）**：`K_TILES_TOTAL=2`、`kStages=2` → `kUnroll = K_TILES_TOTAL-kStages = 0`，**无 steady-state
  主循环**，只有「prologue 填 2 tile + drain 算 2 tile」。所以 gemm2 是 prologue+drain 结构。
- A-load（BM=16）：仅 `wave<2` 参与，每 wave 用 `buffer_load_lds` 把 8 行 inter 直送 LDS（sorted-direct，
  `cached_actual_row = m_row + wave*8 + row_off`，`gemm2_a4w4.cuh:294-297`）。
- atomic 路径不用 `s_setprio`/AGPR/持久网格（那是 BM=128 nonatomic 的事）。

<a id="14-精化伪代码"></a>

### 1.4 精化伪代码

对应 `run_one` 的 atomic 分支（`gemm2_a4w4.cuh:284-418`）。K=512 仅 2 tile → 无 steady-state，结构是
「一次性加载全部 + 2-tile MFMA drain + atomic epilogue」。每行一条语句。

```cpp
// ── grid 入口（gemm2_a4w4.cuh:420+，非持久网格）──
total_m_blocks = cumsum_tensor[0] / 16
// 非持久：m_block_idx = pid / 28; n_block_idx = pid % 28  （kXcdSwizzle=0）
e = sorted_expert_ids[m_block_idx]

// ── run_one(m_block_idx, n_block_idx, e) ──
m_row   = m_block_idx * 16
c_zero  = {0,0,0,0}
// A 是 sorted-direct：cached row = 排序位本身（非 m_indices gather）
if wave < 2:
    cached_actual_row[wave] = m_row + wave*8 + lane/8
for j in 0..3:
    b_load_s_base[j] = (e*N_OUT + n_block_idx*256 + wave_n*64 + j*16) * (K/2)
for mw in 0..1:
    b_scale_s_base[mw] = (e*kBS_per_expert_dw + (mni_base+mw)*kBS_stride_n0_dw) * 4
chunk_base       = m_row / BM        // BM=16
a_scale_s_base[0]= chunk_base * kAS_per_chunk_dw * 4

// ── 一次性加载（两个 K-tile 的 A→LDS、B→寄存器、scale）──
issue_a_load_lds(slot=0, kt=0, cached_actual_row)     // buffer_load_lds（wave<2 各 8 行）
issue_a_load_lds(slot=1, kt=1, cached_actual_row)
sched_barrier(0)
issue_a_scale_load_atomic()                           // a_scale_v[0][0..1] = 2× buffer_load_b32 @ +0/+256
issue_b_scale_load_ku<0>(b_scale_v[0])                // 2× buffer_load_b32
issue_b_scale_load_ku<1>(b_scale_v[1])                // 2×
for j in 0..3: issue_b_load_j<0>(b[0], j)             // 4 j × 2 b128
for j in 0..3: issue_b_load_j<1>(b[1], j)             // 4 j × 2 b128

// ── 2-tile MFMA drain ──
for S in 0..1:
    kt    = K_TILES_TOTAL - kStages + S = S           // 0, 1
    slot_ = kt % 2
    s_waitcnt vmcnt(S==0 ? 23 : 22)                   // 跨 wave 正确性栅栏（loads 落地后再 ds_read）
    s_barrier
    issue_a_ds_read(slot_)                            // ds_read_b128 s_Aq → a
    issue_a_scale_ds_read_ku_atomic<kt>()             // a_scale_aiter = a_scale_v[*][kt]
    for J in 0..3:
        issue_mfma_cluster<J, kInit=(S==0)>(slot_)    // BM=16: mfma_f4f4_vgpr<0,in_b> + <2,2+in_b>

// ── atomic epilogue ──
__syncthreads()
apply_atomic_bf16_epilog<N_OUT, BM>(accm, out_bf16, sorted_token_ids, sorted_weights,
    m_row, n_block_idx, wave_n, lane, tid, M, lds.lds_acc)
```

atomic epilogue（`mxfp4_epilogs.hpp:149-215`，每行一条语句）：

```cpp
// 1) cshuffle：accm → lds_acc（单一 down-proj 列，无 gate/up 拆分）
for i in 0..kMChunksEpi-1:                  // BM=16 → 1
    row_base = i*16 + (lane/16)*4
    for J in 0..3:
        col = wave_n*64 + J*16 + (lane%16)
        for v in 0..3: lds_acc[(row_base+v)*256 + col] = accm[i][J][v]
__syncthreads()
// 2) 取行 + ×weight + atomic（M_REPS = BM/8 = 2）
m_lane = tid/32;  n_lane = tid%32;  col_start = n_lane*2
for mr in 0..M_REPS-1:                       // 2
    row_in_block = mr*8 + m_lane
    sorted_pos   = m_row + row_in_block
    token_id     = sorted_token_ids[sorted_pos] & 0x00FFFFFF
    if token_id >= M: continue               // padding guard
    weight       = sorted_weights[sorted_pos]
    for s in 0..3:                           // kNAtomic=4
        v[s][0] = lds_acc[row_in_block*256 + col_start + s*64 + 0]
        v[s][1] = lds_acc[row_in_block*256 + col_start + s*64 + 1]
    row_addr = &out[token_id*7168 + n_block_idx*256 + col_start]
    for s in 0..3:
        pkbf16 = { (bf16)(v[s][0]*weight), (bf16)(v[s][1]*weight) }
        atomic_pk_add_bf16(row_addr + s*64, pkbf16)     // global_atomic_pk_add_f16/bf16
```

<a id="15-每操作主指令与计数"></a>

### 1.5 每操作主指令与计数

「一个 warp 处理整个 kernel（2 K-tile + atomic epilogue）」，已用 `.s` 核对。tiling：`K_TILES_TOTAL=2`、
每 tile 4 J、BM=16 → `kMChunks=1`、`kSubBlocks=1`、epilogue `M_REPS=BM/8=2`、`kNAtomic=4`。

| # | 操作 | 主指令 | 计数 | 推导 |
|---|---|---|---|---|
| 1 | GEMM MFMA | `v_mfma_scale_f32_16x16x128_f8f6f4` | **16** | 2 MFMA/J × 4 J × 2 tile = 16（与 .s `v_mfma`=16 一致）|
| 2 | B(a16w4) 载入 | `buffer_load_dwordx4`（`issue_b_load_j`）| 16 | 4 J × 2 b128 × 2 tile = 16 |
| 3 | A(inter) 载入 | `buffer_load`(direct-to-LDS, `issue_a_load_lds`) | 2 | 2 个 K-tile 各 1（wave<2）|
| 4 | A_scale 载入 | `buffer_load_dword`（`issue_a_scale_load_atomic`）| 2 | a_scale_v[0][0..1] 2× b32 |
| 5 | B_scale 载入 | `buffer_load_dword`（`issue_b_scale_load_ku` ×2）| 4 | 2 ku × 2 mw = 4 |
| — | **buffer_load 合计** | — | **24** | 16+2+2+4（与 .s `buffer_load`=24 一致）|
| 6 | A 读 LDS | `ds_read_b128`（`issue_a_ds_read`）| — | 与 cshuffle 读共同构成 .s `ds_read`=12 |
| 7 | cshuffle 写 LDS | `ds_write_b32`（accm→lds_acc）| — | 构成 .s `ds_write`=8 |
| 8 | topk 归约 atomic | `global_atomic_pk_add_f16`（`atomic_pk_add_bf16`）| **8** | `M_REPS=2 × kNAtomic=4 = 8`（与 .s `global_atomic`=8 一致）|
| 9 | ×weight | `v_pk_mul`/`v_cvt_pk_bf16`（`v[s]*weight` 转 bf16）| — | epilogue 每对元素 |

> 三个锚点（MFMA 16、buffer_load 24、global_atomic 8）均与 ISA dump 精确吻合。无 `cvt_scalef32_pk_fp4`
> （atomic 路径输出 bf16，不再量化；只有 `kMxfp4Out` nonatomic 路径才有）。

<a id="16-调度原语"></a>

### 1.6 调度原语

| 原语 | 是否真实 ISA | 语义 |
|---|---|---|
| `__builtin_amdgcn_sched_barrier(0)` | 否 | AMDGPU `SCHED_BARRIER`(mask=0) 编译期硬栅栏，阻止任何类别跨越重排；这里隔开「加载段」与「MFMA drain 段」。|
| `s_waitcnt vmcnt(23)` / `vmcnt(22)`（内联 asm）| 是 | 等待 VMEM 计数降到 23/22，确保 B/A 的 `buffer_load` 落地后才 `ds_read`/MFMA 消费；是**跨 wave 正确性栅栏**而非性能旋钮（`gemm2_a4w4.cuh:398-403`）。`s_barrier` 紧随其后做 workgroup 同步。|

atomic 路径**不用** `s_setprio`（那是 BM=128 nonatomic + 持久网格场景）；本 kernel `kUseAGPR=kPersistent=false`。

## 2. 完整 GM/LDS/VGPR 布局图

记号：`lane=tid%64`、`wave=tid/64`、`wave_n=wave`。fp4x2 = 1 字节 2 个 fp4。

<a id="21-a-inter"></a>

### 2.1 A：inter（sorted-direct，GM → LDS → VGPR）

A 是 gemm1 输出的中间量 `inter_sorted_quant[max_sorted, 512/2]`（**sorted 序**），行 = 排序位本身，
不经 `m_indices` gather（与 gemm1 的 A 不同）。

**GM → LDS**（`issue_a_load_lds`，BM=16 仅 `wave<2`，`gemm2_a4w4.cuh:141-165`）：

```cpp
// cached_actual_row[wave] = m_row + wave*8 + lane/8   （sorted 行号，直接用）
lds_row = wave * 8                           // wave0 → 行0..7，wave1 → 行8..15
mask    = lds_swizzle_mask<BK/2=128>(lds_row + lane/8)
voffset = (((lane % 8) * 16) ^ mask) + cached_actual_row[wave] * (K/2)   // K/2=256 字节/行
buffer_load_lds(A_q_rsrc, &s_Aq[slot][lds_row][0], /*size=*/16, voffset, kt*(BK/2), 0, 0)
```

**LDS → VGPR**（`issue_a_ds_read`，`gemm2_a4w4.cuh:168-182`）：

```cpp
lane_row = lane % 16
lane_col = (lane / 16) * 16
mask     = lds_swizzle_mask<128>(lane_row)
for k in 0..1:
    lds_col = (lane_col + k*64) ^ mask
    a[0][k] = *(i32x4*)&s_Aq[lds_slot][lane_row][lds_col]    // ds_read_b128
```

**可视化（GM sorted 行 → LDS 行，BM=16）**：

| wave | lane | sorted 行（GM）| LDS 行 |
|---:|---:|---:|---:|
| 0 | 0..7 | m_row+0 | 0 |
| 0 | 8..15 | m_row+1 | 1 |
| ... | ... | ... | ... |
| 0 | 56..63 | m_row+7 | 7 |
| 1 | 0..7 | m_row+8 | 8 |
| ... | ... | ... | ... |
| 1 | 56..63 | m_row+15 | 15 |

wave 2,3 不参与 A-load（16 行 × 256 字节 = 4096 字节，仅需 wave 0,1 的 128 lane 即可覆盖）。

<a id="22-b-a16w4"></a>

### 2.2 B：a16w4 权重 w2（GM → VGPR，不过 LDS）

B（`w2` = `shuffle_weight_a16w4(w2,16,False)`，`[385,7168,512/2]` fp4x2）寄存器驻留，公式与 gemm1 同形
（仅 N_OUT、K 值不同）：

```cpp
b_load_s_base[j] = (e*N_OUT + n_block_idx*256 + wave_n*64 + j*16) * (K/2)   // N_OUT=7168, K/2=256
v_voff = (lane/16)*256 + (lane%16)*16 + K_C*2048
buffer_load_b128_imm<0,    aux=kUseNT?2:0>(b[..][j][0], B_q_rsrc, v_voff, b_load_s_base[j])
buffer_load_b128_imm<1024, aux>          (b[..][j][1], B_q_rsrc, v_voff, b_load_s_base[j])
```

`gate_up=False`（w2 无 gate/up 交织，是单纯的 down 权重）。`(lane/16, lane%16)` 选 a16w4 的
`[KLane=4, NLane=16]`，每 lane 一个 16B=32 fp4 段；`j∈0..3` 覆盖 256 列 N-tile 的 4 个 16 列子块。
一个 lane 的 `b[..][j][0..1]` = 256 fp4 = MFMA B 操作数（K=128 × 2 段）。

<a id="23-scale"></a>

### 2.3 A_scale / B_scale（make_preshuffle tile）

常量（K=512 → `kAS_c_k1=(512/32)/4/2=2`、`kAS_per_chunk_dw=1*2*64=128`；`kBS_c_n1=7168/16/2=224`、
`kBS_c_k1=2`、`kBS_stride_k0_dw=64`、`kBS_stride_n0_dw=128`、`kBS_per_expert_dw=224*128=28672`）。

**A_scale**（atomic 路径，寄存器，`issue_a_scale_load_atomic`，`gemm2_a4w4.cuh:184-193`）：

```cpp
chunk_base       = m_row / BM                       // BM=16
a_scale_s_base[0]= chunk_base * kAS_per_chunk_dw * 4
v_voff           = ((lane/16)*16 + (lane%16)) * 4
a_scale_v[0][0]  = buffer_load_b32_imm<  0>(A_scale_rsrc, v_voff, a_scale_s_base[0])  // ku=0
a_scale_v[0][1]  = buffer_load_b32_imm<256>(A_scale_rsrc, v_voff, a_scale_s_base[0])  // ku=1
```

**B_scale**（`issue_b_scale_load_ku<KU>`，`gemm2_a4w4.cuh:240-248`）：

```cpp
v_voff = ((lane/16)*16 + (lane%16)) * 4
IMM    = KU * (kBS_stride_k0_dw * 4)                 // KU∈{0,1}
b_scale_v[KU][mw] = buffer_load_b32_imm<IMM>(B_scale_rsrc, v_voff, b_scale_s_base[mw])  // mw∈{0,1}
```

**MFMA 消费 scale**（CDNA4 ISA §7.2.1，line 4321-4362）：block size=32（K），128/32=4 scale/行，M=N=16 →
64 个 8-bit scale = 1/4 VGPR。`mfma_f4f4_vgpr<AB,BB>` 的 `op_sel_a=AB∈{0,2}`、`op_sel_b=BB∈{in_b,2+in_b}`
按 2-bit OPSEL 码选 e8m0 dword 的字节（`00`→[7:0]…`11`→[31:24]）。gemm2 BM=16(pack_M=1) 用 `{0,2}`
选两个 K 子组的字节（与 gemm1 epilogue 写 e8m0 时的 `(ku,ikxdl)` 布局配对，见 `mxfp4_gemm1` §2.5）。

<a id="24-累加器与-cshuffle"></a>

### 2.4 累加器与 cshuffle（VGPR → LDS）

`accm[kMChunks=1][4]`（`f32x4`×4，每 lane 64 f32）。down 投影单累加器（无 gate/up）。cshuffle 写
`lds_acc[16*256]`（`mxfp4_epilogs.hpp:161-172`）：

```cpp
row_base = i*16 + (lane/16)*4               // i=0；lane/16∈0..3 → 行 0..15
col      = wave_n*64 + J*16 + (lane%16)     // 4 wave × 64 = 256 列；无 gate/up 偏移
for v in 0..3: lds_acc[(row_base+v)*256 + col] = accm[i][J][v]
```

读回按 `(m_lane=tid/32, n_lane=tid%32)`，`col_start = n_lane*2`，每行取 `kNAtomic=4` 段 × 2 列。

```
lds_acc row r (r=0..15):  col[0:256) = 一行 down-proj 结果（4 wave 各 64 列）
读回：lane(m_lane=r%8 of mr) 取 col_start=n_lane*2，4 段（s*64）各 2 列 → 8 个值 ×weight 后 atomic
```

<a id="25-输出"></a>

### 2.5 输出：atomic 累加到 out[token]（LDS → GM）

```cpp
sorted_pos = m_row + mr*8 + m_lane          // mr∈0..1
token_id   = sorted_token_ids[sorted_pos] & 0x00FFFFFF
if token_id >= M: continue                  // padding guard（漏掉 padding 行）
weight     = sorted_weights[sorted_pos]
row_addr   = &out[token_id*7168 + n_block_idx*256 + n_lane*2]    // out[token, n]
for s in 0..3:
    atomic_pk_add_bf16(row_addr + s*64, { (bf16)(v[s][0]*weight), (bf16)(v[s][1]*weight) })
```

`global_atomic_pk_add_f16/bf16`（CDNA4）对 2×bf16 打包做原子加：topk 个 (token,slot) 落到**同一**
`out[token]` 行，硬件原子加自然汇总成最终结果（输出缓冲已被 sort_quant 预清零）。每 wave 写 4 段 × 跨 64 列 =
覆盖该 wave 的 256 列 N-tile。

<a id="26-驻留汇总"></a>

### 2.6 张量驻留汇总

| 张量 | GM | LDS | VGPR（每 lane）| 备注 |
|---|---|---|---|---|
| A=inter | `[max_sorted,256]` | `s_Aq[2][16][128]` | `a[1][2]`=i32x4×2 | sorted-direct（无 gather）|
| A_scale | tile e8m0 | —（atomic 寄存器）| `a_scale_v[1][2]` | 2 b32 |
| B=w2 a16w4 | `[385,7168,256]` | — | `b[2][4][2]`=i32x4 | 寄存器驻留 |
| B_scale | `[385*7168,16]` | — | `b_scale_v[2][2]` | make_preshuffle |
| accm | — | 经 `lds_acc[16*256]` | `accm[1][4]`=f32x4×4 | 单 down-proj 累加器 |
| out | `[M,7168]` bf16（预清零）| — | epilogue 临时 `v[4][2]` | atomic_pk_add 汇总 topk |

资源：VGPR 85、SGPR 35、LDS 16384 B、spill 0。

<a id="27-调用链与计数"></a>

### 2.7 底层调用链与 issue 计数

```
mxfp4_moe_gemm2_a4w4 (host, mxfp4_moe.py:160)
└─ mxfp4_moe_gemm2_a4w4_kernel (mxfp4_moe_gemm.cu)        // 按 kernelName 查 g2_atomic_lookup
   └─ gemm2::launch_atomic<655360,385,512,7168,9,16,true,0>
      └─ gemm2::kernel<...,Atomic,...>                     // gemm2_a4w4.cuh:51
         ├─ run_one(m_block_idx, n_block_idx, e)           // :284
         │  ├─ issue_a_load_lds      // inter sorted-direct → LDS (buffer_load_lds)
         │  ├─ issue_b_load_j        // w2 a16w4 → 寄存器 (buffer_load_b128)
         │  ├─ issue_a_scale_load_atomic / issue_b_scale_load_ku   // scale (buffer_load_b32)
         │  ├─ issue_a_ds_read       // ds_read_b128 s_Aq → a
         │  ├─ issue_mfma_cluster<J> → mfma_f4f4_vgpr      // v_mfma_scale_f32_16x16x128_f8f6f4
         │  └─ apply_atomic_bf16_epilog                    // mxfp4_epilogs.hpp:149
         │     └─ atomic_pk_add_bf16 → global_atomic_pk_add_f16   // acc*weight → out[token]
```

**whole-kernel issue 计数（warp，已核对 .s）**：

| 指令 | 计数 | 推导 |
|---|---:|---|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16 | 2/J × 4 J × 2 tile |
| `buffer_load` | 24 | B 16 + A 2 + A_scale 2 + B_scale 4 |
| `ds_read` | 12 | A 读 + cshuffle 读 |
| `ds_write` | 8 | cshuffle 写 |
| `global_atomic`（pk_add_f16）| 8 | M_REPS=2 × kNAtomic=4 |
| `cvt_scalef32_pk_fp4` | 0 | atomic 输出 bf16，不再量化 |
| VGPR / SGPR / LDS / spill | 85 / 35 / 16384 / 0 | `.amdhsa_*` |

> 锚点（MFMA 16、buffer_load 24、global_atomic 8）均与 ISA dump 吻合。与 FlyDSL gemm2 的接口/布局差异
> （尤其 inter 是 sorted-major vs token-major）见 `mxfp4_06_gemm2.md §7`。

