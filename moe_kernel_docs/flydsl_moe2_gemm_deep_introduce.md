# FlyDSL MOE2 GEMM (`mfma_moe2_*_cshuffle_*_vscale_*`) 深度介绍 — down 投影 + atomic topk 归约

> 本文是 `fly_fn`（tuned FlyDSL 路径：`fused_moe(hidden, fly_w...)` → opus 排序 + FlyDSL 2-stage GEMM）中
> MoE **stage2 GEMM** 的深度剖析，遵循 `.cursor/rules/kernel-deep-introduction-doc.mdc`。kernel 由 Python
> 构建 MLIR、JIT 编译为 gfx950 机器码。

## 文档元信息（benchmark / trait / 构建配置）

- **目标 kernel（编译符号）**：`mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_pm1`
  = tile_m=32, tile_n=128, tile_k=256, accumulate=**atomic**, doweight_stage2=true（编译符号 `vscale`）,
  cshuffle epilogue。（`atomic`/`reduce` 不进符号名，仅进 lru-cache key。）
- **源码**（Python 构建 MLIR）：
  - kernel builder：`aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py::compile_mixed_moe_gemm2:2882`
  - 内核体（嵌套 fn）：`moe_gemm2:3185`；launcher：`launch_mixed_moe_gemm2:4863`；名构造 `:2995-2999`
  - 复用 helper：`mfma_preshuffle_pipeline.py`（`make_preshuffle_b_layout:168`、`make_preshuffle_scale_layout:111`、
    `swizzle_xor16:29`、`lds_load_packs_k64`、`prefetch_x_to_lds`）；epilogue：`mfma_epilogues.py`（cshuffle `:85`，
    `write_row_to_lds` 内做 `vscale`，`store_pair` 做 atomic）
  - host：`aiter/aiter/ops/flydsl/moe_kernels.py::flydsl_moe_stage2:1050`
  - 选择 wrapper：`aiter/aiter/fused_moe.py::_flydsl_stage2_wrapper:909`
  - 权重 = `fly_w2 = aiter.ops.shuffle.shuffle_weight(w2_qt, layout=(16,16))`
- **ISA dump**（gfx950 / CDNA4，FlyDSL JIT 最终汇编）：
  - `aiter/flydsl_dump/mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_pm1/21_final_isa.s`（541 行）
  - 中间 MLIR 阶段同目录，用于 IR→ISA 映射
- **ISA 资源元数据（已核对 21_final_isa.s）**：`.amdhsa_next_free_vgpr 68`（`.vgpr_count 68`）、
  `.amdhsa_next_free_sgpr 40`（`.sgpr_count 46`）、`.agpr_count 0`、`.group_segment_fixed_size 13312`、
  `.vgpr_spill_count 0`。
- **指令语义参考**：`.cursor/rules/amd-instinct-cdna4-instruction-set-architecture.txt`、
  `.cursor/rules/amd-cdna-4-architecture-whitepaper.txt`。
- **benchmark 形状（KIMI）**：NE=385、model_dim=D_HIDDEN=7168、inter_dim=D_INTER=512、TOPK=9，gfx950 MI355X。
  a4w4：A=fp4（gemm1 输出的中间量，**token-major** `[tok,topk,256]`，经 sorted_token_ids gather）、W=fp4、
  e8m0 per-32；输出 bf16，atomic 累加到 `out[token]`。

## 目录

- [1. Kernel 概览与软件流水](#1-kernel-概览与软件流水)
  - [1.1 Buffering：inter(A) / a2_scale / w2 / w2_scale / out](#11-buffering)
  - [1.2 epilogue 归约拆成子阶段：cshuffle → vscale → atomic](#12-归约子阶段)
  - [1.3 Wave / 调度结构](#13-wave--调度结构)
  - [1.4 精化伪代码](#14-精化伪代码)
  - [1.5 每操作主指令与计数](#15-每操作主指令与计数)
  - [1.6 调度原语](#16-调度原语)
- [2. 完整 GM/LDS/VGPR 布局图](#2-完整-gmldsvgpr-布局图)
  - [2.1 A：inter（token-major，经 sorted_token_ids gather → LDS → VGPR）](#21-a)
  - [2.2 B：w2（make_preshuffle，GM → VGPR）](#22-b)
  - [2.3 a2_scale / w2_scale（make_preshuffle tile）](#23-scale)
  - [2.4 累加器与 cshuffle epilogue（vscale + atomic）](#24-cshuffle)
  - [2.5 输出：atomic 累加到 out[token]（LDS → GM）](#25-输出)
  - [2.6 张量驻留汇总](#26-驻留汇总)
  - [2.7 底层调用链与 issue 计数](#27-调用链与计数)

---

## 1. Kernel 概览与软件流水

FlyDSL stage2 GEMM（DSL 生成）。对每个「(token,slot) → 专家」计算 fp4×fp4 的 down 投影
（`K=inter_dim=512` → 仅 **2 个 K-tile**，`N=model_dim=7168`），**单累加器**（无 gate/up）。cshuffle epilogue
内按 `sorted_weights`（编译符号 `vscale`）乘上路由权重，topk 归约由 `accumulate=atomic` 决定：
`atomic fadd`（agent scope）累加到 `out[token]`，topk slot 自然汇总。

一个 WG = 256 线程（4 wave，`mixed:2920`），算一个 `tile_m=32 × tile_n=128` 输出 tile。grid：
`by = block_id("x")` = N-tile（沿 model_dim），`gx = ceil(model_dim/tile_n)`；`gy` = 持久网格 `cu_num` 或
`ceil(size_expert_ids/persist_m)`。

> 与 HIP gemm2（M8）的根本差异：A（inter）是 **token-major** `[tok,topk,256]`，需按 sorted_token_ids
> **gather**；HIP 的 inter 是 sorted-major、**sorted-direct 读**（见 `mxfp4_06_gemm2.md §7.1`）。

<a id="11-buffering"></a>

### 1.1 Buffering：inter(A) / a2_scale / w2 / w2_scale / out

LDS `.group_segment_fixed_size = 13312 B`（A 的 ping-pong + cshuffle 输出）。`.vgpr_count 68`、
`.sgpr_count 46`、AGPR 0、spill 0 —— 比 gemm1 小（K 短、单累加器）。

| 张量 | Global memory | LDS | VGPR |
|---|---|---|---|
| A = inter | `inter[token_num, topk, 512/2]` fp4x2（gemm1 输出，**token-major**；按 sorted_token_ids gather）| `lds_x_ping/pong` 双缓冲 | A-pack（`lds_load_packs_k64` 产出 i64）|
| a2_scale | `[sorted_size, 512/32]` e8m0（sorted-tile）| 视配置 | scale dword |
| w2 | `fly_w2[E, 7168, 512]` fp4x2（`shuffle_weight((16,16))`）| 不过 LDS（寄存器驻留）| B-pack（i32x*）|
| w2_scale | `[E*7168, 512/32]` e8m0 | 不过 LDS | scale dword |
| out | `out[token, 7168]` bf16（atomic 模式预清零）| 经 cshuffle `lds_out` 重排 | 单 f32 累加器 `acc`（`vec4_f32`×N-rep）|

<a id="12-归约子阶段"></a>

### 1.2 epilogue 归约拆成子阶段：cshuffle → vscale → atomic

down 投影本身平凡（2 K-tile MFMA），核心在 cshuffle + vscale + atomic epilogue（`mfma_epilogues.py`）：

1. **cshuffle**：`acc` 经 `lds_out` 重排成连续 N 向量。
2. **vscale**（`write_row_to_lds` 内）：对每行乘路由权重 `v = v * tw`（`tw = sorted_weights[sorted_pos]`），
   由 `doweight_stage2 = (sorted_weights is not None)` 开启（编译符号 `vscale`）。
3. **atomic**（`store_pair`）：`AtomicRMWOp fadd`（agent scope）把 `v*tw` 转 bf16 累加到 `out[token]` 行
   （token 由 sorted_token_ids 还原）。topk 个 slot 落同一 `out[token]` → 原子加汇总。

> 对照 `reduce` 模式（大 M）：改为 plain non-temporal store 到 `out[token*topk+slot]`，再由 host `torch.sum`
> 归约（[F7]）。本实例是 atomic。

<a id="13-wave--调度结构"></a>

### 1.3 Wave / 调度结构

- WG = 4 wave = 256 线程。4 wave 切分 tile_n=128 → 每 wave 32 列。
- **K 很短（2 tile）**：tile_k=256、K=512 → 2 个 K-tile，LDS ping-pong 双缓冲（`lds_x_ping/pong`），
  `prefetch_x_to_lds` 预取下一 tile A，`compute_tile` 算当前 tile（`mixed:4348-4415`）。
- `rocdl.s_waitcnt(0)` + `gpu.barrier()` 在 tile 边界做同步（`mixed:4345-4346`）。
- `_sbm{N}`（sort_block_m）允许 tile_m 小于排序 block_m（本实例未触发该后缀）。

<a id="14-精化伪代码"></a>

### 1.4 精化伪代码

对应 `moe_gemm2`（`mixed_moe_gemm_2stage.py:3185+`）的 atomic / vscale / cshuffle 路径。每行一条语句。

```python
# ── 编译期 layout ──
layout_b       = make_preshuffle_b_layout(c_n=E*model_dim, c_k=inter_dim//pack_K, kpack_bytes=16)
layout_a_scale = make_preshuffle_scale_layout(c_mn=sorted_size, c_k=inter_dim)
layout_b_scale = make_preshuffle_scale_layout(c_mn=E*model_dim, c_k=inter_dim)

# ── 运行期索引 ──
by      = block_id.x                       # N-tile（沿 model_dim），gx=ceil(model_dim/tile_n)
wgid    = persistent / by 映射
expert  = expert_ids[m_block]
# A 是 token-major：每 sorted 行经 sorted_token_ids 还原 (token,slot) 再 gather
acc = vec4_f32(0)                          # 单累加器（无 gate/up）

# ── A gather → LDS（ping），预取 ──
prefetch_x_to_lds(k=0, lds_x_ping)         # 按 sorted_token_ids gather inter[token,slot]
b_ping = load_b_tile(0)                    # w2 make_preshuffle → 寄存器
a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(0)

# ── K 主循环（K=512 → 2 tile，ping-pong）──
for kt in 0..1:                            # 2 个 K-tile
    rocdl.s_waitcnt(0)
    gpu.barrier()
    a_pack = lds_load_packs_k64(row_a_lds, col_base, lds_x_cur)   # ds_read A-pack（XOR16）
    prefetch_x_to_lds(next_k, lds_x_other)                        # 预取下一 tile A
    b_next = load_b_tile(next_k_bk)
    a_scale_next, b_scale_next = prefetch_ab_scale_tile(next_k)
    acc = compute_tile(acc, b_cur, lds_x_cur, a_scale_cur, b_scale_cur, a_pack, ...)
        # 内部：E_M·E_N·E_K 个 rocdl.mfma_scale_f32_16x16x128_f8f6f4
    hot_loop_scheduler()                   # sched_group_barrier 提示
    gpu.barrier()

# ── cshuffle + vscale + atomic epilogue ──
write_row_to_lds(acc):                     # cshuffle 重排 + vscale
    tw = sorted_weights[sorted_pos]
    v  = v * tw                            # vscale（×路由权重）
__syncthreads()
store_pair(...):                           # atomic 模式
    token = sorted_token_ids[sorted_pos] & 0x00FFFFFF
    AtomicRMWOp.fadd(out[token*7168 + n], (bf16)v)   # agent scope，topk 汇总
```

<a id="15-每操作主指令与计数"></a>

### 1.5 每操作主指令与计数

「一个 wave 处理整个 kernel（2 K-tile + epilogue）」，已用 `21_final_isa.s` 核对。tiling：tile_m=32 →
`E_M=2`；tile_n=128，4 wave 各 32 列 → `E_N=2`；tile_k=256 → `E_K=2`；K=inter_dim=512 → 2 K-tile；单 down-proj。

| # | 操作 | 主指令 | 计数 | 推导 |
|---|---|---|---|---|
| 1 | down-proj MFMA | `v_mfma_scale_f32_16x16x128_f8f6f4` | **16** | `E_M·E_N·E_K · 2 K-tile = 2·2·2·2 = 16`（与 .s `v_mfma_scale`=16 一致；单流，无 gate/up）|
| 2 | A gather + B + scale 载入 | `buffer_load` | 27 | A gather DMA + w2 B-pack + a2/w2 scale，合计 27（.s 实测）|
| 3 | A 读 LDS | `ds_read`（`lds_load_packs_k64`）| 18 | A-pack + scale + cshuffle 读，合计 18 |
| 4 | A/输出 写 LDS | `ds_write` | 17 | gather A 入 LDS + cshuffle 写 lds_out，合计 17 |
| 5 | topk 归约 atomic | `global_atomic_pk_add_f16`（AtomicRMWOp fadd）| **8** | epilogue 每 wave 8 次原子加（与 .s `global_atomic`=8 一致）|
| 6 | vscale | `v_pk_mul` / `v_cvt_pk_bf16` | — | `v*tw` 转 bf16 |

> 锚点：MFMA **16**（单 down-proj，无 gate/up，故是 gemm1 448 的 1/28×… 实为 2·2·2·2）、`global_atomic` **8**
> 均与 ISA dump 吻合。无 `v_exp`（gemm2 无激活）、无 `v_cvt_pk_fp4`（输出 bf16）。

<a id="16-调度原语"></a>

### 1.6 调度原语

| 原语 | 是否真实 ISA | 语义 |
|---|---|---|
| `rocdl.sched_barrier(0)` / `hot_loop_scheduler()`（sched_group_barrier 提示）| 否 | 编译期调度提示/栅栏，约束 LLVM 把 MFMA 与 DMA/ds_read 交错；不发射机器码。|
| `rocdl.s_waitcnt(0)` | 是 | `s_waitcnt` 等待所有计数归零，确保上一 tile 的 load/store 落地后再进下一 tile（跨 wave 正确性）。|
| `gpu.barrier()` | 是（`s_barrier`）| workgroup 同步，配合 ping-pong 双缓冲。|

atomic 模式不用 `s_setprio`（持久网格 + nonatomic 才更可能用）；本实例 AGPR 0。

## 2. 完整 GM/LDS/VGPR 布局图

记号：`tx=thread_id.x`（0..255）、`wave=tx/64`、`lane=tx%64`。fp4x2 = 1 字节 2 个 fp4。

<a id="21-a"></a>

### 2.1 A：inter（token-major，经 sorted_token_ids gather → LDS → VGPR）

A = gemm1 输出的中间量 `inter[token_num, topk, 512/2]` fp4x2，**token-major**。gemm2 对每个 sorted 行用
`sorted_token_ids` 还原 `(token, slot)` 后 **gather** `inter[token, slot, :]`（与 HIP gemm2 的 sorted-direct
读不同——这是两条流水线 stage1↔stage2 契约的根本差异）。

**GM → LDS（gather DMA）**：`prefetch_x_to_lds` 按 gather 索引把每 lane 的 16B 段送入 `lds_x_ping/pong`，
XOR16 swizzle（`swizzle_xor16(row, col, k_blocks16) = col ^ ((row & (k_blocks16-1))*16)`）。

**LDS → VGPR（MFMA A-pack）**：`lds_load_packs_k64(row_a_lds, col_base, lds_x_cur)` 取 i64 A-pack
（一个 K64 micro-step），ping/pong 交替。

**可视化（sorted 行 → token gather）**：

| sorted_pos | sorted_token_ids[sorted_pos] | (token, slot) | gather 源 |
|---:|---|---|---|
| m_row+0 | packed0 | (packed0&0xFFFFFF, …) | inter[token0, slot0, :] |
| m_row+1 | packed1 | (token1, slot1) | inter[token1, slot1, :] |
| ... | ... | ... | ... |

即 LDS 里第 r 行 = 第 r 个 sorted 行对应的 `inter[token,slot]`（间接寻址），与 HIP 的「LDS 行=排序位直读」
形成对比。

<a id="22-b"></a>

### 2.2 B：w2（make_preshuffle，GM → VGPR）

B = `fly_w2[E, 7168, 512/2]` fp4x2（`shuffle_weight((16,16))`），寄存器驻留，layout 与 gemm1 B 同形
（仅 c_n、c_k 取值不同）：

```python
# make_preshuffle_b_layout(c_n=E*model_dim=385*7168, c_k=inter_dim//pack_K, kpack_bytes=16)
# shape  = (n0=c_n/16, c_k0=c_k_bytes/64, klane=4, 16, kpack=16)
# stride = (c_k0*1024, 1024, 256, 16, 1)
b_off = n0_idx*stride_n0 + k0_idx*1024 + klane*256 + nlane*16 + kpack
```

单 down-proj（无 gate/up），故只有一条 B 流（gemm1 有 gate+up 两条，MFMA ×2）。`load_b_tile`/`load_b_tile_lo`
按此 layout 把每 lane 的 B-pack 取进寄存器。

<a id="23-scale"></a>

### 2.3 a2_scale / w2_scale（make_preshuffle tile）

`make_preshuffle_scale_layout`（与 gemm1 / HIP **完全同形**）：

```python
# shape  = (c_mn1, c_k1, 4, 16),  c_mn1=c_mn/16/2,  c_k1=(c_k/32)/4/2
# stride = (c_k1*64, 64, 16, 1)
```

- a2_scale：`layout_a_scale(c_mn=sorted_size, c_k=inter_dim)`，按 sorted-row 取。
- w2_scale：`layout_b_scale(c_mn=E*model_dim, c_k=inter_dim)`。
- MFMA `rocdl.mfma_scale_f32_16x16x128_f8f6f4` 按 2-bit op_sel 选 e8m0 dword 字节（CDNA4 §7.2.1：block
  size=32，128/32=4 scale/行，64 个 8-bit scale = 1/4 VGPR）。

<a id="24-cshuffle"></a>

### 2.4 累加器与 cshuffle epilogue（vscale + atomic）

单累加器 `acc`（`vec4_f32`×N-rep）。epilogue（`mfma_epilogues.py` cshuffle）：

```python
# 1) acc 经 lds_out 重排成连续 N
write_row_to_lds(acc):
    tw = sorted_weights[sorted_pos]        # 路由权重
    v  = v * tw                            # vscale（doweight_stage2=True）
__syncthreads()
# 2) store_pair（atomic 模式）
store_pair:
    token = sorted_token_ids[sorted_pos] & 0x00FFFFFF
    AtomicRMWOp.fadd(out[token*7168 + n], (bf16)v)    # agent scope
```

`vscale` 把路由权重在 epilogue 内乘掉（省一遍读写），与 HIP gemm2 在 atomic epilogue 里 `acc*weight` 等价。

<a id="25-输出"></a>

### 2.5 输出：atomic 累加到 out[token]（LDS → GM）

```python
token = sorted_token_ids[sorted_pos] & 0x00FFFFFF   # padding 行被跳过
AtomicRMWOp.fadd(out[token*7168 + n_block*tile_n + n], (bf16)(v*tw))   # agent scope
```

`global_atomic_pk_add_f16/bf16`（CDNA4）对 bf16 做原子加：topk 个 (token,slot) 落同一 `out[token]` 行自动汇总
（atomic 模式输出缓冲预清零）。

> 与 HIP gemm2 一致：都用 atomic fadd 到 `out[token]` + epilogue 内乘 weight。差异在 A 的 major 序
> （token-major gather vs sorted-major direct）与 reduce 模式的归约衔接（FlyDSL `torch.sum` vs HIP
> `scatter_reduce`）——见 `mxfp4_06_gemm2.md §7`。

<a id="26-驻留汇总"></a>

### 2.6 张量驻留汇总

| 张量 | GM | LDS | VGPR（每 lane）| 备注 |
|---|---|---|---|---|
| A=inter | `[tok,topk,256]` fp4x2 | `lds_x_ping/pong` | A-pack（i64）| **token-major**，sorted_token_ids gather |
| a2_scale | `[sorted,16]` e8m0 | 视配置 | scale dword | make_preshuffle |
| w2 | `[385,7168,256]` fp4x2 | — | B-pack | 寄存器，make_preshuffle |
| w2_scale | `[385*7168,16]` | — | scale dword | make_preshuffle |
| acc | — | 经 `lds_out` cshuffle | f32x4×N-rep | 单 down-proj 累加器 |
| out | `[tok,7168]` bf16（预清零）| — | epilogue 临时 | atomic_pk_add 汇总 topk |

资源：VGPR 68、SGPR 46、AGPR 0、LDS 13312 B、spill 0。

<a id="27-调用链与计数"></a>

### 2.7 底层调用链与 issue 计数

```
fly_fn → fused_moe(hidden, fly_w...) → fused_moe_ → fused_moe_2stages
└─ _flydsl_stage2_wrapper (fused_moe.py:909)
   └─ flydsl_moe_stage2 (moe_kernels.py:1050)
      └─ compile_mixed_moe_gemm2 (mixed_moe_gemm_2stage.py:2882)   // JIT 构建 MLIR
         └─ launch_mixed_moe_gemm2 (:4863) → moe_gemm2 (:3185)     // gfx950 机器码
            ├─ prefetch_x_to_lds (inter gather → LDS, sorted_token_ids)
            ├─ load_b_tile (w2 make_preshuffle → 寄存器)
            ├─ lds_load_packs_k64 (ds_read A-pack)
            ├─ compute_tile → rocdl.mfma_scale_f32_16x16x128_f8f6f4 (单 down-proj)
            └─ cshuffle epilogue → write_row_to_lds (vscale) → store_pair (AtomicRMWOp fadd)
```

**whole-kernel issue 计数（wave，已核对 21_final_isa.s）**：

| 指令 | 计数 | 推导 |
|---|---:|---|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16 | `E_M·E_N·E_K·2 K-tile = 2·2·2·2`（单流）|
| `buffer_load` | 27 | A gather + w2 B + a2/w2 scale |
| `ds_read` | 18 | A-pack + scale + cshuffle |
| `ds_write` | 17 | gather A 入 LDS + cshuffle |
| `global_atomic`（pk_add）| 8 | epilogue 每 wave 8 次 |
| `v_exp` / `v_cvt_pk_fp4` | 0 / 0 | 无激活、bf16 输出 |
| VGPR / SGPR / AGPR / LDS / spill | 68 / 46 / 0 / 13312 / 0 | `.amdhsa_*` / metadata |

> 锚点（MFMA 16、global_atomic 8）均与 ISA dump 吻合。与 HIP gemm2 的接口/布局差异（尤其 inter token-major
> gather vs sorted-major direct）见 `mxfp4_06_gemm2.md §7`。

