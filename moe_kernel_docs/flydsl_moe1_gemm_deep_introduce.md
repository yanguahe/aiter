# FlyDSL MOE1 GEMM (`mfma_moe1_silu_mul_*`) 深度介绍 — gate/up + SiLU·mul（bf16 输出）

> 本文是 `fly_fn`（tuned FlyDSL 路径：`fused_moe(hidden, fly_w...)` → opus 排序 + FlyDSL 2-stage GEMM）中
> MoE **stage1 GEMM** 的深度剖析，遵循 `.cursor/rules/kernel-deep-introduction-doc.mdc`：事实可追溯到源码 +
> ISA，指令语义取自 CDNA4 ISA 文档。（注：此 kernel 由 Python 构建 MLIR、JIT 编译为 gfx950 机器码。）

## 文档元信息（benchmark / trait / 构建配置）

- **目标 kernel（编译符号）**：`mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32`
  = tile_m=32, tile_n=128, tile_k=256, gate_mode=SEPARATED, out_dtype=**bf16**（本实例**不**融合 fp4 再量化）,
  use_async_copy=true, persist_m=1。
- **源码**（Python 构建 MLIR）：
  - kernel builder：`aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py::compile_mixed_moe_gemm1:92`
  - 内核体（嵌套 fn）：`moe_gemm1:462`；launcher：`launch_mixed_moe_gemm1:2806`；名构造 `:266-269`
  - 复用 helper：`mfma_preshuffle_pipeline.py`（`make_preshuffle_b_layout:168`、
    `make_preshuffle_scale_layout:111`、`swizzle_xor16:29`、`buffer_copy_gmem16_dwordx4:560`、
    `lds_store_16b_xor16:585`、`lds_load_pack_k32:666`）；epilogue：`mfma_epilogues.py`（cshuffle）
  - host：`aiter/aiter/ops/flydsl/moe_kernels.py::flydsl_moe_stage1:694`
  - 选择 wrapper：`aiter/aiter/fused_moe.py::_flydsl_stage1_wrapper:845`
  - 权重 = `fly_w1 = aiter.ops.shuffle.shuffle_weight(w1_qt, layout=(16,16))`（SEPARATED 布局）
- **ISA dump**（gfx950 / CDNA4，FlyDSL JIT 最终汇编）：
  - `aiter/flydsl_dump/mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32/21_final_isa.s`（2159 行）
  - 中间 MLIR 阶段（`00_*` … `11_rocdl_attach_target` … `12_convert_scf_to_cf` … `21_final_isa.s`）同目录，用于 IR→ISA 映射
- **ISA 资源元数据（已核对 21_final_isa.s）**：`.amdhsa_next_free_vgpr 129`（`.vgpr_count 126`）、
  `.amdhsa_next_free_sgpr 96`（`.sgpr_count 46`）、`.agpr_count 0`、`.group_segment_fixed_size 41088`、
  `.vgpr_spill_count 0`。
- **指令语义参考**：`.cursor/rules/amd-instinct-cdna4-instruction-set-architecture.txt`、
  `.cursor/rules/amd-cdna-4-architecture-whitepaper.txt`。
- **benchmark 形状（KIMI）**：NE=385、model_dim=D_HIDDEN=7168、inter_dim=D_INTER=512、TOPK=9，gfx950 MI355X。
  a4w4：A=fp4、W=fp4、e8m0 per-32；本实例 out_dtype=bf16（gate/up SiLU·mul 后直接写 bf16，**不**再量化）。

## 目录

- [1. Kernel 概览与软件流水](#1-kernel-概览与软件流水)
  - [1.1 Buffering：a / a1_scale / w1(gate) / w1(up) / w1_scale / out](#11-buffering)
  - [1.2 融合数学拆成子阶段：gate GEMM + up GEMM → SiLU·mul → bf16](#12-融合数学子阶段)
  - [1.3 Wave / 调度结构](#13-wave--调度结构)
  - [1.4 精化伪代码](#14-精化伪代码)
  - [1.5 每操作主指令与计数](#15-每操作主指令与计数)
  - [1.6 调度原语](#16-调度原语)
- [2. 完整 GM/LDS/VGPR 布局图](#2-完整-gmldsvgpr-布局图)
  - [2.1 A：activation（GM → LDS → VGPR，async DMA）](#21-a)
  - [2.2 B：gate / up 两条 make_preshuffle 流（GM → VGPR）](#22-b)
  - [2.3 A_scale / B_scale（make_preshuffle tile）](#23-scale)
  - [2.4 累加器与 cshuffle epilogue（SiLU·mul → bf16）](#24-cshuffle)
  - [2.5 输出：inter bf16（token-major）](#25-输出)
  - [2.6 张量驻留汇总](#26-驻留汇总)
  - [2.7 底层调用链与 issue 计数](#27-调用链与计数)

---

## 1. Kernel 概览与软件流水

FlyDSL stage1 GEMM（DSL 生成）。对每个「M-block × N-tile × 专家」计算 fp4×fp4 的 **gate** 与 **up** 两路
（`SEPARATED` 模式：两个独立 B 流、两个 f32 累加器），epilogue 做 `y = SiLU(gate)·up`，**本实例直接写 bf16**
（out_dtype=bf16，无 `_fp4q` 融合再量化）。沿 K=model_dim=7168 做 28 个 K-tile（tile_k=256）的缩放 MFMA。

一个 WG = `num_waves = min(4, tile_n/32) = 4` 个 wave = 256 线程，算一个 `tile_m=32 × tile_n=128` 的输出 tile
（gate 与 up 各 128 列）。grid：`by = block_id("x")` = N-tile（沿 inter_dim），`bx_persist = block_id("y")` =
M-block（persist_m=1）；SEPARATED 模式 `gx = ceil(inter_dim/tile_n/2)`（一个 WG 同时算 gate 与 up）。

> 与 HIP gemm1（M7）的关键差异：A 的来源是外部 opus-sort/quant 产出的 fp4（FlyDSL **不**做 inline-quant，
> 本实例也不再量化输出）；权重是 `shuffle_weight((16,16))` 的 SEPARATED 两流，而非 HIP 的 a16w4 gate‖up 交织
> （详见 `mxfp4_05_gemm1.md §7`）。

<a id="11-buffering"></a>

### 1.1 Buffering：a / a1_scale / w1(gate) / w1(up) / w1_scale / out

LDS `.group_segment_fixed_size = 41088 B`（远大于 HIP gemm1 的 16384）：A 用 LDS ping-pong 双缓冲 +
cshuffle 输出缓冲。`.vgpr_count 126`、`.sgpr_count 46`、AGPR 0、spill 0。

| 张量 | Global memory | LDS | VGPR |
|---|---|---|---|
| a（activation）| `[token_num, model_dim/2]` fp4x2（来自 opus quant，sorted 经 sorted_token_ids gather）| `lds_x_ping`/`lds_x_pong` 双缓冲（`tile_m × _eff_lds_stride`）| MFMA A-pack（`lds_load_pack_k32` 产出 i64 pack）|
| a1_scale | `[sorted_size, K/32]` e8m0（sorted-tile）| 视配置入 LDS / 寄存器 | scale dword |
| w1 gate | `fly_w1[E, 2*512, 7168]` fp4x2（`shuffle_weight((16,16))`，前 512 行=gate）| 不过 LDS（寄存器驻留）| gate B-pack（i32x*）|
| w1 up | 同上（后 512 行=up）| 不过 LDS | up B-pack（独立寄存器流）|
| w1_scale | `[E*1024, 7168/32]` e8m0 | 不过 LDS | scale dword |
| out | `inter[token, topk, 512]` bf16（**token-major**）| 经 cshuffle `lds_out` 重排 | **两个** f32 累加器（gate `acc_g`、up `acc_u`），各 `vec4_f32`×N-rep |

> 与 HIP 不同：HIP gemm1 输出 fp4+e8m0 且 sorted-major；FlyDSL 本实例输出 **bf16、token-major**
> `[tok,topk,512]`，gemm2 再按 sorted_token_ids gather（见 `flydsl_06` / `mxfp4_06 §7`）。

<a id="12-融合数学子阶段"></a>

### 1.2 融合数学拆成子阶段：gate GEMM + up GEMM → SiLU·mul → bf16

1. **gate GEMM**（主循环 28 K-tile）：A-pack × gate B-pack → `acc_g`（`rocdl.mfma_scale_f32_16x16x128_f8f6f4`，
   按 e8m0 在 MFMA 内反量化）。
2. **up GEMM**（同循环，第二条 B 流）：A-pack × up B-pack → `acc_u`。
3. **SiLU·mul（epilogue）**：cshuffle 把 `acc_g`/`acc_u` 经 `lds_out` 重排，`_act_vec4`/`_silu_mul_vec4`
   计算 `y = SiLU(gate)·up`（SiLU 用 `v_exp_f32` 的 sigmoid 近似）。
4. **写 bf16**：本实例 out_dtype=bf16 → 直接 `v_cvt_pk_bf16` 转 bf16 写 `inter`（**无** fp4 再量化）。

> 对照：含 `_fp4q` 的实例（小/中 M）会在 epilogue 内对每 32 元素求 amax→e8m0→软件 `_f32_to_e2m1` 压 fp4
> 并写 sorted-tile scale；本 bf16 实例没有这步（`v_cvt_pk_fp4`=0）。

<a id="13-wave--调度结构"></a>

### 1.3 Wave / 调度结构

- WG = 4 wave = 256 线程（`num_waves=min(4,tile_n/32)=4`，`mixed:171`）。4 wave 切分 tile_n=128 →
  每 wave 32 列（gate 与 up 各算）。
- **K 主循环 LDS ping-pong 双缓冲**（`lds_x_ping`/`lds_x_pong`）：一边 DMA 下一 K-tile 的 A，一边 MFMA
  消费当前 tile，B-major 调度（B 驻寄存器、A 在 LDS 复用）。
- **`_async`**：`raw_ptr_buffer_load_lds`（global→LDS 直 DMA，绕过 VGPR），对应 ISA 的 `buffer_load`
  direct-to-LDS。
- 调度栅栏 `rocdl.sched_barrier(0)` 分隔阶段（见 §1.6）；本实例 ISA 含 `s_setprio`（wave 错峰）。

<a id="14-精化伪代码"></a>

### 1.4 精化伪代码

对应 `moe_gemm1`（`mixed_moe_gemm_2stage.py:462+`）的 SEPARATED / bf16-out / async 路径。每行一条语句。

```python
# ── 编译期 layout（mixed:510-532）──
layout_b       = make_preshuffle_b_layout(c_n=E*2*inter_dim, c_k=k_in//pack_K, kpack_bytes=16)
layout_a_scale = make_preshuffle_scale_layout(c_mn=size_expert_ids*sort_block_m, c_k=model_dim)
layout_b_scale = make_preshuffle_scale_layout(c_mn=E*2*inter_dim, c_k=model_dim)
layout_lds     = make_layout((tile_m, _eff_lds_stride))         # A 的 LDS 双缓冲

# ── 运行期索引 ──
tx         = thread_id.x                                        # 0..255
by         = block_id.x                                         # N-tile（沿 inter_dim）
bx_persist = block_id.y                                         # M-block（persist_m=1）
# SEPARATED：gx = ceil(inter_dim/tile_n/2)；一个 WG 同算 gate 与 up
expert     = expert_ids[bx_persist]
# num_valid_ids[0] = 有效 sorted-row 数；越界 block 提前返回

# ── 初始化两个累加器（gate / up）──
acc_g = vec4_f32(0)   # 每 wave 持 E_M*E_N 份
acc_u = vec4_f32(0)

# ── K 主循环（K=7168 → 28 个 tile_k=256；LDS ping-pong 双缓冲）──
prefetch: async A[tile 0] → lds_x_ping        # raw_ptr_buffer_load_lds（绕 VGPR）
for kt in 0..27:
    cur  = (kt % 2) ? lds_x_pong : lds_x_ping
    next = the other
    if kt+1 < 28: async A[tile kt+1] → next   # 与本 tile MFMA 重叠
    a_pack    = lds_load_pack_k32(cur, ...)    # ds_read → i64 A-pack（XOR16 swizzle）
    gate_bpk  = load gate B-pack (寄存器, make_preshuffle_b_layout)
    up_bpk    = load up   B-pack (寄存器)
    a_sc, b_sc= load a1_scale / w1_scale (e8m0)
    sched_barrier(0)
    for (e_m, e_n, e_k) in tile:               # E_M=2, E_N=2/wave, E_K=2
        s_setprio(1)
        acc_g += mfma_scale_f32_16x16x128_f8f6f4(a_pack, gate_bpk, a_sc, b_sc, op_sel...)
        acc_u += mfma_scale_f32_16x16x128_f8f6f4(a_pack, up_bpk,   a_sc, b_sc, op_sel...)
        s_setprio(0)
        sched_barrier(0)

# ── cshuffle epilogue（SiLU·mul → bf16）──
__syncthreads()
write acc_g, acc_u → lds_out (cshuffle 重排成连续 N)
__syncthreads()
for each N-vec of 4:
    gate4 = read lds_out (gate 列)
    up4   = read lds_out (up 列)
    y4    = _silu_mul_vec4(gate4, up4)         # SiLU(gate)*up，SiLU 用 v_exp_f32
    out_bf16 = v_cvt_pk_bf16(y4)               # 本实例直接 bf16，无 fp4 再量化
    store inter[token, topk, n] = out_bf16     # token-major
```

<a id="15-每操作主指令与计数"></a>

### 1.5 每操作主指令与计数

「**一个 wave 处理整个 kernel（28 K-tile + epilogue）**」，已用 `21_final_isa.s` 核对。
tiling：tile_m=32 → `E_M=32/16=2`；tile_n=128，4 wave 各 32 列 → `E_N=32/16=2`；tile_k=256 →
`E_K=256/128=2`；K=7168 → 28 K-tile；gate+up 两路。

| # | 操作 | 主指令 | 计数 | 推导 |
|---|---|---|---|---|
| 1 | gate+up MFMA | `v_mfma_scale_f32_16x16x128_f8f6f4` | **448** | `E_M·E_N·E_K · 2(gate+up) · 28 = 2·2·2·2·28 = 448`（与 .s `v_mfma_scale`=448 一致）|
| 2 | A async 载入 + B 载入 + scale 载入 | `buffer_load`（含 direct-to-LDS）| 341 | A async DMA(每 tile) + gate/up B-pack + a1/w1 scale，合计 341（.s 实测）|
| 3 | A 读 LDS | `ds_read`（`lds_load_pack_k32`）| 118 | A-pack 读 + scale 读 + cshuffle 读，合计 118（.s 实测）|
| 4 | A/输出 写 LDS | `ds_write` | 17 | async A 入 LDS / cshuffle 写 lds_out，合计 17 |
| 5 | SiLU | `v_exp_f32`（`_silu_mul_vec4` 的 sigmoid）| **16** | epilogue 每 N-vec 一次 exp2；与 .s `v_exp`=16 一致 |
| 6 | 输出转 bf16 | `v_cvt_pk_bf16_f32` | — | 本实例 bf16 输出（无 `v_cvt_pk_fp4`，.s 实测=0）|

> 主锚点：MFMA **448** 完整可推导（`E_M·E_N·E_K·2·28`），是「gate+up SEPARATED 两路」的直接体现
> （HIP gemm1 BM=16 因 tile_m=16 且 N_OUT 一次算只 224）。`v_exp`=16 印证「bf16 实例只有 SiLU，无 fp4 量化」。
> `v_cvt_pk_fp4`=0 印证本实例不融合中间量化。

<a id="16-调度原语"></a>

### 1.6 调度原语

| 原语 | 是否真实 ISA | 语义 |
|---|---|---|
| `rocdl.sched_barrier(0)`（FlyDSL）| 否 | 降为 AMDGPU `SCHED_BARRIER`(mask=0) 编译期硬栅栏，阻止任何类别跨越重排，隔开 K-loop 各阶段（DMA / ds_read / MFMA）。与 HIP `__builtin_amdgcn_sched_barrier(0)` 等价，只是 FlyDSL 用 MLIR rocdl intrinsic 发出。|
| `s_setprio(1/0)`（`rocdl.asm_s_setprio`）| 是 | `S_SETPRIO SIMM16`：用户 wave 优先级 0–3（0 最低）。围住 MFMA 簇做 wave 错峰，让另一 wave 在本 wave 算 MFMA 时占用内存端口。|

FlyDSL 仅允许 `sched_barrier` / `s_setprio` 这两类「内联调度原语」，其余排序交由 LLVM 后端。

## 2. 完整 GM/LDS/VGPR 布局图

记号：`tx=thread_id.x`（0..255）、`wave=tx/64`、`lane=tx%64`。fp4x2 = 1 字节 2 个 fp4。

<a id="21-a"></a>

### 2.1 A：activation（GM → LDS → VGPR，async DMA）

A = `[token_num, model_dim/2]` fp4x2（opus quant 产出），按 `sorted_token_ids` 取 sorted 行。

**GM → LDS（async 直 DMA）**：`raw_ptr_buffer_load_lds`（`_async` 路径）把每 lane 16 字节直送 LDS，
绕过 VGPR。LDS 用 `layout_lds = make_layout((tile_m, _eff_lds_stride))`（行 = sorted-row、列 = K 字节）；
写入用 `lds_store_16b_xor16`（`mfma_preshuffle_pipeline.py:585`）的 XOR16 swizzle：

```python
# swizzle_xor16(row, col, k_blocks16):  col ^ ((row & (k_blocks16-1)) * 16)
# k_blocks16 = tile_k_bytes / 16
col_swz = col_local ^ ((row_local & (k_blocks16 - 1)) * 16)
lds[row_local * _eff_lds_stride + col_swz] = vec16   # 16B = 32 fp4
```

**LDS → VGPR（MFMA A-pack）**：`lds_load_pack_k32`（`:666`）按 `swizzle_xor16(curr_row, col_base, k_blocks16)`
取一个 i64 pack（一个 K32 micro-step 的 A 操作数），ping/pong 双缓冲交替。

**可视化（lane → A 的 16B，固定 K-tile）**：4 wave 协作把 `tile_m=32 × tile_k=256` 的 A 经 async DMA 填入
`lds_x_ping/pong`，XOR16 swizzle 消 bank conflict；MFMA 时每 lane 从 LDS 取自己的 K32 pack。

<a id="22-b"></a>

### 2.2 B：gate / up 两条 make_preshuffle 流（GM → VGPR）

B = `fly_w1[E, 2*512, 7168]` fp4x2（`shuffle_weight((16,16))`）。SEPARATED 模式把它当**两条独立流**：
expert e 的 **gate** 在行 `[e*2*inter_dim, e*2*inter_dim+inter_dim)`、**up** 在 `[+inter_dim, +2*inter_dim)`。

**B layout**（`make_preshuffle_b_layout`，N-major `(0,1,3,4,2,5)`，`mfma_preshuffle_pipeline.py:210-230`）：

```python
# shape  = (n0, c_k0, klane=4, 16, kpack_elems=16)
#        = (c_n/16, c_k_bytes/64, 4, 16, 16)
# stride = (stride_n0, stride_k0, stride_klane, stride_nlane, 1)
stride_nlane = kpack_elems              = 16
stride_klane = 16 * stride_nlane        = 256
stride_k0    = 4  * stride_klane        = 1024
stride_n0    = c_k0 * stride_k0
# 寻址：b_off = n0_idx*stride_n0 + k0_idx*1024 + klane*256 + nlane*16 + kpack
```

内层 `[klane=4, nlane=16, kpack=16]` 与 HIP a16w4 的 `[KLane=4, NLane=16, KPack=16]` 形状相同，但
**外层 gate/up 处理不同**：HIP 把 gate‖up 交织进 N_OUT=1024 一个流；FlyDSL SEPARATED 用两个独立 B-pack 寄存器
（gate / up）、两个累加器，故 MFMA 数 ×2（见 §1.5 的 448）。

**可视化（一个 wave 的 N 切分）**：tile_n=128，4 wave 各取 `n0` 维的 32 列（16 列 × 2 个 n0），gate 与 up
各走一遍 → 每 wave 每 K-tile = `E_M(2)·E_N(2)·E_K(2)·2 = 16` MFMA。

<a id="23-scale"></a>

### 2.3 A_scale / B_scale（make_preshuffle tile）

`make_preshuffle_scale_layout`（`mfma_preshuffle_pipeline.py:126-150`）—— 与 HIP gemm1 的 make_preshuffle
scale **完全同形**：

```python
# shape  = (c_mn1, c_k1, 4, 16),  c_mn1 = c_mn/16/2,  c_k1 = (c_k/32)/4/2
# stride = (stride_n0, stride_k0, stride_klane, 1)
stride_klane = 16
stride_k0    = 4 * 16 = 64
stride_n0    = c_k1 * 64
```

- A_scale：`layout_a_scale(c_mn=size_expert_ids*sort_block_m, c_k=model_dim)`，按 sorted-row 取（与 gemm2 一致）。
- B_scale：`layout_b_scale(c_mn=E*2*inter_dim, c_k=model_dim)`。
- MFMA 消费：`rocdl.mfma_scale_f32_16x16x128_f8f6f4` 的 op_sel/op_sel_hi 按 2-bit OPSEL 码选 e8m0 dword 字节
  （CDNA4 ISA §7.2.1：block size=32，128/32=4 scale/行，64 个 8-bit scale = 1/4 VGPR）。

> 这正是 `mxfp4_06 §7` 指出的「scale 布局两边一致、可复用」之处；差异只在权重布局与 inter 的 major 序。

<a id="24-cshuffle"></a>

### 2.4 累加器与 cshuffle epilogue（SiLU·mul → bf16）

两个累加器 `acc_g`（gate）、`acc_u`（up），各 `vec4_f32` × N-rep。epilogue（`mfma_epilogues.py` cshuffle）：

```python
# 1) acc_g / acc_u 经 lds_out 重排成连续 N（gate 与 up 对齐到同一逻辑列）
write acc_g → lds_out[gate 区];  write acc_u → lds_out[up 区]
__syncthreads()
# 2) 读回 → SiLU·mul → bf16
for n-vec of 4:
    gate4 = lds_out[gate 列]
    up4   = lds_out[up 列]
    y4    = _silu_mul_vec4(gate4, up4)        # SiLU(g)=g*sigmoid(g) 用 v_exp_f32；再 *up
    out_bf16 = v_cvt_pk_bf16_f32(y4)
```

`_silu_mul_vec4`（`mixed:1942` 附近）的 SiLU 用 `v_exp_f32` 算 sigmoid（这是 .s 里 16 个 `v_exp` 的来源）。

<a id="25-输出"></a>

### 2.5 输出：inter bf16（token-major）

本实例 out_dtype=bf16，输出 `inter[token, topk, inter_dim=512]` bf16，**token-major**：

```python
# 行 = token*topk + slot（由 sorted_token_ids 还原），列 = inter_dim
store inter[token, slot, n] = out_bf16
```

> 与 HIP gemm1 的根本差异：HIP 输出 **fp4 + e8m0、sorted-major** `[max_sorted, 256]`；FlyDSL 本实例输出
> **bf16、token-major** `[tok, topk, 512]`。这决定了 gemm2 的取数方式不同（FlyDSL gemm2 需按 sorted_token_ids
> gather，HIP gemm2 sorted-direct 读）——见 `mxfp4_06_gemm2.md §7.1`。

<a id="26-驻留汇总"></a>

### 2.6 张量驻留汇总

| 张量 | GM | LDS | VGPR（每 lane）| 备注 |
|---|---|---|---|---|
| a | `[token,3584]` fp4x2 | `lds_x_ping/pong`（tile_m×stride）| A-pack（i64）| async DMA，XOR16 swizzle |
| a1_scale | `[sorted,224]` e8m0 | 视配置 | scale dword | make_preshuffle，sorted-row |
| w1 gate | `[385,1024,7168]` 前半 | — | gate B-pack | 寄存器，N-major preshuffle |
| w1 up | 同上 后半 | — | up B-pack | 第二条流 |
| w1_scale | `[385*1024,224]` | — | scale dword | make_preshuffle |
| acc_g/acc_u | — | 经 `lds_out` cshuffle | 2× f32x4×N-rep | 两个累加器 |
| out=inter | `[tok,topk,512]` bf16 | — | epilogue 临时 | **token-major**（非 sorted）|

资源：VGPR 126、SGPR 46、AGPR 0、LDS 41088 B、spill 0。

<a id="27-调用链与计数"></a>

### 2.7 底层调用链与 issue 计数

```
fly_fn → fused_moe(hidden, fly_w...) → fused_moe_ → fused_moe_2stages
└─ _flydsl_stage1_wrapper (fused_moe.py:845)
   └─ flydsl_moe_stage1 (moe_kernels.py:694)
      └─ compile_mixed_moe_gemm1 (mixed_moe_gemm_2stage.py:92)  // JIT 构建 MLIR
         └─ launch_mixed_moe_gemm1 (:2806) → moe_gemm1 (:462)   // gfx950 机器码
            ├─ make_preshuffle_b_layout / make_preshuffle_scale_layout   // 布局
            ├─ raw_ptr_buffer_load_lds (async A→LDS) + lds_store_16b_xor16
            ├─ lds_load_pack_k32 (ds_read A-pack)
            ├─ rocdl.mfma_scale_f32_16x16x128_f8f6f4 (gate + up 两流)
            └─ cshuffle epilogue → _silu_mul_vec4 (v_exp) → v_cvt_pk_bf16
```

**whole-kernel issue 计数（wave，已核对 21_final_isa.s）**：

| 指令 | 计数 | 推导 |
|---|---:|---|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 448 | `E_M·E_N·E_K·2(gate+up)·28 = 2·2·2·2·28` |
| `buffer_load` | 341 | A async + gate/up B + a1/w1 scale |
| `ds_read` | 118 | A-pack + scale + cshuffle |
| `ds_write` | 17 | async A 入 LDS + cshuffle |
| `v_exp` | 16 | SiLU sigmoid |
| `v_cvt_pk_fp4` | 0 | bf16 输出，不再量化 |
| VGPR / SGPR / AGPR / LDS / spill | 126 / 46 / 0 / 41088 / 0 | `.amdhsa_*` / metadata |

> 主锚点 MFMA=448 完整推导，体现 SEPARATED gate+up 两路；其余为 .s 实测总数。与 HIP gemm1 的接口/布局/精度
> 差异见 `mxfp4_05_gemm1.md §7`。

