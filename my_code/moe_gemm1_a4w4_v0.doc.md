# gfx1250 `moe_gemm1_a4w4_v0` ISA 完整介绍与深度解析

<!-- markdownlint-disable MD013 MD033 MD060 -->

本文解析当前文件：

```text
my_code/moe_gemm1_a4w4_v0.s
SHA256 = 3aaa52af56f6bffaaea2197b2267d048da090c91c7827cd18ef8c0edba7f0d70
```

目标 kernel symbol 为：

```text
a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4
```

尽管 symbol 保留历史前缀 `a8w4_tdm`，紧随其后的 `fp4` 才是 activation
格式：本 kernel 是 **MXFP4 activation × MXFP4 weight（A4W4）**，不是 A8W4。

本文只做静态分析，没有修改或执行 ISA，没有运行 GPU，也没有执行 Git 或远程
操作。所有 `Lxxx` 均指当前 `my_code/moe_gemm1_a4w4_v0.s`，不沿用旧 dump 或
旧分析文档的行号。

## 目录

- [0. 摘要与一页式契约](#sec-0)
- [1. 功能语义与 grouped MoE 行布局](#sec-1)
- [2. Kernel object、资源与 launch geometry](#sec-2)
- [3. 184 B kernarg ABI](#sec-3)
- [4. Entry、16-M-tile swizzle 与 expert lookup](#sec-4)
- [5. A/B/Scale physical layout 与 descriptor](#sec-5)
- [6. LDS、`wpt4` TDM pipeline 与同步](#sec-6)
- [7. K=7168 compute hot loop 与 WMMA 映射](#sec-7)
- [8. Epilogue：clamp、SiLU、BF16 与 masked store](#sec-8)
- [9. Control flow、labels 与寄存器生命周期](#sec-9)
- [10. Correctness invariants、边界与性能瓶颈](#sec-10)
- [11. 静态指令计数、FLOPs 与 logical bytes](#sec-11)
- [12. 与 production runner / ISA runner 的对应关系](#sec-12)
- [13. 静态验证结果、资料索引与仍未知项](#sec-13)

<a id="sec-0"></a>

## 0. 摘要与一页式契约

### 0.1 一句话语义

对每个 routed row `r` 和 intermediate column `j`，kernel 计算同一 expert 的
gate/up 两个 MXFP4 dot product，在 F32 accumulator 中完成 block-scale
累加，随后执行：

```text
g = min(gate, +limit)
u = clamp(up, -limit, +limit)
sigmoid(g) = 1 / (1 + exp2(-log2(e) * g))
y[r,j] = BF16(g * sigmoid(g) * u)
```

当前 production 参数 `limit=7.0`。没有 bias；输出是 SiLU-gated BF16，
shape 为 `[1, contiguous_m, inter_dim] = [1, 9216, 3072]`，但只有 3072 个
routed rows 是语义有效行。

### 0.2 固定 specialization

| 属性 | 当前 v0 |
|---|---|
| Target | `amdgcn-amd-amdhsa--gfx1250` |
| Code object | version 6 |
| Symbol | `a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4` |
| Operand | MXFP4 E2M1 × MXFP4 E2M1，E8M0 block-32 scale |
| Compile-time `K` | 7168 |
| Experts | 96 |
| WG tile | raw GEMM `M64 × N256 × K256` |
| Wave tile | raw GEMM `M64 × N64` |
| Workgroup | 128 threads = 4 wave32 |
| LDS buffers | `b2`，两个 `0xae00`-byte slot |
| TDM ownership | `wpt4`，每个 operand job 由四个 wave 分段共同搬运 |
| Stage-1 activation | `act1` = SiLU gate × up |
| Bias / quantized output | `has_bias=0` / `stage1_quant_out=0` |
| Output | BF16，激活后每 WG 为 `M64 × N128` |
| Cluster | `cluster_n=1`，普通 launch，无 multicast、无 cluster barrier |

### 0.3 固定测试的关键数字

```text
tokens                  = 512
topk                    = 6
routes                   = 512 * 6 = 3072
experts                  = 96
valid rows / expert      = 3072 / 96 = 32
aligned rows / expert    = align_up(32, 64) = 64
actual aligned span      = 96 * 64 = 6144
static contiguous_m      = 9216
raw GEMM N               = 2 * 3072 = 6144
N tiles                  = 6144 / 256 = 24
static M tiles           = 9216 / 64 = 144
grid                     = (144 * 24, 1, 1) = (3456,1,1)
block                    = (128,1,1)
active WGs               = 96 * 24 = 2304
sentinel-tail WGs        = 48 * 24 = 1152
K tiles                  = 7168 / 256 = 28
```

### 0.4 证据等级

本文明确区分：

- **源码事实**：来自当前 FlyDSL、runner、route/quant/shuffle 源码。
- **ISA 事实**：当前 `.s` 中直接可见的指令、label、metadata。
- **硬件文档事实**：来自本地 CDNA5 ISA 与 MI400 Shader Programming Guide。
- **静态推导**：由上述事实和固定 shape 算术推出。
- **未知/待实测**：缓存命中、HBM transaction、wait latency、occupancy 等不能由
  静态文本确定的量。

<a id="sec-1"></a>

## 1. 功能语义与 grouped MoE 行布局

### 1.1 从 token/topk 到 routed rows

`my_code/run_gemm_a4w4.sh` 固定：

```text
experts=96, tokens=512, topk=6
model_dim=7168, inter_dim=3072
data_format=a4w4, act=silu, no-bias
AITER_MOE_EXPERT_BALANCE=true
```

balanced score 在
`op_tests/test_flydsl_grouped_gemm_gfx1250.py:L980-L1023` 中按 6 个 expert
为一组循环。因为 `96/6=16` 且 `512=32*16`：

```text
count[e] = 32, e=0..95
```

route kernel 的 atomic 返回值是每条 route 的 0-based `slot`：

```text
slot(route)       = atomic_fetch_add(counter[e], 1)
row_masked(route) = e * max_m + slot(route)
```

每个 expert 的 slot **集合**是 `0..31`；具体哪条
`(token, topk_position)` 获得哪个 slot 取决于 atomic 执行顺序，不能把
balanced score 的生成顺序误当成最终 slot 顺序。

### 1.2 `valid`、`aligned`、`contiguous` 是三个不同概念

当前 t64 specialization 的 route 参数是：

```text
valid_routes = 3072
max_m        = align_up(valid_routes, 64) = 3072

contiguous_m
  = align_up(valid_routes + experts*64 - topk, 64)
  = align_up(3072 + 96*64 - 6, 64)
  = align_up(9210, 64)
  = 9216
```

`contiguous_psum_remap` 对每个 expert 的 count 做 64-row alignment：

```text
aligned_count[e] = align_up(count[e], 64) = 64
starts[e]        = 64e
psum[e]          = starts[e] + count[e] = 64e + 32
row_contiguous   = starts[e] + slot
```

其中 `psum[e]` 是 valid exclusive end，不是 aligned end。

| 区域 | 行范围 | 行数 | 含义 |
|---|---:|---:|---|
| valid routed rows | 每个 expert 的 `64e..64e+31` | 3072 | 真正输入/输出 |
| per-expert padding | 每个 expert 的 `64e+32..64e+63` | 3072 | 同一 active M64 tile 内 |
| actual aligned span | `0..6143` | 6144 | 96 个 M64 expert tile |
| static sentinel tail | `6144..9215` | 3072 | 48 个完整 M64 tail tile |
| static capacity | `0..9215` | 9216 | graph-safe `contiguous_m` |

几个实例：

| expert | `starts` | `psum` | valid contiguous rows | padding |
|---:|---:|---:|---|---|
| 0 | 0 | 32 | `0..31` | `32..63` |
| 1 | 64 | 96 | `64..95` | `96..127` |
| 95 | 6080 | 6112 | `6080..6111` | `6112..6143` |

### 1.3 Gate/up 的 GUGU 语义

数学输入通常写成 GGUU：

```text
[g0,g1,...,g3071, u0,u1,...,u3071]
```

本 kernel 要求 W1 与 W1 scale 先改成 GUGU：

```text
[g0,u0,g1,u1,...,g3071,u3071]
```

raw GEMM 的 `N=6144`，相邻 raw columns `(2j,2j+1)` 是一个
`(gate_j,up_j)` pair。epilogue 每两个 raw F32 accumulators 生成一个 BF16
输出，因此：

```text
raw N / WG       = 256
activated N / WG = 128
global output N  = 6144 / 2 = 3072
```

传入未 interleave 的 GGUU weight 不会触发错误检查，而会得到数值错误。

### 1.4 完整软件链中的位置

固定 A4W4 路径是：

```text
BF16 hidden [512,7168]
  -> route / atomic slot
  -> tile-aligned psum + masked-to-contiguous remap
  -> per-1x32 MXFP4 quant + scatter
  -> A-scale preshuffle
  -> 本 GEMM1 A4W4 + SiLU -> BF16 [1,9216,3072]
  -> 独立 MXFP4 quant + scale preshuffle
  -> GEMM2
  -> gather-reduce -> BF16 [512,7168]
```

所以本 kernel 自身的终点是 BF16 intermediate，不包括后续 GEMM2 或
gather-reduce。

### 1.5 固定 shape 与 runtime 字段

当前 object 不是任意 shape 的通用 GEMM：

| 维度/属性 | 固定还是 runtime | 当前契约 |
|---|---|---|
| `K` | compile-time | 7168，恰有 28 个 K256 |
| experts | compile-time | 96-entry lookup |
| tile / wave grid | compile-time | `64×256×256` / `1×4` |
| FP4、activation、bias | compile-time | A4W4、`act1`、no-bias |
| `i32_m` | runtime ABI | 当前必须为 static capacity 9216 |
| `i32_n` | runtime ABI | 当前必须为 raw GUGU width 6144 |
| `f32_swiglu_limit` | runtime ABI | production 为 7.0 |
| pointers / Tensor companions | runtime ABI | pointer live；多数 companion 在本 specialization 中 dead |

源码算式对 `i32_m/i32_n` 使用 ceil-div，但这不等于当前 artifact 已验证任意
M/N tail。本文和两个 ISA runners 都把它限定在上述固定 workload。

<a id="sec-2"></a>

## 2. Kernel object、资源与 launch geometry

### 2.1 当前 code-object metadata

ISA L1-L7 给出 target 与 symbol；L1508-L1652 给出 kernel descriptor 和
AMDGPU metadata。

| 资源/属性 | 当前值 | 证据 |
|---|---:|---|
| target | `gfx1250` | L1、L1652 |
| code object version | 6 | L2 |
| fixed LDS | 89,088 B = `0x15c00` = 87 KiB | L1509、L1634 |
| private/scratch | 0 | L1510、L1640 |
| kernarg | 184 B，align 8 | L1511、L1635-L1637 |
| required workgroup | `(128,1,1)` | L1641-L1644 |
| wavefront | 32 | L1520、L1651 |
| SGPR | 88，spill 0 | L1529、L1646-L1647 |
| metadata VGPR | 239，spill 0 | L1650-L1651 |
| AGPR | 0 | L1554 |
| named barrier | 0 | L1530、L1556 |
| dynamic stack | false | L1521、L1649 |

VGPR 必须保留两个口径：

```text
.amdhsa_next_free_vgpr 257       # L1528
.vgpr_count: 239                 # L1650
.set ...num_vgpr, 239            # L1553
highest explicit ISA register v238
```

当前 ISA 没有 `s_set_vgpr_msb`，也没有显式访问 `v239+`。因此本文不把
`next_free_vgpr=257` 简化成“实际使用 257 个 VGPR”；最终 metadata、显式最高
编号和 descriptor directive 存在口径差异，原因仅凭 `.s` 不能确定。

### 2.2 Entry scheduling mode

L8：

```text
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
```

把 `SCHED_MODE.DEP_MODE` 设为 2。MI400 Guide §3.4.9 明确说明该模式禁用
`VA_VDST` 与 `VM_VSRC` 的部分硬件依赖检查，因此 ISA 必须依赖显式
`s_wait_alu`、`s_delay_alu` 和 compiler scheduling 保证正确性。

L9：

```text
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
```

把 `MODE.REPLAY_MODE` 设为 1，即 multi-VMEM-group mode（MI400 Guide
§3.4.3）。这两条指令是当前 binary 的事实，不应从旧 kernel 推断。

### 2.3 Wave/WG 输出分工

一个 WG 有四个 wave32：

```text
wave = thread_idx.x >> 5 = 0..3
lane = thread_idx.x & 31
```

每个 wave 计算 raw `M64×N64`，四个 wave 只沿 raw N 拼接：

| wave | raw M | raw N within WG | activated output N within WG |
|---:|---|---|---|
| 0 | `0..63` | `0..63` | `0..31` |
| 1 | `0..63` | `64..127` | `32..63` |
| 2 | `0..63` | `128..191` | `64..95` |
| 3 | `0..63` | `192..255` | `96..127` |

```text
raw accumulator WG tile = M64 × N256
BF16 output WG tile      = M64 × N128
```

### 2.4 Launch geometry

对当前固定输入：

```text
grid    = (3456,1,1)
block   = (128,1,1)
cluster = (1,1,1)
```

`cluster_n=1` 走普通 `kernel.launch` 分支
（`mxfp4_preshuffle_gfx1250_tdm.py:L1273-L1287`），所以：

- 没有 cluster launch attribute；
- TDM descriptor 的 `workgroup_mask=0`；
- 没有 `s_barrier_signal -3` / `s_barrier_wait -3`；
- 每个 WG 独立执行一个 tile，不是 persistent kernel。

这与参考总文档中的 cluster/persistent/batch-Z 设计不同，不能机械套用。

<a id="sec-3"></a>

## 3. 184 B kernarg ABI

### 3.1 Lowering 结构

FlyDSL kernel 签名见
`mxfp4_preshuffle_gfx1250_tdm.py:L204-L218`。四个 `fx.Tensor` 被 lower 为：

```text
pointer + packed layout companion
```

裸 `fx.Pointer` 只占一个 64-bit pointer。最终得到 17 个 metadata rows，
展开成下表的逐字段 184 B ABI。

表中“production 值”指 t64 固定测试契约。runtime pointer 不伪造具体地址。
stride 是 descriptor 的 element stride，不是 byte stride。

| Offset | Size | 字段 / type | production 固定值 | 意义 |
|---:|---:|---|---|---|
| `0x00` | 8 | `arg_c.ptr / global_buffer` | runtime ptr | BF16 output base |
| `0x08` | 4 | `c_size0 / i32` | 1 | output dim0 |
| `0x0c` | 4 | `c_size1 / i32` | 9216 | output dim1 |
| `0x10` | 4 | `c_size2 / i32` | 3072 | output dim2 |
| `0x14` | 8 | `c_stride0 / i64` | 28,311,552 = `0x1b00000` | BF16 elements |
| `0x1c` | 8 | `c_stride1 / i64` | 3072 = `0xc00` | BF16 elements |
| `0x24` | 4 | padding | 无语义 | 对齐下一个 pointer |
| `0x28` | 8 | `arg_a / global_buffer` | runtime ptr | packed MXFP4 A |
| `0x30` | 8 | `arg_b / global_buffer` | runtime ptr | preshuffled GUGU W1 |
| `0x38` | 8 | `arg_scale_a.ptr / global_buffer` | runtime ptr | preshuffled E8M0 A scale |
| `0x40` | 4 | `sa_size0 / i32` | 1 | int32 Tensor dim0 |
| `0x44` | 4 | `sa_size1 / i32` | 2304 = `0x900` | int32 Tensor dim1 |
| `0x48` | 4 | `sa_size2 / i32` | 224 = `0xe0` | int32 Tensor dim2 |
| `0x4c` | 8 | `sa_stride0 / i64` | 516,096 = `0x7e000` | int32 elements |
| `0x54` | 8 | `sa_stride1 / i64` | 224 = `0xe0` | int32 elements |
| `0x5c` | 4 | padding | 无语义 | 对齐下一个 pointer |
| `0x60` | 8 | `arg_scale_b.ptr / global_buffer` | runtime ptr | flattened int32 E8M0 B scale |
| `0x68` | 4 | `sb_size0 / i32` | 33,030,144 = `0x1f80000` | flattened int32 length |
| `0x6c` | 4 | padding | 无语义 | 对齐下一个 pointer |
| `0x70` | 8 | `arg_m_tile_map / global_buffer` | runtime ptr | `int32 psum[96]` |
| `0x78` | 8 | `arg_bias / global_buffer` | aliases A in production | dead：`has_bias=0` |
| `0x80` | 8 | `arg_quant_scale.ptr / global_buffer` | aliases C in production | dead：`stage1_quant_out=0` |
| `0x88` | 4 | `qs_size0 / i32` | 1 | dummy Tensor dim0 |
| `0x8c` | 4 | `qs_size1 / i32` | 9216 | dummy Tensor dim1 |
| `0x90` | 4 | `qs_size2 / i32` | 3072 | dummy Tensor dim2 |
| `0x94` | 8 | `qs_stride0 / i64` | 28,311,552 | dummy BF16 elements |
| `0x9c` | 8 | `qs_stride1 / i64` | 3072 | dummy BF16 elements |
| `0xa4` | 4 | `i32_m` | 9216 = `0x2400` | static `contiguous_m` |
| `0xa8` | 4 | `i32_n` | 6144 = `0x1800` | raw gate+up N |
| `0xac` | 4 | `f32_swiglu_limit` | 7.0, bits `0x40e00000` | SiLU 路径也实际使用 |
| `0xb0` | 4 | `f32_situ_beta` | 4.0, bits `0x40800000` | `act1` dead field |
| `0xb4` | 4 | `f32_situ_linear_beta` | 25.0, bits `0x41c80000` | `act1` dead field |

有效 metadata payload 为 172 B，三个 4 B padding 共 12 B：

```text
172 + 12 = 184 B = 0xb8
```

### 3.2 Tensor storage 与 ABI view

| Tensor | ABI-visible / physical shape | Storage bytes |
|---|---|---:|
| C | BF16 `[1,9216,3072]` row-major | 56,623,104 = `0x03600000` |
| A payload | uint8 `[1,9216,3584]` | 33,030,144 = `0x01f80000` |
| B payload | uint8 `[96,6144,3584]` preshuffled | 2,113,929,216 = `0x7e000000` |
| A scale | uint8 `[1,2304,896]`，ABI view int32 `[1,2304,224]` | 2,064,384 = `0x001f8000` |
| B scale | uint8 `[96,192,7168]`，ABI flatten int32 | 132,120,576 = `0x07e00000` |
| `m_tile_map` | int32 `[96]` | 384 = `0x180` |

### 3.3 Preload 与真正被读取的字段

Descriptor 指定：

```text
user_sgpr_count          = 4
kernarg_segment_ptr      = 1
kernarg_preload_length   = 2 dwords
kernarg_preload_offset   = 0
```

因此入口：

```text
s[0:1] = kernarg segment pointer
s[2:3] = preloaded arg_c pointer
```

当前 ISA 的 kernarg loads：

| ISA | ABI bytes | 内容 |
|---|---|---|
| L10 `s_load_b96 s[48:50], ..., 0xa4` | `0xa4..0xaf` | `m,n,limit` |
| L117 `s_load_b64 ..., 0x70` | `0x70..0x77` | `m_tile_map` |
| L327 `s_load_b128 ..., 0x28` | `0x28..0x37` | A、B pointers |
| L328 `s_load_b64 ..., 0x38` | `0x38..0x3f` | A-scale pointer |
| L423 `s_load_b64 ..., 0x60` | `0x60..0x67` | B-scale pointer |

Tensor shape/stride companions、bias、quant-scale 和两个 SiTU beta 均不被当前
机器码读取；它们仍是 ABI 的一部分，不能从 launch payload 删除。

<a id="sec-4"></a>

## 4. Entry、16-M-tile swizzle 与 expert lookup

### 4.1 一维 grid 的 16-M-tile swizzle

源码 `mxfp4_preshuffle_gfx1250_tdm.py:L236-L257` 定义：

```text
TILES_PER_GROUP = 16
total_m_tiles   = ceil(9216/64)  = 144
total_n_tiles   = ceil(6144/256) = 24
blocks_per_group = 16*24 = 384

group       = bid_x // 384
in_group    = bid_x % 384
m_tile      = 16*group + (in_group % 16)
n_tile      = in_group // 16
blk_m       = 64*m_tile
blk_n       = 256*n_tile
```

本例 144 恰好可被 16 整除，因此 9 个 group 都是完整 group。更紧凑地：

```text
bid_x = 384*g + 16*n + ml
g  = 0..8
n  = 0..23
ml = 0..15

m_tile = 16*g + ml
n_tile = n
```

静态枚举得到 3456 个互不重复的 `(m_tile,n_tile)`，恰好覆盖
`[0,143]×[0,23]`，没有 hole 或 collision。

| `bid_x` | `(m_tile,n_tile)` | 说明 |
|---:|---|---|
| 0 | `(0,0)` | 第一个 expert/N tile |
| 15 | `(15,0)` | 同 N tile 的第 16 个 M tile |
| 16 | `(0,1)` | N 前进，M group 内回到 0 |
| 383 | `(15,23)` | group 0 结束 |
| 384 | `(16,0)` | group 1 开始 |
| 2303 | `(95,23)` | 最后一个 active WG |
| 2304 | `(96,0)` | 第一个 sentinel-tail WG |
| 3455 | `(143,23)` | grid 末尾 |

ISA L10-L149 是 compiler 展开的除法、余数和 swizzle。其复杂 reciprocal/divide
序列只是上述整数映射的 lowering，不代表 persistent 调度器。

### 4.2 96-entry fixed-step upper-bound

源码语义是找第一个：

```text
psum[expert] > blk_m
```

当前 ISA 发出 8 次 scalar load：

```text
L134, L154, L164, L174, L188, L196, L208, L219
```

索引在每次读取前 clamp 到 95。valid expert 的最后一次精确 `psum` load 位于
L238，随后 L322 形成：

```text
mn_oob = psum[expert] - blk_m
```

对 `m_tile=e=0..95`：

```text
psum[e-1] <= 64e < psum[e]
expert = e
mn_oob = (64e+32) - 64e = 32
```

### 4.3 当前 v0 的 tail sentinel 实际是 97

这里有一个容易被旧文档写错的细节。数学 upper-bound 对
`blk_m>=6144` 应返回 96；但当前源码固定执行 8 轮，即使 `lo==hi==96` 也继续
执行。`mid` clamp 到 95 后，`psum[95]<=blk_m` 再次令 `lo=97`。

对 48 个 tail M tiles 的静态枚举结果是：

```text
m_tile 96..143 -> raw expert register value 97
```

它不是 96。正确性仍成立，因为 L226-L227 只要求：

```text
expert > 95 -> branch .LBB0_5
```

tail 路径在 L227 直接跳到 L1504-L1505，不执行 L238，不会访问
`psum[96]`，也不会发 TDM、DS、WMMA 或 output store。

### 4.4 Active 与 tail workgroups

```text
active M tiles = 0..95  -> 96*24 = 2304 WGs
tail M tiles   = 96..143 -> 48*24 = 1152 WGs
```

active WG 内仍有 32 valid + 32 padding rows；tail WG 则是整个 M64 tile 都被
入口跳过。两种 padding 不可混淆。

<a id="sec-5"></a>

## 5. A/B/Scale physical layout 与 descriptor

### 5.1 A payload：row-major packed FP4

A 的数学 layout 是 `[M,K]` FP4，每 byte 打包两个 E2M1：

```text
physical uint8 shape = [1,9216,K/2] = [1,9216,3584]
A_byte(row,kbyte)    = A + row*3584 + kbyte
```

A payload 本身没有 B 那样的 16×16 tile shuffle。每个 K256 只取
`256/2=128` bytes/row。

对 TDM owner wave `w`：

```text
rows       = blk_m + 16w .. blk_m + 16w + 15
K bytes    = kt*128 .. kt*128 + 127
row stride = 3584 B
payload    = 16*128 = 2048 B
```

`mn_oob=32` 使 wave0/1 的 A rows in-bound，wave2/3 的 A rows 为 TDM OOB
zero-fill。

### 5.2 B payload：GUGU + 16-row × 16-byte preshuffle

`moe_shuffle_weight` 先做 GGUU→GUGU，再执行：

```text
[E,N,Kbytes]
 -> [E,N/16,16,Kbytes/32,2,16]
 -> permute [E,N/16,Kbytes/32,2,N_in16,Kbyte_in16]
```

本例：

```text
[96,384,112,2,16,16]
```

对逻辑 packed-byte 坐标 `(e,n,kb)`：

```text
n16    = n // 16
nin    = n % 16
k32    = kb // 32
khalf  = (kb // 16) % 2
kin16  = kb % 16

physical order = [e,n16,k32,khalf,nin,kin16]
```

一个 N16 super-row 的完整 K stride：

```text
Kp16 = (K/2)*16 = 57344 B = 0xe000
```

一个 expert 的 B stride：

```text
(N/16)*Kp16 = 384*57344 = 22,020,096 B = 0x1500000
```

每个 K256、每个 N16 super-row 取 2048 contiguous bytes。wpt4 的 wave `w`
搬运当前 N256 tile 中的 4 个 N16 rows：

```text
n16 = blk_n/16 + 4w .. blk_n/16 + 4w+3
payload = 4*2048 = 8192 B
```

### 5.3 A scale physical layout

每个 E8M0 byte 对应一个 K32 block。A scale 的物理 byte layout 是：

```text
[batch, M/64, (K/32)/4, wmma_row, lane16, byte_in_K128]
= [1,144,56,4,16,4]
```

等价 ABI allocation：

```text
uint8 [1,2304,896]
int32 view [1,2304,224]
```

索引含义：

```text
m64_tile   = row // 64
wmma_row   = (row % 64) // 16
lane16     = row % 16
k128_group = k32 // 4
byte4      = k32 % 4
```

一个 M64 tile 的 A-scale stride：

```text
56*4*16*4 = 14,336 B = 0x3800
```

一个 K256 stage 含 `2*4*16=128` i32 = 512 B。wpt4 把这段 physical inner
layout 平分为四个 contiguous 32-i32 segment：

```text
wave w -> i32 indices 32w..32w+31 -> 128 B
```

### 5.4 B scale physical layout

W1 scale 同样先做 GUGU，再由 `shuffle_scale_n32k4` 变成：

```text
[E,N/32,(K/32)/4,N_in32,byte4]
= [96,192,56,32,4]
```

ABI-visible uint8 shape 是 `[96,192,7168]`，随后 flatten/view 为 int32。

```text
one N32 super-row stride = 56*32*4 = 7168 B = 0x1c00
one expert stride        = 192*7168 = 1,376,256 B = 0x150000
```

每个 N32 row、每个 K256 读取 64 i32 = 256 B。wave `w` 负责两个 N32
super-rows：

```text
n32 = blk_n/32 + 2w, blk_n/32 + 2w+1
payload = 2*256 = 512 B
```

### 5.5 每个 wave 的 global pointer 公式

令 `kt=0..27`、`w=0..3`，字节地址可写成：

```text
A segment:
  A + (blk_m+16w)*3584 + kt*128
  16 rows, row stride 3584, inner 128 B

B segment:
  B + e*22,020,096
    + (blk_n/16+4w)*57,344
    + kt*2,048
  4 rows, row stride 57,344, inner 2,048 B

SA segment:
  SA + (blk_m/64)*14,336 + kt*512 + w*128
  contiguous 128 B

SB segment row j=0,1:
  SB + e*1,376,256
     + (blk_n/32+2w+j)*7,168
     + kt*256
  each row inner 256 B
```

这些是 physical-layout 地址，不是数学 row-major scale 地址。

### 5.6 Semantic TDM descriptor 与当前 ISA sites

| Operand | 每 wave segment | LDS row/payload | slot0 / slot1 / steady site |
|---|---:|---|---|
| A | `16×128 B` | 144-B row pitch，含 16-B pad | L377 / L539 / L665 |
| B | `4×2048 B` | contiguous 2048-B rows | L511 / L544 / L674 |
| SA | 32 i32 | 128 B contiguous | L517 / L548 / L685 |
| SB | `2×64 i32` | 512 B payload | L538 / L553 / L695 |

每个 operand 的 group-1 descriptor 在 prologue 和 steady refill 中复用；
group-0 global/LDS base 随 slot、`kt` 和 wave segment 改写。

按 CDNA5 descriptor 的 `data_size` unit 重写，四类 load 的语义字段是：

| Operand | `data_size` | `tile_dim0` contiguous inner | `tile_dim1` outer | global outer stride | OOB / pad |
|---|---:|---:|---:|---:|---|
| A | 1 B | 128 | 16 | 3584 B | outer extent `mn_oob-16w`；每 128 B 加 16 B LDS pad |
| B | 1 B | 2048 | 4 | 57,344 B | 无 N/K OOB |
| SA | 4 B | 32 | 1 | 3584 i32 | 无 row OOB；physical inner quarter |
| SB | 4 B | 64 | 2 | 1792 i32 | 无 N/K OOB |

这里列的是由当前 FlyDSL shape/stride 和 ISA LDS offsets 共同建立的 semantic
descriptor；packed SGPR bitfields 由 compiler 生成，不把旧 kernel 的 hex
descriptor words 移植到本文件。当前 descriptor 原始 dword、逐 wave
`tensor_dim1` 和代表性地址展开另见
[`moe_gemm1_a4w4_v0_tdm_ld_st.doc.md`](moe_gemm1_a4w4_v0_tdm_ld_st.doc.md)；
本篇保留完整 kernel 主线，不重复堆砌全部 D# bitfield。

因为 `cluster_n=1`：

```text
workgroup_mask = 0
```

CDNA5 ISA §10.11.3 规定 nonzero mask 才把 load 改成
`CLUSTER_LOAD_ASYNC`；当前 kernel 因而是普通 TDM global→LDS load。

<a id="sec-6"></a>

## 6. LDS、`wpt4` TDM pipeline 与同步

### 6.1 两槽 LDS layout

源码 `mxfp4_preshuffle_gfx1250_tdm.py:L148-L178` 对当前 shape 化简：

| Region | 计算 | Bytes | slot0 |
|---|---:|---:|---|
| A | `64*(128+16)` | 9216 = `0x2400` | `[0x0000,0x2400)` |
| B | `(256/16)*(128*16)` | 32768 = `0x8000` | `[0x2400,0xa400)` |
| SA | `1*(2*4*16)*4` | 512 = `0x0200` | `[0xa400,0xa600)` |
| SB | `(256/32)*(256/4)*4` | 2048 = `0x0800` | `[0xa600,0xae00)` |
| `PITCH` | 512-B aligned sum | 44544 = `0xae00` | — |

完整 ring：

| Region | slot0 | slot1 |
|---|---|---|
| A | `[0x0000,0x2400)` | `[0xae00,0xd200)` |
| B | `[0x2400,0xa400)` | `[0xd200,0x15200)` |
| SA | `[0xa400,0xa600)` | `[0x15200,0x15400)` |
| SB | `[0xa600,0xae00)` | `[0x15400,0x15c00)` |

```text
ARENA = 2*0xae00 = 0x15c00 = 89,088 B
```

它与 L1509/L1634 的 fixed group segment 完全一致。epilogue 在所有 input
TDM/DS 生命周期结束后复用 `[0,0x4000)`，不额外增加 LDS allocation。

### 6.2 `wpt4` 的真正含义

`num_waves_per_tensor_tdm=4` 令四个 logical jobs 的 owner list 都是：

```text
(wave0,wave1,wave2,wave3)
```

这不是“四个 wave 各自完整复制四个 tensor”，而是每个 tensor 的 outer/inner
extent 被四个 wave 平分：

| wave | A rows | B N16 rows | SA physical i32 | SB N32 rows | payload/K256 |
|---:|---|---|---|---|---:|
| 0 | `0..15` | `0..3` | `0..31` | `0..1` | 10,880 B |
| 1 | `16..31` | `4..7` | `32..63` | `2..3` | 10,880 B |
| 2 | `32..47` | `8..11` | `64..95` | `4..5` | 10,880 B |
| 3 | `48..63` | `12..15` | `96..127` | `6..7` | 10,880 B |

```text
per wave/K256 = A 2048 + B 8192 + SA 128 + SB 512 = 10880 B
per WG/K256   = 4*10880 = 43520 B nominal payload
LDS footprint = 44544 B  # 多出的 1024 B 是 64 rows * 16-B A padding
```

与旧 `wpt2` 分析相比，当前 v0 每个 wave 每 K tile 发 **4 条** input TDM，
并且只有一份共同 steady loop；旧文档中的“两组 owner branch、每 wave 2 条
TDM”不适用于本文件。

### 6.3 b2 pipeline

```text
K_TILES = 7168/256 = 28
TDM_PER = 4 jobs/wave/K256
```

高层时序：

```text
Prologue:
  issue slot0/kt0: 4 TDM per wave
  issue slot1/kt1: 4 TDM per wave
  outstanding upper bound = 8 per wave

Steady, kt=0..25 (26 iterations):
  s_wait_tensorcnt 4
  WG barrier                        # current slot 四类 segment 全部可见
  compute one K256
  WG barrier                        # 四个 wave 都不再读取 current slot
  issue same slot for kt+2: 4 TDM per wave

Drain:
  kt26/slot0: s_wait_tensorcnt 4 -> barrier -> compute
  kt27/slot1: s_wait_tensorcnt 0 -> barrier -> compute
```

动态 input TDM 数：

```text
per wave = 8 prologue + 26*4 refill = 112
per WG   = 4*112 = 448
```

### 6.4 当前 ISA 对应

| 阶段 | ISA |
|---|---|
| 8 个 prologue TDM sites | L377、L511、L517、L538、L539、L544、L548、L553 |
| steady label | `.LBB0_3`，L578 |
| current ready | L579 `s_wait_tensorcnt 4` |
| pre-compute WG barrier | L580 / L589 |
| post-compute WG barrier | L654 / L664 |
| 4 个 refill TDM | L665、L674、L685、L695 |
| 26-trip backedge | L693-L696，比较 `0x1a00` 后回 `.LBB0_3` |
| drain label | `.LBB0_4`，L697 |
| kt26 ready/barrier | L698-L701 |
| kt27 ready/barrier | L768-L776 |

`0x1a00 = 26*0x100` 是 compiler 的 K-tile byte/index induction 终点，静态
枚举和源码的 `n_steady=28-2=26` 一致。

### 6.5 为什么 wait/barrier 都需要

CDNA5 ISA §10.11.1 的硬件事实：

- tensor instruction 不按 lane 执行；
- 忽略 EXEC；
- 每条 instruction 对 issuing wave 的 `TENSORcnt` 加一；
- 同一 wave 的 tensor ops 按 issue 顺序完成；
- 不同 wave 的 tensor ops 彼此无序。

因此 `s_wait_tensorcnt 4` 只证明**本 wave**较老的四条 current-slot jobs 已经
完成；紧随其后的 WG barrier 才把四个 wave 的 A/B/SA/SB segment 合并为整个
WG 可见的完整 LDS tile。

CDNA5 ISA §5.7.1.4 规定 DScnt 按 LDS **machine instruction**计数；
`ds_load_2addr_b32` 虽然返回两个地址，仍只令 DScnt 加一。

<a id="sec-7"></a>

## 7. K=7168 compute hot loop 与 WMMA 映射

### 7.1 每个 K256 的 DS 需求

一个 wave 的 raw output tile 是 M64×N64。每个 K128：

| 数据 | 逻辑覆盖 | machine loads |
|---|---|---:|
| A payload | 4 个 M16 × K128 | 8 × `ds_load_b128` |
| B payload | 2 个 N32 × K128 | 8 × `ds_load_b128` |
| A scale | 4 M16 对应的 K32 scales | 2 logical b32 addresses |
| B scale | 2 N32 对应的 K32 scales | 2 logical b32 addresses |

两个 K128 合计：

```text
32 ds_load_b128
4  ds_load_2addr_b32
= 36 DS machine instructions
= 40 logical LDS addresses
```

steady body L591-L653 完整展示这种结构。SB 的两条 dual-load 在 KSL0 提前
取回 KSL0/KSL1 两组 scale；SA 则在 L592 与 L620 分别取 KSL0/KSL1。

### 7.2 WMMA 硬件形状与 host 轴

CDNA5 ISA §7.12.6 / instruction reference 对
`V_WMMA_SCALE_F32_32X16X128_F4` 的定义：

```text
D(32x16 F32) = A(32x128 FP4) * B(128x16 FP4) + C(32x16 F32)
block scale = 32 K elements
```

当前 FlyDSL 把 host weight B 作为 hardware Matrix A，把 host activation A
作为 hardware Matrix B，所以 hardware `N32×M16` 结果在软件语义上对应
host `M16×N32` fragment。

每个 wave：

```text
M repetitions = 64/16 = 4
N repetitions = 64/32 = 2
fragments      = 4*2 = 8
WMMA/K128      = 8
WMMA/K256      = 16
```

### 7.3 Accumulator 映射

由 steady KSL0 的 A/B LDS offsets 与 WMMA operands 可重建：

| Logical fragment | wave-local output | F32 accumulator |
|---|---|---|
| `F00` | `M[0:15], N[0:31]` | `v112:v127` |
| `F01` | `M[0:15], N[32:63]` | `v96:v111` |
| `F10` | `M[16:31], N[0:31]` | `v80:v95` |
| `F11` | `M[16:31], N[32:63]` | `v64:v79` |
| `F20` | `M[32:47], N[0:31]` | `v48:v63` |
| `F21` | `M[32:47], N[32:63]` | `v32:v47` |
| `F30` | `M[48:63], N[0:31]` | `v16:v31` |
| `F31` | `M[48:63], N[32:63]` | `v0:v15` |

KSL0 与 KSL1 都读写同一 accumulator block；它们不是 16 个独立 output
fragments。

第一条实例（L610）：

```text
v_wmma_scale_f32_32x16x128_f4
  v[112:127],        # F00 D
  v[128:143],        # weight/B payload
  v[160:167],        # activation/A payload
  v[112:127],        # F00 C
  v230,              # B scale
  v232               # A scale
```

### 7.4 Steady K256 的实际 schedule

L578-L696 每次 backedge 执行 118 条 assembly mnemonics：

| 区域 | 当前 ISA | 关键内容 |
|---|---|---|
| ready | L579-L590 | tensor wait、barrier、四类 LDS base |
| KSL0 DS 前缀 | L591-L608 | 16 b128 + 3 dual scale |
| KSL0 WMMA | L609-L641 | 8 WMMA，穿插 KSL1 prefetch |
| KSL1 scale/data | L620-L640 | 剩余 16 b128 + 1 dual scale |
| KSL1 WMMA | L642-L653 | 8 WMMA |
| reuse/refill | L654-L696 | barrier、4 TDM、loop induction/backedge |

DScnt wait 序列：

```text
KSL0: 9 -> 4 -> 4 -> 0
KSL1: 6 -> 4 -> 2 -> 0
```

这些值表示 wait 返回时 `DScnt<=threshold`，不是 latency 或 cycle 数。

### 7.5 两个 drain K tiles

`.LBB0_4` 内连续放了两个 unrolled K256 body：

| Tile | Ready | DS | WMMA | DScnt waits |
|---|---|---:|---:|---|
| kt26 | L698-L701 | L703-L750 | L722-L767 | `9,4,4,8,4,2,0` |
| kt27 | L768-L776 | L778-L826 | L797-L838 | `9,4,4,8,4,2,0` |

两者都不再发 input TDM。kt26 允许 kt27 的四条 jobs outstanding，所以等 4；
kt27 是最后一槽，必须等 0。

### 7.6 Full-K 动态工作量

```text
per wave:
  28 K256 * 16 WMMA = 448 WMMA
  28 * 36           = 1008 DS machine loads
  28 * 40           = 1120 logical LDS reads

per active WG:
  4*448  = 1792 WMMA
  4*1008 = 4032 DS machine loads
```

每条 WMMA 覆盖：

```text
32*16*128 = 65,536 FMA = 131,072 FLOP
```

<a id="sec-8"></a>

## 8. Epilogue：clamp、SiLU、BF16 与 masked store

### 8.1 没有独立 dequant epilogue

E8M0 scale 已在每条 `v_wmma_scale_f32_32x16x128_f4` 内应用。进入 epilogue
时 `v0:v127` 已是 scaled F32 accumulators，不再有“先把 int/FP4 accumulator
乘 scale”的独立 dequant pass。

这是本 kernel 与普通 integer GEMM epilogue 的重要区别。

### 8.2 Pipeline fence 与 input LDS 生命周期结束

最后一条 WMMA 是 L838。L839 起 compiler 开始交错安排 epilogue。

关键同步：

| ISA | 作用 |
|---|---|
| L910 `s_wait_tensorcnt 0` | drain 本 wave input TDM |
| L917 `s_barrier_signal -1` | 告知本 wave 不再需要 input ring |
| L980 `s_barrier_wait -1` | 所有 wave 都可安全复用 LDS |
| L984 | 第一条 output `ds_store_2addr_b64` |

### 8.3 SiLU 公式与当前机器码

源码 `gemm_common_gfx1250.py:L160-L175` 的 `act1` 分支：

```text
g = min(g, limit)
u = med3(u, -limit, limit)
t = exp2(-log2(e)*g)
sigmoid = rcp(1+t)
out = g*sigmoid*u
```

ISA 使用 literal：

```text
0xbfb8aa3b = -log2(e) in F32
```

每个 wave 有：

```text
64*64 raw accumulators / 32 lanes = 128 raw F32 values/lane
= 64 (gate,up) pairs/lane
```

所以当前静态 epilogue 恰有：

```text
64 v_med3_num_f32
64 v_exp_f32_e32
64 epilogue v_rcp_f32_e32
64 v_pk_mul_f32
32 v_cvt_pk_bf16_f32
```

`v_cvt_pk_bf16_f32` 把两个 F32 转成一个 packed BF16 dword。CDNA5 ISA
§7.6 规定普通 convert 使用 RNE；当前 metadata round mode 也为 0。

### 8.4 Output LDS address map

定义：

```text
w       = compute wave 0..3
wm      = M16 fragment 0..3
wn      = raw N16 fragment 0..3
lane16  = lane % 16
kgrp    = lane // 16
p       = activated value inside one packed b64, 0..3

row     = 16*wm + lane16
raw_col = 64*w + 16*wn + 8*kgrp + 2*p
out_col = raw_col / 2
lds_byte = 2*(row*128 + out_col)
```

因此：

```text
output LDS shape = BF16 [64,128]
row stride       = 128*2 = 256 B = 0x100
range            = [0x0000,0x4000)
```

脚本穷举 `w,wm,wn,lane,p` 得到：

```text
elements   = 8192
unique     = 8192
holes      = 0
collisions = 0
row range  = 0..63
col range  = 0..127
```

源码每 wave 有 16 个 logical b64 stores；compiler 合并为 8 个
`ds_store_2addr_b64` sites（L984、L1050、L1180、L1469-L1473）。每个 site
对 wave32 aggregate 写 512 B，故：

```text
8*512 B/wave * 4 waves = 16,384 B = 0x4000
```

### 8.5 第二个 barrier 与 TDM output store

所有 output DS stores 后：

```text
L1476 s_wait_dscnt 0
L1477 s_barrier_signal -1
L1501 s_barrier_wait -1
L1502 tensor_store_from_lds s[8:11], s[0:7]
L1503 s_wait_tensorcnt 0
```

这里不是“wave0 单独发一个 WG-wide store”。`make_tdm_store(...,
num_warps=4)` 把 64 output rows 分给四个 wave；当前 ISA 的单个静态 site 会被
四个 wave 各执行一次：

```text
store wave q -> rows 16q..16q+15, all 128 output columns
```

L987-L996 的 wave-id arithmetic 形成 `16*q` row offset 和 `0x1000*q` LDS
chunk offset。对本例 `mn_oob=32`：

```text
q=0,1 -> 各写 16 valid rows
q=2,3 -> outer bound 为 0/OOB，global writes 被丢弃
```

global base：

```text
C + 2*(blk_m*3072 + blk_n/2)
```

每个 output TDM wave 的 semantic descriptor 是：

```text
data_size             = 2 B (BF16)
tile_dim0             = 128 elements
tile_dim1             = 16 rows
tensor_dim0_stride    = 3072 BF16 elements
global row origin     = blk_m + 16*q
LDS byte origin       = 0x1000*q
workgroup_mask        = 0 / store 时本来也会被硬件忽略
outer valid rows      = clamp(mn_oob - 16*q, 0, 16)
```

一个 active WG 最终写：

```text
32 rows * 128 cols * 2 B = 8192 B
```

CDNA5 ISA §10.11.6 规定 TDM load OOB 读零、store OOB 丢弃，这正是 A padding
和 output padding 的硬件边界语义。

<a id="sec-9"></a>

## 9. Control flow、labels 与寄存器生命周期

### 9.1 当前文件全部 function labels

| Label | 行 | 作用 |
|---|---:|---|
| symbol entry | L7 | kernel 入口 |
| `.LBB0_3` | L578 | 26-trip steady K256 loop |
| `.LBB0_4` | L697 | 两个 drain K tiles，随后 epilogue |
| `.LBB0_5` | L1504 | unified end |
| `.Lfunc_end0` | L1550 | assembler function end |

当前文件没有旧 dump 中的 `.LBB0_1/.LBB0_2` owner loops。

### 9.2 三条 conditional branch

| ISA | 条件 | 作用 |
|---|---|---|
| L227 `s_cbranch_scc1 .LBB0_5` | `expert>95` | 跳过 1152 tail WGs |
| L324 `s_cbranch_scc1 .LBB0_4` | wave-first thread id `>127` | 在 required block=128 下不可达 |
| L696 `s_cbranch_scc1 .LBB0_3` | steady induction 未到 `0x1a00` | 26-trip loop |

expert 分支是 WG-uniform，因为 `blk_m` 只依赖 `bid_x`。这保证四个 wave 不会在
后续 workgroup barrier 上分歧。

### 9.3 重要 SGPR 生命周期

| Register | 早期意义 | 后期意义 |
|---|---|---|
| `s[0:1]` | kernarg pointer | valid path 中被 descriptor/address scratch 复用 |
| `s[2:3]` | preloaded C pointer | 一直保留到 output global base 构造 |
| `s[4:5]` | L117 后是 `m_tile_map` pointer | lookup 后被 TDM descriptor scratch 复用 |
| `s34` | `blk_m` | 保留到 output address |
| `s48` | L10 后是 `i32_m` | L322 后是 `mn_oob=32` |
| `s49` | `i32_n=6144` | swizzle、stride 和 output descriptor |
| `s50` | `limit=7.0f` | 整个 SiLU epilogue 使用 |
| `s54` | expert lookup result | valid setup 后可复用 |
| `s55` | steady K-tile induction | L663 每轮递增 |
| `s58` | refill K byte/index progress | L693 与 `0x1a00` 比较 |
| `s60` | `wave=thread_idx.x>>5` | LDS/TDM segment 选择 |

### 9.4 重要 VGPR 生命周期

| Register | Compute 前/中 | Epilogue |
|---|---|---|
| `v0:v127` | 8 个 F32 accumulator blocks | clamp/SiLU/BF16 packed values |
| `v128:v159` | B/weight fragments | output address/data temporaries |
| `v160:v229` | A/B prefetched fragments | output temporaries |
| `v230:v235` | packed SB/SA scales | 生命周期结束后可复用 |
| `v236:v237` | KSL1 A scale | 生命周期结束后可复用 |
| `v238` | A LDS base/address | compute 结束后死亡 |
| `v192`、`v193` | lane-derived address/kgrp state | output mapping继续使用 |
| `v194:v197` | A/B/SA/SB LDS base components | ring address构造 |

`v0:v127` 在 prologue 有大段 zero initialization，steady/drain 的每条 WMMA
都原地累加这些 blocks；最后一个 K tile 后才允许 epilogue 改写它们。

<a id="sec-10"></a>

## 10. Correctness invariants、边界与性能瓶颈

### 10.1 必须成立的 correctness invariants

1. **Specialization 必须匹配。**
   `K=7168`、`E=96`、A4W4、`act1`、no-bias、BF16 output 都编入 symbol。
2. **Raw N 必须按 GUGU 配对。**
   W1 payload 与 scale 都必须是 `[g0,u0,g1,u1,...]`。
3. **Scale layout 必须匹配。**
   SA 必须使用 M64/K128/WMMA-row preshuffle；SB 必须使用 n32k4。
4. **`psum` 单调且表示 valid exclusive end。**
   本例必须是 `psum[e]=64e+32`。
5. **Active tile 的 outer bound 必须在 `(0,64]`。**
   本例恒为 32；padding 由 A-load zero-fill 与 output-store drop 处理。
6. **四个 wave 必须走同一 expert branch。**
   否则 workgroup barrier 会失配。
7. **TDM 不能靠 EXEC suppression。**
   tensor instructions 忽略 EXEC；tail 依赖 L227 scalar branch。
8. **block 必须是 128。**
   wave mapping、`wpt4` segment、barrier member count 和 metadata 都依赖四个
   wave32。
9. **grid 不能把 experts 再乘一次 batch。**
   96 experts 已编码在 contiguous-M 与 B expert dimension 中。
10. **output stride 使用 `i32_n/2`。**
    raw N=6144，但 C row stride 是 3072 BF16 elements。

### 10.2 边界行为

- 每个 active tile 的后 32 行仍执行 WMMA 和 SiLU，但不写 global。
- A payload padding rows 由 TDM OOB 返回 0；SA padding 区域仍可能被搬入 LDS，
  但只影响最终被 mask 的 fragments。
- sentinel M tiles 在 expert lookup 后整体退出。
- N 与 K 在固定测试中都精确整除，没有 N/K tail。
- 空 expert / 非 balanced histogram 是源码企图支持的更一般情形，但本文没有
  运行验证；当前 fixed-step lookup 的 raw tail value 97 说明不能把内部
  `expert` register 当成严格 canonical upper-bound。
- `f32_situ_beta`、`f32_situ_linear_beta` 对 `act1` 不生效。

### 10.3 潜在性能瓶颈

以下是静态风险，不是实测结论：

1. **B surface 主导。** 唯一 logical B payload+scale 约 2.09 GiB。
2. **A 被 N tiles 重复读取。** `cluster_n=1` 时同一 M64 A tile 被 24 个 N WG
   重复请求；没有 TDM multicast。
3. **50% M-lane useful ratio。** active M64 中只有 32 行有效，执行 FLOPs 是
   useful FLOPs 的 2 倍。
4. **1152 个 tail WGs。** 占静态 grid 的 1/3；每个仍执行约 221 条 entry/
   lookup mnemonics 后退出。
5. **TDM descriptor 数。** wpt4 每 wave/K256 发四个较小 segment，均衡了
   payload，但增加 descriptor issue 数；是否优于 wpt2 必须实测。
6. **高资源占用。** 87 KiB LDS 与 239 metadata VGPR 可能限制 residency。
   仅按 LDS 容量，320 KiB/WGP 最多容纳 3 个这样的 WG；最终 occupancy 还受
   VGPR、wave placement 和硬件限制共同决定。
7. **Epilogue TRANS 压力。** 每 wave 有 64 `v_exp_f32` + 64 `v_rcp_f32`。
8. **频繁 barrier。** 每个 active wave 动态执行 56 次 WG signal 和 56 次
   WG wait；实际 stall 时间未知。

<a id="sec-11"></a>

## 11. 静态指令计数、FLOPs 与 logical bytes

### 11.1 当前 `.s` 静态 mnemonic 分类

脚本只统计 symbol body 中的 assembly mnemonics，不把 directive、label、
metadata 算作 instruction：

| 类别 | 静态数 |
|---|---:|
| SALU/control/SMEM (`s_*`) | 618 |
| regular VALU (`v_*` excluding WMMA) | 700 |
| WMMA | 48 |
| DS | 116 |
| TDM | 13 |
| **总计** | **1495** |

说明：

- `v_dual_*` 按一条 assembly mnemonic 计；
- 4-DWORD WMMA 也按一条 mnemonic 计；
- 这不是 encoded DWORD 数、issue slot 数或 dynamic count。

关键 opcode：

| Opcode | 静态 sites | 行 |
|---|---:|---|
| `tensor_load_to_lds` | 12 | L377、L511、L517、L538、L539、L544、L548、L553、L665、L674、L685、L695 |
| `tensor_store_from_lds` | 1 | L1502 |
| `v_wmma_scale_f32_32x16x128_f4` | 48 | 16 steady + 32 drain |
| `ds_load_b128` | 96 | 3 个静态 K256 bodies × 32 |
| `ds_load_2addr_b32` | 12 | 3 × 4 |
| `ds_store_2addr_b64` | 8 | epilogue |
| `s_wait_tensorcnt` | 5 | L579、L698、L768、L910、L1503 |
| `s_wait_dscnt` | 23 | compute + output drain |
| `s_barrier_signal/wait` | 6 / 6 | steady、drain、epilogue |
| `v_exp_f32_e32` | 64 | SiLU |
| `v_rcp_f32_e32` | 64 | SiLU |
| `v_cvt_pk_bf16_f32` | 32 | output conversion |

### 11.2 Static text 与 dynamic path

| Region | ISA | static mnemonics |
|---|---|---:|
| valid entry/prologue | L8-L577 | 570 |
| one steady body | L579-L696 | 118 |
| drain + epilogue | L698-L1505 | 807 |
| tail entry to branch | L8-L227 | 220 |

展开 26-trip loop：

```text
active WG per-wave dynamic mnemonics
  = 570 + 26*118 + 807
  = 4445

tail WG per-wave path
  = 220 + s_endpgm
  = 221
```

这是控制流展开后的 mnemonic 数，不等价于 cycles。

### 11.3 理论 FLOPs

按 GEMM convention，FMA=2 FLOPs；不把 SiLU transcendental 算进 GEMM FLOPs。

```text
one active WG:
  2*64*256*7168 = 234,881,024 FLOP

all 2304 active WGs:
  2*(96*64)*6144*7168
  = 541,165,879,296 FLOP
  = 0.541165879296 TFLOP per dispatch

useful valid-row math:
  2*3072*6144*7168
  = 270,582,939,648 FLOP
  = 0.270582939648 TFLOP
```

执行/useful 比例正好为 2，因为每个 expert 的 64-row tile 只有 32 valid rows。

### 11.4 Useful unique logical bytes

以下口径与 test runner 的 effective-byte helper 一致：每个 valid A row 一次，
每个 active expert 的完整 B surface 一次；不含 LDS、descriptor、cache line、
replay 或 metadata。

| Surface | 公式 | Bytes |
|---|---:|---:|
| A payload | `3072*(7168/2)` | 11,010,048 = 10.5 MiB |
| A scale | `3072*(7168/32)` | 688,128 = 0.65625 MiB |
| B payload | `96*6144*(7168/2)` | 2,113,929,216 = 2016 MiB |
| B scale | `96*6144*(7168/32)` | 132,120,576 = 126 MiB |
| **useful reads** | sum | **2,257,747,968 B** |
| BF16 output | `3072*3072*2` | 18,874,368 = 18 MiB |
| **useful R+W** | sum | **2,276,622,336 B** |

### 11.5 Descriptor-level logical transfer bytes

如果按当前 2304 个 active WG 的 tile requests 计，而不做 cache/multicast
合并：

| Surface | Logical bytes |
|---|---:|
| A payload，考虑 `mn_oob=32` | 264,241,152 = 252 MiB |
| A scale，当前 descriptor 不做 row OOB | 33,030,144 = 31.5 MiB |
| B payload | 2,113,929,216 = 2016 MiB |
| B scale | 132,120,576 = 126 MiB |
| **logical input requests** | **2,543,321,088 B = 2425.5 MiB** |
| output | 18,874,368 = 18 MiB |
| **logical requests + output** | **2,562,195,456 B = 2443.5 MiB** |

额外地，96-entry lookup 的 scalar requests 按“不考虑 scalar cache”计：

```text
2304 active WGs * 9 loads * 4 B
+ 1152 tail WGs * 8 loads * 4 B
= 119,808 B
```

这些量都不是 HBM 实测。TDM OOB 是否触发外部 transaction、WGP$/L2 reuse、
request coalescing、cache line overfetch 与实际 HBM bytes 均未知。

<a id="sec-12"></a>

## 12. 与 production runner / ISA runner 的对应关系

### 12.1 `run_gemm_a4w4.sh` 的语义参数

脚本与本文固定语义完全对应：

| Script | Kernel contract |
|---|---|
| `--data-format a4w4` | symbol 中 `_fp4`，A4W4 |
| `--experts 96` | `_e96` |
| `--tokens 512 --topk 6` | 3072 routes，balanced 时 32/expert |
| `--model-dim 7168` | `_K7168` |
| `--inter-dim 3072` | raw N=6144，output N=3072 |
| `--act silu` | `_act1` |
| `--no-bias` | 无 `_bias`，bias ABI dead |
| `AITER_MOE_EXPERT_BALANCE=true` | `psum[e]=64e+32` |

### 12.2 当前 CSV 与当前 v0 artifact 的选择差异

当前 `aiter/configs/tuned_grouped_fmoe.csv:L73` 对同一 A4W4 case 写的是：

```text
tile_m=32, tile_n=256, tile_k=256
m_warp=1, n_warp=4, num_buffers=2
cluster_n=-1, waves_per_tensor_tdm=4, next_stage_prefetch=1
```

也就是说，在**没有额外环境变量或 launcher injection**时，当前
`run_gemm_a4w4.sh` 的语义 case 会命中 t32 配置，而不是本文的 t64 symbol。

本文 v0 的 t64/grid3456 契约与以下显式 runner 设置一致：

```text
AITER_TDM_TILE_M=64
```

`my_code/isa_runner/moe_gemm1_pipeline_launch_compare.py:L91-L101` 正是这样固定
环境，并把 target symbol 写成当前 `_wpt4`。因此：

- 脚本的模型/routing/activation 参数与本文一致；
- 当前 `.s` 的 tile/geometry 需要 t64 override 或直接 ISA runner launch；
- 不能声称“当前 bare script 必然自然选择此 symbol”。

这是当前工作树重新核验得到的差异，不沿用旧文档对 CSV L73 的描述。

### 12.3 Standalone `gemm_batch_isa_runner`

`my_code/isa_runner/gemm_batch_isa_runner.py` 对当前 symbol 固定：

```text
kernarg_size = 184
grid         = (3456,1,1)
block        = (128,1,1)
cluster      = (1,1,1)
planes       = 1
```

它通过 production shuffle/quant helpers 构造 A、B、SA、SB 与 `m_tile_map`，
并显式检查：

```text
routes=3072
valid rows/expert=32
contiguous_m=9216
working WGs=2304
tail WGs=1152
```

standalone runner 与 production ABI 有几个 dead-field/验证值差异：

| 字段 | production | standalone runner |
|---|---|---|
| `arg_bias` | aliases A | separate zero F32 buffer |
| `arg_quant_scale` | aliases C | one-element dummy F32 buffer |
| quant descriptor | `[1,9216,3072]` | manually packed `[1,1,1]` |
| `swiglu_limit` | 7.0 | 默认约 `3.0e38`，避免 saturation 掩盖 GEMM error |
| SiTU betas | 4.0 / 25.0 | 默认 1.0 / 1.0 |

这些差异不会改变当前 dead fields；`swiglu_limit` 是 live field，所以
standalone validation 的数值输出故意不等同于 production limit=7.0 的输出。

<a id="sec-13"></a>

## 13. 静态验证结果、资料索引与仍未知项

### 13.1 本文完成的静态验证

对当前 `.s` 和固定 shape 做了以下只读检查：

1. 文件为 LF-only，63,470 bytes，SHA256 如文首。
2. function body 静态统计为 1495 assembly mnemonics。
3. labels 只有 L7、L578、L697、L1504、L1550 所列五个。
4. 16-M swizzle 穷举 3456 个 `(Mtile,Ntile)`：
   `unique=3456, holes=0, collisions=0`。
5. 8-step expert lookup 穷举 144 个 M tiles：
   valid tiles 精确映射 0..95、`mn_oob=32`；tail raw result=97。
6. output LDS map 穷举 8192 个 BF16 elements：
   `unique=8192, holes=0, collisions=0`。
7. LDS region 算术得到 `PITCH=0xae00`、`ARENA=0x15c00`，与 metadata 一致。
8. K pipeline 算术得到 26 steady +2 drain、448 WMMA/wave。
9. ABI flattened fields、padding、最终 offset `0xb8` 与 metadata/runner 一致。
10. useful bytes、descriptor logical bytes 与 FLOP 数分别独立枚举。

### 13.2 硬件事实来源

主要本地来源：

1. `mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt`
   - §2.2：wave/workgroup/LDS；
   - §5.6：workgroup barrier；
   - §5.7.1：DScnt/TENSORcnt；
   - §5.7.2.1：expert scheduling mode；
   - §7.6：convert 与 RNE；
   - §7.12.6、instruction reference p.475：block-scaled F4 WMMA；
   - §10.11：TDM、descriptor、OOB。
2. `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`
   - §3.3.2：VGPR allocation；
   - §3.4.3：`MODE.REPLAY_MODE`；
   - §3.4.9：`SCHED_MODE.DEP_MODE`；
   - §4.3.7：dependency scheduling；
   - §4.6.12.6：block-scale WMMA。

MI400 Guide 另有一条重要 stepping 限定：它写明
`V_WMMA_SCALE_F32_32X16X128_F4` 存在于 MI450-B0、而非 MI450-A0。当前
artifact target 只写 `gfx1250`，没有给出可由本文确认的具体 stepping；因此不把
“所有 gfx1250 stepping 都支持该 opcode”作为无条件结论。

### 13.3 软件来源

- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
- `aiter/ops/flydsl/kernels/gemm_common_gfx1250.py`
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py`
- `aiter/ops/flydsl/grouped_moe_gfx1250.py`
- `aiter/ops/flydsl/kernels/moe_contiguous_psum.py`
- `aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py`
- `aiter/ops/flydsl/moe_kernels.py`
- `aiter/ops/shuffle.py`
- `aiter/fused_moe.py`
- `op_tests/test_flydsl_grouped_gemm_gfx1250.py`
- `my_code/isa_runner/gemm_batch_isa_runner.py`
- `my_code/isa_runner/moe_gemm1_pipeline_launch_compare.py`
- `my_code/run_gemm_a4w4.sh`
- `aiter/configs/tuned_grouped_fmoe.csv`

历史背景可参见
[`gemm1_a4w4_tdm_silu_gfx1250_deep_dive.md`](gemm1_a4w4_tdm_silu_gfx1250_deep_dive.md)，
但其中旧 final ISA 的行号、wpt2 owner loops、64 个静态 WMMA sites 和旧 CSV
描述均不能用于当前 v0。

### 13.4 仍未知或必须实测的项目

1. 当前 `.s` 没有逐指令 PC/Vaddr 列，因此本文只引用准确 source line/label；
   encoded instruction address 需要对应 code object 的 disassembly。
2. `.amdhsa_next_free_vgpr=257` 与 metadata/set 的 239 口径差异原因未知。
3. 实际 occupancy、wave 到 SIMD 的 placement、VGPR/LDS 最终 residency 未测。
4. DScnt/TENSORcnt wait latency、barrier stall、TRANS latency 未测。
5. WGP$/L2 hit rate、TDM request 合并、实际 HBM read/write bytes 未测。
6. A OOB zero-fill 在外部 memory hierarchy 中是否完全不产生 request，静态文本
   不能证明。
7. output DS bank conflict、TDM destination bank behavior 未测。
8. 当前 bare `run_gemm_a4w4.sh` 与 t64 artifact 的准确选择链需要外部
   `AITER_TDM_TILE_M=64` 或 runner injection；脚本自身没有固定该变量。
9. MI450-A0/B0 或其它 gfx1250 silicon stepping 的实际运行目标未由 artifact
   metadata 建立。
10. 本文没有做数值 correctness 或性能运行；logical traffic 不应被引用为
    profiler/HBM 实测结果。
