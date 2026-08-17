# gfx1250 A4W4 MoE GEMM1：TDM、网格与端到端流水

本文只分析 `my_code/run_gemm_a4w4.sh` 当前触发的 A4W4 GEMM1，并且只保留下列三章。
基准形状是 `experts=96, tokens=512, topk=6, model_dim=7168,
inter_dim=3072, act=silu, no-bias`；不把旧 A8W4 文档中的
`experts=384`、`tokens=4096`、`inter_dim=768` 或 `b3` 结论移植到这里。

**目录**

- [1. Grid swizzle and expert lookup](#sec-1-grid-swizzle-and-expert-lookup)
- [2. t64/b2 A4W4 test launch: M/N/K partition and grid](#sec-2-t64-b2-a4w4-test-launch-mnk-partition-and-grid)
- [3. End-to-end software pipeline](#sec-3-end-to-end-software-pipeline)

**范围与证据。** 软件主证据是：

- `my_code/run_gemm_a4w4.sh:L1-L25`；
- `op_tests/test_flydsl_grouped_gemm_gfx1250.py:L261-L443,L1050-L1197`；
- `aiter/fused_moe.py:L693-L805`；
- `aiter/ops/flydsl/grouped_moe_gfx1250.py:L409-L697,L825-L1123`；
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py:L28-L150`；
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py:L44-L1294`；
- route、contiguous-psum、quant/preshuffle 与 shuffle helpers；
- `aiter/configs/tuned_grouped_fmoe.csv:L1,L73`；
- `my_code/flydsl_dump/moe_a4w4/a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1/21_final_isa.s`。

硬件语义以本地 `amd-instinct-cdna5-instruction-set-architecture.txt`（下称
**CDNA5 ISA**）和 `MI400_Shader_Programming#65.txt`（下称 **MI400 Guide**）
为准。文中“文档事实”来自 CDNA5 ISA §5.7.1、§7.12.6、§10.11 和
MI400 Guide §4.6.12.6、§4.10；“实现推断”则由当前源码与最终 ISA 共同推出。
本次只做静态分析，没有运行 GPU。

**先给结论。** 当前 production specialization 是
`a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1`。名字的
`a8w4_tdm` 是历史前缀；真正区分本例的是紧随其后的 `fp4`：
源码 `mxfp4_preshuffle_gfx1250_tdm.py:L186-L202` 在
`a_is_fp4=1` 时生成 `_afp="fp4"`，而最终 ISA 又只出现 A4W4 专用
`v_wmma_scale_f32_32x16x128_f4`。因此这里是 FP4 activation × FP4 weight，
不是 A8W4。

<a id="sec-1-grid-swizzle-and-expert-lookup"></a>
## 1. Grid swizzle and expert lookup

### 1.1 从 route 到真正传入 GEMM 的 `m_tile_map`

测试脚本设置 balanced routing：

```text
routes = tokens * topk = 512 * 6 = 3072
routes / expert         = 3072 / 96 = 32
```

`test_flydsl_grouped_gemm_gfx1250.py:L279-L285` 每个 token 依次激活 6 个
expert，并以 96 为周期轮转；3072 恰好是 96 的 32 倍，所以本例每个 expert
精确得到 32 条 route。这不是“平均约 32”，而是当前 deterministic balanced
构造的精确结果。

GEMM 前端没有直接把 `topk_id` 当 expert 索引交给主 kernel。实际链路为：

1. `flydsl_moe_topids_to_rows` 先用每 expert 的 atomic counter 建立 masked row：
   `row_masked = expert * max_m + slot`；
2. `contiguous_psum_remap` 对 `ceil(count[e]/tile_m)*tile_m` 做前缀和，并把
   masked row 原地改写成 `starts[e] + slot`；
3. 返回的 `psum[e] = starts[e] + count[e]` 被作为
   `flydsl_grouped_gemm_a8w4_masked(..., m_tile_map=psum)` 的参数。

对应代码在 `moe_kernels.py:L2432-L2560`、
`grouped_moe_gfx1250.py:L515-L531` 和
`moe_contiguous_psum.py:L199-L359`。当前 `tile_m=64`，故：

```text
aligned_count[e] = align_up(32, 64) = 64
starts[e]        = 64*e
psum[e]          = starts[e] + 32 = 64*e + 32,  e=0..95
```

这里最容易误读：`m_tile_map` 是 **actual end** `64e+32`，不是 aligned end
`64(e+1)`。每个 expert 的连续 tile 前 32 行是真 route，后 32 行是 tile
alignment padding。

route kernel 使用的临时 stride 与 GEMM 连续布局也不同：

```text
max_m                    = align_up(3072, 64) = 3072
masked row before remap  = expert*3072 + slot
contiguous row after     = expert*64   + slot
```

### 1.2 `cluster_n=1` 与一维 swizzle

CSV L73 的 `cluster_n=-1` 经
`grouped_gemm_mxfp4.py:L28-L41` 解析为 1；脚本也没有设置
`AITER_FLYDSL_MXFP4_CLUSTER_N`。虽然本例 24 个 N tiles 可被 2、3、4 整除，
当前实现不会自动选择 cluster，所以：

```text
cluster_n   = 1
swz_id      = bid_x
local_n     = none
n_units     = total_n_tiles
```

因此没有 cluster multicast，也没有 cluster-local block id。kernel
`mxfp4_preshuffle_gfx1250_tdm.py:L236-L260` 的通式在本例化简为：

```text
TILES_PER_GROUP = 16
total_n_tiles   = ceil(6144 / 256) = 24
total_m_tiles   = ceil(9216 / 64)  = 144
blocks_per_group = 24 * 16 = 384

group            = bid_x // 384
group_first_tile = group * 16
in_group         = bid_x % 384
group_tiles      = min(144 - 16*group, 16) = 16
m_tile           = 16*group + (in_group % 16)
n_tile           = in_group // 16
blk_m            = m_tile * 64
blk_n            = n_tile * 256
```

144 可被 16 整除，所以 9 个 group 全都是完整的 16-M-tile group。等价的
block 编号写法是：

```text
bid_x = 384*g + 16*n + ml
g=0..8, n=0..23, ml=0..15

=> m_tile = 16*g + ml
=> n_tile = n
```

同一个 N tile 内 M tile 变化最快；16 个 M tiles 完成后，N 才前进 256。
`cluster_n=1` 时 `a_mcast_mask=0`，A/SA 会被每个 N workgroup 独立读取。

### 1.3 96-entry lower-bound、padding 与 sentinel

kernel 在 `mxfp4_preshuffle_gfx1250_tdm.py:L265-L287` 对 96-entry `psum`
执行固定展开的 upper-bound：

```text
lo = 0, hi = 96
repeat ceil(log2(96)) + 1 = 8 times:
    mid = (lo + hi) >> 1
    go_right = psum[min(mid, 95)] <= blk_m
    lo = go_right ? mid + 1 : lo
    hi = go_right ? hi      : mid
expert = lo
```

也就是找第一个满足 `psum[expert] > blk_m` 的 expert。对
`m_tile=e, blk_m=64e`（`e>0`）：

```text
psum[e-1] = 64e - 32 <= 64e
psum[e]   = 64e + 32 >  64e
=> expert=e
=> mn_oob = psum[e] - blk_m = 32
```

`e=0` 时左边界直接是 `lo=0`，同样得到 `expert=0, mn_oob=32`。
所以 0..95 号 M tile 都能定位到对应 expert，但 TDM A-load 与 output-store
只允许 32 个有效 outer rows。tile 的后 32 行仍参与统一 WMMA 控制流，其
global A OOB 部分按 TDM descriptor 语义读零，最终 store 又被 `mn_oob=32`
截掉。CDNA5 ISA §10.11.2、§10.11.6 明确规定：load tile 超出 tensor
dimension 的部分返回零，store 超界部分被丢弃；这是硬件文档事实。

最后一个真实边界是：

```text
psum[95] = 95*64 + 32 = 6112
```

当 `m_tile>=96`、即 `blk_m>=6144` 时，96 个 `psum` 都不大于 `blk_m`，
lower-bound 返回 `expert=96`。这是 kernel 的 padding sentinel；
`if expert < n_experts` 失败后不发 TDM、不做 DS/WMMA，也不进 epilogue。
本例没有空 expert；若一般 routing 中某 expert 计数为 0，重复的 `psum`
会被同一 upper-bound 自然跳过。

静态容量可拆成：

```text
real route rows             = 3072
per-expert alignment rows   = 96*(64-32) = 3072
post-expert capacity tail   = 9216-96*64 = 3072
static contiguous_m         = 9216
```

也就是说，前 6144 行是 96 个“32 valid + 32 pad”的 expert tiles，后 3072
行才是直接命中 `expert=96` 的 sentinel tail。

### 1.4 ISA 对照

最终 ISA 中 8 次 psum lookup、sentinel branch 和有效 expert 的 `mn_oob`
load 都可与 ELF Vaddr 对齐。下表的 relative PC 以 kernel 入口
`Vaddr 0x1900` 为零：

| 作用 | ISA 行 | Vaddr | relative PC |
|---|---:|---:|---:|
| 8 次 psum lookup | L133,153,163,173,188,196,208,219 | `0x1b1c, 0x1b74, 0x1ba4, 0x1bd0, 0x1c10, 0x1c34, 0x1c6c, 0x1ca0` | `+0x021c .. +0x03a0` |
| sentinel 跳转到 end | L227 | `0x1ccc` | `+0x03cc` |
| valid expert 的 `mn_oob` load | L232 | `0x1ce4` | `+0x03e4` |
| 第一条 A/SA-owner prologue TDM | L274 | `0x1dc0` | `+0x04c0` |
| `s_endpgm` | L1672 | `0x40e0` | `+0x27e0` |

源码先形成 `mn_oob` 再进入 `if expert<n_experts`；compiler 最终把 sentinel
branch 调度到 L227，并把最后一次 psum load 放在 branch 之后。于是 sentinel
不会越界读取 `psum[96]`。这是最终 ISA 的控制流事实，不应只按 Python 源码的
文本顺序理解。

<a id="sec-2-t64-b2-a4w4-test-launch-mnk-partition-and-grid"></a>
## 2. t64/b2 A4W4 test launch: M/N/K partition and grid

### 2.1 为什么不是 `b3`，也不是 adapter launch

`run_gemm_a4w4.sh:L14-L25` 直接执行 production
`test_flydsl_grouped_gemm_gfx1250.py --scenario bench`，没有调用 ISA replay
adapter。配置 lookup 使用 padded token key
`_get_padded_m(512)=512`，命中 `tuned_grouped_fmoe.csv:L73`：

```text
tile_m/tile_n/tile_k = 64/256/256
m_warp/n_warp        = 1/4
num_buffers          = 2
cluster_n            = -1 -> 1
waves_per_tensor_tdm = -1 -> default 2
next_stage_prefetch  = 1, but b2 makes next_stage_on=0
```

最终 symbol 精确写成
`...t64x256x256_w1x4_b2_K7168_e96_act1`，且没有 `_prefetch` 后缀。
`mxfp4_preshuffle_gfx1250_tdm.py:L111-L112` 只有在
`next_stage_prefetch && num_buffers>=3` 时才打开 next-stage carry；所以 CSV
中的 `next_stage_prefetch=1` 在 `b2` specialization 中被编译掉。

CSV L73 虽有旧的 `split_k1=2` 列，但当前 TDM dispatcher
`grouped_moe_gfx1250.py:L1023-L1043` 并不读取该列，
`launch_gemm_a8w4_tdm` 也没有 split-K 参数。因此本例不能解释成
“2-way split-K”：`b2` 只表示两槽 LDS ring。

### 2.2 M：routes、alignment、capacity 与有效 tiles

前端在 `grouped_moe_gfx1250.py:L472-L478` 同时考虑 GEMM1/GEMM2 的 tile-M：

```text
valid_routes = T*topk = 512*6 = 3072
logical M per expert   = 3072/96 = 32

align_m = max(tile_m, tile_m2) = max(64,64) = 64
max_m   = align_up(valid_routes, align_m)
        = align_up(3072,64) = 3072

upper_bound = T*topk + E*align_m - topk
            = 3072 + 96*64 - 6
            = 9210
contiguous_m = align_up(9210,64) = 9216
```

`contiguous_m` 是 CUDA-Graph-safe 静态容量，不是实际 route 数，也不是
`E*max_m`。对应 M tiles：

```text
static M tiles   = 9216/64 = 144
expert M tiles   = 96      # 每 expert 一个 M64 tile
sentinel M tiles = 144-96  = 48
```

每个 expert tile 中只有前 32 行有效，因此“96 个有效 M tiles”不等于
“96*64 条 route”；其真实有效行仍是 `96*32=3072`。

### 2.3 N：raw gate/up 与激活后输出

GEMM1 调用位于 `grouped_moe_gfx1250.py:L630-L659`：

```text
raw GEMM N = two_inter = 2*inter_dim = 2*3072 = 6144
N tiles    = 6144/256 = 24
```

W1 在测试准备阶段被改成 GUGU row order
`[g0,u0,g1,u1,...]`，所以 epilogue 将相邻 raw N columns 组成
`(gate,up)`。每个 raw `N256` workgroup 最终写：

```text
STORE_N = tile_n/2 = 128
output N = raw N/2 = 3072
```

24 个 N workgroups 恰好覆盖 `24*128=3072` 个激活后列；没有 N tail。

### 2.4 Grid、block、有效与跳过 workgroups

launcher 在 `mxfp4_preshuffle_gfx1250_tdm.py:L1256-L1287` 使用：

```text
grid  = (ceil(contiguous_m/tile_m) * ceil(N/tile_n), 1, 1)
       = (144 * 24, 1, 1)
       = (3456, 1, 1) workgroups

block = (m_warp*n_warp*wave32, 1, 1)
       = (1*4*32, 1, 1)
       = (128, 1, 1) threads
```

静态与动态有效数量为：

| 项 | 计算 | 数量 |
|---|---:|---:|
| static workgroups | `144 M tiles * 24 N tiles` | 3456 |
| GEMM-valid workgroups | `96 expert tiles * 24 N tiles` | 2304 |
| sentinel-skipped workgroups | `48 tail tiles * 24 N tiles` | 1152 |
| valid output rows per valid WGP | `mn_oob` | 32 |

swizzle 的前 6 个 group（`g=0..5`）覆盖 96 个 expert M tiles；后 3 个
group（`g=6..8`）全部命中 sentinel。

### 2.5 单个 WGP 的 M/N wave partition

四个 wave 的逻辑坐标来自
`mxfp4_preshuffle_gfx1250_tdm.py:L227-L234`：

```text
warp_tile_m = 64/1 = 64
warp_tile_n = 256/4 = 64
wave_m      = wave//4 = 0
wave_n      = wave%4  = 0..3
```

所以所有 wave 都覆盖相同 M64，并沿 raw N 切开：

| wave | raw accumulator tile | SiLU 后 output tile |
|---:|---|---|
| 0 | `M64 × N[0:64]` | `M64 × Nout[0:32]` |
| 1 | `M64 × N[64:128]` | `M64 × Nout[32:64]` |
| 2 | `M64 × N[128:192]` | `M64 × Nout[64:96]` |
| 3 | `M64 × N[192:256]` | `M64 × Nout[96:128]` |

`wmma_m_rep=64/16=4`，`wmma_n_rep=64/16=4`。A4W4 不使用 4 条
N16 指令；专用 F4 opcode 一条覆盖两个相邻 N16，因此：

```text
mma_n_rep = wmma_n_rep/2 = 2
WMMA per K128 per wave = 4 M16 * 2 N32 = 8
```

CDNA5 ISA 的文档形状是
`D(32x16)=A(32x128)*B(128x16)+C`。当前 kernel 把 weight 放在硬件
Matrix A、activation 放在硬件 Matrix B；源码再把物理 `N32×M16`
accumulator 拆成逻辑 `M16×N32` 的两个 N16 fragments。这一 operand/逻辑轴
对应是实现推断，依据
`mxfp4_preshuffle_gfx1250_tdm.py:L617-L682,L986-L1000` 和最终 opcode；
指令本身的 32×16×128 形状及 FP4×FP4 则是 CDNA5 ISA §7.12.6、
instruction reference p.475 的文档事实。

### 2.6 K：每个有效 WGP 完整遍历 7168

K 没有跨 workgroup 切分：

```text
K           = 7168
tile_k      = 256
K_TILES     = 7168/256 = 28
WMMA_K      = 128
KWS         = 256/128 = 2
```

每个 K256 tile 是 KSL0/KSL1 两个 K128 body。每 wave：

```text
8 WMMAs/K128 * 2 = 16 WMMAs/K256
16 * 28          = 448 WMMAs/full-K
```

四个 wave 合计每个有效 WGP 动态执行：

```text
64 WMMAs/K256
64 * 28 = 1792 WMMAs/full-K
```

每条专用 WMMA 已经覆盖两个 N16；这里的 16/K256 不能替换成旧 A8W4
wide pipeline 的“16+16”。

三维切分可压缩为：

```text
M：跨 WGP 按 M64；每 expert 一个 tile，其中 32 valid + 32 pad。
N：跨 WGP 按 raw N256；WGP 内 4 waves 各 raw N64；激活后每 WGP 写 N128。
K：不 split-K；每个有效 WGP 依次执行 28 个 K256、每个含 2 个 K128。
Grid：(3456,1,1)，是 swizzled M×N 的一维展平，不存在 K grid 维。
```

<a id="sec-3-end-to-end-software-pipeline"></a>
## 3. End-to-end software pipeline

### 3.1 BF16、route 与 A4 quant/preshuffle

测试创建 `hidden[T,K] = BF16[512,7168]`，W1 logical shape 是
`[96,6144,3584 bytes]`（一个 byte 打包两个 FP4），W1 raw scale 是
`[96,6144,224]` E8M0 bytes；见
`test_flydsl_grouped_gemm_gfx1250.py:L355-L420`。

`fused_moe.py:L729-L757` 在 gfx1250 且未设置 `AITER_FORCE_A8W4=1` 时把
`q_dtype_a` 固定为 `fp4x2`。随后 grouped path 完成：

```text
BF16 hidden
  -> route atomic map (12 blocks, 256 threads)
  -> contiguous psum/remap (1 block, 512 threads)
  -> route-indexed per-1x32 MXFP4 quant + scatter + A-scale preshuffle
  -> GEMM1 A4W4 TDM kernel
```

当前 A1 quant 输出 shape：

```text
a1_payload: [1, 9216, 7168/2] = [1,9216,3584] uint8
a1_scale  : [1, 9216/4, (7168/32)*4]
          = [1,2304,896] uint8/E8M0
```

`wmma_rep=4`，每个 scale 对应 K 方向 32 个 FP4。route-indexed quant
使用 `source_row=route//6`，把同一个 token 的 6 条 route 分别写到各自
contiguous destination row。因为 `grid_x=ceil(3072/8)=384 < 512`，
`moe_kernels.py:L3004-L3055` 选择 K-split quant：

```text
mx_blocks_per_row  = 7168/32 = 224
mx_blocks_per wave = 8
grid_y             = 224/8 = 28
quant grid         = (384,28,1), block=(256,1,1)
```

gfx1250 使用 8-wide
`v_cvt_scalef32_pk8_fp4_bf16`；相关事实和 layout 见
`moe_fused_route_quant_scatter.py:L92-L130,L157-L245,L1266-L1505`。
padding row 不由 route quant 写入，但 GEMM 的 `mn_oob=32` 保证它不会作为
有效 global row 被消费。

### 3.2 W1/W1-scale 的 GUGU 与 n32k4 layout

测试输入首先保持 logical GGUU：

```text
[g0..g3071, u0..u3071]
```

然后 `moe_shuffle_weight(..., is_guinterleave=True, gate_up=True)` 先改成：

```text
[g0,u0,g1,u1,...,g3071,u3071]
```

再对 packed-byte weight 做 16-row × 16-byte preshuffle。W1 scale 也先做
相同 GUGU row interleave，再折叠成 n32k4：

```text
raw W1 scale      : [96,6144,224]
preshuffled scale : [96,6144/32,224*32]
                  = [96,192,7168] bytes
```

`shuffle.py:L117-L145,L240-L290` 给出了精确 permutation。n32k4 使一个
`ds_load_b32` 取得同一 WMMA K128 所需的 4 个 E8M0 bytes。相邻 GUGU raw
columns 正是 epilogue 的 `(gate,up)` 配对；如果把 GGUU 直接传给该 kernel，
数值会错。

### 3.3 b2 ring 与 LDS layout

对 `t64x256x256, a_is_fp4=1`，
`mxfp4_preshuffle_gfx1250_tdm.py:L148-L178` 化简为：

| LDS region / slot | 计算 | bytes | slot0 range |
|---|---:|---:|---:|
| A FP4 | `64*(256/2+16)` | 9216 | `0x0000..0x23ff` |
| B FP4 | `(256/16)*(256/2*16)` | 32768 | `0x2400..0xa3ff` |
| SA E8M0 | `1*(2*4*16)*4` | 512 | `0xa400..0xa5ff` |
| SB E8M0 | `(256/32)*(256/4)*4` | 2048 | `0xa600..0xadff` |
| `PITCH` | 512-byte aligned sum | 44544 = `0xae00` | — |

两槽 ring：

```text
slot0 = LDS + 0x00000
slot1 = LDS + 0x0ae00
ARENA = 2*0xae00 = 0x15c00 = 89088 bytes
```

这与 final ISA L1676 的
`.amdhsa_group_segment_fixed_size 89088` 精确相同。当前 ISA 没有旧文档中的
LDS 全量 zero-init；静态计数中 `ds_store_b128=0`。A row 的 16-byte padding
只用于 LDS pitch，实际 `load_a` 不读 pad；expert 尾行由 descriptor OOB
语义补零，最终 output 又由 `mn_oob` 限界。

### 3.4 每个 K256 的四类 TDM job 与 wave ownership

`num_waves_per_tensor_tdm=2`，四个 logical waves 被分成 `(0,1)` 与 `(2,3)`。
源码 `mxfp4_preshuffle_gfx1250_tdm.py:L347-L372,L446-L500` 的 job 分配为：

| owner waves | job | full-WGP K256 tile | nominal per owner |
|---|---|---:|---:|
| 0,1 | A | `64×128 B = 8192 B`，每 row LDS 再 pad 16 B | `32×128 = 4096 B` |
| 0,1 | SA | `128 i32 = 512 B`，按 inner K/WMMA layout 平分 | `64 i32 = 256 B` |
| 2,3 | B | `16×2048 B = 32768 B` | `8×2048 = 16384 B` |
| 2,3 | SB | `8×64 i32 = 2048 B` | `4×64 i32 = 1024 B` |

所以每 wave 每 K256 issue 两条 TDM：

```text
wave0/1: A + SA = nominal 4352 B/wave
wave2/3: B + SB = 17408 B/wave
WGP nominal payload = 43520 B
LDS stage footprint = 44544 B    # 多出的 1024 B 是 A row padding
```

balanced case 的 A outer extent 只有 32：wave0 的 A half 是有效数据，wave1
的 A half 全部越过 `mn_oob` 并补零；SA 仍按 preshuffled layout 搬运，但这些
padding-row accumulator 最终不会写到 global。

CDNA5 ISA §10.11.1 的文档事实是：tensor instruction 不按 lane 执行、
忽略 EXEC，并由每 wave 的 TENSORcnt 跟踪；因此“2 TDM/wave/Ktile”和
“8 TDM/WGP/Ktile”是动态 issue 数，不是把一个 site 误乘 lane 数。

### 3.5 A4W4 dedicated WMMA 与 K128 DS body

每个 K128 的 `load_state` 读取：

```text
A : 4 M16 * 2 ds_load_b128              = 8 b128
B : 2 N32 * (2 halves * 2 ds_load_b128) = 8 b128
SA: 2 logical b32 loads
SB: 2 logical b32 loads
```

compiler 把两对 b32 合成两个 `ds_load_2addr_b32`，所以每 wave/K128 是：

```text
16 ds_load_b128 + 2 ds_load_2addr_b32
= 18 DS machine instructions
= 20 logical LDS accesses

8 v_wmma_scale_f32_32x16x128_f4
```

steady body 对两个 K128 slice 的精确拆分是：

| 子阶段 | 本 slice 内 TDM | DS machine instructions | WMMA | `s_wait_dscnt` |
|---|---:|---:|---:|---|
| KSL0 | 0 | 18 | 8 | `9,4,4,0` |
| KSL1 | 0 | 18 | 8 | `6,4,2,0` |
| K256 tile tail | 2 | 0 | 0 | — |

也就是说，不能把 K256 的两条 TDM 平均后声称“每 K128 精确一条”；当前
post-compute ISA 把两条都放在 KSL1 完成后的 tile tail。tile-ready
`s_wait_tensorcnt 2` 则位于 KSL0 之前。

每个 steady K256 body 因而是：

| per wave / K256 | 数量或阈值 |
|---|---|
| TDM loads | 2，在该 tile compute 后为 `kt+2` 写回被复用 slot |
| DS machine instructions | `32 b128 + 4 dual-b32 = 36` |
| logical LDS accesses | 40 |
| dedicated WMMAs | 16（KSL0 8 + KSL1 8） |
| tensor wait | `s_wait_tensorcnt 2` 一次 |
| KSL0 DS waits | `9 -> 4 -> 4 -> 0` |
| KSL1 DS waits | `6 -> 4 -> 2 -> 0` |
| workgroup barriers | 两组 signal/wait |

CDNA5 ISA §5.7.1 说明 DScnt 按 LDS **instruction** 增减，`S_WAIT_DSCNT n`
等待 `DScnt<=n`；所以 dual-address 指令在 counter 统计上是一条 machine
instruction，而不是两条。§7.12.6 则规定该 WMMA 是 4-DWORD
VOP3PX2、FP4×FP4、FP32 accumulate，并使用 block-32 scale。

### 3.6 源码 b2 控制流与 compiler steady schedule

由于 `tile_m<=64` 且 `num_buffers=2`，源码选择
`mxfp4_preshuffle_gfx1250_tdm.py:L872-L917` 的 **post-compute** 分支，
不是旧 A8W4 文档的 mid-compute prefetch：

```text
Prologue:
  issue(slot0, kt0)
  issue(slot1, kt1)

Steady: n_steady = 28 - 2 = 26
  for kt=0..25:
    wait_tensorcnt 2
    WG barrier
    compute kt: KSL0 + KSL1
    WG barrier
    issue(reused slot, kt+2)

Drain:
  kt26: wait_tensorcnt 2; compute; no new TDM
  kt27: wait_tensorcnt 0; compute; no new TDM
```

compiler 为两组 TDM owners 生成两个互斥 steady loop；每个 wave 只执行其中
一个：

| dynamic path | ISA range | header / backedge | trip count |
|---|---|---|---:|
| A+SA owner waves 0/1 | L560-L658 | `0x2478 / 0x2790` | 26 |
| B+SB owner waves 2/3 | L763-L861 | `0x297c / 0x2c90` | 26 |

两份 steady body 都是 99 条静态 ISA，且各含 2 TDM、32
`ds_load_b128`、4 `ds_load_2addr_b32`、16 WMMA、8 DS waits。第一份 body
的精确 KSL anchors 为：

| phase | ISA lines | Vaddr |
|---|---|---|
| current-ready tensor wait / barrier | L560-L571 | `0x2478..0x24bc` |
| first DS read | L573 | `0x24c4` |
| KSL0 WMMA / DS-wait thresholds | L591-L623 | `0x2550..0x267f` |
| KSL1 WMMA / DS-wait thresholds | L624-L635 | `0x2680..0x270f` |
| post-compute barrier | L636,L646 | `0x2710,0x2740` |
| issue next A/SA or B/SB jobs | L647,L657 | `0x2744,0x2784` |
| loop backedge | L658 | `0x2790` |

这里的 compiler 顺序与高层源码一致到“post-compute issue”粒度，但 DS loads、
DS waits 和 16 条 WMMA 在 KSL 内被 machine scheduler 交错。不能把 source
`load_state()` 当成先完整执行 20 次 LDS access、再连续执行 8 条 WMMA。

两个 drain body 各保留 16 条 WMMA，但不再含 tensor load：

| tile | ready wait | WMMA range | DS wait thresholds |
|---|---|---|---|
| kt26 | L863 / `0x2c94`: `tensorcnt 2` | L889-L934 / `0x2d44..0x2f0b` | `9,4,4,8,4,2,0` |
| kt27 | L935 / `0x2f0c`: `tensorcnt 0` | L964-L1005 / `0x2fcc..0x317f` | `9,4,4,8,4,2,0` |

最终 ISA 的静态 site count 是：

```text
tensor_load_to_lds                   12
tensor_store_from_lds                 1
v_wmma_scale_f32_32x16x128_f4        64
ds_load_b128                         128
ds_load_2addr_b32                     16
s_wait_tensorcnt                       6
s_wait_dscnt                          31
```

64 个静态 WMMA sites 是“两份互斥 steady body各16 + 两个 drain各16”；
它不等于 runtime full-K 数。runtime 是每 wave 448、每有效 WGP 1792。

### 3.7 no-bias SiLU、BF16 store 与 GEMM2 交接

`--no-bias` 使 `_b1=None`、symbol 无 `_bias` 后缀，最终 ISA 也没有
`global_load*` bias 指令。A4W4 又使：

```text
_is_fp4    = true
_fuse_quant = (not _is_fp4) and (_b1 is None) = false
```

所以 GEMM1 必定先输出 BF16 `y`，再由独立 A4 quant/preshuffle kernel 准备
GEMM2；no-bias 并不会打开 A8W4 才有的 fused FP8-qout epilogue。

当前 `--swiglu-limit` 沿 CLI 默认值为 7.0。虽然
`grouped_moe_gfx1250.py:L543-L549` 附近注释说 limit 只用于 swiglu，实际
helper `gemm_common_gfx1250.py:L160-L175` 和最终 ISA 都表明 Silu 分支仍执行：

```text
g = min(gate, +7)
u = clamp(up, -7, +7)
sigmoid(g) = rcp(1 + exp2(-log2(e)*g))
out = g * sigmoid(g) * u
```

这是源码/ISA 可验证的当前行为；上述邻近注释是过时描述。ISA
L1006-L1634 可见 `v_min/max/med3`、常数 `0xbfb8aa3b=-log2(e)`、
64 个 `v_exp_f32`、64 个 epilogue `v_rcp_f32` 和 32 个
`v_cvt_pk_bf16_f32`。

source 在 full-K 后先做 `pipeline_fence(0)`，再运行 epilogue。machine
scheduler 将这个 fence 的独立部分穿插进 SiLU：

| 同步点 | ISA / Vaddr | 含义 |
|---|---|---|
| post-compute tensor drain | L1077 / `0x3358` | 当前 wave 的 TDM 全部完成 |
| post-compute barrier signal | L1084 / `0x3378` | 发出 WG rendezvous |
| post-compute barrier wait | L1147 / `0x34ec` | 第一条 LDS output store 前完成 |
| output LDS DS wait | L1643 / `0x4054` | 所有 output DS stores 完成 |
| output barrier | L1644,L1668 / `0x4058,0x40cc` | 四个 wave 的 LDS tile 可交给 TDM |
| output TDM store | L1669 / `0x40d0` | LDS BF16 -> global `y` |
| final tensor wait | L1670 / `0x40dc` | output TDM 完成 |

output store 的逻辑 view 是：

```text
LDS source  : BF16 [64,128]
GM base row : blk_m
GM base col : blk_n/2
GM stride   : raw_N/2 = 3072
outer bound : mn_oob = 32
```

因此一个有效 WGP 实际写 `BF16[32,128]`。GEMM1 的静态输出 tensor 是
`y[1,9216,3072]`，但只有 route rows 被后续消费。

随后 `grouped_moe_gfx1250.py:L660-L697` 执行：

```text
y BF16[1,9216,3072]
  -> per-1x32 MXFP4 quant + scale preshuffle
  -> a2_payload[1,9216,1536] uint8
  -> a2_scale[1,2304,384] E8M0
  -> GEMM2 A4W4: M=9216, N=7168, K=3072, same psum
  -> grouped_out BF16[1,9216,7168]
  -> gather-reduce(topids_to_rows, topk weights)
  -> final BF16[512,7168]
```

这才是当前 A4W4 的完整软件交接：GEMM1 内部终点是 BF16 gated-SiLU，
GEMM2 的 A4 输入由独立 quant/preshuffle 阶段产生。

**静态复核结果。** 本地 Python checker 直接解析
`19_gpu_module_to_binary.mlir` 的内嵌 ELF，并将 opcode 顺序与
`21_final_isa.s` 行号配对；结果为：symbol 入口 `0x1900`、function size
`0x27e4`、`reqd_workgroup_size=(128,1,1)`、wavefront 32、
LDS 89088 B、4 waves/WGP，且上述 64 WMMA、12 tensor-load、1 tensor-store、
两个 26-trip backedge 与所有列出的 Vaddr 一致。这里没有生成额外脚本文件。
