# gfx1250 A4W4 MoE GEMM1：TDM、网格与端到端流水

本文只分析 `my_code/run_gemm_a4w4.sh` 的唯一测试：`a4w4,
experts=96, tokens=512, topk=6, model_dim=7168, inter_dim=3072,
act=silu, no-bias, routing=balanced, t64/n256/k256/w1x4/b2`，并且只保留下列
三章。这里假定没有额外环境变量覆盖脚本或命中的 CSV 配置；脚本设置的
`AITER_MOE_EXPERT_BALANCE=true` 因而就是本文的 routing。

**Scope 约定。** 除明确标为 **【通式】** 或 **【仅作排除对照】** 的内容外，
下文每个公式、shape、数量和 ISA 统计都只属于上述唯一测试。每个【通式】后
都立即给出【本例代入】；源码中带 `a8w4` 的历史函数名、kernel 前缀，以及
`b3`/adapter，只在明确的命名或排除对照中出现，不代表本文测试配置。

**目录**

- [1. Grid swizzle and expert lookup](#sec-1-grid-swizzle-and-expert-lookup)
- [2. t64/b2 A4W4 test launch: M/N/K partition and grid](#sec-2-t64-b2-a4w4-test-launch-mnk-partition-and-grid)
  - [2.7 Kernel 输入参数 / ABI](#sec-2-7-kernel-input-abi)
- [3. End-to-end software pipeline](#sec-3-end-to-end-software-pipeline)

**范围与证据。** 软件主证据是：

- `my_code/run_gemm_a4w4.sh:L1-L25`；
- `op_tests/test_flydsl_grouped_gemm_gfx1250.py:L57-L81,L261-L302,L317-L443,L1050-L1197`；
- `aiter/fused_moe.py:L693-L805`；
- `aiter/ops/flydsl/grouped_moe_gfx1250.py:L409-L822,L825-L1123`；
- `aiter/ops/flydsl/moe_kernels.py:L2398-L2560,L2949-L3076`；
- `aiter/ops/flydsl/kernels/moe_route_maps.py:L153-L210`；
- `aiter/ops/flydsl/kernels/moe_contiguous_psum.py:L199-L368`；
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py:L28-L150`；
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py:L44-L1294`；
- quant/preshuffle 与 shuffle helpers；
- `aiter/configs/tuned_grouped_fmoe.csv:L1,L73`；
- `my_code/flydsl_dump/moe_a4w4/a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1/21_final_isa.s`。

硬件语义以本地 `amd-instinct-cdna5-instruction-set-architecture.txt`（下称
**CDNA5 ISA**）和 `MI400_Shader_Programming#65.txt`（下称 **MI400 Guide**）
为准。文中“文档事实”来自 CDNA5 ISA §5.7.1、§7.12.6、§10.11 和
MI400 Guide §4.6.12.6、§4.10；“实现推断”则由当前源码与最终 ISA 共同推出。
本次只做静态分析，没有运行 GPU。

**先给结论。** 【本例】production specialization 是
`a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1`。名字的
`a8w4_tdm` 是【仅作命名对照】的历史前缀；真正区分本例的是紧随其后的 `fp4`：
源码 `mxfp4_preshuffle_gfx1250_tdm.py:L186-L202` 在
`a_is_fp4=1` 时生成 `_afp="fp4"`，而最终 ISA 又只出现 A4W4 专用
`v_wmma_scale_f32_32x16x128_f4`。因此这里是 FP4 activation × FP4 weight，
不是 A8W4。

<a id="sec-1-grid-swizzle-and-expert-lookup"></a>
## 1. Grid swizzle and expert lookup

### 1.1 从 route 到真正传入 GEMM 的 `m_tile_map`

【本例】脚本 `run_gemm_a4w4.sh:L2,L14-L25` 设置 balanced routing 和唯一
shape。测试的 balanced 生成代码
`test_flydsl_grouped_gemm_gfx1250.py:L57-L76,L261-L288` 每个 token 连续置 6 个
expert score 为 1，并每次向后轮转 6；96/6=16 个 token 正好覆盖全部 expert，
而 512=32×16，因此：

```text
routes = tokens * topk = 512 * 6 = 3072
count[e]                = 3072 / 96 = 32,  e=0..95
```

这里的 32 是 atomic counter 完成后的 `count[e]`，**不是某条 route 的
`slot`**。`moe_kernels.py:L2432-L2560` 为非 EP 本例创建清零 counter，并以
`ceil(3072/256)=12` 个 256-thread blocks 启动 route kernel；真正的 atomic
及 masked-row 写法在 `moe_route_maps.py:L153-L184`。

**【通式】** 对扁平 route
`r = token*topk + topk_position`，`atomic_fetch_add(counter[e],1)` 返回加一前的
旧值：

```text
slot(r)       = atomic_fetch_add(counter[e], 1)  # old value, 0-based
row_masked(r) = e*max_m + slot(r)
```

**【本例代入】** `max_m=align_up(3072,64)=3072`，每个 expert 最终恰有
32 次 atomic，因此该 expert 实际出现的 slot **集合**是 `0..31`，masked
route rows 是 `e*3072+[0,31]`。哪条 `(token, topk_position)` route 获得集合
中的哪个 slot 取决于 atomic 执行顺序，不能把 balanced score 生成顺序误当成
slot 顺序，更不能把 `slot` 写成固定值 32。

GEMM 前的本例调用链是：

1. `grouped_moe_gfx1250.py:L474-L478,L527-L531` 计算
   `max_m=3072`、静态 `contiguous_m=9216`，调用 route 和 psum/remap；
2. `moe_contiguous_psum.py:L256-L301` 对每个 count 的 64-row 对齐值做 exclusive
   prefix sum，`L318-L328` 再把 masked row 原地改成 contiguous row；
3. `grouped_moe_gfx1250.py:L565-L575` 用改写后的 route row 做 A1
   quant/scatter，`L630-L659` 把 `psum` 传给 GEMM；
4. `grouped_gemm_mxfp4.py:L63-L150` 将该 `psum` 作为 `m_tile_map` 交给最终
   TDM GEMM。`starts` 只用于 row remap；GEMM expert lookup 收到的是 `psum`。

**【通式】** 对任意 route histogram 和 `tile_m`：

```text
aligned_count[e] = align_up(count[e], tile_m)
starts[0]        = 0
starts[e+1]      = starts[e] + aligned_count[e]
psum[e]          = starts[e] + count[e]          # valid exclusive end
row_contiguous   = starts[e] + slot
```

**【本例代入】** `tile_m=64, count[e]=32`：

```text
aligned_count[e] = align_up(32,64) = 64
starts[e]        = 64e
psum[e]          = 64e+32
row_contiguous   = 64e+slot,  e=0..95, slot∈[0,31]
```

所以 **只在本 balanced 测试中**，`starts` 与 `psum` 分别是公差 64 的等差数列。
若仍固定本例的 `tile_m=64`、只把 routing histogram 放宽，则【通式】是
`starts[e+1]-starts[e]=align_up(count[e],64)`、
`psum[e]=starts[e]+count[e]`；紧接的【本例代入】仍是差值 64 和
`psum[e]=64e+32`。一般 histogram 的 `count[e]` 不同，二者就不必是等差数列。

本例三个 expert 的 slot/row 边界如下；区间均为闭区间，`psum` 自身是 valid
exclusive end：

| expert `e` | `count[e]` / slot 集合 | masked route rows | `starts[e]` / `psum[e]` | contiguous valid | alignment padding |
|---:|---|---|---|---|---|
| 0 | `32` / `0..31` | `0..31` | `0 / 32` | `0..31` | `32..63` |
| 1 | `32` / `0..31` | `3072..3103` | `64 / 96` | `64..95` | `96..127` |
| 95 | `32` / `0..31` | `291840..291871` | `6080 / 6112` | `6080..6111` | `6112..6143` |

因此本例必须区分：每 expert `count=32`；每条 route 的
`slot∈[0,31]`；每 expert 有 32 行 alignment padding；psum scan 得到的 actual
aligned span 是 `96*64=6144`；调用方保留的 static `contiguous_m` capacity
是 9216；其 tail sentinel rows 是 `6144..9215`。psum kernel
`L298-L301` 会算出 actual span 6144，但
`grouped_moe_gfx1250.py:L528-L530` 有意忽略该第三返回值 `_`，GEMM launch
继续使用 graph-safe 静态容量 9216。

### 1.2 `cluster_n=1` 与一维 swizzle

【本例】CSV L73 的 `cluster_n=-1` 经
`grouped_gemm_mxfp4.py:L28-L41` 解析为 1；脚本也没有设置
`AITER_FLYDSL_MXFP4_CLUSTER_N`。虽然本例 24 个 N tiles 可被 2、3、4 整除，
当前实现不会自动选择 cluster，所以：

```text
cluster_n   = 1
swz_id      = bid_x
local_n     = none
n_units     = total_n_tiles
```

【本例 launch 参数】`cluster_n=1` 使用普通非 cluster launch，grid 只在 X
维展开。由 static `contiguous_m=9216`、raw GEMM `N=6144` 和
`tile_m/tile_n=64/256` 得：

```text
grid.x = ceil(9216/64) * ceil(6144/256)
       = 144 * 24
       = 3456

block.x = m_warp * n_warp * wave32
        = 1 * 4 * 32
        = 128

grid    = (3456, 1, 1)
block   = (128, 1, 1)       # 4 waves/WG
cluster = (1, 1, 1)         # 不设置 cluster launch attribute
```

所以硬件静态启动 3456 个 workgroups。其中 `96*24=2304` 个 WG 对应
96 个实际 expert M tiles 与 24 个 N tiles；其余 `48*24=1152` 个 WG
落在 static tail 的 sentinel M tiles，并在 expert lookup 后跳过。

因此本例没有 cluster multicast，也没有 cluster-local block id。kernel
`mxfp4_preshuffle_gfx1250_tdm.py:L236-L260` 的【通式】是源码中的
16-M-tile group swizzle；紧接的【本例代入】为：

```text
TILES_PER_GROUP = 16
total_n_tiles   = ceil(6144 / 256) = 24
total_m_tiles   = ceil(9216 / 64)  = 144
blocks_per_group = 24 * 16 = 384

group            = bid_x // 384
group_first_tile = group * 16
in_group         = bid_x % 384
group_tiles      = min(144 - 16*group, 16) = 16,  group=0..8
m_tile           = 16*group + (in_group % 16)
n_tile           = in_group // 16
blk_m            = m_tile * 64
blk_n            = n_tile * 256
```

各变量在本例中的范围如下。`bid_x` 是一维 grid 中的物理 workgroup id；
其余变量均由它确定：

| 变量 | 本例范围 | 含义 |
|---|---|---|
| `bid_x` | `0..3455` | grid 共 `144*24=3456` 个 workgroups |
| `group` | `0..8` | 每组覆盖 16 个连续 M tiles，共 `144/16=9` 组 |
| `group_first_tile` | `0,16,...,128` | 当前 group 的首个 M-tile id，即 `16*group` |
| `in_group` | `0..383` | 当前 workgroup 在 `16 M tiles * 24 N tiles` 组内的线性 id |
| `group_tiles` | 恒为 `16` | 当前 group 中有效的 M-tile 数；最后一组也恰有 16 个 |
| `in_group % 16` | `0..15` | 当前 group 内的局部 M-tile id，记作 `ml` |
| `n_tile` | `0..23` | N-tile id，对应 raw GEMM N 方向的 24 个 `N256` tiles |
| `m_tile` | `0..143` | 静态 M-tile id；对固定 `group=g`，范围为 `16g..16g+15` |
| `blk_m` | `0,64,...,9152` | 当前 tile 的 M 起点；tile 覆盖 `blk_m..blk_m+63` |
| `blk_n` | `0,256,...,5888` | 当前 tile 的 raw N 起点；tile 覆盖 `blk_n..blk_n+255` |

因此 `blk_m` 的静态覆盖终点是 `9152+63=9215`，正好覆盖
`contiguous_m=9216` 的容量；`blk_n` 的覆盖终点是 `5888+255=6143`，
正好覆盖 raw GEMM `N=6144`。这些是 launch 的静态范围，其中
`m_tile=96..143` 对应最后 48 个 sentinel M tiles，会在后续 expert lookup
后跳过。

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

### 1.3 96-entry `psum` expert lookup、padding 与 sentinel

【本例】kernel 在 `mxfp4_preshuffle_gfx1250_tdm.py:L265-L287` 对 96-entry
`psum` 执行固定展开的 upper-bound：

```text
lo = 0, hi = 96
repeat ceil(log2(96)) + 1 = 8 times:
    mid = (lo + hi) >> 1
    go_right = psum[min(mid, 95)] <= blk_m
    lo = go_right ? mid + 1 : lo
    hi = go_right ? hi      : mid
expert = lo
```

也就是找第一个满足 `psum[expert] > blk_m` 的 expert。这里再次强调，本例
`psum[e]=64e+32` 是 expert 的 **valid exclusive end**，不是 aligned end
`64(e+1)`。对本例
`m_tile=e, blk_m=64e`（`e>0`）：

```text
psum[e-1] = 64e - 32 <= 64e
psum[e]   = 64e + 32 >  64e
=> expert=e
=> mn_oob = psum[e] - blk_m = 32
```

`e=0` 时左边界直接是 `lo=0`，同样得到 `expert=0, mn_oob=32`。
所以本例 0..95 号 M tile 都定位到同号 expert，且
`mn_oob=psum[e]-blk_m=32`。这就是 `m_tile_map=psum` 区分 valid 与 padding
的方式：每个 expert 只有一个以 `64e` 开始的 M64 tile，前 32 行有效；
`64e+32..64e+63` 的 32 行 padding 不会另起 WGP，而是由 A-load 和
output-store 的 outer bound 32 截掉。统一 WMMA 控制流仍计算 M64，A 的越界
部分按 TDM descriptor 语义读零，store 越界部分被丢弃。CDNA5 ISA
§10.11.2、§10.11.6 对 load/store OOB 的规定是硬件文档事实。

最后一个真实边界是：

```text
psum[95] = 95*64 + 32 = 6112
```

当本例 `m_tile>=96`、即 `blk_m>=6144` 时，96 个 `psum` 都不大于
`blk_m`，upper-bound 返回 `expert=96`。这是 static-capacity tail sentinel；
`if expert < n_experts` 失败后不发 TDM、不做 DS/WMMA，也不进 epilogue。
【通式算法说明；本例无空 expert】若其它 histogram 的某 expert 计数为 0，
重复的 `psum` 会被同一 upper-bound 跳过；紧接回到【本例】，96 个
`count[e]` 全是 32，不存在这条空-expert 路径。

【本例】静态容量按“行数”拆成：

```text
valid route rows（总数）          = 3072
per-expert alignment padding（总数） = 96*(64-32) = 3072
actual aligned span               = 3072+3072 = 6144  # rows 0..6143
static-capacity sentinel tail      = 9216-6144 = 3072 # rows 6144..9215
static contiguous_m capacity       = 9216
```

也就是说，本例前 6144 行是 96 个“32 valid + 32 padding”的 expert tiles；
后 3072 行、即 `6144..9215`，才是直接命中 `expert=96` 的 48 个完整
sentinel M64 tiles。`count=32`、每 expert padding 32、actual span 6144、
static capacity 9216 和 tail 3072 是五个不同概念。

### 1.4 ISA 对照

【本例 final ISA】8 次 psum lookup、sentinel branch 和有效 expert 的
`mn_oob` load 都可与 ELF Vaddr 对齐。下表的 relative PC 以 kernel 入口
`Vaddr 0x1900` 为零：

| 作用 | ISA 行 | Vaddr | relative PC |
|---|---:|---:|---:|
| 8 次 psum lookup | L133,153,163,173,188,196,208,219 | `0x1b1c, 0x1b74, 0x1ba4, 0x1bd0, 0x1c10, 0x1c34, 0x1c6c, 0x1ca0` | `+0x021c .. +0x03a0` |
| sentinel 跳转到 end | L227 | `0x1ccc` | `+0x03cc` |
| valid expert 的 `mn_oob` load | L232 | `0x1ce4` | `+0x03e4` |
| 第一条 A/SA-owner prologue TDM | L274 | `0x1dc0` | `+0x04c0` |
| `s_endpgm` | L1672 | `0x40e0` | `+0x27e0` |

源码先形成 `mn_oob` 再进入 `if expert<n_experts`；compiler 最终把 sentinel
branch 调度到 L227，并把仅供 valid expert 计算 `mn_oob` 的 psum load 放在
L232、即 branch 之后；8 次 lookup load 本身又把索引 clamp 到 95。因此
sentinel 路径既不执行 L232，也不会读取 `psum[96]`。这是本例 final ISA 的
控制流事实，不应只按 Python 源码文本顺序理解。

<a id="sec-2-t64-b2-a4w4-test-launch-mnk-partition-and-grid"></a>
## 2. t64/b2 A4W4 test launch: M/N/K partition and grid

### 2.1 为什么不是 `b3`，也不是 adapter launch

【本例；`b3`/adapter 仅作排除对照】`run_gemm_a4w4.sh:L14-L25` 直接执行 production
`test_flydsl_grouped_gemm_gfx1250.py --scenario bench`，没有调用 ISA replay
adapter。配置 lookup 使用 padded token key
`_get_padded_m(512)=512`，命中 `tuned_grouped_fmoe.csv:L73`：

```text
tile_m/tile_n/tile_k = 64/256/256
m_warp/n_warp        = 1/4
num_buffers          = 2
cluster_n            = -1 -> 1
waves_per_tensor_tdm = -1 -> 本例解析值 2
next_stage_prefetch  = 1, but b2 makes next_stage_on=0
```

【本例】最终 symbol 精确写成
`...t64x256x256_w1x4_b2_K7168_e96_act1`，且没有 `_prefetch` 后缀。
`mxfp4_preshuffle_gfx1250_tdm.py:L111-L112` 只有在
`next_stage_prefetch && num_buffers>=3` 时才打开 next-stage carry；所以 CSV
中的 `next_stage_prefetch=1` 在 `b2` specialization 中被编译掉。

【仅作排除对照】CSV L73 虽有未被本路径消费的 `split_k1=2` 列，但当前 TDM dispatcher
`grouped_moe_gfx1250.py:L1023-L1043` 并不读取该列，
历史函数名 `launch_gemm_a8w4_tdm` 也没有 split-K 参数。因此本例不能解释成
“2-way split-K”：`b2` 只表示两槽 LDS ring。完整 production kernarg 映射见
[§2.7](#sec-2-7-kernel-input-abi)；旧 104 B adapter ABI 不适用于本次 launch。

### 2.2 M：routes、alignment、capacity 与有效 tiles

【本例】前端在 `grouped_moe_gfx1250.py:L472-L478` 同时考虑
GEMM1/GEMM2 的 tile-M。下面 `upper_bound` 第一行是实现【通式】，其下一行
立即给出【本例代入】：

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

本例 `contiguous_m=9216` 是 CUDA-Graph-safe 静态容量，不是 3072 条实际
route，也不是 `E*max_m`；1.1 中 psum scan 的 actual aligned span 仍是
6144。对应本例 M tiles：

```text
static M tiles   = 9216/64 = 144
expert M tiles   = 96      # 每 expert 一个 M64 tile
sentinel M tiles = 144-96  = 48
```

每个 expert tile 中只有前 32 行有效，因此“96 个有效 M tiles”不等于
“96*64 条 route”；其真实有效行仍是 `96*32=3072`。

### 2.3 N：raw gate/up 与激活后输出

【本例】GEMM1 调用位于 `grouped_moe_gfx1250.py:L630-L659`：

```text
raw GEMM N = two_inter = 2*inter_dim = 2*3072 = 6144
N tiles    = 6144/256 = 24
```

本例 W1 在测试准备阶段被改成 GUGU row order
`[g0,u0,g1,u1,...]`，所以 epilogue 将相邻 raw N columns 组成
`(gate,up)`。每个 raw `N256` workgroup 最终写：

```text
STORE_N = tile_n/2 = 128
output N = raw N/2 = 3072
```

24 个 N workgroups 恰好覆盖 `24*128=3072` 个激活后列；没有 N tail。

### 2.4 Grid、block、有效与跳过 workgroups

launcher 在 `mxfp4_preshuffle_gfx1250_tdm.py:L1256-L1287` 使用以下
【通式】，同一 block 内紧接【本例代入】：

```text
【通式】grid = (ceil(contiguous_m/tile_m) * ceil(N/tile_n), 1, 1)
【本例代入】 = (ceil(9216/64) * ceil(6144/256), 1, 1)
             = (144 * 24, 1, 1)
             = (3456, 1, 1) workgroups

【通式】block = (m_warp*n_warp*wave32, 1, 1)
【本例代入】  = (1*4*32, 1, 1)
              = (128, 1, 1) threads
```

【本例】静态与动态有效数量为：

| 项 | 计算 | 数量 |
|---|---:|---:|
| static workgroups | `144 M tiles * 24 N tiles` | 3456 |
| GEMM-valid workgroups | `96 expert tiles * 24 N tiles` | 2304 |
| sentinel-skipped workgroups | `48 tail tiles * 24 N tiles` | 1152 |
| valid output rows per valid WGP | `mn_oob` | 32 |

swizzle 的前 6 个 group（`g=0..5`）覆盖 96 个 expert M tiles；后 3 个
group（`g=6..8`）全部命中 sentinel。

### 2.5 单个 WGP 的 M/N wave partition

【本例】四个 wave 的逻辑坐标来自
`mxfp4_preshuffle_gfx1250_tdm.py:L227-L234`：

```text
warp_tile_m = 64/1 = 64
warp_tile_n = 256/4 = 64
wave_m      = wave//4 = 0
wave_n      = wave%4  = 0..3
```

所以本例所有 wave 都计算相同 M64，并沿 raw N 切开；由于 balanced
`mn_oob=32`，每个 wave 最终只写这个 M64 的前 32 行：

| wave | raw accumulator tile | 本例 SiLU 后有效 BF16 store |
|---:|---|---|
| 0 | `M64 × N[0:64]` | `M=blk_m+[0,31], Nout[0:32]` |
| 1 | `M64 × N[64:128]` | `M=blk_m+[0,31], Nout[32:64]` |
| 2 | `M64 × N[128:192]` | `M=blk_m+[0,31], Nout[64:96]` |
| 3 | `M64 × N[192:256]` | `M=blk_m+[0,31], Nout[96:128]` |

【本例】`wmma_m_rep=64/16=4`，`wmma_n_rep=64/16=4`。A4W4 不使用 4 条
N16 指令；专用 F4 opcode 一条覆盖两个相邻 N16，因此：

```text
mma_n_rep = wmma_n_rep/2 = 2
WMMA per K128 per wave = 4 M16 * 2 N32 = 8
```

【硬件指令形状，不是另一个测试 shape】CDNA5 ISA 的文档形状是
`D(32x16)=A(32x128)*B(128x16)+C`。当前 kernel 把 weight 放在硬件
Matrix A、activation 放在硬件 Matrix B；源码再把物理 `N32×M16`
accumulator 拆成逻辑 `M16×N32` 的两个 N16 fragments。这一 operand/逻辑轴
对应是实现推断，依据
`mxfp4_preshuffle_gfx1250_tdm.py:L617-L682,L986-L1000` 和最终 opcode；
指令本身的 32×16×128 形状及 FP4×FP4 则是 CDNA5 ISA §7.12.6、
instruction reference p.475 的文档事实。

### 2.6 K：每个有效 WGP 完整遍历 7168

【本例】K=7168 没有跨 workgroup 切分：

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

每条专用 WMMA 已经覆盖两个 N16；【仅作排除对照】这里的 16/K256 不能替换
成旧 A8W4 wide pipeline 的“16+16”。

【本例】三维切分可压缩为：

```text
M：跨 WGP 按 M64；每 expert 一个 tile，其中 32 valid + 32 pad。
N：跨 WGP 按 raw N256；WGP 内 4 waves 各 raw N64；激活后每 WGP 写 N128。
K：不 split-K；每个有效 WGP 依次执行 28 个 K256、每个含 2 个 K128。
Grid：(3456,1,1)，是 swizzled M×N 的一维展平，不存在 K grid 维。
```

<a id="sec-2-7-kernel-input-abi"></a>
### 2.7 Kernel 输入参数 / ABI

【源码事实】production FlyDSL kernel 的源码签名有 13 个参数，顺序见
`mxfp4_preshuffle_gfx1250_tdm.py:L204-L218`；launch 端以同一顺序构造 `kargs`
并启动 kernel（L1256-L1287）。其中 `arg_c`、`arg_scale_a`、`arg_scale_b`、
`arg_quant_scale` 是 `fx.Tensor`，其余 pointer 是裸 `fx.Pointer`。因此
`00_origin.mlir:L30` 的 13 个参数经 `01_fly_rewrite_func_signature.mlir:L30-L58`
展开成 **17 个 metadata rows**：每个 `fx.Tensor` 都变成一个
`global_buffer` pointer 加一个 packed `by_value` layout descriptor。最终 ISA
metadata L1742-L1804 给出 `.kernarg_segment_size=184`、segment alignment 8 B；
这不是旧 adapter 的 104 B。

【源码事实】本例启用了 kernarg-preload compile hint
（`tensor_shim.py:L20-L29`；
`mxfp4_preshuffle_gfx1250_tdm.py:L1290-L1293`），但 **final code object 实际只
preload 2 dword**：ISA L1678-L1685 是
`user_sgpr_count=4, kernarg_segment_ptr=1, preload_length=2,
preload_offset=0`。所以用户 SGPR 映射为：

```text
s[0:1] = 隐式 kernarg-segment pointer
s[2:3] = kernarg byte 0..7，即 preload 的 arg_c 64-bit pointer
```

入口没有为 `arg_c` 发 `s_load`；output 地址形成时 ISA L1653 仍直接使用
`s[2:3]`。其余真正被 specialization 消费的字段由入口
L10,L116,L234-L235,L313 从 `s[0:1]` 显式读取。下表的“对齐”是 lowered type
与实际 offset 所要求的 ABI packing：pointer 为 8 B，scalar dword 为 4 B；
Tensor companion 是 packed struct（descriptor 起点 8 B 对齐，但内部 i64
不额外补齐）。final metadata 没有逐项 `.name`/`.align`，故表中的参数名来自
上述源码签名与 rewritten signature 的顺序配对，而不是臆造 metadata 名。
所有 pointer 地址都是 **runtime 64-bit pointer**，不写伪造地址。

| kernel 参数 / metadata row | final register；kernarg byte offset | size / ABI alignment / `value_kind` | 当前测试具体值 | 语义/单位；pointer tensor 的 dtype、logical/physical layout 与 storage |
|---|---|---|---|---|
| `arg_c` pointer | preload `s[2:3]`；`0..7` (`0x00..0x07`) | 8 B / 8 B / `global_buffer` | runtime 64-bit pointer | 【源码事实】GEMM1 SiLU 后 `y`，`torch.bfloat16`，logical=physical visible shape `[1,9216,3072]`，contiguous row-major；row stride `3072*2 = 6,144 B (0x1800)`。【布局推导】storage `1*9216*3072*2 = 56,623,104 B (0x03600000)`。这里的 output `N=3072` 不等于 raw GEMM `i32_n=6144`。 |
| `arg_c.$layout` companion | 不加载；`8..35` (`0x08..0x23`) | 28 B / packed / `by_value` | `@8,@12,@16` 的 i32 shape=`1,9216,3072`；`@20,@28` 的 i64 stride=`28,311,552 (0x01b00000), 3,072 (0x0c00)` | 【源码事实】`!fly.memref<bf16, global, (?,?,?):(?i64,?i64,1)>` 的 hidden descriptor；stride 单位是 BF16 elements，末维固定 stride 1，故不另传。当前 kernel 只取 base iterator，final ISA 不读取这 28 B。 |
| `arg_a` | `s_load_b128` 后 `s[4:5]`；`40..47` (`0x28..0x2f`) | 8 B / 8 B / `global_buffer` | runtime 64-bit pointer | 【源码事实】`a1_payload` 是 `torch.uint8` on-wire MXFP4 E2M1（2 FP4/byte）；logical capacity `[1,9216,7168]` FP4，physical packed row-major `[1,9216,3584]` bytes，A payload **没有** tile preshuffle；TDM row stride `7168/2 = 3,584 B (0x0e00)`（kernel L339-L360,L446-L460）。【布局推导】storage `1*9216*3584 = 33,030,144 B (0x01f80000)`；只有 route 对应行被 quant kernel 写入，padding/sentinel 不由 GEMM 当有效行读取。 |
| `arg_b` | 同一 `s_load_b128` 后 `s[6:7]`；`48..55` (`0x30..0x37`) | 8 B / 8 B / `global_buffer` | runtime 64-bit pointer | 【源码事实】W1 为 `torch.uint8` on-wire MXFP4；数学 logical GGUU `[96,6144,7168]` 先变成 kernel 所需 GUGU row order `[g0,u0,g1,u1,...]`，再做 packed-byte tile shuffle（test L360-L420；`shuffle.py:L116-L141,L191-L204`）。visible byte shape 仍是 `[96,6144,3584]`；精确 storage order `[E,N/16,(K/2)/32,2,N_in16,Kbyte_in16]=[96,384,112,2,16,16]`，等价合并为 `[96,384,224,16,16]`。每个 N16 super-row stride `16*3584 = 57,344 B (0x0e000)`，每 expert stride `384*57344 = 22,020,096 B (0x01500000)`。【布局推导】storage `96*6144*3584 = 2,113,929,216 B (0x7e000000)`。 |
| `arg_scale_a` pointer | `s_load_b64` 后 `s[24:25]`；`56..63` (`0x38..0x3f`) | 8 B / 8 B / `global_buffer` | runtime 64-bit pointer | 【源码事实】每个 E8M0 byte 覆盖 K32；logical `[1,9216,224]` bytes。quant helper 直接写 folded scale，u8 allocation visible shape `[1,2304,896]`，wrapper 再传 `.view(torch.int32)`，故 ABI Tensor visible shape `[1,2304,224]` i32（`moe_kernels.py:L2949-L2994`；route-quant layout 见 `moe_fused_route_quant_scatter.py:L39-L55,L686-L747`）。【布局推导】byte storage order `[1,M/64,(K/32)/4,wmma_row,lane16,byte4]=[1,144,56,4,16,4]`；M64 tile stride `56*4*16*4 = 14,336 B (0x3800)`；storage `9216*224 = 2,064,384 B (0x001f8000)`。 |
| `arg_scale_a.$layout` companion | 不加载；`64..91` (`0x40..0x5b`) | 28 B / packed / `by_value` | i32 shape：`@64=1,@68=2304,@72=224`；i64 stride：`@76=516,096 (0x0007e000), @84=224 (0x00e0)` | 【源码事实】ABI `i32` Tensor 的 hidden descriptor；stride 单位为 i32 elements，末维固定 1。`516096*4 = 2,064,384 B (0x001f8000)`、`224*4 = 896 B (0x0380)` 分别是 batch 与 visible dim-1 byte stride。kernel 只消费 base pointer。 |
| `arg_scale_b` pointer | `s_load_b64` 后复用 `s[4:5]`；`96..103` (`0x60..0x67`) | 8 B / 8 B / `global_buffer` | runtime 64-bit pointer | 【源码事实】W1 raw E8M0 logical shape `[96,6144,224]`；先按 W1 同样变成 GUGU，再由 n32k4 fold 成 u8 visible `[96,192,7168]`，随后 caller 传 `.reshape(-1).view(torch.int32)`（`shuffle.py:L239-L289`；`grouped_moe_gfx1250.py:L581-L582`）。【布局推导】byte storage order `[E,N/32,(K/32)/4,N_in32,byte4]=[96,192,56,32,4]`；N32 super-row stride `56*32*4 = 7,168 B (0x1c00)`，expert stride `192*7168 = 1,376,256 B (0x150000)`；storage `96*6144*224 = 132,120,576 B (0x07e00000)`。 |
| `arg_scale_b.$layout` companion | 不加载；`104..107` (`0x68..0x6b`) | 4 B / packed / `by_value` | `@104` i32 length=`33,030,144 (0x01f80000)` | 【源码事实】flatten 后 ABI 是 `!fly.memref<i32,global,?:1>`，所以 companion 只带一个动态 i32 length；固定 stride 1 不传。该 length 乘 4 正好得到 `132,120,576 B (0x07e00000)`。 |
| `arg_m_tile_map` | `s_load_b64` 后 `s[8:9]`；`112..119` (`0x70..0x77`) | 8 B / 8 B / `global_buffer` | runtime 64-bit pointer | 【源码事实】`psum`，`torch.int32[96]` contiguous；本 balanced 测试 `psum[e]=64e+32`，即 `32,96,...,6112`。stride 4 B；storage `96*4 = 384 B (0x0180)`。kernel L265-L287 做 96-entry upper-bound lookup。 |
| `arg_bias` | final ISA 不加载；`120..127` (`0x78..0x7f`) | 8 B / 8 B / `global_buffer` | runtime 64-bit dummy pointer，**与 `arg_a` 同地址** | 【源码事实】`--no-bias` 令 `bias=None`；wrapper `grouped_gemm_mxfp4.py:L107-L109` 以 `ptr_arg(a)` 填这个统一签名槽位。它实际指向上面的 packed-A storage，不是 bias tensor；`has_bias=0` 是 compile-time specialization，kernel L1132-L1156 的 bias load 整段被删除，final ISA 也不读取 offset `0x78`，故没有额外 storage、绝不会解引用为 bias。 |
| `arg_quant_scale` pointer | final ISA 不加载；`128..135` (`0x80..0x87`) | 8 B / 8 B / `global_buffer` | runtime 64-bit dummy pointer，**与 `arg_c` 同地址** | 【源码事实】A4W4 的 `_fuse_quant=false`，GEMM1 走 BF16 output + 独立 quant；caller 未传 `quant_scale` 时 wrapper L109 令 `quant_scale_tensor=out`。所以此槽实际指向 `arg_c` 的 BF16 `[1,9216,3072]`，不是 E8M0 scale。`stage1_quant_out=0` 在 compile time 删除 kernel L1028-L1129 的 scale store，final ISA 不读取 offset `0x80`，无额外 storage。 |
| `arg_quant_scale.$layout` companion | 不加载；`136..163` (`0x88..0xa3`) | 28 B / packed / `by_value` | i32 shape：`@136=1,@140=9216,@144=3072`；i64 stride：`@148=28,311,552 (0x01b00000), @156=3,072 (0x0c00)` | 【源码事实】dummy alias 仍按实际 `out` Tensor 生成完整 BF16 rank-3 descriptor，与 `arg_c.$layout` 数值相同；compile-time dead path 使 pointer 与 descriptor 都不被 final ISA 读取。 |
| `i32_m` | L10 `s_load_b96` 后 `s40`；`164..167` (`0xa4..0xa7`) | 4 B / 4 B / `by_value` | `9,216 (0x00002400)` | 【源码事实】static contiguous-M capacity，单位 rows；不是 valid routes 3072，也不是 actual aligned span 6144。 |
| `i32_n` | 同一 load 后 `s41`；`168..171` (`0xa8..0xab`) | 4 B / 4 B / `by_value` | `6,144 (0x00001800)` | 【源码事实】**激活前 raw gate+up width**。kernel L1215-L1230 以 `out_stride=i32_n/2=3072`、`out_col=blk_n/2` 写 `arg_c`，所以 global output view 的 N 是 3072。 |
| `f32_swiglu_limit` | 同一 load 后 `s42`；`172..175` (`0xac..0xaf`) | 4 B / 4 B / `by_value` | `7.0f`，bits `0x40e00000` | 【源码事实】脚本未覆盖 CLI 默认 7.0（test L1097）。虽然本例 `stage1_act=1` 是 SiLU 而非 SwiGLU，这个参数**没有被编译掉**：helper `gemm_common_gfx1250.py:L160-L175,L297-L343` 仍做 `g=min(g,+7)`、`u=clamp(u,-7,+7)`；ISA L1006-L1622 多次直接使用 `s42`。 |
| `f32_situ_beta` | final ISA 不加载；`176..179` (`0xb0..0xb3`) | 4 B / 4 B / `by_value` | `4.0f`，bits `0x40800000` | 【源码事实】来自 test CLI 默认值（L1099-L1103），而不是 grouped helper 自身的 1.0 默认。参数为统一签名保留；`stage1_act=1` 使 `is_situv2=false`，kernel L1017-L1024 在 compile time 令 `situ_c=None`，故 final ISA 不读取此 dword。 |
| `f32_situ_linear_beta` | final ISA 不加载；`180..183` (`0xb4..0xb7`) | 4 B / 4 B / `by_value` | `25.0f`，bits `0x41c80000` | 【源码事实】来自 test CLI 默认值（L1105-L1108）；同样只对 `stage1_act=3` 的 SiTUv2 有效，本 `act1` final ISA 不读取。最后一个字段结束于 byte 184 (`0xb8`)。 |

【ABI 完整性复核】final metadata 的 17 rows 有效 payload 共 172 B；为满足后续
8-B pointer alignment，compiler 在 `36..39` (`0x24..0x27`)、
`92..95` (`0x5c..0x5f`) 和 `108..111` (`0x6c..0x6f`) 各插入 4 B
padding。于是 `172 B rows + 12 B padding = 184 B (0x00b8)`，并且最后字段
`180+4=184`，与 ISA L1678、L1803-L1804 两处完全一致。入口所有
kernarg-segment loads 也只覆盖表中 offset：`0xa4`、`0x70`、`0x28`、`0x38`
和 `0x60`；不存在沿用旧 104 B adapter 字段顺序的空间。

【不是 kernarg】grid/block、LDS 大小和 workgroup id 都是 launch/code-object
状态，不在上表 184 B 内。本例 launcher L1256-L1287 给出
`grid=(3456,1,1)`、`block=(128,1,1)`；`cluster_n=1` 走普通 launch 分支，
不传 cluster attribute（有效默认 geometry 为 `(1,1,1)`），也没有
dynamic-shared-memory 参数，所以
**dynamic LDS request=0 B**。共享 arena 已静态编入 code object，ISA
L1675-L1678 与 metadata L1802 给出固定 group segment
`89,088 B (0x15c00)`；wavefront size 32、required workgroup size
`(128,1,1)` 见 ISA L1805-L1819。隐式 workgroup-id X 由 kernel descriptor
L1690-L1692 启用，也不是 kernarg。

<a id="sec-3-end-to-end-software-pipeline"></a>
## 3. End-to-end software pipeline

### 3.1 BF16、route 与 A4 quant/preshuffle

【本例】测试创建 `hidden[T,K] = BF16[512,7168]`，W1 logical shape 是
`[96,6144,3584 bytes]`（一个 byte 打包两个 FP4），W1 raw scale 是
`[96,6144,224]` E8M0 bytes；见
`test_flydsl_grouped_gemm_gfx1250.py:L355-L420`。

在本文 scope 中未设置额外 override；`fused_moe.py:L729-L757` 因而在
gfx1250 把 `q_dtype_a` 固定为 `fp4x2`。随后 grouped path 完成：

```text
BF16 hidden
  -> route atomic map (12 blocks, 256 threads)
  -> contiguous psum/remap (1 block, 512 threads)
  -> route-indexed per-1x32 MXFP4 quant + scatter + A-scale preshuffle
  -> GEMM1 A4W4 TDM kernel
```

本例 A1 quant 输出 shape：

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

【本例】测试输入首先保持 logical GGUU：

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

【本例】对 `t64x256x256, a_is_fp4=1, num_buffers=2`，
`mxfp4_preshuffle_gfx1250_tdm.py:L148-L178` 化简为：

| LDS region / ring slot | 本例计算 | bytes | ring slot0 range |
|---|---:|---:|---:|
| A FP4 | `64*(256/2+16)` | 9216 | `0x0000..0x23ff` |
| B FP4 | `(256/16)*(256/2*16)` | 32768 | `0x2400..0xa3ff` |
| SA E8M0 | `1*(2*4*16)*4` | 512 | `0xa400..0xa5ff` |
| SB E8M0 | `(256/32)*(256/4)*4` | 2048 | `0xa600..0xadff` |
| `PITCH` | 512-byte aligned sum | 44544 = `0xae00` | — |

这里的 `ring slot0/slot1` 是 **b2 LDS buffer 槽**，与 1.1 中“某条 route 的
atomic allocation slot”完全不同。本例两槽 ring 为：

```text
lds_ring_slot0 = LDS + 0x00000
lds_ring_slot1 = LDS + 0x0ae00
ARENA          = 2*0xae00 = 0x15c00 = 89088 bytes
```

这与本例 final ISA L1676 的
`.amdhsa_group_segment_fixed_size 89088` 精确相同。【仅作排除对照】本例 ISA
没有旧文档中的 LDS 全量 zero-init；静态计数中 `ds_store_b128=0`。A row 的
16-byte padding 只用于 LDS pitch，实际 `load_a` 不读 pad；expert 尾行由
descriptor OOB 语义补零，最终 output 又由 `mn_oob` 限界。

### 3.4 每个 K256 的四类 TDM job 与 wave ownership

【本例】`num_waves_per_tensor_tdm=2`，四个 logical waves 被分成 `(0,1)` 与
`(2,3)`。
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

本 balanced 测试的 A outer extent 只有 32：wave0 的 A half 是有效数据，wave1
的 A half 全部越过 `mn_oob` 并补零；SA 仍按 preshuffled layout 搬运，但这些
padding-row accumulator 最终不会写到 global。

CDNA5 ISA §10.11.1 的文档事实是：tensor instruction 不按 lane 执行、
忽略 EXEC，并由每 wave 的 TENSORcnt 跟踪；因此“2 TDM/wave/Ktile”和
“8 TDM/WGP/Ktile”是动态 issue 数，不是把一个 site 误乘 lane 数。

### 3.5 A4W4 dedicated WMMA 与 K128 DS body

【本例】`wmma_m_rep=4, wmma_n_rep=4, mma_n_rep=2`。源码
`mxfp4_preshuffle_gfx1250_tdm.py:L684-L689` 的计数先按**逻辑 LDS
access**计算：

```text
DS_A     = 2
DS_B     = 2
sb_pairs = 2
sa_pairs = 2
BS_DS    = 4*2 + 2 + 2 = 12
STATE_DS = 4*2 + 12    = 20 logical accesses / K128
```

这里的 `STATE_DS=20` 不是最终 DScnt 增量；`emit_hints()` 在当前源码
L804-L805 立即 `return`，而且 backend 还会合并 scale load。按应用层
A/B/SA/SB 命名，每个 K128 的真实数据需求为：

| K128 slice | activation A | weight B | SA | SB | dedicated WMMA |
|---|---:|---:|---:|---:|---:|
| KSL0 | `4 M16 * 2 = 8 b128` | `2 N32 * 4 = 8 b128` | 2 logical b32 | 2 logical b32 | 8 |
| KSL1 | `8 b128` | `8 b128` | 2 logical b32 | 2 logical b32 | 8 |

所以每个 K128 是 **16 条 b128、4 个 logical b32 地址、8 条
`v_wmma_scale_f32_32x16x128_f4`**。8 条 WMMA 来自
`4 M16 * 2 N32`；一条 dedicated F4 WMMA 已经完成硬件
`32×16×128`，不能再按 A8W4 的 N16 指令数翻倍。

scale 的最终机器合并不是“每个 KSL 各自紧邻的两条 dual-load”。第一份
steady body 清楚显示：

```text
L573 / 0x24c4: ds_load_2addr_b32 v[230:231], ...  # SB0/SB1, N32 group 0
L574 / 0x24cc: ds_load_2addr_b32 v[232:233], ...  # SA0, two M32 scale rows
L581 / 0x2504: ds_load_2addr_b32 v[234:235], ...  # SB0/SB1, N32 group 1
L602 / 0x25c0: ds_load_2addr_b32 v[236:237], ...  # SA1, two M32 scale rows

L592 / 0x2554: ... v230, v232                     # one KSL0 WMMA
L625 / 0x2684: ... v231, v236                     # one KSL1 WMMA
```

也就是两条 SB dual-load 分别把同一 N32 group 的 `SB0/SB1` 跨 KSL
装进相邻 VGPR；SA 则是 KSL0、KSL1 各一条 dual-load。两个 KSL 合起来恰好：

```text
32 ds_load_b128 + 4 ds_load_2addr_b32
= 36 DS machine instructions
= 40 logical LDS accesses
```

如果按**最终 ISA 的 issue 时间**而不是按数据归属切分，steady K256 是：

| machine-scheduled region | physical DS issued | WMMA executed | DScnt waits |
|---|---:|---:|---|
| L573-L601：KSL0 setup/前五条 WMMA | `16 b128 + 3 dual` | 5 × KSL0 | `9,4,4,0` |
| L602-L623：预取 KSL1/收尾 KSL0 | `16 b128 + 1 dual` | 3 × KSL0 | — |
| L624-L635：KSL1 compute | 0 | 8 × KSL1 | `6,4,2,0` |

因此不能把共享的 SB dual-load 重复归到两个 KSL，也不能把 K256 的两条
TDM 平均成“每 K128 精确一条”。当前 post-compute ISA 在 KSL1 完成后的
tile tail 才发两条 `kt+2` TDM；tile-ready `s_wait_tensorcnt 2` 位于
KSL0 之前。

每个 steady K256 body 因而是：

| per wave / K256 | 数量或阈值 |
|---|---|
| TDM loads | 2，在该 tile compute 后为 `kt+2` 写回被复用 LDS ring slot |
| DS machine instructions | `32 b128 + 4 dual-b32 = 36` |
| logical LDS accesses | 40 |
| dedicated WMMAs | 16（KSL0 8 + KSL1 8） |
| tensor wait | `s_wait_tensorcnt 2` 一次 |
| KSL0 DS waits | `9 -> 4 -> 4 -> 0` |
| KSL1 DS waits | `6 -> 4 -> 2 -> 0` |
| workgroup barriers | 两组 signal/wait |

这与 2.6 的 full-K 算术一致：
`8 WMMA/K128 * 2 K128/K256 * 28 K256 = 448 WMMA/wave`；
四个 wave 合计 `1792 WMMA/WGP`。

CDNA5 ISA §5.7.1 说明 DScnt 按 LDS **instruction** 增减，`S_WAIT_DSCNT n`
等待 `DScnt<=n`；所以 dual-address 指令在 counter 统计上是一条 machine
instruction，而不是两条。§7.12.6 则规定该 WMMA 是 4-DWORD
VOP3PX2、FP4×FP4、FP32 accumulate，并使用 block-32 scale。

### 3.6 源码 b2 控制流与 compiler steady schedule

【本例】由于 `tile_m=64` 且 `num_buffers=2`，源码选择
`mxfp4_preshuffle_gfx1250_tdm.py:L872-L917` 的 **post-compute** 分支，
不是【仅作排除对照】的旧 A8W4 mid-compute prefetch。以下 `lds_slot` 都是
b2 ring slot，不是 route slot：

#### 3.6.1 高层 b2 ring：先装两槽，再算 26 个 steady，最后 drain

```text
Prologue:
  issue(lds_slot0, kt0): 2 TDM/wave, 8 TDM/WGP
  issue(lds_slot1, kt1): 2 TDM/wave, 8 TDM/WGP
  after both issues: TENSORcnt <= 4/wave  # async completion may make it lower
  no standalone barrier here

Steady: n_steady = 28 - 2 = 26
  for kt=0..25:
    s = kt % 2
    wait_tensorcnt 2       # current kt complete; at most next kt+1's two jobs remain
    WG barrier             # all A/B/SA/SB owner waves now expose current LDS slot
    compute kt             # one K256: KSL0(8 WMMA) + KSL1(8 WMMA)
    WG barrier             # every wave has stopped reading lds_slot[s]
    issue(lds_slot[s], kt+2): 2 TDM/wave, 8 TDM/WGP
    after issue: TENSORcnt <= 4/wave

Drain:
  kt26/slot0: wait_tensorcnt 2; WG barrier; compute; no new TDM
  kt27/slot1: wait_tensorcnt 0; WG barrier; compute; no new TDM
```

Prologue 的源码是两次 `issue(i,i)`；final ISA 为每个 slot 各生成两个互斥
owner path：

| LDS slot / Ktile | waves 0/1: A+SA sites | waves 2/3: B+SB sites |
|---|---|---|
| slot0 / kt0 | L274,L286 / `0x1dc0,0x1e08` | L367,L378 / `0x1f98,0x1fd8` |
| slot1 / kt1 | L401,L410 / `0x2058,0x2098` | L430,L442 / `0x210c,0x2154` |

所以每个 wave 动态执行 4 条 prologue TDM，每个 WGP 执行 16 条；上表的
8 个静态 sites 是互斥 branch sites，不能当成“每 wave 8 条”。四条 issue
之后没有单独 prologue barrier；两个 owner path 的首个 steady header
L560-L571 / `0x2478..0x24bc` 与 L763-L774 / `0x297c..0x29c0`
分别执行 `tensorcnt<=2 + WG barrier`，才使 kt0 对四个 owner waves
全部可见。

26 个 steady iteration 分别回填 kt2..kt27，所以整个 full-K 的 input TDM
动态计数为：

```text
per wave: 4 prologue + 26*2 steady = 56
per WGP : 16 prologue + 26*8 steady = 224
```

Drain 不再发 input TDM。

#### 3.6.2 源码意图与 final ISA 排序不是同一层

`compute_ktile()` 的源码意图是：

```text
load_state(rmem[0], KSL0): SB0 -> SA0 -> B0 -> A0
k_step(KSL0):
  load_state(rmem[1], KSL1): SB1 -> SA1 -> B1 -> A1
  8 * WMMA(KSL0)
sched_barrier
k_step(KSL1):
  8 * WMMA(KSL1)
sched_barrier
```

这里 `rmem[0/1]` 是两个 register-memory state slot，不是 b2 LDS ring。
因为本例 `next_stage_on=0`、`prefetch_kt=None`，源码也没有在
`compute_ktile()` 内发 TDM；TDM 确实在外层 compute 后发出。

最终 machine scheduler 则把 KSL0 的最后 2 条 A load 延后到前两条 WMMA
之间，把 KSL1 的 17 条物理 DS 穿插到 KSL0 最后三条 WMMA 周围，并跨 KSL
合并 SB scale。下面的 10 步描述**最终 ISA 实际顺序**。

#### 3.6.3 单个 steady K256 的实际 10 步 pipeline

以下以 waves 0/1 的 A+SA owner loop 为锚点。区间是半开 Vaddr
`[begin,end)`；“总指令”包含地址/依赖/loop-control 指令，DS/WMMA/TDM
另行列出。DScnt 只写可证明的上界：DS 可异步完成，所以“发 q 条”只能令
上界增加 q，不能声称 counter 动态精确取到该值。

1. **READY current kt，保留 next kt+1 在 flight**
   - ISA L560-L572 / `[0x2478,0x24c4)`，13 条总指令。
   - 关键指令：1 × `s_wait_tensorcnt 2`、1 × WG signal、1 × WG wait；
     无 DS/WMMA/TDM。
   - 同 wave 的 TDM 按 issue 顺序完成；wait 返回时 `TENSORcnt<=2`，所以
     current kt 的两条 job 已完成，最多只剩 next kt+1 的两条。
   - 上一 KSL1 以 `s_wait_dscnt 0` 收尾，因此本步后进入 DS body 时
     `DScnt=0`。

2. **发 KSL0 scale/B0/A0 前缀**
   - ISA L573-L581 / `[0x24c4,0x250c)`，9 条且全部是 DS。
   - 3 × `ds_load_2addr_b32`：两个 SB dual 跨 KSL 返回
     `SB0+SB1`，另一个返回两个 SA0；4 × B0 b128；2 × A0 b128。
   - 从 `DScnt=0` 发 9 条后只能写 `DScnt<=9`。

3. **补齐 B0，继续 A0**
   - ISA L582-L590 / `[0x250c,0x2550)`，9 条总指令：
     4 × B0 b128、4 × A0 b128、1 × `s_wait_alu`。
   - 至此 B0 的 8 条已齐，A0 已发 6/8；本轮累计 17 条 DS，
     所以 `DScnt<=17`，实际值可能更低。

4. **第一批两条 KSL0 WMMA**
   - ISA L591-L594 / `[0x2550,0x2578)`，4 条：
     `wait_dscnt 9 -> WMMA#1 -> wait_dscnt 4 -> WMMA#2`。
   - wait 返回上界依次为 `DScnt<=9`、`DScnt<=4`；无新 DS。

5. **补最后两条 A0，完成 KSL0 前五条 WMMA**
   - ISA L595-L601 / `[0x2578,0x25c0)`，7 条：
     2 × A0 b128、2 × DScnt wait、3 × WMMA。
   - 从上一步 `<=4` 发 2 条，故先有 `DScnt<=6`；随后
     `wait 4 -> WMMA#3 -> wait 0 -> WMMA#4,#5`。
   - `wait 0` 后才可精确写 `DScnt=0`；KSL0 的全部
     `8 B0 + 8 A0` b128 此时都已完成并可供后续 WMMA 安全消费。

6. **开始 KSL1：SA1 + B1 前六条，并穿插 KSL0 WMMA#6**
   - ISA L602-L610 / `[0x25c0,0x260c)`，9 条总指令：
     1 × SA1 dual、6 × B1 b128、1 × `s_wait_alu`、1 × KSL0 WMMA。
   - 新发 7 条 DS，所以 `DScnt<=7`。

7. **补完 B1，发 A1 前六条，并穿插 KSL0 WMMA#7**
   - ISA L611-L620 / `[0x260c,0x2660)`，10 条总指令：
     2 × B1 b128、6 × A1 b128、1 × `s_wait_alu`、1 × KSL0 WMMA。
   - 在前一步上界上再发 8 条，故 `DScnt<=15`。

8. **补完 A1，以 KSL0 WMMA#8 收尾**
   - ISA L621-L623 / `[0x2660,0x2680)`，3 条：
     2 × A1 b128、1 × KSL0 WMMA。
   - KSL1 这一波实际发出的机器 DS 合计
     `1 dual + 8 B1 + 8 A1 = 17`，故 wait 前 `DScnt<=17`。
     KSL1 的另外两个 SB1 logical values 已由步骤 2 的跨 KSL dual-load
     提前返回。

9. **四批完成 8 条 KSL1 WMMA**
   - ISA L624-L635 / `[0x2680,0x2710)`，12 条：
     4 × DScnt wait、8 × WMMA。
   - 精确顺序是
     `wait6 -> 2 WMMA -> wait4 -> 2 -> wait2 -> 2 -> wait0 -> 2`。
   - 返回上界依次 `<=6, <=4, <=2, =0`；至此本 K256 共执行
     `8 KSL0 + 8 KSL1 = 16` 条 dedicated F4 WMMA。

10. **REUSE barrier，回填同一 LDS slot 为 kt+2**
    - ISA L636-L658 / `[0x2710,0x2794)`，23 条总指令：
      1 × WG signal、1 × WG wait、2 × `tensor_load_to_lds`、
      1 × loop backedge 和 18 条 descriptor/address/control 指令。
    - `DScnt` 保持 0。barrier 保证四个 wave 都不再读
      `lds_slot[kt%2]`，然后当前 wave 发自己负责的两条 kt+2 TDM。
    - header 后原有 `TENSORcnt<=2`，再发两条只可推出
      `TENSORcnt<=4`；下一 iteration 的步骤 1 再收敛到 `<=2`。

十步总数严格相加为：

```text
13+9+9+4+7+9+10+3+12+23 = 99 static instructions
DS    = 32 b128 + 4 dual-b32 = 36
WMMA  = 2+3+1+1+1+8 = 16
TDM   = 2
DS waits = 2+2+4 = 8
WG barriers = 2 signal/wait pairs
```

#### 3.6.4 两个 owner branch、共同 compute body 与差异

compiler 为两组 TDM owners 生成两个互斥 steady loop；每个 wave 只执行其中
一个：

| dynamic path | ISA range | header / backedge | trip count |
|---|---|---|---:|
| A+SA owner waves 0/1 | L560-L658 | `0x2478 / 0x2790` | 26 |
| B+SB owner waves 2/3 | L763-L861 | `0x297c / 0x2c90` | 26 |

第二份 10 步的 ISA line ranges 分别是
L763-L775、L776-L784、L785-L793、L794-L797、L798-L804、
L805-L813、L814-L823、L824-L826、L827-L838、L839-L861；
每步计数和 DScnt 阈值与第一份相同。差异只在 owner payload 与 descriptor
推进：

- waves 0/1 每轮发 A+SA，waves 2/3 每轮发 B+SB；
- 两边的 TDM bytes 不同，见 3.4，但共同 compute body 都是
  `32 b128 + 4 dual + 16 WMMA + 8 DS waits`；
- A+SA branch 退出后由 L659 转到第二个 owner test，该 test 把 waves 0/1
  送往共同 drain、只让 waves 2/3 进入 B+SB setup；B+SB branch 从 L861
  直接落入共同 drain。

因此不能把“两份静态 loop”相加成单 wave 动态计数，更不存在四个 wave
各自再执行四份 compute body。第一份 body 的关键 anchors 为：

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

#### 3.6.5 Drain：为什么 kt26 等 2、kt27 等 0

steady kt24 的 tail 已发 kt26，kt25 的 tail 已发 kt27；每个 Ktile 对当前
wave 都是两条 TDM，而且同 wave TDM 有序。因此：

- kt26 的 `s_wait_tensorcnt 2` 允许较新的 kt27 两条仍 outstanding，却保证
  较老的 kt26 两条已完成；随后 WG barrier 汇合四类 owner；
- kt26 不再 issue 新 TDM，所以到 kt27 必须 `s_wait_tensorcnt 0`，把最后
  两条完全 drain 后才能读 slot1。

两个 drain body 各保留 16 条 WMMA，但不再含 tensor load：

| tile | ready wait | physical body | WMMA range | DS wait thresholds |
|---|---|---|---|---|
| kt26 | L863 / `0x2c94`: `tensorcnt 2` | `32 b128 + 4 dual`, 0 TDM | L889-L934 / `0x2d44..0x2f0b` | `9,4,4,8,4,2,0` |
| kt27 | L935 / `0x2f0c`: `tensorcnt 0` | `32 b128 + 4 dual`, 0 TDM | L964-L1005 / `0x2fcc..0x317f` | `9,4,4,8,4,2,0` |

Drain 少一个 DScnt wait 且中段使用 `wait 8`，是“不含 tail TDM/loop
backedge”的 final body 中实际观察到的另一静态排布；它不改变每 tile 的
40 个 logical LDS accesses 或 16 条 WMMA。

本节没有给任何 wait “平均 cycles”。本文的 pipeline 重建只使用当前源码与
final ISA；本次没有从匹配的 ATT wave sample 重新计算或引用 wait latency，
因此本节不声称任何本例实测 wait latency，也不移植历史 A8W4 ATT 平均值。

本例 final ISA 的静态 site count 是：

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

【本例】`--no-bias` 使 `_b1=None`、symbol 无 `_bias` 后缀，最终 ISA 也没有
`global_load*` bias 指令。A4W4 又使：

```text
_is_fp4    = true
_fuse_quant = (not _is_fp4) and (_b1 is None) = false
```

所以本例 GEMM1 必定先输出 BF16 `y`，再由独立 A4 quant/preshuffle kernel
准备 GEMM2；【仅作排除对照】no-bias 并不会打开 A8W4 才有的 fused FP8-qout
epilogue。

本例 `--swiglu-limit` 未显式传参，沿 CLI 默认值为 7.0。虽然
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

【本例】output store 的逻辑 view 是：

```text
LDS source  : BF16 [64,128]
GM base row : blk_m
GM base col : blk_n/2
GM stride   : raw_N/2 = 3072
outer bound : mn_oob = 32
```

因此一个有效 WGP 实际写 `BF16[32,128]`。GEMM1 的静态输出 tensor 是
`y[1,9216,3072]`，但只有 route rows 被后续消费。

【本例】随后 `grouped_moe_gfx1250.py:L660-L697` 执行：

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

这才是本例 A4W4 的完整软件交接：GEMM1 内部终点是 BF16 gated-SiLU，
GEMM2 的 A4 输入由独立 quant/preshuffle 阶段产生。

**静态复核结果。** 本地 Python checker 直接解析
`19_gpu_module_to_binary.mlir` 的内嵌 ELF，并将 opcode 顺序与
`21_final_isa.s` 行号配对；结果为：symbol 入口 `0x1900`、function size
`0x27e4`、`reqd_workgroup_size=(128,1,1)`、wavefront 32、
LDS 89088 B、4 waves/WGP，且上述 64 WMMA、12 tensor-load、1 tensor-store、
两个 26-trip backedge 与所有列出的 Vaddr 一致。这里没有生成额外脚本文件。
