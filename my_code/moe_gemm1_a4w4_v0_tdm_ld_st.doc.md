# gfx1250 `moe_gemm1_a4w4_v0` 的 TDM Load/Store 数据 Pattern

<!-- markdownlint-disable MD013 MD024 MD033 MD060 -->

本文独立分析当前工作区中的
`my_code/moe_gemm1_a4w4_v0.s`。目标 symbol 是：

```text
a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4
```

当前 `.s` 共 1660 行，固定资源为 wave32、128 threads/WG、
`89088 B = 0x15c00` LDS 和 184-byte kernarg。本文的 ISA 行号全部针对这份
1660 行版本重新核对；旧 1827 行 final ISA 只用于发现可能需要检查的问题，
不作为行号、控制流或计数依据。

本文只把以下指令称为 TDM：

- `tensor_load_to_lds`：Global Memory → LDS；
- `tensor_store_from_lds`：LDS → Global Memory；
- `s_wait_tensorcnt`：等待本 wave 的异步 TDM op。

`ds_load_*`、`ds_store_*` 属于 LDS 指令；WMMA 只作为 TDM
producer-consumer 的背景。本文没有运行 GPU。

`.s` 与 exact-ISA runner 是本文 specialization 的权威；production FlyDSL
源码用于解释它的 lowering 和数据 layout。本文不额外假定当前 CSV/env
selector 在无 override 时一定选择这份 t64 artifact。

证据标签如下：

- **【当前 ISA 事实】**：直接来自 `moe_gemm1_a4w4_v0.s`；
- **【生产源码事实】**：来自当前 FlyDSL kernel、quant/preshuffle、shuffle
  或 grouped-MoE 调用路径；
- **【runner/ABI 事实】**：来自 exact-ISA runner 和 C++ launcher；
- **【硬件文档事实】**：来自本地 CDNA5 ISA 或 MI400 Shader Programming
  Guide；
- **【推导/枚举】**：由上述事实做地址展开、集合计数或只读 Python 枚举；
- **【未知边界】**：现有静态材料不能唯一确定的运行时/微架构行为。

区间统一写成左闭右开 `[begin,end)`。本文严格区分：

1. **nominal descriptor payload**：`tile_dim* × data_size` 描述的字节数，
   包括会因 OOB 而补零/丢弃的位置；
2. **in-range Global bytes**：descriptor 坐标中通过 `tensor_dim*` OOB
   检查的地址字节；
3. **application-useful bytes**：属于 32 条有效 routed rows 的数学输入/输出；
4. **LDS allocation footprint**：包含 A padding 和双槽 ring 的地址空间；
5. **HBM transaction bytes**：必须由 PMC/trace 测量，不能由前四项替代。

## 1. 结论摘要

### 1.1 固定几何

```text
user workload:          experts=96, tokens=512, topk=6
GEMM1 raw shape:        M-capacity 9216, Nraw 6144, K 7168
one active expert tile: M64（32 valid + 32 padding）× Nraw256
one compute wave:       M64 × Nraw64 -> SiLU output M64 × Nout32
one WG:                 M64 × Nraw256 -> SiLU output M64 × Nout128
K body:                 K256 = 2 × K128
grid/block/cluster:     (3456,1,1) / (128,1,1) / (1,1,1)
active/tail WG:         2304 / 1152
```

### 1.2 当前 `_wpt4` 的关键差异

**【生产源码事实 + 当前 ISA 事实】** `wpt4` 不是“一种 tensor 交给一个
specialist wave”。四个 wave **每个都发 A、B、ScaleA、ScaleB 四条
descriptor**，但各发该 tensor 的四分之一：

| wave `w` | A producer | B producer | ScaleA producer | ScaleB producer |
|---:|---|---|---|---|
| 0 | M rows `[0,16)` | N rows `[0,64)` | KSL0、M rows `[0,32)` | N rows `[0,64)` |
| 1 | M rows `[16,32)` | N rows `[64,128)` | KSL0、M rows `[32,64)` | N rows `[64,128)` |
| 2 | M rows `[32,48)`，本例全 OOB | N rows `[128,192)` | KSL1、M rows `[0,32)` | N rows `[128,192)` |
| 3 | M rows `[48,64)`，本例全 OOB | N rows `[192,256)` | KSL1、M rows `[32,64)` | N rows `[192,256)` |

每 wave、每 K256 的 nominal payload 是：

```text
A 0x800 + B 0x2000 + ScaleA 0x80 + ScaleB 0x200
= 0x2a80 = 10880 B
```

每 WG、每 K256：

```text
nominal descriptor payload = 4 * 0x2a80 = 0xaa00 = 43520 B
in-range Global bytes       = 0x9a00 = 39424 B
application-useful bytes    = 0x9900 = 39168 B
LDS stage footprint         = 0xae00 = 44544 B
```

差额有明确来源：

- nominal → in-range 少 `4096 B`：A 的 wave2/3 descriptor 全 OOB；
- in-range → useful 少 `256 B`：ScaleA 仍读取 32 条 padding rows 的 scale；
- nominal payload → LDS footprint 多 `1024 B`：A 的每行 `128 B + 16 B pad`。

### 1.3 Load/store 总体结论

1. 四类 load 都是 2D D#，`count=1`、gather off、descriptor iteration
   off；K 推进由 scalar 地址更新完成。
2. A 是未 preshuffle 的 packed row-major MXFP4，TDM 每行取 128 B，并在
   LDS 每 128 B 插 16 B padding。
3. B 是 GUGU 后的 `16 rows × 16 packed bytes` tile preshuffle；一条
   per-wave descriptor 取 4 个 N16 super-row，每个连续 2048 B。
4. ScaleA 是 M64/K128/WMMA-row preshuffle，四个 wave 沿 descriptor
   dim0 切分；ScaleB 是 GUGU 后的 n32k4 preshuffle，四个 wave 沿 N 切分。
5. exact launch 没有 cluster：所有 load 的 `workgroup_mask=0`、
   `early_timeout=0`，无 cluster barrier；只有 `s_barrier_* -1` WG barrier。
6. 所有 TDM instruction 都没有显式 TH modifier，故 `TH=0=RT`。
7. 输入使用两个 `0xae00` 槽。q0/q1 先填两槽，26 次 steady body 消费 q0..q25
   并回填 q2..q27，最后 drain q26/q27。
8. output 先在 LDS `[0,0x4000)` 拼成 BF16 `[64,128]`。随后四个 wave
   各发一条 M16×N128 store descriptor；本例只有前两条产生有效 Global
   store，后两条因 `tensor_dim1=0` 全部丢弃。

## 2. Workload、路由与物理布局

### 2.1 代表性输入

`my_code/run_gemm_a4w4.sh` 的代表性测试是：

```text
experts   = 96
tokens    = 512
topk      = 6
model_dim = K = 7168
inter_dim = 3072
activation = SiLU
bias       = none
```

balanced routing 给出：

```text
routes          = 512 * 6 = 3072
count[e]        = 3072 / 96 = 32, e=0..95
aligned rows/e  = align_up(32,64) = 64
```

contiguous route map 的本例值为：

```text
starts[e] = 64e
psum[e]   = 64e + 32               # valid exclusive end
```

kernel 的 96-entry upper-bound 查找找到第一个
`psum[e] > blk_m` 的 expert。因此对 `m_tile=e<96`：

```text
blk_m  = 64e
expert = e
mn_oob = psum[e] - blk_m = 32
```

这里的 32 是每个 expert tile 的 valid M extent，不是 tile-M 64，也不是
route atomic 的某个固定 slot。

### 2.2 五个数据面的 shape

令：

```text
P = K/2  = 3584 = 0x0e00 B/row       # two FP4 values per byte
S = K/32 =  224 = 0x00e0 B/row       # one E8M0 byte per K32
```

| tensor | 数学/逻辑 shape | runtime storage / physical view | 总 allocation |
|---|---|---|---:|
| A payload | `[1,9216,7168]` FP4 | uint8 `[1,9216,3584]`，packed row-major | `33030144 B = 0x1f80000` |
| B payload | `[96,6144,7168]` FP4 | uint8 `[96,384,112,2,16,16]` preshuffle | `2113929216 B = 0x7e000000` |
| ScaleA | `[1,9216,224]` E8M0 bytes | byte order `[144,56,4,16,4]`；ABI view int32 `[1,2304,224]` | `2064384 B = 0x1f8000` |
| ScaleB | `[96,6144,224]` E8M0 bytes | uint8 `[96,192,56,32,4]`；ABI flatten int32 | `132120576 B = 0x07e00000` |
| output C | `[1,9216,3072]` BF16 | contiguous row-major | `56623104 B = 0x03600000` |

只有 3072 routed rows 是 application-valid。前 6144 capacity rows 是
96 个 `32 valid + 32 padding` tile；`[6144,9216)` 是 48 个 sentinel M64
tiles。

### 2.3 Gate/up row order

**【生产源码事实】** W1 在 preshuffle 前从 GGUU：

```text
[g0,g1,...,g3071,u0,u1,...,u3071]
```

变成 GUGU：

```text
[g0,u0,g1,u1,...,g3071,u3071]
```

所以 raw GEMM `Nraw=6144` 的相邻两列是一个 `(gate,up)` pair，SiLU 后
输出宽度减半：

```text
Nout = Nraw/2 = 3072
one raw N256 WG -> one activated N128 output tile
one raw N64 wave -> one activated N32 output stripe
```

### 2.4 A、B、ScaleA、ScaleB 的线性 offset

以下 offset 均相对对应 tensor base，`kb` 是 packed-K byte index，
`ks` 是 K32 scale index。

#### A payload：不做 tile preshuffle

```text
A_off(m,kb) = m*P + kb
```

route quant 只写 valid routed rows；本 kernel 用 A descriptor 的
`tensor_dim1` 令 padding rows 从 Global OOB 补零。

#### B payload：GUGU + 16×16 packed-byte preshuffle

```text
B_off(e,n,kb)
  = e*(6144*P)
  + floor(n/16)*(16*P)
  + floor(kb/16)*256
  + (n mod 16)*16
  + (kb mod 16)
```

因此：

```text
one N16 super-row stride = 16*P = 57344 B = 0xe000
one expert stride        = 6144*P = 22020096 B = 0x1500000
```

物理连续的 256-B 基元是 16 rows × 16 packed bytes；一个 K256 对每个
N16 super-row 连续占 8 个基元，即 `8*256=2048 B`。

#### ScaleA：M64/K128/WMMA-row preshuffle

令：

```text
mtile = floor(m/64)
wm    = floor((m mod 64)/16)
lane  = m mod 16
kd    = floor(ks/4)        # one dword contains 4 E8M0 bytes
b     = ks mod 4
```

则：

```text
SA_off(m,ks)
  = mtile*0x3800
  + kd*0x100
  + wm*0x40
  + lane*4
  + b
```

等价 physical shape：

```text
[M64 tile][K128 scale-dword][M16/WMMA row][lane16][4 scale bytes]
= [144][56][4][16][4]
```

一个 M64 tile stride 是：

```text
56 * 4 * 16 * 4 = 14336 B = 0x3800
```

#### ScaleB：GUGU + n32k4

```text
SB_off(e,n,ks)
  = e*(6144*S)
  + floor(n/32)*(32*S)
  + floor(ks/4)*128
  + (n mod 32)*4
  + (ks mod 4)
```

所以：

```text
one N32 super-row stride = 32*S   = 7168 B   = 0x1c00
one expert stride        = 6144*S = 1376256 B = 0x150000
```

## 3. Grid 3456、16-M-tile swizzle 与 tail

### 3.1 静态 launch

```text
static M tiles = 9216/64  = 144
raw N tiles    = 6144/256 = 24
grid.x         = 144*24   = 3456
block.x        = 4*32     = 128
cluster        = (1,1,1)
```

生产源码使用 16-M-tile group swizzle。因为 144 可被 16 整除，本例九组
都是完整 group：

```text
g  = bid_x // 384
in = bid_x % 384
ml = in % 16
n  = in // 16

m_tile = 16*g + ml
n_tile = n
blk_m  = 64*m_tile
blk_n  = 256*n_tile

bid_x = 384*g + 16*n + ml
```

一组内 M tile 变化最快：对固定 `n`，依次发出 16 个 M64 WG。

### 3.2 Active 与 sentinel

| 项 | M tile | N tile | WG 数 |
|---|---|---|---:|
| active | `0..95` | `0..23` | `96*24 = 2304` |
| sentinel tail | `96..143` | `0..23` | `48*24 = 1152` |
| launch 总计 | `0..143` | `0..23` | `3456` |

**【当前 ISA 事实】** `m_tile_map` upper-bound 位于 L150-L222；L224-L227
检查 `expert<96`。tail 得到 `expert=96` 后直接跳到 L1504-L1505 的
`.LBB0_5: s_endpgm`：

- 不加载 A/B/ScaleA/ScaleB pointer；
- 不发 TDM；
- 不执行 LDS/WMMA/SILU/store；
- 但 code object 的 `0x15c00` fixed LDS 是 per-WG dispatch 资源，tail WG
  在驻留时仍按同一 kernel resource contract 分配。

只读枚举得到 3456 个 `(m_tile,n_tile)` pair 全部唯一，无 hole、无
collision；active `bid_x=[0,2304)`，tail `bid_x=[2304,3456)`。

## 4. 184-byte ABI

### 4.1 Metadata 结构

**【当前 ISA 事实】**

- `.amdhsa_kernarg_size 184`：L1511；
- kernarg pointer 在 `s[0:1]`；
- preload 2 dword：L1512-L1518，因此 `arg_c` pointer 在 `s[2:3]`；
- metadata 的 17 rows：L1576-L1634；
- 实际入口 load offset：`0xa4`（L10）、`0x70`（L117）、
  `0x28/0x38`（L327-L328）、`0x60`（L423）。

FlyDSL signature 有 13 个逻辑参数。四个 `fx.Tensor` 各 lower 为
`pointer + packed layout companion`，其余 pointer 没有 companion。

### 4.2 代表性 production launch 的逐字段值

表中 stride 对 Tensor companion 都是 **该 Tensor element** 为单位；
pointer 的 64-bit runtime 地址静态未知。“本例值”采用
代表性脚本的 user-level scalar 和 grouped-MoE 对该 t64 specialization 的
production-style argument wiring。

| 参数/字段 | byte offset | size | production 本例值 | 是否被当前 ISA 使用 |
|---|---:|---:|---|---|
| `arg_c` pointer | `0x00` | 8 | BF16 output base；preload `s[2:3]` | 是 |
| `arg_c.$layout`: size0/1/2 | `0x08/0x0c/0x10` | 12 | `1,9216,3072` | 否 |
| `arg_c.$layout`: stride0/1 | `0x14/0x1c` | 16 | `28311552 (0x1b00000), 3072 (0xc00)` BF16 elements | 否 |
| ABI padding | `0x24` | 4 | 未使用 | 否 |
| `arg_a` | `0x28` | 8 | packed A uint8 base | 是，L327 |
| `arg_b` | `0x30` | 8 | preshuffled B uint8 base | 是，L327 |
| `arg_scale_a` pointer | `0x38` | 8 | ScaleA base，随后按 int32/dword 解释 | 是，L328 |
| `arg_scale_a.$layout`: size0/1/2 | `0x40/0x44/0x48` | 12 | `1,2304,224` int32 | 否 |
| `arg_scale_a.$layout`: stride0/1 | `0x4c/0x54` | 16 | `516096 (0x7e000), 224 (0xe0)` int32 elements | 否 |
| ABI padding | `0x5c` | 4 | 未使用 | 否 |
| `arg_scale_b` pointer | `0x60` | 8 | flattened ScaleB int32 base | 是，L423 |
| `arg_scale_b.$layout`: size0 | `0x68` | 4 | `33030144 (0x1f80000)` int32 | 否 |
| ABI padding | `0x6c` | 4 | 未使用 | 否 |
| `arg_m_tile_map` | `0x70` | 8 | int32 `psum[96]` | 是，L117 |
| `arg_bias` | `0x78` | 8 | no-bias path 的 dummy，生产调用 alias A | 否 |
| `arg_quant_scale` pointer | `0x80` | 8 | qout0 path 的 dummy，生产调用 alias C | 否 |
| `arg_quant_scale.$layout` size0/1/2 | `0x88/0x8c/0x90` | 12 | `1,9216,3072` | 否 |
| `arg_quant_scale.$layout` stride0/1 | `0x94/0x9c` | 16 | `28311552,3072` | 否 |
| `i32_m` | `0xa4` | 4 | `9216 = 0x2400` rows | 是，L10 |
| `i32_n` | `0xa8` | 4 | `6144 = 0x1800` raw gate/up columns | 是，L10 |
| `f32_swiglu_limit` | `0xac` | 4 | `7.0f`, bits `0x40e00000` | 是，L10；SiLU 也执行 clamp |
| `f32_situ_beta` | `0xb0` | 4 | `4.0f`, bits `0x40800000` | act1 编译掉 |
| `f32_situ_linear_beta` | `0xb4` | 4 | `25.0f`, bits `0x41c80000` | act1 编译掉 |

metadata rows 的 payload 合计 172 B，加 `0x24/0x5c/0x6c` 三个 4-B
padding，正好是 184 B。最后字段结束于 `0xb8`。

### 4.3 Exact-ISA standalone runner 的同一 ABI

`gemm_batch_isa_runner.py` 与 production launch 使用完全相同的 184-B
offset/size contract，但对 **当前 ISA 不读取** 的 companion/dummy 字段允许
填入不同、仍然合法的值。两种上下文必须区分：

| 字段 | production pipeline | exact-ISA standalone runner | 当前 ISA |
|---|---|---|---|
| ScaleA companion | int32 view：shape `[1,2304,224]`，stride `[516096,224]` | underlying uint8 view：shape `[1,2304,896]`，stride `[2064384,896]` | companion 不读取；pointer相同 byte storage |
| ScaleB companion `size0@0x68` | flattened int32 length `33030144` | `96` | 不读取 |
| `arg_bias@0x78` | no-bias dummy alias A | 独立 zero float32 `[6144]` | 不读取 |
| `arg_quant_scale@0x80` | qout0 dummy alias C | 独立 zero float32 `[1]` | 不读取 |
| quant companion | C-like `[1,9216,3072]`、stride `[28311552,3072]` | `[1,1,1]`、stride `[1,1]` | 不读取 |
| `f32_swiglu_limit@0xac` | `7.0f` | 默认 `3.0e38f`，bits `0x7f61b1e6`；可由 runner env 覆盖 | 读取，故两种运行的数值 clamp 不同 |
| SiTU beta 两项 | `4.0f,25.0f` | packer 默认 `1.0f,1.0f` | act1 编译掉 |

standalone runner 还量化整个 9216-row test allocation，使 padding/tail storage
有确定测试数据；`m_tile_map` 仍只允许每个 active tile 的前 32 行有效，
tail WG 仍在入口退出。production route-quant 则只把 routed rows 定义为
application-valid。两者不改变本文的 descriptor、OOB、grid 或 LDS pattern。

### 4.4 ABI 中没有的状态

以下不是 kernarg：

- `grid=(3456,1,1)`、`block=(128,1,1)`、`cluster=(1,1,1)`；
- 89,088-B fixed LDS；
- workgroup-id X；
- baked `K=7168`、`E=96`、tile `64×256×256`、`b2`、`wpt4`。

这些由 symbol specialization、kernel descriptor 和 launch API 决定。

## 5. gfx1250 TDM 硬件语义

### 5.1 本文依赖的文档事实

CDNA5 ISA §10.11.1-§10.11.6，printed pp.140-147
（文本抽取 `--- Page 149 ---` 到 `--- Page 156 ---`）规定：

- tensor instruction 不按 lane 执行，忽略 `EXEC`；
- 每条 instruction 完成一次只返回一个 Tensor-Done；
- 同一 wave 的 tensor load/store 相互有序，不同 wave 无序；
- 2D Global 地址为：

```text
global_addr + data_size_bytes * (x + y*tensor_dim0_stride)
```

- `tile_dim*`、`tensor_dim*`、stride 都以 `data_size` element 为单位；
- load OOB 返回零，store OOB 丢弃；
- load 的 padding 只跳过 LDS 地址，不写零；
- store 忽略 padding，没有“反 padding”；
- `workgroup_mask!=0` 的 load 使用 `CLUSTER_LOAD_ASYNC`；
  store 忽略 `workgroup_mask`。

MI400 Shader Programming Guide §4.10.1-§4.10.8，printed pp.197-206
（文本抽取 pages 207-216）给出一致语义，并补充：

- TDM 内部拆成 `GLOBAL_LOAD_ASYNC_TO_LDS`、`CLUSTER_LOAD_ASYNC` 或
  `GLOBAL_STORE_ASYNC_FROM_LDS`；
- direct-copy eligibility 条件见 §4.10.2 printed pp.199-200；
- 每 wave 最多 3 个 tensor op 从 issue 到 XACK 在途，每 SIMD 最多 6 个；
  issue-to-done 的 `TENSORcnt` 是 6 bit，最大 63（§4.10.8 p.206）。

### 5.2 D# group

当前所有 TDM 都只传 Group 0 和 Group 1：

| group | 内容 |
|---|---|
| Group 0，4 SGPR | `count/is_restore/gather`、byte `lds_addr`、57-bit byte `global_addr`、`type` |
| Group 1，8 SGPR | mask/flags、tensor dims、tile dims、strides |
| Group 2/3 | 不传；没有 3D-5D、gather 或 descriptor iteration |

所有 Group 0：

```text
count       = 1
is_restore  = 0
gather      = 0
type        = 2 ("image")
```

Group 0 的 global high dword 被 OR `0x80000000`；高 bit 是 `type=2`
编码的一部分，不是 pointer bit。

### 5.3 公共 flags、mask 与 temporal hint

| op | Group-1 dword0 | `data_size` | mask | pad | iterate | atomic | early timeout | instruction TH |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A load | `0x07100000` | `0` = 1 B | 0 | on | 0 | 0 | 0 | `RT` |
| B load | `0x00000000` | `0` = 1 B | 0 | off | 0 | 0 | 0 | `RT` |
| ScaleA load | `0x00020000` | `2` = 4 B | 0 | off | 0 | 0 | 0 | `RT` |
| ScaleB load | `0x00020000` | `2` = 4 B | 0 | off | 0 | 0 | 0 | `RT` |
| C store | `0x00010000` | `1` = 2 B | 0 | off | 0 | 0 | 0 | `RT` |

A 的 padding 字段实际是：

```text
pad_interval encoding = 4 -> 32 DWORD = 128 B
pad_amount   encoding = 3 ->  4 DWORD =  16 B
```

**【当前 ISA 事实】** 12 条 load 和 1 条 store 都没有 `TH_*` modifier。
CDNA5 ISA Table 10/11（printed pp.35-36）规定 TH code 0 是默认 `RT`，
即 near/far cache 均按 regular temporal 处理。

exact launch 的 `cluster=(1,1,1)`，生产源码令 `a_mcast_mask=0`。因此：

- A 也不是 multicast；
- 所有 load 都走普通 Global→LDS async-copy 路径；
- `early_timeout` 不适用并保持 0；
- 当前 ISA 没有 `s_barrier_* -3`。

### 5.4 `0x7fffffff` 是实际 tensor dim

当前 descriptor 的多数 `tensor_dim0/1` 不是逻辑矩阵 shape，而是
`0x7fffffff`。原因是生产源码向 atom 的对应 OOB extent 传 `None`。

这表示“该轴不由 descriptor 做实际边界 clamp”，不是 tensor 真有
2,147,483,647 个 element。合法性由固定 shape、pointer allocation 和 exact
launch contract 保证。A 与 C 的 outer 轴例外，它们使用 `mn_oob`。

## 6. Descriptor 总表与原始 dword

令：

```text
e      = expert id, 0..95
w      = wave id, 0..3
q      = K256 phase, 0..27
s      = q mod 2
blk_m  = 64e
blk_n  = 256*n_tile
```

### 6.1 Load descriptor 逐字段实际值

| 字段 | A | B | ScaleA | ScaleB |
|---|---|---|---|---|
| `data_size` | 1 B | 1 B | 4 B | 4 B |
| `global_addr` | `A+(blk_m+16w)*0xe00+q*0x80` | `B+e*0x1500000+(blk_n/16+4w)*0xe000+q*0x800` | `SA+e*0x3800+q*0x200+w*0x80` | `SB+e*0x150000+(blk_n/32+2w)*0x1c00+q*0x100` |
| `lds_addr` | `s*0xae00+w*0x900` | `s*0xae00+0x2400+w*0x2000` | `s*0xae00+0xa400+w*0x80` | `s*0xae00+0xa600+w*0x200` |
| `tensor_dim0` | `0x7fffffff` bytes | `0x7fffffff` bytes | `0x7fffffff` dwords | `0x7fffffff` dwords |
| `tensor_dim1` | `max(32-16w,0)` | `0x7fffffff` | `0x7fffffff` | `0x7fffffff` |
| `tile_dim0` | `128` bytes | `2048` bytes | `32` dwords = 128 B | `64` dwords = 256 B |
| `tile_dim1` | `16` | `4` | `1` | `2` |
| `tensor_dim0_stride` | `3584` bytes | `57344` bytes | `3584` dwords = 14336 B | `1792` dwords = 7168 B |
| `tile_dim2/stride1` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| nominal payload/wave | `2048 B` | `8192 B` | `128 B` | `512 B` |
| descriptor iteration | off | off | off | off |
| mask / cluster | 0 / none | 0 / none | 0 / none | 0 / none |

注意 A 的软件 K step 是 `+0x80`，B 是 `+0x800`，ScaleA 是
`+0x200`，ScaleB 是 `+0x100`。它们是 Group 0 `global_addr` 的软件更新，
不是 Group 2 `global_addr_increment`。

### 6.2 本例 Group-1 `s[8]` dword dump

以下按低 dword 到高 dword列出。A 的 `tensor_dim1` 随 wave 变化：

```text
A wave0:
07100000 ffff0000 00207fff 00800000
00000010 00000e00 00000000 00000000

A wave1:
07100000 ffff0000 00107fff 00800000
00000010 00000e00 00000000 00000000

A wave2 / wave3:
07100000 ffff0000 00007fff 00800000
00000010 00000e00 00000000 00000000

B:
00000000 ffff0000 ffff7fff 08007fff
00000004 0000e000 00000000 00000000

ScaleA:
00020000 ffff0000 ffff7fff 00207fff
00000001 00000e00 00000000 00000000

ScaleB:
00020000 ffff0000 ffff7fff 00407fff
00000002 00000700 00000000 00000000
```

当前 ISA 锚点：

- A setup/首次 issue：L327-L377；
- B setup/首次 issue：L379-L404、L511；
- ScaleA setup/首次 issue：L405-L423、L517；
- ScaleB setup/首次 issue：L423-L538；
- q1 的四条 load：L539-L553；
- steady refill：L654-L695。

## 7. A load：row-major M16×K256/packed

### 7.1 Global 二维地址

一条 wave-`w` A descriptor：

```text
y = 0..15
x = 0..127

A_global(w,y,x,q)
  = A
  + (blk_m + 16w)*P
  + q*128
  + y*P
  + x
```

逻辑坐标：

```text
M row          = blk_m + 16w + y
packed-K byte  = 128q + x
logical K      = [256q,256q+256)
```

四个 descriptor 的 nominal tile 合起来是 M64×128 packed bytes =
8192 B。但本例 `mn_oob=32`：

| wave | `tensor_dim1` | nominal | in-range Global | 行为 |
|---:|---:|---:|---:|---|
| 0 | 32 | 2048 B | 2048 B | rows 0..15 正常 load |
| 1 | 16 | 2048 B | 2048 B | rows 16..31 正常 load |
| 2 | 0 | 2048 B | 0 B | rows 32..47 全部 OOB→0 |
| 3 | 0 | 2048 B | 0 B | rows 48..63 全部 OOB→0 |
| **WG** | — | **8192 B** | **4096 B** | M32 valid + M32 zero |

### 7.2 LDS padding

令当前 slot 的 A base 为 `A_s=s*0xae00`：

```text
A_lds(row,kb_in_q) = A_s + row*144 + kb_in_q
row       = 0..63
kb_in_q   = 0..127
```

每行：

```text
[row*144, row*144+128)      data 或 OOB zero
[row*144+128,row*144+144)   16-B padding hole
```

每个 producer wave 拥有一个 `16*144=2304 B=0x900` LDS segment：

| wave | LDS allocation range（slot-relative） | data bytes | pad holes |
|---:|---|---:|---:|
| 0 | `[0x0000,0x0900)` | 2048 | 256 |
| 1 | `[0x0900,0x1200)` | 2048 | 256 |
| 2 | `[0x1200,0x1b00)` | 2048 OOB-zero | 256 |
| 3 | `[0x1b00,0x2400)` | 2048 OOB-zero | 256 |
| **合计** | `[0x0000,0x2400)` | **8192** | **1024** |

只读枚举得到 8192 个 data destination 全唯一，`collisions=0`；
`[0,0x2400)` 内恰有 1024 个 padding holes。

### 7.3 Consumer

生产端按 M16 分给四个 wave；消费端不同：

- 四个 compute wave 都计算同一个 M64；
- 每个 compute wave 都读取完整 A data set 8192 B；
- 每 wave、每 K256 是 16 条 `ds_load_b128`；
- 四个 wave 对 A LDS 的 read 地址集合相同。

因此 A 是“分布式四-wave producer + WG 内全 wave共享 consumer”。

## 8. B load：preshuffled N64×K256/packed

### 8.1 一条 descriptor 的物理 pattern

对 wave `w`：

```text
y = 0..3                       # 4 个 N16 super-row
x = 0..2047

B_global(w,y,x,q)
  = B
  + e*0x1500000
  + (blk_n/16 + 4w)*0xe000
  + q*0x800
  + y*0xe000
  + x
```

把连续 `x` 分解：

```text
x = 256*t + 16*r + u
t = 0..7
r = 0..15
u = 0..15
```

对应逻辑坐标：

```text
N row         = blk_n + 64w + 16y + r
packed-K byte = 128q + 16t + u
```

所以一条 descriptor 的地址集合是 N64×128 packed bytes = 8192 B，但在
Global 物理空间中表现为 4 条 2048-B chunk，chunk stride 为 `0xe000`。

### 8.2 四 wave覆盖

| wave | N rows | chunk 数 | nominal/in-range/useful |
|---:|---|---:|---:|
| 0 | `[blk_n+0,blk_n+64)` | 4 | 8192 B |
| 1 | `[blk_n+64,blk_n+128)` | 4 | 8192 B |
| 2 | `[blk_n+128,blk_n+192)` | 4 | 8192 B |
| 3 | `[blk_n+192,blk_n+256)` | 4 | 8192 B |
| **WG** | `[blk_n,blk_n+256)` | **16** | **32768 B** |

N=6144 是 N256 的整数倍，active expert 的 B storage 完整，因此 B
descriptor 不需要 runtime OOB bound。

### 8.3 LDS producer-consumer 对齐

当前 slot 的 B region：

```text
B_s = s*0xae00 + 0x2400
```

| wave | producer/consumer LDS range | bytes |
|---:|---|---:|
| 0 | `[B_s+0x0000,B_s+0x2000)` | 8192 |
| 1 | `[B_s+0x2000,B_s+0x4000)` | 8192 |
| 2 | `[B_s+0x4000,B_s+0x6000)` | 8192 |
| 3 | `[B_s+0x6000,B_s+0x8000)` | 8192 |

每个 compute wave 只读取自己 N64 的 8192 B B segment，使用 16 条
`ds_load_b128`。B 是 producer wave 与 consumer wave 一一对齐的分片。

## 9. ScaleA load：沿物理 KSL/M-row 交错切分

### 9.1 Descriptor 地址

ScaleA atom 以 int32 为 element：

```text
SA_global_dword(w,x,q)
  = SA_dword_base
  + e*3584
  + q*128
  + w*32
  + x

x = 0..31
```

byte 形式就是：

```text
SA + e*0x3800 + q*0x200 + w*0x80 + 4*x
```

一条 descriptor 是 32 dword = 128 B；四条连续覆盖一个 K256 的
512-B physical block。

### 9.2 物理块到逻辑 scale 的映射

令 `z=32w+x`：

```text
ksl        = floor(z/64)           # 0 或 1
rem        = z mod 64
wm         = floor(rem/16)         # M16 id 0..3
row        = 16*wm + (rem mod 16)
scale      = 8q + 4*ksl + byte_in_dword
```

| producer wave | K128 half | M rows | descriptor bytes | application-useful |
|---:|---:|---|---:|---:|
| 0 | KSL0 / scales `[8q,8q+4)` | `[0,32)` | 128 | 128 |
| 1 | KSL0 / scales `[8q,8q+4)` | `[32,64)` | 128 | 0 |
| 2 | KSL1 / scales `[8q+4,8q+8)` | `[0,32)` | 128 | 128 |
| 3 | KSL1 / scales `[8q+4,8q+8)` | `[32,64)` | 128 | 0 |

与 A 不同，ScaleA descriptor 没有 outer OOB clamp，所以四条都从 Global
读取。生产 route quant 不把 padding scale 当 application-valid 数据；这里
把 wave1/3 的 256 B 单列为“in-range 但非 application-useful”。

### 9.3 LDS 与 consumer

```text
SA_s = s*0xae00 + 0xa400
wave w destination = [SA_s+w*0x80, SA_s+(w+1)*0x80)
```

四段无 hole、无 collision 地覆盖 `[SA_s,SA_s+0x200)`。

所有 compute wave 都需要 M64 的两个 K128 scale plane，因此每个 compute
wave 读取完整 512-B ScaleA region。编译器把 8 个逻辑
`ds_load_b32` 地址中的 ScaleA/ScaleB 对合并成
`ds_load_2addr_b32`。

## 10. ScaleB load：n32k4 N64×K256

### 10.1 Global pattern

ScaleB atom也以 int32 为 element。对 wave `w`：

```text
y = 0..1
x = 0..63 dword

SB_global(w,y,x,q)
  = SB
  + e*0x150000
  + (blk_n/32 + 2w)*0x1c00
  + q*0x100
  + y*0x1c00
  + 4*x
```

把一个 256-B inner chunk 解码：

```text
K-scale group  = floor(x/32)       # 0/1，各含 4 scales
N row in N32   = x mod 32
scale index    = 8q + 4*group + byte_in_dword
```

一条 descriptor 覆盖两个 N32 super-row：

```text
2 * 64 dword * 4 B = 512 B
```

四个 wave 合计：

| wave | N rows | bytes |
|---:|---|---:|
| 0 | `[blk_n+0,blk_n+64)` | 512 |
| 1 | `[blk_n+64,blk_n+128)` | 512 |
| 2 | `[blk_n+128,blk_n+192)` | 512 |
| 3 | `[blk_n+192,blk_n+256)` | 512 |
| **WG** | `[blk_n,blk_n+256)` | **2048** |

### 10.2 LDS 与 consumer

```text
SB_s = s*0xae00 + 0xa600
wave w destination = [SB_s+w*0x200, SB_s+(w+1)*0x200)
```

每个 compute wave 读取与自己 N64 对应的 512-B ScaleB segment。四个
consumer segment 无重叠，合计覆盖完整 `0x800` region。

## 11. 双槽 LDS ring 与 K pipeline

### 11.1 完整 LDS 地址图

| region | slot0 | slot1 | 每槽 nominal payload | 每槽 footprint |
|---|---|---|---:|---:|
| A | `[0x0000,0x2400)` | `[0xae00,0xd200)` | `0x2000` | `0x2400` |
| B | `[0x2400,0xa400)` | `[0xd200,0x15200)` | `0x8000` | `0x8000` |
| ScaleA | `[0xa400,0xa600)` | `[0x15200,0x15400)` | `0x0200` | `0x0200` |
| ScaleB | `[0xa600,0xae00)` | `[0x15400,0x15c00)` | `0x0800` | `0x0800` |

```text
slot0: [0x00000,0x0ae00)
slot1: [0x0ae00,0x15c00)
fixed LDS end: 0x15c00 = 89088 B
```

四个 region 没有 inter-region gap；A 内部有 64×16 B 的 row padding。
output epilogue 在 input TDM 完全 drain 后复用 arena 开头
`[0,0x4000)`，不另增 LDS allocation。

### 11.2 Software K iteration

```text
Q = K/256 = 7168/256 = 28
q = 0..27
slot(q) = q mod 2
```

28 个 phase 在 K 方向恰好首尾相接：

```text
A/B packed-K: q -> [128q,128q+128), union [0,3584)
SA/SB scale:   q -> [8q,8q+8), union [0,224)
```

完整 K 的 logical-coordinate 枚举得到：

```text
A valid = 32*3584  = 114688 unique bytes
B       = 256*3584 = 917504 unique bytes
ScaleA  = 64*224   = 14336 unique bytes（其中 7168 application-useful）
ScaleB  = 256*224  = 57344 unique bytes
跨 q holes=0，collisions=0
```

控制流：

```text
prologue:
  issue q0 -> slot0       # 4 TDM/wave
  issue q1 -> slot1       # 4 TDM/wave

steady, kt=0..25:
  wait_tensorcnt 4
  WG barrier
  consume q=kt from slot kt%2
  WG barrier
  issue q=kt+2 into released slot   # 4 TDM/wave

drain:
  q26/slot0: wait_tensorcnt 4; barrier; compute
  q27/slot1: wait_tensorcnt 0; barrier; compute
```

当前 ISA 锚点：

- prologue 8 条静态 site：L377、L511、L517、L538-L553；
- 26-trip steady loop：L578-L696，backedge L696；
- q26 drain：L698-L767；
- q27 drain：L768-L838；
- epilogue 前再次 `s_wait_tensorcnt 0`：L910。

### 11.3 Hardware tile traversal 与 descriptor iteration

每条 descriptor 内，TDM hardware 仍按 `tile_dim1 × tile_dim0` 遍历
二维 tile。这是 descriptor 的正常二维地址生成，不等于
`iterate_enable`。

当前所有 Group-1 `iterate_enable=0`：

- 无 Group 2；
- 无 `global_addr_increment`；
- 无 `lds_addr_increment`；
- 无 `iterate_count`。

所以本文的 28 次 K phase 是 software loop；每条 descriptor 的二维
`x/y` 展开是 hardware tile traversal；不存在第三层 descriptor iteration。

### 11.4 Wait 与 barrier

一条 K tile 对每 wave 有 4 个 TDM job。双槽 prologue 在程序顺序上发 8 条，
`s_wait_tensorcnt 4` 利用同-wave in-order 保证较老 slot 的四条已经完成，
同时允许下一槽至多四条尚未 Tensor-Done。

这不与硬件“每 wave 最多 3 条从 issue 到 XACK 在途”矛盾：

- 第四条及后续 instruction 必要时由硬件 stall，直到较早 descriptor XACK；
- XACK 之后 descriptor 不再占 3-entry issue limit，但 Tensor-Done 之前仍计入
  6-bit `TENSORcnt`；
- 静态 ISA 不能给出该 backpressure 的实际 cycle 数。

当前 barrier 全部是：

```text
s_barrier_signal -1
s_barrier_wait   -1
```

它们是 WG barrier。没有 cluster barrier `-3`。pre-compute barrier 汇合四个
producer wave；post-compute barrier 保证四个 consumer wave 都不再读取旧
slot 后才覆盖它。

## 12. WMMA consumer 背景

每个 K256、每个 compute wave：

| 项 | 动态数量 | LDS unique data set |
|---|---:|---:|
| A `ds_load_b128` | 16 | 完整 A data 8192 B |
| B `ds_load_b128` | 16 | 自己 N64 的 B 8192 B |
| ScaleA logical b32 addresses | 4 | 完整 ScaleA 512 B |
| ScaleB logical b32 addresses | 4 | 自己 N64 的 ScaleB 512 B |
| `ds_load_2addr_b32` machine instruction | 4 | 合并上述 8 个 b32 地址 |
| `v_wmma_scale_f32_32x16x128_f4` | 16 | 2 K128 × 8 |

只读逐 byte 枚举 consumer 地址得到，对每个 compute wave：

```text
A:      8192 unique bytes，等于完整 A data destination，0 collision
B:      8192 unique bytes，等于该 wave B segment，0 collision
ScaleA:  512 unique bytes，等于完整 SA region，0 collision
ScaleB:  512 unique bytes，等于该 wave SB segment，0 collision
```

所有 slot-relative consumer 地址 `<0xae00`；slot1 整体加 `0xae00` 后
仍 `<0x15c00`。

专用 F4 WMMA 的文档 shape 是 32×16×128；本 kernel 在源码中把 weight
作为硬件 Matrix A、activation 作为硬件 Matrix B，再把 N32 accumulator
拆成两个逻辑 N16 fragment。本文不从 WMMA opcode反推 TDM transaction。

## 13. SiLU output staging 与 TDM store

### 13.1 Raw N256 → output N128

四个 compute wave 的 output 所有权：

| compute wave | raw gate/up columns | activated output columns（WG-relative） |
|---:|---|---|
| 0 | `[0,64)` | `[0,32)` |
| 1 | `[64,128)` | `[32,64)` |
| 2 | `[128,192)` | `[64,96)` |
| 3 | `[192,256)` | `[96,128)` |

当前 act1 epilogue 对相邻 `(gate,up)` 做：

```text
g   = min(gate, +7)
u   = clamp(up, -7, +7)
out = g * sigmoid(g) * u
```

结果转换为 BF16。

### 13.2 LDS staging pattern

output LDS view 是：

```text
shape     = [64,128] BF16
row pitch = 128*2 = 256 B = 0x100
range     = [0,0x4000)
```

源码坐标：

```text
row = 16*wm + lane16
col = 32*compute_wave + 8*wn + 4*kgrp + p

wm=0..3, wn=0..3, kgrp=0..1, p=0..3
LDS byte = row*256 + col*2
```

当前机器代码使用 8 个静态 `ds_store_2addr_b64` site
（L984、L1050、L1180、L1469-L1473）。每 wave 动态执行这 8 条，四 wave
合计 32 条。

逐 byte 枚举：

```text
enumerated bytes = 16384
unique bytes     = 16384
range            = [0x0000,0x4000)
holes            = 0
collisions       = 0
```

虽然只有 M32 最终写 Global，所有 wave 仍在 LDS 生成完整 M64×N128 staging。

### 13.3 四条 Global store descriptor

store atom 的 `num_warps=4` 令四个 wave 按 M16 切分 staging。令 store wave
为 `w`：

```text
lds_addr(w)    = w*0x1000
global_addr(w) = C
               + (blk_m+16w)*(3072*2)
               + blk_n
```

最后一项是 `out_col=blk_n/2` 乘 BF16 2 B，恰好等于 raw `blk_n` bytes。

| store 字段 | 实际值 |
|---|---|
| `data_size` | 1 → 2 B/BF16 |
| `tensor_dim0` | `0x7fffffff` BF16 |
| `tensor_dim1` | `max(32-16w,0)` |
| `tile_dim0` | 128 BF16 = 256 B |
| `tile_dim1` | 16 rows |
| `tensor_dim0_stride` | 3072 BF16 = 6144 B |
| `workgroup_mask` | 0（store 本来也忽略该字段） |
| padding/gather/iteration/atomic | 全 0 |
| TH | 默认 `RT` |
| nominal payload | `128*16*2 = 4096 B = 0x1000`/wave |

本例的 valid/padding store mask 不是 `EXEC` bitmask，而是四个 descriptor 的
outer tensor dim：

| store wave | LDS M rows | `tensor_dim1` | nominal | useful Global | dropped |
|---:|---|---:|---:|---:|---:|
| 0 | `[0,16)` | 32 | 4096 B | 4096 B | 0 |
| 1 | `[16,32)` | 16 | 4096 B | 4096 B | 0 |
| 2 | `[32,48)` | 0 | 4096 B | 0 | 4096 B |
| 3 | `[48,64)` | 0 | 4096 B | 0 | 4096 B |
| **WG** | `[0,64)` | — | **16384 B** | **8192 B** | **8192 B** |

Nout128 全部 valid，没有 N padding/tail。

对应 Group-1 dword：

```text
C wave0:
00010000 ffff0000 00207fff 00800000
00000010 00000c00 00000000 00000000

C wave1:
00010000 ffff0000 00107fff 00800000
00000010 00000c00 00000000 00000000

C wave2 / wave3:
00010000 ffff0000 00007fff 00800000
00000010 00000c00 00000000 00000000
```

descriptor 和 issue 位于当前 ISA L1478-L1503。四个 wave 都发 store，
不是 wave0 发一条 M64×N128 store。

### 13.4 Store 同步闭环

```text
input/WMMA finished
  s_wait_tensorcnt 0                 # L910
  WG barrier                         # signal L917, wait L980
  all waves stage BF16 to LDS
  s_wait_dscnt 0                     # L1476
  WG barrier                         # signal L1477, wait L1501
  each wave tensor_store_from_lds    # L1502
  each wave s_wait_tensorcnt 0       # L1503
```

第二个 barrier 保证任一 store wave 读取自己的 M16 row strip 时，四个
compute wave 的 N stripes 都已经写完。

## 14. Byte、descriptor 与 allocation 审计

### 14.1 每 K256、每 active WG

| tensor | nominal descriptor payload | in-range Global bytes | application-useful bytes | LDS footprint |
|---|---:|---:|---:|---:|
| A | 8192 | 4096 | 4096 | 9216 |
| B | 32768 | 32768 | 32768 | 32768 |
| ScaleA | 512 | 512 | 256 | 512 |
| ScaleB | 2048 | 2048 | 2048 | 2048 |
| **load 合计** | **43520 (`0xaa00`)** | **39424 (`0x9a00`)** | **39168 (`0x9900`)** | **44544 (`0xae00`)** |

这里 ScaleA 的“useful”只计 M32 valid rows；descriptor 确实对完整 M64
physical block 发出 in-range load。

### 14.2 每 active WG 的完整 K7168

`Q=28`：

| tensor | descriptor 数 | nominal payload | in-range bytes | useful bytes |
|---|---:|---:|---:|---:|
| A | 112 | 229376 | 114688 | 114688 |
| B | 112 | 917504 | 917504 | 917504 |
| ScaleA | 112 | 14336 | 14336 | 7168 |
| ScaleB | 112 | 57344 | 57344 | 57344 |
| **load 合计** | **448** | **1218560 (`0x129800`)** | **1103872 (`0x10d800`)** | **1096704 (`0x10bc00`)** |

此外每 WG：

```text
output store descriptors = 4
nominal store payload     = 16384 B = 0x4000
useful output bytes       =  8192 B = 0x2000
```

### 14.3 2304 个 active WG 的逐-WG求和

该口径保留 A/ScaleA 在 24 个 N WG 之间的重复：

| tensor | nominal descriptor payload sum | in-range sum | per-WG useful sum |
|---|---:|---:|---:|
| A | 528482304 | 264241152 | 264241152 |
| B | 2113929216 | 2113929216 | 2113929216 |
| ScaleA | 33030144 | 33030144 | 16515072 |
| ScaleB | 132120576 | 132120576 | 132120576 |
| **load 合计** | **2807562240** | **2543321088** | **2526806016** |

动态 input TDM instruction：

```text
per wave       = 28*4 = 112
per active WG  = 112*4 = 448
active grid    = 448*2304 = 1032192
```

动态 output TDM：

```text
4/WG * 2304 = 9216
nominal store payload = 16384*2304 = 37748736 B
useful Global output  =  8192*2304 = 18874368 B
```

1152 个 tail WG 不发 TDM，所以 active-grid 动态 TDM 数就是整个 dispatch
的动态 TDM 数。

### 14.4 Active grid 的 unique application-useful 地址

去重 A/ScaleA 在 24 个 N tiles 之间的重复后：

| tensor | unique useful 公式 | bytes |
|---|---:|---:|
| A | `3072 valid rows * 3584` | 11010048 |
| ScaleA | `3072 * 224` | 688128 |
| B | `96 * 6144 * 3584` | 2113929216 |
| ScaleB | `96 * 6144 * 224` | 132120576 |
| **read 合计** | — | **2257747968** |
| BF16 output | `3072 * 3072 * 2` | **18874368** |
| **unique useful read+write** | — | **2276622336** |

这与 runner 的 pure-Python effective-byte 口径一致。它不是 descriptor
traffic，也不是 HBM traffic。

若只按 descriptor 的 in-range 地址去重，但仍把 ScaleA 的 M padding
算进去，则 read 地址集合是 `2258436096 B`；它比 application-useful read
多 `96*32*224=688128 B`。

### 14.5 LDS allocation

```text
one stage       = 44544 B = 0xae00
two-slot ring   = 89088 B = 0x15c00
output staging  = 16384 B = 0x4000，复用 ring 开头
```

把 fixed allocation 机械乘 WG 数：

```text
2304 active WG instances * 89088 = 205258752 B
3456 all WG instances    * 89088 = 307888128 B
```

这两个乘积只是“逻辑 WG allocation instance 总和”，不是同时驻留的物理
LDS，也不是流量。实际同时驻留量由 CU/WGP 调度和其它 resource 决定。

## 15. 代表性手算：expert37、N tile7、q3

在要求的
`experts=96,tokens=512,topk=6,model_dim=7168,inter_dim=3072` 下，取：

```text
e       = m_tile = 37
n_tile  = 7
q       = 3                  # logical K [768,1024)
g       = floor(37/16) = 2
ml      = 37 mod 16 = 5
bid_x   = 384*2 + 16*7 + 5 = 885
blk_m   = 37*64  = 2368 = 0x940
blk_n   = 7*256  = 1792 = 0x700
psum[37]= 37*64+32 = 2400
mn_oob  = 2400-2368 = 32
slot    = q mod 2 = 1
```

### 15.1 四个 load 的 Group 0 地址

下表 Global offset 均相对本 tensor base：

| wave | A offset | B offset | ScaleA offset | ScaleB offset |
|---:|---:|---:|---:|---:|
| 0 | `0x00818180` | `0x30f21800` | `0x00081e00` | `0x030f2300` |
| 1 | `0x00826180` | `0x30f59800` | `0x00081e80` | `0x030f5b00` |
| 2 | `0x00834180`，A OOB | `0x30f91800` | `0x00081f00` | `0x030f9300` |
| 3 | `0x00842180`，A OOB | `0x30fc9800` | `0x00081f80` | `0x030fcb00` |

例如 wave0：

```text
A:
  y=0..15
  [0x00818180 + y*0x0e00,
   0x00818200 + y*0x0e00)

B:
  y=0..3
  [0x30f21800 + y*0x0e000,
   0x30f22000 + y*0x0e000)

ScaleA:
  [0x00081e00,0x00081e80)

ScaleB:
  [0x030f2300,0x030f2400)
  [0x030f3f00,0x030f4000)
```

B 的每条 2048-B chunk映射到 16 个 N rows ×128 packed-K bytes；
ScaleB 两条 256-B chunk映射到 2 个 N32 super-row。

### 15.2 slot1 LDS destination

| wave | A start | B start | ScaleA start | ScaleB start |
|---:|---:|---:|---:|---:|
| 0 | `0x0ae00` | `0x0d200` | `0x15200` | `0x15400` |
| 1 | `0x0b700` | `0x0f200` | `0x15280` | `0x15600` |
| 2 | `0x0c000` | `0x11200` | `0x15300` | `0x15800` |
| 3 | `0x0c900` | `0x13200` | `0x15380` | `0x15a00` |

A 每段 footprint 是 `0x900`；B/ScaleA/ScaleB 每段 payload/footprint
分别是 `0x2000/0x80/0x200`。所有终点不超过 `0x15c00`。

### 15.3 Output store

WG output tile 的 byte base：

```text
C relative offset
  = blk_m*(3072*2) + blk_n
  = 2368*0x1800 + 0x700
  = 0x00de0700
```

四条 store：

| store wave | LDS source | Global relative start | `tensor_dim1` | useful |
|---:|---:|---:|---:|---:|
| 0 | `0x0000` | `0x00de0700` | 32 | 4096 B |
| 1 | `0x1000` | `0x00df8700` | 16 | 4096 B |
| 2 | `0x2000` | `0x00e10700` | 0 | 0 |
| 3 | `0x3000` | `0x00e28700` | 0 | 0 |

有效 Global output 坐标是：

```text
M rows      = [2368,2400)
Nout cols   = [blk_n/2, blk_n/2+128) = [896,1024)
payload     = 32*128*2 = 8192 B
```

## 16. 只读 Python 枚举与静态 opcode 复核

本次使用 stdin 形式的只读 Python checker；没有新增 checker 文件。枚举内容：

1. 3456 个 swizzled `(m_tile,n_tile)`；
2. q3 与完整 28-phase K range 的 A/B/ScaleA/ScaleB
   逐 byte/逐 logical-coordinate 集合；
3. A padding holes 和所有 LDS region 边界；
4. 每个 compute wave 的 A/B/ScaleA/ScaleB DS consumer 地址；
5. output LDS 的 `wave×row×col×byte`；
6. 四条 store 的 nominal/useful/drop 集合；
7. 28 K phases、descriptor 数和全 grid byte arithmetic。

关键断言结果：

```text
grid pairs=3456, unique=3456, active=2304, tail=1152

A LDS:
  data destinations=8192
  padding holes=1024
  collisions=0
  allocation=[0,0x2400)

logical q3:
  A valid=4096 unique bytes
  B=32768 unique bytes, exact N256 × packed-K[384,512)
  ScaleA=512 unique bytes, useful-by-wave=[128,0,128,0]
  ScaleB=2048 unique bytes, exact N256 × scale[24,32)

full K:
  A/B/ScaleA/ScaleB unique = 114688/917504/14336/57344
  packed-K=[0,3584), scale=[0,224)
  cross-q holes=0, collisions=0

per compute wave LDS consumer:
  A/B/ScaleA/ScaleB unique = 8192/8192/512/512
  collisions=0

output LDS:
  bytes=16384, unique=16384
  range=[0,0x4000), holes=0, collisions=0

store:
  nominal=16384, useful=8192, dropped=8192
```

对当前 1660 行 ISA 的行首 mnemonic 静态计数：

| opcode | 静态 site 数 |
|---|---:|
| `tensor_load_to_lds` | 12 |
| `tensor_store_from_lds` | 1 |
| `s_wait_tensorcnt` | 5 |
| `s_wait_dscnt` | 23 |
| `s_barrier_signal` | 6 |
| `s_barrier_wait` | 6 |
| `ds_load_b128` | 96 |
| `ds_load_2addr_b32` | 12 |
| `ds_store_2addr_b64` | 8 |
| `v_wmma_scale_f32_32x16x128_f4` | 48 |

静态 site 不能直接当动态次数：

- 12 load sites = q0 四条 + q1 四条 + rolled steady body四条；
- steady 四条动态重复 26 次；
- 48 WMMA sites = steady、q26 drain、q27 drain 各 16；
- 每 active WG 动态 WMMA 仍是 `28*16*4=1792`。

## 17. HBM transaction 与未知边界

### 17.1 静态材料可以确定

- D# 每个字段、单位、mask、padding、TH、Global/LDS 地址集合；
- OOB 补零/丢 store 的坐标；
- nominal payload、in-range bytes、application-useful bytes；
- 双槽 ring 与 output staging 的 allocation 边界；
- 同-wave TDM in-order、跨-wave unordered；
- `s_wait_tensorcnt 4/0` 和 WG barrier 的依赖上界；
- 当前 launch 没有 cluster/multicast；
- 当前四种 load 的第一维分别为 128 B、2048 B、128 B、256 B，
  Global/LDS offset 都保持至少 4-B 对齐。

### 17.2 不能静态确定

1. 一条 descriptor 最终拆成多少条 128-B/256-B GLX request。MI400 Guide
   §4.10.2 给出 direct-copy eligibility，不给当前运行的唯一 transaction
   trace。
2. 精确 HBM read/write bytes。cache hit、cache-line 合并、replay、alignment、
   partition 和运行时 pointer base 都会影响；descriptor payload 与 unique
   address set 都不能代替 PMC。
3. A wave2/3 OOB-zero 与 C wave2/3 dropped-store 在 TDM 内部经历哪些
   micro-op。架构语义可确定结果，不能确定内部 cycle/request 序列。
4. burst 内第四条 TDM 因 3-entry issue-to-XACK limit 实际 stall 多少 cycle，
   以及 `s_wait_tensorcnt` 的实际等待时间。
5. TDM 与 DS/WMMA 的逐 cycle overlap、仲裁和 cache retention效果。
6. ScaleA padding rows 的具体 byte 内容。descriptor 会读这些 in-range
   地址，但生产 route quant 不把它们定义为 application-valid；数学结果依赖
   A padding rows 被 OOB-zero，而不是依赖这些 scale 值。
7. “active-grid allocation instance 总和”不说明同时驻留多少 WG；需要结合
   实际 CU/WGP resource scheduling。

## 18. 静态审计摘要与资料索引

### 18.1 审计摘要

- 目标确认：当前 1660 行 `moe_gemm1_a4w4_v0.s`，不是旧 1827 行版本。
- ABI：184 B，offset/load/metadata 一致；3 个 4-B ABI padding。
- Grid：3456 unique tiles；2304 active，1152 sentinel，tail 不发 TDM。
- K loop：28×K256；双槽 b2；每 wave 4 descriptors/K phase。
- Input TDM：448 descriptors/active WG，1,032,192/active grid。
- Store TDM：4 descriptors/active WG，9,216/active grid。
- Descriptor：全部 2D、count1、iteration off、mask0、RT；只有 A 开 padding。
- LDS：stage `0xae00`，ring `0x15c00`；load/store枚举无 collision 或越界。
- Useful traffic：active-grid unique reads `2257747968 B`，BF16 writes
  `18874368 B`；二者都不是实测 HBM traffic。

### 18.2 当前 ISA 锚点

- `my_code/moe_gemm1_a4w4_v0.s`
  - kernarg scalar load、swizzle：L10-L149
  - 96-entry expert lookup / sentinel branch：L150-L238
  - wave/LDS base 与 A descriptor：L239-L377
  - B/ScaleA/ScaleB descriptor：L379-L538
  - q1 prologue：L539-L553
  - K-loop setup：L554-L578
  - 26-trip steady body：L579-L696
  - q26/q27 drain：L698-L838
  - SiLU 与 output LDS staging：L839-L1477
  - output store descriptor：L1478-L1503
  - resource/ABI metadata：L1508-L1655

### 18.3 生产源码与 runner

- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
  - specialization、LDS sizes：L105-L201
  - swizzle/expert lookup：L221-L289
  - TDM job split与 descriptor construction：L339-L500
  - LDS consumer坐标：L543-L616
  - b2 pipeline：L872-L917
  - BF16 staging / TDM store：L1007-L1253
  - launch kargs/grid：L1256-L1287
- `aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py`
  - ScaleA preshuffle定义：L39-L52
  - quant layout：L157-L210
  - scale byte store：L433-L458
  - route-indexed physical row mapping：L1404-L1468
- `aiter/ops/shuffle.py`
  - GUGU row interleave / B shuffle：L116-L204
  - ScaleB n32k4：L239-L289
  - compact F4 layout说明：L318-L335
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py`
  - cluster/waves selection：L28-L60
  - exact launch argument wiring：L63-L150
- `aiter/ops/flydsl/grouped_moe_gfx1250.py`
  - contiguous-M、route/psum、GEMM1 inputs：L409-L659
- `my_code/isa_runner/gemm_batch_isa_runner.py`
  - 184-B ABI：L203-L300
  - representative workload/grid：L330-L470
  - exact input layouts：L750 以后
- `my_code/isa_runner/moe_gemm1_cpp_launcher.cpp`
  - exact symbol/resource contract：L25-L38
  - 184-B tensor/shape/stride验证：L175-L324

### 18.4 本地硬件资料

- `mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt`
  - §4.1 Table 10/11，TH policies，printed pp.35-36
  - §10.11.1 Tensor Instructions，p.140
  - §10.11.2 Tensor and Tile Addressing，pp.141-142
  - §10.11.3 Optional Operations / Iteration，pp.142-143
  - §10.11.4 Tensor DMA Descriptor，pp.143-146
  - §10.11.6 OOB Behavior，p.147
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`
  - §4.10.1-§4.10.4，printed pp.197-204
  - §4.10.7 OOB Behavior，p.205
  - §4.10.8 Performance and Tracking，p.206
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/TX/mi400_tensor_dma#72.txt`
  - §5.1 Tensor Inflight Limitations
  - §7.1 constants

`gemm1_a4w4_tdm_silu_gfx1250_deep_dive.md` 与
`MAB_TDMs_tdm_load_pattern.md` 仅用于交叉检查术语和遗漏项；本文所有
descriptor 数字、当前 ISA 行号、wpt4 分工、wait threshold 和静态计数均已
针对 `moe_gemm1_a4w4_v0.s` 重新解析。
