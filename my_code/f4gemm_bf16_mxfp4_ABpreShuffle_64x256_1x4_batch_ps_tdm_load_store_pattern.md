# gfx1250 `64x256_1x4_batch_ps` GEMM 的 TDM Load/Store 数据 Pattern

<!-- markdownlint-disable MD013 MD033 MD060 -->

本文专门分析
`my_code/f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.s`
中的 `tensor_load_to_lds` 与 `tensor_store_from_lds`。分析对象是当前工作区里的
ISA 文本，不是早期设计草案。

本文只把以下操作称为 TDM：

- `tensor_load_to_lds`：Global Memory → LDS；
- `tensor_store_from_lds`：LDS → Global Memory；
- `s_wait_tensorcnt`：等待上述异步 TDM op。

`global_prefetch_b8`、`ds_load_*`、`ds_store_*` 分别属于普通 Global prefetch
和 LDS 指令；本文只在它们决定 TDM 的生产/消费关系时讨论它们。

文中使用四种证据标签：

- **ISA/descriptor 事实**：字段或立即数直接见于目标 `.s`；
- **源码事实**：来自 `shuffle.py` 或 runner 的输入布局；
- **硬件文档事实**：来自本地 CDNA5 ISA、MI400 Shader Programming Guide
  或 TDM 设计资料；
- **推导**：由以上事实做地址展开、集合计数或控制流跟踪得到；
- **未知边界**：现有 ISA 和本地文档不足以唯一确定的微架构行为。

区间统一写成左闭右开 `[begin,end)`；“payload”只统计被 descriptor 描述的
有效数据字节，“envelope”是第一字节到最后一字节之间的地址跨度，两者在
strided 2D tile 中不是同一个量。

## 1. 结论摘要

目标 kernel 的核心几何是：

```text
wave32:                    M64 x N64 output
4 waves / WG:              M64 x N256 output
clusterDim (N x M x Z):    4 x 1 x 1 WG
one cluster task:          M64 x N1024 output
one reduction body:        K256
```

WG 内 wave 与 TDM 角色：

| logical wave | 输出区域（WG 相对坐标） | 唯一 TDM load 职责 | 单 K256 descriptor payload |
|---:|---|---|---:|
| 0 | `M[0,64) × N[0,64)` | A data | `0x2000 = 8192 B` |
| 1 | `M[0,64) × N[64,128)` | B data | `0x8000 = 32768 B` |
| 2 | `M[0,64) × N[128,192)` | ScaleA | `0x0200 = 512 B` |
| 3 | `M[0,64) × N[192,256)` | ScaleB | `0x0800 = 2048 B` |

四条 descriptor 合计：

```text
one WG, one K256:
  A 0x2000 + B 0x8000 + SA 0x0200 + SB 0x0800
  = 0xAA00 = 43520 B
```

关键结论如下。

1. 四类 load 都是 **2D、byte mode、无 padding、无 descriptor iteration**；
   K 方向的 `+0x800` 或 `+0x100` 是 scalar 指令更新 `global_addr`，不是
   `global_addr_increment` descriptor 字段。
2. A/B 的物理连续单元不是“一条逻辑 row”，而是 AB preshuffle 后的
   `16 rows × 16 packed bytes = 256 B` tile。一次 K256 descriptor 沿连续
   维合并 8 个这样的 tile。
3. ScaleA/ScaleB 的物理连续单元是
   `32 rows × 4 E8M0 bytes = 128 B` tile。一次 K256 descriptor 沿连续维
   合并 2 个这样的 tile。
4. A 和 ScaleA 在 cluster 内由 4 个 N 方向 WG 请求相同地址，
   `workgroup_mask=0x000f`；B 和 ScaleB 按 WG 的 N256 分片，
   `workgroup_mask=1<<wg_x`，requester set 大小为 1。
5. WG 内四个 compute wave 都读取同一份 A/ScaleA LDS tile；B/ScaleB
   各自按 N64 切成四份，由 wave `w` 读取第 `w` 份。
6. 输入使用四槽 ring。首个 task 先填 slot0/1/2，再填 slot3；steady body
   消费一个 slot，并向该 slot 预取 `q+4` 的 K256 tile。
7. 输出先由四个 wave 把四个 N64 quarter 拼成 LDS 中完整的
   `M64×N256` BF16 row-major tile，再由 wave0 发出唯一一条
   `tensor_store_from_lds`，payload 为 `0x8000 = 32768 B`。

## 2. 符号、batch 基址与 preshuffle 坐标

### 2.1 本文符号

```text
b             batch index = TTMP7[31:16]
mt            logical M tile index
nt            logical N tile index of one WG
w             logical wave id, 0..3
q             K256 body index, 0..Q-1
Q             K / 256

M0            64 * mt
N0            256 * nt
P             K / 2 bytes       # packed MXFP4 row extent
S             K / 32 bytes      # E8M0 scale row extent
```

固定 `clusterDim=(4,1,1)` 下，persistent task 到 tile 坐标的映射是：

```text
C_N = N / 1024
C_M = M / 64

cluster_n = task mod C_N
cluster_m = task div C_N

mt = cluster_m
nt = 4*cluster_n + wg_x

task_next = task + (1 << (log2_grid_x + log2_grid_y))
```

同一个 cluster 的 4 个 WG 具有相同 `task/mt/cluster_n`，只由
`wg_x=0..3` 选择四个连续 N256 tile。不同物理 cluster 沿上述
`task_next` 序列持久化领取后续 M64×N1024 区域。

目标 ISA 从 `s[2:11]` 取得五个 base pointer，从 `s12:s16` 取得五个
row byte stride，从 `s17:s19` 取得 `M,N,K`。入口 L25-L55 用 batch index
和追加在 120-byte ABI 尾部的五个 u64 stride 调整 base pointer。

因此 batch 平面的基址为：

```text
D_b  = ptr_D      + b * (M * N * 2)
A_b  = ptr_A      + b * (M * P)
B_b  = ptr_B      + b * (N * P)
SA_b = ptr_ScaleA + b * (M * S)
SB_b = ptr_ScaleB + b * (N * S)
```

**ISA/descriptor 事实。** batch 只改变这五个 base；后面的 descriptor、
ring、wait 和 store 控制流不含跨 batch 共享。`clusterDim.z=1`，所以同一个
cluster 不会混入两个 batch。

### 2.2 A/B 的 16×16 packed-byte preshuffle

**源码事实。** `aiter/ops/shuffle.py` L318-L335 把原始
`[rows,P]` byte tensor 做：

```text
[rows/16, 16, P/16, 16]
    -> permute(0, 2, 1, 3)
    -> [rows/16, P/16, 16, 16]
```

令原始逻辑 row 为 `r`，packed-K byte 坐标为 `kb`，则 preshuffle 后的
线性 byte offset 为：

```text
weight_off(r,kb)
  = floor(r/16)  * (16*P)
  + floor(kb/16) * 256
  + (r mod 16)   * 16
  + (kb mod 16)
```

所以物理连续的 256 B tile 是：

```text
[one row-block of 16 rows] × [16 packed K bytes per row]
= 16 rows × 32 logical FP4 K values
= 256 B
```

这也是为什么 descriptor 的 dim0 不能按未 shuffle 的单 row 来解释。

### 2.3 ScaleA/ScaleB 的 32×4 preshuffle

**源码事实。** `shuffle_scale_f4` 在 `shuffle.py` L292-L315 把原始
`[rows,S]` E8M0 tensor 做：

```text
[rows/32, 32, S/4, 4]
    -> permute(0, 2, 1, 3)
    -> [rows/32, S/4, 32, 4]
```

令 scale row 为 `r`，K32 scale 坐标为 `ks`，则物理 offset 为：

```text
scale_off(r,ks)
  = floor(r/32) * (32*S)
  + floor(ks/4) * 128
  + (r mod 32)  * 4
  + (ks mod 4)
```

物理连续的 128 B tile 是：

```text
32 rows × 4 scales/row
= 32 rows × K128
= 128 B
```

### 2.4 TDM 的硬件地址语义

**硬件文档事实。**

- CDNA5 ISA §10.11.1-10.11.4 规定 tensor op 不是 per-lane 指令，忽略
  `EXEC`；2D 地址为
  `global_addr + data_size*(x + y*tensor_dim0_stride)`。
- `tile_dim*` 和 `tensor_dim*` 都以 `data_size` 为单位；
  `lds_addr` 本身以 byte 为单位。
- `TENSOR_LOAD_TO_LDS` 把 tile 按 descriptor 顺序紧凑写入 LDS；只有
  `pad_enable=1` 才在 LDS 插空。
- `TENSOR_STORE_FROM_LDS` 不执行“反 padding”；store 时 padding 字段被忽略。
- MI400 Shader Programming Guide §4.10.1-4.10.4 给出相同语义，并说明
  TDM 在内部拆成 `GLOBAL_LOAD_ASYNC_TO_LDS`、
  `CLUSTER_LOAD_ASYNC` 或 `GLOBAL_STORE_ASYNC_FROM_LDS`。

目标所有 descriptor 的 `data_size=0`，编码语义为 1 B。因此本文后面写的
dim、stride 和 tile width 数字都同时是“descriptor element 数”和 byte 数。

## 3. Descriptor 公共解码

### 3.1 目标使用的 12-SGPR 2D descriptor

所有 load 都写入 `s[32:43]`，store 写入 `s[80:91]`：

| group | load SGPR | store SGPR | 关键内容 |
|---|---|---|---|
| Group 0 | `s32:s35` | `s80:s83` | `count`、gather、`lds_addr`、57-bit `global_addr`、`type` |
| Group 1 | `s36:s43` | `s84:s91` | mask/flags、tensor dim、tile dim、stride |
| Group 2/3 | 未传 | 未传 | 目标不是 3D-5D、gather 或 descriptor iteration |

Group 0 的固定构造：

```text
count        = 1
is_restore   = 0
gather       = 0
type         = 2 ("image")
global_addr  = low32 + (high25 << 32)
```

`s35`/`s83` 的构造是
`(pointer_hi & 0x01ffffff) | 0x80000000`；高位 `0x8` 是 `type=2`，
不是 global address 的一部分。

### 3.2 四类 load 的公共 flags

目标先清零 `s36:s43`，再构造以下字段：

| Group-1 字段 | 编码值 | 语义 |
|---|---:|---|
| `data_size` | 0 | 1 B |
| `atomic_barrier_enable` | 0 | descriptor 完成后不请求有效 LDS atomic arrival |
| `iterate_enable` | 0 | 不使用 descriptor iteration |
| `pad_enable` | 0 | `s_bitset0_b32 s36,20` 明确清 bit20 |
| `early_timeout` | 1 | `s_bitset1_b32 s36,21` |
| `pad_interval/pad_amount` | 0 | padding 关闭 |
| `tile_dim2` | 0 | 2D descriptor |
| `tensor_dim1_stride` | 0 | 第三维未使用 |

因此 load flags dword 是：

```text
A / ScaleA:  s36 = 0x0020000f
B / ScaleB:  s36 = 0x00200000 | (1 << wg_x)
```

**硬件文档事实。** `early_timeout=1` 表示 multicast 数据从 GL2 返回后，
GL1 可立即对已经到达的 requester 返回，不继续等待晚到 requester。它不改变
地址集合或 payload。

### 3.3 Temporal hint

`TH` 是 instruction field，不在 D# 内。CDNA5 ISA 的 cache-control 表把
默认 `RT` 定义为 regular temporal，把 `NT` 定义为 near/far cache 都不预期
复用。

目标实际编码是：

| 动态路径 | temporal hint |
|---|---|
| wave0 / A data | `TH_LOAD_NT` |
| wave1 / B data | 默认 `RT` |
| wave2 / ScaleA 首个 task 的 q0-q3 cold prologue | 默认 `RT` |
| wave2 / ScaleA steady refill、跨 task fast restart | `TH_LOAD_NT`，因为与 wave0 共用主体 |
| wave3 / ScaleB | 默认 `RT` |
| D store | 默认 store `RT` |

这一差异只影响 cache retention hint，不改变下面的坐标和字节范围。

## 4. A data：M64 广播 tile

### 4.1 A descriptor

目标 A 初始化位于 ISA L321-L392，首次四槽 issue 位于
L398/L413/L428/L620。

| 字段 | 值 | 含义 |
|---|---:|---|
| issuer | wave0 | WG 内唯一 A TDM specialist |
| `global_addr(q)` | `A_b + M0*P + q*0x800` | 当前 M64、K256 的 preshuffle 起点 |
| `tensor_dim0` | `16*P = 8*K` | 一个 M16 super-row 的完整 packed-K byte extent |
| `tensor_dim1` | `(M-M0)/16` | 从当前 tile 到矩阵末端还剩多少个 M16 block |
| `tensor_dim0_stride` | `16*P` | 相邻 M16 block 的物理间距 |
| `tile_dim0` | `0x800 = 2048` | 8 个连续 16×16 packed-byte tile |
| `tile_dim1` | 4 | M64 含 4 个 M16 block |
| `workgroup_mask` | `0x000f` | cluster 内四个 N WG 请求相同 A |
| software address step | `+0x800` | 下一个 K256；`s56=0x800` |
| descriptor payload | `0x800*4 = 0x2000` | 8192 B |

### 4.2 Global 二维展开

对固定 `q`，descriptor 坐标是：

```text
y = 0..3
x = 0..0x7ff

A_global(y,x)
  = A_b + M0*P + q*0x800 + y*(16*P) + x
```

把连续维 `x` 分解为：

```text
x = 256*t + 16*r + u
t = 0..7       # K 方向的 16-packed-byte tile
r = 0..15      # row_in_M16
u = 0..15      # packed byte inside one row
```

则该 byte 对应 shuffle 前坐标：

```text
logical M row      = M0 + 16*y + r
packed-K byte      = 128*q + 16*t + u
logical K interval = [256*q, 256*q+256)
```

**推导。** 一条 A descriptor 的地址集合恰好是：

```text
A[M0:M0+64, packed_K_byte 128q:128q+128]
```

它在 Global Memory 中表现为 4 条连续 `0x800`-byte chunk，chunk 起点间隔
`16*P`：

| 项 | 数值 |
|---|---:|
| 连续 chunk 数 | 4 |
| 每 chunk | `0x800 = 2048 B` |
| chunk stride | `16*P = 8*K B` |
| payload | `0x2000 = 8192 B` |
| address envelope | `3*(16*P)+0x800` |

### 4.3 LDS layout 与 WG 内广播

无 padding，所以 TDM 在 A slot 中紧凑生成：

```text
A_lds[y][t][r][u]
shape = [4 M16][8 K32][16 rows][16 packed bytes]
byte offset in slot
      = y*0x800 + t*0x100 + r*0x10 + u
```

四个 compute wave 的 A LDS base 都是 `lane_id*16 + A_slot_base`。
每个 wave 用 16 条 `ds_load_b128` 覆盖完整 `[A_slot,A_slot+0x2000)`。
所以：

- Global/TDM 侧：每 WG 每 K256 只有 wave0 的一条 8192-B descriptor；
- LDS consumer 侧：4 个 wave 都读取同一份 8192-B A tile；
- cluster 侧：4 个 WG 的 wave0 都发匹配请求，地址相同且 mask 为 `0xf`；
  目标不是“一个 leader 替四个 WG 发指令”。

## 5. B data：N256 分片 tile

### 5.1 B descriptor

目标 B 初始化位于 ISA L633-L705，首次四槽 issue 位于
L841/L856/L871/L932。

| 字段 | 值 | 含义 |
|---|---:|---|
| issuer | wave1 | WG 内唯一 B TDM specialist |
| `global_addr(q)` | `B_b + N0*P + q*0x800` | 当前 N256、K256 的 preshuffle 起点 |
| `tensor_dim0` | `16*P = 8*K` | 一个 N16 super-row 的完整 packed-K byte extent |
| `tensor_dim1` | `(N-N0)/16` | 剩余 N16 block 数 |
| `tensor_dim0_stride` | `16*P` | 相邻 N16 block 的物理间距 |
| `tile_dim0` | `0x800 = 2048` | 一个 N16×K256 chunk |
| `tile_dim1` | 16 | N256 含 16 个 N16 block |
| `workgroup_mask` | `1 << wg_x` | 只包含当前 N WG |
| software address step | `+0x800` | 下一个 K256 |
| descriptor payload | `0x800*16 = 0x8000` | 32768 B |

### 5.2 Global pattern

```text
y = 0..15
x = 256*t + 16*r + u

B_global(y,x)
  = B_b + N0*P + q*0x800 + y*(16*P) + x

logical N row = N0 + 16*y + r
packed-K byte = 128*q + 16*t + u
```

因此：

| 项 | 数值 |
|---|---:|
| 连续 chunk 数 | 16 |
| 每 chunk | `0x800 = 2048 B` |
| chunk stride | `16*P = 8*K B` |
| payload | `0x8000 = 32768 B` |
| address envelope | `15*(16*P)+0x800` |

不同 `wg_x` 的 `N0` 相差 256，故同一 cluster 的四个 B descriptor
覆盖互不重叠的 N256：

```text
wg_x=0: Ncluster+  0 .. Ncluster+255
wg_x=1: Ncluster+256 .. Ncluster+511
wg_x=2: Ncluster+512 .. Ncluster+767
wg_x=3: Ncluster+768 .. Ncluster+1023
```

### 5.3 LDS 中按 wave 分片

完整 B slot 的紧凑形状是：

```text
B_lds[y][t][r][u]
shape = [16 N16][8 K32][16 rows][16 packed bytes]
size  = 16 * 0x800 = 0x8000
```

ISA L114-L126 令 wave `w` 的 B lane base 多出 `w*0x2000`。因此：

```text
wave w reads:
  [B_slot + w*0x2000, B_slot + (w+1)*0x2000)

logical N:
  [N0 + 64*w, N0 + 64*(w+1))
```

每个 wave 仍执行 16 条 `ds_load_b128`，wave aggregate 为 8192 B；
四个 wave 的 B LDS 地址不重叠，合起来恰好覆盖 descriptor 的 32768 B。

## 6. ScaleA 与 ScaleB

### 6.1 ScaleA descriptor

目标 ScaleA 初始化位于 ISA L945-L1016，首次四槽 issue 位于
L1153/L1168/L1183/L1244。

| 字段 | 值 |
|---|---:|
| issuer | wave2 |
| `global_addr(q)` | `SA_b + M0*S + q*0x100` |
| `tensor_dim0` | `32*S = K` bytes |
| `tensor_dim1` | `(M-M0)/32` |
| `tensor_dim0_stride` | `32*S = K` bytes |
| `tile_dim0` | `0x100 = 256` |
| `tile_dim1` | 2 |
| `workgroup_mask` | `0x000f` |
| software address step | `+0x100` |
| payload | `0x200 = 512 B` |

连续维分解：

```text
x = 128*t + 4*r + u
t = 0..1
r = 0..31
u = 0..3

SA_global(y,x)
  = SA_b + M0*S + q*0x100 + y*(32*S) + x

logical M row = M0 + 32*y + r
scale index   = 8*q + 4*t + u
```

ScaleA 的地址集合是 `M64 × 8 scales`，对应逻辑 K256。Global 中是两条
256-B chunk，起点间隔 `32*S=K` bytes；address envelope 为
`32*S+0x100`。

LDS slot 形状：

```text
SA_lds[y][t][r][u]
shape = [2 M32][2 K128][32 rows][4 scales]
size  = 0x200
```

四个 wave 都从同一个 ScaleA slot 读 4 条 `ds_load_b32`，每条 wave
aggregate 128 B，完整覆盖同一份 512-B SA tile。

### 6.2 ScaleB descriptor

目标 ScaleB 初始化位于 ISA L1257-L1329，首次四槽 issue 位于
L1466/L1481/L1496/L1557。

| 字段 | 值 |
|---|---:|
| issuer | wave3 |
| `global_addr(q)` | `SB_b + N0*S + q*0x100` |
| `tensor_dim0` | `32*S = K` bytes |
| `tensor_dim1` | `(N-N0)/32` |
| `tensor_dim0_stride` | `32*S = K` bytes |
| `tile_dim0` | `0x100 = 256` |
| `tile_dim1` | 8 |
| `workgroup_mask` | `1 << wg_x` |
| software address step | `+0x100` |
| payload | `0x800 = 2048 B` |

Global 坐标：

```text
y = 0..7
x = 128*t + 4*r + u

logical N row = N0 + 32*y + r
scale index   = 8*q + 4*t + u
```

因此 ScaleB 是 8 条 256-B chunk，stride 为 `32*S=K` bytes，
address envelope 为 `7*(32*S)+0x100`。

完整 LDS slot 形状：

```text
SB_lds[y][t][r][u]
shape = [8 N32][2 K128][32 rows][4 scales]
size  = 0x800
```

ISA L130-L136 给 wave `w` 增加 `w*0x200`，所以：

```text
wave w reads:
  [SB_slot + w*0x200, SB_slot + (w+1)*0x200)

logical N:
  [N0 + 64*w, N0 + 64*(w+1))
```

每个 wave 用 4 条 `ds_load_b32` 读取 512 B，四份无重叠地覆盖 2048 B。

## 7. 四槽 LDS ring 的完整地址图

目标头部注释 L8-L10、四个 wave 入口的 `s95:s98` 以及 metadata
`.amdhsa_group_segment_fixed_size 206848` 共同给出以下精确布局。

| operand | slot0 | slot1 | slot2 | slot3 | 每 slot payload |
|---|---|---|---|---|---:|
| A | `[0x00000,0x02000)` | `[0x02000,0x04000)` | `[0x04000,0x06000)` | `[0x06000,0x08000)` | `0x2000` |
| ScaleA | `[0x08000,0x08200)` | `[0x08200,0x08400)` | `[0x08400,0x08600)` | `[0x08600,0x08800)` | `0x0200` |
| ScaleB | `[0x08800,0x09000)` | `[0x09000,0x09800)` | `[0x09800,0x0A000)` | `[0x0A000,0x0A800)` | `0x0800` |
| B | `[0x0A800,0x12800)` | `[0x12800,0x1A800)` | `[0x1A800,0x22800)` | `[0x22800,0x2A800)` | `0x8000` |

图示：

```text
0x00000  A0  0x02000  A1  0x04000  A2  0x06000  A3  0x08000
0x08000 SA0  0x08200 SA1  0x08400 SA2  0x08600 SA3  0x08800
0x08800 SB0  0x09000 SB1  0x09800 SB2  0x0A000 SB3  0x0A800
0x0A800  B0  0x12800  B1  0x1A800  B2  0x22800  B3  0x2A800
0x2A800 output staging, 0x8000 bytes                         0x32800
```

一个 stage 的四类地址不是连续区间，但有效 byte 总量固定为 `0xAA00`。
整个四槽输入区：

```text
4 * 0xAA00 = 0x2A800 = 174080 B = 170 KiB
```

输出 staging 为 `[0x2A800,0x32800)`，32 KiB；metadata 声明的
`206848 B = 0x32800 = 202 KiB` 正好覆盖二者，没有尾部缺口。

### 7.1 每个 compute wave 的 LDS 消费区间

令当前 stage 的四个 base 分别为 `A_s,SA_s,SB_s,B_s`：

| wave `w` | A | ScaleA | B | ScaleB |
|---:|---|---|---|---|
| 0 | 整个 `[A_s,A_s+0x2000)` | 整个 `[SA_s,SA_s+0x200)` | `[B_s,B_s+0x2000)` | `[SB_s,SB_s+0x200)` |
| 1 | 同 wave0 | 同 wave0 | `[B_s+0x2000,B_s+0x4000)` | `[SB_s+0x200,SB_s+0x400)` |
| 2 | 同 wave0 | 同 wave0 | `[B_s+0x4000,B_s+0x6000)` | `[SB_s+0x400,SB_s+0x600)` |
| 3 | 同 wave0 | 同 wave0 | `[B_s+0x6000,B_s+0x8000)` | `[SB_s+0x600,SB_s+0x800)` |

这张表同时说明：

- A/ScaleA 在 WG 内广播，四个 wave 的 LDS read 地址重复；
- B/ScaleB 在 WG 内分片，四个 wave 的 LDS read 地址互斥；
- TDM specialist 身份和输出 N64 所有权是两个不同维度，不能把
  “wave1 发 B”误写成“B 只供 wave1 计算”。

## 8. K 扫描、stage 轮转与生产消费

### 8.1 K256 地址递推

目标使用：

```text
s58 = current logical K position
s59 = K
s58 += 0x100
```

所以：

```text
Q = K / 256
q = 0..Q-1

A/B global_addr step   = 0x800
  = 16 rows * (K256/2 packed bytes per row)

SA/SB global_addr step = 0x100
  = 32 rows * (K256/32 scale bytes per row)
```

对每个 logical row：

```text
A/B q covers packed bytes [128q,128q+128)
SA/SB q covers scales      [8q,8q+8)
```

Q 个 useful descriptor 因而在 K 方向无洞、无重叠地覆盖：

```text
packed data: [0,K/2)
scale data:  [0,K/32)
```

### 8.2 首次填充与 steady refill

首个 persistent task 的顺序是：

```text
issue q0 -> slot0
issue q1 -> slot1
issue q2 -> slot2
wait oldest TDM / prepare first DS
issue q3 -> slot3

body q:
  consume slot(q mod 4)
  issue future q+4 into the slot just released
```

目标有两套共享 compute 顺序：

- wave0(A) 与 wave2(ScaleA) 进入 `.Lbranch_000000007158`；
- wave1(B) 与 wave3(ScaleB) 进入 `.Lbranch_000000008e6c`。

两套顺序都执行 `P0→P1→P2→P3`，动态每 body、每 wave 恰有：

```text
1 tensor_load_to_lds
32 ds_load_b128       # 16 A + 16 B
8  ds_load_b32        # 4 SA + 4 SB
16 v_wmma_scale_f32_32x16x128_f4
1 s58 += 0x100
```

因此每 WG、每 K256 body 是 4 条 load descriptor、128 条
`ds_load_b128`、32 条 `ds_load_b32` 和 64 条 WMMA。

### 8.3 跨 persistent task 的 ring phase

若同一个物理 cluster 处理多个 logical task，代码不会强行把下一 task
重新放回 slot0。令 task `t` 的起始 ring phase 为 `phi_t`：

```text
phi_0     = 0
phi_(t+1) = (phi_t + Q) mod 4
slot(t,q) = (phi_t + q) mod 4
```

这是由末端 `s92=0/1/2/3` 和 L2440-L2714 的 fast-restart 分发表推导出的。
例如 `Q mod 4 = 1` 时，下一 task 的 q0 从 slot1 开始；若
`Q mod 4 = 0`，所有 task 都仍从 slot0 开始。

代表 shape `K=7168` 有 `Q=28=7*4`，所以 phase 不跨 task 旋转。

### 8.4 `s_wait_tensorcnt` 与 barrier

**硬件文档事实。**

- tensor op 在同一 wave 内按序完成，不同 wave 间无序；
- `s_wait_tensorcnt n` 只保证 `TENSORcnt<=n`；
- MI400 TDM 设计资料 §5.1/§7.1 给出的限制是每 wave 最多 3 个、
  每 SIMD 最多 6 个 user TDM descriptor in flight；
- workgroup barrier 是 `s_barrier_signal -1` /
  `s_barrier_wait 0xffff`；cluster user barrier 是 `-3` /
  `0xfffd`。

**推导。** cold prologue 每个 specialist 连续建立 3 个 in-flight load 后执行
`s_wait_tensorcnt 2`。因为同 wave 的 tensor op 有序，返回时最老 descriptor
已经完成；四个 specialist 随后通过 WG barrier 汇合，所以消费者开始读取
某 stage 时，A/B/SA/SB 四类生产者都已达到对应完成点。

steady body 同样在回填前使用 `s_wait_tensorcnt 2`，并用 WG barrier 把
per-wave TENSORcnt 观察传给其他 wave。`s_wait_tensorcnt 2` 不是 drain，
也不表示“恰好还有两条”。

每次 P3 ring wrap，wave0 为每个 WG 发一次 `s_barrier_signal -3`，cluster
内所有 wave 执行 `s_barrier_wait 0xfffd`。这使四个 WG 的 multicast
requester 和四槽所有权在回到 P0 前重新会合。

epilogue 则使用 `s_wait_tensorcnt 0`，确保 input TDM 和最终 output store
分别彻底完成。

### 8.5 下一 persistent task 的提前 load

目标把 `s70=K`、`s71=K+0x200`。每次 issue 后的 scalar 更新会：

1. 在下一 K 坐标仍 `<s70` 时，给当前 descriptor 的 `global_addr`
   加 `s56`；
2. 首次越过 K 时，切到前瞻 task 的 base/dim；
3. 允许前瞻 task 的 q0、q1 两个有效 descriptor；
4. 到 `K+0x200` 后把 `s39` 中的 `tile_dim0` 清零，使后续两次 lookahead
   成为零 payload TDM NOP。

前瞻 task 坐标是 `Mtile_next=s69`、`Ntile_next=s68`。task 边界
L3859-L3864 把 descriptor 地址推进两个 `s56`，恢复 dim 字段；fast-restart
分发表再加载 q2、q3。于是下一 task 在进入 compute 前已有完整四槽前导。

**末 task 的特殊情况。** 当没有下一 task 时，L244-L246 把前瞻坐标设为
当前坐标。由此静态控制流推导，最后仍会执行两条有效但不再消费的 q0/q1
重复 load，随后两条 `tile_dim0=0` 的零 payload TDM。它们不属于 useful
矩阵覆盖，统计 traffic 时必须单列。

## 9. D store：四个 wave 拼成一次 M64×N256 写回

### 9.1 四个 wave 写入 output LDS

output staging 是 `[0x2A800,0x32800)`，每个 D row 的 LDS stride 为
`0x200 = 512 B`。ISA L139-L146 构造：

```text
lane_row  = lane_id & 15
lane_half = lane_id >> 4

v91 =
    0x2A800
  + lane_row * 0x200
  + wave_id  * 0x080
  + lane_half * 0x010
```

epilogue L3690-L3705 的 16 条静态 `ds_store_b128` 可写成：

```text
offset(m_block,g) = m_block*0x2000 + g*0x20
m_block = 0..3
g       = 0..3
```

每条 `ds_store_b128` 每 lane 写 16 B。lane0..15 与 lane16..31 具有相同
`lane_row`，但 `lane_half` 相差 `0x10`，所以二者在同一 row 写相邻的
两个 16-B half。四个 `g` 覆盖一个 wave 的 128-B N64 quarter。

对任意 row `m=0..63`：

```text
wave w writes:
  [0x2A800 + m*0x200 + w*0x80,
   0x2A800 + m*0x200 + (w+1)*0x80)
```

四个 wave 的区间互斥，合起来是完整 512-B row。每 wave 写：

```text
64 rows * 128 B = 8192 B = 0x2000
```

每 WG 动态执行 `4 waves * 16 = 64` 条 `ds_store_b128`，覆盖
`0x8000` 个唯一 LDS bytes；静态 Python 枚举得到
`collisions=0, holes=0, range=[0x2A800,0x32800)`。

### 9.2 D store descriptor

目标 store 构造和 issue 位于 ISA L3710-L3748。

| 字段 | 实际值 | 含义 |
|---|---:|---|
| issuer | wave0 only | L3710-L3711 的 scalar wave-id branch |
| `count` | 1 | 每 WG 每 task 一条 store |
| `lds_addr` | `0x2A800` | 完整 output staging 起点 |
| `global_addr` | `D_b + M0*(N*2) + N0*2` | WG 输出 tile 左上角 |
| `workgroup_mask` | 0 | store 不做 multicast；硬件也忽略该字段 |
| `data_size` | 0 | byte mode |
| `tensor_dim0` | `0x200` | 当前 tile 每 row 512 B |
| `tensor_dim1` | `min(max(M-M0,0),64)` | valid M rows；合法 full tile 时为 64 |
| `tile_dim0` | `0x200` | 256 BF16 columns |
| `tile_dim1` | `0x40` | 64 rows |
| `tensor_dim0_stride` | `N*2` | D 的真实 Global row byte stride |
| gather/iteration/padding/atomic | 0 | 全部关闭 |

合法 launch 强制 `M%64=0`、`N%1024=0`，所以实际 task 总是完整
M64×N256，clamp 不会产生 partial tile。

Global 写地址公式：

```text
D_addr(m,n)
  = D_b + (M0+m)*(N*2) + (N0+n)*2

m = 0..63
n = 0..255
```

每 row 是 512-B 连续 store，row 起点 stride 为 `2*N`。因此：

| 项 | 数值 |
|---|---:|
| 连续 Global row 数 | 64 |
| 每 row | `0x200 = 512 B` |
| payload | `64*0x200 = 0x8000 = 32768 B` |
| address envelope | `63*(2*N)+0x200` |

虽然每个 compute wave拥有不同 N64 output quarter，Global TDM store 不按
wave 分成四条；wave0 的 descriptor 一次读取 LDS 中四个 quarter 拼出的
完整 row。

### 9.3 Store 的同步闭环

目标 epilogue 的顺序是：

```text
all waves:
  s_wait_dscnt 0
  s_wait_tensorcnt 0
  s_wait_alu depctr_va_vdst(0)
  WG barrier

all waves:
  convert F32 -> packed BF16
  16 x ds_store_b128 per wave
  s_wait_dscnt 0
  WG barrier

wave0 only:
  tensor_store_from_lds
  s_wait_tensorcnt 0

all waves:
  WG barrier
  cluster barrier
```

第二个 WG barrier 确保 wave0 看到其他三个 wave 的 LDS quarter。store 后的
WG barrier 则把 wave0 的 `TENSORcnt=0` 完成事实传播给非 issuer wave，
避免它们提前复用 output LDS。最后的 cluster barrier 保证四个 WG 在切换
persistent task 前都完成当前 task。

## 10. 可手算实例：`batch=96,M=64,N=6144,K=7168`

该 shape 是已有 runner 文档记录的实际使用形状。这里取：

```text
b       = 0
Mtile   = 0        -> M0 = 0
Ntile   = 2        -> N0 = 512, wg_x = 2
q       = 3        -> logical K [768,1024)
```

基本 stride：

```text
P = K/2  = 3584 = 0x0E00 B
S = K/32 =  224 = 0x00E0 B

16*P = 57344 = 0x0E000 B
32*S =  7168 = 0x01C00 B
N*2  = 12288 = 0x03000 B
```

batch stride：

| tensor | byte stride |
|---|---:|
| D | `0x0C0000` |
| A | `0x038000` |
| B | `0x1500000` |
| ScaleA | `0x003800` |
| ScaleB | `0x150000` |

任意 batch `b` 只需在下列相对 offset 上再加对应的
`b*batch_stride`。

### 10.1 四类 load 的实际 descriptor 数值

| 字段 | A | B | ScaleA | ScaleB |
|---|---:|---:|---:|---:|
| `tensor_dim0` | `0xE000` | `0xE000` | `0x1C00` | `0x1C00` |
| `tensor_dim1` | `4` | `(6144-512)/16=352=0x160` | `2` | `(6144-512)/32=176=0xB0` |
| `tile_dim0` | `0x800` | `0x800` | `0x100` | `0x100` |
| `tile_dim1` | `4` | `16` | `2` | `8` |
| `stride0` | `0xE000` | `0xE000` | `0x1C00` | `0x1C00` |
| mask | `0xF` | `0x4` | `0xF` | `0x4` |

由 Group-1 packing 得到的 `s36:s43` 是：

```text
A:
  0020000f e0000000 00040000 08000000
  00000004 0000e000 00000000 00000000

B:
  00200004 e0000000 01600000 08000000
  00000010 0000e000 00000000 00000000

ScaleA:
  0020000f 1c000000 00020000 01000000
  00000002 00001c00 00000000 00000000

ScaleB:
  00200004 1c000000 00b00000 01000000
  00000008 00001c00 00000000 00000000
```

这里的第一项 `0x00200000` 是 `early_timeout`，不是 padding。

### 10.2 Global 地址区间

全部 offset 都相对本 batch 对应 tensor base。

A 的 `q=3` 起始 offset 为 `3*0x800=0x1800`，四条 chunk：

| M rows | Global byte interval |
|---|---|
| `[0,16)` | `[0x01800,0x02000)` |
| `[16,32)` | `[0x0F800,0x10000)` |
| `[32,48)` | `[0x1D800,0x1E000)` |
| `[48,64)` | `[0x2B800,0x2C000)` |

它们共 8192 B，逻辑 packed-K bytes 均为 `[384,512)`。

B 的起点：

```text
N0*P + q*0x800
= 512*0xE00 + 0x1800
= 0x1C1800
```

16 条 chunk 为：

```text
y=0..15:
  [0x1C1800 + y*0xE000,
   0x1C2000 + y*0xE000)
```

最后一条是 `[0x293800,0x294000)`；payload 为 `0x8000`，
逻辑覆盖 `N[512,768)`、packed-K byte `[384,512)`。

ScaleA 两条 chunk：

```text
[0x00300,0x00400)
[0x01F00,0x02000)
```

ScaleB 起点：

```text
N0*S + q*0x100
= 512*0xE0 + 0x300
= 0x1C300
```

8 条 chunk 为：

```text
y=0..7:
  [0x1C300 + y*0x1C00,
   0x1C400 + y*0x1C00)
```

最后一条是 `[0x28700,0x28800)`。ScaleA/ScaleB 的逻辑 scale index
均为 `[24,32)`，对应 K `[768,1024)`。

由于这是首个 task 且 `q=3`，stage 为 slot3，LDS destination 分别是：

| operand | LDS interval |
|---|---|
| A | `[0x06000,0x08000)` |
| ScaleA | `[0x08600,0x08800)` |
| ScaleB | `[0x0A000,0x0A800)` |
| B | `[0x22800,0x2A800)` |

### 10.3 D store 地址

该 WG 的 D Global 起点：

```text
D_b + M0*(N*2) + N0*2
= D_b + 0x400
```

row0 写 `[D_b+0x400,D_b+0x600)`；row1 写
`[D_b+0x3400,D_b+0x3600)`；row63 写
`[D_b+0xBD400,D_b+0xBD600)`。

store descriptor 的 Group-1 `s84:s91`：

```text
00000000 02000000 00400000 02000000
00000040 00003000 00000000 00000000
```

它从 LDS `[0x2A800,0x32800)` 读 32768 B，并写到 64 条相隔
`0x3000` 的 512-B Global row。

## 11. 指令与 byte 计数复核

### 11.1 每 K256 body 的动态有用量

| 单位 | TDM load | descriptor payload | `ds_load_b128` | `ds_load_b32` | WMMA |
|---|---:|---:|---:|---:|---:|
| 每 wave | 1 | 由 specialist 决定 | 32 | 8 | 16 |
| 每 WG | 4 | `0xAA00` | 128 | 32 | 64 |
| 每 4-WG cluster | 16 | `0x2A800`（shader descriptor 口径） | 512 | 128 | 256 |

cluster 内的唯一 Global 地址集合要去掉 A/ScaleA 的四 WG 重复：

```text
unique bytes / cluster / K256
  = A 0x2000 + SA 0x0200
  + 4*(B 0x8000 + SB 0x0800)
  = 0x24200 = 147968 B
```

`0x2A800` 是四个 WG 的 descriptor payload 总和；
`0x24200` 是去重后的地址集合。二者都不是精确 HBM transaction bytes。

### 11.2 `K=7168` 的 useful 总量

```text
Q = 7168 / 256 = 28 bodies
28 = 7 complete P0/P1/P2/P3 cycles
```

每 WG、每 logical task 的 useful load：

| operand | descriptor 数 | useful bytes |
|---|---:|---:|
| A | 28 | `28*0x2000 = 0x38000 = 229376 B` |
| B | 28 | `28*0x8000 = 0xE0000 = 917504 B` |
| ScaleA | 28 | `28*0x0200 = 0x03800 = 14336 B` |
| ScaleB | 28 | `28*0x0800 = 0x0E000 = 57344 B` |
| 合计 | 112 | `0x129800 = 1218560 B` |

每 WG 同时执行：

```text
WMMA            = 28 * 64  = 1792
ds_load_b128    = 28 * 128 = 3584
ds_load_b32     = 28 * 32  = 896
useful TDM load = 28 * 4   = 112
TDM store       = 1 per logical task
```

对该 batch-96 例子，每个物理 cluster 只有一个 logical task。沿一条
specialist wave 的实际控制流：

```text
4 cold-prologue issues + 28 body issues = 32 tensor_load_to_lds

其中：
  28 = 当前 task 的 useful q0..q27
   2 = 末端重复 q0/q1 payload
   2 = tile_dim0=0 的零 payload TDM
```

所以每 WG 实际执行 128 条 load instruction，但 useful descriptor 是 112 条。
若一个物理 cluster 连跑 `T` 个 task，跨 task q0/q1 prefetch 属于下一 task，
fast restart 每次再发 q2/q3；对当前 `Q=28`：

```text
executed load instructions per specialist
  = 4 + T*Q + 2*(T-1)
  = T*Q + 2*T + 2
```

### 11.3 全文件静态 opcode 计数

只读 Python 对当前 `.s` 的行首 mnemonic 计数得到：

| opcode | 静态出现数 |
|---|---:|
| `tensor_load_to_lds` | 56 |
| 其中带 `TH_LOAD_NT` | 20 |
| 其中默认 TH | 36 |
| `tensor_store_from_lds` | 1 |
| `s_wait_tensorcnt` | 24 |
| `s_wait_dscnt` | 50 |
| `s_barrier_signal` | 67 |
| `s_barrier_wait` | 75 |
| `ds_load_b128` | 512 |
| `ds_load_b32` | 128 |
| `ds_store_b128` | 16 |
| `v_wmma_scale_f32_32x16x128_f4` | 128 |
| `s_addk_co_i32 s58,0x100` | 8 |

56 条静态 load 的结构是：

```text
16  首个 task：4 operands * 4 slots
16  s94==1 的完整重建分支；s94 在 L147 固定为 0，正常 fast path 不取
16  fast-restart phase 分发表
 8  2 套 wave schedule * 4 phases 的 steady refill
--
56
```

静态数不能直接当作一次动态执行量；phase、wave specialization 和
persistent restart 都包含互斥分支。

### 11.4 地址集合枚举

对第 10 节实例，把 preshuffle 坐标逐 byte 展开得到：

```text
A:  8192 entries,  8192 unique, rows 0..63,   packed-K 384..511
B: 32768 entries, 32768 unique, rows 512..767, packed-K 384..511
SA:  512 entries,   512 unique, rows 0..63,   scales 24..31
SB: 2048 entries,  2048 unique, rows 512..767, scales 24..31
```

独立展开 epilogue 的
`wave × m_block × store_group × lane × 16 bytes`：

```text
enumerated bytes = 32768
unique bytes     = 32768
minimum          = 0x2A800
maximum excl.    = 0x32800
collisions       = 0
holes            = 0
```

这两组枚举分别交叉验证 load 的 logical tile 与 store 的 LDS 拼接。

## 12. 硬件 transaction 与分析边界

### 12.1 可以从现有资料确定

- descriptor 的维度、stride、mask、flags、payload 和 Global/LDS 地址集合；
- TDM op 忽略 `EXEC`，同 wave 有序、跨 wave 无序；
- 非零 `workgroup_mask` 的 load 使用 cluster async-copy 路径；
- store 忽略 `workgroup_mask`，本目标显式置零；
- `tile_dim0=0` 是 TDM NOP，不产生 tile memory copy；
- MI400 TDM 设计资料的目标能力是每周期生成最多 256-B 连续 chunk，并会按
  alignment 和剩余长度选择内部 payload/address mode；
- 目标 A/B `tile_dim0=2048`、scale `tile_dim0=256`，LDS 地址均至少
  4-B 对齐；若 runtime Global base 至少 128-B 对齐，K/tile offset 会保持
  该对齐，满足编程指南给出的 direct-copy eligibility 条件。

### 12.2 不能由当前文本唯一确定

1. 一条 2048-B 或 256-B descriptor row 最终拆成多少条 128-B/256-B
   GLX request、每条使用哪种内部 address-lane payload。TDM 设计资料给出
   mode 规则，但实际 base alignment、实现版本和 arbitration 仍会影响序列。
2. `workgroup_mask=0xf` 的四个 A/SA requester 实际是否在每个 cache line
   都合并成功。multicast 资料说明首个命中组请求可发往 GL2、后续命中请求
   不再重复发送，但 `early_timeout`、到达时序和 cache 状态决定实际结果。
3. 精确 HBM read/write bytes、GL1/GL2 hit、cache-line replay、压缩或
   partition 行为。descriptor payload 和唯一地址集合都不能替代 PMC。
4. TDM 与 DS/WMMA 的逐 cycle 重叠、每个 `s_wait_tensorcnt` 的实际 stall
   周期。静态 ISA 只给出依赖上界。
5. TDM 内部所谓“lane”是生成 async-copy request 的内部地址槽，不是该
   tensor 指令的 shader lane transaction；不能从 wave32 推导“一 lane
   搬多少 Global byte”。

### 12.3 本地资料的一致性说明

- CDNA5 ISA §10.11 与 MI400 Shader Programming Guide §4.10 对本文使用的
  `data_size`、2D 地址、padding、iteration、mask 和 TENSORcnt 语义一致。
- `mi400_tx_api#65.txt` 把同一 16-bit 字段称为 `cluster_mask`，CDNA5 ISA
  称为 `workgroup_mask`；bit 位置和用途一致，只是术语不同。
- MI400 Tensor DMA 内部资料比 programming guide 给出更多 request-mode
  细节；本文只用它界定可能的拆分方式，不把某一种内部 transaction 序列
  冒充为目标运行时事实。

## 13. 资料依据与 ISA 锚点

目标与软件布局：

- `my_code/f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.s`
  - batch base：L18-L56
  - wave/WG/persistent 坐标：L71-L320
  - A descriptor：L321-L392
  - B descriptor：L633-L705
  - ScaleA descriptor：L945-L1016
  - ScaleB descriptor：L1257-L1329
  - 首个 task 四槽 load：L398-L1557 的四条 wave 分支
  - persistent descriptor rebuild/fast restart：L1570-L2714
  - K256 hotloop：L2848-L3580
  - output LDS staging 与 TDM store：L3600-L3758
  - task 切换与 descriptor 复位：L3759-L3872
  - LDS metadata：L3880-L4038
- `aiter/ops/shuffle.py`
  - `shuffle_scale_f4`：L292-L315
  - `shuffle_weight_f4`：L318-L335
- `my_code/isa_runner/gemm_batch_isa_runner.py`
  - batch-major shape 与 stride 构造
- `my_code/f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.doc.md`
  - 用于交叉核对 batch/persistent 几何和 epilogue 背景；本文的 descriptor
    数字仍以目标 `.s` 为准

硬件资料：

- `mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt`
  - §10.11.1 Tensor Instructions
  - §10.11.2 Tensor and Tile Addressing
  - §10.11.3 Tensor Instruction Variants and Optional Operations
  - §10.11.4 Tensor DMA Descriptor
  - scalar wait 与 barrier 章节
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`
  - §4.10.1-§4.10.4
  - §4.10.8 Performance and Tracking
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/TX/mi400_tensor_dma#72.txt`
  - §5.1 Tensor Inflight Limitations
  - §7.1 constants
  - §7.6 descriptor scheduling 与 address generation
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/TX/mi400_tx_api#65.txt`
  - descriptor bitfield 表
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/subsystem/CMM/GLX/Design/MI400_Multicast_Feature#9.txt`
  - multicast requester 与 GL1→GL2 request 合并行为
