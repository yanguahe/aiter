# gfx1250 MXFP4xMXFP4 64x256_1x4 Persistent GEMM Design

<!-- markdownlint-disable MD013 MD033 MD060 -->

<a id="toc"></a>

## Table of Contents

- [1. Wave Tile and TDM Specialization](#section-1-wave-tile)
  - [1.1 Four Waves in a 1x4 N Strip](#section-1-1-wave-output-strip)
  - [1.2 Four-Slot LDS Ring](#section-1-2-lds-ring)
  - [1.3 Final Kernel Contract](#section-1-3-final-kernel-contract)
- [2. 几何结构与 Batch-Z 持久化调度](#section-2-cluster-grid)
  - [2.1 wave 条带与 workgroup cluster 的区别](#section-2-1-wave-vs-cluster)
  - [2.2 逻辑 X/Y 任务空间与形状约束](#section-2-2-logical-grid)
  - [2.3 物理持久化 X/Y 映射](#section-2-3-persistent-launch)
  - [2.4 Batch-Z 平面与基址指针](#section-2-4-batch-z-pointers)
  - [2.5 同步、隔离与 batch=1 兼容](#section-2-5-isolation-compatibility)
  - [2.6 启动实例](#section-2-6-launch-examples)
- [3. End-to-End Software Pipeline](#section-3-software-pipeline)
  - [3.1 wave0/2: `B-current -> A-current -> A-next -> B-next`](#section-3-1-wave02-flow)
  - [3.2 wave1/3: `A-current -> B-current -> B-next -> A-next`](#section-3-2-wave13-flow)
- [4. P0 Detailed Pipeline and Wave-Local Tile Coverage](#section-4-p0-details)
  - [4.1 wave0/2 Specialization Schedule](#section-4-1-wave02-details)
    - [4.1.1 wave0/2 Wave-Local Tile Coverage](#section-4-1-1-wave02-host-tile)
  - [4.2 wave1/3 Specialization Schedule](#section-4-2-wave13-details)
    - [4.2.1 wave1/3 Wave-Local Tile Coverage](#section-4-2-1-wave13-host-tile)
- [5. Epilogue Design and Final Store Contract](#section-5-epilogue-design)
  - [5.1 Output Geometry and Fragment Mapping](#section-5-1-output-geometry)
  - [5.2 Candidate Physical VGPR Layout](#section-5-2-vgpr-layout)
  - [5.3 BF16 Conversion and LDS Staging](#section-5-3-bf16-lds-staging)
  - [5.4 Required TDM-Store Contract and Synchronization](#section-5-4-tdm-store-sync)
  - [5.5 Final Drain, P3 Wrap, and Persistent-Task Transition](#section-5-5-final-drain-transition)
  - [5.6 Resource and Validation Checklist](#section-5-6-resource-validation)
- [6. Cluster TDM Multicast](#section-6-cluster-tdm-multicast)
  - [6.1 WG Bit Matrix](#section-6-1-wg-bit-matrix)
  - [6.2 Operand Multicast Masks](#section-6-2-operand-multicast-masks)
  - [6.3 Payload and Cluster Coverage](#section-6-3-payload-cluster-coverage)

<a id="section-1-wave-tile"></a>

## 1. Wave Tile and TDM Specialization

<a id="section-1-1-wave-output-strip"></a>

### 1.1 Four Waves in a 1x4 N Strip

Each wave retains the same `64x64` compute tile. The four waves are arranged
as one host-M wave by four host-N waves, so the WG output is exactly
`64x256`. All four waves have local M origin zero; logical wave `w` has local
N origin `64*w`.

| Wave | WG-relative M | WG-relative N | N-origin formula | TDM specialization |
|---:|---|---|---:|---|
| 0 | `[0,63]` | `[0,63]` | `64*0 = 0` | host A data |
| 1 | `[0,63]` | `[64,127]` | `64*1 = 64` | host B data |
| 2 | `[0,63]` | `[128,191]` | `64*2 = 128` | SA |
| 3 | `[0,63]` | `[192,255]` | `64*3 = 192` | SB |

For logical WG tile indices `(Mtile,Ntile)` and logical wave ID `w`:

```text
Mbase(w) = 64 * Mtile
Nbase(w) = 256 * Ntile + 64 * w
```

Thus wave ID contributes only to N. There is no wave-ID bit that selects a
high-M half in this design.

Each wave still owns eight `32x16` hardware output fragments and executes
16 WMMA instructions per K256 body: two K128 accumulations for each of eight
fragments. The wave0/2 and wave1/3 schedules in Chapters 3 and 4 are retained
only as alternative software schedules associated with TDM specialization.
They do not denote low-M/high-M wave pairs. Every wave covers the same local
M64 and a distinct N64 quarter.

<a id="section-1-2-lds-ring"></a>

### 1.2 Four-Slot LDS Ring

For one WG and one K256 body, the logical TDM payloads are:

| Operand | Payload derivation | Per-slot payload |
|---|---|---:|
| A data | `M64 * K256 * 4 bits` | `0x2000 = 8192 B = 8 KiB` |
| SA | `M64 * (K256 / K32) * 1 B` | `0x0200 = 512 B` |
| B data | `N256 * K256 * 4 bits` | `0x8000 = 32768 B = 32 KiB` |
| SB | `N256 * (K256 / K32) * 1 B` | `0x0800 = 2048 B = 2 KiB` |

The following correctness-first candidate packs the four input arrays
contiguously, keeps a `0x8000` stride between B slots, and reserves a separate
output region. It inserts no explicit layout gap:

| Operand | slot0 | slot1 | slot2 | slot3 | Array end |
|---|---:|---:|---:|---:|---:|
| A data | `0x00000` | `0x02000` | `0x04000` | `0x06000` | `0x08000` |
| SA | `0x08000` | `0x08200` | `0x08400` | `0x08600` | `0x08800` |
| SB | `0x08800` | `0x09000` | `0x09800` | `0x0A000` | `0x0A800` |
| B data | `0x0A800` | `0x12800` | `0x1A800` | `0x22800` | `0x2A800` |

The resulting fixed LDS allocation is:

```text
input ring [0x00000,0x2A800)     = 0x2A800 = 170 KiB
output staging [0x2A800,0x32800) = 0x08000 =  32 KiB
candidate fixed LDS end           = 0x32800 = 202 KiB
```

CDNA5 permits up to 320 KiB of LDS for a WG (CDNA5 ISA Section 2.2, local
text L743-L748), so 202 KiB is within the documented architectural capacity.
It can still reduce residency, and the target object must request the full
`0x32800` group segment.

The semantic TDM-load row counts follow the non-K dimensions. In byte mode,
one packed-data row is `K256/2 = 0x80` bytes and one scale row is
`K256/K32 = 0x08` bytes. This gives the following unassembled descriptor
candidate:

| Operand | Logical rows | Candidate row width | Candidate semantic tile | Payload |
|---|---:|---:|---:|---:|
| A data | 64 | `0x80` B | `tile_dim0=0x80, tile_dim1=64` | `0x2000` |
| SA | 64 | `0x08` B | `tile_dim0=0x08, tile_dim1=64` | `0x0200` |
| B data | 256 | `0x80` B | `tile_dim0=0x80, tile_dim1=256` | `0x8000` |
| SB | 256 | `0x08` B | `tile_dim0=0x08, tile_dim1=256` | `0x0800` |

CDNA5 defines tile dimensions in `data_size` units (Section 10.11.2 and
Section 10.11.4, local text L10215-L10231 and L10487-L10514). The table fixes
the target's logical row counts and payloads, but it does not claim encoded
descriptor words. Whether the target can expose the semantic rows directly
or must regroup the same bytes depends on the AB-preshuffle layout and
strides. Descriptor packing, legal dimensions, B-base/TDM destination
alignment, and bank behavior are target-assembly and hardware validation
boundaries.

For ring slot `s`, every wave uses the same A and SA base because there is no
wave-M offset. Each wave consumes its own local N64 quarter of the WG-wide B
and SB payload:

```text
A_wave_base(s,w)  = A_slot_base[s]
SA_wave_base(s,w) = SA_slot_base[s]
B_wave_base(s,w)  = B_slot_base[s]  + w * 0x2000
SB_wave_base(s,w) = SB_slot_base[s] + w * 0x0200
```

The offsets are respectively one N64 FP4 payload
(`64*256/2 = 0x2000`) and one N64 scale payload
(`64*(256/32) = 0x200`). A/SA have no wave-M high-half offset. No formula in
this target may use `(wave_id & 1)` for A/SA or `(wave_id >> 1)` for B/SB.
The base arithmetic is exact; alignment, DS access legality, and bank-conflict
behavior must be measured on the eventual target ISA.

<a id="section-1-3-final-kernel-contract"></a>

### 1.3 Final Kernel Contract

The following decisions are hard preconditions for this documentation design.
No corresponding assembled target ISA exists yet.

| Property | Final decision |
| --- | --- |
| Kernel symbol basename | `f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps` |
| Wave/WG geometry | Wave `64x64`; wave grid `(host M, host N) = (1,4)`; WG `64x256`. |
| Cluster geometry | `(cluster_y/M, cluster_x/N) = (1,4)`; four WGs per cluster. |
| M precondition | `M % (WG_M * cluster_M) = M % (64*1) = M % 64 == 0`. |
| N precondition | `N % (WG_N * cluster_N) = N % (256*4) = N % 1024 == 0`. |
| K precondition | `K % 256 == 0`; full K256 bodies only. |
| Boundary/tail policy | No M/N boundary tile, K tail, or partial cluster. |
| Epilogue store | One 64-row by 256-BF16 `tensor_store_from_lds` per WG, issued by wave0. |
| Candidate fixed LDS | Input ring `[0x00000,0x2A800)` plus output `[0x2A800,0x32800)`, ending at `0x32800 = 202 KiB`; no explicit layout gap. |

Accepted inputs therefore satisfy:

```text
M % 64   == 0
N % 1024 == 0
K % 256  == 0
```

The logical WG grid `(N/256, M/64)` must divide exactly into clusters of four
WGs along N and one WG along M. Chapter 5 specifies the 64-row output store.
Its width, descriptor encoding, and LDS layout remain implementation
validation work; a historical 256x256 reference ISA provides migration
evidence only for the 64-row-height idiom.

<a id="section-2-cluster-grid"></a>

## 2. 几何结构与 Batch-Z 持久化调度

本章描述的是**已经实现**的 `_batch_ps` ISA
(`f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.s`)与
`gemm_batch_isa_runner.py`,不是待选方案。本章中的行号 `Lxxxx` 均指该 `.s`
文件的行号。

**一句话总结.** batch 版本原封不动地保留了非 batch 版本在 X/Y 平面上的
cluster 持久化遍历,只是在物理 grid 的 Z 方向为每个矩阵多铺一层平面。整个
batch 化改动只有 kernel 入口处一次性的"五个基址指针各加一个偏移";之后的
描述符构造、K 流水、epilogue、持久化重启全部沿用原有指令流。

**本章速读.**

| 想知道的事 | 答案 | 小节 |
| --- | --- | --- |
| kernel 名字里的 `1x4` 指什么 | 一个 WG 内 4 个 wave 沿 N 排开,与 cluster 形状无关 | 2.1 |
| 一个 cluster 产出多大输出 | `M64 x N1024`(4 个 WG,每个 `M64 x N256`) | 2.1 |
| 一共有多少个调度任务 | `T_XY = (M/64) * (N/1024)`,每个 Z 平面各有一份 | 2.2 |
| 实际启动多少个 cluster | `min(T_XY, 64)` 个,每个 cluster 循环领任务 | 2.3 |
| batch 号从哪里来 | 物理 grid 的 Z 坐标,即 `TTMP7[31:16]` | 2.4 |
| 不同 batch 之间会互相等待吗 | 不会,所有同步都封闭在单个 cluster 内部 | 2.5 |

**全章符号.**

```text
M, N, K    矩阵维度
W_M, W_N   逻辑 WG 网格      = (M/64, N/256)
C_M, C_N   逻辑 cluster 网格 = (M/64, N/1024)
T_XY       单个 Z 平面内的 cluster 任务总数 = C_M * C_N
p          一个物理 cluster 领到的首个任务号
S          持久化递推步长,由 ABI 编码,恒为 2 的幂
b          batch 号,等于物理 grid 的 Z 坐标
```

<a id="section-2-1-wave-vs-cluster"></a>

### 2.1 wave 条带与 workgroup cluster 的区别

kernel 名字里的 `1x4` 描述的是**一个 workgroup 内部**四个 wave 的输出分工,
不是 workgroup cluster 的形状。两层恰好在 N 方向上都是 4,但它们是硬件层级
里不同的两层,必须分开看:

| 层级 | 实现的形状 | 覆盖的输出 |
| --- | --- | --- |
| 一个 wave | 一个 wave32,一块 `64x64` 输出 tile | `M64xN64` |
| 一个 WG | 4 个 wave,排布 `(M,N)=(1,4)`;HIP block `(128,1,1)` | `M64xN256` |
| 一个 workgroup cluster | HIP `clusterDim=(x=N,y=M,z)=(4,1,1)`;4 个 WG | 单个 Z 平面内的 `M64xN1024` |

所以一个 cluster = 4 个 WG = 16 个 wave:

```text
一个 cluster task 覆盖 M64 x N1024:

  N 方向 ------------------------------------------------------->
  +----------------+----------------+----------------+----------------+
  | WG wg_x=0      | WG wg_x=1      | WG wg_x=2      | WG wg_x=3      |
  | N[0:255]       | N[256:511]     | N[512:767]     | N[768:1023]    |
  | w0 w1 w2 w3    | w0 w1 w2 w3    | w0 w1 w2 w3    | w0 w1 w2 w3    |
  +----------------+----------------+----------------+----------------+
  每个 w 是一个 wave,占 N64;四个 wave 的 M 起点都是 0(M 方向只有 64 行)
```

WG 内部,wave `w` 拥有 WG 相对 N 区间 `[64*w, 64*w+63]`;cluster 内部,
`wg_x` 不同的 WG 拥有不同的 `N256` tile。完整的输出原点是:

```text
Mbase = 64 * Mtile
Nbase = 256 * Ntile + 64 * wave_id
```

**硬件文档事实.** CDNA5 定义的层级是
`Grid -> Cluster -> Workgroup -> Wave -> Work-item`;一个 cluster 的所有 WG
调度在同一个 shader engine 上,可以使用 cluster barrier 和 multicast load
(CDNA5 ISA 2.3-2.3.1 节,手册第 10-11 页,本地文本 L751-L815)。MI400
Shader Programming Guide 独立给出了同样的层级以及 TTMP6/TTMP7/TTMP9 的
cluster 状态布局(2.3-2.3.1 节,手册第 24-25 页,本地文本 L1699-L1766)。
扩展 compute AQL packet 也区分了"以 cluster 为单位的三维 grid 计数"和
"以 WG 为单位的三维 cluster 尺寸"(`CLUSTER_COUNT_*` 对 `CLUSTER_SIZE_*`;
packet reference 第 12 页,本地文本 L599-L649)。workgroup-cluster 调度资料
同样指出:grid 在 cluster 意义下可以是三维、cluster 在 WG 意义下可以是三维,
且**不支持残缺 cluster**(Workgroup Clusters Scheduling 第 4 节,第 4 页,
本地文本 L277-L304)。

**ISA 证据.** kernel 入口没有把 `4` 写死,而是直接从 TTMP 读硬件真值:

| ISA 指令(行号) | 取到的位域 | 含义 |
| --- | --- | --- |
| `s_bfe_u32 s22, ttmp8, 0x50019`(L58/L79) | `ttmp8[29:25]` | `wave_id`,WG 内 wave 号 `0..3` |
| `s_bfe_u32 s49, ttmp6, 0x40000`(L83) | `ttmp6[3:0]` | `wg_x`,cluster 内 WG 的 X 号 `0..3` |
| `s_bfe_u32 s50, ttmp6, 0x40004`(L82) | `ttmp6[7:4]` | `wg_y`,本设计恒为 `0` |
| `s_bfe_u32 s51, ttmp6, 0x4000c` 再 `+1`(L81/L85) | `ttmp6[15:12]+1` | `nwg_x = clusterDim.x = 4` |
| `s_bfe_u32 s52, ttmp6, 0x40010` 再 `+1`(L80/L84) | `ttmp6[19:16]+1` | `nwg_y = clusterDim.y = 1` |

这些位域含义与上述硬件文档一致(TTMP8 的 `waveIDinGroup[4:0]` 见 CDNA5 ISA
本地文本 L2146-L2152)。2.2 会用到这一点:`C_N` 的除数 `1024` 不是常量,而是
由读回来的 `nwg_x` 现算出来的。

**实现推断.** runner 和 ISA 把 cluster X 分配给 N、cluster Y 分配给 M,所以
实现出来的 cluster 输出足迹是 `(1*64)x(4*256) = M64xN1024`。cluster Z 只有
一层 WG 深,不承担任何输出范围;软件把 cluster 网格的 Z 坐标当作 batch
选择器使用。

<a id="section-2-2-logical-grid"></a>

### 2.2 逻辑 X/Y 任务空间与形状约束

逻辑任务空间分两步切出来:

```text
第一步:按 WG tile 切分输出
  W_N = N / 256                              # N 方向的 WG 个数
  W_M = M / 64                               # M 方向的 WG 个数

第二步:沿 N 每 4 个 WG 打包成一个 cluster task
  C_N = W_N / clusterDim.x = N / (256*4) = N / 1024
  C_M = W_M / clusterDim.y = M / (64*1)  = M / 64

  T_XY = C_N * C_M                           # 一个调度平面内的 cluster 任务数
```

非 batch kernel 的逻辑 WG 网格是 `(W_N,W_M,1)`,逻辑 cluster 网格是
`(C_N,C_M,1)`。batch runner 报告的是 `(W_N,W_M,batch)` 与 `(C_N,C_M,batch)`,
但追加的这个逻辑 Z 计数**不会改变 `T_XY`**:每个 Z 平面各自拥有一份 `T_XY`
任务计数。在展平后的 cluster 任务编号里,N 是变化最快的维度。

**ISA 证据.** `T_XY` 由 kernel 自己算,除数来自 2.1 从 TTMP6 读到的
`nwg_x/nwg_y`:

```text
L92-L97  : sh   = ctz(nwg_x) + 8            # nwg_x=4  -> 1<<sh = 1024
           C_N  = (N + (1<<sh) - 1) >> sh   # s61
L98-L103 : sh   = ctz(nwg_y) + 6            # nwg_y=1  -> 1<<sh = 64
           C_M  = (M + (1<<sh) - 1) >> sh
L104     : T_XY = C_M * C_N                 # s29
```

其中 `+8` 是 `log2(256)`、`+6` 是 `log2(64)`,即 WG tile 的 N/M 尺寸。由于
下表强制 `N % 1024 == 0`、`M % 64 == 0`,这里的向上取整永远等于精确整除。

runner 强制的启动域:

| 量 | 实现的约束 | 结果 |
| --- | --- | --- |
| `M` | 正 uint32,且 `M % 64 == 0` | 全部是完整 `M64` WG tile;`clusterDim.y=1` 不带来更强的整除要求 |
| `N` | 正 uint32,且 `N % 1024 == 0` | 全部是完整 `N256` WG tile,且 4 个 N 方向 WG 正好凑满一个 cluster |
| `K` | 正 uint32,且 `K % 256 == 0` | 全部是完整 K256 主体,同时满足 K32 scale 与 K128 scale-shuffle 的前提 |
| `batch` | 整数 `1 <= batch <= 65535` | 遵守 runner 的 HIP grid-Z 上限;实际启动的最大 batch 号是 65534 |
| M/N/K 尾块 | 无 | 不构造边界 WG、残缺 cluster 或 K 尾巴 |

`C_N` **不要求**是 2 的幂:`C_N` 是 2 的幂时 ISA 走移位/掩码路径,否则走整数
除法路径。当 `T_XY` 较小且不是 2 的幂时,物理 cluster 网格 X 也会是非 2 的幂,
而独立编码的递推步长仍是 2 的幂 —— 2.3 会解释为什么这样安全。

ABI 另有两项非整除检查。五个继承下来的行 stride(`N*2`、`K/2`、`K/2`、
`K/32`、`K/32`)必须放得进 uint32;每个新增的 batch stride 是 uint64、为正,
且 runner 保守地要求 `batch_stride * batch <= 2^64-1`。

还有一处继承下来的校验缺口:ISA 用 32 位标量寄存器保存 `T_XY`、持久化任务号
以及矩阵内部的 tile 原点字节运算,而 runner 并没有显式证明这些派生值不会回绕。
由于编码步长 `S <= 64`,一个充分的调度器上界是 `T_XY <= 2^32-64`,这样最后一次
`task+S` 前瞻也放得下。同样地,A/B/sA/sB/D 的矩阵内 tile 原点偏移都必须放得进
uint32,一个简单的保守条件是每个单矩阵的字节跨度不超过 `2^32`。这些是实现安全
上限,不是新增的整除规则,所要求的 `64x65536x32768` 例子满足它们。新增的 64 位
batch 偏移**不会**把继承下来的矩阵内寻址变成 64 位。

<a id="section-2-3-persistent-launch"></a>

### 2.3 物理持久化 X/Y 映射

这一节分五步来看:先决定发多少个 cluster,再让每个 cluster 算出自己的起始
任务号,然后把任务号翻译成 tile 坐标,接着按固定步长往后领任务,最后论证
覆盖的完备性。

#### 第 1 步:runner 决定启动多少个物理 cluster

runner 根据逻辑任务数 `T_XY` 选择物理 seed 数 `C`,并**独立地**编码递推
步长 `S`:

| 情况 | 物理 cluster 网格 `(x,y)` | 物理 seed 数 `C` | 编码 `log2_grid_(x,y)` | 递推步长 `S` |
| --- | --- | ---: | --- | ---: |
| `T_XY <= 64` | `(T_XY, 1)` | `T_XY` | `(log2(S), 0)` | `next_pow2(T_XY)` |
| `T_XY > 64` | `(16, 4)` | `64` | `(4, 2)` | `64` |

```text
physical HIP WG grid  = (cluster_grid_x*4, cluster_grid_y, batch)
physical cluster grid = (cluster_grid_x, cluster_grid_y, batch)
clusterDim            = (4,1,1)
block                 = (128,1,1)
编码递推步长 S        = 1 << (log2_grid_x + log2_grid_y)
```

**实现推断.** 对**非 2 的幂且 `T_XY <= 64`** 的情况,物理启动规模与编码递推
步长是**故意不一致**的:实际只启动 `T_XY` 个单行 seed cluster,而沿用的旧
log2 ABI 编码的是向上取到的 2 的幂 `S`。`T_XY > 64` 时,已验证过的 `(16,4)`
拓扑、64 个 seed、步长 64 保持不变。每个物理 cluster 含 4 个 WG,所以一个
平面共启动 `4*C` 个 WG。

**启动接口核对.** CDNA5 ISA 与 MI400 指南把 cluster 网格 X 定义为独立的
32 位 `TTMP9`、cluster 网格 Y/Z 放在 `TTMP7`;它们**没有**定义 kernel 的
`log2_grid_x/y` 这一软件 ABI 字段。在 runner 里,HIP 的 `gridDimX/Y/Z` 和
`clusterDim` 由物理 `LaunchGeometry.grid` 与 `.cluster` 填充,而两个 log2 值
单独打进 kernarg。`LaunchGeometry` 和 HIP 配置构造器都**不要求**物理 cluster
数等于 `1 << (log2_grid_x+log2_grid_y)`。

#### 第 2 步:每个 cluster 算出自己的起始任务号 `p`

对一个物理 cluster,gfx1250 在 `TTMP9` 给出 cluster 网格 X、在 `TTMP7[15:0]`
给出 cluster 网格 Y、在 TTMP6 给出 cluster 内 WG 的 X/Y 以及各维尺寸减一。
按实现的 `clusterDim=(4,1,1)` 解码:

```text
cx   = TTMP9
cy   = TTMP7 & 0xffff
wg_x =  TTMP6       & 0xf              # 0..3
wg_y = (TTMP6 >> 4) & 0xf              # 恒为 0

p = cx + (cy << log2_grid_x)           # L86-L88,起始 X/Y cluster 任务号
```

等价地,对物理 HIP WG 坐标 `(gx,gy,gz)` 就是 `cx=gx/4`、`wg_x=gx%4`、
`cy=gy`、`wg_y=0`。同一个 cluster 内的 4 个 WG 拿到**相同的 `p`** 并同步
推进;`wg_x` 只用来在这个 cluster task 内部选出 4 个逻辑 N tile 中的一个。

#### 第 3 步:把任务号翻译成 tile 坐标

```text
for task = p; task < T_XY; task += S:
    cluster_n = task % C_N              # N 是最快维
    cluster_m = task / C_N

    Ntile = cluster_n * nwg_x + wg_x    # nwg_x = 4
    Mtile = cluster_m * nwg_y + wg_y    # nwg_y = 1,wg_y = 0

    wave_M_origin = Mtile * 64
    wave_N_origin = Ntile * 256 + wave_id * 64
```

对应到 ISA:入口处若 `p >= T_XY` 立即结束(L148-L149);分解 `task` 时,
`C_N` 是 2 的幂走移位/掩码(L150-L158),否则走整数除法(L159-L186);
随后得到 `Mtile -> s55`、`Ntile -> s54`(L188-L191)。

#### 第 4 步:`+S` 递推与一次任务前瞻

源码实现的是**一次任务前瞻**:处理当前任务的同时,先把 `p+S` 也映射出来
(`Mtile_next -> s69`、`Ntile_next -> s68`),并记下"当前任务是不是最后一个"
(`s60`,L192-L197)。当前任务的 epilogue 结束后,任务切换按如下顺序进行:

```text
1. WG barrier,再由每个 WG 的 wave0 发 cluster barrier,全体等待  (L3751-L3758)
2. 每个 wave 把输出指针按当前任务的精确字节原点减回去             (L3759-L3769)
3. 若 s60 表示没有下一个任务,则 s_endpgm                          (L3770-L3771)
4. 否则把前瞻坐标提升为当前坐标                                    (L3772-L3774)
5. 从"已按 batch 平面调整过的基址"重建 A/B/sA/sB 描述符            (L3775-L3790)
6. 给 D 指针加上新任务的字节原点,并复位 K 计数器                  (L3791-L3802)
7. 再形成一次 p+S 前瞻,然后按 wave_id 跳回各自的流水入口          (L3803-L3872)
```

上面第 2 条的"减回去"是必需的:`D` 是绝对指针,每个任务都会在它上面叠加
自己 tile 的字节原点,所以换任务之前必须精确退回。其字节量为:

```text
D_delta(Mtile,Ntile,wave_id)
    = ((Mtile*64)*N + Ntile*256 + wave_id*64) * sizeof(BF16)

D_wave_ptr -= D_delta(current)          # 换任务前
D_wave_ptr += D_delta(next)             # 换任务后
```

ISA 里 `(Mtile*64)*N*2` 是用预加载的行 stride `strideD0 = N*2`(`s12`)乘出来
的,而不是重新算 `N*2`。

#### 第 5 步:为什么覆盖既完备又不重复

- **`T_XY <= 64`**:物理 Y 只有一行,所以 `cy=0`,起始式退化为 `p = cx`,与
  `log2_grid_x` 无关。X 方向恰好启动 `cx = 0..T_XY-1`,因此**每个 seed 都有
  用**。又因为 `S >= T_XY`,所有 `p+S >= T_XY`,**每个 cluster 跑完一个任务
  就退出**,递推步长根本轮不到用第二次。这就是"物理 X 数量不是 2 的幂也安全"
  的原因;反过来说,如果需要跑第二轮,这个组合就不再是完整覆盖了。
- **`T_XY > 64`**:物理 X/Y 是 `(16,4)`,编码 log2 X/Y 是 `(4,2)`,于是
  `p = cx + 16*cy` 恰好覆盖 `[0,64)`。任意非负 task 都有唯一分解
  `task = p + k*64`(`0 <= p < 64`),所以这 64 条持久化序列不重不漏地覆盖
  全部逻辑任务。

因此物理 cluster `p` 恰好访问 `p, p+S, p+2S, ... < T_XY`,不会抢走其他物理
cluster 的任务。ISA 并没有把 `64` 写死在这个映射里:它用 ABI 的
`log2_grid_x` 和 `log2_grid_y` 同时构造起始 `p` 和
`1 << (log2_grid_x+log2_grid_y)` 递推步长,runner 的静态契约检查会核对这些
指令依赖。

<a id="section-2-4-batch-z-pointers"></a>

### 2.4 Batch-Z 平面与基址指针

**硬件文档事实.** 在 cluster 模式下,gfx1250 把 cluster 网格 Z 暴露在
`TTMP7[31:16]`、Y 在 `TTMP7[15:0]`、X 在 `TTMP9`,cluster 内 WG 的 X/Y/Z 以及
cluster 各维尺寸放在 `TTMP6`(CDNA5 ISA 2.3.1 节,本地文本 L769-L815)。
计算着色器初始化表进一步说明 TTMP7 保存 grid Z/Y,并在 grid Y/Z 使能时加载
(CDNA5 ISA 3.5.3.1 节,手册第 29-30 页,本地文本 L2121-L2166;MI400 Shader
Programming Guide 3.5.5.1 节,手册第 52-53 页,本地文本 L3693-L3739)。

runner 的启动是:

```text
physical HIP WG grid  = (cluster_grid_x*4, cluster_grid_y, batch)
clusterDim            = (4,1,1)
physical cluster grid = (cluster_grid_x, cluster_grid_y, batch)
```

因为 `clusterDim.z=1`,每个 cluster 的 `wg_z=0`,它的 cluster 网格 Z 坐标也就
等于物理 WG 的 Z 坐标。所以 batch ISA 直接实现为:

```text
batch_id = TTMP7[31:16] = TTMP7 >> 16      # L25
```

把这个值称作 batch ID 是**软件层面的设计推断**,不是硬件定义:硬件文档只把它
定义为 cluster 网格 Z。它之所以成为 batch ID,是因为 runner 恰好为每个矩阵
编排了一层 cluster、并且 cluster 深度为 1。

**batch 故意不并入原来的 X/Y 任务计数器.** 起始任务只用 `TTMP7[15:0]` 和
`TTMP9`,`T_XY` 里只有 `C_N*C_M`,`+S` 递推里没有任何 Z 项。TTMP7 的高半部分
只被下面这段指针前缀消费。于是每个 Z 平面互不影响地遍历同一套 X/Y 任务号和
逻辑 M/N tile。

120 字节的 batch ABI 保留了原来的 80 字节前缀,并追加五个小端 uint64 字节
stride:

| 调整后的指针 | 预加载 SGPR / ABI 偏移 | 连续 batch-major 布局 | 字节 stride |
| --- | --- | --- | ---: |
| 输出 `D`(runner 里叫 `C`) | `s[2:3]`;stride `s[22:23]` @ 80 | BF16 `[batch,M,N]` | `M*N*2` |
| `A` | `s[4:5]`;stride `s[24:25]` @ 88 | uint8 packed FP4 `[batch,M,K/2]` | `M*(K/2)` |
| `B` | `s[6:7]`;stride `s[26:27]` @ 96 | uint8 packed FP4 `[batch,N,K/2]` | `N*(K/2)` |
| `sA` | `s[8:9]`;stride `s[28:29]` @ 104 | uint8 E8M0 `[batch,M,K/32]` | `M*(K/32)` |
| `sB` | `s[10:11]`;stride `s[30:31]` @ 112 | uint8 E8M0 `[batch,N,K/32]` | `N*(K/32)` |

runner 严格要求这些形状、dtype 和"外层 batch 连续"的 stride。AB 预 shuffle
改变的是原 kernel 在矩阵**内部**的访问解释,不改变被用作 batch stride 的矩阵
连续字节跨度。

对每个指针,记 `stride = stride_hi*2^32 + stride_lo`、`b = batch_id`,入口前缀
做一次完整的两肢(two-limb)乘法(L26-L55):

```text
offset_lo = low32(b * stride_lo)                              # s_mul_i32
offset_hi = low32(high32(b * stride_lo) + b * stride_hi)      # s_mul_hi_u32
offset    = offset_lo + (offset_hi << 32)
base_b    = base_0 + offset                                   # s_add_nc_u64
```

主机侧的地址跨度检查保证这是精确的无符号 64 位乘积,而不是有意的回绕。

**为什么这段必须放在 kernel 最前面.** 五个 stride 预加载在 `s22..s31`,而这些
寄存器紧接着就被复用作别的用途 —— 例如 L58 就把 `s22` 覆盖成了 `wave_id`。
因此这次指针调整只发生一次,且必须早于保存 `D` 基址(L56-L57)和任何输入
描述符的构造。此后所有 current/next 描述符生成、K 流水、输出 store、`D` 回退
以及持久化重启,都原封不动地运行在这五个已按 batch 调整过的基址之上。

<a id="section-2-5-isolation-compatibility"></a>

### 2.5 同步、隔离与 batch=1 兼容

**硬件文档事实.** workgroup barrier 同步的是一个 WG 内的所有 wave。cluster
barrier(`-3`)统计的是一个 cluster 内的 WG 数,等成员 WG 都发过信号后释放
该 cluster 的全部 wave。文档明确推荐的用法是:先用 workgroup barrier 同步,
再由**每个 WG 出一个 wave** 发 cluster barrier 信号,cluster 内所有 wave 都
必须 wait(CDNA5 ISA 5.6.6 节,手册第 50-51 页,本地文本 L3319-L3367;MI400
Shader Programming Guide 4.3.6.6 节,手册第 82-83 页,本地文本 L5364-L5412)。
multicast mask 同样只能点名**同一个 cluster 内**的 WG(CDNA5 ISA 10.7 节,
手册第 134-135 页,本地文本 L9880-L9919;MI400 Shader Programming Guide
4.9.8 节,手册第 190-191 页,本地文本 L13779-L13814)。

**ISA 证据.** 任务切换处的序列正是文档推荐的模式(L3751-L3758):

```text
s_barrier_signal -1 / s_barrier_wait 0xffff    # 先做 WG 内同步
if wave_id == 0: s_barrier_signal -3           # 每个 WG 只由 wave0 发 cluster 信号
s_barrier_wait 0xfffd                          # cluster 内所有 wave 一起等
```

**实现推论.** 因为 `clusterDim.z=1`,一个 cluster 永远不会同时包含两个 Z 平面
的 WG。已有的 WG barrier、cluster barrier 以及 A/SA/B/SB multicast mask 因此
天然局限在单个 batch 平面内。不存在 grid 级 barrier、跨 Z 的 multicast 位、
共享 LDS 分配或跨 batch 指针。不同 batch 可以并发推进、进度不同,彼此既不需要
同步,也不共享操作数负载。

**`batch=1` 的情形.** 唯一的 Z 坐标是 0,五个 64 位乘积全部精确为零,所以前缀
执行完之后 C/D、A、B、sA、sB 的基址与原非 batch 指针**逐位相同**。描述符、
每个任务的 tile 映射、流水、barrier、epilogue、指针回退和终止逻辑都还是原来的
指令流。batch 符号仍然会执行那段零偏移前缀、仍然消费 120 字节 ABI,而它的物理
X/Y 调度是自适应的:`T_XY <= 64` 时只启动确实有用的那些起始任务号,并且可能
编码一个比物理 seed 数更大的递推步长。另一条路是:batch runner 也允许在
`batch=1` 时直接使用原来的非 batch 符号,那条兼容路径完整保留原有的固定几何、
80 字节 ABI 和执行调度。

<a id="section-2-6-launch-examples"></a>

### 2.6 启动实例

#### 2.6.1 精确的 64x65536x32768 启动

对 `M=64,N=65536,K=32768`,runner 推导出:

```text
W_N = 65536/256 = 256
W_M = 64/64     = 1
C_N = 256/4     = 64
C_M = 1/1       = 1
T_XY            = 64                       # 每个 Z 平面 64 个 cluster 任务

batch_stride_D  = 64*65536*2       =    8388608 B = 0x00800000
batch_stride_A  = 64*(32768/2)     =    1048576 B = 0x00100000
batch_stride_B  = 65536*(32768/2)  = 1073741824 B = 0x40000000
batch_stride_sA = 64*(32768/32)    =      65536 B = 0x00010000
batch_stride_sB = 65536*(32768/32) =   67108864 B = 0x04000000
```

这些正是 `make_contiguous_batch_strides` 返回的精确值。
`make_batched_launch_geometry` 给出:

| Batch | 逻辑 WG 网格 | 逻辑 cluster 网格 | 物理 HIP WG 网格 | `clusterDim` | 物理 cluster 网格 | Block / 编码递推步长 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `(256,1,1)` | `(64,1,1)` | `(256,1,1)` | `(4,1,1)` | `(64,1,1)` | `(128,1,1)` / 64 |
| 2 | `(256,1,2)` | `(64,1,2)` | `(256,1,2)` | `(4,1,1)` | `(64,1,2)` | `(128,1,1)` / 64 |

batch=1 时,256 个 WG 组成 64 个物理 cluster;batch=2 时,512 个 WG 组成 128 个
物理 cluster,分成两个完全相同的平面,每个平面 256 个 WG、64 个 cluster。
在任一平面内:

```text
p = cx,其中 cx = 0..63、cy = 0
p 恰好覆盖 0..63
cluster_n = p
cluster_m = 0
Ntile = 4*p + wg_x
Mtile = 0
下一个任务 p+64 >= T_XY,所以每个物理 cluster 跑完一个任务就退出
```

Z=0 时基址偏移全为零。在 batch=2 的启动里,Z=1 走的是完全相同的 `p=0..63`
X/Y 映射,只是从"各自基址 + 上表对应 stride"开始。两个平面 X/Y 遍历相同,是
因为 ISA 的 `p` 和 `T_XY` 都不含 Z;它们地址上互不重叠,则由那一次 Z 派生的
基址调整保证。

#### 2.6.2 实测的 batch-96 64x6144x7168 启动

对 `batch=96,M=64,N=6144,K=7168`,逻辑空间是:

```text
W_N = 6144/256 = 24
W_M = 64/64    = 1
C_N = 24/4     = 6
C_M = 1/1      = 1
T_XY           = 6                         # 每个 Z 平面 6 个 cluster 任务

物理 seed 数 C = T_XY = 6
递推步长 S     = next_pow2(6) = 8
cluster 网格   = (6,1,96)
HIP WG 网格    = (6*4,1,96) = (24,1,96)
log2_grid_x/y  = (3,0)
编码递推步长   = 8
```

这是 2.3 里"物理规模与编码步长故意不一致"的典型例子:只发 6 个 cluster,却
编码步长 8。物理 Y 只有一行,所以起始 `p` 恰好是 `cx=0..5`,**六个 cluster
全都有用**;下一个任务是 `p+8 >= 8 > 6`,所以没有第二轮。

与之前的"最大拓扑"相比,每个 Z 平面从 64 cluster / 256 WG 降到
6 cluster / 24 WG;96 个平面合计是 576 cluster / 2304 WG,而不是
6144 cluster / 24576 WG。这消除了实测到的过量启动,同时仍然保持单次 kernel
dispatch、grid Z 等于 batch。

<a id="section-3-software-pipeline"></a>

## 3. End-to-End Software Pipeline

Every wave consumes one local A `M64xK256` tile, one local B `N64xK256`
quarter, and their corresponding scales, so the steady body remains 40 DS
loads and 16 WMMA instructions per wave. The two schedule families below are
specialization-driven software templates. They do not assign different
host-M regions: all waves have local `M[0:63]`, while wave `w` has WG-relative
`N[64*w:64*w+63]`.

Per wave and K256 body, the local operand traffic is exactly 16
`ds_load_b128` operations for A data, 16 for B data, four `ds_load_b32`
operations for SA, and four for SB. Thus `16+16+4+4=40` DS loads feed eight
C/D fragments and 16 WMMA operations.

The wave0/2 versus wave1/3 split is retained from the 128x128 reference ISA's
local instruction ordering because it remains consistent with the same four
TDM specialist roles. It does not prove that the new 1x4 output placement
needs this split. Keeping the split is an unassembled scheduling candidate;
a target implementation may reschedule it after validating dependencies,
register lifetimes, and issue behavior. The 16-WMMA and 40-DS dynamic counts
remain the wave-local invariant.

<a id="section-3-1-wave02-flow"></a>

### 3.1 wave0/2: `B-current -> A-current -> A-next -> B-next`

This template groups the host-A-data specialist (wave0) with the SA
specialist (wave2). It is a scheduling group, not an M-axis pair: wave0 owns
WG-relative N quarter `[0:63]` and wave2 owns `[128:191]`, while both own
local M `[0:63]`.

```text
Prologue (wave0/2 specialization schedule)
issue TDM A/SA slot0/body0              # 2 TDM in this wave group; wave0=A, wave2=SA
issue TDM A/SA slot1/body1              # 2 TDM in this wave group
issue TDM A/SA slot2/body2              # 2 TDM in this wave group
s_wait_tensorcnt 2                      # one wait per wave; ensures the oldest slot0 is ready
s_barrier_signal -1
s_barrier_wait 0xffff                   # rendezvous for A/B/SA/SB slot0 from wave0/1/2/3

SA-current first half # 2 ds_load_b32/wave
    ds_ld32_as0 (0_0)  # current
    ds_ld32_as1 (0_1)  # current

A-current first half # 8 ds_load_b128/wave
    ds_ld128_a0 (0_0)  # current
    ds_ld128_a1 (0_1)  # current
    ds_ld128_a2 (0_2)  # current
    ds_ld128_a3 (0_3)  # current
    ds_ld128_a4 (0_4)  # current
    ds_ld128_a5 (0_5)  # current
    ds_ld128_a6 (0_6)  # current
    ds_ld128_a7 (0_7)  # current

SB-current first half # 2 ds_load_b32/wave
    ds_ld32_bs0 (0_0)  # current
    ds_ld32_bs1 (0_1)  # current

B-current first half # 8 ds_load_b128/wave
    ds_ld128_b0 (0_0)  # current
    ds_ld128_b1 (0_1)  # current
    ds_ld128_b2 (0_2)  # current
    ds_ld128_b3 (0_3)  # current
    ds_ld128_b4 (0_4)  # current
    ds_ld128_b5 (0_5)  # current
    ds_ld128_b6 (0_6)  # current
    ds_ld128_b7 (0_7)  # current
                                        # total 20 DS/wave = 16 b128 + 4 b32
issue TDM A/SA slot3/body3              # 2 TDM in this wave group; wave0=A, wave2=SA

SB-current second half                  # 2 ds_load_b32/wave
    ds_ld32_bs2 (1_0)  # current
    ds_ld32_bs3 (1_1)  # current

B-current second half                   # 8 ds_load_b128/wave
    ds_ld128_b8 (1_0)  # current
    ds_ld128_b9 (1_1)  # current
    ds_ld128_b10 (1_2)  # current
    ds_ld128_b11 (1_3)  # current
    ds_ld128_b12 (1_4)  # current
    ds_ld128_b13 (1_5)  # current
    ds_ld128_b14 (1_6)  # current
    ds_ld128_b15 (1_7)  # current

P0/body0:
SA/A/SB/B-current --> slot0
SA/A/SB/B-current first half 20 DS + SB/B-current second half 10 DS issued
steady total 40 DS/wave, 16 WMMA; 2 TDM in this wave group

s_wait_dscnt 10                         # wait1, SA/A/SB/B current first half ready

SA-current second half # 2 ds_load_b32/wave
    wmma0 (0_0)  # K0
    wmma1 (0_1)  # K0
    ds_ld32_as2 (1_0)  # current
    ds_ld32_as3 (1_1)  # current

A-current second half # 8 ds_load_b128/wave
    ds_ld128_a8 (1_0)  # current
    ds_ld128_a9 (1_1)  # current
    ds_ld128_a10 (1_2)  # current
    ds_ld128_a11 (1_3)  # current
    wmma2 (0_2)  # K1
    wmma3 (0_3)  # K1
    ds_ld128_a12 (1_4)  # current
    ds_ld128_a13 (1_5)  # current
    ds_ld128_a14 (1_6)  # current
    ds_ld128_a15 (1_7)  # current

s_wait_dscnt 10                                     # wait2, SB/B current second half ready

SA/A-next first half + WMMA G1 local M[0:31],N[32:63] + WG barrier  # 2 ds_load_b32 + 8 ds_load_b128 / wave
    s_wait_tensorcnt 2
    s_barrier_signal -1
    wmma4 (1_0)  # K0
    wmma5 (1_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    s_barrier_wait 0xffff
    ds_ld32_as0 (0_0)                   # next
    ds_ld128_a0 (0_0)                   # next
    ds_ld128_a1 (0_1)                   # next
    ds_ld128_a2 (0_2)                   # next
    ds_ld128_a3 (0_3)                   # next
    wmma6 (1_2)  # K1
    wmma7 (1_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as1 (0_1)                   # next
    ds_ld128_a4 (0_4)                   # next
    ds_ld128_a5 (0_5)                   # next
    ds_ld128_a6 (0_6)                   # next
    ds_ld128_a7 (0_7)                   # next

s_wait_dscnt 10                         # wait3, SA/A current second half ready
s_barrier_signal -1

SB/B-next first half + WMMA G2 K0 local M[32:63],N[0:31] + issue TDM  # 1 ds_load_b32 + 4 ds_load_b128 / wave + 1 TDM / wave
    wmma8 (2_0)  # K0
    s_barrier_wait 0xffff
    issue TDM A/SA slot0/body4          # 1 TDM/wave, 2 total in this wave group
    wmma9 (2_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs0 (0_0)                   # next
    ds_ld128_b0 (0_0)                   # next
    ds_ld128_b1 (0_1)                   # next
    ds_ld128_b2 (0_2)                   # next
    ds_ld128_b3 (0_3)                   # next

SB/B-next first half + WMMA G2 K1 local M[32:63],N[0:31]  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma10 (2_2)  # K1
    wmma11 (2_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs1 (0_1)                   # next
    ds_ld128_b4 (0_4)                   # next
    ds_ld128_b5 (0_5)                   # next
    ds_ld128_b6 (0_6)                   # next
    ds_ld128_b7 (0_7)                   # next

SB/B-next second half + WMMA G3 K0 local M[32:63],N[32:63] + finish  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma12 (3_0)  # K0
    loop control SALU
    wmma13 (3_1)  # K0
    loop induction update
    loop control compare
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs2 (1_0)                   # next
    ds_ld128_b8 (1_0)                   # next
    ds_ld128_b9 (1_1)                   # next
    ds_ld128_b10 (1_2)                  # next
    ds_ld128_b11 (1_3)                  # next

SB/B-next second half + WMMA G3 K1 local M[32:63],N[32:63]  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma14 (3_2)  # K1
    wmma15 (3_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs3 (1_1)                   # next
    ds_ld128_b12 (1_4)                  # next
    ds_ld128_b13 (1_5)                  # next
    ds_ld128_b14 (1_6)                  # next
    ds_ld128_b15 (1_7)                  # next
    loop control branch

P0->P1 boundary
loop branch not taken            # candidate steady path enters P1 directly
phase advance: P0 next becomes P1 current; P1 reuses the corresponding VGPR

P1/body1:
SA/A/SB/B-current --> slot1
SA/A/SB/B-current first half 20 DS + SB/B-current second half 10 DS issued
steady total 40 DS/wave, 16 WMMA; 2 TDM in this wave group

s_wait_dscnt 10                         # wait1, SA/A/SB/B current first half ready

SA-current second half # 2 ds_load_b32/wave
    p1_wmma0 (0_0)  # K0
    p1_wmma1 (0_1)  # K0
    ds_ld32_as2 (1_0)  # current
    ds_ld32_as3 (1_1)  # current

A-current second half # 8 ds_load_b128/wave
    ds_ld128_a8 (1_0)  # current
    ds_ld128_a9 (1_1)  # current
    ds_ld128_a10 (1_2)  # current
    ds_ld128_a11 (1_3)  # current
    p1_wmma2 (0_2)  # K1
    p1_wmma3 (0_3)  # K1
    ds_ld128_a12 (1_4)  # current
    ds_ld128_a13 (1_5)  # current
    ds_ld128_a14 (1_6)  # current
    ds_ld128_a15 (1_7)  # current

s_wait_dscnt 10                         # P1 wait2, SB/B current second half ready

SA/A-next first half + WMMA G1 local M[0:31],N[32:63] + WG barrier  # 2 ds_load_b32 + 8 ds_load_b128 / wave
    s_wait_tensorcnt 2
    s_barrier_signal -1
    p1_wmma4 (1_0)  # K0
    p1_wmma5 (1_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    s_barrier_wait 0xffff
    ds_ld32_as0 (0_0)                   # next
    ds_ld128_a0 (0_0)                   # next
    ds_ld128_a1 (0_1)                   # next
    ds_ld128_a2 (0_2)                   # next
    ds_ld128_a3 (0_3)                   # next
    p1_wmma6 (1_2)  # K1
    p1_wmma7 (1_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as1 (0_1)                   # next
    ds_ld128_a4 (0_4)                   # next
    ds_ld128_a5 (0_5)                   # next
    ds_ld128_a6 (0_6)                   # next
    ds_ld128_a7 (0_7)                   # next

s_wait_dscnt 10                         # P1 wait3, SA/A current second half ready
s_barrier_signal -1

P1 current second half ready, A-next first half issued  # next step enters P1 WMMA_G2 + barrier wait/TDM
```

<a id="section-3-2-wave13-flow"></a>

### 3.2 wave1/3: `A-current -> B-current -> B-next -> A-next`

This template groups the host-B-data specialist (wave1) with the SB
specialist (wave3). It is also non-spatial: wave1 owns WG-relative N quarter
`[64:127]` and wave3 owns `[192:255]`, while both own local M `[0:63]`.

```text
Prologue (wave1/3 specialization schedule)
issue TDM B/SB slot0/body0              # 2 TDM in this wave group; wave1=B, wave3=SB
issue TDM B/SB slot1/body1              # 2 TDM in this wave group
issue TDM B/SB slot2/body2              # 2 TDM in this wave group
s_wait_tensorcnt 2                     # one wait per wave; ensures the oldest slot0 is ready
s_barrier_signal -1
s_barrier_wait 0xffff                  # rendezvous for A/B/SA/SB slot0 from wave0/1/2/3

SB-current first half # 2 ds_load_b32/wave
    ds_ld32_bs0 (0_0)  # current
    ds_ld32_bs1 (0_1)  # current

B-current first half # 8 ds_load_b128/wave
    ds_ld128_b0 (0_0)  # current
    ds_ld128_b1 (0_1)  # current
    ds_ld128_b2 (0_2)  # current
    ds_ld128_b3 (0_3)  # current
    ds_ld128_b4 (0_4)  # current
    ds_ld128_b5 (0_5)  # current
    ds_ld128_b6 (0_6)  # current
    ds_ld128_b7 (0_7)  # current

SA-current first half # 2 ds_load_b32/wave
    ds_ld32_as0 (0_0)  # current
    ds_ld32_as1 (0_1)  # current

A-current first half # 8 ds_load_b128/wave
    ds_ld128_a0 (0_0)  # current
    ds_ld128_a1 (0_1)  # current
    ds_ld128_a2 (0_2)  # current
    ds_ld128_a3 (0_3)  # current
    ds_ld128_a4 (0_4)  # current
    ds_ld128_a5 (0_5)  # current
    ds_ld128_a6 (0_6)  # current
    ds_ld128_a7 (0_7)  # current
                                        # total 20 DS/wave = 16 b128 + 4 b32
issue TDM B/SB slot3/body3              # 2 TDM in this wave group

SA-current second half                  # 2 ds_load_b32/wave
    ds_ld32_as2 (1_0)  # current
    ds_ld32_as3 (1_1)  # current

A-current second half                   # 8 ds_load_b128/wave
    ds_ld128_a8 (1_0)  # current
    ds_ld128_a9 (1_1)  # current
    ds_ld128_a10 (1_2)  # current
    ds_ld128_a11 (1_3)  # current
    ds_ld128_a12 (1_4)  # current
    ds_ld128_a13 (1_5)  # current
    ds_ld128_a14 (1_6)  # current
    ds_ld128_a15 (1_7)  # current

P0/body0:
SB/B/SA/A-current --> slot0
SB/B/SA/A-current first half 20 DS + SA/A-current second half 10 DS issued
steady total 40 DS/wave, 16 WMMA; 2 TDM in this wave group

s_wait_dscnt 10                         # wait1, SB/B/SA/A current first half ready

SB-current second half # 2 ds_load_b32/wave
    wmma0 (0_0)  # K0
    wmma1 (0_1)  # K0
    ds_ld32_bs2 (1_0)  # current
    ds_ld32_bs3 (1_1)  # current

B-current second half # 8 ds_load_b128/wave
    ds_ld128_b8 (1_0)  # current
    ds_ld128_b9 (1_1)  # current
    ds_ld128_b10 (1_2)  # current
    ds_ld128_b11 (1_3)  # current
    wmma2 (0_2)  # K1
    wmma3 (0_3)  # K1
    ds_ld128_b12 (1_4)  # current
    ds_ld128_b13 (1_5)  # current
    ds_ld128_b14 (1_6)  # current
    ds_ld128_b15 (1_7)  # current

s_wait_dscnt 10                         # wait2, SA/A current second half ready

SB/B-next first half + WMMA G1 local M[32:63],N[0:31] + WG barrier  # 2 ds_load_b32 + 8 ds_load_b128 / wave
    s_wait_tensorcnt 2
    s_barrier_signal -1
    wmma4 (1_0)  # K0
    wmma5 (1_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    s_barrier_wait 0xffff
    ds_ld32_bs0 (0_0)                   # next
    ds_ld128_b0 (0_0)                   # next
    ds_ld128_b1 (0_1)                   # next
    ds_ld128_b2 (0_2)                   # next
    ds_ld128_b3 (0_3)                   # next
    wmma6 (1_2)  # K1
    wmma7 (1_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs1 (0_1)                   # next
    ds_ld128_b4 (0_4)                   # next
    ds_ld128_b5 (0_5)                   # next
    ds_ld128_b6 (0_6)                   # next
    ds_ld128_b7 (0_7)                   # next

s_wait_dscnt 10                         # wait3, SB/B current second half ready
s_barrier_signal -1

SA/A-next first half + WMMA G2 K0 local M[0:31],N[32:63] + issue TDM  # 1 ds_load_b32 + 4 ds_load_b128 / wave + 1 TDM / wave
    wmma8 (2_0)  # K0
    s_barrier_wait 0xffff
    issue TDM B/SB slot0/body4          # 1 TDM/wave, 2 total in this wave group
    wmma9 (2_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as0 (0_0)                   # next
    ds_ld128_a0 (0_0)                   # next
    ds_ld128_a1 (0_1)                   # next
    ds_ld128_a2 (0_2)                   # next
    ds_ld128_a3 (0_3)                   # next

SA/A-next first half + WMMA G2 K1 local M[0:31],N[32:63]  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma10 (2_2)  # K1
    wmma11 (2_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as1 (0_1)                   # next
    ds_ld128_a4 (0_4)                   # next
    ds_ld128_a5 (0_5)                   # next
    ds_ld128_a6 (0_6)                   # next
    ds_ld128_a7 (0_7)                   # next

SA/A-next second half + WMMA G3 K0 local M[32:63],N[32:63] + finish  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma12 (3_0)  # K0
    loop control SALU
    wmma13 (3_1)  # K0
    loop induction update
    loop control compare
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as2 (1_0)                   # next
    ds_ld128_a8 (1_0)                   # next
    ds_ld128_a9 (1_1)                   # next
    ds_ld128_a10 (1_2)                  # next
    ds_ld128_a11 (1_3)                  # next

SA/A-next second half + WMMA G3 K1 local M[32:63],N[32:63]  # 1 ds_load_b32 + 4 ds_load_b128 / wave
    wmma14 (3_2)  # K1
    wmma15 (3_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_as3 (1_1)                   # next
    ds_ld128_a12 (1_4)                  # next
    ds_ld128_a13 (1_5)                  # next
    ds_ld128_a14 (1_6)                  # next
    ds_ld128_a15 (1_7)                  # next
    loop control branch

P0->P1 boundary
loop branch not taken            # candidate steady path enters P1 directly
phase advance: P0 next becomes P1 current; P1 reuses the corresponding VGPR

P1/body1:
SB/B/SA/A-current --> slot1
SB/B/SA/A-current first half 20 DS + SA/A-current second half 10 DS issued
steady total 40 DS/wave, 16 WMMA; 2 TDM in this wave group

s_wait_dscnt 10                         # wait1, SB/B/SA/A current first half ready

SB-current second half # 2 ds_load_b32/wave
    p1_wmma0 (0_0)  # K0
    p1_wmma1 (0_1)  # K0
    ds_ld32_bs2 (1_0)  # current
    ds_ld32_bs3 (1_1)  # current

B-current second half # 8 ds_load_b128/wave
    ds_ld128_b8 (1_0)  # current
    ds_ld128_b9 (1_1)  # current
    ds_ld128_b10 (1_2)  # current
    ds_ld128_b11 (1_3)  # current
    p1_wmma2 (0_2)  # K1
    p1_wmma3 (0_3)  # K1
    ds_ld128_b12 (1_4)  # current
    ds_ld128_b13 (1_5)  # current
    ds_ld128_b14 (1_6)  # current
    ds_ld128_b15 (1_7)  # current

s_wait_dscnt 10                         # P1 wait2, SA/A current second half ready

SB/B-next first half + WMMA G1 local M[32:63],N[0:31] + WG barrier  # 2 ds_load_b32 + 8 ds_load_b128 / wave
    s_wait_tensorcnt 2
    s_barrier_signal -1
    p1_wmma4 (1_0)  # K0
    p1_wmma5 (1_1)  # K0
    s_wait_alu depctr_va_vdst(0)
    s_barrier_wait 0xffff
    ds_ld32_bs0 (0_0)                   # next
    ds_ld128_b0 (0_0)                   # next
    ds_ld128_b1 (0_1)                   # next
    ds_ld128_b2 (0_2)                   # next
    ds_ld128_b3 (0_3)                   # next
    p1_wmma6 (1_2)  # K1
    p1_wmma7 (1_3)  # K1
    s_wait_alu depctr_va_vdst(0)
    ds_ld32_bs1 (0_1)                   # next
    ds_ld128_b4 (0_4)                   # next
    ds_ld128_b5 (0_5)                   # next
    ds_ld128_b6 (0_6)                   # next
    ds_ld128_b7 (0_7)                   # next

s_wait_dscnt 10                         # P1 wait3, SB/B current second half ready
s_barrier_signal -1

P1 current second half ready, B-next first half issued  # next step enters P1 WMMA_G2 + barrier wait/TDM
```

<a id="section-4-p0-details"></a>

## 4. P0 Detailed Pipeline and Wave-Local Tile Coverage

Chapter 4 describes the unchanged per-wave `64x64xK256` compute body. Table
coordinates are wave-local unless explicitly prefixed by `WG-relative`.
For wave `w`, convert a listed output `(m,n)` to the WG tile with:

```text
WG-relative M = m
WG-relative N = 64*w + n
host M        = wg_m_origin + m
host N        = wg_n_origin + 64*w + n
```

Thus the two schedule families can share local coverage tables without
implying a two-dimensional wave grid.

<a id="section-4-1-wave02-details"></a>

### 4.1 wave0/2 Specialization Schedule

<a id="section-4-1-1-wave02-host-tile"></a>

#### 4.1.1 wave0/2 Wave-Local Tile Coverage

The same local tables apply to both waves. Wave0 adds N origin `0`; wave2
adds N origin `128`. Both add M origin `0`.

**A data: one `ds_load_b128 = M16xK64` per cell**

| Wave-local M / K | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| M0 [0:15] | `ds_ld128_a0` (`0_0`) | `ds_ld128_a1` (`0_1`) | `ds_ld128_a4` (`0_4`) | `ds_ld128_a5` (`0_5`) |
| M1 [16:31] | `ds_ld128_a2` (`0_2`) | `ds_ld128_a3` (`0_3`) | `ds_ld128_a6` (`0_6`) | `ds_ld128_a7` (`0_7`) |
| M2 [32:47] | `ds_ld128_a8` (`1_0`) | `ds_ld128_a9` (`1_1`) | `ds_ld128_a12` (`1_4`) | `ds_ld128_a13` (`1_5`) |
| M3 [48:63] | `ds_ld128_a10` (`1_2`) | `ds_ld128_a11` (`1_3`) | `ds_ld128_a14` (`1_6`) | `ds_ld128_a15` (`1_7`) |

**A scale: one `ds_load_b32 = M32xK128 scales` per cell**

| Wave-local M / K | K0 [0:127] | K1 [128:255] |
|---|---|---|
| M0 [0:31] | `ds_ld32_as0` (`0_0`) | `ds_ld32_as1` (`0_1`) |
| M1 [32:63] | `ds_ld32_as2` (`1_0`) | `ds_ld32_as3` (`1_1`) |

**B data: one `ds_load_b128 = N16xK64` per cell**

| Wave-local N / K | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| N0 [0:15] | `ds_ld128_b0` (`0_0`) | `ds_ld128_b1` (`0_1`) | `ds_ld128_b4` (`0_4`) | `ds_ld128_b5` (`0_5`) |
| N1 [16:31] | `ds_ld128_b2` (`0_2`) | `ds_ld128_b3` (`0_3`) | `ds_ld128_b6` (`0_6`) | `ds_ld128_b7` (`0_7`) |
| N2 [32:47] | `ds_ld128_b8` (`1_0`) | `ds_ld128_b9` (`1_1`) | `ds_ld128_b12` (`1_4`) | `ds_ld128_b13` (`1_5`) |
| N3 [48:63] | `ds_ld128_b10` (`1_2`) | `ds_ld128_b11` (`1_3`) | `ds_ld128_b14` (`1_6`) | `ds_ld128_b15` (`1_7`) |

**B scale: one `ds_load_b32 = N32xK128 scales` per cell**

| Wave-local N / K | K0 [0:127] | K1 [128:255] |
|---|---|---|
| N0 [0:31] | `ds_ld32_bs0` (`0_0`) | `ds_ld32_bs1` (`0_1`) |
| N1 [32:63] | `ds_ld32_bs2` (`1_0`) | `ds_ld32_bs3` (`1_1`) |

**Each cell in the K0 table computes `M16xN32xK128`, with a K range of `[0,127]`:**

| K0 [0,127] / wave-local M,N | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma0` (`0_0`) | `wmma4` (`1_0`) |
| M1 [16,31] | `wmma1` (`0_1`) | `wmma5` (`1_1`) |
| M2 [32,47] | `wmma8` (`2_0`) | `wmma12` (`3_0`) |
| M3 [48,63] | `wmma9` (`2_1`) | `wmma13` (`3_1`) |

**Each cell in the K1 table computes the second K128 accumulation over `[128,255]` for the same MxN output fragment:**

| K1 [128,255] / wave-local M,N | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma2` (`0_2`) | `wmma6` (`1_2`) |
| M1 [16,31] | `wmma3` (`0_3`) | `wmma7` (`1_3`) |
| M2 [32,47] | `wmma10` (`2_2`) | `wmma14` (`3_2`) |
| M3 [48,63] | `wmma11` (`2_3`) | `wmma15` (`3_3`) |

<a id="section-4-2-wave13-details"></a>

### 4.2 wave1/3 Specialization Schedule

<a id="section-4-2-1-wave13-host-tile"></a>

#### 4.2.1 wave1/3 Wave-Local Tile Coverage

The same local tables apply to both waves. Wave1 adds N origin `64`; wave3
adds N origin `192`. Both add M origin `0`.

**A data: one `ds_load_b128 = M16xK64` per cell**

| Wave-local M / K | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| M0 [0:15] | `ds_ld128_a0` (`0_0`) | `ds_ld128_a1` (`0_1`) | `ds_ld128_a4` (`0_4`) | `ds_ld128_a5` (`0_5`) |
| M1 [16:31] | `ds_ld128_a2` (`0_2`) | `ds_ld128_a3` (`0_3`) | `ds_ld128_a6` (`0_6`) | `ds_ld128_a7` (`0_7`) |
| M2 [32:47] | `ds_ld128_a8` (`1_0`) | `ds_ld128_a9` (`1_1`) | `ds_ld128_a12` (`1_4`) | `ds_ld128_a13` (`1_5`) |
| M3 [48:63] | `ds_ld128_a10` (`1_2`) | `ds_ld128_a11` (`1_3`) | `ds_ld128_a14` (`1_6`) | `ds_ld128_a15` (`1_7`) |

**A scale: one `ds_load_b32 = M32xK128 scales` per cell**

| Wave-local M / K | K0 [0:127] | K1 [128:255] |
|---|---|---|
| M0 [0:31] | `ds_ld32_as0` (`0_0`) | `ds_ld32_as1` (`0_1`) |
| M1 [32:63] | `ds_ld32_as2` (`1_0`) | `ds_ld32_as3` (`1_1`) |

**B data: one `ds_load_b128 = N16xK64` per cell**

| Wave-local N / K | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| N0 [0:15] | `ds_ld128_b0` (`0_0`) | `ds_ld128_b1` (`0_1`) | `ds_ld128_b4` (`0_4`) | `ds_ld128_b5` (`0_5`) |
| N1 [16:31] | `ds_ld128_b2` (`0_2`) | `ds_ld128_b3` (`0_3`) | `ds_ld128_b6` (`0_6`) | `ds_ld128_b7` (`0_7`) |
| N2 [32:47] | `ds_ld128_b8` (`1_0`) | `ds_ld128_b9` (`1_1`) | `ds_ld128_b12` (`1_4`) | `ds_ld128_b13` (`1_5`) |
| N3 [48:63] | `ds_ld128_b10` (`1_2`) | `ds_ld128_b11` (`1_3`) | `ds_ld128_b14` (`1_6`) | `ds_ld128_b15` (`1_7`) |

**B scale: one `ds_load_b32 = N32xK128 scales` per cell**

| Wave-local N / K | K0 [0:127] | K1 [128:255] |
|---|---|---|
| N0 [0:31] | `ds_ld32_bs0` (`0_0`) | `ds_ld32_bs1` (`0_1`) |
| N1 [32:63] | `ds_ld32_bs2` (`1_0`) | `ds_ld32_bs3` (`1_1`) |

**Each cell in the K0 table computes `M16xN32xK128`, with a K range of `[0,127]`:**

| K0 [0,127] / wave-local M,N | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma0` (`0_0`) | `wmma8` (`2_0`) |
| M1 [16,31] | `wmma1` (`0_1`) | `wmma9` (`2_1`) |
| M2 [32,47] | `wmma4` (`1_0`) | `wmma12` (`3_0`) |
| M3 [48,63] | `wmma5` (`1_1`) | `wmma13` (`3_1`) |

**Each cell in the K1 table computes the second K128 accumulation over `[128,255]` for the same MxN output fragment:**

| K1 [128,255] / wave-local M,N | N0 [0,31] | N1 [32,63] |
|---|---:|---:|
| M0 [0,15] | `wmma2` (`0_2`) | `wmma10` (`2_2`) |
| M1 [16,31] | `wmma3` (`0_3`) | `wmma11` (`2_3`) |
| M2 [32,47] | `wmma6` (`1_2`) | `wmma14` (`3_2`) |
| M3 [48,63] | `wmma7` (`1_3`) | `wmma15` (`3_3`) |

<a id="section-5-epilogue-design"></a>

## 5. Epilogue Design and Final Store Contract

This chapter specifies a natural epilogue for the wave-tile `64x64`, WG-tile
`64x256` design. It is design documentation; no assembled gfx1250 ISA exists
for the `f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps` target. The following
evidence labels are used throughout the chapter:

| Label | Meaning |
|---|---|
| Hardware fact | A behavior stated by the CDNA5 ISA or MI400 Shader Programming Guide. |
| 128x128 reference-ISA fact | A local compute or specialization property observed in `f4gemm_bf16_mxfp4_ABpreShuffle_128x128_4x4_ps.s`. It is historical evidence, not a target-geometry fact. |
| 256x256 reference-ISA fact | An output-path instruction or descriptor idiom observed in `my_code/fmha/dump_asm/hsa/gfx1250/f4gemm/f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.s`. It is historical evidence, not a target-address or resource fact. |
| Static derivation | Arithmetic or mapping derived from hardware facts and the geometry contract in Chapters 1-4. |
| Candidate choice | A proposed allocation, descriptor, or schedule that is not final until assembled and run on the target. |
| Validation boundary | A property that cannot be made bit-exact from the available text and must be checked in target assembly, disassembly, metadata, or hardware execution. |

<a id="section-5-1-output-geometry"></a>

### 5.1 Output Geometry and Fragment Mapping

**Hardware facts.** `v_wmma_scale_f32_32x16x128_f4` computes a 32x16
single-precision C/D matrix from FP4 inputs. The CDNA5 ISA states this contract
at L26180-L26200. The C/D storage description at CDNA5 ISA L7367-L7409 and
MI400 Shader Guide L10676-L10718 shows that a 32x16 F32 C/D matrix occupies
16 VGPRs across a wave32. Because host B supplies the hardware Matrix A and
host A supplies the hardware Matrix B, this hardware result is oriented as
host N32 x host M16 and must be transposed into host M x N order during LDS
staging.

**Static derivation.** One 32x16 fragment contains 512 F32 values:

```text
32 * 16 = 512 F32 values
512 / 32 lanes = 16 F32 values per lane
16 values per lane * 4 bytes = 16 VGPRs per fragment
```

The 64x64 wave tile has four wave-local M blocks and two wave-local N blocks,
so it has
eight independent C/D fragments. The K0 and K1 instructions shown below both
accumulate into the same named fragment; they are not separate output
fragments. This table repeats the exact Chapter 4 coordinates and pairings.

| Fragment | Wave-local output coordinates | wave0/2 K0 -> K1 | wave1/3 K0 -> K1 |
|---|---|---|---|
| `F00` | `M[0:15], N[0:31]` | `wmma0 -> wmma2` | `wmma0 -> wmma2` |
| `F01` | `M[0:15], N[32:63]` | `wmma4 -> wmma6` | `wmma8 -> wmma10` |
| `F10` | `M[16:31], N[0:31]` | `wmma1 -> wmma3` | `wmma1 -> wmma3` |
| `F11` | `M[16:31], N[32:63]` | `wmma5 -> wmma7` | `wmma9 -> wmma11` |
| `F20` | `M[32:47], N[0:31]` | `wmma8 -> wmma10` | `wmma4 -> wmma6` |
| `F21` | `M[32:47], N[32:63]` | `wmma12 -> wmma14` | `wmma12 -> wmma14` |
| `F30` | `M[48:63], N[0:31]` | `wmma9 -> wmma11` | `wmma5 -> wmma7` |
| `F31` | `M[48:63], N[32:63]` | `wmma13 -> wmma15` | `wmma13 -> wmma15` |

The `wmmaN` labels describe the Chapter 4 schedule, not fixed physical VGPR
numbers. A code generator must map both members of each K0/K1 pair to the same
physical 16-VGPR accumulator block.

Therefore each wave owns:

```text
8 fragments * 32 * 16 = 4096 BF16 outputs
wave output = 64 * 64 = 4096 BF16 outputs
```

The four wave tiles concatenate only along host N:

| Logical wave `w` | WG-relative M | WG-relative N | Coordinate transform |
|---:|---|---|---|
| 0 | `[0,63]` | `[0,63]` | `(m,n) -> (m,n)` |
| 1 | `[0,63]` | `[64,127]` | `(m,n) -> (m,64+n)` |
| 2 | `[0,63]` | `[128,191]` | `(m,n) -> (m,128+n)` |
| 3 | `[0,63]` | `[192,255]` | `(m,n) -> (m,192+n)` |

The combined WG output is exactly:

```text
4 waves * 4096 elements/wave = 16384 BF16 elements
64 rows * 256 columns         = 16384 BF16 elements
```

<a id="section-5-2-vgpr-layout"></a>

### 5.2 Candidate Physical VGPR Layout

The following is one explicit, internally non-overlapping **candidate choice**.
It is selected for simple addressing and review, not because target metadata
already guarantees it.

| Fragment | Wave-local coordinates | F32 accumulator block | Packed-BF16 staging block |
|---|---|---|---|
| `F00` | `M[0:15], N[0:31]` | `v256:v271` | `v128:v135` |
| `F01` | `M[0:15], N[32:63]` | `v272:v287` | `v136:v143` |
| `F10` | `M[16:31], N[0:31]` | `v288:v303` | `v144:v151` |
| `F11` | `M[16:31], N[32:63]` | `v304:v319` | `v152:v159` |
| `F20` | `M[32:47], N[0:31]` | `v320:v335` | `v160:v167` |
| `F21` | `M[32:47], N[32:63]` | `v336:v351` | `v168:v175` |
| `F30` | `M[48:63], N[0:31]` | `v352:v367` | `v176:v183` |
| `F31` | `M[48:63], N[32:63]` | `v368:v383` | `v184:v191` |

This layout has 128 contiguous accumulator VGPRs in `v256:v383` and 64
contiguous packed-BF16 staging VGPRs in `v128:v191`. Each fragment converts as
follows:

```text
512 F32 values / fragment
= 16 F32 values per lane
= 8 packed BF16 dwords per lane
= 8 packed-BF16 VGPRs per fragment

8 fragments * 8 packed VGPRs = 64 packed-BF16 VGPRs per wave
```

The staging allocation is a lifetime reuse choice. `v128:v191` may be reused
only after all final input-operand and prefetch-address uses of those physical
registers have ended. This epilogue reserves `v0:v127` for lane/LDS addresses,
vector-valued descriptor/index temporaries, and low operand banks; the TDM
descriptor groups themselves remain in SGPRs. `v192:v255` is intentionally
unassigned by this epilogue and remains available to the full-kernel allocator
for operand banks or additional temporaries. The allocation of those lower
regions is outside this epilogue table and still has to be proven
non-overlapping by assembled code.

**VGPR MSB requirement.** CDNA5 ISA L1155-L1197 documents that
`s_set_vgpr_msb` appends two address MSBs independently to the destination and
source operand fields. The accumulator bank `v256:v383` therefore requires
bank value `01` on the WMMA C/D fields. For a VOP3 packed conversion from
physical `v256:v257` to physical `v128`, the conceptual candidate is:

```text
# Candidate encoding; verify the immediate and physical registers in objdump.
s_set_vgpr_msb 0x05
# dst bank=00, src0 bank=01, src1 bank=01, src2 is unused
v_cvt_pk_bf16_f32 v128, v0, v1
# physical operation intended: v128 <- pack_bf16(v256, v257)
```

The documented immediate ordering is `{dst, src2, src1, src0}`. The final
emitter must set the appropriate MSBs around every WMMA, conversion, address,
and DS instruction, and must restore the low bank before low-bank DS sources
or addresses are used. The 256x256 reference ISA demonstrates
`s_set_vgpr_msb` transitions around conversion and DS sequences, for example
at L6522-L6523 and L6587-L6588. It does not establish the register assignment
for this target.

**Dependency requirement.** Before conversion reads the final WMMA results,
the candidate executes `s_wait_alu depctr_va_vdst(0)`. Before DS consumes
freshly converted staging VGPRs, it executes another required VA_VDST wait
unless target scheduling mode and independently verified spacing make that
wait redundant. Before any DS-source VGPR is overwritten, `s_wait_dscnt 0`
must establish that the DS operations have consumed their sources; the MI400
Shader Guide L5918-L5963 describes the relevant RAW, WAW, and WAR boundaries.

The highest explicitly allocated physical VGPR is `v383`, so the candidate
minimum next-free boundary is **at least 384**. This is not a
`next_free_vgpr` metadata claim. Extra operand banks, compiler temporaries,
allocation granularity, or an assembler restriction may raise the emitted
value, and only assembled target metadata may define the final value.

<a id="section-5-3-bf16-lds-staging"></a>

### 5.3 BF16 Conversion and LDS Staging

**256x256 reference-ISA facts.** The reference output path uses
`v_cvt_pk_bf16_f32` at L6458-L6521, `ds_store_b128` at L6588-L6620,
`s_wait_dscnt 0` at L6621, and `tensor_store_from_lds` at L6622, with a
second historical store at L6809. Its lane-address setup and DS offset pattern
provide an implementation idiom only; its tile geometry, LDS bases, and store
count do not describe this target.

CDNA5 ISA L32879-L32889 defines `V_CVT_PK_BF16_F32` as converting two F32
inputs to one packed BF16 dword using round-to-nearest-even. This is the
architecturally appropriate conversion family for a normal BF16 output unless
the target requires a different rounding contract.

The migrated layout principle is to use the low four lane bits as the
wave-local M row, lane bit 4 to select a wave-local N subrange, and consecutive
packed pairs to transpose the hardware N32 x M16 fragment into a host
M16 x N32 row-major tile.

The candidate sequence is:

```text
1. Complete the final K phase and stop issuing a real future prefetch.
   Use a scalar branch, or a validated null TDM descriptor with tile_dim0=0;
   EXEC masking alone is not sufficient because tensor instructions ignore EXEC.

2. Drain speculative and final-pipeline work:
   s_wait_dscnt 0
   s_wait_tensorcnt 0
   s_wait_alu depctr_va_vdst(0)

3. For each F00..F31 accumulator block:
   issue 8 v_cvt_pk_bf16_f32 operations
   map the 16 F32 values/lane to 8 packed BF16 dwords/lane

4. Before DS reads the conversion destinations:
   s_wait_alu depctr_va_vdst(0)

5. Apply the migrated lane/layout transpose and write each wave's 64x64
   N quarter to the row-major 64x256 WG staging tile.

6. Complete all output DS writes:
   s_wait_dscnt 0
```

For an exact candidate address map, define:

```text
lane_row    = lane_id & 15
lane_n_half = lane_id >> 4          # 0 or 1 for wave32

fragment_m = 0, 16, 32, or 48
fragment_n = 0 or 32
store_group = 0 or 1                # first or second four packed VGPRs
wave_n_origin = 64 * logical_wave_id

lds_address =
    0x2A800
  + (fragment_m + lane_row) * 0x200
  + (wave_n_origin + fragment_n
     + 16 * store_group + 8 * lane_n_half) * 2
```

One `ds_store_b128` writes four packed dwords, or eight BF16 values, per lane.
For a fragment, `store_group=0` uses the first four staging VGPRs and
`store_group=1` uses the next four. Lanes 0-15 and 16-31 address the same 16
wave-local M rows but disjoint eight-column wave-local N spans. This produces two
`ds_store_b128` instructions per 32x16 fragment without overlap.

The candidate DS count is therefore rigorous for this specific full-EXEC
`ds_store_b128` mapping:

```text
one ds_store_b128 wave aggregate = 32 lanes * 16 bytes = 512 bytes = 0x200
one wave output                  = 64 * 64 * 2 = 8192 bytes = 0x2000
candidate DS writes per wave     = 0x2000 / 0x200 = 16
candidate DS writes per WG       = 4 * 16 = 64
```

These are candidate instruction counts for the specified DS width and address
map. Another correct code generator may use a different DS width or schedule;
its exact count is target-codegen dependent and must be checked in
disassembly.

The row-major WG staging geometry is:

| Quantity | Candidate value |
|---|---:|
| One wave output | `64x64x2 = 0x2000` bytes |
| One WG output | `64x256x2 = 0x8000` bytes |
| WG output row stride | `256x2 = 0x200` bytes |
| `output_lds_base` | `0x2A800` |
| Output staging range | `[0x2A800,0x32800)` |

The wave row-zero staging origins are:

| Logical wave | WG-relative N origin | Staging origin |
|---:|---:|---|
| 0 | 0 | `output_lds_base + 0x000` |
| 1 | 64 | `output_lds_base + 0x080` |
| 2 | 128 | `output_lds_base + 0x100` |
| 3 | 192 | `output_lds_base + 0x180` |

Each increment of `0x080` is 64 BF16 columns. These are row-major N-quarter
origins, not contiguous `0x2000`-byte wave payloads. Every wave is interleaved
at the full `0x200` WG row stride, so DS addressing must use the row/lane map
above. There is no wave-M offset and no `0x4000` M-quadrant offset.

The mapping has complete static coverage:

```text
fragment_m + lane_row
    = {0,16,32,48} + [0,15]
    = every row in [0,63] exactly once per fragment-N block

fragment_n + 16*store_group + 8*lane_n_half + [0,7]
    = every wave-local column in [0,63] exactly once

wave_n_origin + wave-local column
    = disjoint WG columns [0,63], [64,127], [128,191], [192,255]

covered elements = 64 rows * 256 columns = 16384
covered bytes    = 16384 * 2 = 32768 = 0x8000
```

The row sets, fragment-N sets, store groups, lane halves, and wave-N origins
are pairwise disjoint at each nesting level. Therefore the candidate has
16384 unique element addresses with no holes or collisions. A tagged-lane
runtime test must still confirm that the physical WMMA lane distribution
matches the assumed migration mapping.

An independent temporary Python enumeration expanded every wave, fragment,
store group, lane, and eight-element `ds_store_b128` span. It produced:

```text
elements=16384, collisions=0, holes=0, extras=0
minimum byte address=0x2A800
maximum byte address exclusive=0x32800
```

The output region is deliberately separate from the input ring:

```text
input ring [0x00000,0x2A800)      0x2A800 = 170 KiB
output staging [0x2A800,0x32800)  0x08000 =  32 KiB
fixed LDS end                     0x32800 = 202 KiB
```

This 202-KiB total is the candidate fixed LDS requirement. There is no
explicit layout gap: B starts at the SB end `0x0A800`, and output staging
starts at the B end `0x2A800`. The separate, immediately adjacent output
region makes the first implementation easier to prove and prevents a late
input load or speculative TDM fill from overlapping output staging. Reusing a
drained input-ring slot is a future optimization and is permitted only after
a complete DS/TDM lifetime and barrier proof.

<a id="section-5-4-tdm-store-sync"></a>

### 5.4 Required TDM-Store Contract and Synchronization

**Hardware facts.** CDNA5 ISA L10147-L10156 states that tensor operations
ignore EXEC, increment the issuing wave's TENSORcnt, complete in order within
that wave, and are unordered with tensor operations from other waves.
Descriptor `tile_dim0` and `tile_dim1` are 16-bit fields in data-size units
(CDNA5 ISA L10487-L10503), and `tensor_dim0_stride` is in data-size elements
(L10505-L10514). For a store, `workgroup_mask` is ignored (L10276-L10279);
the candidate descriptor sets it to zero so that the descriptor
unambiguously requests no multicast behavior.

**256x256 reference-ISA facts.** The reference descriptor construction at
L6389-L6434 uses byte-oriented dimensions, `tile_dim1=64`, and a global row
stride supplied through the descriptor. Its stores at L6622 and L6809 are
64-row `tensor_store_from_lds` operations. This is direct evidence for the
required 64-row form; it does not prove this target's `0x200` row width,
packed descriptor bits, LDS address, or SGPR allocation.

The design requires exactly one 64-row by 256-BF16 store for the complete
`64x256` WG tile, issued only by logical wave0:

| Descriptor property | Required semantic value |
| --- | --- |
| Issuer | logical wave0 only, selected by a scalar wave-ID branch |
| Global tile base | `ptr_D + (wg_m_origin * N + wg_n_origin) * 2` |
| LDS tile base | `0x2A800` |
| `count` | exactly one valid descriptor and one store issue |
| `workgroup_mask` | `0` |
| `data_size` | byte mode, 1 byte per descriptor element |
| `tensor_dim0` | `0x200` bytes for the tile's row subrange |
| `tensor_dim1` | `64` rows |
| `tile_dim0` | `0x200` byte-mode elements, equal to 512 row bytes |
| `tile_dim1` | `64` rows |
| `tensor_dim0_stride` | `N * 2` bytes (`0x1000` for `N=2048`) |
| Descriptor padding, gather, iteration, atomic arrival | disabled |

Byte mode makes `tile_dim0=0x200` equal to the required `256*2=512` row
bytes. An element-mode descriptor with two-byte elements and `tile_dim0=256`
is semantically equivalent, but it is not the primary candidate. The packed
descriptor words, semantic-size encoding convention, SGPR assignments,
reserved bits, and accepted LDS/global alignments remain target-assembly and
hardware validation boundaries.

The exact WG synchronization skeleton is:

```text
# All four waves have issued their N-quarter DS writes.
all waves:
    s_wait_dscnt 0
    s_barrier_signal -1
    s_barrier_wait 0xffff       # complete 64x256 staging tile is visible

if logical_wave_id == 0:        # scalar control-flow branch
    tensor_store_from_lds candidate_64_row_descriptor  # exactly once per WG
    s_wait_tensorcnt 0

all waves:
    s_barrier_signal -1
    s_barrier_wait 0xffff       # wave0 has completed the output TDM
```

TENSORcnt is per wave, so the three non-issuing waves cannot wait on wave0's
counter. The second WG barrier is required to carry wave0's observed
completion to those waves before any wave reuses output LDS or begins a new
persistent task. After this WG rendezvous, the cluster-task transition adds a
cluster barrier as specified in Section 5.5.

<a id="section-5-5-final-drain-transition"></a>

### 5.5 Final Drain, P3 Wrap, and Persistent-Task Transition

The P0/P1/P2 phases continue directly to the next phase. P3 is the four-slot
ring boundary and must use one converged decision for all four WGs in the 1x4
cluster. The following pseudocode is the candidate control protocol:

```text
phase = P0

for each persistent logical cluster task:
    while true:
        run_current_K_body(phase)

        if phase == P0:
            phase = P1
            continue
        if phase == P1:
            phase = P2
            continue
        if phase == P2:
            phase = P3
            continue

        # P3 ring-wrap protocol. Exactly one wave per WG signals.
        if logical_wave_id == 0:
            s_barrier_signal -3

        if another_K_body_exists:
            all waves: s_barrier_wait 0xfffd
            phase = P0
            continue

        # Final K path. If P3 already signaled, it must also complete the
        # matching cluster wait; the final path may not abandon that protocol.
        suppress_future_prefetch_with_scalar_branch_or_null_descriptor()
        all waves: s_barrier_wait 0xfffd

        # Drain every class that can touch operands, the input ring, or C/D.
        all waves:
            s_wait_dscnt 0
            s_wait_tensorcnt 0
            s_wait_alu depctr_va_vdst(0)

        convert_pack_and_stage_64x64_N_quarter()
        run_WG_output_store_protocol_from_Section_5_4()

        # No WG may advance while another WG in the physical cluster is still
        # using current-task state or output staging.
        if logical_wave_id == 0:
            s_barrier_signal -3
        all waves: s_barrier_wait 0xfffd

        next_cluster_task = current_cluster_task + persistent_stride
        if next_cluster_task >= logical_cluster_task_count:
            terminate_all_WGs_in_this_physical_cluster()
            return                         # converged exit; no fall-through

        # Reinitialize only after the completed output-store and cluster waits.
        clear_f32_accumulators(v256:v383)
        reset_K_index_and_phase_to_P0()
        rebuild_A_B_SA_SB_and_output_tile_pointers(next_cluster_task)
        rebuild_TDM_descriptors_and_lane_LDS_addresses()
        reset_input_ring_ownership_state()
        current_cluster_task = next_cluster_task
        break
```

For the target shape's N-fastest logical task order, pointer reconstruction
after that barrier uses the following static coordinate derivation:

```text
cluster_n = next_cluster_task % 2
cluster_m = next_cluster_task / 2
Mtile     = cluster_m
Ntile     = 4*cluster_n + wg_x

wg_m_origin = 64  * Mtile
wg_n_origin = 256 * Ntile
wave_m_origin = wg_m_origin
wave_n_origin = wg_n_origin + 64*logical_wave_id
```

The A/SA pointers depend on `wg_m_origin`, B/SB depend on `wg_n_origin`, and
D depends on both origins. All TDM descriptors must be rebuilt from those new
coordinates. These formulas are part of the candidate launcher/swizzle
contract, not assembled-ISA facts.

The future prefetch decision must be effective before a real TDM issue. Tensor
instructions ignore EXEC, so an EXEC-masked instruction is not suppressed.
A scalar branch is sufficient. A descriptor with `tile_dim0=0` is an
alternative only if target validation confirms that it performs no transfer;
it still has to complete its tensor-counter protocol and must have atomic
arrival disabled.

All four WGs derive `another_K_body_exists`, `next_cluster_task`, and task
termination from the same cluster-task state. Exactly one wave in every WG
signals each cluster barrier, and all four waves in every WG wait. Thus each
cluster-barrier generation has four WG signals and 16 wave waits. No WG may
independently skip a signal or wait. For the 576-task logical cluster grid and
64 physical clusters, every physical cluster executes nine tasks with stride
64, but the uniform termination predicate is still required for every
supported shape.

Accumulator clearing and pointer/descriptor initialization may be optimized
or overlapped later. The correctness-first candidate serializes them after
the second WG barrier and the task-boundary cluster barrier. There must be no
outstanding output TDM before output staging or any overlapping input LDS is
reused for the next persistent task.

<a id="section-5-6-resource-validation"></a>

### 5.6 Resource and Validation Checklist

The candidate resource summary is:

| Resource | Candidate requirement | Boundary |
| --- | ---: | --- |
| F32 accumulators per wave | `8 * 16 = 128 VGPRs`, `v256:v383` | Unchanged because each wave remains `64x64` |
| Packed BF16 staging per wave | `8 * 8 = 64 VGPRs`, `v128:v191` | Unchanged; reuse only after input lifetimes end |
| Candidate minimum next-free VGPR boundary | `>= 384` | Not metadata; target assembly may raise it |
| Input LDS ring | `0x2A800 = 170 KiB` | Four contiguous payload arrays; no explicit layout gap |
| Output LDS staging | `0x8000 = 32 KiB` | Separate `[0x2A800,0x32800)` region |
| Candidate fixed LDS total | `0x32800 = 202 KiB` | Must be emitted and accepted as group-segment size |
| Output TDM | one `64x256` BF16 store per WG | `tile_dim1=64`; byte width `0x200` |

The final `next_free_vgpr`, group-segment fixed size, SGPR count, descriptor
SGPR placement, and occupancy must come from the assembled gfx1250 object.
They must not be copied from this candidate table into metadata without that
evidence.

Required validation in the intended ROCm gfx1250 container and on target
hardware:

1. **Assembler:** assemble the exact candidate with the target ROCm toolchain.
   Confirm acceptance of gfx1250 opcodes, `s_set_vgpr_msb`, a `0x32800`
   group segment, all four input descriptors, and the one-store output
   descriptor.
2. **Objdump:** disassemble the object and verify every WMMA C/D range,
   conversion source/destination, DS address/data range, wait, barrier, and
   exactly one `tensor_store_from_lds` issue per WG. Confirm that physical
   `v256:v383` and `v128:v191` do not alias after MSB expansion.
3. **Metadata:** inspect `next_free_vgpr`, SGPR use, kernel descriptor fields,
   `.group_segment_fixed_size`, wave32 mode, and any resource-allocation
   granularity. Reject a build whose metadata does not cover every emitted
   physical register and LDS byte.
4. **Input descriptors and ring:** verify semantic byte-mode tiles
   `0x80*64`, `0x08*64`, `0x80*256`, and `0x08*256` for A, SA, B, and SB,
   respectively; all four slot bases; and every wave-local B/SB quarter
   offset. Confirm any required AB-preshuffle regrouping, descriptor encoding,
   target legality of B bases `0x0A800/0x12800/0x1A800/0x22800`, TDM
   destination alignment, no overlap in `[0,0x2A800)`, and acceptable DS bank
   behavior.
5. **Conversion and layout:** run a tagged-fragment microtest that gives every
   fragment, lane half, row, and column a distinguishable value. Verify
   round-to-nearest-even BF16 packing, the hardware N32 x M16 to host M16 x
   N32 transform, N-quarter bases `0x000/0x080/0x100/0x180`, `0x200` row
   stride, and exactly 16384 unique elements with no overlap or holes in
   `[0x2A800,0x32800)`.
6. **Shape-domain gate:** reject any invocation unless `M % 64 == 0`,
   `N % 1024 == 0`, and `K % 256 == 0`. Verify exact 1x4 cluster division and
   the absence of M/N boundary, K-tail, and partial-cluster paths.
7. **Grid mapping:** for `M=18432,N=2048,K=7168`, verify logical WG grid
   `(8,288)`, logical cluster grid `(2,288)`, 576 cluster tasks, physical
   cluster grid `(16,4)`, WG launch `(64,4)`, stride 64, and nine tasks per
   physical cluster.
8. **Numerical comparison:** compare the complete kernel against the trusted
   BF16 reference for the target `M=18432, N=2048, K=7168` shape, including
   constant, random, signed, zero, overflow, NaN, and rounding-boundary cases
   supported by the test harness.
9. **Pipeline and barriers:** verify 40 DS loads and 16 WMMA instructions per
   wave per K256 body. Stress repeated persistent tasks and verify
   P3 continuation, final null/skip behavior, both WG barriers, both cluster
   barrier protocols, four-wave WG convergence, four-WG cluster convergence,
   one cluster signal per WG, all-wave waits, and uniform termination.
   Confirm that no DS or TDM operation accesses a region after reuse.
10. **Output TDM:** verify that exactly one wave0-issued 64-row
   `tensor_store_from_lds` uses LDS base `0x2A800` and writes the complete
   `64x256` tile (32 KiB) with a global row stride of `N * sizeof(BF16)` (4096
   bytes for `N=2048`) and no multicast. Check wave0 TENSORcnt completion
   before the second WG barrier.
11. **Cluster multicast:** verify `nwg_x=4`, `nwg_y=1`, the one-row WG bit
   mapping, A/SA mask `0xf`, B/SB mask `1<<x`, and 16 total shader TDM load
   issues per cluster per K256 body. Measure request combining separately;
   do not infer memory-transaction counts from masks.
12. **Occupancy and performance:** measure achieved occupancy, VGPR/LDS-limited
   residency, DS bank conflicts, TDM throughput, barrier cost, and end-to-end
   kernel performance. Performance results may choose a later input-ring
   reuse optimization only after the correctness proof remains intact.

<a id="section-6-cluster-tdm-multicast"></a>

## 6. Cluster TDM Multicast

No assembled ISA exists for this target. The 128x128 reference ISA and the
256x256 reference ISA are used only as historical evidence for wave
specialization and descriptor-building idioms. Their instruction addresses,
masks, payload sizes, LDS bases, and 4x4 cluster dimensions do not prove this
design. The target geometry below is a static derivation from the 1x4 contract
and the documented CDNA5 cluster/TDM semantics.

<a id="section-6-1-wg-bit-matrix"></a>

### 6.1 WG Bit Matrix

The hardware facts are:

1. CDNA5 ISA Section 2.3 states that a workgroup cluster is scheduled on one
   shader engine, contains at most 16 WGs, and places each member WG on a
   separate WGP (manual pages 9-10; local text L751-L758).
2. CDNA5 ISA Section 2.3.1 defines `WG_in_Cluster` as the logical WG ID and
   defines the TTMP6 fields (manual page 10; local text L769-L815):
   `nwg_y-1` is in bits 19:16, `nwg_x-1` is in bits 15:12, `wg_y` is in
   bits 7:4, and `wg_x` is in bits 3:0.
3. MI400 Shader Programming Guide Section 2.3.1 independently gives the same
   layout (manual page 24; local text L1720-L1766).
4. The same guide's SGPR-initialization description says that SPI creates
   CS waves in typewriter order, with X as the innermost coordinate
   (`for Z, for Y, for X`; local text L3728-L3734).

For this design, x is host N and y is host M. The semantic cluster dimensions
after decoding the TTMP6 dimension-minus-one fields are `nwg_x=4` and
`nwg_y=1`; consequently `wg_y=0` is the only legal y coordinate. With X
innermost, the flattened ID is:

```text
WGinCluster = x + nwg_x * y
            = x + 4 * y
            = x                 # because y=0

               host N / x
             x=0        x=1        x=2        x=3
host M / y
y=0          bit0/WG0   bit1/WG1   bit2/WG2   bit3/WG3
```

Only bits 0 through 3 represent WGs in this cluster; bits 4 through 15 must
remain zero. This one-row matrix is a static derivation from the documented
TTMP6 fields, X-innermost order, and the required 1x4 launch geometry.

The CDNA5 Tensor DMA group-1 descriptor defines bits 15:0 as
`workgroup_mask`, with one bit per cluster WG (Section 10.11.4, manual page
143; local text L10399-L10408). The multicast protocol says that each selected
WG must make the same memory request through one wave in that WG (Section
10.7, local text L9884-L9918). Consequently, a set mask bit selects a
requester and destination WG; it does not elect one WG as a cluster leader.
In this design, every selected requester issues through its operand-specialized
wave.

<a id="section-6-2-operand-multicast-masks"></a>

### 6.2 Operand Multicast Masks

Historical reference ISA provides migration evidence for assigning wave0 to A
data, wave1 to B data, wave2 to SA, and wave3 to SB. Its cluster masks and
line addresses do not apply here. The masks below are rebuilt from the
target's one M row and four N columns.

| Operand | Specialist | Reuse axis | Required mask |
| --- | --- | --- | --- |
| A data | wave0 | same M64 tile across all four N WGs | `0xf` |
| B data | wave1 | no M reuse because `nwg_y=1` | `1 << x` |
| SA | wave2 | same M64 scale tile across all four N WGs | `0xf` |
| SB | wave3 | no M reuse because `nwg_y=1` | `1 << x` |

The unassembled target mask-construction candidate decodes TTMP6 and then
builds:

```text
nwg_x = ((TTMP6 >> 12) & 0xf) + 1 = 4
nwg_y = ((TTMP6 >> 16) & 0xf) + 1 = 1
wg_y  =  (TTMP6 >> 4) & 0xf       = 0
wg_x  =   TTMP6       & 0xf       = x

row_bits = (1 << nwg_x) - 1       = 0xf
A_SA_mask = row_bits << (nwg_x*wg_y)
          = 0xf << 0              = 0xf
B_SB_mask = 1 << (wg_x + nwg_x*wg_y)
          = 1 << x
```

For A and SA, all WGs have the same cluster M coordinate and request the same
M-side operand tile. The constant `0xf` mask selects WG0 through WG3. All
four A requesters, and separately all four SA requesters, must issue the same
request through their specialized wave. The hardware may then combine
matching requests that arrive within its multicast window.

For B and SB, fixed N/x would normally select a cluster column. That column
contains only the requesting WG because `nwg_y=1`:

```text
WGinCluster = x + 4*0 = x
mask = 1 << WGinCluster = 1 << x
```

The complete examples are:

```text
A/SA, any requester x=0..3 -> mask 0xf -> WG0, WG1, WG2, WG3
B/SB, x=0 -> mask 0x1 -> WG0 only
B/SB, x=1 -> mask 0x2 -> WG1 only
B/SB, x=2 -> mask 0x4 -> WG2 only
B/SB, x=3 -> mask 0x8 -> WG3 only
```

The B/SB masks are nonzero and therefore select the documented cluster-load
path, but each has a requester set of size one and a multicast reuse factor of
one. There is no cross-M reuse to claim. The exact descriptor word,
`early_timeout` policy, and persistent-restart reconstruction are target-ISA
validation boundaries.

<a id="section-6-3-payload-cluster-coverage"></a>

### 6.3 Payload and Cluster Coverage

One WG computes a `64x256` output tile for one K256 body. FP4 consumes four
bits per value, and one E8M0 scale byte covers K32, so each WG-local
destination receives:

| Operand | Logical payload derivation | Bytes |
| --- | --- | ---: |
| A data | `M64 * K256 * 4 bits` | `8 KiB = 0x2000` |
| SA | `M64 * (K256 / K32) * 1 byte` | `512 B = 0x0200` |
| B data | `N256 * K256 * 4 bits` | `32 KiB = 0x8000` |
| SB | `N256 * (K256 / K32) * 1 byte` | `2 KiB = 0x0800` |

The CDNA5 descriptor layout places `tile_dim0` in `s39[31:16]` and
`tile_dim1` in `s40[15:0]`; both are in `data_size` units (Section 10.11.4,
local text L10487-L10514). The target semantic row counts and unassembled
byte-mode descriptor candidate are:

| Operand | Rows | Bytes per row | Candidate semantic tile | Payload |
| --- | ---: | ---: | ---: | ---: |
| A data | 64 | `0x80` | `tile_dim0=0x80, tile_dim1=64` | `0x2000` |
| SA | 64 | `0x08` | `tile_dim0=0x08, tile_dim1=64` | `0x0200` |
| B data | 256 | `0x80` | `tile_dim0=0x80, tile_dim1=256` | `0x8000` |
| SB | 256 | `0x08` | `tile_dim0=0x08, tile_dim1=256` | `0x0800` |

The row counts are fixed by target geometry. Mapping these semantic rows onto
the actual AB-preshuffled global layout, including any equivalent regrouping
required by its strides, is a target-codegen validation boundary. No packed
descriptor word is claimed here.

The corresponding non-overlapping four-slot bases are:

| Operand | slot0 | slot1 | slot2 | slot3 | End |
| --- | ---: | ---: | ---: | ---: | ---: |
| A data | `0x00000` | `0x02000` | `0x04000` | `0x06000` | `0x08000` |
| SA | `0x08000` | `0x08200` | `0x08400` | `0x08600` | `0x08800` |
| SB | `0x08800` | `0x09000` | `0x09800` | `0x0A000` | `0x0A800` |
| B data | `0x0A800` | `0x12800` | `0x1A800` | `0x22800` | `0x2A800` |

Within a slot, all waves use the same A/SA base. Wave `w` uses B offset
`w*0x2000` and SB offset `w*0x0200`, selecting its N64 quarter. The payload
arrays are contiguous, contain `0x2A800 = 170 KiB`, and have no explicit
layout gap. B starts at `0x0A800` with a `0x8000` slot stride, and output
occupies `[0x2A800,0x32800)`. B-base legality, TDM destination alignment, and
DS bank behavior remain target-ISA validation boundaries.

The 1x4 logical cluster contains four WGs and covers
`M64xN(4*256) = M64xN1024`. At shader-issue level, one K256 body has:

| Operand class | Requester identity groups | Shader TDM load issues | Static reuse |
| --- | --- | ---: | --- |
| A | one identical group of four WGs | 4 | potentially combine 4-to-1 upstream |
| SA | one identical group of four WGs | 4 | potentially combine 4-to-1 upstream |
| B | four distinct groups of one WG | 4 | reuse factor 1 |
| SB | four distinct groups of one WG | 4 | reuse factor 1 |
| Total | ten identity groups | 16 | not a transaction count |

Equivalently, every WG has four specialized waves and issues four TDM loads
per K256 body, so `4 WGs * 4 issues/WG = 16` shader TDM issues. Each operand
class has four requester instructions before any multicast combining. There
is no cluster leader that removes requester instructions.
Tensor instructions also ignore EXEC (CDNA5 ISA Section 10.11.1, local text
L10147-L10155), so EXEC masking cannot change this count.

A nonzero TDM `workgroup_mask` makes `TENSOR_LOAD_TO_LDS` use
`CLUSTER_LOAD_ASYNC` rather than `GLOBAL_LOAD_ASYNC` (CDNA5 ISA Section
10.11.3, local text L10269-L10279; MI400 Shader Programming Guide Section
4.10.3, manual page 200, local text L14296-L14305). The upstream semantics
are therefore different for the two reuse classes:

```text
A or SA:
    four selected WGs each issue the same specialized-wave request
    -> GL1 may combine matching requests that arrive in time
    -> the returned data is delivered to each requester's WG-local LDS slot

B or SB:
    one selected WG issues each N-specific request
    -> no cross-M requester exists in this 1x4 cluster
    -> reuse factor is one
```

This is multicast into separate WGP-local LDS destinations, not one
cluster-wide LDS allocation. The MI400 guide states that GL1 can merge at
most five requests into one data return (Section 4.9.8, manual page 190,
local text L13779-L13794), so an A/SA group of four is within that documented
limit. It does not prove that a particular dynamic group combines; arrival
timing and timeout policy still matter. Nothing in this static analysis establishes
the GL1-to-GL2 request count, cache-line transaction count, HBM transaction
count, achieved bandwidth, or performance; those require measurement.

No 64x256 target correctness or performance result is available. Any
bitwise-correct result from the historical 128x128 kernel validates only that
historical kernel and does not validate this design.
