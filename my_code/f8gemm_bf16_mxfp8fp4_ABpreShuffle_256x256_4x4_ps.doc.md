# gfx1250 MXFP8×MXFP4 256×256_4×4 Persistent GEMM 解析

本文解析
`f8gemm_bf16_mxfp8fp4_ABpreShuffle_256x256_4x4_ps`
在 `M=18432, N=2048, K=7168, a8w4, A-preshuffle=1, output=bf16`
下的 CO 选择、tile/grid、K-loop、TDM cluster multicast 与同步。

分析基于仓库当前源码、7994 行反汇编和本地 CDNA5/MI400 硬件资料。文中的
“ISA Lx / Vaddr”均指：

`my_code/fmha/dump_asm/hsa/gfx1250/mxfp8fp4gemm/f8gemm_bf16_mxfp8fp4_ABpreShuffle_256x256_4x4_ps.s`

术语约定：

- host `A`：MXFP8 activation，亦记作 `X`；
- host `B`：MXFP4 weight，亦记作 `W`；
- `ScaleA/ScaleB`：分别记作 `SA/SB`；
- WMMA 的 Matrix A 是 host `B/W`（FP4），Matrix B 是 host `A/X`（FP8）；
- “文档事实”来自硬件文档；“ISA 推断”来自当前二进制的数据流和控制流。

<a id="summary"></a>
## 结论摘要

- 当前 256-active-WGP gfx1250 配置下，selector 的两个 a8w4/apre1 候选均合法；
  256×256_4×4 有 576 个逻辑 WG tile、需 3 轮，64×512_4×1 有 1152 个、
  需 5 轮，因此严格选择前者。
- `_4x4` 表示 **4×4 workgroup cluster**，不是 wave 数。每个 WG 的
  `block=128`，CDNA5 wave32，因此恰有 4 个 wave。
- 每个 WG 计算一个 `256×256` 输出 tile；4 个 wave 按 2×2 象限各计算
  `128×128`。每个 K-body 覆盖 `K=128`，`K=7168` 共 56 个 body。
- 第一组 8 条 WMMA 的 src VGPR 在 wave0/2 形成 `2×4`、在 wave1/3
  形成 `4×2` 笛卡尔积，独立确认两者分别计算
  `M64×N32×K128`、`M32×N64×K128`；该结论不依赖 DS load 形状。
- 逻辑输出 grid 是 `(8,72,1)`，共 576 个 WG tile、36 个 logical cluster
  task；persistent 实际 launch 是 `(64,4,1)`，共 256 个 WG、16 个 cluster。
- 每 wave、每 K-body 有 64 条 WMMA、48 条 `ds_load_b128`、8 条独立
  `ds_load_b32`、1 条 `tensor_load_to_lds`；没有
  `ds_load_2addr_b32`。
- wave0/2 的 `DScnt` wait 上界序列是 `4→10→4→18→0`，wave1/3 是
  `8→18→8→10→0`。两组 wave 的 A/B DS 预取次序也互换。
- cluster 中每个 WG 都发 TDM，不是 row/column leader 代发。A/SA 使用
  `0xf << (4*y)`，B/SB 使用 `0x1111 << x`。每 cluster、每 K-body 仍发
  64 条 TDM 指令。
- multicast 保留每个 WG 的独立 LDS 副本，并可把 4 个匹配 requester 的
  上游请求/返回合并；理论复用上限是 4×，**不等于保证 HBM 只读一次**。

<a id="toc"></a>
## 目录

1. [Kernel 与 CO 选择](#kernel-selection)
2. [M/N/K tile、4-wave 象限与 TDM 角色](#tile-wave)
3. [4×4 cluster、逻辑任务与 persistent grid](#cluster-grid)
4. [K-loop、4-slot ring 与 DS bundle tile](#kloop-ring)
   - [4.3 一个 wave 聚合的 DS tile 与 bundle](#ds-bundle-tiles)
     - [4.3.4 第一组 WMMA src 笛卡尔积独立核验](#wmma-src-cartesian)
   - [4.4 Operand bundle 的 DS 发射与跨 body 滚动](#ds-bundle-issue-order)
5. [真实 1..9 steady pipeline](#steady-pipeline)
6. [Cluster TDM multicast](#cluster-multicast)
7. [TENSORcnt、WG barrier 与 cluster barrier](#synchronization)
8. [计数汇总、事实边界与限制](#counts-limits)

<a id="kernel-selection"></a>
## 1. Kernel 与 CO 选择

### 1.1 调用和筛选链

测试在 `intype=="a8w4"` 时选择 `aiter.gemm_a8w4_mxfp8`，并传入
`dtype=bf16`、`a_preshuffle=bool(apre)`：
`op_tests/test_mxfp8fp4gemm.py` L125–L144。wrapper 默认
`kernelName=""`，转成 `None` 后进入自动选择：
`aiter/ops/gemm_op_a8w4.py` L35–L61。

C++ selector：

1. 从 `hipDeviceProp.multiProcessorCount` 读取评分除数 `num_cu`
   （源码变量名）：`asm_mxfp8fp4gemm.cu` L59–L66。
2. 先按 gfx、`b_intype=1`、`a_preshuffle=1` 过滤：
   `asm_mxfp8fp4gemm.cu` L69–L75。
3. 检查 N/M tile 与 cluster 整除条件：L77–L94。
4. 计算
   `local_round = ceil(tg_num / num_cu)`，先最小化轮数；只有同轮时才考虑
   空余数和 `tile_m*tile_n/(tile_m+tile_n)`：L96–L114。

CSV 中只有两个 a8w4/apre1 候选：
`hsa/gfx1250/mxfp8fp4gemm/mxfp8fp4gemm.csv` L4–L5。

| 候选 | 合法性 | 逻辑 tile 数 `tg_num` | 当前 256 配置轮数 | tie-break 效率 |
|---|---|---:|---:|---:|
| 256×256，cluster 4×4 | `M/256=72`、`N/256=8`，分别可被 4 整除 | `72×8=576` | `ceil(576/256)=3` | `128` |
| 64×512，cluster 4×1 | `M/64=288`、`N/512=4`，N tile 数可被 4 整除 | `288×4=1152` | `ceil(1152/256)=5` | `512/9≈56.89` |

MI455X whitepaper L137–L142 记录 8 个 XCD、每 XCD 32 个 WGP，即 256
active WGP。本文的 3/5 轮结论对应当前 runtime property 为 256 的配置。
两候选轮数严格不同，因此遍历顺序和同轮 tie-break 都不会改变结果。

### 1.2 最终 symbol 与 CO

最终选择：

```text
symbol:
_ZN5aiter48f8gemm_bf16_mxfp8fp4_ABpreShuffle_256x256_4x4_psE

CO:
hsa/gfx1250/mxfp8fp4gemm/
f8gemm_bf16_mxfp8fp4_ABpreShuffle_256x256_4x4_ps.co
```

证据：

- CSV L4 同时给出 `knl_name` 和 `co_name`；
- C++ L211–L223 将二者传给 `AiterAsmKernel`；
- 反汇编入口 ISA L6 / Vaddr `0x1900` 与 symbol 精确匹配。

`data-init`、`scale-init` 和 `seed` 只参与输入/reference 生成，`mode` 只决定
迭代数和 profiler；它们不进入 selector。K 出现在 heuristic cache key 中，
但不参与这两个候选的 tile 评分；本 shape 另满足 launcher 的 `K%128==0`
约束（`asm_mxfp8fp4gemm.cu` L145–L159）。

<a id="tile-wave"></a>
## 2. M/N/K tile、4-wave 象限与 TDM 角色

### 2.1 三层 tile

| 层级 | M | N | K |
|---|---:|---:|---:|
| 单条 WMMA | 16 | 16 | 128 |
| 单 wave、单 K-body | 128 | 128 | 128 |
| 单 WG、单 K-body | 256 | 256 | 128 |
| 单 WG、完整本 shape | 256 | 256 | 7168 |
| 单 4×4 cluster、单 K-body | 1024 | 1024 | 128 |

`v_wmma_scale_f32_16x16x128_f8f6f4` 可见于 ISA L1020 /
Vaddr `0x2B10`；K-loop 在 ISA L1160 / `0x30D0` 每 body 增加
`0x80=128`。

### 2.2 为什么是 4 个 wave

host 固定 `bdx=128`，注释明确为 `4 waves * 32 threads`：
`asm_mxfp8fp4gemm.cu` L264–L267。CDNA5 使用 wave32，反汇编入口
ISA L7 也设置 `UC_VERSION_W32_BIT`。

因此：

```text
128 threads / 32 threads per wave = 4 waves
```

再次强调：文件名中的 `_4x4` 来自 CSV 的
`cluster_x=4, cluster_y=4`，不是“四个 wave”的编码。

### 2.3 2×2 输出象限

ISA L11 / `0x1918` 从 `ttmp8` 取得 `wave_id=s68`，L205–L212 按
`s68=0..3` 分流。输出 descriptor 在 ISA L286–L307 构造：

```text
Mbase = 256 * Mtile + 128 * (wave_id & 1)
Nbase = 256 * Ntile + 128 * (wave_id >> 1)
```

| wave | M 范围 | N 范围 | 额外 TDM 角色 |
|---:|---|---|---|
| 0 | `[0,127]` | `[0,127]` | host A / X |
| 1 | `[128,255]` | `[0,127]` | host B / W |
| 2 | `[0,127]` | `[128,255]` | ScaleA / SA |
| 3 | `[128,255]` | `[128,255]` | ScaleB / SB |

每 wave 有 `128×128/(16×16)=64` 个 accumulator fragment，所以每
K-body 执行 64 条 WMMA；4 个 wave 合计覆盖完整 `256×256` WG tile。
TDM specialization 是额外 load 职责，四个 wave 仍都执行 WMMA。

<a id="cluster-grid"></a>
## 3. 4×4 cluster、逻辑任务与 persistent grid

### 3.1 一个 cluster 覆盖什么

CSV L4 配置：

```text
tile_m=256, tile_n=256
cluster_x=4, cluster_y=4
persistent=1, wg_max=256
```

`gdx` 对应 N，`gdy` 对应 M（C++ L232–L247），故：

```text
cluster_x=4 -> N: 4 * 256 = 1024
cluster_y=4 -> M: 4 * 256 = 1024
```

一个 logical cluster task 包含 16 个 WG，单 K-body 计算
`1024×1024×128`，完整 K 计算 `1024×1024×7168` 并写出一个
`1024×1024` 区域。multicast 不改变每个 WG 对其 `256×256` 输出 tile
的所有权。

### 3.2 逻辑 grid

对 `M=18432,N=2048`：

```text
N tiles = 2048 / 256 = 8
M tiles = 18432 / 256 = 72

logical grid = (gdx,gdy,gdz) = (8,72,1)
logical WG tasks = 8 * 72 = 576
```

按 cluster 分组：

```text
N cluster tasks = 8 / 4 = 2
M cluster tasks = 72 / 4 = 18
total logical cluster tasks = 2 * 18 = 36
36 * 16 WG/cluster = 576 logical WG tasks
```

### 3.3 Persistent 实际 launch

C++ L250–L267 用 `wg_max=256` 覆盖逻辑 launch grid：

```text
cluster_size = 4 * 4 = 16
physical clusters = 256 / 16 = 16
gdx = 16 * cluster_x = 64
gdy = cluster_y = 4
gdz = 1
block = (128,1,1)
```

| 名称 | 值 | 含义 |
|---|---:|---|
| 逻辑 grid | `(8,72,1)` | 576 个输出 tile task |
| 逻辑 cluster task | 36 | 每 task 含 16 个输出 tile |
| 实际 launch grid | `(64,4,1)` | persistent 物理 grid |
| 实际 launched WG | 256 | `64×4` |
| 实际 launched cluster | 16 | `(64/4)×(4/4)` |

ISA L33–L54 由初始 persistent cluster ID 计算 36 个逻辑任务；
L127 / Vaddr `0x1B64` 执行 `s80 += 16`。因此物理 cluster `p` 处理：

```text
p, p+16, p+32, ...  (task < 36)
```

- `p=0..3`：各处理 3 个 logical cluster task；
- `p=4..15`：各处理 2 个；
- 三轮分别覆盖 256、256、64 个 logical WG tile，总计 576。

<a id="kloop-ring"></a>
## 4. K-loop、4-slot ring 与 DS bundle tile

### 4.1 56 个 K-body

每个 body 的 K 步长为 128：

```text
7168 / 128 = 56 bodies
56 = 14 * 4 phases
```

wave0 的四个同构 phase：

| phase | ISA 行 | Vaddr |
|---|---|---|
| P0 | L1018–L1170 | `0x2B08–0x3158` |
| P1 | L1171–L1323 | `0x315C–0x37AC` |
| P2 | L1324–L1476 | `0x37B0–0x3E00` |
| P3 | L1477–L1629 | `0x3E04–0x4454` |

P3 继续时由 ISA L1630 / `0x4458` 回到 P0。因为
`7168=14×512`，本 shape 动态执行 `P0→P1→P2→P3` 共 14 轮，最后从
P3 尾退出；为其他 `K mod 512` 准备的 P0/P1/P2 尾路不在本 shape 执行。

### 4.2 LDS ring

ISA L141–L162 构造四组 ring 区域。以下是区域锚点，实际 DS 地址还叠加
lane/wave 布局偏移：

| operand | slot 0 | slot 1 | slot 2 | slot 3 |
|---|---:|---:|---:|---:|
| A/X | `0x01000` | `0x09800` | `0x12000` | `0x1A800` |
| B/W | `0x23000` | `0x27000` | `0x2B000` | `0x2F000` |
| SA | `0x00000` | `0x00400` | `0x00800` | `0x00C00` |
| SB | `0x33000` | `0x33400` | `0x33800` | `0x33C00` |

每个 phase 消费一个 slot，并在计算期间通过 TDM 回填已消费的 slot，供后续
ring 轮次使用。pipeline 的基本单位是：

```text
K=128 + 64 WMMA/wave + 4-slot LDS/TDM ring
```

它不是 MoE GEMM1 的 K=256、16+16 WMMA 模板。

<a id="ds-bundle-tiles"></a>
### 4.3 一个 wave 聚合的 DS tile 与 bundle

本节只解释 DS payload 对应的逻辑 tile；bundle 的发射顺序和跨 body 滚动见
[4.4](#ds-bundle-issue-order)，完整 wait/WMMA/TDM 时序仍以
[第 5 节](#steady-pipeline)为准。以下 `m,n,k` 都是
**相对当前 wave 的 `128×128` 象限和当前 `K=128` body** 的闭区间坐标：
`m,n,k∈[0,127]`。它们不是 global 坐标，也不是单个 lane 的 tile。令
`M_b,N_b∈{0,64}` 分别表示该 wave 内第一/第二个 X/W bundle 的起点；下列
ISA 行/Vaddr 取 `M_b=N_b=0` 的首个常驻 bundle 作为实例。

#### 4.3.1 单位换算与 preshuffle 约束

CDNA5 ISA L10773–L10778 规定普通 `DS_LOAD` 是每 thread 返回一个值；
L36747–L36755 明确 `DS_LOAD_B128` 返回 128 bit，L35616–L35621 明确
`DS_LOAD_B32` 返回 32 bit。结合本 kernel 的 wave32：

| DS 指令 | 单 lane | 32 lanes 合计 | 当前 preshuffle 下的逻辑 tile |
|---|---:|---:|---|
| X `ds_load_b128` | 16 B | 512 B | `M16×K32` FP8 |
| W `ds_load_b128` | 16 B | 512 B | `N16×K64` FP4 |
| X_scale `ds_load_b32` | 4 B | 128 B | `M32×4` 个 E8M0 byte，即 `M32×K128` |
| W_scale `ds_load_b32` | 4 B | 128 B | `N32×4` 个 E8M0 byte，即 `N32×K128` |

`512 B` 本身只能给出 payload 大小，逻辑形状由 preshuffle 和 LDS lane
地址共同确定：

- X 是 1 B/element，`shuffle.py` L28–L38 将 `[M,K]` 排成
  `[M/2,K/128,2,128]`。ISA L130–L144 / `0x1B70–0x1BBC` 令
  `r=lane&15` 选择 16 个 M row、`lane>>4` 选择当前 32-K 片内的前/后
  16 B；同组 DS offset `0,32,64,96` 再选择四个 K32 片。因此一条是
  `M16×K32`，连续四条正好组成一个 WMMA 的 `M16×K128` X operand。
- W 的每个 packed byte 含两个 FP4。`shuffle.py` L41–L54（通用置换定义见
  L9–L25）给出的实际展平次序是
  `[N/16,(K/2)/16,16,16]`。一个 `16×16` packed-byte tile 是
  `256 B = N16×K32`；ISA L147–L152 / `0x1BD0–0x1BF0` 又令 lane
  地址按 16 B 连续递增，所以一条指令的 512 B 正好跨相同 N16 的两个
  相邻 packed-K tile，即 `N16×K64`。这排除了仅凭字节数猜出的
  `N8×K128`；连续两条才组成一个 WMMA 的 `N16×K128` W operand。
- scale 输入是 `[rows,K/32]`、每项 1 B。`shuffle.py` L57–L75 将其排成
  `[rows/32,(K/32)/4,32,4]`；ISA L153–L162 /
  `0x1BF8–0x1C24` 令 32 lanes 各取连续 4 B。因此一条
  `ds_load_b32` 覆盖 32 rows 的四个 scale，也就是 `rows32×K128`。
  消费端 L1020 / `0x2B10` 与 L1023 / `0x2B30` 的
  `matrix_b_scale` row 选择、L1020 与 L1034 / `0x2BA0` 的
  `matrix_a_scale` row 选择，又确认一个 scale VGPR 覆盖相邻两个
  16-row WMMA fragment，而不是单个 16-row fragment。

#### 4.3.2 W bundle：`N64×K128`

这里的 `W_data0..7` 是按首个 bundle 的 DS issue 顺序定义的文档标签，
不是反汇编中的符号。代表性实例为 ISA L998–L1005 /
`0x2A88–0x2AC0`；`W_scale0/1` 来自 L995–L996 /
`0x2A74–0x2A7C`。

| 项 | ISA 行 / Vaddr | 相对 N 区间 | 相对 K 区间 |
|---|---|---|---|
| `W_data0` | L998 / `0x2A88` | `[N_b,N_b+15]` | `[0,63]` |
| `W_data1` | L999 / `0x2A90` | `[N_b,N_b+15]` | `[64,127]` |
| `W_data2` | L1000 / `0x2A98` | `[N_b+16,N_b+31]` | `[0,63]` |
| `W_data3` | L1001 / `0x2AA0` | `[N_b+16,N_b+31]` | `[64,127]` |
| `W_data4` | L1002 / `0x2AA8` | `[N_b+32,N_b+47]` | `[0,63]` |
| `W_data5` | L1003 / `0x2AB0` | `[N_b+32,N_b+47]` | `[64,127]` |
| `W_data6` | L1004 / `0x2AB8` | `[N_b+48,N_b+63]` | `[0,63]` |
| `W_data7` | L1005 / `0x2AC0` | `[N_b+48,N_b+63]` | `[64,127]` |
| `W_scale0` | L995 / `0x2A74` | `[N_b,N_b+31]` | `[0,127]` |
| `W_scale1` | L996 / `0x2A7C` | `[N_b+32,N_b+63]` | `[0,127]` |

所以完整 W bundle 是 `8×512 B=4096 B` 的 FP4 data，加两条 scale
load，逻辑覆盖 `N64×K128`。第二个 bundle 使用相同展开并令 `N_b=64`：
scale 在 ISA L1021–L1022 / `0x2B20–0x2B28`，data 在
L1024–L1032 / `0x2B40–0x2B88`；两 bundle 合计覆盖当前 wave 的 N128。

#### 4.3.3 X bundle：`M64×K128`

代表性 `X_data0..15` 为 ISA L978–L993 /
`0x29F0–0x2A68`；`X_scale0/1` 为 L975–L976 /
`0x29DC–0x29E4`。

| 项 | ISA 行 / Vaddr | 相对 M 区间 | 相对 K 区间 |
|---|---|---|---|
| `X_data0` | L978 / `0x29F0` | `[M_b,M_b+15]` | `[0,31]` |
| `X_data1` | L979 / `0x29F8` | `[M_b,M_b+15]` | `[32,63]` |
| `X_data2` | L980 / `0x2A00` | `[M_b,M_b+15]` | `[64,95]` |
| `X_data3` | L981 / `0x2A08` | `[M_b,M_b+15]` | `[96,127]` |
| `X_data4` | L982 / `0x2A10` | `[M_b+16,M_b+31]` | `[0,31]` |
| `X_data5` | L983 / `0x2A18` | `[M_b+16,M_b+31]` | `[32,63]` |
| `X_data6` | L984 / `0x2A20` | `[M_b+16,M_b+31]` | `[64,95]` |
| `X_data7` | L985 / `0x2A28` | `[M_b+16,M_b+31]` | `[96,127]` |
| `X_data8` | L986 / `0x2A30` | `[M_b+32,M_b+47]` | `[0,31]` |
| `X_data9` | L987 / `0x2A38` | `[M_b+32,M_b+47]` | `[32,63]` |
| `X_data10` | L988 / `0x2A40` | `[M_b+32,M_b+47]` | `[64,95]` |
| `X_data11` | L989 / `0x2A48` | `[M_b+32,M_b+47]` | `[96,127]` |
| `X_data12` | L990 / `0x2A50` | `[M_b+48,M_b+63]` | `[0,31]` |
| `X_data13` | L991 / `0x2A58` | `[M_b+48,M_b+63]` | `[32,63]` |
| `X_data14` | L992 / `0x2A60` | `[M_b+48,M_b+63]` | `[64,95]` |
| `X_data15` | L993 / `0x2A68` | `[M_b+48,M_b+63]` | `[96,127]` |
| `X_scale0` | L975 / `0x29DC` | `[M_b,M_b+31]` | `[0,127]` |
| `X_scale1` | L976 / `0x29E4` | `[M_b+32,M_b+63]` | `[0,127]` |

所以完整 X bundle 是 `16×512 B=8192 B` 的 FP8 data，加两条 scale
load，逻辑覆盖 `M64×K128`。第二个 bundle 令 `M_b=64`：scale 在
ISA L1051–L1052 / `0x2C80–0x2C88`，data 在 L1055–L1073 /
`0x2CA4–0x2D4C`；两 bundle 合计覆盖当前 wave 的 M128。steady P0
对下一 body 的首个 X/W bundle 分别在 L1096–L1118 /
`0x2E58–0x2F24` 和 L1132–L1144 / `0x2FE0–0x304C` 重复同一映射。

<a id="wmma-src-cartesian"></a>
#### 4.3.4 从第一组 WMMA src 笛卡尔积独立核验

本小节刻意不使用 [4.3.1–4.3.3](#ds-bundle-tiles) 的 DS payload
形状，而只看 WMMA 硬件契约、当前指令的物理 src VGPR、scale selector
和独立 accumulator。CDNA5 ISA L26134–L26169 规定
`V_WMMA_SCALE_F32_16X16X128_F8F6F4` 计算
`D=A(16×128)×B(128×16)+C`，每个 operand 的一个矩阵分布在整个 wave；
S3、S4 分别提供 Matrix A、B 的 scale；反汇编中的 `matrix_a_scale`
（`SCALE_OPSEL[0]`）和 `matrix_b_scale`（`SCALE_OPSEL_HI[0]`）分别选择
对应 scale VGPR 的 lanes 0..15 或 16..31。对应到本 kernel：

| 指令输入 | 硬件矩阵 | kernel 数据及原 GEMM 方向 | 一个 src 组 |
|---|---|---|---|
| `srcA`（第一输入） | Matrix A，`16×128` | host `B/W`，MXFP4，原 N 方向的 `N16×K128` | 8 VGPR |
| `srcB`（第二输入） | Matrix B，`128×16` | host `A/X`，MXFP8；其 16 个硬件列是原 M 方向，payload 记作 `M16×K128` | 16 VGPR |

`matrix_a_fmt:MATRIX_FMT_FP4` 将第一输入选为 FP4；未显示的 Matrix B
format 为编码值 0，即 FP8 E4M3（CDNA5 ISA L26143–L26156）。8/16 VGPR
也与硬件 data layout 的 FP4/FP8 wave32 规定一致（L7838–L7839、
L7929–L7930）。因此硬件结果的 row/column 次序是 `W_N×X_M`；本文仍按
原 GEMM 惯例写成 `M×N`，即 X 组数决定 M、W 组数决定 N。

物理寄存器注释也已按 `s_set_vgpr_msb` 复核。CDNA5 ISA
L1155–L1197 规定把 `SIMM16[7:0]` 中每个 operand 的两位 MSB 拼到
8-bit VGPR 地址前：L1019 的低字节 `0x0b` 令
`srcA: v128+768=v896`、`srcB: v128+512=v640`；L2735 的低字节
`0x0d` 令 `srcA: v128+256=v384`、`srcB: v128+768=v896`，且两处
dst/src2 MSB 都为 0。scale 的 `SCL_SRC0/1` 明确忽略这些 MSB
（L15305–L15310），所以以下 `v196/v197/v200/v201` 就是物理 VGPR。
下文 `r0` 表示无 row1 suffix，`r1` 表示 `MATRIX_SCALE_ROW1`。

**`v200` 的 wave32 lane-private scale 布局。** `v200` 是一个 VGPR
编号，不是全 wave 共享的 scalar：同一编号在 wave32 中有 32 份
lane-private 32-bit 值。每 lane 的 32 bit 打包 4 个 E8M0 byte，分别对应
该 W/N row 在 K128 内的四个 K32 block scale：

```text
v200 lanes  0..15 -> W0 / local N  0..15，每 lane 4 bytes
v200 lanes 16..31 -> W1 / local N 16..31，每 lane 4 bytes
```

因此 lanes 0..15 和 lanes 16..31 分别组成第一个、第二个 N16 scale
矩阵，**不是每 lane 同时保存两个 N16 组**；合计
`N32×4 E8M0 bytes = 128 B`，恰与一条 wave 聚合
`ds_load_b32` 的 `32 lanes×4 B=128 B` 相符。这里的 N 坐标均为当前
wave/bundle 的局部坐标。

硬件依据是 CDNA5 ISA L26158–L26169：S3 是 Matrix A 的 `16×4`
scale 矩阵，每个 scale 覆盖 32 个 K-values，`SCALE_OPSEL[0]=0/1`
分别选择 lanes 0..15 / 16..31。这里“Matrix A”是硬件命名；按上表映射到
本 kernel 的 host `B/W`，所以其 16 行沿原 GEMM 的 W/N 方向。当前 WMMA
又直接验证 selector：ISA L1020 的
`W0=N[0:15]×K128` 使用 `S3=v200` 且无 row1 suffix，即默认
`matrix_a_scale` row0、读取 lanes 0..15；L1034 的
`W1=N[16:31]×K128` 仍使用 `S3=v200`，但显式
`matrix_a_scale:MATRIX_SCALE_ROW1`、读取 lanes 16..31。这是 VGPR
的 lane-private vector scale 输入，不是 scalar 共享 VGPR。

**wave0/2：2 个 W 组 × 4 个 X 组。** wave0 第一组是 ISA
L1020、L1023、L1028、L1033–L1037：

- W 行：`W0=v896:903, S3=v200/r0`；
  `W1=v904:911, S3=v200/r1`。
- X 列：`X0=v640:655, S4=v196/r0`；
  `X1=v656:671, S4=v196/r1`；
  `X2=v672:687, S4=v197/r0`；
  `X3=v688:703, S4=v197/r1`。

| `srcA=W` \ `srcB=X` | X0 | X1 | X2 | X3 |
|---|---|---|---|---|
| W0 | L1020 → `d/c v4:11` | L1023 → `d/c v36:43` | L1028 → `d/c v68:75` | L1033 → `d/c v100:107` |
| W1 | L1034 → `d/c v12:19` | L1035 → `d/c v44:51` | L1036 → `d/c v76:83` | L1037 → `d/c v108:115` |

这 8 条穷举 `2×4` 组合，故按原 GEMM 方向覆盖
`(4×M16)×(2×N16)×K128 = M64×N32×K128`。wave2 的同构副本
L4420、L4423、L4428、L4433–L4437 具有完全相同的物理 src、scale
和 selector。

**wave1/3：4 个 W 组 × 2 个 X 组。** wave1 第一组是 ISA
L2736、L2739、L2744、L2749、L2754、L2759–L2761：

- W 行：`W0=v384:391, S3=v200/r0`；
  `W1=v392:399, S3=v200/r1`；
  `W2=v400:407, S3=v201/r0`；
  `W3=v408:415, S3=v201/r1`。
- X 列：`X0=v896:911, S4=v196/r0`；
  `X1=v912:927, S4=v196/r1`。

| `srcA=W` \ `srcB=X` | X0 | X1 |
|---|---|---|
| W0 | L2736 → `d/c v4:11` | L2754 → `d/c v36:43` |
| W1 | L2739 → `d/c v12:19` | L2759 → `d/c v44:51` |
| W2 | L2744 → `d/c v20:27` | L2760 → `d/c v52:59` |
| W3 | L2749 → `d/c v28:35` | L2761 → `d/c v60:67` |

这里同样穷举 `4×2` 组合，覆盖
`(2×M16)×(4×N16)×K128 = M32×N64×K128`。wave3 的同构副本
L6103、L6106、L6111、L6116、L6121、L6126–L6128 逐项相同。

两张矩阵的每个格子都有不同的 `d/c` VGPR fragment；硬件每条产生一个
`16×16` C/D 矩阵且 opcode 固定 `K=128`，所以 8 条正是 8 个独立
`16×16×K128` 输出/累加 tile。这个结论只来自 WMMA src 的唯一组数及其
完整笛卡尔积，**不依赖 DS load 形状推断**；前面的 DS 映射只进一步给出
这些组在 bundle 内的相对坐标。

<a id="ds-bundle-issue-order"></a>
### 4.4 Operand bundle 的 DS 发射与跨 body 滚动

本节在 [4.3](#ds-bundle-tiles) 的 tile 映射之上，只补充 bundle 粒度的
issue 顺序；[第 5 节](#steady-pipeline)仍负责完整的 1..9 pipeline。
先严格约定名称：

- `current`/`next` 分别指当前 `K=128` body 和下一个 `K=128` body。
- “前半/后半”只切分 operand 的非 K 维 row：在当前 wave 的 M128/N128
  空间内，X 前、后半分别是 M `[0,63]`、`[64,127]`，W 前、后半分别是
  N `[0,63]`、`[64,127]`。
- 因而 **W 后半本身仍是完整的 `N64×K128` bundle，X 后半本身仍是完整的
  `M64×K128` bundle**；“半”不是把一个 bundle 的 scale 或 payload 再减半。
- 每个 bundle 内都是 scale 先于 payload：W 为
  `2×ds_load_b32(W_scale) → 8×ds_load_b128(W_data)`，共 **10 条物理 DS**；
  X 为 `2×ds_load_b32(X_scale) → 16×ds_load_b128(X_data)`，共
  **18 条物理 DS**。

#### 4.4.1 P0 入口的 28 条 DS 与 wait 后缀

wave0/P0 在 step 1 入口前先发 X 前半、再发 W 前半；略去
`s_set_vgpr_msb` 后，当前 ISA 是：

```text
L975–L976   / 0x29DC–0x29E4  X_scale ×2
L978–L993   / 0x29F0–0x2A68  X_data  ×16
L995–L996   / 0x2A74–0x2A7C  W_scale ×2
L998–L1005  / 0x2A88–0x2AC0  W_data  ×8
L1018       / 0x2B08         s_wait_dscnt 4
```

即 ISA L975–L1005 的 DS issue 序列是
`X_scale2 → X_data16 → W_scale2 → W_data8`，合计 `18+10=28` 条。
L1018 / `0x2B08` 只保证 `DScnt≤4`。结合 LDS 同 wave 按 issue 顺序完成并
按序递减 DScnt，可能仍 outstanding 的只能是这个 28 条序列的至多最后 4
条，也就是 W 的 `W_data4..7`（L1002–L1005 /
`0x2AA8–0x2AC0`）；它们不是必然仍有 4 条 pending。

wave1/P0 把两个 bundle 的顺序反过来：

```text
L2691–L2692  / 0x5718–0x5720  W_scale ×2
L2694–L2701  / 0x572C–0x5764  W_data  ×8
L2703–L2704  / 0x5770–0x5778  X_scale ×2
L2706–L2721  / 0x5784–0x57FC  X_data  ×16
L2734        / 0x5844         s_wait_dscnt 8
```

因此 ISA L2691–L2721 是
`W_scale2 → W_data8 → X_scale2 → X_data16`，仍为 28 条；L2734 /
`0x5844` 的 `DScnt≤8` 只允许序列末尾至多 8 条尚未完成，即最后 8 条
X data（L2714–L2721 / `0x57C4–0x57FC`）。wave2 的同构副本为
L4375–L4405 / `0x837C–0x8460`，并在 L4418 / `0x84A8` wait 4；
wave3 的同构副本为 L6058–L6088 / `0xAFD8–0xB0BC`，并在
L6101 / `0xB104` wait 8。

这里的 4/8 都只是 wait 后的计数器上界，而不是可由静态 ISA 确定的动态
精确值。上述“末尾至多 4/8 条”还依赖 LDS 的 in-order completion；
计数器定义和硬件依据见 [7.1](#dependency-counters)。

#### 4.4.2 当前后半与 next 前半如何滚动

把 [第 5 节](#steady-pipeline)的计算和同步折叠为省略号后，wave0/2 的
bundle 序列是：

```text
X_current 前半 → W_current 前半 → wait DScnt≤4
→ W_current 后半(step 1)
→ …〔step 2: wait≤10；step 3 入口: wait≤4〕
→ X_current 后半(step 3)
→ … → X_next 前半(step 6) → W_next 前半(step 8)
```

wave0/P0 的四个 loop 内发射点分别是 W current 后半 L1021–L1032 /
`0x2B20–0x2B88`、X current 后半 L1051–L1073 /
`0x2C80–0x2D4C`、X next 前半 L1096–L1118 /
`0x2E58–0x2F24`、W next 前半 L1132–L1144 /
`0x2FE0–0x304C`。

wave1/3 的顺序相反：

```text
W_current 前半 → X_current 前半 → wait DScnt≤8
→ X_current 后半(step 1)
→ …〔step 2: wait≤18；step 3 入口: wait≤8〕
→ W_current 后半(step 3)
→ … → W_next 前半(step 6) → X_next 前半(step 8)
```

wave1/P0 的对应发射点是 X current 后半 L2737–L2758 /
`0x585C–0x5924`、W current 后半 L2775–L2787 /
`0x59FC–0x5A68`、W next 前半 L2812–L2824 /
`0x5B94–0x5C00`、X next 前半 L2840–L2862 /
`0x5CDC–0x5DA8`。step 6/8 发出的 `next` 前半在控制流进入下一个 phase
后就成为其 `current` 前半，这正是跨 body 滚动关系；其余 WMMA、wait、
barrier 和 TDM 插槽不在此重复。

<a id="steady-pipeline"></a>
## 5. 真实 1..9 steady pipeline

以下以 wave0/P0 的 ISA L1018–L1170 为地址基准；wave1/P0 的对应副本为
L2734–L2886 / `0x5844–0x5E94`。

DS 指令的 wave 聚合 tile、bundle 展开见 [4.3](#ds-bundle-tiles)；不依赖
DS 形状、直接从第一组 8 条 WMMA src 笛卡尔积作出的独立核验见
[4.3.4](#wmma-src-cartesian)。两组 wave 的 bundle issue/滚动关系见
[4.4](#ds-bundle-issue-order)；本节只保留 wait、barrier、计算和下一
bundle 发射之间的完整 steady 时序。

硬件文档规定 `S_WAIT_DSCNT n` 只保证继续执行时 `DScnt≤n`
（CDNA5 ISA L20007–L20019；另见[计数器事实](#dependency-counters)）。
所以表中只写 wait 后上界；发射后的实际值可在任意时刻异步下降，不能静态
伪装成精确动态值。

| 步 | wave0/P0 范围 | wait 后上界 | 计算、同步与新发射 |
|---:|---|---|---|
| 1 | L1018–L1037 / `0x2B08–0x2BDF` | w0/2: `DScnt≤4`; w1/3: `≤8` | 8 WMMA。w0/2 发当前 B/W 后半 `8×b128 + 2×b32(SB)`；w1/3 发当前 A/X 后半 `16×b128 + 2×b32(SA)`。 |
| 2 | L1038–L1046 / `0x2BE0–0x2C63` | w0/2: `≤10`; w1/3: `≤18` | 8 WMMA，无新 DS。 |
| 3 | L1047–L1076 / `0x2C64–0x2D83` | w0/2: `≤4`; w1/3: `≤8` | 8 WMMA。w0/2 发当前 A/X 后半 `16×b128 + 2×b32(SA)`；w1/3 发当前 B/W 后半 `8×b128 + 2×b32(SB)`。 |
| 4 | L1077–L1085 / `0x2D84–0x2E07` | w0/2: `≤18`; w1/3: `≤10` | 8 WMMA，无新 DS；累计完成前 32 条 WMMA。 |
| 5 | L1086–L1093 / `0x2E08–0x2E4F` | `DScnt≤0`, `TENSORcnt≤2` | 排空当前 DS，等待本 wave 最老相关 TDM，`s_barrier_signal -1`，在 barrier wait 前穿插 3 WMMA。 |
| 6 | L1094–L1119 / `0x2E50–0x2F3B` | 无新 wait | `s_barrier_wait -1` 后再做 5 WMMA。w0/2 发下一 body 的 A/X 前半 `16×b128 + 2×b32(SA)`；w1/3 发下一 body 的 B/W 前半 `8×b128 + 2×b32(SB)`。 |
| 7 | L1120–L1128 / `0x2F3C–0x2FC7` | 发射前 `TENSORcnt≤2` | 1 条 `tensor_load_to_lds` 回填 ring slot，故发射后静态上界至多 `≤3`；同时 8 WMMA。每个 wave 只回填其 specialized operand。 |
| 8 | L1129–L1159 / `0x2FC8–0x30CF` | 无新 wait | 8 WMMA。w0/2 发下一 body 的 B/W 前半 `8×b128 + 2×b32(SB)`；w1/3 发下一 body 的 A/X 前半 `16×b128 + 2×b32(SA)`；并更新 TDM descriptor/pointer。 |
| 9 | L1160–L1170 / `0x30D0–0x3158` | 无新 wait | `K_offset += 128`、比较 K、最后 8 WMMA、条件跳转到下一 phase 或尾部。 |

关键同步片段很短：

```text
L1086  0x2E08  s_wait_dscnt 0
L1087  0x2E0C  s_wait_tensorcnt 2
L1088  0x2E10  s_barrier_signal -1
...
L1094  0x2E50  s_barrier_wait 0xffff
...
L1120  0x2F3C  tensor_load_to_lds
```

两组 wave 的 DS 顺序互换，但单 body 总数完全相同：

```text
8 groups * 8 WMMA = 64 WMMA
4 DS bundles = 48 ds_load_b128 + 8 ds_load_b32
1 tensor_load_to_lds
```

8 条 scale load 是 8 条独立物理 DS 指令；当前反汇编中没有
`ds_load_2addr_b32`。

<a id="cluster-multicast"></a>
## 6. Cluster TDM multicast

### 6.1 Cluster 内 WG bit 矩阵

CDNA5 ISA L769–L813 定义 `TTMP6[3:0]=wg_x`、
`TTMP6[7:4]=wg_y`。当前 4×4 kernel 在 ISA L34–L37 构造：

```text
WGinCluster = wg_x + 4 * wg_y
```

其中 x 是 N 方向，y 是 M 方向：

```text
             x/N=0   x/N=1   x/N=2   x/N=3
y/M=0        bit0    bit1    bit2    bit3
y/M=1        bit4    bit5    bit6    bit7
y/M=2        bit8    bit9    bit10   bit11
y/M=3        bit12   bit13   bit14   bit15
```

CDNA5 ISA L10399–L10408 规定 D# Group1 `[15:0]` 是
`workgroup_mask`，bit `i` 对应 `WGinCluster=i`。

### 6.2 四种 operand 的 mask

| operand | 复用方向 | mask | 反汇编构造 | 首个 TDM |
|---|---|---|---|---|
| A/X | 固定 M，跨 4 个 N WG | `0xf << (4*y)` | L264–L267 / `0x1E14–0x1E24` | L282 / `0x1E70` |
| SA | 固定 M，跨 4 个 N WG | `0xf << (4*y)` | L3665–L3668 / `0x77C0–0x77D0` | L3682 / `0x7810` |
| B/W | 固定 N，跨 4 个 M WG | `0x1111 << x` | L1982–L1984 / `0x4B58–0x4B68` | L1998 / `0x4BAC` |
| SB | 固定 N，跨 4 个 M WG | `0x1111 << x` | L5349–L5351 / `0xA418–0xA428` | L5365 / `0xA46C` |

例如：

```text
A/SA row y=2: 0x0f00 -> WG 8,9,10,11
B/SB col x=1: 0x2222 -> WG 1,5,9,13
```

### 6.3 谁发请求：每 WG，不是 leader

ISA L205–L212 只按 **WG 内 wave ID** 分工：

- wave0 发 A/X TDM；
- wave1 发 B/W TDM；
- wave2 发 SA TDM；
- wave3 发 SB TDM。

每个 cluster WG 都包含这四个 wave，代码中没有按 `wg_x/wg_y` 选
row/column leader 后抑制其他 WG 的分支；tensor 指令还忽略 EXEC
（CDNA5 ISA L10147–L10151）。因此每个 WG 都发自己的四条 TDM。

这也符合 multicast 协议：mask 中每个 WG 都应由一个 wave 发同地址、同
mask 请求；未及时到达者会在 timeout 后得到单独 broadcast
（CDNA5 ISA L9884–L9915；MI400 Shader Guide L13783–L13814）。

一个 logical cluster、一个 K-body：

| operand | shader TDM issue | multicast 组 |
|---|---:|---:|
| A/X | 16 | 4 个 row 组 |
| SA | 16 | 4 个 row 组 |
| B/W | 16 | 4 个 column 组 |
| SB | 16 | 4 个 column 组 |
| 合计 | **64** | **16 个四-requester 组** |

multicast 不减少 shader descriptor/TDM issue 数。

### 6.4 硬件机制与 LDS 副本

文档事实：

1. 一个 cluster 最多 16 个 WG、位于同一 Shader Engine，且每个 WG 在不同
   WGP 上：CDNA5 ISA L751–L758。
2. `D#.workgroup_mask!=0` 时，`TENSOR_LOAD_TO_LDS` 使用
   `CLUSTER_LOAD_ASYNC` 而非普通 `GLOBAL_LOAD_ASYNC`：
   CDNA5 ISA L10269–L10279；MI400 Guide L14296–L14305。
3. multicast async load 直接写入各 requester workgroup 可读的 LDS：
   CDNA5 ISA L9921–L9925、L10142–L10144。
4. MI400 GL1 multicast 设计文档 L198–L209 说明 GL1 跟踪匹配请求以避免
   重复发往 GL2；L416–L423 说明首个请求发往 GL2，后续命中 requester
   不再重复发往 GL2。

这里不是 cluster-wide LDS。每个 WG 仍有自己的逻辑 LDS allocation：

```text
一个合并返回
  -> requester WG0 的 LDS offset X
  -> requester WG1 的 LDS offset X
  -> requester WG2 的 LDS offset X
  -> requester WG3 的 LDS offset X
```

offset 相同不代表地址空间共享；普通 DS 指令不能直接读取另一个 WG 的 LDS。
所以 16 个 WG 最终仍有 16 份 LDS 副本。

### 6.5 4× 是上界，不是 HBM 保证

每个 row/column 组有 4 个匹配 requester，理想情况下可把 4 份独立的
GL1→GL2 请求/返回 payload 合成 1 组，即理论上游复用上限 4×。MI400
Shader Guide L13793–L13794 说明 GL1 最多可 merge 5 个请求，本 kernel 的组
大小 4 在此上限内。

但下列说法不成立：

- “TDM 指令数减少 4×”；
- “LDS 只保留一份”；
- “HBM 保证只读一次”；
- “一定获得 4× 带宽或性能”。

是否形成一个最终 DRAM transaction 还受 requester 到达/timeout、cache-line
拆分、GL2 命中与调度影响。硬件资料明确支持的是匹配请求可在 GL1 合并并减少
GL1→GL2 重复流量，不是对 HBM transaction 数作绝对保证。

<a id="synchronization"></a>
## 7. TENSORcnt、WG barrier 与 cluster barrier

<a id="dependency-counters"></a>
### 7.1 计数器事实

- `DScnt`：每条 LDS 指令发射加 1；load 返回 VGPR 或 store 写入 LDS 后减
  1。计数单位是指令，不是 byte、DWORD 或 lane：
  CDNA5 ISA L3499–L3511。
- 同一 wave 发出的 LDS 指令在 LDS 路径及 DScnt decrement 上保持顺序；
  LDS 指令按序完成：CDNA5 ISA L3575–L3586；MI400 Shader Guide
  L5792–L5805。
- `S_WAIT_DSCNT n`：只保证 `DScnt≤n`：
  CDNA5 ISA L20007–L20019。
- `TENSORcnt`：每条 TDM transfer 发射加 1，完成减 1：
  CDNA5 ISA L3529–L3533；MI400 Guide L5618–L5622。
- `S_WAIT_TENSORCNT n` 同样只保证 `TENSORcnt≤n`：
  CDNA5 ISA L20041–L20045。
- 同一 wave 的 tensor 指令按序完成，不同 wave 之间无序：
  CDNA5 ISA L10152–L10156；MI400 Guide L14100–L14104。

因此 steady loop 的 `s_wait_tensorcnt 2` 不是“全部 TDM 已清零”，也不能写成
“动态值恰好为 2”。它利用同一 wave 的 in-order completion 和已知 ring
深度，保证即将复用/读取的最老相关 slot 已完成。

### 7.2 WG barrier：汇合四个 specialized wave

CDNA5 ISA L3044–L3052、L3128–L3146 定义 `-1/0xffff` 为 workgroup
barrier。steady loop 中每个 specialized wave：

1. 等待自己的旧 TDM 达到 `TENSORcnt≤2`；
2. `s_barrier_signal -1`；
3. 四个 wave 都到达后通过 `s_barrier_wait 0xffff`；
4. 才开始读取本 WG LDS 中由 A/B/SA/SB TDM 准备的数据。

每个 WG 自己就是四种 multicast 的 requester，各 specialized wave 会收到
自己的 Tensor-Done；因此 hot K-loop 用 WG barrier 即可保证本 WG 的 LDS
readiness，不需要每 body 做 cluster barrier。晚到 requester 即使因 timeout
单独返回，也会等待自己的 done；影响的是合并效率，不是正确性。最后两句是由
当前 requester/control-flow 推出的 ISA 结论。

### 7.3 Cluster barrier：启动、排空与 task 边界

`-3/0xfffd` 是 cluster user barrier。硬件按 WG 计数，建议先做 WG barrier，
再由每 WG 一个 wave signal，所有 wave wait：
CDNA5 ISA L3319–L3327。

当前 kernel 正好这样使用：

- 启动：wave0 在 ISA L278–L282 / `0x1E5C–0x1E70` 先等 WG barrier，
  signal cluster barrier，再等待并发首个 A TDM；wave1/2/3 只等待
  cluster barrier，例如 L1995–L1998。
- 排空/切换：wave0 在 ISA L1802–L1806 / `0x47CC–0x47DC` 先
  `s_wait_tensorcnt 0`，再依次完成 WG 和 cluster barrier。

职责区分：

| 同步原语 | 范围 | 本 kernel 用途 |
|---|---|---|
| `s_wait_dscnt` | 单 wave 的 LDS dependency | 约束 DS consumer readiness |
| `s_wait_tensorcnt` | 单 requester wave | 约束 TDM completion |
| WG barrier `-1` | 一个 WG 的 4 个 wave | 汇合 A/B/SA/SB readiness |
| cluster barrier `-3` | 4×4 的 16 个 WG | cluster 启动、排空、persistent task 边界 |

<a id="counts-limits"></a>
## 8. 计数汇总、事实边界与限制

### 8.1 Steady body 计数

以下计数来自 16 份 `4 wave × 4 phase` body 的逐段核对；一次动态 body 只走
当前 wave 的一个 phase 副本。

| 单位 | TDM | `ds_load_b128` | `ds_load_b32` | `ds_load_2addr_b32` | WMMA |
|---|---:|---:|---:|---:|---:|
| 每 wave、每 K=128 body | 1 | 48 | 8 | 0 | 64 |
| 每 WG、每 body（4 waves） | 4 | 192 | 32 | 0 | 256 |
| 每 logical cluster、每 body（16 WG） | 64 | 3072 | 512 | 0 | 4096 |

每 wave/body 另有：

- 153 条反汇编指令（按 phase Vaddr 范围静态计数）；
- 5 条 `s_wait_dscnt`；
- 1 条 `s_wait_tensorcnt 2`；
- 1 次 WG barrier signal/wait 配对；
- 3 条 K-loop 控制指令。

仅计算 56 个 steady K-body，不含 prologue 初始填充和 epilogue store：

```text
per wave: 56 * 64 = 3584 WMMA
per WG:   4 * 3584 = 14336 WMMA
```

算术量核对：

```text
14336 * (2 * 16 * 16 * 128)
= 2 * 256 * 256 * 7168
```

与一个 WG 完整 `256×256×7168` GEMM 的 FLOP 数一致。

### 8.2 不应混淆的数量

```text
_4x4                         = cluster_x × cluster_y
4 waves                      = block128 / wave32
576                          = logical WG output tasks
36                           = logical cluster tasks
(64,4,1) / 256 WG / 16 cl.  = physical persistent launch
56                           = K=128 steady bodies
64 TDM/cluster/body          = shader issue，不是 HBM transaction
```

### 8.3 分析边界

- CO 选择、分支路径、K 步长、指令形式与静态条数可由当前源码/ISA确定。
- 3/5 轮结论依赖当前 `multiProcessorCount=256`；设备属性变化时应重新代入
  selector 公式。
- `DScnt`、`TENSORcnt` 的精确逐 cycle 值和 wait latency 不能由静态 ISA
  确定；本文只给 wait 后上界与发射增量。
- multicast 的匹配、timeout 和 cache 行为决定实际 GL1/GL2 收益；本文不把
  理论 4×复用上限外推为 HBM 单次读取或 4×性能。
- 全文件静态计数包含四个 wave 专用副本和四个 phase 副本，不能当作一次
  动态 K-body。

### 8.4 主要证据文件

- `op_tests/test_mxfp8fp4gemm.py`
- `aiter/ops/shuffle.py`
- `aiter/ops/gemm_op_a8w4.py`
- `csrc/py_itfs_cu/asm_mxfp8fp4gemm.cu`
- `hsa/gfx1250/mxfp8fp4gemm/mxfp8fp4gemm.csv`
- `my_code/fmha/dump_asm/hsa/gfx1250/mxfp8fp4gemm/f8gemm_bf16_mxfp8fp4_ABpreShuffle_256x256_4x4_ps.s`
- `C:\Users\yanguahe\Documents\hardware_file\MI450\amd-instinct-cdna5-instruction-set-architecture.txt`
- `C:\Users\yanguahe\Documents\code\llm-wiki\mi400_hw_wiki\raw\papers\mi400_hd_txt\architecture\subsystem\SH\MI400_Shader_Programming#65.txt`
- `C:\Users\yanguahe\Documents\code\llm-wiki\mi400_hw_wiki\raw\papers\mi400_hd_txt\subsystem\CMM\GLX\Design\MI400_Multicast_Feature#9.txt`
