# gfx1250 MXFP4×MXFP4 256×256_4×4 Persistent GEMM 解析

本文解析
`f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps`
在 `M=18432, N=2048, K=7168, a4w4, A-preshuffle=1, output=bf16`
下的调用入口、preload ABI、CO/symbol 选择、tile/grid、K-loop、四槽 LDS ring、
`v_wmma_scale_f32_32x16x128_f4`、TDM cluster multicast 与同步。

分析基于仓库当前源码、7043 行反汇编和本地 CDNA5/MI400 硬件资料。文中的
“ISA Lx / Vaddr”均指：

`my_code/fmha/dump_asm/hsa/gfx1250/f4gemm/f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.s`

术语约定：

- host `A`：MXFP4 activation，原 GEMM 的 M 方向；
- host `B`：MXFP4 weight，原 GEMM 的 N 方向；
- `ScaleA/ScaleB`：分别记作 `SA/SB`；
- WMMA 的 Matrix A 是 host `B`，WMMA 的 Matrix B 是 host `A`；
- “文档事实”来自硬件资料；“源码事实”来自 Python/C++/CSV；
  “ISA 推断”来自当前二进制的数据流和控制流。

<a id="summary"></a>
## 结论摘要

- 正确测试入口是 `op_tests/test_f4gemm.py`。对本 shape 应使用
  `--intype mxfp4 --apre 1 -mnk 18432,2048,7168 -d bf16`；该测试同时跑
  unified `gemm_a4w4` 和强制 symbol 的低层 `asm` 候选。它没有导入
  `bench_init.py`，所以这里的输入开关是 `--init constant/random`，不是
  `--data-init/--scale-init`。
- CSV 中 `intype=7,a_preshuffle=1` 只有一个候选，最终 CO 是
  `f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.co`，symbol 是
  `_ZN5aiter45f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_psE`。
- `_4x4` 是 **4×4 workgroup cluster**。一个 WG 为 block128，即 4 个
  wave32；每个 WG 计算 `256×256`，四个 wave 按 2×2 象限各负责
  `128×128`，并额外按 wave0/1/2/3 分担 A/B/SA/SB 的 TDM。
- F4 的真实 K-body 是 **K=256**，不是照搬 F8 的 K=128。每 body 内有两个
  K=128 WMMA 子步；`K=7168` 恰好是 28 个 body，即四个 phase/slot 循环 7
  轮，最终从 P3 退出。
- 每 wave、每 K=256 body 动态执行 64 条
  `v_wmma_scale_f32_32x16x128_f4`。它们是 32 个独立
  `N32×M16` accumulator fragment，每个 fragment 分别用 K0、K1 两条
  WMMA 累加，故输出仍是 `M128×N128`，不是 64 个独立输出 fragment。
- 硬件 Matrix A 为 `32×128` FP4、使用 16 个 VGPR，来自 host B；
  Matrix B 为 `128×16` FP4、使用 8 个 VGPR，来自 host A。S3 是
  Matrix A/host B 的 `32×4` scale；S4 是 Matrix B/host A 的
  `4×16` scale，并由 `matrix_b_scale:MATRIX_SCALE_ROW1` 在
  lanes 0..15 与 16..31 间选 row。
- 每条 `ds_load_b128` 的 wave 聚合 payload 是 512 B，对应当前 preshuffle
  的 `rows16×K64` FP4；每条 `ds_load_b32` 聚合 128 B，对应
  `rows32×K128` 的 E8M0 scale。一个 bundle 是
  `16×b128 + 4×b32 = rows64×K256`。
- 每 wave/body 有 4 个 bundle，即 64 条 `ds_load_b128`、16 条
  `ds_load_b32`、0 条 `ds_load_2addr_b32`、1 条
  `tensor_load_to_lds`、6 条 `s_wait_dscnt` 和 64 条 WMMA。
- wave0/2 的 DScnt wait 上界序列为
  `8→20→8→20→8→10`；wave1/3 为
  `8→20→8→20→8→4`。所有数字都只是 wait 返回后的上界，不是精确
  outstanding 数。
- 逻辑输出 grid 是 `(8,72,1)`，共 576 个 logical WG task、36 个
  logical cluster task。F4 launcher 的实际 persistent launch 是
  **`(16,16,1)`、256 WG、16 cluster**；物理 cluster `p` 处理
  `p,p+16,p+32,...`。
- multicast mask 与 F8 的拓扑相同：A/SA 用 `0xf << (4*y)`，B/SB 用
  `0x1111 << x`；每 cluster、每 K=256 body 仍有 64 条 shader TDM issue。
  但 F4 的 K-body、TDM payload、ring 地址、DS/WMMA pipeline、周期性
  cluster barrier 和实际 launch 形状都不同，二者绝非完整同构。

<a id="toc"></a>
## 目录

1. [测试入口、selector、CO 与 preload ABI](#kernel-selection)
2. [WMMA 契约、wave tile 与 TDM specialization](#tile-wave)
3. [4×4 cluster、逻辑任务与 persistent grid](#cluster-grid)
4. [K=256 body、四槽 ring 与 DS bundle](#kloop-ring)
   - [4.4 单条 DS 与 preshuffle tile](#ds-unit)
   - [4.5 Bundle 顺序和跨 body 滚动](#ds-bundle-order)
   - [4.6 WMMA src 笛卡尔积独立核验](#wmma-cartesian)
5. [真实 1..9 steady pipeline](#steady-pipeline)
6. [Cluster TDM multicast](#cluster-multicast)
7. [DScnt、TENSORcnt、WG/cluster barrier](#synchronization)
8. [计数汇总与 F8 关键差异](#counts-differences)
9. [分析边界与证据文件](#limits-evidence)

<a id="kernel-selection"></a>
## 1. 测试入口、selector、CO 与 preload ABI

### 1.1 正确测试入口

当前 gfx1250 F4GEMM 专用测试是：

```text
op_tests/test_f4gemm.py
```

针对本文 shape 的命令形式为：

```text
python op_tests/test_f4gemm.py \
  --intype mxfp4 --apre 1 \
  --init constant random \
  -mnk 18432,2048,7168 -d bf16
```

本文遵照任务约束没有实际运行 GPU。入口和数据准备的源码事实如下：

1. `_prep_mxfp4` 生成 packed FP4 `A[M,K/2]`、`B[N,K/2]` 和
   E8M0 `SA[M,K/32]`、`SB[N,K/32]`：
   `op_tests/test_f4gemm.py` L62–L89。
2. `apre=1` 时 A 使用 `shuffle_weight_f4`，B 总是使用同一 preshuffle；
   SA/SB 使用 `shuffle_scale_f4(...,7)`：L81–L85。
3. unified 候选调用 `aiter.gemm_a4w4`，传
   `apreshuffle=True,bpreshuffle=True` 且不传 global scale：
   L127–L142。因此 gfx1250 wrapper 进入 MXFP4 分支：
   `aiter/ops/gemm_op_a4w4.py` L120–L166。
4. `asm` 候选直接构造长度为 45 的 mangled symbol 并传给
   `gemm_mxfp4_asm`：`test_f4gemm.py` L144–L169。

`op_tests/bench_init.py` L3–L28 定义了独立的 data/scale 分布轴，但
`test_f4gemm.py` 没有导入它。当前 F4 测试的 `constant` 路径直接填
`A=0x22,B=0x33,scale=0x7f`，`random` 路径调用 per-1×32 quant
（`test_f4gemm.py` L62–L80）；因此不能把其他 benchmark 的
`--data-init`、`--scale-init` 或 seed 语义写到这个入口上。

### 1.2 Selector 与唯一候选

`gemm_a4w4` 在 gfx1250 且没有 NVFP4 global scale 时调用
`gemm_mxfp4_asm`，默认 `kernelName=""` 被转成 `None`：
`gemm_op_a4w4.py` L146–L166、L297–L318。C++ 随后：

1. 用 `intype=7`、`a_preshuffle=1` 过滤：
   `csrc/py_itfs_cu/asm_f4gemm.cu` L59–L79。
2. 要求 M/N 被 tile 整除，并要求 tile grid 被 cluster 维度整除：
   L80–L93。
3. 用 `ceil(tg_num/num_cu)`、空余 CU 和
   `tile_m*tile_n/(tile_m+tile_n)` 评分：L95–L113。

CSV 对 MXFP4/apre1 只有一行：
`hsa/gfx1250/f4gemm/f4gemm.csv` L3。

| 项 | 当前值 |
|---|---:|
| `tile_m × tile_n` | `256×256` |
| `intype` | `7`（MXFP4） |
| `a_preshuffle` | `1` |
| `cluster_x × cluster_y` | `4×4` |
| M tiles | `18432/256 = 72` |
| N tiles | `2048/256 = 8` |
| logical WG tiles | `72×8 = 576` |

`72%4==0` 且 `8%4==0`，所以候选合法。由于没有第二个 MXFP4/apre1
候选，`num_cu` 的评分值不会改变本 shape 的选择。

### 1.3 最终 symbol 与 CO

最终选择为：

```text
symbol:
_ZN5aiter45f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_psE

CO:
hsa/gfx1250/f4gemm/
f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.co
```

证据闭环：

- CSV L3 同时登记 `knl_name` 与 `co_name`；
- C++ L248–L250 把两者交给 `AiterAsmKernel`；
- ISA L2 的 ELF 来源路径就是该 CO；
- ISA L6 / Vaddr `0x1900` 的入口 symbol 精确匹配。

### 1.4 MXFP4 preload ABI

该 CO 使用 preload SGPR 模式。C++ packed `KernelArgs` 的逻辑顺序见
`asm_f4gemm.cu` L34–L57；MXFP4 发送 80 B，尾部 dw18/19 不作 global
scale，而复用为 persistent `log2_grid_x/log2_grid_y`
（L195–L200、L317–L329）。

| preload SGPR | kernarg dw | 含义 |
|---|---:|---|
| `s[2:3]` | 0..1 | `ptr_D` |
| `s[4:5]` | 2..3 | `ptr_A` |
| `s[6:7]` | 4..5 | `ptr_B` |
| `s[8:9]` | 6..7 | `ptr_ScaleA` |
| `s[10:11]` | 8..9 | `ptr_ScaleB` |
| `s12` | 10 | `strideD0`（byte） |
| `s13..s16` | 11..14 | A/B/SA/SB byte stride |
| `s17,s18,s19` | 15..17 | M、N、K |
| `s20,s21` | 18..19 | `log2_grid_x`,`log2_grid_y` |

ISA L10–L12 立即使用 `s[2:3]` 和 wave ID，L43–L56 使用
`s17/s18/s19` 构造 logical task 数，和上述 ABI 一致。

### 1.5 Shape 约束的事实边界

需要区分三层约束：

- Python 测试显式检查 MXFP4 `K%32==0`：
  `test_f4gemm.py` L120–L125。
- `shuffle_weight_f4` 要求 rows%16、packed-K%16，即 K%32：
  `aiter/ops/shuffle.py` L318–L335。
- `shuffle_scale_f4(...,7)` 以 `32×4` tile reshape scale；
  当前无 padding 路径，因此实际测试输入还需 rows%32 且
  `(K/32)%4==0`，即 K%128：L292–L315。
- C++ selector/launcher 对本 4×4 变体进一步要求
  `M%(256×4)==0`、`N%(256×4)==0`：
  `asm_f4gemm.cu` L265–L279。

当前 `M=18432,N=2048,K=7168` 同时满足这些条件，并且 K 还恰好被 256
整除，所以本文的动态路径没有 K-tail。C++ 对 MXFP4 只显式检查
`K%32==0`（L157–L166）；本文不据此臆测当前二进制对所有
`K mod 256` 的数值语义，只分析无 tail 的 K=7168 路径。

<a id="tile-wave"></a>
## 2. WMMA 契约、wave tile 与 TDM specialization

### 2.1 `32×16×128 F4` 的硬件契约

CDNA5 ISA 文档 L26180–L26205 明确：

```text
D = MatrixA(32×128, FP4) × MatrixB(128×16, FP4)
    + C(32×16, F32)
```

每条指令由整个 wave32 合作完成一个 `32×16×128` 矩阵乘。矩阵布局文档
L7586–L7632 又给出：

- 16×128 FP4 Matrix A 使用 8 个 VGPR；
- 32×128 是两个连续的 16×128，因此 Matrix A 使用 **16 VGPR**；
- 同一布局表可按文档 L7360–L7361 转用于 Matrix B；128×16 FP4 的总
  payload 为 1024 B，即 wave32 每 lane 32 B，因此使用 **8 VGPR**；
- 32×16 F32 C/D 使用 16 个 VGPR：L7409。

MI400 Shader Guide L10215–L10223 和 L11037–L11048 给出同一尺寸与
SRC0/SRC1 角色。当前 ISA 的首条实指令（L4846 /
`0x7168`）也正是 `srcA=v[776:791]` 16 个 VGPR、
`srcB=v[520:527]` 8 个 VGPR、`d/c=v[100:115]` 16 个 VGPR。

### 2.2 硬件 A/B 与 host A/B 的反向角色

当前数据流为：

| 指令输入 | 硬件矩阵 | 当前 kernel 数据 | 原 GEMM 方向 |
|---|---|---|---|
| `srcA` / S3 | Matrix A `32×128` | host B、SB | N32×K128 |
| `srcB` / S4 | Matrix B `128×16` | host A、SA | K128×M16 |
| C/D | `32×16` | 输出 fragment | N32×M16 |

证据不是按名称猜测：

- prologue 的 host A LDS load 写入物理 `v520...`：
  ISA L984–L999 / `0x29FC–0x2A74`；
- host B LDS load 写入物理 `v776...`：
  L1001–L1016 / `0x2A80–0x2AF8`；
- L4846 把 `v776:791` 放在第一输入、`v520:527` 放在第二输入。

因此硬件输出行是原 N、列是原 M。本文仍按 GEMM 接口惯例把 wave tile
写成 `M×N`。

### 2.3 4 个 wave 与 2×2 输出象限

C++ 固定 `bdx=128`，注释为 `4 wave * 32 thread`：
`asm_f4gemm.cu` L332–L335。ISA L7 也设置
`UC_VERSION_W32_BIT`，所以每 WG 恰有 4 个 wave。

输出基址在 ISA L273–L287 / `0x1DF0–0x1E38` 构造：

```text
Mbase = 256 * Mtile + 128 * (wave_id & 1)
Nbase = 256 * Ntile + 128 * (wave_id >> 1)
```

| wave | 相对 M | 相对 N | TDM specialization |
|---:|---|---|---|
| 0 | `[0,127]` | `[0,127]` | host A data |
| 1 | `[128,255]` | `[0,127]` | host B data |
| 2 | `[0,127]` | `[128,255]` | SA |
| 3 | `[128,255]` | `[128,255]` | SB |

wave 分支入口是 ISA L321–L328 / `0x1EE8–0x1F04`。四个 specialized
入口的首个 TDM 分别为：

| wave | LDS slot0 anchor | 首个 TDM |
|---:|---:|---|
| 0 / A | `0x00000` | L412 / `0x20C0` |
| 1 / B | `0x30000` | L1634 / `0x3530` |
| 2 / SA | `0x10000` | L2336 / `0x417C` |
| 3 / SB | `0x22000` | L3044 / `0x4DE4` |

specialization 只分担 global→LDS 工作；四个 wave 都进入 WMMA 主体并各自
计算一个 `128×128` 象限。

### 2.4 S3/S4 的 lane 布局与 row 选择

CDNA5 ISA L26191–L26205 直接定义 S3 是 Matrix A 的 `32×4` scale
matrix、S4 是 Matrix B 的 `4×16` scale matrix，每个 8-bit scale 覆盖
K32，并由 `SCALE_OPSEL_HI[0]` 选择 S4 的部分。再结合 wave32、一条
`ds_load_b32` 的 128 B aggregate 和当前 row0/row1 指令配对，可推出本
kernel 的 lane 布局：

- S3：32 lanes 各对应一个 Matrix A row，每 lane 的 32 bit 打包四个
  8-bit scale，分别覆盖四个 K32 block；
- S4：一个 wave32 VGPR 同时装两份 16-column scale，lanes 0..15 是
  row0，lanes 16..31 是 row1；`SCALE_OPSEL_HI[0]` 选择二者。

映射回当前 kernel：

```text
S3=v92, lanes 0..31
  -> host B local N[0:31]，每 lane 4×E8M0，覆盖 K[0:127]

S4=v82, lanes 0..15
  -> host A local M[0:15]，每 lane 4×E8M0，覆盖 K[0:127]

S4=v82, lanes 16..31
  -> host A local M[16:31]，每 lane 4×E8M0，覆盖 K[0:127]
```

ISA L4846 无 row1 suffix，读取 S4 row0；L4849 /
`0x7188` 对同一个 `v82` 使用
`matrix_b_scale:MATRIX_SCALE_ROW1`，读取 row1。全文件没有
`matrix_a_scale:MATRIX_SCALE_ROW1`：32-row Matrix A 的 S3 已使用全部
32 lanes，不需要二选一。

这里的 `matrix_a_reuse`、`matrix_b_reuse` 是 operand cache reuse hint，
不是 scale row；CDNA5 ISA L7140–L7143 规定 reuse hint 只在与前一条
相应矩阵相同的情况下使用。F4 opcode 的数据类型固定为 FP4，CDNA5 ISA
L26202–L26205 又规定普通 OPSEL/OPSEL_HI 必须为 0，scale selector 是被
重载的独立语义，不能把 row1 suffix解释为 FP4 nibble OPSEL。

<a id="cluster-grid"></a>
## 3. 4×4 cluster、逻辑任务与 persistent grid

### 3.1 一个 logical cluster task

CSV L3 指定 `tile=256×256, cluster=4×4`。C++ 中 x 对应 N tiles、y
对应 M tiles（`asm_f4gemm.cu` L265–L266），所以：

```text
cluster N = 4 * 256 = 1024
cluster M = 4 * 256 = 1024
```

一个 logical cluster task 有 16 个 WG，完整输出区域为 `1024×1024`。
在一个 F4 K-body 内，它覆盖 `1024×1024×256`；对本 shape 的完整 K
则覆盖 `1024×1024×7168`。

### 3.2 逻辑 grid

```text
N tiles = 2048 / 256 = 8
M tiles = 18432 / 256 = 72

logical WG grid = (8,72,1)
logical WG tasks = 8 * 72 = 576

N cluster tasks = 8 / 4 = 2
M cluster tasks = 72 / 4 = 18
logical cluster tasks = 2 * 18 = 36
```

ISA L45–L57 计算每个维度的 cluster-task 数和总数 `s29`；L164–L203
把扁平 task ID 分解为 cluster N/M，再叠加 `wg_x/wg_y` 得到当前
`s54=Ntile,s55=Mtile`。

### 3.3 F4 的实际 persistent launch 是 `(16,16,1)`

F4 launcher 与 F8 文档中的旧 launch 形状不同。源码固定：

```text
PERSISTENT_TG = 256 WG
cluster_size = 4 * 4 = 16 WG
physical clusters = 256 / 16 = 16
PERSISTENT_GY = 4 cluster rows
gridX = 16 / 4 = 4 cluster columns

HIP WG grid:
gdx = gridX * cluster_x = 4 * 4 = 16
gdy = gridY * cluster_y = 4 * 4 = 16
gdz = 1
```

证据为 `asm_f4gemm.cu` L258–L315、L332–L335。

| 名称 | 值 |
|---|---:|
| logical WG grid | `(8,72,1)` |
| logical WG tasks | 576 |
| logical cluster tasks | 36 |
| physical cluster grid | `(4,4,1)` |
| physical WG launch | `(16,16,1)` |
| launched WG / cluster | 256 / 16 |
| block | `(128,1,1)` |

MXFP4 把 `log2_grid_x=2,log2_grid_y=2` 放入 dw18/19。ISA L204–L207 /
`0x1CB8–0x1CC4` 计算：

```text
persistent stride = 1 << (2 + 2) = 16 cluster tasks
```

物理 cluster `p` 因而处理：

```text
p, p+16, p+32, ...  (task < 36)
```

- `p=0..3`：各处理 3 个 logical cluster task；
- `p=4..15`：各处理 2 个；
- 三轮覆盖 256、256、64 个 logical WG task，总计 576。

task epilogue 在 ISA L6835–L6938 恢复下一 task 的 tile/pointer；其中
L6872–L6877 再次以 16 为步长预计算后继 task。不能把 physical grid
`16×16` 误写成逻辑输出 grid。

<a id="kloop-ring"></a>
## 4. K=256 body、四槽 ring 与 DS bundle

### 4.1 28 个 body、每 body 两个 WMMA K 子步

控制变量 `s58` 在每个 phase 尾执行：

```text
s58 += 0x100
compare s58 < K
```

代表性位置是 ISA L5021–L5028 / `0x7834–0x788C`。因此：

```text
body K step = 0x100 = 256
7168 / 256 = 28 bodies
28 = 7 * 4 phases
```

单条 WMMA 的 K 固定为 128，所以一个 body 有：

```text
K0 = body K[0:127]
K1 = body K[128:255]
```

同一 accumulator fragment 会分别执行 K0 和 K1 两条 WMMA。例如
L4846 / `0x7168` 与 L4852 / `0x71A8` 都读写物理
`d/c=v[100:115]`，但 operand/scale 分别是 K0、K1。

### 4.2 两组 wave、四个 phase 与实际回边

编译器共享了两套主体：

- wave0/2 使用第一套；
- wave1/3 使用第二套。

| wave 组 | phase | 核心 ISA 行 | Vaddr |
|---|---|---|---|
| 0/2 | P0 | L4844–L5028 | `0x7160–0x788C` |
| 0/2 | P1 | L5031–L5215 | `0x7898–0x7FC4` |
| 0/2 | P2 | L5218–L5402 | `0x7FD0–0x86FC` |
| 0/2 | P3 | L5405–L5594 | `0x8708–0x8E48` |
| 1/3 | P0 | L5605–L5789 | `0x8E74–0x95A0` |
| 1/3 | P1 | L5792–L5976 | `0x95AC–0x9CD8` |
| 1/3 | P2 | L5979–L6163 | `0x9CE4–0xA410` |
| 1/3 | P3 | L6166–L6352 | `0xA41C–0xAB50` |

P3 继续时：

- wave0/2：L5594 / `0x8E48` 回到 `0x7158`；
- wave1/3：L6352 / `0xAB50` 回到 `0x8E6C`。

本 shape 动态执行 `P0→P1→P2→P3` 七轮，最终 K 比较在 P3 变假。
P0/P1/P2 为其他 K 终点准备的退出支路在本 shape 不走。

### 4.3 四槽 LDS ring

TDM slot anchors 直接由四个 specialized 入口给出：

| operand | slot0 | slot1 | slot2 | slot3 | 单 slot payload |
|---|---:|---:|---:|---:|---:|
| A data | `0x00000` | `0x08000` | `0x12000` | `0x1A000` | 32 KiB |
| SA | `0x10000` | `0x10800` | `0x11000` | `0x11800` | 2 KiB |
| SB | `0x22000` | `0x22800` | `0x23000` | `0x23800` | 2 KiB |
| B data | `0x30000` | `0x38000` | `0x40000` | `0x48000` | 32 KiB |

payload 可独立核算：

```text
FP4 data: 256 rows * K256 / 2 = 32768 B = 0x8000
E8M0 scale: 256 rows * (256/32) = 2048 B = 0x800
```

A slot1 与 slot2 之间插入 SA ring；SB ring 与 B data ring 之间还有对齐
空洞。当前代码使用的 LDS 地址窗口延伸到 `0x50000`，但反汇编没有
kernel metadata，所以这只是 **地址布局事实**，不是对声明
`group_segment_fixed_size` 的猜测。

prologue 先滚动装入 slot0/1/2，再在首批 DS 之后装入 slot3；以 wave0 为例，
前三条 TDM 在 ISA L412、L427、L442，slot3 在 L1018。steady P0 在
L4937 回填刚消费的 slot0，P1/P2/P3 分别回填 slot1/2/3。因而 ring 的
生产距离是四个 K=256 body，而不是四个 K=128 WMMA 子步。

<a id="ds-unit"></a>
### 4.4 单条 DS 与 preshuffle tile

#### 4.4.1 Wave aggregate bytes

CDNA5 ISA L35616–L35621、L36747–L36755 规定每 lane 的
`DS_LOAD_B32`/`B128` 返回 4/16 B。当前 wave32 因而有：

| DS | 单 lane | wave aggregate |
|---|---:|---:|
| `ds_load_b128` | 16 B | 512 B |
| `ds_load_b32` | 4 B | 128 B |

#### 4.4.2 FP4 data：单条 `rows16×K64`

`shuffle_weight_f4` 把 `[rows,K/2 packed bytes]` 排为
`[rows/16,(K/2)/16,16,16]`：
`aiter/ops/shuffle.py` L318–L335。一个 preshuffle tile 是：

```text
16 rows * 16 packed bytes
= 256 B
= rows16 * K32 FP4
```

ISA 的 DS lane 地址连续覆盖 512 B，例如 A 的 L984–L985 /
`0x29FC–0x2A04`。所以一条 wave aggregate `b128` 跨相邻两个 256 B
tile，逻辑覆盖：

```text
rows16 × K64 FP4 = 1024 FP4 = 512 B
```

这个结论对 A/B 都相同；区别只在 WMMA 组装：

- Matrix A/host B 的 `N32×K128` 需要 `2×2=4` 条 `b128`；
- Matrix B/host A 的 `M16×K128` 需要 2 条 `b128`。

例如首个 Matrix A src `v776:791` 来自 L1001–L1004 的 offsets
`0,512,2048,2560`；首个 Matrix B src `v520:527` 来自
L984–L985 的 offsets `0,512`。

#### 4.4.3 Scale：单条 `rows32×K128`

MXFP4 scale 输入是 `[rows,K/32]`，`shuffle_scale_f4(...,7)` 排为
`[rows/32,(K/32)/4,32,4]`：`shuffle.py` L292–L315。一个 tile：

```text
32 rows * 4 E8M0 bytes = 128 B
```

正好由一条 wave aggregate `ds_load_b32` 读取，因此逻辑覆盖
`rows32×K128`。ISA L975–L982 / `0x29B8–0x29F0` 的 8 条 scale load
可解释为：

```text
SA: 4 个 M32×K128 tile
SB: 4 个 N32×K128 tile
```

它们共同准备当前 wave 的 `M128/N128 × K128` 首个子步。

#### 4.4.4 一个 bundle：`rows64×K256`

严格定义一个 A 或 B bundle 为一个 operand 在非 K 维的 64-row 半片，
但覆盖完整 K=256 body：

| bundle 组成 | 数量 | bytes | 逻辑覆盖 |
|---|---:|---:|---|
| FP4 data | `16×b128` | 8192 | `rows64×K256` |
| E8M0 scale | `4×b32` | 512 | `rows64×(K256/32)` |
| 合计 | 20 DS | 8704 | 一个完整 operand bundle |

当前 wave 的 A 有 M 前/后两个 bundle，B 有 N 前/后两个 bundle。
每个 K=256 body 因此需要四个 bundle，总计
`64×b128 + 16×b32`。

<a id="ds-bundle-order"></a>
### 4.5 Bundle 顺序和跨 body 滚动

“前/后半”在本节只指当前 wave 的非 K 维：

```text
A 前/后半 = M[0:63] / M[64:127]
B 前/后半 = N[0:63] / N[64:127]
```

每个半片都覆盖完整 K256，不能误写成 K128 半片。

P0 入口前，prologue 已准备 `A_current 前半 + B_current 前半`。
wave0/2 的 body 内顺序为：

```text
B_current 后半
→ A_current 后半
→ A_next 前半
→ B_next 前半
```

代表性地址：

| bundle | scale | data |
|---|---|---|
| B current 后半 | L4847–L4851 / `0x7178–0x71A0` | L4854–L4873 / `0x71C8–0x7280` |
| A current 后半 | L4887–L4893 / `0x7328–0x7358` | L4897–L4916 / `0x7384–0x743C` |
| A next 前半 | L4940–L4946 / `0x7550–0x7580` | L4949–L4969 / `0x759C–0x7658` |
| B next 前半 | L4977–L4983 / `0x76B8–0x76E8` | L4987–L5006 / `0x7714–0x77CC` |

wave1/3 交换 A/B 顺序：

```text
A_current 后半
→ B_current 后半
→ B_next 前半
→ A_next 前半
```

代表性地址为：

- A current：L5608–L5633 / `0x8E8C–0x8F84`；
- B current：L5648–L5676 / `0x903C–0x9140`；
- B next：L5701–L5731 / `0x9264–0x937C`；
- A next：L5738–L5766 / `0x93CC–0x94D0`。

跨 phase 后，`next 前半` 成为下一 body 的 `current 前半`；下一 body
再补它的 current 后半。这就是每 body 只显式发四个 bundle、却能持续
供应完整 A/B wave tile 的滚动关系。

<a id="wmma-cartesian"></a>
### 4.6 WMMA src 唯一组与笛卡尔积独立核验

本节不使用上一节的 DS payload 形状，只使用硬件 WMMA 契约、物理 src
VGPR、scale row 和独立 C/D fragment。

CDNA5 ISA L1155–L1197 规定 `s_set_vgpr_msb` 把逻辑 8-bit VGPR 编号
扩展到 0..1023；以下均采用反汇编注释已经还原的物理 VGPR。

#### 4.6.1 K0 的四个 Matrix A/host B 组

| 原 N 区间 | Matrix A src（16 VGPR） | S3 |
|---|---|---|
| `[0,31]` | `v776:791` | `v92` |
| `[32,63]` | `v808:823` | `v94` |
| `[64,95]` | `v8:23` | `v96` |
| `[96,127]` | `v40:55` | `v98` |

#### 4.6.2 K0 的八个 Matrix B/host A 组

| 原 M 区间 | Matrix B src（8 VGPR） | S4 / row |
|---|---|---|
| `[0,15]` | `v520:527` | `v82 / row0` |
| `[16,31]` | `v536:543` | `v82 / row1` |
| `[32,47]` | `v264:271` | `v86 / row0` |
| `[48,63]` | `v280:287` | `v86 / row1` |
| `[64,79]` | `v552:559` | `v84 / row0` |
| `[80,95]` | `v568:575` | `v84 / row1` |
| `[96,111]` | `v296:303` | `v88 / row0` |
| `[112,127]` | `v312:319` | `v88 / row1` |

ISA 在 P0 中完整枚举 `4×8=32` 个 K0 组合。代表点包括：

| 输出 fragment | ISA | srcA(B) × srcB(A) |
|---|---|---|
| N0×M0 | L4846 / `0x7168` | `v776:791 × v520:527` |
| N0×M16 | L4849 / `0x7188` | `v776:791 × v536:543` |
| N32×M0 | L4875 / `0x728C` | `v808:823 × v520:527` |
| N64×M0 | L4885 / `0x7314` | `v8:23 × v520:527` |
| N0×M32 | L4930 / `0x74D8` | `v776:791 × v264:271` |
| N96×M112 | L5025 / `0x785C` | `v40:55 × v312:319` |

指令交错安排 K0/K1，不能按连续行数机械分组。按 src/scale 配对筛出
K0 后，四个 N32 组和八个 M16 组的每个组合都恰好
出现一次，并写入不同的 16-VGPR C/D fragment。因此 K0 独立覆盖：

```text
(8 * M16) × (4 * N32) × K128
= M128 × N128 × K128
```

#### 4.6.3 K1 对同一 32 个 fragment 再累加

K1 把上述组替换为：

```text
B/N: v792:807, v824:839, v24:39, v56:71
      with S3=v93,v95,v97,v99

A/M: v528:535, v544:551, v272:279, v288:295,
     v560:567, v576:583, v304:311, v320:327
     with S4=v83,v87,v85,v89 and row0/row1
```

例如 L4852 / `0x71A8` 与 L4846 写同一 `v100:115`；前者是 K1，
后者是 K0。故一个 K=256 body 的 64 条 WMMA 是：

```text
32 output fragments * 2 K128 accumulations
```

而不是 `64×(32×16)` 个互不相同的输出 tile。32 个 fragment 各含
`32×16` 元素，正好：

```text
32 * 32 * 16 = 16384 = 128 * 128
```

这独立核验了单 wave 的 `M128×N128` 输出 tile。

<a id="steady-pipeline"></a>
## 5. 真实 1..9 steady pipeline

以下以 wave0/2 的 P0（ISA L4844–L5028）和 wave1/3 的 P0
（L5605–L5789）为基准。CDNA5 ISA L20007–L20019 规定
`s_wait_dscnt n` 只保证返回时 `DScnt≤n`；表中不把任何 wait 写成精确
动态 outstanding 值。

| 步 | wave0/2 P0 | wave1/3 P0 | wait 后上界与动作 |
|---:|---|---|---|
| 1 | L4844–L4873 / `0x7160–0x7280` | L5605–L5634 / `0x8E74–0x8F8C` | `DScnt≤8`；8 WMMA。w0/2 发 B-current 后半 `4×b32+16×b128`；w1/3 发 A-current 后半。 |
| 2 | L4874–L4882 / `0x7288–0x72FC` | L5635–L5643 / `0x8F9C–0x9010` | `DScnt≤20`；8 WMMA，无新 DS。 |
| 3 | L4883–L4916 / `0x730C–0x743C` | L5644–L5677 / `0x9020–0x9148` | `DScnt≤8`；8 WMMA。w0/2 发 A-current 后半；w1/3 发 B-current 后半。 |
| 4 | L4917–L4925 / `0x7444–0x74B8` | L5678–L5686 / `0x9158–0x91CC` | `DScnt≤20`；8 WMMA，无新 DS。 |
| 5 | L4926–L4935 / `0x74C8–0x751C` | L5687–L5696 / `0x91DC–0x9230` | `DScnt≤8`、`TENSORcnt≤2`；WG signal；barrier wait 前穿插 4 WMMA。 |
| 6 | L4936–L4954 / `0x7520–0x75C4` | L5697–L5709 / `0x9234–0x92A0` | WG wait 后再做 4 WMMA；发 1 条 TDM 回填当前 ring slot。w0/2 开始 A-next 前半并发 `4×b32+6×b128`；w1/3 发 B-next 的 `4×b32`。 |
| 7 | L4955–L4973 / `0x75CC–0x7690` | L5710–L5734 / `0x92B0–0x93A4` | w0/2 `DScnt≤10`，补 A-next 剩余 `10×b128`；w1/3 `DScnt≤4`，发 B-next `16×b128`；各 8 WMMA。 |
| 8 | L4974–L5006 / `0x76A0–0x77CC` | L5735–L5767 / `0x93B4–0x94D8` | 8 WMMA。w0/2 发 B-next `4×b32+16×b128`；w1/3 发 A-next；同时推进 TDM descriptor。 |
| 9 | L5007–L5028 / `0x77D4–0x788C` | L5768–L5789 / `0x94E8–0x95A0` | 8 WMMA；`s58+=256`、比较 K、跳下一 phase 或退出。 |

合计核对：

```text
WMMA: 8+8+8+8+4+4+8+8+8 = 64
DS:   4 bundles * (16 b128 + 4 b32)
    = 64 b128 + 16 b32
TDM:  1 tensor_load_to_lds
wait: 6 s_wait_dscnt + 1 s_wait_tensorcnt
WG:   1 signal + 1 wait
```

关键同步片段（wave0/2 P0）：

```text
L4926  0x74C8  s_wait_dscnt 8
L4927  0x74CC  s_wait_tensorcnt 2
L4928  0x74D0  s_barrier_signal -1
...
L4935  0x751C  s_barrier_wait 0xffff
L4937  0x7530  tensor_load_to_lds
```

两套 P0 的 wait 上界序列：

```text
wave0/2: 8 -> 20 -> 8 -> 20 -> 8 -> 10
wave1/3: 8 -> 20 -> 8 -> 20 -> 8 -> 4
```

其中末个 10/4 来自不同的 next-bundle 切分位置：w0/2 已发
`4 scale + 6 data`，w1/3 只发了 4 条 scale。LDS 可在任意两条 shader
指令之间异步完成，所以这些数字始终只是上界。

<a id="cluster-multicast"></a>
## 6. Cluster TDM multicast

### 6.1 WG bit 矩阵

CDNA5 ISA L2134–L2142 给出 TTMP6 中 cluster 维度和 `wg_x/wg_y`。
当前 4×4 的 `WGinCluster` 位序为：

```text
WGinCluster = x + 4*y

             x/N=0   x/N=1   x/N=2   x/N=3
y/M=0        bit0    bit1    bit2    bit3
y/M=1        bit4    bit5    bit6    bit7
y/M=2        bit8    bit9    bit10   bit11
y/M=3        bit12   bit13   bit14   bit15
```

CDNA5 multicast 协议 L9884–L9918 规定 mask bit 对应
`WGinCluster`，且每个被 mask 命中的 WG 都应由一个 wave 发相同请求。

### 6.2 四种 operand 的 mask

| operand | 复用方向 | mask | ISA 构造 |
|---|---|---|---|
| A | 固定 M，跨 4 个 N WG | `0xf << (4*y)` | L377–L387 / `0x2008–0x2040` |
| SA | 固定 M，跨 4 个 N WG | `0xf << (4*y)` | L1786–L1796 / `0x38B4–0x38EC` |
| B | 固定 N，跨 4 个 M WG | `0x1111 << x` | L1080–L1095 / `0x2C54–0x2CA4` |
| SB | 固定 N，跨 4 个 M WG | `0x1111 << x` | L2489–L2504 / `0x4504–0x4554` |

例如：

```text
A/SA, y=2 -> 0x0f00 -> WG 8,9,10,11
B/SB, x=1 -> 0x2222 -> WG 1,5,9,13
```

拓扑与 F8 相同，但 descriptor 的 tile 是 F4 的 K256 payload：

| operand TDM | 每 requester slot payload |
|---|---:|
| A data | `M256×K256 FP4 = 32 KiB` |
| B data | `N256×K256 FP4 = 32 KiB` |
| SA | `M256×8 E8M0 = 2 KiB` |
| SB | `N256×8 E8M0 = 2 KiB` |

### 6.3 每 WG 都发，不是 leader 代发

ISA 只按 WG 内 wave ID 在 L321–L328 分工，没有按 `wg_x/wg_y` 选一个
row/column leader 并抑制其他 WG。每个 WG 都有 wave0/1/2/3，故每个
WG 每 body 发四条 TDM。

一个 logical cluster、一个 K=256 body：

| operand | shader TDM issue | 匹配 requester 组 |
|---|---:|---:|
| A | 16 | 4 个 row 组 |
| SA | 16 | 4 个 row 组 |
| B | 16 | 4 个 column 组 |
| SB | 16 | 4 个 column 组 |
| 合计 | **64** | **16 个四-requester组** |

multicast 不减少 shader 指令数。CDNA5 ISA L10147–L10156 还明确
tensor 指令忽略 EXEC，因此不能用 lane mask 推导某些 WG 不发。

### 6.4 私有 LDS、副本和 GL1 合并上界

文档事实：

1. cluster 最多 16 WG、位于同一 Shader Engine，且每个 WG 在不同 WGP：
   CDNA5 ISA L751–L758。
2. `D#.workgroup_mask!=0` 的 `TENSOR_LOAD_TO_LDS` 使用
   `CLUSTER_LOAD_ASYNC`：CDNA5 ISA L10269–L10279；
   MI400 Guide L14296–L14305。
3. 每个 requester wave 都收到 done，async multicast 返回直接写其所在
   WG 的 LDS：CDNA5 ISA L9888–L9922。
4. GL1 设计资料 `MI400_Multicast_Feature#9.txt` L198–L209、
   L416–L423 说明首个匹配请求发往 GL2，后续命中请求不重复发往 GL2。

因此这是：

```text
一个合并返回
  -> requester WG0 的私有 LDS slot
  -> requester WG1 的私有 LDS slot
  -> requester WG2 的私有 LDS slot
  -> requester WG3 的私有 LDS slot
```

不是 cluster-wide 共享 LDS。16 个 WG 仍各保留独立逻辑副本，普通 DS
不能跨 WG 读取。

四个 requester 在 MI400 Guide L13793–L13794 的“最多合并 5 个请求”
上限内，所以理想 GL1→GL2 请求/返回复用上限是 4×。但它不保证：

- shader TDM issue 减为四分之一；
- LDS 只保留一份；
- HBM transaction 必为一次；
- 带宽或性能必为 4×。

实际合并受请求到达、timeout、cache line 拆分和 GL2 命中影响。

<a id="synchronization"></a>
## 7. DScnt、TENSORcnt、WG/cluster barrier

### 7.1 DScnt 和 TENSORcnt

硬件文档事实：

- DScnt 按 LDS **指令**计数，不按 byte 或 lane：
  CDNA5 ISA L970–L974、L3583–L3586。
- 同一 wave 的 LDS 指令保持完成与 DScnt decrement 顺序：
  CDNA5 ISA L3575–L3586；MI400 Guide L5792–L5805。
- `S_WAIT_DSCNT n` 只保证 `DScnt≤n`：
  CDNA5 ISA L20007–L20019。
- tensor 指令以 TENSORcnt 计数，同 wave 内有序、不同 wave 无序：
  CDNA5 ISA L10147–L10156；MI400 Guide L14097–L14104。
- `S_WAIT_TENSORCNT n` 只保证 `TENSORcnt≤n`：
  CDNA5 ISA L20041–L20045。

所以 hotloop 的 `s_wait_tensorcnt 2` 不是排空，也不能写成“此刻恰好
outstanding 2 条”。它利用同 wave TDM 有序性保护最老 ring slot。

### 7.2 每 body 的 WG barrier

每个 body 中四个 specialized wave 都执行：

1. `DScnt≤8`；
2. `TENSORcnt≤2`；
3. `s_barrier_signal -1`；
4. 在独立 WMMA 间穿插后 `s_barrier_wait 0xffff`；
5. 再发当前 wave 负责的 slot 回填 TDM。

`-1/0xffff` 是 workgroup barrier（CDNA5 ISA L3128–L3149）。
它汇合 A/B/SA/SB 四个 requester wave，使每个 WG 看到自己的四类 LDS
数据准备进度。

### 7.3 P3 ring-wrap 的 cluster barrier

F4 hotloop 不是“只有 task 边界才有 cluster barrier”。P3 中：

- 仅 wave0 在 L5535–L5537 / `0x8C48–0x8C50` 执行
  `s_barrier_signal -3`；
- 若 K-loop 继续，wave0/2 在 L5593、wave1/3 在 L6351 执行
  `s_barrier_wait 0xfffd`，然后回到 P0。

因此每四个 K256 body（每轮四槽 ring）有一次 cluster-wide wrap
同步。CDNA5 ISA L3319–L3327 规定 cluster barrier 按 WG 计数：每 WG
一个 wave signal，所有 wave wait；当前代码正是 wave0 signal、四 wave
wait。

本 shape 有 7 个四-phase cycle：

- 前 6 次 P3 后继续，执行 wrap wait；
- 第 7 次 P3 是最终 K 退出，走 L5595–L6363 的收束路径，再在输出/task
  边界完成 cluster convergence。

startup 还在 wave0 的 L407–L410 先做 WG barrier，再 signal/wait cluster；
其他 wave 在各自入口等待同一 cluster barrier。输出完成后，epilogue 用
`tensor_store_from_lds` 写回（静态实例 L6622、L6809），并在
L6835–L6938 决定是否进入下一个 persistent task。

<a id="counts-differences"></a>
## 8. 计数汇总与 F8 关键差异

### 8.1 单 K=256 steady body

| 单位 | TDM load | `b128` | `b32` | `2addr_b32` | DScnt wait | WMMA |
|---|---:|---:|---:|---:|---:|---:|
| 每 wave | 1 | 64 | 16 | 0 | 6 | 64 |
| 每 WG（4 wave） | 4 | 256 | 64 | 0 | 24 | 256 |
| 每 cluster（16 WG） | 64 | 4096 | 1024 | 0 | 384 | 4096 |

每 wave/body 另有：

- 1 条 `s_wait_tensorcnt 2`；
- 1 次 WG barrier signal/wait；
- 1 次 `s58+=256` 和 K 比较/分支；
- P3 额外参与一次 ring-wrap cluster barrier。

### 8.2 本 shape 的 28-body 动态总量

只计 K hot bodies，不把 prologue 初填和 epilogue store 混入：

```text
per wave:
  WMMA = 28 * 64 = 1792
  b128 = 28 * 64 = 1792
  b32  = 28 * 16 = 448
  TDM load = 28

per WG:
  WMMA = 4 * 1792 = 7168
```

算术核对：

```text
7168 WMMA/WG * (2 * 32 * 16 * 128 FLOP/WMMA)
= 2 * 256 * 256 * 7168
```

与一个 WG 的完整 `256×256×7168` GEMM 完全一致。

### 8.3 全文件静态计数不能当动态计数

对 7043 行文本按 opcode 模式核对得到：

| opcode | 全文件静态出现数 | 单 wave/body 动态数 |
|---|---:|---:|
| `v_wmma_scale_f32_32x16x128_f4` | 512 | 64 |
| `ds_load_b128` | 768 | 64 |
| `ds_load_b32` | 192 | 16 |
| `ds_load_2addr_b32` | 0 | 0 |
| `tensor_load_to_lds` | 56 | 1 |
| `s_wait_dscnt` | 50 | 6 |

512 条 WMMA 的静态结构是：

```text
2 套 wave-parity 主体 * 4 phase * 64 WMMA = 512
```

一次动态 body 只选择一套主体中的一个 phase。DS/TDM 静态数还包含
初始 task、persistent 后继 task 的 setup 副本，绝不能整文件相加后当作
单次执行量。ISA L6941–L7042 还是 shader padding 的 `s_code_end`，也不在
动态路径。

### 8.4 与 F8 256×256_4×4 版本的关键差异

| 维度 | 当前 F4 | 现有 F8×F4 文档 |
|---|---|---|
| WMMA | `32×16×128 F4` | `16×16×128 F8F6F4` |
| 硬件 Matrix A/B src | 16 / 8 VGPR | 8 / 16 VGPR |
| Matrix A/B host 角色 | B / A | B(FP4) / A(FP8) |
| K-body | 256 | 128 |
| 本 shape body 数 | 28 | 56 |
| wave/body WMMA | 64（32 fragment×2 K 子步） | 64（64 fragment×1 K 子步） |
| wave/body `b128` | 64 | 48 |
| wave/body `b32` | 16 | 8 |
| DScnt wait 数 | 6 | 5 |
| wait 上界 | `8,20,8,20,8,10/4` | `4/8,10/18,4/8,18/10,0` |
| data slot | 32 KiB / K256 | A/B 尺寸不对称 / K128 |
| physical WG launch | `(16,16,1)` | `(64,4,1)` |
| hotloop cluster sync | 每 4 body ring-wrap | 无每-body wrap；主要在启动/边界 |

相同的只有高层 tile/cluster 拓扑：

- `256×256` WG tile；
- 4 wave 的 2×2 象限；
- 4×4 cluster；
- A/SA row multicast、B/SB column multicast；
- 每 cluster、每“各自定义的一个 body”有 64 条 shader TDM issue。

最后一项不能用于等 K 流量比较：F4 body 是 K256，F8 body 是 K128。
按相同 K256 比较，F8 会经历两个 body，而当前 F4 只经历一个。

<a id="limits-evidence"></a>
## 9. 分析边界与证据文件

### 9.1 可以确定的事实

- Python/C++/CSV 可确定测试入口、intype/apre、symbol、CO、ABI 和 launch。
- ISA 可确定 wave 分支、persistent stride、K 步长、实际回边、ring 地址、
  DS/WMMA/TDM 指令形式及静态/动态路径计数。
- preshuffle 源码、wave aggregate bytes、WMMA src/scale/accumulator 数据流
  三者互相独立核验 tile 形状。

### 9.2 不能由当前文本确定的内容

- 当前 `.s` 只有 `.text` 反汇编，搜索不到 `.amdhsa` metadata。
  因此本文不猜 `next_free_vgpr`、声明 VGPR 数、occupancy 或
  `group_segment_fixed_size`。
- 反汇编确实通过 `s_set_vgpr_msb` 访问到物理 `v995`，LDS ring 地址也
  延伸到 `0x50000`；这些是“被访问编号/地址范围”，不是资源 metadata。
- DScnt/TENSORcnt 的逐 cycle 精确值、wait latency、TDM/DS 实际重叠度
  不能由静态 ISA 得出。所有 wait 数均只写 `≤` 上界。
- multicast 的匹配、timeout、cache line 和 GL2/HBM transaction 需要运行时
  观测；本文只给协议支持的 GL1 合并上界，不外推实际性能。

### 9.3 主要证据文件

- `op_tests/test_f4gemm.py`
- `op_tests/bench_init.py`
- `aiter/ops/gemm_op_a4w4.py`
- `aiter/ops/shuffle.py`
- `csrc/py_itfs_cu/asm_f4gemm.cu`
- `hsa/gfx1250/f4gemm/f4gemm.csv`
- `hsa/gfx1250/f4gemm/f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.co`
- `my_code/fmha/dump_asm/hsa/gfx1250/f4gemm/f4gemm_bf16_mxfp4_ABpreShuffle_256x256_4x4_ps.s`
- `C:\Users\yanguahe\Documents\hardware_file\MI450\amd-instinct-cdna5-instruction-set-architecture.txt`
- `C:\Users\yanguahe\Documents\code\llm-wiki\mi400_hw_wiki\raw\papers\mi400_hd_txt\architecture\subsystem\SH\MI400_Shader_Programming#65.txt`
- `C:\Users\yanguahe\Documents\code\llm-wiki\mi400_hw_wiki\raw\papers\mi400_hd_txt\subsystem\CMM\GLX\Design\MI400_Multicast_Feature#9.txt`
