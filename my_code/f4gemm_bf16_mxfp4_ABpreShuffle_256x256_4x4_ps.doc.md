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
5. [End-to-end software pipeline](#steady-pipeline)
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

该符号的 MXFP4 kernarg 是 **80 B = 20 dword**，不是整个 C++
`KernelArgs` 的 88 B。字段声明/offset 在 `asm_f4gemm.cu` L34–L57，赋值在
L168–L188；MXFP4 的 `arg_size=sizeof(KernelArgs)-8=80` 在 L195–L200。
dw18/19 占用结构体中名为 `GlobalScaleA/B` 的 8 B，但 MXFP4 路径在
L317–L329 用 `memcpy` 写入两个 `uint32_t` persistent-grid 参数；这里的
bit pattern 是整数，不是 float global scale。

CO 本身给出了相同且更底层的 ABI 约束：metadata 的
`.kernarg_segment_size=80`；kernel descriptor 的
`COMPUTE_PGM_RSRC2=0x000013ac` 解码为 `USER_SGPR_COUNT=22`，其
0-based dw14（word 15）`0x00140408` 解码为
`ENABLE_SGPR_KERNARG_SEGMENT_PTR=1`、`KERNARG_PRELOAD_LENGTH=20`、
`KERNARG_PRELOAD_OFFSET=0` 和 wave32。因而 **`s[0:1]` 是系统提供的隐式
kernarg-segment pointer**，不对应下表字段；20 个 kernarg dword 从 byte
offset 0 开始，由 CP 依次 preload 到 **`s2..s21`**。ISA 入口 L10–L11
直接把 `s[2:3]` 复制为 D 指针，没有任何显式 `s_load`，与该解码一致。
ISA L42–L57 又直接把 `s19/s18/s17` 用作 K/N/M；L257–L272 逐一配对
`s[4:5]/s13`（A）、`s[6:7]/s14`（B）、`s[8:9]/s15`（SA）和
`s[10:11]/s16`（SB），排除了仅凭 C++ 字段顺序猜 SGPR 映射。

| 字段 | preload SGPR | kernarg dword | byte offset | 当前 shape 的具体值 | 语义/单位；pointer 所指 tensor |
|---|---|---:|---:|---|---|
| `ptr_D` | `s[2:3]` | 0..1 | 0..7 (`0x00..0x07`) | runtime 64-bit pointer（地址不固定） | `torch.bfloat16` / BF16 output；logical=physical shape `[M,N]=[18432,2048]`，contiguous row-major；2 B/elem，row stride `4096 B`；storage `18432*2048*2 = 75,497,472 B (0x04800000)`。 |
| `ptr_A` | `s[4:5]` | 2..3 | 8..15 (`0x08..0x0f`) | runtime 64-bit pointer（地址不固定） | 当前测试为 `torch.uint8` on-wire packed MXFP4（每 byte 两个 E2M1）；logical shape `[M,K]=[18432,7168]`，2-D physical byte tensor shape `[M,K/2]=[18432,3584]`；16×16 packed-byte preshuffle 的实际存储次序为 `[M/16,(K/2)/16,16,16]=[1152,224,16,16]`（`row_tile,k_byte_tile,row_in_tile,byte_in_tile`），再 flatten 为上述 2-D shape；storage `18432*3584 = 66,060,288 B (0x03f00000)`。 |
| `ptr_B` | `s[6:7]` | 4..5 | 16..23 (`0x10..0x17`) | runtime 64-bit pointer（地址不固定） | 当前测试为 `torch.uint8` on-wire packed MXFP4；logical shape `[N,K]=[2048,7168]`，2-D physical byte tensor shape `[N,K/2]=[2048,3584]`；同一 16×16 packed-byte preshuffle 存储次序 `[N/16,(K/2)/16,16,16]=[128,224,16,16]`，再 flatten 为 2-D；storage `2048*3584 = 7,340,032 B (0x00700000)`。 |
| `ptr_ScaleA` | `s[8:9]` | 6..7 | 24..31 (`0x18..0x1f`) | runtime 64-bit pointer（地址不固定） | `torch.uint8` on-wire E8M0，1 B/scale、每个 scale 覆盖 K32；logical shape `[M,K/32]=[18432,224]`，wrapper-visible physical 2-D shape 仍为 `[18432,224]`；32×4 preshuffle 存储次序 `[M/32,(K/32)/4,32,4]=[576,56,32,4]`（`row_tile,k_scale_tile,row_in_tile,scale_in_tile`），再 flatten 为上述 2-D shape；storage `18432*224 = 4,128,768 B (0x003f0000)`。 |
| `ptr_ScaleB` | `s[10:11]` | 8..9 | 32..39 (`0x20..0x27`) | runtime 64-bit pointer（地址不固定） | `torch.uint8` on-wire E8M0，1 B/scale、每个 scale 覆盖 K32；logical shape `[N,K/32]=[2048,224]`，wrapper-visible physical 2-D shape 仍为 `[2048,224]`；32×4 preshuffle 存储次序 `[N/32,(K/32)/4,32,4]=[64,56,32,4]`，再 flatten 为上述 2-D shape；storage `2048*224 = 458,752 B (0x00070000)`。 |
| `strideD0` | `s12` | 10 | 40..43 (`0x28..0x2b`) | `4096 B (0x1000)` | D 的真实 contiguous row byte stride：`N*sizeof(bf16)=2048*2=4096`。 |
| `strideA0` | `s13` | 11 | 44..47 (`0x2c..0x2f`) | `3584 B (0x0e00)` | `K/2=7168/2`，packed A 的原始单-row byte extent；也是 flatten 后 2-D tensor 的表面 stride0，但不是 preshuffle 16×16 tile stride。 |
| `strideB0` | `s14` | 12 | 48..51 (`0x30..0x33`) | `3584 B (0x0e00)` | `K/2=7168/2`，packed B 的原始单-row byte extent；不是 preshuffle tile stride。 |
| `ScaleA_stride0` | `s15` | 13 | 52..55 (`0x34..0x37`) | `224 B (0x00e0)` | `K/32=7168/32`，未 shuffle scale 的单-row byte extent；不是 32×4 tile stride。 |
| `ScaleB_stride0` | `s16` | 14 | 56..59 (`0x38..0x3b`) | `224 B (0x00e0)` | `K/32=7168/32`，未 shuffle scale 的单-row byte extent；不是 32×4 tile stride。 |
| `M` | `s17` | 15 | 60..63 (`0x3c..0x3f`) | `18432 (0x4800)` | 逻辑 A/D row 数、输出 M dimension length；单位是 rows，不是 bytes。 |
| `N` | `s18` | 16 | 64..67 (`0x40..0x43`) | `2048 (0x0800)` | 逻辑 B row 数和 D column 数、输出 N dimension length；单位是 rows/columns，不是 bytes。 |
| `K` | `s19` | 17 | 68..71 (`0x44..0x47`) | `7168 (0x1c00)` | unpacked logical FP4 reduction length；单位是 logical FP4 values/row，不是 packed bytes（后者为 3584）也不是 scale 数（后者为 224/row）。 |
| `log2_grid_x` | `s20` | 18 | 72..75 (`0x48..0x4b`) | `2 (0x00000002)` | `log2` **physical persistent cluster-grid X**；`gridX=4`，不是 logical WG-grid X=8，也不是 logical cluster-grid X=2。 |
| `log2_grid_y` | `s21` | 19 | 76..79 (`0x4c..0x4f`) | `2 (0x00000002)` | `log2` **physical persistent cluster-grid Y**；`gridY=4`，不是 logical WG-grid Y=72，也不是 logical cluster-grid Y=18。 |

表中 pointer 的“logical”语义来自 reference 使用的 shuffle 前 buffers：
`test_f4gemm.py` L62–L80 生成/解释
`A[M,K/2]`、`B[N,K/2]`、`SA[M,K/32]`、`SB[N,K/32]`；L81–L89
才把 preshuffled buffers 交给候选。`gemm_a4w4` 的 gfx1250 wrapper
只把 A view 成 `[M,K/2]` 并原样下传 A/B/scale（
`gemm_op_a4w4.py` L144–L166、L297–L319），所以 CO 收到的 pointer
确实指向表中的 physical/preshuffled storage，而 reference 仍按 logical
矩阵解释。`shuffle_weight_f4` 的
`view(rows/16,16,(K/2)/16,16) -> permute(0,2,1,3)` 见
`shuffle.py` L318–L335；`shuffle_scale_f4` 的
`view(rows/32,32,(K/32)/4,4) -> permute(0,2,1,3)` 见 L292–L315。
两个函数最后都 flatten 回原 2-D tensor shape；因此 4-D shape 描述的是
线性存储次序，不是 wrapper 看见的 `Tensor.shape`。

五个 stride 不是从 `Tensor.stride()` 动态读取；C++ 在
`asm_f4gemm.cu` L157–L173 只由 `Ndim`、`Kdim` 和 dtype byte 数构造它们。
ISA 又给出 preshuffle 后的实际解释：

- D 未 shuffle；ISA L273–L286 用 `s12` 乘 M-row offset，并把 N offset
  乘 2，所以 `s12=4096 B` 就是真实相邻 D row stride。
- A/B 的 `s13/s14=3584 B` 是原始 packed-row extent。ISA 以
  `tile_index*256*stride` 定位 256-row tile（L257–L272），并分别在
  L350–L370、L1052–L1072 计算 `stride<<4`。所以 16-row preshuffle
  super-row stride 是 `16*3584 = 57,344 B (0x0e000)`；单个
  16×16 packed-byte tile 是 `256 B (0x0100)`。kernarg 本身两者都不是。
- SA/SB 同理：`s15/s16=224 B` 是原始 scale-row extent，ISA 在
  L1759–L1779、L2461–L2481 计算 `stride<<5`，得到 32-row super-row
  stride `32*224 = 7,168 B (0x1c00)`；单个 32×4 scale tile 是
  `128 B (0x0080)`。

dw18/19 的算法也不涉及对 72 取近似 log。C++ L252–L315 对当前配置计算：

```text
cluster_size = 4*4 = 16 WG
persistent clusters = PERSISTENT_TG/cluster_size = 256/16 = 16
gridY = PERSISTENT_GY = 4
gridX = persistent_clusters/gridY = 4
log2_grid_x = exact_log2(4) = 2
log2_grid_y = exact_log2(4) = 2
physical WG launch = (gridX*cluster_x, gridY*cluster_y) = (16,16)
```

代码先显式要求 `clusters/gridX/gridY` 都是 2 的幂（L289–L304），再用
右移循环计数（L312–L315），所以这里是 **exact integer log2**，不是
floor/ceil log、bit-width，也不是 fast-div magic。更准确的字段名应是
`log2_persistent_cluster_grid_x/y`。ISA L39–L41 用 `s20` 把物理 cluster
坐标 flatten；L204–L207 及 L6872–L6877 用
`1 << (s20+s21) = 16` 作为下一 persistent logical-cluster task 的步长。

相反，M 方向的 72 是 `M/256=72` 个 **logical WG rows**。ISA L45–L57
利用 `s17/s18` 和 TTMP 中的 4×4 cluster dimensions，以
`(dim+1024-1)>>10` 分别得到 logical cluster grid
`(ceil(N/1024),ceil(M/1024))=(2,18)`，再得到 36 个 logical cluster
tasks；当前 selector 已保证整除，所以这里的 ceil 恰好等于精确除法。
因此 72、18 和 36 都不写入 dw18/19。

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

本节交叉使用硬件 WMMA 契约、物理 src VGPR、scale row、§4.4–§4.5 的
DS producer 映射和独立 C/D fragment。

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
| `[32,47]` | `v552:559` | `v84 / row0` |
| `[48,63]` | `v568:575` | `v84 / row1` |
| `[64,79]` | `v264:271` | `v86 / row0` |
| `[80,95]` | `v280:287` | `v86 / row1` |
| `[96,111]` | `v296:303` | `v88 / row0` |
| `[112,127]` | `v312:319` | `v88 / row1` |

`v552:583`/`v84:85` 来自 A/SA current 前半的后 32 行，对应
`M[32:63]`；`v264:295`/`v86:87` 来自 current 后半的前 32 行，对应
`M[64:95]`，两组不能按 WMMA 的动态出现位置互换。

ISA 在 P0 中完整枚举 `4×8=32` 个 K0 组合。代表点包括：

| 输出 fragment | ISA | srcA(B) × srcB(A) |
|---|---|---|
| N0×M0 | L4846 / `0x7168` | `v776:791 × v520:527` |
| N0×M16 | L4849 / `0x7188` | `v776:791 × v536:543` |
| N32×M0 | L4875 / `0x728C` | `v808:823 × v520:527` |
| N64×M0 | L4885 / `0x7314` | `v8:23 × v520:527` |
| N0×M32 | L4860 / `0x71F8` | `v776:791 × v552:559` |
| N0×M64 | L4930 / `0x74D8` | `v776:791 × v264:271` |
| N96×M112 | L5025 / `0x785C` | `v40:55 × v312:319` |

指令交错安排 K0/K1，不能按连续行数机械分组。按 DS producer 与
src/scale 配对筛出 K0 后，四个 N32 组和八个 M16 组的每个组合都恰好
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

A/M: v528:535, v544:551, v560:567, v576:583,
     v272:279, v288:295, v304:311, v320:327
     with S4=v83,v85,v87,v89 and row0/row1
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
## 5. End-to-end software pipeline

本节先串起 `TDM producer → 四槽 LDS ring → DS consumer → WMMA →
drain/epilogue` 的端到端软件流水线，再由后面的 §5.1 和 §5.2 展开两份
真实 P0 指令流。全文使用 host operand 名称：`A/SA` 是 M 方向的 MXFP4
data/E8M0 scale，`B/SB` 是 N 方向的 data/scale；硬件 WMMA Matrix A/B
与 host B/A 的反向角色见 [第 2 节](#tile-wave)。

统一语义如下：

- `K0=K[k+0:k+127]` 与 `K1=K[k+128:k+255]` 共同组成一个 K256 body；
  `P0/P1/P2/P3` 是四槽 ring phase，不是 K0/K1。
- 下文一个 A 或 B bundle 都包含对应 data+scale：
  `16×ds_load_b128(data) + 4×ds_load_b32(scale)`，覆盖非 K 维
  `64 rows × K256`。每个 `current 后半` 或 `next 前半` 都是完整 K256
  半组，不是 K128。
- `next` 严格采用 **DS consumer** 语义：P0 中它是下一个 K256
  body，即 `slot1/body1`；同一时刻 **TDM producer** 回填的是刚消费的
  `slot0/body4`。因此 `A-next/B-next` 绝不能与 `slot0/body4` 混为一谈。
- `s_wait_dscnt n` 与 `s_wait_tensorcnt n` 返回时只分别保证
  `DScnt≤n` 与 `TENSORcnt≤n`；下列 wait 值和后文各阈值都是上界，
  不是精确 outstanding 数。

以下两份 Prologue 故意用相同格式重复列出四次 prime，并按 specialized
wave 组聚合：wave0/2 只 issue A/SA，wave1/3 只 issue B/SB；每组每个
slot 有 2 条 TDM，合并才是 4 TDM/WG。实际 wave0/1/2/3 分别只 issue
A/B/SA/SB 各自的一条 tensor 指令。下面 DS 数量按一个 compute wave
统计；若换算成整个 WG，DS 数量再乘 4。

**wave0/2：`B-current → A-current → A-next → B-next`**

```text
Prologue（wave0/2 聚合）
issue TDM A/SA slot0/body0              # 本 wave 组 2 TDM；wave0=A，wave2=SA
issue TDM A/SA slot1/body1              # 本 wave 组 2 TDM
issue TDM A/SA slot2/body2              # 本 wave 组 2 TDM
s_wait_tensorcnt 2                     # 每 wave 1 条 wait；保证最老 slot0 ready
WG barrier                             # 汇合 wave0/1/2/3 的 A/B/SA/SB slot0

SA-current 前半 # 4 ds_load_b32/wave
    ds_ld32_as0..as3 (0_0..0_3)  # current

SB-current 前半 # 4 ds_load_b32/wave
    ds_ld32_bs0..bs3 (0_0..0_3)  # current

A-current 前半 # 16 ds_load_b128/wave
    ds_ld128_a0..a15 (0_0..0_15)  # current

B-current 前半 # 16 ds_load_b128/wave
    ds_ld128_b0..b15 (0_0..0_15)  # current
                                        # 合计 40 DS/wave = 32 b128 + 8 b32
issue TDM A/SA slot3/body3              # 本 wave 组 2 TDM；wave0=A，wave2=SA
进入 P0/body0                           # current 前半 40 DS 已发出；由入口 wait1 判定依赖

Steady P0 / body0
entry: A/SA/B/SB-current 前半 40 DS 已发出  # 此处不声称 40 条全部 ready
s_wait_dscnt 8                         # wait1；最老 32 条 ready，B b8..b15 可未完成

SB-current 后半 # 4 ds_load_b32/wave
    wmma0 (0_0)  # K0
    ds_ld32_bs4..bs5 (1_0..1_1)  # current
    wmma1 (0_1)  # K0
    ds_ld32_bs6..bs7 (1_2..1_3)  # current

B-current 后半 # 16 ds_load_b128/wave
    wmma2..wmma3 (0_2..0_3; K1,K1)
    ds_ld128_b16..b21 (1_0..1_5)  # current
    wmma4..wmma5 (0_4..0_5; K0,K0)
    ds_ld128_b22..b27 (1_6..1_11)  # current
    wmma6..wmma7 (0_6..0_7; K1,K1)
    ds_ld128_b28..b31 (1_12..1_15)  # current
    s_wait_dscnt 20                  # wait2
    wmma8..wmma15 (1_0..1_7; K0,K0,K1,K1,K0,K0,K1,K1)

SA-current 后半 # 4 ds_load_b32/wave
    s_wait_dscnt 8                   # wait3
    wmma16 (2_0)  # K0
    ds_ld32_as4..as5 (1_0..1_1)  # current
    wmma17 (2_1)  # K0
    ds_ld32_as6..as7 (1_2..1_3)  # current

A-current 后半 # 16 ds_load_b128/wave
    wmma18..wmma19 (2_2..2_3; K1,K1)
    ds_ld128_a16..a21 (1_0..1_5)  # current
    wmma20..wmma21 (2_4..2_5; K0,K0)
    ds_ld128_a22..a27 (1_6..1_11)  # current
    wmma22..wmma23 (2_6..2_7; K1,K1)
    ds_ld128_a28..a31 (1_12..1_15)  # current
    s_wait_dscnt 20                  # wait4
    wmma24..wmma31 (3_0..3_7; K0,K0,K1,K1,K0,K0,K1,K1)

同步
    s_wait_dscnt 8                   # wait5
    s_wait_tensorcnt 2
    WG barrier signal
    wmma32..wmma35 (4_0..4_3; K0,K0,K1,K1)
    WG barrier wait

SA-next 前半 # 4 ds_load_b32/wave
    wmma36 (5_0)  # K0
    issue TDM A/SA slot0/body4        # 1 TDM/wave；本 wave 组共 2 条
    wmma37 (5_1)  # K0
    ds_ld32_as0..as1 (0_0..0_1)  # next
    wmma38 (5_2)  # K1
    ds_ld32_as2..as3 (0_2..0_3)  # next

A-next 前半 # 16 ds_load_b128/wave
    wmma39 (5_3)  # K1
    ds_ld128_a0..a5 (0_0..0_5)  # next
    s_wait_dscnt 10                  # wait6；A-next 在 a5/a6 之间切分
    wmma40..wmma41 (6_0..6_1; K0,K0)
    ds_ld128_a6..a11 (0_6..0_11)  # next
    wmma42..wmma43 (6_2..6_3; K1,K1)
    ds_ld128_a12..a15 (0_12..0_15)  # next
    wmma44..wmma47 (6_4..6_7; K0,K0,K1,K1)

SB-next 前半 # 4 ds_load_b32/wave
    wmma48 (7_0)  # K0
    ds_ld32_bs0..bs1 (0_0..0_1)  # next
    wmma49 (7_1)  # K0
    ds_ld32_bs2..bs3 (0_2..0_3)  # next

B-next 前半 # 16 ds_load_b128/wave
    wmma50..wmma51 (7_2..7_3; K1,K1)
    ds_ld128_b0..b5 (0_0..0_5)  # next
    wmma52..wmma53 (7_4..7_5; K0,K0)
    ds_ld128_b6..b11 (0_6..0_11)  # next
    wmma54..wmma55 (7_6..7_7; K1,K1)
    ds_ld128_b12..b15 (0_12..0_15)  # next

finish
    wmma56 (8_0)  # K0
    loop control SALU  # ISA L5008–L5011
    wmma57 (8_1)  # K0
    loop control SALU  # ISA L5013–L5016
    wmma58 (8_2)  # K1
    loop control SALU  # ISA L5018–L5021
    wmma59 (8_3)  # K1
    loop control compare  # ISA L5023
    wmma60..wmma63 (8_4..8_7; K0,K0,K1,K1)
    loop control branch  # ISA L5028
                                        # steady 合计 80 DS/wave；本 wave 组 2 TDM
phase advance: slot1/body1 的 next 前半成为 P1 current 前半

P0→P1 交界
entry: L5028 branch not taken           # 本 shape steady path 直接 fall through 到 L5031
entry: P0 已发出 slot1/body1 的 A/SA/B/SB-next 前半 DS
entry: P1 沿用对应 VGPR                 # 不重发 current 前半 DS

P1 / body1 current 后半                 # p1_wmmaN 为 P1-local 编号；ISA L5031–L5112
s_wait_dscnt 8                         # P1 wait1；只保证 DScnt≤8

SB-current 后半 # 4 ds_load_b32/wave
    p1_wmma0 (0_0)  # K0
    ds_ld32_bs4..bs5 (1_0..1_1)  # current；slot1/body1
    p1_wmma1 (0_1)  # K0
    ds_ld32_bs6..bs7 (1_2..1_3)  # current；slot1/body1

B-current 后半 # 16 ds_load_b128/wave
    p1_wmma2..p1_wmma3 (0_2..0_3; K1,K1)
    ds_ld128_b16..b21 (1_0..1_5)  # current；slot1/body1
    p1_wmma4..p1_wmma5 (0_4..0_5; K0,K0)
    ds_ld128_b22..b27 (1_6..1_11)  # current；slot1/body1
    p1_wmma6..p1_wmma7 (0_6..0_7; K1,K1)
    ds_ld128_b28..b31 (1_12..1_15)  # current；slot1/body1
    s_wait_dscnt 20                  # P1 wait2
    p1_wmma8..p1_wmma15 (1_0..1_7; K0,K0,K1,K1,K0,K0,K1,K1)

SA-current 后半 # 4 ds_load_b32/wave
    s_wait_dscnt 8                   # P1 wait3
    p1_wmma16 (2_0)  # K0
    ds_ld32_as4..as5 (1_0..1_1)  # current；slot1/body1
    p1_wmma17 (2_1)  # K0
    ds_ld32_as6..as7 (1_2..1_3)  # current；slot1/body1

A-current 后半 # 16 ds_load_b128/wave
    p1_wmma18..p1_wmma19 (2_2..2_3; K1,K1)
    ds_ld128_a16..a21 (1_0..1_5)  # current；slot1/body1
    p1_wmma20..p1_wmma21 (2_4..2_5; K0,K0)
    ds_ld128_a22..a27 (1_6..1_11)  # current；slot1/body1
    p1_wmma22..p1_wmma23 (2_6..2_7; K1,K1)
    ds_ld128_a28..a31 (1_12..1_15)  # current；slot1/body1
    s_wait_dscnt 20                  # P1 wait4
    p1_wmma24..p1_wmma31 (3_0..3_7; K0,K0,K1,K1,K0,K0,K1,K1)

P1 current 后半完成                    # 下一条 L5113 进入 P1 同步段
```

P0 steady 顺序为 `B-current 后半 → A-current 后半 → 同步/slot0→body4 → A-next 前半（由 wait10 切分）→ B-next 前半`。
本 shape 的 L5028 不跳转，直接落入 P1 L5031；P0 为 slot1/body1 发出的 next 前半在 P1 成为 current 前半，P1 不重发这些 DS。
P1 补齐 current 后半并完成同步后，L5124 将 slot1 回填为 body5。

**wave1/3：`A-current → B-current → B-next → A-next`**

```text
Prologue（wave1/3 聚合）
issue TDM B/SB slot0/body0              # 本 wave 组 2 TDM；wave1=B，wave3=SB
issue TDM B/SB slot1/body1              # 本 wave 组 2 TDM
issue TDM B/SB slot2/body2              # 本 wave 组 2 TDM
s_wait_tensorcnt 2                     # 仅保证 TENSORcnt≤2；同 wave TDM 有序，故最老 slot0 ready
s_barrier_signal -1                    # 本 wave 到达 workgroup barrier
s_barrier_wait 0xffff                  # 等四个 specialist 都 signal 后再读 A/B/SA/SB slot0

SB-current 前半 # 4 ds_load_b32/wave
    ds_ld32_bs0..bs3 (0_0..0_3)  # current

SA-current 前半 # 4 ds_load_b32/wave
    ds_ld32_as0..as3 (0_0..0_3)  # current

B-current 前半 # 16 ds_load_b128/wave
    ds_ld128_b0..b15 (0_0..0_15)  # current；ID 按动态 occurrence，不按 offset 重排

A-current 前半 # 16 ds_load_b128/wave
    ds_ld128_a0..a15 (0_0..0_15)  # current
                                        # 合计 40 DS/wave = 32 b128 + 8 b32
                                        # wave1 DS ISA L1682–L1723；wave3 L3092–L3133
issue TDM B/SB slot3/body3              # 本 wave 组 2 TDM；wave1 L1725，wave3 L3135
进入 P0/body0                           # current 前半 40 DS 已发出；由入口 wait1 判定依赖

Steady P0 / body0                       # ISA L5605–L5789
entry: A/SA/B/SB-current 前半 40 DS 已发出  # 此处不声称 40 条全部 ready
s_wait_dscnt 8                         # wait1；仅保证 DScnt≤8，即至少最老 32 条 ready
                                        # Prologue 顺序 SB4、SA4、B16、A16；最新 A a8..a15 可未完成

SA-current 后半 # 4 ds_load_b32/wave
    wmma0 (0_0)  # K0
    ds_ld32_as4..as5 (1_0..1_1)  # current
    wmma1 (0_1)  # K0
    ds_ld32_as6..as7 (1_2..1_3)  # current

A-current 后半 # 16 ds_load_b128/wave
    wmma2 (0_2)  # K1
    ds_ld128_a16..a21 (1_0..1_5)  # current
    wmma3..wmma4 (0_3..0_4; K1,K0)
    ds_ld128_a22..a27 (1_6..1_11)  # current
    wmma5..wmma6 (0_5..0_6; K0,K1)
    ds_ld128_a28..a31 (1_12..1_15)  # current
    wmma7 (0_7)  # K1
    s_wait_dscnt 20                  # wait2；仅保证 DScnt≤20
    wmma8..wmma15 (1_0..1_7; K0,K0,K1,K1,K0,K0,K1,K1)

SB-current 后半 # 4 ds_load_b32/wave
    s_wait_dscnt 8                   # wait3；仅保证 DScnt≤8
    wmma16 (2_0)  # K0
    ds_ld32_bs4..bs5 (1_0..1_1)  # current
    wmma17 (2_1)  # K0
    ds_ld32_bs6..bs7 (1_2..1_3)  # current

B-current 后半 # 16 ds_load_b128/wave
    wmma18 (2_2)  # K1
    ds_ld128_b16..b21 (1_0..1_5)  # current；ID 按 ISA 动态 occurrence
    wmma19..wmma20 (2_3..2_4; K1,K0)
    ds_ld128_b22..b27 (1_6..1_11)  # current
    wmma21..wmma22 (2_5..2_6; K0,K1)
    ds_ld128_b28..b31 (1_12..1_15)  # current
    wmma23 (2_7)  # K1
    s_wait_dscnt 20                  # wait4；仅保证 DScnt≤20
    wmma24..wmma31 (3_0..3_7; K0,K0,K1,K1,K0,K0,K1,K1)

同步
    s_wait_dscnt 8                   # wait5；仅保证 DScnt≤8
    s_wait_tensorcnt 2               # 仅保证 TENSORcnt≤2，不是 tensor 排空
    s_barrier_signal -1              # signal 只表示本 wave 到达
    wmma32..wmma35 (4_0..4_3; K0,K0,K1,K1)
    s_barrier_wait 0xffff            # 等全部 workgroup waves signal

SB-next 前半 # 4 ds_load_b32/wave
    wmma36 (5_0)  # K0
    issue TDM B/SB slot0/body4        # 1 TDM/wave；本 wave 组共 2 条；ISA L5698
    wmma37 (5_1)  # K0
    ds_ld32_bs0..bs1 (0_0..0_1)  # next
    wmma38 (5_2)  # K1
    ds_ld32_bs2..bs3 (0_2..0_3)  # next
    wmma39 (5_3)  # K1

B-next 前半 # 16 ds_load_b128/wave
    s_wait_dscnt 4                   # wait6；仅保证 DScnt≤4；ISA L5710
    wmma40 (6_0)  # K0
    ds_ld128_b0..b5 (0_0..0_5)  # next；ID 按 ISA 动态 occurrence
    wmma41..wmma42 (6_1..6_2; K0,K1)
    ds_ld128_b6..b11 (0_6..0_11)  # next
    wmma43..wmma44 (6_3..6_4; K1,K0)
    ds_ld128_b12..b15 (0_12..0_15)  # next
    wmma45..wmma47 (6_5..6_7; K0,K1,K1)

SA-next 前半 # 4 ds_load_b32/wave
    wmma48 (7_0)  # K0
    ds_ld32_as0..as1 (0_0..0_1)  # next
    wmma49 (7_1)  # K0
    ds_ld32_as2..as3 (0_2..0_3)  # next

A-next 前半 # 16 ds_load_b128/wave
    wmma50 (7_2)  # K1
    ds_ld128_a0..a5 (0_0..0_5)  # next
    wmma51..wmma52 (7_3..7_4; K1,K0)
    ds_ld128_a6..a11 (0_6..0_11)  # next
    wmma53..wmma54 (7_5..7_6; K0,K1)
    ds_ld128_a12..a15 (0_12..0_15)  # next
    wmma55 (7_7)  # K1

finish
    s_add_co_u32 s24, s58, 0x500  # ISA L5768
    s_cmp_lt_u32 s24, s70  # ISA L5769
    s_cselect_b64 s[34:35], s[34:35], s[62:63]  # ISA L5770
    s_cselect_b32 s36, s36, s64  # ISA L5771
    wmma56 (8_0)  # K0；ISA L5772
    s_cselect_b32 s37, s37, s65  # ISA L5773
    s_cselect_b32 s38, s38, s66  # ISA L5774
    s_cselect_b32 s70, s70, s71  # ISA L5775
    s_cselect_b64 s[26:27], s[56:57], 0  # ISA L5776
    wmma57 (8_1)  # K0；ISA L5777
    s_add_nc_u64 s[34:35], s[34:35], s[26:27]  # ISA L5778
    s_cmp_lt_u32 s24, s71  # ISA L5779
    s_cselect_b32 s39, s39, 0  # ISA L5780
    s_addk_co_i32 s58, 0x100  # ISA L5781
    wmma58 (8_2)  # K1；ISA L5782
    s_cmp_lt_i32 s58, s59  # ISA L5783
    wmma59 (8_3)  # K1；ISA L5784
    wmma60..wmma63 (8_4..8_7; K0,K0,K1,K1)
    s_cbranch_scc0 1388  # ISA L5789；本 shape P0 不跳转
                                        # steady 合计 80 DS/wave；本 wave 组 2 TDM
phase advance: slot1/body1 的 next 前半成为 P1 current 前半

P0→P1 交界
entry: L5789 branch not taken           # 本 shape steady path 经 L5790–L5791 落入 P1 L5792
entry: P0 已发出 slot1/body1 的 A/SA/B/SB-next 前半 DS
entry: P0 next DS 动态次序为 SB4、B16、SA4、A16
entry: P1 沿用对应 VGPR                 # 不重发 current 前半 DS

P1 / body1 current 后半                 # p1_wmmaN 为 P1-local 编号；ISA L5792–L5873
s_wait_dscnt 8                         # P1 wait1；仅保证 DScnt≤8，即至少最老 32 条 ready
                                        # P0 next 顺序下，最新 A a8..a15 可未完成

SA-current 后半 # 4 ds_load_b32/wave
    p1_wmma0 (0_0)  # K0
    ds_ld32_as4..as5 (1_0..1_1)  # current；slot1/body1
    p1_wmma1 (0_1)  # K0
    ds_ld32_as6..as7 (1_2..1_3)  # current；slot1/body1

A-current 后半 # 16 ds_load_b128/wave
    p1_wmma2 (0_2)  # K1
    ds_ld128_a16..a21 (1_0..1_5)  # current；slot1/body1
    p1_wmma3..p1_wmma4 (0_3..0_4; K1,K0)
    ds_ld128_a22..a27 (1_6..1_11)  # current；slot1/body1
    p1_wmma5..p1_wmma6 (0_5..0_6; K0,K1)
    ds_ld128_a28..a31 (1_12..1_15)  # current；slot1/body1
    p1_wmma7 (0_7)  # K1
    s_wait_dscnt 20                  # P1 wait2；仅保证 DScnt≤20
    p1_wmma8..p1_wmma15 (1_0..1_7; K0,K0,K1,K1,K0,K0,K1,K1)

SB-current 后半 # 4 ds_load_b32/wave
    s_wait_dscnt 8                   # P1 wait3；仅保证 DScnt≤8
    p1_wmma16 (2_0)  # K0
    ds_ld32_bs4..bs5 (1_0..1_1)  # current；slot1/body1
    p1_wmma17 (2_1)  # K0
    ds_ld32_bs6..bs7 (1_2..1_3)  # current；slot1/body1

B-current 后半 # 16 ds_load_b128/wave
    p1_wmma18 (2_2)  # K1
    ds_ld128_b16..b21 (1_0..1_5)  # current；slot1/body1；动态 occurrence
    p1_wmma19..p1_wmma20 (2_3..2_4; K1,K0)
    ds_ld128_b22..b27 (1_6..1_11)  # current；slot1/body1
    p1_wmma21..p1_wmma22 (2_5..2_6; K0,K1)
    ds_ld128_b28..b31 (1_12..1_15)  # current；slot1/body1
    p1_wmma23 (2_7)  # K1
    s_wait_dscnt 20                  # P1 wait4；仅保证 DScnt≤20
    p1_wmma24..p1_wmma31 (3_0..3_7; K0,K0,K1,K1,K0,K0,K1,K1)

P1 current 后半完成                    # 下一条 L5874 进入 P1 同步段

slot1/body5 回填锚点
    s_wait_dscnt 8                   # ISA L5874
    s_wait_tensorcnt 2               # ISA L5875；仅保证 TENSORcnt≤2
    s_barrier_signal -1              # ISA L5876
    p1_wmma32..p1_wmma35 (4_0..4_3; K0,K0,K1,K1)  # ISA L5878–L5881
    s_mov_b32 s33, s96               # ISA L5882；选择 slot1 LDS base
    s_barrier_wait 0xffff            # ISA L5883
    p1_wmma36 (5_0)  # K0；ISA L5884
    issue TDM B/SB slot1/body5        # 1 TDM/wave；本 wave 组共 2 条；ISA L5885
```

wave1/3 TDM 只生产 B/SB；P0 steady 的真实顺序为 `A-current 后半 → B-current 后半 → WG barrier/slot0→body4 → B-next 前半 → A-next 前半`，六个 DS wait 上界依次为 `8→20→8→20→8→4`。
L5789 对本 shape 不跳转并落入 P1 L5792；slot1/body1 的 next 前半转为 P1 current 前半且不重发 DS，P1 同步后由 L5885 将 B/SB slot1 回填为 body5。
四槽 ring 由 P0/P1/P2/P3 依次消费 slot0/1/2/3 并回填 body+4；本 shape 以 K256 为 body，共 `28 bodies = 7 × 4 phases`。

### 5.1 wave0/2：B-current → A-current → A-next → B-next

以下伪流保持 P0 ISA L4844–L5028 的动态顺序：`wmmaID (g_n)` 中 `g` 是 0-based pipeline group ID，`n` 是该 group 内 0-based 动态 occurrence ID；DS 的 `(h_i)` 表示前/后半组及组内动态 occurrence，`# current/# next` 标明 body 归属；所有 ID 均按动态出现顺序，DS/TENSOR wait 阈值仅是计数上界；本 shape 的 L5028 不跳转并落入 P1。

#### Prologue：首次进入 P0 前

本 Prologue 只重建首次 P0 的 TDM/DS/WMMA readiness，并非逐条完整 ISA；纯地址计算及重复 SALU descriptor/pointer 更新折叠为 `[collapsed]`，其余 readiness 锚点保留；首次 `s58=0` 时 `slot0/1/2/3→body0/1/2/3`，P0 消费 body0 后回填 slot0/body4，且 Prologue 不计入 P0 `185 ISA / 64 WMMA / 80 DS` totals。

**wave0：A-data TDM specialist。** 它使用 A ring `0x00000/0x08000/0x12000/0x1A000`、`s56=0x800`，并在 TDM opcode 上带 `th:TH_LOAD_NT`。该路径先 prime slot0/1/2，再清零 accumulator。

```text
s_cmp_eq_u32 s22, 0  # ISA L321 / 0x1EE8; wave_id == 0
s_cbranch_scc1 6  # ISA L322 / 0x1EEC; taken to L329 / 0x1F08
s_mov_b32 s95, 0  # ISA L329 / 0x1F08
s_mov_b32 s96, 0x8000  # ISA L330 / 0x1F0C
s_mov_b32 s97, 0x12000  # ISA L331 / 0x1F14
s_mov_b32 s98, 0x1a000  # ISA L332 / 0x1F1C
[collapsed] build A-data TDM descriptor/pointers  # ISA L333–L387 / 0x1F24–0x2040
s_mov_b32 s56, 0x800  # ISA L388 / 0x2044; A-data descriptor base increment = 0x800
[collapsed] finish A-data TDM descriptor/pointers  # ISA L389–L406 / 0x204C–0x20A4
s_barrier_signal -1  # ISA L407 / 0x20AC; WG rendezvous before cluster barrier
s_barrier_wait 0xffff  # ISA L408 / 0x20B0
s_barrier_signal -3  # ISA L409 / 0x20B4; wave0 signals cluster barrier
s_barrier_wait 0xfffd  # ISA L410 / 0x20B8; cluster rendezvous
s_mov_b32 s33, 0  # ISA L411 / 0x20BC; A slot0 / body0
tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT  # ISA L412 / 0x20C0; issue A slot0 / body0
[collapsed] advance A descriptor to body1  # ISA L413–L423 / 0x20CC–0x20F8
s_barrier_signal -1  # ISA L424 / 0x20FC
s_barrier_wait 0xffff  # ISA L425 / 0x2100; all four WG specialists issued slot0
s_mov_b32 s33, 0x8000  # ISA L426 / 0x2104; A slot1 / body1
tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT  # ISA L427 / 0x210C; issue A slot1 / body1
[collapsed] advance A descriptor to body2  # ISA L428–L438 / 0x2118–0x2144
s_barrier_signal -1  # ISA L439 / 0x2148
s_barrier_wait 0xffff  # ISA L440 / 0x214C; all four WG specialists issued slot1
s_mov_b32 s33, 0x12000  # ISA L441 / 0x2150; A slot2 / body2
tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT  # ISA L442 / 0x2158; issue A slot2 / body2
[collapsed] advance A descriptor to body3  # ISA L443–L453 / 0x2164–0x2190
s_barrier_signal -1  # ISA L454 / 0x2194
s_barrier_wait 0xffff  # ISA L455 / 0x2198; all four WG specialists issued slot2
[collapsed] clear 32×16 accumulator VGPRs (512 v_mov + 3 s_set_vgpr_msb)  # ISA L456–L970 / 0x219C–0x29A4
s_wait_tensorcnt 0x2  # ISA L971 / 0x29A8; return guarantees TENSORcnt<=2
s_barrier_signal -1  # ISA L972 / 0x29AC; slot0 is complete for this specialist
s_barrier_wait 0xffff  # ISA L973 / 0x29B0; WG rendezvous makes A/B/SA/SB slot0 readable
```

**wave2：SA TDM specialist。** 它使用 SA ring `0x10000/0x10800/0x11000/0x11800`、`s56=0x100`，TDM opcode 不带 `th:TH_LOAD_NT`。该路径先清零 accumulator，再 prime slot0/1/2。

```text
s_cmp_eq_u32 s22, 2  # ISA L325 / 0x1EF8; wave_id == 2
s_cbranch_scc1 1579  # ISA L326 / 0x1EFC; taken to L1738 / 0x37AC
s_mov_b32 s95, 0x10000  # ISA L1738 / 0x37AC
s_mov_b32 s96, 0x10800  # ISA L1739 / 0x37B4
s_mov_b32 s97, 0x11000  # ISA L1740 / 0x37BC
s_mov_b32 s98, 0x11800  # ISA L1741 / 0x37C4
[collapsed] build SA TDM descriptor/pointers  # ISA L1742–L1796 / 0x37CC–0x38EC
s_mov_b32 s56, 0x100  # ISA L1797 / 0x38F0; SA descriptor base increment = 0x100
[collapsed] finish SA TDM descriptor/pointers  # ISA L1798–L1815 / 0x38F8–0x3950
s_barrier_signal -1  # ISA L1816 / 0x3958; WG rendezvous before cluster barrier
s_barrier_wait 0xffff  # ISA L1817 / 0x395C
s_barrier_wait 0xfffd  # ISA L1818 / 0x3960; wave2 waits for wave0 cluster signal
s_wait_tensorcnt 0x0  # ISA L1819 / 0x3964; return guarantees TENSORcnt=0
[collapsed] clear 32×16 accumulator VGPRs (512 v_mov + 3 s_set_vgpr_msb)  # ISA L1820–L2334 / 0x3968–0x4170
s_mov_b32 s33, 0x10000  # ISA L2335 / 0x4174; SA slot0 / body0
tensor_load_to_lds s[32:35], s[36:43]  # ISA L2336 / 0x417C; issue SA slot0 / body0; no TH_LOAD_NT suffix
[collapsed] advance SA descriptor to body1  # ISA L2337–L2347 / 0x4188–0x41B4
s_barrier_signal -1  # ISA L2348 / 0x41B8
s_barrier_wait 0xffff  # ISA L2349 / 0x41BC; all four WG specialists issued slot0
s_mov_b32 s33, 0x10800  # ISA L2350 / 0x41C0; SA slot1 / body1
tensor_load_to_lds s[32:35], s[36:43]  # ISA L2351 / 0x41C8; issue SA slot1 / body1
[collapsed] advance SA descriptor to body2  # ISA L2352–L2362 / 0x41D4–0x4200
s_barrier_signal -1  # ISA L2363 / 0x4204
s_barrier_wait 0xffff  # ISA L2364 / 0x4208; all four WG specialists issued slot1
s_mov_b32 s33, 0x11000  # ISA L2365 / 0x420C; SA slot2 / body2
tensor_load_to_lds s[32:35], s[36:43]  # ISA L2366 / 0x4214; issue SA slot2 / body2
[collapsed] advance SA descriptor to body3  # ISA L2367–L2377 / 0x4220–0x424C
s_barrier_signal -1  # ISA L2378 / 0x4250
s_barrier_wait 0xffff  # ISA L2379 / 0x4254; all four WG specialists issued slot2
s_wait_tensorcnt 0x2  # ISA L2380 / 0x4258; return guarantees TENSORcnt<=2
s_barrier_signal -1  # ISA L2381 / 0x425C; slot0 is complete for this specialist
s_barrier_wait 0xffff  # ISA L2382 / 0x4260; WG rendezvous makes A/B/SA/SB slot0 readable
```

两条路径没有在同一 PC 立即合并：初始阶段由 wave0 在 L409 signal cluster barrier，wave2 在 L1818 wait；每个 slot prime 后再通过对应 WG barrier 对齐四个 specialist。最终 wave0 L971–L973 与 wave2 L2380–L2382 的 `wait_tensorcnt 2 + WG barrier` 构成 DS-read rendezvous：每个 specialist 至少完成自己最老的 slot0 TDM，且四个 wave 都到达后，每个 wave 才能读取 slot0 的 A/B/SA/SB。TENSORcnt 只保证上界；同一 wave 的 TDM 按 ISA 规定有序完成。

**Converged common prologue：current 前半 group0 DS-read。** wave0 和 wave2 在各自代码副本中执行完全相同的 40 条 DS load；下列每行同时给出两份锚点。

```text
s_set_vgpr_msb 0xc000  # ISA L974 / 0x29B4; wave2 mirror: L2383 / 0x4264
ds_ld32_as0 (0_0)  # current; wave0 L975/0x29B8; wave2 L2384/0x4268; ds_load_b32 v82, v80
ds_ld32_as1 (0_1)  # current; wave0 L976/0x29C0; wave2 L2385/0x4270; ds_load_b32 v83, v80 offset:128
ds_ld32_as2 (0_2)  # current; wave0 L977/0x29C8; wave2 L2386/0x4278; ds_load_b32 v84, v80 offset:256
ds_ld32_as3 (0_3)  # current; wave0 L978/0x29D0; wave2 L2387/0x4280; ds_load_b32 v85, v80 offset:384
ds_ld32_bs0 (0_0)  # current; wave0 L979/0x29D8; wave2 L2388/0x4288; ds_load_b32 v92, v81
ds_ld32_bs1 (0_1)  # current; wave0 L980/0x29E0; wave2 L2389/0x4290; ds_load_b32 v93, v81 offset:128
ds_ld32_bs2 (0_2)  # current; wave0 L981/0x29E8; wave2 L2390/0x4298; ds_load_b32 v94, v81 offset:256
ds_ld32_bs3 (0_3)  # current; wave0 L982/0x29F0; wave2 L2391/0x42A0; ds_load_b32 v95, v81 offset:384
s_set_vgpr_msb 0x80  # ISA L983 / 0x29F8; wave2 mirror: L2392 / 0x42A8
ds_ld128_a0 (0_0)  # current; wave0 L984/0x29FC; wave2 L2393/0x42AC; ds_load_b128 v[8:11] /*v[520:523]*/, v72
ds_ld128_a1 (0_1)  # current; wave0 L985/0x2A04; wave2 L2394/0x42B4; ds_load_b128 v[12:15] /*v[524:527]*/, v72 offset:512
ds_ld128_a2 (0_2)  # current; wave0 L986/0x2A0C; wave2 L2395/0x42BC; ds_load_b128 v[16:19] /*v[528:531]*/, v72 offset:1024
ds_ld128_a3 (0_3)  # current; wave0 L987/0x2A14; wave2 L2396/0x42C4; ds_load_b128 v[20:23] /*v[532:535]*/, v72 offset:1536
ds_ld128_a4 (0_4)  # current; wave0 L988/0x2A1C; wave2 L2397/0x42CC; ds_load_b128 v[24:27] /*v[536:539]*/, v72 offset:2048
ds_ld128_a5 (0_5)  # current; wave0 L989/0x2A24; wave2 L2398/0x42D4; ds_load_b128 v[28:31] /*v[540:543]*/, v72 offset:2560
ds_ld128_a6 (0_6)  # current; wave0 L990/0x2A2C; wave2 L2399/0x42DC; ds_load_b128 v[32:35] /*v[544:547]*/, v72 offset:3072
ds_ld128_a7 (0_7)  # current; wave0 L991/0x2A34; wave2 L2400/0x42E4; ds_load_b128 v[36:39] /*v[548:551]*/, v72 offset:3584
ds_ld128_a8 (0_8)  # current; wave0 L992/0x2A3C; wave2 L2401/0x42EC; ds_load_b128 v[40:43] /*v[552:555]*/, v72 offset:4096
ds_ld128_a9 (0_9)  # current; wave0 L993/0x2A44; wave2 L2402/0x42F4; ds_load_b128 v[44:47] /*v[556:559]*/, v72 offset:4608
ds_ld128_a10 (0_10)  # current; wave0 L994/0x2A4C; wave2 L2403/0x42FC; ds_load_b128 v[48:51] /*v[560:563]*/, v72 offset:5120
ds_ld128_a11 (0_11)  # current; wave0 L995/0x2A54; wave2 L2404/0x4304; ds_load_b128 v[52:55] /*v[564:567]*/, v72 offset:5632
ds_ld128_a12 (0_12)  # current; wave0 L996/0x2A5C; wave2 L2405/0x430C; ds_load_b128 v[56:59] /*v[568:571]*/, v72 offset:6144
ds_ld128_a13 (0_13)  # current; wave0 L997/0x2A64; wave2 L2406/0x4314; ds_load_b128 v[60:63] /*v[572:575]*/, v72 offset:6656
ds_ld128_a14 (0_14)  # current; wave0 L998/0x2A6C; wave2 L2407/0x431C; ds_load_b128 v[64:67] /*v[576:579]*/, v72 offset:7168
ds_ld128_a15 (0_15)  # current; wave0 L999/0x2A74; wave2 L2408/0x4324; ds_load_b128 v[68:71] /*v[580:583]*/, v72 offset:7680
s_set_vgpr_msb 0x80c0  # ISA L1000 / 0x2A7C; wave2 mirror: L2409 / 0x432C
ds_ld128_b0 (0_0)  # current; wave0 L1001/0x2A80; wave2 L2410/0x4330; ds_load_b128 v[8:11] /*v[776:779]*/, v76
ds_ld128_b1 (0_1)  # current; wave0 L1002/0x2A88; wave2 L2411/0x4338; ds_load_b128 v[12:15] /*v[780:783]*/, v76 offset:512
ds_ld128_b2 (0_2)  # current; wave0 L1003/0x2A90; wave2 L2412/0x4340; ds_load_b128 v[16:19] /*v[784:787]*/, v76 offset:2048
ds_ld128_b3 (0_3)  # current; wave0 L1004/0x2A98; wave2 L2413/0x4348; ds_load_b128 v[20:23] /*v[788:791]*/, v76 offset:2560
ds_ld128_b4 (0_4)  # current; wave0 L1005/0x2AA0; wave2 L2414/0x4350; ds_load_b128 v[24:27] /*v[792:795]*/, v76 offset:1024
ds_ld128_b5 (0_5)  # current; wave0 L1006/0x2AA8; wave2 L2415/0x4358; ds_load_b128 v[28:31] /*v[796:799]*/, v76 offset:1536
ds_ld128_b6 (0_6)  # current; wave0 L1007/0x2AB0; wave2 L2416/0x4360; ds_load_b128 v[32:35] /*v[800:803]*/, v76 offset:3072
ds_ld128_b7 (0_7)  # current; wave0 L1008/0x2AB8; wave2 L2417/0x4368; ds_load_b128 v[36:39] /*v[804:807]*/, v76 offset:3584
ds_ld128_b8 (0_8)  # current; wave0 L1009/0x2AC0; wave2 L2418/0x4370; ds_load_b128 v[40:43] /*v[808:811]*/, v76 offset:4096
ds_ld128_b9 (0_9)  # current; wave0 L1010/0x2AC8; wave2 L2419/0x4378; ds_load_b128 v[44:47] /*v[812:815]*/, v76 offset:4608
ds_ld128_b10 (0_10)  # current; wave0 L1011/0x2AD0; wave2 L2420/0x4380; ds_load_b128 v[48:51] /*v[816:819]*/, v76 offset:6144
ds_ld128_b11 (0_11)  # current; wave0 L1012/0x2AD8; wave2 L2421/0x4388; ds_load_b128 v[52:55] /*v[820:823]*/, v76 offset:6656
ds_ld128_b12 (0_12)  # current; wave0 L1013/0x2AE0; wave2 L2422/0x4390; ds_load_b128 v[56:59] /*v[824:827]*/, v76 offset:5120
ds_ld128_b13 (0_13)  # current; wave0 L1014/0x2AE8; wave2 L2423/0x4398; ds_load_b128 v[60:63] /*v[828:831]*/, v76 offset:5632
ds_ld128_b14 (0_14)  # current; wave0 L1015/0x2AF0; wave2 L2424/0x43A0; ds_load_b128 v[64:67] /*v[832:835]*/, v76 offset:7168
ds_ld128_b15 (0_15)  # current; wave0 L1016/0x2AF8; wave2 L2425/0x43A8; ds_load_b128 v[68:71] /*v[836:839]*/, v76 offset:7680
```

`# prologue DS count: per wave=40 (32×b128 + 8×b32); WMMA=0`

DS 发射后，各 specialist 再向 slot3 发一条 TDM，并把 descriptor 推进到 body4；wave0/wave2 最终都 branch 到同一 L4842。

```text
# wave0 specialized tail
s_mov_b32 s33, 0x1a000  # ISA L1017 / 0x2B00; A slot3 / body3
tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT  # ISA L1018 / 0x2B08; issue A slot3 / body3
[collapsed] advance A descriptor to body4  # ISA L1019–L1029 / 0x2B14–0x2B40
s_branch 4484  # ISA L1030 / 0x2B44; branch target L4842 / 0x7158

# wave2 specialized tail
s_mov_b32 s33, 0x11800  # ISA L2426 / 0x43B0; SA slot3 / body3
tensor_load_to_lds s[32:35], s[36:43]  # ISA L2427 / 0x43B8; issue SA slot3 / body3; no TH_LOAD_NT suffix
[collapsed] advance SA descriptor to body4  # ISA L2428–L2438 / 0x43C4–0x43F0
s_branch 2904  # ISA L2439 / 0x43F4; branch target L4842 / 0x7158

# literal common PC for wave0 and wave2
s_nop 0  # ISA L4842 / 0x7158
s_set_vgpr_msb 0  # ISA L4843 / 0x715C
# boundary: next executed instruction is P0 L4844 s_wait_dscnt 0x8
```

进入 P0 L4844 `s_wait_dscnt 8` **之前**：accumulator 的 512 个 F32 VGPR 已清零，WMMA 仍为 0；每个 wave 已发出 `AS0..3 + BS0..3 + A0..15 + B0..15` 共 40 条 current-group0 DS，故只能写 `DScnt≤40`，执行 L4844 后才保证 `DScnt≤8`。wave0 与 wave2 各自已发 4 条 prologue TDM（每 slot 一条）；在 `wait_tensorcnt 2` 后又发 slot3，因此入口只能写 `TENSORcnt≤3`。slot0/body0 已被 rendezvous 保证可供 DS 读取，slot1/2/3 分别预取 body1/2/3；P0 仍需在步骤1补 B-current 后半 group1、在步骤3补 A-current 后半 group1。

`# prologue total for this section: wave0 TDM=4; wave2 TDM=4; DS per wave=40; WMMA=0; not included in P0 totals`

1. **等待与 B-current 后半**（ISA L4844–L4873 / `0x7160–0x7280`）

```text
s_wait_dscnt 0x8  # ISA L4844 / 0x7160
s_set_vgpr_msb 11  # ISA L4845 / 0x7164
wmma0 (0_0)  # K0; ISA L4846 / 0x7168
ds_ld32_bs4 (1_0)  # current; ISA L4847 / 0x7178; ds_load_b32 v96, v81 /*v849*/ offset:512
ds_ld32_bs5 (1_1)  # current; ISA L4848 / 0x7180; ds_load_b32 v97, v81 /*v849*/ offset:640
wmma1 (0_1)  # K0; ISA L4849 / 0x7188
ds_ld32_bs6 (1_2)  # current; ISA L4850 / 0x7198; ds_load_b32 v98, v81 /*v849*/ offset:768
ds_ld32_bs7 (1_3)  # current; ISA L4851 / 0x71A0; ds_load_b32 v99, v81 /*v849*/ offset:896
wmma2 (0_2)  # K1; ISA L4852 / 0x71A8
wmma3 (0_3)  # K1; ISA L4853 / 0x71B8
ds_ld128_b16 (1_0)  # current; ISA L4854 / 0x71C8; ds_load_b128 v[8:11], v76 /*v844*/ offset:8192
ds_ld128_b17 (1_1)  # current; ISA L4855 / 0x71D0; ds_load_b128 v[12:15], v76 /*v844*/ offset:8704
ds_ld128_b18 (1_2)  # current; ISA L4856 / 0x71D8; ds_load_b128 v[16:19], v76 /*v844*/ offset:10240
ds_ld128_b19 (1_3)  # current; ISA L4857 / 0x71E0; ds_load_b128 v[20:23], v76 /*v844*/ offset:10752
ds_ld128_b20 (1_4)  # current; ISA L4858 / 0x71E8; ds_load_b128 v[24:27], v76 /*v844*/ offset:9216
ds_ld128_b21 (1_5)  # current; ISA L4859 / 0x71F0; ds_load_b128 v[28:31], v76 /*v844*/ offset:9728
wmma4 (0_4)  # K0; ISA L4860 / 0x71F8
wmma5 (0_5)  # K0; ISA L4861 / 0x7208
ds_ld128_b22 (1_6)  # current; ISA L4862 / 0x7218; ds_load_b128 v[32:35], v76 /*v844*/ offset:11264
ds_ld128_b23 (1_7)  # current; ISA L4863 / 0x7220; ds_load_b128 v[36:39], v76 /*v844*/ offset:11776
ds_ld128_b24 (1_8)  # current; ISA L4864 / 0x7228; ds_load_b128 v[40:43], v76 /*v844*/ offset:12288
ds_ld128_b25 (1_9)  # current; ISA L4865 / 0x7230; ds_load_b128 v[44:47], v76 /*v844*/ offset:12800
ds_ld128_b26 (1_10)  # current; ISA L4866 / 0x7238; ds_load_b128 v[48:51], v76 /*v844*/ offset:14336
ds_ld128_b27 (1_11)  # current; ISA L4867 / 0x7240; ds_load_b128 v[52:55], v76 /*v844*/ offset:14848
wmma6 (0_6)  # K1; ISA L4868 / 0x7248
wmma7 (0_7)  # K1; ISA L4869 / 0x7258
ds_ld128_b28 (1_12)  # current; ISA L4870 / 0x7268; ds_load_b128 v[56:59], v76 /*v844*/ offset:13312
ds_ld128_b29 (1_13)  # current; ISA L4871 / 0x7270; ds_load_b128 v[60:63], v76 /*v844*/ offset:13824
ds_ld128_b30 (1_14)  # current; ISA L4872 / 0x7278; ds_load_b128 v[64:67], v76 /*v844*/ offset:15360
ds_ld128_b31 (1_15)  # current; ISA L4873 / 0x7280; ds_load_b128 v[68:71], v76 /*v844*/ offset:15872
```

`# count: source=30; WMMA=8; DS=20 (b128=16, b32=4); DS-wait=1`

2. **第二个 DS wait 与 8 条 WMMA**（ISA L4874–L4882 / `0x7288–0x72FC`）

```text
s_wait_dscnt 0x14  # ISA L4874 / 0x7288
wmma8 (1_0)  # K0; ISA L4875 / 0x728C
wmma9 (1_1)  # K0; ISA L4876 / 0x729C
wmma10 (1_2)  # K1; ISA L4877 / 0x72AC
wmma11 (1_3)  # K1; ISA L4878 / 0x72BC
wmma12 (1_4)  # K0; ISA L4879 / 0x72CC
wmma13 (1_5)  # K0; ISA L4880 / 0x72DC
wmma14 (1_6)  # K1; ISA L4881 / 0x72EC
wmma15 (1_7)  # K1; ISA L4882 / 0x72FC
```

`# count: source=9; WMMA=8; DS=0 (b128=0, b32=0); DS-wait=1`

3. **等待与 A-current 后半**（ISA L4883–L4916 / `0x730C–0x743C`）

```text
s_wait_dscnt 0x8  # ISA L4883 / 0x730C
s_set_vgpr_msb 0xb58  # ISA L4884 / 0x7310
wmma16 (2_0)  # K0; ISA L4885 / 0x7314
s_set_vgpr_msb 0x5818  # ISA L4886 / 0x7324
ds_ld32_as4 (1_0)  # current; ISA L4887 / 0x7328; ds_load_b32 v86, v80 offset:512
ds_ld32_as5 (1_1)  # current; ISA L4888 / 0x7330; ds_load_b32 v87, v80 offset:640
s_set_vgpr_msb 0x1858  # ISA L4889 / 0x7338
wmma17 (2_1)  # K0; ISA L4890 / 0x733C
s_set_vgpr_msb 0x5818  # ISA L4891 / 0x734C
ds_ld32_as6 (1_2)  # current; ISA L4892 / 0x7350; ds_load_b32 v88, v80 offset:768
ds_ld32_as7 (1_3)  # current; ISA L4893 / 0x7358; ds_load_b32 v89, v80 offset:896
s_set_vgpr_msb 0x1858  # ISA L4894 / 0x7360
wmma18 (2_2)  # K1; ISA L4895 / 0x7364
wmma19 (2_3)  # K1; ISA L4896 / 0x7374
ds_ld128_a16 (1_0)  # current; ISA L4897 / 0x7384; ds_load_b128 v[8:11] /*v[264:267]*/, v72 offset:8192
ds_ld128_a17 (1_1)  # current; ISA L4898 / 0x738C; ds_load_b128 v[12:15] /*v[268:271]*/, v72 offset:8704
ds_ld128_a18 (1_2)  # current; ISA L4899 / 0x7394; ds_load_b128 v[16:19] /*v[272:275]*/, v72 offset:9216
ds_ld128_a19 (1_3)  # current; ISA L4900 / 0x739C; ds_load_b128 v[20:23] /*v[276:279]*/, v72 offset:9728
ds_ld128_a20 (1_4)  # current; ISA L4901 / 0x73A4; ds_load_b128 v[24:27] /*v[280:283]*/, v72 offset:10240
ds_ld128_a21 (1_5)  # current; ISA L4902 / 0x73AC; ds_load_b128 v[28:31] /*v[284:287]*/, v72 offset:10752
wmma20 (2_4)  # K0; ISA L4903 / 0x73B4
wmma21 (2_5)  # K0; ISA L4904 / 0x73C4
ds_ld128_a22 (1_6)  # current; ISA L4905 / 0x73D4; ds_load_b128 v[32:35] /*v[288:291]*/, v72 offset:11264
ds_ld128_a23 (1_7)  # current; ISA L4906 / 0x73DC; ds_load_b128 v[36:39] /*v[292:295]*/, v72 offset:11776
ds_ld128_a24 (1_8)  # current; ISA L4907 / 0x73E4; ds_load_b128 v[40:43] /*v[296:299]*/, v72 offset:12288
ds_ld128_a25 (1_9)  # current; ISA L4908 / 0x73EC; ds_load_b128 v[44:47] /*v[300:303]*/, v72 offset:12800
ds_ld128_a26 (1_10)  # current; ISA L4909 / 0x73F4; ds_load_b128 v[48:51] /*v[304:307]*/, v72 offset:13312
ds_ld128_a27 (1_11)  # current; ISA L4910 / 0x73FC; ds_load_b128 v[52:55] /*v[308:311]*/, v72 offset:13824
wmma22 (2_6)  # K1; ISA L4911 / 0x7404
wmma23 (2_7)  # K1; ISA L4912 / 0x7414
ds_ld128_a28 (1_12)  # current; ISA L4913 / 0x7424; ds_load_b128 v[56:59] /*v[312:315]*/, v72 offset:14336
ds_ld128_a29 (1_13)  # current; ISA L4914 / 0x742C; ds_load_b128 v[60:63] /*v[316:319]*/, v72 offset:14848
ds_ld128_a30 (1_14)  # current; ISA L4915 / 0x7434; ds_load_b128 v[64:67] /*v[320:323]*/, v72 offset:15360
ds_ld128_a31 (1_15)  # current; ISA L4916 / 0x743C; ds_load_b128 v[68:71] /*v[324:327]*/, v72 offset:15872
```

`# count: source=34; WMMA=8; DS=20 (b128=16, b32=4); DS-wait=1`

4. **第四个 DS wait 与 8 条 WMMA**（ISA L4917–L4925 / `0x7444–0x74B8`）

```text
s_wait_dscnt 0x14  # ISA L4917 / 0x7444
wmma24 (3_0)  # K0; ISA L4918 / 0x7448
wmma25 (3_1)  # K0; ISA L4919 / 0x7458
wmma26 (3_2)  # K1; ISA L4920 / 0x7468
wmma27 (3_3)  # K1; ISA L4921 / 0x7478
wmma28 (3_4)  # K0; ISA L4922 / 0x7488
wmma29 (3_5)  # K0; ISA L4923 / 0x7498
wmma30 (3_6)  # K1; ISA L4924 / 0x74A8
wmma31 (3_7)  # K1; ISA L4925 / 0x74B8
```

`# count: source=9; WMMA=8; DS=0 (b128=0, b32=0); DS-wait=1`

5. **DS/TENSOR wait、WG signal/wait**（ISA L4926–L4935 / `0x74C8–0x751C`）

```text
s_wait_dscnt 0x8  # ISA L4926 / 0x74C8
s_wait_tensorcnt 0x2  # ISA L4927 / 0x74CC
s_barrier_signal -1  # ISA L4928 / 0x74D0
s_set_vgpr_msb 0x58a7  # ISA L4929 / 0x74D4
wmma32 (4_0)  # K0; ISA L4930 / 0x74D8
wmma33 (4_1)  # K0; ISA L4931 / 0x74E8
wmma34 (4_2)  # K1; ISA L4932 / 0x74F8
wmma35 (4_3)  # K1; ISA L4933 / 0x7508
s_mov_b32 s33, s95  # ISA L4934 / 0x7518
s_barrier_wait 0xffff  # ISA L4935 / 0x751C
```

`# count: source=10; WMMA=4; DS=0 (b128=0, b32=0); DS-wait=1; tensor-wait=1; WG-signal=1; WG-wait=1`

6. **4 条 WMMA、1 TDM 与 A-next 起始**（ISA L4936–L4954 / `0x7520–0x75C4`）

```text
wmma36 (5_0)  # K0; ISA L4936 / 0x7520
tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT  # ISA L4937 / 0x7530
wmma37 (5_1)  # K0; ISA L4938 / 0x753C
s_set_vgpr_msb 0xa727  # ISA L4939 / 0x754C
ds_ld32_as0 (0_0)  # next; ISA L4940 / 0x7550; ds_load_b32 v82, v80 /*v848*/ offset:2048
ds_ld32_as1 (0_1)  # next; ISA L4941 / 0x7558; ds_load_b32 v83, v80 /*v848*/ offset:2176
s_set_vgpr_msb 0x27a7  # ISA L4942 / 0x7560
wmma38 (5_2)  # K1; ISA L4943 / 0x7564
s_set_vgpr_msb 0xa727  # ISA L4944 / 0x7574
ds_ld32_as2 (0_2)  # next; ISA L4945 / 0x7578; ds_load_b32 v84, v80 /*v848*/ offset:2304
ds_ld32_as3 (0_3)  # next; ISA L4946 / 0x7580; ds_load_b32 v85, v80 /*v848*/ offset:2432
s_set_vgpr_msb 0x27a7  # ISA L4947 / 0x7588
wmma39 (5_3)  # K1; ISA L4948 / 0x758C
ds_ld128_a0 (0_0)  # next; ISA L4949 / 0x759C; ds_load_b128 v[8:11] /*v[520:523]*/, v73 /*v841*/
ds_ld128_a1 (0_1)  # next; ISA L4950 / 0x75A4; ds_load_b128 v[12:15] /*v[524:527]*/, v73 /*v841*/ offset:512
ds_ld128_a2 (0_2)  # next; ISA L4951 / 0x75AC; ds_load_b128 v[16:19] /*v[528:531]*/, v73 /*v841*/ offset:1024
ds_ld128_a3 (0_3)  # next; ISA L4952 / 0x75B4; ds_load_b128 v[20:23] /*v[532:535]*/, v73 /*v841*/ offset:1536
ds_ld128_a4 (0_4)  # next; ISA L4953 / 0x75BC; ds_load_b128 v[24:27] /*v[536:539]*/, v73 /*v841*/ offset:2048
ds_ld128_a5 (0_5)  # next; ISA L4954 / 0x75C4; ds_load_b128 v[28:31] /*v[540:543]*/, v73 /*v841*/ offset:2560
```

`# count: source=19; WMMA=4; DS=10 (b128=6, b32=4); TDM=1`

7. **末个 DS wait 与 A-next 完成**（ISA L4955–L4973 / `0x75CC–0x7690`）

```text
s_wait_dscnt 0xa  # ISA L4955 / 0x75CC
wmma40 (6_0)  # K0; ISA L4956 / 0x75D0
wmma41 (6_1)  # K0; ISA L4957 / 0x75E0
ds_ld128_a6 (0_6)  # next; ISA L4958 / 0x75F0; ds_load_b128 v[32:35] /*v[544:547]*/, v73 /*v841*/ offset:3072
ds_ld128_a7 (0_7)  # next; ISA L4959 / 0x75F8; ds_load_b128 v[36:39] /*v[548:551]*/, v73 /*v841*/ offset:3584
ds_ld128_a8 (0_8)  # next; ISA L4960 / 0x7600; ds_load_b128 v[40:43] /*v[552:555]*/, v73 /*v841*/ offset:4096
ds_ld128_a9 (0_9)  # next; ISA L4961 / 0x7608; ds_load_b128 v[44:47] /*v[556:559]*/, v73 /*v841*/ offset:4608
ds_ld128_a10 (0_10)  # next; ISA L4962 / 0x7610; ds_load_b128 v[48:51] /*v[560:563]*/, v73 /*v841*/ offset:5120
ds_ld128_a11 (0_11)  # next; ISA L4963 / 0x7618; ds_load_b128 v[52:55] /*v[564:567]*/, v73 /*v841*/ offset:5632
wmma42 (6_2)  # K1; ISA L4964 / 0x7620
wmma43 (6_3)  # K1; ISA L4965 / 0x7630
ds_ld128_a12 (0_12)  # next; ISA L4966 / 0x7640; ds_load_b128 v[56:59] /*v[568:571]*/, v73 /*v841*/ offset:6144
ds_ld128_a13 (0_13)  # next; ISA L4967 / 0x7648; ds_load_b128 v[60:63] /*v[572:575]*/, v73 /*v841*/ offset:6656
ds_ld128_a14 (0_14)  # next; ISA L4968 / 0x7650; ds_load_b128 v[64:67] /*v[576:579]*/, v73 /*v841*/ offset:7168
ds_ld128_a15 (0_15)  # next; ISA L4969 / 0x7658; ds_load_b128 v[68:71] /*v[580:583]*/, v73 /*v841*/ offset:7680
wmma44 (6_4)  # K0; ISA L4970 / 0x7660
wmma45 (6_5)  # K0; ISA L4971 / 0x7670
wmma46 (6_6)  # K1; ISA L4972 / 0x7680
wmma47 (6_7)  # K1; ISA L4973 / 0x7690
```

`# count: source=19; WMMA=8; DS=10 (b128=10, b32=0); DS-wait=1`

8. **B-next 前半与 8 条 WMMA**（ISA L4974–L5006 / `0x76A0–0x77CC`）

```text
s_set_vgpr_msb 0xa7f4  # ISA L4974 / 0x76A0
wmma48 (7_0)  # K0; ISA L4975 / 0x76A4
s_set_vgpr_msb 0xf434  # ISA L4976 / 0x76B4
ds_ld32_bs0 (0_0)  # next; ISA L4977 / 0x76B8; ds_load_b32 v92, v81 offset:2048
ds_ld32_bs1 (0_1)  # next; ISA L4978 / 0x76C0; ds_load_b32 v93, v81 offset:2176
s_set_vgpr_msb 0x34f4  # ISA L4979 / 0x76C8
wmma49 (7_1)  # K0; ISA L4980 / 0x76CC
s_set_vgpr_msb 0xf434  # ISA L4981 / 0x76DC
ds_ld32_bs2 (0_2)  # next; ISA L4982 / 0x76E0; ds_load_b32 v94, v81 offset:2304
ds_ld32_bs3 (0_3)  # next; ISA L4983 / 0x76E8; ds_load_b32 v95, v81 offset:2432
s_set_vgpr_msb 0x34f4  # ISA L4984 / 0x76F0
wmma50 (7_2)  # K1; ISA L4985 / 0x76F4
wmma51 (7_3)  # K1; ISA L4986 / 0x7704
ds_ld128_b0 (0_0)  # next; ISA L4987 / 0x7714; ds_load_b128 v[8:11] /*v[776:779]*/, v77
ds_ld128_b1 (0_1)  # next; ISA L4988 / 0x771C; ds_load_b128 v[12:15] /*v[780:783]*/, v77 offset:512
ds_ld128_b2 (0_2)  # next; ISA L4989 / 0x7724; ds_load_b128 v[16:19] /*v[784:787]*/, v77 offset:2048
ds_ld128_b3 (0_3)  # next; ISA L4990 / 0x772C; ds_load_b128 v[20:23] /*v[788:791]*/, v77 offset:2560
ds_ld128_b4 (0_4)  # next; ISA L4991 / 0x7734; ds_load_b128 v[24:27] /*v[792:795]*/, v77 offset:1024
ds_ld128_b5 (0_5)  # next; ISA L4992 / 0x773C; ds_load_b128 v[28:31] /*v[796:799]*/, v77 offset:1536
wmma52 (7_4)  # K0; ISA L4993 / 0x7744
wmma53 (7_5)  # K0; ISA L4994 / 0x7754
ds_ld128_b6 (0_6)  # next; ISA L4995 / 0x7764; ds_load_b128 v[32:35] /*v[800:803]*/, v77 offset:3072
ds_ld128_b7 (0_7)  # next; ISA L4996 / 0x776C; ds_load_b128 v[36:39] /*v[804:807]*/, v77 offset:3584
ds_ld128_b8 (0_8)  # next; ISA L4997 / 0x7774; ds_load_b128 v[40:43] /*v[808:811]*/, v77 offset:4096
ds_ld128_b9 (0_9)  # next; ISA L4998 / 0x777C; ds_load_b128 v[44:47] /*v[812:815]*/, v77 offset:4608
ds_ld128_b10 (0_10)  # next; ISA L4999 / 0x7784; ds_load_b128 v[48:51] /*v[816:819]*/, v77 offset:6144
ds_ld128_b11 (0_11)  # next; ISA L5000 / 0x778C; ds_load_b128 v[52:55] /*v[820:823]*/, v77 offset:6656
wmma54 (7_6)  # K1; ISA L5001 / 0x7794
wmma55 (7_7)  # K1; ISA L5002 / 0x77A4
ds_ld128_b12 (0_12)  # next; ISA L5003 / 0x77B4; ds_load_b128 v[56:59] /*v[824:827]*/, v77 offset:5120
ds_ld128_b13 (0_13)  # next; ISA L5004 / 0x77BC; ds_load_b128 v[60:63] /*v[828:831]*/, v77 offset:5632
ds_ld128_b14 (0_14)  # next; ISA L5005 / 0x77C4; ds_load_b128 v[64:67] /*v[832:835]*/, v77 offset:7168
ds_ld128_b15 (0_15)  # next; ISA L5006 / 0x77CC; ds_load_b128 v[68:71] /*v[836:839]*/, v77 offset:7680
```

`# count: source=33; WMMA=8; DS=20 (b128=16, b32=4)`

9. **最后 8 条 WMMA 与 loop control**（ISA L5007–L5028 / `0x77D4–0x788C`）

```text
wmma56 (8_0)  # K0; ISA L5007 / 0x77D4
s_add_co_u32 s24, s58, 0x500  # ISA L5008 / 0x77E4
s_cmp_lt_u32 s24, s70  # ISA L5009 / 0x77EC; sets SCC=(s24 < s70)
s_cselect_b64 s[34:35], s[34:35], s[62:63]  # ISA L5010 / 0x77F0; select predicate from L5009: s24 < s70
s_cselect_b32 s36, s36, s64  # ISA L5011 / 0x77F4; select predicate from L5009: s24 < s70
wmma57 (8_1)  # K0; ISA L5012 / 0x77F8
s_cselect_b32 s37, s37, s65  # ISA L5013 / 0x7808; select predicate from L5009: s24 < s70
s_cselect_b32 s38, s38, s66  # ISA L5014 / 0x780C; select predicate from L5009: s24 < s70
s_cselect_b32 s70, s70, s71  # ISA L5015 / 0x7810; select predicate from L5009: s24 < s70
s_cselect_b64 s[26:27], s[56:57], 0  # ISA L5016 / 0x7814; select predicate from L5009: s24 < s70
wmma58 (8_2)  # K1; ISA L5017 / 0x7818
s_add_nc_u64 s[34:35], s[34:35], s[26:27]  # ISA L5018 / 0x7828
s_cmp_lt_u32 s24, s71  # ISA L5019 / 0x782C; sets SCC=(s24 < s71)
s_cselect_b32 s39, s39, 0  # ISA L5020 / 0x7830; select predicate from L5019: s24 < s71
s_addk_co_i32 s58, 0x100  # ISA L5021 / 0x7834
wmma59 (8_3)  # K1; ISA L5022 / 0x7838
s_cmp_lt_i32 s58, s59  # ISA L5023 / 0x7848; sets SCC=(s58 < s59)
wmma60 (8_4)  # K0; ISA L5024 / 0x784C
wmma61 (8_5)  # K0; ISA L5025 / 0x785C
wmma62 (8_6)  # K1; ISA L5026 / 0x786C
wmma63 (8_7)  # K1; ISA L5027 / 0x787C
s_cbranch_scc0 1391  # ISA L5028 / 0x788C; branch predicate from L5023: s58 >= s59; K=7168 steady P0: not taken, fall through to P1; tail target excluded
```

`# count: source=22; WMMA=8; DS=0 (b128=0, b32=0)`

wave0/2 单条动态 P0 路径汇总：

- `# total: source ISA=185; WMMA=64 (8+8+8+8+4+4+8+8+8); K0=32, K1=32; opcode=32×16×128`
- `# C/D pair check: 32 fragments, each K0=1 + K1=1`；例如 `wmma0`（L4846 / `0x7168`, K0）与 `wmma2`（L4852 / `0x71A8`, K1）累加同一 `v[100:115]` fragment。
- `# total: physical DS=64 b128 + 16 b32 = 80`
- `# current 后半: A/AS IDs=16..31/4..7; B/BS IDs=16..31/4..7`
- `# next 前半: A/AS IDs=0..15/0..3; B/BS IDs=0..15/0..3`
- `# total: bundle=4 × (16 b128 + 4 b32)`
- `# total: TDM=1; s_wait_dscnt=6; tensor-wait=1 (s_wait_tensorcnt 2)`
- `# total: WG barrier=1 signal + 1 wait; K-offset=+256`

#### 5.1.1 wave0/2 P0 的 host tile 覆盖

A data、SA、B data、SB 各自按动态 occurrence 独立编号；坐标表只把 occurrence 放回逻辑 M/N/K tile，不能按坐标或 offset 反向重编号。
group0/1 仅切非 K 维：group0=`A M[0:63] / B N[0:63]`，group1=`A M[64:127] / B N[64:127]`；每组覆盖完整 K256，`0_i`/`1_i` 不是 K0/K1，data/scale ID 分别按 `0..15/16..31` 与 `0..3/4..7` 延续。
首 body 的 group0 来自 prologue（wave0 L975–L1016，wave2 L2384–L2425），group1 来自 P0 L4847–L4916；后续 body 由上一 P3 next L5501–L5570 经 L5592–L5594 滚回下一 P0。

**A data：每格一条 `ds_load_b128 = M16×K64`**

| Host M ↓ / K → | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| M0 [0:15] | `ds_ld128_a0` (`0_0`) | `ds_ld128_a1` (`0_1`) | `ds_ld128_a2` (`0_2`) | `ds_ld128_a3` (`0_3`) |
| M1 [16:31] | `ds_ld128_a4` (`0_4`) | `ds_ld128_a5` (`0_5`) | `ds_ld128_a6` (`0_6`) | `ds_ld128_a7` (`0_7`) |
| M2 [32:47] | `ds_ld128_a8` (`0_8`) | `ds_ld128_a9` (`0_9`) | `ds_ld128_a10` (`0_10`) | `ds_ld128_a11` (`0_11`) |
| M3 [48:63] | `ds_ld128_a12` (`0_12`) | `ds_ld128_a13` (`0_13`) | `ds_ld128_a14` (`0_14`) | `ds_ld128_a15` (`0_15`) |
| M4 [64:79] | `ds_ld128_a16` (`1_0`) | `ds_ld128_a17` (`1_1`) | `ds_ld128_a18` (`1_2`) | `ds_ld128_a19` (`1_3`) |
| M5 [80:95] | `ds_ld128_a20` (`1_4`) | `ds_ld128_a21` (`1_5`) | `ds_ld128_a22` (`1_6`) | `ds_ld128_a23` (`1_7`) |
| M6 [96:111] | `ds_ld128_a24` (`1_8`) | `ds_ld128_a25` (`1_9`) | `ds_ld128_a26` (`1_10`) | `ds_ld128_a27` (`1_11`) |
| M7 [112:127] | `ds_ld128_a28` (`1_12`) | `ds_ld128_a29` (`1_13`) | `ds_ld128_a30` (`1_14`) | `ds_ld128_a31` (`1_15`) |

**A scale：每格一条 `ds_load_b32 = M32×K128 scales`**

| Host M ↓ / K → | K0 [0:127] | K1 [128:255] |
|---|---|---|
| M0 [0:31] | `ds_ld32_as0` (`0_0`) | `ds_ld32_as1` (`0_1`) |
| M1 [32:63] | `ds_ld32_as2` (`0_2`) | `ds_ld32_as3` (`0_3`) |
| M2 [64:95] | `ds_ld32_as4` (`1_0`) | `ds_ld32_as5` (`1_1`) |
| M3 [96:127] | `ds_ld32_as6` (`1_2`) | `ds_ld32_as7` (`1_3`) |

**B data：每格一条 `ds_load_b128 = N16×K64`**

| Host N ↓ / K → | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| N0 [0:15] | `ds_ld128_b0` (`0_0`) | `ds_ld128_b1` (`0_1`) | `ds_ld128_b4` (`0_4`) | `ds_ld128_b5` (`0_5`) |
| N1 [16:31] | `ds_ld128_b2` (`0_2`) | `ds_ld128_b3` (`0_3`) | `ds_ld128_b6` (`0_6`) | `ds_ld128_b7` (`0_7`) |
| N2 [32:47] | `ds_ld128_b8` (`0_8`) | `ds_ld128_b9` (`0_9`) | `ds_ld128_b12` (`0_12`) | `ds_ld128_b13` (`0_13`) |
| N3 [48:63] | `ds_ld128_b10` (`0_10`) | `ds_ld128_b11` (`0_11`) | `ds_ld128_b14` (`0_14`) | `ds_ld128_b15` (`0_15`) |
| N4 [64:79] | `ds_ld128_b16` (`1_0`) | `ds_ld128_b17` (`1_1`) | `ds_ld128_b20` (`1_4`) | `ds_ld128_b21` (`1_5`) |
| N5 [80:95] | `ds_ld128_b18` (`1_2`) | `ds_ld128_b19` (`1_3`) | `ds_ld128_b22` (`1_6`) | `ds_ld128_b23` (`1_7`) |
| N6 [96:111] | `ds_ld128_b24` (`1_8`) | `ds_ld128_b25` (`1_9`) | `ds_ld128_b28` (`1_12`) | `ds_ld128_b29` (`1_13`) |
| N7 [112:127] | `ds_ld128_b26` (`1_10`) | `ds_ld128_b27` (`1_11`) | `ds_ld128_b30` (`1_14`) | `ds_ld128_b31` (`1_15`) |

**B scale：每格一条 `ds_load_b32 = N32×K128 scales`**

| Host N ↓ / K → | K0 [0:127] | K1 [128:255] |
|---|---|---|
| N0 [0:31] | `ds_ld32_bs0` (`0_0`) | `ds_ld32_bs1` (`0_1`) |
| N1 [32:63] | `ds_ld32_bs2` (`0_2`) | `ds_ld32_bs3` (`0_3`) |
| N2 [64:95] | `ds_ld32_bs4` (`1_0`) | `ds_ld32_bs5` (`1_1`) |
| N3 [96:127] | `ds_ld32_bs6` (`1_2`) | `ds_ld32_bs7` (`1_3`) |

A data 前/后半为 `v520:583`/`v264:327`，SA 为 `v82:85`/`v86:89`，B data 为 `v776:839`/`v8:71`，SB 为 `v92:95`/`v96:99`；B 坐标须结合 physical dst、WMMA src 与动态 offset `0,512,2048,2560,1024,1536,3072,3584,4096,4608,6144,6656,5120,5632,7168,7680` 判定，ID 按 occurrence 而非 offset/坐标排序，故会出现 `b0,b1,b4,b5` 等非单调坐标排列。
下表坐标是 wave-local `M128×N128×K256`，不是全局下标；host A（M×K）对应硬件 Matrix B，host B（N×K）对应硬件 Matrix A，K0/K1 分列为两张 M×N 表。
`wmma0..wmma63` 严格按 P0 ISA L4844–L5028 动态顺序编号；`(g_n)` 的 `g` 是 0-based pipeline group ID，`n` 是该 group 内 0-based 动态 occurrence ID（group0/1/2/3/6/7/8 各 8 条，group4/5 各 4 条）；代表性校正映射为 `wmma4=L4860=M2N0K0`、`wmma32=L4930=M4N0K0`、`wmma48=L4975=M4N2K0`。

K0 表的每个格子计算 `M16×N32×K128`，K 范围为 `[0,127]`：

| K0 [0,127] / Host M ↓, Host N → | N0 [0,31] | N1 [32,63] | N2 [64,95] | N3 [96,127] |
|---|---:|---:|---:|---:|
| M0 [0,15] | `wmma0` (`0_0`) | `wmma8` (`1_0`) | `wmma16` (`2_0`) | `wmma24` (`3_0`) |
| M1 [16,31] | `wmma1` (`0_1`) | `wmma9` (`1_1`) | `wmma17` (`2_1`) | `wmma25` (`3_1`) |
| M2 [32,47] | `wmma4` (`0_4`) | `wmma12` (`1_4`) | `wmma20` (`2_4`) | `wmma28` (`3_4`) |
| M3 [48,63] | `wmma5` (`0_5`) | `wmma13` (`1_5`) | `wmma21` (`2_5`) | `wmma29` (`3_5`) |
| M4 [64,79] | `wmma32` (`4_0`) | `wmma33` (`4_1`) | `wmma48` (`7_0`) | `wmma49` (`7_1`) |
| M5 [80,95] | `wmma36` (`5_0`) | `wmma37` (`5_1`) | `wmma52` (`7_4`) | `wmma53` (`7_5`) |
| M6 [96,111] | `wmma40` (`6_0`) | `wmma41` (`6_1`) | `wmma56` (`8_0`) | `wmma57` (`8_1`) |
| M7 [112,127] | `wmma44` (`6_4`) | `wmma45` (`6_5`) | `wmma60` (`8_4`) | `wmma61` (`8_5`) |

K1 表的每个格子计算同一个 M×N output fragment 在 `[128,255]` 上的第二次
K128 累加：

| K1 [128,255] / Host M ↓, Host N → | N0 [0,31] | N1 [32,63] | N2 [64,95] | N3 [96,127] |
|---|---:|---:|---:|---:|
| M0 [0,15] | `wmma2` (`0_2`) | `wmma10` (`1_2`) | `wmma18` (`2_2`) | `wmma26` (`3_2`) |
| M1 [16,31] | `wmma3` (`0_3`) | `wmma11` (`1_3`) | `wmma19` (`2_3`) | `wmma27` (`3_3`) |
| M2 [32,47] | `wmma6` (`0_6`) | `wmma14` (`1_6`) | `wmma22` (`2_6`) | `wmma30` (`3_6`) |
| M3 [48,63] | `wmma7` (`0_7`) | `wmma15` (`1_7`) | `wmma23` (`2_7`) | `wmma31` (`3_7`) |
| M4 [64,79] | `wmma34` (`4_2`) | `wmma35` (`4_3`) | `wmma50` (`7_2`) | `wmma51` (`7_3`) |
| M5 [80,95] | `wmma38` (`5_2`) | `wmma39` (`5_3`) | `wmma54` (`7_6`) | `wmma55` (`7_7`) |
| M6 [96,111] | `wmma42` (`6_2`) | `wmma43` (`6_3`) | `wmma58` (`8_2`) | `wmma59` (`8_3`) |
| M7 [112,127] | `wmma46` (`6_6`) | `wmma47` (`6_7`) | `wmma62` (`8_6`) | `wmma63` (`8_7`) |

完整性检查：每张表都是 `8×4=32` 个格子；K0/K1 各覆盖全部
`M128×N128` output tile。两表合计64条，`wmma0..wmma63` 各出现一次；
脚本对 `(K,M,N)` 的 `2×8×4=64` 个组合也逐项检查为各一次，无重复、
无遗漏。

### 5.2 wave1/3：A-current → B-current → B-next → A-next

以下伪流保持 wave1/3 P0 ISA L5605–L5789 的 185 条动态文本顺序：`wmmaID (g_n)` 中 `g` 是 0-based pipeline group ID，`n` 是该 group 内 0-based 动态 occurrence ID；DS 的 `(h_i)` 表示前/后半组及组内动态 occurrence，`# current/# next` 标明 body 归属；A/SA/B/SB 各自编号，尤其 B data 不按 offset 或逻辑坐标重排。

硬件文档只保证 `S_WAIT_DSCNT n`/`S_WAIT_TENSORCNT n` 返回时对应计数 `≤n`（CDNA5 ISA L20007–L20043），同 wave LDS 与 Tensor 指令分别有序完成（CDNA5 ISA L3575–L3586、L10147–L10156；MI400 Guide L5792–L5805、L14097–L14104），WG barrier 则由 signal/arrive 与 wait 两段组成（CDNA5 ISA L3044–L3052）；MI400 Guide 的 XNACK 讨论 L18451 另称 LDS 可乱序完成且附近 L18454 标有 `FIXME`，与前述章节冲突，所以下文的 oldest-first 推断明确采用目标 gfx1250 的 CDNA5 ISA 顺序模型，不把阈值写成精确 outstanding，也不从静态 ISA 推断实测时延。

#### Prologue：首次进入 P0 前

本 Prologue 与 §5.1 一样只重建 readiness，不是 kernel 入口到 P0 的逐条完整 ISA；纯地址计算、重复 descriptor/pointer SALU 与 accumulator 清零可折叠，但 branch、TDM issue、wait、barrier 和 40 条 current 前半 DS 均保留真实锚点。首次 `s58=0` 时 `slot0/1/2/3→body0/1/2/3`，P0 随后消费 body0 并回填 slot0/body4；Prologue 不计入 P0 totals。

**wave1：B-data TDM specialist。** B ring 为 `0x30000/0x38000/0x40000/0x48000`，`s56=0x800`，TDM opcode 不带 `th:TH_LOAD_NT`；该路径先清零 accumulator，再 prime slot0/1/2。

```text
s_cmp_eq_u32 s22, 1  # ISA L323 / 0x1EF0; wave_id == 1
s_cbranch_scc1 788  # ISA L324 / 0x1EF4; taken to L1031 / 0x2B48
s_mov_b32 s95, 0x30000  # ISA L1031 / 0x2B48
s_mov_b32 s96, 0x38000  # ISA L1032 / 0x2B50
s_mov_b32 s97, 0x40000  # ISA L1033 / 0x2B58
s_mov_b32 s98, 0x48000  # ISA L1034 / 0x2B60
[collapsed] build B-data TDM descriptor/pointers  # ISA L1035–L1095 / 0x2B68–0x2CA4
s_mov_b32 s56, 0x800  # ISA L1096 / 0x2CA8; B-data descriptor base increment = 0x800
[collapsed] finish B-data TDM descriptor/pointers  # ISA L1097–L1114 / 0x2CB0–0x2D08
s_barrier_signal -1  # ISA L1115 / 0x2D10; WG rendezvous before cluster barrier
s_barrier_wait 0xffff  # ISA L1116 / 0x2D14
s_barrier_wait 0xfffd  # ISA L1117 / 0x2D18; wait for wave0 cluster signal
[collapsed] clear 32×16 accumulator VGPRs (512 v_mov + 3 s_set_vgpr_msb)  # ISA L1118–L1632 / 0x2D1C–0x3524
s_mov_b32 s33, 0x30000  # ISA L1633 / 0x3528; B slot0 / body0
tensor_load_to_lds s[32:35], s[36:43]  # ISA L1634 / 0x3530; issue B slot0 / body0
[collapsed] advance B descriptor to body1  # ISA L1635–L1645 / 0x353C–0x3568
s_barrier_signal -1  # ISA L1646 / 0x356C
s_barrier_wait 0xffff  # ISA L1647 / 0x3570; all four WG specialists issued slot0
s_mov_b32 s33, 0x38000  # ISA L1648 / 0x3574; B slot1 / body1
tensor_load_to_lds s[32:35], s[36:43]  # ISA L1649 / 0x357C; issue B slot1 / body1
[collapsed] advance B descriptor to body2  # ISA L1650–L1660 / 0x3588–0x35B4
s_barrier_signal -1  # ISA L1661 / 0x35B8
s_barrier_wait 0xffff  # ISA L1662 / 0x35BC; all four WG specialists issued slot1
s_mov_b32 s33, 0x40000  # ISA L1663 / 0x35C0; B slot2 / body2
tensor_load_to_lds s[32:35], s[36:43]  # ISA L1664 / 0x35C8; issue B slot2 / body2
[collapsed] advance B descriptor to body3  # ISA L1665–L1675 / 0x35D4–0x3600
s_barrier_signal -1  # ISA L1676 / 0x3604
s_barrier_wait 0xffff  # ISA L1677 / 0x3608; all four WG specialists issued slot2
s_wait_tensorcnt 0x2  # ISA L1678 / 0x360C; return guarantees TENSORcnt<=2
s_barrier_signal -1  # ISA L1679 / 0x3610; oldest B slot0 is complete
s_barrier_wait 0xffff  # ISA L1680 / 0x3614; WG rendezvous makes all slot0 operands readable
```

**wave3：SB TDM specialist。** SB ring 为 `0x22000/0x22800/0x23000/0x23800`，`s56=0x100`，TDM opcode 同样不带 `th:TH_LOAD_NT`；该路径在 cluster rendezvous 后先执行 `s_wait_tensorcnt 0`，再清零 accumulator 并 prime slot0/1/2。

```text
s_cmp_eq_u32 s22, 3  # ISA L327 / 0x1F00; wave_id == 3
s_cbranch_scc1 2364  # ISA L328 / 0x1F04; taken to L2440 / 0x43F8
s_mov_b32 s95, 0x22000  # ISA L2440 / 0x43F8
s_mov_b32 s96, 0x22800  # ISA L2441 / 0x4400
s_mov_b32 s97, 0x23000  # ISA L2442 / 0x4408
s_mov_b32 s98, 0x23800  # ISA L2443 / 0x4410
[collapsed] build SB TDM descriptor/pointers  # ISA L2444–L2504 / 0x4418–0x4554
s_mov_b32 s56, 0x100  # ISA L2505 / 0x4558; SB descriptor base increment = 0x100
[collapsed] finish SB TDM descriptor/pointers  # ISA L2506–L2523 / 0x4560–0x45B8
s_barrier_signal -1  # ISA L2524 / 0x45C0; WG rendezvous before cluster barrier
s_barrier_wait 0xffff  # ISA L2525 / 0x45C4
s_barrier_wait 0xfffd  # ISA L2526 / 0x45C8; wait for wave0 cluster signal
s_wait_tensorcnt 0x0  # ISA L2527 / 0x45CC
[collapsed] clear 32×16 accumulator VGPRs (512 v_mov + 3 s_set_vgpr_msb)  # ISA L2528–L3042 / 0x45D0–0x4DD8
s_mov_b32 s33, 0x22000  # ISA L3043 / 0x4DDC; SB slot0 / body0
tensor_load_to_lds s[32:35], s[36:43]  # ISA L3044 / 0x4DE4; issue SB slot0 / body0
[collapsed] advance SB descriptor to body1  # ISA L3045–L3055 / 0x4DF0–0x4E1C
s_barrier_signal -1  # ISA L3056 / 0x4E20
s_barrier_wait 0xffff  # ISA L3057 / 0x4E24; all four WG specialists issued slot0
s_mov_b32 s33, 0x22800  # ISA L3058 / 0x4E28; SB slot1 / body1
tensor_load_to_lds s[32:35], s[36:43]  # ISA L3059 / 0x4E30; issue SB slot1 / body1
[collapsed] advance SB descriptor to body2  # ISA L3060–L3070 / 0x4E3C–0x4E68
s_barrier_signal -1  # ISA L3071 / 0x4E6C
s_barrier_wait 0xffff  # ISA L3072 / 0x4E70; all four WG specialists issued slot1
s_mov_b32 s33, 0x23000  # ISA L3073 / 0x4E74; SB slot2 / body2
tensor_load_to_lds s[32:35], s[36:43]  # ISA L3074 / 0x4E7C; issue SB slot2 / body2
[collapsed] advance SB descriptor to body3  # ISA L3075–L3085 / 0x4E88–0x4EB4
s_barrier_signal -1  # ISA L3086 / 0x4EB8
s_barrier_wait 0xffff  # ISA L3087 / 0x4EBC; all four WG specialists issued slot2
s_wait_tensorcnt 0x2  # ISA L3088 / 0x4EC0; return guarantees TENSORcnt<=2
s_barrier_signal -1  # ISA L3089 / 0x4EC4; oldest SB slot0 is complete
s_barrier_wait 0xffff  # ISA L3090 / 0x4EC8; WG rendezvous makes all slot0 operands readable
```

上述 `wait_tensorcnt 2` 的硬件语义只是 `TENSORcnt≤2`；结合每条路径此前发出的三条且同 wave 有序完成的 TDM，ISA 数据流可推出最老 slot0 已完成。随后 WG barrier 汇合 A/B/SA/SB 四个 specialist，但不把不同 wave 的 TENSORcnt 合并成一个计数。

**Converged common prologue：current 前半 group0 DS-read。** wave1 与 wave3 执行相同的 40 条 DS，真实顺序为 `SB → SA → B data → A data`；每行给出两份锚点。

```text
s_set_vgpr_msb 0xc000  # wave1 ISA L1681 / 0x3618; wave3 L3091 / 0x4ECC
ds_ld32_bs0 (0_0)  # current; wave1 L1682/0x361C; wave3 L3092/0x4ED0; ds_load_b32 v92, v81
ds_ld32_bs1 (0_1)  # current; wave1 L1683/0x3624; wave3 L3093/0x4ED8; ds_load_b32 v93, v81 offset:128
ds_ld32_bs2 (0_2)  # current; wave1 L1684/0x362C; wave3 L3094/0x4EE0; ds_load_b32 v94, v81 offset:256
ds_ld32_bs3 (0_3)  # current; wave1 L1685/0x3634; wave3 L3095/0x4EE8; ds_load_b32 v95, v81 offset:384
ds_ld32_as0 (0_0)  # current; wave1 L1686/0x363C; wave3 L3096/0x4EF0; ds_load_b32 v82, v80
ds_ld32_as1 (0_1)  # current; wave1 L1687/0x3644; wave3 L3097/0x4EF8; ds_load_b32 v83, v80 offset:128
ds_ld32_as2 (0_2)  # current; wave1 L1688/0x364C; wave3 L3098/0x4F00; ds_load_b32 v84, v80 offset:256
ds_ld32_as3 (0_3)  # current; wave1 L1689/0x3654; wave3 L3099/0x4F08; ds_load_b32 v85, v80 offset:384
s_set_vgpr_msb 64  # wave1 ISA L1690 / 0x365C; wave3 L3100 / 0x4F10
ds_ld128_b0 (0_0)  # current; wave1 L1691/0x3660; wave3 L3101/0x4F14; ds_load_b128 v[8:11] /*v[264:267]*/, v76
ds_ld128_b1 (0_1)  # current; wave1 L1692/0x3668; wave3 L3102/0x4F1C; ds_load_b128 v[12:15] /*v[268:271]*/, v76 offset:512
ds_ld128_b2 (0_2)  # current; wave1 L1693/0x3670; wave3 L3103/0x4F24; ds_load_b128 v[16:19] /*v[272:275]*/, v76 offset:2048
ds_ld128_b3 (0_3)  # current; wave1 L1694/0x3678; wave3 L3104/0x4F2C; ds_load_b128 v[20:23] /*v[276:279]*/, v76 offset:2560
ds_ld128_b4 (0_4)  # current; wave1 L1695/0x3680; wave3 L3105/0x4F34; ds_load_b128 v[24:27] /*v[280:283]*/, v76 offset:1024
ds_ld128_b5 (0_5)  # current; wave1 L1696/0x3688; wave3 L3106/0x4F3C; ds_load_b128 v[28:31] /*v[284:287]*/, v76 offset:1536
ds_ld128_b6 (0_6)  # current; wave1 L1697/0x3690; wave3 L3107/0x4F44; ds_load_b128 v[32:35] /*v[288:291]*/, v76 offset:3072
ds_ld128_b7 (0_7)  # current; wave1 L1698/0x3698; wave3 L3108/0x4F4C; ds_load_b128 v[36:39] /*v[292:295]*/, v76 offset:3584
ds_ld128_b8 (0_8)  # current; wave1 L1699/0x36A0; wave3 L3109/0x4F54; ds_load_b128 v[40:43] /*v[296:299]*/, v76 offset:4096
ds_ld128_b9 (0_9)  # current; wave1 L1700/0x36A8; wave3 L3110/0x4F5C; ds_load_b128 v[44:47] /*v[300:303]*/, v76 offset:4608
ds_ld128_b10 (0_10)  # current; wave1 L1701/0x36B0; wave3 L3111/0x4F64; ds_load_b128 v[48:51] /*v[304:307]*/, v76 offset:6144
ds_ld128_b11 (0_11)  # current; wave1 L1702/0x36B8; wave3 L3112/0x4F6C; ds_load_b128 v[52:55] /*v[308:311]*/, v76 offset:6656
ds_ld128_b12 (0_12)  # current; wave1 L1703/0x36C0; wave3 L3113/0x4F74; ds_load_b128 v[56:59] /*v[312:315]*/, v76 offset:5120
ds_ld128_b13 (0_13)  # current; wave1 L1704/0x36C8; wave3 L3114/0x4F7C; ds_load_b128 v[60:63] /*v[316:319]*/, v76 offset:5632
ds_ld128_b14 (0_14)  # current; wave1 L1705/0x36D0; wave3 L3115/0x4F84; ds_load_b128 v[64:67] /*v[320:323]*/, v76 offset:7168
ds_ld128_b15 (0_15)  # current; wave1 L1706/0x36D8; wave3 L3116/0x4F8C; ds_load_b128 v[68:71] /*v[324:327]*/, v76 offset:7680
s_set_vgpr_msb 0x40c0  # wave1 ISA L1707 / 0x36E0; wave3 L3117 / 0x4F94
ds_ld128_a0 (0_0)  # current; wave1 L1708/0x36E4; wave3 L3118/0x4F98; ds_load_b128 v[8:11] /*v[776:779]*/, v72
ds_ld128_a1 (0_1)  # current; wave1 L1709/0x36EC; wave3 L3119/0x4FA0; ds_load_b128 v[12:15] /*v[780:783]*/, v72 offset:512
ds_ld128_a2 (0_2)  # current; wave1 L1710/0x36F4; wave3 L3120/0x4FA8; ds_load_b128 v[16:19] /*v[784:787]*/, v72 offset:1024
ds_ld128_a3 (0_3)  # current; wave1 L1711/0x36FC; wave3 L3121/0x4FB0; ds_load_b128 v[20:23] /*v[788:791]*/, v72 offset:1536
ds_ld128_a4 (0_4)  # current; wave1 L1712/0x3704; wave3 L3122/0x4FB8; ds_load_b128 v[24:27] /*v[792:795]*/, v72 offset:2048
ds_ld128_a5 (0_5)  # current; wave1 L1713/0x370C; wave3 L3123/0x4FC0; ds_load_b128 v[28:31] /*v[796:799]*/, v72 offset:2560
ds_ld128_a6 (0_6)  # current; wave1 L1714/0x3714; wave3 L3124/0x4FC8; ds_load_b128 v[32:35] /*v[800:803]*/, v72 offset:3072
ds_ld128_a7 (0_7)  # current; wave1 L1715/0x371C; wave3 L3125/0x4FD0; ds_load_b128 v[36:39] /*v[804:807]*/, v72 offset:3584
ds_ld128_a8 (0_8)  # current; wave1 L1716/0x3724; wave3 L3126/0x4FD8; ds_load_b128 v[40:43] /*v[808:811]*/, v72 offset:4096
ds_ld128_a9 (0_9)  # current; wave1 L1717/0x372C; wave3 L3127/0x4FE0; ds_load_b128 v[44:47] /*v[812:815]*/, v72 offset:4608
ds_ld128_a10 (0_10)  # current; wave1 L1718/0x3734; wave3 L3128/0x4FE8; ds_load_b128 v[48:51] /*v[816:819]*/, v72 offset:5120
ds_ld128_a11 (0_11)  # current; wave1 L1719/0x373C; wave3 L3129/0x4FF0; ds_load_b128 v[52:55] /*v[820:823]*/, v72 offset:5632
ds_ld128_a12 (0_12)  # current; wave1 L1720/0x3744; wave3 L3130/0x4FF8; ds_load_b128 v[56:59] /*v[824:827]*/, v72 offset:6144
ds_ld128_a13 (0_13)  # current; wave1 L1721/0x374C; wave3 L3131/0x5000; ds_load_b128 v[60:63] /*v[828:831]*/, v72 offset:6656
ds_ld128_a14 (0_14)  # current; wave1 L1722/0x3754; wave3 L3132/0x5008; ds_load_b128 v[64:67] /*v[832:835]*/, v72 offset:7168
ds_ld128_a15 (0_15)  # current; wave1 L1723/0x375C; wave3 L3133/0x5010; ds_load_b128 v[68:71] /*v[836:839]*/, v72 offset:7680
```

`# prologue DS count: per wave=40 (32×b128 + 8×b32); WMMA=0`

DS 发射后，wave1/wave3 分别向自己的 B/SB slot3 发一条 TDM，将 descriptor 推进到 body4，并汇合到 P0 L5603。

```text
# wave1 specialized tail
s_mov_b32 s33, 0x48000  # ISA L1724 / 0x3764; B slot3 / body3
tensor_load_to_lds s[32:35], s[36:43]  # ISA L1725 / 0x376C; issue B slot3 / body3
[collapsed] advance B descriptor to body4  # ISA L1726–L1736 / 0x3778–0x37A4
s_branch 5552  # ISA L1737 / 0x37A8; branch target L5603 / 0x8E6C

# wave3 specialized tail
s_mov_b32 s33, 0x23800  # ISA L3134 / 0x5018; SB slot3 / body3
tensor_load_to_lds s[32:35], s[36:43]  # ISA L3135 / 0x5020; issue SB slot3 / body3
[collapsed] advance SB descriptor to body4  # ISA L3136–L3146 / 0x502C–0x5058
s_branch 3971  # ISA L3147 / 0x505C; branch target L5603 / 0x8E6C

# literal common PC for wave1 and wave3
s_nop 0  # ISA L5603 / 0x8E6C
s_set_vgpr_msb 0  # ISA L5604 / 0x8E70
# boundary: next executed instruction is P0 L5605 s_wait_dscnt 0x8
```

进入 P0 L5605 前，每个 wave 已按 `BS0..3 + AS0..3 + B0..15 + A0..15` 发出 40 条 current-group0 DS，故只能写 `DScnt≤40`；L5605 后才保证 `DScnt≤8`，在上述 CDNA5 ISA 的 LDS 顺序模型下可推出至少最老 32 条 ready、最新 `A8..15` 仍可能未完成。wave1 与 wave3 各自发出 4 条 Prologue TDM（每 slot 一条），不是单 wave 4 类 operand；`wait_tensorcnt 2` 后又发 slot3，因此 P0 入口只能写 `TENSORcnt≤3`。slot0/body0 已经 rendezvous，slot1/2/3 预取 body1/2/3。

`# prologue total for this section: wave1 TDM=4; wave3 TDM=4; DS per wave=40; WMMA=0; not included in P0 totals`

1. **等待与 A-current 后半**（ISA L5605–L5634 / `0x8E74–0x8F8C`）

```text
s_wait_dscnt 0x8  # ISA L5605 / 0x8E74; return guarantees DScnt<=8
s_set_vgpr_msb 13  # ISA L5606 / 0x8E78
wmma0 (0_0)  # K0; ISA L5607 / 0x8E7C
ds_ld32_as4 (1_0)  # current; ISA L5608 / 0x8E8C; ds_load_b32 v86, v80 /*v336*/ offset:512
ds_ld32_as5 (1_1)  # current; ISA L5609 / 0x8E94; ds_load_b32 v87, v80 /*v336*/ offset:640
wmma1 (0_1)  # K0; ISA L5610 / 0x8E9C
ds_ld32_as6 (1_2)  # current; ISA L5611 / 0x8EAC; ds_load_b32 v88, v80 /*v336*/ offset:768
ds_ld32_as7 (1_3)  # current; ISA L5612 / 0x8EB4; ds_load_b32 v89, v80 /*v336*/ offset:896
wmma2 (0_2)  # K1; ISA L5613 / 0x8EBC
ds_ld128_a16 (1_0)  # current; ISA L5614 / 0x8ECC; ds_load_b128 v[8:11], v72 /*v328*/ offset:8192
ds_ld128_a17 (1_1)  # current; ISA L5615 / 0x8ED4; ds_load_b128 v[12:15], v72 /*v328*/ offset:8704
ds_ld128_a18 (1_2)  # current; ISA L5616 / 0x8EDC; ds_load_b128 v[16:19], v72 /*v328*/ offset:9216
ds_ld128_a19 (1_3)  # current; ISA L5617 / 0x8EE4; ds_load_b128 v[20:23], v72 /*v328*/ offset:9728
ds_ld128_a20 (1_4)  # current; ISA L5618 / 0x8EEC; ds_load_b128 v[24:27], v72 /*v328*/ offset:10240
ds_ld128_a21 (1_5)  # current; ISA L5619 / 0x8EF4; ds_load_b128 v[28:31], v72 /*v328*/ offset:10752
wmma3 (0_3)  # K1; ISA L5620 / 0x8EFC
wmma4 (0_4)  # K0; ISA L5621 / 0x8F0C
ds_ld128_a22 (1_6)  # current; ISA L5622 / 0x8F1C; ds_load_b128 v[32:35], v72 /*v328*/ offset:11264
ds_ld128_a23 (1_7)  # current; ISA L5623 / 0x8F24; ds_load_b128 v[36:39], v72 /*v328*/ offset:11776
ds_ld128_a24 (1_8)  # current; ISA L5624 / 0x8F2C; ds_load_b128 v[40:43], v72 /*v328*/ offset:12288
ds_ld128_a25 (1_9)  # current; ISA L5625 / 0x8F34; ds_load_b128 v[44:47], v72 /*v328*/ offset:12800
ds_ld128_a26 (1_10)  # current; ISA L5626 / 0x8F3C; ds_load_b128 v[48:51], v72 /*v328*/ offset:13312
ds_ld128_a27 (1_11)  # current; ISA L5627 / 0x8F44; ds_load_b128 v[52:55], v72 /*v328*/ offset:13824
wmma5 (0_5)  # K0; ISA L5628 / 0x8F4C
wmma6 (0_6)  # K1; ISA L5629 / 0x8F5C
ds_ld128_a28 (1_12)  # current; ISA L5630 / 0x8F6C; ds_load_b128 v[56:59], v72 /*v328*/ offset:14336
ds_ld128_a29 (1_13)  # current; ISA L5631 / 0x8F74; ds_load_b128 v[60:63], v72 /*v328*/ offset:14848
ds_ld128_a30 (1_14)  # current; ISA L5632 / 0x8F7C; ds_load_b128 v[64:67], v72 /*v328*/ offset:15360
ds_ld128_a31 (1_15)  # current; ISA L5633 / 0x8F84; ds_load_b128 v[68:71], v72 /*v328*/ offset:15872
wmma7 (0_7)  # K1; ISA L5634 / 0x8F8C
```

`# count: source=30; WMMA=8; DS=20 (b128=16, b32=4); DS-wait=1`

2. **第二个 DS wait 与 8 条 WMMA**（ISA L5635–L5643 / `0x8F9C–0x9010`）

```text
s_wait_dscnt 0x14  # ISA L5635 / 0x8F9C; return guarantees DScnt<=20
wmma8 (1_0)  # K0; ISA L5636 / 0x8FA0
wmma9 (1_1)  # K0; ISA L5637 / 0x8FB0
wmma10 (1_2)  # K1; ISA L5638 / 0x8FC0
wmma11 (1_3)  # K1; ISA L5639 / 0x8FD0
wmma12 (1_4)  # K0; ISA L5640 / 0x8FE0
wmma13 (1_5)  # K0; ISA L5641 / 0x8FF0
wmma14 (1_6)  # K1; ISA L5642 / 0x9000
wmma15 (1_7)  # K1; ISA L5643 / 0x9010
```

`# count: source=9; WMMA=8; DS=0 (b128=0, b32=0); DS-wait=1`

3. **等待与 B-current 后半**（ISA L5644–L5677 / `0x9020–0x9148`）

```text
s_wait_dscnt 0x8  # ISA L5644 / 0x9020; return guarantees DScnt<=8
s_set_vgpr_msb 0xda1  # ISA L5645 / 0x9024
wmma16 (2_0)  # K0; ISA L5646 / 0x9028
s_set_vgpr_msb 0xa121  # ISA L5647 / 0x9038
ds_ld32_bs4 (1_0)  # current; ISA L5648 / 0x903C; ds_load_b32 v96, v81 /*v337*/ offset:512
ds_ld32_bs5 (1_1)  # current; ISA L5649 / 0x9044; ds_load_b32 v97, v81 /*v337*/ offset:640
s_set_vgpr_msb 0x21a1  # ISA L5650 / 0x904C
wmma17 (2_1)  # K0; ISA L5651 / 0x9050
s_set_vgpr_msb 0xa121  # ISA L5652 / 0x9060
ds_ld32_bs6 (1_2)  # current; ISA L5653 / 0x9064; ds_load_b32 v98, v81 /*v337*/ offset:768
ds_ld32_bs7 (1_3)  # current; ISA L5654 / 0x906C; ds_load_b32 v99, v81 /*v337*/ offset:896
s_set_vgpr_msb 0x21a1  # ISA L5655 / 0x9074
wmma18 (2_2)  # K1; ISA L5656 / 0x9078
ds_ld128_b16 (1_0)  # current; ISA L5657 / 0x9088; ds_load_b128 v[8:11] /*v[520:523]*/, v76 /*v332*/ offset:8192
ds_ld128_b17 (1_1)  # current; ISA L5658 / 0x9090; ds_load_b128 v[12:15] /*v[524:527]*/, v76 /*v332*/ offset:8704
ds_ld128_b18 (1_2)  # current; ISA L5659 / 0x9098; ds_load_b128 v[16:19] /*v[528:531]*/, v76 /*v332*/ offset:10240
ds_ld128_b19 (1_3)  # current; ISA L5660 / 0x90A0; ds_load_b128 v[20:23] /*v[532:535]*/, v76 /*v332*/ offset:10752
ds_ld128_b20 (1_4)  # current; ISA L5661 / 0x90A8; ds_load_b128 v[24:27] /*v[536:539]*/, v76 /*v332*/ offset:9216
ds_ld128_b21 (1_5)  # current; ISA L5662 / 0x90B0; ds_load_b128 v[28:31] /*v[540:543]*/, v76 /*v332*/ offset:9728
wmma19 (2_3)  # K1; ISA L5663 / 0x90B8
wmma20 (2_4)  # K0; ISA L5664 / 0x90C8
ds_ld128_b22 (1_6)  # current; ISA L5665 / 0x90D8; ds_load_b128 v[32:35] /*v[544:547]*/, v76 /*v332*/ offset:11264
ds_ld128_b23 (1_7)  # current; ISA L5666 / 0x90E0; ds_load_b128 v[36:39] /*v[548:551]*/, v76 /*v332*/ offset:11776
ds_ld128_b24 (1_8)  # current; ISA L5667 / 0x90E8; ds_load_b128 v[40:43] /*v[552:555]*/, v76 /*v332*/ offset:12288
ds_ld128_b25 (1_9)  # current; ISA L5668 / 0x90F0; ds_load_b128 v[44:47] /*v[556:559]*/, v76 /*v332*/ offset:12800
ds_ld128_b26 (1_10)  # current; ISA L5669 / 0x90F8; ds_load_b128 v[48:51] /*v[560:563]*/, v76 /*v332*/ offset:14336
ds_ld128_b27 (1_11)  # current; ISA L5670 / 0x9100; ds_load_b128 v[52:55] /*v[564:567]*/, v76 /*v332*/ offset:14848
wmma21 (2_5)  # K0; ISA L5671 / 0x9108
wmma22 (2_6)  # K1; ISA L5672 / 0x9118
ds_ld128_b28 (1_12)  # current; ISA L5673 / 0x9128; ds_load_b128 v[56:59] /*v[568:571]*/, v76 /*v332*/ offset:13312
ds_ld128_b29 (1_13)  # current; ISA L5674 / 0x9130; ds_load_b128 v[60:63] /*v[572:575]*/, v76 /*v332*/ offset:13824
ds_ld128_b30 (1_14)  # current; ISA L5675 / 0x9138; ds_load_b128 v[64:67] /*v[576:579]*/, v76 /*v332*/ offset:15360
ds_ld128_b31 (1_15)  # current; ISA L5676 / 0x9140; ds_load_b128 v[68:71] /*v[580:583]*/, v76 /*v332*/ offset:15872
wmma23 (2_7)  # K1; ISA L5677 / 0x9148
```

`# count: source=34; WMMA=8; DS=20 (b128=16, b32=4); DS-wait=1`

4. **第四个 DS wait 与 8 条 WMMA**（ISA L5678–L5686 / `0x9158–0x91CC`）

```text
s_wait_dscnt 0x14  # ISA L5678 / 0x9158; return guarantees DScnt<=20
wmma24 (3_0)  # K0; ISA L5679 / 0x915C
wmma25 (3_1)  # K0; ISA L5680 / 0x916C
wmma26 (3_2)  # K1; ISA L5681 / 0x917C
wmma27 (3_3)  # K1; ISA L5682 / 0x918C
wmma28 (3_4)  # K0; ISA L5683 / 0x919C
wmma29 (3_5)  # K0; ISA L5684 / 0x91AC
wmma30 (3_6)  # K1; ISA L5685 / 0x91BC
wmma31 (3_7)  # K1; ISA L5686 / 0x91CC
```

`# count: source=9; WMMA=8; DS=0 (b128=0, b32=0); DS-wait=1`

5. **DS/TENSOR wait、WG signal/wait**（ISA L5687–L5696 / `0x91DC–0x9230`）

```text
s_wait_dscnt 0x8  # ISA L5687 / 0x91DC; return guarantees DScnt<=8
s_wait_tensorcnt 0x2  # ISA L5688 / 0x91E0; return guarantees TENSORcnt<=2
s_barrier_signal -1  # ISA L5689 / 0x91E4
s_set_vgpr_msb 0xa15e  # ISA L5690 / 0x91E8
wmma32 (4_0)  # K0; ISA L5691 / 0x91EC
wmma33 (4_1)  # K0; ISA L5692 / 0x91FC
wmma34 (4_2)  # K1; ISA L5693 / 0x920C
wmma35 (4_3)  # K1; ISA L5694 / 0x921C
s_mov_b32 s33, s95  # ISA L5695 / 0x922C; select slot0 destination for this specialist
s_barrier_wait 0xffff  # ISA L5696 / 0x9230
```

`# count: source=10; WMMA=4; DS=0 (b128=0, b32=0); DS-wait=1; tensor-wait=1; WG-signal=1; WG-wait=1`

6. **4 条 WMMA、1 TDM 与 B-next scale**（ISA L5697–L5709 / `0x9234–0x92A0`）

```text
wmma36 (5_0)  # K0; ISA L5697 / 0x9234
tensor_load_to_lds s[32:35], s[36:43]  # ISA L5698 / 0x9244; wave1=B slot0/body4; wave3=SB slot0/body4
wmma37 (5_1)  # K0; ISA L5699 / 0x9250
s_set_vgpr_msb 0x5e1e  # ISA L5700 / 0x9260
ds_ld32_bs0 (0_0)  # next; ISA L5701 / 0x9264; ds_load_b32 v92, v81 /*v593*/ offset:2048
ds_ld32_bs1 (0_1)  # next; ISA L5702 / 0x926C; ds_load_b32 v93, v81 /*v593*/ offset:2176
s_set_vgpr_msb 0x1e5e  # ISA L5703 / 0x9274
wmma38 (5_2)  # K1; ISA L5704 / 0x9278
s_set_vgpr_msb 0x5e1e  # ISA L5705 / 0x9288
ds_ld32_bs2 (0_2)  # next; ISA L5706 / 0x928C; ds_load_b32 v94, v81 /*v593*/ offset:2304
ds_ld32_bs3 (0_3)  # next; ISA L5707 / 0x9294; ds_load_b32 v95, v81 /*v593*/ offset:2432
s_set_vgpr_msb 0x1e5e  # ISA L5708 / 0x929C
wmma39 (5_3)  # K1; ISA L5709 / 0x92A0
```

`# count: source=13; WMMA=4; DS=4 (b128=0, b32=4); TDM=1`

L5698 是每条动态 wave 路径仅一条 TDM：wave1 的 descriptor 令其生产 B，wave3 的 descriptor 令其生产 SB；它不生产 A/SA，也不是单 wave 同时发 4 条 TDM。

7. **末个 DS wait 与 B-next data**（ISA L5710–L5734 / `0x92B0–0x93A4`）

```text
s_wait_dscnt 0x4  # ISA L5710 / 0x92B0; return guarantees DScnt<=4
wmma40 (6_0)  # K0; ISA L5711 / 0x92B4
ds_ld128_b0 (0_0)  # next; ISA L5712 / 0x92C4; ds_load_b128 v[8:11] /*v[264:267]*/, v77 /*v589*/
ds_ld128_b1 (0_1)  # next; ISA L5713 / 0x92CC; ds_load_b128 v[12:15] /*v[268:271]*/, v77 /*v589*/ offset:512
ds_ld128_b2 (0_2)  # next; ISA L5714 / 0x92D4; ds_load_b128 v[16:19] /*v[272:275]*/, v77 /*v589*/ offset:2048
ds_ld128_b3 (0_3)  # next; ISA L5715 / 0x92DC; ds_load_b128 v[20:23] /*v[276:279]*/, v77 /*v589*/ offset:2560
ds_ld128_b4 (0_4)  # next; ISA L5716 / 0x92E4; ds_load_b128 v[24:27] /*v[280:283]*/, v77 /*v589*/ offset:1024
ds_ld128_b5 (0_5)  # next; ISA L5717 / 0x92EC; ds_load_b128 v[28:31] /*v[284:287]*/, v77 /*v589*/ offset:1536
wmma41 (6_1)  # K0; ISA L5718 / 0x92F4
wmma42 (6_2)  # K1; ISA L5719 / 0x9304
ds_ld128_b6 (0_6)  # next; ISA L5720 / 0x9314; ds_load_b128 v[32:35] /*v[288:291]*/, v77 /*v589*/ offset:3072
ds_ld128_b7 (0_7)  # next; ISA L5721 / 0x931C; ds_load_b128 v[36:39] /*v[292:295]*/, v77 /*v589*/ offset:3584
ds_ld128_b8 (0_8)  # next; ISA L5722 / 0x9324; ds_load_b128 v[40:43] /*v[296:299]*/, v77 /*v589*/ offset:4096
ds_ld128_b9 (0_9)  # next; ISA L5723 / 0x932C; ds_load_b128 v[44:47] /*v[300:303]*/, v77 /*v589*/ offset:4608
ds_ld128_b10 (0_10)  # next; ISA L5724 / 0x9334; ds_load_b128 v[48:51] /*v[304:307]*/, v77 /*v589*/ offset:6144
ds_ld128_b11 (0_11)  # next; ISA L5725 / 0x933C; ds_load_b128 v[52:55] /*v[308:311]*/, v77 /*v589*/ offset:6656
wmma43 (6_3)  # K1; ISA L5726 / 0x9344
wmma44 (6_4)  # K0; ISA L5727 / 0x9354
ds_ld128_b12 (0_12)  # next; ISA L5728 / 0x9364; ds_load_b128 v[56:59] /*v[312:315]*/, v77 /*v589*/ offset:5120
ds_ld128_b13 (0_13)  # next; ISA L5729 / 0x936C; ds_load_b128 v[60:63] /*v[316:319]*/, v77 /*v589*/ offset:5632
ds_ld128_b14 (0_14)  # next; ISA L5730 / 0x9374; ds_load_b128 v[64:67] /*v[320:323]*/, v77 /*v589*/ offset:7168
ds_ld128_b15 (0_15)  # next; ISA L5731 / 0x937C; ds_load_b128 v[68:71] /*v[324:327]*/, v77 /*v589*/ offset:7680
wmma45 (6_5)  # K0; ISA L5732 / 0x9384
wmma46 (6_6)  # K1; ISA L5733 / 0x9394
wmma47 (6_7)  # K1; ISA L5734 / 0x93A4
```

`# count: source=25; WMMA=8; DS=16 (b128=16, b32=0); DS-wait=1`

步骤6–7 合成 `B-next/SB-next 前半 = 16 b128 + 4 b32`；L5710 的 4 仍只是上界，不能写成恰有 4 条未完成。

8. **A-next 前半与 8 条 WMMA**（ISA L5735–L5767 / `0x93B4–0x94D8`）

```text
s_set_vgpr_msb 0x5ef2  # ISA L5735 / 0x93B4
wmma48 (7_0)  # K0; ISA L5736 / 0x93B8
s_set_vgpr_msb 0xf232  # ISA L5737 / 0x93C8
ds_ld32_as0 (0_0)  # next; ISA L5738 / 0x93CC; ds_load_b32 v82, v80 /*v592*/ offset:2048
ds_ld32_as1 (0_1)  # next; ISA L5739 / 0x93D4; ds_load_b32 v83, v80 /*v592*/ offset:2176
s_set_vgpr_msb 0x32f2  # ISA L5740 / 0x93DC
wmma49 (7_1)  # K0; ISA L5741 / 0x93E0
s_set_vgpr_msb 0xf232  # ISA L5742 / 0x93F0
ds_ld32_as2 (0_2)  # next; ISA L5743 / 0x93F4; ds_load_b32 v84, v80 /*v592*/ offset:2304
ds_ld32_as3 (0_3)  # next; ISA L5744 / 0x93FC; ds_load_b32 v85, v80 /*v592*/ offset:2432
s_set_vgpr_msb 0x32f2  # ISA L5745 / 0x9404
wmma50 (7_2)  # K1; ISA L5746 / 0x9408
ds_ld128_a0 (0_0)  # next; ISA L5747 / 0x9418; ds_load_b128 v[8:11] /*v[776:779]*/, v73 /*v585*/
ds_ld128_a1 (0_1)  # next; ISA L5748 / 0x9420; ds_load_b128 v[12:15] /*v[780:783]*/, v73 /*v585*/ offset:512
ds_ld128_a2 (0_2)  # next; ISA L5749 / 0x9428; ds_load_b128 v[16:19] /*v[784:787]*/, v73 /*v585*/ offset:1024
ds_ld128_a3 (0_3)  # next; ISA L5750 / 0x9430; ds_load_b128 v[20:23] /*v[788:791]*/, v73 /*v585*/ offset:1536
ds_ld128_a4 (0_4)  # next; ISA L5751 / 0x9438; ds_load_b128 v[24:27] /*v[792:795]*/, v73 /*v585*/ offset:2048
ds_ld128_a5 (0_5)  # next; ISA L5752 / 0x9440; ds_load_b128 v[28:31] /*v[796:799]*/, v73 /*v585*/ offset:2560
wmma51 (7_3)  # K1; ISA L5753 / 0x9448
wmma52 (7_4)  # K0; ISA L5754 / 0x9458
ds_ld128_a6 (0_6)  # next; ISA L5755 / 0x9468; ds_load_b128 v[32:35] /*v[800:803]*/, v73 /*v585*/ offset:3072
ds_ld128_a7 (0_7)  # next; ISA L5756 / 0x9470; ds_load_b128 v[36:39] /*v[804:807]*/, v73 /*v585*/ offset:3584
ds_ld128_a8 (0_8)  # next; ISA L5757 / 0x9478; ds_load_b128 v[40:43] /*v[808:811]*/, v73 /*v585*/ offset:4096
ds_ld128_a9 (0_9)  # next; ISA L5758 / 0x9480; ds_load_b128 v[44:47] /*v[812:815]*/, v73 /*v585*/ offset:4608
ds_ld128_a10 (0_10)  # next; ISA L5759 / 0x9488; ds_load_b128 v[48:51] /*v[816:819]*/, v73 /*v585*/ offset:5120
ds_ld128_a11 (0_11)  # next; ISA L5760 / 0x9490; ds_load_b128 v[52:55] /*v[820:823]*/, v73 /*v585*/ offset:5632
wmma53 (7_5)  # K0; ISA L5761 / 0x9498
wmma54 (7_6)  # K1; ISA L5762 / 0x94A8
ds_ld128_a12 (0_12)  # next; ISA L5763 / 0x94B8; ds_load_b128 v[56:59] /*v[824:827]*/, v73 /*v585*/ offset:6144
ds_ld128_a13 (0_13)  # next; ISA L5764 / 0x94C0; ds_load_b128 v[60:63] /*v[828:831]*/, v73 /*v585*/ offset:6656
ds_ld128_a14 (0_14)  # next; ISA L5765 / 0x94C8; ds_load_b128 v[64:67] /*v[832:835]*/, v73 /*v585*/ offset:7168
ds_ld128_a15 (0_15)  # next; ISA L5766 / 0x94D0; ds_load_b128 v[68:71] /*v[836:839]*/, v73 /*v585*/ offset:7680
wmma55 (7_7)  # K1; ISA L5767 / 0x94D8
```

`# count: source=33; WMMA=8; DS=20 (b128=16, b32=4)`

9. **最后 8 条 WMMA 与 loop control**（ISA L5768–L5789 / `0x94E8–0x95A0`）

```text
s_add_co_u32 s24, s58, 0x500  # ISA L5768 / 0x94E8
s_cmp_lt_u32 s24, s70  # ISA L5769 / 0x94F0; sets SCC=(s24 < s70)
s_cselect_b64 s[34:35], s[34:35], s[62:63]  # ISA L5770 / 0x94F4; predicate from L5769
s_cselect_b32 s36, s36, s64  # ISA L5771 / 0x94F8; predicate from L5769
wmma56 (8_0)  # K0; ISA L5772 / 0x94FC
s_cselect_b32 s37, s37, s65  # ISA L5773 / 0x950C; predicate from L5769
s_cselect_b32 s38, s38, s66  # ISA L5774 / 0x9510; predicate from L5769
s_cselect_b32 s70, s70, s71  # ISA L5775 / 0x9514; predicate from L5769
s_cselect_b64 s[26:27], s[56:57], 0  # ISA L5776 / 0x9518; predicate from L5769
wmma57 (8_1)  # K0; ISA L5777 / 0x951C
s_add_nc_u64 s[34:35], s[34:35], s[26:27]  # ISA L5778 / 0x952C
s_cmp_lt_u32 s24, s71  # ISA L5779 / 0x9530; sets SCC=(s24 < s71)
s_cselect_b32 s39, s39, 0  # ISA L5780 / 0x9534; predicate from L5779
s_addk_co_i32 s58, 0x100  # ISA L5781 / 0x9538; K offset += 256
wmma58 (8_2)  # K1; ISA L5782 / 0x953C
s_cmp_lt_i32 s58, s59  # ISA L5783 / 0x954C; sets SCC=(s58 < s59)
wmma59 (8_3)  # K1; ISA L5784 / 0x9550
wmma60 (8_4)  # K0; ISA L5785 / 0x9560
wmma61 (8_5)  # K0; ISA L5786 / 0x9570
wmma62 (8_6)  # K1; ISA L5787 / 0x9580
wmma63 (8_7)  # K1; ISA L5788 / 0x9590
s_cbranch_scc0 1388  # ISA L5789 / 0x95A0; predicate from L5783: s58 >= s59; this shape falls through to P1
```

`# count: source=22; WMMA=8; DS=0 (b128=0, b32=0)`

wave1/3 单条动态 P0 路径汇总：

- `# total: source ISA=185 (30+9+34+9+10+13+25+33+22); WMMA=64 (8+8+8+8+4+4+8+8+8); K0=32, K1=32; opcode=32×16×128`
- `# C/D pair check: 32 fragments, each K0=1 + K1=1`；例如 `wmma0`（L5607 / `0x8E7C`, K0）与 `wmma2`（L5613 / `0x8EBC`, K1）累加同一 `v[100:115]` fragment。
- `# total: physical DS=64 b128 + 16 b32 = 80`
- `# current 后半: A/AS IDs=16..31/4..7; B/BS IDs=16..31/4..7`
- `# next 前半: B/BS IDs=0..15/0..3; A/AS IDs=0..15/0..3`
- `# total: bundle=4 × (16 b128 + 4 b32)`
- `# total: TDM=1/wave; s_wait_dscnt=6; tensor-wait=1 (s_wait_tensorcnt 2)`
- `# DS-wait upper bounds: 8→20→8→20→8→4`
- `# total: WG barrier=1 signal + 1 wait; K-offset=+256`

因此 P0 的真实 operand 发射顺序是 `A-current 后半 → B-current 后半 → B-next 前半 → A-next 前半`；其中步骤6先发 4 条 SB-next scale，步骤7再发 16 条 B-next data。L5789 对本 shape 不跳转，经过 L5790–L5792 落入 P1；P0 已发出的 next 前半在 P1 成为 current 前半而不重发，P1 同步后 L5885 / `0x997C` 仅由 wave1 回填 B slot1/body5、由 wave3 回填 SB slot1/body5。

#### 5.2.1 wave1/3 P0 的 host tile 覆盖

A data、SA、B data、SB 各自按动态 occurrence 独立编号；坐标表只把 occurrence 放回逻辑 M/N/K tile，不能按坐标或 offset 反向重编号。group0/1 仅切非 K 维：group0=`A M[0:63] / B N[0:63]`，group1=`A M[64:127] / B N[64:127]`；每组覆盖完整 K256，`0_i`/`1_i` 不是 K0/K1。

首 body 的 group0 来自 wave1 L1682–L1723、wave3 L3092–L3133，group1 来自 P0 L5608–L5676；动态 DS 总顺序与 wave0/2 不同：wave1/3 Prologue 为 `SB0..3, AS0..3, B0..15, A0..15`，P0 再发 `AS4..7, A16..31, BS4..7, B16..31, BS0..3(next), B0..15(next), AS0..3(next), A0..15(next)`。

**A data：每格一条 `ds_load_b128 = M16×K64`**

| Host M ↓ / K → | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| M0 [0:15] | `ds_ld128_a0` (`0_0`) | `ds_ld128_a1` (`0_1`) | `ds_ld128_a2` (`0_2`) | `ds_ld128_a3` (`0_3`) |
| M1 [16:31] | `ds_ld128_a4` (`0_4`) | `ds_ld128_a5` (`0_5`) | `ds_ld128_a6` (`0_6`) | `ds_ld128_a7` (`0_7`) |
| M2 [32:47] | `ds_ld128_a8` (`0_8`) | `ds_ld128_a9` (`0_9`) | `ds_ld128_a10` (`0_10`) | `ds_ld128_a11` (`0_11`) |
| M3 [48:63] | `ds_ld128_a12` (`0_12`) | `ds_ld128_a13` (`0_13`) | `ds_ld128_a14` (`0_14`) | `ds_ld128_a15` (`0_15`) |
| M4 [64:79] | `ds_ld128_a16` (`1_0`) | `ds_ld128_a17` (`1_1`) | `ds_ld128_a18` (`1_2`) | `ds_ld128_a19` (`1_3`) |
| M5 [80:95] | `ds_ld128_a20` (`1_4`) | `ds_ld128_a21` (`1_5`) | `ds_ld128_a22` (`1_6`) | `ds_ld128_a23` (`1_7`) |
| M6 [96:111] | `ds_ld128_a24` (`1_8`) | `ds_ld128_a25` (`1_9`) | `ds_ld128_a26` (`1_10`) | `ds_ld128_a27` (`1_11`) |
| M7 [112:127] | `ds_ld128_a28` (`1_12`) | `ds_ld128_a29` (`1_13`) | `ds_ld128_a30` (`1_14`) | `ds_ld128_a31` (`1_15`) |

**A scale：每格一条 `ds_load_b32 = M32×K128 scales`**

| Host M ↓ / K → | K0 [0:127] | K1 [128:255] |
|---|---|---|
| M0 [0:31] | `ds_ld32_as0` (`0_0`) | `ds_ld32_as1` (`0_1`) |
| M1 [32:63] | `ds_ld32_as2` (`0_2`) | `ds_ld32_as3` (`0_3`) |
| M2 [64:95] | `ds_ld32_as4` (`1_0`) | `ds_ld32_as5` (`1_1`) |
| M3 [96:127] | `ds_ld32_as6` (`1_2`) | `ds_ld32_as7` (`1_3`) |

**B data：每格一条 `ds_load_b128 = N16×K64`**

| Host N ↓ / K → | K[0:63] | K[64:127] | K[128:191] | K[192:255] |
|---|---|---|---|---|
| N0 [0:15] | `ds_ld128_b0` (`0_0`) | `ds_ld128_b1` (`0_1`) | `ds_ld128_b4` (`0_4`) | `ds_ld128_b5` (`0_5`) |
| N1 [16:31] | `ds_ld128_b2` (`0_2`) | `ds_ld128_b3` (`0_3`) | `ds_ld128_b6` (`0_6`) | `ds_ld128_b7` (`0_7`) |
| N2 [32:47] | `ds_ld128_b8` (`0_8`) | `ds_ld128_b9` (`0_9`) | `ds_ld128_b12` (`0_12`) | `ds_ld128_b13` (`0_13`) |
| N3 [48:63] | `ds_ld128_b10` (`0_10`) | `ds_ld128_b11` (`0_11`) | `ds_ld128_b14` (`0_14`) | `ds_ld128_b15` (`0_15`) |
| N4 [64:79] | `ds_ld128_b16` (`1_0`) | `ds_ld128_b17` (`1_1`) | `ds_ld128_b20` (`1_4`) | `ds_ld128_b21` (`1_5`) |
| N5 [80:95] | `ds_ld128_b18` (`1_2`) | `ds_ld128_b19` (`1_3`) | `ds_ld128_b22` (`1_6`) | `ds_ld128_b23` (`1_7`) |
| N6 [96:111] | `ds_ld128_b24` (`1_8`) | `ds_ld128_b25` (`1_9`) | `ds_ld128_b28` (`1_12`) | `ds_ld128_b29` (`1_13`) |
| N7 [112:127] | `ds_ld128_b26` (`1_10`) | `ds_ld128_b27` (`1_11`) | `ds_ld128_b30` (`1_14`) | `ds_ld128_b31` (`1_15`) |

**B scale：每格一条 `ds_load_b32 = N32×K128 scales`**

| Host N ↓ / K → | K0 [0:127] | K1 [128:255] |
|---|---|---|
| N0 [0:31] | `ds_ld32_bs0` (`0_0`) | `ds_ld32_bs1` (`0_1`) |
| N1 [32:63] | `ds_ld32_bs2` (`0_2`) | `ds_ld32_bs3` (`0_3`) |
| N2 [64:95] | `ds_ld32_bs4` (`1_0`) | `ds_ld32_bs5` (`1_1`) |
| N3 [96:127] | `ds_ld32_bs6` (`1_2`) | `ds_ld32_bs7` (`1_3`) |

wave1/3 的 current 前/后半 physical VGPR 分别为：A data `v776:839`/`v8:71`，SA `v82:85`/`v86:89`，B data `v264:327`/`v520:583`，SB `v92:95`/`v96:99`。B 的动态 offset 序列为 `0,512,2048,2560,1024,1536,3072,3584,4096,4608,6144,6656,5120,5632,7168,7680`，所以坐标表必然出现 `b0,b1,b4,b5` 等非单调排列；这保留了 ISA occurrence ID，没有按 offset 重排。

下表坐标是 wave-local `M128×N128×K256`。host A（M×K）对应硬件 Matrix B，host B（N×K）对应硬件 Matrix A；K0/K1 由两侧 data VGPR、`v82..89`/`v92..99` scale 索引及相同 C/D fragment 配对共同核验，而不是套用 wave0/2 的动态编号。

`wmma0..wmma63` 严格按 P0 L5605–L5789 动态顺序编号。wave1/3 与 wave0/2 的编号差异是实质性的：这里 `M2N0K0=wmma8`、`M3N0K0=wmma12`、`M4N0K0=wmma16`、`M5N0K0=wmma20`，不能复制 §5.1 的 `wmma4/5/32/36`；例如 `wmma0` 的 B/A 源为 `v264:279`/`v776:783`，`wmma2` 为 `v280:295`/`v784:791`，二者使用 K0/K1 scale 并累加同一 `v100:115`。

K0 表的每个格子计算 `M16×N32×K128`，K 范围为 `[0,127]`：

| K0 [0,127] / Host M ↓, Host N → | N0 [0,31] | N1 [32,63] | N2 [64,95] | N3 [96,127] |
|---|---:|---:|---:|---:|
| M0 [0,15] | `wmma0` (`0_0`) | `wmma1` (`0_1`) | `wmma32` (`4_0`) | `wmma40` (`6_0`) |
| M1 [16,31] | `wmma4` (`0_4`) | `wmma5` (`0_5`) | `wmma33` (`4_1`) | `wmma41` (`6_1`) |
| M2 [32,47] | `wmma8` (`1_0`) | `wmma9` (`1_1`) | `wmma36` (`5_0`) | `wmma44` (`6_4`) |
| M3 [48,63] | `wmma12` (`1_4`) | `wmma13` (`1_5`) | `wmma37` (`5_1`) | `wmma45` (`6_5`) |
| M4 [64,79] | `wmma16` (`2_0`) | `wmma17` (`2_1`) | `wmma48` (`7_0`) | `wmma56` (`8_0`) |
| M5 [80,95] | `wmma20` (`2_4`) | `wmma21` (`2_5`) | `wmma49` (`7_1`) | `wmma57` (`8_1`) |
| M6 [96,111] | `wmma24` (`3_0`) | `wmma25` (`3_1`) | `wmma52` (`7_4`) | `wmma60` (`8_4`) |
| M7 [112,127] | `wmma28` (`3_4`) | `wmma29` (`3_5`) | `wmma53` (`7_5`) | `wmma61` (`8_5`) |

K1 表的每个格子计算同一个 M×N output fragment 在 `[128,255]` 上的第二次 K128 累加：

| K1 [128,255] / Host M ↓, Host N → | N0 [0,31] | N1 [32,63] | N2 [64,95] | N3 [96,127] |
|---|---:|---:|---:|---:|
| M0 [0,15] | `wmma2` (`0_2`) | `wmma3` (`0_3`) | `wmma34` (`4_2`) | `wmma42` (`6_2`) |
| M1 [16,31] | `wmma6` (`0_6`) | `wmma7` (`0_7`) | `wmma35` (`4_3`) | `wmma43` (`6_3`) |
| M2 [32,47] | `wmma10` (`1_2`) | `wmma11` (`1_3`) | `wmma38` (`5_2`) | `wmma46` (`6_6`) |
| M3 [48,63] | `wmma14` (`1_6`) | `wmma15` (`1_7`) | `wmma39` (`5_3`) | `wmma47` (`6_7`) |
| M4 [64,79] | `wmma18` (`2_2`) | `wmma19` (`2_3`) | `wmma50` (`7_2`) | `wmma58` (`8_2`) |
| M5 [80,95] | `wmma22` (`2_6`) | `wmma23` (`2_7`) | `wmma51` (`7_3`) | `wmma59` (`8_3`) |
| M6 [96,111] | `wmma26` (`3_2`) | `wmma27` (`3_3`) | `wmma54` (`7_6`) | `wmma62` (`8_6`) |
| M7 [112,127] | `wmma30` (`3_6`) | `wmma31` (`3_7`) | `wmma55` (`7_7`) | `wmma63` (`8_7`) |

完整性检查：K0/K1 各有 `8×4=32` 个格子，两表合计 64 条，`wmma0..wmma63` 各出现一次；每个 `(M,N)` 的 K0/K1 条目使用同一 C/D fragment，`(K,M,N)` 的 `2×8×4=64` 个组合无重复、无遗漏。四类 DS 集合也分别完整：A/B data 各为 `0..31`，AS/BS 各为 `0..7`；P0 physical DS 恰为 `64 b128 + 16 b32`。

### 5.3 同步与末端切分的 ISA 锚点

wave0/2：

```text
L4926  0x74C8  s_wait_dscnt 8
L4927  0x74CC  s_wait_tensorcnt 2
L4928  0x74D0  s_barrier_signal -1
L4935  0x751C  s_barrier_wait 0xffff
L4937  0x7530  tensor_load_to_lds
L4955  0x75CC  s_wait_dscnt 10
```

wave1/3：

```text
L5687  0x91DC  s_wait_dscnt 8
L5688  0x91E0  s_wait_tensorcnt 2
L5689  0x91E4  s_barrier_signal -1
L5696  0x9230  s_barrier_wait 0xffff
L5698  0x9244  tensor_load_to_lds
L5710  0x92B0  s_wait_dscnt 4
```

所以两条路径的六个 DS wait 上界序列分别是
`8→20→8→20→8→10` 和 `8→20→8→20→8→4`。末端 10/4 只来自
next-bundle 的拆分发射位置：wave0/2 已发 `4 scale + 6 data`，
wave1/3 只发 `4 scale`；它们不是 wait latency，更不是 ATT 未观测到的
平均 cycle 数。

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
