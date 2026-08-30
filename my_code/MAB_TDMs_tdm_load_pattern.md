# `MAB_TDMs` 的 A/B TDM Load Pattern

本文分析对象：

- 反汇编：`my_code/MAB/MAB_TDMs.disasm.s`
- 问题规模：`M=16, N=65536, K=16384`
- 数据类型：FP16/BF16，每个元素 2 B
- 物理 workgroup grid：`(1,256,1)`
- workgroup：`(128,1,1)`，即 4 个 wave32
- cluster：`(1,4,1)`，共 64 个 cluster

这里的 “load” 仅指矩阵数据从 Global Memory 搬入 LDS 的
`tensor_load_to_lds`。后续 `ds_load_b128` 是 LDS→VGPR，不属于 Global
Memory load。

## 1. 总体二维分块

### 1.1 矩阵布局

TDM descriptor 把 K 放在维度 0，因此 K 是连续维：

| 矩阵 | 逻辑形状 | TDM tensor 形状 | Global 连续维 | 一整行字节数 |
|---|---:|---:|---|---:|
| A | `[M,K] = [16,16384]` | `[K,M] = [16384,16]` | K | `16384×2 = 32768 B` |
| B | `[N,K] = [65536,16384]` | `[K,N] = [16384,65536]` | K | `16384×2 = 32768 B` |

### 1.2 输出 tile 与 launch 分块

| 层级 | M 范围 | N 范围 | K 范围 | 数量 |
|---|---:|---:|---:|---:|
| 单个 wave32 | 16 | 64 | 16384 | 每 WG 4 个 |
| 单个 WG | 16 | 256 | 16384 | 256 个 |
| 单个 cluster（4 WG） | 16 | 1024 | 16384 | 64 个 |
| 整个 dispatch | 16 | 65536 | 16384 | 1 |

cluster 和 WG 的 N 坐标关系为：

```text
cluster_y = 0..63
local_wg_y = 0..3
wg_n = cluster_y * 4 + local_wg_y = 0..255
WG 的 N 起点 = wg_n * 256
```

反汇编中的对应代码：

```text
s_mul_i32    s91, s91, 4
s_add_co_i32 s31, s91, s94       ; s31 = cluster_y*4 + local_wg_y
```

每个 WG 覆盖：

```text
M = [0, 16)
N = [s31*256, s31*256 + 256)
K = [0, 16384)
```

## 2. TDM descriptor 的公共设置

A 和 B 都使用常量 `0x07590000 | workgroup_mask`。根据 MI400 Shader
Programming Guide §4.10.4 的 Group-1 descriptor 定义，字段解码如下：

| 字段 | 编码值 | 含义 |
|---|---:|---|
| `workgroup_mask` | 动态 | A 为 `0x000f`；B 为 one-hot |
| `data_size` | 1 | 每元素 2 B |
| `atomic_barrier_enable` | 0 | TDM 完成后不自动发送 LDS atomic barrier |
| `iterate_enable` | 1 | 一个 descriptor 内执行多次 tile iteration |
| `pad_enable` | 1 | 写入 LDS 时插入 padding |
| `early_timeout` | 0 | 不请求立即 multicast timeout |
| `pad_interval` | 5 | 每 64 DWORD，即每 256 B 数据插入一次 padding |
| `pad_amount` | 3 | 编码为 4 DWORD，即插入 16 B |

所以每条 Global 行的布局为：

| 位置 | Global Memory | LDS |
|---|---:|---:|
| 有效数据 | 256 B | 256 B |
| padding | 0 B | 16 B |
| 合计占用 | 256 B | 272 B |

padding 只改变 LDS 地址，不改变 Global Memory 的请求地址或字节量。

## 3. A 的 TDM load pattern

### 3.1 A descriptor

A descriptor 位于反汇编约 197–227 行：

| descriptor 字段 | 值 | 解释 |
|---|---:|---|
| `global_addr` | 当前 A 的 K-block 起点 | 指向 `A[:, k]` |
| `tensor_dim0` | 16384 | K |
| `tensor_dim1` | 16 | M |
| `tensor_dim0_stride` | 16384 | 相邻 M 行相隔一个完整 K |
| `tile_dim0` | 128 | 一次读取 K128 |
| `tile_dim1` | 8 | 每次 iteration 读取 8 个 M 行 |
| `iterate_count` | 编码 1 | 实际执行 2 次 |
| `global_addr_increment` | `8×16384` 个元素 | 第二次 iteration 跳到后 8 个 M 行 |
| `workgroup_mask` | `0x000f` | multicast 到 cluster 内 4 个 WG |

### 3.2 一个 K128 中的二维访问

令：

```text
q = 0..127
k0 = q * 128
```

则单条 A descriptor 的二维 tile 为：

| descriptor iteration | M 行 | K 列 | 连续行数 | 每行字节 | iteration 字节 |
|---:|---:|---:|---:|---:|---:|
| 0 | `[0,8)` | `[k0,k0+128)` | 8 | 256 B | 2048 B |
| 1 | `[8,16)` | `[k0,k0+128)` | 8 | 256 B | 2048 B |
| **合计** | `[0,16)` | `[k0,k0+128)` | **16** | **256 B** | **4096 B** |

从 `s67` 的 wave 分支和不同的 `s69` LDS 起点可以看出，wave0、wave1
各发出一条 A descriptor。两条 descriptor 的 Global 地址范围相同，但写入
不同的 LDS 区域：

| 发出 wave | Global A tile | 原始数据字节 | LDS 作用 |
|---:|---|---:|---|
| wave0 | `A[0:16,k0:k0+128]` | 4096 B | 第一份 A 布局 |
| wave1 | `A[0:16,k0:k0+128]` | 4096 B | 第二份 A 布局 |
| **descriptor payload 合计** | 同一组 A 地址读取两遍 | **8192 B** | 两份 LDS 副本 |

因此要区分：

| 口径 | 每个 K128 的 A 字节 |
|---|---:|
| 数学上唯一的 A tile | 4096 B |
| TDM descriptor 搬运量 | 8192 B |
| 每个目标 WG 的 LDS 接收量 | 8192 B |

Global 地址公式：

```text
A_addr(m,q) = A_base + 2 * (m*16384 + q*128)
```

在固定 `q` 下，实际空间 pattern 是：

| 访问顺序 | Global 范围 | 大小 | 到下一行起点的距离 |
|---:|---|---:|---:|
| 0 | `A[0][k0:k0+128]` | 256 B | 32768 B |
| 1 | `A[1][k0:k0+128]` | 256 B | 32768 B |
| … | … | … | … |
| 15 | `A[15][k0:k0+128]` | 256 B | — |

也就是说，单个 K128 phase 对 A 是“16 条 256 B 连续短行，行起点间隔
32 KiB”。

### 3.3 A 的 cluster multicast

A descriptor 的 `workgroup_mask=0x000f`：

| cluster 内 WG | A tile | mask 中对应 bit | 是否需要相同 A |
|---:|---|---:|---|
| WG0 | `A[0:16,k0:k0+128]` | 1 | 是 |
| WG1 | `A[0:16,k0:k0+128]` | 1 | 是 |
| WG2 | `A[0:16,k0:k0+128]` | 1 | 是 |
| WG3 | `A[0:16,k0:k0+128]` | 1 | 是 |

文档明确说明：`TENSOR_LOAD_TO_LDS` 的 `workgroup_mask` 非零时，TDM 会把
普通 `GLOBAL_LOAD_ASYNC` 改成 `CLUSTER_LOAD_ASYNC`。因此两条 A
descriptor 分别把各自的 A 副本 multicast 到 cluster 内 4 个 WG 的 LDS。

这减少的是 cluster 内重复请求；不同 cluster 仍访问同一份 A，跨 cluster 的
复用取决于 GL1/GL2 cache。

## 4. B 的 TDM load pattern

### 4.1 B descriptor

B descriptor 位于反汇编约 230–269 行：

| descriptor 字段 | 值 | 解释 |
|---|---:|---|
| WG 的 B 起点 | `s31×256` 行 | 每个 WG 负责不同的 N256 |
| `tensor_dim0` | 16384 | K |
| `tensor_dim1` | 65536 | N |
| `tensor_dim0_stride` | 16384 | 相邻 N 行相隔一个完整 K |
| `tile_dim0` | 128 | 一次读取 K128 |
| `tile_dim1` | 64 | 每次 iteration 读取 64 个 N 行 |
| `iterate_count` | 编码 1 | 实际执行 2 次 |
| `global_addr_increment` | `64×16384` 个元素 | 第二次 iteration 跳到后 64 个 N 行 |
| `workgroup_mask` | `1 << local_wg_y` | one-hot，只投递到当前 WG |

### 4.2 wave2/wave3 对 B 的二维分工

令当前 WG 的 N 起点为：

```text
n0 = s31 * 256
```

wave2 和 wave3 各自发出覆盖 N128 的 descriptor：

| descriptor 发出者 | iteration | N 行 | K 列 | 字节数 |
|---|---:|---:|---:|---:|
| wave2 | 0 | `[n0,n0+64)` | `[k0,k0+128)` | 16384 B |
| wave2 | 1 | `[n0+64,n0+128)` | `[k0,k0+128)` | 16384 B |
| wave3 | 0 | `[n0+128,n0+192)` | `[k0,k0+128)` | 16384 B |
| wave3 | 1 | `[n0+192,n0+256)` | `[k0,k0+128)` | 16384 B |
| **WG 合计** | — | `[n0,n0+256)` | `[k0,k0+128)` | **65536 B** |

Global 地址公式：

```text
B_addr(n,q) = B_base + 2 * (n*16384 + q*128)
```

固定 `q` 时，单个 WG 的 B pattern 是：

| N 行范围 | 行数 | 每行连续读取 | 行起点间隔 | 总字节 |
|---|---:|---:|---:|---:|
| `[n0,n0+256)` | 256 | 256 B | 32768 B | 65536 B |

B 的 mask 是当前 WG 的 one-hot bit，所以不同 WG 不共享 B：

| `local_wg_y` | mask | WG 的 N 范围（cluster 内相对值） |
|---:|---:|---:|
| 0 | `0x1` | `[0,256)` |
| 1 | `0x2` | `[256,512)` |
| 2 | `0x4` | `[512,768)` |
| 3 | `0x8` | `[768,1024)` |

## 5. 一个 cluster 的完整二维 pattern

令 `c=cluster_y`，则该 cluster 的 N 起点为 `c×1024`：

| cluster 内 WG | 全局 N 范围 | A tile | B tile | A mask | B mask |
|---:|---:|---|---|---:|---:|
| 0 | `[c×1024+0,c×1024+256)` | `A[0:16,k0:k0+128]` | 对应 N256 | `0xf` | `0x1` |
| 1 | `[c×1024+256,c×1024+512)` | 同上 | 对应 N256 | `0xf` | `0x2` |
| 2 | `[c×1024+512,c×1024+768)` | 同上 | 对应 N256 | `0xf` | `0x4` |
| 3 | `[c×1024+768,c×1024+1024)` | 同上 | 对应 N256 | `0xf` | `0x8` |

因此每个 K128 phase 要同时区分唯一地址集合与 descriptor 搬运量：

| 数据 | 唯一 Global 地址字节 | TDM descriptor payload | cluster 内是否复用 |
|---|---:|---:|---|
| A | 4096 B | 8192 B（wave0/1 各一份） | 每份均 multicast 到 4 WG |
| B | `4×65536 = 262144 B` | 262144 B | 4 WG 地址互不重叠 |
| **合计** | **266240 B** | **270336 B** | — |

这里的“唯一 Global 地址字节”只描述地址集合；“descriptor payload”统计
descriptor 所描述的源数据量。两者都不等同于精确 HBM transaction 字节，
实际 HBM 流量还受 cluster request 合并、GL1/GL2 命中和内部请求粒度影响。

## 6. K 方向扫描

K 被分为 128 个 K128 phase：

| phase `q` | K 范围 | 每行 Global 起始偏移 |
|---:|---:|---:|
| 0 | `[0,128)` | `0x0000` |
| 1 | `[128,256)` | `0x0100` |
| 2 | `[256,384)` | `0x0200` |
| … | … | … |
| 127 | `[16256,16384)` | `0x7f00` |

因为一个 K128 是 `128×2 = 0x100 B`，反汇编使用：

```text
s_add_co_u32 s70, s70, s97
```

其中 `s97=0x100`，使下一个 descriptor 沿同一行向后推进 256 B。128 个
phase 最终连续覆盖每一行的完整 32 KiB K 数据。

### 每个 WG/cluster 的读取量口径

| 数据 | 每个 K128 的唯一 tile | 每个 K128 的 descriptor payload | 128 个 K128 的 payload |
|---|---:|---:|---:|
| A | 4096 B | 8192 B | 1048576 B（1 MiB） |
| B（单 WG） | 65536 B | 65536 B | 8388608 B（8 MiB） |
| **单 WG LDS 接收/消费口径合计** | **69632 B** | **73728 B** | **9437184 B（9 MiB）** |

A 是所有 WG 共同需要的数据，而且每个目标 WG 中保留两份 LDS A 副本；
cluster multicast 又会把每份副本投递到 4 个 WG。因此这些数字不能直接
作为 HBM 读取量。B 的 N tile 互不重叠，所以 B 仍是该规模下的主要
Global/HBM 流量。

## 7. Global request 与 LDS layout

### 7.1 预期的 Global transaction 形态

MI400 Shader Programming Guide §4.10.2 说明，在以下条件下 TDM 可以生成
direct-copy compliant 请求：

- fetch 为 128 B 或 256 B；
- Global 地址至少 128 B 对齐；
- tile 第一维为 128 B，或至少 256 B；
- LDS 地址至少 4 B 对齐。

本 kernel 的每行 tile 正好为 256 B，并且正常 HIP allocation 与 K128
偏移满足 128 B 对齐，所以属于 direct-copy eligible pattern。TDM 内部会
拆成 128/256 B 的异步 Global→LDS 请求；具体选择多少条 128 B 或 256 B
transaction 属于微架构行为，不能仅从 ISA 静态确定。

### 7.2 LDS padding 和流水 stage

| 项目 | 值 | 说明 |
|---|---:|---|
| Global 行有效数据 | 256 B | K128 FP16/BF16 |
| LDS 行占用 | 272 B | 256 B 数据 + 16 B padding |
| A 的 8 行子块 | `8×272 = 2176 B = 0x880` | B 的 LDS 区域从 `+0x880` 开始 |
| B 的 64 行子块 | `64×272 = 17408 B = 0x4400` | 每次 descriptor iteration |
| 两条 A descriptor | `2×16×272 = 8704 B = 0x2200` | 两份 A 副本 |
| 两条 B descriptor | `2×128×272 = 69632 B = 0x11000` | 合计 N256 |
| 一个 K128 的 padding 后 payload | `0x13200` | A 两份 + B 一份完整 N256 |
| stage stride | `0x4ca0 = 19616 B` | `s62` |
| 四路 stage/plane 跨距 | `4×0x4ca0 = 0x13280` | `s63`；比 `0x13200` 多 128 B 对齐空间 |

代码通过 `s60/s61 += 0x4ca0` 并在 `0x13280` 处回绕，在四个 LDS stage
之间轮转。descriptor iteration 和不同发出 wave 还会使用额外的
`0x13280` LDS 偏移，因此物理 LDS 布局不是一个简单连续的 A+B 矩阵。

## 8. 发射与等待

反汇编中静态出现 5 条 `tensor_load_to_lds`，但它们位于 prologue、循环和
尾部路径中，动态执行次数远大于 5。

| wave | 主要 TDM 职责 | tile |
|---:|---|---|
| wave0 / wave1 | 发出 A descriptor、推进 A 流水 | M16×K128 |
| wave2 / wave3 | 发出 B descriptor、推进 B 流水 | 各 N128×K128 |

硬件文档明确规定：

- 每个 wave 最多有 3 个 TDM op 等待 XACK；
- 每个 SIMD 最多有 6 个；
- TDM descriptor 可以在不同 wave 之间切换执行；
- 同一 wave 的 tensor 指令按顺序完成；
- `S_WAIT_TENSORCNT` 用于等待异步 TDM 完成。

kernel 使用：

```text
s_wait_tensorcnt 0x2
```

在等待点允许最多 2 个未完成 descriptor，随后可再发出一个，因此能够维持
每 wave 最多 3 个 descriptor 在途。之后通过 barrier 协调 wave，再用
`ds_load_b128` 从 LDS 读取 A/B 数据供 WMMA 使用。

## 9. Pattern 总结

| 特征 | A | B |
|---|---|---|
| Global tensor 形状 | K16384×M16 | K16384×N65536 |
| 每次连续 K 宽度 | 128 元素 = 256 B | 128 元素 = 256 B |
| 每个 descriptor 的行数 | `8×2 = 16` | `64×2 = 128` |
| 完成一个 WG tile 所需分工 | wave0/1 各复制一次完整 M16 | wave2/3 各覆盖 N128 |
| 每 WG、每 K128 唯一地址字节 | 4096 B | 65536 B |
| 每 WG、每 K128 descriptor payload | 8192 B | 65536 B |
| Global 行起点间隔 | 32768 B | 32768 B |
| cluster mask | `0xf` | one-hot |
| cluster 内复用 | 两份 A 副本分别做 4 WG multicast | 无 |
| LDS 行布局 | 256 B + 16 B padding | 256 B + 16 B padding |
| K 扫描 | 128 个 K128 phase | 128 个 K128 phase |

## 10. 资料依据

- `my_code/MAB/MAB_TDMs.disasm.s`
  - cluster/WG ID：约 15–29 行
  - A descriptor：约 197–227 行
  - B descriptor：约 230–269 行
  - TDM 流水：约 368–418 行
  - `s_wait_tensorcnt` 与 LDS/WMMA：约 378–600 行
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`
  - §4.10.2 Tensor and Tile Addressing
  - §4.10.3 Tensor Instruction Variants and Optional Operations
  - §4.10.4 Tensor DMA Descriptor
  - §4.10.8 Performance and Tracking

文档明确事实包括 descriptor 字段含义、padding、iteration、cluster
multicast、TENSORcnt 和 direct-copy 条件；A/B tile 数值、地址范围及流量是
依据反汇编立即数和本次问题规模推导得到的。
