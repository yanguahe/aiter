# gfx950 / CDNA4（MI350X & MI355X）硬件资料速查

<!-- markdown-toc-generator:start -->
## Table of Contents

- [1. gfx950 架构层次：XCC 与 XCD 是什么关系？](#1-gfx950-架构层次xcc-与-xcd-是什么关系)
  - [一颗芯片上的数量（MI350X / MI355X 相同）](#一颗芯片上的数量mi350x-mi355x-相同)
  - [一个 CU 里有几个 SIMD / SQ / SQC？](#一个-cu-里有几个-simd-sq-sqc)
- [2. 支持的数据类型 & 对应算力峰值](#sec-2--支持的数据类型-对应算力峰值)
  - [峰值算力（Peak Theoretical，Table 2，CDNA4-WP:1145-1173）【文档】](#峰值算力peak-theoreticaltable-2cdna4-wp1145-1173文档)
- [3. 内存 / 缓存层次：容量与带宽](#3-内存-缓存层次容量与带宽)
  - [HBM](#hbm)
  - [AMD Infinity Cache™（LLC，位于 IOD）](#amd-infinity-cachellc位于-iod)
  - [Infinity Fabric™ 的作用](#infinity-fabric-的作用)
  - [L2 Cache（每个 XCD 一份）](#l2-cache每个-xcd-一份)
  - [L1 Vector Data Cache（每个 CU 一份，即 TCP）](#l1-vector-data-cache每个-cu-一份即-tcp)
  - [LDS（Local Data Share，片上便签存储，非 cache）](#ldslocal-data-share片上便签存储非-cache)
- [4. 关于 “L0 缓存” 的澄清](#4-关于-l0-缓存-的澄清)
- [5. 重要频率参数](#5-重要频率参数)
- [6. MI350X vs MI355X 的区别](#6-mi350x-vs-mi355x-的区别)
- [附：常用换算备忘](#附常用换算备忘)

<!-- markdown-toc-generator:end -->

> 来源：`.cursor/rules/` 下的 AMD 官方资料 +  bundled HW corpus。
> 标注规范遵循 `flydsl-align-reference-kernel.mdc` Stage 4：每条结论给出**文件 + 行号**，并区分
> **【文档】**（资料明确写出）与**【推算】**（由文档给出的 per-cycle 宽度 × 频率换算，非原文直接给出）。
>
> 引用的资料文件（均在 `cursor_rules/fmha_flydsl_new_api_opt/.cursor/rules/`）：
> - `amd-cdna-4-architecture-whitepaper.txt`（简称 **CDNA4-WP**）
> - `amd-cdna-3-white-paper.txt`（简称 **CDNA3-WP**，CDNA4 多处“沿用上代”，结构细节以此为准）
> - `mi300-mi350-hw-docs.txt`（简称 **CORPUS**，754 个内嵌文档的大语料）

---

## 1. gfx950 架构层次：XCC 与 XCD 是什么关系？

**结论：在 MI300/MI350 这代里，XCD 与 XCC 指的是同一个东西的两个视角——XCD 是物理芯粒（die），XCC 是该 die 上的逻辑计算核，二者 1:1。**

- **XCD = Accelerator Complex Die（加速器复合芯粒）**，是物理 chiplet。
  CDNA4-WP:37 —— “Each AMD Instinct™ MI350 Series GPU integrates 8 vertically stacked accelerator complex dies (XCD)…”。【文档】
- **XCC = ACCELERATED COMPUTE CORE（加速计算核）**，是术语表/harvesting 文档里的逻辑单元名。
  CORPUS:60624-60625 术语表：“XCC = ACCELERATED COMPUTE CORE”；CORPUS:60684 “MI300 project fundamentally changes GFXIP architecture by **grouping 4 SE into XCC**”。【文档】
- 即：一个 XCD 物理 die 上承载一个 XCC（= 4 个 Shader Engine 的集合）。软件/驱动里常见的 `XCC_ID`（CDNA4-ISA Table 7，CORPUS:6020 / ISA:1094）就是这个 die 的编号。所以**“几个 XCD = 几个 XCC”**。【文档+推断】

### 一颗芯片上的数量（MI350X / MI355X 相同）

| 层级 | 数量 | 来源 |
|---|---|---|
| XCD（= XCC） / 芯片 | **8** | CDNA4-WP:37、CDNA4-WP:1127-1128（Table 2: XCD=8）【文档】 |
| SE（Shader Engine） / XCD | **4** | CDNA4-WP:757-761（每个 XCD 画了 4 个 SHADER ENGINE）；CORPUS:60661 “4 SE in one XCC”【文档】 |
| SE / 芯片 | **32**（8×4） | 推算【推算】 |
| CU / XCD | **36 物理（32 激活）**，排成 4 组 × 9 CU | CDNA4-WP:108-109 “36 CDNA 4 Compute Units organized as four arrays of 9 CUs, of which 32 are active”【文档】 |
| CU / SE | **9 物理（8 激活）** | 36/4=9；32/4=8【推算】 |
| CU / 芯片 | **256 激活**（最多 256） | CDNA4-WP:111-112 “up to 256 CUs”；Table 2 COMPUTE UNITS=256（WP:1130-1132）【文档】 |
| Stream Processor（SP，= ALU 通道） / 芯片 | **16,384** | CDNA4-WP:1133-1135（Table 2）→ 16384/256 = **64 SP/CU**（=4 SIMD×16 lane）【文档】 |
| Matrix Core / 芯片 | **1,024** | CDNA4-WP:1136-1138、WP:822 → 1024/256 = **4 Matrix Core/CU**【文档】 |

> 对比上代 MI300（CDNA3）：8 XCC，每 SE **10 CU**（CORPUS:60662），共 ≤304 CU（CDNA3-WP:86）。
> CDNA4 把每 SE 的 CU 略减（10→9），但每个 CU 更强（见 §2/§3）。

### 一个 CU 里有几个 SIMD / SQ / SQC？

| 单元 | 数量 | 说明与来源 |
|---|---|---|
| **SIMD** | **4 / CU**（每个 SIMD16，16 lane） | CORPUS:521616 “…four simds for each compute unit”；CORPUS:498032 “The SQ operates on a ‘4 phase’ system where it rotates working on each of the 4 SIMDs, 1 per clock cycle”；CORPUS:539199 “10 Waves/SIMD * 4 SIMD/CU = 40 waves per CU”。【文档】 4 SIMD × 16 lane = 64 lane/CU，正好对上 16384 SP / 256 CU = 64。 |
| **SQ**（Sequencer / 标量+指令发射，含 scalar ALU） | **1 / CU** | SQ 以 4-phase 轮转驱动该 CU 的 4 个 SIMD（CORPUS:498032、498052），即一个 CU 一个 SQ 实例。【文档+推断】 |
| **SQC**（Scalar/Instruction Cache，标量常量 + 指令 L1） | **1 / 2 CU**（两个相邻 CU 共享） | CDNA4-WP:148-149 “the 64KB, 8-way set-associative instruction cache is **shared between two adjacent CUs**”；CDNA3-WP:121-122 同。注：CORPUS:63444 “SQC cannot be harvested with a single CU”，印证 SQC 服务一对 CU。【文档】 |

> 备注：SQC 即“64KB / 8-way 指令缓存（+标量数据缓存）”，每 2 个 CU 一份；这是 gfx950 标量/取指路径上的最低级缓存（见 §4 关于 “L0” 的说明）。

---

<a id="sec-2--支持的数据类型-对应算力峰值"></a>
## 2. 支持的数据类型 & 对应算力峰值

**新增/支持的数据类型（CDNA4 Matrix Core）**：FP64、FP32、FP16、BF16、FP8(E4M3/E5M2)，
**新引入 OCP micro-scaling 格式 MXFP8、MXFP6（E3M2/E2M3）、MXFP4（E2M1）**，以及 INT8、INT4。
（CDNA4-WP:526-595、600-603：“introduce instruction and hardware support for industry-standard
micro-scaling formats including MXFP8, MXFP6, and MXFP4”。【文档】）

> 注意：上代硬件支持的 **TF32 在 CDNA4 已移出硬件**，改由 BF16 软件仿真（CDNA4-WP:607-609）。【文档】

### 峰值算力（Peak Theoretical，Table 2，CDNA4-WP:1145-1173）【文档】

| 数据类型 | MI350X | MI355X |
|---|---|---|
| FP64 Vector | 72.1 TF | 78.6 TF |
| FP32 Vector | 144.2 TF | 157.3 TF |
| FP64 Matrix | 72.1 TF | 78.6 TF |
| FP32 Matrix | 144.2 TF | 157.3 TF |
| FP16 \| FP16(稀疏) | 2.3 PF \| 4.6 PF | 2.5 PF \| 5.0 PF |
| BF16 \| BF16(稀疏) | 2.3 PF \| 4.6 PF | 2.5 PF \| 5.0 PF |
| FP8 \| FP8(稀疏) | 4.6 PF \| 9.2 PF | 5.0 PF \| 10 PF |
| FP6/FP4 \| (稀疏) | 9.2 PF \| 18.5 PF | 10 PF \| 20 PF |
| INT8/INT4 \| (稀疏) | 4.6 POPs \| 9.2 POPs | 5.0 POPs \| 10 POPs |

> 两款峰值差异仅来自频率（2.2 vs 2.4 GHz），约 1.087×。FP6/FP4 是本代最大亮点，相对上代 MI325X 的 FP8
> 提升约 3.85×（CDNA4-WP:57-59）。FP6 与 FP4 算力相同（10 PF），均为 16834 FLOPs/clock/CU（WP:677-688）。【文档】

---

## 3. 内存 / 缓存层次：容量与带宽

### HBM
| 项目 | MI350X / MI355X | 来源 |
|---|---|---|
| HBM 类型 | HBM3E | CDNA4-WP:1175-1177【文档】 |
| 容量 | **288 GB**（8 stack × 36 GB） | CDNA4-WP:841-842、1175-1177【文档】 |
| 接口 | 1024-bit × 8 stack（12-Hi） | CDNA4-WP:39、1182-1184【文档】 |
| 引脚速率 | 8 Gbps（比 MI325X 快 >33%） | CDNA4-WP:839-841【文档】 |
| 峰值带宽 | **up to 8.0 TB/s** | CDNA4-WP:60、824、1185-1187【文档】 |

### AMD Infinity Cache™（LLC，位于 IOD）
| 项目 | 值 | 来源 |
|---|---|---|
| 容量 | **256 MB**，16-way 组相联，memory-side cache | CDNA4-WP:836-837、1194-1196【文档】 |
| 结构 | 每个 HBM stack 16 条并行 channel，每 channel 64B 宽，挂 2MB banked 阵列；扇出到 8 个 stack | CDNA4-WP:837-838【文档】 |
| 带宽 | CDNA4-WP 未直接给聚合数；上代 CDNA3 的 **LLC→XCD 聚合 17.2 TB/s**（CDNA3-WP:281-282），CDNA4 “组织基本不变”（WP:836），可作量级参考 | CDNA3-WP:281-282【文档】 / CDNA4 量级【推断】 |

### Infinity Fabric™ 的作用
**作用：把芯片内的 8 个 XCD 与 2 个 IOD 连成一体（on-package），并对外把节点内 8 颗 GPU 全互联——
即“片内胶水 + 片间 scale-up 互联”。** （CDNA4-WP:38、811-816、955-960）【文档】
- 片内：XCD 的 L2 把流量汇聚后经 IF 接入 IOD 上的 Infinity Cache / HBM；两个 IOD 之间直连（比 CDNA3 快约 14%，WP:834-835）。Advanced Package bisection 5.5 TB/s（WP:814）。【文档】
- 片间：8 条 16-bit 全双工 IF 链路，38.4 Gbps/lane，每链路单向 76.8 GB/s；其中 1 条复用为 PCIe Gen5 接 host。
  每 GPU P2P 聚合 **1075.2 GB/s**，总聚合 **1203.2 GB/s**（>1 TB/s）。（CDNA4-WP:955-960、1203-1210、1307-1312；PCIe Gen5 to host 128 GB/s，WP:819-820）【文档】

### L2 Cache（每个 XCD 一份）
| 项目 | 值 | 来源 |
|---|---|---|
| 容量 / 组相联 | **4 MB**，16-way（每 XCD 一份，共 8 份） | CDNA4-WP:723-724、1191-1193【文档】 |
| channel | 16 条并行 channel（每条 256KB），读出 128B/line，写 64B/line | CDNA4-WP:724-725；CDNA3-WP:289-293【文档】 |
| 读带宽（per XCD） | **2 KB/clock**（文档原文，CDNA3 结构 CDNA4 沿用） | CDNA3-WP:292 “combined throughput of 2KBytes/clock for each XCD”【文档】 |
| 读带宽（per XCD，换算） | MI355X@2.4G ≈ **4.92 TB/s**；MI350X@2.2G ≈ 4.51 TB/s | 2048B×freq【推算】 |
| 读带宽（整芯 8 XCD，换算） | MI355X ≈ **39.3 TB/s**；MI350X ≈ 36.0 TB/s | ×8【推算】（对比 CDNA3 文档聚合 34.4 TB/s，CDNA3-WP:296） |
| 策略 | writeback + write-allocate，XCD 内一致；CDNA4 新增可缓存 non-coherent DRAM 数据、writeback 后保留副本 | CDNA4-WP:725-729【文档】 |

### L1 Vector Data Cache（每个 CU 一份，即 TCP）
| 项目 | 值 | 来源 |
|---|---|---|
| 容量 / 组相联 / line | **32 KiB**，64-way，128B cache line | CDNA4-WP:722-723、1188-1190【文档】 |
| 带宽（到 core） | 128 B/clock（CDNA3 已把 line/带宽翻倍） | CDNA3-WP:259-263【文档】 |
| 带宽（per CU，换算） | MI355X@2.4G ≈ 307 GB/s；MI350X@2.2G ≈ 282 GB/s | 128B×freq【推算】 |
| 带宽（整芯 256 CU，换算） | MI355X ≈ **78.6 TB/s**；MI350X ≈ 72.1 TB/s | ×256【推算】 |
| 一致性 | 非常宽松，需显式 `s_waitcnt`/`buffer_inv` 同步 | CDNA3-WP:263-264【文档】 |

### LDS（Local Data Share，片上便签存储，非 cache）
| 项目 | CDNA4(gfx950) | CDNA3(gfx942) | 来源 |
|---|---|---|---|
| 容量 / CU | **160 KB**（bank 数增加） | 64 KB | CDNA4-WP:715-716；CDNA3-WP:256【文档】 |
| 读带宽 | **256 B/clock**（翻倍） | 128 B/clock | CDNA4-WP:716【文档】 |
| 新特性 | 支持从 L1 直接 load 入 LDS（省 VGPR、降延迟）；MFMA transpose-load | — | CDNA4-WP:718-721【文档】 |
| 带宽（整芯，换算）| MI355X@2.4G ≈ 157 TB/s | — | 256B×256CU×2.4G【推算】 |

---

## 4. 关于 “L0 缓存” 的澄清

CDNA（gfx9 系列，含 gfx950）**没有 RDNA 那种独立的 “L0 vector cache” 命名**。CDNA 的数据缓存层级是：

- **L1 = 每 CU 的向量数据缓存（TCP）**，32KB —— 这是 CDNA 数据路径上最低一级 cache（见 §3）。
- **L2 = 每 XCD 的共享缓存（TCC）**，4MB。
- **LLC = Infinity Cache**，256MB（IOD 上）。
- 比 L1 更低的是寄存器堆（每 SIMD 的 VGPR/SGPR）与 LDS（便签存储），它们不是 cache。
- 标量/取指路径上有 **SQC**（每 2 CU 共享的 64KB 指令+标量常量缓存），有时被类比为标量侧的 L1/“L0”。

所以若问 “L0 容量/带宽”：在 gfx950 语境下严格说不存在 L0 数据 cache；如把“最贴近 ALU 的存储”当作 L0，
则对应寄存器堆 + LDS（160KB/CU，256 B/clk）或 per-CU L1（32KB，128 B/clk）。【文档+说明】

---

## 5. 重要频率参数

| 参数 | MI350X | MI355X | 来源 |
|---|---|---|---|
| Max Engine Clock（Peak GPU 频率） | **2,200 MHz** | **2,400 MHz** | CDNA4-WP:1139-1141（Table 2）【文档】 |
| HBM3E 引脚数据率 | 8 Gbps | 8 Gbps | CDNA4-WP:839-841【文档】 |
| Infinity Fabric lane 速率 | 38.4 Gbps | 38.4 Gbps | CDNA4-WP:957-958【文档】 |
| PCIe（host） | Gen5（128 GB/s） | Gen5（128 GB/s） | CDNA4-WP:819-820、1211-1213【文档】 |

> 工艺/封装相关频率背景：XCD 用 TSMC N3P，IOD 用 N6（CDNA4-WP:81-87、826）。【文档】

---

## 6. MI350X vs MI355X 的区别

**核心：两者是同一颗 CDNA4 die 的不同功耗/散热档位——计算/内存规格几乎完全相同，差别在散热方式、功耗、峰值频率，
进而带来约 ~8.7% 的峰值算力差。**（CDNA4-WP:50-55、991-996、1062-1067、Table 2）

| 维度 | MI350X | MI355X | 来源 |
|---|---|---|---|
| 散热 | 风冷 / Passive（AC） | 直接液冷 DLC / Liquid | WP:50-51、994-996、1240-1242【文档】 |
| 最大功耗 | **1000 W** | **1400 W** | WP:50-53、1243-1245【文档】 |
| Peak 引擎频率 | 2200 MHz | 2400 MHz | WP:1139-1141【文档】 |
| 机架形态 | UBB8，4RU 托盘（drop-in 兼容 MI325X） | 2RU（另有 1OU 更高密度方案） | WP:992-996【文档】 |
| 峰值算力 | 略低（见 §2，例 FP8 4.6 PF / FP4 9.2 PF） | 略高（FP8 5.0 PF / FP4 10 PF） | Table 2【文档】 |
| Form Factor | OAM | OAM | WP:1237-1239【文档】 |

**两者完全相同的项**（WP Table 2，1122-1245）：架构 CDNA4；XCD=8；CU=256；Stream Processor=16384；
Matrix Core=1024；晶体管 185 Billion；HBM3E 288 GB / 8.0 TB/s；接口 1024-bit×8；L1=32KiB；L2=4MB；
Infinity Cache=256MB；7×16 IF + 1×16 PCIe Gen5；P2P 聚合 1075.2 GB/s / 总 1203.2 GB/s；
SR-IOV、最多 8 partition、全芯 ECC 等 RAS 特性。【文档】

> 一句话：**MI355X = MI350X 解锁更高功耗/液冷后的高频版**；规模一致，算力差仅来自 2.2→2.4 GHz。

---

## 附：常用换算备忘
- 每 CU lane 数：4 SIMD × 16 = 64；整芯 256 CU × 64 = 16384 SP。
- Matrix：1024 Matrix Core / 256 CU = 4 Matrix Core/CU。
- L2 per-XCD 读带宽换算：2048 B × f；整芯 ×8。
- L1 per-CU 读带宽换算：128 B × f；整芯 ×256。
- LDS per-CU 读带宽换算：256 B × f；整芯 ×256。
