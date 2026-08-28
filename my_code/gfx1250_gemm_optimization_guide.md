# gfx1250 / MI450-B0 GEMM 优化指南

> AMD Internal。本文面向 gfx1250（CDNA5 / MI450-B0、MI455X）手写或生成式低精度 GEMM
> kernel。内容综合了 `MI450-B0 GEMM Pipeline.pptx`、对应会议文字记录以及 MI400/CDNA5
> 的 ISA、Shader Programming、TX/TDM、GL1/GL2 和性能计数器资料。
>
> 分享中的 silicon 数字只代表当时的样机、固件、时钟和 kernel 版本。本文把“硬件规格”“公式推导”
> 和“分享中的实测观察”分开陈述；任何最终取舍都应在目标 stepping、真实随机数据和目标 shape 上复测。

## 1. 结论先行

### 1.1 Compute-bound GEMM

Compute-bound 的目标不是单纯增加 WMMA 数量，而是让四个 SIMD 的 XDL pipeline 在真实运行频率下持续有
有效工作，同时确保 TDM、LDS、GL1、GL2、指令供给和 epilogue 都不会让 XDL 断粮。

优先级通常如下：

1. 选择峰值相同但 co-execution 空间更大的 WMMA 形状。
2. 用足够大的 `tile_m × tile_n` 提高 A/B 复用；对称数据类型和对称复用下优先接近方形。
3. 用 pre-shuffle、对齐和 TDM descriptor 尽量生成 256B direct-copy 请求。
4. 在 cluster 内按 A/B 的实际复用方向做 multicast；不要把“cluster 最多 16 个 WG”和“单次返回最多合并
   5 个请求”混为一谈。
5. 用多级流水满足 `(stages - 1) × compute_cycles_per_stage >= measured_latency`。
6. 把地址计算、LDS 访问和其他独立指令塞入 WMMA 的合法 co-execution/独立 issue 空间，并显式处理数据冒险。
7. 缩短 prologue/epilogue，保持热点代码适配 64KB WGP I$，必要时使用 persistent 调度。
8. 同时报 cycles、平均频率和真实数据性能；静态输入可能严重高估 compute-bound 吞吐。

### 1.2 Memory-bound GEMM

Memory-bound 的首要目标是减少真正穿过 HBM/GL2 的字节数，并保持足够并发隐藏延迟，而不是机械照搬
compute-bound 的“大 tile、occupancy=1、所有输入都 pre-shuffle”。

优先级通常如下：

1. 先确定瓶颈位于 HBM、GL2、GL1、TCP/WGP$、TDM tracking、LDS，还是并发不足造成的 latency bound。
2. 只缓存和 multicast 会复用的数据。Decode/GEMV-like GEMM 常见的是 activation 可跨 N tile 共享，
   weight 基本流式读取。
3. 根据输入字节宽度、cluster 复用倍数和实际 shape 选择矩形 tile；非对称 A8W4 并不天然以方形最优。
4. 尽量让每次有效访问成为对齐的 128B 或 256B 请求；优先 256B direct copy，但不要为了“纯 256B”
   反而制造非对齐或 indirect copy。
5. memory-bound kernel 通常需要更多 resident WG、更多独立请求或 grouped/persistent 调度来隐藏延迟。
6. pre-shuffle 必须计算端到端成本。长期复用的 weight/scale 通常值得；在线 activation 未必值得。
7. 当 `M×N` 并行度不足时再考虑 split-K/stream-K；把额外 partial-C 写回与 reduction 流量计入成本。
8. 尽量融合 dequant、bias、activation、scale 和输出转换，避免多次 HBM 往返。

## 2. gfx1250 GEMM 所需的硬件模型

### 2.1 WGP 与 XDL

- 一个 WGP 有 4 个 SIMD32 和 4 个 SQ，原生 wave32。
- 一个 workgroup 的 waves 分布在同一 WGP 的 4 个 SIMD 上。
- MI450 每个 SIMD 的物理 VGPR pool 为 1024 个 wave32 VGPR；单 wave 最多可分配 1024 个 VGPR，
  分配粒度为 16 个 VGPR。
- WGP 内每个 SIMD-pair 共享一条 WGP$/LDS 总线，并配有一个 TDM；因此实现层面通常是每 WGP
  两个 TDM 实例，而不是按 128B/256B 各放一个 TDM/FIFO。
- 16-bit 及更小数据类型的 WMMA 在独立 XDL pipeline 执行，可与部分 core/side/trans 指令
  co-execute。

公开 whitepaper 用“每 WGP 一个 TDM unit”的高层表述，内部 Shader/TDM 架构资料则明确为
per-SIMD-pair、当前配置两个实例。本文在讨论调度和资源时采用后者；两者可理解为 block-level
与 implementation-instance 两种粒度，而不是 128B/256B 两种 TDM。

FMA 按 2 operations 计数。MI450-B0 的理论峰值可由 WMMA 形状直接推出：

```text
FP8/BF8:
  16×16×128 WMMA = 2×16×16×128 = 65,536 ops/instruction
  8 cycles/SIMD，4 SIMD/WGP
  65,536 / 8 × 4 = 32,768 ops/clk/WGP

all-FP4:
  16×16×128 WMMA = 65,536 ops/instruction
  4 cycles/SIMD，4 SIMD/WGP
  65,536 / 4 × 4 = 65,536 ops/clk/WGP

或：
  32×16×128 WMMA = 131,072 ops/instruction
  8 cycles/SIMD，4 SIMD/WGP
  131,072 / 8 × 4 = 65,536 ops/clk/WGP
```

因此分享中的 `FP8 32K ops/clk/WGP` 和 `FP4 64K ops/clk/WGP` 与硬件资料一致。

需要避免误读 Shader Programming Guide §5.6 的 export-fuse 计算表。该表在 “Reduced Rate VALU”
章节列出 F8 `8192`、F4 `16384`，但同一章节明确说明这些 fuses 不存在于 MI450/XCD-ML，且
block-scale 的 `ops×bits` 还带出口管制权重；它不是 MI450-B0 四 SIMD 并行后的产品 raw peak。
32K/64K 可同时由 WMMA 形状与周期、四 SIMD，以及 whitepaper 的
`20.13/40.26 PF @ 256 WGP, 2.4GHz` 相互验证。

### 2.2 产品级峰值与 roofline 转折点

公开 CDNA5 whitepaper 给出的 MI455X 峰值为：

- OCP MXFP8 / FP8：约 20.13 PFLOP/s。
- OCP MXFP4：约 40.26 PFLOP/s。
- HBM4：最高 23.3 TB/s。

分享中的 22 TB/s 是更早期 MI450-B0 口径或取整值，不应替代目标机器上的实测 sustained bandwidth。

用公开峰值粗略计算理想 HBM roofline 转折点：

```text
MXFP8 ridge point = 20.13 PF/s / 23.3 TB/s ≈ 864 FLOP/byte
MXFP4 ridge point = 40.26 PF/s / 23.3 TB/s ≈ 1,728 FLOP/byte
```

这只是产品级上界。真实判断应使用：

```text
ridge_runtime = measured_sustained_compute / measured_sustained_HBM_BW
```

其中 compute peak 必须按 kernel 运行期间的平均 gfx clock 修正，而不能只用标称时钟。

### 2.3 内存层次

对 GEMM 最实用的简化路径是：

```text
HBM4
  ↕
GL2：192MB/GPU，跨 FCD 的共享读写 cache
  ↕
GL1：每个 SE 的分布式 buffer/fabric，不是普通命中型 cache
  ↕
TCP / WGP$：每 WGP 的 vector cache
  ↕
LDS 或 VGPR
  ↕
XDL
```

需要特别纠正两个常见误解：

1. GL1 在该架构中主要是 buffer/fabric，不能把两个相同请求简单理解成“第二个会 GL1 cache hit”。
2. TDM direct copy 不会绕过 GL1/GL2 路由。它绕过的是 WGP$ 的数据 staging/cache-line allocation：
   返回数据经 TX/TCP 控制路径直接写入 LDS。Indirect copy 则先写入 WGP$，再从 WGP$ 读出并写入 LDS。

### 2.4 WGP$、LDS 与 I$

- WGP$ 与 LDS 共用 384KB SRAM，以 64KB 为分区粒度。
- 至少保留 64KB WGP$，因此单 workgroup 最多使用 320KB LDS。
- LDS 有 64 个 32-bit bank；地址、payload 和跨 segment 访问会影响冲突。
- 每 WGP 有 64KB instruction cache。
- 每 SIMD 另有 4KB Instruction Store，用于从 WGP I$ 预取并供给 waves。

所以“4-stage、每 stage 64KB，总计 256KB”在容量上可行，但 occupancy 不只由 LDS 决定，还受
VGPR、barrier、wave 数和 cache/LDS partition 共同约束。

### 2.5 TDM：descriptor、direct copy 与 indirect copy

TDM 指令用 SGPR 中的 descriptor 描述最高 5D tensor/tile；数据不需要先进入 VGPR。一个 TDM
指令不是一个 64KB 的总线 transaction，而是由硬件拆成一系列不超过 256B 的 async-copy 请求。

对同一份没有 cache reuse 的数据，direct copy 并不会减少 HBM→GL2→GL1 的字节数；它减少的是
WGP$ allocation/staging 以及 WGP$→LDS 的额外内部搬运。真正能减少上游独特字节数的是 tile reuse、
cache hit、cluster multicast、压缩存储或 kernel fusion。

TDM direct-copy 的关键条件包括：

- 当前 fetch 为 128B 或 256B。
- global address 至少 128B 对齐；为了稳定生成 256B direct request，实际应按 256B 边界组织。
- 第一维连续字节数为 128B，或大于等于 256B。
- LDS 地址至少 4B 对齐。
- padding interval 至少 128B，且最好按 128B 粒度。

资料存在一处粒度差异：Shader Programming Guide 把 global alignment 概括为“至少 128B”，而更细的
Tensor DMA/TX 示例明确把从 `address % 256 == 128` 发出的 256B 请求标成 non-direct，并通过先发
128B 来对齐后续 256B。实际优化应遵循更细的 TX 规则：128B request 按 128B 对齐，256B request
按 256B 对齐。

这里的条件是“第一维连续**字节数**”，不是数学 GEMM 的 `tile_k` 元素数：

```text
FP8:  K=128 elements → 128B/row
FP4:  K=256 elements → 128B/row（假设 2 个 FP4/byte）
```

若数据预排布能跨行或跨块形成连续 256B，TDM 可以发 256B 请求。反之，单纯把数学 `tile_k`
写成 256 并不能保证 256B 请求。

TDM tracking 限制：

- 每 wave 最多 3 个 tensor op 处于 issue-to-XACK 阶段。
- 每 SIMD 最多 6 个。
- `TENSORcnt` 为 6 bit，issue-to-completion 最多跟踪 63。
- 每 WGP 的 direct-copy tracking 可表示约 512 个 128B 或 256B copy；每个不超过 256B 的 TDM
  copy 消耗一个 tracking entry。
- 256B copy 因而能在相同 tracking 数下维持约两倍于 128B copy 的在途字节数。
- direct tag 不可用时，硬件首先 fallback 到 indirect copy，而不是立刻停止 issue；当 indirect 路径的
  WGP$、LFIFO、slot 等资源也饱和后才会进一步 backpressure。

以 512 个 tracking entry 粗略估算，全部为 256B 时可覆盖约 128KB direct-copy payload。若 WGP$
配置为 64KB，分享中所说的“约 128KB direct + 64KB cache = 192KB”可作为容量直觉，但不是一个
保证可用的、单一硬件 FIFO 上限。

### 2.6 128B、256B 与混流

硬件资料没有“128B FIFO 和 256B FIFO 各一条”的结构说明。两种尺寸在同一请求、tracking 和
per-port LFIFO/OFIFO 体系中处理。

已知事实是：

- WGP vector cache line 为 128B。
- 普通 256B 请求需要同一 vector instruction 的两个 128B line 在同一 request cycle miss，且无
  slot、fault、scope/temporal 等冲突，才能合成一个 256B GL1 请求。
- direct copy 不使用上述 slot combining；direct-copy 数据格式本身决定是 128B 还是 256B。
- MI450 ECR 中记录了 128B/256B 混合以及 direct/indirect 混合的特定性能问题，详见第 9 节。

不要使用“永远禁止 128B+256B 混合”的规则。TDM 规格明确要求在某些 128B-offset 场景先发
128B prefix，再发 256B-aligned body，必要时再发 128B tail；这通常优于从非 256B 对齐地址强行发
256B 而导致 indirect/split。

### 2.7 Cluster 与 multicast

- 一个 cluster 最多包含 16 个 workgroup，全部调度到同一个 SE，每个 workgroup 位于独立 WGP。
- cluster 可为 1D、2D 或 3D。
- multicast mask 为 16 bit。
- MI400 GL1 的一次数据返回最多合并 5 个相同请求，最高约 5× return amplification。

因此，分享中“cluster 最多 5 个”的说法不准确。正确理解是：

- cluster 容量上限是 16 WG。
- 单次相同数据返回的合并上限是 5 个请求。
- `4×4` cluster 共 16 WG 是合法且常用的。
- 对 `4×4` output-tile cluster，A 通常沿 N 方向给 4 个 WG 共享，B 沿 M 方向给 4 个 WG 共享；
  每个实际 multicast group 是 4，正好低于 5 的合并上限。

所有参与者必须提交相同地址和一致 mask。请求到达不齐会 timeout，晚到请求可能退化成另一笔
multicast/unicast，降低放大率。

## 3. 先判断是哪一种 bound

### 3.1 完整 GEMM 字节模型

对 `C = A×B + C`：

```text
FLOPs = 2MNK

Bytes ≈ sA·MK + sB·KN
        + scale_A + scale_B
        + padding/alignment waste
        + C_read + C_write
        + split_K_partial + reduction
        + pre/post-processing traffic
```

`sA`、`sB` 是实际存储字节数，而不是 accumulator 类型。例如：

- A8W8：约 `sA=1, sB=1`，另加 microscale。
- A8W4：约 `sA=1, sB=0.5`，另加 scales 与 packing。
- A4W4：约 `sA=0.5, sB=0.5`，另加 scales 与 packing。

端到端 arithmetic intensity：

```text
AI = useful_FLOPs / actual_HBM_bytes
```

必须使用计数器中的实际 HBM/GL2 流量，而不是只用理想矩阵大小。Multicast、cache reuse、padding、
尾块、重放和 partial output 都会改变分母。

### 3.2 四类常见状态

#### Compute-bound

- XDL 有效利用率接近按实际频率修正后的峰值。
- HBM、GL2、GL1 均未接近可持续上限。
- 减少输入流量帮助有限；减少 XDL 空洞、指令数、hazard 和频率损失更重要。

#### HBM/GL2 bandwidth-bound

- 实际 HBM 或 GL2→EA 字节率接近平台可持续值。
- XDL 有大量无工作周期，但不是 VALU hazard 导致。
- 需要减少独特 A/B/C 字节、增加跨 tile 复用、multicast、融合或压缩存储。

#### On-chip feed-bound

- HBM 未满，但 GL1 return、TCP slot/LFIFO、direct-tag、LDS bank/segment 或 WGP issue 出现明显 stall。
- 常见原因是 128B 请求过多、direct/indirect 混流、LDS 冲突、multicast timeout 或 tile 形状使
  单 WGP 需求超过局部供给。

#### Latency/overhead-bound

- HBM 与 XDL 都不满。
- `M×N` 并行 tile 太少、stage/occupancy 不足，或 prologue/epilogue/launch 占比过大。
- Decode GEMM、小 batch、短 K、grouped small GEMM 常见。

### 3.3 不要只看一个 utilization counter

MI400 的 `SQ_INST_CYCLES_VALU_WMMA`、`SQ_INST_CYCLES_VALU` 和
`SQ_INST_CYCLES_VALU_COEXEC` 在生产 kernel 常启用的 `DISABLE_VALU_ARB_STALL=1` 下存在已知
计数缺陷：满载 XDL 可能只显示约 52%。

建议：

1. 已知单一数据类型时，用 WMMA FLOP/IOP perfmon 除以按实际频率计算的理论峰值推导 XDL utilization。
2. 用 thread trace/ATT 检查 WMMA issue 间距、`S_WAIT_*`、Idle/Stall。
3. 将 cycles、频率、FLOPs 和内存字节交叉验证。
4. 混合数据类型时，单靠 FLOP perfmon 也可能无法准确还原 XDL utilization。

## 4. Tile 的定量选择

### 4.1 单 WGP tile 的 arithmetic intensity

忽略输出与 scale 时，一个 WGP 完成 `Mt×Nt×Kt` 的输入成本为：

```text
FLOPs_tile = 2·Mt·Nt·Kt
Bytes_tile = sA·Mt·Kt + sB·Nt·Kt

AI_tile = 2·Mt·Nt / (sA·Mt + sB·Nt)
```

若 A、B 分别通过 multicast/cache 被复用 `rA`、`rB` 次，则上游独特流量近似为：

```text
Bytes_unique = sA·Mt·Kt/rA + sB·Nt·Kt/rB

AI_unique = 2·Mt·Nt /
            (sA·Mt/rA + sB·Nt/rB)
```

### 4.2 “方形最好”的适用条件

固定 `Mt·Nt=P`，最小化输入字节可得：

```text
Mt / Nt = (sB/rB) / (sA/rA)
        = sB·rA / (sA·rB)
```

只有当 `sA/rA == sB/rB` 时，方形 tile 才是带宽意义上的最优。

例：

- A8W8 且 A/B 复用相同：接近方形合理。
- A8W4 且复用相同：理想比值 `Mt/Nt≈0.5`，从纯输入字节看应偏向更宽的 N tile。
- Decode 中 A 可跨多个 N tile multicast、B 几乎不复用：应减少昂贵的独特 B tile 数，同时受实际
  `M` 很小和并行度需求约束，不能机械使用方形。

实际 tile 还必须满足 WMMA 形状、VGPR、LDS、cluster 方向、尾块和 wave mapping，公式只是候选生成器。

### 4.3 分享中的 256×256 与 128×512

对 FP8、`Kt=128`，两个 tile 的输出面积相同，均需 256 个 `16×16` WMMA：

```text
总 WMMA 执行量 = 256 instructions × 8 cycles
四个 SIMD 并行  = 256 × 8 / 4 = 512 cycles
```

分享 PPT 写了 512 cycles，但没有显式写出 `/4 SIMD`；会议现场已补充了这一点。

输入：

```text
256×256:
  A = 256×128×1B = 32KB
  B = 128×256×1B = 32KB
  total = 64KB
  demand = 64KB / 512 = 128B/clk/WGP

128×512:
  A = 128×128×1B = 16KB
  B = 128×512×1B = 64KB
  total = 80KB
  demand = 80KB / 512 = 160B/clk/WGP
```

所以 128×512 的局部输入需求高 25%。该推导正确，但未包含 scale、padding、重放和尾块。

### 4.4 GL1 96B/clk/WGP 的含义

MI450 一个 SE 有 16 个 active WGP，并连接 12 个 GL1C。GL1C read 目标为 128B/clk。若所有 WGP
同时竞争且公平平均：

```text
12 × 128B/clk / 16 WGP = 96B/clk/WGP
```

这能解释为什么单 WGP 需求 128B/clk 或 160B/clk 时，必须依赖 multicast、非均匀活跃度或其他复用。
但 96B/clk 是拓扑平均值，不是每个 WGP 的硬限额：

- 地址到 GL1C 的分布可能不均。
- 256B return 由相邻 128B channel 配对。
- 其他客户端、timeout、credit、harvest 和瞬时仲裁都会改变可用带宽。
- 并非所有 16 WGP 永远同时满负荷。

## 5. Compute-bound 优化方法

### 5.1 选择 WMMA 形状

MI450-B0 关键低精度 WMMA：

| Instruction / data | 执行周期 | Issue co-exec slot | 峰值特征 |
|---|---:|---:|---|
| `16×16×64 FP8/BF8` | 4 | 1 | 与 K=128 FP8 相同 ops/clk，短 K 适用 |
| `16×16×128 FP8/BF8` | 8 | 3 | 指令更少、更多 I slot |
| `16×16×128 F8F6F4`，两输入均 F4 | 4 | 1 | 64K ops/clk/WGP |
| `16×16×128 F8F6F4`，包含 F6/F8 | 8 | 3 | 32K ops/clk/WGP |
| `32×16×128 F4` | 8 | 3 | 与 16×16 all-F4 同峰值，更大 accumulator footprint |

选择规则：

1. 峰值相同且 K 足够时，优先考虑 I slot 更多、指令数更少的形状。
2. 更大 M 形状会增加 accumulator/VGPR footprint，可能降低 tile 或 occupancy。
3. 小 K、尾块或 VGPR 紧张时，短指令可能更合适。
4. scaled WMMA 的 LD_SCALE 与 WMMA 在硬件中是紧邻的隐式组合；scale 必须立即服务下一条 WMMA，
   调度时不能把它当作完全独立的普通 VALU。

### 5.2 填充 co-execution 空间

MI450-B0 的 `X/I` pattern 与分享 slide 5 一致：

```text
16×16×64 FP8/BF8:       X X X I
16×16×128 FP8/BF8:      X X X I X X I I
16×16×128 all-F4:       X X X I
32×16×128 F4:           X X X I X I X I
```

- `I` 周期允许开始一条可 co-execute 的新指令。
- `X` 周期只能让已经开始的 multicycle 指令继续执行，并且该周期不能再次读取 VGPR。
- SALU、VMEM、LDS 有各自 issue 资源，但仍会受到源寄存器、总线、scoreboard 和依赖限制。

适合填充的工作：

- 与当前 accumulator、A/B source 无重叠的地址计算或转换。
- 下一 stage 的 LDS/VMEM 操作。
- 独立 permute、scale 准备、loop-control。
- 可证明无 RAW/WAR/WAW 的 normal/trans 指令。

硬件不能检测所有 XDL co-execution 数据冒险。对相关 VGPR：

- 插入足够数量的独立 VALU 或 `V_NOP`。
- 不要把 `S_DELAY_ALU` 的存在等同于所有 same-wave co-exec hazard 都已解决。
- 使用 ATT 验证实际 issue 间距和 Stall/Idle，而不是只凭静态排程。

### 5.3 Tile 与 occupancy

Compute-bound 大矩阵通常可从“一 WGP 一 WG、4 waves、occupancy=1”的大 tile 开始：

- `256×256`：对称、复用高、便于 4×4 cluster。
- `128×512`：相同输出面积，但输入流量更高；只在 shape、wave mapping 或 N 方向并行性有明确收益时使用。

必须同时满足：

```text
VGPR_per_SIMD ≥ sum(resident waves' VGPR allocation)
LDS_per_WG × resident_WG ≤ allocated_LDS
barrier / wave / descriptor resources are sufficient
grid_tiles ≥ active_WGPs needed to fill GPU
```

如果全 GPU 没有足够 tile，occupancy=1 的大 tile 反而会降低并行度。边界 tile 利用率低时也应准备较小
或不同 aspect ratio 的 kernel。

### 5.4 Pre-shuffle 策略

#### Weight

长期复用的 weight 最适合离线或在前置 quant/pack kernel 中完成 pre-shuffle：

- 匹配 WMMA lane/VGPR layout。
- 避免 LDS bank conflict。
- 减少或消除 LDS padding。
- 更容易形成对齐 256B TDM direct-copy 请求。

#### Scale

Microscale 数量小但布局离散。把多个 scale block 打包到连续 256B 区域通常有利于：

- 避免大量 B8/B32 细粒度读取。
- 简化 LD_SCALE 前的数据准备。
- 提高 direct-tag 的 payload/entry。

“scale 必须 shuffle”应理解为当前高性能 kernel/layout 的工程选择，而不是所有合法 WMMA 的 ISA
功能要求。

#### Activation

Activation 通常在线生成：

- 不 shuffle：可在 LDS 中每 128B interval pad 16B 等方式消除 bank conflict。
- shuffle：可能提高请求宽度并去掉 padding，但必须支付在线转换、额外读写和生产者复杂度。

Compute-bound 分享中，A pre-shuffle 曾带来约 10% silicon 差异；这是特定 kernel/shape 观察，不是
通用硬件常数。

#### “Always 256B”的修正

Pre-shuffle 只能让布局具备 256B 条件。最终是否为 256B direct request 还取决于：

- global base/stride 是否 256B 对齐。
- TDM 展开模式和 tile 边界。
- padding 是否保持 128B 粒度。
- direct tracking 是否耗尽。
- 已知 hardware errata 是否触发。

### 5.5 Cluster multicast

Compute-bound 大 GEMM 的常见 `4×4` cluster：

```text
WG(m,n):
  A tile 随 m 变化、沿 n 方向共享 → multicast 到同一行的 4 个 WG
  B tile 随 n 变化、沿 m 方向共享 → multicast 到同一列的 4 个 WG
```

收益：

- A/B 各自的上游独特流量理论上约缩小 4×。
- 可将 FP8 `256×256` 的理想本地 `AI=256 FLOP/B` 放大到上游约 `1024 FLOP/B`，
  接近并超过公开 FP8 HBM ridge point。
- 对 A4W4，相同 tile 的理想本地 `AI≈512 FLOP/B`，4× 复用后约 `2048 FLOP/B`，
  才有机会支撑更高的 FP4 compute roof。

要求：

- 参与 WG 使用一致地址、mask 和 cluster ID。
- 对应 waves 的请求时序不能相差过大，否则 timeout/downgrade。
- cluster launch 必须一次性找到足够 WGP 资源，过大 cluster 会降低调度灵活性。

### 5.6 带宽预算

对每个 stage，至少计算：

```text
B_local = A_bytes + B_bytes + scale_bytes + padding_bytes
C_stage = 该 stage 的实际 XDL compute cycles
BW_WGP_required = B_local / C_stage

B_unique_upstream = A_bytes/rA + B_bytes/rB + ...
BW_GL1_GL2_HBM_required = B_unique_upstream / C_stage × active_WGPs
```

同时检查：

- TDM request 是否主要为 256B direct。
- WGP$ indirect copy 是否消耗额外 read/write port。
- LDS bank/segment conflict 是否使 512B/cycle 理论 SRAM 带宽失效。
- GL1 multicast 是否 timeout。
- GL2/HBM channel 是否均衡。

分享中的 `(128+256)/2=192B` 只适用于 A/B 字节数和请求数大致相等、A 为 128B、B 为 256B
的特定情形。一般应使用加权平均：

```text
average_payload_per_request =
  total_payload_bytes / total_request_count
```

对 `128×512` 等非对称 tile，简单算术平均会误导。

### 5.7 多级流水

若 TDM 在 stage `i` issue，在 `S-1` 个 stage 后消费：

```text
(S - 1) × C_stage >= L_effective
S >= ceil(L_effective / C_stage) + 1
```

分享示例：

```text
L_effective = 1000 cycles（实测假设）
C_stage = 512 cycles
S_min = ceil(1000/512)+1 = 3

S=4:
issue-to-consume distance = 3×512 = 1536 cycles
```

公式正确，但 `1000 cycles` 不是硬件常数。它随 GL2/HBM hit/miss、page translation、cluster timeout、
争用和频率变化。建议用目标 workload 的 P90/P95 latency 或直接观察 `S_WAIT_TENSORCNT` stall，
而不是只用均值。

stage 上限至少同时受以下条件限制：

```text
S × LDS_bytes_per_stage + epilogue_workspace + barriers ≤ LDS allocation
TDM descriptors / direct tags / LFIFO / cache capacity sufficient
VGPR allocation permits desired residency
```

### 5.8 Wave 分工与 wait

四 wave WG 可把 A、B、scale 或不同 stage 的 TDM issue 分散到不同 waves，以利用 per-wave
3-descriptor 限制，但必须注意：

- per-SIMD 仍有 6 个 issue-to-XACK 限制。
- 两个 SIMD-pair/TDM 的请求要尽量均衡。
- 同 wave 的 TDM load/store 共用 `TENSORcnt`。
- 跨 wave 消费 LDS 数据必须通过正确 barrier/atomic 协调；不能把另一个 wave 的计数器当作本 wave
  已完成的证明。

### 5.9 Prologue、persistent 与 I$

可行策略：

- prime `S-1` 个输入 stage 后进入 steady loop。
- persistent kernel 中，把下一 tile/round 的 prologue 移到当前 tile main loop 尾部。
- 若尾 tile 无效，利用安全 OOB/descriptor 逻辑，但不能依赖未证明的“无效 TDM 一定免费”。
- 缩短热点代码、复用循环体，避免大量复制展开超过 64KB WGP I$。
- 必要时使用 instruction prefetch，但仍要保持控制流和热点 footprint 可预测。

分享 slide 12 的 `10241-cycle kernel ramp-up` 不能当作当前硬件固定开销。会议后续明确指出该段来自
当时 trace/trap-handler 采集方式，当前环境已经看不到同样开销。`~3000-cycle prologue/epilogue`
也只是当时 kernel 的实测量级。

### 5.10 Epilogue

WMMA D-register layout 往往不能直接生成完整连续 cache line。三类路径：

1. LDS permutation → TDM store
   - 使用 LDS。
   - load/store TDM 都进入 `TENSORcnt`，等待 store 可能连带等待更早的 load。
2. LDS permutation → async store
   - 使用 LDS。
   - 使用 `ASYNCcnt`，可与 TDM load 的 `TENSORcnt` 分开等待。
3. VGPR permutation / clauses → `buffer_store`
   - 不占 LDS。
   - 使用 `STOREcnt`。

选择依据：

- LDS 是否已接近 320KB。
- 是否需要单独等待输出 store。
- VGPR permute 与 clause 是否能在 WMMA 空隙中隐藏。
- 最终 stores 是否形成连续 128B/256B 请求。
- 是否能融合 bias、activation、quantization，避免额外 kernel。

### 5.11 功耗与频率

硬件有 Predictive Instruction Throttling（PIT）和 di/dt 缓解逻辑，会根据指令功耗历史对 SP
进行 stall/throttle。数据位翻转也确实影响动态功耗。

因此：

- 静态/全零数据用于功能和理想 issue-rate 检查，不可作为产品性能结论。
- 随机或真实模型数据会提高 switching activity，可能降低 sustained clock。
- 分享中的 `1.7→1.2GHz`、约 `-30%` 是特定样机实测，不是保证值。
- 比较优化时同时报告：
  - kernel cycles；
  - 平均/分位 gfx clock；
  - PFLOP/s；
  - 功耗；
  - 输入数据分布与初始化方式。
- 使用 A/B reuse hint 可降低不必要的矩阵源读取和功耗；是否提高最终 PFLOP/s必须实测。

## 6. Memory-bound 优化方法

### 6.1 先区分 bandwidth-bound 与 latency-bound

两者都可能表现为 XDL idle：

- bandwidth-bound：增加 stage/occupancy 不再提升，HBM 或某级 bytes/clk 已饱和。
- latency-bound：HBM 未满，增加独立 WG、stage、batch 或 outstanding requests 后性能继续上升。

调优顺序：

1. 固定真实数据和时钟统计方式。
2. 测 HBM/GL2 bytes、GL1/TCP stall、TDM request、XDL useful FLOPs。
3. 扫 occupancy、stage、batch/group 数。
4. 若带宽已饱和，转向减字节；若未饱和，转向增加并发和改善请求形态。

### 6.2 Tile：不要强制 occupancy=1

Memory-bound 常见最优点可能是：

- 更小 LDS stage。
- 2 个或更多 resident WG/WGP。
- 更少 accumulator VGPR。
- 更窄 M、较宽 N 的 decode 专用 tile。

大 tile 只有在其复用收益大于以下代价时才有利：

- 并行 tile 数下降。
- 边界浪费。
- VGPR/LDS 使 occupancy 降低。
- cluster 一次性资源要求变高。
- 首个输出延迟上升。

建议至少生成/测试以下候选族：

```text
Compute-oriented: 256×256, 128×512
Balanced:          128×256, 256×128
Decode/small-M:     16/32/64 × 256/512
Tail kernels:       smaller M/N without large masked tiles
```

### 6.3 请求粒度与对齐

普通 wave32 `GLOBAL_LOAD/STORE`：

```text
B32: 32 lanes × 4B = 128B logical request
B64: 32 lanes × 8B = 256B logical request
B128:32 lanes ×16B = 512B，理想为两个 256B request
B96: 32 lanes ×12B = 384B，容易形成 256B+128B 或多个 128B
```

要求 lane 地址连续，128B 请求起址按 128B 对齐，256B 请求按 256B 对齐。不同 wave/instruction
不会合并为一个请求。

TDM/direct-copy 的内部计算不同：每个 copy cycle 使用 16 个地址，B64 通常形成 128B，B128
形成 256B。TDM 指令本身不应按“4 waves×32 threads”计算 transaction 大小。

物理数据 beat 与逻辑 request 也不同：

- TCP/GL1 write source path 为 128B 级数据传输。
- GL2 client source FIFO 为 128B 宽。
- 一个逻辑 256B store 会使用两个 128B data beat。

因此分析计数器时必须区分 request-size counter 和 data-beat counter。

### 6.4 Direct copy 是否值得

适合 TDM→LDS：

- 数据会被多个 waves/多次 WMMA 复用。
- descriptor 能生成对齐 128/256B 请求。
- LDS layout/padding 能减少消费端冲突。

不一定适合：

- 数据只用一次且直接进入 VGPR 更简单。
- tile 很小，descriptor setup/barrier 占比较大。
- 需要大量不规则 gather、尾块或细粒度 mask。
- direct tracking 经常耗尽，主要走 indirect。

Indirect copy 会使用 WGP$ 作为 staging，并额外消耗 cache/LDS 读写端口和 buffering。硬件规格明确
指出该路径可能无法达到 back-to-back full throughput。

因此，如果计数器显示 HBM 已经完全饱和而 TCP/LDS 仍有余量，把 indirect 全部改成 direct 可能只带来
小幅收益；如果瓶颈在 WGP$ 容量、internal ports 或 latency，direct 的收益才会明显放大。

### 6.5 增加有效在途量

按如下顺序增加并发：

1. 将 128B 请求重排为对齐 256B direct 请求，提高每 tracking entry 的 payload。
2. 增加独立 TDM descriptor，但遵守每 wave 3、每 SIMD 6。
3. 增加 stage，直到 wait stall 消失或资源开始反噬。
4. 增加 resident WG，利用另一 WG 隐藏当前 WG 的 memory round trip。
5. 对 grouped/decode workload 使用 persistent work queue，确保 256 个 WGP 持续有 tile。

不要把“更多 bytes issued”当作有效在途量。若它们集中到同一 GL1/GL2 channel、触发 slot thrash、
timeout 或 indirect fallback，只会增加排队。

### 6.6 Memory-bound 下的 multicast

根据真实共享关系设 mask：

- Dense training/大 M、N：A 沿 N 共享，B 沿 M 共享。
- Decode/small-M：activation A 常能跨多个 N tile 共享；不同 N tile 的 weight B 不同。
- MoE/grouped：不同 expert 的 weight 不共享，跨 expert multicast 通常无意义。

对 decode，常见 cluster 是 `1×Ncluster`：

- 只 multicast activation。
- weight 保持 unicast/stream。
- `Ncluster` 通常从 2～4 开始；大于 5 不会得到单次返回更高的 GL1 merge amplification，
  但 cluster 本身仍可合法大于 5。

若 multicast timeout/downgrade 高，先同步请求到达、缩小 group 或调整 wave 分工，不要只增大 cluster。

### 6.7 Memory-bound 下的 pre-shuffle

#### Weight/scale

若 weight 会被许多 token/batch 重复使用，离线 pre-shuffle 通常值得：

- producer 成本被多次 GEMM 摊薄；
- 256B direct request 提高在途字节；
- 消费端 LDS/VGPR layout 更简单。

#### Activation

activation shuffle 通常在线发生，应比较：

```text
saved_GEMM_cycles × reuse_count
    vs.
shuffle_read + shuffle_write + launch/sync + extra_storage
```

分享中“memory-bound 加 cluster 后，A shuffle 与否约 1～2%”是一次 silicon 观察，且现场明确提醒
不要过早泛化。它说明某些 memory-bound case 已由 HBM/其他流量主导，并不说明 activation shuffle
永远无用。

### 6.8 Split-K / stream-K

使用条件：

- `M×N` tile 数不足以占满 GPU。
- K 很大，额外 K partition 能显著增加并行度。

成本：

- partial C 写回。
- reduction kernel 或 atomic。
- 更多 output traffic 与同步。
- accumulator 转换和数值误差风险。

判据：

```text
saved_idle_time >
partial_write_time + reduction_time + extra_launch/sync
```

Compute-bound 大 M/N 通常不需要 split-K；decode/small batch 更可能受益。

### 6.9 融合和输出

Memory-bound 最有价值的优化往往不是主循环指令调度，而是减少完整 tensor 往返：

- fused dequant / scale。
- fused bias / activation。
- fused residual。
- 直接写最终精度和最终 layout。
- grouped GEMM 内共享 metadata 与调度。

输出 store 应：

- 先把 WMMA D layout 变成连续 cache-line layout。
- 优先完整 128B/256B 写，避免大量 partial line。
- 在 LDS permutation 与 VGPR permutation/clauses 之间按 LDS 容量和 co-exec 空间选择。

## 7. 128B/256B 的专项建议

### 7.1 想要纯 128B 普通 VMEM 流

每 wave 使用全 EXEC 的 `GLOBAL_*_B32`，每 lane 连续 4B：

```text
addr(wave,lane) = aligned128_base + wave_id×128 + lane_id×4
```

4-wave WG 产生 4 个独立 128B logical requests，共 512B；不会跨 wave 合成 256B。

### 7.2 想要纯 256B 普通 VMEM 流

每 wave 使用 `GLOBAL_*_B64`，每 lane 连续 8B：

```text
addr(wave,lane) = aligned256_base + wave_id×256 + lane_id×8
```

4-wave WG 理想产生 4 个 256B logical requests，共 1024B。

但要保持 GL1→GL2 也为 256B，需要两个 128B WGP$ line 同时 miss、无 slot/fault/policy 冲突。若一半
hit、一半 miss，下游仍会看到 128B。

### 7.3 TDM 流

- 目标 128B：让展开后的 direct cycle 使用 16×8B。
- 目标 256B：让展开后的 direct cycle 使用 16×16B，并保证 256B 对齐。
- 大 tile 可由连续多个 256B cycle 组成。
- 128B-aligned 但非 256B-aligned 的行，优先 128B peel + 256B body，而不是强行从错误边界发 256B。

### 7.4 已知混流风险的规避顺序

1. 先避免 direct 128B 与 indirect 256B 混合。
2. 再避免同一高压 GL2 RW-write 流中频繁交错 128B/256B。
3. 若必须有 128B prefix/tail，保持其 direct，并让 256B body 对齐。
4. 可尝试把不同尺寸分 phase，但把它当 silicon workaround，必须在目标 stepping 实测。

## 8. Prologue / Epilogue / Persistent 的端到端取舍

### 8.1 Fixed-cost 的正确计算

分享 slide 12 的 “fixed-cost share” 实际使用的是 `overhead/main_loop`：

```text
K=16K: 16K / 65,536 ≈ 25%
K=1K : 16K / 4,096  ≈ 400%
```

若“share”定义为总时间占比，则应为：

```text
K=16K: 16K / (65,536+16K) ≈ 19.6%
K=1K : 16K / (4,096+16K)  ≈ 79.6%
```

写报告时应明确分母。

### 8.2 Persistent 适用情形

适合：

- 大量 homogeneous tiles。
- grouped/decode 小 GEMM，可由 work queue 持续喂给所有 WGP。
- 需要把下一 tile prologue 与当前 tile main loop overlap。

风险：

- 动态 work queue/atomic 开销。
- cluster/persistent 组合降低调度弹性。
- 长时间占用全部 WGP，影响系统公平性。
- 代码膨胀和 I$ 压力。
- 尾块、不同 shape 和不同 layout 分支使热点 loop 失去规律。

## 9. 分享关键信息核查

| 分享主张 | 判定 | 修正/条件 |
|---|---|---|
| FP8 32K、FP4 64K ops/clk/WGP | 正确 | 可由 WMMA 形状、周期和 4 SIMD 推导 |
| HBM 22TB/s | 旧口径/近似 | 公开 MI455X whitepaper 为最高 23.3TB/s；调优应使用实测 sustained BW |
| GL1 是 buffer | 正确 | 它不是普通命中型 L1 cache |
| TDM “一次 load 64KB request” | 表述不严谨 | 一个 descriptor 可描述大 tile，但内部拆为多笔不超过 256B 的 async copy |
| Direct copy 绕过 cache | 条件正确 | 绕过 WGP$ 数据 staging/line allocation，仍经过 GL1/GL2 和 TX/TCP 控制路由 |
| Direct copy 要求 K 为 128/256 | 不严谨 | 条件是第一维连续字节数为 128B 或 ≥256B；数学 K 需按数据位宽换算 |
| Pre-shuffle 后 always 256B | 过于绝对 | 还需 256B 对齐、合适展开、tracking 可用且不触发 errata |
| Cluster 最多 5 个 WG | 错误 | cluster 最多 16 WG；GL1 单次返回最多合并 5 个相同请求 |
| 4×4 cluster | 正确 | 16 WG 合法；A/B 常分别按 4-WG 行/列 multicast |
| Co-execution `X/I` 表 | 正确 | 与 MI450-B0 Shader Programming Guide 一致 |
| 通常一 WGP 一 WG | Compute-bound 条件成立 | memory/latency-bound 可能需要多个 resident WG |
| Tile 越大越好 | 条件成立 | 受并行度、尾块、occupancy、LDS/VGPR 和 latency 影响 |
| 方形 tile 最好 | 仅对称成本下正确 | 一般最优比为 `Mt/Nt=sB·rA/(sA·rB)` |
| 256×256 需要 128B/clk | 正确 | FP8 K=128、忽略 scale/padding；512 cycles 必须包含 `/4 SIMD` |
| 128×512 需要 160B/clk | 正确 | 同上，输入比 256×256 多 25% |
| GL1 平均只有 96B/clk/WGP | 合理拓扑上界 | `12×128/16`，不是每 WGP 固定硬限额 |
| stage 下界公式 | 正确的保守模型 | latency=1000 是实测假设，不是规格 |
| 384KB shared、最多 320KB LDS | 正确 | 至少保留 64KB WGP$，分区粒度 64KB |
| 128KB direct + 64KB cache≈192KB | 可作直觉 | 由 512×256B tracking 与 64KB WGP$ 推导，不是单一 FIFO 的保证上限 |
| code 保持 64KB 内即可无 I$ miss | 方向正确但不保证 | 还受映射、控制流、4KB/SIMD IS 和预取影响 |
| kernel ramp-up 10241 cycles | 已被现场纠正 | 来自旧 trace/trap-handler 采集，当前环境不应作为固定开销 |
| Compute-bound A pre-shuffle 约 +10% | 实测观察 | 仅适用于分享中的 kernel/机器/数据 |
| Memory-bound 加 cluster 后 shuffle 差 1～2% | 实测观察 | 现场明确建议继续测试，不可泛化 |
| 随机输入造成约 30% 频率差 | 机制合理、数字为实测 | 数据翻转影响功耗，PIT 可 throttle；幅度不是固定规格 |

### 9.1 与 128B/256B 相关的 ECR

- `DEGFXMI400-12263`（GL2）：RW write memory bandwidth 因混合 256B 和 128B 请求下降；
  ECR summary 中两个版本栏均为 `No Fix`。
- `DEGFXMI400-12257`（GL1）：direct 128B 与 indirect-copy GL2-hit 256B load case 记录为约
  75% performance；早期版本 `No Fix`，后续版本有 RTL fix。
- `DEGFXMI400-12753`（VMEM/TDM）：FP6 128×128 tile 出现意外 indirect copy；两个版本栏均
  `No Fix`。

这些 ECR 支持“特定混流会掉性能”，但不支持“TDM 有按 128B/256B 划分的两条 FIFO”这一解释。

## 10. Profiling 与验证

### 10.1 建议流程

1. 用真实 shape、真实随机/模型数据、固定软件栈做基线。
2. 记录：
   - kernel wall time；
   - shader cycles；
   - 平均 gfx clock；
   - useful FLOPs；
   - HBM/GL2/GL1/TCP bytes；
   - request 128B/256B 分布；
   - direct/indirect 和 multicast 状态。
3. 先用 `rocprofv3 --kernel-trace --stats` 确认准确 kernel symbol 和资源占用。
4. 查询生成的数据库，核实 SGPR/VGPR/LDS、dispatch 次数和目标 kernel。
5. 再对代表性较小 shape 做 ATT/thread trace，检查主循环 issue/stall；大型真实 shape 用于最终性能。
6. 每次只改一个维度：tile、WMMA、stage、shuffle、cluster、epilogue 或 persistent。

### 10.2 计算侧

- `SQ_VALU_WMMA_FLOP_FP8/FP6/FP4/...`
- `SQ_INSTS_VEC32_VALU_WMMA`
- `SQ_INSTS_VALU_COEXEC`
- `SQ_INST_ISSUE_VALU_STALL`
- `SQ_INST_ISSUE_TDM_STALL`
- ATT 中 WMMA issue 间距、Latency、Stall、Idle

注意第 3.3 节的 XDL cycle perfmon 缺陷。

### 10.3 TX/TCP/TDM

- `TX_PERF_SEL_VMW_GL1_READ_REQ_256B`
- `TX_PERF_SEL_VMW_GL1_WRITE_REQ_256B`
- 对应 128B request counters
- `TX_PERF_SEL_VCA_ASYNC_DIRECT_MAX_REQUEST`
- `TX_PERF_SEL_VCA_TDM_LOAD_BANDWIDTH_BYTES`
- `TX_PERF_SEL_VCA_TDM_STORE_BANDWIDTH_BYTES`
- `TX_PERF_SEL_VMW_LFIFO_STALL`
- `TX_PERF_SEL_VMW_SLOT_STALL_*`
- LDS bank/segment conflict stalls

关键比率：

```text
request_256_ratio = req_256 / (req_128 + req_256)
direct_fallback_rate = ASYNC_DIRECT_MAX_REQUEST / async_copy_requests
payload_per_tdm_req = TDM_bytes / TDM_request_cycles
```

### 10.4 GL1 multicast

- `GL1C_REQ_MULTICAST_128B`
- `GL1C_REQ_MULTICAST_256B`
- `GL1C_MULTICAST_TRACKER_DEPLETS`
- `GL1C_REQ_MULTICAST_TIMEOUT*`
- `GL1C_REQ_MULTICAST_DOWNGRADE`
- `GL1C_STALL_GL1_GL2_REQ_CREDIT`
- `GL1C_STALL_GL1_GL2_DATA_CREDIT`

不要只看发出了 multicast 指令；要确认 timeout/downgrade 足够低且实际 upstream bytes 下降。

### 10.5 GL2/HBM

- GL2 lookup hit/miss rate。
- GL2 read/write request bytes。
- `GL2C_EA_WRREQ_STALL`
- `GL2C_TOO_MANY_EA_WRREQS_STALL`
- 各 EA VC request/data stall。
- HBM channel bytes 与分布。

## 11. 推荐的调优顺序

### 11.1 Compute-bound

1. 验证 WMMA 指令、ops/cycle 与真实平均频率。
2. 修正 WMMA hazard，填充合法 co-exec slot。
3. 扫 `tile_m/tile_n`，保证 GPU tile 数足够。
4. 按字节和复用选择 aspect ratio，而不是只看方形。
5. 确保 weight/scale 请求主要为 256B direct。
6. 启用并验证 A/B multicast group。
7. 从 3 stage 起扫到 4 stage，观察 wait stall、LDS、direct fallback。
8. 优化 prologue/epilogue 和代码 footprint。
9. 用真实随机数据重新检查 cycles、clock、power。

### 11.2 Memory-bound

1. 判断 HBM bound 还是 latency/on-chip bound。
2. 用小 tile/高 occupancy 和大 tile/高 reuse 两族做对照。
3. 只 multicast 真正共享的 operand。
4. weight/scale 优先离线 pre-shuffle；activation 做端到端收益评估。
5. 提高 256B direct 比例和 channel 均衡。
6. 扫 stage 与 resident WG，找到带宽平台点。
7. `M×N` 不足时测试 split-K/stream-K。
8. 融合 epilogue，减少 C 和中间 tensor 的 HBM 流量。
9. 对 grouped/decode 测 persistent scheduler。

## 12. 参考资料

分享材料：

- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\MI450\MI450-B0 GEMM Pipeline.pptx`
- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\MI450\MI450-B0 GEMM Pipeline.vtt`

主要硬件资料：

- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\MI450\amd-instinct-cdna5-instruction-set-architecture.txt`
- `C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt\MI450\amd-cdna5-whitepaper.txt`

以下省略路径均相对于：

`C:\Users\yanguahe\Documents\code\wk_sp1\mi400_hw_wiki\raw\papers\mi400_hd_txt`

- `...\architecture\subsystem\SH\MI400_Shader_Programming#65.txt`
  - §1.4/1.5 hardware overview；§2.3 clusters；§3.3 VGPR/LDS；§4.6.12 WMMA；
    §4.9.8～4.10 multicast/async/TDM；§5.3～5.7 scheduling/performance。
- `...\architecture\subsystem\TX\mi400_tensor_dma#72.txt`
  - §4 performance requirements；§7.2 configuration；§7.6 descriptor interleave、地址展开与
    128B/256B direct-copy 对齐优化。
- `...\architecture\subsystem\TX\mi400_merge_tcp_lds_cu#85.txt`
  - §3.2 latency hiding；§3.3 bandwidth、direct tracking、256B combining、indirect-copy 吞吐。
- `...\block\tx\mi400_tx_MAS#156.txt`
  - §9 performance；§10.5.4 async/direct copy；§10.6 TCP/LDS 实现与性能计数器。
- `...\subsystem\CMM\GLX\Design\MI400_GL1_CH_GLARB_Fabric_MAS#12.txt`
  - §2 request/return flow；§10.4 GL1 router、12 GL1C、128B/256B 返回。
- `...\subsystem\CMM\GL2\Design\MI400_GL2_MAS#8.txt`
  - §2.2.6 request sizes/fill；§7.2 bandwidth；§8.3 input/source/latency/output FIFO。
- `...\architecture\system\Performance\mi450_design_ecr_summary#3.txt`
  - MI450 stepping ECR。
- `...\architecture\system\PerformanceCounters\mi400_appnote_xdl_perfmon#2.txt`
  - XDL utilization perfmon 缺陷与 workaround。
- `...\architecture\system\PerformanceCounters\MI400_perfmon_spec\`
  - SQ、TX、GL1、GL2 指标定义。

