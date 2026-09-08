# gfx1250 MoE GEMM1 const0 cycle 上限分析

## 结论

分析对象：

```bash
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-const0
```

这里的“性能上限”按**完成一次 kernel dispatch 所需的理论最少
`GFXCLK` cycles** 表示。`perf-const0` 的 `--iters 20` 输出是多次 dispatch
的平均时间，下面所有 cycle 数均指单次 dispatch。

结论分为两层：

1. 只做标准 compute/memory roofline 时，主导项是 MXFP4 WMMA，理论最少为：

   ```text
   516,096 GFXCLK cycles
   ```

2. 对当前这份汇编进一步计入 320 KiB LDS 导致的单 WG residency，以及不能和相邻
   WG 的 WMMA 重叠的 prologue、SiLU/output tail 和无效 tail WG 后，可从现有资料和
   ATT dynamic instruction stream 得到更紧的乐观下限：

   ```text
   617,877 GFXCLK cycles/dispatch
   ```

   因而，对**当前 kernel**，建议把约 `618k GFXCLK cycles` 作为采用最大物理并行
   能力时的硬件/结构性能 ceiling；它在本次 ATT 加权平均
   `GFXCLK=2084.136 MHz` 下对应：

   ```text
   617,877 / 2084.136 MHz = 296.467 us
   ```

这个数仍是乐观下限：它假设 cache、TDM、barrier、指令取指和各 pipeline 都能达到
文档峰值，并忽略不可由吞吐公式确定的启动/排空 latency。硬件资料还注明 WGP mode
下 LDS 可能一次只能使用一个 SIMD-pair；如果该限制适用于这里，结构下限会提高到
`686,997 cycles`。文档没有把这个 “may be limited” 条件说明为 MI455X 的固定行为，
所以能无条件确定的是 `cycles >= 617,877`，真实可达到的最少 cycle 只会更大。

本次 ATT 实测为：

```text
kernel wall time                         = 551.160 us
occupancy shader_timestamp maximum       = 1,150,033 cycles
realtime.json gfx_clock maximum          = 1,149,752 cycles
derived GFXCLK from occupancy maximum    = 2086.568 MHz
weighted mean GFXCLK                     = 2084.136 MHz
```

以 `1,150,033 cycles` 为比较基准：

```text
纯 XDL roof 利用率          = 516,096 / 1,150,033 = 44.88%
当前 kernel 结构下限利用率 = 617,877 / 1,150,033 = 53.73%
```

普通 20-iteration profiler 的 `515.603 us` 与 ATT 不是同一次运行。若仅用 ATT 的
加权平均频率作近似换算，它约为 `1,074,587 cycles`，对应当前 kernel 结构下限的
`57.50%`。这个换算仅供对照，精确 cycle 应使用同一次 ATT capture 的
`shader_timestamp`。

## 1. 假设与边界

- 所有频率统一视为 `GFXCLK`，即令 `fG=fL=fE=fF=fU=fM`。
- HBM4 pins 因而按 `12,288 B/GFXCLK-cycle` 计算。
- `const0` 只改变输入值，不改变地址、tile 数、WMMA 数或 store 数。现有资料没有说明
  该路径存在 zero compression 或 zero-skip，因此不能因为输入全为零而减少理论工作量。
- 内存部分分别列出 ideal unique-byte、4x4 multicast source traffic 和每个 WG 的
  destination delivery traffic，避免把三个不同层级的 byte 数混为一谈。
- bandwidth、WMMA、LDS 和大部分普通 VALU 可以流水重叠。标准 roofline 对这些资源取
  `max(...)`，不能全部相加。
- 当前 kernel 使用完整 `327680 B` LDS allocation。MI400 每个 WGP 最多可分配
  `320 KiB` LDS，因此同一 WGP 不能同时 resident 两个该 kernel 的 WG。ATT 也观测到
  每个 physical SIMD 最多只有一个 active wave slot。

## 2. 硬件依据

本分析使用以下本地资料：

- `mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-cdna5-whitepaper.txt`
  - 第 6 页：MI455X 有 `8 XCD × 2 SE/XCD × 16 WGP/SE = 256 WGP`。
  - 第 22 页：`256` active WGP、`2400 MHz`、OCP MXFP4 `40.26 PFLOP/s`、
    HBM4 `23.3 TB/s`。
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`
  - 第 18 页：每个 WGP 有 `4 SIMD32`；每个 SIMD-pair 共享 LDS/WGP$ bus。
  - 第 10 页：WGP$ 共 `384 KiB`，其中最多 `320 KiB` 可作为 LDS；单个 WG
    最多分配 `320 KiB`。
  - 第 241–242 页：Wave32 `V_WMMA_F32_32x16x128_F4` 在 MI450/XCD-ML 上的
    independent/dependent repeat rate 都是 `8 cycles`；`SCALE16` variant 性能相同。
  - 第 229–231 页：Wave32 Trans32 repeat rate 为 `2 cycles`，WMMA、Trans 和
    core/side MACC 支持规定范围内的 co-execution。
  - 第 234–235 页：`DS_LOAD_B32` independent repeat rate 为 `1 cycle`，
    `DS_LOAD_B128` 为 `2 cycles`，`DS_STORE_B64` 为 `2 cycles`。
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt`
  - 第 464–467 页附近：`V_WMMA_SCALE_F32_32X16X128_F4` 执行
    `D(32x16) = A(32x128) * B(128x16) + C(32x16)`。
- `my_code/mi450_mi455_wg_to_hbm_bandwidth.md`
  - 全 GPU、统一按 `GFXCLK` 后的上限：

    | 路径 | 上限 |
    |---|---:|
    | WGP local SRAM physical buses | `131,072 B/cycle` |
    | TCP/WGP read | `65,536 B/cycle` |
    | TCP/WGP write | `32,768 B/cycle` |
    | GL1 read | `24,576 B/cycle` |
    | GL1 write | `12,288 B/cycle` |
    | GL2 cached，每方向 | `24,576 B/cycle` |
    | GL2 ↔ EA，每方向 | `12,288 B/cycle` |
    | HBM4 read+write shared | `12,288 B/cycle` |

## 3. Workload 和 launch 规模

用户参数产生：

```text
experts              = 96
tokens               = 16,384
topk                 = 6
valid routed rows M  = 16,384 × 6 = 98,304
rows/expert          = 1,024
raw GEMM N           = 2 × 3,072 = 6,144
K                    = 7,168
output N after SiLU  = 3,072
tile                 = 256 × 256 × 256
active M tiles       = 98,304 / 256 = 384
N tiles              = 6,144 / 256 = 24
active WGs           = 384 × 24 = 9,216
contiguous M tiles   = 480
total WGs            = 480 × 24 = 11,520
early-exit tail WGs  = 11,520 - 9,216 = 2,304
block                = 128 threads = 4 waves
cluster              = 4 × 4 WGs
```

全 GPU 有 `256 WGP × 4 SIMD32 = 1024 SIMD32`。因为每个 WG 使用全部
`320 KiB` LDS，每个 physical SIMD 同时只有一个本 kernel wave：

```text
active rounds/physical SIMD = 9,216 WGs / 256 WGP = 36
tail rounds/physical SIMD   = 2,304 WGs / 256 WGP = 9
total rounds                = 45
```

ATT occupancy 与此完全一致：`256` 个 physical slot key，每个 slot 顺序执行
`45` 个 wave lifetime，其中前 `36` 个 active wave 各执行 `7,589` 条 dynamic
instructions，后 `9` 个 early-exit wave 各执行 `29` 条。

## 4. MXFP4 WMMA compute roof

总 GEMM 工作量：

```text
FLOPs
= 2 × M × N × K
= 2 × 98,304 × 6,144 × 7,168
= 8,658,654,068,736 FLOPs
```

每条 `32x16x128` WMMA 完成：

```text
2 × 32 × 16 × 128 = 131,072 FLOPs
```

Wave32 repeat rate 是 `8 cycles`，所以全 GPU peak 为：

```text
1024 SIMD32 × 131,072 FLOPs / 8 cycles
= 16,777,216 FLOP/GFXCLK-cycle
```

在 `2.4 GHz` 时等于 `40.2653 PFLOP/s`，与 MI455X 白皮书的 OCP MXFP4
`40.26 PFLOP/s` 相符，也验证了这里应使用 `8-cycle` throughput，不能把
Wave64/latency 列误当成 `16` 或 `32-cycle` throughput。

因此：

```text
C_WMMA = 8,658,654,068,736 / 16,777,216
       = 516,096 cycles
```

ATT dynamic trace 给出相同结果：

```text
WMMA/active wave = 1,792
cycles/active WG = 1,792 × 8 = 14,336
active WG rounds = 36
C_WMMA            = 14,336 × 36 = 516,096 cycles
```

## 5. Memory/TDM/LDS roof

### 5.1 Ideal unique bytes 到 HBM

```text
A payload = 98,304 × 7,168 / 2  =   352,321,536 B
A scale   = 98,304 × 7,168 / 32 =    22,020,096 B
B payload = 96 × 6,144 × 7,168 / 2  = 2,113,929,216 B
B scale   = 96 × 6,144 × 7,168 / 32 =   132,120,576 B
read total                              = 2,620,391,424 B
output BF16 = 98,304 × 3,072 × 2       =   603,979,776 B
read + write                            = 3,224,371,200 B
```

HBM read/write pins 共享 `12,288 B/cycle`：

```text
C_HBM,ideal = 3,224,371,200 / 12,288
            = 262,400 cycles
```

这是最乐观的 HBM 下限，要求每个 logical input byte 只从 HBM 取一次，A 在不同
N-cluster 之间的重复访问全部命中 cache。

### 5.2 4x4 multicast source traffic

一个 `256×7168` MXFP4 payload 加 scale 为：

```text
256 × 7,168 / 2 + 256 × 7,168 / 32 = 974,848 B
```

一个 4x4 cluster 需要 4 个不同 A tile 和 4 个不同 B tile。active cluster 数为：

```text
(384 / 4) × (24 / 4) = 576 clusters
```

所以 cluster source read 为：

```text
576 × 8 × 974,848 = 4,492,099,584 B
```

若这些 source read 全部落到 HBM、没有跨 cluster cache reuse，则：

```text
C_HBM,no-cross-cluster-reuse
= (4,492,099,584 + 603,979,776) / 12,288
= 414,720 cycles
```

它仍低于 `C_WMMA=516,096 cycles`。

GL1/GL2 cached read path 的对应下限为：

```text
4,492,099,584 / 24,576 = 182,784 cycles
```

### 5.3 TDM destination delivery

multicast source 只取一次数据，但四个 destination WG 都必须收到自己的 LDS copy：

```text
per active WG input = 2 × 974,848 = 1,949,696 B
all destinations    = 9,216 × 1,949,696
                    = 17,968,398,336 B
```

按 TDM/TCP `256 B/cycle/WGP × 256 WGP = 65,536 B/cycle`：

```text
C_TDM,destination = 17,968,398,336 / 65,536
                  = 274,176 cycles
```

### 5.4 LDS → operands

每个 active wave 的 ATT dynamic counts 为：

```text
DS_LOAD_B128 = 1,824
DS_LOAD_B32  =   456
```

按 Wave32 independent repeat rate `2` 和 `1` cycles，并按 WGP 内两个
SIMD-pair 理想并行：

```text
per active WG
= 4 waves × (1,824 × 2 + 456 × 1) / 2 SIMD-pairs
= 8,208 cycles

C_DS-load = 8,208 × 36 = 295,488 cycles
```

这比只按 raw byte bus 得到的 `279,072 cycles` 略高，因为 `DS_LOAD_B32`
没有填满 256 B/cycle bus。

硬件资料同时注明，WGP mode 下两个 SIMD-pair **可能**一次只允许一个 pair 使用 LDS。
若实硅采用该限制，DS-load 项会翻倍为 `590,976 cycles`。文档措辞是 “may be
limited”，所以不能把它当成已经确认的固定峰值。由于 prologue 和 output tail 与
WMMA/DS body 串行，采用这个限制时当前 kernel 的结构下限会进一步变为：

```text
36 × (945 + 16,256 + 1,875) + 9 × 29
= 686,997 cycles
```

### 5.5 Roofline 汇总

| 资源 | 乐观下限 |
|---|---:|
| MXFP4 XDL | `516,096 cycles` |
| HBM，ideal unique bytes | `262,400 cycles` |
| HBM，无跨 cluster reuse | `414,720 cycles` |
| GL1/GL2 cached read | `182,784 cycles` |
| TDM destination delivery | `274,176 cycles` |
| DS-load，两个 SIMD-pair 并行 | `295,488 cycles` |
| DS-load，假设 WGP mode 单 pair | `590,976 cycles` |

按文档给出的最大物理并行能力，标准 roofline 为：

```text
max(516,096, 262,400, 414,720, 182,784, 274,176, 295,488)
= 516,096 cycles
```

因此这个规模首先是 compute-bound，而不是 HBM-bound。

## 6. 当前汇编的更紧结构下限

ATT 对一个稳定 active wave 的 dynamic stream 分段如下：

| 阶段 | Dynamic instructions | 关键资源 |
|---|---:|---|
| first WMMA 前的 prologue | `945` | 地址、descriptor、accumulator init、首批 TDM/LDS |
| first 到 last WMMA | `4,769` | 其中 `1,792` 条 WMMA |
| last WMMA 后的 output tail | `1,875` | SiLU、BF16 pack、LDS/TDM store |
| 合计 | `7,589` | 每个 active wave |

一个 SIMD 上本 kernel 只有一个 resident wave。按每个 wave 每 cycle 最多推进一条
instruction encoding 的乐观前端模型：

- prologue 在 first WMMA 之前，至少 `945 cycles`；
- WMMA body 的 XDL 下限为 `1,792 × 8 = 14,336 cycles`；body 中其余
  `2,977` 条指令在理想 co-execution 下可以放入 WMMA 的空隙；
- output tail 依赖最后的 accumulator，不能移到 WMMA 前，至少 `1,875 cycles`；
- tail 内有 `256 EXP + 256 RCP`，Trans32 throughput 下限为
  `512 × 2 = 1,024 cycles`，低于 `1,875-cycle` instruction issue floor；
- 每个 early-exit tail wave 动态执行 `29` 条指令。

所以当前 kernel 的乐观结构下限为：

```text
C_active-round
= 945 + 14,336 + 1,875
= 17,156 cycles

C_current-kernel
= 36 × 17,156 + 9 × 29
= 617,877 GFXCLK cycles
```

这是按两个 SIMD-pair 均可工作的最大物理能力计算、比 `516,096-cycle` 标准
roofline 更适合当前汇编的性能 ceiling。如果 WGP mode 的单 SIMD-pair LDS 限制生效，
则应采用上一节的 `686,997 cycles`。它也说明，单独把
SiLU 的 `EXP/RCP` 吞吐加到 WMMA 上只能得到：

```text
516,096 + 36 × 1,024 = 552,960 cycles
```

该值没有计入当前汇编必须执行的 prologue 和其余 output-tail instructions，因此不能
作为整个 kernel 的最终 cycle 上限。

## 7. ATT 对照与剩余差距

ATT capture 的关键结果保存在：

```text
my_code/moe_gemm1_act1_optimized/att_const0_analyze.log
```

远端完整 trace 位于：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/moe_gemm1_act1_optimized/att/opt_const0_att/
```

对照如下：

| 指标 | Cycles | 按 2084.136 MHz 换算 | 相对 ATT 实测 |
|---|---:|---:|---:|
| 纯 XDL roof | `516,096` | `247.631 us` | `44.88%` |
| XDL + Trans32 tail | `552,960` | `265.319 us` | `48.08%` |
| 当前 kernel 结构下限 | `617,877` | `296.467 us` | `53.73%` |
| 若 WGP mode LDS 单 SIMD-pair | `686,997` | `329.632 us` | `59.74%` |
| ATT occupancy maximum | `1,150,033` | `551.160 us`（同次 capture） | `100%` |

ATT 中一个 warm active wave 通常约 `24.7k–30.2k cycles`，而结构下限为
`17,156 cycles/active round`。主要剩余差距来自现有 roofline 无法精确量化的
TDM completion、cluster barrier、LDS/TDM contention、cache/HBM latency、指令取指和
pipeline bubble。第一个 active wave 还观测到约 `124k cycles` 的 cold-start lifetime，
它显著抬高了这次单 dispatch ATT 的总 cycle。

仅凭现有带宽表无法把这些 latency 项压缩成一个唯一、可证明可达的精确 cycle 数；需要
继续采集 TDM/GL1/GL2/HBM counter 或逐阶段 shader timestamp，才能把
`1,150,033 - 617,877 = 532,156 cycles` 的差距进一步归因。

## 8. 复现命令

运行正式 const0 benchmark，得到 20 次迭代的平均 kernel 时间：

```bash
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-const0
```

采集一次 const0 kernel 的 ATT，并调用 `get_isa_runner_att.sh --ana-att` 计算同次
capture 的 `GFXCLK`、wall time、per-SE shader cycles、occupancy 和 wave lifetime：

```bash
bash my_code/moe_gemm1_act1_optimized/run_att_const0.sh
```

`run_att_const0.sh` 内部的核心命令是：

```bash
TRACE_ROOT=my_code/moe_gemm1_act1_optimized/att \
HIP_VISIBLE_DEVICES=0 \
bash my_code/get_isa_runner_att.sh \
  moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1 \
  opt_const0_att \
  "python my_code/moe_gemm1_act1_optimized/att_launch_opt.py" \
  --ana-att
```

其中 `att_launch_opt.py` 只加载预编译的：

```text
my_code/moe_gemm1_act1_optimized/act1_opt.co
```

并启动一次 kernel，避免在 `rocprofv3` 内调用 clang/COMGR。
