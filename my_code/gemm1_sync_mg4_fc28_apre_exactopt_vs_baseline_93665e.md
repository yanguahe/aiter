# `sync_mg4_fc28_apre_exactopt` 相对 `baseline_93665e` 的 GEMM1 优化分析

## 1. 分析范围

本文比较下面两个 `gfx1250` MoE GEMM1 specialization：

```text
baseline_93665e
sync_mg4_fc28_apre_exactopt
```

目标 shape 为：

```text
experts   = 96
tokens    = 16384
topk      = 6
model_dim = 7168
inter_dim = 3072
dtype     = BF16 output, MXFP4 activation, MXFP4 weight
act       = exact SiLU(gate) * up
tile      = M256 x N256 x K256
waves     = 2 x 2 = 4 waves/workgroup
buffers   = 4
cluster_n = 4
```

分析使用的仓库 HEAD 为：

```text
c812bf626a7116d4112ec184255349015e01a306
```

`baseline_93665e` 并不是临时切换 Git HEAD，而是由
`my_code/run_gemm1_baseline_93665e.py` 调用
`use_gemm1_baseline_93665e()`，选择仓库内固定保存的历史 kernel：

```text
aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_93665e.py
SHA256=c8b02647cacf457b772e5e38f9d08999e995e6f221917a45846e32a22c8927fe
baseline commit=93665e8417afe1f07cb9bbe1c4902c38da8e3fa3
```

`sync_mg4_fc28_apre_exactopt` 则是当前
`mxfp4_preshuffle_gfx1250_tdm.py` 的 compile-time specialization。它由
`my_code/reproduce_compare.sh` 设置以下开关：

```bash
AITER_FLYDSL_GEMM1_MMA_GROUP=4
AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1
AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2
AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0
AITER_FLYDSL_GEMM1_WMMA_REUSE=1
AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE=1
```

对应 symbol 为：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p
```

这里的 symbol 没有 `_wpt2`，因为当前命名规则把 WPT2 当作默认值；显式
`_wpt1` 才表示 WPT1。

## 2. 性能结果与分层归因

用户给出的同轮结果是：

| case | GEMM1 | fused MoE | GEMM1 相对 `baseline_93665e` | fused MoE 相对 `baseline_93665e` |
|---|---:|---:|---:|---:|
| `baseline_93665e` | `689.116 us` | `1664.90 us` | `0.00%` | `0.00%` |
| `sync_mg4_fc8` | `622.669 us` | `1595.48 us` | `9.64%` | `4.17%` |
| `sync_mg4_fc28` | `619.572 us` | `1597.08 us` | `10.09%` | `4.07%` |
| `sync_mg4_fc28_apre` | `564.597 us` | `1447.19 us` | `18.07%` | `13.08%` |
| `sync_mg4_fc28_apre_exactopt` | `546.656 us` | `1427.16 us` | `20.67%` | `14.28%` |

按连续版本拆分：

| 变化层级 | GEMM1 绝对变化 | GEMM1 相对上一步 | fused MoE 绝对变化 | fused MoE 相对上一步 |
|---|---:|---:|---:|---:|
| `93665e -> sync_mg4_fc8` | `-66.447 us` | `+9.642%` | `-69.42 us` | `+4.170%` |
| `mg4/fc8 -> mg4/fc28` | `-3.097 us` | `+0.497%` | `+1.60 us` | `-0.100%` |
| `mg4/fc28 -> apre` | `-54.975 us` | `+8.873%` | `-149.89 us` | `+9.385%` |
| `apre -> exactopt` | `-17.941 us` | `+3.178%` | `-20.03 us` | `+1.384%` |

因此，总体 `20.67%` 的 GEMM1 提升可以分为三块：

1. 当前分支公共 prefill pipeline 相对 93665e 的组合收益，约 `9.64%`；
2. A/ScaleA preshuffle 的 producer-consumer 协同收益，约 `8.87%`；
3. WPT2、XDL arbitration、WMMA reuse 和两阶段 output store 的组合收益，约
   `3.18%`。

`mg4/fc28` 在这次单样本中贡献约 `0.50%`。这个差异较小，必须视为需要多轮同机
复测的调度收益，不能把它当成稳定常数。

这次命令设置了 `RUN_VERIFY=0`，所以表中的 `random pass` 为 `not-run`。此前
`RUN_VERIFY=1` 的历史同机测试中，所有 exact 版本都通过 random-input 验证并得到
相同 output hash；本文不会把本次 const0 benchmark 本身描述成 random correctness
证据。

## 3. 两个版本保持不变的计算语义

两个版本都执行：

```text
MXFP4 A x MXFP4 B
  -> FP32 WMMA accumulation
  -> clamp gate/up
  -> exact sigmoid through exp/reciprocal
  -> SiLU(gate) * up
  -> BF16 grouped intermediate
```

它们都保留：

- `M256/N256/K256` tile；
- `w2x2` 四 wave workgroup；
- 四级 K-tile pipeline；
- 每个 accumulator 的 K 累加顺序；
- runtime `m_tile_map` expert lookup；
- exact-SiLU 数学路径；
- 动态 expert 边界和 OOB 处理。

`sync_mg4_fc28_hard` 和 `sync_mg4_fc28_relu` 改变 activation 数学语义，不属于
本文讨论的 exact 优化。

## 4. 当前公共 pipeline 相对 93665e 的优化

### 4.1 cluster 从 `4x1` 扩展为 `4x4`

`baseline_93665e` 使用：

```text
cluster=(4,1,1)
```

四个 peer 只沿 N 方向共享同一份 A/ScaleA。B/ScaleB 会由不同 M tile 的
workgroup 重复加载。

当前 prefill specialization 使用：

```text
cluster=(4,4,1)
```

- A/ScaleA 沿 N 方向 multicast；
- B/ScaleB 沿属于同一 expert 的 M 方向 multicast；
- full cluster 在启动及四槽 ring wrap 处执行 workgroup/cluster barrier；
- partial/sentinel cluster 使用动态 mask，避免错误地把不同 expert 的 B/ScaleB
  合并到同一个 multicast group。

以一个完整 `4x4` macro-cluster 和一个 K256 tile 为例：

```text
A payload      = 256 * 256 / 2 = 32768 B
B payload      = 256 * 256 / 2 = 32768 B
ScaleA         = 256 * (256/32) = 2048 B
ScaleB         = 256 * (256/32) = 2048 B
```

忽略 cache 命中和协议开销，仅计算 descriptor 请求的唯一 payload：

```text
4x1 方式折算到 4x4 区域：
  A/SA = 4 * (32768 + 2048)
  B/SB = 16 * (32768 + 2048)
  total = 696320 B

4x4 双向 multicast：
  A/SA = 4 * (32768 + 2048)
  B/SB = 4 * (32768 + 2048)
  total = 278528 B
```

理论上减少 `417792 B`，即约 `60%` 的这一级重复输入请求。实际收益还受 GL1/GL2
命中、TDM descriptor 开销、cluster barrier 和负载不均衡影响，所以不能直接按
60% 换算成 wall time。

硬件依据来自 MI400 Shader Programming Guide §4.10.3：当
`D#.workgroup_mask != 0` 时，`TENSOR_LOAD_TO_LDS` 使用
`CLUSTER_LOAD_ASYNC` 向多个 workgroup 的 LDS multicast。

### 4.2 四槽 ring 的完整同步协议

当前实现不只增加 multicast mask，还增加了与 `4x4` cluster 匹配的生命周期管理：

- 启动时保证所有 participating WG 已经进入相同 generation；
- 每四个 K tiles、即 LDS ring wrap 前再次同步；
- 保留每个 WG 自身的 `s_wait_tensorcnt`；
- `_rcw` 只放松不必要的 LDS drain，不删除 cluster barrier。

这部分是正确性协议。此前对纯汇编版本的实验也证明，直接删除 steady cluster
barrier 会 deadlock，不能把 barrier 数量简单当成可删除开销。

### 4.3 batch-8 exact-SiLU epilogue

93665e 按较小 fragment 依次执行 `exp -> add -> rcp -> mul`，独立 sigmoid chain
之间的 overlap 有限。当前 specialization 默认：

```text
epilogue_batch_wn=8
```

每个 `wm` 一次收集 8 个 logical N fragments，也就是 32 个 `(gate, up)` pair，
分阶段发出独立的 `exp2`、denominator add、reciprocal 和乘法。这样可以提高 TRANS
pipeline 的并行度，同时限制临时 VGPR 的生命周期。

这仍然是 exact-SiLU；它改变的是发射顺序，不改变公式。

### 4.4 output LDS row skew

93665e 的 active-SiLU 路径使用：

```text
STORE_PAD=0
```

当前 prefill specialization 使用：

```text
STORE_N     = 128 BF16
STORE_PAD   = 8 BF16
STORE_PITCH = 136 BF16 = 272 B
```

MI400 的 LDS 有 64 个、每个 4-byte 宽的 bank。`272 B = 68 dwords`，相邻 row
的起始 bank 前进 4，从而打散 `ds_store_b64` 的 lane-to-bank 映射。padding 列由
TDM descriptor 的 OOB extent 丢弃，不写入 global output。

硬件依据来自 MI400 Shader Programming Guide §4.7.1 和 §4.10.2：LDS bank
conflict 会串行化访问；TDM tile 可以通过 tile shape/OOB 描述跳过 global 边界外的
列。

### 4.5 descriptor/control 与 compiler schedule hints

当前源码把 TDM descriptor construction 放入 owning-wave 分支，使 WPT1 时每个
wave 只构造自己实际发出的 descriptor。`emit_hints()` 还显式向 compiler 描述：

- 当前 K128 的 DS read 数；
- WMMA group；
- 下一 K128 的 DS read 分布；
- tail TDM issue 的位置。

这些变化与 4x4 cluster、batch-8 epilogue、output skew 一起组成
`93665e -> sync_mg4_fc8` 的公共优化栈。现有 benchmark 只能给出它们的组合收益
`9.64%`，不能从这组数据中进一步精确拆分。

## 5. `mg4/fc28` 的调度变化

每个 K128、每个 wave 有：

```text
32 physical WMMA
40 logical LDS reads
```

其中 40 个 LDS reads 来自：

```text
A payload  : 8 wm * 2 ds_read_b128 = 16
B payload  : 8 wn * 2 ds_read_b128 = 16
ScaleA     : 4 ds_read_b32          = 4
ScaleB     : 4 ds_read_b32          = 4
```

`MMA_GROUP=4`、`FENCE_COVER_MMA=28` 的意图是：

1. 在典型 K256 tile 的第一个 K128 中，先保留一个 4-WMMA group；
2. 尽早把下一状态的 DS reads 放进调度窗口；
3. 最后使用连续 28 条 WMMA 覆盖 fence/数据到达延迟。

这个开关只影响 compiler 的 scheduling regions，不改变：

- K tile 遍历顺序；
- accumulator 的 K 累加顺序；
- TDM 数据内容；
- exact-SiLU；
- cluster 同步语义。

本轮 `mg4/fc8 -> mg4/fc28` 的 GEMM1 差异是 `3.097 us`，约 `0.50%`；MoE e2e
反而慢 `1.60 us`。因此它属于小幅、易受频率与机器状态影响的调度优化。

## 6. A/ScaleA preshuffle

这是 `sync_mg4_fc28_apre` 相对 `sync_mg4_fc28` 的核心变化，也是本轮第二大
收益来源。

### 6.1 producer：从 route 重复量化改为 token 一次量化

对本 shape：

```text
routes = tokens * topk = 16384 * 6 = 98304
```

普通 routeks 路径会按 routed row 读取并量化 activation；同一个 source token
可能被重复处理 6 次。A-preshuffle fast path 改成三个阶段：

```text
moe_quant_token_fd7168_fp4_pk8
  每个 source token 量化一次

moe_invert_route_rows_tk6
  建立 grouped_row -> source_token 映射

moe_scatter_preshuffled_a_fd7168_r32_lds
  每 32 个 grouped rows 经 LDS transpose 后写出 A payload 与 ScaleA
```

它增加了 inverse/scatter launch，却把最重的 7168-element quantization 从每 route
一次降到每 token 一次。因而 A-preshuffle 对 fused MoE 的收益比只看 GEMM1
consumer 更明显。

### 6.2 consumer：A payload 的物理布局

普通 A view：

```text
global outer = 256 rows
inner        = 128 bytes per K256 FP4 row
LDS row      = 128 + 16 bytes padding
```

A-preshuffle view：

```text
global/LDS outer = 256 / 16 = 16
inner            = (Ktile/2) * 16 = 2048 bytes
LDS padding      = 0
```

对应的 LDS lane mapping 变为：

```text
lds_a_lane_off = (wmb/16) * A_LDS_ROW + kgrp*256 + lane16*16
load_a offset  = wm * A_LDS_ROW + ksl*1024
second b128    = first + 512
```

consumer 不再在 GEMM hotloop 前后为普通 row-major A 做额外的 LDS padding/重排。

### 6.3 ScaleA 的物理布局

ScaleA 从按 wave-M tile 的 `(k128, wm, lane16)` 排列，改为：

```text
outer = tile_m / 32 = 8 super-rows
inner = tile_k / 4  = 64 dwords
```

这使 ScaleA 与 preshuffled A 使用一致的 32-row/K-block 组织，DS load 可以直接
得到 WMMA 所需的 scale lane。

### 6.4 收益边界

本轮：

```text
GEMM1:   619.572 -> 564.597 us, +8.873%
MoE e2e: 1597.08 -> 1447.19 us, +9.385%
```

GEMM1 的收益主要来自更直接的 A/ScaleA TDM 和 LDS mapping；e2e 还包含“每 token
量化一次”的 producer 收益。两者不能混为同一项 kernel 内优化。

## 7. `exactopt` 的四项附加优化

### 7.1 GEMM1-only WPT2

```text
AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2
```

WPT1 的四个 wave 分别负责 A、B、ScaleA、ScaleB。每个 K256 tile 的数据量为：

| tensor | bytes |
|---|---:|
| A | `32768` |
| B | `32768` |
| ScaleA | `2048` |
| ScaleB | `2048` |

因此 WPT1 的 owner 负载约为：

```text
wave0 = 32 KiB
wave1 = 32 KiB
wave2 =  2 KiB
wave3 =  2 KiB
```

WPT2 把 owner 分成 `(wave0,wave1)` 和 `(wave2,wave3)` 两组：

```text
wave0/1 各自加载 1/2 A + 1/2 ScaleA = 17 KiB
wave2/3 各自加载 1/2 B + 1/2 ScaleB = 17 KiB
```

总字节数不变，但 descriptor work 和 TDM transfer 在四个 wave 间更均衡。每个 wave
每个 K tile 发出两个 descriptor。MI400 Shader Programming Guide §4.10.8 规定每个
wave 最多 3 个 TDM op in flight、每个 SIMD 最多 6 个，因此该 ownership 在硬件
上有足够的 descriptor slot，但 wait threshold 必须按两个 descriptor/wave 重新计算。

这个 override 只作用于 GEMM1，避免改变 GEMM2 的调度和性能。

### 7.2 `DISABLE_XDL_ARB_STALL=0`

当前 M256/N256/4-wave specialization 的 automatic default 会设置
`SCHED_MODE.bit[2]`，允许同一 wave 连续 issue 多条 WMMA。`exactopt` 显式设为 0，
保留正常 XDL arbitration stall。

MI400 Shader Programming Guide §4.3.7.4.2 和 CDNA5 ISA §5.7.2.1 说明：

- bit 2 为 1 时，一个 wave 可以连续 issue 多条 WMMA；
- 这样可能阻塞其他 wave 的 co-execution；
- 文档认为它主要可能对“每 SIMD 只有一个 wave”的情形有利；
- 实际最优值仍取决于指令顺序和同一 WGP 内各 wave 的进度。

历史单项实验中，`DISABLE_XDL_ARB_STALL=0` 相对当时的 A-preshuffle 起点提升约
`1.91%`。该数字来自不同机器状态，只用于说明方向有效，不能直接与本轮数据相加。

### 7.3 WMMA A/B operand reuse

```text
AITER_FLYDSL_GEMM1_WMMA_REUSE=1
```

WMMA 遍历采用 snake order：偶数 `wm` 正向遍历 `wn`，奇数 `wm` 反向遍历。
这样：

- 同一 `wm` 内相邻 WMMA 复用 A operand；
- 相邻 snake row 的边界 WMMA 复用 B operand。

源码只在相邻 operand 确实相同的位置设置：

```text
reuseA = wn_raw > 0
reuseB = wm > 0 and wn_raw == 0
```

CDNA5 ISA §7.12 明确要求，reuse bit 只能在相邻 WMMA 确实复用对应 matrix 时设置，
否则结果 undefined。因此这不是可以全局打开的“性能 bit”，而是与 WMMA 遍历顺序
绑定的正确性约束。

历史单项实验中 A+B reuse 的收益约 `0.94%`；同样只应作为方向性证据。

### 7.4 两阶段 output TDM store

普通路径完成全部 128 个 M rows 的 exact-SiLU 和 LDS staging 后，再统一发出一个
完整 output store。`exactopt` 把它改成：

```text
activate/stage wm0..wm3
  -> workgroup barrier
  -> wave_n==0 发出前 64 rows 的 output TDM

同时执行 wm4..wm7 exact-SiLU/staging
  -> workgroup barrier
  -> 发出后 64 rows 的 output TDM

最后 tensor_wait(0)
```

单个 workgroup 的 activated output 为：

```text
256 rows * 128 BF16 columns = 65536 B
```

两阶段各搬运 32768 B。第一半 TDM 与第二半 exact-SiLU 的 VALU/TRANS work
重叠，最终 bytes 和数学结果不变。

最终 ATT 记录中，加入两阶段 overlap 后：

| phase | cycles/wave |
|---|---:|
| prologue | `6130` |
| WMMA core | `21702` |
| epilogue | `3499` |
| total | `31331` |

最终 `s_wait_tensorcnt 0` 只剩约 `26 cycles/wave`。这说明 output TDM 的大部分
drain 已经被后半 epilogue 覆盖。

历史 paired run 中：

```text
apre       = 564.321 us
exactopt   = 546.078 us
improvement= 3.23%
```

历史候选数据还显示，WPT2+xdl0+reuse 为约 `538.443 us`，再加入两阶段 output
store 后约 `533.058 us`，对应约 `1.0%` 的额外降低。不同测量窗口存在频率漂移，
这些数字只用于估计各机制的量级。

## 8. 完整 feature matrix

| 机制 | `baseline_93665e` | `sync_mg4_fc28_apre_exactopt` | 性质 |
|---|---|---|---|
| cluster geometry | `4x1` | `4x4` | 减少 B/ScaleB 重复流量 |
| A/ScaleA multicast | 有 | 有 | 两者共有 |
| B/ScaleB multicast | 无 | 有 | 当前公共 pipeline |
| four-stage ring synchronization | 仅 1-D 协议 | 完整 2-D cluster 协议 | 正确性与复用 |
| next-stage prefetch | 有 | 有 | 两者共有 |
| TDM ownership | WPT1 | GEMM1-only WPT2 | 平衡四 wave 的输入负载 |
| A payload layout | 普通 row-major + LDS pad | producer 预先重排 | 减少 consumer 重排/冲突 |
| ScaleA layout | wave-M/k128 layout | 32-row super-row layout | 匹配 preshuffled A |
| epilogue batching | scalar/small fragment | batch 8 | 隐藏 TRANS latency |
| active output LDS pitch | 无 pad | `128+8` BF16 | 降低 LDS bank conflict |
| compiler schedule | 无有效 hints | `mg4/fc28` | DS/TDM/WMMA 排序 |
| XDL arbitration bit 2 | 未设置 | 显式不设置 | 保留 wave 间 co-execution |
| WMMA operand reuse | 无 | A+B exact reuse | source-cache hint |
| output store | 完整 tile 后统一发出 | 两个 64-row phase | 与后半 epilogue overlap |
| activation | exact SiLU | exact SiLU | 数学语义相同 |
| expert lookup | runtime binary search | runtime binary search | non-balanced 兼容性相同 |

## 9. 如何理解 20.67% 的总收益

`sync_mg4_fc28_apre_exactopt` 的收益不是“某一条更快的 WMMA 指令”造成的。它同时
减少了三类时间：

1. **输入流量和 owner imbalance**：4x4 multicast、A preshuffle、WPT2；
2. **hotloop stall**：`mg4/fc28`、正常 XDL arbitration、合法的 WMMA reuse；
3. **epilogue 暴露时间**：batch-8 exact-SiLU、LDS row skew、两阶段 output TDM。

其中可从现有逐级 benchmark 较可靠地看到：公共 pipeline 与 A-preshuffle 是两个
最大的收益块；最后四项 exact-only 优化属于几个百分点的收尾优化。单项历史数字
不能线性相加，因为它们会竞争相同的 DS/TDM/WMMA overlap window。

## 10. 复现命令

只比较两个目标版本，并执行 random correctness：

```bash
CASE_LIST=baseline_93665e,sync_mg4_fc28_apre_exactopt \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh
```

同时保留中间层，观察性能 waterfall：

```bash
CASE_LIST=baseline_93665e,sync_mg4_fc8,sync_mg4_fc28,sync_mg4_fc28_apre,sync_mg4_fc28_apre_exactopt \
ROUNDS=5 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh
```

采集 ATT：

```bash
CASE_LIST=baseline_93665e,sync_mg4_fc28_apre,sync_mg4_fc28_apre_exactopt \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=1 \
bash my_code/reproduce_compare.sh
```

性能比较必须使用同一次启动后的同机交错结果。MI450 机器的动态频率和系统状态会
明显漂移，不能把不同机器或重启前后的绝对微秒数直接相减。

## 11. 主要证据

软件：

- `my_code/reproduce_compare.sh`
- `my_code/run_gemm1_baseline_93665e.py`
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py`
- `aiter/ops/flydsl/grouped_moe_gfx1250.py`
- `aiter/ops/flydsl/moe_kernels.py`
- `aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py`
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_93665e.py`
- `my_code/gemm1_apre_12pct_optimization_STATUS.md`

硬件资料：

- `MI400_Shader_Programming#65.txt` §4.3.7.4.2：
  `DISABLE_XDL_ARB_STALL`；
- `MI400_Shader_Programming#65.txt` §4.6：WMMA matrix reuse；
- `MI400_Shader_Programming#65.txt` §4.7.1：64-bank、4-byte/bank LDS；
- `MI400_Shader_Programming#65.txt` §4.10.3、§4.10.8：TDM multicast、
  `early_timeout`、descriptor in-flight limits；
- `amd-instinct-cdna5-instruction-set-architecture.txt` §5.7.2.1、§7.12。
