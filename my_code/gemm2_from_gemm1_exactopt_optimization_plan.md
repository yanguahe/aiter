# 从 GEMM1 `exactopt` 迁移到 MoE GEMM2 的优化分析与实施方案

<!-- markdownlint-disable MD013 -->

## 0. 结论摘要

本文分析当前 `sync_mg4_fc28_apre_exactopt` 相对 `baseline_93665e` 的 GEMM1
优化中，哪些机制可以迁移到同一 A4W4 MoE 的 GEMM2。分析基于本地仓库：

```text
branch = hyg_gfx1250_gemm_a4w4
HEAD   = 6059ea13149d2bb853f5217582bf868d50a64831
date   = 2026-09-16
```

目标 workload：

```text
experts     = 96
tokens      = 16384
topk        = 6
model_dim   = 7168
inter_dim   = 3072
data_format = A4W4
act         = SiLU
bias        = false
```

当前 GEMM2 的核心 shape 是：

```text
M = grouped routed rows
N = 7168
K = 3072
tile = M256 x N256 x K256
waves = 2 x 2 = 4 waves/workgroup
buffers = 4
cluster = 4 x 1
WPT = 1
stage1_act = 0
```

迁移结论按优先级排列：

| 优化机制 | GEMM2 适用性 | 当前状态 | 建议 |
|---|---|---|---|
| 合法的 WMMA A/B operand reuse | 高 | snake order 已有，reuse bit 未开 | 第一批实现 |
| `4x4` 双向 multicast | 高 | 只有 `4x1` A multicast | 结构性主候选 |
| WPT2 owner balance | 高 | GEMM2 仍为 WPT1 | 在 `4x4` 前后分别消融 |
| `mg4/fc28` 类 schedule hints | 中高 | GEMM2 被 specialization gate 强制关闭 | 在最终 input layout 上重新 sweep |
| 两阶段 output TDM | 中高 | GEMM2 统一 store 整个 output tile | 基于 ATT 的 output-tail 证据实现 |
| A/ScaleA preshuffle | 条件适用 | producer/consumer 代码已有能力但未接线 | 必须计入 A2 quant producer 成本 |
| 4 槽 ring 的 2-D cluster 同步 | 必须 | GEMM2 当前未启用 | 与 `4x4` 一起启用，不能单独删除 barrier |
| descriptor 只在 owner wave 构造 | 已有 | 公共代码已经覆盖 GEMM2 | 保留，不重复开发 |
| output LDS row skew | 已有 | GEMM2 已使用 `STORE_PAD=16` | 保留，不照搬 GEMM1 的 pad8 |
| 正常 XDL arbitration | 已有 | GEMM2 当前没有设置 bit 2 | 保持，不建立 no-op case |
| batch-8 exact-SiLU | 不适用 | GEMM2 无 activation epilogue | 不迁移 |
| 每 token 只量化一次 | 不适用 | GEMM2 输入是 expert-specific routed rows | 不迁移 |

建议的主线不是把 GEMM1 的所有开关机械地打开，而是：

```text
baseline trace
  -> exact WMMA reuse
  -> 4x4 B/ScaleB multicast
  -> WPT2 balance
  -> 条件式 A/ScaleA preshuffle
  -> 按最终 layout 重扫 schedule hints
  -> 两阶段 output TDM
  -> 只组合各自独立胜出的版本
```

## 1. 当前 GEMM2 的数据流和性能起点

### 1.1 当前调用链

A4W4 路径中，GEMM1 先产生 BF16 grouped intermediate：

```text
GEMM1 exact-SiLU output: y[global_grouped_row, 3072] BF16
```

随后：

```text
flydsl_moe_fused_quant_preshuffle(
    y,
    source_topk=0,
    quant_mode="fp4",
)
```

把每个 grouped row 量化为 GEMM2 的 FP4 A payload 和 E8M0 ScaleA。最后 GEMM2 调用：

```text
flydsl_grouped_gemm_a8w4_masked(
    grouped_out,
    a2_payload,
    w2,
    a2_scale,
    w2_scale,
    m_tile_map,
    N=7168,
    K=3072,
    stage1_act=0,
)
```

对应代码入口：

- `aiter/ops/flydsl/grouped_moe_gfx1250.py`：A2 quant 与 GEMM2 调用；
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py`：specialization 参数选择；
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`：实际 TDM/WMMA kernel。

当前 tuned row 位于 `aiter/configs/tuned_grouped_fmoe.csv:74`：

```text
tile_m2/tile_n2/tile_k2 = 256/256/256
m_warp2/n_warp2         = 2/2
num_buffers2             = 4
cluster_n                = 4
waves_per_tensor_tdm     = 1
next_stage_prefetch      = 1
```

### 1.2 现有性能观测

用户给出的当前观测为：

```text
GEMM2 latency          = 405.022 us
GEMM2 executed         = 4,329,327,034,368 FLOP
GEMM2 throughput       = 10,689.1 TFLOP/s
GEMM2 effective read   = 1,283,457,024 B
GEMM2 output write     = 1,409,286,144 B
GEMM2 effective R+W    = 2,692,743,168 B
GEMM2 effective R+W BW = 6.047 TB/s
MoE e2e                = 1367.10 us
```

这里的 `effective R+W` 是 unique-logical traffic，不是物理 HBM traffic。它忽略 tile
重复请求、cache hit、multicast、padding 和 replay，只适合作为固定 workload 下的有效工作量口径。

有效流量拆分为：

| 项目 | bytes | 占 GEMM2 effective R+W |
|---|---:|---:|
| A payload | `150,994,944` | `5.61%` |
| ScaleA | `9,437,184` | `0.35%` |
| B payload | `1,056,964,608` | `39.25%` |
| ScaleB | `66,060,288` | `2.45%` |
| BF16 grouped output | `1,409,286,144` | `52.33%` |

因此 GEMM2 与 GEMM1 的侧重点不同：

- GEMM2 的 B/ScaleB 是主要 input surface；
- GEMM2 output 是整个有效 R+W 中最大的单项；
- GEMM2 没有 exact-SiLU TRANS chain；
- GEMM2 的 K 只有 `3072/256=12` 个 K tiles，而 GEMM1 有 28 个，prologue/epilogue
  在总时间中的相对占比可能更高。

### 1.3 当前资源证据

现有本地 GEMM2 ISA artifact：

```text
my_code/gemm1_cycle_105pct_20260909/dump_baseline/
  a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1/
  21_final_isa.s
```

其中 metadata 为：

```text
LDS   = 294,912 B = 288 KiB
VGPR  = 732
SGPR  = 54
spill = 0
```

该 artifact 不是当前 HEAD 的权威重新 dump，只能用于建立资源量级。实施 P0 时必须重新生成
current-HEAD GEMM2 code object。即便如此，现有数值已经说明：

- LDS 大于 `160 KiB`，无法达到 2 WG/WGP；
- VGPR 也大于 512，无法达到 2 waves/SIMD；
- 当前 GEMM2 基本处于 occupancy 1 区间；
- 小幅减少 LDS/VGPR 不会自动提高 occupancy，但可以减少访问开销或给后续调度留下空间。

## 2. GEMM1 优化逐项迁移判断

### 2.1 `4x4` cluster 与双向 multicast：高价值、可迁移

当前 GEMM2 是：

```text
cluster = (4, 1, 1)
```

同一行的 4 个 WG 共享 M tile、使用不同 N tile，因此只有 A payload 通过
`a_mcast_mask` 沿 N multicast。当前 `cluster_m=1` 时：

- A multicast；
- ScaleA 不 multicast；
- B 不 multicast；
- ScaleB 不 multicast。

目标改为：

```text
cluster = (4, 4, 1)
```

则：

- A/ScaleA 沿 N 方向共享；
- B/ScaleB 沿属于同一 expert 的 M peer 共享；
- cluster 跨 expert 边界时继续使用当前动态 `column_mask`；
- sentinel tail 不进入 cluster barrier，保留当前安全 fallback。

对一个完整 `4x4` macro-cluster、一个 K256 tile：

```text
A payload = 32 KiB
ScaleA    =  2 KiB
B payload = 32 KiB
ScaleB    =  2 KiB
```

当前 `4x1` 行为折算到同一 `4x4` 区域：

```text
A  :  4 * 32 KiB = 128 KiB
SA : 16 *  2 KiB =  32 KiB
B  : 16 * 32 KiB = 512 KiB
SB : 16 *  2 KiB =  32 KiB
total             = 704 KiB
```

目标 `4x4`：

```text
A  : 4 * 32 KiB = 128 KiB
SA : 4 *  2 KiB =   8 KiB
B  : 4 * 32 KiB = 128 KiB
SB : 4 *  2 KiB =   8 KiB
total            = 272 KiB
```

descriptor-level payload 理论减少：

```text
704 KiB -> 272 KiB
reduction = 432 KiB = 61.36%
```

balanced 主场景下，每个 expert 恰好有：

```text
1024 rows / M256 = 4 M tiles
```

因此一个 M 方向 4-WG cluster 可以完整覆盖一个 expert，B/ScaleB multicast 几何非常理想。
全 workload 共有：

```text
96 M-clusters * 7 N-clusters * 12 K-tiles = 8064 macro-cluster/Ktile units
```

上述模型对应约 `3.57 GB` 的 descriptor-level payload 减少。这个数不是物理 HBM 节省；
GL1/GL2 命中、TDM 合并和 replay 会改变实际流量，必须由 PMC 或 thread trace 验证。

实现时可以复用当前 kernel 中已经存在的：

```text
b_mcast_mask
full_cluster
cluster_sync()
ring-wrap cluster barrier
partial/sentinel fallback
```

主要改动是新增严格的 GEMM2 specialization gate，使 `cluster_m=4` 不再只受
`stage1_act==1 && K==7168` 控制。

正确性要求：

- 保留每个 WG 的 runtime `m_tile_map` binary search；
- 不假设 balanced rows；
- B/ScaleB mask 只能覆盖同一 expert 的 M peers；
- full cluster 的 startup/ring-wrap barrier 不得删除；
- sentinel cluster 必须继续走无 cluster barrier 的安全路径。

### 2.2 4 槽 ring 的 2-D 同步：不是独立优化，而是 `4x4` 的必要协议

GEMM2 当前已有 4-buffer ring 和 next-stage prefetch，但只运行 1-D cluster 协议。
启用 `cluster_m=4` 后必须同时启用：

```text
startup cluster synchronization
每 4 个 K tiles 的 ring-wrap synchronization
每 WG 自身的 s_wait_tensorcnt
安全的 LDS reuse fence
```

GEMM2 有 12 个 K256 tiles，恰好经历 3 个四槽 generation。任何“先删 barrier 再看性能”
的做法都不允许；此前 GEMM1/persistent 实验已经证明，错误的 generation 协议可能 deadlock，
而不是只产生轻微数值误差。

### 2.3 WPT2：可直接迁移，目标是消除 owner imbalance

GEMM2 每个 K256 tile 的四类输入与 GEMM1 相同：

| tensor | bytes |
|---|---:|
| A | `32768` |
| B | `32768` |
| ScaleA | `2048` |
| ScaleB | `2048` |

当前 WPT1：

```text
wave0 -> A      = 32 KiB
wave1 -> B      = 32 KiB
wave2 -> ScaleA =  2 KiB
wave3 -> ScaleB =  2 KiB
```

WPT2 后：

```text
wave0/1 -> 各 1/2 A + 1/2 ScaleA = 17 KiB/wave
wave2/3 -> 各 1/2 B + 1/2 ScaleB = 17 KiB/wave
```

每个 wave 每个 K tile 发出两个 descriptor。MI400 Shader Programming Guide §4.10.8
规定每 wave 最多 3 个 tensor ops 从 issue 到 XACK 同时在途、每 SIMD 最多 6 个。单个
generation 的两个 descriptor 小于 per-wave 上限，但 next-stage prefetch 可能让前后 generation
短暂重叠并触发 descriptor backpressure，必须通过 ATT/PMC 验证。WPT4 单个 generation 就需要
每 wave 发出 4 个 descriptor，更容易触及该限制，不应作为首选。

实现建议：

- 新增 `AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM`；
- 不复用 GEMM1-only env，避免调 GEMM2 时意外改变 GEMM1；
- `TDM_PER`、`s_wait_tensorcnt` threshold 和 compiler hint 必须按每个 owner wave 的真实
  descriptor 数重新计算；
- 先分别测试 `cluster_m=1` 下的 WPT1/WPT2，再测试 `cluster_m=4` winner 上的 WPT1/WPT2，
  区分 owner balance 与 multicast 的收益。

### 2.4 A/ScaleA preshuffle：布局可迁移，“每 token 量化一次”不可迁移

GEMM1 A preshuffle 有两个不同收益来源：

1. producer 把相同 token 的 6 次 route quant 降为一次；
2. consumer 使用 WMMA-friendly A/ScaleA physical layout。

对 GEMM2，第一项不成立。GEMM2 输入是：

```text
SiLU(W1_expert(token)) * Up(W1_expert(token))
```

同一个 token 路由到不同 expert 后，中间向量已经不同，不能跨 route 只量化一次。强行复用会改变
GEMM2 输入和最终 MoE 数学结果。

可以迁移的只有第二项：

```text
A payload:
  row-major + 16B LDS pad
    -> [row//16][K16B block][row%16][16B]

ScaleA:
  [wave-M][k128][wm][lane16]
    -> [row//32][scale-dword][row%32]
```

对 GEMM2 K256 tile：

```text
当前 A LDS stage = 256 * (128 + 16) = 36,864 B
apre A LDS stage = 16 * 2,048       = 32,768 B
每 ring slot 减少                         4,096 B
四槽 ring 共减少                         16 KiB
```

预计 LDS 从约 `288 KiB` 降到约 `272 KiB`，仍然不能达到 2 WG/WGP，因此不能把收益解释为
occupancy 提升。潜在收益来自更直接的 TDM/LDS mapping 和更规则的 `ds_load_b128` bank
分布。

当前 producer 代码实际上已经支持 `a_preshuffle=True`，但 stage2 调用没有传入。第一版实验只需：

1. 新增 `AITER_FLYDSL_GEMM2_A_PRESHUFFLE`；
2. A2 quant 调用传 `a_preshuffle=True`；
3. GEMM2 consumer 同时传 `a_preshuffle=True`；
4. producer 和 consumer 必须成对切换，禁止布局不匹配。

风险在 producer：stage2 quant 本来每 grouped row 只执行一次，直接写 A-preshuffle 地址可能降低
global store 合并效率。因此必须同时记录：

```text
A2 quant producer time
GEMM2 time
MoE e2e time
```

只有 consumer 收益大于 producer 回退时才能保留。若直接 `_apre` producer 回退明显，但 GEMM2
consumer 单独收益成立，第二阶段才考虑专用 32-row LDS transpose producer；不能一开始就增加新 kernel。

### 2.5 `mg4/fc28` schedule hints：计算几何相同，但必须重新 sweep

GEMM2 和 GEMM1 使用相同的 M256/N256/K256/w2x2 FP4 WMMA geometry。每个 K128、每 wave
仍然有：

```text
32 physical WMMA
40 logical LDS reads
```

因此 GEMM1 的调度思路可迁移：

```text
把下一 K128 的 DS reads 分散到 WMMA groups 之间
保留尾部连续 WMMA 覆盖 fence/data-ready latency
把 tail TDM issue 分散到合适的 scheduling slots
```

但不能直接断言 `mg4/fc28` 仍是 GEMM2 最优值，因为：

- GEMM2 只有 24 个 K128 steps，GEMM1 有 56 个；
- GEMM2 没有 exact-SiLU epilogue；
- GEMM2 output tile 大一倍；
- `4x4`、WPT2 和 Apre 会改变 TDM/DS wait 分布。

实施方法：

1. 先确定最终保留的 cluster、WPT 和 A layout；
2. 新增 GEMM2-only `schedule_hints` gate；
3. 固定其他变量，扫描：

```text
MMA_GROUP       = 2, 4, 8
FENCE_COVER_MMA = 8, 12, 20, 24, 28
```

4. 首轮保留 `mg4/fc8` 与 `mg4/fc28` 两个锚点；
5. 小于 `max(0.5%, 3*CV)` 的差异视为噪声，不合入默认路径。

### 2.6 正常 XDL arbitration：GEMM2 已经具备，不需要迁移

当前 kernel 只有在：

```text
fp4_prefill_schedule && xdl_arb_off
```

时才写 `SCHED_MODE.bit[2]`。GEMM2 的 `stage1_act=0, K=3072` 使
`fp4_prefill_schedule=False`，所以当前 GEMM2 没有设置
`DISABLE_XDL_ARB_STALL`，已经保持 normal arbitration。

因此 `AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0` 对当前 GEMM2 是 no-op，不应建立一个
虚假的“优化 case”。

特别注意：如果后续把 GEMM2 纳入新的 `fp4_*_schedule` gate，不能沿用代码中的 automatic
`xdl_arb_off=1`，否则可能在无意中打开 bit 2。GEMM2 specialization 必须显式以：

```text
disable_xdl_arb_stall = 0
```

为基线，只有独立实验证明 bit 2 有利时才新增候选。MI400 Shader Programming Guide
§4.3.7.4.2 指出 bit 2 允许同一 wave 连续发出多条 WMMA，但可能阻塞其他 wave 的
co-execution。

### 2.7 WMMA operand reuse：最直接、最低风险的迁移项

GEMM2 当前源码已经采用 snake order：

```text
wm0: wn0 -> wn1 -> wn2 -> wn3
wm1: wn3 -> wn2 -> wn1 -> wn0
wm2: wn0 -> wn1 -> wn2 -> wn3
...
```

但 `grouped_gemm_mxfp4.py` 只在 GEMM1 prefill specialization 下传入
`wmma_reuse`，所以 GEMM2 最终 ISA 没有 `matrix_a_reuse` / `matrix_b_reuse`。

第一版只需增加 GEMM2-only reuse selector，保持 traversal 不变：

```text
reuseA = wn_raw > 0
reuseB = wm > 0 and wn_raw == 0
```

每 K128 的理论分布：

```text
32 WMMA total
24 matrix_a_reuse
 7 matrix_b_reuse
 1 no-reuse
```

验收不能只看 symbol，必须审计 final ISA 中每个带 reuse 的 WMMA：

- 对 `matrix_a_reuse`，相邻 WMMA 的对应 activation VGPR range 和 scale selector 必须一致；
- 对 `matrix_b_reuse`，相邻 WMMA 的对应 weight VGPR range 和 scale 必须一致；
- K128/K256 边界的第一条 WMMA 不得继承上一阶段的 reuse；
- random output hash128 必须与 baseline 完全一致。

CDNA5 ISA §7.12 明确规定，reuse bit 在 operand 不相同时会导致 undefined result。该项预计是
小幅优化，GEMM1 的约 `0.94%` 只能作为方向性证据，不能作为 GEMM2 预期值。

### 2.8 output LDS row skew：GEMM2 已经有更适合 b128 的版本

GEMM1 active-SiLU 输出使用 `ds_store_b64`，因此 exactopt 采用：

```text
STORE_N=128 BF16
STORE_PAD=8 BF16
```

GEMM2 无 activation，输出路径使用 `ds_store_b128`，当前已经采用：

```text
STORE_N=256 BF16
STORE_PAD=16 BF16
STORE_PITCH=272 BF16=544 B=136 dwords
```

相邻 row 的起始 bank 前进：

```text
136 mod 64 = 8 banks
```

这正是源码为 b128 store 保留的 eight-dword skew。因而：

- 不应把 GEMM1 的 pad8 机械移植到 GEMM2；
- 默认保留 pad16；
- 只有 GEMM2 ATT/PMC 明确显示 output `ds_store_b128` bank-conflict stall 时，才做
  `pad=0/8/16/24` 独立 sweep；
- padding 列必须继续由 output TDM descriptor 的 inner OOB extent 丢弃，global output bytes
  不得增加。

### 2.9 两阶段 output TDM：可迁移，但 overlap window 比 GEMM1 更短

当前 GEMM2 在所有 FP32 accumulators 完成 BF16 conversion 和 LDS staging 后，统一发出一个
完整 output TDM。单 WG output tile 为：

```text
256 rows * 256 BF16 columns = 131,072 B
```

建议沿用 GEMM1 的两阶段框架：

```text
stage wm0..wm3 output to LDS
  -> workgroup barrier
  -> wave_n==0 owners 发出前半 output TDM

继续 stage wm4..wm7 output to LDS
  -> workgroup barrier
  -> 发出后半 output TDM

final tensor_wait(0)
```

每阶段总 output 为：

```text
128 rows * 256 BF16 = 65,536 B
```

实现时需要修改现有 `issue_output_half()` 的 activation-specific 地址：

```text
GEMM1 active output:
  out_stride  = i32_n / 2
  out_col_off = blk_n / 2

GEMM2 passthrough output:
  out_stride  = c_stride = i32_n
  out_col_off = c_inner_off = blk_n
```

GEMM2 的 remaining epilogue 只有 FP32→BF16 conversion、`ds_store_b128` 和少量控制指令，
没有 exact-SiLU TRANS work，所以可覆盖 output TDM 的计算窗口短于 GEMM1。另一方面，GEMM2
output tile 是 GEMM1 active output 的两倍，最终 output drain 可能更重。是否获益必须由 ATT 中的：

```text
final s_wait_tensorcnt 0 cycles
first-phase TDM 与后半 conversion/store 的 overlap
额外 workgroup barrier cycles
LDS/TDM contention
```

共同判断。首轮只测试 two-phase；不要直接做 four-phase。GEMM1 历史 four-phase 已出现额外
barrier/descriptor 开销抵消收益的情况。

### 2.10 batch-8 exact-SiLU：不适用

GEMM2 `stage1_act=0`，没有：

```text
exp2 -> add -> rcp -> mul
```

因此不存在可以通过 batch-8 隐藏的 TRANS dependency chain。GEMM2 epilogue 只需要 FP32
accumulator 到 BF16 的转换和 output staging。不能为了复用代码而人为引入 batch-8 分组或额外
临时 VGPR。

## 3. 推荐实施顺序

### P0：重新建立 GEMM2 baseline 和 all-SIMD ATT

先使用新脚本建立同一启动状态下的起点：

```bash
ROUNDS=5 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh e2e-const0 --gemm2

ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh e2e-random --gemm2
```

抓取 current baseline：

```bash
CASE_LIST=baseline \
AITER_ATT_SHADER_ENGINE_MASK=0xf \
AITER_ATT_SIMD_LIST=0,1,2,3 \
bash my_code/reproduce_compare.sh att --gemm2
```

P0 必须记录：

- exact GEMM2 symbol 和 ISA hash；
- GEMM2、A2 quant producer、MoE e2e timing；
- VGPR、SGPR、LDS、scratch；
- 每 SIMD active wave/resident WG；
- setup/binary-search/descriptor、initial wait、K hotloop、output staging、final output drain cycles；
- `s_wait_tensorcnt`、`s_wait_dscnt`、`s_barrier_wait` 的 site-level cycles；
- output-owner 与 non-owner wave 的尾部差异。

P0 之前不应对 GEMM1 trace 的阶段占比直接套用到 GEMM2。

### P1：`gemm2_reuse`

最小改动：只允许 GEMM2 selector 把 `wmma_reuse=1` 传入现有 snake traversal。

验收：

```text
random hash128 与 baseline 相同
logits_diff/rel_l2 不变
final ISA reuse topology 合法
无 VGPR/SGPR spill
GEMM2 median 至少改善 max(0.5%, 3*CV)
```

若没有稳定收益，关闭该 bit，不进一步改变 traversal。

### P2：`gemm2_cluster4x4`

新增 GEMM2-only specialization gate，启用 `cluster_m=4` 和现有双向 multicast/sync 协议。

先保持：

```text
WPT1
A row-major
schedule_hints=0
normal XDL arbitration
full output store
```

这样只测 cluster 变量。结构性改动的保留门槛建议为：

```text
GEMM2 >= 3%
MoE e2e 不退化
random/non-balanced 全通过
无 deadlock、无偶发 hash 变化
```

如果 balanced 有收益但 non-balanced 明显回退，不能作为无条件默认；应根据每个 expert 的
M-tile 分布建立 runtime/dispatch gate，而不是删除动态 expert mask。

### P3：`gemm2_cluster4x4_wpt2`

在 P2 winner 上启用 GEMM2-only WPT2，并保留一个 `cluster4x4_wpt1` 对照。检查：

- A/B owner 的 `TENSORcnt` stall 是否趋于一致；
- Scale owner 是否不再过早到达 barrier；
- descriptor setup 增量是否小于 owner-balance 收益；
- 每 wave 两个 in-flight tensor ops 是否未出现新 backpressure。

如果 WPT2 只在 `cluster_m=1` 有效、在 `4x4` 无效，应保留对最终组合更快的版本，不强求与
GEMM1 feature matrix 完全一致。

### P4：`gemm2_apre`

先做最小 producer-consumer 成对接线，不新增 kernel：

```text
A2 quant producer a_preshuffle=True
GEMM2 consumer a_preshuffle=True
```

必须同时报告：

```text
A2 quant producer delta
GEMM2 delta
MoE e2e delta
```

保留门槛建议：

```text
GEMM2 >= 3%
MoE e2e >= 1%
producer 不出现无法解释的 >10% 回退
```

如果 GEMM2 加速但 e2e 退化，直接回滚第一版。只有 consumer 收益足够大且 producer 的 scattered
store 被 trace 证明是唯一阻碍时，才设计 32-row LDS transpose producer。

### P5：在最终 input path 上重扫 schedule hints

候选命名示例：

```text
gemm2_mg2_fc12
gemm2_mg4_fc8
gemm2_mg4_fc20
gemm2_mg4_fc24
gemm2_mg4_fc28
gemm2_mg8_fc24
```

先单变量 sweep，再把 winner 叠加到 P3/P4 保留版本。必须 dump ISA，确认：

- DS reads 没有重新聚集成长 burst；
- tail WMMA group 确实覆盖 fence；
- `s_wait_dscnt`/`s_wait_tensorcnt` stall 与 wall time 同方向；
- VGPR live range 和 spill 未恶化。

### P6：`gemm2_ostore2p`

在最终 input/hotloop winner 上实现两阶段 output TDM。先单独比较：

```text
full-store
two-phase-store
```

保留门槛：

```text
GEMM2 >= 1%
final output wait 显著下降
新增 barrier 没有把收益转移成 s_barrier_wait
output hash128 与 baseline 相同
```

如果后半 BF16 conversion/store 窗口不足以覆盖第一阶段 TDM，应如实判定该方向对 GEMM2 无效，
不要继续增加 phase 数。

### P7：组合 winner，并做逐项回拆

推荐累计命名：

```text
baseline
reuse
reuse_c4
reuse_c4_wpt2
reuse_c4_wpt2_apre          # 仅 P4 净收益为正时
reuse_c4_wpt2_apre_sh
reuse_c4_wpt2_apre_sh_ostore2p
```

最终 winner 必须再做一次逆向 ablation：分别关闭 reuse、WPT2、schedule、output overlap，确认每个
保留机制在组合状态下仍有正贡献。各单项收益不能直接线性相加，因为它们竞争相同的
TDM/DS/WMMA overlap window。

## 4. 代码改造边界

### 4.1 `grouped_gemm_mxfp4.py`

新增严格 GEMM2 target predicate，例如：

```text
a_is_fp4
K == 3072
tile_m/tile_n/tile_k == 256/256/256
m_warp/n_warp == 2/2
num_buffers == 4
stage1_act == 0
stage1_quant_out == 0
out_is_f16 == 0
has_bias == 0
cluster_n == 4
next_stage_prefetch == 1
n_experts == 96
```

新增 GEMM2-only selectors，避免 GEMM1/GEMM2 相互污染：

```text
AITER_FLYDSL_GEMM2_CLUSTER_M
AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM
AITER_FLYDSL_GEMM2_A_PRESHUFFLE
AITER_FLYDSL_GEMM2_SCHEDULE_HINTS
AITER_FLYDSL_GEMM2_MMA_GROUP
AITER_FLYDSL_GEMM2_FENCE_COVER_MMA
AITER_FLYDSL_GEMM2_WMMA_REUSE
AITER_FLYDSL_GEMM2_OVERLAP_OUTPUT_STORE
```

GEMM2 的 `disable_xdl_arb_stall` 默认必须是 0，不能因扩展 specialization gate 而继承 GEMM1
automatic bit-2 行为。

### 4.2 `grouped_moe_gfx1250.py`

当 `AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1` 时，必须同时修改：

```text
flydsl_moe_fused_quant_preshuffle(..., a_preshuffle=True)
flydsl_grouped_gemm_a8w4_masked(..., a_preshuffle=True)  # GEMM2 call
```

只改 producer 或只改 consumer 都会造成 physical layout 不匹配。

GEMM1 调用继续使用自己的 env 和 exactopt 配置；GEMM2 tuning 不能改变 GEMM1 symbol/hash。

### 4.3 `mxfp4_preshuffle_gfx1250_tdm.py`

建议把当前单一的：

```text
fp4_prefill_common
fp4_prefill_schedule
```

拆成：

```text
fp4_gemm1_prefill_common
fp4_gemm2_prefill_common
fp4_tdm_schedule = gemm1_schedule or gemm2_schedule
```

但 activation-only 行为必须继续单独 gate：

```text
epilogue_batch_wn       # 仅 GEMM1 activation
silu_*                  # 仅 GEMM1 activation
GEMM1 output stride / 2 # 仅 GEMM1 activation
```

GEMM2 可共享：

```text
cluster_m=4
cluster_sync
WPT2
A/ScaleA preshuffle
schedule_hints
WMMA reuse
two-phase output framework
```

### 4.4 `reproduce_compare.sh`

当前脚本已支持：

```bash
bash my_code/reproduce_compare.sh --gemm2
bash my_code/reproduce_compare.sh e2e-random --gemm2
bash my_code/reproduce_compare.sh att --gemm2
```

实施候选时，在 `--gemm2` 模式逐步扩展 case table；默认只保留 `baseline`，避免在实现前制造
名义 case。汇总继续以 GEMM2 latency、effective R+W TB/s、TFLOP/s、MoE e2e 和 hash128 为
统一判据。

## 5. 正确性验证矩阵

所有候选必须保持：

- GEMM2 Python/C++ 调用接口不变；
- FP4 payload、E8M0 scale 和 FP32 WMMA accumulation 语义不变；
- BF16 grouped output 不变；
- `m_tile_map` binary search 保留；
- non-balanced expert distribution 兼容；
- gather-reduce 输入布局和 row identity 不变。

最低测试集合：

| 场景 | 目的 |
|---|---|
| balanced E96/T16384/topk6 | 主性能场景 |
| random unbalanced routing | 验证动态 expert mask 和 binary search |
| hot-expert skew | 验证 `4x4` cluster 跨 expert mask |
| empty experts | 验证空区间和下一 expert 边界 |
| 每 expert active rows 为 1/15/16/17/31/32/33/255/256/257 | 验证 OOB、tile 和 super-row 边界 |
| const0 | 稳定性能与未初始化输出检查 |
| random finite values | 主正确性与 hash128 |
| 特殊 BF16/FP4 rounding 输入 | Apre producer bit equivalence |

对于只改变调度、multicast、reuse、output overlap 的候选，要求同一输入下：

```text
MoE output hash128 == baseline MoE output hash128
logits_diff        == baseline logits_diff
rel_l2             == baseline rel_l2
```

对于 Apre，除最终 output 外还要比较：

```text
candidate A2 payload == reference preshuffle(A2 baseline payload)
candidate ScaleA     == reference 32-row preshuffle(ScaleA baseline)
```

padding row 可以保持未定义，但必须证明 GEMM2 从不读取有效 extent 外的数据。

## 6. 性能与 ATT 验收协议

### 6.1 timing

- 只在目标机器所有 GPU/KFD 空闲时记录性能；
- 每个 case 至少 5 个交错 rounds；
- 奇数轮正序、偶数轮逆序；
- 同一启动状态、同一容器、同一 clock/partition；
- 报告 median、全部 samples 和 CV；
- const0 用于稳定 timing，random 用于 correctness 和真实性复核。

### 6.2 ATT 重点

baseline 和每个结构性 winner 都抓 SIMD0–3，重点比较：

```text
initial s_wait_tensorcnt
steady s_wait_tensorcnt
ring-wrap s_barrier_wait
s_wait_dscnt
WMMA hotloop active cycles
BF16 conversion + ds_store_b128 cycles
output TDM issue 到 final tensor_wait(0) 的 tail
```

对 `4x4`：验证 B/ScaleB owner 的 TDM wait 和 cluster barrier；对 WPT2：验证 4 个 wave 的
生命周期是否更接近；对 two-phase output：验证第一阶段 TDM 是否真正与后半 epilogue overlap。

### 6.3 资源

每个候选记录：

```text
.vgpr_count
.sgpr_count
.group_segment_fixed_size
.private_segment_fixed_size
spill count
```

硬门槛：

```text
SGPR <= 106
VGPR <= 1024
LDS  <= 320 KiB/WGP
spill = 0
```

当前已经是 occupancy 1；“资源减少但 occupancy 不变”不能单独作为性能成功证据。

### 6.4 保留/回滚门槛

| 类型 | 建议保留门槛 |
|---|---|
| reuse/schedule 小改动 | `>= max(0.5%, 3*CV)` GEMM2 改善 |
| WPT2 | `>=1%` GEMM2 改善，E2E 不退化 |
| `4x4` cluster | `>=3%` GEMM2 改善，random/non-balanced 全通过 |
| Apre layout | `>=3%` GEMM2 且 `>=1%` E2E 改善 |
| two-phase output | `>=1%` GEMM2，且 final wait 明显下降 |
| 最终组合 | `>=5%` GEMM2，MoE e2e 有稳定正收益 |

任何候选出现以下情况立即回滚：

- random hash 非预期变化；
- non-balanced deadlock 或跨 expert 数据污染；
- TDM descriptor 超限、spill 或 LDS 超限；
- 只提高“计算出来的 TB/s”，但 wall time 没有改善；
- producer 回退抵消 consumer 收益；
- 不同启动状态之间的数字被错误地作为单变量结果比较。

## 7. 预期收益边界

在没有 GEMM2 all-SIMD ATT 和物理 traffic counter 前，不能给出可信的总收益承诺。当前只能给出
方向性判断：

1. `4x4` B/ScaleB multicast 是理论请求减少最大的候选；
2. GEMM2 output 占 effective R+W 的 `52.33%`，output-tail 优化值得优先测量；
3. WPT2 和 WMMA reuse 是实现成本较低的收尾项；
4. schedule hints 需要在最终 layout/cluster 上重新调参；
5. Apre 缺少 GEMM1 的 token-dedup 收益，必须以 producer+consumer e2e 结果决定去留。

一个现实目标是先争取稳定的 `5%～10%` GEMM2 wall-time 改善；是否能超过该范围取决于 P0
确认的 B-side TDM wait 和 output drain 占比。如果 ATT 显示 hotloop 已接近纯 WMMA 吞吐、output
tail 很短且 multicast 主要命中 cache，就应如实降低预期，而不是继续增加无证据的复杂调度。

## 8. 硬件资料依据

1. `MI400_Shader_Programming#65.txt` §4.10.2：
   TDM tensor/tile dimension、stride 与 OOB addressing。
2. `MI400_Shader_Programming#65.txt` §4.10.3：
   `D#.workgroup_mask != 0` 时，`TENSOR_LOAD_TO_LDS` 使用
   `CLUSTER_LOAD_ASYNC` 向多个 WG 的 LDS multicast。
3. `MI400_Shader_Programming#65.txt` §4.10.8：
   每 wave 最多 3 个 tensor ops 从 issue 到 XACK 同时在途，每 SIMD 最多 6 个。
4. `MI400_Shader_Programming#65.txt` §4.7.1：
   LDS 为 64 个 4-byte banks，bank conflict 会降低吞吐。
5. `MI400_Shader_Programming#65.txt` §4.3.7.4.2：
   `DISABLE_XDL_ARB_STALL` 的行为及其对 wave co-execution 的影响。
6. `amd-instinct-cdna5-instruction-set-architecture.txt` §7.12：
   WMMA matrix reuse 的正确性限制；operand 不相同时设置 reuse 会导致 undefined result。

## 9. 最终推荐

第一轮开发只做以下三项：

```text
1. gemm2_reuse
2. gemm2_cluster4x4
3. gemm2_cluster4x4_wpt2
```

原因是它们分别覆盖：

- WMMA operand delivery；
- GEMM2 最大 input surface——B/ScaleB 的重复请求；
- TDM owner imbalance。

三项都有明确的现有代码基础，不需要改变 GEMM2 数学接口。完成三项并取得 all-SIMD ATT 后，再
决定是否投入 Apre producer-consumer 改造和 two-phase output。`batch-8 exact-SiLU`、pad8 和
“每 token 只量化一次”不应进入 GEMM2 开发清单。

## 10. A/ScaleA preshuffle 实施结果（2026-09-16）

### 10.1 最终实现

GEMM2 增加独立开关：

```text
AITER_FLYDSL_GEMM2_A_PRESHUFFLE
```

producer 和 consumer 成对切换。最终 producer 没有采用单 kernel 直接写 preshuffled layout，
而是：

```text
moe_fused_quant_preshuffle_routeks_fd3072_r8_fp4_pk8_srcrow_noKS_compact
  -> 把有效 routes 量化到 compact row-major payload/scale

moe_invert_route_rows
  -> grouped_row -> compact route

moe_scatter_preshuffled_a_fd3072_r32_lds
  -> 32-row LDS transpose，写出最终 A/ScaleA preshuffled layout
```

该实现为目标 shape 增加约 `153.5 MiB` 临时 GPU buffer：compact payload、compact scale 和
`grouped_row -> route` map。它们由 PyTorch caching allocator 管理，不改变外部接口；该内存成本需在
后续更大并发/多实例场景继续评估。

GEMM2 consumer 对应变为：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_wpt1_apre
```

`my_code/reproduce_compare.sh --gemm2` 当前包含：

```text
baseline
apre
```

两者的 GEMM1 都固定为 `sync_mg4_fc28_apre_exactopt`。

### 10.2 被否决的 direct producer

第一版直接启用已有 routeks `_apre` store：

```text
moe_fused_quant_preshuffle_routeks_fd3072_r8_fp4_pk8_srcrow_noKS_apre
```

它保持正确性并使 GEMM2 加速，但一个 warp 的相邻 MX blocks 在目标 layout 中相隔 256B，
global store 合并效率显著下降。d01-3 五轮结果：

```text
GEMM2: 449.699 -> 413.168 us, +8.12%
MoE:   1481.23 -> 1548.18 us, -4.52%
```

producer 从约 `108–111 us` 增长到约 `211–216 us`，完全抵消 consumer 收益，因此该 direct
producer 已被替换，不作为保留实现。

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-d01-3_20260916T123109Z_gemm2_e2e-const0
```

### 10.3 最终 compact + LDS scatter 性能

d01-3、GPU 空闲、五轮交错 const0：

| case | GEMM2 samples (us) | median | GEMM2 gain | MoE samples (us) | median | MoE gain |
|---|---|---:|---:|---|---:|---:|
| `baseline` | `447.851, 450.001, 450.405, 447.853, 448.517` | `448.517` | `0.00%` | `1479.40, 1479.93, 1481.96, 1485.97, 1477.10` | `1479.93` | `0.00%` |
| `apre` | `421.927, 419.991, 421.963, 420.622, 414.977` | `420.622` | `+6.22%` | `1460.90, 1461.33, 1462.11, 1460.04, 1454.22` | `1460.90` | `+1.29%` |

GEMM2 effective metrics：

```text
effective R+W: 5.460 -> 5.822 TB/s
executed:       9652.5 -> 10292.7 TFLOP/s
```

producer 五轮 median 分解：

```text
baseline routeks producer = 108.1 us

compact quant             = 69.1 us
invert                     =  4.6 us
LDS scatter                = 42.5 us
candidate producer total   = 116.2 us
producer delta             = +8.1 us
```

consumer 节省约 `27.9 us`，覆盖 producer 增量后，MoE e2e 净减少约 `19.0 us`。

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-d01-3_20260916T123634Z_gemm2_e2e-const0
```

### 10.4 正确性

d01-3 random-input baseline/apre：

```text
logits_diff = 3.39799e-06
rel_l2      = 0.00260689
pass        = True

MoE output hash128 = 1556fc617347e2dabc9cff19dbfd822b
ref output hash128 = 1a5d22911ba167160b4f2c12092a5193
```

两版 metrics 和 hash 完全一致。const0 同样为零误差且 hash 相同。

random 运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-d01-3_20260916T123537Z_gemm2_e2e-random
```

另外使用：

```text
AITER_REPRO_EXPERT_BALANCE=false
```

完成 non-balanced random 验证，两版同样得到：

```text
logits_diff = 3.48778e-06
rel_l2      = 0.00264113
pass        = True

MoE output hash128 = 10ef188b89c427fde6c8b322b5fd4133
ref output hash128 = d043e1891d95c1af3da5a30e37b9042a
```

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-d01-3_20260916T124252Z_gemm2_e2e-random
```

### 10.5 当前决策

最终 compact + LDS scatter 版本满足：

- producer/consumer physical layout 成对切换；
- GEMM2 接口、计算公式和 FP32 accumulation 不变；
- random/const0 correctness 通过；
- GEMM2 稳定提升 `6.22%`；
- MoE e2e 稳定提升 `1.29%`。

因此保留该 `apre` case。下一步不应继续微调 direct scattered store；更有依据的后续方向是文档
P1/P2/P3 中的 WMMA reuse、`4x4` B/ScaleB multicast 和 WPT2。

## 11. a07-3：producer 融合优化（2026-09-16）

本节是在第 10 节三-kernel producer 基础上的后续优化。第 10 节记录的 compact + invert + LDS
scatter 仍作为兼容 fallback 保留，但不再是 `apre` case 的默认 producer。

### 11.1 优化目标与原始瓶颈

a07-3 上原始结果为：

```text
baseline routeks producer = 69.1 us

compact quant             = 67.7 us
invert                     =  2.4 us
LDS scatter                = 37.6 us
three-kernel total         = 107.6 us
```

三段式实现虽然把 GEMM2 consumer 加速了约 `26.3 us`，但 producer 增加了约 `38.5 us`，因此
MoE e2e 出现回退。仅优化 `invert` 或 launch overhead 不足以达到目标，必须消除 compact
payload/scale 的完整 global-memory 写回和再次读取。

### 11.2 最终 fused rowgroup producer

默认 `apre` producer 已替换为单 kernel：

```text
moe_quant_preshuffled_a_fd3072_rpw2_pf2_direct
```

其主要设计如下：

1. 直接读取 GEMM1 已经生成的 grouped BF16 rows，不再生成 route-order compact payload/scale。
2. 直接写最终 GEMM2 A/ScaleA preshuffled layout，删除中间 compact buffer、inverse map 和 LDS
   scatter kernel。
3. gfx1250 的一个 wave 包含 8 个 4-lane MX32 subgroup。最终配置使用
   `rows_per_wave=2`，每个 wave 同时覆盖 2 个相邻 row 和 4 个 K block，使相同 K block 的
   payload 地址在 wave 内形成连续的 32B 区间，而不是原 warp-per-row direct producer 的离散
   16B store。
4. 使用 `prefetch_depth=2`：先发出两个独立 `buffer_load_b128`，再消费其结果，减少 ISA 中每次
   load 后立即 `s_wait_loadcnt 0` 所造成的串行等待。
5. 每个 `expert_tile_m=256` tile 通过 `m_tile_map` binary search 查找所属 expert，再用
   `row < m_tile_map[expert]` 屏蔽 expert padding 和 capacity tail。因此没有假设每个 expert 的
   token 数相同，保留 non-balanced 路由语义。
6. 量化继续使用与原 routeks producer 相同的 `emit_mx_e8m0_scale`、`_ROUND_MODE`、
   `v_cvt_scalef32_pk8_fp4_bf16` 和 4-lane MX32 reduction，未降低精度。

被删除的目标 shape 临时存储包括：

```text
compact payload
compact scale
grouped_row -> compact route map
```

合计约 `153.5 MiB`。对于不满足 rowgroup 约束的 shape，代码自动回退到第 10 节的
three-kernel producer，以保持通用功能兼容。

相关调试/回退环境变量：

```text
AITER_REPRO_GEMM2_APRE_PRODUCER=rowgroup|three_kernel
AITER_REPRO_GEMM2_APRE_RPW=1|2|4|8
AITER_REPRO_GEMM2_APRE_PREFETCH=1|2|4|8
```

最终默认值为：

```text
producer = rowgroup
rows_per_wave = 2
prefetch_depth = 2
```

### 11.3 mapping 与 prefetch 搜索结果

在 a07-3 上得到的关键方向性结果：

| variant | producer time | 结论 |
|---|---:|---|
| three-kernel | 约 `107.6 us` | compact global round-trip 成本过高 |
| rowgroup `rpw8`，未打包 ScaleA | 约 `83.1 us` | 128B 连续 payload store 有效，但循环较长 |
| rowgroup `rpw4` | median `77.8 us` | 优于 rpw8，但仍高于 baseline producer |
| rowgroup `rpw2`, `prefetch=1` | median `68.1 us` | 首次低于 baseline，但余量较小 |
| rowgroup `rpw2`, `prefetch=4` | 初测约 `74.1 us` | VGPR/live-range 成本超过 latency overlap 收益 |
| rowgroup `rpw2`, `prefetch=2` | median `66.6 us` | 最终保留版本 |

`rpw8` 路径还验证了将连续四个 E8M0 byte 在 VGPR 中打包后使用 dword store；它从约
`83.1 us` 降至短测约 `72.8 us`，但仍不及 `rpw2 + prefetch2`，所以没有作为默认配置。

### 11.4 a07-3 最终五轮性能

测试条件：GPU/KFD 空闲，五轮正反序交错，`const0`，GEMM1 固定为
`sync_mg4_fc28_apre_exactopt`。命令为：

```bash
CASE_LIST=baseline,apre ROUNDS=5 E2E_ITERS=20 RUN_VERIFY=1 RUN_ATT=0 \
  bash my_code/reproduce_compare.sh --gemm2
```

producer：

| producer | samples (us) | median | 相对 baseline producer |
|---|---|---:|---:|
| baseline routeks | `69.4, 69.0, 69.2, 68.9, 69.5` | `69.2` | `0.00%` |
| fused rowgroup | `66.6, 66.5, 66.6, 66.5, 66.7` | `66.6` | `+3.76%` |

与旧三段 producer 相比，耗时从 `107.6 us` 降至 `66.6 us`：

```text
降低 38.10%
加速 1.62x
```

GEMM2 consumer 与 MoE e2e：

| case | GEMM2 samples (us) | median | GEMM2 gain | GEMM2 effective R+W | GEMM2 executed | MoE samples (us) | median | MoE gain |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `baseline` | `396.096, 397.630, 396.871, 398.096, 397.067` | `397.067` | `0.00%` | `6.168 TB/s` | `10903.3 TFLOP/s` | `1343.69, 1348.25, 1346.72, 1346.87, 1342.66` | `1346.72` | `0.00%` |
| `apre` fused | `367.764, 367.956, 367.383, 368.544, 368.201` | `367.956` | `+7.33%` | `6.656 TB/s` | `11765.9 TFLOP/s` | `1313.90, 1317.29, 1312.94, 1319.65, 1318.27` | `1317.29` | `+2.19%` |

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260916T154625Z_gemm2_e2e-const0
```

### 11.5 功能等价性验证

目标 shape 的 balanced const0：

```text
pass = True
logits_diff = 0
rel_l2 = 0
MoE output hash128 = 21291d9023c8af8a6324fe20f346a967
ref output hash128 = 21291d9023c8af8a6324fe20f346a967
```

目标 shape 的 non-balanced random：

```text
pass = True
logits_diff = 3.48778e-06
rel_l2 = 0.00264113
MoE output hash128 = 10ef188b89c427fde6c8b322b5fd4133
ref output hash128 = d043e1891d95c1af3da5a30e37b9042a
```

此外，直接比较 three-kernel 与 fused rowgroup producer 的完整输出 buffer：

| case | routing | K | payload diff | ScaleA diff |
|---|---|---:|---:|---:|
| E=8，512 routes | balanced | 1024 | 0 | 0 |
| E=8，454 routes | non-balanced | 2048 | 0 | 0 |
| E=8，2706 routes | non-balanced | 3072 | 0 | 0 |

因此最终版本不仅保持 MoE tolerance，还对所有被读取的 A payload/ScaleA 数据保持逐字节等价。

## 12. GEMM2 consumer 后续优化记录（a07-3，2026-09-16）

本节以第 11 节的 fused rowgroup producer 和 A/ScaleA preshuffle consumer 为起点，按本文
P1/P2/P3/P5/P6 的顺序继续优化 GEMM2。所有性能数据均来自测试开始前确认 GPU/KFD 空闲的
a07-3；发生其他任务中途抢占时，该轮数据作废，不参与结论。

### 12.1 P1：WMMA operand reuse

新增 GEMM2-only `AITER_FLYDSL_GEMM2_WMMA_REUSE`，复用现有 snake traversal 中已经满足
operand 邻接约束的位置：

```text
reuseA = wn_raw > 0
reuseB = wm > 0 and wn_raw == 0
```

random 输入正确性通过，输出 hash 与无 reuse 版本一致。但在 WPT2 最终组合上的逆向消融为：

```text
reuse=0: 332.999 us
reuse=1: 332.858 us
delta:   0.04%
```

该差异低于 `0.5%` 噪声门槛，因此不将 reuse 加入默认 GEMM2 winner。

### 12.2 P2：4x4 cluster 与双向 multicast

新增 GEMM2-only `cluster_m=4` 实验路径，复用现有：

- A/ScaleA 沿 N 方向 multicast；
- B/ScaleB 仅在同一 expert 的 M peers 间 multicast；
- full cluster startup/ring-wrap barrier；
- 跨 expert 和 sentinel tail 的安全 mask/fallback。

random 正确性通过且未发生 barrier hang，但三轮 const0 结果为：

| case | GEMM2 median | MoE e2e median |
|---|---:|---:|
| `apre`，`cluster_m=1` | `368.992 us` | `1316.51 us` |
| `apre_cluster4x4` | `395.952 us` | `1344.82 us` |

`cluster_m=4` 相对 1-D `apre` 回退约 `7.31%`。GEMM2 仅有 12 个 K256 tile，当前 workload
无法摊薄额外的 2-D cluster synchronization，因此该方向不进入默认组合。

### 12.3 P3：WPT2 owner balance

新增 GEMM2-only：

```text
AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=2
```

在 `cluster_m=1`、reuse 关闭、schedule hints 关闭时，三轮 const0 结果为：

| case | GEMM2 samples (us) | median | MoE e2e samples (us) | median |
|---|---|---:|---|---:|
| `apre` WPT1 | `368.888, 368.133, 368.424` | `368.424` | `1318.85, 1314.92, 1316.31` | `1316.31` |
| `apre_wpt2` | `359.797, 359.803, 360.322` | `359.803` | `1312.51, 1310.83, 1306.83` | `1310.83` |

WPT2 在该组合中的收益：

```text
GEMM2: +2.34%
MoE:   +0.42%
```

在最终 schedule/output 组合上关闭 WPT2 后，GEMM2 从 `332.999 us` 回退到 `342.004 us`，
进一步确认 WPT2 的独立贡献约为 `2.63%`，因此保留。

### 12.4 P5：schedule hints sweep

在 `A preshuffle + fused producer + WPT2 + cluster_m=1 + reuse=0` 上扫描：

```text
MMA_GROUP       = 2, 4, 8
FENCE_COVER_MMA = 8, 12, 20, 24, 28
```

单轮筛选结果：

| setting | GEMM2 us | MoE e2e us |
|---|---:|---:|
| `mg2/fc8` | 344.797 | 1287.28 |
| `mg2/fc12` | 343.472 | 1286.17 |
| `mg2/fc20` | 343.579 | 1287.22 |
| `mg2/fc24` | 342.390 | 1286.17 |
| `mg2/fc28` | 342.560 | 1285.40 |
| `mg4/fc8` | 341.565 | 1292.48 |
| `mg4/fc12` | 338.480 | 1281.89 |
| `mg4/fc20` | 339.028 | 1282.96 |
| `mg4/fc24` | 338.599 | 1282.20 |
| `mg4/fc28` | 338.487 | 1281.87 |
| `mg8/fc8` | 339.404 | 1284.04 |
| `mg8/fc12` | 339.434 | 1283.10 |
| `mg8/fc20` | 338.442 | 1282.16 |
| `mg8/fc24` | 339.132 | 1283.69 |
| `mg8/fc28` | 337.407 | 1280.46 |

对前四名进行五轮交错复测：

| case | GEMM2 samples (us) | median | MoE e2e samples (us) | median |
|---|---|---:|---|---:|
| `apre_wpt2`，无 hints | `359.914, 359.922, 359.403, 358.781, 360.807` | `359.914` | `1310.47, 1308.92, 1309.53, 1304.32, 1310.62` | `1309.53` |
| `mg4/fc12` | `338.878, 338.195, 338.983, 338.980, 337.216` | `338.878` | `1287.15, 1284.66, 1290.79, 1286.17, 1284.94` | `1286.17` |
| `mg4/fc28` | `336.715, 337.594, 336.906, 337.009, 337.587` | `337.009` | `1281.82, 1286.14, 1285.14, 1286.51, 1289.10` | `1286.14` |
| `mg8/fc20` | `339.658, 338.977, 337.585, 338.987, 338.871` | `338.977` | `1283.94, 1285.86, 1288.02, 1284.45, 1286.09` | `1285.86` |
| `mg8/fc28` | `338.165, 338.104, 336.341, 337.897, 338.130` | `338.104` | `1285.37, 1288.59, 1286.85, 1283.33, 1284.52` | `1285.37` |

`mg4/fc28` 与 `mg8/fc28` 的差异小于 `0.5%`，不能证明两者存在统计显著差异。最终选择
`mg4/fc28`，原因是其中位数最低、样本分布更集中，并与已经验证过的 GEMM1 scheduling 形状一致。

五轮运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260916T172451Z_gemm2_e2e-const0
```

### 12.5 P6：two-phase output TDM

GEMM2 passthrough epilogue 的两阶段 output TDM 已接通：

```text
out_stride  = i32_n
out_col_off = blk_n
```

第一阶段在部分 WM rows 完成 BF16 conversion 和 LDS staging 后发出 output TDM，剩余 rows 的
conversion/store 与第一阶段 TDM 重叠；第二阶段完成后统一执行最终 `tensor_wait(0)`。

对 `output_store_split_wm=1..7` 的单轮 sweep：

| split WM | first slice rows/wave-M | GEMM2 us | MoE e2e us |
|---:|---:|---:|---:|
| 1 | 16 | 337.156 | 1280.87 |
| 2 | 32 | 333.213 | 1278.35 |
| 3 | 48 | 331.714 | 1275.52 |
| 4 | 64 | 333.806 | 1280.23 |
| 5 | 80 | 335.338 | 1277.05 |
| 6 | 96 | 336.490 | 1278.31 |
| 7 | 112 | 335.020 | 1277.21 |

对 split2/3/4 进行五轮复测：

| case | GEMM2 samples (us) | median | MoE e2e samples (us) | median |
|---|---|---:|---|---:|
| full-store | `337.636, 338.214, 337.592, 337.499, 338.140` | `337.636` | `1280.01, 1285.58, 1284.84, 1289.40, 1286.33` | `1285.58` |
| split2 | `334.303, 334.221, 332.951, 334.077, 333.628` | `334.077` | `1279.64, 1283.88, 1284.94, 1279.56, 1281.98` | `1281.98` |
| split3 | `334.284, 332.245, 332.557, 332.757, 333.816` | `332.757` | `1282.01, 1280.28, 1279.28, 1286.11, 1284.39` | `1282.01` |
| split4 | `335.134, 333.780, 333.238, 333.301, 334.225` | `333.780` | `1279.94, 1279.91, 1280.50, 1282.38, 1283.22` | `1280.50` |

按 GEMM2 主指标选择 split3：相对 full-store 提升 `1.45%`。split4 的 e2e 中位数略低，但差值
只有约 `1.5 us`，属于系统噪声量级，不能据此牺牲 GEMM2 kernel 的稳定优势。

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260916T174453Z_gemm2_e2e-const0
```

### 12.6 当前最终组合及逆向消融

当前 winner：

```text
A/ScaleA preshuffle producer = rowgroup rpw2/prefetch2
GEMM2 TDM owners             = WPT2
schedule hints               = mg4/fc28
WMMA reuse                   = 0
cluster_m                    = 1
output TDM                   = two-phase, split_wm=3
```

三轮逆向消融：

| case | GEMM2 median | 相对最终版 |
|---|---:|---:|
| final | `332.999 us` | `0.00%` |
| WPT1 | `342.004 us` | `-2.70%` |
| reuse=1 | `332.858 us` | `+0.04%`，噪声范围 |

因此保留 WPT2，关闭 reuse；`cluster_m=4` 同样保持关闭。与该轮 baseline `397.740 us` 相比，
最终 GEMM2 为 `332.999 us`，提升 `16.28%`；MoE e2e 从 `1345.00 us` 降到
`1280.91 us`，提升 `4.77%`。

### 12.7 output wave split：当前稳定 winner

在 two-phase output TDM 的每个 M slice 内，原实现只让 `wave_n==0` 发出覆盖全部 rows 的
descriptor。新实现保持 descriptor 的 N extent、global stride、LDS pitch 和两阶段边界不变，只把
每个 slice 的 M rows 均分给两个 `wave_n`：

```text
wave_n=0 -> slice 的前半 rows
wave_n=1 -> slice 的后半 rows
```

因此 4 个 wave 都参与 output TDM，未改变 output 地址、数据类型或有效 extent。balanced const0
五轮同轮结果：

| case | GEMM2 samples (us) | median | MoE e2e median |
|---|---|---:|---:|
| two-phase, single output wave | `333.446, 333.431, 333.515, 331.976, 333.089` | `333.431` | `1279.36 us` |
| two-phase, two output waves | `330.138, 330.355, 330.878, 328.630, 330.222` | `330.222` | `1277.73 us` |

GEMM2 稳定提升约 `0.96%`。另一组五轮复测为：

```text
330.432, 330.029, 329.799, 330.622, 329.968 us
median = 330.029 us
```

random 与 const0 均通过，最终保留 case：

```text
apre_wpt2_mg4_fc28_ostore2p_ow2
```

ATT 对比目录：

```text
single-wave output:
  .codex_tmp/gemm2_att_final_baseline_all_20260916T190653Z
two-wave output:
  .codex_tmp/gemm2_att_output_wave_split_20260916T192319Z
```

trace 中 WMMA 动态 latency 占比仍约 `55%～58%`，但 output-owner 的 final
`s_wait_tensorcnt 0` 更均衡，关键 SIMD 的长尾缩短。这个结果说明收益来自分散 output TDM
ownership，而不是减少 output bytes 或改变 BF16 conversion。

### 12.8 已拒绝实验总表

下面所有实验都以 `apre_wpt2_mg4_fc28_ostore2p_ow2` 或它的直接前驱为对照。除明确标为
错误的版本外，random/const0 的 `logits_diff`、`rel_l2` 和 hash 均与对照一致。失败版本的源码均已
删除，不进入默认脚本。

| 实验 | 性能结果 | ATT 主要变化 | 结论 |
|---|---|---|---|
| output N-column split | random 出现 NaN、漏写及 padding 越界写 | active span 略增，barrier 占比也未下降 | descriptor/LDS slice 语义错误，禁止复用 |
| direct global scales | 约 `683.6 us`，对照约 `376.0 us` | 平均 active span `35,319 cycles`；`s_wait_loadcnt` 约 `11.14%`，另有长 `s_wait_loadcnt_dscnt` | ScaleA/ScaleB 从 TDM/LDS 改成逐 wave global load 破坏了隐藏延迟能力 |
| ring buffer b3 | `428.122 us`，对照 `372.599 us` | active span `18,337 cycles`；`s_wait_tensorcnt=17.29%`、`s_barrier_wait=15.49%` | 少一个 stage 使 steady-state TDM/barrier 等待显著增加 |
| 4-buffer initial prefetch=3 | const0 表面可运行，random 出现 NaN | active span `15,388 cycles`；等待未形成稳定下降 | generation fence 不完整，属于错误 pipeline，不得采用 |
| binary search 固定 7 steps | `328.838 us`，对照 `330.029 us`，仅快 `0.36%`；e2e 反而回退 | active span `15,511 cycles`，等待占比未改善 | 收益低于噪声门槛，恢复通用 binary search |
| `early_timeout=0` | `375.361 us`，对照 `365.626 us` | active span 与对照接近，但 tensor/barrier wait 没有下降 | 当前 multicast 的 early return 有效，保留默认值 |
| `tile_m=512 / 8-wave WG` | 约 `3.136 ms`，并改变误差/hash | active span `77,977 cycles`，`s_wait_loadcnt=47.25%` | 资源与同步压力失控，且未满足等价性要求 |
| `TILES_PER_GROUP=4/8/32` | `384.814 / 393.587 / 397.794 us`，对照 `377.243 us` | `tpg4` active span `15,535 cycles`，未降低关键 wait | 降低调度分散度反而恶化负载均衡 |
| three-phase output | `329.226 us`，对照 `330.107 us`，仅快 `0.27%`；e2e 略退 | active span `15,495 cycles`；barrier/dscnt 占比上升 | 多一个 phase 的收益低于门槛，不保留 |
| `m_warp=4,n_warp=1` | `410.894 us`，对照 `392.426 us` | active span `16,810 cycles`；`s_wait_dscnt=6.68%` | 去掉 output 跨-wave barrier 不足以抵消更差的 operand/load mapping |
| output tail descriptor split | `387.820 us`，对照 `373.038 us` | active span `15,506 cycles`；barrier 占比上升 | 额外 descriptor/setup 成本高于 tail 缩短收益 |
| `DISABLE_XDL_ARB_STALL=1` | `411.936 us`，对照 `391.826 us` | active span `15,919 cycles`；WMMA 占比降到 `46.35%`，barrier/load wait 增加 | normal XDL arbitration 对 4-wave GEMM2 更合适 |

这里的 ATT 百分比是被抓取 active waves 的动态 latency 汇总，不等同于可直接相加的 wall-time
占比；它用于判断 stall 类型和相对变化。各 trace 目录为：

```text
.codex_tmp/gemm2_att_failed_nsplit_20260916T191558Z
.codex_tmp/gemm2_att_direct_scales_b4_20260916T193639Z
.codex_tmp/gemm2_att_ring_b3_20260916T195837Z
.codex_tmp/gemm2_att_initial_prefetch3_invalid_20260916T200139Z
.codex_tmp/gemm2_att_tight_bisect7_20260916T200931Z
.codex_tmp/gemm2_att_early_timeout0_20260916T202129Z
.codex_tmp/gemm2_att_tilem512_20260916T203413Z
.codex_tmp/gemm2_att_tpg4_20260916T204427Z
.codex_tmp/gemm2_att_output_three_phase_20260916T205404Z
.codex_tmp/gemm2_att_m4n1_20260916T211435Z
.codex_tmp/gemm2_att_output_tail_split_20260916T212418Z
.codex_tmp/gemm2_att_xdl_arb_off_20260916T213532Z
```

### 12.9 WMMA reuse 拆分复测

为区分 `matrix_a_reuse` 和 `matrix_b_reuse`，分别测试 A-only、B-only、A+B。三个版本 random
均通过，ISA 也确认 reuse bit 只出现在相邻 operand 确实相同的位置。B-only 是单轮筛选中唯一看似
有利的版本，但五轮交错复测为：

| case | GEMM2 samples (us) | median | MoE e2e median |
|---|---|---:|---:|
| no reuse | `330.540, 329.821, 329.222, 330.365, 329.633` | `329.821` | `1275.73 us` |
| B-only reuse | `330.064, 329.332, 330.951, 329.098, 330.422` | `330.064` | `1278.01 us` |

相对 no-reuse，B-only 慢约 `0.07%`，属于噪声且 e2e 更差。四个 SIMD 的 ATT 平均值：

| variant | active span median | WMMA latency | `s_wait_tensorcnt` | `s_barrier_wait` |
|---|---:|---:|---:|---:|
| no reuse | `15,251 cycles` | `57.04%` | `11.89%` | `8.06%` |
| A-only | `15,279 cycles` | `56.95%` | `11.88%` | `8.15%` |
| B-only | `15,280 cycles` | `57.59%` | `11.56%` | `8.15%` |
| A+B | `15,356 cycles` | `57.19%` | `11.75%` | `8.37%` |

符合预期的是 reuse bit 没有增加 WMMA 指令数，也没有破坏精度；不符合预期的是它没有缩短 WMMA
动态 latency 或关键 wave span，反而让 A+B 的 span 增加约 `0.68%`。说明当前 GEMM2 的瓶颈不是
相邻 WMMA operand scoreboard reuse，继续改 traversal 只会增加正确性风险。相关 selector 和测试 case
已删除。

trace：

```text
.codex_tmp/gemm2_att_reusea_20260916T214948Z
.codex_tmp/gemm2_att_reuseb_20260916T215058Z
.codex_tmp/gemm2_att_reuseab_20260916T215202Z
```

### 12.10 `tile_k=512 / b2`：减少 K-loop 次数失败

该实验把 GEMM2 从 12 个 K256 tiles 改为 6 个 K512 tiles，并把 ring 从 4 槽改为 2 槽。接口、
FP4/Scale 布局、WMMA 数量、binary search 和 output epilogue 均不变。random 与 const0 都通过，
两种输入的误差和 hash 与 winner 一致。

五轮同轮结果：

| case | GEMM2 samples (us) | median | MoE e2e median |
|---|---|---:|---:|
| K256/b4 winner | `331.356, 329.256, 329.489, 328.349, 330.717` | `329.489` | `1277.00 us` |
| K512/b2 | `357.473, 357.242, 357.857, 357.648, 358.221` | `357.648` | `1306.57 us` |

K512/b2 慢约 `8.55%`。ATT 四 SIMD 平均：

| metric | K256/b4 | K512/b2 | 变化 |
|---|---:|---:|---:|
| active span median | `15,251` | `17,422 cycles` | `+14.23%` |
| WMMA latency share | `57.04%` | `49.00%` | 被等待时间稀释 |
| `s_wait_tensorcnt` | `11.89%` | `16.36%` | `+4.47 pp` |
| `s_barrier_wait` | `8.06%` | `9.72%` | `+1.66 pp` |
| `s_wait_dscnt` | `3.19%` | `6.88%` | `+3.69 pp` |

资源为：

```text
K256/b4: LDS 278528 B, VGPR 804, SGPR 58
K512/b2: LDS 278528 B, VGPR 755, SGPR 62
```

符合预期的是 VGPR 从 804 降至 755；不符合预期的是 occupancy 仍为 1，且两个更大的 K512 stage
失去 next-stage prefetch 后显著拉长 tensor/DS/barrier 等待。减少 loop/control 指令无法补偿该回退，
因此该路径已删除。

trace：

```text
.codex_tmp/gemm2_att_tilek512_b2_20260917T025903Z
```

### 12.11 B/ScaleB `TH_HT`：近端 high-temporal 无收益

动机是同一 expert 的多个 M tiles 会读取相同 B/ScaleB，而 A/ScaleA 随 M tile 变化。实验仅对
B 和 ScaleB 的 input TDM 设置 `cache_modifier=2`，最终 ISA 中对应为：

```text
tensor_load_to_lds ... th:TH_LOAD_HT
```

A/ScaleA、地址计算、descriptor extent、WPT2 ownership、barrier 和 WMMA 顺序均未改变。random
正确性通过，误差与 hash 和 winner 完全相同。五轮 const0：

| case | GEMM2 samples (us) | median | MoE e2e median |
|---|---|---:|---:|
| default cache policy | `329.452, 329.691, 330.222, 330.011, 329.838` | `329.838` | `1277.44 us` |
| B/ScaleB `TH_HT` | `329.832, 330.544, 329.642, 330.871, 330.496` | `330.496` | `1277.90 us` |

候选慢约 `0.20%`，没有稳定收益。ATT 四 SIMD 平均：

| metric | default | B/ScaleB `TH_HT` |
|---|---:|---:|
| active span median | `15,251` | `15,692 cycles` |
| WMMA latency share | `57.04%` | `56.50%` |
| `s_wait_tensorcnt` | `11.89%` | `12.40%` |
| `s_barrier_wait` | `8.06%` | `8.43%` |
| `s_wait_dscnt` | `3.19%` | `3.15%` |

符合预期的是 cache hint 不改变指令数、资源和结果；不符合预期的是 B/ScaleB 的复用没有转化为
更短的 tensor wait。MI450 ISA 的 TH 编码表说明 `TH_HT=2` 是 high-temporal；结合当前 kernel
约 272 KiB LDS 占用，近端 WGP$ 可用容量有限，把 B/ScaleB 同时标成近端 high-temporal 并未改善
实际命中，反而轻微拉长了 active span。该版本不保留。

trace：

```text
.codex_tmp/gemm2_att_bscale_ht_20260917T035732Z
```

### 12.12 B/ScaleB `TH_NT_HT`：远端 high-temporal 仍然回退

`TH_HT` 可能在 LDS 已占约 272 KiB 时挤压 WGP$，因此继续测试更精确的
`TH_NT_HT=6`：近端 CU/SE cache non-temporal，远端 MALL high-temporal。根据 CDNA5
`CPol` 定义：

```text
TH_RT    = 0
TH_HT    = 2
TH_NT_RT = 4
TH_NT_HT = 6
```

该实验仍只修改 B/ScaleB descriptor 的 TH，A/ScaleA 保持默认。random 正确性通过，误差和最终
MoE hash 与 winner 相同。五轮 const0：

| case | GEMM2 samples (us) | median | MoE e2e median |
|---|---|---:|---:|
| default cache policy | `328.565, 328.824, 328.951, 329.358, 328.679` | `328.824` | `1272.99 us` |
| B/ScaleB `TH_NT_HT` | `331.765, 332.147, 332.406, 331.264, 332.208` | `332.147` | `1276.82 us` |

GEMM2 回退约 `1.01%`。诊断 ATT 四 SIMD 平均：

| metric | default | B/ScaleB `TH_NT_HT` |
|---|---:|---:|
| active span median | `15,251` | `15,199 cycles` |
| active span mean | `12,602` | `12,911 cycles` |
| active span max | `26,381` | `27,030 cycles` |
| `s_wait_tensorcnt` | `11.89%` | `11.81%` |
| `s_barrier_wait` | `8.06%` | `7.95%` |
| `s_wait_dscnt` | `3.19%` | `2.95%` |

符合预期的是部分 wave 的 wait 占比和 median span 略降；不符合预期的是 mean/max span 增加，说明
cache hint 只让一部分请求受益，同时加重了长尾。wall time 明确回退，不能用局部 trace 改善替代整体
性能结论。

另需说明：生产 JIT 路径的 random/const0 均正确，但把带 `TH_NT_HT` 的文本 ISA 重命名后交给
standalone replay，preflight 出现 missing/unexpected writes。为满足诊断要求，ATT 使用
`AITER_TRACE_ALLOW_INCORRECT=1` 抓取；因此本节 trace 只用于观察调度和等待趋势，不作为正确性
证据。正确性结论来自未改写 ISA 的真实 MoE 调用。该版本不保留。

trace：

```text
.codex_tmp/gemm2_att_bscale_nt_ht_diag_20260917T041104Z
```

### 12.13 output quadrant split：修正正确性后仍无稳定收益

该方向试图消除 output 两阶段中的跨-wave workgroup barrier：每个 wave 本来就计算一个互不重叠的
`M128 x N128` quadrant，因此理论上可以只等待本 wave 的 LDS store，再由本 wave 发 output TDM。

第一版直接让 TDM 从原来的共享 row-major output LDS 中读取 128-column 子区。random 出现 NaN，
单轮约 `398.2 us`。原因是 TDM store 的 LDS tile 是线性迭代的；只把 tensor inner extent 改成
128 不能表达共享 LDS 中 `STORE_PITCH=272` 的跨 row 间隔，导致下一 row 的 source 起点错误。

第一版诊断 trace：

```text
.codex_tmp/gemm2_att_output_quadrant_bad_20260917T042324Z
```

其 active span median 为 `15,564 cycles`，比 winner 增加约 `2.05%`；
`s_wait_tensorcnt` 也从 `11.89%` 增至 `13.07%`。因此即使忽略错误结果，也没有性能证据。

第二版为每个 wave 分配独立的连续 LDS quadrant：

```text
quadrant logical N = 128 BF16
quadrant pad       = 16 BF16
quadrant pitch     = 144 BF16
per-wave LDS       = 128 * 144 * 2 B = 36 KiB
4-wave total       = 144 KiB
```

TDM descriptor 的 inner tile 为 144，OOB extent 为 128；这样 padding 被丢弃，下一 row 的 LDS
起点也正确。random/const0 均通过，hash 与 winner 相同。五轮 const0：

| case | GEMM2 samples (us) | median | MoE e2e median |
|---|---|---:|---:|
| shared LDS row split | `734.185, 330.062, 330.273, 330.654, 329.817` | `330.273` | `1275.78 us` |
| private quadrant LDS | `338.822, 330.466, 329.809, 328.168, 329.743` | `329.809` | `1277.41 us` |

首轮样本存在系统级异常长尾，使用中位数后 quadrant 仅快约 `0.14%`，而 e2e 略慢。ATT：

| metric | winner | private quadrant |
|---|---:|---:|
| active span median | `15,251` | `15,252 cycles` |
| active span mean | `12,602` | `13,131 cycles` |
| active span max | `26,381` | `23,139 cycles` |
| `s_wait_tensorcnt` | `11.89%` | `11.44%` |
| `s_barrier_wait` | `8.06%` | `8.89%` |
| `s_wait_dscnt` | `3.19%` | `2.95%` |

符合预期的是 final tensor wait 略降、max span 缩短；不符合预期的是整体 median 没变，mean 和
barrier 占比反而增加。说明 output barrier 并不是当前 barrier latency 的主要来源，hotloop 的 wave
到达偏差仍占主导。该复杂布局低于 `0.5%` 保留门槛，已删除。

trace：

```text
.codex_tmp/gemm2_att_output_quadrant_private_20260917T043227Z
```

### 12.14 all-input `TH_NT_RT`：GEMM1 经验不能直接迁移

GEMM1 persistent 路径曾从 all-input `TH_NT_RT` 获益，因此在 GEMM2 winner 上把 A/B/ScaleA/ScaleB
的 input TDM 全部设置为 `cache_modifier=4`。random 与 const0 均保持正确。

五轮 const0：

| case | GEMM2 samples (us) | median | MoE e2e median |
|---|---|---:|---:|
| default input TH | `330.343, 330.493, 402.312, 565.450, 328.193` | `330.493` | `1281.67 us` |
| all-input `TH_NT_RT` | `355.220, 356.224, 371.535, 358.457, 382.419` | `358.457` | `1306.45 us` |

虽然该轮被其他任务插入了两个明显 outlier，但 default 的正常样本仍约 `328～330 us`，而
`TH_NT_RT` 五个样本全部在 `355 us` 以上，回退约 `8.46%`，结论明确。ATT：

| metric | default | all-input `TH_NT_RT` |
|---|---:|---:|
| active span median | `15,251` | `16,658 cycles` |
| active span mean | `12,602` | `13,789 cycles` |
| WMMA latency share | `57.04%` | `54.97%` |
| `s_wait_tensorcnt` | `11.89%` | `12.01%` |
| `s_wait_dscnt` | `3.19%` | `3.84%` |
| `s_wait_kmcnt` | `2.38%` | `2.96%` |

GEMM2 的 B/ScaleB 占 input surface 绝大部分，并在同 expert 的 M tiles 间存在实际 cache reuse；把它们
标记为近端 non-temporal 破坏了这种复用。该结果也解释了为什么不能把 GEMM1 的 all-input hint
机械迁移到 GEMM2。版本已删除。

trace：

```text
.codex_tmp/gemm2_att_all_input_nt_rt_20260917T044223Z
```

### 12.15 output TDM `TH_NT_RT`：小幅表面收益低于保留门槛

input hint 失败后，最后测试 output-only `TH_NT_RT=4`。它只修改两条 two-phase
`TENSOR_STORE_FROM_LDS`：

```text
tensor_store_from_lds ... th:TH_STORE_NT_RT
```

目的为 output 大流量绕开近端 cache，同时在远端保持 regular temporal，使紧随其后的
`moe_gather_reduce` 仍有复用机会。random/const0 均通过，结果和 hash 不变。

五轮 const0：

| case | GEMM2 samples (us) | median | MoE e2e median |
|---|---|---:|---:|
| default output TH | `335.386, 335.590, 335.638, 336.246, 336.324` | `335.638` | `1291.75 us` |
| output `TH_NT_RT` | `334.460, 334.561, 335.006, 334.327, 334.759` | `334.561` | `1289.44 us` |

表面上 GEMM2 快约 `0.32%`，e2e 快约 `0.18%`，但低于此前统一采用的 `0.5%` promotion
门槛。ATT 也没有支持稳定加速：

| metric | default | output `TH_NT_RT` |
|---|---:|---:|
| active span median | `15,251` | `15,808 cycles` |
| active span mean | `12,602` | `14,239 cycles` |
| active span max | `26,381` | `31,403 cycles` |
| `s_wait_tensorcnt` | `11.89%` | `13.29%` |
| `s_barrier_wait` | `8.06%` | `8.22%` |

单轮 random 曾出现约 `2.15%` 的表面改善，但与五轮 const0 和 ATT 不一致，判断为运行顺序/系统
噪声放大。由于 trace 中 tensor wait 和长尾均变差，不能把 `0.32%` 中位数差异认定为可靠优化。
该 hint 未合入默认 winner，相关 selector 和 case 已删除。

trace：

```text
.codex_tmp/gemm2_att_output_nt_rt_20260917T065725Z
```

### 12.16 当前停止点与进一步优化难点

当前稳定 winner 仍为：

```text
apre_wpt2_mg4_fc28_ostore2p_ow2
```

经过本轮实验，剩余瓶颈不能再由一个低风险 scheduling/cache knob 消除：

- WMMA 动态 latency 约占 `57%`，但合法 reuse、XDL arbitration 和 wave-grid 变体均无收益；
- `s_wait_tensorcnt` 约 `12%`，ring b3、K512/b2、extra prefetch、direct scales 和 cache hints
  都使等待或长尾恶化；
- `s_barrier_wait` 约 `8%`，但 output private-quadrant 去 barrier 后 wall time不变，说明主要 barrier
  仍来自 hotloop 中不同 owner/consumer wave 的到达偏差；
- LDS 约 `272 KiB`、VGPR 约 `804`，occupancy 仍为 1。现有 tile 形状的小改动无法跨过 occupancy
  阈值，已测 `m4n1`、`tile_m=512` 都明显回退；
- GEMM2 只有 12 个 K256 tiles，可用于隐藏 prologue、TDM 和 output drain 的 steady-state 窗口比
  GEMM1 更短。

后续若继续，合理方向已不是微调现有开关，而是结构性重写：例如 expert-aware persistent
workgroup 调度以保证 B/ScaleB 在同一 WGP 上驻留，或者重新设计能降低 LDS/VGPR 到 occupancy 2
的 tile/ring。两者都需要重新证明 non-balanced 调度、binary search、cluster 安全和完整数值等价，
开发与验证成本显著，且当前 trace 没有足够证据保证收益。因此本轮在此暂停，不再做无依据的参数
sweep。

### 12.17 清理后的最终回归

最终源码和 `reproduce_compare.sh` 已删除本轮所有失败 selector/case，只保留逐级 winner：

```text
baseline
apre
apre_wpt2
apre_wpt2_mg4_fc28
apre_wpt2_mg4_fc28_ostore2p
apre_wpt2_mg4_fc28_ostore2p_ow2
```

a07-3 清理后 balanced const0 五轮：

| case | GEMM2 samples (us) | median | MoE e2e median | pass |
|---|---|---:|---:|:---:|
| baseline | `403.149, 404.317, 401.854, 401.913, 403.922` | `403.149` | `1358.79 us` | True |
| final winner | `333.665, 334.199, 335.056, 334.148, 334.938` | `334.199` | `1289.37 us` | True |

最终 winner 相对同轮 baseline：

```text
GEMM2 improvement = 17.10%
MoE e2e improvement = 5.11%
GEMM2 throughput = 12,954.4 TFLOP/s
effective R+W = 7.328 TB/s
```

balanced random：

```text
pass = True
logits_diff = 3.39799e-06
rel_l2 = 0.00260689
MoE output hash128 = 1556fc617347e2dabc9cff19dbfd822b
ref output hash128 = 1a5d22911ba167160b4f2c12092a5193
```

non-balanced random（`AITER_REPRO_EXPERT_BALANCE=false`）：

```text
pass = True
logits_diff = 3.48778e-06
rel_l2 = 0.00264113
MoE output hash128 = 10ef188b89c427fde6c8b322b5fd4133
ref output hash128 = d043e1891d95c1af3da5a30e37b9042a
```

non-balanced 路径继续执行 runtime `m_tile_map` binary search，没有加入 balanced rows shortcut。
所有测试结束后 `/data/yanguahe/code/gpu_users.sh` 报告无残留 GPU/KFD 进程。
