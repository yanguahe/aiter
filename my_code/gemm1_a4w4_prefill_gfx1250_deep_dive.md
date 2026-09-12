# gfx1250 A4W4 MoE GEMM1 prefill：七个优化版本深度解析

本文只分析以下唯一测试及其 7 个当前分支 GEMM1 specialization：

```bash
ROUNDS=1 RUN_VERIFY=0 RUN_ATT=0 bash my_code/reproduce_compare.sh
```

```text
data_format = a4w4
experts     = 96
tokens      = 16384
topk        = 6
model_dim   = 7168
inter_dim   = 3072
activation  = silu
bias        = false
routing     = balanced
GEMM1 tile  = M256 x raw-N256 x K256
wave grid   = 2 x 2, Wave32
LDS ring    = 4 slots
cluster     = N4 x M4
```

7 个版本是：

```text
sync_mg4_fc8
sync_mg2_fc12
sync_mg4_fc28
sync_mg4_fc28_apre
sync_mg4_fc28_apre_exactopt
sync_mg4_fc28_hard
sync_mg4_fc28_relu
```

`baseline_93665e` 只作为性能与语义参照，不计入上述 7 个当前分支版本。它通过
`my_code/run_gemm1_baseline_93665e.py` 选择仓库内保留的
`mxfp4_preshuffle_gfx1250_tdm_93665e.py`，不需要切换 Git 状态。

**Scope 约定。** 除明确标为 **【通式】**、**【非 balanced 情况】** 或
**【历史验证】** 的内容外，文中的 shape、数量、字节数和性能数字都只属于上述
唯一测试。源码中的 `a8w4_tdm` 是历史命名前缀；symbol 中紧随其后的 `fp4`
才表示本例 activation 与 weight 都采用 MXFP4。

本轮 benchmark 使用 `--const-init 0`，并且 `RUN_VERIFY=0`、`RUN_ATT=0`：

- 表中每个 “median” 实际只有一个 sample，不能据此判断亚百分比差异是否稳定；
- 本轮没有执行 random-input correctness，也没有产生本轮 hash；
- 文中关于 exact/hash 的结论只引用此前 `RUN_VERIFY=1` 的历史结果，并明确标注；
- 本文没有生成新的 ISA 或 ATT capture，所有指令级描述只写源码能够直接证明的
  结构，以及已有状态文档中已经记录的历史 ATT 证据。

**目录**

- [1. Grid swizzle and expert lookup](#sec-1-grid-swizzle-and-expert-lookup)
- [2. t256/b4 A4W4 prefill launch：M/N/K partition、ABI 与七个 specialization](#sec-2-t256-b4-a4w4-prefill-launch)
  - [2.7 Kernel 输入参数 / ABI](#sec-2-7-kernel-input-abi)
  - [2.8 七个 specialization 与 symbol](#sec-2-8-seven-specializations)
- [3. End-to-end software pipeline 与各版本优化机制](#sec-3-end-to-end-software-pipeline)

**范围与证据。** 软件主证据是：

- `my_code/reproduce_compare.sh:L1-L409`；
- `my_code/run_gemm1_baseline_93665e.py:L1-L23`；
- `op_tests/test_flydsl_grouped_gemm_gfx1250.py:L90-L135,L983-L1044,L1059-L1298`；
- `aiter/configs/tuned_grouped_fmoe.csv:L74`；
- `aiter/ops/flydsl/grouped_moe_gfx1250.py:L430-L725,L994-L1146`；
- `aiter/ops/flydsl/moe_kernels.py:L2904-L3172`；
- `aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py:L1582-L1850`；
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py:L14-L153,L156-L381`；
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py:L60-L1819`；
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_93665e.py`；
- `aiter/ops/flydsl/kernels/gemm_common_gfx1250.py:L141-L251,L373-L419`；
- `my_code/gemm1_cycle_105pct_20260909_STATUS.md`；
- `my_code/gemm1_apre_12pct_optimization_STATUS.md`。

硬件语义以本地资料为准：

- `MI400_Shader_Programming#65.txt`（下称 **MI400 Guide**）§4.3.6.6、
  §4.3.7.4.2、§4.6、§4.10；
- `amd-instinct-cdna5-instruction-set-architecture.txt`（下称 **CDNA5 ISA**）
  §10.11、§11.2.2.1 及 WMMA SCALE 指令章节。

文中会区分三类结论：硬件文档直接规定的称为“文档事实”；当前 Python/FlyDSL
实现直接给出的称为“源码事实”；由 shape、源码和性能共同得到的称为“实现推断”。

**先给结论。**

1. 7 个版本共享同一个 `M256/N256/K256, w2x2, b4, cluster 4x4`
   prefill 主体，也都保留动态 `m_tile_map` binary search；没有任何版本依赖
   `balanced_rows_per_expert` 或静态 expert 边界，因此 kernel 本身兼容
   non-balanced token 分布。
2. `sync_mg4_fc8`、`sync_mg2_fc12`、`sync_mg4_fc28` 只改变 compiler
   scheduler hint 的分组形状，不改变数据、WMMA 累加次序或 exact-SiLU 公式。
3. `sync_mg4_fc28_apre` 在 `sync_mg4_fc28` 上增加 A/ScaleA preshuffle：
   producer 先把 activation 写成 GEMM1 直接消费的物理布局，consumer 使用专用
   TDM descriptor、LDS layout 和 DS load mapping。
4. `sync_mg4_fc28_apre_exactopt` 再叠加 GEMM1-only WPT2、WMMA reuse hint、
   `DISABLE_XDL_ARB_STALL=0` 和两阶段 output TDM overlap。它是本轮最快的
   exact-SiLU 版本：`546.656 us`。
5. `sync_mg4_fc28_hard` 和 `sync_mg4_fc28_relu` 都基于**不含 A preshuffle**
   的 `sync_mg4_fc28`。它们通过改变 activation 语义降低 epilogue 成本，不能
   与 exact 版本混称为精度等价优化。
6. 本轮 `RUN_VERIFY=0`。此前 random-input 验证表明五个 exact 版本可保持
   baseline hash；hard/ReLU 只通过 production `logits_diff < 0.01` gate，输出
   hash 与 exact 路径不同。这是历史证据，不是本轮重新验证的结果。

<a id="sec-1-grid-swizzle-and-expert-lookup"></a>
## 1. Grid swizzle and expert lookup

### 1.1 从 route histogram 到 `m_tile_map`

测试脚本设置 `AITER_MOE_EXPERT_BALANCE=true`。balanced score builder 每个
token 选择连续 6 个 expert，并把起始位置按 6 轮转。由于：

```text
experts / topk = 96 / 6 = 16
tokens         = 16384 = 1024 * 16
```

每 16 个 token 恰好覆盖全部 96 个 expert 一次，因此：

```text
routes   = tokens * topk = 16384 * 6 = 98304
count[e] = routes / experts = 98304 / 96 = 1024, e=0..95
```

route kernel 首先在一个按 expert 分段的临时命名空间中分配行。这里
`max_m=align_up(98304,256)=98304`：

```text
slot(route)       = atomic_fetch_add(counter[expert], 1)
row_masked(route) = expert * max_m + slot(route)
```

`slot` 是 atomic 返回的旧值；balanced 只保证每个 expert 最终拥有
`0..1023` 这一 slot 集合，不保证某一条 route 获得哪个具体 slot。

随后 `contiguous_psum_remap` 按每个 expert 的 `tile_m=256` 对齐计数，并把
masked row 改写成连续 grouped row。

**【通式】**

```text
aligned_count[e] = align_up(count[e], tile_m)
starts[0]        = 0
starts[e+1]      = starts[e] + aligned_count[e]
psum[e]          = starts[e] + count[e]       # valid exclusive end
row_contiguous   = starts[e] + slot
```

**【本例代入】** `count[e]=1024` 已经是 256 的整数倍：

```text
aligned_count[e] = 1024
starts[e]        = 1024 * e
psum[e]          = 1024 * (e + 1)
```

因此本例没有 per-expert alignment padding：

| expert | `starts[e]` | `psum[e]` | valid grouped rows | M256 tiles |
|---:|---:|---:|---|---:|
| 0 | 0 | 1024 | `0..1023` | `0..3` |
| 1 | 1024 | 2048 | `1024..2047` | `4..7` |
| 95 | 97280 | 98304 | `97280..98303` | `380..383` |

`psum` 而不是 `starts` 被作为 `arg_m_tile_map` 传入 GEMM1。它表示每个
expert 的有效 exclusive end，并同时服务于 expert lookup 与当前 tile 的
M 方向 OOB 限界。

### 1.2 静态 `contiguous_m`、有效 M tiles 与 sentinel tail

为了保持 CUDA Graph 可捕获性，launch 使用一个由 shape 决定的静态容量，
而不是读取 GPU 上刚计算出的 histogram 后再动态改变 grid。

**【通式】** 当前 grouped-MoE 路径计算：

```text
align_m      = max(tile_m_gemm1, tile_m_gemm2)
upper_bound  = tokens*topk + experts*align_m - topk
contiguous_m = align_up(upper_bound, align_m)
```

**【本例代入】** 两个 GEMM 都是 `tile_m=256`：

```text
align_m      = 256
upper_bound  = 98304 + 96*256 - 6
             = 122874
contiguous_m = align_up(122874,256)
             = 122880
```

于是必须区分：

```text
valid route rows            = 98304
actual balanced aligned span= 98304
static contiguous_m capacity= 122880
static M256 tiles            = 122880 / 256 = 480
valid M256 tiles             = 98304  / 256 = 384
sentinel M256 tiles          = 480 - 384 = 96
sentinel rows                = 98304..122879
```

这里的 96 个 sentinel tiles 不是 expert padding。它们只是静态 capacity
超过本次真实 routed-row span 的尾部，进入 kernel 后会由
`expert < n_experts` 的统一分支跳过。

### 1.3 4×4 cluster 下的二维 grid 与 16-M-tile swizzle

当前 prefill specialization 满足 `fp4_prefill_schedule`，源码因此令：

```text
cluster_n = 4        # 来自 tuned config
cluster_m = 4        # 当前 FP4 prefill kernel 内部派生
block      = 2 * 2 * 32 = 128 threads = 4 waves
```

N 方向共有：

```text
raw GEMM N = 2 * inter_dim = 6144
N tiles    = 6144 / 256 = 24
N units    = 24 / cluster_n = 6
```

M 方向共有：

```text
M tiles = 480
M units = ceil(480 / cluster_m) = 120
```

launcher 的源码 grid 是：

```text
grid.x = M units * N tiles = 120 * 24 = 2880
grid.y = cluster_m         = 4
grid.z = 1
cluster = (4,4,1)
```

所以物理 workgroup 数量仍是：

```text
2880 * 4 = 11520 = 480 M tiles * 24 N tiles
```

每个 cluster 含 16 个 workgroups，总 cluster 数为：

```text
11520 / 16 = 720
```

源码在 cluster 粒度执行 DeepGEMM 风格的 16-M-tile group swizzle。设
`bid_x=block_idx.x`、`bid_y=block_idx.y`：

```text
local_n  = bid_x % 4
swz_id   = bid_x // 4
local_m  = bid_y

group_m_units  = 16 / cluster_m = 4
blocks_per_group = N units * group_m_units = 6 * 4 = 24
group       = swz_id // 24
in_group    = swz_id % 24
m_unit      = 4*group + (in_group % 4)
n_unit      = in_group // 4

m_tile = 4*m_unit + local_m
n_tile = 4*n_unit + local_n
blk_m  = 256*m_tile
blk_n  = 256*n_tile
```

本例 `m_units=120`，正好形成 30 个完整 group。每个 group 覆盖 16 个连续
M tiles，并在其中让 4 个 M-units 比 N-unit 变化更快。一个固定的
`(m_unit,n_unit)` 对应一个 4×4 cluster：Y 方向 4 个 peer 改变 M tile，X
方向 4 个 peer 改变 N tile。

balanced 下：

```text
每 expert M tiles = 1024 / 256 = 4
```

恰好等于 `cluster_m=4`。因此有效区中的一个 4×4 cluster 的四个 M peers
属于同一 expert：

```text
valid M units / expert units = 384/4 = 96
valid clusters               = 96 experts * 6 N units = 576
sentinel clusters            = (120-96) * 6 = 144
```

这只是本例 balanced histogram 带来的理想对齐，不是 kernel 的正确性前提。

### 1.4 8-step binary search：为何仍兼容 non-balanced routing

每个 workgroup 都对 96-entry `m_tile_map=psum` 做固定次数 upper-bound
查找：

```text
lo = 0
hi = 96
repeat ceil(log2(96)) + 1 = 8 times:
    mid = (lo + hi) >> 1
    mid_clamped = min(mid, 95)
    go_right = psum[mid_clamped] <= blk_m
    lo = go_right ? mid + 1 : lo
    hi = go_right ? hi      : mid
expert = lo
```

其语义是寻找第一个满足 `psum[expert] > blk_m` 的 expert。balanced 本例中：

```text
m_tile = 4*e + j, j=0..3
blk_m  = 1024*e + 256*j
expert = e
mn_oob = psum[e] - blk_m
       = 1024 - 256*j
       ∈ {1024,768,512,256}
```

四个 M tiles 都是完整 M256，因此 load/store 的 M bound 不截断任何有效行。
对 sentinel tail，固定 8-step 实现可能得到任意 `expert >= 96` 的 sentinel
值；源码只依赖 `expert < n_experts` 为 false，并在潜在数组读取前把索引 clamp
到 95，因此不应把 sentinel 的具体数值硬编码成 96。

**【非 balanced 情况】** `count[e]` 不再相等时，`starts/psum` 仍由运行时
histogram 生成，binary search 不变。一个 4-M cluster 可能跨越 expert
边界；当前实现没有假设四个 M peers 属于同一 expert，而是为每个 peer 计算：

```text
first_m(expert) = ceil(previous_psum / tile_m)
end_m(expert)   = ceil(current_psum  / tile_m)
```

然后构造只包含“同 expert M peers”的 `b_mcast_mask`。因此：

- A/ScaleA 只沿固定 M row 的 4 个 N peers multicast；
- B/ScaleB 只沿同一 expert 的 M-peer 子集 multicast；
- cluster 跨 expert 时不会错误共享另一 expert 的 weight；
- 尾部 cluster 若含 sentinel M tile，则不使用要求全 cluster 到齐的 barrier；
- `m_tile_map` binary search、动态 expert 边界和 `mn_oob` 全部保留。

这正是 7 个版本都能兼容 non-balanced routing 的核心。balanced 本例只让
`b_mcast_mask` 恰好覆盖完整四个 M peers，从而获得最理想的 weight multicast。

### 1.5 与 gfx1250 硬件语义的对应关系

以下是硬件文档事实，而不是从性能数字反推：

- CDNA5 ISA §10.11 规定 TDM 通过 SGPR descriptor 描述 tensor/tile，在 global
  memory 与 LDS 之间搬运数据，并可与其它 shader 指令并行；完成状态由
  `TENSORcnt` / `S_WAIT_TENSORCNT` 管理。
- `D#.workgroup_mask != 0` 时，`TENSOR_LOAD_TO_LDS` 使用
  `CLUSTER_LOAD_ASYNC` 而不是普通 `GLOBAL_LOAD_ASYNC`；store 路径忽略该 mask。
- descriptor 的 `early_timeout` bit 允许 GL1 在数据返回时先向已经发出匹配请求
  的 requesters 返回，而不是无限等待尚未到达的 peer。
- CDNA5 ISA §10.11.2、§10.11.6 规定 load 超出 tensor positive bound 时读零，
  store 超界时丢弃。这是 `mn_oob` 能处理 expert 尾 tile 的硬件基础。
- MI400 Guide §4.3.6.6 规定 cluster barrier 统计 workgroup；推荐 workgroup
  内先同步，再由一个 wave 发 `s_barrier_signal -3`，cluster 中所有 waves
  执行 `s_barrier_wait -3`。

当前 `sync_*` 路径遵守上述协议。名字中的 `sync` 很重要：它不再使用历史
`skip_cluster_sync` 实验。`_rcw` 只在 ring wrap 时省去额外的完整 LDS drain，
仍保留 workgroup signal/wait 与 cluster signal/wait；它不是“跳过 cluster
同步”。

<a id="sec-2-t256-b4-a4w4-prefill-launch"></a>
## 2. t256/b4 A4W4 prefill launch：M/N/K partition、ABI 与七个 specialization

### 2.1 配置如何命中 `t256x256x256_w2x2_b4`

本测试的 `tokens=16384, topk=6, A4W4, Silu` 命中
`tuned_grouped_fmoe.csv:L74`。GEMM1 关键配置是：

```text
tile_m / tile_n / tile_k = 256 / 256 / 256
m_warp / n_warp          = 2 / 2
num_buffers              = 4
cluster_n                = 4
waves_per_tensor_tdm     = 1
next_stage_prefetch      = 1
```

当前 kernel 又在 `fp4_prefill_schedule` 条件成立时派生 `cluster_m=4`，并默认
打开：

```text
epilogue_batch_wn        = 8
schedule_hints           = 1
relax_cluster_wrap_dscnt = 1
tdm_early_timeout        = 1
```

因此 `sync_mg4_fc8` 不是“只设置两个环境变量的原始 93665e kernel”。它是
当前分支完整 prefill 公共优化栈上的默认 schedule shape。`baseline_93665e`
则通过单独的旧 kernel 文件执行 93665e 语义，其 cluster 只有 N 方向 4 peers，
没有当前 4×4 cluster、batched exact-SiLU 与启用后的 schedule-hint 路径。

### 2.2 M/N/K partition

本例 GEMM1 的数学形状按 expert 分组后为：

```text
M = 每个 expert 的 routed rows，物理上拼成 contiguous-M
N = 2 * inter_dim = 6144             # GUGU gate/up raw columns
K = model_dim       = 7168
```

跨 workgroup 的 partition：

```text
M tile = 256 rows
N tile = 256 raw gate/up columns
K tile = 256 reduction elements
```

因此每个有效 workgroup 完成：

```text
raw GEMM tile      = M256 x N256 x K7168
activation output = M256 x N128 BF16
```

N 减半是因为 W1 已按 GUGU 顺序排列：

```text
[g0,u0,g1,u1,...]
```

epilogue 把相邻 `(gate, up)` 两列合成一个输出。24 个 raw N256 tiles 最终
覆盖 `24*128=3072` 个 activation-output columns，没有 N tail。

K 方向不做 split-K：

```text
K tiles       = 7168 / 256 = 28
K128 per tile = 256 / 128  = 2
K128 steps    = 28 * 2     = 56
```

### 2.3 单个 workgroup 的 wave partition

block 有 128 threads，即四个 Wave32：

| wave | `wave_m` | `wave_n` | raw GEMM responsibility | activation output |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | M `0..127`, raw N `0..127` | M `0..127`, out N `0..63` |
| 1 | 0 | 1 | M `0..127`, raw N `128..255` | M `0..127`, out N `64..127` |
| 2 | 1 | 0 | M `128..255`, raw N `0..127` | M `128..255`, out N `0..63` |
| 3 | 1 | 1 | M `128..255`, raw N `128..255` | M `128..255`, out N `64..127` |

每个 wave 的 raw tile 是 `M128×N128`：

```text
wmma_m_rep = 128 / 16 = 8
wmma_n_rep = 128 / 16 = 8 logical fragments
mma_n_rep  = 8 / 2    = 4 physical FP4 WMMA positions
```

源码使用 `v_wmma_scale_f32_32x16x128_f4` 对应的 FlyDSL intrinsic。一个
K128 step 每 wave 执行：

```text
wmma_m_rep * mma_n_rep = 8 * 4 = 32 physical WMMA
```

完整 K7168 因而是：

```text
per wave = 32 * 56 = 1792 physical WMMA
per WG   = 1792 * 4 = 7168 physical WMMA
```

MI400 Guide 的 WMMA 章节说明该 scaled FP4 WMMA 是 Wave32 指令、FP32
accumulate，并按 K32 block scale 使用 A/B scale。这里的 32/1792/7168 是
当前源码循环次数，不是从 profiler 时间反推的吞吐率。

### 2.4 每个 K256 的 TDM payload 与 4×4 multicast

每个 workgroup、每个 K256 的四类 logical input tile 为：

| tensor | logical tile | global bytes |
|---|---:|---:|
| A payload | `256 × (256/2)` | 32,768 B |
| B payload | `256 × (256/2)` | 32,768 B |
| ScaleA | `256 × (256/32)` | 2,048 B |
| ScaleB | `256 × (256/32)` | 2,048 B |
| total without reuse | — | 69,632 B/WG/K256 |

balanced 本例的一个 4×4 cluster 中：

- 4 个 local-M rows 各有一份 A/ScaleA，并沿 4 个 local-N peers 共享；
- 4 个 local-N columns 各有一份 B/ScaleB，并沿 4 个 local-M peers 共享；
- 四个 local-M peers 恰属于同一 expert，因此 B/ScaleB mask 是完整列。

若只按 logical unique bytes 计，不考虑 cache line、协议和 descriptor 开销：

```text
without multicast = 16 * 69632 = 1,114,112 B/cluster/K256
balanced multicast= 4*(32768+2048) + 4*(32768+2048)
                  = 278,528 B/cluster/K256
```

这是理想的 4× 去重关系，不等同于实测 HBM traffic，也不能直接换算成 4×
kernel 加速。实际仍包含每个 requester 的 descriptor/control、同步、LDS read、
WMMA 和 epilogue 成本。

### 2.5 四槽 LDS ring：普通 A 与 A preshuffle

对本例 `tile_m=tile_n=tile_k=256`，普通 A layout 的单槽 LDS 分解为：

| region | 计算 | bytes |
|---|---:|---:|
| A | `256 * (128 payload + 16 pad)` | 36,864 |
| B | `16 * 2048` | 32,768 |
| ScaleA | `2 * 256 dword` | 2,048 |
| ScaleB | `8 * 64 dword` | 2,048 |
| `PITCH` | 已是 512-B aligned | 73,728 |

四槽 ring 为：

```text
4 * 73728 = 294912 B = 288 KiB/WG
```

A preshuffle 后：

| region | 计算 | bytes |
|---|---:|---:|
| A | `16 outer * 2048-byte WMMA-friendly row` | 32,768 |
| B | unchanged | 32,768 |
| ScaleA | `8 outer * 64 dword` | 2,048 |
| ScaleB | unchanged | 2,048 |
| `PITCH` | 已是 512-B aligned | 69,632 |

四槽 ring 降为：

```text
4 * 69632 = 278528 B = 272 KiB/WG
```

减少的 16 KiB/WG 全部来自取消普通 A 的每行 16-byte LDS padding。更重要的
变化不是这 16 KiB 本身，而是 A/ScaleA 已在 global memory 中按 GEMM1 的
WMMA consumption 顺序组织，TDM 能直接搬入对应 LDS shape，DS load 不再从
普通 row-major A 再做同样的寻址重排。

### 2.6 b4 next-stage pipeline

`next_stage_prefetch=1` 且 `num_buffers=4`，所以 `next_stage_on=1`。当前
prefill 走 mid-compute prefetch 分支：

```text
K_TILES = 28
PRE     = num_buffers = 4
n_steady= 28 - 4 = 24
```

高层控制流是：

```text
Prologue:
  issue kt0, kt1, kt2, kt3 into LDS slots 0..3
  wait until slot0 is readable
  preload slot0/K128-0 into rmem

Steady, kt=0..23:
  compute current K256
    K128-0: consume current rmem, preload current K128-1
    K128-1: consume current rmem, issue kt+4, preload next tile K128-0
  every 4 consumed K tiles: synchronized cluster ring wrap

Drain:
  compute remaining kt24..kt27
  issue no new input TDM
```

每个 K256 有 A/B/ScaleA/ScaleB 四个 logical jobs。WPT1 与 WPT2 的区别
只在 owner 数和 descriptor 切分：

| mode | owner mapping | per owner wave / K256 | total tensor instructions / WG / K256 |
|---|---|---:|---:|
| WPT1 | wave0=A, wave1=B, wave2=SA, wave3=SB | 1 | 4 |
| WPT2 | waves0-1=A+SA groups；waves2-3=B+SB groups | 2 half-tile jobs | 8 |

WPT2 没有减少搬运字节，也增加了 tensor-instruction 数量；它的价值是把每个
大 tile 分给两个 waves，并重新平衡 descriptor setup 与 TDM issue 的时序。
是否更快是实现与调度问题，不是由字节数单独决定。

<a id="sec-2-7-kernel-input-abi"></a>
### 2.7 Kernel 输入参数 / ABI

当前 FlyDSL kernel 源码签名有 13 个参数；所有 7 个版本保持完全相同的调用
接口。由于本轮 `RUN_ATT=0`，本文不虚构最终 code object 的 kernarg byte
offset，只给源码可证明的逻辑 ABI：

| 参数 | 本例值 / shape | 作用 |
|---|---|---|
| `arg_c` | BF16 `[1,122880,3072]` | GEMM1 fused-SiLU 输出；实际只消费 valid routed rows |
| `arg_a` | packed FP4 bytes，逻辑 `[1,122880,7168]` | 普通 routeks A 或 `_apre` A；storage bytes 均为 `[1,122880,3584]`，物理 permutation 不同 |
| `arg_b` | packed FP4 W1，logical `[96,6144,7168]` | GGUU 先转 GUGU，再做 weight preshuffle |
| `arg_scale_a` | E8M0；allocation `[1,15360,1792]` bytes 后以 `int32` view 传入 | 每 K32 一个 scale；普通与 `_apre` 的物理解释不同 |
| `arg_scale_b` | E8M0 W1 scale，logical `[96,6144,224]` | GUGU + n32k4 preshuffle 后传入 |
| `arg_m_tile_map` | `int32[96]` | 动态 `psum`，用于 8-step expert upper-bound 与 `mn_oob` |
| `arg_bias` | dummy pointer | `--no-bias`，`has_bias=0` 使 load 在 compile time 删除 |
| `arg_quant_scale` | dummy/alias | A4W4 GEMM1 输出 BF16，`stage1_quant_out=0`，该路径不使用 |
| `i32_m` | 122880 | static contiguous-M capacity，不是 valid routes |
| `i32_n` | 6144 | activation 前 raw gate/up width；output stride 是 3072 |
| `f32_swiglu_limit` | 7.0 | 本 SiLU 路径仍用于 clamp gate/up |
| `f32_situ_beta` | test 默认值 | `act1` specialization 不使用 |
| `f32_situ_linear_beta` | test 默认值 | `act1` specialization 不使用 |

`fx.Tensor` 在 lowering 时还会携带 layout descriptor，因此机器 ABI row 数会
多于 13；没有匹配本轮 symbol 的 final ISA/metadata 时，不应把 decode 文档中
另一个 specialization 的 offset 原样移植到这里。

<a id="sec-2-8-seven-specializations"></a>
### 2.8 七个 specialization 与 symbol

下面列出脚本实际期望的 symbol。`mg4/fc8` 和 WPT2 是源码命名默认值，因此
存在两个容易误读的规则：

- `sync_mg4_fc8` 的 symbol 不带 `_mg4_fc8`；
- `sync_mg4_fc28_apre_exactopt` 使用 WPT2，但 symbol 不带 `_wpt2`；反而
  WPT1 才显式带 `_wpt1`。

| case | exact activation | A preshuffle | WPT | schedule | 额外机制 | kernel symbol |
|---|:---:|:---:|---:|---|---|---|
| `sync_mg4_fc8` | 是 | 否 | 1 | `mg4/fc8` | 当前公共 prefill 栈 | `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1_eb8_sh_rcw` |
| `sync_mg2_fc12` | 是 | 否 | 1 | `mg2/fc12` | 更细的前部 WMMA groups | `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1_eb8_sh_rcw_mg2_fc12` |
| `sync_mg4_fc28` | 是 | 否 | 1 | `mg4/fc28` | 长 closing WMMA group | `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1_eb8_sh_rcw_mg4_fc28` |
| `sync_mg4_fc28_apre` | 是 | 是 | 1 | `mg4/fc28` | 专用 A/ScaleA layout 与 load mapping | `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1_eb8_apre_sh_rcw_mg4_fc28` |
| `sync_mg4_fc28_apre_exactopt` | 是 | 是 | 2 | `mg4/fc28` | `_xdl0_reuse_ostore2p` | `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p` |
| `sync_mg4_fc28_hard` | 否 | 否 | 1 | `mg4/fc28` | hard-SiLU approximation | `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1_eb8_sh_rcw_mg4_fc28_silu_hard` |
| `sync_mg4_fc28_relu` | 否 | 否 | 1 | `mg4/fc28` | ReLU-gate approximation | `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1_eb8_sh_rcw_mg4_fc28_silu_relu` |

七者不改变外部 GEMM1 ABI；差异都通过 compile-time specialization 进入同一
launcher。`baseline_93665e` 的 symbol 与当前 `sync_mg4_fc8` 不同，它没有
`_eb8_sh_rcw` 后缀，也不包含当前 4×4 cluster 的 M 方向扩展。

<a id="sec-3-end-to-end-software-pipeline"></a>
## 3. End-to-end software pipeline 与各版本优化机制

### 3.1 公共端到端路径

本测试从 BF16 hidden 到 GEMM1 输出的公共路径是：

```text
BF16 hidden [16384,7168]
  -> topk=6 route map，共 98304 routes
  -> contiguous psum/remap，生成动态 m_tile_map
  -> A4 per-1x32 quant + ScaleA layout
  -> A4W4 GEMM1，FP32 accumulate
  -> exact 或 approximate SiLU-gated epilogue
  -> BF16 grouped intermediate [1,122880,3072]
  -> 后续 A4 quant/preshuffle、GEMM2、gather-reduce
```

W1 的 logical GGUU layout：

```text
[g0..g3071, u0..u3071]
```

在进入 GEMM1 前变成 GUGU：

```text
[g0,u0,g1,u1,...,g3071,u3071]
```

然后 payload 采用 GEMM weight preshuffle，ScaleB 采用 n32k4 layout。这个
顺序使 epilogue 从每个 accumulator fragment 中直接取得相邻 `(gate,up)`。
改变 GUGU/GGUU 解释会改变数学结果，不属于性能等价变换。

### 3.2 公共 current-branch 优化栈相对 `baseline_93665e`

`sync_mg4_fc8` 已经包含当前 prefill 公共优化栈。与保留的 93665e kernel
相比，主要结构变化包括：

1. **cluster 从 4×1 扩展为 4×4。**
   旧 kernel 只沿 N 共享 A/ScaleA；当前 kernel 又沿 M 对同 expert peers
   共享 B/ScaleB，并为 non-balanced 边界动态生成 mask。
2. **同步的四槽 ring。**
   当前路径在启动和每四个 K tiles 的 ring wrap 使用 workgroup + cluster
   barrier；`_rcw` 只放松额外的 LDS drain，不删除同步。
3. **`epilogue_batch_wn=8`。**
   每个 `wm` 一次收集 8 个 logical N fragments、共 32 个 `(gate,up)` pairs，
   把所有独立 `exp2`、加法、`rcp` 和最终乘法分阶段发出，提升 TRANS latency
   overlap，同时控制临时 VGPR 生命周期。
4. **`STORE_PAD=8`。**
   BF16 activated output 以 136-element LDS row pitch staging；源码明确把
   8-column skew 用于分散 lane16 的 LDS bank 映射，padding 不写回 global。
5. **启用 schedule hints 与 descriptor/control 路径优化。**
   当前 `emit_hints()` 不再是空函数，并对 DS read、TDM issue 与 WMMA group
   给 compiler 明确的调度分区。

本轮只能测得这些变化的组合效果：

```text
baseline_93665e = 689.116 us
sync_mg4_fc8    = 622.669 us
improvement     = 9.64%
```

不能把这 9.64% 单独归因于 cluster、epilogue batch 或任意一个小改动。

### 3.3 `mg*/fc*` 到底控制什么

每个 K128、每个 wave 的源码 compute body具有：

```text
physical WMMA = 32
logical LDS reads represented to scheduler = 40
```

40 个 DS read 来自：

```text
A payload: 8 wm * 2 ds_read_b128 = 16
B payload: 8 logical wn * 2       = 16
ScaleA: 4 ds_read_b32             = 4
ScaleB: 4 ds_read_b32             = 4
total                             = 40
```

在典型 steady K256 的第一个 K128，源码把 `FENCE_COVER_MMA` 条 WMMA 留作
closing pure-MFMA group，其余 WMMA 按 `MMA_GROUP` 分组，并把下一 K128 的
DS reads 分散到这些 group 周围：

```text
mma_total      = 32 - FENCE_COVER_MMA
schedule_slots = mma_total / MMA_GROUP
```

三个 exact、无 A-preshuffle schedule case 为：

| case | 前部 WMMA | 前部 groups | closing WMMA | 调度意图 |
|---|---:|---:|---:|---|
| `sync_mg4_fc8` | 24 | `6 × 4` | 8 | 较均匀地穿插 next-K128 DS read |
| `sync_mg2_fc12` | 20 | `10 × 2` | 12 | 更细粒度、更多 scheduler groups |
| `sync_mg4_fc28` | 4 | `1 × 4` | 28 | 尽早安排大部分 next-state DS read，再以长 WMMA group 覆盖 fence |

第二个 K128 没有这段 `fc` tail reservation，仍按 `mg2` 或 `mg4` 对 32 条
WMMA 分组。`mg/fc` 只改变 compiler scheduling hints：

- 不改变 56 个 K128 的遍历顺序；
- 不改变每个 accumulator 的 K 累加顺序；
- 不改变 TDM tensor 内容或 LDS address；
- 不改变 exact-SiLU 公式；
- 不删除 binary search 或 cluster synchronization。

### 3.4 `sync_mg4_fc8`

这是当前分支 prefill 公共优化栈的默认 schedule shape：

```text
MMA_GROUP      = 4
FENCE_COVER_MMA= 8
A preshuffle   = 0
WPT            = 1
activation     = exact-SiLU path
```

它的主要价值不是 `mg4/fc8` 两个数字本身，而是作为“当前同步实现”的基准点：
4×4 cluster、动态 expert mask、b4 next-stage pipeline、batch-8 epilogue、
`STORE_PAD=8` 和 `_rcw` 都已经启用。

本轮结果：

```text
GEMM1   = 622.669 us, +9.64% vs 93665e
MOE e2e = 1595.48 us, +4.17% vs 93665e
```

### 3.5 `sync_mg2_fc12`

该版本只把 schedule shape 改成：

```text
MMA_GROUP      = 2
FENCE_COVER_MMA= 12
```

它增加前部 scheduler group 数，使 DS-read 分散粒度更细，同时保留 12 条
closing WMMA。理论上这可能减少局部 dependency bubble，但也会增加调度约束
数量，并限制 compiler 自由度。

本轮结果：

```text
GEMM1   = 623.041 us, +9.59% vs 93665e
MOE e2e = 1597.83 us, +4.03% vs 93665e
```

相对 `sync_mg4_fc8`，GEMM1 仅慢 `0.060%`。在单 sample 条件下这应视为
持平，而不是证明 `mg2/fc12` 必然更差。

### 3.6 `sync_mg4_fc28`

该版本使用：

```text
MMA_GROUP      = 4
FENCE_COVER_MMA= 28
```

典型 steady K256 的第一个 K128 只留下一个 `4-WMMA` 前部 group，把其余
28 条 WMMA 作为 closing group。它的意图是让 next-state LDS loads 尽早进入
调度窗口，再用较长的 independent WMMA 区间覆盖其依赖等待。

本轮结果：

```text
GEMM1   = 619.572 us, +10.09% vs 93665e
MOE e2e = 1597.08 us, +4.07% vs 93665e
```

相对 `sync_mg4_fc8`，GEMM1 提升 `0.497%`。它是三个纯 schedule variants
中本轮最快者，但单轮结果不足以断言 0.5% 是稳定收益。后续四个版本都以它
作为直接对照基线。

### 3.7 `sync_mg4_fc28_apre`：A/ScaleA producer-consumer 协同布局

该版本在 `sync_mg4_fc28` 上增加：

```text
AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1
```

它不是只改 GEMM 内一个 address expression，而是 producer 与 consumer 的
成对协议。

#### 3.7.1 producer：每 token 量化一次，再按 grouped row scatter

普通 routeks producer 以 route 为工作单位；topk=6 时，同一个 source token
可能被读取和量化 6 次。`_apre` 的 fast path 改成：

```text
moe_quant_token_fd7168_fp4_pk8
  每个 source token 只做一次完全相同的 FP4/MX32 quant

moe_invert_route_rows_tk6
  把 route -> grouped_row 反转成 grouped_row -> source_token

moe_scatter_preshuffled_a_fd7168_r32_lds
  每个 WG 处理连续 32 个 grouped rows，经 LDS transpose 后
  合并写出 A payload 与 ScaleA 的 GEMM-friendly layout
```

本 shape 满足 fast path 的全部 guard：FP4、`source_topk=6>1`、非 EP dead
tail、`7168 % 1024 == 0`、`122880 % 32 == 0`，且各 buffer offset 小于 2 GiB。
不满足 guard 的其它 shape 自动退回原 routeks producer。

#### 3.7.2 consumer：专用 descriptor、LDS layout 与 DS mapping

普通 A 的 TDM view 是：

```text
global: 256 rows x 128 bytes, row stride = K/2
LDS:    256 rows x (128+16) bytes
```

`_apre` A view 变成：

```text
global/LDS outer = 256/16 = 16
inner            = PACK_TK*16 = 2048 bytes
LDS padding      = 0
```

对应 lane base 与每个 `wm/ksl` 的 offset 也切换到 preshuffled mapping：

```text
lds_a_lane_off = (wmb/16)*A_LDS_ROW + kgrp*256 + lane16*16
load_a offset  = wm*A_LDS_ROW + ksl*1024
second b128    = first + 512
```

ScaleA 从“每 wave-M tile 的 `(k128,wm,lane16)` layout”切换为按 32-row
super-row 与 K 分块的布局：

```text
global outer stride = K/4 dwords
TDM outer           = tile_m/32 = 8
TDM inner           = tile_k/4  = 64 dwords
```

数学上的 A 元素和 scale 不变，变化的是 producer 写出的 permutation、TDM
descriptor shape、LDS address 和 DS load mapping。

#### 3.7.3 性能与等价性边界

本轮结果：

```text
GEMM1   = 564.597 us
MOE e2e = 1447.19 us
```

相对 `sync_mg4_fc28`：

```text
GEMM1 improvement   = 8.873%
MOE e2e improvement = 9.385%
```

GEMM1 gain 来自 consumer 更直接的 A/ScaleA 数据通路；e2e 还包含 producer
从“每 route 重复量化”变成“每 token 一次量化 + inverse/scatter”的收益。

**【历史验证】** `gemm1_cycle_105pct_20260909_STATUS.md` 记录：最终三-kernel
producer 复用原 routeks 的 `emit_mx_e8m0_scale`、round mode、
`v_cvt_scalef32_pk8_fp4_bf16` 和 pack mapping；多组 balanced/unbalanced
shape 的有效 routed rows 做到了 payload/ScaleA byte-exact，并且完整 random
MoE 输出保持 exact baseline hash。本轮 `RUN_VERIFY=0` 没有重新验证这些结论。

### 3.8 `sync_mg4_fc28_apre_exactopt`：四项 exact-only 优化

该版本保留 `sync_mg4_fc28_apre` 的数学路径，再同时启用四项不改变数值语义的
优化。

#### 3.8.1 GEMM1-only WPT2

```text
AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2
```

只改变 GEMM1；GEMM2 仍使用原配置。A 与 ScaleA 由 waves 0/1 分担，B 与
ScaleB 由 waves 2/3 分担。每个 tensor tile 被切成两个 descriptor，而不是
让一个 wave 独占整个 tile。

#### 3.8.2 保留 XDL arbitration stall

```text
AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0
```

其它当前 prefill cases 使用 automatic default；对本 `M256/N256/4-wave`
specialization，其有效值为 1，会设置 `SCHED_MODE.bit[2]`。exactopt 显式设 0，
不写该 disable bit。

MI400 Guide §4.3.7.4.2 的文档事实是：设置该 bit 可让一个 wave 连续 issue
多个 WMMA，但可能阻塞其它 wave 的 co-execution，主要可能在单 wave/SIMD
时有利。exactopt 的选择是实测结果，不应简化成“关闭 stall 一定更快”或
“保留 stall 一定更快”。

#### 3.8.3 exact WMMA A/B reuse hint

```text
AITER_FLYDSL_GEMM1_WMMA_REUSE=1
```

`mma_rows()` 使用 snake traversal：偶数 `wm` 的 `wn` 正向，奇数 `wm`
反向。源码只在 operand 确实连续复用时设置 hint：

```text
reuseA = current wm 内 wn_raw > 0
reuseB = wm > 0 且 wn_raw == 0 的相邻 snake 边界
```

MI400 Guide 的 WMMA 章节明确规定 reuse bit 只能在相邻指令确实复用对应 matrix
时设置，否则结果未定义。当前条件正是按这一限制构造；它只提示 source cache，
不改变 accumulator 的 K 顺序或 FP32 运算。

#### 3.8.4 两阶段 output TDM overlap

```text
AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE=1
```

每 wave 有 8 个 `wm`。普通路径先完成全部 exact-SiLU、把完整 M256×N128
BF16 tile 写入 LDS，再统一 barrier 并发出 output TDM。两阶段路径改为：

```text
activate/stage wm0..wm3
  -> workgroup barrier
  -> wave_n==0 owners 发出第一半 64-row output TDM

同时继续 activate/stage wm4..wm7
  -> workgroup barrier
  -> 发出第二半 64-row output TDM

最后 tensor_wait(0)
```

每个 half descriptor 仍带 `mn_oob` 与 `STORE_N=128` bound。已经完成的前半
LDS rows 在第二半 exact-SiLU 计算期间向 global memory 搬运，最终 bytes 与
普通 TDM store 相同。

**【历史 ATT】** `gemm1_apre_12pct_optimization_STATUS.md` 记录，保留的两阶段
版本把 final `s_wait_tensorcnt 0` 降到约 `26 cycles/wave`；对应 144 个完整
active waves 的平均 phase split 为：

| phase | cycles/wave |
|---|---:|
| prologue | 6,130 |
| WMMA core | 21,702 |
| epilogue | 3,499 |
| total | 31,331 |

这是此前 a07-3 ATT 的历史证据，不是本轮 benchmark 的新 capture。

#### 3.8.5 组合结果

本轮结果：

```text
GEMM1   = 546.656 us, +20.67% vs 93665e
MOE e2e = 1427.16 us, +14.28% vs 93665e
```

相对 `sync_mg4_fc28_apre`：

```text
GEMM1 improvement   = 3.178%
MOE e2e improvement = 1.384%
```

这是 WPT2、`xdl0`、reuse 和 two-phase output overlap 的组合收益。本轮没有
逐项 ablation，不能把 3.178% 全部分配给其中某一项。

### 3.9 exact-SiLU、hard-SiLU 与 ReLU-gate 的精确公式

本例 CLI 默认 `f32_swiglu_limit=7.0`。令 raw accumulators 中相邻两列为
`g`（gate）与 `u`（up），并定义：

```text
g_hi = min(g, 7)
u_c  = clamp(u, -7, 7)
```

#### exact-SiLU path

`sync_mg4_fc8`、`sync_mg2_fc12`、`sync_mg4_fc28`、
`sync_mg4_fc28_apre` 和 `sync_mg4_fc28_apre_exactopt` 使用：

```text
sigmoid(g_hi) = rcp(1 + exp2(-g_hi * log2(e)))
y              = g_hi * sigmoid(g_hi) * u_c
```

这里“exact”表示保持 93665e/current production helper 的同一算法语义与
输出，而不是声称硬件 `exp2`/`rcp` 等于无限精度实数函数。

batch-8 实现一次收集 32 个 independent `(g,u)` pairs，按以下阶段组织：

```text
all clamp
-> all exp2
-> all (1+exp)
-> all rcp
-> all final multiply
```

这改变 independent instructions 的排布，不改变单个元素的公式。

#### `sync_mg4_fc28_hard`

hard-SiLU 用线性 hard-sigmoid 代替 sigmoid：

```text
sig_hard = clamp(0.193 * g_hi + 0.5, 0, 1)
y_hard   = g_hi * sig_hard * u_c
```

它删除每个输出的 `exp2` 与 `rcp` 链，改用 FMA、clamp 和 multiply。除
activation helper 外，它与 `sync_mg4_fc28` 保持相同的普通 A layout、WPT1、
`mg4/fc28`、cluster 和 output-store 路径。

本轮结果：

```text
GEMM1   = 594.398 us
MOE e2e = 1567.96 us
```

相对同数据通路的 `sync_mg4_fc28`：

```text
GEMM1 improvement   = 4.063%
MOE e2e improvement = 1.823%
```

#### `sync_mg4_fc28_relu`

ReLU-gate 进一步把 gate activation 改成：

```text
g_relu = clamp(g, 0, 7)
y_relu = g_relu * u_c
```

它完全删除 sigmoid 近似，不再计算 `exp2`、`rcp` 或 hard-sigmoid FMA。其余
数据通路仍与 `sync_mg4_fc28` 相同，也不包含 A preshuffle/exactopt。

本轮结果：

```text
GEMM1   = 576.660 us
MOE e2e = 1548.27 us
```

相对 `sync_mg4_fc28`：

```text
GEMM1 improvement   = 6.926%
MOE e2e improvement = 3.056%
```

#### 历史精度证据

以下来自此前 random-input `RUN_VERIFY=1`，不是本轮结果：

| activation path | `logits_diff` | `rel_l2` | output SHA256 | 语义 |
|---|---:|---:|---|---|
| exact variants | `3.39799e-06` | `2.60689e-03` | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` | 与 baseline 输出一致 |
| hard-SiLU | `3.2639e-05` | `8.0800e-03` | `91f3c3c87e7e17e854bcc5c3dbb7032f0f5a039a033a205a1cba04206799b5ca` | approximation |
| ReLU-gate | `6.4241e-05` | `1.1336e-02` | `ca4a57024a6c0ee78991a0a2dcfd227852945fd356657fe64df596f62e98bdd4` | larger semantic approximation |

两种近似都通过当时的 production `logits_diff < 0.01` gate，但 hash 已改变，
所以“通过容差”不能写成“功能/精度等价”。

### 3.10 本轮性能总表与正确读法

用户提供的单轮结果如下，原样保留：

| case | GEMM1 samples (us) | GEMM1 median us | GEMM1 vs 93665e | MOE e2e samples (us) | MOE e2e median us | MOE e2e vs 93665e | random pass | hash |
|---|---|---:|---:|---|---:|---:|:---:|---|
| baseline_93665e | 689.116 | 689.116 | +0.00% | 1664.90 | 1664.90 | +0.00% | not-run | not-run |
| sync_mg4_fc8 | 622.669 | 622.669 | +9.64% | 1595.48 | 1595.48 | +4.17% | not-run | not-run |
| sync_mg2_fc12 | 623.041 | 623.041 | +9.59% | 1597.83 | 1597.83 | +4.03% | not-run | not-run |
| sync_mg4_fc28 | 619.572 | 619.572 | +10.09% | 1597.08 | 1597.08 | +4.07% | not-run | not-run |
| sync_mg4_fc28_apre | 564.597 | 564.597 | +18.07% | 1447.19 | 1447.19 | +13.08% | not-run | not-run |
| sync_mg4_fc28_apre_exactopt | 546.656 | 546.656 | +20.67% | 1427.16 | 1427.16 | +14.28% | not-run | not-run |
| sync_mg4_fc28_hard | 594.398 | 594.398 | +13.74% | 1567.96 | 1567.96 | +5.82% | not-run | not-run |
| sync_mg4_fc28_relu | 576.660 | 576.660 | +16.32% | 1548.27 | 1548.27 | +7.01% | not-run | not-run |

按“只改变一个直接父版本”的增量关系看：

| child vs direct parent | GEMM1 | MOE e2e | 解读 |
|---|---:|---:|---|
| `mg2/fc12` vs `mg4/fc8` | `-0.060%` | `-0.147%` | 单 sample 下基本持平 |
| `mg4/fc28` vs `mg4/fc8` | `+0.497%` | `-0.100%` | GEMM1 小幅更快，e2e 在噪声范围 |
| `apre` vs `mg4/fc28` | `+8.873%` | `+9.385%` | producer-consumer A layout 是主要结构性收益 |
| `exactopt` vs `apre` | `+3.178%` | `+1.384%` | 四项 exact scheduling/output 优化的组合收益 |
| `hard` vs `mg4/fc28` | `+4.063%` | `+1.823%` | 以 activation 近似换取 epilogue 降时 |
| `relu` vs `mg4/fc28` | `+6.926%` | `+3.056%` | 更激进的 activation 语义变化 |

最终应按两条互不混淆的轴理解这 7 个版本：

```text
exact 数据/数学路径：
  sync_mg4_fc8
    -> sync_mg2_fc12 或 sync_mg4_fc28        # scheduler shape
    -> sync_mg4_fc28_apre                     # A producer-consumer layout
    -> sync_mg4_fc28_apre_exactopt            # WPT2/xdl0/reuse/output overlap

近似 activation 对照：
  sync_mg4_fc28
    -> sync_mg4_fc28_hard
    -> sync_mg4_fc28_relu
```

在“接口、功能、精度等价”的约束下，本轮应把
`sync_mg4_fc28_apre_exactopt` 视为最终 exact 版本；hard/ReLU 的性能数字只
用于量化 exact-SiLU epilogue 的潜在成本上限，不能作为等价替代。
