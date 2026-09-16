# all-NT-RT source 与 final WPT2 adjacent 的 thread-trace 对比

## 结论

`final WPT2 adjacent` 的目标是把原来 `32/32/2/2 KiB` 的四-wave TDM
ownership 改为每个 wave `17 KiB`，以缩小 wave 到达 barrier 的时间差。这个目标只在
workgroup 内的局部 task 延迟上实现了；它没有缩短共享 TDM 的实际关键路径，反而引入了更多
TDM descriptor 和 scalar/control 指令。

在 a07-3 的同机结果中：

```text
all-NT-RT source GEMM1 median = 506.090 us
final WPT2 adjacent median    = 512.701 us
regression                    = 6.611 us, 1.31%
```

ATT 的完整 selected-SIMD trace span 也给出几乎相同的方向：

```text
all-NT-RT source median span = 1,231,236 cycles
final WPT2 median span       = 1,249,387 cycles
regression                   = 18,151 cycles, 1.47%
```

后者性能更低的主要原因是：

1. gfx1250 的 TDM 按 SIMD pair 共享。source 的两个 physical SIMD pair 本来都承担
   `34 KiB/stage`，WPT2 没有减少任一 TDM 的 byte traffic，也没有改善 pair 级负载。
2. WPT2 把每个 pair 的 input descriptor 数从 `2` 条/stage 增至 `4` 条/stage。每个
   wave 的 input TDM 数从一条/stage 增至两条/stage。
3. steady task 中动态 `tensor_load_to_lds` 从约 `32` 增至 `62` 条，动态指令从
   `7,341.5` 增至 `7,812.5` 条，增加 `6.42%`。
4. scalar/control issue timeline 从 `1,961` 增至 `3,302.5 cycles/task`，增加
   `1,341.5 cycles`、`68.41%`。新增的 `s_cmp_lg_u32`、`s_cbranch_scc0`、descriptor
   address update 和第二条 TDM issue 是主要来源。
5. `s_wait_tensorcnt` stall 从 `1,507.5` 增至 `1,717 cycles/task`，增加 `13.90%`。
   同一 wave 内 payload TDM 后紧邻的 scale TDM 仍受 in-order completion 和
   descriptor-slot/backpressure 约束。
6. 预期要降低的六次 K-ring `s_barrier_wait 0xfffd` 只从 `2,674.5` 降至
   `2,622.5 cycles/task`，仅减少 `52 cycles`、`1.94%`；四个 owner 的 ring-wait
   spread 反而从 `705` 增至 `772 cycles/task`。

因此，这次移植把部分原本属于早到 wave 的 barrier idle 转换成了所有 wave 都必须执行的
descriptor/control 工作。barrier stall 是并行等待，不能把四个 wave 的等待时间相加后视为
可直接回收的 kernel 时间。由于 source 的两个共享 TDM 已经是 byte-balanced，WPT2 没有降低
最后到达者的关键路径，只增加了所有路径上的管理成本。

## 对比版本

```text
all-NT-RT source:
my_code/moe_gemm1_act1_optimized/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s
SHA256=8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169

final WPT2 adjacent:
my_code/moe_gemm1_act1_optimized/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_exactopt_wpt2_owner.s
SHA256=dee177fc1c01e2b1c3efea59b5d200fd43d0993d7543a5e8acb9af9de452a358
```

WPT2 生成器为：

```text
my_code/moe_gemm1_act1_optimized/build_all_nt_rt_exactopt_wpt2_owner.py
SHA256=dbcad3d7e78402313215e906822df6111de10432a523f3b8f1cf66579470a332
```

这里的 `adjacent` 表示每个 stage 按 `payload -> scale` 连续发射两条 input TDM。
曾测试过在两条 TDM 中间插入一次 WMMA 的 stagger 版本，GEMM1 从 `512.701 us` 回退到
`517.920 us`，因此最终版本保留 adjacent 顺序。

## 测试和 trace 条件

正式性能结果目录：

```text
my_code/moe_gemm1_act1_optimized/history_runs/
  heliosr-1b114-a07-3_20260916T123333Z_e2e-const0/
```

三个 round 交替运行两个 case：

| case | GEMM1 samples | median | MoE e2e samples | median |
|---|---:|---:|---:|---:|
| all-NT-RT source | `506.090, 506.233, 504.650 us` | `506.090 us` | `1333.76, 1332.52, 1333.30 us` | `1333.30 us` |
| final WPT2 adjacent | `514.738, 512.701, 510.155 us` | `512.701 us` | `1341.57, 1341.52, 1336.73 us` | `1341.52 us` |

两个版本的 const0 MoE e2e 均正确：

```text
logits_diff=0
rel_l2=0
MoE output hash128=21291d9023c8af8a6324fe20f346a967
ref output hash128=21291d9023c8af8a6324fe20f346a967
```

WPT2 的 random MoE e2e 也已通过：

```text
logits_diff=3.3980e-06
rel_l2=2.6069e-03
pass=True
```

性能运行前后均确认 GPU/KFD 空闲。ATT 在同一台 a07-3 上以 `sclk=2400 MHz`、GPU use
`0%`、VRAM `0%` 的状态开始，命令为：

```bash
AITER_HISTORY_CASE_LIST=\
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt,\
candidate \
AITER_HISTORY_CANDIDATE=\
my_code/moe_gemm1_act1_optimized/\
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_\
exactopt_wpt2_owner.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
RUN_VERIFY=0 \
AITER_ATT_E2E_ITERS=2 \
AITER_ATT_SIMD_LIST=0,1,2,3 \
AITER_ATT_TIMEOUT_SECONDS=300 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att
```

ATT 结果目录：

```text
my_code/moe_gemm1_act1_optimized/history_runs/
  heliosr-1b114-a07-3_20260916T131009Z_att/att/
```

两个 case 的 SIMD0、SIMD1、SIMD2、SIMD3 均有且仅有一份非空 `.att`、一份
`code.json` 和一份 wave JSON。分析使用：

```text
my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py
SHA256=6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

persistent task 以连续出现的以下序列划分：

```asm
s_add_co_u32 s28, s28, 16
s_cmp_lt_u32 s28, 0x240
s_cbranch_scc0 5
```

每个 SIMD 的第一个 interval 被丢弃，以下 steady-task 数据来自每个 capture 的 17 个
interval。本文主要使用每个 capture 的 median，再对四个 capture 取 median。不同分类的
median 不能严格相加，表格用于比较变化方向和定位瓶颈。

## ownership 和共享 TDM 负载

本次 trace 中 logical owner 到 physical SIMD 的映射为：

| physical SIMD | all-NT-RT source | final WPT2 adjacent |
|---:|---|---|
| SIMD0 | full A，`32 KiB/stage` | A lower half + ScaleA lower half，`17 KiB/stage` |
| SIMD1 | full ScaleB，`2 KiB/stage` | B upper half + ScaleB upper half，`17 KiB/stage` |
| SIMD2 | full ScaleA，`2 KiB/stage` | B lower half + ScaleB lower half，`17 KiB/stage` |
| SIMD3 | full B，`32 KiB/stage` | A upper half + ScaleA upper half，`17 KiB/stage` |

MI400 Shader Programming Guide §1.4.2.1 和 §1.5 说明，每个 SIMD pair 共享通向
LDS/WGP$ 的 bus 和一个 TDM。按本次实际映射，两个版本的 pair 级工作量为：

| physical SIMD pair | all-NT-RT source | final WPT2 adjacent |
|---|---|---|
| SIMD0 + SIMD1 | `32 + 2 = 34 KiB/stage`，2 descriptors | `17 + 17 = 34 KiB/stage`，4 descriptors |
| SIMD2 + SIMD3 | `2 + 32 = 34 KiB/stage`，2 descriptors | `17 + 17 = 34 KiB/stage`，4 descriptors |

WPT2 只改善了 pair 内两个 wave 的 byte 分配，没有降低共享 TDM 的总 bytes。对真正执行数据
搬运的共享单元来说，输入流量不变，descriptor 数翻倍。

## 完整 selected-SIMD trace span

完整 wave JSON 的 `duration` 与 profiler 的方向一致：

| SIMD | source owner | source cycles | WPT2 owner | WPT2 cycles | change |
|---:|---|---:|---|---:|---:|
| 0 | A | 1,230,470 | A0 + ScaleA0 | 1,255,546 | `+2.04%` |
| 1 | ScaleB | 1,222,306 | B1 + ScaleB1 | 1,277,956 | `+4.55%` |
| 2 | ScaleA | 1,232,002 | B0 + ScaleB0 | 1,233,786 | `+0.14%` |
| 3 | B | 1,277,749 | A1 + ScaleA1 | 1,243,228 | `-2.70%` |
| cross-SIMD median | — | **1,231,236** | — | **1,249,387** | **`+1.47%`** |

这里的 span 是 selected-SIMD capture 的时间范围，用来比较两个版本的相对方向；它不是完整
dispatch 的绝对 kernel cycle 数。四个 SIMD 是顺序抓取的独立 dispatch，因此不能把四行相加。

source 的最慢 capture 是 SIMD3 的 full-B owner。WPT2 确实缩短了该 capture，但新的最慢
capture 变成 SIMD1 的 B1+ScaleB1，其 span 为 `1,277,956 cycles`，与 source 的最慢值
`1,277,749 cycles` 基本相同。换言之，关键路径只是从一个 owner 移到了另一个 owner，并未被
缩短。

## steady task 延迟和 wave 均衡

| case | role | SIMD | median cycles/task | dynamic instructions/task | input TDM/task |
|---|---|---:|---:|---:|---:|
| source | A | 0 | 33,034 | 7,347 | 32 |
| source | ScaleB | 1 | 31,878 | 7,346 | 32 |
| source | ScaleA | 2 | 33,160 | 7,336 | 32 |
| source | B | 3 | 37,110 | 7,337 | 32 |
| WPT2 | A0 + ScaleA0 | 0 | 32,054 | 7,776 | 62 |
| WPT2 | B1 + ScaleB1 | 1 | 32,030 | 7,853 | 62 |
| WPT2 | B0 + ScaleB0 | 2 | 32,348 | 7,849 | 62 |
| WPT2 | A1 + ScaleA1 | 3 | 32,515 | 7,760 | 62 |

四个 owner 的 task median spread 从：

```text
source: 37,110 - 31,878 = 5,232 cycles
WPT2:   32,515 - 32,030 =   485 cycles
```

降到了约十分之一，说明 WPT2 的 per-wave ownership 均衡确实生效。四-owner task median
从 `33,097` 降到 `32,201 cycles`，降低 `2.71%`。

但完整 trace 中动态 event median 从 `264,323` 增至 `281,216.5`，增加 `6.39%`；完整
span 反而增加 `1.47%`。WPT2 的 cycles/event 从约 `4.658` 降至 `4.443`，表明单条指令
平均等待更少，但总指令数增加得更多。这也解释了为何局部 steady-task median 看起来更好，最终
kernel wall time 却更差。

## wait 对比

下表为四个 owner 的 per-capture median 再取 median：

| category | source cycles/task | WPT2 cycles/task | change | WPT2 task share |
|---|---:|---:|---:|---:|
| all `s_barrier_wait` stall | 6,288.0 | 5,474.5 | `-813.5` / `-12.94%` | 17.00% |
| `s_wait_dscnt` stall | 4,556.0 | 3,649.0 | `-907.0` / `-19.91%` | 11.33% |
| `s_wait_tensorcnt` stall | 1,507.5 | 1,717.0 | `+209.5` / `+13.90%` | 5.33% |
| `s_wait_idle` stall | 9.0 | 9.0 | 0 | 0.03% |
| six K-ring `s_barrier_wait 0xfffd` | 2,674.5 | 2,622.5 | `-52.0` / `-1.94%` | 8.14% |

使用互不重叠的 issue-timeline 分类，cycle 减少较多的部分如下。占比以 WPT2 representative
steady task 的 `32,201 cycles` 为分母：

| rank | reduced category | source cycles/task | WPT2 cycles/task | saved cycles/task | share of WPT2 task | share of all listed savings |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `DScnt wait` timeline | 4,826.0 | 4,003.0 | **823.0** | **2.56%** | **36.84%** |
| 2 | barrier-wait timeline | 6,366.0 | 5,549.0 | **817.0** | **2.54%** | **36.57%** |
| 3 | WMMA issue timeline | 9,490.0 | 9,187.5 | **302.5** | **0.94%** | **13.54%** |
| 4 | LDS-read issue timeline | 3,395.5 | 3,125.0 | **270.5** | **0.84%** | **12.11%** |
| 5 | LDS-write issue timeline | 145.0 | 124.0 | **21.0** | **0.07%** | **0.94%** |
| | total listed savings | — | — | **2,234.0** | **6.94%** | 100% |

这里的 WMMA、LDS-read 和 LDS-write 指令数没有减少。相应 timeline 下降表示这些指令之后
暴露给 selected wave 的 issue gap 变短，而不是 GEMM FLOP 或 LDS request 数减少。按纯
wait-stall 字段统计，`s_wait_dscnt` 实际 exposed stall 从 `4,556` 降至 `3,649
cycles/task`，减少 `907 cycles/task`，相当于 WPT2 task 的 `2.82%`；它与上表的
`DScnt wait` timeline 是同一现象的两种统计口径，不能相加。

barrier savings 内部以 workgroup barrier 为主。按 exact-site median 求和：

| barrier subset | source cycles/task | WPT2 cycles/task | saved cycles/task | share of WPT2 task |
|---|---:|---:|---:|---:|
| workgroup `s_barrier_wait 0xffff` | ~2,546.0 | ~1,658.0 | **~888.0** | **~2.76%** |
| all cluster `s_barrier_wait 0xfffd` | ~3,228.0 | ~3,047.5 | **~180.5** | **~0.56%** |
| six dominant K-ring cluster waits | 2,674.5 | 2,622.5 | **52.0** | **0.16%** |

exact-site median 的求和与 per-task total median 不满足严格可加性，因此上表用于判断 barrier
savings 的来源。它清楚显示：大部分减少来自 workgroup barrier；原计划要优化的 dominant
K-ring cluster barrier 只贡献了约 `52 cycles/task`。

六次 K-ring wait 的各 owner 汇总如下：

| case | owner stalls，cycles/task | cross-owner median | owner spread |
|---|---|---:|---:|
| source | A `2790`，ScaleB `2468`，ScaleA `3173`，B `2559` | 2,674.5 | 705 |
| WPT2 | A0/SA0 `2600`，B1/SB1 `2121`，B0/SB0 `2645`，A1/SA1 `2893` | 2,622.5 | 772 |

WPT2 没有降低 dominant K-ring barrier 的占比：source 为约 `8.08%`，WPT2 为约
`8.14%`。硬件文档 §4.3.6.6 说明 cluster barrier 要等待 cluster 内所有 workgroup
signal；因此只均衡一个 workgroup 内的四个 wave，不足以保证 cluster 的最后到达者提前。

## 为什么减少的 `2,234 cycles` 大于增加的 `1,661 cycles`，kernel 仍然回退

前面的增加/减少表不能相减后预测完整 kernel 时间，原因有三项：

1. 表格只统计 `trace_segment_cycles.py` 选中的 steady-task interval。相同 task boundary
   同时作为 start/end 时，36 个 boundary occurrence 被配成了交替 interval；丢弃第一个后，
   每个 capture 的表格只使用 17 个 interval，没有覆盖首次 full-setup task，也没有覆盖另一
   组交替 task。
2. 每个分类先在一个 SIMD capture 内取 median，再对四个独立 capture 取 median。不同分类的
   median 通常来自不同 task、不同 SIMD，数学上不满足可加性。
3. barrier stall 是 per-wave idle。早到 wave 减少 1,000 cycles 的等待，不代表 cluster 的
   最后到达者或完整 dispatch 同时缩短 1,000 cycles。

为得到可相加的结果，直接用全部连续 task boundary 将每个 selected-SIMD trace 划分成：

```text
initial/full-setup region + first task
35 consecutive persistent body tasks
post/final drain region
```

下表对四个 SIMD capture 取 arithmetic mean；同一列的三段可以严格相加为 full trace mean：

| trace region | source cycles | WPT2 cycles | WPT2 - source | share of WPT2 full-trace mean |
|---|---:|---:|---:|---:|
| initialization + first full-setup task | 39,479.0 | 62,155.0 | **+22,676.0** | **+1.81%** |
| following 35 persistent body tasks | 1,195,262.8 | 1,186,808.5 | **-8,454.3** | **-0.68%** |
| post/final drain | 3,938.8 | 2,251.3 | **-1,687.5** | **-0.13%** |
| full selected-SIMD trace mean | **1,238,680.5** | **1,251,214.8** | **+12,534.3** | **+1.00%** |

这给出了无矛盾的加法关系：

```text
+22,676.0 - 8,454.3 - 1,687.5 = +12,534.2 cycles
```

完整 trace 覆盖约 36 个 task。把上述成本摊到每个 task：

```text
first-task/full-setup penalty amortized = 22,676 / 36 = +629.9 cycles/task
steady body saving                    =  8,454 / 35 = -241.5 cycles/task
post saving amortized                 =  1,687 / 36 =  -46.9 cycles/task
full-trace net                        = 12,534 / 36 = +348.2 cycles/task
```

因此 steady body 的确变快，但第一 task 的冷启动/full-setup penalty 更大，最终仍净回退。
full-trace mean 回退 `1.00%`，full-trace median 回退 `1.47%`，与 profiler 的 `1.31%`
方向和量级一致。

首次 full-setup region 的四个 physical SIMD 分别为：

| SIMD | source cycles | WPT2 cycles | increase |
|---:|---:|---:|---:|
| 0 | 41,500 | 56,539 | +15,039 |
| 1 | 41,170 | 81,117 | **+39,947** |
| 2 | 34,617 | 50,339 | +15,722 |
| 3 | 40,629 | 60,625 | +19,996 |

最大问题在 WPT2 SIMD1 的 `B1 + ScaleB1` 初始路径。该 capture 在首次 task 中出现两处很长
的 workgroup barrier stall：

```text
0x9d48  s_barrier_wait 0xffff  stall=8,335 cycles
0x95dc  s_barrier_wait 0xffff  stall=7,109 cycles
```

另外，WPT2 首次 task 中在 SiLU/output epilogue 的多条 `v_pk_mul_f32`、`v_exp_f32`、
`v_rcp_f32`、`v_swap_b32`、`v_cvt_pk_bf16_f32` 和 `ds_store_b64` 后出现约
`1.7k–3.1k cycles` 的无 issue gap。对应指令的 trace latency 仍只有 `1–2 cycles`、stall
字段为 0，所以这些 gap 不是 SiLU 指令本身变慢；它们表示下一条指令/依赖尚未 ready 时暴露的
scheduler、scoreboard 或同步空泡。initial region 中归因到普通 VALU 后方的 issue gap 的
cross-SIMD median 从约 `4,028` 增至 `26,929.5 cycles`，增加 `22,901.5 cycles`。

WPT2 的第一个 task 尚未进入稳定的 next-task stage-0 prefetch/output-overlap 状态，同时要建立
payload half 和 scale half 两套 descriptor，并发射双倍 input TDM。后续 persistent task 可以
用已有 pipeline 隐藏一部分成本，所以只看 steady interval 会得到偏乐观的结论。完整 kernel
回退的直接原因是这个首次 full-setup/first-task penalty，而不是 steady body 中减少项小于增加项。

## 新增 descriptor/control 成本

WPT2 的 current-task input load 理论上从每 wave `28` 条变为 `56` 条。本文 interval 还包含
persistent boundary/prefetch 路径，因此动态计数表现为：

```text
tensor_load_to_lds: 32 -> 62 per task interval
TDM issue group:    34 -> 64 events/task  # 另外两条是 output tensor store
```

两条 adjacent input TDM 的动态 issue timestamp 间隔稳定为 `19 cycles`。每个 task 约有
31 组这样的 pair，仅 `payload -> scale` 串行 issue 距离就约为：

```text
31 * 19 = 589 cycles/task
```

这个距离中包含两条 TDM 之间的 scalar setup，并可能与其他 execution pipe 并行，不能再次与
后续各分类的 cycle 数相加；它用于说明每个 stage 新增了实际的串行 descriptor issue 路径。

从 source 到 WPT2，主要 opcode 的动态增量如下。`issue-gap` 是 ATT 中从该指令 timestamp
到下一条指令 timestamp 的归因，不等同于 ISA 手册中的单指令 architectural latency：

| opcode | extra hits/task | extra issue-gap cycles/task |
|---|---:|---:|
| `s_mov_b32` | +90.0 | +120.5 |
| `s_and_b32` | +38.0 | +53.0 |
| `s_add_co_u32` | +33.5 | +40.5 |
| `s_add_co_ci_u32` | +33.5 | +35.0 |
| `s_cbranch_scc0` | +31.0 | +478.5 |
| `s_cmp_lg_u32` | +31.0 | +301.0 |
| `tensor_load_to_lds` | +30.0 | +90.0 |
| `s_or_b32` | +30.0 | +40.0 |
| `s_cselect_b32` | +29.0 | +36.5 |
| `s_cmp_lt_u32` | +28.0 | +28.0 |

全部 SALU/control 的动态 hits 从 `869.5` 增至 `1,310/task`，增加约 `50.66%`；其
issue timeline 从 `1,961` 增至 `3,302.5 cycles/task`，增加 `68.41%`。其中
`s_cmp_lg_u32` 和 `s_cbranch_scc0` 是每条附加 scale descriptor 的有效范围检查；后者后面
紧接第二条 TDM，因此它的 issue gap 还包含 TDM descriptor slot/TX FIFO backpressure。

硬件文档 §4.10.1 和 §4.10.8 给出三项与这里直接相关的约束：

1. `TENSORcnt` 按 TDM instruction 计数，而不是按搬运 byte 数计数；
2. 同一 wave 的 tensor instructions 按序完成；
3. 从 issue 到 XACK，每 wave 最多 3 个、每 SIMD 最多 6 个 tensor ops in flight，TDM
   请求还会经过 VMEM-ARB/TX FIFO。

因此，把一个 32 KiB descriptor 拆成两个 16 KiB descriptor，再给同一 wave 增加一个
1 KiB scale descriptor，并不是零成本的 byte-preserving 变换。即使总 bytes 不变，WPT2
也更频繁地触及 descriptor issue、XACK slot 和 FIFO arbitration。trace 中
`s_wait_tensorcnt` 的动态次数仍为 `31/task`，但 steady ring threshold 从 source 的
`0x2` 改为 WPT2 的 `0x4` 后，stall 增加了 `209.5 cycles/task`。

## code footprint 和资源占用

| metric | source | WPT2 | change |
|---|---:|---:|---:|
| decoded `code.json` rows | 7,397 | 8,619 | `+1,222`, `+16.52%` |
| kernel symbol size | 46,696 B | 53,728 B | `+7,032 B`, `+15.06%` |
| code object size | 52,112 B | 59,152 B | `+7,040 B`, `+13.51%` |
| static `tensor_load_to_lds` sites | 60 | 120 | `+100%` |

两个版本的资源元数据完全相同：

```text
.amdhsa_group_segment_fixed_size 327680  # 320 KiB LDS/WG
.amdhsa_next_free_vgpr 1024
.amdhsa_next_free_sgpr 104
.amdhsa_private_segment_fixed_size 0
```

因此 WPT2 没有通过 occupancy 获得补偿。MI400 Shader Programming Guide §1.5 和 §5.7.1
说明每个 SIMD 有独立 SQ 和 4 KiB Instruction Store cache，WGP 共享 64 KiB L1 instruction
cache。WPT2 的代码体积增长可能增加 instruction-fetch 压力，但本次 ATT 没有给出可将
`1.31%` 回退单独归因于 I-cache 的直接 counter；它应视为次要风险，而不是已证实的主因。

## 为什么“barrier 更少”仍然更慢

source 的 ScaleA/ScaleB wave 很早到达 barrier，因此它们记录了较多 barrier stall。该
stall 与 A/B owner 正在执行的工作并行存在，不能按四个 wave 求和后当作串行开销。WPT2 将
A/B 的一半 payload 和对应 scale 分给每个 wave 后：

1. 四个 wave 的 task elapsed time 更接近；
2. workgroup barrier stall 明显减少；
3. 共享 TDM pair 的 bytes 和最慢 capture span 没有下降；
4. 每个 wave 都新增第二条 descriptor、范围检查、地址更新和 TDM issue；
5. 原先的并行 idle 被替换成了会占用 SQ/TDM issue 资源的真实指令。

这就是局部 task median 降低 `2.71%`，而完整 trace span 增加 `1.47%`、profiler 时间增加
`1.31%` 的原因。WPT2 在这份 persistent kernel 中改善了表面上的 wave 均衡，却没有改善
真正受共享 TDM 和 cluster 最后到达者限制的吞吐。

## 后续优化建议

如果继续沿 WPT2 路线，首先应降低 descriptor/control 成本，而不是继续调整 barrier：

1. 将 secondary scale descriptor 的不变量和 half offset 移出 28-stage hotloop，避免每
   stage 重复执行 `s_cmp_lg_u32`、`s_cbranch_scc0`、`s_mov_b32` 和地址选择。
2. 让 payload/scale descriptor 使用 resident SGPR state，只在 ring wrap 或 task boundary
   更新高位/界限；目标是把新增 SALU/control hits 从约 `440.5/task` 压到接近 0。
3. 保持现有 `persistent_overlap` 的 output-LDS 生命周期和 next-task prefetch 时序，先只替换
   descriptor state machine。
4. 每个实验同时观察三项门槛：完整 selected-SIMD trace span、六次 K-ring barrier 和
   `s_wait_tensorcnt`。仅降低早到 wave 的 barrier stall，不应视为性能收益。

从本次结果看，更低风险的方向仍是保留 source 的一条 input TDM/wave/stage ownership，在
不增加 descriptor 数的前提下优化 source 的 B 路径、LDS read schedule 或 K-ring 前的独立
工作。source 已在共享 TDM pair 层面做到 byte-balanced，这一点应作为下一轮设计的约束。

## 硬件资料

本报告使用本地以下资料核对 gfx1250 行为：

```text
mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/
  MI400_Shader_Programming#65.txt
```

相关章节：

- §1.4.2.1、§1.5，pages 17–19：LDS/WGP$、每 SIMD pair 的共享 TDM、SQ 和 instruction cache；
- §4.3.6.6：cluster barrier 的完成条件；
- §4.10.1，page 197：TENSORcnt 计数和同一 wave 内 tensor instruction 的 in-order 完成；
- §4.10.8，page 206：每 wave/SIMD 的 TDM in-flight descriptor 限制和 VMEM-ARB/TX FIFO；
- §5.7.1–§5.7.2，pages 239–240：instruction fetch 和 per-SIMD issue 能力。
