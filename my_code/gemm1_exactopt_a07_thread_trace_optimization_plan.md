# `sync_mg4_fc28_apre_exactopt` a07-3 thread-trace 分析与后续优化方案

## 1. 目标与结论

本文分析 `my_code/reproduce_compare.sh` 中的：

```text
sync_mg4_fc28_apre_exactopt
```

对应 GEMM1 symbol：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p
```

分析基于 2026-09-14 在 a07-3 新抓取的 SIMD0～SIMD3 四组独立 ATT
thread trace。最重要的结论是：

1. active wave 的平均生命周期为 `31,915 cycles`。其中 full-K hotloop
   占 `72.97%`，它仍是绝对主体。
2. 所有显式 `s_wait_*` 与 `s_barrier_wait` 的 stall 合计约
   `4,516 cycles/wave`，占 active-wave 生命周期的 `14.15%`。
3. 最大 wait 类别是 `s_barrier_wait`：`2,540 cycles/wave`，占 `7.96%`；
   其次是 `s_wait_tensorcnt`：`1,004 cycles/wave`，占 `3.15%`。
4. B/ScaleB owner 明显比 A/ScaleA owner 更晚完成 input TDM。A/ScaleA
   owners 因而主要把时间耗在 barrier 上，B/ScaleB owners 则主要耗在
   `s_wait_tensorcnt` 上。这是当前最明确、最值得优先处理的 pipeline imbalance。
5. 两个 output-owner waves 的 final `s_wait_tensorcnt 0` 平均约
   `824 cycles/wave`，非 output owners 只有约 `58 cycles/wave`。当前两阶段
   output overlap 已经有效，但第二阶段 output TDM 仍集中在同一组 owner 上。
6. 除 wait/barrier 外，representative hotloop 中只有一次
   `ds_load_b128` 超过 100 cycles；全体 576 条 active-wave 样本里的 LDS
   极端长尾非常稀少，对平均生命周期的独占贡献小于 `0.1%`，不是系统性瓶颈。
7. code object 使用 `278,528 B` LDS、`804 VGPR`、`58 SGPR`，无 VGPR/SGPR
   spill。LDS 和 VGPR 都独立把 occupancy 限制在每 WGP 一个 workgroup、每
   physical SIMD 一个 active wave slot。
8. 在不改变接口、数学、精度和动态 expert lookup 的前提下，下一步最合理的
   顺序是：
   - 先把已有相邻实现验证过的 all-input `NT_RT` hint 移植到 FlyDSL；
   - 再做 B/ScaleB-first 的同 TDM-pair issue stagger；
   - 然后把第二阶段 output TDM owner 轮换到当前 non-output wave；
   - 最后尝试把 invariant output/input descriptor setup 提前到 exact-SiLU
     的 VALU/TRANS 区间。
9. 这些改动的现实组合收益预计约 `3%～6%`。若仍要求再降 10% 以上，必须
   重新设计 tile/ring/occupancy，而不是继续调整一个普通 scheduling knob。

## 2. 抓取范围与可复现信息

### 2.1 远端环境

```text
host   = heliosr-1b114-a07-3.mnb.dcgpu
branch = hyg_gfx1250_gemm_a4w4
HEAD   = 25b2b9bb050a90f26a08a0e5048d9598dcf697c1
container = hyg_fyd1
GPU    = 1 x gfx1250
```

抓取前和抓取后均执行 `/data/yanguahe/code/gpu_users.sh`；两次都确认没有其它
GPU/KFD 进程。没有执行 `git clean`、`git checkout .` 或任何远端源码清理。

### 2.2 抓取命令的有效配置

`get_isa_runner_att.sh` 以 `--all-simd --ana-att` 运行，测试命令为：

```bash
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1 \
AITER_FLYDSL_GEMM1_MMA_GROUP=4 \
AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28 \
AITER_FLYDSL_GEMM1_SILU_HARD=0 \
AITER_FLYDSL_GEMM1_SILU_RELU=0 \
AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2 \
AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0 \
AITER_FLYDSL_GEMM1_WMMA_REUSE=1 \
AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE=1 \
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario verify \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --iters 2 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --const-init 0
```

本次 const-input correctness 为：

```text
logits_diff = 0
rel_l2      = 0
pass        = True
output_sha256 = bc3404a6147c932d6ff7c322658fc3258b7a0ca1b81adf2a381f73ed1348020e
```

### 2.3 原始 trace

远端未压缩目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_exactopt_trace_analysis_20260914/
  a07-3_sync_mg4_fc28_apre_exactopt_allsimd_20260914T0827Z/
```

大小约 `831 MiB`。压缩包：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_exactopt_trace_analysis_20260914/
  a07-3_sync_mg4_fc28_apre_exactopt_allsimd_20260914T0827Z.tar.gz
```

大小约 `139 MiB`。四个 capture 的 UI 目录分别是：

```text
SIMD0: ui_output_agent_5367_dispatch_16410
SIMD1: ui_output_agent_65042_dispatch_16410
SIMD2: ui_output_agent_44445_dispatch_16410
SIMD3: ui_output_agent_47394_dispatch_16410
```

### 2.4 本地分析产物

本地目录：

```text
my_code/gemm1_exactopt_trace_analysis_20260914/
```

包含：

- `phase_segments.json`：四阶段、每个 point 使用 trace CSV 中连续三条指令；
- `full_active_path.json`：active path drill-down 配置；
- `phase_segments_output.txt`：`trace_segment_cycles.py` compare-mode 输出；
- `full_active_path_output.txt`：full-path compare-mode 输出；
- `full_active_path_representative_trace.txt`：full-path representative timeline；
- `hotloop_representative_trace.txt`：K hotloop representative timeline；
- `analyze_exactopt_trace.py`：导入指定 parser 后生成聚合统计；
- `exactopt_trace_metrics.json`：机器可读的全部统计；
- `capture_summary.log`、`analyze_att_capture.log`：抓取状态和基础 ATT 分析。

分析使用规则文件指定的 `trace_segment_cycles.py`，固定 SHA256 为：

```text
6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

执行方式：

```bash
python3 my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py \
  my_code/gemm1_exactopt_trace_analysis_20260914/phase_segments.json

python3 my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py \
  my_code/gemm1_exactopt_trace_analysis_20260914/full_active_path.json \
  --specific-part-representative-trace
```

## 3. 统计口径与限制

### 3.1 四个 SIMD capture 是独立运行

SIMD0～SIMD3 是四次独立 ATT capture，不是四个同步时钟域的一次联合记录。
跨 capture 的绝对 timestamp 不能直接相减。本文只比较：

- 每个 wave 内部的 `end_ts-start_ts`；
- 每个 capture 内相同阶段的 cycle；
- 四个 capture 的等权平均；
- 同一静态 instruction 的 `stall`、`latency` 和动态 hit 数。

### 3.2 `latency` 与 `stall` 不能混为一谈

trace event 形状是：

```text
[timestamp, type, stall, latency, code_idx]
```

- 对 `s_wait_*`、`s_barrier_wait`，`stall` 是该 wave 真正暴露的阻塞时间，适合
  累加并除以整个 active-wave lifetime。
- 对 WMMA、LDS load、TDM 等异步或可重叠指令，`latency` 可能和后续独立指令
  重叠，不能直接相加后宣称占用同样多的 kernel 时间。
- 本文对非 wait 指令另用“到下一条动态指令的 timestamp gap”做互斥归因。
  该归因能完整分摊 wave lifetime，但标签表示“该指令之后的 issue gap”，不一定
  等于该指令自身执行延迟。例如 TDM issue 前后的 gap 可能被标到相邻 `s_mov`。

### 3.3 active 与 sentinel wave 分开

每个 SIMD-select capture 中：

```text
active waves   = 144
sentinel waves = 36
```

active wave 以动态执行过 `v_wmma_scale_f32_32x16x128_f4` 判定；sentinel wave
在 `m_tile_map` lookup 后提前退出。四个 capture 共提供 576 条 active-wave
样本。

## 4. Capture 健康度与整体时间

四个独立 capture 的最大 dispatch span：

| capture | 最大 shader cycles | 对应 SE |
|---|---:|---:|
| SIMD0-select | 1,200,924 | SE2 |
| SIMD1-select | 1,211,008 | SE1 |
| SIMD2-select | 1,225,916 | SE0 |
| SIMD3-select | 1,187,806 | SE1 |

16 个 capture×SE clock series 的加权平均 GFXCLK 为 `2,075.645 MHz`。最长的
SIMD2/SE0 span 对应约 `585.88 us`；kernel-trace 数据库中本次普通 profiling
dispatch 为 `549.713 us`。ATT 会改变执行环境，因此后续使用 cycle 比例判断
瓶颈，不把 ATT 时间当作新的 benchmark 基线。

occupancy 记录显示：

```text
每个 physical SIMD key 的 distinct slot IDs = 1
每个 physical SIMD key 的最大同时 active slots = 1
每个 slot 顺序执行 45 条 wave lifetimes
```

WGP completion imbalance 的 16 组结果均约 `0.011%`，最大仅 `0.0119%`。
所以 dispatch 尾部的 WGP 数量不均衡不是当前瓶颈。

active 与 sentinel 的平均 lifetime：

```text
active wave   = 31,915 cycles
sentinel wave =    731 cycles
```

每个 slot 对应 36 条 active waves 和 9 条 sentinel waves。按 wave-cycle work
估算，sentinel 只占：

```text
9*731 / (36*31915 + 9*731) = 0.57%
```

因此当前不应优先为消除 static-capacity sentinel grid 设计复杂机制。

## 5. LDS、SGPR、VGPR 与 occupancy

从本次 capture 对应 code object 的 AMDHSA metadata 读取：

| 资源 | code-object 值 | spill | 硬件含义 |
|---|---:|---:|---|
| LDS / group segment | `278,528 B = 272 KiB` | — | 4 个 69,632-B input ring slots |
| logical SGPR count | `58` | `0` | 包括当前 descriptor/control live state |
| logical VGPR count | `804` | `0` | accumulator、两槽 K128 rmem state 与 epilogue 临时值 |
| private segment | `0 B` | — | 没有 scratch allocation |

rocprof kernel-symbol 数据同时报告 `sgpr_count=128`。这与
MI400 Guide §3.3.1.1 一致：CDNA5 每个 wave 固定分配 128 个 physical SGPR，
其中 106 个是普通 SGPR、106/107 为 VCC。`58` 是 code object 实际声明的
logical SGPR 使用量，`128` 是运行时看到的固定 physical allocation。

资源占比：

```text
LDS: 278528 / 327680 = 85.00% of 320 KiB/WGP
VGPR metadata: 804 / 1024 = 78.52% of one SIMD's register pool
VGPR allocation granularity: ceil(804/16)*16 = 816 VGPR/wave
```

MI400 Guide §3.3.2.1 规定 MI450 每 SIMD 有 1024 VGPR、Wave32 以 16 VGPR
为粒度分配；§3.3.4 规定每 WGP 最多 320 KiB LDS、以 2 KiB 为粒度分配。
因此：

```text
2 waves/SIMD requires VGPR <= 512
2 workgroups/WGP requires LDS <= 160 KiB
```

当前若想翻倍 residency，必须同时做到：

```text
VGPR: 816 -> <=512，至少减少 37.25%
LDS : 272 KiB -> <=160 KiB，至少减少 41.18%
```

只减少少量 VGPR 或 LDS 不会改变 occupancy。此前 `tile_m=128,b3` 虽把 LDS
降到约 153 KiB，但 GEMM1 回退约 28%；所以“为了 occupancy 小幅压资源”不是
当前合适的第一优化方向。

## 6. Active-wave 阶段分解

四个阶段使用 trace CSV 中连续三条实际指令作为 anchor。最后另用每条 wave 的
完整 lifetime 补上 final LDS barrier、output descriptor/store 和最终
`s_wait_tensorcnt 0`。

四个 SIMD-select capture 等权汇总：

| 阶段 | cycles/active wave | 占完整 active wave |
|---|---:|---:|
| setup、binary search、descriptor 与 initial input TDM issue | 4,309.6 | 13.50% |
| initial ready wait + 完整 K hotloop | 23,288.3 | 72.97% |
| 第一半 exact-SiLU + 第一阶段 output TDM | 1,592.0 | 4.99% |
| 第二半 exact-SiLU，直到 final LDS barrier signal | 1,988.9 | 6.23% |
| final LDS barrier、output issue/drain 与 `s_endpgm` tail | 736.3 | 2.31% |
| 合计 | 31,915.1 | 100.00% |

因此：

```text
exact-SiLU/output 区域总计 = 1592.0 + 1988.9 + 736.3
                           = 4317.2 cycles
                           = 13.53%
```

K hotloop 接近四分之三；但其中仍包含 input TDM readiness、LDS reuse 和
cluster ring-wrap 等可优化 wait，并不全是不可压缩的 WMMA 算术。

## 7. 显式 wait 的 cycle 占比

以下数值直接累加 active-wave trace event 的 `stall` 字段，再除以完整
active-wave lifetime。四个 capture 等权：

| wait 类别 | stall cycles/wave | active-wave 占比 |
|---|---:|---:|
| `s_barrier_wait` | 2,540.3 | 7.96% |
| `s_wait_tensorcnt` | 1,003.7 | 3.15% |
| `s_wait_dscnt` | 580.9 | 1.82% |
| `s_wait_kmcnt` | 364.0 | 1.14% |
| `s_wait_xcnt` | 27.0 | 0.08% |
| **合计** | **4,515.8** | **14.15%** |

完全消除全部显式 wait 的不现实理论上限是时间下降 `14.15%`，即约
`1.165x` speedup。实际必须保留数据依赖和 cluster correctness，因此可实现
收益会显著低于这个上限。

硬件语义依据：

- MI400 Guide §4.3.7.2：`DScnt` 对每条 LDS 指令递增，并在 LDS load 数据到达
  VGPR 或 store 写入 LDS 后递减；`S_WAIT_DSCNT N` 等待 `DScnt<=N`。
- 同节规定 SMEM 使用 `KMcnt`；scalar-memory loads 可乱序返回，因此依赖点
  通常只能使用 `S_WAIT_KMCNT 0`。
- MI400 Guide §4.10.1：每条 TDM 指令递增 `TENSORcnt`，完成后递减；同 wave
  TDM 有序，不同 waves 之间无序。
- MI400 Guide §4.3.6.6：workgroup/cluster barrier 等待的是其它 wave/WG 的
  到达，不是本 wave 单独的 memory counter。

### 7.1 主要 wait site

把两个 owner branch 中语义相同的 PC 合并后：

| 语义位置 | ISA PC | stall cycles/wave | 占比 |
|---|---|---:|---:|
| initial input ready：TENSOR wait + WG barrier | `0x2604 + 0x267c` | 951.0 | 2.98% |
| steady cluster ring-wrap barrier | `0x30f0 / 0x41a4` | 825.7 | 2.59% |
| steady LDS-ready WG barrier | `0x34f4 / 0x45b0` | 601.2 | 1.88% |
| final LDS/output tail | `0xa2f0 + 0xa2f8 + 0xa3ac` | 738.4 | 2.31% |
| startup cluster barrier | `0x1f28` | 142.7 | 0.45% |
| steady `s_wait_tensorcnt` | `0x34e0 / 0x459c` | 189.4 | 0.59% |
| steady `s_wait_dscnt 0` | `0x34ec / 0x45a8` | 279.3 | 0.88% |
| ring-wrap local WG barrier | `0x38c0 / 0x497c` | 60.0 | 0.19% |
| first-half output LDS drain | `0x8944` | 64.5 | 0.20% |

最大系统性损失不是某一条孤立 wait，而是：

```text
input TDM owner 到达偏差
  -> 某组 wave 停在 TENSORcnt
  -> 另一组 wave 提前到达并停在 WG/cluster barrier
```

### 7.2 四个 logical wave 路径

根据第一组 TDM branch 和 output-store branch 的动态命中路径，本次 capture 中
physical SIMD selection 与 logical wave 的对应关系是：

| logical wave | ATT capture | input owner | output owner |
|---:|---|---|:---:|
| 0 | SIMD0-select | A + ScaleA | 是 |
| 1 | SIMD3-select | A + ScaleA | 否 |
| 2 | SIMD2-select | B + ScaleB | 是 |
| 3 | SIMD1-select | B + ScaleB | 否 |

这是本次 capture 中观测到的映射，不应推广成硬件永远把 logical wave N
调度到 physical SIMD N。

各路径统计：

| logical role | active-wave cycles | all wait stall | wait 占比 | non-wait cycles | setup | hotloop | epilogue 1 | epilogue 2 | tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wave0 A/SA + output | 32,287.4 | 5,076.6 | 15.72% | 27,210.8 | 4,019.1 | 23,495.2 | 1,623.6 | 2,177.2 | 972.3 |
| wave1 A/SA | 30,915.8 | 4,572.2 | 14.79% | 26,343.6 | 3,634.4 | 23,483.4 | 1,560.2 | 1,802.6 | 435.2 |
| wave2 B/SB + output | 32,898.5 | 4,171.5 | 12.68% | 28,727.0 | 5,101.5 | 22,949.2 | 1,629.6 | 2,252.6 | 965.7 |
| wave3 B/SB | 31,558.5 | 4,243.0 | 13.44% | 27,315.5 | 4,483.5 | 23,225.2 | 1,554.6 | 1,723.1 | 572.1 |

关键不平衡有两组。

第一组是 input TDM：扣除 final output wait 后，B/ScaleB owners 的
`s_wait_tensorcnt` 约为 `995 cycles/wave`，A/ScaleA owners 约为
`130 cycles/wave`。对应地，A/ScaleA owners 在 barrier 中等待得更久：

```text
A/ScaleA owner barrier stall ≈ 3,257 cycles/wave
B/ScaleB owner barrier stall ≈ 1,823 cycles/wave
```

这说明 B/ScaleB 路径是 input readiness 的慢侧，A/ScaleA wave 的高 barrier
stall 只是慢侧到达较晚的结果。只延迟 A/ScaleA 来“降低 barrier 数字”不会
缩短 critical path；必须让 B/ScaleB 更早 issue 或更早完成。

第二组是 output ownership：

```text
output-owner final s_wait_tensorcnt 0 ≈ 824 cycles/wave
non-output final s_wait_tensorcnt 0  ≈  58 cycles/wave

output-owner tail mean    ≈ 969 cycles
non-output-owner tail mean≈ 504 cycles
```

第一阶段 output store 已与第二半 SiLU 大量重叠，但第二阶段 store 的 owner
仍明显处于 critical path。

## 8. 其它长 latency 指令

`--specific-part-representative-trace` 使用 `long_lat_thr=100`。四个 full-path
representative waves 中，所有超过 100 cycles 的稳定事件都是
`s_wait_tensorcnt` 或 `s_barrier_wait`。

每个 logical role 最接近其 capture 平均值的 representative interval 中，最突出的
单次等待为：

| logical role | representative cycles | 主要长等待 |
|---|---:|---|
| wave0 A/ScaleA + output | 31,316 | `0x267c s_barrier_wait 0xffff`: 1,046 cycles |
| wave1 A/ScaleA | 30,471 | `0x267c s_barrier_wait 0xffff`: 821 cycles；`0x30f0 s_barrier_wait 0xfffd`: 526 cycles |
| wave2 B/ScaleB + output | 31,945 | `0x2604 s_wait_tensorcnt 0x6`: 903 cycles |
| wave3 B/ScaleB | 30,987 | `0x2604 s_wait_tensorcnt 0x6`: 1,228 cycles；`0x41a4 s_barrier_wait 0xfffd`: 199 cycles |

这组 representative 结果与全样本聚合一致：B-side 直接停在 input TDM wait，
A-side 则更早到达并在 workgroup barrier 等待 B-side。

K-hotloop representative traces 中：

- SIMD0 出现一次 `ds_load_b128`，latency `411 cycles`；
- 其余三个 representative hotloops 的 >100-cycle 项全部是 wait/barrier。

扫描全部 576 条 active waves 后，确实能看到极少数更大的
`ds_load_b128` outlier，例如最高约 `8,413 cycles`。但该静态 PC 在全部样本中
只命中一次，摊到平均 active wave 的互斥 issue-gap 贡献仅约：

```text
14.6 cycles/wave = 0.046%
```

其它长 LDS outlier 的平均贡献更小。因此它们适合继续监控，但不足以解释当前
kernel 的稳定性能上限。

按“当前指令到下一条动态指令的 timestamp gap”把完整 active-wave lifetime
互斥分摊，结果为：

| 前置指令类别 | cycles/wave | 占比 |
|---|---:|---:|
| WMMA | 13,445.4 | 42.13% |
| SALU/control | 5,191.9 | 16.27% |
| LDS read | 3,661.9 | 11.47% |
| other VALU | 2,991.3 | 9.37% |
| `s_barrier_wait` | 2,657.3 | 8.33% |
| EXP/RCP | 1,105.9 | 3.47% |
| `s_wait_tensorcnt` | 1,036.7 | 3.25% |
| `s_wait_dscnt` | 835.9 | 2.62% |
| LDS write | 403.9 | 1.27% |
| `s_wait_kmcnt` | 386.3 | 1.21% |
| TDM instruction | 170.5 | 0.53% |
| `s_wait_xcnt` | 28.0 | 0.09% |

这张表不是硬件执行单元利用率：例如某个 `s_mov_b32` 后面的 gap 可能实际来自
紧邻的 TDM descriptor/issue 或 dependency。可以可靠得出的结论是：

- WMMA 是最大且必要的计算主体；
- SALU/descriptor/control 仍有显著占比；
- LDS read 是第二大数据通路成本，但大多数 latency 已被 WMMA overlap；
- 系统性 >100-cycle 暴露主要集中在同步和 TDM readiness，而不是普通 VALU。

## 9. 当前性能瓶颈判断

按优先级排序：

### 9.1 第一瓶颈：B/ScaleB TDM readiness 与跨 wave 同步偏差

直接证据：

- B/ScaleB owners 的 input `TENSORcnt` stall 约为 A/ScaleA 的 `7.6x`；
- A/ScaleA owners 随后在 WG/cluster barrier 中等待；
- initial readiness 与 steady WG/cluster waits 合计占 active wave 的数个百分点；
- WGP completion imbalance 只有约 `0.011%`，问题发生在单个 workgroup/cluster
  内部，而不是 dispatch 尾部。

### 9.2 第二瓶颈：单 residency 下无法由其它 wave 隐藏 wait

LDS 与 VGPR 都把 occupancy 限制为一个 workgroup/WGP、一个 wave/physical
SIMD。遇到 TENSOR/LDS/barrier stall 时，没有第二个 resident wave 可供同一
SIMD 切换，因此 wait 会直接暴露到 wall time。

### 9.3 第三瓶颈：output owner 的第二阶段 TDM tail

final output tail 仍占 `2.31%`。两个 output owners 比 non-output peers 多承担
约 `765 cycles/wave` 的 final `TENSORcnt` stall，并且在 second-half epilogue
和 descriptor 路径上也更慢。

### 9.4 第四瓶颈：descriptor/SALU 与 LDS issue 密度

exclusive issue-gap 中 SALU/control 占 `16.27%`、LDS read 占 `11.47%`。
其中不少 SALU gap 紧邻 `tensor_load_to_lds`，不能解释成普通 `s_mov` 本身很慢；
它反映的是 descriptor construction、依赖和 TDM issue 形成的串行窗口。

### 9.5 不是当前主要瓶颈的部分

- sentinel work：约 `0.57%`；
- `m_tile_map`/SMEM 的全部 `s_wait_kmcnt`：约 `1.14%`；
- 偶发 `ds_load_b128` 极端 latency：平均贡献低于 `0.1%`；
- final WGP load balance：最大 completion imbalance 仅 `0.0119%`。

## 10. 后续优化方案

下面的方案均要求：

- 不改变 GEMM1 外部 ABI；
- 不改变 FP4/ScaleA/ScaleB 数据含义；
- 不改变 FP32 accumulation 与 exact-SiLU 公式；
- 保留动态 `m_tile_map` binary search；
- 保留 non-balanced routing 支持；
- 不删除 correctness-critical WG/cluster barrier。

### 10.1 P0：把已验证的 input-TDM `NT_RT` hint 移植到 FlyDSL exactopt

当前 FlyDSL `make_tdm_atom()` 已提供 `cache_modifier` 参数，但本 kernel 的
四类 input jobs 仍使用默认值 0。相关 persistent 实现上已经有同 shape、同
数据流的实验证据：所有 input TDM 使用 `NT_RT` 时保持 random exact，并取得
约 `0.97%` GEMM1 改善；只改 B/ScaleB 为 `NT_HT` 或混合 hint 的结果更弱。

建议先增加一个仅对目标 GEMM1 specialization 生效的 compile-time
`input_tdm_cache_modifier`，在 `add_tdm_loads()` 构造 A/B/ScaleA/ScaleB atom 时
统一传入 `NT_RT`：

```python
fx.rocdl.make_tdm_atom(
    ...,
    cache_modifier=input_tdm_cache_modifier,
)
```

该改动不改变 descriptor shape、地址、multicast mask、同步、数值或 ABI，且
实现面最小。它应先单独验证，以确认 persistent kernel 上的 cache 行为能否迁移
到当前 FlyDSL exactopt。现实预期约 `0.5%～1.0%`；若三轮同机收益低于
`0.5%`，不继续组合。

### 10.2 P1：B/ScaleB-first 的同 TDM-pair issue stagger

这是针对当前第一瓶颈的优先候选。

MI400 Guide 说明每个 SIMD pair 共享一个 TDM。当前 capture 的 physical mapping
使每个 SIMD pair 中同时存在一个 A/ScaleA owner 和一个 B/ScaleB owner。当前
`emit_hints()` 对两个 owner group 使用相同的 `tdm_schedule`，它们会在相近的
WMMA scheduling slot 竞争 TDM issue。

建议保持 owner、descriptor、数据范围和同步全部不变，只增加 compile-time
owner-class schedule：

```text
B/ScaleB owner: 保持或前移现有两个 sched_vmem slot
A/ScaleA owner: 向后错开一个 MMA_GROUP slot
```

目标不是延迟 A 来美化它的 barrier 数字，而是让同 SIMD-pair 中更慢的 B 请求
优先进入 TDM，使 B readiness 更早，从而同时降低：

- B owners 的 `s_wait_tensorcnt`；
- A owners 的后续 `s_barrier_wait`；
- cluster ring-wrap 到达偏差。

只测试两个有明确依据的 stagger：一组错开 1 个 `mg4` slot，另一组错开 2 个
slot。不要做无边界的 knob sweep。

相关 persistent 实现还有一条可借鉴但不能直接外推的证据：只把 B/ScaleB
steady TDM 提前时 random e2e 能通过，而同时提前 A/ScaleA/B/ScaleB 八条 TDM
曾造成 GPU hang。因此 FlyDSL prototype 也必须只移动 B-side，并保留现有
barrier generation 和 A-side 时序。

预期：现实收益 `1.5%～3%`；如果 B-side `TENSORcnt` 没有下降，应立即停止该
方向。历史的“cross ScaleA/ScaleB owner groups”改了 ownership 且回退
`0.45%`，本方案不重复该做法。

### 10.3 P2：第二阶段 output TDM owner 轮换

当前两次 `issue_output_half()` 都由 `wave_n==0` 执行，因此 logical wave0/2
同时承担第一、第二阶段 output descriptor/store。建议：

```text
第一半 output: 保持 wave_n==0
第二半 output: 改由同一 wave_m 的 wave_n==1
```

也就是：

```text
M-half 0: wave0 发第一半，wave1 发第二半
M-half 1: wave2 发第一半，wave3 发第二半
```

TDM source LDS 地址、global 地址、shape、`mn_oob` 和最终 bytes 全部保持不变，
只改变 issuer wave。所有 waves 本来就经过相同的 workgroup barrier，所以接口和
数值语义不变。

证据：output-owner tail 平均 `969 cycles`，non-output owner 为 `504 cycles`；
第二半 epilogue 到 final barrier 的 owner/non-owner 差也约 `452 cycles`。

理论上可重分配的窗口接近 `900 cycles`，占 `2.8%`；考虑新的 owner 也必须
承担第二次 store，现实收益预计 `1%～2%`。该方案保持两阶段 output，不重复
已经验证回退 `0.47%` 的 four-phase output 实验。

### 10.4 P3：在 exact-SiLU 中预构造第二阶段 output descriptor

当前 logical SGPR 使用量只有 58，低于 106 个普通 SGPR 上限，也没有 spill；
因此可尝试在不增加 VGPR/LDS 的前提下，把第二阶段 output TDM 的 invariant
descriptor fields 提前构造并保持在 SGPR 中：

```text
第一阶段 output 发出后
  -> 在第二半 v_exp/v_rcp/VALU 链期间穿插第二阶段 descriptor SALU
  -> final LDS barrier 后只修正动态地址/extent并立即 issue
```

目标是减少 output owner 在 final barrier 前后的串行 SALU/descriptor 窗口，
同时缩短 non-output owners 在 `0xa2f8 s_barrier_wait 0xffff` 的等待。

门禁：每个候选都必须检查 `.sgpr_count <= 106`、SGPR spill=0、VGPR spill=0。
若 descriptor live range 推高 VGPR 或造成 SGPR spill，立即回退。预期收益
`0.5%～1.5%`。

### 10.5 P4：仅在 P1 有效后尝试 region-safe 的提前 B refill

如果单纯 schedule stagger 能稳定降低 B-side wait，再考虑更深的 K pipeline：

- 当前 K128 所需 B/ScaleB 已全部从 LDS 读入 rmem 后，允许下一 K tile 的
  B/ScaleB TDM 提前复用对应 LDS sub-region；
- A/ScaleA sub-region 保持原复用时刻；
- 用现有 `DScnt` dependency 明确证明旧 B/ScaleB LDS reads 已完成；
- 不扩大 b4 ring，不删除 ring-wrap cluster barrier。

这相当于把 ring slot 从“整槽统一复用”细化为“A-side/B-side 分区复用”，可给
更慢的 B payload 多出接近一个 K128 compute window 的传输 lead。

这是高风险方案：必须证明 TDM overwrite 与所有 DS reads 无 WAR hazard，并对
random、non-balanced、不同 expert 边界做验证。只有 P1 证明 B lead 确实能转化
成 wall-time 收益后才值得实现。

### 10.6 暂不建议的方向

| 方向 | 原因 |
|---|---|
| 删除 binary search | 用户要求保留；`s_wait_kmcnt` 总占比仅 1.14%，历史 bisect7 还更慢 |
| 跳过 WG/cluster barrier | barrier 很贵，但它保护 multicast/ring reuse；不能以潜在 hang 或数据竞争换性能 |
| 单纯减少几个 VGPR/LDS | 不跨过 `512 VGPR` 和 `160 KiB LDS` 两个阈值就不会获得双 residency |
| 直接改 `tile_m=128,b3` | 虽满足 LDS 双驻留门槛，历史 exact 版本约回退 28% |
| 再拆 four-phase output | 历史 exact 实验回退 0.47% |
| 继续普通 `mg/fc` sweep | 当前三个 schedule case 差异已接近噪声，无法处理 TDM-pair 与 owner imbalance |
| hard-SiLU/ReLU | 改变 activation 语义，不满足精度等价要求 |
| 普通 persistent loop | 历史实现增加 SGPR/VGPR、发生 spill/同步回退，未优于 exactopt |

## 11. 收益上限与执行顺序

从 trace 看：

```text
所有显式 wait 的绝对上限     = 14.15%
exact-SiLU/output 全区域      = 13.53%
当前 final output tail         =  2.31%
sentinel 消除上限             =  0.57%
```

但 barrier/TENSOR/DScnt 中多数是正确性依赖，不可能全部删除。建议按以下顺序
实施，每一步只保留同轮、同机、通过 random/non-balanced 验证的版本：

1. `exactopt_input_nt_rt`
2. `exactopt_tdm_pair_stagger_bfirst_1slot`
3. `exactopt_tdm_pair_stagger_bfirst_2slot`
4. 选出 P1 winner 后测试 `exactopt_output_owner_rotate`
5. 在 winner 上测试 `exactopt_output_descriptor_hoist`
6. 只有 P1 明确有效时才实现 region-safe early B refill

现实组合预期：

```text
P0 input NT_RT hint        0.5%～1.0%
P1 B-first stagger         1.5%～3.0%
P2 output owner rotation   1.0%～2.0%
P3 descriptor overlap      0.5%～1.5%
组合后                     3.0%～6.0%（不能简单线性相加）
```

若前三项都无法稳定超过噪声，当前 4×4/b4/WPT2/exact-SiLU 设计已接近其局部
最优点。要获得 6% 以上，必须重新设计 tile/ring，使 LDS 与 VGPR 同时跨过双
residency 门槛，或实现经过严格 WAR/cluster 证明的分区 ring pipeline。

## 12. 每个候选的验证门禁

每次实验至少执行：

1. full-shape random balanced verification；
2. full-shape random non-balanced verification；
3. 与 exactopt baseline 比较 GEMM1 输出或最终 MoE hash；
4. 同机交错顺序至少 3 轮 timing；
5. 检查 `.sgpr_count`、`.vgpr_count`、spill 和 LDS；
6. 对 winner 再抓 all-SIMD ATT，确认下降的是 B-side `TENSORcnt`、对应 A-side
   barrier 或 output-owner tail，而不是把 stall 转移到另一个 critical wave；
7. 测试后再次检查 GPU/KFD，无残留进程。

止损标准：

- 单项收益小于约 `0.5%` 且 wait site 没有按预测下降；
- 任意 random/non-balanced hash 或 production accuracy gate 变化；
- 出现 SGPR/VGPR spill；
- cluster barrier generation 不一致、hang 或 watchdog；
- 只是降低某个提前到达 wave 的 barrier stall，却没有缩短最慢 logical wave 或
  dispatch span。
