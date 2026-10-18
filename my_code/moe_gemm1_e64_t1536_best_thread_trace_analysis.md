# E64/T1536 non-balanced 最优版 GEMM1：thread trace 与瓶颈分析

采集日期：2026-10-17；文档完成日期：2026-10-18（Asia/Shanghai）。本文对应 `a07-3` 的 `dsr1_ep4` 容器，以及原始完整 MoE benchmark 的 random、non-balanced workload。采集与初步审核于 **2026-10-17 15:28:50 UTC** 完成，分析使用已保存的四项 SIMD-select capture。

目前最有依据的优化重点是 **steady K 中 TDM/LDS 供数、WMMA 调度与 WG rendezvous 的耦合**。32 个完整 compute waves 的 elapsed lifetime 中，steady K 占 **67.17%**，三次 N-task 交接合计占 **14.09%**；主要 stall 热点是 WG barrier、WMMA、`tensorcnt` 和 LDS load/`dscnt`。实际执行的四份完整 ELF 与已验收最优版逐字节一致，private/VGPR spills/SGPR spills 均为 0。

这些数据定位了等待和调度发生的位置，尚不足以把瓶颈定为单纯 HBM 饱和、LDS bank conflict 或 activation。本文分别标明采样事实、源码/ISA 解释及待验证方向，不把 stall 百分比换算成承诺可节省的 μs。

## 1. 收尾结果与所抓版本

正式保留的 GEMM1 为：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist
```

源码：[mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py)。保留 cyclic64KiB data segments、独立 scale arena、packed BF16 amax 和 `MMA_GROUP=8`；没有改变完整 padded M/N/K、56 次 K128 累加顺序、BF16 rounding、native FP4/per32 E8M0、routing、tail/empty expert 或外部布局。

| 性能口径 | GEMM1（μs） | 解释 |
|---|---:|---|
| 同批冻结 baseline 三轮中值 | 99.152 | 正式性能基准 |
| 已验收最优版三轮中值 | **94.888** | 改善 4.264 μs，约 **4.30%** |
| v89 收尾新配对的 review 中值 | 94.703 | 同一正式源码的补测，不替换历史成绩 |
| v89 shared 收尾新配对中值 | 95.034 | 三轮均慢于对应 review，本次不合入 |
| 距离 80 μs | **14.888** | 相对 94.888 μs 还需改善约 **15.69%** |

v89 首批候选原始中值 94.357 μs 如实保留；当时没有按性能淘汰，而是保留待验收。收尾补测的 review/v89 为 `94.703/94.901`、`94.315/96.092`、`94.783/95.034` μs，六次前置与首次六秒后置均全机 idle，未复现收益。完整 11 cases equivalence、dynamic-input/poison/eager/graph replay 通过；十组 fallback 完整 ELF 与 review 相同，target descriptor/metadata 审核通过。没有因较快单批结果切换正式版本。

收尾详情与全部样本见 [优化记录](moe_gemm1_e64_t1536_nonbalanced_optimization.md) 和 [优化 Q&A](moe_gemm1_e64_t1536_nonbalanced_opt_qa.md)。本次收尾之后只抓取和分析正式版，未生成新 kernel 候选。

## 2. 环境、版本绑定与采集条件

| 项目 | 本次实际值 |
|---|---|
| host | `yanguahe@heliosr-1b114-a07-3.mnb.dcgpu` |
| container | `dsr1_ep4` |
| remote repo | `/data/yanguahe/code/wk_sp1/aiter`，host/container 共享 |
| local repo | `C:/Users/yanguahe/Documents/code/wk_sp1/aiter` |
| local / remote branch | `moe_a4w4_pr_refactor_wt` / `moe_a4w4_pr_refactor_wt2` |
| HEAD | `339589f900b3a880ec447f605886fe803cbf9649` |
| GPU agent | `gfx1250`，agent report：8 XCC、256 WGP、1024 SIMD、wave32、320 KiB LDS/WGP |
| profiler | `_rocm_sdk_core/bin/rocprofv3`，1.3.5 |
| decoder | `_rocm_sdk_devel/lib/librocprof-trace-decoder.so.0.2` |
| 采集脚本 | [get_att_b_rocm_sdk.sh](get_att_b_rocm_sdk.sh) |
| ATT 选择 | `--all-simd --iteration 28 --ana-att --archive` |
| 详细 instruction trace 目标 | YAML `att_target_cu: 1`，每个选中 SE 的 target WGP/SIMD |
| SE mask | `0xf`，本次看到 SE0–SE3 |
| 单项限时 | `timeout --signal=TERM --kill-after=5s 120s` |

源码、用户验证脚本和实际 code object 的绑定如下。Git HEAD 不能代替工作区源码 hash。

| 对象 | SHA256 |
|---|---|
| 正式 persistent source | `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790` |
| 用户 `verify_gemm1_fused_apre_quant.py` | `fecea66b97862c9e8ed822027871a7572d1ecd355bad62f297a5f8386b8d8c9d` |
| 本次 `get_att_b_rocm_sdk.sh` | `e91cd9fd3421bfc9fc04222af88bf05c2eaac27095c31fe3a0fecb610236c012` |
| CPU review 与四份 ATT 完整 ELF | `5f023d44fb65f0a38a8c22907775afeab049d92a0bad0698fec831901eb486ff` |
| 上述 ELF 的 `.text` | `507c1a9aedd0db3e3d06b83063870e6c0e56564228c89ff381f5f71b07748985` |

四份 target ELF 的 `.text`、`.rodata`、`.note` 以及完整 ELF 都相同。`.text` 大小为 28,800 B。这个审核证明 trace 抓到的指令、descriptor 和 metadata 是正式 review；不把它解释成 profiler 环境与正式 eager 计时完全等效。

每次 preflight/ATT 前均检查全机 GPU/KFD 所有权、利用率和 VRAM；有占用则每 6 秒等待。计时期间没有并发空闲检查。采集后的首次六秒检查为全机 idle。每项 benchmark、编译和分析均限时 120 秒，用户验证脚本与 kernel 在采集期间保持上述 hash。

为执行两分钟限制，采集脚本的 `run_logged` 将 `kill-after` 从 30 秒调整为 5 秒，`--ana-att` 分析也经过该限时入口。本次没有改变 benchmark 参数或 kernel 功能。

## 3. 原始 trace、归档和复现命令

远端原始目录和归档：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/thread_trace_runs/gemm1_best_review_iter28_att_20261017/
/data/yanguahe/code/wk_sp1/aiter/my_code/thread_trace_runs/gemm1_best_review_iter28_att_20261017.tar.gz
```

本地完整归档：[gemm1_best_review_iter28_att_20261017.tar.gz](../.codex_tmp/gemm1_t1536_20261017/logs/gemm1_best_review_iter28_att_20261017.tar.gz)。大小 **110,767,473 B，约 106 MiB**，SHA256：

```text
703afb86d505f23308962c596977f3fd61c98fba2a3bf79dc578af62490e32e5
```

本地选择性解包的证据目录：

```text
.codex_tmp/gemm1_t1536_20261017/captures/gemm1_best_review_iter28_att_20261017/
```

完整 tar.gz 已核对 SHA256；选择性解包保留 wave/code/occupancy/realtime、code object、CSV、YAML 和日志。归档完成后新增的 ELF/stall/phase/bottleneck 报告另行保存，未覆盖原始归档。

原 workload 在容器仓库根目录执行，`PYTHONPATH` 保证加载这棵仓库：

```bash
env PYTHONPATH="$PWD" \
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=false \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=5s 120s \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench --data-format a4w4 --act silu --no-bias \
  --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048 --iters 20
```

重抓 trace 时，在 host 使用已有 [trace driver](../.codex_tmp/gemm1_t1536_20261017/trace_review_final_host.sh)，选一个未使用的 label：

```bash
cd /data/yanguahe/code/wk_sp1/aiter
TRACE_LABEL=gemm1_best_review_iter28_att_reproduce_001 \
bash .codex_tmp/gemm1_t1536_20261017/trace_review_final_host.sh
```

driver 在 host 核对源码和 GPU 空闲，随后在 `dsr1_ep4` 内调用：

```bash
AITER_ATT_TIMEOUT_SECONDS=120 \
bash my_code/get_att_b_rocm_sdk.sh \
  a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist \
  "$TRACE_LABEL" "$workload" \
  --all-simd --iteration 28 --ana-att --archive
```

这里的 `workload` 是上面的完整命令转为单行；driver 已精确构造该变量。脚本自己逐次执行空闲检查，不加 `--no-wait-idle`。每项 SIMD-select 是独立运行，不能把四项拼成一次同步的 GPU 全景。

只复现 CPU 分析、不重抓 GPU trace：

```bash
trace_dir=my_code/thread_trace_runs/gemm1_best_review_iter28_att_20261017
timeout --signal=TERM --kill-after=5s 120s \
python3 .codex_tmp/gemm1_t1536_20261017/analyze_review_bottlenecks.py \
  "$trace_dir" --output "$trace_dir/gemm1_bottleneck_reproduce_001.json"
```

输出必须使用未存在的文件名，脚本会拒绝覆盖。当前正式分析报告为 `gemm1_bottleneck_analysis_v2.json`。

## 4. 采集完整性与 workload 核对

| SIMD-select capture | `.att` 数 | decoded wave files | 完整 compute waves | 目标 dispatch ID | 解码状态 |
|---:|---:|---:|---:|---:|---|
| 0 | 4 | 10 | 8 | 301 | 通过 |
| 1 | 4 | 18 | 8 | 301 | 通过 |
| 2 | 4 | 20 | 8 | 301 | 通过 |
| 3 | 4 | 14 | 8 | 301 | 通过 |
| 合计 | 16 | 62 | **32** | 各独立运行均为 301 | 没有 `.long` 或 `v_illegal` |

每个完整 compute wave 均有 **3,584 次 WMMA**：4 个 persistent N-tasks × 每 task 56 次 K128 × 每 K128 16 次 WMMA。`num_insts == num_stitched`，没有把不完整 compute wave 混入统计。其余 30 个 decoded waves 没有 WMMA，单独保留，不计入 compute phase/stall 分母。

原始 kernel trace 共有 46 次目标 GEMM1 dispatch；目标第 28 次为 dispatch301，四份 ATT UI 路径也均为 dispatch301。该选择已越过 benchmark warmup；本文使用“目标 kernel 第 28 次 dispatch”，不擅自换算成 torch profiler 第几轮。

preflight 中 46 次 dispatch 的平均为 **94.668 μs**、中值 **94.281 μs**，第 28 次为 **93.180 μs**。preflight 内的原 eager table 为 `94.893 μs/count19`，原 sanity `logits_diff=3.4733e-06`、`rel_l2=2.6357e-03` 通过。这些是采集诊断数据，正式优化成绩仍使用第 1 节的 94.888 μs。

### 有效 compute WG 与静态 capacity WG

preflight CSV 的 `Grid_Size_X=114688` 是 work-items，`Workgroup_Size_X=256`，因此实际 launch **448 WG**。结合 [grouped launcher](../aiter/ops/flydsl/grouped_moe_gfx1250.py)、persistent kernel 与 [benchmark accounting](test_flydsl_grouped_gemm_gfx1250.py) 源码：

```text
valid routes = 1536 × 8 = 12288
contiguous_m capacity = align_up(12288 + 64 × 256 − 8, 256) = 28672
capacity M256 tiles = 28672 / 256 = 112
N tiles = 4096 / 256 = 16
PERSISTENT_TASKS = 4
launched WG = 112 × 16 / 4 = 448

reported executed GEMM1 FLOPs = 962072674304
executed M256 tiles = FLOPs / (2 × 256 × 4096 × 7168) = 64
compute WG = 64 × 16 / 4 = 256
capacity-tail WG = 448 − 256 = 192
```

对应代码为 `grouped_moe_gfx1250.py` 的 `contiguous_m` worst-case capacity，以及 persistent launcher 的 `grid`、`persistent_m_context` 和 `if expert < n_experts`。随机 routing 的本次 executed-work accounting 表明 64 个 active M256 tiles；capacity-tail WG 进入统一 setup 后跳过 compute。它们不是 448 个等长的 GEMM WG。约 192 routes/expert 的均值也不意味着可以跳过 M256 padded 行或改成 const0 特例。

本次每个独立 capture 的 occupancy 有 896 次 wave start 和对应 end，等于 112 个 WG 的 8-wave launch，覆盖 mask 内的四个 SE。详细 instruction waves 只来自每个 SE 的 target WGP，范围更小；不能把 32 个 compute waves 当作全 GPU 的全部 waves。

## 5. phase 时间：大部分仍在 steady K

32 个完整 waves 的 elapsed lifetime 合计为 **4,299,004 shader cycles**。这是 wave lifetime 的和，包含并行 waves，不能当作单次 kernel 的 wall time。

每 task 按 896 次 WMMA 分组。前 768 次对应 24 个 steady K256 tiles，即 48 次 K128；最后 128 次对应四个 drain K256 tiles。elapsed 边界采用第一条 WMMA 的时间、第一条 drain WMMA 的时间和最后一条 WMMA 的 decoder end 时间；三段 task handoff 包含当前 epilogue、output 发布与下一 task prologue。这五类 span 精确覆盖每个 wave 的 `[begin,end)`。

| elapsed phase | 合计 shader cycles | 占采样 wave lifetime | 每 wave 平均 cycles |
|---|---:|---:|---:|
| initial prologue | 224381 | 5.219% | 7011.906 |
| **steady K** | **2887673** | **67.171%** | **90239.781** |
| drain K | 413009 | 9.607% | 12906.531 |
| **三次 task handoff 合计** | **605829** | **14.092%** | **18932.156** |
| final epilogue | 168112 | 3.910% | 5253.500 |

**直接观察：** steady K 加 drain K 占 76.78%，因此核心供数与 compute 调度仍然值得优先分析。N-task handoff 也显著；但 14.09% 不能全算作 output TDM drain，其中还包含 SiLU、BF16/FP4 quant、LDS staging、descriptor/setup、input readiness 和 successor prologue。

**判断边界：** phase 是按 decoder WMMA 边界划出的 elapsed 区间。旧 `gemm1_phase_analysis.json` 的 instruction duration 归属与 elapsed span 是不同口径；本文时间占比使用 v2 的 `phase_elapsed`，不将两者混用。

## 6. stall 归属与关键 ISA

下表汇总 decoder 的 instruction `stall` 归属，分母仍为 4,299,004 个采样 wave cycles。steady 栏也使用同一个总分母，以便看出热点集中在哪一阶段。

| instruction 类别 | 全 wave stall 归属 | steady K 部分 | 解释 |
|---|---:|---:|---|
| **`s_barrier_wait`** | **18.110%** | **12.537%** | WG rendezvous 是最大的归属热点 |
| WMMA | 15.774% | 13.405% | matrix 指令处有明显 issue/依赖等待；不能直接判为 matrix pipeline 饱和 |
| `s_wait_tensorcnt` | 6.503% | 4.823% | 等待本 wave 的 TDM completion 距离 |
| LDS load | 5.224% | 4.647% | LDS issue/依赖路径有成本，原因需进一步区分 |
| `s_wait_dscnt` | 3.718% | 2.850% | LDS operands/reader 完成等待 |
| `s_wait_storecnt_dscnt` | 2.054% | 0 | 主要位于 output 发布前 |
| TRANS | 0.352% | 0 | 仅是归到 TRANS 的 stall；不是 SiLU 全部执行成本 |
| `s_wait_storecnt` | 0.006% | 0 | 本次 direct ScaleA store wait 归属很小 |

这些数值不能直接相加再乘 94.888 μs，声称删除 wait 就能省下相应时间。并行 waves 可相互覆盖，barrier 可能等待另外一个仍在执行必要工作或等待供数的 wave；decoder 指令归属也不是全 GPU 的硬件 busy counters。

### 6.1 热点 PC

`code.json` 的 PC index 与 ELF object offset 均列出，便于复查。`code.json` 的 source-string 为空，源码关联依赖控制流/ISA 对照，不冒称具有逐行 DWARF 映射。

| PC index | ELF object offset | ISA | stall / 采样 lifetime | 控制流位置 |
|---:|---|---|---:|---|
| 1492 | `0x3954` | `s_barrier_wait 0xffff` | 4.878% | steady carry/reader rendezvous |
| 259 | `0x1dc8` | `s_barrier_wait 0xffff` | 3.925% | output publication |
| 1664 | `0x3eb4` | `s_barrier_wait 0xffff` | 2.871% | steady carry/reader rendezvous |
| 1845 | `0x4444` | `s_barrier_wait 0xffff` | 2.577% | steady carry/reader rendezvous |
| 256 | `0x1dbc` | `s_wait_storecnt_dscnt 0x0` | 2.054% | FP4 payload/ScaleA publication 前 |
| 931 | `0x2db8` | `s_wait_tensorcnt 0x0` | 1.980% | successor compute/fence 路径 |
| 1661 | `0x3ea8` | `s_wait_tensorcnt 0x2` | 1.850% | steady TDM readiness/复用路径 |
| 2028 | `0x49d8` | `s_barrier_wait 0xffff` | 1.654% | steady carry/reader rendezvous |

steady 热点的实际 ISA 片段：

```text
0x3948  s_wait_tensorcnt 0x2
0x394c  s_wait_dscnt 0x0
0x3950  s_barrier_signal -1
0x3954  s_barrier_wait 0xffff
0x3958  tensor_load_to_lds ...
0x3964  s_set_vgpr_msb 0x555
0x3968  v_wmma_scale_f32_32x16x128_f4 ...
```

output 发布处：

```text
0x1dbc  s_wait_storecnt_dscnt 0x0
0x1dc4  s_barrier_signal -1
0x1dc8  s_barrier_wait 0xffff
0x1dcc  s_wait_tensorcnt 0xa
0x1dd0  tensor_store_from_lds ...
0x1ddc  s_wait_storecnt 0x0
```

**源码/ISA 解释：** refill 前要满足 input completion 与旧 LDS readers 已读完；output TDM 消费 LDS 前要保证各 wave 的 payload staging 已完成。`s_barrier_signal -1`/`wait 0xffff` 在这里是 WG barrier。它们不能按“耗时高”直接删除，更不能把跨 wave 安全性寄托于某一个 wave 的 `TENSORcnt`。

硬件依据：Shader Programming Guide §4.10.1，printed p.197，规定同一 wave 的 Tensor loads/stores completion 有序，跨 wave 无序，Tensor 与其他 memory 指令也不保证相互顺序。ISA §5.6.2–5.6.3，printed p.45–48，说明 WG barrier 等待成员 waves signal。

### 6.2 同 SIMD 的两个 peer 显示到达错位

只比较同一 capture、SE、physical WGP 和 hwSIMD 上的 slot0/slot1，得到 16 组 peers。它们 lifetime 大部分重叠，每组匹配 **93 次 steady rendezvous**。

| 指标 | 观察结果 |
|---|---|
| `slot1.signal − slot0.signal` 的逐组中值 | **113–163 cycles** |
| `slot1.release − slot0.release` 的逐组中值 | 全部为 **1 cycle** |
| 哪个 slot 承担更多 barrier stall | 16 组中 15 组为 slot0，1 组为 slot1 |

**推导：** peers 到达 barrier 的时间有系统性错位，而 release 很接近，早到 wave 经常承担更多等待。正式 M256 layout 的 `wave_m = wave % 2`、`wave_n = wave // 2` 改变 operand/owner 分布，同 SIMD peers 的工作和供数角色值得对照。

**限制：** 两个 peer 不是整个 WG 的八个 waves。不能据它们确认八个 waves 的最终到达者，也不能从独立的 SIMD0–3 captures 拼出一次真实八 wave 同步时间线。改变分工后仍须证明每一代 LDS reader/owner 的安全性。

### 6.3 WMMA 边界存在较长间隔

统计单 wave 相邻 WMMA decoder 时间戳的差：

| 相邻 WMMA 的位置 | 样本数 | mean cycles | median cycles |
|---|---:|---:|---:|
| 同一 K128 内 | 107520 | 17.205 | 8.0 |
| K128 边界，排除 K256/task 边界 | 3584 | 235.781 | 112.5 |
| K256 边界，排除 task 边界 | 3456 | 174.729 | 103.0 |
| N-task 边界 | 96 | 6326.063 | 4708.0 |

**观察与推导：** K128/K256 边界明显比组内间隔长，与 LDS load、carry、TDM readiness 和 WG fence 热点一致；单看增加静态 WMMA 连续长度不足以解决边界等待。

Shader Programming Guide §5.7.5 printed p.233 的 `V_WMMA_F32_32x16x128_F4` 行记载 MI450/XCD-ML independent/dependent repeat rate 均为 8 cycles。但 decoder 记录也出现单 wave gap 小于 8 cycles、两个 peers 的合并 gap 为 0 的情况，因此这里仅报告 decoder 时间戳间隔。**不使用“WMMA 次数 × 8 / lifetime”推定实际 matrix utilization，也不由中值 8 认定全 GPU matrix pipeline 已满载。**

## 7. N-task 交接：已有 overlap，主要看 successor 路径

正式版已在上一 task 最后一个 K256 compute 前发出下一 task 的 I0/I1 TDM，使其与最后 K tile 和 epilogue 重叠。output 使用已释放的 LDS 区域；successor 先等待 I0/I1 可用并 prime/compute tile0，再完成旧 output 与 reader fence，随后 refill 后续 ring slots。

源码锚点是 `prefetch_next`、`run_task` 的 `PRE/n_steady`、successor prologue、最后一个 drain tile 中的 `dispatch_wave_job(prefetch_next, next_jobs)`，以及 quant epilogue 的 `issue_quant_output`。

本目标的 **`OUTPUT_IN_RELEASED_STAGE=True`**。`PERSISTENT_TASKS>1 and not OUTPUT_IN_RELEASED_STAGE` 才执行的 final reader wait 在本目标没有生成。因此不存在一个可直接挪到前面、立即消掉的“epilogue 前 final reader wait”。此前这一假设已在源码/ELF 审核中排除。

task handoff 的 14.09% 值得优化，但下次应拆开量化与 payload staging、output publication、I0/I1 readiness、successor kt0 和后续 refill 各自依赖。v89 shared/disjoint 只是共用 steady kt0 和调整 output placement：shared 新配对未确认收益，disjoint 筛选也未优于正式版。没有证据支持靠压缩静态 body 或单纯分离 output arena 就能稳定提速。

### 7.1 是否试过去掉 persistent，每个 WG 只执行一个 N-task？

**试过，且候选基于本次抓 trace 的同一正式最优源码。** 2026-10-17 已测试 `cyclic64_mg8_ntasks1_v2`，将 `M256_FUSED_LAYOUT` 内的 `PERSISTENT_TASKS` 从 4 改成 1。与正式 review 的源码 diff 只有该编译期选择；由原有 `PERSISTENT_TASKS == 1` 分支执行一次 `run_task(0)`，使用 generic cluster swizzle。保留 `t256/w2x4/b4`、完整 K7168 和 N4096、全部 16 个 N256 tiles、原 quant 和 routing 语义。候选仍沿用 `_persist` symbol 后缀，判断是否 persistent 要看 task count/代码，不能只看 kernel 名。

| 方案 | 每 WG 的 N-task 数 | 已有 GEMM1 数据 | 验证与结论 |
|---|---:|---:|---|
| 正式 `cyclic64_mg8_delivery_review` | 4 | 正式三轮中值 94.888 μs | 本次 ATT 对应版本 |
| `cyclic64_mg8_ntasks1_v2` | **1** | **单次筛选 98.032 μs** | 完整 11 cases equivalence 和 dynamic/poison/eager/graph replay 通过；未做三轮配对和完整 ELF resource/fallback 审核，没有该版 ATT |
| `cyclic64_mg8_ntasks2_v2` | 2 | 单次 93.944 μs；配对三轮 94.699/95.915/95.496 μs，中值95.496 μs | 对应 review 三轮95.789/95.555/95.561 μs，中值95.561 μs；方向不一致，target 另有4个SGPR spills，未采用 |

1-task 原 benchmark `count=19`，sanity `logits_diff=3.4733e-06`，executed GEMM1 FLOPs 仍为962072674304。前置全机 idle，但立即后置为 `busy=3`、无 KFD 进程；没有该样本的六秒后置复核，也没有同批4-task/1-task配对。因此 **98.032 μs 是已试过的单次原始筛选值，不足以声明稳定回退多少，也不足以否定所有 non-persistent 方案**。2-task 旧配对的后置也出现 busy 与其他任务/CPU 编译记录，不能把它提升为新的严格空闲收益证据。

对本 workload，4-task 改成1-task仍在一次 kernel dispatch 内完成，但 launch 从 **448 WG 增至1792 WG**，其中实际 compute WG 从 **256增至1024**；每个 WG 的四级 input ring 仍为272 KiB LDS，不会因为只做一次 task 就自动缩小。各 N-task 的必要 SiLU、BF16 rounding、FP4/E8M0 quant、payload staging和发布仍要执行，只是转到更多 WG 的最终 epilogue。

原14.09%的 handoff 同时包含当前 task 的 epilogue与下一 task的prologue，不是全部可删除的串行同步。1-task不再有WG内部successor，代价是每个独立WG重新setup并prime pipeline；原源码还启用1-task的完整 `pipeline_fence(outstanding=0)`。4-task已有的I0/I1跨task prefetch overlap也不再发生。更多WG可能改善调度，也可能暴露更多setup/prime/drain成本，净结果需要实测，不能用 `94.888 × (1−14.09%)` 推算1-task性能。

源码/launch能确认上述拓扑和依赖变化；没有1-task ATT或memory counters，不能量化是哪项造成98.032 μs。下一轮若重试，应把当前最优4-task与1-task做严格空闲的多轮交替，审核ELF/descriptor/resources，再用1-task trace对照相同完整计算。当前仅补查已有记录，未恢复GPU优化。

证据：[1-task候选源码](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_mg8_ntasks1_v2.py)、[1-task原benchmark](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks_v2_20261017_cyclic64_mg8_ntasks1_v2_bench.log)、[完整equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks_v2_20261017_cyclic64_mg8_ntasks1_v2_equivalence.log)、[dynamic replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks_v2_20261017_cyclic64_mg8_ntasks1_v2_replay.log)、[2-task三轮原始controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks2_v2_pair_20261017_controller.log)。完整实验记录见优化文档“完整 N workload 拆成 1/2 个 N-task”。硬件调度与LDS依据为Shader Programming Guide §2.2–2.3、§5.7.3；这里没有用peak wave创建率估算实际调度μs。

## 8. 资源与 occupancy：没有 spills，LDS 限制驻留

| ELF 资源项 | 实际值 |
|---|---:|
| VGPR count | 470 |
| allocated VGPR（dispatch event） | 480 |
| SGPR count | 106 |
| group segment / LDS | **278528 B = 272 KiB** |
| private segment | 0 B |
| VGPR spills / SGPR spills | **0 / 0** |

rocprof CSV 的 `VGPR_Count=240` 与 ELF metadata 的 470 口径不同，本文用 ELF metadata 和 dispatch allocation 做资源判断，不把 240 当作 kernel 实际 VGPR count。

每 WGP 最大 LDS 为 320 KiB，而 `2 × 272 KiB > 320 KiB`，所以两个同款 WG 无法同时驻留同一 WGP。occupancy 也观察到每 physical SIMD 最多两个同时活跃 slots，与 256-thread/8-wave WG 分布在四个 SIMD 的情况一致。

**推导：** 一个同款 WG 内多个 waves 同时进入共享 fence 时，不能再靠另一同款驻留 WG 覆盖等待；供数和 wave 到达错位更容易暴露。VGPR pressure 也较高，但本次没有 spills，不把问题称为 scratch 带宽瓶颈。

**限制：** 缩小 LDS 不自动等于提速。本次只有约 256 个有效 compute WG，还要考虑 VGPR allocation、cluster 调度、prefetch 距离和整体可用并行度。此前 K128/b4 共用 prologue 虽降低到 156672 B，却筛选为 108.465 μs，已经说明仅看 LDS/VGPR 数不足以选择方案。

### occupancy 字段与负载均衡解释

本次 decoder 3.1.0 的 `occupancy_fields` 实际有 11 个字段：

```text
time, cu, simd, wave_id, start, kernel_id,
me_id, pipe_id, is_ext, workgroup_id, cluster_id
```

`workgroup_id` 字段存在，但本次全部 **0**，`is_ext` 也全部 0；end rows 的 `cluster_id` 全部 0，不能用它将 start/end 无条件归到某个逻辑 cluster。旧通用分析日志中“没有 logical WG ID”的说法应理解为本次没有可区分的有效 WG ID，而非 schema 缺少字段。v2 报告已保存这些字段分布。

每项 capture 只观察 mask 内四个 SE、每 SE 16 个 physical WGP，共 **64 WGP**，不是全 GPU 的 256 WGP。physical-WGP completion imbalance 按 `final-end span / median(WGP envelope)` 定义，16 项 capture×SE 的中值 **2.612%**、最大 **9.641%**。这描述采样 physical WGP 的结束时间散布；一个 WGP 可以先后执行多个 capacity/compute WG，不能把它称为逻辑 expert/WG 负载均衡百分比。

硬件依据：Shader Programming Guide §2.2–2.3、§3.3.4；MI455X 白皮书 printed p.7 的 8 XCD×2 SE×16 WGP 架构，以及 printed p.10 的 320 KiB LDS/WGP。实际 agent 信息与这些资源数量相符。

## 9. 时钟、memory counters 与尚未证实的结论

`analyze_att_capture.py` 使用 REALTIME delta 估计 shader frequency。本次 metadata 的 reference frequency 为 0，脚本采用默认 **100 MHz**，在该假设下得到：

| capture | REALTIME-tick 加权平均（MHz） |
|---|---:|
| SIMD0-select | 1479.614 |
| SIMD1-select | 1498.027 |
| SIMD2-select | 1454.349 |
| SIMD3-select | 1470.928 |
| 16 项 capture×SE 合并加权 | **1475.881** |

16 项 series mean 的范围为 1417.784–1528.923 MHz。Shader Programming Guide §3.4.13 说明 REALTIME 是 fixed-frequency counter，但泛化描述分别出现“typically 100MHz”和 discrete GPU typically 27MHz，不能据该通用段落确认本机 reference frequency。上表必须带 **100 MHz 假设**使用；未校准前不声称测得固定降频、不套用白皮书 2.4 GHz 推算剩余收益，也不能将独立 captures 的差异归因于 SIMD 本身。

本次没有采集 HBM read/write bytes、cache miss、LDS bank-conflict 或全 GPU matrix-busy hardware counters。benchmark 的 effective R+W 是唯一 logical useful traffic，排除了 tiling 重复、LDS 和 metadata，不是 measured HBM traffic。因此不能用日志中的约 11 TB/s effective 指标判定 HBM 饱和。

同样，LDS load/`dscnt` stall 说明供数路径值得分析；Shader Programming Guide §5.7.6 printed p.235–236 说明 SIMD-pair 共享 LDS issue，`DS_LOAD_B128` independent repeat rate 为 2 cycles、dependent 约 58 cycles，bank conflict 会降低性能。但本次 counters 无法区分带宽、bank conflict、read-port 或其他依赖各自的份额。§5.3.2 printed p.218–219 的 VGPR bank/source-cache 机制也只能支持后续检查方向，不能代替测量。

## 10. 下一轮可验证的优化方向

此处是暂停后可继续的实验方向，本次没有实施，也没有承诺收益。

| 优先级 | 方向 | 本次支持证据 | 下一步需要证明 |
|---:|---|---|---|
| 1 | 调整 steady K 的 reader release、READY 与 TDM issue 距离 | steady K 占 67.17%，barrier/tensorcnt/LDS 热点集中在 K 边界 | 逐 owner/slot/generation 的依赖图；移动独立工作后 wait 是否缩短；完整精度/replay/fallback 不变 |
| 2 | 优化 LDS→VGPR→WMMA 的分组与 operand lifetime | K128/K256 边界间隔大，高 VGPR allocation，ISA 反复切换 VGPR bank | 保留每 accumulator 的 56 次 K128 顺序；新 ISA 与 trace 证明真实 overlap；zero spills |
| 3 | 减少 N-task successor 暴露的交接等待 | 三次 handoff 占 14.09%，publication PC 热，已有 I0/I1 overlap | 分开量化/staging/readiness/refill 的依赖；不覆盖仍被 TDM 或 readers 使用的 output/input LDS |
| 4 | 平衡 wave owner/operand 角色 | 同 SIMD peers signal 中值差 113–163 cycles，release 接近 | 获取同一次 WG 的完整成员证据；保持跨 wave completion 的正确性，不能只移走早到 wave 的等待 |
| 5 | 结合调度与 grid 数量评估更小 LDS footprint | 272 KiB 限制同款 WG 驻留，已有小 LDS 候选反而更慢 | 同时检查 compute WG 数、VGPR、cluster placement 与 prefetch 距离；以空闲配对结果决定 |

所有方向都需保留 GEMM1 的当前功能、通用性、random/non-balanced 路由、完整 padded 计算、BF16/native FP4/per32 E8M0 数值语义。性能结论继续使用原完整 benchmark、空闲 GPU、交替多轮样本；ATT 耗时用于诊断。

## 11. 可复核证据

- [采集 controller 与源码审核日志](../.codex_tmp/gemm1_t1536_20261017/logs/gemm1_best_review_iter28_att_20261017_controller.log)。
- [正式 bottleneck v2 报告](../.codex_tmp/gemm1_t1536_20261017/captures/gemm1_best_review_iter28_att_20261017/gemm1_bottleneck_analysis_v2.json)：elapsed spans、stall 归属、peer 对照、PC、occupancy schema 与 preflight。
- [四份完整 ELF 审核](../.codex_tmp/gemm1_t1536_20261017/captures/gemm1_best_review_iter28_att_20261017/gemm1_review_elf_audit.json)。
- [原始 stall 汇总](../.codex_tmp/gemm1_t1536_20261017/captures/gemm1_best_review_iter28_att_20261017/gemm1_stall_analysis.json) 与 [原始 phase 归属](../.codex_tmp/gemm1_t1536_20261017/captures/gemm1_best_review_iter28_att_20261017/gemm1_phase_analysis.json)。
- [时钟与 physical occupancy 原始分析日志](../.codex_tmp/gemm1_t1536_20261017/captures/gemm1_best_review_iter28_att_20261017/logs/analyze_att_capture.log)。
- [review ELF 资源审核](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_delivery_review_inspection.log)。
- [只读 bottleneck 分析脚本](../.codex_tmp/gemm1_t1536_20261017/analyze_review_bottlenecks.py)。v2 去掉了由 WMMA repeat rate 推算 utilization 的初始估算，补充了实际 occupancy extension 字段审核；旧报告单独保留，不作为该项结论依据。
- [容器 CPU 分析复现报告](../.codex_tmp/gemm1_t1536_20261017/captures/gemm1_best_review_iter28_att_20261017/gemm1_bottleneck_analysis_v2_remote_verify.json)。在容器内重新解析原始 capture，仅规范化 Windows/Linux 路径后，与本地 v2 JSON 全部内容一致。
- [最终环境与残留检查](../.codex_tmp/gemm1_t1536_20261017/logs/gemm1_best_review_trace_final_environment.log)：2026-10-17 16:06:28 UTC，源码、用户验证脚本、文档、分析脚本/JSON 和归档 SHA256 核对通过；没有 GPU/KFD 进程、GPU busy=0，没有本任务 test/compiler/profiler 残留。
- [硬件 Shader Programming Guide](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming%2365.txt)：§2.2–2.3、§3.3.4、§3.4.13、§4.10.1、§5.3.2、§5.7.5–5.7.6。
- [CDNA5 ISA](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt)：§5.6 WG barrier。
- [MI455X 白皮书](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-cdna5-whitepaper.txt)：架构与 LDS 资源。

以上证据链接指向本地已下载的文件。远端原始 capture 与分析 JSON 位于第 3 节的 `my_code/thread_trace_runs/gemm1_best_review_iter28_att_20261017/`；本地完整归档包含该目录的原始采集输出。最终环境日志的文档 hash 对应加入本条检查记录前的版本，最终文档另行同步并核对 hash。

原始 captures、106 MiB 归档、候选源码与所有性能样本均保留。正式版本仍为 94.888 μs 的 review，优化任务按用户要求暂停。
