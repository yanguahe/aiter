# E64/T1536 non-balanced OPTX v128：thread trace 与瓶颈分析

分析日期：2026-10-18。v128 是采集时 OPTX 路线的最低单次筛选版本，GEMM1 为 **98.525 μs**，仍慢于正式冻结路径最近三轮中值 **94.279 μs**，未经过三轮配对验收，未采纳。后续v153完成完整验证和三轮配对，中值91.017 μs，动态分析另见 [v153 thread trace分析](moe_gemm1_e64_t1536_optx_v153_thread_trace_analysis.md)。本文保留v128的历史证据，不把它作为后续版本的动态瓶颈证据。

四 SIMD 采样中，compute 占 wave lifetime 的 **82.4565%**，三次 N-task handoff 合计 **8.7341%**。循环中的 WG rendezvous、LDS 读取与 WMMA 调度仍是后续需要验证的重点。现有证据没有 HBM 饱和或 LDS bank-conflict counter，不能把等待直接解释为某一种硬件资源饱和，也不能将并行 wave 的百分比换算为可节省的 kernel μs。

## 1. workload、实现和对象绑定

环境为 `yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`、`dsr1_ep4`、共享仓库 `/data/yanguahe/code/wk_sp1/aiter`。远端分支 `moe_a4w4_pr_refactor_wt2`，HEAD 为 `339589f900b3a880ec447f605886fe803cbf9649`；工作区 source hash 单独核对。

候选 [optx_core_v128.py](../.codex_tmp/gemm1_t1536_20261017/candidates/optx_core_v128.py) 使用两份 physical A128/ScaleA128、G1、固定窗口顺序、完全无效 M16 WMMA skip、B/ScaleB register cache、F1→S0 与 S1→下一 F0 的完整 operand carry，以及每 wave 独立的 output publication。没有改变有效 accumulator 的 K0→K55 顺序、SiLU、BF16 rounding、E8M0/native FP4 conversion、canonical output layout、routing 或有效行精度。

精确 kernel 名：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g1_s4_sk1_st0_sn0_bcache_merge_wavewindow_wavepub128_carryF_carryS
```

| 对象 | SHA256 / 数值 |
|---|---|
| 正式 persistent source | `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790` |
| v128 source | `a4fe50a1cd51798576e5e5a432e3013d9abe3688fa87324f3fef9395cece11d3` |
| 原性能实际 ELF 与四份 ATT ELF 的 `.text` | `fbefb57e358ab8299c9b2d5088f638838d8a52fb1fab702af9a9601aa13e569a`，29952 B |
| 实际 `.rodata` | `0e5b442c760275611bba7fec915fa47adf5069024d6e9f9ef38643dcaf5caab8`，64 B |
| 实际 `.note` | `13513c73a5f586489f23d90011996e930257ad293e451fa2294e6237fe9a117a`，1548 B |
| VGPR / SGPR | 448 / 107 |
| LDS | 295936 B，即 289 KiB |
| private / VGPR spill / SGPR spill | 0 / 0 / 0 |

`compare_gpu_objects.py` 核对原性能实际 ELF 与 SIMD0–3 的 `out_gfx1250_code_object_id_15.out`，五份完整 ELF 逐字节一致；不仅是资源数量相同。报告为 [四 SIMD object 绑定](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v128_att_object_binding_20261018.log)。CPU dummy signature 的 `.text` 是另一值，不能拿它代替真实 GPU 对象。

v128 的 random seed17 验证及 dynamic/poison/eager/baseline-graph/OPTX-graph replay 已通过：所有有效 route 的 payload、ScaleA、GEMM2、MoE 四项 bitwise mismatch 为0。尚未完成该版全部11-case、M_e边界和fallback采纳验收。

## 2. 采集和复现

使用用户 [get_att_b_rocm_sdk.sh](get_att_b_rocm_sdk.sh)，`--all-simd --ana-att`，SIMD0–3 的 ATT 与分析 status 全为0。trace workload 使用两轮诊断运行，不作为原 `--iters 20` 性能成绩。所有 profiler workload 和分析进程分别限时120秒，性能/ATT期间没有本任务并发CPU编译、数值验证或空闲检查。

远端原始 capture：

```text
my_code/thread_trace_runs/optx_v128_fullcarry_att_20261018_r1/
```

本次只下载分析报告、ISA和日志，不下载 `.out`。四个 SIMD-select capture 是独立运行，不能全局对齐成一次完整设备执行。

从 host 复现 capture，label 使用新名字：

```bash
kernel=a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g1_s4_sk1_st0_sn0_bcache_merge_wavewindow_wavepub128_carryF_carryS
bash .codex_tmp/gemm1_t1536_20261017/run_imported_optx_trace_host.sh \
  optx_core_v128 <fresh_label> "$kernel" \
  TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0 \
  AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SNAKE=0 \
  AITER_GFX1250_GEMM1_OPTX_SKIP=1 AITER_GFX1250_GEMM1_OPTX_STAGGER=0
```

在容器中只复现分析：

```bash
capture=my_code/thread_trace_runs/optx_v128_fullcarry_att_20261018_r1
timeout --signal=TERM --kill-after=5s 120s \
python3 .codex_tmp/gemm1_t1536_20261017/analyze_optx_skip_trace.py \
  "$capture" --output <fresh_phase_report.json>
timeout --signal=TERM --kill-after=5s 120s \
python3 .codex_tmp/gemm1_t1536_20261017/analyze_optx_precise_pcs.py \
  "$capture" --output <fresh_pc_report.json>
```

09:25:19 UTC前置全机idle；采集结束09:29:34后置idle。09:29:40的post6发现刚退出的短进程/KFD条目，09:29:47恢复idle；没有本任务残留，不把首post6样本写成全部idle。

原性能复现仍用 `run_imported_optx_perf_host.sh optx_core_v128 <fresh_label>` 加相同两个BLAS变量和四个OPTX参数，执行用户原完整random/non-balanced命令、`--iters 20/count 19`。reference计算在计时后执行，其耗时不计入GEMM1。

## 3. elapsed phase 与实际 WMMA 数

32条完整compute waves，incomplete为0。每条wave四个task、每task两条output TDM；各自完成相同的正向K遍历。WMMA总数分布是2688×24、3136×6、2240×2，相对完整padded 3584条，本次采样少23.4375%。这来自完全无效M16跳过；不同capture采样的expert不同，不能把它与旧trace的20.3125%之差视为新增收益。

| phase | 合计 cycles | 采样 wave lifetime 占比 |
|---|---:|---:|
| prologue | 275969 | 6.2373% |
| compute | 3648258 | **82.4565%** |
| handoff | 386435 | **8.7341%** |
| final epilogue | 113801 | 2.5721% |
| 合计 | 4424463 | 100% |

96个handoff，mean4025.3646、median4029.5、min3335、max5241 cycles。handoff按上一task最后WMMA完成到下一task首WMMA开始划分，包含epilogue、output publication、successor setup/READY/prime，不是纯TDM store时间。

原最优trace的handoff为14.0923%，但capture clock和采样条件不同。不得以14.09%→8.73%直接宣称相应的μs收益；v128的原benchmark实测仍为98.525 μs。

## 4. stall归属与逐PC证据

| 指令类别 | calls | stall占wave lifetime |
|---|---:|---:|
| `s_barrier_wait` | 3616 | **13.1057%** |
| WMMA | 87808 | 8.4199% |
| `s_wait_dscnt` | 102732 | **5.9002%** |
| `s_wait_tensorcnt` | 7488 | 2.1353% |
| DS_load | 193728 | 1.7057% |
| `s_wait_storecnt` | 128 | 1.2851% |

这些是decoder归属到指令上的stall，可能包含来自更早指令或较慢peer的依赖，不能相加当作独立可删除的kernel墙钟时间。

旧phase分析的`top_pcs`按相同asm合并，不是真正的独立PC排名。新脚本按`capture + pc_index + code_object + vaddr`绑定；以下地址属于各自的`code.json`，四份capture不能混用PC index：

| capture | PC index | vaddr | 指令 | calls | stall cycles | 占全部采样lifetime |
|---|---:|---:|---|---:|---:|---:|
| SIMD1 / dispatch31 | 1582 | 15620 | `s_barrier_wait 0xffff` | 388 | 97421 | 2.2019% |
| SIMD3 / dispatch31 | 919 | 11532 | `s_barrier_wait 0xffff` | 388 | 79282 | 1.7919% |
| SIMD0 / dispatch31 | 919 | 11532 | `s_barrier_wait 0xffff` | 388 | 68932 | 1.5580% |
| SIMD2 / dispatch31 | 1582 | 15620 | `s_barrier_wait 0xffff` | 388 | 67982 | 1.5365% |

3584个相邻barrier区间，cycles中值1010、mean1128.1479；WMMA中值24条、DS_load中值54条；barrier指令自身stall中值147.5、mean146.6922 cycles。区间不是纯K-step，含setup/closing边界，不宜把mean当作每个tile固定成本。

DScnt后的首个非setup opcode用于描述发射位置，不证明wait在等待后面的load：

| 后续首个opcode | wait calls | wait归属stall占lifetime |
|---|---:|---:|
| `ds_load_b128` | 55680 | 2.7541% |
| `v_wmma_scale_f32_32x16x128_f4` | 32620 | 2.4384% |
| `ds_load_2addr_b32` | 3456 | 0.3594% |
| `ds_load_2addr_stride64_b32` | 3456 | 0.1562% |

`dscnt0x1a`与`dscnt0x11`各27648次，分别归属1.3865%和1.3474% lifetime。大量wait紧邻LDS loads，说明需要检查operand register复用、读完后refill与下一批读取的距离；没有证据把这些compiler生成的wait定性为LLVM bug，更不能直接删掉。

## 5. 对后续实验的约束与已取得的结果

硬件事实：MI400 SPG §4.3.7.2.4（printed p.90）以及CDNA5 ISA §5.7.1.4（p.55）说明LDS instructions按序退休，DScnt对load表示数据已返回VGPR、对store表示已写LDS。SPG §5.3.6–5.3.6.1（p.223–224）说明一对SIMD共享LDS/TA通路、return bus为64 DWORD/cycle；这些事实不保证某种读取排列一定加速。

本次trace支持先验证循环供数和rendezvous，而不是仅继续缩短handoff。根据证据已经做了以下对照，均保留正式kernel、原benchmark、有效K顺序及精度：

| 方向 | 实测或门禁结果 | 决定 |
|---|---|---|
| v135/v136，同K128的F/S交织并复用B/ScaleB | 99.038 / 98.842 μs | 未优于正式版，不采纳 |
| v137/v138，两组READY/RELEASE合并 | 98.995 / 99.000 μs | barrier次数减半未带来优势，不采纳 |
| v139，全部typed raw LDS reads | CPU `.text`与v128逐字节相同 | 无需重复GPU |
| v140，仅A/scales typed raw LDS reads | 静态ISA少44条、DScnt wait少6条，单次99.015 μs | 不采纳 |
| v141/v143，cached有效行谓词及组合invalid-read skip | 各6/10个SGPR spill | CPU拒绝，未GPU |
| v142，跳过完全无效M16对应的A LDS reads | 427/107、无spill，random/replay通过；单次109.947 μs | 不采纳 |
| v144/v146，M32分支跳无效A/ScaleA reads | 104.977 / 107.290 μs | 细分支跳读方向停止 |
| v147/v148，完整DS或WMMA4/DS/WMMA4 hints | CPU `.text`与v128相同 | 未重复GPU |
| v149，四组WMMA2交织DS | 99.120 μs、random/replay通过 | 不采纳 |
| v150/v151，F0期间提前S1或S0的独立banks | 100.935 / 101.726 μs | 未显示优势 |
| v152，F1期间读齐S0/S1的第三bank | 96.643 μs、463/107、无spill、random/replay通过 | 保留后续对照，未三轮配对、未采纳 |

v142继续保持ScaleA/B/ScaleB读取、TDM请求、READY/RELEASE与输出流程，仅不读取不会被有效WMMA消费的A fragment；M16只要含一行有效就保留整组读取。8224项row/source审计与错误window/跳partial-M16负对照通过，GPU有效四项bitwise独立验证。性能结果见 [OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md) 后续追加。

## 6. 报告和审核限制

- [phase/stall分析](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v128_fullcarry_att_analysis.json)，SHA256 `1a455a81e18a55867bbcff36e1a712206b2773282bcfb8e5f289e701932d59b4`。
- [逐PC与barrier区间分析](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v128_precise_pc_analysis.json)，SHA256 `1dca1658f78adfe1d56ac0e66b83f88a912e718f85ab71f68638a030f8847a40`。
- [实际ISA资源](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v128_att_actual_inspection_20261018.log)、[object绑定](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v128_att_object_binding_20261018.log)。
- source模型、CPU编译、ATT与GPU数值比较分别提供不同证据；任意一个不能单独代替完整功能验收或三轮配对性能。
- 正式最优版旧trace文档保留历史，见 [最优版GEMM1分析](moe_gemm1_e64_t1536_best_thread_trace_analysis.md)。不恢复已取消的GEMM2 trace归档。
