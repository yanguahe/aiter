# E64/T1536 non-balanced OPTX v153：thread trace 与瓶颈分析

2026-10-19最新说明：本文件ATT与分析保持绑定**历史v153 ELF**。当前生产版已更新为v190 runtime-E，E64/E65共享同一GEMM1实际编译对象；最新E64/E65性能中值为91.688/137.596 μs。v190的四SIMD ATT已重新抓取、完成实际ELF绑定并写入 [v190 thread trace分析](moe_gemm1_e64_t1536_optx_v190_thread_trace_analysis.md)，不使用本文件的动态百分比代替新对象证据。新验收详见 [OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。以下内容记录v153当时的实现、fallback和验收结果。

分析日期：2026-10-18；集成与指定 baseline 复测更新于 2026-10-19。v153 已集成正常入口并通过完整数值门禁。按用户指定 baseline commit **`339589f900b3a880ec447f605886fe803cbf9649`**，独立源码树、独立进程的三轮原完整 benchmark 中值为 **98.553 / 91.915 μs**，改善 **6.638 μs（6.74%）**，距 80 μs 差 **11.915 μs**。此前 trace 验收批次 94.625 / 91.017 μs 的冻结对照来自 commit `3f1213c795a47e68bdde2b661349c77e4afa0bd3`；两批独立保留，不混算。v153 最新三轮实际完整 ELF 与本次 ATT 和历史验收版一致。

E64/T1536/topk8 与 E65/shared1/T1536/topk9、K7168/inter2048、random non-balanced 的 seed0/17/43 完整 MoE e2e 补验通过。有效 payload、ScaleA、GEMM2 和最终 MoE 输出与指定 commit 的 SHA256 一致，原 FP32 reference 精度正常。E65 保持 generic parent fallback；自动 OPTX 只覆盖经过验证的 E64 bucket。复现、每轮数据和负载限制见 [OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

四 SIMD ATT 与上述性能实际完整 ELF 逐字节一致。32 条完整 sampled waves 中，compute 区间占 wave lifetime 的 **80.0184%**，三次 N-task handoff 合计 **9.8785%**，WG barrier 归因 stall 为 **7.8580%**。compute 区间含供数、同步和控制流，不代表 XDL utilization；这些并行 wave 统计不能换算成可直接省掉的 kernel μs。

## 1. workload、调度与实际对象

环境为 `yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`、`dsr1_ep4`、共享仓库 `/data/yanguahe/code/wk_sp1/aiter`。host 分支为 `moe_a4w4_pr_refactor_wt2`，HEAD 为 `339589f900b3a880ec447f605886fe803cbf9649`，但该主仓库中冻结 parent 的 persistent 工作区文件来自 `3f1213...`，v153 也由工作区源码加载；实际身份以下表 source/ELF 为准。指定 commit 的最新 baseline 使用独立 detached worktree，不覆盖主仓库。workload 保持 E64/T1536/topk8、random non-balanced、K7168/inter2048、SiLU、无 bias，原 `--iters 20` benchmark 未修改。

[v153 source](../.codex_tmp/gemm1_t1536_20261017/candidates/optx_core_v153.py) 为 G1/S4、固定窗口顺序、双份 physical A128/ScaleA128、完全无效 M16 WMMA skip、B/ScaleB register cache、F/S operand carry 和每 wave 独立 output publication。F1 期间读齐 S0/S1 到三份 A/ScaleA register banks，完成原 `tensor_wait(n)` 和全部 `DScnt` 退休后 signal，将 S0 的最多 8 条 WMMA 放在 signal/wait 间。完整 WG wait 后才 refill 当前 LDS stage、读取下一 F0。没有引入新的 WMMA reuse hints。

精确 kernel symbol：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g1_s4_sk1_st0_sn0_bcache_merge_wavewindow_wavepub153_earlybanks_splitS0_S1rows0_carryF_carryS
```

| 对象或资源 | 实测值 |
|---|---|
| 正式 persistent source SHA256 | `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790` |
| v153 source SHA256 | `a2edbe99eb6e2c550199952bff966374dbffe4f2ac66f71f654c816deb4f1a26` |
| actual `.text` SHA256 / bytes | `8fd295fbcc7ac94f8df983671badb01403b2f1f3c4270c6ec8b9081690d56c85` / 29056 |
| actual `.rodata` SHA256 / bytes | `2278abb0150338dc184b487eabe9870c560a436ac8458221cd70e8fcb45d62ae` / 64 |
| actual `.note` SHA256 / bytes | `ca0e10c80f6081d9147085b01135483446da21103bee9aa9805e6ceb4d7c22fc` / 1604 |
| actual VGPR / SGPR | 463 / 107 |
| LDS | 295936 B，即 289 KiB |
| private / VGPR spill / SGPR spill | 0 / 0 / 0 |
| static instructions / static WMMA | 4768 / 256，含互斥路径，不能当作动态执行计数 |

`compare_gpu_objects.py` 比较性能保存的实际 object 与 SIMD0–3 的 `out_gfx1250_code_object_id_15.out`，五份完整 ELF 全部相同。报告为 [ELF 绑定](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v153_splitS0_att_elf_binding_20261018.log)。CPU dummy signature 曾给出 VGPR461 和另一 `.text`，不能代替实际 GPU 资源和对象。

11 项 routing/token 用例、十项 `M_e=0/127/128/129/143/144/145/255/256/257` 边界、E65 和 E65/shared1 fallback、dynamic/poison/eager/baseline-graph/OPTX-graph replay 均通过。有效 canonical FP4 payload、ScaleA、GEMM2、MoE 四项 bitwise mismatch=0；atomic sorting 的行排列可以不同，比较各自 row map 解码后的每条有效 route，并要求相同 row set。每个有效 accumulator 的 56 次 K128 贡献顺序保持正向。

## 2. 采集、负载记录和复现

使用 [get_att_b_rocm_sdk.sh](get_att_b_rocm_sdk.sh) 的 `--all-simd --ana-att`，四个 SIMD 的 ATT 和 analysis status 均为0。每项 profiler workload 和分析分别包 `timeout --signal=TERM --kill-after=5s 120s`。采集期间没有本任务并发 CPU 编译、正确性验证或 idle polling。

远端 capture：

```text
my_code/thread_trace_runs/optx_v153_splitS0_att_20261018_r1/
```

12:01:36 UTC 前置全机 idle，12:06:13 立即后置 idle。12:06:19 的 post6 发现他人 `hyg_fyd1` T6144 PID1381041，报告已运行6秒，接近采集退出边界；12:06:26 恢复 idle。本次 ATT 为诊断采样，不用它给出的 duration 替代正式性能数据，也不把这个后置边界信息改写成首 sample 全部 idle。没有本任务残留，没有杀他人进程。

只下载分析 JSON、ISA、日志和 SHA manifest，未下载 `.out`。十项下载证据逐项核 SHA256，见 [manifest](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v153_splitS0_att_evidence_20261018.sha256)。四个 SIMD-select 是独立运行，不能把四份时钟全局对齐成一次完整设备执行。

host 复现 capture，使用新的 label：

```bash
kernel=a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g1_s4_sk1_st0_sn0_bcache_merge_wavewindow_wavepub153_earlybanks_splitS0_S1rows0_carryF_carryS
bash .codex_tmp/gemm1_t1536_20261017/run_imported_optx_trace_host.sh \
  optx_core_v153 <fresh_label> "$kernel" \
  TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0 \
  AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SNAKE=0 \
  AITER_GFX1250_GEMM1_OPTX_SKIP=1 AITER_GFX1250_GEMM1_OPTX_STAGGER=0
```

container 中复现分析：

```bash
capture=my_code/thread_trace_runs/optx_v153_splitS0_att_20261018_r1
timeout --signal=TERM --kill-after=5s 120s \
python3 .codex_tmp/gemm1_t1536_20261017/analyze_optx_skip_trace.py \
  "$capture" --output <fresh_phase_report.json>
timeout --signal=TERM --kill-after=5s 120s \
python3 .codex_tmp/gemm1_t1536_20261017/analyze_optx_precise_pcs.py \
  "$capture" --output <fresh_pc_report.json>
```

两个 BLAS 变量用于绕过独立 reference FP32 matmul 的 HIPBLASLt fault；基准和候选一致，没有改变输入、计时、工作量或阈值。

## 3. phase 和有效 WMMA

32 条完整 sampled waves，`incomplete=[]`，累计 wave lifetime 为 3858272 cycles；每条 wave 完成四个 N-task、8 次 output TDM。这里的 cycles 是并行 lifetime 总和。

| phase | cycles 总和 | lifetime 占比 |
|---|---:|---:|
| prologue | 277135 | 7.1829% |
| compute 区间 | 3087329 | 80.0184% |
| 三次 N-task handoff | 381138 | 9.8785% |
| final epilogue | 112670 | 2.9202% |

96 次 handoff 的 mean/median/min/max 为 **3970.1875 / 3962 / 3334 / 4983 cycles**。它由上一 task 的最后 WMMA 到下一 task 的第一 WMMA 定义，包含 epilogue、输入准备和控制流；不能把它当作纯通信或纯 barrier 成本。

动态 WMMA 分布为：24 条 wave 各2688，6条各3136，2条各2240，总87808。相对相同 sampled waves 的全 padding 3584/wave，减少 **23.4375%**，来自完全没有有效行的 M16 skip；不是全机吞吐统计，也不是减少有效计算。partial M16 保留整组 WMMA。

## 4. stall、逐 PC 和等待区间

| instruction 类别 | attributed stall cycles | lifetime 占比 |
|---|---:|---:|
| WMMA | 342870 | 8.8866% |
| `s_barrier_wait` | 303184 | 7.8580% |
| `s_wait_dscnt` | 118937 | 3.0826% |
| `s_wait_tensorcnt` | 108353 | 2.8083% |
| DS loads | 84341 | 2.1860% |
| `s_wait_kmcnt` | 60614 | 1.5710% |
| `s_wait_xcnt` | 55045 | 1.4267% |
| `s_wait_storecnt` | 42795 | 1.1092% |

逐 PC 保留 capture、code object 和 vaddr，避免把相同 asm 的不同位置混为一个热点：

| capture | PC index / vaddr | instruction | calls | stall cycles | 总 lifetime 占比 |
|---|---|---|---:|---:|---:|
| SIMD3 | 921 / `0x2d0c` | `s_barrier_wait 0xffff` | 432 | 84980 | 2.2025% |
| SIMD3 | 2200 / `0x4c58` | `s_barrier_wait 0xffff` | 432 | 62747 | 1.6263% |
| SIMD1 | 1542 / `0x3c5c` | `s_barrier_wait 0xffff` | 432 | 33537 | 0.8692% |
| SIMD2 | 1542 / `0x3c5c` | `s_barrier_wait 0xffff` | 432 | 30849 | 0.7996% |
| SIMD0 | 921 / `0x2d0c` | `s_barrier_wait 0xffff` | 432 | 26237 | 0.6800% |
| SIMD3 | 497 / `0x241c` | prologue `s_barrier_wait 0xffff` | 4 | 14478 | 0.3752% |

3584 个相邻 WG wait 完成区间的 cycles mean/median 为970.2944/824，范围666–8051；区间内 WMMA mean/median 为24.4431/24，DS load mean/median 为54.0536/54。每个区间的 barrier stall mean/median 为72.1535/15，最大7278，说明均值中包含长尾。区间跨越两次 wait，包含 task 边界，不是 signal→wait 区间；不能据此称每次 split barrier 中间执行24条 WMMA。实际 ISA CFG audit 的8个 split body 每条路径最多8条，4个 prologue body 为0。

`s_wait_dscnt` 后首个非 delay 指令的归因统计中，紧邻 WMMA 为63200 cycles（1.6380%），紧邻下一 DS load 为20576（0.5333%），紧邻 signal 为28772（0.7457%）。这支持继续检查供数和 release 时机，但不能仅由相邻位置判定 compiler bug、LDS bank conflict 或 HBM 饱和。现有报告没有对应的 bandwidth 或 bank-conflict counter。

报告：[phase JSON](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v153_splitS0_att_phase_20261018.json)、[precise PC JSON](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v153_splitS0_att_precise_20261018.json)。v128 的旧 trace 保持绑定 v128，不同 capture 的百分比不直接相减成 μs。

## 5. 硬件依据和下一步可检验假设

本地 MI400 Shader Programming Guide §4.3.7.2.4（printed p.90）记载：LDS read 在 DScnt 退休后，返回值已可在 VGPR 中使用。CDNA5 ISA §5.6–5.6.1（printed pp.45–46）记载 signal 与 wait 分离，barrierComplete 保存已完成但尚未 wait 的状态。由此可以在当前 S operands 已完成 LDS read、尚未复用对应 VGPR 的前提下，在 signal/wait 之间执行 WMMA；这不允许省略对所有 readers 的完整 WG release。

Shader Programming Guide §5.3.6（printed p.223）以及 §5.7.7（p.236）记载 LDS/TEX return bus 由 SIMD pair 共享，64 DWORD 返回宽度。这解释了仅减少物理 LDS 容量不保证供数提速，但不是本 capture 的实测饱和证据。

后续 v156/v157 保持全部 S operands 已在 VGPR、原 `tensor_wait(n)`、`s_wait_dscnt(0)`、完整 WG signal/wait，分别在 wait 前计算0或4条 S0 WMMA；wait 后先 refill，再完成其余 S0。下一 F0 register load 必须等 S0 的 A0/B0 最后使用完成，S1 使用独立 A bank/B1。两者已通过源码 scope、代际协议、CPU 无 private/spill、random17 四项 bitwise 及 dynamic/poison/eager/graph replay，原完整命令单次为 **94.823 / 92.491 μs**，未优于 v153，不采纳，不扩为三轮验收。实际两者 VGPR463/SGPR107/LDS295936 B，private/spills0；证据见优化记录中的 early refill 小节。

继续检查两个 stage 合并 READY/RELEASE 与早读 S banks 的组合：v158/v159 重复奇偶 S0 body，CPU 产生 private196 B/VGPR spill608，淘汰；v160/v161 将公共 S0 WMMA 移到奇偶分支外，CPU 为 VGPR463/SGPR106、无 private/spill。新的 constrained CFG 审计分别找到12个 signal/WAIT 区间、四个奇偶关联谓词及八个 EXEC 恢复区域，并捕获五项错误谓词/EXEC/memory 负对照；奇数 stage 到达 WAIT，偶数 stage 同时跳过 WAIT/refill。旧无约束审计的失败日志保留。两版随后random17及dynamic/poison/eager/graph replay有效四项bitwise通过，原完整benchmark分别99.759/98.960 μs，未优于v153，不采纳。该控制流证明不代替READY/RELEASE、WMMA data hazards或GPU数值验证。

## 6. trace指导的后续实验与瓶颈限制

v162/v163在v153供数调度下让每WG只执行1/2个N-task，相应增加WG数以保留完整M/N；单次94.041/93.048 μs，actual VGPR443/462、SGPR68/106，无spill。即使减少handoff和register占用，仍没有超过v153，说明不能把9.8785% handoff简单扣成μs收益。

候选级LLVM调度对照v164/v165/v166分别92.432/92.736/93.008 μs；三/四register banks分布S预读v167/v168为96.740/95.810 μs。v167后置记录有身份不完整的KFD PID2082811，不能证明严格无重叠；原样本保留且未采纳。WMMA次序v169/v170/v171为92.240/92.312/93.803 μs；M32 skip v174为92.525 μs；混合physical halves的窗口v177为94.180 μs。上述均为原完整benchmark单次screen，数值/replay通过、actual LDS295936 B且无private/spill，没有新的三轮采纳结果。

复制全有效快速路径v172/v173出现VGPR spill1858/1290，CPU拒绝；打包行谓词修正版v176出现SGPR spill10，CPU拒绝。最后S1 WMMA与F epilogue交错v178/v179因owner分支重复展开，静态指令7560、SGPR spill10；共享epilogue修正版v180/v181降到4849/4856条，仍SGPR spill4，未运行GPU。v182/v183禁machine/postRA LICM后无spill，实际分别VGPR453/452、SGPR102、LDS295936 B，random/replay有效数值正确，原完整GEMM1单次91.464/93.113 μs，没有显示超过v153的收益，停止该交错方向。两项actual barrier CFG通过，但不将它当作完整WMMA数据hazard证明；没有新增ATT capture或把单次screen写成稳定配对收益。

硬件依赖继续按本地SPG §4.3.7.2.4（printed p90）执行DScnt退休，按§4.10.1（p197）保持同wave input/output TDM队列顺序；新epilogue交错还需按§4.6.12.1（p148–150）审核WMMA/VALU RAW、WAW、WAR coexecution距离。已有CFG只证明signal→WAIT路径无memory，不覆盖这些数据hazard。

目前证据仍指向compute中的供数、WG同步、WMMA及register/编译调度的组合限制；没有HBM饱和或LDS bank conflict counters，不能将某个单资源断言为唯一瓶颈。实验详情、复现命令和actual对象绑定见[优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。四SIMD ATT本身没有重抓或改写。

共享 A128/ScaleA128 槽方向按用户指令保持放弃。下一步结论依赖实际 GPU 结果，不能因请求更早或资源相同就宣称性能提升。
