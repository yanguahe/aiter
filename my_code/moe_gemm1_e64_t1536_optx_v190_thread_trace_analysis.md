# E64/T1536 non-balanced OPTX v190：thread trace 与瓶颈分析

分析日期：2026-10-19。当前最优 v190 runtime-E 在 E64/T1536/topk8 与 E65/shared1/T1536/topk9 使用同一个真实 GEMM1 编译对象。用户指定 baseline commit 为 `339589f900b3a880ec447f605886fe803cbf9649`；最新三轮原完整 benchmark 的 E64 baseline/v190 中值为 **99.896 / 91.688 μs**，改善 **8.22%**，距 80 μs 差 **11.688 μs**。E65 中值为 **155.189 / 137.596 μs**，改善 **11.34%**。

本文件绑定新 v190 capture，历史 [v153 分析](moe_gemm1_e64_t1536_optx_v153_thread_trace_analysis.md) 保留原对象和数据。不能将旧 trace 的比例或实际对象改写为 v190 的证据。

## 1. workload、实现与实际对象

在 `yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`、`dsr1_ep4`、共享仓库 `/data/yanguahe/code/wk_sp1/aiter` 采集。保持 E64/T1536/topk8、K7168/inter2048、A4W4、SiLU、无 bias、random non-balanced。诊断采集使用原 fixture 和正常生产 selector，`--iters2` 只用于 ATT；正式性能始终来自用户原 `--iters20`、profiler count19 的完整 benchmark。

v190 基于已验证的 v153 三 A/ScaleA register banks、双份 physical A128/ScaleA128 LDS、G1/S4、四个 N-task、B/ScaleB register cache、F/S carry 和每 wave 独立 output publication。expert 数为 runtime `fx.Int32`，E64/E65 统一 TH6 默认编译参数，保留 override。每个有效 accumulator 的 56 次 K128 按正向顺序累加，全部有效 route 和量化输出保留。

精确 kernel symbol：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_edyn_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g1_s4_sk1_st0_sn0_bcache_merge_wavewindow_wavepub190_edyn_earlybanks_splitS0_S1rows0_carryF_carryS
```

| 对象或资源 | 实测值 |
|---|---|
| OPTX source SHA256 | `f8be1c42936214a3728fcf8273d0a2b6279ab6731c69fe225730f0bd3b1763f5` |
| selector SHA256 | `db26b0d02a7e4c374b0b29f204e6ef371c390aa4324eb9801f4da50fa6d38a68` |
| actual 完整 ELF SHA256 | `24bb623573932faadb22ca2d89d3e84a903adc64c9e116cac4138dd5b686f93c` |
| actual `.text` SHA256 | `7b0b0ca23994eba0fd308c388c5925b356051b553ca0e31e7e738a9c4a6805ad` |
| VGPR / SGPR | 463 / 107 |
| LDS | 295936 B，即 289 KiB |
| private / VGPR spill / SGPR spill | 0 / 0 / 0 |
| static instructions / WMMA | 4710 / 256，包含互斥路径，不是动态执行计数 |

五个 profiled 进程保存的实际对象与四份 SIMD `.out` 在远端读取并核对 SHA256，九份完整 ELF 全部相同；也与 E64/E65 正确性、replay 和正式性能保存的对象相同。没有下载 `.out`。

seed0/17/43 正常入口 E64→E65→E64 或逆序，cache 始终一个 artifact，Python identity 与完整 ELF 一致，无 parent GEMM1 fallback。八项 tensor hash 均与指定 baseline 一致：route IDs、route weights、valid row set、canonical payload、canonical M32 ScaleA、canonical GEMM2、MoE e2e、MoE reference。E64 logits_diff 约 `3.473–3.477e-06`，E65 约 `3.777–3.779e-06`，原 gate `0.01` 未改变，全部 finite。E65 包含每 token expert64/weight1 的 1536 条 shared route。动态输入、poison、eager、parent graph/runtime graph replay 也通过。

## 2. 采集与负载边界

使用 [get_att_b_rocm_sdk.sh](get_att_b_rocm_sdk.sh) 的 `--all-simd --ana-att`，SIMD0–3 ATT 和 analysis 均 status0。每个 profiler workload 和分析命令分别限制 `timeout --signal=TERM --kill-after=5s 120s`。采集期间本任务 CPU 会话闲置，没有并发编译、正确性验证或 idle polling。

远端 capture：

```text
my_code/thread_trace_runs/optx_v190_edyn_e64_att_20261019_r1/
```

host UTC 日志：18:27:03 首次 idle 检查发现他人 E65 验证 PID3601281，按六秒轮询等待；18:27:09 全机 idle 后才开始采集。18:30:02 立即 post 无 KFD、busy0、VRAM176582656 B。18:30:08 的 post6 发现他人 E65 benchmark PID3620179，报告运行5秒，启动在本采集退出后；18:30:14 恢复 idle。没有杀他人进程，没有自有残留，不把这些记录统一描述为首 sample 全部 idle。

四 SIMD-select 是四次独立运行，各有自己的时钟，不能拼成一次全设备 timeline。诊断采集耗时不代替正式 benchmark。

## 3. phase 与无效计算跳过

32 条完整 sampled compute waves，`incomplete=[]`，每条 wave 完成四个 N-task、八次 output TDM。累计 lifetime 为 **4164139 cycles**；这是并行 wave lifetime 总和。

| phase | cycles 合计 | lifetime 占比 |
|---|---:|---:|
| prologue | 292226 | 7.0177% |
| compute 区间 | 3379180 | 81.1495% |
| 三次 N-task handoff | 379963 | 9.1246% |
| final epilogue | 112770 | 2.7081% |

compute 区间同时包含 WMMA、LDS 供数、TDM readiness、同步与控制流，不能称为 XDL utilization。handoff 从上一 task 最后 WMMA 的 decoder end 到下一 task 第一 WMMA，包含 epilogue、发布、输入准备和控制流，不能称为纯 barrier 成本。

96 次 handoff mean/median/min/max 为 **3957.9479 / 3894.5 / 3361 / 5231 cycles**。26 条 wave 各2688次 WMMA、6条各3136次，总88704次；相对同一 sampled waves 的全 padding 3584/wave，减少 **22.65625%**。只跳完全无有效行的 M16，partial M16 保留，不减少有效 route 的 K 贡献。这也不是全机动态计算量统计。

## 4. stall、逐 PC 和 wait 区间

| instruction 类别 | attributed stall cycles | lifetime 占比 |
|---|---:|---:|
| `s_barrier_wait` | 491777 | 11.8098% |
| WMMA | 339684 | 8.1574% |
| `s_wait_tensorcnt` | 167129 | 4.0135% |
| `s_wait_dscnt` | 120563 | 2.8953% |
| DS loads | 98078 | 2.3553% |
| `s_wait_kmcnt` | 64586 | 1.5510% |
| `s_wait_xcnt` | 63664 | 1.5289% |
| `s_wait_storecnt` | 40742 | 0.9784% |

下面保留各 capture 的 PC index 和 vaddr，避免把相同 asm 的不同位置混合为单个热点。

| capture | PC index / vaddr | instruction | calls | stall cycles | 总 lifetime 占比 |
|---|---|---|---:|---:|---:|
| SIMD3 | 879 / `0x2c4c` | `s_barrier_wait 0xffff` | 432 | 93051 | 2.2346% |
| SIMD3 | 2158 / `0x4b98` | `s_barrier_wait 0xffff` | 432 | 61338 | 1.4730% |
| SIMD1 | 2798 / `0x5b4c` | `s_barrier_wait 0xffff` | 432 | 60659 | 1.4567% |
| SIMD2 | 2798 / `0x5b4c` | `s_barrier_wait 0xffff` | 432 | 55184 | 1.3252% |
| SIMD1 | 1500 / `0x3b9c` | `s_barrier_wait 0xffff` | 432 | 50023 | 1.2013% |
| SIMD2 | 1450 / `0x3a74` | `s_wait_tensorcnt 0x2` | 388 | 45584 | 1.0947% |

3584 个相邻 WG wait 完成区间 cycles mean/median 为 **1051.4375 / 833**，范围689–11337；barrier attributed stall mean/median为 **124.1222 / 14**，最大7615，存在长尾。区间内 WMMA mean24.6931、median24；这一区间跨两次 wait，包含循环/task边界，不是 signal→wait，因此不能称每次 split barrier 中间执行24条 WMMA。实际 ISA CFG audit 为四个 prologue 无 WMMA、八个 split 每路径最多8条。

`s_wait_dscnt` 后首个非 delay 指令的归因中，紧邻 WMMA占1.5497%、下一 DS load占0.4941%、signal占0.7026%。这些数据支持继续检查供数时机和 wave 到达时差，不证明是 LLVM bug、HBM 饱和或 LDS bank conflict；现有 capture 没有相应 counter。`kmcnt`/`xcnt` 的静态热点包含 runtime expert 查找和后续 scalar address/setup，不能把整个类别都归为可消除的 lookup 时间。

## 5. 硬件依据与下一步假设

本地 MI400 Shader Programming Guide：

- §4.6.12，printed p144：WMMA忽略EXEC并强制所有lane参与；padding skip需真实scalar branch。
- §4.6.12.1，p148–150：WMMA/VALU的RAW、WAW、WAR和coexecution距离仍必须满足，源码依赖或barrier CFG不能代替完整数据hazard检查。
- §4.3.7.2.4，p90：LDS读DScnt退休后，返回值可在VGPR中使用。
- §4.10.1，p197：同wave tensor load/store按序完成，跨wave无此保证，TDM也不自动与其他memory排序。
- §4.7.1、§5.3.6，p160、p223：LDS/WGP$共享容量、LDS segments和SIMD pair共享return通路是硬件事实；不能由资源量断言本kernel已饱和。

新隔离候选分别检验：v193将task级ScaleA global store等待延后至kernel退出，保留所有stores及最终STOREcnt/TENSORcnt完成；v194用wave连续读取runtime prefix并ballot/popcount计算expert upper bound；v196/v197按physical F半区分配wait前的已缓存S1计算，保留DS退休、完整WG wait和全部正向K贡献。前者针对0.9784% STOREcnt暴露，后者针对prologue及barrier到达时差；这些都是可检验假设，没有预先承诺对应μs收益。

v196 CPU出现一个SGPR spill，拒绝运行GPU；v197约束LLVM scalar live range后为VGPR451/SGPR102/LDS295936 B、无private/spill，12个signal/WAIT CFG区间通过，每split路径最多12条WMMA。v193/v194为463/107、同LDS、无private/spill，12个CFG区间通过。三者各一seed E64→E65→E64完整MoE八项hash与baseline相同、同artifact验收通过，dynamic/poison/eager/两种graph各五次replay通过。

后续原完整benchmark中v194 E64/E65为92.405/138.229 μs、v197为96.101/146.347 μs，均停止。v193独立进程三轮E64中值91.504 μs，v190同批91.949 μs，但反向一轮慢1.520 μs；E65三轮v193均快，中值136.528 μs，v190为137.719 μs。进一步同一HIP context E64原完整三轮，v190为90.931/90.549/91.492 μs，v193为90.973/91.582/91.064 μs，两轮更慢，稳定收益未确认，v193不采纳。两个批次不混算，也不以较低单次改写正式v190中值。

首K128 C=0方向分别试剥离、rolled branch和branch-local compiler barrier；无spill的v200/v201正确性通过，但E64/E65为114.111/179.430及105.658/163.488 μs，停止。v201 actual ISA确实含inline-zero WMMA，不能把负结果归因于C0没有生效。v203/v204独立LDS read排序虽将静态DScnt由109降到106，实际E64/E65为92.409/138.786及92.637/139.000 μs，同批v190为92.200/137.719 μs，两版停止。这些结果不能证明所有供数调度方法无效，只排除已测试的具体实现。

完整K256 step部分展开与wave-uniform控制条件的实验已完成。v205两步展开出现一个SGPR spill；v206四步展开出现private44 B、VGPR spill10、SGPR spill23，均CPU拒绝；v209只禁postRA LICM仍spill1，也未运行GPU。v207仅scalarize原tensor-wait选择条件，v208另scalarize task/refill/expert guards，资源462/106、LDS295936 B、private/spills0；v210两步展开禁machine/postRA LICM为451/102、无spill。三者双规模八hash/e2e与dynamic/poison/eager/两graph各五次replay通过；实际E64/E65为92.923/139.536、93.322/142.349、92.820/137.443 μs。本批v190开头91.106/137.883、结尾92.531/137.299 μs，没有可靠收益，三版停止。该筛选批次与正式三轮独立，不将尾部所有权疑点改写为首sample就idle。

针对task入口和scalar descriptor工作，v211/v212将owner不变的GROUP1、current/next source base、LDS base与rotation提前准备，所有28次input请求、OOB/stride/multicast/early-timeout、wait、正向K贡献及量化输出保留。CPU分别462/105、461/98，LDS295936 B、无private/spill，static4654/3460；双规模八hash/e2e及dynamic/replay均通过，每版E64/E65一个artifact且actual完整ELF与CPU一致。v211原E64/E65为91.498/137.961 μs，v212为93.088/140.011 μs；v190同批首尾为91.534/91.471和136.838/137.264 μs。两版不采纳，静态代码下降没有带来目标耗时收益。

v212 ISA的steady refill在`0x2c84`与`0x2c90`使用runtime `s_lshl_b32`，在`0x2c8c`将K offset sign-extend至64-bit；它是具体地址生成证据，不能由此认定回退唯一原因。v213恢复K-loop外A/B data与共享scale的固定stride分派；v214另用Int64形成K byte offset。两版CPU465/104、LDS295936 B、private/spills0、static4172、九个CFG区间通过，双规模八hash/e2e、dynamic/poison/eager/两graph replay已过。实际E64/E65为94.243/138.431、92.892/139.075 μs，同批v190首尾为92.308/91.437和138.255/137.565 μs，两版停止。减少静态地址生成代码没有获得目标耗时收益，不能以单次e2e起伏替代GEMM1目标。

后续v215在runtime E支持下把完全无效capacity tile的四N-task setup整体绕过，144fixture/323328个判定case、受保护AST及actual CPU检查通过，资源463/106、LDS295936 B、private/spills0、static4708。seed17双规模完整e2e八hash及dynamic/replay通过，同一个artifact/完整ELF，但原完整筛选E64/E65为92.048/138.183 μs，慢于首尾v190的91.886/137.569与91.873/136.635 μs，停止而不合入。v216仅移动无效tile的accumulator初始化，但actual`.text`与v190相同，因此停止而不重复GPU计时。此方向针对capacity tail的非计算工作，不从sampled compute waves的prologue占比推导可省μs。

SiLU分组v217/218保留clamp/tanh/原公式/舍入，仅将独立TRANS的CHUNK4改成2/8；两版CPU463/107、LDS295936 B、private/spills0，完整e2e八hash及dynamic/replay通过，同规模共用实际artifact。原完整E64/E65为92.898/138.794和92.521/137.409 μs，同批v190首尾90.994/137.599与91.523/136.918 μs，两版均停止。CHUNK16的v219与v218同`.text`，仅CPU。没有新候选ATT或counter，不能将这组负结果定量归因为TRANS吞吐、cache或HBM。

所有具体方法、命令、actual ELF、pre/post边界与决定写入[OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。v205–v210的171项/71116596 B、v211/v212的92项/13323968 B、v213/v214的93项/13690522 B和v217–v219的115项/23006259 B证据已下载并全部核SHA256，无`.out`；对应汇总JSON为`logs/runtime205_v210_control_summary_20261019.json`、`logs/runtime211_v212_prepared_summary_20261019.json`、`logs/runtime213_v214_conststride_summary_20261019.json`和`logs/runtime217_v219_actchunk_summary_20261019.json`。

目前主要证据指向 compute 中供数、WG同步、WMMA和register/compiler调度的组合限制。9.1246% handoff、11.8098% barrier stall不能直接从91.688 μs中扣除；下一步由实际原benchmark和数值门禁决定。共享A LDS槽方向按用户指令保持放弃。

## 6. 复现与本地证据

host从共享仓库根目录使用新label采集：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_runtime_expert_trace_host.sh \
  64 optx_v190_edyn_e64_att_repro_01
```

分析已有capture，每项分别120s：

```bash
task=.codex_tmp/gemm1_t1536_20261017
capture=my_code/thread_trace_runs/optx_v190_edyn_e64_att_20261019_r1
timeout --signal=TERM --kill-after=5s 120s \
  python3 "$task/analyze_optx_skip_trace.py" "$capture" \
  --output "$task/logs/optx_v190_phase_repro_01.json"
timeout --signal=TERM --kill-after=5s 120s \
  python3 "$task/analyze_optx_precise_pcs.py" "$capture" \
  --output "$task/logs/optx_v190_pc_repro_01.json"
```

下载并核验174项任务证据、29038731 B，没有`.out`；manifest为 `logs/runtime190_v192_evidence_20261019.sha256`。本地可读：[phase JSON](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v190_edyn_e64_att_20261019_r1_phase.json)、[precise PC JSON](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v190_edyn_e64_att_20261019_r1_precise.json)、[完整 ELF 绑定](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v190_edyn_e64_att_20261019_r1_elf_binding.json)、[实际 ISA](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v190_edyn_TH6_seed0_20261019_step0_e64_actual_gemm1.lowered.isa.log)、[性能与e2e汇总](../.codex_tmp/gemm1_t1536_20261017/logs/runtime190_performance_e2e_replay_att_summary_20261019.json)。只下载任务所需分析与日志，raw capture仍在远端。
