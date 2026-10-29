# gfx1250 MoE GEMM1：E65/T1536/topk9 non-balanced 优化记录

## 当前已验收结果

2026-10-19已完成最新要求：E64/E65在正常生产入口复用同一个compiled GEMM1，无需额外UNIFIED开关。当前已验收lookup7共用默认的完整合格三轮中值：E65 GEMM1 **107.379 μs**、GEMM2 **94.3843 μs**、MoE e2e **256.893 μs**；E64分别 **95.3059 /60.4664 /209.078 μs**。两规模seed0/17共112项零字节差异、eager/graph canary、dynamic A/ScaleA/psum和原reference门限均通过，实际cache1/同compiled object/同ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`。本地与a07-3 live两源SHA一致，原GEMM2/helpers未改。E65距80 μs还差 **27.379 μs**；当前正式记录仍约107 μs，不宣称达到80 μs或稳定刷新纪录。

共用lookup7前一批为E65 107.752 /94.3317 /258.016 μs、E64 96.3286 /60.0702 /208.956 μs，同批GEMM1分别低0.558%/1.061%和0.221%/1.198%。固定反向对照已完成，E65 lookup8在两侧lookup7之间，其稳定提速未证明；E64 lookup7两侧均快于lookup8。早期lookup8共用v1为E65 108.741 /94.4689 /260.549 μs、E64 95.9637 /60.9281 /209.023 μs，E65比两侧旧控制慢0.914%/1.583%。这些按批次保留，不挑选最快中值替换当前默认。详细原始表、排除记录、实现和验收见后文共用章节。

原用户命令的旧E65-only group8默认 GEMM1 完整合格三轮中值为 **107.330 μs**，同轮相邻group16对照为 **108.462 μs**，改善1.044%。原 HEAD 历史中值为 **153.405 μs**；当时耗时下降30.03%，该记录保留为历史。此前group16默认的独立中值107.170 / 107.223 μs保留为历史验收，不能据此声称本轮刷新绝对纪录。此前M32默认为109.251/109.148 μs。106.795 μs、106.997 μs分别是M-order和group8候选完整组中值，不替代当前共用默认验收。

2026-10-18按用户要求完成原HEAD与同一group8最优版的正反向重测，并回收186个raw证据member独立验收。三个合格baseline组中值为154.063 /153.987 /153.659 μs，三个合格best组中值为107.628 /107.381 /106.900 μs；相邻正反向改善30.106%–30.430%。这是已验收代码的复测，不将106.900 μs挑为新优化纪录，历史107.330 μs记录保留。本轮MoE e2e组中值由299.164–299.892 μs降至256.399–257.823 μs，实际GEMM2 ELF一致但best路径下GEMM2中值为93.3788–94.5659 μs，高于baseline的89.6808–90.4144 μs。完整组、排除组和计时口径见后文重测章节。

2026-10-18用户补充验收要求：除E65/shared-experts1/T1536/topk9目标规模外，还须确保E64/shared-experts0/T1536/topk8、model_dim7168/inter_dim2048、random non-balanced routing的完整MoE e2e结果和精度正常。E65保留有效GEMM1 payload/ScaleA、GEMM2和最终MoE输出的严格对比及reference检查；E64需独立完成完整路径验收，既有独立GEMM1结果不足以替代。该补充在手头`static_swizzle_v1`实验结束后执行，两规模验收通过前不据GEMM1单项正确性认定整体通过。

该补充已完成当前生产源码的双规模/双seed完整验收：E65/topk9/shared1的`logits_diff`为3.7793e-06 /3.7788e-06，E64/topk8/shared0为3.4733e-06 /3.4766e-06。每个seed均28项零字节差异，涵盖GEMM1有效payload/ScaleA、GEMM2有效行、最终MoE输出、eager/graph canary及dynamic replay；原reference门限保持。E64本次使用当前production dispatch，actual GEMM1 ELF为原fallback，不使用独立复制kernel。验收=`dual_moe_current_downloaded_acceptance_v1.json`，完整证据与复现命令见后文；有限测试通过不抹去历史偶发GPU fault的未定位记录。

旧E65-only默认限定E65/contiguous_m30464/non-balanced/t256/w2x4/b4/K7168/N4096/SiLU fused-quant守卫，参数为 `(N_TASKS,STAGED_QUANT_SCALE,ROW_TAIL_PREDICATION,HALF_A_PREFETCH,HALF_SCALE_PREFETCH,ROW_TAIL_A_GROUPS,PAIR_A_PREFETCH,TASK_M_ORDER,TASK_M_GROUP)=(1,1,3,1,1,2,1,1,8)`。旧版两个seed各28条零字节差异，全部HEAD有效输出、各三次eager/graph canary和三组dynamic input/psum replay通过；T1537/seed17另28条通过。旧group8 ELF为`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`。当前共用默认采用相同数值pipeline，真实expert数传runtime；目标E64/28672、E65/30464均按同配置与hint0编译，其他scope及偏离默认的显式旧配置保留原fallback。后文各阶段数据按时间保留，早期“最终”或“默认”结论由本节及后续验收更新。

## 目标与口径

在 random、non-balanced routing 下，将 E65/T1536/topk9、shared-experts=1、model_dim=7168、inter_dim=2048、A4W4、SiLU、无 bias 的 GEMM1 优化到 80 μs 以下。最初目标为90 μs，后续用户要求以80 μs为当前目标。64 个 routed experts 的 12288 条 route 和 shared expert 的 1536 条 route 均保留，共 13824 条有效 route。

2026-10-18用户澄清：此前“减少实际计算”的限制有误；允许跳过不影响有效结果的无效计算，前提是不降低kernel通用性和结果精度。本任务后续按最新消息中的80 μs执行，保留旧90 μs目标的历史记录。允许裁剪全padding M16/M32、OOB等无效工作；13824条有效routes对应的全部有效输出、56次K128更新及顺序、SiLU、BF16 rounding、FP4/E8M0和canonical布局保持。不得伪造计时、使用const0特例或放宽精度标准。

正式数据使用用户原命令的完整 MoE eager profiler 中 GEMM1 的 `device_time_avg`，保持 warmup、iters、计时与精度门槛。隔离 kernel 计时与 ATT cycles 仅用于诊断。每次性能测试前后保存整机 KFD、GPU 利用率、VRAM 与 boot ID；存在占用时每 6 秒重查，计时期间不运行监控。

## 环境与源码起点

- 本地 worktree：`C:/Users/yanguahe/Documents/code/wk_sp1/aiter_dsr1_e65`。
- 分支：`codex/dsr1_e65_gemm1_20261017`。
- 起点：原 `aiter` 的 `moe_a4w4_pr_refactor_wt` HEAD `339589f900b3a880ec447f605886fe803cbf9649`。
- host：`yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`。
- host/container 共享仓库：`/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`。
- 容器：`dsr1_ep4`；镜像：`rocm/fw-bringup:gfx1250-atom-20260930-na`。
- 起始 persistent 源码 SHA256：`b1f0ff87c886915b737cebbb77f61ca67bd213abde41f913c7e9400a251bf6b4`。
- 起始测试脚本 SHA256：`10b856f4b9b94fce538ed3c21b01b0211aa238788f414f9768ed846a7ac09c1d`。
- 原始仓库与远端均保留无关文件和修改；仅同步本任务所需文件。

## 复现命令

先在 host 完成整机空闲检查，再在 `dsr1_ep4` 容器运行：

```bash
cd /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
PYTHONPATH="$PWD" \
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=false \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 65 \
  --tokens 1536 \
  --topk 9 \
  --shared-experts 1 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20
```

`PYTHONPATH` 指向指定共享仓库，避免误用容器默认安装的 `/app/aiter`。host 自动控制脚本为 `my_code/run_dsr1_e65_gemm1_host.py`；它保存源码快照、SHA256、完整命令、每轮原始日志和空闲检查。正式运行示例：

```bash
python3 my_code/run_dsr1_e65_gemm1_host.py baseline --runs 3
```

## 基线与精度

实际生产 GEMM1 symbol 为：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e65_act1_q1r8_cn4_cm1_prefetch_apre_persist
```

起始源码下，完整命令三轮正式样本如下，均 `count=19`，原测试精度全部通过，`logits_diff=3.7793e-06`、`rel_l2=2.7493e-03`：

| 轮次 | GEMM1（μs） | MoE e2e（μs） | 原始日志 |
|---:|---:|---:|---|
| 1 | 153.405 | 299.172 | `.codex_tmp/dsr1_e65_gemm1_20261017/baseline_settled/run_1.log` |
| 2 | 153.709 | 299.710 | `.codex_tmp/dsr1_e65_gemm1_20261017/baseline_settled/run_2.log` |
| 3 | 152.657 | 298.747 | `.codex_tmp/dsr1_e65_gemm1_20261017/baseline_settled/run_3.log` |
| **中值** | **153.405** | **299.172** | |

距 90 μs 边界为 63.405 μs，需降低 41.33%。E64/topk8 和旧 balanced 优化记录的数据不能作为 E65/topk9 基线。

三轮前检查均整机空闲。进程退出后的即时检查短暂保留已退出 PID 的 KFD entry、约 6 GB VRAM；`fuser` 无 owner，`gpu_users.sh` 无活跃进程，`/proc/<pid>` 已不存在。6 秒后的独立恢复检查均 KFD 空、busy=0、VRAM=176582656 B，boot ID 一致。即时和恢复快照均保存于 `idle_*.jsonl`，不删除即时记录。最初未区分 driver 退出清理的三轮 151.497、153.819、153.178 μs 保留在 `baseline/`，未计入正式基线统计。

## 取得性能提升的优化

### 拆分连续 N-task，改善 non-balanced routing 的尾部

原 persistent WG 依次计算四个 N-task。将其拆成两个或一个 N-task，增加较短 WG 数量，保留所有有效 routes、K128 累加顺序、激活、BF16 rounding、量化和输出布局。两任务使用 `m_tile = bid_x // (cluster_n * 2)`、`n_group = (bid_x // cluster_n) % 2`；单任务使用已有 contiguous-M swizzle。每个 N-task 的输入和输出 lifetime fences 保留。

| 组 | GEMM1 三轮（μs） | GEMM1 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | 验收 |
|---|---|---:|---|---|---|
| 原 HEAD `baseline_settled` | 153.405, 153.709, 152.657 | 153.405 | 89.811, 90.094, 89.744 | 299.172, 299.710, 298.747 | 三轮有效 |
| `nt2_perf_a` | 129.289, 128.794, 129.616 | 129.289 | 89.132, 90.579, 90.509 | 274.426, 276.113, 276.031 | 三轮有效 |
| `nt1_perf_b` | 120.018, 118.541, 117.672 | **118.541** | 93.594, 94.294, 94.214 | 269.551, 268.301, 266.319 | 三轮有效 |
| 反向 `parent_perf_c` | 153.098, 153.621, 153.450 | 不将三轮合并统计 | 89.645, 90.140, 90.058 | 298.841, 299.573, 298.597 | 第一轮并发 CPU 编译初始化 KFD，排除；后两轮有效 |

nt1 的有效中值较原 HEAD 降低 **22.73%**，仍比 90 μs 多 **28.541 μs**，没有达到目标。nt1 的 MoE e2e 中值为 268.301 μs。GEMM2 源码没有改动，但实测约 4 μs 的变化必须保留，不能将 GEMM1 的独立收益直接当作 e2e 收益。

复现时，在原命令前分别增加 `AITER_GFX1250_GEMM1_N_TASKS=2` 或 `AITER_GFX1250_GEMM1_N_TASKS=1`。当前仍处于候选验证阶段；最终默认选择、原 HEAD 精度绑定和 replay 验证完成后另行记录。

```bash
# host 控制脚本：每轮独立执行整机空闲门禁，容器运行原 benchmark。
python3 -u my_code/run_dsr1_e65_gemm1_host.py NEW_NT1_LABEL --runs 3 \
  --env AITER_GFX1250_GEMM1_N_TASKS=1
# 正确性诊断；这里的 parent_seed0.pt 来自 patched 默认四任务路径。
python3 -u my_code/run_dsr1_e65_gemm1_host.py NEW_VERIFY_LABEL \
  --env AITER_GFX1250_GEMM1_N_TASKS=1 \
  --command python3 -u my_code/verify_dsr1_e65_gemm1.py \
  --seed 0 --compare .codex_tmp/dsr1_e65_gemm1_20261017/parent_seed0.pt
```

nt1/nt2 seed 0、nt1 seed 17 对 patched 默认四任务路径及 standalone producer 的有效输出对照通过：payload 14,155,776 B、ScaleA 884,736 B、GEMM2 有效行 198,180,864 B、MoE 输出 22,020,096 B，均逐字节一致。seed 0 原精度 `logits_diff=3.7793e-06`、`rel_l2=2.7493e-03`；seed 17 为 `3.7788e-06`、`2.7491e-03`。未经修改的原 HEAD exact 对照尚需补齐；不能将 patched 对照误写成原 HEAD 对照。

### Thread trace 证据

使用 `my_code/get_att_b_rocm_sdk.sh` 在 `dsr1_ep4` 采集原路径与 nt1 的 SIMD0、四个 SE，使用 random 原 workload；诊断 iters=2。ATT/decoder/analyzer 成功，目标 symbol、PC 映射完整，无 `.long`。raw archive、decoded waves、ISA 和统计保存在 `.codex_tmp/dsr1_e65_gemm1_20261017/e65_g1_{baseline,nt1}_20261017/`。

| 观测 | 原四任务 | nt1 |
|---|---|---|
| 四个 SE 的采样跨度（cycles） | 136820, 269647, 138591, 264972 | 182658, 181749, 179289, 179573 |
| 每个有效 wave 的动态 WMMA 数 | 3584 | 896；WG 数量增加，全部 N 范围保持 |
| nt1 有效 wave duration P50 | — | 35456.5 cycles |
| nt1 入口至首次 WMMA P50 | — | 4766 cycles |
| nt1 首次至最后 WMMA issue P50 | — | 25494.5 cycles |
| nt1 epilogue/退出 P50 | — | 5024 cycles |

阶段边界统一为首条与末条 WMMA 的 issue timestamp，末条 WMMA 归入 epilogue。旧版按末条 WMMA completion 分界得到25509.5/5009 cycles；此处15 cycles的重新划分来自分析边界变化，没有重新采集 trace，也不表示性能变化。

LDS 282624 B、8-wave WG、实测每 SIMD 2 waves。nt1 steady 的 WG wait（idx1061）432 条记录 P50 112.5 cycles，DS wait（idx1059）P50 88.5 cycles；另一组 DS/WG wait 为 71/32 cycles。证据支持拆任务后尾部更均匀；入口、同步、LDS operand 搬运和 epilogue 是下一步候选。`occupancy.json` 的 workgroup_id 均为 0，不能将 wave 唯一绑定到 logical expert。ATT cycles 不是正式 kernel μs，也不能独立相加预测性能。

硬件依据：`MI400_Shader_Programming#65.txt` §4.10.1（p197）说明 TDM 同 wave 按顺序完成，但跨 wave、相对其他 memory 类型无序，因此没有删除输入/输出 lifetime waits。`mi400_lds_arch#13.txt` §7.3 说明不同 segment 可使两个 port 访问相同 banks；这仅支持布局实验的原理，尚未证明当前 kernel 的具体 bank conflict。CDNA5 ISA §5.7.2.1（p56）说明关闭 XDL arbitration stall 主要可能有利于单 wave/SIMD；本次有两个 wave/SIMD，没有据此盲目启用。

## 未保留的实验与无效样本

| 实验 | 精度/资源/性能结果 | 处理 |
|---|---|---|
| t256 cyclic input stage split、80KB pitch | LDS327680、VGPR456、SGPR70、zero spill；seed0 payload 差异9,114,958 B | 精度失败，撤出，未计时 |
| MMA_GROUP=4 | seed0 exact；GEMM1 118.279, 119.053, 119.534 μs，中值119.053；e2e 268.871, 268.633, 270.170 μs | 三轮有效但无收益，撤出 |
| M128/w1x4/b3、保留256行 route alignment | seed0 exact；LDS159744、VGPR461、SGPR68、zero spill；133.393 μs 有效，132.168 μs 有外部 owner；第三轮rc134 | 无有效收益，撤出 |
| nt1 WMMA reuse1 | seed0 exact；首次正式运行rc134，profiler 后 reference stage capture `Memory access fault ... Page not present or supervisor privilege` | 失败运行全部计时无效，待限定复核 |
| wave-parallel expert lookup | seed0/17 exact；首组仅第一轮通过 ownership 门禁 | 性能收益尚未成立，继续相邻对照 |

`nt1_perf_a` 的119.540、119.774 μs 因外部 E64任务 owner 排除，118.193 μs 单轮有效；`parent_perf_b` 的154.069 μs 有效，155.158、152.005 μs 排除。所有日志保留，不从受污染组挑选有利值。失败捕获发生时还观察到其他 E64 verify进程，现有证据不能将 fault 唯一归因于某个候选或外部任务。

## 2026-10-17 继续实验的阶段记录

相邻 nt1 对照 `nt1_control_d` 三轮均有效：GEMM1 119.470、118.827、118.490 μs，中值118.827；GEMM2 93.3070、93.3957、93.7942 μs；e2e 268.320、268.170、267.266 μs。parallel lookup 首组 `nt1_plookup_perf_a` 为117.094、117.796、120.653 μs，后两轮检测到外部 owner，不能用三轮中值宣称收益。第一轮有效但仍需重复对照。其 seed0、17 有效输出全部逐字节一致。

以下仅为编译证据，尚未完成 GPU 验证或正式性能测试，**不是已取得的性能收益**。所有候选保持原 F32 累加、SiLU、BF16 rounding、FP4/E8M0、实际 routes 与输出布局。

| nt1 编译候选 | LDS（B） | VGPR/SGPR | static instructions | ELF SHA256 | 状态 |
|---|---:|---|---:|---|---|
| tied rounded-vector reuse | 282624 | 458/68 | 4138 | `cb997bc4888ee7172f6b48fdc54ea343f9877adb9354a0a36ea26ff60007bd8a` | zero spills/private；待精度 |
| N4096 descriptor | 282624 | 458/68 | 4057 | `d4ccace3c09c4b1aa780b1e0b49962af9cc628721aa7361c6b5d273415537b3a0` | zero spills/private；待精度 |
| 三缓冲 pipeline | 211968 | 458/68 | 3949 | `76041b5f694ade2ef9937976aa56c99182a99eb48ce2ad15fbdbbad931be1722` | zero spills/private；待精度 |
| M swizzle group=1 | 282624 | 458/68 | 4176 | `c274724e0973ec9f3a9d0b0f6e431308b76aec75b6622219d2ef47a9986dae887` | zero spills/private；待精度 |
| 内部 K128、外部 t256 | 141312 | 424/68 | 3570 | `bfca047603d391b63f97af5b10049bf451b94f91ac2605cfa0089b26aded25eec` | zero spills/private；待精度；保留原 K128 累加顺序 |
| 内部 M128/K128、外部256行 expert alignment | 106496 | 248/68 | 2218 | `59a5a2f3919a1d365da39f6d7ff7f3b2c6b739561c3e11cab44dd53521078f227` | zero spills/private；待精度；拆成更多 WG，全部有效行保留 |

原任务公共源文件的审计脚本 `my_code/audit_dsr1_e65_source.py` 已确认：GEMM2 function、`_amax_inputs_from_bf16x16` 的 AST 与 HEAD 相同；原 benchmark、`grouped_moe_gfx1250.py`、`quant_utils.py`、`gemm_common_gfx1250.py` 逐字节相同。诊断脚本增加 `--original-head`、`--replays`，用于绑定原始 SHA 的精度参考以及相同地址的 eager/CUDAGraph canary；脚本准备完成不代表这些 GPU 验证已经通过。

等待期间检测到外部 E64 `verify_gemm1_fused_apre_quant.py` 进程持续占用 GPU，busy=100%、VRAM约4.4GB。本任务没有杀进程或竞争 GPU；精度和性能运行均排队等待。需要修改候选或编译时仅中断本任务尚未 dispatch 的 idle waiter，保存日志，编译完成后恢复6秒空闲轮询。

### 后续资源候选与运行时等待诊断

当前工程指令允许已有 GPU 占用时进行正确性验证。控制脚本新增显式 `--correctness-only --command ...`，此类 manifest/result 固定标记为正确性诊断，所有耗时无效，不进入正式性能汇总。原 HEAD seed0 的 `original_head_seed0_c0` 已按原 SHA 加载两份生产源，但尚未完成验证；两次 native stack 都停在 `hsa_executable_freeze -> RegionMemory::Freeze -> BlitKernel::SubmitLinearCopyCommand -> hsa_signal_wait_scacquire`。因此不能宣称原 HEAD exact gate 已通过，也不能把这个等待归因于 GEMM1。日志与调用栈完整保留，没有结束或重置任何 GPU 进程。

为了提高剩余等待时间的利用率，继续仅编译下列候选，未 GPU dispatch、未计时：

| nt1 候选 | LDS（B） | VGPR/SGPR | static instructions | 状态 |
|---|---:|---|---:|---|
| M64/K128 | 90112 | 152/74 | 1892 | zero spills/private；待 GPU 验证 |
| M128/K256、单 operand rmem slot、b4 | 212992 | 261/68 | 2582 | zero spills/private；待 GPU 验证 |
| M128/K256、单 operand rmem slot、b3 | 159744 | 245/68 | 2394 | zero spills/private；待 GPU 验证 |
| 上组加 N4096 descriptor | 159744 | 245/68 | 2293 | zero spills/private；待 GPU 验证 |
| 上组加 tied rounded-vector reuse | 159744 | 245/68 | 2268 | zero spills/private；待 GPU 验证 |
| 上组保留原 rounding、使用 unsigned task indices | 159744 | 245/68 | 2231 | zero spills/private；待 GPU 验证 |
| M64/K256、单 operand rmem slot、b3 | 133632 | 162/68 | 1796 | zero spills/private；待 GPU 验证 |
| 原 t256 nt1、最少固定步数 expert search | 282624 | 458/68 | 4169 | zero spills/private；待 GPU 验证 |

M64/M128 只拆分内部 M tile，外部专家布局、256 行 expert alignment、全部有效 routes、输入/输出格式保持原样。单 register slot 在每个 K128 的原 WMMA 完成后才覆盖 operand，顺序仍为原 K128 累加顺序。b3 保留所有 input lifetime fences；输出在最后 K 计算与输入排空之后复用 LDS。N4096 仅将已知的 descriptor 维度常量化，运行时 ABI 保留。最少固定步数为 `E.bit_length()`，E65 为7步；CPU upper_bound/有效 expert/尾部 sentinel 枚举已覆盖75495组，GPU exact gate 仍未完成。

MI400 Shader Programming Guide §3.3.2.1（p32）记载每 SIMD 1024 VGPR、wave32 按16个分配。M128/K256/b3/rmem1 的245 VGPR分配到256，8-wave WG与159744 B LDS从资源预算上允许两个 WG，但这只是预算，必须由实际 trace 确认。该机器KFD公开的LDS是320 KiB，不能直接使用旧LDS设计稿中的384 KiB估算实际occupancy。

编译复现：

```bash
python3 my_code/compile_dsr1_e65_gemm1.py NEW_COMPILE_DIR \
  --tasks 1 --internal-m 128 --buffers 3 --rmem-slots 1 --nconst 1
```

精度通过后，候选的正式复现需要在原命令前增加：

```bash
AITER_GFX1250_GEMM1_N_TASKS=1 \
AITER_GFX1250_GEMM1_INTERNAL_M=128 \
AITER_GFX1250_GEMM1_PIPELINE_BUFFERS=3 \
AITER_GFX1250_GEMM1_RMEM_SLOTS=1 \
AITER_GFX1250_GEMM1_NCONST=1 \
<原 benchmark 命令>
```

上述环境变量仅用于候选试验，尚未成为默认生产选择。最佳合格性能仍为118.541 μs；没有把编译资源或静态指令减少记作实际性能提升。

### 非负 task index 地址计算候选

M128/K256/b3/rmem1/N4096 的 ISA 在 swizzle 和三缓冲 `% 3` 中保留了 signed division 的符号修正。新增默认关闭的 `AITER_GFX1250_GEMM1_UNSIGNED_INDICES=1`，仅将非负的 block/thread index、正尺寸的 swizzle 除法、非负 K tile 的环形索引改为 unsigned division/modulo，再转回原 Int32。expert psum、OOB 长度和所有数值计算继续使用原逻辑。

同一 M128/K256/b3/rmem1/N4096 组合的 static instructions 从2293降到2231；LDS159744 B、VGPR245、SGPR68保持，zero spills/private。ELF SHA256为 `f26bb883c5c6a113fe21082b66d0d2586e39b70931a882d66069b92a9c23ac52`。这只能证明编译后指令减少，尚未通过 GPU exact/replay，也没有实测性能收益。

```bash
python3 my_code/compile_dsr1_e65_gemm1.py NEW_COMPILE_DIR \
  --tasks 1 --internal-m 128 --buffers 3 --rmem-slots 1 \
  --nconst 1 --unsigned-indices 1
```

新增 CPU 地址审计逐字节对比 M64/M128/M256 的 FP4 payload 和 canonical M32 ScaleA 输出，覆盖完整 N4096 与多个 M 起点；另检查 K128/K256、b3/b4 的 issue/read 生命周期。它们是静态映射证据，不能替代 GPU 精度与 canary replay。实际性能与原 HEAD 精度验证仍等待运行时恢复；整机空闲检查每6秒持续保留在 `idle_resume_2102.jsonl`，性能测试尚未启动。

内部 M64/M128 的后续修订将 expert lookup 固定到原256行父 tile 的起点。此前内部 subtile 若完全落在某个 expert 的尾部 padding，直接对 subtile 起点 upper_bound 会选到下一个 expert，进行没有有效 route 的额外运算；这些输出 padding 本来未定义，没有构成此前有效行的精度证据。修订后保留父 expert，并仅在 `mn_oob > 0` 时执行内部 subtile，使整个 padding subtile跳过。全部有效 routes 和每个有效元素的原 K128 WMMA 更新仍保留；这不修改原 t256生产路径。重新编译的候选须使用独立目录/SHA，不能把前述ELF直接算作修订后候选。

修订后 M128/K256/b3/rmem1/N4096/unsigned 的 ELF SHA256为 `a35d79ed4c0b9f0cd47ec02ff7e50fee13f1d67d7ba381ef3b079bd6fecbf5ec`，2235 static instructions；加 parallel lookup 的 ELF为 `725e235b6dde4368afb2adbd9e320502c72cf55f6828124dbc38f09203456a1c`，2215 instructions。两者均 LDS159744 B、VGPR245、SGPR68、zero spills/private，仍未 GPU 验证。

### 默认路径与原 HEAD 的编译比对

以相同 fake tensor layout、参数和工具链，分别编译当前源码的默认 tasks4以及保存并核对原SHA的未修改 HEAD。两个完整ELF逐字节相同，SHA256均为 `958d31dd24a22c8a38af0ec1113a27ee8c5bf69e8ffb124ff222758f11c68110`，LDS282624 B、VGPR480、SGPR106、4386 static instructions、zero spills/private。产物分别保存于 `compile_current_default_audit/` 与 `compile_original_default_audit/`。

```bash
python3 my_code/compile_dsr1_e65_gemm1.py NEW_CURRENT_DEFAULT_DIR
python3 my_code/compile_dsr1_e65_gemm1.py NEW_ORIGINAL_DEFAULT_DIR --original-head
cmp NEW_CURRENT_DEFAULT_DIR/kernel.elf NEW_ORIGINAL_DEFAULT_DIR/kernel.elf
```

该比对证明默认参数在这份完整编译产物上没有变化；不等于已经完成原 HEAD seed0/17的动态对照，后者仍卡在运行时加载等待中。GPU exact/replay 和正式性能系列仍须补齐。新的CPU审计已通过1005组任务覆盖、75495组expert边界、75495组父 tile 与有效 subtile所有权、1462272个输出字节地址以及224个input ring步骤。

### Input fragment 地址审计补充

`audit_dsr1_e65_mapping.py` 新增独立 inverse permutation：以公开的 `shuffle_weight_f4` / `shuffle_scale_f4` 布局解码 source byte，再对比 consumer fragment 的逻辑 row/K 坐标。覆盖内部 M64/M128/M256、K128/K256、全部7168维K、两个K128 register subtiles、所有 A/B/ScaleA/ScaleB 唯一 fragments，以及不同expert与M/N起点。通过8716288个payload字节地址、544768个scale字节地址；未触及16 B的TDM padding区域。此检查不能验证GPU可见性或backend实际pointer lowering，不能替代device exact/replay。

```bash
python3 my_code/audit_dsr1_e65_mapping.py
```

源码/工具同步核对：persistent SHA256 `f966ab82704b024733844c94607c99c2da13139746e6ce9138fa45f488134efb`；launcher `bfe0c2c336faaedbf74f5ad65727802fa45246684535fadfd9c86a95ff76d611`；compile script `60d9b2b001e32b4501202e89b94c2bd63994ffe2cc8fd61bac2623f53c6e8d30`；新增mapping audit `be577b29cf5887ddda01c8a19bb8417b72ef26d17fde73f6f83ced4d36326f89`。

截至2026-10-16 21:49 UTC，外部E64 GPU进程818565与本轮原HEAD验证进程930575仍存在，整机busy=100%、VRAM5741613056 B。原HEAD验证日志最后更新为20:23:49 UTC，尚无精度结果；之前两个native stack均停在运行时代码对象加载的copy completion wait，不能认定目标GEMM1已执行。仅核对了进程、日志、driver记录与CPU源码，没有修改/结束GPU进程，没有新的有效性能数据。独立host waiter每6秒继续检查，原HEAD动态gate、新候选GPU精度及三轮原命令计时仍待完成。

### 生产 tensor layout、显式 carry 地址与实际 ELF 核对

编译诊断脚本默认改为生产 ScaleA view：`(1, 3808, 448)`、`torch.int32`、stride `(1705984, 448, 1)`，storage 6823936 B。旧 fake shape为`(1, 952, 7168)`，新增`--tensor-layout legacy`仅用于复现旧产物。输出 ScaleA 的原 shape `(1, 3808, 512)`、`torch.uint8`本来正确，没有改动。脚本记录所有 tensor 的 shape/stride/dtype/storage bytes与生产源码SHA。

相同rank的TorchTensorJitArg动态传递shape/stride；生产layout重编译后的旧M128候选完整ELF与原fake layout产物逐字节相同，SHA均为`a35d79ed4c0b9f0cd47ec02ff7e50fee13f1d67d7ba381ef3b079bd6fecbf5ec`。这排除了此编译诊断中fake shape造成code object差异的情况，不替代动态正确性验证。

内部M/K候选的drain tail改用`ptr_to_idx(buf_ptr(...))`明确传递下一stage的LDS地址；已有t192分支继续显式地址，已测t256 nt1和默认四任务保留原lowering。公共`make_sgpr_opaque`和GEMM2没有改动。LLVM LangRef的“Input constraints”明确只有`"=s,0"`这类tied constraint保证输入输出同寄存器，原空asm的`"=s,s"`本身不保证identity。源码历史也记录过tied/real `s_mov`仍出现random错误，所以这里只消除新候选对该假设的依赖，不能宣称已唯一定位或修复某个GPU错误。

| 新编译目录 | LDS（B） | VGPR/SGPR | static instructions | ELF SHA256 | GPU状态 |
|---|---:|---|---:|---|---|
| `compile_m128_parent_production_layout` | 159744 | 245/68 | 2235 | `a35d79ed4c0b9f0cd47ec02ff7e50fee13f1d67d7ba381ef3b079bd6fecbf5ec` | 旧carry对照；未验证 |
| `compile_m128_explicit_carry` | 159744 | 256/68 | 2237 | `40383f30cfdb603443ce215e836c54f61eafea7e7edb7aaed92fa83f2e1dab93` | 优先exact/replay；未计时 |
| `compile_nt1_trace_match` | 282624 | 458/68 | 4180 | `10df3ead8f2f3c88288d05e748d8fd6f38df4905f30b7c448ba923fd24d2c92f` | 与已测nt1实际ATT ELF完全相同 |
| `compile_m64k128_b3_explicit` | 67584 | 126/68 | 1455 | `c41de4f0e5f677e5dc484f7490c65c07c4c3b46108a84bf5865a836e2bb73094` | 未验证、未计时 |
| `compile_m128k128_b3_explicit` | 79872 | 248/68 | 1954 | `5c7d0ff3d678b289af849c67e551cba994afd633a81030e3423e3166bbbfd0d9` | 未验证、未计时 |

上述产物均zero spills/private。M128显式carry从245到256 VGPR仍处于相同256分配档，没有跨越资源档位。M64/K128较小LDS提供更多resident WG预算，同时会增加WG数、重复B请求和K同步；不能仅由LDS或指令数预测正式耗时。

`compile_nt1_trace_match/kernel.elf`与`e65_g1_nt1_20261017/thread_trace/simd0/kernel/rpf_v3/out_gfx1250_code_object_id_22.out`完整32768 B逐字节相同。因此新内部候选修改没有改变已有nt1测试所用code object。

```bash
# 只编译，禁止GPU dispatch；默认production tensor layout。
python3 my_code/compile_dsr1_e65_gemm1.py NEW_M128_DIR \
  --tasks 1 --internal-m 128 --buffers 3 --rmem-slots 1 \
  --nconst 1 --unsigned-indices 1
python3 my_code/compile_dsr1_e65_gemm1.py NEW_M64K128_DIR \
  --tasks 1 --internal-m 64 --internal-k 128 --buffers 3 \
  --nconst 1 --unsigned-indices 1
```

### ATT阶段统计补充

`analyze_dsr1_e65_gemm1_att.py`增加每wave阶段统计，沿用上述issue时间边界。nt1共60个waves，其中36个有效waves各896次WMMA，24个跳过；原四任务共14个waves，其中8个有效waves各3584次WMMA。全部PC映射完整。

| nt1有效wave观测 | P50（cycles） |
|---|---:|
| wave duration | 35456.5 |
| entry span | 4766 |
| compute span | 25494.5 |
| epilogue span | 5024 |
| compute阶段WG waits累计 | 2486.5 |
| compute阶段DS waits累计 | 2379.5 |
| entry阶段tensor waits累计 | 1288.5 |
| epilogue阶段tensor waits累计 | 914.5 |

这些分别是各wave统计量的中值，不能相加作为kernel总耗时预测。原四任务的compute span中值122409 cycles包含三次中间task epilogue，与nt1单task compute不是相同边界，不能按四倍直接比较。

本轮两端生产源码SHA已一致：persistent `7072a3124306005017411584410299fe54c39a8a65b826ff612670bfec60365e`，launcher `bfe0c2c336faaedbf74f5ad65727802fa45246684535fadfd9c86a95ff76d611`；编译脚本SHA为`47a6bce533e425acd9c0f8e3cc3a5f1d246c16f9beca4bd329e0bb85ad00e6f8`。analyzer SHA为`ee181d544a074eccb14d5ab9d56564b0ef03d5d488e4f0baf4a0a45f5365843e`。

截至2026-10-16 22:36:32 UTC，boot ID仍为`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，GPU PID818565/930575均在，busy=100%、VRAM5741613056 B；原HEAD验证日志仍813 B、mtime20:23:49 UTC。独立host waiter每6秒检查且不自动dispatch，本轮没有新增GPU进程或有效性能数据。恢复后须先完成原HEAD与候选exact、动态replay，再做三轮原benchmark与相邻nt1反向对照；118.541 μs仍是最佳有效性能，90 μs目标尚未达到。

### f01-1 独立环境与原 HEAD 动态精度验收

a07-3 持续被外部 E64 验证进程及原 HEAD 运行时加载等待占用，因此在历史已使用的 f01-1 建立独立验证目录 `/data/yanguahe/code/wk_sp1/aiter_dsr1_e65_verify_20261017`，使用已有 `dsr1_ep4` 容器。该容器此前处于 exited，启动命令为 `sleep infinity`，启动后整机四卡均空闲。没有覆盖 `/app/aiter_moe_a4w4_pr_refactor_wt`。

两台机器的 PyTorch 均为 `2.11.0+rocm10.1.0a20260811-bkc.20260820`，镜像均为 `rocm/fw-bringup:gfx1250-atom-20260930-na`。f01 原安装 FlyDSL 为 `0.3.4.1`，本次仅通过独立 PYTHONPATH 使用从 a07 复制的 `0.3.3.dev886`，没有修改已安装的包。任务源码与现有 AITER 扩展也放在独立目录并核对 SHA256。源码 archive SHA256 `8201fbabc75560ba418b89f20b4380eed5d2fefed5ef9ff42ae19c3bd603488d`，FlyDSL archive SHA256 `aa3376de4f6977ed8a7b3ab0e39f6ec50575d6922a892b2ebf6343d644b42128`。f01 boot ID为 `1ed0e1e6-fdf0-45c0-b947-5e48834b63da`。

`f01_original_seed0`、`f01_original_seed17` 加载并核对未经修改 HEAD 的两份源码，均与 standalone producer 的 payload/ScaleA、GEMM2 有效行和完整 MoE 输出逐字节一致。`f01_m128_explicit_seed0`、`f01_m128_explicit_seed17` 的 M128/K256/b3/rmem1/N4096/unsigned 候选对这些原 HEAD `.pt` 也全部一致，且各通过三次 eager 和三次 CUDAGraph canary replay。

`f01_nt1_dynamic_seed0`、`f01_nt1_dynamic_seed17` 补齐现有 nt1 对原 HEAD 的严格对照、三次 eager 和三次 graph canary。动态 replay 在原 captured pointers 上更新 random FP4 payload、E8M0 ScaleA，随后更新 expert counts 并重新构建生产256行 alignment prefix sum。第一组保持13824条真实 route，后两组覆盖7569/8048条有效行、3/4个空 expert，以及 M16/32/64/128/256 边界；shared expert count保留。每组 eager 和已捕获 graph 的有效 payload/ScaleA均与独立加载的原 HEAD 逐字节一致。最后恢复原输入，完整 MoE 对照仍逐字节一致。该动态测试仅用于未计时的功能验收，没有替换正式 benchmark 的 routing。

### f01-1 同机性能对照

下面使用用户原完整 MoE eager benchmark，warmup5、iters20、GEMM1/GEMM2 count19。所有九轮 `returncode=0`、`ownership_qualified=true`，每轮前后检查整机全部 GPU，计时期间没有并行监控或编译。f01 与 a07 的性能数据单独记录，不能直接混算。

| f01 组 | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| `f01_original_perf_a`，原 HEAD | 199.172, 200.540, 196.404 | 199.172 | 129.422, 128.835, 129.422 | 413.40, 421.59, 410.89 |
| `f01_nt1_perf_a` | 150.721, 154.680, 154.056 | 154.056 | 127.376, 131.359, 130.358 | 352.21, 361.63, 362.19 |
| `f01_m128_perf_a`，M128/K256/b3/rmem1/N4096/unsigned | 167.521, 168.802, 169.262 | 168.802 | 128.888, 129.641, 130.887 | 370.19, 372.71, 390.22 |

nt1较 f01 同机原 HEAD降低22.65%。M128组合比同机 nt1慢9.57%，未取得性能收益，不作为默认选择。a07 最佳有效中值仍为118.541 μs，距90 μs差28.541 μs。

f01候选复现使用 host 控制脚本新增的显式 `--repo`，容器 PYTHONPATH 如下：

```bash
python3 -u my_code/run_dsr1_e65_gemm1_host.py NEW_F01_LABEL --runs 3 \
  --repo /data/yanguahe/code/wk_sp1/aiter_dsr1_e65_verify_20261017 \
  --env PYTHONPATH=/data/yanguahe/code/wk_sp1/aiter_dsr1_e65_verify_20261017/.codex_tmp/dsr1_e65_gemm1_20261017/flydsl_env:/data/yanguahe/code/wk_sp1/aiter_dsr1_e65_verify_20261017 \
  --env AITER_GFX1250_GEMM1_N_TASKS=1
```

### 内部 K128 候选精度失败与隔离结果

| 精度诊断标签 | 配置变化 | 首次 canary eager payload差异 | ScaleA差异 | 处理 |
|---|---|---:|---:|---|
| `f01_m128k128_dynamic_seed0` | M128/K128/b3/N4096/unsigned | 126 B | 未继续检查 | 失败，未计时 |
| `f01_m64k128_dynamic_seed0` | M64/K128/b3/N4096/unsigned | 374 B | 未继续检查 | 失败，未计时 |
| `f01_m128k128_reuse0_seed0` | 上述 M128，关闭 WMMA reuse | 138 B | 未继续检查 | reuse不能单独解释，未计时 |
| `f01_m128k128_diagnostics_seed0` | 原 M128/K128/b3，再次运行，增加差异坐标输出 | 33 B | 0 B | 差异位于局部 M16/N32块，未计时 |
| `f01_m128k128_b4_seed0` | M128/K128，恢复四缓冲 | 433 B | 0 B | 三缓冲不能单独解释，未计时 |

这些差异均发生在动态输入更新之前；nt1 使用同一 replay 工具已经通过。诊断脚本现在先报告 payload 和 ScaleA 两种差异及坐标，再失败退出，严格逐字节标准没有放宽。内部 K128 未成为生产默认路径，不能将其编译资源改善记为实测收益。

新增默认关闭的 `AITER_GFX1250_GEMM1_INDEPENDENT_A=1` 仅隔离 A 的 TDM multicast，保留输入/输出格式、cluster geometry、计算和同步。此候选仍需验收；没有根据文档直接断言 multicast 是根因。硬件依据仍为 Shader Guide §4.10.1（p197）：TDM 仅保证同 wave 顺序，不能删除跨 wave 的 lifetime fences。

截至2026-10-16 23:47:02 UTC，a07 的 PID818565/930575仍存在，busy100%、VRAM5741613056 B，6秒 waiter继续运行。没有结束任何 GPU进程。

### K128 operand 存储隔离与性能淘汰

`f01_m128k128_aload_seed0` 关闭 A/ScaleA multicast，改为每个 WG 独立 TDM load 后，首次 canary eager 仍出现 890 B payload 差异、0 B ScaleA 差异，位于局部 M16/N32 块。因此 A multicast 不能单独解释此错误，未对失败候选计时。

`AITER_GFX1250_GEMM1_OPERAND_PINGPONG=1` 将当前 K128 operand 和下一 carry 放入不同的 rmem slots，按 `current_kt % 2` 选择。K128 的累加顺序和输出处理不变。`f01_m128k128_rping_seed0/17` 均通过对未修改 HEAD 的严格逐字节比较、各三次 eager/graph canary 和三组更新输入/routing 的 graph replay。这只能证明存储分离消除了这两个 seed 的重跑差异，尚未唯一定位根因。

| f01 组 | GEMM1 三轮（μs） | 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `f01_m128k128_rping_perf_a` | 471.902, 470.094, 466.733 | 470.094 | 131.795, 130.613, 128.321 | 675.61, 677.77, 675.02 |

三轮原 benchmark 均 `returncode=0`、`ownership_qualified=true`，前后整机空闲检查通过。`compile_f01_m128k128_pingpong` 为 LDS79872 B、VGPR512/SGPR68、private36 B、8 VGPR spills、3560 static instructions；完整24480 B ELF SHA256 `e86574b2836d9c40858552755bdcafc21bc3c61474477c16e3d016c43088ca775`。两个分支同时维护更多 live operand 导致资源明显恶化，实测无收益，不作为生产选择。

单 slot K128 产物的最后 WMMA 后紧接着 DS load，部分 destination VGPR 与该 WMMA 的 operand 重叠。当前 `compile_m128k128_b3_explicit/kernel.isa` 只有对 `HW_REG_WAVE_MODE` 的设置，没有进入 `SCHED_MODE 2` 的 `S_SETREG`；源码 GEMM1 compile hint 为 `amdgpu-expert-scheduling-mode=False`。Shader Guide §4.3.7.4.1（p92）记载的是 SCHED_MODE2 下暴露的 WAR hazard，不能据此直接断言本 kernel 的错误就是该 hazard。

新增默认关闭的 `AITER_GFX1250_GEMM1_OPERAND_REUSE_FENCE=1` 用于依赖隔离：在所有 accumulator 上绑定 `s_wait_alu depctr_va_vdst(0)`，加 memory clobber 和 scheduling boundaries，要求下一次同 slot 的 DS loads 位于 wait 之后。该实验先核对实际 ISA 与资源，再进行严格精度验证；未验证前不记录为修复或收益。

`compile_f01_m128k128_rfence` 的 ISA 确认所有六个 carry 点的 wait 均位于末尾 WMMA 与下一轮 DS loads 之间；LDS79872 B、VGPR205/SGPR68、zero spills/private、1986 instructions，ELF SHA256 `87b1c6451e6d5c3c72783c2d975ab0fa71fc79e682aba4b08e259b4c5fd3f8eeb`。但 `f01_m128k128_rfence_seed0` 首次 canary eager 仍有43 B payload差异、0 B ScaleA差异，局部坐标为 grouped_row2880 起、byte224–239。显式 wait 不足以消除错误，未计时，不能记录为 bug 修复。

### t256 nt1 的 descriptor/lookup/三缓冲组合

候选使用 `N_TASKS=1`、`PIPELINE_BUFFERS=3`、`NCONST=1`、`PARALLEL_LOOKUP=1`、`UNSIGNED_INDICES=1`、`MIN_SEARCH=1`。`compile_f01_nt1_nconst_b3_plookup` 为 LDS211968 B、VGPR458/SGPR68、zero spills/private、3736 static instructions，ELF SHA256 `2a697377df2aa0bb71ec26afc4011d7889ec75113f9052bcb842fa607359a2877`。

`f01_nt1_nconst_b3_plookup_seed0/17` 均通过原 HEAD 全部有效输出逐字节对照、各三次 eager/graph canary，以及三组同地址 random FP4/E8M0、expert counts/prefix sum 更新的动态 replay。原完整 MoE 精度仍分别为 `3.7793e-06/2.7493e-03`、`3.7788e-06/2.7491e-03`。

| f01 相邻组 | GEMM1 三轮（μs） | GEMM1 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `f01_nt1_nconst_b3_plookup_perf_a` | 149.123, 152.563, 150.314 | 150.314 | 130.039, 129.584, 132.026 | 356.01, 356.28, 360.16 |
| `f01_nt1_control_b`，原 nt1 | 149.753, 152.890, 149.384 | 149.753 | 128.513, 129.299, 129.314 | 353.71, 357.56, 354.06 |

六轮均 rc0、ownership合格、无live post owners。相邻对照显示组合候选慢0.37%，没有稳定收益，不作为最佳选择。不得用较早 nt1 的154.056 μs与本轮候选对比来宣称收益。

### 同一 t256 WG 内增加 M compute waves

新增默认关闭的 `AITER_GFX1250_GEMM1_COMPUTE_M_WARP=4`，将同一256行tile的 M compute分成四组，外部 tile/alignment、有效routes和K128累加顺序保留。16-wave WG中前8 waves保留原A/B/ScaleA/ScaleB TDM owners，后8 waves仅计算，但全部参与原WG lifetime fences。quant payload和canonical M32 ScaleA布局保持。

| f01 编译候选 | LDS（B） | VGPR/SGPR | VGPR spills / private（B） | static instructions | ELF SHA256 |
|---|---:|---|---|---:|---|
| `compile_f01_t256_w4x4`，双operand slot | 282624 | 256/68 | 10 / 36 | 2495 | `4baf330dd1ad44641bdb8c90295f12a5798d9811357521066ff5e53b7b07c3faa` |
| `compile_f01_t256_w4x4_rmem1`，单operand slot | 282624 | 245/68 | 0 / 0 | 2445 | `b81f840994fad4668c1006542f575de3b34d860297e7588bd745bd7806e5d0d22` |

前者有spills，不计时；后者先做两个seed的严格canary/dynamic验收，再决定性能测试。资源预算允许每SIMD四个waves，但并不证明实际occupancy或收益。硬件依据仍为 Shader Guide §3.3.2.1（p32）的1024 VGPR/SIMD、16粒度分配，实际LDS320 KiB。


### 2026-10-17 两个 seed 的 w4x4 / M64-K256 验收与相邻对照

`f01_t256_w4x4_rmem1_seed0/17` 均通过 payload、ScaleA、GEMM2 有效行和完整 MoE 的逐字节对照、各三次 eager/graph canary 以及三组动态输入/routing replay。对应单 slot 的三轮正式结果如下。w4x4 的四个 compute wave groups 保持同一外部 t256 tile、256 行 expert alignment 和原 K128 累加顺序；增加 waves 没有减少实际有效计算。

| f01 相邻组 | GEMM1 三轮（μs） | 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `f01_t256_w4x4_rmem1_perf_a` | 149.428, 149.355, 151.438 | 149.428 | 128.357, 127.280, 127.729 | 351.50, 351.31, 354.35 |
| `f01_nt1_control_b` | 149.753, 152.890, 149.384 | 149.753 | 128.513, 129.299, 129.314 | 353.71, 357.56, 354.06 |
| `f01_m64k256_rmem1_perf_a` | 268.031, 268.305, 267.881 | 268.031 | 127.102, 128.464, 126.586 | 470.121, 469.467, 464.099 |
| `f01_nt1_control_c` | 149.555, 154.286, 148.985 | 149.555 | 129.825, 128.907, 127.918 | 354.937, 359.221, 352.267 |

上述十二轮均 rc0、ownership 合格、没有 live post owners。w4x4 相比相邻 nt1 仅低约0.22%，不能认定稳定收益。M64/K256 在 `f01_m64k256_rmem1_seed0/17` 也通过全部严格验收，但比相邻 nt1 中值慢79.22%，不作为生产选择。M64 的环境组合为 `N_TASKS=1 INTERNAL_M=64 PIPELINE_BUFFERS=3 RMEM_SLOTS=1 NCONST=1 UNSIGNED_INDICES=1 MIN_SEARCH=1`，每项加 `AITER_GFX1250_GEMM1_` 前缀。减少 LDS/每 wave register 资源没有自动带来性能收益；更多 WG 和重复 B/ScaleB 请求仍须由 trace 分析，不能直接断言唯一瓶颈。

CPU 对照 `f01_original_vs_a07_parent_cpu.log` 已逐字节确认：f01 使用未修改 HEAD 的 seed0/17 保存结果与 a07 patched 默认四任务的 seed0/17 参考 `.pt` 全部一致。因此原 HEAD 的精度参考已完成绑定；该结论来自已有原 HEAD GPU 输出加 CPU 字节对照，不能描述成 a07 上此前挂起的原 HEAD 验证成功。

f01 准备了独立 COMGR，`libamd_comgr.so.3.5.0` SHA256 为 `593588e40049f7d721d64c5dad77f03fa74e22c3fb63892d1cb14e8efc5e763b`，仅放在任务目录。尚未在 f01 运行 ATT，准备环境不表示 decoder 验收完成。

用户随后明确要求只在 a07-3 测试；f01 的所有已安排验证均已结束，后续不再在 f01 安排测试、编译或采集。已有 f01 数据保留为历史，单独记录，不能并入 a07 的目标性能。

### a07 恢复后的 mode2 与 N cluster 对照

a07 的 boot ID 保持 `967d0634-cfc3-4c4f-a711-dc6c7995ef5b`。之前原 HEAD runtime 加载等待的 `original_head_seed0_c0` 最终以 rc143 结束，验证没有完成；本轮没有向该进程发信号。新增 `a07_nt1_resume_seed0` 完成全部逐字节、eager/graph canary 和动态 replay，rc0。其 post 出现其他任务 owner，因此只使用精度，不使用任何耗时。

`run_dsr1_e65_gemm1_mode2.py` 仅在测试进程内将 G1 launcher 的 LLVM `amdgpu-expert-scheduling-mode` 设置为 True，然后原样运行原 benchmark 或 verifier。生产 compile hints、G2 和 benchmark 源码保持。`compile_a07_nt1_mode2` 的实际 ISA 进入 `HW_REG_WAVE_SCHED_MODE[1:0]=2`，插入138条 static `s_wait_alu`；LDS282624 B、VGPR458/SGPR68、zero spills/private、4319 instructions，ELF SHA256 `ac2952a73cf7f7caf68777c57cbfbe16bd5bd908ff40aaf84420d51ee7f95d27`。`a07_nt1_mode2_seed0/17` 均通过原有效输出逐字节、各三次 eager/graph canary、三组动态输入/routing replay。

| a07 组 | GEMM1 三轮（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | 有效性 |
|---|---|---|---|---|
| `a07_nt1_mode2_perf_a` | 120.040, 119.631, 119.249 | 93.6636, 94.4542, 94.3088 | 268.042, 269.134, 269.132 | 仅第三轮 ownership 合格；前两轮 post 外部 owner |
| `a07_nt1_control_e` | 118.290, 118.201, 118.142 | 94.8105, 93.1028, 93.3556 | 268.626, 264.863, 268.017 | 三轮 ownership 均不合格；不统计中值 |

这些受占用影响的组不能用于稳定收益结论。mode2 唯一合格样本119.249 μs，没有构成新的最佳三轮结果。Shader Guide §4.3.7.4.1（p92）说明 mode2 将部分 VALU/VMEM hazard 交由软件处理；已核对 actual mode2 与 wait，不由全局环境变量推断模式。

`INTERNAL_CLUSTER_N=1/2` 仅在 G1 内改变 cluster geometry 和相应 swizzle；每个有效 tile 的全部 K128 更新、数值、输入/输出格式、外部256行 expert alignment 保留。CPU 完整 N/M 覆盖审计增加两种 geometry；`a07_nt1_cn1_seed0/17`、`a07_nt1_cn2_seed0/17` 全部通过原有效字节、各三次 eager/graph canary 和三组动态 replay。cn1/cn2 compile 分别为4157/4178 instructions，资源均LDS282624 B、VGPR458/SGPR68、zero spills/private；ELF SHA256 分别为 `fabd14af09e7000484ca389b6fc70ccee3efc13d02b568c74051995144e2394fd`、`751ddbba38d070af6063eb1ca0ae590d960963e89c911a0b95996f70481e99d10`。

| a07 相邻组 | GEMM1 三轮（μs） | 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `a07_nt1_cn1_perf_a` | 118.490, 120.449, 119.736 | 119.736 | 94.3235, 93.3474, 94.4183 | 268.565, 269.895, 269.185 |
| `a07_nt1_cn2_perf_a` | 118.475, 119.650, 121.332 | 119.650 | 94.3952, 94.8146, 93.8426 | 268.626, 269.527, 270.810 |
| `a07_nt1_cn4_perf_a` | 118.423, 118.562, 119.194 | 118.562 | 93.9247, 94.0976, 94.5636 | 267.063, 268.354, 270.407 |

九轮全部 rc0、ownership 合格、没有 live post owners。cn1/cn2 分别比相邻 cn4 慢0.99%/0.92%，没有性能收益，继续保留 cn4。Shader Guide §2.3（p24）明确 cluster 在同一 SE、不同 WGP 中调度；减少 cluster 大小可能改变调度，但不保证收益。实测已否定该组合的收益假设。

### ScaleA TDM 写出与分离 A/ScaleA segment（2026-10-17）

`STAGED_QUANT_SCALE=1` 保留原 FP4 payload 写出，把有效 lane 的 `global_store_b8` ScaleA 改为 canonical M32 LDS staging，再由已有 `issue_quant_scale_output()` 发起 TDM store。`SEPARATED_A_SEGMENTS=1 PIPELINE_BUFFERS=3` 则保持原输出，把同一个 K256 stage 的 wave_m0 A/ScaleA 放在前部、wave_m1 放在 stage+64 KiB；每 stage 96 KiB、共294912 B，consumer 与 producer 使用相同 stage，不循环错移。两项都保留全部有效 routes、原 K128 累加顺序、输入/输出 lifetime fences 与数值格式。

分离布局地址为 A0=0、ScaleA0=16640、B=17664、ScaleB=50944、A1=65536、ScaleA1=82176 B；使用83200 B。Shader Guide 与 LDS §7.3 仅支持不同 segment 可供两 port 访问的实验依据，不能单凭布局断言 bank conflict 或性能收益。

| 编译目录 | LDS（B） | VGPR/SGPR | static instructions | ELF SHA256 |
|---|---:|---|---:|---|
| `compile_a07_nt1_qscale` | 282624 | 458/68 | 4082 | `d503d96e30d1746baeaea5ba0b7bc96d1ea14de6d14e832e83eba8170a1b3a11` |
| `compile_a07_nt1_aseg96` | 294912 | 453/68 | 4020 | `1e14e2396e30abe6794f70fb10c5ba8f92a37049d66da190e9cf74f09aea678b` |

两份产物均 zero spills/private。`a07_nt1_qscale_seed0/17`、`a07_nt1_aseg96_seed0/17` 四组 rc0、ownership 合格；全部有效 payload、ScaleA、GEMM2、MoE 与原 HEAD 参考逐字节一致，各三次 eager/graph canary 及三组相同指针 random FP4/E8M0、counts/prefix sum 更新 replay 通过。

以下第一组计时受到间歇外部 owner 影响，完整保留但不混算三轮中值，也不据此宣称稳定收益：

| 组 | GEMM1 三轮（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | 空闲验收 |
|---|---|---|---|---|
| `a07_nt1_qscale_perf_a` | 117.727, 116.432, 118.984 | 93.8553, 93.6908, 94.0725 | 267.884, 265.944, 268.246 | 仅第一轮合格 |
| `a07_nt1_aseg96_perf_a` | 120.773, 119.907, 120.811 | 94.7451, 93.4737, 93.6655 | 271.373, 268.107, 270.277 | 三轮均不合格 |
| `a07_nt1_control_f` | 119.388, 121.515, 120.693 | 93.3155, 94.1591, 94.5215 | 266.311, 272.577, 272.767 | 后两轮合格 |

复现原 benchmark 的候选环境：`AITER_GFX1250_GEMM1_N_TASKS=1 AITER_GFX1250_GEMM1_STAGED_QUANT_SCALE=1`，或 `AITER_GFX1250_GEMM1_N_TASKS=1 AITER_GFX1250_GEMM1_PIPELINE_BUFFERS=3 AITER_GFX1250_GEMM1_SEPARATED_A_SEGMENTS=1`。完整主机命令、源码快照、日志和全部前后 idle 状态保存在同名 artifact 目录。

测试控制脚本新增失败返回码传播：任何 rc 非零（包括120秒超时）使串行 `&&` 队列停止，防止失败候选后继续性能测试。120秒 TERM、5秒后 SIGKILL 只作用于本测试进程组。

### packed BF16 amax、TDM owner 数量与 carry prefetch 分布

`PACKED_QUANT_AMAX=1` 复用原文件已有 `_amax_inputs_from_bf16x16`，在 BF16 RNE 之后求 packed maximumNumber，再由未修改的 native helper 做 FP32 clamp、peer reduction 和 E8M0 rounding。原 t192 分支及 GEMM2 AST 保持。`a07_nt1_pkamax_seed0/17` 全部严格字节、canary、动态 replay 通过。

`INTERNAL_TDM_WAVES=1/4` 仅在 t256 nt1 调整每个 tensor 的 owner 数量。1-wave 保留4个 owner、其余wave计算；4-wave 的每wave发两个完整 tensor 的分片。每stage全部有效源字节、目的地址以及随实际请求数计算的 TENSORcnt 门槛保持。`a07_nt1_tdmw1_seed0/17`、`a07_nt1_tdmw4_seed0/17` 均严格通过。第一次编译漏放宽仅接受2-wave的候选静态条件，compile rc1、无dispatch；修正候选条件后使用 `_v2` 独立产物，失败日志保留。

`CARRY_MMA_COVER=8/14` 只改变 compiler scheduling hints，把下一 K128 的 DS reads 分散到更多当前 WMMA groups 中；原 hardware READY/reader/WG fence 和每个 accumulator 的 K128 更新顺序不变。原 t256 closing cover 为12个 physical WMMA（旧“28”注释来自非FP4逻辑计数）。两项先核对实际 ISA/resources，再严格精度，最后原 benchmark。

| 编译目录 | LDS（B） | VGPR/SGPR | static instructions | ELF SHA256 |
|---|---:|---|---:|---|
| `compile_a07_nt1_pkamax` | 282624 | 458/68 | 4082 | `48f535a9f5742f1763067b3b5bc9df618559b88c414e93daa183c9cc879569909` |
| `compile_a07_nt1_tdmw1_v2` | 282624 | 458/68 | 4251 | `f2d9824a9ad7a422f902ac0b9f3ca871de2c47fb523d9f0f65583928de8932e6` |
| `compile_a07_nt1_tdmw4_v2` | 282624 | 458/68 | 3794 | `2e77154e4c665e9ef4c0f926a3e4c7f5e892f5d998fd3d8fafd7a4da7c2492d6` |
| `compile_a07_nt1_fc8` | 282624 | 450/68 | 4223 | `7bd595863d927a671f4fcec46a6b7b6ff4630682eb451e423303cb6d82308b0d` |
| `compile_a07_nt1_fc14` | 282624 | 466/68 | 4160 | `e83c44abb873b6b7c80ee86eae249d1bbb6c32e614761d650540179d852b550e` |

全部 zero spills/private。编译资源变化是诊断证据，不作为性能收益。

用户新增要求：自己启动的测试超过2分钟未结束须及时 kill。host 控制脚本已将每轮容器测试放入 GNU `timeout --signal=TERM --kill-after=5s 120s` 独立进程组；在120秒发 TERM，未退出时5秒后强制终止该组。超时 rc124/137 明确记录并排除性能；空闲等待不算 GPU 测试时间。不会向无关任务发信号。ATT 每个 profiled pass 同样应设置 `AITER_ATT_TIMEOUT_SECONDS=120`。

### 2026-10-17 a07 相邻复测与 M16 padding 候选

恢复后确认 boot ID 不变，原九组串行队列已经提交，没有重复 dispatch。以下保留全部轮次；`ownership_qualified=false` 的轮次不用于性能结论。每轮仍使用原完整 MoE eager benchmark、warmup5、iters20、count19。

| a07 组 | GEMM1 三轮（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | 验收与结论 |
|---|---|---|---|---|
| `a07_nt1_control_g` | 119.595, 119.536, 117.872 | 94.9895, 94.4962, 94.6270 | 270.736, 269.442, 268.474 | 前两轮有效，第三轮外部 owner；不统计完整三轮中值 |
| `a07_nt1_pkamax_perf_a` | 120.236, 116.704, 117.972 | 93.4990, 94.5490, 93.6297 | 268.600, 266.509, 266.431 | 前两轮有效，第三轮外部 owner；未证明稳定收益 |
| `a07_nt1_qscale_perf_b` | 118.513, 117.212, 116.805 | 93.9101, 94.7387, 94.3677 | 266.587, 267.902, 267.296 | 三轮有效，中值117.212；需反向对照确认收益 |
| `a07_nt1_aseg96_perf_b` | 120.324, 119.281, 120.326 | 93.3872, 93.1046, 93.3683 | 269.433, 268.113, 269.792 | 三轮有效，中值120.324，无收益 |
| `a07_nt1_tdmw1_perf_a` | 118.110, 117.012, 118.724 | 94.4415, 94.2475, 93.5052 | 268.487, 267.113, 267.736 | 三轮有效，中值118.110，尚未证明超过量化写出候选 |
| `a07_nt1_tdmw4_perf_a` | 118.891, 121.191, 120.213 | 93.3513, 92.8158, 94.0176 | 267.576, 269.014, 269.855 | 第一轮外部 owner；后两轮无收益 |
| `a07_nt1_fc8_perf_a` | 119.150, 119.932, 119.445 | 93.8363, 92.4491, 92.8706 | 268.265, 267.812, 269.331 | 三轮有效，中值119.445，无收益 |
| `a07_nt1_fc14_perf_a` | 118.543, 120.055, 119.433 | 93.4399, 94.5617, 93.6507 | 267.173, 268.679, 269.151 | 三轮有效，中值119.433，无收益 |
| `a07_nt1_control_h` | 121.613, 119.626, 119.173 | 94.5594, 94.1484, 93.5178 | 269.910, 270.140, 267.086 | 三轮有效，中值119.626 |

ScaleA TDM 写出的117.212 μs比历史nt1最佳118.541 μs低1.12%，比队列末尾nt1对照119.626 μs低2.02%。当前数据仍不足以保证跨顺序稳定收益；下一轮安排反向相邻对照，不把时钟或机器漂移归因于代码。90 μs目标尚未达到。

新增默认关闭的 `AITER_GFX1250_GEMM1_ROW_TAIL_PREDICATION=1` 只在同一 t256 WG 中跳过整个无有效行的 M16 WMMA group。每个 group 的 wave-uniform 条件为 `wave_m * warp_tile_m + wm * 16 < mn_oob`；任何部分有效 group 都保留所有 lanes、全部56次 K128 更新和原累加顺序，13824条真实 routes 不变。全部输入 TDM、DS operand loads、READY/reader/WG fences、SiLU、BF16 rounding、FP4/E8M0与输出布局保留。该候选不能以 EXEC lane mask 实现：Shader Guide §4.6.12（p144）明确 WMMA 忽略 EXEC mask并强制所有bits为1。只有完全没有有效行的group才允许通过wave-uniform分支跳过整条指令。

`audit_dsr1_e65_mapping.py`增加0..256有效行、m_warp2/4的8224个group覆盖检查，证明每条有效行被且仅被覆盖一次，3872个完全padding group可以跳过。它是CPU覆盖证据，尚不等于GPU精度或性能验收。两端生产源GEMM2 AST、公共amax helper AST、原benchmark、grouped_moe、quant_utils和gemm_common保持。下一步先检查实际ISA中的分支及zero spills，再做seed0/17严格逐字节、eager/graph canary、动态输入/routing replay；失败候选不计时。

第一版 rowtail 的 `compile_a07_nt1_rowtail` 为495 VGPR/70 SGPR、LDS282624 B、zero spills/private、5027 instructions，ELF SHA256 `d64133940dd9cfa6b68d4840f1644209c68825a2d685571f71c57fbe2ceed7873`。`a07_nt1_rowtail_seed0` 首次 eager canary 的 payload 有19610 B差异，ScaleA零差异，rc1，未计时且串行队列停止。实际 ISA 包含跳过WMMA的scalar branches；编译成功并不代表数值正确。

rowtail 独立修订改为显式drain carry LDS地址，使用新 `_rowtail_explicit` symbol，避免公共 `make_sgpr_opaque()` 的空 `"=s,s"` asm 在新寄存器分配下不能保证输入输出identity的问题。公共helper及既有nt1 lowering保持。`compile_a07_nt1_rowtail_explicit` 为511 VGPR/70 SGPR、LDS282624 B、zero spills/private、5152 instructions，ELF SHA256 `47aa5908bd60a497f5a96328ca6b2759b73bfd0409ed69b4fa75e49f0e9c82583`。seed0/17全部有效payload、ScaleA、GEMM2、MoE与原HEAD绑定参考逐字节一致，各3次eager/graph canary和3组dynamic replay全部通过。只能确认显式地址消除了这些测试中的差异，不能据此唯一定位原错误根因。

| 相邻复测 | GEMM1 三轮（μs） | 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `a07_nt1_rowtail_perf_a` | 118.431, 117.223, 117.423 | 117.423 | 93.6065, 94.6355, 94.6714 | 268.379, 268.350, 268.413 |
| `a07_nt1_control_i` | 118.827, 118.322, 121.043 | 118.827 | 95.1098, 94.6441, 95.1731 | 270.489, 269.095, 272.366 |
| `a07_nt1_qscale_perf_c` | 118.300, 116.158, 118.884 | 118.300 | 94.9622, 94.4266, 93.4609 | 268.067, 266.460, 268.763 |

九轮全部rc0、ownership合格。rowtail相邻中值低1.18%；qscale反向对照低0.44%，幅度接近噪声，不能直接承诺稳定收益。候选组合仍须独立编译和精度验收，未默认启用。

新增 `COMPACT_N=128 PIPELINE_BUFFERS=3 N_TASKS=1` 候选，将内部N256/w2x4改成两个N128/w2x2，输出MX block映射仍canonical，所有N4096、有效routes、原56次K128更新保持。该候选希望减少每WG accumulator/LDS资源并增加可并行WG数量，不依据资源预算宣称occupancy或收益。静态覆盖检查49152个row/MX pair、786432个payload bytes、49152个ScaleA bytes全部通过；尚待实际ISA、严格GPU字节与性能验收。

### compact N 与 rowtail/ScaleA 组合验收

`compile_a07_nt1_n128` 为LDS159744 B、VGPR458/SGPR68、3520 instructions、zero spills/private；ELF SHA256 `a1356fad0ad7c5c208e335481f740025d1140c99cc282ac48aa1ac44288795c3b`。`compile_a07_nt1_rowtail_qscale` 为LDS282624 B、VGPR511/SGPR70、5052 instructions、zero spills/private；ELF SHA256 `7cbd2cb028cc6a5263c1fc90fabe34710158bf19f31095015d52e07bf3a74a9f`。两项各在seed0/17通过全部有效payload、ScaleA、GEMM2、MoE逐字节对照，以及各三次eager/graph canary与三组同地址动态输入/routing replay。

| a07 相邻组 | GEMM1 三轮（μs） | 完整三轮中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | 验收 |
|---|---|---:|---|---|---|
| `a07_nt1_n128_perf_a` | 123.793, 122.593, 123.312 | 不统计 | 92.8116, 92.9697, 95.0298 | 272.233, 270.994, 273.344 | 第三轮存在外部owner；前两轮无收益 |
| `a07_nt1_rowtail_qscale_perf_a` | 117.761, 113.270, 115.768 | **115.768** | 94.4183, 94.7829, 94.6733 | 268.799, 264.504, 267.287 | 三轮rc0、ownership合格 |
| `a07_nt1_control_j` | 117.927, 119.146, 118.187 | 不统计 | 94.2348, 93.7793, 94.4583 | 268.196, 269.208, 268.784 | 第三轮存在外部owner |

115.768 μs是截至此组的最低有效完整三轮中值，比原HEAD的153.405 μs低24.5349%，距离90 μs仍差25.768 μs。组合的稳定收益还需反向相邻复测；113.270 μs只是单轮，不能作为稳定结果。所有正式轮次继续使用原完整MoE benchmark、random non-balanced routing、warmup5、iters20、count19。生产默认仍未启用组合。

### XDL arbitration policy 候选

新增默认关闭的 `AITER_GFX1250_GEMM1_XDL_NO_STALL=1`，仅为单N-task的t256 fused GEMM1设置 `HW_REG_WAVE_SCHED_MODE` bit2。Shader Guide §4.3.7.4.2（p93）与CDNA5 ISA §5.7.2.1（p56）说明 `DISABLE_XDL_ARB_STALL` 允许连续issue多条WMMA，但可能减少其他wave的co-execution机会，主要可能有利于单wave/SIMD。现有nt1实测两个wave/SIMD，所以这里只把它作为需验证的调度实验，不提前宣称收益。

两份候选的ISA均包含 `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1`。输入/输出lifetime fences、有效计算和K128累加顺序不变；开关本身不改变carry地址策略。关闭开关重新编译的nt1 ELF仍为 `10df3ead8f2f3c88288d05e748d8fd6f38df4905f30b7c448ba923fd24d2c92f`，与已验收的原nt1完全一致。

| 编译目录 | LDS（B） | VGPR/SGPR | instructions | ELF SHA256 |
|---|---:|---|---:|---|
| `compile_a07_nt1_xdl` | 282624 | 458/68 | 4196 | `4203f92d4fb25d0a3c8f6fb45e7f1649f133e2da002ba80a9935ed9af7553026e` |
| `compile_a07_nt1_rowtail_qscale_xdl` | 282624 | 511/70 | 5059 | `895955fd8d8a301a8b992566d3c8f91a913ae932f60f53102b5d53a0a6ee7f59` |
| `compile_a07_nt1_xdl_control` | 282624 | 458/68 | 4180 | `10df3ead8f2f3c88288d05e748d8fd6f38df4905f30b7c448ba923fd24d2c92f` |

三份产物均zero spills/private。当前persistent源码SHA256 `6d92536f2e1b93ce8b8a80162f8a39585fa4188e09ecb975effcb4727d81abde`；grouped wrapper为 `2f1b83975c3142106c4c2da295ebb31aca816d30c3291eaea8e2c49615a5ea2b`；compile脚本为 `6f68c822a330f9518e70954b1658211dbb02ab5dd5ea718e564edfaa4ea73152`。两端SHA一致，公共源/GEMM2审计与CPU mapping审计全部通过。GPU精度与性能验收完成后再补结果。

seed0/17的nt1/XDL与rowtail/ScaleA/XDL四项诊断均rc0；全部有效payload、ScaleA、GEMM2和完整MoE逐字节一致，各三次eager/graph canary、三组同地址动态输入/routing replay通过。原MoE精度仍分别为 `logits_diff=3.7793e-06 rel_l2=2.7493e-03`、`3.7788e-06/2.7491e-03`。

| a07 相邻组 | GEMM1 三轮（μs） | 完整三轮中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | 有效性 |
|---|---|---:|---|---|---|
| `a07_nt1_control_k` | 117.605, 116.869, 118.929 | 不统计 | 93.4715, 93.9691, 94.2515 | 266.258, 266.083, 268.656 | 三轮均外部owner，排除 |
| `a07_nt1_rowtail_qscale_perf_b` | 115.682, 113.917, 119.072 | 不统计 | 94.5889, 94.6100, 94.3528 | 266.399, 262.908, 268.967 | 三轮均外部owner，排除 |
| `a07_nt1_xdl_perf_a` | 121.143, 121.094, 120.205 | 不统计 | 93.8534, 93.1472, 94.3763 | 270.037, 270.865, 270.910 | 第一轮排除；后两轮合格，无新收益 |
| `a07_nt1_rowtail_qscale_xdl_perf_a` | 114.784, 117.316, 116.192 | 116.192 | 94.4351, 94.3953, 94.5976 | 265.364, 267.500, 266.813 | 三轮合格，未超过不带XDL组合 |
| `a07_nt1_control_l` | 120.105, 120.571, 117.628 | 120.105 | 94.1444, 93.7818, 94.6695 | 269.864, 270.081, 267.685 | 三轮合格 |

当前数据没有证明XDL override在已有组合上产生稳定附加收益，暂不作为默认策略。尤其不能从被外部owner污染的前两组取有利单轮作为对照。

### rowtail/ScaleA 组合的反向相邻复测

| 固定顺序 | GEMM1 三轮（μs） | 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `a07_nt1_rowtail_qscale_perf_c` | 115.110, 117.390, 116.264 | 116.264 | 93.6530, 93.4971, 94.4290 | 265.293, 266.521, 266.467 |
| `a07_nt1_control_m` | 120.923, 120.346, 119.861 | 120.346 | 94.3658, 93.8217, 92.9614 | 271.522, 269.805, 268.831 |
| `a07_nt1_rowtail_qscale_perf_d` | 113.915, 117.775, 114.988 | **114.988** | 94.5048, 95.3062, 94.3172 | 264.221, 269.071, 265.001 |

九轮全部rc0、ownership合格、无live post owners。前后两组相对于中间对照分别降低3.39%/4.45%，支持组合收益。最低有效完整三轮中值更新为114.988 μs，比原HEAD低25.043%，距离90 μs仍差24.988 μs；没有把单轮113.915 μs当作稳定结果。不同批次中值仍有漂移，最终默认路径需要在收敛后的源码上复测。

### 新组合 ATT 与交错 M16 候选

在a07/dsr1_ep4使用原 `get_att_b_rocm_sdk.sh` 采集rowtail/ScaleA组合，SIMD0、四SE、random原workload、诊断iters2、每个测试120秒限制。raw、decoded、目标symbol和PC映射验收通过，无缺失PC。实际code object与 `compile_a07_nt1_rowtail_qscale/kernel.elf` 完整逐字节相同，SHA为 `7cbd2cb028cc6a5263c1fc90fabe34710158bf19f31095015d52e07bf3a74a9f`。新trace保存在 `e65_g1_rowtail_qscale_20261017/`，archive为 `a07_e65_rowtail_qscale_att.tar.gz`，SHA256 `a8e4ef94a825a3dda88ab9de7f3513ae21a726857bb4a5e38b0f382b697205d8`。

采集控制记录 `a07_nt1_rowtail_qscale_att/result_1.json` 为 `returncode=0`、`timed_out=false`，但采集后出现外部 live owner `2355791`，`ownership_qualified=false`。raw/decoder/ELF/PC 验收成立；下面的动态指令计数可用于确认分支执行，周期数据只能用于提出诊断方向，不能当作空闲条件下的可靠绝对周期对照、正式性能数据或与旧 trace 比较得出的收益。

采样56waves，其中20个跳过；其余的动态WMMA数为336、448、560、672、896，说明wave-uniform padding predication实际生效。896组20waves的compute span中值24949.5 cycles、compute WG waits累计中值6676.5 cycles、DS waits1090 cycles；336组6waves的compute span仍为24265 cycles。steady WG wait的432条记录中值262.5 cycles。少做padding WMMA的wave并没有按比例缩短compute span，证据支持继续检查WG内wave进度及同步限制，尚不能唯一定位硬件FIFO或cache瓶颈。这些是采样wave cycles，不是正式kernel耗时，也不能直接与旧trace相减预测μs。

新增默认关闭的 `AITER_GFX1250_GEMM1_INTERLEAVED_M_ROWS=1`，需要 `N_TASKS=1 ROW_TAIL_PREDICATION=1`。同一t256 WG的两个M wave groups从连续M128改成交错M16：logical group=`wm*2+wave_m`，每侧仍有八个M16位置，partial expert的有效group数量相差最多一个。全部真实rows、56次K128累加、SiLU、BF16 rounding、FP4/E8M0与canonical输出布局保持；不重排routing。A LDS reads步长改成两个M16，ScaleA用本M16的16-lane scales复制到opsel0，payload TDM stride改成两个M16，ScaleA staging按原canonical M32地址写入。必要同步保留。

CPU审计通过49152个row/MX pair、786432个payload输出字节、49152个scale输出字节、1032192个input fragment字节和4112个row-group边界。源码审计确认GEMM2、公共amax、原benchmark、grouped_moe、quant_utils、gemm_common保持。实际ISA与device精度/性能仍待验收，不把映射审计当作优化收益。

### 交错 M16 与单 register slot 的实际结果

| 编译目录 | VGPR/SGPR | VGPR spills / private（B） | static instructions | ELF SHA256 |
|---|---|---|---:|---|
| `compile_a07_nt1_interleaved_m`，双 slot | 512/70 | 16 / 40 | 5166 | `3d8b230aec8a3cf80e8db7888e11180a280244decef062832601932ab58efa11` |
| `compile_a07_nt1_interleaved_m_rmem1` | 372/70 | 0 / 0 | 5144 | `fc1c21fdf4f450e895201d66fef5c758d5d467c6df81358d66ebb13597eca841` |
| `compile_a07_nt1_rowtail_qscale_rmem1`，连续 M128 | 368/70 | 0 / 0 | 5045 | `5d6397123a66ac93b75ccc3006ad7623fdc8897ebc430074f92a6808ca1f5631` |

三者 LDS 均为282624 B。双 slot 交错 M16 有 spills，未 dispatch 或计时。两个单 slot 方案在 seed0/17 的四项诊断全部 rc0、ownership 合格，原 HEAD 绑定参考的有效 payload/ScaleA/GEMM2/MoE 逐字节一致，各三次 eager/graph canary 与三组动态输入/routing replay 通过，每项28条零差异报告。原 MoE 精度分别为 `3.7793e-06/2.7493e-03`、`3.7788e-06/2.7491e-03`。

| 串行顺序 | GEMM1 三轮（μs） | 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `a07_nt1_rowtail_qscale_rmem1_perf_a` | 130.538, 126.911, 130.822 | 130.538 | 92.3542, 91.2221, 92.4322 | 278.348, 274.041, 278.706 |
| `a07_nt1_interleaved_m_rmem1_perf_a` | 131.723, 129.876, 129.515 | 129.876 | 93.5751, 92.9508, 91.2136 | 281.272, 278.867, 277.317 |
| `a07_nt1_rowtail_qscale_control_e`，原双 slot 组合 | 116.030, 116.182, 114.568 | 116.030 | 94.3087, 94.5658, 94.4373 | 266.556, 266.400, 263.870 |
| `a07_nt1_interleaved_m_rmem1_perf_b` | 132.853, 130.888, 131.999 | 131.999 | 92.7189, 92.1326, 92.2762 | 280.682, 279.115, 279.851 |

十二轮全部 rc0、ownership 合格、无 live post owners。相对于中间双 slot 对照，连续行单 slot 慢12.50%，交错 M16 的两组慢11.93%/13.76%。不保留这些方案为生产选择，资源减少并未变成实际性能收益。M16 双 slot 未通过资源门槛，因此这里没有证明在保留双 slot overlap 后的交错分配是否有收益。

下一候选按完整 M32 group 交错：`logical_m16=(wm//2)*4+wave_m*2+wm%2`，沿用原 ScaleA 的 M32/opsel 结构和两个 register slots；输出 payload 使用三维 TDM descriptor 保持 canonical 地址。全部有效行、56次 K128 更新、数值和必要同步保持。该方案仍须独立地址、实际 ISA、精度及正式性能验收。

### M32 交错与 A operand 尾部 predication 的淘汰

| M32 编译目录 | 输出方案 | static instructions | ELF SHA256 |
|---|---|---:|---|
| `compile_a07_nt1_interleaved_m32_v2` | 三维 payload TDM | 5093 | `afea2388f5c987c76cce08dad7a262dcfcc53205fd3708f0e8569f790e43fd20` |
| `compile_a07_nt1_interleaved_m32_canonical` | 原二维 TDM owner、canonical LDS staging | 5113 | `b775b0c8a7fc4e4126bfaf5db038c76eed4421c97a7915b8e31d762ee9bccac9` |
| `compile_a07_nt1_interleaved_m32_direct` | 直接写出 ScaleA | 5217 | `a745fa380193b6ca2c8c39dd526a3c80a3ce92866657a0c4e19d51ecf7e84e64` |

三份产物均 LDS282624 B、VGPR511/SGPR70、zero spills/private，但 seed0 首次 eager canary 都失败。三维 TDM 方案有效 payload 零差异、ScaleA 差17258 B；canonical staging 方案 payload 零差异、ScaleA 差15793 B；direct 方案 payload 差32321 B、ScaleA 差426248 B。各失败队列停止，没有执行 seed17，也没有对失败候选计时。这些结果不能记作修复或性能收益；恢复原二维输出后仍失败，三维 descriptor 不能单独解释问题，根因尚未定位。

`ROW_TAIL_A_LOADS=1` 只尝试对完全无有效行的 M16 group 跳过 A operand DS loads，所有有效 group、B/scales 和同步保持。`compile_a07_nt1_rowtail_qscale_aload` 为 LDS282624 B、VGPR484/SGPR70、20 VGPR spills、private44 B、5811 instructions，ELF SHA256 `883ec66531ee94b39204fd6c8fa2e48369388949c8e09670d75ef26c347397ce`。资源不合格，没有 GPU dispatch 或计时，撤出生产候选。

### packed BF16 amax 加入 rowtail/ScaleA 组合

`compile_a07_nt1_rowtail_qscale_pkamax` 为 LDS282624 B、VGPR511/SGPR70、zero spills/private、4948 instructions；完整 ELF SHA256 `f65b78dcb78d3c50abd6dd54db8d6aeea8417bae298fd657b0f9c28e8e17bdf1`。使用原 `_amax_inputs_from_bf16x16`，没有改它的 AST、native scale helper、GEMM2 或原 benchmark。

`a07_nt1_rowtail_qscale_pkamax_seed0/17` 都 rc0、未超时、ownership 合格、无 live post owners。两个 seed 各三次 eager/graph canary 和三组同地址动态输入/routing replay 通过；有效 payload14155776 B、ScaleA884736 B、GEMM2198180864 B、MoE22020096 B 与原 HEAD 绑定参考逐字节一致。原 MoE 精度分别 `3.7793e-06/2.7493e-03` 和 `3.7788e-06/2.7491e-03`。

| 相邻固定顺序 | GEMM1 三轮（μs） | 三轮中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `a07_nt1_rowtail_qscale_pkamax_perf_a` | 112.624, 118.870, 113.377 | 113.377 | 94.0643, 94.4183, 93.2059 | 262.084, 268.887, 263.013 |
| `a07_nt1_rowtail_qscale_control_f` | 117.626, 115.985, 116.609 | 116.609 | 94.1377, 94.3317, 93.5941 | 267.572, 266.069, 266.164 |
| `a07_nt1_rowtail_qscale_pkamax_perf_b` | 116.327, 116.525, 114.769 | 116.327 | 95.0951, 94.7598, 94.5722 | 267.354, 267.085, 265.640 |

九轮都 rc0、ownership 合格，使用用户原完整 MoE benchmark，warmup5、iters20、count19。候选前后两组相对于中间对照分别低2.77%/0.24%，幅度不一致，尚不足以认定 packed amax 的稳定附加收益。113.377 μs 是这一候选的一组有效三轮中值，不能用单轮112.624 μs 替代稳定结果，也不能仅按最低批次选择默认路径。

复现候选在原命令前增加 `AITER_GFX1250_GEMM1_N_TASKS=1 AITER_GFX1250_GEMM1_ROW_TAIL_PREDICATION=1 AITER_GFX1250_GEMM1_STAGED_QUANT_SCALE=1 AITER_GFX1250_GEMM1_PACKED_QUANT_AMAX=1`；中间对照去掉最后一项。完整 command、sources、pre/post/settled ownership 和 run logs 保存于同名 artifact 目录。

### 独立 M128 wave-group 候选

新增默认关闭的 `SPLIT_M_WAVES=1`，把原 t256/w2x4 中两个连续 M128 wave groups 拆成两个 M128/w1x4 WG。每个 wave 的 M128/N64 fragment 形状、全部有效行、56次 K128 更新及原累加顺序保持。expert lookup 使用原256行父 tile 的起点，完全 padding 的内部 M128 subtile 跳过；外部路由、expert alignment、SiLU、BF16 rounding、FP4/E8M0 与 canonical 输出布局保持。

该实验针对 partial expert 的 WG 内同步限制，也增加了重复 B/ScaleB 请求；资源减少或 WG 拆分不能直接证明收益。Shader Guide §4.3.7.4.2（p93）说明 XDL stall 的 multi-wave tradeoff，§5.7.6（p235）说明 LDS 的 SIMD-pair 共享和较长 dependent latency；这些是实验依据，不能唯一确定当前瓶颈。实际 ISA、严格精度与原完整 benchmark 尚待验收。

后续编译 b3 为 LDS159744 B、VGPR505/SGPR68、3962 instructions、ELF SHA256 `1f4440e70980f3e9ca77de9bd0c14fdc49aaa8b2a9f126a7d723d41a99ebb007`；b4 为 LDS212992 B、VGPR505/SGPR68、4015 instructions、SHA256 `b751849d78f78b37ccebe0384fe46337568db9114042a31b15f899b9ceb61d229`。两项均 zero spills/private。b3/b4 各 seed0/17 全部有效字节、eager/graph canary、dynamic replay 与原 HEAD 一致。前三项 post 存在外部 owner，因此只使用正确性结果；b4/seed17 ownership 合格。

| 相邻顺序 | GEMM1 三轮（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | ownership 验收 |
|---|---|---|---|---|
| `a07_split_mwave_b3_perf_a` | 131.031, 130.215, 130.217 | 92.2739, 91.6394, 92.5459 | 279.083, 274.235, 277.380 | 仅第三轮合格 |
| `a07_nt1_rowtail_qscale_control_g` | 116.339, 116.654, 120.179 | 93.8784, 93.8532, 94.4606 | 266.688, 265.636, 270.977 | 仅前两轮合格 |
| `a07_split_mwave_b4_perf_a` | 145.014, 146.718, 147.228 | 92.6579, 92.7863, 92.4806 | 291.633, 295.116, 293.079 | 三轮都不合格 |

全部 rc0，没有超时，但不能统计任一组三轮有效中值。b3 唯一有效样本130.217 μs 明显慢于相邻有效对照116.339/116.654 μs；b4 没有有效性能证据。该细化方案不作为生产选择。此前历史记录已测过基础 M128/w1x4/b3，约133.393 μs；本次增加 rowtail、canonical ScaleA TDM 与双 seed 动态验收仍没有收益，因此结束这个方向。

### 分两段 A operand prefetch

新增默认关闭 `HALF_A_PREFETCH=1`。双 register slots 保持，下一 K128 的 B/scales 和 FRONT A rows 提前读取，当前 FRONT WMMA 完成后才读取下一 BACK A rows。有效 fragment 地址、每个 accumulator 的 K128 累加顺序、TDM input ring 和所有 lifetime fences 保持。目的是缩短 A operand 的 live range，验证是否能保留 overlap 同时减少 register 压力；不是减少实际运算。

`compile_a07_w4_halfapre` 的 w4x4 仍为 VGPR256/SGPR68、10 VGPR spills、private36 B、2683 instructions，ELF SHA256 `0f273ce6088d78bf75f99a0a6ea9b48e8a6f641ff0f5445fe0520303a415d9c5`，资源失败，不 dispatch。`compile_a07_w2_rowtail_qscale_halfapre` 为 LDS282624 B、VGPR470/SGPR70、zero spills/private、5099 instructions，ELF SHA256 `c7e085ca0081459a888c2dc151c482b7838136b50b4c19abd3fea247ce81f08e`，进入严格精度验收。

关闭新开关重编译的 rowtail/ScaleA 控制产物完整 ELF SHA256 仍为 `7cbd2cb028cc6a5263c1fc90fabe34710158bf19f31095015d52e07bf3a74a9f`，证明新实验没有改变已验收组合的实际 code object。源码审计确认 GEMM2 AST、公共 amax AST、原 benchmark、grouped_moe、quant_utils、gemm_common 保持。

`a07_w2_rowtail_qscale_halfapre_seed0`、`a07_w2_rowtail_qscale_halfapre_seed17` 均 rc0、未超时、ownership 合格、没有 live post owners。每个 seed 的三次 eager/graph canary 和三组动态输入/routing replay 均通过；全部有效 payload 14155776 B、ScaleA 884736 B、GEMM2 198180864 B、MoE 22020096 B 与原 HEAD 绑定参考逐字节一致。原 MoE 精度 seed0 为 `3.7793e-06/2.7493e-03`，seed17 为 `3.7788e-06/2.7491e-03`。性能验收按 halfA 三轮、原组合三轮、halfA 三轮的固定相邻顺序进行；尚不根据 VGPR 减少宣称性能收益。

| 相邻组 | GEMM1 三轮（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | ownership 验收 |
|---|---|---|---|---|
| `a07_w2_rowtail_qscale_halfapre_perf_a` | 115.102, 114.503, 114.936 | 94.1085, 94.6649, 94.6038 | 264.781, 264.664, 265.820 | 仅第三轮合格 |
| `a07_nt1_rowtail_qscale_control_h` | 115.355, 117.063, 114.398 | 95.2532, 93.9270, 93.8912 | 266.022, 266.547, 261.657 | 三轮均不合格 |
| `a07_w2_rowtail_qscale_halfapre_perf_b` | 114.686, 114.659, 111.492 | 93.9249, 94.2284, 94.9389 | 264.329, 265.640, 261.059 | 三轮均不合格 |

九轮均 rc0、未超时，但任一组三轮完整性能中值都无法验收。失败轮次的结束或六秒 settled 检查发现外部占用，全部保留并排除；111.492 μs 不作为有效最佳。唯一合格 halfA 样本114.936 μs尚不能证明相对于原组合的稳定收益，因此暂不默认采用。

### 尾部 predication 的整段快路径

新增默认关闭的 `ROW_TAIL_PREDICATION=2`，在每个 FRONT/BACK M16 范围先判断最后一个 group 是否有效；如果全部有效，则顺序执行原 WMMA 清单；如果第一个 group 也无效，则整段跳过；其余情况保留逐个 group 的判断。保留每条有效行的全部56次 K128 更新、原累加顺序、同一 fragment 和 operand reuse 规则，不以 EXEC lane mask 跳过部分有效 group。输入/输出地址、routing、同步和数值格式保持。

CPU 覆盖审计新增3084个整段分支边界检查，涵盖有效行0..256、w2x4/w4x4，全部与原 M16 group 判定一致。实际 ISA、spills、严格 GPU 字节和正式性能仍须独立验收。

两版完整快路径资源均失败：`compile_a07_nt1_rowranges_qscale` 为 VGPR512/SGPR68、private664 B、2309 VGPR spills、11192 instructions、ELF SHA256 `fe0d9e15be967a56a0da5d69708ce59c8136085aecc2ff6541d431e861bdca40`；加 halfA 的 `compile_a07_nt1_rowranges_qscale_halfapre` 为 VGPR512/SGPR68、private600 B、2773 VGPR spills、10976 instructions、SHA256 `01a95003f734c7b78d90d62385bf250a908eeaa2b12362f1f2cb33f091a14682`。两版 LDS282624 B，没有 dispatch 或计时。关闭新快路径重编译控制 ELF仍为 `7cbd2cb028cc6a5263c1fc90fabe34710158bf19f31095015d52e07bf3a74a9f`。

后续 `ROW_TAIL_PREDICATION=3` 合并相邻两个 M16 group 的判断，不复制完整 WMMA 快路径；每对group的第一组有效时顺序执行两组，至多多算一个全 padding M16 group。任何有效 group 的全部56次 K128 更新及顺序保持，输出仍只写有效行。CPU 新增3084项pair边界验收，逐个确认有效group全部保留、无有效group丢失。实际资源、精度和性能待验收。

`compile_a07_nt1_rowpairs_qscale` 有3 VGPR spills/private16 B，VGPR512/SGPR68、4554 instructions、ELF SHA256 `9aa205f1b62594d23b9de05b1637db4c3f09f52436f53752ac1976c8c6d29e22`，不 dispatch。加入 halfA 后的 `compile_a07_nt1_rowpairs_qscale_halfapre` 为 LDS282624 B、VGPR477/SGPR68、zero spills/private、4603 instructions、ELF SHA256 `8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f`，进入 seed0/17 严格精度验收。

`a07_nt1_rowpairs_qscale_halfapre_seed0/17` 均 rc0、未超时，每个 seed 的全部有效字节、三次 eager/graph canary 和三组动态 replay 通过；每项28条零字节差异报告。payload 14155776 B、ScaleA 884736 B、GEMM2 198180864 B、MoE 22020096 B 与原 HEAD 绑定参考一致，MoE 原精度仍为 `3.7793e-06/2.7493e-03` 和 `3.7788e-06/2.7491e-03`。两项结束/settled 检查均出现外部占用，故只使用正确性结果，不使用任何耗时。

| 相邻组 | GEMM1 三轮（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | ownership 验收 |
|---|---|---|---|---|
| `a07_nt1_rowpairs_qscale_halfapre_perf_a` | 113.795, 111.920, 113.192 | 94.6482, 93.6763, 95.1582 | 263.855, 259.374, 263.721 | 三轮均不合格 |
| `a07_nt1_rowtail_qscale_control_i` | 113.573, 113.793, 119.104 | 94.3400, 94.4668, 93.5603 | 264.080, 262.655, 268.496 | 仅第一轮合格 |
| `a07_nt1_rowpairs_qscale_halfapre_perf_b` | 113.904, 113.244, 111.612 | 94.3698, 94.3510, 94.3994 | 263.726, 262.906, 262.356 | 仅第三轮合格 |

九轮均 rc0、未超时，未取得完整有效三轮组。111.612 μs 是有效单轮候选，113.573 μs 是本队列的有效单轮控制；二者只支持继续验证，不证明稳定附加收益，不替代114.988 μs的历史完整三轮最佳。

### 生产代码收敛与默认选择

全部失败或无稳定收益的 geometry、lookup、reuse、内部 K/M、XDL、输出重排等实验已保存到 `.codex_tmp/dsr1_e65_gemm1_20261017/production_before_cleanup/` 和对应每组 source snapshots；生产修改从629行新增/147行删除收敛到152行新增/16行删除，仅保留 N-task 拆分、M16/M32 pair predication、canonical ScaleA TDM 和 halfA。公共 amax、GEMM2 AST、原 benchmark、grouped_moe、quant_utils、gemm_common 审计保持。

收敛源码独立编译的三份实际 ELF与既有验收产物完整相同：原四任务 fallback=`958d31dd24a22c8a38af0ec1113a27ee8c5bf69e8ffb124ff222758f11c68110`；nt1/rowtail/ScaleA=`7cbd2cb028cc6a5263c1fc90fabe34710158bf19f31095015d52e07bf3a74a9f`；nt1/pair/ScaleA/halfA=`8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f`。三者均zero spills/private。资源与指令数分别480/106/4386、511/70/5052、477/68/4603，LDS282624 B。

默认 dispatch 仅对已测试的 `E65 + contiguous_m=30464 + non-balanced + t256/w2x4/b4 + K7168/N4096 + SiLU fused quant` 选择 `N_TASKS=1 ROW_TAIL_PREDICATION=1 STAGED_QUANT_SCALE=1`；支持任意实际 expert counts、空专家、partial rows 和同地址 routing 更新。该选择只用静态形状/布局，不读取 random seed 或激活值。其他规模保留原四任务 fallback；环境变量仍能明确选择四任务或 pair/halfA对照。最终默认路径的两个 seed 精度与原完整 benchmark 正在重新验收。

`a07_clean_default_seed0/17` 已重新通过原 HEAD 绑定参考的全部有效字节、每seed三次eager/graph canary、三组动态input/routing replay，rc0、未超时、ownership合格、无live post owners。新增 `audit_dsr1_e65_dispatch.py` 直接执行生产AST中的默认选项分支，确认E65目标得到 `(1,1,1,0)`，E64/E66、其他capacity、t192、balanced、direct scale/vm192等fallback仍为 `(4,0,0,0)`，显式四任务和pair/halfA覆盖正确。它是CPU dispatch验收，邻近规模仍待device通用性检查。

| 收敛版相邻对照 | GEMM1 三轮（μs） | 中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） |
|---|---|---:|---|---|
| `a07_clean_default_perf_a`，nt1/rowtail/ScaleA | 118.024, 117.314, 116.420 | 117.314 | 94.1503, 93.9249, 93.7161 | 268.398, 267.230, 266.039 |
| `a07_clean_rowpairs_halfapre_perf_a` | 114.115, 112.532, 113.251 | **113.251** | 94.0832, 94.6037, 94.6839 | 263.821, 262.432, 263.545 |
| `a07_clean_default_perf_b`，nt1/rowtail/ScaleA | 117.219, 112.962, 115.572 | 115.572 | 94.1232, 94.5195, 94.1041 | 266.278, 263.513, 265.617 |

九轮均rc0、未超时、ownership合格、无live post owners。pair/halfA相比前后对照分别降低3.46%/2.01%；有效完整三轮最佳更新为113.251 μs，相对原HEAD153.405 μs降低26.18%，距离90 μs差23.251 μs。继续反向三轮确认与a07原脚本ATT，不从单轮113.573或111.612预测稳定收益。

### pair/halfA 反向确认及实际 code object

`a07_clean_rowpairs_halfapre_perf_b` 的 GEMM1 为111.492、113.333、113.320 μs，GEMM2为94.3255、94.9076、93.6088 μs，MoE e2e为261.457、264.633、263.228 μs。三轮均rc0、未超时；第一轮六秒settled检查出现外部owner `2648295`，排除，因此不统计完整三轮中值。后两轮ownership合格，支持约113.3 μs的结果，没有采用第一轮111.492 μs作为性能结论。

使用用户原 `get_att_b_rocm_sdk.sh` 在a07/dsr1_ep4采集pair/halfA，SIMD0、四SE、诊断iters2，raw/decoded/目标symbol/PC映射验收通过。实际 `out_gfx1250_code_object_id_22.out` 与 `compile_a07_clean_rowpairs_qscale_halfapre/kernel.elf` 完整逐字节一致，SHA256 `8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f`。采集目录为 `e65_g1_rowpairs_halfapre_att_20261017/`，本次UI目录是 `ui_output_agent_32217_dispatch_58`，不能与旧agent18872混用。

采集后的post/settled检查存在外部live owner `2658363`，`ownership_qualified=false`；以下cycles只提供诊断方向，不作为有效性能对照。50个sampled waves中12个没有WMMA，其他动态WMMA计数为448×11waves、672×5、896×22，missing PCs为空。896组entry/compute/epilogue span中值分别5276.5/23644/4309.5 cycles，compute内WG wait累计中值4276.5 cycles、DS wait2286 cycles、tensor wait81 cycles。448/672组compute仍为23144/23445 cycles，减少padding WMMA没有按比例缩短采样wave跨度，继续关注WG内wave进度和同步限制，不能唯一归因于某个FIFO或cache。steady WG wait两处456条记录的中值为159/72 cycles，steady DS wait0为59 cycles，末尾tensor wait0为1060.5 cycles。profiler诊断日志里的异常小μs不作为正式kernel性能。

根据收敛源码的完整相邻收益及反向有效样本，目标默认更新为 `(N_TASKS,STAGED_QUANT_SCALE,ROWTAIL,HALFA)=(1,1,3,1)`；原用户命令不需要新增优化开关。静态shape/layout守卫保持，其他规模默认 `(4,0,0,0)`。显式 `ROW_TAIL_PREDICATION=1 HALF_A_PREFETCH=0` 保留旧组合对照；显式 `N_TASKS=4` 保留原四任务对照。该默认仍需两个random seed、动态replay、相邻规模device正确性和原命令正式性能验收，结果随后补充。

### 原用户命令的默认路径验收

`a07_final_pair_default_seed0/17` 已通过全部有效payload/ScaleA/GEMM2/MoE逐字节原HEAD对照、各三次eager/graph canary和三组动态输入/routing replay。两项rc0、未超时、ownership合格、无live post owners，原MoE精度保持。

| 最终默认相邻组 | GEMM1 三轮（μs） | 完整有效中值（μs） | GEMM2 三轮（μs） | MoE e2e 三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `a07_final_pair_default_perf_a`，原用户命令 | 111.838,115.283,114.031 | **114.031** | 95.0571,94.3635,94.5489 | 262.594,264.675,264.707 | 三轮合格 |
| `a07_final_rowtail_control_perf_a`，旧组合 | 115.943,116.325,116.818 | 116.325 | 94.7428,94.2834,93.2248 | 265.613,266.045,265.935 | 三轮合格 |
| `a07_final_head_control_perf_a`，原四任务 | 154.510,152.959,155.132 | 154.510 | 89.4763,89.4444,90.6991 | 300.508,297.968,301.974 | 三轮合格 |
| `a07_final_pair_default_perf_b`，反向原命令 | 114.990,111.728,113.495 | 不统计 | 94.6058,94.2981,94.4035 | 264.788,262.340,265.109 | 前两轮合格；第三轮排除 |

全部rc0、未超时。最终默认首组114.031 μs比本轮相邻原四任务154.510 μs低26.199%，比旧组合116.325 μs低1.973%；MoE e2e中值264.675 μs比原四任务300.508 μs低11.924%。114.031 μs仍比90 μs高24.031 μs。历史最佳完整有效组113.251 μs保留为历史结果，不替代默认原命令的本轮验收。GEMM2实测由约89.48 μs到94.55 μs，代码/AST保持，不把GEMM1下降直接当作全链路同幅收益。

### B-only TDM temporal hint 的有限实验

查阅 Shader Guide §4.2.1 Table19（p62）及CDNA5 ISA §10.11.1 Table61（p140）：TH字段控制数据在cache hierarchy中的保留优先级；TDM指令有3-bit TH字段。已有kernel的 `tdm_b_th` 仅传给B payload TDM，地址、tile、ScaleB、A和所有同步保持。本轮只测试TH1/2/4/5/6，不使用TH3的last-use/discard-dirty语义或reserved TH7，也不从文档推测必然收益。

使用独立 `run_dsr1_e65_gemm1_bth.py` 在GEMM1调用时设置既有 `tdm_b_th`，不修改原benchmark、生产默认或GEMM2，不用公共 `AITER_TDM_B_TH` 同时改变两阶段。独立compile脚本确认实际ISA、zero spills/private，随后双seed逐字节/动态replay通过才跑原完整benchmark。host runner的 `--gemm1-b-th` 记录该driver及原benchmark的完整参数，保持warmup5/iters20/count19；收益须经原TH0前后相邻对照确认。

五种hint已通过实际ISA比对：都为LDS282624 B、VGPR477/SGPR68、zero spills/private、4603 instructions，指令清单相对TH0只改变索引340/480/587/694/1474的五条B `tensor_load_to_lds` TH字段。地址、operand、全部WMMA、wait和barrier指令一致。ELF依次为TH1=`228be67e5f2fbe50d4e629ced2f580379a1df9aa014b2e6c60a6db97c0c26282`、TH2=`d8f5b28d92f34b2d505da02b9daec50914d33fe1eb8028ee81a5dfdda101e220`、TH4=`50f61912a3975c2f96276776f02e857b6ebd91cdec482525e6f10532070ed5c9`、TH5=`084c8b696ea36d60d21b7d277d4e2df0d2e63989105db35907fc26c6241b018d`、TH6=`48b8e8cbcf18c343c85f8067b2bef840e1915880b7d67219238ec0e4b68e9cc6`。资源相同不代表性能相同，双seed及正式对照结果随后记录。

### 邻近规模的 device 通用性验收

本轮补做E65/T1024、T1535、T1537、T4608，以及E64/E66的T1536，均random non-balanced、topk9、shared-experts=1、seed17，没有改变生产routing、shape或fallback选择。每个规模独立运行原HEAD绑定参考，再执行当前默认。十二项实际测试都rc0、没有超过120秒；显式 `correctness_only=true`，全部耗时不计入性能。

| 规模 | 有效routes | MoE有效字节 | 验收 |
|---|---:|---:|---|
| E65/T1024 | 9216 | 14680064 | 两阶段BF16有效行及MoE逐字节相同；原fallback调用参数相同 |
| E65/T1535 | 13815 | 22005760 | 两阶段BF16有效行及MoE逐字节相同；原fallback调用参数相同 |
| E65/T1537 | 13833 | 22034432 | fused payload14164992 B、ScaleA885312 B、GEMM2198309888 B、MoE逐字节相同；各三次eager/graph canary与三组dynamic replay通过 |
| E65/T4608 | 41472 | 66060288 | 两阶段BF16有效行及MoE逐字节相同；原fallback调用参数相同 |
| E64/T1536 | 13824 | 22020096 | 两阶段BF16有效行及MoE逐字节相同；原fallback调用参数相同 |
| E66/T1536 | 13824 | 22020096 | 两阶段BF16有效行及MoE逐字节相同；原fallback调用参数相同 |

T1537的原HEAD `logits_diff=1.5996e-04 rel_l2=1.7886e-02`，当前输出逐字节相同，未放宽原门槛；其他规模logits_diff约3.78e-06。目标形状的双seed/dynamic检查见前节；本表不把相邻规模的fallback精度通过当作<90 μs或EP4服务性能证据。

诊断过程保留失败记录：最初对T1535使用仅适用于fused quant的verifier，原HEAD捕获缺少fused字段而rc1；改为按BF16 grouped row物理顺序跨进程比较时，MoE零差异但中间行不同，因为atomic routing的有效行顺序可变化。后续通过真实 `topids_to_rows` 将两阶段有效行还原到token/topk顺序，T1535同shape逐字节通过。T1537实际采用fused quant，已改用严格payload/ScaleA检查器。失败诊断没有产生性能数据，也没有为了通过比较修改生产kernel、routing或精度门槛。

### B-only TH 初筛结果

TH1/2/4/5/6各两个random seed均rc0、未超时，全部有效payload/ScaleA/GEMM2/MoE逐字节原HEAD对照、各三次eager/graph canary与三组dynamic replay通过。十项都为显式correctness-only，不使用耗时。

| 固定串行顺序 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| TH0 `a07_pair_halfapre_bth0_control_a` | 111.132,112.635,112.738 | 不统计 | 94.8526,94.9011,93.7648 | 261.695,264.725,262.853 | 仅第三轮合格 |
| TH2 `a07_pair_halfapre_bth2_perf_a` | 111.229,112.015,110.761 | 111.229 | 94.8315,95.1182,95.0316 | 262.948,264.636,262.990 | 三轮合格 |
| TH6 `a07_pair_halfapre_bth6_perf_a` | 111.003,111.104,111.161 | 111.104 | 94.2896,95.0466,94.7073 | 262.085,261.149,262.256 | 三轮合格 |
| TH4 `a07_pair_halfapre_bth4_perf_a` | 110.727,111.254,111.667 | 111.254 | 94.3951,93.3557,95.0403 | 262.060,260.414,262.195 | 三轮合格 |
| TH5 `a07_pair_halfapre_bth5_perf_a` | 113.803,114.632,111.760 | 113.803 | 94.5301,93.5454,94.1968 | 263.246,263.027,261.990 | 三轮合格 |
| TH1 `a07_pair_halfapre_bth1_perf_a` | 109.951,110.898,110.335 | **110.335** | 93.7752,94.2075,94.4120 | 259.636,260.258,260.144 | 三轮合格 |
| TH0 `a07_pair_halfapre_bth0_control_b` | 112.283,110.801,112.787 | 112.283 | 94.2874,93.2965,94.9474 | 261.316,261.009,263.880 | 三轮合格 |

21轮均rc0、没有超时，前置TH0两轮因外部owner排除。TH1最低完整有效组110.335 μs，比本轮后置TH0低1.735%，比原HEAD历史153.405 μs低28.076%，仍比90 μs高20.335 μs。TH6低1.050%，TH2/4约低0.9%，TH5无收益。由于初筛多个候选而前置完整TH0无效，尚不能据最低批次接入默认；后续安排TH1→TH0→TH6→TH0→TH1相邻复测，收益稳定才更新生产默认。

### B-only TH 反向复测与默认决定

以下15轮全部rc0、未超时、ownership合格，无本轮测试进程残留。保持原完整MoE benchmark、warmup5/iters20/count19及GEMM2不变。

| 固定串行顺序 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| TH1 `a07_pair_halfapre_bth1_perf_b` | 111.809,110.122,110.316 | **110.316** | 94.1083,92.3246,93.6339 | 261.520,259.154,259.838 |
| TH0 `a07_pair_halfapre_bth0_control_c` | 112.702,112.755,112.350 | 112.702 | 93.7035,93.2333,95.0929 | 262.360,260.736,263.291 |
| TH6 `a07_pair_halfapre_bth6_perf_b` | 110.915,111.246,110.889 | 110.915 | 93.1259,94.3803,94.0069 | 260.353,260.418,261.799 |
| TH0 `a07_pair_halfapre_bth0_control_d` | 110.527,112.323,111.083 | 111.083 | 94.2306,95.6748,93.7920 | 260.257,263.823,260.983 |
| TH1 `a07_pair_halfapre_bth1_perf_c` | 111.123,110.982,111.530 | 111.123 | 95.0172,94.7386,95.2807 | 261.492,261.995,262.288 |

TH1前组比相邻TH0低2.117%，后组则比相邻TH0慢0.040 μs；TH6比后置TH0仅低0.151%。TH字段的稳定附加收益不足，生产默认继续TH0。110.316 μs是候选的完整有效组，不替代默认原命令114.031 μs的验收性能，也不能据此宣称接近90 μs的稳定默认性能。

### split carry fence 的有限同步实验

Shader Guide §4.3.6（p74–77）及CDNA5 ISA §5.6（p45–48）明确允许先signal、后wait；barrierComplete由每wave的wait消费。候选仅对当前GEMM1 halfA路径开启：carry点保留tensor_wait及DS wait0，再signal(-1)，执行已在register中的当前FRONT WMMA，随后wait(-1)，才issue未来TDM及读取下一LDS slot。全部wave都执行匹配signal/wait，padding谓词只包围WMMA；56次K128更新及每个accumulator的顺序保持。公共helper和GEMM2不改，默认关闭；先检查真实ISA/zero spills、双seed严格精度和dynamic replay，再做相邻性能对照。

`compile_a07_pair_halfapre_splitcarry0` 在新源码中显式关闭候选，完整ELF仍等于已验收默认 `8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f`。候选 `compile_a07_pair_halfapre_splitcarry1` 为LDS282624 B、VGPR469/SGPR68、4521 instructions、zero spills/private，ELF=`f9ca6bf309d8d0c44d56d1f3fd6fbeee670f701cd3ed43bfdd2b9d2911f408dc`。实际ISA的七处carry fence在signal/wait间各有8条WMMA、没有DS/TDM/global memory指令，signal前保留tensor_wait及DS wait0。两版本static WMMA均256条，WG signal/wait均10对；ISA自动验收记录为 `splitcarry_isa_acceptance.json`。

`a07_pair_halfapre_splitcarry_seed0/17` 均rc0、未超时，通过payload14155776 B、ScaleA884736 B、GEMM2198180864 B、MoE22020096 B的原HEAD逐字节对照，各三次eager/graph canary与三组dynamic input/psum replay。显式correctness-only，所有耗时不作为性能数据。正式相邻三轮计时按TH0默认→splitcarry→TH0默认→splitcarry顺序进行，复现时只新增 `AITER_GFX1250_GEMM1_SPLIT_CARRY_FENCE=1`，其余命令与benchmark保持。

| splitcarry 首轮相邻组 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `a07_pair_halfapre_splitcarry0_control_a` | 112.044,112.998,112.719 | 不统计 | 93.655,94.3275,93.3556 | 261.218,262.368,261.671 | 前两轮排除，第三轮合格 |
| `a07_pair_halfapre_splitcarry1_perf_a` | 112.856,118.562,114.676 | 114.676 | 94.2307,94.0936,93.8533 | 262.883,267.686,264.099 | 三轮合格 |
| `a07_pair_halfapre_splitcarry0_control_b` | 113.312,112.215,112.833 | 112.833 | 93.8301,93.9417,93.0752 | 262.290,262.026,261.936 | 三轮合格 |
| `a07_pair_halfapre_splitcarry1_perf_b` | 115.026,115.433,113.383 | 115.026 | 93.2249,93.6172,93.9417 | 264.264,264.994,261.700 | 三轮合格 |

十二轮全部rc0、未超时，无live post owners；前两轮settled检查有外部占用，保留日志但排除。候选两组比中间默认慢1.633%/1.943%，不采用。虽然signal/wait间覆盖了8条WMMA且静态VGPR下降，下一operand的DS预读也整体推迟；仅由静态ISA不能确定回退的唯一原因。再做范围更小的候选splitcarry2：只用第一个M32 pair的4条WMMA覆盖wait，第二FRONT pair仍放在wait后覆盖future LDS reads；累计顺序、输入完成与LDS复用门禁保持。该候选仍默认关闭，先ISA/双seed验收，再相邻计时。

splitcarry2实际ISA七处carry点在signal/wait之间各4条WMMA，无DS/TDM/global memory指令，static WMMA256条、WG signal/wait10对保持。ELF=`d5ee0d815dade36424f2ea0b9c5d604b463a95466f56ee8f1dafaa207a3ec26f`，LDS282624 B、VGPR485/SGPR68、4545 instructions、zero spills/private。`a07_pair_halfapre_splitcarry2_seed0/17`均rc0、未超时，原HEAD全部有效字节、各三次eager/graph canary与三组dynamic replay通过；correctness-only，不使用耗时。

| splitcarry2 相邻组 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `a07_pair_halfapre_splitcarry0_control_c` | 111.861,115.688,112.724 | 112.724 | 94.3171,92.8264,93.7308 | 262.351,263.277,262.301 | 三轮合格 |
| `a07_pair_halfapre_splitcarry2_perf_a` | 111.745,113.966,115.431 | 113.966 | 94.9600,95.1498,94.1483 | 263.115,265.153,265.098 | 三轮合格 |
| `a07_pair_halfapre_splitcarry0_control_d` | 115.515,115.363,112.599 | 不统计 | 93.8638,94.0493,93.7121 | 264.706,265.691,262.035 | 第二轮排除，其他合格 |

九轮全部rc0、未超时，无live post owners。splitcarry2比前置默认中值慢1.102%，没有稳定收益；后置默认第二轮settled外部占用，不用剩余两轮拼中值。splitcarry1/2都撤出生产源码，保存独立源码快照及实际ISA、精度、计时证据，不留下默认无收益开关。恢复后两个生产文件SHA应回到wrapper=`74a449d5a9656db834c1d0bc67695e96634258320e85501a788b8178264079ce`、persistent=`819f0c782f23a676f3ae004c8170537c9b52919ed3e8624063ba3cbe0ab157c8`；默认的输入/输出lifetime fences保持，继续补可靠ATT。

### 默认 pair/halfA 的无污染 ATT 验收

`a07_final_pair_default_att` 使用用户原 `get_att_b_rocm_sdk.sh` 在a07/dsr1_ep4采集SIMD0、四SE，诊断iters2，rc0、未超时、ownership合格，无live post owners。GPU占用时host每6秒等待，整机idle后开始，采集期间没有并发监控或编译。raw、decoded、目标symbol和PC映射验收通过，actual code object id22与当前默认ELF逐字节相同，SHA=`8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f`。

trace目录=`e65_g1_final_pair_default_att_20261017/`，实际UI=`thread_trace/simd0/kernel/rpf_v3/ui_output_agent_2944_dispatch_58`。四SE采样共40个waves，动态WMMA数448×9、672×7、896×24，missing PCs为空。896组entry/compute/epilogue span中值为4091/23558/4218 cycles，compute中WGwait累计4360、DSwait2202、tensorwait81 cycles。448组compute中值23134 cycles；padding WMMA减少仍没有使wave跨度按比例缩短。

steady WG wait两处各480条记录中值167.5/72 cycles，DS wait0中值55.5/37 cycles；入口tensorwait3中值1099、末尾tensorwait0中值1063 cycles。数据支持后续检查WG内wave进度、LDS operand搬运和控制指令开销，不能唯一归因于硬件FIFO/cache。ATT cycles仍不是正式kernel μs，不把trace阶段独立相加预测90 μs，也不将采集log的profiler耗时作为正式性能。

### 提前生成 wave-uniform M32 predicates

新的有限候选仅对现有pair rowtail提前生成M32有效行predicate，每个predicate由真实 `wmb + wm*16 < mn_oob` 决定：rowuniform1将0/1值经过readfirstlane，rowuniform2经过ballot，再以非零判定保持原group的所有WMMA。Shader Guide §4.6（V_READFIRSTLANE）及§4.6.12（WMMA忽略EXEC）支持使用wave-uniform值，全部有效group、原累加顺序及barriers保持；没有改成lane EXEC predication。默认关闭；实际ISA/精度/计时分别验收。

| 实际编译 | LDS（B） | VGPR/SGPR | instructions | ELF SHA256 |
|---|---:|---|---:|---|
| rowuniform0控制 | 282624 | 477/68 | 4603 | `8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f` |
| rowuniform1 | 282624 | 477/74 | 4598 | `a096234e4199715e0a2ea7fdfadaa1e8fe9cd25cfd1dd8e62796b99652f4a278` |
| rowuniform2 | 282624 | 477/68 | 4579 | `5413aecb5df31b915f15c6983e5d84c8d1d2688bb2ffabed7cc2cc3a02f49d67` |

三者zero spills/private、static WMMA256条、WG signal/wait10对、DS b128384条、input TDM20条、output TDM2条保持。编译器已将原predicate大部分scalar化，提前生成只减少少量静态控制指令，不能预判收益。双seed逐字节/dynamic验收通过后才执行原完整benchmark的相邻三轮对照。

rowuniform1/2的seed0/17四项全部rc0、未超过120秒，原HEAD有效payload14155776 B、ScaleA884736 B、GEMM2198180864 B、MoE22020096 B逐字节一致；各三次eager/graph canary与三组同地址dynamic input/psum replay通过。四项为correctness-only；外部占用不影响这里的数值验收，全部耗时不作为性能数据。

| 固定串行顺序 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| rowuniform0 `a07_pair_halfapre_rowuniform0_control_a` | 111.448,113.126,111.188 | 不统计 | 94.1313,93.7561,93.5643 | 260.782,262.497,259.342 | 仅第一轮合格 |
| rowuniform2 `a07_pair_halfapre_rowuniform2_perf_a` | 114.355,113.145,112.941 | 不统计 | 93.9966,93.2859,94.4646 | 264.295,261.799,262.838 | 前两轮合格，第三轮排除 |
| rowuniform1 `a07_pair_halfapre_rowuniform1_perf_a` | 113.742,112.428,115.093 | 113.742 | 94.7070,94.0471,93.6486 | 264.341,262.011,264.674 | 三轮合格 |
| rowuniform0 `a07_pair_halfapre_rowuniform0_control_b` | 113.196,113.959,112.810 | 113.196 | 94.4794,94.3084,94.2431 | 264.765,263.851,262.370 | 三轮合格 |

十二轮都rc0、未超时。readfirstlane候选比相邻默认中值慢0.482%；ballot没有完整有效三轮组，两个有效样本也未显示明确优势。两者不采用，实验源码保存在 `rowuniform_experiment_sources/`，实际编译、精度、计时与ownership证据保留。最新默认组113.196 μs支持约113–114 μs的已有水平，不作为新代码收益。

### 按M32/M64整组跳过全padding A预读

无污染ATT中，448/672/896动态WMMA组的compute段DS loads均约1485条。此观测说明现有WMMA rowtail并未按有效行数减少operand搬运；不等同于已经证明DS是唯一瓶颈。最后一组有限候选 `ROW_TAIL_A_GROUPS=2/4` 分别以M32 pair或M64 half为单位，只在整组行都为padding时跳过A payload DS loads。两条M16中的任何一条可能执行WMMA时，M32 pair的两个A operands都加载；M64按首行判定，可额外加载padding pair，但不漏有效pair。ScaleA/B、所有同步、两个register slots及56次K128累加保持。

该实验沿用已验收pair/halfA，默认关闭，先编译检查实际ISA和spills；资源不合格的版本不dispatch、不计时。与先前逐M16、无halfA的失败方案分别记录，避免把不同源码或失败数据混用。

| 实际编译候选 | VGPR/SGPR | VGPR spills / private B | instructions | ELF SHA256 |
|---|---|---|---:|---|
| M64 half，`rowaload4` | 486/68 | 1 / 8 | 4436 | `e6a3e35d6ae1af09819112659cd6247bb993120b2a836930defa7f627d7d16fb` |
| M32 pair，`rowaload2` | 486/68 | 9 / 20 | 4847 | `3d2b106b47bc2ba5dfabb702406462a2ef710e7bc02605f7261257687b1664a4` |
| M64 half，padding slot写零，`rowaload8` | 464/68 | 15 / 32 | 5307 | `0be50016018ca03aa64f612d825393ef797812d2f03f76651e324408caf771c8` |
| M32 pair，padding slot写零，`rowaload6` | 480/68 | 20 / 44 | 5693 | `0bc845e152e20172a9d328bfefda049312383fe0dd1ec104bcc8da03dc8ca429` |

四者LDS都是282624 B。后两项给绝不执行WMMA的全padding group定义register值，尝试消除条件加载造成的旧slot loop-carried状态；它们增加spills，不能依据这次结果唯一判断compiler内部原因。四者全部仅编译，没有GPU dispatch、精度结果或性能数据，不采用。实验源码和compile目录保留；关闭候选重编译的默认ELF仍等于已验收 `8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f`。

## 已验收代码与阶段结论

目标90 μs尚未达到。保留原用户命令直接启用的已验收默认 `(N_TASKS,STAGED_QUANT_SCALE,ROW_TAIL_PREDICATION,HALF_A_PREFETCH)=(1,1,3,1)`；最新两个predicate和四个A预读候选均撤出生产，不留下无收益开关。功能、全部13824条有效routes、56次K128累加及顺序、SiLU、BF16 rounding、FP4/E8M0和canonical布局保持。动态psum由kernel读取，不读取seed或按输入数值选择路径。

可重复的正式验收采用完整三轮组：原HEAD历史中值153.405 μs；最终默认原用户命令验收114.031 μs，相比历史基线降低25.667%，相比同期原四任务154.510 μs降低26.199%。收敛阶段完整有效组113.251 μs，最新相邻默认组113.196 μs；后续控制组中也有111–113 μs的完整有效中值，这些环境/批次变化不算新增代码收益。未采用的TH1候选最低完整有效组110.316 μs，因反向复测不稳定不作为默认结果。以114.031 μs验收口径计算，还需降低24.031 μs、即当前耗时的21.074%，才能达到90 μs边界。

瓶颈证据来自有效ATT而不是profiler采集μs：896组compute span P50为23558 cycles，其中compute WGwait累计4360 cycles、DSwait累计2202 cycles；448/672组WMMA减少仍约23134/23284 cycles，DS loads数量近似固定。默认有477 VGPR、282624 B LDS、zero spills。证据支持重点是WG内wave进度、同步与LDS operand搬运，并受register/lifetime限制；不构成某个硬件FIFO、cache或链路的唯一归因，也不能把trace cycles直接换算为正式kernel μs。改变内部geometry、register slots、M分配、carry fence或padding A预读的多组试验没有得到满足精度/资源/性能的稳定更快版本，详见前面各节。

已验收两个生产文件SHA256：wrapper=`74a449d5a9656db834c1d0bc67695e96634258320e85501a788b8178264079ce`；persistent=`819f0c782f23a676f3ae004c8170537c9b52919ed3e8624063ba3cbe0ab157c8`。GEMM2 AST、公共amax/helper和原benchmark/routing保持。原默认及邻近规模的严格精度/动态验证见前节。恢复后 `compile_a07_final_restored_default/fallback` 独立重编译通过：默认ELF=`8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f`、原四任务fallback ELF=`958d31dd24a22c8a38af0ec1113a27ee8c5bf69e8ffb124ff222758f11c68110`，两者与原已验收产物完整一致且zero spills/private。CPU source/dispatch审计通过，确认GEMM2/公共文件保持、13项目标和fallback选择正确。代码在本地worktree与a07共享仓库同步，不commit或push。

### 继续优化：padding operand的LLVM状态和spill位置

用户继续优化后，补做 `tailaundef4/2`：在M64/M32条件加载前将不会被WMMA消费的全padding slot定义为LLVM `undef`。CPU覆盖审计枚举有效行0..256、两个M wave bases、FRONT/BACK/full和两种group宽度，共3084项，通过“所有执行WMMA的group必定加载A”验收。它不是lane EXEC predication；Shader Guide §3.2（p29）说明EXEC=0不能保证跳过LDS，§4.6.12（p144）说明WMMA强制全EXEC，因此仍需scalar branch。

初次 `tailaundef4` compile因现有嵌套import使新 `llvm_dialect` 名称被遮蔽而rc1，未dispatch；改成独立module名 `tail_llvm` 后重编译。`compile_a07_pair_halfapre_tailaundef4_v2` 为VGPR462/SGPR68、15 spills/private32 B、4570 instructions，ELF=`a616a84d057386bee198151b6cb90c2b60a3b92f60b88fb5b67e395d1623b116d`；`tailaundef2_v2`为VGPR478/SGPR68、18 spills/private44 B、4871 instructions，ELF=`3a2af05f63d6416e5d0cc2db0ffec5bc148933e256618cb099599c714d37a4ba3`。两项LDS282624 B，仅编译，不dispatch、不计时、不采用。关闭候选的default ELF仍完全相同。

随后对已有M64 `rowaload4`检查spill位置：只有PC `0x4834` 的scratch store和 `0x4CC8` 的scratch load，全部在最后unrolled drain、所有runtime K-loop backedge之外，不是每次steady K迭代访问scratch。由此调整早先“有spills即不dispatch”的筛选决定：只对此有明确spill范围的版本补双seed全部有效字节、canary/dynamic replay与默认相邻计时，验证padding A搬运下降能否抵消一次drain spill成本。`rowaload2/6/8`和`tailaundef`仍不dispatch。这个候选默认关闭，尚未采用；原已验收默认数据和结论保持。

`a07_pair_halfapre_rowaload4_seed0/17`都rc0、未超时，原HEAD有效payload14155776 B、ScaleA884736 B、GEMM2198180864 B、MoE22020096 B逐字节一致，各三次eager/graph canary和三组动态input/psum replay通过，每项28条零字节差异。显式correctness-only，GPU占用时只采用数值结果。

| M64 A预读对照顺序 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `a07_pair_halfapre_rowaload0_control_a` | 115.711,114.014,114.183 | 114.183 | 94.1632,94.2034,94.5806 | 265.965,264.580,264.943 | 三轮合格 |
| `a07_pair_halfapre_rowaload4_perf_a` | 116.027,111.781,114.111 | 不统计 | 94.1737,94.4456,94.6839 | 266.307,262.685,265.460 | 前两轮合格，第三轮排除 |
| `a07_pair_halfapre_rowaload0_control_b` | 112.293,113.782,113.128 | 113.128 | 94.1736,93.7752,94.5806 | 261.856,263.823,264.080 | 三轮合格 |
| `a07_pair_halfapre_rowaload4_perf_b` | 110.807,112.958,110.914 | 不统计 | 93.3913,93.4063,94.7723 | 258.493,262.536,261.116 | 仅第二轮合格 |

十二轮rc0、未超时。候选没有完整有效三轮组，110.807/110.914 μs排除，不能与默认比较并宣称提升。有效候选111.781–116.027 μs仍不足以证明稳定收益，因此不更新默认。继续有限改动 `HALF_SCALE_PREFETCH=1`：把下一BACK的ScaleA与下一BACK A一起延后预读，拆成两个2-wide scale register slots，试图缩短scale live range和消除drain spill。有效输入地址、scale值、WMMA及K128累加顺序保持；先compile，再独立精度和相邻计时。

### BACK ScaleA延后预读的实际编译与首轮对照

`HALF_SCALE_PREFETCH=1`将一个4-wide ScaleA register slot拆成两个2-wide slots，下一FRONT只读取pairs0/1，当前FRONT WMMA之后读取下一BACK A和ScaleA pairs2/3。当前FRONT/BACK仍使用本K128原scale值；全部有效WMMA、56次K128更新及顺序、输入地址和同步保持。默认仍显式关闭新实验，不依据register数变化宣称性能收益。

| 编译目录 | VGPR/SGPR | spills/private B | instructions | ELF SHA256 |
|---|---|---|---:|---|
| `compile_a07_pair_halfapre_halfscale0_rowaload0` | 477/68 | 0/0 | 4603 | `8b3a05176d953b8eba8ae8f8de50549e4bbfaba39830bc4bcb87c431ce1e193f` |
| `compile_a07_pair_halfapre_halfscale1_rowaload4` | 486/68 | 0/0 | 4504 | `a27c6e9928261b967e45522fc132a82d478f34a0cc15aadadfeae86fc2de1f3a` |
| `compile_a07_pair_halfapre_halfscale1_rowaload0` | 469/68 | 0/0 | 4552 | `97ee52b832abd44f15e3ffaedc9b5f0ad8dfde696106f1a3ddf44d674d87477b` |

三项LDS均282624 B，static WMMA256条、WG signal/wait10对、input TDM20条、output TDM2条保持。组合方案消除了此前rowaload4唯一的drain spill；资源验收与实际计时分别记录。persistent源码SHA256=`74bafd174bc174c729a1949ada6978c19f6934ae6b765771099dfe72a5bffee3`，关闭新实验的默认完整ELF与已验收产物一致。

`a07_pair_halfapre_halfscale1_rowaload4_seed0/17`及`...rowaload0_seed0/17`四项均rc0、未超时，每项28条`differing_bytes=0`。原HEAD全部有效payload14155776 B、ScaleA884736 B、GEMM2198180864 B、MoE22020096 B逐字节一致，各三次eager/graph canary及三组同地址dynamic FP4/E8M0/psum replay通过。显式correctness-only，只采用精度，所有耗时无效。

| 固定顺序：halfscale / rowaload | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| 0/0 `control_a` | 114.644,112.709,113.900 | 113.900 | 95.0065,94.7218,94.9790 | 265.651,264.495,265.754 | 三轮合格 |
| 1/4 `perf_a` | 111.908,110.982,111.594 | 111.594 | 95.0571,93.8531,93.8911 | 263.196,262.275,262.351 | 三轮合格 |
| 0/0 `control_b` | 111.224,112.257,113.377 | 不统计 | 95.0024,94.6544,94.2855 | 261.769,263.195,264.348 | 第一轮排除，其余合格 |
| 1/0 `perf_a` | 113.284,111.176,111.317 | 111.317 | 93.8321,94.7009,94.4583 | 262.431,261.457,261.972 | 三轮合格 |
| 0/0 `control_c` | 113.145,112.108,113.904 | 113.145 | 93.8322,94.2771,94.3170 | 262.543,260.743,264.261 | 三轮合格 |

完整label均以前缀`a07_pair_halfapre_halfscale{halfscale}_rowaload{rowaload}_`命名，原始manifest/result/idle/log位于同名artifact目录。十五轮均rc0、未超时，无live post owners；中间控制第一轮因外部GPU占用排除，不从剩余两轮拼中值。两种候选首轮约1.6%–2.0%收益，尚不足以改变生产默认；接着按1/0→0/0→1/4→0/0→1/0的反向顺序各三轮复测。

```bash
# host控制脚本，容器仍运行用户原完整MoE benchmark。
python3 -u my_code/run_dsr1_e65_gemm1_host.py NEW_HALF_SCALE_M64_LABEL --runs 3 \
  --env AITER_GFX1250_GEMM1_HALF_SCALE_PREFETCH=1 \
  --env AITER_GFX1250_GEMM1_ROW_TAIL_A_GROUPS=4
```

反向15轮全部rc0、未超时、ownership合格、无live post owners：

| 反向顺序：halfscale / rowaload | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| 1/0 `perf_b` | 113.824,111.775,112.711 | 112.711 | 94.2161,93.9123,93.8827 | 264.361,261.926,262.726 |
| 0/0 `control_d` | 111.739,112.711,112.549 | 112.549 | 95.1879,94.0494,95.4830 | 263.281,261.363,264.294 |
| 1/4 `perf_b` | 111.927,110.796,110.463 | **110.796** | 93.7963,93.8279,93.5581 | 261.835,260.481,260.243 |
| 0/0 `control_e` | 112.941,114.550,111.323 | 112.941 | 94.4139,94.2433,93.9925 | 262.923,264.972,261.190 |
| 1/0 `perf_c` | 112.426,113.771,114.410 | 113.771 | 93.7942,94.2960,93.5392 | 262.449,264.141,263.608 |

单独延后ScaleA的反向两组比相邻控制慢0.144%/0.735%，不作为独立默认收益。M64组合反向比前后控制低1.558%/1.899%，与首轮收益方向一致；候选最低完整组110.796 μs比原HEAD历史153.405 μs低27.775%，距90 μs仍差20.796 μs。该数字暂为环境开关候选结果，原用户命令的默认验收仍是114.031 μs，后续默认接入和实际code object/ATT另行验收。

更细M32预读组合`compile_a07_pair_halfapre_halfscale1_rowaload2`实际编译为LDS282624 B、VGPR486/SGPR68、4 VGPR spills/private12 B、4841 instructions，ELF=`f9c861718f77e0d069ff1e29524a19c8b15c0b5eb5f18452b3188be7f9a4e8448`。ISA共有四个scratch store和四个scratch load，PC位于`0x55E8..0x6128`，全部在unrolled drain、八条runtime backedge范围之外；branch immediate按signed16-bit dwords解码。静态验收保存在`halfscale_rowaload2_spill_scope.json`。基于这个明确范围，补两个seed精度及M64→M32→M64相邻三轮计时，不把spills直接当成精度失败或性能结论；尚无GPU结果时不采用。

M32组合`a07_pair_halfapre_halfscale1_rowaload2_seed0/17`均rc0、未超时，每项28条零字节差异、三次eager/graph canary、三组dynamic replay和原HEAD有效输出对照通过；correctness-only，不使用耗时。实际相邻计时如下，九轮全部ownership合格：

| M64→M32→M64顺序 | GEMM1三轮（μs） | 中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| M64 `a07_pair_halfapre_halfscale1_rowaload4_perf_c` | 112.059,110.109,110.215 | 110.215 | 94.4899,93.6107,94.2392 | 263.168,259.518,260.366 |
| M32 `a07_pair_halfapre_halfscale1_rowaload2_perf_a` | 119.812,115.730,115.001 | 115.730 | 93.8364,94.7135,93.4398 | 270.108,266.577,265.231 |
| M64 `a07_pair_halfapre_halfscale1_rowaload4_perf_d` | 110.984,113.403,108.907 | 110.984 | 93.7687,92.9804,95.0508 | 260.108,261.999,260.032 |

M32比前后M64中值慢5.004%/4.276%，不采用；108.907 μs只是单轮，不作为默认性能。M64四组完整有效中值111.594/110.796/110.215/110.984 μs，支持已有约1.5%–2%的组合收益，但最终原命令默认性能仍需单独验收。

### M64组合的有效ATT及pair A预读候选

`a07_pair_halfapre_halfscale1_rowaload4_att`在a07/dsr1_ep4整机idle后使用用户原`get_att_b_rocm_sdk.sh`采集SIMD0/fourSE、random目标、diagnostic iters2，rc0、未超时、ownership合格、无live post owners。raw/decoder/symbol/PC验收通过；actual code object id22与编译ELF完整相同，SHA=`a27c6e9928261b967e45522fc132a82d478f34a0cc15aadadfeae86fc2de1f3a`。UI=`e65_g1_halfscale_m64_att_20261017/thread_trace/simd0/kernel/rpf_v3/ui_output_agent_13963_dispatch_58`，验收记录=`halfscale_m64_att_acceptance.json`。

48个sampled waves中10个跳过，剩余448/672/896-WMMA组分别10/6/22个waves，missing PCs为空。compute span中值为21249/23462.5/23796 cycles，compute段DS loads中值为1046/1486/1486条。896组WGwait累计中值2762.5、DSwait2152 cycles。原有效默认ATT的448组DS loads约1485、compute span23134 cycles，支持M64 padding裁剪减少了这组operand搬运；896组span并未明显缩短。不同采样wave、批次、entry排队不同，不把所有阶段中值独立相加或将cycles换算正式μs，也不唯一归因某个FIFO/cache。

据此新增默认关闭`PAIR_A_PREFETCH=1`：下一K128的B/scales及A pair0提前加载，当前pair0/1/2完成后依次加载下一pair1/2/3，BACK ScaleA仍在当前FRONT完成后加载。保留两套register slots、每个accumulator原56次K128更新与順序、全部有效行、input TDM及DS/WG fences。每个执行WMMA的M32 pair必定加载两个M16 operands；M64 predicate可能额外加载padding pair。公共helper、GEMM2和原benchmark保持。

| 新compile目录的rowaload/pairapre | VGPR/SGPR | spills/private B | instructions | ELF SHA256 |
|---|---|---|---:|---|
| 4/0控制 | 486/68 | 0/0 | 4504 | `a27c6e9928261b967e45522fc132a82d478f34a0cc15aadadfeae86fc2de1f3a` |
| 4/1 | 486/68 | 0/0 | 4414 | `d04f27fef41a40499c35d94cb2a4bb2cc08c2ef4b50654c999465965b76b32b0` |
| 2/1 | 486/68 | 0/0 | 4445 | `7efb631594f354e56a06f2e23d51fc061cf38411b7656dd65256d7988453140a` |
| 0/1 | 469/68 | 0/0 | 4619 | `6cd6e39455e0b5d419871cabe36bc417666dd58b3f2f5bf846b3fe21c5ec73df` |
| 0/0控制 | 469/68 | 0/0 | 4552 | `97ee52b832abd44f15e3ffaedc9b5f0ad8dfde696106f1a3ddf44d674d87477b` |

五项LDS均282624 B。persistent源码SHA=`80ec0f9256becb2300efc157e4c9c0cd31816438a0fc4e25720911e16f38a082`；两个控制ELF与先前候选完全一致。pairapre三项均zero spills，M32组合的drain spill已消除，资源减少不作为性能收益；六项双seed精度和M64控制相邻三轮正式计时正在验收。

### pair A分段预读的精度和首轮相邻计时

`rowaload2/4/0 + pairapre1`三个候选各两个seed的六项验证全部rc0、未超时，每项28条`differing_bytes=0`。原HEAD全部有效payload/ScaleA/GEMM2/MoE字节、各三次eager/graph canary和三组同地址dynamic FP4/E8M0/psum replay通过。六项为correctness-only，耗时不作为性能；外部owner仅作为机器状态记录，不杀外部进程。

| 固定顺序：rowaload / pairapre | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| 4/0 `control_a` | 112.022,114.959,111.596 | 不统计 | 94.7474,93.7039,93.9231 | 261.473,263.406,260.690 | 三轮均排除 |
| 2/1 `perf_a` | 108.520,108.676,108.269 | 不统计 | 93.7206,92.7677,94.1592 | 258.430,255.669,257.844 | 仅第一轮合格 |
| 4/0 `control_b` | 109.391,111.459,111.870 | 111.459 | 92.1475,93.7226,94.5004 | 257.667,260.750,262.828 | 三轮合格 |
| 4/1 `perf_a` | 113.299,112.247,109.481 | 112.247 | 93.3685,93.7058,93.1449 | 263.121,261.837,257.900 | 三轮合格 |
| 0/1 `perf_a` | 112.922,112.785,113.352 | 112.922 | 94.5491,94.0261,93.2147 | 262.605,263.203,261.846 | 三轮合格 |
| 4/0 `control_c` | 109.723,110.520,112.070 | 110.520 | 94.2603,93.5054,94.0473 | 260.292,259.882,261.694 | 三轮合格 |

十八轮全部rc0、未超时，队列已完成。首个控制三轮及M32后两轮在post/settled发现外部GPU owner，保留并排除，不挑剩余样本组成中值。M64分段与不裁剪分段版比相邻M64控制慢，暂不采用；M32仅有效单轮108.520 μs，尚不能宣称稳定提升或代替默认验收。补做M64控制→M32→M64控制→M32→M64控制的十五轮确认，依然每轮先整机idle门禁，120秒timeout，计时期间不并发监控或编译。

| M32相邻复测顺序 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| 4/0 `control_d` | 112.507,109.955,111.863 | 111.863 | 94.5050,93.3914,94.3467 | 262.318,259.800,262.442 | 三轮合格 |
| 2/1 `perf_b` | 111.433,109.629,108.973 | 不统计 | 93.5748,91.9917,94.6946 | 261.082,258.080,259.879 | 第一轮排除 |
| 4/0 `control_e` | 111.144,110.980,111.726 | 111.144 | 94.0031,95.0277,92.9278 | 260.886,261.504,260.545 | 三轮合格 |
| 2/1 `perf_c` | 109.114,110.681,106.769 | 不统计 | 92.8117,93.9207,94.9708 | 258.818,259.636,257.500 | 第一轮排除 |
| 4/0 `control_f` | 110.697,109.044,112.011 | 110.697 | 94.7053,93.5981,94.8084 | 261.388,257.629,263.185 | 三轮合格 |

十五轮rc0、未超时。M32两个组仍缺完整ownership验收，106.769 μs仅为合格单轮，不能作为默认或最佳三轮。下一批预先固定相同五组顺序；只因ownership污染整组重试，每组最多三次，不依据性能挑选重试或选最低批次。该规则写在`queue_pairapre_m32_idle_retry.sh`，全部失败组保留。

下一步静态分析查阅旧pipeline记录的“Rejected paired-K256 GEMM1 fences”：M192 balanced旧方案的相邻控制72.613 μs、pair fence80.851 μs，fences减少并未获益，因此不直接移植这个失败方案。继续检查B payload分段预读能否缩短live range和LDS burst；任何新候选仍须实际ELF、严格精度和相邻性能验收。

### M32分段预读的完整ownership重试确认

以下固定顺序只按ownership失败重试整个三轮组，不按耗时选择重试。原始目录名均以`a07_pair_halfapre_halfscale1_`开头。每次完整原MoE benchmark的warmup5/iters20/count19及GPU空闲门禁保持。

| 顺序及label后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `rowaload4_pairapre0_control_g_try1` | 112.644,109.667,110.531 | 不统计 | 93.4253,93.7289,94.3929 | 262.286,258.578,261.091 | 第一轮排除 |
| `rowaload4_pairapre0_control_g_try2` | 109.848,110.219,112.433 | 110.219 | 94.1694,94.1548,93.8512 | 259.870,260.213,262.501 | 三轮合格 |
| `rowaload2_pairapre1_perf_d_try1` | 109.401,107.777,109.042 | **109.042** | 93.3345,94.4099,93.4528 | 258.346,257.430,258.318 | 三轮合格 |
| `rowaload4_pairapre0_control_h_try1` | 114.356,109.787,114.459 | 不统计 | 95.0888,92.9107,94.6037 | 265.739,258.408,264.460 | 后两轮排除 |
| `rowaload4_pairapre0_control_h_try2` | 109.785,110.615,111.507 | 不统计 | 93.5942,94.3721,93.9819 | 259.307,261.123,261.755 | 三轮排除 |
| `rowaload4_pairapre0_control_h_try3` | 111.572,111.189,110.455 | 不统计 | 93.9630,94.5258,94.3466 | 261.863,260.644,259.691 | 第一、三轮排除 |
| `rowaload2_pairapre1_perf_e_try1` | 107.623,108.243,109.496 | 不统计 | 94.7178,91.4139,93.8679 | 257.901,254.574,259.574 | 前两轮排除 |
| `rowaload2_pairapre1_perf_e_try2` | 107.465,109.297,107.967 | **107.967** | 94.7326,93.8426,91.9153 | 257.781,258.455,256.010 | 三轮合格 |
| `rowaload4_pairapre0_control_i_try1` | 111.279,108.964,109.238 | 109.238 | 93.2645,93.4691,92.8978 | 260.006,257.568,258.184 | 三轮合格 |

全部27轮rc0、未超时，队列结束。M32前组比紧邻前置M64低1.068%；反向M32组比紧邻后置M64低1.164%。中间M64没有完整有效组，保持“不统计”，不能把两端单轮或不同批次拼中值。两组独立有效M32中值109.042/107.967 μs支持约1.1%的附加收益；最低组107.967 μs比原HEAD历史153.405 μs低29.619%，仍距90 μs差17.967 μs。原用户命令默认验收仍114.031 μs，新组合暂未默认接入。

### 延后B operand预读与w4x4资源候选

`HALF_B_PREFETCH=1`只把下一K128第二个N32 B fragment从预读头部移到当前FRONT WMMA之后，当前FRONT/BACK仍使用当前slot原B和ScaleB。两套register slots、全部实际routes、K128累加顺序、input TDM及所有DS/WG fences保持，默认关闭。`COMPUTE_M_WARP=4`候选把同一t256的M compute分给四组，原前8个waves仍为TDM owners，其余只计算，16个waves共同参与所有fences；新halfScale/pairA按实际M64每wave的两个M32 pairs调整，canonical payload/ScaleA输出与原ABI保持。

当前本地和a07同步源码：wrapper SHA=`4c73ea62976f40169fbaefaf5cc261476d6a628f505c13da849208f20b671b6a`，persistent SHA=`b18320be51a9d2ce9ba4319db33065191ff4ab152e13ada537c73de5de91200d`。CPU source/dispatch和py_compile通过。关闭新参数的M64控制实际ELF仍完整等于`a27c6e99...e1f3a`。第一次compile队列在成功编译后的host chmod遇到root-owned输出权限而退出；修正为host `sudo -n chmod -R a+rwX`后复用已完成产物继续，没有重复该次编译或GPU dispatch。

### halfB精度、性能与w4x4编译结果

halfB两个候选实际LDS均282624 B、VGPR486/SGPR68、zero spills/private。M64 `rowaload4/pairapre0/halfb1` 为4550 instructions、ELF=`e3c3efa81f3c74168f31ca0a52ceef8ee3641957d960103ca08e28242d674e633`；M32 `rowaload2/pairapre1/halfb1` 为4465 instructions、ELF=`8b647648682d72026b4b096b2c9efb76af294ed414c20efa6380a3abe75151f4`。关闭halfB的M64控制仍是4504 instructions、ELF=`a27c6e9928261b967e45522fc132a82d478f34a0cc15aadadfeae86fc2de1f3a`。

两个halfB候选各seed0/17的四项正确性全部rc0、未超时，每项28条`differing_bytes=0`，原HEAD有效payload、ScaleA、GEMM2、MoE全部字节、各三次eager/graph canary及三组同地址dynamic input/psum replay通过。correctness-only的耗时不作为性能。固定顺序五组三轮如下，label前缀为`a07_halfscale1_`：

| rowaload/pairapre/halfB及后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| 4/0/0 `control_a` | 110.660,110.723,109.850 | 不统计 | 94.0998,94.8799,94.8419 | 260.169,260.964,260.626 | 三轮排除 |
| 4/0/1 `perf_a` | 113.586,110.389,113.080 | 113.080 | 95.1623,94.6122,94.0322 | 265.220,260.063,263.359 | 三轮合格 |
| 2/1/0 `control_a` | 108.131,108.140,107.813 | 108.131 | 94.8862,94.2749,94.5744 | 258.088,258.242,258.569 | 三轮合格 |
| 2/1/1 `perf_a` | 109.061,108.435,109.517 | 109.061 | 91.7596,92.8411,93.2396 | 256.467,256.448,257.859 | 三轮合格 |
| 2/1/0 `control_b` | 109.584,108.996,106.921 | 108.996 | 92.7632,94.3000,93.6024 | 258.046,260.358,256.955 | 三轮合格 |

十五轮全部rc0、未超时，队列完成。halfB的M32中值比前后控制慢0.860%/0.060%，没有附加收益，不接入默认。M64缺少有效前置控制且也未显示比已有M64更快，不采用；污染的三轮完整保留，不拼中值。GEMM2/e2e变化保留，不能拿e2e下降代替GEMM1优化证据。

w4x4把每wave的M compute减半后，未复用A的四项实际资源如下；全部只编译，无GPU dispatch、无性能或精度结果：

| rowaload/pairapre/halfB | VGPR/SGPR | VGPR spills / private B | instructions | ELF SHA256 |
|---|---|---|---:|---|
| 2/0/0 | 256/68 | 612/208 | 3318 | `bab0a8e3418b4a8e41cb7df11f939efad84d101d14531669f56573a31b92caa5` |
| 2/1/0 | 256/68 | 628/208 | 3281 | `86bd1934da7514f67897f92e667a1557f96dfeade1df8f7ae4d25f98cc045878c` |
| 0/1/0 | 256/68 | 312/100 | 3130 | `debaa74560f3c954c16b5311ff9621ef398db56b46f9238e550b3e27170babd16` |
| 2/1/1 | 256/68 | 882/204 | 3405 | `41f36563db2388181fa64aa653dc49ac9cd9d55c992e747063d7fc66d758cbbb` |

### w4x4的单套A registers逐pair复用

默认关闭的`ROLLING_A_PREFETCH=1`限定`COMPUTE_M_WARP=4`、`PAIR_A_PREFETCH=1`、`ROW_TAIL_A_GROUPS=0`。保留两套B/ScaleA/ScaleB slots，A共享单套register tensors；每个当前M32 pair完成所有WMMA后才把该pair替换为下一K128 operand。load_pair_head只预读下一B/scales，不提前覆盖当前A。最后没有下一operand的K128保留原路径。每个accumulator仍按原56次K128顺序更新，M compute分配保持完整t256 canonical输入/输出布局。

halfB队列结束后才同步这项源码，在a07/dsr1_ep4完成两个独立compile，120秒限制。wrapper SHA=`941081f1e28fbfcc215511b3fac3b4d5a8ec5a4b34a01a060f80c858f0d31878`，persistent SHA=`27ef471f80eaa33bb69b673c588dfdf2e8defd217a5e5927bdf46a4df448b5e4`，两端SHA相同；GEMM2 AST、公共helper、benchmark和routing审计通过。结果如下，两项LDS均282624 B：

| w4 rollingA候选 | VGPR/SGPR | spills/private B | instructions | ELF SHA256 |
|---|---|---|---:|---|
| halfB0 | 254/68 | 0/0 | 2978 | `a3e490ceac309408a21d52ea776d96c6c05e65f1202a634326427f347f844bb5` |
| halfB1 | 254/68 | 0/0 | 2979 | `d530e10a4a3a01ba22a24d1d841c1a90eca6deacdca20834d2db8447592e6cbb` |

单套A复用消除了这两个候选的spills。资源合格不等同于精度正确或性能提高；双seed精度完成后，按预先固定M32控制→w4 halfB0→w4 halfB1→M32控制顺序计时，仅ownership污染允许整组三轮重试，不按耗时挑选。脚本为`queue_w4_rolling_compile.sh`和`queue_w4_rolling_verify_perf.sh`。

w4 rollingA两个候选各seed0/17的四项验证全部rc0、未超时，每项28条零字节差异，原HEAD全部有效输出、各三次eager/graph canary和三组dynamic replay通过。固定顺序四组三轮均第一次就完成ownership验收，无整组重试，结果如下：

| 固定顺序及label后缀 | GEMM1三轮（μs） | 中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| M32控制 `rowaload2_pairapre1_halfb0_mw0_rolla0_control_a_try1` | 109.196,108.243,108.977 | 108.977 | 93.8259,93.0015,94.0513 | 259.505,256.890,260.072 |
| w4/halfB0 `rowaload0_pairapre1_halfb0_mw4_rolla1_perf_a_try1` | 122.865,123.845,122.300 | 122.865 | 93.2586,93.1491,93.1280 | 271.398,273.848,270.897 |
| w4/halfB1 `rowaload0_pairapre1_halfb1_mw4_rolla1_perf_a_try1` | 122.972,122.408,121.817 | 122.408 | 92.2738,92.0504,93.8426 | 270.515,270.388,270.880 |
| M32控制 `rowaload2_pairapre1_halfb0_mw0_rolla0_control_b_try1` | 108.174,110.596,108.074 | 108.174 | 93.4232,94.0642,92.5141 | 257.129,261.046,256.313 |

完整labels以前缀`a07_w4_halfscale1_`开头。十二轮rc0、未超时、ownership合格，队列完成。w4/halfB0比前置M32控制慢12.744%，w4/halfB1比后置M32控制慢13.158%，不采用。更低VGPR、zero spills与更多waves没有转化为目标workload的性能收益；不能只由这些指标唯一判断回退原因。

### M32 pairapre的无污染ATT与TDM owner分配实验

`a07_halfscale1_rowaload2_pairapre1_att`使用用户原`get_att_b_rocm_sdk.sh`在a07/dsr1_ep4采集SIMD0、fourSE、random目标命令、diagnostic iters2。rc0、未超时、ownership合格，无live post owners。raw、decoded、目标symbol和PC验收通过；actual code object id22与`compile_a07_pair_halfapre_halfscale1_rowaload2_pairapre1/kernel.elf`完整逐字节相同，SHA=`7efb631594f354e56a06f2e23d51fc061cf38411b7656dd65256d7988453140a`。UI为`e65_g1_m32_pairapre_att_20261017/thread_trace/simd0/kernel/rpf_v3/ui_output_agent_32891_dispatch_58`，记录为`m32_pairapre_att_acceptance.json`。

52个sampled waves中14个跳过；448/672/896-WMMA组分别9/7/22个waves，missing PCs为空。按first/last WMMA issue边界的compute段观测如下：

| 动态WMMA数 | compute span P50（cycles） | DS loads P50（条） | compute WGwait累计P50（cycles） | compute DSwait累计P50（cycles） |
|---:|---:|---:|---:|---:|
| 448 | 27775 | 1050 | 9355 | 884 |
| 672 | 21860 | 1270 | 1801 | 783 |
| 896 | 23454 | 1490 | 1235.5 | 1857.5 |

M32裁剪的DS loads随有效WMMA组数变化，672组相对M64控制的1486条降为1270条。448组仍有长WGwait，compute span没有随工作量比例缩短。不同采样wave和entry/调度批次不相同，不用这些组P50相加或直接换算正式μs，也不由该观测唯一定位某个cache/FIFO。

下一项默认关闭`TDM_OWNER_LAYOUT=1/2`：A/B/ScaleA/ScaleB owners从原`(0,1)/(2,3)/(4,5)/(6,7)`分别改为`(0,4)/(1,5)/(2,6)/(3,7)`或`(0,5)/(1,4)/(2,7)/(3,6)`。每个tensor仍是两个owners，每wave每stage仍一条request；owner ordinal保持原0/1分片的source/LDS地址，不改multicast mask、ring和input/output lifetime fences。两个M128 compute组各承担每个tensor的一半请求，检查owner分配是否改善等待；不由wave编号断言物理FIFO或SIMD placement。

CPU枚举0..256有效行、四个producer、两个owner布局，1712384个source/destination 16-byte chunk检查通过，全部有效分片各一次覆盖。脚本/结果为`audit_m32_tdm_owners.py`与`m32_tdm_owner_mapping_audit.json`。这不是GPU可见性或精度证明。第一次候选compile因FlyDSL的`select(0,1)`不接受两个Python int而rc1，无dispatch；改用候选owners的`wave_m`作为0/1 ordinal，原控制分支继续原`wave-first`。修订使用独立`_v2`目录，不覆盖失败记录或重复已完成控制编译。原owners0控制ELF仍与已验收M32完整一致。

### TDM owner对照结果与默认收敛

owners1_v2/owners2_v2均为LDS282624 B、zero spills/private。owners1为VGPR486/SGPR68、4784 instructions、ELF=`7352bcf45778139d277b1a262984669bc77503d4cde013e1e9c03bdd51a2ff91`；owners2为VGPR486/SGPR75、4821 instructions、ELF=`d5347708938b7284be5d9e2c41f5194a1cd172221b13fb225f4f2be68d0bed3c`。两候选各seed0/17均rc0、未超时，每项28条`differing_bytes=0`，payload/ScaleA/GEMM2/MoE全部HEAD有效字节、各三次eager/graph canary及三组dynamic replay通过。correctness-only的ownership=false不作为性能证据。

固定owners0→owners1→owners0→owners2→owners0顺序，每组三轮只因ownership污染重试整组，不按耗时挑样本。完整label前缀为`a07_m32_halfscale1_pairapre1_`。

| label后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `owners0_control_a_try1` | 108.427,109.304,107.674 | 108.427 | 93.4717,94.2518,94.1103 | 257.987,259.711,257.502 | 三轮合格 |
| `owners1_perf_a_try1` | 110.172,108.988,111.726 | 不统计 | 93.7329,92.3773,90.6147 | 260.043,257.631,258.731 | 第3轮污染 |
| `owners1_perf_a_try2` | 113.495,107.773,109.188 | 109.188 | 94.4057,94.4880,93.4906 | 263.760,258.151,258.752 | 三轮合格 |
| `owners0_control_b_try1` | 110.413,109.647,108.717 | 不统计 | 94.0197,93.8511,93.2564 | 260.534,259.709,258.957 | 第3轮污染 |
| `owners0_control_b_try2` | 106.799,107.925,108.893 | 不统计 | 94.7472,93.9228,94.1819 | 256.141,257.586,259.239 | 第1轮污染 |
| `owners0_control_b_try3` | 109.898,110.489,108.216 | 109.898 | 93.5453,93.8891,93.0625 | 259.248,261.069,257.824 | 三轮合格 |
| `owners2_perf_a_try1` | 111.340,110.400,110.912 | 不统计 | 94.4856,94.0850,93.6886 | 260.835,260.538,260.127 | 第1轮污染 |
| `owners2_perf_a_try2` | 110.714,108.644,112.245 | 不统计 | 94.9812,93.8764,94.7136 | 261.630,255.990,263.692 | 第1/3轮污染 |
| `owners2_perf_a_try3` | 109.245,109.618,109.976 | 109.618 | 93.1996,93.9987,94.0892 | 258.371,259.214,260.097 | 三轮合格 |
| `owners0_control_c_try1` | 110.406,107.216,106.932 | 107.216 | 92.9950,94.0534,93.7204 | 259.619,256.404,255.376 | 三轮合格 |

三十轮全部rc0、未超时、无自身残留。layout1相对相邻控制慢0.702%/快0.646%；layout2快0.255%/慢2.240%，没有稳定附加收益，两个候选撤出生产。owners0的107.216 μs是该独立完整控制组中值，更新此前107.967 μs的历史最低有效组记录；106.799/106.932 μs等单轮不能作为稳定结果。全量摘要为`m32_tdm_owners_complete_summary.json`。

无收益的halfB、w4/rollingA、owner-layout实验生产代码和compiler参数已删除，完整源码保存于`prefetch_owners_before_convergence/`。稳定M32组合接入原守卫的默认值 `(N_TASKS,STAGED_QUANT_SCALE,ROWTAIL,HALFA,HALFSCALE,ROWALOAD,PAIRAPRE)=(1,1,3,1,1,2,1)`，只限定E65/contiguous_m30464/non-balanced/t256相关几何，不按seed、数值或固定expert counts选择。HALFA、ROWTAIL、HALFSCALE单独关闭时，依赖的默认rowaload/pairapre也正确关闭。其他规模仍原四任务fallback，GEMM2、公共helpers、benchmark和routing保持，CPU dispatch/source审计通过。

收敛后本地/a07的wrapper SHA=`526e62d3a77b53f970c9809ea84153103b75a067bdb754c3b1c8f7c3036a17e5`，persistent SHA=`4e05d4d69c4670fdbdf55879a1b69dbd5dbd36a22f333eee287d3afe955574a4`；compiler SHA=`3f81bc372f64e11fb34588cc3b1ffa664ee183b992c39032697b99cd8d1493d2`。完整源码保存于`m32_converged_source/`。

### M32收敛版的编译、精度与原命令默认验收

独立重编译的M32、M64和原四任务fallback三份ELF与此前已验收产物完整一致，均zero spills/private，记录为`m32_converged_compile_acceptance.json`。M32为VGPR486/SGPR68、4445 instructions、SHA=`7efb631594f354e56a06f2e23d51fc061cf38411b7656dd65256d7988453140a`；M64为486/68、4504 instructions、SHA=`a27c6e9928261b967e45522fc132a82d478f34a0cc15aadadfeae86fc2de1f3a`；原fallback为480/106、4386 instructions、SHA=`958d31dd24a22c8a38af0ec1113a27ee8c5bf69e8ffb124ff222758f11c68110`。两seed各28条零字节差异，全部HEAD有效输出、各三次eager/graph canary及三组dynamic replay通过。

固定原命令新默认→旧默认→新默认顺序。旧默认只加`AITER_GFX1250_GEMM1_HALF_SCALE_PREFETCH=0`，相关rowaload/pairapre默认随依赖关闭；未改benchmark和计时方法。污染组仅按ownership整组三轮重试，不按耗时重试或拼中值；全部rc0、未超时。

| label后缀（前缀`a07_m32_converged_`） | GEMM1三轮（μs） | 有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `default_a_try1` | 108.619,108.698,108.123 | 不统计 | 93.109,93.862,93.940 | 257.202,258.333,258.103 | 第2/3轮污染 |
| `default_a_try2` | 108.456,108.796,109.538 | 不统计 | 93.229,93.554,93.004 | 257.850,258.096,258.803 | 第2轮污染 |
| `default_a_try3` | 110.341,109.251,108.764 | 109.251 | 93.086,94.751,94.825 | 259.490,260.129,258.043 | 三轮合格 |
| `old_default_a_try1` | 114.052,112.734,115.007 | 114.052 | 94.093,93.602,94.441 | 263.924,262.307,264.609 | 三轮合格 |
| `default_b_try1` | 109.148,108.342,109.367 | 109.148 | 94.416,93.579,94.559 | 260.222,257.627,258.662 | 三轮合格 |

新默认109.251/109.148 μs相对相邻旧默认114.052 μs低4.21%/4.30%，支持稳定附加收益。原命令默认验收更新为109.148 μs，相对原HEAD历史153.405 μs下降28.85%，距离90 μs仍差19.148 μs。历史独立控制组107.216 μs单独保留，不能代替原命令默认验收。GEMM2/e2e原数据全部保留。队列结束后整机idle、KFD owners为空、VRAM176582656 B，无本轮测试进程残留；全量摘要为`m32_converged_complete_summary.json`。

### M32交错计算与strided输出的精度及性能淘汰

新实验按完整M32 pairs交错两组M compute：wave_m0的M16 starts为0,16,64,80,128,144,192,208；wave_m1为32,48,96,112,160,176,224,240。两组有效pairs数量相差最多1。两套operand slots、halfScale/pairA、56次K128更新及全部有效行保持；每个compute wave仍在原局部LDS staging payload/ScaleA，payload使用3D TDM `(4,2,256)`，global M16 strides为`(4*global_M16_stride,global_M16_stride,1)`，ScaleA使用M32 stride乘2的2D TDM。全局布局保持canonical，输出LDS不跨两组compute waves互写。

CPU逆permutation和TDM destination审计通过257个row边界、2304个pair operands、917504个input payload bytes、57344个input scales、786432个output payload及49152个output scales，无覆盖缺失或重叠。审计为`audit_interleaved_m32_stride.py`、`interm32_stride_mapping_audit.json`；这不是device精度证明。新实验源码SHA：wrapper=`4cf7ab1747de1df88862817867cf378d547eb7c4da1bd84e71b9ca3a0e05391f`，persistent=`aba805e211487439fe47bec6021437e09b21d6acbba8a9d43e3be6b0a0c5047c`，compiler=`77d8e7d113806c22cfe076392bb4ed3e5ccc58a109604a47dba3e31d1af0c4ec`，完整源码保存于`interm32_stride_source/`。

关闭实验的控制ELF仍与M32默认完整一致；交错candidate为LDS282624 B、VGPR486/SGPR70、zero spills/private、4461 instructions、ELF=`01b403928ad0c8706658b4e7fe1c97c67e280e8a2d5ae194f7756587b6887095`。seed0/17各rc0、未超时、28条零字节差异，HEAD全部有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary及三组dynamic replay通过。此前交错M32失败候选的根因没有唯一定位；本次新候选通过不能直接写成修复了旧bug。

固定默认→交错→默认顺序，每组完整三轮，仅ownership污染允许整组重试，labels前缀`a07_interm32_stride`：

| label后缀 | GEMM1三轮（μs） | 有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_try1` | 110.957,112.097,108.138 | 110.957 | 93.6191,93.6824,94.2853 | 260.726,261.307,258.257 | 三轮合格 |
| `1_perf_a_try1` | 116.540,118.102,118.796 | 118.102 | 94.1147,93.8576,93.1912 | 267.005,268.446,268.483 | 三轮合格 |
| `0_control_b_try1` | 108.920,108.561,108.389 | 不统计 | 94.6945,93.6636,94.3130 | 259.713,258.645,258.655 | 第2轮污染 |
| `0_control_b_try2` | 108.871,109.515,109.232 | 109.232 | 94.1125,93.2335,93.2987 | 258.809,258.193,258.518 | 三轮合格 |

交错candidate比前后有效控制慢6.44%/8.12%，不采用；全部rc0、未超时，队列结束后整机idle、无本轮残留。完整摘要为`interm32_stride_complete_summary.json`。

### 交错输出路径的独立隔离与ATT

保留上述交错compute但恢复原连续2D TDM output，compute waves把各自M32 pairs写入原M128/N64-blocked canonical LDS staging；原WG/DS fence后由原owner输出完整连续slice。CPU覆盖/目的地址审计通过257个row边界、49152个payload producer bytes、3072个scale producer bytes、786432个payload consumer及49152个scale consumer，无重叠。`interm32_canonical2`编译为LDS282624 B、VGPR486/SGPR70、zero spills/private、4498 instructions、ELF=`4990dd06633385a18b681814308d6a3b058b19ad3ff7ad9a752fc5952dfa5dfb`。但seed0首次eager replay的有效payload差16454 B，ScaleA差0 B，首差异grouped_row200/byte0；rc1、未超时，停止队列，无seed17或性能计时。源码保存于`interm32_canonical_source/`。

进一步只把candidate输出arena移到input ring之外的LDS offset282624 B：`interm32_arena3`编译为LDS300032 B、VGPR486/SGPR70、zero spills/private、4526 instructions、ELF=`e3474e887b86ef295da7fc8134d51e2f37fdbf353d62fbc7e6e52252d932643b8`。seed0首次eager replay仍有payload2460031 B差异，ScaleA也失败，首payload差异grouped_row160/byte0；rc1、未超时，停止，无性能数据。存储区域分离不足以消除该candidate的错误，现有证据不能唯一解释为输入/输出alias、某个TDM FIFO或cache；没有删除同步、放宽字节标准或采用失败候选。完整源码保存于`interm32_arena_source/`，失败摘要/日志完整保留。

已通过精度但较慢的strided candidate在a07/dsr1_ep4使用原`get_att_b_rocm_sdk.sh`采集SIMD0/fourSE、random目标、diagnostic iters2，rc0、未超时、ownership合格。四SE raw `.att`、decoded wave、symbol/PC验收通过，actual code object id22与candidate ELF完整一致；记录为`interm32_stride_att_acceptance.json`，UI为`e65_g1_interm32_stride_att_20261017/thread_trace/simd0/kernel/rpf_v3/ui_output_agent_19137_dispatch_58`。

66个sampled waves中30个跳过，36个有效waves均896 WMMA；未采到448/672组，不能据此断言目标全部partial experts的行为。该组entry/compute/epilogue span P50分别5617/23604.5/4967.5 cycles，compute WGwait累计1255.5、DSwait2157.5 cycles，epilogue WGwait514、TDMwait1119 cycles。默认的896组对应4233/23454/4337.5 cycles、compute WGwait1235.5/DSwait1857.5，epilogue WGwait49/TDMwait1090.5。完整计算组compute接近，entry与epilogue更长；采样wave和批次不同，不能直接相减预测kernel μs，不能从该数据唯一证明strided output是全部回退原因。ATT周期不作为新的正式kernel性能。

### M32分段prefetch的compiler scheduling hints实验

默认M32的原closing cover为12个physical WMMA，next-A已逐pair加载，原提示仍按较少WMMA slots安排未来DS reads。新默认关闭参数`PREFETCH_MMA_COVER=0/4/8`只调整`emit_hints`的WMMA/DS调度窗口，不改任何TDM/DS/WG hardware fences、input/output地址、有效routes、K128更新及顺序、数值或输出布局。默认-1控制保持原12-cover路径。

四项compile均LDS282624 B、VGPR486/SGPR68、zero spills/private。控制ELF完整等于已验收M32；0/4/8候选分别为4443 instructions、ELF=`a0bcc9bd8a33a11b271a825a7425a3fb2485d82a8a936c0c02488911a5d1f038` / `50126bfc8a7e5fd214b08c0a776ce73e23405cb9654456c3e5796652f721fe02` / `e3edc4a8be11975a2c4a4c7d774845ea7bc5fbb6e4806e042bda9be81db91a68`。各候选先独立双seed严格验收，再固定控制→0→4→8→控制三轮计时。脚本为`queue_m32_prefcover_compile.sh`及`queue_m32_prefcover_verify_perf.sh`；正式结果待本队列完成，不能把编译指令变化或初步单轮耗时写成已取得收益。

### prefetch cover的完整正向和反向复测

0/4/8三个候选各seed0/17的六项正确性验证全部rc0、未超时，每项28条`differing_bytes=0`。HEAD全部有效payload/ScaleA/GEMM2/MoE输出、各三次eager/graph canary和三组dynamic input/psum replay通过；这些correctness-only运行的计时不作为性能。完整摘要为`m32_prefcover_complete_summary_v1.json`。

下列顺序在运行前固定，正向为控制→0→4→8→控制，反向为0→控制→0。只因ownership污染整组三轮重试，每组最多三次，不按耗时选择重试或拼中值。label前缀为`a07_m32_prefcover`：

| label后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `base_control_a_try1` | 110.284,109.251,108.998 | 109.251 | 94.8736,92.9782,93.8597 | 261.857,257.995,257.643 | 三轮合格 |
| `0_perf_a_try1` | 107.602,107.997,109.534 | 107.997 | 92.1622,95.0844,93.9734 | 254.835,258.872,258.678 | 三轮合格 |
| `4_perf_a_try1` | 110.786,108.623,109.215 | 109.215 | 94.2453,94.0704,94.1735 | 261.048,259.022,258.954 | 三轮合格 |
| `8_perf_a_try1` | 109.175,108.899,108.420 | 不统计 | 93.4484,93.9777,93.8807 | 258.485,258.508,258.341 | 第3轮污染 |
| `8_perf_a_try2` | 108.657,109.896,108.863 | 108.863 | 93.4969,93.1807,93.2059 | 258.436,258.816,257.102 | 三轮合格 |
| `base_control_b_try1` | 109.428,110.027,108.968 | 109.428 | 92.7507,93.4885,94.5364 | 258.404,259.832,260.279 | 三轮合格 |
| `0_perf_b_try1` | 110.373,109.070,109.755 | 不统计 | 93.8299,93.1173,94.5364 | 259.672,257.782,260.302 | 第2轮污染 |
| `0_perf_b_try2` | 107.585,106.721,107.916 | 107.585 | 93.6148,92.9591,93.1155 | 256.964,254.453,257.344 | 三轮合格 |
| `base_control_c_try1` | 107.522,108.551,109.494 | 108.551 | 94.4878,94.7621,93.5769 | 256.992,259.194,258.910 | 三轮合格 |
| `0_perf_c_try1` | 107.252,108.035,108.659 | 108.035 | 93.4568,90.0855,94.2265 | 256.975,254.321,258.613 | 三轮合格 |

三十轮全部rc0、未超时，两批队列结束。cover0首组比前后控制低1.148%/1.308%；反向第二组比前后控制低1.684%/0.890%；末组比相邻前置控制低0.475%。收益幅度较小但各完整相邻组方向一致，因此只把cover0接入原目标守卫的默认值，4/8不接入默认。106.721 μs仅为单轮，不作为默认或最佳稳定结果；此前独立控制组107.216 μs也不替代原命令默认验收。

目标默认新增`PREFETCH_MMA_COVER=0`，依赖`N_TASKS=1`、rowtail3、halfA、halfScale和pairA；任一依赖关闭时cover默认回到-1，其他几何仍-1。CPU实际dispatch block及GEMM2/公共helper/benchmark/routing源码审计通过。wrapper SHA=`ea7bff582b04d5d63cb8ac27547f7ef5ede05cfd935456157761b1c55fe581c1`，persistent SHA=`c2aa42064ae7226839e7fe00df34a3faeea0f95f5e2b1a9e9d9256c978bfc590`，compiler SHA=`d2bb9e15d3a535e6b0c5cf0cd6a7b14f4f35c9c6be5cf2cc87d32210bf122236`，两端SHA一致。`queue_m32_cover0_default.sh`先验证两个seed及T1537边界，再以新默认→旧cover→新默认顺序独立原命令验收，完整性能结果待补齐。

### cover0独立默认验收与撤回

两项T1536 seed0/17及T1537 seed17均rc0、未超时，各28条零字节差异，全部有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary和三组dynamic input/psum replay通过。T1537的有效payload14164992 B、ScaleA885312 B、GEMM2198309888 B、MoE22034432 B均与原HEAD参考逐字节一致，原MoE精度为`3.7789e-06/2.7491e-03`。这些correctness-only结果不作为性能。

| 固定顺序（label前缀`a07_m32_cover0_`） | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| `default_a_try1` | 109.922,108.300,110.700 | 109.922 | 94.3951,93.8469,94.6879 | 259.600,257.900,259.728 |
| `old_cover_a_try1` | 108.349,108.844,108.870 | 108.844 | 94.2264,92.9613,93.3578 | 258.765,258.642,258.927 |
| `default_b_try1` | 111.417,109.137,107.539 | 109.137 | 94.8338,93.2251,92.6304 | 262.217,258.615,255.628 |

九轮均rc0、未超时、ownership合格；两组cover0中值比相邻旧cover慢0.990%/0.269%。独立默认验收未复现前面约0.5%–1.7%的小幅收益，因此撤回cover实验的生产参数与默认选择，完整源码保存在`m32_cover0_before_withdrawal/`，不继续按耗时重试挑选更快批次。保留默认仍为顶部109.148 μs，未更新成107.585 μs或任意单轮。全量摘要=`m32_cover0_default_complete_summary_v1.json`。

cover0在a07/dsr1_ep4用原`get_att_b_rocm_sdk.sh`抓取SIMD0/fourSE，rc0、未超时、ownership合格，无live post owners。raw/decoded/symbol/PC验收通过；actual code object id22与独立compile ELF完全相同，SHA=`a0bcc9bd8a33a11b271a825a7425a3fb2485d82a8a936c0c02488911a5d1f038`，UI=`e65_g1_m32_cover0_att_20261017/thread_trace/simd0/kernel/rpf_v3/ui_output_agent_1838_dispatch_58`。记录=`m32_cover0_att_acceptance.json`。

56个sampled waves中20个跳过，448/672/896组各10/6/20个有效waves，missing PCs为空。三组compute span P50为20838/21544/21542 cycles，compute WGwait为2703/1499/928.5、DSwait为844.5/769.5/1899.5 cycles。896组epilogue P50为3994.5 cycles，WGwait53.5、TDMwait992.5。不同采样组和批次不可直接相减预测μs；较短采样compute span没有转化为独立完整benchmark的稳定收益，不能用ATT替代正式性能。

### FP4 LDS stores合并与输出padding裁剪候选

基于epilogue及output wait观测，新增默认关闭的两个独立参数。`PACKED_QUANT_STORE=1/2`在原SiLU、BF16 RNE、native FP4转换之后，以identity selectors的`permlanex16`交换lane与lane xor16的两段halfwords，lane0..15合并为每行两个b64或一个b128 LDS store。原4条全wave b16 stores的全部字节保持canonical；exchange在mask EXEC之前完成，不改scale、输入、计算、累加顺序或任何同步。`ROW_TAIL_QUANT_OUTPUT=1`仅将TDM output的tensor维度限制到当前真实行数对应的M16/M32组，tile及source pitch保持；每个部分有效组仍完整写出，只有整个padding组被OOB排除。

硬件依据：Shader Guide §4.6.9（p137）/CDNA5 ISA §7.10的`V_PERMLANEX16_B32`从对侧16 lanes取数；Shader Guide §4.10.1（p197–202）的tensor_dim可表示sub-portion、tile超出tensor_dim产生OOB。文档事实支持这些地址与数据操作，不证明device精度或性能。CPU审计覆盖257个row边界、4210688个payload bytes和2236928个有效payload/scale bytes，确认b64/b128 packing逐字节一致，全部有效组保留。审计=`audit_m32_quant_output.py`/`m32_quant_output_mapping_audit.json`。

当前两端源码SHA相同：wrapper=`46850bd7243291daa7be01f3f2f8285959591e6206a73394e73ef5dd40fce6f1`，persistent=`1c0ee6be418d2c40cb3756d42887e62b4c1661949fdb939cb2f1ce359d186e55`，compiler=`e6abef6b83736f5837fafc5945d4a8f7bb8a0cd8786c4a86ad1d46e83f67cc5e`。CPU source/dispatch及py_compile通过，默认及fallback参数保持。依次compile控制0/0、b641/0、b1282/0、OOB0/1、组合2/1，先验实际ISA与resources，再独立双seed精度和相邻性能；尚无新device精度或性能结论。

### 合并输出的实际compile验收

第一次b64 candidate的compile rc1：native转换返回的`ArithValue`没有`.ir_value()`接口。只将该返回值包装为`fx.Int32`后再调用`permlanex16`，使用独立`_v2`编译目录，失败记录和v1源码完整保留，没有GPU dispatch或性能数据。修订persistent SHA=`d25f6478e03181d6bd55bbf5828bc17cc970bb2f9cf9bb2a4793dc646d3bbb9b`，其余两份源SHA保持，两端相同；v2源码保存于`m32_quant_output_v2_source/`。

以下五份actual ELF均LDS282624 B、VGPR486/SGPR68、zero spills/private。静态WMMA256条、DS b128 loads384条、WG signal/wait各10条、input TDM20条、output TDM2条、native FP4 conversion16条全部保持。控制ELF完整等于已验收M32默认；审计=`m32_quant_output_compile_acceptance.json`。

| packed store / output tail | instructions | payload LDS stores | permlanex16 | ELF SHA256 |
|---|---:|---|---:|---|
| 0/0控制 | 4445 | 32条b16 | 8 | `7efb631594f354e56a06f2e23d51fc061cf38411b7656dd65256d7988453140a` |
| 1/0 | 4604 | 16条b64 | 24 | `9de65ffb3c094f2472ba3c4fb438ceef63712274b8847852e74121c1e511d404` |
| 2/0 | 4598 | 8条b128 | 24 | `c38114378d05bcadd2dbf7dbc83fcf649d6ca884975c025837274c19aaa4fefd` |
| 0/1 | 4487 | 32条b16 | 8 | `c75edf75b0644122d3202b5487c7ec4fffa6f7f1c00fa699499aff904e53c0e6` |
| 2/1 | 4643 | 8条b128 | 24 | `04bd5ccd33366dedb5e0cb3d826f6e986320cb4e934da84ceb04f3b0855e638b` |

减少LDS store条数增加了cross-lane和packing指令，不依据条数直接断言性能收益。四项candidate各两个seed，任一精度失败使队列停止；全部通过后才按控制→b64→b128→控制→OOB→组合→控制固定顺序做三轮原benchmark计时。脚本=`queue_m32_quant_output_verify_perf.sh`；只因ownership污染整组最多重试三次。

### 输出合并和padding裁剪的双seed及首轮完整性能

四项候选各seed0/17均rc0、未超时，每项28条零字节差异；HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary及三组同地址dynamic input/psum replay通过。正确性诊断没有使用耗时。固定顺序结果如下，label前缀为`a07_m32_qstore`，后缀包括`_v2_try1/2`；全量摘要为`m32_quant_output_complete_summary_v1.json`。

| 固定顺序（packed/tail） | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| 0/0控制a | 109.776,108.347,109.814 | 109.776 | 93.3767,92.8685,92.9107 | 258.411,256.674,258.383 | 三轮合格 |
| 1/0 b64 | 110.710,110.550,109.683 | 110.550 | 94.3634,92.1665,93.3050 | 261.965,258.807,258.735 | 三轮合格 |
| 2/0 b128 | 109.576,109.411,108.574 | 109.411 | 95.1732,93.6425,94.1210 | 260.075,259.501,259.460 | 三轮合格 |
| 0/0控制b | 109.979,108.300,111.992 | 109.979 | 94.0850,93.8425,93.7036 | 260.211,258.187,261.875 | 三轮合格 |
| 0/1 OOB try1 | 109.955,109.232,107.722 | 不统计 | 93.3324,93.6530,92.9023 | 259.167,259.364,256.941 | 第2/3轮污染 |
| 0/1 OOB try2 | 108.781,108.329,108.005 | 108.329 | 92.6576,94.0894,94.6355 | 257.346,257.864,258.544 | 三轮合格 |
| 2/1组合 | 111.825,110.236,112.260 | 111.825 | 95.6518,94.8231,93.3242 | 263.600,261.417,262.240 | 三轮合格 |
| 0/0控制c | 109.635,108.646,109.700 | 109.635 | 94.8106,94.3276,95.3269 | 260.226,258.802,260.361 | 三轮合格 |

24轮均rc0、未超时。b64慢约0.52%–0.71%，组合慢约1.68%–2.00%，不采用。b128仅快0.33%–0.52%，OOB快1.19%–1.50%；先按OOB→控制→b128→控制固定顺序反向确认。每组三轮仅按ownership重试最多三次，不根据耗时延长或挑样本；未更新生产默认或109.148 μs的独立验收。反向脚本=`queue_m32_quant_output_reverse_v1.sh`。

### 保留默认的T1537边界验证

`a07_m32_retained_t1537_seed17_v1`在两项输出实验参数均0、旧cover及M32保留组合下运行，rc0、未超时、无live post owners。每项28条零字节差异，各三次eager/graph canary与三组dynamic input/psum replay通过。有效payload14164992 B、ScaleA885312 B、GEMM2198309888 B、MoE22034432 B与绑定原HEAD参考`e65_t1537_generic_quant_head_seed17.pt`一致；MoE精度为`3.7789e-06/2.7491e-03`。它是正确性证据，不是新增性能数据。

### 基于运行时psum的M-task排列候选

默认关闭`TASK_M_ORDER=1/2`分别把最后一个expert的M tiles提前、将所有有效M tiles逆序。有效范围由当前psum最后两项计算，不硬编码shared expert counts、seed或数据值；原contiguous-M swizzle和N cluster保留，容量尾部tiles不变。全部有效(M,N) tasks各执行一次，输入/输出不移动，K128更新、数值路径和同步均保持。希望检查较长shared-expert tiles位于原grid末段的尾部影响，不从任务顺序直接宣称硬件调度改善。

硬件依据：Shader Guide §3.5（p53）记载CS waves的typewriter创建顺序，以及同WG/cluster初始grid状态。这不确定实际WGP分配或kernel关键路径，仍需正式相邻测试。CPU审计直接执行生产AST中的permutation算术，仅替换pointer接口，覆盖477种空expert、邻近expert数、row边界及dynamic counts拓扑，113526个permutation points、1816416个cluster swizzle points全部通过；每个有效tile唯一覆盖，capacity sentinel不移动。记录=`audit_m32_task_m_order_v1.py`/`m32_task_m_order_mapping_audit_v1.json`。实际编译和device验证随后进行，参数尚未成为默认。

### 输出候选的反向结果与撤回

反向顺序在运行前固定为OOB→控制→b128→控制。OOB的三次完整组三轮均被ownership污染，没有合格反向中值；不再追加超过预定次数的复测，不从第三组两条合格单轮拼中值。label前缀为`a07_m32_qstore`：

| label后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_qtail1_perf_b_reverse_v1_try1` | 108.418,109.565,110.702 | 不统计 | 93.1807,94.0662,94.5344 | 256.555,257.886,261.257 | 三轮污染 |
| `0_qtail1_perf_b_reverse_v1_try2` | 109.164,109.312,108.650 | 不统计 | 93.9333,94.0746,92.9845 | 258.837,259.168,256.077 | 三轮污染 |
| `0_qtail1_perf_b_reverse_v1_try3` | 109.184,109.513,109.755 | 不统计 | 92.7274,91.4285,93.4821 | 256.313,256.713,259.431 | 第1轮污染 |
| `0_qtail0_control_d_reverse_v1_try1` | 109.196,110.312,107.824 | 109.196 | 93.6572,93.5729,93.2271 | 258.683,260.589,256.604 | 三轮合格 |
| `2_qtail0_perf_b_reverse_v1_try1` | 110.407,110.470,109.605 | 110.407 | 94.8444,93.7541,93.9333 | 262.052,260.117,260.394 | 三轮合格 |
| `0_qtail0_control_e_reverse_v1_try1` | 107.978,106.607,110.508 | 107.978 | 94.2010,93.7393,94.1041 | 257.594,256.614,260.588 | 三轮合格 |

b128反向比相邻控制慢1.11%/2.25%，首轮不足0.6%的改善没有复现。b64、b128和组合无稳定收益；OOB没有完整反向证据，因此四项输出候选均不采用。全部rc0、未超时，原始污染和合格记录都保留在`m32_quant_output_all_summary_v2.json`。实验参数与对应epilogue分支撤出生产，完整v2源码和带排列实验的源码分别保存于`m32_quant_output_v2_source/`、`m32_task_m_order_v1_source/`。

### M-task排列的actual ELF、双seed和正反向性能

三份实际compile均LDS282624 B、VGPR486/SGPR68、zero spills/private。order0的ELF完整等于已验收M32默认。WMMA256、DS b128 loads384、DS b32 loads42、WG signal/wait各10、input TDM20、output TDM2、native FP4 conversion16和permlanex16 8条均保持。记录=`m32_task_m_order_compile_acceptance_v1.json`。

| order | instructions | actual ELF SHA256 |
|---:|---:|---|
| 0 | 4445 | `7efb631594f354e56a06f2e23d51fc061cf38411b7656dd65256d7988453140a` |
| 1 | 4482 | `ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d` |
| 2 | 4464 | `e4f56e850f2c2236214157b6c73e0003d409b0448c0bd6976835eba022e8324b` |

第一次compile控制队列在order0成功后因host无权chmod容器root生成文件停止；没有compile失败或dispatch。修正为在`dsr1_ep4`里chmod输出目录后resume，跳过已完成summary，不重复order0。两个控制日志均保留。order1/2各seed0/17均rc0、未超时，每项28条零字节差异；HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary及三组dynamic input/psum replay全部通过。这些correctness-only结果不作为性能。

正向顺序固定控制→order1→order2→控制，反向固定order1→控制→order1；每组三轮只因ownership污染整组重试，最多三次，不能按耗时挑样本。label前缀为`a07_m32_morder`：

| label后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 107.693,108.874,109.422 | 108.874 | 94.3802,93.9165,93.8300 | 257.981,257.985,259.172 | 三轮合格 |
| `1_perf_a_v1_try1` | 106.795,107.488,106.320 | 106.795 | 95.1141,94.5680,94.9159 | 257.875,258.088,257.470 | 三轮合格 |
| `2_perf_a_v1_try1` | 107.187,108.821,106.086 | 107.187 | 94.6312,93.8932,94.1420 | 257.672,258.640,256.538 | 三轮合格 |
| `0_control_b_v1_try1` | 109.825,108.093,107.109 | 108.093 | 94.7155,94.5406,93.8786 | 259.918,259.184,258.914 | 三轮合格 |
| `1_perf_b_reverse_v1_try1` | 107.594,108.047,108.169 | 108.047 | 94.4435,93.8426,94.3676 | 257.411,258.624,257.668 | 三轮合格 |
| `0_control_c_reverse_v1_try1` | 107.562,109.150,109.124 | 不统计 | 93.7329,93.8532,94.0175 | 257.329,258.232,258.923 | 第1/2轮污染 |
| `0_control_c_reverse_v1_try2` | 111.657,110.902,109.753 | 不统计 | 93.5600,95.6161,93.4485 | 261.484,261.858,259.290 | 第1/2轮污染 |
| `0_control_c_reverse_v1_try3` | 108.886,107.899,111.007 | 108.886 | 95.4851,94.1190,94.1863 | 260.523,257.103,261.717 | 三轮合格 |
| `1_perf_c_reverse_v1_try1` | 107.261,109.243,107.147 | 107.261 | 94.9411,94.0852,95.1015 | 256.778,259.085,257.829 | 三轮合格 |

27轮全部rc0、未超时。order1首轮比相邻控制快1.91%/1.20%，反向两组比控制快0.77%/1.49%，方向一致；order2只作首轮对照，不扩大复测，撤出生产。order1接入原E65/contiguous_m30464/non-balanced守卫，依赖N_TASKS1与rowtail3；其他geometry以及显式关闭相应依赖时仍选择order0。此时顶部109.148 μs仍是此前独立默认验收，不能用106.795 μs替代新默认尚未完成的验收。

收敛后wrapper SHA=`d8bc366bb26c13c6d307549d2119d100e17c3ef8f45b56c3eb49e5fbec8e67a6`，persistent SHA=`682518c9a6efa6bf4c308b7f7f3a3dcc90401a90800d9cfb40354a0aef090346`，compiler SHA=`8b164cadfa680d6fcdb6a1bdeb71a540bd8451433cd2134cffa28fbaabf7cb25`；本地/a07相同。GEMM2/公共helpers/benchmark/routing和CPU默认dispatch审计通过。独立队列`queue_m32_morder_default_v1.sh`先要求order1、order0、四任务fallback三份ELF完整匹配前述产物，再双seed和T1537 device验收，最后原用户命令新默认→旧顺序→新默认三轮相邻测试。完整结果待队列完成。

### M-task排列收敛版的独立默认验收

`m32_morder_default_compile_acceptance_v1.json`确认去掉全部输出和order2实验后，order1、order0与四任务fallback的实际ELF完整相同于已有验收产物，zero spills/private。`a07_m32_morder_default_seed0_v1`、`seed17_v1`与`t1537_seed17_v1`均rc0、未超时，各28条零字节差异，HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary以及三组dynamic input/psum replay全部通过；correctness-only耗时未使用。

以新默认原命令→旧顺序→新默认原命令固定顺序，每组三轮，九轮均ownership合格、rc0、未超时，没有因耗时追加复测。完整前缀为`a07_m32_morder_`：

| label后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| `default_a_v1_try1` | 107.463,105.886,107.170 | 107.170 | 94.5447,95.3798,94.3995 | 258.487,256.342,257.926 |
| `old_order_a_v1_try1` | 108.347,109.557,108.893 | 108.893 | 93.0500,94.7472,93.9164 | 257.070,259.812,258.132 |
| `default_b_v1_try1` | 107.223,106.936,108.401 | 107.223 | 92.2973,94.0935,94.2517 | 254.489,257.023,257.664 |

两组新默认相对本轮相邻旧顺序低1.58%/1.53%，与初筛和反向一致，因此保留order1。原用户命令默认验收更新为107.170/107.223 μs，距90 μs仍约17.2 μs；105.886 μs是单轮，不能作为达标数据。完整摘要=`m32_morder_default_complete_summary_v1.json`，源码=`m32_morder_default_v1_source/`。队列后整机idle、KFD owners为空，VRAM176582656 B，没有本轮测试残留。

### 两组M128 compute wave priority的有限实验

默认关闭`COMPUTE_WAVE_PRIORITY=1/2`，分别在计算阶段对wave_m0或wave_m1执行`S_SETPRIO 1`，计算后所有waves恢复`S_SETPRIO 0`，再进入原epilogue。全部operands、有效WMMA、owner分配、ring和input/output lifetime fences保持，原计时/routing不变。目标是检查两组M128计算量不一致时的wave进度影响，不由priority设置直接断言硬件critical path或FIFO。

Shader Guide §5.2/5.2.2（p212–213）说明USER_PRIO与wave age参与仲裁，合作waves较接近的进度有助于barrier，并明确警告WG equal-priority模式与不同USER_PRIO会出现不均匀份额；ISA §15.5（p291）的S_SETPRIO将SIMM16低两位设置为user priority。此实验同时比较两组方向，完整三轮相邻控制，任何精度或资源失败都不计时。原130us任务的WG_RR_EN没有稳定收益，本次不重复修改该descriptor位。

实验源码SHA为wrapper=`78c6341db24e83324cbad60c58d686768fb8bef5405c8802de5e08eae9f332db`，persistent=`96b03ba30b344bfc3ed0cd262bdbb0ae04946d9a1c13d64701330a967c73c0c3`，compiler=`9796a2743479628998b182593987ffaeab5121a3178d9c5cbaa77a4a926a64fc`；两端相同。CPU dispatch和GEMM2/公共helper/benchmark/routing源码审计通过。`queue_m32_compute_priority_v1.sh`要求控制ELF完整匹配、有效WMMA/loads/TDM/store/barrier数量保持且候选仅两条S_SETPRIO、zero spills/private；随后双seed严格device验收，再固定0→1→0→2→0原完整benchmark。正式结果待队列完成。

### wave priority的完整验收与淘汰

两份priority产物均LDS282624 B、VGPR486/SGPR70、zero spills/private、4491 instructions；priority1 ELF=`5cfcdfc1a7ca4801038761408755ac53a900d62c8e37d8a957a13fd8c39922b79`，priority2=`4841b7fdc8da35eabecb3d10c8319bb3afffbff512481f5b696f2b0037a96632a`。actual ISA确有两条S_SETPRIO，WMMA、DS loads、TDM、output stores和WG barriers数量保持，控制完整等于新默认。两个方向各seed0/17全部rc0、未超时，每项28条零字节差异以及全部eager/graph/dynamic replay通过。记录=`m32_compute_priority_compile_acceptance_v1.json`。

固定0→1→0→2→0，15轮全部ownership合格，无污染重试、rc0、未超时；label前缀为`a07_m32_cprio`：

| label后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| `0_control_a_v1_try1` | 107.206,108.228,111.870 | 108.228 | 92.5269,95.5338,95.0254 | 255.821,259.621,263.220 |
| `1_perf_a_v1_try1` | 109.609,107.661,107.168 | 107.661 | 94.6249,94.1231,93.4822 | 261.295,258.229,256.180 |
| `0_control_b_v1_try1` | 106.493,110.537,107.290 | 107.290 | 93.9862,94.2600,95.7507 | 256.229,260.886,259.625 |
| `2_perf_a_v1_try1` | 110.560,109.744,108.872 | 109.744 | 95.3798,95.3354,95.6749 | 261.577,260.302,262.056 |
| `0_control_c_v1_try1` | 107.438,107.242,107.214 | 107.242 | 94.0978,94.6017,94.7050 | 256.827,257.715,258.038 |

priority1相比前后控制分别快0.52%/慢0.35%，没有一致改善；priority2慢2.29%/2.33%。两项均撤出生产，不追加按耗时挑选的复测。全部完整摘要=`m32_compute_priority_complete_summary_v1.json`，完整源码保留于`m32_compute_priority_v1_source/`。默认仍为107.170/107.223 μs，不改成控制108.228 μs或任意单轮。

### 完整M128/N64 compute tiles在wave间重排

默认关闭`COMPUTE_WAVE_LAYOUT=1/2`。输入TDM owners保留原physical wave IDs；把完整M128/N64 compute tile及其自己的canonical LDS output slice映射到另一physical wave。layout1的physical→logical序列为`[0,4,1,5,2,6,3,7]`，layout2为`[0,1,4,5,2,3,6,7]`。逻辑wave_m/wave_n负责同一个原tile的A/B/ScaleA/ScaleB、全部56次K128更新、SiLU和FP4量化；payload producer和TDM consumer同时按logical wave计算LDS offset，canonical ScaleA地址保持。没有在wave内部交错M16/M32 groups，不沿用此前较慢的strided输出。

Shader Guide §2.3（p24）记载WG waves分布在4个SIMD且共享LDS，§6.10（p267）记载multiple-of-four WG的predictable SIMD rotation；这支持测试wave映射，不能由logical编号唯一确定当前ATT里每个wave的physical位置或critical path。CPU直接执行生产permutation AST，枚举257种row边界，25264128个有效compute points、6316032个payload及394752个ScaleA points，各有效tile一次覆盖，canonical LDS producer/consumer一致。记录=`m32_compute_wave_layout_mapping_audit_v1.json`。

完整实测仍需actual ISA、zero spills/private、双seed/eager/graph/dynamic验证以及固定0→1→0→2→0原命令相邻测试。实验源码SHA为wrapper=`28b4d9cee960ed531ab5f7974fe63da4bee94868b6b335046671be5669a7145b`，persistent=`4e03191cfe573d90c08d7aa905d97507e752a4ee90cd601c593d6c3a54ef15bb`，compiler=`abbc5ebe09d0149bfc9495c8c2cacb4a07ba2e668413c0592b518ac9f8680828`；性能尚未取得。

### wave tile重排的完整验收与淘汰

actual control ELF仍与已验收M-order默认`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`完整一致。layout1为VGPR486/SGPR70、4498 instructions，ELF=`cba571c7f997d30b0d4463ab78f0081c20e25340d26d677666bb6e6bd75543cff`；layout2为486/68、4477 instructions，ELF=`32a5c00e24e18e943a9fdcc2d7c2a84bea10996360ca777aecf5e3f1fb99de02b`。两项均LDS282624 B、zero spills/private，WMMA、DS loads、TDM、output stores和WG barriers条数保持。记录=`m32_compute_wave_layout_compile_acceptance_v1.json`。

两种布局各seed0/17四项device验证均rc0、未超时，每项28条零字节差异；HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay通过。正确性诊断中的ownership=false不用于性能。

固定0→1→0→2→0原完整benchmark的合格组如下，label前缀为`a07_m32_cwl`：

| label后缀 | GEMM1三轮（μs） | 完整有效中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| `0_control_a_v1_try2` | 112.323,107.389,107.301 | 107.389 | 94.6124,94.0641,95.4326 | 263.385,257.671,257.595 |
| `1_perf_a_v1_try1` | 109.215,110.411,108.136 | 109.215 | 94.2453,94.6165,94.9032 | 259.361,260.734,258.567 |
| `0_control_b_v1_try1` | 107.619,106.985,107.010 | 107.010 | 94.9728,93.3978,96.0818 | 258.185,255.621,259.573 |
| `2_perf_a_v1_try1` | 110.951,111.667,112.013 | 111.667 | 95.6941,93.8744,94.5469 | 261.904,261.248,262.733 |
| `0_control_c_v1_try1` | 107.332,107.849,106.704 | 107.332 | 95.0107,94.7811,95.7444 | 257.776,258.944,258.223 |

首个控制`0_control_a_v1_try1`的107.142/106.462/106.801 μs全部ownership不合格，整组不统计中值；按预先限定最多三次重试完成try2，不按耗时重试或拼接单轮。其余15轮全部合格、rc0、未超时。layout1比前后控制慢1.700%/2.061%，layout2慢4.352%/4.039%，均撤出生产，不追加性能筛选。完整摘要=`m32_compute_wave_layout_complete_summary_v1.json`，实验源码保留于`m32_compute_wave_layout_v1_source/`。结束后整机idle、KFD owners为空、VRAM176582656 B，无本任务GPU进程残留。

恢复独立已验收的M-order默认三份源码，默认仍为107.170/107.223 μs。后续验证连续M32区间按当前有效行数分给两组compute waves：输入TDM和256行parent expert alignment保持；每组自己的连续canonical output通过tensor_dim裁掉另一组拥有的行，避免新增strided output或跨wave LDS producer。先审计实际地址、完整行覆盖和output范围，再做actual ISA与device验收；当前没有这项候选的性能结果。

### runtime连续M分配的精度与首轮完整对照

默认关闭的`AITER_GFX1250_GEMM1_RUNTIME_M_SPLIT=1`使用当前parent的`mn_oob`，将`ceil(clamp(mn_oob,0,256)/32)`个M32区间连续分给两个M128 compute halves。第一组从row0开始，第二组从`ceil(live_pairs/2)*32`开始；A和ScaleA的LDS基址、WMMA/prefetch谓词及global output基址一起跟随真实起点。每组仍向自己的连续私有LDS output slice写入，通过运行时`tensor_dim`限制payload M16和ScaleA M32范围，两个区间没有重复有效行。256行parent、expert alignment、physical TDM owners、input ring和全部同步保持。没有使用固定seed、expert counts或减少有效route。

Shader Guide §4.10.1（p197）记载TDM同wave有序、跨wave及相对其他memory无序；§4.10.3 descriptor（p202–203）允许tensor_dim表示sub-portion；§4.10.7（p205）说明超出tensor_dim的store dropped。它们支持地址/生命周期设计，不能替代device验收。CPU审计直接执行实际split/output extent AST，覆盖257种row边界、8421376 compute points、2105344 payload points、131584 ScaleA points及9216 operand points，确认全部有效行唯一覆盖、canonical producer/consumer一致。

actual control ELF完整等于已验收默认`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`，4482 instructions。candidate ELF=`cd845ff099a78baf80d11e496214e004d65513e48b30ec5713f9523e31a596e3`，4539 instructions。两项都是VGPR486/SGPR68、LDS282624 B、zero spills/private，有效WMMA、DS、TDM、output stores及barrier条数保持。compile验收=`m32_runtime_m_split_compile_acceptance_v1.json`。

`a07_m32_rmsplit1_seed0_v1`及`seed17_v1`均rc0、未超时，各28条零字节差异：原HEAD全部有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary及三组同地址dynamic input/psum replay通过。固定0→1→0正式原命令结果如下：

| label（前缀`a07_m32_rmsplit`） | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| `0_control_a_v1_try1` | 107.990,107.642,107.050 | 107.642 | 94.7029,95.6918,93.5748 | 259.189,258.957,255.979 |
| `1_perf_a_v1_try1` | 107.786,107.233,107.821 | 不统计，第2轮污染 | 94.0514,93.3682,94.6669 | 256.275,256.140,258.309 |
| `1_perf_a_v1_try2` | 106.715,106.213,110.406 | 不统计，第2/3轮污染 | 94.0154,91.7153,94.8671 | 254.619,253.583,260.670 |
| `1_perf_a_v1_try3` | 107.916,106.558,106.959 | 106.959 | 95.0572,93.8574,94.1989 | 258.525,256.144,257.314 |
| `0_control_b_v1_try1` | 109.342,106.972,108.258 | 108.258 | 94.5996,94.9095,94.2580 | 258.520,257.827,258.149 |

十五轮均rc0、未超时。两次污染candidate整组排除，按预先限定最多三次规则重试，不拼接单轮。合格candidate比前后控制快0.6345%/1.1999%，仍需固定1→0→1反向对照及独立默认验收；当前默认保持107.170/107.223 μs，未达到90 μs。全部首轮结果=`m32_runtime_m_split_complete_summary_v1.json`；结束后整机idle、KFD owners为空、VRAM176582656 B，无本任务GPU进程残留。

### 下一项epilogue候选：tied rounded-vector reuse

旧130us记录在保留vector SiLU/F32 amax的条件下，以空LLVM asm `"", "=v,0", has_side_effects=False`复用已rounded BF16向量，曾取得约1.99%的GEMM1收益。本任务早期`compile_nt1_roundreuse`也减少过转换指令，但device等待被占用中断，没有性能结论。当前M32/M-order版本的actual ISA有176条静态`v_cvt_pk_bf16_f32`，因此在独立候选快照中准备默认关闭的`AITER_GFX1250_GEMM1_ROUND_REUSE=1`。

candidate仅将已完成BF16 rounding的vector bitcast为Int32，以无副作用tied identity供原F32 amax及native FP4两个consumer复用。SiLU、F32 amax树、E8M0、FP4、全部输入/有效WMMA/累加顺序、output布局和同步保持。实际epilogue AST剥离identity后与已验收M-order默认完全一致；GEMM2/helper不变。审计=`m32_round_reuse_source_audit_v1.json`，候选源码=`m32_round_reuse_v1_source/`；此时仅CPU准备完成，尚未编译、GPU验证或性能计时。反向runtime split对照结束前不替换远端生产源码，也不并发编译或传大文件。

### runtime M split的反向确认与撤回

固定1→0→1反向对照完成。下列七组全部保留，label前缀为`a07_m32_rmsplit`；只按ownership整组三轮最多重试三次，不从污染组选择单轮：

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `1_perf_b_reverse_v1_try1` | 107.303,107.760,107.642 | 不统计 | 94.8736,93.8826,95.6854 | 258.708,257.247,259.952 | 第2/3轮污染 |
| `1_perf_b_reverse_v1_try2` | 107.625,107.737,105.521 | 不统计 | 94.0135,94.6078,94.9454 | 256.279,258.164,256.146 | 三轮污染 |
| `1_perf_b_reverse_v1_try3` | 106.833,107.501,105.553 | 106.833 | 94.6671,96.1071,95.9110 | 257.738,259.082,257.099 | 三轮合格 |
| `0_control_c_reverse_v1_try1` | 107.104,110.984,107.792 | 不统计 | 95.3965,95.0720,95.2870 | 258.115,261.858,258.651 | 第3轮污染 |
| `0_control_c_reverse_v1_try2` | 108.367,107.358,107.450 | 不统计 | 94.1018,93.8447,94.2306 | 257.930,258.024,257.006 | 第1/2轮污染 |
| `0_control_c_reverse_v1_try3` | 107.022,106.862,106.753 | 106.862 | 94.4794,93.4589,94.3571 | 257.668,256.087,257.253 | 三轮合格 |
| `1_perf_c_reverse_v1_try1` | 107.379,107.069,106.312 | 107.069 | 94.4773,95.1625,95.4809 | 258.107,258.339,258.535 | 三轮合格 |

二十一轮全部rc0、未超时。合格candidate_b仅比后置控制快0.0271%，candidate_c比前置控制慢0.1937%，未复现首组0.63%–1.20%的稳定收益。因此不采用runtime M split，不追加按耗时筛选。完整源码双端保存于`m32_runtime_m_split_v1_source/`，生产参数已撤回，接下来的round-reuse以精确已验收M-order默认为控制。完整正反向摘要=`m32_runtime_m_split_forward_reverse_summary_v1.json`。结束后整机idle、KFD owners为空、VRAM176582656 B，无自身GPU残留；GPU外部任务导致的自动等待不计入测试时长。

此前未回收的wave-layout等增量证据已下载本地：`m32_wave_layout_new_evidence_v1.tar.gz`，SHA256=`270be46567dcb9067178ad2243d02fdbd16c5436e5a56926a4677ce275f40035`，2755份regular files全部逐文件核对SHA256通过；验收=`m32_wave_layout_evidence_unpack_acceptance_v1.json`。下载仅在反向队列启动前进行，没有并发性能计时。

### rounded-vector复用的actual compile与数值边界

控制ELF仍完整等于默认`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`。candidate ELF=`34f4e7e4689c5f2413ff4a17fca57edf17cfeb70c6f272ae66e8c46948e785e3`，instructions4482→4433，静态`v_cvt_pk_bf16_f32`176→64。两项都是VGPR486/SGPR68、LDS282624 B、zero spills/private。WMMA、DS payload/scales、TDM、output stores、WG barriers/tensor waits、native FP4 conversion、packed F32 add/mul及tanh条数保持，编译验收=`m32_round_reuse_compile_acceptance_v1.json`。减少转换指令本身不证明kernel性能提升。

独立correctness-only `verify_m32_round_reuse_edges_v1.py`直接从当前kernel AST提取实际tied identity。六组覆盖全部65536 BF16 raw编码、每个BF16值的F32 rounding边界0x7fff/0x8000/0x8001、65536行random F32 bit patterns及12800行mixed signed zero/subnormal/NaN/Inf peer16，合计5447680个F32输入值；rounded BF16、E8M0/forward scale、native FP4全部16343040 B逐字节一致。rc0、未超时，无性能数据。验收=`m32_round_reuse_edge_acceptance_v1.json`。

candidate源码双端SHA：wrapper=`99950c53f596acb72fce348940aa04f81ebdd32b9665aa2607b48c75bf3630e3`、persistent=`6c42ca6c495778a42c1726eee342f12bd04d8f83418957ca3dca54f6b05d855c`、compiler=`3b617211780ed72f31f7eca78eba95f31eb4acac630a96849f4187bcc2cd05aa`。生产默认`ROUND_REUSE=0`，原benchmark、routing及精度门槛保持。下一步完成两个random seed严格device/replay，再固定0→1→0三轮原命令对照；完整结果尚待补齐，默认性能仍为107.170/107.223 μs。

### rounded-vector复用的双seed与首轮完整性能

`a07_m32_roundreuse1_seed0_v1`及`seed17_v1`均rc0、未超时，每项28条`differing_bytes=0`。原HEAD全部有效payload14155776 B、ScaleA884736 B、GEMM2198180864 B、MoE22020096 B、各三次eager/graph canary与三组同地址dynamic input/psum replay通过；MoE原精度分别为`3.7793e-06/2.7493e-03`和`3.7788e-06/2.7491e-03`。这些correctness-only运行不提供性能结论。

固定0→1→0首轮原命令完整结果如下，label前缀为`a07_m32_roundreuse`：

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| `0_control_a_v1_try1` | 108.328,110.590,107.442 | 108.328 | 94.6988,92.8621,94.4773 | 258.797,259.998,258.438 |
| `1_perf_a_v1_try1` | 106.029,106.843,109.519 | 106.843 | 95.0549,94.1567,95.0361 | 256.807,257.419,260.494 |
| `0_control_b_v1_try1` | 108.131,109.182,107.202 | 108.131 | 94.6609,92.7527,94.8337 | 258.219,258.443,258.231 |

九轮全部ownership合格、rc0、未超时。candidate中值比前后控制快1.3708%/1.1911%，尚未经过反向及独立默认验收；默认保持107.170/107.223 μs。结束后整机idle、KFD owners为空、VRAM176582656 B，无自身GPU进程残留，记录=`m32_round_reuse_post_idle_v1.json`。接下来只执行预先固定1→0→1反向对照，脚本=`queue_m32_round_reuse_reverse_v1.sh`，每组三轮最多三次ownership污染整组重试，不挑单轮或根据耗时继续筛选。

### rounded-vector复用的反向结果与撤回

固定1→0→1反向结果如下，label前缀为`a07_m32_roundreuse`。首个candidate组第3轮ownership污染，按事前规则整组三轮重试一次；保留全部样本，不将污染组单轮合并到其他组。

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `1_perf_b_reverse_v1_try1` | 106.643,108.884,106.527 | 不统计 | 95.4915,94.2873,94.2621 | 258.210,258.862,256.681 | 第3轮污染 |
| `1_perf_b_reverse_v1_try2` | 107.421,106.521,107.499 | 107.421 | 92.1982,93.4106,93.6233 | 255.644,256.559,257.141 | 三轮合格 |
| `0_control_c_reverse_v1_try1` | 109.706,107.271,106.892 | 107.271 | 93.9862,94.8526,92.7444 | 259.440,258.921,256.064 | 三轮合格 |
| `1_perf_c_reverse_v1_try1` | 106.413,106.909,107.024 | 106.909 | 93.9355,94.4541,94.5174 | 256.123,256.349,257.158 | 三轮合格 |

十二轮全部rc0、未超时。合格candidate_b比后置控制慢0.1398%，candidate_c比前置控制快0.3375%；首轮1.19%–1.37%改善没有稳定复现，因此撤回rounded-vector reuse生产参数，不继续按耗时重测筛选。完整候选源码双端保留于`m32_round_reuse_v1_source/`，控制仍是精确已验收的M-order默认107.170/107.223 μs。全量摘要=`m32_round_reuse_complete_summary_v1.json`；结束后整机idle、KFD owners为空、VRAM176582656 B，无本任务GPU进程残留，记录=`m32_round_reuse_reverse_post_idle_v1.json`。

### 同一t256的w4x2 compute分工：精度通过、性能淘汰

以精确已验收的M-order默认为控制，新增默认关闭的`AITER_GFX1250_GEMM1_COMPUTE_WAVE_PARTITION=1`。八个waves仍处理同一完整t256/N256/K256 parent，compute由w2x4变为w4x2，每wave处理M64/N128。physical input TDM owners、input ring、padded input LDS、expert alignment、每个有效accumulator的56次K128更新和顺序及全部必要同步保持。每wave生成两组独立MX blocks，使用自己的连续LDS payload slice并保持TDM consumer同步；ScaleA仍为canonical M32，两组N waves各写两列。

`audit_m32_wave_partition_v1.py`直接执行生产AST的几何和地址，覆盖257种row边界、8421376个compute points、2105344个payload bytes、131584个ScaleA points及3158016个operand范围检查，全部有效区域唯一覆盖。load/WMMA/pipeline嵌套函数及GEMM2/helper/benchmark源码审计通过。candidate快照=`m32_wave_partition_v1_source/`；wrapper/persistent/compiler SHA分别为`c2b7980405e16167e2bf9a28c1586c39a963c33ff4f7371e8011cc7ad73a7f9f`、`1820a9095e4956d739c1594711594c35e923494754dd99f2669dd4fc1532497d`、`c7b411c3b6803d92633540331e143cc80dde7fe65d1644c796397169f08632ce`，两端一致。

actual控制ELF完整等于默认`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`，4482 instructions、VGPR486/SGPR68。candidate ELF=`4cd7c8968eb6f29e137adfbe1737f2f9e7a688384f5115df63d07f0266229bd0`，4255 instructions、VGPR503/SGPR68；两者LDS282624 B、zero spills/private。static WMMA256、DS b128 loads384、input TDM20/output TDM2、WG signal/wait10/10、BF16 conversions176、tanh128、FP4 conversions16及原packed add/mul和output stores保持。DS b32 sites42→56、tensorwait sites27→26来自owner-body lowering，源码硬件同步不变。首次acceptance parser误把disassembler的`/*v[physical]*/`注释当成operand，发生在GPU前；仅修正parser去掉注释后验收通过，未改kernel或重编译。失败parser和日志均保留；验收=`m32_wave_partition_compile_acceptance_v1.json`。

`a07_m32_wpart1_seed0_v1`及`seed17_v1`均rc0、未超时，各28条零字节差异：原HEAD有效payload14155776 B、ScaleA884736 B、GEMM2198180864 B、MoE22020096 B及各三次eager/graph canary、三组同地址dynamic input/psum replay全部通过。correctness-only计时未使用。

原定性能顺序0→1→0，每组三轮、只因ownership污染最多三次整组重试。首个控制三次尝试均污染后停止首段；随后只完成已预定的candidate和后控制，没有第四次first-control，也不拼接单轮。完整label前缀=`a07_m32_wpart`：

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 107.347,107.486,107.870 | 不统计 | 93.2102,94.0684,94.4775 | 255.805,257.869,258.919 | 第3轮污染 |
| `0_control_a_v1_try2` | 110.451,107.254,106.934 | 不统计 | 94.3698,94.0408,93.3831 | 259.370,256.528,254.942 | 第1/3轮污染 |
| `0_control_a_v1_try3` | 107.001,106.951,107.250 | 不统计 | 95.1121,93.4338,94.3742 | 257.898,255.179,257.890 | 第2轮污染 |
| `1_perf_a_v1_try1` | 116.131,115.971,116.831 | 116.131 | 93.6929,94.2031,94.5449 | 266.048,266.776,266.469 | 三轮合格 |
| `0_control_b_v1_try1` | 107.490,106.516,106.765 | 106.765 | 93.8658,95.3735,95.0149 | 256.605,257.418,257.919 | 三轮合格 |

十五轮全部rc0、未超时。candidate比固定后置控制慢8.7725%，没有采用理由，撤出生产并恢复`m32_morder_default_v1_source/`的三份精确SHA；receipt=`m32_wave_partition_withdrawal_v1.json`。first-control结束后整机idle/KFD空、VRAM176582656 B；后续检查出现同容器的外部E64/T1536/topk8任务，不属于本任务残留，未杀进程。恢复生产源码和采集M-order默认ATT前每6秒等待整机空闲。完整摘要=`m32_wave_partition_complete_summary_v1.json`。下一项ATT仅用于阶段和同步诊断，当前正式默认仍为107.170/107.223 μs，未达到90 μs。

wave-partition增量raw证据已回收本地，`m32_wave_partition_new_evidence_v1.tar.gz`为976880 B、114份regular files，SHA256=`824e96651ccb902946e0c0929ec632426ed9abbbbc8196d49e0e0d4cda1ec42e`；全部逐文件SHA通过，receipt=`m32_wave_partition_evidence_unpack_acceptance_v1.json`。

### M-order当前默认的无污染ATT及下一项M32分支候选

`a07_m32_morder_default_att_v1`在a07/dsr1_ep4使用原`get_att_b_rocm_sdk.sh`采集SIMD0、fourSE、random目标workload，diagnostic iters2。rc0、未超过120秒、ownership合格、无live post owners。actual code object id22与独立compile默认ELF完整逐字节一致，SHA=`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`；源SHA绑定到精确M-order默认。raw `.att`、decoded waves、目标symbol和PC完整、无`.long`，验收=`m32_morder_default_att_acceptance_v1.json`；UI=`e65_g1_m32_morder_default_att_v1_20261017/thread_trace/simd0/kernel/rpf_v3/ui_output_agent_3928_dispatch_58`。

60个sampled waves中22个跳过，448/672/896组各5/12/21个有效waves。沿用first/last WMMA issue边界，不相加P50预测kernel μs：

| 动态WMMA数 | entry span P50（cycles） | compute span P50（cycles） | epilogue span P50（cycles） | compute WGwait累计P50（cycles） | compute DSwait累计P50（cycles） |
|---:|---:|---:|---:|---:|---:|
| 448 | 4370 | 20995 | 4055 | 2617 | 856 |
| 672 | 4153.5 | 21599 | 4027.5 | 1326.5 | 739 |
| 896 | 4068 | 21750 | 3979 | 1009 | 2000 |

896组epilogue WGwait/TDMwait累计P50为71/1052 cycles。两处steady WGwait各456次，P50为33.5/51 cycles；入口WGwait38次P50为1094.5 cycles，末尾tensorwait0 P50为973.5 cycles。不同M-order、order0采样wave和批次不相同，不把span差异直接换算性能收益，也不把这些观测唯一归因于bank、cache或FIFO。独立post检查再次看到外部E64任务，不属于本任务残留；原result的即时及settled门禁均合格。

完整ATT包=`m32_morder_default_att_evidence_v1.tar.gz`，29884108 B、152份regular files、SHA256=`5827abae775a985517b4b43976e1ddff46c84e29021adbed1404d8756b609fa6`，全部逐文件SHA通过；receipt=`m32_morder_default_att_evidence_unpack_acceptance_v1.json`。

下一项默认关闭`AITER_GFX1250_GEMM1_PAIR_PREFETCH_SCOPE=1`，将下一K128的A pair预读移到相同M32 predicate控制的当前pair WMMA前，使用同一scalar branch。原来下一pair的load在上一pair末尾，各有一处row predicate；候选保留两套independent operand slots，B和ScaleA/ScaleB预读、input/output地址、TDM/WG hardware fences、每个有效accumulator的56次K128更新顺序与所有数值运算不变。此项减少分支的同时可能缩短A预读覆盖距离，必须以actual ISA、资源、双seed和完整性能对照判断。

CPU审计`audit_m32_pair_scope_v1.py`执行实际k_step AST，覆盖257种row边界、两个M128 halves、56个K128位置，共28784次k_step、129024个source M16 updates及63360次future pair loads；全部有效更新唯一保留，最后K128无next loads，默认关闭分支的AST与已验收控制完全一致。原mma/load地址函数和GEMM2/helpers保持。源码快照=`m32_pair_scope_v1_source/`，两端部署要求SHA及host idle；实际compile/精度/性能尚待完成，不能记为性能提升。

### pair scope机器码完全相同，撤回候选

`compile_a07_m32_pairscope0_v1`和`compile_a07_m32_pairscope1_v1`在a07/dsr1_ep4只编译，均未超过120秒。关闭候选的ELF完整等于已验收默认`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`；打开候选ELF为`de979320cf7ee0b7cf42180e706f6a74b9fcad0cf38e54ba6bc8f1eb1cc12e46`。两者均4482 executable instructions、VGPR486/SGPR68、LDS282624 B、zero spills/private。

独立解析ELF64 section headers得到两份完整`.text`均29312 B，逐字节相同，SHA256=`ddf5c40d348d0badebf4a5eeabcbdb376d80a10122e42193a40cecffa5f23677`。4582条address/encoding记录也全部相同；其中包含padding记录，不能作为executable instruction count。默认ISA已经把next-pair A loads和同pair当前WMMA放进同一scalar branch，源码scope改写没有改变该调度。验收=`m32_pair_scope_native_equivalence_v1.json`，决定撤回；未执行候选GPU精度、计时或ATT，不把不同ELF symbol当成kernel优化。

### 下一项有限候选：缓存M32的SGPR条件值

LLVM `IntrinsicsAMDGPU.td`的`int_amdgcn_sched_barrier`、`SIInstructions.td`的`SCHED_BARRIER`以及`AMDGPUMCInstLower.cpp`明确：mask约束编译器调度，pseudo size0，最终仅输出comment。这类hint不提供硬件memory fence。进一步核对默认ISA，每个M32 branch仍有` s_and_b32 / s_cselect_b32 / s_cmp_lg_u32`的条件转换链。前述readfirstlane/ballot候选没有稳定收益，本轮不重复它们；改用真实wave-uniform条件的0/1值经过空LLVM asm `"", "=s,0", has_side_effects=False`，缓存四个SGPR值供原load/WMMA条件判断复用。

新候选`AITER_GFX1250_GEMM1_ROW_PREDICATE_SCALAR=1`默认关闭，以精确M-order默认为控制。tied input/output保证数值identity；LLVM `AMDGPUTargetTransformInfo.cpp::isInlineAsmSourceOfDivergence`把SGPR输出识别为uniform。它没有硬件wait、没有改lane EXEC，也未修改任何有效route、K128更新及顺序、SiLU、BF16 rounding、FP4/E8M0、地址或布局。Shader Guide §4.6.12的WMMA全EXEC要求仍保留；§5.3.2的source-cache invalidation事实不能用于断言当前scalar branch导致cache miss。

CPU执行实际`row_group_is_live` AST，覆盖row边界-1..257和30464、两个M128 halves、56个K128和候选开关，共232960个predicate checks、261632个source M16 updates；还原条件后原load/WMMA AST完全一致。原`k_step/compute_ktile`和硬件同步AST、GEMM2/helpers不变。审计=`m32_rowpred_scalar_source_audit_v1.json`；三源快照=`m32_rowpred_scalar_v1_source/`。后续只有actual native code有实质变化、资源合格时才进入GPU验证和完整性能对照，当前正式默认仍107.170/107.223 μs，目标未达到。

### M32条件生成候选的编译结果与撤回

所有编译仅在a07-3/dsr1_ep4进行，每次120秒限制，没有候选GPU dispatch、精度计时或ATT。scalar v1的打开分支因`.select(1,0)`需要typed参数而编译失败；关闭分支的ELF完整等于已验收默认。v2改用`fx.Int32(1)/fx.Int32(0)`后编译成功，但目标`s_and_b32 / s_cselect_b32 / s_cmp_lg_u32`转换链仍在，增加控制指令，淘汰。

随后以LLVM inline asm的`={scc},s,s`直接生成比较结果。CDNA5 ISA §15.4（p268）明确`S_CMP_LT_I32`设置`SCC = S0.i32 < S1.i32`；LLVM scalar输出uniform与i1 SCC约束由对应源码/测试支持，不预判最终branch lowering。SCC v3因`T.i1`不存在失败；v4改`ir.IntegerType.get_signless(1)`后有spills；v5把纯比较设`has_side_effects=False`允许CSE。

| 候选 | executable instructions | VGPR/SGPR | VGPR spills / private（B） | ELF SHA256 | 判断 |
|---|---:|---|---|---|---|
| scalar v2 | 4500 | 486/74 | 0 / 0 | `0e5a2a1871a7c1753f2ff8e972bb479c3f20f65986263a7abff4a7ca65011be1` | 转换链保留，指令增加 |
| SCC v4，side effects | 4867 | 485/68 | 8 / 20 | `2f0ec763300468cff115ea3838411be697e3f55802b930ddafee4f8a239072f22` | 资源失败 |
| SCC v5，pure compare | 4480 | 486/68 | 0 / 0 | `9e619e836cf254a5fd98ba1318240254b222d27c38c993ccdf66b5ce91a18ef2` | 转换链与98条static `s_cbranch_scc1`保留 |
| 已验收默认 | 4482 | 486/68 | 0 / 0 | `ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d` | 继续保留 |

四项LDS均282624 B。v5全部opcode counts与默认相同，只有`s_delay_alu`339→337；predicate SGPR编号和部分prime/drain调度有变化，ISA并非逐字节等价，但没有实现消除目标控制开销。此处不把两条static delay减少记录成性能提升，也不为这个缺少实质变化的版本重复完整GPU队列。scalar v2和SCC v3/v4/v5的CPU有效更新、predicate与原功能审计通过，不替代GPU精度验收。

双端通过`withdraw_m32_rowpred_v1.py`恢复精确`m32_morder_default_v1_source/`三源；host在整机idle后核对前后SHA，记录=`m32_rowpred_withdrawal_host_v1.json`，本地=`m32_rowpred_withdrawal_local_v1.json`。wrapper/persistent/compiler恢复为`d8bc366bb26c13c6d307549d2119d100e17c3ef8f45b56c3eb49e5fbec8e67a6` / `682518c9a6efa6bf4c308b7f7f3a3dcc90401a90800d9cfb40354a0aef090346` / `8b164cadfa680d6fcdb6a1bdeb71a540bd8451433cd2134cffa28fbaabf7cb25`。全部失败源码、raw编译日志和候选产物保留。

### 允许无效计算裁剪后的M16尾部候选

在已验收M32 pair/halfScale/pairA pipeline上新增默认关闭`AITER_GFX1250_GEMM1_ROW_TAIL_M16=1/2`。mode1在每个pair原first-M16 predicate内部，为second-M16增加真实有效行判断，只跳过全padding的后半M16；mode2另外把prime/future A operand loads裁剪到同样的M16有效范围。部分有效M16仍保留全部lanes及原56次K128更新与顺序；B/ScaleA/ScaleB、TDM、同步、SiLU/BF16/FP4、canonical输出布局与GEMM2/helpers保持。它使用当前`wmb/mn_oob`，不绑定seed或routing counts。

CPU直接执行实际`mma_rows/load_a_rows/load_pair_data/k_step` AST，覆盖260种边界（-1..257及30464）、两个M128 halves和全部56个K128，执行87360次k_step，检查370944次有效M16更新与392160次A fragments；两候选在该枚举中合计裁剪14336次全padding M16 updates。所有有效accumulator保留56次有序更新，关闭分支AST完整等于默认；其他45个nested函数、GEMM2/helper与原benchmark/routing审计通过。记录=`m16_tail_source_audit_v1.json`，快照=`m16_tail_v1_source/`。此时仅CPU审计完成，actual ISA/resources、GPU精度及性能待验收，正式最佳仍107.170/107.223 μs，80 μs目标尚未达到。

### M16尾部裁剪的精度通过，性能淘汰

三份actual ELF均LDS282624 B、zero spills/private。mode0完整等于已验收默认ELF；mode1为496 VGPR/68 SGPR、4864 instructions、162条static `s_cbranch_scc1`，ELF=`fa47ae2ad70e17f3e18ad9f24a792fb9b26b6751545fe0e81deaaa56d4f67eee`；mode2为496/70、5659 instructions、285条branch，ELF=`c6fa98fb15adab12b032473b0a1e7c9d63d14e780cbd96bc677dce31b50cae7a`。默认对应486/68、4482 instructions、98条branch。static有效计算/量化/内存/同步清单保持，runtime全padding M16被跳过；验收=`m16_tail_compile_acceptance_v1.json`。

两mode各seed0/17共四项device验证rc0、未超过120秒，各28条零字节差异。全部HEAD有效payload、ScaleA、GEMM2、MoE、各三次eager/graph canary与三组同地址dynamic input/psum replay通过；correctness-only耗时没有使用。

固定0→1→0→2→0完整三轮计时，仅ownership污染整组重试最多三次。label前缀=`a07_m16_tail`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 108.049,107.446,109.552 | 108.049 | 94.6206,93.8658,94.4711 | 258.096,257.443,259.445 | 三轮合格 |
| `1_perf_a_v1_try1` | 109.086,108.420,108.028 | 108.420 | 93.7753,95.2026,94.3318 | 258.603,259.473,258.042 | 三轮合格 |
| `0_control_b_v1_try1` | 107.159,106.750,108.251 | 不统计 | 93.0415,94.4477,94.3360 | 256.447,256.266,258.815 | 第3轮污染 |
| `0_control_b_v1_try2` | 107.708,108.184,106.396 | 107.708 | 94.6733,93.7669,94.9558 | 258.307,257.882,257.685 | 三轮合格 |
| `2_perf_a_v1_try1` | 121.668,120.733,121.307 | 不统计 | 93.8869,92.2844,93.4842 | 271.012,268.638,271.440 | 第2轮污染 |
| `2_perf_a_v1_try2` | 127.052,121.130,122.912 | 122.912 | 91.7026,93.0479,93.2355 | 274.350,270.287,272.109 | 三轮合格 |
| `0_control_c_v1_try1` | 107.197,107.010,107.463 | 107.197 | 93.6488,94.0829,94.9664 | 257.483,255.518,258.559 | 三轮合格 |

21轮rc0、未超时；mode1比前后控制慢0.343%/0.661%，mode2慢14.116%/14.660%。没有保留理由，不按耗时追加复测，完整源码和失败/合格组均保留。`withdraw_m16_tail_v1.py`在两端恢复精确M-order三源；summary=`m16_tail_complete_summary_v1.json`。结束后整机idle、KFD owners为空、VRAM176582656 B，无本任务GPU进程残留。此结果证明允许裁剪无效计算后，裁剪仍需综合branch、register和prefetch代价；不能直接把减少WMMA数当作kernel收益。正式最佳仍107.170/107.223 μs。


### 2/4 N-task候选的预取地址修复及性能淘汰

在精确M-order默认上有限尝试2/4 N-task复用，保持M32/halfA/halfScale/pairA、canonical输出、每个有效accumulator的56次K128更新及顺序。第一版2-task同输入eager/graph canary全部一致，但首次同地址dynamic input更新的有效payload有1417015 B差异，rc1、未超时，停止队列且没有计时。核查发现fused launcher下一task的B/ScaleB预取用`tile_idx+1`而漏了`persistent_n_group`基址；修正为`next_blk_n=((n_unit+1)*cluster_n+local_n)*tile_n`。4-task只有一组，该表达式原来等价；1-task不执行该分支。独立GEMM2 launcher没有改动。

实际AST地址审计36400次通过，其他kernel AST、数值、内存同步和输出布局保持。修正后2/4-task各seed0/17共四项rc0、未超时，每项28条零字节差异：原HEAD有效payload、ScaleA、GEMM2/MoE、各三次eager/graph canary、三组同地址input/psum更新全部通过。2-task v2 ELF=`4140199d70d93fce0f2f8abf5b84482941678fd8240d58603d739bafcce66379`，5215 instructions；4-task=`36b99ce17658e493c9fb61ed1999842360d948ad6db74575d9171deff143e484`，5187 instructions。两项VGPR486/SGPR107、LDS282624 B，1个VGPR spill/private8 B，SGPR spills分别22/20，明确不能记为zero-spill。scratch和SGPR lane spill位于successor bootstrap/descriptor setup，保守八条steady K backedge区间之外，并不表示其动态成本为零。

固定1→2→1→4→1，每组三轮原命令全部ownership合格、rc0、未超过120秒，没有重试。label前缀=`a07_m32_multitask`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `1_control_a_v2_try1` | 107.377,106.253,106.902 | 106.902 | 95.3818,95.4915,93.3661 | 259.368,257.283,255.487 | 三轮合格 |
| `2_perf_a_v2_try1` | 122.553,121.988,122.614 | 122.553 | 89.4192,88.6665,89.6934 | 268.513,267.159,268.782 | 三轮合格 |
| `1_control_b_v2_try1` | 107.343,108.591,107.385 | 107.385 | 93.5347,95.8415,94.0387 | 256.517,260.081,257.179 | 三轮合格 |
| `4_perf_a_v2_try1` | 148.649,148.522,146.53 | 148.522 | 90.0559,90.3743,89.1472 | 294.65,294.594,291.863 | 三轮合格 |
| `1_control_c_v2_try1` | 108.728,106.911,106.584 | 106.911 | 93.9018,95.6751,94.6945 | 258.497,258.811,257.77 | 三轮合格 |

2/4-task都明显回退，淘汰调优开关，不根据耗时追加筛选。地址bug fix作为独立改动保留，继续优化以精确M-order默认加这条修复为基础。完整验收=`m32_multitask_compile_acceptance_v2.json`，全部结果=`m32_multitask_complete_summary_v2.json`，源码=`m32_multitask_v2_source/`。结束后整机idle、KFD空、VRAM176582656 B，无本任务GPU进程残留。正式默认性能仍107.170/107.223 μs。

### 完全越界parent workgroup的提前capacity判断：无收益撤回

默认ATT的22个无WMMA waves，span P50为897 cycles、240 instructions、8个scalar load sites、41条cselect。有限候选`AITER_GFX1250_GEMM1_CAPACITY_GUARD=1`在bisect/descriptor之前复制现有group16 contiguous-M swizzle，检查真实parent起点`cap_m_tile*tile_m < cap_map[n_experts-1]`，仅跳过完全无有效行的parent。task_m_order只排列live前缀，partial有效tile及原`run_task`全部保持，最终tensorwait0仍执行。不是根据seed/count写死route裁剪。

CPU实际AST859140 geometry checks、28 routing cases通过，保留800580、排除58560个parent/N tiles，覆盖空expert及capacity/group/N边界。CDNA5 ISA §5.5（p45）支持signed dword branch target解码；§5.6（p45–46）说明barrier早退规则，本候选跳过路径不发任何barrier。mode0 ELF完整等于默认FFB；mode1 ELF=`3919065e5aee581d998f6cac61be7d5ab502be1b0fd1a1cc4bb5c9436f16601c`，4483 instructions、VGPR486/SGPR70、LDS282624 B、zero spills/private。容量branch PC0x1DCC直接跳至0x8B4C的tensorwait0/endpgm；跳过路径168 instructions、3个scalar load sites，没有DS/TDM/WMMA/barrier。该静态缩短不等于kernel收益。

seed0/17均通过28条zero-byte comparisons及MoE原精度、dynamic input/psum、eager/graph replay，rc0且未超时。固定0→1→0三轮组结果如下，label前缀=`a07_capacityguard`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 109.894,107.237,106.999 | 107.237 | 94.4477,93.8343,93.6382 | 260.262,256.656,256.593 | 三轮合格 |
| `1_perf_a_v1_try1` | 106.253,107.284,107.486 | 107.284 | 94.9075,94.7175,95.0003 | 257.021,256.57,258.643 | 三轮合格 |
| `0_control_b_v1_try1` | 110.208,107.551,106.841 | 不统计 | 94.1694,93.9037,94.4415 | 260.203,257.166,257.097 | 第1轮污染，整组排除 |
| `0_control_b_v1_try2` | 106.904,107.973,107.153 | 不统计 | 94.8485,92.8033,94.9052 | 257.468,257.325,257.11 | 第2/3轮污染，整组排除 |
| `0_control_b_v1_try3` | 106.959,106.464,108.412 | 106.959 | 94.3192,94.509,94.6291 | 257.127,256.325,259.564 | 三轮合格 |

后控制两组污染后按事前规则整组三轮重试至第三次，15轮全部rc0、未超时。candidate107.284 μs比前后合格控制107.237/106.959 μs慢0.0438%/0.3039%，没有收益，不做反向计时或按耗时追加筛选。撤回capacity参数，候选源码保留于`capacity_guard_v1_source/`；完整结果=`capacity_guard_complete_summary_v1.json`。结束后整机idle、KFD空、VRAM176582656 B，无本任务GPU进程残留；`capacity_guard_post_idle_v1.json`。

### 下一项有限候选：K loop外的all-live-M32路径

`AITER_GFX1250_GEMM1_TAIL_LOOP_SPLIT=1`在进入K pipeline前用当前wave的`wmb+(wmma_m_rep-2)*WMMA_M < mn_oob`判断是否全部四个M32 pairs会执行。满足时采用去掉重复M32 compute/A-load判断的constexpr路径，否则完整保留原pair predicate。它只外移原分支，不减少任何当前有效或已执行的pair计算；TDM owners、ring、barrier各phase、双套operand、SiLU/BF16 rounding/FP4/E8M0及canonical epilogue保持。

CPU直接执行两条实际helper/k_step AST，覆盖260个row边界、58240次k_step、247296次有效M16更新和266304次A fragments；两路径source事件序列逐项相同，每个有效accumulator仍56次有序K128更新。去掉constexpr特化与helper包装后，关闭候选`run_task` AST等于精确默认加独立N-group地址fix。GEMM2/helpers和实际同步AST保持。audit=`tail_loop_split_source_audit_v1.json`。

关闭候选的ELF完整等于默认`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`；原4-task fallback完整等于`958d31dd24a22c8a38af0ec1113a27ee8c5bf69e8ffb124ff222758f11c68110`，证明独立地址修复不改这两路径。candidate ELF=`0d24bb2bec47f443adb14ef3684e05dde78f0aafbec4788489954198e6b3a719`，6248 instructions、VGPR458/SGPR78，LDS282624 B且zero spills/private。静态K/TDM/barrier总数因两个互斥pipeline而增加，不能当作动态实际重复工作；epilogue static counts完整保持，八个owner K loop入口与十二条backedge解码通过。compile验收=`tail_loop_split_compile_acceptance_v1.json`。双seed、原HEAD严格字节及dynamic replay的GPU验收已完成；计时及撤回结果见下一节。

### tail loop split精度通过，候选三组ownership污染后撤回

`a07_tailloopsplit1_seed0_v1`与`seed17_v1`均rc0、未超时，每项28条`differing_bytes=0`。HEAD全部有效payload、ScaleA、GEMM2、MoE输出、各三次eager/graph canary及三组同地址input/psum更新通过。这两项为correctness-only，耗时不作为性能。

事前固定0→1→0，每段完整三轮仅因ownership污染重试、最多三次。候选第三组仍不合格后不追加第四组，只补完原定后控制。全部27轮rc0、未超过120秒，label前缀=`a07_tailloopsplit`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 107.651,109.329,107.767 | 不统计 | 94.2791,95.5821,94.783 | 258.071,261.416,258.366 | 第1轮污染 |
| `0_control_a_v1_try2` | 108.376,107.147,108.091 | 不统计 | 94.88,94.3108,91.416 | 259.119,257.649,256.623 | 第1/3轮污染 |
| `0_control_a_v1_try3` | 107.421,107.044,107.250 | 107.250 | 94.7283,94.3001,94.0324 | 258.174,257.276,255.904 | 三轮合格 |
| `1_perf_a_v1_try1` | 110.609,109.139,109.169 | 不统计 | 93.8469,95.0424,94.2517 | 261.833,260.563,259.825 | 第2轮污染 |
| `1_perf_a_v1_try2` | 109.791,110.170,108.989 | 不统计 | 95.0951,92.8097,94.3254 | 261.495,260.340,260.041 | 三轮污染 |
| `1_perf_a_v1_try3` | 111.092,109.926,110.409 | 不统计 | 94.547,95.5759,94.2286 | 260.698,261.742,260.409 | 第1轮污染 |
| `0_control_b_v1_try1` | 106.725,108.675,106.831 | 不统计 | 93.4421,94.8821,94.4774 | 255.081,259.958,257.721 | 第1轮污染 |
| `0_control_b_v1_try2` | 107.897,107.077,108.511 | 不统计 | 94.2096,94.4943,93.9944 | 257.167,257.053,258.678 | 第2/3轮污染 |
| `0_control_b_v1_try3` | 106.970,107.370,106.487 | 106.970 | 93.7456,93.925,94.315 | 257.426,257.017,256.995 | 三轮合格 |

没有完整合格候选中值，不宣称已证明性能收益或有效回退，不拼接污染组中的单轮。候选已从生产撤回，独立N-group地址fix保留；完整记录=`tail_loop_split_complete_summary_v1.json`，结束标记=`tail_loop_split_perf_finished_v1.json`，整机idle、KFD owners为空、VRAM176582656 B，`tail_loop_split_post_idle_v1.json`。正式默认仍107.170/107.223 μs。

### steady K loop的2/4倍展开：4倍精度正确但性能回退，撤回

以`m32_morder_addressfix_v1_source/`为基础，默认`AITER_GFX1250_GEMM1_STEADY_K_UNROLL=1`。候选2/4只展开24次steady K256循环，按原顺序执行全部计算、TDM与reader fences；不包含前述capacity/tail split开关。CPU实际AST覆盖480事件组、15360 Ktile events及30720 K128 events，三mode的计算顺序、TDM ring地址及barrier事件完全一致，关闭分支run_task与基础AST相同，GEMM2/helpers保持。记录=`steady_k_unroll_source_audit_v1.json`。

全部compile仅在a07-3/dsr1_ep4，每次120秒。实际ELF与反汇编验收=`steady_k_unroll_compile_acceptance_v1.json`：

| mode | executable instructions | VGPR/SGPR | VGPR spills / private（B） | ELF SHA256 | 后续 |
|---:|---:|---|---|---|---|
| 1控制 | 4482 | 486/68 | 0/0 | `ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d` | 完整等于已验收默认 |
| 2 | 5310 | 502/86 | 10/24 | `a53a03f43404f6b760b68bedc303ae496c4e20d42fd0994164bbeab74026ea5b` | 资源筛除，没有GPU dispatch |
| 4 | 6934 | 509/94 | 0/0 | `beb540e9e3291e232bbf44bfe645a9e8cabb805011b9c75895f8001698bc1c03` | 进入严格精度及原命令计时 |

三份LDS均282624 B。依据CDNA5 ISA §5.5（p45）的signed16-bit DWORD branch规则解码，每mode八条backedge对应四个owner loop，静态barrier sites分别为2/4/8，WMMA保守区间分别28或32、60或64、124或128；没有移除任何硬件同步，expanded static counts不等于dynamic重复计算。mode2的scratch位于上述保守区间之外，但本次不为新增spills版本追加GPU队列。mode4的BF16、SiLU、FP4、ScaleA及output static counts与控制保持。

`a07_steadykunroll4_seed0_v1`与`seed17_v1`均rc0、未超时、各28条零字节差异。HEAD有效payload、ScaleA、GEMM2、MoE、各三次eager/graph canary及三组同地址dynamic input/psum replay通过；correctness-only耗时无效。

事前固定1→4→1，每组三轮，只有ownership污染允许整组最多重试三次。12轮全部rc0、未超过120秒，label前缀=`a07_steadykunroll`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `1_control_a_v1_try1` | 107.927,109.901,107.275 | 107.927 | 95.0908,94.2685,95.0192 | 259.144,260.416,258.881 | 三轮合格 |
| `4_perf_a_v1_try1` | 113.345,115.032,116.683 | 115.032 | 94.5827,94.2832,94.6059 | 264.502,265.971,268.282 | 三轮合格 |
| `1_control_b_v1_try1` | 107.604,107.727,108.089 | 不统计 | 93.2523,94.2368,95.6265 | 257.202,258.153,260.470 | 第3轮污染 |
| `1_control_b_v1_try2` | 107.206,107.484,108.209 | 107.484 | 95.1223,93.0626,93.0098 | 258.639,257.620,257.367 | 三轮合格 |

mode4比两侧控制慢6.583%/7.023%，没有收益，不追加按耗时筛选的复测。后控制try1的107.727 μs曾在即时进度中被使用，最终ownership记录将其排除，以try2的107.484 μs为准；原始组全部保留。summary=`steady_k_unroll_complete_summary_v1.json`；最终post整机idle、KFD空、VRAM176582656 B，boot=`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`。双端通过`withdraw_steady_k_unroll_v1.py`恢复精确`m32_morder_addressfix_v1_source/`，保留独立2-task地址fix，不留下unroll开关。正式默认仍107.170/107.223 μs。

### 单一steady K loop的byte-offset ring carry候选

展开方案增加代码与register后，继续有限候选`AITER_GFX1250_GEMM1_STEADY_RING_CARRY=1`。原四缓冲ring每轮以`kt%4`/`(kt+1)%4`计算当前及carry-target地址，实际ISA仍有相应`s_and_b32`和两条`s_mul_i32 ... 0x11400`。候选用2-wide Int32 rmem保存当前byte offset，每轮加PITCH并在4*PITCH处回零；当前slot也是K+4 refill的目标，向原issue函数传入同一个byte offset。保持单一rolled compute body，全部K128更新及顺序、input/output addresses、descriptor emit、DS/WG/TDM fences和epilogue保持。不直接移植旧130us文档已失败的packed K512、pair-role或A4/B6 carry方案；没有改变数据排布、request大小或数目。

CPU直接执行实际steady_mid、do_issue及issue地址前缀，覆盖720事件组、19530 Ktile/39060 K128 events及19530 TDM address cases，current/carry/refill地址与控制逐项一致。移除新参数及constexpr分支后run_task AST等于精确已验收基础，原计算/同步/epilogue及GEMM2/helper AST保持。审计=`steady_ring_carry_source_audit_v1.json`，源码=`steady_ring_carry_v1_source/`。实际native资源/地址简化与双seed严格精度验收已完成，完整性能队列及撤回决定见下节。

### ring carry的实际编译与精度通过，完整后控制污染后撤回

mode0完整ELF等于默认`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`，4482 instructions、VGPR486/SGPR68。mode1 ELF=`08397d26900e7980e45e26d14b4d8b587bb414c20ac93324287e92bdc216381b`，4476 instructions、VGPR486/SGPR70。两者LDS282624 B、zero spills/private。四个owner steady loop实际两条pitch multiplication及两处ring modulo被add/compare/select回绕替换，八条backedge及每区间2个barrier sites保持；全部input/output/epilogue opcodes计数相同。compile acceptance=`steady_ring_carry_compile_acceptance_v1.json`。这项native简化本身不证明kernel性能收益。

`a07_steadyringcarry1_seed0_v1`与`seed17_v1`均rc0、未超过120秒，各28条零字节差异：HEAD有效payload、ScaleA、GEMM2、MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay通过。全部correctness-only耗时不使用。

事前固定0→1→0完整三轮组，只因ownership污染整组重试、最多三次；后控制第三组仍污染后结束，没有追加第四次。18轮rc0、未超时，label前缀=`a07_steadyringcarry`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 106.619,107.010,107.162 | 不统计 | 93.7309,94.9854,95.1435 | 256.007,258.094,258.328 | 第1轮污染 |
| `0_control_a_v1_try2` | 107.062,108.037,109.285 | 108.037 | 94.5322,94.6375,92.5186 | 256.831,257.341,256.821 | 三轮合格 |
| `1_perf_a_v1_try1` | 107.916,109.268,107.560 | 107.916 | 96.0946,95.0213,93.1428 | 259.851,258.706,255.717 | 三轮合格 |
| `0_control_b_v1_try1` | 110.084,108.309,107.980 | 不统计 | 94.9918,94.5913,94.5554 | 260.834,258.114,258.213 | 第2轮污染 |
| `0_control_b_v1_try2` | 110.251,106.190,107.826 | 不统计 | 94.1381,93.9291,93.7499 | 259.256,255.918,256.593 | 第3轮污染 |
| `0_control_b_v1_try3` | 107.274,106.974,107.575 | 不统计 | 94.6903,94.7850,95.3966 | 257.850,256.475,259.233 | 第1轮污染 |

候选合格中值107.916 μs只比前控制108.037 μs低0.112%，三个完整后控制都不合格，不能确认稳定收益；不拼接单轮、也不以candidate中值替代独立默认验收。双端通过`withdraw_steady_ring_carry_v1.py`恢复精确`m32_morder_addressfix_v1_source/`三源，独立N-group地址fix保留，生产ring开关已移除。全部source snapshots、污染与合格组保留。summary=`steady_ring_carry_complete_summary_v1.json`；队列post及host撤回门禁均整机idle、KFD owners为空、VRAM176582656 B，boot=`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，没有本任务GPU进程残留。正式最佳仍107.170/107.223 μs，80 μs目标尚未达到。

### ScaleA随M32 pair分段预读的有限候选

新候选`AITER_GFX1250_GEMM1_PAIR_SCALE_PREFETCH=1`默认关闭，依赖原M32/halfA/halfScale/pairA路径。将每套ScaleA从两个2-wide FRONT/BACK register slots改为四个2-wide identity pair slots，下一K128的scale与对应A pair在同一个真实有效行predicate内预读；prime读取全部执行pair的scale，全padding pair的scale不读取。每个执行M32 pair仍保留两条M16的原全部56次K128更新、相同A/B/ScaleA/ScaleB值与顺序。TDM request、地址、READY/reader/WG fences、BF16 rounding/FP4/E8M0、canonical epilogue和GEMM2/helpers保持。

CPU直接执行实际prime及28次`compute_ktile` AST，覆盖260个parent row边界、两个M128 halves、两个mode，共1040个pipeline cases、58240次K128 step、247296次有效M16更新。每个执行pair的当前A/B/ScaleA/ScaleB逐项一致，关闭分支AST等于精确基础；compute/TDM/硬件同步事件原序一致。枚举中ScaleA loads从116480变为65408，只减少全padding pair的51072次scale loads；这些枚举统计不是实际workload的性能或计算量测量。记录=`pair_scale_prefetch_source_audit_v1.json`。

a07-3/dsr1_ep4 compile每项120秒、两项rc0。mode0完整ELF仍默认FFB，VGPR486/SGPR68、4482 instructions；mode1 ELF=`774940b14cabdd98a3a303e18f3c46a6b79c042e7e0bf74c02fbd9a9fec341b5`，VGPR496/SGPR68、4607 instructions。两项LDS282624 B、zero spills/private，八条backedge及各区间2个barrier sites保持。所有WMMA、A/B b128 loads、input/output TDM、epilogue opcodes计数保持；static scale dwords均96，但scalar DS load合并减少，`ds_load_b32`42→80、两种2addr sites合计27→8，`s_cbranch_scc1`98→117。不能把runtime padding裁剪或zero spills直接当作kernel收益。验收=`pair_scale_prefetch_compile_acceptance_v1.json`。

`a07_pairscaleprefetch1_seed0_v1`及`seed17_v1`均rc0、未超过120秒，各28条零字节差异；HEAD有效payload、ScaleA、GEMM2、MoE、各三次eager/graph canary及三组同地址dynamic input/psum replay通过。两项correctness-only，不使用计时；外部owners不影响这些已完成的字节比对口径。候选源码=`pair_scale_prefetch_v1_source/`，三源manifest=`pair_scale_prefetch_source_manifest_v1.json`。

固定0→1→0完整三轮性能队列已经结束，仅ownership污染整组重试、最多三次，label前缀=`a07_pairscaleprefetch`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 107.008,107.278,107.301 | 107.278 | 95.3716,94.8148,94.2203 | 258.683,258.439,257.793 | 三轮合格 |
| `1_perf_a_v1_try1` | 110.069,110.128,109.757 | 110.069 | 95.7806,95.8585,95.4917 | 261.742,261.996,260.512 | 三轮合格 |
| `0_control_b_v1_try1` | 107.843,106.966,108.134 | 不统计 | 95.1944,96.5984,94.1168 | 259.455,259.233,256.044 | 第3轮污染 |
| `0_control_b_v1_try2` | 108.250,106.576,106.692 | 不统计 | 94.7011,95.4558,94.7537 | 259.700,257.898,258.181 | 第1轮污染 |
| `0_control_b_v1_try3` | 107.971,108.699,107.763 | 不统计 | 95.5337,94.6271,95.0679 | 259.558,259.686,258.571 | 第2轮污染 |

15轮rc0、未超过120秒。候选合格中值110.069 μs比合格前控制107.278 μs慢2.6017%；三个完整后控制均受ownership污染，不能据此宣称完整双侧稳定回退，但该候选没有稳定收益的证据，因此淘汰，不追加第四次或按单轮耗时筛选。双端通过`withdraw_pair_scale_prefetch_v1.py`恢复精确`m32_morder_addressfix_v1_source/`三源，独立N-group地址fix保留。summary=`pair_scale_prefetch_complete_summary_v1.json`，撤回receipt=`pair_scale_prefetch_withdrawal_local_v1.json`及`pair_scale_prefetch_withdrawal_host_v1.json`。队列post与host撤回门禁均整机idle、KFD owners为空、VRAM176582656 B，无本任务GPU进程残留。正式最佳仍107.170/107.223 μs，80 μs目标尚未达到。

### B-only LDS padding的独立候选

进一步检查M方向B/ScaleB multicast时，现有`cluster_m=4`路径需要按runtime expert交集构造column mask及额外cluster同步。当前t256父tile保持原expert alignment，non-balanced相邻父tile不保证使用同一expert的权重；没有把balanced复用比例或旧130us任务的内部M64/M96结果作为当前收益证据，也没有部署M方向候选。

先新增默认保持16的`AITER_GFX1250_GEMM1_B_LDS_PADDING=0/128`，只在原E65/t256/M32/halfA/halfScale/pairA单N-task路径有限实验。A仍每1024 B插入16 B，B分别不插padding或插128 B；B TDM descriptor、B row/subtile stride及后续ScaleA/ScaleB/stage base随之调整，所有有效WMMA、input READY/reader lifetime fences、output TDM、SiLU、BF16 rounding/FP4/E8M0、canonical布局保持。硬件依据为Shader Guide §4.10.2（p200）：非128 B粒度的padding会使一些操作采用indirect copy；资料未保证当前kernel实际copy模式、也未证明此处是唯一瓶颈。

CPU直接执行actual constants和`load_b` AST，三个padding模式、四个stage、28个K256、四个N compute waves、全部lanes及K128 subtiles，检查11010048个B fragment字节的公开canonical inverse映射。全部模式的逻辑B fragment hash相同，无load触及padding。A row保持2080 B，B row为2080/2048/2304 B；LDS为282624/280576/296960 B，均在320KB内。默认nested AST归一化后等于精确基础，全部有效56次K128更新函数、GEMM2/helpers/routing/benchmark保持。审计=`b_lds_padding_source_audit_v1.json`，快照=`b_lds_padding_v1_source/`。目前仅CPU审计完成，实际compile、双seed精度与性能尚待验收；正式最佳仍107.170/107.223 μs。

### B-only padding的actual compile和双seed精度

三项actual compile仅在a07-3/dsr1_ep4运行，每项120秒、rc0。所有WMMA、DS b128、TDM、WG barriers/tensorwait及量化数学的static清单保持；八条K backedges、各区间两处barrier sites保持。三项均zero VGPR/SGPR spills、private0：

| B padding（B / 每1024 B） | LDS（B） | VGPR/SGPR | executable instructions | ELF SHA256 |
|---:|---:|---|---:|---|
| 16，默认 | 282624 | 486/68 | 4482 | `ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d` |
| 0 | 280576 | 486/68 | 4488 | `5709c0def46c998967b570a8d489a1973d6473b60bf4dc574b9807219d5f9c35` |
| 128 | 296960 | 486/70 | 4481 | `1c42f870d6619966f49046a5e825fd5d58657514de2a5d154e7f2296d5c2658b` |

默认ELF完整等于此前独立验收。验收=`b_lds_padding_compile_acceptance_v1.json`，源码三源SHA分别为`97c1fc621fecaafcf52dbd3b8f407c37c136e3e87bb6d8c1101c7dc5f526dc71` / `b879c6054e5d7bcaacfb326dfbc7e15e7ed196a257b9a4021fadac8cbf7ffd9d` / `d4e60858c5e15b6e2a60fd45a655c0f96d6adf6d52e2ce71111c970f5d6acfa0`。

`a07_bldspad0_seed0_v1`、`seed17_v1`和`a07_bldspad128_seed0_v1`、`seed17_v1`四项均rc0、未超过120秒，各28条零字节差异；HEAD全部有效payload、ScaleA、GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay通过。四项均correctness-only，没有使用耗时。验收=`b_lds_padding_correctness_acceptance_v1.json`。固定16→0→16→128→16完整三轮性能队列已开始，最多三次ownership整组重试；当前尚无新有效性能结论，正式默认仍107.170/107.223 μs。

### B-only padding完整性能及撤回

固定16→0→16→128→16已经完成，前控制仅因ownership污染整组三轮重试一次，18轮均rc0、未超过120秒，label前缀=`a07_bldspad`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `16_control_a_v1_try1` | 106.335,108.559,109.858 | 不统计 | 94.7199,94.3424,94.5913 | 256.553,259.279,259.364 | 第1/3轮污染 |
| `16_control_a_v1_try2` | 106.683,107.590,107.501 | 107.501 | 94.1991,92.4174,93.8996 | 256.856,257.028,256.633 | 三轮合格 |
| `0_perf_a_v1_try1` | 108.330,107.183,106.753 | 107.183 | 94.3214,95.0024,94.9117 | 259.115,258.647,257.814 | 三轮合格 |
| `16_control_b_v1_try1` | 107.482,107.079,106.995 | 107.079 | 94.4797,93.8132,93.6086 | 258.038,258.581,257.086 | 三轮合格 |
| `128_perf_a_v1_try1` | 109.407,107.585,106.763 | 107.585 | 93.7964,95.7127,94.2770 | 259.172,259.595,257.639 | 三轮合格 |
| `16_control_c_v1_try1` | 108.094,107.982,106.234 | 107.982 | 94.3298,94.5257,93.6718 | 257.894,258.377,255.670 | 三轮合格 |

无padding比前控制快0.296%，比后控制慢0.097%；128 B padding比前控制慢0.473%，比后控制快0.368%。两项都没有两侧一致改善，不据此启动反向或按耗时追加选择。两端执行`withdraw_b_lds_padding_v1.py`恢复精确`m32_morder_addressfix_v1_source/`三源，独立N-group地址fix保留，padding生产参数撤出。summary=`b_lds_padding_complete_summary_v1.json`，receipt=`b_lds_padding_withdrawal_local_v1.json`及`b_lds_padding_withdrawal_host_v1.json`。队列post和host撤回门禁均整机idle、KFD空、VRAM176582656 B，无本任务GPU进程残留。正式最佳仍107.170/107.223 μs。

pair ScaleA及B padding新增raw证据已回收：`through_b_padding_closeout_evidence_v1.tar.gz`，2257790 B、272份regular files，SHA256=`3ad42046e9a30dd17ac97f930e7abcb55f0632311c5a5fe67b1743530dd7c906`；全部member SHA验收通过，receipt=`through_b_padding_closeout_unpack_acceptance_v1.json`。旧默认ATT专用receipt和此前增量验收保留。

### M swizzle分组大小的有限候选

以精确已验收默认加独立N-group地址fix为基础，新增默认16的`AITER_GFX1250_GEMM1_TASK_M_GROUP=8/32`。仅改变原contiguous-M swizzle的`TILES_PER_GROUP`，保留runtime最后expert优先的M-order排列、相同t256/M32计算分工、所有有效rows与56次有序K128更新、全部input/output地址/同步和量化。不改变routing和benchmark；此处block枚举排列不能保证实际硬件launch顺序或cache复用收益。历史旧nt1的group1与当前M-order组合不同，不重复已结束的group1实验。

CPU直接执行实际swizzle与runtime M-order AST，350种expert ends/空expert/0..256边界/容量尾部拓扑、8/16/32三种group，共2049984个M/N排列点。每个capacity及live M/N tile恰好覆盖一次，N cluster peers选择同一M，capacity-only M排列保持。默认nested AST归一化后等于精确基础，全部compute/operand/TDM/epilogue函数及GEMM2/helpers保持。审计=`task_m_group_source_audit_v1.json`，快照=`task_m_group_v1_source/`。实际compile、双seed精度和完整性能待验收，当前正式最佳仍107.170/107.223 μs。

### M swizzle分组的actual compile与双seed精度

三项actual compile均在a07-3/dsr1_ep4完成，rc0、未超过120秒；各4482 executable instructions、VGPR486/SGPR68、LDS282624 B，zero VGPR/SGPR spills和private segment。group16完整ELF仍为已验收默认`ffb045d35e3c249bce5f068458c534ff98e807dcfe84977a5b17c97def44ee0d`；group8为`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`，group32为`c9ff8ac272f36dabf0c0b0b302f4f93ecadfdcb8c277144ee71e3cb8141636f6`。全部native math、input/output TDM与硬件同步static counts保持，八条backedges及各区间两个barrier sites保持。验收=`task_m_group_compile_acceptance_v1.json`；这些结果只证明code object和资源合格，不能证明实际cache复用或性能收益。

group8/32各seed0/17的四项GPU精度验证均rc0、未超过120秒，各28条零字节差异，原HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay全部通过。四项为correctness-only，计时不使用；验收=`task_m_group_correctness_acceptance_v1.json`。固定16→8→16→32→16性能队列已经开始，每组三轮、仅ownership污染允许整组重试、最多三次。正式默认仍107.170/107.223 μs，尚无新的完整性能结论。

### M swizzle分组的完整正向与反向结果

预先固定的16→8→16→32→16顺序完成；group16前控制和group32首次各因第一轮ownership污染整组三轮重试一次，其他组首次即合格。group8比前后合格控制快0.459%/0.601%，因此按固定8→16→8顺序做一次反向确认，没有按耗时追加筛选。完整label前缀=`a07_mgroup`，30轮全部rc0、未超过120秒：

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `16_control_a_v1_try1` | 107.216,107.556,107.577 | 不统计 | 94.1565,94.5064,94.2620 | 257.006,258.051,258.165 | 第1轮污染 |
| `16_control_a_v1_try2` | 107.168,107.958,107.490 | 107.490 | 94.6353,93.8928,93.7202 | 257.401,257.954,256.513 | 三轮合格 |
| `8_perf_a_v1_try1` | 106.997,107.018,106.647 | 106.997 | 93.2919,93.8719,94.4201 | 256.944,256.774,257.604 | 三轮合格 |
| `16_control_b_v1_try1` | 107.161,107.644,109.369 | 107.644 | 95.2741,94.2913,94.8606 | 258.376,258.007,260.535 | 三轮合格 |
| `32_perf_a_v1_try1` | 109.166,109.253,109.419 | 不统计 | 94.7976,94.4033,93.7305 | 259.877,260.337,259.394 | 第1轮污染 |
| `32_perf_a_v1_try2` | 107.378,109.864,110.619 | 109.864 | 94.8439,94.5255,95.2972 | 258.762,260.282,262.365 | 三轮合格 |
| `16_control_c_v1_try1` | 107.828,106.894,107.712 | 107.712 | 94.5108,94.6014,94.7343 | 258.821,257.775,259.156 | 三轮合格 |
| `8_reverse_a_v1_try1` | 108.104,107.741,107.016 | 107.741 | 94.6563,95.0104,94.2979 | 260.325,259.044,257.828 | 三轮合格 |
| `16_reverse_control_v1_try1` | 108.234,110.488,108.178 | 108.234 | 94.9516,94.7512,94.6838 | 258.985,260.687,258.778 | 三轮合格 |
| `8_reverse_b_v1_try1` | 107.661,107.503,108.629 | 107.661 | 94.6943,95.2109,94.5215 | 257.443,258.191,258.581 | 三轮合格 |

反向两组group8比中间group16低0.456%/0.529%，与正向方向一致，保留这个小幅收益；group32比两侧控制慢2.061%/1.999%，撤出可选范围。全部结果=`task_m_group_complete_summary_v1.json`，固定反向脚本=`queue_task_m_group_reverse_v1.sh`；正向及反向post均整机idle、KFD owners为空、VRAM176582656 B，无本任务GPU进程残留。

收敛源码只将原target守卫、单N-task、rowtail3和M-order开启条件下的默认group改为8；显式关闭M-order、使用其他rowtail或四任务fallback时仍为group16。kernel nested AST完整等于已经双seed验收的candidate，其他计算、TDM、同步和GEMM2保持；CPU实际dispatch审计通过。源码=`task_m_group_default_v1_source/`，三源SHA为`5b6ff443b2c391ab666e04aa720e36de6888918d0600165d295d0e922e446e38` / `d29108153f58a778f6510c201a58e12eaa1a45cabd15a457548aecd5aa5cdae5` / `1dfd0816f18e363d9dd27396db7485a55e07a3b05cfbd3b68ccec49153cb526e`。独立默认compile、双seed/T1537和原用户命令验收待执行，正式最佳暂仍107.170/107.223 μs；106.997 μs为candidate完整组，未替代独立默认数据。

### group8默认的独立编译、精度与原命令验收

`queue_task_m_group_default_v1.sh`已经结束，group8、group16和四任务fallback的完整ELF分别等于此前验收的`7c643f...`、`ffb045...`及原HEAD`958d31...`。前两项4482 instructions、VGPR486/SGPR68；fallback4386 instructions、VGPR480/SGPR106；均LDS282624 B、zero spills/private，rc0、未超时。默认seed0、seed17和T1537/seed17三项分别28条零字节差异，原HEAD有效payload/ScaleA/GEMM2/MoE、eager/graph canary及同地址dynamic input/psum replay全部通过。验收为`task_m_group_default_compile_acceptance_v1.json`和`task_m_group_default_correctness_acceptance_v1.json`。

事前固定默认→旧group16→默认；每组三轮，仅ownership污染允许整组重试，最多三次。全部18轮rc0、未超过120秒，原始污染样本保留，label前缀=`a07_mgroup_`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `default_a_v1_try1` | 107.370,107.246,106.196 | 不统计 | 95.8497,95.1265,94.7723 | 258.937,258.039,256.521 | 第2轮污染 |
| `default_a_v1_try2` | 107.634,107.467,108.557 | 不统计 | 93.3829,94.8759,94.7091 | 256.692,258.651,259.405 | 第2轮污染 |
| `default_a_v1_try3` | 107.625,106.761,106.708 | 不统计 | 95.3099,94.2095,95.3859 | 259.905,256.877,258.440 | 第2轮污染 |
| `old_group16_v1_try1` | 110.278,108.462,106.923 | 108.462 | 93.9902,94.7616,94.9623 | 259.861,258.798,257.604 | 三轮合格 |
| `default_b_v1_try1` | 107.575,109.051,106.887 | 不统计 | 94.7534,94.8377,94.2496 | 259.160,260.696,257.647 | 第1轮污染 |
| `default_b_v1_try2` | 106.059,107.330,107.505 | **107.330** | 95.5735,95.6895,94.7512 | 257.938,258.868,258.356 | 三轮合格 |

当前group8默认中值107.330 μs，比相邻group16对照108.462 μs低1.044%；结合前一节正反向一致的小幅改善保留group8。前默认三次整组均污染，不能声称本节本身具有完整双侧合格对照，也没有追加第四次或挑单轮。当前独立默认比原HEAD历史153.405 μs低30.03%，距80 μs仍差27.330 μs；不能称其刷新历史107.170 μs。GEMM2及e2e完整组中值分别95.5735/258.356 μs。最终post整机idle、KFD空、VRAM176582656 B，boot=`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`；本任务没有GPU进程残留。闭环=`task_m_group_default_acceptance_v1.json`，全部组=`task_m_group_complete_summary_v1.json`。

本轮397份raw日志、ELF/ISA/notes/mlir及验收证据增量回收通过逐member SHA验收。包为`through_mgroup_default_evidence_v1.tar.gz`，3737550 B，SHA256=`4bfed7d55c81e94e7551287f41df315dce5ecab5f7f678776a497b74498a2d38`。

### 在ScaleA跨wave barrier等待期间提交wave私有payload

新增默认关闭的`AITER_GFX1250_GEMM1_EARLY_QUANT_PAYLOAD=1`，仅允许原E65/t256/w2x4/b4/K7168单N-task、SiLU fused-quant及staged ScaleA路径。原顺序是全部DS stores→WG barrier→payload TDM→ScaleA TDM。候选顺序为本wave`s_wait_dscnt(0)`→`s_barrier_signal -1`→本wave payload TDM→`s_barrier_wait -1`→ScaleA TDM；最终`tensor_wait(0)`和epilogue前完整input release fence保留。没有修改有效计算、WMMA累加顺序、量化、TDM descriptor数或全局输出地址。

硬件依据为Shader Guide §4.7.2（p160–161）：DScnt跟踪DS完成，DS同wave有序；§4.10.1（p197）：TDM相对DS无序，所以必须在TDM读取前显式等待本wave的DS stores。CDNA5 ISA §5.6（p45–46）允许将barrier signal和wait分开。CPU直接执行实际payload与ScaleA LDS地址表达式，259种row边界、2244608个payload store字节和132608个scale store字节通过：每wave独享2048 B payload，全部payload区`[0,16384)`与ScaleA区`[16384,17408)`不重叠；每行ScaleA四个字节仍由同M组的四个N waves写入，ScaleA TDM保留全WG DS完成和wait依赖。关闭分支的完整launcher AST与已验收group8默认相同，全部compute/input及GEMM2/helper函数保持。审计=`early_quant_payload_source_audit_v1.json`。

### early quant payload的实际验收与撤回

实际编译的mode0完整ELF等于group8默认`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`；mode1为`6e48892f3014fd30ee857b35723e58f689ad661b964d2f761bbfd0e9368a7c81`，4481 instructions、VGPR486/SGPR68、LDS282624 B、zero spills/private。actual ISA确认DS done→signal→payload→wait→ScaleA顺序，compute、DS/TDM数量和barrier pairs保持；这只证明实现符合候选设计。`early_quant_payload_compile_acceptance_v1.json`记录完整验收。

seed0/seed17各28条零字节差异，原HEAD有效payload、ScaleA、GEMM2/MoE、eager/graph canary及同地址dynamic input/psum replay通过，rc0、未超时。两项为correctness-only，耗时不作性能数据。精度记录=`early_quant_payload_correctness_acceptance_v1.json`。

事前固定0→1→0，每组完整三轮原命令；只因ownership污染整组重试，最多三次。共12轮rc0、未超过120秒，label前缀=`a07_earlyqpayload`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 107.927,107.765,109.083 | 107.927 | 95.5233,95.4748,95.4201 | 259.105,259.743,261.032 | 三轮合格 |
| `1_perf_a_v1_try1` | 106.898,107.465,108.184 | 107.465 | 95.2344,94.9478,94.2751 | 258.061,258.964,258.393 | 三轮合格 |
| `0_control_b_v1_try1` | 107.341,110.234,106.624 | 不统计 | 94.4269,94.7434,95.9744 | 257.591,260.880,258.573 | 第1轮污染，整组排除 |
| `0_control_b_v1_try2` | 106.947,106.822,107.379 | 106.947 | 95.1312,92.0907,95.5338 | 257.644,254.972,258.524 | 三轮合格 |

candidate比前控制快0.428%、比后控制慢0.484%，没有两侧一致改善，淘汰；没有追加反向计时或根据耗时筛选。`withdraw_early_quant_payload_v1.py`已在双端恢复精确group8默认三源SHA，生产开关和分支撤出，candidate源码保留在`early_quant_payload_v1_source/`。完整结果=`early_quant_payload_complete_summary_v1.json`，撤回记录=`early_quant_payload_withdrawal_local_v1.json`和`early_quant_payload_withdrawal_host_v1.json`。结束后整机idle、KFD owners为空、VRAM176582656 B，无本任务GPU进程残留。当前默认仍为107.330 μs。

本项114份raw日志、ELF/ISA/notes/mlir和撤回证据已增量回收，包`through_early_quant_payload_evidence_v1.tar.gz`为1067608 B，SHA256=`03624e458151bfcb96fa994c7277b65c8cdea8f52a50ce2a849eb2740f760560`。全部regular members逐项SHA验收通过，receipt=`early_quant_payload_evidence_unpack_acceptance_v1.json`。

### 保持M256 parent的N128/N64、M64 compute组合

新增默认关闭的`AITER_GFX1250_GEMM1_COMPACT_N_GEOMETRY=1/2`，分别在GEMM1 launcher内部选择`t256x128x256/w4x2/b3`和`t256x64x256/w4x1/b3`。两项每wave计算M64/N64，保留外部256行expert alignment、全部N4096覆盖、同一输入/输出分配及13824条有效routes；每个有效accumulator仍按原顺序执行56次K128更新，SiLU、BF16 rounding、FP4/E8M0保持。其与此前w4x2/N256候选不同：N tile和每wave N范围同步缩小，quant epilogue仍是一组原MX32 block；并不是重新执行旧w4x2/N256试验。

较小N tile只产生canonical ScaleA每个四字节dword中的1/2字节，因此LDS按实际N64 blocks紧凑暂存，再用`(scale_rows32,32,Q_SCALE_COLS)`三维TDM和global strides`(global_scale_stride,4,1)`写出，保持原M32/K布局。payload仍用每wave连续M16二维TDM，输出前完整input release fence、DS/WG barrier和最终tensorwait0保持。三buffer下不删除reader fences，也不把并发预算当成已测occupancy。

硬件依据为Shader Guide §3.3.2.1（p32）：gfx1250 wave32 VGPR按16粒度分配、MI450为1024 VGPR/SIMD；§4.7.1（p160）：LDS/WGP$总384 KiB，实际LDS最多320 KiB；§4.10.4（p202–203）：TDM descriptor支持多维tensor和对应strides。资料支持这项资源/布局实验，不能从编译前预算推断实际吞吐或唯一硬件瓶颈。

`audit_compact_n_geometry_v1.py`直接执行生产geometry、input lane bases和output地址AST，257种row边界，3684352次有效M16/K128更新检查、1572864 payload producer/consumer字节、98304 ScaleA字节、5505024 input payload字节和344064 input scale字节通过；112次三buffer K128 ring步骤保持有效输入所有权。原load/compute/TDM/fence helpers和GEMM2/公共helper保持；关闭candidate并归一化新增常量后的完整launcher AST等于精确group8默认。N128/N64的LDS预算分别159744/133632 B。源码=`compact_n_geometry_v1_source/`，审计=`compact_n_geometry_source_audit_v1.json`；actual compile与严格device/replay已通过，性能正在固定队列中验证，当前默认仍107.330 μs。

### compact N的actual compile与严格精度验收

三项compile仅在a07-3/dsr1_ep4执行，全部rc0、未超过120秒。actual ELF、native指令、zero spills/private、compute backedges和最终output fencing验收通过；mode0完整ELF精确等于已经验收的group8默认。

| mode | 内部geometry | executable instructions | VGPR/SGPR | LDS（B） | ELF SHA256 |
|---:|---|---:|---|---:|---|
| 0 | t256x256x256/w2x4/b4 | 4482 | 486/68 | 282624 | `7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692` |
| 1 | t256x128x256/w4x2/b3 | 2726 | 294/68 | 159744 | `0a5be8d4b36d8248fbef16baef500d0ad0640a6fdc2e4e6ce66fcbe8b5a7d4b1` |
| 2 | t256x64x256/w4x1/b3 | 2369 | 307/68 | 133632 | `632c875d62c891df7c9742b6013e8559eb36680005cb40ae2750d18972193cfb9` |

每项output仍先完成DS stores及WG barrier，再提交payload/ScaleA TDM，最后执行`s_wait_tensorcnt 0x0`和`s_endpgm`。static WMMA sites为256/112/80，来自geometry、owner重复路径和三buffer drain改变，不能解释为减少有效计算；每个有效accumulator仍执行56次有序K128更新。资源减少不是实测occupancy：按Shader Guide §3.3.2.1的16-VGPR分配粒度，294/307分别向304/320取整，MI450的1024 VGPR/SIMD预算仍需与WG映射及其他资源一起判断。

首次native验收在GPU dispatch前因parser只接受`s_barrier_wait -1`而失败，actual disassembler输出为等价的`s_barrier_wait 0xffff`。只修正验收parser，kernel/ELF没有重编译或修改；首次失败脚本和日志原样保留为`accept_compact_n_geometry_compile_v1_parser_before_fix.py`及`compact_n_geometry_compile_accept_control_v1_parser_failure.log`。第二次native验收全部通过，记录=`compact_n_geometry_compile_acceptance_v1.json`。

mode1/2各seed0/seed17四项GPU正确性验证全部rc0、未超过120秒，每项28条`differing_bytes=0`。原HEAD有效payload、canonical ScaleA、GEMM2/MoE输出、各三次eager/graph canary及三组同地址dynamic input/psum replay通过。两候选为correctness-only，所有耗时不作为性能数据；验收=`compact_n_geometry_correctness_acceptance_v1.json`。队列结束整机idle、KFD owners为空、VRAM176582656 B、boot=`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，没有本任务GPU进程残留。

随后按预先固定0→1→0→2→0原命令性能队列，每段完整三轮，只有ownership污染允许整组三轮重试，最多三次；两个相邻控制都支持收益时才安排固定反向确认。队列=`queue_compact_n_geometry_perf_v1.sh`，结果如下。

### compact N的完整原命令性能及淘汰

固定0→1→0→2→0全部完成，最后控制第一次第3轮ownership污染，仅因ownership整组三轮重试一次；18轮均rc0、未超过120秒。完整label前缀=`a07_compactngeom`，全部原始样本保留。

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 108.699,109.316,108.568 | 108.699 | 95.0951,94.4373,94.2624 | 259.345,259.711,257.424 | 三轮合格 |
| `1_perf_a_v1_try1` | 151.371,151.685,151.459 | 151.459 | 89.7736,90.1529,90.2901 | 296.951,296.911,297.275 | 三轮合格 |
| `0_control_b_v1_try1` | 107.297,108.467,106.970 | 107.297 | 95.1521,95.2978,95.0192 | 258.495,259.811,258.276 | 三轮合格 |
| `2_perf_a_v1_try1` | 165.765,165.443,167.117 | 165.765 | 91.2387,91.1102,91.3823 | 311.905,311.081,314.250 | 三轮合格 |
| `0_control_c_v1_try1` | 107.132,107.394,107.708 | 不统计 | 94.6313,94.2581,94.7851 | 256.770,257.156,258.183 | 第3轮污染，整组排除 |
| `0_control_c_v1_try2` | 107.223,106.217,109.179 | 107.223 | 96.5899,94.0029,94.1885 | 259.739,256.110,259.039 | 三轮合格 |

mode1比前后控制慢39.338%/41.159%，mode2比前后控制慢54.492%/54.598%。两项没有保留理由，不启动反向或按耗时追加筛选。较小N tile增加WG数和输入/输出request几何，VGPR294/307仍按304/320分配，资源与请求因素都可能影响；这些是候选分析，现有数据不能唯一确定FIFO/cache或实际occupancy原因。完整汇总=`compact_n_geometry_complete_summary_v1.json`，结束标记=`compact_n_geometry_perf_finished_v1.json`。最后整机idle、KFD owners为空、VRAM176582656 B，boot保持，未出现本任务GPU进程残留。

使用`withdraw_compact_n_geometry_v1.py`按当前candidate SHA与精确默认快照恢复三源，不使用Git清理。candidate及失败parser证据保留。下一项仅编译的方向是N128下A register生命周期复用，必须实际VGPR低于或等于256、zero spills/private及同步合格后才考虑device验收；不直接把更低register数或此前另一geometry的rollingA结果当作性能收益。当前默认仍107.330 μs，80 μs目标尚未达到。

双端撤回均已完成，receipts=`compact_n_geometry_withdrawal_local_v1.json`及`compact_n_geometry_withdrawal_host_v1.json`，恢复三源精确等于`task_m_group_default_manifest_v1.json`；独立N-group地址bug fix保留。179份raw日志、完整ELF/ISA/notes/mlir、parser失败与撤回证据增量回收通过逐member SHA验收。包`through_compact_n_geometry_evidence_v1.tar.gz`为1558011 B，SHA256=`d5ab90d344cf08340c54b9d3870a928fe771bea03796e686beff91e186660f2f`，receipt=`compact_n_geometry_evidence_unpack_acceptance_v1.json`。

### compact N128下的单套A register生命周期候选

新候选`AITER_GFX1250_GEMM1_COMPACT_N_A_REUSE=1`仅允许compact N128/w4x2/b3，默认关闭。两套B、ScaleA、ScaleB仍独立，A register tensors共享：下一K128的head只预读B/scales；每个当前M32 pair的全部WMMA消费者发出后，才用下一K128相同pair替换A。末次无下一K128的路径保持，原56次有序K128更新、数值和全部hardware fences保持，不删除同步或手动移除compiler hazard处理。该方向与旧t256/N256/w4x4 rollingA的geometry、WG数和LDS驻留预算不同，旧方案的122 μs不能作为本项收益证据。

`audit_compact_n_a_reuse_v1.py`直接执行actual shared-slot assignment及prime、28个K256 compute helpers，260种边界、2080个pipeline cases、116480次K128 steps、247296个有效M16更新和261632个已执行M16更新通过。每个当前A/B/ScaleA/ScaleB符号值与原K128一致，compute/TDM/hardware fence事件原序一致，A/ScaleA loads多重集保持；output/input地址、原WMMA helpers和GEMM2 AST未改。关闭复用后的完整launcher AST等于compact-N控制，关闭compact-N的原group8路径继续保持。审计=`compact_n_a_reuse_source_audit_v1.json`；源码=`compact_n_a_reuse_v1_source/`。本节只记录CPU符号审计，native register复用、resource预算及device结果仍需actual验收。

### A register复用的两版actual compile与门槛淘汰

两版仅在a07-3/dsr1_ep4编译，共五项全部rc0、未超过120秒，没有GPU dispatch、数值计时或ATT。v1共享A tensors；v2在相同pipeline下把四个LDS lane base的Int32截断提前到base生成，原DS consumers本来已经使用Int32。CPU逐项3072个地址对照一致且位于159744 B arena内；其余G1/GEMM2/输出及WMMA/fence AST保持。

| 项目 | executable instructions | VGPR/SGPR | LDS（B） | ELF SHA256 |
|---|---:|---|---:|---|
| group8默认，两版各一份 | 4482 | 486/68 | 282624 | `7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692` |
| compact N128控制，v1 | 2726 | 294/68 | 159744 | `0a5be8d4b36d8248fbef16baef500d0ad0640a6fdc2e4e6ce66fcbe8b5a7d4b1` |
| A reuse v1 | 2675 | 259/68 | 159744 | `b8d8037ee94213fcb3fbaefd67d7e3f43379e1aa2479843584d34b228c4fbf15` |
| A reuse v2，Int32 bases | 2649 | 258/68 | 159744 | `c2f0b95e41d73f00eec5e1aef722aad8e73e942ac9a6dc7d68c9bc759a4f642a6` |

全部zero spills/private，default完整ELF与已验收group8默认相同；两项reuse保持112个static WMMA、224个DS b128 loads、16/2个TDM load/store及9/9个WG signal/wait，最终output DS/WG fence与tensorwait0通过。259/258按Shader Guide §3.3.2.1的16-VGPR粒度均分配到272，仍没有通过事前规定的≤256门槛；因此停止该geometry/A-reuse方向，不降低门槛或用编译资源直接宣称性能。决定=`compact_n_a_reuse_compile_decision_v2.json`。

`withdraw_compact_n_a_reuse_v2.py`已经在双端恢复精确`task_m_group_default_manifest_v1.json`三源；candidate源码、CPU审计、五份ELF/ISA/notes/mlir和全部原始日志保留。撤回receipts=`compact_n_a_reuse_withdrawal_local_v2.json`及`compact_n_a_reuse_withdrawal_host_v2.json`，host撤回前整机idle、KFD空、VRAM176582656 B。当前默认仍107.330 μs，没有本轮新增有效性能。

### 量化SiLU的独立tanh分批调度候选

默认关闭的`AITER_GFX1250_GEMM1_QUANT_TANH_CHUNK=2/8`仅作用于原E65/t256单N-task、rowtail3/staged-ScaleA的G1 fused-quant epilogue。默认4继续调用未修改的`batched_silu_swiglu`；candidate使用G1文件内该helper的公式副本，只把独立tanh的CHUNK从4改成2/8。全部clamp、tanh输入、`sig=0.5*(1+tanh)`、乘法结合顺序、BF16 RNE、FP4/E8M0及canonical output保持，公共helper和GEMM2不改。没有加入硬件同步变更或跳过有效计算。

依据已有M-order默认ATT，有效448/672/896组epilogue span P50为4055/4027.5/3979 cycles；不同组P50不能相加、换算为kernel μs或据此唯一定位瓶颈。Shader Guide §4.3.7.3（p91–92）规定TRANS co-execution的hazard需要独立VALU或V_NOP，§5.3.5.2.2（p221–222）记录trans32与core/side可并行；保留compiler hazard处理，先检查actual ISA变化及资源，再严格device/replay和固定相邻性能。

CPU直接执行实际新旧helper AST，6种batch长度、两种activation分支、三种CHUNK的36项逐符号表达式和tanh调用值/顺序完全相同。只移除新增call-selection及诊断suffix后，完整G1 launcher AST等于group8默认；GEMM2、amax helper、输入、compute和hardware fences保持。源码=`quant_tanh_chunk_v1_source/`，审计=`quant_tanh_chunk_source_audit_v1.json`；本节仅记录设计和CPU审计，尚无device或性能结论。

### tanh分批调度的actual compile、精度和完整性能淘汰

三项actual compile仅在a07-3/dsr1_ep4运行，全部rc0、未超过120秒。CHUNK4完整ELF等于group8默认；CHUNK2/8分别为4532/4406 instructions，默认4482。三项均VGPR486/SGPR68、LDS282624 B、zero spills/private；256个static WMMA、384个DS b128 loads、20/2个TDM load/store、10/10个WG signal/wait、128条tanh及16条native FP4转换保持，最终output fencing验收通过。CHUNK8在资源不变时减少指令，按事前编译选择进入device；CHUNK2增加指令，没有执行device或性能。该选择不能当作性能收益。验收=`quant_tanh_chunk_compile_acceptance_v1.json`。

`a07_quanttanhchunk8_seed0_v1`与`seed17_v1`各rc0、未超时，28项逐字节比较全部`differing_bytes=0`；原HEAD有效payload、ScaleA、GEMM2/MoE、各三次eager/graph canary及三组同地址dynamic input/psum replay通过。验收=`quant_tanh_chunk_correctness_acceptance_v1.json`，正确性运行的计时不作性能数据。

固定4→8→4原命令对照，每组三轮，只有ownership污染允许整组最多重试三次。下列所有组和原始日志均保留，label前缀为`a07_quanttanhchunk`：

| 后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `4_control_a_v1_try1` | 107.358,110.811,106.516 | 不统计 | 94.7432,94.5553,94.8398 | 257.827,260.924,257.196 | 第2轮污染 |
| `4_control_a_v1_try2` | 107.022,108.108,108.115 | 不统计 | 94.4288,94.8484,93.4021 | 256.713,258.531,258.518 | 第2/3轮污染 |
| `4_control_a_v1_try3` | 105.534,107.826,107.223 | 不统计 | 95.0255,94.4543,95.3186 | 254.666,258.207,258.537 | 第1轮污染 |
| `8_perf_a_v1_try1` | 107.722,105.355,107.832 | **107.722** | 94.4817,93.8723,94.1885 | 258.524,254.137,257.587 | 三轮合格 |
| `4_control_b_v1_try1` | 109.089,105.669,107.240 | 不统计 | 93.8765,95.5126,92.6850 | 260.504,257.639,256.273 | 第2轮污染 |
| `4_control_b_v1_try2` | 106.240,108.979,106.122 | **106.240** | 94.2579,93.4338,94.6881 | 257.408,258.382,256.119 | 三轮合格 |

18轮全部rc0、未超过120秒。候选合格中值比后置控制慢1.3949%，前置控制三组均缺完整ownership验收，没有两侧一致收益，因此淘汰，不追加反向计时或第四次ownership重试。105.355 μs仅为候选单轮，106.240 μs是本批控制组中值，均不替代独立默认107.330 μs。完整摘要=`quant_tanh_chunk_complete_summary_v1.json`。`withdraw_quant_tanh_chunk_v1.py`按candidate SHA与精确group8快照恢复三源，撤回实验参数与G1-local helper，保留源码及全部证据。结束后整机idle、KFD空、VRAM176582656 B，boot保持`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，无本任务GPU进程残留。

已有M-order默认ATT的细分分析保存在`default_epilogue_detail_analysis_v1.json`。896组128条tanh，相邻issue gap P50为4 cycles，first→last tanh span P50为2382 cycles；payload TDM→wave end P50为1162 cycles。它复用旧group16实际ELF `ffb045...`的已验收采集，没有新抓group8 trace。各segment中值不能相加或换算kernel μs。`default_cost_model_inspection_v1.json`确认原benchmark的executed FLOP与effective traffic来自Python逻辑模型，并非实测HBM请求；不能据9.95 TB/s的逻辑数字唯一确定memory瓶颈。

### 只改变A的LDS padding

新增默认16的`AITER_GFX1250_GEMM1_A_LDS_PADDING=0/128`，仅允许原E65/t256/w2x4/b4/K7168单N-task、rowtail3、staged ScaleA、halfA/halfScale、pairA和rowaload2守卫。每1024 B的A padding由16 B改成0/128 B，并同步改变A consumer subtile/row stride和stage bases/pitch；B始终保持原16 B padding。计算分工、13824条有效routes、每个有效accumulator的56次有序K128更新、全部input/output fences、SiLU、BF16 RNE、FP4/E8M0及canonical output保持。GEMM2和公共helper未改。

实验依据为Shader Guide §4.10.2（p199–200）：padding interval应为128 B或更大，非128 B粒度的padding可能使部分操作采用indirect copy。文档没有证明当前kernel的实际copy模式，也不能由此断言padding是主要瓶颈；先用实际编译和设备对照验证。

CPU直接执行实际A TDM和consumer地址AST，11010048个A fragment字节坐标逐项一致；归一化新增参数、常量及选择后，默认完整launcher AST等于group8，46个nested函数保持。A-only布局预算如下；这些预算不是实测occupancy或性能。

| A padding（B） | A row（B） | B row（B） | stage pitch（B） | LDS预算（B） |
|---:|---:|---:|---:|---:|
| 16 | 2080 | 2080 | 70656 | 282624 |
| 0 | 2048 | 2080 | 70144 | 280576 |
| 128 | 2304 | 2080 | 74240 | 296960 |

源码=`a_lds_padding_v1_source/`，manifest=`a_lds_padding_source_manifest_v1.json`，CPU验收=`a_lds_padding_source_audit_v1.json`。三源SHA为`178bf451d3981bdd1f33bc9ea69c99dbeb0bbe09be705f9d3ca667e6c554b432` / `5af87c195979ff09d2dad3f55a9a0b7049b1d13cf627a8bfc19260dc79ad9596` / `253e7f27941544459f0b26215a3a031ee788edd568af914c0c980d3d74ea1362`。actual compile首先要求默认16完整ELF等于group8、math/DS/TDM/fence数量保持、K backedges及output fencing通过；zero spills/private且按16粒度分配VGPR≤512后才进入device。之后seed0/17各28项零字节差异、eager/graph canary和同地址dynamic input/psum replay全部通过才可计时。

事前固定16→0→16→128→16原命令对照，仅按compile/精度门槛省略失败候选；每段完整三轮，只有ownership污染允许整组最多重试三次，不按耗时追加或选单轮。实现与CPU审计完成时尚无本项device或性能结论，当前默认仍107.330 μs。

A-padding v1的首个默认compile在导入阶段超过120秒，rc124，队列立即停止且没有生成ELF。随后只加`faulthandler`的诊断运行也在120秒终止；重复stack位于Python `ast.iter_child_nodes` / `fix_missing_locations`，没有kernel dispatch。已读取容器实际`flydsl/compiler/ast_rewriter.py`：`RewriteBoolOps.visit_BoolOp`对长`and`链的已有result在test/body/orelse重复引用，导致递归遍历展开量快速增加。该项新增guard含长链，拆为五条相同适用范围的constexpr asserts；16384组独立布尔组合全相等，实际A fragment地址重新审计仍11010048项一致。没有修改或替换安装的FlyDSL rewriter。

修正源码=`a_lds_padding_v2_source/`，persistent SHA=`1816d7a51caeddf04138f7b5de7eec54879768ab8ca6ebcfff324c078e830d6c`，wrapper/compiler SHA保持v1。`a_lds_padding_guard_equivalence_v2.json`保存guard证明。生成v2 shell时首次写入CRLF，`bash -n`发现`$'do\r'`；已将生成脚本改为逐字节LF写入，语法检查通过后才重启compile。CRLF失败日志保留为`queue_a_lds_padding_compile_v2_crlf_failure.control.log`。随后默认16已生成完整ELF，精确等于原group8的`7c643f...`，4482 instructions、VGPR486/SGPR68、LDS282624 B、zero spills/private；该时间点candidate compile/精度和性能尚待完成。

### A-only padding的完整验收与性能淘汰（2026-10-18）

v2三项actual compile全部在a07-3/dsr1_ep4完成，rc0、未超过120秒。default完整ELF仍相同，math/DS/TDM/fence数量、八条K backedges和最终DS→signal→wait→payload→ScaleA→tensorwait0全部通过，三项均zero spills/private。`a_lds_padding_compile_acceptance_v2.json`中的实际资源如下：

| A padding（B） | instructions | VGPR/SGPR | LDS（B） | ELF SHA256 |
|---:|---:|---|---:|---|
| 16 | 4482 | 486/68 | 282624 | `7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692` |
| 0 | 4489 | 486/70 | 280576 | `6c7185a9de9c904cd630a9c6b0f01bb9cc614c902664781155904f01bf46c482` |
| 128 | 4482 | 486/70 | 296960 | `b06c2a5e2409d51be04288daa7873be6e90fef977ed909050534c642124884f8e` |

padding0/128各seed0/17共四项GPU验证均rc0、未超过120秒，各28项`differing_bytes=0`；HEAD有效payload、ScaleA、GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay全部通过。该四项是correctness-only，计时不使用；精度验收=`a_lds_padding_correctness_acceptance_v2.json`。

事前固定16→0→16→128→16完整三轮性能队列已结束。27轮全部rc0、未超过120秒；前控制三次整组均污染，padding0和后控制各因ownership整组重试一次，没有第四次控制或单轮筛选。label前缀为`a07_aldspad`：

| 后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `16_control_a_v2_try1` | 106.968,109.675,105.738 | 不统计 | 95.3713,95.0360,93.9672 | 258.010,260.643,255.824 | 第3轮污染 |
| `16_control_a_v2_try2` | 107.377,107.303,106.828 | 不统计 | 94.8209,93.6255,94.1802 | 258.316,257.070,256.574 | 第1/2轮污染 |
| `16_control_a_v2_try3` | 107.497,106.873,108.408 | 不统计 | 93.9987,93.4528,94.4815 | 258.181,253.547,257.135 | 第2轮污染 |
| `0_perf_a_v2_try1` | 108.222,106.289,106.217 | 不统计 | 94.0599,95.0741,95.7423 | 257.886,257.802,257.068 | 第1/3轮污染 |
| `0_perf_a_v2_try2` | 110.938,107.611,108.395 | **108.395** | 93.6993,94.6987,93.8218 | 260.468,257.573,258.767 | 三轮合格 |
| `16_control_b_v2_try1` | 106.253,108.115,107.144 | **107.144** | 95.5652,92.9319,95.0719 | 257.472,257.207,258.158 | 三轮合格 |
| `128_perf_a_v2_try1` | 106.955,109.076,110.151 | **109.076** | 93.8805,95.1414,93.6550 | 257.069,259.589,259.943 | 三轮合格 |
| `16_control_c_v2_try1` | 107.001,107.550,108.591 | 不统计 | 94.1568,95.1353,94.6292 | 256.584,257.732,258.225 | 第1/3轮污染 |
| `16_control_c_v2_try2` | 106.687,107.811,107.379 | **107.379** | 94.6229,92.1136,93.9838 | 257.491,256.248,257.504 | 三轮合格 |

padding0比相邻合格控制慢1.1676%；padding128比前后合格控制慢1.8032%/1.5804%。两项淘汰，不追加反向或根据耗时加样本。完整结果=`a_lds_padding_complete_summary_v2.json`，SHA256=`264868997e4cda9979a43fe929b750b8937c6056b2c14fa8ed8a4501de9b4928`。`withdraw_a_lds_padding_v2.py`按candidate SHA和`task_m_group_default_v1_source/`精确恢复双端三源，保留候选、启动失败、全部GPU原始日志与编译产物。队列结束及host撤回门禁均整机idle、KFD空、VRAM176582656 B、boot不变，没有本任务GPU进程残留。当前正式默认仍107.330 μs。

### 当前group8默认的独立ATT

撤回后使用用户原`get_att_b_rocm_sdk.sh`在a07-3/dsr1_ep4采集SIMD0、四SE、random原目标、diagnostic iters2，rc0、未超过120秒、ownership合格、无live post owners。actual code object与独立默认compile的完整ELF逐字节相同，SHA256=`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`；missing PCs为空、无`.long`，56个waves和四份SE raw通过验收。UI=`e65_g1_group8_default_att_v1_20261018/thread_trace/simd0/kernel/rpf_v3/ui_output_agent_21693_dispatch_58`，receipt=`group8_default_att_acceptance_v1.json`。

| 动态WMMA数 | 有效wave数 | compute span P50（cycles） | compute WG wait累计P50 | compute DS wait累计P50 | epilogue span P50 | epilogue tensor wait累计P50 |
|---:|---:|---:|---:|---:|---:|---:|
| 448 | 6 | 20714 | 2711.5 | 832 | 4036.5 | 986.5 |
| 672 | 9 | 21817 | 1511 | 813 | 4123 | 959 |
| 896 | 21 | 22177 | 1082 | 2022 | 4195 | 1154 |

另20个capacity-only waves没有WMMA，跨度P50为897 cycles。各有效组compute span没有按动态WMMA数同比缩短，显示padding裁剪后的剩余时间同时包含operand请求、调度和必要的WG同步；它不能唯一定位某个硬件FIFO/cache、证明HBM饱和或直接换算kernel μs。不同组和segment的中值不能相加。这份ATT绑定当前group8，而此前`ffb045...`采集属于group16，不混称为同一code object。

A-padding增量证据已回收本地，archive=`through_a_padding_closeout_evidence_v2.tar.gz`，2372516 B、267个regular members逐SHA通过，archive SHA=`ddfc74a1b36dd564ffca3ab187135f3861e65a1c2c9128b9e13a828c443d0ad6`。当前group8的raw/decoded/ISA/UI全部另存`group8_default_att_evidence_v1.tar.gz`，29685011 B、135个regular members逐SHA通过，SHA=`65a2eb5ae4e94d9412f7317cc9e6aa16d4a569e6e641cfcb0f6af8a6c7a8f957`。传输和打包均在性能及ATT结束后完成。

### compact N128下只复用一组B registers

新增默认关闭的`AITER_GFX1250_GEMM1_COMPACT_N_B_REUSE=1`，仅在compact N128/w4x2/b3、已有A-reuse2条件下实验。第一组B仍独立ping-pong，第二组B registers在当前K128全部WMMA消费者完成后才替换为下一K128值；下一B0与ScaleA/ScaleB仍预读，两套scales独立、共享A逐pair更新保持。这里只缩短operand生命周期，不减少有效routes或改变每个accumulator的56次有序K128更新、SiLU、BF16 RNE、FP4/E8M0、canonical output、input/output fences及GEMM2。

`audit_compact_n_b_reuse_v1.py`直接执行实际新增shared-slot assignment、load/compute/prefetch AST，260种row边界、2080个pipeline cases、116480次K128 steps、247296个有效M16更新和261632个执行M16更新通过；每次WMMA读取的A/B/ScaleA/ScaleB与原K128相符，load多重集、compute/TDM/hardware fence事件保持。关闭新参数后的完整launcher AST等于A-reuse2 archive，关闭compact-N的group8路径保持；GEMM2/helpers及全部输入输出地址函数保持。

源码=`compact_n_b_reuse_v1_source/`，manifest三源SHA为`8721ecd37d34ba3b66f14cca93e4a354510f6079b27dbcb9dc18936610d8b671` / `3d868ae5d3d14a7dca012591dc5e15667073091ecb7334df44a932a4ad51c631` / `a591646ded88e6e9ad5cf4c1a73b79ff7172f36fc4161789831674782f520453`。保持事前门槛：actual VGPR≤256、zero spills/private、默认完整ELF相同、compute/同步/输出验收通过后才做双seed精度；精度通过后才按固定group8→candidate→group8完整三轮原命令对照，ownership污染整组最多三次。VGPR预算和CPU审计不代表实际occupancy或性能；此时尚无device结果，正式默认仍107.330 μs。

两个actual compile均rc0、未超过120秒、无GPU dispatch；关闭candidate的完整ELF精确等于默认。shared-B候选=`fc2bab881ee803291a32636da1f745b904d56cb893d9d1d5120dc74f24825a23`，2665 instructions、VGPR226/SGPR68、LDS159744 B、zero spills/private。按Shader Guide §3.3.2.1的16-VGPR粒度分配为240，已低于事前256门槛；这是资源结果，不能报告性能收益。112个static WMMA、224个DS b128 loads、16/2个input/output TDM、9/9个WG signal/wait及最终output fencing保持；八条backedges每区间一条signal加一条wait均通过。

首个compile验收脚本把“两个barrier sites”误写为两条signal和两条wait，且预期compact只有四条backedges，实际默认和compact均有八条。已通过`compact_n_b_reuse_backedge_inspection_{default,sharedb}_v1.json`核对实际区间，修正parser后验收通过；未改kernel、重编译或放宽hardware fence要求。失败parser源码和日志保留。验收=`compact_n_b_reuse_compile_acceptance_v1.json`，双seed/device replay随后进行。

shared-B候选的seed0/17验证均rc0、未超过120秒，各28项零字节差异；MoE、eager/graph canary及同地址dynamic input/psum replay全部通过。验收=`compact_n_b_reuse_correctness_acceptance_v1.json`，两项log SHA分别为`f8c58763d3c0435c24076404cd8da0f0acfe4ea13e4fdf07de2aaef2d27782ab2`和`d982c7d9a3af5bfd505d598b4a1dda7faca14e037ce55771d45f675ca1ffc93e8`。这是correctness-only结果，不使用其中计时。

固定正向性能方案=`queue_compact_n_b_reuse_perf_v1.sh`，顺序group8→shared-B→group8，每段原用户命令完整三轮。仅ownership污染允许整组三轮重试，最多三次；不按单轮耗时追加或筛选。测前自动每6秒等待全机GPU空闲，测试120秒上限不变；只有相对两侧完整合格中值均改善，才执行一次固定反向确认。正式默认仍107.330 μs，shared-B尚无正式性能结论。

### shared-B的完整性能：资源降低未转化为收益

固定正向三段全部第一次完成ownership验收，九轮rc0、未超过120秒，没有按耗时筛选或追加。label前缀=`a07_compactnbreuse_`，完整记录=`compact_n_b_reuse_complete_summary_v1.json`：

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） |
|---|---|---:|---|---|
| `group8_control_a_v1_try1` | 107.412,106.438,109.093 | 107.412 | 94.1251,95.2091,93.6274 | 257.183,257.097,258.885 |
| `sharedb_perf_a_v1_try1` | 115.239,115.895,115.587 | 115.587 | 95.0804,94.8231,94.7114 | 266.170,266.334,265.889 |
| `group8_control_b_v1_try1` | 107.001,109.464,106.887 | 107.001 | 94.8569,96.0439,95.0803 | 257.055,262.039,257.756 |

shared-B比前后控制慢7.611%/8.024%，不接入默认、不启动反向。VGPR226、zero spills与LDS159744 B仍没有达到原group8性能，不能据资源预算断言实际occupancy或唯一瓶颈。下一步用用户原脚本采集同一候选ELF的SIMD0/fourSE ATT，检查阶段跨度和wait，再恢复精确group8三源。正式默认仍107.330 μs。

shared-B ATT三次采集均rc0、未超过120秒，第一次及try2/try3结束检查均发现其他任务的live GPU/KFD owner，因此三次均不用于性能分析。外部任务为`yadai`中的`op_tests/test_moe_ep.py`，try2还出现另一项`dsr1_ep4/.codex_tmp/gemm1_t1536_20261017/compile_candidate_cpu.py`；没有杀死这些进程，没有追加第四次或把受污染的wave cycles描述为瓶颈证据。三个trace及首个验收失败保留，结束记录=`compact_n_b_reuse_att_finished_v1.json`，`accepted=false`。

双端`withdraw_compact_n_b_reuse_v1.py`已恢复`task_m_group_default_manifest_v1.json`三源精确SHA，独立N-group地址修复保留，未执行Git清理。host自动每6秒等空闲后于`2026-10-18T05:07:40Z`完成撤回，KFD为空；receipts=`compact_n_b_reuse_withdrawal_{local,host}_v1.json`。正式九轮性能验收和被拒绝ATT单独保留，不能混合。

### compact N128扩大A multicast cluster的有限候选

新增默认4的`AITER_GFX1250_GEMM1_COMPACT_N_CLUSTER=8`，仅允许compact N128/A-reuse2/shared-B组合。每个WG仍t256x128x256/w4x2/b3、相同计算与canonical输出，A multicast mask和launch definition/site统一改为8个N peers；不改每个有效accumulator的56次有序K128更新、SiLU、BF16 RNE、FP4/E8M0、routing、input/output fences及GEMM2。理论上每个A broadcast覆盖的N范围从512扩大到1024列，是否降低实际请求成本取决于设备结果。

硬件依据为Shader Guide §2.3（p24）：cluster可包含最多16个WG，同一SE内各WG在不同WGP运行，WGP可承载来自不同cluster的WG。它支持8-WG设计边界，不证明实际co-residency或吞吐收益。没有从被ownership拒绝的shared-B ATT推断这项收益。

`audit_compact_n_cluster_v1.py`执行实际constants、M swizzle/M-order和multicast mask AST，272种expert拓扑/0..256尾部/空expert/capacity tail、2071552个M/N排列点通过。每个live/capacity tile只覆盖一次，cluster peers选择相同parent M，mask精确包含4/8 peers；除cluster常量与守卫外完整launcher AST等于已经双seed验收的shared-B版本，全部nested compute/load/epilogue/input-output地址及GEMM2保持。LDS仍159744 B，关闭compact时group8保持。

源码=`compact_n_cluster_v1_source/`，manifest三源SHA为`3f10ba45e9e4505acf22374eaf0af8624529e06b17a3972abbbd233f7ce89260` / `e7312c86fd09734a051e8f9d17b81e444108668dac037f85d727b560809d18df` / `7c6af7c7729d97b698e8f7c0e1533c540b7c9aef96b9c498b352c0e7d9818fb6`。仅上传任务三源；固定actual compile顺序default→cn4→cn8，每项120秒。必须default/cn4完整ELF保持、cn8 VGPR≤256、zero spills/private及hardware fences合格后才做device精度，精度通过后才做原命令完整性能对照。当前尚无cn8 native/device性能结论，正式默认仍107.330 μs。

三项actual compile全部rc0、未超过120秒、无GPU dispatch。default完整ELF仍`7c643f...`、cn4仍`fc2bab...`，cn8为`70de03433751b868c0a222aef58ed800697f7b1a94345bca28810a581fa58733f`，2665 instructions、VGPR226/SGPR68、LDS159744 B、zero spills/private。112个static WMMA、224个DS b128、16/2个input/output TDM、9/9个WG signal/wait及八条compute backedges和最终output fencing均通过。资源门槛通过，已进入双seed严格device验收；验收=`compact_n_cluster_compile_acceptance_v1.json`。编译合格不代表性能收益。

cn8的seed0/17 GPU验收均rc0、未超过120秒，各28项零字节差异，HEAD全部有效payload/ScaleA/GEMM2/MoE、eager/graph canary和同地址dynamic input/psum replay通过。两项均correctness-only，不使用计时；验收=`compact_n_cluster_correctness_acceptance_v1.json`，log SHA为`56439ddc490b6433e69d803a0b519371f1fd767516d81ac8e7328e382f44947a2`及`bd4427d3c06917d981c7a8736224deca1382b4aaa5330638f81c4f1e0333e6ea4`。

固定group8→cn8→group8原命令性能队列已安排，每段完整三轮，只因ownership污染允许整组最多三次尝试。脚本=`queue_compact_n_cluster_perf_v1.sh`，尚无新正式性能结论。

### compact cn8的完整性能及撤回

固定三段队列已结束，候选第一次第1轮ownership污染，仅因此重跑整组三轮；十二轮全部rc0、未超过120秒，没有按耗时选择或追加反向。label前缀=`a07_compactncluster_`：

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `group8_control_a_v1_try1` | 108.389,108.707,109.662 | 108.707 | 93.9079,94.8251,93.7140 | 257.613,259.825,258.988 | 三轮合格 |
| `cn8_perf_a_v1_try1` | 127.392,127.609,127.997 | 不统计 | 90.5283,91.4138,91.2009 | 273.388,274.150,274.378 | 第1轮污染 |
| `cn8_perf_a_v1_try2` | 127.881,127.280,127.487 | **127.487** | 91.1965,90.2772,91.7026 | 274.268,272.697,273.919 | 三轮合格 |
| `group8_control_b_v1_try1` | 106.809,107.161,110.293 | 107.161 | 91.3442,94.6247,94.3782 | 253.745,257.679,260.127 | 三轮合格 |

cn8比前后控制慢17.276%/18.968%，不接入默认，不进行反向计时。完整摘要=`compact_n_cluster_complete_summary_v1.json`，SHA256=`54d6d61ea452ffc2d9dc8c4b13c85f565ba8e8348e9b5d68a8e43eb0d359542a`。Shader Guide §4.9.8（p191）记载GL1最多合并5个请求形成一次data return，因此8个N peers不能按8倍data return amplification估算；资料与本组回退都不能唯一确定当前kernel的实际merge率或某个cache/FIFO瓶颈。

`withdraw_compact_n_cluster_v1.py`已在双端按candidate SHA守卫恢复`task_m_group_default_manifest_v1.json`三源精确SHA；全部四组raw样本与候选源码保留，未执行Git清理。receipts=`compact_n_cluster_withdrawal_{local,host}_v1.json`。队列postcheck与host恢复门禁均整机idle、KFD owners空、VRAM176582656 B，boot不变，没有本任务GPU进程残留。当前独立默认仍为107.330 μs，距80 μs差27.330 μs。

### 原group8下A payload multicast策略的独立候选

新增默认0的`AITER_GFX1250_GEMM1_A_MCAST_POLICY=1/2`，仅用于E65/K7168原t256/w2x4/b4单N-task、cluster4x1的GEMM1。mode0保持现有A payload multicast和early timeout；mode1保持同一mask，改用standard timeout；mode2仅将A payload改为各WG独立TDM load。B、ScaleA、ScaleB仍使用原策略，input/output地址、ring、每次reader fence、计算顺序及全部数值功能不变，GEMM2保持。该实验没有扩大cluster、改变tile或有效routes。

Shader Guide §4.9.8（p191–192）记载multicast load强制绕过WGP$，early timeout先向数据返回时已提交的requesters广播，后来的请求形成另一批；§4.10.3（p201）说明非零descriptor mask把TDM load转换成CLUSTER_LOAD_ASYNC；descriptor bit21的early timeout定义位于§4.10.4（p204）。这支持比较已有4 peers的请求策略，但不能先断言本kernel存在低merge率、cache命中收益或FIFO瓶颈。

`audit_a_mcast_policy_v1.py`直接执行实际四类job的mask和timeout表达式，三mode×四个local-N的12项策略选择符合设计；移除新增参数、守卫、suffix并还原两处选择后，完整G1 launcher AST等于精确group8默认，wrapper新增keyword以外完整AST相同。B/ScaleA/ScaleB job、全部load/compute/output地址与hardware fence、公共helper及GEMM2 AST保持。源码=`a_mcast_policy_v1_source/`，审计=`a_mcast_policy_source_audit_v1.json`。

先固定0→1→2 actual compile，每项120秒。要求mode0完整ELF等于`7c643f...`，candidate zero spills/private、按16粒度分配VGPR≤512、native有效math/DS/TDM/同步数量及最终output fencing通过；然后mode1/2各seed0/17严格逐字节、eager/graph/dynamic验收。通过门槛后，性能顺序固定0→1→0→2→0，每段完整三轮，仅ownership污染允许整组最多三次，不挑单轮或按耗时追加；只有对两侧控制均改善才进行固定反向确认。此时尚无新device/performance结论，独立默认仍107.330 μs。

三项actual compile已完成，全部rc0、未超过120秒，4482 instructions、VGPR486/SGPR68、LDS282624 B、zero spills/private。mode0完整ELF保持`7c643f...`；mode1=`5f4eface60f5bdc9bb71fc8f6fd42736879de4f70cefca8845ce65fd1462b153`；mode2=`062ccab24a65af1e0be67c6c1a62f350525685f97e975719e898e6897e547113`。native math/DS/TDM数量、八条compute backedges各一处signal加wait及最终output fence通过，验收=`a_mcast_policy_compile_acceptance_v1.json`。

两候选各seed0/17共四项严格GPU验收全部rc0、未超过120秒，各28项零字节差异，HEAD有效payload/ScaleA/GEMM2/MoE、eager/graph canary与同地址dynamic input/psum replay均通过。验收=`a_mcast_policy_correctness_acceptance_v1.json`，correctness-only的计时不使用。固定0→1→0→2→0原命令性能队列在此后启动，不由编译或精度结果声称性能收益。

shared-B/cn8结束后的283份新增raw日志、ELF/ISA/notes/mlir及撤回证据增量回收通过逐member SHA验收。archive=`through_compact_cluster_closeout_evidence_v1.tar.gz`，2502335 B，SHA256=`9e4580deaa3eb27c4474860f1f7bb3980721b666adcc9475904de8a1159f3a56`；独立receipt=`compact_cluster_closeout_evidence_unpack_acceptance_v1.json`。被ownership拒绝的shared-B trace目录原样保留远端，没有用于性能解释。

复用当前group8合格ATT的WMMA gap离线分析为`group8_wmma_gap_analysis_v1.json`。448/672/896组中大于64 cycles的相邻WMMA issue间隔累计P50分别14178.5/13542/12757 cycles，其中穿插了下一operand DS loads、row predicate、ring地址计算与WG同步；该“间隔累计”包含必要指令工作，不能当作全部可消除stall。不同中值不相加，也不换算kernel μs或唯一定位硬件环节。

### A payload multicast策略的完整性能及撤回

固定0→1→0→2→0队列已结束，共21轮全部rc0、未超过120秒。中间和最后的控制第一次分别第3轮、第2轮ownership污染，仅因此整组三轮重试一次；没有根据耗时加样本、拼接单轮或追加反向。label前缀=`a07_amcastpolicy_`：

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 107.531,107.419,108.357 | 107.531 | 95.5655,94.4858,95.0426 | 259.742,258.162,259.427 | 三轮合格 |
| `1_perf_a_v1_try1` | 107.252,106.072,108.798 | **107.252** | 93.4635,93.2209,93.9294 | 256.189,254.759,259.240 | 三轮合格 |
| `0_control_b_v1_try1` | 107.927,107.377,108.256 | 不统计 | 93.6343,93.9759,95.0890 | 257.460,258.076,260.090 | 第3轮污染 |
| `0_control_b_v1_try2` | 106.966,106.567,108.819 | 106.966 | 95.1416,94.3806,94.5764 | 258.002,256.406,259.092 | 三轮合格 |
| `2_perf_a_v1_try1` | 109.344,108.823,109.319 | **109.319** | 94.7873,94.9582,94.8507 | 259.912,259.073,258.729 | 三轮合格 |
| `0_control_c_v1_try1` | 106.829,106.314,107.141 | 不统计 | 95.4285,95.6708,94.5323 | 257.214,257.852,257.896 | 第2轮污染 |
| `0_control_c_v1_try2` | 106.479,107.132,106.538 | 106.538 | 94.7957,95.4874,94.1042 | 257.196,258.693,256.227 | 三轮合格 |

mode1相对前控制快0.259%，相对后控制慢0.267%，没有两侧一致收益；mode2相对前后控制慢2.202%/2.613%。均淘汰，不接入默认或追加反向计时。完整摘要=`a_mcast_policy_complete_summary_v1.json`，SHA256=`435b6bfde121a86a4445781bdd404ca084d328e8079b7ba4b795395fd6a9ab6d`。这些实验未证明A payload timeout或WGP$策略是当前kernel的主要瓶颈。

`withdraw_a_mcast_policy_v1.py`已在双端按candidate SHA守卫恢复`task_m_group_default_manifest_v1.json`三源精确SHA；全部候选源码和原始样本保留，独立N-group地址修复保持。receipts=`a_mcast_policy_withdrawal_{local,host}_v1.json`。队列结束及host撤回门禁均整机idle、KFD owners空、VRAM176582656 B、boot保持`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，没有本任务GPU进程残留。独立最佳默认仍107.330 μs，距80 μs差27.330 μs。

A-policy关闭后的199份新增raw日志、ELF/ISA/notes/mlir及撤回证据已增量回收，archive=`through_a_policy_closeout_evidence_v1.tar.gz`，1813650 B、SHA256=`3be514a08054c6a93b2ac20f475998833dc4121d7b4de0e7ec8a29b09bee3cc5`，逐member SHA验收通过，独立receipt=`a_policy_closeout_evidence_unpack_acceptance_v1.json`。传输和打包均在性能队列结束后执行。

### 保留K128 carry的全operand K512双buffer候选

新增默认关闭的`AITER_GFX1250_GEMM1_K512_TWO_BUFFER=1`，只允许原E65/K7168/t256/w2x4/b4单N-task、cluster4x1及当前rowtail3/staged-ScaleA/halfA/halfScale/rowaload2/pairA守卫。launcher内部改为t256x256x512/w2x4/b2，每个K512仍执行四次原有K128更新，偶数KWS继续使用两套独立register slots，最后一次向slot0预读下一K512的首个K128。M/N geometry、canonical输入输出、全部13824条有效routes、有效accumulator的56次更新及顺序、SiLU、BF16 RNE、FP4/E8M0、GEMM2和公共helper保持。

与旧130us记录（`moe_pipeline_130us_optimization.md`的“ATT diagnosis and rejected K512 transfer unit”）不同，本项没有多N-task或t192 cyclic layout，保留原mid-compute issue和register carry，不采用原`num_buffers<=2`的post-compute/no-carry路径。初始resident/issue lead保持1024个K元素，LDS保持282624 B；但双buffer的boundary READY等待会排空当时全部请求，实际overlap可能降低，因此不能把相同初始lead描述成已证明等效吞吐。候选仅用于检验减少TDM descriptors和跨tile同步是否足以抵消这项成本。

Shader Guide §4.10.1（p197–198）说明同wave的TDM loads/stores有序，但跨wave和相对其他memory类别无序；因此每次boundary仍执行tensor_wait及完整DS/WG reader fence后再refill已释放buffer，同时从另一个已完成的buffer预读carry。§4.10.8（p206）给出3/wave、6/SIMD的issue-to-XACK限制及63的TENSORcnt限制；descriptor数减少不证明吞吐提升。原完整output input-release fence、output DS/WG及最终tensorwait0保持。

`audit_k512_two_buffer_v1.py`直接执行实际constants、consumer lane地址以及prime/steady/drain/load/compute AST。mode0/1、260种row边界、两个M-wave位置共1040个pipeline cases、58240次K128 steps、261632个执行M16更新通过；11010048 payload字节和688128 ScaleA/ScaleB字节逐项对应同一global K128/M/N坐标。当前buffer refill必须晚于其最后reader release、carry target READY在DS reads前、K input tiles按序恰好issue一次；有效M16更新与默认逐项相同。关闭candidate的完整launcher/wrapper AST归一化相同，46个nested load/compute/output函数及GEMM2保持。

源码=`k512_two_buffer_v1_source/`，manifest三源SHA为`23502b48c4636a26858dcbfd9f4f13aa538f70e15a2223894b796337c82db505` / `9e3cb27f813bce53a65ac0c4aa72e8bf09db35235e4acba8435cf956aa83f40d` / `b7fe84ddd57c658b64ab031cd7fe8de3477124e5bf53e594047e3e8769ffaac3`。CPU审计=`k512_two_buffer_source_audit_v1.json`；这些符号结果不是native指令、cross-wave时序、实际occupancy或性能结果。先固定mode0→mode1 actual compile，要求mode0完整ELF精确等于group8、候选zero spills/private、按16粒度分配VGPR≤512及compute backedges/output fences通过，才做seed0/17严格逐字节、eager/graph/dynamic验收。只有精度合格才启动固定0→1→0原命令完整三轮性能对照，ownership污染整组最多三次；当前尚无本项device/performance结论，独立默认仍107.330 μs。

v1两项actual compile均rc0、未超过120秒，无GPU dispatch。mode0完整ELF仍`7c643f...`；K512候选为`ece4601a7b813e48ab1deef7b74afa54c697a0cb160464b61b3e05a80eb77cd2`，5022 instructions、VGPR504/SGPR68、LDS282624 B，但有1个VGPR spill和8 B private segment，未通过zero-spill门槛，不执行精度或性能。actual ISA在0x621C有`scratch_store_b32`，在0x64FC有对应`scratch_load_b32`；这是native编译诊断，不是计时。

有限v2仅提前把四种lane-specific LDS base加法转换为Int32；原DS copy helpers已经将base及byte offset截断为Int32，arena全部在282624 B内。344064项实际base结果及原payload/scale字节坐标全部一致，1040个pipeline cases重新通过；关闭candidate的完整launcher/wrapper AST仍等于精确默认。修改不涉及operand生命周期、WMMA、TDM/ring/refill、数值或hardware fence。源码=`k512_two_buffer_v2_source/`，persistent SHA=`c43b9acf79287a108d33e565347d3f794d70a7618666da5e5e24977afa6ca49b`，wrapper/compiler保持v1；审计=`k512_two_buffer_source_audit_v2.json`。v2只用于消除地址临时值带来的编译压力，仍按相同zero-spill门槛验收，没有降低精度标准或timing规则。

v2两项actual compile均rc0、未超过120秒，candidate为`c1824c7e1b92d7be4dc19b05287994bc27320f853f7478f17498dd0a9252d7c9`，5017 instructions、VGPR503/SGPR68、LDS282624 B、zero spills/private；mode0完整ELF仍精确等于group8。实际四个owner rolled regions共八条backedges，candidate区间分别60/64 WMMA sites，对应原28/32扩大KWS；每个区间一条tensorwait0、一条signal和一条wait，完整DS reader release先于refill TDM。static WMMA384和DS b128576来自较长rolled body及两stage drain；不表示减少有效计算或动态执行量。128条tanh、16条FP4转换以及payload/ScaleA两处TDM保持，两个native退出路径都先tensorwait0。验收=`k512_two_buffer_compile_acceptance_v2.json`，完整inspection=`k512_two_buffer_native_inspection_v2.json`。

seed0/17两项GPU验证均rc0、未超过120秒，各28条`differing_bytes=0`，HEAD全部有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay通过。原MoE精度分别`3.7793e-06/2.7493e-03`和`3.7788e-06/2.7491e-03`；两项是correctness-only，耗时不使用。验收=`k512_two_buffer_correctness_acceptance_v2.json`。接下来按事前固定0→1→0完整三轮原命令队列，只有ownership污染允许整组最多三次，精度/compile合格不提前宣称性能收益。

### K512双buffer完整性能与撤回

固定0→1→0原命令队列结束，共21轮全部rc0、未超过120秒。前控制、candidate第一次各第2轮ownership污染；后控制第一次第1/2轮污染，第二次第1轮污染。只因此重跑完整三轮组，最多三次，没有根据耗时追加或拼接单轮。label前缀=`a07_k512carry2_`：

| label后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v2_try1` | 111.241,106.927,106.727 | 不统计 | 94.4689,95.4325,93.4926 | 261.069,257.542,255.979 | 第2轮污染 |
| `0_control_a_v2_try2` | 109.759,107.493,107.394 | **107.493** | 93.6551,94.0535,94.4668 | 259.882,257.146,258.118 | 三轮合格 |
| `1_perf_a_v2_try1` | 118.752,120.417,123.780 | 不统计 | 92.5882,95.0214,92.8558 | 266.414,270.933,272.377 | 第2轮污染 |
| `1_perf_a_v2_try2` | 120.974,121.541,127.326 | **121.541** | 94.2264,92.1306,93.5263 | 270.159,269.751,276.682 | 三轮合格 |
| `0_control_b_v2_try1` | 108.492,108.686,107.423 | 不统计 | 94.2055,93.7183,91.5467 | 257.677,258.763,254.795 | 第1/2轮污染 |
| `0_control_b_v2_try2` | 106.603,105.962,107.244 | 不统计 | 94.0302,95.5926,94.9770 | 257.170,257.418,257.487 | 第1轮污染 |
| `0_control_b_v2_try3` | 108.007,110.550,107.651 | **108.007** | 94.7894,95.6959,94.4183 | 259.465,262.547,257.860 | 三轮合格 |

K512比前后控制慢13.069%/12.531%，不保留，不启动反向计时。完整摘要=`k512_two_buffer_complete_summary_v2.json`，SHA256=`97e94da56b1f52288f16513c8e405ac67d5efb8381acf887eedf0f5e34039fcc`。减少descriptor和boundary次数没有得到收益；native两个buffer的READY tensorwait0、较长body和503 VGPR是已经确认的设计/资源变化，现有计时不能唯一确定哪个因素解释全部回退，不用未采集的ATT推断cache/FIFO。

`withdraw_k512_two_buffer_v2.py`已按candidate SHA守卫在双端恢复`task_m_group_default_manifest_v1.json`三源精确SHA，候选v1/v2及全部原始组保留，独立下一N-group预取地址fix保持。receipts=`k512_two_buffer_withdrawal_{local,host}_v2.json`。性能结束及host撤回门禁均整机idle、KFD空、VRAM176582656 B，boot仍为`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，没有本任务GPU进程残留。正式独立默认仍107.330 μs，距80 μs差27.330 μs。

### 在原K256/b4路径提前生成Int32 LDS bases

新增默认关闭的`AITER_GFX1250_GEMM1_LDS_I32_BASES=1`，仅允许原E65/K7168/t256/w2x4/b4单N-task、cluster4x1及当前rowtail3/staged-ScaleA/halfA/halfScale/rowaload2/pairA守卫。四种`lds_*_base`由先做index宽度加法、在copy helper处转Int32，变为先转Int32再加。`gemm_common_gfx1250.py::make_lds_copy_ops`已经在每个DS consumer处计算`Int32(base)+Int32(byte_offset)`，全部arena仅282624 B；本项不改变tile、TDM ring、operand lifetime或hardware fences，不再采用K512/b2。

`audit_lds_i32_bases_v1.py`直接执行实际base和lane offset AST，8个waves×32个lanes×56个K128×4种base共57344项结果相同并位于原arena。关闭candidate的完整G1 launcher/wrapper AST归一化相同，42个nested load/compute/TDM/epilogue函数保持；GEMM2、公共helpers、benchmark/routing保持。首次CPU脚本漏提供`TDM_DESCRIPTOR_VERSION`而失败，仅补完整constants环境后通过，未改kernel。源码=`lds_i32_bases_v1_source/`，manifest三源SHA为`e8fe0444c7f73a1d9a69f98b385220624c5b5b390fdca526d11aa94f38f91c36` / `f880b7d5dd9998da972c6179a8c19aa473bd220d0d15875002c45dd2c833a3a0` / `051c5b53b189731e6e2201c5caf8106fb55f765fc72552b3384034d5b9b6a4f7`。

两项actual compile全部rc0、未超过120秒、无GPU dispatch。关闭candidate完整ELF仍`7c643f...`；candidate=`4372771c73779113d9962df2469c56f08c7f20d9cfa78dfc9ed06a7973069e96`，instructions4482→4456、静态`v_add_nc_u64_e32`11→0；VGPR486/SGPR68、LDS282624 B、zero spills/private保持。八条K backedges及四targets、每region28/32 WMMA sites、READY tensorwait2→DS reader release→WG signal/wait→refill TDM顺序、payload/ScaleA output及两条exit最终tensorwait0验收通过。96个static scale dwords保持，scalar DS loads合并方式改变；编译器生成的DS waits/delay和VALU配对也变化，因此不将纯指令减少描述成已经取得性能提升。验收=`lds_i32_bases_compile_acceptance_v1.json`，inspection=`lds_i32_bases_native_inspection_v1.json`。

seed0/17两项严格GPU验证均rc0、未超过120秒，各28项零字节差异；原HEAD全部有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay通过。correctness-only计时不使用；验收=`lds_i32_bases_correctness_acceptance_v1.json`。随后固定0→1→0原命令完整三轮性能对照，仅ownership污染整组最多三次；只有两侧完整控制均支持收益才进行一次固定反向确认。尚无本项正式性能结论，独立默认仍107.330 μs。

### Int32 LDS bases的完整性能与撤回

固定0→1→0队列完成，共27轮全部rc0、未超过120秒。每段仅按ownership污染重试完整三轮，最多三次；没有拼接单轮、第四次重试或按耗时追加。全部label前缀=`a07_ldsi32_`：

| 后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 108.368,109.061,106.774 | 不统计 | 94.1609,93.9122,95.2025 | 258.581,259.320,257.774 | 第1/2轮污染 |
| `0_control_a_v1_try2` | 108.070,108.416,106.824 | 不统计 | 93.7119,95.2235,95.3545 | 258.036,257.612,256.714 | 第1/3轮污染 |
| `0_control_a_v1_try3` | 106.774,107.014,107.741 | 不统计 | 96.5078,94.5426,95.4766 | 258.636,256.582,260.219 | 第2/3轮污染 |
| `1_perf_a_v1_try1` | 107.273,108.115,110.305 | 不统计 | 95.3503,94.9580,94.6946 | 259.210,259.629,261.318 | 第2轮污染 |
| `1_perf_a_v1_try2` | 108.178,107.128,109.262 | 不统计 | 95.3648,94.4012,95.2489 | 258.699,257.506,260.034 | 第1轮污染 |
| `1_perf_a_v1_try3` | 113.352,106.464,106.965 | **106.965** | 95.0171,93.9988,94.5891 | 264.295,255.971,256.574 | 三轮合格 |
| `0_control_b_v1_try1` | 107.604,108.275,108.152 | 不统计 | 94.7283,94.8757,94.5744 | 258.173,259.543,258.402 | 第2轮污染 |
| `0_control_b_v1_try2` | 107.746,108.701,107.701 | 不统计 | 94.3993,93.5896,93.2755 | 257.941,258.307,257.232 | 第3轮污染 |
| `0_control_b_v1_try3` | 107.208,106.497,106.200 | **106.497** | 94.8401,94.7895,92.4996 | 257.821,256.945,254.392 | 三轮合格 |

candidate比合格后控制慢0.4395%；前控制三组均未得到完整ownership验收，不能报告双侧稳定回退或收益。没有接入依据，撤出candidate，不进行反向计时。106.965 μs是本项candidate组中值，106.497 μs是后控制组中值，都不替代独立默认107.330 μs。完整摘要=`lds_i32_bases_complete_summary_v1.json`，SHA256=`66cc6d56817dfa2bc827fbb247fd101879297c8d4a53db8a395f308b068d81b9`。

`withdraw_lds_i32_bases_v1.py`已按candidate SHA守卫在双端恢复`task_m_group_default_manifest_v1.json`三源精确SHA，保留全部candidate/raw/native证据及独立N-group地址fix。receipts=`lds_i32_bases_withdrawal_{local,host}_v1.json`。队列结束及host撤回门禁均整机idle、KFD空、VRAM176582656 B，boot仍`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，没有本任务GPU进程残留。下一项先核对正式routing的M128全padding范围，再决定operand裁剪是否值得部署；当前已验收默认仍107.330 μs。

### 正式routing的M128范围检查与证据回收

`inspect_live_wave_extents_v1.py`仅包装原GEMM1 launcher读取实际psum，保留原random seed0/routing并调用完整原verify；经既有host runner在a07-3/dsr1_ep4执行，rc0、未超过120秒。64个routed experts的counts为164–224，shared expert为1536；共13824条有效routes、70个active parent tiles，其中64个partial、6个完整256。`whole_padding_M128_halves=0`、`partial_M128_halves=64`，HEAD有效输出严格比较通过。因此目标输入中没有可通过跳过整个M128 half而减少B/Scale搬运的工作；本项没有修改kernel、部署候选、编译或记录性能。

诊断summary=`live_wave_extents_seed0_v1.json`，runner原结果=`live_wave_extents_seed0_runner_result_v1.json`。该项为correctness-only，`ownership_qualified=false`；即时post快照曾见退出中的PID3147308、busy2和VRAM7067209728 B，但`live_post_owners=[]`，随后settled快照KFD空、busy0、VRAM176582656 B、整机idle，PID再次检查已经退出。这里不使用任何诊断耗时。group8已验收ATT的20个无WMMA waves属于capacity尾部，TDM loads和DS loads均为0，不能把它们视为仍有operand搬运的全padding compute halves。

K512和Int32 LDS候选关闭后的395份新增日志、ELF/ISA/notes/mlir及撤回证据已回收，archive=`through_ldsi32_closeout_evidence_v1.tar.gz`，3479099 B、SHA256=`d048f763d0b9b8d748c9cd4b79e226dc8e6bd9f11e20d9cab8d1bd5f3e41e318`，所有member SHA匹配，独立receipt=`ldsi32_closeout_evidence_unpack_acceptance_v1.json`。该包在extent诊断前生成，不包含后者的logs/result；打包与传输均在正式性能队列结束后进行。

### 将N32 FP4 WMMA拆为两条N16指令的有限候选

新增默认关闭`AITER_GFX1250_GEMM1_WMMA_N16=1`，仅适用当前E65/K7168/t256/w2x4/b4、单N-task、rowtail3/staged-ScaleA/halfA/halfScale/rowaload2/pairA、cluster4x1。保留原parent geometry、TDM request/ring、A和两侧scales、原pair预读和全部hardware fences；每个N32 weight fragment拆成相同字节的两个N16 fragment，原vec16 accumulator改为两个vec8 accumulator。每个有效输出仍保持原56次K128更新及顺序，epilogue使用原N16视图，SiLU、BF16 RNE、FP4/E8M0、canonical ABI、GEMM2和公共helpers保持。

Shader Guide §4.6.12的C/D布局（p150）明确32x16的前后八个VGPR对应两个16x16矩阵；CDNA5 ISA §15.11（p465–466）明确N16形式的S3/S4为16x4/4x16，`SCALE_OPSEL[0]`选择lane0–15或16–31。candidate按`wn//2`取原N32的ScaleB并以`wn%2`选择N16，activation scale仍按`wm//2`和`wm%2`选择。文档§5.7.5给出MI450 FP4 N16的4/4与N32的8/8 repeat rates，这不证明拆指令后kernel更快；本项用于检验较小operand/accumulator带来的native调度变化，不减少有效计算，也不预设80 μs可达。

`audit_wmma_n16_v1.py`执行actual constants、B-load及mma_row AST，比较458752 operand dwords、229376 scale bytes，32个实际mma_row call参数正确。260种row边界×8个waves共2080 cases、37376 retained N16 output tiles均有56次有序K128更新；关闭candidate完整launcher/wrapper AST等于精确group8，42个nested pipeline/epilogue函数保持。初始CPU脚本将每wave的mma count写成两倍并漏排除派生cache tuple，已修正诊断脚本，未改kernel math。

源码=`wmma_n16_v1_source/`，三源SHA为`db9795b6f2d52d655da1bc1a892413ae9076f84c4e6c33c5466688944ce80135` / `3fc9a8252965a7f4c26556f608fae5bc94f4ba2c352f32e4f8e1fd9e10c6af75` / `1e47358687fea1df605e1e09e94911224e2b23588c3e3a86cebd1920fa8f0764`。先固定关闭→打开actual compile，要求关闭完整ELF仍`7c643f...`、candidate zero spills/private、原compute/output fences及K backedges通过；之后seed0/17、eager/graph canary、同地址dynamic replay的28项严格零字节差异均通过，才启动0→1→0原命令三轮完整性能，ownership污染整组最多三次，双侧一致收益才做一次固定反向。当前尚无本项device/performance结论，独立默认仍107.330 μs。

两项actual compile已完成，rc0、未超过120秒、无GPU dispatch。关闭完整ELF仍`7c643f...`；candidate=`a4262b9e2485cb0db6f8b23f745fc64b6eb4ae3e8c88337ecc40482f7524618ce`，VGPR479/SGPR68、LDS282624 B、zero spills/private、4820 instructions。N16 WMMA静态512 sites替代N32的256，DS b128384、TDM loads20、payload/ScaleA stores2、WG signal/wait10对、tanh128和FP4转换16保持。原八条backedges变为每region56/64条N16 WMMA；各region tensorwait2→DS wait0→WG signal/wait→refill保持，两条exit最终tensorwait0及输出完整DS/WG fence通过。compiler生成的epilogue VALU配对、DS waits/delay及VGPR bank切换也变化，因此不把VGPR减少或指令repeat rate直接当成收益。验收=`wmma_n16_compile_acceptance_v1.json`，inspection=`wmma_n16_native_inspection_v1.json`。

seed0/17严格验证均rc0、未超过120秒，各28项零字节差异，HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic replay通过；原MoE精度仍`3.7793e-06/2.7493e-03`和`3.7788e-06/2.7491e-03`。初始summary额外要求correctness log含symbol而失败，原verify本来不打印symbol；改用原manifest的环境开关、源码SHA绑定验收完成，未修改原精度门槛，也未重跑这两项。correctness-only的所有计时不使用，receipt=`wmma_n16_correctness_acceptance_v1.json`。额外独立验证将实际loaded ELF绑定到上述candidate SHA，通过后才进入正式性能队列。

### N16完整性能队列与撤回

固定0→1→0，每段完整三轮，仅因ownership污染整组最多三次，未追加第四组或反向。label前缀=`a07_wmman16`：

| label后缀 | GEMM1三轮（μs） | 合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `0_control_a_v1_try1` | 109.327,106.820,108.178 | 不统计 | 95.1373,94.0893,94.8863 | 260.196,256.823,258.297 | 第1轮污染 |
| `0_control_a_v1_try2` | 107.756,107.058,106.554 | 不统计 | 95.8814,94.1357,95.7569 | 259.775,257.291,257.879 | 第1轮污染 |
| `0_control_a_v1_try3` | 108.408,108.985,106.495 | 不统计 | 95.0572,95.0928,94.4520 | 258.755,260.265,256.488 | 第2/3轮污染 |
| `1_perf_a_v1_try1` | 117.048,111.380,112.848 | 不统计 | 94.4878,95.2448,95.4134 | 267.510,262.374,264.378 | 第2轮污染 |
| `1_perf_a_v1_try2` | 113.504,110.740,109.647 | 不统计 | 94.3232,94.8547,93.3430 | 263.119,261.993,259.250 | 第3轮污染 |
| `1_perf_a_v1_try3` | 111.520,109.749,110.845 | 不统计 | 94.5764,94.3255,94.6628 | 262.436,259.684,260.711 | 第2/3轮污染 |
| `0_control_b_v1_try1` | 106.900,106.674,108.914 | 106.900 | 94.3613,94.1441,94.4393 | 256.930,256.844,259.372 | 三轮合格 |

21轮全部rc0、未超过120秒。前控制和candidate均无完整合格中值，不能报告稳定收益或用表中单轮拼接性能；后控制106.900 μs也不替代独立默认107.330 μs。summary=`wmma_n16_complete_summary_v1.json`，SHA256=`12c807a30cbae4f68c48388dbb4d55f8874abca4794e5f648d5f50d38fafb83c`。候选没有接入依据，不继续追加按耗时挑选的复测。

`withdraw_wmma_n16_v1.py`在双端按candidate SHA守卫恢复`task_m_group_default_manifest_v1.json`三源精确SHA，保留candidate/raw/native/双seed证据及独立N-group地址fix。receipts=`wmma_n16_withdrawal_{local,host}_v1.json`。性能队列结束和host恢复门禁均整机idle、KFD owners空、VRAM176582656 B，boot保持`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，没有本任务GPU进程残留。恢复期间遇到外部任务时仍自动每6秒等待，未杀外部进程。正式最佳默认仍107.330 μs，距80 μs差27.330 μs。

N16关闭后的185份新增raw日志、ELF/ISA/notes/mlir、实际extent和撤回证据已增量回收。archive=`through_wmma_n16_closeout_evidence_v1.tar.gz`，1634634 B、SHA256=`9a9c24a537bcc25f2175d24cd1e5872bb1ce729a6e4dd548a26b31f19242c741`，全部regular member SHA通过，独立receipt=`wmma_n16_closeout_evidence_unpack_acceptance_v1.json`。打包和传输均在性能结束后完成。

### 仅合并steady段的单descriptor候选

新增默认关闭`AITER_GFX1250_GEMM1_SHARED_TDM_STEADY=1`，仅适用于E65/K7168/t256/w2x4/b4单N-task、cluster4x1、当前rowtail3/staged-ScaleA/halfA/halfScale/rowaload2/pairA及B TH0组合。prologue和drain保持原owner分支，steady段在loop外选择当前wave的一份rank-2 descriptor，再共用原compute loop；24次refill只更新LDS stage与完整i64 global地址，保留每wave每次一条TDM、原READY/DS/WG同步和28个K256。全部有效输出的56次K128更新及顺序、SiLU、BF16 RNE、FP4/E8M0、canonical ABI、GEMM2和公共helpers保持。

旧130us文档中的四live descriptor共用loop产生25 SGPR spills，单descriptor stream则无spills但GEMM1 74.333 μs对71.093 μs回退。本项仅用于有限检验当前单N-task/M32分段路径的native变化，没有从代码缩小预设收益，也不恢复旧多N-task stream。若关闭新开关不能复现默认完整ELF、candidate新增spills/private或native无实质变化，则在GPU验证前撤回。

Shader Guide §4.10.1（p197–198）明确TDM忽略EXEC，同wave TDM有序、跨wave及相对其他memory类别无序；§4.10.2（p199–200）规定正方向OOB load返回零，§4.10.4（p202–204）定义descriptor。candidate逐字段匹配当前FlyDSL `CopyAtom.cpp`的rank-2/numWarps1生产lowering，包括`None` extent=`INT32_MAX`、OOB clamp、padding、mask/early-timeout、element stride与跨4 GiB进位，不采用旧helper默认tile extent。

CPU审计=`audit_shared_tdm_steady_v1.py`，直接执行实际descriptor与issue AST；259种row边界、四种global地址、四种mask、8waves×24次refill共795648 cases通过，比较的有效输入字节量为5300551680 B。关闭candidate的完整launcher/wrapper AST等于精确group8，44个nested compute/load/fence/epilogue函数保持，原`issue_owned`与默认`issue` AST相同。receipt=`shared_tdm_steady_source_audit_v1.json`，三源manifest=`shared_tdm_steady_source_manifest_v1.json`。CPU等价不是GPU数值或性能验收；正式默认仍107.330 μs。

首项关闭candidate编译读取独立快照，`a07_sharedsteady_0_compile_v1`在120秒被runner终止，rc124，没有生成ELF，候选编译未启动；外部GPU owners原样保留。原部署等待仅中断本任务尚未dispatch的host waiter，remote live三源精确保持group8。没有延长超时或运行GPU benchmark。编译失败本身不能归因于kernel时序或数值错误。

v2移除原`issue`外层转发包装：原`issue`恢复完整AST，只在`compute_ktile.do_issue`的候选分支直接调用独立stream emitter。复用同一descriptor与地址算法的795648项CPU检查再次通过；关闭candidate完整launcher/wrapper AST相同，43个未涉及该调用site的nested函数保持，全部math/load/fence/epilogue及GEMM2保持。源码=`shared_tdm_steady_v2_source/`，manifest/audit suffix为`v2`；两项snapshot compile仍各限制120秒，增加40秒Python stack诊断，以区分前端包装与实际codegen阶段。

v2第一次queue脚本含CRLF，shell在启动compiler前退出；修正为LF后保留原失败日志，真实关闭编译仍在120秒被runner终止，rc124，没有ELF。40/80秒stack均落在Python `ast.iter_child_nodes`，尚未进入LLVM codegen。检查本地FlyDSL `visit_BoolOp`发现，14项连续`and`守卫中，每次构造`IfExp`均多次引用已有结果，后续AST遍历重复展开。v3只将这项constexpr守卫改为等价`all((...))`，不改公共FlyDSL、runtime算法或120秒上限；795648项descriptor检查再次通过。

### shared steady v3的编译验收

独立snapshot关闭→打开compile均rc0、未超时，无GPU dispatch。关闭完整ELF仍`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`。候选ELF=`7c3b4ab1478963268cc9b9affa325c0d81eb71ff2265ae5ad9f6680960748d10`，3777 instructions、29384 B，较默认4482 instructions/34096 B减少705条静态指令；VGPR486、SGPR70、LDS282624 B、zero spills/private保持。单份steady loop替代四份owner loop，原28/32 WMMA的两种backedge、tensorwait2→DS wait0→WG signal/wait→refill和最终payload/ScaleA output fencing均通过。静态WMMA/DS/TDM sites减少是删除重复代码，不能解释为跳过有效K更新或直接等同于性能提升。

CPU/native验收=`shared_tdm_steady_source_audit_v3.json`、`shared_tdm_steady_native_inspection_v3.json`及`shared_tdm_steady_compile_acceptance_v3.json`。三源SHA为`9f1142297fa9e13b258983e5e2c115119740b6b91c6396db621671876ee980ef` / `a5a2e5155c881b29d8f8deda14f7b60471cfa1b358bf0fe0ed45a817a405b941` / `07f6874e545530b033ed854b866310ce9ecec534b179cd52eeaf85da2b42d325`。remote live仍精确group8，只上传到候选snapshot目录。外部GPU owner存在时不做正式计时；双seed device验证也使用独立snapshot，并将实际loaded ELF绑定到上述候选SHA。当前仍无新正式性能结论，最佳默认107.330 μs不变。

独立snapshot的seed0/17 device验证均rc0、未超过120秒，各28项零字节差异，包括HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary及三组同地址dynamic input/psum replay。两项实际loaded ELF均精确匹配`7c3b4ab...`；验收=`shared_tdm_steady_snapshot_correctness_acceptance_v3.json`，log SHA分别为`87427a723fc74a79a0e828ce79dec2667184ec6eed9dcc60dc2eb946bb4fe131`及`cadc5f31f2a073b44be206c418b56d84e54d22ab532155ab0d5987928d6f221a`。correctness-only运行的所有计时无效。

固定性能队列=`queue_shared_tdm_steady_deploy_perf_v3.sh`：先整机idle门禁和三源SHA守卫，再独立live loaded-ELF binding，随后原完整命令0→1→0，每段完整三轮。只因ownership污染允许整组最多三次尝试，不按耗时筛选或追加；两侧完整合格中值均有收益才执行一次固定反向。等待期间自动每6秒检查整机GPU/KFD，候选没有默认接入依据前仍关闭。

### shared steady固定队列在控制路径fault后停止

`a07_sharedsteady_0_control_a_v3_try1`第1/2轮rc0，但结束及settled均有外部live owners；第3轮在profiler后的reference stage capture出现`Memory access fault ... Page not present or supervisor privilege`，rc134。该轮`SHARED_TDM_STEADY=0`，完整ELF为原group8默认，没有执行候选steady算法；结束同样出现外部owner。当前不能将fault唯一归因于候选、原kernel或其他任务。三轮均无性能验收，不统计中值，候选性能段和后控制未启动。execution failure按事前规则停止队列，没有把它当作ownership-only重试或延长120秒。

`withdraw_shared_tdm_steady_v3.py`已在双端按candidate SHA守卫恢复精确group8三源，候选默认关闭代码退出live树，全部snapshot/CPU/native/双seed/failed performance logs保留。host恢复门禁于`2026-10-18T08:30:10Z`整机idle，KFD owners为空、VRAM176582656 B、boot未变；本任务PID3664234/3667717/3687612均已退出。receipts=`shared_tdm_steady_withdrawal_{local,host}_v3.json`。本项没有新正式性能结论，最佳默认仍107.330 μs。

### WMMA LD_SCALE unused字段的严格隔离对照

默认ELF中256个`v_wmma_scale_f32_32x16x128_f4`的第一对DWORD内，`LD_SCALE [58:50]`均为`0x80`，例如PC`0x2FF4`的`CC350000 0202291A CC8840F0 1FC2F100`。Shader Guide rev65 §11.3.7 Table116（p322，TXT lines24052–24057）和CDNA5 ISA §14.3.7 Table97（TXT lines15311–15313）均要求此字段为`0x100`。LLVM `VOP3PInstructions.td`的`VOP3PX2e`将该字段设为`?`，scaled-WMMA指定`postEncodeVOP3<true,true,false>`；`AMDGPUMCCodeEmitter.cpp`随后为缺少的source写入inline zero编码`0x80`。

这说明实际encoding不符合上述字段表，但不证明它造成当前stall。Shader Guide §4.6.12（p168）描述设置zero会增加stall；同一文档§1.1及rev61.1对应DEGFXMI400-12891条目进一步明确`0..127`会被当作SGPR读取并增加cycles，当前`0x80=128`不在该范围内。现有资料没有明确给出`0x80`相对`0x100`的silicon性能差异，因此本项采用隔离实验，不先称为已确认性能bug。

新增默认关闭`AITER_GFX1250_GEMM1_LDSCALE_UNUSED_VGPR0=1`。仅对完整SHA为`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`的已验收group8 GEMM1，在engine初始化前将256处unused字段改为`0x100`。其余完整ELF保持原样。候选不改公共LLVM/FlyDSL、persistent math、GEMM2、routing或benchmark，使用原JitFunction的同一host launcher IR和dispatch。

CPU审计=`ldscale_unused_binary_audit_v1.json`，只改变256个字节，严格逐PC匹配原ISA的256个WMMA sites，所有有效operand、WMMA/TDM/fences的完整PC和编码其余位保持。candidate ELF=`e50cc51d52a0248ed5ed3b34fdc9f506ab3db2e89c61c1c78844add60d2afa16`，34096 B。a07-3/dsr1_ep4中的native验收=`ldscale_unused_native_acceptance_v1.json`：4482条decoded instructions及PC完全相同，metadata完全相同，VGPR486/SGPR68/LDS282624 B/zero spills/private保持。关闭开关的wrapper AST等于精确group8，persistent和compile工具字节未改。CPU首轮发现全hex重新编码后的ELF literal检测只接受原混合字面量，已修为同时接受两种形式并显式拒绝空replacement；这是隔离helper的预device问题，修复后CPU和native验收通过。

独立snapshot seed0/17验证各28项零字节差异、rc0、未超过120秒，包括HEAD有效payload/ScaleA/GEMM2/MoE、三次eager/graph canary、三组同地址dynamic input/psum replay；实际loaded ELF均绑定到上述candidate SHA。acceptance=`ldscale_unused_snapshot_correctness_acceptance_v1.json`，log SHA分别为`df6b76ba7108d706537dc6c65b01c050d954c6fa66c2288683212530a466407f8`和`0f1aad652c5e1cfe0b0cafb8414723a31725a4a215f1755a1e749bb7c52200295`。correctness-only期间外部owner保留，所有计时不使用；remote live仍精确group8。

正式对照计划=`queue_ldscale_unused_deploy_perf_v1.sh`：先整机idle与默认SHA守卫部署默认关闭开关，再独立live loaded-ELF binding，随后按`0_control_a → 1_perf_a → 0_control_b`运行原完整命令，每段完整三轮。仅ownership污染整组最多三次；execution failure立即停止，无按耗时筛选、追加或拼接。两侧完整合格控制均支持收益才做一次固定反向，并据此决定是否存在值得实现为可维护源码的收益。当前尚无本项正式性能数据，独立默认107.330 μs不变。

### LD_SCALE关闭控制再次fault，候选性能未启动

`a07_ldscale_0_control_a_v1_try1`三轮均rc0，但ownership均不合格；整组按事前规则重试。`try2`第1轮rc0、ownership不合格，第2轮在profiler后的reference stage capture发生`Memory access fault by GPU node-2 ... on address 0x4160000. Reason: Page not present or supervisor privilege.`，rc134。第3轮、LD_SCALE候选性能段和后控制均未启动。该故障轮开关为0，未执行LD_SCALE字段修改；故障前的profiler数字不作为有效性能数据，也未将execution failure当作ownership-only重试。

两次独立候选队列都在关闭候选的额外capture路径fault，前次shared-steady控制地址为`0x48a000`。结束检查均出现外部live GPU owner，但现有证据不能唯一归因于外部任务、默认kernel或runtime。源码流程明确：原bench/profiler完成后，test安装`grouped.stage_output_capture`并额外调用一次MoE；这个hook关闭fused quant，GEMM1输出BF16且`stage1_quant_out=0`，因此由非persistent的`launch_gemm_a8w4_tdm_optimized`执行，再经standalone quant、GEMM2和hash/reference。故障位置日志不能直接证明被测fused persistent GEMM1出错。

`withdraw_ldscale_unused_v1.py`已在本地按SHA守卫恢复精确group8三源，删除本项新增helper，独立N-group地址修复及全部候选snapshot/精度证据保留；host恢复脚本仍自动每6秒等待整机idle后执行同一恢复。后续先对精确默认snapshot进行独立诊断，逐stage记录runtime tensor地址、shape、stream和实际ELF，并仅在capture阶段添加同步定位故障；原benchmark文件、正式计时路径、routing、GEMM2和公共helpers保持。诊断只用于正确性和故障定位，全部计时无效。正式最佳仍107.330 μs，距80 μs差27.330 μs。

### 精确默认的逐stage与reference诊断

host已于`2026-10-18T09:05:40.995Z`通过整机idle、KFD空、VRAM176582656 B门禁恢复精确group8三源并移除LD_SCALE helper，receipt=`ldscale_unused_withdrawal_host_v1.json`。本轮重新核对remote branch=`moe_a4w4_pr_refactor_wt`、HEAD=`339589f900b3a880ec447f605886fe803cbf9649`、`dsr1_ep4` image=`rocm/fw-bringup:gfx1250-atom-20260930-na`及三源SHA均保持；所有诊断仅在a07-3/dsr1_ep4、通过120秒host runner执行，没有改变原benchmark或production modules。

`diagnose_reference_capture_v1.py`加载精确默认snapshot并逐stage记录tensor地址、shape、stride、stream和actual ELF。`a07_capture_diag_sync_v1`中额外BF16 GEMM1、standalone quant、GEMM2、gather及两个hash均显式同步完成，之后`_torch_moe_ref`出现fault，地址`0x28020000`，rc134。实际ELF分别为fused persistent=`7c643f...`、BF16 GEMM1=`be5584151798dd09e88ecb6dfdde7432eacafb37d5f1b085d9f6a3adc454658c`、GEMM2=`840d2c768ad0b92ee446af222164513d8d190e75ddf5e6a664fc2b4b5a20cca0`。这修正了此前仅依据日志称为“capture fault”的位置判断；同步完成不等于排除潜在数据破坏或runtime状态问题。

同脚本`--trace-reference-ops`的`a07_capture_diag_refops_v1`逐PyTorch op同步后，fault落在reference第2038项`aten.mm.default`，A=`float32 [164,7168] stride [7168,1]`，B=`float32 [7168,4096] stride [1,7168]`，fault地址`0x39d0000`；前55个expert的GEMM1 reference mm和此前2037项op均同步完成。原fault定位没有证明被测fused persistent GEMM1错误，也不能据此确定BLAS、allocator、driver或其他任务的根因。

隔离对照均为correctness-only、rc0、未超过120秒，所有profiler数字弃用：

| host runner label | 前序与诊断 | 结果 |
|---|---|---|
| `a07_native_mm_only_v1` | 不import AITER，相同164行FP32 mm连续3次并检查小CPU投影 | 通过 |
| `a07_native_mm_sequence_v1` | 不dispatch AITER，按真实65个expert counts依次FP32 mm | 全部通过 |
| `a07_reference_only_v1` | 原`_torch_moe_ref`独立运行 | 通过 |
| `a07_reference_shuffle_v1` | 原weight/scale shuffle后reference | 通过 |
| `a07_reference_eager_prelude_v1` | 一次生产MoE后reference | 通过 |
| `a07_reference_profile_prelude_v1` | 原warmup5/iters20 profiler后直接reference | 通过 |
| `a07_reference_capture_nohash_v1` | 原profiler→capture→reference，省略hash | 通过 |
| `a07_reference_capture_hash_v1` | 原profiler→capture→两个原hash→reference | 通过 |
| `a07_reference_release_hash_v2` | 上项并丢弃额外capture返回输出 | 通过 |
| `a07_reference_matchalloc_hash_v2` | 丢弃capture输出、保留profiler trace、匹配bias physical分配、不提前构造routing histogram | 通过 |
| `a07_reference_async_matchalloc_v2` | 上项reference省略逐mm同步 | 通过 |

独立reference脚本只检查完成与finite，没有冒充完整候选精度验收。除最后一项外，reference的mm均添加同步；这些正结果说明shape、hash或capture返回对象生命周期单独不足以复现，但不证明完整流程已稳定或排除这些因素与其他状态的组合。PyTorch=`2.11.0+rocm10.1.0a20260811-bkc.20260820`，HIP=`7.16.26315`。原完整bench命令再以correctness-only运行一次作为preflight，禁止把诊断阶段数字记为性能。

### WMMA operand reuse hint隔离候选

Shader Guide rev65 §4.6（TXT10415–10417）与§4.6 WMMA encoding（TXT11594–11597），CDNA5 ISA §7.12（TXT7141–7143、8170–8173）说明A/B reuse hint指向下一条WMMA的operand，并要求与上一条的instruction type相同；§5.3.2（TXT15414–15442）描述VGPR banking和source cache。文档不足以证明默认hint引发了当前瓶颈，故按隔离对照验收。

原native `v_wmma_scale_f32_32x16x128_f4`的matrix A是`wt[wn]`16VGPR，matrix B是activation8VGPR；当前N32的`mma_n_rep=2`、M16蛇形顺序在偶数row尾和同M32 pair的奇数row头使用同一weight。候选`AITER_GFX1250_GEMM1_WMMA_OPERAND_REUSE=1`只在这类保证两row执行的边界增加A hint，保留原B hint和首row无hint。v1曾拟在首row也增加B hint，CPU检查通过后因上述instruction-type限制收紧为v2，v1没有编译、dispatch或计时。

v2仅限制当前E65/K7168/t256/w2x4/b4、单N-task、rowtail3/stagedScale/halfA/halfScale/rowaload2/pairA、cluster4x1组合，开关默认0，不改GEMM2或公共helpers。CPU审计`audit_wmma_operand_reuse_v2.py`直接执行实际`mma_row/mma_rows/mma_rows_full`，覆盖520项row/wave边界、4768次operand调用、1192项A和1990项B next-operand一致性；hint不跨M32 predicate或K128结束，全部有效output保留56次有序更新。关闭开关完整launcher/wrapper AST等于精确默认，44个未改nested函数及所有有效math、TDM、DS、hardware fences和epilogue保持。

源码=`wmma_operand_reuse_v2_source/`，manifest=`wmma_operand_reuse_source_manifest_v2.json`，三源SHA=`2d1488bed9029534b280b3582fb2c7f8bf52f4a9a00d02cc22fc4e92b5216458` / `4f2943ae0c93e70dac506c26f78a8219b8781c46c466ccbe9ba1e299364664ea` / `31b49c050354c4c9429bede6acefe568bbb815cd2658d171f2f6d99e0605ef3a`。实际compile固定关闭→打开，每项120秒、无dispatch、remote live保持精确默认。关闭完整ELF须等于`7c643f...`，然后检查candidate无spill/private、有效operand与累加顺序、native hint边界及原output fencing，才进入双seed严格精度验收。当前没有本项正式性能结论，最佳仍107.330 μs。

两项actual compile均rc0、未超过120秒、无dispatch。关闭完整ELF仍为精确默认`7c643f...`；candidate=`448584a7fff4bbf06b05c5da2400497ba14e423d26a7736592f123bdeed9b61a`，34128 B、4482 instructions、VGPR486/SGPR68、LDS282624 B、zero spills/private。native全部PC、顺序、有效operand、scale selector、memory及hardware fences保持，只改变64个matrix A hint bits，原112个matrix B hints完整保持；额外32 B来自symbol suffix。验收=`wmma_operand_reuse_compile_acceptance_v2.json`。

独立snapshot的seed0/17均rc0、未超过120秒，各28项零字节差异，原HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary及三组同地址dynamic input/psum replay通过。实际loaded ELF均匹配上述candidate，live binding也通过。验收=`wmma_operand_reuse_snapshot_correctness_acceptance_v2.json`；双seed log SHA分别为`3326b095f50e9647e37ec5fc87cf317ccf8845b74d48f9b9710289878563b4b2`、`d42fdc58a8d44f8bc034ccdc4fa4d730848de86f2d075deda2ad729e335b8256`。上述验证均为correctness-only，不记录为性能。

### operand reuse的关闭控制fault及撤回

固定`0_control_a → 1_perf_a → 0_control_b`队列在前控制第三组第3轮发生execution failure后停止，没有启动candidate性能或反向。前两组三轮均rc0但ownership全部不合格，仅因此整组重试；第三组第1轮合格，第2轮污染，第3轮profiler后出现`Memory access fault by GPU node-2 ... on address 0x5160000. Reason: Page not present or supervisor privilege.`，rc134、未超时。开关始终0，symbol无`_operand_reuse1`。这没有证明candidate hint错误；本轮尚未逐op定位新fault，不能直接将之前定位到reference的结果当作本次fault PC证据。

| label（前缀`a07_operandreuse_`） | GEMM1三轮（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | 验收 |
|---|---|---|---|---|
| `0_control_a_v2_try1` | 106.489,108.071,108.108 | 94.3614,94.8716,93.5898 | 256.324,257.342,257.389 | 全部ownership污染，不统计中值 |
| `0_control_a_v2_try2` | 106.704,107.404,112.475 | 94.7451,95.2742,94.3235 | 257.633,258.032,262.862 | 全部ownership污染，不统计中值 |
| `0_control_a_v2_try3` | 108.222,111.016,未完成 | 93.8428,94.5277,未完成 | 255.538,262.328,未完成 | 第1轮合格；第2轮污染；第3轮rc134 |

完整失败摘要=`wmma_operand_reuse_failed_queue_summary_v2.json`，SHA256=`0d0e8d899a8d178fcd9a659c03374dbd8528037499d4fbf344c6420a2f4515b1`。没有拼接单轮、统计污染中值、追加第四次或忽略故障后的精度验证。`withdraw_wmma_operand_failure_v2.py`已在双端按candidate SHA守卫恢复精确`task_m_group_default_manifest_v1.json`三源，candidate及全部compile/device/raw证据保留。receipts=`wmma_operand_reuse_withdrawal_{local,host}_v2.json`。host恢复门禁于`2026-10-18T09:56:42.386696Z`整机idle、KFD owners为空、VRAM176582656 B、boot保持；九轮的`live_post_owners`均为空，没有本任务GPU进程残留。

当前正式最佳仍107.330 μs，距80 μs差27.330 μs。后续继续检查完整验证流程的reference mm/backend，并只在独立snapshot编译下一项GEMM1候选；不能把此前独立诊断通过描述为已修复完整流程的偶发fault。

### BACK A registers的部分共享：v1资源拒绝与v2收紧

本轮在原w2x4/N256、K256/b4、single-N-task、rowtail3/stagedScale/halfA/halfScale/rowaload2/pairA、cluster4x1路径尝试缩短A registers的双缓冲生命周期。`AITER_GFX1250_GEMM1_A_PAIR_REUSE`默认0；v1的mode1只共享BACK四个M16 rows，保持FRONT、B、ScaleA和ScaleB双套。每个当前M32 pair的全部WMMA消费者结束后，才覆盖其下一K128的A operands。既有w4x4 rollingA和compactN full-A reuse已测，本项保持原geometry和有效计算。

CPU审计直接执行实际load/WMMA/k_step/compute_ktile AST：260种row边界、1040项pipeline cases、58240次K128 steps、247296次有效M16更新、523264次实际WMMA。有效A/B/scales、hint、56次原有序K128更新、load多重集、TDM/hardware fence事件均与默认相同；关闭candidate的完整launcher/wrapper AST与精确group8一致，45个nested函数保持。CPU不能替代native hazard或device验证。v1源码=`a_pair_reuse_v1_source/`，三源manifest=`a_pair_reuse_source_manifest_v1.json`，SHA为`55f45d7f3523f78f7edd4b1bd1f612786265becf2c77309060e7bae10939c07f` / `af94622217f09f37902c8071a0c686c9a3effcd1344e5031564f20a1726114fd` / `d12d74076f5bce862b7f3860531d8af44822f3c154bb9ef0819469c97906729a`。

固定关闭→打开两项actual compile均rc0、未超过120秒、无dispatch，remote live保持精确默认。关闭完整ELF仍`7c643f...`；v1 candidate=`c364ec85ba1ac02254e9bf7c059c080334e66e84db6a7e80240109c6579cc387`，34376 B、4542 instructions、VGPR453/SGPR68、LDS282624 B，但出现1 VGPR spill/8 B private segment，资源验收拒绝，未运行GPU精度或性能。native在PC `0x4B60`保存一个ScaleA值到scratch，在`0x4EE0`回读、`0x4EF0`等待LOADcnt、`0x4EF4`作为WMMA scale operand消费；这个位置不能唯一确定完整register-allocation原因或量化成本。证据=`a_pair_reuse_resource_rejection_v1.json`，v1源码和raw完整保留。

v2的mode2只共享最后一个M32 pair（M16 rows6/7），前三pairs的A保持双套，BACK ScaleA的原预读位置保持。相同CPU覆盖再次通过；源码=`a_pair_reuse_v2_source/`，三源SHA为`55f45d7f3523f78f7edd4b1bd1f612786265becf2c77309060e7bae10939c07f` / `11302c8f0c20409981c3c42f4a6b1b97084a883eee568fec932005a28fbebe49` / `46dff64f48dfedf4f392d2f59dc70b7fb092b4612c93207d4950319c816039fd`。固定0→2两项actual compile均rc0、未超过120秒、无dispatch：candidate=`5769a906055ccbf65065b47cf579ae6313ce3f63b7b800b05506386e924d9d1c`，34248 B、4513 instructions、VGPR470/SGPR68、LDS282624 B、zero spills/private。较少VGPR没有证明occupancy或性能提升。

`accept_a_pair_reuse_compile_v2.py`验收八条K backedges/四targets、每region28/32 WMMA sites、READY tensorwait2→DS reader release→WG signal/wait→refill TDM顺序、payload/ScaleA output fencing、两条exit最终tensorwait0。原256个WMMA、384个DS b128 loads、20/2个TDM load/store、10/10个WG signal/wait、96个scale dwords、128个tanh及16个FP4转换sites保持；register assignment、DS/WMMA预读与编译器生成的wait/delay安排有变化，不能仅靠static opcode count证明跨wave安全。验收=`a_pair_reuse_compile_acceptance_v2.json`和`a_pair_reuse_native_inspection_v2.json`。随后在a07-3/dsr1_ep4运行独立snapshot seed0/17的HEAD逐字节、eager/graph canary、同地址dynamic input/psum严格验证，每项120秒。当前尚无本项device或正式性能结论，107.330 μs不变。

### reference BLAS日志诊断补充

`a07_reference_blas_trace_v1`使用精确默认snapshot，在correctness-only下设置`ROCBLAS_LAYER=1 HIPBLASLT_LOG_MASK=1`，运行`diagnose_reference_capture_v1.py --sync-stages --trace-reference-ops --seed 0`。rc0、未超过120秒，完整原MoE/capture/hash/reference通过，`logits_diff=3.7792784448342734e-06`；此前定位到的第2038项`aten.mm.default`本次也同步完成。没有找到`rocblas_sgemm`或`hipblasLtMatmul`日志，因此无法据此确定实际BLAS backend，也不能声称偶发fault已经修复。本项所有计时无效。

raw=`reference_blas_trace_v1/{events.jsonl,acceptance.json,...}`及`a07_reference_blas_trace_v1/{run_1.log,result_1.json,...}`已回收本地；actual fused/BF16 GEMM1/GEMM2 ELFs仍为`7c643f...`/`be558415...`/`840d2c76...`。结束出现live KFD owner `397299`，随后该PID的`/proc`条目已不存在；没有杀进程。既有完整对照的execution failure仍保留并使该队列停止，未用本次正结果覆盖故障样本。

### A-pair v2双seed通过，额外live验证fault后停止

独立snapshot seed0/17两项均rc0、未超过120秒，各28项零字节差异，包括原HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay；actual loaded ELF均精确匹配`5769a906...`。验收=`a_pair_reuse_snapshot_correctness_acceptance_v2.json`，双seed log SHA分别为`7a9393c70849739b389e0cda103c07093c370775cc948e48ecfc7ee2f766acca7` / `727622a83823e72e5d907cc8bf11e37aff7ae3fdcd6c1b3bff287ebe13fff5a34`。所有correctness-only计时无效。

`run_a_pair_reuse_perf_v2.py`固定计划为原完整命令0→2→0，每段完整三轮、仅ownership污染整组最多三次、execution failure停队列，两侧均支持收益才固定反向。host先通过idle/SHA守卫部署默认关闭开关；随后`a07_apairreuse_live_binding_v2`的额外无replays精度验证在已打印actual candidate ELF后fault，地址`0x69e00000`，rc134、未超时。本次没有逐stage故障位置证据，不能把此前reference mm定位当作本次结论。前控制、candidate性能、后控制和反向全部未启动，没有本项性能数据。完整summary=`a_pair_reuse_complete_summary_v2.json`，SHA=`086032bccb785804483a0c44d6dc41f332694acc2f125248696036f2689234df`，原snapshot正结果和本次失败日志均保留。

`withdraw_a_pair_reuse_v2.py`已双端按candidate SHA守卫恢复精确group8三源；receipts=`a_pair_reuse_withdrawal_{local,host}_v2.json`。结束`live_post_owners=[]`，summary post整机idle、KFD空、VRAM176582656 B、boot保持，没有本任务GPU进程残留。下一步仅在独立snapshot、correctness-only下增加stage/reference同步定位；不忽略failure、不延长120秒、不修改原benchmark、routing、GEMM2或公共helpers。正式最佳仍107.330 μs，距80 μs差27.330 μs。

### A-pair故障的同步诊断通过，根因尚未确定

`a07_apairreuse_fault_diag_v2`在独立candidate snapshot、seed0下运行`diagnose_a_pair_reuse_fault_v2.py`，对capture的GEMM、quant、gather和hash前后加同步，并对原`_torch_moe_ref`逐PyTorch op同步、记录tensor地址/shape/stride。该项rc0、未超过120秒，4710项reference op均完成，原HEAD有效payload/ScaleA/GEMM2/MoE四项零字节差异通过；没有开启额外canary/replays，也不替代之前两个seed的28项验收。actual ELF仍为`5769a906055ccbf65065b47cf579ae6313ce3f63b7b800b05506386e924d9d1c`，events SHA256=`252f38c8fcfd778512e6700449923ad56dd3a7e1bce8925ae0ae6fa31aa88147`，`live_post_owners=[]`。

增加同步改变了诊断执行顺序；本次没有复现原异步fault，因此不能称为production bug fix，也没有解锁或重启失败的A-pair v2性能队列。host dmesg中该次失败对应PID573810，在10:28:03–04记录多个XCD的TCP/GC_UTCL2 read protection faults，status=`0x00C04061`、`PERMISSION_FAULTS=0x3`；这些日志不足以将根因唯一定位到GEMM1、reference BLAS、allocator或driver。

新增490份任务raw/ISA/ELF/诊断证据已增量回收，archive=`through_apairdiag_evidence_v3.tar.gz`，4960210 B，SHA256=`15f69a1dc99574c0e24121933d618cd1933e9c78142f0ab844f97d8a99cc7b10`；逐member SHA全部通过，receipt=`through_apairdiag_evidence_unpack_acceptance_v3.json`。当前双端live三源仍为精确group8默认。

### 下一项有限实验：只共享FRONT A pairs

基于精确group8，在独立`front_a_reuse_v1_source/`中新增默认关闭的`AITER_GFX1250_GEMM1_FRONT_A_REUSE=1/2`，分别共享第一个/前两个M32 pairs的A register slots。BACK A、B、ScaleA、ScaleB仍独立双套；下一K128的head不覆盖共享FRONT A，在当前同pair全部WMMA消费者结束后才读取下一K128的同pair，非共享pairs保留原预读位置。它与上一项只共享BACK的实现和此前w4x4/compactN geometry不同。

本项只适用当前E65/K7168/t256/w2x4/b4、singleN、cluster4x1及原rowtail3/stagedScale/halfA/halfScale/rowaload2/pairA组合。CPU直接执行实际pipeline AST，通过260个row边界、1560项pipeline cases、87360次K128 steps、370944次有效M16更新和784896次WMMA；当前A/B/scales、有效输出56次原顺序更新、load多重集及TDM/hardware fence事件保持。关闭candidate完整launcher/wrapper AST等于精确默认，43个未涉及的nested函数保持；GEMM2和公共helpers未改。CPU等价不能代替native hazard或GPU精度验收。

三源SHA为`123b8e92275531f2b02691b354c0c2fa7e02fbf4b5e72fe226c36067b571522b` / `5b5bb5968527da8e52a5a31dbc439f91fc6b753f477e4cff2129f611ddf869b5` / `d05684a7b6215102c2b18efbb6df072552967798b207fe798d810e8cb85a7a42`；manifest/audit=`front_a_reuse_source_{manifest,audit}_v1.json`。接下来固定0→1→2 actual compile，每项120秒、无dispatch；要求关闭完整ELF保持默认、candidate zero spills/private及原native input/output fencing通过，才进入双seed严格精度验证。当前没有本项device或性能结论，最佳仍107.330 μs。

### FRONT A复用的编译验收与mode2淘汰

固定0→1→2三项actual compile均rc0、未超过120秒、无dispatch。关闭完整ELF精确等于group8默认，双端live三源仍保持默认SHA；mode1进入独立snapshot的seed0/17严格验证，mode2不运行GPU精度或性能。

| FRONT A mode | ELF SHA256 | VGPR / SGPR | instructions | VGPR spills / private B | 结论 |
|---:|---|---|---:|---|---|
| 0 | `7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692` | 486 / 68 | 4482 | 0 / 0 | 完整默认ELF一致 |
| 1 | `ba7e91ec3fe12af672337a15b7272b469d64e60b783e46dbb598054298dda5d2` | 470 / 68 | 4512 | 0 / 0 | 仅共享rows0/1，继续device验收 |
| 2 | `8fc84626c136367cd4105a3e2e952ea5a36717ef62fd53e104fc2a22789c0226` | 454 / 68 | 4578 | 6 / 16 | 资源验收拒绝 |

三者LDS均282624 B。`accept_front_a_reuse_compile_v1.py`对mode0/1验收八条K backedges/四targets、每region28/32 WMMA sites、READY tensorwait2→DS release→WG signal/wait→refill、payload/ScaleA output fencing和两条exit最终tensorwait0。原256 WMMA、384 DS b128 loads、20/2 TDM load/store、10/10 WG signal/wait、96 scale dwords、128 tanh及16 FP4转换sites保持。

mode1相对默认的opcode counts只增加30条`s_set_vgpr_msb`（375→405），同时register assignment和DS调度发生变化。较少VGPR不能证明occupancy、cross-wave安全或性能收益。CPU审计fixture已核对本路径实际`STATE_DS=30/MMA_GROUP=2`，没有继承上一项BACK审计的38/4。证据=`front_a_reuse_compile_acceptance_v1.json`、`front_a_reuse_native_inspection_v1.json`和`front_a_reuse_resource_rejection_v1.json`；此时无本项正式性能结论。

### FRONT A mode1双seed通过，准备完整性能对照

`queue_front_a_reuse_snapshot_correctness_v1.sh`在a07-3/dsr1_ep4只验证mode1：seed0/17均rc0、未超过120秒，各28项零字节差异。原HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary和三组同地址dynamic input/psum replay通过；实际加载ELF均为`ba7e91ec3fe12af672337a15b7272b469d64e60b783e46dbb598054298dda5d2`。验收=`front_a_reuse_snapshot_correctness_acceptance_v1.json`；seed0/17原始日志SHA分别`839552136b3b0a27a4733583da45bf314fa0acf2d31f41689d3d3fc5cb100fe6` / `c892478a2144e8f54182cc6f74387bcc03a6e51ee2a6749ac2864c91d7f344b9`。correctness-only期间存在其他GPU使用者，其耗时均无效；live三源仍为精确默认。

新队列=`run_front_a_reuse_perf_v1.py`，不重启上一项BACK失败队列。固定原完整命令`0_control_a → 1_perf_a → 0_control_b`，每段完整三轮，只允许ownership污染整组最多三次，不按耗时追加或拼接。只有两侧完整合格控制都支持收益，才运行一次固定`1_reverse_a → 0_reverse_control → 1_reverse_b`。每个容器测试120秒上限，execution failure立即停止；结束或失败后自动按SHA守卫恢复默认，candidate及全部raw保留。正式计时前后使用整机GPU/KFD门禁，遇占用每6秒自动等待。当前没有本项正式性能数据，独立默认仍107.330 μs。

### FRONT A性能队列：默认关闭控制fault，候选未计时

额外live binding验证rc0、未超过120秒，四项HEAD字节对照通过，实际candidate ELF仍为`ba7e91ec...`。随后原完整命令的`a07_frontareuse_0_control_a_v1_try1`运行第1轮rc0，打印106.759 μs，但ownership不合格；第2轮在打印原profiler表之后出现`Memory access fault`，地址`0x26820000`，rc134、未超时。此轮`AITER_GFX1250_GEMM1_FRONT_A_REUSE=0`，使用原默认kernel；没有本次逐stage或fault PC证据，不能直接将此前reference mm定位当作本次结论，也不能据此证明生产kernel无问题。

按事前规则停止队列，第3轮、candidate性能、后控制和反向全部未启动；没有完整三轮中值，106.759 μs不作为新最佳。完整summary=`front_a_reuse_complete_summary_v1.json`，SHA256=`3839060e0c01fa1ec27765309ff5796b7e2783a59b10d7ecc470ebf095a6b1c8`。第1/2轮log SHA分别`0f745e7b0a4c27e41a0082d339aa8c6374522a1a085f9d47e5c8d4873dd2e7d1` / `696645aac0afc4bcfbe0a2955a656397f2a4b608a03b5f0003ba967cd6d8cfe3`。

`withdraw_front_a_reuse_v1.py --host`已自动按candidate/default SHA守卫恢复精确group8三源，receipt=`front_a_reuse_withdrawal_host_v1.json`；本地live从未部署本项候选，仍为同一默认SHA。两轮`live_post_owners=[]`，队列post整机idle、KFD空、VRAM176582656 B、boot保持，没有本任务GPU进程残留。保留FRONT源码和已通过的双seed证据，不解锁或重启失败性能队列。下一步检查原完整流程的native dispatch和reference backend证据；当前正式最佳仍107.330 μs，80 μs目标未达到。

### reference BLAS实际绑定与原完整命令的native trace

`reference_blas_interpose_v1.cpp`按已安装的hipBLASLt/rocBLAS ABI，通过`LD_PRELOAD`和`dlsym(RTLD_NEXT)`调用原函数，只记录native library、API、tensor/workspace指针、host时间和返回status；没有增加GPU同步、修改数据或生产kernel。a07-3/dsr1_ep4独立构建rc0、无dispatch、未超过120秒。C++源码SHA为`4cf558fd22931436010c9afbba2c58963dd9187ef789d03961e2d2d62f2229bb6`，产物`libreference_blas_interpose_v1.so`为`04976179e7e365f9b771682da81c63ea341e22b994007d42c173f889ea194ee5`。PyTorch默认`preferred_blas_library()`报告`_BlasBackend.Cublaslt`，allocator为`native`；这两项本身不作为实际dispatch绑定证据。

`a07_reference_blas_interpose_v1`在精确默认snapshot下运行既有逐stage/逐reference-op同步诊断，rc0、未超过120秒，完整原MoE/capture/hash/reference精度通过。logger实际解析到`/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib/libhipblaslt.so.1`的`hipblasLtMatmul`，共130次调用和130次返回，全部status0，没有`rocblas_sgemm`调用。76 MiB（79691776 B）workspace在该进程内始终使用同一指针。同步reference的130项`aten.mm.default`与native A/B指针逐项匹配；第2038项、native call55的`float32 [164,7168] stride[7168,1] × [7168,4096] stride[1,7168]`本次完成。此项增加了同步，不能据此声称完整异步流程的偶发fault已修复，所有耗时无效。

首次`queue_reference_native_trace_v1.sh`因host runner的`argparse.REMAINDER`对rocprof子命令中`--`的解析失败，未进入容器测试或GPU dispatch；原失败日志保留。修订v2由独立Python wrapper内启动`rocprofv3 --kernel-trace --hip-runtime-trace --memory-allocation-trace --output-format csv json --output-config true`及用户原完整benchmark命令，同时启用上述logger。原benchmark打印完整精度summary，`pass=True`、`logits_diff=3.7793e-06`，trace文件写出后rocprof进程CPU `SIGSEGV`退出，host runner rc245、未超时。没有复现GPU `Memory access fault`，也没有修复结论。注入rocprof时原PyTorch profiler出现0.828/8.29 μs等异常数字，全部弃用，禁止作为GEMM1/GEMM2或MoE性能数据。

`analyze_reference_native_runtime_v2.py`显式读取`Kernel_Name`，用HIP `Correlation_Id`与native调用的host区间进行一一关联。6837条kernel dispatch中130条为同一`Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT32x32x16_MI16x16x1_...` symbol；它们分别由130项`hipExtModuleLaunchKernel`发出，并与130项`hipblasLtMatmul`形成严格双射，全部API返回status0。symbol全名SHA为`bae1cb1faaf2d69fd4b7e48159acd54c70a4491c5e61ddcf84ef5ab0e037e285`。API成功返回属于异步状态，不能唯一确定之前fault的BLAS、allocator、driver或memory corruption根因。shape与指针审计来自独立的同步诊断，未冒充失败运行的fault PC或地址证据。验收=`reference_native_runtime_binding_acceptance_v2.json`；原CSV与JSONL完整保留。

增量回收`through_frontareuse_evidence_v1.tar.gz`为2832628 B、124份regular members，SHA256=`6c3e4e16d9aee2fe9b71d2949302c9e3adbfae9eff7d2742a3e5a8908fb98b0a`，所有member SHA通过，独立receipt=`through_frontareuse_evidence_unpack_acceptance_v1.json`。该包按原规则不包含CSV/C++/.so；native kernel/HIP CSV另行下载并纳入上述验收的input SHA，C++本地源码和远端.so保留。结束出现外部live KFD owner939771，其命令为其他E64优化验证，不属于本任务残留，未杀进程。重新核对双端live三源、remote branch/HEAD、容器image和boot均保持精确group8默认；正式最佳仍107.330 μs。

### 原t256/w2x4下只共享第二组B registers的有限候选

基于精确group8的独立`partial_b_reuse_v1_source/`，新增默认关闭`AITER_GFX1250_GEMM1_PARTIAL_B_REUSE=1`。原parent geometry、cluster4x1、K256/b4、双套A/B0/ScaleA/ScaleB及全部input/output fences保持；B1 registers共享，下一K128的head预读省略B1，在当前K128所有M32 pairs的WMMA消费者完成后才装入下一B1。末次没有下一K128时不覆写。它与已淘汰compact N128/w4x2/b3且共享A的B实验不同，不复用旧性能结果，不在live默认部署。

CPU执行实际`load_lds_data/k_step/compute_ktile`与原`load_a/load_b/mma_row`AST，覆盖260个row边界、两个M128 halves，共1040个pipeline cases、58240次K128 steps和523264次WMMA。每项当前A/B/scales正确、有效accumulator仍56次原有序更新、hint及TDM/hardware fence事件相同，payload/scale/weight load多重集保持；默认关闭完整launcher/wrapper AST等于精确group8，其他42个nested函数及GEMM2/helpers保持。审计=`partial_b_reuse_source_audit_v1.json`；CPU等价不作为native hazard或GPU精度通过。

wrapper/persistent/compiler SHA分别为`3f8a4ce645ecb56f77d0921da8f688e6c62e47f69490fe9f430dd8b80c7ff753` / `7089e91e8729f26e6c8f7fa4d9fa0e4198deb608d64721e8a88e4b1c9777d364` / `ee3dbc5eba72cd806a61d5a48b50a1f3f96d172d69cf51ad32f3e0de79d67c88`。a07-3/dsr1_ep4固定关闭→打开actual compile已经启动，每项120秒、无dispatch、live三源保持默认。只有关闭完整ELF相同、candidate zero spills/private及原native fences合格后才运行双seed严格精度，然后新建原完整命令固定性能队列；之前失败队列保持停止。此时正式最佳仍107.330 μs，80 μs目标未达。

两项actual compile均rc0、未超过120秒、无dispatch。关闭完整ELF仍`7c643f...`；candidate=`66e16de854ab3d26d3a0dcf9d0c6e3126ea321439d25479148c48c37eb59fe1c`，34384 B、4528 instructions、VGPR456/SGPR68、LDS282624 B、zero spills/private。原256 WMMA、384 DS b128 loads、20/2 input/output TDM、10对WG signal/wait、96 scale dwords、128 tanh和16 FP4转换sites保持。八条K backedges/四targets、每region28/32 WMMA、READY tensorwait2→DS reader release→WG signal/wait→refill、output DS/WG fencing与两条exit最终tensorwait0验收通过。opcode增量为`s_wait_dscnt +31`、`s_set_vgpr_msb +22`、`s_delay_alu -4`、`v_nop -3`；较少VGPR同时增加了部分compiler等待，不能据此预测收益。验收=`partial_b_reuse_compile_acceptance_v1.json`，native inspection=`partial_b_reuse_native_inspection_v1.json`。

独立snapshot seed0/17均rc0、未超过120秒，各28项零字节差异，原HEAD有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary及三组同地址dynamic input/psum replay通过；actual ELF均严格匹配`66e16de...`。验收=`partial_b_reuse_snapshot_correctness_acceptance_v1.json`，双seed log SHA为`5d3a258e5095aaf33619d6986e15d4cefd27e6bd96602aba897d811412f46488` / `a93630ec73d412d67237dc05f642873b22ca4af3931c2604a0de38a0b3bcd106`。两项`live_post_owners=[]`，所有correctness-only耗时弃用。新独立固定`0_control_a→1_perf_a→0_control_b`原命令性能队列已经启动，每段三轮、只因ownership污染整组最多三次；execution failure停队列并SHA守卫自动恢复默认，两侧均改善才固定一次反向。目前没有本项正式性能结论。

### B1性能队列在默认控制第二轮fault后停止

`a07_partialbreuse_0_control_a_v1_try1`第1轮rc0、ownership合格，GEMM1/GEMM2/MoE e2e分别为106.027/93.7604/255.014 μs；第2轮在原profiler表之后发生`Memory access fault by GPU node-2 ... on address 0x2710000. Reason: Page not present or supervisor privilege.`，rc134、未超时。该轮`AITER_GFX1250_GEMM1_PARTIAL_B_REUSE=0`，symbol仍为精确group8默认，无`_partialbreuse1`。本次没有逐op或fault PC定位，不能把之前reference第2038项的同步诊断结论直接应用到本次，也不能据此判断B1候选正确或错误。

按事前规则停止queue；第3轮、candidate性能、后控制和反向均未启动。没有完整三轮中值，106.027 μs只是单轮记录，不替代正式107.330 μs。summary=`partial_b_reuse_complete_summary_v1.json`，最终文件SHA256=`91fad2f77e6c397d8a6c69c2fc53646c5b4d482febfe74416b0b1e8062f11ca8`；两个run log SHA为`f7171d061ec0635b5c13b67a376f0056251121a507ebc2916a75b9d3e907bebb` / `6030e7b2c4c63ce45cde2ea1c0cc27bc4ba4a75f173a1fc2314f9fda797a41eb`。恢复receipt在summary写入`default_restored`之前保存的summary SHA与最终SHA不同，保留原文件，不改写旧receipt掩盖这个先后差别。

`withdraw_partial_b_reuse_v1.py --host`自动等待外部E64测试PID1268694结束，`2026-10-18T11:53:52.908691Z`整机idle、KFD空、VRAM176582656 B、boot保持后，按三源SHA守卫恢复精确`task_m_group_default_manifest_v1.json`。receipt=`partial_b_reuse_withdrawal_host_v1.json`，等待原始记录=`partial_b_reuse_auto_restore_control_v1.log`。本地live没有部署候选；checkpoint重新核对双端三源、remote branch/HEAD与容器image均一致。两轮`live_post_owners=[]`，没有本任务GPU进程残留。当前a07-3另有其他测试/trace任务，未结束外部进程；正式性能继续等待整机空闲，不重启本项失败队列。候选snapshot、compile、双seed和失败raw完整保留，最佳107.330 μs不变，80 μs目标尚未达到。

### 原完整命令的无新增同步native BLAS日志诊断

`queue_original_blas_calls_v1.py`在整机idle门禁后，通过120秒host runner运行`diagnose_original_blas_calls_v1.py`。该项以`runpy`执行未修改的原benchmark及全部原参数，仅`LD_PRELOAD`前述native logger；没有rocprof或新增GPU同步。开始时间为`2026-10-18T12:10:32Z`，rc0、未超时，完整原精度通过，`logits_diff=3.77928e-06`。本次没有复现GPU fault，不能报告fault已修复，也不解锁B1、FRONT A或其他已停止的性能队列。

130项`hipblasLtMatmul`调用与130项返回的ID均严格为0..129，全部status0；workspace为79691776 B，进程内同一指针，stream=`(nil)`。events=`original_blas_calls_diag_events_v1.jsonl`，summary=`original_blas_calls_diag_summary_v1.json`，完整host run=`a07_original_blas_calls_diag_v1/`。两份脚本SHA分别为`e0a27dfd200b1b30a24f3c59c43037bcd46eb5ef1322fec5ba4bbec0783a8769` / `ce29213767a3d9f4a69e5fa1d108cf17e51b4822df32eda780290340cf602a2a`。日志、events、summary已回收，本地独立CPU验收另存`original_blas_calls_diag_acceptance_v1.json`，绑定各原文件SHA。native API成功返回属于异步状态，不是失败运行的fault PC、地址或根因证据。

该项为correctness-only，所有profiler数字弃用，不作为新GEMM1/GEMM2或MoE性能。结束`live_post_owners=[]`，整机idle、KFD空、VRAM176582656 B、boot保持。双端live仍精确group8默认，正式最佳107.330 μs，距80 μs差27.330 μs。

### M32条件转换的精确ELF隔离实验

重新核查CDNA5 ISA §15.1（p214）的`S_AND_B32`、§15.1（p220）的`S_CSELECT_B32`及§15.5（p288）的`S_CBRANCH_SCC0/1`：AND非零时SCC=1，CSELECT将其转换为0/1，后续`S_CMP_LG_U32 temp,1`再反转SCC。默认native在M32 compute body前重复此链；此前源码SGPR/SCC缓存与pair scope未消除转换，因此这次仅做exact-ELF有限隔离，不重跑这些已淘汰source候选。

`analyze_m32_scc_chains_v1.py`对默认`7c643f...`的4582条instruction records（含100条padding）逐条解析signed relative branch，筛出54处包含原四条WMMA、没有TDM的M32 body。两条successor路径上，临时SGPR与SCC均在再次读取前被覆写。`m32_scc_direct_v1_source/`新增默认关闭`AITER_GFX1250_GEMM1_M32_SCC_DIRECT=1`，仅对这份完整SHA匹配的GEMM1删除54对CSELECT/COMPARE，并将54项分支改为SCC0；全部relative branches映射到原surviving instruction边界。未知specialization保持原binary，不改GEMM2、routing、原benchmark或公共helpers。

CPU审计=`m32_scc_direct_binary_audit_v1.json`，包含131108项predicate cases、54个两侧liveness proofs、完整retained instruction bytes及branch mapping。所有WMMA、DS/TDM、`s_set_vgpr_msb`、`s_delay_alu`、barrier、量化与数值指令编码保持；ELF `.text`以外逐字节相同，末尾增加108条`S_CODE_END`保留section size。根据ISA §15.5（p283–284）的`S_DELAY_ALU`定义，本项保留其位置和编码，没有把缩短条件链当作可以无审计删除硬件delay的依据。

candidate ELF=`6252e64566f74af5b5e9ac1fddec8000ec23911cbc7300a119d1f2468bf08003`，34096 B，native执行指令4482→4374，VGPR486/SGPR68/LDS282624 B、zero spills/private。a07-3/dsr1_ep4实际`llvm-objdump/readelf`验收全部保留指令、54处反向分支、其余relative branch目标、metadata相同、no ELF relocations及无`.long`。前两项无dispatch checker分别因空白字符和objdump把负relative immediate打印为unsigned i16而rc1，失败raw保留，修正checker后容器项`a07_m32_scc_direct_native_v3`rc0、未超时。外层queue误将结束时外部KFD PID1620881当作禁止项而assert；独立closeout保留此失败与原post，native检查本身无GPU dispatch，不将外部ownership转成性能合格。acceptance=`m32_scc_direct_native_acceptance_v1.json`，closeout=`m32_scc_direct_native_closeout_v1.json`。

独立snapshot双seed严格device验证已启动，各使用原HEAD参考、3次eager/graph canary及3组同地址dynamic input/psum replay，120秒限制保持。CPU/native等价尚不作为GPU精度或性能通过；双端live仍精确group8默认，正式最佳107.330 μs不变。

双seed已完成：`a07_m32_scc_direct_snapshot_seed0_v1` / `seed17_v1`均rc0、未超过120秒，各28项零字节差异，覆盖原HEAD有效payload/ScaleA/GEMM2/MoE、三次eager/graph canary及三组同地址dynamic input/psum replay；实际initialized ELF均精确匹配`6252e645...`。双seed log SHA分别为`e863ce71e728d9c2c30131866cb97984f5d7ba5b9637fe9e38ed20c0e891753f` / `93edfac08e71d0c732ae0e842849a119ee133242107fd503d5a8ea7259510449f`。验收=`m32_scc_direct_snapshot_correctness_acceptance_v1.json`，correctness-only所有耗时无效。

固定性能队列=`run_m32_scc_direct_perf_v1.py`，SHA=`c5bc2744f192c60de825178df7fb54e872a4453c1c87a7d8f3db7a2cb9f45b57`。计划为未修改的原完整命令`0_control_a→1_perf_a→0_control_b`，每段三轮，仅ownership污染整组最多三次，execution failure立即停止；两侧控制都支持收益才固定一次`1_reverse_a→0_reverse_control→1_reverse_b`。部署与恢复均整机idle/SHA守卫，source helper只默认关闭接入，本地live不部署本项。candidate正式运行必须记录已验收ELF的patch record，保持原host launcher和dispatch；有限binary实验无论性能是否有收益，结束都恢复默认，收益需再转成可维护源码。本项性能队列已启动，尚无新正式性能结论。

### M32 SCC隔离的完整正反向性能与自动恢复

固定队列已正常结束，27轮原完整benchmark均rc0、未超过120秒，没有本项execution failure。以下只列完整三轮ownership合格的组，中值分别计算；不拼接受污染组，不把ATT cycles作为kernel耗时。

| 固定组 | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2中值（μs） | MoE e2e中值（μs） |
|---|---|---:|---:|---:|
| control_a try2 | 108.174 / 106.643 / 107.950 | 107.950 | 93.1807 | 257.483 |
| candidate try3 | 106.350 / 106.858 / 106.162 | 106.350 | 92.4302 | 254.023 |
| control_b try1 | 108.393 / 107.655 / 107.623 | 107.655 | 94.6818 | 258.120 |
| reverse_candidate_a try1 | 105.700 / 107.090 / 106.225 | 106.225 | 92.3751 | 254.633 |
| reverse_control try1 | 107.343 / 107.292 / 107.362 | 107.343 | 94.0347 | 258.057 |
| reverse_candidate_b try1 | 107.010 / 106.793 / 105.778 | 106.793 | 92.8727 | 253.875 |

正向候选比两侧控制减少1.4822% / 1.2122%；反向两组比中间控制减少1.0415% / 0.5124%。正反向均支持小幅收益；GEMM2没有修改，表中的波动不作为GEMM2优化归因。control_a try1、candidate try1/try2存在ownership污染，全部三轮排除，中值保持null，raw完整保留。

结束时整机idle、KFD空、VRAM176582656 B、boot保持`967d0634-cfc3-4c4f-a711-dc6c7995ef5b`，各轮本任务`live_post_owners=[]`。`m32_scc_direct_withdrawal_host_v1.json`在`2026-10-18T12:48:02Z`记录自动SHA守卫恢复精确group8默认并移除live helper；candidate snapshot和证据保留。最终summary SHA256=`12e0e77685db455f85b57ad1cf9a65f1219183fe893d54341857c2887d24e9cc`；receipt内`summary_sha256_before_restore_flag`在添加`default_restored`前保存，保持其先后事实。

该结果属于精确ELF有限隔离，不把106.225 μs替代可维护默认的107.330 μs。默认相对原HEAD历史153.405 μs仍减少30.03%，距80 μs差27.330 μs。下一项在独立snapshot尝试仅对M32 compute use的整数比较输入加入tied SGPR identity，阻止跨use条件CSE；load predicates、原有效计算、同步、GEMM2/helpers保持。与此前缓存四个条件值及直接`={scc}`的候选不同，本项不缓存i1条件，不用精确SHA的binary patch作生产实现。先进行CPU功能和actual ISA资源验收，再决定是否运行双seed严格精度及新固定性能对照。

### M32局部条件的源码实现：v1资源淘汰，v2编译与精度通过

新增默认关闭的`AITER_GFX1250_GEMM1_M32_PREDICATE_INPUT=1`，launcher参数为`m32_predicate_input=0`，候选symbol带`_m32predinput`。在条件使用点，以空LLVM inline asm的`=s,0` tied identity返回原整数bound，再执行原整数比较；这是阻止条件跨use合并的compiler barrier，不是memory fence，不缓存i1条件。v1只对compute predicate处理、保留原load scope，CPU的232960项predicate检查通过，但actual codegen拆成两个分支：4482→4728 instructions、VGPR486/SGPR68、4个VGPR spills、private12 B，branch sites增加。根据事前资源门槛淘汰v1，没有执行其GPU精度或性能测试；失败源码和产物保留于`m32_predicate_input_v1_source/`。

v2将同一M32 pair的未来A pair loads及当前WMMA放进同一个局部条件，保留两个operand slots和原K128更新顺序。未来pair loads只从相邻位置移入同pair scope，原head的B/scales与BACK ScaleA时机保持；`load_pair_data`增加`already_live`，避免重复predicate。实际AST审计`audit_m32_predicate_input_v2.py`覆盖2080项case、523264次M16有效更新和256960次未来pair loads；原load/sync events多重集相同，关闭路径event顺序与默认一致，未涉及nested函数、GEMM2及公共helpers保持。

固定关闭→打开actual compile均rc0、未超过120秒、无GPU dispatch。关闭完整ELF仍为默认`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`；候选ELF为`a9cf7d64fe70c9524185ce9aa4272865af46ef9b9ad55f31201d68898ce80557`，33736 B、4359 instructions，较默认少123条。VGPR486/SGPR68/LDS282624 B、zero spills/private保持，没有新增branch sites；原WMMA、DS/TDM及数值opcode数量保持。八条K backedges/四targets、28/32 WMMA region、READY tensorwait2→DS reader release→WG signal/wait→refill、payload/ScaleA output fencing、两条exit tensorwait0与96个scale dwords均经native检查。验收为`m32_predicate_input_compile_acceptance_v2.json`，源码manifest为`m32_predicate_input_source_manifest_v2.json`。

独立snapshot的seed0/17均rc0、未超过120秒，各28项零字节差异，包括原HEAD有效payload/ScaleA/GEMM2/MoE、三次eager/graph canary与三组同地址dynamic input/psum replay。实际loaded ELF均匹配上述候选SHA；验收为`m32_predicate_input_snapshot_correctness_acceptance_v2.json`，双seed log SHA分别为`d2c94408a56d39089636626b6215b3dfb4e2c6592a455ae520ebf940fe3cc2af`和`439fe1df10bb3b291855256744d8ba9c921417f49e1082292ad152b1594e979f`。这些correctness-only计时不作为性能数据。

正式固定队列为`run_m32_predicate_input_perf_v2.py`，SHA256=`d40f1a7132b4c59b7568462d645cb22322eff537af956c1114849173b3590040`。仅在a07-3/dsr1_ep4测试；部署前整机idle、branch/HEAD/image/boot及默认三源SHA重检均通过，开关默认0。独立live验证已通过并绑定实际loaded ELF。随后按原完整用户命令`0_control_a→1_perf_a→0_control_b`各三轮；仅ownership污染允许整组最多三次，execution failure立即停止。两侧控制均支持收益时才运行一次固定`1_reverse_a→0_reverse_control→1_reverse_b`。队列结束自动在整机idle与SHA守卫后恢复默认，所有raw保留。本项正式性能尚未完成，当前默认最佳仍107.330 μs。

### M32局部条件完整性能队列：候选三组ownership未通过

队列结束，共18轮原完整benchmark均rc0、未超过120秒。前控制第一次第3轮、candidate第一次第3轮、candidate第二次第1/3轮及candidate第三次第1轮ownership不合格，整组排除，candidate没有完整三轮合格中值。因此没有正反向收益结论、不追加第四组、不设为默认；105.572 μs只是candidate单轮，不能替代107.330 μs正式默认。

| label后缀（前缀`a07_m32predinput_`） | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2中值（μs） | MoE e2e中值（μs） |
|---|---|---:|---:|---:|
| `0_control_a_v2_try1` | 106.603 / 106.599 / 107.096 | 不统计 | 不统计 | 不统计 |
| `0_control_a_v2_try2` | 107.056 / 106.890 / 109.397 | 107.056 | 94.9833 | 258.206 |
| `1_perf_a_v2_try1` | 109.458 / 106.206 / 106.415 | 不统计 | 不统计 | 不统计 |
| `1_perf_a_v2_try2` | 106.869 / 108.534 / 107.453 | 不统计 | 不统计 | 不统计 |
| `1_perf_a_v2_try3` | 106.725 / 105.924 / 105.572 | 不统计 | 不统计 | 不统计 |
| `0_control_b_v2_try1` | 108.509 / 106.983 / 106.479 | 106.983 | 94.3889 | 258.061 |

结束时外部PID1959076在同容器执行另一个E64优化目录的验证，不属于本任务；保留post快照并自动每6秒等空闲，没有杀外部进程。`2026-10-18T13:28:16.683797Z`整机idle、KFD空、VRAM176582656 B、boot保持后，队列自动恢复精确group8三源。receipt=`m32_predicate_input_withdrawal_host_v2.json`，summary=`m32_predicate_input_complete_summary_v2.json`；本地始终同一默认。全部compile/correctness/raw性能证据包`m32_predicate_input_raw_evidence_v2.tar.gz`为2081408 B，SHA256=`6c990b9c6e9a71007dde8380714e48145e72111b8d67fc55440cfc0b24303794`，逐member SHA、18轮raw和两个控制中值验收通过，receipt=`m32_predicate_input_complete_evidence_acceptance_v2.json`。

### 新候选：ballot upper_bound与readlane取expert end

仅在独立`expert_ballot_lookup_v1_source/`中新增默认关闭的`AITER_GFX1250_GEMM1_EXPERT_BALLOT_LOOKUP=1`。E≤96时lane-parallel读取至多三组32项expert ends，对`valid & end<=blk_m`使用`ballot + ctpop`计算原upper_bound，再从已加载值中`readlane`取当前expert end。有效expert结果与原固定8步bisect一致，capacity tail保留原`E+1` sentinel及负`mn_oob`；重复psum和空expert保持。该候选不采用此前三个shuffle-xor reduction的parallel lookup，也不合入已结束的M32性能队列。原psum、routing、TDM、compute、所有56次K128有效更新、数值和GEMM2保持。

CPU实际AST检查覆盖E=1/2/3/4/7/8/16/31/32/33/63/64/65/66/95/96、空expert、重复psum及有效/capacity行边界，共54968组；所有source读取下标在界内，expert/eb64/mn_oob逐项相同。关闭候选完整launcher/wrapper AST与默认一致，46个compute/load/fence/epilogue内部函数未改。源码审计=`expert_ballot_lookup_source_audit_v1.json`，manifest=`expert_ballot_lookup_source_manifest_v1.json`。硬件依据为CDNA5 ISA §15.3的BCNT语义与§15.8的READLANE语义；实际降低哪些memory/control指令须以native检查为准，不将CPU load计数视为HBM流量或性能。

固定关闭→打开actual compile已在a07-3/dsr1_ep4启动，每项仍120秒、无GPU dispatch，remote live保持group8默认。只有关闭完整ELF相同、candidate zero spills/private及原compute/output fences合格，才继续双seed严格device验证。当前正式最佳仍107.330 μs，距离80 μs差27.330 μs。

两项actual compile均rc0、未超过120秒。关闭完整ELF仍精确默认`7c643f...`，candidate为`b136c6392033d65ea04ed5a474d1ffb20c0e39411207c5edc9be33fa485082bd`，33984 B、4422 instructions，较默认少60条。VGPR486、SGPR70、LDS282624 B、zero spills/private；八条K backedges/四targets、原28/32 WMMA region、READY/DS reader release/WG/refill、payload/ScaleA fencing及两条exit tensorwait0通过。lookup native包含两条`global_load_b32`、三个`s_bcnt1_i32_b32`和一条`v_readlane_b32`；最后一组E65只有一个有效expert end，编译器复用原last-end读取。原九个`s_load_b32` sites消失，所有有效math及DS/TDM opcode counts保持。验收=`expert_ballot_lookup_compile_acceptance_v1.json`。

独立snapshot的seed0/17各28项零字节差异、rc0、未超过120秒，HEAD有效payload/ScaleA/GEMM2/MoE、三次eager/graph canary及三组同地址dynamic input/psum replay通过，实际loaded ELF均匹配candidate。验收=`expert_ballot_lookup_snapshot_correctness_acceptance_v1.json`，双seed log SHA分别为`5e0b62e0e956b212c688be526af816aa7f57a044be35184ab0decd25941c55e3`与`0b34c7b7e6c3e4d59bafd34957e09af715a33f7c32117f358f0d2e804d03dfa9`。correctness-only所有计时弃用。

新的固定完整原命令性能队列=`run_expert_ballot_lookup_perf_v1.py`，SHA256=`6e59b992de9a1939ebc176503b4755309769c6c09c5e54775fb414029478190a`。先整机idle/SHA守卫临时部署默认关闭开关，再独立live binding；固定0→1→0各三轮，只有ownership污染整组最多三次，两侧一致收益才固定反向1→0→1。execution failure停队列，最后自动恢复默认。该queue不重启此前任何失败或已结束的候选实验；此时没有本项正式性能中值。

### ballot lookup完整固定性能：前控制污染，未提升默认

18轮原完整benchmark均rc0、未超过120秒，candidate独立live binding通过并匹配`b136c639...`。前控制三次整组均ownership不合格，后控制首次也不合格；保留全部raw，不拼接单轮，不追加第四次前控制或反向。

| label后缀（前缀`a07_eballot_`） | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2中值（μs） | MoE e2e中值（μs） |
|---|---|---:|---:|---:|
| `0_control_a_v1_try1` | 107.779 / 107.208 / 109.610 | 不统计 | 不统计 | 不统计 |
| `0_control_a_v1_try2` | 108.661 / 106.675 / 106.839 | 不统计 | 不统计 | 不统计 |
| `0_control_a_v1_try3` | 107.197 / 106.879 / 107.528 | 不统计 | 不统计 | 不统计 |
| `1_perf_a_v1_try1` | 105.831 / 105.667 / 105.892 | 105.831 | 94.5004 | 255.013 |
| `0_control_b_v1_try1` | 107.362 / 107.016 / 108.387 | 不统计 | 不统计 | 不统计 |
| `0_control_b_v1_try2` | 110.896 / 107.427 / 106.270 | 107.427 | 95.0762 | 258.179 |

candidate比后控制低1.4857%，但没有合格前控制，不能认定稳定正反向收益或将105.831 μs替代107.330 μs正式默认。`2026-10-18T13:49:10.684487Z`整机idle、KFD空、VRAM176582656 B、boot保持后，队列SHA守卫自动恢复精确group8三源。本地live一直保持默认。summary=`expert_ballot_lookup_complete_summary_v1.json`，receipt=`expert_ballot_lookup_withdrawal_host_v1.json`。

全部source、compile、双seed、live binding及18轮raw包`expert_ballot_lookup_raw_evidence_v1.tar.gz`为1540930 B，SHA256=`ec28ce079bde622b22d84ae3e539b1b767d0d7946319ef43142af27b8c097c16`，逐member SHA、实际loaded ELF和两个合格中值验收通过；acceptance=`expert_ballot_lookup_complete_evidence_acceptance_v1.json`。这项仍是默认关闭的独立候选，不重启已结束队列，目标80 μs尚未达到。

### 最后两个K256按M32 pair完成累加并立即量化

旧130us记录的final-K128-only retirement有完整失败性能和ATT，重叠预算有限，本项不照搬其结果。独立`pair_tail_quant_v1_source/`新增默认关闭`AITER_GFX1250_GEMM1_PAIR_TAIL_QUANT=1`；正常pipeline计算Ktile0..25，不carry Ktile26。保留原完整input tensor/DS/WG fence，随后将Ktile26/27的B与scales预读到register，按当前M32 pair依次执行K12852/53/54/55，然后立即进入与原epilogue完全相同的SiLU、BF16 RNE、F32 amax、FP4/E8M0和canonical LDS输出。每个有效accumulator仍有原56次有序更新，所有有效routes和数值标准保持，GEMM2/helper/原benchmark不改。

CPU实际AST审计520组row-boundary/wave cases、247296项有效WMMA更新和363776次operand reads通过；原operand read多重集相同，quant math及输出地址AST逐项相同，关闭完整launcher/wrapper AST等于精确group8。剩余input仅在stages2/3的`[141312,211968)`和`[211968,282624)`；output使用此前完成读取的stage0前部`[0,17408)`，不与仍需读取的tail input重叠。原output DS→WG→payload/ScaleA TDM→final tensorwait0完整保持。

硬件依据仍为Shader Guide §4.7.2（p160–161）的DS同wave顺序与DScnt、§4.10.1（p197）的TDM相对DS无序及跨wave可见性要求，以及§4.6.12.1的WMMA RAW/WAW/WAR规则。旧文档已指出§4.6.12（p147）与§5.3.5.2（p221–223）关于WMMA/TRANS coexecution的表述冲突；本项不假定tanh能与WMMA硬件并行，实际ISA、严格精度及原命令性能决定是否保留。

源码SHA分别为persistent=`945b8b4eb50a49800271f36be86920314705eaf771460c1dd71202071889e3c9`、wrapper=`817bf9d70697b5745e75558bed0d7b2d143914423fc14f7e564a6a7305425067`、compiler=`d4fb347a50f27c450a545c2aa6a5767cdbaaa606d8c45610aa5a81e5a8e67e13`。审计=`pair_tail_quant_source_audit_v1.json`；接下来固定关闭→打开actual compile，每项120秒、无GPU dispatch，先验收control完整ELF相同、candidate zero spills/private及tail LDS lifetime/有效K顺序，再决定device验证。此时没有本项正式性能，默认最佳仍107.330 μs。


### pair-tail v1/v2 编译资源淘汰与 v3 生命周期收紧

两版固定关闭→打开 actual compile 已结束，各项 rc0、未超过120秒、无 GPU dispatch；两版 control 完整 ELF 都等于 group8 默认 `7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`。本地与远端 live 三源仍为 `task_m_group_default_manifest_v1.json`，原 benchmark SHA 不变。以下都是编译资源，没有本项 GPU 精度、性能或 ATT 结论。

| 独立候选 | instructions | VGPR / SGPR | LDS B | VGPR spills / private B | scratch store / load sites | ELF SHA256 |
|---|---:|---|---:|---|---|---|
| v1：四套 K128 B/scales 同时驻留 | 4704 | 472 / 68 | 282624 | 20 / 84 | 20 / 44 | `1c2b72d98ccd61daa997b49991d9957e547c94ed2ac1f630c4961794173084a9` |
| v2：按 N32 column 复用 B registers | 4811 | 472 / 68 | 282624 | 16 / 68 | 16 / 40 | `ee9d1959d438b9633b024e08cc67fa48bc0b045245ca0fbe7f624c04be3fff92` |

v2 保留每个 accumulator 的56次原序 K128 更新、原量化数学和输出地址，但 tail A 因第二个 N32 column 重读，operand reads 为373120；不能沿用 v1 的363776或声称全部 read 多重集相同。CPU 审计为 `pair_tail_quant_source_audit_v2.json`。原始 `kernel.elf/isa/notes/launcher.mlir` 与 host `result/run/idle/manifest/sources` 均回收，验收为 `pair_tail_quant_resource_rejection_acceptance_v1_v2.json`。首次验收脚本把 v2 的64位 SHA 字符串抄成65位而拒绝，核对实际 ELF、summary 后只修正期望字符串，编译产物未改写、未重编译。

native 中 v2 的 scratch sites 从 PC `0x53dc` 延续至 `0x8698`。初始 stores 紧接 tail ScaleA/ScaleB 的 `ds_load_b32`，其后 scratch loads 与末尾 WMMA 混排；这为进一步缩短 tail scale 生命周期提供依据，不把静态 spills 计数转换成 kernel μs。Shader Guide §3.3.5（p37）说明 scratch 是 thread-private global memory。根据事前 zero-spill/private 门槛，两版均停止在编译阶段，不给这两版追加 GPU 计时。

独立 v3 改为每个 live M32 pair 仅持有当前一个 K128 的 B/scales，按52→53→54→55完成后立即进入相同 epilogue。保留 immutable stages2/3、完整 input fence 和原 output DS/WG/TDM/final completion；空 `=v,0` tied LDS-base identity 限制 LLVM 跨 pair 合并纯 LDS reads，它是 compiler barrier，不是硬件同步。tail B/ScaleB 明确增加每 live pair 重读；A 保持原 read 多重集，ScaleA 仅加载 live pairs。此实现代价需 actual ISA 和正式固定性能对照判断，不能仅因生命周期更短推定收益。两侧 live 仍为精确默认，正式最佳107.330 μs、距80 μs差27.330 μs。

v3 的固定 compile 在 candidate 使用未导入的 `llvm.inline_asm` 时发生 `NameError: name 'llvm' is not defined`，rc1、未超时、无dispatch，立即停止该 queue。control 已成功编译且 ELF 精确默认；失败candidate没有完整 ELF。`pair_tail_quant_compile_finished_v3.json` 与失败 raw 保留，未重启该队列。问题来自新增候选遗漏 dialect import，未修改默认或既有产物。

独立 v4 只在候选 `const_expr(pair_tail_quant)` 内补入 `from flydsl._mlir.dialects import llvm`，新 symbol=`_pairtailq2localimp`，重新执行实际 AST 的520组 CPU 审计通过；审计显式检查该导入，模拟执行时才用 identity stub。有效247296项更新、实际261632 WMMA calls、operand reads370496，每个有效 accumulator 的56次原序保持，原 quant/output AST 与关闭完整launcher/wrapper保持。persistent SHA=`e610c9a9694a02af7126b48b9bdfa1af915ba5a54b5c782c1a73d809ed282f5f`，wrapper/compiler 与 v1/v2相同。

v4 固定关闭→打开 actual compile 均rc0、未超过120秒、无dispatch。control完整ELF仍默认；candidate ELF=`5b69de4f2340de606fe6a70ec738396500881dd5f2da7e0baf41c98f523b2e0c`，4619 instructions、VGPR472/SGPR68、LDS282624 B、zero spills/private。已消除前两版 scratch，但静态资源通过不等于实际device精度或性能通过；下一步验收 native tail/input/output fences、严格双seed/graph/dynamic replay和实际loaded ELF，再运行原完整命令固定三轮对照。正式默认107.330 μs保持。

v4 native 验收=`pair_tail_quant_compile_acceptance_v4.json` / `pair_tail_quant_native_inspection_v4.json`：八条 steady K backedges、四 targets和各28/32 WMMA region保持；tail前 `tensorwait0→DScnt0→WG signal/wait`完整，tail内64条 WMMA、32条 ROW1 scales、reuse hints关闭，第一个 pair16条更新后量化，后续pair继续WMMA。之后没有input TDM，output不alias stages2/3，末尾DS/WG→payload/ScaleA TDM及两条exit tensorwait0通过。最后两块K256从原pipeline转入immutable tail，static WG signal/wait10→8并非删除仍需的跨wave可见性门禁。tail B重读实际增加96个`ds_load_b128`sites；scale dwords亦增加，不将资源收紧等同于流量减少。

snapshot seed0/17严格device验证均rc0、未超过120秒，各28条零字节差异：原HEAD全部有效payload/ScaleA/GEMM2/MoE、各三次eager/graph canary、三组同地址dynamic input/psum replay通过。实际loaded ELF均精确匹配`5b69de4f...`。验收=`pair_tail_quant_snapshot_correctness_acceptance_v4.json`，下载后独立验收=`pair_tail_quant_downloaded_correctness_acceptance_v4.json`；双seed log SHA=`cb70d5e983b2b600be853cb6675418809eb35756e751f0b7026665260a495d9a` / `989752cbb13fd81cc98ad7b3516eff5728910877fd4f7dc2042413d4c5b6ea51`。两项`live_post_owners=[]`；correctness-only计时全部弃用。

固定原完整命令性能队列=`run_pair_tail_quant_perf_v4.py`：host branch/HEAD、dsr1_ep4 image/status、boot和两端三源SHA核对后，整机idle/SHA守卫临时部署默认关闭候选，独立live ELF binding；0→1→0各三轮，仅ownership污染允许最多三次整组尝试，execution failure停队列并自动恢复默认，两侧控制一致改善才固定一次反向1→0→1。原benchmark和GEMM2/helper/routing不改，不重启历史失败队列。本项尚无新的正式性能中值，默认107.330 μs保持。

### pair-tail v4 正式性能回退与恢复默认（2026-10-18）

固定 `0_control_a→1_perf_a→0_control_b` 已完成，九轮原完整 benchmark 全部 rc0、未超过120秒，三组均首次通过完整 ownership 门禁，没有重试、拼接或按耗时筛选。

| label后缀（前缀 `a07_pairtailq_`） | GEMM1三轮（μs） | GEMM1中值（μs） | GEMM2中值（μs） | MoE e2e中值（μs） |
|---|---|---:|---:|---:|
| `0_control_a_v4_try1` | 108.161 / 107.022 / 107.636 | 107.636 | 94.7470 | 257.664 |
| `1_perf_a_v4_try1` | 112.139 / 112.380 / 114.682 | 112.380 | 94.2748 | 262.825 |
| `0_control_b_v4_try1` | 109.378 / 108.783 / 106.128 | 108.783 | 94.2434 | 258.776 |

候选比两侧控制慢4.4074% / 3.3066%，按事前规则淘汰，不运行反向、不追加性能或 ATT。实际 zero spills/private 没有转化为收益；tail B重读增加96个 `ds_load_b128` sites、ScaleB等读取亦增加，这是静态代价证据，不能唯一认定它解释全部回退。GEMM2没有修改，其波动不作为 GEMM2 优化归因。

`2026-10-18T14:36:49.572120Z` 整机 idle、KFD空、VRAM176582656 B、boot保持后，队列按三源SHA守卫恢复精确 `task_m_group_default_manifest_v1.json`。receipt=`pair_tail_quant_withdrawal_host_v4.json`；本地始终同一默认。九轮 `live_post_owners=[]`，没有本任务GPU进程残留。正式最佳仍107.330 μs，距80 μs差27.330 μs。

完整 v1–v4 source、CPU/native、双seed、live binding、九轮性能raw和恢复记录已回收：`pair_tail_quant_raw_evidence_v4.tar.gz`，2487665 B、243个regular members，SHA256=`e3bbfd56a613fc9c9ef218e3861f7b7a2d02c81f738845e6f3799df451da5a8b`。独立逐member SHA及九轮原表、ownership、三组中值和本地恢复SHA验收通过，记录=`pair_tail_quant_complete_evidence_acceptance_v4.json`。receipt里的summary SHA保存于追加 `default_restored` 标志之前，保留原先后事实。

本项性能driver有明确版本差异：SFTP异步上传期间，本地给 `run_pair_tail_quant_perf_v4.py` 补入branch/HEAD/image/boot四项assertions；远端实际运行旧版，SHA=`1a030434bbe852524d85ea102f10367d568b93f5064828d324546c00e6eaecaa`，以 `run_pair_tail_quant_perf_v4_actual_remote.py` 独立回收。当前本地补断言版SHA=`54fb3250297bbdc59b5516c7f56351ac8eab2c95c05817751f201208b8623ee6`，没有覆盖为远端版本或宣称两端一致。执行前已在host只读核对上述四项和live SHA，原runner、计时命令及ownership门禁相同；保留这一限制，不因此重跑性能挑选批次。

### group8 默认 SIMD1 ATT 补齐与 descriptor 来源绑定（2026-10-18）

既有 SIMD0 trace 仅覆盖 native owner loops0/2，因此补采 SIMD1，未重采同一覆盖。`a07_group8_default_simd1_att_v1_try1` 为 rc0、未超120秒、ownership合格、`live_post_owners=[]`；四个SE raw非空，50个waves中38个有WMMA、12个无WMMA。实际 object22 与独立默认完整ELF逐字节相同，SHA256=`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`；动态ISA index→byte PC→native opcode全部一致，无missing PC、无`.long`。源码、原benchmark、pre/post/settled ownership与boot均保持。

新增 `group8_owner_descriptor_provenance_v1.json` 把源码四个 `add_tdm_loads`、kernarg offsets40/48/56/96和精确native pointer chains绑定，确认 loops0/1/2/3 分别为A/B/ScaleA/ScaleB；不能只按loop排列认定角色。B链含PC`0x1ff0`的A/B合并kernarg load、`0x2148`的B base、`0x36e8`的owner减2及`0x3700`的源地址；ScaleB链含PC`0x22b0`的offset0x60 load、`0x430c`的owner减6和`0x4328`的源地址。SIMD1实际覆盖B/ScaleB，补齐此前A/ScaleA的缺口。

| native owner | 有效waves | 每wave WMMA counts | compute span中值 cycles | steady tensorwait2个体中值 cycles | 每wave steady tensorwait2累计中值 cycles | 每wave WG wait累计中值 cycles |
|---|---:|---|---:|---:|---:|---:|
| B / loop1 | 19 | 896×19 | 22031 | 3 | 72 | 491 |
| ScaleB / loop3 | 19 | 448×8、672×10、896×1 | 21715 | 3 | 72 | 1349 |

两组WMMA间隔中值都是8 cycles；gap>64的累计中值分别12614/14064 cycles。B/ScaleB的compute原opcode计数中值含1308/1088条 `ds_load_b128`、225/222条conditional branches、各27对WG signal/wait及各24条input TDM。独立中值不能相加、不能换算kernel μs；短steady TDM wait不排除其他memory影响，也不能唯一确定HBM、TA、FIFO、cache或LDS是瓶颈。后续按DS/控制/同步的实际位置选新实验，不重启已结束或失败的方向。

证据=`group8_default_simd1_att_acceptance_v1.json`、`group8_simd1_owner_waits_analysis_v1.json`、`group8_owner_descriptor_provenance_v1.json`。raw包=`group8_default_simd1_att_raw_evidence_v1.tar.gz`，2591170 B、101 regular members，SHA256=`36f79a97422bee6ac8d0e108bf0411c5c7f1a0630b595b86c4ec891b4b461bd2`；仅保留本trace目标object22，回收后逐member SHA验收通过。ATT日志中的102.360 μs等为诊断读数，不替代正式107.330 μs中值。

### 按用户要求重测原 HEAD baseline 与已验收 group8（2026-10-18）

用户要求重测baseline和最优版，暂停尚未实现的尺寸常量/swizzle新候选。baseline固定HEAD=`339589f900b3a880ec447f605886fe803cbf9649`，使用 `baseline_settled/sources` 的wrapper/persistent；两文件与本地 `git show HEAD:path`逐字节一致。best固定 `task_m_group_default_v1_source` 的已验收group8，不使用binary隔离或已撤回候选。`baseline_best_retest_source_acceptance_v1.json`记录两组SHA、原benchmark、GEMM2 AST一致性与runner/verify SHA。

固定序列为baseline→best→baseline各三轮；两侧均有完整合格组且支持提升时，固定反向best→baseline→best各三轮。仅ownership污染允许整组最多三次尝试，execution failure立即停止；不拼接单轮、不按耗时筛选、不延长120秒。每轮在a07-3/dsr1_ep4执行原完整MoE eager benchmark、warmup5/iters20/random non-balanced E65/T1536/topk9/shared-experts1，性能前后整机GPU/KFD门禁，每6秒等待空闲，计时期间不传输、编译、ATT或监控。

每版正式计时前单独运行有效payload/ScaleA/GEMM2/MoE逐字节对比，四项均零字节差异，实际GEMM1 ELF分别绑定baseline=`958d31dd24a22c8a38af0ec1113a27ee8c5bf69e8ffb124ff222758f11c68110`、best=`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`；两版实际GEMM2 ELF也逐字节一致。原benchmark文件不改。host初始队列=`run_baseline_best_retest_v1.py`，SHA256=`ca7f033ab4c585670e4ddd49c6a93e2624c09b3f82e747f42581a67b5446a1e4`。

v1在前六轮GPU运行均rc0后，CPU日志解析误把`[bench kernel timing]`中的截断symbol当成完整名字，触发`AssertionError`。这是验收脚本问题，没有GPU execution failure或timeout；v1恢复精确默认并保留六轮raw。v2仅把名字检查限定为含`" CUDA "`的完整profiler行，SHA256=`8ead6a79803cc10929ef56b51bfdff42e64d68bf253342a0c336ebc0be4d5c70`。v2逐项重新验收已有六轮，不重跑这些轮次，随后只执行尚未完成的后控制及反向。`parser_only_resume`记录原summary SHA、六个rc0和`restarted_failed_device_queue=false`。

27轮性能运行全部rc0、未超120秒。18轮组成下表六个完整合格组，另外9轮组成三个ownership污染组，全部整组排除。各组中值独立计算，不合并单轮、不挑最低值。

| 固定顺序/组suffix | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|---:|
| 前baseline `head_a_v1_try1` | 154.063 /154.846 /152.708 | 154.063 | 90.1171 | 299.240 |
| best `best_a_v1_try1` | 107.628 /108.958 /105.907 | 107.628 | 94.5638 | 257.823 |
| 后baseline `head_b_v2_try2` | 154.057 /153.713 /153.987 | 153.987 | 90.4144 | 299.892 |
| 反向best `best_b_v2_try1` | 108.540 /107.124 /107.381 | 107.381 | 94.5659 | 257.641 |
| 反向baseline `head_c_v2_try3` | 153.659 /154.215 /153.108 | 153.659 | 89.6808 | 299.164 |
| 反向best `best_c_v2_try1` | 107.320 /106.664 /106.900 | 106.900 | 93.3788 | 256.399 |

上述目录统一加前缀`a07_retest18_`。排除组为`head_b_v2_try1`（153.273 /152.735 /151.957 μs）、`head_c_v2_try1`（157.043 /153.619 /151.637 μs）及`head_c_v2_try2`（153.058 /154.765 /155.680 μs）；这些读数仅为排除记录，没有性能中值或优化结论。

正向best对前/后baseline分别改善30.140% /30.106%；反向前/后best对baseline分别改善30.117% /30.430%。本次确认约30%的GEMM1提升可复现，80 μs目标仍未达到。GEMM2源码与actual ELF相同，读数变化只能报告为本轮完整MoE运行中的伴随现象，现有证据不能唯一归因cache、时钟或其他硬件环节。

`BASELINE_BEST_RETEST_V2_DONE`后按SHA守卫精确恢复`task_m_group_default_manifest_v1.json`；整机idle、KFD空、VRAM176582656 B、boot未变，无本轮GPU进程残留。raw包=`baseline_best_retest_raw_evidence_v1.tar.gz`，1290096 B、186 regular members，SHA256=`a85fd02548d81d7157219a9a19a79a706950f80a3c869b3982f1c76eeddb0d9a`。本地`accept_baseline_best_retest_v1.py`逐member SHA、27轮日志/ownership、原完整命令、两版源快照/actual ELF、六组中值、parser-only恢复和live默认SHA均独立验收通过；验收=`baseline_best_retest_downloaded_acceptance_v1.json`。包保留全部合格与排除组，不含parent大tensor文件。

### host 已知 M/N 尺寸的静态 swizzle 候选（2026-10-18）

按用户要求完成baseline/best重测后，继续独立`static_swizzle_v1_source`，默认关闭开关=`AITER_GFX1250_GEMM1_STATIC_SWIZZLE=1`。只在当前E65/contiguous_m30464/N4096及既有group8/rowtail/prefetch守卫内由wrapper提供`known_m_tiles=119, known_n=4096`，两项均为launcher末尾Constexpr参数，不增加kernel runtime kernarg。把112个full-group M tiles和7个tail-group M tiles分别做常量除法，同时让B batch stride与ScaleB batch offset使用同一实际N常量；动态expert lookup、M-order及原fallback保持。

`audit_static_swizzle_v1.py`执行实际mapping AST，全部1904 WG在关闭/打开状态与默认的`(m_tile,n_unit,local_n)`一致，映射双射及cluster peers一致；7260个live/last-expert边界的M-order仍为同一双射。关闭分支还原后persistent/wrapper/compiler三个完整module AST均与默认相同，全部nested load/compute/quant/output functions及GEMM2/public helpers不改，`n64`之后完整AST保持。source audit=`static_swizzle_source_audit_v1.json`，这只是CPU等价性证据，尚无device精度或性能结论。

源SHA：persistent=`ec50c9b254f9f58c82bc9dae12e8086a009d24c0ea56e0d214f42e2cbd50adcc`，wrapper=`894a6f95cad78175d58d16b7454a41b9b4929d4523b50d640f204a90f4ec6473`，compiler=`8467d049f312eeff3f73cc26242b3d624972f2456f920fc0635d0bdf320c91d8`。独立快照和本项审计/驱动共9个文件打包同步，archive SHA=`c829bceb081d11c01dd144d59c4b6cb4ca0bf9f932f7e83f2d346b7116042bb5`，不覆盖live源码。接着固定0→1 actual compile，每项120秒、无dispatch；要求control完整ELF精确默认、candidate zero spills/private和native计算/同步验收，再决定严格双seed/graph/dynamic及正式0→1→0性能。当前默认107.330 μs记录保持。

实际`a07_staticswz_0_compile_v1`于`2026-10-18T16:00:32Z`启动，到120秒上限由既有runner终止，rc124/`timed_out=true`；`queue_static_swizzle_compile_v1.py`立即停止，mode1未执行。日志只到`aiter/jit/core.py`的`module_aiter_core`导入，没有创建compile输出目录、没有生成ELF或dispatch证据。不能从本次超时判断候选device精度、资源或性能，也不重启原失败队列或延长上限。control log SHA=`9554da2e7b43b45970cd700492b00580b4762f8f9455024c7daa2cb91e127144`，raw log SHA=`842d777f9b203dff83e25955f0ef0236c3ed4982dbaa25c782c3de24e9e875682`。

结束`live_post_owners=[]`，live三源始终为精确group8默认。post/settled曾出现外部同容器`bench_commit_tree.py`任务，不能称整机一直空闲；本项为无dispatch correctness-only编译，不使用任何耗时作性能。随后`2026-10-18T16:04:57Z`独立整机检查idle、KFD空、VRAM176582656 B，无本项残留。按用户新要求先补齐当前生产dispatch的E65/E64完整MoE验收，保留超时原始记录。

完成双规模验收后，独立`diagnose_static_swizzle_import_v1.py`仅导入模块、不compile、不dispatch，约1.972秒已导入AITER和FlyDSL，随后卡在候选`@flyc.jit`初始化。20/40/60秒等faulthandler栈指向`flydsl/compiler/ast_rewriter.py:542–575 visit_BoolOp`的`ast.fix_missing_locations`，到120秒rc124终止，无live残留。raw log SHA=`64a99646f311eec4d51ab139fa513b039fe596e3acbd0a1af5fef6278eada39e`。读取实际rewriter可见每个后续`and` operand都把前一result重复嵌入condition/body/orelse，长BoolOp展开成本随条件数迅速增长；栈结合新增长14条件guard定位为CPU AST改写问题，不把该超时归因GPU kernel。

新建`static_swizzle_v2_source`，只把新14条件compile-time guard改为等价`all(tuple)`；逐operand AST相同，恢复该guard后整个module AST与v1相同，wrapper/compiler字节保持。persistent SHA=`5527978c3c14039074a89bf2855433ae20e419d574e9addd381bc2fa3629b16a`，其余仍v1；`static_swizzle_guard_rewrite_acceptance_v2.json`与重新执行的1904WG/7260M-order源码审计通过。此前CPU builder查找assert的unparse前缀失败发生在任何snapshot写入前，修为AST结构定位后构建成功，没有GPU运行。v2使用新labels固定0→1编译，不重启v1失败queue、不延长120秒。当前仍无static-swizzle正式性能。

v2关闭项rc0，生成完整ELF与默认`7c643f5c...`精确一致；开启项在实际MLIR构建时rc1，`n_unit`在动态`scf.if`前被FlyDSL视为`None`，报错要求初始化为相同类型。队列立即停止，无candidate ELF或dispatch。raw log SHA=`1273afb78db2d69a6c40b91c2a44689730f4fb96158656861c2f25f4a6e93352`；关闭项post记录有外部owner2908563，开启项`live_post_owners=[]`，这些无dispatch编译不使用任何耗时作性能。

建立独立`static_swizzle_v3_source`，只在开启分支的动态if前增加`m_unit=fx.Int32(0)`和`n_unit=fx.Int32(0)`，每条有效路径仍覆盖这两个结果。移除两条类型初始化后完整module AST与v2一致，关闭后三个module AST还原默认，1904WG/7260M-order及全部nested函数重新验收。persistent SHA=`388dc83a26b175451964483e2ff09560dca2a7ee879f3f9a257a82a065855c24`；archive=`static_swizzle_inputs_v3.tar.gz`，60929 B，SHA256=`502d5235b07bf2caf267b56839275b04c4d0d4368c83866662a8f3669369f47b`。远端逐member SHA验收后固定0→1无dispatch编译，每项仍120秒；live保持默认，没有新的性能结论。

### static-swizzle v3编译、双规模完整精度与正式性能收尾（2026-10-19）

v3固定0→1无dispatch编译均rc0、未超过120秒。关闭ELF仍为默认`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`；开启ELF=`f5b4ad99b6b85d3c200500df8847bed5283fd985fab0828c091597d3e314c92c`，34096→33744 B、4482→4373条instructions，两处`v_s_rcp_f32`消除。VGPR486/SGPR68/LDS282624 B、private/spills0和176 B runtime kernarg保持。实际256条WMMA、384条`ds_load_b128`、20/2 input/output TDM、128条tanh、数值转换opcode保持；八条steady backedges/四targets、28/32 WMMA regions、READY tensorwait2→DS reader release→WG signal/wait→refill及最终tensorwait0通过native验收。

候选源码在E65开启的seed0/17完整MoE验证，以及E64关闭的seed0/17原fallback完整验证均rc0、未超120秒，各28项零字节差异。包括原HEAD有效GEMM1 payload/ScaleA、GEMM2有效行、最终MoE、各三次eager/graph canary及三组同地址dynamic A/ScaleA/psum replay。actual E65 ELF绑定`f5b4ad99...`；E64仍绑定原`addf6e2a...`。这不是在E64开启static specialization，reference门限未改，诊断计时不作性能。

固定0→1→0原完整benchmark结束，共15轮全部rc0、未超过120秒，无execution failure。每组三轮，仅按ownership污染整组最多三次；全部前缀=`a07_staticswz_`：

| 后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2合格中值（μs） | MoE e2e合格中值（μs） | ownership |
|---|---|---:|---:|---:|---|
| `0_control_a_v3_try1` | 110.168,107.320,107.136 | 107.320 | 94.7261 | 257.675 | 三轮合格 |
| `1_perf_a_v3_try1` | 106.369,105.450,105.899 | 105.899 | 94.6502 | 255.346 | 三轮合格 |
| `0_control_b_v3_try1` | 106.637,107.581,108.140 | 不统计 | 不统计 | 不统计 | 第3轮污染 |
| `0_control_b_v3_try2` | 107.999,106.660,107.887 | 不统计 | 不统计 | 不统计 | 第1/3轮污染 |
| `0_control_b_v3_try3` | 107.387,107.760,106.295 | 不统计 | 不统计 | 不统计 | 第1/3轮污染 |

候选比前置合格控制快1.324%，但后控制三次ownership尝试耗尽，没有完整双侧对照，不能认定稳定收益、不启动反向、不追加第四次、不重启已结束queue，也不把105.899 μs替代独立已验收默认107.330 μs。即时live列表为空不足以忽略settled污染。summary=`static_swizzle_complete_summary_v3.json`，`finished=true/performance_data=false/execution_failure=null/reverse_started=false/default_restored=true`；其中`performance_data=false`表示未成立完整双侧性能结论，两项合格组各自保留。

队列结束post见外部PID3047276运行其他目录的`bench_imported_optx.py optx_core_v187`，未杀该任务；按6秒门禁等待，到`2026-10-18T16:40:07.397441Z`整机idle、KFD空、VRAM176582656 B，SHA守卫精确恢复三源默认。receipt=`static_swizzle_withdrawal_host_v3.json`。本轮raw包=`static_swizzle_raw_evidence_v3.tar.gz`，2197923 B、273 regular members、SHA256=`12d2cec6f2e5f714ab8427830ccd9cedddcf9ae80c4820a6b3199dd705ec4460`，包含v1/v2失败、v3候选source/native、四项完整精度、全部15轮raw及撤回记录，不含tensor dumps。已本地独立核验member SHA、ELF/ISA、28项比较、完整命令、ownership、全部中值及恢复，receipt=`static_swizzle_raw_downloaded_acceptance_v3.json`。canonical远端receipts解包到独立`static_swizzle_raw_download_v3/`，保留已有alias本地receipts而未覆盖。

### E65/E64 当前生产路径的完整MoE精度验收（2026-10-19）

在手头static-swizzle编译队列因120秒超时停止后，按用户新要求补验**当前生产dispatch**的两个完整规模，均为T1536/model_dim7168/inter_dim2048/A4W4/SiLU/no-bias、random non-balanced。E65为topk9/shared-experts1、13824有效routes；E64为topk8/shared-experts0、12288有效routes。既有`a07_fallback_routes_e64_t1536_default_seed17`实际上是E64/topk9/shared1，不能拿来替代本次指定的E64规模。

E65复用同一live源码SHA的已验收`a07_mgroup_default_seed0_v1`和`a07_mgroup_default_seed17_v1`，重新核对raw SHA、源快照、原完整MoE有效输出及全部28项比较，避免重复已完成测试；本次重测的best actual GEMM1/GEMM2 ELF也已分别绑定`7c643f5c...` /`840d2c76...`。E64独立运行`a07_dualmoe_current_e64_seed{0,17}_v1`，不导入side-copy GEMM1，不设置GEMM1候选环境开关，host runner明确设置`AITER_MOE_EXPERT_BALANCE=false`并执行完整MoE reference检查。两项均rc0、未超120秒。

| 当前生产规模 | seed | logits_diff | rel_l2 | HEAD/固定parent有效payload、ScaleA、GEMM2、MoE差异 | eager/graph/dynamic |
|---|---:|---:|---:|---|---|
| E65/topk9/shared1 | 0 | 3.7793e-06 | 2.7493e-03 | 四项均0 bytes | 全通过 |
| E65/topk9/shared1 | 17 | 3.7788e-06 | 2.7491e-03 | 四项均0 bytes | 全通过 |
| E64/topk8/shared0 | 0 | 3.4733e-06 | 2.6357e-03 | 四项均0 bytes | 全通过 |
| E64/topk8/shared0 | 17 | 3.4766e-06 | 2.6369e-03 | 四项均0 bytes | 全通过 |

每seed的28项包含四项完整MoE有效输出逐字节对比、三次eager/graph canary共12项、三组同地址dynamic A/ScaleA/psum更新的eager/graph共12项。GEMM1结果恢复原输入后继续真实GEMM2和MoE combine，最终MoE/reference的`logits_diff<0.01`原门限不变，同时保留rel_l2读数；不只验证GEMM1。E64有效payload/ScaleA分别为12582912 /786432 B，GEMM2有效输出176160768 B，最终MoE22020096 B；固定生产parent的四项均0 bytes差异。standalone producer、GEMM2及MoE输出也均bitwise一致。dynamic replay包含空expert及M16/M32/M64/M128/256边界，恢复原输入后完整输出仍一致。

E64实际GEMM1 ELF=`addf6e2a0436d2b17137d37757615ff8dda939b837b354fb2a7450c28e64bf9c`，实际GEMM2 ELF=`6982c0689aa1c8934ba07fa210b1b17d9210c79ffbc062e31acb5e999e84582c`，两seed一致；capacity28672、GEMM1 hint6、GEMM2 hint0均记录在binding，证明走当前fallback。E64 parent来自`verify_baseline_seed{0,17}_layout_v2/effective.pt`，actual ELF也为同一fallback，参考文件SHA分别`856689303dab0e652b4a805971c22801a12d296e4fde15f5df5b1984407aea429` /`36ac1ac511a37b919f74e1f0451c040f0363b271a7fab6274e93be0ff114e94ac`。本次不是E64优化copy性能或旧失败queue的重启。

用户追问E64/E65是否同一GEMM1：二者来自同一`mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py::launch_gemm_a8w4_tdm_fused_persistent`和`t256/w2x4/b4`基础geometry，但运行不同编译specialization。E65命中`n_experts==65/contiguous_m==30464/non-balanced`调优守卫，启用单N-task、rowtail3、staged-ScaleA、halfA/halfScale、rowaload2、pairA、M-order1/group8；E64为`n_experts==64/contiguous_m==28672`，保持原默认4 N-tasks和未开启这些E65开关的路径。`n_experts`为Constexpr，本身也参与编译区别。这里fallback表示同一launcher的原默认路径，不能把它解释成E65最优配置已经在E64开启并验收。

2026-10-19用户明确收紧接入要求：**E64和E65必须使用同一个GEMM1，不走额外编译specialization，并记录E64性能**。此前E64 fallback精度通过仍是有效历史证据，但不满足这一新要求。停止尚未部署/执行的raw-LDS草稿实验，先统一两规模的真实expert数runtime参数、去除tensor shape参与GEMM1编译的依赖、使用共同调优配置与TDM hint；在同一进程核对编译缓存项和实际ELF一致后，再验证两个完整MoE并分别记录性能。当前尚未完成统一实现，旧已验收默认性能107.330 μs仍仅指E65，不把它当作新统一kernel结果。

复现E64当前生产路径，在a07-3主机repo根目录执行以下命令；label及binding输出首次运行均不得已存在：

```bash
python3 -u my_code/run_dsr1_e65_gemm1_host.py \
  a07_dualmoe_current_e64_seed0_v1 --correctness-only --command \
  python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/verify_current_dual_moe_v1.py 64 0 \
  --experts 64 --tokens 1536 --topk 8 --shared-experts 0 \
  --model-dim 7168 --inter-dim 2048 --seed 0 --replays 3 --dynamic-replays 3 \
  --compare .codex_tmp/side_e64_gemm1_20261018/verify_baseline_seed0_layout_v2/effective.pt
```

seed17同时替换label、位置seed、`--seed`和parent路径。E65沿用原`my_code/verify_dsr1_e65_gemm1.py --seed {0,17} --replays 3 --dynamic-replays 3 --compare .codex_tmp/dsr1_e65_gemm1_20261017/parent_seed{0,17}.pt`，相同host runner确保120秒和false routing。所有correctness-only运行的计时弃用，没有新性能结论。

本地逐member SHA、source manifest、完整命令/精确geometry、rc/timeout、每seed28项比较、精度与actual ELF绑定独立验收通过，receipt=`dual_moe_current_downloaded_acceptance_v1.json`。raw包=`dual_moe_current_raw_evidence_v1.tar.gz`，784548 B、81 regular members，SHA256=`d3d3b5d60f2b03653813a818d835f3fa5075f95fa66385ccae68d58aa1a1323e`，包含既有E65双seed、本次E64双seed、固定parent来源证据及static-swizzle超时记录，不含大tensor文件。E64两项`live_post_owners=[]`，结束`2026-10-18T16:09:36Z`整机idle、KFD空、VRAM176582656 B、live源码未改。两规模有限验收通过，历史完整profiler之后的偶发GPU fault仍保留为未定位，不能从本次通过推定已修复。

### E64/E65 同一个已编译 GEMM1：runtime expert count 与完整精度验收（2026-10-19）

统一候选 `unified_gemm1_v1_source/` 保留真实 E64/E65 tensor、权重、routing 和 allocation，GEMM1 launcher 使用共同 Constexpr selector `n_experts=0`，实际 expert 数通过新增 `runtime_n_experts: fx.Int32` 传入 kernel。expert lookup、M-order 和 capacity-tail guard 使用真实 runtime count。没有把 E64 补成 E65，也没有按数据值裁剪有效 routes。两个规模共用原 E65 group8 的 single-task/rowtail3/staged-ScaleA/halfA/halfScale/rowaload2/pairA/M-order1/group8 配置，GEMM1 TDM hint 固定0；GEMM2 AST 保持。FlyDSL 的 Torch tensor 默认 `dynamic_layout=True`，原 Tensor ABI 本身已支持动态 shape/stride，无需改成 raw pointer ABI。

source manifest：persistent=`ecb38530e6a31071078b6450e85cda543b8661ba7dcbc9e040a966d8aaee20ae`，wrapper=`98b701a70afd8c27dd651b4823ae5999bf21f984765988aebd3331f15405347f`。CPU 审计保留47个 nested load/math/output/sync 函数和完整 GEMM2；固定8步 lookup 对2–128 experts、空 expert 与 capacity 边界共2544 cases 通过，实际 map 读取无 OOB。此审计只说明所覆盖的源码逻辑，不代替 device 验收。

在 a07-3/dsr1_ep4 同一个无 dispatch 编译进程按 E65→E64→E65 调用，tensor shape 和真实 runtime count 变化，`cache_entries=1`、cache key、compiled object id 和 IR 均保持相同。共同 ELF SHA256=`cb20b50ee53ec51daaf0eaf3011b0f07ac98f814ad13c4214a581c9eb2cb68ab`，34360 B、4547 instructions，VGPR486/SGPR70/LDS282624 B、private/spills0，runtime kernarg180 B，新增 expert scalar 在 offset176/size4。kernel 名包含 `_eruntime_`。native 验收全部有效 WMMA、量化、TDM、LDS 指令和同步；八条 steady backedges/四 targets、28/32 WMMA regions、READY tensorwait2→DS reader release→WG signal/wait→refill 以及最终 tensorwait0 保留，branch targets 均为指令边界。receipt=`unified_gemm1_compile_acceptance_v1.json`。

随后 seed0、seed17 各在同一实际 GPU 进程依次运行 E65/topk9/shared1 和 E64/topk8/shared0 完整 MoE。两项进程均 rc0、未超过120秒，始终仅一个 GEMM1 cache entry、同一 compiled object，实际 loaded ELF 与上述共同 ELF 一致。各规模各 seed 完成28项零字节差异：四项有效 payload/ScaleA/GEMM2/MoE 与固定原基线对比，三次 eager/graph canary 共12项，三组同地址 dynamic A/ScaleA/psum 更新共12项。完整四 workload 共112项；dynamic replay 包含空 experts 和 M16/M32/M64/M128/256 尾行，之后恢复原输入继续 GEMM2 和 combine。

| 统一 GEMM1 workload | seed | logits_diff | rel_l2 | 28项比较 |
|---|---:|---:|---:|---|
| E65/topk9/shared1 | 0 | 3.7793e-06 | 2.7493e-03 | 全部0 bytes |
| E65/topk9/shared1 | 17 | 3.7788e-06 | 2.7491e-03 | 全部0 bytes |
| E64/topk8/shared0 | 0 | 3.4733e-06 | 2.6357e-03 | 全部0 bytes |
| E64/topk8/shared0 | 17 | 3.4766e-06 | 2.6369e-03 | 全部0 bytes |

原 `logits_diff<0.01` 门限保持，所有 correctness-only 耗时弃用。binding 的 `geometry.tdm_b_th` 记录上游 wrapper 传入值（E65为0，E64为6），统一 wrapper 在实际 GEMM1 launch 前将它覆盖为0；两个规模实际加载相同 ELF 也已验证这一点，不能将上游字段误写成统一 kernel 的不同 hint。下载后的 raw log、source/geometry/ELF、112项比较独立验收通过，receipt=`unified_gemm1_correctness_downloaded_acceptance_v1.json`。这一步仍在独立 opt-in snapshot 中，默认 live 暂时保留旧版，正式双规模性能和默认接入待完成；旧已验收 E65 107.330 μs不作为统一候选的新性能。

正式性能队列 `run_unified_gemm1_dual_perf_v1.py` 固定原 E65/E64 前控制→统一 E65/E64→原 E64/E65 后控制，各三轮；仅 ownership 污染允许整组最多三次，execution failure 立即停止。新增 host runner 正式 `--experts/--topk/--shared-experts` 参数，不通过 `--command` 诊断入口报告 E64 性能，原 benchmark 文件、warmup5/iters20 和计时不改。计时中不并行 GPU 检查、编译或传输；source 切换由整机 idle 与 SHA 守卫控制，队列结束恢复旧默认后再决定共用版接入。

初版队列的两组旧前控制各三轮合格；随后共用E65三个GPU进程均rc0且ownership合格，但profiler parser不识别`_eruntime_`，正式表GEMM1单元格为空，汇总时`ValueError`使队列停止。三轮rounded raw profiler结果保留，不计为正式GEMM1数据。新增准确的runtime symbol解析，只在E64/E65和当前fp4/K7168/SiLU/t256/w2x4/b4/q1r8/cn4共同geometry识别GEMM1，wrong stage/geometry及duplicate ambiguity仍拒绝；120个旧数字symbol fixture与旧parser完全一致。`timed_workload_AST_unchanged=True`。既有A8W4 fixture的TB/s期望字符串在修改前后同一处失败，本项未改该无关fixture，不能声称所有旧fixture通过。

修parser后使用新队列`run_unified_gemm1_dual_perf_v2.py`，复用两组已经完整合格的旧前控制，仅重测共用两组和后控制，不重启失败队列或拼接轮次。最终六组18轮全部rc0、单次120秒内、完整三轮ownership合格，无污染重试；每轮count19，原warmup5/iters20/testGraph=False、真实random non-balanced routes和原计时保持。下表均为完整MoE profiler，单位μs：

| workload | 路径/顺序 | GEMM1三轮 | GEMM1中值 | GEMM2中值 | MoE e2e中值 |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 旧前控制 | 107.756 /108.005 /106.837 | 107.756 | 94.0873 | 256.972 |
| E64/topk8/shared0 | 旧前控制 | 98.6815 /99.0062 /99.9992 | 99.0062 | 62.0034 | 213.938 |
| E65/topk9/shared1 | 共用v1 | 107.695 /110.356 /108.741 | 108.741 | 94.4689 | 260.549 |
| E64/topk8/shared0 | 共用v1 | 97.1233 /95.9637 /95.1162 | 95.9637 | 60.9281 | 209.023 |
| E64/topk8/shared0 | 旧后控制 | 99.5356 /98.7535 /99.5166 | 99.5166 | 61.3710 | 212.554 |
| E65/topk9/shared1 | 旧后控制 | 107.046 /107.035 /107.948 | 107.046 | 95.2258 | 258.396 |

E64共用v1的GEMM1相对两侧旧版控制减少3.073%/3.570%，e2e减少2.297%/1.661%；E65 GEMM1增加0.914%/1.583%，e2e增加1.392%/0.833%。共用版没有刷新E65最优性能，后续优化保持同一个compiled object要求。生产默认在队列结束后自动恢复旧manifest；不能把已通过的opt-in共用候选写成默认已集成。

原始证据包`unified_gemm1_raw_evidence_v1.tar.gz`为1417874 B、195个regular members，SHA256=`d914268c0ce3eb1c800d4bb092afaf59cd1783126cb5dd7ddf5dc2040fca0895`。包含实际编译、完整精度、18个正式轮及3个parser-missing轮、source transitions和工具源码，排除大tensor文件。下载至独立目录`unified_gemm1_raw_download_v1/`，不覆盖alias receipts；`unified_gemm1_raw_downloaded_acceptance_v1.json`逐member SHA及完整数据验收通过。

在共用ABI上另行检查LDS operand load lowering：mode0原UniversalCopy、mode1 raw LLVM b128、mode2 raw LLVM b128+b32。地址/fragment/对齐32768个CPU案例一致，每个mode按E65→E64→E65证明同compiled object。三版均4547 instructions、VGPR486/SGPR70/LDS282624 B、private/spills0，实际`.text`逐字节相同，29568 B、SHA256=`a30d2597b108154a229ec6cd3fc5cd015695039f53815b4ceb59bbfea6809595`。ELF差异来自名称，因此停止此方向，没有运行GPU正确性或性能，不记录虚构收益。receipt=`unified_raw_lds_native_rejection_v1.json`。

随后仅把共用selector0的expert lookup固定步数8→7，positive Constexpr旧count路径保持。真实lookup AST的37345个CPU案例覆盖2–65 experts、12种routing pattern、空expert/shared1536/capacity边界，所有有效expert/row bound一致，每次map访问in bounds；其余kernel AST逐项不变。独立`unified_lookup7_v1_source/` persistent SHA=`29cfe04e826aa2617c4c1f57cb4fd0a0211f615ca14715a5bba227b35cda7d64`，wrapper与共用v1相同。

a07-3/dsr1_ep4无dispatch编译E65→E64→E65仍复用一个cache entry、cache key、compiled object和IR。lookup7 ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`，4531条指令，比lookup8少16条；VGPR486、SGPR68、LDS282624 B、private/spills0、kernarg180 B。差异全部落在一次scalar map读取及其索引/比较/select/wait/delay；数值、LDS、TDM和全部既有同步指令数量保持，八steady backedges/四targets、28/32 WMMA区域及最终tensorwait0验收通过。receipt=`unified_lookup7_compile_acceptance_v1.json`。

lookup7随后在同一实际GPU进程分别完成seed0/17的E65→E64完整MoE，每规模每seed28项，共112项零字节差异；eager/graph canary、三组dynamic A/ScaleA/psum replay及最终MoE均通过，原reference阈值不变，实际loaded ELF为上述一个共用ELF。两项进程均rc0、未超120秒，live默认未改。正式新队列`run_unified_lookup7_dual_perf_v1.py`固定lookup8 E65/E64前控制→lookup7 E65/E64→lookup8 E64/E65后控制，各三轮，已经完成。27轮全部rc0、未超时；lookup7 E65、lookup7 E64与lookup8 E64后控制首组因ownership各整组排除一次，六组最终完整合格共18轮，不从污染组拼接：

| workload | lookup/顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 8步前控制 | 108.357 /107.781 /111.625 | 108.357 | 94.2326 | 258.611 |
| E64/topk8/shared0 | 8步前控制 | 95.1247 /96.5415 /96.7206 | 96.5415 | 61.1771 | 209.329 |
| E65/topk9/shared1 | 7步候选 | 107.752 /106.828 /108.273 | 107.752 | 94.3317 | 258.016 |
| E64/topk8/shared0 | 7步候选 | 96.3286 /95.7001 /96.9841 | 96.3286 | 60.0702 | 208.956 |
| E64/topk8/shared0 | 8步后控制 | 97.6505 /96.1889 /97.4965 | 97.4965 | 61.8875 | 212.182 |
| E65/topk9/shared1 | 8步后控制 | 110.402 /107.191 /108.907 | 108.907 | 94.0746 | 260.059 |

本批E65 GEMM1低0.558%/1.061%，E64低0.221%/1.198%；e2e分别低0.230%/0.786%与0.178%/1.520%。这是小幅同批改善，尚非稳定反向确认或新的默认绝对纪录，距80 μs仍27.752 μs。源切换和最终恢复门禁均整机idle，最后KFD空、busy0、VRAM176582656 B、boot不变。初版raw打包工具误引用仅本地alias的`unified_gemm1_summary_v1.json`，在GPU队列已经完成后失败，未运行后续compile；新v2工具改读原canonical`compile_a07_unified_gemm1_v1/summary.json`，未重跑GPU。

完整包`unified_lookup7_raw_evidence_v2.tar.gz`为1685814 B、234个regular members，SHA256=`5d71c760030711f77a91c7055db61512f1bb9b63912c5379b1762af59a3cc83e`；独立`unified_lookup7_raw_download_v2/`逐member验收通过，包含27轮及所有排除记录、同ELF完整精度、native和source transitions。receipt=`unified_lookup7_raw_downloaded_acceptance_v2.json`。

下一项`unified_mgroup4_v1_source/`只允许共同selector0的M swizzle group=4，保留lookup8和所有numerical/operand/output/sync/GEMM2 AST，不重启旧group1/16/32实验。CPU直接运行实际swizzle/M-order AST，590种E64/E65拓扑、2211776个M/N排列点，每个capacity/live tile恰好一次，N cluster peers同M、capacity-tail排列保持。persistent SHA=`a4885f154fd363919b0c1dd714475799c20de79b66c2d327fe6158d065afe341`，wrapper仍共用v1。actual编译4547 instructions、VGPR486/SGPR70/LDS282624 B、private/spills0、kernarg180 B，ELF=`1126e0ad6f67b0f45a71f21c0ab851e62dc0fe9ece6a1e6d600dee6a3f8854a3`；E65→E64→E65仍同compiled object，全部opcode counts及八steady backedges的数值/同步验收通过。双seed双规模完整112项零字节差异已通过，实际同ELF绑定。性能队列已经停止：group8前控制E65为108.756 μs、E64为95.1944 μs，group4 E65三轮107.181 /107.414 /108.758 μs、中值107.414 μs；group4 E64连续三个整组均ownership不合格，达到事先设定上限，未运行后控制。没有GPU执行失败或超时，finally恢复旧源码，末次整机idle。group4未采纳，107.414 μs不作为新最优，不能根据未完成对照推断cache收益。

两pair A提前预读候选`unified_pairlead2_v2_source/`保留共用lookup8/group8，两套rmem槽及所有input READY/release/fences。CPU修订版覆盖1040个pipeline case、58240次K128更新、261632次M16 accumulator更新和727552次operand read；有效accumulator均保持56次原序更新。最初CPU草稿把GEMM2的同名nested function计入GEMM1审计而assert失败，未上传；v2限制在GEMM1后通过。实际no-dispatch编译E65→E64→E65同对象，ELF=`ba0f3bae272dc91bd024ca7e18e3643ff820a3f644546c9463d005b9ce4d293f`，4916条指令，比共用lookup8多369条；VGPR486/SGPR70、LDS282624 B、private12 B、VGPR spill4，静态各4条`scratch_store_b32`/`scratch_load_b32`。资源门禁拒绝，未做GPU精度或性能，未接入默认。编译rc0/未超时后，host收尾脚本误引用不存在的`task_m_group_default_manifest_v2.json`，该独立脚本失败保留，不重启队列。native拒绝receipt=`unified_pairlead2_native_rejection_v3.json`；关于VGPR分配和scratch Load/Store的硬件依据为Shader Programming Guide #65 §3.3.2.1、§2.4，未将spill读数当作kernel性能或唯一硬件瓶颈。

group4全部raw/ownership排除、双seed精度、native/source与两pair编译/host错误合并归档为`unified_stopped_raw_evidence_v1.tar.gz`，1728175 B、227个regular members，SHA256=`53cfb15b0598e5b9cb9c3101fa4bf22559684c477a6bc17cc145246ea257e4cf`。独立`unified_stopped_raw_download_v1/`逐member验收通过，receipt=`unified_stopped_raw_downloaded_acceptance_v1.json`；没有包含tensor大文件或另一OPTX工作链。

本轮默认接入准备为`unified_default_v2_source/`：目标两个capacity bucket共同启用runtime count，既有N tasks/rowtail/ScaleA/A预读/M-order控制若偏离共用配置则保留旧参数路径，显式`AITER_TDM_B_TH`也保留。其他expert数/capacity、balanced、t192、direct-scale原scope保持。实际dispatch AST的6项共用配置与76项legacy参数对照通过，receipt=`unified_default_dispatch_acceptance_v2.json`；后者证明参数与原逻辑相同，不把原本不合法的override组合称为新增支持。kernel bytes沿用验收版本。v1准备工具在末尾对AST list调用`ast.dump`失败，保留未上传草稿；v2用`ast.Module`修正后通过。首版check driver生成器引号错误尚未执行，v3修正并对输出全部AST检查通过。无UNIFIED开关的独立snapshot双seed双规模112项通过，随后远端正常生产imports/默认dispatch同样完成seed0/17共112项、实际cache1/同对象/同ELF；正式三轮性能进行中，未提前宣称性能验收完成。

固定一次反向`lookup7→lookup8→lookup7`已完成，共24轮均rc0/未超120秒，lookup8 E65前两个整组三轮因ownership排除，六个完整合格组共18正式轮。原计时/warmup5/iters20/随机数据和reference门限保持：

| workload | lookup/顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 7步前控制 | 108.960 /107.250 /108.884 | 108.884 | 95.0444 | 260.784 |
| E64/topk8/shared0 | 7步前控制 | 95.6390 /96.3286 /94.7072 | 95.6390 | 60.9723 | 208.003 |
| E65/topk9/shared1 | 8步中间对照 | 107.851 /108.155 /108.659 | 108.155 | 94.8378 | 258.898 |
| E64/topk8/shared0 | 8步中间对照 | 97.7727 /96.0271 /95.5273 | 96.0271 | 60.7088 | 211.387 |
| E64/topk8/shared0 | 7步后控制 | 96.1156 /95.0213 /95.0338 | 95.0338 | 60.6815 | 208.675 |
| E65/topk9/shared1 | 7步后控制 | 107.640 /108.271 /107.425 | 107.640 | 94.5659 | 258.362 |

本批lookup8 E65相对两侧lookup7为-0.670%/+0.478%，e2e为-0.723%/+0.207%，因此lookup7的E65收益未稳定反向复现。E64 lookup8相对两侧慢0.406%/1.045%，e2e慢1.627%/1.300%。采用lookup7作为默认共用候选的依据为减少16条scalar指令、原精度/同步不变及E64两侧收益，不能把其107.640 μs后控制中值称为E65新纪录或把多个批次最快中值拼成默认值。完整raw=`unified_lookup_reverse_raw_evidence_v2.tar.gz`，1260093 B、162 regular members、SHA256=`95f1dcc8b5242e0408244f313239a0802ea31842e0c9d2051957c51c5b503449`；独立逐member、原profile表数值及ownership前后验收通过，receipt=`unified_lookup_reverse_raw_downloaded_acceptance_v2.json`。

### 共用GEMM1默认接入完成及正式性能（2026-10-19）

`run_unified_default_live_v2.py 7`已完成。normal production imports不替换module、不传UNIFIED环境开关，先各seed运行E65→E64完整MoE，cache始终1项、同compiled object、同一actual ELF。每seed56项零字节，共112项，包括四项HEAD有效输出、eager/graph canary和dynamic input/psum；snapshot事前验收另112项。所有GPU程序rc0、未超120秒。对其他scope和显式旧配置的76项实际dispatch AST对照参数保持；这并不把旧不合法组合升级为支持。正式性能仍使用用户原入口和warmup5/iters20/testGraph=False，9轮均rc0/未超时；E65首组三轮因ownership不合格整组排除，第二组与E64第一组三轮合格，没有拼接。

| workload | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|---:|
| E65/T1536/topk9/shared1 | 107.172 /107.379 /108.015 | **107.379** | **94.3843** | **256.893** |
| E64/T1536/topk8/shared0 | 94.9283 /95.3059 /95.7361 | **95.3059** | **60.4664** | **209.078** |

本地和a07-3最终live源码：wrapper SHA=`798a865d7c11a0914b3c335d141948cbf77384d12135e822e339487b474e60b1`，persistent SHA=`29cfe04e826aa2617c4c1f57cb4fd0a0211f615ca14715a5bba227b35cda7d64`。native VGPR486/SGPR68、LDS282624 B、private/spills0、4531条；kernarg新增真实expert count at offset176/size4。原GEMM2、`quant_utils.py`、`gemm_common_gfx1250.py`和MoE routing源码与HEAD保持。benchmark仅动态名称解析及对应fixture变更，计时、输入、计算和reference AST一致；原fixture既有TB/s预期失败已独立标记，未为通过fixture改数据。最终整机KFD空、busy0、VRAM176582656 B、boot不变；未commit/push。

完整包`unified_default_raw_evidence_v2.tar.gz`，986792 B、128 regular members、SHA256=`f22c35958091355830a9c1e3f1d339994027daa4498866988ea05090dd9bb1a2`。独立逐member、actual ELF/对象绑定、原profile行和全部ownership记录验收通过，receipt=`unified_default_raw_downloaded_acceptance_v2.json`。本地CPU工具`my_code/audit_dsr1_e65_dispatch.py`已跟随默认更新并通过；`my_code/audit_dsr1_e65_source.py`检查保留的benchmark AST与原GEMM2/helpers均通过。

复现仍用本文用户原E65命令，不加候选开关。E64只替换`--experts 64 --topk 8 --shared-experts 0`。host包裹执行：

```bash
python3 -u my_code/run_dsr1_e65_gemm1_host.py <new_e65_label> --runs 3
python3 -u my_code/run_dsr1_e65_gemm1_host.py <new_e64_label> \
  --runs 3 --experts 64 --topk 8 --shared-experts 0
```

E65距80 μs仍27.379 μs。前述group8 ATT对应旧`7c643f...`，可支持方向分析，不能冒充新ELF的ATT。旧steady B/ScaleB owner tensorwait2 P50=3 cycles/每wave累计72，完整896 WMMA wave compute约22177 cycles；长WMMA gap包含DS/row predicate/address/WG必需工作。当前没有证据把差距唯一归因HBM、FIFO、cache或某一硬件单元。两pair lookahead被spill拒绝，是具体资源瓶颈证据，不能直接换算kernelμs。

下一项有限组合为`unified_pairlead_b1_v2_source/`，默认保持lookahead1/B1reuse0；候选lookahead2/B1reuse1，在下一B1 load前等待所有当前K128 consumer用完共享B1，A/B0/scales仍双槽。CPU实际WMMA validator覆盖四组合、2080项pipeline、116480次K128 steps、1046528次WMMA，当前A/B/scales和hint、有效acc56次原序、load多重集和TDM/hardware fence事件一致。v1草稿全文件替换撞到GEMM2同名注释，在生成snapshot前assert；v2限制GEMM1区域，所有GEMM2/math/output AST不变后通过。该候选随后完成native/device/正式性能，两个规模回退，已淘汰；详见下一节，当前已验收默认不改。

### 两pair A预读与B1复用：精度通过，正式性能回退（2026-10-19）

本项在当前共用lookup7/group8源码上组合lookahead2与B1共享。candidate wrapper SHA=`4aeae83023ba9ee66d3523afbf740128177431836c8fb27c9bd0eac850bfb20b`，persistent SHA=`0816a3c2a69eb633cbcef318cc48cdc7107392656118d28dd67999fd19d24202`。原两pair预读初版spill拒绝与旧B1控制fault的队列保持关闭，本项是新的有限组合，没有重启旧失败队列。

两项no-dispatch编译均rc0、未超过120秒，E65→E64→E65每mode复用一个compiled object。关闭候选完整ELF与当前默认`9a12287...`逐字节一致、4531 instructions/VGPR486/SGPR68；组合ELF=`db2302f78b9cca3ecc0394a319bdf8eed19ab69ab35ae1868154feffe341f767`、4936 instructions/VGPR456/SGPR68。两者LDS282624 B、private与spills均0、kernarg180 B、真实expert count offset176/size4保持。组合多405条静态指令：`s_set_vgpr_msb`+82，`s_and_b32/s_cbranch_scc1/s_cmp_lg_u32/s_cselect_b32`各+60，`s_delay_alu`+56，`s_wait_dscnt`+30，`v_nop`-3。WMMA、DS b128、TDM、量化、output stores及既有barrier数量保持；降低VGPR未证明occupancy或性能改善。

native验收正确保留objdump physical VGPR annotations，B1为v[272:287]，CFG覆盖7039个状态、52处DS写入、104处WMMA消费者，所有探索路径的共享B1 DS readiness通过。实际ELF只有入口WAVE_MODE设置，没有写SCHED_MODE，normal DEP_MODE interlocks保留；不把源码compiler hint或其他mode2实验当成本项进入了expert mode的证据。八条K backedges/四targets、28/32 WMMA区域、tensorwait2→DS reader release→WG signal/wait→refill、payload/ScaleA DS/WG fencing和最终tensorwait0保持。硬件依据为Shader Guide #65 §4.3.7.2、§4.3.7.4、§5.3.1.3；§4.6.12.1的WMMA/VALU hazard表不能代替DS dependency检查。receipt=`unified_pairlead_b1_compile_acceptance_v2.json`；此审计范围不等于穷尽所有hardware hazards。

snapshot seed0/17各在一个实际GPU进程运行E65→E64，共112项零字节差异，完整MoE/reference、eager/graph canary、dynamic A/ScaleA/psum更新均通过。之后live normal imports额外seed0双规模56项通过。实际loaded ELF精确匹配上述组合ELF，cache1/同compiled object；精度门限、真实routing/weights和有效routes保持。全部correctness-only耗时不记为性能，验证脚本的旧结束banner原文保留，JSON和actual ELF是本项绑定依据。

正式队列=`run_unified_pairlead_b1_dual_perf_v2.py`，固定默认E65/E64→组合E65/E64→默认E64/E65，每组完整三轮，仅ownership污染允许整组最多三次。27轮全部rc0、未超过120秒；组合E64第一组、默认E64后控制第一/二组因ownership整组排除，18轮正式合格。没有计时中监控、传输、编译或ATT，没有按耗时重试或拼接轮次。

| workload | 路径/顺序 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 当前默认前控制 | 107.155 /107.853 /107.666 | 107.666 | 94.2875 | 257.767 |
| E64/topk8/shared0 | 当前默认前控制 | 96.5540 /95.3839 /94.8906 | 95.3839 | 60.3779 | 208.747 |
| E65/topk9/shared1 | lookahead2/B1reuse1 | 119.000 /118.901 /122.060 | 119.000 | 94.3719 | 269.121 |
| E64/topk8/shared0 | lookahead2/B1reuse1 | 106.255 /111.389 /107.379 | 107.379 | 63.7766 | 223.375 |
| E64/topk8/shared0 | 当前默认后控制 | 96.4782 /97.0412 /95.8055 | 96.4782 | 60.3799 | 209.358 |
| E65/topk9/shared1 | 当前默认后控制 | 107.977 /107.356 /108.768 | 107.977 | 93.0712 | 256.201 |

组合E65 GEMM1相对两侧慢10.527%/10.209%，e2e慢4.405%/5.043%；E64 GEMM1慢12.576%/11.299%，e2e慢7.008%/6.695%。GEMM2源码不改，其读数仍按同轮记录，不把其差异归为GEMM2优化。候选淘汰，不接入本地live或默认；远端finally在全机idle/SHA守卫下恢复当前共用lookup7，最终KFD空、busy0、VRAM176582656 B。原正式默认E65 107.379 μs与E64 95.3059 μs保持，80 μs仍未达。

重复predicate、VGPR MSB设置与DS waits是具体native开销证据，但静态计数无法唯一归因此次回退或换算为kernel μs。完整raw包为`unified_pairlead_b1_raw_evidence_v2.tar.gz`，1858814 B、237 regular members、SHA256=`8816cb4e4a12c8b17a81654e239ca8ac44e02c1786e219d5208a11d269489080`；独立下载后逐member、27轮原profile行、source/actual ELF及ownership验收通过，receipt=`unified_pairlead_b1_raw_downloaded_acceptance_v2.json`。当前共用默认ELF的SIMD0/1、四SE已通过原`get_att_b_rocm_sdk.sh`采集及独立raw验收，详见下节；旧`7c643f...`ATT继续保留为旧版证据。

### 当前共用默认的 SIMD0/1 ATT 与局部M32条件候选（2026-10-19）

原脚本诊断命令为当前完整random workload、iters2，分别`--simd 0 --ana-att`和`--simd 1 --ana-att`，每项120秒限制。两项首次采集均rc0、无超时、ownership合格、无live post owners，分别四SE。actual code_object_id22与当前共用默认完整ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`一致。UI分别为`e65_g1_unified_default_simd0_att_v3_try1_20261019/thread_trace/simd0/kernel/rpf_v3/ui_output_agent_41929_dispatch_58`及`e65_g1_unified_default_simd1_att_v3_try1_20261019/thread_trace/simd1/kernel/rpf_v3/ui_output_agent_49397_dispatch_58`；每项60个sampled waves，24个无WMMA。

| SIMD | WMMA数 | 有效sampled waves | compute span中值 cycles | WMMA issue gap中值 cycles |
|---:|---:|---:|---:|---:|
| 0 | 448 | 6 | 20346.0 | 8 |
| 0 | 672 | 10 | 21636.5 | 8 |
| 0 | 896 | 20 | 21856.0 | 8 |
| 1 | 448 | 8 | 20743.5 | 8 |
| 1 | 672 | 9 | 21429.0 | 8 |
| 1 | 896 | 19 | 21665.0 | 8 |

全compute的`s_wait_tensorcnt 0x2`每wave累计中值为75 cycles，不能称单条wait中值。独立下载验收后，`code.json`字段5逐条匹配actual native byte PC与opcode，再沿用既有owner-loop analyzer。随后已完成新ELF逐owner descriptor寄存器来源绑定：source `add_tdm_loads` 的 `gA/gB/gSA/gSB`、actual kernarg pointer offset及native pointer加法链逐项核对，不凭loop顺序指定角色。receipt=`unified_default_owner_descriptor_provenance_v3.json`，详细分析=`unified_default_simd{0,1}_owner_waits_v3.json`。

| owner | kernarg pointer offset B | steady native loop bounds | refill PC | 采样 |
|---|---:|---|---:|---|
| A | 40 | `[12244,13664]` | 13040 | SIMD0，18 waves |
| B | 48 | `[14340,15760]` | 15136 | SIMD1，18 waves |
| ScaleA | 56 | `[15908,17328]` | 16704 | SIMD0，18 waves |
| ScaleB | 96 | `[17460,18880]` | 18256 | SIMD1，18 waves |

每个owner的steady `tensorwait2` per-wave累计中值均72 cycles；工具所报各wave单次wait中值的再中值为3 cycles，不冒充所有单条记录合并后的P50。两SIMD独立采集，仍不能相加中值或直接换算kernel μs。

采样中padding WMMA减半没有使wave compute span同比减少；896组约225–226个CSELECT/CBRANCH、1308条DS b128。这支持继续检查operand、条件和同步调度，不证明某个独占硬件瓶颈。两SIMD独立采集、采样waves并非全部WGs；独立中值不能相加，也不转换为kernel μs。raw归档`unified_default_att_raw_evidence_v3.tar.gz`为4604740 B、215 regular members、SHA256=`26544590df53017d6f39062beffebcdaea0014af6c78542a19aebf062c3daab3`，raw `.att`、decoded waves、PC/opcode、actual ELF和source/ownership记录独立下载验收通过，receipt=`unified_default_att_raw_downloaded_acceptance_v3.json`。

据此准备新的`unified_m32_local_v3_source/`，并非重启已关闭的旧E65-only性能队列。仍使用当前runtime count/lookup7/group8、两套独立A/B/scales rmem；将同一M32 pair的未来A loads与当前WMMA放入一个局部条件，以空LLVM asm `=s,0` tied identity返回真实`mn_oob`后进行原整数比较。它只限制compiler CSE，不是memory fence，不更改lane EXEC或有效行范围。默认关闭`AITER_GFX1250_GEMM1_M32_LOCAL=0`，仅当前共用scope可开启；其他旧参数路径保留。wrapper SHA=`61017103a9f534cd3a6dcad6bf7806647beb8c8b55510ef3359cd0cc6b737e00`、persistent SHA=`afbc7adb3e76af5fda31c90b58ca0e193469f593d355a6f04457119be0c7dbf0`。

实际AST条件审计2080个case、523264项M16更新/256960次未来pair loads通过；完整operand审计1040项pipeline、58240次K128 step、523264次WMMA，通过每个有效accumulator的56次原序及A/B/ScaleA/ScaleB、scale selectors/reuse hints、loads多重集和TDM/hardware fences对照。初版CPU audit将含list的hardware event直接交`Counter`而失败；改为完整event的`repr`比较后通过，修订时一次checker缩进错误同样保留，不涉及生产源码或GPU。原GEMM2、量化math、output/输入地址functions保持。

固定no-dispatch compile0→1均rc0、120秒内，E65→E64→E65每mode一个compiled object。关闭完整ELF精确当前默认；打开ELF=`18d81298f90b9c8ba5900ba9a583971b48d10486166988cec0cdd2d82d1d0e8f`，4531→4408 instructions，减少123条，VGPR486/SGPR68、LDS282624 B、zero spills/private及kernarg180 B保持。AND/CSELECT/CMP_LG各少64、CMP_GE/MOV各多64、delay少59，原WMMA/DS/TDM/数值/barrier opcode counts保持。native八backedges/四targets及28/32 WMMA区间、input READY/release、output DS/WG/TDM、两条exit tensorwait0、actual count offset176/size4通过。编译queue的末尾banner继承`V2`原文保留；新目录、JSON/source及actual ELF绑定使用v3，未将banner作为验收依据。

snapshot seed0/17双规模共112项零字节差异，HEAD有效payload/ScaleA/GEMM2/MoE、eager/graph canary及dynamic A/ScaleA/psum replay通过；实际E64/E65 cache1/同对象/同ELF。两项均rc0、未超时，correctness-only耗时全部无效。live normal imports另seed0双规模56项通过。receipt=`unified_m32_local_correctness_acceptance_v3.json`。固定完整三轮性能顺序为0 E65/E64→1 E65/E64→0 E64/E65，反向固定为1 E65/E64→0 E64/E65→1 E65/E64。只有ownership污染可整组最多三次，执行/精度/解析失败停止；本轮36轮全部首次rc0、未超时、ownership合格，没有重试、拼接或按时间筛选，计时中不并发传输、编译、ATT或GPU监控。

| workload | 固定组/路径 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 正向默认前控制 | 108.762 /107.132 /107.623 | 107.623 | 94.0428 | 258.944 |
| E64/topk8/shared0 | 正向默认前控制 | 95.7950 /95.7717 /96.0120 | 95.7950 | 60.3968 | 208.835 |
| E65/topk9/shared1 | 正向局部M32候选 | 107.505 /106.700 /107.729 | 107.505 | 94.4519 | 258.044 |
| E64/topk8/shared0 | 正向局部M32候选 | 94.9938 /94.8695 /96.6068 | 94.9938 | 60.2660 | 208.108 |
| E64/topk8/shared0 | 正向默认后控制 | 98.9681 /96.6426 /94.3823 | 96.6426 | 61.0862 | 211.452 |
| E65/topk9/shared1 | 正向默认后控制 | 107.967 /108.289 /107.501 | 107.967 | 94.7641 | 257.970 |
| E65/topk9/shared1 | 反向局部M32前组 | 107.410 /108.526 /107.703 | 107.703 | 94.7176 | 258.733 |
| E64/topk8/shared0 | 反向局部M32前组 | 95.3037 /93.3892 /95.1752 | 95.1752 | 59.9244 | 207.307 |
| E64/topk8/shared0 | 反向默认中间控制 | 94.9199 /96.2757 /96.2398 | 96.2398 | 60.5612 | 208.918 |
| E65/topk9/shared1 | 反向默认中间控制 | 107.377 /107.826 /107.497 | 107.497 | 94.6988 | 258.617 |
| E65/topk9/shared1 | 反向局部M32后组 | 108.325 /107.503 /108.772 | 108.325 | 95.2364 | 259.271 |
| E64/topk8/shared0 | 反向局部M32后组 | 95.3945 /95.5675 /96.5244 | 95.5675 | 61.5312 | 208.667 |

正向E65 GEMM1相对两侧快0.109642%/0.427909%，e2e为-0.347565%/+0.028686%；反向两侧候选相对中间默认GEMM1慢0.191633%/0.770254%，e2e慢0.044854%/0.252884%。E65稳定收益未复现，因此候选淘汰，不继续按耗时追加组、不接入默认。E64正向快0.836369%/1.706080%、反向快1.106195%/0.698568%，仍不能按E64另设default，必须遵守两规模共用一个compiled GEMM1的要求。默认E65 107.379 μs、E64 95.3059 μs保持，80 μs目标尚未达到。

raw归档=`unified_m32_local_raw_evidence_v3.tar.gz`，2226344 B、313 regular members、SHA256=`4413f900b52b7d23f87c20ba44094963feb382cbd5b84f053e33d6b6c4c37373`。逐member、36轮原profile数据、双seed/live精度、source/actual ELF、ownership及新owner provenance独立下载验收通过，receipt=`unified_m32_local_raw_downloaded_acceptance_v3.json`。归档初版漏传两个仅在本地产生的ATT工具/receipt而两次报`FileNotFoundError`，在正式计时结束后补传、确认文件齐备再成功打包；保留两个失败log，不影响上述GPU执行或性能记录。反向finally已恢复精确共用默认，post全机idle、KFD空、VRAM176582656 B，无本任务GPU残留。

### 共用runtime ABI的K loop外all-live M32候选（2026-10-19）

新的有限snapshot为`unified_m32_split_v2_source/`，默认关闭`AITER_GFX1250_GEMM1_ALL_LIVE_M32_SPLIT=0`。仅当前共同selector0 scope可开启：在K pipeline前按真实`wmb+(wmma_m_rep-2)*WMMA_M < mn_oob`选择同ELF内的两条wave-uniform路径。条件满足时所有原M32 pairs都会执行，可移去每K128的重复M32 compute/A-load判断；否则保留原条件。末pair部分有效仍按原first-row predicate执行全部lanes，输出只写有效行。不是按64/65编译两套specialization，不修改实际routing、有效routes、56次K128更新及顺序、双套A/B/scales、SiLU/BF16/FP4/E8M0、canonical output及TDM/WG fences。此前E65-only旧tail split队列已因三组ownership污染关闭；本项是新的共用ABI候选，不重启旧队列或沿用其无效性能。

CPU实际AST覆盖260个row边界/两个M128 halves，58240次K128 steps、523264次WMMA。两条路径的所有有效accumulator 56次原序、A/B/ScaleA/ScaleB与selector/hint、loads多重集、TDM/hardware fence events通过；关闭run_task完整AST等于当前共用默认，35个nested math/address/output/sync functions、GEMM2/helpers保持。两份receipt=`unified_m32_split_scope_audit_v2.json`及`unified_m32_split_operand_audit_v2.json`。wrapper SHA=`54249ea181df16a8ff2fede3e1c10f478253eb49703eeeb83d79c4fa6acba166`、persistent SHA=`b41f531e850c4ae328e532fa262552865eb63b6655f13b58b54753afe4f3f167`。

两项无dispatch编译均rc0、120秒内，各mode按E65→E64→E65复用cache1/同对象/同IR。关闭ELF完整等于当前`9a12287...`；候选ELF=`55dc98003818a4b25c96584a75793059e0c932f92e2eeaefb3f1d45f2e6cf595`，6298 instructions、VGPR458/SGPR78、LDS282624 B、zero spills/private、kernarg180 B。静态指令增加是同ELF内互斥pipeline的代码体积，不表示runtime重复执行全部计算。actual CFG的12条backedges/8个targets、20/24/32 WMMA regions、input READY与reader release、output DS/WG/TDM及最终tensorwait0均通过；native验收不等同于穷尽hardware hazards。

seed0/17每项在同一实际进程E65→E64，共112项零字节差异；HEAD有效payload/ScaleA/GEMM2/MoE、eager/graph canary及dynamic A/ScaleA/psum replay全部通过，原reference门限保持。live正常imports另seed0双规模56项通过，actual两规模cache1/同对象/同完整candidate ELF。两项snapshot与一项live均rc0、未超120秒；correctness-only耗时全部无效。receipts为`unified_m32_split_compile_acceptance_v2.json`及`unified_m32_split_correctness_acceptance_v2.json`。

固定0 E65/E64→1 E65/E64→0 E64/E65的完整原命令性能已经结束；每组原三轮、warmup5/iters20/eager、120秒上限。只因ownership污染允许整组最多三次尝试；E64候选try1的第3轮不合格，仅因此重跑完整try2，其余六个合格组均第一次完成。全部21轮rc0、未超时，无execution failure；没有按耗时筛选或拼接。计时中不并发本任务传输、编译、ATT或GPU监控。

| workload | 固定组/路径 | GEMM1三轮 μs | GEMM1合格中值 μs | GEMM2合格中值 μs | MoE e2e合格中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前控制 | 108.347 /111.117 /110.265 | 110.265 | 95.1627 | 261.240 |
| E64/topk8/shared0 | 默认前控制 | 96.4613 /96.0753 /95.6685 | 96.0753 | 59.3531 | 208.154 |
| E65/topk9/shared1 | K loop外all-live候选 | 109.928 /112.137 /111.364 | 111.364 | 95.0298 | 262.208 |
| E64/topk8/shared0 | 候选try1，ownership污染整组排除 | 98.1862 /96.7227 /97.9057 | — | — | — |
| E64/topk8/shared0 | 候选try2 | 96.6259 /98.7997 /97.9055 | 97.9055 | 61.8917 | 213.269 |
| E64/topk8/shared0 | 默认后控制 | 95.0381 /94.7915 /97.4967 | 95.0381 | 60.7216 | 209.179 |
| E65/topk9/shared1 | 默认后控制 | 107.248 /107.353 /108.964 | 107.353 | 94.7535 | 257.202 |

候选E65 GEMM1比两侧默认慢0.996690%/3.736272%，e2e慢0.370540%/1.946330%；E64 GEMM1慢1.904964%/3.017106%，e2e慢2.457315%/1.955263%。本项淘汰，不启动反向确认、不追加计时、不接入默认。VGPR下降没有转化为性能收益，不能据该对照唯一归因instruction cache、某个FIFO或cache环节。两端精确共用默认保持，正式E65 107.379 μs、E64 95.3059 μs不被本批控制中值替换。

raw包`unified_m32_split_raw_evidence_v2.tar.gz`为1700476 B、213 regular members、SHA256=`066e21cea6db1bd5c041fb11a5a0295775066af7ea2ae8f4a5a0356276034d1c`。逐member、21轮原profile/完整command/source/ownership、112+56项精度及actual ELF独立下载验收通过，receipt=`unified_m32_split_raw_downloaded_acceptance_v2.json`。队列finally恢复默认，post全机idle、KFD空、VRAM176582656 B，无本轮GPU进程残留。复现候选是在原完整命令前增加`AITER_GFX1250_GEMM1_ALL_LIVE_M32_SPLIT=1`并使用对应snapshot；关闭开关等于当前默认。闭合队列及原labels不重启。

### 完整M swizzle group使用固定除数，尾组保留runtime尺寸（2026-10-19）

新有限snapshot起点为`unified_runtime_swizzle_v1_source/`，开关`AITER_GFX1250_GEMM1_RUNTIME_SWIZZLE=0`默认关闭，仅共用selector0/group8的scope可开启。目标workload的N为4096，但launcher的`N: fx.Int32`实际上是runtime参数，早期将其当成constexpr常量的说明有误。完成修订的v4在runtime确认N4096且当前WG属于完整M group时，才用字面固定32/8除数；尾组及其他N完整保留原runtime映射。E64/E65没有新增expert-count或M-capacity编译specialization，真实count/routing与M-order保持。

初版实际AST对120项geometry、87216个WG映射点验收通过，其中81024个full-group点、6192个runtime-tail点；覆盖E64/112与E65/119个M tiles、整组/尾组及256行以内的capacity边界。所有M/N/local peer覆盖与默认逐点相同，无重复遗漏。关闭后两个完整module AST精确还原默认；全部nested compute/operand/address/output/sync functions及GEMM2 AST保持。初版wrapper SHA=`8faab89baa82ecbaa988b98cf947e4bf5afbfc7e42ffbd4d011c662adc076ff2`、persistent SHA=`b22c1d8a3119749cf8dce54fab9068aab1024846f8e3bf3242a033d5635b0050`。

v1关闭控制在120秒内未完成，rc124，候选未运行；另一个同限stage/stack诊断同样rc124，确认卡在module import的`@flyc.jit` AST重写，而非LLVM或GPU执行。新增13项长`and`被`flydsl/compiler/ast_rewriter.py:541–575 visit_BoolOp`反复引用已展开result，`ast.fix_missing_locations/iter_child_nodes`遍历量随条件数量放大。v2改为`all((...))`后关闭控制成功且完整默认ELF一致，但打开候选立即rc1：`RuntimeError: cannot evaluate dynamic 'Boolean' as Python bool during tracing`，来自runtime `N == 4096`。不能称该条件为纯constexpr。两版队列均依规定失败停止，没有延长120秒或记录性能。

v3将constexpr守卫拆成五项短assert，排除runtime N，并在mapping中增加`(i32_n == 4096) & (swz_id < full_blocks)`；字面4个N cluster units对应完整group的32/8除数，runtime尾组与其他N使用原公式。CPU扩展至N1024/2048/3072/4096/6144/8192、720项geometry、523296个WG点，其中81024个fast-path点、442272个fallback点，逐点相同且完整覆盖。各nested functions、两module关闭AST与GEMM2保持。v3工具生成漏改f-string label中的`_v2`，host runner因已有label拒绝启动，rc2；queue误引用旧result中的rc0，这个旧status不能作为新执行结果，v3没有新compile或GPU dispatch。另一次remote CPU审计因为已随输入包传入receipt而触发存在性assert；没有覆盖该receipt，后续工具只在完整重算record与已有receipt精确相等时允许通过。上述失败记录均保留。

v4仅修订工具版本和receipt复核方式，candidate源码与v3逐字节一致：wrapper SHA=`8faab89baa82ecbaa988b98cf947e4bf5afbfc7e42ffbd4d011c662adc076ff2`，persistent SHA=`a6d4a0e6c2ac12d3f7ee15a0bd263ee51a2f3b82801dc2d3ca62861ea5440372`。两端523296点CPU验收通过，输入包68836 B/14 regular members，SHA=`40446034c6291afa62433abc7241e8f167abf14699cb8485e34fe7e13bfdadbe`，逐member传输核验通过。各工具按AST literal处理包含`{mode}/{seed}/{attempt}`的JoinedStr并检查candidate版本，不改任何`unified_default_*_v2`输入。

v4无dispatch编译0→1均rc0、120秒内，E65→E64→E65各mode始终cache1/同compiled object/同IR。关闭完整ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`精确默认；候选ELF=`2d200a1b95699c265e26aa9c913d0ae0d9a28ec7dd142ef50ba5122d57d6b9f0`，4531→4576 instructions，VGPR486保持、SGPR68→70、LDS282624 B、private/spills0、kernarg180 B及count offset176/size4保持。八backedges/四targets、28/32 WMMA regions、READY/DS reader release/WG/refill、output DS/WG/TDM和最终tensorwait0检查通过，原WMMA、DS/TDM、数值及barrier opcode counts保持，normal DEP_MODE interlocks保留；native有限审计不等于穷尽hardware hazards。actual `.elf/.isa/.notes/.mlir/summary`已下载并核对完整ELF SHA。

v4 snapshot seed0/17每进程E65→E64，共112项零字节差异通过，包含HEAD有效payload/ScaleA/GEMM2/最终MoE、eager/graph canary和同地址dynamic A/ScaleA/psum replay，原reference门限保持；实际两规模同compiled object/同candidate ELF。两项均rc0、未超120秒，correctness-only耗时无效。receipt=`unified_runtime_swizzle_correctness_acceptance_v4.json`。固定0 E65/E64→1 E65/E64→0 E64/E65、各完整三轮的原benchmark队列已结束：21轮rc0、均在120秒内；E64后控制try1第1轮ownership污染，整组三轮排除，仅因此重跑try2。候选E65 GEMM1比两侧默认慢1.420381%/1.361137%，MoE e2e慢0.689807%/0.622460%，因此淘汰，不追加反向、不修改生产默认。

| workload | 固定组/路径 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前 | 107.788 /108.018 /107.689 | 107.788 | 94.5553 | 258.478 |
| E64/topk8/shared0 | 默认前 | 96.4234 /95.0507 /94.6672 | 95.0507 | 61.2255 | 208.106 |
| E65/topk9/shared1 | runtime swizzle候选 | 111.684 /109.319 /106.993 | 109.319 | 94.7429 | 260.261 |
| E64/topk8/shared0 | runtime swizzle候选 | 95.5187 /94.6735 /94.4731 | 94.6735 | 60.7279 | 208.637 |
| E64/topk8/shared0 | 默认后try1，整组排除 | 96.2928 /94.6901 /95.8311 | — | — | — |
| E64/topk8/shared0 | 默认后try2 | 96.6616 /94.9052 /96.5523 | 96.5523 | 59.1866 | 208.414 |
| E65/topk9/shared1 | 默认后 | 107.689 /107.851 /108.857 | 107.851 | 94.9076 | 258.651 |

完整summary=`unified_runtime_swizzle_dual_perf_summary_v4.json`；raw归档=`unified_runtime_swizzle_raw_evidence_v4.tar.gz`，2323354 B、306 regular members，SHA256=`7c72e73eff426d13f135f5d025f172bb2596db1820d7589faf652b5dc17f8936`。独立下载后逐member、21轮原profile、ownership、112+56项精度及actual source/ELF重算通过，receipt=`unified_runtime_swizzle_raw_downloaded_acceptance_v4.json`。host队列为`python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_runtime_swizzle_dual_perf_v4.py`；本次label已经存在，不重启旧队列。两端精确恢复共用默认，正式E65 107.379 μs、E64 95.3059 μs记录保持。

### 共用ABI的WMMA operand reuse：完整性能无稳定收益（2026-10-19）

本项从当前runtime count/lookup7/group8默认重新构建有限候选`unified_operand_reuse_v1_source/`。仅在同一M32 pair的even→odd边界增加matrix-A hint，原matrix-B hints保持；hint指向下一条WMMA、不跨predicate/K128末尾、首WMMA无hint。依据Shader Guide #65 §4.6及CDNA5 ISA §7.12，不从reuse字段单独推断性能瓶颈。wrapper SHA=`28370b5bc7665799e470e48e578b4632a9640da36877f06cd7777a6732ea49b8`，persistent SHA=`7078fdace8abe51731e1015661b9a2d6210ba3bbd79256cbd4e12d774e62e6cb`。

CPU actual AST审计522项row/wave边界、4832次operand calls通过。关闭完整module AST精确默认；native所有PC、顺序、operands和memory/fence编码保持，仅64个word0 bit13变化，原112个matrix-B hints保持。关闭完整ELF=`9a12287...`；候选完整ELF=`23a3bb556adbe5a3757d0b89541bf27902d93844f916fda499206f9eb18caad3`。两者均4531 instructions、VGPR486/SGPR68、LDS282624 B、zero spills/private、kernarg180 B及真实count offset176/size4。E65→E64→E65每mode cache1、同compiled object/IR。

seed0/17双规模112项零字节完整MoE、HEAD有效payload/ScaleA/GEMM2/MoE、eager/graph canary和同地址dynamic replay全部通过，live额外56项通过；实际E64/E65同compiled object/同candidate ELF。固定前默认→候选→后默认共24轮rc0、120秒内。E64后控制try1第3轮与E65后控制try1第2轮ownership污染，仅因此各整组三轮排除和重试，未按耗时选择。

| workload | 固定组/路径 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前 | 107.824 /107.474 /109.517 | 107.824 | 93.4336 | 257.789 |
| E64/topk8/shared0 | 默认前 | 96.0059 /96.0314 /95.1311 | 96.0059 | 60.3124 | 208.243 |
| E65/topk9/shared1 | operand reuse候选 | 110.158 /107.271 /107.933 | 107.933 | 94.4919 | 257.258 |
| E64/topk8/shared0 | operand reuse候选 | 95.9935 /94.2241 /96.8155 | 95.9935 | 59.6632 | 207.444 |
| E64/topk8/shared0 | 默认后try1，整组排除 | 96.9503 /96.0311 /96.6743 | — | — | — |
| E64/topk8/shared0 | 默认后try2 | 95.3881 /94.9601 /95.0383 | 95.0383 | 59.1592 | 206.684 |
| E65/topk9/shared1 | 默认后try1，整组排除 | 107.223 /108.186 /110.476 | — | — | — |
| E65/topk9/shared1 | 默认后try2 | 106.786 /110.092 /108.024 | 108.024 | 94.2898 | 257.841 |

E65 GEMM1相对前/后为+0.101091%/-0.084241%，E64为-0.012916%/+1.005068%；无稳定双规模收益，淘汰、不反向、不追加计时。summary=`unified_operand_reuse_dual_perf_summary_v1.json`，默认已恢复，post `2026-10-19T00:08:12.637643Z`整机idle、KFD空、VRAM176582656 B；该post属于本次结束时状态，不替代后续空闲重检。

raw归档=`unified_operand_reuse_raw_evidence_v1.tar.gz`，1748834 B、219 regular members，SHA256=`f97a6cfaab9234da0774224061fb5d93b5d60fe07b03c72bbb345dad46707324`。独立下载验收24轮原profile/ownership、中值、source/ELF与64hint bits通过，receipt=`unified_operand_reuse_raw_downloaded_acceptance_v1.json`。最初source preparation/audit失败JSON与原logs保留。raw checker首版误读失败JSON的`no_dispatch`字段，实际键为`remote_compile_or_dispatch=false`；修正后通过，失败单独保存在`unified_operand_reuse_raw_checker_failure_v1.json`，后建checker/失败JSON不冒充原archive members。

### LDS b64 payload lowering：编译资源拒绝，未dispatch（2026-10-19）

候选`unified_lds_b64_v1_source/`将原load_a/load_b每个b128拆成两个aligned b64，再按原dword顺序拼回，输入TDM布局、compute、输出、同步不改。wrapper SHA=`2afacd3864bf2fc2d4110bd79aa8b9238468d0a53807372164758a624c107375`，persistent SHA=`ec7bd51e5cea522072ca5c0a81d68f720e68121f166429a89472669076becc19`。Shader Guide #65 §5.7.6 Table86记载b64 independent repeat1、b128 repeat2，不能直接将该文档数字换算为kernel性能收益。

CPU actual AST审计20480 loader cases、786432B，覆盖四stage/八wave/所有lane/两K128，dword地址、顺序及load多重集完全相同，关闭完整module AST精确默认。首checker遗漏load_b内部`load_half`的changed subset allowlist，assert失败后只修checker，candidate源码不变，记录=`unified_lds_b64_audit_preparation_failure_v1.json`。输入包55732 B/11 members、SHA=`cf511682e942c14e2864309de8b8cc30f96ec7f7659ac4ff76633b05409e53cf`，远端逐member SHA核验通过。

两项compile均rc0、120秒内，每mode E65→E64→E65 cache1/同对象/同IR；关闭完整ELF精确默认。候选ELF=`cf30ce53c7cdef2af5966fc03cd59913a28e702d7ad8ed66947f029868f9c18d`，36936 B、4941 instructions、VGPR486/SGPR68/LDS282624 B、kernarg180 B，但VGPR spills5/private12 B。实际352条`ds_load_2addr_b64`、64条`ds_load_b64`，新增moves/地址/MSB切换，以及5条scratch stores/9条scratch loads。按zero-spill门槛淘汰，未GPU dispatch、未计时、未live部署。这是明确资源证据，不是实测性能回退。

两mode完整`.elf/.isa/.notes/.mlir/summary`已下载，独立native拒绝receipt=`unified_lds_b64_native_rejection_v1.json`，`selected_for_device=false`、`performance_data=false`。原compile队列已结束，不重跑。

### 共用ABI的MMA schedule group1：编译通过，完整验证fault（2026-10-19）

候选`unified_mma_group1_v1_source/`只把当前GEMM1的compiler hint `MMA_GROUP=2`改为`1`，原closing cover12、payload/scales loads、56次有序K128有效更新、quant/output、hardware fences和runtime count ABI保持；GEMM2及原benchmark不改。wrapper SHA=`962b9a6a7a3ed3503686b5a62891fc619238a59dde34832038e9d99ef53f704e`，persistent SHA=`26bbf07b85480ea1320237b0e70eb7f67e53e2e92d49e5a6566d73b5a3824165`。

两端CPU审计80个实际hint cases/2840 events，WMMA/DS/TDM总数和closing cover12保持，关闭完整module AST精确默认，所有nested numerical/address/load/output/fence函数及GEMM2 AST相同。host首次复核已有JSON receipt时，Python tuple与JSON list相比较产生assert；checker在比较前JSON规范化后重算通过，原日志/失败receipt保留，candidate源码不改。输入包63318 B/14 members、SHA=`af9f5572906a968a190d68db2a492ec3449d1e43c905a29ecb3cc49f88e812b0`，逐member传输通过；修订checker另行传输，不改写原input包。

两项no-dispatch compile均rc0、120秒内。关闭完整ELF精确默认；候选完整ELF=`4accaa4d94e7b7110248b9672b8b99a78d35fb2d9cbd96e02a6a9fc05274f694`，34384 B、4527 instructions，相对默认4531少4条`s_wait_dscnt`。VGPR486/SGPR68/LDS282624 B、zero spills/private、kernarg180/count176/4保持；实际E65→E64→E65每mode同compiled object/cache1。原WMMA/DS/TDM/数值/barrier counts保持，11个DScnt0、barrier/tensorwait顺序、八backedges/四targets/28–32 WMMA regions及output DS/WG/TDM/final tensorwait0通过。native有限审计=`unified_mma_group1_compile_acceptance_v1.json`。

新增实际physical VGPR DS completion检查覆盖所有CFG条件分支及K-loop backedges：默认4531条/17331次register访问、候选4527条/17320次访问，显式DS RAW/WAW unresolved conflicts均为空。依据CDNA5 ISA §5.7/§5.7.1.4的in-order LDS与`wait <= count`语义；这是instruction-issue和explicit operands模型，不能穷尽implicit EXEC skipping、ALU/WMMA multicycle或所有hardware hazards。首checker遗漏`global_prefetch_b8`地址read分类，修正后通过，失败receipt保留。验收=`unified_mma_group1_ds_read_audit_v1.json`。

`a07_unified_mma_group1_seed0_v1`在E65 eager/graph canary与三组dynamic replay中24项payload/ScaleA均零差异，但在后续原完整MoE reference阶段fault：`HSA_STATUS_ERROR_MEMORY_FAULT`、address=`0x6ba0000`，fault日志kernel为`Cijk_Alik_Bljk_...`，rc134、未超120秒。此时实际candidate ELF已绑定；完整HEAD四项及最终`[moe] pass`尚未完成，E64和seed17未执行，不能称为112项完整精度通过。结束存在外部stress/MLA GPU owners，未杀进程；不能据此唯一归因外部任务、GEMM1、BLAS、allocator或driver。

原精度队列按execution failure停止，未部署live、未启动性能、不重新运行旧队列。独立诊断固定默认/候选各一次，仅记录原reference `aten.mm` tensor地址与实际`hipblasLtMatmul`调用，不新增GPU同步，全部计时无效；它不能用来覆盖失败样本或自动解锁性能。主机复现=`python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/queue_unified_mma_group1_fault_diagnostic_v1.py`，每个容器程序120秒限制，空闲每6秒轮询，诊断结果另补。

#### group1固定故障诊断及证据收尾（2026-10-19）

固定mode0/default和mode1/candidate各seed0一次的异步reference/native日志诊断已完成，每mode均E65→E64，56项`differing_bytes=0`、两次原`[moe] pass=True`、eager/graph canary和三组dynamic A/ScaleA/psum replay通过。默认actual完整ELF=`9a12287...`，candidate=`4accaa4d...`；每mode双规模cache1/同compiled object/同完整ELF。每mode记录258项reference `aten.mm.default`与258项`hipblasLtMatmul`，指针逐项对应、全部native API返回status0、tensor shape/stride未超storage范围。本次未复现原fault，异步API status0不等于失败运行的fault PC或根因证据。

mode0的GPU完整验证和`acceptance.json`已完成，随后最后一次`emit("diagnostic_finished", **record)`因`seed`重复keyword产生`TypeError`，runner rc1、未超时。已保留原脚本SHA=`dfecd7e5...`和失败raw/rc，只将`emit`改为基础dict后`row.update(details)`。离线核对56项比较、actual ELF、258项调用及原始文件SHA通过，receipt=`unified_mma_group1_fault_diag_mode0_offline_acceptance_v1.json`；没有为日志工具错误重跑默认GPU。原固定队列停止后，仅运行尚未执行的mode1，rc0、未超120秒，receipt=`unified_mma_group1_fault_diag_remaining_finished_v1.json`。原group1精度队列的rc134、未完成项目和性能关闭状态保持。

诊断增加CPU日志改变host发射时序，虽然没有新增GPU同步，也不能证明原异步fault已经修复。本项candidate没有正式性能、没有live部署、没有双seed完整验收；正式默认E65 107.379 μs、E64 95.3059 μs保持。当前资料不能唯一确定GEMM1、reference BLAS、allocator、driver或外部任务中哪一项是根因。

LDS b64资源拒绝和group1编译/原fault/两mode诊断raw限定归档为`unified_lds_mma_closeout_raw_v1.tar.gz`，1432603 B、163 regular members、SHA256=`87a862f8effd0c68f9ba4a26685afd85cb9a111b64e3d331a90d82e5e1d0fcdb`。逐member SHA、四项compile、实际双规模ELF、5spill/private12 B拒绝、原rc134与24项局部比较、两mode各56项严格比较及各258项native调用独立下载验收通过，receipt=`unified_lds_mma_closeout_raw_downloaded_acceptance_v1.json`。归档checker初次误把`execution_failure`当作`rows`成员，修正读取真实字段、未重跑GPU；失败receipt保留在包内。

### 共用ABI的A-only LDS b64读取候选（2026-10-19）

从精确共用默认新建独立`unified_a_lds_b64_v1_source/`，默认关闭`AITER_GFX1250_GEMM1_LDS_A_PAYLOAD_B64=0`。只将A payload每个b128拆为两个aligned b64，B payload loader、ScaleA/ScaleB、hint totals、56次原序K128更新、有效routes、quant/output、全部input/output lifetime fences及GEMM2保持。之前A+B b64已因5 VGPR spills/private12 B拒绝；本项通过收紧到A-only检查是否避免那个具体资源代价，不把旧candidate重跑或资源减少预设为提速。

本地实际AST审计20480个loader cases/786432 B，覆盖四stage、八wave、所有lane和两个K128 subtiles；返回dword地址/顺序与memory多重集完全相同，B loader AST精确默认、所有numerical/compute/fence及GEMM2保持，关闭完整module AST等于当前默认。source SHA：wrapper=`294742a1e603136ac5f01fa9963983601a97153622dc3c4e62d2e5602b386858`、persistent=`e8264bb46ec72cf432e4950baa01de4e9870253e9fdb2efe08af64ac2e55f6a3`。CPU receipt=`unified_a_lds_b64_source_audit_v1.json`，不能替代actual native或GPU精度。硬件依据为CDNA5 ISA §15.15的DS b64/2addr b64地址语义和Shader Guide §5.7.6 Table86的LDS执行率/长dependent latency；文档不保证当前kernel收益。

任务文件已逐项上传a07-3，实际compile尚未启动；没有本项GPU或性能结果，没有覆盖live默认。2026-10-19用户报告机器hang并明确要求等待重启通知，随即停止SSH/SFTP重连、编译与GPU测试，期间只继续本地分析、审计工具和文档工作。不能将此前成功的连接/boot或GPU空闲状态沿用到重启后，也不能从时间先后把机器hang归因本项或历史reference fault。收到用户通知后先重新核对boot、容器、HEAD、任务源码和全机GPU状态，再启动新boot对应的compile/验证标签。

#### 重启后A-only实际编译与资源拒绝（2026-10-19）

用户通知机器已重启后恢复a07-3，boot=`1d5146ac-1863-43d0-8211-f0761013cc4a`。主机分支/HEAD与指定仓库相符，原`dsr1_ep4`容器启动后image仍为`rocm/fw-bringup:gfx1250-atom-20260930-na`。live wrapper/persistent/host runner/benchmark四项SHA重核一致。首次复核发现断连前上传的8项任务source/tool/manifest在remote为0 B；保存原manifest和传输失败receipt，重新逐文件SFTP `put -f`、SHA核验后才继续。不能从这个现象确定文件何时丢失或机器hang原因。

实际launcher AST推导的A/B row stride均2080 B、K128 subtile stride1040 B、stage pitch70656 B、四stage arena282624 B；20480 loader cases/786432 B的地址、返回dword顺序及memory多重集相同，A/B region和arena均不越界，E64/E65及开关两态geometry相同。v2 receipt比较遇到tuple/JSON list及Python版`ast.unparse`格式差异，v3先转换JSON-domain并比较表达式AST后，本地/remote通过；原失败receipt保留。独立标准库ELF/MessagePack parser、全`.text`ISA字节绑定、metadata/notes/resources/ABI交叉核对及14项negative/logic checks通过。有限DS CFG模型依据ISA §5.7.1.4，Shader Guide §6.9的out-of-order措辞与其§4.3.7.2.4存在冲突，不将该模型称为完整hardware hazard证明。

v2编译driver残留旧argv赋值，mode0在读取`--expected-boot`时报`ValueError`、rc1，尚未编译或dispatch；失败driver/raw/status保留。修订v3使用新标签，关闭→打开两项no-dispatch compile均rc0、未超120秒，各自E65→E64→E65同compiled object/cache1。实际结果：

| A-only mode | instructions | VGPR/SGPR | spills/private B | 完整ELF SHA256 | 决定 |
|---:|---:|---|---|---|---|
| 0 | 4531 | 486/68 | 0/0 | `9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b` | 精确当前默认 |
| 1 | 4747 | 487/68 | 1/8 | `d40a0a99a4f18d40f8022b91ad6f471f37ff557352468b50816835319957f2a7` | 资源拒绝，未dispatch |

LDS282624 B和kernarg180 B保持。关闭完整ELF精确默认；candidate五件套`.elf/.isa/.notes/.mlir/summary`下载后逐项SHA及binary独立验收，与remote资源拒绝receipt相符。A-only比旧A+B的5 spills/private12 B更少，但仍未达到zero-spill门槛，不能当作实测提速或回退。此队列已关闭，没有本项GPU精度、性能或live部署。主机复现=`python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/queue_unified_a_lds_b64_compile_v3.py`；它绑定该boot与新标签，历史完成标签不重跑。receipt=`unified_a_lds_b64_compile_acceptance_boot1d5146ac_v3.json`，`passed=false/selected_for_device=false/performance_data=false`。

正式默认仍E65 GEMM1/GEMM2/MoE e2e=`107.379/94.3843/256.893 μs`、E64=`95.3059/60.4664/209.078 μs`，两规模完整精度和同ELF要求已验收，距80 μs差27.379 μs。本轮尚未运行GPU程序。重启初查有他人`jimguo12 ./bwd.out`使用KFD，性能前必须重新检查全机空闲，不以correctness-only或no-dispatch compile耗时替代正式性能。

#### 默认ATT的WMMA operand来源补充（2026-10-19）

`analyze_unified_default_operand_provenance_v1.py`使用已下载SIMD0/1四SE原trace，逐PC绑定当前默认完整ELF并传播physical VGPR的identity MOV，包括`v_mov_b64_e32`与并行`v_dual_mov_b32`。120个sampled waves中的72个compute waves，全部WMMA四类输入operand均追溯到LDS load，没有未识别输入。native matrix A是logical weights/B（16 VGPR），matrix B是activation/A（8 VGPR）；ScaleB/ScaleA各1 VGPR，保持原selectors。

896-WMMA组的各wave“最近相关DS issue→WMMA issue距离中值”的再中值，SIMD0为A414/B581/ScaleA484/ScaleB591 cycles，SIMD1为408/569.5/474/580 cycles。这些是issue距离，不是DS completion latency、kernel μs或独占硬件瓶颈。较长相邻WMMA issue gap常跨M32条件/operand refill或K256的TDM/WG同步区间；分类只描述中间指令，不能把整段gap归因branch本身，更不能相加独立中值预测提速。先保留b128和所有hardware fences，下一项只调整compiler的DS/MFMA hint顺序，以actual native资源和完整精度判断是否值得计时。分析receipt=`unified_default_operand_provenance_v1.json`。首版误将`code.json`的kernel名称注释作为指令匹配，修正跳过注释后通过，没有重采GPU trace。

### 共用ABI的carry段DS-first compiler hints候选（2026-10-19）

从精确当前默认创建`unified_ds_first_v1_source/`，默认关闭`AITER_GFX1250_GEMM1_DS_FIRST_CARRY=0`。只在K256的第二个K128且存在next-stage carry时，把每个future `sched_dsrd` group放到对应`sched_mfma(2)`前，TDM hints的位置、总数、MMA group2和closing cover12保持。所有实际A/B/scale loader继续b128/b32，原两套rmem、有效route与每accumulator的56次有序K128更新、量化、canonical ABI及全部hardware fences/GEMM2 AST保持。此候选没有更改hardware SCHED_MODE，也没有把hint当作hardware barrier。

本地和a07-3 host实际AST审计80组/1960 hint events通过，其中20组carry事件顺序改变、其他60组精确默认；WMMA/DS/TDM hint总量相同。关闭完整wrapper/kernel module AST精确默认。source SHA：wrapper=`07047da54b3858e5508aa6e05137c063a28b45795e2d96bbe8f9158a850f42b5`、persistent=`ecf95a6f842b2410cfcb259b3f3e73686d893c2d96ebad846dda01e807feef57`。逐文件SFTP fsync及remote SHA核对完成，只存独立snapshot，没有覆盖live默认。

固定关闭→打开no-dispatch编译已在a07-3/dsr1_ep4启动，新boot标签=`boot1d5146ac_v1`，每项120秒。actual完整ELF、zero spills/private、DS physical RAW/WAW及input/output native fences通过后，才可进入双seed双规模完整MoE验证；本项尚无device或性能结论。硬件事实只采用Shader Guide §5.7.6 Table86的LDS执行率/延迟与ISA §5.7的counter语义，ATT来源距离和compiler顺序不能保证性能收益。

编译已结束，关闭→打开均rc0/120秒内，E65/E64同compiled object/cache1，candidate完整ELF=`7ce607e86cc4aa9c408e99d736a6c3991ac1e6b47b86fea576697812f36071cb`，4531 instructions、VGPR486/SGPR68、zero spills/private。初native receipt仅判断完整ELF不同，错误地将symbol/metadata变化当作actual schedule变化；下载后独立比较两份`.text`逐字节相同、4531条instruction及physical operands也全部相同，`.text` SHA256均`47604f4f0e98953a88fa1f771960a8a624badb2667d7a1f9688a9cb26f8b8e48`。

修正native gate必须检查executable `.text`变化，原`unified_ds_first_compile_acceptance_boot1d5146ac_v1.json`保留但被新`unified_ds_first_noop_downloaded_rejection_boot1d5146ac_v2.json`明确取代。metadata-only negative case与改一byte executable positive case均通过。candidate为无实际指令变化的实验，停止在no-dispatch阶段，未运行GPU正确性、性能或live部署；编译资源合格不表示存在提速。

### 共用ABI的完整B预读延后一个M32 pair候选（2026-10-19）

新的独立`unified_late_b_v1_source/`仍以当前共用lookup7/group8默认为起点，默认关闭`AITER_GFX1250_GEMM1_LATE_B_PREFETCH=0`。把下一K128的两个N32 B fragments从head移动到当前M32 pair0全部WMMA发出之后；两个完整B slots保持独立，不在当前consumer完成之前覆写其B。A/ScaleA/ScaleB仍双槽、head scales和next A pair0位置保持，实际LDS读仍b128，原TDM ring、全部reader/READY/output hardware fences、compiler hint序列及GEMM2保持。它与历史halfB只延后一个fragment、B1共享或lookahead2/B1reuse不同，不沿用那些已淘汰结果。

CPU运行实际`load_lds_data/k_step/compute_ktile/mma_*` AST，覆盖260个row边界和两个M128 halves，1040 pipeline cases、58240次K128 steps、523264次WMMA通过：所有当前A/B/ScaleA/ScaleB identities正确，有效accumulator仍56次原序更新，selectors/reuse hints、load多重集及TDM/hardware fence/compiler hint事件保持；关闭完整module AST和完整event顺序精确默认。首audit同名nested functions误入未改变的GEMM2，另一次CPU environment漏`TDM_PER`，只修工具后通过，失败receipt保留，candidate source不变。

source SHA：wrapper=`b0fecbfbfe9f8f9dd4b6e9b0ef7fbb7b8153567213318b98f7de222816fcb546`、persistent=`72e6ddf6230b155f058a3342c97cddff00fd9a347af86cec5f0bf64187811d2b`。只上传独立snapshot和必要工具，live默认仍精确当前版本。下一步actual no-dispatch compile要求两规模同对象/完整ELF、zero spills/private、实际`.text`变化及numerical/DS/TDM/native fences通过后才运行完整MoE验证；本项尚无GPU精度或性能。

固定mode0→1 actual compile已完成，均rc0/120秒内、每mode E65→E64→E65同对象/cache1。关闭完整ELF精确当前默认；candidate完整ELF=`ab190ab0ef7eeaf7c535ff6875f0925f8f3f451a0dfab53c02dfdfb28910b58c`，4538 instructions、VGPR486/SGPR70、LDS282624 B、zero spills/private、kernarg180 B。`.text`确实变化，保护的WMMA/DS/TDM/数值opcode counts保持，delta=`s_delay_alu -10/s_set_vgpr_msb +21/s_wait_dscnt -4/s_mul_i32 -1/s_mulk_i32 +1`。四条减少的DScnt waits均为非零threshold，11条DScnt0与全部input/output hardware fences保持。physical VGPR CFG的4538条/17333次访问/7555 transitions没有显式DS RAW/WAW未解决冲突；不等同完整hardware hazard证明。

candidate五件套独立下载后全SHA、ISA字节/binary metadata、fences与remote receipt再次核对通过，native=`unified_late_b_compile_acceptance_boot1d5146ac_v1.json`。随后固定seed0→17，每个进程E65→E64，完整GPU精度验收已通过：两个进程rc0/120秒内，112项有效payload/ScaleA/GEMM2/MoE逐字节比较全为零差异，eager/graph canary和动态A/ScaleA/psum replay通过，每次动态更新保持相同指针，覆盖3/4个空expert。四次原reference MoE门限均pass=True；E65 seed0/17 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`2.7493e-03/2.7491e-03`，E64分别`3.4733e-06/3.4766e-06`、`2.6357e-03/2.6369e-03`。两规模均cache1/同compiled object/同完整ELF=`ab190ab0...`，没有按E新增specialization。

两目录raw、result、manifest及两个actual binding均已下载，独立重算原日志/binding SHA、逐项比较数量、original精度pass、来源SHA、执行rc、同ELF和120秒门限通过；receipt=`unified_late_b_correctness_downloaded_acceptance_boot1d5146ac_v1.json`。这些运行是snapshot imports的correctness-only；期间有其他GPU用户，本项没有有效性能数据，两次结束均无本任务GPU进程残留。生产live仍精确默认。下一步先核对候选正常live imports，再固定默认E65/E64→候选E65/E64→默认E64/E65；每组完整三轮，仅ownership污染允许整组最多三次，执行/精度/解析失败停止，不以耗时筛选。

#### 完整性能队列在默认E64控制的ownership门槛停止

固定队列执行到候选部署前即停止，六组三轮均为精确默认、18轮rc0/120秒内，原random完整MoE精度均True。前置E65第三组ownership合格；前置E64三组全部有ownership污染，按事前最多三次规定停止，不追加第四组或用单轮拼接。完整label前缀=`a07_unified_late_b_mode0_`：

| 后缀 | GEMM1三轮（μs） | 完整合格中值（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership |
|---|---|---:|---|---|---|
| `e65_control_a_boot1d5146ac_v1_try1` | 107.521 /110.131 /110.137 | 不统计 | 95.6778 /94.6825 /95.1106 | 259.746 /260.150 /260.492 | 第1轮污染 |
| `e65_control_a_boot1d5146ac_v1_try2` | 108.210 /108.124 /108.362 | 不统计 | 94.3137 /94.4739 /94.5897 | 258.000 /259.142 /257.691 | 第3轮污染 |
| `e65_control_a_boot1d5146ac_v1_try3` | 107.664 /109.056 /112.453 | **109.056** | 93.5462 /94.3516 /96.3082 | 257.196 /259.981 /266.735 | 三轮合格 |
| `e64_control_a_boot1d5146ac_v1_try1` | 93.9866 /94.6868 /97.6975 | 不统计 | 59.4398 /59.2017 /60.9157 | 205.597 /203.677 /273.319 | 三轮污染 |
| `e64_control_a_boot1d5146ac_v1_try2` | 98.7896 /97.2776 /95.1209 | 不统计 | 61.6156 /61.3437 /59.3913 | 213.660 /210.676 /206.414 | 第1/2轮污染 |
| `e64_control_a_boot1d5146ac_v1_try3` | 94.7623 /95.6856 /95.5846 | 不统计 | 59.5409 /58.7840 /60.4391 | 207.080 /204.982 /209.557 | 第2/3轮污染 |

唯一合格默认E65组的GEMM2/e2e中值为94.3516/259.981 μs，仅属本批控制，不替换正式107.379 μs。候选正常live binding、候选正式计时均未执行，不能判断其提速或回退；该队列关闭，snapshot/native及112项精度证据保留，live两源SHA精确默认。post仍有其他用户的退出中KFD项/busy3，本任务18轮子进程均已退出，没有杀外部进程。限域raw包1428982 B/172 regular members，SHA256=`e30d3ca272d3ba195f25410ec2c356e76233a6c2013ecd50848f2df4fe97b7a3`；逐member SHA、两份native/ELF/.text、112项精度及18轮完整profile/ownership独立验收通过，receipt=`unified_late_b_stopped_raw_downloaded_acceptance_boot1d5146ac_v1.json`。

### 共用ABI的B-only LDS b64候选（2026-10-19）

继续从精确共用默认创建`unified_b_lds_b64_v1_source/`，默认关闭`AITER_GFX1250_GEMM1_LDS_B_PAYLOAD_B64=0`。仅把B payload每个b128拆为两个aligned b64，原A、ScaleA/ScaleB、双slot/pair预读、全部有效WMMA与56次原序K128更新、input/output hardware fences、canonical ABI及GEMM2保持。A-only与A+B此前因1/5 spills拒绝；B-only用于区分B的实际lowering资源成本，不重启旧队列，不预设存在收益。

实际AST loader审计20480项/786432 B，覆盖四stage、八wave、全部lane和两个K128：相同dword地址/返回顺序/多重集，A loader及numerical/compute/fence/GEMM2 AST不变，关闭完整module等于默认，E64/E65仍共同selector0/runtime count。source SHA：wrapper=`9515f31ce68e937678cef7b9b69491b11f8ff6bda099c227550ae0c2039f264d`、persistent=`9de5b08f165a123359a3d3e16e736a3f3d3fcdaa3e4ae1d9fc7293ca7e82c5f2`。native gate保留zero spills/private、真实`.text`变化、硬件fences和显式DS readiness门槛；只对本项改变的payload DS load opcodes比较总dwords，其他数值/DS/TDM不放宽。硬件语义来自CDNA5 ISA §15.15（p739）和Shader Guide §5.7.6 Table86（p235），不能由b64/2addr条数预测收益。此时仅完成本地审计及任务包，尚未actual compile、GPU精度或性能。

### B-only LDS b64 actual compile与本地native验收（2026-10-19）

上述候选的mode0→mode1编译已经在a07-3/dsr1_ep4结束，两项均rc0、未超过120秒、未dispatch。每mode的E65→E64→E65均保持cache1和同compiled object；关闭完整ELF精确等于生产默认`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`。开启完整ELF=`f1c729ebc8d7f824dfef8c9ddc19cb571d0f80d78dbe28d2516c1b67767bfcb1`、4698 instructions、VGPR487/SGPR68，zero spills/private、LDS282624 B及180 B canonical runtime-count ABI保持。相对默认4531 instructions/VGPR486，资源变化本身不是性能收益。

已下载两mode的ELF/ISA/notes/launcher MLIR/summary及原编译日志/result/manifest，独立复核所有SHA、每条ISA编码绑定到`.text`、binary metadata与notes/summary一致、launcher嵌入ELF逐字节一致。candidate `.text` SHA=`df67949d93daa304cd8810ad70b49562a05ec9c8208f4837681b61ffea2a8ee2`，确实不同于默认；原numerical/scalar DS/store/TDM opcode counts、input release/output completion hardware fences和11条DScnt0保持。两mode的static payload DS load总量均为1536 dwords；本项允许payload load opcode改变，未放宽数值或同步门槛。显式physical VGPR的有限CFG DS检查无RAW/WAW冲突，仍不能代替完整GPU正确性。

验收工具=`check_unified_b_lds_b64_native_download_v1.py`，receipt=`unified_b_lds_b64_native_downloaded_acceptance_boot1d5146ac_v1.json`，均位于任务目录。接下来固定seed0→17，每进程E65→E64，做原MoE reference、有效payload/ScaleA/GEMM2/MoE逐字节、eager/graph canary和同地址dynamic input replay；通过后才正常live binding及完整三轮性能。此时尚无本项GPU精度或性能结果，双端live生产默认不变，正式E65/E64 GEMM1仍107.379/95.3059 μs。

### B-only LDS b64双seed双规模完整GPU精度（2026-10-19）

固定seed0→17、每进程E65→E64的两项correctness-only已经完成，rc0、均在120秒以内。112项有效payload/ScaleA/GEMM2/MoE逐字节比较全部`differing_bytes=0`；四次原MoE reference均`pass=True`，E65 seed0/17的logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`2.7493e-03/2.7491e-03`，E64为`3.4733e-06/3.4766e-06`、`2.6357e-03/2.6369e-03`。eager/graph canary及三组同地址dynamic A/ScaleA/psum replay通过，包含3/4个空expert。原reference门限保持，两规模cache1、同compiled object和完整candidate ELF=`f1c729eb...`。

原log/result/manifest/actual binding已经完整下载，独立逐SHA、112项比较、四次reference及cache/ELF验收通过，receipt=`unified_b_lds_b64_correctness_downloaded_acceptance_boot1d5146ac_v1.json`。这些是snapshot imports的正确性验证，存在外部GPU users，所有耗时无效；未发现本任务GPU进程残留，live默认SHA保持。继续固定默认E65/E64→候选E65/E64→默认E64/E65的三轮组对照，候选正式计时前额外做正常live imports双规模绑定。ownership污染整组最多三次尝试，执行/精度/解析失败立即停止，finally恢复默认；不重启此前已停止late-B队列。

#### B-only性能队列因三组ownership污染停止

固定队列停在第一个默认E65前控制，三组三轮共9轮全部rc0、120秒内、原random完整MoE精度True；但三组均有ownership污染，没有完整合格组。label前缀为`a07_unified_b_lds_b64_mode0_e65_control_a_boot1d5146ac_v1_`：

| 尝试 | GEMM1三轮（μs） | GEMM2三轮（μs） | MoE e2e三轮（μs） | ownership / 性能口径 |
|---|---|---|---|---|
| `try1` | 106.687 /108.568 /106.738 | 94.4539 /93.9164 /95.3756 | 255.056 /256.800 /256.951 | 第1/3轮污染，整组排除 |
| `try2` | 107.250 /107.655 /108.344 | 93.7605 /95.0951 /95.6034 | 256.256 /257.968 /259.469 | 三轮污染，整组排除 |
| `try3` | 111.197 /107.619 /107.845 | 95.8772 /95.7992 /94.7408 | 262.707 /258.240 /257.865 | 三轮污染，整组排除 |

按事前最多三次规定关闭队列，不追加第四组、不从不同组抽取单轮拼成中值。summary为`finished=false/performance_data=false/execution_failure=null/default_restored=true`，异常原文为`Three ownership-contaminated whole groups: E65 mode0`。candidate正常live部署、live binding及正式计时都未执行，不能据此宣称候选提速或回退；native及112项精度证据保留，双端live源码仍精确默认。10:12 UTC的最终post有外部`jimguo12 ./bwd_dqdkdv.out`使用KFD，属于历史负载记录，后续性能前需重新检查，不杀外部进程。

已生成并下载限域raw归档`unified_b_lds_b64_stopped_raw_evidence_boot1d5146ac_v1.tar.gz`，1120512 B、136 regular members，SHA256=`b57c35a28e0f25a685e10ace55eb045ea55c35e38f439e880eb12178c4650fcc`。本地独立重算每member SHA、两mode完整ELF/ISA/.text/binary metadata/launcher、112项精度及九轮原profile/ownership全部通过。receipt为`unified_b_lds_b64_stopped_raw_downloaded_acceptance_boot1d5146ac_v1.json`。没有产生新正式性能纪录；默认E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。

### 共用ABI的M32 runtime bit mask候选（2026-10-19）

从精确共用默认新建`unified_m32_mask_v1_source/`，默认关闭`AITER_GFX1250_GEMM1_M32_RUNTIME_MASK=0`。按真实`wmb + pair*32 < mn_oob`把四个原M32 first-row条件编码进单个Int32 mask，经SGPR tied identity保存；在原load/WMMA条件使用点读取该mask的一bit，局部identity仅限制compiler CSE，不是hardware fence。保留部分有效M32的整pair计算、两个operand slots、56次原序K128累加、全部load/TDM/hardware fence/compiler hint事件、canonical ABI和GEMM2。它与旧缓存四个0/1 SGPR条件及all-live双pipeline不同，没有按E64/E65增加编译specialization或绑定seed。

本地与a07-3 host执行实际mask、load、compute AST，260种row边界/两个M128 halves、1040 pipeline cases、58240 K128 steps、523264 WMMA通过：两个mode的operand identities、WMMA及load/TDM/fence/hint完整event顺序精确默认；关闭两module完整AST保持，43个nested functions/GEMM2不变。wrapper SHA=`2d0ce3c34757ac03034d50bbb5225834de57d0bab33cca36fd6255e237382069`，persistent=`b3f771f19046bd45e6a4d89c7eee5bb206743ec12c62cf380d511c2572039f34`。仅上传任务snapshot/tools并核对SHA，live生产默认未改。

下一步在a07-3/dsr1_ep4固定mode0→1 no-dispatch compile，每项120秒。native必须保持zero spills/private、同compiled object/cache1/完整ELF共用、180 B runtime-count ABI、原数值/DS/TDM及hardware fences，并实际改变`.text`且减少目标条件转换proxy sites，才进入完整MoE精度。CDNA5 ISA §15.4（pp268–270）记载scalar compare/bitcmp写SCC的语义，不保证LLVM将本项表达式lower为某个opcode或取得性能收益。此时没有新device或正式性能结果，107.379/95.3059 μs默认保持。

#### M32 mask v1资源拒绝与v2局部条件收紧

v1关闭→打开无dispatch编译均rc0、120秒内，两mode各E65→E64→E65 cache1/同compiled object。关闭完整ELF精确默认；打开ELF=`2471c8e586125cb44f2769af36e64e2049d9d7fea73a3cb0fd769c888a874c31`，4685 instructions、VGPR486/SGPR70/LDS282624 B/kernarg180 B，但出现6个VGPR spills、private16 B。因此native gate拒绝，不dispatch、不运行该版本GPU精度或性能。

已下载两mode的native五件套独立复核ELF/ISA/.text/binary metadata/summary/launcher及SHA，receipt=`unified_m32_mask_native_downloaded_acceptance_boot1d5146ac_v1.json`。v1的opcode delta显示`s_and_b32 -61/s_cselect_b32 -61/s_cmp_lg_u32 -64`，同时`s_bitcmp0_b32 +128/s_cbranch_scc1 +60/s_mov_b32 +127/s_set_vgpr_msb +65`及各6条scratch load/store；目标转换减少，但branch重复和spill使它不满足dispatch资格。静态证据不能换算kernel μs。

v2仅收紧同一个M32 pair的future A load与current WMMA为同一局部mask条件，load_pair_data增加already_live，原head B/scales与BACK ScaleA位置保持，未来A仅从相邻scope移入同pair；双slot、原K128次序、load多重集/TDM/hardware fences/compiler hints保持。CPU实际AST再次覆盖1040 pipeline cases/58240 K128 steps/523264 WMMA，全部有效operand/update和完整hardware events相同，关闭完整module及event顺序精确默认。v2 wrapper SHA保持，persistent=`78013d108d82a1b519f962ce5053fa29da3d0b4b82bac8d207b0085d7b4e02cd`。v1失败证据保留，新v2先做独立no-dispatch compile，不重启v1队列、不修改live默认。

#### M32 mask v2实际编译和独立native验收

v2两项编译均rc0、120秒内、未dispatch，各mode E65→E64→E65 cache1/同对象/完整ELF共用。关闭完整ELF精确默认；候选ELF=`b46ef980bb5e30b553d3b020721f76414b6d0126b9bc8b1f38ec9c951d030f13`、`.text` SHA=`8f7e747cdf1d1fc68819a15fee495f8502fb2e03332cbc8107569e46e5841523`，4420 instructions（默认4531），VGPR486/SGPR70、zero spills/private、LDS282624 B、kernarg180 B与count offset176/size4保持。

两mode native五件套及原compile log/result/manifest/idle已下载；独立检查每条ISA编码绑定ELF `.text`、binary metadata与notes/summary、launcher嵌入ELF及所有SHA通过。有限physical DS CFG无未解决RAW/WAW，原数值/DS/TDM opcode与全部hardware fences/11条DScnt0保持。delta为`s_and_b32 -61/s_cselect_b32 -61/s_cmp_lg_u32 -64/s_delay_alu -59`，新增`s_bitcmp0_b32 +68/s_mov_b32 +67`；原branch数保持，v1新增branch和scratch已消除。receipt=`unified_m32_mask_native_downloaded_acceptance_boot1d5146ac_v2.json`。这些实证足以进入完整GPU精度，不能把111条static指令减少换算为性能收益。

下一步固定seed0→17、每进程E65→E64，验证原MoE reference、HEAD有效payload/ScaleA/GEMM2/MoE逐字节、eager/graph canary及dynamic A/ScaleA/psum replay；每程序仍120秒，correctness-only耗时不作性能。live生产默认保持，正式E65/E64 GEMM1仍107.379/95.3059 μs。

#### M32 mask v2双seed完整GPU精度独立验收（2026-10-20）

a07-3/dsr1_ep4的seed0/17队列均rc0、120秒内完成；下载原始run/result/manifest/idle和actual binding后，用`check_unified_m32_mask_correctness_download_v2.py`重新独立核验。E65→E64每个进程共用一个compiled GEMM1、cache1及完整ELF=`b46ef980bb5e30b553d3b020721f76414b6d0126b9bc8b1f38ec9c951d030f13`，snapshot源码SHA与native验收一致。全部112项有效payload/ScaleA/GEMM2/MoE逐字节比较为0，eager/graph canary和同地址动态A/ScaleA/psum replay通过，四次原MoE reference精度通过，未放宽标准。

| workload | seed | logits_diff | rel_l2 |
|---|---:|---:|---:|
| E65/topk9/shared1 | 0 | 3.7793e-06 | 0.0027493 |
| E64/topk8/shared0 | 0 | 3.4733e-06 | 0.0026357 |
| E65/topk9/shared1 | 17 | 3.7788e-06 | 0.0027491 |
| E64/topk8/shared0 | 17 | 3.4766e-06 | 0.0026369 |

独立receipt=`unified_m32_mask_correctness_downloaded_acceptance_boot1d5146ac_v2.json`，SHA256=`81c0ce418f1ebe0d1cf1d3c931b6c60d225a1e15dcad519613d56ed9d7a2836b`。这是correctness-only snapshot验证，有外部GPU owner时也允许执行，全部计时无效；结束检查无本项GPU进程残留。candidate正常live imports仍须另做binding。

接下来固定性能顺序为默认E65→默认E64→候选E65→候选E64→默认E64→默认E65，每组完整三轮，只允许ownership污染整组最多三次尝试，任何执行/精度/解析错误立即停止。`run_unified_m32_mask_dual_perf_v2.py`在host运行，先等待全机空闲，candidate部署后验证正常live imports双规模同对象/完整ELF，finally在idle/SHA守卫下恢复默认。尚无新正式性能收益，当前E65/GEMM1 107.379 μs、E64 95.3059 μs；E65距80 μs差27.379 μs。

#### M32 mask v2性能队列在默认前控制停止（2026-10-20）

队列已在第一个默认E65控制的三次ownership尝试后关闭。全部九轮原random non-balanced完整MoE均rc0、120秒内、精度True；三组分别第1/3/1轮出现外部live GPU owner，完整三轮组均排除，不计算正式中值、不拼轮、不追加第四次。候选正常live部署、live binding及正式计时均未执行，因此没有候选性能收益或回退结论。

| 默认E65控制 | GEMM1三轮 μs | GEMM2三轮 μs | MoE e2e三轮 μs | ownership合格轮 |
|---|---|---|---|---|
| try1 | 108.505 /108.296 /108.034 | 94.8465 /94.9011 /95.4850 | 258.033 /258.293 /257.519 | 2、3 |
| try2 | 109.886 /109.266 /107.191 | 94.7872 /93.3597 /91.7975 | 260.291 /257.510 /254.946 | 1、2 |
| try3 | 107.978 /111.549 /107.956 | 94.3929 /93.4926 /95.6474 | 258.073 /261.637 /259.416 | 2、3 |

summary=`unified_m32_mask_dual_perf_summary_boot1d5146ac_v2.json`，queue rc1仅来自`Three ownership-contaminated whole groups: E65 mode0`；`execution_failure=null`。finally核对精确默认SHA保持，post于2026-10-19T16:04:41Z为KFD空、busy0、VRAM176582656 B、全机idle，没有本项GPU进程残留。此为当时状态，不代表后续机器持续空闲。

#### M32 mask v3显式SCC bit test源码候选（2026-10-20）

v2 native的局部identity生成每次`s_mov_b32 temp,mask`再`s_bitcmp0_b32 temp,pair`；下一新snapshot v3保持v2的同pair future A/current WMMA局部scope，只将helper改为`llvm.inline_asm`的`s_bitcmp1_b32 $1,pair`与`={scc},s`输出，真实mask仍只读。依据CDNA5 ISA §15.4 p269的bitcmp1语义，返回SCC等于选定位；这不是hardware memory fence，也不改lane EXEC。与旧直接比较SCC实验不同，本项以已合并scope的四bit mask为输入，必须验收actual native的mask-copy减少、zero spills/private及资源。

本地actual AST审计1040 pipeline cases/58240 K128 steps/523264 WMMA通过，保持每个有效accumulator的56次原序更新、operand identities、load多重集/TDM/hardware fences，关闭完整module/event顺序精确默认，canonical ABI/GEMM2保持。snapshot wrapper SHA=`2d0ce3c34757ac03034d50bbb5225834de57d0bab33cca36fd6255e237382069`，persistent=`cc1ff29bb0a4ad201aeee98c346c5bcf65758313aca1849ccf7c03c38e9ee805`。仅上传该snapshot和必要tools，先进行a07-3/dsr1_ep4 no-dispatch实际编译，不重启v2队列，不提前修改live默认；CPU语义不替代native或GPU精度验证。

#### M32 mask v2原始证据归档及v3编译淘汰（2026-10-20）

v1/v2限域raw归档`unified_m32_mask_stopped_raw_evidence_boot1d5146ac_v2.tar.gz`已下载并独立验收：1569138 B、177 regular members，SHA256=`3337beaa73d9938dea1d98c4eff63d6226ea757d6a2450b3f4d1c74f840e7e9f`。逐member SHA、v1资源拒绝/v2 native五件套、112项精度及九轮原profile/ownership全部通过，receipt=`unified_m32_mask_stopped_raw_downloaded_acceptance_boot1d5146ac_v2.json`。首次checker误把原日志中`pass=True`的位置限定在`logits_diff`之后；已按实际日志格式修正并重新独立核验，仅修正CPU解析，没有修改raw包或重跑GPU，失败准备记录保留于`unified_m32_mask_raw_checker_preparation_failure_v2.json`。零完整合格性能组的结论保持，v2队列关闭。

v3在a07-3 host完成actual AST审计，在dsr1_ep4固定mode0→1 no-dispatch编译，两项均rc0、120秒内；各mode E65→E64→E65同compiled object/cache1/完整ELF，关闭完整ELF精确默认。候选ELF=`6a2406da0fb91559b6cec12a247e42f611bd4b34a95851c128cebffa55b3b577`、`.text` SHA=`6ea40d3f198c52dc4d86a86b11382e60f64b9c323316a84bccde27b336f2c0eb`，4557 instructions（v2为4420、默认4531），VGPR486/SGPR70、zero spills/private、LDS282624 B、kernarg180 B、count offset176/size4保持。68条显式`s_bitcmp1_b32`保留且重复mask拷贝减少，但返回SCC后LLVM又生成转换；相对默认`s_and_b32 -61/s_cselect_b32 +7/s_cmp_lg_u32 +4/s_delay_alu +9`，目标转换proxy为385→335（v2为199）。

native gate以`The candidate did not reduce the v2 executable instruction count`拒绝v3；已淘汰，未dispatch、没有该版本GPU精度或性能数据。两mode五件套及原compile log/result/manifest/idle下载后独立ISA字节/ELF/binary metadata/launcher/SHA及DS/fences验收通过，receipt=`unified_m32_mask_native_downloaded_acceptance_boot1d5146ac_v3.json`明确`candidate_native_qualified=false`。live生产默认保持，正式E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs差27.379 μs。

下一候选仅调整GEMM1 accumulator的rmem创建/存储顺序，再映射回原logical `wm * mma_n_rep + wn`索引；保持实际WMMA snake执行次序、每个有效accumulator的56次原序K128更新、输出、同步、GEMM2及canonical ABI。先CPU实际allocation/mapping身份审计，再检查actual native是否减少`s_set_vgpr_msb`且zero spills/private；若`.text`相同或没有目标改善，直接拒绝，不dispatch。Shader Programming Guide §3.3.2.3记载该指令设置MODE的VGPR MSBs，§5.3.3及§5.7.1.2说明部分control instruction可co-issue，因此static数量减少不能自动换算为cycle或μs收益。此处仅记录下一方向，尚无该候选代码、编译或性能结果。

#### 共用ABI的accumulator storage snake v1源码审计（2026-10-20）

从精确默认建立`unified_acc_storage_snake_v1_source/`，默认关闭`AITER_GFX1250_GEMM1_ACC_STORAGE_SNAKE=0`；仅增加编译期实验选项`acc_storage_snake`，没有按真实expert数specialize。原`c_frags`按logical行列建立，候选先按原WMMA snake顺序分配`acc_storage`，再映射回同一个logical索引：t256的logical→storage为`[0,1,3,2,4,5,7,6,8,9,11,10,12,13,15,14]`。因此所有原compute、zeroing和epilogue均保持原索引/AST，不改56次K128次序、valid routes、load/TDM/fences或GEMM2；live生产默认未改。

本地CPU actual AST执行64种allocation/mapping cases，检查全部allocated objects互异、映射双射、16/8-dword原零初始化及logical读写身份；把这些实际allocated objects注入原独立pipeline interpreter，1040 pipeline cases、58240 K128 steps、523264 WMMA全部与默认完整event顺序相同，46个nested functions保持。关闭两module完整AST还原默认，receipt=`unified_acc_storage_snake_source_audit_v1.json`。审计准备期间修正了parent函数范围和AST语句缩进问题，仅CPU工具变化，诊断记录保留；候选源码没有因工具错误改动或GPU重跑。

wrapper SHA=`018afae205204fceb4f9484eaf1a4577ac21e6235496669ebfa559dbe6bf14ea`，persistent=`95c650d6fb39ccee87bf34dbcc62b300e3600c910205a08bb22c2563188eff26`。仅上传snapshot及必要tools，并核对a07-3分支/HEAD/boot/image/live SHA和候选SHA；下一步固定0→1 no-dispatch compile，每项120秒。native必须实际改变`.text`、减少目标`s_set_vgpr_msb`且不增加总指令数、zero spills/private、同对象/完整ELF服务E64/E65、count ABI及原numeric/DS/TDM/fences保持，才进入完整GPU精度。现阶段没有该候选性能数据，正式E65/E64仍107.379/95.3059 μs。

#### Accumulator storage snake v1实际native拒绝及寄存器布局证据

a07-3 host的actual AST审计通过；dsr1_ep4固定两项compile均rc0、120秒内，关闭完整ELF精确默认，各mode E65→E64→E65同compiled object/cache1/完整ELF。候选ELF=`e3584e0960b6f5b4a44a532311a06fd64e9f199cbe64b92267be2d84351e4d03`，`.text` SHA=`5c5f01176d404234c9f318e18f23ff64d79fe804a8dfaceef99e6b72470ff8af`；指令4531→4549，`s_set_vgpr_msb`375→391，VGPR486/SGPR68、zero spills/private、LDS282624 B/count ABI保持。native gate拒绝：目标MSB sites与总指令数都增加；另有VOPD配对变化造成单opcode计数改变，不能仅凭该配对delta断言math改变。该候选淘汰，未dispatch、无GPU精度或性能。

两mode原五件套及compile logs/results/manifests/idle下载后，独立ELF/ISA字节、binary metadata、launcher/SHA、physical DS CFG与hardware fences验收通过，receipt=`unified_acc_storage_snake_native_downloaded_acceptance_boot1d5146ac_v1.json`。第一次host普通chmod因容器root输出权限被拒绝，随后仅对本项两个生成目录用`sudo -n chmod -R a+rwX`成功；下载时idle文件名由误写的`.log`纠正为实际`.jsonl`，没有重跑compile或GPU。post没有本项live GPU owners。

`analyze_unified_acc_storage_native_v1.py`逐条绑定两版ELF的physical WMMA operands：两者均有256 accumulator dwords，其中240在v0..v255、16在高256 bank，一个M16/N32 accumulator仍跨MODE的MSB选择；创建顺序并没有把全部accumulator留在同一个bank。新增16个MSB sites全部落在first→last static WMMA区，四组28/32-WMMA backedge区间各从28/32变为29/34个MSB sites。native还有240条`v_mov_b64_e32`分布在两个零初始化路径，各120条；它们是零向量复制，不能当作每次K128搬运。分析receipt=`unified_acc_storage_native_analysis_v1.json`，这是static布局和互斥路径证据，不是dynamic stage耗时或唯一瓶颈。

下一方向是把GEMM1全accumulator零初始化延后到首次input LDS prime之后，缩短零值live range并避免padding expert的无效初始化；保留每个有效accumulator在第0次WMMA前得到原+0、后续56次原序更新和完整输出。须从精确默认另建snapshot、直接审计实际prime/zero路径，然后先no-dispatch native检查zero-copy/resource目标变化；此时尚未实施该方向。正式默认及107.379/95.3059 μs保持。

#### Accumulator late zero v1实际源码及CPU审计（2026-10-20）

从精确共用默认创建`unified_acc_zero_late_v1_source/`，默认关闭`AITER_GFX1250_GEMM1_ACC_ZERO_LATE=0`，仅将GEMM1原`zero_accumulators()`从`run_task`早期移至首次`load_lds_data(rmem_slots[0], stage_addr, 0)`之后；load/TDM和hardware fences位置、原+0值、所有有效routes与56次原序K128更新、compute/epilogue/GEMM2、canonical runtime-count ABI保持。作用域限定为已验收的共用selector0、t256/w2x4/b4、Ntask1、rowtail3及现有prefetch配置，E64/E65真实expert数仍是同一个runtime参数，不新增expert-count specialization。padding expert在开启模式下不执行无效prime/compute/zero。

本地及a07-3 host的actual AST审计已通过：192个expert/prime/zero边界cases中，每个有效accumulator恰好得到一次原+0，在首次prime之后、首WMMA之前；实际primed/zeroed对象注入原pipeline后，1040 cases/58240 K128 steps/523264 WMMA全部保持原operand、scale selector、reuse hint和完整event顺序。45个nested函数及GEMM2不变，关闭完整module AST精确还原默认。receipt=`unified_acc_zero_late_source_audit_v1.json`；source wrapper/persistent SHA=`0d9b366d9d8532091248c6ab3780c275fbfcc4b8c9273dd36e826cbea2fde504`/`9de283ce717ae2c3c8e814191f2ba5f193519d5da2eac36cb3775669debfdf3b`。已只上传本项snapshot/tools并核对SHA，live默认未改。

接下来固定0→1 no-dispatch编译、每项120秒。事前native gate要求关闭完整ELF精确默认、开启`.text`实际变化、zero-copy proxy `v_mov_b64_e32`减少且总指令不增加，zero spills/private、共用compiled object/ELF/ABI、原numeric/DS/TDM counts及physical DS CFG/hardware fences保持；通过后才完整GPU精度及正式计时。这些static门槛不能直接推导μs收益，CPU审计也不替代GPU验证。当前正式E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs差27.379 μs。

#### Late zero v1实际编译：executable完全相同，淘汰

a07-3/dsr1_ep4的固定0→1编译均rc0、120秒内、未dispatch；每mode E65→E64→E65同compiled object/cache1/完整ELF。开启ELF=`d9d63caf00c3edbcbf6a4b3024eb40d8b17fb46f3fb0f8bf2a52d7fcca42f1c9`，但两mode的executable `.text`逐字节相同，SHA均=`47604f4f0e98953a88fa1f771960a8a624badb2667d7a1f9688a9cb26f8b8e48`。4531 instructions、VGPR486/SGPR68、zero spills/private、LDS282624 B/count ABI全部相同；240个zero-copy proxy与375个MSB sites也未变。完整ELF差别来自kernel命名/metadata，不构成native优化，不运行重复性能。

两mode五件套及原logs/results/manifests/idle已下载，独立验收source/ELF/每条ISA编码/binary metadata/launcher/SHA、完整`.text`字节相等及DS/fences通过。receipt=`unified_acc_zero_late_native_downloaded_acceptance_boot1d5146ac_v1.json`，candidate_native_qualified=false；队列已关闭、post无本项GPU owner。初CPU checker误取共享parser不存在的`words`字段，已改为实际`word/size/ins`并保留完整`.text`字节比较，准备失败记录保留，未改候选或重跑编译/GPU。

下一项以首次K128的原+0作为WMMA C输入，试图消除有效accumulator的初始化复制；保持第0次在内的全部56次原序更新。Shader Guide §4.6 VOP3P fields（p146）及CDNA5 ISA §7.12（VOP3P fields，p96）明确WMMA SRC2可为even VGPR或inline constant zero；ISA §15.11 pp465–466仍定义D=A*B+C。只能使用相同+0，不能改为undef或丢弃有效计算。先建立新的独立snapshot，审计首次与rolled pipeline事件/有效accumulator/epilogue边界，再验收actual ISA；尚未实施或取得新性能。

#### First-WMMA +0 v1实际源码与完整pipeline CPU审计

新独立snapshot=`unified_acc_first_zero_v1_source/`，默认关闭`AITER_GFX1250_GEMM1_ACC_FIRST_ZERO=0`。仅在共用selector0/t256/w2x4/b4/K7168/Ntask1/rowtail3的scope，把原tile0移到owner rolled loop前执行一次，rolled `steady_start=1`；tile0的首K128使用原FP32 +0向量作为C，其后及全部剩余tile仍读写原独立accumulator。未计算的M32 pairs继续初始化为原+0，避免epilogue内部读取未定义padding；每个有效pair保留第0次在内的56次原序K128更新。`issue()`仍按真实wave选择原job，原routing/ABI/GEMM2和其他scope保持；E64/E65不新增specialization。

本地实际AST执行4160个pipeline cases/232960 K128 steps/2093056 WMMA，覆盖全部八wave的原owner分支、两M128 halves及-1..257/30464 row边界；576个expert边界cases覆盖E64/E65有效末expert及padding。首次C为逐位+0、后续完整C history的dyadic FP32 oracle、有效accumulator最终值、28个tile完整次序、WMMA operand/scale/reuse selectors、loads多重集/顺序及TDM/hardware fence/compiler hint事件与默认一致。41个nested函数及GEMM2不变，关闭完整module AST精确默认，receipt=`unified_acc_first_zero_source_audit_v1.json`。CPU Restore误访问GEMM2同名helper的新增参数检查已修正并留diagnostics，候选源码未因此改动。

source wrapper/persistent SHA=`c5461d3ee15adc42937c4c7c0a93868e11f63b794b3ce3f751443d2eb6fa0763`/`a463fe4013465c4c8261654cc7db44e6be1df675be78ddf7afc56b2d0933ae1d`。接下来a07-3 host复核CPU audit、dsr1_ep4固定0→1 no-dispatch编译，每项120秒；必须看到16个首K128 inline-zero sites且不在backedge内、zero-copy减少、zero spills/private和共用count ABI保持。peel会增加32条static WMMA、816 B static LDS reads及一组carry fence，事前逐项绑定，不能把static数量增长误认为runtime多计算或把减少zero-copy直接换算μs。尚无本项GPU精度/性能，live默认及107.379/95.3059 μs保持。

#### First-WMMA +0实际编译与验收契约修正（2026-10-20）

a07-3 host actual AST复核rc0；dsr1_ep4固定0→1 no-dispatch编译均rc0、120秒内，各mode E65→E64→E65同compiled object/cache1/完整ELF。关闭完整ELF精确默认；开启ELF=`e13389f41529c8cdba9a18215e418fd58983a779a09b36a80e5f5655a28fcc14`，`.text` SHA=`3e39c0c15e016a595470daf016a4c09f7bea3680933ff116b63355df7c63b743b`，4531→4718 instructions、VGPR486→492/SGPR68→74、zero spills/private、LDS282624 B/count ABI保持。首16条WMMA实际使用inline +0 C，后16条对应相同physical accumulator D/C及相同scale/reuse modifiers；zero-copy proxy240→96。总指令增长187条，在事前peel上限600条内，不表示已取得性能收益。

原v1 native gate以`Unexpected protected numerical/DS/TDM opcode counts changed`、`Hardware fence sequence changed`拒绝，队列rc1关闭，未GPU dispatch。两mode五件套及原logs/results/manifests/idle已下载并重新独立绑定，receipt=`unified_acc_first_zero_native_downloaded_acceptance_boot1d5146ac_v1.json`保留原拒绝、`candidate_native_qualified=false`。CPU checker的Windows参数引用及manifest字段名准备错误已修正并留diagnostics，没有重跑编译、修改候选或覆盖原证据。

逐条诊断确认原gate的static契约有遗漏：拆开VOPD两个half并归一化`_e32/_e64`后，epilogue全部数值运算数量保持；剩余差别精确为WMMA256→288、LDS b128 384→432/b32 42→52/stride64 b32 14→15，总static LDS bytes/lane6528→7344，及展开首tile的TDM20→24。4个新增TDM sites的actual kernarg pointer链与wave predicates已分别绑定A/B/ScaleA/ScaleB，不凭disassembly顺序指定owner。原source的动态load/TDM/fence事件完整相同，有效accumulator仍56次原序K128更新，static展开不代表runtime增加有效计算。

原gate还把`s_barrier_wait -1`与native打印`0xffff`误作不同；actual static增量为一组carry `tensorwait2/signal/wait`和三条`tensorwait10`。保守native CFG outstanding上界分析不使用predicate事实、不假定异步完成，三条新增`tensorwait10`入口最大3/4/5，均不可能等待；没有移除原完成fence。依据CDNA5 ISA §5.7.1、§15.6的`TENSORCNT <= SIMM16[5:0]`及Shader Guide §4.10.1的同wave TDM计数/in-order语义，这只是counter与路径证据，不换算cycles或μs。八backedges/四targets、input/output fences、physical DS显式RAW/WAW仍通过。

使用同一份已编译ELF的新独立qualification=`unified_acc_first_zero_native_qualification_boot1d5146ac_v2.json`已通过；绑定原失败receipt SHA、源码、完整native及静态展开诊断，不重启旧队列、不放宽zero-spill/资源/ABI/精度标准。复现CPU核查=`py -3 -X utf8 .codex_tmp/dsr1_e65_gemm1_20261017/qualify_unified_acc_first_zero_native_v2.py`。接下来才做双seed双规模完整GPU验证；目前没有该候选GPU精度或正式性能，live默认及E65/E64 GEMM1 107.379/95.3059 μs保持。

#### First-WMMA +0双seed双规模完整GPU精度通过（2026-10-20）

a07-3/dsr1_ep4固定seed0→17、每进程E65→E64已完成，两程序均rc0、120秒内，原队列rc0关闭。下载两seed的完整run/result/manifest/idle、source snapshots、actual binding及queue/control logs后，执行`py -3 -X utf8 .codex_tmp/dsr1_e65_gemm1_20261017/check_unified_acc_first_zero_correctness_download_v2.py`独立核验通过。每规模每seed28项、共112项有效payload/ScaleA/GEMM2/MoE逐字节比较全部零差异，eager/graph canary及同地址dynamic A/ScaleA/psum replay均通过，四次原MoE reference精度正常，精度标准保持。

| workload | seed | logits_diff | rel_l2 |
|---|---:|---:|---:|
| E65/topk9/shared1 | 0 | 3.7793e-06 | 0.0027493 |
| E64/topk8/shared0 | 0 | 3.4733e-06 | 0.0026357 |
| E65/topk9/shared1 | 17 | 3.7788e-06 | 0.0027491 |
| E64/topk8/shared0 | 17 | 3.4766e-06 | 0.0026369 |

两个进程内E64/E65实际复用同一个compiled GEMM1、cache1及完整ELF=`e13389f41529c8cdba9a18215e418fd58983a779a09b36a80e5f5655a28fcc14`，与no-dispatch native qualification绑定一致。独立receipt=`unified_acc_first_zero_correctness_downloaded_acceptance_boot1d5146ac_v2.json`，SHA256=`95ff662ac09cb2ea38cde8fda926ab6ba783c3a099f29b02f49cc38a3ac68f3b`。这是correctness-only snapshot imports，有外部GPU owner时也允许执行，全部计时无效；post无本项GPU进程残留，live默认两源SHA保持。

下一步使用已准备的`run_unified_acc_first_zero_dual_perf_v2.py`，host在全机idle门禁下固定默认E65→默认E64→候选E65→候选E64→默认E64→默认E65，各组完整三轮；仅ownership污染允许整组最多三次尝试，执行/精度/解析失败立即停止。candidate部署后另做正常live imports双规模56项与同对象/完整ELF binding，finally恢复精确默认。目前本候选尚无正式性能，当前E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。

#### First-WMMA +0完整正式性能：E65回退，淘汰（2026-10-20）

固定默认E65/E64→候选E65/E64→默认E64/E65队列已rc0结束，共24轮原random non-balanced完整MoE、120秒上限内、原精度True。candidate正常live imports先通过额外seed0双规模56项零字节比较和同compiled object/cache1/完整ELF binding。计时期间没有本任务传输、编译、ATT或并发GPU监控；仅E65 candidate try1第3轮、E64后控制try1第2轮ownership不合格，各整组三轮排除，按事前规则重试try2，不按耗时重试或拼轮。

| workload | 完整合格组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前控制try1 | 107.952 /109.677 /109.466 | 109.466 | 94.8674 | 259.456 |
| E64/topk8/shared0 | 默认前控制try1 | 95.1226 /95.9385 /94.9138 | 95.1226 | 61.0588 | 206.933 |
| E65/topk9/shared1 | candidate try2 | 107.940 /109.930 /110.444 | 109.930 | 94.2476 | 259.617 |
| E64/topk8/shared0 | candidate try1 | 97.0560 /95.6452 /94.8417 | 95.6452 | 61.0335 | 207.749 |
| E64/topk8/shared0 | 默认后控制try2 | 97.1062 /93.9119 /96.1554 | 96.1554 | 61.4085 | 208.681 |
| E65/topk9/shared1 | 默认后控制try1 | 108.399 /108.152 /108.000 | 108.152 | 94.9113 | 258.123 |

E65 candidate GEMM1比两侧默认慢0.423876%/1.643983%，MoE e2e慢0.062053%/0.578794%；E64 GEMM1比前控制慢0.549396%、比后控制快0.530599%，收益不一致。因此淘汰first-WMMA +0，不追加反向或按耗时筛选，不接入生产。两个排除组的GEMM1原读数分别107.442/106.698/108.146和94.7321/94.5170/95.4784 μs，中值保持null，原raw全部保留；这些读数不能替代合格组。

所有性能run/result/manifest/idle/source snapshots、plan/summary、正常live binding及部署/恢复host receipts已下载；`check_unified_acc_first_zero_perf_download_v2.py`重新解析24轮原profile、精度、SHA、全机ownership和六个完整三轮中值，独立验收通过。receipt=`unified_acc_first_zero_perf_downloaded_acceptance_boot1d5146ac_v2.json`，SHA256=`1790d90c4a46858fee5e591610f2b0fdd217c659095268591c70ab345eceb29a`；summary SHA256=`6105da0170a94da62e1b46a5d94ecaebbbcc7162b7e2c674bea3b7b3c8472f90`。finally恢复精确默认，2026-10-19T18:09:15Z的post全机idle、KFD空、busy0、VRAM176582656 B，无本项GPU进程残留。此为当时状态，后续性能前仍须重查。正式默认E65/E64 GEMM1保持107.379/95.3059 μs，距80 μs差27.379 μs。

#### Scale LDS base tied identity候选：只约束地址lowering（2026-10-20）

默认完整ELF的operand provenance已确认：steady各owner的ScaleA/ScaleB大多已生成2addr读取，但prime和最后固定drain tiles仍保留独立`ds_load_b32`及重复base adds。新独立`unified_scale_base_tied_v1_source/`仅在两种`lds_sa_base/lds_sb_base`完成lane-specific地址加法后，将低32位地址经过空LLVM asm `"", "=v,0", has_side_effects=False`，再恢复原index返回类型；普通LDS load、原scale地址/字节顺序、slots与全部hardware fences保持。tied identity保证输入输出同寄存器，不沿用旧公共helper的`=s,s`；它是compiler约束，不是memory fence。没有在asm里执行DS load，也没有在每个scale consumer新增硬件wait。

作用域仍为共用selector0、t256/w2x4/b4/K7168、Ntask1、rowtail3/stagedScaleA/halfA/halfScale/rowaload2/pairA/M-order1/group8，默认关闭`AITER_GFX1250_GEMM1_SCALE_BASE_TIED=0`；不按E64/E65真实expert数新增specialization。actual AST审计172032项scale地址/读取顺序、688128 B及9216项modulo32 consumer cases通过；44个nested函数、全部numeric/pipeline/TDM/fences、每个有效accumulator的56次原序K128更新及GEMM2 AST不变，关闭完整module AST精确默认。receipt=`unified_scale_base_tied_source_audit_v1.json`，wrapper/persistent SHA=`f82f9c368249f75e645ccfe6f3443b620c9be1b81e6422a8636aab12073e7c95`/`a00238736fa197ee821315d96633cf6ac5c485be773ee411130f8f47219f1098`。

CDNA5 ISA §15.15 pp721–722定义`DS_LOAD_2ADDR_STRIDE64_B32`的两个offset分别乘256 B，不能当成连续b64。Shader Guide §5.7.6 Table86记载LDS服务率、latency及bank conflict影响，但未给当前2addr pair的确定加速结论。因此先a07-3/dsr1_ep4固定mode0→1 no-dispatch编译、每项120秒：必须`.text`实际改变、scale DS issue sites减少且总指令不增长、zero spills/private、原scale dwords/numeric/payload/output/TDM/fences保持、同compiled object/完整ELF/count ABI保持并通过physical DS CFG。之后才双seed完整GPU精度和固定完整三轮性能；当前尚无本项device或性能数据，live默认及正式107.379/95.3059 μs保持。

#### Scale base tied v1实际编译与独立native验收

a07-3 host actual AST复核rc0；dsr1_ep4固定mode0→1 no-dispatch编译均rc0、120秒内，每mode E65→E64→E65 cache1/同compiled object/完整ELF。关闭完整ELF精确默认；candidate ELF=`db4ebc4eeae3df6cbc9a1aaed9f654f779a39ca9efdf4175b20fa2db4e92a8ee`、`.text` SHA=`fb825c5cc34244c09cb31a065ce2eaa5b2cc74898cf9be92994ba86b90a17f47`，指令4531→4404、VGPR486→496/SGPR68、zero spills/private，LDS282624 B、kernarg180 B/count offset176 size4保持。

42条独立scale `ds_load_b32`全部合并：scale DS issue69→48、总96 dwords保持；`s_set_vgpr_msb`375→312、normalized `v_add_nc_u32 -34`、compiler DS wait sites -15。VOPD两个half拆开统计后全部numeric/payload/output/TDM操作数保持；eight backedges及四owner targets、原hardware fence序列、physical DS RAW/WAW fixed-point模型均通过。scale pairing减少静态sites并不自动保证服务rate/μs收益，VGPR增加10的影响也须GPU性能确认。

两mode五件套、原run/result/manifest/source/idle及queue/control logs已下载，`check_unified_scale_base_tied_native_download_v1.py`独立重算SHA、ISA逐条编码、binary metadata、embedded launcher ELF、共用count ABI和DS/fences通过。receipt=`unified_scale_base_tied_native_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`c975419b1a20623548cf708a638f29f1cc02ef2400f231bd2318b399bb3cac5a`；原qualification SHA=`b7fbe8f3ea4c08e7b678d14a231efff793f3061cc29121fee233f4acb356bed7`。初checker误将过滤后的host audit stdout与完整JSON比较，仅改工具字段投影并保留diagnostic，没有改候选或重跑编译。native队列rc0关闭，post无本项owner。接下来固定seed0→17、每进程E65→E64完整112项及原MoE reference验证，计时无效；正常live imports另绑定后才固定三轮正式性能，live默认尚未改。

#### Scale base tied v1双seed双规模完整GPU精度

固定seed0→17、每进程E65→E64已经在a07-3/dsr1_ep4完成，两程序均rc0、120秒内；每规模每seed28项、共112项有效payload/ScaleA/GEMM2/MoE逐字节比较均零差异，eager/graph canary与三组same-address dynamic A/ScaleA/psum replay通过。四次原MoE reference通过：E65 seed0/17 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`，E64为`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`，原精度阈值保持。

两进程E64/E65 cache1、同compiled object/完整ELF=`db4ebc4eeae3df6cbc9a1aaed9f654f779a39ca9efdf4175b20fa2db4e92a8ee`，与native验收一致。全部原run/result/manifest/source/idle、actual binding和queue/control logs已下载，`check_unified_scale_base_tied_correctness_download_v1.py`独立重算后通过。receipt=`unified_scale_base_tied_correctness_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`00eb9b7c96c4ebcaa30b57d1679e99d1cecba9867e74b4e1c466995d974302f3`。correctness-only所有耗时无效，post无自身GPU残留，live默认保持。

下一步`run_unified_scale_base_tied_dual_perf_v1.py`固定默认E65→默认E64→候选E65→候选E64→默认E64→默认E65，各组三轮；只有ownership污染可整组最多三次尝试，execution/precision/reporting失败立即停止，finally恢复默认。candidate正常live入口先额外双规模56项及完整ELF绑定，性能前全机idle、计时期间不并发监控/传输/编译/ATT。只有两个规模的GEMM1相对两侧控制均改善且e2e均不回退，才允许一次固定反向确认；原first-zero队列仍关闭。性能checker在启动前移除旧实验的结果特定断言并保存preparation correction，不改变新实验计时政策或精度。尚无本项正式性能，正式107.379/95.3059 μs保持。

#### Scale base tied v1正式性能收尾：E65 e2e未满足保留条件（2026-10-20）

原固定六组队列rc0关闭，共27轮原random non-balanced完整MoE，执行/精度全部通过、各项120秒内。候选normal live imports额外双规模56项零字节比较与同compiled object/cache1/完整ELF绑定通过。三组ownership污染均整组三轮排除并按原规则重试try2，没有拼轮、按耗时追加、修改计时或重开已关闭队列。

| workload | 完整合格组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前控制try2 | 107.246 /107.670 /108.199 | 107.670 | 92.9238 | 255.936 |
| E64/topk8/shared0 | 默认前控制try2 | 97.1511 /94.2014 /97.2521 | 97.1511 | 60.4268 | 210.097 |
| E65/topk9/shared1 | candidate try1 | 108.071 /107.645 /107.335 | 107.645 | 95.3654 | 258.103 |
| E64/topk8/shared0 | candidate try2 | 95.9156 /94.7494 /94.1993 | 94.7494 | 59.3196 | 205.623 |
| E64/topk8/shared0 | 默认后控制try1 | 95.6984 /96.3940 /95.7447 | 95.7447 | 60.3906 | 207.689 |
| E65/topk9/shared1 | 默认后控制try1 | 107.520 /108.461 /107.883 | 107.883 | 94.7685 | 258.869 |

E65 candidate GEMM1相对两侧控制为-0.023219%/-0.220609%，MoE e2e为+0.846696%/-0.295903%；E64 GEMM1为-2.472128%/-1.039535%，e2e为-2.129493%/-0.994757%。E65 GEMM1改善不足以认定有稳定收益，且e2e比前侧控制回退，因此事前双规模/e2e门槛未满足，候选淘汰，不追加反向确认，不接入生产。GEMM2源码和AST保持；它的计时波动不能单独归因于本项GEMM1地址改动。

| 排除整组 | GEMM1三轮原读数 μs | 污染轮次与原证据 |
|---|---|---|
| E65默认前控制try1 | 107.822 /108.446 /107.632 | run2外部live owner PID2924373，原日志为OPTX replay |
| E64默认前控制try1 | 95.9936 /96.1876 /96.6388 | run2 post已退出PID2942092及GPU活动，settled后恢复idle |
| E64 candidate try1 | 93.3286 /96.3985 /95.6099 | run1外部live owner PID2963818，原日志为OPTX edge correctness |

这些组的GEMM1/GEMM2/e2e中值均为null，完整raw保留。没有杀死或干预外部任务。性能结束的2026-10-19T18:57:25Z post全机idle、KFD空、busy0、VRAM176582656 B，无本项GPU残留；后续性能前重新检查。

全部run/result/manifest/source/idle、plan/summary、queue/control logs、normal live binding与transition receipts已下载；raw包=`unified_scale_base_tied_perf_raw_boot1d5146ac_v1.tgz`，1111105 B，SHA=`df579eafa7a2630213da08bcf0ec8906a0246b1033aeea7964519d59b439cad0`。独立checker v1误要求所有排除组也无live owner，在CPU验收中失败；保留原checker和`unified_scale_base_tied_perf_checker_excluded_owner_failure_v1.json`，v2改为匹配原owner列表并重算污染组，只要求合格轮及最终post owner为空。候选、原计时/状态/排除规则全部保持，未重跑GPU队列。

`check_unified_scale_base_tied_perf_download_v2.py`独立重算27轮、六组合格中值、三组排除、SHA/命令/精度/ELF及ownership通过。receipt=`unified_scale_base_tied_perf_downloaded_acceptance_boot1d5146ac_v2.json`，SHA=`7cfe554c62cfb5fe934d1fc966942bfb5430ce3e178d32d36bf0caabc9db6273`；原summary SHA=`949dfcdc4c1cbff835285c7236a7add06f8ac69344c46f4bd5c5f2e6bdb6c86a`。finally恢复精确默认wrapper/persistent SHA=`798a865d7c11a0914b3c335d141948cbf77384d12135e822e339487b474e60b1`/`29cfe04e826aa2617c4c1f57cb4fd0a0211f615ca14715a5bba227b35cda7d64`。正式保留E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs仍27.379 μs；上表为本轮完整对照，不把默认测量的随机差异记作代码收益。

离线分区=`unified_scale_base_tied_native_regions_v1.json`：loop外指令3750→3630、scale DS sites45→24、MSB247→176；四个steady owner loops合计只少7条指令，每loop MSB32→34、scale DS sites均仍6。这说明静态收益主要在prime/drain，并未证明steady提速或这次e2e波动的硬件原因；尚无本候选ATT。下一独立方向将tied scale identity限于prime/fixed drain，保持steady原地址路径，再依次验收native、完整精度与正式性能。此时新方向尚未实现或测量。

#### Prime/drain限定scale identity：native拒绝（2026-10-20）

独立`unified_scale_fixed_tied_v1_source/`仅在`prime_stage`与constexpr drain的scale LDS base使用空LLVM tied identity，steady保留原表达式。actual AST审计172032项地址/读取顺序、18432项modulo32 consumer cases通过；关闭module精确默认，GEMM2、56次原序K128及hardware fences保持。开关=`AITER_GFX1250_GEMM1_FIXED_SCALE_TIED=0/1`。

a07-3/dsr1_ep4固定两项no-dispatch编译均rc0/120秒内，双规模同compiled object/cache1/完整ELF、zero spills/private保持。candidate ELF=`91f11aa4920a2a0bf7c1ecd2f4e3594c5d1e33c6b8195de8516687abcd11cd23`，4531→4417 instructions、VGPR486→498、SGPR68/LDS282624 B/180 B runtime count ABI保持。但四个steady owner loops每个均MSB32→34、指令+1，未通过事前steady不增长门槛，因此淘汰，未dispatch或测试性能。源码局部改动不能保证LLVM全局寄存器分配局部不变；静态总指令减少不构成性能收益。

raw包=`unified_scale_fixed_tied_native_raw_boot1d5146ac_v1.tgz`，409054 B、SHA=`5b2cb8265e34998294a4cfe7769958815941c8d0c6ac66eae28548213f025d03`。完整两mode五件套/run/source/idle/原门槛已下载复核，拒绝receipt=`unified_scale_fixed_tied_native_downloaded_rejection_boot1d5146ac_v1.json`、SHA=`a10cfb19728e33d758fd3295f8220c36866bbc90ec3a594777be7446a098b2ea`。初CPU Restore工具误识别GEMM2同名helper，新参数存在时才删除的修正及失败记录保留；候选没有修改。生产默认不变。

#### 初始input issue：现有ATT分析及rolled-loop关闭（2026-10-20）

复用精确默认ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`已经验收的SIMD0/1、四SE ATT，没有新采集。对896-WMMA sampled waves，以下均为wave cycles或执行指令数中值：

| 指标 | SIMD0（20 waves） | SIMD1（19 waves） |
|---|---:|---:|
| entry span cycles | 7045.5 | 6364 |
| first→fourth input TDM issue cycles | 1833 | 2497 |
| first→fourth input TDM区间指令数 | 178 | 134 |
| fourth input TDM→first WMMA cycles | 3404 | 2981 |

first→fourth input TDM区间有12条重复owner CBRANCH以及SALU/descriptor工作。分析=`analyze_unified_default_entry_prime_v1.py`、`unified_default_entry_prime_analysis_v1.json`（SHA=`2fb26c6359caeb9ced7bffc6f8fbcf5168e2c1d6ffb17c0a41f94f6587bd711b`）。这些是issue间隔，不是TDM completion latency或kernel μs；独立中值不可相加，也不能唯一归因硬件瓶颈。Shader Guide §4.10.1 p197记载TDM忽略EXEC、同wave有序、跨wave及相对其他memory无序，因此保持原tensor wait及WG ready/release/output fences。

独立`unified_prime_loop_v1_source/`尝试owner选择一次后runtime四轮initial issue。768项actual dispatch AST cases、1872 copy events及各owner原序四stage检查通过，issue/descriptor/56次K128/steady/GEMM2/fences AST保持。a07-3两项no-dispatch编译rc0/120秒内，关闭精确默认；candidate ELF=`927950a5b4d83764b2f992778e101219d5bee78125731e9af298fb98d2183b7d`，4531→4273 instructions、VGPR486/SGPR68、zero spills/private。实际LLVM完全展开四轮，只有原8条steady backedges，static input TDM sites仍20，未满足本版预设四个pure-TDM loops/8 sites门槛；原队列rc1关闭，不修改门槛或重开，不运行GPU精度/性能。

raw包=`unified_prime_loop_native_raw_boot1d5146ac_v1.tgz`，470144 B、46 regular members、SHA=`977ee62d9b09a2bad35852785258de2167cf041439a709f0d62ba553364646dda`。完整下载逐member SHA、binary/ISA/launcher、source/run/timeout/count ABI及原拒绝独立复核通过，receipt=`unified_prime_loop_native_downloaded_rejection_boot1d5146ac_v1.json`。静态少12个owner branches仍是下一独立实现的假设，不能据此恢复本版测试。

#### 显式四stage owner选择一次：独立源码/native实验（2026-10-20）

新snapshot=`unified_prime_owner_v1_source/`、开关=`AITER_GFX1250_GEMM1_PRIME_INPUT_OWNER=0/1`。在同一canonical scope中先`dispatch_wave_job(prime_inputs)`，helper内用`range_constexpr(PRE)`显式按stage0→3调用原`issue(..., my_jobs)`。这版直接保留四个owner各四处TDM，而不是要求runtime循环；事前门槛为原20 static input sites、不新增backedges、四stage内部无重复owner branch、entry/总指令减少、steady MSB/指令不增长、原数值/descriptor/fences/共用ABI及zero spills保持。旧rolled-loop队列保持关闭。

wrapper/persistent SHA=`886cd23edba68c94dde832b9265dff9312fe3ef37fcf0cc72516799d4ee7b8ec`/`23b0f6ea9c2f41e46cbd831b9c0e2a808b41839cf141add49360a63470b14253`。本地与a07-3的actual AST审计768 cases/1872 copy events通过，关闭完整module精确默认，issue/descriptor/steady/数值56次原序K128/GEMM2/fences保持。固定no-dispatch两项rc0/120秒内，candidate ELF=`036059fecc06b3550428098de8f8baa874f580ab132cb36399613385f2c0e262`、4273 instructions、VGPR486/SGPR68、zero spills/private，E64/E65同compiled object/cache1/完整ELF。远端native门槛通过，继续下载独立核查、双seed/双规模完整GPU精度，再normal live binding和固定完整三轮性能。目前没有该候选正式性能；默认E65/E64 GEMM1仍107.379/95.3059 μs。

#### Prime owner v1完整精度及独立验收（2026-10-20）

native raw包471309 B、46 regular members，SHA=`deec5bdba60e705d1893b1ee9dcfb431053a282e950ce8b835068a6fdca39c11f`。独立下载重算source/run/binary/ISA/metadata/launcher与事前gate通过，receipt=`unified_prime_owner_native_downloaded_acceptance_boot1d5146ac_v1.json`、SHA=`3a5b0abed3153bd13d850ca48e5663ad6905ca4e7b48794de6ca1a25e21bfa12`。first DS前static instructions874→618；四个constexpr owner组first→fourth TDM静态区间指令由138/80/80/80降到12/15/15/15，组内重复owner branch各3→0。这里是静态路径长度，不能冒充ATT wave cycles或kernel μs；生产的有效route/每owner四次TDM与56次原序K128保持。

固定seed0→17，每进程E65→E64完整验证均rc0/120秒内，共112项`differing_bytes=0`，包括有效payload/ScaleA/GEMM2/最终MoE、eager/graph canary以及三组same-address dynamic A/ScaleA/psum replay，原空expert和M16/M32/M128边界检查保持。双规模实际同compiled object/cache1/完整candidate ELF=`036059fecc06b3550428098de8f8baa874f580ab132cb36399613385f2c0e262`。原MoE reference阈值不变，E65 seed0/17 logits_diff=`3.7793e-06/3.7788e-06`，E64=`3.4733e-06/3.4766e-06`。

correctness raw包217906 B、29 regular members、SHA=`22802d3bfeb78caba7d2e2900df00df7cb4159429ec46ffb80d68657b441ad88e`，逐member SHA通过。CPU checker v1错误地要求correctness-only时也无外部owner而失败；保留v1及`unified_prime_owner_correctness_checker_external_owner_failure_v1.json`，v2只改为绑定原外部owner日志并确保没有本任务残留，112项数值/reference/replay/实际ELF检查保持，没有重跑GPU。原外部PID3402170执行`verify_runtime_n256_ring_edges.py ... optx_v231_n256s2_edges5_10_20261020_r1`，没有干预，所有验证耗时无效。

下载独立精度验收=`unified_prime_owner_correctness_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`f9607b906df4165e95e9c28de3abafea00451939c1852348cf757a212ead1ddb`。下一固定性能队列为默认E65→默认E64→候选E65→候选E64→默认E64→默认E65，每组三轮；仅ownership污染可整组最多三次，执行/精度/解析失败立即停止。normal live入口另做双规模56项/共用完整ELF绑定，finally恢复精确默认。每轮全机idle门禁、计时中不并发本项监控/编译/传输/ATT。只有两规模GEMM1相对两侧控制均改善且e2e均不回退，才做一次事前固定反向确认。此时尚无prime-owner正式性能，保留107.379/95.3059 μs。

#### Prime owner v1正式性能在默认前控制停止（2026-10-20）

原六组固定队列rc1关闭，只执行了E65默认前控制的三次完整三轮尝试。九轮执行及原random MoE精度均通过、单项未超过120秒；三个完整组均未通过ownership门槛，全部GEMM1/GEMM2/e2e中值为null。不拼接合格单轮、不追加第四次尝试、不重开该队列。候选normal live binding及性能尚未执行，先前已通过的native/112项完整精度验收保持；不能把本次无候选计时写成已证明候选回退。

| 排除整组 | GEMM1三轮原读数 μs（全部不作性能） | 排除原因 |
|---|---|---|
| E65默认前控制try1 | 107.531 /109.306 /107.107 | run3 settled时外部PID3450241运行OPTX E64前控制 |
| E65默认前控制try2 | 107.227 /107.427 /107.391 | run1 post有外部live owner PID3455048运行OPTX E65前控制 |
| E65默认前控制try3 | 107.750 /112.490 /111.338 | run1/run3 settled有外部PID3471867/3479459运行OPTX候选/后控制 |

没有干预外部进程。finally核对两源精确默认并恢复；queue post记录了外部OPTX owner，不能宣称当时全机idle。后续独立打包检查没有本项live GPU owner。64项raw包=`unified_prime_owner_stopped_perf_raw_boot1d5146ac_v1.tgz`、458087 B、SHA=`31083e833fbc2397a9236243521cdb0a2e78eddcde2f9c05536e5937ce41a983`。`check_unified_prime_owner_stopped_perf_download_v1.py`独立重算全部九轮命令/精度/日志SHA/ownership/整组排除/默认恢复通过，receipt=`unified_prime_owner_stopped_perf_downloaded_acceptance_boot1d5146ac_v1.json`、SHA=`9133e4c005c483a099830273946661dde7aca88d45bdbc4ab172bcf48704e17f`。修正早期段落完整ELF多写一个`3`的转录错误，未改任何原raw或receipt。

#### 原两组ScaleA slots下的M64无consumer读取裁剪（2026-10-20）

新独立snapshot=`unified_half_scale_guard_v1_source/`、默认关闭`AITER_GFX1250_GEMM1_HALF_SCALE_GUARD=0/1`。仍用原两个2-wide FRONT/BACK ScaleA slots，prime及future读取保留每个可能执行的M64 half的两条原scale dwords；仅当该half首行`wmb + half*64 >= mn_oob`时跳过整个half读取。至少一个M32 consumer有效时，两条scale仍全部加载。原逐pair scale裁剪版本曾增加scalar分支且性能无收益，保持关闭；此项不改成四组pair slots，也不重测prime-owner。

actual AST完整pipeline审计1040 cases、58240次K128 step、494592次有效M16/N32 accumulator更新通过，原56次更新顺序、每个有效A/B/ScaleA/ScaleB consumer、所有TDM/硬件同步事件保持。CPU边界枚举ScaleA loads116480→72576，仅跳过43904次padding reads；这不是目标routing的实测计算量或kernel收益。关闭完整module AST精确默认，GEMM2/canonical ABI/有效routes保持。source wrapper/persistent SHA=`f028634bef39a06b50cd1b63f23b8c99215cad3787e9db3217133c1e2f87ba1c`/`8c840165aea9f4d203be932cc6f751979429d47f6a5e2ff4fc1b53755bc83cbd`。

依据Shader Guide §4.10.1 p197及§5.7.6 Table86，TDM/其他memory无序、LDS服务率受共享和bank conflicts影响；本候选保留原READY/release/output fences，不能由读取减少推断确定硬件加速。先固定mode0→1 no-dispatch编译、单项120秒，要求同compiled object/完整ELF/count ABI、zero spills/private、VGPR≤498/SGPR≤70、原LDS/backedges/数值/同步以及每steady loop至多新增四个scalar conditional branches。通过后才完整GPU精度与固定三轮性能；此时生产默认和107.379/95.3059 μs正式成绩保持。

#### M64 ScaleA guard：前端超时诊断、v2 native拒绝与证据收尾（2026-10-20）

v1固定编译队列在mode0的candidate module `@jit`初始化阶段达到120秒，被host runner终止，rc124关闭；尚未建立artifact目录或调用launcher，mode1没有运行，无GPU dispatch。精确默认模块的独立import/FakeTensorMode/null-pointer诊断rc0、约2.47秒完成；只给原compiler增加phase prints/faulthandler、移除诊断后完整AST相同的诊断项再次rc124，栈停在`flydsl/compiler/ast_rewriter.py visit_BoolOp` / `ast.fix_missing_locations`。新配置校验有15项连续`and`。v2仅改为tuple equality与三个短assert，数值/读取裁剪/TDM/fences AST保持；实际两mode编译均rc0、120秒内，支持长BoolOp校验是本例阻塞因素，不扩大为FlyDSL通用根因结论。

v2双规模E65→E64→E65仍cache1、同compiled object/完整ELF，关闭完整ELF精确默认。candidate ELF=`162fba81ce6919f11ba911b40e4dc1ecd7b7d39583937307795a2694784da1fd`，4531→4674 instructions、VGPR486→487/SGPR68→70、LDS282624 B/count ABI保持，但private16 B、VGPR spill4；依据原zero-spill/private门槛拒绝，native队列rc1关闭。未运行该候选GPU正确性或性能，不替换生产默认。

四个steady owner loops各conditional branches9→11、MSB32→44、instructions约+21/22；这在原至多新增四个branch门槛内，但带来明确静态开销。12个scratch sites位于最后drain的PC `0x53a8`至`0x5ed8`，包含四个store、八个load。原数值/payload/TDM opcode数量、96个scale dwords保持；此处不是目标routing的动态流量或性能测量。第三条native拒绝已追到工具要求` s_wait_dscnt 0x0`恰好11条，而candidate有15条；没有因此证明device时序race，旧拒绝和门槛原样保留。Shader Guide §3.3.5 p37定义scratch为thread-private global memory；不能根据这些静态sites计算kernel μs或唯一确定register allocation根因。

v1 timeout、两项诊断、v2两mode五件套/source/audit/gate/run/result/manifest已归档下载，105个regular members、896841 B，raw包=`unified_half_scale_guard_rejected_raw_boot1d5146ac_v2.tgz`，SHA=`dca4592d0d7e3989eac6a2273a8030c69c37b899ecfa7b966da132892e097809`。`check_unified_half_scale_guard_rejected_download_v2.py`逐member重算SHA并重新执行原qualifier，拒绝结果完全一致，receipt=`unified_half_scale_guard_rejected_downloaded_acceptance_boot1d5146ac_v2.json`，SHA=`1bb38f0ef55b7d4f3110301fb57b0c0ba15aebcd3dda5c90b6ddf09992497d35`。两源码SHA仍为精确默认，无本项compile/GPU进程残留；旧队列不重开。

下一独立修订只在无任何consumer的M64 ScaleA half中定义两项占位零值，避免条件未命中时继续保留旧slot的SSA值。它不使用const0输入、不改变有效scale或有效WMMA/K顺序，旨在验证死值生命周期是否解释spill；目前仅是待验收假设，事前zero-spill/原分支/数值/同步门槛保持。正式默认E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs差27.379 μs。

#### M64 ScaleA guard v3：死值定义仍产生spill，独立队列关闭（2026-10-20）

上述假设现已完成验证。v3在无任何M32 consumer的M64 half中，向原两个2-wide ScaleA slots写入两个占位`fx.Int32(0)`；有效scale、数值顺序、TDM及硬件同步保持。实际AST审计1040 pipeline cases、58240次K128 steps、494592次有效M16/N32更新通过，每个有效accumulator仍执行56次原序K128更新；边界枚举仅裁剪43904次padding ScaleA reads，不代表目标routing的实测收益。

固定mode0→1实际编译均rc0、120秒内，关闭完整ELF精确默认，各mode E65→E64→E65仍同compiled object/cache1。candidate ELF=`02bf47b5dff29ff697badfcc2f00f668b8f595d4986047cbf73a16458654e802`，4705 instructions、VGPR486/SGPR70、LDS282624 B/count ABI保持，但private20 B、VGPR spill5；15个scratch sites位于drain PC `0x5104`至`0x5c74`。原zero-spill/private门槛拒绝，队列rc1关闭，未dispatch候选或测性能。额外DScnt0仍触发原静态计数assert；这不能作为已验证device race。定义死值没有消除spill，不继续重开这条路线。

raw包=`unified_half_scale_guard_rejected_raw_boot1d5146ac_v3.tgz`，94 regular members、852479 B、SHA=`467ab130d65b9a385e60671e843a493e64f9f68f71c40bae0063a8e319a9388d`。逐member SHA、两mode native/source/audit/run及原qualifier独立复核通过；receipt=`unified_half_scale_guard_rejected_downloaded_acceptance_boot1d5146ac_v3.json`、SHA=`73e4ecb1a3b3ff9209624d908a8ddb9e120c54cb80b37ee62f39d74de06ed78d`。v3 checker复用v2的原规则，个别输出标签仍保留v2文字，实际验收绑定v3文件及hash。

继续方向是runtime连续M32分配：例如192有效行从原128/64改为96/96，full256保持128/128。原input TDM owners、LDS ring、硬件fences和私有output LDS slices保持；输出global起点随实际行起点移动，TDM tensor extent裁剪每wave拥有的M16/M32 groups。Shader Guide §4.10.2 p199和§4.10.7 p205记载超tensor正方向边界的stores丢弃；仍需实际地址/coverage、native、完整MoE精度验证，不由分配更均匀推断加速。准备脚本v1误用`@jit`定位GEMM2 decorator，实际为`@flyc.jit`，因此在persistent写出前停止，仅留下wrapper partial snapshot；未上传、编译或dispatch。原失败及partial保留，使用独立v2修正准备定位。正式默认和107.379/95.3059 μs保持。

#### 连续M32行分配：native及完整精度通过（2026-10-20）

独立候选`unified_contiguous_m_split_v3_source/`使用真实有效行数计算`split_m=((clamp(mn_oob,0,256)+63)>>6)<<5`、`wmb=wave_m*split_m`。第一个M wave的有效结束为`min(mn_oob,split_m)`，第二个保持`mn_oob`；每wave有效行数裁剪到0–128。164行由128/36变为96/68，192行由128/64变为96/96，224行及full256分配保持。它改变wave之间的工作分配，全部有效计算及K顺序保持，总有效WMMA更新数量不减少，不按E64/E65增加compiled specialization。

input TDM仍使用原`mn_oob`、owners、ring和fences；compute/A-load/quant谓词使用该wave的结束行。每wave向原私有output LDS slice写入，payload的global M16起点为`wmb//16`、extent为`ceil(wave_m_rows/16)`，ScaleA对应M32起点/extent。Shader Guide §4.10.1 p197明确TDM忽略EXEC，本候选依靠实际tensor extent裁剪有效store，不能只靠lane mask；§4.10.7 p205记载超tensor正边界stores丢弃。

准备v2实际编译mode0报`TypeError: Variable(s) ['owned_rows'] initialized as None before a dynamic if/else (scf.if)`，mode1未运行、队列rc1关闭。新temporary名称与原嵌套epilogue local冲突；v3仅把三处新引用改为`split_owned_rows`，原嵌套`owned_rows`不变，地址/数值alpha等价。较早preflight把省略`unchanged_nested_functions`的stdout与完整receipt比较而失败，也已保留，仅修独立CPU preflight，未重跑GPU。

actual AST完整审计通过：1040 pipeline cases、58240 K128 steps，每mode247296次有效M16/N32更新；1794048个实际A/ScaleA LDS地址、7800个裁剪output descriptors、100224个有效输出行。有效M16计算覆盖恰好一次，每有效accumulator保留56次原序K128更新，A/ScaleA地址留在原stage范围，private output slices及有效输出无重叠。包括空expert、-1..257及30464边界；关闭完整module AST精确默认，完整GEMM2 AST保持。

固定mode0→1 no-dispatch编译均rc0/120秒内，每mode E65→E64→E65同compiled object/cache1。candidate完整ELF=`dfca5097cca41db10a2f7a419c196195f48038511b0bdf29a4b58c4fb2bcc2e8`；4551 instructions（默认4531）、VGPR486/SGPR74（默认68）、zero spills/private、LDS282624 B、kernarg180 B/runtime count offset176保持。原8 steady backedges、numerical/DS/TDM/scales/fences及physical DS CFG验收通过，mode0完整ELF精确默认。native raw79 members/637281 B已下载逐SHA独立复核，包SHA=`0f1f69f0a7cafadd0c245a8aaa7cbe13b4a6121b64e4f78f4e3ebb9162fa4b1d`，独立receipt SHA=`c51e93ad7a9e9ecd444fdc31001982615b7782c7c602f950bb937250003808d0`。

seed0、17各进程E65→E64完整GPU精度均rc0/120秒内，112项逐字节比较全部零差异，四次原MoE reference pass=True、原`logits_diff<0.01`不变。E65 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`；E64为`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`。eager/graph canary、same-address动态A/ScaleA/psum与empty experts通过，实际加载E64/E65同compiled object/cache1/上述完整ELF。

GPU精度raw34 members/223123 B下载逐member SHA通过，包SHA=`525f9b6b07dba3369ef2fddc58e82f3fb26839d3d5d4d1a57b06053c27d91b68`。原download checker v3误把外部OPTX PID36233当成本项残留；独立v4又错误假定该owner在seed0初始idle snapshot已存在，均保留失败和工具。v5依据原pre/post/settled记录精确绑定外部命令及本项无live残留，保留原112项/reference/replay/source/ELF门槛，不重跑GPU。独立精度receipt=`unified_contiguous_m_split_correctness_downloaded_acceptance_boot1d5146ac_v3.json`（记录checker revision v5），SHA=`8f5421dd470967f2ae4a7f4b9edcc324ee812f584c79a31913d47be9928d5cf8`。所有精度运行均correctness-only、耗时无效。

正式性能完成固定默认E65/E64→候选E65/E64→默认E64/E65各三轮，normal live入口另做双规模56项零字节比较与实际ELF绑定，finally恢复默认。空闲前每6秒等待，单项120秒timeout，计时中不并发GPU监控/传输/编译/ATT；仅ownership污染允许整组最多三次，执行/精度/解析失败立即停止。完整队列rc0，27轮原benchmark均执行和原MoE精度通过、未超120秒；三组ownership污染按事前规则整组排除，未拼轮或按耗时重试。

| workload | 合格完整组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前try3 | 109.356 /109.702 /107.003 | 109.356 | 94.4794 | 257.486 |
| E64/topk8/shared0 | 默认前try1 | 96.0438 /98.8669 /98.6477 | 98.6477 | 61.3835 | 209.571 |
| E65/topk9/shared1 | 连续M32候选try1 | 107.347 /107.157 /108.134 | 107.347 | 94.9286 | 257.322 |
| E64/topk8/shared0 | 连续M32候选try2 | 95.4218 /94.0682 /94.4097 | 94.4097 | 59.7518 | 205.230 |
| E64/topk8/shared0 | 默认后try1 | 94.1947 /94.3053 /96.2105 | 94.3053 | 59.5934 | 205.283 |
| E65/topk9/shared1 | 默认后try1 | 107.417 /107.623 /107.651 | 107.623 | 94.6502 | 257.188 |

候选E65 GEMM1相对前/后控制为−1.837119%/−0.256451%，e2e为−0.063693%/+0.052102%；E64 GEMM1为−4.296096%/+0.110704%，e2e为−2.071374%/−0.025818%。两规模相对两侧控制均改善、e2e均不回退的两项事前保留条件都未满足，候选淘汰，不启动反向确认、不追加计时、不接入生产。小幅差异不足以唯一归因具体硬件瓶颈。

整组排除保留原值：E65默认前try1 GEMM1为107.689/109.365/109.890 μs，run2外部OPTX live PID122111；try2为106.763/108.317/107.743 μs，run1/run3 settled外部PID129411/136869。E64候选try1为96.6047/94.3635/95.3839 μs，run2外部CPU compile初始化KFD PID162742。全部这些组的中值为null，没有干预外部进程，不能拿其中较快单轮替换合格组。

完整raw包`unified_contiguous_m_split_perf_raw_boot1d5146ac_v3.tgz`有179 regular members、1207886 B，SHA=`69253c786c581fe52d66319306fd88140a1267ff1069e40b7566ccfecd07dbb6`。下载逐member SHA后由`check_unified_contiguous_m_split_perf_download_v3.py`独立复核27轮原命令/精度/完整分组/ownership/正常入口共用ELF/默认恢复通过；receipt=`unified_contiguous_m_split_perf_downloaded_acceptance_boot1d5146ac_v3.json`，SHA=`e5308739e2964355dfaa51535889a25c7337a737cd16c54e80c8969f36f2fdb6`。

队列后及独立打包后检查全机KFD空、busy0、VRAM176582656 B，无本项live GPU残留。默认wrapper/persistent SHA仍`798a865d7c11a0914b3c335d141948cbf77384d12135e822e339487b474e60b1`/`29cfe04e826aa2617c4c1f57cb4fd0a0211f615ca14715a5bba227b35cda7d64`。正式最佳E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs差27.379 μs，不将本轮默认测量波动记作代码收益。后续诊断采集精确默认kernel的GL2C counters，所有profiling耗时无效，逻辑effective read bytes不称为真实HBM bytes。

本项复现工具均在`.codex_tmp/dsr1_e65_gemm1_20261017/`；已有labels不可覆盖。实际运行命令为：

```bash
# a07-3 host；编译与GPU程序由runner进入dsr1_ep4
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_contiguous_preflight_compile_v4.py
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_contiguous_correctness_host_v3.py
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_contiguous_perf_host_v3.py
```

这些是已执行的固定队列命令；新复现必须另建label及receipt目录，并保留相同精度、计时与ownership规则。

#### 精确默认GL2C诊断与A multicast timeout候选（2026-10-20）

SDK `rocprofv3 --list-avail`列出gfx1250的`GL2C_READ_SECTORS/HIT/MISS`；可用定义的dimensions为`DIMENSION_INSTANCE[0:95] DIMENSION_AID[0:1]`。依据SDK与本地`subsystem/CMM/GL2/Verification/gl2_perfmon_counter_test_mapping#2.txt`行28–29、57，前者计read requests中的32B sectors，HIT/MISS是cache事件、UC reads计miss。不能把read request量称为HBM bytes，也不能把hit-event ratio称为byte命中率。

固定默认E65完整原workload仅将诊断`--iters`改为2，kernel filter精确完整symbol、请求第10次目标GEMM1；无源码或数值修改。device hook绑定实际默认ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`、34360 B、cache1/同compiled object。采用SDK core/devel相同library路径，启动前记录真实rocminfo并只在profiled进程树使用shim，所有diagnostic耗时与逻辑effective TB/s全部无效。

v1因外层host runner的`argparse.REMAINDER`仍把后续`--`作为自身separator而报`unrecognized arguments`，未启动container/GPU；raw和工具保留。v2在container内从plan读取完整argument vector修正，仅启动方式变化。GPU profiler/workload约9.623秒正常rc0、原MoE logits_diff=`3.7793e-06`、rel_l2=`0.0027493`，原0.01门限保持。host收尾对container root输出`chmod`报权限错误、外层rc1，原finished记录已保存GPU rc0及ownership合格；仅以`sudo -n chmod -R a+rwX`修复本项输出权限，没有重跑GPU或覆盖raw。

| 计数项 | 原始值 | 有限解释 |
|---|---:|---|
| GL2C_READ_SECTORS | 46045030 | read requests中的32B sectors，对应1473440960 B |
| GL2C_HIT | 1718564 | cache hit events |
| GL2C_MISS | 4127923 | cache miss events，含UC reads |
| HIT/(HIT+MISS) | 29.394814% | event比例，不是byte命中率 |

CSV只有一个dispatch148/Agent2/Queue2/Process71501的三项聚合结果，grid487424、WG256、LDS282624 B、scratch0；未保留每instance/AID breakdown。profiler所报VGPR248/SGPR128不能替代actual ELF metadata486/68。本次requested iteration10、hook观察到10次GEMM1，尚未独立证明它属于哪一个warmup/profile phase，因此只作默认kernel memory层诊断，不用于正式计时或反推带宽。立即post保留短暂已退出PID339240/KFD及VRAM，live owners为空/fuser空；六秒settled恢复KFD空、busy0、VRAM176582656 B，无本项残留。

下载原CSV、完整日志/idle/plan/finished/actual binding后独立`check_unified_default_memory_counters_download_v2.py`逐SHA和完整symbol/source/ELF/精度/ownership复核通过，receipt=`unified_default_memory_counters_downloaded_acceptance_boot1d5146ac_v2.json`、SHA=`8aa532480547e07515a61eddf1129246f65314625e8858b4c20726724c1db57d`。1.473GB为L2请求量，现有三项counter不能唯一分离A/B/scale、next-level请求或HBM饱和。

下一独立`unified_a_mcast_timeout_v1_source/`只将GEMM1 A payload multicast的descriptor `early_timeout`关闭，使用standard timeout；其他job、地址、mask、LDS ring、K顺序、全部fences和GEMM2保持。Shader Guide §4.9.8 pp191–192明确晚到wave在early timeout下另产生L2请求，§4.10.4 Table80 p203定义descriptor bit21；这些事实给出可验证假设，不证明当前workload有收益。E64/E65仍共用runtime selector0/同ELF；配置偏离共同默认时实验直接拒绝，不新增实际expert数specialization。

source wrapper/persistent SHA=`7e3258fab74bd7d70b0fb9f476924783ca67329d1545c9553cd54799484553de`/`67713b5070a03758afb5ef568d46c07b9f57c6a4e4c68b026942ee7e4dba09f6`。actual AST审计7168 timeout cases/896项A timeout变化通过；其余所有input地址/masks/retirement/fences相同，关闭完整module AST精确默认、GEMM2 AST保持。原pipeline520边界cases/261632次WMMA更新对照通过，每有效accumulator56次原序K128保持。先a07-3/dsr1_ep4固定mode0→1、no-dispatch/单项120秒编译：必须默认完整ELF相同、仅五A descriptor literal变化、register资源原486/68、zero spill/private、原LDS/canonical ABI与数值/DS/TDM/fences不变；通过后才双seed双规模精度及固定完整三轮性能。目前该候选无正式性能，保留107.379/95.3059 μs。

#### A multicast timeout：动态参数校验修正（2026-10-20）

v1固定mode0→1的no-dispatch编译队列rc1关闭：mode0 rc0、完整ELF精确匹配默认`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`，mode1 rc1、未超时、artifact目录为空。原报错为`RuntimeError: cannot evaluate dynamic 'Boolean' as Python bool during tracing`，位置是新配置guard的tuple equality。`N`的原声明为`fx.Int32`，不能在tracing中要求Python bool；这是本项候选校验代码错误，没有candidate device dispatch，不是已验证的生产kernel race。原队列、失败日志和mode0产物保留，不重开v1。

独立`unified_a_mcast_timeout_v2_source/`仅将动态`N==4096`检查移到普通Python wrapper，launcher保留其他constexpr配置guard。A payload timeout、descriptor地址/mask、ring、数值和全部fences与v1相同。wrapper/persistent SHA=`d40784f1c54433469486cb40d50aedb9bacf7d253081ff2ff002e6766dc21b19`/`a3afd3706ce1137ca6c3fcc6e207023fe0fee75362c32943faa6d52c348e360d`。CPU actual AST再次通过7168项timeout cases、896项仅A变化、520项pipeline cases和261632次WMMA更新，原56次K128次序、完整GEMM2及共用runtime ABI保持。编译门槛保持并增加descriptor Group1 bit21的literal、encoded word和后续实际TDM descriptor use绑定；不只根据五条`s_mov_b32`变化认定bit正确。

旧非unified E65已试过standard timeout，前后控制差异−0.259%/+0.267%，没有稳定收益，已淘汰。当前新实验针对精确unified默认、要求E64/E65同compiled object/完整ELF及双规模完整验收；它不推翻旧结论。v2已上传a07-3/dsr1_ep4并启动独立两项编译，单项120秒上限。尚无v2 candidate性能，正式E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。

#### A-only multicast timeout v2：native、双规模精度与正向性能（2026-10-20）

固定mode0→1的no-dispatch编译队列rc0，两个程序均在120秒内结束。mode0完整ELF精确匹配默认，mode1完整ELF=`6e3c95affca822922b99b02c0f0a73e1db092af0929dacd6fc14b48aab0ef77c`。两者均4531 instructions、VGPR486/SGPR68、LDS282624 B、zero spills/private、kernarg180 B；E65→E64→E65实际共用compiled object/cache1与完整ELF。native `.text`仅五处A descriptor Group1 literal从`0x07f0000f`变为`0x07d0000f`，xor为`1 << 21`；逐条核对encoded word及后续实际TDM descriptor use，中间无覆写，mask15/data_size1B保持。五处literal PC为8552/9392/9968/10544/12212，使用它们的TDM PC为8572/9416/9992/10568/13040；其他instructions字节一致，原physical DS CFG与native contract通过。

完整92项/1081950 B native raw包SHA=`b48f428e84e85b44069bf11c3aad6d8b3d7f5cd3eec29b5de50c1eed5718e943`；独立下载receipt=`unified_a_mcast_timeout_native_downloaded_acceptance_boot1d5146ac_v2.json`，SHA=`5d5f1f829c9a42425b81730ad01e83da330fd49ffc9ef06d49f17eabd9a2e5aa`。这只证明编译差异与绑定，不单独证明GPU精度或性能。

随后固定seed0→17、每进程E65→E64通过112项逐字节比较，所有差异0；四项原完整MoE reference通过，原0.01门限保持。eager/graph canary、same-address动态A/ScaleA/psum replay与empty experts通过，实际共用上述compiled object/完整ELF。E65 seed0/17 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`；E64为`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`。两个程序rc0且120秒内，正确性阶段耗时全部无效。post出现短暂KFD entries，live owners为空，六秒settled为空，无本项残留。

完整33项/266195 B正确性raw包SHA=`cb36f7ba0ac74a8c76fac1b7e062107e38aa44b0f51bfeecbd4589d450275220`；独立receipt=`unified_a_mcast_timeout_correctness_downloaded_acceptance_boot1d5146ac_v2.json`，SHA=`162002ef745b5cbd1279a81156f5c6fe8c094b0a7a53d9f15dfd00876c6adc32`。

正向完整性能按事前顺序默认E65→默认E64→候选E65→候选E64→默认E64→默认E65完成，队列rc0。六个合格组三轮共18轮；另有一组三轮因ownership检查不合格而整组排除，共保留21轮原benchmark。所有程序执行/原MoE精度通过、均120秒内，计时中不并发监控/传输/编译/ATT，未拼轮或按耗时重试。切换候选后正常live imports另做双规模56项逐字节比较全零，cache1/同compiled object/完整ELF绑定通过。

| workload | 合格组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前try1 | 108.150 /107.838 /107.387 | 107.838 | 94.4942 | 257.131 |
| E64/topk8/shared0 | 默认前try1 | 94.2517 /98.8987 /96.8556 | 96.8556 | 60.8523 | 209.369 |
| E65/topk9/shared1 | A-only timeout候选try1 | 107.067 /108.420 /106.769 | 107.067 | 94.5552 | 256.846 |
| E64/topk8/shared0 | A-only timeout候选try2 | 97.5916 /94.0767 /94.1991 | 94.1991 | 60.0172 | 205.285 |
| E64/topk8/shared0 | 默认后try1 | 97.1317 /96.5668 /95.7254 | 96.5668 | 60.1080 | 208.422 |
| E65/topk9/shared1 | 默认后try1 | 109.030 /107.478 /108.673 | 108.673 | 94.9940 | 258.678 |

E65 GEMM1相对前/后控制−0.714961%/−1.477828%，e2e−0.110838%/−0.708216%；E64 GEMM1−2.742743%/−2.451878%，e2e−1.950623%/−1.505119%。两项正向事前条件通过，允许进行一次固定反向确认；目前不能把这组小幅改善当作已接入默认的新成绩，也不能唯一归因到具体硬件瓶颈。

排除的E64候选try1三轮GEMM1原读数94.5913/95.5463/94.2179 μs，中值null；run1立即post有KFD PID806803/807287、busy1、VRAM5912989696 B，`gpu_users`显示已退出entry，live owners/fuser为空，六秒settled恢复空闲。保守ownership规则仍排除整组，未将其两个合格单轮拼入try2，也未认定这些已退出PID属于某个外部任务。

性能raw包`unified_a_mcast_timeout_perf_raw_boot1d5146ac_v2.tgz`包含145 regular members、1204607 B，SHA=`33060b35f42c7d2e63b660d0bd7d3fa15f708c241de225868b5603055cf22ed3`。下载逐member SHA后由`check_unified_a_mcast_timeout_perf_download_v2.py`独立复核21轮原命令/三轮组/精度/ownership/源码/共用ELF/默认恢复，receipt SHA=`f92944fc3ed9a5dbd6b372f321f14d904fa3b0d48373f61a2962eedda5e55ce8`。finally及独立打包后默认两源SHA精确恢复，全机KFD空/busy0/VRAM176582656 B，无本项live GPU残留。

固定反向顺序为候选E65→候选E64→默认E64→默认E65→候选E65→候选E64，各组三轮。仍只允许ownership污染整组最多三次，执行/精度/解析失败立即关闭；要求反向两侧候选在双规模GEMM1均改善、完整MoE e2e均不回退，否则淘汰且不追加计时。当前反向队列已启动、尚未收尾，生产默认107.379/95.3059 μs保持，E65距80 μs仍差27.379 μs。

本项复现工具在`.codex_tmp/dsr1_e65_gemm1_20261017/`，实际在a07-3 host执行、runner进入dsr1_ep4：

```bash
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_a_mcast_timeout_perf_host_v2.py
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_a_mcast_timeout_reverse_host_v2.py
```

以上队列各执行一次，已有labels和receipts不覆盖。新复现必须另建labels并保留相同完整workload、精度和ownership规则。

#### A-only multicast timeout：反向确认未复现，淘汰（2026-10-20）

固定候选E65/E64→默认E64/E65→候选E65/E64反向队列rc0结束。六个合格组三轮18轮，另五个ownership不合格组按事前规则完整排除，共33轮原benchmark/精度通过、均120秒内。没有按耗时重试、拼轮或追加第二个反向队列。全部原值、process/KFD/VRAM边界和exact source/ELF provenance保留。

| workload | 合格组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 反向候选前try1 | 107.332 /109.848 /110.598 | 109.848 | 94.2371 | 260.194 |
| E64/topk8/shared0 | 反向候选前try1 | 94.4709 /98.0339 /94.7619 | 94.7619 | 61.0988 | 206.862 |
| E64/topk8/shared0 | 反向默认中间try3 | 96.8662 /95.0487 /93.5874 | 95.0487 | 61.5037 | 209.385 |
| E65/topk9/shared1 | 反向默认中间try1 | 110.955 /106.908 /107.853 | 107.853 | 94.6163 | 257.769 |
| E65/topk9/shared1 | 反向候选后try3 | 109.019 /109.565 /107.545 | 109.019 | 94.2159 | 257.384 |
| E64/topk8/shared0 | 反向候选后try2 | 95.9552 /95.1941 /95.2301 | 95.2301 | 59.8613 | 205.985 |

反向E65候选两侧相对中间默认的GEMM1为+1.849740%/+1.081101%，e2e为+0.940765%/−0.149359%；E64 GEMM1为−0.301740%/+0.190850%，e2e为−1.204957%/−1.623803%。双规模GEMM1两侧均改善及e2e不回退未成立，淘汰候选，不接入正式默认、不追加该方向计时。正向107.067/94.1991 μs保持为这次候选的独立批次结果，不挑它替换默认成绩。

五个排除组的原GEMM1三轮为：E64中间默认try1 `95.9805/94.1862/95.6813`、try2 `96.8914/95.3860/94.5110`；E65后候选try1 `110.959/107.296/109.109`、try2 `106.744/106.991/106.426`；E64后候选try1 `94.8232/94.1379/95.2785` μs，全部中值null。原日志记录外部OPTX settled PID906610/917230/928026/957306/973297，以及短暂已退出KFD/GPU状态；未干预外部进程，未将E65污染try2的较快读数用于收益。

完整反向raw包`unified_a_mcast_timeout_reverse_raw_boot1d5146ac_v2.tgz`有219 regular members、1754578 B，SHA=`c57ca81744bcf44d6516b24eb9cb5bb12ffe80eb9a11fbf5adac31d7d899025e`。逐member SHA、33轮原benchmark/profile/完整三轮组/ownership/源码/共用ELF/default恢复独立复核通过，receipt=`unified_a_mcast_timeout_reverse_downloaded_acceptance_boot1d5146ac_v2.json`、SHA=`0c0ff211f0d5370d4734c64da27764837ad5f2f1c83c37111d0569f2d591dc1c`；`stable_forward_and_reverse_benefit=false`。默认两源SHA精确恢复，queue post及独立打包时全机idle/KFD空/busy0/VRAM176582656 B，无本项live GPU残留。

正式默认仍E65 GEMM1/GEMM2/e2e `107.379/94.3843/256.893 μs`、E64 `95.3059/60.4664/209.078 μs`，E65距80 μs差27.379 μs。下一项只做精确默认GL2C EA counters诊断：分两项采集32/64/128/256B请求数及EA_RDREQ/LEVEL/DRAM，实际调用栈额外确认requested第10次GEMM1所属阶段；未修改生产kernel。每项120秒、全机idle门禁/原MoE reference保持，全部profiling耗时无效，GL2→EA加权bytes不能直接等同物理HBM实测bytes或饱和证据。诊断队列已启动、尚无结果。

### E64/T1536/topk8：复制已验收 group8 GEMM1 的独立接入测试（2026-10-18）

本项按用户要求复制当前已验收 GEMM1 到 `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_gemm1_e64.py`。源为 `task_m_group_default_v1_source/` 中的 persistent 文件，SHA256=`d29108153f58a778f6510c201a58e12eaa1a45cabd15a457548aecd5aa5cdae5`；新文件 SHA256=`0dce35d584b593895da7a07fc083acd88c174dbb4931f5d210dce86392f43301`。仅保留 GEMM1 与其 helper/import/compile_hints，并将 group8 guard 的 `n_experts == 65` 扩展为 `n_experts in (64, 65)`。将该 guard 还原后完整 GEMM1 AST 与已验收源相同，数值计算和同步逻辑保持。原 dispatch/kernel 文件不改，本项通过 `my_code/run_gemm1_e64_copy.py` 在单个进程内接入，并使用独立 production snapshot 固定 GEMM2 与 MoE 配置。

workload 为 A4W4、SiLU、no-bias、random non-balanced，E64/T1536/topk8/model_dim7168/inter_dim2048、shared_experts=0，有效 routes=12288，实际 contiguous_m 容量=28672。采用 t256/w2x4/b4、N tasks1、staged quant scale1、rowtail3、half A/scale prefetch1、rowaload2、pair A1、M-order1、M-group8，balanced_m192/direct-scale/drain-output 均为0。实际生产调用 B TDM hint=`tdm_b_th=6`。全部操作仅在 a07-3/dsr1_ep4；remote branch=`moe_a4w4_pr_refactor_wt`、HEAD=`339589f900b3a880ec447f605886fe803cbf9649`、image=`rocm/fw-bringup:gfx1250-atom-20260930-na`。

无 dispatch 编译通过：新文件 ELF=`0c5952afb691424ddffb63736bf378a184447713a0b1d6c25506234f13da5949`、34096 B、VGPR486/SGPR68/LDS282624 B、zero spills/private。原 E64 默认 ELF=`addf6e2a0436d2b17137d37757615ff8dda939b837b354fb2a7450c28e64bf9c`。最初离线脚本用 hint0，继而尝试 hint3，均与实际调用不同；通过实际参数记录确认 hint6 后重新编译，实际 loaded ELF 严格匹配。早期 mismatch 与收尾 root 权限错误保留原始日志。

新文件 seed0/17 的严格精度均 rc0、未超过120秒：与原 E64 基线的有效 payload/ScaleA/GEMM2/MoE 四项逐字节差异均0；各三次 eager/graph canary及三组同地址动态 A/ScaleA/psum replay通过，包括空 expert 和 M16/M32/M128 边界。payload/ScaleA 与 standalone producer一致。logits_diff 分别为3.4733e-06、3.4766e-06，原门限0.01保持；实际加载 ELF 均匹配上述新文件 ELF。正确性运行的所有耗时不记为性能。证据=`.codex_tmp/side_e64_gemm1_20261018/verify_copy_seed{0,17}_v3/`。

首次完整 eager MoE 性能队列的原始读数如下。每轮使用原 benchmark、warmup5/iters20/testGraph=False，整机 pre/post GPU/KFD 检查、单次120秒上限；计时中不并发监控。未形成完整合格三轮组，不计算验收中值或改善百分比。

| 路径 | 轮次 | GEMM1 μs | GEMM2 μs | MoE e2e μs | 验收状态 |
|---|---:|---:|---:|---:|---|
| 原 E64 默认 | 1 | 99.1433 | 61.3541 | 212.810 | rc0/精度通过，settled 出现外部 E65 GPU owner，性能无效 |
| 原 E64 默认 | 2 | 98.9346 | 63.7260 | 213.326 | rc0/精度通过，settled 出现外部 E65 GPU owner，性能无效 |
| 原 E64 默认 | 3 | 100.7100 | 59.6105 | 214.669 | 单轮 rc0/精度/ownership/ELF 合格；不是三轮中值 |
| 新文件 group8 | 1 | 92.7002 | 59.4648 | 204.077 | rc0/精度通过，post 有外部 live owner，性能无效 |
| 新文件 group8 | 2 | 93.0331 | 59.4879 | 203.634 | rc0/精度通过，settled 外部占用，性能无效 |
| 新文件 group8 | 3 | 97.2（profiler表） | 61.0（profiler表） | — | profiler后 `Memory access fault ... address 0x25820000 ... Page not present or supervisor privilege`，rc134，性能无效 |

队列目录=`perf_baseline_a_try1_v3/`、`perf_copy_a_try1_v3/`，第三轮故障后停止该队列，没有重启或拼接其中读数。正式 E65 最佳107.330 μs不受本项影响，不能将E64结果作为E65优化记录。

后续 `diagnose_post_fault_v1` 在计时后逐capture stage/hash/reference同步，rc0，各阶段均完成，精度通过；本次没有复现fault，不代表偶发fault已修复，也没有定位失败运行的fault PC或根因，诊断数字全部无效。

新增独立 `bench_gemm1_e64_copy_isolated.py` 用真实 production payload/scale/weight/psum 和仓库原 `run_perftest` 只计 GEMM1，以区别完整MoE路径。首版复核错误地用未对齐 expert end 作为下一 expert start，包含未定义padding，payload差异32918 bytes、scale0，assert失败，93.5 μs读数弃用。第二版仅重建当前输入对照，仍存在同一行号错误，在计时前repeat检查失败，未计时。v3修正为原验证脚本的256行对齐公式，并检查由psum推导的有效counts与真实routing一致，计时前后均只比较有效行。随后 `isolated_copy_valid_rows_v3` 三轮读数为91.005895 / 87.201263 / 88.776000 μs，前两轮合格，第三轮结束时外部 `verify_optx_replay.py` 的 GPU owner 存在，ownership不合格；整组不统计正式中值，88.776 μs也不列为有效性能。

独立 GEMM1 最终验收：`isolated_copy_valid_rows_v4` 与 `isolated_baseline_valid_rows_v3` 两组各三轮全部rc0、未超时、整机pre空闲且post无live GPU owner/fuser用户、settled空闲，boot保持；计时期间本项不并发监控。每轮使用相同seed0随机non-balanced routing、12288条有效routes、原 `run_perftest`、warmup5/iters20/testGraph=False。两组routing counts、source manifest和测量参数逐项相同，实际loaded ELF匹配各自已验收compile；有效payload和ScaleA计时前后逐字节差异均0。

| 独立 GEMM1 路径 | 第1轮 μs | 第2轮 μs | 第3轮 μs | 完整合格中值 μs |
|---|---:|---:|---:|---:|
| 原 E64 默认 | 95.901579 | 92.249632 | 92.760947 | 92.760947 |
| 新文件 group8 | 87.723842 | 87.795842 | 90.554684 | 87.795842 |

此入口下GEMM1中值减少4.965105 μs，耗时下降5.3526%。保留三轮原始波动，未选最快轮或拼接污染组。此数据只描述独立eager GEMM1，不能与上面完整MoE profiler的100.710 μs直接计算收益，也不能推出新的GEMM2/MoE e2e结论。完整MoE队列的偶发GPU fault仍未定位；单独测量通过不代表该fault已修复。

最终验收=`.codex_tmp/side_e64_gemm1_20261018/isolated_final_acceptance.json`；原始证据包=`side_e64_gemm1_raw_evidence_v1.tar.gz`，1001481 B，SHA256=`4cbec4b8c180c6ee33b2fb5b6920ee4afd14257270abc752d663baeac79bea64`。包包含本项raw日志、整机空闲记录、result/summary、编译/精度/实际ELF及源快照，排除大tensor文件、core和其他任务目录。最终driver SHA逐项记录在验收JSON，不沿用脚本修订前的旧manifest。

完整 MoE 复现命令（容器 repo 根目录）：

```bash
ENABLE_CK=0 AITER_MOE_EXPERT_BALANCE=false AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u my_code/run_gemm1_e64_copy.py \
  --variant copy --script bench \
  --binding-output .codex_tmp/side_e64_gemm1_20261018/manual_binding.json \
  --scenario bench --data-format a4w4 --act silu --no-bias \
  --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 \
  --shared-experts 0 --model-dim 7168 --inter-dim 2048 --iters 20
```

正式 host runner 使用 `python3 -u my_code/run_gemm1_e64_copy_host.py <新label> --stage perf --variant copy --runs 3`，包含空闲轮询与120秒超时；直接复现时也必须遵守同一空闲门禁。改为 `--variant baseline` 运行固定生产快照的原E64默认路径。独立GEMM1测量用 `--stage isolated`，其结果不能冒充完整MoE profiler/e2e。

本次合格的独立 GEMM1 复现命令在a07-3主机repo根目录执行；host runner会进入dsr1_ep4，GPU占用时每6秒等待，单次GPU程序120秒上限。重跑时为label使用未存在的新目录名。

```bash
python3 -u my_code/run_gemm1_e64_copy_host.py \
  isolated_copy_valid_rows_v4 --stage isolated --variant copy --runs 3
python3 -u my_code/run_gemm1_e64_copy_host.py \
  isolated_baseline_valid_rows_v3 --stage isolated --variant baseline --runs 3
```

### 找回复制源 E65 group8：E64/E65 完整 MoE 精度及性能验收（2026-10-19）

本项使用当时复制 E64 文件所依据的 E65 group8 旧版。该旧版仍保存在本地 worktree，以下两个文件与当时的 `task_m_group_default_manifest_v1.json` 完全一致：

```text
.codex_tmp/dsr1_e65_gemm1_20261017/task_m_group_default_v1_source/
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py
  aiter/ops/flydsl/grouped_gemm_mxfp4.py
```

旧 E65 kernel SHA256=`d29108153f58a778f6510c201a58e12eaa1a45cabd15a457548aecd5aa5cdae5`，wrapper SHA256=`5b6ff443b2c391ab666e04aa720e36de6888918d0600165d295d0e922e446e38`。当前工作区同名文件已经变化，因此本项将旧版固定在 `.codex_tmp/side_dual_moe_20261019/original_snapshot/` 中，以进程内加载方式运行；本地和远端的固定源码逐项 SHA256 相同。当前共用 unified 路径不参与本项测量。

E64 复制文件为 `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_gemm1_e64.py`，SHA256=`0dce35d584b593895da7a07fc083acd88c174dbb4931f5d210dce86392f43301`。其中完整 GEMM1 相对旧 E65 只将 group8 guard 中的 `n_experts == 65` 扩展为 `n_experts in (64, 65)`；还原这一处条件后，完整 GEMM1 AST 相同。计算、量化、地址计算和同步逻辑保持。新文件只保留 GEMM1 与其所需 helper/import，GEMM2 使用固定旧版原路径。本轮没有新的 kernel 数值或同步修改。

两种规模均使用 t256/w2x4/b4、`persistent_n_tasks=1`、`staged_quant_scale=1`、`row_tail_predication=3`、`half_a_prefetch=1`、`half_scale_prefetch=1`、`row_tail_a_groups=2`、`pair_a_prefetch=1`、`task_m_order=1`、`task_m_group=8`；`balanced_m192`、`partial_m_direct_scale`、`partial_m_drain_output` 均为0。保留原接入参数：旧 E65 GEMM1 的 B TDM hint0，E64 复制版的实际接入 hint6。

工作负载为 A4W4、SiLU、no-bias、random non-balanced routing，T1536/model_dim7168/inter_dim2048。E64/topk8/shared0 的有效 routes=12288、contiguous_m 容量28672；E65/topk9/shared1 的有效 routes=13824、contiguous_m 容量30464，其中 shared expert 接收1536条 route。输入、权重和真实 routing 保持，未采用 const0 或放宽精度标准。

所有 GPU 程序均在 a07-3 的 `dsr1_ep4` 容器运行。远端 repo=`/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`，branch=`moe_a4w4_pr_refactor_wt`、HEAD=`339589f900b3a880ec447f605886fe803cbf9649`，image=`rocm/fw-bringup:gfx1250-atom-20260930-na`。每个 GPU 程序均采用 `timeout --signal=TERM --kill-after=5s 120s`。

#### 完整 MoE 精度

E64/E65 各 seed0、seed17 的严格验证共四项，全部 rc0、120秒内完成。每项包含与固定基线的有效 payload/ScaleA/GEMM2/最终 MoE 四项逐字节比较、各三次 eager/graph canary，以及三组同地址动态 A/ScaleA/psum 更新重放。共112项 `differing_bytes=0`，覆盖空 expert 和行尾边界。另与 standalone producer 路径对照的 GEMM2 valid rows、最终 MoE output 均为 `bitwise_mismatch=0`。这些零字节比较对应固定基线及相同量化语义的 producer 对照；完整 PyTorch reference 精度另按原 `logits_diff < 0.01` 标准检查。

| workload | seed | logits_diff | rel_l2 | 完整输出 |
|---|---:|---:|---:|---|
| E64/topk8/shared0 | 0 | 3.473346e-06 | 0.002635657 | (1536,7168)、BF16、全部 finite |
| E64/topk8/shared0 | 17 | 3.476638e-06 | 0.002636903 | (1536,7168)、BF16、全部 finite |
| E65/topk9/shared1 | 0 | 3.779278e-06 | 0.002749280 | (1536,7168)、BF16、全部 finite |
| E65/topk9/shared1 | 17 | 3.778831e-06 | 0.002749115 | (1536,7168)、BF16、全部 finite |

随后每种规模在单独进程内运行 seed0/17/101：完整 profiler 后处理、capture、hash、PyTorch reference 和实际输出检查均通过。每个 seed 再做三次完整 eager、三次完整 graph 重放，各规模9次 eager+9次 graph，合计18次 eager+18次 graph，最终输出逐字节相同。seed101 的 logits_diff 分别为 E64 `3.4876599434021036e-06`、E65 `3.775772632252015e-06`。所有 correctness-only 阶段的耗时不作为性能数据。

所有实际加载 ELF 均逐项核对，避免只凭 kernel 名推断使用了旧版：

| workload | stage | actual ELF SHA256 |
|---|---|---|
| E64 copy | GEMM1 | `0c5952afb691424ddffb63736bf378a184447713a0b1d6c25506234f13da5949` |
| E64 | GEMM2 | `6982c0689aa1c8934ba07fa210b1b17d9210c79ffbc062e31acb5e999e84582c` |
| E65 recovered group8 | GEMM1 | `7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692` |
| E65 | GEMM2 | `840d2c768ad0b92ee446af222164513d8d190e75ddf5e6a664fc2b4b5a20cca0` |

#### 完整 MoE 三轮性能

使用固定原 benchmark 的 `run_moe`/`run_perftest`，warmup5/iters20/`testGraph=False`；原 profiler 去首轮规则保持，GEMM1/GEMM2 的 count 均为19。GEMM1/GEMM2 取完整 MoE 中相应 kernel 的 `device_time_avg`，e2e 沿用原 benchmark 返回的完整 production path 计时，三列分别计算各自三轮中值。每轮都有全机 GPU/KFD pre/post/settled 检查；占用时每6秒等候，计时中不并发本项监控、传输、编译或 ATT。

| workload | 合格组/轮次 | GEMM1 μs | GEMM2 μs | MoE e2e μs |
|---|---|---:|---:|---:|
| E64 copy/topk8/shared0 | try2/run1 | 93.269263 | 60.428474 | 205.759263 |
| E64 copy/topk8/shared0 | try2/run2 | 94.591263 | 58.429684 | 205.870947 |
| E64 copy/topk8/shared0 | try2/run3 | 93.511737 | 59.070474 | 205.010526 |
| **E64 copy/topk8/shared0** | **三轮中值** | **93.511737** | **59.070474** | **205.759263** |
| E65 recovered group8/topk9/shared1 | try1/run1 | 107.893211 | 94.445684 | 257.733842 |
| E65 recovered group8/topk9/shared1 | try1/run2 | 107.380737 | 94.745158 | 258.712105 |
| E65 recovered group8/topk9/shared1 | try1/run3 | 107.764684 | 93.857263 | 257.153842 |
| **E65 recovered group8/topk9/shared1** | **三轮中值** | **107.764684** | **94.445684** | **257.733842** |

两组最终六轮全部 rc0、未超时、精度/actual ELF/ownership 合格。E64 第一组三轮虽然执行与精度通过，但 run1 的 post `gpu_users.sh` 显示已退出的 PID1517838，settled 仍有 KFD owner，因此整组三轮排除，仅因 ownership 完整重测 try2，没有按耗时筛选或拼接轮次。该排除组的原始读数保留如下，不计算合格中值：

| E64 排除组 | GEMM1 μs | GEMM2 μs | MoE e2e μs |
|---|---:|---:|---:|
| try1/run1 | 93.981842 | 59.553526 | 204.720053 |
| try1/run2 | 93.085737 | 59.905526 | 206.469947 |
| try1/run3 | 93.165737 | 59.391105 | 204.329474 |

本轮 E64 的93.511737 μs来自完整 MoE；上节87.795842 μs来自独立 eager GEMM1，两者测量入口不同，不能相互替代或直接计算性能回退。上述旧 E65 数据同样只验收恢复的 group8 旧版，不替换其他章节当前 unified 路径的正式记录。

#### 历史 fault 核查及证据

重新核对上节失败日志，fault address 应为 `0x2582000`；上节 `0x25820000` 是转录多写一个0。2026-10-18 13:47:58 的 dmesg 记录了 Python PID2097448、地址 `0x0000000002582000`，同 PID 另有地址 `0x134000`；日志中 `Faulty UTCL2 client ID: TCP (0x8)`、`PERMISSION_FAULTS:0x3`、`RW:0x0` 保留在原始证据中。现有日志没有 fault kernel/PC，无法确定根因，也不能把该 fault 归因给随后出现的其他 E65 进程。本轮完整 MoE 验证和性能队列未复现此 fault；这不等于已经定位或修复偶发 fault。本轮没有依据猜测增加 drain 或修改 kernel 同步。

最终运行目录为 `.codex_tmp/side_dual_moe_20261019/recovered_original_e2e_v1/`。raw 包为 `.codex_tmp/side_dual_moe_20261019/recovered_dual_moe_raw_evidence_v1.tar.gz`，672311 B、141 regular members，SHA256=`40cbe7bfb64b439616cd992cf2ff045e625d0d5fc71aeebcf92b966d8b9ebaf6`。包含本项 command、idle、raw、result/summary、profile、stress events、actual ELF、旧版源码、driver、provenance 和旧 fault 摘录；排除大 tensor、core 和其他任务目录。已下载至本地 `downloaded_evidence_v1/`，逐 member SHA256、112项原始零字节比较、三轮 profile、完整 e2e、源代码及实际 ELF 均通过独立复核，receipt=`recovered_dual_moe_raw_downloaded_acceptance_v1.json`。

本轮结束检查记录于 `final_post_state.json`：没有本项 runner/GPU 进程残留，KFD owners为空、busy0、VRAM176582656 B、全机 idle。

#### 复现

在 a07-3 主机的远端 repo 根目录运行，host runner 自动进入 `dsr1_ep4`，设置原 workload 环境、固定旧版 snapshot，执行双 seed 严格精度、三 seed 完整 e2e 重放和两规模各完整三轮性能。label 必须为未存在的新目录名，下例仅用于新一轮复现，不覆盖已完成的目录：

```bash
python3 -u my_code/run_recovered_e64_e65_moe_host.py \
  recovered_original_e2e_recheck_01 --stage all
```

`--stage checks` 仅做正确性，`--stage perf` 仅做两规模完整性能；性能数据的有效性仍要求精度验收成立及全机空闲门禁通过。driver 为 `my_code/run_recovered_e64_e65_moe.py` 和 `my_code/run_recovered_e64_e65_moe_host.py`，输入 provenance=`source_manifest.json`/`recovered_source_receipt.json`；如果固定源码或支撑源码 SHA256 变化，runner 会停止并报告，而不会静默换用当前 E65 路径。

#### 补充：一个 WG 有8个 wave 时的寄存器预算

依据本地 `MI400_Shader_Programming#65.txt` §1.5、§2.2，gfx1250 使用 wave32，一个 WG 在同一 WGP 的4个 SIMD32之间分布。按8个 wave 均匀分布、每个 SIMD 驻留2个 wave，且不计其他 WG 的寄存器占用，§3.3.2.1 记载的1024 VGPR/SIMD资源池给出每 wave 512 VGPR预算，分配粒度为16。这里每个 VGPR 对应32个 lane 各一个32-bit值；512是此驻留条件的预算，单个 compute wave 的架构寻址上限为1024（§3.3.2.3）。更多驻留 wave 或其他资源约束需另行计算。

§3.3.1.1 明确每个 wave 有106个普通 SGPR（s0–s105），另有 VCC_LO/VCC_HI 两个专用位置和16个 trap temporary SGPR；物理分配为128个位置/每 wave。CDNA5 ISA §3.3.1.1 与该普通 SGPR 数量一致。白皮书第10页的128 SGPR/wave描述的是物理预算，不能解读为128个任意使用的普通 SGPR，也不需随 WG 的8个 wave 再等分。

恢复的 GEMM1 code object 报告 VGPR486/SGPR68；VGPR486向上分配到496，每 SIMD两个 wave 占992 VGPR，满足上述预算。该预算只证明寄存器容量允许，不单独证明全部资源条件下的实际 occupancy。

### 精确共用默认 EA counters：原始证据验收及单位限制（2026-10-20）

已在 a07-3/dsr1_ep4 完成两个独立 SDK counter dispatch：请求尺寸和 EA 在途请求量。使用原 E65/T1536/topk9/shared1 random non-balanced 完整 MoE workload，仅诊断 `--iters 2`，完整默认 symbol filter、requested GEMM1 iteration10、`--kernel-trace`；生产源码及默认 ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b` 保持。两项 child rc0、未超120秒，原 `logits_diff=3.7793e-06`、0.01精度门限通过。全部 profiling 耗时无效，不产生新正式 kernel 性能。

| 独立 dispatch | counter | 聚合值 |
|---|---|---:|
| sizes /PID94817 /Dispatch148 | GL2C_EA_RDREQ_32B | 1 |
| sizes /PID94817 /Dispatch148 | GL2C_EA_RDREQ_64B | 39 |
| sizes /PID94817 /Dispatch148 | GL2C_EA_RDREQ_128B | 0 |
| sizes /PID94817 /Dispatch148 | GL2C_EA_RDREQ_256B | 4058955 |
| latency /PID95061 /Dispatch148 | GL2C_EA_RDREQ | 4059744 |
| latency /PID95061 /Dispatch148 | GL2C_EA_RDREQ_LEVEL | 1624319825 |
| latency /PID95061 /Dispatch148 | GL2C_EA_RDREQ_DRAM | 32062287 |

两次进程虽各自 Dispatch_Id 都为148，仍是不同 dispatch，不能合并推导同一次调用的流量分解、命中率或带宽。实际 SDK config 的 gfx1250 event 为 RDREQ117、32B119、64B120、128B121、256B122、LEVEL129、DRAM130；definitions dimensions=`INSTANCE[0:95]/AID[0:1]`，CSV没有 dimension breakdown。CSV资源248/128不能代替实际 native VGPR486/SGPR68、LDS282624 B、zero spills/private、kernarg180 B。

按 SDK FETCH_SIZE 尺寸加权公式，sizes 的 GL2→EA 请求量为 **1039095008 B**，即1014741.21875 KiB。latency 的 LEVEL/RDREQ≈**400.104**；本地 `gl2_metrics#1.adoc.txt` “GL2C-EA Interface - Latencies” 将该比值标为 cycles，尚未验证 counter clock domain 与 wave clock 的关系，故不换算为 ATT cycles、ns 或 kernel μs。

资料存在单位表述冲突：`MI400_perfmons_list#8.txt` event130（行2381）和 `mi400_perfmon_spec#7.txt` 行4502–4505将 DRAM 值描述为32/64/96/128B variable-size requests；`gl2_metrics#1.adoc.txt` “GL2C-EA Interface - Bandwidth”（行534–536）却使用 **`GL2C_EA_RDREQ_DRAM *32`** 表示 HBM request bytes。按后者得到 **1025993184 B**。DRAM/RDREQ约7.9与 sector 计数解释相容，但现有资料不能唯一确定其单位，不能据此声称 HBM 已饱和。该字节值与sizes加权值来自不同 dispatch，不能相减或解释为实测 cache miss bytes。MI455X 白皮书公开peak也不能替代 a07实际配置与本次有效运行时间。

本次额外记录实际调用栈并绑定保存的 `aiter/test_common.py`：1次memory probe、5次warmup、2次cuda.Event、2次`run_iters_rotate`；第10次 GEMM1 是**第二次 profiler rotate call**。这纠正了先前按“两次memory probe、第10次首个profiler”推算的假设。旧GL2采集没有保存同样的stack，不能用新证据追认旧采集阶段已独立证明。

原55个 regular members/1645463 B raw完整下载逐member SHA256验收通过，包SHA=`48674a270b650883b2df7e5bc12d78f7a9ff0a662493a552be3773ca26a81c5a`。离线checker v1误用 frame 文本heuristic、不认识`return worker(...)`；v2修成实际AST函数/行核对后又误假设 kernel_trace 有`Process_Id`字段。两次失败及旧工具保留，v3使用实际`Thread_Id`、`<Process_Id>_kernel_trace.csv`文件名及counter CSV共同绑定，全部核对通过；未重跑 GPU、未调整精度或计时门槛。receipt=`unified_default_ea_counters_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`be81deb7717ddf8af03b6b798e63ee679c5cba04eb37041af90f235a33d26032`。

原外层 shell queue `.rc` 文件没有留存；summary finished=True、无exception、queue log CLOSED banner与两个child statuses rc0均保存，不能声称外层也留有rc0文件。即时post短暂已退出KFD entry的live owners为空，六秒settled全机idle；package后2026-10-20T00:19:52Z记录KFD空、busy0、VRAM176582656 B，无本项live残留。这只是当时状态，后续性能前继续重新检查。

原始采集复现入口为任务目录内`run_unified_default_ea_counters_host_v1.py`；已完成固定两项队列，不重新执行带旧label的脚本。原plan中完整profiler argument vector、SDK config、CSV、call stack、source、ELF及idle均归档。离线验收命令如下；receipt已存在时不重复执行写入。

```powershell
py -3 -X utf8 .codex_tmp/dsr1_e65_gemm1_20261017/check_unified_default_ea_counters_download_v3.py
```

### 共用 WMMA/DS group4 调度候选：编译前审查（2026-10-20）

从精确`unified_default_v2_source`另建`unified_mma_group4_v1_source`，只将 GEMM1 `MMA_GROUP 2→4` compiler scheduling hints opt-in；保留原数值、全部有效routes、每个有效accumulator的56次原序K128更新、load/store地址、ring、hardware fences及完整GEMM2。E64/E65仍共用runtime expert count和完整ELF，不按实际expert数specialize。开关=`AITER_GFX1250_GEMM1_MMA_SCHEDULE_GROUP4=1`，候选suffix=`_mmagroup4`，正常生产默认不改。

actual源码helper CPU审计80组边界/1520次hint events通过，WMMA/DS/TDM提示总量及closing cover12保持，关闭后完整module AST精确默认。编译前发现生成工具仍传旧`a_multicast_standard_timeout=mode`及引用不存在的`unified_default_source_manifest_v1.json`，已在首次上传/运行前修正为`schedule_mma_group4=mode`及v2默认manifest；不是device错误或已验证生产bug。

新native门槛核对两mode完整ELF/ISA/embedded launcher、zero spills/private、原LDS/180 B count ABI、同compiled object/cache1、所有branch targets/steady backedges、DS physical CFG、原数值与DS/TDM操作数量、输入release和输出completion fences。不能沿用A timeout的“五处literal差异”或旧group1的固定指令数。旧group1故障队列及旧E65-only group4淘汰结论保持；本项仅为当前共用ABI/rowtail/pairA的独立候选，此时尚无新candidate ELF、device精度或性能。正式E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。

#### group4实际编译与双seed完整MoE精度验收（2026-10-20）

a07-3/dsr1_ep4固定mode0→1 no-dispatch compile队列rc0，两项child rc0/120秒内。关闭完整ELF精确默认；开启完整ELF=`6de4f08672db2b2f561a3b3f63ca8904ce557043b5d738842266314a50b4c22d`，实际`.text`变化，4531→4529 instructions，VGPR486/SGPR68、zero spills/private、LDS282624 B、kernarg180 B/count offset176保持。normalized delta为`s_set_vgpr_msb +5 /s_wait_dscnt −4 /v_nop −3`，原11个zero DScnt lifetime waits和所有Tensorcnt/barrier sequence保持；原数值、DS/TDM操作数量相同，physical DS CFG无显式RAW/WAW conflicts。指令数量变化不直接换算为性能。

编译过程中已有外部GPU owner PID1271106，no-dispatch compile的耗时全部无效，未中断或干预其程序。native raw52 regular members/635297 B包SHA=`70e9b57933eb6c300a8789d5e36169a95a2721f7017bdc6d803b024a0f998cce`，逐member SHA、两mode五件套、实际共享object/ELF、源码与同步依赖独立下载复核通过，native receipt SHA=`b68e00172afdb0f4300d972237ad8df5eefa2904584b69329699e080bdf8abcc`。打包当时已全机idle/KFD空，无本项live GPU残留。

随后固定seed0→17、每进程E65→E64，112项原HEAD有效payload/ScaleA/GEMM2/MoE逐字节比较全零、四项原reference通过。E65 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`；E64分别`3.4733e-06/3.4766e-06`和`0.0026357/0.0026369`。eager/graph canary、same-address dynamic A/ScaleA/psum及empty expert检查通过，E64/E65正常参数复用同compiled object/cache1及上述完整ELF，0.01门限不改。两项child rc0/120秒内，correctness-only耗时无效，post live owners为空。

精度raw35 regular members/269492 B包SHA=`b6fcd5961f59e6df4686c48f9f271dbb77cc5c89ca7b537729dcc56530b9cdd7`，独立下载核对全部原比较/reference/replay/ELF/命令与SHA通过，correctness receipt SHA=`ede5e8fe868a717b30589d7cf0f606c8f90a1d2d924ab82572aae03c53292e5e`。模板生成的package/checker f-string版本及默认lookup7 v2 receipt引用也在各自首次上传/运行前审查修正，原generated preparation保留，没有为工具修订重复GPU队列。

固定完整性能队列已启动：默认E65→默认E64→候选E65→候选E64→默认E64→默认E65，每组三轮，仅ownership污染整组最多三次尝试，执行/精度/解析失败立即关闭。候选进入正常live入口后须另56项完整比较和同ELF绑定，finally恢复精确默认；计时期间不并发传输、监控、编译或ATT。双规模GEMM1相对两侧控制均改善、MoE e2e均不回退才能进入一次固定反向确认，否则淘汰、不追加计时。此时没有本候选正式性能结论。

```bash
# 已完成的一次固定编译和精度队列；旧label不可重开。
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_mma_group4_compile_host_v1.py
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_mma_group4_correctness_host_v1.py
# 本轮固定性能入口，host runner自动进入dsr1_ep4并执行空闲/120秒门禁。
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_mma_group4_perf_host_v1.py
```

#### group4性能队列关闭：默认E64三组ownership不合格（2026-10-20）

固定六组队列仅完成默认E65前控制及默认E64的三次完整组三轮尝试。12次原benchmark全部rc0、精度通过、120秒内；外层rc1来自`Three ownership-contaminated whole groups: E64 mode0`，`execution_failure=null /finished=false /performance_data=false`。默认E64三组均排除，中值保留null。候选尚未部署live，normal live binding、候选性能和后控制均未执行；不能据此判断group4提速或回退，不重开该队列。

| 默认前控制组 | GEMM1原三轮 μs | 合格GEMM1中值 μs | 合格GEMM2中值 μs | 合格MoE e2e中值 μs |
|---|---|---:|---:|---:|
| E65/try1 | 108.357 /107.760 /106.430 | 107.760 | 94.1295 | 256.437 |
| E64/try1 | 96.5540 /94.9138 /94.5426 | null | null | null |
| E64/try2 | 94.8609 /95.6199 /97.5874 | null | null | null |
| E64/try3 | 95.0214 /96.2064 /94.3233 | null | null | null |

三个排除轮的原`result_*.json /idle_*.jsonl`独立复核如下。立即post的部分KFD entries已退出，不能按PID猜测其身份；随后实际保存的`gpu_users`给出了另一项OPTX验证程序的命令。

- E64/try1/run3：post KFD=`1358850`、busy1、VRAM5912883200 B、live owners空；settled KFD=`1359969`、VRAM176664576 B，命令为`verify_runtime_expert_moe_scale_prepared257.py --seed 17 --label optx_v257_scale_b_capacity_moe_seed17_20261020_r1`。
- E64/try2/run2：post KFD=`1366198/1366520`、busy1、VRAM5912993792 B，live owner=`1366520`；settled仍有该owner、VRAM4403597312 B，命令为`verify_runtime_expert_replay_scale_prepared257.py --label optx_v257_scale_b_capacity_replay_20261020_r1`。
- E64/try3/run1：post KFD=`1374172`、busy1、VRAM5912887296 B、live owners空；settled KFD=`1375031`、VRAM176660480 B，命令为`verify_runtime_scale_prepared257_edges.py --start 0 --stop 5 --label optx_v257_scale_b_capacity_edges0_5_20261020_r1`。

这些ownership检查证明采样门槛未满足，并不证明外部程序一定与每个计时区间重叠。保留保守整组排除，不从九轮中拼出新中值。没有杀死或干预外部程序。finally于2026-10-20T00:53:35Z核对默认两源精确SHA、KFD空/busy0/VRAM176582656 B，未发现本项live GPU残留。

停止专用raw包=`unified_mma_group4_perf_stopped_raw_boot1d5146ac_v1.tgz`，679652 B、85 regular members，SHA256=`9f3809fed1370e1c273fdb8463120e7bebc3380145b50ff482d769908ec8c598`。独立下载逐member SHA、12轮原命令/精度/ownership、一个E65合格控制、三个E64完整排除组、原rc1及缺失candidate stages验收通过，receipt=`unified_mma_group4_perf_stopped_downloaded_acceptance_boot1d5146ac_v1.json`，SHA256=`a8ec56c5a97b08a4425a682964b348ac3d93d433bd029cd44938ef07ca07ae3d`。未使用要求六组合格的完整性能checker为停止队列补造结果。打包时全机idle、无本项GPU残留。

正式可保留结果仍为E65 GEMM1/GEMM2/e2e=`107.379/94.3843/256.893 μs`、E64=`95.3059/60.4664/209.078 μs`，E65距80 μs差27.379 μs。本次107.760 μs只是默认前控制，不替换已完成双侧验收的最优记录。

### GEMM1 post-RA scheduler关闭候选：native指令增加，未dispatch（2026-10-20）

从精确共用默认创建`unified_no_post_sched_v1_source/`，仅将GEMM1的`compile_hints["llvm_options"]["enable-post-misched"]`改为False，GEMM2同名选项保持True。98个实际function AST、全部数值/地址/DS/TDM/hardware fences与默认相同，runtime expert count及有效accumulator的56次原序K128更新保持。准备工具首次误认为完整模块只有一个同名选项而本地assert失败；下一次audit因manifest尚不存在而失败。这两次尚未上传或启动device队列，原失败记录保存在`unified_no_post_sched_preparation_failure_v1.json`。修正为仅检查GEMM1所在半部后，源码和传输验收通过。

a07-3/dsr1_ep4固定mode0→1 no-dispatch编译的两个child均rc0、120秒内。mode0完整ELF精确默认；mode1完整ELF=`3d6292ce18f78b6a81cca5c9ebdbb35ecc1badc84178f67995bc2a36e8dd3aae`，指令4531→4782。VGPR486/SGPR68、LDS282624 B、zero spills/private、kernarg180 B及count offset176/size4保持；两种mode各自E65→E64→E65共用一个compiled object/cache1/完整ELF。normalized delta为`s_delay_alu +82 /s_set_vgpr_msb +160 /s_wait_dscnt −9 /s_wait_xcnt −1 /s_nop −1 /v_nop −17 /v_bitop2_b32 +1 /v_or_b32 −1`；保护的数值与DS/TDM操作数量、原11个DScnt0及Tensorcnt/barrier sequence保持，有限physical DS CFG未发现显式冲突。

唯一native拒绝原因是`Disabling post-RA scheduling increased executable instruction count`，因此外层queue rc1、两个compiler child rc0、`execution_failure=null /finished=true`。候选淘汰并关闭队列，未做GPU精度、性能或live部署；指令增加不是实测性能回退。生产默认SHA保持。

原54 regular members/628205 B归档SHA=`d781de76cf06c77e0057e2ea2fcd97941382141f24b4df2d39a4759cbf544bd3`，两mode ELF/ISA/notes/launcher/summary、原命令/exit status/source/同步条件全部下载独立核验通过；receipt=`unified_no_post_sched_rejected_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`5b7ae6c1caf7b620349f8dc234e3b1041f54979f5affa8499820e48a9560a71e`。package时有另一项OPTX的PID1641757，命令为`verify_runtime_scale_prepared258_edges.py --start 15 --stop 21`；保留原owner，不干预，编译/归档不产生正式计时。本项没有live GPU残留。

### 保留post-RA scheduler的bottomup候选：FlyDSL enum setter失败（2026-10-20）

`unified_post_sched_bottomup_v1_source/`只在GEMM1增加`"misched-postra-direction": "bottomup"`，`enable-post-misched=True`和完整GEMM2保持。98个function AST与默认相同；source persistent SHA=`34588ec3efa2a22db5319f9732fbb6d8879b7c47eec9e5b2d22191527397b00d`，wrapper精确默认，两端传输验收通过。

固定mode0→1 no-dispatch队列中，默认mode0 rc0、完整ELF精确默认、E65/E64同object/cache1。mode1在`flydsl/compiler/llvm_options.py:35`调用`_fly.set_llvm_option_str(name, value)`时报`RuntimeError: basic_string::_M_construct null not valid`，child rc1、未超时。外层rc1，`finished=false /execution_failure=mode1`。没有生成candidate ELF或native acceptance，未dispatch、未精度/性能，也未部署live；队列关闭，不重开。

实际远端`MachineScheduler.cpp:191–201`将此选项声明为`cl::opt<MISched::Direction>`，而`FlyDSL/lib/CAPI/LlvmConfig/llvm.cc:70–85`的string setter仅根据`ValueRequired`区分bool，将其`static_cast<cl::opt<std::string> *>(opt)`后读取旧值。这一类型不匹配与当前异常相符，不能将它描述为GEMM1时序race或证明bottomup调度有无性能收益。没有改共享FlyDSL安装或用int setter绕过enum类型检查。

实际远端LLVM/FlyDSL源码SHA及章节摘录保存于`no_mem_cluster_remote_llvm_excerpts_v1.json`，不以本地另一版LLVM代表实际运行版本。原48 regular members/480804 B raw包SHA=`9fb1dcfc2447d063a4413ff5cbe185e68833ce8f5ba1d78c28ce20b9e733d577`，原失败stack、rc1、默认native五件套、源/命令与candidate产物缺失全部独立下载核验通过。receipt=`unified_post_sched_bottomup_failed_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`f3ca28339000a8c840613daacd7a3b9d0e48ee175fca96f6bb2b1f1aae789008`。package检查默认SHA保持，KFD owners空、无本项live残留。只完成一次固定编译队列，没有GPU重跑。

### 关闭memop clustering：实际native及完整双规模精度通过（2026-10-20）

独立`unified_no_mem_cluster_v1_source/`只在GEMM1增加`"misched-cluster": False`，post-RA scheduler保持开启。实际远端`MachineScheduler.cpp:258–260`明确此项为bool；其2058–2074行控制load/store clustering mutation的创建，`AMDGPUTargetMachine.cpp:1440–1463`的post-RA scheduler使用这些mutations。该选项也影响pre-RA的clustering，不将它称为仅改变post-RA。原98个function AST、有效计算、load/store地址、硬件同步、GEMM2、benchmark与canonical runtime ABI保持；persistent SHA=`a6bf1d5e4cd42d3328fa682bf14f7ac2ab6965510ddf4131595b181980f58c84`，wrapper精确默认。

固定mode0→1 no-dispatch编译queue rc0、两项child rc0/120秒内；每mode E65→E64→E65同object/cache1/完整ELF。关闭完整ELF精确默认，candidate=`6c638b85aad8e496c3c486c9c0064d25ed572ed20cc6a8502d56023571be2540`，4531→4530 instructions，normalized delta只有`s_nop −1`。VGPR486/SGPR68、LDS282624 B、zero spills/private、180 B/count offset176保持；原numeric/DS/TDM counts、11个DScnt0、Tensorcnt/barrier sequence及有限physical DS CFG通过。实际`.text` SHA从`47604f4f0e98953a88fa1f771960a8a624badb2667d7a1f9688a9cb26f8b8e48`变为`b0093a08c46737e6c551490a61dc854c9d0fe1a7754cba1ecfd1de6e7f082319`。离线diff的96个regions包含寄存器重命名、次序变化和branch位移，不能把这些regions全部称为减少的stall或性能收益。

53 regular members/630486 B native raw包SHA=`b65a6d3b11ade6f1849e5b924cd1daeeb62997d9b142457acfe888a0a0d2a7a4`，完整两mode五件套与原命令/source/fences/download SHA独立验收通过；native receipt SHA=`63fa029dacd67db934db380c4357126bd06743648f65603896b27b617299ffdd`。

随后固定seed0→17，每进程E65→E64，两个child均rc0/120秒内：共112项有效payload/ScaleA/GEMM2/MoE零字节差异，原reference、eager/graph canary、同地址dynamic A/ScaleA/psum及empty expert检查通过。E65 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`；E64为`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`，0.01门限保持。E64/E65实际同compiled object/cache1/上述完整ELF，无extra expert specialization。seed0的post观察到外部owner1782686，correctness-only耗时全部无效，未将其作为性能；seed17 post owners空。

精度35 regular members/269338 B raw包SHA=`e7e6d6d16d81bc08328410d0589366bcdf6d104b7eed390306c688140ebdcc64`，全部原比较/reference/replay/命令/ELF绑定下载独立验收通过；receipt=`unified_no_mem_cluster_correctness_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`9db56b0d5d11e56d43be7af3d2343a907855d4cd09d42e7c5a105fbc1a001691`。

固定完整性能队列已启动：默认E65/E64→候选E65/E64→默认E64/E65，每组三轮，仅ownership整组最多三次、执行/精度/解析失败停止；normal live另56项完整比较和同ELF绑定。性能前全机idle检查、占用每6秒等待，计时中不并发本轮传输/监控/编译/ATT，finally恢复默认。此时没有新正式性能结论，生产最优仍E65 107.379 μs/E64 95.3059 μs、距80 μs差27.379 μs。以下是本轮一次固定执行入口，完成的旧label不重开：

```bash
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_no_mem_cluster_compile_host_v1.py
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_no_mem_cluster_correctness_host_v1.py
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_no_mem_cluster_perf_host_v1.py
```

#### memop clustering性能队列关闭：默认E65三组ownership污染（2026-10-20）

固定队列只执行了默认E65前控制的三次完整三轮组，九次原benchmark均rc0、精度通过、未超120秒；三组均有ownership污染，全部中值保留null。外层rc1为`Three ownership-contaminated whole groups: E65 mode0`，`execution_failure=null /finished=false /performance_data=false`。候选未部署live、未做normal live binding、没有候选性能或后控制，不能判断本候选提速或回退。队列关闭，不重开或拼接轮次。

| 默认E65排除组 | GEMM1原三轮 μs | ownership三轮 | 合格GEMM1/GEMM2/e2e中值 |
|---|---|---|---|
| try1 | 107.807 /125.395 /107.381 | True /False /True | null /null /null |
| try2 | 108.633 /106.727 /108.800 | False /False /False | null /null /null |
| try3 | 107.098 /108.070 /107.820 | True /True /False | null /null /null |

try1/run2的live PID1831669属于`jli-mi450`，命令为`python op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a8w4 --experts 384 --tokens 512 16384 6144 --topk 6 --model-dim 7168 --inter-dim 768 --act silu --no-bias --no-check-aot-cache`，post busy2/VRAM22752534528 B。try2/run1、run2的live PID1847541/1851788属于`zan_att`，运行E64/T1536 A4W4 grouped GEMM；run3的PID1859498为另一个OPTX的`verify_runtime_expert_replay_scale_prepared260.py --label optx_v260_scale_b_capacity_replay_20261020_r1`。这些原命令与owners均保留，未杀死或干预。try3/run3立即post有已退出KFD entry1919764、busy1/VRAM6076690432 B；六秒settled恢复idle。该已退出entry没有可核对的命令，不能推断来源或证明某个计时区间必然发生重叠。

finally于2026-10-20T01:59:40Z核对默认两源精确SHA，KFD空/busy0/VRAM176582656 B；打包时同样没有本项live GPU残留。停止专用71 regular members/550788 B原始包=`unified_no_mem_cluster_perf_stopped_raw_boot1d5146ac_v1.tgz`，SHA=`44f34d2463ff0dc814dc9455b22a9f7b2d286797c1295c24a7fb06a77f475469`。独立下载复核九轮原命令、精度、ownership、null中值、rc1和candidate stages缺失通过；receipt=`unified_no_mem_cluster_perf_stopped_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`257befc46808d64c0de8d37c3053ac78247fa3463de863d4f2b95b7b5b19968e9`。

### 关闭可选delay-ALU插入pass：独立源码审计（2026-10-20）

本节的准备阶段记录随后由下方实际native、完整精度及性能验收更新。

#### delay-ALU实际native与双seed完整MoE精度独立验收（2026-10-20）

固定两mode no-dispatch队列及两个compiler child均rc0、120秒内。mode0完整ELF精确当前默认；mode1 ELF=`edd3e867ec099281565c959e0c57059a73727798c3ab26472d9671971eeaaa90`，`.text` SHA=`e6d2c252fbccedceb15978bf789c51f616a857525ea32e0c9a5154f555477c02`。4531→4176 instructions，唯一normalized数量变化为`s_delay_alu −355`；4176条保留指令逐条完全相同，112个branch目标按保留指令ordinal相同。原VGPR486/SGPR68、LDS282624 B、zero spills/private、180 B kernarg与count offset176/size4保持，原MSB/nops/DS/TDM/waits/barriers不改、有限physical DS CFG通过。每mode实际E65→E64→E65共用cache1/同compiled object/完整ELF。

53 regular members/628605 B native raw包=`unified_no_delay_alu_native_raw_boot1d5146ac_v1.tgz`，SHA=`2abc6b8e9d03e86f5975443bce973b0d3f9a4bdd7f53ca31d329b37c9d4da968`；独立下载验收receipt SHA=`97c41926b4261242b21529cd4a58e5cc7343070aca2c12737e616c3e47c33cd2`。mode1编译post有外部owner2024923，no-dispatch耗时无效、不作为性能、未干预。

随后固定seed0→17，每进程E65→E64，两个child均rc0/120秒内。112项有效payload/ScaleA/GEMM2/MoE逐字节差异为0；四项原MoE reference以及eager/graph canary、同地址dynamic A/ScaleA/psum、empty experts检查全部通过。E65两seed的logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`；E64=`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`。原0.01门限保持，两规模actual同candidate compiled object/cache1/完整ELF，无expert specialization。seed0 post有外部owner2069831，seed17 post owners空；correctness-only所有耗时无效，不能从中提出性能结论。

35 regular members/269650 B完整精度raw包=`unified_no_delay_alu_correctness_raw_boot1d5146ac_v1.tgz`，SHA=`6d71a13ed6044857a3f9da0ae87afef1f7ffb46e7f2741fe145e5fb8f0594e95`；独立逐member/source/ELF/112项比较/reference/replay/120秒门禁下载验收通过，receipt=`unified_no_delay_alu_correctness_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`12fbb0b1d34e02a6e763c28acb84fe609db858359c8d774b594f89a95e2c0cc4`，已上传核SHA。当前本地和远端live默认两源精确SHA保持。

在查看候选性能前已准备、审查、上传和逐SHA验证固定性能工具及完整六组raw checker；新队列已启动：默认E65→默认E64→候选E65→候选E64→默认E64→默认E65，每组完整三轮，仅ownership整组最多三次，执行/精度/解析失败即停止，normal live另56项严格比较和actual ELF绑定，finally恢复默认。原完整命令、warmup与iters保持；全机idle门禁与每6秒等待、单项120秒限制保持；计时中无本项传输/编译/打包/ATT/GPU监控。此时尚无新正式性能，最优107.379/95.3059 μs及距80 μs差27.379 μs保持。以下各入口只用于本次固定执行，旧已关闭label不得重开：

```bash
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_no_delay_alu_compile_host_v1.py
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_no_delay_alu_correctness_host_v1.py
python3 -u .codex_tmp/dsr1_e65_gemm1_20261017/run_unified_no_delay_alu_perf_host_v1.py
```

#### delay-ALU实验准备阶段的原记录

当前默认实际ISA有355条`s_delay_alu`。CDNA5 ISA §5.8（p57，文本行3620–3635）明确该指令可选、用于尝试减少依赖stall，并非数值正确性要求；此结论不能扩展到`v_nop`、DScnt、Tensorcnt、WG barrier或MODE修改。实际远端`AMDGPUTargetMachine.cpp:524–529`将`amdgpu-enable-delay-alu`定义为bool，2046–2047/2755–2757控制`AMDGPUInsertDelayAlu`pass。该pass源码`AMDGPUInsertDelayAlu.cpp:460–476`按`hasDelayAlu`和max waves启用，源码SHA及摘录保存在`delay_alu_remote_llvm_excerpts_v1.json`。初次源码查询使用了不存在的`SIInsertDelayAlu.cpp`，只是read-only路径错误；已核对实际文件名，不涉及device执行。

独立`unified_no_delay_alu_v1_source/`仅给GEMM1增加`"amdgpu-enable-delay-alu": False`，post scheduler仍开启，GEMM2保持。persistent SHA=`d93fb39720f64890b017a7c8221fc6217de28af72e5f9c3740aeee5da9221639`，wrapper精确默认。98项实际function AST完全相同，原有效计算、每accumulator的56次K128更新、所有地址/DS/TDM/output/hardware fences和runtime count ABI保持；已上传task snapshot/tools并逐项核SHA。固定mode0→1 no-dispatch编译已启动，单项120秒；native要求所有355条optional delay消失、每条保留指令和branch目标按ordinal完全相同、原资源和zero spills/private保持。没有把减少355条static指令直接当成性能提升；目前没有本候选GPU精度或正式性能，生产最优107.379/95.3059 μs保持。

#### delay-ALU性能队列关闭：候选E65三组ownership不合格（2026-10-20）

固定队列实际完成默认E65三次三轮、默认E64三次三轮、候选E65三次三轮，共27次原random non-balanced完整MoE benchmark。全部child rc0、原精度True、120秒内；两个默认第三组合格，候选E65三组均ownership不合格。按事前三次上限关闭，外层rc1、`finished=false/performance_data=false/execution_failure=null`，异常原文=`Three ownership-contaminated whole groups: E65 mode1`。未执行候选E64或两个后控制，也未重开、拼轮、按时间筛选或补造六组合格结果。

| workload / 路径 | GEMM1三轮 μs | GEMM1完整合格中值 μs | GEMM2完整合格中值 μs | MoE e2e完整合格中值 μs |
|---|---|---:|---:|---:|
| 默认E65前控制try3 | 108.032 /108.865 /108.121 | 108.121 | 94.6714 | 258.289 |
| 默认E64前控制try3 | 94.9579 /95.1519 /95.4407 | 95.1519 | 60.1902 | 207.937 |
| 候选E65 try1 | 109.384 /109.485 /110.900 | null | null | null |
| 候选E65 try2 | 109.105 /112.323 /111.916 | null | null | null |
| 候选E65 try3 | 109.030 /109.371 /109.127 | null | null | null |

两个默认前控制仅为本批数据，不替换正式默认记录。候选实际normal live入口seed0 E65→E64的另56项有效payload/ScaleA/GEMM2/MoE字节比较、原reference及replay通过，cache1/同compiled object/完整ELF=`edd3e867ec099281565c959e0c57059a73727798c3ab26472d9671971eeaaa90`，与no-dispatch和双seed snapshot验收一致。候选未形成合格完整性能组，不能据原读数宣称提速或回退。

原ownership记录逐轮保留。候选try1/run2的post entry2206023已退出、busy1/VRAM6076579840 B，settled恢复idle；无法核对该entry的命令。try2/run3 post entries2216860/2217627以及settled PID2218049的另一项OPTX `verify_runtime_expert_replay_scale_prepared262.py`使组不合格。try3/run1 post entry2224591及settled已退出entry2225945；run3 live PID2229131属于`zan_att`，命令为`python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --act silu --no-bias --iters 128 --no-check-aot-cache --data-init uniform --scale-init uniform`，post busy6/VRAM16901275648 B。没有干预这些外部程序；检查时占用只能说明门槛不合格，不能反推每个计时区间必然重叠，也不能把未识别的已退出PID归因其他任务。

finally等待全机idle，于2026-10-20T02:35:39Z恢复默认两源精确SHA，KFD空/busy0/VRAM176582656 B、无本项live GPU残留。停止专用raw包=`unified_no_delay_alu_perf_stopped_raw_boot1d5146ac_v1.tgz`，1488070 B、181 regular members，SHA=`2b5ddf9d3344db1ebb9b84fc297b06c07c615bbc59e359e01a1ad802582c6602`。下载逐member SHA及独立原命令/27轮精度/完整三轮组/ownership/null medians/56项live/两个source transitions/原rc1和缺失stages验收通过，receipt=`unified_no_delay_alu_perf_stopped_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`47e9c3a9c168437a882eab42138516d0347ba7d41fe9c5fdccbe3756bd87238d`，已上传核SHA。打包时另有外部OPTX2357449和其他kernel测试2362619，无本项owner；当时没有性能测试。

正式可保留E65 GEMM1/GEMM2/e2e仍`107.379/94.3843/256.893 μs`，E64仍`95.3059/60.4664/209.078 μs`；E65距80 μs差27.379 μs。本轮没有新的生产kernel精度或race问题。后续继续从精确默认分析，已关闭候选队列不重开。

### Paired LDS payload读取：资料与既有失败核查后关闭（2026-10-20）

已查阅旧pipeline记录的paired B64、align8、N512交错读取和weight-major traversal实验；没有重复已发生的前端拒绝或spill方案。当前A/B LDS row为2080 B、K128 subtile为1040 B、stage为70656 B。Shader Guide §4.7.2 Tables61–62 pp160–161规定two-address各offset为8-bit，B64乘8，stride64再乘64；普通paired B64的offset reach为2040 B，当前跨row或跨subtile配对通常还需要额外base调整。当前实际FlyDSL rocdl API未提供普通read2 intrinsic，LLVM有DS_READ2ST64_B64 selection，不能据此假定可直接替换当前b128读取。

CDNA5 ISA §15.15 pp738–739中，`DS_LOAD_2ADDR_B64`公式各offset乘8 B；`DS_LOAD_2ADDR_STRIDE64_B64` prose写256 B，但公式乘512 B，存在文档冲突。Shader Guide §5.7.6 Table86 p235的continuous throughput表不能证明本kernel配对后必然收益。本方向没有生成新snapshot、编译或GPU dispatch。

### 关闭VOPD：actual native未减少VGPR-MSB，拒绝（2026-10-20）

独立snapshot仅为GEMM1增加`"amdgpu-enable-vopd": False`，post-RA scheduler和delay-ALU pass保持，wrapper及GEMM2精确默认；98个function AST完全相同。persistent源码SHA=`a8ea77eab4838d6afa4e5a3d2d2daf9f0e44baf4ff731b09d85d31966df86eef`。实际a07-3 LLVM bool选项和CreateVOPD pass已核对，源码摘录见`vopd_remote_llvm_excerpts_v1.json`；硬件依据为Shader Guide §4.6.8 pp134–135、§5.3.2 p218及§5.3.5 pp221–222，只支持配对及bank规则，不能从规则推定性能瓶颈。

固定mode0→1 no-dispatch队列的两个child均rc0、120秒内；每mode实际E65→E64→E65复用cache1/同compiled object/同完整ELF、原runtime count ABI。事前native门槛比较拆开VOPD后的primitive operations，要求原数值/DS/TDM/fences和zero spills/private保持、寄存器及各steady primitive数量不增，并且总MSB及至少一个steady interval的MSB减少。

| actual native metric | 默认mode0 | no-VOPD mode1 |
|---|---:|---:|
| executable instructions | 4531 | 4712 |
| unpaired primitive operations | 4753 | 4712 |
| VOPD pairs | 222 | 0 |
| VGPR-MSB sites | 375 | 375 |
| allocated VGPR /SGPR | 486 /68 | 486 /68 |
| LDS B /private B | 282624 /0 | 282624 /0 |
| VGPR /SGPR spills | 0 /0 | 0 /0 |

候选完整ELF=`aeeb1c61050032d67ff2b135634e8b0a6ff594554b1f468dd00185ec6007bf77`。Normalized opcode唯一delta为`s_delay_alu −41`，11个DScnt0、原Tensorcnt/barrier sequence和有限physical DS CFG通过。四owner共8个backedge：owner0短/长primitive为185/198，其他owners为184/197，两mode逐interval完全相同；每interval的MSB仍28/32，raw仅因拆开两对VOPD而多2条。因此原拒绝理由为`VOPD removal did not reduce total VGPR-MSB switches`和`No steady owner interval reduced VGPR-MSB switches`。外层rc1为native门槛拒绝，`compile_finished.finished=true/execution_failure=null`；没有GPU精度、性能、ATT或live部署，不重开队列。

原始包=`unified_no_vopd_native_raw_boot1d5146ac_v1.tgz`，628821 B、53 regular members，SHA=`0e80c9aa456787bb6838c60bf05752ba3cf5e9072ddb92141be6cc1df94a0738`。最初下载checker因JSON mode keys为字符串、本地qualify为整数而在最后字段比较失败；原失败及原checker保留。v2仅normalize该JSON key表达，重新核验全部member SHA、原ELF/ISA/launcher/resources、source manifests、原命令/timeout/同object、DS/fences及原拒绝报告，没有重复编译或GPU程序。独立验收receipt=`unified_no_vopd_native_downloaded_acceptance_boot1d5146ac_v1.json`，SHA=`d63616a3ff1d7d12f29ff771e088a558331eb322079e8eb6927d6052cc1b9072`，已上传核SHA。

默认源和正式结果保持：E65 `107.379/94.3843/256.893 μs`、E64 `95.3059/60.4664/209.078 μs`，E65距80 μs差27.379 μs。native static统计不是新增kernel性能数据；本轮未发现新的生产kernel race。

## 优化收尾：B LDS地址实验关闭，保留已验收最优版（2026-10-20）

用户要求将手头实验做完后结束优化。本节关闭最后的B地址实验，不启动新的优化候选，也不重开既有关闭队列。三个独立snapshot只调整B load消费点的地址表达，普通DS loads、有效数值计算及同步语义保持；所有编译或GPU child均限制120秒。

| B地址候选 | executable instructions | 地址primitive | VGPR-MSB sites | VGPR /SGPR | 结论 |
|---|---:|---:|---:|---|---|
| 共用默认 | 4531 | 222 | 375 | 486 /68 | 保留 |
| `unified_b_base_tied` | 4434 | 191 | 310 | 498 /68 | native资源门槛拒绝；VGPR按16粒度从496增加到512，未跑GPU精度或性能 |
| `unified_b_lane_tied` | 4536 | 221 | 407 | 486 /70 | steady primitive及SGPR增加，native拒绝；未跑GPU精度或性能 |
| `unified_b_subtile_tied` | 4488 | 195 | 386 | 486 /68 | native通过，完整精度通过；没有合格的候选性能数据，未接入生产 |

最后的`subtile_tied`在B load消费点使用空tied asm identity分离K128 subtile base。CPU实际地址映射覆盖28672 cases、917504 dwords及2048项modulo32检查，均与原实现相同。unpaired primitive由4753降到4711，LDS282624 B、private0、zero spills、180 B kernarg和runtime expert count ABI保持。必须同时报告其MSB总数增加375→386、steady短/长区间28/32→32/36，不能将它描述为MSB减少。candidate完整ELF=`b6a9b7cb2a2669502e298569d2be84c2c93b6a03963026c644ad7979f6c47e49`。

在a07-3/dsr1_ep4完成snapshot双seed0/17的E65/shared1→E64/shared0严格验收，共112项有效payload/ScaleA/GEMM2/MoE零字节差异，原reference、eager/graph及same-address dynamic input/psum/empty专家均通过。每个seed的两规模复用同compiled object/cache1/同candidate完整ELF。该精度证据不能替代正式性能对照。

固定六组三轮性能队列停在默认E65前控制：按事前上限尝试三组，三组均有ownership不合格轮，九次原benchmark都rc0、原精度True、未超时，但三列中值全部null。外层rc1=`Three ownership-contaminated whole groups: E65 mode0`，`execution_failure=null`。没有candidate live部署、normal live binding、candidate性能或后控制；无法判断该候选提速或回退，队列按用户收尾要求关闭。

| 默认E65前控制 | GEMM1原三轮 μs | 每轮ownership合格 | 合格GEMM1 /GEMM2 /MoE e2e中值 |
|---|---|---|---|
| try1 | 108.252 /108.819 /107.105 | True /False /False | null /null /null |
| try2 | 107.295 /106.797 /107.942 | True /False /False | null /null /null |
| try3 | 108.684 /107.624 /108.062 | True /True /False | null /null /null |

原post/settled所有权记录中有`/tmp/b32_work/pilot.py`及另一独立OPTX benchmark。已退出的KFD entries不能猜测来源；未干预外部任务，不将检查时负载直接等同于每个计时区间必然重叠。原控制数据保留以供审计，不拼轮、不用单轮替换最优正式中值。

所有以下文件都位于`.codex_tmp/dsr1_e65_gemm1_20261017/`，raw包已经下载逐member SHA及实际ISA/ELF独立验收，receipt已上传核对：

| 证据 | raw包 /regular members /B | raw SHA256 | 下载验收receipt SHA256 |
|---|---|---|---|
| base tied native拒绝 | `unified_b_base_tied_native_raw_boot1d5146ac_v1.tgz` /59 /687665 | `2e10e531c0421c8c72e030214da2211eeb94bce8fd412f5c168e417058fc3964` | `2b9711ccb5a85a5c23de90645d79ed718e4575a3ca4d0cbc13941ee42433288b` |
| lane tied native拒绝 | `unified_b_lane_tied_native_raw_boot1d5146ac_v1.tgz` /56 /695836 | `ebe3f53a039b29b850fb2f871433216dd7fb24b1a76964a353f998dbc56f3c3d` | `718ed3a683ae9fa723e722a1f7ba58a517ccb82f83e1df34eeab3f59f6a66a9c` |
| subtile tied native | `unified_b_subtile_tied_native_raw_boot1d5146ac_v1.tgz` /56 /694055 | `89af0734aa73f741e963a93c7c8c72a5c7ae1c4b0b4b126639de437a96898028` | `b268b46c9efe42f563b4808829f86d95ee3354d6c9ea4ea2f289ca981477e91d` |
| subtile tied双seed完整精度 | `unified_b_subtile_tied_correctness_raw_boot1d5146ac_v1.tgz` /35 /269811 | `1af533b354e42b7b0b1a866e156c9ba793015f78e9b07d15e89bb996783f9a55` | `49ee117365ad273ae94a7336ac4000948ddcd98665c89a1b853edac327d0f364` |
| subtile tied性能队列关闭 | `unified_b_subtile_tied_perf_closed_raw_boot1d5146ac_v1.tgz` /73 /1516975 | `a744f827b382ed5bce065cd076be4be133bfaf143df2f63a6cfd5e34dc81902a` | `6637b0539171a06e8c1feb7be7bac4f8b0f83c240a73880e8e2e110951d5eccd` |

一次性复现入口分别为`run_unified_b_base_tied_compile_host_v1.py`、`run_unified_b_lane_tied_compile_host_v1.py`、`run_unified_b_subtile_tied_compile_host_v1.py`及`run_unified_b_subtile_tied_perf_host_v1.py`。原label与已封存receipt不可覆盖或重开；需要独立复现时采用新的证据目录和label。

保留的最优版仍为共用runtime-E /lookup7 /group8，完整ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`：

| random non-balanced workload，T1536 /7168 /2048 | GEMM1 μs | GEMM2 μs | MoE e2e μs |
|---|---:|---:|---:|
| E65 /topk9 /shared1 | 107.379 | 94.3843 | 256.893 |
| E64 /topk8 /shared0 | 95.3059 | 60.4664 | 209.078 |

E65未达到80 μs，距目标27.379 μs，也未达到最初90 μs目标；相对历史原HEAD153.405 μs约改善30%。后续控制批次没有取代上述正式验收。原有效routes、全部有效K128更新、SiLU、BF16 rounding、FP4/E8M0、canonical布局和精度标准保持。

瓶颈证据和边界：已验收group8 ATT绑定历史E65-only ELF `7c643f...`，并非最新runtime-E完整ELF；其中448/672/896动态WMMA组compute span P50为20714/21817/22177 cycles，epilogue为4036.5/4123/4195 cycles，compute DS wait累计为832/813/2022 cycles、WG wait为2711.5/1511/1082 cycles。裁剪padding后compute span未按WMMA数同比缩短，说明剩余工作包含operand搬运、地址/控制和必要WG同步。最新默认native仍有375个MSB sites、steady短/长区间28/32 sites，以及VGPR486、LDS282624 B。Shader Guide §3.3.2.1 p32、§4.3.7.2.4 p90、§5.7.6 p235分别支持资源粒度、DS completion和LDS吞吐/依赖规则。不能把WMMA issue间隔当completion latency、把不同阶段中值相加换算μs、把不同dispatch EA counters直接对齐，或据逻辑流量断言HBM饱和；现有证据未唯一确定某个硬件FIFO/cache/链路为瓶颈。

优化关闭时的原host记录=`e65_optimization_final_host_state_boot1d5146ac_v1.json`，2026-10-20T04:28:35Z，SHA=`bfbe2af0b04ad010058bee71270798b1e711617fcfba78ba5e117b8d0381b2c5`，本任务进程残留为空。当时外部PID3720150运行`/tmp/b32_work/sweep.py time ... --cube`；该历史状态发生在以下准入补充之前，原记录保留，不覆盖。

## 收尾补充：E65/topk9/shared0复用当前最优kernel（2026-10-20）

用户问：当前最优版能否支持`--experts 65 --topk 9`、没有shared expert，只修改kernel准入条件而无需修改kernel代码？结论：可以。此次生产代码仅修改`aiter/ops/flydsl/grouped_moe_gfx1250.py`中的profile准入及对应debug文字；`grouped_gemm_mxfp4.py`、persistent kernel、routing benchmark和原精度标准逐字节不变。这是功能准入补充，不是继续优化。

原`_find_routed_e64_topk8_token_profile`依严格metadata只接受routed E64/topk8：physical E65/topk9在shared1时映射到routed64/8，在shared0时仍是routed65/9，所以此前没有选到exact t256 profile。新增`_find_e65_topk9_nonbalanced_token_profile`只接受T1536、E65/topk9/shared0、7168/2048、A4W4、SiLU、无bias、non-balanced、非EP且非EP scatter，复用既有`t256_persistent_any_routing`。旧selector的范围和metadata语义保持；其他token bucket或配置不据本次有限验证扩展。

底层最优准入原已覆盖`(n_experts, contiguous_m)=(65,30464)`，kernel从runtime expert count和实际`psum`读取每expert有效行尾，不固定expert64为shared。新场景保留65列score的真实`fused_topk(...,9,True)`，13824条有效route全部来自routed experts，不追加权重1的shared route、不伪造metadata。capture确认两seed共有65个expert count、总routes13824；expert64只按真实top-k出现，未固定每token必选。

CPU执行实际production profile选择块，并与修改前的全部旧routed profiles逐项对照；新入口命中exact t256，原E64/shared0与E65/shared1、各旧token interval、显式旧控制、bias/EP/其他shape以及非法metadata处理保持。执行原底层dispatch AST审计及Python语法检查全部通过。

a07-3/dsr1_ep4先为真实shared0双seed保存原HEAD compute sources的完整对照，再正常imports执行新E65/shared0→原E65/shared1→E64/shared0；每child均`timeout --signal=TERM --kill-after=5s 120s`。两个seed各84项、共168项零字节差异，覆盖有效payload/ScaleA、GEMM2有效行、最终MoE、三次eager/graph canary和三组same-address dynamic random input/ScaleA/psum（含empty专家）。三个规模实际均cache1、同compiled object、同34360 B完整ELF，精确等于保留最优ELF `9a12287...`；没有新增expert/shared编译specialization。

| workload | seed0 logits_diff /rel_l2 | seed17 logits_diff /rel_l2 | 严格字节对比 |
|---|---|---|---|
| 新E65/topk9/shared0 | 3.4699e-06 /2.6344e-03 | 3.4695e-06 /2.6342e-03 | 每seed28项全部0差异 |
| 原E65/topk9/shared1 | 3.7793e-06 /2.7493e-03 | 3.7788e-06 /2.7491e-03 | 每seed28项全部0差异 |
| 原E64/topk8/shared0 | 3.4733e-06 /2.6357e-03 | 3.4766e-06 /2.6369e-03 | 每seed28项全部0差异 |

该补充仅运行correctness-only，所有耗时无效，没有新shared0正式性能数据。不能将shared1的107.379 μs移作新场景性能。完整验证后host对容器root生成的输出执行普通`chmod`失败，外层rc1；四个GPU child、全部数值与绑定检查此前已rc0通过。失败日志原样保留，后续使用`sudo -n chmod -R a+rwX`仅修复本项输出权限，独立复核原日志并关闭，未重跑GPU测试。最终无本任务进程/KFD owner残留。

准入文件SHA由`bd47bf66ec34f7e208e6cc07e50a62ed7121c8d74b892faa4d729481eb56c479`变为`97ac61a37ca72623ed1988c0f573bcb1a5f4793a4f7903076d67526c9d422668`；wrapper仍`798a865d...`，persistent仍`29cfe04e...`。本地worktree改好后，仅上传这处生产文件的stage及本项验证工具，远端使用原SHA守卫、原字节备份和原子替换；无Git clean/reset/checkout/commit/push，未覆盖其他任务修改。

新场景复现命令（其余选项与原任务相同；benchmark的`--shared-experts`默认0，也可明确传0）：

```bash
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=false \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
PYTHONPATH="$PWD" \
timeout --signal=TERM --kill-after=5s 120s \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache \
  --experts 65 --tokens 1536 --topk 9 --shared-experts 0 \
  --model-dim 7168 --inter-dim 2048 --iters 20
```

在a07-3 host全机idle检查后才可使用该命令的性能读数；精度可在外部负载存在时验证。核验工具为`verify_e65_no_shared_parent_v1.py`、`verify_e65_no_shared_triplet_v1.py`和`check_e65_no_shared_admission_v1.py`。新raw=`e65_no_shared_admission_raw_v1.tgz`，52 regular members、508540 B，SHA=`f85cccd8128a8f4bfa3b61469aac1bd025e6471ca18071606508756cdaaa8d6b`；下载独立验收=`e65_no_shared_admission_v1/downloaded_acceptance.json`，SHA=`9363dc47728e00255c1531c1d1d6176f8be6576ebcfeb5b86a743ccf171ab076`。两个较大的parent `.pt`留在host，SHA封存在receipt中，未声称下载。补充关闭记录=`e65_no_shared_admission_v1/closeout_host.json`，SHA=`e00c9ad00eef71d5aa6ff43cf54caab981ce53d9478ee7115f3cee8ecf69bc9f`。原raw、外层权限失败和优化关闭历史均保留；本轮优化结束。
