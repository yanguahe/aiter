# gfx1250 MoE GEMM1：E64/T1536/topk8 non-balanced 优化记录

## 最新状态（2026-10-17）

最新正式成绩为整理版原命令三轮交替测量的中值 **94.888 μs**，同批 baseline 为 **99.152 μs**，减少 **4.264 μs（4.30%）**；距 80 μs 为 **14.888 μs**。整理版样本为 94.888、94.332、94.893 μs，原第 4 批中值 95.065 μs 仍保留，不能用最低单次替代中值。

已将有实测收益的 mg8 整理版写入本地与远端 persistent GEMM1，source SHA256 为 `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790`。目标及十组 fallback ELF 静态核对、完整 routing/tail、buffer/replay 和上述三轮复测均已通过。下文早期“尚未合入”的标题保留各批实验当时的状态；当前本地源码状态以“本地正式源码的当前改动”和“整理版完整验收”为准。

用户明确授权清理 host PID `818565`、`930575` 后，已核对两个进程的完整命令并发送 SIGTERM，二者均已退出。随后完成整理版验收。后续 epilogue、payload、split READY/REUSE、ScaleA cache、FRONT A seed、named DATA、N-task 数量、output arena、wave 映射、更大 MMA_GROUP、完整 K256 提前 refill 和 16-wave geometry 均未确认稳定收益。v69/v70 共享 producer 循环在 CPU 阶段检出 SGPR spill；v71–v79 的完整实验结果见文末。最新 v80 row forward reuse 三轮均快，中值 94.387 μs、同批 review 94.954 μs，改善 0.60%，但硬件资料对 reuse 条件存在冲突，尚未采用。v81 K128 rotation 三轮有一轮回退，未确认稳定收益；四 SIMD ATT 已完成，实际 ELF 与 CPU 候选逐字节一致。v84 的 2/4-row epilogue 分组完整精度通过，筛选 94.922/95.940 μs，后置发现其他 E65 任务，未采用；8-row allocated VGPR496，被本轮 480 门槛拒绝。v85 native 16x16 FP4 WMMA 的 group16 完整精度通过，筛选 97.806 μs，后置发现其他 E65 任务，未采用；group8 allocated VGPR496，仅 CPU 筛选。正式源码与 94.888 μs 中值不变；最新证据与限制见文末。

最新 v86–v88 检查 K128 input ring。v86 的 b4/b8 在 CPU 阶段均新增 4 个 SGPR spills，未运行 GPU；v87 共用首 task 与后续 task 的 steady body 后消除 spills，完整 equivalence/replay 通过，原命令筛选为 108.465/108.056 μs；v88 保留当前 SSA operands 再提前读取下一组，完整精度通过，但筛选进一步回退至 114.566/116.736 μs。四个 GPU 筛选的前置与六秒后置均为全机 idle，均未进入三轮复测或合入。详细版本、复现与原始证据见文末。

v89 保留正式 K256 ring，首 task 与 successor 共用 steady kt0。shared/disjoint output 两项完整精度通过，筛选 94.492/95.021 μs；shared 首批三轮中值 94.357 μs，但第二轮 review 与其他 E65 工作负载存在重叠。收尾补做的六次原命令运行前置与首次六秒后置均 idle，review / shared 中值为 **94.703 / 95.034 μs**，候选三轮均慢，未确认稳定收益，未合入。shared 十组 fallback 的完整 ELF 均相同，target descriptor/metadata 审核通过。正式中值仍为 94.888 μs，源码与证据保留。详见文末及专项 Q&A。

已完成正式最优版四项 SIMD-select thread trace，实际完整 ELF 与验收版一致。采样 complete waves 的 steady K 占 67.17%、三次 N-task handoff 占 14.09%；主要证据指向 TDM/LDS 供数、WMMA 调度与 WG rendezvous 的耦合。采集复现、归档、源码/ELF hash、phase/stall 数据及瓶颈判断单独记录在 [最优版 GEMM1 thread trace 与分析](moe_gemm1_e64_t1536_best_thread_trace_analysis.md)。本次收尾后没有展开新 kernel 优化。

## 目标与记录规则

优化 `E64/T1536/topk8/model_dim=7168/inter_dim=2048` 的 A4W4、SiLU、无 bias GEMM1。输入与权重使用原测试的 random 初始化，routing 使用 `AITER_MOE_EXPERT_BALANCE=false`。模型推理的 shared expert 假设单独计算，本 workload 只计算 routed experts。

目标是在用户指定的 eager profiler 测试口径下，将 GEMM1 稳定优化到 **80 μs 以下**。保留当前 kernel 的通用性、完整计算、BF16 rounding、per32 MXFP4 scale、native FP4 转换及原有精度标准。

本文只将通过精度验证、且同机空闲交替对比确认有收益的优化方法列入“取得性能提升的优化”。每项记录方法、适用条件、源码版本、复现命令、全部正式样本、统计数据及精度结果。ATT 中的耗时仅用于诊断，不作为正式性能结果。失败和未验证实验的原始日志另存，不将它们写成已取得的性能提升。

用户于本轮追加要求：本任务自行启动的测试若超过两分钟未结束，应及时终止。后续 benchmark、quant 和私有 correctness/replay 每个命令均包在 `timeout --signal=TERM --kill-after=5s 120s` 内；120 秒到时向测试 process group 发送 SIGTERM，最多再等 5 秒后 SIGKILL，保留日志及 exit code。GPU 空闲等待不属于测试执行时间。该授权不用于终止其他任务的进程。

## 测试环境与原始基线（2026-10-17）

- 本地仓库：`C:/Users/yanguahe/Documents/code/wk_sp1/aiter`。
- host：`yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`。
- 容器：`dsr1_ep4`。
- 测试仓库：`/data/yanguahe/code/wk_sp1/aiter`，host/container 共享此路径。
- 起始 HEAD：`339589f900b3a880ec447f605886fe803cbf9649`。
- 起始 persistent 源码 SHA256：`b1f0ff87c886915b737cebbb77f61ca67bd213abde41f913c7e9400a251bf6b4`。
- 本地与远端的 selector、launcher、persistent kernel 和原 benchmark 脚本 SHA256 一致。
- 每次正式测试前在 host 检查全部 GPU 的 KFD 进程、sysfs 利用率及 VRAM；有占用时每 6 秒重查。计时期间不并发运行空闲检查。

容器默认安装包指向 `/app/aiter`，所以在指定仓库测试时使用 `PYTHONPATH=$PWD`。首次测试发现指定仓库中的旧 `module_aiter_core.so` 缺少当前源码所需的 `ActivationType.Relu2`。已备份该 `.so`，使用当前源码与容器 PyTorch 重编译。此环境修复不计为 GEMM1 优化收益；下表都是修复完成后、源码未修改时的基线。

### 复现命令

在 `dsr1_ep4` 容器内执行，性能结果须以前述 host 空闲检查为前提：

```bash
cd /data/yanguahe/code/wk_sp1/aiter
PYTHONPATH="$PWD" \
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=false \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=5s 120s \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20
```

默认 warmup 为 5。GEMM1 数据来自同一次完整 MoE eager profiler 的 `device_time_avg`；原脚本剔除第一轮后，GEMM1 `count=19`。未修改计时、采样与精度门槛。

### 基线数据

目标 kernel：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist
```

| 轮次 | GEMM1（μs） | 原测试精度 | 原始日志 |
|---:|---:|---|---|
| 1 | 97.526 | pass，`logits_diff=3.4733e-06` | [baseline_fixed_1.log](../.codex_tmp/gemm1_t1536_20261017/logs/baseline_fixed_1.log) |
| 2 | 99.036 | pass，`logits_diff=3.4733e-06` | [baseline_fixed_2.log](../.codex_tmp/gemm1_t1536_20261017/logs/baseline_fixed_2.log) |
| 3 | 98.502 | pass，`logits_diff=3.4733e-06` | [baseline_fixed_3.log](../.codex_tmp/gemm1_t1536_20261017/logs/baseline_fixed_3.log) |
| **中值** | **98.502** | **全部通过** | |

当前基线距 80 μs 为 **18.502 μs**；达到 80 μs 边界需降低约 **18.78%**。用户此前约 104 μs 的观测与本轮 98.502 μs 不是同轮交替对比，不将差值算作代码优化收益。

## 取得性能提升的优化

### 1. M256 的 cyclic-owner LDS 布局（初步交替验证，尚未合入）

方法：保持 `t256/w2x4/b4/K7168` 和四个 N-task，重新安排每个 K256 stage 的 A、B、ScaleA、ScaleB。wave_m=0 的 A/ScaleA 放在当前 slot 前部，wave_m=1 的 A/ScaleA 放在下一 slot 尾部；B/ScaleB 保留在当前 slot 中部。M256 每 slot 使用 80 KiB，四 slot 共 320 KiB。原 M192 布局保持原样。

全部 K128 WMMA 和每个 accumulator 的 K 累加次序保持不变。输入的 A-preshuffle、M32 ScaleA，以及输出 FP4 payload、M32 ScaleA 的格式保持不变。原 reader/READY/output fences 继续保留。当前仅确认 seed0 目标规模；在更多 routing、复用 buffer 和其他原支持规模验证完成前，不合入生产源码。

候选源码 SHA256：`412d35943c4ca1626ef6d16f5e64519c8fa81fa037d8068cc9f639d3fbeabff9`。

| 交替轮次 | 顺序 | 基线 GEMM1（μs） | cyclic-owner GEMM1（μs） | 减少（μs） |
|---:|---|---:|---:|---:|
| 1 | baseline → candidate | 98.585 | 97.885 | 0.700 |
| 2 | candidate → baseline | 98.707 | 98.163 | 0.544 |
| 3 | baseline → candidate | 98.656 | 97.128 | 1.528 |
| **中值** | | **98.656** | **97.885** | **0.771（0.78%）** |

三轮目标规模精度均通过，`logits_diff=3.4733e-06`。独立 fused-vs-standalone 诊断的 payload、ScaleA、GEMM2 valid rows 和 MoE 输出均为 `mismatch=0`。最低单次筛选为 97.370 μs，不以此代替交替复测中值。97.885 μs 仍距 80 μs 17.885 μs。

复现：候选临时覆盖 persistent 源码后执行前述原 benchmark。下面的 host 脚本执行三轮交替对比，并在退出时恢复基线；它会逐次完成全 GPU 空闲门禁，不修改 benchmark 的计时口径。

```bash
bash .codex_tmp/gemm1_t1536_20261017/pair_candidates_host.sh cyclic80
```

本轮实际交替命令还包含两个候选，完整顺序与数据见原始日志：

```bash
bash .codex_tmp/gemm1_t1536_20261017/pair_candidates_host.sh cyclic80 fence4 round_packed
```

- [candidate source](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic80.py)
- [precision log](../.codex_tmp/gemm1_t1536_20261017/logs/screen_cyclic80_quant.log)
- baseline：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_baseline_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_baseline_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_baseline_3.log)。
- candidate：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_cyclic80_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_cyclic80_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_cyclic80_3.log)。

### 2. cyclic-owner 加 packed BF16 amax（第二批交替验证，尚未合入）

保持第 1 项的 LDS 布局，在 M256 epilogue 中复用已有 `_amax_inputs_from_bf16x16`：对同一批已经 rounded 到 BF16 的值，用 packed `maximumNumber` 得到绝对最大值，再交给原有 E8M0 clamp/peer reduction/rounding 和 native FP4 conversion。K 累加、activation、BF16 rounding 和输出格式均保持原样。

候选源码 SHA256：`43c5164fc405b40f598c7e2ce384325422a3433d17228a3c115b2875ca22b98a`。

| 轮次 | 基线 GEMM1（μs） | cyclic80_round（μs） | 减少（μs） |
|---:|---:|---:|---:|
| 1 | 98.962 | 97.509 | 1.453 |
| 2，反向 | 98.532 | 96.864 | 1.668 |
| 3 | 97.769 | 96.904 | 0.865 |
| **中值** | **98.532** | **96.904** | **1.628（1.65%）** |

这批三轮的目标规模精度均通过；独立诊断 payload、ScaleA、GEMM2 valid rows、MoE output 均为 mismatch=0。复现候选后仍执行本文开头的原 benchmark；交替命令为：

```bash
PAIR_BATCH=paired_v2_repro bash .codex_tmp/gemm1_t1536_20261017/pair_candidates_host.sh cyclic80_round cyclic80_postoff
```

`cyclic80_postoff` 第三轮回退，不列为取得收益的方法。三轮 baseline 已另存为 `paired_v2_baseline_{1,2,3}.log`，保留第一批本地基线日志；脚本后续使用独立 batch 名且拒绝覆盖同名日志。

- [candidate source](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic80_round.py)
- baseline：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v2_baseline_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v2_baseline_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v2_baseline_3.log)。
- candidate：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_cyclic80_round_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_cyclic80_round_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_cyclic80_round_3.log)。

### 3. 四个 64 KiB 数据 segment 与独立 ScaleA/ScaleB 区域（第三批交替验证，尚未合入）

M256 每个 segment 放本 stage 的 A0（16 KiB）、B（32 KiB）和前一个 stage 的 A1（16 KiB），四个数据 segment 共 256 KiB。ScaleA/ScaleB 放到独立的 16 KiB 区域，writer 与 reader 一起修改地址，输入不再额外 padding。`wave_m=wave%2`、`wave_n=wave//2` 仅改变物理 wave 的 M/N 分工映射。完整 K128 WMMA、各 accumulator 的 K 顺序、原有 tensor/reader/output fences 和外部布局均保留。

候选源码 SHA256：`071a77506ac52de3020f739c51c357b3c05fb2fe45bb6a5f9cba534c85fe8b0b`。

| 轮次 | 基线 GEMM1（μs） | cyclic64_scales（μs） | cyclic64_scales_interleave（μs） |
|---:|---:|---:|---:|
| 1 | 97.857 | 97.802 | 97.041 |
| 2，反向 | 100.596 | 97.633 | 95.272 |
| 3 | 99.071 | 96.775 | 94.973 |
| **中值** | **99.071** | **97.633** | **95.272** |

interleave 三轮均优于同轮基线，中值减少 3.799 μs（3.83%）；仍距 80 μs 15.272 μs。此处不将不同批次中值相减当作增量收益。三轮都通过原 MoE 精度门槛，`logits_diff=3.47335e-06`；seed0 独立诊断全部 mismatch=0。

完整 production-launch 等价验证已通过：T1536 的 random seeds 0/17/43，以及 seed17 的 skew、sparse16、hot8、tiny_expert（7 条 route）、balanced；另测 T1537/T1792/T2048 random seed17。每例的 canonical GEMM1 payload、M32 ScaleA、GEMM2 valid rows 和完整 MoE output 均 bitwise mismatch=0。routing sorter 的 row_map 顺序可以不同，因此在 canonical route 顺序比较输出，不要求 row_map 本身相同。详见 [equivalence log](../.codex_tmp/gemm1_t1536_20261017/logs/equivalence_cyclic64_scales_interleave.log)。

```bash
PAIR_BATCH=paired_v3_repro bash .codex_tmp/gemm1_t1536_20261017/pair_candidates_host.sh cyclic64_scales cyclic64_scales_interleave
```

- [candidate source](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_scales_interleave.py)
- baseline：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v3_baseline_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v3_baseline_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v3_baseline_3.log)。
- candidate：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v3_cyclic64_scales_interleave_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v3_cyclic64_scales_interleave_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v3_cyclic64_scales_interleave_3.log)。

### 4. 64 KiB segment、packed BF16 amax 与 MMA_GROUP=8（第四批交替验证，尚未合入）

在第 3 项的 interleave 布局上，组合第 2 项的精确 packed BF16 amax，并将 `sched_mfma` 的 grouping 从 2 改成 8。grouping 只安排同一批完整 WMMA 的发射，不改变任何 accumulator 的 K 累加顺序。native E8M0/FP4 conversion、activation、BF16 rounding、tail、reader/producer/output fences 保持原逻辑。

候选源码 SHA256：`2bc14677d077551b03a3b3d459cdeddc78c7bc8dbdf5555e07611d4da7e5a788`。

| 轮次 | 基线 GEMM1（μs） | interleave（μs） | interleave_round_mg8（μs） |
|---:|---:|---:|---:|
| 1 | 99.500 | 96.261 | 95.065 |
| 2，反向 | 98.500 | 96.240 | 96.392 |
| 3 | 98.133 | 95.013 | 93.763 |
| **中值** | **98.500** | **96.240** | **95.065** |

mg8 三轮均优于同轮基线，中值减少 3.435 μs（3.49%）；距 80 μs 仍有 15.065 μs。第二轮 mg8 比 interleave 慢 0.152 μs，不能据此声称此组合每轮都优于单独的 interleave。93.763 μs 是单轮数据，不替代中值成绩。

三轮原 MoE 精度都通过，`logits_diff=3.4733e-06`。复用真实 production GEMM1 callable 的 buffer/replay 检查也通过：T1536 random、T1536 tiny_expert、T2048 random；对 initial、payload XOR 0x88 的 sign_flip、沿输入方向 roll 32 的 alternate_values，分别比较 eager、baseline graph 和 candidate graph。每种 graph 含 3 次 launch、重复 5 次，有效 payload/ScaleA 全部 bitwise mismatch=0。完整多 routing 等价验证仍待补齐，尚未合入生产文件。

```bash
PAIR_BATCH=paired_v4_repro bash .codex_tmp/gemm1_t1536_20261017/pair_candidates_host.sh cyclic64_scales_interleave cyclic64_interleave_round_mg8
VERIFY_SCRIPT=verify_gemm1_replay.py VERIFY_LABEL=replay_cyclic64_mg8_repro bash .codex_tmp/gemm1_t1536_20261017/verify_candidate_host.sh cyclic64_interleave_round_mg8
```

- [candidate source](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_interleave_round_mg8.py)
- baseline：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v4_baseline_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v4_baseline_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v4_baseline_3.log)。
- candidate：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v4_cyclic64_interleave_round_mg8_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v4_cyclic64_interleave_round_mg8_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/paired_v4_cyclic64_interleave_round_mg8_3.log)。
- [buffer/replay log](../.codex_tmp/gemm1_t1536_20261017/logs/replay_cyclic64_mg8.log)

## 当前瓶颈证据与排除结果

使用指定 `my_code/get_att_b_rocm_sdk.sh` 对原完整 GEMM1 symbol 抓取四 SIMD 的 ATT。52 个 decoded waves 中 32 个执行真实 WMMA，其余为容量 sentinel。每个完整 valid wave 执行 3584 WMMA，即 `4 N tasks × 56 K128 × 16 WMMA`。decoder 未出现 `.long` 或 illegal instruction。

valid wave 周期加权的暴露 stall 归属为：WMMA 12.04%、`barrier_wait` 11.22%、`s_wait_dscnt` 6.99%、DS load 3.70%、`s_wait_tensorcnt` 3.35%。现有 SiLU 使用 `v_tanh_f32`；将其归入 TRANS 后，TRANS 暴露 stall 为 0.24%。这些比例是采样 wave 的指令归属，不能直接相加当作可消除耗时。

当前基线的输出包括 fused SiLU、BF16 rounding、per32 E8M0 scale 和 native FP4 conversion。只计 unfused GEMM、忽略 quant 开销不满足本任务。`masked_mma` 无收益且已从候选路线排除；`drain_carry` 未通过精度验证，未进入性能比较。其他回退候选和完整日志留在专项临时目录，不列为取得收益的方法。

baseline ATT 的真实 GEMM1 ELF metadata：`.vgpr_count=480`、`.sgpr_count=106`、`.group_segment_fixed_size=282624`、`.private_segment_fixed_size=0`、`.vgpr_spill_count=0`、`.sgpr_spill_count=0`。本轮没有证据将主要回退归因于 spill。资源原始输出：[baseline_resources.log](../.codex_tmp/gemm1_t1536_20261017/logs/baseline_resources.log)。

原 ATT：`my_code/thread_trace_runs/gemm1_t1536_baseline_att_20261017`。本地汇总：[trace_analysis_v2.log](../.codex_tmp/gemm1_t1536_20261017/logs/trace_analysis_v2.log)。

第四批 mg8 候选使用同一脚本抓取四 SIMD ATT：`my_code/thread_trace_runs/gemm1_cyclic64_mg8_att_20261017`。32 个 valid waves 中每个完整 wave 仍执行 3584 WMMA。采样 valid-wave 总周期从基线 4,439,910 降到候选 3,966,657；这属于诊断采样，不作为原 benchmark 的性能数字。

| 暴露 stall 的指令归属 | 基线 | mg8 候选 |
|---|---:|---:|
| `s_wait_dscnt` | 6.99% | 4.03% |
| `barrier_wait` | 11.22% | 16.94% |
| `s_wait_tensorcnt` | 3.35% | 5.36% |
| DS load | 3.70% | 5.49% |
| WMMA | 12.04% | 17.15% |

候选 ELF：VGPR=470、SGPR=106、LDS=278528 B、private=0、VGPR/SGPR spills=0。比例变化不能直接相加当作可消除的 kernel 时间；现有证据支持继续研究同步、LDS 复用和 wave 到达不均，未支持将主瓶颈归因于 spill。

- [mg8 ATT analysis](../.codex_tmp/gemm1_t1536_20261017/logs/gemm1_cyclic64_mg8_att_20261017_analysis.log)
- [mg8 ELF metadata](../.codex_tmp/gemm1_t1536_20261017/logs/cyclic64_mg8_resources.log)

后续 `cyclic64_named_data` 实验申请四个 named DATA barriers，保留 WG FREE。其正确性运行未完成；GDB 栈显示停在 `rocr::core::BusyWaitSignal::WaitRelaxed` / `hsa_signal_wait_scacquire`，不是仍在编译。该方案未通过精度门禁，没有有效性能结果，不合入。现场 source/lowered IR、ELF 和 HSA wait 栈已保存；在确认本任务进程可以终止后，继续使用正常基线执行候选验证。

静态 IR/ISA 已定位该实验的死锁原因：Python `wave == 0 or wave == 4`、`1 or 5`、`2 or 6` 在 DSL 编译中只保留左侧 predicate，wave4/5/6 落入了不匹配的 signal 分支。按 emitted IR 的分支计算，四个 barrier 每代实际分别收到 2/6/2/6 次 signal，而 INIT member count 均为 4；其中 barrier0 只能收到 2 次，joined waves 无法通过 DATA wait。候选 `cyclic64_named_data_orfix` 已将这三处改为 DSL 的 `|`；修正版尚未运行验证，不能视为已解决或已取得性能收益。

原实验 ELF 的 `COMPUTE_PGM_RSRC3.NAMED_BAR_CNT=1`（按 4 个 barrier 一组编码），且确实发出四组 INIT/JOIN 指令，因此不能仅凭 `amdgpu.max_num_named_barrier` 的未解析 linker symbol 判断“没有分配 named barrier”。此失败候选还出现 SGPR spill=8、VGPR=500，与已验证 mg8 的无 spill 资源情况不同。

硬件依据：`MI400_Shader_Programming#65.txt` §4.7.1（LDS segments/banks）、§4.10.1（TDM 同 wave 有序及跨 wave 同步限制）、§5.3.2（VGPR source-cache）和 §5.7.6（LDS 发射与延迟）；WMMA reuse 依据 CDNA5 ISA §7.12。布局与调度收益为实测推导，硬件文档没有保证某种布局一定更快。

## 可交付候选的适用范围与静态核对（2026-10-17，GPU 验证待补）

第四批筛选源码修改范围较宽，因此不能直接将该文件整体合入。候选 `cyclic64_mg8_delivery.py` 将三个已有优化限定在原 launcher 支持的 `t256/w2x4/b4/K7168/E64/SiLU`、fused FP4 quant、`quant_wmma_rep=8`、`cluster_m=1`、实际 compute M256 路径，且排除 `partial_m_direct_scale` 和 `partial_m_drain_output`。不对 token 数、routing 分布、输入值、expert 的有效行数增加特例。其他原支持路径继续采用原布局、wave 映射、grouping 和 quant reduction。

- [delivery candidate](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_mg8_delivery.py)，SHA256：`171bee6edf8e1962bdbe858816d55cc1a1a311d842be2aea6fc58128982491d1`。
- [reuse1 candidate](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_mg8_delivery_reuse1.py)，SHA256：`b3d7b6cffe4448025fb711be49f584c260e3039afda4a56f69cc7d56a828319f`。其 WMMA reuse hint 尚待完整正确性及正式交替复测，不列为有效成绩。

在容器中用 `COMPILE_ONLY=1`、`ARCH=gfx1250`、CPU dummy tensors、null pointer 和 `Stream(None)` 编译；只读取生成的 IR/ELF，不创建 GPU executor、不 launch。以下十组 fallback 的整个 ELF（包括 `.text`、`.rodata` 和 `.note`）与冻结 baseline 完全相同：

| fallback 配置 | ELF 与 baseline |
|---|---|
| `t192/q6` | 完全相同 |
| `t256/balanced_m192` | 完全相同 |
| `t256/cluster_m=4` | 完全相同 |
| `t256/E128` | 完全相同 |
| `t256/w2x4`、non-fused BF16 输出 | 完全相同 |
| `t256/w2x2`、non-fused BF16 输出 | 完全相同 |
| GEMM2 launcher、`K2048` | 完全相同 |
| GEMM2 launcher、`K3072` | 完全相同 |
| `partial_m_direct_scale=1` | 完全相同 |
| `partial_m_drain_output=1` | 完全相同 |

目标配置的 production `tdm_b_th=6` 也已核对：delivery 生成的整个 ELF 与第四批已抓取 ATT 的 mg8 对象完全相同。`.text` SHA256 为 `507c1a9aedd0db3e3d06b83063870e6c0e56564228c89ff381f5f71b07748985`，资源仍为 VGPR 470、SGPR 106、LDS 278528 B、private 0、spills 0。最初 dummy 参数用 `tdm_b_th=0`，该对象与 ATT 的 TDM temporal hint 不同；目标核对应使用 6，不能混用两份对象。

证据：[fallback ELF comparison](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_scope_cyclic64_mg8_delivery.json)、[production hint comparison](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_target_th6_comparison.log)。这是静态兼容性证据，不能代替 delivery 自身的完整多 routing、tail、eager/replay 验证，也没有产生新的性能数据。此处保存的是收紧范围后的原始候选；本地合入状态见后文“本地正式源码的当前改动”。

## 已有 ATT 的 K 循环与 N-task 交接分析

仅重新分析已有 baseline/mg8 ATT，不再次使用 GPU。按每 wave 的 16 条 WMMA 划分 K128 block、每 56 个 block 划分 N-task；32 个 valid waves 都具有完整的 3584 条 WMMA。下表是各 phase 的 stall 归属占全部 valid-wave sample cycles 的比例。

| phase / 指令归属 | baseline | mg8 |
|---|---:|---:|
| `steady_K / barrier_wait` | 7.0160% | 10.7250% |
| `steady_K / s_wait_dscnt` | 5.9395% | 3.0171% |
| `steady_K / s_wait_tensorcnt` | 1.6134% | 3.2554% |
| `task_handoff / barrier_wait` | 0.9311% | 2.8108% |

同一个 K128 block 内，相邻 WMMA 发射间隔的平均值由 23.58 cycles 降到 16.99 cycles，两者中值均为 8 cycles；N-task 交接处的平均间隔由 4210 cycles 增加到 4819 cycles，中值由 3932.5 增加到 4657.5 cycles。这支持继续研究 K 循环同步和 N-task 交接。`task_handoff` 包含前一 task 最后一个 K block 的尾部以及 epilogue/prologue，不能称为纯 output TDM 时间，更不能把这些 stall 比例相加当作可消除的 kernel 时间。

脚本：[analyze_trace_phases.py](../.codex_tmp/gemm1_t1536_20261017/analyze_trace_phases.py)。结果：[phase comparison](../.codex_tmp/gemm1_t1536_20261017/logs/existing_att_phase_comparison.json)。复现只读取已有 trace：

```bash
python3 .codex_tmp/gemm1_t1536_20261017/analyze_trace_phases.py \
  my_code/thread_trace_runs/gemm1_t1536_baseline_att_20261017 \
  my_code/thread_trace_runs/gemm1_cyclic64_mg8_att_20261017
```

## named DATA 后续静态实验与当前运行状态

`cyclic64_named_data_owners` 利用 `steady_mid` 已选定的 `my_jobs[0].waves` 区分 A/ScaleA 与 B/ScaleB owner，再按 wave parity 选择 DATA signal，减少热循环中重复判断完整 wave-id 分支。CPU 编译通过，VGPR 从 orfix 的 500 降到 486，SGPR 102、LDS 278528 B、private 0，但 SGPR spill 仍为 8；ISA 的 `v_writelane_b32` / `v_readlane_b32` 确认此处是 SGPR 保存到 VGPR lane。没有 GPU 正确性或性能结果，不启用此同步实验。证据：[owners resources](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_named_owners_resources.log)。

上述两个阻塞进程已按用户明确授权核对并终止，现场日志/IR/ELF 保留。named DATA owners/orfix 仍只是静态候选，未追加 GPU 实验；当前正式源码与成绩见“整理版完整验收与正式复测”。

## ScaleA 输出候选的静态筛选

针对 ATT 中增大的 N-task 交接等待，新增两类仅改变 ScaleA 写出方式的候选，完整 activation、BF16 rounding、native E8M0/FP4 和 M32 global 地址均保留。以下只有 CPU 编译与 ISA 资源结果，没有新增 GPU 性能数据。

TDM 版本把 ScaleA bytes 写入 canonical M32 LDS source，再由每个 M half 的一个 wave 发出 TDM。其 source 与下一 N-task 的四个 input-scale stage 分离。下一 task 的 READY wait 按 owner 实际 outstanding output 数设置：ScaleA output owner 留 2 个，其余 wave 留 1 个；统一留 2 个会使没有第二个 output 的 wave 过早读取输入。分别编译独立 ScaleA source、released stage-2 source 两种布局。

packed-store 版本保持原 TDM 数量和 wait distance：各 N64 wave 将原始 ScaleA byte 写入 LDS 的相邻字节，WG barrier 后每个 M half 的 wave_n=0 写出 assembled dword。进一步让 32 个 lane 分别写 M32 行，减少静态 unroll 指令，但该版本产生 SGPR spill。

| 候选 | VGPR | SGPR | LDS（B） | SGPR spills | 静态 WMMA 指令数 | 状态 |
|---|---:|---:|---:|---:|---:|---|
| 已验证 mg8 delivery | 470 | 106 | 278528 | 0 | 288 | 第 4 批已有性能与部分 replay 证据 |
| `cyclic64_mg8_tdm_scale` | 474 | 107 | 279552 | 12 | 288 | 不进入正式实现 |
| `cyclic64_mg8_tdm_scale_stage2` | 474 | 107 | 279552 | 12 | 288 | 不进入正式实现 |
| `cyclic64_mg8_scale_store32` | 472 | 106 | 279552 | 0 | 288 | 保留为待 GPU 验证候选 |
| `cyclic64_mg8_scale_store32_lanes` | 486 | 102 | 279552 | 8 | 288 | 不进入正式实现 |
| `cyclic64_mg8_scale_store32_lanes_stage2` | 486 | 102 | 278528 | 8 | 288 | 不进入正式实现 |

288 是包含 mutually exclusive paths 的静态 ISA 数量，不是每 wave 的运行计数；完整运行仍须验证 3584 WMMA 和实际输出。新增 spill 仅说明需要额外保存 SGPR，不能代替性能测试宣称必然回退。

证据：[TDM scale inspection](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_tdm_scale_inspection.log)、[packed store inspection](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_scale_store32_inspection.log)、[32-lane variants](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_scale_store32_lanes_inspection.log)。生成器和候选保存在专项目录，正式 benchmark/计时脚本没有修改。

## 本地正式源码的当前改动

已将第 4 批有收益的 mg8 delivery 整理后写入本地 `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py`，仅修改 `launch_gemm_a8w4_tdm_fused_persistent`：M256 内部 cyclic data/scale arena、wave M/N interleave、packed BF16 amax、MMA_GROUP=8 和匹配的 LDS reader/writer 地址。原同步协议、输入输出布局、所有 K contribution、GEMM2 launcher 及其他 module definitions 保留；named DATA、TDM ScaleA 和 packed global-store 实验均未合入。

整理后的 source SHA256：`b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790`。production `tdm_b_th=6` 的整个 ELF 与原 delivery 完全相同，证据：[reviewed target ELF](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_delivery_review_comparison.log)。AST 核对确认只改变 GEMM1 launcher，`git diff --check` 通过。其他十组 fallback 的整理版整个 ELF 核对也已通过，证据：[reviewed scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_scope_cyclic64_mg8_delivery_vs_cyclic64_mg8_delivery_review.json)。

本地与远端正式源码均为上述整理版，完整多 routing/tail 与 eager/replay 验收及原命令三轮交替复测已完成；最新中值 94.888 μs。没有 commit/push，80 μs 目标仍未达成。新的 CPU/GPU 筛选候选仍保存在专项目录，不混入正式源码。

固定 tmux session 中的 [wait_review_validation_host.sh](../.codex_tmp/gemm1_t1536_20261017/wait_review_validation_host.sh) 已完成等待、source hash 门禁与两项完整正确性检查。其后性能阶段由 exact-file permission 修复的 resume 脚本完成，详见下文，未覆盖未知变更。

## disjoint ScaleA stores 的完成等待与 LDS barrier scope

每个 persistent N-task 写入 canonical M32 ScaleA 的不同列。后续 task 读取 `arg_scale_a`，不读取 `arg_quant_scale`；GEMM2 在整个 GEMM1 kernel 完成后才消费输出。新候选把 task 内的显式 `s_wait_storecnt(0)` 合并到 kernel 结束：保留全部 ScaleA stores、全部输出 TDM，以及最终 `tensor_wait(0)` 后的显式 `s_wait_storecnt(0)`。

仅移动显式 wait 不足以消除 task 交接处的 global store completion 等待。默认 `gpu.barrier()` 同步 global 和 LDS address spaces，LLVM 因而在 epilogue rendezvous 前生成 `s_wait_storecnt_dscnt(0)`。候选 `cyclic64_mg8_lds_epilogue_finalwait` 将该处 barrier 限定为 `#gpu.address_space<workgroup>`：原 workgroup rendezvous、LDS 可见性、READY/REUSE 和 TDM completion 协议均保留，ScaleA global stores 在 kernel 结束时完成。

ISA 已确认此处的 `s_wait_storecnt_dscnt(0)` 变为 `s_wait_dscnt(0)`。目标 `.rodata` descriptor 与 `.note` metadata 完全相同；十组 fallback 的完整 ELF 全部与 mg8 整理版一致，整理版 fallback 已与冻结 baseline 核对一致。此候选尚未执行 GPU 正确性或性能测试，未合入本地正式源码。

| 候选 | VGPR | SGPR | LDS（B） | spills | 静态验证结论 |
|---|---:|---:|---:|---:|---|
| `cyclic64_mg8_storewait_final` | 470 | 106 | 278528 | 0 | 显式 wait 已移到结束，但默认 barrier 仍等待 global store completion |
| `cyclic64_mg8_lds_epilogue_finalwait` | 470 | 106 | 278528 | 0 | 限定 LDS scope，十组 fallback 完整 ELF 一致 |
| `cyclic64_mg8_scale_store32_lds_finalwait` | 472 | 106 | 279552 | 0 | 组合此前 ScaleA packed store，增加 LDS pack/read，保留为备选 |

纯 LDS-scope 候选 source SHA256：`fa6cf9a356515ee1c11ce0e5046f6052d625c354c32936eb9a583ca360d4f01f`；`.text` SHA256：`0b7e9f9c17f804c554802fe19170d3a782af838fca43a993c482da32f7d62a40`。其 288 条静态 WMMA、8 条静态 `global_store_b8`、11 组 WG signal/wait 和原 TDM 操作全部保留。

已有 mg8 ATT 中，`task_handoff / s_wait_storecnt_dscnt` 的暴露 stall 占 valid-wave sample cycles **1.0914%**，显式 `s_wait_storecnt` 占 **0.00484%**。combined wait 同时包含 DS completion，不能据此断言 1.0914% 都来自 global stores，更不能直接换算为可优化的 μs。这个方向只用于验证是否能减少无数据依赖的等待。

生成、编译与静态范围核对的复现命令：

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v22.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v23.py
# In dsr1_ep4; CPU arguments, no executor or GPU launch.
python3 -u .codex_tmp/gemm1_t1536_20261017/compile_candidate_cpu.py \
  cyclic64_mg8_lds_epilogue_finalwait --tdm-b-th 6
# On the host; all compilations still run inside dsr1_ep4.
bash .codex_tmp/gemm1_t1536_20261017/compile_storewait_scope_host.sh
```

证据：[storewait-only ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_storewait_final_inspection.log)、[LDS-scope ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_lds_epilogue_finalwait_inspection.log)、[十组 fallback 核对](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_lds_storewait_scope_checked.json)。首次使用通用比较脚本时，预期变化的 target 也被列为 mismatch；十组 fallback 当时全部一致。后来显式排除 target、检查这十组 fallback 的脚本返回 0，未将预期 target 差异作为失败的兼容性结果。

硬件依据：`MI400_Shader_Programming#65.txt` §4.3.7.2（printed pp.86–89）分别定义 STOREcnt 和 DScnt completion，并说明 counter overflow 会由硬件阻塞发射；§4.3.2 记载 `S_ENDPGM` 等待 outstanding 指令。本候选仍保留显式 final store wait。LLVM `GPUBarrierOpLowering` 的 address-space lowering 与本机生成 ISA共同支持上述 barrier scope 结论；文档未保证该改动一定有性能收益。

## named DATA 指针选择：仅静态实验

候选 `cyclic64_named_data_ptrselect` 使用 compiler-assigned barrier0 指针，加 uniform 的 16/32-byte offset 选择 owner 对应的两个 DATA barriers，减少重复 wave-id 分支。INIT、JOIN、member count=4、WG FREE 和全部计算保持原逻辑。CPU 编译通过，资源为 VGPR **487**、SGPR **102**、LDS **278528 B**、private **0**、SGPR spill **8**。

此处仍有 epilogue 的 8 个 SGPR lane 保存/恢复，未消除 spill。没有 GPU 正确性或性能数据，未启用新的 named barrier 实验，也不宣称其有实测回退。原实验的 signal-count 错误与这个候选能否正确运行是两件事：仍须核对 emitted branches、各代 signal count 和受控 GPU 正确性。证据：[ptrselect ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_named_ptrselect_inspection.log)，生成器：`build_candidates_v24.py`。

## 普通 cyclic 地址下的 drain carry 复用：未采用

早期 `drain_carry` 在原 baseline 的 opaque carry target 路径上未通过精度。当前 mg8 整理版的 M256 cyclic 路径用普通 `buf_ptr` 计算 carry target，A/ScaleA owner-stage rotation 与 B/ScaleB stage 都对 lowering 可见，因此重新编译了只在 `M256_FUSED_LAYOUT` 内使用 carried rmem 的候选。其他路径仍保留原 drain reload。

候选仅让 drain tiles 1..3 的 subtile-0 消费前一 tile 已载入的 A/B/ScaleA/ScaleB，不修改全部 K contribution、每个 accumulator 的 K 顺序、register ping-pong、reader/TDM/output fences。`cyclic64_mg8_normal_drain_carry` 与组合 LDS-scope 的 `_lds` 版本都已编译通过：VGPR **474**、SGPR **106**、LDS **278528 B**、private/spills **0**，288 条静态 WMMA 和原 TDM/store/barrier 数量保留。

但最终 ISA 的 `ds_load_b128` 均为 **456**，与 mg8 整理版相同；计入 dual-address 指令对应的两个 dword 后，scale DS load word 总数也未减少。编译器已经能在原 drain unroll 中消除一部分相同-address reload，改为 carry 主要改变 load 时机、scale load 合并和寄存器分配，不能把源码少调用 `load_lds_data` 当作真实减少相同数量的物理 LDS fetch。VGPR 从 470 增至 474，静态指令从 4381 增至 4393。因此暂不采用此改动，GPU 正确性与性能均未验证。

source SHA256：普通版本 `a515316e04e66175cc2449ff0c237dd12fc3241a28d36c05e3b917200c72cae0`，LDS-scope 组合 `d666d334b5d2e46d8ac2b5b4569ae6f1e9306d0d04517ab5ec457ba32b643099`。生成器：`build_candidates_v25.py`；证据：[drain carry ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_normal_drain_carry_inspection.log)、[十组 fallback 与 target 字段核对](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_normal_drain_carry.json)。十组 fallback 完整 ELF 全部一致；target metadata 仅 VGPR count 由 470 增至 474，descriptor 仅 instruction prefetch 长度改变，ABI/layout/control fields 保持一致。

## FP4 payload 合并写入 LDS：待 GPU 验证

保留原 SiLU、BF16 rounding、E8M0 scale 与每条 native `v_cvt_scalef32_pk8_fp4_bf16`。每个 lane 仍生成原 `packed_i32`；lane 与 `lane^16` 交换它之后，kgrp0 lane 将两组原 halfword 合成 canonical dwords，再写入原 payload LDS 字节地址。exchange 在收窄到 kgrp0 的 predicate 之前执行；M16 有效性 predicate 对两个 kgrp 相同，因此 partial expert tail 不需要从 inactive peer 取数据。两种版本均只改变 bounded M256 fused GEMM1 的内部 staging，不改变对 GEMM2 暴露的格式。

dword 版本按每个 conversion 生成两个 dword；b128 版本合并该行两条 conversion 的四个 dword，以一次 b128 写出该 lane16 的完整 16 字节。CPU 字节/地址代数检查覆盖 **65536** 组 random packed values、8 waves、8 wm 与16 lane16，全部 mismatch=0，且 b128 地址 16-byte aligned。此检查验证拼接公式和地址映射，不是 GPU quantization、shuffle 或完整 MoE 正确性结果。

| 候选 | 静态 payload stores | 静态总指令 | VGPR/SGPR | LDS（B） | spills |
|---|---|---:|---|---:|---:|
| mg8 整理版 | 16 `ds_store_b16` + 16 `ds_store_b16_d16_hi` | 4381 | 470/106 | 278528 | 0 |
| `cyclic64_mg8_payload_store32` | 16 `ds_store_2addr_b32` | 4544 | 470/106 | 278528 | 0 |
| `cyclic64_mg8_payload_store128` | 8 `ds_store_b128` | 4489 | 470/106 | 278528 | 0 |

`_lds` 组合版本同时采用前述 LDS-only epilogue scope 和 final store wait，资源和 payload store 数相同。全部版本仍有 288 条静态 WMMA、456 条 `ds_load_b128`、8 条 `global_store_b8`、11 组 WG rendezvous 以及原 TDM 操作。store 指令数减少的代价是 shuffle、packing 和 EXEC 管理，不能单凭指令数判定 kernel 更快。所有 payload 候选尚无 GPU 正确性或性能数据，均未合入生产源码。

source SHA256：dword `e339b28b60b1ecc40e94333311629b7a9a3dfdd65ac823692e5a42043b7b5aaf`，b128 `3783354136f2098f24891907b19add9c46b623ef715a499a2658ec7d9f580bd6`；b128 `_lds` 为 `baa41a2c1cf742fb7be6ea6516bf1f0da8132506de4e80f904cdbb6f3ffebe29`。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v26.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v27.py
# In dsr1_ep4; compile-only.
python3 -u .codex_tmp/gemm1_t1536_20261017/compile_candidate_cpu.py \
  cyclic64_mg8_payload_store128 --tdm-b-th 6
```

证据：[dword ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_payload_store32_inspection.log)、[b128 ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_payload_store128_inspection.log)、[字节/地址代数](../.codex_tmp/gemm1_t1536_20261017/logs/payload_store_byte_algebra.json)。硬件依据：CDNA5 ISA §15.14 `V_PERMLANEX16_B32`（printed p.581）定义跨相邻16-lane rows 的 gather；`MI400_Shader_Programming#65.txt` §5.7.6（printed p.235）记载 LDS indexed throughput 与 bank conflict 影响。两份文档未给出上述 composite epilogue 的性能保证。

### 合并 store 的 byte packing 与寄存器存活范围

`cyclic64_mg8_payload_store128_perm` 以两条 `V_PERM_B32` 替代每次 conversion 的 shift/mask/OR 拼接。byte pool 为 `{peer_i32, packed_i32}`，selector `0x05040100` 拼出第一组 native halfwords，`0x07060302` 拼出第二组；原 native FP4 conversion、BF16 rounding、E8M0 scale、kgrp exchange 与 canonical LDS 地址保持原样。CPU byte selector 代数检查 **131082** 项全部 mismatch=0，包括 random 与边界 bit patterns；该检查不验证 GPU shuffle 和 MoE。

全 lane b64 版本让两个 kgrp 各写原 canonical 行的不同 8-byte half，以减少 kgrp0-only payload predicate。初版携带两组 lane-dependent selectors；第二版改成固定 selector，选择 low/high 的 native operands。第三条路线让每个 conversion 的两个 kgrp 各写一个 canonical dword，仅使用 lane-derived selector。三者都完成 CPU 编译，但当前均产生 SGPR spills，暂不采用。

| 候选 | 静态 payload stores | 静态总指令 | VGPR/SGPR | SGPR spills | 状态 |
|---|---|---:|---|---:|---|
| mg8 整理版 | 32 halfword stores | 4381 | 470/106 | 0 | 已有第 4 批实测 |
| b128 shift/mask | 8 `ds_store_b128` | 4489 | 470/106 | 0 | 十组 fallback 完整 ELF 一致，待 GPU 验证 |
| `payload_store128_perm` | 8 `ds_store_b128` | 4443 | 470/106 | 0 | `_lds` 版本十组 fallback 完整 ELF 一致，待 GPU 验证 |
| `payload_store64_perm` | 8 `ds_store_b64` | 4508 | 488/102 | 8 | 暂不采用 |
| `payload_store64_constperm` | 8 `ds_store_b64` | 4532 | 486/102 | 8 | 固定 selector 未消除 spill，暂不采用 |
| `payload_store32_laneperm` | 8 `ds_store_2addr_b32` | 4510 | 480/102 | 8 | 暂不采用 |

表中各 `_lds` 组合具有相同的资源和静态总指令数，LDS 都为 278528 B，private segment/VGPR spills 都为 0。SGPR spills 的 ISA 表现为八组 `v_writelane_b32` / `v_readlane_b32` 保存和恢复。b64 初版的十组 fallback 完整 ELF 也全相同，但 target 被资源门槛拒绝；scope script 返回 2 的原因是 spills 与 metadata contract 差异，而不是 fallback 失败。没有任何新 payload 候选的 GPU 正确性或性能结果，不把静态 store 数减少计入性能提升。

b128-perm source SHA256：普通版本 `9fd85e0ddb4fe4de0b2c66cb24d6e48f1f4cc0ce9301ec675751f8fc07adcd9a`，LDS-only 组合 `9f959f0ec96c046e9e5caa59c1de5c12573eb2f91df298533a2b954c500048cb`；`.text` SHA256 分别为 `5256b7f54dfdbdd8c2bc27314dd05499983b8ae3af0a4b9c3d380ce7246bc860`、`70ea06d8ceb6e69e0756a4d3c342a5da5c25069e7930adfe6774ddf9ccecaae1`。证据：[b128-perm ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_payload_store128_perm_inspection.log)、[byte selectors](../.codex_tmp/gemm1_t1536_20261017/logs/payload_perm_byte_algebra.json)、[b128-perm LDS scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_payload_store128_perm_lds.json)、[b64 ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_payload_store64_perm_inspection.log)、[fixed selector ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_payload_store64_constperm_inspection.log)、[dword ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_payload_store32_laneperm_inspection.log)。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v28.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v29.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v30.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v31.py
# In dsr1_ep4; CPU arguments only, no GPU launch.
python3 -u .codex_tmp/gemm1_t1536_20261017/compile_candidate_cpu.py \
  cyclic64_mg8_payload_store128_perm_lds --tdm-b-th 6
# On host; all fallback compilation and ELF audits run in dsr1_ep4.
bash .codex_tmp/gemm1_t1536_20261017/compile_bounded_candidate_scope_host.sh \
  cyclic64_mg8_payload_store128_perm_lds
```

硬件依据：CDNA5 ISA §15.14 `V_PERM_B32`（printed pp.573–574）定义 low input 在 `S1`、high input 在 `S0`，selector bytes 0..7 按 byte 选择；这支持拼接公式。新候选的速度只能由后续原命令的空闲交替测试确定。

## 整理版完整验收与正式复测（2026-10-17）

用户授权清理两个阻塞进程后，原实验恢复 baseline，再将整理版放入真实 production launcher 验收。11 组完整 MoE 输入覆盖 T1536 random seeds 0/17/43、skew、sparse16、hot8、tiny_expert、balanced，以及 T1537/T1792/T2048 random seed17。canonical GEMM1 FP4 payload、M32 ScaleA、GEMM2 valid rows 和 MoE output 全部 bitwise mismatch=0；原 `LOGITS_DIFF_TOL=0.01` 未修改。row_map 是 sorter 的物理行排序，不作为逐元素相等要求。

buffer/replay 覆盖 T1536 random、T1536 tiny_expert 和 T2048 random；initial、sign_flip、alternate_values 在同一组输入/输出 buffer 上执行 eager、baseline graph、candidate graph，输出先 poison，每种路径重复 5 次。graph 每次含 3 次 GEMM1 launch；payload/ScaleA 全部 mismatch=0。

验收后运行原 benchmark，第二轮逆序；下表每个值都是原 eager profiler 的 `device_time_avg`、`count=19`，不是私有 correctness launcher 的计时。

| 轮次 | 顺序 | baseline GEMM1（μs） | 整理版 GEMM1（μs） | 减少（μs） |
|---:|---|---:|---:|---:|
| 1 | baseline → review | 99.498 | 94.888 | 4.610 |
| 2 | review → baseline | 97.865 | 94.332 | 3.533 |
| 3 | baseline → review | 99.152 | 94.893 | 4.259 |
| **中值** | | **99.152** | **94.888** | **4.264（4.30%）** |

三轮原 MoE 精度均通过，`logits_diff=3.4733e-06`。性能前每次完成全机 GPU/KFD/VRAM 门禁；上一运行结束后的短暂 sysfs busy 尾值为 1–3%，下一次检查等待它归零再启动，计时期间没有轮询。测试完成无 KFD 进程残留，远端保留整理版，本地/远端源码 SHA256 相同。

原等待脚本在完整 equivalence/replay 已成功后，因 host 对其它 root-owned CPU 编译日志递归 chmod 失败而退出，未开始性能测试。已以 exact-file container chmod 修复后续流程，沿用两份已通过的精度日志，用 [resume_review_bench_host.sh](../.codex_tmp/gemm1_t1536_20261017/resume_review_bench_host.sh) 完成六次原命令，不重复精度计算或改动计时口径。

- [完整 equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_20261017_equivalence.log)、[完整 replay](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_20261017_replay.log)。
- baseline：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_20261017_baseline_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_20261017_baseline_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_20261017_baseline_3.log)。
- review：[第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_20261017_review_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_20261017_review_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_20261017_review_3.log)。
- [复测 controller log](../.codex_tmp/gemm1_t1536_20261017/logs/delivery_review_final_resume.log)。

## 同一 dispatch 内的 wave 到达差异（已有 ATT）

只重读已有 mg8 ATT，不新增 trace。分别在 SIMD0、SIMD1 的独立 capture 内，将同一 dispatch 的 slot0/slot1 按累计 WMMA index 对齐。两份 capture 的绝对 clock 不可互相比对。

| sampled pair | steady rendezvous 数 | slot1−slot0 signal 中值（cycles） | release 差中值（cycles） | slot0 / slot1 barrier stall 总 cycles |
|---|---:|---:|---:|---:|
| SIMD0 | 93 | 118 | 1 | 19127 / 8081 |
| SIMD1 | 93 | 133 | 1 | 19625 / 9293 |

slot1 的 sampled DS_load/WMMA stall 更多，WMMA 平均 issue gap 约 21.1–21.6 cycles，slot0 约 17.1–17.2 cycles。signal 到达差与 barrier stall 差相符，支持研究“先到 wave 等较慢 peer”的推理；两条 sampled wave 不能识别全八条 wave 的最后到达者，也不能将这些 cycles 换成可节省的 kernel μs。脚本：[analyze_att_slot_pairs.py](../.codex_tmp/gemm1_t1536_20261017/analyze_att_slot_pairs.py)，结果：[slot pair comparison](../.codex_tmp/gemm1_t1536_20261017/logs/existing_att_slot_pair_comparison.json)。

## bounded split READY/REUSE 调度：静态筛选与待验收候选

仅在 `M256_FUSED_LAYOUT`、KWS=2、已有 `next_stage_wait` 的路径调整当前 WG rendezvous 的时机。first k128 完成当前 stage 的 second-k128 LDS operands 和原 tensorcnt outstanding-distance wait 后 signal；signal 到 wait 之间只放依赖已完成 operands 的 WMMA。second k128 在 reused-slot TDM issue 和 next-stage carry reads 前 wait。保持原 WG member count、barrier 代次、所有 TDM、WMMA、K 累加次序、epilogue 和 N-task output/input fences，不采用 named barrier。

`splitready_scoped` 先完成 DS reads 再 signal，增加 register lifetime；`splitready_front` 让 FRONT WMMA 先覆盖 LDS loads，再完成 DS/tensor wait 与 signal，用 BACK WMMA 覆盖 rendezvous。FRONT2/FRONT3 改变 FRONT 的 partition，保留同一组完整 WMMA。下表为目标 `tdm_b_th=6` 的最终 ISA/ELF，数量包含互斥代码路径，不能当作动态执行量或实测收益。

| 候选 | VGPR / SGPR | private（B） / spills | 静态指令数 | `s_wait_dscnt` | 七处 compute signal→wait 的 WMMA |
|---|---|---|---:|---:|---|
| mg8 review | 470 / 106 | 0 / 0 | 4381 | 原版 | 0 |
| `splitready_scoped` | 489 / 106 | 0 / 0 | 4338 | 41 | 16–17 |
| `splitready_front`（FRONT4） | 462 / 106 | 0 / 0 | 4413 | 67 | 10–11 |
| `splitready_front2` | 482 / 106 | 12 / 4 VGPR spills | 4385 | — | 13–14 |
| `splitready_front3` | 470 / 106 | 0 / 0 | 4381 | 60 | 11–12 |
| `splitready_front_lds` | 462 / 106 | 0 / 0 | 4413 | 67 | 10–11 |
| `splitready_front3_lds` | 470 / 106 | 0 / 0 | 4381 | 60 | 11–12 |

全部 LDS 为 278528 B；FRONT2 因 spill 排除。FRONT4 的 encoded VGPR allocation 从 480 降为 464，但不能仅由此宣称 occupancy 或性能改善。两个 `_lds` 组合准确应用已编译的 LDS-only epilogue/final store wait 改动，保留 kernel 结束前 `s_wait_tensorcnt 0` 与 `s_wait_storecnt 0`。

source SHA256：

```text
splitready_scoped     ac8beb022164dcce1f7b452b6c139bb777d578319208f0c8b01a16b308bef67c
splitready_front      a757c082d46a78b75a0672d7610249bee5e3f271e785fa51d53666fa2761ce8f
splitready_front3     6e98fa641455a3fefd02913d2ab3ee36fb01a9de109a3fa0bea15363c2adab6a
splitready_front_lds  922e14a5b37db3bfb2484e6118bf7ac735d471ceb975ab3f73f9489d168cf0e2
splitready_front3_lds 6b2562f7db932bf70b661264514dd303e7dfac2392866ba1e3d3b0ddec1d1006
```

scoped、FRONT4、FRONT3 及两个 `_lds` 版本，十组 fallback 完整 ELF 均与 mg8 review 相同；target 只有已逐字段解释的 instruction prefetch/VGPR allocation 差异，ABI/layout/control contract 保持。FRONT3 的 target descriptor 和 metadata 完全相同。最终 ISA 的全部 signal→wait 区间没有 memory 或 branch，七处 compute 区间实际含上表 WMMA；没有将源码排列直接当作最终 scheduler 结果。

- [scoped scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_splitready_scoped.json)、[FRONT4 scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_splitready_front.json)、[FRONT3 scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_splitready_front3.json)。
- [FRONT4 LDS scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_splitready_front_lds.json)、[FRONT3 LDS scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_splitready_front3_lds.json)。
- [ISA placement script](../.codex_tmp/gemm1_t1536_20261017/audit_splitready_isa.py)、[FRONT4 LDS placement](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_compile_cyclic64_mg8_splitready_front_lds_m256_q8_b0_cm1_pd0_dr0_sq1_e64_th6.splitready_placement.json)、[FRONT3 LDS placement](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_compile_cyclic64_mg8_splitready_front3_lds_m256_q8_b0_cm1_pd0_dr0_sq1_e64_th6.splitready_placement.json)。

### source contract、ring 与乱序事件检查

[AST contract](../.codex_tmp/gemm1_t1536_20261017/logs/splitready_source_contract.json) 限定到 fused GEMM1 launcher 内比较同名 nested functions，确认纯 scoped/FRONT4/FRONT3 的 `load_lds_data`、A/B/ScaleA/ScaleB readers、`mma_rows`、TDM issue/prefetch、地址公式、quant output 等函数完全相同，run_task 剔除唯一调度改动后 AST 相同。不能用全 module 的 ast.walk 让 GEMM2 同名函数覆盖 GEMM1。

[ring/counter 代数](../.codex_tmp/gemm1_t1536_20261017/logs/splitready_ring_algebra.json) 共 207 checks，mismatch=0：kt+4 准确复用 kt 范围而不覆盖 kt+1..3；每 owner-wave 的 ordered TDM 到 kt+3 后 tensorcnt≤2 保证到 kt+1 完成。实际 M256 quant payload 为 16 KiB，范围 `[245760,262144)`，scale 直接写 global；payload 与下一 task I0/I1 不重叠，但与 K2 的 A1 重叠，所以原 prologue 在 refill K2/K3 前必须 drain previous output，不能删除。

[乱序事件模型](../.codex_tmp/gemm1_t1536_20261017/logs/splitready_interleavings.json) 对八条 wave、28 个 K256 tiles、四 buffer、per-owner ordered TDM、随机 DS completion 及 `barrierComplete` 运行 1024 trials，每轮 1336 events，全部通过。负对照提前 REUSE signal 在 1023/1024 轮检出 `WRITE before REUSE`；删除 READY wait 在 6/1024 轮检出 `READ before READY`。模型仅覆盖 steady/drain，不执行 emitted ISA，也不证明 cluster multicast 或硬件时序，GPU 完整 bitwise/replay 验证仍是必需门槛。

硬件依据：`MI400_Shader_Programming#65.txt` §4.3.6/4.3.6.1 记载 WG `barrierComplete` bit 在已完成时让 WAIT 立即通过并清零，§4.3.7.2.4 记载 DS completion；CDNA5 ISA §10.11.1（printed p.140）明确同 wave Tensor load/store ordered、不同 wave unordered，且不与其它 memory 类型自动排序。这些事实支持本候选必须保留 DS completion、tensorcnt 与跨 wave rendezvous；不能由资料推导候选一定更快。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v32.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v33.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v34.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v35.py
# In dsr1_ep4; compile-only, no GPU launch.
python3 -u .codex_tmp/gemm1_t1536_20261017/compile_candidate_cpu.py \
  cyclic64_mg8_splitready_front --tdm-b-th 6
# On host; fallback compilation and audits execute in dsr1_ep4.
bash .codex_tmp/gemm1_t1536_20261017/compile_bounded_candidate_scope_host.sh \
  cyclic64_mg8_splitready_front
```

FRONT4/FRONT3 后续已通过独立 GPU 完整 equivalence/replay，但原命令筛选没有性能收益，未替换正式源码。完整结果见后文“清理后的候选筛选与交替排除结果”。本节的静态数量仍只表示编译产物。

## target descriptor 差异的字段核对

最初通用范围脚本把 target descriptor 完全一致作为必需条件，因此 drain 与 b128 的 scope 最初返回 2，尽管十组 fallback 完整 ELF 全相同。已逐字段检查 64-byte AMDHSA descriptor，差异只在 `COMPUTE_PGM_RSRC3.INST_PREF_SIZE[11:4]`。它随函数代码长度改变；不能使用包含 shader padding/alignment 的整段 `.text` 字节数代替函数长度。

| target | ELF function bytes | INST_PREF_SIZE | metadata resource 差异 |
|---|---:|---:|---|
| mg8 整理版 | 28372 | 222 | 基准 |
| 普通 drain carry | 28504 | 223 | VGPR 470 → 474 |
| b128 shift/mask | 29228 | 229 | 无 |
| b128-perm LDS-only | 29000 | 227 | 无 |

`descriptor_scope.py` 显式允许上述 prefetch 字段随 ELF `STT_FUNC` size 按 128-byte 向上取整改变。新增 FRONT4/scoped 资源筛选还允许 `COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT[5:0]` 改变：必须是 wave32，编码须准确等于 metadata `ceil(vgpr_count/16)-1`，其余 bit 不得改变。其余 descriptor/reserved bytes 必须一致，特别保留 NAMED_BAR_CNT 和所有 control bits。metadata 比较保留全部 ABI、argument offsets、wave/workgroup/cluster geometry、LDS/private 与 spill 字段；仅 VGPR/SGPR count 变化单独记录，任何 spill 拒绝。十组 fallback 仍要求完整 ELF 相同。重新读取原编译产物核对 drain、b128、b128-perm LDS-only 与通过资源门槛的 splitready 候选均返回 0，没有通过忽略 `.rodata` 或 `.note` 规避差异；首次不认识 VGPR allocation 差异的失败报告仍保留。

证据：[raw descriptor delta](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_descriptor_delta.json)、[b128 audited scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_payload_store128.json)、[drain audited scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_normal_drain_carry.json)、[field audit implementation](../.codex_tmp/gemm1_t1536_20261017/descriptor_scope.py)。LLVM 依据：`llvm/include/llvm/Support/AMDHSAKernelDescriptor.h` 的字段位宽/偏移、`AMDGPUBaseInfo.cpp::getVGPREncodingGranule/getEncodedNumVGPRBlocks` 与 `AMDGPUAsmPrinter.cpp`。硬件资料 `MI400_Shader_Programming#65.txt` §5.5（printed pp.225–226）说明 initial instruction prefetch 的 128-byte 单位；CDNA5 ISA §3.3.2.1（printed p.16）记载 wave32 VGPR 的 16-register allocation granularity。此核对是兼容性证据，不能代替 GPU 正确性或性能验证。

## 后续验证入口与验收顺序

新增 [validate_imported_candidate.py](../.codex_tmp/gemm1_t1536_20261017/validate_imported_candidate.py)：私有导入候选，只在进程内替换 GEMM1 callable；真实 producer、GEMM2、canonical route 比较和 replay 检查沿用原验证脚本，共享 live module 文件只读。它不能用于改变正式性能计时入口。候选串行验证，原死锁现场已清理，没有并发堆叠 GPU 验证进程。

mg8 整理版完整 equivalence/replay 已完成。[screen_validated_candidates_host.sh](../.codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh) 已串行验证 LDS-scope、payload b128-perm、splitready FRONT4/FRONT3，并继续用于 ScaleA cache；每个候选先完成完整 equivalence/replay，才在全 GPU 空闲时用本文开头的原 benchmark 筛选，通过者再做至少三轮交替比较，第二轮反向。测试均有上述 120 秒终止限制。没有通过门禁的实验不替换正式源码。

```bash
# In dsr1_ep4; untimed production correctness checks.
python3 -u .codex_tmp/gemm1_t1536_20261017/validate_imported_candidate.py \
  cyclic64_mg8_delivery_review --check equivalence
python3 -u .codex_tmp/gemm1_t1536_20261017/validate_imported_candidate.py \
  cyclic64_mg8_delivery_review --check replay
python3 -u .codex_tmp/gemm1_t1536_20261017/validate_imported_candidate.py \
  cyclic64_mg8_lds_epilogue_finalwait --check equivalence
```

两个阻塞进程已退出，当前正式中值为 **94.888 μs**，80 μs 目标未达成。新增候选的结果将依据原始日志追加，单次筛选不替代正式交替中值。

## 清理后的候选筛选与交替排除结果

四个候选均通过与冻结 baseline 的完整 equivalence/replay：11 cases 的 canonical payload、ScaleA、GEMM2 valid rows 和 MoE 输出均 bitwise mismatch=0；三种动态输入、poison 和 eager/graph replay 检查也全部通过。正式源码仍为 mg8 review，私有正确性 import 不修改 shared live module。

| 候选 | 原命令单次 GEMM1（μs） | 后续结论 |
|---|---:|---|
| `cyclic64_mg8_lds_epilogue_finalwait` | 93.775 | 三轮没有稳定收益 |
| `cyclic64_mg8_payload_store128_perm` | 94.779 | 三轮中值回退 |
| `cyclic64_mg8_splitready_front` | 95.799 | 未观察到筛选收益，不合入 |
| `cyclic64_mg8_splitready_front3` | 97.590 | 未观察到筛选收益，不合入 |

LDS-only 与 b128-perm 在同批全 GPU 空闲下，与 review 串行比较三轮；第二轮先测候选再测 review：

| 轮次 | mg8 review（μs） | LDS-only（μs） | b128-perm（μs） |
|---:|---:|---:|---:|
| 1 | 94.752 | 95.080 | 94.642 |
| 2 | 94.608 | 94.210 | 95.038 |
| 3 | 93.992 | 94.572 | 95.650 |
| **中值** | **94.608** | **94.572** | **95.038** |

LDS-only 中值仅差 0.036 μs（约 0.038%），三轮方向不一致，不能宣称稳定提升。b128-perm 中值增加 0.430 μs（约 0.45%），减少 store 指令不代表实际更快。splitready 在 signal→wait 间确有 10–12 条 WMMA，且资源/bitwise 正确，但筛选耗时没有降低；因此不因静态 overlap 就合入。

全部单项精度与性能日志已下载到本地。证据：[screen controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_screen_20261017_controller.log)、[三轮 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_epilogue_pair_20261017_controller.log)、[LDS-only equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_screen_20261017_cyclic64_mg8_lds_epilogue_finalwait_equivalence.log)、[LDS-only replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_screen_20261017_cyclic64_mg8_lds_epilogue_finalwait_replay.log)。controller 保留每个样本、source hash 和空闲检查。间歇出现的 E65 其他任务先每 6 秒等待其结束，未终止或与其竞争性能。

```bash
# On the host; validation and timing execute inside dsr1_ep4.
SCREEN_BATCH=<new_screen_batch> bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg8_lds_epilogue_finalwait cyclic64_mg8_payload_store128_perm \
  cyclic64_mg8_splitready_front cyclic64_mg8_splitready_front3
PAIR_BATCH=<new_pair_batch> VALIDATION_BATCH=<new_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/pair_verified_candidates_host.sh \
  cyclic64_mg8_lds_epilogue_finalwait cyclic64_mg8_payload_store128_perm
```

`<new_..._batch>` 须替换为未使用过、仅含字母/数字/下划线/短横线的名称；脚本拒绝覆盖已有验收日志。

## 完整 ScaleA 跨 N-task cache：实验 bug 与修正版验收

仅在原 `M256_FUSED_LAYOUT` 范围，将原 mutable ScaleA/ScaleB ring 改为 8 KiB ScaleB ring，后放 56 KiB canonical M32 全 K ScaleA cache。首个 N-task 加载一次 exact ScaleA bytes，四个 N-task 的 readers 使用相同 cache；A/B payload 和 ScaleB 继续正常 prefetch，完整 K128 WMMA、每 accumulator 的 K 顺序、activation/quant 及外部格式均不变。该方案只复用输入，未减少矩阵计算。

首版 v36 的完整 equivalence 在 T1536 random seed0 被 bitwise 门禁拒绝：payload mismatch=31829、ScaleA mismatch=0、MoE mismatch=10417165、GEMM2 mismatch=43021832。虽然原 logits gate 仍通过（logits_diff≈1.32584e-03 < 0.01），本任务还要求与原 GEMM1 bitwise 一致，故没有测性能或替换正式源码。失败源码及日志原样保留。

根因是生成器删除了正常 `mutable_jobs` / `next_jobs` 中的 ScaleA job，却保留了 `cached_a_jobs = jobs`。下一 N-task 的 stage4 refill 因而仍把旧 cyclic ScaleA 写到新的 ScaleB ring。v37 以新名字 `cyclic64_mg8_cache_full_sa_v2` 将这份 refill 列表改为 `mutable_jobs`；该实验 bug 不在当前正式 review 源码中。

修正版 source SHA256：`5b132ca58632f87cdda89d447f150ded9cf8f8eb46f8c18a62f2f4960a3168e1`。目标 CPU 编译资源 VGPR=486、SGPR=98、LDS=327680 B、private=0、spills=0；静态指令 4203、WMMA 288、`ds_load_b128` 456、TDM load 31。LDS 从 272 KiB 增至 320 KiB，不能只依据 TDM 变少判断收益。硬件资料 `MI400_Shader_Programming#65.txt` §1.1 / §4.7 记载 WGP$/LDS 的划分和每 WG 最大 320 KiB；从该上限无法推出实际性能提升。

修正版完整 equivalence 的 11 cases 和动态输入/replay 检查全部 mismatch=0。2026-10-17 01:02:46 UTC 性能前全机空闲，原命令单次 GEMM1=93.798 μs，count=19，logits_diff=3.4733e-06。此处仅为筛选结果，尚不能替代正式三轮成绩。性能后无 KFD 残留，sysfs 短暂 busy 尾值已后续等待归零。正式 review 已恢复。

证据：[首版失败](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_20261017_cyclic64_mg8_cache_full_sa_equivalence.log)、[v37 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v37.py)、[修正版 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cache_full_sa_cyclic_v2_inspection.log)、[完整 equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_20261017_cyclic64_mg8_cache_full_sa_v2_equivalence.log)、[完整 replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_20261017_cyclic64_mg8_cache_full_sa_v2_replay.log)、[原 benchmark](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_20261017_cyclic64_mg8_cache_full_sa_v2_bench.log)。

该候选 LDS 变化单独由 [audit_scale_cache_scope.py](../.codex_tmp/gemm1_t1536_20261017/audit_scale_cache_scope.py) 核对，要求 descriptor 和 metadata 都精确为 278528→327680 B 后，才比较其余 ABI/control 字段；没有扩大通用 descriptor audit 的允许范围。十组 fallback 的完整 ELF 全部相同，目标其余 ABI/control 字段审核通过：[scope 结果](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_scale_cache_scope_cyclic64_mg8_cache_full_sa_v2.json)。

### ScaleA full-cache 修正版三轮交替结果

原 benchmark、random non-balanced、iters20、逐次全机空闲门禁；第 2 轮逆序。

| 轮次 | mg8 整理版（μs） | ScaleA full-cache v2（μs） |
|---:|---:|---:|
| 1 | 94.391 | 93.668 |
| 2，逆序 | 94.275 | 95.264 |
| 3 | 94.815 | 94.205 |
| 中值 | **94.391** | **94.205** |

中值仅减少 0.186 μs（0.20%），第二轮回退 0.989 μs，未确认稳定收益，不采用。增加 48 KiB LDS 并没有带来明确的原口径提升。GPU 空闲等待中出现其他 E65/shared-expert 任务时，脚本按 6 秒重查等待其结束，未终止或与之并行计时。

```bash
PAIR_BATCH=postcleanup_sa_cache_v2_pair_20261017 \
VALIDATION_BATCH=postcleanup_sa_cache_v2_20261017 \
bash .codex_tmp/gemm1_t1536_20261017/pair_verified_candidates_host.sh \
  cyclic64_mg8_cache_full_sa_v2
```

原始记录：[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_pair_20261017_controller.log)，整理版 [第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_pair_20261017_cyclic64_mg8_delivery_review_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_pair_20261017_cyclic64_mg8_delivery_review_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_pair_20261017_cyclic64_mg8_delivery_review_3.log)，候选 [第 1 轮](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_pair_20261017_cyclic64_mg8_cache_full_sa_v2_1.log)、[第 2 轮](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_pair_20261017_cyclic64_mg8_cache_full_sa_v2_2.log)、[第 3 轮](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_sa_cache_v2_pair_20261017_cyclic64_mg8_cache_full_sa_v2_3.log)。

## FRONT A seed：完整精度通过，首轮未取得收益

候选 `cyclic64_mg8_front_a_seed` 在原 M256 fused 路径将两个 ping-pong seed 中的 A 缩为前 4 个 M16 fragments；后 4 个 A fragments 在当前 K128 内加载并消费，B 和 ScaleA/ScaleB 仍保留在 seed。FRONT WMMA 覆盖 BACK 读取，BACK WMMA 覆盖下一 seed 读取。原 ring 地址、TDM、所有 reader/task fences、每个 accumulator 的 K 顺序、完整 epilogue 和外部布局保持不变。其他路径继续采用原逻辑，没有对 token 数或输入值添加特例。

源码 SHA256：`b6d400b7847f2086a3f08c815fddbcdf09168d7e694a1cbf56e400c8a7131a70`。目标 CPU 编译资源为 VGPR 373、SGPR 106、LDS 278528 B、private 0、spills 0，VGPR 低于正式版的 470。ISA 静态 `s_wait_dscnt` 为 134，出现更多细粒度 operand readiness 等待；静态计数包括互斥路径，不能按 VGPR 降幅或指令数推算运行时间。

完整 11 cases 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output 全部 mismatch=0；动态输入、输出 poison、eager 和 graph replay 全部 mismatch=0。十组 fallback 完整 ELF 与整理版一致，target ABI/control 审核通过。[AST 审查](../.codex_tmp/gemm1_t1536_20261017/audit_front_a_seed_source.py) 还核对 `mma_rows`、reader、TDM/quant 函数、剔除 operand schedule 后的 `run_task` 以及 generic compute AST 均相同；每 wave/task 为 896 WMMA，四 task 为 3584。

2026-10-17 01:29:06 UTC 全机空闲后运行原 benchmark，得到 **100.035 μs**（count19，logits_diff `3.4733e-06`）。这是筛选单次，不是正式中值，没有取得收益，不采用。controller 已恢复整理版源码。性能后未看到残留 KFD 进程；瞬时 busy=3% 保留在日志中，后续空闲检查确认归零。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v38.py
python3 .codex_tmp/gemm1_t1536_20261017/audit_front_a_seed_source.py
# On host; numerical checks use process-local imports, then original benchmark.
SCREEN_BATCH=postcleanup_front_a_20261017 \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg8_front_a_seed
```

原始记录：[候选](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_mg8_front_a_seed.py)、[ISA/resources](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_front_a_seed_inspection.log)、[AST contract](../.codex_tmp/gemm1_t1536_20261017/logs/front_a_seed_source_contract.json)、[fallback/ABI scope](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_front_a_seed.json)、[完整 equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_front_a_20261017_cyclic64_mg8_front_a_seed_equivalence.log)、[完整 replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_front_a_20261017_cyclic64_mg8_front_a_seed_replay.log)、[原命令性能](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_front_a_20261017_cyclic64_mg8_front_a_seed_bench.log)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_front_a_20261017_controller.log)。

硬件资料中的 VGPR allocation 粒度为 wave32 每 16 个（CDNA5 ISA §3.3.2.1）；LDS 上限为每 WGP 320 KiB（CDNA5 白皮书 LDS 一节）。本候选虽将 VGPR allocation 从 480 降为 384，但 LDS 仍为 278528 B，因此不能据此声称增加了常驻 WG 数。首轮性能也没有支持寄存器减少即加速的判断。

## 按 M/N 数据依赖划分 named DATA：精度通过，首轮未取得收益

候选 `cyclic64_mg8_named_data_scoped` 从正式版生成，仅修改已有 M256 fused 分支。WG FREE 仍覆盖所有当前 stage 的 DS readers，原 TDM refill 仍等待 FREE；next-stage READY 改由四个 named DATA barriers 表示对应 M half / N128 plane 的 A、B、ScaleA、ScaleB producer completion。每个 DATA barrier 的 member count 为 4，reader 在初始化后 JOIN 对应 barrier，后续仅等待已经 JOIN 的 DATA barrier。producer 分支使用已选定的 `my_jobs[0].waves` 与 wave parity，避免早期 `or` 分支丢失的问题。

source SHA256：`982384e84dd7752dd04ce471f110f67f61a101f8f8e4578c2c31459637db211d`。生成器核对原 MMA、K 顺序、LDS reader/writer、TDM issue/prefetch、output/quant 函数 AST 不变，并枚举四组 DATA 各收到准确的四个 producer signals：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/named_data_scoped_source_contract.json)。硬件依据为 Shader Programming Guide §4.3.6.5.5：wave 仅接收其 JOIN 的 named barrier 通知，但允许 signal 任意 barrier；仍须实测验证跨代复用。

完整 11 cases 的 payload、ScaleA、GEMM2 valid rows、MoE output 全部 bitwise mismatch=0；完整动态输入/poison/eager/graph replay 全部 mismatch=0。原命令的空闲单次为 **98.831 μs**（count19，logits_diff `3.4733e-06`），未取得收益，不采用。源码已恢复正式版。

目标 CPU ELF 为 VGPR 486、SGPR 102、LDS 278528 B、private 0、VGPR spills 0、SGPR spills 8；ISA 存在八组 `v_writelane_b32` / `v_readlane_b32`。这是该候选新增的资源开销，不能推广为正式 kernel 的主要瓶颈，也未单独证明它解释了全部回退。此同步实验改变 named barrier allocation，因此没有将它描述为通过原严格 descriptor contract。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v39.py
# On host, inside the fixed task tmux session.
SCREEN_BATCH=postcleanup_named_full_20261017 \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg8_named_data_scoped
```

记录：[候选](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_mg8_named_data_scoped.py)、[ISA/resources](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_named_scoped_inspection.log)、[equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_named_full_20261017_cyclic64_mg8_named_data_scoped_equivalence.log)、[replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_named_full_20261017_cyclic64_mg8_named_data_scoped_replay.log)、[原命令](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_named_full_20261017_cyclic64_mg8_named_data_scoped_bench.log)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_named_full_20261017_controller.log)。

## 完整 N workload 拆成 1/2 个 N-task：三轮无稳定收益

仅在已有 `M256_FUSED_LAYOUT` 分支内，将每个 WG 的四个 N256 task 改为一个或两个；增加对应 WG 数，仍对完整 4096 列计算十六个 N256 tile，每个 M/N pair 覆盖一次。1-task 使用原 generic cluster swizzle；2-task 用两个 task groups，`persistent_m_tile`、`n_unit` 与 next-task B/ScaleB offsets 一起调整。原 K128 WMMA、SiLU、BF16 rounding、E8M0/native FP4 conversion、routing/tail/empty expert 逻辑及所有 input/output fences 保留。没有缩小矩阵或减少实际计算。

v40 首版把动态 `N == 4096` 用来选择 Python 编译期 `PERSISTENT_TASKS`，两候选都在 tracing 被拒绝：`RuntimeError: cannot evaluate dynamic 'Boolean' as Python bool during tracing`。原 `N` 类型为 `fx.Int32`，不能为绕过报错改动 launcher ABI。v41 改用原有编译期 `M256_FUSED_LAYOUT`；实际 fused persistent wrapper 的该配置使用 N4096。失败源码和日志保留，这不是正式源码中的 bug。

修正版源码 SHA256：

```text
cyclic64_mg8_ntasks1_v2  6e7b5164433ca8508dddbaa5d022e1cf3200ece914d9cd4c7577c540687bc867
cyclic64_mg8_ntasks2_v2  8eb37dc13221d9c12e2f2274307aa88d8c4ac218af02d79be162b079820d6b63
```

两候选的完整 11 cases equivalence 全部 canonical payload、ScaleA、GEMM2 valid rows、MoE output mismatch=0；完整动态输入/poison/eager/graph replay 也全部 mismatch=0。原命令空闲筛选单次分别为 **98.032 μs** 和 **93.944 μs**，后者进入三轮交替确认，第 2 轮逆序：

| 轮次 | mg8 review（μs） | 2-task v2（μs） |
|---:|---:|---:|
| 1 | 95.789 | 94.699 |
| 2，逆序 | 95.555 | 95.915 |
| 3 | 95.561 | 95.496 |
| **中值** | **95.561** | **95.496** |

中值仅差 **0.065 μs（0.068%）**，第二轮回退 0.360 μs，没有确认稳定收益，不采用。第一轮单次 93.944 μs 不替代三轮中值，也不与旧批正式中值相减当作收益。controller 于 2026-10-17 02:05:03 UTC 完成并恢复正式 review。遇到其他 E65 工作负载或 CPU compile-only 进程登记 KFD 时，脚本按六秒等待，未并行计时或清理其进程。

2-task CPU target 资源：VGPR=480、SGPR=106、LDS=278528 B、private=0、VGPR spills=0、**SGPR spills=4**。十组 fallback 完整 ELF 相同；target descriptor 仅 instruction prefetch length 222→223，逐字段核对通过，但 metadata 的新增 spills 导致严格 scope gate 返回 2。没有把该候选描述为通过完整资源验收；spills 的出现也不能独自解释整轮性能。1-task 因首轮无收益未继续完整资源核对。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v40.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v41.py
# On host; use unused batch labels for a fresh reproduction.
SCREEN_BATCH=<new_ntasks_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg8_ntasks1_v2 cyclic64_mg8_ntasks2_v2
PAIR_BATCH=<new_ntasks_pair_batch> VALIDATION_BATCH=<new_ntasks_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/pair_verified_candidates_host.sh \
  cyclic64_mg8_ntasks2_v2
```

源码：[v40 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v40.py)、[v41 修正](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v41.py)、[1-task](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_mg8_ntasks1_v2.py)、[2-task](../.codex_tmp/gemm1_t1536_20261017/candidates/cyclic64_mg8_ntasks2_v2.py)。原始证据：[首版失败](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks_20261017_controller.log)、[修正版完整筛选](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks_v2_20261017_controller.log)、[2-task equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks_v2_20261017_cyclic64_mg8_ntasks2_v2_equivalence.log)、[2-task replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks_v2_20261017_cyclic64_mg8_ntasks2_v2_replay.log)、[三轮 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_ntasks2_v2_pair_20261017_controller.log)、[ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_ntasks2_inspection.log)、[scope 与 spills 门禁](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_ntasks2_v2.json)。

## WMMA queue / multicast timeout：独立 issue-policy 实验

[v42 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v42.py) 从正式 review 分别生成两个独立候选，均只修改原 `M256_FUSED_LAYOUT`。`queue_scoped` 允许八-wave GEMM1 写 `WAVE_SCHED_MODE[2]=1`，保留 DEP_MODE 和编译器 dependency checks；`mcast_normal_scoped` 仅将带 workgroup mask 的 TDM descriptor `early_timeout` 改为 0。没有合并两项设置，也没有删减 request、改动地址或输出。

硬件依据：Shader Programming Guide §3.4.9 / §4.3.7.4.2 说明 bit2 允许一条 wave 连续 queue 数条 WMMA，但可能阻碍其它 wave 的 co-execution，因此八-wave 配置下收益只是待验证假设；该设置不等于 DEP_MODE=1/2。§4.9.8 / TDM descriptor bit21 说明 normal timeout 等待请求合并或标准 timeout，early timeout 在 GL2 数据返回时立即向已有请求者返回；更大的合并机会也可能增加等待，不能由资料推断加速。

[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/issue_policy_source_contract.json) 逐个恢复唯一 policy expression 后，整个 module AST 与 review 完全相同。它限定改动范围，不能代替 GPU correctness。两个 source SHA256：

```text
cyclic64_mg8_queue_scoped         6bbfdc6fe7448321baddef7b9f44c09e2d295e5e6743b80afaabf3f69fba943f
cyclic64_mg8_mcast_normal_scoped  443395afd0fcb7fa7ac76840dd3ca71ca43794c4bd79580541d44fc624038d66
```

两候选完整 11 cases equivalence 和动态输入/poison/eager/graph replay 全部 bitwise mismatch=0。原命令空闲筛选分别为 94.690 μs、94.077 μs；随后做三轮交替，第 2 轮候选在前：

| 轮次 | mg8 review（μs） | queue scoped（μs） | normal-timeout scoped（μs） |
|---:|---:|---:|---:|
| 1 | 96.166 | 95.070 | 97.669 |
| 2，逆序 | 93.990 | 94.787 | 95.226 |
| 3 | 94.802 | 94.344 | 97.977 |
| **中值** | **94.802** | **94.787** | **97.669** |

queue 中值仅差 0.015 μs（0.016%），第二轮回退 0.797 μs；normal timeout 中值回退 2.867 μs（3.02%），三轮均慢。两者都未确认稳定收益，不采用，controller 于 2026-10-17 02:38:53 UTC 完成并恢复正式 review。该 batch 尚未做完整 CPU fallback 资源审核，不能据源表达式改动小就写成全部通过。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v42.py
# On host; numerical checks and original benchmark execute inside dsr1_ep4.
SCREEN_BATCH=<new_issue_policy_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg8_queue_scoped cyclic64_mg8_mcast_normal_scoped
PAIR_BATCH=<new_issue_policy_pair_batch> VALIDATION_BATCH=<new_issue_policy_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/pair_verified_candidates_host.sh \
  cyclic64_mg8_queue_scoped cyclic64_mg8_mcast_normal_scoped
```

证据：[完整筛选 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_issue_policy_20261017_controller.log)、[三轮 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_issue_policy_pair_20261017_controller.log)。每次实际测试仍有 120 秒终止保护。不同批次 review 中值存在波动，因此不能用候选筛选单次与较早批次比较，冒充已确认收益。

## 第一个 K256 A 的 launch-local cache：精度通过，筛选没有收益

[v43 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v43.py) 在原 `M256_FUSED_LAYOUT` 的四个 N-task 之间缓存同一 M tile 的第一个 K256 A。缓存区为 `[278528,311296)`，与原输入 ring、scale arena、quant output 全部 disjoint。每次 launch 的 task0 由原 A-owner waves 加载一次完整 32 KiB A；后续 task 的 K0 A 使用该只读 LDS 区，其余 K tiles、B/ScaleA/ScaleB、数学和输出仍执行原逻辑。没有跨 launch/global cache。

源码 SHA256 `4d1542c0e88936b9ed036e6b966bcb843b852683fdcf70946d391fbfd40d42f9`。首 task 的 A-owner stream 是 cacheK0、K1、K2、K3，恰好与其他 owner 一样有四个 requests；`tensor_wait(3)` 加原 WG rendezvous 保证 K0 cache 已完成再读取。后续 task 的 A-owner 仅发 nextK1，原 `tensor_wait(1)`/WG rendezvous 保留；首次读取的 K0 cache 已由 task0 fence 完成。动态输入/replay 必须验证每个新 launch 重新加载缓存。

CPU target 编译得到 VGPR=470、SGPR=106、LDS=311296 B、private=0、VGPR/SGPR spills=0。新增 LDS 32 KiB 不增加已有单-WG常驻数，也不能据此推断加速。[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/first_a_cache_source_contract.json) 确认 MMA、reader、K-loop、quant/TDM emit 等函数 AST 相同，只有输入缓存路由和其分配改变。

完整 11 cases equivalence 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output 全部 bitwise mismatch=0；动态输入、输出 poison、eager/graph replay 也全部 mismatch=0。2026-10-17 02:45:51 UTC 全机空闲后运行原 benchmark，单次 GEMM1 为 **96.318 μs**，count=19，logits_diff=`3.4733e-06`。没有取得筛选收益，不采用，也未继续扩大 fallback/显式 LDS descriptor scope 验收。controller 于 02:46:13 UTC 完成并恢复正式 review；该单次结果不能与旧批中值相减声称改善或回退幅度。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v43.py
# On host; choose a fresh batch label when reproducing.
SCREEN_BATCH=<new_first_a_cache_batch> \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg8_first_a_cache
```

证据：[CPU ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_first_a_inspection.log)、[完整 equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_first_a_cache_20261017_cyclic64_mg8_first_a_cache_equivalence.log)、[完整 replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_first_a_cache_20261017_cyclic64_mg8_first_a_cache_replay.log)、[原命令筛选](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_first_a_cache_20261017_cyclic64_mg8_first_a_cache_bench.log)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_first_a_cache_20261017_controller.log)。

## 独立 output LDS 与提前 stage2/3 refill：三轮未确认稳定收益

[v44 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v44.py) 在原 `M256_FUSED_LAYOUT` 中将 output 放到 `[278528,295936)`，与四个 input data segments `[0,262144)` 和 scale ring `[262144,278528)` 完全分离。`output_arena` 仅改分配；`output_arena_early` 另外将下一 N-task 的 I2/I3 提前到旧 output O 完成前。每条 owner wave 的顺序为 `I0,I1,O,I2,I3`：`tensor_wait(3)` 与 WG rendezvous 保证 I0/I1 可读，计算 K0 后 `pipeline_fence(2)` 保证 O 已完成且 K0 reader 释放，再发 I4。后续原 K-loop completion fences 保证 I2/I3 readiness，output 区域在重用前仍已完成旧 O。

完整 input/WMMA workload、K 累加顺序、activation/BF16/E8M0/native FP4、canonical payload/ScaleA 和 routing/tail 逻辑保持相同。[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/output_arena_source_contract.json) 核对 protected 函数 AST 不变；还原 allocation/prologue 两项编辑后，整个 module AST 与正式 review 一致。硬件依据为 Shader Programming Guide §4.10.1（同 wave TDM ordered、不同 wave unordered，因此仍需 completion/rendezvous）及 §4.7.1（每 WGP 最多 320 KiB LDS）。

| 候选 | source SHA256 | VGPR / SGPR | LDS（B） | private / spills |
|---|---|---|---:|---|
| `cyclic64_mg8_output_arena` | `499fa180dd774cf486c2c888c8b3e95aa24760cfc072a221e3a26466c6981279` | 470 / 106 | 295936 | 0 / 0 |
| `cyclic64_mg8_output_arena_early` | `5711c899a8af82ecd978ebc94dd48b884db3203f296f1997e3557b80cb05320e` | 467 / 106 | 295936 | 0 / 0 |

两版本完整 11 cases equivalence、动态输入/poison/eager/graph replay 全部 bitwise mismatch=0。原命令空闲筛选单次分别为 **94.414 μs**、**93.708 μs**，随后完成三轮交替，第 2 轮候选在前：

| 轮次 | mg8 review（μs） | 只移动 output arena（μs） | 独立 arena + early refill（μs） |
|---:|---:|---:|---:|
| 1 | 95.093 | 95.382 | 94.737 |
| 2，逆序 | 94.300 | 94.789 | 94.216 |
| 3 | 94.747 | 96.335 | 94.752 |
| **中值** | **94.747** | **95.382** | **94.737** |

只移动 arena 三轮均慢，中值增加 0.635 μs（0.67%）；early refill 中值仅减少 0.010 μs（0.011%），第 3 轮慢 0.005 μs，无法确认稳定收益。两者不采用，也没有继续完整 fallback/显式 LDS descriptor scope 验收。controller 于 2026-10-17 03:21:59 UTC 完成并恢复正式 review；原筛选单次 93.708 μs 不替代三轮中值。其他 E65 工作负载出现时按六秒等待，未竞争 GPU 或终止其进程。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v44.py
# On host; use fresh batch names for reproduction.
SCREEN_BATCH=<new_output_arena_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg8_output_arena cyclic64_mg8_output_arena_early
PAIR_BATCH=<new_output_arena_pair_batch> VALIDATION_BATCH=<new_output_arena_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/pair_verified_candidates_host.sh \
  cyclic64_mg8_output_arena cyclic64_mg8_output_arena_early
```

原始证据：[CPU resource controller](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_output_arena_controller.log)、[完整筛选 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_output_arena_20261017_controller.log)、[三轮 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_output_arena_pair_20261017_controller.log)。

## 三种 wave 映射：完整精度通过，筛选未取得收益

[v45 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v45.py) 只修改原 `M256_FUSED_LAYOUT` 的 `wave_m` / `wave_n`，每个候选仍为同一 2×4 个 M128/N64 compute tiles 的双射。A/B/ScaleA/ScaleB producer 的 owner 不变，所有 K128 contribution、每 accumulator 的 K 顺序、quantization、输出地址和完成等待保留。256 行和全部 N 列仍覆盖一次。

| 候选 | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | spills |
|---|---:|---|---:|---:|
| `cyclic64_mg8_wave_slotxor` | 95.447 | 470 / 106 | 278528 | 0 |
| `cyclic64_mg8_wave_pair` | 95.546 | 471 / 106 | 278528 | 0 |
| `cyclic64_mg8_wave_pairxor` | 95.958 | 471 / 106 | 278528 | 0 |

三者完整 11 cases equivalence 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output，以及完整动态输入/poison/eager/graph replay 全部 bitwise mismatch=0。未观察到筛选收益，没有继续三轮交替或扩大 fallback 验收，不采用。以上为原 eager profiler、count=19 的空闲单次；不能把它们与较早正式中值作严格配对比较。controller 于 2026-10-17 03:37:59 UTC 完成并恢复 review。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v45.py
# On host; use a fresh batch name.
SCREEN_BATCH=<new_wave_mapping_batch> \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg8_wave_slotxor cyclic64_mg8_wave_pair cyclic64_mg8_wave_pairxor
```

原始证据：[mapping contract](../.codex_tmp/gemm1_t1536_20261017/logs/wave_mapping_source_contract.json)、[CPU controller](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_wave_mapping_controller.log)、[完整筛选 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wave_mapping_20261017_controller.log)。各候选逐项 equivalence/replay/bench 与 CPU inspection 日志已下载到本地同目录。

## 更大 MMA_GROUP 与 M128/N128 小 tile：精度通过，未采用

[v48 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v48.py) 将原 M256 `MMA_GROUP` 分别改为 12、16；其余完整 module AST 不变。单纯扩大 scheduling group 没有改变 WMMA、LDS 读取、TDM、fences 或 quantization。Shader Programming Guide §5.3.2 说明 source-cache / VGPR-bank 行为受指令和 EXEC/memory 变动影响，但没有保证扩大 group 必然加速。

| 候选 | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | private / spills |
|---|---:|---|---:|---|
| `cyclic64_mg12_review` | 96.042 | 470 / 106 | 278528 | 0 / 0 |
| `cyclic64_mg16_review` | 94.863 | 470 / 106 | 278528 | 0 / 0 |
| `split128_mn128_mg8_v3` | 172.339 | 274 / 102 | 139264 | 0 / SGPR 8 |
| `split128_mn128_mg8_pad_v3` | 170.720 | 274 / 103 | 141312 | 0 / SGPR 8 |

小 tile 由 [v46 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v46.py) 将每个原 M256/N256 task 拆成四个 M128/N128 WG，内部配置 `w2x2/b4/q4`。producer 和两阶段 256 行 expert alignment 不变，每 WG 保留四个 N-task，通过两个 N-task groups 覆盖完整 N4096。完整 padded M/N/K 计算量与原方案相同；没有按有效行减少计算。每 accumulator 的 K128 顺序、SiLU/BF16/E8M0/native FP4 和外部 payload/ScaleA 布局不变。

首版存在两项实验实现问题，在进入 GPU 验证前修正：

- `psum` 记录未 padding 的 expert ends。第二个 M128 half 可能从 parent M256 的尾部 padding 开始，按该 half 地址 bisect 会错误选择下一 expert。[v47 修正](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v47.py) 以 `blk_m & -256` 查找 parent owner，input/output 仍用实际 `blk_m`，valid extent 为 `max(psum[e]-blk_m,0)`。506 个 half 地址代数检查通过，随后再做实际 GPU 验证。这是新 tile 拆分引入的候选问题，不是已采用 review 的问题。
- 首版和 v2 CPU compilation 各达到 120 秒限制并终止。35/70/105 秒诊断栈均停在 `flydsl/compiler/ast_rewriter.py::visit_BoolOp` / `ast.fix_missing_locations`，发生在 module import 的 `@flyc.jit` rewriting，尚未进入 MLIR/LLVM。21 项 `and` guard 的展开反复嵌入已生成表达式。[v49 修正](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v49.py) 用同一组 constexpr predicates 的 `all((...))` 避免该展开，保留 ownership 修正；恢复 guard 后其余完整 AST 与 v2 一致。v3 两个版本 CPU 编译均成功，未修改 FlyDSL 安装或放宽超时。

四个候选完整 11 cases equivalence、完整动态输入/poison/eager/graph replay 均 bitwise mismatch=0。每次原 benchmark 的 sanity 仍为 `logits_diff=3.4733e-06`，kernel count=19。mg12/mg16 未确认收益；小 tile 显著回退，因此四者均未采用、没有再做三轮或扩大 fallback 验收。CPU ISA 证实小 tile 各有八组 `v_writelane_b32` / `v_readlane_b32` 保存/恢复 SGPR，但该证据不能单独解释全部回退。

资源判断依据：Shader Programming Guide §4.7.1（printed p.160）记载每 WGP 最多 320 KiB LDS；§3.3.2.1（printed p.32）记载 MI450 每 SIMD 的 1024 VGPR pool 和 wave32 16-register allocation granularity。原 review 的 278528 B LDS 不能容纳第二个同尺寸 WG；小 tile 的 LDS 降到约一半只是放宽其中一项资源限制，不能据此宣称实际驻留数或性能提高。§5.7.6（printed p.235）还记载 SIMD-pair 共享 LDS issue 且 bank conflicts 会降低性能。这里没有为小 tile 抓取新 ATT，因此未将额外 WG/request 数或 spills 推测写成已证明的耗时占比。

```bash
# Reproduce the private sources, retaining parent ownership and the rewritten guard.
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v46.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v47.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v48.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v49.py
# On host; every numerical/benchmark command runs inside dsr1_ep4.
SCREEN_BATCH=<new_mma_geometry_batch> \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  cyclic64_mg12_review cyclic64_mg16_review \
  split128_mn128_mg8_v3 split128_mn128_mg8_pad_v3
```

controller 于 **2026-10-17 03:52:52 UTC** 完成，确认恢复 `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790`。其他 E65 工作负载出现时按六秒检查等待，未终止其进程。四个候选逐项 GPU 日志已全部下载，前一轮 output arena 的逐项验证、计时和九个三轮日志也已补齐。

证据：[本批 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_mma_geometry_20261017_controller.log)、[大 MMA group contract](../.codex_tmp/gemm1_t1536_20261017/logs/larger_mma_groups_source_contract.json)、[tile workload contract](../.codex_tmp/gemm1_t1536_20261017/logs/split128_source_contract.json)、[parent ownership](../.codex_tmp/gemm1_t1536_20261017/logs/split128_parent_ownership_contract.json)、[guard AST contract](../.codex_tmp/gemm1_t1536_20261017/logs/split128_guard_rewrite_contract.json)、[卡点诊断栈](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_split128_mn128_mg8_v2_diagnostic_compile.log)、[v3 CPU controller](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_larger_mma_split128_v3_controller.log)、[小 tile ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_split128_mn128_mg8_v3_inspection.log)、[padding 版 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_split128_mn128_mg8_pad_v3_inspection.log)。

## Wave priority：完整精度通过，筛选未取得收益

[v50 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v50.py) 分别测试 wave4..7、wave0..3 的 `USER_PRIO=1`，以及每个 K128 WMMA 区间临时 priority1、完成后恢复0。所有实验仅作用于原 `M256_FUSED_LAYOUT`，输入地址、全部 WMMA、每 accumulator 的 K 顺序、quantization 和同步完成条件保留。去掉新增 priority statements 后，完整 module AST 与 review 相同。

已有同 dispatch ATT 的 sampled peer 在 steady rendezvous 到达时间有差异，但不能据此确定整个 WG 的最后到达者。Shader Programming Guide §5.2 / §5.2.1 / §5.2.2 记载 `S_SETPRIO`、priority 调整和 WG round robin 的相互影响，并没有保证不等 priority 会加速；本实验因此只作为待验证假设。

| 候选 | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | private / spills |
|---|---:|---|---:|---|
| `cyclic64_mg8_prio_upper` | 95.523 | 470 / 106 | 278528 | 0 / 0 |
| `cyclic64_mg8_prio_lower` | 94.901 | 470 / 106 | 278528 | 0 / 0 |
| `cyclic64_mg8_prio_wmma` | 95.394 | 486 / 106 | 278528 | 0 / 0 |

三者完整 11 cases equivalence 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output，以及完整动态输入/poison/eager/graph replay 全部 bitwise mismatch=0。单次均使用原 eager profiler、count=19、全机空闲条件；sanity 为 `logits_diff=3.4733e-06`。没有观察到筛选收益，未继续三轮交替或扩大 fallback 验收，三者均不采用。不能把本批单次结果与较早 review 中值作严格配对比较。

首轮 wrapper 的 CPU upper compilation 已成功，随后 `docker exec chmod` 因未指定仓库 cwd 而失败，未启动 GPU workload。修正 wrapper 的 `-w "$PWD"` 后复用已成功的 compile log，再继续三者 inspection 和 GPU 验证。成功 controller 于 **2026-10-17 04:08:34 UTC** 完成，确认恢复正式源码 `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790`。04:15:59 UTC 重新检查机器，未发现使用 GPU 的进程。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v50.py
# On host; use an unused batch label and fresh inspection paths.
SCREEN_BATCH=<new_wave_priority_batch> \
bash .codex_tmp/gemm1_t1536_20261017/priority_candidates_host.sh \
  cyclic64_mg8_prio_upper cyclic64_mg8_prio_lower cyclic64_mg8_prio_wmma
```

证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/wave_priority_source_contract.json)、[首轮 wrapper 失败](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wave_priority_20261017_controller.log)、[成功 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wave_priority_v2_20261017_controller.log)、[upper ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_prio_upper_inspection.log)、[lower ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_prio_lower_inspection.log)、[WMMA priority ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_prio_wmma_inspection.log)。逐项 CPU compilation 和 GPU equivalence/replay/bench 日志均保存在同一本地目录。

## LDS 读取顺序与 K128-0 分组：三个通过，一个 memory fault，均未采用

[v51 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v51.py) 只在原 `M256_FUSED_LAYOUT` 内调整完整 A/B/scales 读取顺序，或将 K128-0 的 scheduling group 改为 4/2 并保留 closing 4 WMMA。K128-1 仍用 group8，实际 LDS load 集合、全部 WMMA、每 accumulator 的 K 顺序、完成等待、quantization 和输出地址保留。恢复修改的函数/表达式后，完整 module AST 与正式 review 相同；这只证明源码修改范围，不证明运行时正确性。

| 候选 | 完整 equivalence / replay | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | spills |
|---|---|---:|---|---:|---:|
| `cyclic64_mg8_lds_a_first` | 通过 / 通过 | 95.797 | 471 / 106 | 278528 | 0 |
| `cyclic64_mg8_lds_ab_interleave` | memory fault / 未运行 | 未计时 | 471 / 106 | 278528 | 0 |
| `cyclic64_mg8_k0_group4_tail4` | 通过 / 通过 | 96.069 | 444 / 106 | 278528 | 0 |
| `cyclic64_mg8_k0_group2_tail4` | 通过 / 通过 | 96.834 | 444 / 106 | 278528 | 0 |

通过的三个候选完整 11 cases equivalence 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output，以及动态输入/poison/eager/graph replay 全部 bitwise mismatch=0。计时使用用户原 eager profiler、count=19 和全机 GPU 空闲条件，sanity 为 `logits_diff=3.4733e-06`。没有观察到筛选收益，因此未进行三轮交替，均未采用。降低 VGPR 到 444 本身未带来加速。

交错读取候选通过 seeds0/17/43 random、skew、sparse、hot8 六组后，在后续 `tiny_expert` 验证附近发生 `Memory access fault by GPU node-2 ... Reason: Page not present or supervisor privilege.`，进程退出码 134。它没有通过完整精度验证，没有运行 replay 或 benchmark。故障根因尚未证明，不能称为已定位的 compiler bug，也不能把 controller 的 exit0 解释为四个候选全部通过。wrapper 按正确性门禁拒绝该候选，继续验证其余候选，最后恢复正式 review；未进行 GPU reset。

Shader Programming Guide §5.7.6 说明 SIMD-pair 共享 LDS issue、bank conflicts 会影响吞吐；`HGEMM_Optimizations_And_Recommendations#1.txt` 建议 A/B 使用不同 segments 并交错读取。这些资料提供实验方向，并不保证本 kernel 的重排正确或加速。本批结果也没有新的 ATT，不能把降低寄存器数或某种读取顺序当作已证明的性能解释。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v51.py
# On host; use an unused batch label and fresh inspection paths.
SCREEN_BATCH=<new_lds_prefetch_batch> \
bash .codex_tmp/gemm1_t1536_20261017/priority_candidates_host.sh \
  cyclic64_mg8_lds_a_first cyclic64_mg8_lds_ab_interleave \
  cyclic64_mg8_k0_group4_tail4 cyclic64_mg8_k0_group2_tail4
```

controller 于 **2026-10-17 04:38:11 UTC** 完成并恢复正式源码 `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790`。04:42:05 UTC 再次核对 SHA、GPU/KFD、busy 和 VRAM，没有本任务测试残留。原始证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/lds_prefetch_groups_source_contract.json)、[完整 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_lds_prefetch_v51_20261017_controller.log)、[fault equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_lds_prefetch_v51_20261017_cyclic64_mg8_lds_ab_interleave_equivalence.log)。逐项 CPU compilation、ISA/resource、equivalence/replay/bench 日志已保存在同一本地目录。

## 两个 M128 组独立三级 pipeline：完整精度通过，明显回退

[v52 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v52.py) 保留原 M256/N256 task、八条 compute waves 和全部 K7168 contribution，让 wave parity 对应的两个 M128 组各使用三级输入 ring 和四成员 named barrier。每个组的四条 wave 分别负责 A128、完整 B256、ScaleA128、完整 ScaleB256，只有 B/ScaleB 搬运翻倍，实际矩阵计算与每 accumulator 的 K 顺序不变。外部 routing、256 行 expert alignment、全部 output 地址及数值 epilogue 保留，GEMM2 源码不变。

四级独立 rings 需要 `2 × 4 × (16384 + 32768 + 1024 + 2048) = 417792 B`，超过 Shader Programming Guide §4.7.1 的 320 KiB LDS 上限。三级改为 **313344 B**，不需要越过容量限制。每 half 的 input producer 和 reader 使用同一 ring 地址；当前 stage 的 DS completion、next stage 的 tensor completion 都在组 barrier signal 前完成。INIT 与各 wave 的 JOIN 通过一次 WG rendezvous 完成，再开始首代 signal。output 复用 input arena 之前仍执行全 WG input completion，下一 N-task refill 之前完整等待旧 output，不采用原跨 task 的 I0/I1 overlap。

| 候选 | 完整 equivalence / replay | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | private / spills |
|---|---|---:|---|---:|---|
| `split_mhalf_b3_named_drain` | 通过 / 通过 | **114.585** | 473 / 107 | 313344 | 0 / 0 |

完整 11 cases 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output，以及动态输入/poison/eager/graph replay 全部 bitwise mismatch=0。计时使用原完整 MoE eager profiler、count=19、全机空闲条件，sanity `logits_diff=3.4733e-06`。没有性能收益，未做三轮复测，未采用。该结果说明这个完整实现的额外开销未被分组同步收益抵消；没有为它抓新 ATT，不能把全部回退单独归因于 B 搬运翻倍、三级 buffer 或 output drain。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v52.py
# On host; use an unused batch label and fresh inspection paths.
SCREEN_BATCH=<new_mhalf_batch> \
bash .codex_tmp/gemm1_t1536_20261017/priority_candidates_host.sh \
  split_mhalf_b3_named_drain
```

controller 于 **2026-10-17 05:06:18 UTC** 完成并恢复正式 review。05:07:03 UTC 再次核对 SHA 与 GPU/KFD，无本任务残留；期间出现的其他 E65 benchmark 均按六秒等待，未终止其进程。证据：[完整地址/源码 contract](../.codex_tmp/gemm1_t1536_20261017/logs/split_mhalf_pipeline_source_contract.json)、[CPU resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_split_mhalf_b3_named_drain_inspection.log)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_mhalf_v52_20261017_controller.log)。逐项 CPU compilation、equivalence/replay/bench 日志已下载到同一本地目录。

## 精确 producer/consumer dependency union：代次修正通过，性能回退

[v53 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v53.py) 保留四级 cyclic ring、全部 input/output TDM、完整 WMMA 与每 accumulator 的 K128 顺序。每个 listener 的 named barrier 只统计其下一块数据的 producer，以及即将复用 LDS 区域的 consumer。八个 barrier 的 member count 为 `[4,6,6,7,6,4,7,6]`，从全 WG rendezvous 缩小为实际依赖的 union。首版直接重用同一组八个 barrier：快 wave 可能在慢成员完成本代前发出下一代 signal，使硬件计数无法区分代次。`cyclic64_mg8_union_tail` 的 equivalence 在 120 秒达到限制，exit=124，进程组按要求终止，未完成完整 case、未运行 replay、没有性能数据。front 版本仅编译与静态 inspection，未运行 GPU。首版不是可用优化，不采用。

[CPU 乱序事件模型](../.codex_tmp/gemm1_t1536_20261017/check_union_barrier_generations.py) 的 2048 个随机调度中，单组复用全部检出 mixed generation；奇偶两组全部通过。模型假定同 wave TDM 有序，只证明抽象事件协议，不能代替 emitted ISA/GPU 验证，也不能将首版超时根因称为已在硬件上证实。

[v54 修正](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v54.py) 分配两组共 16 个 named barrier，按 `current_kt & 1` 选本代；每个 listener 在 signal 前 JOIN 本代，仍只同时 JOIN 一个 named barrier。输入加载、WMMA、量化及 task-boundary 完成等待未改。Shader Programming Guide §4.3.6.5.5 规定每 wave 只能 JOIN 一个 named barrier、JOIN 清其 complete bit；§4.10.1 规定 TDM 只在同 wave 有序，跨 wave 的 completion 仍须建立同步关系。这里不能用“同 wave 有序”替代 LDS reader 的完成等待。

| 候选 | 完整 equivalence / replay | 原命令单次 GEMM1（μs） | VGPR / SGPR | SGPR spill |
|---|---|---:|---|---:|
| `cyclic64_mg8_union_parity_tail` | 全部 bitwise mismatch=0 | 101.728 | 500 / 106 | 4 |
| `cyclic64_mg8_union_parity_front` | 全部 bitwise mismatch=0 | 103.990 | 474 / 106 | 4 |

两版 LDS 均为 278528 B，private=0、VGPR spill=0。完整 11 cases 覆盖 random seeds0/17/43、skew、sparse、hot8、tiny expert、balanced、T1537/T1792/T2048；canonical payload、ScaleA、GEMM2 valid rows、MoE output 均 mismatch=0，dynamic input、poison、eager/graph replay 全部通过。原命令 count=19，sanity 仍为 `logits_diff=3.4733e-06`。两个奇偶版都没有筛选收益，不采用，也未做三轮。相比 review 增加 SGPR 保存/恢复、JOIN、多个 signal；这些静态证据支持同步开销未必更少，但没有新 ATT，不能把全部回退单独归因于 SGPR spill 或 signal 数。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v53.py
python3 .codex_tmp/gemm1_t1536_20261017/check_union_barrier_generations.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v54.py
# On host; use a fresh batch label. Every container test has a 120 s limit.
SCREEN_BATCH=<new_union_parity_batch> \
bash .codex_tmp/gemm1_t1536_20261017/priority_candidates_host.sh \
  cyclic64_mg8_union_parity_tail cyclic64_mg8_union_parity_front
```

v54 controller 于 **2026-10-17 05:33:35 UTC** 完成恢复 review，逐项 CPU compilation、ISA/resource、equivalence/replay/bench 日志已下载。证据：[首版 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_union_v53_20261017_controller.log)、[抽象乱序结果](../.codex_tmp/gemm1_t1536_20261017/logs/union_barrier_interleavings.json)、[奇偶版完整 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_union_v54_20261017_controller.log)、[tail ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_union_parity_tail_inspection.log)、[front ISA](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_union_parity_front_inspection.log)。

## Producer wave 重映射：完整精度通过，未取得筛选收益

[v55 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v55.py) 只在原 `M256_FUSED_LAYOUT` 重映射 A/B/ScaleA/ScaleB 的 producer wave groups；compute wave mapping、四级 ring、全部 K contributions、input/output bytes 和所有 completion fences 不变。每 wave 每 K256 仍发出一条 input TDM。非连续 owner group 的 ordinal 不能继续用 `wave - group[0]`；改为第一个 owner=0、第二个=1，并同时用于 global segment offset 与 cyclic destination。

首版把两个 Python `int` 直接传给 FlyDSL `.select(0,1)`，CPU compilation 报 `AttributeError: 'int' object has no attribute '_CAPIPtr'`，三个版本均未进入 GPU。[v56 修正](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v56.py) 使用 `.select(fx.Int32(0),fx.Int32(1))`，保留首版 artifacts，生成 `_v2` 新文件。去掉三个 producer/ordinal edits 后整个 module AST 与 review 一致；静态双射检查覆盖全部 wave0..7，但正确性仍以实际 GPU 验证为准。

| 候选 | A / B / ScaleA / ScaleB groups | 原命令单次 GEMM1（μs） | VGPR / SGPR | SGPR spill |
|---|---|---:|---|---:|
| `cyclic64_mg8_producers_stride4_v2` | `(0,4)/(1,5)/(2,6)/(3,7)` | 96.057 | 480 / 107 | 2 |
| `cyclic64_mg8_producers_stride2_v2` | `(0,2)/(1,3)/(4,6)/(5,7)` | 95.034 | 470 / 102 | 0 |
| `cyclic64_mg8_producers_within_reader_v2` | `(0,1)/(2,6)/(4,5)/(3,7)` | 95.222 | 471 / 107 | 0 |

三个 `_v2` 候选的完整 11 cases equivalence 与完整 dynamic input/poison/eager/graph replay 全部通过，canonical payload、ScaleA、GEMM2 valid rows、MoE output 均 bitwise mismatch=0。LDS=278528 B，private=0、VGPR spill=0；原命令 count=19，sanity=`3.4733e-06`。三个版本都未筛选出值得继续三轮验证的收益，未采用。这些单次结果不能与旧批中值相减作为已确认的回退百分比。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v56.py
# On host; use a fresh batch label.
SCREEN_BATCH=<new_producer_wave_batch> \
bash .codex_tmp/gemm1_t1536_20261017/priority_candidates_host.sh \
  cyclic64_mg8_producers_stride4_v2 \
  cyclic64_mg8_producers_stride2_v2 \
  cyclic64_mg8_producers_within_reader_v2
```

controller 于 **2026-10-17 05:51:12 UTC** 完成，正式 SHA 恢复为 `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790`；05:55:35 UTC 全机 idle，KFD 无进程、busy=0、VRAM=176582656 B。证据：[首版 CPU failure](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_producers_v55_20261017_controller.log)、[v2 contract](../.codex_tmp/gemm1_t1536_20261017/logs/producer_wave_v2_source_contract.json)、[完整 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_producers_v56_20261017_controller.log)。逐项验证/bench 与 CPU inspection 归档在同一本地 logs 目录。

## 完整 K256 operands staging 后提前 refill：精度通过，筛选无收益

[v57 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v57.py) 仅改变原 `M256_FUSED_LAYOUT` 的 steady K256 路径。完整当前 K256 的 A/B/ScaleA/ScaleB 已进入两个 register slots 后，先计算 K128-0 的前 2 或 4 个 M16 rows，再执行原 READY/REUSE fence 并发下一 input TDM，随后计算其余 K128-0 和完整 K128-1。下一块 carry 读取继续由 K128-1 覆盖。每个 steady K256 仍有一个完整 WG fence，四级 ring、全部 input transfers、每 accumulator 的 K128 顺序、epilogue 和 N-task 边界协议均保留；fallback 完整原 body 不变。

移除新增 body 后整个 module AST 与 review 相同；完整 padded M256/N256/K7168 的 WMMA 索引和每 accumulator 的 K128 顺序枚举也相同。该静态核对只界定改动，不代替 GPU 验证。

| 候选 | 完整 equivalence / replay | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | private / spills |
|---|---|---:|---|---:|---|
| `cyclic64_mg8_refill_after_rows2` | 全部 bitwise mismatch=0 | **97.901** | 470 / 106 | 278528 | 0 / 0 |
| `cyclic64_mg8_refill_after_rows4` | 全部 bitwise mismatch=0 | **96.723** | 470 / 106 | 278528 | 0 / 0 |

两者完整 11 cases equivalence 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output，以及 dynamic input、poison、eager/graph replay 均 mismatch=0。计时使用原完整 MoE eager profiler、count=19、全机空闲条件，sanity=`logits_diff=3.4733e-06`。未取得筛选收益，不做三轮、不采用。CPU ISA 两者均为 4381 条静态指令；`s_wait_dscnt` 静态数量分别为 80/88。没有新的 ATT，不能由这些数量单独推导实际停顿时间或全部回退原因。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v57.py
# On host; use a fresh batch label. Every container test has a 120 s limit.
SCREEN_BATCH=<new_full_k256_refill_batch> \
bash .codex_tmp/gemm1_t1536_20261017/priority_candidates_host.sh \
  cyclic64_mg8_refill_after_rows2 cyclic64_mg8_refill_after_rows4
```

controller 于 **2026-10-17 06:06:29 UTC** 完成并恢复 review；06:30 UTC 显式复核正式 SHA 与全机 GPU/KFD，无测试残留，busy=0、VRAM=176582656 B。证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/full_k256_early_refill_source_contract.json)、[完整 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_refill_v57_20261017_controller.log)、[rows2 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_refill_after_rows2_inspection.log)、[rows4 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_refill_after_rows4_inspection.log)。两者 CPU compilation、equivalence/replay/bench 日志已全部下载到同一本地 logs 目录。

## 16-wave 完整 M256 task：ScaleA 实验修正后通过精度，性能回退

[v58 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v58.py) 保留同一个完整 padded M256/N256/K7168 task 和四个 persistent N-task，将 compute 从两个 M128 partition 改为四个 M64 partition（`w4x4/q4`）。输入仍搬运完整 A256/B256 及全部 scales，四级 linear 64 KiB data ring、独立 scale ring 和独立 output arena 共占 295936 B LDS。单个 K128 operand register slot 在全部当前 WMMA 完成后复用；原 counted READY/REUSE 和 task boundary 完成条件保留。

首版 `linear64_mg8_waves16_rmem1` 在第一组 random seed0 equivalence 失败：`MoE accuracy failed: 0.6554650326015814`。新的 reader 使用 linear ScaleA，producer 却保留了正式 cyclic 布局的 owner offsets，第二 owner 向下一 stage 写入。32 个 stage/superrow 地址检查中有16个不一致；[v59 修正](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v59.py) 在该实验分支禁用 cyclic ScaleA owner offsets，32个检查全部一致。首版未运行 replay 或 benchmark；这一问题只属于新候选，不是正式 review 的 bug。

修正版 `linear64_mg8_waves16_rmem1_v2` 完整11 cases 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output，以及动态输入/poison/eager/graph replay 全部 bitwise mismatch=0。CPU resource 为 VGPR237（allocated240）、SGPR106、LDS295936 B，private/spills=0；静态 ISA 2946 instructions。原命令空闲单次为 **110.995 μs**，count=19、sanity `logits_diff=3.4733e-06`。没有取得筛选收益，未做三轮交替，未采用。

Shader Programming Guide §3.3.2.1（printed p.32）规定 MI450 每 SIMD 1024 VGPR、wave32 按16分配；allocated240 通过四-wave/SIMD 的 VGPR 容量门槛。§4.7.1（printed p.160）规定 LDS 上限320 KiB；本候选未超过。通过静态门槛不证明实际 occupancy 增加或性能提高，本批没有新 ATT，不能将全部回退单独归因于某个资源或 barrier。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v58.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v59.py
# On host; the successful source alone is shown below.
SCREEN_BATCH=<new_waves16_single_slot_batch> \
bash .codex_tmp/gemm1_t1536_20261017/screen_validated_candidates_host.sh \
  linear64_mg8_waves16_rmem1_v2
```

两个 controller 分别于 **07:02:49 / 07:07:46 UTC** 完成恢复 review。证据：[首版失败 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_waves16_v58_20261017_controller.log)、[修正版 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_waves16_v59_20261017_controller.log)、[ScaleA producer mapping](../.codex_tmp/gemm1_t1536_20261017/logs/waves16_scale_producer_v2_source_contract.json)、[修正版 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_linear64_mg8_waves16_rmem1_v2_inspection.log)。逐项 CPU/GPU 原始日志和 ISA 保存在同一本地目录。

## 16-wave quadrant operand prefetch：两项完整精度通过，仍无收益

[v60 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v60.py) 基于修正后的 linear ScaleA 版本，只改变当前/下一 K128 operands 的寄存器存活与 WMMA overlap。每个 seed 保留两个 A M16 fragments、B0 和全部 scales；当前临时加载另两个 A fragments 及 B1。Q0 的两条 WMMA 覆盖当前临时读取，随后保留原 READY/REUSE fence/refill/carry，Q1/Q2/Q3 覆盖下一 seed 读取。完整 operand bytes 和每 accumulator 的56个 K128 contribution 顺序相同；input/output 地址、任务划分及量化不变。

| 候选 | 完整 equivalence / replay | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | private / spills |
|---|---|---:|---|---:|---|
| `linear64_waves16_quadrant_reuse0` | 通过 / 通过 | **109.386** | 238 / 106 | 295936 | 0 / 0 |
| `linear64_waves16_quadrant_reuse1` | 通过 / 通过 | **109.388** | 238 / 106 | 295936 | 0 / 0 |

两者都先通过 CPU allocated-VGPR≤256、LDS≤320 KiB、private/spills=0 的 gate，静态 ISA 为2966 instructions；再通过完整11 cases equivalence 和动态输入/poison/eager/graph replay，所有 canonical payload、ScaleA、GEMM2 valid rows、MoE output 均 bitwise mismatch=0。原 eager profiler count=19，sanity `logits_diff=3.4733e-06`，测量前按全机 GPU 空闲检查等待。reuse0 关闭显式 operand reuse，reuse1 为相邻同 weight 的 WMMA 设置 reuseA；本批没有新 ATT，不将两项几乎相同的单次耗时解释为普遍的硬件结论。

二者没有取得筛选收益，未做三轮交替，也未合入正式 kernel。controller 于 **2026-10-17 07:16:38 UTC** 完成并恢复正式 SHA `b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790`；07:28:03 UTC 再次核对源码 SHA、GPU/KFD，无本任务残留。期间出现的其他 E65 benchmark 按六秒等待，没有终止其进程。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v60.py
# On host; CPU compilation, resource gate, numerical checks, then original benchmark.
SCREEN_BATCH=<new_waves16_quadrant_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  linear64_waves16_quadrant_reuse0 linear64_waves16_quadrant_reuse1
```

证据：[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_waves16_v60_20261017_controller.log)、[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/waves16_quadrant_source_contract.json)、[reuse0 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_linear64_waves16_quadrant_reuse0_inspection.log)、[reuse1 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_linear64_waves16_quadrant_reuse1_inspection.log)。两项逐项 compile/inspection/equivalence/replay/bench 原始日志已下载到同一本地目录。

## 按槽位展开 steady K256：精度通过，新增 spill，未采用

[v61 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v61.py) 仅在 `M256_FUSED_LAYOUT` 中，将完整 steady K256 step 以两级/四级展开。保留首 task 从kt0、后续 task 从kt1开始，全部迭代到kt23，再执行原 drain kt24..27。每 step 的完整 input TDM、原 counted READY/REUSE、K128 contributions 和 task boundary 不变；四级展开让 ring slot 在编译时为常量。CPU 源码模型核对两种起点的完整 kt/ring-slot 序列与 review 相同。

首版两个 CPU compilation 都报错，尚未生成可运行 kernel：`Variable(s) ['group'] initialized as None before a dynamic if/else (scf.if)`。新增 runtime `group` induction variable 被 FlyDSL rewriter 视为跨动态 branch 的结果。[v62 修正](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v62.py) 使用原有 `kt` induction variable 的 step-sized interval，加 constexpr offsets，完整迭代序列相同；未修改 FlyDSL 安装。首版没有 GPU 性能数据。

| 修正版 | 完整 equivalence / replay | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | private（B） | VGPR / SGPR spills |
|---|---|---:|---|---:|---:|---|
| `cyclic64_mg8_steady_unroll2_v2` | 通过 / 通过 | **98.848** | 498 / 107 | 278528 | 0 | 0 / 8 |
| `cyclic64_mg8_steady_unroll4_v2` | 通过 / 通过 | **237.496** | 512 / 107 | 278528 | 724 | 361 / 16 |

两者完整11 cases equivalence 和动态输入/poison/eager/graph replay 全部 bitwise mismatch=0，原 eager profiler count=19、sanity `logits_diff=3.4733e-06`。计时前检查全部 GPU 空闲，其他 E65 workload 出现时按六秒等待，没有终止其进程。静态 ISA 分别为6358/10698 instructions；数目包含互斥路径，不等于动态执行次数。

展开没有取得筛选收益，二者均不采用、不做三轮交替。四级版明确新增 private memory、VGPR 和 SGPR spills，属于回退证据；没有为它抓取新 ATT，不能将全部增加的 μs 定量归因于 spill。后续 CPU gate 要求 private/VGPR spill/SGPR spill 全为0，避免继续对有明显资源问题的候选做 GPU 性能筛选。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v61.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v62.py
# On host; original batch before the stricter zero-spill screening gate.
SCREEN_BATCH=<new_steady_unroll_batch> \
bash .codex_tmp/gemm1_t1536_20261017/priority_candidates_host.sh \
  cyclic64_mg8_steady_unroll2_v2 cyclic64_mg8_steady_unroll4_v2
```

controller 于 **2026-10-17 07:46:17 UTC** 完成恢复 review；07:49:52 UTC 显式核对正式 SHA、文档 SHA 与全机 GPU/KFD，无本任务残留，busy=0、VRAM=176582656 B。证据：[首版 CPU controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_steady_unroll_v61_20261017_controller.log)、[修正版 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_steady_unroll_v62_20261017_controller.log)、[原 step 序列 contract](../.codex_tmp/gemm1_t1536_20261017/logs/steady_unroll_source_contract.json)、[修正 contract](../.codex_tmp/gemm1_t1536_20261017/logs/steady_unroll_v2_source_contract.json)、[两级 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_steady_unroll2_v2_inspection.log)、[四级 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_steady_unroll4_v2_inspection.log)。逐项 CPU/GPU 原始日志已下载。

## A/ScaleA 与 B/ScaleB 独立 prefetch 分级：完整精度通过，未取得收益

[v63 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v63.py) 保留八个 compute wave、完整 M256/N256/K7168 和四个 persistent N-task，将八份 data staging 容量分别分配为 A 三级/B 五级或 A 两级/B 六级。ScaleA/ScaleB 跟随对应 data ring，output arena 独立；LDS 总量为295936 B。当前 reader 完成原 full-WG carry fence 后才复用 slot。下一 task 的 I0/I1 在上一 task 最后一次读取完成后发出，并仍排在 output TDM 之前；没有减少 input request、matrix contribution 或 epilogue。

| 候选 | 完整 equivalence / replay | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | private / spills |
|---|---|---:|---|---:|---|
| `linear64_mg8_a3_b5_prefetch` | 通过 / 通过 | **111.155** | 511 / 94 | 295936 | 0 / 0 |
| `linear64_mg8_a2_b6_prefetch` | 通过 / 通过 | **116.759** | 490 / 98 | 295936 | 0 / 0 |

两项均先通过 allocated-VGPR≤512、LDS≤320 KiB、private/VGPR spill/SGPR spill 全为0的 CPU gate；完整11 cases 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output，以及动态输入/poison/eager/graph replay 全部 bitwise mismatch=0。正式计时仍使用原完整 MoE eager profiler、count=19、random non-balanced，sanity=`logits_diff=3.4733e-06`；每次计时前通过全机空闲检查。两项未取得筛选收益，不做三轮交替，不合入。没有新 ATT，不能把回退定量归因于某一 ring 深度或某一个 wait。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v63.py
# On host; use a fresh batch label.
RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_differential_ring_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  linear64_mg8_a3_b5_prefetch linear64_mg8_a2_b6_prefetch
```

controller 于 **2026-10-17 08:00:01 UTC** 完成并恢复 review。证据：[完整 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_differential_ring_v63_20261017_controller.log)、[A3/B5 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_linear64_mg8_a3_b5_prefetch_inspection.log)、[A2/B6 ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_linear64_mg8_a2_b6_prefetch_inspection.log)。逐项 CPU compilation、inspection、equivalence、replay 和原 benchmark 日志已下载到同一本地目录。

## 八-wave quadrant operand 调度：CPU 发现 spill，尚未 GPU 验证

[v64 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v64.py) 沿用正式八-wave cyclic64 geometry。每个 K128 seed 仅保存 FRONT 四个 A fragments、B0 和全部 scales；BACK 四个 A fragments 与 B1 作为当前临时值读取。Q0覆盖当前临时读取，Q1/Q2/Q3覆盖下一 seed 读取；`row_snake` 仅反转第二列的 M 遍历。完整 operand bytes、全部 M/N/K contributions 与每 accumulator 的56个 K128顺序相同，显式 reuse 关闭。这里只是源码 contract，不构成 GPU 正确性结果。

| 候选 | CPU 编译 | VGPR / SGPR | LDS（B） | private（B） | VGPR / SGPR spills | GPU 验证 / 性能 |
|---|---|---|---:|---:|---|---|
| `cyclic64_waves8_quadrant_plain` | 成功 | 402 / 106 | 278528 | 12 | 2 / 0 | 未运行 |
| `cyclic64_waves8_quadrant_row_snake` | 成功 | 402 / 106 | 278528 | 12 | 2 / 0 | 未运行 |

两项均因 private/spill 非0被 CPU gate 拒绝，不能写成性能提升，也不能用静态 WMMA 数量证明数值正确。plain ISA 中存在两个 `scratch_store_b32` 及相应 `scratch_load_b32`，存储值来自 LDS scale 读取；后续仅研究这些临时值的存活与调度，不放宽 gate。controller 于 **2026-10-17 08:04:13 UTC** 完成，没有替换正式 kernel。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v64.py
RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_waves8_quadrant_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_waves8_quadrant_plain cyclic64_waves8_quadrant_row_snake
```

证据：[完整 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_waves8_quadrant_v64_20261017_controller.log)、[plain ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_waves8_quadrant_plain_inspection.log)、[row_snake ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_waves8_quadrant_row_snake_inspection.log)。两项 compile/inspection 日志及 plain ISA 已下载。

补充已有 ATT 诊断：[K128 分析脚本](../.codex_tmp/gemm1_t1536_20261017/inspect_existing_att_k128.py) 重读 SIMD0 同 dispatch 两个 slot，各3584个 WMMA；steady K128内16个 WMMA首末 span 中值为241/291 cycles。decoded wave events 未单列 `s_set_vgpr_msb`，但对应 code range 存在18/19条；部分 internal instructions 不独立显示为 decoded event，因此不能推断未执行。Shader Programming Guide §3.3.2.3 说明 `S_SET_VGPR_MSB` 不需要 `S_SETREG_B32 MODE` 的额外 delay；§5.3.2 说明 source-cache reuse 与 memory-write-hit invalidation。上述事实支持继续研究 operand 存活与 LDS/WMMA交错，不证明可节省多少 μs。

## Quadrant 的 scale 存活调整：消除 spill，完整精度通过，仍未取得收益

[v65 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v65.py) 基于 v64 plain。`late_scales` 仅把下一 seed 的 scale 读取延后到关闭 quadrant，使当前 scale 临时值的存活不再与整批 next seed 重叠。`single_seed` 进一步在 Q0/Q1 已完整消费 FRONT 后复用 FRONT slot，Q2 已消费 B0 后再覆盖 B0，Q3后覆盖 scales。完整 payload/scale 读取、每 accumulator 的56个 K128贡献、原 carry fence、task boundary 和外部布局不变。

| 候选 | 完整 equivalence / replay | 原命令单次 GEMM1（μs） | VGPR / SGPR | LDS（B） | private / spills |
|---|---|---:|---|---:|---|
| `cyclic64_waves8_quadrant_late_scales` | 通过 / 通过 | **102.540** | 410 / 106 | 278528 | 0 / 0 |
| `cyclic64_waves8_quadrant_single_seed` | 通过 / 通过 | **102.378** | 394 / 106 | 278528 | 0 / 0 |

两项先通过 CPU zero-spill gate，再通过完整11 cases equivalence、动态输入/poison/eager/graph replay；payload、ScaleA、GEMM2 valid rows、MoE output 全部 bitwise mismatch=0。单次性能来自全机空闲的原完整 MoE eager profiler、count=19，sanity=`logits_diff=3.4733e-06`。资源改善没有带来筛选收益，未做三轮交替，未合入。正式 review 的 VGPR470/private0/spills0 与94.888 μs中值仍不变。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v65.py
RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_quadrant_scale_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_waves8_quadrant_late_scales cyclic64_waves8_quadrant_single_seed
```

controller 于 **2026-10-17 08:19:12 UTC** 完成并恢复 review，随后显式核对正式SHA一致。证据：[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_waves8_quadrant_v65_20261017_controller.log)、[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/waves8_quadrant_v65_source_contract.json)、[late scales ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_waves8_quadrant_late_scales_inspection.log)、[single seed ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_waves8_quadrant_single_seed_inspection.log)。两项 compile/inspection/equivalence/replay/bench 日志已全部下载。

## VOP3PX2 固定字段核对：发现 compiler 编码与 ISA 文档不一致

CDNA5 ISA §14.3.7 Table 97（printed p.191）明确要求 load-scale 部分的 `bits[58:50]` 固定为`0x100`。Shader Programming Guide §4.6（printed p.159）记录硬件问题`DEGFXMI400-12891`，描述该字段若设为zero会产生额外stall；它没有定量说明`0x80`与`0x100`的差别。因此不能只凭字段不一致就认定本workload一定提速。

只读[ELF audit](../.codex_tmp/gemm1_t1536_20261017/audit_wmma_scale_constant.py) 以llvm-objdump提供instruction boundary，直接读取对应ELF字节，确认正式review的288条静态`v_wmma_scale_f32_32x16x128_f4`均为`0x80`。该对象SHA256为`5f023d44fb65f0a38a8c22907775afeab049d92a0bad0698fec831901eb486ff`。本地LLVM `VOP3PInstructions.td`中`VOP3PX2e`把该字段设为`?`，调用`postEncodeVOP3<true,true,false>`；`AMDGPUMCCodeEmitter.cpp`将通用unused source填为inline zero `0x80`。这说明实际encoder没有按该Table固定值编码，尚不构成性能收益证据。

当前[进程内probe](../.codex_tmp/gemm1_t1536_20261017/wmma_constant_probe.py) 仅针对review的M256 fused specialization，在code load前修改上述固定字段为`0x100`。原完整kernel定义、code size、operand、metadata、K顺序、控制流和等待保持原样，逐byte断言修改仅落在该9-bit字段。它不修改LLVM、FlyDSL安装或disk cache；性能阶段临时使用同一完整review定义和私有wrapper，退出后恢复正式review。该probe已完成CPU、完整equivalence/replay及原命令筛选，三轮未确认稳定收益，不合入。

证据：[真实ELF字段报告](../.codex_tmp/gemm1_t1536_20261017/logs/review_wmma_constant_elf_audit.json)、[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/wmma_constant_probe_source_contract.json)、[v66 generator](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v66.py)。

### v66 完整验证与性能对比：没有确认稳定收益

首版CPU probe预期一个`bin` attribute，实际同一ELF同时出现在两个target attributes；核对内容相同后同时修改。完整原ELF为32888 B，288条静态WMMA的固定字段各改变一个byte，共288 bytes；所有其余ELF字节完全相同，包括metadata、operand、控制流和等待。patch后的ELF SHA256为`516a4bcd5b1dbbfe03c5331fd6efd5f9c07e83f05832bad830669fb9796184ec`。VGPR470/SGPR106/LDS278528 B，private和全部spills为0。

完整11 cases equivalence的canonical payload、ScaleA、GEMM2 valid rows和MoE output均为bitwise mismatch=0；动态输入、poison、eager/graph replay也全部通过。私有candidate源码SHA256为`4f682c346e1ceea93ee0feea8daefbf1b2429a422ed610dc7a44f31f9fea92c9`，实际验证的helper SHA256为`1dd460a99996e78d2b69a820104c03aeaf0c70ebe7d4be4c76621ecd0d8ca57c`。二者必须同时核对，不能只用candidate SHA标识实验版本。

最初将candidate临时复制到正式module路径后，relative helper目录改变，benchmark报`ModuleNotFoundError: No module named 'wmma_constant_probe'`。该次没有有效性能结果。后续review和probe都加同一个私有task目录到`PYTHONPATH`，执行原benchmark及全部原计算，不修改脚本、precision gate或计时。筛选分别为review96.735 / probe94.475 μs，count=19、原sanity通过。三轮交替结果如下，第二轮逆序；它们不替代正式中值：

| 轮次 | review（μs） | probe（μs） | review−probe（μs） | 对比限制 |
|---:|---:|---:|---:|---|
| 1 | 94.806 | 93.562 | +1.244 | review结束时其他E65验证进程已进入；不用于严格性能结论 |
| 2，逆序 | 95.371 | 94.494 | +0.877 | 前置idle门禁通过 |
| 3 | 94.595 | 95.125 | −0.530 | 前置idle门禁通过，probe回退 |

三轮原sanity均符合原`logits_diff<0.01` gate。第一轮review的reference hash与其它样本不同，`logits_diff=1.0762e-04`；生产输出hash相同，但post-run发现其他GPU进程，不能将该轮用于bitwise等价或严格性能判断。其余运行sanity为`3.4733e-06`。现有剩余两轮方向不一致，没有稳定收益，不能根据最低单次声称字段修正加速。它证明了encoder与文档固定值不一致和probe在完整验证中的数值等价，未证明该字段是当前80 μs目标的主要瓶颈。

```bash
# On host; helper SHA and full numerical validation logs are hard gates.
bash .codex_tmp/gemm1_t1536_20261017/time_validated_wmma_v66_host.sh
bash .codex_tmp/gemm1_t1536_20261017/pair_wmma_v66_host.sh
```

controller于**2026-10-17 08:48:23 UTC**退出并恢复正式review SHA。完整验证/筛选/三轮日志、CPU inspection及patch前后ELF已下载本地。证据：[v2 controller与导入失败](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma_constant_v66_v2_20261017_controller.log)、[equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma_constant_v66_v2_20261017_cyclic64_mg8_wmma_constant_v100_probe_equivalence.log)、[replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma_constant_v66_v2_20261017_cyclic64_mg8_wmma_constant_v100_probe_replay.log)、[成功筛选controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma_constant_v66_v3_20261017_controller.log)、[三轮controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma_constant_v66_pair_20261017_controller.log)、[ELF逐byte报告](../.codex_tmp/gemm1_t1536_20261017/logs/wmma_constant_516a4bcd5b1dbbfe.json)、[CPU resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_wmma_constant_v2_inspection.log)。

## M32 pair stripe 与 M64/N128：完整精度通过，未取得筛选收益

v67 将原两个 M128 compute halves 改为交错的完整 M32 pairs，reader 访问原 physical cyclic owners。每 wave 的一次 output TDM 改为 `[4,2,256]`，保留 canonical payload/ScaleA。28,672项地址核对通过，input TDM、ring、READY/REUSE、task fences、完整 padded M/N/K 和每 accumulator 的 K128 累加顺序保持。candidate SHA256为`e051698c53922575b86e4720b32d27140073a33dac51a6f37e24229f7304654f`。

v68 保持完整 M256/N256/K7168 和八个 compute waves，将内部 partition 改为 M64/N128（`w4x2`）。每 wave 仍有256个F32 accumulator、每K128仍读取30个原模型LDS operands；physical A/ScaleA owners不变。每 wave 的两组 per32 quant blocks 通过一次 `[4,512]` output TDM 写入原外部布局。candidate SHA256为`14c06dfd6761b0be8a6627e945fd5cc32ea879e0109ef60553f6d16fb4cefd43`。

| 候选 | 完整 equivalence / replay | 原命令筛选 GEMM1（μs） | VGPR / SGPR | LDS（B） | 样本限制 |
|---|---|---:|---|---:|---|
| `cyclic64_mg8_m32_pair_stripe` | 通过 / 通过 | 98.125 | 473 / 106 | 278528 | post-run发现其他E65验证进程，不能作为严格空闲性能数据 |
| `cyclic64_mg8_m64n128` | 通过 / 通过 | 98.865 | 479 / 106 | 278528 | post-run busy=3，空闲检查未通过，不能作为正式性能数据 |

两项CPU private、VGPR/SGPR spills均为0。完整11 cases 的canonical payload、ScaleA、GEMM2 valid rows、MoE output，以及动态输入/poison/eager/graph replay均bitwise mismatch=0。v68原sanity=`logits_diff=3.4733e-06`、count=19；不能把受限筛选样本与94.888 μs正式中值相减来声称确定回退幅度。两项没有显示进入三轮复测的收益，未采用。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v67.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v68.py
RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_m32_stripe_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_m32_pair_stripe
RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_m64n128_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_m64n128
```

v67/v68 controller分别于08:59:19 / 09:05:54 UTC完成并恢复review。09:09:49 UTC再次核对boot、branch、HEAD、正式SHA以及全部GPU/KFD，均符合本批环境，没有本任务测试残留。证据：[v67 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_m32_pair_stripe_v67_20261017_controller.log)、[v68 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_m64n128_v68_20261017_controller.log)、[M32 source contract](../.codex_tmp/gemm1_t1536_20261017/logs/m32_pair_stripe_source_contract.json)、[M64/N128 source contract](../.codex_tmp/gemm1_t1536_20261017/logs/m64n128_source_contract.json)、[v68 equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_m64n128_v68_20261017_cyclic64_mg8_m64n128_equivalence.log)、[v68 replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_m64n128_v68_20261017_cyclic64_mg8_m64n128_replay.log)、[v68 benchmark及后置检查](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_m64n128_v68_20261017_cyclic64_mg8_m64n128_bench.log)。逐项CPU和GPU日志已下载本地。

## 单个 shared steady loop：CPU 检出 SGPR spill，未进入 GPU 测试

v69 保持 `issue`、reader、WMMA、`compute_ktile`、fence和quant/output函数不变，只在 `M256_FUSED_LAYOUT` 将四组producer的steady循环合并。`issue()`在循环内选择原单个owner job，每 wave 请求数和tensorcnt距离均保持。candidate SHA256为`512f7d9563df6960e10b389eff975e979b1cdba37d98fddc419a08c0c4373983`。

CPU静态ISA由review的288条WMMA降到192条，资源为VGPR481（allocated496）、SGPR107、LDS278528 B、private0/VGPR spill0，但**SGPR spill34**，出现34条`v_writelane_b32`和56条`v_readlane_b32`。这些是静态互斥路径的合并，不能解释为减少实际矩阵计算；每个动态wave仍须完成原全部K128贡献。新增spill使它未通过zero-spill门禁，未运行GPU correctness/benchmark，不合入。

Shader Programming Guide §5.3.2（printed p.218–219）说明GPR banking及source-cache在memory-write-hit、EXEC改变等情况下失效；§5.7.6（printed p.235–236）说明SIMD-pair共享LDS发射以及B128依赖延迟。已有ATT显示两个同SIMD slot的steady到达差，支持检查producer分支调度，但文档和trace均未保证共享循环一定提速。v70进一步测试两组producer共享循环以限制descriptor状态存活，CPU结果见下一节；未启动GPU测试。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v69.py
RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_shared_steady_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_shared_steady_loop
```

controller于09:37:32 UTC完成；live review未被替换。证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/shared_steady_loop_source_contract.json)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_shared_steady_v69_20261017_controller.log)、[CPU ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_shared_steady_loop_inspection.log)。

## 两组 producer 共享循环：CPU 仍有 spill，未进入 GPU 测试

v70将四组producer分别按data/scale或A/B配成两对。每wave只发出原属于自己的request，两个steady实例保留原完整K128计算、reader与fences。两项target资源均为VGPR469（allocated480）、SGPR107、LDS278528 B、private/VGPR spill0，但各有**8个SGPR spill**。静态WMMA为224条，静态指令分别为4136/4179；静态数量包含互斥控制流，不能当成动态计算减少。

| 候选 | source SHA256 | SGPR spill | 状态 |
|---|---|---:|---|
| `cyclic64_mg8_shared_pairs_data_scale` | `5e19e73f11ba237d8cc648b8db2d8a009bd7ef1838ad8256abe88668fdcdd943` | 8 | CPU gate淘汰 |
| `cyclic64_mg8_shared_pairs_a_b` | `ed97233ac99f81e262ea8e128818f4cfe0745befbe495d97333528c5df5553a7` | 8 | CPU gate淘汰 |

两项没有GPU正确性或性能数据，也没有合入。controller于09:44:17 UTC完成。复现采用新batch名，每个容器内compile/inspection命令均限时120秒：

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v70.py
RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_shared_pairs_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_shared_pairs_data_scale cyclic64_mg8_shared_pairs_a_b
```

证据：[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_shared_pairs_v70_20261017_controller.log)、[data/scale CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_shared_pairs_data_scale_inspection.log)、[A/B CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_shared_pairs_a_b_inspection.log)。

## Per-wave descriptor 预构造：完整验收通过，三轮没有稳定收益

v71在`M256_FUSED_LAYOUT`的steady入口先选出每wave唯一的原producer job，再构造其原rank-2 TDM descriptor。共享循环只更新global/LDS地址。GROUP1的`INT32_MAX` no-clamp、OOB zero fill、source pitch、cyclic owner placement、multicast early_timeout和B-only caller cache policy均保留；global地址使用carry-safe i64。prologue、drain、跨N-task input/output交接、WMMA、SiLU、BF16 rounding、native per32 E8M0/FP4和外部布局保持原逻辑。

candidate source SHA256为`acc8abaeb6e524c8db51413651533cb42d4f2e928cf5feebbe0e5113395f445e`。source contract核对8736项descriptor/地址，关键reader/WMMA/fence/quant/output函数AST相同。CPU target为VGPR485（allocated496）、SGPR85、LDS278528 B，private/VGPR/SGPR spills均0；3976条静态instructions、192条静态WMMA，text SHA256为`b14b0552cdef22492e42e544bc99635ceda5106486102afe1c753cbbfa788b92`。代码实例变少、SGPR降低没有改变动态计算量，也不保证性能收益。

完整11 cases的canonical payload、ScaleA、GEMM2 valid rows和MoE output均bitwise mismatch=0；覆盖random seeds0/17/43、skew、sparse、hot8、tiny expert、balanced及T1537/T1792/T2048。动态input、poison和eager/graph replay均通过。十组fallback完整ELF相同；target descriptor只改变instruction prefetch长度222→193及allocated VGPR480→496，metadata仅VGPR/SGPR count变化，ABI/geometry/LDS/private/spill/control逐字段审核通过。

原命令筛选为94.507 μs，立即post-run busy=3，不能单独作为新收益。随后三轮交替使用原benchmark和count=19，第二轮逆序：

| 轮次 | mg8 review（μs） | prepared steady（μs） | review−candidate（μs） |
|---:|---:|---:|---:|
| 1 | 95.171 | 94.747 | +0.424 |
| 2，逆序 | 95.165 | 95.034 | +0.131 |
| 3 | 94.479 | 95.217 | −0.738 |
| **中值** | **95.165** | **95.034** | **+0.131** |

三轮原sanity均为`logits_diff=3.4733e-06`、`rel_l2=2.6357e-03`，与原精度门槛一致。每次前置全机idle通过；立即post-run busy=2/3、KFD无进程且VRAM恢复到176582656 B，6秒后的独立采样均busy=0。两次后置采样原文均保留；不能仅凭这组采样断言延迟的全部原因。第三轮候选回退，中值差只有0.131 μs，没有确认稳定提升，**未合入**。pair于09:58:27 UTC恢复review，fallback controller于10:01:41 UTC完成。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v71.py
RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_prepared_steady_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_prepared_steady
# The pair script also requires the exact completed numerical-validation logs.
PAIR_BATCH=<new_prepared_steady_pair_batch> \
bash .codex_tmp/gemm1_t1536_20261017/pair_v71_verified_host.sh
bash .codex_tmp/gemm1_t1536_20261017/compile_v71_scope_host.sh
```

证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/prepared_steady_source_contract.json)、[CPU ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_prepared_steady_inspection.log)、[equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_prepared_steady_v71_20261017_cyclic64_mg8_prepared_steady_equivalence.log)、[replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_prepared_steady_v71_20261017_cyclic64_mg8_prepared_steady_replay.log)、[三轮controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_prepared_steady_v71_pair_20261017_controller.log)、[十组fallback及target审核](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_mg8_prepared_steady.json)。全部逐轮benchmark与后置日志已下载本地。

## Bounded shift 地址与循环外 B policy：完整精度通过，未确认收益

v72基于v71的per-wave descriptor，只把有界relative LDS/global offset的动态multiply改为shift；pitch与K advance均为2的幂。global pointer的加法仍为i64，保留跨4 GiB carry。504项地址枚举通过，reader、WMMA、fences和output函数AST不变；完整input bytes、K128顺序和外部布局保留。

| 候选 | CPU VGPR / SGPR | private（B） / VGPR spill | 完整 equivalence / replay | 原命令筛选 GEMM1（μs） |
|---|---|---|---|---:|
| `cyclic64_mg8_prepared_shifted` | 485 / 83 | 0 / 0 | 全部 bitwise mismatch=0 | 94.621 |
| `cyclic64_mg8_prepared_b_hoisted` | 512 / 85 | 1544 / 545 | CPU门禁淘汰，未运行 | — |
| `cyclic64_mg8_prepared_shifted_b_hoisted` | 512 / 83 | 1544 / 545 | CPU门禁淘汰，未运行 | — |

三个版本LDS278528 B、SGPR spill0。shifted的source SHA256为`c6356780e62f50493503e6d147ab133352cb76c793af9e0cf48ba8cbf7f9b15f`，text SHA256为`1e427c5a0554a596aef216bafc113378541d934d87c4ddb5a7e664a1cd71ffc5`，静态3978 instructions。完整11 cases及动态输入、poison、eager/graph replay均通过，原sanity=`logits_diff=3.4733e-06`、count=19。前置全机idle通过，立即post-run busy=3、6秒后KFD无进程/busy=0/VRAM=176582656 B；两个原始采样均保留。只有单次筛选，未做三轮和扩展fallback验收，不能宣布稳定提升，未合入。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v72.py
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_address_policy_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_prepared_shifted cyclic64_mg8_prepared_b_hoisted \
  cyclic64_mg8_prepared_shifted_b_hoisted
```

controller于10:15:07 UTC完成恢复review。证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/prepared_address_policy_v72_source_contract.json)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_prepared_policy_v72_20261017_controller.log)、[shifted CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_prepared_shifted_inspection.log)、[benchmark及立即后置采样](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_prepared_policy_v72_20261017_cyclic64_mg8_prepared_shifted_bench.log)、[6秒后采样](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_prepared_policy_v72_20261017_cyclic64_mg8_prepared_shifted_bench_post.log)。

## Production TH6 的 B cache policy：实际 ISA 已确认，未取得可靠筛选收益

早期cache实验的guard只改变`tdm_b_th==0`，而当前production传6；因此v73在`M256_FUSED_LAYOUT && tdm_b_th==6`范围实际测试0/1/2。CDNA5 ISA §4.1.1 Table10（printed p.36）将0/1/2/6分别定义为RT/NT/HT/NT_HT；这只是retention优先级提示，文档没有保证某个hint对本workload更快。

只读ISA audit对照正式review逐条核对instruction地址、长度和machine words。每个对象的40条静态TDM load中，只有10条B load的VIMAGE `TH[54:52]`由6改为对应hint；所有其他instruction bytes相同。三个对象均为VGPR470/SGPR106/LDS278528 B，private及VGPR/SGPR spills全0；静态4381 instructions、288 WMMA。不能把静态数量解释为动态计算减少。

| B hint | source SHA256 | 完整 equivalence / replay | 原命令筛选 GEMM1（μs） | 后置采样限制 |
|---|---|---|---:|---|
| TH0 / RT | `b2af1ea907a22cf67a5d3c169d10dba097cb76b958b6869f5d52b1a4d83767c8` | 全部 bitwise mismatch=0 | 94.956 | 立即busy=3，6秒后idle |
| TH1 / NT | `7805572fa2ce326124ec928342853658df53a5b573038dfe057e33d9ac7f4bef` | 全部 bitwise mismatch=0 | 94.627 | 立即busy=3；6秒后发现其他E65任务，不能用于严格性能结论 |
| TH2 / HT | `e1b26698b519c3b3bb901bea5985a52506218f268f0e9439507eeded6a7747f5` | 全部 bitwise mismatch=0 | 95.804 | 立即busy=3，6秒后idle |

完整11 cases覆盖random/skew/sparse/hot8/tiny/balanced及T1537/T1792/T2048，canonical payload、ScaleA、GEMM2 valid rows、MoE output均bitwise mismatch=0；动态输入/poison/eager/graph replay全部通过。三个benchmark的前置idle均通过，原sanity=`3.4733e-06`、count=19。TH1后置发现host PID3807824的其他E65/topk9/shared-expert任务，等待6秒后恢复idle，未终止其他进程。全部数据只用于本批筛选；没有明确收益，不做三轮、不合入，正式94.888 μs中值保持。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v73.py
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_production_cache_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_b_from6_th0 cyclic64_mg8_b_from6_th1 cyclic64_mg8_b_from6_th2
python3 .codex_tmp/gemm1_t1536_20261017/audit_b_cache_v73.py
```

controller于10:27:36 UTC完成恢复review；10:28:48 UTC正式source SHA复核一致，10:28:54 UTC全机idle，无本任务残留。证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/production_b_cache_v73_source_contract.json)、[真实ISA逐指令核对](../.codex_tmp/gemm1_t1536_20261017/logs/production_b_cache_v73_emitted_isa_audit.json)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_production_b_cache_v73_20261017_controller.log)、[TH1 benchmark](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_production_b_cache_v73_20261017_cyclic64_mg8_b_from6_th1_bench.log)、[TH1后置采样](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_production_b_cache_v73_20261017_cyclic64_mg8_b_from6_th1_bench_post.log)、[最终环境检查](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_production_b_cache_v73_final_environment_20261017.log)。三个候选逐项CPU、equivalence/replay/bench/post日志已下载本地。

## 相邻 K256 成对 rendezvous：展开版本 CPU spill 淘汰

v74保留原四级cyclic ring与K256 TDM请求，把相邻两个K256成对计算；第二个step的closing fence先完成两个current reader和两个next input，再refill已释放的两个slots。所有56个K128 contribution和每accumulator顺序相同，原drain和跨N-task output等待保留。1734项地址/顺序检查通过，但静态模型不能代替GPU正确性。

candidate=`cyclic64_mg8_paired_k256`，source SHA256=`6068a6d798208f10219a307981cfd5fee84cebaabb4769335db92d78219bdb14`。CPU编译成功，VGPR512/SGPR106/LDS278528 B，**private1360 B/VGPR spill2877**，SGPR spill0。静态8097 instructions/544 WMMA，出现scratch load/store；这不能视为动态计算变化。由于zero-spill门禁拒绝，未运行GPU correctness、replay或benchmark，也未替换正式源码。v75随后用单个rolled body保留相同pair协议并完成验证，结果见下一节。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v74.py
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_paired_k256_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_paired_k256
```

v74 controller于10:35:18 UTC结束。证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/paired_k256_v74_source_contract.json)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_paired_k256_v74_20261017_controller.log)、[CPU ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_paired_k256_inspection.log)。

## 单个 rolled body 的 K256 pair：完整精度通过，筛选未改善

v75把v74两次完整compute展开改为原单个rolled K256 body，只在odd kt执行pair的ready/reader fence及两次refill；even kt从已完成的current pair读取，不写input slots。每N-task的steady rendezvous为12次，原四级ring、全部28个K256 TDM和56个K128 accumulation顺序保持。最后四个drain step及output交接逻辑不变。CPU移除了v74的spill，但同步更少不保证整个pipeline更快。

candidate=`cyclic64_mg8_paired_k256_rolled`，source SHA256=`de148edfc064846c904119d9a464df6ea488cb2f43cfaca181ff907f5057c8b9`；CPU VGPR485/SGPR106/LDS278528 B，private及spills全0，text SHA256=`9e5dca65123633883e03a201e608e2d509d03a0ca6cebe1ceeb2b85223079b98`，静态4487 instructions/288 WMMA。完整11 cases equivalence和动态input/poison/eager/graph replay全部bitwise mismatch=0，原benchmark count19、sanity=`3.4733e-06`。

单次筛选为**100.495 μs**，未显示进入三轮的收益。前置全机idle通过；立即post-run busy3/无KFD，6秒后发现另一任务的E65 CPU compiler进程在KFD列表中，继续按6秒等待，10:41:42 UTC恢复idle。由于后置占用限制，该样本不列为严格空闲性能成绩，不能用它计算确定的回退幅度。没有新ATT，不能将该耗时定量归因于pair lookahead、branch或某一种wait。不合入，正式review已恢复。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v75.py
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_paired_rolled_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_paired_k256_rolled
```

证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/paired_k256_rolled_v75_source_contract.json)、[CPU ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_paired_k256_rolled_inspection.log)、[完整controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_paired_k256_rolled_v75_20261017_controller.log)、[benchmark及立即post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_paired_k256_rolled_v75_20261017_cyclic64_mg8_paired_k256_rolled_bench.log)、[6秒后post及占用等待](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_paired_k256_rolled_v75_20261017_cyclic64_mg8_paired_k256_rolled_bench_post.log)。完整equivalence/replay原始日志已下载本地。

## M128/b2 单 register slot：完整精度通过，未取得筛选收益

v76保留原256行expert alignment与完整padded M/N/K，把每个原M256 task拆成两个M128 WG，每WG仍为八个waves、M64/N64 compute partition、四个N-task。parent expert用`blk_m & -256`选择，读写采用实际half地址，valid extent clamp到0。两级linear data+scale ring、独立output LDS和单个K128 register slot降低资源，但下一task必须先等旧output完成，不做跨task input overlap。514项地址与完整计算覆盖检查通过。

candidate=`linear64_m128n256_b2_rmem1`，source SHA256=`8f097043d289048fe4cf9b8ac9db1e8989f725107598ec7b6443a943ff77ac59`，text SHA256=`c30de40fc72b58dc7d34aa8c01d63b1eb521531b9acf115075d072eea4c3d141`。CPU VGPR227（allocated240）/SGPR105/LDS113152 B，private及spills全0；静态1997 instructions/96 WMMA。单WG资源降低不代表整个workload计算或请求减少，也不能直接证明实际occupancy与性能。

完整11 cases equivalence和动态input/poison/eager/graph replay全部bitwise mismatch=0。原命令单次筛选为**172.590 μs**，count19、sanity=`3.4733e-06`，未显示收益，因此不做三轮、不合入。前置等待期间有其他E65任务，均等其退出后才开始计时；立即post-run busy3/KFD无进程/VRAM176582656 B，6秒后idle。两次采样均保留，没有新ATT，不能将全部增加耗时定量归因于WG数、input lead或某个wait。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v76.py
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=256 SCREEN_BATCH=<new_m128_b2_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  linear64_m128n256_b2_rmem1
```

controller于10:54:36 UTC恢复review。证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/m128_b2_v76_source_contract.json)、[CPU ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_linear64_m128n256_b2_rmem1_inspection.log)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_m128_b2_v76_20261017_controller.log)、[benchmark及立即post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_m128_b2_v76_20261017_linear64_m128n256_b2_rmem1_bench.log)、[6秒后post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_m128_b2_v76_20261017_linear64_m128n256_b2_rmem1_bench_post.log)。完整equivalence/replay与CPU日志已下载。

## 当前 review 的 K512/b2：安全交接验证通过，未取得筛选收益

v77从当前正式review派生，只在原M256 fused specialization将内部搬运粒度改为K512、两个linear input stages。全部56个K128和每accumulator顺序保持，原`mma_rows`、`k_step`、`compute_ktile`等函数AST不变。data `[0,262144)`、scale `[262144,278528)`、output `[278528,295936)` 完全分离；672项完整request byte区间检查通过。最后K512有四个K128，第一块carry不能释放整块输入，因此next-task I0/I1移到完整final compute及DS/WG reader fence之后。后续task的owner顺序为I0,I1,O，`tensor_wait(1)`使两个input可读并允许disjoint output继续执行。

candidate=`linear128_k512_b2_review`，source SHA256=`349b8bbd00a149636ba9d168d1305e9a9c851cb0f2f34006a65ff959d255a725`，text SHA256=`2ce2b9bcf8ad7e662f4d610309068864f5de93832bb7c78e07de711e068b36e8`。CPU VGPR469（allocated480）/SGPR102/LDS295936 B，private与spills全0，静态4107 instructions/384 WMMA。K512减少request次数但保留完整输入bytes；静态compute实例变化不能解释为动态计算减少。

完整11 cases equivalence通过，canonical payload、ScaleA、GEMM2 valid rows、MoE output均bitwise mismatch=0；动态input/poison/eager/graph replay全部通过。原命令单次筛选为**100.145 μs**，count19、sanity=`3.4733e-06`。前置全机idle通过，立即post-run busy2/KFD无进程/VRAM176582656 B，6秒后idle，原始采样均保留。没有显示进入三轮的收益，不合入。此候选相较历史K512草稿保留安全的跨task input/output overlap，但仍未改善当前口径；没有新ATT，不能定量分摊input lead、linear布局或final fence的影响。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v77.py
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_k512_b2_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  linear128_k512_b2_review
```

controller于11:05:37 UTC完成并恢复正式review，随后正式SHA复核一致。证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/k512_b2_v77_source_contract.json)、[CPU ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_linear128_k512_b2_review_inspection.log)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k512_b2_v77_20261017_controller.log)、[完整equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k512_b2_v77_20261017_linear128_k512_b2_review_equivalence.log)、[replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k512_b2_v77_20261017_linear128_k512_b2_review_replay.log)、[benchmark及立即post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k512_b2_v77_20261017_linear128_k512_b2_review_bench.log)、[6秒后post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k512_b2_v77_20261017_linear128_k512_b2_review_bench_post.log)。全部日志已下载本地。

## B/ScaleB K2/K3 提前交接：完整精度通过，三轮未取得收益

借鉴历史pipeline记录中B/ScaleB-only successor handoff，但以当前M256 review重新生成，不能引用历史balanced/const0的耗时作为当前成绩。K2/K3 B目标分别为`[147456,180224)` / `[212992,245760)`，ScaleB为`[272384,274432)` / `[276480,278528)`；均与output `[245760,263168)` 不重叠。当前cluster_m1下B/ScaleB没有multicast mask，属于WG-private搬运。每候选312项区间、per-wave completion距离与完整request顺序检查通过；原TDM descriptors、reader、WMMA、K128顺序、activation/quant/output函数保持。

`cyclic64_mg8_weight_handoff_early_v78`在final reader WG fence之后、quant之前发下一task B2/B3及ScaleB2/ScaleB3，再发O；后续entry先保留原wait并将深层weights等完，K2/K3只补A/ScaleA。该版本增加每非final task一次reader rendezvous。

`cyclic64_mg8_weight_handoff_queued_v78`复用原output-publication rendezvous，顺序为I0,I1,O,W2,W3，不增加reader barrier。weight owners的entry `tensor_wait(3)`完成I0/I1，K0之后`tensor_wait(2)`完成O，A/ScaleA owners分别保留1/0；随后补A2/A3并恢复原steady completion距离。它避免将已释放weight区域和output arena混为同一依赖，但性能收益仍需实测。

两项先通过CPU zero-private/zero-spill gate，随后完整11 cases equivalence和全部dynamic/poison/eager/graph replay的canonical payload、ScaleA、GEMM2 valid rows及MoE output均bitwise mismatch=0。硬件依据：Shader Programming Guide §4.10.8（printed p.206）的3 tensor ops/wave、6/SIMD是issue-to-XACK限制，tensorcnt completion上限为63；§4.10.1及CDNA5 ISA §10.11.1（printed p.140）规定同wave TDM完成有序、跨wave无序。因此仍保留per-owner completion和WG reader/visibility fences，不能由issue数量预设收益。

| 候选 | source SHA256 | VGPR / SGPR | LDS（B） | 原命令筛选 GEMM1（μs） | 样本限制 |
|---|---|---|---:|---:|---|
| `cyclic64_mg8_weight_handoff_early_v78` | `561c51497fb20e05fca60de730d88220e51d64a454f04e61ac8d61dc7d42309f` | 470 / 106 | 278528 | 94.260 | 单次筛选，后续三轮未确认收益 |
| `cyclic64_mg8_weight_handoff_queued_v78` | `f7733154c75e3d0557be6f7f011e342d21587daa71fa767feee9f7eae1008da0` | 467 / 106 | 278528 | 95.723 | 后置发现其他E65任务，不用于严格性能结论 |

early的text SHA256为`da6bd96a7ccffa66ae7c4924f74a4ba8da1e139450178a6795c83de12c20c363`；queued为`fdfab65334367f91828f2c0d16cf3552ffff424eafbe476ca8510917fe13e52e`。early因筛选有小幅迹象而进入三轮交替，第2轮逆序：

| 轮次 | review（μs） | early（μs） |
|---:|---:|---:|
| 1 | 94.180 | 94.688 |
| 2，逆序 | 94.747 | 94.926 |
| 3 | 94.452 | 94.937 |
| **中值** | **94.452** | **94.926** |

三轮候选均慢，中值增加0.474 μs，未取得稳定收益，不合入。六次原benchmark的count均为19、sanity均为`logits_diff=3.4733e-06`，六次前置全机idle以及六次6秒delayed post均通过；立即post原始采样仍随各benchmark日志保留。不能由历史weight-handoff收益推导当前M256也会提速。screen于11:23:03 UTC、pair于11:30:42 UTC结束并恢复正式review。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v78.py
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_weight_handoff_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_weight_handoff_early_v78 cyclic64_mg8_weight_handoff_queued_v78
# Pair requires the exact successful equivalence/replay logs and source hashes.
PAIR_BATCH=<new_weight_handoff_pair_batch> \
bash .codex_tmp/gemm1_t1536_20261017/pair_v78_verified_host.sh
```

证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/weight_handoff_v78_source_contract.json)、[screen controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_weight_handoff_v78_20261017_controller.log)、[三轮controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_weight_handoff_early_v78_pair_20261017_controller.log)、[early CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_weight_handoff_early_v78_inspection.log)、[queued CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_weight_handoff_queued_v78_inspection.log)。两项完整CPU/GPU日志、六份pair benchmark及六份post日志已下载本地。

## Column-snake WMMA：完整精度通过，未取得筛选收益

v79仅将原M256 `mma_rows`改为在同一个K128内按weight column遍历独立M accumulator。FRONT/BACK调用、全部16个WMMA/K128、每accumulator的56个K128顺序、reader/TDM/fence/quant/output均保留；112项完整step顺序检查通过。`nohint`关闭显式reuse；`reuse`将与前一条相同的weight/activation operand标为reuse。后续重读CDNA5 ISA §7.12（printed p.107）确认它描述的是当前matrix将在**下一条**指令被复用，因此v79的backward-looking hint不能称为符合该文档的forward-looking设置。v80以新名字单独验证修正，不覆盖已有实验。

| 候选 | source SHA256 | VGPR / SGPR | LDS（B） | 原命令筛选 GEMM1（μs） |
|---|---|---|---:|---:|
| `cyclic64_mg8_column_snake_nohint_v79` | `b77751e01e992175e78dc09940ffd1e48cc438db9f60ca38da3371731e2a114c` | 506 / 106 | 278528 | 97.142 |
| `cyclic64_mg8_column_snake_reuse_v79` | `d7e9761370e009e13417567be2ab87ae5a04bd92b413ab7dc8c3b5fb63b10bed` | 506 / 106 | 278528 | 96.307 |

两项CPU private/VGPR spill/SGPR spill全0，allocated VGPR512；static 4422 instructions / 288 WMMA。text SHA256分别为`080547f71086223128f6fe6566ed02165a93ba02a1cbe888d9456d99f73e71db` / `dc16382d882fe5180718a30e628a35d094a5556eda4a96ebd91166d5363fd832`。完整11 cases equivalence和全部dynamic/poison/eager/graph replay均bitwise mismatch=0；原sanity=`3.4733e-06`、count19。

两次计时前全机idle通过，但后置分别发现其他E65验证/benchmark任务，故两项不列为严格空闲性能数据；没有显示进入三轮的收益，不合入。不能将增加的VGPR或hint方向单独归因为确定的耗时差。controller于11:37:36 UTC完成恢复review，11:40:38–39 UTC重新核对boot、branch、HEAD、容器和全机idle；正式source SHA仍为`b36f7ec...`。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v79.py
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_column_snake_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_column_snake_nohint_v79 cyclic64_mg8_column_snake_reuse_v79
```

证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/column_snake_v79_source_contract.json)、[完整controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_column_snake_v79_20261017_controller.log)、[nohint CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_column_snake_nohint_v79_inspection.log)、[reuse CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_column_snake_reuse_v79_inspection.log)。逐项CPU compilation、equivalence/replay/bench/post原始日志已下载本地。

## Forward reuse 与 K128/K256 priority rotation：完整精度及三轮复测

v80 从 review 保留完整 WMMA 遍历和累加次序，只在 `M256_FUSED_LAYOUT` 改 reuse attributes。row 版本将当前 operand 在后续 source WMMA 中仍相同时设置 hint；column 版本基于 v79 column-snake，并使用相同的 forward 判断。v81 则从 review 保留原 WMMA 和依赖，只对目标路径设置初始 `USER_PRIO`：wave0–3 为 2，wave4–7 为 0，在已有 K128 或 K256 scheduler boundary 发 `s_setprio_inc_wg 2`，kernel 结束前恢复 priority0。

| 候选 | source SHA256 | 首次原命令筛选（μs） |
|---|---|---:|
| `cyclic64_mg8_row_forward_reuse_v80` | `ecea392e55075612df76e44d65d374fe5bdf0ea6bbaf09b5ca0ddb13b1619654` | 93.773 |
| `cyclic64_mg8_column_forward_reuse_v80` | `282fc1214888449557c4c04cc970392ed357d86f47ae6d06192008c60760b724` | 96.371 |
| `cyclic64_mg8_rotate_k128_v81` | `cf61c59171a558d388c461b06f9b7eac6bdba284a8fe46594b0070763e0102bf` | 93.826 |
| `cyclic64_mg8_rotate_k256_v81` | `adb0430572ffdb4eca9f683192d84613ef8117d2a57b888e48219994349dc84f` | 94.355 |

四项 CPU private/spills 均为 0，完整 11 cases equivalence 和动态输入、poison、eager/graph replay 全部 bitwise mismatch=0。11 cases 包括三个 T1536 random seeds，以及 skew、sparse、hot8、tiny expert、balanced、T1537、T1792、T2048。row reuse、两项 rotation 的 VGPR470/SGPR106/LDS278528 与 review 相同；column 为 VGPR506/SGPR106/LDS278528。十组 fallback 验收尚未对这四项补做。

随后仅对 row reuse 与 K128 rotation 做三轮原命令对比，第 2 轮完整逆序。9 次前置全机 idle、9 次 6 秒后的后置 idle 均通过；日志保留立即 post 的退出 PID/短暂 busy 原文，另存 delayed post，没有改写原始记录。所有样本 `count=19`、原 sanity `logits_diff=3.4733e-06`。

| 轮次 | review（μs） | K128 rotation（μs） | row forward reuse（μs） |
|---:|---:|---:|---:|
| 1 | 95.163 | 94.690 | 95.038 |
| 2，逆序 | 94.743 | 95.336 | 94.387 |
| 3 | 94.954 | 94.224 | 94.290 |
| **中值** | **94.954** | **94.690** | **94.387** |

row reuse 三轮分别减少 0.125、0.356、0.664 μs，中值减少 0.567 μs（0.597%）；rotation 第二轮回退 0.593 μs，中值减少 0.264 μs（0.278%），不列为稳定收益。曾在进度说明中误把 reuse 说成有一轮回退，已更正；实际有回退的是 rotation。未另做 row reuse 独立配对，未扩大 K256/column 比较，也没有把筛选最低单次写成正式成绩。

### reuse 的硬件资料冲突与采用条件

本轮重新查阅完整本地文本，发现同一文档存在不一致：

- CDNA5 ISA §7.12，printed p.107，lines8170–8173，VOP3PX2 load-scale 描述为当前 A/B matrix 将在 **next instruction** 复用。
- 同一 §7.12，printed p.97，lines7141–7143，先说 **next instruction**，紧接着又说 `Reuse should only be set to 1 if the current instruction is the same as the previous instruction, otherwise results are undefined.`
- Shader Programming Guide §4.6 对应 lines10415–10417 也有 previous/undefined 限制，而 lines11594–11597 又给出 next 描述。

从现有资料不能直接统一这两种条件。有限 bitwise 测试通过不能替代通用性保证，因此 v80 row reuse 虽有三轮收益，**暂不合入**。不能只引用 p.107 将此事写成已经确定的合法优化，也不能据此断言正式 review 存在精度 bug。

对实际 ISA 的静态 fallthrough 核对发现：review 的 126 个 B hints 全部通过 forward adjacency 核对；row reuse 的 126 个 A hints 全部通过，144 个 B hints 中 141 个通过，3 个跨 barrier 未认证。该检查仅核对相邻静态 WMMA 的寄存器、scale 和可见 clobber，不是完整 CFG/dataflow 或硬件语义证明。column 有更多未认证位置且未显示收益。原始报告没有被重写。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v80.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v81.py
# On host; each CPU/GPU invocation runs in dsr1_ep4 with a 120-second limit.
SCREEN_POST_DELAY_SECONDS=6 RESOURCE_VGPR_MAX=512 SCREEN_BATCH=<new_v80v81_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/resource_gated_candidates_host.sh \
  cyclic64_mg8_rotate_k128_v81 cyclic64_mg8_rotate_k256_v81 \
  cyclic64_mg8_row_forward_reuse_v80 cyclic64_mg8_column_forward_reuse_v80
# This driver binds the archived validation batch and exact source hashes.
# It records a six-second post sample for each arm and reverses round 2.
PAIR_BATCH=<new_v80v81_pair_batch> \
bash .codex_tmp/gemm1_t1536_20261017/pair_v80v81_verified_host.sh
```

证据：[完整筛选](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_forward_reuse_rotation_v80v81_20261017_controller.log)、[三轮 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_forward_reuse_rotation_v80v81_pair_20261017_controller.log)、[逐样本审核](../.codex_tmp/gemm1_t1536_20261017/logs/v80v81_pair_audit.json)、[实际 ISA adjacency](../.codex_tmp/gemm1_t1536_20261017/logs/forward_reuse_emitted_isa_audit.json)、[v80 source contract](../.codex_tmp/gemm1_t1536_20261017/logs/forward_reuse_v80_source_contract.json)、[v81 source contract](../.codex_tmp/gemm1_t1536_20261017/logs/priority_rotation_v81_source_contract.json)。controller SHA256：`2a0d58914d12d0b1883f1cf5e2bf3bf12fe4a2dac69ae98628aedd63835c6117`。

## K128 rotation 四 SIMD ATT：实际 ELF 核对及诊断限制

使用用户指定 `my_code/get_att_b_rocm_sdk.sh`、`--all-simd --ana-att` 抓取完整 GEMM1。driver 对 live source 做 ownership/hash 核对并恢复 review；每个 profiler/workload/analyzer 独立限制 120 秒。v82 是诊断脚本的编号，没有新增 tuning candidate。遇到其他 GPU 任务时按 6 秒等待，未终止其他进程。

capture：`my_code/thread_trace_runs/gemm1_rotate_k128_v81_att_20261017`。四 SIMD status、总 `att_status`、`analyze_status` 均为 0；32 个 valid waves 每个 3584 WMMA，decode 无 `.long`/illegal。2026-10-17 12:26:47 UTC 恢复正式 review，随后全机 idle 通过。

v83 的只读 ELF audit 逐个 SIMD 在实际 code-object 中找到完整 GEMM1 defined symbol，四个 capture 均对应 `out_gfx1250_code_object_id_15.out`，并与 CPU candidate **完整 ELF 逐字节一致**：

```text
ELF SHA256:   08823bd22092b0741acc6cc4a2a9a60d7c81912f1fb76adb6db77e2e8dc09f7a
.text SHA256: 6c2278519ab4ad73702578ec2c4b59dac4fe4dab390ad60413e110831056f40b
VGPR470 / SGPR106 / LDS278528 / private0 / spills0
```

phase stall 归属占各自采样 valid-wave cycles：

| phase / 指令归属 | review ATT | K128 rotation ATT |
|---|---:|---:|
| steady_K / WMMA | 14.5983% | 14.9945% |
| steady_K / `s_wait_dscnt` | 3.0171% | 3.2986% |
| steady_K / `s_wait_tensorcnt` | 3.2554% | 2.6427% |
| steady_K / `barrier_wait` | 10.7250% | 9.8481% |
| task_handoff / `barrier_wait` | 2.8108% | 2.3453% |

| WMMA issue gap | review mean / median（cycles） | rotation mean / median（cycles） |
|---|---:|---:|
| 同一 K128 内 | 16.994 / 8 | 16.972 / 8 |
| K128 boundary | 204.623 / 109 | 187.651 / 112 |
| K256 boundary | 139.843 / 103 | 140.495 / 103 |
| N-task handoff | 4818.823 / 4657.5 | 4551.750 / 4478.5 |

各自 sampled valid-wave cycles 合计为 3,966,657 / 3,873,096。这是不同 capture，clocks 和采样调度不同，不能把上述差换算成 eager 可节省 μs；同一 K128 内的发射均值几乎相同，三轮 eager 也未确认稳定收益，故不采用 rotation。

只在各自同一 SIMD0 capture 内比较 slot0/1，93 个 steady rendezvous 的 slot1−slot0 signal 中值为 review118 / rotation136 cycles，release 差中值均 1 cycle。两 slot 的 barrier stall 合计分别为 19127/8081 与 18273/5875 cycles；rotation 每 slot 确实动态执行 224 次 `s_setprio_inc_wg 2`。它没有缩小这对 sampled peer 的 signal 差，且两条 wave 无法确定整个 WG 的最后到达者。

硬件依据：Shader Programming Guide §5.2.1，printed p.213，`S_SETPRIO_INC_WG` 只改变同 WG、**同 SIMD** 当前已创建 waves 的 USER_PRIO，约 4 cycles 生效，不影响其他 SIMD。§5.3.2 记载 source-cache/bank constraints。文档没有保证本 priority arrangement 一定加速。

```bash
# On host; reusing an existing completed capture directory is rejected.
TRACE_LABEL=<new_rotation_capture_label> \
bash .codex_tmp/gemm1_t1536_20261017/trace_verified_candidate_v82_host.sh
# In dsr1_ep4; read existing objects, without new GPU execution.
timeout --signal=TERM --kill-after=5s 120s \
python3 .codex_tmp/gemm1_t1536_20261017/audit_rotation_trace_elf_v83.py \
  my_code/thread_trace_runs/<new_rotation_capture_label> \
  .codex_tmp/gemm1_t1536_20261017/logs/cpu_compile_cyclic64_mg8_rotate_k128_v81_m256_q8_b0_cm1_pd0_dr0_sq1_e64_th6.lowered.elf \
  .codex_tmp/gemm1_t1536_20261017/logs/<new_elf_audit_name>.json
```

证据：[capture controller](../.codex_tmp/gemm1_t1536_20261017/logs/gemm1_rotate_k128_v81_att_20261017_controller.log)、[phase 报告](../.codex_tmp/gemm1_t1536_20261017/logs/gemm1_rotate_k128_v81_att_20261017_phase_comparison.json)、[slot-pair 报告](../.codex_tmp/gemm1_t1536_20261017/logs/rotate_k128_v81_slot_pair_audit.json)、[实际 ELF audit](../.codex_tmp/gemm1_t1536_20261017/logs/rotate_k128_v81_att_elf_audit_v83.json)。phase wrapper 使用新文件名，没有覆盖旧 `existing_att_phase_comparison.json`。driver SHA256：`ffd02f5425f731ab50ad4eef70f0adc597509ac300d951ff8b81b216baa11ab1`；phase wrapper SHA256：`6f571cb0423371f2a9ad53d16b69279d49e0c892b6ca29c9c32c58c1347e0058`。

## 相邻 M-row BF16 activation 分组：完整精度通过，未取得筛选收益

进一步读源码发现目标 M256 的 `OUTPUT_IN_RELEASED_STAGE` 已启用，没有可独立向后移动的 final reader wait。未对不存在的等待删除同步或声称修复。

v84 仅在原 `M256_FUSED_LAYOUT` epilogue，把 2、4、8 个 M16 row 的相同 activation expression 和 BF16 rounding 先写入 register slots，再逐 row 执行原 per32 E8M0 scale、native FP4 conversion、payload/ScaleA store body。每个 row 保留原 uniform valid guard；所有 K loops、WMMA、TDM、output completion、外部 layout 和其他 signature 保持原逻辑。目标是检查寄存器调度变化能否减少 epilogue 串行成本，收益尚未确定。

[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/quant_rowgroup_v84_source_contract.json) 对每候选验证 1026 个 extent/wave_m guard visitation cases，protected functions AST 不变、还原编辑后与 review 整个文件完全一致。该静态检查不代替 GPU 精度或性能。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v84.py
# On host; per-invocation 120-second limit, whole-machine idle gates for timing.
SCREEN_BATCH=<new_quant_rowgroup_v84_batch> RESOURCE_VGPR_MAX=480 \
bash .codex_tmp/gemm1_t1536_20261017/screen_bounded_v84_host.sh \
  cyclic64_mg8_quant_rowgroup2_v84 cyclic64_mg8_quant_rowgroup4_v84 \
  cyclic64_mg8_quant_rowgroup8_v84
```

controller 只修改通过精度验证的私有候选计时窗口，hash/ownership 检查后恢复正式 review，仅处理本次输出权限；不改 benchmark，也不操作用户的验证脚本。

| 候选 | source SHA256 | VGPR / allocated VGPR | static instructions | 精度 / replay | 原命令筛选（μs） |
|---|---|---|---:|---|---:|
| `cyclic64_mg8_quant_rowgroup2_v84` | `9b024c6c5e39d453448e7030ca2a9524b398e9c41567c43ce18c98ddbc29405a` | 470 / 480 | 4421 | 完整通过 / 完整通过 | 94.922 |
| `cyclic64_mg8_quant_rowgroup4_v84` | `c1eb21fa17f8a0d19b86e8c45a41ef02ec0456ebabe18612df05399466ab9775` | 470 / 480 | 4425 | 完整通过 / 完整通过 | 95.940 |
| `cyclic64_mg8_quant_rowgroup8_v84` | `5b0bd8e6505e26cfa84431ec226c85dfd882c752b606ce3814a95e5ffd03161a` | 482 / 496 | 4490 | 未运行 GPU | 未运行 GPU |

三项 SGPR106/LDS278528/private0/spills0；2/4-row 的全部 11 cases canonical payload、ScaleA、GEMM2 valid rows、MoE output 与动态/poison/eager/graph replay 都是 bitwise mismatch=0。原 benchmark `count=19`、sanity `logits_diff=3.4733e-06`，计时前全机 idle。

两次 benchmark 的六秒 post 分别发现其他 E65 PID70587、83345；post 原始日志保留这些进程，之后按六秒等待其退出。虽然它们没有证明与计时重叠，也不能将最终 wait 到 idle 覆盖最初 post，因此不将两项作为严格空闲性能对比。筛选未显示扩大三轮的收益，两项不采用；未做十组 fallback，也没有用最低样本作为成绩。

8-row 因 allocated VGPR496 超过本轮与 review 相同的 480 门槛，仅在 CPU 淘汰；这是本轮资源实验门槛，不是硬件不支持 496 VGPR，不能写成 kernel 编译错误或必然精度/性能问题。controller 于 2026-10-17 12:51:02 UTC 完成并恢复 review SHA。全部 CPU、精度、筛选、post 原始日志已下载本地。

证据：[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_quant_rowgroup_v84_20261017_controller.log)、[2-row CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_quant_rowgroup2_v84_inspection.log)、[4-row CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_quant_rowgroup4_v84_inspection.log)、[8-row CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_quant_rowgroup8_v84_inspection.log)、[2-row benchmark](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_quant_rowgroup_v84_20261017_cyclic64_mg8_quant_rowgroup2_v84_bench.log)、[2-row post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_quant_rowgroup_v84_20261017_cyclic64_mg8_quant_rowgroup2_v84_bench_post.log)、[4-row benchmark](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_quant_rowgroup_v84_20261017_cyclic64_mg8_quant_rowgroup4_v84_bench.log)、[4-row post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_quant_rowgroup_v84_20261017_cyclic64_mg8_quant_rowgroup4_v84_bench_post.log)。

## Native FP4 16x16 WMMA 拆分：完整精度通过，筛选没有收益

v85 仅在原 `M256_FUSED_LAYOUT` 将一条 native FP4 `32x16x128` WMMA 拆成两个 native FP4 `16x16x128` WMMA。weight vec16 与 accumulator vec16 分成两个 vec8，结果拼回原 accumulator；两个 half 的 `scaleAType` 分别为 0/1，`scaleBType=wm % 2`，`fmtA=fmtB=4`，新指令不设置 operand reuse。保留原八个 waves、256 个 F32 accumulators/lane、每个 accumulator 的全部 56 个 K128 次序、TDM/reader/fences、量化和外部 layout。scheduler 的 physical WMMA 总数和 closing group 数同步乘二。

硬件依据：CDNA5 ISA §7.12.2（printed p.99）的 accumulator 两个 vec8 组织、§7.12.6.1–2（printed pp.104–107）的 FP4 vec8 和 scale lane selectors；Shader Programming Guide §5.7（printed pp.232–233）记载 16x16 FP4 repeat rate 4 cycles、32x16 FP4 8 cycles。这个等吞吐拆分仅是调度实验，文档没有保证它更快。

| 候选 | source SHA256 | VGPR / allocated VGPR | static instructions / WMMA16 | 精度 / replay | 原命令筛选（μs） |
|---|---|---|---|---|---:|
| `cyclic64_mg8_wmma16_group8_v85` | `367519a93ca5ccd55a4f104da3203ac9a8ffdb8afc0a897803f0ae5d123f4f64` | 484 / 496 | 4752 / 576 | 未运行 GPU | 未运行 GPU |
| `cyclic64_mg8_wmma16_group16_v85` | `54d32d4a6baf778b66ba01d3f56d343402b570cc05cf8aa42718564f19344dd8` | 475 / 480 | 4718 / 576 | 完整通过 / 完整通过 | 97.806 |

两项 SGPR106/LDS278528/private0/spills0。group8 超过本批与 review 相同的 allocated480 门槛，仅 CPU 淘汰；这不是硬件不支持496 VGPR。group16 完整11 cases的canonical payload、ScaleA、GEMM2 valid rows、MoE output及动态输入/poison/eager/graph replay全部 bitwise mismatch=0；未补十组 fallback。576 是包含不同控制流路径的静态 WMMA16 数，不是动态执行计数；源码完整 wave 模型为 `4 N tasks × 56 K128 × 32 WMMA16 = 7168`，本批没有新 ATT。

原 benchmark `count=19`、sanity `logits_diff=3.4733e-06`、workload exit0，13:02:14 UTC 全机 idle 后才计时，结束时间13:02:41 UTC。六秒后13:02:47 UTC发现其他 E65 PID121703，按六秒等待至13:03:57 UTC退出；原始post保留占用，不能用最后idle覆盖最初post。因此97.806 μs不列为严格空闲性能成绩，不能计算确定回退幅度。筛选没有显示进一步三轮的收益，不采用，controller结束后恢复正式review。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v85.py
# On host; each CPU/GPU invocation runs in dsr1_ep4 with a 120-second limit.
SCREEN_BATCH=<new_wmma16_v85_batch> RESOURCE_VGPR_MAX=480 \
bash .codex_tmp/gemm1_t1536_20261017/screen_bounded_v84_host.sh \
  cyclic64_mg8_wmma16_group8_v85 cyclic64_mg8_wmma16_group16_v85
```

证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/wmma16_v85_source_contract.json)、[controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma16_v85_20261017_controller.log)、[group8 CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_wmma16_group8_v85_inspection.log)、[group16 CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_mg8_wmma16_group16_v85_inspection.log)、[完整精度](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma16_v85_20261017_cyclic64_mg8_wmma16_group16_v85_equivalence.log)、[replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma16_v85_20261017_cyclic64_mg8_wmma16_group16_v85_replay.log)、[benchmark](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma16_v85_20261017_cyclic64_mg8_wmma16_group16_v85_bench.log)、[post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_wmma16_v85_20261017_cyclic64_mg8_wmma16_group16_v85_bench_post.log)。所有引用日志已下载本地，未覆盖历史实验文件。

## K128 input ring：消除 prologue spills 后仍未取得性能收益

v86 仅在原 `M256_FUSED_LAYOUT` specialization 中将内部搬运粒度 K256 改成 K128，分别使用 b4/b8 linear input ring、独立 scale ring 与 output arena。保留原 launcher ABI、完整 padded M256/N256/K7168、每 accumulator 的 56 次原顺序 K128、native WMMA、SiLU/BF16/per32 E8M0/native FP4 和外部 payload/ScaleA 布局。1344 项 request byte 区间、64 项 accumulator 顺序的静态检查通过；这些检查不能代替 GPU 正确性。

| 候选 | source SHA256 | VGPR / SGPR | LDS（B） | SGPR spills | 验证状态 |
|---|---|---|---:|---:|---|
| `linear32_k128_b4_rmem1_v86` | `8498ebaa5b89b58579adae15f39b7396f43390d7206f6b4b06fccda2e2bb7802` | 428 / 106 | 156672 | 4 | CPU gate 拒绝，没有 GPU 数值或性能数据 |
| `linear32_k128_b8_rmem1_v86` | `850646b4ca416dfc58b5f3680eadb9e1b33055c6ac3fc8ab1f18ebd1ce009fa0` | 429 / 106 | 295936 | 4 | CPU gate 拒绝，没有 GPU 数值或性能数据 |

两项 private/VGPR spills 为 0，但 SGPR spills 不满足本批严格 zero-spill 门禁。ISA 出现 `v_writelane_b32/v_readlane_b32` 保存两组跨 prologue 指针；不将更少 VGPR 描述成已经加速。v87 移除 successor 专用 kt0 compute，在 I0/I1 可读后只填 I2..I[PRE−1]，首 task 与 successor 都进入同一个 steady kt0。该 body 保留原 READY/reader fence，在覆盖 input0 前完成所有 readers；per-owner `I0,I1,O,I2,...` 顺序及 completion distance 也保证 output arena 在再次写入前完成。此调整消除新增 spills。

| 候选 | source SHA256 | VGPR / SGPR | LDS（B） | 完整 equivalence / replay | 原命令单次筛选（μs） |
|---|---|---|---:|---|---:|
| `linear32_k128_b4_common_prologue_v87` | `a2618c7bb4acbd909bc9050fb6519686cc5868c6cfab8bb1e92c39b925043c75` | 417 / 102 | 156672 | 通过 / 通过 | 108.465 |
| `linear32_k128_b8_common_prologue_v87` | `390908258e57be172f023e9437596757924bc4d2b5f0f6202dd756f5b47633ec` | 409 / 102 | 295936 | 通过 / 通过 | 108.056 |

v88 基于 v87，在覆写单 named rmem slot 前，将当前完整 A/B/scales 保留为 SSA operands；先调用下一组 LDS load，再让当前 WMMA 只消费保留的旧值。新增 `mma_rows_values` 与原 native intrinsic/attributes 相同，每 accumulator 顺序、reader/TENSOR/output fences 和量化布局不变。该 source 只表达 overlap 意图，没有新 ATT 证明实际 scheduler 的 overlap 距离。

| 候选 | source SHA256 | VGPR / SGPR | LDS（B） | 完整 equivalence / replay | 原命令单次筛选（μs） |
|---|---|---|---:|---|---:|
| `linear32_k128_b4_ssa_prefetch_v88` | `854df92b2112a1076783cd2f19b5cc0dcee2551554c590a86c09e5ca5f1976b1` | 434 / 102 | 156672 | 通过 / 通过 | 114.566 |
| `linear32_k128_b8_ssa_prefetch_v88` | `e24fa95bb07251fb2991929ebdb1934fd335b4c7a31f82ea2dbe0614c5bbbfa6` | 419 / 102 | 295936 | 通过 / 通过 | 116.736 |

v87/v88 四项均 private/VGPR spill/SGPR spill=0，完整 11 cases 的 canonical payload、ScaleA、GEMM2 valid rows、MoE output 及 dynamic-input/poison/eager/graph replay 全部 bitwise mismatch=0。四次原完整 MoE eager profiler 均 `count=19`、sanity `logits_diff=3.4733e-06`；前置及六秒后置全机 idle 均通过。后置原始日志从第一次检查就是 idle，没有用最终等待结果掩盖其他任务占用。上述是单次筛选，不能代替三轮交替结果；均未显示收益，未继续三轮或十组 fallback，不合入。

硬件依据：Shader Programming Guide §5.7.6（printed p.235–236）说明 SIMD-pair 共享 LDS issue，B128 independent repeat rate=2 cycles、dependent 约58 cycles；§4.10.1 规定同 wave TDM completion 有序、跨 wave 无序。资源减少或 source 的预读取顺序均不能保证性能；没有该候选的新 ATT，不定量归因回退于某个 wait、request 数或寄存器数量。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v86.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v87.py
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v88.py
# On host; each CPU/GPU invocation is limited to 120 s. Use unused batch labels.
SCREEN_BATCH=<new_k128_ring_batch> RESOURCE_VGPR_MAX=480 \
bash .codex_tmp/gemm1_t1536_20261017/screen_bounded_v84_host.sh \
  linear32_k128_b4_rmem1_v86 linear32_k128_b8_rmem1_v86
SCREEN_BATCH=<new_k128_common_batch> RESOURCE_VGPR_MAX=480 \
bash .codex_tmp/gemm1_t1536_20261017/screen_bounded_v84_host.sh \
  linear32_k128_b4_common_prologue_v87 linear32_k128_b8_common_prologue_v87
SCREEN_BATCH=<new_k128_ssa_batch> RESOURCE_VGPR_MAX=480 \
bash .codex_tmp/gemm1_t1536_20261017/screen_bounded_v84_host.sh \
  linear32_k128_b4_ssa_prefetch_v88 linear32_k128_b8_ssa_prefetch_v88
```

controller 分别于 13:28:09 / 13:36:58 / 13:48:18 UTC 完成，并恢复正式 review `b36f7ec...`。本地与远端 review SHA 以及用户 verification script SHA 已再次核对一致，未改变正式 94.888 μs 中值。

证据：[v86 contract](../.codex_tmp/gemm1_t1536_20261017/logs/k128_ring_v86_source_contract.json)、[v87 contract](../.codex_tmp/gemm1_t1536_20261017/logs/k128_common_prologue_v87_source_contract.json)、[v88 contract](../.codex_tmp/gemm1_t1536_20261017/logs/k128_ssa_prefetch_v88_source_contract.json)、[v86 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k128_ring_v86_20261017_controller.log)、[v87 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k128_common_v87_20261017_controller.log)、[v88 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k128_ssa_v88_20261017_controller.log)。三批逐项 CPU inspection、v87/v88 equivalence/replay/bench/post 原始日志均已下载本地。

## K256 共用 steady prologue：完整精度通过，首批配对的 review 有并发，候选保留待验收

v89 保留正式 K256/b4 cyclic64 data segments 和独立 scale arena，只调整原 M256 fused specialization 的 successor prologue。shared-output 版本在入口完成旧 output，再 prime I0、填 I2/I3，进入与首 task 相同的 steady kt0；原 body 在覆盖 I4 前完成 READY/reader fence。disjoint-output 版本将 output 放在独立 arena，旧 output 可暂时 outstanding，原 per-owner completion distance 保证其在后续 output staging 前完成。原 `mma_rows`、`k_step`、`compute_ktile`、reader/input TDM/output descriptor 函数保持，全部 padded M/N/K、56 次 K128 累加顺序和数值/外部布局不变。

| 候选 | source SHA256 | VGPR / SGPR | LDS（B） | 原命令筛选（μs） |
|---|---|---|---:|---:|
| `cyclic64_k256_common_shared_v89` | `d9de5d0bc6f5a3097455d6bfbc574a685e2f28a677d83dd853c7601c66777e65` | 467 / 106 | 278528 | 94.492 |
| `cyclic64_k256_common_disjoint_v89` | `021c88605df9f08ae4c4b70394592eb329353abc442292b2b065ac5022481b5b` | 467 / 106 | 295936 | 95.021 |

两项 CPU private/VGPR spill/SGPR spill=0，完整 11 cases equivalence 和 dynamic-input/poison/eager/graph replay 全部 bitwise mismatch=0。两次筛选前置和六秒后置全机 idle；原 benchmark `count=19`、sanity `logits_diff=3.4733e-06`。静态 code instances 减少不代表动态计算减少，也不自动证明提速。

shared-output 进入三轮交替，第二轮逆序。原始样本如下，保留全部数据，但该批不作为严格空闲性能结论：

| 轮次 | review（μs） | shared v89（μs） | 样本条件 |
|---:|---:|---:|---|
| 1 | 95.557 | 94.519 | 两项前置与首次六秒 post 均 idle |
| 2，逆序 | 94.951 | 94.283 | review 存在实际运行重叠；candidate post 的 E65 约在候选结束 3 秒后启动 |
| 3 | 97.018 | 94.357 | review post 的 E65 约在 review 结束 3 秒后启动；candidate post 短暂 busy1 后 idle |
| **原始中值** | **95.557** | **94.357** | **存在受限样本，不作为正式收益** |

第二轮 review 的 workload 起止为 14:25:23–14:25:40 UTC，14:25:46 post 发现 E65 host PID348062 已运行22秒，表明该任务在 review 运行期间启动。前置 idle 只能证明启动前状态，不能用最终等待 idle 掩盖随后出现的竞争负载。所有 post 原文及候选样本均保留；没有据该批中值声称稳定提升，也未合入或完成十组 fallback 验收。需要新的完整空闲配对才能确认候选收益。

用户指出 **94.357 μs 比已验收的 94.888 μs 更快**，这一数值判断正确，差为 **0.531 μs（0.56%）**。候选没有被删除或按性能淘汰，其状态为保留、待验收。第一轮同批 review 95.557 / candidate 94.519 μs，前置与首次六秒 post 均 idle，提供了 **1.038 μs（1.086%）** 的初步正向证据；跨批中值和含并发 review 的配对不足以确认稳定增益，但也不能据此否定候选。特别是 candidate 第二轮 14:24:35–14:24:52 UTC，post 14:24:58 看到 PID346153 已运行 3 秒，启动约为 14:24:55 UTC，晚于候选结束；该记录不证明候选计时被干扰。第三轮 candidate post 看到已退出 PID/busy1 也无法确定并发。收尾前优先完成 v89 的空闲配对和十组 fallback 验收，不展开 N128 等新方向。

```bash
python3 .codex_tmp/gemm1_t1536_20261017/build_candidates_v89.py
# On host, use unused batch names; every individual CPU/GPU command is bounded to 120 s.
SCREEN_BATCH=<new_k256_common_screen_batch> RESOURCE_VGPR_MAX=480 \
bash .codex_tmp/gemm1_t1536_20261017/screen_bounded_v84_host.sh \
  cyclic64_k256_common_shared_v89 cyclic64_k256_common_disjoint_v89
PAIR_BATCH=<new_k256_common_pair_batch> VALIDATION_BATCH=<new_k256_common_screen_batch> \
bash .codex_tmp/gemm1_t1536_20261017/pair_bounded_v90_host.sh \
  cyclic64_k256_common_shared_v89
```

screen 于 14:19:47 UTC、pair 于 14:27:13 UTC 完成，恢复正式 review SHA。14:43:23 UTC 再次检查全机 idle，无本任务编译/验证/benchmark/controller 残留；本地/远端 review 和用户 verification script SHA 保持一致。N128/4-wave 完整 M256 方案尚在静态设计阶段，没有生成候选或性能数据。

证据：[source contract](../.codex_tmp/gemm1_t1536_20261017/logs/k256_common_prologue_v89_source_contract.json)、[screen controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_20261017_controller.log)、[shared CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_k256_common_shared_v89_inspection.log)、[disjoint CPU](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_cyclic64_k256_common_disjoint_v89_inspection.log)、[三轮 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_pair_20261017_controller.log)、[第二轮 review](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_pair_20261017_cyclic64_mg8_delivery_review_2.log)、[第二轮 review post](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_pair_20261017_cyclic64_mg8_delivery_review_2_post.log)。六份配对 benchmark 及六份 post 日志已下载本地。

## v89 收尾验收：空闲复测未确认收益，保留 review

用户要求手头尝试做完后收尾暂停，随后追加抓取最优版 GEMM1 的 thread trace 并分析瓶颈。收尾仅补验收已有 v89，不生成新 kernel 候选。2026-10-17 **14:59:02–15:01:22 UTC** 完成三轮原完整 MoE eager profiler 配对，第二轮逆序；每个命令均使用 120 秒 timeout，六次前置和首次六秒后置从第一次采样就是全机 idle。所有 benchmark `count=19`、sanity=`logits_diff=3.4733e-06`、`rel_l2=2.6357e-03`、exit code=0。

| 轮次 | review（μs） | shared v89（μs） | candidate − review（μs） |
|---:|---:|---:|---:|
| 1 | 94.703 | 94.901 | +0.198 |
| 2，逆序 | 94.315 | 96.092 | +1.777 |
| 3 | 94.783 | 95.034 | +0.251 |
| **中值** | **94.703** | **95.034** | **+0.331（+0.35%）** |

首批候选中值 94.357 μs 的数值优势仍如实保留；新的空闲配对未复现收益，不能仅采用较快批次，**本次不合入 v89**。此判断不表示它有数值错误，也不能用没有新 ATT 的推测量化 prologue drain、static code size 或 VGPR 数量的贡献。完整 11 cases equivalence 和动态输入/poison/eager/graph replay 已通过。详见 [K256 共用 body Q&A](moe_gemm1_e64_t1536_nonbalanced_opt_qa.md)。

**15:01:45–15:04:58 UTC** 在容器中完成十组 CPU-only fallback 编译，每项命令单独限时 120 秒，未并发运行 benchmark。完整 ELF 十项均与 review 相同；target `.text` 改变，descriptor 仅 `INST_PREF_SIZE` 随实际 code length 调整，metadata 仅 VGPR470→467，ABI/geometry/LDS/control contract 全部通过。target VGPR467/SGPR106/LDS278528 B，private/VGPR spill/SGPR spill 均为0。没有扩大审核 allowlist。

实际复现命令（host）：

```bash
PAIR_BATCH=postcleanup_k256_common_v89_recheck_20261017 \
VALIDATION_BATCH=postcleanup_k256_common_v89_20261017 \
bash .codex_tmp/gemm1_t1536_20261017/pair_bounded_v90_host.sh \
  cyclic64_k256_common_shared_v89
bash .codex_tmp/gemm1_t1536_20261017/compile_v89_scope_host.sh
```

重新执行时给配对使用新的 `PAIR_BATCH`，避免覆盖已完成证据。15:05:41 UTC 及 15:13:05 UTC 再次确认正式源码仍为 `b36f7ec...`，用户 verification script 仍为 `fecea66...`，没有本任务编译/验证/benchmark/controller 残留；15:05:41 的全机空闲门禁通过。GEMM1 正式成绩保留 94.888 μs、相比冻结 baseline 提升4.30%、距80 μs仍差14.888 μs。

证据：[复测 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_recheck_20261017_controller.log)、[六次测量摘要](../.codex_tmp/gemm1_t1536_20261017/logs/v89_closeout_pair_summary.json)、[fallback controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_scope_20261017_controller.log)、[十组 ELF 和 target audit](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_k256_common_shared_v89.json)、[收尾环境](../.codex_tmp/gemm1_t1536_20261017/logs/v89_closeout_environment_20261017.log)。shared/disjoint 两项完整 equivalence/replay 日志、六份新 benchmark 和六份后置日志已下载本地。
