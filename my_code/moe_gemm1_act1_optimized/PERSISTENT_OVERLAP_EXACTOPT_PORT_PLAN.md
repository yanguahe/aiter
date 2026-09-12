# 将 FlyDSL `exactopt` 机制迁移到纯汇编 `persistent_overlap` 的实施方案

## 1. 目标与基线

目标 kernel：

```text
my_code/moe_gemm1_act1_optimized/
moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_persistent_overlap.s
```

当前 SHA256：

```text
ec3af906acebfdef77db5f734d01fbef45035dfcfdd799f557693aa73dd0d601
```

当前 contract：

```text
grid    = (16,16,1)
cluster = (4,4,1)
block   = (128,1,1)
LDS     = 327680 B = 320 KiB
VGPR    = 1024
SGPR    = 104 numbered SGPR
ABI     = production 184-byte GEMM1 ABI
```

它以 16 个 physical cluster persistent 地遍历 576 个有效 logical cluster task，
并把前一 task 的 output TDM drain 延迟到下一 task 的 descriptor/address setup 之后、
第一条 input TDM 之前。

本方案只设计新的候选版本，不直接覆盖当前保留的
`persistent_overlap.s`。所有生成脚本、候选 ISA、静态审计和结果继续放在
`my_code/moe_gemm1_act1_optimized/` 下。

## 2. 为什么不能把 FlyDSL 的 20.67% 直接搬过来

`sync_mg4_fc28_apre_exactopt` 相对 `baseline_93665e` 的大部分收益，在当前纯汇编
kernel 中已经存在：

- 4x4 cluster；
- A/B/ScaleA/ScaleB multicast；
- ABpreShuffle A/ScaleA layout；
- exact-SiLU；
- packed/dual-issue epilogue；
- 两阶段 output TDM；
- output LDS 双缓冲；
- 跨 persistent task 的 output-drain overlap；
- WMMA `matrix_a_reuse` / `matrix_b_reuse` bit。

而且两个 benchmark harness 不同：

- `my_code/reproduce_compare.sh` 使用当前仓库 FlyDSL MoE；
- `benchmark_history.sh` 使用自包含 `repo_snapshot` 和 C++ ASM injection；
- FlyDSL exactopt 使用普通大 grid；纯汇编版本使用 16-cluster persistent grid；
- 周边 quant/GEMM2 实现和 profiler 上下文不同。

因此，`546.656 us` 与 `541.037 us` 不能直接用来计算一个跨实现的百分比。所有候选
都必须在 `benchmark_history.sh` 内与当前 `persistent_overlap` 同进程、同机、交错
比较。

## 3. 当前 feature audit

| FlyDSL 优化 | `persistent_overlap` 当前状态 | 决策 |
|---|---|---|
| 4x4 cluster + 双向 multicast | 已有；8 个 descriptor setup 保留 bit 21，3 组核心 cluster barrier | 保留，不重做 |
| ABpreShuffle A/ScaleA | 已有；地址和 quant producer 已对齐 | 保留 |
| four-stage K pipeline | 已有 | 保留 |
| batch-8 exact-SiLU | 已有更低层实现：`v_dual_*`、`v_pk_mul_f32`、双临时组 EXP/RCP pipeline | 不回退到 FlyDSL 代码形态 |
| two-phase output store | 已有；第一半与 bank 2/3 epilogue overlap | 保留 |
| cross-task output drain overlap | 纯汇编独有 | 保留 |
| WMMA reuse | 已出现：静态 ISA 中 `matrix_a_reuse=128`、`matrix_b_reuse=128` | 先审计拓扑，不盲目增加 |
| `DISABLE_XDL_ARB_STALL=0` | **未采用**；入口显式把 `SCHED_MODE.bit[2]` 设为 1 | 第一优先级 |
| GEMM1-only WPT2 | **未采用**；来源是 production `..._wpt1`，当前每个 wave 负责一种 tensor | 第二优先级 |
| `mg4/fc28` schedule | 纯汇编有固定指令序，但尚未证明与 exactopt 相同 | 第四优先级 |
| active output `STORE_PAD=8` | 未采用；当前 compact row pitch 为 128 B | 第三优先级实验 |

### 3.1 已经存在的 exact-SiLU 优化

当前纯汇编每个 accumulator bank 使用：

```text
v_swap_b32
v_dual_min_num_f32
v_pk_mul_f32
v_exp_f32
v_dual_add_f32
v_rcp_f32
v_cvt_pk_bf16_f32
```

并用两套临时 VGPR 在当前 batch 的 reciprocal 与下一 batch 的 EXP/preparation 间
做软件流水。静态计数为：

```text
v_exp_f32       = 256
v_rcp_f32       = 256
v_pk_mul_f32    = 384
v_dual_* instr  = 256（每条包含两个 operation）
v_swap_b32      = 128
```

这已经比 FlyDSL 的高层 batch-8 表达更直接，不应把 epilogue 替换回生成代码。

### 3.2 当前 XDL arbitration 状态

入口现在执行：

```asm
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
```

第一条保留 SCHED_MODE 2；第二条把
`DISABLE_XDL_ARB_STALL` 设为 1。FlyDSL exactopt 显式选择 0，因此这是最小、最清晰
的缺失项。

### 3.3 当前 TDM ownership 是 WPT1

当前 kernel 源自 production symbol：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1
```

汇编在 wave-specific 分支中分别构造 A、B、ScaleA、ScaleB descriptor。每个 K256
tile 的 owner load 量大致为：

```text
A owner      = 32 KiB
B owner      = 32 KiB
ScaleA owner =  2 KiB
ScaleB owner =  2 KiB
```

这会让 A/B owner 到达 LDS-ready barrier 的时间明显晚于 scale owner。

### 3.4 当前 output LDS layout

当前 active output 明确使用 compact pitch：

```asm
; Compact activated output uses a 128-byte LDS row pitch.
v_mul_u32_u24_e64 v91, v4, 0x80
```

对每个 wave 的 64-column BF16 slice：

```text
row payload = 64 * 2 B = 128 B = 32 dwords
```

按 MI400 的 64-bank、4-byte/bank 地址模型，row 起始 bank 为：

```text
(row * 32 + kgrp * 2) mod 64
```

16 个 `lane16` 只落在少量重复起始 bank 上。已有 double-LDS ATT 中，两个主要
epilogue `s_wait_dscnt 0` 分别约为 `269` 和 `264 cycles/wave`，说明 output DS
drain 仍值得单独检查；但这项 attribution 同时包含 queue backlog，不能直接断言
全部都是 bank conflict。

## 4. 实施总原则

1. 每项机制先生成独立候选，再组合，不一次修改多个变量。
2. 所有变体由 Python build script 从固定 SHA 的 `persistent_overlap.s` 生成；使用
   exact string/count assertions，源文件结构变化时立即失败。
3. 不修改保留版 ISA，直到候选通过 random MoE e2e correctness 和同机交错性能。
4. const0 只用于正式性能；正确性必须显式运行 `e2e-random`。
5. `RUN_VERIFY=1 ... e2e-const0` 当前不会额外执行 random e2e，不能把它当成 random
   correctness gate。
6. 每次机器重连或重启后先重新测 `persistent_overlap`，再测候选；只比较同一轮结果。
7. 不删除 cluster barrier、不跨未完成 output TDM 覆盖 LDS、不超过每 wave 3 个
   in-flight TDM descriptor。

建议新增：

```text
my_code/moe_gemm1_act1_optimized/build_exactopt_port_variants.py
my_code/moe_gemm1_act1_optimized/audit_exactopt_port.py
my_code/moe_gemm1_act1_optimized/exactopt_port/
```

候选命名：

```text
persistent_overlap_xdl0.s
persistent_overlap_wpt2.s
persistent_overlap_xdl0_wpt2.s
persistent_overlap_xdl0_wpt2_opad8.s
persistent_overlap_exact_sched.s
```

## 5. 阶段 0：建立可验证的 ISA 对照

### 5.1 固定 FlyDSL reference

先用当前 HEAD 生成并保存：

```text
sync_mg4_fc28_apre
sync_mg4_fc28_apre_exactopt
```

的 final ISA、code object metadata 和 ATT。建议命令：

```bash
CASE_LIST=baseline_93665e,sync_mg4_fc28_apre,sync_mg4_fc28_apre_exactopt \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=1 \
bash my_code/reproduce_compare.sh
```

需要从 final ISA 自动提取：

- `SCHED_MODE` 写入；
- 每个 K256 tile、每个 wave 的 TDM descriptor 数；
- descriptor 的 global offset、tile outer/inner、LDS offset、OOB extent 和
  `workgroup_mask`；
- `s_wait_tensorcnt` threshold；
- 每个 K128 的 WMMA/DS/TDM 指令序列；
- `matrix_a_reuse` / `matrix_b_reuse` 的动态拓扑；
- output row pitch、half-store owner 和 descriptor shape。

不能仅根据 FlyDSL Python 猜 descriptor bitfield；最终生成 ISA 才是移植模板。

### 5.2 静态审计脚本

`audit_exactopt_port.py` 至少检查：

```text
kernel symbol
184-byte kernarg metadata
cluster_dims=(4,4,1)
group_segment_fixed_size=327680
VGPR/SGPR count
tensor_load_to_lds / tensor_store_from_lds count
s_wait_tensorcnt value distribution
workgroup and cluster barrier count
WMMA count and reuse-bit count
output LDS address interval
```

任何 candidate 违反 ABI、grid、LDS 上限或 barrier generation 都不进入 GPU 测试。

## 6. 阶段 1：只迁移 `DISABLE_XDL_ARB_STALL=0`

### 6.1 修改

保留：

```asm
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
```

删除或改成 0：

```asm
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1
```

优先选择删除第二条写入，让 bit 2 保持 kernel 启动时的 0。不要关闭 SCHED_MODE 2，
因为当前汇编依赖显式 `s_wait_alu` / `s_wait_dscnt` hazard protocol。

### 6.2 理由

硬件文档说明，bit 2 为 1 会允许同一 wave 连续 issue WMMA，但可能阻塞其他 wave
的 co-execution。FlyDSL A-preshuffle 的历史单项实验中，bit 2 设为 0 获得约
`1.91%` 收益。

纯汇编的 WMMA/DS 交错与 FlyDSL 不同，所以这里只能把它当作低风险 A/B test，
不能预设一定变快。

### 6.3 验收

- 汇编成功，metadata 完全不变；
- random e2e hash 与同轮 `persistent_overlap` 相同；
- const0 e2e GEMM1 median 至少不回退；
- ATT 对比 WMMA attributed cycles、DS issue 间隔和 wave progress。

## 7. 阶段 2：实现 GEMM1-only WPT2 input TDM

这是最有价值、也最需要谨慎实现的缺失机制。

### 7.1 ownership 设计

按 FlyDSL exactopt 的分组：

```text
wave 0/1: A half + ScaleA half
wave 2/3: B half + ScaleB half
```

每个 wave 每个 K256 tile 搬运：

```text
16 KiB payload + 1 KiB scale = 17 KiB
```

总数据量仍为 `68 KiB/K256 tile/workgroup`，但四个 wave 的 TDM 工作量完全均衡。

### 7.2 descriptor 拆分

对每类 tensor：

| tensor | 原 WPT1 tile | WPT2 每个 owner |
|---|---|---|
| A | outer 16 × inner 2048 B | outer 8 × inner 2048 B |
| B | outer 16 × inner 2048 B | outer 8 × inner 2048 B |
| ScaleA | outer 8 × inner 64 dwords | outer 4 × inner 64 dwords |
| ScaleB | outer 8 × inner 64 dwords | outer 4 × inner 64 dwords |

第二个 owner 的 global/LDS outer offset 分别加上半个 outer extent。A/ScaleA 继续
使用 N-row multicast mask；B/ScaleB 继续使用同 expert M-column mask；bit 21
`early_timeout` 保持不变。

### 7.3 descriptor register 复用

当前 kernel 的 SGPR 空间接近上限，不能为第二个 descriptor 长期保留一套新 SGPR。
实现应在每个 wave branch 内：

1. 构造并 issue payload half descriptor；
2. 复用同一组 `s[32:43]` descriptor SGPR；
3. 构造并 issue scale half descriptor；
4. 返回共用 compute path。

这样每个 wave 同时最多保留两个已发出的 TDM op，但不增加 persistent descriptor
state。

### 7.4 `TENSORcnt` 重新设计

MI400 Shader Programming Guide §4.10.8 的硬限制是：

```text
per wave in-flight descriptor <= 3
per SIMD in-flight descriptor <= 6
```

WPT1 当前常见模式是 `s_wait_tensorcnt 0x2` 后再 issue 一个 descriptor。WPT2 不能
机械地在同一点连续多发一条，否则可能从 2 个 outstanding 增加到 4 个。

实施时必须以 exactopt final ISA 为准，选择下列之一：

- 发 descriptor pair 前等待到 `tensorcnt <= 1`；
- 先 issue 一个、在第二个前做更精确 threshold；
- 把两个 issue 分散到不同 WMMA group，使前一个先获得 XACK。

每个 cluster peer 必须以相同顺序发出 pairwise-matching multicast descriptor。任何
只修改一个 wave path 或一个 ring generation 的做法都有 deadlock 风险。

### 7.5 persistent boundary

保留现有跨 task 顺序：

```text
previous task output TDM remains in flight
  -> next task scalar/address/descriptor setup
  -> s_wait_tensorcnt 0 before first input TDM/barrier
  -> issue WPT2 input descriptors
```

WPT2 只改变当前 task 的 input owner，不改变 output LDS 生命周期。最终 task 仍执行
`s_wait_idle`。

### 7.6 验收

- random standalone 多 seed、多 launch；
- random MoE e2e 至少 3 轮，相同 hash；
- const0 MoE e2e 同轮交错 5 轮；
- ATT 中初始和 steady LDS-ready barrier 应下降；
- 不出现新的 descriptor-slot stall、额外 `s_wait_tensorcnt 0` 或 cluster imbalance。

## 8. 阶段 3：output LDS row skew

### 8.1 最小改动方案

保持当前“每个 wave 独立存 64 output columns”的组织，只把 row pitch 从：

```text
64 BF16 = 128 B = 32 dwords
```

改为：

```text
64 + 8 BF16 = 72 BF16 = 144 B = 36 dwords
```

对应改动：

```asm
v_mul_u32_u24_e64 v91, v4, 0x80
```

改为：

```asm
v_mul_u32_u24_e64 v91, v4, 0x90
```

当前 bank 起点：

```text
(row*32 + kgrp*2) mod 64
```

候选 bank 起点：

```text
(row*36 + kgrp*2) mod 64
```

`gcd(36,64)=4`，因此 16 个 `lane16` 可以分布到 16 个不同的 4-bank group；再加
`kgrp*2` 后，两组 lane 覆盖不同起始 bank。理论映射明显优于 128-byte pitch。

### 8.2 double-output-LDS 地址调整

每个 half 目前为：

```text
64 rows * 128 B = 8192 B = 0x2000
```

改成 padded pitch 后：

```text
64 rows * 144 B = 9216 B = 0x2400
```

因此第二 half 的：

```asm
v_add_nc_u32_e32 v91, 0x2000, v91
s_add_co_u32 s81, s81, 0x2000
```

必须同步改为 `0x2400`，并重新编码 output TDM descriptor：

- LDS tile inner extent 使用 72 BF16；
- global/OOB inner extent 仍为 64 BF16；
- global row stride 保持 3072 BF16；
- 不设置 TDM `pad_enable`。硬件文档说明 store path 忽略该字段；正确方式与
  FlyDSL exactopt 一样，是扩大 LDS tile pitch并用 OOB 丢弃 pad columns。

### 8.3 地址安全证明

生成脚本必须枚举四个 wave、两个 half 的 `[start,end)` LDS 区间，并证明：

- 区间互不重叠；
- 最大地址 `< 327680`；
- 第一 half TDM 完成前，第二 half 不覆盖第一 half；
- 下一 persistent task 第一条 input TDM 前仍有 `s_wait_tensorcnt 0`。

如果 compact per-wave padded layout 无收益，再考虑 exactopt 的 shared full-row
layout；后者需要 wave_n peers 协作 staging 和只由 `wave_n==0` 发 output descriptor，
改动面更大，不作为第一版。

### 8.4 验收重点

ATT 重点比较：

```text
ds_store_b64 attributed stall
两个 epilogue s_wait_dscnt 0
workgroup barrier wait
first/second tensor_store_from_lds issue 到完成
```

如果 `s_wait_dscnt` 没有下降，立即撤销该候选；增加 padding 只会扩大 output TDM
tile 和 descriptor work。

## 9. 阶段 4：把 `mg4/fc28` 的最终指令顺序迁入 hotloop

FlyDSL 的 `mg4/fc28` 是 compiler scheduling hint，纯汇编中没有可以直接设置的
等价环境变量。必须比较 final ISA 后，显式重排指令。

### 9.1 提取模板

对 exactopt 和纯汇编各自的一个 steady K256 tile，生成序列：

```text
TDM issue
s_wait_tensorcnt
DS A reads
DS B reads
DS ScaleA/ScaleB reads
32 WMMA for K128-0
32 WMMA for K128-1
s_wait_alu / s_wait_dscnt
workgroup/cluster barrier
```

记录每条指令的：

```text
operand VGPR range
destination accumulator
K128 generation
LDS buffer generation
dependency wait
```

### 9.2 重排规则

目标不是照抄地址，而是复现 scheduling shape：

1. K128-0 前部只保留一个 4-WMMA group；
2. 尽早发出 next-state 的 40 个 DS reads；
3. 以 28-WMMA closing group 覆盖 LDS/TDM fence；
4. K128-1 继续按 4-WMMA groups 调度；
5. 每个 accumulator 的 K 顺序完全不变；
6. 不跨 cluster ring generation 移动 load/store；
7. 在 SCHED_MODE 2 下保留所有显式 RAW/WAR wait。

纯汇编是大规模展开代码，必须通过脚本识别重复模板并生成，不能手工修改几十处
相似片段。

### 9.3 预期

本轮 FlyDSL `mg4/fc8 -> mg4/fc28` 只有约 `0.50%` GEMM1 差异，因此这一阶段
优先级低于 WPT2 和 XDL arbitration。只有 ATT 显示 pure ASM 的 DS-read burst 或
fence stall 明显高于 exactopt 时才继续。

## 10. 阶段 5：审计 WMMA reuse topology

当前纯汇编已经有：

```text
512 static WMMA
128 matrix_a_reuse
128 matrix_b_reuse
```

所以“打开 reuse bit”已经不是一个可执行方案。需要做的是：

1. 解析相邻 `v_wmma_scale_f32_32x16x128_f4`；
2. 比较实际 A/B source VGPR range、scale select 和 bank；
3. 只在相邻 operand 完全一致时保留或增加 reuse；
4. 将当前 traversal 与 exactopt snake traversal 的动态 reuse 比例对比；
5. 每次改变 traversal 都保持每个 C accumulator 的 K 累加顺序。

CDNA5 ISA §7.12 明确说明，reuse bit 与真实相邻 operand 不一致时结果 undefined。
因此该阶段必须由静态 checker 拒绝非法标记，不能依赖 const0 或少量 random 测试
碰巧通过。

历史 FlyDSL 实验中 A+B reuse 只有约 `0.94%` 单项收益，且后续 column-major
traversal 没有提升，所以本阶段排在 hotloop scheduling 之后。

## 11. 不建议重复实施的方向

以下机制已经存在，或已有明确负面实验：

- 再增加 packed SiLU：当前纯汇编已经使用 packed/dual pipeline；
- 再拆 output 为四阶段：FlyDSL 历史结果比两阶段慢约 `0.47%`；
- 合并成单个 128-row output TDM：纯汇编历史测试略慢；
- 用 `buffer_store` 替代 output TDM：历史 LDS-buffer-store 回退约 `4.7%`，
  VGPR-direct store 回退约 `25%` 以上；
- 删除 steady cluster barrier：历史纯汇编实验 deadlock；
- WPT4：历史 FlyDSL 实验出现 NaN；
- 近似 SiLU / ReLU：改变数学语义，不满足本任务要求；
- 继续缩小 persistent grid：当前 16 cluster 已覆盖 256 CU，先解决单 task stall。

## 12. 候选组合顺序

按以下顺序构造和筛选：

| 阶段 | candidate | 目的 |
|---|---|---|
| A | `persistent_overlap_xdl0` | 单独验证 arbitration bit |
| B | `persistent_overlap_wpt2` | 单独验证 TDM owner balance |
| C | `persistent_overlap_xdl0_wpt2` | 检查两者是否互补 |
| D | `..._xdl0_wpt2_opad8` | 降低 output DS drain |
| E | `persistent_overlap_exact_sched` | 迁移 `mg4/fc28` 指令顺序 |
| F | winner + reuse retune | 仅在静态 checker 证明合法时尝试 |

每一阶段都保留上一阶段 winner；失败版本不覆盖历史文件。

## 13. 正确性与性能流程

### 13.1 random MoE e2e correctness gate

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/exactopt_port/CANDIDATE.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=3 RUN_VERIFY=0 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

必须满足：

- `pass=True`；
- `logits_diff < 0.01`；
- `rel_l2` 不高于同轮 `persistent_overlap`；
- output hash 与同轮 `persistent_overlap` 相同；
- 连续多轮无 hang、无偶发 mismatch。

### 13.2 正式 const0 性能

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/exactopt_port/CANDIDATE.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=5 RUN_VERIFY=0 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

正式指标使用 MoE e2e profiler 中的 GEMM1 median。测试顺序应交错，并同时记录
fused MoE median。换机器、SSH 失联后重连或机器重启时，必须在同一轮重新测
`persistent_overlap`。

### 13.3 ATT

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/exactopt_port/CANDIDATE.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att
```

重点观察：

- kernel 总 GFXCLK cycles；
- 每 persistent task 的 normalized cycles；
- initial/steady `s_barrier_wait`；
- `s_wait_tensorcnt`；
- `s_wait_dscnt`；
- WMMA attributed stall；
- 四个 wave/SIMD 的完成偏差；
- 是否出现新 spill 或 descriptor-slot pressure。

## 14. 接受与止损标准

候选只有同时满足以下条件才保留：

1. random MoE e2e 连续通过并与同轮 baseline hash 一致；
2. const0 GEMM1 median 在同轮至少有可重复改善；
3. ATT cycle 同方向下降，排除仅由 GFXCLK 漂移造成的微秒变化；
4. `.amdhsa_group_segment_fixed_size` 不超过 `327680`；
5. 不增加 VGPR/SGPR spill；
6. 不增加无法解释的 cluster barrier 或 TDM wait；
7. 测试后无残留 GPU/KFD 进程。

对小于约 `0.5%` 的单轮变化，默认视为噪声，需要增加轮数或反向顺序复测。某一
候选即使单独持平，若它是下一阶段的必要基础，可以暂时保留为实验分支，但不能替换
正式 `persistent_overlap`。

## 15. 收益预期

可迁移的最大收益块不是 A preshuffle、4x4 multicast 或 SiLU pipeline，因为这些
已经存在。剩余候选的历史 FlyDSL 量级大致为：

```text
DISABLE_XDL_ARB_STALL=0 约 0%～2%
WPT2                    约 1%～3%
mg4/fc28                约 0%～1%
reuse topology retune   约 0%～1%
output LDS skew         未知，取决于纯汇编的 DS drain
```

这些收益重叠，不能相加。对当前 `persistent_overlap`，较现实的第一阶段目标是
`2%～4%` 的同轮 GEMM1 cycle 降低；如果 WPT2 同时显著降低 initial/steady
LDS-ready barrier，且 output padding 降低 `s_wait_dscnt`，才有机会超过这个范围。

最先应该实施的是 `xdl0` 的单行 A/B test，然后实现完整 WPT2 descriptor/wait
协议。它们分别提供最低风险和最高结构性收益的信息。

## 16. 硬件依据

- MI400 Shader Programming Guide §4.3.7.4、§4.3.7.4.2：SCHED_MODE 2 和
  `DISABLE_XDL_ARB_STALL`；
- MI400 Shader Programming Guide §4.6、CDNA5 ISA §7.12：WMMA A/B reuse
  的合法性条件；
- MI400 Shader Programming Guide §4.7.1：LDS 为 64 banks、4 bytes/bank，
  每 WGP 最多 320 KiB LDS；
- MI400 Shader Programming Guide §4.10.3：`workgroup_mask` multicast 和
  descriptor bit 21 `early_timeout`；
- MI400 Shader Programming Guide §4.10.8：每 wave 最多 3 个、每 SIMD 最多
  6 个 TDM descriptor in flight；TDM 可以在 descriptor 间切换；
- MI455X whitepaper 第 10 页：每 WGP 独立 TDM、LDS/DRAM direct transfer、
  multicast 和 320 KiB LDS。
