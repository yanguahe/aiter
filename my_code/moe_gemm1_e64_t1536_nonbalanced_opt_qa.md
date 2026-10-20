# gfx1250 MoE GEMM1：E64/T1536 non-balanced 优化问答

记录日期：2026-10-17。本文整理本次会话中关于 M128/N256 方案、已确认最优版的 tile/LDS，以及输出 FP4 scale 写回方式的问答。

讨论范围为 `E64/T1536/topk8/model_dim=7168/inter_dim=2048`、random non-balanced routing、A4W4、融合 SiLU、无 bias，shared expert 单独计算。性能口径为原完整 MoE eager profiler 的 GEMM1 `device_time_avg`，`--iters 20` 对应 `count=19`。

本文所称“已确认最优版”是三轮交替复测中值为 **94.888 μs** 的 mg8 整理版。同批 baseline 中值为 **99.152 μs**，减少 **4.264 μs（4.30%）**，距离 80 μs 仍差 **14.888 μs**。这些是已有测试结果；本次整理问答没有重新运行 GPU 测试。

对应源码：

```text
aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py
SHA256: b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790
```

## 问题 1：前面尝试的 M128/N256 tile 方案性能如何？

查到的对应实验是 v52 的 `split_mhalf_b3_named_drain`，GEMM1 单次筛选结果为 **114.585 μs**，没有显示性能收益，未采用。

这个实验仍保留一个完整的 **M256/N256 WG task**，在 WG 内分成两个各自计算 M128/N256 的组。它没有将 launcher 的 WG tile 改成独立 M128/N256；实际 profiler 中的 kernel 名仍包含 `t256x256x256_w2x4_b3`。

每个组使用三级输入 ring 和四成员 named barrier，四条 wave 分别负责 A128、完整 B256、ScaleA128、完整 ScaleB256。完整 padded M/N/K、K7168 的全部贡献、每个 accumulator 的 K128 累加顺序，以及 activation、BF16 rounding、FP4/E8M0 和外部布局均保留。B/ScaleB 搬运量翻倍；下一 N-task refill 前完整等待旧 output，这版没有采用原跨 task 的 I0/I1 overlap。

| 项目 | v52 结果 |
|---|---|
| GEMM1 单次 `device_time_avg` | 114.585 μs，count=19 |
| VGPR / SGPR | 473 / 107 |
| 每 WG LDS | 313344 B = 306 KiB |
| private / VGPR spill / SGPR spill | 全部为 0 |
| 完整 11 cases equivalence | canonical payload、ScaleA、GEMM2 valid rows、MoE output 全部 bitwise mismatch=0 |
| 动态输入、poison、eager/graph replay | 全部通过 |
| 原 benchmark sanity | `logits_diff=3.4733e-06` |
| 三轮交替复测 | 未做 |

四级独立 rings 需要 `2 × 4 × (16384 + 32768 + 1024 + 2048) = 417792 B`，超过 320 KiB LDS 上限，所以实验采用三级 rings，合计 313344 B。容量依据为 `MI400_Shader_Programming#65.txt` §4.7.1（printed p.160）。

**性能数据的限制：** controller 和 benchmark 启动前均记录了空闲状态，但运行结束后的即时检查为 `busy=3`，日志提示 `Post-run GPU state needs inspection.`。因此 114.585 μs 只作为已有单次筛选记录，不能与 94.888 μs 的正式三轮中值相减，声称已经严格确认具体回退幅度。这个候选没有新 ATT，也不能把耗时变化单独归因于 B 搬运翻倍、三级 buffer 或 output drain。

证据：

- [v52 生成器](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v52.py)
- [原 benchmark 与即时后置检查](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_mhalf_v52_20261017_split_mhalf_b3_named_drain_bench.log)
- [完整 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_mhalf_v52_20261017_controller.log)
- [CPU ISA/resource](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_split_mhalf_b3_named_drain_inspection.log)
- [完整 equivalence](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_mhalf_v52_20261017_split_mhalf_b3_named_drain_equivalence.log)
- [完整 replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_mhalf_v52_20261017_split_mhalf_b3_named_drain_replay.log)

## 问题 2：当前性能最优版采用什么 tile？总共用了多少 LDS？输出 FP4 的 scale 是否经过 LDS/TDM？

### Tile 和 wave 分工

已确认最优版采用 **M256/N256/K256 tile，w2x4/b4**，共 8 条 wave，每个 WG 顺序计算 4 个 persistent N-task：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist
```

每条 wave 负责 M128/N64。`K256` 是输入 pipeline tile，完整 K7168 仍分成 K128 contribution 顺序累加。这里 N256 是激活前的 gate+up 宽度；融合 SiLU 和逐元素乘法后，单个 N-task 产生 **M256/N128 的 FP4 输出**。

### LDS 总量与输出区复用

编译后 ELF 的 `.group_segment_fixed_size` 为 **278528 B = 272 KiB，每 WG**，组成如下：

| 用途 | LDS 大小 |
|---|---:|
| A/B 数据 ring：4 个 64 KiB cyclic segments | 256 KiB |
| 输入 ScaleA：4 个 stage，每 stage 2 KiB | 8 KiB |
| 输入 ScaleB：4 个 stage，每 stage 2 KiB | 8 KiB |
| **总分配量** | **272 KiB** |

每个数据 segment 放本 stage 的 A0（16 KiB）、B（32 KiB）和前一个 stage 的 A1（16 KiB）；输入 ScaleA/ScaleB 位于独立的四级 scale arena。

单个 N-task 的 FP4 输出 payload 为 `256 × 128 × 4 / 8 = 16384 B`，复用已释放的输入 LDS 区域 **[245760, 262144)**，不额外增加总 LDS 分配。

### FP4 payload 和输出 scale 的写回路径

**当前 t256 版本的输出 scale 没有经过 LDS，也没有通过 TDM store 写出。** 两种输出分别使用下面的路径：

| 输出内容 | 实际写回路径 |
|---|---|
| FP4 payload | 在寄存器中量化、打包 → `lds_store_b16` → WG barrier → TDM store 到 global |
| 每行、每 32 个输出元素对应的 E8M0 scale | 在寄存器中生成 → `fx.ptr_store` 直接写 global → `s_wait_storecnt(0)` 等待完成 |

输出 scale 按 GEMM2 所需的 canonical M32 布局直接写入 `arg_quant_scale`，作为 GEMM2 的输入 ScaleA。该输出 scale 与上表中用于 GEMM1 输入的 ScaleA/ScaleB arena 是不同的数据。

源码中 `tile_m == 256 or partial_m_direct_scale` 分支选择直接 global store；`issue_quant_scale_output()` 定义的 LDS → TDM 路径供其他配置使用，当前 t256 分支不调用它。TDM 硬件路径的依据为 CDNA5 ISA §10.11.1、§15.17：`TENSOR_STORE_FROM_LDS` 将 LDS 中的数据写到 global。

证据：

- [persistent kernel 源码](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py)：`M256_FUSED_LAYOUT`、`SPLIT_SCALE_ARENA`、`ARENA_B`、`issue_quant_output()` 和 `issue_quant_scale_output()`，以及 `fx.ptr_store(e8m0_byte, scale_ptr + dst_byte)` 分支。
- [正式版 ELF/resource 核对](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_delivery_review_comparison.log)
- [优化记录](moe_gemm1_e64_t1536_nonbalanced_optimization.md)：正式三轮复测中值与当前源码 SHA256。

## 问题 3：输出 scale 经 LDS/TDM 的路径，之前尝试过吗？性能如何？

**针对当前 t256 尝试过，但只完成 CPU 编译与 ISA 筛选，没有 GPU 实测耗时。** 做过两版候选：

- `cyclic64_mg8_tdm_scale`：将输出 scale 写入独立的 canonical M32 LDS source，再由每个 M half 的一个 wave 发出 TDM store。
- `cyclic64_mg8_tdm_scale_stage2`：将输出 scale source 放到已释放的 stage-2 区域，再由 TDM 写出。

| 版本 | VGPR | SGPR | 每 WG LDS | SGPR spills | GPU 性能数据 |
|---|---:|---:|---:|---:|---|
| 已确认最优版，scale 直接写 global | 470 | 106 | 278528 B（272 KiB） | 0 | 正式三轮中值 94.888 μs |
| `cyclic64_mg8_tdm_scale` | 474 | 107 | 279552 B（273 KiB） | 12 | 无 |
| `cyclic64_mg8_tdm_scale_stage2` | 474 | 107 | 279552 B（273 KiB） | 12 | 无 |

两版 TDM 候选均因新增 12 个 SGPR spill，在静态筛选阶段被淘汰，未运行 GPU 正确性验证或 benchmark，也未合入正式源码。**无法据此判断输出 scale 经 LDS/TDM 实际快多少或慢多少**；新增 spill 不能代替性能测量证明必然回退。

证据：

- [TDM scale 生成器](../.codex_tmp/gemm1_t1536_20261017/build_candidates_v19.py)
- [独立 scale source 的 CPU inspection](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_tdm_scale_inspection.log)
- [stage-2 source 的 CPU inspection](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_tdm_scale_stage2_inspection.log)
- [优化记录](moe_gemm1_e64_t1536_nonbalanced_optimization.md)中的“ScaleA 输出候选的静态筛选”。

## 问题 4：K256 共用 steady body 的原始中值 94.357 μs 比最优版更好，为何未采用？

**数值上确实更好，先前没有按性能淘汰这个候选。** `cyclic64_k256_common_shared_v89` 的首批三轮为 94.519、94.283、94.357 μs，中值 **94.357 μs**，比已验收版本的历史中值 94.888 μs 少 **0.531 μs（0.56%）**。当时源码和日志均保留，状态为待验收、未合入。此前将整批样本笼统标为受限，容易让人误以为候选被否定，应更精确说明。

首批第一轮同批 review / candidate 为 95.557 / 94.519 μs，两项前置和首次六秒后置均 idle，是初步正向证据。明确存在并发的是第二轮 review：运行 14:25:23–14:25:40 UTC，后置 14:25:46 发现另一个 E65 测试已运行 22 秒。候选第二轮于 14:24:52 结束，14:24:58 后置发现的 E65 测试已运行 3 秒，约在 14:24:55 启动，晚于候选结束，因此这个后置记录不证明候选计时被干扰。跨批中值和含并发 review 的配对不够确认稳定增益，但也不构成淘汰候选的理由。

按用户要求在暂停前完成这个已有尝试，补做三轮原命令交替测试，第二轮逆序。六次运行前置与首次六秒后置均为全机 idle，`count=19`、原精度 sanity 均通过，原始结果如下：

| 轮次 | review（μs） | K256 共用 body（μs） | candidate − review（μs） |
|---:|---:|---:|---:|
| 1 | 94.703 | 94.901 | +0.198 |
| 2，逆序 | 94.315 | 96.092 | +1.777 |
| 3 | 94.783 | 95.034 | +0.251 |
| 中值 | **94.703** | **95.034** | **+0.331（+0.35%）** |

**新配对未复现首批收益，候选三轮均慢，因此本次不合入。** 两批原始数据都保留，不能只选择较快的首批中值。这个结论限于本次原完整 MoE eager profiler 测量；没有新的 v89 ATT，不能将变化定量归因于 prologue drain、VGPR 数或 static code size。完整 11 cases equivalence 和动态输入/poison/eager/graph replay 已通过，canonical payload、ScaleA、GEMM2 valid rows、MoE output 全部 bitwise mismatch=0。源码和证据仍保留供后续复查，正式采用版本仍为 review。

复现本次配对（在 host 执行，使用未用过的 `PAIR_BATCH`，每个 benchmark 均有 120 秒 timeout）：

```bash
PAIR_BATCH=<new_k256_common_recheck_batch> \
VALIDATION_BATCH=postcleanup_k256_common_v89_20261017 \
bash .codex_tmp/gemm1_t1536_20261017/pair_bounded_v90_host.sh \
  cyclic64_k256_common_shared_v89
```

证据：[首批配对 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_pair_20261017_controller.log)、[空闲复测 controller](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_recheck_20261017_controller.log)、[六次测量摘要](../.codex_tmp/gemm1_t1536_20261017/logs/v89_closeout_pair_summary.json)、[完整精度](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_20261017_cyclic64_k256_common_shared_v89_equivalence.log)、[动态 replay](../.codex_tmp/gemm1_t1536_20261017/logs/postcleanup_k256_common_v89_20261017_cyclic64_k256_common_shared_v89_replay.log)。六份复测 benchmark 和六份后置日志已下载本地。

补充收尾结果：十组 fallback 的完整 ELF 与 review 全部相同，目标 descriptor/metadata 审核通过，VGPR467/SGPR106/LDS278528 B，private 和 VGPR/SGPR spills 均为0。该静态兼容性结果不改变“空闲复测未确认收益、本次不合入”的决定。证据：[十组 fallback 与 target audit](../.codex_tmp/gemm1_t1536_20261017/logs/cpu_bounded_scope_cyclic64_k256_common_shared_v89.json)。

## 问题 5：三次 N-task 交接占14.09%，是否试过去掉persistent，让每个WG只执行一个N-task？

**试过。** `cyclic64_mg8_ntasks1_v2` 基于与当前最优版相同的review，仅在原M256 fused编译期分支将每WG task数4→1，保留t256/w2x4/b4、完整M/N/K及数值功能。完整11 cases equivalence和dynamic/poison/eager/graph replay通过，原命令单次筛选为 **98.032 μs/count19**；没有进入三轮配对或完整ELF/fallback审核，也没有该候选的ATT。原前置idle，立即后置busy3/无KFD，缺少六秒复核。因此该值说明已经测试过这一方案，不能据跨批94.888 μs中值计算确定回退，也不能据单次筛选否定non-persistent方向。

还试过每WG两个task：单次93.944 μs，三轮原始候选中值95.496 μs、对应review95.561 μs，方向不一致；target新增4个SGPR spills，未采用。旧配对后置存在busy及其他任务/CPU编译记录，没有作为新的严格空闲收益结论。

14.09%包含当前task必要的SiLU/quant/staging/output发布和下一task prologue，不是全可删同步。1-task将这些工作分到更多WG，launch448→1792、compute WG256→1024，各WG仍占272KiB LDS；独立WG重复setup/prime/drain，并失去原跨task I0/I1 prefetch overlap。更多WG的调度是否抵消这些成本需要同批实验，不能直接从94.888 μs减掉14.09%。候选仍使用原`_persist`名字，实际task数以代码为准。

完整说明、两方案全部数据与原始证据已补入 [thread trace报告第7.1节](moe_gemm1_e64_t1536_best_thread_trace_analysis.md#71-是否试过去掉-persistent每个-wg-只执行一个-n-task)。本次只回答并补记已有实验，没有恢复暂停的GPU优化。
