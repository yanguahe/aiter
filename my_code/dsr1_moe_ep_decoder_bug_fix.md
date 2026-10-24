# DSR1 MoE EP decoder bug fix 记录

本文件按主题整理此前的问答；重复提出的问题合并为一份回答。后续每个问题的回答也同步追加到这里，结论变化时保留核对依据，并更新原有表述。性能的逐轮记录保存在 [dsr1_moe_ep_decoder_opt.md](dsr1_moe_ep_decoder_opt.md)，这里只引用解释问题所需的 GEMM1、GEMM2 和通信 kernel 数据。

更新时间：2026-10-14。当前生产实现保持精度正确的双 persistent 基线，全面 kernel 优化仍暂停。本次用户要求继续后，已完成 thread trace 的隔离诊断、归档及 GPU 空闲记录复核，详见第7节：入口等待由 LOAD 主导，同机输出目标对照指向 EP/P2P 有效输出流量对后续 loads 的影响；尚未唯一确定具体硬件队列或链路，也未宣称生产性能已经修复。3组结束检查出现其它 KFD 进程的 trace 已排除出耗时对照。四卡通信的独立说明及第9节保留，完整内容见 [moe_framework_wiki_ep4.md](moe_framework_wiki_ep4.md)。

## 1. 下一 N-task input prefetch 与上一 output TDM 的时序 race 是 bug 吗？设计上能否重叠？

**是实现 bug，设计上可以重叠。** 已确认的错误是 LDS 生命周期管理：output DS stores 复用了仍有其他 wave 读取的 final K tile 区域。prefetch 与 output TDM 同时存在本身不构成错误；需要保证输入、输出的 LDS 区域互不覆盖，并在复用区域前完成跨 wave release。

这个结论针对 2026-10-11/12 的单卡 `t192/w2x4/b4` 问题。后续四卡新增的 EP scatter 等待适配见第 4 节，两次工作的改动应分别理解。

### 修复了哪些问题？

GEMM1 exact 路径将 quant payload/ScaleA 放到已释放的 stage2 前部：

- `PITCH=65536`，output 的 LDS 范围为 `[131072,144128)`。
- penultimate carry fence 释放该区域。final K tile 使用 stage3 前部及 stage0 tail；next I0/I1 的区域也不覆盖这片 output，next I1 tail 为 `[179648,192896)`。
- 保留 next I0/I1 prefetch、下一任务的 input-ready fence，以及回填 stage2/3 前的 output-done fence；删除 exact quant 路径中的冗余 `tensor_wait(2)`。
- 保留原五 segment allocation。

GEMM2 将两个 M wave 的第一片 output 一起放入已释放的 stage2，`PITCH=60416`，第一片占 `52224` bytes。第一片 output 的 WG barrier 释放 final stage3 后，才写第二片 output。动态 slice LDS address 固定在 VGPR 中，防止 DS immediate 的 16-bit 范围截断；保留两 tile A cache、全部 K tile ScaleA cache及原 input ring pitch，删除 cached 路径中的冗余 wait。

为消除正确同步后的性能成本，GEMM1 quant amax 使用已取整 BF16 的 packed `V_PK_MAX_NUM_BF16` tree，后续仍执行原 scale clamp、peer reduction、E8M0 rounding 和 FP4 conversion。这项是性能优化，不应单独称为时序 bug 修复。

硬件依据：[MI400 Shader Programming Guide](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming%2365.txt) §4.10.1，印刷页197，规定同 wave Tensor 按序完成，跨 wave及相对其他 memory instructions 无序；§4.7，印刷页160–161，描述 LDS segment 和 DS immediate 范围。具体地址安全性还结合生成 ISA 与重复数值验证检查。

### T1536、T4608 相对旧无 drain 还差多少？

2026-10-12 已接入实际默认路径，并完成性能验收。下表为相对旧无 drain 的耗时中值变化，负值表示更快：

| tokens | data | GEMM1 | GEMM2 | MoE e2e |
|---:|---|---:|---:|---:|
| 1536 | random | −2.93918% | −2.24165% | −2.35408% |
| 1536 | const0 | −1.38267% | −1.56128% | −1.16042% |
| 4608 | random | −0.57487% | −3.39512% | −1.22223% |
| 4608 | const0 | −1.01528% | −2.65926% | −1.30496% |

每组在整机 GPU 空闲后做 `24 × 201` 相邻 A/B，保留全部样本，不丢首轮、不做 IQR filtering。12 项 median 和 paired median 均不回退，但单轮仍有波动，不能保证每次运行都更快。这里的 `MoE e2e` 是一次完整 eager MoE 的所有 GPU kernel 耗时之和，与后续四卡16层 graph replay 的 wall-clock 口径不同。

累计9120次 stage 重放的有效输出差异为0。T1536/T4608 exact 的 raw payload、ScaleA、GEMM2 均一致；部分非对齐 boundary 的 whole-allocation raw payload 有差异，不能声称所有边界 bitwise equal。packed amax 对75,497,472个值的验证包含全部 BF16 encodings、NaN/Inf/denormal，scale bits、E8M0 和 FP4 payload 均零差异。

依据：[默认路径验收报告](../.codex_tmp/t192_overlap_20261012/resume_evidence/default_route_report_local.json)、[正确性汇总](../.codex_tmp/t192_overlap_20261012/resume_evidence/production_summary.json)、[原始修复记录](e64_topk8_token_range_kernel_dispatch_head_vs_worktree.md)。这是当时单卡配置的验收结果，不能直接用来保证后续四卡 `_epscatter` 或当前新补丁的所有配置性能。

## 2. K7168 GEMM1 修了哪些问题，四卡 benchmark 才正确？ScaleA 是原有 bug 吗？

**这是原代码中已经存在的实现 bug，四卡接入后把它暴露了出来。** 问题出在 persistent GEMM1 fused quant 输出 ScaleA 与 A-preshuffled GEMM2 之间的布局契约上。

涉及的 GEMM1 kernel：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist
```

### 原版已经允许这个配置

修复前的代码基线为 `e749e6cef3392295f7a1c491a1d9f51a1b75bbb2`。其中的支持条件已经接受 `K7168/N4096/t192/w2x4/b4`、`stage1_act=1`、`stage1_quant_out=1`、`E64`、`cluster_n=4`、`cluster_m=1`，并选择 `launch_gemm_a8w4_tdm_fused_persistent`。

因此，该 kernel 的 fused quant 输出本来就应当满足 A-preshuffled GEMM2 的 ScaleA 布局要求。这次修复的是已有实现中的错误。

支持条件和 launcher 选择位于 [grouped_gemm_mxfp4.py](../aiter/ops/flydsl/grouped_gemm_mxfp4.py) 中的 `_supports_gfx1250_a_preshuffle_resolved()` 和 `flydsl_grouped_gemm_a8w4_masked()`。

### 错误地址公式与四卡规模无关

`t192/w2x4` 的每个 M wave 负责 96 行，`quant_wmma_rep=6`。对于一个输出 K128 分块，每行有 4 个 E8M0 scale bytes，所以每个 M wave 要写出 `96 × 4 = 384B` ScaleA。

原实现将这 384B 作为连续数据写出。GEMM2 则按 canonical M32 group 读取：每个 M32 group 内，排列该 group 的全部输出 K128 分块，然后才进入下一个 M32 group。原实现由此混排了 M group 与 K 分块。

以 `blk_m=0`、`wave_m=0`、`blk_n=0`、GEMM1 激活后输出宽度为 2048 为例：

- 输出共有 `2048 / 128 = 16` 个 K128 分块。
- 每个 M32 group 在一个 K128 分块上的 ScaleA 占 `32 × 4 = 128B`。
- 相邻 M32 groups 的 global stride 应为 `16 × 128 = 2048B`。

| ScaleA 数据块 | 原实现写入的 byte offset | GEMM2 布局要求的 byte offset |
|---|---:|---:|
| rows 0–31，第一个 K128 | 0 | 0 |
| rows 32–63，第一个 K128 | 128 | 2048 |
| rows 64–95，第一个 K128 | 256 | 4096 |

第一个 M32 group 恰好正确，后两个已经写到了错误位置。GEMM2 按自己的布局读取时，会取到不对应当前数据的 scale，从而产生数值错误。

这个地址公式没有 `world_size` 或 `tokens_per_rank` 条件，因此，相同 producer/consumer 组合在单卡上也存在同样的布局错误。这里对单卡的判断来自地址公式推导，本轮没有额外执行单卡复现。

### 为什么在四卡接入后暴露

原四卡 FP4 dispatch 路径没有接入这里的 A-preshuffle producer，实际使用 generic TDM kernel。本轮接入 GEMM1、GEMM2 双 persistent 后，才执行到了这段已有的错误输出逻辑。

四卡接入改变了该四卡配置实际选中的路径，暴露了原四卡路径没有执行到的布局错误；它没有改变 GEMM2 对 canonical M32 ScaleA 的要求。历史单卡是否曾执行同一组合，需要另外核对，不能据此称历史单卡完全没有覆盖。

### 修复方法

修复位于 [mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py) 的 `launch_gemm_a8w4_tdm_fused_persistent()` 内，具体是 `issue_quant_scale_output()`。

每个 M wave 的 384B ScaleA 改为三个 128B 的 M32 groups，按正确的 global stride 写出：

```python
q_dst_scale_dwpr = i32_n // 256
scale_rows32 = quant_wmma_rep // 2
scale_row32_bytes = 32 * 4
global_scale_stride = fx.Int64(q_dst_scale_dwpr) * scale_row32_bytes
global_scale_off = (
    fx.Int64(blk_m) // 32 + fx.Int64(wave_m * scale_rows32)
) * global_scale_stride + (
    fx.Int64(blk_n) // 256
) * scale_row32_bytes
```

对应的 global view 使用 `(scale_rows32, scale_row32_bytes)` 和 `(global_scale_stride, 1)`；LDS source view 使用同样的 shape 和 `(scale_row32_bytes, 1)`，使每个 M32 group 写到 GEMM2 预期的位置。

### 修复归属与结论范围

- **四卡 FP4 wire 接入属于新增功能支持。** 上游 Mori ScaleA 的 224B 有效宽度与 256B 实际 source pitch，也是在该输入接入中处理的独立问题。
- **GEMM1 fused ScaleA 写错布局属于已有 bug 的修复。** 本节的 M96/M32 问题发生在 GEMM1 输出端。
- 不能据此认定历史所有单卡结果都错了。需要确认当时是否实际执行了该 persistent fused quant 与 A-preshuffled GEMM2 的组合。

四卡精度验证的配置、结果和验证范围见 [dsr1_moe_ep_decoder_opt.md](dsr1_moe_ep_decoder_opt.md)。单卡相同组合的错误判断是静态地址分析结论，不能替代对应配置的运行验证。

### 指定 GEMM1 kernel 的直接修复与整条四卡路径的区别

对这个 K7168 GEMM1 本身，直接的数值正确性修复是上述 `issue_quant_scale_output()` 的 M96/M32 global layout。上游 Mori ScaleA pitch、inverse route `-1`、共同 expert alignment 属于另几处接入和边界处理；GEMM2 scatter ownership 和等待适配也发生在 GEMM2，不应说成指定 GEMM1 的修改。

四卡验证使用 `E256/topk8/hidden7168/inter2048`、1536 tokens/rank、16层，以及原 FP32 RefModel 的 `logits_diff/logits_tol`：

| seed | expert_balance | base logits_diff | fused logits_diff | logits_tol | 结果 |
|---:|---|---:|---:|---:|---|
| 0 | True | 0.1842963171 | 0.1842538070 | 0.3609625668 | PASS |
| 1 | True | 0.1838829516 | 0.1838064915 | 0.3609625668 | PASS |
| 2 | False | 0.2309923653 | 0.2310255178 | 0.3609625668 | PASS |

原 benchmark 的 seed0 验证和独立 no-profiler 运行均通过；后者保留原模型、routing 和 graph，时间值为0，不作为性能数据。正式性能命令的 `--acc_verify 0` 本身不验证精度。

证据：[seed0](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/accuracy_persistent_both16_noprof_seed0_tables.json)、[seed1](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/accuracy_persistent_both16_noprof_seed1_tables.json)、[不均衡路由](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/accuracy_persistent_both16_noprof_unbalanced_tables.json)、[冻结基线复验](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/accuracy_verified_baseline_seed0_tables.json)。

## 3. 上面的 ScaleA bug 为何之前 MoE 单测没测出来？

**已确认测试存在覆盖和 oracle 的盲区，但现有证据还不能唯一解释历史目标 case 为何 PASS。** 不能简单回答“从未跑过 t192”，也不能认定所有精度检查都绕过了错误路径。

1. 常规 pytest 的 geometry 不覆盖目标 fused quant 配置。[test_flydsl_grouped_gemm_gfx1250.py](test_flydsl_grouped_gemm_gfx1250.py) 中 `run_moe()` 默认 `E4/T8/topk2/model512/inter512/use_bias=True`；`test_grouped_a4w4_silu_matches_torch_ref()` 也使用小 shape 和默认 bias，不能验证 `E64/K7168/t192/w2x4` fused quant 输出。
2. 中间 stage hash 来自另一轮调用。`_run_grouped_via_fused_moe()` 先执行生产调用保存最终 `out`，随后安装 `stage_output_capture` 再调用一次获取中间数据；而 grouped 路径的 `_fuse_quant` 要求 `_stage_output_capture is None`。因此这些 stage hashes 来自关闭 fused quant、另做 standalone quant 的路径，不能证明生产 fused epilogue 满足 M32 布局。原生产最终 `out` 仍与 reference 比较，不能说最终精度检查被完全绕过。
3. [verify_gemm1_fused_apre_quant.py](verify_gemm1_fused_apre_quant.py) 的 `_decode_scale_rows()` 使用 `rows_per_tile = wmma_rep * 16`，t192 对应M96，其 self-test 也按同一布局编码和解码。这不是独立于 producer 的 M32 consumer oracle。不过脚本还有 standalone 和 e2e 比较，不能仅凭这一点解释全部历史 PASS。
4. const0 对 scale 错排不敏感；相同 scale 的错排可能不改变数值。但是历史 random case 是否存在这种退化，没有证据确认。

必须保留的反证是：2026-10-12 历史 profile 确实有 t192 `_persist`，random 和 const0 都有 PASS。旧 `production_exact_random.jsonl` 还报告 T1536/T4608、seeds0/7 的 `logits_diff≈3.38e-6`、`scale_mismatch=0`、`gemm2_byte_mismatch=0`。这些结果不能删除，也不能凭推测解释为容差过宽、缓存加载旧代码或所有 scales 都相同。

更充分的验证应独立按 GEMM2 的 canonical M32 契约检查**生产 fused quant** 输出的完整 ScaleA bytes，使用非恒定 scales 并确认实际 symbol和代码来源。现有新增四卡 producer oracle 验证的是上游 Mori 输入转换，不是 GEMM1 fused epilogue，应分别看待。

依据：[历史 exact 检查](../.codex_tmp/t192_overlap_20261012/resume_evidence/production_exact_random.jsonl)、[历史默认路径验收](../.codex_tmp/t192_overlap_20261012/resume_evidence/default_route_report_local.json)、[grouped 的 fused quant 条件](../aiter/ops/flydsl/grouped_moe_gfx1250.py)。

## 4. `_persist_epscatter` GEMM2 比普通 `_persist` 加了什么？为何约两倍？

两个完整 symbol：

```text
普通 GEMM2:
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist

EP scatter GEMM2:
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter
```

两版 K-loop 和 WMMA 计算量相同。`_epscatter` 将 route 加权、indexed 输出和跨 rank 回写融合进 GEMM2，增加的主要是 epilogue 与通信工作。

### 新增的逻辑

[gemm2_scatter.py](../aiter/ops/flydsl/kernels/mega_moe_gfx1250/gemm2_scatter.py) 的 `make_bf16_scatter_view()` 解析 Mori symmetric window；`load_bf16_scatter_rows()` 读取 route weights/destinations，处理无效 destination，计算目的行：

```python
peer = destination // ep_destination_stride
slot = destination % ep_destination_stride
destination_row = peer * slots_per_rank + slot
```

这些逻辑在当前 `run_task()` 中，同一 WG 的7个 N-task重复执行。epilogue 对每个输出元素增加 `acc = acc * ep_weights[wm]`，即 FP32 multiply。

普通版的完整 `192×256` output 分两片，每片96行；每片4个 wave 各写连续24行，合计8次 TDM stores，每 wave 共1次。scatter 使用32-bit row indices，每条最多8行，因此每片12次，合计24次；第二片 ownership 旋转4个 wave，使8个 wave 每个恰好3次。完整 WG 的7个 N-task 输出 TDM stores 由56次增加到168次。

每次 `store_bf16_scatter_group()` 还通过8次 `readlane` 提取 indices并构造 gather descriptor。输出的逻辑元素数未增加，descriptor、地址计算、发射和 P2P 完成开销增加。

[CDNA5 ISA](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt) §10.11.3.2，印刷页143，明确32-bit indices每条最多8个，16-bit最多16个，并支持 LDS-to-global scatter。

### scatter 接入怎样保证 overlap 的时序正确？

next I0/I1 在旧 output stores 之前由相应 wave 发出。普通版的下一任务 prologue 使用 `tensor_wait(1)`；scatter 根据每 wave 的3次 output stores，使用 `tensor_wait(3)`，随后 WG barrier，保证所有 requester waves 的输入已到达。第二片 ownership 旋转是使这个逐 wave计数成立的必要条件。

之后先消费 stage0，使上一 output 继续 drain，再在复用 stages2/3 前执行 `pipeline_fence(outstanding=0)`。两版在这个位置本来都有 full fence，本次不是新增了这个 fence 位置；scatter/P2P 增加的 outstanding work 可能延长原有等待。

这套推导依赖同 wave Tensor 按序完成以及各 wave 实际发出的 store 数量；不能只等待一个 wave 来证明整个 WG 的输入或输出完成。不能通过去掉必要 wait/barrier 来追求速度。

### 冻结基线实测与结论范围

来源：[bench_verified_baseline_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_verified_baseline_tables.json)。这里只列 GEMM1、GEMM2 和相关通信 kernel。

| mode / kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| base `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.44468 | 42.25 | 98.23645 | 98.08136 | 98.35873 | 99.10218 |
| base `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 58.98674 | 42.25 | 58.54060 | 58.75255 | 59.27709 | 59.37673 |
| fused `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.35451 | 42.75 | 97.91154 | 97.96423 | 98.53923 | 99.00302 |
| fused `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 119.68991 | 42.75 | 120.23997 | 118.95344 | 121.00330 | 118.56292 |
| base `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 79.05920 | 42.25 | 80.33405 | 79.59988 | 77.99378 | 78.30910 |
| fused `ep_combine_fused_0` | 17.71442 | 42.75 | 17.54300 | 17.78091 | 17.65142 | 17.88235 |
| fused `ep_combine_fused_sync_0` | 7.67293 | 42.75 | 8.35316 | 8.58677 | 6.96937 | 6.78244 |

GEMM2 增加 `60.70317 μs`、`+102.90986%`，为普通版的约 `2.0291×`，四个 rank 均有相近差距。没有分项计时，不能将全部增加值精确归因于某一项，也没有证明 occupancy 或 spill 翻倍。

同轮16层 e2e 中值为 base `7029.27325 μs`、fused `6347.07750 μs`，fused 耗时降低约9.71%。部分 combine 工作进入 GEMM2，因此单独 GEMM2 的时长增加不等于整条 fused 路径性能更差。该轮 fused mean/max 有长尾，完整统计保留在性能文档中。

metadata hoist、descriptor复用和 split `2+4` 候选已单独保存，当前生产仍使用精度正确基线。`2+4` 最近配对结果有约0.64%–1.00%的 fused e2e收益，但暂停前尚未完成全部候选验收，不能当成已接入的最终优化。2026-10-14 的 ATT 及隔离对照已确认：最大的新增等待暴露在下一 N-task 入口的 LOAD wait，EP/P2P 有效输出目标/流量是主要影响因素；单纯去掉 window geometry 读取没有消除问题，具体测量与结论范围见第7节。

## 5. 224B/256B ScaleA、dropped route `-1`、192 expert alignment 都是四卡特殊规模 bug 吗？

**三项性质不同。** 之前将它们笼统称为“修 bug”不准确，核对后按下表分类：

| 项目 | 触发条件 | 归属 |
|---|---|---|
| 有效 ScaleA 224B / source pitch 256B | 实际 row pitch 大于有效宽度 | 接入 Mori prequantized 输入时补充 padded stride 支持 |
| inverse route map 的 `-1` | 有效扫描范围内包含无效 grouped row | 已有 helper 的边界处理缺失 |
| 两阶段统一192行 alignment | 将一个或两个 GEMM tile 改为192 | tile 配置匹配；原256/256组合兼容 |

### Mori source pitch

7168维每 token 有 `7168/32=224B` 有效 ScaleA，Mori 的实际 source pitch 是256B，这是合法 padding。读取地址应按256B推进，仅读取224B有效内容。

原 producer 服务本地 BF16 quant 的紧凑 `token_scale`，按224B计算 row stride 对那个输入正确；原 `a_preshuffle + prequantized_scale` 明确抛出 `NotImplementedError`。本轮新增 `src_scale_bytes_per_row/src_scale_dwords_per_row`，caller 传 `int(token_scale.shape[-1])`，补上 padded Mori 输入支持。不能说 Mori 生成了错误数据，也不能说原有紧凑输入公式本来就错。

若将 padded输入误按有效宽度寻址，第二个 token 就可能读错，不需要四卡或1536 tokens才触发。

### 无效 route 的 `-1`

`build_moe_invert_route_rows_module()` 原来只判断 `route < valid_route_count`，随后按读取的 row写入 inverse map。扫描范围内若 row 为 `-1`，还需要先判断 `fx.Int32(row) >= 0`，否则没有满足合法 row 的写入前提。

`-1` 可以表示不属于本 rank的 expert route，并不限于 capacity drop；均衡路由也可能出现。单卡全 local 的普通路由往往不触发，不代表 helper 边界处理完整。没有证据证明每轮 benchmark 都已因这项产生实际数值错误。

### 192 expert alignment

原 EP tuned row 使用 GEMM1/GEMM2 `256/256`，属于兼容配置。改为 persistent t192 后使用 `192/192`，使 expert row排列与两阶段 tile相匹配。

原代码已有：

```python
_align_m = max(tile_m, tile_m2)
if _align_m % tile_m or _align_m % tile_m2:
    raise ValueError(...)
```

256不能整除192，只改一阶段为192会被现有 guard拒绝。本轮没有新增这个 guard，原256/256也不存在因此静默跨 expert 的 bug。统一192是接入新配置所需的调整。

### 上游 producer 的独立验证

[verify_moe_fp4_ep_apreshuffle.py](verify_moe_fp4_ep_apreshuffle.py) 在单卡、64 tokens、topk8下使用独立 PyTorch gather + reshape/permute oracle，检查 payload、canonical M32 ScaleA全部 bytes，包括 padding和 dead-tail。

| feature_dim | source_scale_pitch | experts | 无效 route / dead-tail | 每项 replay | 结果 |
|---:|---:|---:|---|---:|---|
| 7168 | 256 | 64 | 有 | 3 | PASS |
| 7168 | 224 | 4 | 无 | 3 | PASS |
| 2048 | 128 | 4 | 有 | 3 | PASS |
| 2048 | 64 | 4 | 无 | 3 | PASS |

这证明相应上游 producer case 的处理正确，也说明触发条件不限于四卡规模；它不替代 GEMM1 fused epilogue 的 M32独立验证。

源码：[moe_fused_route_quant_scatter.py](../aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py)、[moe_kernels.py](../aiter/ops/flydsl/moe_kernels.py)、[grouped_moe_gfx1250.py](../aiter/ops/flydsl/grouped_moe_gfx1250.py)。

## 6. 手动四卡 benchmark 为什么 kernel 名仍与原来一样，没有 `_persist`？

此前确认的原因是：benchmark 脚本从 refactor目录运行，但实际导入了 `/app/aiter` 的 main版本。交互容器的 `/etc/bash.bashrc:72` 设置了：

```bash
export PYTHONPATH=/app/aiter:${PYTHONPATH}
```

当时日志明确显示 `/app/aiter/aiter/jit/module_aiter_core.so`，因此选中 GEMM1 `t256/w1x4/b4`、GEMM2 `t256/w2x2/b3`。refactor目录的源码和配置仍与已验证基线一致。

在同一容器终端先执行：

```bash
cd /app/aiter_moe_a4w4_pr_refactor_wt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import aiter; print(aiter.__file__)'
```

应输出 `/app/aiter_moe_a4w4_pr_refactor_wt/aiter/__init__.py`，然后再执行原 benchmark 命令。已验证这样修改导入路径后指向正确配置，不需要因此重新安装，也未修改 `/etc/bash.bashrc`。

正确 symbol 为第2节的 K7168 GEMM1和第4节的两个 K2048 GEMM2。仅从当前目录或安装成功，不能推断进程实际导入了哪一份 AITER。

## 7. `_persist_epscatter` 性能成本很大，能抓 GEMM2 thread trace 吗？

**可以，使用 `rocprofv3` 的 Advanced Thread Trace (ATT)，按完整 kernel 名筛选。** 初次采集保留四卡 Mori EP/P2P workload，仅对一个 rank 启用 ATT。单卡本地 C输出不能复现当前 `_epscatter` 的跨 rank回写成本。

### 2026-10-14 最初的工具检查（后续实际兼容性结果见下文）

在 `f01-1/dsr1_ep4` 中只读确认下列路径存在：

| 工具 | 路径 / 检查结果 |
|---|---|
| 自编译 rocprofv3 | `/data/yanguahe/code/wk_sp1/rocprof-install/bin/rocprofv3` |
| profiler环境 | `/data/yanguahe/code/wk_sp1/rocprof_env.sh` |
| decoder | `/data/yanguahe/code/wk_sp1/decoder_new/librocprof-trace-decoder.so`，`ldd`未见缺失依赖 |
| COMGR | `/data/yanguahe/code/wk_sp1/comgr_new/libamd_comgr.so.3` 指向 `libamd_comgr.so.3.5.0` |
| rocprofv3版本 | `1.1.0`，revision `07ce631b1384f6c9aefc6244e8c481fa48ae8062`，报告 ROCm `7.12.0` |
| 当前非交互导入 | `/app/aiter_moe_a4w4_pr_refactor_wt/aiter/__init__.py` |

未经 source 的 shell 默认找到的是 `/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin/rocprofv3`。最初准备使用上述自编译工具，但实际运行证明它不适配 `dsr1_ep4` 的 ROCm 10.1 runtime，已改用容器配套 profiler/decoder。下面保留最初参数示例作为历史记录；当前执行入口及兼容性结果见本节后续补充。

这次只检查路径、版本和 `--help` 的 ATT参数。该 profiler与当前容器 runtime 的实际采集兼容性、此 kernel 的解码完整性还需一次 capture 验证，不能仅凭版本输出宣布成功；也没有启动工具链重建。

### 为什么要使用单独 runner？

原 `DeviceMoEPipeline.bench()` 即使设置 `--profile_table 0` 仍执行 `torch.profiler`，该选项只控制表格打印。外层 ATT 采集应避免再套 Kineto。

新增 [trace_mega_moe_without_profiler.py](trace_mega_moe_without_profiler.py) 只替换 benchmark 的计时/profile方法，保留原模型、routing、kernel选择和graph；全部rank运行同一runner。其时间统计为0，不能作为性能记录。新增 [trace_mega_moe_rank.sh](trace_mega_moe_rank.sh) 通过 `LOCAL_RANK` 仅给选定rank加 `rocprofv3`，其他rank照常参与通信。

不应把 `rocprofv3` 直接套在 `torchrun` 外面来假设只采了 rank0；rank wrapper使进程选择明确。`--save_trace 1` 导出的是 Kineto/Chrome kernel timeline，不是指令级 thread trace。

### 具体采集方法

将上述两个 helper传入 refactor仓库的 `my_code/`。每次采集前在 **f01-1主机**运行 `/data/yanguahe/code/gpu_users.sh`，检查全部GPU/KFD进程、busy及VRAM；脚本不存在时用 `fuser -v /dev/kfd` 和可用的 GPU工具。若忙则每6秒重新检查，空闲后才开始；采集期间不同时轮询，不杀其它进程。结束后再次核对残留进程。

以下命令在 **dsr1_ep4容器**执行。保留原 `E256/topk8/7168/2048/1536 tokens/rank/16层`，将 `--combine` 设为 `fused` 确保执行 `_epscatter`，迭代减为2仅用于 trace：

```bash
cd /app/aiter_moe_a4w4_pr_refactor_wt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import aiter; print(aiter.__file__)'

trace_root="$PWD/my_code/thread_trace_runs/gemm2_persist_epscatter_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$trace_root"
chmod u+x my_code/trace_mega_moe_rank.sh

# 本轮只采 SIMD0。后续每轮重新检查 GPU 空闲后再设置 simd=1/2/3。
simd=0
capture_dir="$trace_root/thread_trace/simd${simd}/kernel/rpf_v3"
mkdir -p "$capture_dir"
config="$trace_root/att_simd${simd}.yaml"
cat > "$config" <<YAML
jobs:
  - kernel_include_regex: '^a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter$'
    kernel_iteration_range: "[8]"
    output_file: out
    output_directory: "$capture_dir"
    output_format: [csv]
    truncate_kernels: false
    sys_trace: false
    advanced_thread_trace: true
    att_target_cu: 1
    att_shader_engine_mask: "0xf"
    att_simd_select: "$simd"
    att_buffer_size: "0x10000000"
    att_library_path: ["/data/yanguahe/code/wk_sp1/decoder_new"]
YAML

AITER_ATT_CONFIG="$config" \
AITER_ATT_LOCAL_RANK=0 \
MORI_V2_KERNEL_BACKEND=hip \
MORI_EP_TOKOFF_EXT="${MORI_EP_TOKOFF_EXT:-0}" \
MEGA_DISPATCH=mori FLYDSL_DUMP_IR=0 \
MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 \
AITER_FORCE_A8W4=0 AITER_MOE_EXPERT_BALANCE=True \
torchrun --standalone --nproc_per_node=4 --no-python \
    ./my_code/trace_mega_moe_rank.sh \
    -e 256 -k 8 -hd 7168 -id 2048 --layers 16 \
    -tpr 1536 -q a4w4_mxfp4 --combine fused \
    --warmup 5 --iters 2 --prof_replays 0 \
    --acc_verify 0 --profile_table 0 --save_trace 0 --dispatch_wire fp4 \
    > "$trace_root/simd${simd}.log" 2>&1
```

**每个 SIMD 是独立一次四卡运行，开始前均需重新完成主机空闲检查。** 本次结束后再将 `simd` 改为1、2、3，分别重新生成YAML并采集。没有使用 `HIP_VISIBLE_DEVICES=0`，因为全部rank必须继续访问四卡。helper明确关闭 `AITER_SMI_MONITOR`，避免无效的零时间SMI replay和采集期间并发GPU轮询。

`kernel_iteration_range: "[8]"` 用于先选取一次目标 dispatch打通采集，它不是 persistent 内部 N-task编号，也不能据此称为 warmup后的稳态。后续应依据该SDK实际dispatch计数选择 warmup后的 graph replay occurrence；如果需要比较两个版本，应保持相同rank、layer/routing和replay位置。第一次可以只跑 SIMD0、SE mask `0x1`减小数据量，打通后再按上面的 SIMD0–3、SE mask `0xf`补齐。

`att_target_cu` 是 profiler的硬件单元选择参数，与 EP rank和device编号不同。根据本地 [MI400 Shader Programming Guide](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming%2365.txt) 第8章，印刷页287，每个SE同时只在一个WGP采集指令执行；即使覆盖所有SIMD，也不能把它描述为所有GPU、所有WG的完整指令记录。

### 怎样判断采集成功，以及 trace 重点看什么？

每个采集的 SIMD都应有非空 `.att`、可读取 `code.json`、decoded wave JSON，且 `code.json` 包含完整目标symbol和非空指令列表。检查 `.long/v_illegal`，日志中的 buffer overflow、decoder失败或空 wave也必须处理。仅 `rocprofv3` 返回0不够。

现有 [analyze_att_capture.py](analyze_att_capture.py) 可检查已解码trace的wave/occupancy和clock分布：

```bash
python3 my_code/analyze_att_capture.py --dir "$trace_root" --no-plot
```

详细指令时序由decoded wave与 `code.json` 关联。重点比较：

- epilogue 的 `readlane`、destination计算及descriptor构造，24次indexed stores的发射间隔；
- `s_wait_tensorcnt`及WG barrier的等待，尤其跨 N-task复用stages2/3前的output-done fence；
- next I0/I1 到达和stage0计算是否实际覆盖上一轮output等待；
- route加权的VALU工作、VMEM依赖，以及WMMA是否出现额外空档。

ATT能显示所采wave的issue/wait和执行空档；仅凭长wait不能唯一确定是P2P链路、cache、descriptor还是其它rank造成的，需要结合普通 `_persist` 对照、通信timeline与必要的额外诊断。trace本身会扰动运行，其耗时不应替代此前空闲条件下的正式性能数据。

此前问答交付的是采集方法与 helper。本轮用户已明确要求执行采集；实际进展记录如下，尚未验证成功的步骤不能写成已完成结果。

### 2026-10-14 实际采集进展与失败证据

冻结性能基线的普通 `_persist` GEMM2 为 `58.986739 μs`，`_persist_epscatter` 为 `119.689910 μs`，增加 `60.703171 μs / 102.91%`。同一组四卡16层 e2e 为 `7029.27325 / 6347.07750 μs`，fused 仍快约 `9.71%`；原因是 combine 的部分工作融合进 GEMM2，不能只用 GEMM2 kernel 时间判断整体收益。原始数据：[bench_verified_baseline_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_verified_baseline_tables.json)。

最初在 `f01-1/dsr1_ep4` 执行 dispatch-based ATT 的尝试失败；以下是失败路径的证据，后续 device-based ATT 已成功：

- 自编译 `rocprofv3 1.1.0 / ROCm 7.12` 在 `import torch` 阶段触发 `Configuration request occurred outside of valid rocprofiler configuration period`，这是 profiler/runtime 兼容性失败，不能归因于 GEMM2。
- 换用容器内配套 `rocprofv3 1.3.5`，revision `44405e956f3708ccd53f244666f7fd768eea6bbc`，以及 `_rocm_sdk_devel/lib/librocprof-trace-decoder.so`。已有 `comgr_new` 前置；同时前置 devel/core library 解决 `libhsa-amd-aqlprofile64.so` bare-name 加载。工具导入和 HSA 初始化检查通过。
- case `fused_simd0_iter129_aql` 加载了完整目标 symbol 的 code object 36，随后出现 `device-memory ring-buffer requested but profiling requires system-memory InterceptQueue`、GPU null-address fault 和 queue sync timeout。queue interception 是当前嫌疑，尚未证实其因果，也不能据此断言 cluster dispatch 不受支持。日志：[fused_simd0_iter129_aql_capture.log](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/fused_simd0_iter129_aql_capture.log)。
- 新增 [trace_mega_moe_agent.cpp](trace_mega_moe_agent.cpp)，使用 SDK device-based ATT API，没有配置 kernel-dispatch interception service。由 Python runner 对稳态原 graph replay 显式 start/stop。原四卡 workload 已成功完成，两版 SIMD0–3 均获得有效 raw 和 decoded wave，见下面的验收表。
- 新增 [decode_mega_moe_agent.cpp](decode_mega_moe_agent.cpp)，离线保存 wave 的 timestamp、stall、duration、PC/code-object、occupancy、realtime 和 decoder INFO；有效 trace 还需完整 ISA 映射、目标 symbol 验收和 buffer-loss 检查。

原目标机器出现其它 GPU/KFD 工作负载，原四卡任务在 tmux 中每6秒检查一次整机空闲，空闲前不启动采集、不杀其它任务。另对映射机器做只读可用性检查；`f01-2` 在本轮恢复四卡空闲，因此创建独立诊断容器 `codex-dsr1-gemm2-att-20261014` 和独立目录 `/home/yanguahe/codex_dsr1_gemm2_att_20261014`，没有覆盖已有仓库或其它容器。仅复制原实现必需的 AITER 源码/已构建 extension、Mori Python/CCO runtime 和采集 helper。迁移包 SHA256 为 `c9a668a317ec99bfe05afa7126b7b2d2dbebda2a7ca8262999cb82ad6da7db41`；两端核对一致。

`f01-2` 诊断容器是已有 `ee6d7ff29b6e` 镜像的独立实例，PyTorch `2.11.0+rocm10.1.0a20260822`，FlyDSL `0.3.4.1`，配套 profiler revision `d134b7823f61f46816022525b98f128f5128ded9`。最初备用等待曾取消，后来恢复空闲后完成了14组生产对照、隔离及确认采集，各组前后均记录整机空闲。原机生产 trace / split-wait 是主证据，备用结果是备用机内部的因果对照。两机 runtime 版本不同，不能跨机相减 wave cycles 或混作原机性能。备用执行脚本：[run_capture.sh](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/run_capture.sh)。

### 当前静态证据（不能替代 thread trace）

两份 code object 的反汇编均无 `.long/v_illegal`。最初先提取普通 `_persist` 的 FlyDSL cache 和失败运行加载的 scatter code object做静态检查；随后成功采集两版实际运行的目标 code objects，进一步核对目标 symbol、PC 和指令流。各次运行固化的 Mori window 地址不同，不能要求完整 code object 的 SHA256 相同；控制流、PC和指令形状保持一致。

| 项目 | 普通 `_persist` 缓存 | `_persist_epscatter` 实际 code object |
|---|---:|---:|
| LDS | 305152 B | 305152 B |
| VGPR | 383 | 412 |
| SGPR | 98 | 106 |
| VGPR / SGPR spills | 0 / 0 | 0 / 0 |
| private segment | 0 B | 0 B |
| 静态 WMMA 指令数 | 216 | 216 |
| 静态指令数（不含 padding） | 2578 | 2881 |

新增输出逻辑包含 route weights、destination 映射和 descriptor 构造。完整192×256 tile 的动态 TDM output stores 从每 WG `8` 增至 `24`，每 wave 从 `1` 增至 `3`；7个 N-task 合计从 `56` 增至 `168`。32-bit gather 每条最多8个 indices，这是 ISA §10.11.3.2，印刷页143规定的限制；源码每 store 提取8个 row indices。跨 N-task 的 input-ready threshold 从普通 `tensor_wait(1)` 调整为 scatter `tensor_wait(3)`，stages2/3 复用前仍有 full wait/barrier。相同 LDS 和无 spill 不等于已证明 occupancy 相同，也不能证明增加的60.70μs全部来自 store/P2P。需要 decoded wave 判定 output issue、input 到达、fence 和 WMMA空档的实际时间关系。

证据：[static_comparison.json](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/static_comparison.json)、[普通版反汇编](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/base_cached.disasm)、[scatter 版反汇编](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/fused_failed_code36.disasm)。生产 persistent 源码 SHA256 保持 `165c0b8eb3c2578555008f730776712754b7f7fe54539fecdca4dbe76d3e7f85`，`gemm2_scatter.py` 保持 `d482d6c5a92245cd84625b761ed076811155c9f95a12e9100bb2e1ee85fbb8e5`。

### 2026-10-14 成功采集与归档

实际目标为 `f01-1` 的 `dsr1_ep4` 容器 `/app/aiter_moe_a4w4_pr_refactor_wt`。四个 rank 均执行原 E256 / topk8 / hidden7168 / intermediate2048 / T1536 / 16层、balanced routing、seed0、Mori、FP4 dispatch workload；仅 rank0 配置 device-based ATT。warmup5 后采一轮原 graph replay。runner关闭 Kineto，只报告0时长；本轮不把 ATT 时序作为新的性能 benchmark。

已验证工具为容器配套 SDK `1.3.5` / revision `44405e956f3708ccd53f244666f7fd768eea6bbc`，ROCm `10.1`；decoder为 `_rocm_sdk_devel/lib/librocprof-trace-decoder.so`，COMGR为 `comgr_new/libamd_comgr.so.3.5.0`，objdump为 `llvm-project/mlir_install/bin/llvm-objdump`。PyTorch `2.11.0+rocm10.1.0a20260811-bkc.20260820`，FlyDSL `0.3.4.1`，Mori `1.2.3.post2.dev43+g6cd0ca41b`。没有重建或更换生产工具链。

| SIMD | 普通版 raw ATT / 目标 wave / 有效 wave | scatter版 raw ATT / 目标 wave / 有效 wave | 两版每有效 wave 的 N-task数 | 验收 |
|---:|---|---|---:|---|
| 0 | 4 / 674 / 128 | 4 / 804 / 128 | 7 | PASS |
| 1 | 4 / 704 / 128 | 4 / 708 / 128 | 7 | PASS |
| 2 | 4 / 662 / 128 | 4 / 692 / 128 | 7 | PASS |
| 3 | 4 / 646 / 128 | 4 / 818 / 128 | 7 | PASS |

每 case 的4份 raw对应 SE0–3，全部非空；code.json可读，target symbol准确，PC映射无缺失，反汇编无 `.long/v_illegal`，decoder `INFO_RECORDS=0`，raw chunks 的 `flags=1`（END），没有 buffer-full 位2/4。decoder保留了 event 的 cases 未出现 `PACKET_LOSS` 或 `TT_STALL_BEGIN/END`。目标中包含未执行 WMMA 的 capacity-tail/empty expert wave；分析有效计算时只使用执行 WMMA 的128个 wave，不能用全部目标 wave 的中值。

原始采集结束的 SDK cleanup有 `Failed to destroy queue: 4107` warning；未发现本次测试的残留 GPU 进程。该 warning保留在日志中，不作为生产 GEMM2 的故障证据。后续完整空闲记录复核发现：普通版 SIMD3 graph、普通版 SIMD0 eager，以及原机 `global_window` 补采的结束检查出现其它 KFD 进程，不能排除中途重叠。它们的 raw / ISA / decoder 验收仍通过，但不再纳入耗时对照，详见本节末尾的空闲记录复核。

复现入口是 [run_capture.sh](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/run_capture.sh)，在远程主机 task root中执行：

```bash
bash run_capture.sh fused 0 0xf 129 fused_device_simd0_graph_b device graph
bash run_capture.sh base  0 0xf 129 base_device_simd0_graph    device graph
# SIMD1–3分别以对应SIMD号和新的case label执行；8个已采case保留完整命令与manifest。
```

`device`模式直接调用 SDK start/stop；YAML中的 `kernel_iteration_range=129` 没有被device API使用，实际采集区域由 runner圈选，不能声称这里选择了第129次 GEMM2。graph replay中有16层，分析器依靠实际 code object/symbol筛选 GEMM2 wave。`att_target_cu=1`，SE mask `0xf`，每SE buffer `0x10000000`。

完整 raw、code objects、decoded wave、code.json和转换后的目标 wave 已保存到本地 [gemm2_att_complete.tar.gz](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gemm2_att_complete.tar.gz)，`1056372189` bytes；SHA256 `f8054abc70be67b5bf837006e5a62dd626a1cb241603bec67ea2ef36e33d6014`，本地/远端核对一致。验收索引：[capture_manifest.json](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/capture_manifest.json)。远端 task root为 `/data/yanguahe/code/wk_sp1/.codex_tmp/f01_1_dsr1_gemm2_att_20261014`。

### trace证实的主要瓶颈：下一 N-task 首条 WMMA 之前

scatter版跨 N-task入口的 ISA为：

```text
0x29fc  s_wait_tensorcnt 0x3
0x2a00  s_wait_loadcnt_dscnt 0x0
0x2a04  s_barrier_signal -1
0x2a10  s_barrier_wait 0xffff
        ... LDS reads / first WMMA ...
```

普通版的对应位置为 `0x26f0 s_wait_tensorcnt 0x1`、`0x26f8 s_wait_dscnt 0x0`、WG barrier；没有 LOADcnt drain。这是当前两版生成 ISA 的实际差异，虽源码两处都调用 `workgroup_barrier()`，其backend会按当前 memory dependency发 wait。

| SIMD | 普通版有效wave span中值 / cycles | scatter版有效wave span中值 / cycles | scatter `0x2a00` 等待中值 / cycles | 等待平均 / cycles |
|---:|---:|---:|---:|---:|
| 0 | 73339.5 | 237500.5 | 17848.5 | 19286.36 |
| 1 | 71460.5 | 235633.5 | 17107.5 | 18151.63 |
| 2 | 73682.0 | 239310.0 | 17377.5 | 18958.83 |
| 3 | 排除：结束检查有其它 KFD 进程 | 238180.5 | 18357.5 | 19361.29 |

每SIMD该wait命中768次，即128个有效wave × 6次任务切换。SIMD0按真实persistent outer backedge切分7个N-task，得到下列timestamp区间的平均值：

| 每 N-task 区间 | 普通版 / cycles | scatter版 / cycles | 增量 / cycles |
|---|---:|---:|---:|
| 首条 WMMA 前 | 1858.83 | 20119.92 | 18261.09 |
| 首条至末条 WMMA | 7728.59 | 10165.15 | 2436.55 |
| 末条 WMMA 后至下一task/退出 | 952.64 | 1433.10 | 480.46 |
| N-task 总跨度 | 10540.06 | 31718.16 | 21178.10 |

首条WMMA前区间占新增task跨度约86.23%。`0x2a00` 每有效wave的6次wait累计平均115718.17cycles，约占新增7-task wave跨度的78.06%；这是采样指令时序内的比例，不能说正式benchmark的60.70μs中精确有78%属于这个wait。

representative wave的一次切换中，`s_wait_tensorcnt 3` 从4758865至4758868即通过；下一条组合wait从4758868至4790474，阻塞31606cycles，WG barrier随后只约143cycles。证据：[fused SIMD0 representative trace](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/fused_simd0_representative_trace_v3.txt)、[普通版对照](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/base_simd0_representative_trace_v3.txt)、[boundary_analysis.json](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/boundary_analysis.json)。

scatter每wave每task的3次输出store，从第一条到最后一条的发射跨度平均约770cycles；末条WMMA至首store平均636cycles。store快速发射并不代表P2P已经完成。生产版 stage0计算后、复用stages2/3前的 `0x2e50 s_wait_tensorcnt 0` 中值3cycles；在此采样中输出成本主要已暴露于更早的任务入口，不能继续将主要新增等待描述为stages2/3处full drain。kernel末尾 `0x6570 s_wait_tensorcnt 0` 还承担最后输出的完成等待，必须单独考虑。

有效计算wave的采样WGP/SE overlap两版均最多2个所选SIMD wave同时存活；没有发现采样位置的有效wave occupancy从2降到1。静态metadata无spill也与trace一致地排除了明显scratch流量。但这只约束采样位置，不能据此证明全GPU的occupancy、调度和cache状态完全相同。

静态图：[gemm2_att_phases.png](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gemm2_att_phases.png)、[PDF](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gemm2_att_phases.pdf)。

### load / LDS / P2P 的机制判断与限制

当前 `make_bf16_scatter_view()` 在每个 `run_task()` 内解析Mori window，生成 `0x1e8c flat_load_b32`（stride4G）和 `0x1ef0 flat_load_b64`（winBase）；`load_bf16_scatter_rows()` 又发出 `0x20d8–0x212c` 的8条 `global_load_b32`，其中6条weight、2条destination vectors。它们的返回值主要在epilogue使用，但下一任务的 WG barrier已先要求 LOADcnt/DScnt归零，形成进入下一轮计算前的阻塞。

[Shader Programming Guide](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming%2365.txt) §4.3.7.2.3，印刷页89–90，明确每条FLAT同时计入DScnt和LOADcnt，两部分分别完成；§2.4，页25–26，规定不同memory类型和跨wave无序。TDM output不直接计入LOADcnt，因此看到LOAD/DS组合wait不能直接解释成“TENSORcnt等待”。[Tensor DMA资料](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/TX/mi400_tensor_dma%2372.txt) §7.3.2.1，页12–13，记录Addresser与TDM共享arbitration；故scatter输出、普通loads以及LDS路径的资源竞争可能延长这个入口等待。但仅靠当前trace尚不能把具体队列/事务的延迟唯一归因于P2P。

额外采集了同步后单独圈选的真实 eager GEMM2：两版raw/ISA/decoded验收通过，各8个有效wave、每wave7task。scatter同一组合wait仍有13380cycles中值、14170.52cycles均值，其wave span中值213519cycles。普通 eager 的结束检查出现其它 KFD 进程，已排除它与scatter的耗时比较，只保留ISA/流程证据。scatter case 保留真实EP输入和live window，但运行额外eager layer并在kernel前后sync，不能当作原graph稳态性能。证据：[scatter eager analysis](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/fused_eager_simd0_analysis_v3.json)、[普通 eager analysis（耗时排除）](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/base_eager_simd0_analysis_v3.json)。scatter本身的长等待说明问题不限于graph replay。

ATT中普通/scatter有效wave跨度比约3.2，正式profile kernel时间比约2.029；两者不是同一指标。每SE只有一个WGP记录详细指令，且trace本身会扰动调度/时钟；不得把采样wave cycles转换成kernel wall-clock替换58.99/119.69μs基线。

### 2026-10-14 隔离诊断结果：LOAD 主导，geometry 读取不是唯一原因

诊断副本只由 `AITER_ATT_DIAGNOSTIC` 显式启用，不修改生产 kernel。`split_wait` 保留原 LOAD/DS full drain 和 WG barrier，拆成 `s_wait_loadcnt 0; s_nop 0; s_wait_dscnt 0`。原机 graph 采集得到128个有效wave、每wave7个N-task，768次任务切换：

| 原机拆分入口 | PC | duration 中值 / cycles | 平均 / cycles | 最大 / cycles |
|---|---|---:|---:|---:|
| `s_wait_loadcnt 0` | `0x2a00` | 17763.5 | 19253.4896 | 52414 |
| `s_wait_dscnt 0` | `0x2a08` | 1 | 1 | 1 |

有效wave span中值236151.5cycles，与原scatter的237500.5cycles接近。**在保留同步语义的这组实验中，入口阻塞由 LOAD 等待承担；先完成 LOAD 后，DS没有剩余等待。** 这不意味着DS事务本身始终只有1cycle，也不能推广到其它PC的所有DS waits。证据：[原机split analysis](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/accepted/fused_diag_split_wait_simd0_graph/analysis_v3/analysis.json)、[代表wave](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/split_wait_representative_trace_v3.txt)。

`global_window` 将两条FLAT geometry读取换成GLOBAL；`const_geometry` 在启动前用当前进程已加载的HIP runtime，通过 `hipMemcpyDefault` 读取真实 `ccoWindowDevice` 的前16B，再把 `winBase/stride4G` 传为constexpr。后者kernel不再加载这两项geometry，仍使用真实EP window及原route输出。`hipPointerGetAttributes` 各rank均返回device memory、`host_pointer=null`、`is_managed=0`；不能把原window说成host memory。证据：[rank0 geometry](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/accepted/fused_diag_const_geometry_eager/window_geometry_rank0.json)。

原机 `const_geometry` 补采前后均空闲，128个有效wave、每wave7task，入口 `0x2a78` wait中值17713.5cycles、均值19046.8255cycles，wave span中值242035cycles。去掉geometry加载后长等待仍在。原机 `global_window` 补采虽raw/ISA验收成功，但结束检查出现其它任务，已排除其耗时；两FLAT改成两GLOBAL的ISA差异仍可核查。

备用机graph的同机对照如下，均128个有效wave、每wave7task，前后空闲；表中duration/跨度均为ATT采样cycles：

| case | 有效wave span中值 | 关键等待中值 | 解释 |
|---|---:|---:|---|
| `fused_diag_original` | 232782.5 | 入口17373 | 原生产scatter |
| `fused_diag_split_wait` | 234071.5 | LOAD17898；DS1 | 原机拆分结果得到复现 |
| `fused_diag_global_window` | 235199 | geometry使用点16432.5；入口2 | 等待移到 `0x21d0 s_wait_loadcnt 9`，随后 `v_readfirstlane` 使用window base |
| `fused_diag_const_geometry` | 237084 | 入口17315 | 不再读geometry，仍有长等待 |
| `base_diag_original` | 75038.5 | — | 普通GEMM2对照 |

因此，“每个N-task重复读取window metadata”不能独自解释约两倍回退。改变地址空间可以移动等待PC，却没有缩短这组整体wave跨度；仅把源码hoist、descriptor cache或减少两次geometry读取也不能当作已经证明的解决方案。

最初两种副本的四卡非profile运行exit0；首个 `--acc_verify 1`诊断运行在全链输出之后的fp32 reference / allreduce阶段发生GPU address0x65000 fault，未宣称诊断精度PASS，也未将该fault归因生产kernel。禁用reference后的复跑及本节各ATT采集正常，原错误日志保留。生产精度结论仍沿用冻结基线，不能用diagnostic exit0替代数值验收。

### 输出目标隔离：有效 EP/P2P stores 显著影响后续 loads

为了保留真实非零输入，以下eager对照使用 `AITER_ATT_DIAGNOSTIC_DEFER=1`：原graph、warmup及额外eager层的dispatch/GEMM1均使用生产kernel，只在选中的GEMM2调用前安装副本。四个rank仍完整参与Mori，rank0采SIMD0/SE0–3；每case8个有效wave、每wave7task、48次入口切换。

- `oob_output` 保留route loads、weights、readlane、每wave每task三条TDM stores以及所有fences，只将输出row indices加到OOB范围，跳过实际global payload写入。早期graph版会影响后续层输入，所以仅作为补充；因果判断使用上述deferred eager输入。
- `oob_sorted` 同样跳过payload，但用有side effects的 `s_and_b32` 保留原destination/readlane依赖，生成严格递增的 `oob..oob+7`，满足两套硬件资料共同的OOB顺序约束。实际trace仍有每wave7次FLAT32、7次FLAT64、56次GLOBAL32、168次readlane、672次packed multiply、168次DS128 stores和21次TDM stores。
- `local_output` 与 `const_geometry` 一样不读geometry，将原peer/slot输出写到本卡独立scratch的不同row，保留有效输出量、三条stores及原fences。每rank scratch为805306368B：12288 slots × 4 peers × 16384B slot stride。它不回写生产EP window，数值输出故意无效。

| 备用机同机eager case | 有效wave span中值 / cycles | 下一N-task入口wait中值 / cycles |
|---|---:|---:|
| 原生产 `fused_diag_original_eager` | 209563.5 | 13482 |
| 不读geometry，仍写EP/P2P `fused_diag_const_geometry_eager` | 222856.5 | 10684 |
| 不读geometry，写本卡scratch `fused_diag_local_eager` | 86879 | 382 |
| 不写payload `fused_diag_oob_eager` | 81473.5 | 9.5 |
| EP/P2P constexpr确认 `fused_diag_const_eager_confirm` | 229672 | 11329 |
| 本卡scratch确认 `fused_diag_local_eager_confirm` | 92400 | 352 |
| OOB确认 `fused_diag_oob_eager_confirm` | 82979.5 | 10.5 |
| 严格递增OOB `fused_diag_oob_sorted_eager` | 92552 | 10.5 |

这些case全部exit0，warnings/missing PCs为空，四份raw均非空且flags=1，decoder INFO为0；前后空闲检查通过。有效wave中每task仍192次WMMA、3条TDM output stores。隔离输出版本的数值故意错误，不能接入生产，也不能将wave span降幅称作kernel性能提升。

**当前证据支持：EP/P2P有效输出目标/流量，使后续memory loads遭遇显著长等待，入口full LOAD drain又将其暴露在下一N-task首条WMMA之前。** 与同样不读geometry而仍写EP/P2P的case相比，写本卡scratch保留了实际写入量却将入口wait从约1.1万cycles降至几百cycles；跳过payload但保留loads/weights/readlane/stores/fences后，wait降至约10cycles。因此仅metadata重复读取、descriptor指令条数或必要stages2/3 fence均不足以解释主要回退。

这里仍有归因边界：local scratch把原16GiB peer stride改成紧凑本卡布局，改变了VA、translation、cache/访问分布；常量参数也影响编译器寄存器和指令安排。实验没有唯一分离P2P链路本身、TDM/Addresser arbitration、cache、translation或具体FIFO。不能由此宣称某个硬件队列存在bug，更不能直接删除保证LDS生命周期和输入完成的wait/barrier。

汇总：[diagnostic_summary.json](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/diagnostic_summary.json)，包含精确symbol、每case wait PC、原source SHA、动态counts和空闲排除标记。主要逐指令证据：[原生产eager](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/accepted/fused_diag_original_eager/analysis_v3/representative_trace.txt)、[本卡scratch](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/accepted/fused_diag_local_eager_confirm/analysis_v3/representative_trace.txt)、[严格递增OOB](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/accepted/fused_diag_oob_sorted_eager/analysis_v3/representative_trace.txt)。同机诊断图：[PNG](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gemm2_att_diagnostics.png)、[PDF](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gemm2_att_diagnostics.pdf)。

#### 2026-10-14 已完成验证与三组 eager GEMM2 对照汇总

- 在 `f01-1/dsr1_ep4` 抓完普通 `_persist` 和 `_persist_epscatter` 的 SIMD0–3、四个 SE，raw、decoded wave、目标 ISA 和 PC 映射均通过验收。普通 SIMD3 的结束检查出现其它 KFD 进程，其耗时对照已排除，raw/ISA 验收结果仍保留。
- 在原机拆开 LOAD/DS wait：LOAD 等待中值为 `17,763.5 cycles`；随后 DS wait 的768次记录全部为 `1 cycle`。这组长等待落在 LOADcnt；DS数据表示先完成 LOAD 后没有剩余 DS 等待。
- 去掉 kernel 内的 window geometry 读取后，长等待仍存在，说明 metadata 重复读取不能单独解释这次回退。
- 在 `f01-2` 保留真实 EP 输入，对所选 eager GEMM2 的输出路径做隔离对照，结果如下。各组均为8个有效wave、每wave7个N-task、48次入口切换，采集前后空闲检查通过。

| eager GEMM2 对照 | 入口 wait 中值 | 有效 wave 跨度中值 |
|---|---:|---:|
| 原 `_persist_epscatter` | 13,482 cycles | 209,563.5 cycles |
| 保留指令和同步，输出行全部 OOB | 9.5 cycles | 81,473.5 cycles |
| 保留有效写入量，输出改到本卡独立 buffer | 382 cycles | 86,879 cycles |

后两组保留了每wave每N-task三条 TDM store 和必要同步，即每个有效wave的7个N-task共21条 TDM store。本卡输出对照已复测，入口wait中值为 `352 cycles`、有效wave跨度中值为 `92,400 cycles`，差异保持明显。证据指向 EP/P2P 输出目标及其对后续 memory 请求的影响，目前还不能唯一确定是哪一个硬件 FIFO、cache 或链路环节；本卡buffer也改变了原peer stride及VA布局，归因限制沿用上文说明。

正式 benchmark 的 GEMM2 仍为 **`58.99 → 119.69 μs`，增加 `102.91%`**。上表是 ATT 中的 wave cycles，不能当作新的 kernel 性能数据；OOB/本卡隔离输出不提供生产数值结果。生产 kernel 基线没有改动。本段补充既有采集结果，没有运行新的GPU测试。

### OOB硬件文档冲突与诊断约束

[Shader Programming Guide](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming%2365.txt) §4.10.3.2，印刷页201，以及 [Tensor DMA资料](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/TX/mi400_tensor_dma%2372.txt) §7.12，要求gather indices递增且不重复；[CDNA5 ISA](../../mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt) §10.11.3.2，印刷页143则明确允许random/repeated indices，但OOB正确处理要求nondecreasing。三份资料在一般indices约束上冲突，不能忽略。

为避免把这种冲突混入输出流量诊断，新增严格递增 `oob_sorted`，使8个indices恰为 `oob..oob+7`，原route依赖仍保留；结果与旧OOB一致。Tensor DMA §7.6.3记载store到OOB区域时global destination skipped；这是跳过payload的依据。该对照满足共同的OOB顺序条件，但不能据此推断任意生产scatter indices都已获得硬件文档一致性保证，也没有把文档冲突直接判定为现有性能回退的bug。

### 完整诊断归档与复现

除上述1.056GB生产完整包外，诊断另存为以下文件，均下载至本地并核对SHA256；原包保留，没有重采/覆盖：

| 归档 | cases | bytes | SHA256 |
|---|---:|---:|---|
| [原机split-wait](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gemm2_att_diagnostics_split_wait.tar.gz) | 1 | 84229394 | `bd04835984b71baa4e2751bccdc6f0da3ae1b3dbd3967cd3df1021bed064e7d3` |
| [原机geometry确认](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gemm2_att_diagnostics_metadata_confirm.tar.gz) | 2 | 151044206 | `5b65c132cbadb79e6b9d07655ca71147fbbc26e655156da4fed7b1475d1b4c7a` |
| [备用机首10组](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/gemm2_att_diagnostics.tar.gz) | 10 | 518177018 | `cc68a5e0a81c985bc2a7ccaa74d8e2275af6060e2e42990af31560cc299019bd` |
| [备用机4组确认](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/gemm2_att_diagnostics_confirm.tar.gz) | 4 | 106091626 | `190540937ce85e7611a4d218f4db9e82f82bd2946f7497fbc927c37ede809460` |

诊断包保留raw ATT、所有captured code objects、code.json、全部映射目标UI wave、decoder events/INFO、source/window manifest、命令及日志；未包含其它kernel的decoded wave JSON，可从raw重建。raw完整性与空闲时序合格是两个独立条件：原机geometry包中的 `global_window` 仍归档，但耗时已排除。

原机复现脚本为 [run_diagnostic_resume_20261014.sh](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/run_diagnostic_resume_20261014.sh)。它固化了当前有效启动脚本，保留6秒空闲轮询、生产路径及全部trace参数；旧 `global_window` 排队在空闲后曾报 `run_diagnostic.sh: line 57: k: command not found`，尚未启动GPU capture，日志保留。新副本两次采集exit0。备用机完整 `run_diagnostic.sh`、`att_env.sh` 及各case命令均包含在归档内。例如从备用机host root执行：

```bash
AITER_ATT_DIAGNOSTIC=split_wait bash run_diagnostic.sh fused 0 new_split_case device graph
AITER_ATT_DIAGNOSTIC=const_geometry AITER_ATT_DIAGNOSTIC_DEFER=1 \
  bash run_diagnostic.sh fused 0 new_const_eager_case device gemm2
AITER_ATT_DIAGNOSTIC=local_output AITER_ATT_DIAGNOSTIC_DEFER=1 \
  bash run_diagnostic.sh fused 0 new_local_eager_case device gemm2
AITER_ATT_DIAGNOSTIC=oob_sorted AITER_ATT_DIAGNOSTIC_DEFER=1 \
  bash run_diagnostic.sh fused 0 new_oob_sorted_case device gemm2
```

原机CPU解码/分析使用taskroot下的绝对脚本路径；容器默认cwd不同，不能用未经核对的相对路径。图表、汇总及排除规则的本地脚本均在原机artifact目录保存。生产persistent/scatter的SHA256与本节冻结值一致，未接入新的生产优化。

### GPU空闲记录复核，以及“有进程在用GPU时是否运行GPU程序”的回答

**不会在已知其它GPU/KFD进程占用时启动GPU程序。** 2026-10-14用户进一步明确约束后，所有GPU运行都要求整机空闲，包括ATT、benchmark和正确性验证；这比工程指令中“正确性可在忙卡时运行”的默认规则更严格，后续按用户要求执行。任意GPU有已有进程、busy不为0或VRAM显示活跃负载，就每6秒检查并等待；可继续CPU静态分析、编译、归档和文档工作，不杀其它任务。

复核了27组已归档生产/诊断trace的host日志：全部有启动前KFD为空、四卡busy=0、VRAM接近runtime基线的记录。24组前后均空闲；以下3组结束检查出现其它进程，不能排除别人于运行中启动任务，因此**排除其所有耗时对照**，raw/ISA仅作非计时证据保留：

| machine / case | 启动前 | 结束后KFD进程 | 处理 |
|---|---|---|---|
| f01-1 `base_device_simd3_graph` | 空闲 | `1847603` | 普通SIMD3时序排除，图表不再展示该base柱 |
| f01-1 `base_device_simd0_eager` | 空闲 | `1866040` | 普通eager耗时及其与scatter相减排除 |
| f01-1 `fused_diag_global_window_simd0_graph_retry` | 空闲 | `1169351/1169608/1169609/1169610/1169611` | 原机GLOBAL改写的耗时排除；等待转移结论使用前后空闲的备用机同机对照 |

证据：[gpu_idle_audit.json](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gpu_idle_audit.json)。每case保留host-log SHA、最后启动前/结束后快照及 `timing_comparison_eligible`；汇总JSON同步标记排除项。启动前快照不能阻止其它用户随后运行，前后均空闲也不能严格证明期间独占，采集时没有并发轮询GPU以免扰动ATT。以上处理撤回了这3组原有时序比较，没有撤回其成功capture/ISA验收。

原机SIMD0–2生产对照、split-wait、const-geometry和备用机全部14组前后空闲，仍支持本文的主要结论。当前两机采集、decode及archive任务已经结束，未发现本轮 `trace_mega_moe` / `run_diagnostic` GPU进程残留；最终归档和文档整理仅使用CPU，不再启动GPU程序。

### “目前用了哪两台机器？”的回答

本次使用 **f01-1和f01-2**。f01-1为 `yanguahe@ctheliosr-1b114-f01-1.amd.com`，在用户指定的 `dsr1_ep4` 容器、`/app/aiter_moe_a4w4_pr_refactor_wt` 抓取生产kernel SIMD0–3及split/geometry诊断。f01-2为 `yanguahe@ctheliosr-1b114-f01-2.amd.com`，创建独立容器 `codex-dsr1-gemm2-att-20261014`，使用 `/home/yanguahe/codex_dsr1_gemm2_att_20261014` 的独立目录做本卡scratch、OOB等补充隔离对照，没有覆盖原有仓库或其它容器。

原机因其它任务占用需要等待，才使用f01-2补充诊断。两机的数据分别归档，只在同一台机器及相同graph/eager口径内比较。“无本轮采集进程残留”指本任务进程已经退出，不代表两台机器目前没有其它用户的GPU任务。

### 独立 `codex-dsr1-gemm2-att-20261014` 容器使用哪个镜像？

2026-10-14在 `f01-2` 主机通过 `docker inspect` 和 `docker image inspect` 核对，镜像为 **`rocm/vllm-dev:nightly_455_wip__torch__1010_b323`**。创建容器时直接使用该机器已有的本地 image ID `ee6d7ff29b6e`，因此容器的 `Config.Image` 记录的是这个ID。完整 image ID 为 `sha256:ee6d7ff29b6eb6d809304cda48b79195ce8e0f914daa57519e5a7baac3b0588d`；registry digest 为 `rocm/vllm-dev@sha256:ff056bfe0195da1648a249a15a9ad143f7428149ba8cbae8dd6c160f44d4660f`。镜像创建时间为 `2026-10-10T18:28:18.88481242Z`。核对证据：[image_identity_20261014.json](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/image_identity_20261014.json)。本次只读取Docker元数据并更新文档，没有启动GPU程序。

2026-10-14 交付核对：本报告及 `trace_mega_moe_without_profiler.py`、`analyze_mega_moe_device_att.py` 已同步到 `f01-1/dsr1_ep4:/app/aiter_moe_a4w4_pr_refactor_wt/my_code/`；旧采集 helper 已先备份。15个报告、图表、汇总和复查脚本保存在原机 task root 的 `delivery_20261014/`，逐文件核对本地/远端SHA256，生产persistent/scatter两文件仍与冻结SHA256一致。核对记录：[delivery_verification.json](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/delivery_verification.json)。两机最后一次本轮采集/解码进程检查均无匹配；本次交付仅执行CPU分析及文件操作，没有启动GPU程序。thread trace采集与回退分析已完成，生产性能修复尚未完成。

## 8. 其它已回答的工作记录问题

### `git add` 的 CRLF 警告是否说明暂存区仍是CRLF？

2026-10-13 当时直接检查暂存 blob，`aiter/ops/flydsl/grouped_moe_gfx1250.py` 的暂存区是纯LF：CRLF为0、单独LF为2512；工作区则CRLF为2335、单独LF为177，是混合换行。`git ls-files --eol` 报 `i/lf w/mixed attr/text eol=lf`。Git按 `text eol=lf` 在add时规范化暂存内容，警告指的是工作区CRLF，不意味着暂存blob仍为CRLF。这些字节计数是当时状态，不是当前新补丁后的重新统计。

### 已暂存的修改是否已commit并push？

2026-10-13 已提交当时暂存的7个文件并推送 `origin/moe_a4w4_pr_refactor_wt`：commit `e749e6cef3392295f7a1c491a1d9f51a1b75bbb2`，message `Fix gfx1250 MoE LDS races and tune E64 token dispatch`。随后四卡接入的新补丁是这次commit之上的未提交修改，不能把历史push结果当作新补丁已经push。

### 四卡 benchmark 接入的代码是否先在本地修改，再传到 f01-1？

2026-10-15核对历史操作记录：**是的，四卡接入的主要源码补丁先在本地 Windows 仓库 `C:/Users/yanguahe/Documents/code/wk_sp1/aiter` 修改，再通过长期 SFTP 会话逐文件上传到 `f01-1`。** 当时的中转目录为 `/data/yanguahe/code/wk_sp1/.codex_tmp/f01_1_dsr1_refactor_p192_20261013/upload/`，随后将对应文件复制到 `dsr1_ep4` 容器内 `/app/aiter_moe_a4w4_pr_refactor_wt` 的相应仓库路径。实际按“本地修改 → 上传 → 远端编译/测试 → 本地继续修正”的方式迭代；安装、编译和四卡 benchmark 在远端执行，远端运行脚本及环境配置也在远端处理。

目前本地保留这套精度正确的双 persistent 基线。此次逐文件核对，下列8个实现/配置文件在本地和原容器中的 SHA256 均与 `persistent_both_source_manifest.json` 冻结值一致：

- `aiter/configs/tuned_grouped_fmoe.csv`
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py`
- `aiter/ops/flydsl/grouped_moe_gfx1250.py`
- `aiter/ops/flydsl/moe_kernels.py`
- `aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py`
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill.py`
- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py`
- `aiter/ops/flydsl/kernels/mega_moe_gfx1250/gemm2_scatter.py`

证据：[源码冻结清单](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/persistent_both_source_manifest.json)、[历史本地修改及SFTP同步记录](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/source_sync_history_20261015.json)。后续 ATT 的隔离诊断副本保存在独立归档中，生产源码仍使用上述冻结基线。本次只核对源码、历史记录和文档，没有运行GPU程序。

### 当前进展、精度优先和暂停状态

main安装及原四卡benchmark、refactor prefill和双persistent四卡接入、独立精度验证与性能归档已经完成，详见 [性能记录](dsr1_moe_ep_decoder_opt.md)。当前双persistent基线的base/fused16层中值为 `7029.273/6347.078 μs`；精度结论与限制见第2节。候选优化未替换已冻结的精度正确基线；用户后续已授权 thread trace 采集和性能回退分析，本轮执行这项诊断，全面 kernel 调优仍未恢复。

后续每次回答同步写入本文件。新的实际采集或性能结论应追加日期、命令、代码来源和artifact链接。此前暂停及四卡通信说明的记录保留；本次用户明确要求继续后，完成了第7节的 thread trace 隔离诊断与归档，全面 kernel 调优仍未恢复，也没有修改生产精度基线。

### 当前会话在 `.codex/sessions` 中对应哪些文件？

2026-10-15核对本地JSONL的 `session_meta` 和最新用户消息：当前会话正在写入的文件为 `C:/Users/yanguahe/.codex/sessions/2026/10/11/rollout-2026-10-11T23-24-58-01a12b91-3152-7ab3-852b-49232fc8c38d.jsonl`，`session_id=01a12b91-3152-7ab3-852b-49232fc8c38d`。文件末尾包含用户本次“这个会话的 session 文件是哪些”的问题。路径中的日期是该会话创建日期，当前记录仍追加到这份文件。

用户开头指定继承的基础工作历史另有两份：

- `C:/Users/yanguahe/.codex/sessions/2026/09/07/rollout-2026-09-07T20-45-46-01a07be7-3b41-7d53-9cf8-9eb72323d943.jsonl`
- `C:/Users/yanguahe/.codex/sessions/2026/09/08/rollout-2026-09-08T22-23-18-01a08166-e38d-7303-ad9f-fab12d849d6d.jsonl`

恢复用户指定的工程历史时，按9月7日基础文件 → 9月8日继承文件 → 当前10月11日会话文件读取。9月8日的 `history_base` 明确限定9月7日文件至 `end_ordinal_exclusive=7275`、`end_byte_offset=33828333`；当前10月11日的 `session_meta` 没有 `forked_from_id/history_base`，9月历史是用户要求读取的上下文。

另核对到 `C:/Users/yanguahe/.codex/sessions/2026/10/14/rollout-2026-10-14T13-04-16-01a138cc-029c-7952-93c1-ab8aa41dbe12.jsonl` 为从当前会话分出的独立会话，`forked_from_id=01a12b91-3152-7ab3-852b-49232fc8c38d`、`forked_from_ordinal_exclusive=11380`，`history_base.end_byte_offset=98397462`。该文件保存另一支工作；本次最新用户消息写在上述10月11日活动文件中。本次仅检查本地记录并补充文档，没有运行GPU程序。

## 9. 四卡 MoE 计算在哪些环节通信？同一 token 的 8 个 expert 在同卡就不需要通信吗？

2026-10-14，用户指定本次说明写入 [moe_framework_wiki_ep4.md](moe_framework_wiki_ep4.md)，已追加完整的「四卡 MoE benchmark 的通信流程」章节，保留该 wiki 先前的 DSR1 服务说明。该章节给出原命令、MoE 概念式、expert 分组、routing 矩阵、base/fused 流程图、有效 payload 推导、kernel 对应关系与源码证据。

**当前 balanced routing 下，同一 token 的 8 个 expert 确实在同一个目的 rank，但它们不一定在 token 的来源 rank。** `expert_id(t,k)=(8*t+k)%256`，expert rank 为 `expert_id//64`；64 可被8整除，同一 token 不跨目的 rank 边界。四个来源 rank 的 IDs 都使用这个公式，没有添加来源 rank 偏移。

例如，来源 rank0 的 token8 选择 expert64–71，需要 rank0→rank1 dispatch，rank1 执行8个 expert，再把结果送回 rank0。每个来源 rank 的1536个 token中，向每个目的 rank发送384个；只有384个在本卡 expert上计算，其余1152个即75%需要跨卡去程和回程。全组有4608个远端源 token。只有目的 expert rank与来源 rank相同，该 token的大块payload才留在本卡；跨rank的协议同步仍存在。

每层的大块通信位置是 GEMM1 前的 dispatch 和 expert 结果回程。GEMM1/SwiGLU/GEMM2的矩阵计算及2048维中间activation留在expert卡。`base`先在expert卡对同一token的本地expert结果加权求和，然后Mori combine由来源卡P2P gather部分和；`fused`在GEMM2 epilogue中把每条route的加权结果P2P写入来源卡，再做`ep_combine_fused_sync`和本地topk slot reduce。

这份测试的dispatch按目的rank去重，所以同一token的输入只发一份。在`combine_quant=none`的BF16回程下，base每远端token回传一份14336B的部分和，fused回传8份14336B的route结果；有效回程payload分别为每来源rank每层15.75MiB和126MiB，比例8倍。这是源码和CPU routing推导，不是实测链路流量，也不能直接转换成耗时比。

此次核对采用`f01-1/dsr1_ep4:/app/aiter_moe_a4w4_pr_refactor_wt`的代码，本地/远端benchmark与MegaMoE文件SHA256一致，4份Mori相关源码下载后SHA256全部一致。CPU枚举确认每token只有1个目的expert rank、每rank有效接收1536个token、每expert192条route。证据：[source_validation.json](../../.codex_tmp/moe_ep4_communication_20261014/source_validation.json)。没有运行新的GPU测试或修改生产kernel。

## 10. AITER_MOE_EXPERT_BALANCE=False 如何改变四卡 benchmark 的 token routing？

2026-10-14，完整说明已补入 [moe_framework_wiki_ep4.md 第 10 节](moe_framework_wiki_ep4.md#ep4-random-routing)。本次核对 routing、dispatch/combine 和当前本地 launcher 源码，并计算理想随机 topk 的数学期望。

`False` 将固定的 `expert_id(t,k)=(8*t+k)%256` 改为每 token 的256个随机 scores取 top8，选出8个不同 expert。seed 为 `4242+100*source_rank+args.seed+layer_idx`，不同来源 rank、不同层使用不同随机序列；routing在capture前生成，base/fused共用，每次replay保留，不会从当前hidden states重算真实router。weights仍独立随机并按token归一化，且切换开关后会因RNG消耗不同而改变实际数值。

同一token通常访问3–4张卡，dispatch仍按目的rank去重，每个已访问rank各接收一份activation/scales；对应rank只计算属于它的expert routes。base由各expert rank先汇总本地expert部分和再回传，fused仍由GEMM2逐route写回来源rank的独立slot。中间activation留在expert卡。

理想均匀抽取256个expert中的8个时，每token平均访问3.6142个目的rank，远端目的rank数期望2.7107；至少有一条远端route的token比例约99.99891956%，远端expert route比例期望仍为75%。每目的rank有效接收rows从精确1536变为期望5551.4332，每expert有效routes从精确192变为期望192、标准差约13.6382。全组有效route数仍为49152；随机负载可能增加M-tile数和padding。

在`combine_quant=none`条件下，每来源rank每层远端有效payload的期望为：FP4 dispatch含有效scales约15.1204MiB，base BF16部分和约56.9239MiB，fused BF16逐route结果126MiB。相比balanced的4.184/15.75/126MiB，dispatch和base期望字节量约增至3.6142倍，fused期望字节量相同；fused/base期望字节比约2.2135。这是理论有效payload，不是实测链路流量或性能比，计算见 [random_route_expectations.json](../../.codex_tmp/moe_ep4_communication_20261014/random_route_expectations.json)。

当前本地`grouped_gemm_mxfp4.py`也读取这个开关：False会关闭相应`t256/w2x4`的balanced A-preshuffle准入，以及未显式覆盖时的`auto_balanced_m192`；相应`t192/w2x4`准入不依赖balance为True，但仍有其它条件。本次没有执行False的GPU benchmark或修改生产kernel。用户已取消的GEMM2 thread trace诊断结果归档和文档更新任务未恢复。

## 11. main 与 moe_a4w4_pr_refactor_wt 的四卡 kernel 列表为什么不同？

2026-10-15，按用户要求使用已经跑好的 `moe_ep4_perf.log` 完成逐项比对，完整回答写入 [moe_framework_wiki_ep4.md 第 11 节](moe_framework_wiki_ep4.md#ep4-kernel-branch-comparison)。未重跑测试或修改 kernel，也未恢复已取消的 thread trace 归档任务。

每种 combine 模式，main 为 19 个 GPU kernel，refactor 为 20 个；另各有3个 HIP host API事件。15个 GPU symbol 相同，4个被替换：GEMM1、GEMM2、接收后的输入 producer 和 global→local LUT；额外新增 `moe_invert_route_rows_tk8`。inverse map 的 `torch.full(-1)` 与原初始化共用 `FillFunctor<int>`，使该 symbol 的计数约翻倍。相同 Mori dispatch、base combine、fused sync/reduce 的整体通信语义保持一致；GEMM2 内部 scatter 的 tile/slice 实现不同。

真实配置选择来自 EP 的静态接收容量6144→CSV token key8192、local E64、topk=-1。refactor CSV 给 GEMM1配置t192/w2x2/b4，fused-quant路径再提升n_warp至4并选persistent；GEMM2配置t192/w2x4/b4。`_find_e64_topk8_token_profile()`排除EP，所以不能说四卡直接命中了单卡T1536 interval。refactor容器工作区含先前未提交补丁，不能把所有差异归为纯origin commit。

文本显示也放大了差异：main列宽120，refactor72；固定4空格的`egrep "kernel    avg_us"`漏掉refactor的两张base表，fused表仍能匹配；`-A21`不足以截取完整事件列表。完整symbol逐项映射见 [kernel_mapping.json](../../.codex_tmp/f01_1_ep4_kernel_compare_20261014/kernel_mapping.json)。

用户现有日志中，GEMM1约为main106–108μs/refactor84–85μs，普通GEMM2约65/53μs，带epscatter的GEMM2两边都约116μs。完整GEMM/通信的四rank字段与e2e记录见wiki第11.10节。日志没有附带当时GPU空闲记录，且acc_verify=0，本次只是整理既有记录，没有新增受控性能或精度验证。

## 12. 当前会话对应哪些 session JSONL 文件？

2026-10-15，核对运行环境中的 `CODEX_THREAD_ID`、`CODEX_SESSION_ID` 与本地 JSONL 第一条 `session_meta`，当前会话直接对应以下两个文件。

当前活动文件，session_id 为 `01a138cc-029c-7952-93c1-ab8aa41dbe12`：

```text
C:\Users\yanguahe\.codex\sessions\2026\10\14\rollout-2026-10-14T13-04-16-01a138cc-029c-7952-93c1-ab8aa41dbe12.jsonl
```

继承的基础历史文件，session_id 为 `01a12b91-3152-7ab3-852b-49232fc8c38d`：

```text
C:\Users\yanguahe\.codex\sessions\2026\10\11\rollout-2026-10-11T23-24-58-01a12b91-3152-7ab3-852b-49232fc8c38d.jsonl
```

活动文件元数据记录：

```text
forked_from_id = 01a12b91-3152-7ab3-852b-49232fc8c38d
forked_from_ordinal_exclusive = 11380
history_base.thread_id = 01a12b91-3152-7ab3-852b-49232fc8c38d
history_base.end_ordinal_exclusive = 11380
history_base.end_byte_offset = 98397462
```

完整恢复本会话时，先读取10月11日基础文件到上述继承边界，再读取10月14日活动文件。基础文件在fork之后新增的内容不属于当前会话继承的历史。基础文件自身没有记录更上游的 `forked_from_id` 或 `history_base`；此前用户指定阅读的9月会话是外部引用的历史文件，不是当前元数据中的直接fork链。本次只读取session文件及更新本问答记录，没有修改JSONL或运行GPU程序。

## 13. a07-3 上本轮优化使用的代码仓在哪个位置？

2026-10-24，依据本轮 E65/T1536 GEMM1 优化记录，使用的远端主机为 `yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`，代码仓路径为：

```text
/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
```

测试容器为 `dsr1_ep4`。主机和容器共享同一仓库目录，容器内也是上述路径。本次仅整理已知仓库位置，没有重新登录远端或运行测试。
