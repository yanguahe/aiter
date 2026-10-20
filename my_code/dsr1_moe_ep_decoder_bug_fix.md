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

## 13. optX 方案实现的 GEMM1 使用多少 VGPR、SGPR、LDS？

2026-10-18，按实际编译的 ELF AMDGPU metadata 回答。连续 ring 的 v8、逐 tile READY 的 v14均为VGPR512、SGPR96、LDS295936 B（289 KiB）；最新逐 row LDS schedule 的 v16为VGPR506、SGPR96、LDS295936 B（289 KiB）。private segment及VGPR/SGPR spill均为0。早期每批输入drain的v6/v7对照为VGPR400、SGPR93、同样289 KiB LDS，不能将其较低寄存器数量当作连续ring版本的资源结果。

VGPR count表示每lane的向量寄存器数量，SGPR count表示每wave的标量寄存器数量，LDS为每workgroup固定分配；这里报告编译器metadata，不由count直接推断硬件按粒度分配后的占用率。LDS拆分为256 KiB data ring、16 KiB scale arena、17 KiB双窗口output arena。v8完整正确性通过但原命令单次125.716μs，v14 random/replay通过但单次123.207μs，均未采纳；v16先取得无spill资源和random数值通过，再进行性能筛选。详细资源表、源码hash、复现与证据见 [optX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

## 14. A/ScaleA 的 LDS 使用是否减少，是否只驻留一份 A128/ScaleA128？

截至2026-10-18的G4/S4和G2/S4候选，尚未减少，也不是每stage只驻留一份。每个逻辑stage分别保留A_F和A_S各16 KiB、ScaleA_F和ScaleA_S各1 KiB；S4合计A=128 KiB、ScaleA=8 KiB。A_S在下一物理stage的tail，仍是独立存储，不是覆盖A_F。G2仅缩小K-group，LDS总量仍为295936 B（289 KiB）。

接下来实施S5槽位复用：F读完并退休DS后，同一A128/ScaleA128区域装入S；B/ScaleB保持到S读完。推导为5×48 KiB data、5×3 KiB scale、17 KiB output，共272 KiB；A合计80 KiB、ScaleA5 KiB。此处是待编译确认的布局设计，不是已取得的性能结果。A/ScaleA在F、S各发一次TDM，B/ScaleB仅发一次，必须逐owner重推tensorcnt并使用模5的跨taskstage编号。

最新G2/S4 v21实编译为VGPR406、SGPR91、LDS295936 B，无private/spill；random17有效payload、ScaleA、GEMM2、MoE bitwise一致，但原命令单次116.443 μs仍慢于原路径。v22/v23的错峰epilogue候选出现spill，均没有运行GPU。详细记录和证据见 [optX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

同日后续进展：S5槽位复用首版已经完成并实际编译，ELF确认LDS278528 B（272 KiB）。每stage确实只保留一份A128/ScaleA128供F、S依次覆盖，A总80 KiB、ScaleA总5 KiB。v26为VGPR445/SGPR107，SGPR spill46；v27/v28/v29将其降为34/23/21，仍未通过无spill门禁，尚无该布局的GPU正确性或性能数据，不能把LDS下降直接当作耗时改善。READY/RELEASE事件模型G4/G2及八项负对照已通过。

同日进一步完成：S5 v33实际编译为VGPR435、SGPR106、LDS278528 B，private及VGPR/SGPR spill均为0；T1536/random17有效FP4 payload、ScaleA、GEMM2、最终MoE bitwise一致，dynamic/poison/eager/graph replay也通过。单份A128/ScaleA128驻留已经落实，但原benchmark单次138.662 μs，仍慢于原persistent路径，尚未采纳。后续rolled F epilogue stagger v39为VGPR472、SGPR95、同样272 KiB LDS、无private/spill，random/replay通过，单次165.346 μs；S5 padding skip v38为479/100、同样272 KiB，单次140.753 μs。详见 [optX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

同日继续从LDS读量调优：G2 B/ScaleB寄存器复用v40为VGPR493/SGPR99、289 KiB LDS、无private/spill，原命令单次108.140 μs；G1较小cache v41为424/96、同样289 KiB、无private/spill，单次108.888 μs。两者random及dynamic/poison/eager/graph replay通过，仍慢于原路径当日中值94.279 μs。此优化减少B/ScaleB的LDS读取，不减少A/ScaleA的S4驻留，也不能把VGPR下降直接当作性能收益。v35 ATT确认padding skip实际少21.09375%的WMMA，但LDS和barrier等待仍显著。下一步继续验证G1跨task epilogue stagger；未启用默认optX。

## 15. 2026-10-18 继续优化的进展：同步合并、carry 和窗口顺序

正式保留路径仍为同一SHA256 `b36f7ec9…e790`：历史已验收中值94.888 μs，当日空闲复测94.684/94.279/94.043 μs，中值94.279 μs，距80 μs还差14.279 μs。未把复测的频率或环境差异算成代码收益，optX默认仍关闭。

G1 B-cache的v44把下一input READY与当前S的LDS reader RELEASE合并到一次WG rendezvous，保持每owner TDM队列和有效K顺序。snake1单次105.979 μs，固定h0→h1的snake0单次105.479 μs；0.500 μs差异未经交替复测，不能确认稳定收益。v44为VGPR424/SGPR97，snake0为426/94，均LDS295936 B、无private/spill。random、M_e128/129及replay门禁分别按优化记录所列覆盖；固定顺序目前完成random/replay。

跨task stagger的v43正确但156.547 μs；共用compute body和局部地址的v46为VGPR432/SGPR102、139.789 μs，random/boundary/replay通过，仍未采纳。共用compute的v45有24个SGPR spill，未GPU；局部地址修正v47为411/84、无spill，112.369 μs，但结束六秒时发现他人GPU测试，不能排除潜在计时重叠，保留为初步筛选值而不报告确定回退幅度。

完整window carry在F1读S0、S0读S1、S1读下一F0，B0等S0消费后才覆盖，B1保留到S1消费。v48发生1个VGPR spill，未GPU；v49仅做F1→S0 carry，413/84、无spill，random/replay正确，112.816 μs；v50恢复owner在K-loop外分派，445/73、无spill，random17、M_e128/129、dynamic/poison/eager/graph replay全部bitwise正确，107.301 μs。三者LDS都为295936 B，没有进一步减少A/ScaleA驻留。

v53只改变LLVM expert scheduling选项，ISA实际进入`SCHED_MODE=2`且保留编译器生成的`s_wait_alu`，424/97、无spill，random/replay正确，107.339 μs，没有确认收益。正式GEMM1本来就显式使用normal mode；此项是编译模式对照，不是新发现的功能bug。

目前新结果仍没有优于正式路径。仅完整h0分支外提v55取得411/86、无spill，random17、M_e128/129、dynamic/poison/eager/graph replay正确，原命令单次112.074 μs；h1两组专用v57为426/86、无spill，另补M_e192/193并通过replay，单次112.490 μs。全prefix循环复制v51和owner版本v56分别出现418和503个VGPR spills，CPU拒绝，未GPU。三组专用及调度hint变体继续验证，不因资源下降认定加速。

同日完成v44固定h0→h1的四SIMD ATT，32个完整compute waves、无incomplete，采样中无效WMMA少20.3125%。compute81.5252%、handoff9.4768%；stall归属dscnt13.1236%、DS_load8.8273%、barrier12.6245%。这支持继续检查LDS供数与短padding分支的调度影响，不能将并行wave比例直接换成可省μs，也不足以断言bank conflict或HBM饱和。报告已下载并核SHA256。继续测试相邻wave在同一时间窗口消费不同physical A half的变体，保持TDM和READY/RELEASE顺序；该阶段尚无性能结论。

方法、复现命令、源码/ISA绑定和原日志见 [optX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。自有workload分别包120s timeout，性能全机idle门禁，未恢复用户取消的GEMM2 trace归档任务，也未修改原精度标准。

同日后续完成optX v58/v60/v61数值验证与原命令筛选，单次113.567/112.688/112.127 μs。v62让相邻wave用 `window XOR wave_m` 消费不同physical A half，producer/READY/RELEASE不变，random17、M_e128/129、replay通过，最低单次101.422 μs；shared映射v63为106.987 μs，LDS payload publication与global ScaleA drain分开的v65为106.137 μs，同样通过所列数值门禁。均未三轮验收，也仍慢于冻结K256。v58/v61的post发现他人E65任务，保留性能限制，不据此确认稳定增益；其余立即post有短暂busy，独立post6 idle。optX默认仍为0，live仍v61。按照用户授权，完成这组深调后已恢复K256路线，详细数据已补入optX优化记录。

K256 v94跳过完全无效M32 WMMA；machine/postRA LICM关闭后review为VGPR493/SGPR86、LDS278528 B、无private/spill。v94/v95均完成11case及dynamic/poison/eager/graph replay，有效canonical payload/ScaleA/GEMM2/MoE bitwise一致。v94三轮交替review/skip为96.476/94.897、93.271/95.350、95.861/94.642 μs，中值95.861/94.897 μs；两轮快、一轮慢，未确认稳定收益，未采纳。v95 stripe约97.1 μs，亦未采纳。compile_hints是launcher-wide，尚不能宣称fallback ELF不变；早期sched2 CPU漏设环境变量，不能用于判定mode2收益。

v96复制三个pair的steady loop出现544个VGPR spills；v97 poison和v98 undef仍spill，未GPU。v98 pure scalar predicate完成11case/replay，单次97.545 μs可能与CPU编译重叠，只作受限筛选。v99整个task外分派，在161..192行只保留六组M16 accumulator，两版都有2个VGPR spill，未GPU。关闭LLVM if/else live-range优化的v100 prepared `.text`与v99相同，没有重复GPU。

v101保留六组专用task，其余tail恢复完整padded WMMA，prepared实编译VGPR484/SGPR88/LDS278528 B，private/VGPR spill/SGPR spill全部0；review仍有1个VGPR spill，未GPU。prepared的11case及dynamic/poison/eager/graph replay已通过，有效四项输出bitwise一致；原完整benchmark单次104.669 μs/count19，未采纳。正式persistent源码SHA仍为 `b36f7ec9…e790`；最近同源码全机idle复测中值94.279 μs，距80 μs仍差14.279 μs。方法、复现命令、性能及ISA证据详见 [non-balanced优化记录](moe_gemm1_e64_t1536_nonbalanced_optimization.md)。

## 16. 四级输入 ring 复用 A128/ScaleA128 的实施结果（2026-10-18）

用户建议上下两半复用同一份A128/ScaleA128，将输入ring降到 `4 × (A16 + B32 + ScaleA1 + ScaleB2) = 204 KiB`。该方案已实现：每stage只驻留一份A128/ScaleA128，F读者退休后才覆盖成S，B/ScaleB保持到两半消费完；两个output windows另占17 KiB，总LDS221 KiB（226304 B）。不是把总kernel LDS说成204 KiB。

四槽跨task顺序需要重新计算READY：F0–F2先于旧两条output TDM，F3后于它们，仅successor首group的step0–2加入旧output计数，step3排除。沿用五槽计数的负对照在p31捕获missing READY；早覆盖A、提前B refill、缺DS RELEASE、错误stage stride和output复用等负对照也捕获。同步和有效K的正向顺序都保留。

v104 G4为VGPR362/SGPR86，无private/spill，原命令单次130.013 μs；G2为135.454 μs。无效M16 skip的v105 G4为134.358 μs（post发现他人任务，受限筛选）；B/ScaleB cache的v106 G2为480/78、无spill，129.591 μs，G4 cache发生2775个VGPR spill而未GPU。将S0 READY/最终F DS RELEASE与S0预读移到F closing WMMA前的v107 G4为407/78、无spill，单次122.235 μs；v108 G2 cache+carry为126.000 μs（post6有他人E65任务，受限筛选）。均慢于正式中值94.279 μs，未替换正式路径，不能将单次差值声称稳定收益。

v104–v108已分别完成记录中列出的random及dynamic/poison/eager/graph replay，有效payload、ScaleA、GEMM2、MoE bitwise一致。v107完整11case累计到120秒自动终止时已通过前六项，剩余五项拆分后全部通过；M_e=0/127/128/129边界通过，其他边界继续分批。v109扩展组内下一tile carry，CPU为410/78/221 KiB、无spill，random17通过，replay继续验证。所有自有单项workload包120s timeout，超时不能记成功，不留自有悬挂进程。

四SIMD ATT采样32条完整compute waves：compute85.9818%、handoff7.6060%，stall归属`dscnt`20.3768%、WG barrier18.6755%、tensorcnt1.7506%。当前证据支持LDS供数与同步暴露等待，不能断言HBM饱和或bank conflict，也不能把wave占比换成可省μs。详见 [四级共享A thread trace分析](moe_gemm1_e64_t1536_s4_shared_a_thread_trace_analysis.md) 和 [optX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

## 17. OPTX 最优耗时与共享 A 槽方向的停止决定（2026-10-18）

问题：OPTX 目前最优 GEMM1 是多少，上下两半复用同一份 A128/ScaleA128 槽是否值得继续。

回答：OPTX 历史最低单次为 v62 的 **101.422 μs**，尚未经过三轮配对验收，不能称为已采纳的稳定性能。正式冻结路径历史验收中值为94.888 μs；当天同源码全机空闲三轮为94.684/94.279/94.043 μs，中值 **94.279 μs**。跨批次环境差异不算代码收益，80 μs目标仍未达到。

共享槽方向实测最低单次为 v110 G4 的 **114.303 μs**；后续split barrier的v114为114.594 μs，skip的v115为116.987 μs，v118组合为119.504 μs，均无优势。v119已通过数值与replay，但尚在等待他人GPU任务释放，按用户指令取消了性能controller，未启动workload、无性能结果。不是把未测候选算作慢，也不是kernel超时或死锁。v120仅CPU，不再测。

按用户最新指令，放弃共享A槽方向，保留源码、ISA和负结果；接下来以v62原双份physical A half为参考，尝试operand carry、split WG rendezvous和更小的epilogue切片。正式persistent源码、有效计算、通用fallback、精度标准和默认selector保持当前状态。新候选先通过CPU/ISA及有效四项bitwise比较，再在全机GPU/KFD空闲时用原完整benchmark计时。

## 18. 继续 OPTX 调优时为何数值验证出现 memory fault？（2026-10-18）

本轮 v121 的首次 random 验证和历史正确 v62 的复验都 exit134。同步定位确认 v62 的 OPTX launch 已完成，随后 reference FP32 matmul fault；使用完全不导入 AITER/FlyDSL、不执行 MoE 的独立脚本也复现同一 hipBLASLt `...UserArgs_MT32x32x16...ISA1250...` memory fault，证明 reference 有独立的后端故障。尚未定位 hipBLASLt 内部根因，不能仅凭这些证据把所有候选判为正确。

只选择 PyTorch `cublas` 后端仍可能经 rocBLAS 内部调用 hipBLASLt。为本任务的验证进程同时设置 `TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0` 后，v121–v123 完整 random seed17 验证成功：有效 canonical payload、ScaleA、GEMM2、MoE 四项 mismatch 都为0，原 logits 阈值未放宽；dynamic/poison/eager/graph replay 也通过。没有修改生产 kernel、benchmark、PyTorch/ROCm 安装或全局容器环境，也没有 reset GPU。

后续性能使用同一原完整 benchmark，baseline/candidate 同时采用上述 reference 后端设置；reference 在计时结束后计算，其耗时不算 GEMM1。被测路径的实际 source、object 和计时结果分别绑定、核对。v121–v123 均为VGPR424、SGPR97、LDS295936 B、无private/spill，当前尚无新有效性能值。

v124–v126 进一步实验每 wave 独立的 output publication，只移除该处不需要跨 wave 数据交换的 WG rendezvous，保留 `s_wait_dscnt(0)`、最终 ScaleA STOREcnt 和 output TENSORcnt drain。input READY/RELEASE 全部保留；source/address 模型与负对照已完成，三版均已通过random seed17和dynamic/poison/eager/graph replay，有效四项bitwise mismatch为0。完整命令和故障日志见 [OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

## 19. OPTX继续优化的进展与有效性能结论（2026-10-18）

截至v132筛选完成，正式冻结路径最近三轮中值仍为 **94.279 μs**，距80 μs差14.279 μs；历史验收中值94.888 μs仍保留。新OPTX v128完整operand carry与每wave output publication的单次为 **98.525 μs**，尚未三轮配对，仍慢于正式路径，没有替换默认kernel。v131/v132较宽payload LDS stores分别101.401/101.857 μs，未采纳；共享A128/ScaleA128槽方向按用户指令继续停止。

v127的133.021 μs在退出前已有他人GPU任务启动，现有日志不足以判断是否与GEMM1计时重叠，因此不用于判定该方案本身回退幅度。v128立即post无KFD，独立post6的他人任务在退出后启动；单次筛选仍不能替代稳定配对收益。四版random17和dynamic/poison/eager/graph replay全部有效四项bitwise mismatch0，实际object均无spill；没有改原计时、精度标准或正式源码。

v133/v134有效行mask首版各有8个SGPR spill，CPU门禁拒绝，不运行GPU；v135/v136 K128窗口交织分别VGPR424/432、SGPR97/107、LDS295936 B、无private/spill，继续严格验证后筛选。完整方法、复现命令、性能与负载证据见 [OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

后续结果：v135/v136 random/replay通过，单次99.038/98.842 μs；v137/v138两组READY/RELEASE合并，使step rendezvous从每wave四task112降为56，input请求和有效K不变，random/replay正确，但单次98.995/99.000 μs，未采纳。v139 typed LDS读取的CPU `.text`与v128相同，未重复GPU；v140部分typed LDS少44条静态ISA、6条DScnt wait，random/replay正确，单次99.015 μs，未采纳。

v128四SIMD ATT与原98.525 μs性能实际完整ELF相同，32条完整wave、incomplete0；compute82.4565%、handoff8.7341%，barrier归属stall13.1057%、DScnt5.9002%。逐PC分析确认热点集中在循环rendezvous，大量DScnt waits紧邻下一批LDS reads；现有counter不足以断言HBM饱和、bank conflict或LLVM bug，不把占比换算为可省μs。详细证据、硬件出处和复现写入 [OPTX v128 thread trace分析](moe_gemm1_e64_t1536_optx_v128_thread_trace_analysis.md)。旧正式版trace历史保留。

根据trace新增v141/v142/v143：cached有效行谓词、逐M16跳无效A读取、两者组合。v141/v143各6/10个SGPR spill，CPU拒绝、未GPU；v142无spill，所有有效payload/ScaleA/GEMM2/MoE及dynamic/poison/graph replay bitwise正确，但原完整benchmark109.947 μs，未采纳。读取量减少不自动带来性能提升，细分支会改变ISA与供数调度；M32粒度对照继续验证。当前正式94.279 μs中值、14.279 μs目标差距及默认OPTX关闭状态保持。

同日M32粒度对照已完成：v144只跳完全无效M32的A reads，为104.977 μs；v145在FlyDSL构造时因条件内namedtuple成员赋值失败，未GPU；v146修正赋值构造后同时跳无效ScaleA word，为107.290 μs。v144/v146实际VGPR427、SGPR107、LDS295936 B、无private/spill，random seed17及dynamic/poison/eager/graph replay有效四项bitwise一致。均未采纳，这一类细分支跳读停止。构造错误是本次实验代码问题，不是新发现的正式kernel bug。

v128完整carry路径的三组LDS/WMMA hints对照中，v147/v148 CPU `.text`与parent相同，未重复GPU；v149四组WMMA2交织DS实际机器码有变化，random/replay通过，原完整benchmark99.120 μs，仍未取得收益。正式冻结源码、默认选择器及目标差距不变；下一步检查独立operand register bank提前S窗口预读，输入LDS仍保留双份A，不恢复已放弃的共享A槽。方法、复现命令与原证据继续写入 [OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

## 20. 独立 register banks 预读 S 窗口的进展（2026-10-18）

本轮v150/v151/v152分别在F0期间预读S1、用四banks预读S0/S1、以及在F1期间读齐S0/S1。原完整random/non-balanced命令单次为 **100.935 / 101.726 / 96.643 μs**；v152是截至这三项结束的OPTX最低单次，比历史v128的98.525 μs低，但没有同批三轮配对，仍慢于正式路径当天中值94.279 μs，不能报告已采纳的稳定收益。正式源码及默认OPTX关闭状态保持，距离80 μs仍差14.279 μs。

三项有效payload、ScaleA、GEMM2、MoE在random seed17及dynamic/poison/eager/graph replay全部bitwise一致；实际VGPR为461/495/463，SGPR107、LDS295936 B，private及两类spill均0。这是register供数调度优化，没有改变功能、精度标准或恢复共享A LDS槽。十五项验证/性能/post/实际ISA日志已下载核SHA；v151的独立post6有退出后新启动的T4608任务，负载限制如实写入OPTX记录。

继续以v152为parent测试split READY/RELEASE：S reads全部完成后先signal，在完整wait前只执行已缓存的S WMMA；refill和下一F0读取仍在wait之后。三项CPU无spill。新增逐事件协议模型直接施加错误bank/ksl、提前B复用、提前refill、缺DS completion和READY overcount等mutation，全部被同一套assertions捕获；模型不代替GPU和实际ISA门禁。实验本身未发现新的正式kernel功能bug。

## 21. split WG barrier候选与三轮配对的进展（2026-10-18）

用户要求汇报进展并继续优化。新v153在F1期间把S0/S1读取到独立register banks，完成原tensor_wait和全部DScnt退休后signal，在WG wait前计算S0的8条WMMA；完整wait之后才refill和读取下一F0，不降低通用功能、有效K顺序或精度。输入LDS仍双份A128/ScaleA128。实际资源VGPR463、SGPR107、LDS295936 B，private和spill全0。把12/16条WMMA放到signal/wait间的v154/v155单次96.276/96.419 μs，没有优势。

v153的11项routing/token用例、十项M_e边界0/127/128/129/143/144/145/255/256/257、E65与E65/shared1 expected fallback、dynamic/poison/eager/baseline-graph/OPTX-graph replay全部通过，有效payload、ScaleA、GEMM2、MoE bitwise mismatch0。实际ISA CFG审计确认所有signal路径汇合到同一WAIT，中間没有memory或第二SIGNAL，不能把审计当作GPU精度验证。

第一批独立进程配对中值基准94.760/候选91.559 μs，第二轮基准post发现他人T3840 PID1225845已运行17秒，启动时间与基准进程重叠，所以3.38%不作为严格性能验收；原日志保留。补测在同一bounded HIP context内分别执行两次原完整benchmark，原参数、warmup、random数据、profiler和--iters20/count19全部保持，第二轮逆序。基准94.625/94.753/94.096 μs，候选91.017/91.192/90.631 μs，中值 **91.017 μs**，同批改善 **3.81%**，三轮均快，距离80 μs还差 **11.017 μs**。原sanity均3.4733e-06、exit0，实际candidate `.text`全部一致。负载首sample与后置他人任务时间详见OPTX记录，不把最终idle说成首sample全部idle。

目前v153是最佳完整验证候选，四SIMD ATT采集、分析和实际完整ELF绑定已完成，交付集成尚未完成；正式persistent、默认selector仍冻结。v156/v157提前refill的原完整命令单次为94.823/92.491 μs，没有进一步收益，不采纳。共享A LDS槽方向保持放弃。所有方法、原命令复现、性能样本与hash继续写入 [OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

## 22. v153 trace 完成后的继续优化进展（2026-10-18）

最佳候选仍为 v153，同批三轮中值91.017 μs，比94.625 μs基准改善3.81%，距80 μs差11.017 μs。四SIMD ATT与该性能实际完整ELF逐字节一致；32条完整waves的compute/handoff/final epilogue为80.0184%/9.8785%/2.9202%，barrier归属stall7.8580%。这是并行wave lifetime统计，不能直接换成可节省μs，也没有HBM饱和或LDS bank conflict counter。详细分析见 [v153 thread trace分析](moe_gemm1_e64_t1536_optx_v153_thread_trace_analysis.md)。

v156/v157保持完整WG release、全部有效K贡献和双份A LDS，只在wait后更早issue refill。random17有效payload、ScaleA、GEMM2、MoE以及dynamic/poison/eager/graph replay均bitwise正确，实际463/107、LDS295936 B，无private/spill；性能94.823/92.491 μs，没有优于v153，停止这两项候选的深调。

下一组试验合并相邻两个stage的READY/RELEASE。v158/v159的CPU编译因重复奇偶WMMA body产生private196 B/VGPR spill608，未执行GPU。v160/v161把公共WMMA移出分支，CPU463/106、LDS295936 B、private/spills0。旧ISA审计无条件探索两边分支，在0x2c00 signal后报告可绕过0x2cf0 WAIT的路径；这不是已经确认的生产kernel bug。检查编译后的奇偶谓词及其saved register，加入有约束CFG证明：执行signal意味着saved mask=-1，途中没有clobber，非零EXEC经八个mask区域全路径恢复，之后相同谓词必达WAIT；偶数路径同时绕过WAIT/refill。五项负对照实际捕获saved mask错误、谓词/EXEC clobber、WAIT前memory及反转WAIT条件。该审计只证明控制流，descriptor代际、WMMA hazard与GPU数值验证保持独立。两版random17及dynamic/poison/eager/graph replay有效四项bitwise正确，原完整benchmark单次99.759/98.960 μs，没有优于v153，未采纳；同步次数减半不自动等于提速。

随后在v153三bank/split barrier上比较每WG只执行1/2个N-task，增加对应WG数量覆盖完整M/N，不减少有效计算。v162/v163源码scope、27360项M/N覆盖、独立1/2task队列模型及九项mutation通过；actual分别443/68和462/106、LDS295936 B、private/spills0。random17及dynamic/poison/eager/graph replay有效四项bitwise一致，原完整benchmark单次94.041/93.048 μs，未优于v153，不采纳。前序正式版的对照和本次供数调度不同；减少N-task不自动等于提速。共享A槽方向保持放弃。

## 23. v162–v183继续优化的进展（2026-10-18）

最佳完整验证仍为v153：三轮同批基准/候选中值94.625/91.017 μs，改善3.608 μs（3.81%），距80 μs差11.017 μs。11项routing/token、十项M_e边界、E65/shared1 fallback和dynamic/poison/eager/graph replay通过，全部有效canonical payload、ScaleA、GEMM2、MoE bitwise一致。正式persistent源码冻结，默认OPTX仍关闭，集成尚未完成。

本阶段完成的原完整GEMM1单次screen如下；均不是新的三轮采纳结果：

| 候选 | 试验 | GEMM1 μs |
|---|---|---:|
| v162 / v163 | 每WG执行1/2个N-task | 94.041 / 93.048 |
| v164 / v165 / v166 | LLVM LICM/post scheduler/expert scheduling对照 | 92.432 / 92.736 / 93.008 |
| v167 / v168 | 三/四register banks分布S预读 | 96.740 / 95.810 |
| v169 / v170 / v171 | N forward/reverse/B-major WMMA次序 | 92.240 / 92.312 / 93.803 |
| v174 | 无复制body的M32 skip | 92.525 |
| v177 | 两physical A half混合窗口 | 94.180 |

上述版本有效数值/replay通过，actual LDS295936 B、private/spills0；没有超过v153。v167后置身份不完整的KFD记录无法证明启动边界，不能声称严格无重叠。v170后置他人PID2196845约14:06:28 UTC启动，晚于本测试14:06:24退出；保留负载记录，不把退出后新任务算成本测试残留。

v172/v173复制全有效快速路径产生private460/428 B与VGPR spill1858/1290，CFG也因signal/WAIT之间scratch loads拒绝；未运行GPU。v175实验构造中`rocdl.readfirstlane`返回值未包`fx.Int32`导致`ArithValue`错误，修正版v176能编译但SGPR spill10，未运行GPU。这些是实验实现/资源失败，不是已确认的生产kernel bug。

v178/v179把最终S1 WMMA和F epilogue交错，保持全部input READY/RELEASE、I0–I3先于F/S output以及56次正向K128。但epilogue在TDM owner分支重复展开，CPU451/107、LDS295936 B、private0/VGPR spill0/SGPR spill10、静态7560条，因此停止，无GPU性能。v180/v181将最后S1及交错epilogue移到统一owner dispatch之后，按1/2个M16切片，静态降到4849/4856条、SGPR spill4，仍不通过资源门禁。v182/v183禁machine/postRA LICM后无private/spill，actual分别453/102和452/102、LDS295936 B、静态4830/4862条，random17及dynamic/poison/eager/graph replay四项bitwise正确。原完整GEMM1单次91.464/93.113 μs，没有显示超过v153的收益，停止最后S1/F epilogue交错方向，不扩大完整matrix或采纳配对。actual barrier CFG通过，不宣称由此完成全部WMMA数据hazard证明。

新共享epilogue模型覆盖2056个row/half/wave case及34952事件，缺S1、重复S1、output早于I3、缺DS完成和缺F最后K五项mutation在同一validator均被捕获；数值、barrier CFG和WMMA data hazards仍独立验证。性能前全机idle，单项120s timeout，计时期间不并发CPU编译/正确性/idle检查。实验及66项下载SHA清单见[OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)，不下载`.out`。

## 24. E65优化验收中的E64 fallback是否使用同一个GEMM1（2026-10-19）

问题：E64/topk8/shared0的完整fallback也通过同样验证，E64和E65走的不是同一个GEMM1 kernel？

回答：二者来自同一个源码launcher，但使用不同编译specialization。当前`aiter_dsr1_e65`工作树中，两者均调用`mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py::launch_gemm_a8w4_tdm_fused_persistent`，基础geometry同为`t256/w2x4/b4`、N4096/K7168。E65/topk9/shared1命中`n_experts==65/contiguous_m==30464/non-balanced`调优守卫，启用1 N-task、rowtail3、staged-ScaleA、halfA/halfScale、rowaload2、pairA和M-order1/group8；E64/topk8/shared0为`n_experts==64/contiguous_m==28672`，保持原默认4 N-tasks及未开启上述E65开关的路径。`n_experts`也是Constexpr，所以实际编译kernel本来就分别生成。

实际完整MoE验证绑定的E65 GEMM1 ELF SHA256为`7c643f5cddb07e3e431ca7d0f28c6e78d2262588c629e16abf8083fccbe3c692`，E64为`addf6e2a0436d2b17137d37757615ff8dda939b837b354fb2a7450c28e64bf9c`。这里fallback指同一launcher的原默认路径；它证明E65修改保持了E64默认完整MoE功能和精度，不代表E65优化配置已在E64启用并验收。两规模均双seed完整28项零字节差异，详见`aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md`及`dual_moe_current_downloaded_acceptance_v1.json`。本问答没有修改E64 dispatch或kernel，也没有重跑已完成的GPU测试。

用户随后明确要求：E64和E65用同一个GEMM1，不走额外编译specialization，且记录E64性能。当前两份不同ELF的旧接入不能满足这一新要求，需统一真实expert数的runtime处理、编译参数与tensor shape依赖，并以同一进程复用编译缓存项/实际ELF为证据，随后验证两规模完整MoE并分别计时。raw-LDS草稿候选暂停，尚未上传、编译或运行GPU；统一实现与新性能未完成前不把旧fallback结果当作要求已经满足。

## 25. 指定 baseline、完整 e2e 与继续优化进展（2026-10-19）

用户指定 baseline 为 commit `339589f900b3a880ec447f605886fe803cbf9649`。已用独立干净 detached worktree 重测，baseline 三轮 98.692/98.102/98.553 μs，中值 **98.553 μs**；正常入口 v153 三轮 91.915/90.864/92.456 μs，中值 **91.915 μs**，改善 **6.638 μs（6.74%）**，三轮均快，距80 μs差11.915 μs。每轮原完整 benchmark、count19、exit0、前置全机 idle，actual source/完整 ELF 已核对。此前约94 μs冻结对照来自 `3f1213...`，保留历史批次，不作为指定 baseline 或混算中值。

v153 已集成正常入口，E64/T1536/topk8 non-balanced bucket 自动启用；原 persistent fallback、旧配置开关和其他支持场景保持。E64/topk8 与 E65/shared1/topk9、T1536/K7168/inter2048、random non-balanced 的 seed0/17/43 完整 MoE e2e 均通过，八项 tensor SHA256 与指定 baseline 一致。E64 logits_diff约3.473–3.477e-06，E65约3.777–3.779e-06，原阈值0.01未放宽、全部finite；E65显式包含每token expert64、权重1，共1536条shared route。验证工具统一consumer M32 ScaleA解码后补验通过，旧decoder及首次CPU/CUDA index错误属于工具问题，未修改生产kernel。

继续优化的v184–v187只改TDM multicast timeout/cache hint，seed17两种规模八项hash均与v153一致。单次GEMM1依次91.253/91.726/92.177/92.327 μs，本批v153对照92.166 μs。v184进一步三轮交替：v153为91.068/92.508/90.642 μs、中值91.068 μs，v184为91.260/91.663/91.391 μs、中值91.391 μs；两轮更慢，未采纳。不能以一次较低screen替换最优版。actual均463/107、LDS295936 B、private/spills0；完整数据与负载限制继续写入[OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

下一组v188/v189检查完全无效capacity tile的task loop和accumulator初始化；有效TDM、56次正向K128、WG同步和canonical输出保持。CPU无spill，v188 SGPR106、v189107，VGPR463、LDS295936 B；ISA CFG门禁通过。此处只报告CPU进展，GPU/e2e及空闲性能筛选仍需完成，没有新增已采纳性能收益。共享A128/ScaleA128 LDS槽方向继续停止。

## 26. 前面的 E65 和 E64 是否走同一个 GEMM1？（2026-10-19）

**最新最优版和候选验证中，不是同一个实现路径。** E64 使用 v153 OPTX，E65/shared1 使用原 persistent fallback。seed0保存的实际编译对象给出以下kernel名：

```text
E64:
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g1_s4_sk1_st0_sn0_bcache_merge_wavewindow_wavepub153_earlybanks_splitS0_S1rows0_carryF_carryS

E65/shared1:
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e65_act1_q1r8_cn4_cm1_prefetch_apre_persist
```

selector明确要求`n_experts == 64`才进入OPTX，OPTX launcher自身也对E64设guard。因此E65的`optx_used=False`，即使显式设OPTX开关也不会绕过该guard。两者虽都是`t256/w2x4/b4/K7168`，不能据此视为同一个优化kernel。

若讨论指定commit的baseline，两种规模都来自同一个`mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py::launch_gemm_a8w4_tdm_fused_persistent`通用launcher，但`n_experts`是constexpr，分别生成`..._e64_..._persist`和`..._e65_..._persist`两个specialization，仍不是同一个编译对象。

所以此前“E64与E65/shared1 e2e均通过”的准确含义是：优化接入后的两种规模完整MoE都正确，且各自与指定baseline一致。**它不证明v153 OPTX已支持或已验证E65/shared1，也不表示E65已获得v153的性能收益。** 本次没有为达到E64数字去改变E65功能，但目前优化覆盖范围确实只到E64。源码guard、JSON的`optx_used`和actual对象名称一致，证据为`logs/v153_e2e_e64_topk8_seed0_20261019*`与`logs/v153_e2e_e65_topk9_s1_seed0_20261019*`。

## 27. E64/E65共用同一个GEMM1编译对象的实现与验证（2026-10-19）

用户明确要求E64和E65使用同一个GEMM1，不为E65增加编译specialization，并记录E65性能。本轮停止v188/v189筛选，以已验证v153为基础实现runtime expert count版本v190。`n_experts`从`Constexpr[int]`改为`fx.Int32`，加入kernel实参；kernel名使用`_edyn`，不包含物理expert数。expert查找改为runtime upper-bound bisect，M256布局和四个N-task保持一致；TDM请求、有效WMMA、READY/RELEASE、epilogue与输出地址逻辑保留。

首次真实GPU验证发现E64/E65配置表分别传入`tdm_b_th=6/0`，虽然数值八项hash已通过，cache仍出现两个对象。保留失败日志并修正selector：两种规模的OPTX均使用已验证TH6默认值，显式`AITER_TDM_B_TH`覆盖仍有效。不通过清空cache、手工别名两个条目或把两个对象改成同名来满足验收。

修正后，正常生产入口完成seed0、17、43的E64→E65→E64或E65→E64→E65验证。每个进程GEMM1 cache始终只有一个artifact，同一artifact的Python identity保持，完整ELF均为`24bb623573932faadb22ca2d89d3e84a903adc64c9e116cac4138dd5b686f93c`，没有调用原persistent GEMM1 fallback。两种规模的八项tensor SHA256均与指定baseline commit `339589f900b3a880ec447f605886fe803cbf9649`一致。E65显式包含每token expert64/权重1的第九条route，共1536条shared route；没有省略shared计算。原`logits_diff<0.01`保持，全部finite。

v190实际VGPR463、SGPR107、LDS295936 B，private及spills均0。12个signal/WAIT CFG区间通过；这不替代完整WMMA数据hazard证明。CPU dummy与实际GPU对象的`.text`均为`7b0b0ca23994eba0fd308c388c5925b356051b553ca0e31e7e738a9c4a6805ad`。本轮验证报告为`logs/optx_v190_edyn_TH6_seed{0,17,43}_20261019.json`，新工具为`verify_runtime_expert_moe.py`。

首轮原benchmark因解析器只识别`_e<数字>`而显示`gemm1 missing`，其e2e为202.91 μs，不据此估计GEMM1。随后仅增加`_edyn`的profile名称解析；原profiler duration/count、输入、迭代、warmup、reference和精度门禁保持。新增动态名称/重复候选拒绝的CPU测试通过；旧静态fixture的effective-byte格式期望失败，在修改前快照同样复现，属于既有测试期望问题，未为此改计时或性能数字。另补18项static/dynamic/ambiguity数值兼容检查全部通过，静态parser结果与修改前版本一致。

## 28. runtime-E三轮性能、replay补验及继续优化进展（2026-10-19）

用户要求汇报并继续优化。本轮完成指定baseline与v190的独立进程、三轮交替原完整benchmark。E64 baseline为99.896/100.779/99.268 μs，v190为91.346/91.688/92.059 μs；中值 **99.896 / 91.688 μs**，改善 **8.208 μs（8.22%）**。E65/shared1 baseline为155.189/155.603/153.188 μs，v190为138.050/137.586/137.596 μs；中值 **155.189 / 137.596 μs**，改善 **17.593 μs（11.34%）**。E64距80 μs还差 **11.688 μs**，尚未达标。每次原`--iters20`、profiler count19、sanity通过、exit0，前置全机idle；实际v190完整ELF均与三seed正确性验收相同，不把单次最低值当稳定成绩。

新增runtime-E replay工具使用真实生产callable，E64→E65→E64同进程复用一个artifact，在initial/sign_flip/alternate_values、poison、eager、parent CUDAGraph/runtime CUDAGraph下各重复五次，全部有效canonical payload/M32 ScaleA bitwise一致。采集fixture时完整MoE精度仍正常。首次工具中的`torch.arange`受fixture默认CUDA device影响，造成CPU/CUDA index混用；显式`device="cpu"`后重跑通过，失败日志保留。该错误在比较工具，不是kernel正确性失败。

继续优化候选v191/v192只改变block到M-tile的映射：最后一个expert的全部tiles提前、或全部有效M-tiles逆序。offset与expert数均来自runtime prefix table，不为E65编译特殊kernel。6291项映射/边界case覆盖2764176个逻辑M/N tiles，保留inactive capacity；TDM、有效K、READY/RELEASE、epilogue AST均不变。CPU均VGPR463/SGPR107/LDS295936 B、无private/spill，同进程E64/E65始终一个artifact。两版各自完成一组seed的两种规模八项hash/e2e验证。v191的E64/E65单次GEMM1为92.886/138.680 μs，v192为92.029/138.769 μs；同阶段v190 E64 control为91.806 μs。没有显示优势，停止该方向，不扩三轮或合入。详细方法、复现、性能与证据继续写入 [OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

## 29. v190新thread trace与继续优化进展（2026-10-19）

当前最优仍为v190，E64三轮GEMM1中值91.688 μs，距80 μs差11.688 μs；E65为137.596 μs。两种规模同一个实际GEMM1编译对象、完整e2e及动态输入replay已通过。新的四SIMD ATT已用用户`get_att_b_rocm_sdk.sh --all-simd --ana-att`完成，四次采集与analysis均status0；五个profiled进程保存对象和四份capture对象的完整ELF全部为`24bb623573932faadb22ca2d89d3e84a903adc64c9e116cac4138dd5b686f93c`，没有将历史v153 trace当作本版证据。

32条完整sampled waves、incomplete为空；累计wave lifetime4164139 cycles。compute区间81.1495%、三次N-task handoff9.1246%、prologue7.0177%、final epilogue2.7081%。归因stall中`barrier_wait`11.8098%、WMMA8.1574%、TENSORcnt4.0135%、DScnt2.8953%、DS loads2.3553%、STOREcnt0.9784%。这些是并行wave统计，不能直接换算为可节省μs；现有trace没有HBM或LDS bank-conflict counters。采集前曾等其他验证进程结束，立即post为空闲；post6的他人E65进程于采集退出后启动，按六秒轮询等待恢复，没有杀他人进程。详细解释将写入独立v190分析文档。

据此新增隔离候选：v193延后各task的ScaleA global STOREcnt等待到kernel退出前，所有stores和最终completion保留；v194用wave连续读prefix并ballot/popcount查找expert，runtime E不变；v196根据wave所属physical F半区，把已缓存的两行S1 WMMA移到signal/wait之间，所有K贡献、DS退休和完整WG wait保持。v193/v194已通过CPU同对象、资源与12个signal/WAIT CFG门禁，均463/107、LDS295936 B、无private/spill；v196出现一个SGPR spill，CPU拒绝、不运行GPU。修正版v197约束LLVM scalar活跃范围后为451/102、同LDS、无private/spill，CPU及CFG门禁通过。三者各一seed双规模八项hash/e2e和initial/sign_flip/alternate_values、poison、eager/两种graph各五次replay均通过，E64→E65→E64始终一个artifact；实际ELF与CPU一致。新独立 [v190 thread trace文档](moe_gemm1_e64_t1536_optx_v190_thread_trace_analysis.md)已经完成。候选正进行性能筛选，尚无新的已采纳性能收益。

## 30. 继续优化：负结果收尾与新的控制开销候选（2026-10-19）

本节是原`aiter`仓库的OPTX E64目标任务链，生产仍为v190，指定baseline仍是`339589f900b3a880ec447f605886fe803cbf9649`。正式三轮E64 GEMM1中值91.688 μs、E65/shared1为137.596 μs；两个规模同一实际编译对象、完整e2e与动态replay通过，80 μs仍未达到。后面的`aiter_dsr1_e65`补充属于独立工作树，不混用其代码、验收或性能。

新完成v203/v204改变LDS独立读取次序，原完整benchmark E64为92.409/92.637 μs、E65为138.786/139.000 μs；同批v190为92.200/137.719 μs。两规模均没有GEMM1优势，停止。虽然static DScnt等待指令由109降到106，实际没有提速；CPU资源小变或静态指令下降不是采纳条件。两版各一seed双规模八hash/e2e、initial/sign_flip/alternate_values、poison、eager与两种graph各五次replay全部通过，没有数值回退。

v193延后ScaleA task级STOREcnt的独立进程三轮曾显示E64中值快0.445 μs，但一轮慢1.520 μs；E65三轮均快。新工具保留同一HIP context，分别执行两次原完整benchmark并绑定实际ELF，三轮正反顺序E64 v190为90.931/90.549/91.492 μs，v193为90.973/91.582/91.064 μs。v193两轮更慢、中值91.064高于本批v19090.931，仍无稳定收益，停止，不替换生产最优版。此前E65收益保留为历史，不跨批拼接或作为E64达标证据。首次误传不存在的seed0验收label在host `sha256sum`失败，未启动GPU；更正为实际seed17验收并用新label重跑，失败日志保留。这是实验runner参数错误，不是kernel新bug。

首K128 C=0候选中，v198/v199/v202产生spill，CPU拒绝；v200 LLVM实际未生成inline-zero，v201加入branch-local compiler barrier后actual ISA真正使用C0，但原E64/E65耗时105.658/163.488 μs（v200更慢），方向停止。两项无spillGPU版本均完整精度和replay通过，没有降低阈值。

完整K256 step的部分展开和wave-uniform pipeline条件实验已完成。v205/v206分别出现SGPR spill1、private44 B/VGPR spill10/SGPR spill23，v209禁postRA LICM仍spill1，均拒绝GPU。v207 scalarize tensor-wait原条件，v208另scalarize task/refill/expert guards，均462 VGPR/106 SGPR、LDS295936 B、private/spills0；v210两步展开禁machine/postRA LICM为451/102、同LDS、无spill。三版双规模八hash/e2e与动态输入/poison/eager/两graph各五次replay均通过，同一个artifact和actual ELF。原完整benchmark E64/E65分别为92.923/139.536、93.322/142.349、92.820/137.443 μs；同批v190首91.106/137.883、尾92.531/137.299 μs。没有可采纳的稳定收益，停止三版，生产v190保留。171项/71116596 B证据已下载并核SHA256，原pre/post samples与独立诊断进程所有权疑点保留。

## 31. 继续优化：prepared TDM descriptor与固定stride实验（2026-10-19）

用户继续要求汇报并优化。原`aiter`生产最优仍是v190，指定baseline commit `339589f900b3a880ec447f605886fe803cbf9649`；正式三轮E64为99.896→91.688 μs，改善8.22%，距80 μs差11.688 μs，E65为155.189→137.596 μs，改善11.34%。两种规模共用同一个真实编译对象，完整e2e、精度和dynamic replay正常。后面独立`aiter_dsr1_e65`工作树的实现、性能和trace不用于这条任务链。

v211/v212把owner不随K变化的TDM GROUP1、current/next global base、LDS base/rotation提前到task入口准备。v211保留四owner外层dispatch，v212合并prepared字段及compute body，仅B的TH6 immediate分派在K-loop外。原INT32_MAX inner extent、runtime OOB clamp、48-bit stride、multicast/early-timeout、28次input请求、四级ring、READY/RELEASE、56次正向K128和SiLU/BF16/FP4输出均保持。999936项整数descriptor/address case覆盖E64/E65、31种tail、snake、跨task和4 GiB carry；模型不代替实际GPU验收。

CPU v211为VGPR462/SGPR105、static4654，v212为461/98、static3460；LDS均295936 B、private/spills全0，CFG分别12/6个区间通过。v211 seed17、v212 seed43两种规模完整MoE八项hash与指定baseline一致，finite及原logits gate0.01正常；dynamic initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph各五次replay通过。每版E64/E65一个artifact，actual完整ELF与CPU相同，E65全部1536条shared route保留。

| 本批版本 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190首control | 91.534 | 136.838 | 204.05 | 280.91 |
| v211 | 91.498 | 137.961 | 202.00 | 282.22 |
| v212 | 93.088 | 140.011 | 204.39 | 285.72 |
| v190尾control | 91.471 | 137.264 | 204.05 | 281.47 |

原完整`--iters20/count19`、sanity通过、exit0；八次pre全机idle，立即post无KFD但busy2–3、六秒及post6恢复idle，计时期间CPU会话闲置。v211 E64与对照持平但E65更慢，v212两规模更慢，两版不采纳、不扩三轮。92项/13323968 B证据已下载并核SHA256，无`.out`。静态指令减少不是性能收益，不能据此宣布达标。

v212 actual ISA在steady refill使用runtime shift，并把Int32 K offset sign-extend到64-bit。v213在K-loop外将A data、B data和共享scales分为固定stride分派；v214另使用Int64形成K byte offset，保留完整global carry。两版2688项新增地址/stride case通过，CPU465/104、LDS295936 B、private/spills0、static4172、九个CFG区间通过，双规模完整e2e八hash和dynamic/poison/eager/两graph各五次replay全过。原完整E64/E65分别94.243/138.431、92.892/139.075 μs；同批v190开头92.308/138.255、结尾91.437/137.565 μs，两版不采纳、不扩三轮。93项/13690522 B证据已下载核SHA256，archive SHA256=`b68a7fd486d704852c7739786763752f070bd36144a261b592c582a70d00f0d3`，无`.out`。post原样保留：立即busy1–3、六秒及post6恢复idle，两项control包含身份不完整的`<已退出>`条目，不能作严格所有权结论。源AST/整数模型只证明相应语义，不称代码生成差异是回退唯一原因。

方法、完整复现命令、原profile与actual ELF汇总见[OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)，瓶颈解释见[v190 thread trace分析](moe_gemm1_e64_t1536_optx_v190_thread_trace_analysis.md)。所有自有命令120s上限；共享A LDS槽方向继续放弃。

## 32. 继续优化：runtime-E 的空 capacity tile（2026-10-19）

原`aiter`任务继续以v190正式最优为准，E64/E65为91.688/137.596 μs，E64距80 μs差11.688 μs；以下不引用独立`aiter_dsr1_e65`工作树数据。v213/v214无稳定收益后停止。新v215把整个四N-task loop的空capacity判定外提，v216只移动accumulator初始化。有效tile的所有TDM、正向56次K128、READY/RELEASE、量化与输出函数保持；同WG和四cluster N peers的live predicate一致，kernel末尾tensor_wait保留。

144组fixture、323328个M/N-task判定case覆盖空expert、partial tile、capacity tail和E65/shared1536；静态AST/整数模型不代替GPU。CPU v215为VGPR463/SGPR106/LDS295936 B、private/spills0、static4708、12个CFG区间通过，actual完整ELF=`77cb2b852f3c71ba497068f61251dcbb0474bfced9d8e0f1cafe02a3f518c637`。v216资源463/107、static4710，actual`.text`与v190相同，停止而不重复GPU计时。

v215 seed17双规模完整MoE八项hash与指定baseline一致，original logits gate0.01、finite及全部1536条shared route保持；E64→E65→E64一个实际artifact/ELF，没有fallback。dynamic initial/sign_flip/alternate_values、poison、eager/两graph各五次replay通过。原完整空闲筛选v215 E64/E65为92.048/138.183 μs，首v190为91.886/137.569、尾为91.873/136.635 μs，无收益而停止，不扩三轮。六次pre均全机idle，立即post无KFD所有者、busy2–3，六秒/post6恢复idle；测量时CPU会话闲置。77 regular-member包已下载逐SHA核验，SHA256=`c3c4f53ab0da7a1ce745f90346266f3464149eba9bb965d30663a00426da70b1`，`.out`未下载。具体source/model/命令与实际数据写入[OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。

## 33. 继续优化：SiLU epilogue 指令分组（2026-10-19）

原`aiter`生产仍是v190，E64/E65正式三轮中值91.688/137.596 μs，E64距80 μs差11.688 μs；指定baseline为339589f900b3a880ec447f605886fe803cbf9649。新的v217/v218/v219仅在隔离helper将tanh的CHUNK从4改成2/8/16，所有输入clamp、原SiLU公式和舍入输出不改，其他compute/sync AST保持。2145项index case与CPU同artifact检查通过，资源463/107、LDS295936 B、private/spills0、12 CFG区间通过。

v218与v219生成相同`.text`，v219只完成CPU，不重复GPU测试。v217 seed17、v218 seed43双规模完整MoE八项hash、原精度门限和全部1536条shared routes通过，一个actual artifact/完整ELF；dynamic三输入、poison、eager/两graph各五次replay亦通过。原完整性能筛选已结束：v217 E64/E65为92.898/138.794 μs，v218为92.521/137.409 μs；同批v190首对照90.994/137.599、尾对照91.523/136.918 μs。两版均未改善E64，停止、不合入、不扩三轮。

候选pre首sample均全机idle，post无KFD/busy2–3，6秒及post6恢复idle；E64尾control先等待他人MLA结束才测试，其post有身份不完整退出条目、post6有独立工作树CPU compile，E65尾control post6亦有退出条目，均后续恢复idle。这些所有权边界原样保留，不统一宣称全部control首sample均idle。冻结包116 regular members，SHA256=`e6e342ae16bf09dfcc24ab728bda454726a524f4d472f3558bcc0a29d7f515ce`，115项23006259 B已本地逐项核SHA，未下载`.out`。依据SPG §5.3.5 pp220–221独立main/TRANS管线调整分组，但这组实现没有取得目标收益。完整方法/数据/复现见[OPTX记录](moe_gemm1_e64_t1536_optx_optimization.md)。

## 34. 为何 `--scenario kernel` 与 `--scenario bench` 的 GEMM1 耗时不同？（2026-10-19）

原`aiter`v190两种scenario实际完整ELF、`.text`、JIT cache key及launch参数相同，均t256x256x256/w2x4/b4、cn4/cm1、TH6、act1/q1r8，没有因scenario换kernel或关闭fused quant。kernel先执行完整MoE，再捕获固定tensor的partial，单独连续测GEMM1；bench每轮执行完整MoE，在routing、quant、GEMM2、combine的上下文中提取同一个GEMM1的device duration。

均原warmup5/iters20/testGraph=False/use_cuda_event=False，丢第一轮后count19。kernel的GEMM1区间只有一个device kernel，bench的GEMM1行也没有把其他kernel或Python gap加进来。日志`avg ... from cuda.Event`来自额外同步循环，不是最终GEMM1数值；零参数callback的rotate_args没有轮换weights。

正常原命令kernel85.9759/87.0495 μs，bench92.2233 μs；诊断批次86.760579/91.074316 μs，差4.313737 μs，批次分开。原bench646.850 μs与外部stress PID1532804–1532809重叠，排除性能。route、counts、psum、W1/W1 scales一致，A/ScaleA整块hash含capacity padding，有效行decode待补齐。两种模式MoE原gate0.01通过，logits约3.473346e-06，但隔离计时的e2e校验来自最初完整调用，未重新compose末次隔离结果。

执行前序、中间地址复用和提交节奏不同已证实；局部cache、memory请求和时钟是待验证机制，没有counter或时钟采样，不能细分成本，不能说896 MiB的全部W1驻留192 MB L2。固定地址G2/quant前序对照已启动但未取回，a07-3随后hang，按用户要求暂停全部远程操作至重启通知，不报告对照结果或hang原因。正式E64/E65仍91.688/137.596 μs，80 μs目标保持完整bench口径。

完整回答、原数据、源码/事件范围、ELF/hash、复现与未完成项已写入[GEMM1 kernel/bench计时差异分析](moe_gemm1_kernel_vs_bench_timing_analysis.md)。本地校验22项1092803 B，两次r2原命令SHA与远端一致；诊断远端二次checksum及有效输入核对待重启。独立`aiter_dsr1_e65`数据只在以下补充章节记录。

## 35. 机器暂停期间继续了哪些本地优化工作？（2026-10-19）

用户要求汇报并继续。原`aiter`任务仍使用指定baseline `339589f900b3a880ec447f605886fe803cbf9649`，生产最优v190的原完整bench E64 GEMM1三轮中值 **91.688 μs**，比baseline99.896 μs快8.22%，距80 μs差 **11.688 μs**；E65/shared1为 **137.596 μs**。两种规模同一个实际编译GEMM1、完整MoE精度及dynamic/replay已通过。a07-3仍按此前指令等待本会话用户明确通知重启，本轮没有SSH/SFTP、上传、远端编译或GPU测试，不将独立工作树的恢复消息和数据用于本任务。

本地新v220把内部N256/8waves改成N128/4waves、N4096两个N-group，完整M256 owner、四个persistent N-tasks、双份physical A128/ScaleA128、每个有效accumulator K0..K55和原quant/output保留。LDS模型295936→218112 B，即289→213 KiB；实际VGPR/SGPR/LDS metadata、spill及性能尚未取得。213 KiB大于每WGP320 KiB LDS的一半，不据此宣称两个WG常驻。原按owner waves识别B/ScaleB在新几何会混淆，候选用tensor/layout字段区分；两descriptor/wave的wait重推为initial6、steady4、successor前三步6、tail2/0/0，旧较小wait属于更保守等待，不能说必然有race。硬件依据和完整推导见专项优化记录。

读取冻结seed0真实E64/E65 route counts，分别有12288/13824条有效route、64/70个live M256、797/893个live M16，E65保留1536条shared route，未拼用E64 routed counts。v220令live WG从256/280增到512/560，input TDM指令合计229376/250880增到458752/501760；有效WMMA、总wave instances、output TDM和有效输出量不变。A/ScaleA重复source请求翻倍，B/ScaleB总surface不变。这些是源码和descriptor账目，未测HBM流量、cache或时钟，不能把LDS下降写成性能收益。

1983 routing fixtures、actual-AST owner/N/reader/output及跨N-task地址模型通过，七项协议负对照全部拒绝。已准备CPU/ISA→双规模完整八hash/原精度→dynamic/poison/eager/graph replay→原完整bench的验收入口，并绑定source/model/ELF/resource/CFG。原benchmark的N128 `_edyn`解析无需修改；本地两规模解析和11项损坏receipt拒绝检查通过。工具检查采用已有v190 ELF与temporary synthetic metadata，不是v220编译或GPU验收；candidate源码、生产selector及原计时逻辑未改变。

本轮记录、logical成本表、源码/model路径及fresh-label复现命令写入[OPTX优化记录v220章节](moe_gemm1_e64_t1536_optx_optimization.md)，总状态同步[non-balanced优化记录](moe_gemm1_e64_t1536_nonbalanced_optimization.md)。本地文档尚未同步远端。恢复后先读取hang前固定context日志并核对有效A/ScaleA，再执行候选门禁及单轮首尾v190 control筛选；没有新的GPU性能数字，80 μs目标仍未达到。以下独立`aiter_dsr1_e65`章节保持原有内容。

用户再次要求汇报并继续后，补齐本地source/scope与命令复核：候选26个受保护函数、LLVM compile policy和MoE/replay的有效输入/输出snapshot functions均保持冻结AST，四个host runner容器/timeout/性能区间检查通过；六份生产source SHA匹配。本地FlyDSL支持把Constexpr当Python值进入body并直线重赋值，但远端版本及实际lowering待核对，不宣称候选已能编译。本轮仍未连接a07-3或运行GPU，正式成绩不变。

最新本地descriptor范围/对齐检查已通过：115136个actual-AST input descriptors、14392项extent和512个output descriptors，覆盖M-tail 0..256、全部K tile及两个N-group；确认A/ScaleA有效范围、LDS区间不重叠及relative global/stride对齐，三个故意引入的地址/范围错误全部拒绝。结果为 `logs/optx_v220_alignment_extent_local_20261019.json`，12.850秒、exit0。实际absolute pointer、发射descriptor、direct-copy及bank/cache行为待编译和GPU验收，没有新的性能成绩。helper与证据保存至 `frozen_helpers/runtime220_n128_fourwave/`，清单与SHA见 `logs/optx_v220_local_preparation_inventory_20261019.json` 和 `logs/optx_v220_local_preparation_20261019.sha256`；冻结副本保存版本，复现仍从task根目录执行。原生产kernel、selector和计时工具保持前述SHA，下面的独立工作链内容保留。

## 36. a07-3 重启后，scenario 差异任务完成了吗？候选进展如何？（2026-10-19）

用户已明确通知a07-3重启，本任务恢复远端诊断；生产v190、selector、原benchmark和计时工具保持冻结SHA，不清理或切换远端分支。重启前固定context日志已取回，只记录全机idle等待，未进入测试，因此不能把机器hang归因于该实验。重启后的首版context helper误收fixture随后的BF16 stage capture，出现`quant_a2`并覆盖同名callable，任何context计时前exit1；修正为第一次正式GEMM2 callable追加后立即撤capture，确认GEMM1 `stage1_quant_out=1`、actual完整ELF仍为v190。这是诊断helper问题，没有修改生产kernel功能。

**`--scenario kernel`与`--scenario bench`差异诊断已完成。** 两模式同一v190实际完整ELF/`.text`/JIT key/launch参数，原warmup5/iters20/count19/device duration公式保持。新诊断隔离GEMM1为86.037263 μs、完整bench为90.783632 μs；原完整MoE `logits_diff=3.473346e-06`、gate0.01通过。这些包含计时前输入readback，仍是诊断批次，不替换正式E64/E65三轮中值91.688/137.596 μs。

有效物理行A/ScaleA hash确实不同，不应说成仅padding不同。进一步捕获首次正式quant的`topids_to_rows`，确认12288条route唯一覆盖各expert有效范围，按token/topk顺序解码后A payload/ScaleA SHA256分别为`e0fc76ae1443de8061ace63d6aa0fda764bd3e452b2be37af01105858e7bb2b7` / `533257615940eb87b1ecf8969f48df3849dbcb88fb021976364c4406db8ced48`，两模式相同；hidden、route_ids/weights、counts和psum也相同。源码per-expert atomicAdd分配slot使expert内物理行序变化，不能据物理hash认定有效输入改变，亦不能把行序变化定为性能主因。

固定字节/地址/code object前序对照G1-only first、G2→G1、quantA1→G1、G2→quantA1→G1、G1-only last分别88.384368/88.556895/88.019263/90.374105/85.978211 μs，输入hash均保持。首尾2.406158 μs漂移如实保留；支持相同kernel受运行上下文影响，不能把全部差值分摊成cache、频率或某个前序kernel的固定惩罚。性能窗口前全机idle；立即post无KFD、busy2，六秒检查的其他`dev_al`进程约在本测试退出4秒后开始，原post/等待不改写。完整原始receipt、时间轴与限制见[GEMM1 kernel/bench分析](moe_gemm1_kernel_vs_bench_timing_analysis.md)。

v220内部N128/四-wave候选实际452/107、LDS218112 B、private0/VGPR spill0、6个已审核SGPR lane spills；调整审计helper的input-prime分类后通过CPU/CFG。v220和v225 N128/b2已补齐三seed双规模完整MoE及dynamic/replay，均同artifact/完整ELF、精度正常，但E64/GEMM1为133.557及127.767 μs，E65为176.670及208.077 μs，均回退，停止N128方向。两者不属同批，不能以相差5.790 μs声明b2稳定提速。249项/44779452 B证据已下载逐SHA核验，未下载`.out`。

v226 A unicast RT和v227 ScaleA multicast early只改变TDM policy，CPU分别463/106、463/107、LDS295936 B、private/spills0；三seed双规模完整八hash及dynamic/poison/eager/两graph replay全部通过。原完整screen E64为92.957/92.582 μs，同批v190首尾90.766/90.735 μs，均无收益，停止而不合入。E65为139.077/137.873 μs，同批首尾138.035/137.919 μs；保留v226 E65退出后独立进程的ownership边界记录，不将全部post写成首sample idle。129项/18698762 B完整证据已下载逐SHA，汇总和复现见[OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。这些是负结果，不是新增生产bug修复。

v228仅跳完全无效A M16及ScaleA M32的LDS reads，partial group、有效K累加、B/ScaleB cache、request数/READY/RELEASE和quant/output保留。CPU461/107、LDS295936 B、private/spills0、12 CFG通过；seed17 E64→E65→E64完整八hash/原0.01门限、1536 shared routes、同artifact/ELF和dynamic/poison/eager/两graph五次replay全部通过。原完整screen E64/E65为107.627/171.426 μs，同批v190首尾为91.441/92.734与136.829/137.242 μs，明显回退，因此停止、不合入、不扩三轮。六项count19、exit0，所有pre idle；立即post无KFD/busy2–3，六秒后及独立post6均idle。69项/10191253 B证据下载逐member/archive SHA通过。

后续TDM surface缩减v229因CPU SGPR spill8及原CFG规则拒绝，未运行GPU，不能称为生产bug或精度失败。v230只缩完全OOB的A groups，保留最小两行、原ScaleA/DS/WMMA与有效K；CPU465/106、LDS295936 B、private/spills0、12 CFG通过。实际源码descriptor审计115136项与四个错误字段负对照通过；seed17双规模完整八hash、1536 shared routes和dynamic/replay通过，同artifact/完整ELF。21个短tail/空expert/hot8完整MoE边界case也均八项mismatch0。首轮失败来自v190 parent解码工具CPU rows与默认CUDA arange混用，仅显式指定工具`device="cpu"`并保留旧helper/失败log，不是kernel bug。原完整screen v230 E64/E65为93.858/141.196 μs，同批v190首尾91.517/91.781与138.061/136.922 μs，两种规模回退，停止、不合入、不扩三轮。所有pre idle，立即post无KFD/busy2–3，六秒后及独立post6 idle，无本任务残留；123项/19321773 B证据已下载逐SHA核验，不含`.out`。两版只在隔离候选中试验，详细事实、限制及复现见[OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。生产仍v190，原三轮正式E64/E65中值91.688/137.596 μs，距离80 μs仍差11.688 μs；以下独立工作树章节保持原有内容。

N256两级input ring v231完整验收通过，但原完整bench性能回退，停止。两种规模seed17八hash、同artifact/完整ELF、原0.01门限、1536 shared routes、dynamic/poison/eager/两graph五次replay和21个空expert/短tail/hot8完整MoEcase保持；CPU463/104、LDS156672 B、private/spills0，原12个CFG区间通过。E64/E65 GEMM1为129.015/220.959 μs，同批v190首尾91.361/91.334与138.822/137.481 μs，未合入、未扩三轮。v232三级ring实际489/107、LDS226304 B，SGPR spill18且原prime CFG规则拒绝，没有GPU正确性或性能数据，不是已证实的生产bug。actual-AST工具首版超过120秒被终止，只缓存compiled AST代码后新label通过，原assert/coverage不变，无候选数值或计时修改，旧helper/timeout证据保留。127项/18926348 Bsource/model/native/完整MoE/replay/边界/性能和helpers已下载逐member/archive SHA核验，不含`.out`；详细硬件出处、对象及ownership边界见[OPTX记录](moe_gemm1_e64_t1536_optx_optimization.md)。下一阶段只在隔离v233/v234/v235做三级LLVM policy CPU筛选，不继承两级GPU验收、不放宽原门禁；生产正式v190 E64/E65三轮91.688/137.596 μs保持。以下独立工作树章节保持原有内容。

## 补充：aiter_dsr1_e65 共用 GEMM1 默认接入与继续优化（2026-10-19）

本节仅记录 `aiter_dsr1_e65` 工作树、a07-3 的 `aiter_t256w2x4_e64_20261007` / `dsr1_ep4` 任务链。原仓另有 OPTX 优化链，本节不沿用其实现或性能数字。用户要求 E64/E65 使用同一个 compiled GEMM1、记录 E64 性能，现已在本工作树正常生产入口完成。

仍采用原 `t256/w2x4/b4`、SiLU fused-quant 数值 pipeline；共同 Constexpr selector 为 `n_experts=0`，真实64/65通过 `runtime_n_experts: fx.Int32` 传入 lookup、M-order 和 tail guards。两规模共同1 N-task、rowtail3、staged ScaleA、half A/ScaleA prefetch、rowaload2、pairA、M-order1/group8及B hint0。真实 tensor、weights、routing 和全部有效 routes 保持，E65 的1536条 shared route保留。没有把E64补成E65或通过清cache/改名伪造复用。lookup固定7步，经37345项实际CPU案例验证；比lookup8 native少16条scalar指令。

默认按E64/contiguous_m28672与E65/contiguous_m30464、non-balanced/t256共同scope自动选择，无需额外UNIFIED开关。显式N_TASKS4/2、rowtail/预读/M-order等旧开关、balanced、t192、direct-scale、其他expert count/capacity保留原参数路径。6项共享默认与76项旧参数路径CPU对照通过；未把旧非法override组合变为新增支持。

正常production imports同一实际GPU进程分别完成seed0/17的E65→E64，GEMM1 cache一直一个对象，Python identity及完整ELF相同：`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`。native VGPR486/SGPR68、LDS282624 B、private/spills0、kernarg180 B，真实expert count offset176/size4。共112项零字节差异包括HEAD有效payload/ScaleA/GEMM2/MoE、eager/graph canary和dynamic A/ScaleA/psum。原 `logits_diff<0.01` 保持，E65约3.779e-06、E64约3.473e-06。GEMM2和公共helpers不改，benchmark仅增加 `_eruntime_` 解析与对应fixture；计时、输入、计算/reference AST与HEAD一致。旧TB/s fixture期望的既有失败修改前后相同，未为此改性能数据。

最新原完整benchmark、warmup5/iters20/eager、每轮120秒上限结果如下。性能前后全机GPU/KFD检查；E65第一组三轮ownership不合格整组排除，E65第二组和E64第一组三轮合格，不按耗时挑选或拼接。

| workload | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|---:|
| E65/T1536/topk9/shared1 | 107.172 /107.379 /108.015 | **107.379** | **94.3843** | **256.893** |
| E64/T1536/topk8/shared0 | 94.9283 /95.3059 /95.7361 | **95.3059** | **60.4664** | **209.078** |

80 μs目标未达到，E65差27.379 μs。共用lookup7上一批107.752 μs的小幅收益未在E65反向稳定复现：lookup8为108.155 μs，两侧lookup7为108.884/107.640 μs；E64两侧lookup7为95.6390/95.0338 μs，均快于lookup8的96.0271 μs。当前采用lookup7不宣称E65稳定提速纪录，不混用另一OPTX工作链数字。

共用M group4通过112项精度，但E64三个完整性能组均ownership污染，上限后停止并恢复源码，未采纳孤立E65中值107.414 μs。两pair A预读初版native4916条、VGPR spill4、private12 B、各4条scratch load/store，资源门禁拒绝，未运行GPU；编译rc0后host收尾manifest路径错误独立保留。这些是实验工具/资源失败，不称生产kernel新bug。继续优化的有限候选将B1在全部当前K128 consumer结束后共享复用，结合lookahead2；CPU2080项pipeline/1046528次实际WMMA的operand、hint、56次原序K128和TDM/fence事件一致，正在no-dispatch编译，尚无新GPU性能。

已有瓶颈方向依据来自旧group8 ATT：B/ScaleB owner steady tensorwait2中值3 cycles/每wave累计72，完整896 WMMA wave compute约22177 cycles；长WMMA gap包含DS loads、row predicate、地址与同步工作，不能全当可消除stall，也不能换算为kernelμs或唯一归因HBM/FIFO/cache。该ATT属于旧 `7c643f...`，不冒充共用新ELF。硬件资料为Shader Programming Guide #65 §2.4、§3.3.2.1、§4.6.12、§4.10.1，未从文档冲突推断WMMA/tanh必然并行。

详细记录见 [aiter_dsr1_e65 优化文档](../../aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md)。默认raw包986792 B/128 regular members，SHA256=`f22c35958091355830a9c1e3f1d339994027daa4498866988ea05090dd9bb1a2`，逐member/ELF/profile/ownership独立验收通过；失败候选raw SHA=`53cfb15b0598e5b9cb9c3101fa4bf22559684c477a6bc17cc145246ea257e4cf`、反向raw SHA=`95f1dcc8b5242e0408244f313239a0802ea31842e0c9d2051957c51c5b503449`，均验收。当前本地/远端wrapper SHA=`798a865d7c11a0914b3c335d141948cbf77384d12135e822e339487b474e60b1`、persistent SHA=`29cfe04e826aa2617c4c1f57cb4fd0a0211f615ca14715a5bba227b35cda7d64`一致；默认验收最后全机空闲、无本轮GPU残留，未commit/push。

## 36. a07-3恢复后原aiter任务的最新进展（2026-10-20）

用户要求汇报并继续。原`aiter`任务的指定baseline仍为commit `339589f900b3a880ec447f605886fe803cbf9649`，正式最优v190的E64 GEMM1中值 **91.688 μs**，对baseline99.896 μs提升8.22%，距80 μs差 **11.688 μs**；E65/shared1为 **137.596 μs**。两规模共用一个真实artifact及完整ELF，完整MoE精度、八hash与dynamic/replay验收保持。本段不采用后面独立`aiter_dsr1_e65`的代码、性能或ATT结论。

a07-3已按用户通知恢复，branch/HEAD/boot/container及生产SHA重新核对。v220 N128/b4和v225 N128/b2均完成seed0/17反向/43完整双规模八hash和dynamic/poison/eager/两graph五次replay，mismatch0且E65的1536条shared route保留；性能均明显回退，停止、不合入、不扩三轮。v220同批首control/候选/尾control的E64为91.717/133.557/93.094 μs，E65为138.200/176.670/138.248 μs；v225为90.838/127.767/92.597与138.121/208.077/138.117 μs。N128两批分开，不跨批比较b2/b4的稳定收益；没有新ATT/counters，不把更小LDS或request账目当作已测occupancy或唯一回退原因。

继续从v190 N256隔离A/ScaleA TDM策略：v226只取消A multicast且保留TH0/RT；v227只令ScaleA使用同四N-peer multicast/early timeout。归一化policy/symbol后完整AST等于v190，LDS地址/request数/READY/RELEASE/全部有效K顺序/输出/runtime-E ABI保持。硬件依据为Shader Guide #65 §4.9.8 p191、§4.10.3及Table80 pp201–203、Table19 p62，multicast绕过WGP$而unicast可能允许局部cache复用，净效果待实测。实际CPU两候选VGPR463、SGPR106/107、LDS295936 B、private/spills0，E64/E65同artifact/完整ELF；ISA/CFG通过。v226完整三seed八hash已通过，replay进行中；v227 GPU验证和候选性能尚未完成，未提前报告收益。

原kernel/bench差异临时任务已完成，正式目标仍使用原完整bench，不用隔离约86 μs替代。新candidate只通过进程内snapshot导入，生产source/selector/原计时未改变；单项120秒、性能全机idle前置和后置门禁保持。方法、资源、完整数据与fresh-label复现已写入[OPTX优化记录](moe_gemm1_e64_t1536_optx_optimization.md)。N128归档工具遇到缺少本地准备helper和高压缩110秒门禁，已补齐并冻结当时helper，失败记录保留；分shard下载SHA验收仍待完成，不称已成功归档。

## 37. 原 aiter 三级 ring 候选修了什么，最新进展如何？（2026-10-20）

本节仅记录原`aiter` E64优化链，不混入后续独立`aiter_dsr1_e65`任务的代码和性能。指定baseline仍为commit `339589f900b3a880ec447f605886fe803cbf9649`，正式最优v190的E64/E65 GEMM1三轮中值为 **91.688 /137.596 μs**，E64对baseline99.896 μs改善8.22%，距80 μs差11.688 μs。两规模共用实际artifact和完整ELF，原完整MoE精度保持。

新S3 v236/v237在stage地址使用空inline asm `=s,s`，输入输出可能不同SGPR，空asm并不实现copy。两版资源/模型/原CFG通过，但完整E64原logits失败。隔离v238/v239新增局部`_stage_scalar_identity`，采用`=s,0`绑定同一SGPR，kernel地址/协议/有效计算、quant/output不变。依据LLVM LangRef Input constraints lines5611–5618；不修改共享`gemm_common_gfx1250.py`，也不把本轮候选失败推广为生产v190普遍数值bug。

完整v236→v238 `.text`30080 B只变两字节，VMA `0x1b00`、offset15574/17498分别对应PC `0x57d4`目标s12→s14、`0x5f58`目标s86→s87；后续consumer原先即读s14/s87。字节对照、ISA连线与修复后的完整GPU结果共同支持绑定bug结论。v237→v239重排较多，不冒称只有两条变化。模型另有复制scope后function globals未更新的工具bug，62误匹配→124判定通过；只修工具绑定，不改candidate或assert，失败诊断保留。

v238/v239为448 VGPR、102/98 SGPR、LDS226304 B、private/spills0。两版actual-AST全地址/协议及八负对照、原12CFG区间、seed17完整E64→E65→E64八hash与指定baseline相同，原0.01门限/finite及1536 shared routes保持。每版同CPU/GPU完整ELF、cache1，没有E65 specialization。换输入/poison/eager/两graph各五次replay payload/ScaleA mismatch0，21个空expert/短tail/257/E65 hot8完整MoE边界八项mismatch0。

原完整bench单轮screen的v238 E64/E65 GEMM1为 **94.779 /148.168 μs**，v239为 **96.272 /149.593 μs**；同批v190首尾91.798/91.013与137.323/136.899 μs。两种规模都回退，停止、不采纳、不扩三轮。所有pre idle；立即post无KFD、busy2–3，六秒复查和post6恢复idle，无本任务残留，计时期间无本任务并发编译/传输/正确性或idle polling。资源下降没有带来收益，没有候选ATT/counters，不唯一归因cache、HBM或occupancy。

295项/58300254 B completed evidence已分五shard下载逐SHA验收，无`.out`；summary v1的e2e regex只在独立v2修正并重新解析原raw，不改计时、不重跑GPU、不覆盖冻结v1。完整字节/对象/方法/复现和性能表见[OPTX记录](moe_gemm1_e64_t1536_optx_optimization.md)。后续v240恢复默认LICM出现SGPR spill2，CPU拒绝、未GPU；v241为461/107/LDS226304 B、private/spills0，actual-AST/原CFG、seed17完整双规模八hash、dynamic/replay及21个完整MoE边界case全部通过，同完整ELF `fe149f0d3a292450e8f4b1ea7ef74517f113a35b7a451b095814c6a494c33cff`。原完整screen E64/E65为96.590/148.227 μs，慢于首尾v190的91.559/91.793和137.455/137.542 μs，停止本轮S3对照、不扩三轮、不合入。126项/19306862 B完整证据已下载逐SHA核验，逐轮pre/post/post6及后置独立任务原样保留；不以最后idle替代逐轮边界，无`.out`。

用户要求继续后，隔离v242/v243/v244给两侧Scale LDS base、仅A、仅B添加局部`=v,0` identity。原地址、DS有效读量、同步、有效K与quant/output不变；actual-AST59392地址/475136scale consumer words通过。CPU均零private/spills、原CFG和E64→E65→E64同artifact通过，但总指令4719/4726/4711超过事前4710门槛，停止CPU、未GPU，不称为精度bug或实测性能回退。仅B的scale issue54→48、`s_wait_dscnt`109→100值得继续检查，v245与原v215无效capacity guard组合，actual源码lookup/guard144fixture和323328判定通过。

v245现已完成CPU和完整GPU验收：实际464 VGPR/106 SGPR/LDS295936 B、private/spills0、4709条指令、原12CFG通过；完整ELF=`7fbd5d7631aa1c6fdea33c24f9729cb9f9e475fd656900517cd438ed0a9f70f1`。seed17 E64→E65→E64八hash与指定baseline相同、原0.01 gate/finite和1536 shared routes保持，同artifact/cache1/完整ELF，没有E65 specialization；dynamic/poison/eager/两graph每路径五次replay mismatch0，21个完整MoE边界case八项mismatch0。

原完整screen的v245 E64/E65 GEMM1为90.697/137.186 μs，同批v190首尾91.338/91.983和136.878/137.517 μs。进一步三轮独立进程交替，E64 v190/v245中值92.006/91.888 μs，仅差0.118 μs且候选第三轮慢1.322 μs；E65中值137.932/137.715 μs。同HIP context E64三轮中值92.131/91.034 μs，但E65为137.491/137.759 μs、反向一轮候选慢1.621 μs。综合未确认稳定收益，v245暂不采纳。275项/37181076 B完成证据已下载逐archive/member SHA、CPU gate、GPU完整ELF、数值及逐轮pre/post/post6独立核验，无`.out`；后置外部任务与`<已退出>`记录保留，不把后续idle当作首post就idle。

v246/v247仅metric bias0/100、v248仅ScaleB raw LLVM load、v250/v251合并DS issue hints均与v245同`.text`，CPU停止、未GPU；v249两侧raw load为4715条指令超过原4710门槛，也未GPU。这些是无native收益或筛选拒绝，不是新增已证实数值bug。v252/v253的独立Scale/operand读取次序已完成native，20个actual-reader AST请求multiset/bank相同，但实际4741/4773条指令超过原4710门槛，CPU停止、未GPU。生产仍v190，正式E64/E65三轮91.688/137.596 μs、距E64目标11.688 μs保持。以下独立工作树章节及其数据保留。
### prepared/body 后续、异常重测与暂停

目前正式最优仍是基于OPTX方案的v190，不是本轮未采纳候选。v254 prepared owner body+ScaleB tied/capacity guard在原完整三轮E64 v190/候选中值92.371/91.462 μs，但第二轮候选慢1.058 μs，E65中值137.620/137.704 μs，收益不一致；v255 shared body E64/E65单轮91.058/140.595 μs，E65回退。两版完整MoE/replay/21边界均通过，未采纳。v256–v263只调整原signed row validity，八版完整双规模八hash、原0.01门限、同artifact/完整ELF、dynamic/poison/eager/两graph五次replay和每版21边界通过，但性能均无稳定收益。

v263先前ISA CFG审计失败是工具bug：encoded 65519应sign extend为-17 DWORD offset，不能当正跳转；backward PC layout也不等于CFG cycle。独立v2按实际encoded字段及有向图拓扑审计，真实六区间/八个backward edges通过，真cycle/memory/越界signal区间/重复signal/错encoded位移/缺DS退休六负对照拒绝。依据CDNA5 ISA §5.5/§15.5 printed pp288–289，`PC_next + signed_offset*4`；只修工具，保留旧失败，不改kernel或放宽精度。v261边界首批漏`--version`、rc2，改用明确成功的`edges0_5_20261020_r1_args2.json`，只修调用/验收文件名。

v264/v265从v255分别只给A/B data LDS base加`=v,0` tied identity，地址AST覆盖59392 case/7602176 consumer dwords，模型不是完整native hazard证明。两版完整GPU验收全部通过、两种expert规模同一个actual artifact/完整ELF。v264 E64/E65 GEMM1为92.278/139.094 μs；v265为92.154/138.957 μs，同批v265首尾v190为90.632/90.804和137.858/136.968 μs，均不采纳。v266 A+B tied版CPU为463/96/LDS295936 B、private/spills0，按用户要求未GPU，不称为精度或性能通过。

| E65/v190 | GEMM1 μs | GEMM2 μs | MoE e2e μs |
|---|---:|---:|---:|
| 旧异常记录（不作稳定对照） | 161.590 | 315.866 | 982.26 |
| 重测 1 | 138.887 | 89.750 | 284.02 |
| 重测 2 | 136.918 | 89.441 | 281.17 |
| 重测 3 | 136.931 | 89.287 | 281.12 |
| 三次重测中值 | **136.931** | **89.441** | **281.17** |

按用户要求用相同原完整bench、fresh labels重测三次E65/v190，三次count19、完整ELF和原sanity保持，logits_diff均3.7793e-06/门限0.01。旧异常未复现，保留旧raw、排除稳定对照；不是修复生产kernel后的收益，不改正式E65中值137.596 μs。旧异常退出后有不完整KFD和随后独立任务，不能唯一确定原因。三次pre最后sample idle，立即post无KFD/busy3；后续post/post6的其他bench、dev_al/defunct记录均保留，六秒等待到idle，不把最后idle替代首post。本任务性能窗口无并发编译/验证/SFTP/归档/idle polling，未杀他人进程。

1048项/95480350 B本轮source/CPU/native/完整MoE/replay/210边界/所有control actual MLIR/异常三次复测及helpers已分五shard下载逐archive/member SHA、actual完整ELF在本地独立核验，无`.out`；[独立汇总](../.codex_tmp/gemm1_t1536_20261017/logs/runtime256_v266_completed_pause_downloaded_summary_20261020.json)与[OPTX详细复现](moe_gemm1_e64_t1536_optx_optimization.md)保留全部原记录。首版freeze误纳入写入中的包装日志，SHA拒绝后只在独立v2修工具，不改GPU结果。**按用户要求，手头v265与异常复测完成后已暂停全部优化**，v266 GPU、ATT/counters及原计划后续均未启动。以下独立`aiter_dsr1_e65`章节及其数字逐字保持。


## 补充：aiter_dsr1_e65 两pair回退、共用默认ATT与局部M32正反向结果（2026-10-19）

本节继续仅记录 `aiter_dsr1_e65` / a07-3 `aiter_t256w2x4_e64_20261007` / `dsr1_ep4` 工作链。当前正式默认仍为E65 GEMM1 **107.379 μs**、GEMM2 **94.3843 μs**、MoE e2e **256.893 μs**；E64分别 **95.3059 /60.4664 /209.078 μs**。两种规模正常入口共用compiled object及完整ELF `9a12287...`，双seed完整MoE/动态replay验收保持，E65距80 μs差27.379 μs。

上一节进行中的lookahead2+B1reuse1已经完整结束：zero spills、双规模共112项零字节差异，live额外56项通过；正式完整三轮中值E65 **119.000 μs**、E64 **107.379 μs**。分别比同批前后默认慢10.527%/10.209%与12.576%/11.299%，因此淘汰，默认不变。native增加predicate、`s_set_vgpr_msb`与DS waits，不能由静态计数唯一解释回退。raw包SHA256=`8816cb4e4a12c8b17a81654e239ca8ac44e02c1786e219d5208a11d269489080`，27轮所有raw/ownership及逐member验收完成；不使用被ownership污染的三组三轮。

当前共用默认已用用户原 `get_att_b_rocm_sdk.sh` 在a07-3完成SIMD0/1、各四SE的独立ATT，actual code_object_id22逐字节匹配新默认ELF。每SIMD60个sampled waves、24个无WMMA，896 WMMA组compute span中值为21856/21665 cycles；448组为20346/20743.5 cycles。padding WMMA减半后span未同比缩短。已通过source `add_tdm_loads`、actual kernarg pointer offsets40/48/56/96与native register chain把四owner明确绑定为A/B/ScaleA/ScaleB；steady tensorwait2每wave累计中值均72 cycles。工具的各wave单次中值再取中值为3 cycles，不冒充所有记录合并P50，也不把两个独立SIMD中值相加或换成kernelμs。raw ATT归档4604740 B/215 regular members，SHA256=`26544590df53017d6f39062beffebcdaea0014af6c78542a19aebf062c3daab3`，raw/decoded/native PC/ELF/source/ownership独立下载验收通过。此证据支持继续检查条件和LDS operand调度，不能唯一归因HBM/cache/FIFO。

局部M32条件候选在同一runtime count ABI中，将future A pair loads与当前pair WMMA放进同局部条件，用空LLVM asm的`=s,0` identity限制CSE。它不是memory fence、不改EXEC，不改有效routes、56次原序K128、scales/量化/输出或GEMM2。实际ELF `18d81298...`比默认少123条instructions，VGPR486/SGPR68/LDS282624 B、zero spills/private、180 B kernarg保持；CPU/native、seed0/17双规模112项零字节及live56项全部通过。

固定正反向原完整benchmark共36轮首次rc0、120秒内、ownership全部合格，没有按时间重试。只记录以下GEMM1/GEMM2与MoE e2e中值；完整每轮数据在主优化文档和raw包。

| workload | 固定顺序/路径 | GEMM1 μs | GEMM2 μs | MoE e2e μs |
|---|---|---:|---:|---:|
| E65/topk9/shared1 | 正向默认前控制 | 107.623 | 94.0428 | 258.944 |
| E64/topk8/shared0 | 正向默认前控制 | 95.7950 | 60.3968 | 208.835 |
| E65/topk9/shared1 | 正向局部M32候选 | 107.505 | 94.4519 | 258.044 |
| E64/topk8/shared0 | 正向局部M32候选 | 94.9938 | 60.2660 | 208.108 |
| E64/topk8/shared0 | 正向默认后控制 | 96.6426 | 61.0862 | 211.452 |
| E65/topk9/shared1 | 正向默认后控制 | 107.967 | 94.7641 | 257.970 |
| E65/topk9/shared1 | 反向局部M32前组 | 107.703 | 94.7176 | 258.733 |
| E64/topk8/shared0 | 反向局部M32前组 | 95.1752 | 59.9244 | 207.307 |
| E64/topk8/shared0 | 反向默认中间控制 | 96.2398 | 60.5612 | 208.918 |
| E65/topk9/shared1 | 反向默认中间控制 | 107.497 | 94.6988 | 258.617 |
| E65/topk9/shared1 | 反向局部M32后组 | 108.325 | 95.2364 | 259.271 |
| E64/topk8/shared0 | 反向局部M32后组 | 95.5675 | 61.5312 | 208.667 |

正向E65 GEMM1快0.11%/0.43%，反向慢0.19%/0.77%，收益没有复现，因此不采用。E64正向快0.84%/1.71%、反向快1.11%/0.70%，仍不能据此按E64另加default specialization。最终双端恢复精确共用默认。raw包2226344 B、313 regular members、SHA256=`4413f900b52b7d23f87c20ba44094963feb382cbd5b84f053e33d6b6c4c37373`，逐member/36轮原profile/精度/actual ELF/ownership/新owner provenance独立验收通过。打包初版漏传本地ATT工具/receipt，两次报`FileNotFoundError`；补齐后成功，两个工具失败log保留，没有重跑GPU测试或替换性能轮次。

共用ABI的K loop外all-live M32路径已经结束。当前wave的最后M32 pair原first-row predicate为真时，进入同ELF内没有重复pair predicate的pipeline；否则保留原条件，E64/E65仍同对象。CPU58240次K128/523264次WMMA的原序、operands和fences通过，actual ELF=`55dc9800...`、6298 instructions、VGPR458/SGPR78、zero spills/private、kernarg180 B及input/output fences通过。双seed双规模112项零字节及live56项精度通过，E64/E65共用同实际compiled object/完整ELF。

固定0 E65/E64→1 E65/E64→0 E64/E65、每组完整三轮的原benchmark共21轮rc0、120秒内。E64候选try1第3轮ownership不合格，整组排除，仅因此重跑完整try2；没有按耗时筛选。GEMM1/GEMM2/MoE e2e合格中值如下：

| workload | 固定组/路径 | GEMM1 μs | GEMM2 μs | MoE e2e μs |
|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前控制 | 110.265 | 95.1627 | 261.240 |
| E64/topk8/shared0 | 默认前控制 | 96.0753 | 59.3531 | 208.154 |
| E65/topk9/shared1 | all-live候选 | 111.364 | 95.0298 | 262.208 |
| E64/topk8/shared0 | all-live候选try2 | 97.9055 | 61.8917 | 213.269 |
| E64/topk8/shared0 | 默认后控制 | 95.0381 | 60.7216 | 209.179 |
| E65/topk9/shared1 | 默认后控制 | 107.353 | 94.7535 | 257.202 |

候选E65 GEMM1慢0.996690%/3.736272%，E64慢1.904964%/3.017106%，均相对两侧默认。因此淘汰，不做反向、不接入默认；减少VGPR没有成为实际收益，不唯一归因hardware FIFO/cache。正式默认E65 107.379 μs、E64 95.3059 μs保持。raw包1700476 B/213 regular members，SHA256=`066e21cea6db1bd5c041fb11a5a0295775066af7ea2ae8f4a5a0356276034d1c`；逐member、21轮原profile/ownership、112+56项精度、source/actual ELF独立下载验收通过。finally已恢复精确默认，post全机idle、KFD空，无本轮GPU残留。

runtime swizzle候选继续沿用共用ABI。这里修正早期说明：目标N为4096，但原launcher的`N: fx.Int32`是runtime参数。v1的13项长`and`导致FlyDSL AST重写在120秒内未完成，rc124；独立stage/stack诊断定位于`ast_rewriter.py:541–575 visit_BoolOp`，候选未dispatch。v2改`all((...))`后关闭控制成功，开启则因runtime `N == 4096`产生tracing Boolean而rc1。v3改短constexpr守卫与runtime N4096分支，但工具漏改f-string label版本，host runner拒绝已有label、没有新compile或dispatch；queue所读旧rc0不能作为新验收。remote CPU复核也曾因已有receipt而assert失败。上述失败保留，没有延长120秒或使用失败日志计时。

修订v4源码与v3相同：runtime确认N4096且为完整M group才走固定32/8除数，尾组及其他N保留原公式。CPU对N1024/2048/3072/4096/6144/8192、720项geometry共523296个WG映射点逐点验收，无重复遗漏；所有compute/operand/量化/地址/sync nested函数及GEMM2保持，关闭AST精确默认。wrapper SHA=`8faab89b...`、persistent SHA=`a6d4a0e6...`。两项no-dispatch编译均rc0、120秒内，E65/E64各mode同compiled object；关闭完整ELF精确`9a12287...`，候选完整ELF=`2d200a1b95699c265e26aa9c913d0ae0d9a28ec7dd142ef50ba5122d57d6b9f0`，4576 instructions、VGPR486/SGPR70、zero spills/private、LDS282624 B及runtime count ABI保持。原WMMA/DS/TDM/数值/barrier counts、所检查input/output fences和normal DEP_MODE保留。

双seed双规模完整MoE 112项零字节差异及live额外56项全部通过，eager/graph canary和同地址dynamic A/ScaleA/psum replay、原reference门限保持，实际两规模同compiled object/同candidate ELF。固定默认E65/E64→候选E65/E64→默认E64/E65各完整三轮对照已经结束，共21轮rc0/120秒内；E64后控制try1因第1轮ownership污染整组三轮排除，只因此重试try2。候选E65 GEMM1 109.319 μs，相对前/后默认107.788/107.851 μs慢1.420381%/1.361137%，MoE e2e也慢0.689807%/0.622460%；E64候选94.6735 μs，相对两侧95.0507/96.5523 μs较快，但不能据此另加E64 specialization。候选淘汰、不反向、不追加计时，双端恢复精确共用默认。raw归档2323354 B/306 regular members、SHA256=`7c72e73eff426d13f135f5d025f172bb2596db1820d7589faf652b5dc17f8936`，所有member、21轮原profile/ownership、精度/source/actual ELF独立下载验收通过。

本轮另完成共用ABI的operand reuse候选：只改变64个matrix-A hint bits，原112个matrix-B hints和全部ISA PC/顺序/operand/memory/fences保持。双seed双规模112项零字节和live56项通过，E64/E65同compiled object/完整ELF=`23a3bb556adbe5a3757d0b89541bf27902d93844f916fda499206f9eb18caad3`。完整24轮固定对照均rc0/120秒内，其中两组仅ownership污染整组排除；E65候选107.933 μs，相对两侧107.824/108.024 μs为+0.101091%/-0.084241%，E64候选95.9935 μs，相对两侧96.0059/95.0383 μs为-0.012916%/+1.005068%。无稳定双规模收益，淘汰、默认保持。raw1748834 B/219 regular members、SHA256=`f97a6cfaab9234da0774224061fb5d93b5d60fe07b03c72bbb345dad46707324`，独立下载逐member/24轮/精度/64hint bits验收通过。最初prepare/audit错误、后来raw checker误读JSON字段的失败记录均保留。

LDS b64 payload候选将各b128拆为两个aligned b64，CPU actual AST的20480 loader cases/786432B地址和dword顺序完全一致。两项no-dispatch compile均rc0/120秒内，共用E64/E65 object/cache1；关闭完整ELF默认，候选ELF=`cf30ce53c7cdef2af5966fc03cd59913a28e702d7ad8ed66947f029868f9c18d`，4941 instructions/VGPR486/SGPR68/LDS282624 B，但5 VGPR spills/private12 B、5 scratch stores/9 scratch loads，按资源门槛拒绝，未dispatch、未计时、未live部署。这是资源证据，不是实测性能回退。

下一项共用MMA group1仅改compiler分组2→1，原closing cover12和数值/memory/hardware fences保持。两端80个actual hint cases/2840 events及关闭完整AST通过；candidate完整ELF=`4accaa4d94e7b7110248b9672b8b99a78d35fb2d9cbd96e02a6a9fc05274f694`，4527 instructions，较默认少4个非零DScnt waits，仍VGPR486/SGPR68/LDS282624 B、zero spills/private、180 B runtime ABI。原WMMA/DS/TDM/数值/barrier counts和11个DScnt0保持；explicit physical VGPR的CFG DS RAW/WAW检查无未解决冲突，不能据此穷尽所有hardware hazards。

E65 seed0的eager/graph及dynamic replay已有24项payload/ScaleA零差异，但后续原完整MoE reference在`Cijk_Alik_Bljk_...` kernel发生`HSA_STATUS_ERROR_MEMORY_FAULT`，fault address=`0x6ba0000`，rc134、未超时。HEAD四项/最终MoE、E64和seed17未完成，故不能声称完整精度通过，也没有本项性能数据。队列停止，生产默认保持。外部stress/MLA owners在post存在，不构成唯一归因。独立固定默认/候选各一次的异步native/reference日志诊断随后按每个GPU程序120秒执行；不以诊断正结果覆盖failure或自动解锁旧性能队列。

固定异步诊断已经完成：mode0/default和mode1/candidate各seed0一次，各自E65→E64共56项零字节差异，原MoE reference门限、eager/graph canary及三组dynamic A/ScaleA/psum replay均通过；双规模每mode均同compiled object/cache1/完整ELF。每mode的258项`aten.mm.default`与258项native `hipblasLtMatmul`指针逐项对应、全部API返回status0、tensor shape/stride在storage界内。本次没有复现原memory fault，但CPU日志改变异步发射时序、API返回也不代表GPU实际执行完成，不能宣称原fault定位或修复。

mode0在完整GPU验证和receipt写完后，末尾日志`emit("diagnostic_finished", **record)`重复传`seed`，产生`TypeError`/runner rc1。仅修诊断工具为基础dict后`row.update(details)`，保留原raw/rc和脚本，离线复核全部完成结果与SHA通过，没有重跑默认GPU。原固定诊断停止后，仅执行尚未运行的candidate，rc0/120秒内；原candidate精度队列的rc134及性能关闭状态保留。本项只有seed0诊断通过，没有双seed完整验收、没有正式性能、没有live部署。

LDS b64拒绝和MMA group1原fault/诊断的raw共1432603 B、163 regular members，SHA256=`87a862f8effd0c68f9ba4a26685afd85cb9a111b64e3d331a90d82e5e1d0fcdb`，已完整下载并逐member/source/ELF/精度/native调用独立验收。初次归档checker误读queue JSON字段，修正读取`execution_failure`，失败receipt保留，GPU未重跑。

下一项独立A-only LDS b64候选保持当前共同runtime expert ABI，仅改变A payload loader，B/scales、有效计算/顺序、同步和GEMM2保持；20480 cases/786432 B的实际AST地址/dword顺序及关闭完整module相等通过。代码已逐项传到a07-3，compile/GPU验证尚未启动。用户报告机器hang并要求等重启通知后再使用，已停止连接尝试和远端工作，期间只整理本地证据/审计工具/文档。收到通知前不使用其他机器替代测试；重启后需重新核对环境与全机GPU空闲状态，历史boot/空闲检查不能沿用。

正式默认E65 107.379 μs、E64 95.3059 μs仍保持，距80 μs差27.379 μs。各完整三轮原数据、复现和证据限定记录在 [aiter_dsr1_e65 优化文档](../../aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md)；本节仅更新本工作链，不覆盖其他OPTX接入记录。

### E65 GEMM1：重启恢复、A-only资源拒绝及当前进展（2026-10-19）

收到用户“机器已重启”通知后恢复a07-3/dsr1_ep4，boot=`1d5146ac-1863-43d0-8211-f0761013cc4a`，指定仓库HEAD/分支、原容器image及live四文件SHA已复核。此前上传的8项任务文件在remote为0 B，保存证据后只重传所需文件并核对SHA；不能据此确定hang原因。实际launcher AST导出的2080 B A/B stride、1040 B K128 stride、70656 B stage pitch、282624 B arena已核对；20480 loader cases/786432 B地址和返回顺序相同，E64/E65 geometry一致。修正JSON tuple/list、AST格式及旧argv残留等工具问题，原失败records保留，未因此重跑GPU。

A-only mode0/1实际no-dispatch compile均rc0、120秒内，各自E65→E64→E65同compiled object/cache1。mode0完整ELF精确默认；mode1完整ELF=`d40a0a99a4f18d40f8022b91ad6f471f37ff557352468b50816835319957f2a7`，4747 instructions、VGPR487/SGPR68、LDS282624 B、kernarg180 B，但1 VGPR spill/private8 B。candidate五件套下载后已逐SHA、ISA字节和binary metadata交叉验收，资源拒绝证据成立。该候选停止在编译阶段，未GPU正确性、未计时、未部署live；不能把资源变化写成性能收益。

当前正式最优仍E65 GEMM1/GEMM2/MoE e2e=`107.379/94.3843/256.893 μs`，E64=`95.3059/60.4664/209.078 μs`。两规模双seed完整MoE精度及正常入口同compiled GEMM1/完整ELF要求已通过，距80 μs还差27.379 μs，本轮没有新增性能纪录。

已对默认SIMD0/1共120个sampled waves中的72个compute waves完成physical VGPR operand provenance，所有WMMA的A/B/ScaleA/ScaleB均有LDS来源，MOV传播包含`v_mov_b64_e32`及`v_dual_mov_b32`。native matrix A对应logical B，matrix B对应logical A，不能按指令名直接反置。896-WMMA组最近DS issue到WMMA issue的每wave中值再中值：SIMD0 A414/B581 cycles，SIMD1 A408/B569.5 cycles；这不是DS完成延迟或kernel性能。长issue gap横跨条件、refill及WG/TDM区间，现有证据不能唯一归因某FIFO/cache。继续保留b128、有序有效计算和全部hardware fences，仅研究compiler hint调度，实际native与完整精度通过后才计时。

本次进展回复后继续创建独立carry DS-first候选：`AITER_GFX1250_GEMM1_DS_FIRST_CARRY=0/1`仅调整第二个K128的future DS/MFMA compiler hint顺序，保留b128、MMA group2、cover12、全部有效计算/数值/同步、GEMM2和共用runtime expert ABI。本地与a07-3 host的80组/1960实际hint事件审计通过，20组carry顺序改变、总数相同、关闭完整AST精确默认。独立snapshot传输SHA已核对，a07-3/dsr1_ep4固定两项no-dispatch编译已启动，每项120秒；尚无本项精度或性能结论，正式默认仍E65 107.379 μs、E64 95.3059 μs。

carry DS-first两项编译已rc0、120秒内，zero spills/private及canonical runtime ABI保持，但candidate完整ELF仅因名称等非指令内容不同。下载后4531条native instruction、physical operands及整段`.text`字节完全等于默认，SHA256均`47604f4f0e98953a88fa1f771960a8a624badb2667d7a1f9688a9cb26f8b8e48`。初native checker把不同完整ELF错误标为实际调度变化；现修正为必须有executable `.text`变化，旧receipt保留并由`unified_ds_first_noop_downloaded_rejection_boot1d5146ac_v2.json`取代。该候选停止、未追加GPU测试或性能数据；继续下一项实际调度实验。

继续准备完整B预读延后一个M32 pair的独立候选，两个B fragments在当前pair0计算后再预读，保留独立双套A/B/scales、b128及全部有效计算/精度/hardware fences。它与历史halfB、共享B1及lookahead2/B1reuse不同。本地1040 pipeline cases、58240次K128 steps、523264次WMMA actual AST审计通过：当前operand/原56次有序更新、load总量和同步/hint事件保持，关闭路径精确默认。此时仅完成本地源码审计与任务文件传输，下一步a07-3/dsr1_ep4 actual compile；仍无新性能纪录。

完整B延后候选actual编译已通过，两mode rc0/120秒内、E65/E64同对象/cache1。candidate实际`.text`变化，完整ELF=`ab190ab0ef7eeaf7c535ff6875f0925f8f3f451a0dfab53c02dfdfb28910b58c`，4538 instructions/VGPR486/SGPR70/zero spills/private；LDS282624 B、180 B canonical runtime ABI、原WMMA/DS/TDM/数值counts和11条DScnt0/全部hardware fences保持。显式DS CFG无未解决RAW/WAW冲突，下载后binary/ISA/metadata再次验收通过。继续固定双seed双规模完整MoE验证；目前只有native资格，未精度或正式性能通过。

完整B延后候选的seed0/17、E65/E64完整GPU验证现已结束，两个进程rc0/120秒内。112项有效payload、ScaleA、GEMM2和MoE逐字节比较全部零差异，四次原reference MoE均pass=True，eager/graph canary及动态A/ScaleA/psum replay保持相同指针并通过，覆盖空expert。E65 seed0/17 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`2.7493e-03/2.7491e-03`，E64为`3.4733e-06/3.4766e-06`、`2.6357e-03/2.6369e-03`。两规模cache1、同compiled object及完整ELF=`ab190ab0...`；raw/result/manifest/binding下载后独立SHA和112项验收通过，receipt=`unified_late_b_correctness_downloaded_acceptance_boot1d5146ac_v1.json`。

这些为snapshot imports的correctness-only运行，期间有其他GPU用户，耗时全部无效；结束未发现本任务GPU残留，live生产默认保持。接下来先做候选正常live imports验收，再执行默认E65/E64→候选E65/E64→默认E64/E65的完整三轮性能对照，空闲前每6秒自动等待，执行/精度/解析失败停止，仅ownership污染整组最多三次。正式已验收性能仍E65 GEMM1 **107.379 μs**、E64 **95.3059 μs**，E65距80 μs差 **27.379 μs**，本项尚无新性能结论。

该固定性能队列已在候选部署前停止：默认E65三组的前两组ownership污染、第三组合格，GEMM1三轮107.664/109.056/112.453 μs，中值109.056 μs，GEMM2/e2e中值94.3516/259.981 μs；随后默认E64三组三轮全部有ownership污染，达到预定上限后停止。18轮rc0/120秒内、原MoE精度均True；五个污染整组排除，不拼轮、不追加第四次。候选正常live binding和正式计时均未运行，不能称提速或回退。限域raw包1428982 B/172 regular members、SHA256=`e30d3ca272d3ba195f25410ec2c356e76233a6c2013ecd50848f2df4fe97b7a3`，已下载并逐member/native/112项精度/18轮profile/ownership独立验收；正式默认性能与live源码保持。

继续准备独立B-only LDS b64候选，仅拆B payload的b128为两个aligned b64，保留原A/scales、双slot、全部有效计算/原K128顺序、hardware fences、canonical runtime ABI与GEMM2。它区别于已因spill拒绝的A-only/A+B方案，不重启旧队列。实际AST的20480 loader cases/786432 B地址与dword顺序审计通过，关闭完整module等于共用默认；下一步仅在a07-3/dsr1_ep4固定两项actual compile，先检查zero spills/private和真实`.text`变化，随后才决定是否做完整精度。当前正式E65 107.379 μs、E64 95.3059 μs保持，距80 μs差27.379 μs。

B-only LDS b64的两mode actual compile现已rc0/120秒内结束、未dispatch，E65→E64→E65每mode同compiled object/cache1。关闭完整ELF精确默认；candidate完整ELF=`f1c729ebc8d7f824dfef8c9ddc19cb571d0f80d78dbe28d2516c1b67767bfcb1`，4698 instructions、VGPR487/SGPR68、zero spills/private、LDS282624 B和180 B runtime-count ABI保持。原numerical/scalar DS/store/TDM counts、11条DScnt0及hardware fences保持，static payload总量均1536 dwords。下载两mode的native五件套和原编译日志/result/manifest后，已逐SHA、ISA编码、binary metadata与launcher嵌入ELF独立验收，`.text`确有变化，显式DS CFG无未解决RAW/WAW。receipt=`unified_b_lds_b64_native_downloaded_acceptance_boot1d5146ac_v1.json`。这只达到GPU验证资格，不能推断性能收益；继续固定双seed双规模完整MoE精度，通过后才测性能，正式默认107.379/95.3059 μs保持。

B-only LDS b64的seed0/17、每进程E65→E64完整GPU精度现已通过，两进程rc0/120秒内，112项有效payload/ScaleA/GEMM2/MoE逐字节比较全部零差异。四次原MoE reference保持原门限并pass=True；E65 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`2.7493e-03/2.7491e-03`，E64为`3.4733e-06/3.4766e-06`、`2.6357e-03/2.6369e-03`。eager/graph canary及同地址dynamic A/ScaleA/psum replay、空expert覆盖通过，两规模cache1、同compiled object和完整ELF=`f1c729eb...`。raw/result/manifest/binding下载后已独立SHA和112项验收，receipt=`unified_b_lds_b64_correctness_downloaded_acceptance_boot1d5146ac_v1.json`。correctness-only期间有外部GPU users，耗时不记录为性能；本任务无GPU残留、live默认保持。继续固定完整三轮性能，候选计时前验收正常live imports，不重启旧late-B队列，正式最优仍107.379/95.3059 μs。

### B-only LDS b64性能控制停止及证据收尾（2026-10-19）

B-only的双seed双规模112项精度通过后，固定性能队列停在第一个默认E65前控制。三组三轮全部rc0、120秒内且原random MoE精度True，但三组均有ownership污染；GEMM1原始三轮分别106.687/108.568/106.738、107.250/107.655/108.344、111.197/107.619/107.845 μs，全部整组排除，不计算正式中值。事前三次上限已到，队列关闭，不追加第四组、不拼轮。candidate正常live部署、live binding及正式计时均未执行，不能说它提速或回退，live源码精确默认保持。

限域raw归档`unified_b_lds_b64_stopped_raw_evidence_boot1d5146ac_v1.tar.gz`为1120512 B、136 regular members，SHA256=`b57c35a28e0f25a685e10ace55eb045ea55c35e38f439e880eb12178c4650fcc`。下载后逐member SHA、两mode native/完整ELF/.text/metadata/launcher、112项精度和九轮profile/ownership独立验收通过，receipt=`unified_b_lds_b64_stopped_raw_downloaded_acceptance_boot1d5146ac_v1.json`。没有新正式性能结果；当前共用默认E65 107.379 μs、E64 95.3059 μs，E65距80 μs差27.379 μs。下一项继续检查K-loop重复row predicate及operand调度；ATT中跨这些指令的间隔不能唯一归因某个hardware单元。

### M32 runtime mask的实际源码审计（2026-10-19）

下一独立候选把真实`wmb + pair*32 < mn_oob`四个first-row条件编码进单个runtime mask，在原use点用局部SGPR identity取bit，尝试减少K-loop条件转换。它保持部分有效pair的全部计算、双slot、有效accumulator的56次原序K128更新、canonical ABI及GEMM2，同一个compiled GEMM1服务E64/E65，没有按seed或expert数新增specialization。本地及a07-3 host的1040 pipeline cases/58240 K128 steps/523264 WMMA actual AST审计通过，operand、load、WMMA及TDM/fence/hint完整事件顺序精确默认，关闭完整module AST还原默认。source wrapper/persistent SHA=`2d0ce3c34757ac03034d50bbb5225834de57d0bab33cca36fd6255e237382069`/`b3f771f19046bd45e6a4d89c7eee5bb206743ec12c62cf380d511c2572039f34`。

只上传新snapshot与必要tools，live默认未改。a07-3/dsr1_ep4已启动固定0→1 no-dispatch编译，每项120秒；实际`.text`变化、减少目标条件转换、zero spills/private、runtime-count ABI和native fences通过后才考虑GPU完整精度。ISA §15.4的scalar bitcmp/compare语义不保证该LLVM表达式会生成预期opcode，更不构成性能结论。当前正式E65/E64 GEMM1仍107.379/95.3059 μs。

M32 mask v1编译现已结束，两项rc0/120秒内、E65/E64每mode共用同对象/cache1，关闭完整ELF精确默认。候选ELF=`2471c8e586125cb44f2769af36e64e2049d9d7fea73a3cb0fd769c888a874c31`，4685 instructions/VGPR486/SGPR70，出现6个VGPR spills和16 B private，被native gate拒绝，未dispatch或做GPU精度/性能。两mode native下载后独立ISA字节、binary metadata、summary/launcher和全部SHA通过，receipt=`unified_m32_mask_native_downloaded_acceptance_boot1d5146ac_v1.json`。其条件转换减少，但多60条static branch及6对scratch load/store，不能记作提速。

继续v2，把同pair的future A load和current WMMA放入一次局部mask条件，保留双slot、原K128次序、head B/scales及BACK ScaleA时机。1040 pipeline cases/58240 K128 steps/523264 WMMA CPU实际AST审计通过，有效operand/updates、load多重集和hardware事件保持；关闭AST还原默认。persistent SHA=`78013d108d82a1b519f962ce5053fa29da3d0b4b82bac8d207b0085d7b4e02cd`，wrapper保持v1，先独立no-dispatch编译，live默认不改。

v2 actual compile与独立native下载验收已通过：两项rc0/120秒内，各mode E65/E64同compiled object/cache1。候选ELF=`b46ef980bb5e30b553d3b020721f76414b6d0126b9bc8b1f38ec9c951d030f13`，4420 instructions，VGPR486/SGPR70、zero spills/private、LDS282624 B/kernarg180 B，真实count仍offset176/size4。ISA/ELF `.text`、binary metadata、notes/summary、launcher和全部SHA重新独立核验；physical DS有限CFG无RAW/WAW未解决冲突，原branch数、numerical/DS/TDM和hardware fences保持。比默认少111条static指令，目标三类转换共少186个sites，增加68条bitcmp和67条mov，没有v1的额外branch/spill。receipt=`unified_m32_mask_native_downloaded_acceptance_boot1d5146ac_v2.json`。接下来固定双seed双规模完整GPU精度，不能从静态指令减少宣称提速，live和正式107.379/95.3059 μs保持。

#### M32 mask v2完整GPU精度复核及当前进展（2026-10-20）

双seed0/17、每进程E65→E64的a07-3/dsr1_ep4运行现已完成；下载原始run/result/manifest/idle和actual binding后独立验收通过。共112项有效payload/ScaleA/GEMM2/MoE逐字节差异为0，eager/graph canary、同地址动态A/ScaleA/psum replay通过，四次原MoE reference通过。E65 seed0/17的logits_diff为3.7793e-06/3.7788e-06，rel_l2为0.0027493/0.0027491；E64为3.4733e-06/3.4766e-06，rel_l2为0.0026357/0.0026369。每个进程两种规模共用同compiled GEMM1、cache1和完整ELF=`b46ef980bb5e30b553d3b020721f76414b6d0126b9bc8b1f38ec9c951d030f13`，源码SHA与native验收一致，均rc0、120秒内，无本项GPU进程残留。

独立receipt=`unified_m32_mask_correctness_downloaded_acceptance_boot1d5146ac_v2.json`，SHA256=`81c0ce418f1ebe0d1cf1d3c931b6c60d225a1e15dcad519613d56ed9d7a2836b`。这是correctness-only snapshot验证，存在外部GPU owner时得到的计时全部无效，不能作为性能收益。下一步先等待全机空闲，固定默认E65→默认E64→候选E65→候选E64→默认E64→默认E65，每组完整三轮，ownership污染整组最多三次；candidate正常live部署后另做双规模binding，结束恢复默认。当前共用正式默认E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs差27.379 μs。

v2性能队列现已关闭：默认E65前控制三次完整三轮组分别第1/3/1轮出现外部live GPU owner，九轮均rc0、120秒内、原MoE精度True，但没有完整合格性能组。GEMM1三组原始读数108.505/108.296/108.034、109.886/109.266/107.191、107.978/111.549/107.956 μs，全部整组排除，不能按最快轮或拼轮计算中值。candidate部署/live binding/正式计时未执行；不追加第四组、不重启队列，不能说candidate快或慢。finally精确默认SHA保持，post当时KFD空/busy0/VRAM176582656 B，全机idle、无本项GPU进程残留。

继续新的v3：在v2同pair局部scope内，以`s_bitcmp1_b32`/`={scc},s`直接返回真实四bit mask的选定位，尝试去掉native中的每次mask拷贝；mask只读、不是hardware fence。ISA §15.4 p269支持bitcmp1的SCC语义，仍需actual native降低mask-copy且zero spills/private才有GPU验证资格。CPU actual AST1040 pipeline cases/58240 K128 steps/523264 WMMA通过，有效56次原序K128更新、operand/load多重集/TDM/fences、canonical ABI/GEMM2保持，关闭AST精确默认。source wrapper/persistent SHA=`2d0ce3c34757ac03034d50bbb5225834de57d0bab33cca36fd6255e237382069`/`cc1ff29bb0a4ad201aeee98c346c5bcf65758313aca1849ccf7c03c38e9ee805`，独立snapshot不覆盖live默认；此为编译前记录，实际结果如下。

#### M32 mask原始证据收尾与v3实际编译结果（2026-10-20）

v1/v2限域raw包为1569138 B、177 regular members，SHA256=`3337beaa73d9938dea1d98c4eff63d6226ea757d6a2450b3f4d1c74f840e7e9f`；下载后逐member SHA、两版native/ELF/.text/metadata/launcher、112项精度、九轮profile/ownership独立验收通过，receipt=`unified_m32_mask_stopped_raw_downloaded_acceptance_boot1d5146ac_v2.json`。首次CPU checker对`pass=True`在日志中的位置假设错误，已按原格式修正并复核，失败准备记录保留；没有修改raw包或GPU重跑。三组性能全ownership污染、零完整合格组，v2没有候选正式计时，关闭结论保持。

v3 actual AST host审计和dsr1_ep4两项no-dispatch编译均完成，rc0/120秒内、每mode E65→E64→E65同对象/cache1/完整ELF，关闭ELF精确默认。候选ELF=`6a2406da0fb91559b6cec12a247e42f611bd4b34a95851c128cebffa55b3b577`，4557 instructions（v2 4420、默认4531）、VGPR486/SGPR70、zero spills/private、LDS282624 B/kernarg180 B/count offset176/size4保持。虽然68条显式bitcmp1保留且mask-copy减少，LLVM对SCC返回值重建转换：相对默认`s_and_b32 -61/s_cselect_b32 +7/s_cmp_lg_u32 +4/s_delay_alu +9`，目标转换proxy385→335，而v2为199。因此native gate以`The candidate did not reduce the v2 executable instruction count`淘汰v3，未dispatch、无该版本GPU精度或性能。两mode五件套、原logs/results/manifests/idle下载后独立验收通过，receipt=`unified_m32_mask_native_downloaded_acceptance_boot1d5146ac_v3.json`明确候选不合格。

下一步尝试按WMMA snake次序创建accumulator rmem，再恢复原logical索引，保持全部有效计算、K128累加次序、输出、fences、GEMM2和共用ABI；CPU actual allocation/mapping及native目标变化合格后才做GPU验证。Shader Programming Guide §3.3.2.3、§5.3.3、§5.7.1.2支持VGPR MSBs与control co-issue语义，static减少`s_set_vgpr_msb`并不自动等于减少cycles。此为该候选实施前记录，实际源码审计如下；当前共用正式E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs差27.379 μs。

#### Accumulator storage snake v1实际源码审计（2026-10-20）

新独立snapshot默认关闭`AITER_GFX1250_GEMM1_ACC_STORAGE_SNAKE=0`，仅改GEMM1的rmem创建和logical映射，t256 logical→storage为`[0,1,3,2,4,5,7,6,8,9,11,10,12,13,15,14]`；全部原compute/zeroing/epilogue/load/fences AST、56次原序K128更新、GEMM2及canonical runtime-count ABI保持，没有按E64/E65新增specialization。64种actual allocation cases验证对象互异、双射、零初始化和logical身份；实际allocated objects注入原pipeline，1040 cases/58240 K128 steps/523264 WMMA完整event顺序精确默认。关闭完整module AST还原，receipt=`unified_acc_storage_snake_source_audit_v1.json`。CPU工具的parent范围和AST缩进准备错误已修正并留诊断，候选源码未因此改动、GPU未重跑。

source wrapper/persistent SHA=`018afae205204fceb4f9484eaf1a4577ac21e6235496669ebfa559dbe6bf14ea`/`95c650d6fb39ccee87bf34dbcc62b300e3600c910205a08bb22c2563188eff26`，只上传必要snapshot/tools并重核a07-3 HEAD/branch/boot/image/live及candidate SHA。接下来固定0→1 no-dispatch编译、每项120秒，必须actual `.text`变化、减少`s_set_vgpr_msb`且不增加总指令数、zero spills/private、共用compiled ELF/count ABI和native numeric/DS/TDM/fences保持才dispatch验证。没有新性能结果，live生产默认保持。

storage snake v1已实际编译并淘汰：两项rc0/120秒内，各mode E64/E65同对象/cache1/完整ELF；关闭完整ELF精确默认。候选ELF=`e3584e0960b6f5b4a44a532311a06fd64e9f199cbe64b92267be2d84351e4d03`，4531→4549 instructions、`s_set_vgpr_msb`375→391，VGPR486/SGPR68/LDS282624 B/zero spills/private/count ABI保持。目标指令数和总指令数均增加，native gate拒绝，未dispatch、无新精度或性能。两mode五件套和原logs/results/manifests/idle下载后独立验收通过，receipt=`unified_acc_storage_snake_native_downloaded_acceptance_boot1d5146ac_v1.json`，post无本项GPU残留。VOPD配对delta不能单独证明math改变。

后续native布局分析确认两版256 accumulator dwords均为低bank240、高bank16；创建次序没有消除高bank accumulator。16个新增MSB sites都在compute静态区，四组28/32-WMMA backedge的MSB计数28/32→29/34。240条native `v_mov_b64_e32`来自两个互斥零初始化路径，各120条，不能算成每次K-loop operand复制或直接换算μs。分析=`unified_acc_storage_native_analysis_v1.json`。接下来从精确默认尝试延后零初始化到首次input LDS prime后，验证所有有效accumulator仍在首WMMA前为原+0、完整56次更新保持，padding expert无效零初始化可省略；先实际CPU路径及native，不提前声称收益。共用正式E65/E64 GEMM1仍107.379/95.3059 μs。

#### Accumulator late zero v1源码及CPU验证（2026-10-20）

新独立snapshot仅将原GEMM1 accumulator的+0初始化从`run_task`早期移到首次input LDS prime之后，load/TDM/hardware fences、有效routes、56次原序K128更新、compute/epilogue/GEMM2及canonical runtime-count ABI保持；默认关闭`AITER_GFX1250_GEMM1_ACC_ZERO_LATE=0`。E64/E65继续共用selector0/runtime expert数，不新增specialization；开启时padding expert不prime/compute/zero。

本地和a07-3 host已执行actual AST审计：192个prime/zero边界cases确认每个有效accumulator在首次prime后、首WMMA前恰好得到一次原+0；实际对象注入原pipeline的1040 cases/58240 K128 steps/523264 WMMA保持原operand、scale/reuse selector及完整event顺序。45个nested函数、GEMM2和关闭完整module AST保持，receipt=`unified_acc_zero_late_source_audit_v1.json`。source wrapper/persistent SHA=`0d9b366d9d8532091248c6ab3780c275fbfcc4b8c9273dd36e826cbea2fde504`/`9de283ce717ae2c3c8e814191f2ba5f193519d5da2eac36cb3775669debfdf3b`，限域上传并核对SHA，live默认未改。

继续a07-3/dsr1_ep4固定0→1 no-dispatch编译、每项120秒；只有开启`.text`实际变化、zero-copy proxy减少且总指令不增加、zero spills/private、共用ELF/ABI、原numeric/DS/TDM counts及physical DS CFG/hardware fences保持后才dispatch做完整GPU精度。CPU/static结果不作为μs收益。目前正式E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs差27.379 μs。

late zero v1实际编译已结束，两mode rc0/120秒内、各E64/E65同对象/cache1。候选ELF=`d9d63caf00c3edbcbf6a4b3024eb40d8b17fb46f3fb0f8bf2a52d7fcca42f1c9`，但executable `.text`与默认逐字节相同、SHA均=`47604f4f0e98953a88fa1f771960a8a624badb2667d7a1f9688a9cb26f8b8e48`，4531 instructions、240个zero-copy proxy、375个MSB sites及全部resources保持。因此native gate拒绝，不重复跑相同executable的性能。原五件套/logs/results/manifests/idle下载后独立验收通过，receipt=`unified_acc_zero_late_native_downloaded_acceptance_boot1d5146ac_v1.json`；CPU checker字段名准备错误已修正留证，没有重跑编译/GPU、post无本项owner。

继续检查首次K128的原+0 C输入能否直接用inline zero，去掉有效accumulator的初始化复制；保留全部56次原序更新。Shader Guide p146与CDNA5 ISA §7.12 p96的VOP3P fields允许WMMA SRC2为even VGPR或inline constant zero，ISA §15.11 pp465–466仍要求原D=A*B+C。下一新snapshot先审计首tile/rolled事件、有效accumulator和输出边界，再native/GPU完整精度；此时没有新性能结果，107.379/95.3059 μs保持。

#### First-WMMA +0 v1源码及CPU审计（2026-10-20）

新snapshot只在已验收共用scope将原tile0执行一次移到owner rolled loop前，`steady_start=1`；tile0首K128使用原+0 C，其后原56次更新/次序保持。未计算M32 pairs继续初始化+0，避免epilogue读取undef；原`issue()`按实际wave选择job，load/TDM/fences、routing/canonical ABI/GEMM2保持，不按E64/E65新增specialization。默认关闭`AITER_GFX1250_GEMM1_ACC_FIRST_ZERO=0`。

本地actual AST的4160 pipeline cases/232960 K128 steps/2093056 WMMA及576 expert边界cases通过；检查全部八owner branches、完整28tile、第一次C逐位+0、后续C history及有效最终值、原operand/scale/reuse selectors、loads顺序/多重集和全部hardware/compiler hint事件。关闭完整module AST精确默认，41个nested函数和GEMM2保持，receipt=`unified_acc_first_zero_source_audit_v1.json`。source wrapper/persistent SHA=`c5461d3ee15adc42937c4c7c0a93868e11f63b794b3ce3f751443d2eb6fa0763`/`a463fe4013465c4c8261654cc7db44e6be1df675be78ddf7afc56b2d0933ae1d`。继续a07-3 host audit和dsr1_ep4固定no-dispatch编译，每项120秒；actual inline-zero及zero-copy/resource/fence门槛合格后才GPU完整精度。目前正式性能107.379/95.3059 μs保持。

#### First-WMMA +0编译结果与native验收脚本问题（2026-10-20）

a07-3 host CPU复核通过；dsr1_ep4两mode no-dispatch编译均rc0/120秒内、E64/E65同对象/cache1/完整ELF。关闭完整ELF精确默认；候选ELF=`e13389f41529c8cdba9a18215e418fd58983a779a09b36a80e5f5655a28fcc14`，4531→4718 instructions、VGPR486→492/SGPR68→74、zero spills/private、LDS282624 B/count ABI保持，16个inline-zero C sites已出现、zero-copy proxy240→96。尚无GPU精度或性能结论。

原v1 gate拒绝opcode/fence计数并关闭rc1；原五件套和logs/results/manifests/idle下载后独立验收保留拒绝receipt，不重启编译。后续拆开VOPD half的operation统计确认epilogue数值数量保持。首tile展开精确增加32 static WMMA、816 B/lane static LDS reads和4个TDM sites，4个actual owner分别绑定原A/B/ScaleA/ScaleB kernarg pointer及wave predicate。source完整动态事件、56次原序K128有效更新保持；static sites增加不能当作多计算。

原gate遗漏4个展开TDM，且将`s_barrier_wait -1`与native `0xffff`误判不同。除一组展开carry fence外，增加三条`tensorwait10`；保守native CFG不假定异步完成或predicate事实，其入口最大outstanding为3/4/5，均不可能等待，原input/output completion fences及显式physical DS RAW/WAW保持。硬件依据CDNA5 ISA §5.7.1/§15.6的`TENSORCNT <= SIMM16[5:0]`和Shader Guide §4.10.1同wave TDM计数/in-order，不能据此推导μs收益。

新独立`unified_acc_first_zero_native_qualification_boot1d5146ac_v2.json`绑定同一ELF、source、原拒绝SHA及展开分析后通过，旧拒绝receipt和队列仍保留，未修改候选或放宽资源/精度标准。CPU checker的Windows参数引用及manifest字段准备错误留diagnostics，没有重跑编译/GPU。下一步双seed0/17、每进程E65→E64的完整MoE精度、逐字节有效payload/ScaleA/GEMM2/MoE、eager/graph/dynamic replay；此时live默认及正式107.379/95.3059 μs保持。

#### First-WMMA +0双seed双规模GPU精度验收与当前进展（2026-10-20）

固定seed0→17、每进程E65→E64在a07-3/dsr1_ep4完成，两程序rc0、120秒内。原始run/result/manifest/idle、source snapshots、binding及queue/control logs已下载，`check_unified_acc_first_zero_correctness_download_v2.py`独立核验通过：112项有效payload/ScaleA/GEMM2/MoE逐字节比较全部零差异，eager/graph canary和同地址dynamic A/ScaleA/psum replay通过，四次原MoE reference精度检查通过，标准保持。

| workload | seed | logits_diff | rel_l2 |
|---|---:|---:|---:|
| E65/topk9/shared1 | 0 | 3.7793e-06 | 0.0027493 |
| E64/topk8/shared0 | 0 | 3.4733e-06 | 0.0026357 |
| E65/topk9/shared1 | 17 | 3.7788e-06 | 0.0027491 |
| E64/topk8/shared0 | 17 | 3.4766e-06 | 0.0026369 |

每进程双规模cache1/同compiled object/完整ELF=`e13389f41529c8cdba9a18215e418fd58983a779a09b36a80e5f5655a28fcc14`，与native qualification一致。receipt=`unified_acc_first_zero_correctness_downloaded_acceptance_boot1d5146ac_v2.json`，SHA256=`95ff662ac09cb2ea38cde8fda926ab6ba783c3a099f29b02f49cc38a3ac68f3b`。本项correctness-only snapshot验证全部计时无效，post无自身GPU残留，live默认未改。

继续固定默认E65→默认E64→候选E65→候选E64→默认E64→默认E65完整三轮性能对照，candidate正常live imports仍先单独绑定，只有全机idle时计时。ownership污染仅整组最多三次尝试，其他失败立即停止，finally恢复默认。此时没有本候选正式性能数据；正式E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs目标差27.379 μs。

#### First-WMMA +0性能对照完成：未提速，撤回候选（2026-10-20）

24轮原完整random non-balanced benchmark均rc0、120秒内、精度True。额外正常live imports先完成双规模56项零字节及同compiled object/cache1/完整ELF验收。E65 candidate try1第3轮与E64后控制try1第2轮ownership污染，按事前规则各整组三轮排除并重试try2，未拼轮或按耗时重试。

| workload | 完整合格组 | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前控制 | 109.466 | 94.8674 | 259.456 |
| E65/topk9/shared1 | candidate | 109.930 | 94.2476 | 259.617 |
| E65/topk9/shared1 | 默认后控制 | 108.152 | 94.9113 | 258.123 |
| E64/topk8/shared0 | 默认前控制 | 95.1226 | 61.0588 | 206.933 |
| E64/topk8/shared0 | candidate | 95.6452 | 61.0335 | 207.749 |
| E64/topk8/shared0 | 默认后控制 | 96.1554 | 61.4085 | 208.681 |

E65 GEMM1比两侧默认慢0.423876%/1.643983%，MoE e2e慢0.062053%/0.578794%；E64 GEMM1相对两侧为+0.549396%/-0.530599%，没有稳定收益。因此候选淘汰，不接入默认或追加按耗时筛选。原raw与全部排除组保留，完整三轮及SHA见优化文档。

下载完整性能run/result/manifest/idle/source snapshots、正常live binding、plan/summary与部署/恢复receipts后，`check_unified_acc_first_zero_perf_download_v2.py`独立重算24轮、六个合格组及两组排除，通过。receipt=`unified_acc_first_zero_perf_downloaded_acceptance_boot1d5146ac_v2.json`，SHA256=`1790d90c4a46858fee5e591610f2b0fdd217c659095268591c70ab345eceb29a`。finally恢复精确默认，2026-10-19T18:09:15Z post全机idle/KFD空/busy0/VRAM176582656 B，无本项GPU残留；后续性能前重查。当前正式E65/E64 GEMM1仍107.379/95.3059 μs，E65距80 μs差27.379 μs。

#### 继续优化：Scale LDS base tied identity源码候选（2026-10-20）

已同步first-WMMA +0的全部精度/正式性能记录和独立receipt；只读核对远端文档后保留原side/OPTX内容，两文档SHA一致。正式默认未刷新，E65/E64 GEMM1仍107.379/95.3059 μs。

下一新snapshot只给ScaleA/ScaleB的lane-specific LDS base加空LLVM tied identity `=v,0`，低32位地址保持原DS consumer语义，原普通LDS load仍由LLVM处理依赖和waitcnt。没有删除hardware fence、修改计算/slots/route/output，也不新增E64/E65 expert-count specialization。现有ISA的steady scale多数已合并，prime/drain有独立b32及重复地址生成；本项只测试是否能改善后者，不能直接推定性能收益。CDNA5 ISA §15.15 pp721–722记载stride64 offset×256 B；Shader Guide §5.7.6没有给该kernel的确定加速保证。

actual AST审计172032项scale地址/顺序、688128 B及9216项modulo32 consumer cases通过；44个nested函数和完整56次原序K128、numeric/pipeline/TDM/hardware fences/GEMM2保持，关闭module精确默认。source=`unified_scale_base_tied_v1_source/`，receipt=`unified_scale_base_tied_source_audit_v1.json`；默认开关`AITER_GFX1250_GEMM1_SCALE_BASE_TIED=0`，wrapper/persistent SHA=`f82f9c368249f75e645ccfe6f3443b620c9be1b81e6422a8636aab12073e7c95`/`a00238736fa197ee821315d96633cf6ac5c485be773ee411130f8f47219f1098`。继续仅a07-3/dsr1_ep4 no-dispatch compile，必须actual `.text`/DS issue改善、总指令不增长、zero spills/private、共用完整ELF/count ABI及physical DS CFG/hardware fences保持，才进入完整GPU精度；此时尚无新性能。

后续实际编译和本地独立验收已通过：mode0→1两项rc0、120秒内，各mode双规模同compiled object/cache1/完整ELF，关闭精确默认；candidate ELF=`db4ebc4eeae3df6cbc9a1aaed9f654f779a39ca9efdf4175b20fa2db4e92a8ee`，4404 instructions（默认4531）、VGPR496/SGPR68、zero spills/private、LDS282624 B/count ABI保持。42条独立scale b32全部合并、scale DS issue69→48、96 dwords保持；MSB375→312、normalized v_add32减少34条。原numeric/payload/output/TDM、hardware fences及physical DS CFG通过，不把static减少换算成μs。

完整两mode native及run证据已下载，receipt=`unified_scale_base_tied_native_downloaded_acceptance_boot1d5146ac_v1.json`、SHA=`c975419b1a20623548cf708a638f29f1cc02ef2400f231bd2318b399bb3cac5a`。初checker打印字段误比较留diagnostic，只修CPU checker，无候选或GPU重跑。继续双seed/双规模完整112项精度、eager/graph和same-address dynamic replay，正常live入口另绑定后才正式性能；目前正式107.379/95.3059 μs未变。

后续双seed0→17、每进程E65→E64完整精度通过，两个程序rc0/120秒内，112项逐字节比较全零、eager/graph和same-address dynamic replay通过，四次原MoE reference正常。E65 seed0/17 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`，E64为`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`。双规模同compiled object/cache1/上述完整candidate ELF，原阈值不变。

下载完整原run/source/manifest/result/idle和binding后独立验收通过；receipt=`unified_scale_base_tied_correctness_downloaded_acceptance_boot1d5146ac_v1.json`、SHA=`00eb9b7c96c4ebcaa30b57d1679e99d1cecba9867e74b4e1c466995d974302f3`，没有自身GPU残留，correctness-only所有耗时无效。继续固定默认E65/E64→候选E65/E64→默认E64/E65各三轮正式性能，normal live另做56项绑定，finally恢复默认；计时前全机idle、期间不并发GPU监控/传输/编译/ATT，只有ownership污染可整组最多三次。此时尚无本项正式性能，正式107.379/95.3059 μs保持。

#### Scale base tied v1正式性能收尾与继续方向（2026-10-20）

固定六组队列rc0结束，共27轮原random non-balanced完整MoE，精度/执行通过且各项120秒内；normal live imports额外双规模56项零字节及同compiled object/cache1/完整candidate ELF绑定通过。三组ownership污染按原规则整组三轮排除并重试try2，没有拼轮、按耗时筛选或重开队列。

| workload | 完整合格组 | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前控制try2 | 107.670 | 92.9238 | 255.936 |
| E65/topk9/shared1 | candidate try1 | 107.645 | 95.3654 | 258.103 |
| E65/topk9/shared1 | 默认后控制try1 | 107.883 | 94.7685 | 258.869 |
| E64/topk8/shared0 | 默认前控制try2 | 97.1511 | 60.4268 | 210.097 |
| E64/topk8/shared0 | candidate try2 | 94.7494 | 59.3196 | 205.623 |
| E64/topk8/shared0 | 默认后控制try1 | 95.7447 | 60.3906 | 207.689 |

E65 GEMM1相对两侧为-0.023219%/-0.220609%，e2e为+0.846696%/-0.295903%；E64 GEMM1为-2.472128%/-1.039535%，e2e为-2.129493%/-0.994757%。E65 e2e未满足事前两侧均不回退要求，候选淘汰，不反向重测或接入生产。GEMM2源码/AST不变，其计时波动不构成本项GEMM1导致回退的根因证据。

排除组为E65默认前控制try1/run2（外部OPTX PID2924373）、E64默认前控制try1/run2（post已退出PID2942092与GPU活动）、E64 candidate try1/run1（外部OPTX PID2963818）。对应三组原GEMM1读数107.822/108.446/107.632、95.9936/96.1876/96.6388、93.3286/96.3985/95.6099 μs，全部中值null，raw保留；没有干预外部进程。完整合格三轮见优化文档。

完整证据已下载，raw包SHA=`df579eafa7a2630213da08bcf0ec8906a0246b1033aeea7964519d59b439cad0`。CPU checker v1错误地要求污染组无live owner，保留原失败和checker，只修改v2匹配原owner列表并独立重算排除；没有改候选、原计时/状态/规则，也没有重跑GPU。独立验收=`unified_scale_base_tied_perf_downloaded_acceptance_boot1d5146ac_v2.json`，SHA=`7cfe554c62cfb5fe934d1fc966942bfb5430ce3e178d32d36bf0caabc9db6273`，27轮/六合格组/三排除组/SHA/精度/ELF/ownership通过。

finally恢复精确默认，两源SHA与原默认一致；2026-10-19T18:57:25Z post全机idle/KFD空/busy0/VRAM176582656 B，无本项GPU残留，后续性能前重查。正式E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。

native离线分区显示改动主要在prime/drain：loop外指令3750→3630，四steady loops合计仅少7条，每loop MSB32→34且scale DS sites仍6。它不是性能计时或硬件根因证明；尚无candidate ATT。下一方向把tied地址约束限于prime/fixed drain，保持steady原路径，先native验收，再完整精度/正式性能。新方向此时尚未实现。

#### 后续：限定scale identity拒绝、初始owner重复工作与显式四stage（2026-10-20）

限定prime/drain的`unified_scale_fixed_tied_v1_source/`已结束，两项no-dispatch编译均rc0/120秒内。虽然steady源码保留，LLVM全局寄存器分配仍改变：VGPR486→498，四steady loops每个MSB32→34、指令+1，因此按事前门槛淘汰，未运行GPU精度或性能。candidate完整ELF=`91f11aa4920a2a0bf7c1ecd2f4e3594c5d1e33c6b8195de8516687abcd11cd23`、总指令4531→4417，zero spills/private、共用count ABI保持。这是隔离优化候选的native拒绝，不能称为生产kernel的精度bug。raw包SHA=`5b2cb8265e34998294a4cfe7769958815941c8d0c6ac66eae28548213f025d03`；下载拒绝receipt SHA=`a10cfb19728e33d758fd3295f8220c36866bbc90ec3a594777be7446a098b2ea`。

复用精确默认SIMD0/1四SE ATT分析initial issue，896-WMMA sampled waves的entry span中值7045.5/6364 cycles，first→fourth input TDM issue为1833/2497 cycles，区间执行178/134条指令并有12条重复owner CBRANCH。fourth TDM→first WMMA为3404/2981 cycles。`unified_default_entry_prime_analysis_v1.json` SHA=`2fb26c6359caeb9ced7bffc6f8fbcf5168e2c1d6ffb17c0a41f94f6587bd711b`。这些不是completion latency或kernel μs，也不能唯一定位硬件瓶颈。依据Shader Guide §4.10.1 p197，TDM同wave有序但跨wave及相对其他memory无序，后续优化保持原ready/release/output同步。

原runtime四stage版本`unified_prime_loop_v1_source/`两项编译rc0，候选4273 instructions、VGPR486/SGPR68、zero spills/private，但LLVM完全展开，static input TDM仍20而没有预设4个pure-TDM backedges，原native gate拒绝、队列rc1关闭，没有GPU精度或性能。完整46项raw已下载逐SHA/native/source/run/原拒绝独立复核，包SHA=`977ee62d9b09a2bad35852785258de2167cf041439a709f0d62ba553364646dda`；receipt=`unified_prime_loop_native_downloaded_rejection_boot1d5146ac_v1.json`。没有因为静态减少而放宽旧门槛或重开队列。

继续新实现`unified_prime_owner_v1_source/`，先选择owner一次，helper内显式constexpr按stage0→3调用原issue。事前接受原20 static input sites、要求四stage内无重复owner分支且entry/总指令减少、steady MSB/指令不增长、原数值/descriptor/fences/ABI与zero spills保持。actual AST审计768 cases/1872 copy events通过，原56次K128/GEMM2保持；双规模同compiled object/cache1/完整ELF=`036059fecc06b3550428098de8f8baa874f580ab132cb36399613385f2c0e262`、4273 instructions，远端native通过。接下来独立下载验收，再双seed/双规模完整MoE精度、normal live binding及固定三轮性能；当前正式默认E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。

后续prime-owner native完整下载复核通过，raw包SHA=`deec5bdba60e705d1893b1ee9dcfb431053a282e950ce8b835068a6fdca39c11f`，独立receipt SHA=`3a5b0abed3153bd13d850ca48e5663ad6905ca4e7b48794de6ca1a25e21bfa12`。first DS前static指令874→618，四组initial TDM内部重复owner branch各3→0，steady指令/MSB保持。随后seed0/17双规模完整112项零字节比较及四项原MoE精度通过，eager/graph/same-address dynamic replay保持，共用完整ELF核对通过。E65 logits_diff=`3.7793e-06/3.7788e-06`，E64=`3.4733e-06/3.4766e-06`，原阈值不变。

验证期间外部OPTX PID3402170有GPU占用，correctness-only允许并行，所有耗时无效。CPU checker v1误要求无外部owner，保留失败记录并仅修v2绑定原owner日志、核对无自身残留，数值/reference/replay/ELF门槛保持，没有重跑GPU。完整29项raw下载SHA=`22802d3bfeb78caba7d2e2900df00df7cb4159429ec46ffb80d68657b441ad88e`，独立精度receipt SHA=`f9607b906df4165e95e9c28de3abafea00451939c1852348cf757a212ead1ddb`。继续normal live双规模56项绑定和固定六组三轮性能，空闲前每6秒等待，计时中不并发监控/编译/传输/ATT，finally恢复精确默认；尚无新正式性能，107.379/95.3059 μs保持。

后续原prime-owner正式队列rc1关闭，仅完成E65默认前控制try1/2/3，各三轮执行和精度通过、未超120秒，但三组ownership全部不合格。污染为try1/run3 settled外部PID3450241、try2/run1 post外部live PID3455048、try3/run1与run3 settled外部PID3471867/3479459，原日志均绑定OPTX workload。九轮耗时全部排除、中值null，未拼轮或追加第四次，未运行candidate normal live binding/性能，原112项GPU精度证据保持。finally精确默认已恢复，queue post当时有外部owner；独立打包时核对无本项live GPU残留。raw64项/458087 B、SHA=`31083e833fbc2397a9236243521cdb0a2e78eddcde2f9c05536e5937ce41a983`；独立全部九轮/原门槛/命令/精度/SHA/ownership验收receipt SHA=`9133e4c005c483a099830273946661dde7aca88d45bdbc4ab172bcf48704e17f`。修正早期ELF多一个`3`的转录错误，原raw/receipts保持。

继续独立`unified_half_scale_guard_v1_source/`：保持原两个2-wide ScaleA slots，以真实M64 half首行判断只跳过无任何有效M32 consumer的ScaleA读取，所有有效A/B/scale、56次原序K128、TDM/READY/release/output同步保持。不重开此前逐pair scale或prime-owner队列。actual AST完整pipeline1040 cases/58240 K128 steps/494592有效M16/N32 accumulator更新通过；枚举裁剪43904次padding scale reads，仅为CPU覆盖统计。关闭完整module精确默认、GEMM2及E64/E65共用count ABI保持。下一步a07-3/dsr1_ep4固定两项no-dispatch编译，按事前资源/分支/数值/同步门槛验收后才完整GPU精度与性能。当前正式E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。

#### M64 ScaleA读取裁剪：前端诊断与native拒绝（2026-10-20）

v1 mode0在candidate module `@jit`初始化中超120秒，runner停止、rc124关闭，尚未创建artifact目录或调用launcher，mode1未运行。默认模块独立import诊断rc0、约2.47秒；原compiler只增加phase prints/faulthandler的诊断再次rc124，栈落在FlyDSL `visit_BoolOp` / `fix_missing_locations`。v2把新15项连续`and`配置校验替换为tuple equality和三个短assert，数值/读取裁剪/TDM/fences AST不变；固定两mode实际编译均rc0/120秒内。证据支持本例长BoolOp校验导致阻塞，不断言FlyDSL一般性根因。

v2 candidate完整ELF=`162fba81ce6919f11ba911b40e4dc1ecd7b7d39583937307795a2694784da1fd`，4674 instructions、VGPR487/SGPR70、LDS282624 B、private16 B/VGPR spill4；双规模同compiled object/cache1/count ABI保持，关闭ELF精确默认。原zero-spill/private门槛拒绝、native队列rc1关闭，未dispatch该候选或记录其性能。四steady owner loops各branch9→11、MSB32→44；四scratch stores/八loads在末尾drain PC `0x53a8`至`0x5ed8`。数值/payload/TDM opcode和96个scale dwords保持；static开销不能折算μs。第三个native拒绝实际是工具要求` s_wait_dscnt 0x0`恰好11条，candidate有15条，不是已验证device race，旧门槛和拒绝原样保留。

105项raw、896841 B完整下载逐member SHA/ELF/ISA/source/run/audit/原qualifier复核通过；包SHA=`dca4592d0d7e3989eac6a2273a8030c69c37b899ecfa7b966da132892e097809`，独立receipt=`unified_half_scale_guard_rejected_downloaded_acceptance_boot1d5146ac_v2.json`、SHA=`1bb38f0ef55b7d4f3110301fb57b0c0ba15aebcd3dda5c90b6ddf09992497d35`。所有超时由本项runner终止，无本项残留、默认两源SHA保持；没有干预外部任务或重开旧队列。

继续修订的假设为：只在确认没有consumer的ScaleA half内定义两项占位零值，以消除无效分支保留旧slot值的生命周期。有效scale、全部有效计算和原功能保持，不使用const0输入；需要重新通过CPU/native/完整MoE精度才能计时。目前正式E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。

#### 后续：ScaleA guard v3拒绝与连续M32分配准备（2026-10-20）

v3已完成上述死值定义验证：无consumer的half向原2-wide slots定义两项占位零值，有效scale和计算不改。actual AST审计1040 cases/58240次K128 steps/494592次有效M16/N32更新通过，每有效accumulator保留56次原序更新，原TDM/fences保持；边界枚举裁剪43904次padding ScaleA reads，不是性能数据。

固定两mode actual compile均rc0/120秒内，关闭完整ELF精确默认；candidate ELF=`02bf47b5dff29ff697badfcc2f00f668b8f595d4986047cbf73a16458654e802`、4705 instructions、VGPR486/SGPR70、LDS282624 B/count ABI保持，但private20 B/VGPR spill5。15个scratch sites在drain PC `0x5104`至`0x5c74`；原zero-spill/private门槛拒绝，队列rc1关闭，未GPU验证或计时。额外DScnt0的原静态计数assert失败不证明device race，定义死值也未消除spill，该路线不重开。

v3完整raw94 members/852479 B下载逐member SHA/native/source/run/audit和原qualifier复核通过；包SHA=`467ab130d65b9a385e60671e843a493e64f9f68f71c40bae0063a8e319a9388d`，独立receipt=`unified_half_scale_guard_rejected_downloaded_acceptance_boot1d5146ac_v3.json`、SHA=`73e4ecb1a3b3ff9209624d908a8ddb9e120c54cb80b37ee62f39d74de06ed78d`。复用checker个别v2标签不改变实际v3文件/hash绑定。

下一独立候选按runtime实际有效行在M32边界连续分成两个前缀，192行尝试96/96、full256仍128/128，E64/E65共享原runtime ABI。input TDM owners/ring/fences和每wave原私有output LDS slice保持，通过global起点及tensor extent裁剪防止有效输出覆盖。Shader Guide §4.10.2 p199/§4.10.7 p205支持超tensor正边界stores丢弃，仍须地址/native/完整精度验收。准备v1的decorator定位误写`@jit`，实际`@flyc.jit`，persistent尚未写出就停止，只保留wrapper partial；未上传/编译/dispatch。保留失败，用独立v2修正。正式E65/E64 GEMM1仍107.379/95.3059 μs，生产默认不变。

#### 连续M32分配的编译诊断、完整精度及验收脚本修正（2026-10-20）

候选按runtime有效行在M32边界连续分成两个前缀，164行由128/36改为96/68，192行为96/96，224行及full256保持；E64/E65共用原runtime count ABI及同compiled object/完整ELF。input TDM owners/ring/fences不改，每wave仍使用原私有output LDS slice，通过global M16/M32起点及tensor extent防止有效输出覆盖。依据Shader Guide §4.10.1 p197，TDM忽略EXEC，因此裁剪依赖descriptor extent；§4.10.7 p205记载超tensor正边界stores丢弃。这是独立性能候选，未发现默认功能需要修改。

v2 mode0实际编译报`TypeError: Variable(s) ['owned_rows'] initialized as None before a dynamic if/else (scf.if)`，mode1未运行，队列rc1关闭。新temporary与原epilogue嵌套local重名触发FlyDSL closure处理；v3仅把三处新引用改名`split_owned_rows`，原嵌套local和地址/数值保持。较早CPU preflight因stdout省略字段而比较失败，也已保留并仅修工具。这些不是生产kernel的已验证device race。

v3 actual AST审计1040 cases/58240 K128 steps、每mode247296有效M16/N32更新、1794048实际A/ScaleA LDS地址、7800裁剪output descriptors和100224有效输出行通过。每有效accumulator保留56次原序更新，有效行覆盖一次，private LDS地址与有效输出不重叠，关闭完整module AST精确默认、完整GEMM2 AST保持。

两mode no-dispatch编译均rc0/120秒内，关闭完整ELF精确默认；candidate ELF=`dfca5097cca41db10a2f7a419c196195f48038511b0bdf29a4b58c4fb2bcc2e8`，4551 instructions、VGPR486/SGPR74、zero spills/private、LDS282624 B、kernarg180 B。原steady/数值/DS/TDM/fences及physical DS CFG通过，E65→E64→E65共用compiled object/cache1。native raw包SHA=`0f1f69f0a7cafadd0c245a8aaa7cbe13b4a6121b64e4f78f4e3ebb9162fa4b1d`，下载独立receipt SHA=`c51e93ad7a9e9ecd444fdc31001982615b7782c7c602f950bb937250003808d0`。

随后seed0/17、每进程E65→E64完整112项逐字节比较全零、四项原MoE reference通过，原阈值不变。E65 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`；E64为`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`。eager/graph canary、same-address动态A/ScaleA/psum与empty experts通过，实际同compiled object/cache1/上述完整ELF。

correctness-only期间外部OPTX PID36233使用GPU，原download checker v3误要求全部live owners为空而失败；独立v4又误假定owner在seed0初始idle snapshot存在。均保留原失败和工具，v5只按原pre/post/settled中的精确外部命令分类，所有数值/reference/replay/source/ELF门槛保持，无GPU重跑、无本项live残留。raw34 members/223123 B包SHA=`525f9b6b07dba3369ef2fddc58e82f3fb26839d3d5d4d1a57b06053c27d91b68`；独立精度receipt SHA=`8f5421dd470967f2ae4a7f4b9edcc324ee812f584c79a31913d47be9928d5cf8`。这两项是证据解析bug修正，精度运行耗时全部无效，性能空闲门槛未变。

#### 连续M32分配：完整性能淘汰与默认保留（2026-10-20）

固定六组三轮完整性能队列rc0，27轮原benchmark执行/原MoE精度通过、均120秒内。normal live另做E65→E64共56项逐字节比较全零、同compiled object/cache1/实际完整候选ELF绑定。各轮全机空闲门禁保持，计时中无本项并发监控/编译/传输/ATT；仅ownership污染可整组最多三次，执行/精度/解析失败立即停止。

| workload | 合格组 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前try3 | 109.356 /109.702 /107.003 | 109.356 | 94.4794 | 257.486 |
| E64/topk8/shared0 | 默认前try1 | 96.0438 /98.8669 /98.6477 | 98.6477 | 61.3835 | 209.571 |
| E65/topk9/shared1 | 候选try1 | 107.347 /107.157 /108.134 | 107.347 | 94.9286 | 257.322 |
| E64/topk8/shared0 | 候选try2 | 95.4218 /94.0682 /94.4097 | 94.4097 | 59.7518 | 205.230 |
| E64/topk8/shared0 | 默认后try1 | 94.1947 /94.3053 /96.2105 | 94.3053 | 59.5934 | 205.283 |
| E65/topk9/shared1 | 默认后try1 | 107.417 /107.623 /107.651 | 107.623 | 94.6502 | 257.188 |

E65 GEMM1相对前/后控制−1.837119%/−0.256451%，e2e−0.063693%/+0.052102%；E64 GEMM1−4.296096%/+0.110704%，e2e−2.071374%/−0.025818%。双规模GEMM1相对两侧控制均改善、e2e均不回退的事前条件均未满足，淘汰候选，不反向确认、不追加计时、不接入默认。资源或负载分配变化没有带来已验收收益，也不能从这些小幅差异唯一定位硬件环节。

排除三完整组：E65默认前try1/run2外部OPTX live PID122111，try2/run1与run3 settled外部PID129411/136869；E64候选try1/run2外部CPU compile初始化KFD PID162742。原raw和null组中值保留，没有干预外部进程或拼凑合格轮。完整179项/1207886 B raw包SHA=`69253c786c581fe52d66319306fd88140a1267ff1069e40b7566ccfecd07dbb6`；逐member SHA及独立27轮命令/精度/分组/ownership/共用ELF/default恢复复核通过，receipt SHA=`e5308739e2964355dfaa51535889a25c7337a737cd16c54e80c8969f36f2fdb6`。

finally和独立结束检查精确默认两源恢复，全机KFD空、busy0、VRAM176582656 B，无本项live GPU残留。正式E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。完整数据、原排除值和复现命令保存在`aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md`。继续采集默认GL2C memory counters；profiling耗时不作为正式性能，逻辑effective read bytes不充当真实HBM bytes。

#### 默认GL2C counters与A multicast timeout诊断（2026-10-20）

已在a07-3/dsr1_ep4完成精确默认kernel的SDK counter诊断，完整原E65 random non-balanced MoE仅诊断iters2，filter为完整默认symbol、requested iteration10；actual默认ELF/cache1/源码SHA绑定通过。原GPU workload/profiler rc0、约9.623秒，完整MoE原logits_diff=`3.7793e-06`、rel_l2=`0.0027493`、门限0.01保持。所有profiling耗时和打印的effective TB/s无效，没有提升正式性能成绩。

CSV一个dispatch148/Agent2的聚合值：`GL2C_READ_SECTORS=46045030`，按SDK定义32B sectors得到1473440960 B的L2 read request量；`GL2C_HIT=1718564`、`GL2C_MISS=4127923`，event比例29.394814%。本地GL2 verification mapping行28–29、57支持这些定义，UC reads计miss。这不是HBM bytes，也不是byte命中率；没有per-instance/AID breakdown。CSV所报VGPR248/SGPR128不代替actual ELF486/68。iteration所属warmup/profile phase尚未独立证明，不能由counter或异常小profiling耗时反推正式带宽或唯一确定瓶颈。

采集v1有host参数转发bug，`argparse.REMAINDER`遇到嵌套`--`报`unrecognized arguments`，未启动container/GPU，原失败保留。v2在container内读取plan argument vector修正，GPU采集成功；结束host `chmod`无权修改container root文件导致外层rc1，原finished已记录GPU rc0/ownership合格。仅修本项权限、无GPU重跑。post保留短暂已退出KFD entry，live owners/fuser为空；六秒后全机idle/KFD空/busy0/VRAM176582656 B，无本项残留。原CSV/log/idle/plan/source/actual ELF下载SHA独立复核通过，receipt SHA=`8aa532480547e07515a61eddf1129246f65314625e8858b4c20726724c1db57d`。

继续新A-only multicast standard timeout候选，依据Shader Guide §4.9.8 pp191–192的晚到wave会在early timeout下另生成L2请求、Table80 descriptor bit21；只改A payload timeout，实际地址/masks/计算/56次原序K128/全部fences保持，E64/E65共用原runtime count/完整ELF，GEMM2 AST不改。actual AST7168 timeout cases/896项变化、520 pipeline cases/261632 WMMA通过，关闭完整module AST精确默认。下一固定no-dispatch编译先核对仅五descriptor literals及原zero-spill资源，再完整双seed精度和固定三轮性能。尚无候选性能，正式107.379/95.3059 μs保持。详细数据和工具见`aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md`。

#### A multicast timeout候选的编译校验错误及修正（2026-10-20）

v1两mode固定no-dispatch队列rc1关闭：mode0 rc0且完整默认ELF保持，mode1 rc1/未超时、没有candidate ELF。新guard把动态`N: fx.Int32`放入Python tuple equality，FlyDSL tracing报`RuntimeError: cannot evaluate dynamic 'Boolean' as Python bool during tracing`。这是候选新增校验代码的实现错误，尚未candidate device dispatch，不能称为生产kernel时序race；原失败和关闭队列保留。

独立v2只将`N==4096`检查移至普通Python wrapper，其他constexpr guard和A-only timeout本体保持。actual AST7168项timeout/896项仅A变化、520项pipeline/261632次WMMA再次通过，原56次K128次序、地址/masks/ring/fences、完整GEMM2及E64/E65共用runtime ABI不改。native门槛保持并增加五A descriptor Group1 bit21的literal/encoded word/后续实际TDM use绑定验证。新队列在a07-3/dsr1_ep4编译、单项120秒上限，未重开v1。

旧非unified E65 standard timeout实验没有两侧一致收益，已淘汰；新实验必须在精确unified默认上完成双规模native/精度/完整性能，不能据旧实验或counter假设宣称提升。当前正式E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs，默认两源不改。

#### A-only multicast timeout v2：完整精度与正向性能通过（2026-10-20）

新候选已在a07-3/dsr1_ep4通过固定mode0→1 no-dispatch编译，两个程序rc0/120秒内，E65→E64→E65共用compiled object/cache1/完整ELF。默认ELF不变，candidate ELF=`6e3c95affca822922b99b02c0f0a73e1db092af0929dacd6fc14b48aab0ef77c`；两mode均4531 instructions、VGPR486/SGPR68、LDS282624 B、zero spills/private、kernarg180 B。actual `.text`仅五处A descriptor Group1 bit21 literal从`0x07f0000f`到`0x07d0000f`，逐条encoded word和后续实际TDM use绑定通过，中间无覆写；其他instructions字节一致。原地址/masks/data size、计算、56次原序K128和全部fences不改，完整GEMM2保持。

native完整92项raw包SHA=`b48f428e84e85b44069bf11c3aad6d8b3d7f5cd3eec29b5de50c1eed5718e943`；独立下载receipt SHA=`5d5f1f829c9a42425b81730ad01e83da330fd49ffc9ef06d49f17eabd9a2e5aa`。随后双seed0/17、每进程E65→E64通过112项零字节比较、四项原MoE reference、eager/graph canary、same-address动态A/ScaleA/psum与empty experts。原0.01门限保持；E65 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`，E64为`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`。完整33项精度raw包SHA=`cb36f7ba0ac74a8c76fac1b7e062107e38aa44b0f51bfeecbd4589d450275220`，独立receipt SHA=`162002ef745b5cbd1279a81156f5c6fe8c094b0a7a53d9f15dfd00876c6adc32`。正确性阶段耗时全部无效，无本项live GPU残留。

正向固定默认E65/E64→候选E65/E64→默认E64/E65六组三轮性能队列rc0，共21轮原random non-balanced benchmark/精度通过、各项120秒内。六合格组如下；另一个E64候选组三轮因post ownership状态不合格而整组排除并按原规则重试。normal live imports另56项比较全零，同compiled object/cache1/完整ELF绑定通过。

| workload | 合格组 | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|---:|
| E65/topk9/shared1 | 默认前 | 107.838 | 94.4942 | 257.131 |
| E65/topk9/shared1 | A-only timeout候选 | 107.067 | 94.5552 | 256.846 |
| E65/topk9/shared1 | 默认后 | 108.673 | 94.9940 | 258.678 |
| E64/topk8/shared0 | 默认前 | 96.8556 | 60.8523 | 209.369 |
| E64/topk8/shared0 | A-only timeout候选try2 | 94.1991 | 60.0172 | 205.285 |
| E64/topk8/shared0 | 默认后 | 96.5668 | 60.1080 | 208.422 |

E65 GEMM1相对两侧−0.714961%/−1.477828%，e2e−0.110838%/−0.708216%；E64 GEMM1−2.742743%/−2.451878%，e2e−1.950623%/−1.505119%。两项正向事前条件通过，继续一次固定候选→默认→候选反向确认；双规模两侧GEMM1改善且e2e不回退才能保留，不能根据小幅正向差异直接宣布稳定收益或具体硬件根因。

排除的E64候选try1原GEMM1为94.5913/95.5463/94.2179 μs，中值null。run1立即post有KFD PID806803/807287、busy1、VRAM5912989696 B，gpu_users显示已退出entry，live owners/fuser为空；六秒settled为空。仍按保守ownership规则排除整组，没有拼轮，也没有把已退出PID归因为特定外部任务。原21轮、空闲记录、source/ELF/正常入口精度及恢复证据完整下载逐SHA复核通过；145项raw包SHA=`33060b35f42c7d2e63b660d0bd7d3fa15f708c241de225868b5603055cf22ed3`，独立性能receipt SHA=`f92944fc3ed9a5dbd6b372f321f14d904fa3b0d48373f61a2962eedda5e55ce8`。

finally及独立打包后默认两源精确恢复，全机KFD空/busy0/VRAM176582656 B，无本项live GPU残留。反向队列已启动、尚未收尾；正式默认E65/E64 GEMM1仍107.379/95.3059 μs，距80 μs差27.379 μs。完整三轮数据、复现命令和限制见`aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md`。

#### A-only timeout反向收益未复现与继续诊断（2026-10-20）

固定候选→默认→候选的双规模反向队列rc0，33轮原完整random non-balanced benchmark/原精度通过、各项120秒内。六个合格组三轮如下；五个ownership不合格组原样保留、中值null，没有拼轮、按时间重试或追加第二次反向确认。

| workload | 合格组 | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|---:|
| E65/topk9/shared1 | 候选前 | 109.848 | 94.2371 | 260.194 |
| E65/topk9/shared1 | 中间默认 | 107.853 | 94.6163 | 257.769 |
| E65/topk9/shared1 | 候选后try3 | 109.019 | 94.2159 | 257.384 |
| E64/topk8/shared0 | 候选前 | 94.7619 | 61.0988 | 206.862 |
| E64/topk8/shared0 | 中间默认try3 | 95.0487 | 61.5037 | 209.385 |
| E64/topk8/shared0 | 候选后try2 | 95.2301 | 59.8613 | 205.985 |

E65 GEMM1两侧候选相对默认+1.849740%/+1.081101%，e2e+0.940765%/−0.149359%；E64 GEMM1−0.301740%/+0.190850%，e2e−1.204957%/−1.623803%。事前双规模两侧均改善且e2e不回退不成立，淘汰候选、不接入默认、不追加同方向计时。正向107.067/94.1991 μs不替代正式默认，不能把计时波动描述为新的生产bug。

排除组包括E64中间默认try1/try2、E65后候选try1/try2、E64后候选try1；原process/KFD记录保留短暂已退出状态和外部OPTX PID906610/917230/928026/957306/973297，没有干预这些任务。尤其E65后候选污染try2的106.744/106.991/106.426 μs没有用于收益。反向219项/1754578 B raw包SHA=`c57ca81744bcf44d6516b24eb9cb5bb12ffe80eb9a11fbf5adac31d7d899025e`，独立逐SHA/33轮原命令/精度/分组/ownership/共用ELF/默认恢复receipt SHA=`0c0ff211f0d5370d4734c64da27764837ad5f2f1c83c37111d0569f2d591dc1c`，稳定收益判定false。

finally和独立打包后全机KFD空/busy0/VRAM176582656 B，无本项live GPU残留，默认两源精确恢复。正式E65/E64 GEMM1仍107.379/95.3059 μs、距80 μs差27.379 μs。继续精确默认EA请求尺寸和在途请求量诊断，新增actual call stack确认采集阶段；生产kernel、有效计算与原精度门限不改，所有profiling耗时无效，尚无新counter结果。完整原三轮及复现命令保存在优化文档。

#### EA counters验收、采集阶段纠正与下一调度候选（2026-10-20）

a07-3/dsr1_ep4的精确共用默认EA诊断两项child均rc0/120秒内，实际默认完整ELF、原E65 random non-balanced完整MoE与0.01精度门限保持，`logits_diff=3.7793e-06`。sizes=`32B:1 /64B:39 /128B:0 /256B:4058955`；另一次latency dispatch=`EA_RDREQ:4059744 /LEVEL:1624319825 /DRAM:32062287`。全部profiling耗时无效，不刷新正式性能。

sizes按SDK公式加权GL2→EA请求量1039095008 B；LEVEL/RDREQ≈400.104仅作为counter clock域比值，不换算wave cycles/ns。资料单位表述冲突：`MI400_perfmons_list#8.txt` event130及`mi400_perfmon_spec#7.txt`称DRAM为variable-size requests，而`gl2_metrics#1.adoc.txt` “GL2C-EA Interface - Bandwidth”用`DRAM*32`表示HBM request bytes，按此得到1025993184 B。约7.9倍counter差与sector解释相容，仍不能唯一确定，未证明HBM饱和。两个字节值来自不同dispatch，不能组合成同dispatch命中率/流量分解；SDK CSV资源248/128不替代实际ELF486/68。

实际stack与归档`aiter/test_common.py`核对得到1次memory probe→5次warmup→2次cuda.Event→2次`run_iters_rotate`，第10次是第二次profiler rotate call，纠正旧“两次probe、第10次首个profiler”的推算。旧GL2采集未保存stack，不能追认其phase已独立证明。外层shell `.rc`没有留存，summary finished/CLOSED banner/两个child rc0均保存，未补造外层rc。

55 regular members/1645463 B raw包逐member SHA/ELF/CSV/stack/source/原精度/ownership下载验收通过，包SHA=`48674a270b650883b2df7e5bc12d78f7a9ff0a662493a552be3773ca26a81c5a`；receipt SHA=`be81deb7717ddf8af03b6b798e63ee679c5cba04eb37041af90f235a33d26032`。checker v1不认识`return worker(...)`的文本heuristic失败，v2误假设trace有`Process_Id`字段失败；旧失败保留，v3用actual AST/`Thread_Id`及PID文件名核对同一份raw，无GPU重跑、未放宽精度或性能空闲门槛。这两项是离线证据工具修正，没有发现新的生产kernel时序race。package当时KFD空/busy0/VRAM176582656 B，无本项live残留。

继续独立`MMA_GROUP 2→4` WMMA/DS调度候选，保留原数值、地址、56次原序K128、全部hardware fences和GEMM2，E64/E65共用原runtime count/完整ELF。CPU实际helper审计80 cases/1520 hint events通过；编译前修正生成工具的两项旧参数/manifest引用，尚未候选device运行。下一步a07-3/dsr1_ep4固定两mode no-dispatch compile、单项120秒，必须通过zero spills/资源/完整ISA/DS CFG/lifetime gates后才双seed完整精度及固定性能。正式E65/E64 GEMM1仍107.379/95.3059 μs、距80 μs差27.379 μs。完整counter表、依据、限制与复现命令追加在优化文档，独立OPTX及此前记录保持。

#### 共用group4调度的native与完整精度通过（2026-10-20）

a07-3/dsr1_ep4固定两mode no-dispatch编译队列rc0、每项120秒内。关闭完整ELF精确默认，candidate ELF=`6de4f08672db2b2f561a3b3f63ca8904ce557043b5d738842266314a50b4c22d`，4531→4529 instructions、VGPR486/SGPR68、zero spills/private、LDS282624 B、180 B count ABI保持。normalized变化为`s_set_vgpr_msb +5 /s_wait_dscnt −4 /v_nop −3`，原11个zero DScnt lifetime waits、Tensorcnt/barrier sequence、数值/DS/TDM操作数量保持，physical DS CFG通过。两个规模在每mode E65→E64→E65仍同compiled object/cache1/完整ELF。

编译时有外部GPU owner1271106；no-dispatch耗时不作性能、未干预其程序。完整52项native raw包SHA=`70e9b57933eb6c300a8789d5e36169a95a2721f7017bdc6d803b024a0f998cce`，独立逐SHA/ELF/ISA/source/同步验收receipt SHA=`b68e00172afdb0f4300d972237ad8df5eefa2904584b69329699e080bdf8abcc`。随后双seed双规模112项严格逐字节比较全零、四项原MoE reference及eager/graph/same-address dynamic/empty experts通过，原0.01门限保持，actual同compiled object/完整ELF。E65 logits_diff=`3.7793e-06/3.7788e-06`、E64=`3.4733e-06/3.4766e-06`。

精度35项raw包SHA=`b6fcd5961f59e6df4686c48f9f271dbb77cc5c89ca7b537729dcc56530b9cdd7`、独立receipt SHA=`ede5e8fe868a717b30589d7cf0f606c8f90a1d2d924ab82572aae03c53292e5e`，所有child rc0/120秒内、post无live GPU owner，精度阶段耗时无效。生成工具剩余版本引用在首次上传前修正并保留原preparation，无GPU重跑。这是优化候选功能验收，没有发现新的生产kernel race，也尚未证明性能收益。

已启动固定默认双规模→候选双规模→默认反序双规模的完整三轮性能队列，空闲每6秒等待、仅ownership整组最多三次、任何执行/精度/解析失败停止。normal live入口另56项比较及同ELF绑定，finally恢复默认；计时中无本项传输/监控/编译/ATT。正式107.379/95.3059 μs和距80 μs差27.379 μs暂保持，候选完整性能结论随后追加。

#### group4性能停止及原始ownership证据（2026-10-20）

固定队列在默认E64前控制连续三组三轮ownership不合格后关闭，外层rc1、`execution_failure=null`。12次原benchmark全部rc0、精度正常、未超120秒；这是性能采样门槛失败，没有新的kernel执行故障。候选没有部署live、没有运行normal live binding、没有候选性能和后控制；不能判断该版本提速或回退。

唯一合格默认E65组三轮GEMM1=`108.357/107.760/106.430 μs`，中值107.760，GEMM2/e2e中值94.1295/256.437 μs。默认E64三组原GEMM1分别`96.5540/94.9138/94.5426`、`94.8609/95.6199/97.5874`、`95.0214/96.2064/94.3233 μs`，中值全部null，整组排除、不拼轮、不重开队列。

原status/idle显示：try1/run3的settled PID1359969为另一项OPTX的`verify_runtime_expert_moe_scale_prepared257.py --seed 17`；try2/run2的live PID1366520为`verify_runtime_expert_replay_scale_prepared257.py`，settled VRAM4403597312 B；try3/run1的settled PID1375031为`verify_runtime_scale_prepared257_edges.py --start 0 --stop 5`。三个立即post另有已退出KFD entries，未凭PID猜测来源。检查时的外部owner说明采样门槛不合格，不能反推每个计时区间都发生重叠；没有杀死或干预外部任务。

finally于2026-10-20T00:53:35Z为默认两源精确SHA、KFD空/busy0/VRAM176582656 B、无本项live GPU残留。85项/679652 B停止专用raw包SHA=`9f3809fed1370e1c273fdb8463120e7bebc3380145b50ff482d769908ec8c598`，独立下载验收receipt SHA=`a8ec56c5a97b08a4425a682964b348ac3d93d433bd029cd44938ef07ca07ae3d`；全部原12轮、排除原因、rc1和缺失candidate stages保留，没有为六组完整checker补造数据。

正式可保留E65 GEMM1/GEMM2/e2e仍`107.379/94.3843/256.893 μs`，E64仍`95.3059/60.4664/209.078 μs`，E65距80 μs差27.379 μs。下一步继续独立源码/依赖方向分析；本轮未发现新的生产race。独立OPTX与side记录保持。

#### post-RA调度候选的编译结果与FlyDSL enum setter问题（2026-10-20）

关闭GEMM1 `enable-post-misched`的独立候选已完成固定两mode no-dispatch编译，两个child均rc0/120秒内。关闭控制完整ELF精确默认；候选ELF=`3d6292ce18f78b6a81cca5c9ebdbb35ecc1badc84178f67995bc2a36e8dd3aae`，指令4531→4782，`s_set_vgpr_msb +160 /s_delay_alu +82`。VGPR486/SGPR68、LDS282624 B、zero spills/private、count ABI、数值/DS/TDM及原hardware fences保持，E64/E65共用同object/完整ELF。native拒绝总指令增加，因此外层rc1、candidate未GPU精度/性能或live部署；不能称实测回退。原准备工具的两次本地失败也保留，未上传或启动device队列。

54项/628205 B原始native包SHA=`d781de76cf06c77e0057e2ea2fcd97941382141f24b4df2d39a4759cbf544bd3`、独立验收receipt SHA=`5b7ae6c1caf7b620349f8dc234e3b1041f54979f5affa8499820e48a9560a71e`；package时外部OPTX PID1641757保持，未干预，默认源精确SHA、无本项live GPU残留。

后续独立bottomup候选只新增GEMM1 `"misched-postra-direction": "bottomup"`，post scheduler开启、GEMM2及所有function AST保持。固定mode0 rc0、默认完整ELF相同；mode1在`flydsl/compiler/llvm_options.py:35`调用`_fly.set_llvm_option_str`时报`RuntimeError: basic_string::_M_construct null not valid`，rc1、未超时，外层rc1后关闭。candidate ELF和native acceptance未生成，没有GPU精度、性能或live部署。

实际远端LLVM将`misched-postra-direction`声明为`cl::opt<MISched::Direction>`；FlyDSL CAPI string setter按`ValueRequired`区分bool后，将该enum `static_cast<cl::opt<std::string> *>(opt)`读取旧值。源码类型不匹配与当前异常相符，属于本实验遇到的compiler API类型问题，不能归为生产GEMM1时序race，也不证明bottomup性能。未修改共享FlyDSL安装、未用int setter绕过enum、不重开失败队列。实际远端源码SHA/章节摘录保存在`no_mem_cluster_remote_llvm_excerpts_v1.json`。

48项/480804 B失败raw包SHA=`9fb1dcfc2447d063a4413ff5cbe185e68833ce8f5ba1d78c28ce20b9e733d577`、独立验收receipt SHA=`f3ca28339000a8c840613daacd7a3b9d0e48ee175fca96f6bb2b1f1aae789008`，原stack、exit statuses、默认五件套及candidate产物缺失均核验，package KFD空、默认源保持，无本项live残留。

#### memop clustering候选的双规模完整精度与后续性能（2026-10-20）

新独立候选仅在GEMM1新增`"misched-cluster": False`；实际LLVM为bool，控制pre-RA/post-RA load/store clustering mutations。post-RA scheduler开启、原数值/地址/56次K128有序更新、全部hardware fences、GEMM2与原benchmark保持。固定两mode no-dispatch编译全部rc0/120秒内，E64/E65共用runtime count/同compiled object/cache1/完整ELF。

candidate ELF=`6c638b85aad8e496c3c486c9c0064d25ed572ed20cc6a8502d56023571be2540`，4531→4530 instructions，normalized仅`s_nop −1`；原VGPR486/SGPR68、LDS282624 B、zero spills/private、180 B ABI、11个DScnt0及Tensorcnt/barrier sequence保持，physical DS CFG通过。实际`.text`变化；离线diff同时包含寄存器重命名、重排和branch位移，不能按变化数量推算性能。53项/630486 B native包SHA=`b65a6d3b11ade6f1849e5b924cd1daeeb62997d9b142457acfe888a0a0d2a7a4`，独立native receipt SHA=`63fa029dacd67db934db380c4357126bd06743648f65603896b27b617299ffdd`。

固定seed0/17、每进程E65→E64的112项完整MoE有效字节比较全零，四项原reference和eager/graph/same-address dynamic/empty experts通过；E65 logits_diff=`3.7793e-06/3.7788e-06`、E64=`3.4733e-06/3.4766e-06`，原0.01门限保持。两个child rc0/120秒内，实际同candidate ELF，无expert specialization。seed0 post有外部owner1782686，correctness-only耗时无效、不作性能、不干预；seed17 post owners空。35项/269338 B精度raw包SHA=`e7e6d6d16d81bc08328410d0589366bcdf6d104b7eed390306c688140ebdcc64`、独立精度receipt SHA=`9db56b0d5d11e56d43be7af3d2343a907855d4cd09d42e7c5a105fbc1a001691`。

已启动固定默认双规模→候选双规模→默认反序双规模完整三轮性能，空闲每6秒等待、仅ownership整组最多三次、执行/精度/解析失败停止，normal live另56项及同ELF绑定、finally恢复精确默认。此时没有该候选正式性能结果，E65/E64仍107.379/95.3059 μs、距80 μs差27.379 μs。本轮全部只用a07-3/dsr1_ep4，没有发现新的生产kernel race。完整复现、native/精度表及限制追加在优化文档，独立OPTX/side与既有历史保持。

#### memop clustering性能队列停止及下一项可选delay-ALU实验（2026-10-20）

以下准备阶段记录由本节后续实际验收补充，不作为未完成状态的当前结论。

#### 可选delay-ALU候选的actual native、完整精度及性能队列（2026-10-20）

固定两mode no-dispatch编译及两个child均rc0/120秒内。mode0完整ELF精确默认，mode1 ELF=`edd3e867ec099281565c959e0c57059a73727798c3ab26472d9671971eeaaa90`，4531→4176 instructions，唯一normalized数量变化为`s_delay_alu −355`。全部4176条保留指令和112个branch目标的保留ordinal完全相同，原VGPR486/SGPR68、LDS282624 B、zero spills/private、180 B kernarg/count offset176/size4、MSB/nops/DS/TDM/waits/barriers保持；有限physical DS CFG通过。E65/E64每mode仍cache1/同compiled object/完整ELF，没有expert specialization。

53项/628605 B native raw包SHA=`2abc6b8e9d03e86f5975443bce973b0d3f9a4bdd7f53ca31d329b37c9d4da968`、独立native receipt SHA=`97c41926b4261242b21529cd4a58e5cc7343070aca2c12737e616c3e47c33cd2`。编译post有外部owner2024923，不干预、不将编译耗时当性能。

固定seed0/17、每进程E65→E64的112项有效payload/ScaleA/GEMM2/MoE比较均零字节差异，四项原reference及eager/graph/same-address dynamic/empty experts检查通过；原0.01门限保持。E65 logits_diff=`3.7793e-06/3.7788e-06`、rel_l2=`0.0027493/0.0027491`；E64=`3.4733e-06/3.4766e-06`、`0.0026357/0.0026369`。两项rc0/120秒内，actual同candidate完整ELF/cache1/对象。seed0 post外部owner2069831、seed17 owners空；correctness-only耗时全部无效。

35项/269650 B精度raw包SHA=`6d71a13ed6044857a3f9da0ae87afef1f7ffb46e7f2741fe145e5fb8f0594e95`，独立下载逐SHA/112项/reference/replay/完整ELF验收receipt SHA=`12fbb0b1d34e02a6e763c28acb84fe609db858359c8d774b594f89a95e2c0cc4`，已上传核对。默认两源SHA保持；该实验未发现新的生产race，删除optional指令不能直接换算性能。

已在查看性能结果前准备和审查完整六组三轮固定队列及raw验收工具并上传核SHA，队列为默认E65/E64→候选E65/E64→默认E64/E65，ownership整组最多三次、执行/精度/解析失败停止，normal live另56项和actual同ELF绑定、finally恢复默认。单项120秒、全机idle/6秒等待保持，计时中不并发本项传输/编译/打包/ATT/GPU监控。此时正式E65/E64仍107.379/95.3059 μs，距80 μs差27.379 μs；原有效计算与GEMM2保持，后续性能结论追加。完整命令与证据在优化文档，独立OPTX/side和既有记录保持。

#### memop clustering停止与delay-ALU准备阶段的原记录

固定队列在默认E65前控制的三次完整三轮组全部ownership不合格后关闭，九次benchmark全部rc0、精度通过、120秒内；外层rc1、`execution_failure=null`，没有candidate live部署、binding、正式性能或后控制。三组原GEMM1为`107.807/125.395/107.381`、`108.633/106.727/108.800`、`107.098/108.070/107.820 μs`，三列中值全部null；不拼轮、不重开，不能将该结果解释为候选提速或回退。

原owners显示try1/run2 PID1831669为`jli-mi450`的E384 A8W4 grouped GEMM，try2/run1/run2 PID1847541/1851788为`zan_att`的E64/T1536 A4W4 grouped GEMM，try2/run3 PID1859498为另一OPTX的`verify_runtime_expert_replay_scale_prepared260.py`。未干预外部任务。try3/run3 post entry1919764已退出，没有实际命令，不能猜测其身份；六秒settled恢复idle。finally于2026-10-20T01:59:40Z为默认两源精确SHA、KFD空/busy0/VRAM176582656 B，未发现本项live GPU残留。71项/550788 B停止raw包SHA=`44f34d2463ff0dc814dc9455b22a9f7b2d286797c1295c24a7fb06a77f475469`，独立九轮/缺失stages下载验收receipt SHA=`257befc46808d64c0de8d37c3053ac78247fa3463de863d4f2b95b7b5b19968e9`。

下一独立候选仅关闭GEMM1的`amdgpu-enable-delay-alu`compiler pass。CDNA5 ISA §5.8 p57将`s_delay_alu`明确为可选性能调度指令；不扩展为可以移除`v_nop`、DS/TDM waits、WG barrier或MODE规则。实际远端LLVM bool选项及pass启用源码已经核对，初次查询错误文件名`SIInsertDelayAlu.cpp`为read-only错误，实际为`AMDGPUInsertDelayAlu.cpp`。当前默认有355条optional delay；native事前要求仅删除这些指令、所有保留指令与CFG目标、VGPR486/SGPR68、LDS282624 B、zero spills/private及共用count ABI完全保持。98个实际function AST和完整GEMM2相同，固定两mode no-dispatch编译已启动、单项120秒；此时没有candidate GPU精度或性能。该优化实验没有发现新的生产race，最优107.379/95.3059 μs保持。

#### delay-ALU候选性能停止及实际入口完整精度（2026-10-20）

实际固定队列完成默认E65、默认E64、候选E65各三次完整三轮，共27次原benchmark。全部child rc0、原MoE精度True、120秒内；两个默认第三组合格，候选E65三组均ownership不合格。外层rc1、`execution_failure=null`，异常=`Three ownership-contaminated whole groups: E65 mode1`。候选E64和两个后控制未执行，按事前三次上限关闭、不重开、不拼轮、不按时间追加。该停止由采样条件引起，没有新的kernel执行或精度故障。

| workload / 路径 | GEMM1三轮 μs | GEMM1完整合格中值 μs | GEMM2完整合格中值 μs | MoE e2e完整合格中值 μs |
|---|---|---:|---:|---:|
| 默认E65前控制try3 | 108.032 /108.865 /108.121 | 108.121 | 94.6714 | 258.289 |
| 默认E64前控制try3 | 94.9579 /95.1519 /95.4407 | 95.1519 | 60.1902 | 207.937 |
| 候选E65 try1 | 109.384 /109.485 /110.900 | null | null | null |
| 候选E65 try2 | 109.105 /112.323 /111.916 | null | null | null |
| 候选E65 try3 | 109.030 /109.371 /109.127 | null | null | null |

候选normal live入口seed0 E65→E64另56项有效payload/ScaleA/GEMM2/MoE字节比较、原reference、eager/graph及dynamic replay通过，actual cache1/同compiled object/同完整ELF=`edd3e867ec099281565c959e0c57059a73727798c3ab26472d9671971eeaaa90`。加上已验收snapshot双seed112项，功能与原精度门限保持。但候选没有完整合格性能组，不能用原读数宣称性能回退或提升。

候选try1/run2的post已退出entry2206023，无法识别来源；try2/run3 post entries2216860/2217627及settled PID2218049的另一OPTX `verify_runtime_expert_replay_scale_prepared262.py`；try3/run1 post entry2224591及settled已退出entry2225945；try3/run3 live PID2229131为`zan_att`的E64/T1536/topk8 A4W4 grouped GEMM。全部原status/idle/命令保留，未干预外部任务，未从检查时状态推断每个计时区间必然重叠。

finally于2026-10-20T02:35:39Z等待全机idle并恢复精确默认两源SHA，KFD空/busy0/VRAM176582656 B、无本项live GPU残留。181项/1488070 B停止专用raw包SHA=`2b5ddf9d3344db1ebb9b84fc297b06c07c615bbc59e359e01a1ad802582c6602`，独立下载验收27轮原命令/精度/ownership/三轮组/null medians/56项live/两次source transition/rc1/缺失stages通过，receipt SHA=`47e9c3a9c168437a882eab42138516d0347ba7d41fe9c5fdccbe3756bd87238d`、已上传核对。打包时外部OPTX2357449及其他kernel测试2362619存在，无本项owner；没有并发性能计时。

正式E65 GEMM1/GEMM2/e2e仍`107.379/94.3843/256.893 μs`，E64仍`95.3059/60.4664/209.078 μs`；距80 μs差27.379 μs。完整数据、复现入口及限制见优化文档。本轮只用a07-3/dsr1_ep4，未发现新的生产kernel race，独立OPTX/side及旧历史保持。

#### Paired LDS方向核查及no-VOPD actual native拒绝（2026-10-20）

已核对旧pipeline的paired B64/align8前端拒绝、N512交错spill和weight-major实验。当前LDS row2080 B、K128 subtile1040 B、stage70656 B；Shader Guide §4.7.2 Tables61–62 pp160–161的B64 two-address各8-bit offset乘8，普通reach2040 B，不能假定当前跨row配对无额外地址成本。实际FlyDSL没有普通read2 intrinsic。CDNA5 ISA §15.15 pp738–739的STRIDE64_B64 prose写256 B、公式乘512 B，保留文档冲突；Table86 p235的throughput不能代替实际收益。本方向未编译或GPU dispatch。

下一独立候选只关闭GEMM1 `amdgpu-enable-vopd`，post scheduler和delay-ALU保持、wrapper/GEMM2默认，98个function AST相同。硬件配对和bank规则已查Shader Guide §4.6.8 pp134–135、§5.3.2 p218和§5.3.5 pp221–222，实际LLVM bool选项及CreateVOPD源码保存于`vopd_remote_llvm_excerpts_v1.json`。固定mode0→1两child均rc0/120秒内、no-dispatch；两规模各mode仍同compiled object/cache1/完整ELF、原180 B ABI/count offset176。

候选完整ELF=`aeeb1c61050032d67ff2b135634e8b0a6ff594554b1f468dd00185ec6007bf77`。raw executable4531→4712，拆开VOPD后的primitive4753→4712，222对VOPD全部消失；normalized唯一变化`s_delay_alu −41`。VGPR486/SGPR68、LDS282624 B、zero spills/private、原数值/DS/TDM、11个DScnt0、Tensorcnt/barrier sequence和有限physical DS CFG保持。然而MSB总数仍375，四owner的8个backedge primitive逐段相同、每段MSB仍28/32。原native拒绝为`VOPD removal did not reduce total VGPR-MSB switches`及`No steady owner interval reduced VGPR-MSB switches`。外层rc1是收益门槛拒绝、不是compiler child失败；未进入GPU精度/性能/ATT/live部署，不重开。

53项/628821 B raw包SHA=`0e80c9aa456787bb6838c60bf05752ba3cf5e9072ddb92141be6cc1df94a0738`。首个下载checker在最后primitive字段比较遇到JSON字符串key与本地整数key不一致，失败记录保留；v2只normalize keys并重新核验每member SHA、实际binary/ISA/launcher/resources、原source/命令/timeout/同object/DS/fences和原拒绝，未重跑compile/GPU。独立receipt SHA=`d63616a3ff1d7d12f29ff771e088a558331eb322079e8eb6927d6052cc1b9072`，已上传核对。

当前正式E65仍`107.379/94.3843/256.893 μs`、E64仍`95.3059/60.4664/209.078 μs`，距80 μs差27.379 μs；两规模完整精度及共用默认ELF验收保持。没有新增生产kernel bug/race，也没有把static ISA差值当作性能。完整表和证据在优化文档，独立OPTX/side及旧历史保持。

### E65 GEMM1优化收尾与最后B地址实验（2026-10-20）

用户要求完成手头实验后结束优化。最后三个独立B LDS地址表达snapshot均已关闭，不再启动新优化或重开旧队列：

| 候选 | native结果 | GPU精度 /性能结论 |
|---|---|---|
| `unified_b_base_tied` | 指令4531→4434、地址primitive222→191、MSB375→310；VGPR486→498，按16分配496→512，资源门槛拒绝 | 无GPU精度或性能，未接入生产 |
| `unified_b_lane_tied` | 指令4531→4536、地址222→221、MSB375→407、SGPR68→70及steady primitive增加，拒绝 | 无GPU精度或性能，未接入生产 |
| `unified_b_subtile_tied` | 指令4531→4488、primitive4753→4711、地址222→195；原VGPR486/SGPR68、LDS282624 B、zero spills/private、kernarg180保持 | 双seed完整精度112项0差异；无合格候选性能、未接入生产 |

最后候选在B load消费点用空tied asm identity分离K128 subtile base，CPU28672 cases、917504 dwords及2048 modulo32检查通过，普通DS loads、全部有效K128更新、数值与同步保持。其MSB总数实际增加375→386、steady短/长28/32→32/36，不能声称减少MSB。candidate完整ELF=`b6a9b7cb2a2669502e298569d2be84c2c93b6a03963026c644ad7979f6c47e49`，snapshot双seedE65/shared1→E64/shared0均同compiled object/cache1/完整ELF，原reference、有效payload/ScaleA/GEMM2/MoE、eager/graph及dynamic input/psum/empty专家通过。

固定六组三轮性能队列只完成默认E65前控制的三次完整组，九次原benchmark均rc0/精度True/120秒内，三组均ownership不合格，GEMM1/GEMM2/e2e中值全部null。原GEMM1三组为`108.252/108.819/107.105`、`107.295/106.797/107.942`、`108.684/107.624/108.062 μs`；ownership分别`True/False/False`、`True/False/False`、`True/True/False`。外层rc1=`Three ownership-contaminated whole groups: E65 mode0`、`execution_failure=null`。candidate未部署live、normal live binding、candidate性能和后控制未执行，无法判断提速或回退，不拼轮或重开。原post/settled有`/tmp/b32_work/pilot.py`及另一独立OPTX benchmark，已退出KFD entry不猜来源，未干预外部任务。

raw证据与独立下载验收均保存在worktree的`.codex_tmp/dsr1_e65_gemm1_20261017/`：

- base native：59项/687665 B，raw SHA=`2e10e531c0421c8c72e030214da2211eeb94bce8fd412f5c168e417058fc3964`，receipt SHA=`2b9711ccb5a85a5c23de90645d79ed718e4575a3ca4d0cbc13941ee42433288b`。
- lane native：56项/695836 B，raw SHA=`ebe3f53a039b29b850fb2f871433216dd7fb24b1a76964a353f998dbc56f3c3d`，receipt SHA=`718ed3a683ae9fa723e722a1f7ba58a517ccb82f83e1df34eeab3f59f6a66a9c`。
- subtile native：56项/694055 B，raw SHA=`89af0734aa73f741e963a93c7c8c72a5c7ae1c4b0b4b126639de437a96898028`，receipt SHA=`b268b46c9efe42f563b4808829f86d95ee3354d6c9ea4ea2f289ca981477e91d`。
- subtile完整精度：35项/269811 B，raw SHA=`1af533b354e42b7b0b1a866e156c9ba793015f78e9b07d15e89bb996783f9a55`，receipt SHA=`49ee117365ad273ae94a7336ac4000948ddcd98665c89a1b853edac327d0f364`。
- subtile性能关闭：73项/1516975 B，raw SHA=`a744f827b382ed5bce065cd076be4be133bfaf143df2f63a6cfd5e34dc81902a`，receipt SHA=`6637b0539171a06e8c1feb7be7bac4f8b0f83c240a73880e8e2e110951d5eccd`。

本次没有发现新的生产kernel bug/race。最优版保留共用runtime-E/lookup7/group8，完整ELF=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`。正式E65/shared1仍`107.379/94.3843/256.893 μs`、E64/shared0仍`95.3059/60.4664/209.078 μs`，E65距80 μs差27.379 μs，未达到最初90 μs目标；相对历史原HEAD153.405 μs约改善30%。旧控制批次不替代正式验收。

明确的瓶颈证据是padding裁剪后仍保留DS operand搬运、地址/控制、WG同步和epilogue成本：历史group8 ATT的448/672/896 WMMA组compute span P50为20714/21817/22177 cycles，DS wait累计832/813/2022、WG wait2711.5/1511/1082 cycles，epilogue4036.5/4123/4195 cycles。该ATT属于旧E65-only ELF `7c643f...`，不可冒称当前runtime-E ELF的trace；最新native的486 VGPR、282624 B LDS和375个MSB sites独立保留。Shader Guide §3.3.2.1 p32、§4.3.7.2.4 p90、§5.7.6 p235支持资源、DS completion和LDS规则，不能据issue距离当completion latency，或把不同中值相加换算μs。EA来自不同dispatch且DRAM单位有资料冲突；现有证据未证明HBM饱和或唯一定位某个FIFO/cache/链路。

原优化关闭状态=`e65_optimization_final_host_state_boot1d5146ac_v1.json`，2026-10-20T04:28:35Z，SHA=`bfbe2af0b04ad010058bee71270798b1e711617fcfba78ba5e117b8d0381b2c5`，无本任务进程残留；外部PID3720150为`/tmp/b32_work/sweep.py time ... --cube`。原记录保留。详细表、候选复现入口、raw包文件名及限制已写入`aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md`。

### 问：当前最优版本支持E65/topk9、没有shared expert，是否只改准入而不用改kernel？（2026-10-20）

答：可以，而且已经完成并验证。生产代码仅修改worktree中`aiter/ops/flydsl/grouped_moe_gfx1250.py`的profile准入及对应debug文字；底层`grouped_gemm_mxfp4.py`和`mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py`逐字节保持，既有routing、量化、精度门限和计时方式保持。此次是停止优化后的准入补充，不继续展开优化实验。

此前上层只接受routed E64/topk8，严格`AITER_MOE_NUM_SHARED_EXPERTS=1`才能将physical E65/topk9映射到routed64/8；shared0是真实routed65/9，因此未获得exact t256 profile。新增`_find_e65_topk9_nonbalanced_token_profile`仅放行当前T1536、7168/2048、A4W4、SiLU、无bias、non-balanced、非EP/非EP scatter的E65/topk9/shared0，复用原`t256_persistent_any_routing`。旧selector范围、shared metadata严格语义及其他shape/bias/EP fallback不变。

底层最优准入原已覆盖`(n_experts, contiguous_m)=(65,30464)`；kernel使用runtime expert count和真实`psum`，不要求expert64是shared。新场景直接保留65列score的`fused_topk(...,9,True)`，13824条真实route；不追加expert64/权重1、不伪装shared1。两seed capture都确认65项expert counts、总routes13824，expert64没有被固定为每token必选。

CPU检查执行实际上层选择块和底层guard AST，与修改前全部旧routed token profiles对照通过；Python语法检查通过。远端只用a07-3/dsr1_ep4，先保存真实shared0双seed的原HEAD compute对照，随后正常imports验证新E65/shared0→原E65/shared1→E64/shared0。两个seed各84项、共168项有效字节比较全部0差异，原reference、payload/ScaleA、GEMM2有效行、完整MoE、三次eager/graph和三组same-address dynamic random input/ScaleA/psum（含empty专家）均通过。

| workload | seed0 logits_diff /rel_l2 | seed17 logits_diff /rel_l2 |
|---|---|---|
| E65/topk9/shared0 | 3.4699e-06 /2.6344e-03 | 3.4695e-06 /2.6342e-03 |
| E65/topk9/shared1 | 3.7793e-06 /2.7493e-03 | 3.7788e-06 /2.7491e-03 |
| E64/topk8/shared0 | 3.4733e-06 /2.6357e-03 | 3.4766e-06 /2.6369e-03 |

三个规模实际均cache1、同compiled object、同34360 B完整ELF，SHA=`9a12287efd4cf4759f15fb5fa71c4f3291262a002dff8f2db2bab274f6fce23b`，精确等于当前最优；没有新增expert/shared编译specialization。新shared0只做correctness-only，所有耗时无效，未记录正式性能，不能套用shared1的107.379 μs。

四个GPU child全部rc0且在120秒内完成；最后host普通`chmod`因容器root生成输出而失败，原外层rc1日志保留。之后`sudo -n chmod -R a+rwX`只修复本项输出，独立核对原日志完成收尾，没有重跑GPU测试。最终无本任务进程/KFD owner残留。

本地worktree修改后，仅上传必要生产文件stage和本项工具；远端按原SHA守卫、原字节备份、原子替换安装。准入文件新SHA=`97ac61a37ca72623ed1988c0f573bcb1a5f4793a4f7903076d67526c9d422668`；wrapper仍`798a865d...`、persistent仍`29cfe04e...`。没有Git clean/reset/checkout/commit/push，独立OPTX/side和既有历史不覆盖。

使用原benchmark命令时，将`--shared-experts 1`换为`--shared-experts 0`，也可直接省略该参数（benchmark默认0）；保留`--experts 65 --tokens 1536 --topk 9 --model-dim 7168 --inter-dim 2048`和原non-balanced环境。完整命令与验证工具已写入优化文档。性能数据仍须经过host全机idle门禁；本次精度补充不提供新性能结论。

证据包=`e65_no_shared_admission_raw_v1.tgz`，52 regular members/508540 B，SHA=`f85cccd8128a8f4bfa3b61469aac1bd025e6471ca18071606508756cdaaa8d6b`；独立下载验收=`e65_no_shared_admission_v1/downloaded_acceptance.json`，SHA=`9363dc47728e00255c1531c1d1d6176f8be6576ebcfeb5b86a743ccf171ab076`，已上传核对。大parent `.pt`仅留host并封存SHA，不声称下载。准入补充关闭receipt=`e65_no_shared_admission_v1/closeout_host.json`，SHA=`e00c9ad00eef71d5aa6ff43cf54caab981ce53d9478ee7115f3cee8ecf69bc9f`。原优化关闭状态和权限失败日志保留；优化工作结束。

### 问：优化文档中E65/shared1和E64/shared0的baseline与最优版性能如何对比？（2026-10-20）

答：按用户最新要求，仅对比GEMM1和MoE e2e。下表取完整合格三轮的各列中值，均为T1536/model_dim7168/inter_dim2048、A4W4、SiLU、无bias、random non-balanced。E65为topk9/shared1，E64为topk8/shared0。数据来自`aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md`：E65 baseline为第102行的原HEAD `baseline_settled`；E64 baseline为第2672行的原默认路径“旧前控制”；最优版为第2730–2731行最终保留的共用runtime-E/lookup7/group8正常生产入口结果。

| workload | 版本 | GEMM1 μs | MoE e2e μs |
|---|---|---:|---:|
| E65/topk9/shared1 | baseline | 153.405 | 299.172 |
| E65/topk9/shared1 | 当前最优 | 107.379 | 256.893 |
| E64/topk8/shared0 | baseline | 99.0062 | 213.938 |
| E64/topk8/shared0 | 当前最优 | 95.3059 | 209.078 |

耗时降幅按`(baseline - 最优) / baseline × 100%`计算：

| workload | GEMM1耗时减少 μs /% | MoE e2e耗时减少 μs /% |
|---|---|---|
| E65/topk9/shared1 | 46.0260 /30.00% | 42.2790 /14.13% |
| E64/topk8/shared0 | 3.7003 /3.74% | 4.8600 /2.27% |

E65的GEMM1耗时下降30.00%，MoE e2e下降14.13%；E64分别下降3.74%和2.27%。以上是不同测试批次的合格中值对比，本轮没有重新跑GPU，不能称为新同批前后重测。E64独立GEMM1 baseline92.760947 μs使用不同测量入口；100.710 μs仅是单轮合格完整MoE样本；两者都不代替本表的完整三轮baseline。原合格“旧后控制”的GEMM1/MoE e2e为99.5166/212.554 μs，继续保留在优化文档，不与前控制混合成新baseline。新E65/shared0尚无正式性能，不包含在本比较中。
