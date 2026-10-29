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


### 问：v3 波动定位目前进展如何，定位到根因后是否继续优化？（2026-10-25）

生产保留 v45，pin=0 正式 GEMM1 三轮为 80.9644 /81.3186 /118.372 μs，
中值 81.3186 μs；异常第三轮完整保留，稳定性补测不覆盖正式结果，本轮没有新 MoE e2e 性能。
按要求先定位 v3 波动，再恢复 kernel 优化；当前没有新生产 kernel 修改。

正常异步相同 ELF 的 module 三轮已通过原 MoE 精度、完整两阶段 ELF、output 字节一致性
以及全部 90 个 GEMM1 profiler 窗口的独立复核。
同一进程、同一 input/output 指针与机器码复现 **90.5134 → 78.8855 → 91.0910 μs**。
更换原 module 的 `CallState` 没有恢复快状态。W1、A1、ScaleA、ScaleB 单独指针对照
均未出现更换该指针即可恢复快状态；A1 补测的 GEMM2 profiler 缺失 event 仍保持采集失败记录，
不追认整批通过，不采用其 GEMM2 耗时。

公共 HSA loader 接口已通过实际 SDK 的 C/Python ABI 核验，首次 serialized 正确性验证
确认四个实例的实际 GPU `.text` 地址不同，加载字节与冻结 ELF 完全相同。
该接口验证轮次有外部 KFD owner 重叠、post/settled 未空闲，保留 controller 未完成，耗时不采用。
正常异步 HSA 对照使用新 label，等待全机空闲后执行，每个测试 child 不超过 120 秒。

现有证据将范围收敛到加载实例相关环节，尚不能唯一确定 GPU code placement、
指令供给或 host JIT 状态，不能直接将 I-cache 宣布为根因。
硬件依据为本地 `MI400_Shader_Programming#65.txt` §1.5、§5.7.1；
其 instruction cache/Instruction Store 描述不能单独证明本次波动原因。
完整数据与复现命令见 `my_code/moe_gemm1_v3_variability_diagnosis_20261025.md`。



### 问：v3 性能波动原因定位到哪里，接下来继续什么？（2026-10-26）

正常异步固定 host 对照的三个 child 均已完成，分别耗时 11.5719 /13.1770 /11.2910 秒，
三轮 idle/owner 门禁全部通过。90 个 GEMM1 profile 的原 19 次均值、完整 event-count、
原 MoE 精度、完整两阶段 ELF、实际加载 `.text` 字节及 output payload/scale 字节身份
通过独立本地复核。固定一个原 host function、`ExecutionEngine`、`CallState` 和原 C getter，
只在计时窗口外切换 GPU function 后仍复现快慢往返，15 个加载实例全部符合原地址预测。
group 0/2 约 78.49–79.71 μs，group 3 约 90.36–90.94 μs；本批没有 group 1。

首次原型将每次 launch 的 `mgpuModuleGetFunction` 交给了 Python observer，190 次查询中
80 次落在 profile 窗口，检查在 8.8175 秒停止。该失败记录不变，局部耗时不采用。
修正后 getter 直接走原 C runtime，native proxy 保留完整普通和 cluster launch ABI。
这属于诊断工具修正，不是新发现的 kernel 功能 bug。

硬件资料 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2 和
`MI400_SQC_UTCL0_Integration_MAS#22.txt` §1/§2 说明 SQC 使用 virtual address/virtual tag，
set hash 为 `address[13:9] ^ address[18:14]`。目前确认 GPU 加载实例地址相关性，并进一步
排除了本次往返由 host JIT 实例变化解释；具体 I-cache conflict 仍需 counter/匹配 ATT 验证。
先完成该机制验证，再恢复 GEMM1 优化；测试继续仅在 a07-3、`dsr1_ep4`、pin=0 下执行。

以上耗时属于 E64/T1536/topk8 random non-balanced 的定位实验，不更新最优版正式成绩。
生产 v45 正式 GEMM1 三轮仍为 80.9644 /81.3186 /118.372 μs，中值 81.3186 μs，
异常第三轮保留；本轮没有新增 MoE e2e 性能。详细命令、表和证据身份见
`my_code/moe_gemm1_v3_variability_diagnosis_20261025.md`。


### 问：v3 波动原因现在定位到哪里，接下来继续什么？（2026-10-26 counters/ATT 补记）

v3 的 code 加载地址相关 I-cache miss 放大已取得直接证据，是快慢档位的主要机制。
相同 ELF、加载 text、一个固定原 host function/ExecutionEngine/CallState、相同输入输出指针和 native cluster proxy，
只切换实际 HIP function 就能往返恢复快慢；counters 随目标实例恢复。四轮共有 120 个 GEMM1 profiler 窗口、
5,524 次 GEMM1 dispatch、22,096 条 counter 行全部独立绑定。每 dispatch 的 SQ_WAVES=7,168，
requests 基本相同；group 0/2 misses 约 5.9k，group 1 约 26.1k，group 3 约 28.8k，后者约为快档 4.9 倍。
SDK 的 nondeterministic/per-SQ/per-Bank 描述保留，事件比值不冒充严格 cache 概率。

新 ATT pilot v3 完成，capture 命令 34.8843 秒，owner/post/settled 通过，生产 v45 恢复。
preflight PID 1911960 和真正 ATT PID 1912205 分开保存；真正 ATT 的 1,381 次 GEMM1 dispatch 全部保留，
按 native before/after 绑定 profiler 窗口与目标 HIP function。五个 code objects 的 SIMD0、四个 SE、
raw/decoded、全部 12,733 条 executable ISA/冻结 ELF、五份加载 text 和 output 字节身份逐个验收通过。
80 个 active waves 按完整 PC sequence 匹配成两类，69 个 inactive waves 保留。
相同 1344 WMMA sequence 的 wave 跨度中值 24,040 → 26,800 cycles；1568 为 26,059.5 → 27,564 cycles。
主要增长落在 mainloop 和 activation/quant；barrier 的长 latency 也可能反映 peer 到达时间变化。
residual 用 event[3] 的 decoded latency，负值保留，不把 hidden wait 或间隙一律命名为 IFETCH stall。

本地 SQC MAS 文档的 virtual address/set hash 可以解释跨 64 KiB code 段时 group 0/2 与 group 1/3
相对 set 映射的差别，但具体 live set/bank 冲突仍未唯一确定。v45 text 为 44,160 B、VA 0x1a00–0xc680，
没有 v3 的 64 KiB 跨段，不能用 v3 规则解释 v45 原 118.372 μs 异常。
外层 rocprof 令 PyTorch profiler 出现约 0.8–0.9 μs 的失真值，全部不采用；ATT cycles 也不是新的 kernel μs。

本轮修改的是 `my_code/get_att_b_rocm_sdk.sh` 的多对象采集/验收功能，默认 code-objects=1 保留；
没有新生产 kernel 修改。pilot v1 cache/RUN_ONLY 失败、v2 单对象验收失败和 CPU packer v1 snapshot 缺失失败
均保持原状态；使用新 v2 archive 完成独立验收，不改旧失败记录。没有为 CPU 打包问题重跑 GPU。

详细表、复现入口、原始档案和独立报告见
[moe_gemm1_v3_variability_diagnosis_20261025.md](moe_gemm1_v3_variability_diagnosis_20261025.md) 的本日新增章节。
机制定位后继续优化：先用新 label 完成之前未通过的 v59 双侧性能对照，全部 pin=0，保留原 timer、有效计算及精度。
截至本节生产 v45 正式三轮仍是 80.9644 /81.3186 /118.372 μs、中值 81.3186 μs，异常第三轮保留；
本轮没有新 MoE e2e 成绩，fixed shared1/E64-E65 共用 compiled GEMM1 历史要求尚未完成。
本地和远端 decoder bug 文档各自追加、独立 CAS 和备份，保留远端较大的历史，绝不用本地小副本覆盖它。


### v59 补测未完成、v45 异步非法访问诊断与 v60 精度验证（2026-10-26）

v3 的 code 加载地址相关 I-cache miss 放大已有固定 host 的往返、counters 与 matched ATT 支持，
因此继续实际 kernel 优化。本节不将该机制推广到 v45 原 118.372 μs 异常，也不将它解释为非法访问原因。
生产仍保留 v45；正式 GEMM1 三轮 80.9644 /81.3186 /118.372 μs，中值 **81.3186 μs**，异常轮次不删除。
本节尚无新的正式 MoE e2e 性能。

#### v59 新 label 的固定对照再次未完成，不采用单侧小幅收益

使用 `f3_target_v59_vs_v45_comparison_20261026_v2`，只更新 label 和 child 的 118s/2s timeout，
原 helper 函数 AST、timer、warmup5/iters20、pin=0 及原精度标准保持。

| 组 | GEMM1 各轮 μs | GEMM1 中值 μs | GEMM2 中值 μs | 状态 |
|---|---|---:|---:|---|
| 前侧 v45 | 81.3268 /81.6158 /81.6432 | 81.6158 | 64.4977 | 三轮完成 |
| v59 | 81.4238 /81.1943 /80.2517 | 81.1943 | 64.1223 | 三轮完成 |
| 后侧 v45 | 第一轮没有最终原表 | — | — | 8.5259 秒返回 134 |

后侧 `g3_base_a1_r1` 日志在性能采集后、PyTorch reference 的 `nonzero/index_put_` 处报告
异步 `hipErrorIllegalAddress`。报错位置不能唯一定位 faulting kernel。
同期 dmesg 的 CPF fault（host PID1510324、17:20:36、地址0x1744000）尚未与该失败 child 的
container PID1920744 建立对应，不能归因；PID 查询未匹配也不能证明没有 GPU fault。
完整双侧对照未完成，不采用前侧单边 0.5164% 改善；旧 20261025/v1 失败状态保持。

该批证据包 `logs/f3_v59_incomplete_comparison_evidence_20261026_v2.tar.gz` 的 106 members 已逐项 SHA256 复核。
archive SHA256：`b08cc20ba5ba926dff3b20fabfb6048ac40c99ce7db39b3d69744086d5ded5ba`。
独立复核为 `logs/f3_v59_incomplete_comparison_independent_local_review_20261026_v2.json`。

#### v45 四项固定诊断均完成；未复现不等于已修复

| 模式 | returncode | child 秒数 | 数据口径 |
|---|---:|---:|---|
| `normal_markers` | 0 | 8.9273 | 诊断，耗时不采用 |
| `phase_fences` | 0 | 8.9274 | 诊断，耗时不采用 |
| `serialized3_markers` | 0 | 8.9272 | 诊断，耗时不采用 |
| `untimed_complete_moe` | 0 | 9.0774 | 诊断，耗时不采用 |

前三项分别保持正常异步调用、增加 phase fences，以及使用 `AMD_SERIALIZE_KERNEL=3`。
原 benchmark 文件未改，仅在 import 后包装 phase 标记和诊断同步，入口仍为原 `module.main()`。
phase journals 的 PID、单调时间、原 reference/perftest/fused 调用成对、原精度完成标记已独立复核。
完整 E64 MoE 输出为 22,020,096 B，与已验证 v45 **零字节差异**；
output SHA256：`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。
四项均 `performance_data=False`，诊断中的 μs 不作为新性能数据；不能宣称非法访问已修复。

#### v60：穿插 ScaleA/ScaleB 与 A/B payload 的 LDS 读取

仅在已有 compact/pin0/interleaved loader 内，将 ScaleA/ScaleB 读取分散到 A/B payload 读取间。
所有请求地址、数量、operand 结果及 surrounding fences 保持，GEMM math、输入 TDM/wait helpers、barrier、
原有效 row predicate、activation/quant 和精度标准不变；没有新增 expert-count specialization。
实际 loader AST 的 1,024 组 mock 比较确认请求和结果一致，guard 外及 pin1 的源码路径保持。

CPU prepare 15.1866 秒、inspect 0.164 秒，原 bounded DS/WMMA 模型经过 7,014 次 transfer 后 issues=[]。
原模型及 transfer AST 未放宽。资源为 798 VGPR /56 SGPR /327680 B LDS，private/spill 均 0；
DS/WMMA、activation/quant、TDM、barrier 静态计数保持。完整 ELF、逐 machine word、source IR、原 caller 签名已本地绑定。
GEMM1 ELF SHA256：`59e4bd1aea39005b5bb1a0aa7e36871c75816f4233a374705a71fa8c78cb8135`；`.text=44416 B`。
source SHA256：`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。

设备验证的 E64 seed0/17、E65 原 t64 routed fallback seed0/17 共四项完整 MoE 均通过原 gate，
每项 22,020,096 B 输出与已验证 v45 **零字节差异**。GEMM2 的完整 ELF 保持
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
E65 reference 复用已封存输出，不冒充本轮新执行。该 fallback 验证没有完成 fixed shared1 或
E64/E65 共用 compiled GEMM1 的历史要求。本节写入时 `v45→v60→v45` 各三轮正在执行，尚不采用 v60。

planar LDS 的有效数据为272 KiB，即使去掉48 KiB空隙，按 Shader Programming Guide §3.3.4 的64 KiB分区
仍需320 KiB；故没有推进仅去空隙来提高 occupancy/cache 的方案。静态布局推导不是性能测量。

实际入口（主机编排，编译/设备 child 均在 a07-3 /dsr1_ep4；旧 label 已存在，复跑必须用新 label）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v45_reference_fault_phase_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v60_prepare_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v60_guard_outside_e65_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v60_vs_v45_comparison_20261026_v1.py
```

v60 精度和 v45 phase 诊断联合证据包为 `logs/f3_v60_precision_and_v45_fault_evidence_20261026_v1.tar.gz`，
99 members 全部 SHA256 独立复核，已有本地文件保持；archive SHA256：`b5725eac04c3d40c9907c46bc1221d21d8229cd54c51262502cfb5ac54e56fe0`。
独立报告为 `logs/f3_v60_precision_independent_local_review_20261026_v2.json` 与
`logs/f3_v45_fault_phases_independent_local_review_20261026_v1.json`。
local reviewer v2 只纠正继承的 controller label、proof 路径及 bounded timeout 字面量，保留 v1 和全部原断言，未重跑 GPU。

以上 `logs/` 相对 `.codex_tmp/hyg_f3_followup_20261024/`。
全程只在 a07-3 测试、pin=0；新 child 用 `timeout --signal=TERM --kill-after=2s 118s`，总上限约120秒。
性能前全机三次连续 idle、间隔至少6秒；占用时自动等待，不杀外部任务，计时期间不传输/编译/额外检查 GPU。
70 μs 目标、fixed shared1 和共用 compiled GEMM1 均尚未完成。后续以文末性能结论为准。

### 2026-10-26：v60 已采用并正式复测，v61 回退，继续检查 v62

本节回答本轮“汇报下进展，然后继续”，并补齐前节之后完成的性能结果。
当前生产源码为 v60，pin=0；以下 workload 均为 E64/T1536/topk8/K7168/inter2048、
A4W4/SiLU、无 bias、random non-balanced，原单测参数 `uniform/auto/seed0`，
A-preshuffle=1、warmup5/iters20。没有改变原 timer 或精度标准。

#### v60 正反向对照与采用

v60 在 compact/pin0/interleaved loader 内穿插 ScaleA/ScaleB 与 A/B payload 的 LDS 读取。
既有地址、请求量、有效 row predicate、GEMM math、activation/quant、输入 TDM/wait 和 barrier 保持。
没有新增 expert-count specialization。E64 seed0/17 和 E65 原 t64 routed fallback seed0/17 的
四项完整 MoE 已通过原 gate；各 22,020,096 B 输出均与已验证参考零字节差异。

两批固定顺序、每组三轮的对照全部完成，18 轮全部有效：

| 批次和组顺序 | 版本 | GEMM1 三轮中值 μs |
|---|---|---:|
| 正向前侧 | v45 | 81.1163 |
| 正向中间 | v60 | 80.6778 |
| 正向后侧 | v45 | 80.8106 |
| 反向前侧 | v60 | 80.9327 |
| 反向中间 | v45 | 81.4708 |
| 反向后侧 | v60 | 80.8253 |

v60 相对四个相邻对照的改善分别为 0.5406%、0.1643%、0.6605%、0.7923%。
合并本次正反向中各版本的九轮原数据，中值为 v45 **81.1163 μs**、v60 **80.9139 μs**，
约改善 **0.25%**。这些是同批比较结果，不与下一节的独立正式组拼成新的性能对比。
改善幅度较小；不能据此宣称所有历史慢档或非法访问已消除。

两端采用 receipt 除 `side` 外一致，旧 v45 源码备份保留。
生产文件为 `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`：

- source SHA256：`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。
- GEMM1 ELF SHA256：`59e4bd1aea39005b5bb1a0aa7e36871c75816f4233a374705a71fa8c78cb8135`。
- GEMM2 ELF SHA256：`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

独立复核为 `logs/f3_v60_vs_v45_comparison_independent_local_review_20261026_v2.json`、
`logs/f3_v60_reverse_reference_independent_local_review_20261026_v1.json`。

#### v60 正式三轮：81.6604 μs

第一次正式组 `f3_target_v60_retained_pin0_formal_20261026_v1` 的第三轮结束后出现外部 owner
`1839129`，因此整组排除；原 GEMM1 三轮 **80.7284 /80.7873 /81.8247 μs** 全部保留，
未终止该外部任务。该任务来自另一个 repo 的 `run_original_source_fanout_global_v119.py`。
排除记录为 `logs/f3_v60_formal_ownership_exclusion_local_20261026_v1.json`。

随后完整重测 `f3_target_v60_retained_pin0_formal_20261026_v2`，三轮均通过原精度、完整 ELF、
运行前全机三次 idle 和运行后 ownership/settled idle 检查，没有排除任何一轮：

| 正式轮次 | GEMM1 μs |
|---|---:|
| r1 | 81.6604 |
| r2 | 81.6815 |
| r3 | 80.7704 |
| 中值 | **81.6604** |

GEMM2 中值为 64.5735 μs。当前距 70 μs 目标还差 **11.6604 μs**。
这是独立正式成绩；本轮没有新增 MoE e2e 耗时，完整 MoE 正确性通过不等于测得 e2e 性能。
本地独立复核为 `logs/f3_v60_retained_pin0_formal_local_review_20261026_v2.json`。
正式组与 v61 CPU 证据联合归档的 134 members 均已独立 SHA256 核对，archive SHA256 为
`b5ef2359262b6bc44e38861a56cbd9ad90dc355c57036cdf05f9127765de016f`。

#### v61：精度通过，双侧性能回退，不采用

v61 将 ScaleA/ScaleB 的读取移到对应 payload 之后。实际 source SHA256 为
`1026ec32f9f7e815d96fd2222411b4dba93c363b66e115c2831dae7107f9acaf`，GEMM1 ELF 为
`c01008636cd3cb678c9d02a270664dad43a9da0a922fe48049beabca5efac09b`。
native 的原 bounded DS/WMMA 模型经过 6,790 次 transfer，issues=[]；模型未放宽。
资源仍为 798 VGPR /56 SGPR /327680 B LDS，private/spill=0。
完整 ELF、逐 machine word、source IR 与原 caller 签名绑定通过。

E64 seed0/17、E65 原 fallback seed0/17 四项完整 MoE 均通过原 gate，
输出与已验证参考零字节差异，GEMM2 ELF 保持。
精度归档的 60 members 已独立核对，archive SHA256 为
`77acd87a2330fb143151b814fa0d769e208518f94212d97a3a2d7e12998cedc4`；
独立报告为 `logs/f3_v61_precision_independent_local_review_20261026_v2.json`。

固定 `v60 → v61 → v60` 的有效性能结果为：

| 顺序 | GEMM1 三轮中值 μs |
|---|---:|
| 前侧 v60 | 81.3355 |
| v61 | 81.4873 |
| 后侧 v60 | 81.0423 |

v61 相对两侧分别回退 **0.1866% /0.5491%**，没有取得双侧改善，所以不采用。
不追加同版反向测试或 ATT。两个因轮后 external ownership 而无效的候选组各完成一轮，
GEMM1 为 **81.3503 /82.0587 μs**；两组均整体排除并保留，未混入有效中值。
后侧 v60 的原三轮 **82.0397 /81.0423 /80.5701 μs** 全部保留。
controller 正常结束并恢复 v60。

性能证据归档 `logs/f3_v61_vs_v60_pin0_performance_evidence_20261026_v1.tar.gz` 的
226 members 全部逐项核对，既有独立文件保持；archive SHA256 为
`04b7b7ea206dd037862d6595806a26a9e3e66eab6ad2efc336e51775faaa3cb5`。
独立性能复核为 `logs/f3_v61_vs_v60_comparison_independent_local_review_20261026_v2.json`。

#### v62：编译和 native 通过，设备精度与性能尚待验证

v62 从已采用 v60 派生，只将每个穿插 payload pair 的请求顺序从 A→B 改为 B→A；
ScaleA/ScaleB 的位置保持 v60。实际 loader AST 的 1,024 组 mock 请求、结果和 barrier 一致，
14 个保护 helper 的 AST 保持；原有效 row predicate 和 pin1/guard 外行为保持。
source SHA256 为 `58d35238443896c775790ec60eb25394ddca892ff03a2d2ece398e6a28a1b140`。

原 caller 的 CPU prepare 15.3867 秒、inspect 0.1138 秒，GEMM1 ELF 为
`aa4fe8f42eda16fa4a1e270280fdf876394840f574c6fa34b232a2799eb33752`，与 v60 不同。
native 原模型检查通过、issues=[]，资源仍为 798 VGPR /56 SGPR /327680 B LDS，
private/spill=0。controller 已恢复 v60；尚未执行 v62 的设备精度或性能测试。
下一步先做完整 MoE 精度验证，通过后执行固定 `v60 → v62 → v60` 性能对照。

#### 复现入口和约束

以下命令在 a07-3 主机编排；compile/native/GPU child 在 `dsr1_ep4` 中执行。
已有 label 不得覆盖，重测需创建新 label 和私有 cache。

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v60_vs_v45_comparison_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v60_reverse_reference_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v60_retained_pin0_formal_20261026_v2.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v61_vs_v60_comparison_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v62_striped_bfirst_cpu_20261026_v1.py
```

v60 base preparation 曾出现两项 outer 工具错误：v1 错在 native 路径、没有启动 source swap/GPU；
v2 四个正常 child 全部成功，但 outer 错读 base 不具有的 `differing_bytes` 字段。
正常 preparation 的 completed=True 已保留，使用保存的完整输出与已验证 v60 输出做 CPU-only
零字节对比并核对 ELF，没有重跑成功的 GPU 验证，也没有改 gate。两份失败记录和 traceback 保留。
独立调和报告为 `logs/f3_v60_as_base_for_v61_preparation_reconciled_20261026_v2.json`。

本节 `logs/` 相对于 `.codex_tmp/hyg_f3_followup_20261024/`。
全程仅 a07-3、pin=0。每个新 child 使用 `timeout --signal=TERM --kill-after=2s 118s`，
总上限约120秒；多轮 controller 和 idle 等待不限两分钟。
性能前全机三次连续 idle、间隔至少6秒；占用时每6秒自动等待，不杀外部进程。
计时期间不传输、不远端编译、不额外查询 GPU，轮后核对 ownership 和测试进程退出。

E65 目前验证的仍为原 t64 routed fallback，不能宣称 fixed shared1 或 E64/E65 共用 compiled GEMM1
的历史要求已经完成。当前没有新 commit/push/pull；用户授权的后续提交和最终复测任务仍保留。
v3 波动的已有证据不被本次调度微调替代：同 ELF 不同加载实例、固定 host 往返、counters
与 matched ATT 支持慢档的 instruction 供给差异，但不能据此唯一确定硬件内部机制，
也不能解释 v45 的 118.372 μs 旧异常或宣称历史非法访问已修复。

## 2026-10-26 进展补充：v62 未采用，v63 性能对照失败后先诊断

问：汇报下进展，然后继续。

答：当前继续保留 **v60、pin=0**。独立正式三轮 GEMM1 为
**81.6604 /81.6815 /80.7704 μs**，中值 **81.6604 μs**；同组 GEMM2 中值
**64.5735 μs**。距离当前 70 μs 目标还差 **11.6604 μs**。
这仍是前一轮完成的有效性能结果，本次诊断没有产生新的正式性能或 MoE e2e 性能。

本地仓库为 `C:/Users/yanguahe/Documents/code/wk_sp1/aiter_dsr1_e65`，远端为
`a07-3:/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`，容器 `dsr1_ep4`，
分支 `hyg_dsr1_conc6144`，HEAD `6f656cf5b02b2f1acf723949652f91b48e2933d7`。
后文 `TASK` 指 `.codex_tmp/hyg_f3_followup_20261024`，`NORMAL` 指
`.codex_tmp/yadai_gemm1_param_20261021`。

### v62：完整精度通过，但双侧性能没有稳定收益

v62 从 v60 派生，仅把每个穿插 payload pair 的 A→B 请求顺序改为 B→A，
ScaleA/ScaleB 位置、地址、请求量、数学、有效 row predicate、TDM/wait/barrier 保持。
CPU prepare、native bounded DS/WMMA、完整 ELF/ISA/IR 和原 caller 绑定通过。
E64 seed0/17、E65 原 fallback seed0/17 四项完整 MoE 通过原精度 gate；
每项 22,020,096 B 输出与已验证参考零字节差异，GEMM2 ELF 保持。

固定 `v60 → v62 → v60` 各三轮的结果为：

| 顺序 | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---:|---:|
| 前侧 v60 | 81.1162 | 64.5485 |
| v62 | 81.0867 | 64.6178 |
| 后侧 v60 | 81.0128 | 64.4976 |

v62 相对前侧改善 **0.0363676%**，相对后侧回退 **0.0912202%**。
九轮全部有效，没有排除组；没有取得双侧改善，所以未采用，不追加同版反向或 ATT。
两份归档均已核对全部 members：

- `TASK/logs/f3_v62_E64_E65_precision_evidence_20261026_v1.tar.gz`：60 members，
  SHA256 `4e1d2bd6c4e95cb7157e006ae0fae5037472756d0e38ca561adee32fe42975d4`。
- `TASK/logs/f3_v62_vs_v60_pin0_performance_evidence_20261026_v1.tar.gz`：192 members，
  SHA256 `e915aaa9238109ac8c40504e711db8ab177454a6d9dab7df57889036c7813f81`。

### v63：合并 ScaleA byte store，精度通过，性能对照第三轮非法访问

v63 在 v60 上，把相邻两个 ScaleA byte store 合并为一次对齐的 i16 store。
guard 为 compact/pin0、FP4 quant、`N_MX_BLKS == 2`；必须保留
`offset_is_bytes=True`。量化值、输出布局、loader、数学和同步保持。
原 byte/address 检查覆盖 65,536 组 byte pair 和 417,792 组地址，两个 byte
同属一个有效 dword 或同为 OOB。CPU prepare 15.2870 秒、inspect 0.1138 秒；
native 原 bounded DS/WMMA 模型检查通过，6,974 次 transfer，`issues=[]`。

GEMM1 为 6,210 条 executable instructions，798 VGPR /56 SGPR /327,680 B LDS，
private/spill=0。ISA 中 16 条 `buffer_store_b8` 改为 8 条 `buffer_store_b16`，
受保护的数学、LDS/TDM 和 barrier 数量保持。完整 ELF、逐 machine word、
source IR 和原 caller 参数、globals 签名绑定通过。

- source SHA256：`ca1f2ee865e62d0d7a0624afc58a8c05e716ae3e228b2285fbe0a44312bf7ed3`。
- GEMM1 ELF SHA256：`a0fbb0b4739a2690aeb211cff5783caac72d8e648842501226d64dfc918c5ec5`。
- source IR SHA256：`ce24579cb2eb88de514468c6d0df462a0226ace3790380c5b3ce0c252d3b8a54`。

E64 seed0/17、E65 原 fallback seed0/17 四项完整 MoE 通过原 gate，
每项 22,020,096 B 输出与已验证参考零字节差异，GEMM2 ELF 保持。
精度证据 `TASK/logs/f3_v63_E64_E65_precision_evidence_20261026_v1.tar.gz`
的 60 members 已独立核对，SHA256 为
`9769380737b874aa3083ebd523ad6cf55c4b26690e57d45ca0425e6ae269bc67`。

**上述 native 模型不完整覆盖 VALU/VMEM/TDM completion；精度验证通过不能覆盖后续异步错误。**

固定性能对照的前侧 v60 三轮完成，GEMM1 /GEMM2 中值为
**81.3439 /64.6768 μs**。候选第一、第二轮完成，原读数分别为
**80.4646 /64.1329 μs** 和 **81.291 /64.4809 μs**；第三轮在 8.5768 秒内退出，
returncode=134，出现 `c10::AcceleratorError` /`hipErrorIllegalAddress`。
错误显示在 GEMM2 profiling 后的 PyTorch reference 路径，stack 含
`nonzero`、`_index_put_impl_`，但异步报错不能据此唯一归因 GEMM1、GEMM2 或 reference。

outer 为 `completed=False, source_restored=True`，后侧 v60 未运行。
**整组性能对照未完成，不用候选局部读数宣布收益，不补跑替换失败轮，不采用 v63。**
原失败 label 为
`f3_target_v63_vs_v60_comparison_20261026_v1_g2_scale16_striped_pin0_a1_r3`。
原 `.manifest.json/.log/.exit/_pre.log` 与失败 summary 全部保留。

### 四阶段诊断已经完成，本轮没有复现，不能宣称修复

启动 `TASK/run_f3_v63_fault_phase_20261026_v1.py`，固定四个模式，各为新 label、
独立 child，复用原 benchmark 和已准备的 v63 私有正常 cache。
所有诊断均为 `performance_data=False`，其中打印的计时不作为性能数据。

| 模式 | 检查内容 | 结果 |
|---|---|---|
| `normal_markers` | 原 CLI、fused MoE、三项 perftest、reference 阶段 marker | returncode=0，原精度 gate 通过 |
| `phase_fences` | 每个阶段返回后同步，reference 前后同步 | returncode=0，所有同步点和原精度 gate 通过 |
| `serialized3_markers` | `AMD_SERIALIZE_KERNEL=3`，保留阶段 marker | returncode=0，原精度 gate 通过 |
| `untimed_complete_moe` | 完整 MoE 对比已验证 v60 seed0 输出 | returncode=0，原 gate 通过，零字节差异 |

四个 child 分别在约 8.83 /8.88 /8.88 /8.93 秒内完成，没有超时。
完整 MoE `logits_diff=3.6105513278025183e-06`，`rel_l2=0.0026872099842876196`，
输出 22,020,096 B，与参考 SHA256 均为
`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`。

每项启动前执行三次连续全机 idle gate、间隔至少6秒。第二项启动前检测到另一个 repo 的
GPU 任务，因此自动每6秒等待，未干预该任务。各项结束后的 ownership /settled ownership
均为空，最后检查无本轮 GPU child 残留。outer 正常完成并恢复 v60。

本轮未复现错误不等于已经修复。尚未建立原错误的完整阶段归属，继续诊断期间不开展新的
kernel 性能优化。复用 child 文件名和部分日志 marker 中的 `V45` 是有意复用；
新 label、outer、实际 source 和 cache 均为 v63，不将历史 v45 当作本轮版本。

### 失败组 cache 后补审计和证据归档

四阶段诊断结束后，使用 `TASK/run_f3_v63_failed_cache_audit_20261026_v1.py`
临时 CAS 切换远端源码到 v63，在容器中运行原 CPU-only audit，使用 g2 group 的原 key inventory，
完成后恢复 v60。**这是 postmortem cache 检查，不能补回原失败轮缺失的 contemporaneous
dispatch 审计，不能据此宣称已经证明原第三轮执行了哪个 ELF。**

保存的 GEMM1 ELF 与 v63 准备阶段一致；GEMM2 ELF 为
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`，保持原版本。
原 caller key、argument/globals 签名和 IR/ELF 检查通过。原失败 summary 和 group summary
逐字节保持，没有把后补记录写成原 `_r3.cache_acceptance.json`。

保留的 dmesg 窄窗口有 26 条 amdgpu 记录：2026-10-25 19:27:50 UTC，
`[gfxhub0] no-retry page fault`，host pid2254359 /pasid44488，
pages `0x000000000054e000`、`0x0000000005da0000`，client `TCP (0x8)`，`RW=0x0`。
原 profiler container PID 为1996883；尚未建立严格的 host/container PID 对应，
不能仅凭时间把上述 fault 唯一归属本 child，也不能由低地址或 RW 字段猜定根因。

失败证据归档为
`TASK/logs/f3_v63_failed_comparison_and_phase_evidence_20261026_v1.tar.gz`，
SHA256 `e9ca906966d49512973aec1171750bdaefc368b61866baf61d85da320b6cde8a`。
165 members 全部核对，111 个原本缺失的文件安全导入；既有文件必须逐字节相同。
独立复核同时检查原失败日志、退出码、三轮 v60、候选两轮、四项 phase journal、
原精度日志、source 恢复和保存 ELF 的 `.text`，报告为
`TASK/logs/f3_v63_failure_and_phase_independent_local_review_20261026_v1.json`。

以下在 a07-3 主机编排；kernel/GPU 和 artifact audit child 在 `dsr1_ep4` 执行。
已有 label 不能覆盖，本轮命令已完成：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v63_fault_phase_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v63_failed_cache_audit_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/pack_f3_v63_failed_comparison_20261026_v1.py
```

### 当前保留状态和下一步

两端生产仍保留 v60，source SHA256 为
`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。
当前没有创建 v64，没有新的正式 MoE e2e 数据，也没有新的 commit/push/pull。
此前已授权的英文 commit、push、远端 pull 和最终复测任务仍保留。

下一步先独立重复 PyTorch reference，再在完整流程中为 reference 增加 ATen 操作和同步边界记录。
这些属于诊断运行，保持 `performance_data=False`；即使全部通过也只记录未复现，不宣称修复。
本次 checkpoint 时这两项进一步诊断尚未启动。

所有新 test/compiler/profiler child 继续用
`timeout --signal=TERM --kill-after=2s 118s`，总上限约120秒；
多轮 controller 和 idle 等待不限两分钟。仅使用 a07-3，统一 pin=0，
计时期间不传输、不远端编译、不额外查询 GPU，不杀外部任务，不清理或丢弃仓库修改。

E65 当前验证的仍为原 t64 routed fallback；**fixed shared1 和 E64/E65 共用 compiled GEMM1
的历史要求尚未完成**，不能误报为通过。本轮不改变原功能、有效计算、精度 gate，
不新增 expert-count specialization。

v3 先前波动定位已有同 ELF 不同加载实例、固定 host 往返、counters 和 matched ATT 的支持。
本轮诊断不替代该证据，也不能用其机制解释 v45 的旧118.372 μs异常或宣称历史非法访问已修复。


## 2026-10-26 进展：reference 隔离及原异步九轮诊断完成

本节回答“汇报下进展，然后继续”，补齐上一 checkpoint 后完成的诊断。
当前生产最优可用版本仍为 v60、`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`。
E64/T1536/topk8/K7168/inter2048、random non-balanced、`uniform/auto/seed0` 的正式
GEMM1 三轮中值仍为 **81.6604 μs**，距70 μs差 **11.6604 μs**；同组 GEMM2 中值64.5735 μs。
本轮没有新增正式性能或 MoE e2e 耗时，诊断中打印的 μs 全部不参与性能比较。

### 已完成的进一步诊断

| 诊断 | 固定计划与范围 | 结果 |
|---|---|---|
| PyTorch reference-only | seed0/17各16次；复用原 initializer AST前缀及 `_torch_moe_ref`；不运行 grouped MoE | 两个 child均returncode=0；32次reference全部通过，每个seed的16个输出hash一致 |
| 完整流程 ATen 逐项同步 | `front_v60 → v63 → back_v60`，每组4641个reference ATen操作逐项同步 | 三个 child均returncode=0；原精度gate通过，fused及reference输入指纹保持 |
| 原异步路径轻量观察 | `front_v60_r1/r2/r3 → v63_r1/r2/r3 → back_v60_r1/r2/r3` | 九个 child全部returncode=0；原精度gate通过，未复现非法访问 |

轻量观察复用原 benchmark和原phase observer，仅增加CPU helper marker及tensor地址metadata。
没有增加GPU操作、输入hash/复制或额外同步；`phase_fences=False`。
每轮原reference helper调用为`per_1x32_f4_quant`两次、`torch_moe_stage1`一次、
`torch_moe_stage2`一次，完整原CLI最终precision summary通过。
逐项同步诊断会改变执行时序，其通过不证明原异步路径安全；九轮轻量观察通过也只表示本轮未复现。

所有诊断均明确`performance_data=False`。九轮轻量child分别约8.78–9.04秒完成，
没有超时；每个test/compiler/profiler child继续使用118秒TERM、2秒后KILL的上限。
固定九轮没有替换原失败轮、没有补齐失败性能组，没有依据读数重跑。

### 进程、源码和证据边界

新controller在child运行时通过`/proc`记录精确host/container `NSpid`映射，
没有执行运行中GPU查询。九个对应host PID均已退出，各轮ownership和settled ownership均为空，
6秒后settled busy均为0，生产源码已恢复为v60。
四轮的即时post snapshot仍报告busy=1，原值保留，不写成“所有即时post均为0”；
这些运行始终只用于诊断，不据此补认性能数据。
对九个精确host PID检查保存的dmesg，没有匹配的child PID记录。
这不证明系统从未发生其他GPU fault，也不能补回原失败轮缺失的PID对应。

原v63性能组第三轮returncode134及`hipErrorIllegalAddress`保持原样。
目前仍不能唯一确定原faulting kernel或根因，**不宣称修复，不采用v63**。
两端v60 source SHA256仍为
`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。
原v3加载地址相关I-cache miss证据与这次非法访问属于不同问题，不能混用解释。

证据以`.codex_tmp/hyg_f3_followup_20261024/`为根目录：

- `logs/f3_reference_only_and_ops_fault_evidence_20261026_v1.tar.gz`：48个members，SHA256
  `690303f49df33965734419e29e9db1d92c7990c4336c3f1d232d50918331a385`。
- `logs/f3_light_reference_fault_evidence_20261026_v1.tar.gz`：102个members，SHA256
  `528b174a942ab53274b59a8fcf02b941a063899527931570347483bda0a45b1e`。
- `logs/f3_reference_fault_detail_independent_local_review_20261026_v1.json`：本地独立复核
  原日志、退出码、32次hash、三组输入指纹、九轮phase/helper/精度、PID及源码恢复，结果通过。

两份archive的全部member均逐SHA256核对后导入，既有本地文件保持。
轻量archive packer首版错误要求所有即时post busy为0，因此在归档前停止；v2仅修正
诊断证据打包的这项假设，保留busy原值并要求settled idle，没有修改原controller或性能门禁，
没有因此重跑GPU测试。

### 继续工作的范围

v63保持隔离。下一步只从保留v60做静态改进检查：研究在现有输入pipeline中使用单个
mainloop跳过最后一对完全无效的M reps，同时避免此前多份mainloop造成的代码膨胀。
先检查历史、地址/数学等价和原native模型；没有通过这些门槛时不运行候选GPU kernel。
不新增persistent、w2x4、pipeline或expert-count specialization，不降低通用性及精度。

E65目前验证的仍为原t64 routed fallback；**fixed shared1和E64/E65共用compiled GEMM1的历史要求尚未完成**。
本轮没有改变这些路径或把fallback验证说成共用kernel验证。
英文commit、push、a07-3主机pull及最终复测的既有授权仍保留，本checkpoint尚未执行新提交。

已完成复现入口如下（在a07-3主机编排，GPU及artifact audit child在`dsr1_ep4`）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_reference_only_fault_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_reference_ops_fault_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_light_reference_fault_20261026_v1.py
```

这些label均已使用；重测须另建label，保留本轮全部记录。

## 2026-10-26：v65 精度通过，性能组因非法访问停止；六轮启动身份诊断完成

本节补齐 v64/v65 的静态检查、设备精度、失败性能组及后续六轮诊断记录。
`TASK=.codex_tmp/hyg_f3_followup_20261024`，
`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。仍只在 a07-3、
`/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`、`dsr1_ep4` 中测试。
两端分支为 `hyg_dsr1_conc6144`，HEAD 为
`6f656cf5b02b2f1acf723949652f91b48e2933d7`。

### 当前采用状态和进度答复

问：汇报下进展，然后继续。

答：正式采用的仍为 v60，`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`。
E64/T1536/topk8/model_dim7168/inter_dim2048、A4W4/SiLU、无 bias、
`uniform/auto/seed0`、random non-balanced、A-preshuffle=1、warmup5/iters20 下，
此前正式三轮 GEMM1 为 **81.6604 /81.6815 /80.7704 μs**，中值 **81.6604 μs**；
同组 GEMM2 中值 **64.5735 μs**，距离当前 70 μs 目标还差 **11.6604 μs**。
本节没有新的正式 MoE e2e 耗时，没有采用 v65，也没有新增 commit/push/pull。

v65 的 E64 和 E65 原 routed fallback 均通过完整 MoE 精度检查，与对应已验证参考输出
零字节差异；正式性能组却在候选第二轮报 `hipErrorIllegalAddress`，因此整组未完成。
后续 v60/v65 各三轮、只增加启动身份记录的诊断全部通过，六个 child 均已退出。
这些成功诊断不证明问题已修复，不使用其中打印的 μs 作性能成绩。
下一步使用 PyTorch 附带的同一套 ROCm SDK 做 API/dispatch trace，定位非法访问归属。

生产 source SHA256 保持
`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`；
GEMM1 ELF SHA256 为
`59e4bd1aea39005b5bb1a0aa7e36871c75816f4233a374705a71fa8c78cb8135`；
GEMM2 ELF SHA256 为
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
E65 本轮验证的是原 t64 routed fallback；fixed shared1 与 E64/E65 共用 compiled
GEMM1 的历史要求仍未完成，不能将 fallback 验证表述为这些要求已经满足。

### v64/v65：单 mainloop 跳过无效 M reps 的检查

v64 尝试在单 mainloop 中跳过最后一对无效 M reps。原 native 模型报告
40 处 `DS overwrite before WMMA retirement`，未运行该候选 GPU kernel。
wrapper 末尾还出现 `KeyError: 'branches'`；原失败记录保持，没有改动模型使候选通过。

v65 在 v64 上扩大 loader 入口/出口两处 compiler boundary 的保护条件：

```python
... and (final_drain_load or interleaved_m_rows)
```

保留原 final-drain 保护，没有在源码中加入新的硬件 wait。CPU、原 native 模型、完整
ELF/ISA/IR/caller 绑定检查通过：9597 次 CFG transfer，`issues=[]`；
798 VGPR、57 SGPR、327680 B LDS、private/spills=0；6127 条 static executable
instructions，`.text=44160 B`。

- snapshot：`TASK/snapshots/f3_v65_single_back_striped_boundary_pin0_v1.py`
- source SHA256：`5085d9a094618a7fdb57f682e045d6ec9e68a42a7291777d872a697eee139acf`
- ELF SHA256：`f4c79e72a8ad3372660aa273f2c16734554a223066cea67c573956cee44e5bf0`

独立 ISA 复核发现 **15 个静态分支位置，每处可跳过 8 条 WMMA**，分支体中没有
DS/TDM/barrier。旧 native metadata collector 只统计 16 条 WMMA 的跳过块，
所以 `observed_pure_16_WMMA_skip_sites=[]` 不表示没有分支。原 hazard 模型仍遍历
完整 CFG，模型与原 JSON 均未修改；静态分支存在也不能当作设备实际跳过次数。
复核文件为
`TASK/logs/f3_v65_failure_and_actual_skip_independent_local_review_20261026_v1.json`。

### 四项完整 MoE 精度验证

v65 在 E64/seed0、E64/seed17、E65 原 routed fallback/seed0、E65 原 routed
fallback/seed17 下均通过原精度 gate，输出与对应参考零字节差异。
每项输出为 22020096 B，GEMM2 ELF 保持原版本。结束后恢复 v60。

以下主机 controller 命令已经执行完毕，不覆盖既有 label：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v65_prepare_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v65_guard_outside_e65_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/pack_f3_v65_precision_20261026_v1.py
```

精度 archive 为 `TASK/logs/f3_v65_E64_E65_precision_evidence_20261026_v1.tar.gz`，
58 members、984074 B，SHA256
`ddfe5d4e076e68528696a293d454709c02a23ce0f954dc15c4584eee86a29d14`。
下载后逐 member SHA256 核验并导入；独立报告为
`TASK/logs/f3_v65_precision_independent_local_review_20261026_v1.json`。

### 原性能对照组未完成，候选没有采用

`run_f3_v65_vs_v60_comparison_20261026_v1.py` 的固定计划为
`v60 → v65 → v60`，每组三轮，使用原 CLI/profiler timer。
前侧 v60 三轮完成，GEMM1/GEMM2 中值为 **80.3362 /64.8962 μs**。
v65 第一轮原记录为 **84.8481 /65.5624 μs**，仅作为未完成组的原始记录保留，
不用于采用决策。第二轮 **8.426660976954736 秒返回134**，出现
`hipErrorIllegalAddress`，没有最终精度表；后侧 v60 没有运行。

失败 label：
`f3_target_v65_vs_v60_comparison_20261026_v1_g2_single_back_striped_boundary_pin0_a1_r2`。
outer 为 `completed=False, source_restored=True`。没有补跑替换失败轮。
失败轮没有 contemporaneous ELF audit，后补 cache 检查不能证明原失败 dispatch 的 ELF。
栈包含 `SetDevice`、`at::_ops::nonzero::call`、`at::_ops::index_put_::call`；
异步错误的报告位置不能唯一定位 faulting kernel。

保存的主机 dmesg 在 UTC 2026-10-25 20:48:29.947418–20:48:30.906045 窗口出现
gfxhub0 no-retry page faults，host pid2668635/pasid45035，涉及 TCP/CPF 和多个低地址页面。
原 Python 日志的 container pid 为2020060；缺少该失败轮严格的 host/container PID 对应，
不能只凭时间将 fault 唯一归属该 child，也不能推断根因。

失败 archive 为 `TASK/logs/f3_v65_failed_comparison_evidence_20261026_v1.tar.gz`，
125 members、1434337 B，SHA256
`f6bc0d1525160e2af02280b935156a124d1ba883d2665bcfb2cc9cf969b29c35`。
全部核验并导入，原失败记录保持。

### 六轮原 CLI 诊断：启动身份可追溯，未复现错误

固定计划 `v60_r1 → v65_r1 → v60_r2 → v65_r2 → v60_r3 → v65_r3` 已完成。
child 用 `runpy.run_path` 执行原未修改 CLI，仅在 GPU 初始化前写启动身份；没有
helper/ATen wrappers、额外 GPU 操作、额外同步或 interphase marker。
controller 在运行中只读 `/proc` 捕获 `NSpid`，不查询 GPU。

| 版本/轮次 | host PID | container PID | 原 CLI 结果 |
|---|---:|---:|---|
| v60/r1 | 2725052 | 2021943 | returncode=0，原精度通过 |
| v65/r1 | 2738847 | 2026700 | returncode=0，原精度通过 |
| v60/r2 | 2741799 | 2027073 | returncode=0，原精度通过 |
| v65/r2 | 2747764 | 2027446 | returncode=0，原精度通过 |
| v60/r3 | 2751649 | 2027819 | returncode=0，原精度通过 |
| v65/r3 | 2754758 | 2028192 | returncode=0，原精度通过 |

六项均在约8.78–9.04秒完成，没有超时。独立本地复核逐项检查原 precision table、
startup journal、`NSpid`、日志 SHA256 和 source 恢复；六个精确 host PID 在 postmortem
时均已退出。同一 boot 的 dmesg 中没有这些精确 PID 的 GPU fault 记录，覆盖本组六轮的
窄时间窗口也没有 amdgpu/KFD 记录。未复现不等于已修复。

即时 post 的 busy 原值依次为 `0/1/2/1/0/1`，settled busy 均为0。
v65/r3 的即时 post 出现外部 owner2755045，来自另一个 repo 的测试，6秒后已退出；
该原记录保持，没有清掉外部进程，也没有把这些诊断耗时改成正式性能数据。

已执行的主机入口：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v65_uninstrumented_reference_fault_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/pack_f3_v65_uninstrumented_fault_20261026_v1.py
```

archive 为 `TASK/logs/f3_v65_uninstrumented_fault_evidence_20261026_v1.tar.gz`，
46 members、13115 B，SHA256
`c014bc84c9718f7b6d6f2e9bfc5032139cb0f56d35149c4531ddf4f1e3ce83d5`。
逐 member 核验并导入，独立报告为
`TASK/logs/f3_v65_uninstrumented_fault_independent_local_review_20261026_v1.json`。

### 后续定位范围

API/dispatch trace 使用 `_rocm_sdk_core/bin/rocprofv3` 与配套 SDK libraries。
本节写入时只完成 controller 静态检查，尚未启动该 trace，不能写成已经采集或定位。
仍使用 pin=0 和原精度标准，各 test/compiler/profiler child 为
`timeout --signal=TERM --kill-after=2s 118s ...`，总上限约120秒。
继续保存失败记录，保留 v60，v63/v65 隔离，不新增 GPU 性能候选。
原 v3 的 code 加载地址相关 I-cache miss 放大证据与本次非法访问是两个问题，不能混用。
目前没有发现或修复确定的输入生命周期 bug、缺失 drain 或其他确定根因。


## 2026-10-26：SDK/API trace 与 v60/v65 direct CLI ATT 验收

本节补齐前节之后完成的 SDK runtime trace、runtime 源码检查，以及两版 direct CLI ATT。
`TASK=.codex_tmp/hyg_f3_followup_20261024`，
`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。全部 GPU child 只在 a07-3 的
`dsr1_ep4` 中运行；Git/controller 在主机。pin 始终为0。

### 当前进度答复与采用状态

问：汇报下进展，然后继续。

答：当前生产仍为 v60，正式 GEMM1 三轮为81.6604 /81.6815 /80.7704 μs，
中值 **81.6604 μs**，距离70 μs还差 **11.6604 μs**。同组 GEMM2 中值64.5735 μs。
本节没有新增正式性能、MoE e2e 耗时或 commit/push/pull，没有采用 v65。

v60 与 v65 的 SIMD0、SE0–3 ATT 均已生成，目标 code object 已绑定各自冻结 ELF，
decoded PC/ISA 与 ELF 内指令字节逐项核验。v65 的无效 M reps 跳过在设备上确实执行，
但 payload DS reads 没有减少。本节继续做该方向的依赖与输入读取静态分析，
没有因一次成功采集而宣称原非法访问已经修复。

E65 当前验证范围仍是原 routed fallback；固定 shared1 路由与 E64/E65 共用 compiled
GEMM1 的历史要求仍未完成。本节没有改变这些支持范围或精度口径。

### SDK API/kernel trace：正常 CSV 与退出阶段失败均保留

使用 PyTorch 附带的 `_rocm_sdk_core/bin/rocprofv3` 及配套 libraries，没有升级工具链。

| 入口/版本 | host PID /container PID | 结果 |
|---|---|---|
| `run_f3_v65_SDK_runtime_dispatch_trace_20261026_v1.py` 首项 v60 | 2820463 /2031072 | 原 CLI 完整精度通过并打印 COMPLETE；随后 profiler 输出阶段 SIGSEGV，returncode139，约12.035秒 |
| `run_f3_v65_SDK_runtime_dispatch_csv_20261026_v2.py` v60 | 2831571 /2031553 | returncode0，完整精度通过，约11.285秒 |
| 同一 CSV 入口 v65 | 2837798 /2032113 | returncode0，完整精度通过，约11.287秒 |

第一项不是原 `hipErrorIllegalAddress` 的复现。它的 CSV 完整，但 JSON 为155525120 B，
尾部截断，解析失败位置155525111；原文件保留，没有补尾、修补或替换。
后两项使用新 label，只输出 CSV；成功不能据此宣称 profiler 退出问题已经修复。

每项有6834次 kernel dispatch、178608条 HIP API记录，GEMM1/GEMM2各47次。
六次 `hipModuleUnload` 与六次 `hsa_executable_destroy` 均在最后一次 kernel 结束之后。
94次 GEMM dispatch 均没有匹配的 HIP API correlation，当前 API trace 因而不能唯一绑定
cluster launch，也不能据此唯一确定原 faulting dispatch。

同一 boot 中三个精确 host PID 均已退出；相应 PID 与窄时间窗口没有 GPU fault 记录。
所有这些 trace 均为 `performance_data=False`，原 profiler 输出的失真耗时继续排除。

已核验 archive：`TASK/logs/f3_v65_SDK_runtime_trace_evidence_20261026_v1.tar.gz`，
44 members、11728693 B，SHA256
`5e21e1a8e5008840c2dad2b693387499e78a702895aea3b54efbf2b96343df60`。
独立报告：`TASK/logs/f3_SDK_runtime_traces_independent_local_review_20261026_v1.json`。
本地日志 alias 中的 trace CSV 保留 `original_source_path`，不把导入后的路径当成远端原路径。

### 安装中的 FlyDSL runtime 检查

已保存实际安装的 `jit_executor.py`、`jit_function.py`、`jit_argument.py`，
前两项与之前 pin 的 SHA256 一致：

```text
jit_executor.py  2e82d9faf3d55714d651aec97f2be9f9a7372b32f1216aa78b731aaac799463e
jit_function.py  f677c8f424ee2698a4c655fa8b7bc5a432c2974c62226a88e619f1cb80e07ed5
```

`CompiledArtifact` 保留 `_ctx`，`CallState` 使用 thread-local packed storage。
目前没有发现或修复确定的 runtime 生命周期 bug；不能把这次检查写成生命周期问题已修复。
记录与原文件位于 `TASK/logs/f3_installed_runtime_snapshot_20261026_v1.json` 及同前缀文本文件。

### v60 已成功采集；旧 controller 的身份验收失败保留

原入口 `run_f3_v60_v65_GEMM1_ATT_diagnostic_20261026_v1.py` 计划 v60/v65 两项，
第一项 v60 的采集 returncode0、elapsed30.079241867992096秒。
raw ATT、decoded waves、目标 code.json 均完整，原精度为
`logits_diff=3.61055e-06`、`rel_l2=0.00268721`、pass=True。

该 controller 随后因 `AssertionError('Missing direct CLI identities')` 停止，
identity 列表为空，v65 未启动。主机用户能读容器 root 进程的 `cmdline/status`，
但读 `environ/cwd` 返回 `PermissionError`；旧观察器跳过该异常，因而没有记录身份。
这是诊断观察器权限问题，不能写成 kernel 采集失败或 GEMM1 故障根因。

没有为了补 metadata 重跑 v60，也没有事后构造 host PID 映射。
原 summary 中空 identity 列表对应的 `all(...)` 为真，不作为已证明具体 PID 退出的依据。
preflight/container PID2044192、ATT/container PID2044437 来自各自完整 CSV，
仅记录容器 PID；post/settled 的 KFD owners 为空、busy0、VRAM176582656 B。
源码已恢复 v60。

### v65 新 label：只读 root 观察器与采集均通过

`observe_f3_ATT_direct_cli_identity_20261026_v2.py` 只读 `/proc`，
用 `sudo -n python3` 获得读取容器 root 进程 `environ/cwd` 的权限。
它只匹配已知原 benchmark argv 与唯一 task label，不输出其他环境变量，
核验 cwd、容器 PID namespace、`NSpid`、numeric UID 和 process start ticks，
没有 NSS 或 GPU 查询。原 benchmark 没有 runpy/helper wrapper、额外 GPU 操作或同步。

只运行尚未采集的 v65：`run_f3_v65_GEMM1_ATT_direct_identity_20261026_v2.py`。
returncode0、elapsed29.827714019978885秒，身份验收通过：

| 阶段 | host PID | container PID |
|---|---:|---:|
| kernel-trace preflight | 2974979 | 2046147 |
| SIMD0 ATT | 2977519 | 2046392 |

两个精确 host PID 已退出，post/settled KFD owners 均为空、busy0、VRAM176582656 B。
同 boot 的精确 PID 日志保留了 CPU `x86/split lock detection` bus lock trap，
未发现对应 GPU fault；覆盖采集的 uptime 窗口也没有 GPU 日志。
没有把 CPU trap 混写成 GPU illegal address，原记录见
`TASK/logs/f3_v65_GEMM1_ATT_direct_identity_20261026_v2.dmesg_exact_PID_review.json`。
v65 原精度同样为 `3.61055e-06 /0.00268721 /True`；结束后恢复 v60。

### 两版实际对象与动态执行证据

两次使用同一用户脚本 `my_code/get_att_b_rocm_sdk.sh`，SHA256
`3ab6143824ace8df39ceab0a96fcf3c7d3fd3f8729e9b196880d147d82933de4`。
采集 SIMD0、SE mask0xf、kernel iteration28、code-objects1，并保留完整 kernel trace。
目标为第28次 GEMM1 dispatch，SDK Dispatch_Id562，code object_id21。

| 版本 | frozen/实际目标 ELF SHA256 | decoded active waves |
|---|---|---|
| v60 | `59e4bd1aea39005b5bb1a0aa7e36871c75816f4233a374705a71fa8c78cb8135` | 16个，每个1792条WMMA |
| v65 | `f4c79e72a8ad3372660aa273f2c16734554a223066cea67c573956cee44e5bf0` | 13个1344条WMMA，3个1792条WMMA |

两版各有四个非空 raw SE0–3 ATT，decoded PC/ISA 全量一致，ISA 的每条机器指令字节与
冻结 ELF 的 `.text` 一致，没有 `.long` 或 `v_illegal`。
v60 的17个、v65 的25个 inactive waves 保留，未当成有效计算 wave。

以下仅为采样 active waves 的动态可见指令数与 cycles，**不是新 kernel μs**：

| 采样类别 | wave数 | 每wave WMMA | 每wave `ds_load_b128` | 每wave可见 `s_cbranch_scc1` | mainloop中值cycles | 有效跨度中值cycles |
|---|---:|---:|---:|---:|---:|---:|
| v60 | 16 | 1792 | 1792 | 59 | 18291.5 | 26404.5 |
| v65 跳过尾行 | 13 | 1344 | 1792 | 92 | 18915 | 25887 |
| v65 完整行 | 3 | 1792 | 1792 | 92 | 22269 | 30090 |

v65 部分 waves 少执行448条WMMA，即25%，但 payload DS reads 保留了全量。
因此当前候选实现了数学上无效 WMMA 的跳过，没有实现无效 payload 读取的跳过。
两版每个 active wave 可见 `tensor_load_to_lds` 为28，`s_wait_tensorcnt`、
`s_barrier_wait` 均为30，输入 TDM 与 barrier 的可见次数保持。
ATT 不记录所有 control 指令，不能由可见 `s_wait_alu` 数量为0推断机器指令被删除。

两次仅采 SIMD0，未绑定为同一批 workgroups，不能由上表计算正式性能提升/回退百分比，
不能把累计 wave phase 当成 dispatch 关键路径，不能将 timestamp gap 全部归因于前一条指令。
它们没有提供候选取得稳定收益的证据；原未完成性能组与非法访问记录仍原样保留。

硬件判断依据本地 Shader Guide §4.3.7.4：WMMA 与其他 VALU 可乱序完成，
非零 VA_VDST wait 的正确性需要检查实际依赖；§4.6.12.1规定 WMMA data hazards。
不能凭减少 WMMA 数量就删掉 DS/WMMA hazard waits，也不能把 compiler boundary
通过或成功 trace 当成所有实际时序已经安全的证明。

### 已封存产物、复核入口与继续方向

| archive | members /bytes | SHA256 |
|---|---|---|
| `TASK/logs/f3_v60_v65_GEMM1_ATT_diagnostic_20261026_v1.ATT_evidence.tar.gz` | 98 /2561399 | `de6e362e8e2be6f0c0ca549cb389e1575b28aa03e01447a03a024340f7b8defe` |
| `TASK/logs/f3_v65_GEMM1_ATT_direct_identity_20261026_v2.ATT_evidence.tar.gz` | 107 /2503608 | `41029c182d41c10743364e45a213b1deb2a5469688feb478174d4de2117767e85` |

两 archive 均已逐 member SHA256 核验并导入；目标 `.out` 仅在 producer 侧计算 receipt，
未下载无关 `.out`。独立复核使用已有冻结 ELF，再逐机器指令字节与 decoded ISA核验，
保留 v60 的 controller error及缺失身份。

```text
TASK/logs/f3_v60_v65_GEMM1_ATT_diagnostic_20261026_v1.independent_local_review.json
TASK/logs/f3_v65_GEMM1_ATT_direct_identity_20261026_v2.independent_local_review.json
TASK/logs/f3_v60_v65_ATT_wave_classes_independent_20261026_v1.json
```

本轮已执行的入口如下；这些 label/产物已封存，不要原样覆盖重跑：

```bash
# a07-3 主机
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v60_v65_GEMM1_ATT_diagnostic_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v65_GEMM1_ATT_direct_identity_20261026_v2.py
# 本地独立复核
python .codex_tmp/hyg_f3_followup_20261024/review_f3_direct_CLI_ATT_local_20261026_v1.py f3_v60_v65_GEMM1_ATT_diagnostic_20261026_v1
python .codex_tmp/hyg_f3_followup_20261024/review_f3_direct_CLI_ATT_local_20261026_v1.py f3_v65_GEMM1_ATT_direct_identity_20261026_v2
```

所有子测试仍用 `timeout --signal=TERM --kill-after=2s 118s ...`，总上限约120秒。
ATT 脚本继承 `PYTORCH_ALLOC_CONF=expandable_segments:True`，这项诊断环境变化也保留，
不能说 ATT 与原非法访问的启动环境完全等价。

继续从两端保留的 v60 和已绑定 ISA研究无效 A payload reads 的条件、DS wait预算与
WMMA operand退休依赖；先做静态模型检查。v63/v65保持隔离，原 faulting kernel与
非法访问根因目前仍未唯一确定。原 v3 的加载地址相关 I-cache miss 放大已有独立证据，
与本次非法访问分别记录，不能用一项的结论替代另一项。


### ?? v65 ATT archive ?????

??? v65 ATT archive ??? SHA256 ????????? member ??? manifest ? archive ???????????
`41029c182d41c10743364e45a213b1deb2a5469688feb478174d4de211767e85`?archive ?107 members?2503608 B?????ELF/ISA???????????????


## 静态模型校准、v66 无效 A payload 与 v67 局部等待（2026-10-26）

本节接续 SDK/ATT 记录。生产继续使用 v60，源码 SHA256 为
`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。
本轮尚未新增正式 kernel 性能或 MoE e2e 性能；v60 最新正式三轮 GEMM1
为 81.6604 /81.6815 /80.7704 μs，中值 81.6604 μs，pin=0。

### 先校准原模型对 mixed VALU completion 的假设

读取本地 Shader Guide §4.3.7.4、§4.3.7.4.1、§4.3.7.4.2、§4.6.12.1 与
§5.3.1.3。文档同时写有 WMMA 与其他 VALU 的 out-of-order completion，以及
VA_VDST 的 in-order counter update；不能据此假设任意 mixed VALU 均按序退休。
实际两版入口先设 SCHED_MODE[1:0]=2，随后设置 bit[2] DISABLE_XDL_ARB_STALL；
两者作用不同。SCHED_MODE 2 的 `VALU ← VGPR ← VMEM` WAR 需要适当的
`S_WAIT_ALU VA_VDST`，不能用“通常已读完 source”替代依赖检查。

原 native 模型按每条后续 `v_*` 增加 WMMA age。本轮增加保守的 WMMA_only stress：
只按后续 WMMA 增加 age，其余 DS/WMMA transfer 规则保持。v60、v65 在
original 和 WMMA_only 两种模型下均 0 issues。这并未定位原非法访问，也不等于
完整 hardware simulation 或所有设备时序已验证。

复核入口和结果：

```text
TASK/review_f3_native_WMMA_ordering_stress_20261026_v2.py
TASK/logs/f3_native_WMMA_ordering_stress_20261026_v2.summary.json
```

v1 的失败是 LLVM physical annotation comment 文本归一化问题，原失败日志保留。
v2 沿用既有 comment normalization，仍逐 PC、ISA 与 ELF 机器字节核验，没有放宽
hazard 标准或删除原失败。

### v66：A reads 和 WMMA 使用相同的尾行条件

v66 基于隔离的 v65，在 `load_active_a()` 内按相同尾行条件跳过数学上无效的
A payload LDS reads，替换三个 loader 调用点。8194 组实际 predicate 执行确认
A-read/WMMA 条件对应、有效行完整覆盖；B/scales/TDM/barrier、原 compile guard
与有效计算表达式保持。它只生成隔离 snapshot，没有落入生产。

| 项目 | v66 结果 |
|---|---|
| snapshot source SHA256 | `b2843b17287af361cbf41cc5349d0be58550231e8f320b638d5d4596a519e368` |
| ELF SHA256 | `7c545982eeb8f5d388a167682830c9e1fa786f2c7557d125d330d8988c638235` |
| 资源 | 798 VGPR、57 SGPR、327680 B LDS、private/spills0 |
| `.text` | 44544 B |
| 原 caller 编译 /inspect | 15.838秒 /0.114秒 |
| GEMM1 target dispatch | 未执行；input preparation 含既有 GPU 操作 |
| original /WMMA_only | 两者均 14处 `DS overwrite before WMMA retirement` |
| 精度、ATT、正式性能 | 均未执行，不作为可采用候选 |

审计进程 returncode0 只表示报告生成完成，报告的 `model_passed=false` 才是候选状态。
不能把 returncode0 写成 native gate 通过。编译结束已恢复 v60，KFD owners 为空。
CPU archive 已下载并逐 member 核验，共20 members、433574 B：

```text
TASK/logs/f3_v66_invalid_A_payload_CPU_evidence_20261026_v1.tar.gz
TASK/logs/f3_v66_native_WMMA_ordering_stress_20261026_v1.summary.json
```

14处均涉及尾行 A payload 对应物理 VGPR。例：PC0x2824 的 wait8 后，
PC0x282c/0x2834 重新写入 v656:663；后一对 reads 前才有 wait0。
PC0x2c1c/0x2c24 对 v640:647 存在同类报告。这是保守 native 模型发现的风险，
不能直接说成设备已复现的 race 或原 v65 illegal address 根因。

### 条件一致性检查没有把完整行风险排除

为避免把相同 `mn_oob > 192` 的多次判断当成独立选择，新建只读 CFG refinement，
保留原 WMMA_only transfer AST SHA256
`e2e95fb0c4ecd2fcb693dff5aba464732f9431468ff9808b46ff4e5f74675e17`。
按实际 ISA 中 s44 的唯一定义、s21 predicate、SCC 与不变的非零 EXEC 推导分支。
再核验 s46 的唯一 `v_readfirstlane_b32 s46, v0`、原 caller `warp_size=32`、
128线程 WG，分别覆盖 wave0–3 的 TDM producer 分工，没有修改 kernel 或 wait 规则。

| CFG 检查 | `mn_oob <= 192` | `mn_oob > 192` |
|---|---|---|
| 仅 row 条件一致性 | 0 issues | 14 issues |
| row + wave0固定分工 | 0 issues | 14 issues |
| row + wave1固定分工 | 0 issues | 10 issues |
| row + wave2固定分工 | 0 issues | 10 issues |
| row + wave3固定分工 | 0 issues | 10 issues |

```text
TASK/logs/f3_v66_row_predicate_CFG_20261026_v1.summary.json
TASK/logs/f3_v66_wave_predicate_CFG_20261026_v2.summary.json
```

因此当前不能将14处报告全部归为相同 row 条件的路径误合并。以上只覆盖既有
DS/WMMA register model 与 active waves；其他 VALU/VMEM hazards、tensor completion
和精确 source-read latency 未完全建模，不作设备正确性或故障根因声明。
v2 reviewer 初次因 inspection 未含 `wavefront_size` 字段而停止，后改为核验原 caller
cache target 的 `warp_size=32`；该失败发生在 case evaluation 前，已在报告中保留。

### v67：仅在需要尾行 A reload 时加局部 retirement wait

v67 在 v66 的条件 A reads 内加入 `s_wait_alu depctr_va_vdst(0)` 和前后 compiler
boundary。每个 reloaded k128 的两条尾行 rep 各执行一次等待；跳过的行不执行
新增 wait/read。实际 A/WMMA predicates、有效计算、B/scales/TDM/barrier requests、
wave layout 与编译 guard 均保持，没有增加 persistent、新 pipeline、register pin
或 expert-count specialization。

隔离源码 SHA256：
`7b048f4d9de28d84444bd8dd573c07befc2e94ce08da9147977271f13dbbea12`。
已仅传输相关 snapshot、controller、plan、review 和 diff 到 a07-3，逐 SHA256 核验。
当前进行原 caller 编译与实际 ISA 检查，尚未给它精度/性能成绩，也没有替换生产 v60。

```bash
# a07-3 主机；controller 在 dsr1_ep4 内启动每个 compiler child
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v67_invalid_A_retirement_cpu_20261026_v1.py
```

每个 child 保持 `timeout --signal=TERM --kill-after=2s 118s`，约120秒硬上限。
只有实际 ISA 通过原 native gate 后才进入设备精度与 pin0 正式对照。
v66 原失败、v65 原非法访问和已完成 ATT 原产物继续保留，不能宣称原 fault 已修复。

### 更正上一条本地 archive SHA 勘误

此前本地追加的勘误因写入编码出现 `?` 且漏了一位 SHA256，不能用那一条校验 archive。
v65 ATT archive 的正确完整 SHA256 为：
`41029c182d41c10743364e45a213b1deb2a5469688feb478174d4de2117767e85`。
archive 共107 members、2503608 B；以已核验的 archive/manifest/member receipts 为准。
本节以 UTF-8/LF 写入，保留两端各自原 QA 前缀，没有全文覆盖历史。

## v67 实际 ISA 通过与 E64 完整 MoE 精度通过（2026-10-26）

v67 编译完成，GEMM1 target 尚未在 CPU 编译阶段 dispatch；compiler child
15.436754秒，inspect child0.113780秒，input preparation 含既有 GPU 操作。
生产已恢复 v60。本节没有新增正式性能数据。

| 项目 | 结果 |
|---|---|
| v67 source SHA256 | `7b048f4d9de28d84444bd8dd573c07befc2e94ce08da9147977271f13dbbea12` |
| v67 GEMM1 ELF SHA256 | `54402e63f3d6db3722a4832dd744457c975c59e07eea7950a08f7e0cd30f48b2` |
| `.text` /资源 | 45312 B；798 VGPR、57 SGPR、327680 B LDS、private/spills0 |
| original native model | `model_passed=true`，0 issues |
| WMMA_only stress | `model_passed=true`，0 issues |
| E64/T1536/topk8 完整 MoE，seed0/17 | 两个原精度 gate 均通过；各22020096 B 与已验证 v60 零字节差异 |
| E65/T1536/topk9 | 原 routed fallback 验证进行中；不代表 fixed shared1 或共用 compiled GEMM1 |
| 正式性能 | 尚未测量；生产仍为 v60/pin0 |

局部 full-row A reload 的 wait0 确实出现在条件内，例如首组 PC0x27c0/0x27c4，
后续 `ds_load_b128` 在 PC0x27cc/0x27d4。原模型与 WMMA_only 的所有 native PC、
ISA 和机器字节仍逐项绑定冻结 ELF；没有用 CFG refinement 把失败直接改成通过。
在 v67 的原 overapproximated CFG 中也已0 issues。
这说明 v66 的既有 bounded model 风险在新对象中已消除，不证明原 v65 illegal address
根因已修复，不等于精确硬件时序模拟或所有运行均安全。

```text
TASK/logs/f3_v67_native_WMMA_ordering_stress_20261026_v2.summary.json
TASK/logs/f3_v67_invalid_A_retirement_pin0_native_20261026_v1.review.json
TASK/logs/f3_v67_cpu_local_binding_20261026_v1.json
TASK/logs/f3_target_v67_invalid_A_retirement_prepare_20261026_v1_controller.summary.json
NORMAL/logs/f3_target_v67_invalid_A_retirement_prepare_20261026_v1.preparation.json
```

CPU evidence archive 已下载并逐 member 核验/导入，共14 members、400327 B，SHA256：
`facfd3494cbfc311bf1b03414a2e3a186f4aab9b171b457523cc559cc2a0361a`。

本轮还保留了几项执行前的工具问题：v67 maker 初次将未变化的 sibling loader AST
也列入 changed 集合，断言在 snapshot 写入前停止；packer 初次对 member 总数设了
不正确的固定最小值，改为逐个检查必需 artifact 后打包；native stress v1 手抄 ELF SHA
多一位，冻结对象 binding 立即拒绝，v2 改为读取已核验 archive 中的实际64位 SHA。
这些失败均未产生 target GEMM1 dispatch，不是设备故障或 kernel 正确性结论。

旧 SSH PTY 的输出曾延后到达；v67 编译实际已完成，随后一次启动被
“label/cache already exists”门禁拒绝。原完成编译没有重跑、覆盖或删除。
SFTP 的批量传输也先等全部命令回到 prompt，再按清单逐 SHA256 验证后启动设备测试。

E64 device preparation 的四个 child（两阶段编译、seed0/17 strict MoE）总计47.842311秒，
各 compiler/test child 仍使用118秒 TERM、再2秒 KILL的硬上限。实际输出与基准逐字节比较，
计时或精度标准未改；GEMM2 全 ELF 保持
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

```bash
# a07-3 主机；子构建/验证在 dsr1_ep4
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v67_prepare_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v67_guard_outside_e65_20261026_v1.py
```

下一步完成 E65 原 fallback 验证，独立复核证据后，以 pin0 原 CLI 测量
v60/v67/v60 三组、每组三轮；只有全机三次连续空闲、计时后无残留且 cache/ELF
验收通过的轮次才作为正式性能，按中值报告，不挑最快读数。

## 2026-10-26：v67 完整精度和 pin0 正式性能结论；继续隔离检查 v68

v67 的 E64/topk8 与 E65/topk9 原 routed fallback 在 seed0/17 下均已完成完整 MoE
验证。各输出为1536×7168 BF16、22,020,096 B，四份输出与各自已验证 reference
均为零字节差异，原 precision gate 通过。E64 的 `logits_diff` 为
3.6105513278025183e-06 /3.6202693316254594e-06，`rel_l2` 为
0.0026872099842876196 /0.002690816530957818；E65 分别为
3.594893112763491e-06 /3.6086727178189905e-06 和
0.002681376878172159 /0.0026865052059292793。

这里的 E65 是原 routed fallback；fixed shared1 和 E64/E65 共用 compiled GEMM1
的历史要求仍未建立。本轮精度通过不能用来声称原 v65 `hipErrorIllegalAddress`
根因已经修复。GEMM2 完整 ELF 保持：
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

### 正式性能：固定 v60 /v67 /v60，每组三轮

只在 a07-3/dsr1_ep4 使用原 `--scenario kernel` CLI、E64/T1536/topk8、
uniform/auto/seed0、A-preshuffle=1、warmup5/iters20。
`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`，原 timer 和 precision gate 未修改。

| 顺序 /版本 | GEMM1 r1 μs | GEMM1 r2 μs | GEMM1 r3 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---:|---:|---:|---:|---:|
| 前侧 v60 | 82.0651 | 80.9244 | 81.2112 | 81.2112 | 64.3078 |
| v67 | 83.8994 | 83.9353 | 83.4168 | 83.8994 | 64.8668 |
| 后侧 v60（attempt2） | 81.1396 | 81.1796 | 80.8358 | 81.1396 | 64.5422 |

v67 相对前侧耗时增加3.3101345627204064%，相对后侧增加3.401298502827224%。
两侧均回退，因此不采用。没有因最快一轮、降低精度或删除有效计算而改变决策。
本地仍为 v60，remote controller 已在 finally 恢复 v60；source SHA256：
`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。
此前保留版三轮中值81.6604 μs仍作为既有 pin0 记录；本轮前后侧是新的对照记录，
不把单次更快读数当成新的优化成果。距70 μs仍约11 μs，尚未达标。

后侧 attempt1 的第一轮 GEMM1/GEMM2=80.9117 /63.7725 μs，但 post/settled
检查发现外部 GPU ownership 或未恢复空闲，因此整组保留并排除，自动等待全机空闲后
从 attempt2 的 r1 开始重测。排除由 ownership 门禁触发，与耗时快慢无关。
九轮有效数据均逐次验收了三次连续全机 idle（间隔≥6秒）、原始表格、实际 source/
cache keys、完整 ELF 与运行后 ownership/settled 状态；未发现测试启动的进程残留。
每个 compiler/test/audit child 都使用118秒 TERM、再2秒 KILL上限，测试和 audit
实际均未超时。没有新增正式 MoE e2e 性能；完整 MoE 的精度运行不能充当 e2e 耗时。

已执行的主机命令（各子构建和 GPU 测试进入 dsr1_ep4；已存在的 label 不可原样重复）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v67_prepare_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v67_guard_outside_e65_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v67_vs_v60_comparison_20261026_v1.py
```

### 证据验收与后续检查

精度 archive：58 members、989365 B，SHA256
`353e0180c0290e8fd9438fee3139ade7b19fe20d3df544ae59c467bf9ff987a1`。
正式性能 archive：207 members，压缩大小2091053 B，SHA256
`cfe926621523086be7a989d8f81c90d7282337eb2db2eca5c991d2615243ed47`。
后者已逐 member 校验并导入，独立 reviewer 从原表重算中值和两侧差值通过，包含
ownership 不合格的 attempt1，而非只保存有效或更快的结果。

```text
TASK/logs/f3_v67_precision_independent_local_review_20261026_v1.json
TASK/logs/f3_target_v67_vs_v60_comparison_20261026_v1.summary.json
TASK/logs/f3_v67_vs_v60_comparison_independent_local_review_20261026_v1.json
TASK/logs/f3_v67_comparison_archive_members_local_20261026_v1.json
TASK/logs/f3_v67_comparison_archive_imported_local_20261026_v1.json
TASK/logs/f3_v67_v60_performance_evidence_20261026_v1.tar.gz.manifest.json
```

v68 只把 v67 的尾行 wm6/wm7 两次局部 `wait0` 合并成一次：wm6 前 wait0，
wm7 后关闭 schedule barrier；两个 payload read 的 row predicate 保持相同。
原 `load_lds_data` 按 rep 升序调用，两次读取之间的 source 没有 WMMA 调用。
8,194组实际 source predicate/覆盖检查通过，有效行、WMMA、B/scales/TDM/
barrier requests、guard、pipeline 和 wave layout 均保持。没有新增 expert-count
specialization，也没有修改 pin=0 或原计时口径。

v68 snapshot/source SHA256：
`d65d2af564f956f0df31ce89088e64bed5e68f5157b236404c1479e51d930c96`。
原 caller CPU 编译已完成，资源为798 VGPR、57 SGPR、327680 B LDS，private/spills0，
compile controller 已恢复 v60。尚须对实际 ISA 跑原 native 和 WMMA_only
依赖 gate，通过后才运行完整 MoE 精度和正式对照；此刻没有 v68 设备精度、ATT
或正式性能结论。v67 回退的 μs 本身不能唯一定位为 `wait0` 成本，v68 用于检验
这一调度改动，不能先宣称已得到收益。

## 2026-10-26：v68 精度通过但正式性能回退；v69 继续细化无效尾行

v68 把同一 row predicate 下的 wm6/wm7 A payload reload 的两次局部
`s_wait_alu depctr_va_vdst(0)`合并为一次，并用 schedule barriers 包住这对读取。
source SHA256 为`d65d2af564f956f0df31ce89088e64bed5e68f5157b236404c1479e51d930c96`；
原 caller 实际 GEMM1 完整 ELF 为
`a9aa389f285cc222d1fbe5b7d185c19d5abf9d453c4d4c2f15f0be3d08ef0a4e`。
资源保持798 VGPR、57 SGPR、327680 B LDS、private/spills0。

原 native model 与 WMMA_only 均0 issues；两者分别耗时2.1108712 /1.7272110秒。
所有实际 PC、ISA 和 machine words 与冻结 ELF 绑定，原 transfer 模型保持，
没有通过修改退休规则来消除疑点。这是既有 bounded model 的通过，仍不能称为
精确硬件模拟，或用它解释/修复原 v65 的 `hipErrorIllegalAddress`。

v68 的 E64/topk8 和 E65/topk9 原 routed fallback 在 seed0/17 下共四次完整 MoE
均通过原 precision gate，各22,020,096 B输出与各自已验证 reference 零字节差异。
E64 preparation controller 的四个编译/验证 child 合计48.044793674秒，各 child
保持118秒 TERM、再2秒 KILL的上限。E65 两个 child 也未超过两分钟。
精度验证后全机三次连续 idle 通过，没有本轮进程残留。E65 fixed shared1 和
E64/E65 共用 compiled GEMM1 的历史要求仍未建立。

### 正式 pin0 性能：v60 /v68 /v60，各三轮

原 E64/T1536/topk8、uniform/auto/seed0、A-preshuffle=1、warmup5/iters20、
`--scenario kernel` CLI 和 precision gate 保持。
`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`，仅在 a07-3/dsr1_ep4 测试。

| 顺序 /版本 | GEMM1 r1 μs | GEMM1 r2 μs | GEMM1 r3 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---:|---:|---:|---:|---:|
| 前侧 v60 | 80.8254 | 82.5165 | 82.3877 | 82.3877 | 64.5272 |
| v68 | 83.6675 | 84.3001 | 84.5909 | 84.3001 | 64.0339 |
| 后侧 v60 | 82.4067 | 81.1354 | 81.4092 | 81.4092 | 64.7676 |

v68 相对前侧增加2.3212202792407184%，相对后侧增加3.5510728517170076%，
两侧均回退，**不采用**。本轮没有排除或重试组；九轮全部合格，所有原读数保持。
独立 reviewer 从原日志表格重算中值、验证三次连续全机 idle（间隔≥6秒）、
实际 source/keys/完整 ELF、原 timer 及 post/settled 状态通过。
GEMM2 完整 ELF 仍为
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

正常对照已结束，controller 已在 finally 恢复 v60；本地也保持 v60，source SHA256
`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。
保留版仍约81–82 μs，尚未达到70 μs。此前81.6604 μs为已记录的 pin0 三轮中值，
本轮81.4092 μs是后侧对照中值，不能把这种组间变化当作源码优化收益。
没有新增正式 MoE e2e 耗时，完整 MoE 精度运行不作为性能数据。
v67/v68 均回退，说明这些减少无效读取/合并等待的具体实现没有带来收益；
目前不能仅凭 μs 唯一判定回退属于哪个等待或调度环节，不重复抓明显回退版本的 ATT。

已执行主机复现命令（子构建和 GPU child 在 dsr1_ep4；固定 label 已存在，不能原样重复）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v68_shared_A_retirement_cpu_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v68_prepare_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v68_guard_outside_e65_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v68_vs_v60_comparison_20261026_v1.py
```

三份 archives 已下载、逐 member 校验并导入，独立 CPU/native、精度和性能复核通过：

| archive | members | 压缩 bytes | SHA256 |
|---|---:|---:|---|
| `f3_v68_shared_A_retirement_CPU_evidence_20261026_v1.tar.gz` | 14 | 400367 | `ef398ee1c5c2e87efac1460bc7103a527b76752ede31e5e666ca6ab0071572e7` |
| `f3_v68_E64_E65_precision_evidence_20261026_v1.tar.gz` | 58 | 989332 | `ab0a05df164754505f0c0f000ad9eb4ea232e11d67bcb502c54d98eb818c32ba` |
| `f3_v68_v60_performance_evidence_20261026_v1.tar.gz` | 190 | 1976428 | `5bfc5715f726dd6d363ca67951a7204656aa3c30c8d6efa8d9f6e106ea09c568` |

```text
TASK/logs/f3_v68_native_WMMA_ordering_stress_20261026_v1.summary.json
TASK/logs/f3_v68_shared_A_retirement_pin0_native_20261026_v1.review.json
TASK/logs/f3_v68_precision_independent_local_review_20261026_v1.json
TASK/logs/f3_v68_precision_post_all_GPUs_idle_20261026_v1.log
TASK/logs/f3_target_v68_vs_v60_comparison_20261026_v1.summary.json
TASK/logs/f3_v68_vs_v60_comparison_independent_local_review_20261026_v1.json
```

### v69：同时细化尾行 WMMA 和 A payload 条件，仍为隔离候选

v69 从 v67 的逐次局部 retirement wait 实现派生：wm0–5 始终执行，wm6 在
`mn_oob > 192`时执行，wm7 在`mn_oob > 224`时执行；对应 A payload 使用完全相同的
条件，每个仍执行的尾行 A read 前保留自己的 wait0。193–224个有效行的 tile
会执行七个 rep，而不是八个；有效行全部覆盖，不减少有效计算。

8,194组实际 A/WMMA/wait predicate 与有效行覆盖检查通过。现有 compile guard、
B/scales/TDM/barrier requests、K loop、wave layout、epilogue和原 timer 均保持，
没有新增 expert-count specialization、persistent、w2x4或新 pipeline。
这项是 source predicate 的真实修改，review 明确记录
`actual_A_and_WMMA_predicates_AST_unchanged=false`，并以实际匹配/覆盖检查验证，
不能沿用“predicate AST保持”来描述它。

旧 v37 已试过 wm7 的224阈值，前/候选/后中值81.5316 /84.1018 /82.2379 μs，
因此淘汰；它还对 wm4/wm5 加128阈值，且只跳过 WMMA、没有跳过对应 A读取。
v69 保持前六个 rep 无条件执行，并将 A读取与计算一起细化、补足局部 retirement，
并非原样重复 v37，也不先宣称这种组合有收益。

v69 source SHA256为
`2034e4388e713131c32a5e8af98b2cb86bba42474bd5a0306d03c9ae108c6110`。
原 caller CPU 编译已完成：798 VGPR、58 SGPR、327680 B LDS、private/spills0，
controller 已恢复 v60。下一步对实际 ELF 跑原 native 和 WMMA_only gate；
尚无 v69 target GEMM1设备精度、ATT或正式性能结果，不声称原 v65故障已修复。

### v69：native、完整 MoE 精度和双侧 pin0 性能已完成，不采用

v69 保留 wm0–5 无条件执行，wm6 在 `mn_oob > 192` 时执行，wm7 在
`mn_oob > 224` 时执行。A payload 与对应 WMMA 使用相同的实际条件，每个活跃
尾行 A read 前有自己的局部 `wait0` 和 compiler boundaries。8,194 组实际条件
与有效行覆盖检查通过；predicate AST 确实改变，不能描述成 predicate AST 不变。
原 compile guard、B/scales/TDM/barrier requests、K loop、wave layout 和 epilogue 保持。

实际 GEMM1 ELF SHA256 为
`65357f36299292a756c31ebfef1fdb7743ceb74b068e1efedfbc50aed18de033`；
资源为 798 VGPR、58 SGPR、327680 B LDS、private/spills0。
原 native 和更保守的 WMMA_only transfer model 均为 0 issues，分别耗时
2.643972299993038 / 1.8998237999621779 秒。模型的既有边界保持：这些通过结果
不能代替精确硬件模拟，也不能据此声称原 v65 的 `hipErrorIllegalAddress` 已修复。

E64/topk8 和 E65/topk9 原 routed fallback 在 seed0/17 下共四次完整 MoE
均通过原 precision gate，每份 22,020,096 B 输出与各自已验证 reference 零字节差异。
E64 的 logits_diff 为 3.6105513278025183e-06 / 3.6202693316254594e-06，
rel_l2 为 0.0026872099842876196 / 0.002690816530957818；E65 的 logits_diff
为 3.594893112763491e-06 / 3.6086727178189905e-06，rel_l2 为
0.002681376878172159 / 0.0026865052059292793。精度阈值和原测试保持。
精度验证后全机三次连续 idle 通过，无本轮进程残留。
E65 fixed shared1 和 E64/E65 共用 compiled GEMM1 的历史要求仍未建立。

正式性能保持 E64/T1536/topk8、uniform/auto/seed0、A-preshuffle=1、warmup5/iters20、
原 `--scenario kernel` CLI；统一 `AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`。
只在 a07-3/dsr1_ep4 测试，各 compiler/test child 保持 118 秒 TERM、再 2 秒 KILL 上限。

| 顺序 /版本 | GEMM1 r1 μs | GEMM1 r2 μs | GEMM1 r3 μs | GEMM1 中值 μs | GEMM2 中值 μs |
|---|---:|---:|---:|---:|---:|
| 前侧 v60 | 80.7007 | 81.0256 | 81.6494 | 81.0256 | 65.1532 |
| v69 | 83.3172 | 83.0264 | 84.1524 | 83.3172 | 64.1813 |
| 后侧 v60，attempt2 | 80.6439 | 80.5448 | 81.5379 | 80.6439 | 64.3079 |

v69 相对前侧增加 2.828241938350352%，相对后侧增加 3.314943845721744%，
两侧均回退，**不采用**。后侧 attempt1 的第一轮原读数为
GEMM1 81.6432 μs / GEMM2 63.7829 μs；测后 ownership/settled gate 不合格，
因此整个 attempt1 排除并保留原始记录。controller 自动等待空闲后按相同固定三轮
重测 attempt2，没有按耗时选择重试。其余九个合格轮次全部保留。

独立 reviewer 从原日志表格重算所有中值，验证每轮三次连续全机 idle（间隔≥6秒）、
实际 source/keys/完整 ELF、原 timer 和 post/settled 状态通过。GEMM2 完整 ELF
保持 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

controller 已在 finally 恢复 v60；两端生产 source SHA256 均为
`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。
HEAD 保持 `6f656cf5b02b2f1acf723949652f91b48e2933d7`，分支为
`hyg_dsr1_conc6144`，boot ID 保持 `cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`。
本轮后侧 v60 中值 80.6439 μs 距 70 μs 差 10.6439 μs；此前 81.6604 μs 是另一组
已记录的 pin0 三轮中值，不能把组间变化当作源码优化收益。没有新增正式 MoE e2e
性能数据，完整 MoE 精度运行不计作性能数据，也不重复抓明显回退候选的 ATT。

已执行主机命令（构建和 GPU child 在 dsr1_ep4，固定 label 已存在，不能原样重用）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v69_row_specific_A_retirement_cpu_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v69_prepare_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v69_guard_outside_e65_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v69_vs_v60_comparison_20261026_v1.py
```

三份 archives 均已下载、逐 member 校验并导入，独立 native、精度和性能复核通过：

| archive | members | 压缩 bytes | SHA256 |
|---|---:|---:|---|
| `f3_v69_row_specific_A_retirement_CPU_evidence_20261026_v1.tar.gz` | 14 | 411827 | `d0c1a7b2799db434b28275404076e23058508b51722c23bff2733639f4defb19` |
| `f3_v69_E64_E65_precision_evidence_20261026_v1.tar.gz` | 58 | 991296 | `45b104925d624c7d095f4c1774aa9e6fd4614407ff1d7552ccff2ab7a2807105` |
| `f3_v69_v60_performance_evidence_20261026_v1.tar.gz` | 207 | 2101190 | `7da8609e402d5eab36f4d0c223ad2c352033b9e728139d5a113b47ad7584202c` |

关键记录：

```text
TASK/logs/f3_v69_native_WMMA_ordering_stress_20261026_v1.summary.json
TASK/logs/f3_v69_precision_independent_local_review_20261026_v1.json
TASK/logs/f3_v69_precision_post_all_GPUs_idle_20261026_v1.log
TASK/logs/f3_target_v69_vs_v60_comparison_20261026_v1.summary.json
TASK/logs/f3_v69_vs_v60_comparison_independent_local_review_20261026_v1.json
```

下一步仅尝试缩小 v69 loader 的出口 compiler boundary：入口边界和尾行局部 wait0
保持，原 final drain 出口边界保持。先 CPU 编译并对实际 ISA 跑原 native 与 WMMA_only
gate，通过后才验证完整 MoE 和正式性能；当前尚未创建或采用该候选。

### v70 /v71：缩小 loader compiler boundary 未改变实际 ELF，不作为新性能结果

v70 从 v69 派生，只将 loader 出口的 compiler boundary 条件从
`final_drain_load or interleaved_m_rows` 缩小为 `final_drain_load`。
入口 boundary、两条活跃尾行 A read 前的局部 wait0 和各自 boundaries 保持。
v71 再从 v70 派生，对入口做同样缩小；出口、尾行局部保护保持。
两版各自 128 组实际 boundary 条件检查通过，原 final drain 的入口/出口保护保持。
两版均保留实际 A/WMMA predicate、B/scales/TDM/barrier requests、K loop、wave layout、
原 timer 和 precision gate，不新增 expert-count specialization、pipeline、persistent 或 w2x4。

| 版本 | source SHA256 | 原 caller 编译 child 秒 | inspect child 秒 |
|---|---|---:|---:|
| v70 | `86136a3e0bc3aec3ead5f31207a5357de30763eb1199cfea34d1f69bfd04818f` | 15.637358336010948 | 0.1137458709999919 |
| v71 | `c056a3db77c7cb634c671048dd97f66d0664aeda9b702f7e90961cf905b463bb` | 15.587088396016043 | 0.11383198096882552 |

两个 controller 均完成并恢复 v60；没有 dispatch 新候选 target GEMM1。
输入 preparation 可能运行既有 quant 等其他 GPU kernel，不能将“target GEMM1 未执行”
写成“整个 preparation 完全没有 GPU 工作”。所有 child 未超过 120 秒。

两份 archive 已下载、逐 member 验收、导入；独立 reviewer 核验了完整对象每个字节、
实际 ISA PC/machine words、原 caller 的 argument/global signatures、原 cache key 生成方式
和 source IR。v70/v71 的 source IR 都与 v69 不同，但完整 GEMM1 ELF 与 v69 逐字节相同：

```text
65357f36299292a756c31ebfef1fdb7743ceb74b068e1efedfbc50aed18de033
```

资源都为 798 VGPR、58 SGPR、327680 B LDS、private/spills0。因此只按完整对象身份
复用 v69 的 target GEMM1 原 native 和 WMMA_only gate，不重复运行相同 native 模型或
GPU 精度/性能测试。两版新 native 次数、新 target GEMM1 case 数、新完整 MoE case 数和
新性能轮数均为 0；没有新成绩，也未采用。v69 的 83.3172 μs 不包装成 v70/v71 新测数据。
这些对象身份检查不声称原 v65 的设备故障已经修复。

| archive | members | 压缩 bytes | SHA256 |
|---|---:|---:|---|
| `f3_v70_loader_exit_narrow_CPU_evidence_20261026_v1.tar.gz` | 14 | 411808 | `9a5c47368e68f19b104cb4b162342d61d40d2c94d0b9e3e4d6343f93f01ac2b4` |
| `f3_v71_loader_entry_narrow_CPU_evidence_20261026_v1.tar.gz` | 14 | 411631 | `23a40a9f3e87ef4d83c35b94280f2041a588a7570a3ee1ba49156cbf9dbb7f3a` |

独立复核记录：

```text
TASK/logs/f3_v70_ELF_identity_independent_local_20261026_v1.json
TASK/logs/f3_v71_ELF_identity_independent_local_20261026_v1.json
```

已执行的主机入口（compile/inspect child 在 a07-3/dsr1_ep4，pin=0；已有 label 不得重用）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v70_loader_exit_narrow_cpu_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v71_loader_entry_narrow_cpu_20261026_v1.py
```

这只证明当前目标配置下这两处 source boundary 缩小没有改变编译结果，不能推断所有
boundary 都可删、其他 shape 同样等价，或 boundary 对性能永远没有作用。此方向先停止。
下一候选计划保留 v65 的全部 A payload 读取，只将 wm7 的 WMMA 条件从成对的 192 阈值
细化为 224，wm0–5 仍无条件执行；避免引入 v69 的条件 A read 和局部 wait0 开销。
先 CPU 编译并检查实际 ISA；原 v65 正式性能流程曾失败，不能借新候选声称该故障已修复。
当前生产继续保留 v60，70 μs 目标尚未达到，未新增正式 MoE e2e 性能或 commit/push/pull。

### v72：保留全部 A 读取，只细化尾行 WMMA，native 与完整 MoE 精度通过

v72 从 v65 的完整 striped loader 派生，只将尾行 WMMA 的成对条件
`(wm // 2 * 2) * M_ROW_STEP < mn_oob` 改为 `wm * M_ROW_STEP < mn_oob`。
wm0–5 始终执行，wm6 的阈值仍为 192，wm7 的阈值变为 224；193–224 个有效行
的 tile 执行七个 rep。保留全部 A payload 读取，没有 v69 的条件 A read 或新增局部 wait0。
所有有效行计算保持，允许跳过的只是不产生有效输出的尾行 WMMA。

实际 WMMA predicate AST 确实改变。执行实际 predicate 的 16,388 组检查通过，
包含启用/禁用 guard、pin 两种值和 0–4096 有效行数；目标 interleaved 行映射的有效行
全部覆盖。`load_lds_data`、A/B/scales read helpers、`emit_row` 有效数学、K loop helpers、
TDM/barrier requests 与 loader boundaries 在 source AST 中保持，原 compile guard 保持。
没有新增 expert-count specialization、pipeline、persistent 或 w2x4。

v37 曾同时对 wm4/wm5 增加 128 阈值并回退；v72 保留前六个 rep 无条件，
只把 wm7 的成对 192 阈值细化为 224，不是原样重复 v37。原 v65 的正式测试曾失败，
失败记录继续保留；本次修改不声称已经定位或修复其 `hipErrorIllegalAddress`。

```text
v72 source SHA256:
f2e529ae56e7a2f672ef2f2e665753f377be7d5e31c7b6d0e56128f3a6fb037a

v72 GEMM1 full ELF SHA256:
b642b7e3b6a716514b09981271d9c79df7c6064194d22ba51b9bcbe248e4ed26
```

原 caller 编译和 inspect child 分别耗时 15.387054517981596 / 0.11370538198389113 秒，
controller 完成并恢复 v60，没有 dispatch target GEMM1；输入 preparation 可用其他 GPU kernel。
资源为 798 VGPR、58 SGPR、327680 B LDS、private/spills0。实际目标 ISA 有 6,245 条静态
executable instructions，v69 为 6,607 条；这只是已绑定对象的静态统计，不能换算成性能收益。

原 native 和更保守的 WMMA_only transfer model 均为 0 issues，分别耗时
2.8307443000376225 / 1.9071418999228626 秒。原 transfer model 和既有 parser 扩展保持，
实际 PC/ISA/machine words 与完整 ELF 绑定；这些仍是 bounded model 检查，不能当作
精确硬件模拟，也不覆盖全部 VALU、VMEM、EXEC 和 TDM completion 情形。

四项新候选完整 MoE 输出均通过原 precision gate，且与各自已验证 reference 零字节差异：

| workload | seed | 输出 bytes | differing bytes | logits_diff | rel_l2 |
|---|---:|---:|---:|---:|---:|
| E64/topk8 | 0 | 22020096 | 0 | 3.6105513278025183e-06 | 0.0026872099842876196 |
| E64/topk8 | 17 | 22020096 | 0 | 3.6202693316254594e-06 | 0.002690816530957818 |
| E65 原 routed fallback/topk9 | 0 | 22020096 | 0 | 3.594893112763491e-06 | 0.002681376878172159 |
| E65 原 routed fallback/topk9 | 17 | 22020096 | 0 | 3.6086727178189905e-06 | 0.0026865052059292793 |

E64 preparation 编排合计 48.44533856300404 秒，各 child 保持 118 秒 TERM、再 2 秒 KILL。
E65 的两个 candidate case 新执行，reference 复用先前已经验证的 v45 baseline；不冒充新执行。
GEMM2 完整 ELF 保持
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
E65 fixed shared1 和 E64/E65 共用 compiled GEMM1 的历史要求仍未建立。

CPU 和精度 archives 均已下载、逐 member 核验、导入；本地独立精度复核通过：

| archive | members | 压缩 bytes | SHA256 |
|---|---:|---:|---|
| `f3_v72_row_specific_WMMA_full_A_CPU_evidence_20261026_v1.tar.gz` | 14 | 396378 | `554ed2d64dcc9528e35ec2252ab23c3a8ce3ecd9da0d1a542391b628de2d951d` |
| `f3_v72_E64_E65_precision_evidence_20261026_v1.tar.gz` | 58 | 985213 | `df0fc33a7d3ebd60964be4da49f36b756102d5cfbe442ff5f72fc6174143b06e` |

复现入口均在 a07-3 主机，构建和 GPU child 在 dsr1_ep4，pin=0；既有 label 不得原样重用：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v72_row_specific_WMMA_full_A_cpu_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v72_prepare_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v72_guard_outside_e65_20261026_v1.py
```

```text
TASK/logs/f3_v72_native_WMMA_ordering_stress_20261026_v1.summary.json
TASK/logs/f3_v72_row_specific_WMMA_full_A_pin0_native_20261026_v1.review.json
TASK/logs/f3_v72_precision_independent_local_review_20261026_v1.json
```

传输校验的第一次尝试因四个附加 metadata 路径保留 Windows 反斜杠而遇到 FileNotFoundError；
没有写入验收 receipt、修改生产 source 或运行候选 GPU。规范化为 POSIX 路径后全部 SHA256
匹配，记录为 `TASK/logs/f3_v72_device_tools_remote_verified_20261026_v2.json`，原错误未伪装成成功。

截至本节，全机 post-precision idle checker 正在每 6 秒重查：a07-3 有外部 GPU/KFD 任务，
因此尚未启动正式性能测试。等待程序在主机运行，不是一个超时的 GPU test child；不终止外部任务。
空闲后计划固定 v60 / v72 / v60 各三轮，保持原 timer、uniform/auto/seed0、A-preshuffle=1、
warmup5/iters20 和 `--scenario kernel`。当前没有 v72 正式性能数据或新的 MoE e2e 耗时，
生产保留 v60，未执行 commit/push/pull，70 μs 目标尚未达到。

## 2026-10-26：v72 旧对照未完成；v73 完整精度验收及连续空闲门禁

问：汇报下进展，然后继续。

答：当前正式保留 **v60、pin=0**。已记录的正式三轮 GEMM1 中值仍为
**81.6604 μs**，距 70 μs 目标差 11.6604 μs。本 checkpoint 没有新增完整、合格的
候选性能对照，也没有新增正式 MoE e2e 耗时。v3 加载实例相关的 I-cache 波动已有
独立证据，不重复诊断；v63/v65 的历史非法访问根因仍未唯一确定，不能宣称已修复。

### v72 旧性能批次：11 个整组被 external ownership 排除

`f3_target_v72_vs_v60_comparison_20261026_v1` 的前侧 v60 有 11 个尝试组被测后
external ownership 门禁排除，合格组为 0；v72 候选尚未进入正式测量。
外部主要工作负载为 lipiwang 的 `padding.out`，约 31.4 GiB VRAM。
第 12 个尝试尚未启动 test child 时停止 controller，原 summary 保留
`completed=false`、`source_restored=true` 和
`RuntimeError('Comparison interrupted with signal 2')`。全部排除读数保留，不参与中值，
没有按快慢选择重试。停止后检查没有本轮 controller、waiter 或 KFD 进程残留。

旧证据包已下载、逐 member SHA256 核验并导入：192 members、1,246,979 compressed bytes，
archive SHA256 为 `eab85940028d203f9151934311c7787a1924156039b0a36565bca15025ff569e`。
原 verifier 已成功的结果复用，没有覆盖重跑。

```text
TASK/logs/f3_target_v72_vs_v60_comparison_20261026_v1.summary.json
TASK/logs/f3_v72_incomplete_archive_verified_local_20261026_v1.json
TASK/logs/f3_v72_incomplete_archive_imported_local_20261026_v1.json
```

### 只加强测前等待，不改变原 timer 或精度口径

因外部任务反复使用短空闲窗口，新增独立 waiter，要求 11 次连续全机空闲观察，
每次间隔至少 6 秒、总跨度至少 60 秒。原三次观察的 waiter 与 v1 工具原样保留。
等待在主机运行，不是一个超过两分钟的 GPU test child；不终止外部进程。
原 measurement child、observer、controller 的保护函数 AST、输入、完整 ELF、
timer、warmup5/iters20、uniform/auto/seed0、原 precision gate 和固定组顺序保持。
新的独立 reviewer 同时核对每轮全部 11 个空闲 observation 和时间跨度。

v72 的新版固定 `v60 / v72 / v60` 对照正在执行；每组仍是三轮，只有完整通过
原精度、对象绑定和测后 ownership/idle 的组才有效。每个 GPU/compiler/profiler child
继续使用 `timeout --signal=TERM --kill-after=2s 118s`，总上限约 120 秒。
计时期间不上传、不远端编译、不额外查询 GPU；只读取保存的进度日志。

```bash
# a07-3 主机；既有 label 不得覆盖重跑
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v72_vs_v60_comparison_stable60_20261026_v2.py
```

### v73：ScaleA/ScaleB 请求错开，完整 MoE 精度已独立验收

v73 从 v60 派生，仅将 compact/pin0 loader 内 ScaleB 的读取条件从偶数
iteration 改为奇数 iteration；ScaleA 保持偶数，A/B payload 顺序保持：

```python
if const_expr(i % 2 == 1 and i // 2 < sb_pairs):
```

v61 曾把两类 scales 都移到对应 payload 之后并回退，v73 没有重做该方案。
所有地址、请求数量、operand 结果、同步、有效行数学和原 compile guard 保持。
1,024 组实际 loader AST exercise 通过；pin1 及 guard 外行为保持。
不假定具体硬件 FIFO 或 cycle 收益，不将请求顺序变化直接写成性能改善。

v73 source SHA256：`2098797e26f2d2cfd9a918d5db1e0279b744b664684558f20dd62269de70a1ba`。
GEMM1 full ELF SHA256：`b55e559bb6d3e294462992036e0b4d0a8b3ac3b9ca6f5e66aba102ec258fddb5`。
实际对象为 798 VGPR /56 SGPR /327680 B LDS，private/spills0；有 6,334 条静态
executable instructions。原 caller compile/inspect child 分别耗时
15.286841446 /0.113630775 秒，没有 dispatch target GEMM1；输入准备可能使用其他 GPU kernel。
原 native 经 7,116 次 transfer 后 0 issues；original/WMMA_only stress 均为 0 issues，
分别耗时约 1.5284643 /1.3001274 秒。它们是 bounded model 检查，不能替代真实硬件验收。

新执行的四项完整 MoE 均通过原 precision gate，且与各自已验证 reference 零字节差异：

| workload | seed | output bytes | differing bytes | logits_diff | rel_l2 |
|---|---:|---:|---:|---:|---:|
| E64/topk8 | 0 | 22020096 | 0 | 3.6105513278025183e-06 | 0.0026872099842876196 |
| E64/topk8 | 17 | 22020096 | 0 | 3.6202693316254594e-06 | 0.002690816530957818 |
| E65 原 routed fallback/topk9 | 0 | 22020096 | 0 | 3.594893112763491e-06 | 0.002681376878172159 |
| E65 原 routed fallback/topk9 | 17 | 22020096 | 0 | 3.6086727178189905e-06 | 0.0026865052059292793 |

E64 preparation controller 合计 47.94527493196074 秒，`completed=true`、
`source_restored=true`。E65 两项 candidate 新执行，reference 复用先前已验证的 v45
输出，没有冒充新执行；E64 reference 复用已完成并调和验收的 v60 preparation。
GEMM2 full ELF 保持 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
E65 fixed shared1 和 E64/E65 共用 compiled GEMM1 的历史要求仍未建立。

精度包 58 members、986356 compressed bytes，SHA256 为
`7f885fdf84632febdf15dfcf1d4c95979a42daefb58a2ee546721e0b98b7867e`，
已下载、逐 member 核验并导入，本地独立 precision reviewer 通过。
v73 的性能 controller 已在本地适配同样的 60 秒连续空闲门禁；尚未上传或启动，
待 v72 对照结束并恢复 v60 后继续。原 v73 三次 idle 性能工具没有运行或覆盖。

```bash
# 已执行的主机入口
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v73_prepare_20261026_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v73_guard_outside_e65_20261026_v1.py
# 已执行的本地独立复核
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v73_precision_independent_local_20261026_v1.py
```

```text
TASK/logs/f3_v73_precision_independent_local_review_20261026_v1.json
TASK/logs/f3_v73_precision_archive_verified_local_20261026_v1.json
TASK/logs/f3_v73_precision_archive_imported_local_20261026_v1.json
TASK/logs/f3_v73_stable60_tools_local_20261026_v2.json
```

### 下一候选仅作本地准备：v74 的互补 ScaleA phase

v74 从 v60 派生，仅将 ScaleA 改到奇数 iteration，ScaleB 仍在偶数，
不是把 v73 原样重复。source SHA256 为
`2673f71630ffcd13d04bcb0ea948e517bad8412e2d6bde676192ed58faa2da6c`。
实际 loader AST 的 1,024 组请求、结果、同步和 guard 外检查通过；本地生产 source 未改。
原 caller 的 untimed compile controller、证据 packer 已准备，尚未上传、编译或 dispatch，
因此没有 ELF/native/精度/性能成绩。需完整通过这些 gates 才能测性能。

本 checkpoint 只在本地追加；远端将在性能对照结束后使用相同 chunk 独立 CAS 追加，
两侧 QA 原历史均保留，不能用整份本地文档覆盖远端。生产仍为 v60，尚未提交、
push 或 pull；已授权的最终提交与复测任务继续保留。

## 2026-10-26：v72 对照因 HEAD 变化中断；核验实际代码后重新绑定

问：汇报下进展，然后继续。

答：当前正式保留 v60、pin=0，既有合格三轮 GEMM1 中值仍为 **81.6604 μs**，
距离 70 μs 目标差 11.6604 μs。本阶段尚未增加完整、合格的候选性能对照，
没有新的正式 MoE e2e 耗时。v73 的完整精度已通过，v74 仍只完成本地静态准备。

### 原 stable60/v2 保留两轮 partial，不改写失败结果

前侧 v60 的两轮原 precision、完整 ELF 绑定和测后 ownership/idle 都通过：

| round | GEMM1 μs | GEMM2 μs |
|---|---:|---:|
| r1 | 81.7634 | 63.7639 |
| r2 | 80.9603 | 64.1625 |

两次 benchmark 和两次 audit child 全部成功且不足 10 秒。第三轮 waiter 在
`2026-10-26T04:19:55+00:00` 完成后，旧 HEAD assertion 失败，第三轮测试和 v72
候选都没有启动；两轮不组成原要求的三轮组，也不更新正式成绩。
原 outer summary 的 `completed=false`、`source_restored=false` 和异常原样保留。

问：这是新的 kernel fault 吗？

答：本次失败发生在启动第三轮 test child 前的 HEAD 检查。branch ref 从
`6f656cf5b02b2f1acf723949652f91b48e2933d7` 前移到
`e99dbac65a92ed23f87df4d99bd38f655a923a3b`，因此旧固定 assertion 触发；
本次证据没有显示新的 kernel fault。独立只读核验确认生产源码仍是 v60，SHA256
为 `8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`，
其余五个实际输入准备、调用和测试文件仍与中断前一致。
实际 tracked 工作区相对旧 HEAD 的差异只有本轮 GEMM1 文件；当前两个暂存文件
`moe_apreshuffled_quant_scatter.py`、`moe_kernels.py` 保持原样，没有覆盖、unstage
或提交。重新接续时，本地和远端 branch ref 均观察为 e99d。

`source_restored=false` 不能当成源码实际已被替换：它来自 finally 的 HEAD assertion
阻止 swap；随后的独立只读 source hash 核验才证明实际仍是 v60。
旧 summary 没有修改成成功。本次 checker 观察没有自己的 controller/waiter/KFD 残留。

失败和 partial evidence 已归档、下载、逐 member SHA256 核验并导入：34 members、
234222 compressed bytes，archive SHA256 为
`3823da39440d55c9c456e74fd082e5a3e344f9b47d2ebef45266eaa53c89324a`。

```text
TASK/logs/f3_v72_stable60_v2_HEAD_change_post_check_20261026_v1.json
TASK/logs/f3_v72_stable60_v2_HEAD_change_archive_verified_local_20261026_v1.json
TASK/logs/f3_v72_stable60_v2_HEAD_change_archive_imported_local_20261026_v1.json
```

### 新 label/v3：只更新已观察 HEAD，并增加 caller source gate

v72/v73 新工具已生成、上传并完成 12 个工具文件的 SHA256 检查。
正常 measurement `child`、`observe`、`save` 的实际 AST 与原工具一致；
`environment()` 仅改变 expected HEAD literal，另在其 top-level `other_shas` 加入
`moe_kernels.py` 的 SHA256 门禁。outer `swap/save/stop` AST 保持，outer environment
仅改变 HEAD literal；receipt 明确记录这一限定，没有宣称整个 environment AST 未变。
旧 label、原失败和 partial 读数保持，不复用 partial 轮次。

原 timer、精度 gate、uniform/auto/seed0、warmup5/iters20、A-preshuffle=1、pin=0、
两阶段完整 ELF 和固定 v60/v72/v60 各三轮计划保持。测前仍要求 11 次连续全机空闲
观察、每 6 秒一次、总跨度至少 60 秒；每个 compiler/test/profiler child 仍是
`timeout --signal=TERM --kill-after=2s 118s`。测量中不上传、不远端编译、不额外查询 GPU。

新 v72 对照已启动，初始遇到外部 GPU 任务，waiter 自动重置连续空闲计数并继续等待。
这个等待不属于超时的 GPU test child，不终止外部任务；本节未提供 v72 新性能成绩。
v73 新性能工具已就绪，待当前对照退出并恢复 v60 后执行。

```bash
# a07-3 主机入口；每个实际 child 在 dsr1_ep4 内运行
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v72_vs_v60_comparison_stable60_20261026_v3.py
# 后续待执行
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v73_vs_v60_comparison_stable60_20261026_v3.py
```

### v74 本地工具接续

v74 source 和 1024 组已通过的 loader AST 检查保持。新 CPU controller/v2 只更新
expected HEAD、增加 `moe_kernels.py` source hash gate；`child/swap/save/stop` AST、
原 caller 编译、两分钟 timeout、实际 ELF/ISA/IR 绑定与 native gate 保持。
新 label/cache 与旧文件分开；尚未上传、远端编译或 dispatch，没有 ELF/native/
设备精度/性能成绩。生产源码仍是 v60，待性能阶段结束后继续本候选准备。

本节只先在本地独立 CAS 追加；远端待性能阶段结束后追加缺失的前一 chunk 和本节，
保留两端各自的 QA 历史。英文 commit/push、主机 pull 和最终复测仍待收尾。

## 2026-10-26：v74 实际对象、原 native 和四项完整 MoE 精度通过

问：汇报下进展，然后继续。

答：生产保留 **v60、pin=0**；正式三轮 GEMM1 中值仍是 **81.6604 μs**，
距 70 μs 目标差 11.6604 μs。本阶段完成 v74 的编译、实际 ISA 绑定和完整精度验收，
没有把这些非计时运行中的数字作为新性能成绩。v72/v73/v74 尚须完成合格性能对照。

### 利用外部占用时段完成非计时准备，计时队列没有并发执行

v72 新 v3 的第一轮始终停在连续空闲等待，尚未启动 benchmark/audit child。
为利用外部 GPU 任务反复占用的等待时间，只向自己的 normal helper PID1370116
发送 SIGSTOP；核对唯一 argv、`State:T`、未生成 normal summary、没有 benchmark/audit
进程，并保存 `f3_v72_wait_phase_pause_for_v74_untimed_20261026_v1.json`。
没有向外部进程发送 signal。原 controller、waiter、计时程序和 metadata 均未修改。

暂停后才上传并运行 v74 的非计时编译、完整精度。恢复计划要求：全部 source 恢复到
已核验 v60；两项精度 controller 都完成并恢复 source；重新执行独立的 11 次、每6秒、
至少60秒全机空闲门禁；核对原 helper 的 PID/argv/暂停状态后才 SIGCONT。
恢复程序不在 SIGCONT 后查询 GPU，正常每轮仍保留原完整 idle/ownership 检查。
这没有修改原 timer、warmup/iters 或 precision gate。

### v74 的实际编译结果与依赖验收

v74 仍只把原 compact/pin0 loader 的 ScaleA 请求从偶数 iteration 改到奇数，
ScaleB 保持偶数；所有地址、请求数量、operand、数学、原 guard、TDM/wait/barrier 保持。
source SHA256 为 `2673f71630ffcd13d04bcb0ea948e517bad8412e2d6bde676192ed58faa2da6c`。
CPU/controller v2 固定当前 e99d HEAD，并增加 caller hash gate；非 environment 的
`child/swap/save/stop` AST 保持。原 v1 工具没有运行或覆盖。

| 项目 | 实际结果 |
|---|---|
| GEMM1 full ELF SHA256 | `d93735b82aabfd3ae36bb4960d5fc96749dc9ecade2382946aeb86c39ac1b8f0` |
| 资源 | 818 VGPR、56 SGPR、327680 B LDS，private/spills0 |
| static executable instructions | 6336 |
| 原 caller compile child | 15.136388314 秒；target GEMM1 未 dispatch，input preparation 可用其他 GPU kernel |
| inspect child | 0.113859563 秒 |
| full ELF/逐 machine word/ISA/IR/原 argument+global signatures | 全部绑定通过 |
| 原 bounded DS/WMMA model | 6897 次 transfer、0 issues |
| original /WMMA_only stress | 均0 issues，worker分别1.2881789 /1.4379464秒 |

原 native 和 stress 在 Windows 本地对已下载、逐 member 核验的实际 ELF/ISA 执行，
没有 GPU dispatch；transfer 规则保持。v74 比 v60 的798 VGPR增加20，静态指令6334→6336。
这些编译差异不能直接证明 occupancy、bank conflict 或性能变化；仍需正常性能对照。
bounded model 通过不构成精确硬件模拟，也不解释或修复历史 v63/v65 非法访问。

CPU evidence：15 members、396123 compressed bytes，SHA256
`5217083e242e9508283b68b8d3510b2bc9cc78fe59cedcbc70ca27e049018192`，
已下载、全部 member 校验、导入，独立 CPU reviewer 通过。初始 CPU binding 的
`native_gate_not_yet_executed=true` 原样保留；新 `cpu_native_binding` 单独记录后续通过。

### 四项完整 MoE 精度，均与各自已验证 reference 零字节差异

| workload | seed | output bytes | differing bytes | logits_diff | rel_l2 |
|---|---:|---:|---:|---:|---:|
| E64/topk8 | 0 | 22020096 | 0 | 3.6105513278025183e-06 | 0.0026872099842876196 |
| E64/topk8 | 17 | 22020096 | 0 | 3.6202693316254594e-06 | 0.002690816530957818 |
| E65 原 routed fallback/topk9 | 0 | 22020096 | 0 | 3.594893112763491e-06 | 0.002681376878172159 |
| E65 原 routed fallback/topk9 | 17 | 22020096 | 0 | 3.6086727178189905e-06 | 0.0026865052059292793 |

E64 preparation controller合计47.651382271秒；E65两项 child分别21.245898439 /8.977359584秒。
每个实际 child 继续采用118秒 TERM、再2秒 KILL上限，没有超时。两个 controller均
`completed=true/source_restored=true`，GEMM2完整 ELF保持
`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。
reference复用已有核验输出，没有伪装成新执行。E65 fixed shared1和E64/E65共用 compiled
GEMM1的历史要求仍未建立，这四项验证不能声明那些要求已完成。

精度 evidence：58 members、983884 compressed bytes，SHA256
`381e14c88d80403bd00f60d6bc7fc6c21a80daef832e3a2a0f886e56d96bcbc2`，
已下载、逐 member 验证、导入，独立 precision reviewer通过。

```bash
# 已执行；主机 controller，实际 child 在 a07-3/dsr1_ep4
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v74_scale_A_odd_B_even_cpu_20261026_v2.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v74_prepare_20261026_v2.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v74_guard_outside_e65_20261026_v2.py
```

```text
TASK/logs/f3_v74_cpu_local_binding_20261026_v2.json
TASK/logs/f3_v74_cpu_native_binding_20261026_v2.json
TASK/logs/f3_v74_precision_independent_local_review_20261026_v2.json
TASK/logs/f3_v74_CPU_archive_verified_local_20261026_v2.json
TASK/logs/f3_v74_precision_archive_verified_local_20261026_v2.json
```

v74固定 v60/v74/v60 各三轮的60秒idle性能工具已准备，normal caller与 timer、原 precision/
完整 ELF gate保持；尚未运行。packer v3只校正初版工具中的 input binding文件引用到已执行
的v2 binding，旧packer原样保留；没有发生候选性能失败或替换任何测试读数。
原两份等待/HEAD中断 chunk已经在远端独立CAS追加，保留较大的远端QA历史。
本节继续两端独立追加，未执行英文 commit/push/pull，最终复测仍待收尾。

## 2026-10-26：接续 v72/v73/v74 串行对照，实际 ELF 静态计数更正

问：汇报下进展，然后继续。

答：生产继续保留 **v60、pin=0**；既有正式三轮 GEMM1 中值仍为 **81.6604 μs**，
距离 70 μs 目标差 11.6604 μs。v72/v73/v74 的既有 native、E64/E65 seed0/17 完整
MoE 精度结果继续有效，本阶段没有重复这些 GPU 验证，也没有新增合格性能数据。

### 已启动串行队列，先重新等待完整空闲窗口

接续时只读核对两端 `hyg_dsr1_conc6144`、HEAD `e99dbac65a92ed23f87df4d99bd38f655a923a3b`、
远端 boot ID `cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`、v60 源 SHA256
`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`、暂停 helper PID1370116
及其 argv/`State:T`，均与前一 checkpoint 一致。`dsr1_ep4` 为 running。
他人暂存的两个 input preparation 文件保留原状态。

已在 a07-3 主机的固定 tmux `codex-f3-serial-20261026` 启动：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/finish_f3_v72_then_compare_v73_v74_20261026_v1.py
```

队列 SHA256 为 `57b72213e4ae5004b70356ac1350e87bed3951501253804874fc9cc49fafd3ac`，
两端核验一致。恢复程序继续要求独立 11 次、间隔6秒、至少60秒的全机连续 idle，
核对源码、已完成精度记录和唯一原 helper 后才 SIGCONT。
截至读取的 `2026-10-26T05:20:48+00:00` 日志，仍处于 fresh idle 等待，尚未恢复 helper，
normal summary 尚未保存，合格组为0。外部 GPU/KFD 任务间歇启动时自动重置连续 idle 计数，
没有向外部任务发送 signal。计时进程的118秒 TERM /再2秒 KILL限制保持。

队列计划依次完成 v60/v72/v60、v60/v73/v60、v60/v74/v60，各组固定三轮；
双侧改善则停在反向验证点，验证或 kernel 失败也停止并保留失败。
运行期间只读取保存的日志，不传输、不远端编译或额外查询 GPU。

### 更正前一 checkpoint 将 v73 静态计数写成 v60 的错误

对已下载、逐 member 核验的实际 full ELF/ISA/inspection，再次逐 machine word 绑定，得到：

| 版本 | static executable instructions | VGPR | SGPR | LDS B | DS | WMMA | s_wait_alu | s_wait_dscnt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v60 | 6250 | 798 | 56 | 327680 | 786 | 640 | 176 | 68 |
| v73 | 6334 | 798 | 56 | 327680 | 786 | 640 | 176 | 62 |
| v74 | 6336 | 818 | 56 | 327680 | 802 | 640 | 191 | 83 |

三版 private/VGPR spills/SGPR spills 均为0。前一 checkpoint 所写“v60 为6334条、
v74 相对增加2条”不准确：**6334条属于 v73；v74 相对 v60 增加86条**。
保留旧记录，在本节明确更正；源、实际 ELF 和既有精度结果没有改变。

相对 v60，v73 增加81条 `s_set_vgpr_msb`、减少6条 `s_wait_dscnt`，并改变部分地址计算
的 dual/scalar 合并。v74 的 `ds_load_2addr_b32` 和 `ds_load_2addr_stride64_b32` 各减少8条、
`ds_load_b32` 增加32条；这是部分成对读取拆成单条的实际编译变化，DS总数增加16。
v74 的 `s_wait_alu`、`s_wait_dscnt` 各增加15，VGPR增加20。
源码 loader 的逻辑读取/结果保持，不应把它说成实际机器指令或物理 DS 请求数量完全不变。

这些都是静态对象计数，不能据此直接推断动态请求数、occupancy、cache 行为或 kernel μs。
后续仍由原入口正式对照决定收益；没有把本次 CPU 分析当作性能测试。

```text
TASK/analyze_f3_v60_v73_v74_bound_static_20261026_v1.py
TASK/logs/f3_v60_v73_v74_bound_static_local_20261026_v1.json
```

### 在本地准备固定反向对照，不更改正在运行的工具

v72/v73/v74 的反向 controller/reviewer 均已在 Windows 本地生成、syntax检查及AST核对通过。
保留原 measurement loop、controller functions、原表/ELF validation loop、reviewer functions，
仅反转组顺序为候选/v60/候选，并以中间 v60 分别比较两侧候选。
每组三轮、原 timer/warmup/iters/precision、完整 ELF/source 和60秒 idle gate 保持。

反向执行前必须存在正向双侧改善及其独立原表/ELF复核记录；自动采用仍为 false。
本阶段只作本地准备，尚未上传或运行反向，也没有修改生产 source。

```text
TASK/make_f3_v72_v73_v74_reverse_stable60_local_20261026_v1.py
TASK/logs/f3_v72_v73_v74_reverse_stable60_tools_local_20261026_v1.json
```

v3 的加载实例相关 I-cache 波动已有此前直接证据，不重复诊断；v63/v65历史非法访问仍未唯一定位。
E65仍是原 routed fallback；本阶段不将既有验证表述为 fixed shared1 或 E64/E65 共用 compiled GEMM1 已完成。
英文 commit/push、远端主机 pull 和最终复测继续待收尾。
本节先在本地独立 CAS 追加，远端待性能阶段结束后追加，保留两端各自 QA 历史。

## 2026-10-26：完成手头对照后，专门修复 v3 的加载地址敏感性

问：做完手头实验之后，暂停尝试新的优化方法，针对 v3 因加载地址产生的快慢分档完成修复和验收。

答：先完成已经启动的 v72/v73/v74 串行对照；只有出现正向双侧改善的候选才补必要的反向验证。之后暂停其他新优化，转入 v3 的 code footprint/layout 修复。当前生产仍为 v60、pin=0，既有正式三轮 GEMM1 中值为 81.6604 μs；正在运行的 v72 前侧 reference 只完成了一轮 82.0841 μs，尚未形成完整候选结果，不能据此采用任何候选。没有新增 MoE e2e 性能。

修复验收要求如下：

- 保留有效计算、无效行跳过、结果精度及已有支持范围。v3 的 4/6/7/8 行覆盖路径是需要保留的计算语义；不能将 v72 前六个 rep 始终执行直接视为完整保留这些收益。
- 使用同一 ELF、相同输入、原 timer，覆盖实际 HSA `load_base` 的 `group = (load_base >> 16) & 3` 的 0–3。每个加载实例均记录，分别统计各组正常计时分布，不能重新加载后只选择最快实例。
- counters 对照单独执行，检查慢组的 I-cache miss 放大是否降低。此前 rocprofv3 外层下失真的原 timer 读数不能作为 kernel 性能成绩。
- 同时考察快档速度和慢档上界。优先缩小、重排热点机器码；有必要时再验证 layout/padding/alignment。函数 alignment 不等同于控制实际 GPU code 基址。
- 不重复整套已完成的根因诊断，不将尚未验收的修复描述为消除了所有波动。

现有 v3 ELF 为 `c29d97ab809d41e6cf028d9c478fb77ba7bfebbf25d7879788a4a616f53cf0de`，`.text=97408 B`；source snapshot 为 `TASK/snapshots/f3_m16_wg_uniform_pin0_boundary8_v3.py`。v45/v60 的约 44 KiB footprint 提供了减少跨段地址敏感性的实例，但此前 v35/v36 的具体 body 收敛实验回退，不能只凭源码变短判断修复收益。v30 instruction prefetch 的候选中值 89.2589 μs、前后 reference 78.3183/79.1279 μs，没有显示收益，不直接重做或当成修复。

本地硬件依据：`mi400_hw_wiki/raw/papers/mi400_hd_txt/block/sq/MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2 记载 MI400 SQC I-cache 的 64 KiB、4 banks、4 ways、32 sets/bank、128 B line，以及 `Set[4:0] = address[13:9] ^ address[18:14]`。精确 bank hash 和实际 live residency 尚未由现有资料与采集唯一确定；布局静态推导必须由不同加载地址下的实测支撑。

本阶段只做本地记录和已有代码/ELF 的静态核对。串行性能对照期间继续只读取保存日志，不并发 SFTP、远端编译或额外 GPU 查询；外部任务占用时原 waiter 每6秒继续等待，全部 pin=0、原 warmup5/iters20 和每个实际 child 的118秒 TERM/再2秒 KILL 限制保持。英文 commit、push、远端主机 pull 和最终复测仍是已授权的后续收尾任务。

## 2026-10-26：v72/v73 失败封存与 a07-3 重启期间的本地工作

问：汇报下进展，然后继续；等 a07-3 重启后我通知你，先干点别的。

答：遵照最新安排，暂停 a07-3 的全部远端操作，包括 SSH、状态查询、上传、编译和 GPU 测试，等待用户明确通知重启完成。先在 Windows 本地补齐失败记录，并准备仅调整 v3 行覆盖 selector 布局的候选和等价性检查。当前本地生产 source 仍为 v60、pin=0，已验收的正式三轮 GEMM1 中值为 **81.6604 μs**，本轮没有新的 MoE e2e 性能，也没有采用新候选。

### 手头性能对照的有效结果和失败边界

| 对照 | v60 前侧 GEMM1 三轮中值 μs | v60 前侧 GEMM2 三轮中值 μs | 候选状态 |
|---|---:|---:|---|
| v72 vs v60 | 81.3986 | 64.3502 | 候选首轮报告失败，没有 accepted round；不接受局部计时 |
| v73 vs v60 | 81.4410 | 63.6731 | 首轮完成，第二轮非法访问；整组未完成，不接受候选性能 |

上述 v60 reference 均使用原入口、原 timer、E64/T1536/topk8/K7168/inter2048、uniform/auto/seed0、warmup5/iters20，pin=0、A-preshuffle=1、`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`。每轮计时前保留全机连续11次 idle 检查、每6秒一次、跨度至少60秒的记录，并由本地独立复核直接核对原表、实际 ELF/source 和 idle 日志。它们是本次失败对照的有效 reference，不能替代此前生产正式成绩，也不能补成候选组的结果。

v72 候选在原 sanity gate 通过后，GEMM2 profiler 未返回有效数据，原测试随后报：

```text
TypeError: unsupported format string passed to NoneType.__format__
```

child returncode=1，耗时8.676555452984758秒。没有最终原表，没有 accepted round；局部出现的 GEMM1 83.23 μs 不作为成绩。未修改 timer、精度标准或测试报告源码，也没有用重跑替换失败。controller 已恢复 v60，测后无 KFD owner。该记录不能解释为 kernel 已发生非法访问。

v73 首轮原表为 GEMM1 82.1219 μs、GEMM2 64.7527 μs，精度 gate 通过；保留这一轮原记录。第二轮在 ATen `nonzero/index_put` 路径异步上报 `hipErrorIllegalAddress`，child returncode=134，耗时8.376593922963366秒，没有 sanity 或最终表。出错 kernel 尚未唯一确定，不能归因于 v3 I-cache，也不能归因于后来看到的外部 GPU 占用。候选组不完整，因此不接受其性能、不自动重跑，controller 已恢复 v60。之后检查发现7个外部 KFD owners，逐个 `/proc/PID/cgroup` 核验均不属于 dsr1_ep4，0 unavailable；没有杀死外部进程。

两个失败 archive 均已在本地逐 member 验证 manifest、SHA256、文件大小后导入，并完成独立原表/ELF/idle 复核：

```text
TASK/logs/f3_v72_stable60_timer_failure_20261026_v1.tar.gz
59 members, 375540 B
SHA256=08860bcbe999e1f44aa0c5896ea387cda0b7706d0c5f5e6ebdcf5fb46f0baeed
TASK/logs/f3_v72_timer_failure_independent_local_review_20261026_v1.json

TASK/logs/f3_v73_stable60_runtime_failure_20261026_v1.tar.gz
71 members, 467078 B
SHA256=941dacea943efba37deb522ec223501d69f6eeb8780139a31df6886f3eb9c7c7
TASK/logs/f3_v73_runtime_failure_independent_local_review_20261026_v1.json
TASK/logs/f3_v73_failure_KFD_container_postcheck_20261026_v1.json
```

### v74 的中断状态

07:13 UTC 在主机 tmux 启动 `run_f3_v74_vs_v60_comparison_stable60_20261026_v2.py`。最后可见内容仅为 reference group label 和 `V74_V60_FIXED_GROUP_START`；当时外部 GPU 有占用，随后 SSH connection reset。**尚未取得 v74 的最新 summary，不能断言其后没有执行 child/round，也不能假定远端生产 source 一定已恢复。**

用户通知重启完成后，先只读核验新 boot ID、branch/HEAD、dsr1_ep4、保存日志、controller 状态、远端 source/hash、cache 和残留，再封存这一中断实验。不会直接重启旧 label/controller。重启前最后确认的 branch/HEAD 为 `hyg_dsr1_conc6144` / `e99dbac65a92ed23f87df4d99bd38f655a923a3b`，boot ID 为 `cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`。

### 本地专项修复方向

已验证的旧 v3 ATT 有64个 active waves，其中42个执行6-rep body、22个执行7-rep body。6-rep 的 observed mainloop line footprint 为16512 B，ELF PC line 范围 `0xc980–0x10980`，跨越64 KiB 边界；7-rep 为18048 B，范围 `0x7a00–0xc000`。这些是已有 trace 的实际 PC 记录，不能当成实际 live cache residency 或精确 bank 冲突测量。

先准备只改变 selector 分支布局的候选，保留4/6/7/8行覆盖，以及所有 mainloop、loader、WMMA、K loop、wait、barrier、epilogue 和接口。源码顺序不保证 LLVM 实际机器码布局；本地 AST/区间检查也不能代替 native、完整精度和 group 0–3 的正常计时/counters 验收。目前没有已验收的 v3 加载地址修复方案。

E65 fixed shared1、E64/E65 共用 compiled GEMM1 的历史要求仍待完成，已有 routed fallback 精度通过不能描述成这些要求已完成。英文 commit、push origin、远端主机 pull 和最终复测继续保留为后续收尾任务，不提交另外两份他人的暂存修改。本节先本地独立 CAS 追加，远端待重启通知和状态核验后独立追加，保留两端各自 QA 历史。

## 2026-10-26：本地准备 v75 selector 布局候选，生产仍保留 v60

问：等 a07-3 重启后我通知你，先干点别的。

答：本轮只做本地工作，已生成 v75 的 v3 selector 布局候选，并完成独立静态复核。没有访问 a07-3，没有上传、编译或启动 GPU 程序。v75 尚未形成实际 ELF 或性能成绩，不能描述为已修复 v3 的快慢分档；当前生产仍为 v60、pin=0，已验收 GEMM1 三轮中值仍为81.6604 μs，没有新 MoE e2e 数据。

### 修改内容及静态证据

v75 只把 v3 selector 的首层 `4 * M_ROW_STEP >= mn_oob` 改成反向的 `4 * M_ROW_STEP < mn_oob`，将原6/7/8-rep 子树放在 true 分支、原4-rep body 放在 else 分支。源码中的叶子顺序由4/6/7/8变为6/7/8/4；没有合并 body，没有增加每条路径的比较次数。

| `mn_oob` 区间（`M_ROW_STEP=32`） | v3/v75 的 `m_reps` | 两版执行的比较次数 |
|---|---:|---:|
| signed int32 最小值至128 | 4 | 1 |
| 129至192 | 6 | 2 |
| 193至224 | 7 | 3 |
| 225至signed int32 最大值 | 8 | 3 |

独立 reviewer 从实际 AST 解析每个比较，把 true/false 条件转换成精确整数区间，沿分支树求交集。结果覆盖完整 signed int32 域，无遗漏或重叠：两版对每个输入调用相同 body、传入相同参数、执行相同数量的 predicate；`M_ROW_STEP=16/32` 都通过区间证明。对实际 interleaved M-wave 布局，还检查了 `mn_oob=0..256` 的全部257种有效行数：两条 M-wave 的行索引互不重复，所有有效行完整覆盖。

恢复 selector 后，新旧整份源码的 AST 完全一致，selector 外的源码字节也完全一致。原 `run_mainloop`、WMMA、TDM、loader、K loop、wait、barrier、epilogue、外层准入条件、fallback 和接口保持；6-rep 的 `depctr_va_vdst(8)`、7-rep 的 `depctr_va_vdst(0)`、8-rep 的 loader boundaries 保持。没有增加 expert/shared-expert 编译 specialization。

这些检查证明的是源码与 selector 语义，不能证明 native 寄存器分配、指令布局或 GPU 数值结果。源码叶子顺序不保证 LLVM 发射顺序，不能据此声称热点已移出64 KiB 边界，也不能声称 `.text` 已缩小。

```text
候选：TASK/snapshots/f3_v3_hot_selector_layout_v75.py
SHA256=0b956a0de360c57a962ef29b6b301ce05a189acc7fc8b18ed124ce5581eb0639

原始 v3 source SHA256：
612308136fba5459a9c3120fee93ead5f446c37c5fa0557acf8a9b69afce6f95

保持的本地生产 v60 source SHA256：
8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008

TASK/make_f3_v75_v3_hot_selector_layout_local_20261026_v1.py
TASK/review_f3_v75_v3_hot_selector_layout_local_20261026_v1.py
TASK/logs/f3_v75_v3_hot_selector_layout_source_20261026_v1.plan.json
TASK/logs/f3_v75_v3_hot_selector_layout_source_20261026_v1.diff
TASK/logs/f3_v75_v3_hot_selector_layout_source_20261026_v1.independent_local_review.json
```

本地首次生成与复核命令如下，已执行通过。工具以独立 checkpoint 保存证据，存在同名输出时拒绝覆盖；再次实验使用新 label，不覆盖旧记录。

```powershell
cd C:/Users/yanguahe/Documents/code/wk_sp1/aiter_dsr1_e65
python .codex_tmp/hyg_f3_followup_20261024/make_f3_v75_v3_hot_selector_layout_local_20261026_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v75_v3_hot_selector_layout_local_20261026_v1.py
```

### 用户通知重启完成后的执行顺序

1. 先只读核验 a07-3 的新 boot ID、主机 branch/HEAD、dsr1_ep4 状态、v74 的保存日志/controller/source/hash/cache 和残留。v74 的最新 round 状态与远端源码恢复情况目前未知，不直接重启旧 controller。
2. 封存 v74 中断实验，再为 v75 分配独立 compile/native/correctness/measurement label 和 cache。仅上传任务所需文件并核验 SHA256；Git 在主机执行，编译和 kernel 验证在 dsr1_ep4 执行。
3. 先看实际 ELF、`.text`、寄存器、private/spills、目标 ISA 和4/6/7/8 body 的 PC 范围，确认源码重排实际产生了什么变化；不能只按源码长度或顺序判断收益。保留原 native 退休距离和精度 gate，任何非法访问或报告失败均原样封存。
4. 使用同一候选 ELF、相同输入、原 timer，覆盖真实 HSA `group=(load_base>>16)&3` 的0–3，记录每个加载实例和每组正常计时分布；不重新加载后只挑最快实例。所有性能测试保持 pin=0、A-preshuffle=1、explicit VGPR partition=0，计时前连续11次全机 idle 检查、间隔6秒、跨度至少60秒；计时期间不并发编译、传输或额外 GPU 查询。
5. counters 单独采集并绑定实际 ELF/load base，检查慢组 miss 放大是否降低；不能把 rocprofv3 外层失真的 timer 当性能。只有同时满足 native、完整精度、快档速度和慢档上界要求，才考虑采用候选。每个实际 compiler/test/profiler child 继续 `timeout --signal=TERM --kill-after=2s 118s`。

另外两份他人的暂存修改保持原样；本轮未执行 Git add/commit/push，也未修改生产 kernel。英文 commit/push、远端主机 pull、最终性能复测及历史 E65 fixed shared1/E64-E65 共用 compiled GEMM1 要求继续保留为后续待办。本节在本地独立 CAS 追加，远端等重启通知和核验后独立追加，保留两端各自 QA 历史。

## 2026-10-26：a07-3 恢复，v75 native/完整精度通过，首组正常计时仍缺地址覆盖

问：a07-3 ok。

答：按通知恢复 a07-3 / dsr1_ep4 的工作，继续专项验证 v3 加载地址快慢分档修复，保持 pin=0、A-preshuffle=1、`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`。两端生产源码仍为 v60；没有采用 v75，没有新 MoE e2e 性能。此前生产正式 E64/T1536/topk8 GEMM1 三轮中值81.6604 μs保持为历史成绩。

### 重启与未完成实验的边界

新 boot ID 为 `4d22137c-d3aa-4a81-ad98-44cb4786b2fe`。主机 branch/HEAD 仍是 `hyg_dsr1_conc6144` / `e99dbac65a92ed23f87df4d99bd38f655a923a3b`；dsr1_ep4 运行，镜像 `rocm/fw-bringup:gfx1250-atom-20260930-na`。六份关键源码逐文件 SHA256 核验与重启前一致。

旧 v74 controller 最后停在 `F3_WAIT_IDLE ..._r1`，无 r1 child log/exit/meta，0 qualified groups。旧 summary 的 `completed=false`、`source_restored=false` 原样保留；另以重启后的实际源码 hash 确认当前为 v60，不能把旧字段改成 true。idle log 中402个 NUL原样保留。没有发现本任务残留进程，没有重启旧 label。

### v75 的实际编译与 native 检查

v75 只重排 v3 行覆盖 selector，完整 signed int32 域中的 body 选择与比较次数不变，selector 外源码字节、mainloop 和准入保持。实际 GEMM1 ELF 为 `efd6cff7bc9c97d538ccd9b9696604c09f6bf548f6e126eb1fe916094ae7f327`，`.text=97024 B`、VA=`0x1a00`、VGPR=804、SGPR=60、LDS=327680 B，private/spills=0。原 v3 `.text=97408 B`、VGPR=799；仅缩小384 B，不能据此宣称修复。

编译 child 19.6434秒，inspection 0.1639秒。机器码字节、ISA/ELF/IR 与 caller 参数绑定通过；原 bounded DS/WMMA transfer model 0 issues，23234 CFG iterations，source IR 显式 wait8/wait0位置均保持各20个。保守 retirement stress 的 original 和 WMMA_only 两种 ordering 均0 issues，child分别约3.48/3.23秒。这些是静态模型结果，不替代 GPU 精度检查。

原 v3 的48-WMMA loop有一组 PC `0xfd80–0x103f8` 跨64 KiB；v75对应 loops移到 `0x10d90` 之后，静态 intervals均不跨段。这里不是 ATT live footprint 或实际 cache residency，不能当成消除 I-cache 冲突的证明。本地 SQC MAS §2 的 set hash依据仍为 `Set[4:0] = address[13:9] ^ address[18:14]`，精确 bank hash和 live residency仍未唯一确定。

### 完整 MoE 精度复核

| workload | seed | logits_diff | rel_l2 | 输出逐字节差异 |
|---|---:|---:|---:|---:|
| E64/T1536/topk8 原完整 MoE | 0 | 3.6105513e-06 | 0.0026872100 | 0（对照 v60） |
| E64/T1536/topk8 原完整 MoE | 17 | 3.6202693e-06 | 0.0026908165 | 0（对照 v60） |
| E65/T1536/topk9 原 routed fallback | 0 | 3.5948931e-06 | 0.0026813769 | 0（对照已保存 routed fallback） |
| E65/T1536/topk9 原 routed fallback | 17 | 3.6086727e-06 | 0.0026865052 | 0（对照已保存 routed fallback） |

全部原精度 gate通过，22,020,096 B BF16输出的原 log、meta及汇总已独立复核。E65 对照实际来自 `f3_v49_wait0_guard_outside_e65_20261025_v1_base_s0/s17`，不能描述成 E65 对 v60 的专门验证。GEMM2完整 ELF仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。E64 preparation controller总耗时51.6075秒，每个实际 child继续118秒 TERM/再2秒 KILL。精度运行耗时不作为性能。

历史 E65 fixed shared1 与 E64/E65共用 compiled GEMM1要求仍未完成；本次 E65 fallback通过不能替代这两项验收。旧 preparation controller继承字段 `environment_only_expected_HEAD_literal_changed=true` 保留原样，准确的生成 proof记载实际变化为 boot literal；不改写已运行原记录。

### 首组加载地址正常计时

`f3_v75_load_groups_normal_20261026_v1` 完成，child耗时10.92997秒。使用同一 ELF、相同输入、同一 host function/CallState/ExecutionEngine和原 timer，保留5个加载实例及全部28次 GEMM1 profiler窗口；每个窗口完整20次 kernel事件，依原 timer去掉首个迭代后以19次均值独立复算一致。原 memory-profiling call、5次warmup及20次辅助 Event call保持，未修改计时或计算。

| 实际 HSA load_base group | 加载实例数 | 原 timer次数 | 中值 μs | 最小 μs | 最大 μs |
|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 28 | 81.153053 | 80.201211 | 81.613842 |
| 0 / 2 / 3 | 0 | 0 | 未覆盖 | 未覆盖 | 未覆盖 |

`group=(public_HSA_load_base>>16)&3`，不能由 HIP module handle或host地址代替。实例实际基址依次为 `0x7135df010000`、`0x7135a6e10000`、`0x7135a6dd0000`、`0x7135a6d90000`、`0x7135a6d50000`。四个clone基址步长0x40000，使5个地址都落在group1；不同地址不等于覆盖不同group。

计时前全机连续11次idle、间隔6秒且跨度至少60秒；计时期间无额外 GPU查询、编译或传输，测后1秒和6秒检查均idle。所有实例 loaded `.text` 与冻结 ELF逐字节相同，输出 payload/scale hash一致，source已恢复v60。这仅为group1正常计时结果，不证明修复；其余group和独立 counters尚待采集。同boot v3对照已串行完成，尚待独立复核。

复现入口（controller在主机，实际测试进入dsr1_ep4）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v75_load_address_acceptance_20261026_v1.py normal --variant v75 --label f3_v75_load_groups_normal_20261026_v1 --runs 1
```

该 label为已完成记录，不覆盖；后续用新label。验收必须覆盖group0–3的全部实例正常计时，并单独采counters，不重新加载后只挑最快实例。

```text
TASK/logs/f3_v75_E64_E65_precision_evidence_20261026_v1.tar.gz
29 members, SHA256=01cae0e7719e0565fd2d03363a404112ac38aefccab7cf2902c214a152497f36
TASK/logs/f3_v75_precision_independent_local_review_20261026_v1.json
TASK/logs/f3_v75_load_groups_normal_20261026_v1_evidence.tar.gz
73 members, SHA256=78b3fb247143a354275e6b6478de0005fcd518b8a4886ccb98970bcfeb80532f
TASK/logs/f3_v75_load_groups_normal_20261026_v1.independent_local_review.json
```

三份文档继续独立 CAS追加，保留两端各自原有问答；另外两份他人的暂存修改不动。英文commit/push origin、远端主机pull及最终性能复测仍是已授权的后续收尾事项，未声称已完成。

## 2026-10-26：排除 v75，完成 v76 精度复核并准备同进程地址组对照

问：汇报下进展，然后继续。

答：继续专项修复 v3 因 GPU code 加载地址产生的快慢分档，暂停其他新优化方向。v75 已有同进程快慢组证据，排除为修复方案；v76 只调整行覆盖 selector 的叶子顺序，机器码检查与 E64/E65 routed fallback 的四项完整 MoE 精度验证已通过。新正常性能及 counters 仍待地址组对照，生产尚未采用 v76，没有新的正式 MoE e2e 成绩。生产保留版本仍为 v60，历史正式 E64/T1536/topk8 GEMM1 三轮中值 81.6604 μs。

### v75 的有效分组证据与被排除记录

`f3_v75_valid_A_normal_20261026_v1` 的 r2 在同一进程、同一实际输入下出现明显分档：

| 实际 HSA load_base group | 实例数 / 原 timer 数 | GEMM1 中值 μs | 范围 μs |
|---:|---:|---:|---|
| 0 | 1 / 16 | 79.1375 | 78.6284–79.9356 |
| 3 | 4 / 12 | 88.9923 | 88.0802–89.8789 |

r1/r2 的全部 28 个窗口分别通过原 timer 独立复算。r3 测后发现外部 GPU/KFD 活动，整轮耗时排除；原 controller `completed=false` 保持不变，不把部分有效数据改写为整组通过。这足以排除 v75 为消除地址敏感性的修复。不同进程内的 expert 行排列仍可能因 routing 的 atomic slot reservation 不同，不能把同 seed 当作逐字节相同的实际输入。

归档 `TASK/logs/f3_v75_valid_A_normal_20261026_v1_evidence.tar.gz` 的 SHA256 为 `748f98b07443fe07f6600a85f4c3a8c0dd5757c055d598a7d2982c8c0f4648f3`；独立复核为 `TASK/logs/f3_v75_valid_A_normal_20261026_v1.partial_round_review_local_v1.json`。

此前 v3/v75 paired 诊断的 v1 失败是预备输出快照计数错误：实际 39 份，旧检查期待 38 份。失败记录原样保留，不接受其性能。修正后的 v2 controller 完成，但第 49 个 GEMM1 窗口 `forward_v3_clone16` 有 20 个 CPU launch events、0 个 GPU events，Chrome trace 同样没有 kernel events，原 timer 返回 0.0 μs；native cluster launch delta 仍为 46，输出正确。该轮正常性能验收失败，原 controller `completed=true` 不改写。日志中的 `ROCTracer produced duplicate flow start: 1` 不能唯一确定丢失原因。

v2 归档 SHA256 为 `e2c8841e1f37e3b63ffd0fc1da89293e08f6c8fceab727059f7d8b66b951505a`。不通过新增平均、补造 duration 或挑选其他窗口绕过缺失事件。

### v76 仅改变 selector 顺序及由此产生的布局

源文件：`TASK/snapshots/f3_v3_hot_pair_first_layout_v76.py`，SHA256=`81a599100f7fb5dc3e565976e2d18491825da01e945d6b741d3d93408aa710bf`。叶子顺序由 4/6/7/8 改为 4/8/6/7：

```python
if 4 * M_ROW_STEP >= mn_oob:
    run_mainloop(..., m_reps=4)
elif 7 * M_ROW_STEP < mn_oob:
    run_mainloop(..., m_reps=8)
elif 6 * M_ROW_STEP >= mn_oob:
    run_mainloop(..., m_reps=6)
else:
    run_mainloop(..., m_reps=7)
```

完整 signed int32 域的 body 选择等价，257 种有效行覆盖检查通过；原 mainloop、TDM loader、wait/barrier、WMMA、K-loop、epilogue、准入及 fallback 保持。比较次数并非全部相同：4-rep 为 1→1，6-rep 为 2→3，7-rep 为 3→3，8-rep 为 3→2。没有新增 expert/shared expert specialization。

实际 GEMM1 ELF SHA256=`9cc6f51d232a95f34aa7827da7b09f2344e167982250033f2924e8550e75f918`，`.text=97664 B`、VGPR799、SGPR60、LDS327680 B，private/spills=0。原 bounded DS/WMMA model 0 issues，25967 CFG iterations；original 和 WMMA_only retirement stress 均 0 issues，原 caller signatures 与 GPU dispatch ABI 相同。

对应 48/56-WMMA backward edges 的静态 PC 范围分别为 `0x7e80–0xb300` 和 `0x2f00–0x6900`，都在第一个 64 KiB 段。这是实际 ELF 的静态布局结果，不能当作 live I-cache footprint、residency 或已修复性能的证明。硬件依据仍为本地 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2 的 `Set[4:0] = address[13:9] ^ address[18:14]`；精确 bank hash 和 live residency 尚未唯一确定。

CPU 归档 SHA256=`48df68eb1015511dc5f5976283048ac84c9ecddc54d49084053cd1c5da0d45fa`；汇总为 `TASK/logs/f3_v76_native_and_layout_local_20261026_v1.json`。

### v76 完整 MoE 精度验证已独立复核

| workload | seed | logits_diff | rel_l2 | 最终输出逐字节差异 |
|---|---:|---:|---:|---:|
| E64/T1536/topk8 原完整 MoE，对照保存的 v60 | 0 | 3.6105513e-06 | 0.0026872100 | 0 |
| E64/T1536/topk8 原完整 MoE，对照保存的 v60 | 17 | 3.6202693e-06 | 0.0026908165 | 0 |
| E65/T1536/topk9 原 routed fallback，对照保存的 fallback | 0 | 3.5948931e-06 | 0.0026813769 | 0 |
| E65/T1536/topk9 原 routed fallback，对照保存的 fallback | 17 | 3.6086727e-06 | 0.0026865052 | 0 |

每份输出均为 `[1536, 7168]` BF16、22,020,096 B，原测试及精度标准未修改；原记录、日志、stage ELF、summary 和已验证 reference 链已逐项核对。E64 preparation controller 耗时 51.7103 秒；两份 E65 child 也正常结束，controller `completed=true/source_restored=true`，结束时源码 SHA256 确认恢复 v60。精度运行耗时不作为性能。GEMM2 ELF 仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

E65 对照实际来自 `f3_v49_wait0_guard_outside_e65_20261025_v1_base_s0/s17`，不是本轮专门运行的 v60 E65 对照。历史 fixed shared1 和 E64/E65 共用 compiled GEMM1 的要求仍未完成，本次 fallback 通过不能替代这两项验收。

```text
TASK/logs/f3_v76_E64_E65_precision_evidence_20261026_v1.tar.gz
29 members, SHA256=2f70125fb295be5f4e7638faae2fda19f4bae5573ded846dfdc833e4b7eddf21
TASK/logs/f3_v76_precision_independent_local_review_20261026_v1.json
```

### 下一步：原 timer 的同进程、同实际输入地址组对照

冻结 v3/v76 ELF，在同一进程里只生成一次 routing/实际输入，共用原 host function、CallState 和 ExecutionEngine。已验证的 native C proxy 仅替换 GPU function，保留完整 15 参数 cluster ABI；filler 只加载、不 dispatch，全部 target/filler 保留到计时结束。加载计划只依据实际 HSA group 覆盖，不能依据速度选实例。

每个实例正序、逆序各使用一次原 timer；每个 timer 仍为 warmup=5、iters=20，原计时函数 AST 保持。减少的是诊断重复窗口数，不减少每次 kernel 的有效计算。新增缺失 GPU events 门禁：任何正常 GEMM1 窗口没有完整 20 次 GPU events，或原返回值不大于零，都拒绝整轮正常性能。独立 reviewer 分别执行旧 v3/v75 每实例 6 窗口和新 v3/v76 每实例 2 窗口的门禁，没有把旧失败协议放宽为通过。

```bash
# host controller；实际 child 在 dsr1_ep4，全部 pin=0
python3 -u TASK/run_f3_v3_v76_paired_acceptance_20261026_v1.py normal \
  --label f3_v3_v76_paired_normal_20261026_v1 --runs 1 --seed 0
```

上述新 label 已启动 controller，记录时仍在等待外部 GPU0 任务结束，尚无该 label 的性能结果。每 6 秒自动检查全机空闲；性能前要求连续 11 次 idle、跨度至少 60 秒。每个实际 child 使用 `timeout --signal=TERM --kill-after=2s 118s`。计时阶段不并发上传、编译或额外 GPU 查询；不杀外部进程。wait controller 等待时间不等于实际测试 child 时间。该 controller 临时使用保存的 v3 源码以恢复原 host callable，结束后通过 CAS 恢复 v60；本地生产源码保持 v60。

后续单独采 `SQC_ICACHE_REQ/HITS/MISSES/SQ_WAVES`，按 `native_selected_cluster_before/after` 与完整 SDK dispatch 序号绑定所有实例。每个原 perftest 的 46 launches 为 probe1 + warmup5 + auxiliary Event20 + profiler20，仅后 20 次用于 counters 对照；另有 1 次初始 GEMM1 dispatch。SDK hits/misses 是 per-SQ/per-Bank、nondeterministic，事件比率不能当作精确 hit rate。counter 模式的原 timer 及 SDK instrumented span 都不当作正式性能。新 reviewer 为 `TASK/review_f3_paired_counters_local_20261026_v1.py`。

修复验收仍需同时覆盖 v3/v76 的 group0–3 正常计时、有限多进程/seed 复测和独立 counters，比较快档速度、慢档上界与 miss 放大。没有通过前不采用候选。三份文档分别 CAS 追加，保留两端原有问答；两份他人的暂存源码不动、不提交。英文 commit/push origin、远端主机 pull 及 Git pull 后最终性能复测继续保留为已授权的后续收尾事项。

## 2026-10-26：v76 首轮完整地址组正常计时，慢组改善但仍未修复

问：汇报下进展，然后继续。

答：a07-3 的外部任务结束并满足全机连续 60 秒空闲后，v3/v76 的第一轮同进程、同实际输入对照完成。v76 保住了快组速度，把 group3 中值从 90.79295 μs 降到 82.41111 μs，但仍有明确的慢组，因此不采用为完整地址敏感性修复。按事先记录的有限验证计划，停止其余两次正常复测；继续单独采 counters，定位剩余分档。v77 只准备了局部 LLVM layout hint 快照，尚未编译或测试。生产保留 v60，没有新的正式 MoE e2e 性能。

### 所有 group0–3 的原 timer 已独立复核

环境仍为 `hyg_dsr1_conc6144/e99dbac65a92ed23f87df4d99bd38f655a923a3b`、a07-3/dsr1_ep4、boot `4d22137c-d3aa-4a81-ad98-44cb4786b2fe`，pin=0、A-preshuffle=1。目标为 E64/T1536/topk8/K7168/inter_dim2048、uniform random non-balanced routing。

| 地址 group | v3 实例 / timer 数 | v3 中值 μs | v3 范围 μs | v76 实例 / timer 数 | v76 中值 μs | v76 范围 μs |
|---:|---:|---:|---|---:|---:|---|
| 0 | 1 / 2 | 78.615842 | 78.411368–78.820316 | 1 / 2 | 77.973763 | 77.411737–78.535789 |
| 1 | 1 / 2 | 82.439658 | 81.397000–83.482316 | 2 / 4 | 77.957947 | 77.485737–78.008684 |
| 2 | 2 / 4 | 78.412289 | 77.947526–79.294842 | 5 / 10 | 78.133053 | 77.443526–78.727632 |
| 3 | 3 / 6 | 90.792947 | 89.533211–91.742684 | 8 / 16 | 82.411105 | 81.245158–82.969789 |

两版在一个进程里共用同一次 routing、完整 GEMM1 输入、host function、CallState 和 ExecutionEngine，实际 GPU function 与加载 `.text` 逐一绑定。23 个 target 实例和 25 个只加载、不 dispatch 的 filler 都保留到正序/逆序结束，不选择最快实例。全部 46 个 GEMM1 profiler 窗口具有完整 20 个 CPU launch events 和 20 个 GPU events，Chrome trace 也逐一通过；以原 timer 的规则去掉首个迭代、取其余 19 个的均值，复算与每个原返回值一致。

每个原 perftest 有 46 次 native cluster launches，首个窗口前另有 1 次初始 GEMM1 dispatch，合计 2117 次。输入前后完整 hash 一致，47 份输出 payload/scale 快照逐字节一致；loader/HSA observer/function lookup 不进入计时窗口。实际 child 14.13557 秒，测前连续 11 次全机 idle、测后 ownership 和 settled idle 均通过，结束后源码恢复 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。没有本轮测试进程残留。

v76 的 group3 减少 8.381842 μs，降幅 9.231821%；各组中值的 `max/min - 1` 从 v3 的 15.789180% 降到 v76 的 5.712256%。v76 group3 的整个实测范围仍高于快组范围，不能把这 5.7% 差距忽略，也不能宣称已消除地址敏感性。没有新 GEMM2 或 MoE e2e 正式性能。

```text
TASK/logs/f3_v3_v76_paired_normal_20261026_v1_evidence.tar.gz
273 members, SHA256=36e87b8917a4cf93deca5ec71e3603d1b6ad9ccdee9233572fdf890a86146372
TASK/logs/f3_v3_v76_paired_normal_20261026_v1.paired_normal_independent_local_review_v2.json
SHA256=ba413a96ff2e820be52f628c4f6f62cf01f0caec1dd35fbaf4f2c9a9b993e0d2
TASK/logs/f3_v76_first_paired_normal_decision_local_20261026_v1.json
```

### 预先规定的复测停止条件已执行

在取得任何 v76 正常计时结果前，保存 `TASK/logs/f3_v3_v76_finite_validation_plan_local_20261026_v1.json`，SHA256=`1e0af7ebc45ea391c1b995734bb62167d81da3f40940680daf692a132e682861`。计划为 seed0/17/0 三个独立进程、每实例正逆序各一个原 timer；任何有效数据明确出现残余地址分档则排除候选、停止多余正常复测。第一轮已证明这一点，所以 seed17 和第二次 seed0 两轮未运行。这个停止不是丢弃慢窗口或重测挑快；全部第一轮数据都保留。

另按计划启动独立 counters controller：

```bash
python3 -u TASK/run_f3_v3_v76_paired_acceptance_20261026_v1.py counters \
  --label f3_v3_v76_paired_counters_20261026_v1 --runs 1 --seed 0
```

记录时该 controller 仍在等待外部 GPU0 任务完成，没有 counters 结论。每 6 秒自动检查，连续 60 秒空闲后才运行；不杀外部任务。前一轮正常测试结束后已经恢复 v60；counter controller 临时使用 v3 的保存源码恢复原 host callable，结束后仍须按 CAS 恢复 v60。本地生产源码保持 v60。

新 counter reviewer 将按 `native_selected_cluster_before/after` 和 SDK GEMM1 dispatch 顺序核对全部 46 次 launch，仅尾部 20 次用于 `SQC_ICACHE_REQ/HITS/MISSES/SQ_WAVES` 对照。counter 模式的原 timer 和 SDK instrumented span 不作为性能，SDK 的 per-SQ/per-Bank、nondeterministic hits/misses 也不解释成精确 hit rate。profiler、SDK config、rocminfo shim 及实际 rocminfo 保存文件的四个 SHA256 在本轮只读重检中与原记录一致；没有用这项文件检查启动 GPU kernel。

### set hash 的静态推导边界

基于本地 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Page2 的 `Set[4:0] = address[13:9] ^ address[18:14]`，对 64 KiB 对齐基址检查了 8,388,608 对地址：同段 PC 的 set 编号整体 XOR 旋转，段内相对 set 关系不随基址改变。相同段内偏移跨到下一段时，group0/1/2/3 的 set XOR 差异分别为 4/12/4/28。这些是映射掩码，不是冲突数量或 miss 倍数。

已保存的 HSA 观测中 `load_base == load_delta`，`.text` 起点为基址加 `0x1a00`。检查文件为 `TASK/logs/f3_v76_documented_set_rotation_local_20261026_v1.json`。该结果只证明文档 hash 下的地址关系，没有假设精确 bank hash，没有推出 live footprint、residency 或延迟。v76 将两个热点 backward-edge 区间放进第一段的静态结果仍成立；首轮正常数据证明它不足以消除整个 kernel 的残余分档。

### v77 仅准备局部分支 layout hint，未编译

`TASK/snapshots/f3_v3_hot_selector_expect_layout_v77.py`，源码 SHA256=`f2079c872bb3e40f74c88d14d8195fb7bedbff6f5d5c8e307759d33cc0085f39`。以 v76 为父版本，对既有 4-rep 和 8-rep selector 的判定增加 `llvm_dialect.intr_expect(..., fx.Boolean(False).ir_value())` 包装，尝试影响编译器 basic-block placement。原比较、叶子选择与比较次数均保持；去掉两处 expect 包装后整棵 AST 与 v76 相同，mainloop、loader、wait/barrier、WMMA、epilogue、准入和接口不变，没有增加 E64/E65 或数据 specialization。

LLVM 本地 LangRef 的 `llvm.expect` Semantics 明确写的是 `This intrinsic is lowered to the val.`；这是概率提示，不能删除实际进入冷分支的 route。本仓库 fmha 的既有代码使用同一 API 包装方式。v77 当前只在快照文件中，未上传、未编译、未运行；是否进入编译/精度/性能阶段须等 v76 counters。不能因源码加入 hint 就假定机器码、footprint 或性能发生改善。

生产版本仍为 v60，历史正式 E64 GEMM1 三轮中值 81.6604 μs继续作为历史记录。E65 fixed shared1 及 E64/E65 共用 compiled GEMM1 的历史要求仍未完成，本轮 E65 routed fallback 精度通过不替代这些验收。两份他人的暂存修改继续保留；英文 commit/push origin、远端主机 pull 和 Git pull 后最终复测仍是已授权的后续收尾事项。

## 2026-10-27：v77 独立源码复核与编译工具准备，counters 继续等待空闲

问：汇报下进展，然后继续。

答：已验收的 v3/v76 第一轮正常计时结论保持：E64/T1536/topk8、random non-balanced、pin=0 下，v76 保住约 78 μs 的快组，将 group3 的 GEMM1 中值从 90.792947 μs 降到 82.411105 μs，但仍有 5.712256% 的地址组中值差距，尚不能采用为完整修复。当前没有新增正式 GEMM1、GEMM2 或 MoE e2e 性能。

本轮继续复用原 a07-3 主机 SSH/tmux 和 SFTP 会话。`f3_v3_v76_paired_counters_20261026_v1` controller 尚在自动等待外部 KFD/GPU 任务结束，没有重复启动相同 label，也没有启动本轮实际 profiler child。等待按原协议自动进行：每 6 秒检查全机，连续 11 次空闲且跨度至少 60 秒后才开始采集。外部任务不属于本轮，不终止或驱逐。实际测试/compiler/profiler child 继续受 `timeout --signal=TERM --kill-after=2s 118s` 限制。

### v77 的独立复核和工具准备

对 v77 另做独立 AST 核对：准确移除两处 `fx.Boolean(llvm_dialect.intr_expect(..., fx.Boolean(False).ir_value()))` 包装后，整棵 AST 与 v76 相同；包装的第一参数仍是原比较，第二参数均为 False。原比较次数、coverage body、mainloop、loader、wait/barrier、WMMA、epilogue、准入和 fallback 保持。以本地 LLVM LangRef 的 `llvm.expect` 章节及其 SHA256 绑定语义 `This intrinsic is lowered to the val.`，只将它作为概率提示。

新增本地工具和 receipt：

```text
TASK/prepare_f3_v77_compile_tools_local_20261027_v1.py
TASK/logs/f3_v77_selector_expect_source_independent_local_20261027_v1.json
TASK/run_f3_v77_selector_expect_layout_cpu_20261027_v1.py
TASK/pack_f3_v77_CPU_evidence_20261027_v1.py
TASK/logs/f3_v77_compile_tools_local_20261027_v1.json
```

`TASK` 仍为 `.codex_tmp/hyg_f3_followup_20261024`。本轮只在本地生成并解析工具；v77 尚未上传、编译或运行。其 compile controller 必须先读到 v76 counters 的独立复核结果和后续决策 receipt，核对两者 SHA256 和完整 group0–3 绑定后才能执行。后续仍需看实际 ELF 是否产生实质布局变化：若机器码和资源相同，直接排除无效 hint；若有变化，先做原 native model、两种 retirement stress、完整精度，再做全部地址组的正常计时和独立 counters。

### 两端历史保留和源码状态

已下载并核对前两次 remote 文档追加 receipt，与各自 prior receipt 的 SHA256 链相符；最新两端追加 chunk 相同，SHA256=`07827467278140830a84d9297867559688e4f31366101795a77bdd38adafafe9`。本地三份文档当前内容均匹配 local receipt，各端不同的原始问答保留。

本地生产仍是 v60，源码 SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。远端 counters controller 可暂用保存的 v3 源码，结束后必须由原 finally/CAS 恢复 v60；本轮不覆盖它。两份已有暂存修改继续保留。E65 fixed shared1 与 E64/E65 共用 compiled GEMM1 的历史验收仍待完成；既有 E65 routed fallback 精度通过不能替代它们。英文 commit/push、远端主机 pull 和 Git pull 后最终复测继续作为已授权的后续收尾事项。

## 2026-10-27：首轮 counters 整轮作废并保留，按同协议重采

问：汇报下进展，然后继续。

答：v76 第一轮已验收的正常 GEMM1 计时保持不变：快组约 78 μs，group3 中值由 v3 的 90.792947 μs 降到 82.411105 μs，下降 9.231821%；地址组中值差距从 15.789180% 缩小到 5.712256%，仍未完成修复。本轮没有新增正式 GEMM1、GEMM2 或 MoE e2e 性能。

### 首轮独立 counters 因外部活动无效，不能用于对比

原 label `f3_v3_v76_paired_counters_20261026_v1` 在连续 11 次全机空闲、跨度 61 秒后启动 profiler。实际 child 17.690261 秒、returncode=0，v3/v76 均覆盖真实 HSA `load_base` 的 group0–3，共 13 个 target 实例、12 个只加载的 filler。保存了 1197 个完整目标 GEMM1 SDK dispatch records、4788 个对应的四 counter records，以及 27 份逐字节相同的输出 payload/scale 快照，完整输入前后 hash 相同。

但是 post 和 settled 两次检查均发现外部 GPU0/KFD owner `3937492`，命令属于另一项 `test_gemm_a8w8_blockscale.py` 工作。该轮 controller 按原有规则标为 `valid=false/completed=false`；本轮全部 counter 值、SDK instrumented span 和原 timer 都不用于性能或 counters 对比。没有将测后外部活动解释成“肯定发生在采集之后”来重新判有效。原始完整记录保持，不筛掉慢窗口或遗漏数据。

远端 production 已由原 finally/CAS 恢复 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`；主机 `ps` 核对本轮 profiler/test child 无残留。未终止外部进程。

```text
TASK/logs/f3_v3_v76_paired_counters_20261026_v1_evidence.tar.gz
120 members, SHA256=521fba1e7773743f075d69284c78c2faa3cebca83458775f0087ab0283aaa21e
TASK/logs/f3_v3_v76_paired_counters_20261026_v1.occupancy_failure_independent_local_review_v1.json
SHA256=8c0db782847afb7cc0ba569847130073cc39d5e608de083295fba33d80f4ac12
```

归档已下载到本地，manifest 与全部 member bytes 逐一核对并保留。单独的失败 reviewer 核对源码/ELF/HSA 绑定、输入输出 identity、测前空闲和 SDK records 完整性，同时明确记录 `usable_for_counter_comparison=false`、`usable_for_performance_comparison=false`；它不替代有效轮次的严格 counter reviewer。

### 同协议、新 label 的有限重采

预先保存 `TASK/logs/f3_v76_counter_external_activity_retry_plan_local_20261027_v1.json`，SHA256=`a0b4913cd8ccc89cbd9743b71e85f65235a3dc6f258be66e77a46c8e5f47c4cc`。原因仅为已确认的外部占用导致原轮无效，不依据耗时或 counter 好坏重测。该计划允许一轮实际 child；原 controller、冻结 ELF、loader、输入协议、counter 配置和原 timer 均保持，旧 label 不重用。

```bash
python3 -u TASK/run_f3_v3_v76_paired_acceptance_20261026_v1.py counters \
  --label f3_v3_v76_paired_counters_20261027_v2 --runs 1 --seed 0
```

启动前在主机核对计划、失败 review、旧 archive、controller 和三个受保护源码 SHA256；保留 `f3_v3_v76_paired_counters_20261027_v2.retry_context.json`。新 controller 已启动，记录时仍在等待全机连续 60 秒空闲。所有 target 实例仍全部正逆序保留和采集，不挑快实例；pin=0、A-preshuffle=1、explicit VGPR partition=0，实际 child 118 秒 TERM/再 2 秒 KILL 的限制保持。结束后仍须核对 source restored、post/settled idle、残留进程、完整 group0–3 与独立 counter 绑定。

### epilogue 的额外静态证据

已对两个实际 ELF 的反汇编再次做字节绑定。v76 的 32 个 `v_cvt_scalef32_pk8_fp4_bf16` 静态 sites 位于 PC `0x14f98–0x1943c`，`tensor_store_from_lds` 位于 `0x19578`，均在第二个 64 KiB 地址段；v3 相应范围为 `0x14d84–0x1938c` 和 `0x194c8`。因此，v76 的两个热点 mainloop backward-edge 区间都位于第一段，不代表共用 epilogue 也已进入第一段。结果保存在 `TASK/logs/f3_v3_v76_common_epilogue_static_local_20261027_v1.json`。

这是实际机器码地址的静态事实，不是 ATT live footprint，也没有证明 epilogue 是残余 miss 的唯一来源。有效 counters 仍是后续决策的门禁。v77 仍只有本地快照、独立源码复核和编译工具准备，尚未上传、编译或执行。当前继续围绕 code layout 的地址敏感性；本地生产保留 v60，两份已有暂存修改与两端原始问答继续保留。

## 2026-10-27：v76 有效 counters、v77 编译与完整精度，准备跨地址配对验收

问：汇报下进展，然后继续。

答：v76 的首轮正常计时仍是已报告的结果：E64/T1536/topk8、random non-balanced、pin=0 下，group3 中值从 v3 的 90.792947 μs 降到 82.411105 μs，快组约 78 μs；残余分档尚未修复。有效重采 counters 已完成独立复核，支持继续研究 code layout。v77 只给 v76 的既有 selector 增加 branch expectation hint，已经完成编译、native 检查以及 E64/E65 routed fallback 的完整 MoE 精度验证；尚无 v77 正式性能。其他优化方向继续暂停，下一步覆盖 v3/v77 实际 `load_base` group0–3。

本节的 `TASK` 为 `.codex_tmp/hyg_f3_followup_20261024`，`NORMAL` 为 `.codex_tmp/yadai_gemm1_param_20261021`。仅在 a07-3 的 `/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007` 主机执行 controller，构建、实际 kernel 和 profiler child 在 `dsr1_ep4` 内执行。分支为 `hyg_dsr1_conc6144`，HEAD=`e99dbac65a92ed23f87df4d99bd38f655a923a3b`，boot ID=`4d22137c-d3aa-4a81-ad98-44cb4786b2fe`。

### v76 counters 有效重采及其限度

此前 `f3_v3_v76_paired_counters_20261026_v1` 因测后外部占用整轮无效的结论保持；没有重新判有效，也没有终止外部任务。按已保存的有限重采计划，以新 label、原 controller 和同一协议执行：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v76_paired_acceptance_20261026_v1.py counters \
  --label f3_v3_v76_paired_counters_20261027_v2 --runs 1 --seed 0
```

该轮测前连续 11 次全机 idle、跨度 60 秒，post/settled idle 均通过。实际 profiler child 为 18.140977 秒，returncode=0。15 个 target、14 个 filler 全部保留，两版均覆盖真实 HSA 地址 group0–3，所有 target 均按正逆序采集。独立检查绑定了 1381 个完整 GEMM1 dispatch 和 5524 个四 counter records；每窗口 46 launches，只统计尾 20 个 counter launches，另保留初始 1 次 launch。所有 dispatch 的 `SQ_WAVES=7168`，实际 ELF/native dispatch/HSA 地址绑定、完整输入前后 hash 与输出字节 identity 通过。

| group | v3 `SQC_ICACHE_MISSES` 中值 | v76 中值 |
|---:|---:|---:|
| 0 | 5875 | 1185 |
| 1 | 26204 | 2618 |
| 2 | 5875 | 1190 |
| 3 | 28560 | 14942 |

v76 group3 的绝对 miss 中值下降 47.682073%，但仍为其 group0 的 12.609283 倍；group3 范围 13454–16120，仍明显高于其他组。这支持继续修复残余地址敏感性，但不能唯一确认具体 live cache 冲突。上述 counters 是 nondeterministic、per-SQ/per-Bank 事件计数，比例不是精确 hit rate。该轮的原 timer 和 SDK instrumented span 均不作为新 kernel 性能。

```text
TASK/logs/f3_v3_v76_paired_counters_20261027_v2_evidence.tar.gz
142 members, SHA256=caf65f7c3e587b653687a6bae756daba390edb5cc0343350c505ae656d524e9a
TASK/logs/f3_v3_v76_paired_counters_20261027_v2.paired_counter_independent_local_review_v1.json
SHA256=1c030cdcbbfaee4eb5bd40fe26dd8dd089cb520b723c0a0bc0e6fd956c2027d2
TASK/logs/f3_v76_paired_counters_decision_local_20261027_v1.json
SHA256=11f630eee204df60042cf5de8968f93e19564f0265077da6832b4a886a499659
```

### v77 的实际机器码变化和 native 检查

`TASK/snapshots/f3_v3_hot_selector_expect_layout_v77.py` 的源码 SHA256=`f2079c872bb3e40f74c88d14d8195fb7bedbff6f5d5c8e307759d33cc0085f39`。移除两处 `llvm_dialect.intr_expect(..., False)` 包装后，整棵 AST 与 v76 相同；计算、selector 比较次数、无效行跳过、同步、准入、fallback 和接口保持。LLVM LangRef 的 `llvm.expect` Semantics 为 `This intrinsic is lowered to the val.`，只作为概率提示。

编译结果：GEMM1 ELF SHA256=`e9680236c033d4a158804e4698d3a3ead85120c18dc52e7184fc0457072f2e46`，`.text` SHA256=`4b6e168053226a3c7379d9d0a71394706ceb910754c9d3a520d8e9f4b547651e`，`.text=97664 B`，VGPR799 / SGPR60 / LDS327680 B，private=0、VGPR/SGPR spills=0。compile child 为 19.743688 秒，inspect child 为 0.113816 秒，目标 GEMM1 未 dispatch。这不是新的性能数据。

两处 `s_mov_b32 s4,-1` / `s_branch` selector stub 移到 `s_endpgm` 之后，主体 PC 大体提前 16 B，一些 128 B 边界变化；没有缩小 `.text`。热点 48/56-WMMA backward-edge 区间仍约为 `0x7e80–0xb300` / `0x2f00–0x6900`，位于第一 64 KiB 地址段。FP4 quant epilogue 仍位于第二段 `0x14f88–0x1942c`，output TDM store 为 `0x19568`。尾部 stub 的向后跳转产生的 320/960-WMMA 静态区间不是新增动态 K-loop 或额外计算。

原 bounded DS/WMMA model 的 25967 次 transfer 为 0 issues，explicit `depctr_va_vdst(8)` / `(0)` 各 20 sites；原 retirement stress 与 WMMA_only retirement stress 均为 0 issues，没有放宽模型。caller signatures 与 GPU dispatch ABI 相同。

```text
TASK/logs/f3_v77_selector_expect_layout_CPU_evidence_20261027_v1.tar.gz
22 members, SHA256=a53ec1b234e613fdb9c2a26744ee172d5a872d2c67b1e00bd6be7894d086ae3a
TASK/logs/f3_v76_v77_layout_comparison_static_local_20261027_v1.json
TASK/logs/f3_v77_v3_static_layout_local_20261027_v1.json
TASK/logs/f3_v77_native_20261027_v1.review.json
TASK/logs/f3_v77_native_WMMA_ordering_stress_20261027_v1.summary.json
```

静态映射依据仍为本地 `mi400_hw_wiki/raw/papers/mi400_hd_txt/block/sq/MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Page2：`Set[4:0] = address[13:9] ^ address[18:14]`。精确 bank hash、live residency 尚未确定；实际 PC 或静态 set 推导不能替代硬件 counters 与正常计时。

### v77 的完整 MoE 精度已通过

使用原有精度协议、pin=0、A-preshuffle=1、`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`，顺序执行 E64 prepare 和 E65 guard controller；两者均用原 finally/CAS 恢复生产 v60：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v77_prepare_20261027_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v77_guard_outside_e65_20261027_v1.py
```

E64 seed0/17 完整 MoE 输出 `[1536,7168]`、22020096 B，与已验证 v60 逐字节相同，原数值精度检查通过。GEMM2 ELF 仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。E64 prepare 的四个 child 分别为 19.743866、13.684389、9.027552、8.827358 秒。

E65 seed0/17 验证的是原 routed fallback，与已保存 `f3_v49_wait0_guard_outside_e65_20261025_v1_base_s0/s17` 参考逐字节相同，原数值精度通过；不能描述成专门对照 v60。两个 child 分别为 20.445169 / 10.930374 秒。

29 个证据 member 已全部下载并独立检查 hash、日志、源码/ELF 绑定和精度记录，没有下载 `.out` 或覆盖现有文件。独立 reviewer 已通过：

```text
TASK/logs/f3_v77_E64_E65_precision_evidence_20261027_v1.tar.gz
SHA256=246211680f4c00ce66c665e044870b03425930e5e1c759794d7fde9f8bb15865
TASK/logs/f3_v77_precision_independent_local_review_20261027_v1.json
SHA256=f92e0a0d0d0c645e791ce3b569291a196b6a69e97571405aa06335d9774df5ef
```

精度期间所有实际 child 使用 `timeout --signal=TERM --kill-after=2s 118s`，均在 120 秒内结束。测后本轮进程无残留，KFD owners 为空，生产源码已恢复 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。

### 尚未运行的配对计时与 counters 有限计划

冻结 v3/v77 ELF，沿用同一实际输入、host callable、CallState、ExecutionEngine 和原 timer；保留全部 target/filler，覆盖两版 group0–3，全部正逆序测量。没有修改每次 launch 的工作量，warmup5/iters20 与原 19-event 均值复算规则保持；缺少任何一个完整 20-GPU-event 窗口即拒绝正常性能验收。

先保存 `TASK/logs/f3_v3_v77_paired_finite_plan_local_20261027_v1.json`，SHA256=`3d87178ecb2e5fbc467180742c941e300479b17d830f92096bbbc86c2523e905`。正常计时最多按 seed0/17/0 顺序运行三次独立 process；若首个有效轮次某组全部耗时范围都高于其他所有组，保留整轮并停止剩余正常复测，认定候选仍未修复。独立 counters 最多一个有效 process；仅在独立确认整轮因外部占用失效时，允许各 action 至多一次重采，不能因数字不好而重测。

下一条正常计时命令为：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v77_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v77_paired_normal_20261027_v1 --runs 1 --seed 0
```

每次性能/profiler child 前仍要求连续 11 次全机 idle、6 秒间隔、至少 60 秒跨度；外部占用时自动等待，不终止其他任务。实际 child 期间不查询 GPU、不传输、不并发编译；结束后检查 post/settled idle、source restored 和残留。正常 timer 与独立 counters 分开，不将 SDK instrumented span 当 kernel 性能。本节没有新增 v77 正式 GEMM1、GEMM2 或 MoE e2e 数据，也没有采用 v77。

E65 fixed shared1 以及 E64/E65 共用 compiled GEMM1 的历史要求仍待验收；E65 routed fallback 正确不能替代这两项。两份已有暂存修改与各端原始问答继续保留；英文 commit/push origin、远端主机 pull 和 Git pull 后最终性能复测仍是已授权的后续收尾事项。

## 2026-10-27：v77 正常计时仍分档，独立 counters 验证残余 miss 放大

问：汇报下进展，然后继续。

答：v77 已完成 E64/E65 原 routed fallback 精度、首轮正常计时和独立 counters。正常计时显示 group0/1/2 中值约 77.8–78.3 μs，group3 为 83.196263 μs；group3 的完整实测范围仍高于其他三组，未消除地址分档。依事前有限计划停止 seed17、随后 seed0 的额外正常复测，不采用 v77。有效 counters 两版均覆盖 group0–3，v77 group3 miss 仍为自身 group0 的 12.251695 倍。后续继续局限于 code footprint/layout 的修复。

`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。所有实际测试均在 a07-3/dsr1_ep4，pin=0、A-preshuffle=1、explicit VGPR partition=0，原 benchmark、timer、warmup5/iters20 和精度标准保持。

### 首轮正常计时：保留全部实例，明确标注缺失 baseline group

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v77_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v77_paired_normal_20261027_v1 --runs 1 --seed 0
```

外部 `E96/T16384/topk6` 工作期间 controller 持续等待，没有终止外部任务。达到连续 11 次全机 idle、至少 60 秒跨度后，实际 child 用时 14.435394 秒，returncode=0；post/settled idle 通过。共保留 24 个 target、29 个 filler，48 个正常 GEMM1 窗口均含完整 20 个 GPU events，并依原规则独立复算 19-event 均值。完整输入 hash 前后相同，所有输出快照字节相同；各实例全部按正逆序计时。

| 实际 HSA group | v3 中值 μs | v3 实测范围 μs | v77 中值 μs | v77 实测范围 μs |
|---:|---:|---|---:|---|
| 0 | 78.557816 | 78.248789–78.837211 | 77.912447 | 77.622579–78.185579 |
| 1 | 82.919947 | 82.564789–83.294474 | 78.274211 | 78.236211–78.312211 |
| 2 | 未覆盖 | 未覆盖 | 77.767184 | 77.382263–78.373316 |
| 3 | 90.483079 | 89.895632–90.800421 | 83.196263 | 82.486684–83.412474 |

本轮 v77 的 group0/1/2/3 分别有 2/1/13/2 个实例，v3 分别为 2/2/0/2。预设最多 24 target 的覆盖过程没有获取 v3 group2，因此 `all_four_groups_both_variants=false`：该轮不是两版完整四组的修复验收，不能用历史或 counters 的 group2 数值补齐正常计时。v77 自身四组均已测量；group3 最小值 82.486684 μs 高于其他组的全部最大值 78.373316 μs，足以证明它仍分档。按事前停止规则保留整轮，停止余下正常复测。四组中值 `max/min - 1` 为 6.981195%。

这里未假设不同 process 的输入 tensor 位模式相同，也未将 v76 上一轮的 82.411105 μs 和本轮 83.196263 μs 当作一组严格配对。新表只记录本轮同 process 内共享输入的实际结果；没有新增正式 MoE e2e 或 GEMM2 性能。

```text
TASK/logs/f3_v3_v77_paired_normal_20261027_v1_evidence.tar.gz
308 members, SHA256=85b2c685d39b1c2b80c341fb72c4517902166185ba56c0cfb2d630b3bffdc066
TASK/logs/f3_v3_v77_paired_normal_20261027_v1.paired_normal_independent_local_review_v2.json
SHA256=850a1f0183ef5de6d216dfda1128c97b0949329e2210d3c9f475f16f90bb8a7a
TASK/logs/f3_v77_first_normal_decision_local_20261027_v1.json
SHA256=6eb4d2984ea69ff207f6d1bc9d09321f7087ae5269dabf0035510a9713e6acbd
```

### 独立 counters：两版均完整覆盖 group0–3

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v77_paired_acceptance_20261027_v1.py counters \
  --label f3_v3_v77_paired_counters_20261027_v1 --runs 1 --seed 0
```

测前 11 次全机 idle、60 秒跨度，post/settled idle 均通过；实际 profiler child 为 17.990817 秒。14 个 target、13 个 filler 全部保留，28 个正逆序窗口均有 46 launches，1289 个完整 SDK GEMM1 dispatch 与 5156 个四 counter records 已独立绑定。每窗口尾 20 个 dispatch 用于 counters 对比，初始 1 次 dispatch 保留但不计入对比。所有 dispatch 的 `SQ_WAVES=7168`，ELF/native function/HSA 地址、输入前后 hash 和输出字节 identity 通过。

| group | v3 `SQC_ICACHE_MISSES` 中值 | v77 中值 | v77 范围 |
|---:|---:|---:|---|
| 0 | 5900 | 1180 | 1105–1250 |
| 1 | 26192 | 2590.5 | 2409–2783 |
| 2 | 5887.5 | 1175 | 1115–1265 |
| 3 | 28572 | 14457 | 13298–15738 |

group3 绝对 miss 中值相对同轮 v3 下降 49.401512%，但仍是 v77 group0 的 12.251695 倍，且完整范围高于其他组。未确认唯一的 live cache 冲突来源。counter 比例只是 nondeterministic、per-SQ/per-Bank 事件计数的经验比例，不是精确 hit rate。SDK instrumented span 和该轮原 timer 均不作为新的 kernel 性能；counters 完整 group2 也不补齐正常计时缺失的 baseline group2。

```text
TASK/logs/f3_v3_v77_paired_counters_20261027_v1_evidence.tar.gz
129 members, SHA256=266187d3318d4871e559c8714f5d6915d6574460dec7b5ed83b0d3f37006f25b
TASK/logs/f3_v3_v77_paired_counters_20261027_v1.paired_counter_independent_local_review_v1.json
TASK/logs/f3_v77_paired_counters_decision_local_20261027_v1.json
SHA256=f22ae4ea4e911308785237a02848b9e694a0d83102606d0cf6f3927f92ab94d4
```

通用本地 importer 只允许日志目录，曾在首个 `counters/.../out_agent_info.csv` member 处拒绝导入，未写入任何 member 文件。没有放宽测试或 native gate；独立 counter reviewer 直接从完整原 archive 读取，先核验 manifest 的全部 129 个 member，再绑定所有 records。原 archive 和 CSV 数据完整保留，未删记录、未补采或篡改有效性。这是本地导入路径限制，不是 GPU collector、kernel 精度或性能失败。

### 接续范围和生产状态

v77 分支 hint 的小布局变化没有完成修复。下一步以 v76 为父版本，只准备一个 common epilogue 前 512 B 的源码级 NOP padding 对照，用来研究 epilogue 与 mainloop 的相对 layout；先检查实际 ELF，再通过原 native、完整精度、全地址正常计时和独立 counters 判断。padding 的执行开销必须计入原 timer，不能跳过有效计算、改 timer 或挑最快加载实例。该思路属于用户指定的 layout/padding 修复范围，不恢复其他一般计算优化方向，也不承诺 padding 会消除所有波动。

本地/远端生产继续是 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`；两次 controller 都已恢复源码。实际 child 使用 118 秒 TERM/再 2 秒 KILL 的限制，均正常在 120 秒以内结束，未终止外部进程。两份已有暂存修改、各端原始问答，以及 E65 fixed shared1/共用 compiled GEMM1 的待办和已授权 Git 收尾任务继续保留。

## 2026-10-27：v78 epilogue padding 的编译和原 native 检查

问：汇报下进展，然后继续。

答：v77 仍有地址分档，group3 正常计时中值为 83.196263 μs，其他组约 77.8–78.3 μs，因此未采用为完整修复。接续的 v78 以 v76 为父版本，只在 common epilogue 前增加一个源码级 128 条 `s_nop 0` 的 512 B padding block；仍只研究用户指定的 code footprint/layout 修复，不恢复一般计算优化。v78 已完成编译、静态检查、原 bounded DS/WMMA model 和两种 retirement stress，完整 device 精度及性能仍待验证。

`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。本地分支为 `hyg_dsr1_conc6144`，HEAD=`e99dbac65a92ed23f87df4d99bd38f655a923a3b`；本轮只使用 a07-3/dsr1_ep4，pin=0、A-preshuffle=1、explicit VGPR partition=0。

### 编译与实际 ELF

编译已经完成，不能重复执行相同 label 的 controller：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v78_epilogue_pad512_cpu_20261027_v1.py
```

GEMM1 仅编译，没有 dispatch；input preparation 可能调用其他 GPU kernel，不产生有效性能数据。compile/inspect child 分别用时 19.592916 / 0.163838 秒，均使用 `timeout --signal=TERM --kill-after=2s 118s`。finally/CAS 已恢复生产 v60。

| 项目 | v78 实际值 |
|---|---|
| `.text` | 98,176 B |
| ELF | 102,632 B |
| VGPR / SGPR | 799 / 60 |
| LDS | 327,680 B |
| private / VGPR spills / SGPR spills | 0 / 0 / 0 |
| 源码 SHA256 | `d8b76f8053ae6f95f400a558352017882e942bf4163a14d338c6de6f0faebc4c` |
| GEMM1 ELF SHA256 | `1b66000a856a65aa2e165f0df76fcc510dcba7865163035b939febd854fb757d` |
| `.text` SHA256 | `bf1a2f3834b40d64fba89715a9e288b0fd1091bdb250553af44ffaeafde5f3c2` |

源码 reviewer 证明移除新增 padding block 后，整棵 AST 与其余源文件 bytes 等于 v76；再通过原 v76 selector 的 signed-int32 等价证明绑定到 v3。未新增 expert-count/data specialization，未改变准入、有效计算、精度、同步或 fallback。padding 的实际执行成本必须计入原 timer。

22 个 CPU archive member 已全部独立校验，14 个导入、8 个已有文件按原 bytes 保留：

```text
TASK/logs/f3_v78_epilogue_pad512_CPU_evidence_20261027_v1.tar.gz
SHA256=456f44f444b2b55739d7a1205bdec42a41bc742e38c5c3d0d53c1ff6a860b6ea
TASK/logs/f3_v78_CPU_evidence_imported_local_20261027_v1.json
```

### padding 并非整个后半段机器码的简单平移

实际 128 条 `s_nop 0` 位于 `0x14648–0x14844`，opcode 数量唯一增量为 `s_nop: +128`。32 个 FP4 quant conversion sites 均移动 +512 B，由 v76 的 `0x14f98–0x1943c` 到 v78 的 `0x15198–0x1963c`；32/48/56/64-WMMA 原静态 backward-edge 区间的 PC 和 decoded instructions 保持相同。

严格全指令顺序检查在 index9547、PC `0x14640` 失败：v76 为 `s_mov_b32 s0, 0`，v78 为 `s_sub_f32 s5, 0, s22`。去掉 NOP 后仍有 17 处 epilogue 附近的编译器指令顺序差异。原失败 checker 保留，没有放宽后重写为通过。后续静态报告明确记录这些差异，不声称完整机器语义等价、精度正确或 cache 已修复。

```text
TASK/review_f3_v78_padding_machine_identity_local_20261027_v1.py
TASK/logs/f3_v78_padding_layout_local_20261027_v2.json
```

### 原 native model 和 retirement stress 已通过

本地顺序运行：

```powershell
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v78_native_20261027_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v78_native_WMMA_ordering_stress_20261027_v1.py
```

原 bounded transfer 函数保持不动，实际 ISA 与 ELF bytes 绑定；26,607 次 CFG transfer iterations 未发现 bounded DS/WMMA issue。source IR 的 explicit waits 为 `(8)` 20 处、`(0)` 20 处，`(12)` 0 处。`original` 与 `WMMA_only` 两种 retirement stress 均通过，child 分别为 3.765850 / 3.171315 秒，issue_count=0。

```text
TASK/logs/f3_v78_native_20261027_v1.review.json
SHA256=7f5176a30cffae4bb3e483a427cabbf6b63fc345b992d38a4b75148409defb6a
TASK/logs/f3_v78_native_WMMA_ordering_stress_20261027_v1.summary.json
SHA256=54254bdd641ab2a16b25dfa0a4ca30089cf5e3b720b53f1ce291d61aafbb68b0
```

这些是现有保守 model 的检查结果，不是完整硬件模拟或完整 device 精度证明。下一步顺序执行 E64 seed0/17 的完整 MoE 输出对比已验证 v60，以及 E65 seed0/17 原 routed fallback 对比保存的 v45 fallback。不能将后者描述成 E65 fixed shared1 或 E64/E65 共用 compiled GEMM1 的验收。两项历史要求仍待完成。

尚未产生 v78 的正常 GEMM1、GEMM2 或 MoE e2e 性能数据，未采用 v78。精度通过后，先固定有限正常计时/counters 计划，再覆盖实际 load_base group0–3；保留全部 target/filler、正逆序窗口和原 timer，不按耗时挑选加载实例。独立 counters 与正常计时分开。

硬件依据为 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Page2 的 `Set[4:0] = address[13:9] ^ address[18:14]`，以及 CDNA5 ISA Table15/PDF Page50、§15.5/PDF Page287 的 `S_NOP`。精确 bank hash 和 live residency 仍未确定，静态 PC/set 映射不代替实测 miss 证据。

本地与远端生产继续保留 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`；两份已有暂存修改和各端原始问答保持。已授权的英文 commit/push origin、远端主机 pull 和 Git pull 后最终性能复测仍待后续收尾。

## 2026-10-27：v78 完整精度、四组正常计时与独立 counters 结论

问：汇报下进展，然后继续。

答：v78 已完成原 native、四个完整 MoE 精度用例、两版 group0–3 的正常计时及独立 counters。512 B epilogue padding 仍未修复地址分档：正常计时 group0/1/2 中值为 77.878658 / 78.175895 / 77.770184 μs，group3 为 84.372526 μs。group3 全部范围仍高于其他三组，依事前计划停止额外正常复测，不采用 v78；生产继续保留 v60。接下来仍只研究 code footprint/layout，优先检查 common epilogue 与六/七行覆盖 body 的相对位置。

`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。所有实际测试均在 a07-3/dsr1_ep4，pin=0、A-preshuffle=1、explicit VGPR partition=0，原输入、timer、warmup5/iters20 与精度标准保持。

### 完整 MoE 精度

主机顺序执行下面两个 controller；Git 环境检查在主机，实际 compile/test child 在指定容器：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v78_prepare_20261027_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v78_guard_outside_e65_20261027_v1.py
```

E64 seed0/17 的完整 MoE BF16 输出 `[1536,7168]`、22,020,096 B 与已验证 v60 逐字节一致，原数值精度通过。E64 prepare controller 共 51.658931 秒，每个实际 child 均在 120 秒内结束。GEMM1 ELF 为 `1b66000a856a65aa2e165f0df76fcc510dcba7865163035b939febd854fb757d`，GEMM2 ELF 仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

E65 seed0/17 验证原 routed fallback，完整输出与保存的 `f3_v49_wait0_guard_outside_e65_20261025_v1_base_s0/s17` 参考逐字节一致，原数值精度通过。两个实际 child 用时 20.845277 / 8.877321 秒。这不能替代 E65 fixed shared1 或 E64/E65 共用 compiled GEMM1 的历史验收。

29 个精度 archive member 已独立核验并导入，未下载 `.out`；完整 reviewer 通过：

```text
TASK/logs/f3_v78_E64_E65_precision_evidence_20261027_v1.tar.gz
SHA256=775aaa7b8ee2aec59c27798ede7cd07ae66a076a7bcfe8c481ca86224475cded
TASK/logs/f3_v78_precision_independent_local_review_20261027_v1.json
SHA256=9c0c4deca8cd4f62386ca1ef58a11b3bf4fb7fac6ed70cebbff9c3d914cb10e5
```

### 事前有限计划与首轮正常计时

先冻结两版实际 ELF，并保存有限计划，SHA256=`30a862640c1ed0cfa5709d911f669d4ae14976e38c40e74b754c329868904986`。正常计时最多按 seed0/17/0 运行三次独立 process；首个有效轮次若某组 min 高于其他所有组 max，保留整轮并停止剩余正常复测。独立 counters 最多一个有效 process。只允许已独立确认整轮外部占用失效时各 action 至多一次重采，不能因数字不好而重测。

上一轮 v77 的 24 target 上限曾缺少 v3 group2；本轮在任何 v78 timer 前，将有限加载上限预设为 48。加载选择仍只依据实际地址覆盖，不依据耗时，达到两版四组覆盖即停止；所有加载的 target 都完整正逆序计时，filler 从不 dispatch。原每实例 timer 和 launch 工作量不变。本轮实际只需要 18 个 target、20 个 filler，未达到新上限。

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v78_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v78_paired_normal_20261027_v1 --runs 1 --seed 0
```

测前连续 11 次全机 idle、6 秒间隔且跨度至少 60 秒，实际 child 为 13.283552 秒，post/settled idle 通过。同 process 内共享实际输入、host callable、CallState 与 ExecutionEngine，完整输入 hash 前后相同，全部输出 payload/scale 字节相同。36 个 GEMM1 正逆序窗口均含原 20 个 GPU events，并独立依原 19-event 均值规则复算：

| group | v3 中值 μs | v3 范围 μs | v78 中值 μs | v78 范围 μs |
|---:|---:|---|---:|---|
| 0 | 78.608211 | 78.267684–79.115263 | 77.878658 | 77.744684–78.012632 |
| 1 | 83.265789 | 82.623789–83.911947 | 78.175895 | 77.689895–78.486947 |
| 2 | 78.807395 | 78.763158–78.851632 | 77.770184 | 77.215632–78.324737 |
| 3 | 91.190237 | 90.492316–91.641316 | 84.372526 | 84.177579–84.567474 |

两版均完整覆盖 group0–3，没有用历史或 counters 数据补齐正常计时。v78 group3 的 min=84.177579 μs，高于其他三组的全部 max=78.486947 μs；四组中值 `max/min - 1` 为 8.489554%。依事前停止规则保留整轮，停止剩余 seed17/seed0 正常复测，未采用 v78。未将历史 v76/v77 的不同 process 数值当作严格配对的 padding 前后对照。

```text
TASK/logs/f3_v3_v78_paired_normal_20261027_v1_evidence.tar.gz
195 members, SHA256=b28f63a26eefed1e2a1e3a654b4a279ddbd1c1a4984052643deec8cd67c8c65a
TASK/logs/f3_v3_v78_paired_normal_20261027_v1.paired_normal_independent_local_review_v2.json
SHA256=702a5d5bae8287aa166e8debea3e864560e32fc8df91e03b70ca093571617ec1
TASK/logs/f3_v78_first_normal_decision_local_20261027_v1.json
SHA256=adb038c840ceae065c7aaed395fb241d18377da086ddb2fa073e241fb16ee51e
```

### 独立 counters：group3 仍明显放大

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v78_paired_acceptance_20261027_v1.py counters \
  --label f3_v3_v78_paired_counters_20261027_v1 --runs 1 --seed 0
```

测前及测后 idle 均通过，实际 profiler child 为 17.890587 秒；保留 13 个 target、12 个 filler 与全部正逆序窗口。完整 archive、SDK dispatch/四 counter records、ELF/native function/HSA 地址、输入 hash 和输出字节已独立绑定；所有 dispatch 的 `SQ_WAVES=7168`，两版均覆盖 group0–3。

| group | v3 `SQC_ICACHE_MISSES` 中值 | v78 中值 | v78 范围 |
|---:|---:|---:|---|
| 0 | 5875 | 2142 | 1962–2286 |
| 1 | 26098 | 2596 | 2420–2772 |
| 2 | 5850 | 2124 | 1980–2214 |
| 3 | 29072 | 18644 | 17064–20303 |

v78 group3 miss 中值是自身 group0 的 8.704015 倍，相对同 process 的 v3 group3 下降 35.869565%，但完整范围仍高于其他三组。counter 是 nondeterministic、per-SQ/per-Bank 事件计数，不是精确 hit rate；SDK instrumented span 和 counters 轮的 timer 都不当作新的 kernel 性能。精确 bank hash、live residency 与唯一冲突 block 仍未确定。

```text
TASK/logs/f3_v3_v78_paired_counters_20261027_v1_evidence.tar.gz
118 members, SHA256=1f429f6fc0290a9dccd2fa06968a91adf18e7e8fbd3ec55832e34c55b20736bd
TASK/logs/f3_v3_v78_paired_counters_20261027_v1.paired_counter_independent_local_review_v1.json
SHA256=858410957538955f55df90c8f33bfd44e57aa86a01340a4140e352f281ff994c
TASK/logs/f3_v78_complete_counters_and_next_layout_decision_local_20261027_v1.json
SHA256=129f1188213587dadb65b88ff5e668b89404b7f1a7c5cc2b929f95b1981e59aa
```

counter reviewer 直接读取原 archive，并先核验全部 member；没有更改旧 importer 的范围或重新判定此前的无效轮次。CSV 和所有记录完整保留。当前结论是这一版 padding 未完成修复，不把 `.text` 大小或静态 PC/set 映射当作 live cache 冲突的唯一证明。

下一方向仍属于用户指定的布局修复：检查能否将相同 epilogue 靠近六/七行覆盖 body，并保留四种覆盖 body、所有有效运算和原 waits/barriers。先做源码等价检查及实际 ELF，确认是否被编译器重新合并、热点区间是否实际改变，再决定后续 native/精度/性能验收；尚未产生下一版代码或性能。

本地/远端生产继续保留 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。normal/counters controller 的 finally/CAS 均恢复 source，post/settled idle 证实本轮无 KFD 进程残留；未终止外部任务。所有实际 compiler/test/profiler child 仍受 118 秒 TERM/再 2 秒 KILL 限制。三份文档各端按 CAS 追加，保留各自原始问答与两份已有暂存修改。E65 fixed shared1/共用 compiled GEMM1 的验收及已授权英文 commit/push、远端主机 pull、Git pull 后最终复测继续保留为待办。

## 2026-10-27：v79 hot/cold epilogue 源码、实际 ELF 与原 native 检查

问：汇报下进展，然后继续。

答：v78 已淘汰，生产保留 v60；继续只研究 v3 的 code footprint/layout 地址分档修复。v79 将相同 epilogue 提取成源码 helper，在六/七行与四/八行两组 mainloop 后各调用一次，并将六/七行区间标为 likely。源码独立检查、一次实际编译、寄存器约束下的 epilogue CFG、原 bounded DS/WMMA model 和两种 retirement stress 已通过。尚未执行 v79 的完整 device 精度或四组正常性能验收，不声称已经修复地址波动。

`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。测试仍只在 a07-3/dsr1_ep4，pin=0、A-preshuffle=1、explicit VGPR partition=0；分支 `hyg_dsr1_conc6144`、HEAD `e99dbac65a92ed23f87df4d99bd38f655a923a3b`，boot ID `4d22137c-d3aa-4a81-ad98-44cb4786b2fe` 已重新核验。

### 源码等价范围

此前已生成的 v79 source 与独立 source reviewer 原样保留，没有重跑生成器。恢复 selector/epilogue region 后，整棵 source AST 等于 v76；提取 helper 的完整 AST 等于原 epilogue，region 外 bytes 相同。对 `M_ROW_STEP=16/32/64` 的 12 个 signed-int32 闭区间，均执行相同的一个 mainloop 与一个 epilogue；不增加 expert/token/data specialization，不改变原准入、fallback、有效运算、wait 或 barrier。没有新增 NOP 或 ELF 注入。

```text
TASK/snapshots/f3_v3_hot_cold_epilogue_layout_v79.py
SHA256=91929fd6170fdd4f9f5a989374133929ca73202dac89f2022d64789acbb4c5a9
TASK/logs/f3_v79_hot_cold_epilogue_layout_source_20261027_v1.plan.json
SHA256=de7bacef269e5c42ae0f0ad78ddc8d0d230fc1aa1b34ba5d9d7e218ef61c7478
TASK/logs/f3_v79_hot_cold_epilogue_source_independent_local_20261027_v1.json
SHA256=b9b0b9634de9dbfd464d651c3f669cc05ca180c2901266620bdc4de172b6d974
```

### 实际编译与布局

主机 controller 运行 Git/HEAD 检查，实际 compiler/ISA inspection child 在指定容器；原 child/swap/save/stop/environment 函数 AST 保持。每个实际 child 仍为 `timeout --signal=TERM --kill-after=2s 118s`，不得将 controller 总用时误作 kernel 性能。

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v79_hot_cold_epilogue_layout_cpu_20261027_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/pack_f3_v79_CPU_evidence_20261027_v1.py
```

GEMM1 未 dispatch；准备输入可能运行其他 GPU kernel，不采集性能。实际 compile/inspect child 用时 20.594717 / 0.163795 秒。source 已由 finally/CAS 恢复。资源为 VGPR799、SGPR60、LDS327680 B，private/spills 均为 0。

v79 的 `.text=115712 B`，ELF=120168 B；相对 v76 `.text=97664 B` 增加 18048 B。编译器对 epilogue 做了部分复制和公共尾部合并，并非简单完整复制两份：静态 `v_cvt_scalef32_pk8_fp4_bf16` 从 32 处变为 61 处，六/七行 epilogue 可达集合为 29 处，四/八行为 32 处，二者集合不相交；静态 WMMA 总数仍为 2000，TDM store 仍只有一处。静态数量不等于每条运行路径执行量，也不单独证明有效工作或 device 精度。

```text
GEMM1 ELF SHA256=9e1f7a36cb06e4ac8e9532e3a270545e9ed0750c1bcf64e8fbb523aa92be3dd9
TASK/logs/f3_v79_hot_cold_epilogue_layout_CPU_evidence_20261027_v1.tar.gz
23 members，SHA256=77d051e4b5d48bdae30ae3d54470fd734b4bebd2d144a68dc0ec3f3be8361a99
TASK/logs/f3_v79_CPU_evidence_imported_local_20261027_v1.json
```

全部 archive member 的 bytes/SHA256 已独立核验，已有文件只核对原 bytes，不覆盖；仅保存新的日志、ELF、IR、ISA 与 inspection，未下载 `.out` 或 cache。

### 保留首次 CFG 检查失败，并核对公共控制点

首次不跟踪寄存器值的 epilogue CFG reviewer，在 cold 路径发现抽象可达集合包含 WMMA，故检查失败。原检查源码和失败 receipt 保留，未修改断言后称其通过：

```text
TASK/review_f3_v79_actual_epilogue_layout_local_20261027_v1.py
TASK/logs/f3_v79_epilogue_CFG_static_v1_failed_20261027_v1.json
```

实际 ISA 证明 cold epilogue 在 `0x1dc1c` 执行 `s_mov_b32 s0, 0`，`0x1dc20` 的 `s_branch` 返回 `0x1ef0`，随后 `s_and_b32 vcc_lo, exec_lo, s0` 必得零，`0x1ef4` 的 `s_cbranch_vccz` 必进入 `0xff50` 输出收尾。首次 reviewer 将这个不可能的 fallthrough 当作可行路径，不能据其失败断言实际多执行了一次 mainloop。

独立 v2 reviewer 只跟踪被指令证明的零值，未知条件仍保留两条边；核验上述机器指令与实际 branch target。六/七行和四/八行 epilogue 的约束 CFG 均无可行 cycle，不含 WMMA，所有可达终点为 `s_endpgm`，每条约束路径恰执行一次 TDM store。两段量化代码实际位于 `0xc1d0–0xfe00` 与 `0x195a8–0x1db98`；六/七行输出终点仍为 `0x10020`，不能声称热点完全处于一个 64 KiB 地址段。

```powershell
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v79_actual_epilogue_layout_local_20261027_v2.py
```

`TASK/logs/f3_v79_actual_epilogue_layout_local_20261027_v2.json` 保存完整 CFG/指令绑定；这仍是静态、有限零值分析，不是完整机器语义或 live I-cache residency 证明。

### 原 native model 与 retirement stress

```powershell
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v79_native_20261027_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v79_native_WMMA_ordering_stress_20261027_v1.py
```

原 bounded transfer/model 函数没有改动，实际 ELF/ISA bytes、source IR 与 v79→v76→v3 的 source 等价链均绑定。34,487 次 CFG transfer iterations 未发现 bounded DS/WMMA issue。source IR 的 explicit waits 仍为 `(8)` 20 处、`(0)` 20 处、`(12)` 0 处。`original` 和 `WMMA_only` 两种 retirement stress 均通过，child 用时 4.839728 / 4.638056 秒，issue_count=0。

```text
TASK/logs/f3_v79_native_20261027_v1.review.json
SHA256=0ce0961dc3e7001f43fffacc23d269543242d7d6896e16bba4a13eeac51429c3
TASK/logs/f3_v79_native_WMMA_ordering_stress_20261027_v1.summary.json
SHA256=b482f2570568ebf31861a48ff575d3abd6931b0983485dd5b22c43a29434728c
```

这些原有保守 model 不代替完整 device 精度。下一步先验证 E64 seed0/17 完整 MoE 输出对比已验证 v60，再验证 E65 seed0/17 原 routed fallback 对比保存的 v45 fallback；不将后者描述为 fixed shared1 或 E64/E65 共用 compiled GEMM1 验收。通过后冻结 ELF、预设有限计划、独立覆盖 load_base group0–3 的原正常 timer 和 counters，全部实例与全部正逆序窗口保留，不挑最快地址。

硬件事实依据为 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Page2 的 SQC I$ 64KB、4 banks/4 ways/32 sets、cache line 32 DW 与 `Set[4:0] = address[13:9] ^ address[18:14]`。branch 解码依据 CDNA5 ISA §5.5、Table20/PDF Page54：SIMM16 为 signed DWORD offset，PC 指向下一条指令。精确 bank hash/live residency 尚不能从这些资料或静态代码范围唯一确定。

生产两端继续保留 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。compile 后已核查没有本轮 KFD 进程残留，原两份暂存修改保留。E65 fixed shared1/共用 compiled GEMM1，以及已授权英文 commit/push、远端主机 pull 和 Git pull 后最终性能复测仍待后续收尾。

## 2026-10-27：v79 完整精度通过，冻结四组有限复测计划

问：汇报下进展，然后继续。

答：v79 的四个完整 MoE 精度用例通过，原数值精度门槛保持，完整 BF16 输出逐字节一致。正常性能尚未测试，不能把准备/精度阶段的耗时列为 GEMM1 成绩。已冻结实际 ELF 与有限四组正常计时/counters 协议；测后发现另有 E96/T16384 测试占用 a07-3 GPU，先按每 6 秒自动等待全机空闲，不终止该外部进程。

`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`；所有实际 child 在 a07-3/dsr1_ep4，pin=0、A-preshuffle=1、explicit VGPR partition=0。

### 完整精度与验收边界

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v79_prepare_20261027_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v79_guard_outside_e65_20261027_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/pack_f3_v79_runtime_evidence_20261027_v1.py precision
```

E64/T1536/topk8 seed0/17 的完整输出 `[1536,7168]`、22,020,096 B，与已验证 v60 完全相同。GEMM1 ELF 为 `9e1f7a36cb06e4ac8e9532e3a270545e9ed0750c1bcf64e8fbb523aa92be3dd9`，GEMM2 ELF 仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。E64 controller 共 53.009696 秒，各实际 compiler/test child 均在 120 秒内结束。

E65/T1536/topk9 seed0/17 验证原 routed fallback，完整输出与保存的 v45 fallback 参考逐字节一致，原精度门槛通过。两个实际 child 为 20.494548 / 8.776924 秒。这些记录不替代 E65 fixed shared1 或 E64/E65 共用 compiled GEMM1 的历史验收；这两项仍待完成。

```text
TASK/logs/f3_v79_E64_E65_precision_evidence_20261027_v1.tar.gz
29 members，SHA256=3948e5ff6762bed419b5cf3e2efbea82f26b97115cfd55be585424e81e9e842a
TASK/logs/f3_v79_E64_E65_precision_evidence_20261027_v1.imported_local_v1.json
TASK/logs/f3_v79_precision_independent_local_review_20261027_v1.json
SHA256=f27c40ef12790f329cd23b9256e17487b22a499ab99cadaaffbe56f537afd1e0
```

使用原 checked importer，核验全部 member bytes/SHA256；仅导入新日志、ELF/IR/ISA 等任务证据，原文件不覆盖，未下载 `.out`。独立 reviewer 核验原数值门槛、完整输出 hash 与保存的基准、完整 GEMM1/GEMM2 ELF、controller/原测试脚本 SHA256、两个 seed 与各实际 child 的 deadline；四个用例均通过。

### 正常计时前冻结的计划

```powershell
python .codex_tmp/hyg_f3_followup_20261024/prepare_f3_v3_v79_paired_tools_local_20261027_v1.py
```

固定使用本次编译的 v79 ELF，未编辑 ELF 或加 ISA 指令。与 v3 的原 GPU dispatch ABI 完全相同。沿用 v78 已有的 48 个有限 target 加载上限；本轮没有再次增加预算。只按两版 load_base group0–3 是否覆盖决定继续加载，达到完整覆盖或 48 target 即停止；所有已加载 target 均在正逆序窗口完整计时，filler 不 dispatch，不按耗时选择地址。

```text
TASK/logs/f3_v3_v79_paired_finite_plan_local_20261027_v1.json
SHA256=9b8acb38f079b48b6cf10069a6d4b40019d67f903b5cc2e3d56e123f3fe4dea4
TASK/logs/f3_v3_v79_paired_protocol_upload_manifest_local_20261027_v1.json
SHA256=97ba8da1cc952ce31ac8278fd8e4363d097729fd46462a7187d003a80b5d83db
TASK/paired_frozen_v1/v79.elf
SHA256=9e1f7a36cb06e4ac8e9532e3a270545e9ed0750c1bcf64e8fbb523aa92be3dd9
```

正常计时最多运行三个独立 process，seed 顺序为 0/17/0，每个 process 一轮。首个有效轮次如果候选某个 group 的 min 高于其他所有 group 的 max，保留整轮并停止其余正常复测，判为分档未修复。独立 counters 最多一个有效 process；只有独立证据确认整轮外部占用失效时，才允许每个 action 至多一次重采，数字不好不是重采理由。

workload 为 E64/T1536/topk8/model_dim7168/inter_dim2048、random non-balanced。两版同 process 共享实际输入、原 host callable/CallState/ExecutionEngine 与 native launch proxy；原 warmup5/iters20、所有 20 个 GPU events、原 19-event 复算规则和每次 launch 的实际工作均保持。正常 timer 与 counters 分开，SDK instrumented span 或 counters 轮 timer 不作性能成绩。

首轮复现命令，必须先通过规定的全机 idle 检查：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v79_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v79_paired_normal_20261027_v1 --runs 1 --seed 0
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v79_paired_acceptance_20261027_v1.py counters \
  --label f3_v3_v79_paired_counters_20261027_v1 --runs 1 --seed 0
```

每轮测前连续 11 次全机 idle、6 秒间隔且跨度至少 60 秒；每个实际 child 仍为 118 秒 TERM/再 2 秒 KILL，测试期间不查询 GPU、不传输、不并发编译；测后 post/settled ownership/idle 重新核验。本轮精度结束时的 ownership 记录为外部 E96/T16384 活动，记录在 `TASK/logs/f3_v79_precision_post_GPU_owners_20261027_v1.json`，因此尚不能开始性能。正确性数据不因其后出现的占用改作无效，但没有使用任何占用期间的耗时。

两端生产源码继续保留 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`；四个精度用例的 finally/CAS 已恢复，原两份暂存修改不动。已授权英文 commit/push、远端主机 pull 和 Git pull 后最终复测继续保留为后续收尾任务。

## 2026-10-27：v79 首轮正常计时采集不完整，整轮排除并限定一次原协议补采

问：汇报下进展，然后继续。

答：v79 的 native 与四个完整 MoE 精度用例已通过。首轮正常计时覆盖了两版的 load-base group0–3，输入、输出和加载代码 bytes 均保持一致，但一个原 profiler 窗口缺少 GPU events，不能作为性能结果。全部记录已经保留，生产仍为 v60；这里不发布任何 v79 性能数字。

### 首轮失败记录与证据

首轮命令在 a07-3 主机执行，实际 child 在 dsr1_ep4，pin=0：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v79_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v79_paired_normal_20261027_v1 --runs 1 --seed 0
```

实际 child 14.184912 秒结束，returncode=1。测前连续 11 次全机 idle、6 秒间隔且跨度至少 60 秒；测后 post/settled 均 idle，没有本轮 GPU 进程残留。finally/CAS 恢复生产源 SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。

24 个 target 全部保留并按正逆序计时，共 48 个 GEMM1 窗口。47 个窗口有 20 个 GPU events；`reverse_v79_clone11`、repetition37 只有 13 个，Chrome trace 同样只有 13 条 kernel events。该窗口仍有 20 条 `hipDrvLaunchKernelEx` CPU events，每个原 perftest 的 native selected launch delta 都是 46。证据确认采集不完整，没有证据表明有效计算被减少；尚不能唯一确定 ROCTracer 内部丢记录的机制。

完整输入 before/after hash、全部 GEMM1 payload/scale 输出 hash 相同，两版四组地址覆盖与加载 ELF/.text bytes 均核验。仍排除整个失败 process，不采用其他 47 个窗口的数字，也不只补测丢记录的单个实例。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
TASK/logs/f3_v3_v79_paired_normal_20261027_v1_evidence.tar.gz
295 members，SHA256=2b737a6018719a9eb53e181088429b1e81cb3a8f4442084a7517af919452cee0
TASK/logs/f3_v79_first_normal_capture_failure_local_20261027_v1.json
SHA256=623c4377f82892e84a86ead04f0dc56ba0ccb4316ee3c59e433ad258dc4364f8
```

使用原 checked importer 核验全部 member，既有证据未覆盖，未下载 `.out`。独立失败 reviewer 核验 CPU/native launch 数、Chrome/device events、四组加载 bytes、输入输出 identity、idle、source restoration；不把残留完整窗口复算为性能成绩。

### 有限补采计划，正常 timer 与门槛不变

原有限计划 SHA256=`9b8acb38f079b48b6cf10069a6d4b40019d67f903b5cc2e3d56e123f3fe4dea4` 保留。针对这一次技术采集失败，在新的正常计时开始前记录以下补充：仅允许一次整个 process 的补采，seed0；若补采再发生技术失败，停止正常采集。此补充不允许因为不利的耗时/counters 重跑。

```text
TASK/logs/f3_v79_one_capture_replacement_plan_local_20261027_v1.json
SHA256=79842ed02fd0aecbc375fdbe53aa634c4831033e44c1aa3ddd52c07d8ac68a18
```

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v79_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v79_paired_normal_20261027_v2 --runs 1 --seed 0
```

原 controller、child、两版 frozen ELF、输入生成方式、warmup5/iters20、20 events 门槛、原 timer 与 19-event 复算规则均未修改。48 target 上限、所有 target 正逆序完整计时、无最快实例选择、实际 child 118 秒 TERM/再 2 秒 KILL、测前和测后 idle 检查保持。最多三个有效 normal process，seed0/17/0；首个有效轮次若某候选组 min 高于其他全部组 max，仍按原规则淘汰。独立 counters 最多一个有效 process，原外部占用补采限制不变。

v79 尚未取得有效正常性能轮次，未宣称地址分档修复，也未替换 v60。E65 fixed shared1 与 E64/E65 共用 compiled GEMM1 的历史验收、已授权英文 commit/push、远端主机 pull 和拉取后最终复测仍保留。

## 2026-10-27：v79 首个有效四组正常计时未见分档，继续既定有限验收

问：汇报下进展，然后继续。

答：v79 首个有效正常 process 的四组中值为 77.884 / 77.831 / 77.788 / 77.789 μs，组间中值差 0.123209%；同一 process 的 v3 group3 仍为 90.243763 μs。本轮未触发事前的分档淘汰条件，但一个有效 process 尚不足以验收。继续原计划的 seed17、seed0 两个独立 process 和单独 counters；生产仍为 v60。

### 原正常 timer 的首个有效 process

所有实际 child 在 a07-3/dsr1_ep4，pin=0、A-preshuffle=1、explicit VGPR partition=0；E64/T1536/topk8/model_dim7168/inter_dim2048、random non-balanced。两版共享实际输入和原 host callable，同一 ELF 的所有已加载 target 都完整计时；未选择最快实例、未改计时或计算。

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v79_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v79_paired_normal_20261027_v2 --runs 1 --seed 0
```

本轮为上节事前限定的一次技术补采，原失败 v1 整轮仍被排除。补采实际 child 12.682642 秒完成；16 target、17 filler，32 个正逆序 GEMM1 窗口，每个窗口完整保留 20 GPU events，独立 reviewer 从全部原始 events 复算并核对原 timer/原 19-event 规则。测前连续 11 次全机 idle、6 秒间隔且跨度至少 60 秒；测后 post/settled idle，无本轮 GPU 进程残留。输入 before/after hashes、所有输出 payload/scale hashes 和实际加载 `.text` bytes 均相同，source finally/CAS 恢复。

以下每个数均为该组所有保留正常 timer 的中值，不是选取加载地址后的最快成绩：

| load-base group | v3 GEMM1 中值 μs | v79 GEMM1 中值 μs | v79 全部正常 timer 范围 μs | v79 target 数 |
|---:|---:|---:|---|---:|
| 0 | 78.619947 | 77.884105 | 77.038632–78.301579 | 2 |
| 1 | 82.620763 | 77.831421 | 77.519421–78.143421 | 1 |
| 2 | 78.423789 | 77.788263 | 77.422316–78.310000 | 5 |
| 3 | 90.243763 | 77.789289 | 77.426632–78.151947 | 1 |

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
TASK/logs/f3_v3_v79_paired_normal_20261027_v2_evidence.tar.gz
153 members，SHA256=0fc5366a306ec51192118ef1e1ddb739ee2ac02d4d064413b0df841c97d1c9f7
TASK/logs/f3_v3_v79_paired_normal_20261027_v2.paired_normal_independent_local_review_v2.json
TASK/logs/f3_v79_first_valid_normal_decision_local_20261027_v1.json
```

原 checked importer 已核验全部 member 并导入，不覆盖既有证据、不下载 `.out`。first-valid decision 仍应用原条件“某一候选组 min 高于其他全部组 max”，本轮 wholly slower group 集合为空，组中值跨度 0.123208953861%。因此继续两个事先约定的独立正常 process，而非按这次数字临时加测：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v79_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v79_paired_normal_20261027_v3 --runs 1 --seed 17
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v79_paired_acceptance_20261027_v1.py normal \
  --label f3_v3_v79_paired_normal_20261027_v4 --runs 1 --seed 0
```

剩余 normal 和独立 counters 尚未完成。该首轮不是生产路径的最终性能验收，也未新增 MoE e2e 性能数据；没有推断精确 bank hash、live residency 或所有加载地址都已消除波动。E65 fixed shared1、E64/E65 共用 compiled GEMM1 的历史验收和已授权的英文 commit/push、远端主机 pull、拉取后最终复测继续保留。

## 2026-10-27：v79 三个正常 process 全部保留，seed17 回退，继续检查布局

问：汇报下进展，然后继续。

答：原有限 normal 计划按 seed0/17/0 全部完成。两次 seed0 中 v79 group0–3 约 77–78 μs，没有观察到 v3 的 group3 约 90–91 μs 分档；seed17 中 v79 四组均约 98.5–98.8 μs，同 process 的 v3 为 89.2–92.3 μs。因为 seed17 回退，v79 未采用，不能只取两次 seed0 的快值作为最优版。生产继续保留 v60。

### 完整正常结果与 routing

下面按 group0/1/2/3 依次列每组所有保留正常 timer 的中值，三个 process 不合并成一个性能数字：

| seed | label 末尾 | v3 四组 GEMM1 中值 μs | v79 四组 GEMM1 中值 μs |
|---:|---|---|---|
| 0 | v2 | 78.619947 / 82.620763 / 78.423789 / 90.243763 | 77.884105 / 77.831421 / 77.788263 / 77.789289 |
| 17 | v3 | 89.672368 / 92.261342 / 89.235868 / 91.823842 | 98.767000 / 98.789053 / 98.639395 / 98.527658 |
| 0 | v4 | 78.548368 / 82.911605 / 78.125526 / 91.001711 | 77.771368 / 77.493026 / 76.976474 / 77.136632 |

三个实际 child 分别为 12.682642 / 13.284098 / 12.181916 秒；两版四组地址全部覆盖，所有原 20 GPU events/19-event timer 复算通过，全部 target 正逆序计时、输入输出 hash 不变。测前 11 次/60 秒全机 idle、测后 post/settled idle 与 source restoration 通过，没有本轮 GPU 进程残留。原采集失败 v1 仍整轮排除；有限正常次数已用完，不再重采正常数据。

已保存的 tile_map/valid-row spans 确认：seed0 有 35 个 expert 走六行覆盖、29 个走七行覆盖，valid rows 为 168–221；seed17 有 32/30/2 个 expert 分别走六/七/八行覆盖，valid rows 为 169–225。八行覆盖由 expert10、expert54 各 225 行触发。两 seed 的有效 routed rows 都是 12,288。该证据确认代码路径不同；A/B/scales、数据/代码地址也随 process/seed 改变，所以不是只改变 routing 的因果隔离实验，不能唯一归因于某个 I-cache bank、capacity、pipeline tail 或 descriptor 行为。

v79 seed17 的所有正常 timer min=97.561895 μs，高于该 process v3 所有正常 timer max=93.811211 μs。这是有效的候选回退记录，不是技术采集失败，不以此为理由重测、换输入或挑地址。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
TASK/logs/f3_v79_three_valid_normal_and_routing_local_20261027_v1.json
SHA256=7c964d69b3c6a1966f48601fcdd73593f39ac0eeb0abc0bdd5412342f764f9b7
```

所有三个 archive member 和独立正常 reviewer 均核验；没有覆盖历史或下载 `.out`。本轮没有修改 kernel 功能、有效计算、准入条件、fallback、timer 或精度标准，没有新增 expert/token/data specialization。

### counters 接续与新增回退诊断的有限边界

原计划的唯一 seed0 counters 继续执行，独立于正常计时，其 timer/SDK span 不作性能。为解释已记录的 seed17 回退，事前另保存一个有明确边界的诊断：seed17 同一 pair、同一 ELF、同一 controller/child，只运行一次，不重采、不新增 normal process；所有 target 的 SDK/counter records 均保留。

```text
TASK/logs/f3_v79_seed17_routing_counter_diagnostic_plan_local_20261027_v1.json
SHA256=c9badb132c3d30809c5c3824bfe8ff26196b547e981272e7174451644f206018
```

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v79_paired_acceptance_20261027_v1.py counters \
  --label f3_v3_v79_seed17_routing_diagnostic_counters_20261027_v1 --runs 1 --seed 17
```

新增诊断针对已经出现的另一随机输入的回退，不用于替换原 seed0 counters，也不以某组 counters 高低为理由增加实例或重测。仍保持 48 target 上限、11 次/60 秒全机 idle、118 秒 TERM/再 2 秒 KILL、child 期间不查询 GPU/编译/传输、post/settled idle。之后只继续 code footprint/layout 修复，优先检查能否保留六/七路径邻接，同时使完全相同的 epilogue 只保留一个公共实例；实际 ELF 未编译前不宣称 code size 或 placement 有改善。

E65 fixed shared1、E64/E65 共用 compiled GEMM1 的历史验收和已授权英文 commit/push、远端主机 pull、拉取后最终复测仍待后续收尾。

## 2026-10-27：v79 全部计时/counters 结束，不采用，继续单次公共 epilogue 布局

问：汇报下进展，然后继续。

答：v79 完成原 native、四个完整 MoE 精度用例、seed0/17/0 三个有效正常 process、原 seed0 counters 和另行事前限定的一次 seed17 回退诊断。seed0 下地址分档被明显压低；seed17 回退记录仍有效，因此不采用 v79，生产保持 v60。下一版只改 code layout，尝试保留六/七路径邻接，同时将完全相同的 epilogue 改为 selector 后的一次公共调用。

### 全部 counters 的独立证据

下面是 `SQC_ICACHE_MISSES` 事件计数中值，不能当作精确 hit rate 或 kernel 耗时：

| seed | load-base group | v3 miss 中值 | v79 miss 中值 |
|---:|---:|---:|---:|
| 0 | 0 | 5925 | 0 |
| 0 | 1 | 26100 | 0 |
| 0 | 2 | 5925 | 0 |
| 0 | 3 | 28914.5 | 470 |
| 17 | 0 | 14848.5 | 21235.5 |
| 17 | 1 | 32387.5 | 21367 |
| 17 | 2 | 14173 | 21256.5 |
| 17 | 3 | 30546.5 | 20741.5 |

seed0 中 v79 group0–2 在本次窗口 miss 全部为 0，group3 中值470；同 process 的 v3 group3 为28,914.5。seed17 中 v79 四组 miss 中值为20,741.5–21,367，与该随机输入触发两个八行覆盖路径以及约98.5–98.8 μs的正常耗时回退同时出现。两版全部 SDK dispatch `SQ_WAVES=7168`，完整 kernel/counter records、native function、实际 HSA load base、ELF/.text bytes、输入与输出 identity 均绑定。

这支持布局在 seed0 下减少 miss，同时在另一个合法 routing 下仍有明显 instruction-cache 压力。不能从该对照唯一判定 capacity、具体 bank/set 冲突、live residency 或 pipeline tail 各自贡献。硬件资料 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Page2 记载 I$64KB、32 sets、4 ways 和 `Set[4:0] = address[13:9] ^ address[18:14]`。v79 实际 `.text=115712 B`，编译器部分复制 epilogue；总 `.text` 大于 cache 不等于所有代码同时 live，仍需看真实路径和计数。

两个实际 profiler child 分别为 18.141398 / 17.139894 秒。均通过11次/60秒全机 idle、post/settled idle、finally/CAS source restoration，无本轮GPU进程残留。seed0为原唯一 counters 验收 process；seed17为已记录回退后的唯一有限诊断，没有再重采。两轮 instrumented timer/SDK span 均不作性能成绩。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
TASK/logs/f3_v3_v79_paired_counters_20261027_v1_evidence.tar.gz
118 members，SHA256=e6eeca2413668e8459da4265fd025ef9e4a6376b2e3a077176ab8ef7d8cb9854
TASK/logs/f3_v3_v79_seed17_routing_diagnostic_counters_20261027_v1_evidence.tar.gz
87 members，SHA256=2dac772d2cf239ac04ef49c867b91b1b1c6f9409d5d28821891e7ad3216baf24
TASK/logs/f3_v79_complete_layout_decision_local_20261027_v1.json
SHA256=9c3e4b76665378c42f051b93b0b9792fd41fce1c44b0c49254f9e64e145c1247
```

原 checked importer 核验全部 member、不覆盖历史、不下载 `.out`；两份独立 counter reviewer 通过。正常性能以之前三个有效 process 的全部原 timer 为准，原首次采集失败仍整轮排除，不挑 seed0 或快地址作为生产成绩。v79 的所有计划运行均结束。

### 下一步保持同样计算，只调整公共尾部布局

保留 v3/v76 的四/六/七/八行覆盖 body、所有有效数学、TDM/LDS/WMMA/wait/barrier、准入和 fallback；沿用 v79 的六/七与四/八 selector 及 LLVM likely hint，将相同 epilogue 只在 selector 后调用一次。该源码结构可能影响 compiler 的复制/合并和 block placement，编译前不宣称实际只有一个 machine epilogue、热点不跨段或性能改善。先查真实 ELF/ISA 和原 native/retirement model；再做完整精度与跨加载地址、跨 seed 正常验收。

E65 fixed shared1、E64/E65 共用 compiled GEMM1 的历史验收，以及已授权英文 commit/push、远端主机 pull、拉取后最终复测仍保留。生产源 SHA256继续为`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`，原两份暂存源码修改未动。

## 2026-10-27：v80 公共 epilogue 的实际编译、原 native 检查、完整精度与有限测速计划

问：汇报下进展，然后继续。

答：v79 的正常计时和两组 counters 已结束，因合法 seed17 回退而淘汰，生产仍保留 v60。继续只研究 v3 的 code footprint/layout 地址分档修复。v80 保留相同的六/七行与四/八行 selector、likely hint 和原 mainloop，将同一 epilogue helper 的源码调用移到 selector 后，只调用一次。v80 的源码等价、一次实际 GEMM1 编译、原 native 模型和四个完整 device 精度用例均已通过独立检查；四组加载地址的正常性能和 counters 尚未执行，不能声称修复已验收或达到 80 μs。

`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。仍只在 a07-3/dsr1_ep4 测试；pin=0、A-preshuffle=1、explicit VGPR partition=0，原 benchmark/timer 不变。分支 `hyg_dsr1_conc6144`、HEAD `e99dbac65a92ed23f87df4d99bd38f655a923a3b`、boot ID `4d22137c-d3aa-4a81-ad98-44cb4786b2fe` 已检查。Git 检查在主机，compiler/test child 在指定容器，每个实际 child 使用 `timeout --signal=TERM --kill-after=2s 118s`。

### 源码和实际机器码

已有 v80 source 与 source reviewer 没有重新生成或重跑。恢复 selector/epilogue region 后，完整 source AST 与 v76 相同；helper 的完整 AST 与原 epilogue 相同。对 `M_ROW_STEP=16/32/64` 的 12 个 signed-int32 闭区间，均执行相同的一个 mainloop 和一个 epilogue。原 loader、算术、wait、barrier、无效行跳过、准入和非目标 compile flags 的 fallback 均保留，没有新增 expert/token/data specialization、NOP padding 或 ELF 注入。

| 实际编译结果 | v79 | v80 |
|---|---:|---:|
| `.text` bytes | 115712 | 97408 |
| 静态 WMMA sites | 2000 | 2000 |
| 静态 `v_cvt_scalef32_pk8_fp4_bf16` sites | 61 | 32 |
| 静态 `tensor_store_from_lds` sites | 1 | 1 |
| VGPR / SGPR | 799 / 60 | 799 / 60 |
| LDS bytes | 327680 | 327680 |
| private / VGPR spills / SGPR spills | 0 / 0 / 0 | 0 / 0 / 0 |

v80 减少了 v79 的机器码复制，但静态数量不是每条路径执行量，也不单独证明 device 精度。v80 `.text` 总大小等于原 v3 的 97408 B，实际 `.text` bytes 和完整 ELF 与已保存的旧版均不相同，不能沿用旧版性能作为新成绩。

```text
TASK/snapshots/f3_v3_likely_pair_single_epilogue_layout_v80.py
source SHA256=93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312
NORMAL/logs/f3_target_v80_single_epilogue_layout_pin0_cpu_20261027_v1_g1.elf
ELF SHA256=49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8
.text SHA256=739d3cc9c731c8378cb758b6ed5af8fc0978ae5795878a6ba1152ca2a55f1f7d
TASK/logs/f3_v80_single_epilogue_layout_CPU_evidence_20261027_v1.tar.gz
24 members，SHA256=40d68e052877befb8b71038c0ea8cc248d1bf5213f284cfccdb4494da3234b57
```

实际 compile/inspect child 用时为 19.743950 / 0.163844 秒，不是 kernel 性能。此阶段 GEMM1 没有 dispatch；输入准备可能运行其他 GPU kernel。所有 archive member 的 bytes/SHA256 已检查，已有本地文件仅核对相同 bytes，不覆盖；未下载 `.out` 或 cache。生产由原 finally/CAS 恢复。

实际 backward-edge 静态区间：六行覆盖的 cache-line PC 范围为 `0x7d80–0xb180`，七行覆盖为 `0x2f00–0x6800`，八行为 `0x11880–0x15380`，四行为 `0x16780–0x19000`。各自的这些 interval 未跨 64 KiB；公共量化的 32 处转换 PC 为 `0xc118–0x10730`，仍跨边界。上述 interval 与 PC 范围不是 ATT 观测的 live footprint，不能据此断言 cache conflict 已消失或 epilogue 是唯一原因。

硬件依据仍为 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1：I$ 64KB、4 banks、4 ways、32 sets、cache line 32 DW，`Set[4:0] = address[13:9] ^ address[18:14]`。实际 branch target 按 CDNA5 ISA §3.2.1/§5.5 的 next-instruction-relative signed DWORD offset 计算。精确 bank hash 和 live residency 未由这些静态分析确定。

### 原 native 模型与完整精度

继续使用原 bounded DS/WMMA transfer model，未放宽 transfer 函数；原模型和 `original` / `WMMA_only` retirement stress 均为 0 issue。实际 ISA instruction/machine bytes、原调用参数/globals signature 和 GPU dispatch ABI 均已绑定。IR 中显式 `s_wait_alu depctr_va_vdst(8)` / `(0)` 各 20 处，原 source wait provenance 保留。这些是原模型覆盖范围内的检查，不是完整硬件执行模拟。

完整输出验证的四个新执行用例及原精度检查均通过，每个输出为 22020096 B、shape `[1536,7168]`：

- E64/T1536/topk8 seed0、seed17：完整 MoE 输出与已验证 v60 的保存输出逐字节相同。
- E65/T1536/topk9 seed0、seed17：验证原 routed fallback，完整输出与保存的 v45 fallback 参考逐字节相同。
- GEMM1 ELF 仍为上述 v80 ELF，GEMM2 完整 ELF 仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

未重复执行已通过的 v60 / v45 参考用例。E65 fixed shared1 和 E64/E65 共用 compiled GEMM1 的最终验收仍待完成，不能以本轮 E65 fallback 验证替代。精度运行中的打印耗时不作为性能数据。

```text
TASK/logs/f3_v80_native_and_layout_local_20261027_v1.json
TASK/logs/f3_v80_precision_independent_local_review_20261027_v1.json
TASK/logs/f3_v80_E64_E65_precision_evidence_20261027_v1.tar.gz
29 members，SHA256=f20c8076cec603ba0561f5226ddca05b7a114bae25176222c480527e97f79082
```

原执行命令（这些证据标签独占创建，不能覆盖后重跑）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_single_epilogue_layout_cpu_20261027_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_prepare_20261027_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_guard_outside_e65_20261027_v1.py
```

### 环境等待与中断记录

开始时以及精度验证结束后，均观察到外部 E96/T16384/topk6/inter_dim3072 的 GPU 测试。没有 kill、驱逐或竞争该任务。每次启动前使用原 waiter 每 6 秒检查全机 `gpu_users.sh --plain`、KFD owners 和 DRM sysfs busy/VRAM，累计至少 11 次、跨度至少 60 秒的连续 idle；实际 child 期间没有并发 GPU 查询、文件传输或编译。

第一次启动包装在等待结束后因远端缺少 `f3_paired_review_idle_20261026_v1.py` 而中断，尚未启动 compiler/test，保留失败记录。补传原检查脚本后，先加载并检查它，再重新执行当前 idle gate；没有复用已过时的第一次等待。GEMM1 实际 compile/inspect 仅运行一遍。`rocm-smi --showuse --showmeminfo vram --showpids` 的一次初始汇总查询触发原 `GetGPUMetricsFormat1` assertion，异常输出未被当作 idle 证据，后续沿用上述原 waiter。

编译后的 post/settled 均 idle；E64 精度后的 post/settled 均 idle；E65 精度后第一次 post 无 KFD owner，6 秒后的 settled 出现外部 E96 进程 PID804888，记录保留，未杀该进程。接下来的性能测试必须重新等待全机 idle。本轮精度结果有效，没有接受该阶段任何性能数据。

### 已冻结的正常计时/counters 计划

```text
TASK/logs/f3_v3_v80_paired_finite_plan_local_20261027_v1.json
SHA256=eeb7fb2c3e5e3dbd0ae5cd6baf18f5b9e28e77b45d9ee9074ea086cb79db09d4
TASK/logs/f3_v3_v80_paired_protocol_upload_manifest_local_20261027_v1.json
SHA256=e5d58ed04f512daf2ac8e54198ea9d60ba6eadc0149a3d1c41ca3e0bb53cb9e8
```

从原 v79 协议仅替换 variant、source/ELF/precision SHA 和路径绑定；逆向恢复这些精确替换后，child、controller 和两个 independent reviewer 的完整 module AST 与原件相同。原 timer、warmup5、iters20、全部 20 个 GPU events 和原 19-event 计算规则均保留。

正常计时最多三个有效 process，seed 顺序 `0/17/0`。同一 process 内只生成一次实际输入，v3/v80 共用全部参数和指针；每个 variant 覆盖 `(public_HSA_load_base >> 16) & 3` 的 group0–3。加载上限仍 48 个 target，只按地址覆盖选择加载，所有加载的 target 均正/逆序计时；fillers 不 dispatch，不查询其 function，不挑最快实例用于生产。若任一 candidate group 的全部计时都慢于其他所有 group（min 大于其他组 max），保留整个有效 process，按原 early-stop 规则停止剩余正常重复。

counters 与正常计时分开：seed0 一次；仅在仍有修复验收可能，或合法 seed17 的正常 process 显示 candidate 回退时，再执行一次 seed17 诊断，最多两个有效 counter process。counter span 不能作为正常 kernel 性能。外部活动使整轮失效后的重试上限沿用原计划；本轮还事先限定一次独立确认的完整 20-event 捕获失败的整 process 补采，全部失败证据保留，不使用其余窗口，也不为慢值补采。

正常计时复现入口（在上述协议文件、缓存和绑定证据已就位后使用新的唯一 label）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v80_paired_acceptance_20261027_v1.py normal --label <new_unique_label> --runs 1 --seed 0
```

当前仍没有 v80 正常性能或修复验收结论。英文 commit/push origin、远端主机 git pull，以及 git pull 后 E65/shared1、E64/shared0 的最终性能复测仍保留为后续任务，未提前标记完成。

## 2026-10-27：v80 三个有限正常 process 的完整核验，下一步单独采 counters

问：汇报下进展，然后继续。

答：v80 的三个预定正常 process（seed 顺序 `0/17/0`）均通过独立核验，v3/v80 每轮均覆盖四个实际加载地址组；全部加载的 target 均正、逆序计时，全部原 20-event 窗口保留，原 19-event timer 重算规则不变。两个 seed 下均未触发预定地址分档 early-stop，seed17 未重演 v79 约98.5–98.8 μs的回退。当前仅完成正常计时，counters 尚待执行，生产仍为 v60。

`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。仍只在 a07-3/dsr1_ep4；E64/T1536/topk8/model_dim7168/inter_dim2048，random non-balanced A4W4、SiLU、无 bias；pin=0、A-preshuffle=1、explicit VGPR partition=0。全部性能使用同一个原 caller/timer，没有改输入生成、有效计算或精度 gate。

### GEMM1 正常计时（μs，完整保留窗口的组内中值）

| process | seed | load-base group | v3 | v80 |
|---|---:|---:|---:|---:|
| v1 | 0 | 0 | 79.264421 | 77.679868 |
| v1 | 0 | 1 | 83.545447 | 77.366868 |
| v1 | 0 | 2 | 78.716211 | 77.749500 |
| v1 | 0 | 3 | 90.462105 | 77.911763 |
| v2 | 17 | 0 | 89.619053 | 89.580132 |
| v2 | 17 | 1 | 92.876816 | 89.873316 |
| v2 | 17 | 2 | 89.894342 | 89.801632 |
| v2 | 17 | 3 | 91.922737 | 89.172316 |
| v3 | 0 | 0 | 78.417289 | 78.482605 |
| v3 | 0 | 1 | 82.284079 | 77.428421 |
| v3 | 0 | 2 | 78.929684 | 77.574947 |
| v3 | 0 | 3 | 90.536368 | 77.870132 |

seed0 两轮 v80 全部原 timer 的最大值均小于79 μs；seed17 四组中值为89.172316–89.873316 μs，同轮 v3 为89.619053–92.876816 μs。seed17 是另一个合法随机输入，其约89–90 μs结果完整保留，不能将seed0的77–78 μs推广成所有 random routing 的耗时，也不能把这个输入差异归因于同一 ELF 的加载地址。不同 process 不预设输入 bytes 相同；每个 process 内两版共用实际完整 GEMM1 输入和指针，before/after hashes 保持不变。

### 完整证据与有限计划

| process | target / filler | actual child 秒 | archive members | archive SHA256 |
|---|---:|---:|---:|---|
| v1 | 15 / 14 | 12.532553 | 139 | `f106afba93bdf67a42e010e263c2e05ac4f7c18a6fd5899906a9c3a6fea804de` |
| v2 | 18 / 18 | 13.033136 | 182 | `02f78afb0fdbe62f282f10c126eb20cc3f7e9d958e50dc5f11f081fd248ed640` |
| v3 | 13 / 12 | 12.182093 | 116 | `95623a4db12ed37beac97692bade066ad14bc700c866f1ce55a343ff23b5bc0d` |

全部独立 reviewer 绑定实际 HSA load base、64 KiB对齐、完整 ELF 和loaded `.text` byte identity、function handle、原GPU dispatch ABI、全部原计时窗口、输入和输出 identity。group 定义仍为 `(public_HSA_load_base >> 16) & 3`；按地址覆盖冻结加载计划，不按读数挑实例；fillers 全部不 dispatch、不查询 function。

每轮均通过连续11次、6秒间隔、跨度至少60秒的全机idle门禁，post/settled均idle；实际child均使用 `timeout --signal=TERM --kill-after=2s 118s`，finally/CAS恢复源码，无本轮GPU进程残留。等待门禁期间发生的外部E96工作负载保留在idle记录中；实际测试时没有并发GPU查询、传输或编译。v80没有正常测试失败、替换或为慢值重采。所有139/182/116个archive成员均由原checked importer验SHA并保留，不下载`.out`。

复现使用已冻结controller；controller临时swap原v3源码，通过相同native proxy选择冻结v80 ELF，仍保留原launch/timer：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v80_paired_acceptance_20261027_v1.py normal --label f3_v3_v80_paired_normal_20261027_v1 --runs 1 --seed 0
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v80_paired_acceptance_20261027_v1.py normal --label f3_v3_v80_paired_normal_20261027_v2 --runs 1 --seed 17
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v80_paired_acceptance_20261027_v1.py normal --label f3_v3_v80_paired_normal_20261027_v3 --runs 1 --seed 0
```

已使用label均为独占创建，复现应换新label；上面是实际完成的三个process，不能再跑来替换本轮数据。三轮normal计划已结束，不增加正常重复。下一步单独seed0 counters；由于正常修复验收仍有可能，按已冻结触发条件再采一次seed17 counters，最多两个有效counter process，instrumented span不能当作正常kernel性能。

```text
TASK/logs/f3_v80_finite_normal_decision_local_20261027_v1.json
SHA256=d35319dbc0e143b54addf2688708d72c0af7190d97a2e2bb6c9161f7abe12158
TASK/logs/f3_v3_v80_paired_finite_plan_local_20261027_v1.json
SHA256=eeb7fb2c3e5e3dbd0ae5cd6baf18f5b9e28e77b45d9ee9074ea086cb79db09d4
TASK/logs/f3_v3_v80_paired_normal_20261027_v1.paired_normal_independent_local_review_v2.json
SHA256=ad76912d801dc727ae065c315e56ade5e4c8d32bfb9fe7abc5423f643ffd2f58
TASK/logs/f3_v3_v80_paired_normal_20261027_v2.paired_normal_independent_local_review_v2.json
SHA256=1e1fd6fc8de8dade6838d9dd1a9b95d507ef9865721b75d61e68bf28bec38369
TASK/logs/f3_v3_v80_paired_normal_20261027_v3.paired_normal_independent_local_review_v2.json
SHA256=39746bb814323d9af674c9121bb543e0ed6f0cca0afb5184eb01d7e0d1eea3f2
```

E65 fixed shared1、E64/E65共用compiled GEMM1的最终验收、英文commit/push origin、远端主机git pull及pull后最终性能复测仍保留为后续任务。当前正常计时没有代替这些验收，也没有修改生产源码。

文档生成检查发现Windows native pipe将首份CPU草稿的中文替换为问号；该草稿未追加到任何正式文档，保留在 `TASK/logs/f3_v80_finite_normal_doc_chunk_20261027_v1.md`。改用此UTF-8 Python文件生成v2，并在CAS追加前核验完整中文、LF和SHA256；没有重跑任何测试。

## 2026-10-27：v80 完成有限跨地址修复验收，进入普通源码/原入口采用检查

问：汇报下进展，然后继续。

答：v80 的三个正常process（seed0/17/0）和两个独立counter process（seed0/17）全部通过核验。两个合法输入下，两版都覆盖实际64 KiB对齐HSA load-base的group0–3，全部target和正常计时保留；v80没有重现v3的明显地址快慢分档，慢组的instruction-cache miss放大也得到改善。因此接受这份v80 ELF在本轮明确范围内的跨地址修复。生产仍保留v60，先做普通源码经原入口调用的固定三轮采用检查；未宣称所有random routing均低于80 μs或历史E65/shared1要求已经完成。

仍为 a07-3/dsr1_ep4、E64/T1536/topk8/model_dim7168/inter_dim2048、random non-balanced A4W4、SiLU、无bias，pin=0、A-preshuffle=1、explicit VGPR partition=0。`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。

### 独立SDK counters：原timer的正常性能见上一节，本表仅为miss事件数

| seed | load-base group | v3 `SQC_ICACHE_MISSES` 中值 | v80 中值 | 中值降幅 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 5875 | 0 | 100.00% |
| 0 | 1 | 26122 | 474 | 98.19% |
| 0 | 2 | 5875 | 0 | 100.00% |
| 0 | 3 | 29109.5 | 3081 | 89.42% |
| 17 | 0 | 14386.5 | 10476 | 27.18% |
| 17 | 1 | 32089.5 | 11301.5 | 64.78% |
| 17 | 2 | 14128.5 | 10452.5 | 26.02% |
| 17 | 3 | 30435 | 11567.5 | 61.99% |

seed0 的v80 group0/2在本次保留的tail20窗口miss全部为0，group1中值474、group3中值3081；对应v3慢组为26122和29109.5。seed17的v80四组中值10452.5–11567.5，v3为14128.5–32089.5。seed17仍有instruction-cache压力，但其miss数量与正常计时均没有重演v79在该合法输入上的大幅回退。全部目标SDK dispatch的 `SQ_WAVES` 都为7168，原完整输入/输出identity与native/source数学等价检查均保留。

每个SDK dispatch已逐条绑定native function、实际HSA load base、完整ELF/loaded `.text` bytes、原GPU dispatch ABI、CPU/native launch区间与输出identity。完整capture records及全部加载target均保留；统计子集仍是每个实例正/逆序各46次中的最后20次，probe1+warmup5+auxiliary Event20+profiler20保持原样，initial reference dispatch不混入组内统计。counter采集中的instrumented timer/SDK span均未采用为正常性能数据。

硬件资料 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1记载I$64KB、4 banks/4 ways/32 sets与 `Set[4:0] = address[13:9] ^ address[18:14]`；`MI400_SH_PerfCounter_Users_Guide#71.txt` §5.1记载ICACHE lookup请求/hit与non-duplicate miss、per-SQ/per-Bank及ND事件属性。表中的miss降幅是实际事件中值的对照，不能用它唯一确定capacity、具体bank/set冲突、live residency或pipeline tail的贡献，也不能声称所有miss已消除。

v80仍保留四/六/七/八行原mainloop和所有有效数学；只改变hot/cold selector的布局提示及公共epilogue的源码调用位置。`.text=97408 B`、量化转换静态sites=32、TDM store静态sites=1；原native/retirement model、四个完整MoE精度用例和完整源码等价已验收。总 `.text` 超过64KB不等于全部代码同时live，正常计时与独立counters是本轮采用判断的依据。

### 两个counter process的全部证据

| seed | target / filler | actual child 秒 | archive members | archive SHA256 |
|---:|---:|---:|---:|---|
| 0 | 21 / 23 | 19.292749 | 256 | `c17a1206dfeff933a4304cfdf2b0126fc579bd8aa4be769c20efbbab4a8d2eb3` |
| 17 | 13 / 12 | 18.041007 | 121 | `863a240bfb57199ebe964b0c531d5e0457af3d3d5771cc67cd4f7f90cdca61a2` |

两轮均通过连续11次、6秒间隔、跨度至少60秒全机idle门禁，post/settled均idle，finally/CAS恢复v60源码，无本轮GPU进程残留；实际child期间不并发GPU查询、编译或传输。seed0启动前的外部E96工作负载保留在pre-swap/idle记录中，自动等待，没有杀死外部进程。v80的整个有限计划没有失败或替换process，不为慢值重采；checked importer验完256/121个archive成员，未下载`.out`。

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v80_paired_acceptance_20261027_v1.py counters --label f3_v3_v80_paired_counters_20261027_v1 --runs 1 --seed 0
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v3_v80_paired_acceptance_20261027_v1.py counters --label f3_v3_v80_paired_counters_20261027_v2 --runs 1 --seed 17
```

上述label已使用，不能重跑替换本轮结果。修复接受范围：同一冻结ELF、上述E64配置、seed0/17、已观察到的64 KiB对齐group0–3、原pin0 timer；它不是对所有随机输入或其他加载对齐的保证。seed0两个process全部计时最大值78.743579 μs，seed17四组中值89.172316–89.873316 μs；全部原正常窗口已写入上一节。

### 后续源码采用检查的冻结范围

这一步是已验证v80源码接入普通runtime的采用检查，与已经结束的paired-address计划分开。复用未经修改的 `TASK/f3_v80_normal_20261027_v1.py measure` 与已完成的v80/v60 preparations，通过原 `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py` 入口固定三轮seed0、warmup5/iters20；没有clone/filler/native proxy，也不按地址或读数选择ELF。核验全部原表、实际cache key/IR/full ELF、原精度、11次/60秒idle及post/settled；每个实际test/audit child仍为118s/2s。controller只做CAS临时源码swap、日志和finally恢复v60，不增加kernel优化、specialization或更改timer。

固定三轮结果全部保留，不因慢值或技术失败追加；仅在独立确认整次action受外部ownership污染时允许最多一次整action补采，不能选择其中的快轮。CPU检查确认原 `child/observe/save` 完整AST相同。CPU reviewer的末尾status print因重复 `passed` keyword报错；已完成的核验JSON在报错前写好，随后只读取并验SHA，未重跑核验或启动GPUchild，该错误保留独立receipt。

```text
TASK/logs/f3_v80_complete_bounded_layout_decision_local_20261027_v1.json
SHA256=fbcce108d4f2fef05f66234f65c6c48fe9eaa0108fc7aa932095e455aa110e54
TASK/logs/f3_v80_source_qualification_20261027_v1.finite_plan.json
SHA256=d8abc08822fb70e66c4dafc92477e29452e4bce3cbeb112a0b90cd9635a232b4
TASK/logs/f3_v3_v80_paired_counters_20261027_v1.paired_counter_independent_local_review_v1.json
SHA256=85df2c84dc08944e9c7de6255af305a58db0eb8e0e8450f905b6439629cf1bc3
binding_SHA256=9ef2a924800e767aaebf174f8d67d9b4c7331d356f8d403d123590befd6ba104
TASK/logs/f3_v3_v80_paired_counters_20261027_v2.paired_counter_independent_local_review_v1.json
SHA256=ea179b63f88048eb250fdf3d10d28bce2eca72f7a3d89ca4d59deec746aa8cd3
binding_SHA256=c3fe1f696a5472ec41fe84028ef673cb3e5bf1dfae620f70c61624fda288185c
```

历史待办继续保留：E65 fixed shared1、E64/E65共用compiled GEMM1的最终验收，以及已授权的英文commit/push origin、远端主机pull和pull后E65/shared1、E64/shared0最终性能复测。当前跨地址修复没有替代这些工作。

## 2026-10-27：v80 原入口采用检查中止，保留失败证据并诊断 profiler 空记录

问：汇报下进展，然后继续。

答：v80 已通过上一节的有限跨加载地址验收，seed0 两个独立 process 的全部保留窗口为 76.668684–78.743579 μs，seed17 仍为约89–90 μs，没有把合法慢输入删掉。普通源码/原入口采用检查完成一轮，但第二轮因原 benchmark 的 profiler 空数据报告错误退出，第三轮没有执行。不能把第一轮当作三轮最终成绩，生产采用尚未完成；两端仍保留 v60。继续先定位测量问题，不修改原 timer、不补选快地址、不开始其他 kernel 优化。

### 原入口采用检查的完整状态

沿用 `pin=0`、A-preshuffle=1、explicit VGPR partition=0、E64/T1536/topk8/model_dim7168/inter_dim2048、random non-balanced、SiLU、无bias、seed0、warmup5/iters20。

| 原计划轮次 | GEMM1 μs | GEMM2 μs | 精度 / 状态 |
|---|---:|---:|---|
| r1 | 77.9181 | 64.5991 | `logits_diff=3.61055e-06`；原完整表、actual keys/IR/full ELF、11次/60秒全机idle及post/settled均验收 |
| r2 | 不采用 | 不采用 | 原精度检查通过；profiler 空数据导致报告异常，未生成完整原表 |
| r3 | 未执行 | 未执行 | r2 非零退出后，固定controller中止 |

r1 两阶段完整ELF分别为 `49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8` / `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`，实际原call keys和IR匹配已完成preparation，未复制ELF替换cache。r2也通过启动前连续11次、6秒间隔、至少60秒全机idle门禁，但缺少完整结果和事先规定的后续验收，不能接受其局部打印的耗时。

三个实际child（r1 test、r1 audit、r2 test）分别为8.676909 /7.124510 /8.827171秒，退出码0/0/1；都使用 `timeout --signal=TERM --kill-after=2s 118s`。外层controller用时1075.267323秒，包含自动等待外部任务和空闲门禁，不是GPU测试超时。controller最终按CAS恢复v60，当前源码SHA仍为 `8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`，两份既有staged edits保持原样。

实际test/audit期间没有并发GPU查询、编译或传输。失败后检查未发现本轮进程残留；20:30:30 UTC另一次保存的观察中存在外部E96/T16384/topk6验证进程，该观察明确不在失败child期间，不能据此判定r2受到外部ownership污染或启用补采。外部任务没有被杀死。

### CPU复现确定的报告处理问题，以及仍不确定的collector原因

原r2日志先出现 `no valida data after post process!`，随后：

```text
TypeError: unsupported format string passed to NoneType.__format__
```

错误位于原 `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py` 的kernel-bench报告格式化。`aiter/test_common.py:get_trace_perf()`在没有有效GPU数据时仍返回0；原 `_rates()` 对非正耗时返回 `(None, None)`；调用方只检查elapsed是否为`None`，继续用 `:.2f` 格式化TFLOPS，因而报错。CPU复现直接执行这三个原函数的完整AST，没有改动源码或计时函数；只提供synthetic记录，不访问GPU。空记录情况复现了相同TypeError，正常记录下原平均公式保持一致。该复现证明报告处理有bug，不证明collector为何漏记。

`ROCTracer produced duplicate flow start: 1` 同时出现在r1成功日志和r2失败日志，不能仅凭此警告认定根因；此前其他实验也保留过类似空数据记录。当前证据不足以唯一确定collector缓冲、flow关联或其他记录路径的贡献。没有把cuda.Event辅助日志改作正式性能数据，也没有放宽精度、改timer或因为失败追加v80正常轮次。

### 证据与状态

`TASK=.codex_tmp/hyg_f3_followup_20261024`，`NORMAL=.codex_tmp/yadai_gemm1_param_20261021`。

```text
TASK/logs/f3_v80_source_qualification_20261027_v1_evidence.tar.gz
bytes=152357, members=25
SHA256=e4c9cda882d954bb5c355ce4b14d64661271f4887c1dc9d13874934c6498e341
TASK/logs/f3_v80_source_qualification_failure_independent_local_review_20261027_v1.json
SHA256=f8c5fa16a5855e4897f3038bceb8039763b29b7435df2c302a2742bbd8f9af44
TASK/logs/f3_empty_profiler_reporting_CPU_reproduction_20261027_v1.json
SHA256=880627b2ddc31fe7609d110c2894c7694d1d63a45c4150cb018830612bd80ea9
TASK/logs/f3_empty_profiler_reporting_CPU_reproduction_20261027_v1.child_receipt.json
SHA256=926635e020009d0fd2059a0fda1bc83e729f9183a00dd132391fbbee23ee38c6
```

全部25个archive成员经checked importer校验；原r1表、r2错误、外层失败、两轮idle和恢复观察均保留，未下载`.out`。独立review结论为 `evidence_review_passed=true`、`source_qualification_passed=false`；上一节的有限跨地址决定仍有效，但没有据此绕过未完成的普通源码采用检查。

### 冻结的一次v60 profiler event诊断

为区分collector漏收和后处理问题，新建独立诊断计划，使用保留v60和原已验证cache，通过相同原入口/CLI，只做一个process。hook首先执行原 `get_trace_perf()`，原处理完成后才保存该profiler对象的全部FunctionEvent、associated kernels和Chrome trace，并原样返回原结果；不新增GPU dispatch、timer或替代值。CPU AST检查确认原函数先执行且结果未经修改返回。这个诊断的全部耗时排除在正式性能数据之外，与已中止的v80采用检查分开，不补做r2/r3，不更改有限跨地址实验，也不采用新的kernel优化。

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/diagnose_f3_profiler_records_20261027_v1.py run
```

```text
TASK/logs/f3_v60_profiler_records_diagnostic_20261027_v1.finite_plan.json
SHA256=41d455e594cfc2fbd07a55e42ceeb66db68f47c7b110394b5be0fe598dc8f49b
TASK/diagnose_f3_profiler_records_20261027_v1.py
SHA256=33867e1cc867fed364b8dad7936c3b65debaf9bf56422ea5ca318cb43dccbafa
```

仍仅在a07-3/dsr1_ep4运行，Git在主机，启动前11次/60秒idle，外部占用自动每6秒等待，每个实际child为118s/2s。脚本已在固定tmux中启动；本章不提前宣称诊断完成或collector根因已解决。

历史待办不变：E65 fixed shared1与E64/E65共用compiled GEMM1的最终验收，以及已授权的英文commit/push origin、远端主机pull和pull后最终性能复测，均未由本次E64跨地址修复代替。

## 2026-10-27：报告处理修复、v80 重启前验收完成，以及新 boot 的 v60 baseline 重测

问：汇报下进展，然后继续。机器重启了，请重测 baseline 的性能。

答：本次 baseline 是当前生产保留的 v60，pin=0、原入口和原 timer。新 boot 下完整三轮 GEMM1 中值为 **83.4505 μs**，GEMM2 中值为 **65.2188 μs**；三个原精度结果均通过。重启前 v80 的原入口中值为 77.6057 μs，这个历史值不替代新 boot 成绩。本轮继续优先完成已经验证的 v80 地址布局修复的源码采用检查，一般新调度优化仍暂停。

### 已完成的报告处理修复与重启前 v80 原入口验收

此前失败轮在进入 benchmark 后处理时已经没有 CUDA profiler 记录。GEMM1/GEMM2 的 None、NaN 或非正耗时，现在统一报告为不可用，避免 `_rates()` 返回 None 后格式化报错。改动仅在 `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py` 中增加4行、删除2行：有效正耗时的公式与输出、原 `perftest/get_trace_perf`、kernel 调用、输入生成和原精度路径均没有修改。

该 guard 没有修复或掩盖 collector 本身；缺少有效计时的轮次仍不能作为性能数据。原失败采用检查继续保留失败状态与完整归档。一次后处理诊断保留 v60，在原 timer 结束后导出 events/Chrome trace，GEMM1/GEMM2 各有20条 CUDA event，没有复现空记录；`duplicate flow start` 警告也出现在成功记录中，不能据此唯一确定 collector 漏记根因。该诊断的耗时未采用为性能数据。

报告修复后的独立三轮计划已在旧 boot 完成，完整归档50个成员在本地独立核验通过，未重跑这次成功 GPU 试验：

| 旧 boot 轮次 | v80 GEMM1 μs | GEMM2 μs |
|---|---:|---:|
| r1 | 77.0702 | 64.9046 |
| r2 | 77.8060 | 64.5567 |
| r3 | 77.6057 | 64.7907 |
| 中值 | 77.6057 | 64.7907 |

三轮 `logits_diff=3.61055e-06`，实际原入口 cache key、IR 和完整 ELF 均匹配 v80/GEMM2；每轮连续11次、间隔6秒且跨度至少60秒的全机 idle，以及 post/settled 检查均通过。原失败计划没有被覆盖，已结束的 paired-address/counter 计划没有重复采集。重启前 finally 已恢复 v60，v80 当时尚未作为生产源码采用。

### 重启后的环境与测量范围

```text
新 boot ID: a94beab0-6a60-46c6-9347-c90fb32d31fd
旧 boot ID: 4d22137c-d3aa-4a81-ad98-44cb4786b2fe
branch: hyg_dsr1_conc6144
HEAD: e99dbac65a92ed23f87df4d99bd38f655a923a3b
host: yanguahe@heliosr-1b114-a07-3.mnb.dcgpu
repo: /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
container: dsr1_ep4
baseline source SHA256: 8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008
GEMM1 ELF SHA256: 59e4bd1aea39005b5bb1a0aa7e36871c75816f4233a374705a71fa8c78cb8135
GEMM2 ELF SHA256: 4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be
PyTorch: 2.11.0+rocm10.1.0a20260811-bkc.20260820
HIP: 7.16.26315
FlyDSL: 0.3.3.dev882
```

两端分支、HEAD、保留源码与报告 guard 的 SHA256、容器和导入路径均重新核对。复用已验证的 v60 preparation/cache，但先通过当前安装环境重新计算原入口 key，并验 IR/full ELF，未复制或选择其他 ELF。`aiter/test_common.py` 保持 SHA256 `90f6ca70aa5567c0138aac18fe9a8a2eb1f383e5806d8d09da828e6f00cb6a54`。重启后的 `numa_balancing=0` 已记录，没有为本轮测量修改该系统设置。

workload 固定为 E64/T1536/topk8/model_dim7168/inter_dim2048、random non-balanced A4W4、SiLU、无bias、seed0、`data-init=uniform`、`scale-init=auto`；pin=0、A-preshuffle=1。使用原 `--scenario kernel`、warmup5/iters20。这次原入口执行了完整 MoE 精度路径，但没有测量新的 MoE e2e 耗时；不把 GEMM1/GEMM2 的和当作 MoE e2e。

### 首次重测整组排除，全部证据保留

首次 `f3_v60_reboot_baseline_20261027_v1` 计划完成r1/r2，原表读数分别为 GEMM1 83.1571/83.1152 μs、GEMM2 65.4991/64.9786 μs，两轮原精度与 ELF 均通过。r2 完成后的 post 检查没有 KFD owner，6秒后的 settled 检查出现 PID622409 的 E64 GPU 工作负载，因此既定条件失败，r3没有执行。

现有 post/settled 记录不能证明新进程与原 timer 实际重叠，也不能把该现象归为 kernel 精度或计时代码错误；按既定条件保守排除整个 action，两个原表均未计入下面新三轮中值。实际 GPU child 都在约8.7秒内完成，正常退出后的 post 无本轮 GPU 残留。外部进程没有被杀死。

完整42成员归档 SHA256 `404e7ede1fa9546c06c654951d7b17b0bb45edd3333be12130dab8d54261595b` 已验，排除独立review SHA256 `c27a858dbff3b3f166087757f8361a891f2367ad65f4589193e785e3e7d2af87`。仅在核实这个环境原因后新建一次完整固定三轮计划 `f3_v60_reboot_baseline_20261027_v2`，不复用旧r1、不因快慢读数重测、不覆盖旧失败状态，所有新轮次都保留。

### 新 boot 的完整 v60 baseline 三轮

| 轮次 | GEMM1 μs | GEMM2 μs | logits_diff | 原精度通过 |
|---|---:|---:|---:|---|
| r1 | 83.4505 | 65.2188 | 3.61055e-06 | True |
| r2 | 83.3576 | 64.6054 | 3.61055e-06 | True |
| r3 | 83.6464 | 65.4380 | 3.61055e-06 | True |
| 中值 | **83.4505** | **65.2188** | — | True |

此前 v60 原入口三轮中值为 81.6604 μs，本次 GEMM1 变化为 +1.7901 μs（+2.19%）。这是不同 boot 的复测结果，不能仅据此唯一归因到时钟、NUMA、温度或加载地址。新 baseline 将用于同 boot 的 v80 采用判断，旧 v80 77.6057 μs 仍单列为历史值。

每轮都通过11次/60秒全机 idle、原精度、真实原入口key/IR/full ELF及 post/settled，实际 test/audit child 均使用 `timeout --signal=TERM --kill-after=2s 118s`；child 运行期间没有并发 GPU 查询、编译或文件传输。独立本地review SHA256 `dd65c64a8d5bed98a1ccccbdd02a3b9071fa8eeb114bfd0cb428b4e396cfc662`，新完整归档 SHA256 `3b527f8a4386df2057da35a94ebbf6f7d6922a1d4b4bce11ed2e19af31ec27fd`。

实际本轮主机启动命令（label已经使用，后续复测需保留旧记录并准备新label）：

```bash
cd /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v60_reboot_baseline_20261027_v2.py run
```

容器内原测试命令：

```bash
AITER_GROUPED_A4W4_A_PRESHUFFLE=1 AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0 \
PYTHONPATH="$PWD" AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 AITER_LOG_MORE=1 AITER_FORCE_GFX1250=1 GPU_ARCHS=gfx1250 CU_NUM=256 \
AITER_MOE_EXPERT_BALANCE=false AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=2s 118s python3 -u \
op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 \
--experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 \
--act silu --no-bias --no-check-aot-cache --warmup 5 --iters 20 \
--seed 0 --data-init uniform --scale-init auto
```

### 当前边界与后续工作

生产源码仍为 v60，v80 新 boot 采用检查尚未在本节完成。v80 已结束的跨地址有限验收覆盖 seed0/17、实际64 KiB对齐 load-base group0–3、同一冻结 ELF；seed0全部保留窗口76.668684–78.743579 μs，seed17各组中值89.172316–89.873316 μs，不能宣称所有random输入均低于80 μs或I-cache miss全部消失。硬件事实与counter证据见前文，本节不重复采集。

E65 fixed shared1、E64/E65共用 compiled GEMM1的历史待办，以及已授权英文commit/push origin、远端主机pull和pull后最终性能复测继续保留，本次baseline数据没有替代这些验收。

## 2026-10-27：补回 a07-3 保存的旧 OPTX 问答

以下907行原先仅存在于a07-3的bug问答文档中，本地缺少。它们是2026-10-18起的历史问答与性能记录，不是本轮的新测量。经比较两端内容、独立备份和SHA256校验后，将原文作为历史补充追加在这里，保留本地此前所有内容；远端原文不改。


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

v246/v247仅metric bias0/100、v248仅ScaleB raw LLVM load、v250/v251合并DS issue hints均与v245同`.text`，CPU停止、未GPU；v249两侧raw load为4715条指令超过原4710门槛，也未GPU。这些是无native收益或筛选拒绝，不是新增已证实数值bug。后续v252/v253仅调整独立Scale/operand读取次序，20个actual-reader AST请求multiset/bank结果通过，仍需新native及完整GPU验收。生产仍v190，正式E64/E65三轮91.688/137.596 μs、距E64目标11.688 μs保持。以下独立工作树章节及其数据保留。

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

## 2026-10-27：新 boot v80 原入口验收的 GPU 空闲等待记录

截至 2026-10-27 08:22:06 UTC（北京时间16:22:06）的保存日志，`f3_v80_reboot_source_qualification_20261027_v2` 仍在等待 r1 的11次/60秒全机 idle 门禁，本 action 尚未启动任何实际 GEMM test/audit child。a07-3 上的其他用户持续执行 benchmark、profile 和 trace；等待程序每6秒自动复查全机 ownership、busy 和 VRAM，没有结束或驱逐其他用户的进程。等待 controller 的总耗时不代表实际 GPU test 耗时；每个实际 child 仍使用118s/2s限时。

重启后的完整 v60 baseline 仍为 GEMM1=83.4505 μs、GEMM2=65.2188 μs（各三轮中值，pin=0、A-preshuffle=1、E64/T1536/topk8、原 precision gate 通过）。新 boot 的 v80 性能尚无有效数据，MoE e2e 耗时也未测。旧 boot 的 v80 原入口中值77.6057 μs，以及跨 load-base group0–3 的 seed0/17正常计时和 counters，继续保留在前文对应记录中，不将它们当成新 boot 的性能结果。

截至该记录，本地 production kernel 保留v60；remote controller 于07:07 UTC临时将 production source换为v80，用于本次冻结的原入口验收。v80尚未正式采用；controller正常结束或普通异常退出后通过CAS恢复v60。待完整三轮结束，继续归档所有原表、核验source/key/IR/full ELF、precision与post/settled状态；达到既定采用条件后，再分别采用两端源码并追加文档。当前未启动其他新的kernel调优方法，E65 fixed shared1、E64/E65共用compiled GEMM1及英文commit/push/pull后的最终复测仍保留为待办。


## 2026-10-27：新 boot 的 v80 原入口三轮完成，80 μs 采用门槛未通过

问：汇报下进展，然后继续；a07-3 is free now。

答：保留已有等待与验收进程，在全机连续11次/60秒空闲门禁通过后完成剩余轮次。新 boot 的 v80 GEMM1 三轮中值为 **81.2470 μs**，同 boot v60 baseline 为 **83.4505 μs**，GEMM1 中值改善 **2.64%**。三轮原精度与来源/计时证据均通过，但预先准备的采用脚本要求三轮 GEMM1 全部低于80 μs，该条件未满足；本章没有采用v80，production仍保留v60。距离handoff的70 μs目标还差 **11.2470 μs**。

### 测量范围与全部原表

仅在 a07-3/dsr1_ep4，branch=`hyg_dsr1_conc6144`，HEAD=`e99dbac65a92ed23f87df4d99bd38f655a923a3b`，boot=`a94beab0-6a60-46c6-9347-c90fb32d31fd`。workload为E64/T1536/topk8/model_dim7168/inter_dim2048、random non-balanced A4W4、SiLU、无bias、seed0、`data-init=uniform`、`scale-init=auto`，pin=0、A-preshuffle=1、原`--scenario kernel`、warmup5/iters20。未测新的MoE e2e耗时。

| 版本 | 轮次 | GEMM1 μs | GEMM2 μs | logits_diff | 原精度通过 |
|---|---|---:|---:|---:|---|
| v60 baseline | r1 | 83.4505 | 65.2188 | 3.61055e-06 | True |
| v60 baseline | r2 | 83.3576 | 64.6054 | 3.61055e-06 | True |
| v60 baseline | r3 | 83.6464 | 65.4380 | 3.61055e-06 | True |
| v60 baseline | 中值 | **83.4505** | **65.2188** | — | True |
| v80 | r1 | 81.3037 | 64.7782 | 3.61055e-06 | True |
| v80 | r2 | 81.1332 | 65.2672 | 3.61055e-06 | True |
| v80 | r3 | 81.2470 | 65.4168 | 3.61055e-06 | True |
| v80 | 中值 | **81.2470** | **65.2672** | — | True |

新boot的全部三轮原读数均保留，首轮81.3037 μs没有因未达80而被替换或排除。性能三轮及三个独立cache audit都在118s/2s限时下正常完成，各轮全机idle与post/settled检查通过；实际child期间没有并发GPU查询、编译或文件传输，正常结束后CAS恢复v60。等待GPU的parent时长与实际test时长分别记录。来源review通过只表示本次原表、精度、source/key/IR/full ELF及空闲证据有效，不代表性能目标或采用门槛通过。

### 已核实的差异与尚未确定的原因

对应三轮baseline/v80 manifest中的完整 `shell_command` 逐字相同；warmup/iters/seed相同，记录的caller、`aiter/test_common.py`、route/quant和launcher源码SHA256相同。manifest差异仅为label、variant、各自原源码对应的runtime cache目录，以及目标GEMM kernel源码SHA256；cache key/IR/full ELF均由真实原入口核验，不按ELF或加载地址选择实例。

重启前相同v80 source/full ELF的原入口三轮中值为 **77.6057 μs**，新boot变化 **+3.6413 μs（+4.69%）**。该跨boot差异不能仅凭相同命令和ELF唯一归因到I-cache、clock、温度或NUMA。现有普通验收manifest未记录实际输入hash、GPU code load_base和时钟/热状态，不能补造这些证据，也不能将本章表格归为特定load-base group。

v80已有的有限跨地址修复验收仍单列在前文：旧boot、同一冻结ELF、合法seed0/17、观察到的64 KiB对齐group0–3。它改善了v3慢组的instruction-cache miss放大；seed17的89–90 μs结果继续保留。该证据不保证任意random输入、其他加载对齐或重启后所有时钟状态都低于80 μs。当前继续优先处理这个地址波动修复的采用边界与必要诊断，一般新GEMM1优化方法仍暂停。

### 证据与复现

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
NORMAL=.codex_tmp/yadai_gemm1_param_20261021
v60 source SHA256=8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008
v80 source SHA256=93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312
v80 GEMM1 full ELF SHA256=49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8
GEMM2 full ELF SHA256=4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be
baseline review SHA256=dd65c64a8d5bed98a1ccccbdd02a3b9071fa8eeb114bfd0cb428b4e396cfc662
v80 source qualification review SHA256=272c23fd51c6dd08ef7cf8f4da194efb1a156e134f40adae0fba9bfa47b36749
v80 qualification archive SHA256=5eff055d5e83dbbd3d7d17e538664067ade6678ce4b2438cf5683c5eb79d2718
comparison SHA256=1dc5979495645821403952f93536117f3d15ff0343ec1e7f26a3242b08615acf
```

本轮主机启动命令如下；label已使用，不得重跑来替换慢轮或覆盖既有证据：

```bash
cd /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_reboot_source_qualification_20261027_v2.py run
```

容器内原入口参数保持：

```bash
AITER_GROUPED_A4W4_A_PRESHUFFLE=1 AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0 \
PYTHONPATH="$PWD" AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 AITER_LOG_MORE=1 AITER_FORCE_GFX1250=1 GPU_ARCHS=gfx1250 CU_NUM=256 \
AITER_MOE_EXPERT_BALANCE=false AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=2s 118s python3 -u \
op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 \
--experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 \
--act silu --no-bias --no-check-aot-cache --warmup 5 --iters 20 \
--seed 0 --data-init uniform --scale-init auto
```

E65 fixed shared1、E64/E65共用compiled GEMM1，以及已授权英文commit/push origin、远端主机pull和pull后的最终性能复测仍为历史待办，本次E64原入口数据没有替代这些工作。

## 2026-10-27：新 boot 诊断首轮保存、缺失两轮补采与 A padding 范围核查

问：汇报下进展，然后继续。

答：新 boot 的正式性能仍为 v60 GEMM1 中值83.4505 μs、v80中值81.2470 μs，改善2.64%，原精度通过；没有新增正式 kernel 或 MoE e2e 性能。v80未达到预先冻结的三轮全部低于80 μs采用门槛，production没有正式切换到v80。继续优先核查地址波动修复的跨boot边界，一般新GEMM1优化方法仍暂停。

### 原输入/加载/时钟诊断的首轮及未完成状态

`f3_newboot_original_context_20261027_v1` 原计划固定执行v60→v80→v60，所有读数均为诊断值，排除在正式性能之外。首轮v60在38.8533秒内正常退出，原精度通过，保存了实际输入hash、scalar geometry、public HSA loader加载信息及timer外的时钟样本。

首轮结束后，`gpu_users.sh`输出包含PID2645661的`<已退出>`行，而同次KFD owners为空、busy=0、VRAM=176582656 B。因此旧controller的`post.ownership_idle`条件失败，缺失的r2/r3没有启动，finally通过CAS恢复v60。6秒后的settled检查完全idle。已核查`gpu_users.sh`先枚举KFD PID再读取`ps`/queues，PID在两步之间退出时仍可能输出该行；之后该PID不在`/proc`或KFD中。现有证据不能断言PID属于本轮，也不能断言其与原timer重叠；没有修改全局GPU检查脚本，没有杀外部进程。

旧action保持incomplete，未覆盖、重开或改称验收通过。31成员归档已独立核验：

```text
archive=.codex_tmp/hyg_f3_followup_20261024/logs/f3_newboot_original_context_20261027_v1_evidence.tar.gz
archive bytes=211164
archive SHA256=3cce64755da14f3e9a2c1c85482e8c78e729f7af1f77211ca6e6c4facd8d0abc
review=.codex_tmp/hyg_f3_followup_20261024/logs/f3_newboot_original_context_20261027_v1.incomplete_local_review.json
```

review中的`passed=True`仅表示保存证据核验通过；`diagnostic_action_completed=False`、`all_post_gates_passed=False`继续保留。首轮GEMM1加载base为`0x7c8d15720000`，64 KiB aligned、group2，`.text`实际地址为`0x7c8d15721a00`；GEMM2加载base为`0x7c8d15170000`、group3。对应full ELF和实际loaded `.text`均核验一致。首轮原timer返回后GEMM1 SCLK样本2224 MHz、GEMM2样本2222 MHz，MCLK1900 MHz、模式auto；这些都是timer外样本，不能当作kernel运行期间的精确时钟，也不能唯一解释旧boot 77.6057 μs与新boot 81.2470 μs的差异。

### 只补缺失的 v80→v60 两轮

补采controller已启动，复用原source/ELF/cache/CLI；没有重复首轮v60，没有filler、加载实例选择或计时修改。worker只替换冻结plan文件名，所有函数AST与v1一致。每个实际child采用118s/2s限时；每个child结束后立即CAS恢复v60，再保存原始post状态。若post非idle，每6秒等待settled idle，不杀进程；下一child重新通过11次、至少60秒的全机idle门禁。

```bash
cd /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
python3 -u .codex_tmp/hyg_f3_followup_20261024/complete_f3_newboot_context_20261027_v2.py run
```

```text
plan=.codex_tmp/hyg_f3_followup_20261024/logs/f3_newboot_context_completion_20261027_v2.finite_plan.json
plan SHA256=90a013e17a6b95741b0d749f5355835922741f0ba4720b211d715bf198f195de
fixed_order=v80,v60
pin=0
performance_data=False
```

截至2026-10-27 10:17:57 UTC的保存日志，summary仍为`completed=False, rounds=[]`，本action尚未启动实际GPU child，仍在累计全机稳定idle窗口。初始`source_restored=False`只是controller初始状态，不表示候选已采用或此时源码已换为v80。结束后再归档、独立核验全部补采证据；不将诊断读数加入正式三轮中值。

### A 的 full buffer hash 与有效输入不能混同

CPU独立比较新boot首轮与旧boot两个seed0 paired-address process：B、ScaleA、ScaleB、psum的完整buffer hash相同，route counts与scalar keyword geometry相同，A完整buffer hash不同。旧paired记录不能冒充旧正式77.6057 μs测量的输入证据；旧正式manifest没有实际输入hash。

进一步依据旧paired保存的psum、geometry、TDM M16读取spans，独立核算如下。两份旧记录的spans完全相同：

| A 区域 | 行/tiles | payload bytes |
|---|---:|---:|
| 有效routed rows | 12288行 | 44040192 |
| TDM实际读取的M16范围，含尾部padding | 797个M16、12752行 | 45703168 |
| 上述M16中的无效尾部padding | 464行 | 1662976 |
| 完全不读取的M16 | 995个M16 | 57057280 |
| 完整分配 | 1792个M16、28672行 | 102760448 |

这里`TDM实际读取`依据kernel的`ceil((expert_row_end - blk_m)/16)` bound及保存的expert spans，不能把该范围的所有bytes都叫作数学上有效输入。旧boot两个独立process的`read_M16_tiles_SHA256`本身也不同：

```text
f3_v3_v80_paired_normal_20261027_v1_r1:
fb491f3837ebf6bd22daff10679ab298146264149e31b678fa056da731b07ae8
f3_v3_v80_paired_normal_20261027_v3_r1:
75963df2f010a1b36f19ad700173a5bc81139159a685f22cb8cefa3d06c211af
```

此差异仍可能含M16尾部padding，现有保存hash不能唯一判定有效行payload是否不同，也不能证明只来自padding。不能要求不同process的M16读取hash天然相同。CPU coverage审计最初对两者相同的无依据assert已保存失败记录，随后改为保留两个实际hash并报告比较结果；没有重新执行任何GPU测量。

```text
coverage=.codex_tmp/hyg_f3_followup_20261024/logs/f3_saved_A_read_coverage_20261027_v1.json
initial CPU audit failure=.codex_tmp/hyg_f3_followup_20261024/logs/f3_saved_A_read_coverage_20261027_v1.initial_assertion_failure.json
```

后续必要的输入审计应分开记录严格有效行、M16尾部padding和完全不读取tiles的hash，保持真实输入不变。当前尚未启动这种额外GPU审计，不补造跨boot有效输入等价的结论。E65 fixed shared1、E64/E65共用compiled GEMM1，以及已授权英文commit/push origin、远端主机pull和pull后最终复测仍保留。

## 2026-10-27：相同 logical hidden 下物理 route map 变化，准备有效 A 的 canonical 核查

问：汇报下进展，然后继续。

答：CPU继续核对旧paired诊断中保存的quant输入，找到比A full hash更具体的差别：两个旧boot独立process的`quant_a1.args[0]`完整hash相同，也与新boot首轮的logical hidden相同；但`row_to_token`和`topids_to_rows`完整hash不同。对应dtype、shape、stride、bytes均相同，route counts与psum相同。因此route counts相同不能证明GEMM1的物理行映射逐字节相同，A的不同hash也不能直接归为padding。新boot正式性能和生产源码状态不变，没有新增正式kernel或MoE e2e耗时。

### 保存证据及其解释边界

```text
comparison=.codex_tmp/hyg_f3_followup_20261024/logs/f3_preserved_quant_source_and_map_comparison_20261027_v1.json
logical hidden / quant source full SHA256:
9b52bd0a7798bf544ac9e9d77fc8ccc46057c7129e771d6a4e9fb0c46f019306

f3_v3_v80_paired_normal_20261027_v1_r1:
row_to_token SHA256=dea3dac9c905f0e9a585822b1b0f46bcbf74b15299ebe089d57bb29b810188a1
topids_to_rows SHA256=67fbbd9eafa6d62a16b06eeb034c26849d5b7363b4d2e4060149a186c437466b

f3_v3_v80_paired_normal_20261027_v3_r1:
row_to_token SHA256=3732079cab24d24b89f20d5b5c95ab88aa1b416c26baa646f9af3ea184ac0e9f
topids_to_rows SHA256=ba027666173386255618801e0e7e6ffce7b019b8b45fab1cf3cd0d1e3983119f
```

`row_to_token`为28672个i32、114688 B，`topids_to_rows`为1536×8个i32、49152 B。上面比较直接基于已归档的原observer记录，没有新增GPU运行。

代码机制可参考`aiter/ops/flydsl/kernels/moe_route_maps.py`的`build_moe_topids_to_rows_module`：每route通过per-expert `llvm.AtomicRMWOp(add)`取得slot，写入`slot + expert * max_m`。`moe_contiguous_psum.py`的remap再写入`final_row = starts[expert] + slot`及`row_to_token[final_row] = route // topk`。源码机制允许slot顺序随执行改变；保存hash的直接事实是物理map bytes不同，不能仅由hash证明这是唯一变化或已解释性能差异。

现有v3/v80跨加载地址对照没有因此失效：同一个paired process中，两版共用同一份实际A/B/Scale/psum，并保留了前后full hash一致的证据。这里发现的是不同process、不同boot之间的输入比较边界，不能把它当作已经定位的新boot +3.6413 μs的唯一原因，也不能把旧paired输入证据替代旧正式77.6057 μs测量中缺失的输入记录。

### 准备中的 A 核查：不改路由、不改数据、不计时

额外输入审计已在本地准备并通过AST语法检查，尚未freeze、上传或启动。先完成并独立核验已有`f3_newboot_context_completion_20261027_v2`缺失两轮，才允许freeze下一份有限计划。下一份计划固定仅一个实际child，复用v60 production source/cache/ELF，执行原`--scenario verify`；不调用`run_perftest`，不新增GEMM launches，不挑选load address，不修改kernel、准入条件或precision gate。

```text
worker=.codex_tmp/hyg_f3_followup_20261024/probe_f3_live_A_original_verify_20261027_v1.py
controller=.codex_tmp/hyg_f3_followup_20261024/run_f3_live_A_original_verify_20261027_v1.py
```

observer只在原verify结束后读取实际输入，核对每个expert的`row_to_token[start:end]`与原topk routes的source-token multiset相符、无效行保持无映射，再按documented `(M/16, K/128, 4, 16, 16-byte)` payload layout在CPU恢复row bytes。分别记录物理有效行、按`(expert, source token)`排列的canonical有效A、TDM读取M16、M16尾部padding和完全不读取tiles的hash。canonical化只用于CPU hash比较，不改变设备上的输入或routing。

每个实际child仍使用118s/2s限时，前置全机11次/至少60秒idle，结束后记录post与每6秒settled检查。只保留observer的region hash和已核验的worker/geometry来源；若未保存原region bytes，review需明确不能在本地独立重算这些bytes的hash。旧正式77.6057 μs没有valid A hash，未来新记录也不能补造这一缺失证据。

截至2026-10-27 10:30:41 UTC，已有v80→v60补采summary仍为`completed=False, rounds=[]`；原等待程序继续每6秒复查全机，尚未启动本action的实际child。一般新GEMM1优化方法仍暂停，80 μs采用门槛未修改。两端三份文档应分别CAS追加本章，保留各自历史，不整文件覆盖。

## 2026-10-27：新 boot 补采与有效 A 审计完成，继续核查 I-cache 和窗口内 clock

问：汇报下进展，然后继续。

答：新 boot 正式三轮仍为 v60 GEMM1 中值83.4505 μs、v80中值81.2470 μs，改善2.64%；原精度通过。v80尚未满足预先冻结的三轮全部低于80 μs采用门槛，两端production继续保留v60。本节完成了此前缺失的v80→v60诊断补采及一次不计时的有效A审计，没有新增或替换正式kernel/MoE e2e性能。一般新的GEMM1优化方法仍暂停。

### 缺失两轮已完成，旧中止 action 继续保持 incomplete

`f3_newboot_context_completion_20261027_v2` 已完成固定v80→v60两轮，实际child分别38.694127/38.619083秒、退出码均0，每轮启动前均通过11次、间隔6秒且跨度至少60秒的全机idle检查。每个child结束后立即CAS恢复v60，保存原始post，再每6秒等待settled idle。本次两轮post均为owners为空、VRAM=176582656 B、busy=2；6秒后的settled均完全idle，没有本轮KFD/测试进程残留。

39成员归档已独立逐SHA核验，并与原v1首轮合并比较。原v1的`diagnostic_action_completed=False`、`all_post_gates_passed=False`没有改变；补采review的通过表示缺失观察已经收齐，不把旧中止action改称成功。

```text
archive=.codex_tmp/hyg_f3_followup_20261024/logs/f3_newboot_context_completion_20261027_v2_evidence.tar.gz
archive bytes=272707
archive SHA256=5ea5bb7f3402f2f232ee2f95900b0e4219dd37bb54284586cd3d96baec5ff82a
review=.codex_tmp/hyg_f3_followup_20261024/logs/f3_newboot_context_completion_20261027_v2.independent_local_review.json
review SHA256=4214c2d7dd7431c1834cd63bebd99f829f9849de4950415d6b0c4e111a0a6235
```

以下为观察用诊断值，全部排除在正式性能三轮之外：

| 保存记录 | 版本 | GEMM1 μs | GEMM2 μs | GEMM1 public HSA load base | 64 KiB aligned |
|---|---|---:|---:|---|---|
| 原v1首轮 | v60 | 83.562000 | 65.427474 | `0x7c8d15720000`，group2 | 是 |
| 本次补采r1 | v80 | 82.001579 | 65.733053 | `0x754319940000`，group0 | 是 |
| 本次补采r2 | v60 | 83.631526 | 65.357895 | `0x72df70638000`，group3 | 否，base低16位为`0x8000` |

两版实际完整ELF和loaded `.text`均匹配保存对象，v60/v80 `.text`仍为44416/97408 B。三份记录的logical hidden/topk ids/topk weights、expert route counts、B、ScaleA、ScaleB、psum、scalar geometry、最终MoE输出及reference均相同；GEMM1/GEMM2物理A buffer的full hash不同。对应原精度均为`logits_diff=3.61055e-06`，gate不变。

v80此次group0诊断值仍有新boot的耗时上移；不能只用旧group3慢档解释全部差异。后侧v60出现了非64 KiB对齐的实际base，也不能预设runtime每次都按64 KiB对齐加载。上述事实不能给未记录load base的正式三轮补上地址分类。

三轮GEMM1原timer返回后的SCLK样本分别2224/2225/2223 MHz，GEMM2样本2222/2222/2221 MHz，MCLK均1900 MHz、模式auto。这些都是timer外样本，不能代表实际运行窗口的平均clock，也不能唯一解释旧boot 77.6057 μs与新boot81.2470 μs的差异。

### 原 verify 的有效 A 审计：不计时、输入不变

在补采review通过后，执行预先准备的v1单child审计：生产source始终v60，仅将原CLI的`--scenario kernel`改为`--scenario verify`。原eager调用和精度检查保留，`run_perftest`调用数为0，没有额外GEMM、filler或加载实例选择。本次实际child30.440099秒、退出码0，结束后owners为空；post busy=3、6秒后settled完全idle，source SHA仍为`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。

此前为占用期间纯verify准备的v2计划已在本地freeze，但由于原补采已进入idle窗口，v2没有上传或执行；没有把它的early-context方案称为运行通过。实际执行的是原v1串行计划：

```text
controller=.codex_tmp/hyg_f3_followup_20261024/run_f3_live_A_original_verify_20261027_v1.py
worker=.codex_tmp/hyg_f3_followup_20261024/probe_f3_live_A_original_verify_20261027_v1.py
plan SHA256=41d15b4e1c57c0f4c8c056905fec593103c8bb48bfb2a785272c38d5d82d8b4e
archive=.codex_tmp/hyg_f3_followup_20261024/logs/f3_live_A_original_verify_20261027_v1_evidence.tar.gz
archive bytes=203370，members=26
archive SHA256=6ef87c7bcebf6f429a7d6a5c0d395031c7132869d6ab6bab893eac2ca3d9b80f
review=.codex_tmp/hyg_f3_followup_20261024/logs/f3_live_A_original_verify_20261027_v1.independent_local_review.json
```

原精度通过，`logits_diff=3.61055e-06`；表中的`us/gemm1_us/gemm2_us`均为空。观察器在原verify结束后，在CPU按documented `(M/16, K/128, 4, 16, 16-byte)`布局恢复row bytes。每个expert有效`row_to_token`与原topk source-token multiset一致，无效行映射均为负。分区结果如下：

| A 区域 | 数量 | bytes | 本次核查 |
|---|---:|---:|---|
| 严格有效routed rows | 12288行 | 44040192 | 按`(expert, source token)`计算canonical hash |
| TDM读取M16，含尾部padding | 797 tiles | 45703168 | 单独记录物理读取hash |
| M16尾部padding | 464行 | 1662976 | 本次全部为零 |
| 完全不读取M16 | 995 tiles | 57057280 | 单独记录，未归入数学有效输入 |
| 完整分配 | 1792 tiles | 102760448 | 保存full hash，未替代有效行hash |

```text
strict_live_rows_by_expert_token_SHA256:
6bfbf8257cc8cb7db6583e73dc83a8d68fc8412b5e244b616ac0f0d90e1e8f29
physical strict_live_rows_SHA256:
08b330ee39c7941c5c9c80dbb07a8372527f481e6c709fca4ea272e1bc48e3fb
row_to_token SHA256:
01f229bfd91beb1032d463854feeebc0c34be6cd1f1997931adb52bc4ebfe03b
```

这次audit确认自身有效行映射正确、尾padding为零；没有保存device A region的原始bytes，本地review核验的是worker来源、geometry、分区计数及observer记录，不能在本地独立重算region bytes hash。只有一次新的canonical记录，旧boot正式测量和旧paired记录未保存该hash；仍不能宣称跨boot有效A逐字节相同，或证明此前所有A full-hash差异只来自route排序。

首次本地review发现表格解析bug：`strip("| ")`删掉了末尾空的计时列，header16列、row误读为13列，导致`ValueError: zip() argument 2 is shorter than argument 1`。已保存初始reviewer和失败JSON，随后只移除外层两个`|`，保留16列中的空值，并明确断言计时为空；重新核验同一归档通过，没有重跑GPU或补造耗时。

```text
initial reviewer=.codex_tmp/hyg_f3_followup_20261024/logs/f3_live_A_original_verify_20261027_v1.initial_local_reviewer.py
failure=.codex_tmp/hyg_f3_followup_20261024/logs/f3_live_A_original_verify_20261027_v1.initial_local_review_parser_failure.json
```

### 下一项必要诊断：同 dispatch 的 SQG clock ratio 与 I-cache counters

已在当前`dsr1_ep4`只读核验live SDK，未升级环境或改clock；`flydsl=0.3.3.dev882`、PyTorch版本不变。当前SDK config完整SHA为`f328ce418eb98923f3bf60374428d71d9c6ce5c3931af6b19af6f4b28576f7ef`，profiler SHA为`baf11fe2bbf64c8389d4f08e81fed2e687f227b9395af3ff4d516ae19b33519d`，当前`rocminfo`单独保存，没有把旧boot metadata冒充本次agent信息。

`SQG_CYCLES`为gfx1250/SQG/event14，`SQG_REFCLKS`为gfx1250/SQG/event32。本地硬件资料`MI400_SH_PerfCounter_Users_Guide#71.txt` §4、p38–39说明同窗口CYCLES与REFCLKS可用于估算平均GFXCLK；REFCLK具体频率没有核实，因此本轮仅记录同dispatch的比值，不输出推算绝对MHz，也不把instrumented窗口clock当作原正式timer的clock。

复用已验收的v3/v80 frozen ELF/native probe，所有probe function/class AST不变；新caller差异已从HEAD原文件逆向逐byte核验，仅为原timer结束后的报告guard。采集同时记录`SQC_ICACHE_REQ/HITS/MISSES`、`SQ_WAVES`、`SQG_CYCLES/REFCLKS`，固定seed0、仅一个实际child，所有加载target正/逆序完整保留。source等待时保留v60，child后立即恢复；仍用11次/60秒全机idle及118s/2s限时，计时/profiling中不并发GPU查询、编译或传输。

```bash
# 本轮冻结的host命令；不能用相同label覆盖已经执行的证据。
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_newboot_v3_v80_SQG_counters_20261027_v2.py \
  counters --label f3_newboot_v3_v80_SQG_counters_20261027_v2 --runs 1 --seed 0
```

```text
plan SHA256=61a51641a472a715d71fa265c2cefceb9b784df48e199fb1a6ee8edc0c87a41d
controller SHA256=433e735c1156923180c908040dbfa51b38584ec34429100bc407f190081a81d8
child SHA256=4f058b96506acc4d4badbe1eadaf8a7d5e1b537d4876a6417acbf66e602392f4
```

counter工具v1仅做过CPU准备，未启动GPU；v2在新receipt中仅保留本次兼容与核验字段，将原工具的历史derivation描述留在原receipt，未改测量顺序、输入、timer、门槛或有效数学。截至2026-10-27 11:21:44 UTC，新controller仍在等待首次idle门禁，尚无actual child或counter结果；观察到外部`zan_att`任务，等待程序继续每6秒复查，没有杀外部进程。后续根据实际counter记录再判断，不能预先认定clock或I-cache已唯一解释新boot的+3.6413 μs。

E65 fixed shared1、E64/E65共用compiled GEMM1，以及已授权的英文commit/push、远端主机pull和pull后最终复测仍保留；本节没有完成或取消这些待办。两端三份文档分别CAS追加，保留各自历史。

## 2026-10-27：新 boot 的 v3/v80 四组 counters 完成核验，SQG clock 读数不能用于归因

问：汇报下进展，然后继续。

答：新 boot 正式三轮结果保持不变：pin=0 的 v60 GEMM1 中值83.4505 μs，v80中值81.2470 μs，改善2.64049%，减少2.2035 μs；原精度均通过，`logits_diff=3.61055e-06`。v80未满足预先冻结的三轮全部低于80 μs门槛，没有执行adopter或修改门槛，两端production仍为v60，没有新的MoE e2e性能。本节完成了已经采集的counter归档核验和跨boot计数对照，没有重新执行已结束的采集或替换正式读数。

### 实际采集、归档和来源核验

`f3_newboot_v3_v80_SQG_counters_20261027_v2`固定一个process、seed0，实际child18.291487秒，退出码0。启动前11次、间隔6秒且跨度至少60秒的全机idle，以及post/settled检查通过。共保留13个目标module和12个从不dispatch的filler；v3/v80均覆盖public HSA load-base group0–3。全部target正序和逆序保留，每个窗口仍为原46次launch，独立counter统计使用各窗口原profiler的tail20；初始冷dispatch单列。

120成员归档已下载，逐member检查bytes/SHA256后，本地reviewer逐dispatch绑定native function、public HSA加载地址、完整ELF、SDK Kernel/Dispatch/Agent ID和原窗口顺序，并核验实际输入在同一process内共用且前后未变、全部输出snapshot逐字节相同。该核验表示绑定及保存证据通过，不表示本节所有counter均适合物理解释。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
archive=TASK/logs/f3_newboot_v3_v80_SQG_counters_20261027_v2_evidence.tar.gz
archive bytes=2092966，members=120
archive SHA256=42fee0295a652490704b8b2caf040d7bc511e6fc13b9694c3fd5f529fd42bfe6
review=TASK/logs/f3_newboot_v3_v80_SQG_counters_20261027_v2.paired_counter_independent_local_review_v1.json
binding SHA256=ff1e17e8518f81c693a321f57b15cd9d914f4b34ecabc049544ddc9170e38faf
analysis=TASK/logs/f3_newboot_v3_v80_SQG_counter_analysis_local_20261027_v1.json
analysis SHA256=62007444bf945a4c918abd16c734d59de5e324ab7bc7d840d0d486e1a2131e69
```

本地CPU核验命令：

```powershell
python .codex_tmp/hyg_f3_followup_20261024/review_f3_newboot_v3_v80_SQG_counters_local_20261027_v1.py f3_newboot_v3_v80_SQG_counters_20261027_v2
python .codex_tmp/hyg_f3_followup_20261024/analyze_f3_newboot_v3_v80_SQG_local_20261027_v1.py
```

命令输出创建独占证据文件，已有输出不能覆盖后重跑。第二个分析器也核验旧seed0/17归档及binding SHA，所有旧记录保留。

### I-cache 的地址相关 miss 放大仍被明显压低

下面是各group内全部保留tail20 dispatch的`SQC_ICACHE_MISSES`中值，不是kernel耗时。旧boot seed0和新boot seed0不是同一process，不能仅据同seed宣称物理有效A逐字节相同。

| group | 旧boot seed0 v3 | 新boot seed0 v3 | 旧boot seed0 v80 | 新boot seed0 v80 |
|---:|---:|---:|---:|---:|
| 0 | 5875.0 | 5862.5 | 0.0 | 0.0 |
| 1 | 26122.0 | 25936.0 | 474.0 | 472.0 |
| 2 | 5875.0 | 5862.5 | 0.0 | 0.0 |
| 3 | 29109.5 | 29365.0 | 3081.0 | 3081.0 |

旧/新seed0的非零group miss中值相差不到0.9%。新boot v80 group1相对v3减少98.18%，group3减少89.51%；group0/2本轮所选tail20范围内计数为零。这支持v80的布局改善在本次新boot、64 KiB对齐的实际加载样本中仍存在，不能宣称所有输入/地址都零miss或跨boot正式耗时漂移已解释。

全部13个counter目标base低16位为0，本轮没有采到非64 KiB对齐目标。此前原入口v60诊断确实出现过base低16位`0x8000`；本轮counter证据不能替代非64 KiB加载条件的验收，也不能给未记录load base的正式三轮补上group标签。

所有原异常值保留：v80有一个tail20 dispatch（ordinal125、SDK Dispatch_Id667、forward `v80_clone2`、base=`0x76e298390000`、group1）miss=16219，SDK instrumented span=103.075 μs。该实例全部40个tail20 miss中值476，forward/reverse分别483/469，reverse范围428–488。没有因单次高值删dispatch、换实例或追加快轮；现有证据不能唯一归因该单次高值。

旧seed17计数仍全部保留：v80各group miss中值10476/11301.5/10452.5/11567.5，对应此前约89–90 μs的正常诊断计时。不能以seed0计数替代seed17，或承诺所有random输入低于80 μs。SDK instrumented span和本轮原timer打印值全部排除在正式性能之外。

硬件依据：`MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1记录I$的64KB、4 banks、4 ways、32 sets及`Set[4:0] = address[13:9] ^ address[18:14]`。SDK将REQ定义为per-SQ/per-Bank，HITS/MISSES标为nondeterministic，因此这里的miss/request只能作为经验事件比值，不能当成严格的cache miss probability。精确bank hash和live residency仍未唯一确定。

### 两项 SQG clock counters 返回同一固定值，clock 证据未成立

本地额外分析直接读取同一归档的原SDK CSV，确认包括初始冷dispatch在内的全部1197个GEMM1 dispatch，每项都为：

```text
SQG_CYCLES  = 12564068（0xbfb664），1197行只有1个唯一值
SQG_REFCLKS = 12564068（0xbfb664），1197行只有1个唯一值
ratio       = 1
```

初始冷dispatch的SDK span144.857 μs，与各温热窗口的span明显不同，但这两项仍不变。不能把ratio=1解释为实际GFXCLK/REFCLK比例、稳定clock、相同boot clock或对正式三轮clock的测量。已在新增analysis中明确标记`SQG_clock_counters_not_validated_for_clock_inference=True`，保留原CSV和原review结果，没有把采集配置存在及binding通过等同于counter物理有效。

当前installed SDK config仍为完整SHA`f328ce418eb98923f3bf60374428d71d9c6ce5c3931af6b19af6f4b28576f7ef`，其中gfx1250 `SQG_CYCLES/REFCLKS`定义为SQG/event14/32，未升级或修改环境。本地资料`MI400_SH_PerfCounter_Users_Guide#71.txt` §4 p38–39说明CYCLES与REFCLKS可用于估算同窗口GFXCLK，但本轮固定读数未满足这种解释所需的测量有效性，REFCLK实际频率也未核实。现有证据未唯一确定是block配置、寄存器读回、SDK汇总还是其他采集环节的问题。

对跨boot正式v80中值77.6057→81.2470 μs的+3.6413 μs，当前可确认的边界是：相同source/full ELF；新boot本次I-cache各group中值保持旧有模式；跨boot有效A bytes、原正式timer实际load base和窗口内clock证据仍不齐。因此不能唯一归因clock或I-cache，也不能重新挑选正式快轮。

### 下一步与保留状态

继续优先核查clock counter采集链路和实际加载对齐的覆盖范围，只进行必要诊断；一般新的GEMM1调度优化仍暂停。所有compiler/test/profiler child继续118s/2s限时，性能/counters前全机11次/60秒idle，运行期间不并发GPU查询、编译或传输。生产source SHA保持`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`，分支/HEAD仍为`hyg_dsr1_conc6144/e99dbac65a92ed23f87df4d99bd38f655a923a3b`，新boot为`a94beab0-6a60-46c6-9347-c90fb32d31fd`。

E65 fixed shared1、E64/E65共用compiled GEMM1，以及已授权英文commit/push、远端主机pull和pull后最终复测仍为待办；本节未完成或取消这些任务。三份主文档两端分别CAS追加，保留各自历史。

## 2026-10-27：完成非 64 KiB 静态覆盖和 loader 分配检查，独立 clock probe 等待空闲

问：汇报下进展，然后继续。

答：本轮继续处理 v3 的加载地址敏感性和 v80 的验收边界，一般新的 GEMM1 调度优化仍暂停。没有修改 production kernel，也没有新增正式性能数据。新 boot 已冻结的 E64/T1536/topk8 原入口三轮中值仍为 v60 GEMM1 **83.4505 μs**、v80 **81.2470 μs**，改善 **2.64049%**，减少 **2.2035 μs**；原精度三轮均通过。v80 未满足三轮全部低于 80 μs 的采用门槛，未执行 adopter，production 仍为 v60。没有新的 MoE e2e 性能，E65 fixed shared1 仍不能宣称已完成共用 compiled GEMM1 验证。

### 非 64 KiB 对齐的静态覆盖

使用与已核验完整 ELF 逐字节绑定的 ISA，扫描 16 个 4 KiB offset × load-base group0–3，保留 4/6/7/8 行 coverage 的原语义。没有初始化 GPU runtime，未采集新性能或 counters。

v80 六行热点循环所选 instruction lines 的 PC 范围为 `0x7d80–0xb180`。当 base 低 16 位为 `0x8000` 时，这些地址跨越 64 KiB fold，部分相对 set 关系随 group 改变；四行 body 也存在类似条件。该结果只是静态映射的几何差异，不是已经实测的 cache conflict 或性能回退，未假定 bank hash 或 live residency。

本次新 boot counters 的 13 个目标实际 base 全部 64 KiB 对齐。此前原入口 v60 出现过 base 低 16 位 `0x8000`，不能据此断言同一 v80 ELF 也会出现这种对齐。

硬件依据仍为 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1 的 `Set[4:0] = address[13:9] ^ address[18:14]`，没有把静态 set 等同性当成严格 cache miss probability。

### loader 对齐需要按完整 load allocation size 和实际分支判断

逐字节核验实际 ELF 的 `PT_LOAD`，不是只看 `.text` 大小。所读 loader 源码 `runtime/hsa-runtime/loader/executable.cpp:1355–1382` 的 `LoadSegmentsV2` 使用最后一个 DataSegment 的 `vaddr + memSize` 作为整个 code object 的分配 size；这些 ELF 的首个 load vaddr 为 0。

| 版本 | `.text` B | 完整 load allocation size B | 条件性的 mmap 起始地址对齐 B |
|---|---:|---:|---:|
| v60 | 44,416 | 57,344 | 32,768 |
| v3 | 97,408 | 110,592 | 65,536 |
| v80 | 97,408 | 110,592 | 65,536 |

表中对齐列以 4 KiB page、max alignment 至少 64 KiB、未指定固定地址、走所检查的 mmap 分支为条件。`libhsakmt/src/fmm.c:905–921` 的循环按照 size 提高 alignment。`amd_loader_context.cpp:300–325` 设置 `AllocateExecutable`；`amd_kfd_driver.cpp:298–332` 将 executable 排除在普通 VRAM suballocator 之外，随后进入 KFD 分配路径。

同时检查了容器已安装 `libhsa-runtime64.so.1` 的 x86 ISA，保留原 opcode 和 PC。`mmap_aperture_allocate_aligned` 中 `0x1e33e0–0x1e33f1` 的比较/倍增循环支持所述按大小提高 alignment 的机制；`hsaKmtAllocMemoryCtx` 也能看到将 explicit alignment 设为 0 的路径。安装库 SHA256 为 `4e1a28dc1dfdb0f34b03193c37920b3421f93839be753b0c537047d3b5e83b08`。

需要保留的限制：`reserved_aperture_allocate_aligned` 是另一个分支，在原请求 alignment 不大于 page 时会对齐 buffer 尾部，不能套用 mmap 起始对齐。按所读算法，v60/v80 的相应 end-alignment offset 分别为 8,192/20,480 B。实际此前 dispatch 走哪个分配分支尚未观测；源码树 HEAD 为 `07ce631b1384f6c9aefc6244e8c481fa48ae8062`，所检查文件相对 HEAD 无差异，但尚无 source-to-installed-build 的完整对应证明。因此表格和安装库静态 ISA 检查仍不能升级成任意 runtime/环境都保证 64 KiB 对齐的结论。当前容器 `HSA_MAX_VA_ALIGN` 未设置，仅表示当前静态检查进程的环境，不代表此前全部 dispatch 的环境。

### 独立 SQG clock health probe 已编译，尚未启动 GPU child

此前全部 1,197 个 GEMM1 dispatch 的 `SQG_CYCLES` 与 `SQG_REFCLKS` 均为同一固定值 `12564068`，不能用于推断实际 clock。为核验采集路径，准备了独立 HIP probe：同一 wave 内读取 `s_get_shader_cycles_u64` 和 `s_sendmsg_rtn_b64 ..., 0x83`，后者带 `s_wait_kmcnt 0`，执行固定整数循环的 6 个动态样本，iterations 顺序为 `4096/65536/1048576/1048576/65536/4096`，全部样本保留，检查 accumulator。这个 probe 不实现或计时 MoE，也不修改原 MoE timer。

v1 compile 因环境缺少 `hipcc` 默认查找的未版本化 `libamdhip64.so` 而失败，耗时 0.814922 秒、退出码 1，原日志保留。v2 保持相同 CPP bytes，只在新 task stage 创建私有 HIP linker root，链接现有 `/opt/rocm/lib/libamdhip64.so.7`；未修改安装文件。v2 compile 耗时 **0.814971 秒**、退出码 0，保存的 ISA 中 TIME、REALTIME 和 KMCNT 等待各有两处，已核验。

截至 `2026-10-27 12:07:35 UTC` 的检查，原 collect controller 仍处于空闲等待阶段，无 counters directory、无 actual GPU child。先后观察到外部任务占用，等待程序继续每 6 秒检查全机；没有杀死外部进程或使用占用期耗时作为性能。CPU-only 静态证据传输在已确认门禁被外部任务阻止、actual profiler 尚未启动的窗口内完成，12 个任务文件的 bytes/SHA256 两端一致。实际 compiler/test/profiler child 继续使用 118s/2s 限时，controller 的空闲等待时间与 child 执行时间分开记录。

硬件依据为 CDNA5 ISA §3.4.11/3.4.11.1：TIME 只在同一 wave 内比较 delta；REALTIME 为固定频率时钟，其本机实际频率尚未核实。该独立 probe 即使完成，也仅能核查本 probe 的读数与 SDK 采集路径，不能补造此前正式 GEMM1 窗口的 clock，不能唯一解释跨 boot 的 77.6057→81.2470 μs 变化。

### 本轮静态证据与后续顺序

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
non64 report=TASK/logs/f3_v80_non64KiB_set_geometry_local_20261027_v1.json
non64 SHA256=1cf1d7583f49267401f8e5bfd14f54e8bf041391a4c7164c702b576de7a1ffd1
ELF size report=TASK/logs/f3_loader_ELF_sizes_local_20261027_v1.json
ELF size SHA256=8e2b09c6b914d78f7448bd227ba67d2c4d9a73cdfb82bd035955754815634b94
loader source/installed ISA report=TASK/logs/f3_loader_static_remote_20261027_v1.json
loader report SHA256=cafd54f05ec2ab813fab79f2a61e81d9481a8dec6b5b342dc4f7e59493e40084
upload receipt=TASK/logs/f3_static_and_review_upload_remote_20261027_v1.json
upload receipt SHA256=f1b38599905a98dafd5f8008a280651521ff4222ee5128fbaf350bb0a0ae694d
local static reviewer=TASK/review_f3_loader_static_local_20261027_v1.py
clock frozen plan SHA256=7aedda512488ad9a47f3391ce88c99ad88ad1b4fc185ff0e3cf5c4e84ee9e9ae
```

继续复用已运行的 collect，完成后核验全部 6 个样本与 SDK dispatch binding，不重跑来替换异常或慢读数。证据归档明确排除 `.out` 中间文件，保留其远端 hash 记录；没有将未下载的完整安装库和中间文件称为本地逐字节核验。根据实际 clock health 结果再确定必要的地址诊断。80 μs 采用门槛不变，70 μs handoff 目标不变；E65 fixed shared1、E64/E65 共用 compiled GEMM1，以及已授权英文 commit/push、远端主机 pull 和 pull 后最终复测继续保留。


## 2026-10-27：独立 probe 确认当前 SQG clock 采集读数无效

问：汇报下进展，然后继续。

答：独立 clock health probe 已完成，本地核验通过。执行量相差 256 倍的整数循环中，直接 TIME/REALTIME delta 正常变化，`SQ_WAVES=1`，但 SDK 的 `SQG_CYCLES` 与 `SQG_REFCLKS` 仍在全部六个 probe dispatch 中返回同一固定值 **12,564,068**。当前这两项 SDK 采集读数不能用于推断实际 clock；同样异常可以在独立 probe 中复现，不能归因于 MoE 计算、v80 的 body 或 MoE 输入布局。现有证据仍未唯一确定是 selector/block 配置、寄存器读回、SDK 汇总或其他采集环节的问题。

这次是 counter 有效性诊断，没有新增正式 GEMM1/GEMM2 或 MoE e2e 性能。新 boot 的正式 E64/T1536/topk8 三轮中值仍为 v60 GEMM1 **83.4505 μs**、v80 **81.2470 μs**，原精度均通过。v80 未满足三轮全部低于 80 μs 的冻结采用条件，production 仍为 v60。

### 六个原始 probe 样本全部保留

同一 wave 内以 `s_get_shader_cycles_u64` 读取 TIME，以 `s_sendmsg_rtn_b64 ..., 0x83` 和 `s_wait_kmcnt 0` 读取 REALTIME。单个动态 probe kernel 依次执行下列固定循环量，逐样本验证 `accumulator = 1 + 17 * iterations`，没有换顺序或追加快样本。

| ordinal | iterations | TIME delta cycles | REALTIME delta ticks | SDK Dispatch_Id | SQG_CYCLES | SQG_REFCLKS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4096 | 81,929 | 3,744 | 1 | 12,564,068 | 12,564,068 |
| 2 | 65536 | 1,310,729 | 58,812 | 3 | 12,564,068 | 12,564,068 |
| 3 | 1048576 | 20,971,529 | 940,492 | 5 | 12,564,068 | 12,564,068 |
| 4 | 1048576 | 20,971,529 | 940,472 | 7 | 12,564,068 | 12,564,068 |
| 5 | 65536 | 1,310,729 | 58,816 | 9 | 12,564,068 | 12,564,068 |
| 6 | 4096 | 81,929 | 3,704 | 11 | 12,564,068 | 12,564,068 |

六条 `SQ_WAVES` 均为 1，metadata 为 gfx1250、wave32、grid1/block32。同一 probe 的两条 1,048,576-iteration 样本的 TIME/REALTIME ratio 分别为 22.298466122/22.2989403193；这里只描述这两条同 wave 观测，REALTIME 的本机实际频率未独立核实，未换算绝对 GFXCLK MHz，也不能推断原正式 GEMM1 窗口的 clock 或旧 boot 的 clock。

硬件依据为 CDNA5 ISA §3.4.11/3.4.11.1：TIME counter 不跨 SIMD 同步，只能用于同一 wave 内的 delta；REALTIME 是固定频率的计数源。`MI400_SH_PerfCounter_Users_Guide#71.txt` §4 p38–39 对 CYCLES/REFCLKS 的解释以 counter 测量有效为前提，本次固定值不满足该解释的条件。

### 空闲、超时、绑定及本地核验修复

保留此前运行的 collect controller，全机11次/至少60秒空闲门禁最终通过。实际 child 在 **2026-10-27 12:18:44 UTC** 执行，耗时 **0.464424 秒**、退出码0，post/settled均空闲，无本轮GPU进程残留。编译/采集 child 均有118s/2s限时；此前长时间只是等待外部任务退出，未并发运行本轮GPU实验。actual child 期间没有GPU查询、编译或文件传输，生产source保持原SHA256。

初版本地 reviewer 预期 trace 中只有六条 probe；实际 SDK trace 还记录了六次 HIP 内部 `__amd_rocclr_copyBuffer`。本地 reviewer 因 trace 数断言而退出1，未修改GPU归档、CPP或计数值。v3 reviewer 明确绑定12条原trace的固定交替顺序：probe Dispatch_Id=1/3/5/7/9/11，copy=2/4/6/8/10/12；copy均在相应probe结束后，18条counter记录只绑定六条probe，全部原copy trace继续保存。这个修复仅改变本地解析与记录，不改变计时/采集。

本地派生器另有一次记录生成失败：手抄归档SHA字面值为65字符，导致生成正确v3 reviewer后停止写receipt。恢复检查直接核验下载归档与manifest，确认原归档完整，派生v3源码逐字等于声明的解析改动，并核验v3结果、原v2与派生器SHA。失败文件与失败状态均保留，没有重复任何GPU试验。该记录错误不表示实际归档SHA不一致。

### 证据和复现命令

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
archive=TASK/logs/f3_SQG_clock_health_20261027_v2_evidence.tar.gz
archive bytes=668344，members=30
archive SHA256=de8d9dded0b0dd3d13a31a0f5d95f5ac1b1a05694f77feaea846412c4ee142ea
review=TASK/logs/f3_SQG_clock_health_20261027_v2.independent_local_review_v3.json
review SHA256=20ad272897af32b729d170a17ba45ae5a86a29666582ea1ca6b48929dd5ea0ef
provenance receipt=TASK/logs/f3_SQG_clock_health_review_v2_parser_failure_and_v3_derivation_20261027.json
provenance SHA256=2481279fa95e8bc337ab9b662b5635fd6fe07906eb74ab43714357ecaf4d0417
```

主机已执行的compile/collect命令如下；对应独占label已经使用，不能重跑覆盖结果：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_SQG_clock_health_20261027_v2.py compile
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_SQG_clock_health_20261027_v2.py collect
```

本地已执行的CPU核验命令：

```powershell
python .codex_tmp/hyg_f3_followup_20261024/review_f3_SQG_clock_health_local_20261027_v3.py
python .codex_tmp/hyg_f3_followup_20261024/recover_f3_clock_health_review_receipt_20261027_v1.py
```

归档排除一个`.out`编译中间文件，manifest保留其远端hash记录；实际host binary、保存ISA、CPP、原日志、counter CSV与门禁记录在本地核验，未下载的中间文件不称为本地逐字节核验。v1缺少未版本化HIP linker name的编译失败证据继续保留；v2使用task私有linker root，没有修改安装环境。

### 对当前优化决策的影响

前文v80压低64 KiB对齐group0–3地址相关I-cache misses的有限实证继续保留，不能把本次SQG读数当成跨boot clock归因。重启后v80 77.6057→81.2470 μs的变化仍未唯一解释，正式三轮不替换，seed17约89–90 μs的历史诊断也不排除。一般新的GEMM1调度优化仍暂停。

下一步使用已有rocgdb仅观测独立module加载中的CPU allocator入口/返回，与public HSA `load_base/load_size`绑定，核实当前环境实际采用mmap还是reserved分配规则。该观测会明确独立加载对照和原benchmark的范围区别；原benchmark此前未记录的分配分支不会被补造。E65 fixed shared1、E64/E65共用compiled GEMM1，以及已授权英文commit/push、远端主机pull和pull后最终复测仍为待办。三份文档两端分别CAS追加，保留各自历史。


## 2026-10-27：加载观测已冻结，纠正主机与容器 HIP 身份绑定

问：汇报下进展，然后继续。

答：本轮继续处理 v3 加载地址快慢分档和 v80 修复验收，没有尝试新的 GEMM1 调度优化，也没有修改 production kernel。重启后保留的正式 E64/T1536/topk8 三轮结果仍为：v60 GEMM1 **83.4505 μs**、v80 **81.2470 μs**，改善 **2.64049%**；三轮原精度均通过。v80 没有达到三轮全部 `<80 μs` 的冻结采用条件，production 仍为 v60。没有新的正式 GEMM1、GEMM2 或 MoE e2e 性能。

此前新旧 boot 对照中，v80 压低实际 64 KiB 对齐 group0–3 的 `SQC_ICACHE_MISSES` 的效果仍成立；独立 clock health probe 则确认当前 `SQG_CYCLES/SQG_REFCLKS` 固定读数不能用于 actual clock 或跨 boot clock 归因。重启后的正式耗时上移尚未唯一归因，原三轮和此前异常记录全部保留。

### 当前加载观测的目的和范围

已通过已安装 debugger 的离线 CPU symbol 检查：HSA allocator 入口分别是 `mmap_aperture_allocate_aligned`（library-relative `0x1e3380`）和 `reserved_aperture_allocate_aligned`（`0x1e2ff0`），四个 HIP load/launch API symbol 也可定位。离线检查没有初始化 GPU runtime。

独立观测冻结了原完整 ELF、public HSA observer 和固定顺序 `v60 → v3 → v80 → v80 → v3 → v60`。不重新编译或改写 ELF，不请求 GEMM launch，不采性能；全部六次加载及失败记录保留。CPU breakpoint 记录入口 size、requested alignment、caller stack 和返回地址，再与 public HSA `load_base/load_size`、完整 ELF storage 和 loaded `.text` SHA256 绑定。设置 `set disable-randomization off`，不设置 code placement、`HSA_MAX_VA_ALIGN` 或 clock。

该实验只能证明当前独立 module 加载的实际 allocator 分支；它不是原 benchmark 此前未记录的分支证据，不构成不同加载地址下的正常计时验收，也不能据此接受 v80。

### 首轮环境校验失败及修复

v1 child 在 **0.214015 秒**退出1，停在 HIP library SHA256 断言处，尚未执行 `ctypes.CDLL`、`hipInit` 或 module 加载。debugger 日志只有启动和 inferior 退出，无 HIP call、HSA allocator hook 或 probe result；post/settled GPU 空闲。失败归档的全部19个 member已在本地核验，失败源码与日志未覆盖。

原因是沿用了此前 clock compile controller 的 `installed_HIP_library_SHA256` 字段。该 controller 在主机执行，其 `Path("/opt/rocm/lib/libamdhip64.so.7").read_bytes()` 读取的是主机文件；task-private linker symlink 和 probe 的实际执行都在容器中，同名绝对路径解析到容器文件。不能把这个主机字段当作容器 runtime 身份。

本次直接读取两端文件确认：

| 对象 | SHA256 | 当前观测范围 |
|---|---|---|
| 主机 HIP | `f49637debeecb69225e5bf44d178fc902f627db0a55e7d8ad12b89a099b2f69f` | 主机 `/opt/rocm/lib/libamdhip64.so.7` |
| dsr1_ep4 HIP | `4a3318b0245da02ee3e1203585db96041c6646b7899d6559fef9cea2a10ce8e6` | 容器 `_rocm_sdk_core/lib/libamdhip64.so.7`，28,320,337 B |
| dsr1_ep4 HSA | `4e1a28dc1dfdb0f34b03193c37920b3421f93839be753b0c537047d3b5e83b08` | 与此前 HSA 观测身份一致 |
| dsr1_ep4 debugger | `9297733ad81af96a1eee10b81a109930ec296ad52177bb394acb0fe2e875aa31` | 与此前 debugger 身份一致 |

这是观测脚本的环境绑定错误。没有证明发生了 runtime 安装替换，也没有证明旧 boot 的实际 HIP 身份。此前 clock probe 的 binary、ISA、原 counter 数据和 TIME/REALTIME 样本不改动；仅明确 compile summary 的 HIP hash 字段来自主机，不能据此声称该轮实际容器 runtime 身份已核验。

v2 使用当前容器 HIP 的真实路径与 hash，保留 v1 probe/GDB observer 和六次输入顺序；controller 只改独占 label。16个 frozen 文件在两端 bytes/SHA256 校验通过。

### 门禁、证据和后续

v2 controller 已开始全机11次、至少60秒的连续空闲等待，每6秒重新检查。最后读取的空闲日志截至 **2026-10-27 12:51:58 UTC**：等待期间新外部 GPU/KFD 工作出现，空闲计数重新开始，在这个时点尚未运行 v2 GPU runtime child。本节尚未得到通过核验的 allocator 返回/HSA 绑定。实际诊断 child 仍设置 `timeout --signal=TERM --kill-after=2s 118s`；parent 等待时间与 child 执行时间分开记录。child 期间不进行 GPU 查询、编译或文件传输。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
v1 archive=TASK/logs/f3_loader_allocator_20261027_v1_evidence.tar.gz
v1 archive SHA256=cc3720e108058d30f6a694e7b24a834ab294a620f9f83e049f323eaa56cfb45d
identity report=TASK/logs/f3_loader_host_container_library_identity_20261027_v2.json
identity report SHA256=4029b09f0013957b833e32603fcce2d9ee1afcee4b6ec1ad04c1ccd0ae87a7cb
failure/derivation receipt=TASK/logs/f3_loader_allocator_v1_failure_and_v2_binding_20261027.json
receipt SHA256=cb07cec668f4617667d0fc4afa7173c88d7c811ab681347006a6d4b0ebc50351
v2 plan=TASK/logs/f3_loader_allocator_20261027_v2.frozen_plan.json
v2 plan SHA256=a112fd0497b9eff6d6613a95e8626ba58e9b2aecd189666a00639e42ff897e00
```

主机已启动命令如下，对应独占 label 不能重跑覆盖：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_loader_allocator_20261027_v2.py
```

下一步等待当前 controller 完成，逐字节核验归档，绑定全部六次 allocator 返回与 HSA 对象，再决定必要的地址修复诊断。只使用 a07-3 / dsr1_ep4，后续性能保持 `pin=0`。E65 fixed shared1、E64/E65共用 compiled GEMM1，以及此前已授权英文 commit/push、远端主机 pull 和 pull 后最终复测继续保留为待办。三份文档分别 CAS 追加，保留各自历史。

## 2026-10-27：六次独立加载的 mmap allocator 返回与 HSA 地址绑定通过

问：汇报下进展，然后继续。

答：第二轮独立加载观测完成，本地全部核验通过。固定六次调用均走 `mmap_aperture_allocate_aligned`，入口 `requested_alignment=4096 B`；allocator 返回地址与 public HSA `load_base`、size 与 `load_size` 精确一致。每次 caller stack 均包含 `LoadSegmentsV2`，完整 ELF storage 与 loaded `.text` SHA256 通过核验，六个 newly observed HSA 对象各不相同，所有 module 保持存活到观测完成后统一卸载。

本轮没有请求 GEMM launch，没有新增正式性能。新 boot 保留三轮的 GEMM1 中值仍为 v60 **83.4505 μs**、v80 **81.2470 μs**，原精度均通过；v80 未满足三轮全部 `<80 μs`，production 仍为 v60。77.6057→81.2470 μs 的跨 boot 变化尚未唯一解释，不能用无效 SQG 读数归因 clock。

### 全部六次加载原样保留

| ordinal | variant | allocator return / HSA load_base | size / HSA load_size B | base low16 | group |
|---:|---|---|---:|---|---:|
| 1 | v60 | `0x71559c630000` | 57,344 | `0x0` | 3 |
| 2 | v3 | `0x7155983e0000` | 110,592 | `0x0` | 2 |
| 3 | v80 | `0x7155983a0000` | 110,592 | `0x0` | 2 |
| 4 | v80 | `0x715598360000` | 110,592 | `0x0` | 2 |
| 5 | v3 | `0x715598320000` | 110,592 | `0x0` | 2 |
| 6 | v60 | `0x71559c608000` | 57,344 | `0x8000` | 0 |

四次 v3/v80 实际基址均按64 KiB对齐，本次只遇到 group2；没有追加加载去挑 group、没有执行 GEMM，也不称为 group0–3 正常计时验收。此前计时/counter 对照的 group 覆盖是另一个已保留实验。

v60 的实际基址同时出现 low16=`0x0` 和 `0x8000`。与此前源码条件分析相符，其57,344 B分配在mmap分支可按32 KiB对齐，不能统一假设 v60 每次也按64 KiB对齐。`.text <64 KiB` 本身不保证实际放置后不会跨64 KiB段。下一步优先检查已保留的正式 baseline HSA 加载记录；尚未证明这个差异解释了 baseline 或 v80 的跨 boot 耗时变化。

这些地址只绑定当前独立 module 加载，不补造原 benchmark 此前未记录的 allocator 分支，不证明 reserved 分支不会出现，也不构成未来所有 runtime/加载地址的对齐保证。

### 等待、执行和归档核验

GPU空闲门禁最终通过：11次连续空闲，跨度至少60秒。v2 parent曾因新外部GPU/KFD任务重新计数；实际child于 **2026-10-27 12:55:19 UTC**附近开始，耗时 **0.564450582秒**、退出码0。child受118s/2s限时保护，期间没有GPU查询、编译或文件传输。post/settled全部GPU空闲，无本轮GPU进程残留；生产source始终保持v60 hash。

归档全部33个member的bytes/SHA256逐项通过本地核验，完整ELF、六个loaded `.text`、CPU breakpoint journal、probe/debugger结果、门禁、控制器和失败修复来源均保留。v1的主机/容器HIP身份绑定失败记录不删除；v2只修正当前容器HIP身份和独占label，observer与原ELF不改动。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
archive=TASK/logs/f3_loader_allocator_20261027_v2_evidence.tar.gz
archive bytes=265466，members=33
archive SHA256=257eb10ef9d580ddaa6a082fa6dff0eedbd3e0e829b445d6466e4f9be548b275
review=TASK/logs/f3_loader_allocator_20261027_v2.independent_local_review.json
review SHA256=827d464cbaebac8e50460c0faaf4c0e0317b9831a654d6b1f6f63a3c6e1381dd
```

已执行的主机命令与本地CPU核验命令：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_loader_allocator_20261027_v2.py
```

```powershell
python .codex_tmp/hyg_f3_followup_20261024/review_f3_loader_allocator_local_20261027_v2.py
```

### 等待期间补充的静态判断

已安装HSA library ISA中，`LoadSegmentsV2` 在PC `0x1655cc`将 `r8d`设为固定`0x100`，随后PC `0x1655e3`调用 `SegmentAlloc`。`RegionMemory::Allocate` 的PC `0xe2697`与`0xe26c8`调用 `Runtime::AllocateMemory(region,size,flags,out,node)`，这个调用签名没有alignment参数。与已保存源码 `executable.cpp:1365–1366`、`amd_loader_context.cpp:300–324`相符；源码树与安装build的完整对应关系仍未证明，但这两个已安装CPU ISA调用点已直接检查。

因此，仅设置函数alignment或ELF `p_align`，不能保证在该加载路径中控制GPU code allocation基址；它们可能改变代码相对布局或allocation size，不能把这种间接效果称为loader显式保证。没有据此修改production、runtime或ELF。

此前16个4 KiB offset静态扫描还显示：在条件性的reserved尾部对齐偏移`0x5000`下，v80六行热点PC `0x7d80–0xb180`跨64 KiB fold，group0–3相对set关系出现3种signature；四/七/八行在这个offset没有跨fold。依据仍是 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1的 `Set[4:0] = address[13:9] ^ address[18:14]`。这只是已保存静态几何的进一步解释，不假设bank hash或live residency，不推断miss率、性能回退或实际reserved路径。

继续暂停一般新的GEMM1调度优化，先检查baseline保留加载记录和v80地址修复的范围。只在a07-3/dsr1_ep4测试，性能保持pin=0；E65 fixed shared1、E64/E65共用compiled GEMM1及已授权commit/push、远端主机pull和pull后最终复测仍为待办。三份文档两端分别CAS追加，保留各自完整历史。


## 2026-10-27：原 caller 的地址记录缺口已确认，计时前只读观测已准备

问：汇报下进展，然后继续。

答：重启后已冻结的正式 E64/T1536/topk8、random non-balanced、pin=0 三轮结果保持不变：v60 GEMM1 中值 **83.4505 μs**，v80 **81.2470 μs**，改善 **2.64049%**、减少 **2.2035 μs**；GEMM2 中值分别为 **65.2188 / 65.2672 μs**。三轮原精度均通过，`logits_diff=3.61055e-06`。没有新增 MoE e2e 性能。v80 未满足冻结 adopter 的三轮全部 `<80 μs` 条件，production 保留 v60；距离 handoff 的 `<70 μs` 目标，v80 中值还差 **11.247 μs**。这两个阈值分别是本次 source 采用条件和后续优化目标，不能混用。

### 原六轮不能事后指定加载地址

本地只读审计已完成：v60 baseline 和 v80 source qualification 两份正式归档，分别为56 / 48个member；逐项检查其中共64个 JSON、JSONL、log、CSV、Python记录，没有 `load_base` 或 `text_gpu_address` 字段。归档 SHA256 与原保留证据一致。此前独立 module 的 allocator/HSA 观测，不能补造这六轮原 benchmark 的历史加载地址或 allocator 分支。

原性能记录仍然有效；这个缺口限制的是逐轮地址归因，不能据此删除、重跑或替换已冻结三轮。旧 boot 相同 v80 source/full ELF 的77.6057 μs与新 boot 的81.2470 μs之间，尚未唯一确定原因；已经判为无效的 `SQG_CYCLES/SQG_REFCLKS` 固定读数仍不得用作 actual clock 归因。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
audit=TASK/logs/f3_original_caller_address_coverage_local_20261027_v1.json
v60 archive SHA256=3b527f8a4386df2057da35a94ebbf6f7d6922a1d4b4bce11ed2e19af31ec27fd
v80 archive SHA256=5eff055d5e83dbbd3d7d17e538664067ade6678ce4b2438cf5683c5eb79d2718
```

### 新观测保留原 caller、timer、输入与精度门禁

已确认原 caller 的 `--scenario kernel` 先执行一次 eager `_call()`，此时 `grouped_moe_gfx1250.kernel_bench_callable` 为list，然后恢复为None，才进入两个原 `run_perftest`。task-private bootstrap 在原 `CompiledArtifact._ensure_engine` 返回后、首次 eager launch之前读取该artifact已经解析的 `gpu.binary`；不重新parse、创建或替换IR对象，不改ELF、cache key、实际计算或callable。

针对两阶段，通过不变的 public HSA observer 核验原完整ELF storage及loaded `.text`，读取真实 `load_base`、`text_gpu_address`、low16、group和实际 `.text` 是否跨64 KiB段。两阶段记录完成后立即恢复原 `_ensure_engine`，再进入原warmup5/iters20、CUDA Event/torch profiler timer和原精度检查。不开启额外SMI replay，不强制加载地址或clock，不做额外module加载。

CPU语法与调用范围检查通过；当前容器的安装 `jit_executor.py` SHA256与冻结值一致。20个冻结文件两端bytes/SHA256一致。本次有限诊断顺序固定为 **v60 → v80 → v80 → v60**，每轮独立process，固定seed0/uniform/auto和pin0；使用各source此前通过原call生成并核验的正常cache。不根据group、读数或seed补跑，不把新诊断替换成原正式成绩。如果自然加载未覆盖group0–3，将保留这个验收范围限制。

```text
observer=TASK/f3_original_caller_load_observer_20261027_v1.py
runner=TASK/run_f3_original_caller_loads_20261027_v1.py
plan=TASK/logs/f3_original_caller_loads_20261027_v1.frozen_plan.json
plan SHA256=48416a088e0e4b907ee1b2112efaa4a036460cc88a744a29230adcea84bfbd4c
```

主机已启动如下命令；该独占label不得重跑覆盖：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_original_caller_loads_20261027_v1.py run
```

截至最近读取的日志（**2026-10-27 13:20:00 UTC**），第一轮连续空闲等待仍被新出现的外部GPU/KFD任务打断，尚未启动本轮实际test child。自动每6秒检查全机，不杀外部进程，要求11次、至少60秒连续空闲。实际child为 `timeout --signal=TERM --kill-after=2s 118s`；parent等待不等于child超时。child期间不并发GPU查询、编译或文件传输，结束后检查ownership和settled idle。source切换仅用于上述固定对照，并以原hash做CAS恢复；不接受v80作为production。

继续暂停一般新的GEMM1调度优化，先补齐原caller的地址证据。只在a07-3/dsr1_ep4测试；E65 fixed shared1、E64/E65共用compiled GEMM1，以及已授权英文commit/push、远端主机pull和pull后最终复测仍为待办。三份文档两端分别CAS追加，保留各自历史。

## 2026-10-27：原 caller 计时前加载观测完成，失效性能轮次保留

问：汇报下进展，然后继续。

答：本轮完成了原 caller 的只读加载地址观测，保留原输入、原 timer、warmup5/iters20、pin0和原精度 gate。固定顺序 `v60 → v80 → v80 → v60` 共执行四次，没有补跑、替换第三轮。四轮中，三轮性能有效；第三轮精度和代码身份通过，但测试后检测到外部GPU/KFD任务，其耗时明确排除。八个两阶段目标code object均完成原完整ELF storage、loaded `.text`和source IR绑定，hook在原计时前恢复。

已冻结的重启后三轮正式结果保持不变：v60 GEMM1中值 **83.4505 μs**，v80 **81.2470 μs**，改善 **2.64049%**、减少 **2.2035 μs**；GEMM2中值为 **65.2188 /65.2672 μs**。三轮原精度通过，`logits_diff=3.61055e-06`。没有新增MoE e2e性能。v80未满足冻结adopter的三轮全部 `<80 μs` 条件，production保留v60；距离handoff的 `<70 μs` 目标仍差 **11.247 μs**。新诊断不能替换正式数据，也不用于重新决定原adopter结果。

### 原定四轮的全部结果

下面是本次独立地址诊断，不是新的正式三轮成绩。第三轮原表中的数值保留为失效原始读数，不纳入有效性能比较或中值。

| ordinal | variant | GEMM1 μs | GEMM2 μs | 性能有效性 | GEMM1 base low16 | group | GEMM1 `.text` 跨64 KiB段 |
|---:|---|---:|---:|---|---|---:|---|
| 1 | v60 | 82.6554 | 65.2505 | 有效 | `0x8000` | 0 | 是 |
| 2 | v80 | 81.8439 | 65.3199 | 有效 | `0x0` | 2 | 是 |
| 3 | v80 | 81.1983（失效原始读数） | 64.9044（失效原始读数） | 排除 | `0x0` | 1 | 是 |
| 4 | v60 | 83.2118 | 65.3198 | 有效 | `0x0` | 3 | 否 |

四轮均通过原MoE e2e精度检查，`logits_diff=3.61055e-06`、`rel_l2=0.00268721`。第三轮的正确性与只读代码身份仍可使用，耗时不得作为有效性能；本轮没有报告新的比较中值或收益百分比。

真实GEMM1加载地址为：

| ordinal | variant | HSA load_base | text_gpu_address |
|---:|---|---|---|
| 1 | v60 | `0x74ace9308000` | `0x74ace9309a00` |
| 2 | v80 | `0x7ba5c5b60000` | `0x7ba5c5b61a00` |
| 3 | v80 | `0x7629e3650000` | `0x7629e3651a00` |
| 4 | v60 | `0x70606e630000` | `0x70606e631a00` |

GEMM2也记录完整地址与原ELF身份：第1/2轮base low16=`0x0`，第3/4轮为`0x8000`；第四轮的GEMM2 `.text` 实际跨64 KiB段，正常计时为65.3198 μs。不能统一假设所有目标module的实际code基址都按64 KiB对齐。

### 外部任务进入、排除第三轮与只补齐未执行第四轮

第三轮实际child退出0，耗时8.827441635秒，代码身份和精度通过。post于 **2026-10-27 13:41:01 UTC**、settled于 **13:41:07 UTC** 检测到外部PID `2371944`，来自`zan_att`，运行：

```text
python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 --experts 1 --tokens 16384 --topk 1 --model-dim 16384 --inter-dim 8192 --act silu --no-bias --iters 64 --no-check-aot-cache --iters 8
```

外部进程VRAM约2.65 GB，KFD owner存在；即使busy采样为0，也不满足空闲要求。该进程在本轮child附近出现，无法证明没有干扰计时。因此控制器将第三轮`performance_valid=False`并停止，归档原失败状态、全部日志和已完成读数，按CAS恢复v60源码；未杀外部进程，也没有重测v80来取得替代结果。

本地先核验37个member的中断归档，明确前两轮性能有效、第三轮排除。随后只冻结并执行原计划中尚未运行的ordinal4；前三轮未重新加载、重跑或改写。最后一轮使用原plan、原observer、原command、原v60 cache和相同输入，只把原command的ordinal1替换为原定ordinal4；源码始终为v60。完整四轮执行结果仍保留第三轮失效状态，不能称为四轮全部性能通过。

四个实际child耗时分别为 **9.127659336 /8.776648217 /8.827441635 /9.178284479秒**，均受`timeout --signal=TERM --kill-after=2s 118s`保护。总等待较长来自外部任务打断11次、至少60秒连续空闲门禁，不是test child超过两分钟。实际child期间没有GPU查询、编译或文件传输。第1/2/4轮post/settled全机空闲，无本轮GPU进程残留；第三轮的外部进程身份和排除原因保留。

### 这次观测支持和不支持的结论

- 已证明原caller可以在首次eager launch前读取实际加载地址，并核对完整ELF和loaded `.text`，随后恢复原执行路径再计时。不修改production runtime、IR、ELF、cache key、callable或计时函数。
- 原caller的v60同时出现32 KiB半段偏移和64 KiB对齐地址。两轮正常耗时分别为82.6554 /83.2118 μs，跨段这一事实本身不足以证明性能会回退，也不能由此解释正式baseline的跨boot上移。
- 有效v80轮实际64 KiB对齐，group2，正常耗时81.8439 μs。因此至少这一轮的`>80 μs`读数不能归因于条件性的reserved offset `0x5000`；本次没有观测到这个offset。仍然不能证明所有runtime/加载路径都不会出现该offset。
- v80本次代码身份覆盖group1/2，但group1的性能排除，只有group2的正常计时有效；本次不是group0–3的完整正常计时验收。此前已保存的新旧boot同ELF地址/counter对照保留原验收范围，不能用本次读数替换或补造其数据。
- 正式原六轮此前未记录的加载地址仍无法事后恢复。本次是新的独立caller诊断，不给旧三轮指定address/group或allocator分支。
- 旧boot相同v80 source/full ELF的77.6057 μs与新boot正式81.2470 μs之间仍未唯一归因。v80降低地址相关I-cache miss的已有证据不变；无效的`SQG_CYCLES/SQG_REFCLKS`固定读数继续排除，不据此声称actual clock变化。

硬件地址关系仍以本地`MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1的`Set[4:0] = address[13:9] ^ address[18:14]`为依据。`.text`是否跨段是地址几何，不直接证明cache miss、精确bank hash、live residency或dispatch瓶颈。

### 证据、复现与文档保留

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
中断归档=TASK/logs/f3_original_caller_loads_20261027_v1_evidence.tar.gz
bytes=173137，members=37
SHA256=6d3672857f490ec591cb93a47e52bef75f55fc20479bb34cbd8ca67c1562d9094
完整归档=TASK/logs/f3_original_caller_loads_20261027_v1_complete_with_exclusion_evidence.tar.gz
bytes=210681，members=49
SHA256=adaf10a512e8b8fb449bb646c5522aba527571b2340f26c697ac6c839f73ec89
完整本地核验=TASK/logs/f3_original_caller_loads_20261027_v1.complete_identity_review.json
SHA256=3da386e8527e4420d30c7184a4cd43f73334b0e5ebdc4f91a399a4f03d3439e4
第四轮补齐plan=TASK/logs/f3_original_caller_finish_unrun_r4_20261027_v1.frozen_plan.json
SHA256=46a407539877e9461c7847853477ac84e0ac9c89524c683efdc0cf24779d4c56
```

49个member的bytes/SHA256、四轮原日志表、完整ELF/loaded `.text`、source IR、计时前hook恢复、11次/至少60秒门禁、post/settled和child限时均经本地独立核验。原中断归档和原summary未覆盖；第四轮补齐summary的前三轮与原summary逐项一致。`evidence_review_passed=True`和`all_performance_valid=False`同时保留，不把证据核验通过写成所有性能轮次通过。

已执行的主机命令：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_original_caller_loads_20261027_v1.py run
python3 -u .codex_tmp/hyg_f3_followup_20261024/finish_f3_original_caller_unrun_r4_20261027_v1.py run
```

已执行的本地核验：

```powershell
python .codex_tmp/hyg_f3_followup_20261024/review_f3_original_caller_exclusion_local_20261027_v1.py partial
python .codex_tmp/hyg_f3_followup_20261024/review_f3_original_caller_exclusion_local_20261027_v1.py complete
```

### 等待期间核对的E64/E65共用要求

当前`launch_gemm_a8w4_tdm`的`n_experts`是`Constexpr[int]`，进入cache key、kernel symbol和expert bitsearch的展开范围。目标标准路径E64是6步，E65是7步。因此，相同源码文件/函数/tile参数不等于同一个compiled GEMM1；只修改E65准入仍会产生不同编译对象。

可研究共同metadata上界或runtime expert边界，但需要检查末项psum分配、重复/空expert、padding sentinel、两阶段alignment和真实weight范围。当前原caller的E65/topk9只是从65个expert随机routing，不定义fixed shared expert；固定shared1仍需按用户历史/模型语义单独构造route并验证完整MoE精度。上述改动尚未实现、编译或测试，不算完成共用要求。

静态范围说明已写入`TASK/notes/E64_E65_common_compiled_GEMM1_scope_20261027_v1.md`。没有开启一般新的GEMM1调度优化，也没有改变本轮原caller输入来完成该要求。E65仍为原t64/w1x4/b3 fallback；fixed shared1、E64/E65共用compiled GEMM1仍未满足，不能把已有fallback验证称为完成。

本轮新增诊断工具、证据和文档；production kernel维持v60 source SHA256。三份文档两端分别CAS追加，保留各自完整历史；原中断记录不删除，未下载`.out`。只在a07-3/dsr1_ep4执行GPU测试，后续性能保持pin0。继续保留已授权的英文commit/push、远端主机pull和pull后最终复测待办。

## 2026-10-27：E64/E65 共用 compiled GEMM1 的正确性原型通过

本轮在 a07-3 / `dsr1_ep4` 完成进程内原型验证，`pin=0`，生产 kernel 和 dispatcher 文件没有修改；没有新增性能数据。当前分支仍为 `hyg_dsr1_conc6144`，HEAD 为 `e99dbac65a92ed23f87df4d99bd38f655a923a3b`。普通新调度优化继续暂停，当前工作是补齐历史要求的共用 GEMM1 和 fixed shared1。

### 原型改动与边界

- 沿用当前 v60 的 GEMM1 计算源码，SHA256 为 `8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。只在测试进程内替换 `_grouped_a8w4_tdm_moe` 和 `grouped_gemm_gfx1250_a8w4` 的代码，退出时恢复。
- 目标 E64/topk8 和 E65/topk9 共用现有 E64 tuning row，统一两阶段的量化布局。在当前分支中该配置是 **t256/w2x2/b4**，不要与早期分支的 w2x4 配置混称。
- GEMM1 共用 constexpr expert 边界65；E64 的 route counter/psum 分配65项，新增末项计数为零。实际 E64 的权重依然只有64个 experts，GEMM2 仍用实际 E。E65 则保留真实 expert64。
- `tile_m == _align_m == 256` 是空末项安全性的明确条件。新增空项的 psum 可能把最后有效行结束位置向上取整，但不会增加 padding-only GEMM1 tile。没有删除有效 route 或有效计算。CPU 检查覆盖 E64 的108组计数和 E65 的18组计数，均通过。
- 准入限定 T1536、7168/2048、A4W4、Silu、BF16、无 bias、非 EP、非 compact、A-preshuffle 和已验证 tile/quant layout。其他调用保持原路径。E65/topk9 是否含 shared expert 不参与 kernel 的额外编译 specialization。

### 完整 MoE 正确性结果

保留原 caller、原初始化、seed0 和原 `logits_diff < 0.01` gate。先运行保留路径，再在相同输入上运行原型。所有输入/权重/scale/routes 的 SHA256 相同，FP32 reference 相同，输出在远端逐字节比较一致。

| workload | 原型 logits_diff | rel_l2 | 与保留路径的完整 MoE 输出 |
|---|---:|---:|---|
| E64/T1536/topk8/shared0，random non-balanced | 3.6105513278025183e-06 | 0.0026872099842876196 | 逐字节一致 |
| E65/T1536/topk9/shared1，random non-balanced | 3.850681558204242e-06 | 0.0027751296292990446 | 逐字节一致 |
| E65/T1536/topk9/shared0，random non-balanced | 3.594893112763491e-06 | 0.002681376878172159 | 逐字节一致 |

shared1 沿用此前分支的定义：先在64个 routed experts 中选择 topk8，再给每个 token 追加 expert ID64、权重1。已检查 shared route 数等于1536。普通 E65 random topk9 没有冒充 shared1。

### 同一编译实例的验收

三个原型 workload 的 GEMM1 使用同一个实际编译对象，cache key、cache namespace、完整 source IR、完整 ELF、HSA loaded object 和 `CallState` 实际调用的 host wrapper 地址全部一致。不是只根据相同源码文件或 symbol 判定共用。

```text
GEMM1 symbol:
a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e65_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg1x2_dsfirst16_planarlds_interleavelds_rolleddrain
ELF SHA256:
ffb11d7044486bc9c947e4c574249331de2f3986843dc08b937d291e28a98776a
source IR SHA256:
bed30f8ded47835324d27a985db7d42335551fc964260013b0a2b98a5f17f56c
本次同一进程内的 host wrapper 地址：0x731209800260
```

只读加载观测沿用原 caller 已验证的 `_ensure_engine` 时点，在 `raw_lookup` 前读取现有 `gpu.binary`；不重新 parse 或替换 IR。再把 `_get_func_exe` 返回地址绑定到实际 `CallState.__call__`。完整 ELF storage 和 loaded `.text` 均经 public HSA loader extension 核验。退出后所有 hook 恢复，生产 sources SHA256 不变。

### 失败记录与复现

v1 工具在 `_sig` 尚未初始化时读取它，v2 工具在 `raw_lookup` 后读取不到 `gpu.binary`；两次均在目标 GEMM1 launch 前停止。已保留两版 frozen plan、脚本及失败日志，没有修改或覆盖失败结果。v3 分别采用原准备工具已有的 `_ensure_sig()` 和原只读观测时点后通过。reference/prototype child 分别为24.701067453 /27.204526885秒，均低于118秒限制；这些数字是整个正确性进程时长，**不是 kernel 性能**。

复现从远端主机仓库根目录执行，controller 在指定容器运行正确性 child，Git 不进入容器：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_E64_E65_common_GEMM1_20261027_v3.py
```

这是一次性 frozen controller，已执行完毕，不要原地重跑或覆盖其结果。两端保留 `f3_E64_E65_common_GEMM1_proto_20261027_v3.frozen_plan.json`、`.summary.json`、reference/prototype JSON、完整 ELF、MLIR 和 loaded text；MoE 输出 binary 保留在远端供逐字节复核，不下载 `.out`。

本轮结束后全机 GPU users 检查没有进程，本轮没有残留。接下来对新 ELF 做资源/布局检查，并另冻结 pin0、原 timer 的 E64/shared0 与 E65/shared1 性能对照。当前只确认正确性原型和同一编译实例，尚未确认性能、不代表生产采用。此前正式 v60/v80 三轮结果仍为83.4505 /81.2470 μs，v80 未采用。

## 2026-10-27：再次重启后的 baseline 复测与共用 GEMM1 资源复核

问：汇报下进展，然后继续。

答：a07-3 本轮再次重启。当前 boot ID 为 `ff39c3dc-ffb5-4510-a458-0110ee739267`，启动时间为 2026-10-27 14:58:51 UTC；上一轮 boot ID 为 `a94beab0-6a60-46c6-9347-c90fb32d31fd`。分支仍为 `hyg_dsr1_conc6144`，HEAD 仍为 `e99dbac65a92ed23f87df4d99bd38f655a923a3b`，`dsr1_ep4` 仍运行镜像 `rocm/fw-bringup:gfx1250-atom-20260930-na`。生产 kernel 和 dispatcher SHA256 未变，生产仍为 v60。

上一轮已冻结的 v60/v80 GEMM1 三轮中值 83.4505 /81.2470 μs 属于旧 boot；原结果及其未通过“三轮全部 <80 μs”采用条件的结论保留，不替换成当前 boot 的成绩。GEMM1 <70 μs 仍为 handoff 优化目标，没有声称达到。

### 本轮固定性能计划

新计划使用同一原 caller 和原 timer，固定 E64/topk8/shared0、E65/topk9/shared1 两个 workload；每种 workload 按 `reference → prototype → prototype → reference → reference → prototype` 的顺序运行，合计12个独立 child。reference 是本分支保留路径，prototype 是此前通过三种完整 MoE 正确性和同一实际 compiled GEMM1 验证的进程内共用原型。不能把其中的 E65 reference 称为原早期分支的153.405 μs baseline。

每个 child 使用 pin=0、seed0、uniform/auto、warmup5、iters20；先沿用原 kernel-bench 的 `testGraph=False`，再沿用原 MoE e2e 的 `testGraph=True`。首次 eager G1/G2 的地址/代码身份观测全部在原 timer 开始前恢复，原 kernel-bench callable 不替换，原 `logits_diff <0.01` gate 不改动。每次计时前执行11次、间隔6秒、跨度至少60秒的全机空闲检查，结束后检查 ownership 和 settled idle。外部占用污染轮保留并排除，不重跑取得替代成绩，也不依照加载地址、seed 或读数挑选实例。全部有效才计算各自三轮中值。

当前已完成新 boot 的第一轮 v60 reference：GEMM1 **83.45615789473686 μs**、MoE e2e **317.04515789473686 μs**。此前共用原型的两轮 GEMM1 已得到 **83.35842105263163 /83.14900000000002 μs**，MoE e2e **315.8803684210527 /312.99821052631546 μs**。这三轮精度和结束后空闲检查均通过；此处仅记录执行进展，**不是完整三轮性能比较或最终采用结论**。后续轮次仍在执行或等待空闲。

等待曾检测到 `dev_al` 的外部 GPU/KFD 任务，包括 `test_gemm_a8w8_blockscale.py` 和 `verify_n_tail_renamed.py`。没有杀死或驱逐外部进程，自动每6秒继续检查。等待 parent 可以超过两分钟；每个实际 test/compiler/profiler child 仍使用 `timeout --signal=TERM --kill-after=2s 118s`，计时期间不并发查询 GPU、编译或传输文件。

### 完整 ELF 资源与布局复核

读取完整 ELF 的 AMDGPU notes 和 section headers，再在本地使用既有 ELF/metadata parser 独立复核，结果如下：

| 项目 | E64 v60 reference | 共用 expert-bound=65 原型 |
|---|---:|---:|
| `.text` bytes | 44,416 | 44,544 |
| 完整 ELF bytes | 48,872 | 49,000 |
| SGPR | 56 | 58 |
| VGPR | 798 | 798 |
| LDS bytes | 327,680 | 327,680 |
| private segment bytes | 0 | 0 |
| VGPR /SGPR spill count | 0 /0 | 0 /0 |
| workgroup size | 128 | 128 |
| wavefront size | 32 | 32 |
| cluster dims | `[4,1,1]` | `[4,1,1]` |

共用原型 `.text` 增加128 B，SGPR增加2；其余上表资源相同。runtime argument ABI 在排除 symbol/name 后逐项一致。不能称为“所有资源完全不变”，也不能只凭 `.text <64 KiB` 保证实际 GPU 放置后不跨64 KiB地址段或保证性能不回退。这里只完成静态资源和布局复核，性能仍以本轮固定原 timer 对照为准。

### 更正上一条文档中的 SHA256 转写错误

前一条正确性进展正文中的 `ffb11d7044486bc9c947e4c574249331de2f3986843dc08b937d291e28a98776a` 多写了一个字符，并非有效的64字符 SHA256。依据实际完整 ELF、原 prototype JSON 和本地独立哈希复核，正确值为：

```text
common GEMM1 full ELF SHA256:
ffb11d7044486bc9c947e4c574249331de2f3986843dc08b937d291e28a9876a
source IR SHA256:
bed30f8ded47835324d27a985db7d42335551fc964260013b0a2b98a5f17f56c
```

此前 frozen plan、原 prototype JSON 和新性能计划从实际二进制读取的 SHA256 均正确，未使用这个正文转写值进行实验绑定；代码和证据文件没有因此变化。两端通过追加更正保留原正文与完整历史，不覆盖整份文档。

### 复现与保留状态

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
plan=TASK/logs/f3_common_GEMM1_new_boot_perf_20261027_v1.frozen_plan.json
plan SHA256=0b9c244af60247b33b20e52098566def70e9ac4bc4311fe03a88caa126bedcd8
resources review=TASK/logs/f3_common_GEMM1_resources_independent_local_review_20261027_v1.json
resources review SHA256=67227c50b1b3e02f202658d4bfb86a40e117a469222fd73dbb133eb359a68e45
```

本轮从 a07-3 主机仓库根目录执行：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_common_GEMM1_perf_20261027_v1.py
```

该 frozen controller 已启动，不要原地重复执行或覆盖结果。Git 仍只在远端主机执行；当前未提交、推送或 pull，避免改变本轮绑定的 HEAD。已授权的英文 commit/push、远端主机 pull、pull 后最终复测继续保留为待办。当前只在 a07-3 /dsr1_ep4 测试；一般新的 GEMM1 调度优化继续暂停，先完成共用要求与地址修复的验收。此原型尚未进入生产。

## 2026-10-28：共用 GEMM1 计时 partial 参数漏改的排除与修复

问：汇报下进展，然后继续。a07-3 is free now。

答：继续复用 a07-3 /dsr1_ep4 的现有固定12轮任务，不重复启动。生产仍为 v60；本次修正发生在进程内共用 GEMM1 原型及其验收工具，尚未修改生产 kernel、dispatcher，也没有 commit/push/pull。所有性能继续使用 pin=0，逐轮全机空闲检查；一般新的调度优化仍暂停。

### 先前两个原型 GEMM1 读数排除

复核发现旧性能计划 `f3_common_GEMM1_new_boot_perf_20261027_v1` 的原型 eager GEMM1 已传入 `_gemm1_experts=65`，但 kernel-bench 所保存的原 `functools.partial` 仍传入 `n_experts=E=64`。因此实际计时的 GEMM1 不是要验收的共用 expert-bound=65 实例。这是实验原型的参数漏改，不能把它解释成计算 kernel 错误或精度标准变化。

| 原 ordinal | variant | 原 GEMM1 μs | 原 MoE e2e μs | 当前判定 |
|---:|---|---:|---:|---|
| 1 | reference | 83.456158 | 317.045158 | baseline 单轮保留 |
| 2 | prototype | 83.358421 | 315.880368 | GEMM1 排除：timed partial expert-bound=64，eager=65 |
| 3 | prototype | 83.149000 | 312.998211 | GEMM1 排除：timed partial expert-bound=64，eager=65 |

上述原日志、JSON、ownership 的原 `performance_valid=true` flag 和完整归档均保留；独立排除记录说明该原 flag 只判断测试结束后的 ownership，不能证明实际 timed artifact 正确。旧计划在第4轮等待空闲时停止，没有启动第4轮 child，没有杀死外部 GPU 进程。旧计划仅有一个有效 reference 单轮，不能计算完整三轮中值、声称原型改善或作为采用依据。

此前 E64/shared0、E65/shared1、E65/shared0 的完整 MoE eager 输出逐字节正确性证据不受此计时参数错误影响；旧 MoE e2e 的原读数保留，但不拼入新的固定比较。

### 修正与计时前的实际参数绑定

修正 snapshot 只将 fused GEMM1 partial 的 `n_experts=E` 改为 `n_experts=_gemm1_experts`；eager 路径、v60 kernel 计算和精度 gate 不变。新验收在首次 eager fused_moe 返回后、原 timer 开始前，临时捕获原 masked launcher 转换后的实际参数，不 dispatch GEMM；使用原 JIT 的完整 key 规则核对 GEMM1/GEMM2 的 partial 与 eager 是否绑定同一 key、namespace、source IR 和 artifact。随后恢复所有观测及绑定 hook，由保留的原 partial 对象执行原 warmup5/iters20 和原 timer。

工具 v2 的第一次预验证因从 grouped 模块读取局部 import 的 masked launcher 而失败，耗时7.625秒，尚未启动性能 timer；失败证据保留，没有作为性能数据。工具 v3 从实际 `aiter.ops.flydsl.grouped_gemm_mxfp4` 模块读取 masked launcher，预验证通过。成功 child 用时 **8.025635秒**，这是整个预验证进程时长，**不是 GEMM1 性能**。E64 的 GEMM1 eager/partial expert-bound 均65，GEMM2 均64；两阶段完整 key、artifact 和 IR 相同，两次完整 MoE 正确性调用均通过，所有 hook 已恢复。

新性能计划仍为两个 workload 各 `reference → prototype → prototype → reference → reference → prototype`，合计12轮。此次新增的实际 partial 绑定是每轮有效性条件；不挑 round、seed 或最快加载实例。全部完成后，从原日志及完整 ELF 独立重算各三轮中值。工具版本号 v3 与早期优化候选 kernel v3 无关。

```text
corrected snapshot=TASK/snapshots/f3_E64_E65_common_GEMM1_corrected_partial_20261027_v4_grouped.py
corrected snapshot SHA256=6b9a829a67fa6d534660d2f63a961989de23a9cd63af2d1912c377eaaad77f59
exclusion review=TASK/logs/f3_common_GEMM1_partial_parameter_error_independent_local_review_20261027_v1.json
exclusion review SHA256=ca070fa680172be958d49f7ce3f2eb3f9b1d2c5b5103bbb662be41f6a0370e59
old incomplete archive SHA256=dd09d6956088e07f1d8187c504f64a890d32814e71af6b2f6487b9040c83f568
new plan=TASK/logs/f3_common_GEMM1_new_boot_perf_20261027_v3.frozen_plan.json
new plan SHA256=ca410e56faac1f927f2bac19f1d523447cfefad4f64d12c13bf846d6e64c4e04
binding preflight=TASK/logs/f3_common_GEMM1_new_boot_perf_20261027_v3_r02_E64_topk8_shared0_prototype_binding_preflight.json
TASK=.codex_tmp/hyg_f3_followup_20261024
```

新 controller 已运行，不能原地重复执行或覆盖结果：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_common_GEMM1_perf_20261027_v3.py
```

旧正式 v60/v80 的三轮中值83.4505 /81.2470 μs保留为上一 boot 的独立结果；v80 未采用。后续仍需完成共用实例的正常性能与地址范围验收。已授权的英文 commit/push、远端主机 pull 和 pull 后最终复测继续保留为待办。

## 2026-10-28：取消后续 E65/shared1，E64 六轮独立复核完成

问：E65/shared1 以后就不要再测了。

答：从此指示起，**后续不再运行 E65/shared1 的性能或正确性测试**，也不为旧待办补测这个 workload。此前 fixed shared1 的历史正确性证据保留，不删除或改写。旧提交/push/pull 后的复测范围也随此指示取消 E65/shared1；以后以 E64/T1536/topk8/shared0、random non-balanced、pin=0 为当前性能主线，其他新范围遵循用户后续明确指示。

收到指示时，第7轮日志刚从等待空闲转为 child 已启动。在 tmux 对本轮 controller 发出 Ctrl-C 后，该已经启动的 child 自行完成并退出；没有启动第8–12轮，没有杀死外部 GPU 进程，也不会补齐 E65/shared1 计划。第7轮没有 controller 的 post/settled 记录，**不作为有效性能结果**，原日志、JSON、ELF、loaded text 均保留。停止后确认本轮 checker 无残留，全机 KFD owners/GPU users 为空，busy=0。

原12轮 frozen plan 和 summary 保持原样，`completed=false`、整个计划的 `all_performance_valid=false` 不改写。另建停止证据和独立 reviewer，只验收在新指示前已经完整结束的 E64 前6轮；因此可以计算 E64 三轮中值，不能称原12轮计划完成，也不计算 E65/shared1 比较。

### E64/T1536/topk8/shared0 的全部有效结果

固定顺序为 `reference → prototype → prototype → reference → reference → prototype`。reference 为当前分支 v60 保留路径，prototype 为使用 v60 计算源码、expert-bound=65 的进程内共用原型。使用原 caller、原 timer、seed0、uniform/auto、warmup5、iters20，GEMM1 kernel-bench 为原 `testGraph=False`，MoE e2e 为原 `testGraph=True`。

| ordinal | variant | GEMM1 μs | MoE e2e μs |
|---:|---|---:|---:|
| 1 | reference | 83.118789 | 307.778737 |
| 2 | prototype | 82.899737 | 311.898316 |
| 3 | prototype | 83.283684 | 315.209526 |
| 4 | reference | 83.019842 | 311.501211 |
| 5 | reference | 83.432684 | 312.019526 |
| 6 | prototype | 82.945316 | 315.466158 |

| variant | GEMM1 三轮中值 μs | MoE e2e 三轮中值 μs |
|---|---:|---:|
| v60 reference | 83.118789 | 311.501211 |
| 共用 expert-bound=65 原型 | 82.945316 | 315.209526 |

GEMM1 中值降低 **0.208706%**（减少0.173474 μs），MoE e2e 中值增加 **1.190466%**（增加3.708316 μs）。这次仅证明原型的有效性能读数和精度，没有建立 e2e 性能收益；暂不将共用原型写入 production。不能把0.21%的 GEMM1 中值变化单独当作稳定优化收益。

六轮 kernel-bench 和 graph MoE 的原 `logits_diff <0.01` gate 全部通过，`logits_diff=3.6105513278025183e-06`、`rel_l2=0.0026872099842876196`。每轮两个实际 timed partial 经原 masked 参数转换后，与 eager 的完整 key、namespace、source IR、artifact 一致；全部计时前 hook 恢复，原 partial 对象、timer 和有效计算保留。各实际 child 为8.276–8.327秒，低于118秒限制；每轮11次/至少60秒全机 idle、post ownership、settled idle 均独立通过。

### 实际加载范围及其限制

| ordinal | variant | HSA load_base | group | base low16 | 实际 `.text` 跨64 KiB段 |
|---:|---|---|---:|---|---|
| 1 | reference | `0x704b89368000` | 2 | `0x8000` | 是 |
| 2 | prototype | `0x75e333490000` | 1 | `0x0` | 否 |
| 3 | prototype | `0x74ceec648000` | 0 | `0x8000` | 是 |
| 4 | reference | `0x71df23e20000` | 2 | `0x0` | 否 |
| 5 | reference | `0x77b3692e0000` | 2 | `0x0` | 否 |
| 6 | prototype | `0x79a6c4560000` | 2 | `0x0` | 否 |

这里是原 caller 的自然加载记录。reference 只出现 group2；prototype 出现group0–2，没有group3。出现 low16=`0x8000` 的两轮实际 `.text` 跨64 KiB段，其余四轮不跨段。**这不是 group0–3 地址修复验收**，也不能由小于64 KiB的 `.text` 推广成每次加载都不跨段。没有追加加载去挑快实例，也不将旧 ELF 的地址验收套到共用 expert-bound=65 的新 ELF。

### 证据和后续

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
archive=TASK/logs/f3_common_GEMM1_new_boot_perf_20261027_v3_evidence.tar.gz
archive members=108
archive bytes=3319602
archive SHA256=a7771eed5f4bd4006af65e012ec7164e811787530393ad00db87830a84a8153c
independent review=TASK/logs/f3_common_GEMM1_E64_scope_stop_independent_local_review_20261028_v1.json
independent review SHA256=7d439e48d5dc1a2ce801ac898a39db69fbda50db9ab0d3022c328f2a4138d990
scope stop=TASK/logs/f3_common_GEMM1_new_boot_perf_20261027_v3.scope_change_stop_20261028_v1.json
```

归档逐member SHA256 和完整 loaded `.text` byte identity 已在本地独立复核，不含`.out`。本地 reviewer 首次适配时仍匹配旧 `bench=True` 字面表达式，未匹配新版 `bench=not args.binding_preflight_only`，CPU 校验因此失败；按实际源码表达式修正，并额外检查每个被验收 child 的 `binding_preflight_only=false` 后通过。没有改变原计时代码、原日志或性能有效性条件。

生产 kernel 和 dispatcher 仍未变，production 继续为 v60，当前未 commit/push/pull。上一 boot 的正式 v60/v80 三轮中值83.4505 /81.2470 μs及v80未采用的结论保留。继续 E64 的加载地址修复/原路径验收；一般新的调度优化仍暂停。已授权英文提交、推送、远端主机pull和pull后复测继续保留，但不再包含 E65/shared1。

## 2026-10-28：E64 的 v80 原路径三轮与后置 v60 基线独立复核

问：汇报下进展，然后继续。

答：生产继续保留 v60，未修改生产 kernel 或 dispatcher，未 commit/push/pull。用户已明确要求以后不要再测试 E65/shared1，本轮及后续待办均遵守；历史记录保留。本轮只在 a07-3 /dsr1_ep4 测 E64/T1536/topk8/shared0、random non-balanced、pin=0。一般新的调度优化继续暂停，先完成加载地址修复和完整 MoE 路径诊断。

### 同一 boot 的前置 v60、v80、后置 v60

沿用原 caller 的输入和 timer：seed0、uniform/auto、warmup5、iters20；GEMM1 kernel-bench 为原 `testGraph=False`，MoE e2e 为原 `testGraph=True`。每轮计时前核对原 masked 参数转换后的 partial 完整 cache key、eager artifact、source IR、实际 loaded ELF，随后恢复全部绑定和观测 hook；没有替换原 partial、减少 timer 次数或调整精度 gate。

| 组别 | GEMM1 三轮 μs | GEMM1 中值 μs | MoE e2e 三轮 μs | MoE e2e 中值 μs |
|---|---|---:|---|---:|
| 前置 v60 reference | 见前一节原固定 ordinal1/4/5 | 83.118789 | 见前一节原固定 ordinal1/4/5 | 311.501211 |
| v80 | 81.510211 / 81.475474 / 81.540737 | 81.510211 | 315.331632 / 318.114895 / 318.185684 | 318.114895 |
| 后置 v60 reference | 83.743211 / 83.656947 / 82.609053 | 83.656947 | 312.850684 / 314.225737 / 309.936474 | 312.850684 |

后置三轮事前冻结，仅执行原定三轮；它们已结束，不重跑。三轮 `completed=true`、`all_performance_valid=true`，整个 child 分别用时8.326170 /8.276252 /8.276252秒，均低于两分钟。v80 三轮也全部有效。后置三轮原 kernel/e2e 的 `logits_diff` 均为 `3.6105513278025183e-06`，原 gate 仍为0.01；两阶段 timed partial 与 eager 代码身份一致。

| v80 相对组别 | GEMM1 耗时变化 | MoE e2e 耗时变化 |
|---|---:|---:|
| 前置 v60 | 降低1.935277%（减少1.608579 μs） | 增加2.123165%（增加6.613684 μs） |
| 后置 v60 | 降低2.566119%（减少2.146737 μs） | 增加1.682659%（增加5.264211 μs） |

v60 自身前后中值上移：GEMM1 0.647456%，e2e 0.433216%。后置 v60 的 e2e 仍低于 v80，因此不能把 v80 的全部 e2e 差异解释成整个时间段的统一漂移；这也不能唯一归因于某个硬件环节。需要检查完整 MoE graph 中 GEMM1/GEMM2 的实际执行表现，不能用孤立 GEMM1 的收益证明 e2e 收益。

生产暂不采用 v80：本轮三轮 GEMM1 都没有低于80 μs，且原 e2e 没有建立收益。上一 boot 的正式 v60/v80 中值83.4505 /81.2470 μs及其冻结采用结论保持原样；新数据属于当前 boot 的独立验收，不替换旧表。当前 v80 中值距离 handoff 的 `<70 μs` 目标仍差11.510211 μs。

### 实际地址范围及相同代码的绑定

当前 boot 为 `ff39c3dc-ffb5-4510-a458-0110ee739267`，两端分支/HEAD仍为 `hyg_dsr1_conc6144` / `e99dbac65a92ed23f87df4d99bd38f655a923a3b`。v80 source SHA256为 `93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`，实际 GEMM1 ELF为 `49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8`；三轮自然加载的 base group 为2/0/2，base low16均为0，`.text`为97,408 B，三轮都跨64 KiB地址段。

后置 v60 source SHA256为 `8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`，GEMM1 ELF为 `59e4bd1aea39005b5bb1a0aa7e36871c75816f4233a374705a71fa8c78cb8135`；三轮自然加载的 base group 为2/1/0，base low16为0/0/0x8000，`.text`为44,416 B；第三轮实际跨64 KiB段，前两轮不跨。

两组 GEMM2 的完整 ELF均为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`，没有改 GEMM2 的有效计算或编译 specialization。这一绑定不证明 graph 中的 GEMM2 耗时必定相同。

以上只是原 caller 自然地址的覆盖，没有人为挑实例，未覆盖本轮所有 group0–3。此前完整同 ELF 四组正常计时/counters 对照保留，不重复或用本轮填补它。地址关系继续以本地 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1的 `Set[4:0] = address[13:9] ^ address[18:14]` 为依据；`.text <64 KiB`不保证实际加载后不跨段，跨段几何也不能独立证明 miss、live residency 或唯一瓶颈。无效的 `SQG_CYCLES/SQG_REFCLKS`固定读数继续排除。

### 证据、复现入口和采用决定

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
v80 archive=TASK/logs/f3_E64_v80_current_boot_original_timer_20261028_v1_evidence.tar.gz
61 members，2797522 bytes
SHA256=88981b3df2d1ff8de2a6b54e88686b101b1e3f35ad3860ce8354baa677e8e90b
v80 independent review=TASK/logs/f3_E64_v80_current_boot_original_timer_independent_local_review_20261028_v1.json
review SHA256=175052fbe273325170bcb5843c5a7c34341e9ef25e99956a2f151fd86d2f6535
post-v60 archive=TASK/logs/f3_E64_v60_after_v80_original_timer_20261028_v1_evidence.tar.gz
58 members，1457511 bytes
SHA256=9d3eadd6626ac1c13ded97d680d45226e5e14f869076d0cc79178e66ea54eae7
post-v60 independent review=TASK/logs/f3_E64_v60_after_v80_original_timer_independent_local_review_20261028_v1.json
review SHA256=4926d32c2ddb17a99e0a5da8aee2810a78152a54fcd57b88a46ddbbfc6f6904e
post-v60 plan SHA256=af44ec0ddc46a23a8298d0fa690929d0903a0bb32b253577879dadfcdb0dde3f
```

两份归档均已下载到本地并逐 member 核验，未下载 `.out` 或完整 MoE 输出 binary。后置 reviewer 还核对前置和候选的独立 review、相同 workload/route fingerprint、原 timer和全部三轮；不存在按读数或加载地址筛选。每轮前11次全机连续idle，间隔6秒、跨度至少60秒；child为 `timeout --signal=TERM --kill-after=2s 118s`，计时期间不并发GPU查询、编译或SFTP。最后 post ownership无owner、settled idle通过，未发现本轮测试残留。

本地复核入口（本轮已执行成功，输出采用独立新文件，不能原地重跑覆盖）：

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_E64_v80_current_boot_original_timer_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_E64_v60_after_v80_original_timer_local_20261028_v1.py
```

继续 E64 的地址修复和完整 MoE 性能诊断，暂不尝试一般新的调度优化。已授权英文commit/push、远端主机pull和pull后最终复测仍为待办；最终复测不再包含E65/shared1。三份文档在两端分别CAS追加，保留各自历史。

## 2026-10-28：完整 MoE graph 的 GEMM1 回退与 counters 独立复核

用户：汇报下进展，然后继续。

答：当前生产仍保留 v60。v80 在本次 boot 下孤立 GEMM1 中值为 81.510211 μs，前置/后置 v60 为 83.118789 / 83.656947 μs；但 v80 MoE e2e 为 318.114895 μs，相对前置/后置 v60 的 311.501211 / 312.850684 μs 回退 2.123165% / 1.682659%，因此没有采用。下面通过已有正常日志及另行采集的 counters 确认，回退主要出现在完整 MoE 路径中的 GEMM1。下一步仍只研究 code footprint/layout，先尝试收敛 activation/quant epilogue 的展开；尚未产生 v81 源码、ELF 或性能成绩。

本轮范围是 E64/T1536/topk8/shared0、random non-balanced、pin=0、A preshuffle=1。按照用户最新要求，不再测试 E65/shared1，包括正确性、性能和之后 Git pull 后的最终复测。已有 E65/shared1 历史证据原样保留；E65/shared0 的功能支持要求仍保留。

### 已有九轮原 graph profile 的复核

从前置 v60、v80、后置 v60 三个已封存归档中提取原 graph profiler 表，每组各三轮；没有重新运行正常计时或增加 profiler window。表中 GEMM1/GEMM2 只打印一位小数，保留该精度，不声称有原 event 的更高精度。

| 组别 | 孤立 GEMM1 三轮中值 μs | MoE e2e 三轮中值 μs | graph 内 GEMM1 中值 μs | graph 内 GEMM2 中值 μs |
|---|---:|---:|---:|---:|
| 前置 v60 | 83.118789 | 311.501211 | 113.5 | 83.1 |
| v80 | 81.510211 | 318.114895 | 119.0 | 82.9 |
| 后置 v60 | 83.656947 | 312.850684 | 112.5 | 83.3 |

独立复核：`.codex_tmp/hyg_f3_followup_20261024/review_f3_E64_existing_graph_profiles_20261028_v1.py`，结果为 `logs/f3_E64_existing_graph_profiles_independent_local_review_20261028_v1.json`，`passed=true`。原 timer 排除第一次后统计 19 次的口径保持不变。孤立 kernel 耗时与 graph 内耗时属于不同执行上下文，不能相互替换。

### 两轮原路径 counters：v60 → v80

事前冻结计划：`.codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_original_graph_counters_20261028_v1.frozen_plan.json`，SHA256=`65e210b60620b483ac79d70f5ee77867f620f5641942c728f5c626dd71bda128`。两轮实际 child 用时 12.332355 / 12.232065 秒，已全部结束，不需要重跑。

原 caller、输入、warmup=5/iters=20、精度 gate=0.01 及 timer 返回值保留。仅在原 profiler context 外围加 CPU ROCTx markers，context 关闭后复制 events。外层 rocprofv3 可能改变调度/序列化，因此其所有 μs 读数均不作为新性能成绩。正常性能仍使用上面的原九轮。

每轮五个原 profile window，其中孤立 GEMM1、完整 eager MoE、完整 graph replay 各含 20 个目标 GEMM1 dispatch。独立 reviewer 将原始 callback 的每个 counter instance 求和，与 CSV 值、dispatch IDs、时间戳、correlation IDs 对齐；又将 graph_exec_id 的 180 个 dispatch 与 graph window 中的 20 个 GEMM1 绑定。每轮 113 个目标 GEMM1 dispatch、452 条 counter CSV 行均完整保留；下面为各 window 全部 20 次的中值，结果文件另保留原 timer 对应的后 19 次统计。

| 版本 | 场景 | `SQC_ICACHE_MISSES` 中值 | `SQ_WAVES` |
|---|---|---:|---:|
| v60 | 孤立 GEMM1 | 0 | 7168 |
| v80 | 孤立 GEMM1 | 0 | 7168 |
| v60 | eager MoE 内 GEMM1 | 82,528 | 7168 |
| v80 | eager MoE 内 GEMM1 | 110,160 | 7168 |
| v60 | graph MoE 内 GEMM1 | 82,532 | 7168 |
| v80 | graph MoE 内 GEMM1 | 110,362.5 | 7168 |

graph miss 中值增加 **33.720860%**。这支持继续研究 instruction-cache footprint；不能唯一归因全部正常耗时差异，也不能据此确认具体 cache bank/set、live residency、冷 cache 或 clock 是唯一原因。`SQC_ICACHE_MISSES` 的文档定义包含 uncached requests，且属于 nondeterministic counter；没有假设 `REQ=HITS+MISSES`。两轮仅使用自然加载地址，不能替代之前对同一 ELF 的 group0–3 验收：

| 版本 | 自然 `load_base` | group | `.text` B | GEMM1 ELF SHA256 |
|---|---|---:|---:|---|
| v60 | `0x7b04c6d30000` | 3 | 44416 | `59e4bd1aea39005b5bb1a0aa7e36871c75816f4233a374705a71fa8c78cb8135` |
| v80 | `0x7f6a3b960000` | 2 | 97408 | `49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8` |

SDK 的 kernel_id、code_object_id、kernel_object、load_base/load_delta/load_size 与公开 HSA observer、原 partial key、完整 ELF、loaded text 逐一核对。两版 GEMM2 ELF 相同，SHA256=`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。没有新增/使用此前已判定无效的 SQG clock counters。

### 复核工具的寄存器语义修正

第一次 CPU reviewer 失败于 `AssertionError: ('.sgpr_count', 56, 128)`。原 reviewer 错误地要求 ELF metadata 的编译器使用数与 SDK 读数直接相等，原失败 stderr、修改前源码和 SHA256 已保留。修正了 reviewer，未改原 ELF、callback、CSV 或性能结果，也没有重跑 GPU 采集。

硬件依据：`MI400_Shader_Programming#65.txt` §3.3.1.1（正文 p30）规定每 wave 固定 128 个 SGPR RAM 位置；§3.3.2.1（正文 p32）规定 wave32 VGPR 以 16 个为单位分配。独立解析 ELF `.symtab` 的 `.kd`、64 B descriptor、`COMPUTE_PGM_RSRC1` 与 wave32 property，结果如下：

| 版本 | metadata SGPR 使用数 | 固定 SGPR RAM 位置 | metadata VGPR 使用数 | descriptor VGPR blocks | gfx1250 wave32 分配数 | 原 SDK `arch_vgpr_count` |
|---|---:|---:|---:|---:|---:|---:|
| v60 | 56 | 128 | 798 | 50 | 800 | 400 |
| v80 | 60 | 128 | 799 | 50 | 800 | 400 |

原 SDK 的 VGPR 400 与通用 parser 的 `(GRANULATED_WORKITEM_VGPR_COUNT + 1) * 8` 相符，不能当作 gfx1250 的实际分配数。所检查 SDK 源码 excerpt 与源码 hash 已封存，但尚未证明它和当前安装 SDK binary 的源码 revision 完全相同，因此不扩大为所有 SDK 版本的判断。保留原 SDK 值，并分别验证编译器使用数、descriptor 分配数和 SDK 读数。两版 spill count 和 private segment 都为 0。

独立 counters reviewer：`.codex_tmp/hyg_f3_followup_20261024/review_f3_E64_original_graph_counters_local_20261028_v1.py`，结果 `logs/f3_E64_original_graph_counters_independent_local_review_20261028_v1.json`，`passed=true`。原失败记录：`logs/f3_graph_counter_reviewer_SGPR_parser_failure_20261028_v1.json`。

归档 `logs/f3_E64_original_graph_counters_20261028_v1_evidence.tar.gz`：62 个 member 的全部 hash 验证通过，6,238,248 B，SHA256=`1dc3f588a210556bd61671dcb9ae2300c9d85df20535b387ca94ffb243de5b29`；没有下载 `.out` 或完整 MoE 输出 binary。

### 复现入口与下一步

已完成的采集入口是在 a07-3 主机仓库根目录执行 `python3 .codex_tmp/hyg_f3_followup_20261024/run_f3_E64_original_graph_counters_20261028_v1.py`，它按冻结计划进入 dsr1_ep4 执行两个 child。该 label 已完成封存，不能直接重跑覆盖；CPU 复核入口是同目录中的 `review_f3_E64_original_graph_counters_local_20261028_v1.py`。

本次 boot=`ff39c3dc-ffb5-4510-a458-0110ee739267`，branch=`hyg_dsr1_conc6144`，HEAD=`e99dbac65a92ed23f87df4d99bd38f655a923a3b`。每轮 counters 前均通过全机 11 次、间隔 6 秒、跨度至少 60 秒的连续 idle；child 运行时不并发查询 GPU/编译/传输。118 秒 TERM/再 2 秒 KILL 保留，post/settled 检查无本次 KFD 残留，未终止外部任务。远端生产源码已逐字节恢复 v60，两端 SHA256 均为 `8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。

下一步仍是用户指定的 code footprint/layout 修复。v80 保留四/六/七/八覆盖 body 及一个公共 epilogue，activation/quant 仍按八个 M16 row 展开；考虑收敛该展开时必须保留有效数学、无效行跳过、原 waits/barriers、精度和支持范围。先编译检查实际 ELF、寄存器/溢出和 native 模型，之后才做完整 MoE 精度、正常计时与不同加载地址验收；源码变短本身不构成收益证据。

本轮尚未 commit/push/pull。此前授权的英文 commit、push origin、远端主机 pull 和 pull 后最终性能复测仍待收尾；最终复测排除 E65/shared1。三份文档分别在两端按 CAS 追加，保留各自历史及既有暂存修改。

## 2026-10-28：v81 runtime row epilogue 的编译、native 和完整 MoE 精度验收

针对“汇报下进展，然后继续”：生产仍保留 v60。v80 孤立 GEMM1 稍快而完整 MoE e2e 回退的结论保留；本轮没有将它设为生产版本。当前只推进 code footprint/layout 修复，没有恢复一般调度优化。所有新运行均在 a07-3/dsr1_ep4，pin=0、`AITER_GROUPED_A4W4_A_PRESHUFFLE=1`、random non-balanced。未执行任何 E65/shared1 测试。

v81 基于 v80，在既有 `interleaved_m_rows and not SUPPORT_EXPLICIT_VGPR_PARTITION` guard 内将八个 activation/quant row body 收敛为一个 runtime row loop。固定八行 bound 通过空 inline asm 的 tied identity `constraints="=s,0"` 保持对 LLVM opaque；八个 accumulator selection arm、所有有效运算、无效 M16 跳过、partial row ScaleA zero 和 owner mask 均保留。四/六/七/八 mainloop、TDM/LDS/WMMA/waits/barriers、准入和 fallback 没有随这一版 epilogue 改动。源码逐行 AST 复核和 1028 个 contiguous output contract 检查已通过。

| 项目 | v80 | v81 |
|---|---:|---:|
| ELF `.text` bytes | 97,408 | 81,920 |
| metadata VGPR used | 799 | 797 |
| metadata SGPR used | 60 | 59 |
| LDS bytes | 327,680 | 327,680 |
| private segment / VGPR spill / SGPR spill | 0 / 0 / 0 | 0 / 0 / 0 |

`.text` 减少 15,488 B（约 15.9%），仍大于 64 KiB。这一结果不能证明快慢分档已消失，也不能替代 device 性能验收。寄存器 used 数量不等于硬件 allocation：依据 `MI400_Shader_Programming#65.txt` §3.3.1.1、§3.3.2.1，应继续区分固定 SGPR RAM slots 与 wave32 VGPR blocks，不能重新采用此前 SDK generic parser 的错误 8 VGPR/block 解释。

v81 source SHA256 为 `66064581c1b7ffb183e0d536621d1452b411c34d729b74bcd3e8adaf6deecf53`；GEMM1 ELF 为 `e96248e5d7eb6287ed5c31f42dad3632db0e163eb0c11a4c76baa75afb9bd931`。既有 bounded DS/WMMA model 与 `WMMA_only` retirement stress 均通过，各约 2.5 秒，CFG iterations 均为 16,714；source IR 中 `depctr_va_vdst(8)/(0)` 各 20 处保留，无 indirect control flow。native 是保守检查模型，精度与性能仍需实际运行验证。

本轮完整 MoE 验证结果如下，全部使用原 `LOGITS_DIFF_TOL=0.01` gate，并比对整个 `(1536,7168)` bf16 输出。参考输出保留在远端，没有下载完整 MoE binary。以下没有性能数据。

| workload | seed | logits_diff | rel_l2 | 输出字节数 | 与参考 differing_bytes |
|---|---:|---:|---:|---:|---:|
| E64/topk8/shared0 | 0 | 3.6105513278025183e-06 | 0.0026872099842876196 | 22,020,096 | 0 |
| E64/topk8/shared0 | 17 | 3.6202693316254594e-06 | 0.002690816530957818 | 22,020,096 | 0 |
| E65/topk9/shared0 | 0 | 3.594893112763491e-06 | 0.002681376878172159 | 22,020,096 | 0 |
| E65/topk9/shared0 | 17 | 3.6086727178189905e-06 | 0.0026865052059292793 | 22,020,096 | 0 |

实际 GEMM2 ELF 仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。已通过 native gate 的 GEMM1 cache `.pkl` bytes/hash/mtime 保持不变。五个实际 child 均在 120 秒内完成：GEMM2 preparation 13.63 秒，四组完整 MoE 验证 8.43 / 8.08 / 20.39 / 8.33 秒；使用 `timeout --signal=TERM --kill-after=2s 118s`。结束时源码恢复 v60，KFD owners 为空，settled idle。E65/shared0 这里只验证 guard 外的原支持路径；不能因此声称 E64/E65 共用同一 compiled GEMM1 的要求已完成。

保留两个已恢复的工具错误：首次编译完成后 artifact export 缺少 `parameter_runtime` 的 import path；修正导出后，host 检查又误以为一个 cache stem 只有一个文件，实际同时有 `.lock` 和 `.pkl`。前者复用已完成的编译 cache，后者在 CPU 上核对目标 cache hash/mtime 后完成 inspection，未重新执行目标 GPU preparation 来替换证据。失败记录、原有效 cache、v60 生产源码均保留。

随后冻结六轮正常 E64/shared0 性能对照：`v60, v81, v81, v60, v60, v81`（AB、BA、AB）。先做一次 v81 的 untimed binding preflight，再由原 caller 分别运行孤立 kernel timer 和完整 MoE graph timer；观察和 binding hook 在 timer 前全部恢复，实际 timed partial key 与 eager artifact 必须相同。每轮要求全机 11 次连续 idle、间隔 6 秒、跨度至少 60 秒；保留全部正常结果和自然 load address，不挑最快实例，不替换慢轮。此 checkpoint 尚无 v81 正式性能成绩，尚未进行 v81 group0–3 或 counters 验收。

本轮已执行的 host 命令（label 已使用，未来复测须创建独立 label/plan，不能覆盖既有记录）：

```bash
cd /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v81_device_prepare_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_E64_v60_v81_original_timer_20261028_v1.py
```

证据路径均相对本仓库：

- `.codex_tmp/hyg_f3_followup_20261024/snapshots/f3_v3_runtime_row_quant_epilogue_v81.py`
- `.codex_tmp/hyg_f3_followup_20261024/logs/f3_v81_source_independent_CPU_review_20261028_v1.json`
- `.codex_tmp/hyg_f3_followup_20261024/logs/f3_v81_compile_and_export_CPU_20261028_v3_evidence.tar.gz`，SHA256 `d837236979e69ac0fd99fe907fb23909b316da4c70bc330a76f58f04f462723d`，19 个 member hash 均通过。
- `.codex_tmp/hyg_f3_followup_20261024/logs/f3_v81_native_CPU_20261028_v1.review.json`
- `.codex_tmp/hyg_f3_followup_20261024/logs/f3_v81_complete_MoE_pin0_20261028_v1.summary.json`
- `.codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_v60_v81_original_timer_20261028_v1.frozen_plan.json`，SHA256 `16efa78fc8ac83414a4ab68a9541a002cbc13607b35d6a758dc7784dc032cf69`。

当前限制：v81 尚无正常性能或跨加载地址修复成绩；`<70 μs` 的 handoff 目标尚未达成，生产仍为 v60。英文 commit/push、远端主机 pull 和 pull 后最终复测仍待收尾；此前 baseline/最优版的历史数据没有因本轮实验被覆盖。

## 2026-10-28：v81 六轮原 timer 成绩及完整 graph 耗时复核

用户：汇报下进展，然后继续。

答：生产仍保留 v60。v81 完整 MoE 的 E64/shared0、E65/shared0 seed0/17 精度和逐字节比较已通过，详见上一节；没有测试 E65/shared1。六轮固定 AB、BA、AB 正常对照已全部完成，v81 的孤立 GEMM1 中值 84.122737 μs，相对 v60 的 83.741158 μs 增加 0.455665%；完整 MoE e2e 中值从 315.172105 降至 286.818158 μs，降低 8.996338%。尚不能把 v81 作为已验收的地址分档修复，正常自然地址只覆盖 group0/1/2，group3 仍缺证据，handoff 的 `<70 μs` 目标未达成。

本轮 workload 为 E64/T1536/topk8/shared0、random non-balanced、pin=0、A preshuffle=1。使用原 caller、原 timer 和原 `LOGITS_DIFF_TOL=0.01`；所有正常轮保留，没有根据读数或加载地址筛选。

| 固定轮次 | 版本 | GEMM1 load group | 孤立 GEMM1 μs | MoE e2e μs |
|---:|---|---:|---:|---:|
| 1 | v60 | 2 | 83.741158 | 309.608053 |
| 2 | v81 | 0 | 84.122737 | 286.818158 |
| 3 | v81 | 1 | 83.547316 | 286.086579 |
| 4 | v60 | 0 | 83.947842 | 315.452211 |
| 5 | v60 | 3 | 82.973789 | 315.172105 |
| 6 | v81 | 2 | 84.905053 | 287.994632 |

### 六轮原 graph 表的独立复核

从同一已封存归档提取全部六轮原 graph profiler 表，没有重新运行 GPU 测试或新增 profile window。表格只打印一位小数，以下保留原精度；原计时按排除第一次后的19次统计。

| 固定轮次 | 版本 | graph 内 GEMM1 μs | graph 内 GEMM2 μs |
|---:|---|---:|---:|
| 1 | v60 | 113.0 | 82.0 |
| 2 | v81 | 96.4 | 75.2 |
| 3 | v81 | 96.3 | 73.9 |
| 4 | v60 | 113.6 | 86.0 |
| 5 | v60 | 113.2 | 86.8 |
| 6 | v81 | 96.2 | 75.1 |
| v60 中值 | v60 | 113.2 | 86.0 |
| v81 中值 | v81 | 96.3 | 75.1 |

孤立 GEMM1 没有建立收益，而 graph 内两阶段耗时均下降。GEMM2 的完整 ELF 仍相同，SHA256=`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`；这只证明机器码相同，不证明执行上下文或耗时相同。不能将 e2e 收益等同于孤立 GEMM1 收益，也不能据此唯一归因某个 cache、FIFO 或 clock 环节。正在采集另行冻结的 v81 原路径 counters，两轮只测 v81，使用既有 v60/v80 counters 作为保留对照，不重跑旧采集；外层 profiler 的所有 μs 不作为正常性能成绩。

硬件解释沿用本地 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1 的 `Set[4:0] = address[13:9] ^ address[18:14]`，以及 `architecture/system.txt` 对 `SQC_ICACHE_MISSES` 包含 uncached requests 的定义和 nondeterministic 限制。代码缩小或当前三种地址表现接近，均不能替代同一 ELF group0–3 的正常 timer 验收。

### 证据和下载恢复

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
plan=TASK/logs/f3_E64_v60_v81_original_timer_20261028_v1.frozen_plan.json
plan SHA256=16efa78fc8ac83414a4ab68a9541a002cbc13607b35d6a758dc7784dc032cf69
archive=TASK/logs/f3_E64_v60_v81_original_timer_20261028_v1_evidence.tar.gz
archive bytes=5493211，118 members
archive SHA256=25bc0b25ab63fa257c473c28264aec36ab1ba774de7465b6e28299bf702b283f
normal reviewer=TASK/review_f3_E64_v60_v81_original_timer_local_20261028_v1.py
normal review=TASK/logs/f3_E64_v60_v81_original_timer_independent_local_review_20261028_v1.json
graph reviewer=TASK/review_f3_E64_v60_v81_graph_profiles_local_20261028_v1.py
graph review=TASK/logs/f3_E64_v60_v81_graph_profiles_independent_local_review_20261028_v1.json
recovery receipt=TASK/logs/f3_v81_normal_archive_early_download_recovery_local_20261028_v1.json
```

曾在远端 tar 尚未结束时提前下载，得到5,178,299 B的 partial，SHA256=`68f5d6472f9b1c81a4ab92cc23ef1fdb0f937f43de4069c6275744c9affc769b`。该文件及原失败记录均保留，未用作性能证据。等远端 archive 和 manifest 完成后重新下载，完整5,493,211 B和全部118 member的hash均通过。第一次 importer 还因终端抄入65字符 expected digest 而失败，修正为上面的64字符完整archive SHA256后成功；旧 failure receipt 中的错误 expected digest 保留，由 recovery receipt 明确纠正。恢复只重做下载/CPU复核，没有重跑性能或替换任何计时轮。

每个实际 child 使用 `timeout --signal=TERM --kill-after=2s 118s`。每轮前全机11次连续 idle、间隔6秒、跨度至少60秒；运行时不并发GPU查询、远端编译或SFTP。两端 production source仍为 v60，SHA256=`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。继续只研究 code footprint/layout；E64/E65 共用同一 compiled GEMM1 的要求仍未完成。英文commit/push、远端主机pull和pull后最终复测仍待收尾，最终复测排除E65/shared1。

## 2026-10-28：v81 原路径 counters 两轮复核与有限地址覆盖计划

用户：汇报下进展，然后继续。

答：v81 两轮原路径 counters 均已结束并通过独立复核，生产源码已恢复 v60。每轮113个目标 GEMM1 dispatch、452条 counter CSV行、五个原 profile window均保留；raw callback各instance求和后与CSV、SDK kernel/code object、公开HSA loaded ELF及原180-dispatch graph逐一对齐。外层 profiler 的所有 μs均排除，不作为新的正常性能成绩。

| 版本 / 轮次 | load group | 孤立 GEMM1 `SQC_ICACHE_MISSES` 中值 | eager MoE GEMM1 miss 中值 | graph MoE GEMM1 miss 中值 | `SQ_WAVES` |
|---|---:|---:|---:|---:|---:|
| 保留 v60 | 3 | 0 | 82,528 | 82,532 | 7168 |
| 保留 v80 | 2 | 0 | 110,160 | 110,362.5 | 7168 |
| v81 第1轮 | 2 | 0 | 79,305 | 79,364 | 7168 |
| v81 第2轮 | 1 | 0 | 78,748.5 | 78,380 | 7168 |

v81 两轮 graph window 中值的中值为78,872，相对保留v60的82,532降低4.434644%。v60/v80的旧采集没有重跑。v81 的正常 graph GEMM1中值96.3 μs及MoE e2e中值286.818158 μs来自上一节六轮原timer，不能用本节外层profiler读数替代。miss下降支持继续研究code footprint/layout，但没有证明所有地址组都相同，也不能唯一解释正常e2e收益。miss包含uncached requests且为nondeterministic counter；没有假设`REQ=HITS+MISSES`，没有使用无效SQG clock counters。

两轮自然`load_base`为`0x7d5794020000`、`0x75b7caed0000`，均64 KiB对齐；GEMM1 ELF仍为`e96248e5d7eb6287ed5c31f42dad3632db0e163eb0c11a4c76baa75afb9bd931`，`.text=81920 B`。resource检查分别保留compiler used、descriptor allocation和SDK原值：metadata VGPR used797，wave32 descriptor50 blocks /800 VGPR；metadata SGPR used59，固定128 SGPR RAM slots；SDK原始arch_vgpr_count400不作为gfx1250实际allocation。private segment/spills为0。

每轮前11次全机连续idle、间隔6秒、跨度至少60秒。两轮实际child用时12.382237 /12.332212秒，均受118秒TERM/再2秒KILL约束；计时/采集期间没有并发GPU查询、编译或SFTP。post ownership为空，settled idle通过，没有本轮进程残留。等待期间存在其他E96/T16384/topk6/inter_dim3072任务，按6秒间隔自动等空闲，没有终止外部进程。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
counter plan=TASK/logs/f3_E64_v81_original_graph_counters_20261028_v1.frozen_plan.json
plan SHA256=b0d24c17516c464b6baa54885e328b5ba0fc3581976d33f47d3dfac2ffa07284
archive=TASK/logs/f3_E64_v81_original_graph_counters_20261028_v1_evidence.tar.gz
archive bytes=6603598，63 members
archive SHA256=41c81dc26e81a859c2263b779e91a5fda5226be4bc93a434259e06b9f99b19b5
reviewer=TASK/review_f3_E64_v81_original_graph_counters_local_20261028_v1.py
review=TASK/logs/f3_E64_v81_original_graph_counters_independent_local_review_20261028_v1.json
```

完整archive和manifest均完成后才下载，63个member的hash全部通过；没有下载`.out`或完整MoE输出binary。

下一轮有限计划已经在实际运行前冻结：只启动一个E64/shared0 seed0的正常地址对照process，使用固定v81/v3 ELF、同一原输入、同一host wrapper和CallState、原warmup5/iters20。最多加载48个target；加载完全按group0–3的地址覆盖决定，先冻结全部加载，再将每个实例正向、反向各计时一次。所有实例和20个原GPU events保留，用原后19个event重算timer；不挑最快实例，不为慢读数或失败轮追加替换测试。filler只加载，不dispatch。

这轮只用于地址修复验收，不替换此前六轮v60/v81正式性能，也不修改有效计算、精度gate或准入；复用当前已纠正physical expert bound的原caller。运行时临时换入已验证v81源码，结束恢复v60。两份target的GPU dispatch ABI相同，child去掉精确标签/hash替换后与既有bounded paired工具整份AST相同。当前尚无这轮正常性能结果。

```text
address plan=TASK/logs/f3_v81_v3_all_addresses_normal_20261028_v1.frozen_plan.json
plan SHA256=1ae685e46954dc07f255d4e18d118438fd4d238baf1ee3e7067297b93dc3ac84
host runner=TASK/run_f3_v81_v3_all_addresses_normal_20261028_v1.py
v81 ELF SHA256=e96248e5d7eb6287ed5c31f42dad3632db0e163eb0c11a4c76baa75afb9bd931
v3 ELF SHA256=c29d97ab809d41e6cf028d9c478fb77ba7bfebbf25d7879788a4a616f53cf0de
```

仍只在a07-3/dsr1_ep4执行，pin=0、A preshuffle=1、random non-balanced。E65/shared1不再测试。`<70 μs`目标和E64/E65共用同一compiled GEMM1的要求仍未完成；英文commit/push、远端主机pull和pull后最终复测仍待收尾。

## 2026-10-28：v81/v3 四地址组正常计时验收与 v82 owner-loop compaction 计划

针对“汇报下进展，然后继续”：固定 v81/v3 ELF 的 group0–3 对照已经采集结束，CPU 独立复核通过，没有重复运行已成功实验。生产继续保留 v60；本节不替换此前六轮正式 v60/v81 成绩，不新增 MoE e2e 或 GEMM2 性能。

同一 E64/T1536/topk8/shared0 seed0 process 共享原输入、host wrapper/CallState 和原 timer，pin=0、A-preshuffle=1、random non-balanced。最多48个 target 的预定地址覆盖加载过程先结束并冻结，之后每个实例正向/反向各计时一次；29个filler仅加载不dispatch。24个重新加载的目标实例、正反向48个窗口及原路径reference两个窗口全部保留，共50个正常 profile window。所有窗口的20个原 GPU events完整，并依原后19个event均值规则复算；目标text、ABI和输出hash均通过绑定检查。

| group | v3 实例/窗口 | v3 中值 μs | v3 范围 μs | v81 实例/窗口 | v81 中值 μs | v81 范围 μs |
|---:|---:|---:|---|---:|---:|---|
| 0 | 2/4 | 82.136684 | 81.491632–82.278000 | 1/2 | 83.646474 | 83.572842–83.720105 |
| 1 | 2/4 | 84.683658 | 84.532105–85.035842 | 4/8 | 83.942579 | 83.502947–84.306263 |
| 2 | 2/4 | 82.226342 | 81.852158–82.617316 | 6/12 | 83.983711 | 83.372211–84.614211 |
| 3 | 1/2 | 89.509868 | 89.174474–89.845263 | 6/12 | 84.042632 | 83.625421–85.172895 |

v3组间中值跨度 7.373184 μs（max/min−1=8.976725%），v81为 0.396158 μs（0.473610%）。全部正常读数范围分别为 81.491632–89.845263、83.372211–85.172895 μs。v81在本次样本中显著缩小组间分档，但快组相对v3有所代价；group0仅一个v81实例、group3仅一个v3实例，不能把有限样本表述为消除了所有生产波动。

此前v81两轮原路径counters只覆盖自然group1/2，不能补成group0–3 counters；正式六轮GEMM1/e2e结果仍为v60 83.741158/315.172105 μs、v81 84.122737/286.818158 μs。地址复核的`cross_address_performance_repair_accepted=false`保留。硬件映射依据为`MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2 的`Set[4:0] = address[13:9] ^ address[18:14]`；静态跨段、ELF大小和backward-edge区间均不能当作ATT live footprint或唯一瓶颈证明。

下一候选v82基于已验证v81，尝试收敛TDM owner外层dispatch造成的K-loop复制：只在既有`interleaved_m_rows and not SUPPORT_EXPLICIT_VGPR_PARTITION`且`job_waves`互不重叠、完整覆盖所有wave时调用公共`fn(None)`。该调用复用原`issue`中已有owner predicate和原jobs顺序；其他布局保留原outer dispatch。四/六/七/八覆盖body、有效运算、无效行跳过、TDM job geometry、waits/barriers和整个epilogue的源码/AST保持原样，没有新增expert/token/seed specialization。

原先“shared K loop”实验合并的是M-row覆盖body；v82保留这些body，收敛的是owner复制，作用位置不同。循环内owner判断可能增加分支或改变compiler scheduling，必须检查实际ELF和native再做device精度及性能，源码变短本身不构成收益。目前仅源码和AST/owner contract检查通过，没有v82 ELF、native、device或性能成绩；生产源码没有改变。后续继续只研究code footprint/layout，禁止E65/shared1测试，E64/E65共用同一compiled GEMM1要求仍未完成。

```text
address plan SHA256=1ae685e46954dc07f255d4e18d118438fd4d238baf1ee3e7067297b93dc3ac84
address archive=TASK/logs/f3_v81_v3_all_addresses_normal_20261028_v1_evidence.tar.gz
address archive SHA256=f73a396f6f406eec11a8aa2016446f46ade40d6f553e823cdad41404de2ad798
address archive bytes=4099162, members=332, all hashes verified
address review=TASK/logs/f3_v81_v3_all_addresses_normal_20261028_v1.binding_independent_local_review.json
v82 snapshot=.codex_tmp/hyg_f3_followup_20261024/snapshots/f3_v81_shared_owner_mainloop_v82.py
v82 source SHA256=a0eb9177daa645984512d307a4250862e546037b03bc7b827c7c907a5d462ff9
v82 plan SHA256=dc10758b80aa87fcf8d80c5aa3f029823d8f477aadd6c4fb4f3e088c52fe9bb6
v82 source review=TASK/logs/f3_v82_source_independent_CPU_review_20261028_v1.json
production v60 source SHA256=8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008
```

复核/下一步复现命令（已成功的地址实验无需重跑；实际GPU child前仍要求全机11次连续idle、间隔6秒、跨度至少60秒，并使用118秒TERM/再2秒KILL）：

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v81_v3_all_addresses_normal_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v82_compile_20261028_v1.py
```

boot=`ff39c3dc-ffb5-4510-a458-0110ee739267`、branch=`hyg_dsr1_conc6144`、HEAD=`e99dbac65a92ed23f87df4d99bd38f655a923a3b`已只读重检。英文commit/push、远端主机pull和pull后最终复测仍待收尾；未执行clean/reset或丢弃修改，未将实验候选设为生产。

## 2026-10-28：v82 common owner-loop 资源 gate 拒绝，v83 两 owner 共用 body 已冻结

针对“汇报下进展，然后继续”：空闲恢复后v82编译已结束，两项实际child分别用时15.436931秒和0.113727秒，均在120秒内。目标GEMM1仅编译/export，没有dispatch；resource gate发现VGPR spill而拒绝，未执行native/device/正常计时或追加同版测试。原失败summary、stderr和ELF全部保留，CPU独立复核通过。

| compiler 项目 | v81 | v82 |
|---|---:|---:|
| `.text` B | 81920 | 46848 |
| metadata VGPR used | 797 | 797 |
| metadata SGPR used | 59 | 95 |
| LDS B | 327680 | 327680 |
| private segment B | 0 | 20 |
| VGPR spill count | 0 | 4 |
| SGPR spill count | 0 | 0 |

v82成功减少35072 B静态text，但产生4个VGPR spill，不能把这个未dispatch的版本当性能成绩，也不能只凭源码重用程度接受它。metadata使用数并非硬件allocation；此表不重新采用SDK generic VGPR count解释，也没有唯一定位导致spill的register live range。循环内owner分支与descriptor存活范围是下一版需验证的方向，尚不是已证明的spill原因。

v83从v81重新派生，保留四/六/七/八M-row覆盖body，只把相邻两个TDM owner合并为一个outer K-loop body；每个wave仍只进入一个body。其原TDM emit geometry、数学、waits/barriers、epilogue源码/AST保持，issue点按原owner predicate筛选该pair的原jobs顺序。overlap或缺失wave的分组、pin1和guard外配置继续使用原dispatch；没有新增expert/token/seed specialization。源码region逆替换和80个owner/job-order contract检查通过。下一步先编译资源及native gate；目前没有v83 ELF、精度或性能成绩。

运行中曾有另一组E96/T16384/topk6测试使用GPU，等待程序按6秒间隔检查，未终止或竞争其任务；随后通过全机11次连续idle（跨度至少60秒）才启动本轮child。v82结束时post/settled idle、KFD owners为空，生产逐字节恢复v60；没有修改或覆盖外部源码，没有clean/reset。主workload仍为E64/shared0、pin=0，禁止E65/shared1测试，E64/E65共用同一compiled GEMM1要求仍未完成。

```text
v82 archive=TASK/logs/f3_v82_owner_loop_compaction_CPU_20261028_v1_evidence.tar.gz
v82 archive SHA256=048099cf539f46dedd7243c4b8bc8a311fcb9202a1cd6b12c87583492190a049
v82 archive bytes=469784, members=22, all hashes verified
v82 ELF SHA256=38c5dafd2a302e8e5a8595f1ec766ebaea66161d5bb5b2208170e9648e962f9d
v82 review=TASK/logs/f3_v82_resource_gate_failure_independent_local_review_20261028_v1.json
v83 snapshot=.codex_tmp/hyg_f3_followup_20261024/snapshots/f3_v81_pair_owner_mainloops_v83.py
v83 source SHA256=bdba6f98ff8291b577394b1fb5f48ceb50d789689663cd05770502f76317639e
v83 source plan SHA256=9bf99da1e7c845f3263befe9e3c3cb37094424656b18c01374f0bb783c30aabf
production v60 source SHA256=8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008
boot=ff39c3dc-ffb5-4510-a458-0110ee739267
branch=hyg_dsr1_conc6144
HEAD=e99dbac65a92ed23f87df4d99bd38f655a923a3b
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v82_resource_failure_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v83_compile_20261028_v1.py
```

正式六轮结果和地址分档验收表继续保留，不因未dispatch的v82被替换。生产仍为v60；英文commit/push、远端主机pull和pull后最终复测仍待收尾。

## 2026-10-28：v83 两 owner 共用 body 编译及 native gate 通过

针对“汇报下进展，然后继续”：v83编译/export与inspection均已结束，实际child分别16.788295秒、0.113743秒，均小于120秒。全机连续11次idle通过后才启动。结束后源码恢复v60、post/settled idle、无本轮KFD残留。

实际`.text=57856 B`，比v81的81920 B减少24064 B（29.375%）。metadata VGPR used797、SGPR used76、LDS327680 B，private segment、VGPR/SGPR spill count全0；v82的4个VGPR spill未出现在这份v83 ELF中。此处只报告编译事实，不将metadata used数当硬件allocation。

原bounded DS/WMMA模型和`WMMA_only` retirement stress分别通过，issue均为0，CFG iterations为12384/12386，CPU child分别2.321275/1.824223秒。source IR中的`depctr_va_vdst(8)/(0)`各12处；相比v81各20处的静态数量下降来自owner复制收敛，单个原mainloop中的显式wait源码仍保留。native模型不覆盖完整VALU/VMEM/EXEC skip/tensor completion，不代替device精度。

本次CPU archive的22个member hash及实际ELF/ISA machine bytes、原caller signatures和dispatch ABI绑定均通过。下一步复用已经通过native的GEMM1 cache，不重编译G1，运行原完整MoE精度gate0.01并在远端比较全部22020096 B输出：E64/shared0及E65/shared0，seed0/17。E65/shared0只验收guard外已有支持路径，不能由此宣称E64/E65共用同一compiled GEMM1；没有E65/shared1测试。

`.text<64 KiB`不保证任意load address下text不跨64 KiB段，更不等于cache冲突已消失。只有完整MoE精度、原正常timer及不同加载地址/独立counters都符合要求，才考虑生产采用。此checkpoint没有v83 device或正常性能成绩，生产仍保留v60。

```text
v83 CPU archive=TASK/logs/f3_v83_pair_owner_loop_CPU_20261028_v1_evidence.tar.gz
v83 CPU archive SHA256=d3e80c6440ab94d0606f0e83227199e5490154bf726afd225874a0cc1ac8f09c
v83 CPU archive bytes=596559, members=22, all hashes verified
v83 ELF SHA256=44dd209cd4251ba30803dce05550d908dda7c136e7df474b8325f5f66b8c9097
v83 source SHA256=bdba6f98ff8291b577394b1fb5f48ceb50d789689663cd05770502f76317639e
v83 native review=TASK/logs/f3_v83_native_CPU_20261028_v1.review.json
v83 native review SHA256=716d6ae2483488e9c2308c30de018f28b69bb9a61ad0e8dc5e81bd699a8724a9
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v83_native_CPU_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v83_device_prepare_20261028_v1.py
```

保留一次CPU派生工具错误：archiver模板没有`V82`大写日志标签，首次派生工具因此assert失败；保留修改前工具源码后仅修复该可选标签替换，未重跑GPU编译或替换有效ELF。两端三份文档继续CAS追加，历史前缀保留，未整文件覆盖。

## 2026-10-28：v83 完整 MoE 精度通过，原 caller/cache binding preflight 及六轮对照推进

针对“汇报下进展，然后继续”：v83完整MoE四组数值验证已经结束，原`LOGITS_DIFF_TOL=0.01`保留；每组整个`(1536,7168)` bf16输出共22020096 B与已验证参考逐字节一致，differing_bytes=0。GEMM2完整ELF保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`，已经通过native的GEMM1 cache bytes/hash/mtime在device验证前后不变，没有重新编译G1。

| workload | seed | logits_diff | differing_bytes |
|---|---:|---:|---:|
| E64/topk8/shared0 | 0 | 3.6105513278e-06 | 0 |
| E64/topk8/shared0 | 17 | 3.62026933163e-06 | 0 |
| E65/topk9/shared0 | 0 | 3.59489311276e-06 | 0 |
| E65/topk9/shared0 | 17 | 3.60867271782e-06 | 0 |

GEMM2 preparation及四组精度child分别用时 13.683678 / 8.326276 / 8.175984 / 20.494375 / 8.175765 秒，均小于120秒，使用118秒TERM/再2秒KILL。a07-3曾有其他E96/T16384测试间歇占用，controller保留6秒自动检查和全机连续11次idle门槛，未终止外部进程；通过门槛后才启动本轮验证。结束时post/settled idle、KFD owners为空，源码恢复v60。

E65/shared0仅验证guard外原支持路径，不构成E64/E65共用同一compiled GEMM1的验收；这项要求仍未完成。没有执行E65/shared1测试。四组正确性没有任何新性能成绩，不能将child wall time当kernel时间。

六轮正常对照已经冻结，顺序为`v60,v83,v83,v60,v60,v83`，只测E64/T1536/topk8/shared0、seed0、uniform/auto、random non-balanced、pin=0、A-preshuffle=1。原warmup5/iters20、caller、timer和精度gate保持，所有轮次保留，不挑address、seed、快实例或读数，不为慢/失败/占用轮追加替换。每个实际child前全机11次连续idle，至少60秒；child运行期间不并发GPU查询、远端编译或SFTP。原caller/cache binding preflight在实际正常timers之前检查，timed partial key必须与已验证eager artifact相同；后续所有binding/observer hooks在计时前恢复。

静态补充：绑定的v83 ELF `.text`区间为`0x1a00–0xfc00`。在假设load_base为64 KiB对齐时，其整个text位于同一64 KiB段内；依据文档set函数，group0–3的静态同set关系仅作set编号置换，CPU检查通过。这不是实际加载地址、ATT live footprint、miss rate、竞争kernel影响或地址分档消失证明。v83跨加载地址正常性能和独立counters仍待完成，不能提前采用生产或宣称修复。

```text
device summary=TASK/logs/f3_v83_complete_MoE_pin0_20261028_v1.summary.json
device summary SHA256=161f4c85e22d2e184bbdfab2777fa9946b47d9674533f4b94ac2719b483d8b68
normal plan=TASK/logs/f3_E64_v60_v83_original_timer_20261028_v1.frozen_plan.json
normal plan SHA256=9ec5b86196265dfec388c9f51910499a1e32d6cea585ad10093584d43eec6536
conditional set geometry=TASK/logs/f3_v83_single_segment_set_geometry_local_20261028_v1.json
v83 ELF SHA256=44dd209cd4251ba30803dce05550d908dda7c136e7df474b8325f5f66b8c9097
v83 source SHA256=bdba6f98ff8291b577394b1fb5f48ceb50d789689663cd05770502f76317639e
```

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v83_device_prepare_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_E64_v60_v83_original_timer_20261028_v1.py
```

以上命令记录实际复现方法，已经成功的device验证无需重跑。本checkpoint尚无新的正常性能成绩，生产选择继续为v60；英文commit/push、远端主机pull及pull后最终复测仍待收尾。

## 2026-10-28：v83 六轮原 timer 对照完成，孤立 GEMM1 未超过 v60

针对“汇报下进展，然后继续”：已启动的固定六轮正常对照全部完成，没有重启、替换慢轮、选择加载地址或修改原 timer。实际顺序 `v60,v83,v83,v60,v60,v83`，workload 为 E64/T1536/topk8/shared0、seed0、uniform/auto、random non-balanced、pin=0、A-preshuffle=1、explicit VGPR partition=0。原 warmup5/iters20 和 `LOGITS_DIFF_TOL=0.01` 保留。

| 轮次 | 版本 | 孤立 GEMM1 μs | MoE e2e μs | graph GEMM1 μs | graph GEMM2 μs |
|---:|---|---:|---:|---:|---:|
| 1 | v60 | 83.8847368421 | 314.3813684211 | 113.3 | 85.3 |
| 2 | v83 | 84.5382105263 | 284.6676842105 | 94.0 | 74.3 |
| 3 | v83 | 84.3085263158 | 290.5145789474 | 93.6 | 79.9 |
| 4 | v60 | 83.5240526316 | 310.5785263158 | 113.1 | 82.4 |
| 5 | v60 | 83.0222631579 | 311.7884210526 | 113.3 | 83.1 |
| 6 | v83 | 83.9920526316 | 278.8631052632 | 93.2 | 71.1 |
| 中值 | v60 | 83.5240526316 | 311.7884210526 | 113.3 | 83.1 |
| 中值 | v83 | 84.3085263158 | 284.6676842105 | 93.6 | 74.3 |

v83 相对 v60 的孤立 GEMM1 增加 0.7844736842 μs（回退 0.9392188950%）；MoE e2e 减少 27.1207368421 μs（改善 8.6984426011%）。graph 中两阶段的原 profile 表只保留 0.1 μs 精度，不能将其作为更高精度的原始 events，也不能把 e2e 或 graph GEMM1 的收益当作孤立 GEMM1 的收益。GEMM2 完整 ELF 未变；这里的 graph GEMM2 时间差还不足以唯一归因于某个 cache、FIFO 或硬件资源。

117 个 archive member 的大小/hash、实际 GPU loaded text、public HSA load_base、dispatch ABI、原 timed partial/cache key、source IR、原 caller/timer、原精度门槛，以及四组完整 MoE byte comparisons 均经本地 CPU 独立复核。所有计时前 binding/observer hooks 已恢复；每轮全机连续 11 次 idle、间隔 6 秒、跨度至少 60 秒，实际 child 使用 118 秒 TERM/再 2 秒 KILL，全部小于 120 秒。测后 ownership/settled idle 通过，没有本轮 KFD 残留。生产源码逐字节恢复 v60，分支和 HEAD 未变。

六轮自然加载只覆盖 GEMM1 base group 0/2，不能验收跨 group 稳定性。下一步冻结 v83 与 v3 的同一 ELF、同一输入和原 host wrapper/CallState，在计时前仅根据实际 public HSA 地址覆盖 group0–3，随后保留全部实例的正反向计时；最大 48 个 target，不选择快实例，不重编译 G1。v83 native 和完整 MoE 精度验收已绑定；此前 v81 counters 的复核只作为已验证诊断环境的依据，不作为 v83 的 counter 成绩。正常地址实验和独立 counters 分开。

截至此 checkpoint，`<70 μs` 目标仍未达成，v83 的跨地址与 counter 验收仍未完成，不能宣称分档已消失。继续修复加载地址敏感性，不启动新的普通调度优化。E65/shared1 已禁止测试；E65/shared0 保留支持验证，E64/E65 共用同一 compiled GEMM1 的要求仍未完成。生产选择继续为 v60，英文 commit/push、远端主机 pull 和 pull 后最终复测仍待收尾。

```text
normal archive=TASK/logs/f3_E64_v60_v83_original_timer_20261028_v1_evidence.tar.gz
normal archive SHA256=4e698f750083c932b4aee19e31cf50289964f9b29d736eee8d6d207bef4bfe82f
normal archive bytes=4372143, members=117, all hashes verified
normal review=TASK/logs/f3_E64_v60_v83_original_timer_independent_local_review_20261028_v1.json
graph review=TASK/logs/f3_E64_v60_v83_graph_profiles_independent_local_review_20261028_v1.json
address plan=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v1.frozen_plan.json
address plan SHA256=1bf25809b19d0353b5105fd03738b897bb3b9070fc7d6538b5fc50587fb69b20
v83 ELF SHA256=44dd209cd4251ba30803dce05550d908dda7c136e7df474b8325f5f66b8c9097
v3 ELF SHA256=c29d97ab809d41e6cf028d9c478fb77ba7bfebbf25d7879788a4a616f53cf0de
v60 source SHA256=8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008
boot=ff39c3dc-ffb5-4510-a458-0110ee739267
branch=hyg_dsr1_conc6144
HEAD=e99dbac65a92ed23f87df4d99bd38f655a923a3b
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_E64_v60_v83_original_timer_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_E64_v60_v83_graph_profiles_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v83_v3_all_addresses_normal_20261028_v1.py
```

上述命令记录本次复现流程；已有成功性能及精度实验无需重跑。本地与远端三份文档分别 CAS 追加，保留各自历史前缀，不整文件覆盖。

## 2026-10-28：v83 首轮地址采样缺少 group1/3，保留全部结果并修正诊断采样

针对“汇报下进展，然后继续”：v83/v3 首轮地址对照实际 child 在 23.199121 秒内正常结束，测前连续 11 次全机 idle 和测后 idle 均通过，源码恢复 v60、无本轮残留。达到预先冻结的 48 个 target 上限后，v3 覆盖 group0–3，v83 仅覆盖 group0/2；因此不能接受这轮为完整跨地址验收。没有重启或替换任何慢实例，全部 48 个 target、96 个正反向正常计时窗口和原路径 2 个 reference 窗口均保留；原 20 events 完整，按原后 19 events 规则复算。

| kernel variant | base group | target instances | 正反向 timer windows | GEMM1 中值 μs |
|---|---:|---:|---:|---:|
| v3 | 0 | 2 | 4 | 82.2422105263 |
| v3 | 1 | 2 | 4 | 84.5434210526 |
| v3 | 2 | 1 | 2 | 82.1673947368 |
| v3 | 3 | 2 | 4 | 88.8647105263 |
| v83 | 0 | 1 | 2 | 84.4317368421 |
| v83 | 1 | 0 | 0 | 未覆盖 |
| v83 | 2 | 40 | 80 | 84.1470789474 |
| v83 | 3 | 0 | 0 | 未覆盖 |

以上是固定 ELF、同进程同输入和同原 host wrapper/CallState 的地址诊断数据，不替换之前六轮正式 v60/v83 性能。未覆盖组保持 None，不填 0、不估算，也不宣称性能波动已修复。所有 archive member、actual loaded text、dispatch ABI、每次 46 个实际 launch、完整输入前后 hash 和每个输出 snapshot 均经 CPU 独立复核。

从原始 public HSA 地址记录独立复核：v83 的 40 个 clone 全部属于 group2，每两个相邻 clone 的 load_base 差均为 -262144 B（39 次）。在此过程中，前置 tiny filler 的数量已经分别出现 1/2/3 个（16/12/12 次），但没有使 v83 的实际 base group 改变。这证明本轮 filler 数量不能保证采样覆盖；尚未证明 tiny filler 的实际 GPU allocation、去重或某个 allocator 内部行为，不能把这个观察扩展为 loader 的通用对齐限制。

下一轮是独立冻结的诊断采样实验，保留旧样本：target v83/v3 的 ELF、输入、数学、精度、host wrapper/CallState 和原 timer 全部不变。filler 改为已经验证的 v3 ELF，只 load 并保留，不查询 kernel function、不 dispatch；filler cadence 在计时前固定为 1/2/3 循环。新增强制 gate：必须先根据实际 public HSA 地址确认两个 variant 均覆盖 group0–3，才开始 comparison timers。larger filler 是否改变地址采样仍须实测，不提前假设成功。下一轮继续全机 idle 门槛和 118 秒 TERM/再 2 秒 KILL，运行期间不并发 GPU 查询、编译或 SFTP。

保留一次 CPU 派生工具失败：把 filler hash 重绑定为已存在的 v3 hash 后，inverse 全局 hash 替换也误触了原 v3 binding，因此精确逆替换 assert 阻止了生成 child/frozen plan。已保留修改前脚本与失败 receipt，仅改为替换完整 filler dictionary field；再次精确逆替换及 AST/bytes 检查通过。没有因此重新编译、dispatch 错误 ELF 或改变生产代码。

文档勘误：上一 checkpoint 的 normal archive SHA256 手工转录时多写了一个 `e`。以实际 archive/manifest 和已通过的独立复核为准，正确值为 `4e698f750083c932b4aee19e31cf50289964f9b29d736ee8d6d207bef4bfe82f`；原 archive、117 个 member 和性能/精度结果均未改变。此处追加更正，保留两端文档历史前缀。

```text
partial address archive=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v1_evidence.tar.gz
partial address archive SHA256=e1c3a7cd7f1d0bd849c3a446da85c109324427cb93580ad27ab7b8b635947133
partial address archive bytes=14404124, members=1167, all hashes verified
partial review=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v1.partial_coverage_independent_local_review.json
loader sampling review=TASK/logs/f3_v83_v1_loader_sampling_independent_local_review_20261028_v1.json
new address plan=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v2.frozen_plan.json
new address plan SHA256=d6937f3374337b675241bf61e5760347adc77388d7aa45d88c3da21cb87184e0
sampling source diff review=TASK/logs/f3_v83_address_sampling_diff_review_local_20261028_v2.json
derivation failure=TASK/logs/f3_v83_address_sampling_derivation_inverse_failure_local_20261028_v2.json
v83 ELF SHA256=44dd209cd4251ba30803dce05550d908dda7c136e7df474b8325f5f66b8c9097
v3 ELF SHA256=c29d97ab809d41e6cf028d9c478fb77ba7bfebbf25d7879788a4a616f53cf0de
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v83_v3_partial_address_coverage_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v83_v1_loader_sampling_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v83_v3_all_addresses_normal_20261028_v2.py
```

首轮精度和正常 timer 数据有效但地址覆盖不足，`cross_address_performance_repair_accepted=false` 保留。第二轮尚无新性能成绩，独立 counters 也未开始。生产继续保留 v60；E65/shared1 不再测试，E64/E65 共用同一 compiled GEMM1 仍未完成，已授权 Git 收尾待验收后执行。

## 2026-10-28：保留第二轮地址采样失败，完成独立 loader-only geometry 观测

针对“汇报下进展，然后继续”：生产仍为 v60，正常六轮结果不变：E64/T1536/topk8/shared0、random non-balanced、pin=0 下，孤立 GEMM1 中值 v60=83.5240526316 μs、v83=84.3085263158 μs；MoE e2e 中值分别为311.7884210526/284.6676842105 μs。v83 孤立 GEMM1 回退0.9392188950%，e2e改善8.6984426011%，不能把后者作为前者的收益。本 checkpoint 没有新的 kernel 性能成绩。

### 第二轮正常地址采样在 comparison timer 前被 gate 拒绝

larger filler 版本实际 child 在38.120177943秒内以 returncode=1 退出。完整48个 target、93个 filler 均保留；v3只覆盖group0/2，v83仅有原实例group2。流程先尝试补齐v3，未进入v83 clone采样阶段，因此不能声称large filler已经在v83 clone上验证无效。原 eager MoE setup 和一次quant profile实际执行过，但没有任何GEMM1 comparison timer window，不能把整个运行说成零GPU dispatch。

verbose archive留在远端，未下载。另存 compact archive，仅省略旧matching对象重复text export，保留全部新target/filler实际loaded text、完整记录和必要源码。本地独立核验341个member的bytes/SHA256、public HSA base及source恢复通过；全部观察原样保留，不替换慢组、不填造缺失组。source恢复v60、测后无本轮KFD残留。

```text
compact archive=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v2_failure_compact_evidence.tar.gz
SHA256=7f8905e73cec9cdb73ad938d1ce7d746e5365312da6ad0abbfd09b3fd5d7716c
bytes=8110920, members=341
review=TASK/logs/f3_v83_v3_address_sampling_failure_independent_local_review_20261028_v2.json
```

### 独立诊断的 device index 误用已修正；原性能记录不变

loader-only v1在`hip.hipSetDevice(plan["device_index"])`处退出，耗时0.163820424秒、returncode=1，尚未执行显式target module load，更未查询kernel function、执行kernel或调用MoE/timer。此次失败全部32个archive member保留并通过CPU独立核验，production source始终不变，测后全机idle、无本轮KFD残留。

原因是诊断计划误把原profiler event的`device_index=2`当成HIP device ordinal。核验原compact archive的完整input描述后确认，原测试张量在`cuda:0`，而event字段确实为2；这两个字段不能互相替代。旧review中名为`device_index_in_original_setup`的值2，仅来自profiler event，并不是设备选择API的参数。此处追加勘误，旧记录和原性能events不改写。

v2仅修正诊断设备选择为HIP ordinal0，显式绑定已验证的HIP library，并记录实际device count和进程已加载library路径；原ELF、76次module load计划、各phase及production源码全部保留。v2实际`hipGetDeviceCount`返回1，当前ordinal0有效。native `mgpuModuleLoad/mgpuModuleUnload` library仍与原测试一致，HSA public loader ABI及镜像读取/hash检查全部保留。不import torch、不查询function、不launch kernel、不调用MoE或timer。

```text
v1 failure archive=TASK/logs/f3_v83_loader_only_geometry_probe_20261028_v1_failure_evidence.tar.gz
SHA256=aa58948ef98e62bc99bae0717d05a32fabac0c88194cef36f9b332a6c144b6c6
bytes=141735, members=32
device correction review=TASK/logs/f3_v83_loader_only_geometry_probe_20261028_v1.device_failure_independent_local_review.json
HIP library=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamdhip64.so.7
HIP library SHA256=4a3318b0245da02ee3e1203585db96041c6646b7899d6559fef9cea2a10ce8e6
native loader SHA256=9536601c5a661fe9d8e83e7c876cbb05ac7cd2fc1d7f655969e168572e770c51
```

### 76次module load全部完成，但不能作为性能验收

v2实际child耗时0.915114812秒、returncode=0，全部76个HIP module handle不同，每次都取得一个独立的新HSA对象。每个实际loaded text与对应已验证ELF逐字节相等，全部public HSA base/size及hash经本地CPU独立核验。所有module保留到7个固定phase结束，再统一卸载；每次相同live HSA对象的text export复用首次路径，原HSA查询、实际mirror读取和hash检查仍保留，避免此前O(N²)重复文件。

| 固定phase | target ELF | 前置filler ELF | target数 | 实际target base groups |
|---|---|---|---:|---|
| v83_none | v83 | 无 | 4 | 1/3 |
| v3_small | v3 | small640B | 8 | 2/3 |
| v83_small | v83 | small640B | 4 | 2 |
| v83_v3 | v83 | v3 | 8 | 2 |
| v83_v81 | v83 | v81 | 8 | 2 |
| v83_v60 | v83 | v60 | 4 | 0/2 |
| v83_gemm2 | v83 | 原GEMM2 | 4 | 0/2 |

v83在这7个phase的合集中实际覆盖group0–3，证明它并非只能加载到group2；每个单独phase仍未覆盖四组。v3仅覆盖group2/3。这个实验没有comparison timer，不能验收跨地址正常性能，更不能证明miss放大消失。全部新HSA对象不同只证明本轮load各自产生独立对象，不补造历史采样的allocator分支或其他未记录事实。

本轮HSA `load_size`：v83=69632 B、v3=110592 B、v81=94208 B、v60=57344 B、原GEMM2=61440 B、small=12288 B。ELF `.text`大小仍为57856/97408/81920/44416/47872/640 B，二者不可混用。v83前四个base为`0x7da7f6c10000`、`0x7da7ee8f0000`、`0x7da7ee8b0000`、`0x7da7ee870000`；首次与后续加载间出现明显VA区间切换，后两步为-262144 B。这是实际地址观测，不能唯一解释为某个通用allocator保证。

```text
loader-only v2 archive=TASK/logs/f3_v83_loader_only_geometry_probe_20261028_v2_evidence.tar.gz
SHA256=55db317db67bdc8aca8ebaa9ade1204885edc83d8aff4cac33432b594e5d4fb39
bytes=2387064, members=195, all hashes verified
review=TASK/logs/f3_v83_loader_only_geometry_probe_20261028_v2.independent_local_review.json
plan SHA256=0d5db24acf7bd3c597b7a24487c1438e259920471374c5108d8208bb07e2d3c3
production v60 source SHA256=8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008
boot=ff39c3dc-ffb5-4510-a458-0110ee739267
branch=hyg_dsr1_conc6144
HEAD=e99dbac65a92ed23f87df4d99bd38f655a923a3b
```

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v83_loader_only_geometry_probe_20261028_v2.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v83_loader_only_geometry_probe_local_20261028_v2.py
```

已有成功loader-only诊断及六轮正常性能无需重跑。下一步先以独立、固定、限时的loader-only计划验证私有CPU VA reservation能否改变实际HSA地址覆盖；这是改变本进程mmap空洞布局的诊断假设，不改runtime、原ELF或production，不使用`MAP_FIXED`，效果必须以实际地址为准。得到可靠采样方法后再恢复原caller/timer的同输入group0–3对照，独立counters仍待采集。

本轮全机连续11次idle、间隔6秒、跨度至少60秒；child保留118秒TERM/再2秒KILL，运行期间没有并发GPU查询、编译或SFTP。post/settled全机idle通过，无本轮残留。继续只在a07-3/dsr1_ep4测试，pin=0、explicit VGPR partition=0、A-preshuffle=1，E65/shared1不再测试。`<70 μs`目标、E64/E65共用同一compiled GEMM1、已授权英文commit/push、远端主机pull及pull后最终复测仍未完成。

## 2026-10-28：私有 VA 采样覆盖四组，v83 正常计时的地址分档明显收窄

针对“汇报下进展，然后继续”：本轮完成了 loader-only 私有 CPU VA reservation 验证，以及同一进程、同一输入、固定 v83/v3 ELF 的正常四地址组对照。生产仍保留 v60；已核验的六轮正式结果保持为孤立 GEMM1 中值 v60=83.5240526316 μs、v83=84.3085263158 μs，MoE e2e 中值分别为311.7884210526/284.6676842105 μs。下面的新数据用于加载地址诊断，不替换上述六轮正式结果。

### 先用零 kernel launch 的固定实验验证采样方法

48次 loader-only load 分六个固定 phase 完成，每个 phase 保留8个独立的新HSA对象。全部实际 loaded text、public HSA base、私有匿名 CPU mapping 以及末尾的 module unload/VA release 经本地CPU独立核验。没有显式function lookup、kernel launch、MoE或comparison timer；production source始终不变，测后无本轮KFD残留。

| 每次load前的私有CPU VA reservation | v3实际base groups | v83实际base groups |
|---|---|---|
| 64 KiB | 0/1/2 | 0/1/2/3 |
| 192 KiB | 0/1/2/3 | 0/1/2/3 |
| 320 KiB | 0/1/2/3 | 1/2/3 |

据此固定选择192 KiB作为后续诊断采样方法。选择只依据地址覆盖，没有使用性能数据，没有选择快实例。CPU mapping不注册到GPU，不使用`MAP_FIXED`，不修改ELF或runtime；它只是帮助诊断覆盖加载地址，不作为生产修复方案。

```text
loader-only VA archive=TASK/logs/f3_v83_loader_only_VA_reservation_20261028_v1_evidence.tar.gz
SHA256=ffdc5979ba8e08e395bc58ac174cae6af6add7a66f050449e66bf194ef38a599
bytes=1919167, members=123, all hashes verified
review=TASK/logs/f3_v83_loader_only_VA_reservation_20261028_v1.independent_local_review.json
```

### 正常原timer对照：v3四组中值差6.56 μs，v83为0.27 μs

workload固定为E64/T1536/topk8/shared0、model_dim=7168/inter_dim=2048、random non-balanced、pin=0、explicit VGPR partition=0、A-preshuffle=1、seed0。保留原caller/CallState和timer，warmup5/iters20；每个窗口20个GPU events，原后19个events平均值由本地reviewer独立重算。

只按实际HSA地址覆盖规则加载，实际共9个target实例（v3=4、v83=5），零filler；两个variant均覆盖group0–3后才开始comparison timer。全部实例按加载顺序正向一次、逆序反向一次，共18个完整窗口；每窗口46次实际GEMM1 launch均经native计数核验。没有重编译GEMM1，没有筛选快实例、快轮次或seed。输入指针、version、前后完整hash不变；原reference及所有正反向输出payload/scale逐字节一致。

| variant | `(public_HSA_load_base >> 16) & 3` | 实例数 | timer窗口数 | 正反向窗口中值 μs | 全部窗口范围 μs |
|---|---:|---:|---:|---:|---|
| v3 | 0 | 1 | 2 | 82.321211 | 81.919421–82.723000 |
| v3 | 1 | 1 | 2 | 84.770132 | 84.382105–85.158158 |
| v3 | 2 | 1 | 2 | 82.489895 | 82.157842–82.821947 |
| v3 | 3 | 1 | 2 | 88.881342 | 88.421684–89.341000 |
| v83 | 0 | 1 | 2 | 84.592079 | 84.247421–84.936737 |
| v83 | 1 | 2 | 4 | 84.324237 | 84.198632–84.611842 |
| v83 | 2 | 1 | 2 | 84.400079 | 84.297789–84.502368 |
| v83 | 3 | 1 | 2 | 84.389447 | 84.384263–84.394632 |

本轮组间中值差从6.560131579 μs降到0.267842105 μs；全部正常窗口范围分别为81.919421–89.341000 μs和84.198632–84.936737 μs。v83改善了v3的慢组，同时损失了快组速度。这里仅完成一个同进程四组正常计时对照，不能据此承诺生产中的所有性能波动消失，也不能把v83的e2e收益当作孤立GEMM1收益。

child实际耗时11.480973324秒、returncode=0。测前全机连续11次idle、间隔6秒、跨度61秒；child运行期间没有并发GPU查询、远端编译或SFTP，post/settled均idle，无本轮残留。生产源码恢复v60并经archive/local reviewer再次核验；全部CPU reservations位于profile windows外、与实际HSA code区间不重叠，末尾全部释放。

```text
normal archive=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v3_evidence.tar.gz
SHA256=1e6ccc4763ce891cd1feb50d85c05357e349f0831cb6ebf37354f805a957e764
bytes=906373, members=108, all hashes verified
normal review=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v3.paired_normal_independent_local_review_v2.json
binding review=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v3.binding_independent_local_review.json
VA review=TASK/logs/f3_v83_v3_all_addresses_normal_20261028_v3.VA_reservation_independent_local_review.json
normal frozen plan SHA256=02473ecf60dd0227106d7ec3b0371fb49af702cb85fcd3f74be79f0bf3ba173f
v83 G1 ELF SHA256=44dd209cd4251ba30803dce05550d908dda7c136e7df474b8325f5f66b8c9097
v3 G1 ELF SHA256=c29d97ab809d41e6cf028d9c478fb77ba7bfebbf25d7879788a4a616f53cf0de
```

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v83_v3_all_addresses_normal_20261028_v3.py
python3 .codex_tmp/hyg_f3_followup_20261024/pack_f3_v83_v3_all_addresses_normal_20261028_v3.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v83_v3_all_addresses_normal_local_20261028_v3.py
```

这些label的结果已存在，应保留；复现另建label，不覆盖或重跑已完成的记录。

### 下一步只做独立counters验收，尚无新counter结论

另冻结一个独立counter process，继续使用同样的192 KiB采样规则、固定v83/v3 ELF、原caller/timer和全部实例正反向顺序，两个variant缺任何group就拒绝comparison windows。只采集`SQC_ICACHE_REQ/HITS/MISSES`及`SQ_WAVES`，通过native累计launch ordinal将SDK CSV逐项绑定到实际ELF、public HSA地址和function handle。SDK instrumented duration不作为正常性能；已判无效的SQG clock counters继续排除。

本轮counter诊断仅更新工具为当前boot/real rocminfo环境，并检查profiler、config、roctx和shim的实际hash；GEMM1不重编译，timer及kernel计算不改。该流程已启动但此checkpoint尚未引用counter结果，`cross_address_performance_repair_accepted=false`保留。

```text
counter frozen plan=TASK/logs/f3_v83_v3_all_addresses_counters_20261028_v1.frozen_plan.json
SHA256=4832c9bfb42c2e99b7eb2438e5c15066ace9024b94f06316bd0b9977da5bddc0
frozen files=65, remote hashes verified
counter tools diff review=TASK/logs/f3_v83_VA_counter_tools_diff_review_local_20261028_v1.json
boot=ff39c3dc-ffb5-4510-a458-0110ee739267
branch=hyg_dsr1_conc6144
HEAD=e99dbac65a92ed23f87df4d99bd38f655a923a3b
```

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v83_v3_all_addresses_counters_20261028_v1.py
```

硬件依据为`MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2：`Set[4:0] = address[13:9] ^ address[18:14]`。v83 `.text`为57856 B，实际相对区间`0x1a00–0xfc00`在本轮64 KiB对齐base下没有跨段；这可支持内部相对set关系不随上述base group变化，不能单独证明miss消失、bank hash或live residency。SDK对`SQC_ICACHE_MISSES`注明包括uncached requests且nondeterministic，最终判断需要独立正常计时和counters共同支撑。

继续暂停普通调度优化，先判断慢组的miss放大是否被压下去。仅在a07-3/dsr1_ep4测试，E65/shared1不再测试。`<70 μs`目标、E64/E65共用同一compiled GEMM1，以及已授权英文commit/push、远端主机pull和pull后最终复测仍未完成。

## 2026-10-28：v83 四组独立 counters 通过，继续核验 balanced row selector

针对“汇报下进展，然后继续”：同一boot的v83/v3独立counter process已经结束，固定ELF、实际HSA地址、function handle、native累计launch ordinal与SDK CSV逐项绑定通过。本轮child耗时16.438163377秒、returncode=0，9个target实例全部保留并各做正反向一次；829次GEMM1 dispatch由native/SDK共同核验，包括原untimed reference一次及18个窗口各46次launch。每个窗口保留原warmup5/iters20和全部实际运算。

按既定协议统计每窗口最后20次profiler dispatch，两个方向合计每实例40次；v3四组各一个实例，v83四组分别为1/1/2/1个实例。以下全部为独立counters，SDK instrumented duration不作为正常性能。

| variant | group | 统计dispatch数 | `SQC_ICACHE_REQ`中值 | `SQC_ICACHE_MISSES`中值 | misses/requests的逐dispatch中值 | `SQ_WAVES`全部读数 |
|---|---:|---:|---:|---:|---:|---:|
| v3 | 0 | 40 | 1877856 | 5800 | 0.308789% | 7168 |
| v3 | 1 | 40 | 1877826 | 25992 | 1.383049% | 7168 |
| v3 | 2 | 40 | 1877796 | 5887.5 | 0.313492% | 7168 |
| v3 | 3 | 40 | 1877886 | 28621.5 | 1.524450% | 7168 |
| v83 | 0 | 40 | 1078464 | 0 | 0% | 7168 |
| v83 | 1 | 40 | 1078554 | 0 | 0% | 7168 |
| v83 | 2 | 80 | 1078644 | 0 | 0% | 7168 |
| v83 | 3 | 40 | 1078404 | 0 | 0% | 7168 |

v3 group1/group3的miss中值分别为group0的4.481379/4.934741倍；v83四组共200次计入统计的dispatch，`SQC_ICACHE_MISSES`范围全部为[0,0]。两版所有统计dispatch的`SQ_WAVES`均为7168，输入版本/指针/完整hash不变，reference及全部输出payload/scale逐字节一致，不能把结果解释成减少wave或漏算有效行。

这与前节独立正常timer的结果相符：v3 group0–3中值为82.321211/84.770132/82.489895/88.881342 μs；v83为84.592079/84.324237/84.400079/84.389447 μs，组间中值差由6.560132 μs降为0.267842 μs。当前观测范围内，v83的代码结构改善了已定位的加载地址敏感性。正常timer和counters为两个独立process，各自内部两版共享实际输入；它们的完整A hash不同，没有假设跨process物理输入相同，也没有将SDK duration混入正常成绩。

上述“0 misses”仅指本轮200次固定profile tail dispatch的实际counter读数，并非宣称所有冷启动、输入或生产条件下永远没有miss。SDK定义中`SQC_ICACHE_MISSES`包括uncached requests且nondeterministic，misses/requests只能作为经验事件比值，不能当作唯一硬件归因或精确cache hit-rate。`observed_four_group_address_sensitivity_mitigation_supported=true`，`all_production_variability_eliminated=false`、`production_promoted=false`。

测前全机11次连续idle、间隔6秒、跨度至少60秒；child保留118秒TERM/再2秒KILL。运行期间没有并发GPU查询、远端编译或SFTP，post/settled均idle，没有本轮KFD残留。全部额外module和私有CPU VA reservation已释放，生产源码恢复v60并经archive独立核验。

```text
counter archive=TASK/logs/f3_v83_v3_all_addresses_counters_20261028_v1_evidence.tar.gz
SHA256=1d8ec4ff344e9aab47d60eac75843e4c1357970d2e33b011b1e09e04146fa16b
bytes=1315267, members=122, all hashes verified
counter dispatch binding=TASK/logs/f3_v83_v3_all_addresses_counters_20261028_v1.paired_counter_dispatch_binding_local_v1.json
counter review=TASK/logs/f3_v83_v3_all_addresses_counters_20261028_v1.paired_counter_independent_local_review_v1.json
VA/binding review=TASK/logs/f3_v83_v3_all_addresses_counters_20261028_v1.VA_reservation_binding_independent_local_review.json
combined review=TASK/logs/f3_v83_four_address_combined_independent_local_review_20261028_v1.json
```

```bash
python3 .codex_tmp/hyg_f3_followup_20261024/pack_f3_v83_v3_all_addresses_counters_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v83_v3_VA_counters_local_20261028_v1.py f3_v83_v3_all_addresses_counters_20261028_v1
python .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v83_four_address_evidence_local_20261028_v1.py
```

已完成的label及review继续保留，无需重复采集。

### v84仅重排原八叶accumulator选择分支，目前没有性能成绩

v83的runtime row epilogue通过八个`if/elif`分支选择原accumulator，正常计时约84 μs，比v3快组仍慢约2 μs。下一候选v84只把该selector重排为深度3的二叉分支树；八个叶节点的原store AST逐项相等，所有`wm=0..7`叶选择一致，518项有效row覆盖对照通过。原row loop bound、无效row条件、`emit_quant_row`、TDM jobs/owner loop、mainloop数学、waits/barriers、输出mask及写入顺序的源码/AST保持不变。没有增加expert/token/seed specialization，没有修改caller、计时或精度。

分支树在源码中的深度不代表已经证实GPU动态分支减少；LLVM可能重排或将两种形式编译成同样代码。先仅编译/export目标GEMM1并检查真实ELF及资源。如果`.text`与v83相同，就不再执行同版GPU数值或性能测试。如果恢复跨64 KiB段的代码区间，或出现spill，也会保留失败证据后停止该候选。编译工具保留原输入preparation，后者可能使用GPU，不能把整个编译流程说成零GPU操作。

```text
v84 snapshot=TASK/snapshots/f3_v83_balanced_row_selector_v84.py
v84 source SHA256=041a04eaa1dd7496efed5b1de294d322ae016eb712b2f3162eed4829dd05d697
source plan=TASK/logs/f3_v84_balanced_row_selector_source_20261028_v1.plan.json
source review=TASK/logs/f3_v84_source_independent_CPU_review_20261028_v1.json
compiler runner=TASK/run_f3_v84_compile_20261028_v1.py
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/make_f3_v84_balanced_row_selector_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v84_compile_20261028_v1.py
```

当前生产仍为v60，正式六轮孤立GEMM1中值83.5240526316 μs，距离70 μs还差13.5240526316 μs。v83对应正式中值84.3085263158 μs，MoE e2e虽改善但不能替代孤立GEMM1目标。继续只在a07-3/dsr1_ep4、E64/shared0、pin=0下测试，E65/shared1不再测。E64/E65共用同一compiled GEMM1和已授权Git收尾尚未完成。

## 2026-10-28：v84 的预设 `.text` gate 拒绝记录与 v85 向量索引计划

针对“汇报下进展，然后继续”：生产仍保留 v60，最新正式六轮对照中 E64/T1536/topk8/shared0、random non-balanced、pin=0 的孤立 GEMM1 中值为83.5240526316 μs，MoE e2e为311.7884210526 μs，尚未达到70 μs。v83的四个加载地址组正常计时与独立counters已经核验，组间GEMM1中值差由v3的6.560132 μs降至0.267842 μs；这支持观测范围内地址敏感性改善，但v83正式孤立GEMM1中值84.3085263158 μs，未替换生产。已有成功采集不重跑。

### v84 编译结束，未进入目标 GEMM1 的 native、精度或性能测试

两个实际child均returncode=0：目标编译/export耗时16.790067819秒，ISA inspection耗时0.113854643秒，均保留118秒TERM/再2秒KILL。controller在预设的`single_static_64KiB_segment_under_observed_base_alignment`检查失败，`completed=false`和原`AssertionError`完整保留；没有执行目标GEMM1的device dispatch，没有v84性能成绩。原输入preparation可能使用其他GPU kernel，不能称整个编译流程零GPU操作。

v84 `.text=59008 B`，相对区间`0x1a00–0x10080`，比v83的57856 B增加1152 B。原runner的资源assert位于失败的`.text`assert之后，不能宣称原runner已通过资源gate。后续独立解析ELF metadata核对结果为VGPR=797、SGPR=76、LDS=327680 B、private=0、VGPR/SGPR spills=0，与v83相同；这是另一次CPU资源复核。

完整ISA分析进一步限定失败含义：v84跨过`0x10000`的32条4B指令全部为尾部`s_code_end`填充，最后一个非填充指令的结束位置为`0xfebc`，非填充代码本身仍在第一段。不能据`.text`门禁失败直接宣称热点执行代码跨段，也不能据此推断实际性能回退。原gate失败不改写为成功，本轮不再给v84补做device或性能测试。

完整编码字节核算中，v83/v84静态branch数为186/177，WMMA sites均为1200。v84多出64条`s_set_vgpr_msb`和96条`v_dual_mov_b32`，两项增加1024 B；包括其他编码变化与尾部填充后总增长1152 B。分支树深度由源码8叶ladder变为3，不等于整体机器码更小。这里是静态指令及编码字节，不能代替动态retirement或耗时。

首次CPU reviewer的opcode histogram未包含带symbol target annotation的branch行；完整ISA analyzer已另行修正parser，逐行包含branch annotation并核对编码字节连续覆盖整个`.text`。先前ELF/metadata/hash和失败状态核验仍保留，本文branch数使用完整analyzer结果。

```text
v84 ELF SHA256=7c9bf99c702169ca23f9575ec71093714b014d3cdd1e58bd568d82bc2dfa461a
v84 .text SHA256=f8a378129e4df3c6e5ef401a262dc188f882df526e55cc70135cca73f22bf3fc
failure archive=TASK/logs/f3_v84_rejected_layout_CPU_20261028_v1_evidence.tar.gz
SHA256=516a0017cde844f2f95f1861af0947690dabf0269e60953f26aca574de0ca3cf
bytes=611955, all manifest members verified
failure review=TASK/logs/f3_v84_rejected_layout_independent_local_review_20261028_v1.json
complete ISA review=TASK/logs/f3_v83_v84_complete_ISA_independent_local_review_20261028_v1.json
```

controller的`finally`已恢复v60，源码SHA256仍为`8666f612497e2f064136a57250aa7014e3ebf1ebb79004124d252c3ff20c8008`。原post/settled记录出现其他容器`jli-mi450`的E96/T16384/topk6测试，占用事实照实保留，不能写成post/settled idle。后续独立进程检查未发现本轮compiler/inspector残留；没有终止外部任务。按6秒自动轮询的全机连续11次idle检查随后完成，日志为`TASK/logs/f3_v84_post_external_followup_idle_20261028_v1.log`，exit=0。下一次实际测试仍重新检查空闲，不复用这次空闲结论。

### v85：以原 accumulator 的动态向量索引替换八叶复制 ladder

以已验收的v83源码为父版本，将相同的accumulator按原row/column/element次序组成向量，通过`wm * output_n_rep * 8 + wn * 8 + p`提取当前row。源码层面不对Float32值进行任何新运算，原`emit_quant_row`的激活、amax、scale、FP4转换、LDS和global输出AST保持。原row loop bound、无效row判定、TDM issue/owner顺序、mainloop、waits/barriers、fallback、caller和计时保持；没有新增expert/token/seed/data specialization。8704项symbolic operand mapping以及518项row覆盖检查完成，生产源码未改变。

动态索引的实际LLVM lowering尚未知，可能生成`V_MOVREL`、选择指令或scratch。依据CDNA5 ISA §3.3.2.2/§3.3.2.3（文档页16–17），必须检查VGPR寻址及资源，不能预先假定指令更少。首先只编译/export并检查真实ELF；出现spill或静态`.text`跨段即保留失败，停止该候选。CPU/native通过后才进入E64/shared0及E65/shared0原精度验证、原timer和四地址组验收。E65/shared1继续禁止测试。

```text
v85 snapshot=TASK/snapshots/f3_v83_indexed_row_selector_v85.py
v85 source SHA256=33606d1af46dcbe1e37cd3908698aa16896e417ded54e3504f1ca271994d0f35
source plan=TASK/logs/f3_v85_indexed_row_selector_source_20261028_v1.plan.json
source review=TASK/logs/f3_v85_source_independent_CPU_review_20261028_v1.json
compiler runner=TASK/run_f3_v85_compile_20261028_v1.py
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v84_rejected_layout_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v83_v84_complete_ISA_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/make_f3_v85_indexed_row_selector_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v85_compile_20261028_v1.py
```

已有label、archive和review不覆盖，复现另建label。仅在a07-3/dsr1_ep4、pin=0下继续；`<70 μs`、E64/E65共用同一compiled GEMM1、已授权的英文commit/push、远端主机pull及pull后最终复测仍待完成。

## 2026-10-28：v85 动态索引的 private memory 门禁拒绝，继续核验短向量 v86

针对“汇报下进展，然后继续”：v85编译及ISA inspection已结束，源码恢复v60；没有启动目标GEMM1的native、device精度或性能测试。正式性能仍引用已完成的六轮原timer对照，v60孤立GEMM1中值83.5240526316 μs、MoE e2e中值311.7884210526 μs，尚未达到70 μs。v83的四地址组稳定性证据保持，不重复采集，也未替换生产。

### v85 的失败是实际 private memory lowering，不是数值或计时失败

目标编译/export child耗时18.341329177秒、returncode=0，inspection child耗时0.163829815秒、returncode=0。原controller先记录资源gate，再因`resource_gate_passed=false`抛出原`AssertionError`，`completed=false`完整保留。两者均使用118秒TERM/再2秒KILL。测前全机连续11次idle、间隔6秒、跨度至少60秒；等待期间发现的其他测试按6秒自动轮询，没有终止或竞争外部任务。运行期间没有SFTP、GPU查询或远端并行编译；post/settled均idle，没有本轮child/KFD残留。

ELF `.text=158848 B`，相对区间`0x1a00–0x28680`；VGPR=797、SGPR=76、LDS=327680 B，metadata的`private_segment_fixed_size=131200`，虽然`vgpr_spill_count=0`和`sgpr_spill_count=0`，仍存在大量private memory。不能把两个spill计数为0解释为没有scratch。

独立核验覆盖完整`.text`编码字节、source hash、原caller的dispatch ABI及源码operand mapping。实际ISA中有8192个`scratch_store_b128`和64个`scratch_load_b32`静态sites，没有`V_MOVREL`指令。这与512-element向量被反复存入private memory再按动态索引读取相符；静态store数8192等于64个读取组件各保存512/4个b128片段。保存的lowered MLIR没有`llvm.alloca`文本行，不能声称已经在这份MLIR里看到64个alloca；实际scratch证据来自最终ELF metadata和ISA。

```text
v85 source SHA256=33606d1af46dcbe1e37cd3908698aa16896e417ded54e3504f1ca271994d0f35
v85 ELF SHA256=a32af233716ba7eeb5a699aef04effc61c4659e32b3e3ed69bd2dc0c9d59500b
failure archive=TASK/logs/f3_v85_rejected_indexed_selector_CPU_20261028_v1_evidence.tar.gz
SHA256=1cd6c89a931f4d26b06000415f67450ec28dbb95445200e1aa52246f300028a4
bytes=784985, all members verified
failure review=TASK/logs/f3_v85_rejected_indexed_selector_independent_local_review_20261028_v1.json
source algebra review=TASK/logs/f3_v85_operand_mapping_independent_local_review_20261028_v1.json
```

源码代数review独立读取旧selector的8个叶节点，核对新旧loop bound和无效row predicate AST，并证明对于任意正`output_n_rep`，`wm * output_n_rep * 8 + wn * 8 + p`恰好读取原`(wm,wn,p)`。Float32值只构造/提取vector，不新增Float32数学；原`emit_quant_row`的激活、amax、scale、FP4转换与所有输出AST保持。但源码语义等价不代表lowering有效率，v85资源失败不作数值结果或性能数据。

### v86 的索引表长度在目标配置下缩为8，先做实际编译gate

新候选从原v83独立派生，每个原column element构造一个包含8个row候选的向量，读取`row_banks[wn * 8 + p][wm]`。由此消除一个512-element索引表，保留相同的所有accumulator位模式、原row条件和激活/量化body。独立AST和代数检查确认外层column/element索引为`wn * 8 + p`，内层row索引为原`wm`，原mainloop/TDM/quant row函数AST及region外bytes保持。没有增加expert/token/seed/data specialization，没有调整精度、caller或计时，生产仍为v60。

较短的vector是否产生`V_MOVREL`、select或scratch，仍以最终ELF为准，不预先承诺收益。先检查private memory/spills、资源及静态`.text`区间；任何门禁失败就停止该版，不运行device或性能测试。若通过，再做native、E64/shared0和E65/shared0完整原精度验证，以及原timer和多加载地址验证。E65/shared1继续禁止测试。

```text
v86 snapshot=TASK/snapshots/f3_v83_small_indexed_row_selector_v86.py
v86 source SHA256=789b6df2eef955efee31ede958fc3cc1cdff3c2aa3df0049072ccc5a2baaac01
source plan=TASK/logs/f3_v86_small_indexed_row_selector_source_20261028_v1.plan.json
source review=TASK/logs/f3_v86_source_independent_CPU_review_20261028_v1.json
operand review=TASK/logs/f3_v86_operand_mapping_independent_local_review_20261028_v1.json
compiler runner=TASK/run_f3_v86_compile_20261028_v1.py
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v85_rejected_CPU_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/make_f3_v86_small_indexed_row_selector_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v86_operand_mapping_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v86_compile_20261028_v1.py
```

现有label、archive、manifest及review继续保留，复现另建label，不下载`.out`、cache或完整MoE输出binary。仅使用a07-3/dsr1_ep4、pin=0。E64/E65共用同一compiled GEMM1和已授权Git收尾仍待完成，不能误报为已经实现。

## 2026-10-28：v86 的实际动态索引开销已封存，继续编译检查 v87 两行显式 select

针对“汇报下进展，然后继续”：生产仍保留v60，正式性能仍引用已完成的六轮原timer对照；E64/shared0、pin=0的孤立GEMM1中值83.5240526316 μs，MoE e2e中值311.7884210526 μs，距离70 μs目标还差13.5240526316 μs。v83已观测四加载地址组中值差0.267842 μs，v3为6.560132 μs；这些已有数据不重复采集。v83孤立GEMM1中值84.3085263158 μs，尚未替换生产。

### v86：没有private memory，但最终ISA的寄存器索引开销使实际指令跨段

编译/export child耗时17.239559820秒，inspection child耗时0.113825973秒，两者returncode=0，均使用118秒TERM/再2秒KILL。最终资源为VGPR=1024、SGPR=76、LDS=327680 B，private bytes及VGPR/SGPR spill计数均为0；独立资源检查通过。原controller的完整`.text`单64 KiB段gate失败，`completed=false`和原AssertionError保持，没有给v86补跑native、目标device精度或性能测试。编译输入preparation可能使用其他GPU kernel，不能将整个流程称为零GPU操作。

独立review连续核算完整`.text`机器码和带symbol annotation的branch。`.text=64384 B`，相对区间`0x1a00–0x11580`，非`s_code_end`指令末端为`0x11388`，跨界部分包含实际指令，并非仅尾部填充。相比v83，`.text`增加6528 B，静态branch从186减到175，WMMA sites均为1200；实际存在64个`v_movrels_b32_e32`，没有scratch指令。额外693个`s_set_vgpr_msb`、85个`s_wait_alu`和多处寄存器搬运与较大的寄存器索引/分配开销相符。静态sites不是动态执行次数，不作为kernel性能数据。不能用源码更短或branch更少推断运行更快。

CDNA5 ISA §3.3.2.3（文档页17）说明，wave32可分配最多1024 VGPR，超过255的寄存器通过VGPR-indexing访问，MODE的DST/SRC0/SRC1/SRC2 MSBits可由`S_SET_VGPR_MSB`更新；表16（本地文本约2837行）说明`S_CODE_END`是shader末端填充、按illegal instruction处理。这些文档解释指令语义，不证明本候选的动态瓶颈或性能收益。

```text
v86 source SHA256=789b6df2eef955efee31ede958fc3cc1cdff3c2aa3df0049072ccc5a2baaac01
v86 ELF SHA256=a3e079775afa78bcbfecafe9d90f76d7b759d41a57f17da33e3a407c443b6ea5
v86 failed archive=TASK/logs/f3_v86_rejected_layout_CPU_20261028_v1_evidence.tar.gz
archive SHA256=13c18dea845a473bf2d785dc06b7e814807c9f41a719bbf629581a7402fe56f5
archive bytes=645575, all manifest members verified
failed review=TASK/logs/f3_v86_rejected_layout_independent_local_review_20261028_v1.json
complete ISA review=TASK/logs/f3_v83_v86_complete_ISA_independent_local_review_20261028_v1.json
```

源码已由controller的`finally`恢复v60，两次原post/settled在`2026-10-28T00:11:05.307939+00:00`/`2026-10-28T00:11:08.344511+00:00`均idle，独立remote review无本轮child残留。本轮没有覆盖旧证据、修改生产选择或改变暂存状态。

### v87：四个相邻row分支加显式Float32 select，先验收真实ELF

新候选从已冻结的v83源码独立派生，仅替换原八叶selector。保留原`selected_row` rmem、opaque row bound、`wmb + wm * M_ROW_STEP < mn_oob`有效row guard和一次`emit_quant_row`调用。四个分支分别覆盖row0/1、2/3、4/5、6/7，以`(fx.Int32(wm) & 1) == 0`在每对原accumulator元素之间做Float32 `select`，不使用动态数组索引，不增加Float32数学或舍入。

独立AST核验原8个operand叶节点及新4个pair的每个operand，完成8704项映射检查，并从既有`inline_zero_acc`/`interleaved_m_rows`准入及`WMMA_M=16`推导合法`wm=0..7`。原mainloop、TDM issue/owner、waits/barriers、激活、amax、scale、FP4转换、LDS/global输出及fallback的函数AST和selector region外bytes保持，没有新增expert/token/seed/data specialization。源码等价不能代替device精度验收。

事前冻结的编译gate继续检查整个`.text`位于一个64 KiB静态地址段，VGPR≤1024、SGPR≤106、LDS≤327680 B，private bytes和两个spill计数均为0。未改写v84/v86旧失败，也不因尾填充调整此次gate。仅通过实际ELF/resource和bounded native检查后，才做E64/shared0、E65/shared0完整原精度、原caller/timer及四加载地址组正常计时与独立counters验收。实际lowering及性能当前未知，生产继续v60。

```text
v87 snapshot=TASK/snapshots/f3_v83_paired_select_selector_v87.py
v87 source SHA256=141e45e8b4f48eac8bb49ced6b601e4bb24cb24749d07412dc2bddcb911428f9
source plan=TASK/logs/f3_v87_paired_select_selector_source_20261028_v1.plan.json
source review=TASK/logs/f3_v87_source_independent_CPU_review_20261028_v1.json
compiler runner=TASK/run_f3_v87_compile_20261028_v1.py
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v86_rejected_CPU_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v83_v86_complete_ISA_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/make_f3_v87_paired_select_selector_local_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v87_paired_select_source_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v87_compile_20261028_v1.py
```

已有label、archive、manifest和review不覆盖，复现另建label。只在a07-3/dsr1_ep4、pin=0下测试，不再测试E65/shared1。每个实际GPU child前仍按6秒轮询并要求全机连续11次idle；child期间不做SFTP、GPU查询或远端并行编译，所有实际compiler/test/profiler child有118秒TERM/再2秒KILL限制。不下载`.out`、cache或完整MoE输出binary。E64/E65共用同一compiled GEMM1、已授权英文commit/push、远端主机pull和pull后最终复测仍待完成。

## 2026-10-28：v87 编译和 bounded native 通过，进入完整 MoE 原精度验收

v87编译/export耗时16.939310111秒，ISA inspection耗时0.113785705秒，两者returncode=0，均由118秒TERM/再2秒KILL保护。原完整`.text`单64 KiB静态段gate和资源gate均通过，source已恢复v60。编译preparation可能执行其他GPU kernel，目标GEMM1尚未dispatch；当前没有v87性能成绩。

最终ELF `.text=58112 B`，相对区间`0x1a00–0xfd00`，非`s_code_end`末端`0xfb50`；比v83增加256 B。metadata VGPR=797、SGPR=76、LDS=327680 B，private segment和两个spill计数全0。完整ISA静态branch从186减为178，WMMA sites均1200；没有MOVREL或scratch。新增64个`s_set_vgpr_msb`以及部分`v_cndmask`/`v_dual_cndmask`，搬运指令减少；静态计数和代码大小不能证明动态性能改善。

原bounded DS/WMMA模型及`WMMA_only` retirement stress均通过，issue=0，CFG iterations=12511/12513，两个CPU child耗时2.028341500/1.922557200秒。派生worker仅修改诊断module名，原transfer AST保持；模型未覆盖完整VALU、VMEM、EXEC skip及tensor completion，不能代替device正确性。

下一步复用此ELF对应的原caller cache，G1不重编译；G2必须核验其完整ELF SHA256保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。使用原完整MoE verifier、原精度gate0.01，在远端比较每例全部22020096 B输出：E64/topk8/shared0、E65/topk9/shared0，各seed0/17。E65/shared0的验证仍不能证明两种expert数共用同一compiled GEMM1；该要求待完成，E65/shared1不再测试。

每个实际GPU child前独立要求全机连续11次idle、间隔6秒、跨度至少60秒，外部占用自动等待；运行期间不做SFTP、GPU查询或远端并行编译。所有实际compiler/test/profiler child有118秒TERM/再2秒KILL限制。完整输出binary只保留在远端，不下载`.out`、cache或完整MoE输出。

```text
v87 CPU archive=TASK/logs/f3_v87_paired_select_selector_CPU_20261028_v1_evidence.tar.gz
archive SHA256=339d9b46f6d8d0ee8cb83ad7c807aadfee93ed03491cf2bd8191834a020adf8f
archive bytes=612265, members=24, all hashes verified
v87 ELF SHA256=c044e4257babacf5d06d5217f6987362f35f4705d0760a5a903260e3dffbfb34
v87 native review=TASK/logs/f3_v87_native_CPU_20261028_v1.review.json
v87 complete ISA review=TASK/logs/f3_v83_v87_complete_ISA_independent_local_review_20261028_v1.json
device runner=TASK/run_f3_v87_device_prepare_20261028_v1.py
```

```bash
python .codex_tmp/hyg_f3_followup_20261024/review_f3_v87_native_CPU_20261028_v1.py
python .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v83_v87_complete_ISA_local_20261028_v1.py
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v87_device_prepare_20261028_v1.py
```

目前生产仍为v60，已复测GEMM1中值83.5240526316 μs，尚未达到70 μs；未新增或挑选性能数据。只有完整原精度、原caller/timer、多加载地址正常计时和独立counters均验收后才考虑生产采用。已授权Git收尾及pull后最终复测仍待完成。

## 2026-10-28：澄清 v80 成绩，v87 完整 MoE 正确性通过并冻结原 timer 对照

问：之前性能最好的不是 v80 么？

答：v80 确实有更快的孤立 GEMM1 成绩。之前进展汇报只列当前保留的 v60 和新候选，遗漏了 v80；保留 reference 不表示它是历史性能最优。v80 重启前原入口三轮为 77.0702 / 77.8060 / 77.6057 μs，中值 **77.6057 μs**。随后重启的原入口三轮中值 **81.2470 μs**，同 boot v60 为 83.4505 μs，改善 2.640487%。当时冻结的三轮全部低于 80 μs 采用条件未通过，该条件与数值正确性通过应分开说明。

更新的当前 boot 下，已封存的原 timer 记录如下；本节没有重跑或替换这些数据：

| 固定顺序中的版本 | 孤立 GEMM1 中值 μs | MoE e2e 中值 μs |
|---|---:|---:|
| 前置 v60 | 83.118789 | 311.501211 |
| v80 | **81.510211** | 318.114895 |
| 后置 v60 | 83.656947 | 312.850684 |

v80 孤立 GEMM1 更快，但完整 MoE e2e 相对前置/后置 v60 回退 2.123165% / 1.682659%，因此后来仍未采用。此前只用早期 80 μs 条件解释当前选择不完整。本轮另有六轮 v60/v83 记录：v60 孤立 GEMM1 / e2e 为 83.524053 / 311.788421 μs，v83 为 84.308526 / 284.667684 μs。v83 孤立 GEMM1 略回退而 e2e 改善，未替换生产；这些记录不能合并成一次同时实测的比较。v80 seed17 约 89–90 μs 的合法随机输入结果保留，不能以 seed0 的历史快值代表所有 random routing，也不能用跨 boot 的数字判断新候选胜出。

问：汇报下进展，然后继续。

答：v87 的编译和完整 MoE device 正确性已结束，并独立核验证据 archive 的全部 39 个成员。仅改 v83 epilogue 内 accumulator selector 的每对 Float32 `select`，保留 mainloop、有效数学、无效行跳过、原 quant、TDM、waits/barriers 和 fallback，没有新增 expert/token/seed/data specialization。v87 `.text=58112 B`，VGPR=797、SGPR=76、LDS=327680 B，private/spills 均为 0，原完整 `.text` 单 64 KiB 段检查和两个 bounded native 模型通过；native 模型不能代替 device 精度。

| 完整 device 用例 | 实际 child 耗时 s | 结果 |
|---|---:|---|
| GEMM2 preparation | 13.784479467 | 原完整 GEMM2 ELF 保持 |
| E64/shared0 seed0 | 8.526603834 | 原精度通过，22020096 B 输出逐字节一致 |
| E64/shared0 seed17 | 8.276274833 | 同上 |
| E65/shared0 seed0 | 20.344408831 | 同上，已有支持路径验证 |
| E65/shared0 seed17 | 8.226270744 | 同上，已有支持路径验证 |

各 child 使用 `timeout --signal=TERM --kill-after=2s 118s`，分别有连续 11 次、6 秒间隔、至少 60 秒的全机 idle 记录。最终 post 无 GPU/KFD owner，短暂 busy=1；settled busy=0、idle=True。原 v60 source 恢复，G1 cache hash/mtime 保持，无本轮 child 残留。正确性过程的 μs 不作为正式性能数据；未下载完整输出 binary、`.out` 或 cache。未测试 E65/shared1。E64/E65 共用同一 compiled GEMM1 尚未实现，E65/shared0 的正确性不能代替该要求。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
TASK/logs/f3_v87_complete_MoE_pin0_20261028_v1_evidence.tar.gz
bytes=715294, members=39
SHA256=6ed2fed4ad94456608d69441663c5092adcd142a87a3ef9a42eae7752ffd2f8f
TASK/logs/f3_v87_complete_MoE_independent_local_review_20261028_v1.json
G1 ELF SHA256=c044e4257babacf5d06d5217f6987362f35f4705d0760a5a903260e3dffbfb34
G2 ELF SHA256=4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be
```

下一步正常性能计划已在执行前冻结：E64/T1536/topk8/shared0、7168/2048、random non-balanced、pin=0、A-preshuffle=1、explicit VGPR partition=0，六轮固定 `v60,v87,v87,v60,v60,v87`，另有一次不计时的 v87 binding preflight。使用原 caller、原 kernel/graph timers、warmup5/iters20、原 precision gate；每个 child 前独立 60 秒 idle 检查，各自 118s/2s 上限。保留全部轮次和自然加载地址，不挑快实例/seed，不修改生产选择。尚无 v87 正式性能成绩；胜过 v60 也不能自动宣称胜过 v80，后续候选选择仍需完整 MoE 和性能参照。

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_E64_v60_v87_original_timer_20261028_v1.py
```

```text
TASK/logs/f3_E64_v60_v87_original_timer_20261028_v1.frozen_plan.json
SHA256=7be4d3e4d582be78ee5e9d9a0965f794c556900133c5e3d2ce8195eb604f4cafa
```

测试仅在 a07-3/dsr1_ep4；分支 `hyg_dsr1_conc6144`、HEAD `e99dbac65a92ed23f87df4d99bd38f655a923a3b`、boot `ff39c3dc-ffb5-4510-a458-0110ee739267`。本轮未改变生产选择、两份暂存源码或 OP test 的 unavailable-time guard。原已授权英文 commit/push、远端主机 pull 和 pull 后最终复测仍待完成，未提前拉取改变实验 HEAD。

## 2026-10-28：按用户最新目标恢复 v80，并仅按孤立 GEMM1 选择优化版本

问：v80 孤立 GEMM1 更快，不要看 MoE e2e 耗时；从优化记录中找到 GEMM1 最快的版本，在它的基础上继续优化。

答：当前 E64/T1536/topk8/shared0、7168/2048、random non-balanced、pin=0 的原入口多轮记录中，应选择 **v80**。它的重启前原入口三轮是77.0702 /77.8060 /77.6057 μs，中值77.6057 μs；随后一个boot中值81.2470 μs；当前boot已封存三轮中值81.510211 μs。当前boot相邻前置/后置v60中值83.118789 /83.656947 μs。不同boot和计时批次仍分开保存，77.6057 μs不能当作今天的新测成绩。

此前以MoE e2e回退或“三轮全部低于80 μs”决定是否采用v80的条件，现由用户明确更新的GEMM1目标替代。后续版本选择以原孤立GEMM1 timer结果为依据，完整MoE e2e仍验证原精度和结果正确性，其耗时不作为选版条件。诊断中v79某地址组76.976474 μs不冒充完整原入口正式多轮成绩；v80合法seed17约89–90 μs的限制仍保留，不挑快seed、加载实例或轮次。

v60/v87六轮实验在第5轮等待GPU空闲期间停止；当时没有本轮GPU child运行。controller的finally逐字节恢复v60，准确终止本轮controller/idle waiter，未终止外部任务。已有四轮保留如下；未补跑最后两轮，也不计算六轮中值或声称实验完成。

| 已完成轮次 | 版本 | GEMM1 μs |
|---:|---|---:|
| 1 | v60 | 83.129684 |
| 2 | v87 | 84.076579 |
| 3 | v87 | 83.732684 |
| 4 | v60 | 83.851000 |

这四轮均保留原有效性记录和原日志。取消后的post/settled中存在外部KFD owner，照实保存，不能改写为空闲。v87此前已完成的完整MoE正确性证据仍保留，不继续以v87作为本轮优化基准。

两端生产kernel现按最新授权恢复已验证v80源码，SHA256=`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。改动前完整备份v60，使用SHA核对和原子替换，保留两份已暂存修改、原test unavailable-time guard及完整index。仅传输任务文件，不执行clean/reset/checkout，不在容器内运行Git。复现环境仍为a07-3/dsr1_ep4、pin=0、A-preshuffle=1、explicit VGPR partition=0；不再测试E65/shared1。

v80基准实际GEMM1 ELF SHA256=`49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8`，GEMM2 ELF=`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。后续候选须保留所有有效数学、无效行跳过、TDM/LDS/WMMA同步及支持范围，经源码/实际ELF和原精度验证后，用相邻v80对照衡量GEMM1收益。本checkpoint尚无新候选性能成绩。E64/E65共用同一compiled GEMM1的历史要求仍未完成，不能用E65/shared0功能验证代替该要求；已授权英文commit/push、远端主机pull及pull后最终复测仍保留为后续收尾任务。

## 2026-10-28：v88 activation 分组无 GEMM1 收益，继续从 v80 调整 clamp/tanh 发射

问：汇报下进展，然后继续。

答：已按照用户最新要求，仅以孤立 GEMM1 耗时选版；完整 MoE 仍必须通过原精度标准。v80/v88 的六轮固定原 timer 对照和独立复核已结束，全部性能有效。v80 中值 **81.405052632 μs**，v88 中值 **81.529473684 μs**；v88 增加 **0.124421053 μs / 0.152841929%**，没有建立收益，不采用。两端生产继续保留 v80。下一候选 v89 保留四值分组，将 clamp 从一次性处理全部 pairs 改为在相应 tanh 分组前处理；256 组源码算式检查和实际编译已通过，尚无 v89 native model、device 正确性或性能成绩。

### 环境和原计时口径

本轮只使用 a07-3/dsr1_ep4，仓库 `/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`，分支/HEAD 为 `hyg_dsr1_conc6144` / `e99dbac65a92ed23f87df4d99bd38f655a923a3b`，boot 为 `ff39c3dc-ffb5-4510-a458-0110ee739267`。目标是 E64/T1536/topk8/shared0、7168/2048、A4W4/SiLU、无 bias、uniform/auto/seed0 的 random non-balanced routing。保持 pin=0、A-preshuffle=1、原 warmup5/iters20 和原 kernel timer。

此次比较没有测量 MoE e2e 性能，原 caller 的完整 MoE 正确性调用继续执行。两阶段原 timed partial、原 eager artifact、source IR、完整 ELF、实际 loaded `.text` 和 caller/cache key 均绑定核验，所有 observation/binding hooks 在原计时前恢复；没有改原 timer 或原迭代数。

### v88 的改动和正确性

v88 从已验证 v80 派生，仅在既有 `interleaved_m_rows and not SUPPORT_EXPLICIT_VGPR_PARTITION` guard 内使用局部 `_batched_silu_swiglu_chunk8`，将 tanh/consumer 的分组从 CHUNK=4 改为8。common helper 默认行为保持原样。显式导入所需的 `as_ir_value`、`fclamp_f32`、`fmin_f32`；没有使用早期缺少 imports 的未上传 draft。

每个值的 clamp、tanh、乘法结合、精度转换与输出顺序保持，N=1..128、SiLU/SwiGLU 共256个符号算式检查通过。原 bounded DS/WMMA model 和 WMMA-only retirement stress 都通过，issues=0，没有放宽模型。实际 caller signatures 和 dispatch ABI 与 v80 一致。

E64/shared0 和 E65/shared0 的 seed0/17 共四项完整 MoE 验证均通过原 logits_diff<0.01 标准，四份 `[1536,7168]` BF16 输出各22,020,096 B，与已验证参考逐字节相同。E64 的 logits_diff 为 `3.6105513278025183e-06` / `3.6202693316254594e-06`。GEMM2 完整 ELF 保持原样，已验证 GEMM1 cache 的 hash/mtime 未变化。这些正确性 child 的时间全部排除性能排名；正确性验证遵守当前 AGENTS.md，不要求 GPU 空闲。

| 实际机器码项目 | v80 | v88 |
|---|---:|---:|
| `.text` B | 97,408 | 97,408 |
| VGPR / SGPR | 799 / 60 | 799 / 60 |
| LDS B | 327,680 | 327,680 |
| private / VGPR spills / SGPR spills | 0 / 0 / 0 | 0 / 0 / 0 |
| 静态 executable instructions | 12,729 | 12,734 |
| 静态 `s_set_vgpr_msb` | 2,034 | 2,032 |
| 静态 `v_nop` | 77 | 84 |

其他静态 opcode 数量与 v80 相同；静态数量不当作动态计算量或可节省的 μs。此版没有缩小 code footprint，不宣称消除了加载地址敏感性。

### 六轮固定原 timer 对照

计划在执行前冻结，顺序为 `v80,v88,v88,v80,v80,v88`，保留全部轮次、输入和自然加载地址，不挑最快一轮或加载实例。每轮前独立要求全机11次、间隔6秒、至少60秒连续空闲；外部任务占用期间仅自动等待，没有杀外部进程。实际 child 全部使用 `timeout --signal=TERM --kill-after=2s 118s`。

| ordinal | 版本 | GEMM1 μs | 自然 load_base group | child 秒 | 性能有效 |
|---:|---|---:|---:|---:|---|
| 1 | v80 | 81.329105263 | 2 | 8.075942 | 是 |
| 2 | v88 | 81.529473684 | 2 | 7.975590 | 是 |
| 3 | v88 | 81.729736842 | 0 | 8.076161 | 是 |
| 4 | v80 | 81.405052632 | 0 | 8.025557 | 是 |
| 5 | v80 | 81.987052632 | 0 | 7.925691 | 是 |
| 6 | v88 | 81.310157895 | 0 | 8.175808 | 是 |

三组预先指定的相邻比较 `(r1,r2)`、`(r4,r3)`、`(r5,r6)` 中，v88 的 GEMM1 耗时降幅为 **-0.246367% / -0.398850% / +0.825612%**。仅第三对更快，全部六轮中值也未建立收益。保留第五轮81.987052632 μs，不将它作为异常值删除；不把第六轮81.310157895 μs冒充最优中值。

自然加载仅覆盖 group0/2，这次不是 group0–3 的完整地址验收。六轮原 kernel 与完整 MoE 的 logits_diff 都为 `3.6105513278025183e-06`。测后 ownership/settled idle 均通过，controller 结束后 v80 源码恢复，没有本轮测试进程残留。已完成的正常比较不再因未覆盖其他 group 而重复；v88 已未见收益，停止该候选后续性能扩展。

之前当前 boot 封存的 v80 三轮中值81.510211 μs，以及其他 boot 的77.6057 /81.2470 μs保留为各自独立批次，不能与本次交替对照混合计算，也不能将不同批次的差异当成源码优化收益。

### v89：分组前 clamp，保持每值算法和输出顺序

v89 使用局部 `_batched_silu_swiglu_clamp_pipeline`，common helper 默认仍保持 CHUNK=4。每个 chunk 先执行它的 gate `fmin_f32` 和 up `fclamp_f32`，随后进入原 tanh/前一组 VALU consumer 调度；entry scheduling boundary、原 per-value 运算和结果列表顺序保留。目标是缩短 clamp 中间值的 live range、让 clamp 的 VALU 与前一组 tanh 的延迟有机会重叠，这只是待实测假设。

源码变换可逆回完整 v80。N=1..128 的 SiLU/SwiGLU 共256个符号算式及 tanh/min/clamp/scheduling-barrier 数量检查均通过。没有更改 mainloop 的 TDM/LDS/WMMA、有效计算、无效行跳过、准入条件和输出布局；只在既有 guard 内选择新 helper。

v89 编译和 inspection child 分别为19.593265969 /0.163906215秒，均成功；目标 GEMM1 没有 dispatch。实际 `.text=97,152 B`，比 v80 少256 B；VGPR799、SGPR60、LDS327,680 B、private/spills0，未降低 peak VGPR。原 native model、完整 MoE 精度及性能尚待验证，不能仅凭源码 live range 或减少256 B推断收益。

### 证据和复现

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
v80 source SHA256=93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312
v80 G1 ELF SHA256=49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8
v88 source SHA256=3ecd502c59ee1eb63bb900972dc3b00793a266d533fec8f2ae75b803cbc8b765
v88 G1 ELF SHA256=f35eb015dfb4204fe6d52ff4455f168de739200f855e1375680704844a0061ee
G2 ELF SHA256=4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be
v88 correctness=TASK/logs/f3_v88_complete_MoE_pin0_20261028_v1.summary.json
v88 normal plan=TASK/logs/f3_E64_v80_v88_original_kernel_timer_20261028_v1.frozen_plan.json
v88 normal archive=TASK/logs/f3_E64_v80_v88_original_kernel_timer_20261028_v1_evidence.tar.gz
archive bytes=6162964
archive SHA256=2d2dedfc82308aa117b4928e702ae86cc445fb657cdbb043d7b678b4347a25ef
independent review=TASK/logs/f3_E64_v80_v88_original_kernel_timer_20261028_v1.independent_local_review.json
v89 source=TASK/snapshots/f3_v80_clamp_pipeline_v89.py
v89 source SHA256=eb64bec2f968f1a1ff7d9bf1150a09d4f7986110ecf384fe65dd1290b9c0ad6b
v89 plan=TASK/logs/f3_v89_clamp_pipeline_source_20261028_v1.plan.json
```

独立 reviewer 核验了109个 archive 成员的全部 hash、六轮原日志、原 timer/partial、完整 ELF/loaded `.text`、原精度和全机空闲证据；没有下载 `.out`、cache 或 MoE 输出 payload。

以下复现控制器在远程主机运行，kernel/compiler child 自动进入 dsr1_ep4；既有 label 是封存记录，复测须重新冻结新 label，不能覆盖或替换旧轮次：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_original_kernel_timer_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_v80_v88_original_kernel_timer_20261028_v1.frozen_plan.json
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_compile_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_v89_clamp_pipeline_source_20261028_v1.plan.json \
  --stem f3_target_v89_clamp_pipeline_pin0_cpu_20261028_v1
```

```powershell
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v80_followup_original_kernel_timer_local_20261028_v1.py --label f3_E64_v80_v88_original_kernel_timer_20261028_v1
```

E65/shared1 不再测试。E65/shared0 的正确性验证仍走已有 t64 fallback；本 checkpoint 没有完成 E64/E65 共用同一 compiled GEMM1 的历史要求，不能误报。已授权英文 commit/push、远端主机 pull 及 pull 后最终复测仍保留为后续收尾任务。

## 2026-10-28：v89 native 与完整 MoE 精度通过，六轮孤立 GEMM1 对照等待全机空闲

问：汇报下进展，然后继续。

答：当前生产仍保留 v80。本次已完成的 v80/v88 原 timer 对照中值分别为81.405052632 /81.529473684 μs，v88 没有收益，不采用。v89 已完成原 bounded native model、WMMA-only retirement stress，以及 E64/shared0、E65/shared0 的 seed0/17 四项完整 MoE 精度验证；六轮 v80/v89 原 timer 对照正在执行前的全机空闲等待，尚无 v89 性能成绩。所有后续选版以孤立 GEMM1 耗时为依据，完整 MoE 结果仍必须正确。

### 已完成的 native 检查

独立 reviewer 核验了 CPU evidence archive 的全部 member hash、实际 ELF/ISA bytes、原 caller signatures 和 GPU dispatch ABI。v89 的两个原模型均通过，未降低模型要求；这些检查不代替 device 数值验证，也不提供 kernel μs。

| 项目 | 结果 |
|---|---|
| source SHA256 | `eb64bec2f968f1a1ff7d9bf1150a09d4f7986110ecf384fe65dd1290b9c0ad6b` |
| G1 ELF SHA256 | `7f6b4da728e825f76041ffc160d3093d0c8cd55bfc657fed9317e0b331c812b6` |
| `.text` / VGPR / SGPR / LDS B | 97,152 /799 /60 /327,680 |
| private / VGPR spills / SGPR spills | 0 /0 /0 |
| original transfer model | issues=0，CFG iterations=27,378 |
| WMMA-only retirement stress | issues=0，CFG iterations=27,378 |
| CPU archive SHA256 | `8f842e82d3c9868ae589e3eab0fbfad611ed656b7a93319da3f9544ec84aaa37` |
| native review SHA256 | `4b5c607d5897e07fe55b69beb39c990d9419c1a838840d814fe175a032050196` |

静态 executable 为12,715条，v80为12,729条；静态 `s_set_vgpr_msb` 为1,964处，v80为2,034处。部分 VALU 配对同时发生变化，不能用静态指令减少或256 B code footprint 变化推算性能收益。

### 完整 MoE device 精度

`f3_v89_complete_MoE_pin0_20261028_v1.summary.json` 已完成，SHA256 为 `4b28034a73ec6700feeab65f12b2ca8205fb0fe2afa989883199a23debe4908f`。四个新运行用例均通过原精度 gate，每份输出为 `[1536,7168]` BF16、22,020,096 B，与已验证参考逐字节一致：

- E64/T1536/topk8/shared0：seed0、seed17。
- E65/T1536/topk9/shared0：seed0、seed17，验证原 t64 fallback。

GEMM2 完整 ELF 保持 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`，已经验收的 GEMM1 cache hash/mtime 保持，device controller 已恢复 v80。正确性允许在已有 GPU 进程期间执行；本轮打印的正确性耗时全部排除性能比较。E65/shared1 未测试；E64/E65 共用同一 compiled GEMM1 的历史要求仍未完成。

### 固定性能计划与继续方向

环境仍是 a07-3/dsr1_ep4，分支 `hyg_dsr1_conc6144`、HEAD `e99dbac65a92ed23f87df4d99bd38f655a923a3b`，boot `ff39c3dc-ffb5-4510-a458-0110ee739267`。E64/T1536/topk8/shared0、7168/2048、uniform/auto/seed0、random non-balanced、pin=0、A-preshuffle=1；原 warmup5/iters20、原 kernel timer、原 partial 和精度检查保持。完整 MoE 使用不计时的正确性调用，不以 MoE e2e 耗时决定采用。

六轮顺序在执行前固定为 `v80,v89,v89,v80,v80,v89`，计划 SHA256 为 `623c41e9d21cf0a75aed4e35962dc7017349b4694cf7884d1f7a2d97a66529c4`。每轮前要求11次、间隔6秒、至少60秒连续全机空闲；外部任务连续占用时继续自动等待，不杀外部进程。每个实际 compiler/test/profiler child 均为118秒 TERM、随后2秒 KILL 上限。不能用重新加载、挑 seed、替换慢轮次或修改计时达到数字。

截至本 checkpoint，第一轮尚未启动实际性能 child。等待期间仅继续本地源码/ISA/历史分析，生产和冻结计划绑定文件保持。旧 K256/b3 在 F3 上已回退，不能将较低 LDS 声明当成新收益；本轮没有创建或测试新的 geometry 候选。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
native review=TASK/logs/f3_target_v89_clamp_pipeline_pin0_cpu_20261028_v1_native.review.json
device summary=TASK/logs/f3_v89_complete_MoE_pin0_20261028_v1.summary.json
normal plan=TASK/logs/f3_E64_v80_v89_original_kernel_timer_20261028_v1.frozen_plan.json
normal host log=TASK/logs/f3_E64_v80_v89_original_kernel_timer_20261028_v1.host.log
normal summary=TASK/logs/f3_E64_v80_v89_original_kernel_timer_20261028_v1.summary.json
```

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_original_kernel_timer_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_v80_v89_original_kernel_timer_20261028_v1.frozen_plan.json
```

已在运行的计划只观察、不重启。正式结果完成后再归档原日志/ELF/loaded `.text`、独立复核全部六轮，并据 GEMM1 中值和三组预设相邻对照决定。此处没有将等待状态写成完成成绩。

## 2026-10-28：保留 v89 中止性能证据，继续验证 v90 六行 WMMA 分组

问：汇报下进展，然后继续。

答：当前生产最优仍是 v80，最近完整有效六轮 v80/v88 对照的孤立 GEMM1 中值为81.405052632 μs；v88 为81.529473684 μs，没有收益，不采用。v89 已通过 E64/shared0、E65/shared0 的 seed0/17 四项完整 MoE 原精度验证，输出与参考逐字节一致，但六轮性能实验在第二轮测后发现外部 KFD owner而中止，不能据此给 v89 排名。继续从 v80 验证只调整既有六行路径 WMMA 分组的 v90，先编译、native 和精度，之后才比较性能。所有选版依据保持为孤立 GEMM1 耗时，MoE e2e 正确性仍须通过。

### v89 中止批次的完整保留

原计划 `f3_E64_v80_v89_original_kernel_timer_20261028_v1` 的顺序为 `v80,v89,v89,v80,v80,v89`，仅运行了前两轮：

| ordinal | 版本 | 日志打印 GEMM1 μs | child 秒 | 性能有效性 |
|---:|---|---:|---:|---|
| 1 | v80 | 81.750789474 | 8.025764 | 有效单轮，不能当成三轮中值 |
| 2 | v89 | 82.043894737 | 8.025907 | 按冻结的测后条件无效，不参与排名 |

两轮 returncode=0，原 kernel 和完整 MoE 精度均通过。第二轮即时 post 没有 owner，3秒后的 settled 检查发现外部 `cpu_checks.py` PID3836624 建立 KFD；之后有外部 pytest PID3837324。仅凭 settled 检查不能证明计时窗口本身受到干扰，仍严格保留冻结有效性规则：第二轮无效，controller 退出 `AssertionError('Retain external ownership/non-idle failure without replacement')`，不补齐本批、不替换第二轮、不与旧轮次混合。没有六轮 medians或配对性能结论。

中止 evidence 已归档并在本地独立复核全部81个成员，archive为2,944,374 B，SHA256=`44b2edc8d3d628df73b8a2f22adc84dedddd963adabfe1e49862f02491ffa0e8`。review明确记录 `performance_data=false`、`v89_not_ranked=true`、`no_pair_or_median_claimed=true`，原 timer/精度规则保持，生产源码已恢复 v80。此前缺少的远端 native reviewer 已上传原文件并核SHA，没有放宽归档检查或重跑GPU测试来修复证据。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
TASK/logs/f3_E64_v80_v89_original_kernel_timer_20261028_v1_interrupted_evidence.tar.gz
TASK/logs/f3_E64_v80_v89_original_kernel_timer_20261028_v1_interrupted_evidence.tar.gz.manifest.json
TASK/logs/f3_E64_v80_v89_original_kernel_timer_20261028_v1.interrupted_independent_local_review.json
```

后续完整独立六轮性能计划采用新增的 external-test precheck，在原11次、间隔6秒、至少60秒全机 idle gate之外，识别尚未建立KFD的已有GPU测试/控制进程。检测到外部占用继续自动等待，不终止外部任务；precheck不能保证未来新任务不会启动，原测后有效性检查继续保留。新controller仅更换pre-idle script路径，未改原timer、child、source CAS/restore或旧冻结计划。新的六轮必须全部新测，不能把本批第一轮拼入新批。

### v90 当前状态和改动范围

v90源码由v80可逆派生，仅对既有 `standard_a_timeout/interleaved_m_rows`、pin0、六行路径且默认MMA hints的 closing WMMA cap从25（原有效22）改成20，普通分组由 `[1,1]` 改为 `[1,2,1]`。其他行覆盖、所有 per-value 数学与舍入、无效行跳过、TDM/LDS/WMMA操作、硬件 waits/barriers、输出和launch helper保持。没有新增 expert/token/seed/data specialization，也未更改准入条件。

实际assignment AST检查24,960项通过，42个 nested arithmetic/load/pipeline/epilogue/launch helper的AST保持；source SHA256=`0eba4b512a88226616583dcf525e86e51caeccefb0417edcf5dcf421489eeac3`。历史v17是v3上所有行路径closing20，v26是v3上六行closing16，本候选并非原样重测它们。

v90 compiler controller已在a07-3/dsr1_ep4启动，截至本checkpoint没有编译结果、native/精度结果或性能成绩。按当前AGENTS.md，compile-only输入准备属于正确性工作，允许在外部GPU占用时执行，所有打印耗时排除性能；目标GEMM1保持compile-only不dispatch。原118秒TERM/随后2秒KILL child上限、实际ELF inspection、source CAS/restore保持。生产源码只由controller在受控编译期间暂时切换，结束必须恢复v80。

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_compile_correctness_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_v90_six_closing20_source_20261028_v1.plan.json \
  --stem f3_target_v90_six_closing20_pin0_cpu_20261028_v1
```

测试范围保持a07-3/dsr1_ep4、E64/T1536/topk8/shared0、7168/2048、random non-balanced、pin=0、A-preshuffle=1。E65/shared1不再测试；E65/shared0仅做既有fallback正确性保护。E64/E65共用同一compiled GEMM1的历史要求仍未完成，不能用fallback通过替代该要求。已授权英文commit/push、远端主机pull和pull后最终复测仍留待优化收尾，当前不pull。

## 2026-10-28：v90 编译与原精度通过，后续测试只保留 E64/topk8

问：`--experts 65 --topk 9` 这个规模请你不要再测了。

答：接受最新约束，后续性能与正确性测试均不再运行 E65/topk9，包括 shared0 和 shared1。主任务保持 E64/T1536/topk8/shared0、model_dim=7168/inter_dim=2048、random non-balanced、pin=0 的孤立 GEMM1。v89、v90 已冻结的性能计划都只包含 E64。此前在这条新指令到达前完成的 E65/shared0 正确性记录继续保留，不能作为新测试或性能成绩；后续候选的正确性controller必须去掉E65用例。

### v90 的已完成验证

v90 compiler与inspection child成功，分别耗时20.044992499 /0.163884230秒，目标GEMM1没有dispatch，所有compile/input preparation时间排除性能。实际G1 ELF SHA256=`06c23e901b582d7492cf6c5beddd8d05e6c044dc8cf31f1382ca66cd14fb52d2`；`.text=97,536 B`，较v80增加128 B；VGPR799、SGPR60、LDS327,680 B，private/VGPR spills/SGPR spills均为0。完整source SHA保持`0eba4b512a88226616583dcf525e86e51caeccefb0417edcf5dcf421489eeac3`。

原bounded native model和WMMA-only retirement stress均通过，issues=0、CFG iterations=27,586。实际原caller signatures、dispatch ABI和保护的静态操作数量通过核验；模型不代替device验证。compile evidence archive SHA256=`bbd47ccb02951080fee2e75a95e5ca5bc0ca1e45bed4c5d939c7c399addbb266`，native review SHA256=`c98c1db3f6e6f85642926e990172e1886465982eebfcf92aad6a93f9a72ae4ce`。

完整MoE正确性controller已结束，原precision gate不变，E64/shared0的seed0/17输出均与既有已验证参考逐字节一致，每份为`[1536,7168]` BF16、22,020,096 B。GEMM2完整ELF保持`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`，已验收G1 cache的SHA256/mtime保持。该controller也在最新禁止指令之前完成了E65/shared0的seed0/17旧fallback保护；之后不再运行该规模。正确性允许在外部负载期间执行，全部打印耗时不用于性能排名。

两端生产仍为v80，远端compiler/device controller均已逐字节恢复。没有把native通过、128 B text增长、或者正确性耗时当作优化收益。

### 实际ISA静态对照与正式性能计划

v80/v90的28个静态K-loop backedge区间已绑定完整ELF与machine bytes。六行路径的7个48-WMMA静态区间，各自比v80增加2条`s_set_vgpr_msb`；32/56/64-WMMA区间的opcode与MSB数量保持。此处的区间包含关系仅用于静态比较，不代表dynamic计算量、live I-cache footprint或可节省的μs。调度改变能否改善DS/WMMA重叠，仍须看原timer实测。

两份新计划分别为：

| 对照 | 六轮固定顺序 | plan SHA256 |
|---|---|---|
| v80/v89 新独立批次 | v80,v89,v89,v80,v80,v89 | `bf912c9a0a877a1e8971199ca1c7de2d84d49f6209e324fa1d94b8f26b1c5d34` |
| v80/v90 | v80,v90,v90,v80,v80,v90 | `6aec3d0cec2c9c921586adc70c3ddb7623cca9cac00ebf915ff865b011802215` |

两份case list均只有E64/topk8/shared0，不含任何E65测试，原timer、warmup5/iters20、precision gate和partial保持。每轮前保留11次、间隔6秒、至少60秒连续全机空闲要求，并增加可识别的外部test/controller进程检查；有占用自动等待，不终止外部任务。性能child期间不执行SFTP、额外GPU查询或并行远端编译。各child保留118秒TERM/随后2秒KILL上限。任何无效/慢/失败轮次完整保留，不选择seed、加载实例或替换轮次；v89旧中止批次不会拼入新批次。

截至本checkpoint，两份新六轮性能计划尚未启动实际child，v90没有性能成绩，v80仍是已确认的生产最优。当前boot最近完整有效v80/v88批次中值为81.405052632 /81.529473684 μs。下一步按顺序运行v89、v90孤立GEMM1对照，再根据实际GEMM1收益选择方向，完整MoE结果继续保持正确。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
TASK/logs/f3_target_v90_six_closing20_pin0_cpu_20261028_v1_native.review.json
TASK/logs/f3_v90_complete_MoE_pin0_20261028_v1.summary.json
TASK/logs/f3_v80_v90_mainloop_static_local_20261028_v1.json
TASK/logs/f3_E64_v80_v89_original_kernel_timer_external_guard_20261028_v2.frozen_plan.json
TASK/logs/f3_E64_v80_v90_original_kernel_timer_external_guard_20261028_v1.frozen_plan.json
```

E64/E65共用同一compiled GEMM1的历史要求仍未完成；保留此限制，不新增E65测试来验收。已授权英文commit/push、主机pull和pull后最终复测留待收尾，当前不pull。

## 2026-10-28：v89 第二批对照因外部 owner 中止，v91 在 native gate 被拒绝

问：汇报下进展，然后继续。

答：当前生产最优仍是v80，最近已完成有效v80/v88批次中值为81.405052632 /81.529473684 μs。v89第二个独立六轮计划运行至第4轮时再次出现外部KFD owner，整批按冻结规则中止，不能给v89排名。v91已经编译，但原bounded native model报告75处`DS overwrite before WMMA retirement`，未进入GPU正确性或性能测试，不采用。继续完成已经通过精度的v90的E64-only性能对照。

### v89 新批次的四轮完整记录

`f3_E64_v80_v89_original_kernel_timer_external_guard_20261028_v2` 原固定顺序为v80,v89,v89,v80,v80,v89，实际只完成前四轮：

| ordinal | 版本 | 原日志 GEMM1 μs | child 秒 | 性能有效 |
|---:|---|---:|---:|---|
| 1 | v80 | 81.876263158 | 8.176308 | 是 |
| 2 | v89 | 81.601736842 | 7.975734 | 是 |
| 3 | v89 | 80.990105263 | 7.975791 | 是 |
| 4 | v80 | 4655.768578947 | 9.327663 | 否 |

第4轮returncode=0、原kernel与完整MoE精度通过，但即时post发现外部KFD owner `160093`，settled时该owner已消失。该轮发生约4.66 ms的明显异常，不能拿它抬高baseline或计算候选收益；仅凭post检查也不能唯一确定竞争或异常耗时的具体硬件原因。controller按原规则退出`AssertionError('Retain external ownership/non-idle failure without replacement')`，没有运行r5/r6，没有medians或正式配对结论。

所有有效/无效轮次保留，不删除4.66 ms数据，不替换失败轮次，不把两个中止批次或旧v80/v88轮次拼在一起。结束时v80源码逐字节恢复，final post/settled没有owners。没有终止外部任务。

本批归档4,647,514 B，包含101个成员，SHA256=`2eb3999bad5f8f36788f3fc4625afb7ec8daa8596e7b8d11bf32f52afe5a930d`；本地独立review全部通过，明确`performance_data=false`、`candidate_not_ranked=true`、`no_pair_or_median_claimed=true`。新增通用中止packer/reviewer只将prefix长度从原2轮拓展为1–5轮；保留原错误类型、连续ordinal、最后一轮无效、之前轮次有效、原timer/精度/ELF/loaded-text绑定及source restore检查。没有修改旧packer、reviewer、归档或冻结文件。

### v91 原指令依赖检查失败，保持生产v80

v91仅改变既有pin0 FP4六行路径每个FRONT/BACK调用内独立WMMA输出的遍历顺序，原每输出数学、K128累加顺序和全部loads/barriers/epilogue保持。8,192项实际源码遍历AST检查通过，41个其他nested helper的AST保持，source SHA256=`76783ca428a7728b0b0fdebdcfc17bdaa2ce82309180eec5f2b348597c18827a`。这些源码检查不能证明生成指令的register retirement安全。

compiler/inspection均成功，耗时19.793455297 /0.163895105秒；目标G1没有dispatch，编译/准备时间不属于性能。G1 ELF SHA256=`e61eb25a762c39883eec010d4bafce78a4980e4057aa548a28999fb240f55c65`；`.text=97,536 B`、VGPR799、SGPR60、LDS327,680 B、private/spills0。compile archive SHA256=`7aeb431d301aca3b24ac6733b0c4b5d61dc37cf3b2966457ce828be653a18ba0`。

原native worker退出1，CFG iterations=27,538，报告75处同类`DS overwrite before WMMA retirement`，涉及low scale register及部分高位input register，例如PC31444的v2/v3、PC31888的v544–547。这是保守模型报告的多个site，不能称为75个已证实的device bug，也不能把它归到生产v80。原model/transfer要求未改变；原model已经失败，未继续WMMA-only worker，没有生成“通过”的native review，也未运行v91完整MoE或性能。

实际ISA静态对照的六行7个48-WMMA backedge区间，`s_set_vgpr_msb`数量仍为41/39，未减少；区间静态指令反而增加1或2。32/56/64-WMMA区间计数保持。此处没有将静态数量当成dynamic工作、live cache footprint或μs。该候选当前未建立寻址指令收益，同时未通过原native gate，停止此候选后续device/performance。若重新调整其retirement，将作为独立候选另行验证。

### 严格保留最新E64-only测试范围

新增`run_f3_v80_followup_device_E64_only_20261028_v1.py`及对应freezer已生成、上传并核SHA。实际source AST中的两个expert loop均只有E64/topk8，seed仍为0/17，原precision gate、逐字节比较、timeout和source CAS/restore保持。compile-only的原stage preparation也已确认仅使用`--experts 64 --tokens 1536 --topk 8`。后续不运行任何E65/topk9正确性或性能测试；以前完成的记录原样保留。

v90已完成的精度证据继续有效，新六轮performance plan仍为`f3_E64_v80_v90_original_kernel_timer_external_guard_20261028_v1`，case list仅E64/topk8/shared0。截至本checkpoint该performance plan尚未启动实际child。接下来保留pin=0、A-preshuffle=1、原timer和11次/6秒/至少60秒连续全机idle gate，自动等待外部占用，不与外部GPU工作竞争；有新任务在测后触发无效仍按原规则完整保留。

```text
TASK=.codex_tmp/hyg_f3_followup_20261024
TASK/logs/f3_E64_v80_v89_original_kernel_timer_external_guard_20261028_v2_interrupted_evidence.tar.gz
TASK/logs/f3_E64_v80_v89_original_kernel_timer_external_guard_20261028_v2.interrupted_independent_local_review.json
TASK/logs/f3_target_v91_six_column_order_pin0_cpu_20261028_v1_native_original.json
TASK/logs/f3_v91_native_model_rejection_local_20261028_v1.json
TASK/logs/f3_v80_v91_mainloop_static_local_20261028_v1.json
```

没有新增生产kernel准入specialization，未更改两份已有staged edits及test guard。E64/E65共用compiled G1的历史要求仍未完成；已授权commit/push、主机pull和pull后最终复测保留为收尾任务，当前不pull。

## 2026-10-28：严格停测 E65/topk9，启动 v90 对照并准备 v92

问：`--experts 65 --topk 9` 这个规模请你不要再测了。

答：后续不运行任何 E65/topk9 的正确性、性能或 profiler 测试，无论 shared0 还是 shared1。当前继续优化并验证 E64/T1536/topk8/shared0、model_dim=7168/inter_dim=2048、random non-balanced 的孤立 GEMM1。保持 pin=0、A-preshuffle=1、原 timer、原精度阈值和完整 MoE 正确性；MoE e2e 耗时不作为选版条件。以前完成的 E65 记录保留，未重测或篡改。

本地、a07-3 主机的分支和 HEAD 重新核实为 `hyg_dsr1_conc6144` / `e99dbac65a92ed23f87df4d99bd38f655a923a3b`；boot 为 `ff39c3dc-ffb5-4510-a458-0110ee739267`。两端生产仍为 v80，kernel source SHA256=`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。未改变 index、其他用户文件或生产选择，未执行 pull、clean、reset。

### v90 的原 timer 六轮对照已启动

已先完成远端上一节文档的独立 CAS 追加，保留远端原问答前缀 735879 B。随后启动已冻结的 `f3_E64_v80_v90_original_kernel_timer_external_guard_20261028_v1`，plan SHA256=`6aec3d0cec2c9c921586adc70c3ddb7623cca9cac00ebf915ff865b011802215`；实际 case 只有 E64/topk8/shared0，顺序为 v80,v90,v90,v80,v80,v90。

全机连续 11 次、间隔 6 秒、跨度至少 60 秒的 idle gate 发现外部 benchmark，自动等待其退出，没有启动我方计时 child 与其竞争，也没有终止外部进程。性能 child 期间不做 SFTP、额外 GPU 查询或并行远端编译。每个实际 child 仍由 118 秒 TERM / 随后 2 秒 KILL 保护。

本 checkpoint 时第 1 轮 v80 已完成，原 GEMM1 timer 为 81.61336842105261 μs，原 kernel 和完整 MoE 精度通过，post/settled 所有权检查通过。剩余轮次尚未完成，不计算中值、不宣称 v90 收益；若之后出现外部 owner 或失败，仍保留整个批次，不替换或拼接轮次。

### v92 只完成本地源代码与逐值运算检查

从已确认的 v80 派生 `_batched_silu_swiglu_chunk2`，仅将既有 `interleaved_m_rows and not SUPPORT_EXPLICIT_VGPR_PARTITION` 路径中的 tanh 调度分组由 CHUNK=4 改为 2。保留所有 clamp、tanh、乘加运算的逐值图和返回顺序；其他路径继续使用原 helper。公共 `gemm_common_gfx1250.py`、mainloop、TDM、硬件 waits/barriers、量化、输出和 kernel 准入保持。

1–128 个 activation pair，分别验证 silu、swiglu，共 256 项实际 helper 的 symbolic 运算图一致检查通过；每值 tanh 数量相同，scheduler barrier 数按 CHUNK=2 改变。新 helper 的 AST 还原名字、docstring 和 CHUNK 后与原 helper 完全一致，完整候选源码可逐字节还原为 v80。以上只证明源码层面的运算一致，不能代替生成指令的 native safety、device 精度或性能验收。

```text
generator=TASK/derive_f3_v92_tanh_chunk2_local_20261028_v1.py
generator SHA256=611d7aa7024d6a4ee180a48beb7b30ad37f87762a688e754f1518177ed4db2da
snapshot=TASK/snapshots/f3_v80_tanh_chunk2_v92.py
source SHA256=7be73ef6bf73759554dd500aac31e6cd468f3ba6c52308f0ba7ff3f1dce184d4
source plan=TASK/logs/f3_v92_tanh_chunk2_source_20261028_v1.plan.json
plan SHA256=fc366b28f81507d17dfbba9c5a95a8de61b4d8d89ce7823df52f072d18e2f1e7
```

截至本 checkpoint，v92 尚未上传、编译或 dispatch，没有 ELF、native review、GPU 精度或性能成绩。待 v90 controller 结束并恢复源码后，再顺序进行 v92 的编译、未改的 bounded native model 和 WMMA-only stress、E64 seed0/17 完整 MoE 原精度与全部 22020096 B 输出比较。

### 结果归档与复核同样改为 E64-only

新增 `pack_f3_v80_followup_original_kernel_timer_E64_only_20261028_v1.py` 和 `review_f3_v80_followup_original_kernel_timer_E64_only_local_20261028_v1.py`，保留旧工具、旧计划和归档。新 wrapper 只将以前的 E65 device 验收要求改为“E65/topk9 已禁用且没有运行”，并核验实际 E64-only device controller 的两个 expert loop 均只包含 (64,8)。原 timer、固定六轮、不替换轮次、source/ELF/loaded-text 绑定、原 precision gate0.01、seed0/17 输出逐字节比较和 child timeout 检查保持。wrapper 的每处替换可逆还原为已绑定 SHA 的原工具；本地 `--help` 通过，没有执行 workload。

v80 仍为生产最优，最近完整有效 v80/v88 对照中值 81.405052632 /81.529473684 μs 保留为原独立批次。v89 两个中止批次继续不能排名，v91 native 拒绝继续不做 dispatch。E64/E65 共用同一 compiled GEMM1 的历史要求仍未完成，不能误报；已授权英文 commit/push、主机 pull 及 pull 后 E64 最终复测仍待收尾，当前不 pull。

## 2026-10-28：v90 整批中止，v92 完成 E64 精度验收，后续性能统一 LOG0

问：测性能时，要设置 `AITER_LOG_MORE=0`；汇报下进展，然后继续。

答：所有后续性能 child 显式设置 `AITER_LOG_MORE=0`，在导入 AITER 前检查实际环境，并在原 timer 执行完后再次核验。只测试 E64/T1536/topk8/shared0、model_dim=7168/inter_dim=2048、random non-balanced；保持 pin=0、A-preshuffle=1，完整 MoE 的原精度与正确性仍是硬性条件，仅按孤立 GEMM1 耗时选版。E65/topk9 的任何正确性、性能和 profiler 测试均已禁用。以前的 LOG1 记录保留，不能与新的 LOG0 批次合并。

### v90：保留不完整批次，不用于排名

冻结批次 `f3_E64_v80_v90_original_kernel_timer_external_guard_20261028_v1` 原计划顺序为 v80,v90,v90,v80,v80,v90，实际完成五轮。原打印 GEMM1 耗时依次为 81.61336842105261、81.98457894736839、82.11757894736844、80.92636842105269、80.87568421052632 μs。上述运行发生在最新 LOG0 指令之前，实际为 LOG1；前四轮通过性能有效性检查，第五轮未通过。

第五轮 post 的 `gpu_users.sh` 返回 PID 288750、289232，均显示 `<已退出>`；同次 `/sys/class/kfd/kfd/proc` owners 为空。settled 和 controller 最终检查恢复空闲。这不能证明是已确认的外部 KFD owner，也不能确认计时窗口内存在 GPU 竞争，来源尚未查明。controller 按既有冻结规则中止，报 `AssertionError('Retain external ownership/non-idle failure without replacement')`。所有 child returncode=0，原 kernel 和完整 MoE 精度通过；生产源码逐字节恢复 v80。

不补第六轮、不替换第五轮、不与其他批次拼接，也不报告 v90 中值或收益。111 个归档成员已独立验 SHA：

```text
archive=TASK/logs/f3_E64_v80_v90_original_kernel_timer_external_guard_20261028_v1_interrupted_evidence.tar.gz
archive bytes=5505877
archive SHA256=d98dd1dded6b70bad2c949f2e05f1a196746b1da0d68de88037a5dbe703923f6
review=TASK/logs/f3_E64_v80_v90_original_kernel_timer_external_guard_20261028_v1.interrupted_independent_local_review.json
review SHA256=4850e983ba111f9ed031fca133343a29bbc39990fed1e2e663aea6da95201536
```

### v92：CHUNK=2 的源码、生成指令与完整 MoE 精度通过

仅在既有 `interleaved_m_rows and not SUPPORT_EXPLICIT_VGPR_PARTITION` 路径用 `_batched_silu_swiglu_chunk2`，将 tanh 调度分组由 4 改为 2。其他路径仍用原 helper，公共 helper 文件保持。256 项逐值 symbolic 运算图检查通过，保留原 tanh、数学与输出顺序；完整候选源码可逐字节还原为 v80。

compile-only 实际准备规模为 E64/topk8，没有 dispatch；生成 G1 ELF `.text=97408 B`、VGPR799、SGPR60、LDS327680 B，private/VGPR spills/SGPR spills 均为0。原 native model 和 WMMA-only retirement stress 都为 issues=0，CFG iterations=27598。以上不是性能数据，也不能单独代替 device 验证。

E64 seed0/17 完整 MoE device 验证已完成，每组全部 22020096 B 输出与原基准逐字节一致，`differing_bytes=0`；原 logits_diff 分别为 3.6105513278025183e-06 / 3.6202693316254594e-06，原阈值仍为0.01。G1 已验 cache SHA/mtime 保持一致，G2 full ELF 与基准一致；三个 child 均在120秒以内结束。correctness controller 原 setup 为 LOG1，全部耗时排除性能，未用其耗时评价候选。finally 恢复生产 v80，settled 无残留且空闲。

```text
v92 source SHA256=7be73ef6bf73759554dd500aac31e6cd468f3ba6c52308f0ba7ff3f1dce184d4
v92 G1 ELF SHA256=bfe2e50c08441e9980a0709fa5a630ff190edaed7cbf984d360927afe8959d42
v92 native review SHA256=44bcce19d64491838b1544586df1104307dcc9e8d6d415945486fa6f54fc322f
device summary=TASK/logs/f3_v92_complete_MoE_pin0_20261028_v1.summary.json
device summary SHA256=6a49d0bbe5e206c7ba78948e6103c13db835c36e615925c312b3cce331064e2a
G2 ELF SHA256=4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be
```

### LOG0 下重新比较 v80/v92，保持原 timer

新增 LOG0 checker、driver、freezer、binding preflight、packer 和 independent reviewer，历史冻结文件不修改。checker 在导入 AITER 前要求实际 `os.environ['AITER_LOG_MORE']=='0'`，输出可核验的 `F3_GEMM1_LOG0_ENV` 标记；原 checker/timer/partial 对象、seed0、warmup5/iters20、原精度、实际 ELF/loaded-text/partial binding 保持。binding preflight 对两版做 E64 正确性与实际 JIT key/ELF 检查，不做性能计时。

新六轮固定顺序仍为 v80,v92,v92,v80,v80,v92，每轮独立通过全机连续11次、间隔6秒、至少60秒的空闲检查；性能 child 期间没有 SFTP、额外 GPU 查询或并行编译。每个实际 child 仍为 `timeout --signal=TERM --kill-after=2s 118s`，中止整批保留且不替换。新计划已冻结，尚未执行 binding preflight 或计时：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_LOG0_binding_preflight_E64_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_log0_v80_v92_original_kernel_timer_20261028_v1.frozen_plan.json
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_original_kernel_timer_E64_log0_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_log0_v80_v92_original_kernel_timer_20261028_v1.frozen_plan.json
```

plan SHA256=`9d6eac4f44eb00bd75e5fabc06896b7d182356bf8c772e8919b2866575989612`。controller 在 a07-3 主机运行，实际 kernel child 仅在 dsr1_ep4 容器运行；Git 只在主机执行。当前两端仍为 `hyg_dsr1_conc6144` / `e99dbac65a92ed23f87df4d99bd38f655a923a3b`，boot=`ff39c3dc-ffb5-4510-a458-0110ee739267`。

当前生产最优仍为 v80；最近完整 LOG1 v80/v88 批次中值 81.405052632 /81.529473684 μs，不能作为新的 LOG0 基准。v92 尚无性能成绩，v91 native 拒绝后没有 dispatch。历史 E64/E65 共用 compiled GEMM1 的要求仍未实现，不误报完成。已授权英文 commit/push、远端主机 pull 及 pull 后 E64 最终复测仍待收尾，当前尚未执行。


## 2026-10-28：v92 LOG0 批次中止，v93 完整 E64 精度通过

问：汇报下进展，然后继续。

答：本 checkpoint 的生产版本仍为 v80。最近 LOG0 批次中有效的 v80 单轮 GEMM1 为81.99963157894742 μs；v92 未取得完整有效批次，v93 尚无性能结论。下文 CPU/native、device 精度及 binding preflight 都不是新的性能数据。

后续只测试 E64/T1536/topk8/shared0、model_dim=7168/inter_dim=2048、random non-balanced，pin=0、A-preshuffle=1；实际性能 child 显式设置 AITER_LOG_MORE=0。E65/topk9 的 shared0/shared1 正确性、性能及 profiler 均禁用。完整 MoE 保留原精度要求；只按孤立 GEMM1 选版，e2e 耗时不参与采用判断。旧 LOG1、不同 boot 及中止批次保持独立。

### v92 的 LOG0 原 timer 对照未完成

冻结计划 f3_E64_log0_v80_v92_original_kernel_timer_20261028_v1 原顺序为 v80,v92,v92,v80,v80,v92，实际只完成两轮：

| ordinal | version | 原 timer 打印 GEMM1 μs | 性能有效 |
|---:|---|---:|---|
| 1 | v80 | 81.99963157894742 | 是 |
| 2 | v92 | 82.69121052631583 | 否 |

第二轮 raw post 的 gpu_users.sh 返回 PID671251、<已退出>、0K、0 queues；同次 KFD owners 为空，busy=3、VRAM=176582656 B。settled 的 users/owners 为空、busy=0。按当时冻结的规则，这一轮未通过，controller 报 AssertionError('Retain external ownership/non-idle failure without replacement') 并中止；源码恢复 v80。该非原子快照不能证明 PID 属于外部任务，也不能确认计时窗口受到 GPU 竞争污染。旧批次没有真实 host/container PID 身份记录，故不追溯改有效性、不补轮次、不计算中值或排名。

98项成员的中止归档已下载，独立核验每个成员 SHA，并确认实际环境标记均为 LOG0：

~~~text
TASK/logs/f3_E64_log0_v80_v92_original_kernel_timer_20261028_v1_interrupted_evidence.tar.gz
bytes=4644488
SHA256=7bc76659c90fca4fc7ca6ed13d674a6d093820422ae5483af788c05cc7388b85
TASK/logs/f3_E64_log0_v80_v92_original_kernel_timer_20261028_v1.interrupted_independent_local_review.json
TASK/logs/f3_E64_log0_v80_v92_original_kernel_timer_20261028_v1.LOG0_environment_review.json
~~~

### v93 提前计算原 amax tree 第一层，数学与完整 MoE 精度保持

从 v80 派生 _batched_silu_swiglu_amax_level1：保留 CHUNK=4，在三个原 activation 值全部生成后，提前计算原 ternary tree 第一层的 abs/maxnumf。第一层分组及顺序保持，后续 peer reduction、FLT_MAX、exponent、scale 和 conversion 原代码保持。只在已有 interleaved_m_rows and not SUPPORT_EXPLICIT_VGPR_PARTITION and not is_situv2 路径使用；其他 activation/scale 路径保留原调用。公共 helper 文件、mainloop、TDM、wait/barrier、输出布局及 kernel 准入保持，没有增加 expert/token/data specialization。

256项逐值 activation 运算图及第一层分组源码检查通过。compiler 约19.54秒完成；ELF 的 .text=97536 B（v80=97408 B），VGPR799/SGPR60/LDS327680 B，private/VGPR spill/SGPR spill 均为0。原 native model 与 WMMA-only retirement stress 均 issues=0、CFG iterations=27663；受保护静态运算数量与 v80 相同。这些是源码和生成指令的验收证据，不是性能提升证据。

~~~text
generator=TASK/derive_f3_v93_activation_amax_level1_local_20261028_v1.py
generator SHA256=ea1e6e574afe355cb63227bf23a939dd0dc70eefec65b4a83c1b2d100c6ef8cf
snapshot=TASK/snapshots/f3_v80_activation_amax_level1_v93.py
source SHA256=88d03f06c1a98dbbf2ec49d99aca0cefb97787dbbe0a80ae27649661f1b17c63
G1 ELF SHA256=57b1b4152c12ae5196561da1f3c7050f12b4d6afb11fda86ff2fd404f3b20f33
native review SHA256=a12e885210d7899acda938d8cfe651f04af0048ec41792e0dc550ff4642cbc48
CPU archive=TASK/logs/f3_target_v93_activation_amax_level1_pin0_cpu_20261028_v1_evidence.tar.gz
CPU archive bytes=989239
CPU archive SHA256=844de19c206f54c8bed88dc8930f229d412c174267234667fd03d8dc9b75fac0
~~~

E64 完整 MoE device 验收通过，原 precision gate=0.01：

| seed | logits_diff | 比较输出字节 | differing_bytes |
|---:|---:|---:|---:|
| 0 | 3.6105513278025183e-06 | 22020096 | 0 |
| 17 | 3.6202693316254594e-06 | 22020096 | 0 |

G2 full ELF 仍为4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be，G1 cache SHA/mtime 保持。三个 child 均 returncode=0，elapsed 分别约13.685、8.376、8.126秒。finally 恢复 v80，settled 空闲且无残留。device setup 原为 LOG1，全部打印耗时排除性能；实际性能使用 LOG0。

~~~bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_device_E64_only_20261028_v1.py \
  --cpu-stem f3_target_v93_activation_amax_level1_pin0_cpu_20261028_v1 \
  --stem f3_v93_complete_MoE_pin0_20261028_v1
~~~

device summary=TASK/logs/f3_v93_complete_MoE_pin0_20261028_v1.summary.json，SHA256=4a8eca66fee6dd8fa3a5a218e319bdf28f623a5b5a07c8133b570b02be746ac09。七项 device/PID CPU 记录已下载，使用远端生成的 JSON manifest 自动核验 bytes 和 SHA，避免终端折行影响手工抄录。

### 真实 host PID 验证已通过，旧中止批次及工具保持

已只读检查主机 gpu_users.sh：先枚举 KFD/proc，随后调用 ps，再读 queues/VRAM，进程可能在这些读取之间退出而显示 <已退出>。这证明快照不是原子的，不能单凭该行判断进程来源。dsr1_ep4 不是 host PID namespace，不能将 os.getpid() 当成 host PID。

新增 f3_child_host_PID_identity_20261028_v1.py：child 在 AITER import 前原子写 container PID、start_ticks 和完整 argv，等待 host ack；主机用 docker top dsr1_ep4 -eo pid,ppid,args 精确匹配 argv，并比较 /proc/<hostPID>/stat 的 start_ticks，再原子写 ack。只读取 CPU 进程信息，均发生在 AITER import/timer 前，没有 GPU 查询或新增 timer。

一次真实 CPU-only Docker child 的 smoke 通过，host PID=884714、container PID=323748，elapsed 约0.12964秒；结束后真实 host PID 已不存在。另有6项明确标为 synthetic 的 classifier 边界检查通过，这些不是 GPU 观察或性能数据。

~~~text
helper SHA256=5f9b4fa51d5118fe4060a94fc655f1c7f799fad13a7ff395efc5961dba1111c3
TASK/logs/f3_host_PID_handshake_CPU_only_20261028_v1.summary.json
TASK/logs/f3_host_PID_handshake_CPU_only_20261028_v1.child_PID_identity.json
TASK/logs/f3_host_PID_handshake_CPU_only_20261028_v1.host_PID_ack.json
~~~

新的冻结规则保留全部 raw post。仅当 raw 的唯一 PID 精确匹配本轮已验证 host PID、显示 <已退出>、该进程已不存在、KFD owners 为空、所有 GPU VRAM 空闲且 settled 空闲时，才能识别为已退出的我方 child。未知 PID、活跃进程、任意 KFD owner、活跃 VRAM 或非空闲 settled 仍拒绝；不杀外部进程。原60秒 pre gate、固定六轮、原 timer、warmup5/iters20、seed0、ELF/loaded-text/partial 绑定和原精度保持。旧 frozen 工具及中止归档逐字节保持，不追溯修改。

### v80/v93 新 LOG0/PID 六轮已冻结，性能待完成

新计划=TASK/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v1.frozen_plan.json，SHA256=626b06f75eb9eaeb8d963243831c6645a04e8bffd610a792658289a6ecc1b432，固定顺序为 v80,v93,v93,v80,v80,v93。两端40项 bound 文件 SHA 一致。LOG0 binding preflight 已通过：两版原 kernel 与完整 MoE 精度、实际 JIT key/ELF 绑定均通过；两个 child 约7.926/8.076秒，未运行性能 timer，finally 恢复 v80。

~~~bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_LOG0_binding_preflight_E64_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v1.frozen_plan.json
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_original_kernel_timer_E64_LOG0_PID_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v1.frozen_plan.json
~~~

截至主机 UTC 2026-10-28T06:05:39，性能 controller 仍在第一轮空闲门禁；门禁观察到预先存在的 E96/T16384/topk6 测试，因此尚未 dispatch 我方性能 child，正每6秒自动复检。性能 child 期间禁止 SFTP、额外 GPU 查询或并行远端编译；每个实际 child 保留 timeout --signal=TERM --kill-after=2s 118s。只有完整六轮全部有效后才给出中值及采用判断；中止整批保留、不补轮次。

生产仍为 v80，source SHA256=93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312。历史 E64/E65 共用同一 compiled GEMM1 的要求尚未完成，不误报；已授权英文 commit/push、远端主机 pull、pull 后 E64 最终复测仍待优化收尾，当前没有执行这些 Git 写操作。

## 2026-10-28：等待 GPU 空闲期间完成 v80/v93 的实际 ISA 对比

问：汇报下进展，然后继续。

答：a07-3 持续有其他 grouped GEMM 测试，v80/v93 的 LOG0/PID 六轮 controller 仍在第一轮空闲门禁，没有 dispatch 我方性能 child。继续每6秒自动复检，不使用占用期间的数据作为性能。v93 完整 E64 MoE 原精度已通过，生产源码仍为 v80，没有新性能结论。

完成了本地 CPU-only 的实际 ELF/ISA 比较，两版 source IR bytes、ELF bytes、disassembly bytes 均核验，原 caller 参数签名及 dispatch ABI 相同：

| 静态项目 | v80 | v93 |
|---|---:|---:|
| .text B | 97408 | 97536 |
| s_set_vgpr_msb | 2034 | 2070 |
| v_nop | 77 | 65 |
| v_dual_max_num_f32 | 87 | 89 |
| v_dual_min_num_f32 | 72 | 70 |
| v_dual_mul_f32 | 9 | 10 |
| v_max_num_f32_e32 | 124 | 119 |
| v_min_num_f32_e32 | 129 | 135 |
| v_mul_f32_e32 | 46 | 43 |

28个可对应的内部 backward-edge 区间，opcode 顺序、静态指令数量及 WMMA 数量保持一致，但实际寄存器操作数不同；后14个区间的 target PC 整体移动0x70。另一个包含整个 task body 的外层 backward-edge 区间，其 opcode 顺序发生变化。这里比较的是实际机器码的静态区间，不能视为 ATT 中观察到的动态热点、live I-cache residency 或加载地址问题已经消除。

v93 的 static s_set_vgpr_msb 增加36、v_nop 减少12，不直接换算为耗时或额外36次 pipeline stall。已查 Shader Programming Guide rev65 §3.3.2.3 “Waves with up to 1024 VGPRs”：S_SET_VGPR_MSB 设置后续指令的 VGPR 地址高位；该指令不产生 S_SETREG_B32 MODE 的额外 delay。这不表示指令完全没有 issue/fetch 成本，也不能证明此处的寄存器布局变化会产生收益或回退；仍需完整性能对照。

~~~text
analyzer=TASK/analyze_f3_v80_v93_static_activation_amax_local_20261028_v1.py
analyzer SHA256=e52355623ea72c187c7fab7ec0eeb5d0c4b7e2db3e63149c253c8d4b18f6a396
record=TASK/logs/f3_v80_v93_static_activation_amax_local_20261028_v1.json
record SHA256=68f7c434fd429ccb8e8ca063c49c524f685e1921b5187d47df7bf9ddfd1a6d99
performance_data=false
GPU_dispatch=false
~~~

截至主机 UTC 2026-10-28T06:16:09，六轮仍未开始，pre gate 观察到其他 GPU/KFD/test 进程，VRAM 约16.8GB。上述分析未修改 kernel、timer、冻结工具或计划，也未运行 E65/topk9。待连续空闲门禁通过后，按原固定六轮继续，不补旧中止批次、不挑地址/seed/轮次。

## 2026-10-28：v80/v93 LOG0/PID 第一批中止，保留原有效性并独立复测

问：汇报下进展，然后继续。

答：生产仍为 v80。v93 已通过完整 E64 MoE 原精度与逐字节输出对比，但第一批性能只完成两轮，第二轮后的 settled 检查发现独立外部 KFD owner，按冻结规则中止。没有完整中值、版本排名或采用结论。继续只测 E64/T1536/topk8/shared0，pin=0、A-preshuffle=1、AITER_LOG_MORE=0，以原孤立 GEMM1 timer 选版，完整 MoE 保留原精度。未运行 E65/topk9。

| ordinal | version | 原 timer 打印 GEMM1 μs | 性能有效 |
|---:|---|---:|---|
| 1 | v80 | 81.57184210526312 | 是，单轮 |
| 2 | v93 | 81.86263157894736 | 否 |

第一轮真实 host PID=1146001、container PID=329268；第二轮 host PID=1160756、container PID=329616。两个真实 host PID 均已退出，身份在 AITER import/timer 前经精确 argv 与 /proc start_ticks 验证。

第二轮即时 post（UTC 2026-10-28T06:18:32.750259+00:00）users/owners 为空；3秒后的 settled（UTC 2026-10-28T06:18:36.009790+00:00）出现外部 KFD owner PID1162840。此前只读进程检查识别其为 yadai 容器中的 python3 op_tests/test_gemm_a4w4.py -mnk 16384,16384,16384，不能将它当作刚退出的我方 child。post 检查本身不能证明计时窗口已被外部任务影响，也不能追溯认可第二轮。controller 保留 AssertionError('Retain external ownership/non-idle failure without replacement')，finally 恢复 v80；未杀外部进程、未补轮次、未拼接旧数据。

123项成员的归档、LOG0 环境与真实 PID 身份已在本地独立核验：

~~~text
archive=TASK/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v1_interrupted_evidence.tar.gz
bytes=4692533
SHA256=c606951d33267da80f3f9280cef06497c2da8b63b6d2640e79a839d2b7ba1186
PID review SHA256=ab40425a5ec55fd450b8932c3f34394269835914addb495eac3b969a3fc21e1e
candidate_not_ranked=true
no_pair_or_median_claimed=true
~~~

已有 LOG0 preflight 固定输入的只读 routing 统计为 experts=64、routes=12288、每 expert 有效行数 min=168/max=221，实际覆盖6-rep的35个 expert、7-rep的29个 expert。此统计只用于确定后续静态分析重点，没有选 seed、改输入或作为性能计时。

新的六轮计划为 TASK/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v2.frozen_plan.json，SHA256=2de4bb2a72eaed94d1035cb9ab9ef7e5793f81cd7d55156622d6c5ab61edec6e。固定顺序仍为 v80,v93,v93,v80,v80,v93；这是全新独立批次，第一批两轮不参与其统计。继续保留实际 JIT key/ELF、原 partial binding、原 timer/warmup5/iters20、seed0及精度标准。每个实际 child 使用 timeout --signal=TERM --kill-after=2s 118s。每轮前等待全机连续11次、间隔6秒空闲；性能 child 期间不并行执行 SFTP、额外 GPU 查询或远端编译。

~~~bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_LOG0_binding_preflight_E64_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v2.frozen_plan.json
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_original_kernel_timer_E64_LOG0_PID_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v2.frozen_plan.json
~~~

本地及远端三份文档分别追加，保留两端各自的原历史；没有整文件覆盖。当前 staged 修改保留，没有执行 Git commit/push/pull。

## 2026-10-28：v80/v93 第二批在第四轮测后遇到外部 owner，继续独立复测

问：汇报下进展，然后继续。

答：生产仍为 v80。v93 完整 E64 MoE 原精度及逐字节输出对比已经通过；第二批性能完成前三轮有效读数，但第四轮即时 post 出现独立外部 KFD owner，按原冻结规则中止。没有六轮中值或采用结论，不将旧批次拼入后续统计。

| ordinal | version | 原 timer 打印 GEMM1 μs | 性能有效 |
|---:|---|---:|---|
| 1 | v80 | 82.59189473684211 | 是，单轮 |
| 2 | v93 | 81.30147368421050 | 是，单轮 |
| 3 | v93 | 82.08584210526315 | 是，单轮 |
| 4 | v80 | 81.51242105263155 | 否 |

第四轮我方真实 host PID1858546 已退出。即时 post（UTC 2026-10-28T06:48:08.213153+00:00）users/owners 包含外部 PID1859585，容器为 zan_att，命令为 python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 --experts 1 --tokens 16384 --topk 1 --model-dim 16384 --inter-dim 8192 --act silu --no-bias --iters 64 --no-check-aot-cache --warmup 2 --iters 101。该快照 busy=54，VRAM=3882659840 B。3秒后的 settled owners 已为空、busy=0；即时 post 的外部 owner 仍令第四轮无效。仅凭测后快照不能确定外部任务是否与原 timer 的窗口重叠，也不能追溯认可本轮。

四个实际 child 均 returncode=0，原 kernel 与完整 MoE 精度通过，真实 host PID 均已退出，source finally 恢复 v80。controller 原报 AssertionError('Retain external ownership/non-idle failure without replacement') 保留；未杀外部任务、未运行第五/六轮、未补接第一批或第二批轮次。

第二批145项成员已归档下载，原中止规则、实际 LOG0 环境、完整 ELF/loaded-text/partial 绑定及真实 PID 记录均由本地独立 reviewer 核验：

~~~text
label=f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v2
archive=TASK/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v2_interrupted_evidence.tar.gz
bytes=6398955
SHA256=bff9eb249b37cd6ecffae4f4e0b3663f1356c6b2b497921968639b237fc19daa
PID reviewer=TASK/review_f3_incomplete_kernel_timer_LOG0_PID_post_external_local_20261028_v2.py
candidate_not_ranked=true
performance_data=false
~~~

本次 external owner 位于即时 post，settled 已退出；新 PID partial reviewer 只针对这一证据位置做检查，原父 reviewer、原有效性公式、冻结文件及全部 raw 数据保持。前三轮 immediate_ownership_empty=true，第四轮=false，外部 post owners 原样写入 receipt。原第一批的 reviewer 和归档不变。

第三批是全新独立六轮：TASK/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v3.frozen_plan.json，SHA256=fcdb0a3c4dcb3957d908bb3313897b57e95350923f2084ac47df14b1a5a4008c。顺序仍为 v80,v93,v93,v80,v80,v93，旧批次不参与统计；继续先做 LOG0 binding preflight，后由 controller 每6秒自动等待全机连续60秒空闲。只测 E64/T1536/topk8/shared0，pin=0、A-preshuffle=1、AITER_LOG_MORE=0；每个实际 child 使用 timeout --signal=TERM --kill-after=2s 118s。性能 child 期间没有 SFTP、额外 GPU 查询或并行远端编译。

生产 source SHA256 仍为93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312。英文 commit/push、远端主机 pull 与 pull 后最终 E64 复测仍待优化收尾；index 中原有 staged 修改保持。

## 2026-10-28：等待 v80/v93 独立对比期间准备 v94 本地源码

问：汇报下进展，然后继续。

答：生产文件保持 v80，v80/v93 第三批独立对比已经完成 LOG0 binding preflight，controller 继续每6秒自动等待全机空闲。前两批中止证据保持，v93 尚未完成性能排名。等待期间只做本地 CPU 源码分析和下一候选准备，没有并行运行远端编译或 profiler。

v94 将已经独立验证过运算图的两个调度改动放进同一个局部 activation helper：每个 CHUNK=4 先 clamp、随后发射该组 tanh；在上一组 consumer 生成后，发射原 ternary amax tree 已就绪的第一层。每个值的 min/clamp、tanh、乘法结合、第一层 maxnumf 分组、后续 scale 数学、BF16 舍入和 FP4 conversion 保持。只选择已有 interleaved_m_rows and not SUPPORT_EXPLICIT_VGPR_PARTITION and not is_situv2 路径，没有新增 expert/token/seed/data specialization。

源码变换可精确逆回 v80；N=1..128、SiLU/SwiGLU 共256项逐值与第一层 maxnumf 运算图检查通过，tanh 和 scheduling-boundary 数量保持。公共 helper 文件、mainloop、TDM/LDS/WMMA、hardware waits/barriers、输出布局、无效行跳过及准入保持。生成器只写 snapshot 与 source plan，不修改生产文件。

~~~text
generator=TASK/derive_f3_v94_clamp_and_amax_level1_local_20261028_v1.py
snapshot=TASK/snapshots/f3_v80_clamp_and_amax_level1_v94.py
source SHA256=d5e85a5ad7bc36afa86bee0c1b8c962acac4df5ae4010072dd89809c07f7c9cd
source plan=TASK/logs/f3_v94_clamp_and_amax_level1_source_20261028_v1.plan.json
exact_arithmetic_graph_checks=256
old_CHUNK=4
new_CHUNK=4
GPU_dispatch=false
performance_data=false
production source SHA256=93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312
~~~

缩短 clamp live range、使其与 TRANS/amax 调度重叠只是待实测假设，不能根据源码重排推算节省的 μs。v94 此时尚未上传用于编译、尚无真实 ELF/native/device 验证或性能成绩。后续实际 child 仍只在 a07-3/dsr1_ep4、E64/topk8/shared0，保持 pin=0、原精度和原 timer；性能使用 AITER_LOG_MORE=0，不运行 E65/topk9。

## 2026-10-28：v80/v93 第三批等待期间的进度和 swizzle 静态检查

问：汇报下进展，然后继续。

答：生产 kernel 继续保持 v80。v93 已通过 E64 seed0/17 完整 MoE 原精度 gate，输出与 reference 逐字节一致；前两批性能对比的中止证据保持，不补轮、不拼接。新独立批次 `f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v3` 首轮 v80 的 GEMM1 为 81.22878947368423 μs，controller 判定该单轮有效，第二轮仍在等待全机连续空闲。完整六轮尚未完成，因此没有新的 v80/v93 中值或排名。历史 LOG1 中值不与本批 LOG0 数据合并。

v94 的 generator、snapshot 和 source plan 已传到 a07-3 的 TASK 对应位置，source/generator SHA256 绑定核对通过。上传在第二轮空闲门禁尚未通过、最近外部占用检查使连续计数归零的等待窗口内完成；没有在性能 child 中执行 SFTP 或远端编译。生产 source SHA256 仍为 `93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。v94 真实编译、native/device 验证和性能仍待完成。

等待期间新增本地 CPU-only swizzle 诊断，直接执行 hash 绑定的原源码区域，与独立分组坐标枚举以及 unsigned 除法模型比较。3060 组几何配置共检查 933240 个 workgroup 坐标，模型内 6266040 次除法满足非负 signed-i32 分子和正 signed-i32 分母条件，坐标和覆盖集合一致。756 个零 grid 配置不执行 kernel 坐标计算。

~~~text
script=TASK/analyze_f3_v80_nonnegative_swizzle_local_20261028_v1.py
receipt=TASK/logs/f3_v80_nonnegative_swizzle_static_20261028_v1.json
source_region_SHA256=748e9bc8b0ee4452dccaac289c1a3c9b53dbe598bee242b249136f3442ba87a2
GPU_dispatch=false
performance_data=false
production_source_unchanged=true
candidate_created=false
~~~

这只是有限几何检查和非负整数域上的除法等价检查。完整运行时 launch 边界、signed-i32 overflow 语义、persistent worker 路径、真实 IR/ELF、设备精度和性能仍未证明，不能据此直接修改生产 swizzle 或声称减少了 μs。该诊断没有创建新的 kernel 候选。

后续仍只在 a07-3/dsr1_ep4 测 E64/T1536/topk8/shared0、random non-balanced，pin=0、A-preshuffle=1，性能实际使用 `AITER_LOG_MORE=0`。原 timer、精度 gate 和有效计算保持；只按 GEMM1 耗时选版，完整 MoE 保留正确性验证。空闲门禁继续每6秒自动检查，全机连续11次、跨度至少60秒；各实际 compiler/test/profiler child 保持118秒 TERM、再2秒 KILL。不运行 E65/topk9，不杀外部任务。

## 2026-10-28：第三批 v80/v93 中止，v94 完整精度通过并进入 LOG0 对比

问：汇报下进展，然后继续。

答：采用版仍为 v80，没有确认新的性能提升。第三批 `f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v3` 在第四轮测后 settled 检查发现外部 KFD owner 而中止，不补轮、不拼接前两批，不计算候选中值或排名。

| ordinal | 版本 | GEMM1 μs | 单轮有效性 |
|---:|---|---:|---|
| 1 | v80 | 81.22878947368423 | 有效，仅单轮 |
| 2 | v93 | 82.52689473684212 | 有效，仅单轮 |
| 3 | v93 | 81.65415789473684 | 有效，仅单轮 |
| 4 | v80 | 82.39205263157890 | 无效，不采用 |

第四轮我方 host PID2379275 已退出；即时 post ownership 为空、busy=5，随后 UTC `2026-10-28T07:27:26.013472+00:00` 的 settled 出现外部 PID2380498，所属 `zan_att` 容器，执行 E1/T16384/topk1/model_dim16384/inter_dim8192 grouped GEMM，busy=9、VRAM1561866240 B。controller 的 finally/post 仍观察到该外部进程，最终 settled 已空闲，生产源码恢复 v80。本批四个我方 child 均已退出。测后快照不能证明计时窗口一定受竞争影响，也不能追溯认可无效轮。

归档 `TASK/logs/f3_E64_LOG0_PID_v80_v93_original_kernel_timer_20261028_v3_interrupted_evidence.tar.gz` 共145项、6406143 B，SHA256=`7481d47dc4f6653845fbe0108da39eb2297f6e07b0f26e29843cda453d52ef40`。下载后，原中止批次、LOG0 环境和真实 host PID 三项独立 reviewer 全部通过；三个 receipts 已传回远端。原冻结规则、四轮 raw 数据及前两批记录保持。暂不重复启动第四批 v93 对比，继续已经准备的 v94 独立候选。

v94 将每 CHUNK=4 的 clamp 与已就绪的原第一层 ternary amax 放入局部 activation pipeline。原逐值数学、amax 分组、后续 scale、BF16/FP4 舍入、mainloop、TDM/LDS/WMMA、hardware waits/barriers、布局、无效行跳过和准入保持，所有 fallback 保留，没有新增 expert/token/data specialization。

v94 的实际 E64 编译及原 native model、WMMA-only retirement stress 均通过：issues=0、两套 CFG iterations=27455；`.text=97280 B`，VGPR799/SGPR60/LDS327680 B，private segment 和 spill 均为0。上述资源与静态检查不用于推算 μs。

~~~text
source_SHA256=d5e85a5ad7bc36afa86bee0c1b8c962acac4df5ae4010072dd89809c07f7c9cd
GEMM1_ELF_SHA256=2e286484d37b0617a265c01eed539d4ae49d785d9f0b179ecb3733bcb1d64311
GEMM2_ELF_SHA256=4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be
CPU_stem=f3_target_v94_clamp_amax_level1_pin0_cpu_20261028_v1
device_stem=f3_v94_complete_MoE_pin0_20261028_v1
native_review_SHA256=b7bb50af6ca973e17fa158b13191bc3a9798d58bcb36a07f484ca7c56e602ee7
device_summary_SHA256=b1995bc0773fee3ebe02c5152479e07559e0bf3289b94750735612ca601881ed
~~~

完整 E64/T1536/topk8/shared0 MoE 的 seed0/17 各22020096 B输出，与已验证 reference 逐字节一致（differing_bytes=0）；logits_diff 分别为 `3.6105513278025183e-06` 和 `3.6202693316254594e-06`，原 gate=0.01。G1 cache SHA/mtime 保持，G2 完整 ELF 保持。编译和正确性过程中的耗时全部排除出性能数据；没有测试 E65/topk9，没有下载完整 MoE binary。

新独立计划 `f3_E64_LOG0_PID_v80_v94_original_kernel_timer_20261028_v1` 已冻结，SHA256=`e58b6adad32518df17fa226eb5eed597dc2125b0b57dca50322ab0af3313caba`，固定顺序 v80/v94/v94/v80/v80/v94。先核对 LOG0 下 eager 与实际 kernel-bench partial 的 JIT key、IR 和 ELF，然后使用原 warmup5/iters20 timer，不改计时迭代、有效数学、输入或精度标准。全部 binding/observation hooks 在 timer 前恢复；不选择地址、seed、实例或快轮次。完整六轮通过之前，没有 v94 的正式性能中值。

~~~bash
python3 -u TASK/run_f3_v80_followup_compile_correctness_20261028_v1.py --plan TASK/logs/f3_v94_clamp_and_amax_level1_source_20261028_v1.plan.json --stem f3_target_v94_clamp_amax_level1_pin0_cpu_20261028_v1
python TASK/review_f3_v80_followup_native_local_20261028_v1.py --stem f3_target_v94_clamp_amax_level1_pin0_cpu_20261028_v1
python3 -u TASK/run_f3_v80_followup_device_E64_only_20261028_v1.py --cpu-stem f3_target_v94_clamp_amax_level1_pin0_cpu_20261028_v1 --stem f3_v94_complete_MoE_pin0_20261028_v1
python3 -u TASK/run_f3_v80_followup_LOG0_binding_preflight_E64_20261028_v1.py --plan TASK/logs/f3_E64_LOG0_PID_v80_v94_original_kernel_timer_20261028_v1.frozen_plan.json
~~~

这些 label 已使用，复现必须另建 label，保留原记录。后续性能仍为 a07-3/dsr1_ep4、E64、pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，每轮前全机连续11次空闲检查、间隔6秒、跨度至少60秒，外部占用自动等待。实际 compiler/test/profiler child 保持118秒 TERM、再2秒 KILL，性能 child 中不执行 SFTP、GPU 查询或远端并行编译。只按 GEMM1 耗时选版，完整 MoE 的原精度验证继续保留。E64/E65 共用 compiled GEMM1 的历史要求和已授权 Git 收尾仍未完成，不能误报。

## 2026-10-28：v94 首批 LOG0 性能中止，补充原计时调用的 CPU 时间边界

问：汇报下进展，然后继续。

答：生产仍保留 v80。v94 已通过 E64 seed0/17 完整 MoE 原精度验证，输出与 reference 逐字节一致；首批性能只完成两轮，第二轮测后出现外部任务，按冻结规则中止。没有新的正式候选中值或排名。

| ordinal | 版本 | 原 timer GEMM1 μs | 单轮有效性 |
|---:|---|---:|---|
| 1 | v80 | 81.43668421052628 | 有效，仅单轮 |
| 2 | v94 | 81.94273684210525 | 无效，不采用 |

第二轮我方 host PID2585132 已退出。即时 post UTC `2026-10-28T07:48:16.950646+00:00` 的 ownership 为空、busy=4；settled UTC `2026-10-28T07:48:20.176702+00:00` 出现外部 KFD owner PID2585845。只读进程检查识别其为 `yadai` 容器中的 E96/T16384/topk6/model_dim7168/inter_dim3072 grouped GEMM。controller 的 finally/post/settled 仍见该外部工作负载；未杀外部进程。两轮我方 child 均已退出，生产 source 恢复 v80。

归档 `TASK/logs/f3_E64_LOG0_PID_v80_v94_original_kernel_timer_20261028_v1_interrupted_evidence.tar.gz` 为4690427 B，共123项，SHA256=`b1018bf0f56d545903bc9ea3bd1123a8a412648da51bbece38110a50504236e3`。原中止、LOG0 环境及真实 host PID 三项独立本地 reviewer 全部通过，receipts 已上传。原 raw 数据、冻结规则和所有此前中止批次保持，不补轮、不拼接，也不据此认可无效轮。

测后快照只能证明该时间点的占用，不能确定外部任务是否与原 GEMM1 计时重叠。下一步在原 `caller.run_moe(...kernel_bench=True...)` 调用前后记录 `CLOCK_BOOTTIME`，在 child 退出及原 post/settled 检查后读取外部 `/proc/<pid>/stat` 的 start_ticks，核对 USER_HZ、boot_id 和 time namespace。该调用窗口包含原输入准备、GEMM1/GEMM2 timer 和其原有验证；它是保守的外层时间窗口，不是新的 GPU kernel timer。

时间戳只用于诊断。若外部进程的创建时间下界晚于调用结束，可以排除这个进程与该调用的重叠；若进程早已存在或创建时间区间与窗口相交，不能据此推断实际 GPU 工作开始时间。有限 tick 分辨率、进程退出或 PID 重用、time namespace 不一致均须保留为无法判断。计时期间不轮询 GPU/KFD，也不增加观察 hooks；原 timer、warmup5/iters20、输入、有效计算和精度 gate 保持。未来若调整测后有效性规则，须先有可靠证据并冻结新规则；本次尚未修改任何有效性判定。

继续只在 a07-3/dsr1_ep4 测 E64/T1536/topk8/shared0、random non-balanced，pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，按孤立 GEMM1 耗时选版。每轮前全机连续11次空闲检查，间隔6秒、跨度至少60秒；外部占用自动等待。每个实际 compiler/test/profiler child 保持118秒 TERM、再2秒 KILL。未测试 E65/topk9，完整 MoE 原精度验证继续保留，Git 收尾仍待最终选版。

## 2026-10-28：CPU 时间窗口诊断验收，缓存 helper 绑定修复准备及 GPU 等待

问：汇报下进展，然后继续。

答：生产仍为 v80，目前没有新的有效性能中值。最近 LOG0 有效单轮 `81.43668421052628 μs` 来自此前 v80/v94 中止批次的第一轮，不能当成三轮中值；历史 LOG1 中值不与当前 LOG0 数据合并。v94 的 E64 seed0/17 完整 MoE 原精度与逐字节输出验证已通过，尚无完整性能采用结论。后续继续只测 E64/topk8/shared0，pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，使用原孤立 GEMM1 timer。

### 原调用 CPU 时间窗口与 root owner 读取

新增诊断只在原 `caller.run_moe(...kernel_bench=True...)` 调用前后读取 `CLOCK_BOOTTIME`，在 timed child 退出并完成原 post/settled 后读取外部进程创建时间。原 caller、timer body、partial、warmup5/iters20、GPU 数学和精度标准保持。原有 eager 绑定观察在内层 timer 开始前恢复；本次没有在 timed iterations 中增加查询或 hooks。外层窗口包含原输入准备、eager 绑定、GEMM1/GEMM2 timer 及其原验证，并非新的 GPU 耗时。

28 项 CPU-only 边界检查覆盖 USER_HZ=100/250/1000/1024、创建时间 tick 区间及边界交叉。主机与 dsr1_ep4 的实际 USER_HZ=100、boot_id、time namespace 和 timens_offsets 一致；容器 CPU 时间位于主机启动前后时间括号内。只读测试发现普通主机用户无法完整读取 root 容器进程 `/proc` 元数据，`sudo -n` 的 CPU-only 读取成功，因此先取消尚未产生任何性能 child 的 CLOCK v1 计划，保留冻结文件、预检与零轮 summary，确认恢复 v80。

权限修复后，主机、容器和特权 CPU reader 的时钟域一致验证通过，对当时外部 root PID2812839 的实际只读 metadata 测试也成功。该测试使用合成 CPU 窗口，不是性能数据。读取失败、PID 消失、身份不稳定、tick 边界交叉或时钟域不一致仍保留为无法确定；进程创建时间不能替代其实际 GPU 工作开始时间。

~~~text
clock/helper=TASK/f3_original_call_CPU_window_20261028_v2.py
helper_SHA256=06a4770961d724aaa9f918246625dd94db0f5f69abbd0d7d143635bf3548acb3
clock_domain_receipt=TASK/logs/f3_original_call_CPU_clock_domain_20261028_v2.summary.json
tools_receipt=TASK/logs/f3_original_call_CPU_window_tools_local_20261028_v2.json
cancelled_before_any_performance_child=f3_E64_LOG0_PID_CLOCK_v80_v94_original_kernel_timer_20261028_v1
new_plan=f3_E64_LOG0_PID_CLOCK_v80_v94_original_kernel_timer_20261028_v2
new_plan_SHA256=495afeb9d41727193897337abe22610709eb037f3f9422227adb2b8967e87954
original_post_rule_unchanged=true
old_interrupted_batches_never_revalidated=true
~~~

新计划仍固定六轮 v80,v94,v94,v80,v80,v94。两版 LOG0 eager/partial/key/IR/ELF 预检后再运行；每轮前全机连续11次空闲检查、间隔6秒、跨度至少60秒。有外部任务则自动等待，不杀进程。每个实际 compiler/test/profiler child 继续118秒 TERM、再2秒 KILL，性能 child 中没有 SFTP、额外 GPU 查询或远端并行编译。截至本 checkpoint，没有运行新的性能 child：此前外部 att_driver 释放后，dev_al 容器的 pytest 又持有 KFD/VRAM，等待门禁继续有效。这不能解释成测试卡住或性能回退。

### 缓存身份审计与准备的修复

本地 CPU 审计确认 v93/v94 的 `launch_gemm_a8w4_tdm` 完整源码相同，发生变化的 `_batched_silu_swiglu_amax_level1` 只在 nested code object 中被引用，launcher 自身的 co_names/co_freevars 均未引用这两个 private helper。已保存的 installed jit_function.py 快照 SHA256 为 `f677c8f424ee2698a4c655fa8b7bc5a432c2974c62226a88e619f1cb80e07ed5`，其 dependency collector 扫描 co_names/closures，未遍历 co_consts。两份已验证 device 记录确有相同 namespace/key、不同实际 IR/ELF；使用不同私有 cache root且实际 ELF 绑定经过核验，因此现有测试身份保持可靠。当前安装源码尚未重新核验，不能将快照分析扩大成当前所有 FlyDSL 版本的结论。

准备的修复在 launcher 中增加两处 callable alias，使原 helper 全局名在 launcher co_names 中可见，并让 nested kernel 捕获对应 alias。移除两个 assignment 并还原调用名后，整个 module AST 与 v94 一致；helper 运算、同步、布局、准入、支持分支和有效计算均保持。没有新增 expert/token/data specialization，也没有修改生产 v80。

~~~text
audit=TASK/analyze_f3_v93_v94_nested_helper_cache_local_20261028_v1.py
audit_receipt=TASK/logs/f3_v93_v94_nested_helper_cache_static_20261028_v1.json
prepared_snapshot=TASK/snapshots/f3_v94_helpers_visible_to_JIT_cache_20261028_v1.py
prepared_source_SHA256=41a2bbd1fd1483927fd48f472e81c6593713f1509b979fea8c64dbe334e6086e
prepared_receipt=TASK/logs/f3_v94_helpers_visible_to_JIT_cache_static_20261028_v1.json
production_source_changed=false
actual_compile_IR_ELF_cache_key_and_device_validation=pending
performance_data=false
~~~

这是缓存依赖修复的本地准备，不是已验收的性能优化。后续须验证实际 installed runtime 的 key、编译 IR/ELF、native 与完整 E64 精度，才能接入。当前优先完成带 CPU 时间诊断的 v80/v94 性能计划，不因为忙卡或不完整数据换版。禁止 E65/topk9 的约束继续覆盖性能、正确性及 profiler；E64/E65 共用 compiled GEMM1 的历史要求与已授权 Git 收尾仍未完成。

## 2026-10-28：当前 cache 依赖的 CPU 验证通过，原性能对照继续等待

问：汇报下进展，然后继续。

答：生产仍保留 v80。最近有效的 `AITER_LOG_MORE=0`、pin=0 单轮 GEMM1 为 `81.43668421052628 μs`；该数据来自此前中止批次第一轮，不能当作完整中值。v94 的 E64 seed0/17 完整 MoE 原精度与逐字节输出验证保持通过，当前六轮性能计划尚未产生 performance child，没有新的有效中值或采用结论。

### 当前安装 runtime 与 helper 依赖

在原 controller 等待外部 GPU 任务的窗口中，仅通过容器文件读取重新保存当前安装的 `/usr/local/lib/python3.12/dist-packages/flydsl/compiler/jit_function.py`。源码为72049 B，SHA256=`f677c8f424ee2698a4c655fa8b7bc5a432c2974c62226a88e619f1cb80e07ed5`，与此前快照一致。当前 `_collect_dependency_sources` 的确读取 co_names/closures，未遍历 nested co_consts。

随后在 dsr1_ep4 运行 CPU-only 依赖验证：只构造三个源码的 top-level function definitions，检查每个 function body AST 不变，再绑定原始完整 Python code object。没有导入 AITER，没有调用 launcher/kernel，没有生成 typed launch 或执行 GPU 工作，也没有修改生产或 installed runtime。调用的是当前安装的 `_collect_dependency_sources` 和 `_jit_function_cache_key`。

| definition-only 源码 | 两个 private helper 是否进入依赖 | installed function source key |
|---|---|---|
| v93 | 否 | `8fda3eb7a057cc7fc9348fc76a7ff88f` |
| v94 | 否 | `8fda3eb7a057cc7fc9348fc76a7ff88f` |
| v94 加 callable aliases | 是 | `c0c7f2d27c74e6f67590f6696c81a0b6` |

在保持 alias launcher 源码相同的情况下，将第一个 helper 的 callable 重绑定到 v93 的原 helper 后，key 变为 `faab5bf4abb6c68e8ba91038314464ea`。这验证了准备的 alias 修复能够让当前安装的 source-key collector 跟踪 helper 内容变化。上表是 definitions-only source-key 诊断，不能当作真实 typed launch 的 cache namespace/key；实际 IR、ELF、native 和完整 E64 device 验证仍待完成。此前已验证的 v93/v94 actual artifacts 使用独立 cache root并核验实际 ELF，原记录保持。

~~~text
installed_snapshot=TASK/logs/f3_installed_jit_function_CPU_snapshot_20261028_v2.py.txt
installed_snapshot_receipt=TASK/logs/f3_installed_jit_function_CPU_snapshot_20261028_v2.json
CPU_checker=TASK/check_f3_installed_helper_cache_dependency_CPU_20261028_v2.py
CPU_checker_SHA256=4cbb17412ccdf23de91c6ee8f42020627e49ca43d9fcef4a086464ff4888003d
CPU_receipt=TASK/logs/f3_installed_helper_cache_dependency_CPU_20261028_v2.json
CPU_receipt_SHA256=95311a5f777b91cebe14682dfa042e330152fba5c7d04a7fd30fe50a8da60242
local_review=TASK/logs/f3_installed_helper_cache_dependency_CPU_local_review_20261028_v2.json
prepared_alias_snapshot_SHA256=41a2bbd1fd1483927fd48f472e81c6593713f1509b979fea8c64dbe334e6086e
production_source_SHA256=93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312
~~~

### 原计时 CPU 时间窗口的复核工具

新增 `review_f3_CPU_window_kernel_timer_local_20261028_v2.py` 已通过 source-only 复核：独立重建原 checker、LOG0/PID wrapper 和 controller 到当前 v2 的全部差异，确认只增加原调用之外的 CPU clock reads 和测后的 CPU metadata 诊断。原 caller/test_common.py SHA、warmup5/iters20、kernel-bench partial、原数学与原 post/settled 性能有效性公式保持。原有 eager/HSA 观察仍在内层 timer 前恢复；本次没有向 timed iterations 加入 GPU 查询或观察 hooks。

该 reviewer 使用 `Fraction` 独立重算外部创建 tick 区间与原 CPU 调用窗口的关系，并核对 raw stat 的 PID/start_ticks、boot_id、USER_HZ 和 time namespace。读取失败、身份不稳定或时钟域不兼容仍为 unknown；进程创建时间不等于 GPU 工作开始时间。完整六轮时另外复用原 full PID/ELF/ABI/timer reviewer，仅适配 active child 文件名；中止批次保持不排名，不追溯认可旧无效轮。归档工具支持零轮中止或第六轮失败，保留原始状态。

~~~text
source_review=TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v94_original_kernel_timer_20261028_v2.CPU_window_sources_local_review.json
CPU_window_reviewer=TASK/review_f3_CPU_window_kernel_timer_local_20261028_v2.py
full_PID_reviewer=TASK/review_f3_CPU_window_full_PID_local_20261028_v2.py
packer=TASK/pack_f3_CPU_window_kernel_timer_20261028_v2.py
active_plan=f3_E64_LOG0_PID_CLOCK_v80_v94_original_kernel_timer_20261028_v2
active_plan_SHA256=495afeb9d41727193897337abe22610709eb037f3f9422227adb2b8967e87954
fixed_order=v80,v94,v94,v80,v80,v94
completed_performance_rounds_at_checkpoint=0
~~~

这轮等待期间，dev_al、zan_att、yadai 等外部任务反复持有 GPU/KFD/VRAM，原全机连续11次、间隔6秒、跨度至少60秒空闲门禁持续自动检查，没有杀外部进程或竞争计时。性能 child 中不进行 SFTP、额外 GPU 查询或远端并行编译；CPU-only 源码/依赖工作和文件传输安排在确定尚未进入性能 child 的等待窗口。所有后续 kernel compiler/test/profiler child 仍执行118秒 TERM、再2秒 KILL。

后续只在 a07-3/dsr1_ep4 测 E64/T1536/topk8/shared0，保持 random non-balanced、pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，按孤立 GEMM1 耗时选版，完整 MoE 精度标准保持。禁止 E65/topk9 的约束继续覆盖性能、正确性和 profiler。生产仍为 v80，E64/E65 共用 compiled GEMM1 的历史要求和已授权 Git 收尾仍待完成。

## 2026-10-28：v94 cache 依赖修复的实际编译与完整 E64 验证通过

问：汇报下进展，然后继续。

答：生产源码仍为 v80，最近有效的 LOG0/pin=0 单轮 GEMM1 为 `81.43668421052628 μs`。修复版 v94 的实际 cache 绑定、完整 ELF/ABI 和 E64 seed0/17 完整 MoE 原精度验收已通过，尚无新的完整性能中值。后续继续仅在 a07-3/dsr1_ep4 测 E64/T1536/topk8/shared0、random non-balanced，性能使用 pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，按孤立 GEMM1 耗时选版。

### 零性能轮的旧计划归档

旧 `f3_E64_LOG0_PID_CLOCK_v80_v94_original_kernel_timer_20261028_v2` 一直等待外部 GPU 任务，未启动任何 performance child、未产生任何计时轮。为先验收已发现的 cache 依赖修复，主动停止我方 controller；未杀外部进程，未选择、删除或替换性能轮，源码恢复 v80。旧计划不重新启动。

其零轮归档为 `TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v94_original_kernel_timer_20261028_v2_interrupted_evidence.tar.gz`，110项、2989840 B，SHA256=`01c0920b1fb03199fd53589846799df8974f87acf284d7fb91cd497f17afae9a`。下载后的 CPU-window 独立复核通过，`rounds=0`、`performance_data=false`；取消原因保存在 `.cancel_before_any_timing_for_cache_fix.json`。旧冻结规则和此前中止批次保持，不拼接数据。

### 实际 cache/ELF 绑定

修复在 `launch_gemm_a8w4_tdm` 增加两个 callable alias，让 nested kernel 使用的 private helper 进入当前 installed JIT dependency collector。移除两个 assignment 并还原调用名后，完整 module AST 与原 v94 相同。原 globals 值全部保持；新增21项明确列出的依赖，来源为两个 helper 及 gemm_common、quant_utils、tensor_shim 等传递依赖。当前 runtime 的 CPU-only source-key 测试已验证 helper 内容变化会触发 key 变化，不能将 definitions-only key 与 actual typed key 混用。

实际 E64 compile-only 和 ISA inspection 已完成，各 child 为19.743秒、0.164秒，退出码均为0；其中输入准备可能使用 GPU，所得耗时全部排除出性能数据。修复版完整 ELF 与原 v94 逐字节相同，ABI 和全部 AMDGPU metadata 相同。

~~~text
source=TASK/snapshots/f3_v94_helpers_visible_to_JIT_cache_20261028_v1.py
source_SHA256=41a2bbd1fd1483927fd48f472e81c6593713f1509b979fea8c64dbe334e6086e
actual_typed_cache_namespace=launch_gemm_a8w4_tdm_d5ba48f69259cb3ca0a11f12de1590fe
actual_typed_cache_key=8716159b89eb9f27
actual_source_IR_SHA256=1bc53597c7939e09611e3f5531c99bc2c5861db8b1b93f4315ed04f3574f5e9c
full_ELF_SHA256=2e286484d37b0617a265c01eed539d4ae49d785d9f0b179ecb3733bcb1d64311
text_bytes=97280
text_SHA256=3ac6b8b05997fb2c94e6be00508c1bc6ee6506d37b48a518350aa50b84952d34
VGPR=799
SGPR=60
LDS_bytes=327680
private_segment_and_VGPR_SGPR_spills=0
compile_archive_SHA256=ab4cbe5a0c62b29a33cf6990cf21864d25afb1770eb6c4300cd119bcc2a852bfd
compile_archive_bytes=988788
~~~

新 native receipt 复用原 v94 已通过的 original/WMMA-only bounded DS/WMMA 检查，前提是核验完整 ELF、ABI、metadata 及 source 数学等价，并逐项核验新增依赖和原 globals 值。没有重跑相同机器码的 native model；`new_native_model_runs=0`，复用的两套报告各 issues=0、CFG iterations=27455。这是相同 ELF 的静态证据复用，不声称新 device 或性能结果。原 conservative model 的限制保持。

~~~text
binding_reviewer=TASK/review_f3_v94_cache_alias_binding_local_20261028_v1.py
new_native_receipt=TASK/logs/f3_target_v94_cache_alias_pin0_cpu_20261028_v1_native.review.json
new_native_receipt_SHA256=7ce56f5de919d9dec16d966dfd3e504b07e337d7f25afbfb25eaffb40b5075e4
reused_original_native_receipt_SHA256=b7bb50af6ca973e17fa158b13191bc3a9798d58bcb36a07f484ca7c56e602ee7
~~~

### 新修复版的完整 E64 device 验证

实际新跑 GEMM2 准备、完整 MoE seed0、完整 MoE seed17，三个 child 分别13.885秒、8.527秒、8.176秒，退出码均为0。E64/T1536/topk8/shared0 两组完整 BF16 输出各22020096 B，均与原 reference 逐字节一致。原精度 gate=0.01 保持；GEMM2 完整 ELF 为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`，与原版相同，已验收 GEMM1 cache 文件的 SHA256 和 mtime 均保持。

| seed | logits_diff | rel_l2 | differing_bytes |
|---:|---:|---:|---:|
| 0 | 3.6105513278025183e-06 | 0.0026872099842876196 | 0 |
| 17 | 3.6202693316254594e-06 | 0.002690816530957818 | 0 |

验证后生产恢复 v80，我方三个 child 均已退出。测后观察到外部 dev_al PID3768458 在 `/workspace/FlyDSL` 执行 `python -m pytest tests/kernels/test_gemm_fp8fp4_gfx1250.py -xq --durations=10` 并持有 GPU/KFD/VRAM；未杀或驱逐其任务。上述运行只用于正确性，其打印的任何时间均不作性能数据。

~~~text
device_summary=TASK/logs/f3_v94_cache_alias_complete_MoE_pin0_20261028_v1.summary.json
device_summary_SHA256=28b9593748ba0cf920af32c047e15874bdb5be1c35706b3049e792091de44e9e
~~~

### 新 LOG0 六轮性能计划

新冻结计划 `f3_E64_LOG0_PID_CLOCK_v80_v94_cache_fixed_kernel_timer_20261028_v1` 的 SHA256=`0429ea2359cfd09cdd1bb11fb3e92ea8d8856420e88eaeba15bc2e17110d575f`。这里 v94 指 cache 修复后的相同 v94 机器码。固定顺序仍为 v80/v94/v94/v80/v80/v94，原 timer、warmup5/iters20、输入、精度标准、测后有效性规则及 CPU-window/PID checker 均保持，不选地址、seed、加载实例或快轮次。CPU source-only 独立复核已经通过。先做两版 LOG0 eager/partial/key/IR/ELF 预检，再进入全机连续11次、间隔6秒、跨度至少60秒空闲门禁；有外部任务则自动等待。当前尚无本计划性能轮。

~~~bash
python TASK/review_f3_v94_cache_alias_binding_local_20261028_v1.py
python3 TASK/run_f3_v80_followup_device_E64_only_20261028_v1.py --cpu-stem f3_target_v94_cache_alias_pin0_cpu_20261028_v1 --stem f3_v94_cache_alias_complete_MoE_pin0_20261028_v1
python TASK/freeze_f3_v94_cache_fixed_CPU_window_local_20261028_v1.py
python3 TASK/run_f3_v80_followup_LOG0_binding_preflight_E64_20261028_v1.py --plan TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v94_cache_fixed_kernel_timer_20261028_v1.frozen_plan.json
python3 TASK/run_f3_original_kernel_timer_CPU_WINDOW_E64_LOG0_PID_20261028_v2.py --plan TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v94_cache_fixed_kernel_timer_20261028_v1.frozen_plan.json
~~~

性能 child 中禁止 SFTP、额外 GPU 查询和并行远端编译；kernel compiler/test/profiler child 继续执行118秒 TERM、再2秒 KILL。E65/topk9 禁止测试继续覆盖正确性、性能及 profiler。生产采用结论尚未改变，E64/E65 共用 compiled GEMM1 的历史要求与已授权英文 commit/push、主机 pull 及 pull 后最终 E64 复测仍未完成。

## 2026-10-28：v94 正式对照继续等待，完成按 M-wave 的尾行覆盖分析

问：汇报下进展，然后继续。

答：生产仍为 v80。最近有效的 LOG0/pin=0 GEMM1 单轮为 `81.43668421052628 μs`，来自已保留的中止批次，不能当成完整性能中值。cache 修复版 v94 的 actual namespace/key、完整 ELF/ABI、原 native 证据绑定与 E64 seed0/17 完整 MoE 原精度验证已通过；本轮不重复这些已完成的验收。截至 UTC 2026-10-28 10:25:12，正在运行的 `f3_E64_LOG0_PID_CLOCK_v80_v94_cache_fixed_kernel_timer_20261028_v1` 仍处于第一轮测前空闲门禁，performance child 与计时轮均为0。a07-3 上外部任务反复接续持有 GPU/KFD/VRAM，未杀外部进程、未竞争计时，也未重新启动或拼接旧性能计划。

### 已有 ELF 的本地静态比较

`TASK/logs/f3_v80_v94_cache_fixed_static_activation_amax_local_20261028_v1.json` 的 SHA256 为 `b4d1b60e830559a755b934400569172f5f2055cdb5f0fa4d2ea60459d99fa711`。v80/v94 `.text` 为97408/97280 B，静态 `s_set_vgpr_msb` 数为2034/1999。28个内部 backward-edge 区间的 opcode 顺序及 WMMA 数量相同，寄存器 operands 有变化。这些是实际已绑定 ELF 的静态事实，不能换算成运行时间、动态指令量或 cache 收益，没有据此采用 v94。

### planar B alignment 的容量限制

重新查阅本地 `MI400_Shader_Programming#65.txt` §4.7.1（正文 p160）和 §4.10.2（正文 p199–200）：LDS/WGP$ 共享384 KiB，按64 KiB segments 分区；TDM direct-copy 的相应条件包含至少4-byte LDS alignment。当前几何的有效输入数据为272 KiB，原64 KiB B alignment使 arena为320 KiB；若仅改为32 KiB，预计为288 KiB，仍未跨过该64 KiB容量分区边界。旧文档已记录去掉间隙也不能单独增加 cache 分区。因此，本轮没有为此创建或运行 v95，也不声称 occupancy 增加。该事实不证明任何重新布局的性能，实际编译与硬件访问仍须单独验证。

### 下一方向的本地静态准备：按各 M-wave 起点判断无效尾行

当前 interleaved M 布局两个 M-wave 的起点为0、16，步长32；原 pin0 selector 使用同一 `mn_oob`，两者均选择同一4/6/7/8-rep body。准备分析的表达式为 `wave_row_oob = mn_oob - wmb`，仅替换已有 selector 中四处 bound 的使用，保留原四个 body。

| expert 有效行数 | 原两个 M-wave 的 reps | 按 wave 起点的 reps | 原 / 预计 WMMA 数每 WG |
|---:|---|---|---:|
| 192 | 6 / 6 | 6 / 6 | 5376 / 5376 |
| 193–208 | 7 / 7 | 7 / 6 | 6272 / 5824 |
| 209 | 7 / 7 | 7 / 7 | 6272 / 6272 |
| 225 | 8 / 8 | 8 / 7 | 7168 / 6720 |

上表是 K7168、tileK256、w2x2、每个 M-wave 含两个 N-wave 的源码工作量计算，**不是性能数据**。193–208行时，第二个 M-wave 的第7个 M16 block 从208行开始，完全无效；225行时，其第8个 block 从240行开始，也完全无效。已从 v80/v94 的实际 selector AST 检查0–256行的全部257种情况：所有有效行完整且唯一覆盖，新增跳过的 block 全部无效；也检查了非负 signed-int32 域的 selector 分段。数学、ScaleA opsel、activation/quant、输出、原准入和 fallback 的源码保持，没有新增 expert/token/seed 编译 specialization。

~~~text
analyzer=TASK/analyze_f3_wave_local_row_coverage_local_20261028_v1.py
receipt=TASK/logs/f3_wave_local_row_coverage_static_local_20261028_v1.json
receipt_SHA256=0ad3f80770678edbbbc3cf0678f52aa454ea56fb9df8d845644affe5f7c8ff95
production_source_changed=false
candidate_kernel_snapshot_created=false
GPU_dispatch=false
performance_data=false
~~~

~~~powershell
python .codex_tmp/hyg_f3_followup_20261024/analyze_f3_wave_local_row_coverage_local_20261028_v1.py
~~~

这里只完成本地准备，尚未创建 kernel 候选、编译或做 device 测试。原 input TDM/fence 源码虽保持，**同一 WG 内不同 M-wave 执行不同 body 的动态 barrier、寄存器 retirement 和 I-cache 影响仍未验证**；减少 WMMA 也不能保证缩短 WG 的最长路径。先完成当前 v80/v94 原 timer 对照，再从有效 GEMM1 最快版派生候选，并完成这些 gate 及完整 E64 精度验证。

继续仅在 a07-3/dsr1_ep4 测 E64/T1536/topk8/shared0，保持 random non-balanced、pin=0、A-preshuffle=1、性能 `AITER_LOG_MORE=0`、原 timer/warmup5/iters20及精度标准。性能 child 中不运行 SFTP、额外 GPU 查询或远端并行编译；每个 compiler/test/profiler child 仍为118秒 TERM、再2秒 KILL。E65/topk9 禁止测试继续覆盖性能、正确性和 profiler；历史共用 compiled GEMM1 要求及已授权 Git 最终收尾仍未完成。

## 2026-10-28：v95 按 M-wave 跳过无效尾行通过验收，v94 性能批次按原规则中止

问：汇报下进展，然后继续。

答：生产保持 v80。新的 v95 只修改原 pin0 row-coverage selector，已通过实际编译、原 native/WMMA-only 检查和 E64 seed0/17 完整 MoE 原精度验收。两组完整 BF16 输出均与 reference 逐字节一致，尚无 v95 性能数据。v80/v94 正式六轮批次在第2轮测后遇到外部 GPU 占用，按原冻结规则中止并完整归档；没有补轮、拼接或根据这轮耗时采用 v94。继续只在 a07-3/dsr1_ep4 测 E64/T1536/topk8/shared0，性能固定 pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，按孤立 GEMM1 耗时选版，完整 MoE 精度标准保持。

### 原 v80/v94 批次的精确状态

`f3_E64_LOG0_PID_CLOCK_v80_v94_cache_fixed_kernel_timer_20261028_v1` 完成了2轮：第1轮 v80 的 GEMM1 为 `81.98705263157898 μs`，原 controller 判定 valid=true；第2轮 v94 的原始记录为 `83.34473684210528 μs`，valid=false，**不得用于性能排名或收益计算**。这一批没有完整中值，第1轮也不能与此前 `81.43668421052628 μs` 的单轮拼接成新批次。

第2轮原 kernel-bench 调用的 CPU outer window 为 `[73400636899850, 73401411153210] ns`，使用 `CLOCK_BOOTTIME`。直接 post 检查的 ownership 为空；3秒后的 settled 检查出现外部 grouped GEMM PID1152204，GPU/KFD/VRAM 已被使用，因此原有效性公式拒绝该轮。我方 host child PID1150572 已退出，没有把外部 PID 当成我方残留。

CPU-only 创建时间诊断核验该外部 PID 的稳定身份、start_ticks=7340231、USER_HZ=100、boot_id 和相同 time namespace，结论为 `definitely_created_after_original_call`。这证明其进程创建晚于上述原调用窗口，**不等于测到了 GPU 工作开始时刻，也没有据此放宽本批有效性规则或追溯认可第2轮**。外部任务未被终止。

整批中止归档下载后的 CPU-window 独立复核通过，`rounds=2`、`performance_data=false`。137个归档成员、4,758,193 B，全部原始轮次、source/IR/ELF、PID/clock 和门禁日志保留。原 v80/v94 计划不重新启动。

~~~text
interrupted_archive=TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v94_cache_fixed_kernel_timer_20261028_v1_interrupted_evidence.tar.gz
interrupted_archive_SHA256=cce38aaa41ea53f8e794788273b026a5d8fb4450b5dfea706d68b51cc7e87e46
independent_review=TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v94_cache_fixed_kernel_timer_20261028_v1.CPU_window_independent_local_review.json
production_source_SHA256=93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312
~~~

### v95 的源码变更与功能边界

从当前 v80 派生，在原 `interleaved_m_rows and not SUPPORT_EXPLICIT_VGPR_PARTITION` selector 内增加 `wave_row_oob = mn_oob - wmb`，仅将四处 bound 的 `mn_oob` 改为 `wave_row_oob`。原4/6/7/8-rep body、计算表达式与舍入、ScaleA opsel、input TDM、fence/barrier、activation/quant、输出、准入及 guard 外 fallback 的源码全部保持。完整源码可逆恢复为 v80，没有新增 expert/token/seed 编译 specialization，也没有更改输入或选择 seed。

已绑定前一轮完成的0–256行全部257种静态覆盖检查，新增跳过的 M16 block 均完全无效。例如193–208行时，两个 M-wave 从7/7-rep 改为7/6-rep；225行时从8/8-rep 改为8/7-rep。这里的工作量减少不是性能收益结论。

~~~text
source_generator=TASK/derive_f3_v95_wave_local_tail_source_local_20261028_v1.py
source_snapshot=TASK/snapshots/f3_v95_wave_local_tail_20261028_v1.py
source_SHA256=fc45d76807a8e0130057eb69007f1177abb4c647c9e2eb642b1346431a46de8e
source_plan=TASK/logs/f3_v95_wave_local_tail_source_20261028_v1.plan.json
source_plan_SHA256=3259e8c1bba0663b445344da4b3e0c3326e239fde20a9ce7b97868a7ae7ec13a
static_coverage_receipt_SHA256=0ad3f80770678edbbbc3cf0678f52aa454ea56fb9df8d845644affe5f7c8ff95
~~~

### 实际 ELF 与 native 检查

实际 E64 compile-only 和 ISA inspection 均退出0，child 用时19.794146/0.163933秒。原 native 与 WMMA-only 两套 model 均通过，issues=0、CFG iterations=27549，各用时3.945863/3.163970秒。实际 typed argument/globals signature 与 dispatch ABI 均与 v80 相同；原 protected static operations 的数量相同，没有引入 call/indirect branch。

~~~text
compile_stem=f3_target_v95_wave_local_tail_pin0_cpu_20261028_v1
GEMM1_ELF_SHA256=e8e76ca32e411aad248af9a45967734a82c4e229c2e685769fd77c34e2bb5921
text_SHA256=42bbf10c772a4fce90d029772a8038b7254f2bf448077603d6f6fbcf83c87248
text_bytes=97408
VGPR=799
SGPR=61
LDS_bytes=327680
private_segment_and_VGPR_SGPR_spills=0
compile_archive_SHA256=41f94a02a3c3bd1990ccf45a9ed3809c6da92c92fdd8065feafb9f168b3bbbb1
native_review=TASK/logs/f3_target_v95_wave_local_tail_pin0_cpu_20261028_v1_native.review.json
~~~

原 conservative bounded DS/WMMA model 保持，未放宽 transfer；它不完整模拟 VALU/VMEM、EXEC skip 或 tensor completion，不能替代 device 正确性，也不提供性能结论。v95 的代码仍跨越64 KiB；没有声称新的 selector 已消除加载地址敏感性，混合6/7-rep body 的 I-cache 和 WG 最长路径影响仍待实测。

### 完整 E64 MoE 验证

实际运行 GEMM2 准备、完整 MoE seed0 和 seed17，三个 child 分别13.835402、8.476390、8.176145秒，均退出0。两组输出各22,020,096 B，均与原 reference 逐字节一致；gate=0.01 保持。测后下载的独立记录与 summary 一致，GEMM2 完整 ELF SHA256 也与基线相同。已验收的 GEMM1 cache SHA256 和 mtime 保持。

| seed | logits_diff | rel_l2 | differing_bytes |
|---:|---:|---:|---:|
| 0 | 3.6105513278025183e-06 | 0.0026872099842876196 | 0 |
| 17 | 3.6202693316254594e-06 | 0.002690816530957818 | 0 |

~~~text
device_summary=TASK/logs/f3_v95_wave_local_tail_complete_MoE_pin0_20261028_v1.summary.json
device_summary_SHA256=41ecdf68fac4c96cea86c8f2dea71f2547a7375cc70816d4ed77fb1591a913e0
GEMM2_ELF_SHA256=4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be
seed0_output_and_reference_SHA256=443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935
seed17_output_and_reference_SHA256=04b4ad0f8e6f51991674e95ffcf432d4a3873c20c5064fbc9212bf52cc0e294a
~~~

这些 compiler/native/device child 的秒数只证明运行在120秒硬期限内；准备和正确性打印的任何 kernel 时间均排除出性能数据。

### 在等待窗口完成准备，保持同一旧性能计划

编译和 device 验证前分别核验我方 controller PID3957640、start_ticks=6732430、tmux pane %6/parent132704，以及实际等待 helper PID668642 的身份；仅对等待中的我方父 controller 发送 SIGSTOP。STOP 后重新确认没有 performance child，原 helper 持续每6秒轮询。正确性运行产生的我方 GPU/KFD 样本确实使门禁重新累计；源码恢复 v80、所有冻结文件 SHA256 不变、helper 未完成且正确性结束后的新 active 样本仍重新累计门禁后，再向同一 controller 发送 SIGCONT。没有沿用正确性之前已完成的旧空闲窗口。

这两次准备均发生在第1轮完成、第2轮性能 child 启动之前，没有重启或替换性能轮，没有修改 timer 或旧计划有效性规则，也没有干预外部任务。相应证据为 `f3_v95_compile_pause_same_timer_20261028_v1.json`、`f3_v95_compile_resume_same_timer_20261028_v1.json`、`f3_v95_device_pause_same_timer_20261028_v1.json`、`f3_v95_device_resume_same_timer_20261028_v1.json`。原计划后来因第2轮 settled 占用中止，是独立保留的结果。

### 下一次性能对照

新的 `f3_E64_LOG0_PID_CLOCK_v80_v95_wave_local_tail_kernel_timer_20261028_v1` 已冻结并通过 CPU source-only 复核，SHA256=`7b20264de991d0288dd2a88371e299c48bb05c82e95dd90ab95e62b8a83e750f`。固定顺序为 v80/v95/v95/v80/v80/v95，case 仅 E64/topk8/shared0。复用原 CPU-window/PID checker、controller、原 timer、warmup5/iters20、输入、精度和 post/settled 有效性规则；旧中止轮次不带入新批次。先做两版 LOG0 eager/partial/key/IR/ELF binding preflight，再进行全机连续11次、间隔6秒、跨度至少60秒空闲门禁。

冻结模板中旧 prototype 的 snapshot/common_bound/boot annotations 移至 `historical_template_annotations` 作为历史 provenance；当前 candidate 的 case、typed keys、IR/ELF 与精度证据均使用本次 E64 验收记录。历史 E64/E65 共用 compiled GEMM1 要求仍未完成，不据此宣称已满足；E65/topk9 的禁测约束保持。

~~~bash
python3 TASK/run_f3_v80_followup_compile_correctness_20261028_v1.py --plan TASK/logs/f3_v95_wave_local_tail_source_20261028_v1.plan.json --stem f3_target_v95_wave_local_tail_pin0_cpu_20261028_v1
python TASK/review_f3_v80_followup_native_local_20261028_v1.py --stem f3_target_v95_wave_local_tail_pin0_cpu_20261028_v1
python3 TASK/run_f3_v80_followup_device_E64_only_20261028_v1.py --cpu-stem f3_target_v95_wave_local_tail_pin0_cpu_20261028_v1 --stem f3_v95_wave_local_tail_complete_MoE_pin0_20261028_v1
python TASK/freeze_f3_v95_CPU_window_local_20261028_v1.py
python TASK/review_f3_CPU_window_kernel_timer_local_20261028_v2.py --label f3_E64_LOG0_PID_CLOCK_v80_v95_wave_local_tail_kernel_timer_20261028_v1 --check-sources-only
python3 TASK/run_f3_v80_followup_LOG0_binding_preflight_E64_20261028_v1.py --plan TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v95_wave_local_tail_kernel_timer_20261028_v1.frozen_plan.json
python3 TASK/run_f3_original_kernel_timer_CPU_WINDOW_E64_LOG0_PID_20261028_v2.py --plan TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v95_wave_local_tail_kernel_timer_20261028_v1.frozen_plan.json
~~~

性能 child 中不进行 SFTP、额外 GPU 查询或并行远端编译；每个 compiler/test/profiler child 继续118秒 TERM、再2秒 KILL。生产仍为 v80，已授权的英文 commit/push、主机 pull 与 pull 后最终 E64 复测在最终选版后完成。

## 2026-10-28：v95 首批性能中止归档，实际 routing 的无效 WMMA 工作量分析

问：汇报下进展，然后继续。

答：生产继续保留 v80。v95 已完成的实际编译、原 native/WMMA-only 验收和 E64 seed0/17 完整 MoE 原精度验证均保持通过，不重复运行。第一批六轮性能对照在第2轮测后检查失败，中止证据已下载并独立复核；没有 v95 的完整有效中值，不据此选版。继续只在 a07-3/dsr1_ep4 测 E64/T1536/topk8/shared0，性能固定 pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，选版只看孤立 GEMM1，完整 MoE 精度要求保持。

### v95 中止批次的原始证据

`f3_E64_LOG0_PID_CLOCK_v80_v95_wave_local_tail_kernel_timer_20261028_v1` 只完成2轮：

| 轮次 | 版本 | 原 GEMM1 timer μs | 原 controller valid |
|---:|---|---:|---|
| 1 | v80 | 82.04205263157893 | true |
| 2 | v95 | 81.3693684210526 | false |

第2轮直接 post 的 users/owners 均为空，VRAM 为176582656 B，busy=4，ownership_idle=true。3秒后 settled 的 `gpu_users.sh` 出现 PID1581465 的 `<已退出>` 记录，owners 为空，busy=2，ownership_idle=false、idle=false。这个 PID 与我方 host child PID1582652 不同；不能套用只允许我方已退出 PID 的例外。原 controller 因 `AssertionError('Retain external ownership/non-idle failure without replacement')` 中止。

该轮 CPU outer call window 为 `[75454977767329, 75455747664823] ns`，使用 `CLOCK_BOOTTIME`。原 CPU-only 诊断的 `external_owners=[]`，因此没有可比的外部进程创建时间，也未观察其 GPU 工作开始时刻。现有证据不能证明具体外部工作是否与 timer 重叠；仍按已冻结规则保留该轮无效状态，**81.3694 μs 不用于排名、收益或中值计算**。有效 v80 单轮82.0421 μs也不与其他中止批次拼接。

续接时核验 controller PID1411084、两轮我方 host child PID1566551/PID1582652 均已不存在，tmux pane 已回到 bash，生产源码恢复 v80 SHA256=`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。没有杀外部进程，没有残留我方测试进程。

完整中止归档147个成员、4784037 B，SHA256=`65b07ed2311700d43e3ab025936664c6db75650bb7a99b3fcd7c4bd1f01ba9f60`。本地独立 reviewer 通过，`performance_data=false`、rounds=2、原有效性公式和中止状态不变。该 label 不重新启动，不补轮，不拼接。

~~~powershell
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_CPU_window_kernel_timer_local_20261028_v2.py --label f3_E64_LOG0_PID_CLOCK_v80_v95_wave_local_tail_kernel_timer_20261028_v1
~~~

### 从已观测 E64 routing 计算 v95 跳过的无效工作量

静态分析读取前一份已核验 v80/v94 中止 archive 中实际 v80 E64 routing counts，没有生成或筛选新输入：12288 routed rows，各 expert 为168–221行，64个 expert 都只有一个 active M256 tile。

| expert 数量 | v80 两个 M-wave 的 rep | v95 两个 M-wave 的 rep |
|---:|---|---|
| 35 | 6/6 | 6/6 |
| 24 | 7/7 | 7/6 |
| 5 | 7/7 | 7/7 |

每一个 N256 tile、跨全部 expert 的 WMMA 工作量从370048降为359296，少10752条，即2.905569%。所有减少的 M16 block 都完全无效；所有有效行的覆盖保持。每个 expert 的最大 M-wave coverage 均保持，**工作量减少不是性能收益，也不能保证 WG 最长路径缩短**；共享发射资源和混合 body 的 I-cache 影响仍需实测。v95 `.text` 仍为97408 B，没有声称消除了加载地址敏感性。

~~~text
static_analysis=TASK/logs/f3_v95_observed_E64_route_work_static_local_20261028_v1.json
analyzer=TASK/analyze_f3_v95_observed_E64_route_work_local_20261028_v1.py
GPU_dispatch=false
performance_data=false
~~~

当前 v80 epilogue 已通过 `if wmb + wm * M_ROW_STEP < mn_oob` 跳过完全无效 M16 block 的 activation/quant。后续不能将这一既有行为作为新优化重复报告。

下一批使用新的独立 v2 label，重新冻结固定 v80/v95/v95/v80/v80/v95 顺序，复用已验收的同一候选、原 timer、warmup5/iters20、输入、精度和 post/settled 规则；旧批次两轮完全排除。先完成 LOG0 binding preflight，再重新累计全机连续11次、间隔6秒、跨度至少60秒的空闲门禁。性能 child 期间不运行 SFTP、额外 GPU 查询或远端并行编译，每个 compiler/test/profiler child 仍118秒 TERM、再2秒 KILL。E65/topk9 禁测继续覆盖正确性、性能和 profiler；历史共用 compiled GEMM1 要求与已授权 Git 最终收尾仍未完成。

### 归档 SHA256 文本更正

上节手工抄录的 v95 中止 archive SHA256 多写了一个字符。以下载后 manifest 与独立 reviewer 已核验的64位值为准：`65b07ed2311700d43e3ab025936664c6db75650bb7a9b3fcd7c4bd1f01ba9f60`。147个成员、4784037 B、2轮中止状态与全部有效性结论保持。归档字节和 verifier 从未修改；新计划冻结时此处 hash assertion 拒绝了错误抄录，未启动 child。更正后重新冻结新计划。

## 2026-10-28：v95 第二批中止，v98 行尾跳过与 retirement 保护通过完整 E64 验证

问：汇报下进展，然后继续。

答：生产仍保留 v80，没有新的完整 LOG0 性能中值。v95 新批次的前两轮有效：v80 为81.5864210526 μs，v95 为82.1280526316 μs；第3轮测后出现独立外部 GPU 任务，原规则拒绝该轮并中止整批。新候选 v96/v97 的实际 native 检查未通过，均未进入 GPU GEMM1 dispatch；v98 在完全跳过无效第七个 M16 block 时增加必要的 WMMA retirement 保护，已通过两套原 native model 及 E64 seed0/17 完整 MoE 原精度验证。v98 尚无性能数据，不据此换版。

### v95 v2 的固定批次与中止证据

| 轮次 | 版本 | 原 GEMM1 timer μs | 原 controller valid |
|---:|---|---:|---|
| 1 | v80 | 81.5864210526 | true |
| 2 | v95 | 82.1280526316 | true |
| 3 | v95 | 81.9995789474 | false |

固定顺序仍为v80/v95/v95/v80/v80/v95；仅完成3轮，没有完整中值。第3轮的直接 post users/owners 为空、busy=3，3秒后 settled 出现独立 PID2489626 的 `bwd_dqdkdv.out`，KFD/VRAM 由外部任务持有。该 PID 与我方 host child PID2491461 不同。CPU-only reader 随后遇到 PID 消失，创建时间关系保留为 unknown，不能证明其 GPU 工作是否与原 timer 重叠。81.9995789474 μs 保持无效，不排名，不与旧批次拼接或补轮。未终止外部任务。

161个归档成员、5663734 B，下载后的独立 reviewer 通过，performance_data=false、rounds=3。原 summary、所有 raw/source/IR/ELF/PID/clock 和有效性规则保持。

~~~text
interrupted_label=f3_E64_LOG0_PID_CLOCK_v80_v95_wave_local_tail_kernel_timer_20261028_v2
archive_SHA256=9a0d2462583202ddeaf70bfd6189800737ab705a732fc8537fd02f56c2525e59
independent_review=TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v95_wave_local_tail_kernel_timer_20261028_v2.CPU_window_independent_local_review.json
~~~

### 暂停恢复与 mixed-body 静态分析

v96 编译准备时只暂停了正在等待空闲的我方父 controller PID1746394、start_ticks=7623787；唯一直接 child 是空闲 helper PID1789531。续接时核验父进程仍为 T、身份不变、没有 performance child，v80 源码及旧计划全部冻结文件 SHA256 不变。helper 未完成旧空闲窗口，持续每6秒检查并记录新的 active 样本后，向同一父进程发送 SIGCONT。v96 没有 GPU dispatch；恢复后原第2轮确实运行并有效，第3轮因上述独立外部占用中止。controller 已退出，生产恢复 v80，没有遗留 SIGSTOP。

`f3_v95_mixed_body_static_local_20261028_v1.json` 绑定实际 ELF/ISA/IR：v95 有24个 expert 的两个 M-wave 分别走完整7/6-rep body，v80 为0个；六/七行 K-loop 的静态区间各占69/76个128 B line，union=145、重叠=0。静态区间不等于 live I-cache footprint；不能据此计算性能收益或唯一归因。已观测 routing 的无效 WMMA 减少2.905569%，各 expert 的最大 M-wave coverage 保持。这是 v98 保留共用7-rep body、只跳过完全无效行尾的动机。

### v96/v97 的失败与 v98 的修复边界

v96 保留共同4/6/7/8-rep selector，只在原 pin0/interleaved、active_m_rep==7、wm==6 且该 M16 block 完全无效时跳过 WMMA。0–256及扩展行数的覆盖证明保持；有效行的 per-row WMMA N 顺序、operands、ScaleA/ScaleB、累加和初始化 AST 与原版相同。没有新增 expert/token/seed specialization。

但实际 ISA 在PC17820/17828和21528/21536处，DS load 复用 `v640:647` 时，bounded model 仍保留先前 WMMA 对这些 operand 的使用。原 model 和 WMMA-only model 同时报告4处 `DS overwrite before WMMA retirement`。该 model 保持未放宽；它对条件 CFG 过近似，并不完整模拟 VALU/VMEM/EXEC/tensor，因此这里只报告 native gate 失败，没有声称已复现 device 数值错误，也不把它扩大为生产 v80 的 bug。

v97 仅将既有 pin0 六行 path 的显式 VA_VDST wait 从8收紧为4，pin1 原值8保持。实际 ISA 的上述风险点仍是编译器插入的 `s_wait_alu depctr_va_vdst(8)`，问题未消除，原 native gate 同样拒绝；没有运行 device 测试。

v98 回到v96，只在完全跳过第七个 M16 block 的 else 分支加入 `s_wait_alu depctr_va_vdst(0)`，并用两个 `rocdl.sched_barrier(0)` 约束编译调度边界。有效第七块的运算路径没有新增 wait；原 input reads、TDM load/store/barrier、既有 waits、调度 hints、activation/quant/output、准入及 fallback 源码保持。等待的是先前 WMMA retirement，确保后续 DS operand 复用安全。两套 actual native model 均通过，issues=0。

本地硬件依据为 `MI400_Shader_Programming#65.txt` §4.3.7.4/§4.3.7.4.1（正文91–93页）：VA_VDST 在 VALU issue 时递增、完成时递减；WMMA 与其他 VALU 的完成顺序以及 runtime 跳过会影响非零 wait 的有效距离，SCHED_MODE 2 要求软件保护这些依赖。此处选择跳过分支的 zero wait，再以实际 ISA 和 device 验证验收；没有通过放宽 model 或精度绕过问题。

| 版本 | .text B | VGPR | SGPR | compile child秒 | native 结果 |
|---|---:|---:|---:|---:|---|
| v96 | 97920 | 799 | 60 | 19.994539 | 拒绝，4处 operand retirement 风险 |
| v97 | 97920 | 799 | 60 | 19.793725 | 拒绝，4处 operand retirement 风险 |
| v98 | 98176 | 799 | 60 | 20.094749 | 两套通过，issues=0 |

三版 LDS 均327680 B、private/spills 均0；上表秒数是非性能 compiler child 用时。v98 的 .text 比v80增加768 B，仍跨64 KiB，不声称已消除全部加载地址敏感性。

~~~text
v96_source_SHA256=8c1c4628e692fec7041a03f75200cdf75361eae589bd2cc159c24f579a699e77
v96_ELF_SHA256=bf53f71f5f25b48c4927200ba6d8745dbb11636634ade8fede6796e0e3d3376a
v97_source_SHA256=60e0c0633d2d2cc80ea6d7ba510a2c747d9f65c39fb425f64e7bbd269f78dce7
v97_ELF_SHA256=78f947340b6e5a307f8b505c17eb67984112529522025eaeba2412daff506769
v98_source_SHA256=6c72e2b8ea00adfc7fff2215d0a8fff39f285b6ab11dfaba7f8612b26e74ed52
v98_ELF_SHA256=9f0e466031d3d4fb57ee09d89f5751b54d0b40513dcf4d4c08ed6ed32ee00c8b
v98_native_review=TASK/logs/f3_target_v98_shared_tail_skip_retirement_pin0_cpu_20261028_v1_native.review.json
~~~

### 完整 E64 MoE 精度验收

只运行E64/T1536/topk8/shared0、random non-balanced，pin=0、A-preshuffle=1。原 gate=0.01保持，两组完整 BF16 输出各22020096 B，均与原 reference 逐字节一致；已验收的 GEMM1 cache SHA256/mtime 保持，GEMM2 完整 ELF 与基线相同。正确性允许在外部 GPU 占用时运行，全部打印的耗时排除出性能记录。

| seed | logits_diff | rel_l2 | differing_bytes |
|---:|---:|---:|---:|
| 0 | 3.6105513278e-06 | 0.00268720998429 | 0 |
| 17 | 3.62026933163e-06 | 0.00269081653096 | 0 |

~~~text
device_summary=TASK/logs/f3_v98_shared_tail_skip_retirement_complete_MoE_pin0_20261028_v1.summary.json
device_summary_SHA256=09f0ccd3452e3b5ad6d28ebdc1a2e768fed0c12b42799899fbc9194452352189
GEMM2_ELF_SHA256=4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be
~~~

### 继续原 GEMM1 timer 的独立性能对照

新计划固定v80/v98/v98/v80/v80/v98，仅E64/topk8/shared0，pin=0、A-preshuffle=1、AITER_LOG_MORE=0。复用已检查的原 caller/timer、warmup5/iters20、输入及精度，先做两版 LOG0 binding preflight，再重新累计全机连续11次、间隔6秒、跨度至少60秒的空闲门禁。performance child 中没有 SFTP、额外 GPU 查询或远端并行编译；每个 compiler/test/profiler child 仍为118秒 TERM、再2秒 KILL。

~~~text
label=f3_E64_LOG0_PID_CLOCK_v80_v98_shared_tail_skip_retirement_kernel_timer_20261028_v1
frozen_plan_SHA256=e0bf323777e354c698f38de82f0b91f62fdcb2e83d57b05bf65a7d1177978f31
performance_result=pending
~~~

首次验证/对照命令如下，首次写入的证据使用独立 label；已有中止归档不重新启动：

~~~bash
python3 TASK/run_f3_v80_followup_compile_correctness_20261028_v1.py --plan TASK/logs/f3_v98_shared_tail_skip_retirement_source_20261028_v1.plan.json --stem f3_target_v98_shared_tail_skip_retirement_pin0_cpu_20261028_v1
python TASK/review_f3_v80_followup_native_local_20261028_v1.py --stem f3_target_v98_shared_tail_skip_retirement_pin0_cpu_20261028_v1
python3 TASK/run_f3_v80_followup_device_E64_only_20261028_v1.py --cpu-stem f3_target_v98_shared_tail_skip_retirement_pin0_cpu_20261028_v1 --stem f3_v98_shared_tail_skip_retirement_complete_MoE_pin0_20261028_v1
python TASK/freeze_f3_v98_CPU_window_local_20261028_v1.py
python TASK/review_f3_CPU_window_kernel_timer_local_20261028_v2.py --label f3_E64_LOG0_PID_CLOCK_v80_v98_shared_tail_skip_retirement_kernel_timer_20261028_v1 --check-sources-only
python3 TASK/run_f3_v80_followup_LOG0_binding_preflight_E64_20261028_v1.py --plan TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v98_shared_tail_skip_retirement_kernel_timer_20261028_v1.frozen_plan.json
python3 TASK/run_f3_original_kernel_timer_CPU_WINDOW_E64_LOG0_PID_20261028_v2.py --plan TASK/logs/f3_E64_LOG0_PID_CLOCK_v80_v98_shared_tail_skip_retirement_kernel_timer_20261028_v1.frozen_plan.json
~~~

后续仅a07-3/dsr1_ep4，E65/topk9 禁止测试仍覆盖精度、性能和 profiler。按孤立 GEMM1 耗时选版，完整 MoE 精度必须保持；历史E64/E65共用 compiled GEMM1 要求仍未完成。已授权英文 commit/push、主机 pull 与 pull 后最终 E64 复测仍待最终选版后执行。

## 2026-10-28：当前最优版本与新的三次中值选版流程

问：到目前为止，性能最好的版本是哪个？

答：当前已验收并保留的 GEMM1 最优版仍为 **v80，pin=0**。最近 `AITER_LOG_MORE=0` 的有效单轮为81.5864210526 μs，来自v80/v95第二批第1轮；这是单轮，不是完整三次或九次中值。该批第3轮测后遇到外部 GPU 任务而中止，原始记录保持，不拼接其他批次。v98 已通过两套原 native 检查和 E64 seed0/17 完整 MoE 原精度、逐字节输出验收，尚无性能结果，不能称为更快版本。

用户先提出“九次测试取中值，有收益就立即替换正式源码”，随后更新为：

> 候选跑3次取中值有收益，就做同期正反向 control，确定有收益后，跑3次取中值性能归档。

执行最新指示，九次中值方案不再使用。设 A 为当前正式版、B 为候选版，流程为：

1. 初筛：B 做三次有效测试取中值。当前 LOG0 尚无新的完整 A 中值，本轮先测同条件 A 三次作为初筛 reference。
2. 初筛有收益后，完成同期正向 A→B→A、反向 B→A→B，每组各三次。初筛 A/B 两组直接作为同一新冻结计划的正向开头，不重跑、不拼接历史批次。正向 B 中值须低于前后两组 A，反向前后两组 B 中值须低于中间 A。
3. 两个方向确认收益后，再做 B 三次取中值性能归档。若归档中值相对最近同期 A 不再有收益，记录实际回退，不挑快轮、不补测来取得较好归档。
4. 完成以上验收和原始记录核验后，立即将本地及 a07-3 正式源码替换为 B；后续优化从 B 派生。MoE e2e 耗时不作为采用条件，完整 MoE 原精度要求保持。

只测 E64/T1536/topk8/shared0、model_dim7168/inter_dim2048、A4W4/SiLU、random non-balanced；性能固定 pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，使用原 caller/timer、warmup5/iters20、gate0.01。E65/topk9 禁测覆盖正确性、性能和 profiler。只用 a07-3/dsr1_ep4，每个 child 118秒 TERM、2秒 KILL。每轮前按既有规则累计全机连续11次、间隔6秒、跨度至少60秒空闲门禁；外部占用时自动等待，不杀外部任务；计时期间不做 SFTP、额外 GPU 查询或远端并行编译。

原v80/v98六轮冻结计划已完成LOG0 binding preflight，两子进程均退出0且小于120秒，正式源码恢复v80，但未启动任何性能轮。该计划保留并不再启动。新冻结计划为 `f3_E64_LOG0_v80_v98_screen_ABA_BAB_archive3_20261028_v1`，最多21轮，仅在组边界按上述已冻结收益规则继续或停止；没有修改原 timer、输入、精度或 kernel 计算。

## 2026-10-28：三次中值流程已启动，GPU 等待期间的实际 ISA 核对

问：汇报下进展，然后继续。

答：正式版仍为 v80，pin=0，没有新的有效性能中值。v98 已通过原 native/WMMA-only 两套检查及 E64 seed0/17 完整 MoE 原精度、逐字节输出验收。新流程已启动：三次初筛，有收益才补同期正向 A→B→A、反向 B→A→B，每组各三次，通过后再三次取中值归档并立即更新正式源码。旧九次中值方案和未启动性能的旧六轮计划均不再使用。

截至已有门禁日志的 UTC 2026-10-28 14:13:38，新 controller 仍等待第一轮全机空闲，实际性能子进程和计时轮均为0。a07-3 有外部短任务及其他 grouped GEMM 测试接续使用 GPU；我方持续每6秒等待，不杀外部进程、不在占用期间记录性能。等待/controller 总时长与单个 GPU test child 的120秒上限分开；没有正在运行超过两分钟的我方 GPU test。性能条件仍仅为 E64/T1536/topk8/shared0、random non-balanced、pin=0、A-preshuffle=1、AITER_LOG_MORE=0、原 timer/warmup5/iters20/gate0.01。

本地静态分析逐条核对实际 ISA 的机器字与已冻结的完整 ELF，v80/v98 均通过；未编译新 kernel、未运行 GPU 或 profiler。结果如下：

| 静态机器码项目 | v80 | v98 | v98 相对 v80 |
|---|---:|---:|---:|
| .text bytes | 97408 | 98176 | +768 |
| decoded instructions | 12729 | 12903 | +174 |
| encoded WMMA instructions | 2000 | 2000 | 0 |
| VA_VDST=0 wait instructions | 133 | 195 | +62 |
| conditional branches | 132 | 156 | +24 |

上述是编码数量，既不是动态执行数量，也不是 kernel 耗时。源码的无效 M16 block guard 已位于整个 `mma_one_row` 调用外，不存在把该 guard 逐 N-column 重复编写的问题。实际差异集中于七行 body 的部分循环区间：v80 两处56-WMMA回边区间含4条零计数 wait，v98 对应区间含15条；尾部区间由3条变为13条。v98 同时存在共享前缀的52/56-WMMA回边区间，区间会重叠，不能相加为动态工作量。四/六/八行区间的这些计数保持。必须等待原 timer 初筛，才能判断减少无效 WMMA 与增加等待/控制流的净影响，不能仅依据源码减少计算就宣称收益。

硬件资料依据：`mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`，§4.3.7.4/§4.3.7.4.1，正文91–93页。SCHED_MODE 2 将 VA_VDST/VM_VSRC 的部分依赖保护交给软件；运行时因 EXEC 跳过的 VALU 不会增加 VA_VDST，改变 EXEC 后可能被跳过的指令不能计入非零 wait 的距离。这里仅用于解释为何必须保留 retirement 验收，并非证明 v98 的等待时间或性能。

本地证据位于 `.codex_tmp/hyg_f3_followup_20261024`：`logs/f3_v80_v98_static_tail_wait_layout_local_20261028_v1.json`、`logs/f3_v80_v98_static_WMMA_backward_regions_local_20261028_v1.json`。复现静态机器字和计数核对：

```powershell
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v80_v98_static_tail_wait_layout_local_20261028_v1.py
```

首次证据已存在时，不覆盖原 JSON；使用独立输出 label 复核。新的正式源码替换工具已经准备并通过语法检查，尚未执行；只有 control、归档和原始数据核验通过才会采用候选。

## 2026-10-28：v99 编译后被 retirement 检查拒绝，继续 v98 的三次中值初筛

问：汇报下进展，然后继续。

答：当前正式源码仍为已验收的 **v80，pin=0**。v98 已通过原 native/WMMA-only 两套检查及 E64 seed0/17 完整 MoE 原精度、逐字节输出验证，但截至恢复性能 controller 的 UTC 2026-10-28 14:49:50，仍为0个性能轮，没有新的有效中值，不能将 v98 称为更快版本。此前81.5864210526 μs是独立旧批次的 v80 单轮，不能充当本轮三次 reference 或与本轮拼接。

等待期间完成了 v99 的编译和实际 ISA 验收尝试：从 v80 派生，保留 v96 的完整无效 M16 block guard，不加入 v98 的 skipped-row else wait，改为在七行 `load_lds_data` 路径已有的 `s_wait_alu depctr_va_vdst(0)` 两侧加 pin0 `rocdl.sched_barrier(0)`。原准入、fallback、每个有效值的 WMMA 数学运算与舍入、输入 TDM、activation、quant 和输出功能保持；源级证明不代替机器码或设备验证。

编译准备 child 19.793567577秒、ISA inspection child 0.113849996秒，均退出0并小于120秒。目标 GEMM1 为 compile-only，打印耗时不属于性能数据。资源为 `.text=97920 B`、VGPR799、SGPR60、LDS327680 B、零 spills。完整 ELF SHA256 为 `e14d157c9942b9fa2048348d3ddfd18c136ed8f2af5478be5e439cddc42465e3`。

原 native model 在3.878571400秒内退出1，保留4处 `DS overwrite before WMMA retirement` 报告：PC17820/17828复用 VGPR640–647，PC21528/21536复用同组 VGPR。失败位置与此前 v97 相同，但逐字节比较确认两版 `.text` 并不相同，不能声称整个机器码没有改变。v99 编码的 VA_VDST=0 wait 为166条，低于 v98 的195条，但这是静态数量，既不能证明动态等待减少，也不能替代正确性验收。原 native model 未修改或放宽；它是保守的 DS/WMMA model，并未完整建模 VALU、VMEM、EXEC skips 和 tensor completion。这里据其失败拒绝候选，不将 model 报告表述为已观察到的设备错误。

因此 v99 **不进行目标 GEMM1 dispatch、不运行完整 MoE 或性能测试**，也不声称精度正确或性能更快。正式源码已恢复 v80，源码 SHA256 仍为 `93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。下一步优先完成已通过精度的 v98 初筛，再处理有实际 ISA 证据支持的 retirement 边界。

为编译临时 `SIGSTOP` 的性能 controller PID3127720/start_ticks8265441已恢复，未遗留暂停。恢复前核验 controller、唯一空闲 helper PID3127733/start_ticks8265445、全部冻结文件 hashes、主机分支/HEAD/boot及正式源码；helper仍在连续每6秒检查，尚未返回 `ALL_GPUS_IDLE`，最新一轮门禁仍有外部 grouped GEMM 测试，因此未消费过时的已完成空闲窗口。恢复 receipt 为 `logs/f3_v99_compile_resume_v98_three_stage_20261028_v1.json`。正式性能计划保持 `f3_E64_LOG0_v80_v98_screen_ABA_BAB_archive3_20261028_v1`，未修改或重新启动旧批次。

后续流程仍为候选三次中值初筛、有收益才完成同期 A→B→A / B→A→B control（各组各三次），确认后再候选三次中值归档；通过后立即替换两端正式源码，下一步从新最优版派生。只测 E64/T1536/topk8/shared0、random non-balanced；pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`、原 caller/timer/warmup5/iters20/gate0.01。完整 MoE 原精度必须保持；MoE e2e 耗时不作为采用条件。E65/topk9 禁测不变。所有性能轮等待全机连续11次/间隔6秒/至少60秒空闲，外部占用自动等待，不杀外部进程；计时期间不做传输、额外 GPU 查询或远端并行编译。

本地证据位于 `.codex_tmp/hyg_f3_followup_20261024`：

- `logs/f3_target_v99_shared_tail_existing_retirement_fence_pin0_cpu_20261028_v1.summary.json`
- `logs/f3_target_v99_shared_tail_existing_retirement_fence_pin0_cpu_20261028_v1_native_original.json`
- `logs/f3_v99_failed_native_wait_inspection_local_20261028_v1.json`
- `analyze_f3_v99_rejected_retirement_local_20261028_v1.py`

编译复现只在 a07-3 主机启动，实际 child 由 controller 进入 dsr1_ep4；Git 始终在主机运行。证据 stem 已存在时使用独立 stem，不覆盖原记录：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v80_followup_compile_correctness_20261028_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_v99_shared_tail_existing_retirement_fence_source_20261028_v1.plan.json \
  --stem f3_target_v99_shared_tail_existing_retirement_fence_pin0_cpu_20261028_v1
```

硬件依据仍为 `MI400_Shader_Programming#65.txt` §4.3.7.4/§4.3.7.4.1（正文91–93页）。软件在 SCHED_MODE 2 下负责必要的依赖保护；源码 scheduling hint 不是实际机器码顺序或 retirement 完成的证明。

已授权的英文 commit/push、远端主机 pull及 pull 后最终 E64 复测继续保留，待最终选版后完成。历史 E64/E65 共用 compiled GEMM1 要求仍未建立，本轮不误报为已经完成。

## 2026-10-28：v100 共享 k_step retirement 边界的实际验收结果

问：汇报下进展，然后继续。

答：v100 已完成编译，尝试在原 pin0 七行 `k_step` 的全部 WMMA 调用之后、`reuse_cur_rmem` 的后续 load 之前设置一个共享 `s_wait_alu depctr_va_vdst(0)` 边界；不改变 v96 的无效行 guard、有效值计算、原准入和 fallback。源码生成器核验了移除新增边界后 `k_step` 的 AST 与原版一致，`mma_rows` 的 AST保持，正式源码未由生成器修改。

编译 child 19.893721518秒、inspection child 0.163851399秒，均退出0、小于120秒，目标 GEMM1 仍为 compile-only。资源为 `.text=97408 B`、VGPR816、SGPR60、LDS327680 B、零 spills；代码大小和 resources 不是性能数据。源码 SHA256 为 `4bec61d51d6977c55bd6ec6905bd2aa06afedb004a666f766344aa60c6e49a89`，完整 ELF SHA256 为 `5b55f1537fe4cdbfba5be31b765082a037f0c1a59df4922853b99ae49c0246b1`。

原 native model 在3.259987600秒内退出1，仍报告4处 `DS overwrite before WMMA retirement`：PC17944/17952与PC21688/21696，复用 VGPR656–663。逐条 ISA 机器字核验通过；其中 PC17936 是 `s_wait_alu depctr_va_vdst(8)`，紧接着 PC17944/17952 的 `ds_load_b128` 复用了此前 WMMA 读取的 VGPR656–663。这一候选新增的共享源码边界未覆盖 model 报出的复用位置，不能只依据其在源码中位于末尾就判定安全。

因此 v100 同样拒绝进入目标 GEMM1 dispatch、完整 MoE和性能测试。没有放宽 native model、设备精度或计时标准，也没有对失败版本补测来获得性能数字。正式版仍保留 v80；v98 已有原精度验收，但仍须等待全机空闲后完成新的三次中值及同期 control 流程。仅 E64/T1536/topk8/shared0，pin=0、`AITER_LOG_MORE=0`，禁止 E65/topk9 测试不变。

证据位于 `.codex_tmp/hyg_f3_followup_20261024/logs`：`f3_target_v100_shared_tail_kstep_retirement_boundary_pin0_cpu_20261028_v1.summary.json`、`f3_target_v100_shared_tail_kstep_retirement_boundary_pin0_cpu_20261028_v1_native_original.json`，以及对应带 manifest SHA256 的编译 evidence archive。native model 的保守范围及完整建模限制同 v99 记录；这里不声称已观察到设备错误或新性能收益。

## 2026-10-28：v101 的 wait 参数优化通过两套原 native 和完整 E64 精度验证

问：汇报下进展，然后继续。

答：已完成 v101 的实际编译与正确性验收。它以当前正式 v80 为基准，沿用 v98 对完全无效 M16 block 的跳过逻辑，只将 skipped-tail 分支内的 `s_wait_alu depctr_va_vdst(0)` 改为 `depctr_va_vdst(4)`，两侧原 scheduling barrier 保留。源码核验了仅一个 wait 参数发生变化、每个有效值的 `mma_one_row` AST 完全相同，可逆恢复到已检查的 v98；未添加专家数量或输入数据特例。

这次候选没有像 v99/v100 一样移动 wait 的源码位置，而是在已有正确路径上测试较小的等待范围。该 nonzero wait 只有通过原 retirement model、WMMA-only stress 和原设备精度后才具备候选资格，不从“参数较大”直接推断其安全或更快。

- 编译准备 child 19.793534180秒、ISA inspection child 0.163842949秒，均退出0、小于120秒，目标 GEMM1 为 compile-only。
- 原 native model 4.585463400秒、WMMA-only model 4.058342100秒，均退出0，issues均为0；两套原 model 的 transfer规则未修改。
- 完整 E64/topk8/shared0 MoE seed0/17 验证分别8.476469432秒、8.276192498秒退出0；两个22020096 B BF16输出均通过原精度，`differing_bytes=0`，与 reference逐字节一致。
- GEMM2准备 child 13.684177831秒退出0，完整 GEMM2 ELF仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`；已验收 GEMM1 cache的内容和mtime未改变。

这些为编译、model和精度 child的墙钟耗时，均排除在性能数据之外；正确性允许外部GPU任务占用。v101的GEMM1正式性能尚未测量。

实际 ISA 机器字与完整 ELF逐条绑定后，对比 v98/v101：PC集合完全相同，只有12条 wait从 `VA_VDST=0` 改为 `4`，`.text`恰好12个字节不同，其余全部 decoded instructions保持相同。编码的zero wait数量从195降为183，VA_VDST=4 wait从8增为20；这只是静态编码计数，不能作为动态执行次数或性能收益。两版`.text`同为98176 B、VGPR799、SGPR60、LDS327680 B、零 spills。v101完整GEMM1 ELF SHA256为 `aa79348099a55e5d238eef327c96666dab5ae090cc6007d43f354c13c19a0cb8`，源码SHA256为 `716d9f1a9cb3e2008fa936347256da47abe4cdf46517a5754bfccacbb438f50a`。

当前状态如下：

| 版本 | 原 native / WMMA-only | 完整 E64 原精度 | 本轮正式性能 | 当前用途 |
|---|---|---|---|---|
| v80 | 已验收 | 已验收 | 尚未形成新的三次中值 | 保留的正式最优版，pin=0 |
| v98 | 通过 / 通过 | seed0/17逐字节一致 | 等待全机空闲，尚无新结果 | 先完成既有三次中值初筛 |
| v99 | 原 model失败，4个issue | 未运行 | 未运行 | 拒绝候选 |
| v100 | 原 model失败，4个issue | 未运行 | 未运行 | 拒绝候选 |
| v101 | 通过 / 通过 | seed0/17逐字节一致 | 尚未运行 | 下一项已准备的候选 |

v101验证完成后，两端正式源码仍为v80。原性能controller已于UTC2026-10-28 15:20:00恢复，PID3127720/start_ticks8265441的state为S；唯一空闲helper PID3127733/start_ticks8265445仍连续每6秒检查，未返回已完成空闲窗口。恢复前重新核验全部冻结hashes、branch/HEAD/boot、v101设备child均退出、validation pane回到bash及正式源码；没有消费过时窗口，没有遗留SIGSTOP。恢复receipt为 `logs/f3_v101_compile_resume_v98_three_stage_20261028_v1.json`。

另已冻结并完成source-only reviewer检查的v101预备计划为 `f3_E64_LOG0_v80_v101_screen_ABA_BAB_archive3_20261028_v1`，SHA256 `77e27f2f85c797dc94793648da1507ad6c670d0aa84ba6ef1246b2c4d803145e`。该计划未启动，既有v98计划未改写；须先结束v98比较。如果v98通过control和归档而被采用，下一轮以采用后的新正式版重新冻结reference，不能继续使用v80 reference或导入旧批次的计时。三次初筛→同期A→B→A / B→A→B各组三次→候选三次归档的规则保持；通过收益和原始记录核验后立即替换正式源码。

后续性能仍仅E64/T1536/topk8/shared0、model_dim7168/inter_dim2048、random non-balanced、pin=0、A-preshuffle=1、`AITER_LOG_MORE=0`，原caller/timer/warmup5/iters20/gate0.01。每轮前全机连续11次、间隔6秒、至少60秒空闲；占用自动等待，不杀外部进程；计时期间不做传输、额外GPU查询或远端并行编译。E65/topk9禁测覆盖精度、性能和profiler。MoE e2e原精度继续保留，耗时不作为采用条件。

本地与远端证据目录仍为 `.codex_tmp/hyg_f3_followup_20261024`：`logs/f3_target_v101_shared_tail_skip_retirement_va4_pin0_cpu_20261028_v1_native.review.json`、`logs/f3_v101_shared_tail_skip_retirement_va4_complete_MoE_pin0_20261028_v1.summary.json`、`logs/f3_v98_v101_actual_wait_parameter_delta_local_20261028_v1.json`。本地复现原native检查：

```powershell
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v80_followup_native_local_20261028_v1.py `
  --stem f3_target_v101_shared_tail_skip_retirement_va4_pin0_cpu_20261028_v1
```

已有证据不覆盖；再次复现使用独立stem和对应已打包的编译evidence。完整精度复现仍在a07-3主机由 `run_f3_v80_followup_device_E64_only_20261028_v1.py` 进入dsr1_ep4执行。硬件依据仍为 `MI400_Shader_Programming#65.txt` §4.3.7.4/§4.3.7.4.1（正文91–93页）；nonzero VA_VDST涉及WMMA与其他VALU的顺序，不能只数源码指令，必须保留原model和设备验证。

## 2026-10-28：v98 计划已有第一条有效 reference，继续等待连续空闲窗口

问：汇报下进展，然后继续。

答：正式最优版仍为 v80、pin=0。已接续原有 `f3_E64_LOG0_v80_v98_screen_ABA_BAB_archive3_20261028_v1` 冻结计划，没有另起一批或导入历史计时。第一轮空闲门禁于 UTC `2026-10-28 15:22:29` 返回 `ALL_GPUS_IDLE`；r01/v80 的原 timer GEMM1 为 **81.99336842105262 μs**，`valid=true`、returncode=0，child 墙钟时间7.985971183秒。原 kernel 和完整 MoE 的 `logits_diff` 均为 `3.6105513278025183e-06`，测试 child 已退出，settled 检查为空闲。该数值只有一轮，不能作为三次中值、归档性能或候选收益。

第二轮在等待全机连续60秒空闲。既有门禁日志显示其他用户短 GPU 任务和尚未结束的 grouped GEMM 测试接续出现；controller 继续每6秒检查，不杀外部进程，不使用占用期间的性能数据。没有重写冻结计划、插入替代轮次或修改既有性能有效性条件。计时期间禁止 SFTP、额外 GPU 查询和远端并行编译。

本地静态分析进一步明确了 v98/v101 的收益范围。已保存的 routing 指纹与本轮 r01 的指纹相同：`ids_SHA256=a543dfbcfe5976bc42809fbe751880b996a3ca88d2e95175a204e74a35fc6a6c`、`weights_SHA256=99940dfb31ef88a29570641fa6138adf0dfb2f0fdef2abcba67a5cd4d040a117`。64个 experts 共12288条 routes；35个 expert 使用共同六行 body，29个使用共同七行 body，其中24个可以跳过第七个完全无效 M16 block。

| 工作量模型 | v80 | v98 | v101 |
|---|---:|---:|---:|
| 相同 N、K 范围下的 M16 block 计数 | 826 | 802 | 802 |
| 完全无效 block 的跳过量 | 0 | 24 | 24 |
| 相对 v80 的 WMMA 工作量减少比例 | 0 | 2.905569% | 2.905569% |

这是已绑定输入和源码的工作量模型，**不是动态 counter 或新的 kernel 性能数据**。保留共同七行 pipeline、原输入读取、TDM/barrier、有效值的 math/rounding 和原 epilogue；新增 branch、retirement wait 以及 code layout 的实际成本仍需原 timer 检验，不能将2.91%直接转换成 μs。v98/v101 都已通过两套原 native model 和 E64 seed0/17 完整 MoE 原精度，各 BF16 输出逐字节一致；这次没有重复设备验收。v99/v100 的原 native model 拒绝状态保留，不放宽检查。

接下来先完成当前 v98 的三次初筛、同期正反向 control 和三次归档。收益通过独立本地 reviewer 后立即替换两端正式源码；没有收益则保留 v80。v101 的冻结预备计划已准备，但没有上传或启动：v98 结束后，按最终正式版本确认其 reference，不能在 v98 已采用时沿用 v80 reference。后续仍只测 E64/T1536/topk8/shared0、random non-balanced，固定 `AITER_LOG_MORE=0`、pin=0、原 caller/timer/warmup5/iters20/gate0.01；不测 E65/topk9，不按 MoE e2e 耗时选版。

本地新增静态证据为 `.codex_tmp/hyg_f3_followup_20261024/logs/f3_v98_v101_observed_work_scope_local_20261028_v1.json`，复现工具为同目录 `analyze_f3_v98_v101_observed_work_scope_local_20261028_v1.py`。当前 frozen plan SHA256仍为 `ceeeeadeaa59e87e8ec3896c1fde352226a5eb389020aa2857d8c9a9699b1ac0`。文档按两端各自已有历史做 CAS 追加，远端 v101 问答及本节待当前性能计划结束后的空档同步，未覆盖远端旧内容。

## 2026-10-28：空闲等待期间，本地核查 skipped-tail wait 的假设范围

问：汇报下进展，然后继续。

答：v98 的原冻结性能计划继续等待第二轮所需的全机连续空闲窗口，正式版仍是 v80/pin=0，本轮仍只有第一条有效 v80 GEMM1=81.99336842105262 μs，尚无三次中值。没有 SFTP、远端并行编译或额外 GPU 查询干扰性能 child；controller 保持每6秒自动复查，外部任务保留。

这段时间完成了一项本地 CPU 诊断：从已验收的 v101 完整 ELF/ISA 读取真实 PC 和机器字绑定，只在内存中的 model 输入里，把12处 skipped-tail 的 `VA_VDST=4` 参数分别假设为5、6、7、8，然后运行原 DS/WMMA transfer model 及已有 WMMA-only stress。没有修改 ELF、ISA log、生产 kernel、model 的原 transfer 规则或冻结性能计划。

| 假设 wait 参数 | 原 bounded model | WMMA-only stress | 设备精度 / 性能 |
|---|---|---|---|
| `VA_VDST=5` | issues=0 | issues=0 | 未运行 |
| `VA_VDST=6` | issues=0 | issues=0 | 未运行 |
| `VA_VDST=7` | issues=0 | issues=0 | 未运行 |
| `VA_VDST=8` | issues=0 | issues=0 | 未运行 |

8个 CPU worker 均退出0、各自小于5秒；hypothetical input 的每条非目标 wait 指令均保持原样，原 input files 保持原字节。这说明在这份实际 v101 指令布局的有限 DS/WMMA model 中，5–8没有被该检查拒绝；**不意味着已得到可采用的新 kernel，也不证明完整 VALU/VMEM/EXEC/tensor 行为或性能**。没有将8称为安全上限，因为更大的值尚未检查。任何后续源码候选仍须重新编译、验证实际 ISA 布局、通过两套原 model 和 E64 完整 MoE 原精度，再按用户规定的三次初筛/control/三次归档计时；不能以本地假设结果替换这些验收。

诊断输入完整 GEMM1 ELF SHA256为 `aa79348099a55e5d238eef327c96666dab5ae090cc6007d43f354c13c19a0cb8`，ISA log SHA256为 `c6357bfc2307b2f0bd1ccbeae4e715089f85a91701714d680a652cd1c091f908`。未修改的原 model SHA256为 `87ba3e65485b01a063fe78c7f2120cad00e320a2a9018bd9229dee1ec7fc21b7`。新本地诊断工具和 summary 为 `.codex_tmp/hyg_f3_followup_20261024/analyze_f3_v101_hypothetical_wait_bounds_local_20261028_v1.py`、`logs/f3_v101_hypothetical_wait_bounds_local_20261028_v1.summary.json`。硬件依据仍是 `MI400_Shader_Programming#65.txt` §4.3.7.4/§4.3.7.4.1，正文91–93页；WMMA 与其他 VALU 的退休顺序及非零 wait 必须按资料和实际依赖检查，不凭静态指令数或固定 cycle 数推断安全。

当前顺序保持：先结束 v98 的既有计划，再根据最终正式版推进已经验收精度的 v101；本次 CPU 诊断没有另起 GPU 性能候选或提前采用版本。只测 E64/topk8/shared0，pin=0、`AITER_LOG_MORE=0`、原 timer 和精度标准不变。已授权的 Git commit/push/远端 pull/最终 E64 复测仍在实验闭合后执行，不改动当前绑定的 HEAD。

## 2026-10-29：v102 的实际 wait 参数候选完成 native 与 E64 原精度验收

问：汇报下进展，然后继续。

答：正式版仍为 v80、pin=0。当前 v98 冻结性能批次只有 r01/v80 一轮有效 GEMM1 计时 **81.99336842105262 μs**，尚无三次中值，也没有 v98 的有效性能结果。期间其他用户短 GPU 任务接续出现，原空闲 helper 每6秒继续检查；没有杀外部进程、修改原 timer 或用占用期间的耗时选版。

在没有 performance child 的维护窗口中，只暂停了自己的性能 controller（PID3127720/start_ticks8265441），原空闲 helper 保持运行。已在该窗口完成 v102 的实际源码编译与正确性验收。v102 基于当前正式 v80，沿用已验收 v98/v101 的完全无效尾 M16 block guard，仅将 v101 skipped-tail 路径的 `s_wait_alu depctr_va_vdst(4)` 改为 `depctr_va_vdst(8)`，两侧 scheduling barrier 保留；有效值的 `mma_one_row` AST、原准入、fallback、输入读取、TDM/barrier、activation、quant 和输出逻辑保持。未添加专家数量、seed、route 或 const0 特例。

| 验收项 | 实际结果 |
|---|---|
| GEMM1 compile-only / ISA inspection | 19.894388279秒 / 0.163803330秒，均退出0 |
| 原 bounded native model / WMMA-only stress | 4.505462900秒 / 4.263837500秒，均退出0、issues=0 |
| GEMM2准备 | 13.934690299秒，退出0，完整 ELF 与原版相同 |
| E64/topk8/shared0 完整 MoE seed0 / seed17 | 8.426841974秒 / 8.226322767秒，均退出0、原精度通过 |
| 两份完整 BF16输出 | 各22020096 B，与各自 reference逐字节一致，`differing_bytes=0` |
| 已验收 GEMM1 cache | 内容 SHA256和mtime保持 |
| 正式源码及本轮测试进程 | 已恢复v80；validation pane回到bash，没有v102测试进程残留 |

以上秒数均为编译、model或正确性 child的墙钟时间，**不是 kernel性能数据**。所有 child均小于120秒，实际命令使用TERM118秒/KILL2秒。完整输出的逐字节比较由未改变原精度规则的 verifier在child内执行；该 verifier在compare模式下不保存候选 `.moe.bin`。归档保留结果JSON、原日志、相关ELF和验证源码，不将未保存的候选原始payload误报为本地重新比较过的文件。

v102源码 SHA256：`9de32536882f4b298785033ce065828f042c273a1476ca3d6e1b7281fd929ca1`；完整GEMM1 ELF SHA256：`681d9fc91b32d00168865344084bbb6c65d4e5ef8a115cc669669b6434a28959`。实际 `.text=98176 B`，VGPR799、SGPR60、LDS327680 B、private0、VGPR/SGPR spills0。完整GEMM2 ELF仍为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。

本地逐条绑定实际机器字后对比v101/v102：PC集合完全相同，只有12条 `VA_VDST=4→8` 的wait不同，`.text`恰好12个字节不同，其余decoded instructions、dispatch ABI和resources均保持。两套原model的transfer规则没有放宽。该证据确认了实际编译变化范围，不能单凭nonzero wait或静态计数证明更快；完整VALU/VMEM/EXEC/tensor行为也不能由有限native model单独证明，设备原精度验收仍保留。

接续顺序保持：先结束原v98计划，再推进已经验收精度的v101，v102作为后续候选准备，不提前采用。收益按用户最新流程：候选三次中值初筛→同期正反向control→候选三次中值归档；通过原始证据核验后立即替换两端正式源码，并从新正式版继续派生。若reference变化，重冻后续计划，不拼接旧轮次。

本次维护后的恢复使用已冻结resumer：先等原helper自然完成，再执行完整11次/至少60秒新空闲检查，保留旧r02 pre-log字节前缀并追加真实新观察，原reviewer通过后才恢复同一controller。当前尚在完成证据传输和文档追加，不能提前称controller已恢复，也不直接SIGCONT消费旧空闲窗口。

后续性能仅E64/T1536/topk8/shared0、model_dim7168/inter_dim2048、random non-balanced；固定`AITER_LOG_MORE=0`、pin=0、A-preshuffle=1、原caller/timer/warmup5/iters20/gate0.01。计时期间不做SFTP、额外GPU查询或远端并行编译；禁止所有E65/topk9测试。MoE e2e原精度继续保留，耗时不作为选版条件。

证据目录为 `.codex_tmp/hyg_f3_followup_20261024`：`logs/f3_target_v102_shared_tail_skip_retirement_va8_pin0_cpu_20261029_v1_native.review.json`、`logs/f3_v102_shared_tail_skip_retirement_va8_complete_MoE_pin0_20261029_v1.summary.json`、`logs/f3_v101_v102_actual_wait_delta_local_20261029_v1.json`。本地实际ISA比较工具为 `analyze_f3_v101_v102_actual_wait_delta_local_20261029_v1.py`；已有证据不覆盖。硬件依据仍为 `MI400_Shader_Programming#65.txt` §4.3.7.4/§4.3.7.4.1（正文91–93页），nonzero VA_VDST须考虑WMMA与其他VALU的顺序及EXEC跳过条件。

## 2026-10-29：等待性能窗口期间，检查v102的9–15假设wait参数

问：汇报下进展，然后继续。

答：v102实际编译、两套原native检查及E64 seed0/17完整MoE原精度均已完成，结果已追加两端三份记录，正式版仍为v80/pin=0。原v98性能计划的恢复程序已启动，但仍等待原空闲helper结束；该helper继续每6秒检查全机，外部短GPU任务持续打断空闲窗口。当前仍只有r01/v80的81.99336842105262 μs一轮有效reference，没有候选中值；未额外启动GPU实验、传输或远端编译。

在本地完成了另一项CPU诊断：绑定已编译v102完整ELF `681d9fc91b32d00168865344084bbb6c65d4e5ef8a115cc669669b6434a28959`及逐条机器字一致的ISA，以v101/v102实际差异证明定位12处skipped-tail wait，只在model的内存输入里分别将`VA_VDST=8`假设为9、10、11、12、13、14、15。14个worker（各参数的原model及WMMA-only stress）均退出0、issues=0，单worker约3.67–5.22秒；原transfer规则、ELF、ISA log、正式源码及既有报告保持原字节。

结果说明这些参数在当前布局的有限DS/WMMA model中均未被拒绝，**不能把8称为模型已证明的安全上限，也不能据此称更大的值更快或可直接采用**。假设参数只存在于内存，没有实际编译成新候选，没有设备精度或性能数据。即使更大参数在model中通过，后续仍须重新从当时正式版派生源码、检查实际编译顺序、通过两套原model和E64完整MoE原精度，再做三次初筛、同期正反向control和三次归档；不会用本地假设结果替换这些条件。

硬件资料`MI400_Shader_Programming#65.txt` §4.3.7.4.1（正文92–93页）记载`VA_VDST`的4个LSB可供`s_wait_alu`等待0..15，并说明EXEC全零跳过VALU以及SALU修改EXEC对wait计数的影响。§4.3.7.4同时提醒WMMA与其他VALU的完成顺序。上述可编码范围与实际依赖安全是不同问题；有限model不完整覆盖VALU/VMEM/EXEC/tensor行为，不能从参数可编码或静态指令数量直接推断安全。

本地工具为`.codex_tmp/hyg_f3_followup_20261024/derive_f3_v102_hypothetical_wait_probe_local_20261029_v1.py`及其生成的`analyze_f3_v102_hypothetical_wait_bounds_local_20261029_v1.py`；summary为`logs/f3_v102_hypothetical_wait_bounds_local_20261029_v1.summary.json`。本项新证据及本节先在本地保存，远端追加待当前性能计划闭合后的空档；此前v102实际验收证据两端均已保存。

后续性能队列和约束保持：先结束v98，再推进v101；reference改变时重新冻结。只测E64/T1536/topk8/shared0、random non-balanced、pin=0、`AITER_LOG_MORE=0`、A-preshuffle=1；原caller/timer/warmup5/iters20/gate0.01，禁测E65/topk9，按GEMM1耗时选版并保留完整MoE原精度。当前暂不改HEAD，已授权的commit/push/远端主机pull/最终E64复测继续保留在实验闭合后完成。

## 2026-10-29：v103尾行统一predicate的实际ISA被原native model拒绝

问：汇报下进展，然后继续。

答：正式源码仍为v80/pin=0。冻结的v80/v98性能计划仍只有r01/v80一轮有效reference，GEMM1为81.99336842105262 μs；没有三次中值或v98候选成绩。v102已完成两套原native model及E64 seed0/17完整MoE原精度验证，等待性能验收。此次在原性能controller暂停、没有performance child的维护窗口检查v103，失败后保留原报告并拒绝该候选。

v103沿用v102尾M16 block跳过逻辑与wait8，尝试在K loop前用`rocdl.readfirstlane`产生wave内一致的`tail_m16_valid`并用于原guard。源码生成器核对了有效计算`mma_one_row`的AST，未新增expert/token/data specialization；正式parent仍为v80。实际compile child退出0，耗时20.044403615989722秒，ISA inspection退出0，耗时0.16381966599146836秒。编译阶段不属于性能测试，目标GEMM1未dispatch；原input preparation允许使用GPU，不能把整个编译准备过程称为完全不使用GPU。

| 项目 | v102 | v103 |
|---|---:|---:|
| `.text` B | 98176 | 98688 |
| VGPR | 799 | 799 |
| SGPR | 60 | 60 |
| LDS B | 327680 | 327680 |
| VGPR/SGPR spills | 0/0 | 0/0 |

v103的原native worker退出1，保留的报告列出两处`DS overwrite before WMMA retirement`：PC23236写VGPR680–683、PC23244写VGPR684–687。对应实际ISA为PC23148的WMMA读取v[680:687]，随后PC23228执行`s_wait_alu depctr_va_vdst(8)`，PC23236/23244的`ds_load_b128`覆写上述寄存器。这是原有限model检出的retirement风险，未运行设备精度验证，未观察或宣称设备数值错误。没有改model、放宽精度或改写失败报告；不继续此版的WMMA-only验收、设备验证及性能测量。

该改写也没有实现预期的静态分支减少。相对v102，`v_readfirstlane_b32`数量差为0，`s_cbranch_scc0`增加2条、`s_cbranch_scc1`增加6条、`s_cbranch_vccz`增加8条、`s_wait_alu`增加35条；`.text`增加512 B。以上为绑定实际ELF的静态计数，不能换算成动态次数或kernel耗时。

v103源码SHA256为`a371a0e40c484ded776af9d7cf3804797017d649a2c5bcb3a4b05db286ec09c4`，GEMM1 ELF为`d0a00af6f1e5a4e98468030d93f456cb8774ef76a9b92ba88b9df9fa22cdd7e0`。compile evidence archive的SHA256为`9454084dd9d30dcdba9ad61491a7495eeb02995d99a34518660641caa6ece3f2e`，已在本地核对。原报告保存在`logs/f3_target_v103_uniform_tail_predicate_pin0_cpu_20261029_v1_native_original.json/.log`，失败分析保存在`analyze_f3_v103_rejected_uniform_tail_predicate_local_20261029_v1.py`及`logs/f3_v103_rejected_uniform_tail_predicate_local_20261029_v1.json`，均位于`.codex_tmp/hyg_f3_followup_20261024/`。

硬件资料`MI400_Shader_Programming#65.txt` §4.3.7.4/§4.3.7.4.1（正文91–93页）明确要求expert scheduling mode由软件保证依赖，并说明EXEC跳过对非零`VA_VDST`等待值的影响。CDNA5 ISA §15.8（正文326–327页）记载`V_READFIRSTLANE_B32`读取最低active lane的值。这些资料支持检查实际ISA依赖及predicate一致性；不能仅凭源码中的readfirstlane或v102的wait参数通过model，就推断不同编译布局的v103安全。

本节及失败证据保存后，只恢复既有resumer，让它在原helper自然结束后完成新的全机11次/至少60秒空闲门禁，再恢复原master；不重启冻结计划、不覆盖r01、不直接恢复master。后续先结束v98性能计划，再推进已完成原精度验收的v101/v102；reference改变时重新冻结。只测试E64/T1536/topk8/shared0、random non-balanced、pin=0、`AITER_LOG_MORE=0`、A-preshuffle=1，保留原caller/timer/warmup5/iters20/gate0.01，禁测E65/topk9，按GEMM1耗时选版并保持完整MoE原精度。

## 2026-10-29：v103失败证据已保存，原resumer恢复，v102性能计划在本地核验完成

问：汇报下进展，然后继续。

答：v103失败报告、实际ISA分析及上一节问答已同步至a07-3并分别核对SHA256，三份文档在两端各自保留历史前缀后追加成功。正式源码仍为v80/pin=0，两端SHA256均为`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。

传输和文档追加结束后，按pause receipt核对boot、master/resumer的PID、start_ticks、UID、argv、children、正式源码、原summary及resumer脚本hash，只向原resumer PID1082628/start_ticks9211415发送SIGCONT；其状态由T恢复为S。没有直接恢复master，也没有重新启动既有计划。远端恢复receipt为`logs/f3_v103_validation_restore_original_resumer_20261029_v1.json`。原resumer继续监督master，并等待原空闲helper自然结束后再执行新的完整11次/至少60秒空闲门禁。

最近保存的空闲日志仍记录其他用户的FMHA/bwd短任务、非零GPU busy及间歇KFD owner，连续空闲窗口反复重置。原v80/v98计划仍只记录r01/v80的81.99336842105262 μs一轮有效数据，r02仍在空闲等待，尚无v98候选三次中值。本轮没有据此宣称性能收益、替换正式源码、重启旧批次或杀外部进程。

等待期间仅在本地完成v102性能计划准备与源码独立核验，未执行GPU测试或远端并行编译：

```powershell
python .codex_tmp/hyg_f3_followup_20261024/freeze_f3_v102_three_stage_CPU_window_local_20261029_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_three_stage_CPU_window_kernel_timer_local_20261028_v1.py --label f3_E64_LOG0_v80_v102_screen_ABA_BAB_archive3_20261029_v1 --check-sources-only
```

两条命令最终均退出0。新冻结计划SHA256为`1e5116bc6fcbdb39480a6a50335a6621560ec735eee3aea55723162e0a7aec58`；源码核验receipt为`logs/f3_E64_LOG0_v80_v102_screen_ABA_BAB_archive3_20261029_v1.CPU_window_sources_local_review.json`，明确`performance_data=false`、`candidate_selected=false`、rounds=0。计划绑定v102已完成的实际compile、两套原native model、seed0/17完整E64 MoE原精度、原caller/timer、两阶段实际ELF/cache与所有相关源码hash；最多21轮，保留三次初筛、同期正反向control、三次归档及原有效性规则。

v102计划目前仅在本地保存，未上传、未binding preflight、未启动。先闭合当前v98计划，再推进既有v101计划；如果前一候选通过验收并被采用，会先从新正式版重新派生、冻结reference，再推进后续候选，不把当前v80 reference计划直接用于新parent。v103维持拒绝状态。

本节先在本地三份文档追加；远端同步留到已有性能计划闭合后的空档，避免与计时并行传输。性能约束保持E64/T1536/topk8/shared0、random non-balanced、pin=0、`AITER_LOG_MORE=0`、A-preshuffle=1、原warmup5/iters20/gate0.01，禁测E65/topk9；完整MoE原精度保持，按孤立GEMM1耗时选版。

## 2026-10-29：v104实际wait候选通过E64精度，补齐VA15 sentinel语义检查

问：汇报下进展，然后继续。

答：正式源码仍为v80/pin=0，SHA256为`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。原v80/v98冻结性能计划仍只有r01/v80一轮有效GEMM1计时81.99336842105262 μs，没有候选成绩或三次中值，不能据此判断收益。此次维护窗口暂停的是自己的原resumer；master已经暂停，原空闲helper继续每6秒检查全机，没有重启性能计划或杀外部进程。

v104从已验收v102派生，唯一源码变化是skipped-tail branch的`s_wait_alu depctr_va_vdst(8)`改为`depctr_va_vdst(15)`，两侧scheduling barrier保留。有效行的math、原guard、TDM/barrier、activation、quant、输出及原准入保持，未采用被原model拒绝的v103 predicate hoist。候选源码SHA256为`a52775a8d2af0d11d0eb791dd0b57a0f15051dc28f9af11c3af5750431784352`，完整GEMM1 ELF为`1b38a9884eb3618c8932efa95b636f3ba3eba60203d4e8343ac9f0e615a80abe`。

| 验证项 | 实际结果 |
|---|---|
| GEMM1 compile / ISA inspection | 19.894秒 / 0.164秒，均退出0 |
| 原native model / 原WMMA-only stress | 均退出0、issues=0，原报告保留 |
| 补充VA15 sentinel审计：original / WMMA-only | 4.612秒 / 4.647秒，均退出0、issues=0 |
| E64/T1536/topk8/shared0，完整MoE seed0 / seed17 | 均退出0、原精度通过，各22020096 B，与各自reference逐字节一致 |
| seed0 / seed17 `logits_diff` | 3.6105513278025183e-06 / 3.6202693316254594e-06 |
| `.text` / VGPR / SGPR / LDS | 98176 B / 799 / 60 / 327680 B |
| private / VGPR spills / SGPR spills | 0 / 0 / 0 |
| GEMM2完整ELF及已验收GEMM1 cache | GEMM2 ELF保持；GEMM1 cache内容及mtime保持 |

表中的秒数是compile、审计或正确性child的墙钟时间，全部小于120秒，**不是kernel性能数据**。设备child保留TERM118秒/KILL2秒；正确性允许在外部GPU占用时执行，打印的计时全部排除于性能统计。没有执行任何E65/topk9测试。

这里发现并修正的是审计解释：`MI400_Shader_Programming#65.txt` §4.3.7.4、Table 37（正文91页）明确写道最大wait field值表示“don't wait on this counter”。因此`VA_VDST=15`禁用该counter的等待，不能解释成等待counter≤15。原有限model把所有数值统一用于retirement过滤，对15的处理不符合该sentinel语义。此前14组9–15内存假设探测及v104两套原model的通过报告保持原样；其中15的旧结果不足以单独证明安全，不改写旧报告为新的结论。

补充审计绑定v104实际ELF和逐条机器字一致的ISA，只在model的内存输入中移除12处`VA_VDST=15`的retirement作用，保留其他字段和原文件；original模式的transfer AST保持，另做WMMA-only更保守检查。两套补充检查仍通过，issues=0。该结果与设备原精度一起支持把v104保留为待测候选；有限DS/WMMA model不覆盖完整VALU/VMEM/EXEC/tensor语义，也不证明性能收益。

原literal比较脚本因为一侧反汇编省略default字段、另一侧展开全部default字段而失败，失败日志继续保留。新encoded-field比较逐条绑定实际机器字，确认v102/v104的PC集合相同，只有12处`s_wait_alu`的bits15:12由8改为15，`.text`恰好12个字节不同，其余decoded instructions、非VA字段、dispatch ABI及resources保持。v104对应机器字为`BF88FF9F`；没有将整条指令称为nop，`HC`字段仍保留。

两组完整MoE输出在原compare child内执行了逐字节比较，`differing_bytes=0`，原gate及精度标准没有改动。compare模式没有保存候选`.moe.bin`，本地核验的是原日志、比较receipt、绑定源码及ELF，不声称重新比较了未保存的候选raw payload。

CPU evidence archive SHA256为`fc46ca9ad944267635ed0d490d1d47c0b800c1d75caf4e07e47035f9d580e1b7`。新增device与VA15证据archive SHA256为`9d80cf01f0e6fde593a03c7a99d3afdeba93ec94076ddbd1e878082b25efaa14`，45个成员在本地逐一核验通过；两份archive两端均已保存，既有失败及原通过报告没有覆盖。证据位于`.codex_tmp/hyg_f3_followup_20261024/`：

- `logs/f3_v104_VA15_disabled_counter_retirement_audit_local_20261029_v1.summary.json`
- `logs/f3_v102_v104_encoded_wait_field_comparison_local_20261029_v1.json`
- `logs/f3_v104_E64_validation_independent_local_review_20261029_v1.json`
- `logs/f3_v104_shared_tail_skip_retirement_va15_complete_MoE_pin0_20261029_v1_evidence.tar.gz`

结论：v104完成本轮实际compile、补充有限依赖审计及E64完整MoE精度验收，**没有性能成绩，未采用**。文档和证据传输结束后，只按本轮pause receipt核对身份并恢复既有resumer，由它等待原helper自然结束，再完成新的11次/至少60秒全机空闲门禁后恢复同一master；不直接恢复master、不重启已冻结计划、不覆盖r01。

后续先闭合v98，再推进已准备v101/v102；v104保留为后续候选。若正式reference改变，先重新派生并冻结后续计划。选版保持三次中值初筛→同期正反向control→三次中值归档，验收有收益后立即替换两端正式源码，以孤立GEMM1耗时选版，完整MoE原精度保持。只测E64/T1536/topk8/shared0、random non-balanced、pin=0、`AITER_LOG_MORE=0`、A-preshuffle=1，原caller/timer/warmup5/iters20/gate0.01不变。计时期间不做SFTP、额外GPU查询或远端并行编译。已授权的英文commit/push/远端主机pull/最终E64复测仍待选版闭合后执行，当前不改冻结HEAD。


## 2026-10-29：v104证据同步完成，原resumer已恢复，后续性能计划在本地准备完毕

问：汇报下进展，然后继续。

答：正式最优版仍为v80、pin=0，当前原v80/v98冻结计划只有r01/v80一轮有效GEMM1计时81.99336842105262 μs，没有候选结果或新的三次中值。v104实际ELF、E64 seed0/17完整MoE精度、VA15 sentinel补充审计和12字节wait-field比较已完成，device archive的45个成员在本地逐一核验通过；两端三份记录分别CAS追加成功，没有覆盖不同的历史前缀。

本轮所有传输及文档追加结束后，恢复程序在远端主机核对boot、branch/HEAD、冻结文件hash、原summary、正式源码，以及master/resumer/helper的PID、start_ticks、UID、argv与children。只向原resumer PID1082628/start_ticks9211415发送SIGCONT，其状态由T恢复为R；没有直接恢复master或新建resumer，没有重启既有计划。远端receipt为`logs/f3_v104_validation_restore_original_resumer_20261029_v1.json`，SHA256为`e746eb4f8068371b04f16e1c7f5cf6f3b631694a32675d88ba563378e6b4d517e`，程序退出0、`passed=true`。

原resumer继续等待旧空闲helper自然结束，再执行新的全机11次/至少60秒空闲门禁，满足原review规则后才恢复同一master。最新只读状态仍为`maintenance_completed=false/controller_resumed=false/error=null`，性能summary仍只有r01。这表示等待门禁，不表示恢复程序失败。最近原pre-log继续记录外部KFD owner、短GPU busy及VRAM升高，连续空闲窗口被打断；原helper继续每6秒检查。没有杀外部进程或修改计时、有效性规则、轮次。恢复后不再进行SFTP或远端并行编译。

等待期间仅在本地完成v104后续性能计划冻结及独立源码核验，未运行GPU测试：

```powershell
python .codex_tmp/hyg_f3_followup_20261024/freeze_f3_v104_three_stage_CPU_window_local_20261029_v1.py
python .codex_tmp/hyg_f3_followup_20261024/review_f3_three_stage_CPU_window_kernel_timer_local_20261028_v1.py --label f3_E64_LOG0_v80_v104_screen_ABA_BAB_archive3_20261029_v1 --check-sources-only
```

两条命令均退出0。新计划SHA256为`1748671b73341542f6ad13000eff3460b0149b53844a893fac4e2a99f38172f5`；源码review receipt为`logs/f3_E64_LOG0_v80_v104_screen_ABA_BAB_archive3_20261029_v1.CPU_window_sources_local_review.json`，`passed=true/performance_data=false/candidate_selected=false/rounds=0`。计划绑定实际GEMM1/GEMM2 ELF、原caller/timer、E64原精度、原native报告以及新增的VA15 sentinel审计和独立device review，不把原model对15的旧解释作为唯一安全依据。

该计划最多21轮：三次reference和三次候选初筛，收益成立才继续同期正反向control和三次归档；原post规则、warmup5/iters20/gate0.01、pin=0、`AITER_LOG_MORE=0`、A-preshuffle=1均保持。v104未获选、没有性能数据。计划目前仅在本地保存，未上传、未remote binding preflight、未启动。

接续顺序保持v98→v101→v102，再根据当时正式源码推进v104；如前一版本通过完整性能验收，立即采用，并从新reference重新派生、冻结后续计划，不混用旧reference或拼接历史轮次。只测E64/T1536/topk8/shared0、model_dim7168/inter_dim2048、random non-balanced，以孤立GEMM1耗时选版并保持完整MoE原精度。禁测全部E65/topk9。已授权的Git add、英文commit/push、远端主机pull及pull后的最终E64复测仍待当前选版计划闭合后执行，`.codex_tmp`不暂存，原staged及`op_tests/...`修改保持。

本节仅在本地三份文档追加；远端同步留到原性能计划闭合后的空档。远端已保存上一节v104实际验证及问答，不在可能进入计时的阶段传输文件。


## 2026-10-29：v105未通过依赖检查，后续性能测试统一warmup2/iters101

问：汇报下进展，然后继续。

答：正式源码仍为v80、pin=0，SHA256为`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。此前v80/v98计划只有r01/v80一轮有效GEMM1计时81.99336842105262 μs，没有候选成绩或三次中值。该轮使用旧warmup5/iters20，只作为原参数历史记录，不用于新参数下的收益比较。

v105从已验证的v98/VA0派生，在共同7-rep body中对整数`mn_oob`使用`readfirstlane`，保留原signed comparison、有效行WMMA、输入读取、waits、TDM/barrier、activation、quant、输出及原准入。没有重复已被拒绝的Boolean predicate scalarization，也没有让同expert的两条M-wave分别选择完整6/7 body。

实际compile和ISA inspection分别耗时20.094597041010275秒、0.1638416630012216秒，均退出0；这些是墙钟诊断时间，不是kernel性能。实际GEMM1 ELF SHA256为`ea33045ba35ec4cb2d2a5026c088cb7a46caf1f15bfe35dfb3483d1658b9988c`。与v98相比，`.text`由98176 B增至98560 B，增加384 B；dispatch ABI、VGPR799、SGPR60、LDS327680 B保持，private及两类spills均为0。protected static operations的计数保持，不能据此声称动态计算量不变或性能提高。

两套有限native检查都未通过，在PC23076、23084报告`DS overwrite before WMMA retirement`，涉及VGPR680–687。original模式保留原transfer AST；WMMA-only模式报告相同两处问题。原model SHA256仍为`87ba3e65485b01a063fe78c7f2120cad00e320a2a9018bd9229dee1ec7fc21b7`，没有放宽检查规则。该model不覆盖完整VALU/VMEM/EXEC/tensor语义，因此这里是候选未满足现有验收条件，不能当作设备上已观察到结果错误的证据。

v105已拒绝，不运行其设备精度测试和性能测试，不替换正式源码。compile archive的20个成员在本地逐一核验通过，archive SHA256为`1e1510b5ca54dd5d58131de03106a2ff4b21328d27096b3f10793c51374aed91`。原失败报告及实际ELF保持；追加的codegen/rejection receipt为`logs/f3_v105_uniform_integer_bound_codegen_and_rejection_local_20261029_v1.json`。未执行任何E65/topk9测试。

问：以后每次执行性能测试，warmup和iters选用`--warmup 2 --iters 101`。

答：从此统一执行这组参数。最新用户指令覆盖之前warmup5/iters20的约束；因此不能恢复仍会使用旧参数的master/resumer，也不能修改旧冻结计划后继续混用r01。

已核验只属于本任务的master PID3127720、resumer PID1082628和原idle helper PID109251的UID、start_ticks、argv与children，确认没有performance child。随后取消这三个自己的控制/等待进程；没有向外部进程发送信号。原master通过其signal/finally流程恢复v80，原r01逐项保留，没有新增旧参数测试。取消脚本的最终`/proc`读取遇到进程恰好退出而中断，其初始receipt和失败保留；后续只读closure不重发signal，确认三者均不存在、原rounds保持、正式源码恢复。closure为`logs/f3_old_timer_plan_cancellation_W2_I101_independent_closure_20261029_v1.json`。

旧计划及原证据已保存为interrupted archive，148个成员，SHA256为`e6eba4aedc2b704f6e53014d20f4ee5217eb263795c8c62445635c2ae740f4eb`。旧summary在取消前另存原字节快照；旧结果不被重新解释为新参数数据。上一轮v104本地冻结的warmup5/iters20计划同样不启动；v101/v102/v104在实际推进前必须重新冻结为warmup2/iters101。

新建v80/v98计划`f3_E64_LOG0_W2_I101_v80_v98_screen_ABA_BAB_archive3_20261029_v1`，SHA256为`82476e3375d97711230eb9b7eec7e73e4e9fbc362ec0ff38ffb3dd051d83d70e`。本地freeze与独立源码review均退出0，`performance_data=false/candidate_selected=false/rounds=0`。新入口要求显式传入`--warmup 2 --iters 101`，在CLI、plan、实际传入原`caller.run_moe`的kwargs及结果receipt中核验一致；原caller、`run_perftest`实现、partial对象、计时方式及原精度gate0.01保持。

新计划从全新的reference和候选开始：每组三次取中值初筛，收益成立才执行同期正向A/B/A和反向B/A/B control，最终候选三次中值归档。每个字母代表同版本的三次测试。验收有收益后立即替换两端正式源码，再从新正式版继续；以孤立GEMM1耗时选版，完整MoE原精度必须通过，MoE e2e耗时不作为采用条件。

远端实际入口为：

```bash
python3 .codex_tmp/hyg_f3_followup_20261024/run_f3_three_stage_LOG0_binding_preflight_E64_W2_I101_20261029_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_W2_I101_v80_v98_screen_ABA_BAB_archive3_20261029_v1.frozen_plan.json \
  --warmup 2 --iters 101

python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_three_stage_original_kernel_timer_CPU_WINDOW_E64_LOG0_PID_W2_I101_20261029_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_W2_I101_v80_v98_screen_ABA_BAB_archive3_20261029_v1.frozen_plan.json \
  --warmup 2 --iters 101
```

Git仍在远端主机执行，实际kernel child仍在a07-3/dsr1_ep4运行；只测E64/T1536/topk8/shared0、model_dim7168/inter_dim2048、A4W4/SiLU、random non-balanced，固定`AITER_LOG_MORE=0`、pin=0、A-preshuffle=1。性能前全机连续11次空闲检查、每6秒一次、至少60秒；外部占用自动等待。实际compile/test/profiler child保持TERM118秒/KILL2秒。计时期间不进行SFTP、额外GPU查询或远端并行编译。

本节记录的是本地准备与旧计划取消，尚无warmup2/iters101新性能结果。两端文档继续独立CAS追加，保留各自不同的历史前缀。最新用户改动优先，原resumer恢复要求在本次被新的timer参数指令替代。已授权的Git add、英文commit/push、远端主机pull和pull后的最终E64复测仍待选版闭合后完成。


## 2026-10-29：warmup2/iters101 的实际 binding preflight 通过，继续等待正常性能对照

问：汇报下进展，然后继续。

答：两端正式源码仍为 v80/pin=0，SHA256 为 `93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。新计划 `f3_E64_LOG0_W2_I101_v80_v98_screen_ABA_BAB_archive3_20261029_v1` 已在 a07-3 主机的 tmux window `codex-f3-v3-layout-20261027:w2-i101-perf` 启动，master PID3098328；没有重启旧计划或恢复旧 resumer。

实际 binding preflight 在 dsr1_ep4 中完成，两个 child 均退出0：

| 版本 | child 墙钟时间 s | kernel logits_diff | 完整 MoE logits_diff |
|---|---:|---:|---:|
| v80 | 7.975845162989572 | 3.6105513278025183e-06 | 3.6105513278025183e-06 |
| v98 | 8.025844493997283 | 3.6105513278025183e-06 | 3.6105513278025183e-06 |

这些是入口绑定与正确性验证，`performance_data=false`，没有执行性能计时；墙钟时间不能作为 kernel 耗时。CLI、plan、传入原 caller 的 warmup/iters 及结果 receipt 均为 `(2,101)`，`AITER_LOG_MORE=0`、pin=0、A-preshuffle=1、原精度 gate0.01 保持。实际 eager/partial/ELF 绑定通过；本地独立 review 核验了 summary 和两份 child receipts，尚未下载或重新比较 preflight raw ELF。

preflight summary SHA256：`1cf634f91189de333886a6dc2f8e8e38bb16f24c4cb3502c12439c6d2d52f724`。独立 review 为 `TASK/logs/f3_E64_LOG0_W2_I101_v80_v98_screen_ABA_BAB_archive3_20261029_v1.W2_I101_binding_independent_local_review.json`。

截至 `2026-10-28T19:28:37+00:00` 附近的已保存 gate 日志，新计划仍在 r01/v80 的连续空闲门禁，rounds=[]、groups=[]、error=null，尚无 warmup2/iters101 性能数据。外部 `bwd.out` 短任务及间歇非零 busy/KFD owner 反复重置空闲窗口。更早的 root PID3167826 属于另一个仓库的 E96 counters 任务，不能归为本轮 checker；未向这些外部进程发送信号。既有 helper 每6秒继续检查，确认连续11次、至少60秒全机空闲后才自动开始性能 child；计时期间不另行查询 GPU、传输文件或远端并行编译。

等待期间在本地分析了 preflight 保存的真实 E64 seed0 routing：35个 expert 使用共同6-rep body，29个使用共同7-rep body。原 M16 计算块合计826，v98 跳过24个完全无效块，占 `24/826 = 2.905569%`；全部完全无效块为29个，另有5个属于6-rep body。该比例只计 M 维块，N/K 复制因子在比值中抵消，不能直接换算成 kernel 耗时收益。分析 receipt 为 `TASK/logs/f3_E64_real_route_M16_invalid_work_analysis_local_W2_I101_20261029_v1.json`，没有 dispatch GPU、修改 kernel 或计时方法。

当前优先闭合 v98 对 v80 的正常计时：三次中值初筛，有收益才完成同期正向 A/B/A、反向 B/A/B control，各字母代表三次测试，随后候选三次中值归档。验收后立即更新两端正式源码，再从新的正式版继续；尚未采用 v98。后续 v101/v102/v104 只能在新的 warmup2/iters101 计划下推进，正式 reference 改变时重新派生比较。只测 E64/T1536/topk8/shared0、random non-balanced，禁测全部 E65/topk9；完整 MoE 原精度保持，以孤立 GEMM1 耗时选版。

本节先在本地三份文档独立 CAS 追加。远端同步及上述两份本地分析 receipts 的上传留到性能计划结束的空档，避免同步与可能开始的计时重叠。


## 2026-10-29：后续 wait 候选完成 warmup2/iters101 的本地计划准备

问：汇报下进展，然后继续。

答：等待 a07-3 的连续空闲窗口期间，在本地完成了 v101、v102、v104 的新参数性能计划冻结和 source-only review。三份计划均为 E64/T1536/topk8/shared0、random non-balanced、pin=0、`AITER_LOG_MORE=0`、`warmup=2 / iters=101`，最多21轮，保留三次初筛、同期正反向 control、三次归档，以及原 caller/timer/precision 和有效性规则。未启动这些计划，未上传它们，未运行新 GPU 测试，未改动当前 v98 计划或两端正式源码。

| 候选 | 新 frozen plan SHA256 | source-only review |
|---|---|---|
| v101 | `b7109e7d101e7969bf58ba2d100896fc1f2f85e4b77c8e8a808dd51cc0d9b2b1` | passed |
| v102 | `09a6266b0daa71db7d5ac07e47c2f0082b9f20290908fea70c07e585f6e65802` | passed |
| v104 | `bbc6cebb1224eab59fa0cdf249b0e8ad2bd9dc0b54bfed4bf6db290204f7ad0e` | passed |

本地入口为 `TASK/freeze_f3_wait_candidate_W2_I101_local_20261029_v1.py --candidate v101|v102|v104`。实际逐项执行 freeze 和 `TASK/review_f3_three_stage_CPU_window_kernel_timer_W2_I101_local_20261029_v1.py --label <对应新label> --check-sources-only`，最终均退出0。生成器首次检查旧计划时发现旧计划没有显式 warmup/iters 字段，因此改为同时检查原 checker AST 中硬编码的 `(5,20)`；失败发生在写新计划和任何 GPU 操作之前，没有放宽新参数检查。原旧计划和原 checker 保持不变。

新计划 label 均为 `f3_E64_LOG0_W2_I101_v80_<candidate>_screen_ABA_BAB_archive3_20261029_v1`，并显式要求当前 W2/I101 v98 计划先结束。它们仍以当前正式 v80 为 reference；若 v98 或后续版本通过验收并被采用，使用前必须从新的正式 reference 重新派生并冻结，不能沿用本节的 v80 对照，也不能导入旧参数计时。v104 保留实际 VA15 sentinel 补充审计的绑定，未将 VA15 解释为 counter≤15。

准备汇总为 `TASK/logs/f3_wait_candidates_W2_I101_preparation_local_20261029_v1.json`，`performance_data=false/candidate_selected=false/rounds_run=0`。当前先完成正在运行的 v98 比较，再依实际正式版推进后续候选；不把本地准备记为性能提升。本节先在本地三份文档独立 CAS 追加，远端同步留到当前性能计划结束的空档。


## 2026-10-29：v106/v107 热区共用 body 编译实验失败，恢复 W2/I101 对照前的收尾

问：汇报下进展，然后继续。

答：正式源码仍为 v80/pin=0，未采用新候选。等待 a07-3 的空闲窗口期间，尝试把 v98 的两份 6/7-rep 热区 body 合为共用 7-rep body，再只调整 selector 的分支树。两项实验均编译完成，但都产生了 scratch spill，未通过既有零 spill 资源检查。没有 dispatch 目标 GEMM1，没有运行设备精度、native retirement 或性能测试；不能把机器码缩小记为性能提升。

| 版本 | `.text` B | VGPR | SGPR | LDS B | private B | `vgpr_spill_count` | 实际 scratch ISA |
|---|---:|---:|---:|---:|---:|---:|---|
| v98 已验证参考 | 98176 | 799 | 60 | 327680 | 0 | 0 | 无 |
| v106 共用 7-rep body | 80000 | 812 | 60 | 327680 | 20 | 20 | 10 条 `scratch_store_b64`、9 条 `scratch_load_b64` |
| v107 重排 selector | 79232 | 797 | 60 | 327680 | 36 | 56 | 28 条 `scratch_store_b64`、27 条 `scratch_load_b64` |

v106 保留第七个完全无效 M16 块的 guard 和 skip-path VA0 retirement，保留 4/8 cold leaves、pin1/non-interleaved fallback、有效行计算和原准入。原 6-rep 区间每 k128/wave 增加 3 条 DS reads。源码检查覆盖 4353 个 row-count cases；这只说明源码分支和行覆盖检查通过，不能替代设备精度。v107 只改为先判断 `4 * M_ROW_STEP < mn_oob`，然后判断 `7 * M_ROW_STEP >= mn_oob`，所有 leaf calls 和 mainloop/epilogue/sync 定义与 v106 相同，truth table 检查覆盖 4355 cases，包含 int32 边界。

编译和 ISA inspect child 均退出 0：v106 墙钟分别为 `18.392138881987194 / 0.1639324099960504 s`，v107 为 `18.491826907003997 / 0.11385168400011025 s`；outer 随后因资源 gate 失败退出，并恢复远端 v80。这些墙钟不是 kernel 性能。v106 的 scratch sites 位于静态 64-WMMA 循环组前后的准备/载入区域，没有落在该 backward interval 内部；现有证据不能唯一确定哪个 live range 导致 spill，不能称为已确定的 C accumulator 或 LDS address spill。

两份失败 archive 各含 20 members，已下载并在本地逐项核验；原 dispatch ABI、实际 ELF/ISA 绑定、源码 plan 和恢复状态均通过审核，失败状态保留，没有重跑替换失败实验：

- v106 archive SHA256：`69201636097f4626fd14569b9155b8ca92a78d78aab5d6ec217322ac5d847aae`；GEMM1 ELF SHA256：`8bba13e5355e1b9b0c8ab2aec5597266c94e6504cf4b4900b7ac5ebe21e82533`。
- v107 archive SHA256：`651c672fca046365bbcbb8d40b7a2b4dafd4ad867d921e0700d67ddd5d0f0ee5`；GEMM1 ELF SHA256：`fd93ea54dc749a48263894a8cf499c9a53ef1d4548a882215f1bec3707a15d11`。
- 独立审核：`TASK/logs/f3_target_v106_common_seven_hot_body_pin0_cpu_20261029_v1_failed_resource_independent_local_review.json` 和 `TASK/logs/f3_target_v107_common_seven_selector_tree_pin0_cpu_20261029_v1_failed_resource_independent_local_review.json`。

为避免编译/传输与可能启动的计时重叠，此轮只对已核验身份的新 W2/I101 master PID3098328（start_ticks=10126753、UID12144）发送了 pidfd SIGSTOP，没有向 idle helper 或外部任务发送信号。旧 W5/I20 master/resumer 保持取消状态。收尾时新计划仍为 rounds=[]、groups=[]、error=null，summary SHA256 `1526d29a6559d771cfcee167fc4c72153b11bcd94aebaa6f85ba0d64340db858` 与暂停前相同，未产生新的有效性能数据。

本节和之前两节分别在两端独立 CAS 追加，保留两端原有文档前缀；所需证据传输和维护 closure 完成后，使用 `TASK/resume_f3_current_W2_I101_after_CPU_maintenance_20261029_v1.py --closure <closure.json>` 恢复同一个 master。该脚本先等待原 idle helper 自然结束，再重新完成连续 11 次、每 6 秒一次、至少 60 秒的全机空闲 gate，最后以 pidfd SIGCONT 恢复原计划，不重启计划、不补跑轮次，恢复后不额外查询 GPU 或传输文件。

下一步继续 v80/v98 的三次中值初筛、同期正向 A/B/A 和反向 B/A/B control、候选三次归档；各字母代表三次测试。性能统一 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin=0、A-preshuffle=1，仅 E64/T1536/topk8/shared0、random non-balanced，禁测全部 E65/topk9。原 caller/timer、gate0.01 和完整 MoE 精度保持。收益验收后立即替换两端正式源码，再从新正式版继续；目前不追加 v108，也不把 v106/v107 作为可测候选。


## 2026-10-29：维护闭合后启动同一 W2/I101 计划的恢复流程，补充 v107 静态 spill 位置

问：汇报下进展，然后继续。

答：v106/v107 的失败证据归档和两端文档同步已完成。远端维护 closure 已生成并下载，本地独立核验 26 个 sealed files 的对应证据；两端文档历史前缀不同，分别按各自 receipt 保留原前缀，最后追加的 Q&A chunk SHA256 相同。两端正式源码仍为 v80/pin=0，没有采用失败候选。

closure 为 `TASK/logs/f3_v106_v107_CPU_maintenance_closed_before_W2_I101_resume_20261029_v1.json`，SHA256 `9a77cfa2cd312e90ea9e3f809c67a7fd71504e0a8ae5ba6af1b34eccfe4818ae`。恢复脚本已在既有 tmux 的新 window `w2-i101-resume`（@96）启动，resumer PID3944356；原 W2/I101 master 仍是 PID3098328，没有重新创建计划。读取的 state 为 `old_idle_helper_finished=false / fresh_idle_gate_completed=false / controller_resumed=false / error=null`，仍在等待原 idle helper 自然结束，然后再执行新的连续11次/至少60秒空闲 gate。

截至已读取的 `2026-10-28T20:14:14+00:00` gate 日志，其他用户 `jimguo12` 的 `bwd_dqdkdv.out` 短任务仍间歇占用 GPU/KFD，空闲窗口反复重置。新性能计划保持 rounds=[]、groups=[]，没有 warmup2/iters101 的有效性能数据。没有向外部进程发送信号，没有计时 child 或新的远端并行编译。恢复流程继续自动每6秒检查；恢复后不额外查询 GPU 或传输文件。

等待期间仅在本地分析已归档的 v107 ELF/ISA，未派生新优化版本。独立核验结果为 28 条 `scratch_store_b64`、27 条 `scratch_load_b64`，PC 范围 33380–84284。7 个包含64条静态 WMMA 的 backward intervals 的并集中，有26个 scratch PCs；相比之下，v106 的对应64-WMMA intervals 中没有 scratch 指令。这里的 interval 是按静态 branch target 划定的代码区间，不是完整动态执行路径，不能从中推断每个 expert 的实际 scratch 次数、具体 spilled live range 或性能成本。两者的零 spill 资源检查失败结论保持不变。

静态分析入口为 `TASK/analyze_f3_v107_spill_intervals_local_20261029_v1.py`，结果为 `TASK/logs/f3_v107_spills_and_static_WMMA_intervals_local_20261029_v1.json`，`GPU_dispatch=false / performance_data=false / candidate_selected=false`。当前先闭合 v80/v98 的三次中值初筛、同期正反向 control 和三次归档，再依据验收结果更新正式源码、推进已准备的 v101/v102/v104；仍只测 E64/topk8，性能 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin=0，原 caller/timer/精度和有效性要求保持。

本节先在本地三份文档独立 CAS 追加。为保持恢复脚本绑定的远端 closure 文档 hash，并避免传输影响后续计时，远端同步和新静态 receipt 上传留到本轮性能计划结束后。


## 2026-10-29：W2/I101 验收后的源码替换步骤已在本地准备，仍等待全机空闲

问：汇报下进展，然后继续。

答：正式版仍为 v80/pin=0。v106/v107 的资源失败已归档并完成两端独立审核/文档同步；两项实验都没有目标 GEMM1 的设备精度或性能结果。v80/v98 的 W2/I101 binding preflight 和完整 MoE 精度通过，当前仍未得到新参数的正常性能数据，不能把旧 W5/I20 的单轮结果并入比较。

截至读取的 `2026-10-28T20:23:24+00:00` 保存日志，外部 `jimguo12` 的 `bwd_dqdkdv.out` 短任务仍在间歇使用 GPU/KFD。新 master PID3098328 保持暂停，resumer PID3944356 仍等待原 idle helper 自然结束；`old_idle_helper_finished=false / fresh_idle_gate_completed=false / controller_resumed=false / error=null`，rounds=[]、groups=[]。恢复流程继续自动每6秒检查，先完成原 gate，再执行新的连续11次/至少60秒空闲 gate，随后恢复同一 master，不创建新的计划或补跑轮次。

等待期间在本地准备了 `TASK/adopt_f3_W2_I101_controlled_candidate_20261029_v1.py`，语法检查通过。该脚本要求最终独立 review 和 archive 都证明21轮完成、三次初筛/同期正反向 control/三次归档通过、原精度和有效性检查通过，且 `(warmup,iters,AITER_LOG_MORE,pin)=(2,101,0,0)`；然后分别在两端以当前正式源码 hash 为前提执行 CAS 替换，备份旧源码并核验暂存区字节保持不变。没有执行或上传 adopter，没有替换正式 kernel，也没有新增优化候选。只有取得实际验收结果后，才使用这个流程。

当前继续等待并完成 v80/v98 的 E64/T1536/topk8/shared0、random non-balanced 性能对照；只以 GEMM1 耗时选版，完整 MoE 原精度仍必须通过。所有性能统一 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin=0、A-preshuffle=1，禁测全部 E65/topk9。收益验收后立即更新两端正式源码，再从新正式版推进后续已准备的候选。最终选版后的英文 commit/push、远端主机 pull 和 pull 后 E64 最终复测仍为待办，未遗忘。

本节先在本地三份文档独立 CAS 追加。远端仍保持维护 closure 绑定的文档 hash；本节、上一节、新静态分析 receipt 和 adopter 的按需传输留到当前计划结束后，避免破坏恢复条件或干扰可能启动的计时。


## 2026-10-29：v107 scratch 分支归属和 scale 寄存器复用的静态核验，进入新的空闲复核

问：汇报下进展，然后继续。

答：两端正式源码仍是 v80/pin=0，W2/I101 计划没有新的有效性能结果。恢复进程和原 idle helper 的身份、状态及日志更新均已检查，没有异常卡住。已保存的100次 gate 观察覆盖 `2026-10-28T20:32:24+00:00` 至 `20:43:23+00:00`，33次观察为空闲，最长连续仅4次、采样跨度19秒，未达到连续11次/至少60秒门槛。这是主机空闲观察，`performance_data=false`，不是 kernel 性能。外部 `jimguo12` 的短 GPU 任务持续打断窗口；保存日志中还出现过其他任务的 E96/T16384/topk6 GEMM 测试。本轮没有启动该规模的测试，也未向这些外部进程发送信号。

随后原 idle helper 已自然结束，恢复流程进入新的连续11次/至少60秒全机空闲复核。读取的 state 为 `old_idle_helper_finished=true / fresh_idle_gate_completed=false / controller_resumed=false / error=null`，当前 plan 仍为 rounds=[]、groups=[]。master 仍是原 PID3098328，resumer 仍是 PID3944356；没有重新创建计划、补跑或导入旧 W5/I20 计时。

等待期间在本地继续检查已归档的 v107 ELF/ISA，没有派生新优化候选或 dispatch GPU。实际 selector 为：

| 条件 | branch / entry PC | 对应 source leaf |
|---|---|---|
| `s46 < 0x81` | 7896 → 29360 | 4-rep |
| 前一条件为假，`s46 > 0xe0` | 7920 → 31576 | 8-rep |
| 两个条件均为假 | 7920 → 7924 | 共用 7-rep |

只固定这两个 selector branch 的选择，保守探索其他全部直接 branch 后，CFG 结果为：

| 固定 selector leaf | 可达 `scratch_store_b64` | 可达 `scratch_load_b64` |
|---|---:|---:|
| 4-rep | 0 | 0 |
| 共用7-rep | 0 | 0 |
| 8-rep | 28 | 27 |

因此，之前位于 PC33380/33932 的两条 scratch stores 虽然在物理布局上早于多数64-WMMA循环，也属于8-rep selector 分支的可达集合，不能仅凭 PC 位置把它们归到共用7-rep 热路径。静态 CFG 没有模拟 SGPR 值传播、EXEC masks 或 loop trip counts；这里的固定边可达性不是动态 trace，也不能替代设备精度、完整 MoE 验证或性能测量。既有零 spill 资源 gate 不变，v107 仍未采用，v106/v107 仍不作为已验收可测候选。

另核验了一个具体的 scale 寄存器复用证据：

1. PC33380：`scratch_store_b64 off, v[0:1], off offset:16 nv`。
2. PC33404/33412：`v0/v1` 被两条 `v_add_nc_u32_e32` 复用于 LDS 地址偏移计算。
3. PC66008：`scratch_load_b64 v[0:1], off, off offset:16 th:TH_LOAD_LU nv`。
4. PC66064/66084：后续 `v_wmma_scale_f32_32x16x128_f4` 的 scale operand 分别使用 `v0/v1`。

CDNA5 ISA §7.12.6/§7.12.6.2（正文pp.103–106）规定 block-scale WMMA 的两个 scale sources，并说明 scale source fields 不使用 VGPR-MSB。以上机器码说明至少一处 scratch 恢复值随后用于低位 scale operand；尚未建立完整 reaching-definition/liveness 证明，不能唯一认定全部 spill 的源变量，更不能把所有 scratch 归为 C accumulator 或 LDS address spill。未来若继续缩小代码并修复资源问题，应重点检查8-rep分支中 scale 值和地址临时量的共存，仍需实际零 spill ELF、native retirement 和设备精度验证。

已保存真实 E64 seed0 routing 的64个 expert count 为168–221，合计12288，全部位于129–224区间；原 body 分布为35个6-rep、29个7-rep。这只限定这份已保存输入。生产 non-balanced routing 可以出现更大 expert count，8-rep 仍是必要支持路径，不能为了当前输入删除它、把它当作不支持范围或忽略其精度。

分析入口为 `TASK/analyze_f3_v107_selector_CFG_scratch_local_20261029_v1.py`，结果为 `TASK/logs/f3_v107_selector_CFG_scratch_and_scale_reuse_local_20261029_v1.json`。分析器首次把PC66080误作WMMA；该位置实际是 `s_wait_dscnt`，检查在写任何结果之前停止。已更正为实际WMMA PC66084并重新核验 ELF/ISA 绑定；未修改生产 kernel、计时参数或资源检查。

下一步仍先完成同一 v80/v98 的 W2/I101 对照：三次初筛、同期正向 A/B/A 和反向 B/A/B control（每个字母三次）、候选三次归档。有收益并通过验收后，立即替换两端正式源码，再从新正式版继续。只测 E64/T1536/topk8/shared0、random non-balanced，`AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin=0、A-preshuffle=1，原 caller/timer/gate0.01 和完整 MoE 精度保持。禁测全部 E65/topk9。

本节先在本地三份文档独立 CAS 追加，远端 chunk 和新静态分析的同步留到当前性能计划结束后；不改写恢复 closure 绑定的远端文档，不传输文件或并行编译干扰可能启动的计时。


## 2026-10-29：v107 的27处 scratch load 后续 scale 用途已形成静态证据，继续当前 W2/I101 对照

问：汇报下进展，然后继续。

答：正式源码仍为 v80/pin=0，SHA256=`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。截至读取的 `2026-10-28T21:39:13+00:00` 保存状态，当前 v80/v98 的 W2/I101 计划仍是 rounds=[]、groups=[]、error=null；恢复流程为 `old_idle_helper_finished=true / fresh_idle_gate_completed=false / controller_resumed=false`。原 master PID3098328 仍在等待恢复，resumer PID3944356 继续按每6秒/连续11次/至少60秒的规则等待全机空闲，没有重启计划、直接 SIGCONT 或补跑轮次。保存日志持续出现外部短 GPU 任务、E96/T16384/topk6 GEMM 测试和间歇非零 busy；这些均不是本轮启动的测试，未向外部进程发送信号。当前仍没有有效的 `--warmup 2 --iters 101` 性能数据，旧 W5/I20 结果不并入当前比较。

等待期间只在本地分析已归档、因资源检查失败而未采用的 v107 ELF/ISA，没有修改生产 kernel、启动新的优化候选、dispatch GPU、并行远端编译或传输文件。

首先，对27处 `scratch_load_b64` 的后续直线代码逐条检查：23处的两个恢复 VGPR 在局部重定义之前都被用作 `v_wmma_scale_f32_32x16x128_f4` 的 scale operand。其中17处用于 WMMA A scale，6处用于 WMMA B scale；另4处遇到 branch，直线分析停止，没有直接归类。独立 reviewer 重新检查了每个报告片段的连续 PC、使用点、目的寄存器覆盖和 ELF/ISA 字节绑定，结果通过。

随后只追踪这4个未归类 load site 的直接 CFG branch，并传播精确 `v_mov_b64_e32` copy，找到了8个恢复值通向后续 B scale operand 的路径 witness：

| scratch load PC | 恢复 VGPR | 后续使用 VGPR / PC | 静态证据 |
|---:|---|---|---|
| 69588 | v[14:15] | v14 / 70252；v15 / 70388 | 跨条件 branch 到 B scale 使用点 |
| 73500 | v[10:11] | v10 / 73884；v11 / 74024 | 跨条件 branch 到 B scale 使用点 |
| 77400 | v[10:11] | v10 / 77784；v11 / 77924 | 跨条件 branch 到 B scale 使用点 |
| 81320 | v[10:11] | v14 / 82740；v15 / 82872 | PC81376 的 `v_mov_b64_e32 v[14:15], v[10:11]` 后跨 `s_branch` 到 B scale 使用点 |

另一个独立 reviewer 沿8条已保存 instruction paths 逐条重建 branch target、fallthrough、vector copy 和寄存器覆盖，检查通过；没有导入或运行分析器。合并两类证据后，27个 load site 都有后续 scale 用途的静态 witness：17个对应 WMMA A scale，10个对应 WMMA B scale。两种证据强度不同：23个是直线片段，4个是带 branch/copy 的 CFG 路径，不能把后者当成已经发生的 device execution。

CDNA5 ISA §3.3.2.3（正文pp.17–18）说明 VGPR-MSB 寻址；§7.12.6.2（正文pp.105–106）说明 WMMA scale sources 忽略 VGPR-MSB，按低256个 VGPR 寻址。上述证据进一步把调查重点落到8-rep分支里 scale 值与其他临时量的共存及寄存器复用。尚未建立完整 store-to-load reaching definitions、源变量映射、SGPR 值传播、MODE dataflow、EXEC masks 或 memory completion 证明，因此不能唯一认定全部 spilled source variables，也不能用静态27处 load site 或路径数推算动态 scratch 次数、cycles 或 kernel μs。高位物理目的寄存器仍使用已保存 disassembler 的 annotations，没有声称新建了完整 MODE 数据流证明。

所有新分析均为 `GPU_dispatch=false / performance_data=false / candidate_selected=false`；原零 spill 资源 gate 不变，v106/v107 仍未采用，没有设备精度、native retirement 或性能结果。kernel 的8-rep支持路径仍须保留，不能因为当前 seed0 的 expert counts 未进入该分支而删除它或忽略其精度。

首次生成分析记录时使用的命令如下，在本地仓库根目录执行：

```powershell
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v107_scratch_load_scale_slices_local_20261029_v1.py
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v107_scale_use_slices_local_20261029_v1.py
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v107_branch_scale_copy_witnesses_local_20261029_v1.py
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v107_branch_scale_copy_witnesses_local_20261029_v1.py
```

四次运行均已通过，生成带 ELF/ISA、分析器和 reviewer SHA256 的 receipts，保存在 `TASK/logs/f3_v107_*scale*20261029_v1.json`。输出使用 exclusive-create；同一路径已有 receipt 时，不重复执行上述生成命令或覆盖记录，复查使用已保存 receipt 和其绑定的 ISA/ELF。

下一步仍先闭合同一 v80/v98 的 E64/T1536/topk8/shared0 对照：三次中值初筛、有收益才完成同期 A/B/A 和 B/A/B control（每个字母三次）、候选三次中值归档；收益验收后立即替换两端正式源码，后续从新的正式版派生。v101/v102/v104 的本地 W2/I101 预备计划仍以 v80 为 reference；若正式版改变，必须重建 reference 和冻结计划，不能直接沿用。

只测 E64/topk8/shared0、random non-balanced，性能固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin=0、A-preshuffle=1。原 caller/timer/gate0.01、完整 MoE 精度及 kernel 支持范围保持；只以 GEMM1 耗时选版。禁测全部 E65/topk9；实际 compiler/test/profiler child 的 TERM118s/KILL2s 上限保持。已授权的英文 commit/push、远端主机 pull 和 pull 后 E64 最终复测仍在最终选版后的待办中。

本节先在本地三份文档独立 CAS 追加。此前三个待同步 chunk、本节及新静态证据的远端传输留到当前性能计划闭合后，继续保持 resumer 所绑定的远端 closure 文档 hashes，避免传输或写入干扰后续计时。

## 2026-10-29：Scale operand 名称澄清、旧 loader 实验回查及 v108 本地准备

问：汇报下进展，然后继续。

答：正式源码仍为 v80/pin=0，没有新的有效性能中值。2026-10-28T22:19:53 UTC 读取既有状态时，W2/I101 的 v80/v98 冻结计划仍为 `rounds=[] / groups=[] / completed=false / candidate_selected=false / error=null`；原 resumer 尚未通过新的连续11次、每6秒一次、跨度至少60秒的全机空闲检查，没有恢复 master。既有日志中的外部任务和非零 busy 继续打断空闲窗口；本轮未启动 GPU 程序、上传文件或并行远端编译，也未杀死外部进程。

本轮在 Windows 本地回查了与 scale 请求顺序有关的旧实验，避免把旧失败方案当成新收益：

| 旧方案 | 已有结果 | 本轮判断 |
|---|---|---|
| v61：两类 scales 在各自 interleaved payload pair 之后读取 | 前侧 v60 / v61 / 后侧 v60 的三轮中值分别81.3355 /81.4873 /81.0423 μs；v61双侧回退0.1866% /0.5491% | 旧 warmup5/iters20 记录，不并入当前 warmup2/iters101；没有重跑 |
| v73：只将 ScaleB reads 从偶数 iteration 移到奇数 | 第一轮82.1219 μs，第二轮 `hipErrorIllegalAddress`，候选组未完成 | 不接受为性能成绩，不据此推断收益或硬件原因 |
| v74：只将 ScaleA reads 从偶数 iteration 移到奇数 | 旧实际对象818 VGPR、private/spills0；没有完整候选性能组 | 编译资源变化不等于性能收益，本轮没有重跑 |

### WMMA 的 A/B 与源码 ScaleA/ScaleB 对应关系

已对 byte-bound v107 源码中的 `mma_rows` 做 AST 核对，并绑定此前两个独立 static witness review。此处的 A/B 必须区分 WMMA matrix operand 与源码 tensor 命名：

| WMMA operand | 本源码传入值 | 对应源码 tensor family | scratch load 后续用途的静态 site 数 |
|---|---|---|---:|
| Matrix A / S3 scale | `wt_value` / `scale_a = sb_k[...]` | 权重 payload / ScaleB (`sb`) | 17 |
| Matrix B / S4 scale | `act_value` / `scale_b = sa_k[wm // 2]` | 输入 activation payload / ScaleA (`sa`) | 10 |

因此，之前的“17处 WMMA A scale、10处 WMMA B scale”对应17处权重 scale family、10处输入 scale family；不能把它们分别写成17处输入 ScaleA和10处权重 ScaleB。

硬件依据是本地 `amd-instinct-cdna5-instruction-set-architecture.txt` 的15.11 VOP3PX2章节、`V_WMMA_SCALE_F32_32X16X128_F4`（页码466–467 of823）：第一、第二 matrix input 分别是A、B，S3、S4分别提供其 scale。本 kernel 将权重作为第一 matrix input、activation作为第二 matrix input。源码 AST 与 ISA 静态用途共同支持的是 operand family 对应关系，仍不是精确 spilled source variable 或 store-to-load reaching definition 证明；没有新增动态 EXEC/MODE、memory completion、动态次数、cycles 或 μs 结论。

本地结果：`TASK/logs/f3_v107_source_scale_operand_roles_local_20261029_v1.json`。v107仍因资源 gate失败而拒绝，原零 spill要求保持。

### v108：只准备8-rep scales后移实验，尚未编译或测试

v108在v107实验分支上，只将8-rep、pin0、interleaved LDS loader中的所有 ScaleA/ScaleB definitions移到全部 A/B payload reads之后。共用7-rep body及selector保持；4-rep、pin1、非interleaved loader和其他rep的请求顺序保持。所有有效load地址、请求数量、values、entry/exit retirement fences、K数学、input TDM和epilogue与v107保持，未新增expert/token/data specialization。

这与v61在每个payload pair之后发起scales不同：本次只针对已定位有scratch的8-rep叶子，把全部scale definitions后移。它只是一项缩短scale live range的假设；LLVM可能重新调度，实际是否消除scratch必须看ELF，不能从源码顺序直接断言。

源码生成器完成1024组实际loader AST exercise（interleaved rows、pin、interleaved loads、active rep1–8、N rep1/2/4/8、ksl0/1、kt0/7）。16组目标请求顺序变化，其余组合顺序一致；所有组合的operand结果、有效请求 multiset和同步一致。这些是源码行为检查，不替代native retirement、真实设备精度或性能验证。

- v108 source SHA256：`4b1158a104a9d1668544a578116d60208ecde21da90b1b4ccfc15296f728a4f4`。
- plan：`TASK/logs/f3_v108_eight_rep_scales_after_payload_source_20261029_v1.plan.json`。
- snapshot：`TASK/snapshots/f3_v108_eight_rep_scales_after_payload_20261029_v1.py`。
- 尚未上传、生成实际ELF、执行native/device检查或测性能；未生成新的性能计划、未替换正式v80。

继续优先闭合已有v80/v98计划；性能统一 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin0、A-preshuffle1，只测E64/T1536/topk8/shared0、random non-balanced，全部E65/topk9禁测。按三次初筛、同期A/B/A和B/A/B control、三次归档验收，收益确认后立即采用，再从新正式版推进。最终英文commit/push、远端主机pull及pull后E64最终复测仍为待办。此节先独立CAS追加本地三份文档，远端保留当前性能计划绑定的文档hash，待计划闭合后追加。

## 2026-10-29：v108 实际编译仍有 spill，维护收尾完成并继续原 W2/I101 对照

问：汇报下进展，然后继续。

答：正式源码仍为 v80/pin=0，SHA256=`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。本轮完成了 v108 的实际编译、ELF/ISA 资源检查、证据包逐项核验，以及进程内导入身份和原 caller 签名的补充核验。v108 减少了 spill，但仍未通过原零 spill gate，因此未执行 native retirement、设备精度、性能或 ATT，未采用。当前没有新的有效 `--warmup 2 --iters 101` GEMM1 性能中值。

### 实际编译结果

v108 在 v107 的8-rep、pin0、interleaved LDS loader中，将全部 ScaleA/ScaleB definitions 移到全部 A/B payload reads 之后；其他 rep、地址、请求数量、operand结果和同步保持。此前1024组 loader AST exercise 通过；实际资源如下：

| 候选 | `.text` B | VGPR | SGPR | LDS B | private segment B | VGPR spill count |
|---|---:|---:|---:|---:|---:|---:|
| v107 | 79232 | 797 | 60 | 327680 | 36 | 56 |
| v108 | 78848 | 800 | 60 | 327680 | 28 | 36 |

v108 的 `.text` 减少384 B，private segment减少8 B，VGPR spill count减少20；VGPR总数增加3。ISA中仍有18处 `scratch_store_b64` 和17处 `scratch_load_b64`。这些是静态site数量，不是动态次数、cycles或性能成本。不能从spill减少推断GEMM1有性能收益，也不能据此放宽原资源检查或忽略8-rep支持路径。

硬件资料 `MI400_Shader_Programming#65.txt` §3.3.5 “Scratch (Private) Memory” 说明scratch是wave-launch时可分配给thread-private data的global memory。这里拒绝v108的直接依据是实际ELF的非零private/spill资源和ISA中的scratch指令；资料和静态指令数量不能给出本候选的运行耗时。

编译child用时18.14161171100568秒，inspect child用时0.16380720099550672秒，两者returncode=0；controller总计21.71603102599329秒，以returncode=1保留资源失败：`AssertionError('Keep resource failure; do not dispatch target GEMM1')`。全部实际child均低于120秒上限，没有失败重跑或替代轮次。

### 编译方式和身份核验

本轮使用 `MetaPathFinder` 在编译进程内将原module name `aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm` 导入到v108 snapshot。正式kernel文件没有临时换版。输入准备可能使用GPU；compile-only截获下目标GEMM1没有dispatch，相关耗时不作为性能数据。

补充review确认：effective module file和launcher source均指向同一v108 snapshot；source hash、stage artifact和ELF/ISA一致；正式v80文件before/after hash相同；`cache_argument_signature`、`cache_globals_signature` 与原v80 stage preparation逐字相等；原wrapper、compiler options、caller和timer保持。新的source产生自己的原生cache namespace，未伪造cache key。

- source SHA256：`4b1158a104a9d1668544a578116d60208ecde21da90b1b4ccfc15296f728a4f4`。
- ELF SHA256：`e6add9ebaf6b3111776431352dbe56e9191d24d8d8fd9913dd3460eae0b9e80a`。
- failed-resource archive：785530 bytes、27 members，SHA256=`667b2a1cbf9348dfea1a45e79e9a51b854c4b8c0ff9b9392fb0bfbd2906137a5`。
- combined import/resource review SHA256：`b39beeaf37c45a3b47b007fc602e2ce2314bd39135f95875a1bab42bd565687a`。

首次运行入口如下；证据保存在同stem下，生成工具使用exclusive-create，已有输出不覆盖：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v108_process_import_compile_20261029_v1.py \
  --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_v108_process_import_compile_source_20261029_v1.plan.json \
  --stem f3_target_v108_eight_rep_scales_after_payload_pin0_process_import_cpu_20261029_v1
```

本地补充核验入口为 `python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v108_process_import_binding_local_20261029_v1.py`；实际结果通过，没有再次编译或dispatch目标kernel。

### 原性能计划恢复状态

维护closure已在a07-3主机生成并通过检查，绑定41个文件，SHA256=`cfad7cc830e65b58e25c24e6865b71bab09ea7384f0e39b2ea41b19eb2556223e`。原master PID3098328和原resumer PID3944356身份、boot、HEAD、冻结plan、空summary和正式v80源码保持；没有重启、添加或替代性能轮次。旧idle helper PID333996已结束。

新的guardian PID2060930/start_ticks115755828/UID12144已启动，负责维护后的全新连续11次、每6秒一次、跨度至少60秒空闲检查。通过后仅恢复同一个原resumer，由原resumer恢复同一个master。本轮未直接SIGCONT master，也未向外部进程发送信号。读取的guardian状态为 `existing_idle_helper_finished=true / fresh_idle_gate_completed=false / original_resumer_resumed=false`。已保存idle日志出现外部短GPU任务，因此暂未获得新的计时样本；持续自动等待。

继续同一v80/v98冻结计划：只测E64/T1536/topk8/shared0、random non-balanced，`AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin0、A-preshuffle1。原caller/timer/gate0.01、完整MoE精度及kernel支持范围保持；全部E65/topk9禁测。按三次初筛、同期A/B/A和B/A/B control（每个字母三次）、三次归档验收，有收益立即替换两端正式源码，再以新正式版继续。MoE e2e耗时不作为采用条件。

本节先独立CAS追加本地三份文档。远端三份文档保持closure绑定的hash，待当前性能计划闭合后独立追加历史chunk和本节；不以本地整份文件覆盖远端。已授权的英文commit/push、远端主机pull和pull后E64最终复测仍为最终选版后的待办，当前冻结计划闭合前不改变HEAD。

## 2026-10-29：v108 残余 scale-use 静态检查及 v109 单 input slot 本地准备

问：汇报下进展，然后继续。

答：正式源码仍为 v80/pin=0，未产生新的有效 W2/I101 性能中值。本轮继续在本地检查已拒绝的v108 ELF/ISA，并准备v109源码；没有启动新的远端compiler、设备精度、性能或profiler child，没有修改正式源码或上传文件。

先更正上一节一个进程身份数字：guardian PID2060930的 `start_ticks` 按远端 `f3_W2_I101_after_v108_maintenance_guardian_launch_20261029_v1.json` 应为 **11575828**；上一节写作115755828，多写了一个5。PID2060930、UID12144、原master/resumer及closure身份均不受此文档录入错误影响，没有因此执行任何进程信号操作。

### v108 剩余 scratch 的静态用途

重新用原 `load_bound_isa` 将v108 ISA逐指令绑定到ELF，并复用已有v107分析中的physical register parsing helpers（只提取函数AST，没有运行旧分析器或改写旧输出）。17处 `scratch_load_b64` 中，11处的两个恢复VGPR在直线代码中、局部重定义之前都被用作WMMA scale。共22个恢复值的用途是8个WMMA A scale、14个WMMA B scale；按本源码 `mma_rows` 的传参，这分别对应权重ScaleB (`sb`) 和输入ScaleA (`sa`)。

其余6处load均在直线追踪时遇到branch，没有在本轮继续跨branch/copy追踪：

| load PC | 恢复VGPR | 停止PC / branch |
|---:|---|---|
| 69508 | v[4:5] | 69540 / `s_cbranch_scc1 957` |
| 73356 | v[4:5] | 73388 / `s_cbranch_scc1 953` |
| 77188 | v[4:5] | 77220 / `s_cbranch_scc1 958` |
| 81040 | v[4:5] | 81104 / `s_branch 213` |
| 81064 | v[14:15] | 81104 / `s_branch 213` |
| 81076 | v[8:9] | 81104 / `s_branch 213` |

这是后续scale-use的静态诊断，未证明精确源码变量、store-to-load reaching definitions、动态EXEC/MODE、memory completion或实际执行次数，不能估算cycles或μs。本地CDNA5 ISA §7.12.6.2 “Matrix Scale Values” 说明scale ISA fields不使用VGPR-MSB；该规则用于将scale source按低256个VGPR解释。高位目的寄存器仍依赖已保存disassembler annotations，没有新建完整MODE dataflow证明。

输出：`TASK/logs/f3_v108_remaining_scratch_scale_slices_local_20261029_v1.json`；生成入口：`python -X utf8 .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v108_remaining_scratch_local_20261029_v1.py`。v108的原零spill gate失败状态保持，未采用。

### v109：仅对既有八行支持分支尝试单 input register slot

v109从v108 snapshot派生，只在 `pin0 / interleaved_m_rows / active_m_rep==8` 的既有runtime coverage分支中，将两个rmem input slots设为同一个对象。原 `k_step` 已有 `reuse_cur_rmem` 路径：同一个slot时，先完成当前全部WMMA，再调用下一k128的loader。v109没有改变 `k_step`、`mma_rows` 或 `compute_ktile` 的AST；在八行loader入口的sched barrier后显式加入 `s_wait_alu depctr_va_vdst(0)` 和后续sched barrier，要求旧WMMA使用先retire，再替换同一slot的payload和scale。

目的只是减少已出现spill的八行叶子中current/next payload与scale同时存活的数量。四行、七行、pin1以及非interleaved M路径保持两个slot；八行支持路径没有删除，也没有新增expert、token或data specialization。输入TDM、WG barrier、有效WMMA数学、activation/quant epilogue、原caller/timer/gate0.01和正式准入保持。

这一取舍会让八行叶子的下一k128 input slot在当前WMMA之后加载，失去这部分current/next slot overlap。八行耗时可能变化，尚未测量；不能从源码的slot数量断言实际VGPR减少、零spill或性能提高，也不能因为当前seed0不走八行而免除该分支的native retirement和设备精度验证。

本地检查结果：

- 1024组实际loader AST exercise通过：有效load地址、数量、operand值和请求顺序保持；32组目标八行组合仅新增完整VA retirement fence及其sched barrier，其余组合fences也保持。
- 192组actual slot constructor与 `k_step` 输入更新模型通过：覆盖interleaved M开关、pin0/1、active rep4/6/7/8、KWS1/2/4、tile数1/2/3/28。每个k128输入按顺序加载、所有模型coverage行读取正确输入；目标单slot分支在当前全部WMMA调用之后才覆盖slot。
- 以上是源码请求和更新顺序模型，替换了WMMA实际算子，不是设备数值检查，也不证明LLVM实际调度、硬件memory completion或完整MoE精度。

source：`TASK/snapshots/f3_v109_eight_rep_single_slot_20261029_v1.py`，SHA256=`f6d03c9b8e91df4c2f9152e97d54aa273225ae5053a6f7bac3aa20baa516e006`。

source plan：`TASK/logs/f3_v109_eight_rep_single_slot_source_20261029_v1.plan.json`；首次生成入口：`python -X utf8 .codex_tmp/hyg_f3_followup_20261024/derive_f3_v109_eight_rep_single_slot_local_20261029_v1.py`。尚未上传、实际编译、native/device检查或测性能，没有生成新的性能计划或替换正式v80。若当前选版后reference改变，需重新核对派生关系和准入，不能直接沿用以v80为reference的冻结计划。

读取的2026-10-28T23:17:45 UTC既有idle日志仍出现外部短GPU任务、grouped GEMM测试和非零busy；guardian的 `fresh_idle_gate_completed=false / original_resumer_resumed=false`，原v80/v98 summary仍为rounds=[]、groups=[]、selection_status=running、candidate_selected=false。本轮未向外部进程发信号，继续每6秒自动检查全机空闲；没有让传输或远端并行编译与可能启动的计时重叠。

下一步继续优先闭合原v80/v98的三次初筛、同期A/B/A和B/A/B control、三次归档；仅记录 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin0、A-preshuffle1下的E64/T1536/topk8/shared0、random non-balanced GEMM1结果。收益验收后立即采用，再以新正式版继续优化；完整MoE精度和支持范围保持，MoE e2e耗时不作为采用条件，全部E65/topk9继续禁测。

本节独立CAS追加本地三份文档；远端chunk、静态证据和v109源码留待当前性能计划闭合后按文件检查同步，不覆盖远端文档，也不改变当前HEAD。已授权的英文commit/push、远端主机pull和pull后E64最终复测仍在最终选版后的待办中。

## 2026-10-29：v109 实际零 spill 编译、独立导入核验与 native gate 拒绝

问：汇报下进展，然后继续。

答：正式源码仍是 v80/pin=0，尚无新的有效 `--warmup 2 --iters 101` 性能中值。v109 已完成实际 compile-only 和独立 ELF/ISA/import 核验，但随后两个既有 native retirement 模型都发现相同的两处风险，因而未采用，也没有进入设备精度、性能或 ATT 测试。

### 实际编译资源

| 版本 | `.text` B | VGPR | SGPR | LDS B | private B | VGPR spills |
|---|---:|---:|---:|---:|---:|---:|
| v108 | 78848 | 800 | 60 | 327680 | 28 | 36 |
| v109 | 78336 | 747 | 60 | 327680 | 0 | 0 |

v109 的实际 compile child 为 18.1413220030081 秒，inspect child 为 0.11381430400069803 秒，两者 returncode 均为 0。与当前 v80 的 `.text=97408 B` 相比，v109 缩小 19072 B；这些是编译资源和 CPU child wall time，不是 kernel 性能数据。

本次编译仍使用原模块名和原 caller 的真实参数，通过 process-local source import 绑定 snapshot。控制器没有替换正式文件，没有 dispatch 目标 GEMM1；输入准备可能使用 GPU。独立核验逐项检查 archive 的全部 32 个文件、实际 ELF/ISA 机器字节、原 ABI、实际 metadata 资源，以及与 v80 相同的 argument/global signatures。零 spill gate 通过，ISA 没有 `scratch_*` 指令。

- source SHA256：`f6d03c9b8e91df4c2f9152e97d54aa273225ae5053a6f7bac3aa20baa516e006`
- ELF SHA256：`3d6ed3778a04593b1734c23dcec5348dd1494f5f3835895f2fdc20434bd58e5d`
- archive SHA256：`2b897fb803114b5f885b61298b44fd0b002d5eaf619ac34f9b681a550f1b37b2`
- compile 独立 review SHA256：`5bc96edecdcf25f77a17b751ebffd2e1aaafa77070f99753be48b2612475dd91`

复现入口（已有输出使用 exclusive-create，不要覆盖已保存的记录）：

```text
# 远端主机归档已有编译结果，不重跑 compiler
python3 -u .codex_tmp/hyg_f3_followup_20261024/pack_f3_process_import_compile_20261029_v3.py --stem f3_target_v109_eight_rep_single_slot_pin0_process_import_cpu_20261029_v1

# 本地独立核验
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_process_import_compile_local_20261029_v2.py --stem f3_target_v109_eight_rep_single_slot_pin0_process_import_cpu_20261029_v1
```

第一次打包使用的 v2 工具在白名单检查处失败：它收集了控制器已 hash 绑定的 `.sh` idle helper，却只允许 `.py/.json/.log/.elf/.mlir/.exit/.txt`。v3 工具先完整检查 inventory，只额外允许已绑定且 SHA256 匹配的 `.sh`；还逐 member 核验并改名保留了 v2 的部分 archive，没有删除或覆盖它，也没有重新编译。旧 v2 文件、旧记录和冻结的性能计划均保留原样。

### 两种 native retirement 检查均拒绝 v109

本地 native reviewer 从既有 common-hot-body reviewer 派生，调整范围仅为 process-local compile manifest 与 v107→v108→v109 source proof chain 的核验。worker 的完整 else 分支 AST 保持，原 DS/WMMA `transfer` 逻辑及另一个 `WMMA_only` ordering stress 规则保持；没有删除、重分类或豁免 hazard。

| ordering 模型 | CFG transfer iterations | issue 数 | gate |
|---|---:|---:|---|
| original | 22799 | 2 | failed |
| WMMA_only | 23938 | 2 | failed |

两者都报告 `DS overwrite before WMMA retirement`：

| PC（十进制） | DS load | 涉及 VGPR |
|---:|---|---|
| 23032 | `ds_load_b128 v[168:171] /*v[680:683]*/, v14 offset:40960` | `v[680:683]` |
| 23040 | `ds_load_b128 v[172:175] /*v[684:687]*/, v14 offset:41472` | `v[684:687]` |

实际 ISA 在 PC 22896–22944 的四条 WMMA 读取 `v[680:687]`，随后仅四条其他 WMMA，PC 23024 执行 `s_wait_alu depctr_va_vdst(8)` 后便出现上述 DS overwrite。该直线片段及跳入其上下文的 branch edges 已逐指令绑定到 ELF 并保存；目前只是实际机器码和保守 retirement 模型的诊断，没有证明完整动态路径，也不是设备数值错误的复现。

本地《MI400 Shader Programming Guide》§4.3.7.4.1、正文页 92 说明 `SCHED_MODE 2` 下软件需保证相关 dependency hazard 的等待；同章说明 WMMA 与其他 VALU 的完成顺序需要区别考虑。因此零 spill 不足以证明 input register 重用安全。后续先定位覆盖路径和 fence 调度，再检查修订候选；不会通过放宽模型、精度 gate 或支持范围来接受 v109。

```text
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v109_process_import_native_local_20261029_v1.py --stem f3_target_v109_eight_rep_single_slot_pin0_process_import_cpu_20261029_v1
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v109_process_import_native_local_20261029_v1.py --stem f3_target_v109_eight_rep_single_slot_pin0_process_import_cpu_20261029_v1 --worker --ordering WMMA_only
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v109_retirement_issue_context_local_20261029_v1.py
```

失败结果分别保存在 `<stem>_native_original.json`、`<stem>_native_WMMA_only.json`；机器码上下文为 `TASK/logs/f3_v109_retirement_issue_context_local_20261029_v1.json`。没有生成通过的 parent native review，没有 dispatch 目标设备 kernel。

### 继续原性能计划

已把通过的 compile 独立 review 上传远端并核对 hash，再启动既有 `close_and_resume_f3_guardian_after_v109_compile_20261029_v1.py`。维护 closure 已完成，确认本轮 compiler/inspect child 均结束，原 v80、HEAD、冻结 plan、master/resumer 身份及远端三份文档保持。新恢复进程 PID2786569、start_ticks11888878、UID12144 先等旧 idle helper 自然完成，再开始全新的 11 次／至少 60 秒空闲检查；它只会恢复原 guardian PID2060930，由原 guardian 恢复原 resumer/master，不重新启动或补充性能轮次。

读取到的原 `f3_E64_LOG0_W2_I101_v80_v98_screen_ABA_BAB_archive3_20261029_v1` summary 仍为 `rounds=[] / groups=[] / selection_status=running / candidate_selected=false / error=null`。a07-3 的外部短 GPU 任务持续打断空闲窗口，自动每 6 秒检查继续运行；没有向外部进程发送信号，没有以占用期间的耗时归档性能。

后续只测 E64/T1536/topk8/shared0、A4W4/SiLU、random non-balanced，固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin0、A-preshuffle1；原 caller/timer/gate0.01 与完整 MoE 精度要求保持，全部 E65/topk9 禁测。采用仅由 GEMM1 的三次初筛、同期 A/B/A 与 B/A/B control、三次归档决定；收益验收后立即替换正式源码，MoE e2e 耗时不是采用条件。

本节 CAS 追加本地三份文档；远端文档、native 新证据与该问答 chunk 等原性能计划闭合后逐文件同步，避免覆盖既有远端历史。已授权的英文 commit/push、远端主机 pull 与 pull 后 E64 最终复测仍保留在最终选版后的待办中，`.codex_tmp` 不暂存。

## 2026-10-29：v110 编译与 native retirement 检查通过，编译维护已收尾

问：汇报下进展，然后继续。

答：正式源码仍是 v80/pin=0。v110 已通过实际编译、独立 ELF/ISA/import 核验以及两种既有 native retirement 模型；设备精度、性能与 ATT 尚未验证。原 v80/v98 的 `--warmup 2 --iters 101` 性能计划还没有有效样本，不能把旧 W5/I20 数字或本轮 CPU 检查用时作为新的 kernel 性能。

### v110 的修改与检查结果

v110 由 v109 派生，只在既有 `pin0 / interleaved_m_rows / active_m_rep==7` loader 中加入三个 `sched_barrier`：原完整 VA wait 前后各一个，loader 尾部一个。原 wait 值保持。1024 组 loader AST exercise 验证了 load 地址、数量、值和请求顺序保持；slot 构造、数学、TDM、WG barrier 与 epilogue 没有修改，八行 single-slot 支持分支保持。

v109 两种 native 模型报告的 `DS overwrite before WMMA retirement` 在 v110 的实际机器码中不再被报告。仍使用原 transfer 逻辑、完整 worker else 分支和 `WMMA_only` ordering stress，没有放宽或豁免 hazard。

| 版本 | `.text` B | VGPR | SGPR | LDS B | private B | VGPR spills |
|---|---:|---:|---:|---:|---:|---:|
| v109 | 78336 | 747 | 60 | 327680 | 0 | 0 |
| v110 | 77568 | 783 | 60 | 327680 | 0 | 0 |

v110 的 `.text` 比 v80 的 97408 B 减少 19840 B（20.37%），比 v109 减少 768 B；VGPR 比 v109 增加 36，实际零 spill gate 仍通过。compile child 为 18.140803781003342 秒，inspect child 为 0.11382597500050906 秒，两者 returncode=0。这些是编译资源及 CPU wall time，不是 GEMM1 耗时，也不能据此认定加载地址敏感性已经消除。

| native ordering 模型 | CFG transfer iterations | issues | CPU wall seconds |
|---|---:|---:|---:|
| original | 20091 | 0 | 3.299141600029543 |
| WMMA_only | 21230 | 0 | 2.922054599970579 |

本地 `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt` 的 §4.3.7.4.1（正文页 92）要求 `SCHED_MODE 2` 下软件保证相关 dependency hazard 的等待；同章说明 WMMA 与其他 VALU 完成顺序需要区别考虑。native 通过只是实际 ISA 与既有保守模型的检查结果，不能代替设备完整 MoE 精度验证或证明全部硬件 hazard。

证据位于 `.codex_tmp/hyg_f3_followup_20261024/`：

- snapshot：`snapshots/f3_v110_seven_loader_boundaries_20261029_v1.py`，source SHA256=`cf4245e6504b96b7c3c31bad84200e797672f1e48a5d2ee8bdd130f8d2bee73c`。
- stem：`f3_target_v110_seven_loader_boundaries_pin0_process_import_cpu_20261029_v1`；ELF SHA256=`aef3ad2df72ec5be99938d78bf569f528c225207ad705387d40318640bd565a9`，archive SHA256=`906555c66b9736eb65f851701dd970e4276f063a28ef8f30e4bd8f7c56e7b7f8`。
- compile review：`logs/<stem>_process_import_compile_independent_local_review.json`，SHA256=`ab44764e775130bbbe584b3dd36322fdfb94eb76c424b759e60dd0e274ba34fd`；全部 34 个 archive members 已独立核验。
- native review 与两个 worker 记录：`logs/<stem>_native.review.json`、`logs/<stem>_native_original.json`、`logs/<stem>_native_WMMA_only.json`。

复核入口（已有输出使用 exclusive-create，保留旧记录，复测需使用新 label）：

```text
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_process_import_compile_local_20261029_v2.py --stem f3_target_v110_seven_loader_boundaries_pin0_process_import_cpu_20261029_v1
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v110_process_import_native_local_20261029_v1.py --stem f3_target_v110_seven_loader_boundaries_pin0_process_import_cpu_20261029_v1
```

### 维护恢复与后续顺序

已上传 compile review 并核对两端 SHA256，执行原 `close_and_resume_f3_guardian_after_v110_compile_20261029_v1.py`。维护 closure 已完成，仅恢复同一个 guardian PID2786569/start_ticks11888878；没有直接恢复旧 master/resumer/guardian，也没有重新启动性能计划。guardian 先等待既有 idle helper 自然结束，再执行编译后的全新 11 次／至少 60 秒空闲检查，随后逐级恢复原计划。

closure SHA256=`55f691e7120f5804a392ea23cda3fef1e0d92bf8e91fa1977d0136b47cb002c78`。恢复后读取到的原 summary 仍为 `completed=false / rounds=[] / groups=[] / selection_status=running / candidate_selected=false / error=null`。自动每 6 秒检查全机空闲的流程继续运行；未向外部进程发送信号。

继续优先闭合原 v80/v98 计划，然后核对当前 reference，推进 v110 的 E64 完整 MoE 设备精度与三阶段性能验收。只测 E64/T1536/topk8/shared0、7168/2048、A4W4/SiLU、random non-balanced；性能固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin0、A-preshuffle1。全部 E65/topk9 禁测；原 caller、timer、gate0.01、完整 MoE 精度和 kernel 支持范围保持。采用仅按 GEMM1 的三次初筛、同期 A/B/A 与 B/A/B control、三次归档决定，收益验收后立即更新正式源码。

本节仅独立 CAS 追加本地三份文档。远端文档与新 native 证据在原性能计划闭合后逐文件同步，保留两端历史。已授权的英文 commit/push、远端主机 pull 与 pull 后 E64 最终复测仍在最终选版后的待办中，`.codex_tmp` 不暂存。

## 2026-10-29：v110 的 E64 完整 MoE 设备精度已通过

问：汇报下进展，然后继续。

答：正式版本仍为 v80/pin=0。本轮 v110 已完成 E64/T1536/topk8/shared0、7168/2048、A4W4/SiLU、random non-balanced 的完整 MoE 设备验证；两组输出均通过原精度 gate，并与既有基线逐字节一致。没有新的有效 `--warmup 2 --iters 101` GEMM1 性能中值，不能据此宣布采用 v110 或认定性能已提升。

| seed | 完整 MoE logits_diff | rel_l2 | output bytes | 与基线 differing bytes |
|---:|---:|---:|---:|---:|
| 0 | 3.6105513278025183e-06 | 0.0026872099842876196 | 22020096 | 0 |
| 17 | 3.6202693316254594e-06 | 0.002690816530957818 | 22020096 | 0 |

输出 shape=`[1536, 7168]`、dtype=`torch.bfloat16`。seed0 输出 SHA256=`443e8c5959dc9513a655d5c9b59b21c93c738df5807cda3123560fbc8e47b935`；seed17 为 `04b4ad0f8e6f51991674e95ffcf432d4a3873c20c5064fbc9212bf52cc0e294a`。两者分别与已验证的 v60 seed0/17 基线完全一致。

实际运行保留了原 `run_f3_v80_followup_device_E64_only_20261028_v1.py`、原 caller 和原完整输出 verifier，没有修改数学或精度标准。最长 child 是 GEMM2 stage preparation，23.900766637001652 秒；两组完整 MoE child 分别为 8.526236091012834、8.176206546006142 秒，returncode 均为 0，每个实际 child 保留 TERM118s/KILL2s 限时。这里的 CPU wall seconds 不是 kernel 性能数据；正确性运行的全部计时均排除出性能归档。

实际 GEMM1 ELF 与已通过 native 检查的 v110 ELF 相同，SHA256=`aef3ad2df72ec5be99938d78bf569f528c225207ad705387d40318640bd565a9`。GEMM2 完整 ELF 保持 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。已核验的 GEMM1 cache 内容 hash 与 mtime 均未变化。

执行期间仅暂停原等待 guardian PID2786569，原 master/resumer/older guardian 保持暂停；没有与本轮性能计时并发。正确性结束后 v80 正式源码恢复，重新检查 GPU/KFD 时没有本轮测试进程残留，随后恢复同一个 guardian，由其执行全新空闲检查后继续原 v80/v98 性能计划。未向外部任务或 idle helper 发信号，也没有新建或补充性能轮次。全部 E65/topk9 均未测试。

证据 stem=`f3_v110_seven_loader_boundaries_complete_MoE_pin0_20261029_v1`，位于 `.codex_tmp/hyg_f3_followup_20261024/logs/`。archive SHA256=`1b918701569ce29e5601b394ba6d1526d2a497fa8fe9145e577ff5ff4653a6ef`，本地独立核验通过全部 45 个文件，包括原精度与逐字节对比 receipts、实际 ELF/IR/native 绑定、每个 child 的 log/exit 和维护恢复 receipt。archive 不包含 MoE 输出 payload 或 cache；本地核验验证已保存的 child 比对记录，没有声称重新比对未下载的输出 payload。

复现入口（原始记录使用 exclusive-create，复跑需新 stem）：

```text
# 远端主机，测试在 dsr1_ep4 中执行；协调原等待任务的维护控制器
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_v110_E64_device_maintenance_20261029_v1.py

# 远端主机，归档已完成的精度证据
python3 -u .codex_tmp/hyg_f3_followup_20261024/pack_f3_E64_device_receipts_20261029_v1.py --stem f3_v110_seven_loader_boundaries_complete_MoE_pin0_20261029_v1 --cpu-stem f3_target_v110_seven_loader_boundaries_pin0_process_import_cpu_20261029_v1

# 本地，独立核验
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v110_E64_device_local_20261029_v1.py
```

最新读取的原 v80/v98 summary 仍为 `rounds=[] / groups=[] / selection_status=running / error=null`；外部 GPU 短任务继续打断空闲窗口。后续性能仍固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin0、A-preshuffle1，按三次初筛、同期 A/B/A 与 B/A/B control、三次归档决定，收益验收后立即采用。当前计划闭合后核对 reference，再开始 v110 性能对照；不混入其他参数的旧数字。完整 MoE 精度与 kernel 支持范围保持，MoE e2e 耗时不作为采用条件。

本节独立 CAS 追加本地三份文档。原计划闭合前保留远端三份文档、HEAD 和正式 kernel；远端问答及新证据稍后逐文件同步。已授权的英文 commit/push、远端主机 pull 和 pull 后 E64 最终复测继续保留，`.codex_tmp` 不暂存。

## 2026-10-29：v110 的 eight-row 精度覆盖已核实，继续原性能对照

问：汇报下进展，然后继续。

答：正式源码仍为 v80/pin=0。v110 已通过编译、原 native retirement 检查及 E64 seed0/17 完整 MoE 原精度验证。本轮进一步核实 seed17 的 routing：expert 10、54 都有 225 行，覆盖既有 `mn_oob > 224` 的 eight-row 选择条件；对应 seed17 的完整 MoE 输出此前已与基线逐字节一致。当前仍无有效的新 `--warmup 2 --iters 101` 性能中值，不能据此采用 v110 或宣布加载地址敏感性已经消除。

### 路由覆盖与检查脚本修正

采用原 `_run_grouped_via_fused_moe` 的完整输入初始化，只临时包装原 `_make_topk`；取得真实 routing 后在 GEMM 前结束，finally 恢复 hook。只检查 E64/T1536/topk8/shared0、7168/2048、A4W4/SiLU、random non-balanced。没有改动正式 kernel、原 caller、性能 seed0 或计时函数。

| seed | 最大 expert 行数 | 超过 224 行的 expert | 已有完整 MoE logits_diff | 已有 rel_l2 | 与基线 differing bytes |
|---:|---:|---|---:|---:|---:|
| 0 | 221 | 无 | 3.6105513278025183e-06 | 0.0026872099842876196 | 0 |
| 17 | 225 | 10、54，各 225 行 | 3.6202693316254594e-06 | 0.002690816530957818 | 0 |

v2 probe 返回失败，原因是检查脚本把 `_make_topk` 返回的原始 fp32 权重 fingerprint，与 caller 转换后的 bf16 权重直接比较。这是辅助检查脚本的问题，不是正式 kernel 的数值错误。v3 保留两套 fingerprint gate，分别核对转换前后的历史记录，并把 seed0 的 64 个 route counts 与原性能 caller 的记录逐项比较；全部一致。v1/v2 失败记录保留，未替换成成功样本。

```text
seed0 ids int32 SHA256:
a543dfbcfe5976bc42809fbe751880b996a3ca88d2e95175a204e74a35fc6a6c
seed0 _make_topk weights fp32 SHA256:
99940dfb31ef88a29570641fa6138adf0dfb2f0fdef2abcba67a5cd4d040a117
seed0 caller weights bf16 SHA256:
4c098198961b67c7411c768b7c01e4b5064e258a4379dc8308d20f5c69c5330e
```

route probe 本身只证明输入覆盖；完整 MoE 精度及逐字节一致性来自已有 device receipts，未把路由检查当作新增的 kernel 精度测试或动态 ISA trace，也没有声称本地重比未下载的输出 payload。

本地独立核验通过五份新下载日志/receipt 的 SHA256，并绑定已有 v110 device summary、review 与 snapshot。路由记录为 `TASK/logs/f3_v110_E64_original_route_coverage_20261029_v3.json`，SHA256=`24eeb1c9e2c738560a96281b8c928d394ba44267d26e33d13fa516e1eeca4166`；核验结果为 `TASK/logs/f3_v110_E64_route_coverage_independent_local_review_20261029_v3.json`。

复现入口（记录使用 exclusive-create，重跑需新 label）：

```text
# a07-3 主机；只暂停自己的等待 guardian，probe 在 dsr1_ep4 中执行
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_E64_route_probe_guarded_20261029_v3.py

# 本地独立核验
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_v110_route_coverage_local_20261029_v3.py
```

probe child 采用 TERM28s/KILL2s 限时，正常结束；其 wall time 不计入 kernel 性能。运行时只暂停已核对 PID/start_ticks/UID/argv 的原 guardian，结束后恢复同一个 guardian；master/resumer/older guardian 保持原状态，旧 idle helper 与外部进程均未收到信号。未重启性能计划、增加轮次或修改远端文档。

### v110 的 code footprint 与下一步性能验收

已有独立 ELF 分析核实 `.text`：v80 为 97408 B，v110 为 77568 B，减少 19840 B（20.37%）。最外层 backward-edge interval 从 95740 B 缩到 75828 B，仍跨 64 KiB；区间包含互斥 body，不能当作动态热点 working set，也不能据此保证不同加载地址的性能一致。分析记录为 `TASK/logs/f3_v80_v110_loop_intervals_local_20261029_v1.json`。

v110 的三阶段性能计划已在本地冻结并完成 sources-only review，尚未上传或运行：

```text
label = f3_E64_LOG0_W2_I101_v80_v110_screen_ABA_BAB_archive3_20261029_v1
plan SHA256 = bf788349a8ac5981ee364f83429355b320a4fbea230025b4705009fcdf8137a8
rounds run = 0
```

先完成原 v80/v98 W2/I101 计划；若它选中 v98，立即采用并重新核对后续 reference 绑定，不能直接启动仍以 v80 为 reference 的旧计划。继续只按孤立 GEMM1 的三次初筛、同期 A/B/A 与 B/A/B control、三次归档决定采用，MoE e2e 耗时不作为采用条件，完整 MoE 原精度必须保持。

截至本轮读取，原 v80/v98 summary 仍为 `completed=false / rounds=[] / groups=[] / selection_status=running / candidate_selected=false / error=null`。a07-3 的外部 FMHA、其他 GEMM 短任务与间歇非零 busy 持续打断空闲窗口；原 helper 每 6 秒自动重查全机，原 guardian 尚未通过 fresh gate。确认连续 11 次、至少 60 秒全机空闲后继续原对照，计时期间不另做 GPU 查询、SFTP 或远端编译。

性能参数固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin0、A-preshuffle1；性能输入保持 seed0。全部 E65/topk9 禁测。正式源码 SHA256 仍为 `93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。

本节先独立 CAS 追加本地三份文档，保留全部原历史；原性能计划闭合前保留远端已绑定文档，闭合后逐文件同步。已授权的最终英文 commit/push、远端主机 pull 与 pull 后 E64/W2/I101 最终复测仍在待办中，`.codex_tmp` 不暂存。

## 2026-10-29：v110 公共 output TDM store 的静态 code placement 核对

问：汇报下进展，然后继续。

答：v110 的完整 E64 精度与 eight-row 输入覆盖已经核实，正式源码仍为 v80/pin0，新的 W2/I101 性能对照继续等待 a07-3 的全机空闲窗口。进一步机器码核对发现，v110 的公共 `tensor_store_from_lds` 从 ELF PC `0x1086c` 移到了 `0xfd9c`；它与所选 seven-row 循环在 64 KiB 对齐假设下的相对 set 关系不再随 group0–3 改变。这是静态映射结果，不能据此宣布性能收益或完成跨地址验收。

### 实际 ELF 中的位置

解析原完整 ELF 和 ISA，并逐条核对 ISA 编码与 ELF bytes；两版均只有一条公共 output TDM store 和一条 `s_endpgm`。所选内层 backward-edge 区间含 52/56 条静态 WMMA，区间不足 4 KiB，合并时去重；这些区间没有被当作已测动态热点。

| 版本 | 公共 `tensor_store_from_lds` ELF PC | `s_endpgm` ELF PC | 所选循环指令 PC 范围 |
|---|---|---|---|
| v80 | `0x1086c` | `0x10880` | `0x2f08–0x6844` |
| v110 | `0xfd9c` | `0xfdb0` | `0x2ef8–0x6ed4` |

硬件依据为本地 `mi400_hw_wiki/raw/papers/mi400_hd_txt/block/sq/MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1：

```text
Set[4:0] = address[13:9] ^ address[18:14]
```

按照该公式，计算所选循环的 512 B instruction lines 与公共 store instruction line 的相对 set 等同性；对 set 编号做规范化后比较 partition signature，不假定 bank hash、live residency、miss latency 或运行时加载地址。

| 假设 `load_base` 低 16 位 | v80 group0–3 的不同相对 set partition 数 | v110 group0–3 的不同相对 set partition 数 |
|---|---:|---:|
| `0x0000` | 3 | 1 |
| `0x8000` | 3 | 3 |
| `0xf000` | 1 | 1 |

扫描保留全部 16 个 4 KiB offset 与四个 group 的结果，未挑选最有利地址。`0x0000` 下，v110 的所选循环与公共 store instruction 均在同一个 64 KiB 地址段；v80 的 store instruction 位于后一个地址段。`0x8000` 下两者仍各有三种关系，说明本次静态结果具有对齐条件，不能推广为任意 loader placement 都已稳定。

本模型只覆盖指定循环 instruction lines 与一个公共 store instruction line，未涵盖所有执行 block；same set 也没有被等同为 same bank、实际冲突或 cache miss。`.text` 总长仍为 77568 B，eight-row 等其他路径可能跨段。完整修复仍需要同一 ELF、原 timer、原输入在实际 load_base group0–3 下的耗时分布与独立 counters 证据。当前 `cross_address_performance_repair_accepted=false`。

复现入口与记录：

```text
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/analyze_f3_v80_v110_store_loop_set_relation_local_20261029_v1.py

TASK/logs/f3_v80_v110_store_loop_set_relation_local_20261029_v1.json
```

本分析未 dispatch GPU、重编译或改写 ELF、修改 loader/alignment、改变正式源码，也未生成性能数据。原 v80/v98 W2/I101 计划继续运行，每 6 秒自动检查全机空闲；尚无候选性能样本，不补跑或替换慢/失败轮次。继续只测 E64/T1536/topk8/shared0、random non-balanced，固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、pin0、A-preshuffle1。

本节独立 CAS 追加本地三份文档，保留原历史。远端文档待原性能计划闭合后同步；已授权的最终英文 commit/push、远端主机 pull 与 pull 后 E64 最终复测仍保留在待办中。

## 2026-10-29：W2/I101 baseline 与 v98 三次初筛完成，继续 v110

问：汇报下进展，然后继续。

答：正式源码仍为 v80/pin=0。已闭合原 v80/v98 性能计划；六次样本全部有效，v98 三次中值比同期 v80 慢约 6.29%，按既定规则在初筛后停止，未进入 control 或采用。原 master 已正常退出，远端正式源码恢复 v80，结束时 GPU/KFD 检查没有本批测试进程残留。

性能仅来自 a07-3 的 `dsr1_ep4`，workload 为 E64/T1536/topk8/shared0、model_dim7168/inter_dim2048、A4W4/SiLU、random non-balanced。全部样本固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`、A-preshuffle1；保留原 caller、原 timer、原 gate0.01 和完整 MoE 精度验证。每轮性能运行前使用既有全机连续 11 次／至少 60 秒空闲检查，每 6 秒轮询；计时期间没有额外 GPU 查询或文件传输。没有使用旧 W5/I20 数字替代新参数样本，也没有选择加载地址或替换慢样本。

| 版本 | 第一次 GEMM1 μs | 第二次 GEMM1 μs | 第三次 GEMM1 μs | 三次中值 GEMM1 μs | 结论 |
|---|---:|---:|---:|---:|---|
| v80 | 81.90383838383835 | 81.4941134020618 | 81.8573535353534 | 81.8573535353534 | 保持正式版本 |
| v98 | 87.07807999999989 | 87.00608999999993 | 86.77350526315786 | 87.00608999999993 | 初筛无收益，不采用 |

这里的 baseline 是当前正式 v80 在最新计时参数下的同期 reference，不是初始未优化 kernel 的历史 baseline。只按孤立 GEMM1 耗时选择版本；未测量或记录其他 kernel 的性能，MoE e2e 耗时不是采用条件。全部 E65/topk9 均未运行。

原计划 label=`f3_E64_LOG0_W2_I101_v80_v98_screen_ABA_BAB_archive3_20261029_v1`，plan SHA256=`82476e3375d97711230eb9b7eec7e73e4e9fbc362ec0ff38ffb3dd051d83d70e`。summary 为 `completed=true / selection_status=rejected_screen_no_gain / candidate_selected=false / all_performance_valid=true / original_source_restored=true`。archive SHA256=`297170f13d43c7219672c9640516e94fe72236a483213dc1898b7306bc83f9b5`，本地已核验全部 211 个 archive members，并从日志、receipt 和实际 ELF/IR/loaded text 重新核对六次计时及中值；review SHA256=`df731e7ba883bbb3db2b7c46c7bd08b93f3c2e4be43268d9a544fb304d1ccf68`。

### v110 后续测试准备与 LOG0 脚本问题

v110 已通过 E64 seed0/17 完整 MoE 原精度验证，与保存的基线逐字节一致；seed17 的 expert10、54 各有225行，已覆盖 eight-row 输入支持分支。不重复把正确性 child 的 wall time 当作 kernel 性能，也不重复跑已经覆盖的精度场景。

旧性能 driver 与 binding preflight 强制要求 CPU preparation prelude 中恰有一个 `AITER_LOG_MORE=1`，再替换为0。v110 的实际 preparation 记录已是 LOG0，旧脚本将因此失败。这是测试准备脚本的兼容问题，不是 GEMM1 数学或精度 bug。

新 v2 脚本允许恰好一个 LOG0 或 LOG1 记录；仅对 LOG1 执行原替换，继续强制最终 prelude 为 LOG0。对实际 v98(LOG1) 与 v110(LOG0) 两份 CPU 记录独立执行处理后，最终 shell bytes 的 SHA256 一致。保留原 caller、timer、child checker、PID 检查、精度 gate 与性能有效性规则；所有实际测试/编译 child 继续保留 TERM118s/KILL2s 限时。旧冻结计划、脚本与失败/慢样本记录均未修改。

新计划 label=`f3_E64_LOG0_W2_I101_v80_v110_screen_ABA_BAB_archive3_20261029_v2`，plan SHA256=`538c1ba8c4d73ec85e6c5b8002b93fa5182bdd1cbef423b7bd302b6170d54242`。本地源码复核已通过，远端 126 个绑定文件 SHA256 全部一致，branch/HEAD/boot 与 v80 正式 source hash 已重检。当前准备运行 binding preflight，之后按三次初筛 → 同期 A/B/A、B/A/B（每个字母三次）→ 三次归档推进；尚无 v110 性能样本或采用结论。

复现入口位于 `.codex_tmp/hyg_f3_followup_20261024/`，已有 label 使用 exclusive-create，复跑需新冻结 label：

```text
# 本地核验 v98 六次初筛证据
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_three_stage_CPU_window_kernel_timer_W2_I101_local_20261029_v1.py --label f3_E64_LOG0_W2_I101_v80_v98_screen_ABA_BAB_archive3_20261029_v1

# a07-3 主机；child 在 dsr1_ep4 中运行，不把 preflight 计时作为性能
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_three_stage_LOG0_binding_preflight_E64_W2_I101_20261029_v2.py --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_W2_I101_v80_v110_screen_ABA_BAB_archive3_20261029_v2.frozen_plan.json --warmup 2 --iters 101
```

后续性能及跨加载地址诊断仍须实测；静态 code footprint/set 分析不能替代性能验收。已授权的英文 commit/push、远端主机 pull 和 pull 后 E64 最新参数最终复测继续保留，`.codex_tmp` 不暂存，`op_tests` 不静默加入用户指定的暂存范围。

## 2026-10-29：v110 W2/I101 初筛闭合，正式版保持 v80，继续 v101

问：汇报下进展，然后继续。

答：v110 三次初筛中值为 82.872132653 μs，同期 v80 为 81.849724490 μs，回退 1.249128%。按既定规则初筛无收益即停止，未进入正反向 control 或归档采用；正式源码仍为 v80/pin0。

| 版本 | 第一次 GEMM1 μs | 第二次 GEMM1 μs | 第三次 GEMM1 μs | 三次中值 GEMM1 μs | 结论 |
|---|---:|---:|---:|---:|---|
| v80 | 81.93786868686861 | 81.80350505050492 | 81.8497244897959 | 81.8497244897959 | 保持正式版本 |
| v110 | 82.87213265306127 | 82.54951063829779 | 82.8727099999999 | 82.87213265306127 | 初筛无收益，不采用 |

这里的 reference 是当前正式 v80 的同期测量，不是初始未优化 kernel 的历史 baseline。六条样本全部有效，每条实际 child 约8秒，结束时本批 child 均已退出；controller 正常结束，源码恢复 v80。

全部运行只在 a07-3/dsr1_ep4，E64/T1536/topk8/shared0、7168/2048、A4W4/SiLU、random non-balanced，固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`、A-preshuffle1。原 caller、timer、gate0.01 和完整 MoE 精度验证保持；每轮性能前执行既有全机空闲门禁，外部任务出现时每6秒等待，不杀外部进程。性能计划期间未执行 SFTP、额外 GPU 查询或远端编译。全部 E65/topk9 未运行。只按 GEMM1 耗时选版，没有记录其他 kernel 耗时或 MoE e2e 耗时。

### v110 的 canonical import 绑定与归档问题

旧 v2 preflight 因 `cache_namespace` 不一致失败，发生在性能测试前，失败记录已保留。临时 snapshot import 与正式文件 import 的 source IR location 不同；只替换386处源文件路径，两份 IR 字节完全一致，完整 ELF 及实际 loaded `.text` 均与已验收版本相同。v3 计划绑定正式 caller 的实际 namespace/IR hash，没有放宽 ELF、精度或 timer 检查，也没有复用旧性能样本。

v110 正式 namespace 为 `launch_gemm_a8w4_tdm_f3b75b7217c77c297c46e6641a93d520`，与 v80 的 namespace 不同。正式 GEMM1 IR SHA256 为 `04dab816e76d8cd9a4bf417c03adbaf71559d023e5ee60ca26a68e219b3a32cb`；完整 GEMM1 ELF SHA256 为 `aef3ad2df72ec5be99938d78bf569f528c225207ad705387d40318640bd565a9`。v3 binding preflight 两条 child 均通过后才启动上述性能计划。

旧 packer 的日志 glob 顺带收集已存在的正确性 archive，文件类型门禁因此拒绝归档。新版本仅排除这些没有被 plan/device 绑定的冗余 archive 及其 manifest，继续收集全部底层绑定成员，保留原始输出排除和 SHA256 检查；没有修改 kernel 或性能有效性。最终归档含258个成员，全部通过本地独立核验。

性能 label=`f3_E64_LOG0_W2_I101_v80_v110_screen_ABA_BAB_archive3_20261029_v3`；plan SHA256=`fe5af138ac3ec9582761f882915509932508a0233da0dc7d2fdb96539f9fb2a1`；archive SHA256=`c02e21bf0cd40bac8e2b280c561c002777dbd2c8b41a7fbce74aa5ea747ddc4e`；review SHA256=`4715757055c07b6cc50dd02b6add8b4b4ee24598f5977ad93ceb8a615f86972f`。

### 下一组 v101

v101 是已有的 row-tail VA4 等待候选。此前实际 ISA 对照证明相对 v98 只有12处 `depctr_va_vdst(0)` → `depctr_va_vdst(4)`，实际 `.text` 只有12个 byte 改变；两种原 native retirement 模型和 E64 seed0/17 完整 MoE 原精度已通过。非零 VA wait 的安全性继续依赖实际 ISA/native 和设备验证，不能仅据等待 cycles 推断。

新 W2/I101 v2 plan=`f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v2`，SHA256=`b9b90daff7c10dc7cc53e38eddb014f70e637765cb27bef833a1536ffaf7e503`，119个文件绑定及本地 source review 已核对。旧 W5/I20 与 W2/I101 v1 计划保持，全部21个预定轮次使用新 label 和新输入调用，未引入旧计时；当前尚无该新批次性能结果。

复现入口（a07-3 主机启动；测试 child 在 dsr1_ep4 执行，已有 label 不覆盖，复跑需冻结新 label）：

```bash
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_three_stage_LOG0_binding_preflight_E64_W2_I101_20261029_v2.py --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v2.frozen_plan.json --warmup 2 --iters 101
python3 -u .codex_tmp/hyg_f3_followup_20261024/run_f3_three_stage_original_kernel_timer_CPU_WINDOW_E64_LOG0_PID_W2_I101_20261029_v2.py --plan .codex_tmp/hyg_f3_followup_20261024/logs/f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v2.frozen_plan.json --warmup 2 --iters 101
```

三次初筛有收益才进入同期 A/B/A、B/A/B（每个字母三次），通过后再做三次归档，确认收益立即替换正式源码。v110 缩小 code footprint 未带来本批速度收益，不能据静态分析宣称跨加载地址问题已修复。本轮未修改正式 kernel 或暂存区；已授权的英文 commit/push、远端主机 pull 与 pull 后 E64 最终复测继续保留。

## 2026-10-29：v101 中断证据闭合，重启后恢复限时查询与新批次

问：汇报下进展，然后继续。

答：正式源码仍是 v80/pin0，最近已完成的三次 E64 GEMM1 中值为 81.409175258 μs。v110 初筛没有收益，不采用；v101 只完成一条候选样本，随后第五轮之前的 GPU 状态读取阻塞。该批未完成，不给 v101 排名，不把其单次结果当作三次中值，也不跨批补两条。

### 中断批次与进程处理

| ordinal | 版本 | GEMM1 μs | 原记录 valid |
|---:|---|---:|---|
| 1 | v80 | 81.15439795918378 | true |
| 2 | v80 | 81.5291818181817 | true |
| 3 | v80 | 81.40917525773193 | true |
| 4 | v101 | 86.26179381443299 | true |

上述四个实际 child 均约8秒完成，均保留原 caller、timer、gate0.01 和完整 MoE 精度验证。第五次 GEMM 测试没有启动；阻塞的是空闲检查中的 `cat /sys/class/drm/card1/device/gpu_busy_percent`，PID414767/start_ticks12789850，状态为 `D`，`wchan=amdgpu_dpm_read_sensor`。这次阻塞不能归因于第五次 GEMM 测试的 kernel，也不能仅凭 D 状态断定整机一定需要重启。

`2026-10-29T02:38:52 UTC` 核对自己的 master314175、waiter401135、sensor414767 的 PID/start_ticks、进程树与 PGID 成员后，只对这三个自己的进程发出 SIGKILL。master 和 waiter 已停止；sensor 当时仍为 D，SIGKILL 挂起，因此当时不能声称没有残留进程。没有向外部 PID 发信号，没有 GPU reset 或自行重启。生产源码已恢复为 v80，SHA256=`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`，暂存区原字节保留。

中断 label=`f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v2`，plan SHA256=`b9b90daff7c10dc7cc53e38eddb014f70e637765cb27bef833a1536ffaf7e503`。summary 明确为 `completed=false / selection_status=interrupted_GPU_sensor_read / candidate_selected=false / all_performance_valid=false / original_source_restored=true`；原四条样本的 valid 和数值没有修改。

本地独立复核已核验全部221个归档成员，archive SHA256=`e7a9fc8f397b3a7f51e34019023313768d8c8f301e66072cf29231f1b81c8c7e`。从原日志、receipt、实际 ELF/IR/loaded text 重新核对四次计时、原精度门禁和源码绑定；另核验取消前后的 summary hash，四条 rounds 和已有 groups 完全相同，取消前生产源码为 v101、取消后归档生产源码为 v80，且没有第五轮 child identity 或结果文件。该归档保留部分观测数据，`performance_data=false`，不用于采用候选。

复核入口：

```text
python -X utf8 .codex_tmp/hyg_f3_followup_20261024/review_f3_three_stage_CPU_window_kernel_timer_W2_I101_local_20261029_v2.py --label f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v2

TASK/logs/f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v2.cancellation_preservation_independent_local_review.json
```

### 查询限时保护与重启后的状态

本轮只通过 CPU metadata 先确认：boot_id 已由 `ff39c3dc-ffb5-4510-a458-0110ee739267` 变为 `47fbe50f-2c15-4168-9cb8-10b9453bb82e`，上述三个旧 PID 均不存在，branch/HEAD 和正式 v80 hash 保持。`dsr1_ep4` 正在运行，但没有把“容器相同”当作 GPU 空闲或时钟域相同的证明。

新 host 查询模块将每次 GPU 状态查询隔离到自己的 session/process group，默认10秒后 TERM，最多再等待2秒后 KILL。只使用非阻塞 `poll()`，不对 D 状态 reader 执行阻塞 `wait()` 或 `communicate()`。在实际查询开始前，launcher 必须先核对已登记的 PID/start_ticks/token/boot guard；查询父进程退出后仍会检查其子进程组。若有同一 boot/start_ticks 的 reader 仍存活，后续轮询只读 `/proc`，不再创建 GPU 查询。查询失败、输出不完整或状态未知都不能被解释为 idle。

9项 CPU 验证已通过：12次快速 JSON 退出，非零退出、截断 JSON、非 object JSON、超时且忽略 TERM、父进程先退出而子进程仍活、已有 reader 阻止重复查询、PID/start_ticks 与 boot 变化、并发查询 lock。超时及父进程先退出的 CPU fixture 分别约0.796/0.370秒处理完成；这些是 CPU 退出路径验证，未读取 GPU、未 dispatch kernel，不是新的 kernel 性能数据，也没有人为复现真实 D 状态。

host/container/privileged CPU reader 的 `CLOCK_BOOTTIME`、USER_HZ、time namespace 和 offsets 已重新检查一致。新模块首次实际 GPU 查询约0.287秒正常返回；当时其他 PID127084 正在使用 GPU，VRAM约60.5 GB，未启动自己的 kernel 测试，也未触碰该任务。随后每6秒自动等待，已有初步日志完成连续11次、至少60秒全机空闲验证。每个正式性能 child 仍须重新执行自己的空闲门禁；该初步空闲记录不能替代后续各轮门禁。

复现入口与证据：

```text
# a07-3 主机执行；纯 CPU 测试，不使用 GPU；同名 QA 输出已存在，复跑须使用新输出版本
timeout --signal=TERM --kill-after=2s 118s python3 -u .codex_tmp/hyg_f3_followup_20261024/check_f3_bounded_GPU_state_CPU_only_20261029_v2.py

TASK/f3_bounded_GPU_state_read_20261029_v2.py
TASK/wait_f3_all_GPUs_and_external_tests_idle_60s_bounded_20261029_v1.py
TASK/logs/f3_bounded_GPU_state_CPU_only_20261029_v2.json
TASK/logs/f3_CPU_clock_domain_after_boot47fbe50f_20261029_v1.summary.json
TASK/logs/f3_boot47fbe50f_environment_and_bounded_GPU_query_20261029_v1.json
```

新 controller 与本地 reviewer 由旧 v2 做可逆文本推导，只替换 host 查询与对应 waiter，并在查询未知时明确中断批次；原 child checker、caller、timer、精度 gate 与 `valid=(post['ownership_idle'] or known_dead) and settled['idle'] and own_process_absent` 方程保持。原冻结脚本、计划和样本不改。

新 v101 plan label=`f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v3_boot47fbe50f_BOUNDED`，SHA256=`79cffc0ea63915c2cbce83f89cedcae57a93cea5aa5333e15fe06d3f09405a15`，133个文件及8个源文件绑定已在本地核对，新 source review 通过。下一步为同步必需的新工具、重新运行 binding preflight，再按三次初筛 → 同期 A/B/A、B/A/B（每个字母三次）→ 三次归档继续。当前没有新 boot 下的 GEMM1 性能样本或采用结论。

后续只测试 E64/T1536/topk8/shared0、7168/2048、A4W4/SiLU、random non-balanced，固定 `AITER_LOG_MORE=0 --warmup 2 --iters 101`、`AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`、A-preshuffle1。只以 GEMM1 耗时选版，保留 MoE e2e 正确性，全部 E65/topk9 禁测。已授权的英文 commit/push、远端主机 pull 与 pull 后 E64 最终复测继续保留；本轮没有 kernel 优化改动或 Git mutation。

## 2026-10-29：优化收尾、baseline 问答与 v80 详细说明

问：先不测性能了，这次优化任务，baseline 版本的性能是多少？

答：本轮从 `hyg_dsr1_conc6144 / 6f656cf5b02b2f1acf723949652f91b48e2933d7 / F3` 接手。原始 F3 的历史 GEMM1 三次中值为 **81.7383 μs（pin=0）**，三次为82.3413 /81.7383 /81.4917 μs；pin=1 对照中值为80.7452 μs。以上使用 `--warmup 5 --iters 20`。后续切换 pin0 后阶段基准 **82.0483 μs** 已经是行尾 v2，不是起点 HEAD 的原始 F3。更早 E64 baseline99.0062 μs属于旧 codex 分支任务。当前 `AITER_LOG_MORE=0 --warmup 2 --iters 101` 没有对原始 F3 和 v80 做完同期配对，不能由跨参数、跨批次数字推导严格总降幅。本轮按用户要求没有启动性能复测。

问：v80 相对于 baseline 做了哪些优化？写个详细介绍放在一个 markdown 文件中。

答：已写入 [v80 相对 F3 baseline 的详细说明](moe_gemm1_v80_vs_f3_baseline_20261029.md)。文档逐项说明 M16 行块交错分配、WG 在 K loop 外选择4/6/7/8档覆盖、跳过 padding WMMA 与A/ScaleA LDS读取、输入及量化输出行地址重映射、较短body的retirement waits/compiler boundaries，以及六/七档likely selector与一次公共epilogue的code layout。文档包含原始baseline身份、源码hash、正常计时与独立counters、精度和支持范围、复现命令及证据索引。

源码核对确认：当前 `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` 与v80 snapshot逐字节相同，SHA256=`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`；F3与v80的文件顶层函数参数签名相同。v80的公共epilogue AST与v76原epilogue一致，恢复selector/epilogue region后整个源码逐字节恢复成v76，region之外的计算没有变化。

v45的dense LDS地址简化和v60的scale/payload LDS读取穿插是另一条实验路线，没有叠加到最终v80。v80相对v3的`.text`总长仍为97408 B；相对未采用v79，代码量115712→97408 B，静态FP4量化sites61→32。这是减少机器码复制，不是少算有效输出。

v80已经完成有限跨加载地址修复验收：固定ELF、E64目标配置、seed0/17、实际64 KiB对齐load base group0–3、原pin0 timer。group3的ICACHE miss中值在seed0/17分别下降89.42% /61.99%；seed17正常计时仍约89–90 μs。不能声称任意routing或任意loader alignment下所有波动都已消失。

### 最后一组已有实验的结果与停止状态

label=`f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v3_boot47fbe50f_BOUNDED`，plan SHA256=`79cffc0ea63915c2cbce83f89cedcae57a93cea5aa5333e15fe06d3f09405a15`。以下保留原ordinal和valid，不补选样本：

| ordinal | 版本 | GEMM1 μs | 原valid |
|---:|---|---:|---|
| 1 | v80 | 80.52848958333333 | true |
| 2 | v80 | 81.22617346938779 | true |
| 3 | v80 | 81.25989898989893 | true |
| 4 | v101 | 85.82230208333321 | true |
| 5 | v101 | 86.28818367346948 | false |

三条v80 reference观测中值为81.22617346938779 μs，固定pin0、A-preshuffle1、LOG0/W2/I101。第五条实际child正常返回，完整MoE原精度通过、自有child已退出，但测后出现外部KFD owner775124，未通过既有ownership门禁，随后settled恢复idle。该外部PID的CPU metadata查询超时，创建时序关系为unknown；现有证据没有证明其GPU工作与原timer重叠。原valid=false不据此改成true，也没有杀死该外部进程。

第六条没有运行，v101没有三次有效中值，不排名、不采用。原summary保留`completed=false / selection_status=running / candidate_selected=false`及原AssertionError，不手改成完成状态。master已经退出，finally恢复正式v80；旧boot下的测后记录确认自有child已退出。后续机器重启仅通过CPU metadata重新核对，不把旧boot的idle记录用于新测试。

本地已独立核验全部246个归档成员、五条原表和valid、原timer/precision/ELF绑定。archive SHA256=`8052861f5fb366dc3d83c08b26b88462899ba3413d0238ea0a2cfffd76e1ccc9a`，review=`TASK/logs/f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v3_boot47fbe50f_BOUNDED.CPU_window_independent_local_review.json`。整批`performance_data=false / candidate_selected=false`；三条reference不是完整候选采用验收或pull后最终成绩。

遵守用户“结束优化”和“先不测性能”的更新：没有新候选、没有补跑v101，没有新性能或正确性dispatch；完整MoE精度以已经验收的记录为准。正式源码保留v80。此前授权的英文commit/push和远端主机pull继续做收尾，pull后的最终性能复测暂停。

### 两端历史核对与 v65 ATT archive SHA256 更正

下载并只读比较远端7个现存任务文件。三份主文档的旧OPTX问答块仅位置不同，远端119884字符的块已逐字节存在于本地2026-10-27“补回a07-3保存的旧OPTX问答”记录中。远端原件已保存在`TASK/publication_20261029_v1/remote_originals/`，本地原件在CAS追加前另存备份；两端独有历史均保留。

对实际文件`TASK/logs/f3_v65_GEMM1_ATT_direct_identity_20261026_v2.ATT_evidence.tar.gz`重新计算的64位SHA256为：

```text
41029c182d41c10743364e45a213b1deb2a5469688feb478174d4de211767e85
```

远端旧表使用上述正确hash。本地三份文档的一处旧表误多写一个`7`，成为65位字符串`41029c182d41c10743364e45a213b1deb2a5469688feb478174d4de2117767e85`。本地此前一段中文被写成问号的更正记录已经保存正确hash，但说明不可读；本节重新用中文明确更正，原表及问号记录保留为历史，不改归档bytes、107个members、2503608 B、ELF/ISA绑定或性能数据，也没有重新采trace。

本次新文档与以上问答以UTF-8/LF写入。所有工作均为CPU读取、归档核验、文档和Git收尾；全部E65/topk9仍禁测。

## 2026-10-29：重启后完成 baseline F3 / v80 最终复测与独立核验

问：baseline 和 v80 的性能再测一下做个对比；a07-3 hang 后应等重启、确认 rocm-smi 正常才使用 GPU，等待期间每隔两分钟尝试 SSH；继续。

答：已完成重启后、授权源码 commit/push/远端 pull 后的六轮同期复测，全部有效样本已经在本地独立核验。仅复测已有 F3/v80，优化工作结束，正式源码保留 v80。

| 版本 | 三次 GEMM1 μs，按各版执行顺序 | 三次中值 μs |
|---|---|---:|
| F3 baseline | 84.008918 / 83.387657 / 83.123265 | **83.387657** |
| v80 | 80.904896 / 81.120394 / 81.942660 | **81.120394** |

v80 比 F3 减少 **2.267263 μs**，耗时降低 **2.718943%**。表格显示六位小数，原始数值、中值和公式完整保存在原 summary 与独立 review。历史 F3 的 81.7383 μs 属于旧 LOG1/W5/I20；这次对比只使用本批次六条新样本。

固定 workload：E64/T1536/topk8/shared0、model_dim=7168/inter_dim=2048、A4W4/SiLU、无 bias、random non-balanced，`--seed 0 --data-init uniform --scale-init auto`。固定 `pin=0`、A-preshuffle=1、`AITER_LOG_MORE=0 --warmup 2 --iters 101`。原 caller、kernel timer、有效计算及原 `logits_diff < 0.01` 门限保持。执行顺序在运行前冻结为 F3/v80/F3/v80/F3/v80，每版三次取中值；没有加载地址选择、替换慢样本、换 seed 或补挑快实例，全部 E65/topk9 仍未测。

六轮 kernel 与完整 MoE eager 精度检查均通过，`logits_diff=3.6105513278025183e-06 < 0.01`，route ids/weights fingerprint 全部一致。本次只记录 GEMM1 耗时；完整 MoE 使用 `bench=False, kernel_bench=False` 验证精度，没有测 MoE e2e 时间。证据中的 `graph_metrics` 是该完整 MoE eager 检查，完整 MoE CUDA graph 验证不在本次范围内。

每轮计时前全机空闲检查连续通过 11 次、间隔 6 秒、跨度至少 60 秒；计时期间没有并发 GPU 查询、传输或编译。六个 child 约 7.98–8.34 秒内正常结束，均有 TERM118s/KILL2s 的 120 秒硬上限。测后自有测试 child 均不存在，最终 `settled.idle=true`，生产源码已恢复为正式 v80。

源码发布与性能身份：`hyg_dsr1_conc6144 / cecc19e93bf112cb1a1670431c80a7dec5a2dc46`，英文 commit 为 `Optimize gfx1250 GEMM1 row coverage and restore fused A-preshuffle path`，已 push origin 并在 a07-3 主机 pull。测试进入 `dsr1_ep4`，共享工作区为 `/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`。F3 使用 commit `6f656cf5b02b2f1acf723949652f91b48e2933d7` 的原始 Git blob，source SHA256=`f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec`；v80 source SHA256=`93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312`。后续文档收尾 commit 只更新记录，性能身份仍是上述源码 HEAD 和两版 hash。

GEMM1 ELF SHA256：F3=`62037bcfa621ff1c2d378e0e5face1479e6af47fbcb5beb8bacf8ee39d003897`，v80=`49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8`。两版 GEMM2 ELF 相同，SHA256=`4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`。原 caller/timer 与实际 ELF、loaded text 的绑定均通过复核。

重启等待按用户要求每隔两分钟尝试 SSH；重启后 boot_id=`7e035a22-f1ba-411b-be47-b404092abe68`。主机旧 `/usr/bin/rocm-smi` 在 `GetGPUMetricsFormat1` assertion，未修复。恢复 GPU 工作前，已验证 `dsr1_ep4` SDK 的 `/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin/rocm-smi --showuse --showmeminfo vram --json` 正常返回：一张 GPU，利用率 0%，VRAM used176582656 B。验收 receipt 同属本次 boot。两版参考 matmul 均采用 `TORCH_BLAS_PREFER_HIPBLASLT=0`，原参考数学和精度门限保持；性能测试未设置 `AMD_SERIALIZE_KERNEL`。旧 boot 的等待批次保留为中断记录，没有产生性能样本。

证据批次为 `f3_E64_F3_vs_v80_LOG0_W2_I101_after_pull_20261029_v4_rocblas_boot7e035a22`，以下记作 `LABEL`，`TASK=.codex_tmp/hyg_f3_followup_20261024`：

- frozen plan：`TASK/logs/LABEL.frozen_plan.json`，SHA256=`01df1baef6f0cb79feabcdf87d0c1bcbbf4bbbcc0173341bd5546f488245fe48`。
- 原 controller 和最终 summary：`TASK/logs/LABEL.controller.log`、`TASK/logs/LABEL.summary.json`，`completed=true / all_performance_valid=true / original_source_restored=true`。
- 证据归档：`TASK/logs/LABEL_evidence.tar.gz`，266 个成员、8784857 B，SHA256=`ffc10fc776399dd95722a77dcecf6d290ebb858d13f3c3c988e4a173452d5aec`。
- 本地独立核验：`TASK/logs/LABEL.independent_local_review.json`，`passed=true`；重新核验全部归档 hash、原 valid 方程、精度、相同 routing、原 caller/timer/AST、实际 ELF/loaded text、rocm-smi receipt，并独立重算六条读数、中值和降幅。
- 重启后 readiness：`TASK/logs/f3_ROCM_SMI_ready_after_reboot_20261029_v3_sdk_boot7e035a22.json`，`passed=true / returncode=0`。

完整优化介绍和此次复测结果见 [v80 相对 F3 baseline 的详细说明](moe_gemm1_v80_vs_f3_baseline_20261029.md) 第 9.5 节。正式 v80、旧失败样本和既有问答历史均保留。
