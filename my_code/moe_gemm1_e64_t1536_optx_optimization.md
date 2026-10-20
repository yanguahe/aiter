# gfx1250 MoE GEMM1 optX 优化记录

## 任务与状态

按用户“异常实验重测，手头实验做完后暂停”的最新要求，2026-10-20 已完成 v265 的完整数值验收和性能筛选，并完成此前异常 E65/v190 的三次原完整 benchmark 重测。GEMM1/GEMM2/MoE e2e 重测中值为 **136.931 /89.441 /281.17 μs**，旧 161.590 /315.866 /982.26 μs 异常未复现；保留旧记录，不将其作为稳定对照。**本轮优化已暂停**，v266 只完成 CPU，未启动其 GPU 验证、性能或新 counters。正式最优仍是基于 OPTX 方案的 v190，E64/E65 原正式三轮中值 **91.688 /137.596 μs**，E64 距80 μs差 **11.688 μs**。

目标是 E64/T1536/topk8、random non-balanced routing 下 GEMM1 稳定低于 80 μs。shared expert 单独计算；有效行 canonical FP4 payload、ScaleA、GEMM2 和 MoE 输出保持 bitwise 一致。允许跳过无效 padding 计算，禁止更改计时、使用 const0 特例、反转 K 或放宽精度阈值。

根据用户指令，先完成手头实验，然后暂停原优化路线，实施 `C:/Users/yanguahe/.cursor/plans/gemm1_optx_优化方案_910c94ab.plan.md`；optX 深度调优后若仍未达标，再恢复原路线。2026-10-18 曾优先实现上下两半复用同一份 A128/ScaleA128 槽，将四级输入 ring 降为 204 KiB。该方向完整实验的最好单次为 v110 的 114.303 μs，仍明显慢于正式路径。按用户最新指令，现已放弃共享 A 槽的深调，转向 v62 双份 A 布局下的其他 optX 优化；保留已有候选与负结果，不再启动新的共享槽实验。

当前生产入口为 **v190 runtime-E**，在经过验证的 E64/T1536/topk8 和 E65/shared1/T1536/topk9 non-balanced bucket 使用**同一个真实 GEMM1 编译对象**。用户指定 baseline commit **`339589f900b3a880ec447f605886fe803cbf9649`** 不变。最新独立源码树、独立进程三轮交替原完整 benchmark：E64 baseline/v190 中值 **99.896 / 91.688 μs**，改善 **8.208 μs（8.22%）**；E65 baseline/v190 中值 **155.189 / 137.596 μs**，改善 **17.593 μs（11.34%）**。E64 距80 μs差 **11.688 μs**，尚未达标。历史 v153 的98.553 / 91.915 μs单独保留，不与新批次混算，也不把其 E65 fallback 验证当作同对象支持。

v153 使用第三 A/ScaleA register bank，F1 期间读齐 S0/S1，完整 DScnt 退休后先 signal，以最多 8 条 S0 WMMA 覆盖 WG wait；仍有两份 physical A128/ScaleA128 LDS。11 项 routing/token、十项 expert 行数边界、E65/shared1 fallback、dynamic/poison/eager/graph replay，以及正常 selector 的 13 项 scope 和 4 项旧配置 fallback 全部通过。最新补验 **E64/T1536/topk8** 与 **E65/shared1/T1536/topk9**，均为 K7168/inter2048、A4W4、SiLU、无 bias、random non-balanced，seed0/17/43 的有效 payload、ScaleA、GEMM2 和最终 MoE 输出 SHA256 与指定 commit 一致，各自原 FP32 reference 精度正常。后续候选须保持这两种规模 MoE e2e 正确。

v190 将 `n_experts` 改为 runtime `fx.Int32`，统一 TH6 默认编译参数，保持 v153 的 M256 数据布局、四个 N-task、有效正向 K 顺序、READY/RELEASE 和量化输出。seed0/17/43 的两种规模完整 MoE e2e 八项 tensor SHA256均与指定 baseline一致；同一进程在 E64/E65之间切换，cache始终一个artifact，实际完整ELF为 `24bb623573932faadb22ca2d89d3e84a903adc64c9e116cac4138dd5b686f93c`。E65包含1536条shared expert64/weight1 route，未省略shared计算。新增 E64→E65→E64 的动态输入、poison、eager、parent graph/runtime graph replay也已通过。

自动启用仍要求严格 M256 fused/SiLU/FP4、exact partial-M、无 scatter、non-balanced等guard，容量对应 `(E,M)=(64,28672)/(65,30464)`；旧不兼容override及其他bucket继续原路径。最新四 SIMD ATT已绑定**v190实际完整ELF**，32条完整sampled waves中compute占81.1495%、三次N-task handoff占9.1246%，详见 [v190 thread trace分析](moe_gemm1_e64_t1536_optx_v190_thread_trace_analysis.md)。[v153分析](moe_gemm1_e64_t1536_optx_v153_thread_trace_analysis.md)保留历史对象及批次，不用其百分比代替v190动态证据。共享 A LDS 槽方向保持放弃。

正式persistent源码仍冻结，历史验收中值94.888 μs、当天独立复测中值94.279 μs保留，跨批差异不计为代码收益。v128的98.525 μs、v152的96.643 μs与各自证据是此前阶段记录；v154/v155为96.276/96.419 μs，没有扩为采纳验收。曾恢复的K256路线详见 [non-balanced优化记录](moe_gemm1_e64_t1536_nonbalanced_optimization.md)。文中早期状态对应当时实验，当前以本节和后续验收记录为准。

## 基准冻结与前序实验收尾

原 kernel SHA256：

```text
b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790
```

这一冻结 persistent 源码对应本地 commit `3f1213c795a47e68bdde2b661349c77e4afa0bd3`。远端主仓库 HEAD 虽为 `339589f900b3a880ec447f605886fe803cbf9649`，当时 persistent 工作区文件已经被该冻结源码覆盖，不能用远端 HEAD 标识实际 kernel。以下历史最优 94.888 μs 和当日复测属于冻结对照，保留原批次；环境变化带来的基准差不能当作代码优化收益。用户最新指定的 `339589...` baseline 由独立干净 worktree 重新测量，见后文。

| 原命令轮次 | GEMM1 μs | count | logits_diff | rel_l2 |
|---:|---:|---:|---:|---:|
| 1 | 94.684 | 19 | 3.4733e-06 | 2.6357e-03 |
| 2 | 94.279 | 19 | 3.4733e-06 | 2.6357e-03 |
| 3 | 94.043 | 19 | 3.4733e-06 | 2.6357e-03 |
| 中值 | **94.279** | | | |

每轮前置与立即、六秒后置均检查全机 GPU/KFD 状态。sysfs 的 `pp_dpm_sclk` 标记 2400 MHz 仅表示 DPM 状态，不能证明 kernel 内持续工作在该频率。

证据：`.codex_tmp/gemm1_t1536_20261017/logs/optx_baseline_20261018_{1,2,3}.log`、`optx_baseline_freeze_20261018_controller.log`。

v91 四个 padding 跳过候选在 CPU resource gate 均出现 spill，全部拒绝，未运行 GPU correctness 或性能测试，未合入：

| 候选 | VGPR / SGPR | private B | VGPR / SGPR spills |
|---|---:|---:|---:|
| stripe M32 | 501 / 106 | 0 | 0 / 2 |
| stripe M16 | 501 / 107 | 0 | 0 / 16 |
| contiguous M32 | 512 / 106 | 16 | 3 / 0 |
| contiguous M16 | 502 / 107 | 0 | 0 / 14 |

## optX-core 实现与依赖

parent task 仍为 M256/N256/K7168，4 个 persistent N-tasks。`blk_n=((tile_idx*4)+local_n)*256`，同一个 WG 每次横移 1024 列。

内部使用 M128/N256 双窗口 F/S，accumulator 按执行顺序绑定。默认 `first_half=tile_idx&1`，偶数 task h0→h1，奇数 task h1→h0；每个有效 accumulator 的 56 次 K128 accumulation 均按原顺序进行。

每个窗口内部 w2x4，M32 pair 交错：wave_m0 负责 pair0/2，wave_m1 负责 pair1/3。每 wave/window/K128 为 8 条 WMMA；两个窗口及四个 task 合计仍是每 wave 3584 条 WMMA。首版不启用 skip 或 stagger。

G4/S4 使用原四级 64 KiB data ring 和独立 scale ring。producer owner 0/1 改为 F/S 的实际 half；B/ScaleB 只请求一次，两个窗口分别从 LDS 读取。input multicast、source pitch、OOB zero fill 保留。OOB/zero TDM 也计入 tensorcnt。

READY/RELEASE 的关键条件：F 和 S 都读取共享 B；F 完成不能释放 stage。S 最后 LDS reads 完成并由完整 WG rendezvous 发布后，才可 refill q+4。closing WMMA 使用已经捕获的 VGPR operands，允许与 refill 重叠。

连续 ring 的目标协议：最后 S group 逐 stage 发出下一 task I0–I3，再发 output stores。同 wave input TDM 完成有序，跨 wave无序；下个 task 第一个 READY 必须将旧 output 的 2 条请求加入 outstanding 计数。q+4 之后的 READY 同时排除仍在队列中的旧 output，不能直接复制原 carry 常量。早期实现因 spill 未进入 GPU 验证；当前 v6/v7 是每批输入 drain 后再计算的诊断对照，尚未恢复连续 ring，不能当作完整 optX-core 的性能结果。

output 使用两个独立 8.5 KiB window arena，总 LDS 分配为 295936 B（289 KiB）。每 wave/window 发一条 3-D payload TDM，覆盖该 wave 的两个 M32 pairs；ScaleA 仍按有效行 canonical M32 地址直接写 global。首版按 F/S 顺序执行 epilogue，每 task 16 条 output TDM；stagger 后续另验。

硬件依据：MI400 Shader Programming Guide §1.5.1.1（printed p.21）说明 16-bit 及以下 WMMA 在独立 XDL pipe 可与 normal/TRANS coexecute；§4.10.1（p.197）说明同 wave TDM loads/stores 有序完成、跨 wave无序；§4.10.8（p.206）区分 issue→XACK 的 3/wave、6/SIMD 与完成 tensorcnt=63。上述事实不直接保证本实现能隐藏特定数量的 μs，须以 ELF、ATT 和交替性能测试判断。

## CPU 候选与资源检查

以下均为 CPU 编译结果；不能用较少的寄存器数量声称 GPU 已提速。

| 候选 | 结构 | VGPR / SGPR | private B | VGPR / SGPR spill | 状态 |
|---|---|---:|---:|---:|---|
| v1 | G4 展开、连续 ring | 512 / 107 | 500 | 220 / 185 | 拒绝 |
| v2 | owner 作用域 | 512 / 107 | 544 | 453 / 19 | 拒绝 |
| v3 | steady/final 分离 | 512 / 103 | 768 | 2130 / 0 | 拒绝 |
| v4 serial | 单 operand slot 压力对照 | 512 / 103 | 544 | 1215 / 0 | 仅 CPU 对照，禁止直接 GPU 测试 |
| v4 sn0 | 固定 half 压力对照 | 512 / 102 | 608 | 1278 / 0 | 拒绝 |
| v5 rolled | runtime K loop、每批 drain | 375 / 107 | 0 | 0 / 76 | 拒绝 |
| v6 scoped | v5 + owner 作用域 | **400 / 93** | **0** | **0 / 0** | CPU 资源通过；GPU 待验 |
| v7 rolled issue | v6 + runtime producer loop，G 参数化 | **400 / 93** | **0** | **0 / 0** | random17、tiny_expert17 数值通过；无性能数据 |
| v8 continuous conservative | S readers 完成后 refill，next-task I0–I3 连续 | 512 / 96 | 0 | 0 / 0 | 完整数值门禁通过；单次125.716 μs，不采纳 |
| v9 common final | 合并 steady/final producer 与计算 body | 512 / 91 | 660 | 1188 / 0 | CPU gate 拒绝 |
| v10 postcompute refill | v9 改为 S closing compute 后 refill | 512 / 91 | 732 | 1318 / 0 | CPU gate 拒绝 |
| v11 release operands | v8 显式清除迭代尾 operand rmem | 512 / 96 | 0 | 0 / 0 | 资源未改善；未单独 GPU 测试 |
| v12 LLVM unroll limits | v11 + scoped LLVM unroll options | 512 / 96 | 0 | 0 / 0 | 资源未改善；未单独 GPU 测试 |
| v13 tile READY | v8 + 每个 F tile 的 READY | 512 / 96 | 164 | 152 / 0 | CPU gate 拒绝；没有 GPU 测试 |
| v14 tile READY + LLVM unroll limits | v13 + scoped LLVM unroll options | 512 / 96 | 0 | 0 / 0 | random17、dynamic/poison/eager/graph replay通过；单次123.207 μs |
| v15 cross-tile prime | 跨 tile operand prime 与 runtime window 选择 | 512 / 106 | 688 | 967 / 0 | CPU gate 拒绝，没有 GPU 测试 |
| v16 row LDS schedule | SA/SB/B/A0 prime，其余 A 逐 M16 row 读取 | 506 / 96 | 0 | 0 / 0 | random17数值通过；单次156.638 μs，放弃此读法 |
| v17 G4 F stagger | F epilogue 插入 S 末组 | 512 / 104 | 388 | 189 / 0 | CPU gate 拒绝 |
| v18 G4 cross-task stagger | 再将 S epilogue 插入下一 task F 首组 | 512 / 107 | 576 | 179 / 8 | CPU gate 拒绝 |
| v19 G2 common body | 单计算 body 与独立 producer 控制 | 392 / 107 | 0 | 0 / 43 | CPU gate 拒绝 |
| v20 G2 scoped | descriptor 限于 owner 分支 | 392 / 89 | 0 | 0 / 0 | random通过，但事件模型发现末尾 READY bug；未测性能 |
| v21 G2 terminal READY | 修复最后 task 最后 group 的 wait 计数 | 406 / 91 | 0 | 0 / 0 | random17通过；单次116.443 μs，尚未采纳 |
| v22 G2 one-slot F stagger | overlap 段只留一份 operand slot | 512 / 92 | 596 | 293 / 0 | CPU gate 拒绝，没有 GPU 测试 |
| v23 G2 one-slot cross-task stagger | v22 再跨 task 处理 S epilogue | 512 / 105 | 720 | 541 / 0 | CPU gate 拒绝，没有 GPU 测试 |

v6 的先前进度文字曾引用 v5 的 375/107、76 SGPR spill；重新从 v6 `.lowered.mlir` 提取 ELF 后，正确资源为 400/93、全部 spill=0。v6/v7 LDS 均为 295936 B。ISA 中 128 条 static WMMA 包含两个 owner 分支各自的循环 body；应按实际分支和 runtime loop 次数核算动态 3584 条/wave，不能把 static 数量直接当作减少的计算量。

v7 的 G2 已参数化，但尚未验证 G2 连续 ring 的 READY/RELEASE 方程；当前验证先使用 G4。不开放 skip/stagger，避免未实现的开关静默改变行为。

v7 使用默认关闭的正式 selector，验证脚本通过 launcher spy 确认 candidate 确实调用 optX。T1536/random seed17、tiny_expert seed17 的有效 route FP4 payload、ScaleA、GEMM2、最终 MoE 均 bitwise mismatch=0；logits 分别为 3.47663789e-06 和 3.47268136e-06，原阈值未改变。route map 由 atomic sorting 生成，两个独立 producer 调用可改变同 expert 内的排列；首次脚本错误地断言 row map 按元素一致，已改为分别用自身 map 解码每条有效 route，并要求 route 数量、唯一性和行集合完全相同。数值比较不放宽，不屏蔽有效行。

证据：`logs/optx_v7_random17_equivalence.log` 保留初次比较器误判，修正后的通过记录为 `logs/optx_v7_random17_equivalence_r2.log`、`logs/optx_v7_tiny17_equivalence.log`。复现每组单独用 120s timeout 运行 `verify_optx_function.py --seed 17` 或 `--routing tiny_expert --seed 17`，不记录其中耗时为性能。

证据：`.codex_tmp/gemm1_t1536_20261017/logs/cpu_compile_optx_core_v{1,2,3,5,6,7}*.lowered.inspection.json`。v6/v7 source 快照分别保存为 `candidates/optx_core_v6.py`、`candidates/optx_core_v7.py`。

v8 是连续 ring 的保守版本：G4 每组 F 开始前保证本组所有输入完成，S 每个 tile 的最后 DS reads 退休并 WG rendezvous 后才 issue q+4；S closing WMMA 只用已经捕获的 operands。最后 group 逐 stage issue 下一 task I0–I3，之后发布两个 output。successor `tensor_wait(2)` 只保证四个输入，不必等旧 output；第一组 refill 后的 wait0 保证旧 output 在本 task epilogue 重用 arena 前退休。其 address/ordered-queue 模型和四项负对照已通过，记录为 `logs/optx_v8_schedule_model.json`。v8 未做 stagger/skip，group READY 的 wait0 仍较保守；没有把该版宣称为性能最优或已经达到80 μs。

## v8 正确性与原命令筛选（2026-10-18）

v8 的 E64/11 cases 已通过：T1536/random seeds0/17/43；T1536/skew、sparse、hot8、tiny_expert、balanced；T1537/1792/2048/random seed17。M_e=127/128/129/143/144/145/255/256/257 均通过。比较所有有效 route 的 canonical FP4 payload、ScaleA、GEMM2 和最终 MoE，bitwise mismatch 全为0，原 logits 阈值不变。

T1536 的 random、tiny_expert 动态输入、poison、eager/graph replay 通过。输入包括 initial、sign_flip、alternate_values；各输入和 replay 路径重复5次。T2048/random 的同类 replay 也已通过，证据为 `logs/optx_v8_t2048_replay.log`。

E65/shared1 使用物理 topk9（8 routed + 1 shared），optX launcher spy 计数0，保留原 fallback。复验 payload、ScaleA、GEMM2、MoE 全部 bitwise mismatch=0，logits_diff=3.77883117e-06。首次比较器错误套用了 A-preshuffle/canonical M32 布局；该 fallback 实际为 generic row-major payload 和 wmma_rep*16 ScaleA，已按 capture 的 `gemm2_a_preshuffle` 解码所有有效行。首次失败和 CPU/GPU device 误判日志保留为原始证据；没有修改 kernel 或放宽数值比较。

10 个合法 public selector fallback 配置，对原 selector、当前 flag0 和当前 flag1 的原 launcher 调用和完整 ELF 均相同。包括 M192/q6、balanced M192、cluster_m4/w2x2、E128、非融合输出/w2x2、GEMM2 K2048/K3072、partial-direct、partial-drain、E65；未把 public 接口本身不支持的 cluster_m4/w2x4 当作已验证病例。

| 原命令筛选 | GEMM1 μs | count | logits_diff | rel_l2 |
|---|---:|---:|---:|---:|
| optX v8 G4/S4，无 skip/stagger | **125.716** | 19 | 3.4733e-06 | 2.6357e-03 |
| 同批 flag0 原路径 | **94.328** | 19 | 3.4733e-06 | 2.6357e-03 |

单次差值为 +31.388 μs（+33.28%），没有确认收益，因此不采纳、不启用默认路径。两次前置全机 GPU/KFD idle。v8 立即后置无 KFD 进程、VRAM恢复176582656 B，busy=2；六秒复核遇到另一项 E65 benchmark 进程1730198，按6秒继续等待到全机 idle。该占用发生在本次计时结束之后，未杀其他进程，也未将它算作 v8 残留；本样本仍只是筛选证据，后续采纳须三轮交替验证。

v8 源码 SHA256 为 `d9593af7ccf64bf49ee5798e435d261f7a168ef5722a43f40e6fc1bb7a1283b4`。trace 用现有 `my_code/get_att_b_rocm_sdk.sh --all-simd --ana-att` 采集到 `my_code/thread_trace_runs/optx_v8_g4s4_att_20261018_r1`；SIMD0–3 ATT status 和 analyze status 全为0，采样耗时不作为原 benchmark 性能。

复现 host controller：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_optx_perf_host.sh <new-label> 1
bash .codex_tmp/gemm1_t1536_20261017/run_optx_perf_host.sh <another-new-label> 0
bash .codex_tmp/gemm1_t1536_20261017/run_optx_trace_host.sh <new-trace-label>
```

证据：`logs/optx_v8_screen_20261018_r1.log`、`logs/optx_baseline_pair_20261018_r1.log`、`logs/optx_v8_e65shared1_fallback_verify_r3.log`、`logs/optx_selector_case*.json`。资源实验 v13、v15 仅在 CPU 编译，均出现 VGPR spills，已拒绝；未将有 spill 的候选直接用于 GPU。

## 当前资源使用量与 v14/v16 筛选

资源取自实际 ELF 的 AMDGPU metadata，VGPR count 是每 lane 的向量寄存器数量，SGPR count 是每 wave 的标量寄存器数量，LDS 是每 workgroup 的固定分配。v8/v14 为 VGPR512、SGPR96；v16 为 VGPR506、SGPR96。三者 LDS 均为295936 B（289 KiB）：256 KiB data ring、16 KiB scale arena、17 KiB双窗口 output arena；private segment 和 VGPR/SGPR spill 全为0。上述 metadata count 不直接等于硬件按粒度分配后的占用率。

v14 源码 SHA256 为 `0d57f8d071e410f262289bb667ee8c6bac536c97b57c34f0d505835dd7ebc096`。T1536/random17 的有效 payload、ScaleA、GEMM2、MoE 均 bitwise mismatch=0；dynamic/poison/eager/graph replay 通过。原 benchmark 单次为123.207 μs、count19，logits_diff=3.4733e-06、rel_l2=2.6357e-03，没有进入采纳配对。前置全机 idle；立即后置无 GPU/KFD 使用进程、busy3、VRAM176582656 B，六秒后置遇到已退出项及busy2，继续6秒轮询至idle；没有杀其他任务。证据为 `logs/optx_v14_random17_verify.log`、`logs/optx_v14_random_replay.log`、`logs/optx_v14_screen_20261018_r1{,_post6}.log` 和 `_source.sha256`。

v16 源码 SHA256 为 `05ee911dbf0b42e004e5006b4145e88f30946cc006885e24658189e8062f0efb`。CPU ISA审核确认 S closing tile 的最后 DS reads 经 `s_wait_dscnt 0` 和 WG barrier 退休后才发 refill，最后两条 WMMA 使用已捕获 operands。random17 的 payload、ScaleA、GEMM2、MoE 均 bitwise mismatch=0。原命令单次156.638 μs、count19、logits_diff=3.4733e-06、rel_l2=2.6357e-03，明显回退，停止推进此逐 row 读法。前置全机 idle；立即后置无KFD使用进程、busy3、VRAM176582656 B，六秒复查idle。

## G2/S4 与末尾 READY 修复

v20 的 random 比较碰巧通过，但 `check_optx_g2.py` 在保守完成顺序下发现 `missing READY: (3,26,0)`：最后 task 不存在 successor，倒数第二 group 不再发下一 task I0/I1，最后 group 只剩当前 q26/q27；照用 wait3/2 会允许尚未完成的输入被读取。v21 将最后 task 最后 group 改为 wait1/0，其余仍用 wait3/2。该修复没有反转 K，也没有省略有效 WMMA。

v21 的事件模型覆盖112个 input tiles、224次窗口读取、每 owner 120个 TDM requests、每 wave 3584条WMMA。错误 successor 计数、过大的 READY outstanding、terminal overcount、F 后提前 refill、缺少 DS RELEASE、output 提前复用六项负对照均捕获错误。模型发现 entry wait2 在当前逐 tile协议中冗余，不带该 fence 的模型也通过，已记录为 `redundant_entry_fence_control`；没有将其伪装为成功的负对照。

v21 源码 SHA256 为 `17f336d1c8ca09c9a794723e03c321bd81fcef07fe9363f60fffd663bd563919`，ELF `.text` SHA256 为 `50046d9c5b21db89213083da2806dd4cafddc4b81bd3d8952f754303bcf4b20c`。random17 有效payload、ScaleA、GEMM2、MoE bitwise mismatch 全为0；logits_diff=3.47663789e-06。原命令单次116.443 μs、count19、logits_diff=3.4733e-06、rel_l2=2.6357e-03。前置全机idle；立即后置无KFD进程、busy3，六秒复查idle。尚未完成该版11 cases/边界/replay，也未进入采纳配对。

v22/v23 虽明确清除未用 operand rmem，仍因额外计算分支与 epilogue 活跃区间出现 spill，均仅 CPU 编译。下一步改为更小的循环 body 和槽位复用，不能声称 stagger 已获得性能收益。

复现 v21 时显式设置 `AITER_GFX1250_GEMM1_OPTX_KGROUP=2`；当前 selector 的 optX 开关仍默认为0，开发候选 v21 只接受G2，不能用 flag1/KGROUP默认4运行该快照。证据保存在 `logs/optx_v21_g2_schedule.json`、`logs/optx_v21_random17_verify.log`、`logs/optx_v21_g2_screen_20261018_r1{,_post6}.log`、`logs/cpu_compile_optx_core_v{17,18,19,20,21,22,23}*.lowered.inspection.json`。

## A/ScaleA 驻留情况与 S5 下一步

截至 G2/S4 v21，A/ScaleA LDS 用量没有减少：每个逻辑 stage 同时保留 A_F=16 KiB、A_S=16 KiB、ScaleA_F=1 KiB、ScaleA_S=1 KiB。S4 合计 A=128 KiB、ScaleA=8 KiB；B=128 KiB、ScaleB=8 KiB，独立 output=17 KiB，总计289 KiB。G2仅改变K-group，不改变ring分配。

正在实施S5：每 stage仅一份 A128=16 KiB、ScaleA128=1 KiB，F 的 DS reads 退休后再装 S 覆盖同一槽位；B=32 KiB、ScaleB=2 KiB驻留到S读完。五级ring推导为240 KiB data +15 KiB scale +17 KiB output=272 KiB，其中A=80 KiB、ScaleA=5 KiB。这个数字是布局推导，须以新ELF确认；尚不能当作已实现或已经提速。

新协议按 `(tile_idx*28+kt)%5` 标识跨task物理stage，不能沿用 `&3` 或每task重置stage。A/ScaleA owner在F、S分别issue，B/ScaleB只在F issue，所以owner的tensorcnt不同；将逐owner推导READY，并把前一tile的DS RELEASE与下一tile的READY rendezvous合并，覆盖terminal/no-successor、partial/zero M、output generations与负对照。依据为SPG §4.10.1（printed p.197）的同wave有序完成、跨wave无序和EXEC忽略；模型仍须接受ISA/GPU复验。

### S5 已编译的槽位复用候选

S5 的实现和 `check_optx_s5.py` 事件模型已完成：F读完后的下一READY rendezvous同时退休DS，再装S；S读完后的下一READY rendezvous再装下一代F/B。每stage只留一份A128/ScaleA128，B/ScaleB跨F/S共享。G4与G2事件模型均覆盖896条input requests、每owner120条含output的TDM、224次窗口读取、每wave3584条WMMA。八项负对照捕获错误F/S READY、terminal overcount、提前A覆盖/B refill、缺少DS RELEASE、每task重置模5、output提前复用。事件模型不代替ISA/GPU验证，也不能用列出的M_e边界说明GPU边界验证已完成。

v24首次IR构造因`.select()`的纯Python int参数报`AttributeError`；v25修正类型后发现`tensor_wait`不能接受runtime值，报`std::bad_cast`。CDNA5 ISA §15.5（printed p.294）记载计数在SIMM16[5:0]，v26改为uniform分支选择常量wait0–5。均只进行CPU构造/编译，没有GPU运行。

| S5候选 | VGPR / SGPR | LDS B | private B | VGPR / SGPR spill | 结果 |
|---|---:|---:|---:|---:|---|
| v26，首版槽位复用 | 445 / 107 | **278528（272 KiB）** | 0 | 0 / 46 | CPU gate拒绝 |
| v27，禁machine LICM | 440 / 107 | 278528 | 0 | 0 / 34 | CPU gate拒绝 |
| v28，issue点构造source row descriptor | 440 / 107 | 278528 | 0 | 0 / 23 | CPU gate拒绝 |
| v29，结合v27/v28 | 438 / 107 | 278528 | 0 | 0 / 21 | CPU gate拒绝 |

这些ELF已经确认A=81920 B（80 KiB）、ScaleA=5120 B（5 KiB）的复用布局，总LDS减少17 KiB；仍有SGPR spill，尚无S5性能数据，未采纳。当前进一步拆分physical owner分支以缩短descriptor活跃范围。源快照及ELF metadata分别在`candidates/optx_core_v{24,25,26,27,28,29}.py`、`logs/cpu_compile_optx_core_v{26,27,28,29}_g4_sn1_sk0_st0.lowered.inspection.json`，模型为`logs/optx_v24_s5_schedule.json`。所有远端命令均使用120s timeout。

### S5 单份 A/ScaleA：消除 spill 后的实际结果

后续把完整64-bit source offset、source row、OOB和output descriptor的计算约束在实际issue处，保持地址宽度、原始请求顺序和所有有效K贡献。默认LLVM优化保留，仅禁用machine/postRA LICM；side-effect tied `s_mov_b32`/`s_mov_b64`用于阻止这些值被外提。该方法是编译作用域控制，不能由资源下降直接推断性能收益。

| 候选 | 改动 | VGPR / SGPR | private B | VGPR / SGPR spill | LDS B |
|---|---|---:|---:|---:|---:|
| v30 | 分成八个physical owner分支 | 468 / 107 | 0 | 0 / 21 | 278528 |
| v31 | local OOB与output descriptor | 438 / 107 | 0 | 0 / 8 | 278528 |
| v33 | 再限制完整64-bit source base的活跃范围 | **435 / 106** | **0** | **0 / 0** | **278528** |

v33源码SHA256为`19b3e116d46ca0899ea3f59761a989e4040465445ac7f1aea7910a08fdd223c8`，ELF `.text` SHA256为`f0bf675367f0b4aa48d5e90d40f64c97a1aa7918582df10197507cdddd3f1862`。T1536/random17有效payload、ScaleA、GEMM2、MoE均bitwise mismatch=0；原logits为3.47663789e-06。initial/sign_flip/alternate_values、poison、eager/baseline graph/optX graph各重复5次，payload和ScaleA全部一致。该版尚未完成完整11 cases和M_e边界矩阵。

原命令单次筛选为**138.662 μs**，count19、logits_diff=3.4733e-06、rel_l2=2.6357e-03。前置全机idle；立即后置无KFD进程，busy2、VRAM176582656 B，六秒复查idle。该版本降低了LDS和寄存器压力，但实测明显回退，不采纳。S5的F/S独立READY rendezvous增加同步频率，是否为主要代价仍需ATT验证，不能只凭布局推断。

证据：`logs/optx_v33_s5_random17_verify.log`、`logs/optx_v33_s5_random_replay.log`、`logs/optx_v33_s5_g4_screen_20261018_r1{,_post6}.log`及`logs/cpu_compile_optx_core_v33_g4_sn1_sk0_st0.lowered.inspection.json`。

### padding skip 与 rolled epilogue stagger

padding skip只在wave-uniform scalar branch中跳过完全无效的M16组，保留input TDM、DS reads、READY/RELEASE和量化协议；有效行仍按原顺序累加56次K128。关闭WMMA reuse提示，避免跳过指令后隐式reuse链发生变化。CPU覆盖全部257个tail长度、263168项行检查；ISA实见`s_cbranch_scc1`跨过两条该行组WMMA，未依赖EXEC mask。SPG说明WMMA忽略EXEC，因此普通lane mask不足以完成这项优化。

| 候选 | 结构 | VGPR / SGPR | private B | VGPR / SGPR spill | LDS B | 当前结果 |
|---|---|---:|---:|---:|---:|---|
| v34 | S5，复用已有runtime S loop交错F epilogue | 483 / 107 | 0 | 0 / 11 | 278528 | CPU拒绝，无GPU测试 |
| v35 | G2/S4，padding skip | **425 / 99** | **0** | **0 / 0** | 295936 | random/replay通过；已测M_e边界通过；单次116.409 μs |
| v36 | G4/S5，padding skip | 480 / 107 | 0 | 0 / 6 | 278528 | CPU拒绝，无GPU测试 |
| v37 | v34，epilogue地址在row scope内计算 | 475 / 106 | 0 | 0 / 8 | 278528 | CPU拒绝，无GPU测试 |
| v38 | v36，输入offset/OOB在owner issue内计算 | **479 / 100** | **0** | **0 / 0** | **278528** | random/replay通过；单次140.753 μs |
| v39 | v37，输入offset/OOB在owner issue内计算 | **472 / 95** | **0** | **0 / 0** | **278528** | random/replay通过；单次165.346 μs |

v34/v37/v39没有展开额外closing WMMA body；最后一个S group的每个runtime step完成后处理一组已完成F accumulator，空half退回顺序epilogue。两条output TDM仍在group结束后发布；本阶段尚未实现跨task延迟S epilogue。v39 input offset保留i64，720项地址代数检查无差异，input请求顺序与S5模型相同。资源门禁通过不代表正确性或性能已经通过。

v35源码SHA256为`9b98c59c0f2b402dafe7d2e754b993a929ad5ddcd8f36a024c5c0167d4c8389d`，v39为`9d6c06f25f15b300168785d9b49474ec4c6652e76d10601f99092520cc2a6c50`。复现编译和验证使用`--source candidates/optx_core_v35.py`并设置`KGROUP=2/SKIP=1`，或`--source candidates/optx_core_v39.py`并设置`KGROUP=4/STAGGER=1`。每次命令独立120s timeout，不将正确性运行的耗时记为性能。

### 后续原命令筛选与 B/ScaleB 寄存器复用（2026-10-18）

以下全部使用用户原完整MoE benchmark，random、non-balanced、iters20、GEMM1 count19；每次前置全机GPU/KFD idle，计时期间不并发自有编译、正确性或idle检查。立即后置没有自有KFD进程，VRAM恢复176582656 B，busy可短暂为2或3；六秒后置均恢复idle。logits_diff约3.4733e-06、rel_l2约2.6357e-03，原精度阈值不变。

| 候选 | GEMM1 μs | 与原复测中值94.279的差值 | 结果 |
|---|---:|---:|---|
| v35 G2/S4 padding skip | 116.409 | +22.130 | 相对v21的116.443只差0.034 μs，不能声称skip带来收益 |
| v38 G4/S5 padding skip | 140.753 | +46.474 | 回退，不采纳 |
| v39 G4/S5 rolled F stagger | 165.346 | +71.067 | 回退，不采纳；没有据此宣称跨task stagger已完成 |
| **v40 G2/S4 Bcache** | **108.140** | **+13.861** | 比v35单次少8.269 μs（7.103%），仅为筛选趋势，未做采纳配对 |
| v41 G1/S4 Bcache | 108.888 | +14.609 | 比v40单次慢0.748 μs，较低寄存器数量未自动转化为收益 |

v35已补M_e=0/127/128/129/144/145/256/257的有效payload、ScaleA、GEMM2、MoE bitwise比较，全部mismatch=0；143、255及完整11case矩阵尚未补齐。v38/v39各自random17和dynamic/poison/eager/graph replay通过，完整矩阵未完成。坏候选未为凑样本继续三轮测试。

v40对同一group的F/S复用B和ScaleB的VGPR数据，保留input TDM、READY、RELEASE、output操作。G2保存四个K128的128个B DWORD与8个ScaleB DWORD；S的每K128 LDS读取由20条降为10条，A/ScaleA仍正常读取。v41的G1只保留两个K128的64个B DWORD与4个ScaleB DWORD，ring仍是S4；最后task末尾READY使用3/2/1/0，避免错误沿用G2末组1/0。它们不减少有效K贡献、不改canonical量化或结果布局。

| Bcache版本 | VGPR / SGPR | LDS B | private B | VGPR / SGPR spill |
|---|---:|---:|---:|---:|
| v40 G2 | 493 / 99 | 295936 | 0 | 0 / 0 |
| v41 G1 | 424 / 96 | 295936 | 0 | 0 / 0 |

v40/v41的random17以及initial/sign_flip/alternate_values、poison、eager/baseline graph/optX graph replay均通过；有效四项数值一致。v41另测M_e=128通过；其余GPU边界和完整case矩阵未完成。`check_optx_bcache_schedule.py`补齐112个input tiles、224次窗口读取、每owner120请求、1792次B/ScaleB寄存器代际检查、output arena四代与有序56次K128。八项负对照捕获错误READY、terminal计数、提前refill、缺少DS RELEASE、提前output复用、错误B位置和提前cache复用。该依赖模型不替代ISA和GPU验证。

源码SHA256：v38=`66db62db7176a89fdb37d8d459cb8a32e1b699fc2ee87a3f9c5a9da6b9b72987`；v40=`6351a6c7cbd211b6d9a9b3487d8d7abf38ed78ab0f62711b345c6adfc151ddfc`；v41=`8195870d833509a10c3ad0fab53c806dc13727fe3e5be8c2f8bf910ebf78c00b`。快照位于`candidates/optx_core_v{38,40,41}.py`。

复现性能先复制所选快照到独立optX module并核对SHA，随后在host执行：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_optx_perf_host.sh <new-label> 1 \
 AITER_GFX1250_GEMM1_OPTX_KGROUP=2 AITER_GFX1250_GEMM1_OPTX_SKIP=1
# v41把KGROUP改为1；v38使用KGROUP=4/SKIP=1；v39使用KGROUP=4/STAGGER=1。
```

controller在dsr1_ep4中执行原命令并包120s timeout；不更改benchmark计时。数值复现使用`verify_optx_function.py --seed 17 --source ...`和`verify_optx_replay.py --tokens 1536 --source ...`，每个调用分别包120s timeout。日志为`logs/optx_v{35,38,39,40,41}_*`，资源证据为对应`.lowered.inspection.json`。

### v35 padding skip ATT：WMMA 确实减少，但仍有 LDS/barrier 等待

使用`my_code/get_att_b_rocm_sdk.sh --all-simd --ana-att`，准确匹配`_optx_g2_s4_sk1_st0_sn1`后缀。目录为`my_code/thread_trace_runs/optx_v35_g2_skip_att_20261018_r1`；SIMD0–3的ATT和analysis status均为0。按每task两条output TDM划分，适配skip后动态WMMA数量，不硬套full padding的3584条。

32个完整waves、无incomplete；10个wave执行3136条WMMA，22个执行2688条，相对3584条平均减少**21.09375%**。合计wave lifetime为5895227cycles，96个handoff总613233cycles，均值6387.84375、median5714cycles。

| elapsed phase | cycles | wave lifetime占比 |
|---|---:|---:|
| prologue | 305007 | 5.1738% |
| compute | 4825784 | 81.8592% |
| handoff | 613233 | 10.4022% |
| final epilogue | 151203 | 2.5648% |

stall归属：barrier17.2472%、DS_load11.8934%、s_wait_dscnt12.4352%、tensorcnt1.6736%、WMMA6.2980%。证据支持“skip确实减少无效WMMA，但LDS读取和WG同步仍暴露等待”，不支持把这些并行wave比例直接相加换算成可省μs。capture clock不同，不能直接拿v8或原trace的cycles宣称wall-time收益。原benchmark的116.409 μs没有确认skip收益。

分析脚本为`analyze_optx_skip_trace.py`，报告为`logs/optx_v35_skip_att_analysis.json`。本阶段继续从减少重复LDS读取与跨task epilogue重叠入手，保留原最优路径默认运行。

### G1 合并同步、shared compute 和 window carry（2026-10-18）

本阶段没有更改正式persistent源码、selector默认值、benchmark计时或有效K128累加顺序。F/S在同一group复用B/ScaleB寄存器；A/ScaleA仍按各窗口读取。所有性能均为用户原完整benchmark的单次筛选值，不能代替三轮交替验收。除v47的负载局限单独说明外，前置全机GPU/KFD idle、运行期间无自有并发任务、结束及六秒检查无自有残留。

| 候选 | 方法 | VGPR / SGPR | LDS B | private B | VGPR / SGPR spill | GEMM1 μs |
|---|---|---:|---:|---:|---:|---:|
| v43 | G1跨task stagger，source/output地址局部构造 | 432 / 92 | 295936 | 0 | 0 / 0 | 156.547 |
| v44 snake1 | 合并下一input READY与当前S reader RELEASE | 424 / 97 | 295936 | 0 | 0 / 0 | 105.979 |
| v45 | v44共用compute body | 414 / 107 | 295936 | 0 | 0 / 24 | CPU拒绝，未GPU |
| v46 | v43共用compute body | 432 / 102 | 295936 | 0 | 0 / 0 | 139.789 |
| v47 | v45把input/output/epilogue地址局部构造 | 411 / 84 | 295936 | 0 | 0 / 0 | 112.369，负载局限 |
| v48 | v47完整window carry | 430 / 88 | 295936 | 8 | 1 / 0 | CPU拒绝，未GPU |
| v49 | v47仅F1→S0 carry | 413 / 84 | 295936 | 0 | 0 / 0 | 112.816 |
| v50 | 完整carry，把owner分派移回K-loop外 | 445 / 73 | 295936 | 0 | 0 / 0 | 107.301 |
| **v44 snake0** | 固定h0→h1窗口顺序对照 | 426 / 94 | 295936 | 0 | 0 / 0 | **105.479** |
| v53 | v44使用编译器生成的SCHED_MODE=2 | 424 / 97 | 295936 | 0 | 0 / 0 | 107.339 |

v44每个G1 step在S最后LDS reads之后，用一次`pipeline_fence`认证q+1 input并退休当前reader，然后issue q+4，与S closing WMMA重叠。首task发I0–I3后wait3；successor kt0–2用wait4包含两条旧output；普通step用wait2；末task kt25/26/27用wait1/0/0。不是删除依赖，而是让同一次WG rendezvous承担READY和RELEASE。模型检查112个input、224个窗口读取、每owner120个请求与四代output arena，六项错误变体都被捕获。random17、M_e128/129和dynamic/poison/eager/graph replay通过。

v46把owner分派限制在input issue内部，使compute只保留共同body；静态ISA指令从v43的15302降为7421。random17、M_e128/129和replay通过，筛选值由156.547变为139.789 μs，仍慢于正式路径；这两个单次结果不构成已验收收益。v47进一步在实际owner/row位置构造地址，保留完整i64 pointer/offset，禁machine/postRA LICM，消除v45的SGPR spill。random/replay通过，但112.369 μs结束六秒检查在02:57:28 UTC发现他人的`op_tests/test_moe_ep.py`，下一次六秒检查恢复idle。立即后置没有活跃进程，不足以排除其与计时的潜在重叠；该值仅作初步筛选，不能据此确定回退幅度，未杀该进程。

完整carry以F0→F1→S0→S1保留有效K顺序，在F1读取下一S0(A/ScaleA)，S0读取S1(A/ScaleA)，合并READY/RELEASE后在S1读取下一F0(A/B/scales)。B0在S0消费之后才允许覆盖，B1保留到S1消费。v48延长ScaleB寄存器活跃范围后出现1个VGPR spill，未进入GPU。v49不保留loop-carried F0，只做F1→S0 carry，random/replay通过。v50完整carry恢复owner在K-loop外分派，无spill；random17、M_e128/129、replay通过，四项有效输出bitwise mismatch=0。依赖模型full/partial覆盖448个window-K128 reads、3584项寄存器代际检查和六项负对照，但不替代ISA/GPU。`replace_B0_before_S0`错误变体同时破坏A和B，模型先捕获A错误，不能宣称只隔离证明了B错误。

v44 snake0的random17和replay通过；105.479与snake1的105.979只差0.500 μs，尚未进行稳定性比较。v53只将LLVM的`amdgpu-expert-scheduling-mode`设为已用环境配置，实际ISA进入`SCHED_MODE=2`并保留编译器生成的`s_wait_alu`；没有手工删wait，random/replay通过。正式GEMM1也显式使用normal mode，不能将v53称为修复原路径配置bug。SPG §3.4.9、§4.3.7.4说明mode2的软件dependency责任；本实验的107.339 μs未确认收益。

本阶段所有候选均未优于正式中值94.279 μs，保持默认optX关闭。资源较少、静态ISA较短或跳过WMMA均不能直接代表实测加速。

复现时先把对应`candidates/optx_core_vXX.py`复制到独立optX module并核SHA，然后从host运行：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_optx_perf_host.sh <new-label> 1 \
 AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SKIP=1
# v43/v46加STAGGER=1；v44固定顺序加SNAKE=0。
```

CPU复现：`compile_optx_cpu.py --kgroup 1 --skip 1 --label core_vXX --source ...`及`inspect_compiled_isa.py logs/cpu_compile_optx_core_vXX_g1_sn1_sk1_st0.lowered.mlir`；stagger或固定顺序分别加`--stagger 1`、`--snake 0`。正确性用`verify_optx_function.py --seed 17 --source ...`、边界加`--expert0-routes 128/129`；replay用`verify_optx_replay.py --tokens 1536 --source ...`。每个调用分别包`timeout --signal=TERM --kill-after=5s 120s`。

日志：`logs/optx_v43_g1_cross_screen_20261018_r1*`、`optx_v44_g1_merge_screen_20261018_r1*`、`optx_v46_g1_shared_cross_screen_20261018_r1*`、`optx_v47_g1_shared_merge_screen_20261018_r1*`、`optx_v49_g1_partial_carry_screen_20261018_r1*`、`optx_v50_g1_owner_window_carry_screen_20261018_r1*`、`optx_v44_g1_sn0_screen_20261018_r1*`、`optx_v53_g1_expert_screen_20261018_r1*`。source、ISA和metadata分别由对应snapshot及`.lowered.inspection.json`绑定；不下载`.out`。

### 行边界专用循环与 LDS/WMMA 调度（2026-10-18）

将full-h0条件和有限的h1有效M16组数分派移到K-loop外，保留其余tail的原通用路径。`full_h0`要求每组至少有一行有效，不假设整个h0的128行均有效。对全部257个tail、两种窗口顺序和wave_m的2056项有效行组检查通过；原input请求、量化、tensorcnt及READY/RELEASE条件保持一致。

| 候选 | 变化 | VGPR / SGPR | LDS B | private B | VGPR / SGPR spill | 原命令GEMM1 μs |
|---|---|---:|---:|---:|---:|---:|
| v51 | 全h1 prefix循环复制/shared | 512 / 84 | 295936 | 524 | 418 / 0 | CPU拒绝，未GPU |
| v55 | 只外提full-h0/shared | 411 / 86 | 295936 | 0 | 0 / 0 | 112.074 |
| v56 | 只外提full-h0/owner分派 | 512 / 99 | 295936 | 672 | 503 / 0 | CPU拒绝，未GPU |
| v57 | full-h0+h1两组专用/shared | 426 / 86 | 295936 | 0 | 0 / 0 | 112.490 |
| v58 | full-h0+h1三组专用/shared | 423 / 86 | 295936 | 0 | 0 / 0 | random、M_e128/129/208/209、replay通过；单次113.567 μs |
| v60 | v55，WMMA4→整组DS→WMMA4 hints | 411 / 86 | 295936 | 0 | 0 / 0 | random/replay通过；单次112.688 μs |
| v61 | v55，四组WMMA2交织DS hints | 411 / 86 | 295936 | 0 | 0 / 0 | random/replay通过；单次112.127 μs，post6发现他人E65任务，性能结论受限 |

v55的random17、M_e128/129及dynamic/poison/eager/baseline graph/optX graph replay通过。v57另补M_e128/129/192/193；有效payload、ScaleA、GEMM2与MoE均bitwise mismatch=0，replay通过。v55/v57原benchmark均count19，原sanity保持约3.4733e-06；前置全机idle，立即后置无自有KFD残留，六秒后置idle。尚未补这两版的完整11case矩阵，没有宣称通过完整验收；筛选结果未优于正式路径，不采纳。

v60/v61只改LLVM scheduling hints，source操作、操作数和数据依赖不变；两版静态ISA均5783条，但`.text` hash不同。CPU门禁通过不能代替GPU数值与性能验证。首次在host运行inspection因`librocm_sysdeps_zstd.so.1`缺失失败，改在已有dsr1_ep4内执行后成功；没有重装或修改工具链。

source SHA256：v55=`98a48dfae3975f2b6c9870ec81fa7803c71d6c467016f76ca91ef2cbead4591e`；v57=`302f16150490f4bc75808cf3f707aaa3562d1e34fe825a755baf29d063948728`；v58=`53ffee2e1214a2a63f7431e48bace4b264a94bdd31f074d3345c52fba8b608ad`。脚本为`build_optx_row_dispatch.py`、`build_optx_schedule_variants.py`、`build_optx_prefix_variants.py`、`build_optx_window_hints.py`。复现使用`KGROUP=1/SKIP=1/SNAKE=0`、对应source snapshot及前节CPU/数值/原命令controller。

### v44 snake0 最新 ATT：供数等待仍在（2026-10-18）

准确匹配`_optx_g1_s4_sk1_st0_sn0_bcache_merge`，使用用户脚本`my_code/get_att_b_rocm_sdk.sh --all-simd --ana-att`。SIMD0–3采集及analysis均status0；source SHA256为`8dbdef4cae01b6108a00a814ef996c2183dfdb8ce2844857603797f0f4f54436`。32个完整compute waves、没有incomplete，四task各有两条output TDM。12个wave有3136条WMMA，20个有2688条，相对full3584在本次采样中平均少20.3125%。v35的21.09375%来自另一份采样，不能把两者差异解释为计算遗漏或加速差值。

| elapsed phase | cycles | 采样wave lifetime占比 |
|---|---:|---:|
| prologue | 304288 | 5.8901% |
| compute | 4211643 | 81.5252% |
| handoff | 489577 | 9.4768% |
| final epilogue | 160557 | 3.1079% |
| 合计 | 5166065 | 100% |

96个handoff mean5099.7604、median5053.5cycles。stall归属：`s_wait_dscnt`13.1236%、DS_load8.8273%、barrier12.6245%、WMMA7.3998%、tensorcnt2.6346%、`s_wait_storecnt_dscnt`2.1286%。具体`dscnt15`与`dscnt24`分别为5.2085%与3.9256%，实际ISA位于F/S读取之后、首个WMMA之前。当前报告的`top_pcs`按相同汇编字符串合并，不能将其误称为独立PC地址热点。

这些数据支持继续检查LDS读取与WMMA之间的距离、短padding分支对调度的影响，以及相邻wave对同一physical A half的集中读取；尚无bank-conflict或HBM饱和counter，不能把等待直接定性为bank conflict。SPG §4.7.1.1记载64 KiB segments；§5.3.6记载SIMD pair共享LDS/TA通路和64 DWORD/cycle return bus。硬件事实不保证修改window分配必然加速。

wave lifetime包含并行waves，capture的频率与原基准不同；不得将9.48%与原14.09%的差额乘benchmark时间当作已实现收益。最终仍以原命令105.479 μs筛选结果衡量。前置、立即及六秒后置全机idle，所有单项workload/analysis限时120s，无自有残留。

证据目录：`my_code/thread_trace_runs/optx_v44_g1_sn0_merge_att_20261018_r1`；报告为[ATT分析](../.codex_tmp/gemm1_t1536_20261017/logs/optx_v44_sn0_att_analysis.json)，SHA256=`0db4ad598317ed9a27ebe324b69665a76a40e1b9619371d21ea44729ddf24dae`，已下载本地并核对。复现：

```bash
AITER_OPTX_ATT_KERNEL_SYMBOL=a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist_optx_g1_s4_sk1_st0_sn0_bcache_merge \
bash .codex_tmp/gemm1_t1536_20261017/run_optx_trace_host.sh <new-label> \
 AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SKIP=1 AITER_GFX1250_GEMM1_OPTX_SNAKE=0
timeout --signal=TERM --kill-after=5s 120s \
python3 .codex_tmp/gemm1_t1536_20261017/analyze_optx_skip_trace.py \
 my_code/thread_trace_runs/<new-label> --output <report.json>
```

该分段脚本适用于顺序epilogue，不能直接用于跨task stagger候选。

### 相邻 wave 的 physical A half 与 output publication（2026-10-18）

v62/v63 保留producer的F/S摆放及TDM request顺序，consumer使用 `physical_half = window XOR wave_m`；accumulator的有效行判定和output row同步调整。每个wave的两个temporal windows仍各消费一次physical half0/1，完整有效行/N256集合和每accumulator的56个正向K128 contributions不变，READY/RELEASE位置不变。这减少同时消费同一physical half的wave数，是调度实验；没有bank-conflict counter，不能据此断言已消除LDS bank conflict。

| optX候选 | 方法 | VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | GEMM1单次 μs |
|---|---|---:|---:|---|---:|
| v62 | v44 + wave-window映射 | 424 / 97 | 295936 | 0 / 0 / 0 | **101.422** |
| v63 | shared/local compute + wave-window | 409 / 86 | 295936 | 0 / 0 / 0 | 106.987 |
| v65 | LDS payload publication与global ScaleA drain分开 | 426 / 94 | 295936 | 0 / 0 / 0 | 106.137 |

v65在output TDM之前保留 `s_wait_dscnt(0)` 和完整WG rendezvous，global ScaleA stores在task末尾用 `s_wait_storecnt(0)` 完成；没有删除任何必须的input/output依赖。源码审计确认protected producer、compute和output descriptor functions相同，并通过缺失DScnt、提前publication、缺失最终STOREcnt三项负对照。CPU模型不是GPU执行证明。

三项均通过random seed17、M_e128/129及dynamic/poison/eager/graph replay，有效route的canonical FP4 payload、ScaleA、GEMM2、最终MoE均bitwise一致。route atomic sorting可改变同expert内row_map顺序，比较时分别按各自map解码；未要求row_map逐元素一致，也未屏蔽有效route。原完整benchmark保持random non-balanced和count19；没有进行三轮稳定性验收，所以表中均为筛选值，不能替代正式中值。

v62/v63/v65前置完成全机idle门禁；立即post出现busy1但没有活跃KFD、VRAM回到176582656 B，六秒post均idle。v61六秒post明确发现他人E65 benchmark，约三秒前启动；是否与计时重叠无法从该sample完全确定，因此保留为受限筛选。v58在等待前以及六秒post的首sample发现他人E65验证，随后六秒轮询恢复idle；同样不用于确定稳定收益。原始post信息保留，不能统一表述为所有立即post都busy0。

复现保存的候选时，用对应 `candidates/optx_core_v{62,63,65}.py`，设置 `AITER_GFX1250_GEMM1_OPTX_KGROUP=1`、`AITER_GFX1250_GEMM1_OPTX_SKIP=1`、`AITER_GFX1250_GEMM1_OPTX_SNAKE=0`，验证脚本支持只在进程内导入快照：

```bash
docker exec -w /data/yanguahe/code/wk_sp1/aiter dsr1_ep4 \
  env PYTHONPATH=/data/yanguahe/code/wk_sp1/aiter ENABLE_CK=0 \
  AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SKIP=1 \
  AITER_GFX1250_GEMM1_OPTX_SNAKE=0 \
  timeout --signal=TERM --kill-after=5s 120s \
  python3 -u .codex_tmp/gemm1_t1536_20261017/verify_optx_function.py \
    --source .codex_tmp/gemm1_t1536_20261017/candidates/optx_core_v62.py --seed 17
```

对应原benchmark、selector/env、source SHA和post日志为 `logs/optx_v{62,63,65}_g1_sn0_{wave_window,shared_wave_window,publication}_screen_20261018_r1*`；correctness为 `logs/optx_v{62,63,65}_sn0_*_{verify,replay}.log`，资源及ISA为 `logs/cpu_compile_optx_core_v{62,63,65}_g1_sn0_sk1_st0.lowered.{inspection.json,isa.log}`。方法源码见 `build_optx_wave_window.py`，publication审计见 `audit_optx_output_publication.py`。这些证据已下载本地并核对SHA256，没有下载 `.out`。

optX已测试G4/G2/G1、S4/S5、单份A/ScaleA、B-cache、padding skip、carry、epilogue stagger、专用行循环、compiler scheduling及wave窗口映射，仍未优于冻结K256版。该阶段曾恢复K256，之后按用户的新指令优先验证四级共享A方案，见下一节。不启用默认optX，也不把101.422 μs描述为已达标。live optX留在v61，SHA256 `54239864862dd9a40f4c0b1d30dec2118c3496582c4693c6d7eca2486a40b473`，默认selector为0。

## 四级共享 A128/ScaleA128：204 KiB 输入 ring（2026-10-18）

用户要求上下两半复用同一份 A128/ScaleA128 槽。本轮将该布局落实到独立候选 v104–v109；正式 persistent 源码、live optX 和 selector 默认值保持冻结。每个 stage 的 A128/ScaleA128 先服务 F，再在所有 F LDS reads 退休后覆盖成 S；B/ScaleB 保留到两半消费完成。F/S 是执行顺序，physical row half 可按原 snake 顺序交换。

| 四级 LDS 项目 | 字节 | KiB |
|---|---:|---:|
| A128：4 × 16 KiB | 65536 | 64 |
| B：4 × 32 KiB | 131072 | 128 |
| ScaleA128：4 × 1 KiB | 4096 | 4 |
| ScaleB：4 × 2 KiB | 8192 | 8 |
| 输入 ring | **208896** | **204** |
| 两个 output windows | 17408 | 17 |
| ELF 总 LDS | **226304** | **221** |

`OPTX_STAGES=4`、`PITCH=49152`、`SCALE_STAGE_PITCH=3072`、`SCALE_ARENA_OFF=196608`、`OUTPUT_LDS_OFF=208896`。输入 204 KiB 与 kernel 总 LDS 221 KiB 是不同口径；相对于此前共享 A 的五槽 272 KiB 总 LDS，减少 51 KiB。每槽只驻留一份 A128/ScaleA128，四槽分别是四个不同 K256 输入。

四槽不能只从五槽把常量 5 改成 4。successor task 的 F0–F2 先于旧 task 两条 output TDM，而 F3 在新 task step0 才补发、位于旧 output 后。因此仅 successor 首 group 的 `step<3` 将两条旧 output 纳入 READY outstanding count，step3 排除。沿用五槽计数的负对照在 `p=31` 捕获 `missing owner READY`。stage 为 `(tile_idx*28+kt)&3`；S0 READY 同时承担最后 F reader 的 RELEASE；B refill 仍等待 S reader 退休。有效 K 始终按正向 56 个 K128 累加。

`check_optx_s4_reuse.py` 覆盖 G4/G2、B-cache 和 S0 carry：896 个 input requests、每 owner 120 个含 output 请求、224 次窗口读、3584 项 A/ScaleA 地址检查；无 skip 时每 wave 3584 条 WMMA。13 项错误计数、过早覆盖、缺失 DS RELEASE、寄存器代际及 output 复用负对照全部被捕获。模型只验证抽象依赖和地址，不代替实际 ISA 或 GPU 结果。

| 候选 | 调整 | VGPR / SGPR | private B / VGPR spill / SGPR spill | 原命令 GEMM1 μs |
|---|---|---:|---:|---:|
| v104 G4 | 四级共享 A，完整 padded WMMA | 362 / 86 | 0 / 0 / 0 | 130.013 |
| v104 G2 | 缩小 K-group | 368 / 82 | 0 / 0 / 0 | 135.454 |
| v105 G4 | 无效 M16 scalar skip、局部 source offset、调度 hints | 409 / 82 | 0 / 0 / 0 | 134.358，负载限制 |
| v106 G2 | v105 + B/ScaleB register cache | 480 / 78 | 0 / 0 / 0 | 129.591 |
| v106 G4 | 较大 B-cache | 512 / 78 | 800 / 2775 / 0 | CPU 拒绝，未运行 GPU |
| v107 G4 | v104 + F closing 区提前 S0 READY/RELEASE 与预读 | 407 / 78 | 0 / 0 / 0 | **122.235** |
| v108 G2 | v106 + S0 carry | 480 / 78 | 0 / 0 / 0 | 126.000，负载限制 |

这些是原完整 `--iters 20` 的单次筛选结果，`count=19`；sanity 为 `logits_diff=3.4733e-06`、`rel_l2=2.6357e-03`。全部候选总 LDS 为 226304 B。v107 比 v104 的单次值快 7.778 μs（约 5.98%），未做三轮配对，不能认定为稳定收益；仍慢于正式同源码空闲复测中值 94.279 μs，未采纳。

v106 G2 保留一个 group 的四个 K128 权重及 scale，共 136 DWORD/lane；S 每 K128 的源码 LDS 指令由 20 降为 10，A/ScaleA 正常读取，所有 TDM、READY/RELEASE 保留。v107/v108 将 S0 rendezvous 移到最后 F tile 的 K128/1 之前：先退休最终 F LDS reads，发 S closing input，预读 S0 到已空闲的 slot0，与 F 最后 WMMA 重叠。没有删除依赖或改变有效累加顺序。

正确性：v104 G4/G2、v105 G4、v106 G2、v107 G4、v108 G2 的 random17 和 dynamic/poison/eager/baseline graph/optX graph replay 通过；有效 canonical payload、ScaleA、GEMM2、最终 MoE 四项 bitwise mismatch 为 0。replay 使用 initial/sign_flip/alternate_values，各 path 五次。v104 G4 另测 M_e=129。v107 完整 11-case 矩阵累计运行到 120 秒，被 timeout 正常终止，已有 random0/17/43、skew、sparse、hot8 六项通过；之后 tiny_expert、balanced 与 T1537/1792/2048 分批通过，合计补齐 11 项。v107 的 M_e 边界矩阵继续按小批次验证，不将 timeout 记为测试成功。

性能前置均经过全机 GPU/KFD idle gate，计时期间无自有并发编译或验证。立即 post 的短暂 busy 保留原样：v104 G4 为 2，v106/v107 为 3，无 KFD，六秒后恢复 idle；独立 post6 idle。v105 在 06:28:47 UTC 退出，立即 post busy2；06:28:53 出现他人 E65 PID2992487、运行 2 秒，之后独立 post6 又见 PID2993850，再恢复 idle。v108 独立 post6 在 06:55:11 见他人 E65 PID3092113、运行 2 秒，06:55:17 idle。没有杀他人进程；v105/v108 只作受限筛选值，不据此确认回退幅度或稳定收益。

四级版 ATT 见 [四级共享 A thread trace 分析](moe_gemm1_e64_t1536_s4_shared_a_thread_trace_analysis.md)。采样显示 compute 85.9818%、handoff 7.6060%，`s_wait_dscnt` stall 20.3768%、`s_barrier_wait` 18.6755%；目前优先处理 LDS 预读与 WG rendezvous 的暴露等待。资源下降尚未转化为原命令加速。

复现四级候选性能，在 host 上使用新 label 保留既有证据：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_imported_optx_perf_host.sh \
  optx_core_v107 <new_label> \
  AITER_GFX1250_GEMM1_OPTX_KGROUP=4 AITER_GFX1250_GEMM1_OPTX_SKIP=0 \
  AITER_GFX1250_GEMM1_OPTX_SNAKE=1 AITER_GFX1250_GEMM1_OPTX_STAGGER=0
```

controller 在 `dsr1_ep4` 中通过 `bench_imported_optx.py` 导入独立候选并运行原完整 benchmark，证明候选实际执行并在计时后保存其 object。不要用只适用于非 optX 的 `bench_imported_candidate.py` 测此候选。验证用 `verify_optx_function.py --source ... --seed 17` 和 `verify_optx_replay.py --source ... --tokens 1536`；矩阵用 `verify_optx_matrix.py --suite full/boundaries --start-case <inclusive> --stop-case <exclusive>`，每个 workload 分别包 `timeout --signal=TERM --kill-after=5s 120s`。builder 是 `build_optx_s4_reuse.py`、`build_optx_s4_bcache.py`、`build_optx_s4_carry.py`；新 v109 在 `build_optx_s4_group_carry.py` 内将组内下一 tile 的 rendezvous 与预读提前到当前 closing WMMA，正在通过 CPU/GPU 门禁，尚无性能结果。

source SHA256：v104=`0439f568f8a061caf398135c6a5fcc7c05a3082480149248e6d8249fe3be151a`，v105=`2d4e169665a809c0df4fdee4fe3710ce5f25795cc82d2ed3bd60f2831ae3ddce`，v106=`40e11b51b82d607519f43f28d09f27f4dff235484ba0d1d9985388647a4ff278`，v107=`4a2ea7de1a5adeedb557b2bcfb3c66e315f16a5cb0d6ae9e6a294d7667182e40`，v108=`64f5f2d902b4c00cc06fef6fbe1cba688fbb15bd0ff50284dce7729a4a4283c2`。日志位于 `logs/optx_v10{4,5,6,7,8}_s4_reuse_*`；CPU 资源与实际 GPU object 的逐项对照将另行写入 manifest，不凭 CPU 编译结果声称二者已匹配。

### 共享槽方向的最终筛选与停止决定

下列数据补齐 v109 之后的探索。各候选的输入 ring 均为 204 KiB，总 LDS 均为 226304 B；列出的无 spill 版本 private segment、VGPR spill、SGPR spill 均为 0。数字是原完整 benchmark 的单次筛选，不能称为稳定优化收益。

| 候选 | 调整 | VGPR / SGPR | GEMM1 μs | 判断 |
|---|---|---:|---:|---|
| v109 G4 | 组内下一 tile carry | 410 / 78 | 122.439 | 回退，不采纳 |
| **v110 G4** | 下一批 LDS reads 集中提前，再执行八条 WMMA | 476 / 78 | **114.303** | 共享槽方向的最低单次，仍慢于正式路径 |
| v111 G4 | LDS reads 分散于四组 WMMA2 | 423 / 78 | 124.039 | 不采纳 |
| v114 G4 | 分开 WG signal/wait，中间四条 WMMA | 442 / 78 | 114.594 | 不采纳 |
| v115 G4 | v110 + 完全无效 M16 WMMA skip | 473 / 86 | 116.987 | 不采纳 |
| v116 G4 | task 外计算有效行谓词，跳过无效 A LDS reads | 465 / 106 | 130.076 | post6 出现他人任务，受限筛选 |
| v117 G4 | v116 去掉 DS scheduling hint | 465 / 106 | 129.928 | 不采纳 |
| v118 G4 | split barrier 与 early DS 组合 | 466 / 78 | 119.504 | 不采纳 |
| v119 G2 | early DS、group carry、B/ScaleB register cache | 512 / 82 | 无 | random/replay 已通过；等待空闲期间按用户指令取消，未启动 benchmark |
| v120 G2 | v119 + 无效 M16 WMMA skip | 498 / 90 | 无 | 仅 CPU，停止此方向后不再启动 GPU 实验 |

v109–v119 已分别完成 random17 的有效 canonical payload、ScaleA、GEMM2、最终 MoE 四项 bitwise 比较，以及 dynamic/poison/eager/graph replay。v107 另已分批补齐 11-case 和 M_e=0/127/128/129/143/144/145/255/256/257 全部十项边界；先前整批 120 秒 timeout 仍保留为超时记录，不记作成功。v119 性能 controller 一直在前置全机空闲轮询，未进入 `bench_imported_optx.py`；取消日志追加 `cancelled_before_workload=1`，未生成计时结果，也不是 kernel 超时或死锁。

v110 的 114.303 μs 比正式当日中值 94.279 μs 高 20.024 μs（约 21.24%），这是不同筛选与基准批次的说明性比较，没有三轮配对。已有 ATT 表明共享槽减少 LDS 容量后仍暴露供数和 WG rendezvous 等待；不能把 LDS 下降当作性能收益，不能由这些 wait 百分比断言 HBM 或 bank conflict。共享槽方向按用户最新指令放弃，源码与日志保留供复查。

新方向以 v62 的双份 physical A half、289 KiB LDS 为起点。`build_optx_v62_carry_split.py` 生成 v121（F1 期间预读 S0）、v122（signal/wait 间四条 S1 WMMA）、v123（两者组合）。这些候选保留原 input producer、output descriptor 和有效 K128 顺序；split 版本仍先 `s_wait_dscnt(0)`、再 signal，完整 WG wait 后才允许 refill。三项已完成 CPU/ISA、random seed17 及 dynamic/poison/eager/graph replay 门禁，尚无性能结论。

### v121–v126：双份 A carry、split READY/RELEASE 与 wave output publication

v121–v123 的 CPU ELF 均为 VGPR424、SGPR97、LDS295936 B、private/VGPR spill/SGPR spill 全为0。v122 的实际 CPU ISA 中，`s_wait_dscnt 0` → `s_barrier_signal -1` → 四条 S1 WMMA → `s_barrier_wait 0xffff` → refill；wave-uniform 无效 M16 分支可以跳过其中对应的 WMMA。等待、refill 和有效 K 累加顺序没有删除或反转。

首次 random 验证出现 memory fault，历史正确 v62 的复验也失败，故暂时停止全部性能测量。同步诊断确认 v62 的 OPTX launch 完成，之后的 reference FP32 matmul 发生 fault；更重要的是，`diagnose_reference_matmul.py` 不导入 AITER/FlyDSL、不运行任何 MoE kernel，也复现 hipBLASLt 的 `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT32x32x16...ISA1250...` memory fault。此证据证明 reference 存在独立故障，不构成任意候选正确性的证明。

只调用 `torch.backends.cuda.preferred_blas_library("cublas")` 不足以解决完整 fixture 的 fault，因为 rocBLAS 仍可内部选择 hipBLASLt。设置 `TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0` 后，v121–v123 的 random seed17 严格验证全部通过：所有有效 route 的 payload、ScaleA、GEMM2、MoE bitwise mismatch 为0，logits_diff 为 `3.47663789e-06`，原阈值未改变。dynamic/poison/eager/baseline-graph/optX-graph replay 同样通过，initial/sign_flip/alternate_values 各 path 五次。没有修改 PyTorch/ROCm 安装、生产 source 或 benchmark，也没有 reset GPU。后端变量只传给本任务进程；后续新性能对照的 baseline 与 candidate 使用同样设置，reference 仍在计时结束后执行。

故障和恢复的证据保留于 `logs/reference_matmul_only_20261018_r1.log`、`reference_matmul_rocblas_only_20261018_r1.log`、`optx_v62_fault_sync2_20261018.log`、`optx_v121_rocblas_reference_20261018.log`、`optx_v121_nohipblaslt_reference_20261018.log` 和 `optx_v12{2,3}_{function,replay}_nohipblaslt_20261018.log`。失败 exit134 均不记为精度通过或性能；每个进程单独限时120s，无自有残留。

v124–v126 在 v62/v121/v123 上只调整 output publication：每 wave 的 output TDM source 是自己的1024-byte payload slice，全部 LDS 写入也来自同一 wave。将该处完整 WG rendezvous 改为显式 `s_wait_dscnt(0)`，保留 task 末 `s_wait_storecnt(0)` 和 kernel 末 `tensor_wait(0)`；所有 input READY/RELEASE rendezvous、TDM 队列距离、source pitch、output descriptor、有效 K128 顺序保持相同。

`build_optx_wave_publication.py` 枚举每 window 8192 个 payload bytes，确认没有跨 wave 读取；全部257种 tail、两种 first-half 顺序的514项有效 output coverage 通过。三个负对照分别捕获读取其他 wave 区域、缺 LDS completion、缺最终 ScaleA STOREcnt completion。这是 source/address 模型，实际 ISA/GPU 验证单独进行。硬件依据为 MI400 SPG §4.3.7.2.4、§4.7.2（printed p.160–161）和 §4.10.1（p.197）：LDS stores 由 DScnt 退休，TDM 不与其它类型 memory instructions 自动排序；因此不能把 `s_wait_dscnt(0)` 一并省去。

| 候选 | 方法 | CPU VGPR / SGPR | LDS B | private / spills | 当前验证 |
|---|---|---:|---:|---|---|
| v121 | v62 + F1→S0 A/ScaleA carry | 424 / 97 | 295936 | 0 / 0 / 0 | random/replay通过 |
| v122 | v62 + split signal/wait，中间四条 WMMA | 424 / 97 | 295936 | 0 / 0 / 0 | random/replay通过 |
| v123 | carry + split | 424 / 97 | 295936 | 0 / 0 / 0 | random/replay通过 |
| v124 | v62 + wave output publication | 424 / 97 | 295936 | 0 / 0 / 0 | random/replay通过 |
| v125 | v121 + wave output publication | 424 / 97 | 295936 | 0 / 0 / 0 | random/replay通过 |
| v126 | v123 + wave output publication | 424 / 97 | 295936 | 0 / 0 / 0 | random/replay通过 |

CPU `.text` 与真实 GPU signature 不预设逐字节一致：v121 CPU 为 `ce3bdb82...`，实际验证 object 为 `6ca64ce3...`，二者 resource metadata 和 static operation counts 相同。真实性能执行后另保存实际 object，核对其 ISA 与资源，不把 CPU hash 当作 GPU 绑定。

复现正确性，在 container 仓库根目录执行，每项独立120s：

```bash
env PYTHONPATH="$PWD" ENABLE_CK=0 \
 TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0 \
 AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SNAKE=0 \
 AITER_GFX1250_GEMM1_OPTX_SKIP=1 AITER_GFX1250_GEMM1_OPTX_STAGGER=0 \
 timeout --signal=TERM --kill-after=5s 120s \
 python3 -u .codex_tmp/gemm1_t1536_20261017/verify_optx_function.py \
 --source .codex_tmp/gemm1_t1536_20261017/candidates/optx_core_v121.py --seed 17
# replay: 换为 verify_optx_replay.py，使用 --tokens 1536。
```

六个候选均已完成 random seed17 和 dynamic/poison/eager/baseline-graph/OPTX-graph replay，有效 canonical payload、ScaleA、GEMM2、MoE 四项 bitwise mismatch 均为0；不是完整11-case/fallback验收。v124–v126 日志为 `logs/optx_v12{4,5,6}_{function,replay}_wavepub_20261018.log`。原完整 `--iters 20` 筛选继续使用同样的两个 reference 后端变量和 G1/SNAKE0/SKIP1/STAGGER0；每项性能前置全机空闲，计时期间不并发本任务编译或验证。

本轮六项原完整benchmark筛选已完成，全部count19、exit0，sanity为原 `logits_diff=3.4733e-06`、`rel_l2=2.6357e-03`。均未优于正式路径，不采纳；不把单次之间约1–2 μs差值作为已确认的稳定代码收益。

| 路径 | 原命令 GEMM1 单次 μs | 说明 |
|---|---:|---|
| frozen baseline，两个BLAS变量同候选 | 95.097 | 退出后立即发现他人E65验证，保留负载限制 |
| v121 carryS | 104.300 | 不采纳 |
| v122 split4 | 100.628 | 不采纳 |
| v123 carryS + split4 | 102.441 | 不采纳 |
| v124 wave publication | **100.366** | 本轮最低单次，仍慢于正式版 |
| v125 carryS + wave publication | 102.637 | 六秒post发现其他E65/FP8任务，受限筛选 |
| v126 carryS + split4 + wave publication | 101.239 | 不采纳 |

六项候选前置均全机idle，立即post为busy2（v126为3）但没有KFD且VRAM回到176582656 B，随后六秒poll恢复idle。v121/v123/v126的独立post6首sample也idle。v122独立post6首sample在08:54:27 UTC发现E65 benchmark PID3919891运行1秒，晚于本项08:54:15退出；v124在08:52:25的中间post出现他人短任务，晚于08:52:19退出，最终恢复idle；v125独立post6发现E65及FP8验证后继续等到idle。baseline立即post PID3867089已有4秒，本日志不足以排除与末尾测量的潜在重叠，因此不用于确认精确配对收益。所有原始post保留，未杀他人进程。

原日志为 `logs/optx_v12{1,2,3,4,5,6}_dualA_nohipblaslt_screen_20261018_r1{,_post6}.log`，baseline为 `base_nohipblaslt_pair_20261018_r1{,_post6}.log`；已下载并对12项候选日志逐项核对 `dualA_v121_v126_perf_logs.sha256`。真实性能执行的object在每项计时结束后保存为 `<label>_actual_gemm1_0.lowered.mlir`，后续单独inspection绑定实际ISA。

### v127–v130：完整 operand carry 与较宽 payload LDS stores

后续独立快照由 `build_optx_duala_carry_wide_output.py` 生成。v127/v128 在 v126/v125 上把 q+1 的 F0 operands 预读到 q 的 S1 WMMA 期间；q+1 READY 与当前所有 S LDS reader RELEASE 完成后才预读。A0 register bank 在 F0→S0→下一F0 之间轮换，B0 必须等 S0 消费后才覆盖，A1/B1 保留到 S1 完成；q+4 refill 与 q+1 所读的 physical A/ScaleA/B/ScaleB 区域分别不相交。末尾 q27 使用独立不carry的closing body，所有有效accumulator仍按K0→K55贡献顺序更新。216项operand/region审计通过，input producer、queue距离、quant/output functions均保持AST一致。模型不是GPU执行证明。

v129/v130只改output payload的LDS写入。原每lane每M16四条b16 store，试验分别用两条b32或一条b64 store；通过 `V_PERMLANEX16_B32` 在 lane l 与 l^16 交换量化后的packed值，再以 `V_PERM_B32` 保持canonical字节顺序。ScaleA、原FP4转换、有效行mask、output TDM source及DScnt/STOREcnt/TENSORcnt完成条件保持不变。硬件依据为CDNA5 ISA §15.14 `V_PERMLANEX16_B32`（printed p.580–581），它按scalar lane selector从相邻16-lane row读数；这里在wave-uniform有效M16分支内、global ScaleA lane分支之前交换。两种写法各1024项payload row地址/字节审计通过，错置kgrp字节顺序的负对照均被捕获。

这些是待CPU/ISA/GPU门禁的实验设计，没有性能结论。继续保持v62双份physical A half、295936 B LDS，不恢复共享A槽路线，也不修改正式kernel、live OPTX或用户benchmark。源码hash与模型保存在 `logs/optx_v127_v130_carry_wide_payload_model.json`。

v127/v128实际CPU资源分别为VGPR443/448、SGPR均107、LDS均295936 B，private和两类spill均0。v129/v130首版CPU构造失败：native FP4 conversion返回 `ArithValue`，不能直接调用 `.ir_value()`；没有形成GPU执行或性能数据。`build_optx_wide_output_value_fix.py`生成新名字v131/v132，用 `fx.Int32` 规范该返回值，保留首版失败源码和日志；原算术与layout不变。

### v127/v128/v131/v132：原完整命令筛选与实际 object（2026-10-18）

四版均通过random seed17严格对比和dynamic/poison/eager/baseline-graph/OPTX-graph replay：有效route的canonical payload、ScaleA、GEMM2、MoE四项bitwise mismatch均0，原logits阈值保持，replay各路径各输入重复五次。尚未补完整11-case及fallback，不是完整采纳验收。

| 候选 | 方法 | 实际 VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 原命令 GEMM1 单次 μs |
|---|---|---:|---:|---|---:|
| v127 | carryF/carryS + split4 + wave publication | 443 / 107 | 295936 | 0 / 0 / 0 | 133.021，退出前已有他人任务，不能判定回退幅度 |
| v128 | carryF/carryS + merged rendezvous + wave publication | 448 / 107 | 295936 | 0 / 0 / 0 | **98.525**，当前OPTX最低单次，未采纳 |
| v131 | v124 + 两条32-bit payload stores | 425 / 97 | 295936 | 0 / 0 / 0 | 101.401，未采纳 |
| v132 | v124 + 一条64-bit payload store | 427 / 97 | 295936 | 0 / 0 / 0 | 101.857，未采纳 |

全部为原完整random non-balanced `--iters 20`，`count=19`、`workload_exit_code=0`，sanity `logits_diff=3.4733e-06`。baseline/candidate均采用同样的两个reference BLAS环境变量。性能前置全机GPU/KFD idle，计时期间无本任务CPU编译、验证或idle检查。

v127在09:07:03 UTC退出，立即post发现dev_al的PID4094671已运行11秒，说明另一工作负载在本进程退出前已启动。现有日志无法确定其与GEMM1计时段是否重叠，不用133.021 μs判断方案本身回退，保留原始样本。之后每六秒自动等候，09:09:19恢复全机idle。

v128在09:10:16退出，立即post无KFD、VRAM176582656 B、busy2，09:10:22恢复idle；独立post6在09:10:28看到hyg_fyd1 T4608 PID4126976运行6秒及dev_al FP8 PID4128691运行1秒，均在本进程退出后启动，09:10:34再次idle。v131在09:11:17退出，09:11:23看到T6144 PID4137808运行5秒；独立post6又看到T2049任务。v132在09:12:25退出，09:12:31看到T3072 PID4146287运行5秒；独立post6看到T3073任务。它们均晚于对应进程退出，不能据此断言计时被干扰，也不能把post首sample写成全部idle。这些仍只是单次筛选，没有配对收益结论。

计时结束保存的实际object逐项inspection资源与CPU相同；v131 ISA用 `ds_store_2addr_b32`、v132用 `ds_store_b64`。实际 `.text` SHA256如下：v127=`eb70a3747761f55471cdab0245a6fe71d16e0cb3ef33ed2a3b358cb0b7b39240`，v128=`fbefb57e358ab8299c9b2d5088f638838d8a52fb1fab702af9a9601aa13e569a`，v131=`81cb28e86603bc5fa9d188d0c1af948f8215df7c1947961450d12228f12631ab`，v132=`d1843511a901c5e83753e632ec714904af65657767683d827d84ae9f85860f53`。

性能证据为 `logs/optx_v{127,128,131,132}_carrywide_screen_20261018_r1{,_post6}.log`；八项日志已下载本地并逐项核对 `logs/dualA_v127_v132_perf_logs.sha256`。actual inspection为 `logs/optx_v{127,128,131,132}_carrywide_actual_inspection_20261018.log`；验证和replay为 `logs/optx_v{127,128,131,132}_{function,replay}_carrywide_20261018.log`。不下载 `.out`。

复现性能在host调用：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_imported_optx_perf_host.sh \
  optx_core_v128 <fresh_label> \
  TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0 \
  AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SNAKE=0 \
  AITER_GFX1250_GEMM1_OPTX_SKIP=1 AITER_GFX1250_GEMM1_OPTX_STAGGER=0
```

### v133–v136：有效行mask与K128窗口交织

`build_optx_duala_row_mask.py`生成v133/v134，打包四个wave-uniform M16有效标志，不移动WMMA、LDS、TDM或rendezvous；8224项tail/half/wave_m检查及遗漏wave_m的负对照通过。但实际CPU分别为413/107和418/107 VGPR/SGPR，均出现8个SGPR spill，虽然private=0和VGPR spill=0，仍拒绝，未运行GPU数值或性能。

`build_optx_k128_interleave.py`生成v135/v136：F0,F1,S0,S1改为F0,S0,F1,S1，同K128的F/S共享B/ScaleB register cache；每个有效accumulator仍保留K0→K55累加顺序。v135不做跨q carry，v136在S1期间预读下一q+1的F0。原input请求、READY/RELEASE、q+4 refill、epilogue和外部布局保持。256条accumulator序列、112项B代际检查通过；模型不代替实际ISA或GPU结果。

v135/v136 CPU资源分别424/97和432/107 VGPR/SGPR，LDS295936 B，private及两类spill全0。两版random seed17和dynamic/poison/eager/baseline-graph/OPTX-graph replay通过，有效payload、ScaleA、GEMM2、MoE四项bitwise mismatch均0。原完整benchmark单次分别 **99.038 / 98.842 μs**，count19、exit0、原sanity正确，未优于正式版，不采纳。CPU资源证据为 `logs/cpu_inspect_optx_v{133,134,135,136}_20261018.log`，模型为 `logs/optx_v133_v134_row_mask_model.json` 和 `logs/optx_v135_v136_K128_interleave_model.json`。

v135前置全机idle，立即post busy2、无KFD、VRAM176582656 B，六秒poll和独立post6 idle。v136在09:24:15 UTC退出，09:24:21见他人T3584 PID52533运行2秒，独立post6见T3839 PID53836；均晚于退出，最后idle。不能据此断言其计时受干扰，也不把post首sample写成全部idle。四项性能日志已下载并核 `logs/optx_v135_v136_perf_logs.sha256`。实际v136 `.text`为 `49a7714b7374c513005a93817c0e34c7ec3214559e4182f9df602f0193e30ca5`。

### v128四SIMD ATT与逐PC分析（2026-10-18）

使用指定脚本 `get_att_b_rocm_sdk.sh --all-simd --ana-att` 精确抓取v128，32个完整compute waves、incomplete0；四份ATT ELF与98.525 μs性能实际ELF逐字节相同，实际VGPR448/SGPR107/LDS295936 B、无private/spill。compute82.4565%、handoff8.7341%，stall归属barrier13.1057%、WMMA8.4199%、DScnt5.9002%、tensorcnt2.1353%。逐PC绑定capture/code_object/vaddr后确认热点主要在循环rendezvous；大量DScnt waits紧邻下一批LDS loads。这些是等待位置证据，不是HBM饱和、bank conflict或LLVM bug的证明，不直接换算μs。

具体phase、96个handoff分布、3584个barrier区间、object绑定、SHA256、复现和硬件出处写入 [v128 thread trace分析](moe_gemm1_e64_t1536_optx_v128_thread_trace_analysis.md)。旧正式版trace文档保留历史；不下载 `.out`，不恢复已取消的GEMM2 trace归档。

### v137–v140：两组READY/RELEASE合并与typed LDS读取

v137基于v125、v138基于v128，仍按q的F/S交替消费B-cache，偶数q不refill，奇数q认证两条后续input、退休两组reader后再issue两个future inputs。每wave四task的step rendezvous由112降为56，每owner input请求仍112条；first-task wait2、successor q1 wait2包含旧output，其他pair wait0。原input/WMMA/quant/output保持，五项错误READY/early-refill/missing-DS-RELEASE负对照被捕获。

| 候选 | 变化 | 实际VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 原命令GEMM1单次μs |
|---|---|---:|---:|---|---:|
| v137 | pair rendezvous + carryS + wave publication | 424 / 96 | 295936 | 0 / 0 / 0 | 98.995 |
| v138 | pair rendezvous + carryF/carryS + wave publication | 448 / 106 | 295936 | 0 / 0 / 0 | 99.000 |
| v139 | v128全部typed raw LDS reads | 448 / 107，CPU | 295936 | 0 / 0 / 0 | 未测：CPU `.text`与v128相同 |
| v140 | v128仅A/scales typed raw LDS reads | 448 / 107 | 295936 | 0 / 0 / 0 | 99.015 |

v137/v138/v140均通过random17及dynamic/poison/eager/graph replay，有效四项bitwise一致；原完整benchmark count19、exit0、原sanity `logits_diff=3.4733e-06`。均未优于正式版或OPTX最低单次，未采纳，也没有扩成完整11-case/fallback/三轮验收。v140只改读取IR形式，地址、宽度、address-space3和依赖不变；静态ISA由4983降为4939条，DScnt waits从223降为217，资源不变。v139 `.text`为parent CPU的 `b6b98ff4c816002d8f94c11eb1a9bef8d08222fe5c31df2d041b9259696fb7b8`，没有重复GPU。

三项性能前置全机idle、期间无本任务CPU编译/验证。v137在09:45:31 UTC退出，立即busy2但无KFD，09:45:37发现T1280 PID223990运行4秒、09:45:44发现E65 PID225263运行2秒，均退出后启动；独立post6 idle。v138在09:46:40退出，立即busy3但无KFD，09:46:46出现T1535 PID233847运行2秒；post6出现E65/T1024任务，均晚于退出，最后idle。v140在09:51:34退出，立即busy2但无KFD，09:51:40出现T1025 PID276472运行2秒；post6出现T1152 PID277778运行3秒，均晚于退出，最后idle。不删除这些post信息、不杀他人任务。

实际 `.text`：v137=`fc87449dd72402c7b2491821c87b38a73984e6e6d2439966aa95bbeec03e0194`，v138=`92faff8a2792f57124cd612ef7e9bb13d25a2104f7a38ffc525a9ca07000569a`，v140=`1ad7953b5c14a3944d62a1884bc5471c9a9ec3802c0636c5bec4dd190a134d8d`。证据为 `logs/optx_v{137,138}_pairrelease_screen_20261018_r1{,_post6}.log`、`optx_v140_rawlds_screen_20261018_r1{,_post6}.log` 与对应actual inspection。14项pair/CPU/precise-PC证据以及5项v140证据分别下载并核 `optx_v137_v140_evidence.sha256`、`optx_v140_evidence.sha256`。

builder为 `build_optx_pair_rendezvous.py`、`build_optx_raw_lds.py`，模型为 `logs/optx_v137_v138_pair_rendezvous_model.json` 和 `optx_v139_v140_raw_lds_source_audit.json`。复现使用前节controller，以对应 `optx_core_vNNN` 和新label替换，KGROUP1/SNAKE0/SKIP1/STAGGER0及两BLAS变量保持。

### v141–v146：无效A读取与谓词成本对照

`build_optx_valid_lds.py`分别生成v141缓存行谓词、v142逐M16跳过无效A LDS reads、v143两者组合。有效行判定覆盖全部257种tail、first-half/window/wave_m，共8224项；错误window和跳过partial-M16负对照被捕获。原TDM请求、READY/RELEASE、ScaleA/B/ScaleB读取、有效WMMA、quant/output保持。只有M16完全没有有效行时才跳过A读取，slot中旧值不会被有效WMMA消费。

| 候选 | 变化 | VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 结果 |
|---|---|---:|---:|---|---|
| v141 | 两个vector cached masks，use-site scalarize | 462 / 107 | 295936 | 0 / 0 / 6 | CPU拒绝，未GPU |
| v142 | 逐M16跳过无效A reads | 427 / 107 | 295936 | 0 / 0 / 0 | random/replay通过，原命令109.947 μs，未采纳 |
| v143 | cached masks + invalid A reads | 438 / 107 | 295936 | 0 / 0 / 10 | CPU拒绝，未GPU |

v142原完整benchmark count19、exit0、sanity正确，前置全机idle，10:01:49 UTC退出后busy2但无KFD、VRAM176582656 B，六秒poll与独立post6 idle，没有本任务并发CPU/GPU工作。静态ISA5528条、DScnt waits274条，相比v128的4983/223反而增加；这是分支及register调度的静态证据，不单独证明全部回退原因，不进行三轮采纳验证。

为区分分支粒度与读量，`build_optx_valid_m32_lds.py`生成v144两条M32分支只skip A、v145同分支也skip完全无效M32 ScaleA。含任一有效行的M32继续保留两组M16的A读取和原WMMA predicates；8224项coverage审计通过。v145在FlyDSL构造时失败，`slot.sa[sm] = ...`使namedtuple进入`scf.if` state，报 `TypeError: RmemSlot.__new__() missing 3 required positional arguments: 'b', 'sa', and 'sb'`，没有GPU执行或性能数据。独立修正版v146先绑定`scale_target = slot.sa`，再条件写入`scale_target[sm]`，构造和CPU门禁通过；没有改运算或有效行判定。

| 候选 | 变化 | 实际 VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 原命令 GEMM1 单次 μs |
|---|---|---:|---:|---|---:|
| v144 | 两个M32分支，跳过完全无效的A reads | 427 / 107 | 295936 | 0 / 0 / 0 | 104.977 |
| v145 | 同分支跳过ScaleA word，首版 | — | — | — | 构造失败，未GPU |
| v146 | 修正条件ScaleA赋值构造 | 427 / 107 | 295936 | 0 / 0 / 0 | 107.290 |

v144/v146均通过random seed17及dynamic/poison/eager/baseline-graph/OPTX-graph replay；有效payload、ScaleA、GEMM2、MoE bitwise mismatch全部0。原完整benchmark的count19、exit0、sanity `logits_diff=3.4733e-06`均保留，未优于正式版或v128，不进行三轮采纳验收。这一类细分支跳读停止，保留负结果，不继续把读取量下降称为性能收益。

两项性能均前置全机idle、期间无本任务CPU编译或其他GPU验证。v144在10:11:06 UTC退出，立即post busy2、无KFD、VRAM176582656 B；10:11:12看到他人T4608 PID429075运行3秒，晚于退出。独立post6在10:11:31看到另一T4608 PID430971运行6秒，也晚于退出，10:11:43恢复idle。v146在10:19:59退出，立即post busy2无KFD；10:20:05看到他人T4095 PID500900运行3秒，独立post6在10:20:17看到T4608 PID503350运行3秒，均退出后启动，最终idle。未杀他人，也不将后置首sample改写为全机idle。

实际ISA v144为5304条/DScnt waits292条，v146为5368条/322条；v128 CPU对照为4983条/223条。分支和分散ScaleA loads改变静态调度，这不是全部回退的定量归因。实际 `.text` v144=`871c131888a839da3fb8abfb5256d0f4eb4394b9e871aa7639b6f09a7a7ddb99`，v146=`05052ecbc1502feaf9d05778cb2f8001aa4894021d717ff5b804776cb3d930ca`；v146源SHA256=`ed5ebe2f9f5e17a03d7019a957310597e1678c422d3ee763cfb2f2ba4f019b8b`。11项原性能、post、验证、实际inspection和构造失败日志已下载并逐项核对`logs/optx_v144_v146_evidence.sha256`。复现沿用前节controller，以`optx_core_v144`或`optx_core_v146`和新label替换。

### v147–v149：v128 carry路径的LDS/WMMA hints对照

`build_optx_v128_window_hints.py`仅更改scheduling intrinsics；移除该改动与symbol后缀后源码逐字节等于v128。与早期v55实验不同，本组保留F0 loop carry、F1→S0与S0→S1 carry、双B/ScaleB cache和每wave output publication。READY/RELEASE、所有reads、有效WMMA和quant/output均保持。

| 候选 | hints | CPU `.text` SHA256 | 结果 |
|---|---|---|---|
| v147 | 完整DS在八条WMMA之前 | `b6b98ff4c816002d8f94c11eb1a9bef8d08222fe5c31df2d041b9259696fb7b8` | 与v128 CPU `.text`相同，未重复GPU |
| v148 | WMMA4→完整DS→WMMA4 | 同上 | 与v128 CPU `.text`相同，未重复GPU |
| v149 | 四组WMMA2交织DS | `aad7f4e7d8cd63caff7206d5687637b29284f8ccfacc86b689223068678543ad` | random/replay通过，原命令99.120 μs，未采纳 |

三版CPU均VGPR448、SGPR107、LDS295936 B、private和两类spill全0、静态4983条指令。v149原完整benchmark count19、exit0、原sanity正确；未补完整11case/fallback/三轮验收，不能报告稳定收益。证据为 `logs/cpu_inspect_optx_v{147,148,149}_20261018.log`、`optx_v149_{function,replay}_v128hints_20261018.log`、`optx_v149_v128hints_screen_20261018_r1{,_post6}.log`。实际 `.text`为`b748714a36c567db8802ebf915eb11a01758fc2ac895a73b4c8d74e23be50e81`，资源及静态计数与CPU逐项一致。10:27:45 UTC退出，10:27:51的T4608 PID571387运行5秒，晚于退出；独立post6于10:28:03出现T6144 PID573299运行4秒及其他仓库verify PID573810运行3秒，也均在退出后启动，10:28:16最终idle。没有将首post写成全部idle，也没有杀其他任务。11项CPU/验证/性能/实际inspection证据已下载并核对 `logs/optx_v147_v152_cpu_and_perf_evidence.sha256`。

### v150–v152：独立 A/ScaleA register banks 的 S 窗口预读

`build_optx_early_second_banks.py`从v128生成三项独立快照。input LDS仍为两份physical A128/ScaleA128，四级ring和input/output TDM、READY/RELEASE、有效K128累加顺序保持；只调整当前input的S窗口A/ScaleA LDS读取时机及目标register bank。B/ScaleB仍各两份cache，不是恢复已经放弃的共享A LDS槽方向。

| 候选 | 方法 | 实际 VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 原命令 GEMM1 μs |
|---|---|---:|---:|---|---:|
| v150 | 第三A bank，F0期间提前S1 reads；F1期间读S0 | 461 / 107 | 295936 | 0 / 0 / 0 | 100.935 |
| v151 | 四A banks，F0期间提前S0 reads；F1期间读S1 | 495 / 107 | 295936 | 0 / 0 / 0 | 101.726 |
| **v152** | 第三A bank，F1期间读齐S0/S1，S0期间不再发S1 reads | **463 / 107** | 295936 | 0 / 0 / 0 | **96.643** |

三项random seed17均有效payload、ScaleA、GEMM2、MoE bitwise mismatch0，原reference logits为`3.47663789e-06`；dynamic/poison/eager/baseline-graph/OPTX-graph replay全部通过。性能执行原完整random/non-balanced benchmark，`--iters20/count19`、exit0、sanity `logits_diff=3.4733e-06`，所有自有单项120s限时。期间没有本任务CPU编译、数值验证或idle检查。v151 VGPR495在512上限内、无spill，作为独立对照测试；历史某批480目标不是这里的硬件上限。

v152是截至这三项结束的OPTX最低单次，但仍慢于正式路径当日中值94.279 μs；相差2.364 μs只是跨批说明，未做三轮配对，不能报告稳定收益或替换默认kernel。v150/v151没有优势；v152保留供split barrier对照。三项实际`.text`分别为`8f6de33b6d5f782cec61da812a993c69ef7e09efb4d9f0a4f93bfcb773bb13d9`、`a504e4de2104ef2082d2ec13239e359a81b65a96f3df0ddd25b612aa88e6d902`、`3c929fee8fd936485c793245282329c48b6c82d6cd8c9a464f0fdc269165b0ad`；CPU dummy signature的hash另存，不能代替GPU实际对象。实际静态指令4957/4832/4832、DScnt waits202/164/165，减少静态wait不自动等于运行收益。

三项前置全机idle均通过，退出后首sample分别busy2/2/3、无KFD、VRAM176582656 B，六秒后idle。v150独立post6于10:44:35从首sample idle。v151于10:45:37退出；独立post6首sample10:45:49见另一仓库verify T4608 PID709551运行6秒，约10:45:43启动，晚于本次退出；10:46:07恢复idle。v152于10:47:36退出，独立post6于10:47:48从首sample idle。保留这些差异，不把最终idle改写成所有post从开始idle，也不由退出后启动的任务自动判定计时重叠。

15项GPU验证、性能、post和实际inspection已下载本地，并逐项核对 `logs/optx_v150_v152_gpu_evidence.sha256`。候选源码SHA分别为`ad1bba3ce413a234206b3ac2e5d64761f08e41267849182415b1a4b57fcac3b1`、`df479e07c977395cd150bd78146b210ac5f4e90d6f92a732e84add7303b7a0ac`、`b84b5eec53fc9aff457c7568e6cfa64340653c67bd930b64f44fc3831b5d3f50`。

补强模型 `audit_optx_earlybanks_protocol.py`逐事件检查四task/28Ktiles、448个operand loads、每wave全部有效row的正向56次贡献、每owner120个ordered tensor requests及output代际。它在同一event stream中实际施加提前refill、错误S1 bank/ksl、提前B复用、提前A0 carry、READY overcount、缺DS completion和wait前refill等mutation，全部被相同assertions捕获。报告为 `logs/optx_v150_v155_earlybanks_protocol_v2.json`。原builder末尾手工tuple“不相等”检查不作为完整协议负对照；v2使用source中refill后、剩余S1 WMMA前的真实carry位置。模型仍不替代实际ISA和GPU验证。

host复现，label必须使用新值：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_imported_optx_perf_host.sh \
 optx_core_v152 <fresh_label> \
 TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0 \
 AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SNAKE=0 \
 AITER_GFX1250_GEMM1_OPTX_SKIP=1 AITER_GFX1250_GEMM1_OPTX_STAGGER=0
```

### v153–v155：独立 S banks 上的 split READY/RELEASE 对照

从v152建立三项快照，所有input LDS reads、TDM/output、quant、有效WMMA本体与K顺序保持。先完成同样的`tensor_wait(n)`及`s_wait_dscnt(0)`，再`signal(-1)`；分别将S0的8条、再加S1前两行的4条、或S0/S1全部16条WMMA放在signal/wait之间。完整`wait(-1)`之后才允许input refill和下一F0 LDS reads，没有删除rendezvous或提前复用input槽。硬件依据为SPG §4.3.7.2.4（printed p90）：LDS reads在DScnt退休后返回值可用于VGPR；barrier和跨wavepublication条件单独保持。

v153/v154/v155 CPU均VGPR461、SGPR107、LDS295936 B、private/VGPR spill/SGPR spill全部0，静态指令4768/4829/4797，CPU `.text`分别`81a7a6eb09f5b432ffec7354ae21416d784192e335c2767b7ef3d4818b73fdfd`、`a8d07aaf9b74b76c16c52c2cfe325ee4c1b504843f97f2bfbbab14a7d19c6c00`、`07097df7633eeb968c04e11ca87172ea975894e06a58eb706765046a0abcfde7`。这些是CPU dummy signature，GPU实际资源与hash以下表为准。

| 候选 | signal/wait间最多WMMA数 | 实际VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 原完整命令单次μs |
|---|---:|---:|---:|---|---:|
| v153 | 8 | 463 / 107 | 295936 | 0 / 0 / 0 | 91.121 |
| v154 | 12 | 461 / 107 | 295936 | 0 / 0 / 0 | 96.276 |
| v155 | 16 | 461 / 107 | 295936 | 0 / 0 / 0 | 96.419 |

实际`.text`分别为`8fd295fbcc7ac94f8df983671badb01403b2f1f3c4270c6ec8b9081690d56c85`、`dbc04eb9757b63859d641d3efeab1be6f752486c801ac5b0a9400fdf4be49a55`、`05a43ef6102a8ba6a8582a3a5bc72c05dfd1f4e47100a5b9c0a6697dcd12b03c`。实际ISA的CFG审计每版找到12个WG signal/wait区间，4个prologue没有中间WMMA，8个split区间最多8/12/16条WMMA；所有分支汇合到同一个WAIT，中間没有memory操作或第二SIGNAL。原wave-uniform padding predicates保留。该审计证明placement，不证明descriptor地址或全部数据hazard，数值与replay门禁单独通过。

三项random17及dynamic/poison/eager/baseline-graph/OPTX-graph replay均通过；原完整random/non-balanced性能count19、exit0、sanity `logits_diff=3.4733e-06`。v153随后补齐11case、全部十项M_e边界0/127/128/129/143/144/145/255/256/257，以及E65、E65/shared1 fallback。按每个valid route自己的row map比较，要求同一valid row set，四项bitwise mismatch全部0。单項驗證包120s，分批每项exit0，不用loop结束标记代替检查。

29项screen、GPU实际ISA/inspection和完整matrix证据已下载并核对 `logs/optx_v153_v155_gpu_evidence_20261018.sha256`；fallback两项另存。v153于11:02:55 UTC退出，立即post发现其他仓库T6144 verify PID856738运行1秒，启动约11:02:54、在sanity/perf计算后，现有记录不能保证进程区间完全无重叠，因此91.121 μs只作screen。v154于11:04:33退出，post6于11:04:45见T3840 PID872013运行7秒，约11:04:38启动；v155于11:06:41退出，post6于11:06:53见T4608 PID891178运行5秒，约11:06:48启动，均晚于退出。三个post最终idle，不改写为首sample全idle，也不杀他人。

### v153三轮原命令配对和负载复核

首次独立进程配对原始数值如下，未作为最终采纳批次：

| 轮次 | 基准μs | v153 μs | 候选少μs |
|---:|---:|---:|---:|
| 1 | 95.268 | 91.559 | 3.709 |
| 2，逆序 | 94.760 | 91.447 | 3.313 |
| 3 | 94.439 | 91.682 | 2.757 |
| 中值 | 94.760 | 91.559 | 3.201 |

第二轮基准11:49:22–11:49:39 UTC、立即post发现他人T3840 PID1225845已运行17秒，约11:49:22启动，与基准进程重叠。这一批不能建立严格空闲配对收益，3.38%不作为正式结果。候选三轮精度与actual `.text`一致仍成立。34项原数据和SCLK/实际ISA证据已核`optx_v153_splitS0_paired_20261018_evidence.sha256`，不删除受限样本。

补测入口`bench_imported_optx_context_pair.py`在一个bounded进程内分别`runpy`执行两次原未修改benchmark，warmup、random输入、profiler、`--iters20/count19`和参数全部保持；两次间只保留正常HIP context，避免释放KFD后再初始化造成的空档。第二轮反向。每个pair整体包120s，前置全机idle，期间没有本任务CPU编译、验证或idle检查；actual object在计时结束后保存。

| 补测轮次 | 基准μs | v153 μs | 候选少μs |
|---:|---:|---:|---:|
| 1，bc | 94.625 | 91.017 | 3.608 |
| 2，cb | 94.753 | 91.192 | 3.561 |
| 3，bc | 94.096 | 90.631 | 3.465 |
| 中值 | **94.625** | **91.017** | **3.608（3.81%）** |

三轮count19、exit0、原sanity `logits_diff=3.4733e-06`，有效输出与前述完整验证一致。各轮candidate实际`.text`与screen一致；基准实际VGPR470/SGPR106/LDS278528 B、无spill，`.text`=`507c1a9aedd0db3e3d06b83063870e6c0e56564228c89ff381f5f71b07748985`。21项性能、post、SCLK、actual inspection和candidate ISA证据下载核`optx_v153_contextpaired_20261018_evidence.sha256`。汇总脚本`summarize_earlybanks_evidence.py`分别保留两批，不混合中值，并将首批标为`acceptance_valid=false`。

补测pair分别于11:57:33/11:58:42/11:59:49 UTC退出，立即post无活跃KFD、busy3/4/3及固定VRAM176582656 B。后续他人T6144 PID1308456（11:57:53运行6秒）、T3840 PID1317259（11:58:48运行2秒）、T4095 PID1325948（11:59:55运行5秒）均在pair退出后启动；独立post6还出现另外任务，全部恢复idle。首pair后置出现已退出PID1306652不能由此反推其计时重叠，但这个不完整身份记录保留，不声称所有post从首sampleidle。SCLK仍只记录DPM状态，不能证明kernel内持续2400MHz。

复现同一HIP context的两次原完整命令，使用新label：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_imported_optx_context_pair_host.sh \
 optx_core_v153 <fresh_label> bc
# 下一轮传cb，第三轮传bc；各轮都执行完整--iters20。
```

v153当前是通过完整数值门禁、三轮补测显示稳定收益的最佳候选，距离80 μs差11.017 μs。默认selector及共享正式源码仍冻结，待trace分析与交付集成；不将CPU成功或最低单次写成已完成默认启用。

### v153 四 SIMD ATT 完成与 early refill 负结果

v153 ATT 的四 SIMD 采集、analysis 和实际完整 ELF 绑定均完成，32 条完整 sampled waves、incomplete0。compute80.0184%、handoff9.8785%、final epilogue2.9202%，WG barrier attributed stall7.8580%；3584 个相邻 WAIT 完成区间的 cycles mean970.294/median824。这些是并行 lifetime 统计，不是可直接省掉的 kernel μs；相邻 WAIT 区间不是 signal→WAIT 区间。五份 performance/ATT 完整 ELF 一致，十项下载证据核 SHA256 通过。详见 [v153 thread trace分析](moe_gemm1_e64_t1536_optx_v153_thread_trace_analysis.md)。

`build_optx_early_refill_banks.py` 在已缓存全部 S operands 的前提下，把 refill 放到完整 WG WAIT 后、S0 剩余 WMMA 前。v156 在 signal/WAIT 间不计算 S0，v157 计算两行即4条 WMMA；下一 F0 A0/B0 register load 仍等全部 S0 最后消费完成，S1 用独立 bank，原 input requests、有效正向 K、quant/output 保持。

| 候选 | 实际 VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 原完整命令 GEMM1 单次 μs | 结果 |
|---|---:|---:|---|---:|---|
| v156，WAIT 后先 refill，再完成 S0 | 463 / 107 | 295936 | 0 / 0 / 0 | 94.823 | 没有优于 v153，不采纳 |
| v157，WAIT 前4条 S0，后先 refill | 463 / 107 | 295936 | 0 / 0 / 0 | 92.491 | 没有优于 v153，不采纳 |

两版 random17 有效 payload、ScaleA、GEMM2、MoE bitwise mismatch 全部0，dynamic/poison/eager/baseline-graph/OPTX-graph replay 通过；原 benchmark count19、exit0、sanity3.4733e-06。没有扩成完整矩阵或三轮配对，也不因 refill 更早就报告收益。actual `.text` 分别为 `81b17c36d27df72ec5ebfca09d244a1aca23a6d6d3f958b266776334d64f6068`、`0eb1d96a6eb0206978a104c15427ec27bd499f11486b7a0d195d9ba1dcac4d26`。

两次前置全机 idle，期间没有本任务 CPU 编译、正确性或 idle 检查。v156 在12:20:17 UTC退出，立即 post 无 KFD、busy3，12:20:23 idle；post6 的其他仓库 verify PID1484073 在12:20:24附近启动，晚于退出。v157 在12:29:43退出，立即 post 无 KFD、busy3；12:29:49 他人 T4608 PID1565158 已运行2秒，晚于退出，后续再有新任务，最终 idle。保留 post 首sample和退出后新任务的差异。13项证据已下载、核 `logs/optx_v156_v157_early_refill_evidence_20261018.sha256`。

### v158–v161：pair READY/RELEASE 与早读 S banks 组合

本组沿用 v153 的三 A/ScaleA register banks 和完整 S 预读，把相邻两 stage 的 READY/RELEASE 合并：偶数 q 不 refill，奇数 q 退休两组 reader 后认证后续 input，完整 WG WAIT 之后依次 issue q-1 和 q 的 future input。first prologue wait2，successor q1 wait2 保留旧 output 顺序，其他 pair wait0；每 wave 四 task 的 step rendezvous112→56，input请求和所有有效 K128 顺序保持。逐事件模型覆盖112 input tiles、448 operand loads、14336 register row checks、每owner120 tensor requests、56 signal/WAIT；七/八项错误READY、提前refill、缺DS completion、错误bank、提前A0 carry负对照捕获。模型不代替ISA与GPU。

| 候选 | 方法 | CPU VGPR / SGPR | LDS B | private B / VGPR spill / SGPR spill | 当前门禁 |
|---|---|---:|---:|---|---|
| v158 / v159 | 分支内重复奇偶 S0 WMMA body | 506 / 106 | 295936 | 196 / 608 / 0 | CPU拒绝，没有GPU运行 |
| v160 | 公共 S0 8条 WMMA 在 pair分支外 | 463 / 106 | 295936 | 0 / 0 / 0 | random/replay通过，单次99.759 μs，不采纳 |
| v161 | 公共 S0 前4条在 pair分支外，其余在WAIT后 | 463 / 106 | 295936 | 0 / 0 / 0 | random/replay通过，单次98.960 μs，不采纳 |

v160/v161静态4826条，CPU `.text` 分别为 `aa7d906f94a59b932cb9d32ced17855a47ef45b5325ed6051f699df820937172`、`d0c27bc673e0c5df51127ecb397ad512e7d37abf6e06ea909574dfc234660a9f`。两版源SHA为 `65233c83b5ea27ff487659a5dd05aa6f58795856b2051914e56cd07e5d622212`、`93f1e4f0e35dda9e913449704cd67eb32c018bbdb9aeb0e7ab2a93937a31b11a`。

旧 `audit_earlybanks_split_isa.py` 无约束探索所有分支组合，在 v160 的 `0x2c00 SIGNAL` 后认为可能绕过 `0x2cf0 WAIT` 到 `0x2dac`，exit1日志保留。新增 `audit_pair_common_isa.py` 没有放宽原assert，而是从真实 ISA 的 `s_and_b32`/`s_bitcmp1_b32`/`s_cselect_b32` 奇偶gate推导：SIGNAL意味着saved mask=-1；SIGNAL前和区间中没有谓词/EXEC clobber；八处lane-mask区域的所有路径都恢复EXEC，full wave32 launch的非零EXEC条件保持；WAIT条件据相同saved mask必成立。偶数gate为0，同时绕过WAIT和refill。每版12个静态interval，四个有关联gate，split最多8或4条WMMA；五项实际mutation捕获错误saved mask、谓词/EXEC clobber、WAIT前memory和反转WAIT条件。控制流证明仍不声称descriptor地址或WMMA数据hazard已证明，GPU门禁保持独立。

constrained CPU CFG本地和远端exit0，证据为 `logs/optx_v160_v161_pair_constrained_cfg_20261018.log` 和两个 `.pair_constrained_cfg.json`；旧失败为 `logs/optx_v160_v161_pair_common_cfg_20261018.log`。builder为 `build_optx_pair_earlybanks.py` / `build_optx_pair_common_compute.py`，代际模型为 `audit_optx_pair_earlybanks_protocol.py`。两版random17有效payload、ScaleA、GEMM2、MoE均bitwise mismatch0，dynamic/poison/eager/graph replay通过；原完整benchmark count19、exit0、sanity3.4733e-06，99.759/98.960 μs未优于v153，不扩成采纳配对，最佳仍为91.017 μs。

v160前置12:58:58 UTC全机idle，12:59:42退出；post6于12:59:54发现另一仓库compile-only任务PID1787579已运行4秒，约12:59:50启动，晚于本次退出；没有杀该进程，13:00:07恢复idle。v161前置全机idle，13:01:37退出后无KFD、busy2、VRAM176582656 B，13:01:43与独立post6 idle；计时期间没有本任务CPU编译、验证或idle检查。保留退出后的任务与busy记录，不将结束后的占用改写成计时期间已知重叠。

### v162–v168：减少 N-task、编译调度和分布 S 预读的后续结果

以v153为parent，v162/v163分别让每WG只执行1/2个N-task，同时增加对应WG数，完整M/N覆盖不变；v164/v165/v166只调整候选的LLVM调度选项；v167/v168把S窗口A/ScaleA预读分散到F0/F1，用3/4份register banks。双份physical A128/ScaleA128 LDS槽、有效正向56次K128、producer/output逻辑保持。

| 候选 | 方法 | 实际 VGPR / SGPR | 静态指令数 | 原完整 GEMM1 单次 μs |
|---|---|---:|---:|---:|
| v162 | 每WG 1个N-task | 443 / 68 | 4803 | 94.041 |
| v163 | 每WG 2个N-task | 462 / 106 | 4830 | 93.048 |
| v164 | 禁machine/postRA LICM | 451 / 98 | 4790 | 92.432 |
| v165 | 禁post machine scheduler | 463 / 107 | 4772 | 92.736 |
| v166 | expert scheduling mode1 | 463 / 107 | 4863 | 93.008 |
| v167 | 三register banks分布S预读 | 461 / 107 | 4797 | 96.740 |
| v168 | 四register banks分布S预读 | 495 / 107 | 4736 | 95.810 |

七项actual LDS均295936 B，private/VGPR spill/SGPR spill全部0；random17有效payload、ScaleA、GEMM2、MoE四项bitwise一致，dynamic/poison/eager/graph replay通过。原完整benchmark count19、sanity3.4733e-06、exit0，actual barrier CFG通过。全部没有超过最佳v153，不扩为采纳配对。资源下降、handoff减少或静态指令变少都不自动等于提速。

v167的负载限制单独保留：13:46:02 UTC退出，13:46:08后置记录中有已退出PID2083222和身份不完整的KFD PID2082811，无法证明这些进程的启动边界，不能声称该样本严格无重叠。后续恢复idle不能消除这项限制。所有性能期间没有本任务CPU编译、验证或并发idle检查。

### v169–v177：WMMA次序与有效行谓词、窗口映射

v169/v170固定N forward/reverse，v171改为B-major，三者只改变独立accumulator之间的WMMA发射次序。v174采用M32粒度skip：M32中任一M16含有效行就保留两组，额外计算只在无效padding。v177把上下physical A half的M32 pairs混入F/S窗口，在M_e=192时窗口有效M16由F[4,2]/S[2,4]变成F[4,4]/S[2,2]，producer、input/output队列和两份A LDS保持。静态模型仅证明覆盖与地址，不预测收益。

| 候选 | 方法 | 实际 VGPR / SGPR | 静态指令数 | 原完整 GEMM1 单次 μs |
|---|---|---:|---:|---:|
| v169 | N forward | 463 / 107 | 4768 | 92.240 |
| v170 | N reverse | 463 / 107 | 4768 | 92.312 |
| v171 | B-major | 463 / 107 | 5312 | 93.803 |
| v174 | M32 skip，无复制body | 463 / 103 | 4451 | 92.525 |
| v177 | 两half混合窗口 | 472 / 104 | 4807 | 94.180 |

五项actual LDS295936 B、private/spills全部0，random17及dynamic/poison/eager/graph replay四项bitwise正确；原benchmark count19、sanity3.4733e-06、exit0。actual CFG每版12个静态signal/WAIT区间，四个prologue无WMMA，八个split区间最多8条，区间无memory。actual `.text`依次为`83ff3dcaf316e0c86626de40e81f494d50221275ad60cc20e180dca4d070a526`、`949dbd46ff2d3e22edb214a65e800e8bc279e67341fd6c2ffed9be542b025243`、`2ca9ca764a44983c49881f33ff97edd785026681c9188ea5e683501bd100ff82`、`487b7b77073c13f945338f2c0cb7cff9d27862f05c8be9c4d081c7b11a6d24fa`、`270221db5dc24f6108ece2b700f135927273ae27b49b1e200c1c2f1a67ce3a29`。CPU dummy signature另存，不能与actual hash混用。

v169于14:02:26 UTC退出，立即无KFD/busy2，14:02:33及post6 idle；v170于14:06:24退出，立即无KFD/busy3，14:06:30他人PID2196845已运行2秒，约14:06:28启动，晚于退出，之后idle；v171于14:11:02退出，立即无KFD/busy3，14:11:08及post6 idle。v174于14:23:18退出，立即无KFD/busy3，14:23:24及post6 idle；v177于14:31:41退出，立即无KFD/busy3，14:31:47及post6 idle。均为单次screen，没有三轮同批采纳收益。

复制快速路径与打包谓词的失败候选保留：

| 候选 | CPU VGPR / SGPR | private B | VGPR / SGPR spills | 静态指令数 | 结果 |
|---|---:|---:|---:|---:|---|
| v172，全有效window与partial分支复制 | 512 / 107 | 460 | 1858 / 0 | 10308 | CPU拒绝，无GPU |
| v173，全有效M32与partial分支复制 | 512 / 107 | 428 | 1290 / 0 | 8882 | CPU拒绝，无GPU |
| v175，单word打包八谓词 | 未编译成功 | — | — | — | `ArithValue`未包`fx.Int32`，保留失败日志 |
| v176，修正scalar包装 | 451 / 107 | 0 | 0 / 10 | 4877 | SGPR spill，拒绝，无GPU |

v172/v173的CFG也因signal/WAIT区间出现`scratch_load_b128`而拒绝，不能写成整个CPU probe通过。v175是实验构造错误，不是生产kernel bug；修正后仍须通过资源门禁。

### v178–v183：最终 S1 WMMA 与 F epilogue 交错

v178/v179保持最后stage的`tensor_wait(n)`、DScnt退休、完整WG signal/WAIT和input refill；最后S1的四个M16分别执行WMMA后，穿插已完成F的SiLU/BF16/FP4/ScaleA/LDS staging，全部完成后DScnt退休再F output TDM，最后原S epilogue。I0–I3仍先于F/S output TDM，有效K顺序和canonical epilogue AST保持。v179加入schedule hints，但两版CPU `.text`相同：`06195fe6cf76d5527b617af819b78fa4aab49d5da033a12366191b0f59ac50a2`。

| 候选 | CPU VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 静态指令数 | 结果 |
|---|---:|---:|---|---:|---|
| v178，无新hints | 451 / 107 | 295936 | 0 / 0 / 10 | 7560 | CPU资源拒绝，无GPU |
| v179，WMMA/DS store hints | 451 / 107 | 295936 | 0 / 0 / 10 | 7560 | CPU资源拒绝，无GPU |
| v180，owner分支外共享1个M16切片 | 451 / 107 | 295936 | 0 / 0 / 4 | 4849 | CPU资源拒绝，无GPU |
| v181，owner分支外共享2个M16切片 | 451 / 107 | 295936 | 0 / 0 / 4 | 4856 | CPU资源拒绝，无GPU |

v178/v179把epilogue内联进TDM owner dispatch，发生重复展开；v180/v181将最终S1推迟到dispatch之后，只保留一份F epilogue，指令膨胀和SGPR spill减少，仍未达zero-spill门禁。四项barrier CFG均通过12个interval、split最多8条、区间无memory；不由CFG通过推断全部WMMA data hazards或GPU结果正确。

v180/v181逐事件模型覆盖0..256有效行、两种first half及两种wave_m，共2056个case/34952事件；在同一个validator上施加缺S1、重复S1、output早于I3、缺DS完成和缺F最后K五项mutation，全部捕获。它不替代实际ISA和GPU验证。v182/v183在v180/v181上禁machine/postRA LICM，消除了剩余spill，随后完成random17、dynamic/poison/eager/baseline-graph/OPTX-graph验证，有效payload、ScaleA、GEMM2、MoE四项bitwise mismatch均0。

| 候选 | 实际 VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 静态指令数 | 原完整 GEMM1 单次 μs |
|---|---:|---:|---|---:|---:|
| v182，共享1个M16交错切片、禁LICM | 453 / 102 | 295936 | 0 / 0 / 0 | 4830 | 91.464 |
| v183，共享2个M16交错切片、禁LICM | 452 / 102 | 295936 | 0 / 0 / 0 | 4862 | 93.113 |

原完整benchmark两项count19、exit0、sanity3.4733e-06；两项actual CFG通过12个signal/WAIT区间，四个prologue无WMMA、八个split最多8条，signal→WAIT无memory。该审计不证明完整WMMA数据hazard；手工检查交错region未看到F/S accumulator重叠，没有宣称已完成自动hazard证明。两项都没有显示优于v153的进一步收益，不扩完整矩阵或三轮配对，停止最终S1/F epilogue交错方向。

v182 actual `.text`为`106b73d1c700f61bd5a898c2fda0612b07e6a2db5dbb98083056911a728f4677`，v183为`5964d56b0655e6f6970449092521ec91984980af3735513158405e3b84394eb3`；CPU dummy `.text`分别为`3026fa57390deb7c58bc0458e2504b5d58bd9a69b932c768193c02691ebcaea0`、`abee11ed87af0e75c714a6c8f20d66e238c96cf4cbb04ca36db2ed92ad34a83c`，不可混用。v182前置14:52:40 UTC idle，14:53:23退出后无KFD、busy3，14:53:29恢复idle，独立post6于14:53:35 idle。v183前置14:59:14 idle，14:59:58退出后无KFD、busy2，15:00:04及独立post6于15:00:10 idle。性能期间另一CPU会话闲置，没有并发编译、数值或idle检查。

实验源码、GPU logs、post、actual inspection/CFG与CPU拒绝证据均保留在`.codex_tmp/gemm1_t1536_20261017/`。下载只包括所需logs/JSON，没有下载`.out`。本组66项下载清单为`logs/optx_v169_v181_download_evidence_20261018.sha256`，同步远端核对结果另存`logs/optx_v169_v181_download_sha_verify_20261018.log`。

每个编译、inspection、CFG、数值或性能单项仍包120s timeout；性能前全机idle，有占用每6秒等待，计时期间另一CPU会话闲置。以下host复现使用新label：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_optx_cpu_probe_host.sh \
 optx_core_v182 <fresh_cpu_label>
bash .codex_tmp/gemm1_t1536_20261017/run_imported_optx_verify_host.sh \
 optx_core_v169 <fresh_function_label> function --seed 17
bash .codex_tmp/gemm1_t1536_20261017/run_imported_optx_perf_host.sh \
 optx_core_v169 <fresh_perf_label> \
 TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0 \
 AITER_GFX1250_GEMM1_OPTX_KGROUP=1 AITER_GFX1250_GEMM1_OPTX_SNAKE=0 \
 AITER_GFX1250_GEMM1_OPTX_SKIP=1 AITER_GFX1250_GEMM1_OPTX_STAGGER=0
```

截至本组结束，最佳完整验证仍为v153的91.017 μs；新候选尚未改变默认路径或正式persistent源码。

## 正常入口集成与两批历史复测（2026-10-18）

v153 已从独立实验快照整理为正常模块 `mxfp4_preshuffle_gfx1250_tdm_prefill_persistent_optx.py`，由 `grouped_gemm_mxfp4.py` 在验证过的 E64 bucket 自动选择。正常入口完成 11 项 routing/token、十项 M_e 边界、E65/shared1 fallback、dynamic/poison/eager/graph replay、13 项 selector CPU scope 和四项旧 override fallback；有效 canonical payload、ScaleA、GEMM2、MoE bitwise mismatch 全部为 0。关闭 OPTX 时，parent launcher 调用及完整 ELF 与集成前一致；这些 scope 检查不等于任意 workload 的性能或精度证明。

两批独立的原完整 benchmark 数据如下。对照均为冻结的 `3f1213c795a47e68bdde2b661349c77e4afa0bd3` persistent 源码，**不是用户随后指定的 `339589...` baseline**。每个值为 count19，第二轮逆序；不跨批次混合中值。

| 批次 | 轮次 | 冻结对照 μs | 正常入口自动 v153 μs |
|---|---:|---:|---:|
| 集成验收 | 1 | 94.648 | 91.095 |
| 集成验收 | 2，逆序 | 94.966 | 91.800 |
| 集成验收 | 3 | 94.650 | 90.737 |
| 集成验收 | 中值 | **94.650** | **91.095** |
| 首次用户复测 | 1 | 93.864 | 90.353 |
| 首次用户复测 | 2，逆序 | 95.148 | 91.363 |
| 首次用户复测 | 3 | 94.307 | 92.179 |
| 首次用户复测 | 中值 | **94.307** | **91.363** |

完整原日志、actual objects 和负载样本分别为 `logs/optx_v153_integrated_pair_20261018_r{1,2,3}*`、`logs/optx_v153_requested_retest_20261018_r{1,2,3}*`。汇总为 `logs/optx_v153_integrated_and_retest_summary_20261018.json`；该汇总明确要求独立复核负载原样本，不把这些历史中值与下一节指定 commit 的复测合并。actual v153 `.text` 均为 `8fd295fb...6c85`，与已验收候选及四 SIMD ATT 的完整 ELF 一致。

## 指定 commit baseline 复测与两种规模 MoE e2e（2026-10-19）

用户明确指定 baseline 为 **`339589f900b3a880ec447f605886fe803cbf9649`**。在主机创建独立 detached worktree `.codex_tmp/gemm1_t1536_20261017/baseline_commit_339589`，核对 HEAD、tracked 和 index 均干净，再在该源码树运行其正常生产入口。最优版在主仓库正常入口运行；二者分别启动独立进程，没有 process-local launcher 替换，也没有用远端主仓库 HEAD 代替实际工作区源码身份。

原 benchmark SHA256 两树均为 `10b856f4b9b94fce538ed3c21b01b0211aa238788f414f9768ed846a7ac09c1d`。仅为该 baseline worktree 建立两个未跟踪的已安装扩展 symlink：`module_aiter_core.so` SHA256 `ff0c308d...ad3b`、`module_moe_asm.so` SHA256 `08b5082d...caa3`。两版使用相同容器及 runtime 扩展，没有重新编译或修改 baseline 的 tracked 代码。

| 轮次 | 执行顺序 | baseline339589 GEMM1 μs | v153 GEMM1 μs | 减少 μs |
|---:|---|---:|---:|---:|
| 1 | baseline → best | 98.692 | 91.915 | 6.777 |
| 2 | best → baseline | 98.102 | 90.864 | 7.238 |
| 3 | baseline → best | 98.553 | 92.456 | 6.097 |
| 中值 | | **98.553** | **91.915** | **6.638（6.74%）** |

三轮均快、count19、exit0。原 sanity 均为 `logits_diff=3.4733e-06`、`rel_l2=2.6357e-03`；目标尚未完成，距 80 μs 仍差 **11.915 μs**。每个 workload 包 120s timeout。六次最终前置均全机 GPU/KFD idle，计时期间 CPU 会话闲置，没有本任务编译、正确性运行或 idle 检查。

第三轮 baseline 前出现他人 compile/KFD PID2827954，按六秒间隔等到 16:02:34 UTC 退出并全机 idle 后才启动。六次立即 post 均无 KFD，VRAM 为 176582656 B，但 busy 首读为 2/3；六秒后恢复 idle，独立 post6 首样本也均 idle。保留全部首读，不能写成“所有 post 首样本 idle”。SCLK 前后为 2400MHz 的 DPM 快照，不证明 kernel 内持续该频率。首次在 host inspection 因缺 `librocm_sysdeps_zstd.so.1` 失败，保留失败日志；改在 `dsr1_ep4` 内 inspection 成功。

| 实际 kernel | source SHA256 | VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | `.text` SHA256 |
|---|---|---:|---:|---|---|
| baseline339589 persistent | `b1f0ff87...b6b4` | 480 / 106 | 282624 | 0 / 0 / 0 | `15bc3165...f631` |
| 正常入口 v153 OPTX | `537fe1e9...083e` | 463 / 107 | 295936 | 0 / 0 / 0 | `8fd295fb...6c85` |

baseline selector SHA256 为 `5329bb55...013b`，最优 selector 为 `bd9d07d3...2c15`。同一侧三次完整 ELF 均逐字节一致；v153 还与历史验收、正常入口集成及四 SIMD ATT 完整 ELF 一致。12 个 signal/WAIT CFG 区间审计通过，只证明该控制流范围，不能作为完整 WMMA data-hazard 证明。

### E64 与 E65/shared1 的完整 MoE e2e

按用户新增要求，以相同 random non-balanced fixture、原 FP32 reference、原量化及两树正常 GEMM1/GEMM2 launchers 独立验证。E65/topk9 显式设 `shared_experts=1`，前八个 route 是 E64 routed experts，每个 token 的第九个 route 为 expert64、权重1；共有 13824 条有效 route，其中 1536 条 shared route。E64 为 12288 条有效 route。每例都没有 dropped valid row，out/reference/GEMM2 全 finite。

| E / topk / shared | seed | logits_diff | rel_l2 | baseline 与 v153 的八项 hash |
|---|---:|---:|---:|---|
| 64 / 8 / 0 | 0 | 3.47334604e-06 | 2.63565709e-03 | 全部一致 |
| 64 / 8 / 0 | 17 | 3.47663789e-06 | 2.63690320e-03 | 全部一致 |
| 64 / 8 / 0 | 43 | 3.47728910e-06 | 2.63715163e-03 | 全部一致 |
| 65 / 9 / 1 | 0 | 3.77927844e-06 | 2.74927984e-03 | 全部一致 |
| 65 / 9 / 1 | 17 | 3.77883117e-06 | 2.74911476e-03 | 全部一致 |
| 65 / 9 / 1 | 43 | 3.77714610e-06 | 2.74850219e-03 | 全部一致 |

八项为 route_ids、route_weights、valid_row_set、canonical_payload、canonical_ScaleA、canonical_GEMM2、MoE_e2e、MoE_reference。有效 payload、ScaleA、GEMM2 和 MoE 与 baseline bitwise 一致，并分别满足原 `LOGITS_DIFF_TOL=0.01`；没有放宽标准。最优树 E64 确认实际使用 OPTX，E65 确认使用 generic parent fallback。验证不计时，不能从这些运行报告 E65 性能。

补验过程修正了验证工具的解码问题：baseline commit 的 `verify_gemm1_fused_apre_quant.py::_decode_scale_rows` 仍按旧 wmma_rep 布局读 ScaleA，导致最初只有该项 hash 不一致，而 GEMM2/MoE 已一致。新工具按 consumer canonical M32 解码后补测全部通过，没有修改两树生产 kernel 或 ScaleA tensor。首次修正漏 `device="cpu"`，受 fixture default device 影响发生 index CPU/CUDA 错误；随后补齐并使用新 label 重跑，失败日志仍保留。这两项属于本次验证工具问题，不是新发现的生产 kernel bug。

### 复现与证据

以下在 host 从远端主仓库根目录执行，runner 自动选择源码树，在 `dsr1_ep4` 内运行原完整 benchmark、保持全部原参数，并实施 idle gate 和 120s timeout。label 必须用新名字。

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_commit339589_retest_host.sh" baseline339589 fresh_baseline_repro_01
bash "$task/run_commit339589_retest_host.sh" best_v153 fresh_best_v153_repro_01
# 三轮交替；第二轮先 best_v153 再 baseline339589。

for seed in 0 17 43; do
  bash "$task/run_commit_tree_e2e_host.sh" baseline339589 64 "$seed" "fresh_baseline_e64_${seed}_repro01"
  bash "$task/run_commit_tree_e2e_host.sh" best_v153 64 "$seed" "fresh_best_e64_${seed}_repro01"
  bash "$task/run_commit_tree_e2e_host.sh" baseline339589 65 "$seed" "fresh_baseline_e65_${seed}_repro01"
  bash "$task/run_commit_tree_e2e_host.sh" best_v153 65 "$seed" "fresh_best_e65_${seed}_repro01"
done
```

上例使用合法的新 label，复现时需改成未使用的名称；label只含字母、数字和下划线。性能 runner 保持用户原 `--scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache --experts64 --tokens1536 --topk8 --model-dim7168 --inter-dim2048 --iters20`；两树同设 `TORCH_BLAS_PREFER_HIPBLASLT=0 ROCBLAS_USE_HIPBLASLT=0`，仅避开此前独立复现的 FP32 reference 后端故障，未改变 GEMM1 计时范围。

汇总为 [指定 baseline 复测与 e2e JSON](../.codex_tmp/gemm1_t1536_20261017/logs/commit339589_vs_v153_retest_and_e2e_summary_20261019.json)，生成工具为 `summarize_commit339589_retest.py`。性能证据下载 manifest `logs/commit339589_vs_v153_retest_download_evidence_20261019.sha256` 共 62 项、4222863 B；e2e manifest `logs/commit339589_v153_e2e_download_evidence_20261019.sha256` 共 43 项、1747216 B，全部核对 SHA256。未下载 `.out`。baseline HEAD、worktree 和 runtime 扩展身份另存 `logs/commit339589_baseline_identity_20261019.log`。

## v184–v189 前序实验收尾（2026-10-19）

v184将multicast的early-timeout策略改为merge，v185/v186分别使用B RT/NT hint，v187使用A HT hint。input requests、masks、LDS地址、completion计数、全部有效K贡献和canonical输出保持；CPU均无private/spill，E64 random17完整有效结果与baseline相同。这里的E65记录仍属于当时fallback验收，不能当作OPTX同对象支持。

| 候选 | 原完整 E64 GEMM1 首轮 μs | 最终判断 |
|---|---:|---|
| v184 multicast merge | 91.253 | 三轮配对未确认收益 |
| v185 B RT | 91.726 | 停止 |
| v186 B NT | 92.177 | 停止 |
| v187 A HT | 92.327 | 停止 |

v184的三轮交替比较：v153为91.068/92.508/90.642 μs，中值91.068；v184为91.260/91.663/91.391 μs，中值91.391。不能用首轮91.253或单轮配对取代三轮结论，因此未采纳。151项证据、18027407 B已补齐本地并逐项核SHA256，manifest为`logs/optx_v184_v187_tdmpolicy_download_20261019.sha256`。

v188/v189分别只在整个capacity M-tile无有效expert时绕过task setup或accumulator初始化，CPU及边界模型通过。随后用户要求E64/E65使用同一个真实编译对象，实验在GPU性能筛选前暂停，转入v190 runtime-E；未重跑这两项旧实验，也不填写不存在的性能结果。

## v190 runtime-E 正常入口与完整验收（2026-10-19）

用户指定baseline为commit `339589f900b3a880ec447f605886fe803cbf9649`。以v153的M256、G1/S4、双份physical A LDS、三A/ScaleA register banks、B/ScaleB缓存和四N-task为基础，将`n_experts`改为runtime `fx.Int32`。E64/E65统一已验证的TH6默认编译参数；显式TH override继续保留，kernel名使用`_edyn`。没有额外E65 specialization。

首次接入时E64/E65分别传TH6/TH0，虽结果hash正常却产生两个cache objects；修正共同默认值后，seed0/17/43正常生产入口在E64/E65间切换，cache始终一个artifact、Python identity相同、完整ELF相同。两种规模八项tensor SHA256与baseline一致：`route_ids / route_weights / valid_row_set / canonical_payload / canonical_ScaleA / canonical_GEMM2 / MoE_e2e / MoE_reference`。E65保留每token expert64/weight1的第九条route，共1536条shared route。原`logits_diff<0.01`保持，E64约3.473–3.477e-06、E65约3.777–3.779e-06，输出finite。initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph各五次replay也通过。

| 对象或资源 | 实测 |
|---|---|
| OPTX source SHA256 | `f8be1c42936214a3728fcf8273d0a2b6279ab6731c69fe225730f0bd3b1763f5` |
| selector SHA256 | `db26b0d02a7e4c374b0b29f204e6ef371c390aa4324eb9801f4da50fa6d38a68` |
| actual完整ELF SHA256 | `24bb623573932faadb22ca2d89d3e84a903adc64c9e116cac4138dd5b686f93c` |
| actual `.text` SHA256 | `7b0b0ca23994eba0fd308c388c5925b356051b553ca0e31e7e738a9c4a6805ad` |
| VGPR / SGPR / LDS | 463 / 107 / 295936 B |
| private / VGPR spill / SGPR spill | 0 / 0 / 0 |

三轮独立进程、独立baseline源码树交替运行用户原完整benchmark，原`--iters20`和profiler count19保持。下面仅比较本批；此前v153和冻结persistent数据不混入中值。

| workload / 版本 | GEMM1三轮 μs | GEMM1中值 μs | GEMM2中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|---:|
| E64 baseline339589 | 99.896 / 100.779 / 99.268 | **99.896** | 63.068 | 215.43 |
| E64 v190 | 91.346 / 91.688 / 92.059 | **91.688** | 58.896 | 201.99 |
| E65/shared1 baseline339589 | 155.189 / 155.603 / 153.188 | **155.189** | 90.014 | 300.31 |
| E65/shared1 v190 | 138.050 / 137.586 / 137.596 | **137.596** | 89.921 | 282.38 |

E64改善8.208 μs（8.22%），距80 μs还差11.688 μs；E65改善17.593 μs（11.34%）。正常入口自动选择严格M256/fused SiLU FP4/exact partial-M/non-scatter/non-balanced bucket `(E,M)=(64,28672)/(65,30464)`；旧不兼容override和其他bucket保留原路径。

原benchmark parser首次不识别动态`_edyn`名称而显示GEMM1 missing，保留失败记录；随后只扩展名称解析，计时、count、warmup、fixture、reference和阈值没有变化。完整汇总见[runtime-E性能/e2e/replay/ATT JSON](../.codex_tmp/gemm1_t1536_20261017/logs/runtime190_performance_e2e_replay_att_summary_20261019.json)，174项证据、29038731 B的manifest为`logs/runtime190_v192_evidence_20261019.sha256`。

host复现使用未使用过的label：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_runtime_expert_check_host.sh" 17 runtime190_e2e_repro01
bash "$task/run_runtime_expert_replay_host.sh" runtime190_replay_repro01
bash "$task/run_runtime_expert_perf_host.sh" baseline339589 64 baseline339589_e64_repro01
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 runtime190_e64_repro01
bash "$task/run_runtime_expert_perf_host.sh" baseline339589 65 baseline339589_e65_repro01
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 runtime190_e65_repro01
```

性能runner逐次检查全机GPU/KFD；有占用每六秒等待。计时和ATT期间本任务CPU会话保持闲置，不并发编译、数值或idle polling。后置原samples单独保留，不将busy尾部或他人退出后启动记录改写为首次就空闲。

## v190 新 ATT 与 v191–v197 实验（2026-10-19）

使用指定`get_att_b_rocm_sdk.sh --all-simd --ana-att`完成v190四SIMD capture。五个profiled进程和四份capture的完整ELF都等于上述v190验收对象。32条完整sampled waves、incomplete为空，累计lifetime4164139 cycles：compute81.1495%、三次N-task handoff9.1246%、prologue7.0177%、final epilogue2.7081%。归因stall为barrier11.8098%、WMMA8.1574%、TENSORcnt4.0135%、DScnt2.8953%、DS loads2.3553%、STOREcnt0.9784%。这些是并行wave统计，不能直接换算可省μs或认定HBM/bank conflict；详细PC、负载和限制见[v190 thread trace分析](moe_gemm1_e64_t1536_optx_v190_thread_trace_analysis.md)。

| 候选 | 方法 | CPU VGPR / SGPR | 原完整 E64 / E65 GEMM1 首轮 μs | 决定 |
|---|---|---:|---:|---|
| v191 | 最后expert的M-tiles提前 | 463 / 107 | 92.886 / 138.680 | 无优势，停止 |
| v192 | 全部有效M-tiles逆序 | 463 / 107 | 92.029 / 138.769 | 无优势，停止 |
| v193 | ScaleA task级STOREcnt移到kernel退出前 | 463 / 107 | 91.024 / 138.031 | 扩三轮，稳定收益未确认 |
| v194 | wave连续读prefix、ballot/popcount upper bound | 463 / 107 | 92.405 / 138.229 | 无优势，停止 |
| v196 | physical F高半区在wait前额外算两行S1 | 451 / 107 | — | SGPR spill1，CPU拒绝 |
| v197 | v196约束machine/postRA LICM | 451 / 102 | 96.101 / 146.347 | 无优势，停止 |

除v196外各版private/spills0、LDS295936 B。v191/v192只改变block映射，6291项映射case覆盖2764176个M/N tiles；有效K、TDM、READY/RELEASE、输出保持。v193保留所有ScaleA stores和最终STOREcnt/TENSORcnt完成；不同task写入互不重叠N列，后续GEMM1不读取该输出ScaleA。v194保留runtime E；v197保留原reader退休、完整WG wait和有效正向K贡献。各版筛选前一seed双规模八hash/e2e通过；v193/v194/v197另完成双规模dynamic/poison/eager/两种graph各五次replay，同一个artifact且actual ELF与CPU相同。v193/v194/v197完整ELF依次为`d3f7311841c86d7a23effcf3f33ad833efba18361375c118d633f88b905a10de`、`f4069f47d965c3d5c0fefd7c80fabd1d6be0b5b42b9a3db038ae1344c6472116`、`ef783c27fbd838a3031bc2fa6819eee61257d03a3634237e146a4f4ce10d3bba`。

v193与v190三轮原完整benchmark交替：r1/r3先190后193，r2反向。以下为独立配对批次，不更新v190指定baseline复测中值。

| workload / 版本 | GEMM1三轮 μs | 中值 μs | MoE e2e中值 μs |
|---|---|---:|---:|
| E64 v190 | 91.949 / 91.144 / 92.736 | **91.949** | 203.84 |
| E64 v193 | 91.388 / 92.664 / 91.504 | **91.504** | 203.53 |
| E65 v190 | 137.289 / 138.547 / 137.719 | **137.719** | 284.06 |
| E65 v193 | 136.528 / 137.765 / 136.502 | **136.528** | 283.27 |

E64中值快0.445 μs，但r2慢1.520 μs，尚未证明稳定收益；E65三轮均快，中值快1.191 μs。不能用单次91.024 μs宣称新最优。后续同一HIP context配对也未确认E64收益，v193已停止，生产继续v190；具体补测见下节。

## v198–v204：首 K128 C=0 与 LDS 读取次序（2026-10-19）

CDNA5 ISA §7.7 printed p85允许WMMA C使用inline constant，A/B仍需VGPR。v198剥离q0并以相同`+0.0`初始化，v199保留原zero定义；两项产生大量spill，在CPU拒绝。v200保留原q0..q27 loop，仅K128/0以q==0分支选C0，但LLVM将其合成`v_cndmask_b32`，actual ISA没有inline-zero WMMA。v201在C0分支内加入compiler scheduling barrier，actual ISA真实生成C0 WMMA；v202尝试side-effecting tied identity，产生spill、拒绝GPU。

| 候选 | CPU VGPR / SGPR | private B / VGPR spill / SGPR spill | 静态指令 / DScnt数 | 原完整 E64 / E65 GEMM1 μs |
|---|---:|---|---:|---:|
| v198 q0 peel、去zero | 512 / 106 | 5880 / 5235 / 0 | 见CPU证据 | — |
| v199 q0 peel、保留zero | 512 / 106 | 6008 / 5683 / 0 | 见CPU证据 | — |
| v200 rolled q==0 | 455 / 107 | 0 / 0 / 0 | 5791 / 152 | **114.111 / 179.430** |
| v201 branch-local barrier | 463 / 107 | 0 / 0 / 0 | 5190 / 125 | **105.658 / 163.488** |
| v202 tied identity | 441 / 107 | 196 / 1056 / 0 | 见CPU证据 | — |

v200/v201双规模八hash/e2e、initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph各五次replay均通过，始终一个actual artifact。所有有效accumulator仍按K0→K55贡献、同样初始加零；没有改变精度或省略shared route。v201完整ELF为`57bcc251a08427033dad5353ec6061a725ac465cd00e1227e106d2f20812c450`，`.text`为`81b1f06d40715c7e937b8daa0923b1f2f1efa97db80db02dd97f6542be28c82b`；v200完整ELF为`cf35fbcffc31603cff53a2c57f483c8de42516c8b6e28def0181f4a535507054`。计时count19、sanity通过，实际对象与验收一致。两版明显回退，首K128 C0方向停止，不扩三轮、不合入生产。静态WMMA数包含互斥分支，不能据其增长声称动态重复计算。

v203/v204仅将`load_lds_data`内部独立读顺序改为B→A→scales或A→B→scales。全部读地址/目标的multiset不变，1792项模型case通过；producer、WMMA、K顺序、READY/RELEASE和quant/output AST不变。两版CPU同对象、LDS295936 B、463/107、private/spills0；static instructions为4710/4742，DScnt均106（v190为109）。各一seed双规模八hash/e2e与dynamic/poison/eager/两种graph各五次replay通过。

| 读取顺序筛选批次 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190同批control | **92.200** | **137.719** | 205.37 | 282.49 |
| v203 B→A→scales | 92.409 | 138.786 | 203.17 | 284.71 |
| v204 A→B→scales | 92.637 | 139.000 | 205.62 | 283.22 |

均为原完整`--iters20/count19`、sanity通过、exit0，并绑定各自e2e验收完整ELF；前置全机idle、性能期间无本任务编译或其他测试，post及post6保留。两版在E64/E65均未优于同批v190，停止，不扩三轮。静态等待指令减少不是实际收益，单次e2e起伏不能替代GEMM1目标。control完整ELF仍为v190的`24bb6235...`；v203/v204依次为`1cf865d66da465d99ab42d46f1eede192cf18384f029c25fb5eea252fa43043d`、`b2dfd02b11da5eb1a7b74c54ff7d73a24f3cc70b637b64fd99e4942b2ba8a541`。

本轮218项/62500014 B与46项/15378864 B证据已下载并全部核SHA256，未下载`.out`；manifest分别为`logs/runtime193_v200_evidence_20261019.sha256`、`logs/runtime201_v202_evidence_20261019.sha256`。汇总工具为`summarize_runtime_latency_evidence.py`，结果见[候选CPU、双规模验证、配对与负载JSON](../.codex_tmp/gemm1_t1536_20261017/logs/runtime193_v202_latency_summary_20261019.json)。

复现候选使用其验收label绑定actual ELF，source只在进程内替换，生产源码不改。原接受记录需存在，repro label需换成未使用的名称：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_runtime_latency_cpu_host.sh" 203
bash "$task/run_runtime_expert_check_host.sh" 17 operand_order_e2e_repro01 \
  --source "$task/candidates/optx_core_v203.py"
bash "$task/run_runtime_expert_replay_host.sh" operand_order_replay_repro01 \
  --source "$task/candidates/optx_core_v203.py"
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 operand_order_e64_repro01 \
  --source "$task/candidates/optx_core_v203.py" \
  --acceptance-label operand_order_e2e_repro01
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 operand_order_e65_repro01 \
  --source "$task/candidates/optx_core_v203.py" \
  --acceptance-label operand_order_e2e_repro01
```

CPU runner已有固定label时会拒绝覆盖；对已有候选可直接读取其资源/ISA证据，重新编译需使用新的独立输出label。每个自有compile、验证、benchmark和分析仍限120s，GPU性能前全机idle，计时期间不并发CPU编译或其他测试。共享A LDS方向保持放弃。

## v193 同一 HIP context 配对补测与控制开销实验（2026-10-19）

新增`bench_runtime_expert_context_pair.py`与host runner，在同一bounded进程依次调用两次`bench_runtime_expert_moe.py`，原输入生成、warmup、profiler、`--iters20/count19`和reference均保持。两次间仅保留正常HIP context；每次实际GEMM1完整ELF分别绑定已完成的e2e验收，比较的是v190与v193，不是旧冻结parent。整个pair120s上限，期间CPU会话闲置。

| E64配对轮次 | 顺序 | v190 GEMM1 μs | v193 GEMM1 μs | v193减v190 μs |
|---|---|---:|---:|---:|
| 1 | bc | 90.931 | 90.973 | +0.042 |
| 2 | cb | 90.549 | 91.582 | +1.033 |
| 3 | bc | 91.492 | 91.064 | −0.428 |
| 中值 | — | **90.931** | **91.064** | 中值差+0.133 |

该独立批次没有显示E64稳定收益，v193不采纳；此前独立进程E65三轮改善保留为历史，不能据此替换用户E64目标的最优实现。新批次不能与指定baseline三轮混算或将90.931当作生产稳定中值。首次runner误传不存在的seed0验收label，停在host `sha256sum`，未启动GPU；更正为实际seed17并用新label重跑，失败日志保留。99项/16939008 B的读取次序与context-pair证据已下载并核SHA256，manifest为`logs/runtime203_v204_contextpair_evidence_20261019.sha256`，汇总为`logs/runtime203_v204_contextpair_summary_20261019.json`；原pre/post所有权sample保留。

v205/v206将完整K256 step按2/4步部分展开；每步原F0/F1/S0/S1、28次READY/RELEASE、原refill/carry与final non-carry closing保持。每版4112项event case覆盖1949696次有效K128贡献，正向K0→K55不变。v207/v208将task/kt等wave-uniform条件以`readfirstlane`表达，分别控制tensor-wait选择及额外task/refill guards；lane-dependent epilogue masks保持。681项整数case通过，runtime E ABI不变。实际CPU与GPU结果已完成，见下面v205–v210章节；共享A LDS槽方向保持放弃。

原配对复现，使用新label：

```bash
bash .codex_tmp/gemm1_t1536_20261017/run_runtime_expert_context_pair_host.sh \
  193 64 runtime193_pair_repro01 bc optx_v193_edyn_latency_e2e_seed17_20261019
# 后续传cb；每项整体仍为120s上限。
```

## v205–v210：循环展开与控制条件的实际结果（2026-10-19）

v209/v210在v205两步展开的基础上分别禁用postRA LICM、或machine与postRA LICM，试图缩短scalar live range。只改变候选的compiler选项，生产source、selector和原benchmark未改。所有候选LDS均295936 B。

| 候选 | 方法 | VGPR / SGPR | private B / VGPR spill / SGPR spill | CPU决定 |
|---|---|---:|---|---|
| v205 | 完整q两步展开 | 452 / 107 | 0 / 0 / 1 | 拒绝，无GPU |
| v206 | 完整q四步展开 | 512 / 107 | 44 / 10 / 23 | 拒绝，无GPU |
| v207 | scalarize tensor-wait条件 | 462 / 106 | 0 / 0 / 0 | 通过，static4742 |
| v208 | 再scalarize task/refill/expert guards | 462 / 106 | 0 / 0 / 0 | 通过，static4774 |
| v209 | v205禁postRA LICM | 452 / 107 | 0 / 0 / 1 | 拒绝，无GPU |
| v210 | v205禁machine及postRA LICM | 451 / 102 | 0 / 0 / 0 | 通过，static6633 |

v207/v208/v210各一seed的E64→E65→E64完整MoE八项hash均与指定baseline一致，全部finite、原logits gate0.01保持；dynamic initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph各五次replay通过。每版始终一个artifact，actual完整ELF分别为`fe2cce9071aee5d2eea6fc0da391183964f5ec4fd8e732e046a222fc9ff23187`、`93f0606d279abf942f9d83712bb54d0fe5e2b7f55233c3feec1458fab0053b8a`、`664f94d950ccf429d7482ed1385181d410499133f8dc0999d7bded905d1d3287`。v210的20个signal/WAIT CFG区间通过，split最多8条WMMA；CFG不代替数值或完整hazard验收。

原完整benchmark筛选如下，原`--iters20/count19`、sanity正确、exit0，actual ELF绑定各自双规模验收，不改正式三轮中值。

| 本批版本 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190开头control1 | **91.106** | **137.883** | 202.73 | 282.82 |
| v207 | 92.923 | 139.536 | 205.10 | 284.30 |
| v208 | 93.322 | 142.349 | 203.97 | 288.57 |
| v210 | 92.820 | 137.443 | 206.93 | 284.93 |
| v190结尾control2 | **92.531** | **137.299** | 204.60 | 281.28 |

v207/v208两种规模均无优势；v210的E64无稳定收益，E65微小优势对尾部control反向。三版停止，不扩三轮、不替换生产v190。性能前全机idle；v210 E65前等另一任务PID558454，尾部E64 control的立即post发现独立diagnostic PID581225。原post首samples保留：其余立即post无KFD，busy2–3后六秒恢复idle；不能统一写成首post就全部idle。此批仅作筛选，不声称严格无所有权疑点的新稳定成绩。

171项/71116596 B证据已下载并核SHA256，未下载`.out`。manifest为`logs/runtime205_v210_control_evidence_20261019.sha256`；精确archive含172个regular members，6990301 B，SHA256=`26e349200718382c7dc3872680350a66b71653776570ce1320e466a2ea1d5531`。读取、actual ELF、e2e/replay及全部原profile复核见[控制开销实验汇总](../.codex_tmp/gemm1_t1536_20261017/logs/runtime205_v210_control_summary_20261019.json)。后续helper新增候选allowlist不改冻结archive。

## v211–v214：task入口准备TDM descriptor与固定stride（2026-10-19）

v211/v212保留四级ring、28次input请求、cross-task prefetch、OOB和完整F/S carry，只将owner不变的`GROUP1`、current/next global base、LDS base与rotation提前到task入口准备。原`make_tdm_atom(None)`的INT32_MAX inner extent、48-bit stride、multicast/early-timeout字段保留，完整global地址采用Int64。v211保留四owner外层dispatch，v212合并prepared字段，共享compute body，仅B immediate TH6的分派留在K-loop外。

每版999936项整数descriptor/address case覆盖E64/E65、31种tail、snake0/1/2、四task、四job、两owner、28个q、cross-task和跨4 GiB地址；负对照检测错误rotation、丢失carry、错误nextN与未clamp OOB。WMMA、reads、quant和同步的受保护AST不变；模型不代替实际ISA/GPU验收。

| 版本 | VGPR / SGPR | LDS B | private / spills | 静态指令 | actual完整ELF SHA256 |
|---|---:|---:|---|---:|---|
| v211 | 462 / 105 | 295936 | 全0 | 4654 | `bca6dcd55f960257cac6b2f0d378e307552353f455ae5acfcaa3533d16919740` |
| v212 | 461 / 98 | 295936 | 全0 | 3460 | `50fc8928af69c00a6e2140700609d0c207b9fa42d6f3979995c2527bd9aca063` |

v211 seed17、v212 seed43双规模完整e2e八项hash与baseline一致；E65保留1536条shared route，原logits gate0.01与finite检查通过。动态三种输入、poison、eager/两graph各五次replay通过，E64/E65一个artifact且actual ELF与CPU相同。12/6个signal/WAIT CFG区间通过，split每路径最多8条WMMA。

| 本批版本 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190开头control1 | **91.534** | **136.838** | 204.05 | 280.91 |
| v211 | 91.498 | 137.961 | 202.00 | 282.22 |
| v212 | 93.088 | 140.011 | 204.39 | 285.72 |
| v190结尾control2 | **91.471** | **137.264** | 204.05 | 281.47 |

原完整`--iters20/count19`、sanity通过、exit0。全八次pre首sample全机idle，立即post无KFD但busy2–3，六秒后及post6恢复idle；性能期间CPU会话闲置，没有本任务编译、验证或idle polling。v211 E64与首尾control持平、E65更慢；v212两种规模更慢。两版停止，不扩三轮、不合入。静态代码缩短不等于动态指令减少或耗时收益。

92项/13323968 B证据已下载、逐项核SHA256，archive含93个regular members、1583262 B、SHA256=`7b2e7cf286f824e796e99f8f9d7181e4eafa3348db540ee7f15228f9c73be045`，无`.out`。manifest为`logs/runtime211_v212_prepared_evidence_20261019.sha256`，汇总见[prepared-input实验JSON](../.codex_tmp/gemm1_t1536_20261017/logs/runtime211_v212_prepared_summary_20261019.json)。

v212 ISA在refill处实际使用runtime `s_lshl_b32`计算K byte offset及LDS pitch，并在Int32→Int64时生成sign extension；这不是回退唯一原因的证明。v213将A/B data与共享scales分为K-loop外的三个固定stride分派；v214另用Int64形成正向K offset。两版在2688项新增地址/stride case及继承descriptor模型中相等，原64-bit carry保留。CPU均465 VGPR/104 SGPR、LDS295936 B、private/spills0、static4172，九个CFG区间通过。v213 seed17、v214 seed43的双规模完整e2e八项hash与指定baseline一致，dynamic/poison/eager/两graph各五次replay全部通过，同一actual artifact和完整ELF分别为`69851d5237d61de5b00254c5ef8489e1fce1a008b07b88de5e810daf085c62e2`、`e2b2de3a5fa405a5321d28bd0172c2b87f5598254096b707d73110d2e867ecf5`。

| 固定stride筛选批次 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190首control1 | 92.308 | 138.255 | 205.04 | 283.19 |
| v213 | 94.243 | 138.431 | 204.27 | 283.51 |
| v214 | 92.892 | 139.075 | 204.98 | 282.77 |
| v190尾control2 | 91.437 | 137.565 | 203.68 | 283.32 |

原完整`--iters20/count19`、sanity及exit0保持，actual ELF与验收对象绑定。八次pre首sample全机idle，性能期间CPU会话闲置；立即post busy1–3，六秒及post6恢复idle。两项v190 control的post首sample包含`<已退出>`身份不完整条目，原记录保留，不将其解释成自有残留或严格排除其他任务边界。两候选均没有目标收益，停止，不扩三轮、不合入生产。93项/13690522 B证据已下载、逐项核SHA256；archive1636535 B、94个regular members、SHA256=`b68a7fd486d704852c7739786763752f070bd36144a261b592c582a70d00f0d3`，无`.out`。manifest为`logs/runtime213_v214_conststride_evidence_20261019.sha256`，汇总见[固定stride实验JSON](../.codex_tmp/gemm1_t1536_20261017/logs/runtime213_v214_conststride_summary_20261019.json)。

host复现用新label，先CPU与正确性，再通过原完整benchmark筛选：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_runtime_latency_cpu_host.sh" 213
bash "$task/run_runtime_expert_check_host.sh" 17 prepared_const_e2e_repro01 \
  --source "$task/candidates/optx_core_v213.py"
bash "$task/run_runtime_expert_replay_host.sh" prepared_const_replay_repro01 \
  --source "$task/candidates/optx_core_v213.py"
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 prepared_const_e64_repro01 \
  --source "$task/candidates/optx_core_v213.py" \
  --acceptance-label prepared_const_e2e_repro01
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 prepared_const_e65_repro01 \
  --source "$task/candidates/optx_core_v213.py" \
  --acceptance-label prepared_const_e2e_repro01
```

CPU runner已有固定label时拒绝覆盖；重新编译已有候选须新输出label。每项自有命令仍120s上限。共享A LDS槽、此前无收益的具体pipeline方向不重开。

## v215/v216：runtime-E 完全无效 capacity tile（2026-10-19）

这是从当前v190重新生成的runtime-E候选，不把旧v188/v189的CPU模型当作新GPU验收。v215把`persistent_expert < n_experts`放到整个四N-task loop外；v216只把accumulator zeroing放到原expert guard内。所有有效tile的TDM、K0→K55、reads、READY/RELEASE、SiLU/BF16/FP4及输出函数保持，四cluster N peers与WG内所有waves共用同一个live predicate。无有效tile在v190本来就没有input/output TDM和WG barrier，两版保留kernel最后的`tensor_wait(0)`。

144个fixture、323328个M/N-task判定case覆盖E1–E256、空expert、重复psum、partial M16/M256、capacity tail及E65/shared1536。模型重现actual runtime upper-bound while-loop，逐项等于所有真实有效行的范围；静态AST/整数检查不代替实际ISA/GPU。CPU v215为463 VGPR/106 SGPR、LDS295936 B、private/spills0、static4708、12个CFG区间通过，完整ELF=`77cb2b852f3c71ba497068f61251dcbb0474bfced9d8e0f1cafe02a3f518c637`。v216为463/107、同LDS、private/spills0、static4710，其`.text` SHA256=`7b0b0ca23994eba0fd308c388c5925b356051b553ca0e31e7e738a9c4a6805ad`与v190相同；完整ELF因kernel名字不同而不同。v216没有新机器码，不运行重复GPU性能。

v215 seed17 E64→E65→E64完整MoE八hash与baseline一致、logits约3.4766e-06/3.7788e-06、finite、无fallback，E65保留1536条shared route，始终一个actual artifact。dynamic三输入、poison、eager/两graph各五次replay通过，actual ELF与CPU一致。原完整性能筛选已结束，两规模无收益，停止v215、不扩三轮、不合入；v216保持CPU-only。

复现使用新label：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_runtime_expert_check_host.sh" 17 capacity_e2e_repro01 \
  --source "$task/candidates/optx_core_v215.py"
bash "$task/run_runtime_expert_replay_host.sh" capacity_replay_repro01 \
  --source "$task/candidates/optx_core_v215.py"
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 capacity_e64_repro01 \
  --source "$task/candidates/optx_core_v215.py" \
  --acceptance-label capacity_e2e_repro01
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 capacity_e65_repro01 \
  --source "$task/candidates/optx_core_v215.py" \
  --acceptance-label capacity_e2e_repro01
```

本批原完整benchmark，`--iters20`、profiler count19、原logits gate0.01不变。数值属于该筛选批次，不混入正式三轮中值。

| version | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190首control1 | 91.886 | 137.569 | 202.98 | 283.37 |
| v215 screen1 | 92.048 | 138.183 | 202.67 | 283.32 |
| v190尾control2 | 91.873 | 136.635 | 203.05 | 282.18 |

六次pre首sample均全机idle；立即post无KFD所有者、VRAM176582656 B，busy2–3，按6秒等待后及post6均idle。测量期间CPU会话闲置，无本任务并发编译/正确性/轮询。冻结包1642106 B、77 regular members，SHA256=`c3c4f53ab0da7a1ce745f90346266f3464149eba9bb965d30663a00426da70b1`；其中76项15274741 B逐项SHA已本地核验，排除`.out`，当时helper保存到`frozen_helpers/runtime215_v216/`。证据为`.codex_tmp/gemm1_t1536_20261017/logs/runtime215_v216_capacity_evidence_20261019.sha256`及`runtime215_v216_capacity_summary_20261019.json`。

## v217–v219：保留公式的 SiLU TRANS/VALU 分组（2026-10-19）

仅在隔离candidate中复制`gemm_common_gfx1250.py::batched_silu_swiglu`，将CHUNK4改成2/8/16；原公共helper不改。clamp、tanh、`0.5*(1+tanh)`、乘法依赖、BF16/E8M0/FP4、output stores及56次正向K128均保持。每版2145项index case覆盖N1–65、每项issue/consume一次，helper还原CHUNK4后与原helper相同；其余函数AST受保护。

参考MI400 Shader Programming Guide §5.3.5 pp220–221，main与TRANS使用独立math管线，dependent main/TRANS需要分别4/6个waitstates或独立指令；这提供了调整独立指令分组的依据，不保证实际收益。CPU三版均463 VGPR/107 SGPR/LDS295936 B、private/spills0、12 CFG区间通过，v217 static4742、v218/v219 static4731。v218与v219的actual`.text`相同，因此v219停止于CPU，不重复GPU验证或计时。

v217 seed17、v218 seed43均完成双规模八hash/e2e和dynamic三输入、poison、eager/两graph各五次replay，全部通过；E64→E65→E64同artifact/full ELF、全部1536条shared route保留。v217完整ELF为`ec66b572e6e0f0b56736b476dc4ef6471b18e43ff58ef72de35ca91d775fe667`，v218为`1ff9bdbfeecfe0efa17ab53e18bacd9ac0c0014865ca27d736cbb3782e30b70a`，actual GPU对象与对应CPU一致。

原完整`--scenario bench --iters20`、warmup5、profiler count19、原logits gate0.01的本批筛选如下，和正式三轮数据分开记录：

| version | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190首control1 | 90.994 | 137.599 | 201.59 | 283.57 |
| v217 CHUNK2 | 92.898 | 138.794 | 204.81 | 284.98 |
| v218 CHUNK8 | 92.521 | 137.409 | 205.07 | 282.12 |
| v190尾control2 | 91.523 | 136.918 | 201.13 | 280.27 |

CHUNK2/8均未改善E64 GEMM1，停止、不合入、不扩三轮；CHUNK16只完成CPU检查，因与CHUNK8生成相同`.text`而没有GPU/e2e/replay/性能记录。生产正式最优仍为v190，E64/E65正式三轮中值91.688/137.596 μs。独立main/TRANS管线提供候选依据，但这组具体分组没有取得目标收益，不能由静态指令数推导动态成本。

候选pre首sample均全机idle，立即post无KFD所有者、busy2–3，6秒及post6恢复idle。v190 E64尾control的pre先发现他人MLA PID1437594，按6秒等待至idle后才启动；其post先有身份不完整的`<已退出>`条目，post6又出现独立工作树的CPU compile PID1444787，之后idle。E65尾control的post6也曾有`<已退出>`条目，之后idle。原始所有权记录保留，不将上述尾control写为首次检查均idle或严格无疑点；候选两次E64均慢于两个对照，当前停止判断不依赖挑选较快样本。计时期间本任务CPU会话闲置。

冻结包2526317 B、116 regular members、SHA256=`e6e342ae16bf09dfcc24ab728bda454726a524f4d472f3558bcc0a29d7f515ce`，其中115项23006259 B逐项SHA256已在本地核验，不包含`.out`。冻结当时helper在`frozen_helpers/runtime217_v219/`，manifest与汇总为`logs/runtime217_v219_actchunk_evidence_20261019.sha256`、`logs/runtime217_v219_actchunk_summary_20261019.json`。

复现时使用新label，E65由同一候选自动选择真实shared1/topk9：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_runtime_expert_check_host.sh" 17 actchunk2_e2e_repro01 \
  --source "$task/candidates/optx_core_v217.py"
bash "$task/run_runtime_expert_replay_host.sh" actchunk2_replay_repro01 \
  --source "$task/candidates/optx_core_v217.py"
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 actchunk2_e64_repro01 \
  --source "$task/candidates/optx_core_v217.py" --acceptance-label actchunk2_e2e_repro01
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 actchunk2_e65_repro01 \
  --source "$task/candidates/optx_core_v217.py" --acceptance-label actchunk2_e2e_repro01
```

CHUNK8使用v218及seed43，其他参数与门限保持；已存在的候选或输出label由脚本拒绝覆盖，复现无需重新生成已有候选。

## 临时任务：kernel / bench 计时差异与机器暂停（2026-10-19）

SiLU CHUNK实验完成后按用户要求转入scenario诊断。两种模式实际v190完整ELF、`.text`、JIT key及launch参数相同；正常原命令kernel85.9759/87.0495 μs、bench92.2233 μs，额外诊断批次86.760579/91.074316 μs。异常bench646.850 μs与外部六个stress进程重叠，排除性能。production及原averaging未修改，诊断不混入正式中值。

源码/时间轴确认隔离GEMM1重复固定partial，bench每轮完整MoE；两者GEMM1行仅统计其device duration。route、psum、weights已一致，A/ScaleA整块hash含padding，有效行待核对。固定地址G2/quant前序对照未取回结果前a07-3 hang，用户要求等重启通知，全部远程操作暂停；本地继续证据与候选模型，不假定对照通过或确定hang原因。详情见[kernel/bench计时差异分析](moe_gemm1_kernel_vs_bench_timing_analysis.md)。正式最优仍91.688/137.596 μs。

## v220：完整 M256 / N128 四-wave 候选，本地准备完成（2026-10-19）

本节最初记录机器暂停期间的本地准备。用户后续明确通知a07-3重启，本任务恢复远端操作，scenario差异诊断已完成并归档，v220已上传核SHA并实际CPU编译。2026-10-20已完成scalar选项筛选、ISA审计helper修正、seed0/17/43双规模完整MoE和dynamic/poison/eager/两graph replay，全部通过且E64/E65同artifact/完整ELF。原完整benchmark筛选E64/E65为 **133.557 / 176.670 μs**，明显慢于同批v190，v220四级ring停止、不合入、不扩三轮。随后v225两级ring也完成完整验证与筛选，E64/E65为 **127.767 / 208.077 μs**，同样停止；当前转回v190 N256的隔离TDM策略实验。生产最优仍为v190，正式E64/E65 GEMM1三轮中值 **91.688 / 137.596 μs**，E64距80 μs差 **11.688 μs**。

### 候选改变与保留项

独立候选为 `.codex_tmp/gemm1_t1536_20261017/candidates/optx_core_v220.py`，SHA256=`414a664749b164eef6eeb2c7923f3819e8975e9df453076c7733913c80ef25cd`。由冻结v190生成，16项源码替换可逆还原。公开调用仍接受原 `t256/w2x4/b4/q1r8`，内部改为 `t256x128x256/w2x2/b4`，四个persistent N-tasks，N4096使用两个N-group。runtime `n_task_base`覆盖完整N，expert lookup和M256 owner保留，E64/E65仍使用runtime expert参数。Constexpr参数在launcher内部重新绑定是否被FlyDSL接受，尚待实际编译。

保留上下两份独立physical A128/ScaleA128、两个M128 compute windows、原SiLU/BF16/MXFP4 quant及输出布局；每个有效accumulator仍按K0→K55完成56次K128。没有复用同一份A/ScaleA槽，不是已经放弃的共享槽路线。每wave仍有16×16个逻辑F32 accumulator words，不能预先声称VGPR减半。

LDS模型为A ring128 KiB、B ring64 KiB、ScaleA ring8 KiB、ScaleB ring4 KiB、output windows9 KiB，合计 **218112 B（213 KiB）**，比v190的295936 B（289 KiB）减少 **76 KiB**。实际VGPR/SGPR/LDS metadata和spill尚未取得。Shader Programming Guide §4.7.1 pp160–161、CDNA5白皮书p10记载每WGP最多320 KiB LDS，LDS/WGP$按64 KiB segments划分；213 KiB大于320 KiB的一半，不能据此宣称两个WG常驻，也没有WGP$收益的counter。

四waves下 `(0,1)`同时承担A/ScaleA，`(2,3)`同时承担B/ScaleB，每wave每K256发出两个input TDM。原按owner `waves`识别下一task B/ScaleB的写法，在新几何会把ScaleB归成B；候选改用不可变tensor/layout字段区分。该问题来自新几何适配，没有据此修改生产v190。

按两descriptor新几何重推 `tensor_wait`：initial6、steady4、successor前三步6、最终tail2/0/0。旧3/2/4在新几何属于更保守等待，不能说旧常量必然造成race。依据CDNA5 ISA §10.11.1 p140，同wave tensor load/store按序完成，跨wave及其他memory不自动排序；模型保留all-wave READY/RELEASE及output LDS store的DScnt wait。SPG §4.10.8 p206的每wave3、每SIMD6 descriptor限制是issue→XACK口径，和issue→completion的TENSORcnt上限63区分；`tensor_wait(6)`不等于允许6个未XACK descriptor。

### 实际 route counts 的工作量与请求成本

使用冻结的 `logs/optx_v190_edyn_TH6_seed0_20261019.json`，从各自record读取真实E64/E65 counts。E65 routed counts与E64不同，没有直接拼用E64 counts加shared route。E64有12288条有效route、64个live M256、797个live M16；E65有13824条有效route、70个live M256、893个live M16，其中1536条shared route保留。

| logical量 | E64 v190 | E64 v220 | E65 v190 | E65 v220 |
|---|---:|---:|---:|---:|
| live WG | 256 | 512 | 280 | 560 |
| 含empty capacity的launched WG | 448 | 896 | 476 | 952 |
| live wave instances | 2048 | 2048 | 2240 | 2240 |
| live N-tasks | 1024 | 2048 | 1120 | 2240 |
| input TDM指令，所有owner waves合计 | 229376 | 458752 | 250880 | 501760 |
| output TDM指令，所有waves合计 | 16384 | 16384 | 17920 | 17920 |
| 有效M16 WMMA指令 | 5712896 | 5712896 | 6401024 | 6401024 |
| A logical source请求，合并前MiB | 697.375 | 1394.750 | 781.375 | 1562.750 |
| ScaleA logical source请求，MiB | 45.390625 | 90.781250 | 50.421875 | 100.843750 |
| B logical source请求，MiB | 896.000 | 896.000 | 980.000 | 980.000 |
| ScaleB logical source请求，MiB | 56.000 | 56.000 | 61.250 | 61.250 |

这些是重复N-task/M-tile的descriptor source surface量，A/ScaleA分别按实际M16/M32边界裁剪，**不是测得的HBM流量**。cluster_m=1源码对A带四N-peer multicast mask，ScaleA没有mask；即使假设每组四peer A请求完全合并，E64 A logical source仍由174.34375增为348.6875 MiB，ScaleA仍翻倍。真实合并、cache命中和HBM事务均未测。B/ScaleB总surface不变，但单descriptor更小且指令数增加；有效计算、总wave数和输出量不变，WG及input descriptor数量翻倍。潜在收益是每WG LDS与N跨度下降，代价是重复搬运、descriptor及调度开销，不能据此预测耗时。

账目脚本从实际候选AST推导几何、TDM mask和LDS constants，结果为 `logs/optx_v220_runtime_n128_fourwave_accounting_20261019.json`；本地0.610秒、exit0、120秒上限，无GPU/网络。

### 本地检查和待执行门禁

`logs/optx_v220_runtime_n128_fourwave_model_20261019_r3.json`记录1983 routing fixtures、196616 actual-AST owner checks、786464 N-tile checks、3027562 valid routed rows、573440 symbolic DS loads、32256跨N-task地址，以及output payload/ScaleA地址检查通过；四wave两descriptor和v190八wave一descriptorcontrol均保留K0..K55。七项错误注入全部拒绝，包括提前READY、遗漏RELEASE、提前refill、提前覆盖B bank、遗漏N-group和错误ScaleA owner。模型14.304秒、exit0，**未执行FlyDSL lowering，不能替代ISA或GPU验收**。

补充descriptor有效范围与对齐检查已完成：`logs/optx_v220_alignment_extent_local_20261019.json`记录115136个actual-AST input descriptors、14392项extent cases和512个output descriptors；本地12.850秒、exit0、120秒上限。覆盖M-tail 0..256、全部28个K256 tile、两个first-half、两个N-group及全部task/peer/window，确认input LDS DWORD对齐、各tensor目的区间不重叠、A按M16和ScaleA按M32裁剪、relative global source/stride 128B对齐、first tile dimension至少256B，以及private output slices的范围与对齐。故意注入misaligned LDS、重叠A1和错误A extent的三个负对照全部拒绝。每个input descriptor的LDS区间均落在单个64 KiB segment内；这只是源码地址统计，不是bank-conflict或硬件事务测量。

上述对齐条件参照Shader Programming Guide §4.10.2 pp199–200，tensor OOB和地址单位参照CDNA5 ISA §10.11.2 pp141–142。absolute allocation pointer、实际发射descriptor和direct-copy路径尚未验证；条件吻合不能代替CPU/ISA、时序协议或数值/replay门禁，也不能据此宣称cache、HBM或耗时改善。

已准备并本地语法检查CPU runner与完整验收入口。CPU要求E64→E65→E64一个artifact/完整ELF，VGPR≤512、LDS218112 B、private和两类spill0，审核split barrier CFG。`runtime_n128_fourwave_gate.py`绑定source、model、compile helper、完整ELF、resource、ISA和CFG；GPU完整MoE验收先核对它，再保留指定commit的有效输出八hash及原FP32 reference门限。replay要求完整双规模MoE receipt，性能要求MoE、replay、source和实际ELF一致。

进一步本地scope复核：26个compute/quant/output等受保护函数AST与v190一致，候选LLVM compile policy也一致；完整MoE/replay helper的fixture调用、canonical snapshot、poison及输入变体functions与冻结v217–v219 helpers相同。四项host runner的容器、120秒timeout与性能区间无并发idle-check通过，六份生产/selector/原benchmark/计时工具source SHA仍匹配。结果为 `logs/optx_v220_source_scope_local_20261019.json`。本地FlyDSL HEAD=`5675194f18f0655ce1f979c517f673533963fa93`，`jit_function.py`把Constexpr作为Python值传入body，`ast_rewriter.py:visit_Assign`没有禁止直线赋值；该源码支持内部重新绑定，但与远端版本是否一致未核对，也不替代v220完整编译和ISA检查。

`logs/optx_v220_validation_helpers_local_20261019_r3.json`是验收**工具**自检：实际原benchmark AST识别E64/E65 N128 `_edyn`名，两阶段各19个profile calls保持，继续拒绝错误K/activation及重复匹配；11种损坏receipt全部拒绝。自检使用temporary synthetic metadata及已有v190 ELF，临时文件已移除，**不属于v220 CPU或GPU验收**。第一次自检因AST stub漏了 `_STAGE1_ACT_ID`失败，补齐原依赖后r2通过，r3补核calls；失败日志保留，生产benchmark未修改。

恢复后已读取hang前固定context日志并完成canonical有效A/ScaleA核对。v220的CPU/model/ISA门禁已通过，下一阶段为seed0/17/43双规模完整MoE与dynamic/poison/eager/graph replay。全部通过后独占空闲性能窗口，先v190首尾control夹单轮v220；无目标收益就停止，不扩三轮，不将筛选最低值纳入正式成绩。未合入生产。

### 复现命令

以下远端命令是验收入口，每个compile/test/analysis内部120秒timeout。已有候选及输出label拒绝覆盖，复现使用fresh label，已有候选无需再次运行builder。CPU资源门禁允许原v220已审核的6个SGPR-to-VGPR-lane spills，仍要求private=0、VGPR spill=0和没有scratch指令；全部CPU/model/ISA门禁通过后才能执行GPU入口。

```powershell
# Local integer/symbolic checks only; use fresh labels.
py -3.14 -X utf8 .codex_tmp/gemm1_t1536_20261017/audit_runtime_n128_fourwave.py --label n128_model_repro01
py -3.14 -X utf8 .codex_tmp/gemm1_t1536_20261017/account_runtime_n128_fourwave.py --label n128_account_repro01
py -3.14 -X utf8 .codex_tmp/gemm1_t1536_20261017/audit_runtime_n128_fourwave_alignment.py --label n128_alignment_repro01
py -3.14 -X utf8 .codex_tmp/gemm1_t1536_20261017/check_runtime_n128_fourwave_helpers_local.py --label n128_helper_repro01
```

```bash
# Remote host, after explicit resume and task-file SHA verification.
task=.codex_tmp/gemm1_t1536_20261017
model=optx_v220_runtime_n128_fourwave_model_20261019_r3
cpu=n128_cpu_repro01
accept=n128_e2e_seed0_repro01
replay=n128_replay_repro01
bash "$task/run_runtime_n128_fourwave_cpu_host.sh" "$cpu" "$model"
bash "$task/run_runtime_expert_check_host.sh" 0 "$accept" \
  --source "$task/candidates/optx_core_v220.py" --model-label "$model" --cpu-label "$cpu"
bash "$task/run_runtime_expert_check_host.sh" 17 n128_e2e_seed17_repro01 --reverse \
  --source "$task/candidates/optx_core_v220.py" --model-label "$model" --cpu-label "$cpu"
bash "$task/run_runtime_expert_check_host.sh" 43 n128_e2e_seed43_repro01 \
  --source "$task/candidates/optx_core_v220.py" --model-label "$model" --cpu-label "$cpu"
bash "$task/run_runtime_expert_replay_host.sh" "$replay" \
  --source "$task/candidates/optx_core_v220.py" --acceptance-label "$accept"

# All gates first. Each call independently waits for all-GPU/KFD idle.
# Original full --scenario bench, warmup5/iters20 and profiler count19.
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 n128_v190_control_before_repro01
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 n128_e64_screen_repro01 \
  --source "$task/candidates/optx_core_v220.py" --acceptance-label "$accept" --replay-label "$replay"
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 n128_v190_control_after_repro01
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 n128_e65_screen_repro01 \
  --source "$task/candidates/optx_core_v220.py" --acceptance-label "$accept" --replay-label "$replay"
```

本轮源码/model/helper/账目/对齐与文档的本地SHA manifest为 `logs/optx_v220_local_preparation_20261019.sha256`，文件角色、冻结副本与尚未完成阶段记录在 `logs/optx_v220_local_preparation_inventory_20261019.json`，helper冻结副本位于 `frozen_helpers/runtime220_n128_fourwave/`。路径均相对repo，manifest不含自身；helper副本用于保存测试时版本，执行入口仍为task根目录，不能直接从冻结目录运行依赖相对路径的脚本。文档尚未同步远端，独立`aiter_dsr1_e65`章节保留，正式性能继续使用前述v190三轮结果。

### 重启后首次实际 CPU 编译

`optx_v220_n128_cpu_resume_20261019_r1`编译与ISA inspection exit0，E64→E65→E64为同一个artifact及完整ELF。actual完整ELF SHA256=`567fc0dd98e0b2b5719ddc467eaa8af84577068997b1d5da2eed77ce68a71432`，`.text`=`8ffb1d5939800aeb099ba59dde6bbc7c962be869bfad6985769567c156725e7d`；452 VGPR、107 SGPR、LDS218112 B，private0、VGPR spill0、SGPR spill6。资源检查exit1后停止，未进行GPU数值、replay或性能。ISA可见将scalar地址/extent存入VGPR lanes再读取的指令，不能把private0解释成完全没有spill。

CPU receipt、完整log、ISA/inspection保存于`logs/optx_v220_n128_cpu_resume_20261019_r1*`。下一步只在独立候选调整scalar生存期/LLVM hoisting，不改变有效计算、量化/输出或生产代码；零spill不是硬件正确性的必然要求，这里是本轮性能候选的资源筛选条件。

### Scalar 候选筛选与 ISA 审计修正（2026-10-20）

为缩短scalar生存期，实际CPU检查了四个独立候选，均未进行GPU测试：

| 候选 | 唯一变化 | VGPR / SGPR | SGPR spills | CPU结论 |
|---|---|---:|---:|---|
| v221 | disable-machine-licm | 450 / 102 | 8 | spill增加；未GPU |
| v222 | disable-postra-machine-licm | 452 / 107 | 6 | `.text`与v220相同；未GPU |
| v223 | 同时禁用两项LICM | 450 / 102 | 8 | `.text`与v221相同；未GPU |
| v224 | 两owner组互斥if/else | 456 / 107 | 6 | 未改善scalar spills；未GPU |

这些候选没有改变生产kernel。6个SGPR spills是`v_writelane_b32` / `v_readlane_b32`保存和取回scalar值，actual ELF private=0，ISA没有scratch指令；它们仍有指令成本，不能称为零spill。ISA §15.14（printed pp638–639）的lane指令语义支持这一解释。零spill是此前自设的效率筛选条件，并非数值正确性要求；现在仅允许原v220这6个已审核lane spills进入真实数值和性能验证，不改变精度门限、有效route或计时。

`optx_v220_n128_cpu_lane_review_20261020_r1`资源检查通过，但旧`audit_earlybanks_split_isa.py`在PC `0x26b0`误判：input prime的`SIGNAL`与`WAIT`之间有` s_mov_b32`、`s_set_vgpr_msb`两条寄存器设置，旧checker按区间长度误认为是reader RELEASE。修正仅发生在审计helper：无DScnt wait的例外须立即跟在input `tensor_load_to_lds`及` s_wait_tensorcnt`之后，所有路径WMMA=0，区间只允许指定scalar register setup；真正reader RELEASE仍要求SIGNAL前完整` s_wait_dscnt 0x0`。这是checker分类修复，没有修改kernel的同步。

本地actual ISA自检确认2个prime和4个RELEASE区间；删除RELEASE的DScnt wait、插入scalar memory、把prime变为WMMA reader、删除TENSORcnt wait、跳过WAIT五项错误注入均被拒绝。另用明确标记的synthetic receipts做14项gate失配检查，包含scalar spill数和scratch指令拒绝；工具自检不是GPU验收。

fresh CPU轮次`optx_v220_n128_cpu_cfg_primefix_20261020_r1`的compile、inspection、resource、CFG及final gate全部exit0。E64→E65→E64始终一个artifact，完整ELF/`.text`与初次CPU相同。6个静态SIGNAL/WAIT区间为2个WMMA=0 prime、4个WMMA=0/2/4/6/8 RELEASE，各路径收敛同一WG WAIT且区间没有memory。新CFG receipt同时绑定ISA SHA256及审计helper SHA256，防止旧证明被误用。此处仍不声称完整memory-hazard证明或GPU正确性。

证据：`logs/split_audit_prime_release_selfcheck_20261020_r1.json`、`runtime_n128_gate_lane_review_selfcheck_20261020_r1.json`、`optx_v220_n128_cpu_cfg_primefix_20261020_r1{,_gate}.json`、同轮log及`_step0_e64.lowered.isa.split_cfg_audit.json`。冻结`frozen_helpers/runtime220_n128_fourwave/`保留此前版本，未重写。

### v220 完整精度、replay 与性能筛选结果

seed0/43按E64→E65→E64、seed17按E65→E64→E65执行完整MoE。每次8项tensor hash与指定baseline一致，E64全部12288条route、E65全部13824条route和1536条shared expert64/weight1 route保留；原logits门限0.01不改，E64约3.473–3.477e-6，E65约3.777–3.779e-6，全部finite。三个进程各自仅一个GEMM1 artifact，实际完整ELF均为`567fc0dd98e0b2b5719ddc467eaa8af84577068997b1d5da2eed77ce68a71432`，与CPU验收相同。

dynamic `initial/sign_flip/alternate_values`、poison、eager/parent graph/runtime graph各五次replay通过，所有有效payload和ScaleA mismatch=0。按用户原完整`--scenario bench --iters20`、warmup5、count19、原FP32 reference做同批首尾control夹候选：

| 本批版本 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190首control | 91.717 | 138.200 | 203.61 | 282.53 |
| v220 N128/b4 | **133.557** | **176.670** | 240.33 | 322.47 |
| v190尾control | 93.094 | 138.248 | 206.07 | 282.63 |

六次pre首sample均全机idle，无KFD所有者。立即post无所有者、VRAM176582656 B、busy2–3，按6秒复查及post6恢复idle；原samples保留，性能期间本任务没有并发编译、正确性或idle polling。未出现自有测试残留。v220在E64/E65都明显回退，停止四级ring，不扩三轮、不合入生产。上述是该筛选批次，正式91.688/137.596 μs不变。

源码账目说明v220 live WG与input TDM请求数翻倍，A/ScaleA重复source请求增加；实际LDS降至213 KiB后仍不能在320 KiB LDS内同时容纳两个WG。它解释了为何单看LDS下降不足以保证优势，但没有v220 ATT或memory counters，不能断言哪项是回退的唯一原因，更不能把源码请求量当实际HBM读取量。

证据：`logs/optx_v220_n128_moe_seed{0,43}_20261020_r1.json`、`optx_v220_n128_moe_seed17_reverse_20261020_r1.json`、`optx_v220_n128_replay_20261020_r1.json`及同轮logs；性能为`optx_v220_n128_{control_before,screen,control_after}_e{64,65}_20261020_r1.log`和actual identity。完成时helper另存`frozen_helpers/runtime220_completed_20261020/`，后续candidate allowlist修改不重写这些副本。

### v225：N128 两级输入 ring 的隔离候选

保留v220四-wave、M256两个M128 window、四个N-task和N4096两组、runtime expert ABI、两份physical A128/ScaleA128以及全部有效正向K0..K55、quant/output。仅把四级input ring改成两级，stage索引使用`num_buffers-1`；没有把上下半A复用为一份。逻辑LDS为2×(A256 32 KiB+B128 16 KiB+ScaleA256 2 KiB+ScaleB128 1 KiB)+9 KiB output=**113664 B（111 KiB）**。

两级协议重新推导：首task prime `tensor_wait(2)`；普通step为0，确保q+1全部input可供下一F carry读取；successor q0/q1在旧task两条output TDM之前issue，只有successor step0保留`tensor_wait(2)`排除这两条旧stores，step1之后wait0。全部input refill仍在reader DScnt=0和WG WAIT之后。两级ring减少prefetch提前量，是否被更多驻留wave补偿需要实测。

actual-AST模型重新运行1983 routing fixtures、196616 owner checks、786464 N-tile checks、3027562 valid rows、28672 reader cases及573440 symbolic DS loads；七个协议/地址负对照通过。112个input Ktiles、每wave232条含output请求、128个有效accumulator streams仍按K0..K55。descriptor对齐/extent另验115136个input、14392种extent、512个output和三个负对照。模型不替代FlyDSL ISA或GPU。

硬件依据为Shader Programming Guide §4.7.1（p160）的LDS最多320 KiB、LDS/WGP$共384 KiB，以及§3.3.2.1（p32）的MI450每SIMD1024 VGPR。实际资源451 VGPR/wave，两个四-wave WG在LDS/VGPR上可容纳；这是资源上界，不证明scheduler实际驻留两WG。第一轮CPU因自设零scalar spill筛选条件失败，记录保留；按v220相同标准审核六个lane spills后，fresh `optx_v225_n128_b2_cpu_lane_review_20261020_r1` 的compile、inspection、resource、CFG及final gate全部exit0，private0/VGPR spill0，无scratch。

v225 source SHA256=`5ce0532694fea557cd4ee62166c6288ee3fa5def9d3fff7173905de7a45e80ee`。源码变化由九项可逆替换绑定，26个compute/quant/output/descriptor相关函数AST不变。生产source、selector、原benchmark与计时工具仍冻结。

实际完整ELF=`4a902ab5e2b2b670f6b3bf1f509487cbb64499ca2a30faa276749c5402967994`，`.text`=`c1494b2ef7eb90be432b8424b216781464511a2870d2593a4f1716148958a21e`。seed0、seed17反向和seed43的完整MoE全部通过，两种规模八项hash与指定baseline一致，E64 logits diff为3.47334604e-6–3.47728910e-6，E65为3.77714610e-6–3.77927844e-6，原0.01门限保持。每进程正反向切换E64/E65均一个真实artifact，GPU实际完整ELF与CPU一致，E65 shared route1536条全部保留。dynamic initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph每种各五次replay全部payload/ScaleA mismatch0，同artifact/完整ELF。

原完整`bench --iters20`、warmup5、GEMM1 count19的同批筛选为：

| 同批版本 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190首control | 90.838 | 138.121 | 200.94 | 283.28 |
| v225 N128/b2 | **127.767** | **208.077** | 239.39 | 353.27 |
| v190尾control | 92.597 | 138.117 | 204.26 | 284.20 |

六次run退出均0，actual完整ELF与各自验收一致，sanity原0.01门限通过。每次pre全机空闲检查后才启动，性能期间无本任务并发编译、正确性或idle polling。各次立即post及post6的原始记录单独保留，最后已恢复idle。E64和E65均显著回退，停止该配置及本轮N128 ring方向，不扩三轮、不合入。b2与b4不是同批对照，不能用127.767与133.557的差直接声明两级ring取得稳定收益；也没有候选ATT或counters，不能定量分摊prefetch lead、WG数、lane spills或实际occupancy的影响。生产仍为v190。

证据label为`optx_v225_n128_b2_moe_seed{0,43}_20261020_r1`、`optx_v225_n128_b2_moe_seed17_reverse_20261020_r1`、`optx_v225_n128_b2_replay_20261020_r1`及`optx_v225_n128_b2_{control_before,screen,control_after}_e{64,65}_20261020_r1`。复现使用下列host入口和fresh label；CPU/model文件已存在时不覆盖：

```bash
task=.codex_tmp/gemm1_t1536_20261017
model=optx_v225_n128_two_stage_model_20261020_r1
cpu=optx_v225_n128_b2_cpu_lane_review_20261020_r1
accept=n128_b2_moe_repro01
replay=n128_b2_replay_repro01
bash "$task/run_runtime_expert_check_host.sh" 0 "$accept" \
  --source "$task/candidates/optx_core_v225.py" --model-label "$model" --cpu-label "$cpu"
bash "$task/run_runtime_expert_replay_host.sh" "$replay" \
  --source "$task/candidates/optx_core_v225.py" --acceptance-label "$accept"
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 n128_b2_perf_repro01 \
  --source "$task/candidates/optx_core_v225.py" --acceptance-label "$accept" --replay-label "$replay"
# E65 records shared_experts=1/topk9; all other workload parameters stay fixed.
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 n128_b2_e65_perf_repro01 \
  --source "$task/candidates/optx_core_v225.py" --acceptance-label "$accept" --replay-label "$replay"
```

## v226/v227：N256 A/ScaleA 请求策略隔离实验（2026-10-20）

在v220/v225明确回退后回到已验收v190，不继续N128 ring。v226仅把A的`wg_mask=a_mcast_mask`改为`None`，保留原TH0/RT；v227仅让ScaleA使用相同四N peers的multicast mask及early timeout。B保持TH6，两份physical A/ScaleA、四级ring、全部request数、READY/RELEASE、四N-task、有效正向K0..K55、quant输出和runtime expert数不变。两候选归一化唯一policy和symbol后完整module AST等于v190，分别57568项peer descriptor身份检查通过；这些检查不替代GPU精度。

硬件依据为本地`MI400_Shader_Programming#65.txt`：§4.9.8 p191说明multicast绕过WGP$，early timeout允许先到requesters返回，GL1最多合并五请求；§4.10.3 p201说明非零`workgroup_mask`令TDM使用`CLUSTER_LOAD_ASYNC`，Table80 pp202–203定义mask和bit21 timeout；Table19 p62定义TH0/RT。取消multicast可能允许本WG复用A缓存，也会取消跨N peers合并，净效果必须实测，不能预先宣称省流量或加速。旧v184仅改timeout、v187仅改A HT，不等同这两项策略。

| 候选 | source SHA256 | actual完整ELF SHA256 | VGPR / SGPR / LDS B | private / spills | static instructions |
|---|---|---|---|---|---:|
| v226 A unicast RT | `a52edade943a626725bc518108171d6b5059e0c72d257cd0e05eac2a1adfe75d` | `ed3b45878437d96a8771b02681f052060f8ce12dec10864fa5e888e06ef443d4` | 463 / 106 / 295936 | 全0 | 4711 |
| v227 ScaleA multicast early | `2e07b0b8d7e4954d10ed40ed73eeca6d659d2dfd5d99604492f7266fd1dd018a` | `a070b9f61e625c0b1fe1829fbd241d76f01cd23ac875a9272d79bac6c474e022` | 463 / 107 / 295936 | 全0 | 4709 |

两候选CPU dummy的E64→E65→E64始终cache1/同artifact和完整ELF，ISA/CFG门禁通过；原DS/TDM/WMMA counts、wait histogram和hardware barrier均保持v190，`.text`确有变化。CPU labels为`optx_v{226,227}_tdm_policy_cpu_20261020_r1`。两版seed0/17反向/43完整双规模八hash均与指定baseline一致，E65保留1536条shared route，logits仍约3.473–3.477e-6/3.777–3.779e-6；dynamic initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph各五次replay均通过。actual GPU对象与各自CPU完整ELF一致。

原完整benchmark单轮筛选已结束，顺序为v190 before E64/E65、v226 E64/E65、v227 E65/E64、v190 after E65/E64：

| 同批版本 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190 首control | 90.766 | 138.035 | 202.02 | 284.37 |
| v226 A unicast RT | 92.957 | 139.077 | 203.50 | 283.98 |
| v227 ScaleA multicast early | 92.582 | 137.873 | 203.06 | 281.49 |
| v190 尾control | 90.735 | 137.919 | 201.77 | 281.25 |

八项count19、exit0、原sanity及ELF绑定通过。E64两版均慢于首尾control，停止、不扩三轮、不合入；E65单次差异不作为采纳依据。正式最优仍是v190原三轮中值91.688/137.596 μs，不能用本批较低control替换正式成绩。

八次pre均全机idle，性能区间没有本任务CPU编译、正确性或idle检查。v226 E65于17:55:38 UTC退出，立即post见独立工作树PID2544688、运行1秒、VRAM0K/queue0，随后才进入GPU；post6又见独立PID2548336，晚于本次退出。保留这组边界疑点，不把它改写为所有post首sample idle；E64两候选post无KFD、busy2–3、六秒后idle。没有自有残留，也没有杀独立任务。没有候选ATT或counter，不定量归因cache、HBM、LDS或occupancy。

复现从host进入共享repo，使用fresh label：

```bash
task=.codex_tmp/gemm1_t1536_20261017
v=226
model=optx_v${v}_runtime_tdm_policy_model_20261020
cpu=tdmpolicy${v}_cpu_repro01
accept=tdmpolicy${v}_e2e_repro01
replay=tdmpolicy${v}_replay_repro01
bash "$task/run_runtime_tdm_policy_cpu_host.sh" "$v" "$cpu" "$model"
bash "$task/run_runtime_expert_check_host.sh" 0 "$accept" \
  --source "$task/candidates/optx_core_v${v}.py" --cpu-label "$cpu" --model-label "$model"
bash "$task/run_runtime_expert_replay_host.sh" "$replay" \
  --source "$task/candidates/optx_core_v${v}.py" --acceptance-label "$accept"
# Only after complete numerical/replay gates; runner waits for all GPUs idle.
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 tdmpolicy${v}_e64_repro01 \
  --source "$task/candidates/optx_core_v${v}.py" --acceptance-label "$accept" --replay-label "$replay"
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 tdmpolicy${v}_e65_repro01 \
  --source "$task/candidates/optx_core_v${v}.py" --acceptance-label "$accept" --replay-label "$replay"
```

每个CPU/数值/性能单项120秒timeout，原benchmark/warmup5/iters20/count19/门限0.01保持。只在进程内导入candidate，不覆盖生产source或selector。N128证据249项/44779452 B、五shard已下载逐member及archive SHA通过，receipt为`logs/runtime220_v225_n128_completed_downloaded_receipt_20261020.json`；原高压缩归档失败保留。v226/v227完成证据129项/18698762 B、五shard也已下载逐SHA通过，不含`.out`，manifest为`logs/runtime226_v227_tdmpolicy_evidence_20261020.sha256`，receipt为`logs/runtime226_v227_tdmpolicy_downloaded_receipt_20261020.json`。原helpers冻结于`frozen_helpers/runtime226_v227_completed_20261020/`，复现入口仍从task根运行。性能/完整e2e与对象汇总为`logs/runtime226_v227_tdmpolicy_summary_20261020.json`。

## v228：仅跳无效 A/ScaleA LDS reads（2026-10-20）

v190已用wave-uniform scalar branch跳完全无效M16的WMMA，仍读取全部A/ScaleA。v228只在`load_lds_data`按相同row-start条件跳无效A M16 reads、无效ScaleA M32 reads；partial group完整保留。每个跳过的register bank写零，B/ScaleB读取和cache、TDM/OOB zero-fill、四级ring、两份physical A/ScaleA、READY/RELEASE、正向K和quant/output不改，E64/E65不增加specialization。完整module AST回滚该helper及五个window实参后等于v190；2056项valid-group检查通过。资料依据为Shader Guide §4.3.7.2.4 p90、§4.6.12 p144和§4.10.1 p197：DS结果必须退休后使用、WMMA需真实scalar branch、跨wave TDM不保序。

CPU label `optx_v228_liveDS_cpu_20261020_r1`实际461 VGPR、107 SGPR、LDS295936 B、private/spills0，12个signal/WAIT CFG区间通过。完整ELF=`847e77dde698c8c63633771d5999b2e67db3bdc98fc0d0e94c66a0a62684dc7f`、`.text`=`77fbfbc7b47ee5c4053eb004e38dd23a9e7be8ef573f337d3e027c244f596591`；static6476较v1904710增加，不能据DS skip宣称加速。

seed17 E64→E65→E64完整MoE八hash与指定baseline一致，同artifact/完整ELF、1536 shared routes和原0.01门限通过；dynamic initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph每种各五次replay全部payload/ScaleA mismatch0。实际GPU ELF与CPU一致。源码SHA256=`b71cb309af01099fc1957404456562b4b85ba6885da27c22c5dc09e825ed6860`，builder=`build_runtime_row_reads.py`，model=`logs/optx_v228_runtime_row_reads_model_20261020.json`，CPU/GPU经`runtime_row_reads_gate.py`绑定对象。

原完整benchmark单轮筛选结束，保持warmup5/iters20/count19、原0.01门限和正常MoE生产/消费路径：

| 同批版本 | E64 GEMM1 μs | E65 GEMM1 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|
| v190 before | 91.441 | 136.829 | 201.43 | 281.59 |
| v228 live-group DS | 107.627 | 171.426 | 221.68 | 319.40 |
| v190 after | 92.734 | 137.242 | 205.35 | 281.66 |

六项exit0、sanity、count19及actual ELF绑定通过。每次pre均全机idle，性能区间无本任务并发编译、正确性或idle检查；立即post无KFD、busy2–3，六秒后与独立post6均idle，无自有残留。两规模均明显慢于首尾control，停止、不合入、不扩三轮。static instructions增加是实际观察，尚无候选ATT/counters，不能据此将回退唯一归因为branch、DS或register压力。

69项/10191253 B完成证据已分三shard下载逐member和archive SHA验证，不含`.out`。manifest=`logs/runtime228_liveDS_completed_evidence_20261020.sha256`，receipt=`logs/runtime228_liveDS_completed_downloaded_receipt_20261020.json`，完整精度/replay/六项性能和ownership汇总=`logs/runtime228_liveDS_completed_summary_20261020.json`；当时helpers冻结于`frozen_helpers/runtime228_completed_20261020/`。汇总器初版用普通substring重复计算exit标记而失败，修正为六条完整controller行匹配后通过；未重跑或替换GPU性能数据。

复现使用新label，每个单项120秒timeout；候选只在进程内导入：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_runtime_expert_check_host.sh" 17 <fresh_acceptance_label> \
  --source "$task/candidates/optx_core_v228.py" \
  --model-label optx_v228_runtime_row_reads_model_20261020 \
  --cpu-label optx_v228_liveDS_cpu_20261020_r1
bash "$task/run_runtime_expert_replay_host.sh" <fresh_replay_label> \
  --source "$task/candidates/optx_core_v228.py" --acceptance-label <fresh_acceptance_label>
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 <fresh_perf_label> \
  --source "$task/candidates/optx_core_v228.py" \
  --acceptance-label <fresh_acceptance_label> --replay-label <fresh_replay_label>
```

## v229/v230：缩减完全 OOB 的 A/ScaleA TDM surface（2026-10-20）

v229保留原DS/WMMA及LDS容量，raw descriptor按有效M16/M32 group缩小A/ScaleA传输高度，零surface用`tile_dim0=0`仍发一条pred1 TDM。Shader Guide #65 §4.10.1 p197及Table80 pp202–203说明Tensor-Done每instruction一次、`tile_dim0=0`为NOP、`tile_dim1=0`为dimension unused；仅修改后者不能表示零行。该版CPU E64→E65→E64确实同artifact/完整ELF，但actual资源451 VGPR/107 SGPR/LDS295936 B，private0、VGPR spill0、**SGPR spill8**。原CFG审计在input-prime的Tensor-Done wait和SIGNAL间遇到`s_set_vgpr_msb`而拒绝。依据§3.3.2.3 “Waves with up to 1024 VGPRs”（printed p34），此指令修改后续vector register寻址mode，本身不是memory操作；当前仍保留原审计失败，且资源门禁独立失败，因此没有运行GPU。不能将该结果记为数值bug或性能回退。

v229源码SHA=`0aef33335304b114ff98292f473b52168aa6f3a9d2669c6560801a768ae95407`，完整ELF=`3b7bb796e543d8cb9f7468ed5c84c0ab7be0bf273a23790b3213be0e4a844307`，`.text`=`ce6134cc968d1d435acfbdedd3bb0813c0dbd55b3e21287655cae3942a7496b9`。原builder的57568项仅是范围检查，地址部分为同公式比较，不能视为实际descriptor审计。仓库`mega_moe_gfx1250/tdm_prims.py`另记载单行/零高维stride的engine hang；它是已有实现经验，不据此断言所有一行TDM非法，也不在此版运行GPU试探。

v230回到v190，只缩A，ScaleA/B/ScaleB全部保留原copy atom。A的`tensor_dim1=min(max(ext,0),8)`，`tile_dim1=max(bound,2)`，最小两行；tiny/empty尾仍写两行OOB zero-fill，避免单行或零行编码。partial M16完整保留，inner tensor bound恢复原`0x7fffffff`；其余global/48-bit stride、multicast mask15/early-timeout、TH0、physical LDS位置、四级ring、READY/RELEASE、正向K0..K55及quant/output不变。未写入的完全无效A行仍会被原DS读取，但原scalar predicate不让它们进入有效WMMA，任何有效组均完整写入。该推理仍需GPU短tail与dynamic验证，不能由AST直接称硬件正确。

v230 source=`0852264141ff224aa2a02c7d668171aa7de6a231b08333885ca4b8b6577b41eb`，actual完整ELF=`1944fbe061fd00b764548c86975b1f05d68ba20699507bcb50c39c2bd344d52a`，`.text`=`806b46d4e308eb082ead41e2871b20b39a6bcc3f107efb88617d30e51fce9c7b`。CPU465 VGPR/106 SGPR/LDS295936 B、private和两类spill0、static4742，12个CFG区间通过；E64→E65→E64只有一个artifact。`audit_runtime_trim_a_tdm.py`执行actual descriptor AST，115136项覆盖0..256 tail、全部28 Ktile、两half和四tensor；四项实际字段负对照被拒绝。原生A prime GROUP1 chain核到`v_med3_i32(ext,0,8)`、`v_max_u32(bound,2)`及field bank；未模拟全部native路径，source/AST及GPU证明分别保留其范围。严格gate=`runtime_trim_a_tdm_gate.py`。

seed17双规模完整MoE八hash/原0.01门限通过，E65保留1536条shared route，同artifact/完整ELF与CPU一致；dynamic initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph各五次replay全部payload/ScaleA mismatch0。随后21个短tail、空expert和热点路由的完整MoE补验全部通过。E64 expert0 counts为0/1/15/16/17/31/32/33/127/128/129/143/144/145/255/256/257；E65为0/1/129/hot8，hot8的前八个routed experts各1536 rows，另外56个为空，shared expert64仍1536 rows。每case核route IDs/weights、valid rows、payload、M32 ScaleA、GEMM2、e2e和reference，八项mismatch0；实际编译对象和完整ELF保持同一份。

首轮边界工具在v190 parent结果解码时因CPU rows与默认CUDA `torch.arange`混用退出1，尚未切换candidate。仅修正工具为显式`device="cpu"`并断言rows/flat/mx设备一致；原helper和失败日志保留，不称kernel数值bug。旧/新helper SHA分别为`861da3d55aa0b2fee7f8677291da3a1227a341a07afda1b259968dd0777e5f8a`/`89d09853422a9970e5aa86a99cadfc43eb366e7b8c97dde5ca85e085fc96b987`，四个fresh r2批次均exit0，每项120秒上限。

原完整benchmark同批首尾v190对照与v230如下，均warmup5/iters20/count19、sanity及actual ELF绑定通过。表内是单轮筛选，不替换正式三轮成绩。

| 同批版本 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 91.517 | 138.061 | 59.043 | 88.793 | 201.60 | 281.52 |
| v230 trimA2 | **93.858** | **141.196** | 59.657 | 89.620 | 205.56 | 285.64 |
| v190 after | 91.781 | 136.922 | 58.463 | 89.270 | 202.12 | 281.06 |

v230两种规模均回退，停止、不合入、不扩三轮。没有v230 ATT/counters，不能将回退定量归因于descriptor开销、HBM或cache。六项pre全机idle；本轮机器只暴露`/sys/class/drm/card1/device`，记录busy0、VRAM176582656/463856467968、无KFD所有者。立即post无所有者、busy2–3，六秒复查及独立post6恢复idle，无本任务残留；性能期间没有本任务并发编译、正确性或idle检查。

123项证据共19321773 B已下载并逐member/archive核SHA256，不含`.out`，包括失败工具、CPU/descriptor审计、完整MoE/replay、21case actual对象及原性能/pre/post。manifest为`logs/runtime230_trimA2_completed_evidence_20261020.sha256`，完整汇总为[本轮证据](../.codex_tmp/gemm1_t1536_20261017/logs/runtime230_trimA2_completed_summary_20261020.json)。三个shard SHA为`d7242b4e057600d729ca3f521e4bdbe71f073316be23eda37416b42defe973d9`、`3c1a6b6b1460f076e080c11318c03e769b48217fe6cd5593784122f949542abe`、`dee139620edcc0f46331341c2ab8e58a47feb0d218614fa47fced21adc44a270`。生产仍v190，正式E64/E65中值91.688/137.596 μs，E64距80 μs差11.688 μs。

fresh-label复现（先使用已完成的CPU/descriptor与双规模replay门禁，边界工具使用修正版本；host仓库根目录，构建/测试由helper进入`dsr1_ep4`）：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_runtime_expert_perf_host.sh" runtime190 64 trimA2_perf_e64_repro01 \
  --source "$task/candidates/optx_core_v230.py" \
  --acceptance-label optx_v230_trimA2_moe_seed17_20261020_r1 \
  --replay-label optx_v230_trimA2_replay_20261020_r1
bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 trimA2_perf_e65_repro01 \
  --source "$task/candidates/optx_core_v230.py" \
  --acceptance-label optx_v230_trimA2_moe_seed17_20261020_r1 \
  --replay-label optx_v230_trimA2_replay_20261020_r1
```

## v231/v232：N256 两级、三级 input ring（2026-10-20）

在 v230 缩减 TDM surface 无收益后，本轮只调整 v190 N256 input ring 容量和相应 stage/READY 距离。两份 physical A128/ScaleA128、8 waves、三 A/ScaleA register banks、B/ScaleB cache、四 N-task、正向 K0..K55、全部有效 routes、原量化和输出保留；不是已放弃的共享 A 单槽方向。候选仍接受原公开 `t256/w2x4/b4` launch 参数，内部 ring 和实际 symbol 分别标记 b2/b3。生产源码、selector 和原 benchmark 不变。

| 候选 | input ring | actual VGPR / SGPR | actual LDS B | private / VGPR spill / SGPR spill | static instructions | 结果 |
|---|---:|---:|---:|---|---:|---|
| v231 | 2 | 463 / 104 | 156672 | 0 / 0 / 0 | 4529 | 完整数值门禁通过，原性能回退，停止 |
| v232 | 3 | 489 / 107 | 226304 | 0 / 0 / 18 | 5418 | CPU 资源和原 CFG 审计拒绝，未运行 GPU |

v231 source SHA256=`ec943bd6cc87eab66126764fe62ae2530c477e2a1dabc65b7ea344676c2da1a5`，actual 完整 ELF=`d6841babd4bdb46e7924c658ffcd6433503e2c0c16f8f12bec071aa121ebc090`，`.text`=`66243847094c0d1b63d4e2f457bdface984346cf87d33230cd98b91848db78d5`。v232 source=`5ed0cffadf99c84a2f37cd810fe0fd760a9203b83ef32f3294b4eb3aed367af7`，CPU 完整 ELF=`1e1699533ff31ad53ffc77ebf42f5708f412d8ced620876555f21589f0c8db6a`，`.text`=`c41224aac70f57c195a7dbabd1b4fe2122c748d982a979f3f96956c2eda55dde`。两版 CPU 均 E64→E65→E64 cache1、同一 artifact 和完整 ELF。v232 prime 的 Tensor-Done wait 与 SIGNAL 之间出现 `s_set_vgpr_msb 0x100`，原审计拒绝；该指令根据 Shader Guide #65 §3.3.2.3 printed p34 修改后续 vector register 寻址，不是 memory 指令。保留原失败，且 SGPR spill 独立不合格；不将 CPU 拒绝称为数值 bug 或性能回退。

v231 initial `tensor_wait(1)`；successor step0 用 wait2 排除前一 task 的两个 output TDM，其余 step wait0，reader DScnt 退休与完整 WG WAIT 之后才 refill。每 owner 每 Ktile 一条 input TDM、每 wave 四 task 共112 input tiles及8 output requests保持。三级版因28 mod3=1，stage使用 `(kt+tile_idx)%3` 连续旋转，没有在 task 边界错误重置。

`audit_runtime_n256_ring.py` 执行实际源码 AST：229376 reader cases、4587520 symbolic DS loads、460544 descriptors/57568 extent cases、32256跨task地址 cases通过，覆盖tail0..256、28 Ktile、四 task、两 first-half。七项错误 READY、遗漏 DS release、提前 refill、提前覆盖 B0、错误 ScaleA owner和错 LDS alignment负对照全部拒绝。模型首版反复解析 AST 超过120秒，被 subprocess timeout终止，无匹配 Python 残留；只缓存 compiled AST 代码，原assert、coverage和候选源码不变，成功记录使用新label `optx_v231_runtime_n256_ring_model_20261020_r2`。初版helper/timeout记录保留。模型是 CPU AST/顺序依赖检查，不是任意异步时序或原生 ISA 证明；v231原CFG另通过12区间，四prime无WMMA、八release最多8条WMMA。

seed17 E64→E65→E64完整MoE的八项hash与指定baseline一致，logits_diff约 `3.47663789e-06 / 3.77883117e-06`，原0.01阈值、finite和1536条shared routes保持。initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph各五次replay通过，payload/ScaleA mismatch0。21个完整MoE边界case全部八项mismatch0，覆盖E64 expert0 counts0/1/15/16/17/31/32/33/127/128/129/143/144/145/255/256/257及E65的0/1/129/hot8；hot8的前八routed experts各1536、其余56空、shared expert64仍1536。所有GPU实际ELF等于CPU对象，两种规模同artifact，无E65 specialization。

原完整 `--scenario bench --iters20` 的 warmup5/count19、sanity和actual ELF绑定保持。同批六项单轮筛选结果：

| 对照/候选 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 91.361 | 138.822 | 59.364 | 89.761 | 203.52 | 283.58 |
| v231 S2 | **129.015** | **220.959** | 59.128 | 90.981 | 240.83 | 366.84 |
| v190 after | 91.334 | 137.481 | 58.910 | 89.666 | 200.71 | 283.09 |

v231两种规模均慢于首尾control，停止，不扩三轮、不合入。LDS减少139264 B并没有带来目标收益；没有候选ATT/counters，不定量归因occupancy、cache、HBM或compiler。硬件依据为Shader Guide #65 §3.3.4 printed p36/§4.7.1 p160：LDS与WGP$共384 KiB，按64 KiB partition；§4.10.1 p197：Tensor-Done每instruction一次、同wave保序、跨wave无保证。仅据LDS容量不能推断实际驻留WG数或cache加速。

六项pre首sample均全机idle；本轮可见GPU仅 `/sys/class/drm/card1/device`，busy0、VRAM176582656/463856467968、无KFD所有者。性能期间没有本任务并发编译、正确性、传输或idle polling。立即post均无所有者、busy2–3；E64四个post/post6最终idle。E65 before在退出后post/post6分别出现PID3458867/3460376，报告运行5/4秒；候选E65退出后post/post6出现PID3475855/3477413，报告运行2/1秒，均晚于本workload退出。原ownership边界保留，按六秒等待恢复idle，没有干预外部任务，没有自有残留，不写成每次post首sample都idle。

127项/18926348 B的source、model、CPU/ISA、完整MoE/replay/边界、原性能及当时helpers已经分三shard下载并逐member/archive SHA核验；无`.out`。manifest=`logs/runtime231_v232_n256_ring_completed_evidence_20261020.sha256`，汇总=[N256 ring完整结果](../.codex_tmp/gemm1_t1536_20261017/logs/runtime231_v232_n256_ring_completed_summary_20261020.json)，当时helpers冻结于`frozen_helpers/runtime231_v232_completed_20261020/`。生产正式v190三轮中值91.688/137.596 μs保持，E64距80 μs仍差11.688 μs。

复现CPU/边界的原labels必须改为fresh labels；性能controller中的验收labels需绑定对应的新CPU/model/MoE/replay记录，不能直接覆盖旧证据。host从共享仓库根运行：

```bash
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_runtime_n256_ring_cpu_host.sh" 231 <fresh_cpu_label>
bash "$task/run_runtime_n256_ring_edges_host.sh" 0 5 <fresh_edge_label>
# 原完成批次的六项首尾control screen，以新的suffix保留旧数据。
bash "$task/run_runtime_n256_ring_screen_host.sh" <fresh_suffix>
```

后续隔离v233/v234/v235仅对原三级v232分别关闭machine LICM、postRA LICM或两者，kernel body除symbol外逐字相同，descriptor、协议、有效数值、quant/output和runtime E ABI保持。它们只进入CPU筛选，不继承v231 GPU验收；原零spill和CFG门禁不放宽。

## v233–v241：三级 ring 的 scalar identity 修复与 LICM 对照（2026-10-20）

本轮继续 N256/S3，保留上下两份 physical A128/ScaleA128、8 waves、四个 N-task、全部有效 routes、正向 K0..K55、原量化和输出及 runtime-E ABI。不是已放弃的共享 A 单槽方向。生产 v190、selector 和用户原 benchmark 均未修改。

| 候选 | 改动 | VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 结论 |
|---|---|---:|---:|---|---|
| v233 | S3 仅禁 machine LICM | 488 / 107 | 226304 | 0 / 0 / 10 | CPU 拒绝，未 GPU |
| v234 | S3 仅禁 postRA LICM | 489 / 107 | 226304 | 0 / 0 / 18 | 与 v232 同 `.text`，资源/原 CFG 拒绝，未 GPU |
| v235 | S3 禁两项 LICM | 488 / 107 | 226304 | 0 / 0 / 10 | CPU 拒绝，未 GPU |
| v236 | unsigned rem3 + 原 scalar opacity | 448 / 102 | 226304 | 0 / 0 / 0 | 模型/原 CFG 通过，首项完整 E64 logits 失败，停止 |
| v237 | bounded rem3 + 原 scalar opacity | 448 / 98 | 226304 | 0 / 0 / 0 | 模型/原 CFG 通过，首项完整 E64 logits 失败，停止 |
| v238 | v236 的 stage identity 改为 `=s,0` | 448 / 102 | 226304 | 0 / 0 / 0 | 完整精度/replay/边界通过，性能回退，停止 |
| v239 | v237 的 stage identity 改为 `=s,0` | 448 / 98 | 226304 | 0 / 0 / 0 | 完整精度/replay/边界通过，性能回退，停止 |
| v240 | v238 恢复默认两项 LICM | 450 / 107 | 226304 | 0 / 0 / 2 | 原 CFG 通过但资源拒绝，未 GPU |
| v241 | v239 恢复默认两项 LICM | 461 / 107 | 226304 | 0 / 0 / 0 | 完整精度/replay/21边界通过，性能回退，停止 |

### scalar identity bug 与证据范围

v236/v237 把 ring stage 接入空 inline asm 的 `=s,s`。该约束允许输入和输出使用不同 SGPR；空 asm 没有复制指令，因此不能保证返回值等于输入。新候选局部 helper `_stage_scalar_identity` 使用 `=s,0`，显式绑定同一寄存器。依据本地 LLVM LangRef 的 Inline Assembler Expressions / Input constraints，lines 5611–5618：整数约束将 input 绑定为指定 output 的同一寄存器。共享 `gemm_common_gfx1250.py` 和生产 v190 没有改动，不把本轮隔离候选失败推广为 v190 的数值 bug。

v236→v238 完整 `.text` 长 30080 B，仅两个字节变化。`.text` VMA 为 `0x1b00`，section offset 15574/17498 分别绑定到 PC `0x57d4`/`0x5f58` 的目标寄存器字段：

```text
PC 0x57d4: s_sub_co_i32 s12, s1, s12  -> s_sub_co_i32 s14, s1, s12
next:      s_add_co_i32 s15, s14, s63
PC 0x5f58: s_add_co_i32 s86, s86, 0xff000019 -> s_add_co_i32 s87, s86, 0xff000019
later:     s_lshl_b32 s83, s87, 12
```

完整字节对照与 ISA producer/consumer 连线共同支持绑定错误结论；v238/v239 的 GPU 完整结果独立验证了修复。v237→v239 的寄存器重排较多，不称其仅有两条变化。ISA 对照脚本仅解析带 operand 的 4328 行，完整 inspection 是 v236/v238 的 4969 条指令；不混用两种计数。

模型初版还出现一个独立工具问题：复制 scope 后，已构造 Python function 的 globals 仍指向旧 task 地址，造成 62 个不匹配。重新把同一实际 `ring_stage`/`ring_owner_stage` AST 绑定到新 mutable scope 后，124 个判定通过。只改模型工具，不改 candidate 和原 assertions；失败诊断保留。这不是 kernel 数值修复的替代证据。

### v238/v239 完整数值与原性能筛选

每版 actual-AST 模型通过 229376 reader cases、4587520 symbolic DS loads、460544 input descriptors、57568 extent cases、32256 跨 task 地址及八个负对照；原 ISA CFG 的 12 区间通过。三级 ring 仍连续旋转 `(kt+tile_idx)%3`，没有在 task 边界重置 stage。模型不是任意异步时序或原生执行证明；Tensor-Done 的依据仍为 Shader Guide #65 §4.10.1 printed p197：同 wave tensor 指令按序，跨 wave 及其他 memory 不自动排序。

seed17 完整 E64→E65→E64 的八项 hash 与指定 baseline 相同，logits_diff 为 `3.47663789e-06 / 3.77883117e-06`，原 `0.01` gate、finite 和 1536 shared routes 保持。GPU actual 完整 ELF 与 CPU 相同，每版 E64/E65 同 artifact/cache1，没有 E65 specialization。initial/sign_flip/alternate_values、poison、eager/parent graph/runtime graph 各五次 replay 的 payload/ScaleA mismatch0。每版 21 个完整 MoE case 覆盖空 expert、M16/M32/M128/M256 边界及257、E65 hot8，八项 mismatch 全零。

原完整 `--scenario bench --iters20` 的单轮首尾 control 筛选如下，warmup5、GEMM1 count19、原 sanity 和实际 ELF 绑定保持。不是正式三轮中值。

| 对照/候选 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 91.798 | 137.323 | 59.134 | 89.457 | 203.60 | 282.23 |
| v238 | **94.779** | **148.168** | 60.296 | 90.376 | 207.34 | 293.03 |
| v239 | **96.272** | **149.593** | 60.047 | 89.847 | 208.26 | 294.47 |
| v190 after | 91.013 | 136.899 | 58.856 | 90.522 | 200.77 | 282.07 |

两版在两种规模均慢于首尾 control，停止、不采纳、不扩三轮。LDS 少 69632 B、VGPR 少15，并没有取得性能收益。没有这两版 ATT/counters，不能定量归因 occupancy、HBM、cache 或某条指令。八项 pre 均 idle；立即 post 无 KFD 所有者、busy2–3，按六秒复查和独立 post6 恢复 idle，没有本任务残留。当前只暴露一张物理加速卡的 `/sys/class/drm/card1/device` 及 xcp 虚拟节点，不将其描述为四张物理 GPU；性能期间本任务没有并发编译、传输、正确性或 idle polling。

v238/v239 source SHA256 分别为 `e34b16ecc1d7f689fa7c8cd44c63dfff24987af22179e049ba84253e5bd74585` / `3a26e43e05305a7074c97cdb9e40a4fc28f19312bb8e6ceb27521493edcd3198`；完整 ELF 分别为 `1bf754ac531b62a2ff6176a492a2db73e4be9b94b659da84876cf65fc12de66b` / `ea58343eef8cea29d1be2245a4ca20b360c6c4ac7dc36d06734260e055f66da1`。

295 项/58300254 B 的 completed evidence 已按五 shard 下载，逐 member/archive SHA256 验收，无 `.out`。manifest 为 `logs/runtime233_v239_s3_stage_completed_evidence_20261020.sha256`，helpers 冻结于 `frozen_helpers/runtime233_v239_s3_stage_completed_20261020/`，结果见[完整汇总](../.codex_tmp/gemm1_t1536_20261017/logs/runtime233_v239_s3_stage_completed_summary_20261020_r2.json)和[下载验收](../.codex_tmp/gemm1_t1536_20261017/logs/runtime233_v239_s3_stage_downloaded_receipt_20261020.json)。汇总 v1 的 e2e regex 错误匹配 `number us`，原输出为 `fused_moe end-to-end us = number`；只在独立 v2 修正解析并重算旧 raw，没有重跑 GPU、改数据或覆盖冻结 v1。r2 汇总 SHA256=`ff4565ba8b2e916060fb9f9de01587ebaf968451e4b94a85bcd712796cccab97`。两个字节的完整证据见[字节绑定](../.codex_tmp/gemm1_t1536_20261017/logs/runtime236_v238_text_bytes_bound_20261020.json)。

### v241 完整结果与复现

v240/v241 只移除 `disable-machine-licm`、`disable-postra-machine-licm` 两项和改变诊断 symbol，kernel body、tied identity、地址、协议、有效数值保持。v240 零 private 但 SGPR spill2，资源 gate 拒绝，未运行 GPU。v241 实际 CPU 为461/107/LDS226304 B、private/spills0、4992条 static instructions，完整 ELF=`fe149f0d3a292450e8f4b1ea7ef74517f113a35b7a451b095814c6a494c33cff`；E64→E65→E64 同 artifact。source=`fce7ad40435f1961188c3bcfd4b9a0a375a27934cd69a5fec9905c11fec0f0c1`。

v241 actual-AST、八负对照及原12区间 CFG 通过。seed17 完整双规模八hash与指定baseline相同、E65 1536 shared routes保持、CPU/GPU完整ELF相同，dynamic/poison/eager/两graph各五次replay mismatch0。21个完整MoE边界case全部八项mismatch0，覆盖空expert、M16/M32/M128/M256边界、257与E65 hot8。独立 v241 helpers 保存全部原 assertions，不覆盖已完成 v238/v239 helpers。

原完整benchmark单轮筛选已结束，保持warmup5/iters20/count19、原sanity和actual ELF绑定。两种规模都慢于首尾v190，停止本轮S3对照，不扩三轮、不合入；生产正式最优仍 v190 三轮 **91.688 / 137.596 μs**，E64 距80 μs差 **11.688 μs**。

| 对照/候选 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 91.559 | 137.455 | 59.374 | 90.003 | 203.12 | 284.87 |
| v241 | **96.590** | **148.227** | 60.138 | 90.066 | 208.96 | 294.52 |
| v190 after | 91.793 | 137.542 | 58.742 | 89.094 | 202.46 | 282.27 |

126项/19306862 B证据已按两个shard下载，逐archive/member SHA256和完整ELF核验，包含原CPU/model/完整MoE/replay/21边界、六轮性能及逐轮pre/post/post6；无`.out`。所有pre最后sample均idle。立即post无活跃KFD，busy1–3；v190 after E64首post含一条`<已退出>`记录，仍原样保留。部分稍后post或post6出现独立任务，随后六秒复查恢复idle，不能将后续恢复写成首post就idle，也不能只凭此排除未知的短时重叠。计时期间本任务无并发编译/传输/正确性或idle polling。完整边界和结果见[本地独立核验汇总](../.codex_tmp/gemm1_t1536_20261017/logs/runtime240_v241_completed_downloaded_summary_20261020.json)，SHA256=`981d49a9f89b4ffaa7468a833f919799f8f1b7c2cfda59bb13931ec67c3d29e9`。

```bash
# Host repository root; labels below identify the completed evidence.
task=.codex_tmp/gemm1_t1536_20261017
# New CPU/model labels preserve prior output; each item is bounded to120s.
bash "$task/run_s3_tied_licm_policy_cpu_host.sh" 241 optx_v241_licm_cpu_repro_01
bash "$task/run_s3_tied_licm241_model_host.sh" 241 optx_v241_licm_model_repro_01 optx_v241_licm_cpu_repro_01
bash "$task/run_s3_tied_licm241_check_host.sh" 17 optx_v241_licm_moe_repro_01 \
  --source "$task/candidates/optx_core_v241.py" \
  --cpu-label optx_v241_licm_cpu_repro_01 --model-label optx_v241_licm_model_repro_01
bash "$task/run_s3_tied_licm241_replay_host.sh" optx_v241_licm_replay_repro_01 \
  --source "$task/candidates/optx_core_v241.py" --acceptance-label optx_v241_licm_moe_repro_01
# All21 full-MoE edge cases must also pass before timing.
# The completed v241 screen can be reproduced with a fresh suffix:
bash "$task/run_s3_tied_licm241_screen_host.sh" repro_02
```

## v242–v245：Scale LDS base 的 tied identity 隔离筛选（2026-10-20）

从v190分别只给ScaleA/ScaleB、仅ScaleA、仅ScaleB的LDS base加局部`=v,0`空inline asm。它保留同一low32地址并约束同寄存器，普通LLVM LDS loads和依赖处理仍存在；不改TDM/waits、四级ring、两份physical A/ScaleA、有效K、quant/output或runtime E。回滚helper/import/address和symbol后完整源码等于v190；actual-AST的59392地址case、475136 low32 scale consumer words通过，包含32-bit wrap对照。依据CDNA5 ISA §15.15 printed pp721–722：2addr offset以4 B为单位，stride64为256 B；LLVM tied input见LangRef lines5611–5618。地址及静态合并不证明性能收益。

CPU同进程E64→E65→E64都只有一个artifact/完整ELF。原12区间CFG、private/spills0、LDS295936 B通过，但三者总指令均超过事前`<=4710`门槛，全部停止在CPU，没有GPU精度或性能数据；未放宽门槛。

| 候选 | 改动 | VGPR / SGPR | 静态总指令 | Scale DS issue / dwords | `s_wait_dscnt` | 结果 |
|---|---|---:|---:|---:|---:|---|
| v190 | 原始 | 463 /107 | 4710 | 54 /96 | 109 | 正式最优 |
| v242 | ScaleA与ScaleB base tied | 463 /105 | 4719 | 48 /96 | 121 | CPU拒绝，未GPU |
| v243 | 仅ScaleA base tied | 463 /105 | 4726 | 52 /96 | 137 | CPU拒绝，未GPU |
| v244 | 仅ScaleB base tied | 464 /107 | 4711 | 48 /96 | 100 | CPU拒绝，未GPU |

v242/v243/v244完整ELF分别为`1ebdb89d47fed933f936586c8b3230c18eb6f1d217fda4f1c3a0ec8707ffdee4`、`c2aeedb2637301a236727799eea55e5fabe49e327ca16d554ad55a3ea6879a47`、`106680ffad8674e496feccee02dd03990a4c67cf89d52847677703f56423a3a4`。三者原DS data384、scale dwords96、WMMA256保持；不能把CPU gate拒绝称为精度bug或实测回退。

隔离v245仅将v244与已验证v215的uniform inactive-capacity guard组合，不继承旧GPU验收。实际源码lookup和guard AST在144个routing fixture、323328个M/N-task判定case通过，shared expert tiles仍保留；kernel有效task body、地址/TDM/DS/WMMA/quant/output保持原AST。原scale地址检查从逐字相同的v244 `calc_window_addr`继承，未称为完整FlyDSL或native证明。v245 source SHA256=`fe6e47ba3322a608e4f49154cc2d0fde1324d4511582e570f987dcca7397782f`。

### v245 完整正确性和性能结论

CPU实际464 VGPR/106 SGPR/LDS295936 B，private及两类spills0；4709条静态指令、scale issue48/dwords96、`s_wait_dscnt`100、data `ds_load_b128`384、WMMA256。原12区间CFG及事前`<=4710`门槛通过。完整ELF=`7fbd5d7631aa1c6fdea33c24f9729cb9f9e475fd656900517cd438ed0a9f70f1`，`.text`=`6e9f03f6431e8d8a68e5e6af0eecda737cab3a96587a29c4c58332fd4d39eca1`。E64→E65→E64在CPU与GPU各自都为同一artifact/cache1，完整ELF相同，没有E65 specialization。

seed17完整双规模八项hash与指定baseline相同，logits_diff为E64 `3.476637889e-06`、E65 `3.778831168e-06`，原`0.01` gate、finite及E65的1536 shared routes保持。initial/sign_flip/alternate_values与poison、eager/parent graph/runtime graph各路径各五次replay的payload/ScaleA mismatch0。21个完整MoE case覆盖空expert、M16/M32/M128/M256边界及257、E65 hot8，八项mismatch全部为0。

原完整`--scenario bench --iters20`单轮首尾screen保持warmup5、GEMM1/GEMM2 count19及原sanity；不是新正式中值。

| 对照/候选 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 91.338 | 136.878 | 59.524 | 89.263 | 203.11 | 280.22 |
| v245 | 90.697 | 137.186 | 58.991 | 89.746 | 201.18 | 282.93 |
| v190 after | 91.983 | 137.517 | 60.294 | 89.470 | 204.20 | 282.89 |

E64首轮有改善，进一步完成三轮独立进程交替与三轮同HIP context配对，第二轮反向，仍调用原完整benchmark。两批独立记录，不混算中值。

| 测量批次 | workload /版本 | 三轮 GEMM1 μs | 中值 μs |
|---|---|---|---:|
| 独立进程交替 | E64 v190 | 92.006 /92.476 /91.167 | 92.006 |
| 独立进程交替 | E64 v245 | 91.888 /90.646 /92.489 | 91.888 |
| 独立进程交替 | E65 v190 | 137.972 /137.932 /137.006 | 137.932 |
| 独立进程交替 | E65 v245 | 138.322 /137.360 /137.715 | 137.715 |
| 同HIP context | E64 v190 | 92.131 /91.937 /92.171 | 92.131 |
| 同HIP context | E64 v245 | 91.034 /91.216 /90.948 | 91.034 |
| 同HIP context | E65 v190 | 138.615 /136.705 /137.491 | 137.491 |
| 同HIP context | E65 v245 | 137.759 /138.326 /137.287 | 137.759 |

独立进程E64中值只快0.118 μs，第三轮反而慢1.322 μs。同context E64三轮都快、中值改善1.097 μs/1.19%，但E65反向轮慢1.621 μs。综合稳定收益未确认，v245暂不采纳，不以较低单次或某一批改写正式v190。正式E64/E65三轮仍为 **91.688 /137.596 μs**，距E64目标 **11.688 μs**。未采集v245 ATT/counters，不唯一归因HBM、cache、occupancy或bank conflict。

275项/37181076 B completed evidence已按三shard下载，逐archive/member SHA256、CPU gate、GPU实际完整ELF、八hash/21case、原计时及逐轮pre/post/post6在本地独立重算通过，无`.out`。汇总见[独立验收JSON](../.codex_tmp/gemm1_t1536_20261017/logs/runtime242_v245_completed_downloaded_summary_20261020.json)，SHA256=`4ec38a15c9518038039f8aaafbec8c7620a3e6d4868b1c69770b370767e9d062`；manifest=`logs/runtime242_v245_completed_evidence_20261020.sha256`。每次pre最后sample均idle；立即post无活跃KFD且busy2–3，六秒复查恢复idle。部分post6出现`jli-mi450`独立任务及`<已退出>`，按六秒轮询后恢复idle，原记录保留；不把最终idle替代首post，也不凭后续idle排除未知短时重叠。计时期间本任务没有并发编译/传输/正确性或idle polling。

```bash
# Host repository root; use fresh labels and the unchanged CPU gate.
task=.codex_tmp/gemm1_t1536_20261017
bash "$task/run_scale_b_capacity245_cpu_host.sh" optx_v245_scale_b_capacity_cpu_repro_01
# Fresh numerical acceptance before any timing; every item is bounded to120s.
bash "$task/run_scale_b_capacity245_check_host.sh" 17 optx_v245_scale_b_capacity_moe_repro_01 \
  --source "$task/candidates/optx_core_v245.py" \
  --cpu-label optx_v245_scale_b_capacity_cpu_repro_01 \
  --model-label optx_v245_scale_b_capacity_source_contract_20261020
bash "$task/run_scale_b_capacity245_replay_host.sh" optx_v245_scale_b_capacity_replay_repro_01 \
  --source "$task/candidates/optx_core_v245.py" --acceptance-label optx_v245_scale_b_capacity_moe_repro_01
# The completed screen/pairs use *_20261020_r1 acceptance labels.
bash "$task/run_scale_b_capacity245_screen_host.sh" repro_02
bash "$task/run_scale_b_capacity245_pairs_host.sh" repro_02
bash "$task/run_scale_b_capacity245_context_pair_host.sh" 245 64 optx_v245_context_repro_01 bc \
  optx_v245_scale_b_capacity_moe_seed17_20261020_r1 \
  optx_v245_scale_b_capacity_replay_20261020_r1
```

### v246–v251 CPU负结果与下一候选

v246/v247仅将LLVM `amdgpu-schedule-metric-bias`设为0/100；v248仅ScaleB改普通LLVM address-space3 i32 raw load；v249两侧都改。raw reader实际AST的20种case覆盖低32-bit wrap和`read_weight=False`，请求地址与bank结果保持。v246/v247/v248实际`.text`均与v245完全相同，CPU同text门槛停止，不重复GPU。v249实际464/106/LDS295936 B、private/spills0、scale issue48/dwords96、DScnt100，但4715条指令超过原4710门槛，CPU停止。未放宽门槛，不称为精度失败或实测性能回退。

根据v245已经合并的`ds_load_2addr_*`，v250只把DS hint容量由20改18，v251另改四个MFMA2 hint组。所有reader、地址、TDM/DS retirement、WG barrier、有效WMMA、量化和输出保持；hint不是hardware wait。两实际`.text`也与v245完全相同，均只CPU、未GPU。v252/v253已完成Scale/operand的B-first与A-first读取次序native筛选；20个actual-reader AST case请求multiset/bank结果相同，但实际4741/4773条指令超过原4710门槛，CPU停止，未运行GPU。

## v252–v266：prepared body、row predicate、data base 与异常复测收尾（2026-10-20）

按用户“异常实验重测，手头实验做完后暂停”的最新要求，2026-10-20 已完成 v265 的完整数值验收和性能筛选，并完成此前异常 E65/v190 的三次原完整 benchmark 重测。GEMM1/GEMM2/MoE e2e 重测中值为 **136.931 /89.441 /281.17 μs**，旧 161.590 /315.866 /982.26 μs 异常未复现；保留旧记录，不将其作为稳定对照。**本轮优化已暂停**，v266 只完成 CPU，未启动其 GPU 验证、性能或新 counters。正式最优仍是基于 OPTX 方案的 v190，E64/E65 原正式三轮中值 **91.688 /137.596 μs**，E64 距80 μs差 **11.688 μs**。

### v252/v253 CPU 筛选与 v254/v255 完整结果

v252 的 B-first、v253 的 A-first Scale/payload 读取顺序已完成新 native 筛选：资源均464 VGPR/106 SGPR/LDS295936 B、private及两类spill0，但静态指令为4741/4773，超过原4710门槛，CPU停止，未运行GPU。此前“仍待新native验收”的表述已更新；模型请求multiset相同不代表已通过GPU或取得性能收益。

v254 将 v211 的 prepared owner input body 与 v245 的 ScaleB tied base、完全无效 capacity guard 组合；v255 则采用 v212 的 prepared shared compute body，同样组合 v245。task入口预先准备owner不变的descriptor字段，v255合并owner分支重复的静态body，保留原64-bit地址、四级ring、两份physical A128/ScaleA128、四个N-task、完整正向K0..K55、wait/READY/RELEASE和quant/output。有效request及数值没有省略；静态指令减半不能解释成有效WMMA计算减半。

| 版本 | VGPR / SGPR | LDS B | private / VGPR spill / SGPR spill | 静态总指令 | Scale DS issue / dwords | `s_wait_dscnt` |
|---|---:|---:|---|---:|---:|---:|
| v190 | 463 /107 | 295936 | 0 /0 /0 | 4710 | 54 /96 | 109 |
| v254 | 463 /104 | 295936 | 0 /0 /0 | 4651 | 48 /96 | 100 |
| v255 | 461 /96 | 295936 | 0 /0 /0 | 3464 | 24 /48 | 68 |

两版均通过seed17完整E64→E65→E64八hash、原0.01精度门限、finite、1536 shared routes、同CPU/GPU完整ELF，以及dynamic/sign_flip/alternate_values、poison、eager/两graph各五次replay和每版21个完整MoE边界。原完整bench首尾筛选如下：

| 版本 / 对照 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 92.826 | 137.354 | 59.262 | 89.219 | 203.740 | 281.740 |
| v254 | 90.593 | 138.086 | 59.414 | 89.632 | 201.680 | 281.970 |
| v190 after | 92.251 | 137.677 | 60.561 | 89.795 | 204.300 | 284.130 |

v254首轮E64改善，追加三轮独立进程交替，第二轮反向：E64 v190为92.517/90.404/92.371 μs，中值92.371；v254为92.491/91.462/91.334 μs，中值91.462。第二轮候选慢1.058 μs，收益不一致。E65 v190中值137.620 μs，v254为137.704 μs，略慢，未采纳。三轮GEMM2/e2e数据及逐轮所有权原记录保留在[独立验收汇总](../.codex_tmp/gemm1_t1536_20261017/logs/runtime246_v255_completed_downloaded_summary_20261020.json)，不混入正式最优中值。

| 版本 / 对照 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 91.106 | 137.360 | 58.181 | 90.140 | 200.190 | 284.810 |
| v255 | 91.058 | 140.595 | 58.398 | 88.375 | 202.720 | 283.150 |
| v190 after | 90.781 | 137.839 | 59.652 | 88.854 | 202.890 | 281.830 |

v255 E64与control持平、E65回退，未扩三轮、未采纳。v246–v255的403项/75484354 B完成证据已下载逐SHA核验；旧freeze漏下14个重复control MLIR，原汇总明确记录使用逐字节SHA一致的代表对象，不冒称旧包包含所有control对象。本轮v256–v266的freeze逐项纳入每次control的actual MLIR，不沿用该缺口。

### v256–v263：缩减 row predicate 的寄存器生存期

八版均从v255只改wave-uniform row validity的形成方式。部分M16有有效行时继续完整累加，只有完全无效行组可跳过；未改变有效routes、DS payload、WMMA active branch、K顺序、TDM或量化输出。依据本地Shader Guide #65 §4.6.12 printed p144：WMMA忽略EXEC mask，真正skip必须由wave级scalar branch完成，不能把EXEC置零当作省略WMMA。

| 候选 | 方法 | VGPR / SGPR | 静态指令 | E64 GEMM1 μs | E65 GEMM1 μs | 同批 v190 E64 before / after μs | 同批 v190 E65 before / after μs |
|---|---|---:|---:|---:|---:|---:|---:|
| v256 | use-site vector compare + cndmask/readfirstlane | 461 /96 | 3714 | 112.601 | 182.708 | 91.327 /91.319 | 137.384 /137.373 |
| v257 | packed vector validity + bfe/readfirstlane | 463 /104 | 3628 | 105.076 | 167.989 | 92.567 /92.607 | 138.198 /136.833 |
| v258 | SALU signed compare + cselect i32 | 461 /96 | 3552 | 93.533 | 142.244 | 91.639 /91.856 | 137.441 /137.793 |
| v259 | packed scalar validity + s_bfe_u32 | 461 /104 | 3496 | 91.656 | 139.467 | 91.081 /90.857 | 138.077 /137.535 |
| v260 | SCC signed compare 输出 i1 | 461 /96 | 3616 | 93.889 | 142.109 | 92.523 /92.215 | 137.030 /137.875 |
| v261 | SCC bitcmp 输出 i1 | 461 /104 | 3620 | 93.168 | 142.075 | 92.112 /91.880 | 138.115 /137.188 |
| v262 | native Boolean signed row predicate | 461 /96 | 3628 | 93.141 | 140.837 | 90.878 /91.445 | 136.945 /136.698 |
| v263 | LLVM ICmp slt i1 / readfirstlane | 461 /96 | 3520 | 94.226 | 142.609 | 91.123 /90.155 | 138.004 /137.970 |

以上八版LDS均295936 B、private/spills0、Scale静态issue24/dwords48、data `ds_load_b128`192、WMMA120、CFG六区间通过；对应CPU/GPU完整ELF在汇总中逐项绑定。所有seed17完整E64→E65→E64八hash与指定baseline相同，原0.01门限及12288/13824有效routes、1536 shared routes保持，同一个实际artifact和完整ELF，没有E65 specialization。三输入variant、poison、eager/两graph各五次replay mismatch0；每版21个完整MoE边界八项mismatch0。所有原完整bench采用warmup5/iters20/count19，均未取得同批稳定收益，停止、不采纳、不扩三轮。

v260/v261源码把结果暴露为SCC/i1，但native仍出现`compare/bitcmp → s_cselect 1,0 → s_cmp → branch`，不能宣称消除了整数转换。v262/v263没有新inline asm；源码i1也不保证原生分支就更快。本轮未抓候选ATT/counters，不唯一归因HBM、cache、occupancy或某条指令。

v263先前CFG审计失败来自工具把65519当无符号正offset；encoded low16=`0xffef`应sign extend为-17。依据本地CDNA5 ISA §5.5及§15.5 printed pp288–289，branch target为`PC_next + signed_offset*4`。独立`audit_earlybanks_split_isa_v2.py`用实际encoded字段核对signed目标，并以CFG拓扑检查真cycle：向较低PC跳转本身不等于cycle。真实v263六区间、八个backward layout edges均通过；真cycle、memory路径、离开区间、重复signal、错误encoded offset、缺完整DS retirement六负对照拒绝，v190/v255对照也通过。只修审计工具，旧失败和原候选保留，未修改或放宽kernel数值门限。v261首批边界命令漏`--version`、rc2，成功记录为`edges0_5_20261020_r1_args2.json`；screen仅更正验收文件名，原精度/性能函数不变。

### v264/v265：A/B data LDS base 的 tied identity

仅在v255 `calc_window_addr`给A或B data base套局部`_lds_vector_identity`，约束`=v,0`保证同一VGPR，不采用空asm `=v,v`当作copy。回滚helper调用及symbol后完整源码等于v255，protected函数AST相同；59392地址case/7602176 consumer dwords包含low32 wrap。模型使用常量和identity stub，只证明覆盖范围内地址相等，不作为完整native hazard或任意异步时序证明。

| 版本 | data base | VGPR / SGPR | LDS B | 静态指令 | GPU状态 |
|---|---|---:|---:|---:|---|
| v264 | 仅A | 463 /96 | 295936 | 3471 | 完整精度/replay/21边界及screen完成，不采纳 |
| v265 | 仅B | 461 /96 | 295936 | 3443 | 完整精度/replay/21边界及screen完成，不采纳 |
| v266 | A+B | 463 /96 | 295936 | 3444 | CPU/model/资源/CFG通过；按用户要求未运行GPU |

三版private/spills均0，v264/v265完整GPU验收与前节相同，E64/E65仍使用同一个actual artifact。v264完整ELF=`1006f66e25978e4efdec5914eafa9550a1664d24bf9d514a68e4a922745ebc2c`，v265=`d8b5be84157ba45539dbb2a25facd7ad33b189eced1ed0ca56225b4339044e5e`。生产仍为v190完整ELF=`24bb623573932faadb22ca2d89d3e84a903adc64c9e116cac4138dd5b686f93c`。

| 版本 / 对照 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 90.779 | 161.590 | 59.005 | 315.866 | 201.630 | 982.260 |
| v264 | 92.278 | 139.094 | 59.766 | 89.693 | 203.900 | 284.430 |
| v190 after | 92.272 | 136.806 | 59.157 | 90.412 | 203.810 | 281.200 |

v264的E65 before为异常记录，不作为稳定收益对照；候选E64与尾control持平且慢于首control，E65慢于正常尾control，不采纳。

| 版本 / 对照 | E64 GEMM1 μs | E65 GEMM1 μs | E64 GEMM2 μs | E65 GEMM2 μs | E64 MoE e2e μs | E65 MoE e2e μs |
|---|---:|---:|---:|---:|---:|---:|
| v190 before | 90.632 | 137.858 | 58.990 | 89.786 | 201.740 | 282.580 |
| v265 | 92.154 | 138.957 | 58.447 | 90.149 | 203.260 | 283.010 |
| v190 after | 90.804 | 136.968 | 59.134 | 89.822 | 203.380 | 282.500 |

v265两规模GEMM1均慢于首尾control，不采纳、不扩三轮。v265相对v255静态指令减少21条没有带来本测试的性能提升；不修改生产kernel、selector或用户benchmark。

### 用户指定的 E65/v190 异常重测

独立三次运行完全相同的原完整`--scenario bench --iters20`命令，使用fresh labels保留旧数据；同完整ELF、warmup5/count19、原sanity和0.01门限均保持。测量前全机idle，性能窗口内无本任务并发编译/正确性/idle polling/SFTP/归档。三次实际GEMM1 ELF与正式v190一致，E65原logits_diff均为`3.7793e-06`。

| E65/v190 | GEMM1 μs | GEMM2 μs | MoE e2e μs |
|---|---:|---:|---:|
| 旧异常记录（不作稳定对照） | 161.590 | 315.866 | 982.26 |
| 重测 1 | 138.887 | 89.750 | 284.02 |
| 重测 2 | 136.918 | 89.441 | 281.17 |
| 重测 3 | 136.931 | 89.287 | 281.12 |
| 三次重测中值 | **136.931** | **89.441** | **281.17** |

异常未在这三次复现，这是同代码复测结果，不计为代码优化收益，也不改写原正式E65中值137.596 μs。旧异常pre为03:08:29 UTC idle，03:08:50退出后出现`<已退出>`及身份不完整KFD记录，03:08:56有独立`jli-mi450`任务；这些采样不能排除未知短时重叠，也不能唯一认定干扰原因。保留异常原raw及全部pre/post/post6，不从较低复测数字声称修了kernel性能bug。

复测三次pre分别04:20:24/04:21:43/04:22:22 UTC idle。立即post无KFD但busy3；随后post/post6仍有其他完整bench、`dev_al /tmp/b32_work/pilot.py`、defunct等独立记录，均按六秒等待到idle，不杀他人进程。r1/r2/r3最后post6分别04:21:42/04:22:22/04:23:07 UTC idle；不把最后idle冒称首post就idle，也不据此证明整个窗口不存在未知瞬时重叠。当前sysfs只暴露`card1/device`一张物理加速卡，不写成四张物理卡测试。

### 证据、复现和暂停边界

本批1048项/95480350 B source、CPU/ISA/完整ELF、完整MoE/replay/210边界、原screen及异常三次复测、每个control actual MLIR和当时helpers已按五shard下载，逐archive/member SHA256及所有实际ELF独立核验，不含`.out`。manifest=`logs/runtime256_v266_completed_pause_v2_20261020.sha256`，SHA256=`94210539379d9e743f682e8dab0d85c92c340e4168d254e5726f60206a239b0d`；[本地独立汇总](../.codex_tmp/gemm1_t1536_20261017/logs/runtime256_v266_completed_pause_downloaded_summary_20261020.json) SHA256=`a80ef4c7e52c3e4a2dfb5e9d668866db8a6d373a0fbec484895946edd3acc86a`。首版归档误纳入仍写入中的freeze输出，SHA门禁拒绝；独立v2排除live包装日志、保留已完成失败记录后重新冻结，不改任何GPU结果或覆盖旧raw。

以下host入口复现已完成实验；使用fresh label/suffix，保留CPU/model/完整数值门禁，并且每项测试120秒上限。各版screen绑定本次已通过的验收labels；若重做CPU/数值，用新labels同步修改对应绑定，不能覆盖旧数据。

```bash
cd /data/yanguahe/code/wk_sp1/aiter
task=.codex_tmp/gemm1_t1536_20261017
# v254/v255 prepared-body screen; v254 also completed three paired rounds.
bash "$task/run_scale_prepared254_screen_host.sh" repro_02
bash "$task/run_scale_prepared254_pairs_host.sh" repro_02
bash "$task/run_scale_prepared255_screen_host.sh" repro_02
# v256/v257 use corrected namespaces; v261 first-edge label needs args2.
bash "$task/run_scale_prepared256_screen_v2_host.sh" repro_02
bash "$task/run_scale_prepared257_screen_v2_host.sh" repro_02
bash "$task/run_scale_prepared261_screen_args2_host.sh" repro_02
# Versions258/259/260/262/263/264/265 use their own unmodified screen.
bash "$task/run_scale_prepared265_screen_host.sh" repro_02
# Same original E65/v190 full benchmark for the requested anomaly retest.
for r in 1 2 3; do
  bash "$task/run_runtime_expert_perf_host.sh" runtime190 65 \
    "optx_v190_capacity264_abnormal_e65_retest_repro_r${r}" || break
done
```

构建/kernel测试只在`dsr1_ep4`，Git只在host。本轮生产kernel、selector和原benchmark SHA仍分别为`f8be1c42936214a3728fcf8273d0a2b6279ab6731c69fe225730f0bd3b1763f5`、`db26b0d02a7e4c374b0b29f204e6ef371c390aa4324eb9801f4da50fa6d38a68`、`9a18ad0d61ea063f09c0f7b04c7ccbf9dcfa864ac68f681757a942865641b7d8`。本轮收尾后暂停；v266未GPU，新的优化、ATT/counters及原计划后续方案均未启动。

## v8 ATT 诊断数据

SIMD0–3均采集、分析成功；32个完整compute waves，每wave3584条WMMA，没有不完整wave。合计lifetime为6514922cycles。

| phase | elapsed cycles | lifetime占比 |
|---|---:|---:|
| prologue | 365667 | 5.6128% |
| compute | 5413664 | 83.0964% |
| handoff | 570884 | 8.7627% |
| final epilogue | 164707 | 2.5281% |

按指令归属累计stall，`s_wait_dscnt`占22.4375%（其中wait6占16.1593%），`s_barrier_wait`占14.6314%，`s_wait_tensorcnt`占14.2760%（其中wait0占14.1552%），WMMA占9.7905%，DS_load占0.4306%。分类是观测到的阻塞位置，不是各资源独立的可节省时间；不同依赖可能相互传递，不能直接把占比换算成μs收益。

原最优trace的handoff为605829cycles、compute为3300682cycles；v8分别为570884、5413664cycles。两个capture的lifetime与clock不同，因此不能只比较14.09%和8.76%就宣称handoff获得对应的μs提升。v8各capture kernel clock均值约1676–1750MHz，合并约1702MHz；最终性能仍用原benchmark衡量。证据为 `logs/optx_v8_att_analysis.json`、`logs/optx_v8_att_clock.log`，原始trace目录为 `my_code/thread_trace_runs/optx_v8_g4s4_att_20261018_r1`。

## 基准 handoff 分解

对原最优版已有 ATT 的 32 条完整 compute waves、96 个 task 边界做分解，总 wave lifetime 为 4299004 cycles。

| span | cycles 合计 | wave lifetime 占比 |
|---|---:|---:|
| epilogue compute/staging | 352638 | 8.2028% |
| publication + successor setup | 226342 | 5.2650% |
| successor READY | 2188 | 0.0509% |
| successor prime | 24661 | 0.5736% |
| handoff 总计 | 605829 | **14.0923%** |

landmarks 绑定原 ELF SHA256 `5f023d44fb65f0a38a8c22907775afeab049d92a0bad0698fec831901eb486ff`。publication 从 PC256 `0x1dbc s_wait_storecnt_dscnt 0` 开始，包含 PC259 barrier、PC261 output TDM 和后续 setup；READY 为 output 后首次 `tensorcnt1` 到 successor WG barrier 结束，prime 到首次 WMMA。publication 不是纯同步耗时，epilogue span 也可能包含 setup；四次 capture 独立，不能全局对齐，不能将占比直接换算为可节省 μs。

证据：`logs/optx_baseline_handoff_decomposition.json`，复现脚本 `decompose_optx_baseline_handoff.py`。G4 ring 模型及提前 refill、错误 outstanding、多代 output 复用等负对照记录在 `logs/optx_core_schedule_model.json`；模型不替代 GPU/ISA 正确性验证。

## 复现

代码本地开发后只复制任务所需文件至 `yanguahe@heliosr-1b114-a07-3.mnb.dcgpu:/data/yanguahe/code/wk_sp1/aiter`，核对 SHA256。构建和测试在 `dsr1_ep4` 内进行，Git 只在 host，保留所有其他未提交修改。

CPU 编译（无 GPU executor）：

```bash
env PYTHONPATH="$PWD" COMPILE_ONLY=1 ARCH=gfx1250 FLYDSL_GPU_ARCH=gfx1250 ENABLE_CK=0 \
timeout --signal=TERM --kill-after=5s 120s \
python3 -u .codex_tmp/gemm1_t1536_20261017/compile_optx_cpu.py --label core_v1
```

原 benchmark（先做全机 idle gate，测量时不并发其他测试或 idle 检查）：

```bash
env PYTHONPATH="$PWD" ENABLE_CK=0 AITER_MOE_EXPERT_BALANCE=false \
AITER_LOG_MORE=1 AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 AITER_GFX1250_GEMM1_OPTX=1 \
timeout --signal=TERM --kill-after=5s 120s \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache \
  --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --iters 20
```

设置 `AITER_GFX1250_GEMM1_OPTX=0` 复现原路径。未通过 CPU 资源和 GPU bitwise 门禁前，不将 optX benchmark 计为可交付性能。
