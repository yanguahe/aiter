# gfx1250 MoE GEMM1 optX 优化记录

## 任务与状态

目标是 E64/T1536/topk8、random non-balanced routing 下 GEMM1 稳定低于 80 μs。shared expert 单独计算；有效行 canonical FP4 payload、ScaleA、GEMM2 和 MoE 输出保持 bitwise 一致。允许跳过无效 padding 计算，禁止更改计时、使用 const0 特例、反转 K 或放宽精度阈值。

根据用户最新指令，先完成手头实验，然后暂停原优化路线，实施 `C:/Users/yanguahe/.cursor/plans/gemm1_optx_优化方案_910c94ab.plan.md`。optX 深度调优后若仍未达标，再恢复原路线。该指令优先于计划文件中的停止规则。

当前没有 optX 候选被采纳。原 persistent 路径历史正式中值为94.888 μs，本轮全机空闲复测中值94.279 μs，距离80 μs还差14.279 μs。新测G2 B/ScaleB寄存器复用v40为108.140 μs，G1较小cache v41为108.888 μs；均通过random/replay，但仍慢于原路径。padding skip v35为116.409 μs，S5 skip v38为140.753 μs，S5 rolled F stagger v39为165.346 μs。v35 ATT确认减少21.09375%的WMMA，但barrier/LDS等待仍显著。开发期间 `AITER_GFX1250_GEMM1_OPTX` 默认为0，原persistent源码保持冻结；当前实验module为v41，下一步继续验证G1跨task epilogue stagger v42。文中早期“正在实施”“尚无数据”描述的是对应阶段；最新结果见后续小节。

## 基准冻结与前序实验收尾

原 kernel SHA256：

```text
b36f7ec94524a1cb0bf20c5152f2a82d630fed71c0db2c1b2559d68f9a43e790
```

原 HEAD 为 `339589f900b3a880ec447f605886fe803cbf9649`。正式历史最优三轮中值为 94.888 μs。2026-10-18 Asia/Shanghai 的当日空闲复测如下；环境变化带来的基准差不能当作代码优化收益。

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
