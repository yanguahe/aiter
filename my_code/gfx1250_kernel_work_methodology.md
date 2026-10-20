# gfx1250 kernel 开发与优化工作方法论

本文从 2026-09-07 至 2026-10-15 的工作历史中提炼通用的方法、流程和判据。覆盖的任务包括：纯汇编 MoE GEMM1 接入与优化、FlyDSL GEMM1/GEMM2 移植与调优、persistent kernel 时序 race 修复、token 区间 kernel 路由、四卡 EP MoE 接入与 thread trace 诊断，以及 MoE 框架文档整理。

文中的具体数值只用来说明方法，不作为当前性能结论。thread trace 的抓取命令和工具见同目录的 [gfx1250_thread_trace_capture_guide.md](gfx1250_thread_trace_capture_guide.md)。

历史来源（按继承顺序）：

```text
C:\Users\yanguahe\.codex\sessions\2026\09\07\rollout-2026-09-07T20-45-46-01a07be7-3b41-7d53-9cf8-9eb72323d943.jsonl   基础历史
C:\Users\yanguahe\.codex\sessions\2026\09\08\rollout-2026-09-08T22-23-18-01a08166-e38d-7303-ad9f-fab12d849d6d.jsonl   主体历史
C:\Users\yanguahe\.codex\sessions\2026\10\11\rollout-2026-10-11T23-24-58-01a12b91-3152-7ab3-852b-49232fc8c38d.jsonl   当前会话
C:\Users\yanguahe\.codex\sessions\2026\10\14\rollout-2026-10-14T13-04-16-01a138cc-029c-7952-93c1-ab8aa41dbe12.jsonl   分支会话
```

---

## 0. 核心原则

1. **正确性是性能数据有效的前提。** 先用 random 数据验证正确性，再测 const0 性能；最终以完整 e2e 单测（例如 MoE 算子单测）作为正确性和性能的口径。
2. **结论必须有证据。** 硬件主张以本地资料为准并注明章节或页码；明确区分「文档事实」「实测结果」「推断」；资料无法确定时直接说明未知。
3. **只做同条件比较。** 同一台机器、同一轮、相邻或交错 A/B、整机 GPU 空闲。跨重启、跨机器、跨 runtime 的绝对时间不可直接比较。
4. **先估上限，再动手。** 用 roofline 或结构下限判断收益空间，用 thread trace 定位真实瓶颈，按「收益上限 × 证据强度」排序候选方案。
5. **小步、隔离、可回滚。** 每个候选单独实现、单独验证；只保留稳定收益，其余代码及时删除，失败的原因和证据要留档。
6. **对「好得离谱」的结果保持怀疑。** 结果低于硬件下限或远快于预期时，先检查是否跳过了计算、跑错了代码或测错了对象。
7. **守住安全边界。** 破坏性操作逐次获得授权；不杀进程、不 reset GPU、不重启机器；Git 在主机执行，构建和测试在容器内执行；不覆盖其他 agent 或用户的修改。
8. **可复现、可追溯。** 记录命令、原始样本、中位数、SHA256、日志路径、机器、容器和 commit。汇报时先给结论，再写证据和限制。

---

## 1. 任务启动：先定义问题，再核对环境

### 1.1 明确目标、约束和交付物

开始前把以下内容固定下来，并在后续汇报中保持同一口径：

| 项目 | 需要明确的内容 | 历史中的例子 |
|---|---|---|
| 性能指标 | 用哪个计时器、哪个 kernel、哪个 workload | 以完整 MoE e2e profiler 中 GEMM1/GEMM2 的 `device_time_avg` 为准，standalone 结果只用于筛选 |
| 目标 | 相对哪个 baseline 改善多少 | 「GEMM1 耗时减少 20% 以上」，目标线由本轮 baseline 中位数计算 |
| 不可变约束 | 规模限制、数学功能、精度、其它规模不回退 | 「不能修改 kernel 对规模的限制、kernel 结果的精度、kernel 的功能」；优化 E96 时守护 E64/T1536 不回退 |
| 交付物 | 写入哪个文档、记录哪些字段 | 稳定收益要记录「优化方法、复现命令、性能数据」 |
| 执行位置 | 哪台机器、哪个容器、哪个目录 | a07-3 的 `hyg_fyd1`；四卡任务在 f01-1 的 `dsr1_ep4` |

用户在任务中途给出的指令优先于历史默认值，例如「只用某台机器」「尽量减少抓 trace 的次数」「日志已经跑好，不要再执行测试」「有进程在用 GPU 时连正确性也不要跑」。

### 1.2 确认「实际运行的是哪份代码、哪个 kernel」

很多「优化无效」或「结果不符合预期」最后都是环境问题。每次开始和每次出现异常时检查：

- 分支、HEAD、未提交修改；本地与远端关键文件的 SHA256 是否一致。
- Python 实际导入路径：

  ```bash
  python3 -c 'import aiter; print(aiter.__file__)'
  ```

  例：`dsr1_ep4` 的 `/etc/bash.bashrc` 写了 `export PYTHONPATH=/app/aiter:${PYTHONPATH}`，在 refactor 目录运行 benchmark 时实际导入的是 main 版本，kernel 名因此没有变化。

- 实际选中的 kernel：从日志里的完整 symbol 或 JSON 确认。profiler 汇总表可能截断 kernel 名，`grep` 的固定空格模式也可能漏掉表格。
- 配置是否命中：CSV 或路由表的 key 是否匹配目标规模，tile、cluster 是否是预期值。
- 编译缓存：旧缓存（如 `~/.flydsl/cache`）可能让测试和 trace 使用之前编译的 tile 配置。
- 编译参数是否一致：例：`hyg_fyd_e2e` 容器的性能回退根因是 optimized launcher 漏传外部 `compile_hints`（尤其是 `amdgpu-expert-scheduling-mode`），而不是最初怀疑的 FlyDSL 版本差异。
- 工具链版本：profiler、decoder、COMGR、LLVM 是否与容器 runtime 对齐。

### 1.3 先测 baseline，并建立「一键复现」入口

- baseline 必须在整机 GPU 空闲时测，至少 3 轮取中位数，同时保存原始样本、日志路径和源码 SHA256。
- 准备一个脚本，在同一次运行中复测 baseline、关键历史版本和当前候选，顺序交错，自动记录主机、HEAD、源码或 ISA 的 SHA256、时钟状态和完整日志。换机器或机器重启后，先运行它重新找到性能「起点」。
- 脚本依赖尽量自包含（例如固定的 repo snapshot），避免依赖会被其他人修改的工作树文件；跨平台使用时注意 LF/CRLF 会让 snapshot 校验失败。

### 1.4 先读历史，避免重复

- 先查阅已有的优化记录、失败实验清单和交接摘要，确认哪些方向已经被证伪。例：metadata hoist、descriptor cache、较小 tile 换 occupancy 等方向在历史中已试过且没有稳定收益。
- 历史中的一次性授权或限制只适用于当时的任务。例：早期「只能修改 `my_code/`」是隔离任务的要求，后来用户明确授权修改生产代码，不能机械沿用旧限制；同样，「某台机器可以忽略其他 GPU 进程直接测」的授权不能推广到其它机器。

---

## 2. 正确性验证体系

### 2.1 分层 oracle

| 层级 | 检查内容 | 作用 |
|---|---|---|
| 中间张量 | producer 输出的 payload、ScaleA 逐字节比较 | 定位布局、量化、stride 问题 |
| 单个 kernel | GEMM1/GEMM2 有效输出行 bitwise 比较或 hash | 定位具体 kernel 的错误 |
| 端到端 | `logits_diff`、`rel_l2` 与精度门限 | 最终验收 |

- 记录 hash 时写清比较的是哪个张量（GEMM1 payload、ScaleA、GEMM2 valid rows、最终 MoE 输出）以及哈希算法（历史中统一为 BLAKE2b-128，32 个小写 hex）。
- 输出 buffer 先填非零哨兵值（例如 BF16 `2.0`），防止「输出本来就是 0」造成假通过。

### 2.2 数据和覆盖范围

- random 数据用于发现布局、索引、同步错误；const0 适合确定性的性能测量与 hash 对比。const0 通过不代表 random 正确。
- 覆盖边界条件：tile 对齐和非对齐（partial M）、token 区间边界（例如 `1024/1025/1536/1537/2048/2049/4096/4097`）、balanced 与 non-balanced routing。
- 多卡和 EP 场景有特有输入，例如 padded source pitch（ScaleA 有效宽度 224B、实际行距 256B）、dropped route 的 `-1`、非本地 expert。这类边界通常可以用单卡、小规模的专用 oracle 覆盖，不必每次跑四卡大规模测试。
- 修复 bug 后回答「为什么之前的单测没有发现」，找到覆盖盲区并补上对应的 oracle。

### 2.3 非确定性错误（race）

- 少量 timed 运行通过不能证明正确。例：exact t192 的跨 N-task overlap 在 T1536 的 20 轮 stage audit 中 17 轮通过、3 轮失败，而 5 轮 timed random 恰好通过了整体精度门限。
- 用长时间、多 seed、多独立进程的 stage replay 做验收。历史中 race 修复的最终验收累计了数千次 stage replay，有效输出差异必须为 0。
- 错误位置在重复运行之间漂移，是跨 wave 同步竞态的典型信号，而不是数值计算或 layout 错误。

### 2.4 对「好得离谱」的结果做反向检查

- 实测 cycle 数低于硬件理论下限，说明要么测量有误，要么结果不对。例：为 const0 输入加入的 zero shortcut 跳过了全部 WMMA，ATT 测得 185,438 cycles，低于 516,096 cycles 的 WMMA 下限；该结果无效，代码已撤销。
- 验收时核对动态工作量：在 ATT 中确认每个有效 wave 实际执行的 WMMA、EXP、RCP 等指令数与理论一致。
- 不能为特定输入（例如全零数据）开捷径来达成性能目标。

---

## 3. 性能测量规范

### 3.1 统一口径

- 主指标：e2e profiler 中目标 kernel 的 `device_time_avg`；standalone 结果只用于快速筛选。
- 区分以下几种时间，不能混用：
  - profiler 中各 kernel 时间之和与 CUDA event 测得的 wall time；
  - ATT 中的 wave cycles 与 kernel 耗时（trace 会扰动调度和时钟，ATT 期间输出的耗时不能作为性能数据）；
  - 采样 wave 的跨度比例与正式 kernel 时间比例（例：ATT 中两版有效 wave 跨度比约 3.2，正式 kernel 时间比约 2.03）。
- 派生指标要定义清楚：effective R+W 是 useful bytes 除以时间，按十进制计算，它不是实测的 HBM 带宽。

### 3.2 GPU 空闲门禁

- 每次性能测试和 thread trace 采集前，检查整机所有 GPU 的 KFD 进程、利用率和 VRAM（优先运行 `/data/yanguahe/code/gpu_users.sh`，否则用 `fuser -v /dev/kfd` 与 `rocm-smi` 或 sysfs）。
- GPU 忙时每 6 秒自动重新检查，等待期间可以做静态分析、编译或正确性验证；不杀进程、不抢占 GPU。
- 性能测试运行期间不要并发执行空闲检查，以免干扰计时；测试结束后再次检查是否有残留进程。
- 如果结束检查发现其他进程出现，就无法排除中途有重叠，这组数据要从耗时对比中排除；raw 数据可以保留作非计时证据。

### 3.3 机器性能不稳定时的比较方法

- MI450 机器的性能会随时间、重启、配置变化而漂移，只关注同一轮内候选相对 baseline 的变化。
- SSH 失联恢复、机器重启或测量出现异常后，先复测 baseline 和关键历史版本，再继续测试候选。
- 采用相邻 A/B 或交错顺序，降低频率漂移的影响。

### 3.4 样本量与判定标准

| 场景 | 做法 |
|---|---|
| 候选初筛 | `ROUNDS=3`，与相邻 baseline 比较中位数 |
| 关键取舍 | 3 次命令 × `ROUNDS=3` 共 9 个样本取中位数。例：早期 3 个样本得出的「t256 `_vm192` 比 t192 快」被 9 样本复测推翻，T1536 默认路由随后改回 t192 |
| 不回退承诺或最终验收 | 长序列相邻 A/B（历史中用过 12×201、24×201 对），同时看 median 和 paired median |

- 收益小于约 1%，或各轮方向不一致，按噪声处理，不保留。
- 记录全部原始样本，不只记录中位数。

---

## 4. 性能分析方法

### 4.1 先算理论上限

1. 把各资源的吞吐折算到同一个时钟（历史中为简化统一按 GFXCLK）下的 cycle 下限：HBM、GL2/GL1、TDM 投递、LDS、WMMA/XDL、Trans/VALU。
2. 叠加结构约束：LDS 和 VGPR 决定每个 SIMD 能驻留几个 wave；串行阶段（例如单 slot 下 epilogue 无法被其他 wave 隐藏）；early-exit 和空 WG 的数量。
3. 用途：
   - 判断剩余优化空间；
   - 校验测量是否合理；
   - 判断目标在当前结构下是否可达。例：单 slot 结构下 WMMA 加 Trans 的下限 552,960 cycles 已经高于 1.05× 目标 541,900.8 cycles，必须改变结构才可能达标。
4. 指令吞吐、LDS 容量、VGPR 容量、TDM 规则等参数要查本地硬件资料，必要时与白皮书的峰值算力交叉验证（例：按 WMMA 8-cycle repeat rate 推出的峰值与白皮书 MXFP4 40.26 PFLOP/s 一致）。

### 4.2 对性能回退做受控拆解

- 每次只改变一个因素，使用相同数据和相同计时方法，逐步复现回退。例：565 us 回退到 710 us 被拆解为「4×4 拓扑与压缩输出 −64 us」「SiLU +75 us」「为正确性关闭 multicast +133 us」，从而确定最有价值的方向是恢复可靠的 multicast/barrier 协议。
- 不要一次叠加多个变化后再归因。例：一个失败版本同时改变了 warp geometry、persistent loop 和 cluster，事后无法确定是哪一项造成回退。

### 4.3 静态分析

- 从 code object 读取 LDS、VGPR、SGPR、spill、private segment；统计静态指令数，并与参考实现（例如 production ISA dump）做 diff。
- 资源数字不等于实际 occupancy，需要时用 trace 中的 occupancy 数据确认。
- 静态证据只能提出假设，不能代替动态 trace 下结论。

### 4.4 thread trace 分析要点

- 按编译后 `code.json` 中的真实 PC 切分 prologue、hotloop、epilogue，不要使用汇编注释中可能已经偏移的旧地址。
- 区分 exposed stall（wave 因依赖未满足而真正停住的时间）和 latency（另含指令自身的执行开销）。判断瓶颈和优化空间时优先看 exposed stall。
- 只统计执行了 WMMA 的有效 wave，排除 empty expert、capacity tail、early-exit 等无效 wave。
- 尽量采全 SIMD0–3。只采一个 SIMD 可能得出偏差结论。例：只看 SIMD3 得到 `s_barrier_wait` exposed stall 30.29%，四个 SIMD 加权后是 24.54%。
- 观察物理 WGP 之间的完成时间不均衡度，判断负载均衡方向是否值得投入（历史中多次测得中位数低于 1%，因此没有优先做负载重排）。
- 区分「逻辑 wave 编号」「trace 文件中的 `wv`」「resident slot」「物理 SIMD」，它们之间的映射必须由 trace 证实，不能互相替代。
- 等待被「转移」不等于被消除：某项改动使显式 wait 下降，但整体 span 可能不变甚至变长。例：改用 `global_store_async_from_lds_b128` 后显式 epilogue stall 从 16.06% 降到 2.60%，但等待转移到 `s_endpgm` 隐含的 wait-idle，kernel 反而从约 59 us 变慢到约 73 us。

### 4.5 动手前估算理想收益

用 trace 数据估算候选方案的收益上限，再决定是否投入。例：对 output LDS 双缓冲版本，persistent mode 只消除调度和重建开销时理想收益约 2.3%；若还能把上一 tile 的 output TDM drain 完全隐藏在下一 tile 的准备阶段，上限约 8.3%。收益上限太低的方向直接放弃或降低优先级。

### 4.6 用隔离诊断建立因果关系

- 构造「保留指令流和同步，只改变一个因素」的诊断副本（数值结果可以故意错误），在同一台机器上对比。四卡 GEMM2 诊断中用过：
  - 拆开组合 wait，区分 LOAD 与 DS 等待；
  - 不再读取 window metadata；
  - 输出改写到本卡独立 buffer；
  - 输出行越界，跳过实际写入。
- 诊断副本只能通过显式开关启用，不能进入生产。记录它额外改变了哪些条件（例如 VA 布局、stride），据此限定结论范围。
- 结论分级表述：「trace 已证实」「隔离对照支持」「仍是机制推断」。不能唯一确定具体硬件环节时明确说明。

### 4.7 使用硬件资料

- 引用到文档、章节和页码。
- 资料之间有冲突时明确报告，并设计同时满足各方约束的实验。例：gather indices 的顺序要求在 Shader Programming Guide、Tensor DMA 资料与 CDNA5 ISA 之间存在冲突，诊断时改用严格递增的 OOB indices。

---

## 5. 优化迭代策略

### 5.1 选择候选方案

- 依据三类信息：trace 中占比最大的 exposed stall、硬件容量计算、历史失败清单。
- 优先实施「上限高、证据强」的结构性方案，不要为每个小参数调整都抓 trace。用户的明确要求是：「尽量减少抓 trace 的次数，重点放在分析 trace，然后根据 gfx1250 的硬件资料制定针对性的优化方案，首先实施可能大幅提升 kernel 性能的优化方案」。
- 可以借鉴其他 kernel 或其他分支已验证有效的方法（例如把 GEMM2 的 persistent、w2x4、TDM owner 均衡移植到 GEMM1），但要按目标 kernel 的实际情况调整并重新验证。

### 5.2 单个候选的标准流程

1. **隔离实现**：独立文件、环境变量开关或 compile-time specialization；生产代码保持可回退（保留备份或使用 `git restore`）。
2. **静态检查**：语法、lint、diff；只上传任务相关文件，核对 SHA256。
3. **正确性**：random 严格验证；按需要补 non-balanced、边界值、重复运行。
4. **性能**：整机空闲后，与相邻 baseline 做 3 轮 const0 对比。
5. **判定**：
   - 有稳定收益：保留，并写入优化记录（方法、命令、样本、中位数、正确性）；
   - 无收益或回退：如果结果明显不符合预期，抓一次 trace 解释原因；然后立即删除代码，在记录中保留失败原因。
6. **叠加后重新建立 baseline**：多个收益叠加后，重新测量组合版本，作为下一轮的起点。

### 5.3 先算资源，再决定是否走某条路线

- 先计算 LDS（320 KiB/WGP）、VGPR（1024/SIMD）、TDM inflight 建议值等资源，以及预期 occupancy。资源上达不到设想时直接停止，不必抓 trace。例：为提高 occupancy 改用 t128，资源上可以驻留 2 个 WG，但每个 expert 从 192 行补齐到 256 行，多出约 33% 的无效 WMMA，最终明显变慢。
- 历史中失败较多的方向：小 tile 换 occupancy、增加 wave 数导致 VGPR 过高、过早发射 future TDM 造成 queue backpressure、增加 ring 深度带来的 SALU/LDS 和代码体积开销抵消收益。

### 5.4 保护其他规模

- 改动只影响目标规模：使用精确 key 的配置（例如 CSV 精确行）或受限的 profile 条件。
- 列出需要守护的规模，每次保留候选前复测，确认不回退。

### 5.5 路由和配置选择

- 对每个边界值，测试所有候选 kernel 的 random 正确性；在结果正确的候选中选择 const0 性能最优者。
- 用「区间 + 左闭右开」表达路由，相邻区间方案相同时合并，代码尽量通用，减少写死数值的 `if`。
- 术语保持一致（例如 persistent 与 persistent full 是否同义），最终配置表写进文档，并标注 cluster 等 launch 参数。

### 5.6 停止与收尾

用户要求暂停或结束时：完成手头实验 → 恢复性能最好的版本 → 删除无稳定收益的代码 → 复测最优版本并写入文档 → 核对本地与远端 SHA256 → 确认没有残留的 GPU 进程。

---

## 6. Bug 定位与修复

1. **复现并缩小范围**：先判断是确定性错误还是非确定性竞态（错误位置是否漂移）。
2. **先排除环境问题**：导入路径、缓存、配置未命中、编译参数漏传、LF/CRLF 导致的校验失败。
3. **race 分析**：
   - Tensor 操作只在同一 wave 内按序完成，跨 wave、相对其他 memory 类型都无序（Shader Programming Guide §4.10.1，印刷页 197）；
   - 检查 LDS 区域的生命周期和 release 协议，确认没有 wave 在其他 wave 读完之前覆盖同一块 LDS；
   - 在生成的 ISA 中确认 fence/barrier 的 PC 早于第一条会覆盖该区域的 store。
4. **准确定性**：区分「原有实现 bug」「新输入或新场景需要补充的支持」「更换配置后的匹配调整」。例：四卡接入中的三项修改分别属于 padded stride 支持、helper 边界处理缺失和 tile 配置匹配，不应统称为「四卡规模触发的 bug」。发现之前的表述不准确时直接更正。
5. **修复时保留原设计意图**：修 race 不等于简单加大同步范围。例：修复后仍保留下一 N-task input prefetch 与上一 output TDM 的 overlap，只是把 output 放到已经释放的 LDS 区域。
6. **修复后的验收**：长时间 stage replay、补充 oracle；如果修复带来性能成本，再做性能补偿，并完成「不回退」验收。

---

## 7. 远程执行与协作规范

- **长连接**：SSH PTY（`ssh -tt -o ServerAliveInterval=30 -o ServerAliveCountMax=3 <user>@<host>`）、长期 SFTP 和 tmux；只有会话失效时才重新连接。
- **执行边界**：Git 只在主机执行；构建和 kernel 测试在指定容器内执行；容器生成的文件在容器内 `chmod -R a+rwX <path>`。`docker exec` 中使用绝对脚本路径，因为容器默认 cwd 不同。
- **破坏性同步门禁**：每台机器、每个工作轮次执行 `sudo git clean -fd` 或 `git checkout .` 之前都要获得明确批准；普通的「去 X 上测试」不构成批准。
- **传输**：只传任务所需文件，必要时比较 SHA256；不批量复制工作区，不批量删除 untracked 文件（历史中发生过误删其他 agent 目录）。
- **与其他 agent 共享工作树**：被别人修改的文件用 `git show HEAD:<path>` 读取；不覆盖他人修改；实验放在独立目录、独立 worktree 或独立容器中。
- **备用机器**：原机器长时间被占用时，可以在其他机器上建独立目录或独立容器做补充诊断，但必须标明机器和 runtime 版本，只在同一台机器内比较。不同机器上的 `/data/...` 不一定是同一个文件系统。
- **机器故障**：机器 hang 或失联时停止 GPU 操作并报告用户；不自行 reset GPU 或重启机器；恢复后先重建 baseline。
- **Git 提交**：commit/push 只在用户明确要求时执行。用户要求「只提交已经 git add 的内容、不要修改文件」时严格照做，commit message 用英文。

---

## 8. 文档、汇报与证据

- 每项稳定优化记录：方法、复现命令（包含全部环境变量）、原始样本与中位数、正确性数据、机器/容器/commit/SHA256、日志路径。
- 汇报先给结论，再给证据和限制；区分已证实的结论与推断；写清数据口径。
- 用户要求「回答写进某个文档」后，后续每个问题的回答都持续写入；修改前先读取最新版本，保留他人新增的内容。
- 证据归档：打包 raw 数据、manifest 和 SHA256，下载后校验；文档中的链接指向真实存在的路径。
- 只使用已有数据回答时（例如用户要求「不要再执行测试」），说明数据来源和日志位置，不重新运行测试。

---

## 9. 常见陷阱

| 现象 | 常见原因 | 处理方法 |
|---|---|---|
| kernel 名与预期不一致 | 导入了其他仓库、命中旧缓存、配置未命中 | 检查 `aiter.__file__`、完整 symbol、配置 key，清理缓存 |
| 性能和之前的记录相差很多 | 机器状态变化、比较的不是同一个 kernel 或同一口径 | 同轮复测 baseline 与历史版本，统一口径 |
| cycle 数低于理论下限 | 跳过了计算或测错对象 | 核对动态指令数和输出正确性 |
| 少量运行通过、偶发错误 | 跨 wave race | 长时间 stage audit，检查 LDS 生命周期 |
| 显式 wait 减少但没有变快 | 等待转移到 barrier、其他 PC 或 `s_endpgm` 的隐含等待 | 看整体 wave span 和等待转移的位置 |
| 降低资源占用想提高 occupancy，结果变慢 | occupancy 实际未达成、增加了无效计算、失去了预取 | 先计算资源与 padding 开销 |
| 汇总表看不全 | 表格截断名字、`grep` 模式固定空格 | 读完整 JSON；用 `grep -E 'kernel[[:space:]]+avg_us'` |
| 带宽数字口径混乱 | 混用 1.024 因子、MB/MiB | 统一十进制，并注明是 useful bytes |

---

## 10. 检查清单

### 开始性能测试前

- [ ] 分支、HEAD、关键文件 SHA256 与预期一致，导入路径正确
- [ ] 已确认目标 kernel 的完整 symbol
- [ ] random 正确性已通过
- [ ] 整机 GPU 空闲
- [ ] 本轮 baseline 已测量

### 宣布完成前

- [ ] e2e 正确性（random 与 const0）通过；存在非确定性风险时已完成长测
- [ ] 相邻 A/B 显示稳定收益，守护的其他规模没有回退
- [ ] 无收益的代码已删除；本地与远端版本一致
- [ ] 方法、命令、数据和限制已写入文档
- [ ] 测试结束后没有残留进程
