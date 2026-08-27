# gfx1250 Grouped-MoE A4W4 带宽优化方案 v2

> **禁用声明：本机 ATT/thread trace 无效；本方案不读取、不引用、不采集、不建议其输出或任何派生 CSV/cycle。**

<!-- markdownlint-disable MD013 -->

## 0. 定位、目标与证据标签

本文是 `post-execution next-round plan`。它独立于旧方案，专门用于下一轮
gfx1250 grouped-MoE A4W4 GEMM1/GEMM2 的 memory-feed 与 wall-time 优化，不重复已经
完成或失败的工作。

目标只有两个：

1. 降低 GEMM1/GEMM2 以及 E2E wall time；
2. 在不制造无效流量的前提下，提高有效物理 memory-feed。

本文统一使用以下标签：

- **[当前代码事实]**：由当前本地源码或配置直接确认。
- **[执行记录事实]**：来自已核准执行记录
  `C:\Users\yanguahe\Documents\code\wk_sp1\cc099ad7-e0e8-4745-a9a0-1d4e95b7b83e.jsonl`。
- **[硬件文档事实]**：来自本文第 15 节列出的 gfx1250/MI400/CDNA5 资料。
- **[推断]**：由代码几何和硬件文档推导，尚未得到目标机器计数器验证。
- **[必须实测]**：必须由同一环境的 wall time、kernel trace/stats 或目标 counter catalog/PMC
  证实。

任何未被当前源码、执行记录或目标机器测量确认的物理行为都写为 `UNKNOWN`。

## 1. 当前仓库与已核准执行状态

### 1.1 Git 与远端边界

- [执行记录事实] 当前本地 branch：`hyg_gfx1250_gemm_a4w4`。
- [执行记录事实] 当前本地 HEAD：
  `b0c6ad78a24e9efac79738d0f4cc020fc20f2bb1`。
- [执行记录事实] 真实远端状态：`UNKNOWN`。下一轮不得根据本地 remote-tracking ref
  推断真实远端 HEAD。
- 第一项执行动作必须重新输出 branch、完整 HEAD、`git status --short`；若与上面不符，
  停止并重新建立基线，不能套用本文性能值。

### 1.2 当前 retained 配置

[当前代码事实] `aiter/configs/tuned_grouped_fmoe.csv:1,73` 的目标行是：

- shape：tokens=512，model_dim=7168，inter_dim=3072，experts=96，topk=6；
- 两阶段均为 `tile_m=32, tile_n=256, tile_k=256`；
- 两阶段均为 `m_warp=1, n_warp=4, num_buffers=2`；
- `cluster_n=-1`，launcher 归一化为 cluster1；
- `waves_per_tensor_tdm=4`，即 wpt4；
- `next_stage_prefetch=1`，但
  `mxfp4_preshuffle_gfx1250_tdm.py:111-112` 只有在 `num_buffers>=3` 时才令
  `next_stage_on=1`，所以当前 b2 下实际为 **off**；
- [当前代码事实] `grouped_moe_gfx1250.py:1030-1051` 从 CSV 读取上述 specialization；
  `grouped_moe_gfx1250.py:1062-1107` 只为 tile/buffer/warp-grid 提供环境覆盖。

[当前代码事实] route-indexed A2 quant 已保留在
`aiter/ops/flydsl/grouped_moe_gfx1250.py:660-675`：
`topids_to_rows=topids_to_rows, source_topk=0, num_valid_routes=_ep_nvr`。它不再是下一轮
开发项，只作为 E2E 辅助优化保留并做回归。

[当前代码事实] balanced case 有 `512*6/96=32` 个 active rows/expert，恰好填满 M32。
因此旧的 “M64 partial-M16” 机制不适用于当前 balanced 主场景。

### 1.3 当前应出现的 kernel symbol

- GEMM1：
  `a8w4_tdm_fp4_t32x256x256_w1x4_b2_K7168_e96_act1_wpt4`
- GEMM2：
  `a8w4_tdm_fp4_t32x256x256_w1x4_b2_K3072_e96_wpt4`

[当前代码事实] symbol 拼接入口是
`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py:186-204`。只有
`num_waves_per_tensor_tdm != 2` 才附加 `_wptN`；实际 next-stage off 时无 `_prefetch`。

[必须实测] 当前没有 M32/wpt4 final ISA/resource artifact。symbol、VGPR、SGPR、LDS、
scratch、block size、grid、waves/WG 和 inferred residency 均必须由 current-HEAD kernel
trace/stats 或当前 code object 确认，不能沿用旧 M64/wpt2 artifact。

### 1.4 已核准 timing；不能跨状态直接比较

| 状态 | E2E | GEMM1 | GEMM2 | A2 quant | 适用边界 |
| --- | ---: | ---: | ---: | ---: | --- |
| 初始 M64/wpt2，五独立进程中位数约值 | 237.67 us | 128.10 us | 72.90 us | 10.60 us | [执行记录事实] 旧基线 |
| retained P1 + M32 + wpt4 记录约值 | 230.50 us | 128.30 us | 69.40 us | 未作为本文基线 | [执行记录事实] 组合状态 |
| 当前 HEAD、默认环境、五独立进程 | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | `UNKNOWN` | P0 必须重测 |

组合状态中 GEMM2 下降约 3.5 us，但 GEMM1 基本持平；因为 route quant、M32 和 wpt4
同时存在，该记录不能证明 wpt4 的独立收益。下一轮必须先做 P1 真正单变量实验。

## 2. 带宽口径纠正

### 2.1 四种不同概念

1. **logical requested rate**
   - kernel 按 tile/descriptor 逻辑请求的 byte 数除以时间；
   - 会包含因 tile 重复、padding、cluster 或实现方式产生的逻辑请求；
   - 19.8/17.4 TB/s 属于此类 tile-level logical requested read/time；
   - 不能称为物理 HBM 带宽或 HBM 利用率。
2. **unique-logical rate**
   - 每个 valid activation row 与 active-expert 的 B/scale surface 只计一次；
   - 用于比较算法必须搬运的有效数据量，不随 M tile、cluster 或 multicast 重复计数；
   - `op_tests/test_flydsl_grouped_gemm_gfx1250.py:430-478` 是 CPU-only arithmetic fixture：
     GEMM1 用 2,257,747,968 B / 144.7 us = 15.603 TB/s，GEMM2 用
     1,128,038,400 B / 76.1 us = 14.823 TB/s；
   - 144.7/76.1 us 不来自执行 JSONL，所以这两个数也不是目标机物理 HBM 数据。
3. **physical counter rate**
   - 目标机器计数器报告的实际 DRAM/EA 读写 byte 数除以严格匹配的 kernel 时间；
   - 必须分 read/write，说明聚合范围、partition、计数器实例数、单位和 replay；
   - 当前 GEMM1/GEMM2 的 HBM/GL2/GL1/TDM/LDS 物理值全部为 `UNKNOWN`。
4. **same-pattern sustained roof**
   - 同 partition、clock、stepping、地址分布、读写 mix、请求尺寸/对齐、并发度下，
     相同访问模式 microbench 的可持续物理带宽；
   - 这是判断 kernel memory-feed 是否接近上限的首选分母，不是产品宣传峰值。

### 2.2 23.3 TB/s 的适用边界

[硬件文档事实] `amd-cdna5-whitepaper.txt` 的 MI455X 产品级规格是 432 GB HBM4、
up to 23.3 TB/s，并支持 1/2/4/8 partition；这只是整卡峰值规格。

[推断] 当前目标机器可能受 partition、memory clock、B0 stepping、harvesting、温度/功耗、
请求 mix、地址 hash 和并发度影响。没有目标机信息时，不能用
`logical_or_unique_rate / 23.3` 得到 HBM 利用率，也不能断言 15--17 TB/s “低”。

判断公式必须写成：

```text
physical efficiency = measured physical bytes / kernel_time / same-pattern sustained roof
```

若分子或分母缺失，结果一律 `UNKNOWN`。wall-time 改善仍然有效，但不能被解释成物理带宽改善。

### 2.3 防止“假带宽优化”的守门条件

- logical requested bytes 可以作为 descriptor 工作量模型，但不能代替 physical bytes。
- unique-logical bytes 应在同 shape、routing 和 correctness 下保持不变。
- physical bytes 相对 baseline 无解释增加超过 1%：立即回滚。
- 即使 TB/s 数字变大，只要 wall time 未达到门槛，或增加来自 padding/replay/无效 expert，
  均 `REJECT`。
- 允许的理想结果是：physical bytes 不变或下降，wall time 下降，same-pattern roof
  利用率上升。

## 3. 当前配置、已实施与已失败矩阵

| 类别 | 配置/工作 | 状态 | 下一轮动作 |
| --- | --- | --- | --- |
| retained | route-indexed A2 quant | 已实现；E2E 辅助项 | 只做 correctness/timing 回归，不再开发 |
| retained | 两阶段 M32/N256/K256/w1x4/b2 | CSV row73 当前值 | P0 按默认环境重建基线 |
| retained | wpt4/cluster1/next-stage 实际 off | 当前 specialization | P1 与 wpt2 单变量对照 |
| artifact | M32/wpt4 final ISA/resources | `UNKNOWN` | P0 kernel trace/stats 确认 |
| physical | HBM/GL2/GL1/TDM/LDS/residency | `UNKNOWN` | P0 catalog/PMC；不可得则保持 `UNKNOWN` |
| failed | M64 上下的 shared N128/N512 | 已失败 | P7 条件满足前禁止原样重复 |
| failed | cluster2/3/4 | 已失败 | P9 有新证据才重开 |
| failed | mixed M64/M32 | 已失败 | 禁止原样重复 |
| failed | wpt1@M64 | 已失败 | P1 只作 M32 负对照，不作主候选 |
| failed | b3 + next-stage-on | 已失败 | 禁止原样重复 |
| untested | pure b3/off | 未测 | 不抢在 P0--P3 前；若测试必须显式令 off |
| obsolete | M64 partial-M16 | M32 balanced 恰好 32 rows 后不适用 | 不开发 |

## 4. 全局实验与验收协议

### 4.1 统计协议

- 每个候选至少 5 个独立进程；不能只在一个长进程内重复。
- 使用随机 activation、weight、scale 和 routing；固定可记录的 seed 集合。
- warmup、iters、graph 模式、CPU/GPU affinity、可见 GPU、容器、进程数全部固定。
- `run_gemm_a4w4.sh:14-25` 当前主场景为 96 experts、512 tokens、topk6、iters100、
  model_dim7168、inter_dim3072、Silu、no-bias；任何偏离必须单列。
- 每个进程报告 E2E、GEMM1、GEMM2、A2 quant；最终报告 median、mean、min/max、CV。
- 同时报告 memory/graphics clock、partition、stepping、温度/功耗状态；不可得写 `UNKNOWN`。
- 低代码改动门槛：`>= max(2%, 3*CV)`。
- kernel 源码改动门槛：GEMM1 或 GEMM2 主目标 `>=3%`，且 E2E 不退化。
- layout/离线格式改动门槛：目标 kernel `>=5%`，且转换/缓存成本纳入 E2E。
- G1/G2 单独判定；允许只保留对其中一个 stage 有效的 specialization。

### 4.2 correctness 协议

每个候选必须覆盖：

1. balanced：96 experts，每 expert 32 rows；
2. unbalanced 随机 routing；
3. skew：少数 hot experts；
4. empty expert；
5. 边界 active rows：1、15、16、17、31、32、33；
6. 随机有限值以及 NaN/Inf 输入传播/清洗语义；
7. candidate-vs-current-HEAD baseline 逐 stage 与 E2E 输出比较；
8. 重复运行确定性、越界、未初始化 padding、race 检查。

使用现有测试的生产 gate 语义
`op_tests/test_flydsl_grouped_gemm_gfx1250.py:116-119`，同时保存更严格的误差分布；
不能只以一次 balanced 输出通过作为正确性证据。

### 4.3 每项统一 PMC 语义

先从目标工具的 **实际 counter catalog** 生成映射表，记录实例聚合与单位；本文资料中的
候选名不能被假定为目标机可用名。

优先语义：

- EA/DRAM read bytes、write bytes、read/write request count；
- 每 HBM/EA/GL2/GL1 channel 的 bytes 或 requests，用于 CV/热点判断；
- 128B/256B direct request、smaller/indirect request 的数量或等价语义；
- GL2 request/fill/hit/miss bytes，GL1 request/return bytes；
- TDM issued/completed tensor op、request-size distribution；
- LDS read/write bytes 或 bank-conflict 等价事件；
- active waves/WGP、resident WG、kernel cycles 与有效 clock。

资料中的 `GC_EA_SE_SARB_DRAM_RD_SIZE_REQ`（文档公式为乘 32 得 bytes）、
`GC_EA_SE_CMD_POP`、`GL2A_BUSY`、`GL1A_REQUEST_GL1Cx`、
`TX_TENSORCNT_*` 等仅是 **catalog 搜索候选**。目标 catalog 若没有精确对应项，不得用名字
相似的 counter 冒充。若 physical PMC 不可得，则所有物理结论保持 `UNKNOWN`，决策只使用
严格单变量 wall-time 与 correctness。

## 5. P0：同环境校准与 current-HEAD 基线

P0 完成前，不允许改 kernel 或 layout。

### 5.1 当前源码入口/参数

- runner：`my_code/run_gemm_a4w4.sh:1-25`；
- dispatch/config：
  `aiter/ops/flydsl/grouped_moe_gfx1250.py:1024-1108`；
- launcher：
  `aiter/ops/flydsl/grouped_gemm_mxfp4.py:17-60,63-150`；
- kernel：
  `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py:52-204`。

### 5.2 严格执行顺序

1. 确认 branch/完整 HEAD/clean status；打印所有 `AITER_*`、`FLYDSL_*`、GPU visibility。
2. 清空会污染 specialization 判断的环境变量，按当前 HEAD/default config 跑 5 独立进程；
   先得到 **无 profiler** timing baseline。
3. 仅做 kernel trace/stats，确认两个 exact symbol、launch 次数、duration、VGPR/SGPR/LDS、
   scratch、block/grid 与可推导 residency；不得沿用旧 artifact。
4. 枚举目标 counter catalog；输出“资料候选名 -> 目标实际名/不可用”的映射。
5. counter 可用时分组做最小 PMC，避免同时启用过多事件改变 timing；counter 不可用就写
   `UNKNOWN`，不得阻塞 wall-time 因果实验。
6. 建立三类同环境 roof：
   - pure read：连续 256B 对齐，覆盖足够大工作集；
   - read+write：匹配 GEMM epilogue 的 read/write mix；
   - B-like TDM-to-LDS：匹配 M32/N256/K256 的 B/SB stride、tile、owner 数和并发度。
7. roof 至少扫 128B/256B 对齐、请求尺寸、工作集、active WG/WGP 与 read/write mix；
   报告平台区而不是单个峰值。

### 5.3 成功、物理 bytes 与停止条件

- wall-time 门槛：P0 不作性能 KEEP，只要求五进程 CV 足以支持后续门槛。
- physical bytes：基线分别报告 logical、unique-logical、physical；缺失字段写 `UNKNOWN`。
- correctness：current-HEAD 主场景及第 4.2 节矩阵通过。
- 停止：symbol 与预期不符、默认环境不干净、资源无法归属到 exact symbol，立即停止后续
  A/B。
- P0 产物：baseline JSON/CSV、kernel stats、counter mapping、三类 pattern roof、
  clock/partition 元数据。

## 6. P1：M32 固定下 wpt2 与 wpt4 真正单变量

### 6.1 当前源码与 specialization 陷阱

- 参数读取：
  `grouped_moe_gfx1250.py:1045-1050`；
- 选择函数：
  `grouped_gemm_mxfp4.py:44-60`；
- owner/TDM_PER：
  `mxfp4_preshuffle_gfx1250_tdm.py:347-372`；
- symbol/cache tag：
  `mxfp4_preshuffle_gfx1250_tdm.py:113-132,186-204`。

[当前代码事实] CSV row73 已写死 `waves_per_tensor_tdm=4`。
`_select_num_waves_per_tensor_tdm()` 在 CSV 值属于 1/2/4 时直接返回，因此仅设置
`AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM=2` **不会改变 specialization**。

保证单变量的方法：

1. 最稳妥：临时实验 patch 只把目标 row 的 `waves_per_tensor_tdm` 改为 2；其余字段 byte-for-byte
   相同；
2. 或增加一个明确的 present-check override，但该 patch 自身先做单元验证；
3. 每次删除/隔离 compile cache，kernel trace 必须看到 wpt2 symbol **没有** `_wpt4`，
   wpt4 symbol必须带 `_wpt4`；
4. 保存 exact CSV diff、env、symbol、resource。只看 env 值不能作为 specialization 证据。

### 6.2 机制与静态风险

- [硬件文档事实] 每 wave 最多 3 个 issue-to-XACK tensor op，每 SIMD pair 最多 6 个。
- [当前代码事实] M32/w1x4 有 4 waves；wpt4 令 `TDM_PER=4`，A/B/SA/SB 四个
  descriptor owner group 均覆盖四个 waves。
- [推断] 每 wave 的四 descriptor 可能产生 backpressure；这是风险假设，不是已发生事实。
- wpt2 令每个 operand 由两 waves 分担，每 wave 总计两个 descriptor；它可能降低 descriptor
  压力，但也可能减少单 operand 并行度。

### 6.3 最小矩阵与验收

| 变体 | M/N/K | buffer | owner | 目的 |
| --- | --- | --- | --- | --- |
| A | 32/256/256 | b2 | wpt4 | current baseline |
| B | 32/256/256 | b2 | wpt2 | 主单变量 |
| C | 32/256/256 | b2 | wpt1 | 仅负对照；不进入默认候选 |

- G1/G2 差异：每 tile 几何相同，G1 有 7168/256=28 个 K stage，G2 有 12 个；
  descriptor 改善更可能在 G1 累积，但 G2 的固定开销占比更大，必须分别判断。
- PMC 语义：TDM issued/completed、request size、EA read bytes/requests、channel CV、
  active residency；实际名字必须从目标 catalog 映射。
- wall-time：低代码 A/B，KEEP 需各目标 `>=max(2%,3*CV)`，且 E2E 不退化。
- physical bytes：理论有效 bytes 不变；实测增加 >1% 且不能解释则回滚。
- correctness：第 4.2 节全覆盖。
- 回滚：symbol 未变化、资源显著恶化、任何 stage >1% 退化且无另一 stage/E2E 补偿。
- 决策：wpt2 胜出则作为 P2 baseline；wpt4 胜出但物理计数显示 descriptor 压力时仍进入 P2。

## 7. P2：shape-aware weighted owner

### 7.1 当前入口与建议 owner map

实现点：

- `mxfp4_preshuffle_gfx1250_tdm.py:347-372`：替换统一 wpt 分组；
- `:383-500`：A/B/SA/SB job 构造与 `split_inner`；
- `:502-540`：按 wave 派发 job；
- `:513-534,730-980`：issue、wait 与 outstanding 计算；
- `:113-132,186-204`：cache tag 与 symbol 必须加入 owner-policy tag。

M32/N256/K256 每 stage 逻辑 payload 近似：

- A：32 x 128B = 4096B，另有 LDS row pad；
- B：16 x 2048B = 32768B；
- SA：256B；
- SB：2048B。

建议起点：

```text
waves/SIMD-pairs（需由 current stats 核实映射）:
  pair0 = {wave0, wave1}
  pair1 = {wave2, wave3}

A  owners = {wave0, wave1}            # 2 x 2048B
B  owners = {wave0, wave1, wave2, wave3}  # 跨两个 SIMD-pair，4 x 8192B
SA owners = {wave0}                   # 1 x 256B，避免四份 64B descriptor
SB owners = {wave2, wave3}            # 2 x 1024B

descriptor/wave = [3, 2, 2, 2]
descriptor/SIMD-pair = [5, 4]
```

若 SA 单 owner 的 256B 几何不能 direct，则备选 `SA={wave0,wave1}`，形成两个 128B 请求。
不能为了平均 descriptor 数把 SA 再拆成 64B。

### 7.2 正确性改动要求

统一标量 `TDM_PER` 不再成立。必须：

1. 为每个 wave specialization 计算 `JOB_COUNT[wave]`；
2. `tensor_wait()`、`pipeline_fence(outstanding=...)`、tail drain、prologue、steady-state、
   `next_stage_wait` 全部使用当前 wave 的真实 outstanding job 数；
3. barrier 仍保持 workgroup-uniform；不能让无 job wave 跳过 collective；
4. `dispatch_wave_job()` 必须确保每个 tile 每个 A/B/SA/SB region 恰好被一个 owner subset
   完整覆盖，无 gap/overlap；
5. cache key、symbol 加 `_ownw1` 等 tag，避免与统一 wpt code object 混淆；
6. 静态打印每个 operand 的 `(outer,inner,stride,owners,bytes/owner)`，并断言
   coverage 与 alignment。

### 7.3 最小矩阵与验收

| 变体 | owner | 其他 |
| --- | --- | --- |
| A | P1 胜出的 uniform owner | M32/N256/K256/b2 |
| B | `[A01,B0123,SA0,SB23]` | 唯一变化 |
| C | `[A01,B0123,SA01,SB23]` | 仅当 B 的 SA256 不 direct |

- 机制：B 保留跨两个 SIMD-pair 的并行 feed，同时减少每-wave descriptor；SA 形成
  >=128/256B 请求。
- G1/G2：G1 28 stage 更敏感于 descriptor 频率；G2 12 stage 更敏感于额外分支/descriptor
  setup 固定成本。
- PMC：direct 128B/256B 与 smaller/indirect 比例、TDM op、EA requests/bytes、channel CV；
  无对应 counter 时写 `UNKNOWN`。
- wall-time：kernel 源码改动，G1 或 G2 `>=3%`，E2E 不退化。
- physical bytes：应不变或下降；增加 >1% 回滚。
- correctness：第 4.2 节，特别增加 1/17/33 rows 与空 expert，检测 split coverage。
- 停止：任何 wait hang、间歇错、symbol/cache alias、资源使 residency 降档、主 stage 未过 3%。

## 8. P3：K512 与 b1/b2 crossover

### 8.1 当前入口与机制

- env/config：
  `grouped_moe_gfx1250.py:1031-1044,1062-1094`；
- LDS pitch/resource：
  `mxfp4_preshuffle_gfx1250_tdm.py:148-178`；
- K loop：
  `mxfp4_preshuffle_gfx1250_tdm.py:755-980`。

[推断] K512 可把每个完整 K 覆盖的 descriptor 次数减半，但单 stage LDS、内部 KWS、
register lifetime 和 refill 间隔增加。b1 可恢复 residency，却缩短隐藏 memory latency 的窗口。
G1 K=7168 可整除 512（14 stages）；G2 K=3072 可整除 512（6 stages）。

### 8.2 禁止混淆的严格顺序

1. K256/b2（P2 baseline）；
2. K512/b2（只改 K）；
3. K512/b1（在步骤 2 上只改 buffer）；
4. K256/b1（回到 K256，只改 buffer，闭合 crossover）。

禁止把 K256/b2 与 K512/b1 的差异作为“改 K 有效”的唯一证据。

### 8.3 验收

- G1/G2 分别保留最佳组合，不强求两阶段相同。
- PMC：physical bytes、request count、request size、active residency、TDM op；
  名称由 catalog 映射。
- wall-time：源码 specialization 变化，目标 stage `>=3%`。
- physical bytes：unique bytes 不变；实测增加 >1% 无解释则回滚。
- correctness：覆盖完整 K、最后 stage、所有 routing 矩阵。
- 停止：scratch 出现、residency 降档且无时间收益、K512 任一尾处理错误、CV 上升到收益不可辨。

## 9. P4：channel-phase 与 expert 地址着色

### 9.1 当前入口与机制

- 当前常量：
  `mxfp4_preshuffle_gfx1250_tdm.py:236-252`，`TILES_PER_GROUP=16`；
- B/SB 地址：
  `:340-345,461-500`；
- grid：
  `:1256-1289`。

先只扫 `TILES_PER_GROUP={1,4,8,16,32}`。它改变 active workgroups 在 expert/M/N 上的时序相位，
不改变 tile、bytes 或计算。

[硬件文档事实] GL2 有 address hashing/XOR/remap；因此不能用简单
`address % channel_count` 推断实际 channel。只有目标 physical per-channel counter 的 CV
可证明失衡。

只有在 `TILES_PER_GROUP` 扫描后 physical channel CV 仍高，才试：

1. expert K-phase rotation：不同 expert 的 K stage 起点/调度相位轮转，不改变逻辑布局；
2. 每 expert 256B color padding：离线 B 与 SB stride 同步加 color，必须更新 preshuffle、
   descriptor stride、cache key 和 shape metadata。

### 9.2 最小矩阵与验收

- A/B：仅五个 `TILES_PER_GROUP` 值，其他固定 P3 最佳值。
- phase rotation：off/on 单变量。
- color padding：0/256B 单变量；只在 channel 证据成立后执行。
- G1/G2：B row 分别为 57,344B=224x256B 和 24,576B=96x256B；SB row 分别为
  7,168B=28x256B 和 3,072B=12x256B。二者 expert stride/hash 相位不同，必须分别选值。
- PMC：per-channel physical bytes/requests 的 mean、max、CV，外加总 physical bytes。
- wall-time：常量扫描 `>=max(2%,3*CV)`；padding 属 layout，`>=5%`。
- physical bytes：phase 调整应不变；padding 增加的静态容量必须单列，执行物理 bytes
  增加 >1% 或无 5% 收益回滚。
- correctness：所有 routing，尤其随机 expert id 与 empty expert。
- 停止：无 channel counter 时不得声称“修复 channel balance”；可仅按 wall time 保留 group 值。

## 10. P5：B/SB two-K-stage superstage

### 10.1 机制与实现点

目标是把两个相邻 K256 的 B/SB 合为一个 K512 descriptor superstage，使 B/SB descriptor
频率减半，同时 A/SA 仍保持 K256 pipeline。

改动点：

- stage size/offset：`mxfp4_preshuffle_gfx1250_tdm.py:148-178`；
- Job 增加 operand kind 与独立 `k_adv`：`:378-500`；
- `issue()` 拆成 `issue_as(k256)` 与 `issue_bsb_super(k512)`：`:513-534`；
- consumer 在偶/奇 K256 选择 superstage 前/后半：`:680-870`；
- wait/job count 按 A/SA 与 B/SB 两个生产节奏计算：`:872-980`。

风险：

- B/SB LDS footprint 翻倍，可能降低 residency；
- producer/consumer epoch 复杂化，奇偶 K256 容易读取错误半区；
- K=7168/3072 都是偶数个 K256 stage，但仍要显式处理通用尾部；
- descriptor 减半不等于 physical bytes 减半；有效 payload 不变。

### 10.2 最小矩阵与验收

- baseline：P3 最佳 uniform K pipeline；
- candidate：仅 B/SB superstage，A/SA K256；
- 可选：B-only superstage，用于分离 SB 复杂度。
- G1/G2：G1 descriptor 节省累计更大；G2 更容易被新增控制开销抵消。
- PMC：B/SB TDM op 数、request-size、physical bytes、resident WG。
- wall-time：源码大改，目标 stage `>=3%`；若 residency 降档，要求 `>=5%` 才保留。
- physical bytes：应基本不变；>1% 无解释回滚。
- correctness：奇/偶 stage、最后 stage、所有 routing/race。
- 停止：LDS 导致 occupancy 降档且时间未过门槛，或 wait 模型不能静态证明。

## 11. P6：B/SB temporal hint

### 11.1 先验证 lowering，禁止猜测

- 当前 TDM atom：
  `mxfp4_preshuffle_gfx1250_tdm.py:383-500`；
- 当前源码调用 `fx.rocdl.make_tdm_atom()` 未显式传 cache 参数。
- [硬件文档事实] CDNA5 ISA 的 tensor load/store 有 TH 字段；GL2 文档说明 scope 与 temporal
  字段在 NC mtype 下解析为 RT/NT/LU/FM 等 policy。

执行前必须：

1. 找到 Python API/IR attribute 到 ISA TH bits 的真实 lowering；
2. 用最小 compile-only artifact 证明指定值改变了目标 tensor instruction 的 TH；
3. 确认 B 与 SB 能分别设置，cache key/symbol 带 policy tag；
4. 未证明 lowering 前禁止性能实验。

不得虚构“GL1 bypass”；不得把 FM 当作无副作用的 streaming hint；不得根据字段名猜 policy。

### 11.2 最小矩阵与验收

按顺序：

1. current/default；
2. B=NT、SB=current；
3. B=current、SB=NT；
4. 仅在 2/3 各自有正收益时试 B=NT、SB=NT。

- G1/G2：G1 B surface 更大、复用间隔更长；G2 更可能受 active expert 的 cache residency
  影响，方向不能预设。
- PMC：GL2 request/fill/hit/miss bytes、EA physical bytes、GL1 requests；名字从 catalog 映射。
- wall-time：低代码但 lowering 风险高，`>=max(2%,3*CV)`。
- physical bytes：若 miss/fill 导致 >1% 增加且无明确 wall-time 收益，回滚。
- correctness：policy 不应改变数值；仍执行完整矩阵。
- 停止：TH 未变化、mtype/scope 不明、只有一次运行收益、任一 stage/E2E 超过 1% 退化。

## 12. P7--P10：有门槛的后续候选

### 12.1 P7：在 M32/新 owner 上条件复测 N128/N512

- 入口：`grouped_moe_gfx1250.py:1031-1039,1062-1094` 与 kernel 的
  `tile_n`/`n_warp`/LDS/epilogue 路径。
- 机制：N128 增 grid/降低 B descriptor 粒度；N512 减 grid/增单 WG B 与 accumulator/LDS。
- 旧 N128/N512 发生在 M64/uniform owner 状态，不能机械外推；但旧结果已失败，也不能无证据
  重跑。
- 触发条件：P2 已 KEEP，且 P0/P2 显示 N 方向 feed 并发不足或 request 几何可改善。
- 最小矩阵：N256 baseline -> N128；回到 baseline -> N512。保持 M/K/buffer/owner 不变。
- 2x2 warp grid 只能在 owner 修正后作低优先级反证，不能与 N 改动捆绑。
- G1/G2：输出 N 与 K 不同，grid/resource crossover 分别判断。
- PMC：physical bytes、request-size、channel CV、resident WG。
- wall-time：源码/config 变体 `>=3%`。
- physical bytes：padding/重复增加 >1% 回滚。
- correctness：N 边界、epilogue、全部 routing。
- 停止：没有新物理/资源证据时不启动；两点均失败后关闭此方向。

### 12.2 P8：same-buffer early refill

- 入口：`mxfp4_preshuffle_gfx1250_tdm.py:733-870,872-980`。
- 机制：当当前 K256 的最后 K128 已完整进入 rmem 后，提前向同一个 LDS buffer 发下一 tile
  的 TDM；保持 LDS 大小和 residency 不变。
- 必须证明：最后 K128 的所有 LDS load 已完成；下一次写不会覆盖尚未消费的 A/B/SA/SB；
  workgroup 内所有 wave 对 epoch 一致。
- 最小矩阵：P3/P5 最佳 baseline vs early-refill-on；不得同时改 buffer 数。
- G1/G2：G1 steady-state 更长；G2 prologue/epilogue 占比更高。
- PMC：TDM/EA feed、physical bytes、active residency；计数名由 catalog 映射。
- wall-time：源码改动 `>=3%`。
- physical bytes：应不变，>1% 回滚。
- correctness：增加高重复随机 seed、race stress、1/17/33 rows。
- 停止：任何间歇性错误、barrier divergence、wait 无法形式化对应 job epoch。

### 12.3 P9：cluster 仅作为 on-chip A/SA 候选

- 入口：`grouped_gemm_mxfp4.py:28-41,110-117`；
  kernel `:423-431,1256-1289`。
- 机制：只考虑 A/SA multicast/共享，目标是减少 on-chip 重复请求；它不是减少 B/SB
  unique HBM bytes 的方案。
- 旧 cluster2/3/4 已失败。只有 P0/P2 明确显示 A/SA on-chip 请求限制且新 owner 改变瓶颈时重开。
- 最小矩阵：cluster1 vs 唯一一个由 N tiles 整除且证据支持的 cluster 值；禁止一次扫 2/3/4。
- G1/G2：A/SA 占比和 grid 不同，分 stage 判定。
- PMC：A/SA source bytes、GL1/GL2 requests、physical bytes、cluster launch geometry。
- wall-time：源码/launch 改动 `>=3%`。
- physical bytes：总量增加 >1% 回滚。
- correctness：cluster peer、尾 grid、empty expert。
- 停止：若无 on-chip 证据或复现旧退化，立即关闭。

### 12.4 P10：B+SB 联合 4352B / 17x256B record

- 入口：离线 weight/scale preshuffle 生产者、metadata/cache，以及 kernel
  `mxfp4_preshuffle_gfx1250_tdm.py:340-345,461-500,680-729`。
- [推断] 对每 N32/K256 record，将 B 4096B 与 SB 256B 组合为
  4352B = 17x256B；目标是统一 B/SB 地址相位和 direct request 几何。
- 必须同时修改：离线布局版本、B/SB base/stride、descriptor、consumer offset、cache key、
  serialization/version guard；旧权重缓存不得静默复用。
- G1/G2：record 几何相同，但 expert surface/复用距离不同；分别测。
- 最小矩阵：旧 layout/current kernel vs 新 layout/配套 kernel；禁止交叉不匹配。
- PMC：128B/256B request、physical bytes、channel CV、GL2 fill。
- wall-time：高风险 layout，目标 stage `>=5%`，并将转换与 cache miss 成本计入 E2E。
- physical bytes：有效 payload不应增加；padding/metadata 后执行 bytes >1% 无解释回滚。
- correctness：版本错配必须 fail-fast；完整 routing、reload、cache reuse、NaN/Inf。
- 停止：小于 5%、cache/version 不可证明、物理流量增加、部署格式成本不可接受。

## 13. 激进项及触发条件

这些项目不进入 P0--P10 的默认流水线。

### 13.1 exact-active-grid / persistent 上限测试

- 触发：P0 显示 active tile 数或 launch 固定开销限制，且 memory roof 利用率随 active WG
  明显上升。
- 入口：`grouped_moe_gfx1250.py` 的 psum/tile map 与 kernel `grid`。
- 最小实验：current grid vs exact-active-grid；再单独测试 persistent。
- 门槛：源码 `>=3%`；physical bytes不增；完整 skew/empty correctness。
- 该项是上限测试，若 scheduler/atomic 开销抵消收益即停止。

### 13.2 多请求并发

- 触发：P2 后 owner 已平衡，计数器仍显示 feed 未达 B-like roof，且资源有余量。
- 入口：job interleave、issue schedule、descriptor state。
- 最小实验：当前并发深度 vs 单个更深深度；不得同时改 layout。
- 门槛：`>=3%`，physical bytes不增，wait/job accounting 可证明。

### 13.3 普通 async copy 替代 TDM

- 触发：目标 catalog/单变量实验明确指出小 SA/SB descriptor 无法有效 direct，且 P2/P5
  无法解决。
- 仅替换一个 operand，先 SA，再 SB；B 不作为首个替换对象。
- 门槛：`>=3%`，资源不导致 residency 降档，physical bytes不增。

### 13.4 fusion

- 触发：GEMM 单体已接近 same-pattern roof，而 E2E 仍由中间写回/读回主导。
- 先量化可删除的 **unique** 与 **physical** read/write bytes，再设计 fusion。
- 门槛：高风险 `>=5% E2E`；不得用增加 GEMM 内无效流量换取表面 TB/s。

## 14. 首轮给硬件 agent 的具体执行清单

严格按以下顺序，不并行开发：

1. 在 host 确认 branch、完整 HEAD、clean status；远端真实状态仍单独核实。
2. 在目标容器打印 exact env、GPU/partition/stepping/clock；确认 CSV row73 默认配置。
3. 清空 specialization override，按固定 warmup/iters、随机输入跑 5 个独立进程 current-HEAD
   baseline；报告 median/CV/clock。
4. 做 kernel trace/stats，确认两个 exact symbol 与 actual resources；若不是 M32/wpt4，停止。
5. 枚举目标 counter catalog，建立候选语义到实际 counter 名映射；可用才做分组 PMC，
   不可用字段全部写 `UNKNOWN`。
6. 跑 pure-read、read+write、B-like TDM-to-LDS 三类 microbench，建立 same-pattern roof。
7. 在固定 M32/N256/K256/b2/cluster1/actual-off 下做 wpt4 与 wpt2 A/B；用 symbol 证明
   specialization 真正改变。wpt1只作负对照。
8. 若 wpt2/wpt4 指向 per-wave descriptor 问题，进入 P2 weighted owner；
   若 owner 差异小而 descriptor/stage 固定开销明显，进入 P3 K512 crossover。
9. 不再开发 route-indexed A2 quant，不再开发 M32，不重复旧失败矩阵。

## 15. 硬件依据、冲突与适用边界

主要依据：

- `my_code/gfx1250_gemm_optimization_guide.md`：
  - §7：layout、LDS padding、128B/256B direct request；
  - §8：HBM-bound 诊断、tile/stage/residency；
  - §10：系统化 sweep 与测量；
  - §11.2：memory-bound 优先级；
  - §12：资料索引。
- `MI450-B0 GEMM Pipeline.pptx/.vtt`：
  B0 pipeline、WMMA co-execution、weight preshuffle、TDM descriptor tracking。
- `amd-instinct-cdna5-instruction-set-architecture.txt`：
  tensor load/store、scope/TH 编码。
- `amd-cdna5-whitepaper.txt`：
  MI455X 产品级 256 active WGP、Wave32、192 MB L2、432 GB HBM4、up to 23.3 TB/s、
  partition 支持。
- `MI400_Shader_Programming#65.txt`：
  hardware overview、VGPR/LDS、WMMA、multicast/async/TDM、调度。
- `mi400_tensor_dma#72.txt`：
  §4 256B/cycle 目标；§7.6 descriptor interleave；128B/256B direct-copy 对齐与地址展开。
- `mi400_merge_tcp_lds_cu#85.txt`、`mi400_tx_MAS#156.txt`：
  TCP/LDS latency hiding、direct tracking、combining、TX counter 语义。
- `MI400_GL1_CH_GLARB_Fabric_MAS#12.txt`：
  GL1 route/return 与 128B/256B flow。
- `MI400_GL2_MAS#8.txt`：
  §2.2.5 mtype/policy、§2.2.6 request size、§7.2 bandwidth、§8.3 hash/FIFO。
- `MI400_perfmon_spec/`：
  EA/TX/GL1/GL2/SQ counter 定义；只能作为 catalog 候选。
- `mi450_design_ecr_summary#3.txt`：
  stepping ECR，实际 B0 行为需与当前机器核对。

冲突与边界：

1. whitepaper 的 23.3 TB/s 是 MI455X 产品峰值；B0 pipeline 分享材料中的约 22 TB/s
   属不同 stepping/上下文。两者都不是当前 partition 的 sustained roof。
2. TDM 文档的 256B/cycle 是模块性能目标，不等于 kernel 可持续 HBM 带宽。
3. 文档定义的 counter 不保证当前目标工具暴露；必须以目标 catalog 为准。
4. 旧 M64 artifact 只能说明旧 specialization，不能用于 current M32/wpt4 的资源或 residency。
5. 每 wave/每 SIMD-pair tensor-op 限制是文档事实；“当前 wpt4 已 backpressure”只是待测推断。
6. GL2 hash 存在使简单 modulo channel 模型不可靠；channel 优化必须依赖目标物理分布。

## 16. 反馈契约

每个实验返回一个可机器解析记录，至少包含：

```yaml
identity:
  branch:
  head:
  dirty_status:
  remote_state: UNKNOWN|...
environment:
  container:
  gpu:
  partition:
  stepping:
  exact_env:
  warmup:
  iters:
  independent_processes:
specialization:
  csv_row:
  gemm1_symbol:
  gemm2_symbol:
  actual_resources:
    vgpr:
    sgpr:
    lds_bytes:
    scratch_bytes:
    block:
    grid:
    residency: UNKNOWN|...
timing_us_unprofiled:
  e2e: {median: null, cv: null}
  gemm1: {median: null, cv: null}
  gemm2: {median: null, cv: null}
  a2_quant: {median: null, cv: null}
roof:
  pure_read: UNKNOWN
  read_write: UNKNOWN
  b_like_tdm_to_lds: UNKNOWN
bytes_and_rates:
  logical_requested: {gemm1: null, gemm2: null}
  unique_logical: {gemm1: null, gemm2: null}
  physical:
    hbm_read: UNKNOWN
    hbm_write: UNKNOWN
    gl2: UNKNOWN
    gl1: UNKNOWN
    tdm: UNKNOWN
    lds: UNKNOWN
  counter_name_mapping:
correctness:
  balanced:
  unbalanced:
  skew:
  empty_expert:
  nan_inf:
decision:
  verdict: KEEP|REJECT
  reason:
  rollback:
```

反馈中必须显式分列 logical requested、unique-logical、physical；不得把缺失 physical
值用 logical/unique 值补齐。`KEEP` 必须同时给出 wall-time 门槛、physical bytes 守门和
correctness 证据；否则一律 `REJECT` 并回滚到记录中的 exact baseline。
