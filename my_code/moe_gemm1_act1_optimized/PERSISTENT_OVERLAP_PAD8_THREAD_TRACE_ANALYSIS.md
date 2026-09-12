# `persistent_overlap_pad8` thread-trace 分析

## 结论

当前 kernel 的主要剩余瓶颈在 K hotloop 的 input TDM 同步和 cluster 同步。
三次 ATT 的 steady persistent task 中，所有显式 wait 的 stall 合计约占
`30%`；其中 `s_wait_tensorcnt` 约占 `12.5%`，`s_barrier_wait` 约占
`11.6%`，`s_wait_dscnt` 约占 `5.8%`。output-pad8 已把 output LDS
drain 的两个主要 `s_wait_dscnt 0` 压到约 `150 cycles/task`，它们已经不是
首要瓶颈；但下一 task prologue 中等待前一 task output TDM 完成的
`s_wait_tensorcnt 0` 仍约占 `3.5%`。

按 issue timeline 划分，WMMA 占约三分之一，LDS read/write 约占八分之一，
显式同步等待约占三分之一。下一轮优化应优先减少 TDM ready 时间和 cluster
peer 到达偏差，而不是继续压缩 SiLU 指令条数。

## 方法

分析使用规则文件指定的 `trace_segment_cycles.py`：

```text
SHA256=6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

数据来自 d01-3 的三次 `persistent_overlap_pad8` ATT capture。每个 capture
包含四个 SE 的 `SIMD3-select` wave trace。脚本使用 task-entry 到下一次
task-entry 的动态时间戳作为一个 persistent task 区间，并展开 36 个相同
sample point。每个 wave 的最后一个 task 没有后继 task-entry，因此得到每个
capture `4 waves × 35 tasks = 140` 个完整区间。占比统计再去掉每个 wave
的第一个 cold task，保留 `136` 个 steady task。

`trace_segment_cycles.py` 的 interval cycle 使用 `end_ts - start_ts`，包含区间
内的 wait stall。指令级 latency 可以与后续独立指令重叠，因此非 wait 指令的
latency 只用于寻找异常长事件；全周期占比采用相邻 instruction issue timestamp
之差，保证各类别加总接近完整 task cycle。

## 完整 task cycle

| capture | steady tasks | mean cycles/task | p50 | p90 |
|---|---:|---:|---:|---:|
| `pad8_att_1` | 136 | 28,850.3 | 27,709.5 | 35,858.5 |
| `pad8_att_2` | 136 | 27,817.2 | 27,665.5 | 29,228.0 |
| `pad8_att_3` | 136 | 28,664.3 | 27,808.5 | 35,091.5 |

三次 capture 的 steady-task mean 中位数为 `28,664.3 cycles/task`。
首个 task 明显受 cold-start/TDM 建链影响，因此不用于 steady-state 占比。

## 整个 kernel wave-lifetime 的 wait 占比

这一统计直接覆盖每个 trace wave 从第一条到最后一条指令，包括 cold
task、后续 steady task 和 final drain。每个 capture 先在四个 SE wave
之间求平均，再对三次 capture 取中位数。

| capture | occupancy max kernel cycles | mean traced-wave cycles |
|---|---:|---:|
| `pad8_att_1` | 1,152,616 | 1,145,688.5 |
| `pad8_att_2` | 1,107,196 | 1,101,049.0 |
| `pad8_att_3` | 1,113,445 | 1,106,442.0 |

occupancy max 的中位数为 `1,113,445 cycles`；四个 traced
wave lifetime 的中位数为
`1,106,442.0 cycles`。
两种口径相差不到 1%，下面用 traced-wave lifetime 作为 wait 占比分母。

| 指令组 | cycles/wave | 整个 wave lifetime 占比 |
|---|---:|---:|
| `s_wait_tensorcnt` | 122,581.0 | 11.08% |
| `s_barrier_wait` | 113,714.2 | 10.28% |
| `s_wait_dscnt` | 61,358.2 | 5.55% |
| `s_wait_kmcnt` | 2,338.5 | 0.21% |
| `s_wait_idle` | 2,447.0 | 0.22% |
| 合计 | 302,439.0 | 27.33% |

三次 capture 的 mean wave lifetime 中位数为 `1,106,442.0 cycles`。
因此从整个 kernel 生命周期看，显式 wait stall 约占 `28%`；去掉 cold
task 后，steady task 中该比例约为 `29%`。两种口径结论一致。

## 阶段占比

下表先对每个 capture 求 steady-task phase mean，再取三次 capture 的中位数。

| 阶段 | cycles/task | 占完整 task |
|---|---:|---:|
| Task prologue + K hotloop | 24,373.5 | 85.03% |
| Output descriptor/address setup | 216.8 | 0.76% |
| SiLU banks 0-1 + first LDS drain | 1,747.9 | 6.10% |
| SiLU banks 2-3 | 1,843.9 | 6.43% |
| Second output descriptor finalization | 118.0 | 0.41% |
| Second output launch + persistent boundary | 240.3 | 0.84% |
| 分段合计 | 28,540.4 | 99.57% |

`Task prologue + K hotloop` 占约 `85%`。两个 SiLU/output staging 段合计约
`12.5%`，descriptor setup 和 persistent boundary 合计约 `2%`。因此即使
把整个 SiLU/output 部分理想化为零成本，上限也只是约 `14%`；实际可优化
空间显著更小。

## `s_wait_*` 与 barrier stall

这些值直接累加 trace event 的 `stall` 字段。对 wait 指令而言，stall
期间 wave 不能 issue，因此可以与完整 task cycle 相除。

| 指令组 | stall cycles/task | task 占比 | 动态次数/task | 平均 stall/次 |
|---|---:|---:|---:|---:|
| `s_wait_tensorcnt` | 3,457.1 | 12.06% | 30.0 | 115.2 |
| `s_barrier_wait` | 3,230.0 | 11.27% | 46.0 | 70.2 |
| `s_wait_dscnt` | 1,630.5 | 5.69% | 171.0 | 9.5 |
| `s_wait_kmcnt` | 34.0 | 0.12% | 1.0 | 34.0 |
| `s_wait_idle` | 14.3 | 0.05% | 1.0 | 14.3 |
| 合计 | 8,365.9 | 29.19% | — | — |

按具体 wait immediate 合并后：

| 指令 | stall cycles/task | task 占比 | 动态次数/task |
|---|---:|---:|---:|
| `s_barrier_wait 0xfffd` | 2,406.6 | 8.40% | 9.0 |
| `s_wait_tensorcnt 0x2` | 2,353.6 | 8.21% | 29.0 |
| `s_wait_dscnt 0x8` | 1,313.8 | 4.58% | 84.0 |
| `s_wait_tensorcnt 0x0` | 998.4 | 3.48% | 1.0 |
| `s_barrier_wait 0xffff` | 823.5 | 2.87% | 37.0 |
| `s_wait_dscnt 0x14` | 150.8 | 0.53% | 56.0 |
| `s_wait_dscnt 0x0` | 150.7 | 0.53% | 3.0 |
| `s_wait_kmcnt 0x0` | 34.0 | 0.12% | 1.0 |
| `s_wait_idle` | 14.3 | 0.05% | 1.0 |
| `s_wait_dscnt 0x4` | 0.9 | 0.00% | 28.0 |

最重要的具体 wait PC 是：

| PC | 指令 | stall cycles/task | task 占比 | 解释 |
|---|---|---:|---:|---|
| `0xac44` | `s_barrier_wait 0xfffd` | 1,965.8 | 6.86% | 每个 K-ring generation 末尾的 cluster barrier；peer 到达偏差最大的单一热点。 |
| `0x36fc` | `s_wait_tensorcnt 0x2` | 1,792.2 | 6.25% | 初始 input TDM 发出后等待 `TENSORcnt<=2`，暴露首批 payload/scale 到达时间。 |
| `0x2dfc` | `s_wait_tensorcnt 0x0` | 998.4 | 3.48% | 下一 task prologue 等待前一 task 的 output TDM 完成，保护即将复用的 LDS，随后才进入 input TDM/WG/cluster 同步。 |
| `0xa6c0` | `s_wait_dscnt 0x8` | 554.4 | 1.93% | steady hotloop 的 LDS read drain，等待下一组 WMMA 所需 operand。 |
| `0x9f88` | `s_wait_dscnt 0x8` | 320.8 | 1.12% | steady hotloop 的另一处 LDS read drain。 |
| `0xa8d0` | `s_barrier_wait 0xffff` | 241.7 | 0.84% | 7.0 次/task；单次最大 latency 中位数 9,022 cycles。 |
| `0x9a10` | `s_wait_tensorcnt 0x2` | 170.1 | 0.59% | 7.0 次/task；单次最大 latency 中位数 946 cycles。 |
| `0x8f64` | `s_nop 0` | 168.1 | 0.59% | 显式 `s_nop 0` hazard spacing；固定成本。 |
| `0xa880` | `s_wait_tensorcnt 0x2` | 166.2 | 0.58% | 7.0 次/task；单次最大 latency 中位数 1,211 cycles。 |
| `0xdc10` | `s_barrier_wait 0xfffd` | 156.3 | 0.55% | persistent task 边界的 cluster barrier。 |
| `0x2e08` | `s_barrier_wait 0xfffd` | 154.4 | 0.54% | 1.0 次/task；单次最大 latency 中位数 1,224 cycles。 |
| `0xa148` | `s_wait_tensorcnt 0x2` | 147.4 | 0.51% | 7.0 次/task；单次最大 latency 中位数 8,239 cycles。 |

其中 `0xac44 s_barrier_wait 0xfffd` 与 `0x36fc s_wait_tensorcnt 0x2`
是最大的两个稳定热点。二者合计约占一个 steady task 的 `13%`。
`0x2dfc s_wait_tensorcnt 0x0` 再占约 `3.5%`。这三处已经解释约六分之一
的 task cycle。需要注意，`0x2dfc` 等待的是前一 persistent task 的
output TDM，而不是当前 task 的 input TDM。

## issue timeline 构成

下表把每条动态指令到下一条动态指令的 timestamp 差归到前一条指令。
这是一种互斥的 wave issue-timeline 分解，适合判断时间消耗在哪类工作上；
它不是各执行单元的独占 busy counter。

| 类别 | cycles/task | 占比 |
|---|---:|---:|
| WMMA issue | 9,632.7 | 33.61% |
| TENSORcnt wait | 3,487.1 | 12.17% |
| LDS read issue | 3,480.0 | 12.14% |
| barrier wait | 3,301.7 | 11.52% |
| DScnt wait | 2,098.8 | 7.32% |
| SALU/control issue | 1,854.5 | 6.47% |
| packed SiLU VALU issue | 1,753.0 | 6.12% |
| other VALU issue | 1,276.0 | 4.45% |
| EXP/RCP issue | 1,000.0 | 3.49% |
| explicit NOP | 238.1 | 0.83% |
| LDS write issue | 148.9 | 0.52% |
| TDM issue | 103.0 | 0.36% |
| other wait | 50.5 | 0.18% |
| other issue | 1.0 | 0.00% |

WMMA issue timeline 约占三分之一，但单条 WMMA 没有形成百 cycle 级的
serial stall；它是必要计算吞吐。`EXP/RCP` 和 packed SiLU 的成本已经被
软件流水覆盖在约 `12.5%` 的完整 epilogue 中，优先级低于 input TDM/cluster
同步。

## 其他长 latency 指令

除 wait/barrier 外，超过 100 cycles 的事件几乎全部是 `ds_load_b128`。
它们的单次最坏 latency 可达到约 `0.6–1.3k cycles`，但同一静态 PC 的
平均 stall 通常只有几十 cycles/task，说明它们是间歇性的 LDS queue/bank
冲突，而不是像 TDM/barrier 那样每个 task 都稳定暴露的大气泡。

| PC | 指令 | 单次最大 latency 中位数 | stall cycles/task |
|---|---|---:|---:|
| `0xaae0` | `ds_load_b128 v[28:31], v72 offset:2560` | 732 | 37.3 |
| `0xab20` | `ds_load_b128 v[44:47], v72 offset:4608` | 616 | 14.4 |
| `0xa3e8` | `ds_load_b128 v[44:47], v75 offset:4608` | 530 | 7.4 |
| `0xab70` | `ds_load_b128 v[68:71], v72 offset:7680` | 517 | 10.6 |
| `0xab10` | `ds_load_b128 v[36:39], v72 offset:3584` | 517 | 7.9 |
| `0x37d4` | `ds_load_b128 v[8:11], v72` | 441 | 34.3 |
| `0xa3a8` | `ds_load_b128 v[28:31], v75 offset:2560` | 438 | 8.8 |
| `0xab68` | `ds_load_b128 v[64:67], v72 offset:7168` | 419 | 5.9 |

`s_nop 0` 的显式 hazard spacing 约 `168 cycles/task`，约占 `0.6%`。它可以
作为低优先级调度微调目标，但删除前必须重新验证 SCHED_MODE 2 下的
RAW/WAR hazard；其收益上限远小于 TDM/barrier。

## output-pad8 已解决的部分

用同三轮 baseline trace 比较 output 区域的三个 `s_wait_dscnt 0`：

| 版本 | output wait stall cycles/task |
|---|---:|
| `persistent_overlap` | 536.0 |
| `persistent_overlap_pad8` | 150.7 |

steady task 中 output wait 从约 `536.0` 降到
`150.7 cycles/task`，下降
`71.89%`。
这说明当前继续优化 output LDS bank mapping 的边际收益已经明显下降。
这里减少的是 LDS store drain；前一 task 的 output TDM 写回仍会在下一
task 的 `0x2dfc s_wait_tensorcnt 0x0` 暴露约 `998 cycles/task`。

## 下一步优化优先级

1. 优先处理 `0x36fc s_wait_tensorcnt 0x2`。它是 input TDM ready 的
   最大单点，应把更多独立 WMMA/DS/address work 移到 wait 前，或改变
   input TDM descriptor 的发出时机；必须保持每 wave 最多 3 个、每 SIMD
   最多 6 个 in-flight TDM descriptor。
2. 缩小 `0xac44 s_barrier_wait 0xfffd` 的 peer 到达偏差。该 wait 每 task
   动态执行 6 次，是最大的单一稳定热点。应比较四个 wave 的 TDM owner
   工作量和 DS-read tail，而不是删除 cluster barrier。
3. 继续隐藏 `0x2dfc s_wait_tensorcnt 0x0` 的 previous-output TDM drain。
   当前 task/address setup 已经覆盖一部分延迟；后续只能在不提前覆盖
   320 KiB LDS、且不破坏 TDM in-order 约束的前提下再前移独立工作。
4. 针对 `0xa6c0`、`0x9f88` 等 `s_wait_dscnt 0x8` 前的 DS burst 重新排程，
   把不依赖目标 VGPR 的 WMMA/SALU 穿插到 wait 前。
5. 最后才考虑 SiLU epilogue 和显式 NOP。当前整个 output/Silu 区间约占
   `14%`，而 output drain 本身已经被 pad8 大幅压低。

## 硬件依据

- MI400 Shader Programming Guide §4.3.7（第 84–88 页）：`S_WAIT_*CNT`
  等待期间 wave 处于 inactive 状态；`DScnt` 跟踪 LDS，`TENSORcnt` 跟踪
  TDM transfer；`S_WAIT_*CNT N` 同时也执行 `S_WAIT_XCNT N`，因此观测到的
  latency 还可能包含地址转换完成时间。
- MI400 Shader Programming Guide §4.3.6：`S_BARRIER_WAIT` 必须等待对应
  workgroup/cluster 的所有成员完成 signal。
- MI400 Shader Programming Guide §4.10.8（第 205–206 页）：每 wave 最多
  3 个、每 SIMD 最多 6 个 TDM descriptor 在 XACK 前处于 in flight；TDM
  可以在多个 wave/descriptor 间切换，并可能被 VMEM arbitration 阻塞。
- CDNA5 ISA §7.10/§7.12：`EXP/RCP` 属于 transcendental pipeline，WMMA
  属于 XDL pipeline，两类完成可与普通 VALU 乱序并行。

## 复现命令

```bash
python3 my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py \
  my_code/moe_gemm1_act1_optimized/pad8_trace_full_task.json

python3 my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py \
  my_code/moe_gemm1_act1_optimized/pad8_trace_full_task.json \
  --specific-part-representative-trace

python3 my_code/moe_gemm1_act1_optimized/analyze_pad8_thread_trace.py
```
