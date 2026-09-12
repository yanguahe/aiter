# GEMM1 persistent cluster 跨 block pipeline 可行性分析

## 1. 结论

普通 persistent mode，即仅让固定数量的 workgroup 在 kernel 内循环领取多个
tile，预计只能带来很小的收益，无法补足当前剩余约 9% 的 GEMM1 性能差距。

真正可能产生较明显收益的是更完整的跨 block pipeline：在计算完当前 block 的
WMMA core 后，让当前 block 的 exact-SiLU epilogue 和 output TDM store，与下一
block 的 `m_tile_map` binary search、descriptor 构造和 input TDM load 交叠。

根据最新 ATT 数据：

- 仅考虑这种跨 block overlap，其绝对理论收益上限约为 `11.2%`。
- 如果再假设 persistent 能完全摊销一部分 block setup，极端理论上限约为
  `12.9%`。
- 比较乐观的可实现上限约为 `6%～8%`。
- 考虑额外同步、TDM 竞争和寄存器压力后，更现实的净收益预计为 `4%～7%`。

因此，这一方向值得实现 prototype，但要达到剩余约 9% 的目标，需要隐藏超过
80% 的现有 epilogue，已经非常接近理论极限。

## 2. 当前 kernel 的 ATT 时间分解

分析对象为当前保留的两阶段 output-store 版本：

```text
sync_mg4_fc28_apre_exactopt
```

ATT 目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/a07-3_20260911T0526_exactopt_att/a07-3_exactopt_ostore2p
```

每个完整 active wave 的平均总时间约为 `31,331 cycles`：

| 阶段 | cycles/wave | 占比 |
|---|---:|---:|
| prologue | 6,130 | 19.6% |
| WMMA core | 21,702 | 69.3% |
| exact-SiLU/output epilogue | 3,499 | 11.2% |
| 合计 | 31,331 | 100.0% |

prologue 可以进一步分解为：

| prologue 子阶段 | cycles/wave | 占总时间 |
|---|---:|---:|
| kernel setup 到首次 `m_tile_map` load | 541 | 1.7% |
| 8-step binary search 及后续 boundary load | 542 | 1.7% |
| boundary load 完成到 initial cluster sync 完成 | 160 | 0.5% |
| cluster sync 完成到第一条 input TDM | 119 | 0.4% |
| input TDM issue、等待及首次 LDS load | 4,768 | 15.2% |

其中，共观察到 11 条相关 `m_tile_map` scalar load：8 条用于 fixed-step binary
search，其余用于 `mn_oob`、expert 起止边界和 cluster mask 计算。因此，纯 binary
search 的占比小于上表中的 `1.7%`；即使完全消除，也不足以提供明显的整体收益。

input TDM startup 的详细分解为：

| TDM startup 子阶段 | cycles/wave | 占总时间 |
|---|---:|---:|
| 8 条 input TDM 的 descriptor/issue span | 3,152 | 10.1% |
| 最后一条 TDM issue 到 `s_wait_tensorcnt` | 39 | 0.1% |
| `s_wait_tensorcnt` | 32 | 0.1% |
| wait 后到 LDS-ready barrier | 27 | 0.1% |
| LDS-ready `s_barrier_wait 0xffff` | 1,076 | 3.4% |
| barrier 完成后首次 LDS→VGPR load | 441 | 1.4% |

这说明 A/B/ScaleA/ScaleB 的启动路径占比较高，但其中很大一部分是实际 descriptor
构造和 TDM issue 指令，并不都是可以免费隐藏的异步等待。

## 3. exact-SiLU 和最终 output TDM 的占比

当前 epilogue 总时间约为：

```text
3,499 cycles/wave，占总时间约 11.2%
```

其中包括：

- exact-SiLU 的 clamp、`v_exp_f32`、`v_rcp_f32` 和乘法；
- FP32→BF16 pack；
- output LDS staging；
- 两阶段 output TDM descriptor 和 store；
- 对应的 LDS/barrier 同步。

当前两阶段 overlap 已经把最终：

```text
s_wait_tensorcnt 0x0
```

降低到约 `26 cycles/wave`，占总时间不到 `0.1%`。因此，不能再把完整 output
TDM store 时间当成跨 block pipeline 的新增收益。新方案的主要收益来源是隐藏
exact-SiLU 和 output staging，而不是继续消除最终 TDM drain。

## 4. 跨 block pipeline 的理想时间模型

当前每个 block 的执行过程可以近似表示为：

```text
P(block i) -> WMMA(block i) -> E(block i)
```

其中：

```text
P    =  6,130 cycles
WMMA = 21,702 cycles
E    =  3,499 cycles
```

如果使用 persistent cluster，并在当前 block 的 WMMA 完成后并行执行：

```text
exact-SiLU/output(block i)
binary-search/descriptor/input-TDM(block i+1)
```

理想 steady-state 时间将变成：

```text
WMMA + max(P, E)
= 21,702 + max(6,130, 3,499)
= 27,832 cycles
```

相对于当前：

```text
(31,331 - 27,832) / 31,331 = 11.17%
```

所以，仅考虑跨 block overlap，绝对理论收益上限约为 `11.2%`。这个上限意味着
整个 epilogue 都被下一 block 的 prologue 完全隐藏，并且没有任何额外控制、同步、
资源竞争或 spill 开销。

如果进一步假设 persistent loop 可以完全摊销 `541 cycles` 的 block setup，则极端
理论模型为：

```text
21,702 + (6,130 - 541) = 27,291 cycles
```

对应：

```text
(31,331 - 27,291) / 31,331 = 12.89%
```

但 `541 cycles` 中仍包含每个新 tile 必需的 block index、swizzle 和地址计算，因此
不能全部摊销；`12.9%` 只能作为不可达或接近不可达的极端上界。

## 5. 达到剩余 9% 需要的 overlap 效率

基于同轮权威数据：

```text
sync_mg4_fc28_apre          = 564.321 us
sync_mg4_fc28_apre_exactopt = 546.078 us
12% 目标                    = 496.603 us
```

当前 exactopt 到目标还需要：

```text
(546.078 - 496.603) / 546.078 = 9.06%
```

换算到 ATT cycles，需要减少约：

```text
31,331 × 9.06% ≈ 2,839 cycles/wave
```

而整个 epilogue 只有 `3,499 cycles/wave`，所以需要隐藏：

```text
2,839 / 3,499 ≈ 81.1%
```

也就是说，要达到目标，跨 block pipeline 必须隐藏超过 80% 的 epilogue，同时
新增开销必须控制在约 `660 cycles/wave` 以内。这是一个非常紧的实现窗口。

## 6. 技术可行性与主要限制

### 6.1 TDM 的异步特性支持该方向

CDNA5 文档说明 TDM 可以在 global memory 与 LDS 之间执行 descriptor-driven
asynchronous transfer。因此，在 next-block input TDM 已经 issue 后，理论上可以
继续执行当前 block 的 VALU/TRANS epilogue，让数据传输在后台进行。

这是跨 block pipeline 可行的硬件基础。

### 6.2 binary search 不能直接视为免费并行

`m_tile_map` binary search 使用依赖串联的 scalar load：下一次搜索位置依赖上一次
load 的结果。同一个 wave 不能把这 8 次搜索一次性全部 issue。

可以尝试把每次 scalar load 的等待窗口与 exact-SiLU 的 VALU/TRANS 指令交错，但：

- wave 仍然只有一条指令流；
- SALU/SMEM 与 VALU/TRANS 只能部分重叠；
- 下一 block 的 expert、`mn_oob` 和 multicast mask 必须全部确定后，才能构造最终
  TDM descriptor。

因此，binary search 的 `542 cycles` 可以部分隐藏，但不能直接从总时间中全部扣除。

### 6.3 input 和 output 必须使用不同的 LDS 区域

当前 kernel 在 WMMA 完成后，复用 input ring 所在的 LDS arena 来暂存 output。
如果在当前 epilogue 期间把下一 block 的 A/B/ScaleA/ScaleB 预取到同一 arena，
output staging 会覆盖这些输入数据。

当前 input ring 占用：

```text
278,528 B ≈ 272 KiB
```

当前 active-SiLU 的完整 output staging 约为：

```text
69,632 B ≈ 68 KiB
```

完全分离需要：

```text
272 KiB + 68 KiB = 340 KiB
```

超过 CDNA5 每 WGP 的 `320 KiB` LDS 容量。

可行方案是继续使用两阶段 output store，但仅分配一个可复用的 half-output arena：

```text
half-output arena ≈ 34 KiB
input ring + half-output arena ≈ 306 KiB
```

这样可以放入 `320 KiB` LDS，但在复用 half-output arena 写第二半结果前，必须确认
第一半 output TDM 已完成。

### 6.4 input 和 output 会竞争同一个 TDM

即使 LDS 地址完全分离，current-block output store 和 next-block input load 仍然由
同一 WGP 的 TDM 执行。两者可以与 exact-SiLU compute 重叠，但不能假设彼此之间
没有带宽、队列和 descriptor issue 竞争。

合理的次序可能是：

1. 完成当前 block 的 WMMA core。
2. 计算下一 block 的 tile/expert/boundary。
3. issue 当前 block 第一半 output TDM。
4. issue 下一 block 的 input TDM。
5. 继续当前 block 第二半 exact-SiLU。
6. 等待第一半 output 完成并复用 half-output arena。
7. issue 第二半 output TDM。
8. 在下一 block 开始 WMMA 前等待 input ready。

实际最优次序需要根据 TDM queue 的完成顺序和 ATT 重新调整。

### 6.5 register 压力可能抵消收益

当前 kernel 已经只能保持每个 physical SIMD 一个 active slot。跨 block pipeline
需要在当前 block accumulator 仍然存活时，同时保存下一 block 的：

- expert 和 `m_tile_map` boundary；
- A/B/ScaleA/ScaleB global offset；
- descriptor 状态；
- multicast mask 和 OOB extent；
- persistent tile-loop 状态。

如果新增 live SGPR/VGPR 导致 spill，或让现有 WMMA/epilogue 的寄存器分配变差，
理论 overlap 收益会迅速消失。

### 6.6 必须以整个 cluster 为 persistent worker

当前使用：

```text
cluster_m = 4
cluster_n = 4
```

即一个 cluster 包含 16 个 WG。CDNA5 ISA 规定 cluster 最多包含 16 个 WG，并且
每个 WG 位于不同 WGP。

由于 input TDM 使用 multicast，ring reuse 使用 cluster barrier，不能让单个 WG
独立从全局 atomic queue 获取下一 tile。否则 cluster 内的 WG 会进入不同的 tile、
expert 或 barrier generation，导致数据错误或 deadlock。

persistent 调度必须满足：

- 以完整 4×4 cluster 为最小 worker；
- cluster 内所有 16 个 WG 同时进入同一个 next macro-tile；
- 使用固定 stride 或 cluster-level queue；
- 最后不足一个完整 cluster 的任务仍要生成一致的 sentinel tile；
- 每个新 M tile 继续执行原 `m_tile_map` binary search，以兼容 non-balanced 分布。

## 7. 现实收益区间

### 7.1 可以较可靠隐藏的部分

在 input TDM 全部 issue 后，到第一条 WMMA 之前有约：

```text
39 + 32 + 27 + 1,076 = 1,174 cycles/wave
```

这些主要是 TDM readiness 和 barrier 等待，适合通过 split-phase wait 与当前
epilogue 重叠。

`m_tile_map` load span 约为 `542 cycles/wave`，其 SMEM latency 也有机会和
exact-SiLU 的 VALU/TRANS 指令部分重叠。

因此，在不假设 descriptor 指令免费执行的情况下，比较可信的原始 overlap 空间约为：

```text
1,174 + 542 ≈ 1,716 cycles/wave
```

对应总时间约 `5.5%`。

### 7.2 乐观部分

如果 descriptor/SALU 指令可以与 exact-SiLU 的 VALU/TRANS 指令较好地交错，并且
persistent loop 能摊销一部分 setup，则还可能多隐藏约 `500～1,000 cycles/wave`。

这会把乐观收益推到约：

```text
6%～8%
```

但还没有扣除：

- persistent tile-loop 控制；
- cluster-level work assignment；
- half-output arena 的复用等待；
- input/output TDM queue 竞争；
- 新增寄存器压力；
- 第一轮和最后一轮无法形成完整 overlap 的边界 bubble。

因此，更现实的净收益预计为 `4%～7%`。

## 8. 建议的 prototype 实现顺序

### 阶段一：只实现 persistent cluster

- 保持现有 4×4 cluster 和单 block 内部计算完全不变。
- 使用固定 cluster stride 遍历 macro-tile，避免 atomic queue。
- 每个 tile 仍执行原 binary search。
- 不做跨 block TDM/epilogue overlap。

目的不是获得最终性能，而是确认：

- cluster 内 16 个 WG 可以稳定同步进入下一 tile；
- sentinel/tail cluster 正确；
- non-balanced 输入正确；
- persistent loop 本身没有明显回退。

如果这一阶段已经回退超过约 `1%～2%`，则不应继续。

### 阶段二：增加独立 half-output LDS arena

- 保留现有 input ring。
- 增加约 34 KiB 的 half-output staging。
- 保持 exact-SiLU 数学操作、BF16 pack 和输出 layout 不变。
- 验证两阶段 output store 在新的 LDS 地址下逐字节一致。

### 阶段三：跨 block prefetch

- 当前 block WMMA 结束后，计算 next tile 的 binary search 和 descriptor。
- 在当前 exact-SiLU 期间 issue next-block input TDM。
- 使用 split-phase workgroup/cluster synchronization，在真正读取 next-block LDS
  前才执行 wait。
- output store 和 input load 的 TDM 顺序需要通过 ATT 调整。

### 阶段四：验收和止损

必须验证：

- balanced random 输出 hash 与 baseline 一致；
- non-balanced random 通过；
- `m_tile_map` binary search 保留；
- exact-SiLU 的操作和累加顺序不变；
- 其他 MoE kernel 的接口与功能不变；
- 无 deadlock、无残留 GPU 进程；
- ATT 中无 SGPR/VGPR spill，且 input readiness wait 确实被 epilogue 覆盖。

建议止损条件：

- persistent-only 阶段回退超过 `2%`；或
- 完成跨 block overlap 后提升仍低于 `3%`；或
- 新增 spill、cluster barrier 不稳定或 non-balanced correctness 无法保证。

## 9. 最终判断

该方案在数学上有约 `11.2%` 的 overlap 上限，确实覆盖当前剩余约 9% 的目标，
因此比普通 persistent mode 更值得尝试。

但达到目标要求隐藏超过 80% 的现有 epilogue，而实际实现同时受到 LDS 容量、
TDM input/output 竞争、cluster lockstep 和寄存器压力限制。更合理的预期是
`4%～7%`，乐观情况下可能接近 `8%`。只有在 descriptor/SALU 与 exact-SiLU
的 VALU/TRANS 调度高度重叠、且没有新增 spill 的情况下，才可能接近或超过 9%。
