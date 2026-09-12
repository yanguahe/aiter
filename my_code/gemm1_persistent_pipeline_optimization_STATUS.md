# GEMM1 persistent cluster 跨 block pipeline 优化记录

## 目标与约束

本轮以 `sync_mg4_fc28_apre_exactopt` 为基础，按照
`my_code/gemm1_persistent_pipeline_analysis.md` 的方案验证 persistent cluster
以及跨 block input-prefetch / exact-SiLU overlap。

必须保持：

- GEMM1 外部接口、数学功能与 exact-SiLU 精度不变；
- 每个新 M tile 仍通过 `m_tile_map` binary search 查找 expert；
- 支持 non-balanced token distribution；
- 不修改其他 MoE kernel 的接口和功能；
- 性能未优于 `sync_mg4_fc28_apre_exactopt` 的实验版本不作为最终保留版本。

## 阶段一：persistent-only prototype

实现方式：

- 以完整 `4x4` cluster 作为 persistent worker；
- 机器有 256 个 CU，使用 16 个 persistent cluster；
- cluster 通过固定 grid stride 遍历 macro-tile，不使用 global atomic queue；
- 每个 tile 保留原 `m_tile_map` binary search；
- non-balanced 的最后一个 partial M cluster 使用零 OOB sentinel 路径，使 16 个
  WG 保持相同的 cluster barrier generation。

a07-3 同轮结果目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260912T085002Z
```

| case | GEMM1 samples (us) | median (us) | vs exactopt | MOE e2e samples (us) | median (us) | random/hash |
|---|---|---:|---:|---|---:|---|
| `sync_mg4_fc28_apre_exactopt` | 534.256, 534.115 | 534.186 | +0.00% | 1373.70, 1376.15 | 1374.93 | pass, baseline hash |
| `persistent-only pc16` | 543.219, 545.226 | 544.223 | -1.88% | 1385.84, 1388.24 | 1387.04 | pass, baseline hash |

阶段一没有超过文档设定的 2% 止损线，但没有直接收益。

code object metadata：

| kernel | SGPR | SGPR spill | VGPR | VGPR spill | LDS |
|---|---:|---:|---:|---:|---:|
| `exactopt` | 58 | 0 | 804 | 0 | 278528 B |
| `persistent-only pc16` | 107 | 32 | 876 | 0 | 278528 B |

CDNA5 文档说明每个 wave 有 106 个普通 SGPR，另有 SGPR106/107 保存 VCC。
因此 persistent-only 已达到该架构的 SGPR 上限并发生 spill；额外 VGPR 数主要来自
SGPR spill 保存。

## 阶段二/三初版：stage-3 output arena + next stages 0..2 prefetch

初版复用 input ring 的 stage 3 作为当前 output arena，并在第二半 exact-SiLU
期间预取下一 tile 的 stages 0..2。随机输入最终可做到 baseline hash 一致，但性能
回退：

| 机器 | 版本 | GEMM1 (us) | 说明 |
|---|---|---:|---|
| d01-3 | 初版，完整 next state 跨 epilogue 存活 | 586.851 | 正确；明显回退 |
| d01-3 | 缩短 next state lifetime | 580.399 | 正确；仍回退 |
| a07-3 | wave-owner descriptor 专门化 | 566.501 | 正确；同机 exactopt 约 532 us |
| a07-3 | 上述版本，`epilogue_batch_wn=4` | 564.609 | 正确；仍明显回退 |

上述版本的 code object 为 `107 SGPR / 53 SGPR spills / 872 VGPR`。降低
`epilogue_batch_wn` 与关闭 kernarg preload 均未消除 spill。

## ATT 证据

跨 block 初版 ATT：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/a07-3_20260912_persistent_att/a07-3_xbp3
```

每个 persistent wave 执行 36 个 tile，共 64,512 条 WMMA。平均 wave span 为
约 1,240,112 cycles，即约 34,448 cycles/tile；原 exactopt 为约
31,331 cycles/tile。

主要新增开销：

- `s_barrier_wait` 合计约 5,119 cycles/tile，原 exactopt 约 2,960
  cycles/tile；
- 迭代末为安全复用 output/input LDS stage 增加的 workgroup barrier 约
  1,490 cycles/tile；
- `s_wait_loadcnt` 约 1,984 cycles/tile，原因是 next tile binary search 已在
  当前 epilogue 执行，但下一迭代又重复执行一次。

该 capture 的完整 wave 覆盖 36 个 tile：

```text
wave span                = 1,240,112 cycles
mean cycles/tile         = 34,448
s_barrier_wait/tile      = 5,119 cycles
s_wait_loadcnt/tile      = 1,984 cycles
s_wait_tensorcnt/tile    =   254 cycles
```

其中迭代末 LDS 复用 barrier 的单点开销约为 `1,490 cycles/tile`。这说明把
stage 3 同时作为 current output arena 与 next input slot，会新增一个无法忽略的
跨 wave rendezvous。

CDNA5 Shader Programming Guide 还说明 SMEM 与 VMEM/TDM 之间切换时硬件会等待
相关 `XCNT` 清零。因此，预取后在下一迭代重新执行 `m_tile_map` scalar loads
不仅重复工作，还会迫使已发出的 input TDM 提前 drain，破坏跨 block overlap。

## 阶段二/三改进版：half-output arena 与 metadata carry

随后按 ATT 证据实现了更完整的结构：

- 使用独立 half-output arena，总 LDS 约 306 KiB；
- 下一 tile 的 4 个 input stage 全部在当前 exact-SiLU 后半段前发出；
- 通过 half-output row padding 传递上一轮已经执行的 binary-search 结果，避免
  下一迭代再次发出 `m_tile_map` SMEM loads；每个 tile 仍执行且只执行一次原
  binary search；
- 分别验证了 output arena 位于 input ring 之后、output arena 位于 LDS offset 0，
  以及增加/调整 TDM wait 与 workgroup barrier 的版本。

这些版本均能完成运行，但 random 输出无法与 baseline 逐字节一致：

```text
logits_diff ≈ 1.48e-3 ～ 1.68e-3
rel_l2      ≈ 5.44e-2 ～ 5.80e-2
output hash != baseline hash
```

最后一个版本的单轮性能为：

```text
GEMM1  = 558.055 us
MOE e2e = 1398.87 us
```

相邻时间窗口内 `sync_mg4_fc28_apre_exactopt` 为约 `532～534 us`，因此该版本
即使忽略精度失败也仍回退约 4%～5%。它不满足功能/精度等价约束，不能保留。

## 最终结论与止损

本轮没有找到优于 `sync_mg4_fc28_apre_exactopt` 的 persistent 版本，所有实验性
kernel 与 `reproduce_compare.sh` case 均已撤销，正式实现继续保持原 exactopt。

撤销后在 a07-3 复测：

```text
run directory:
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-a07-3_20260912T100754Z

random logits_diff = 3.3980e-06
random rel_l2      = 2.6069e-03
random hash        = aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
GEMM1              = 534.775 us
MOE e2e            = 1374.92 us
```

该结果与本轮同机 exactopt 起点 `532～534 us` 一致，确认正式代码已恢复。

当前进一步优化的结构性难点是：

1. dynamic persistent loop 让 kernel 达到 `107 SGPR`，超过 CDNA5 文档给出的
   106 个普通 SGPR，persistent-only 已产生 32 次 SGPR spill；
2. 加入 next-tile descriptor/prefetch 后 spill 增至 53 次，且代码路径增加大量
   `s_wait_loadcnt`；
3. input ring 已占 272 KiB，完整 output arena 无法与其同时常驻 320 KiB LDS；
4. half-output arena 的重映射在当前 FlyDSL/TDM lowering 下没有保持逐字节等价；
5. input/output 共用 TDM，新增同步使 `s_barrier_wait` 高于被隐藏的 startup latency。

若继续这一方向，需要重新实现一个更低层级、显式控制 SGPR lifetime 和 LDS/TDM
descriptor 的专用 GEMM1 kernel，并先单独证明 half-output TDM layout 的逐字节
正确性；继续调整现有 scheduling knob、prefetch 深度或 epilogue batch 没有足够
依据达到剩余约 9% 的目标。

## 远端异常记录

d01-3 上一次已撤销的 raw-pointer ABI 实验导致 verify kernel hang，留下：

```text
PID 219155
container: hyg_fyd1
VRAM: about 10.2 GiB
```

已报告该残留进程；按机器规则未在没有明确授权的情况下终止它。随后再次连接
d01-3 时 SSH 超时，因此当前无法确认该 PID 是否仍存活，也无法完成 d01-3 的
最终远端状态核验。
