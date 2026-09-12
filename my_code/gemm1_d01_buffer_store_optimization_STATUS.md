# d01-3 GEMM1 output buffer-store 优化记录

## 目标与约束

本轮以 `sync_mg4_fc28_apre_exactopt` 为基础，只修改 GEMM1 kernel 的最终
output store 路径：尝试将 `TENSOR_STORE_FROM_LDS` 改为 vector
`buffer_store`，并尽量生成 `BUFFER_STORE_B128`。

必须保持以下条件不变：

- GEMM1 的接口、功能和精度等价；
- 保留每个 M tile 通过 `m_tile_map` binary search 查找 expert 边界的逻辑；
- 兼容 non-balanced token distribution；
- 不修改 MOE 中其他 kernel 的接口和功能；
- 若新版本性能低于 `sync_mg4_fc28_apre_exactopt`，不集成到
  `my_code/reproduce_compare.sh`。

## d01-3 性能起点

测试机器：`heliosr-1b114-d01-3`

测试时间：`2026-09-12`

测试命令：

```bash
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 bash my_code/reproduce_compare.sh
```

测试结果目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-d01-3_20260912T070701Z
```

| case | GEMM1 samples (us) | GEMM1 median us | GEMM1 vs 93665e | MOE e2e samples (us) | MOE e2e median us | MOE e2e vs 93665e | random pass | hash |
|---|---|---:|---:|---|---:|---:|:---:|---|
| baseline_93665e | 696.293, 693.641, 697.489 | 696.293 | +0.00% | 1678.87, 1675.06, 1679.60 | 1678.87 | +0.00% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc8 | 620.235, 618.273, 617.202 | 618.273 | +11.21% | 1599.21, 1598.49, 1598.76 | 1598.76 | +4.77% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg2_fc12 | 626.179, 630.101, 621.828 | 626.179 | +10.07% | 1608.78, 1611.41, 1597.38 | 1608.78 | +4.17% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28 | 614.509, 617.526, 622.296 | 617.526 | +11.31% | 1592.31, 1601.44, 1606.38 | 1601.44 | +4.61% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28_apre | 566.992, 564.325, 557.968 | 564.325 | +18.95% | 1452.49, 1449.36, 1439.79 | 1449.36 | +13.67% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28_apre_exactopt | 548.616, 548.625, 546.055 | 548.616 | +21.21% | 1430.63, 1429.57, 1431.85 | 1430.63 | +14.79% | True | `aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2` |
| sync_mg4_fc28_hard | 599.381, 596.906, 594.627 | 596.906 | +14.27% | 1580.33, 1580.80, 1574.75 | 1580.33 | +5.87% | True | `91f3c3c87e7e17e854bcc5c3dbb7032f0f5a039a033a205a1cba04206799b5ca` |
| sync_mg4_fc28_relu | 582.546, 578.444, 578.257 | 578.444 | +16.93% | 1558.80, 1566.47, 1555.67 | 1558.80 | +7.15% | True | `ca4a57024a6c0ee78991a0a2dcfd227852945fd356657fe64df596f62e98bdd4` |

本轮 buffer-store 实验的直接比较基准为：

```text
sync_mg4_fc28_apre_exactopt GEMM1 median = 548.616 us
sync_mg4_fc28_apre_exactopt MOE e2e median = 1430.63 us
```

## 实验 A：LDS staging 后 buffer-store

### 实现方式

实验版本保持 exact-SiLU 和现有两阶段 epilogue 不变，仅替换每个 half-output
的写回方式：

- activation 结果仍先写入原有 padded LDS output arena；
- 同一 `wave_m` 下的两个 `wave_n` 协作搬运一个 64-row half；
- 每个 lane 从 LDS 读取连续 16B，并用 `vec<4xi32>` raw buffer store 写入
  global memory，对应 `BUFFER_STORE_B128` 路径；
- 每行由 16 个连续 16B chunk 覆盖，保持 global store 合并；
- tail row 继续由动态 `mn_oob` 屏蔽，buffer descriptor 使用 runtime output
  byte extent 做硬件 OOB check；
- `m_tile_map` binary search、GEMM 主循环、exact-SiLU、接口以及其他 MOE
  kernel 均未修改。

### 正确性

在 d01-3 上使用 random input 验证通过：

```text
logits_diff = 3.39799e-06
rel_l2      = 0.00260689
pass        = True
output_sha256 = aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
```

hash 与 `baseline_93665e`、`sync_mg4_fc28_apre_exactopt` 的 random-input
基准一致。

### 同轮交错性能对比

为降低系统漂移影响，按 `TDM -> buffer -> buffer -> TDM -> TDM -> buffer`
顺序交错测试三轮。每轮测试前后均通过 `/data/yanguahe/code/gpu_users.sh`
确认全机没有其他 GPU/KFD 进程。

结果目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-d01-3_20260912T071226Z_exactopt_bufstore
```

| case | GEMM1 samples (us) | GEMM1 median us | vs same-run exactopt | MOE e2e samples (us) | MOE e2e median us | vs same-run exactopt |
|---|---|---:|---:|---|---:|---:|
| `exactopt_tdm` | 550.034, 547.035, 547.659 | 547.659 | +0.00% | 1434.39, 1432.92, 1432.25 | 1432.92 | +0.00% |
| `exactopt_lds_bufstore` | 573.376, 572.292, 578.834 | 573.376 | -4.70% | 1454.80, 1452.01, 1458.68 | 1454.80 | -1.53% |

其中同轮 TDM 版本的 median `547.659 us` 与本文件完整重测起点
`548.616 us` 只相差约 `0.17%`，说明两组测试处于一致的机器状态。buffer-store
版本相对完整重测起点的 GEMM1 回退约 `4.51%`。

### 结论

该方案性能低于 `sync_mg4_fc28_apre_exactopt`，因此：

- 不将该版本集成到 `my_code/reproduce_compare.sh`；
- 正式 kernel 源码已恢复为原 `sync_mg4_fc28_apre_exactopt` 实现；
- 仅保留本文档中的实验设计、正确性和性能结果。

回退的主要原因是显式 buffer-store 路径需要每个 lane 在两个 half 中合计执行
32 次 `ds_read_b128` 和 32 次 `buffer_store_b128`。原 TDM 路径由少量
`TENSOR_STORE_FROM_LDS` descriptor 直接搬运 LDS tile，已经能高效完成合并写回；
新增的 LDS read 指令、VMEM store issue 和 `STOREcnt` drain 成本超过了绕开
output TDM 所能获得的收益。

## 实验 B：VGPR 直接 buffer-store

### 实现方式

本实验按要求彻底去掉 output LDS staging 和 output TDM store：

- exact-SiLU 的 4 个 BF16 结果转换后继续保留在 VGPR 中；
- 每个 lane 的结果为两个 i32 dword；
- 使用两条 `DS_BPERMUTE_B32` 从 `lane ^ 16` 获取同一 row、相邻
  `kgrp` 的另外两个 dword；
- `kgrp == 0` 的 lane 将四个 dword 组成 `vec<4xi32>`，直接发出
  `BUFFER_STORE_B128`；
- 不再执行 output `ds_store_b64`、output `ds_read_b128`、output LDS
  barrier 或 `TENSOR_STORE_FROM_LDS`；
- tail row 仍使用动态 `mn_oob` mask，`m_tile_map` binary search、GEMM
  主循环、exact-SiLU、kernel 接口和其他 MOE kernel 均未修改。

这里的 `DS_BPERMUTE_B32` 只使用 LDS crossbar 做 wave 内寄存器交换，不读写
LDS memory，因此 output payload 从 exact-SiLU 结果到 global store 始终保存在
VGPR 中。

### 正确性

d01-3 random-input 验证通过：

```text
logits_diff = 3.39799e-06
rel_l2      = 0.00260689
pass        = True
output_sha256 = aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
```

### 同轮交错性能对比

测试仍按 `TDM -> buffer -> buffer -> TDM -> TDM -> buffer` 顺序交错三轮，
每轮前后检查全机 GPU/KFD 占用。

结果目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-d01-3_20260912T073133Z_exactopt_vgpr_bufstore
```

本节中的 `exactopt_bufstore` 指 VGPR-direct 实现。

| case | GEMM1 samples (us) | GEMM1 median us | vs same-run exactopt | MOE e2e samples (us) | MOE e2e median us | vs same-run exactopt |
|---|---|---:|---:|---|---:|---:|
| `exactopt_tdm` | 546.810, 549.763, 549.415 | 549.415 | +0.00% | 1434.22, 1430.60, 1432.56 | 1432.56 | +0.00% |
| `exactopt_bufstore` | 701.020, 704.012, 699.914 | 701.020 | -27.59% | 1580.85, 1585.95, 1579.80 | 1580.85 | -10.35% |

同轮 TDM median `549.415 us` 与完整重测起点 `548.616 us` 相差约
`0.15%`。VGPR-direct 版本相对完整重测起点的 GEMM1 回退约 `27.78%`。

### 回退原因与结论

虽然 VGPR-direct 版本完全消除了 output LDS 往返，但当前 accumulator mapping
天然是 row-per-`lane16`：每个 wave 为自己的 16 KiB output 需要发出 64 条
`BUFFER_STORE_B128`，每条还需要两条 `DS_BPERMUTE_B32` 拼接另一个
`kgrp` 的数据，即每个 wave 新增 128 条 cross-lane permute。

更关键的是，每条 `BUFFER_STORE_B128` 只有 `kgrp == 0` 的半个 wave 有效，
这些 lane 写相同 column chunk 的不同 row，lane 间地址相隔一整行
`6144 B`，无法形成 TDM tile store 的连续合并写回。cross-lane 指令和高度
strided global stores 的代价远大于省下的 LDS/TDM 操作。

因此 VGPR-direct `exactopt_bufstore` 也不集成到
`my_code/reproduce_compare.sh`，正式源码继续保留原
`sync_mg4_fc28_apre_exactopt` 的 TDM output store。

## 实验 C：VGPR 直接 `BUFFER_STORE_B64`

### 实现方式

本实验去掉实验 B 中的 `DS_BPERMUTE_B32`：

- 每个 lane 的 exact-SiLU 结果为 4 个连续 BF16，即 8B；
- 将这 8B bitcast 为 `vec<2xi32>`，直接从 VGPR 发出
  `BUFFER_STORE_B64`；
- 所有 32 个 lane 均参与 store；
- 完全不使用 output LDS，不执行 output LDS barrier，也不发出
  `TENSOR_STORE_FROM_LDS`；
- tail row 仍由动态 `mn_oob` mask 保护，其余接口、功能和精度保持不变。

### 正确性

d01-3 random-input 验证通过：

```text
logits_diff = 3.39799e-06
rel_l2      = 0.00260689
pass        = True
output_sha256 = aed13e2b195f531e4dc52010fa2b643b2d59d7ce18ab56c479cc599658f41db2
```

### 同轮交错性能对比

结果目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/heliosr-1b114-d01-3_20260912T074142Z_exactopt_vgpr_b64
```

| case | GEMM1 samples (us) | GEMM1 median us | vs same-run exactopt | MOE e2e samples (us) | MOE e2e median us | vs same-run exactopt |
|---|---|---:|---:|---|---:|---:|
| `exactopt_tdm` | 548.612, 548.794, 548.531 | 548.612 | +0.00% | 1431.22, 1431.89, 1430.60 | 1431.22 | +0.00% |
| `exactopt_bufstore` (`BUFFER_STORE_B64`) | 680.803, 687.624, 690.355 | 687.624 | -25.34% | 1560.73, 1571.98, 1574.18 | 1571.98 | -9.84% |

同轮 TDM median `548.612 us` 与完整重测起点 `548.616 us` 基本完全一致。
`BUFFER_STORE_B64` 相比实验 B 的 direct-B128 GEMM1 median `701.020 us`
改善约 `1.91%`，说明去掉 128 条/每 wave 的 `DS_BPERMUTE_B32` 确实有效；
但相对 TDM 仍回退 `25.34%`。

### 回退原因与结论

每个 wave 仍需执行 64 条 `BUFFER_STORE_B64`。当前 accumulator layout 中，
同一条 vector store 指令的 lane 主要对应不同 row：`lane16` 每增加 1，global
地址就跨越一个 `6144 B` row stride。虽然 `lane` 与 `lane ^ 16` 的两个 8B
store 最终覆盖同一 row 中相邻的 16B，但它们位于 wave 的两个不同半区，无法
形成 TDM 对连续二维 tile 的高效搬运。

因此瓶颈不是只有 `DS_BPERMUTE_B32`，根本问题是 accumulator 的
row-per-lane mapping 与 row-major global store mapping 不匹配。该版本不集成到
`my_code/reproduce_compare.sh`，正式源码继续使用原 TDM output store。
