# v80 相对 F3 baseline 的 GEMM1 优化说明

本文件说明本轮 `hyg_dsr1_conc6144` 优化中，正式保留的 v80 相对起点 F3 改了什么、为什么这样改，以及已有证据支持到什么程度。内容来自保存的源码、实际 ELF/ISA、完整 MoE 正确性记录和性能归档。整理日期为 2026-10-29；初稿完成后，按用户授权在 a07-3 重启并验证容器 SDK 的 `rocm-smi` 后完成了两版同期复测。

最新 E64/T1536/topk8/shared0、pin0、LOG0/W2/I101 复测中，F3 baseline 的 GEMM1 三次中值为 **83.387657 μs**，v80 为 **81.120394 μs**，减少 **2.267263 μs（2.718943%）**。六轮原精度检查全部通过，完整数据、测量条件和独立复核见第 9.5 节。

v80 的主要变化是：**将逻辑 M 方向的 16 行块交错分配给两组 M-wave，在整个 WG 进入 K loop 前选择 4/6/7/8 档有效行覆盖，并通过六/七档路径邻接和一次公共 epilogue 调整机器码布局。** 前两项减少 padding 的 WMMA 和 A/ScaleA LDS 读取，最后一项改善已观察到的加载地址快慢分档。有效行的计算、量化结果和原有输入/输出格式保持。

## 1. baseline 和 v80 的准确身份

本轮接手时的 baseline 是 `6f656cf5b02b2f1acf723949652f91b48e2933d7 / F3`。更早 `codex/dsr1_e65_gemm1_20261017` 阶段的 E64 baseline `99.0062 μs` 属于另一轮任务，不是本文的起点。

| 项目 | F3 baseline | 当前正式保留的 v80 |
|---|---|---|
| 分支 | `hyg_dsr1_conc6144` | `hyg_dsr1_conc6144` |
| 代码身份 | commit `6f656cf5b02b2f1acf723949652f91b48e2933d7` 中的 kernel | v80 source hash 对应的 kernel；初稿核对时工作区 HEAD 为 `e99dbac65a92ed23f87df4d99bd38f655a923a3b` |
| 文件 | `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` | 同一文件 |
| source SHA256 | `f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec` | `93165bbb8acab32dfe0faa9759c76adc8c2408f8f84998dcf3b8879ce6b8d312` |
| 对照 snapshot | Git 中的上述 blob | `.codex_tmp/hyg_f3_followup_20261024/snapshots/f3_v3_likely_pair_single_epilogue_layout_v80.py` |

2026-10-29 初稿的只读核对确认：生产文件与 v80 snapshot 逐字节相同；F3 与 v80 的文件顶层函数参数签名相同。初稿核对时的工作区 HEAD 不是“v80 已经提交”的证明，v80 的身份由上述源码 hash 确定；后续提交及复测状态另行记录。

随后发布并远端 pull 的源码 commit 为 `cecc19e93bf112cb1a1670431c80a7dec5a2dc46`，英文 commit message 为 `Optimize gfx1250 GEMM1 row coverage and restore fused A-preshuffle path`。第 9.5 节复测以该 HEAD 为工作区基准，轮流使用上述 F3 Git blob 和 v80 source hash；测试结束后恢复正式 v80。

交接时记录的 GEMM1 baseline 如下：

| 版本与参数 | 三次 GEMM1 μs | 中值 μs | 使用方式 |
|---|---|---:|---|
| 原始 F3，`pin=0`，`--warmup 5 --iters 20` | 82.3413 / 81.7383 / 81.4917 | **81.7383** | 本轮起点的历史 baseline |
| 原始 F3，`pin=1`，`--warmup 5 --iters 20` | 81.0487 / 80.7177 / 80.7452 | 80.7452 | 交接时的 pin1 对照 |
| 后续行尾 v2，`pin=0`，`--warmup 5 --iters 20` | 82.1558 / 81.4642 / 82.0483 | **82.0483** | 切换 pin0 后继续优化所用的阶段基准 |

第三行已经包含行尾 v2 的优化，不能称为起点 commit 的原始 F3。后续文档中“baseline”也曾指同期 v60 或当前 v80 reference，阅读数值时需要同时确认版本。

以上历史测试使用旧计时参数；交接命令还使用 `AITER_LOG_MORE=1`。当前要求为 `pin=0`、`AITER_LOG_MORE=0 --warmup 2 --iters 101`。不能把不同参数、不同 boot 或不同批次的数值拼接成严格的优化降幅。

原始记录见 [交接文档](moe_gemm1_hyg_dsr1_handoff_20261024.md) 第 1、4 节，以及 [后续优化记录](moe_gemm1_hyg_dsr1_f3_followup_20261024.md) 的行尾 v2/pin0 基准记录。

## 2. 目标计算与 F3 已经具备的能力

本文当前使用的 workload 是 E64/T1536/topk8/shared0、model_dim=7168、inter_dim=2048，A4W4、SiLU、无 bias、random non-balanced routing；保存记录中的输入参数为 `--data-init uniform --scale-init auto --seed 0`，并另有 seed17 正确性及诊断对照。

平均每个 expert 的路由行数为：

```text
1536 × 8 / 64 = 192
```

non-balanced routing 的实际 expert 行数围绕这个平均值变化。一个 256 行 M tile 往往包含较多 padding，因此优化重点是按运行时有效行数安排计算。

F3 已经有原来的 256×256×256 tile、四级 LDS buffer、TDM input prefetch、carried k128、rolled drain、cluster 同步、inline-zero accumulator、scaled WMMA，以及融合 activation 和 MX quant。v80 保留这些基础结构。

本轮 v80 使用既有的 non-persistent GEMM1 路径，`persistent_workers=0`。它不是此前四卡任务中的 persistent `t192/w2x4/b4`，也没有通过修改 drain 规则取得这里的收益。

## 3. 最终 v80 的来源：哪些改动真正保留了

最终源码的主要来源关系为：

```text
F3
  → 行尾 v2
  → M16 交错分配和 WG-uniform 的 4/6/7/8 档覆盖
  → WG pin0 boundary8/v3
  → v76：调整四/八与六/七路径的 selector 顺序
  → v80：likely 六/七档 selector + 一次公共 epilogue
```

本文的 v3 专指 `f3_m16_wg_uniform_pin0_boundary8_v3.py`，source SHA256 为 `612308136fba5459a9c3120fee93ead5f446c37c5fa0557acf8a9b69afce6f95`。历史上其他 family 也有 v3 编号，应以 snapshot 名称和 hash 区分。

| 最终保留的变化 | 对计算或调度的作用 | 性质 |
|---|---|---|
| M16 行块交错分配 | 将有用行更均匀地分配到两组 M-wave | 行映射优化 |
| WG 外层选择 4/6/7/8 档 | 跳过覆盖范围之外的 padding WMMA 和 A/ScaleA LDS 读取 | 无效计算消除 |
| A/ScaleA 与 epilogue 行地址同步重映射 | 保证交错后的操作数、量化 payload 和 scale 仍对应原物理行 | 必要的正确性适配 |
| 按 `active_m_rep` 计算 LDS/WMMA scheduling 数量 | 让较短 body 的 scheduling hints 与实际计算相符 | 调度适配 |
| 六档 wait8、七档 wait0、pin0 八档 loader boundaries | 保护缩短计算距离后的寄存器 retirement 和自动分配 | 必要的依赖处理 |
| 六/七档邻接、likely hint、一次公共 epilogue | 改变热点代码与公共尾部的相对位置，降低已观察到的地址敏感性 | code layout 优化 |

**v45 的 dense LDS 地址简化和 v60 的 ScaleA/ScaleB 与 payload LDS 读取穿插，不在最终 v80 中。** 它们属于以行尾 v2 为基础的另一条实验路线。v80 是回到 v3 的计算结构后继续做布局修复，不能描述成把 v45、v60、v80 的全部改动依次叠加。

当前 v80 loader 仍先读取 ScaleA/ScaleB，再交错读取 A/B payload；v60 则把 scale 读取分散到 payload 读取之间。两者的源码不同，v60 的历史收益不能直接计入 v80 的单项收益。

## 4. M16 行块交错：为什么适合 192 行附近的 routing

### 4.1 F3 的 M-wave 分工

源码的逻辑 M 方向按 16 行块组织。目标 geometry 有两组 M-wave，每组还包含 N 方向的 wave；下面只比较 M 方向的分工。

F3 的行映射为：

```python
wmb = wave_m * warp_tile_m
row_rel = wmb + wm * 16 + lane16
```

对于 `tile_m=256`、`m_warp=2`，`warp_tile_m=128`：

| M-wave | 分配的物理 M16 块 | 分配的行区间 | tile 有 192 个有效行时的有效块数 |
|---|---|---|---:|
| `wave_m=0` | 0–7 | 0–127 | 8 |
| `wave_m=1` | 8–15 | 128–255 | 4 |

F3 的 mainloop 仍按完整覆盖计算；虽然有效行分布为 8/4 个块，两组 M-wave 都执行完整 8 档 WMMA。后来的行尾 v2 能让后一组跳过整段无效计算，但产生 8/4 的执行量差异，需要共同到达的同步点仍受较长一组影响。

### 4.2 v80 的交错映射

v80 在已有 geometry guard 内使用：

```python
M_ROW_STEP = 32
wmb = wave_m * 16
row_rel = wmb + wm * 32 + lane16
```

等价的物理 M16 块编号是：

```text
p = 2 × wm + wave_m
```

| M-wave | 分配的物理 M16 块 | tile 有 192 个有效行时的有效块数 |
|---|---|---:|
| `wave_m=0` | 0、2、4、6、8、10、12、14 | 6 |
| `wave_m=1` | 1、3、5、7、9、11、13、15 | 6 |

这样既能跳过尾部 padding，又能让两组 M-wave 各计算六个 M16 块。256 行完整 tile 仍由两组各八个块覆盖。

这项优化依据运行时 expert 行数，保持原 routing 和所有有效 token×expert 组合。它不依赖 const0 输入，也不把 non-balanced routing 转换成 balanced routing。

代码位置：[kernel 文件](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py) 中的 `interleaved_m_rows`、`M_ROW_STEP`、`wmb` 和 `load_a`。

## 5. WG 外层的 4/6/7/8 档：跳过什么，保留什么

进入 mainloop 前，以本 expert 从当前 tile 起点开始的剩余有效行数 `r=mn_oob` 选择覆盖档位。目标路径的 `M_ROW_STEP=32`，所以四个档位覆盖 128/192/224/256 行。

| 当前 tile 的有效行范围 | 每组 M-wave 的 `m_reps` | WG 覆盖行数 | 相对完整八档的 WMMA 数量减少 |
|---|---:|---:|---:|
| 1–128 | 4 | 128 | 50% |
| 129–192 | 6 | 192 | 25% |
| 193–224 | 7 | 224 | 12.5% |
| 225–256；或 expert 后面还有更多行 | 8 | 256 | 0% |

空的 padding tile 沿用原来的 `has_work` 处理。四档是最小档，因此小于 128 行时仍可能计算一部分 padding；这不是逐行精确裁剪。

表中的比例是 WMMA 数量的源码推导，**不是 kernel 耗时降幅**。目标 K=7168 的全部 K 范围仍参与有效行计算；B 的 N/K 范围、全局 TDM tile 和必要同步也保留。

mainloop 内相应使用：

```python
active_m_rep = m_reps
compute_n_acc = active_m_rep * wmma_n_rep
sa_pairs = (active_m_rep + 1) // 2
STATE_DS = active_m_rep * DS_A + BS_DS
```

`mma_rows`、A LDS loader 和 ScaleA LDS loader 只处理该档位需要的 M 块，WMMA scheduling 和 DS scheduling 的计数也随之调整。原来的 scheduling 上限和规则继续使用，只把数量基础从完整 `n_acc` 改成 `compute_n_acc`。

重要区别是：**档位选择在整个 WG 的 K loop 外完成。** 同一 WG 的所有 wave 使用相同的档位和 TDM/barrier 序列，避免一部分 wave 在 K loop 中途跳过另一部分 wave 仍需参加的同步。

四档 body 同时存在于同一个编译 kernel 中，按 `mn_oob` 运行时选择。优化没有增加按 expert 数量、token 数量或随机 seed 单独生成的专用 body。

代码位置：`run_mainloop`、`load_lds_data`、`emit_hints` 和末尾的 coverage selector。

## 6. A、ScaleA、量化输出同步重映射

M 行交错以后，只改 A 的地址会使 scale 或输出行与对应 accumulator 错位，因此 v80 同时修改了三处地址关系。

### 6.1 A payload

`wmb` 改为 M-wave 的 16 行偏移，`load_a` 的 M 方向步长改成 `M_ROW_STEP`。仍从原有 preshuffled A/LDS 布局读取每个有效物理行的同一组 K 数据。

### 6.2 输入 ScaleA

ScaleA 保留原 `(M128, k128, M16, lane16)` 排列。交错路径重新计算 LDS lane base：

```python
(wave_m * 16 + lane16 + (lane // 16) * 32) * 4
```

并在 `load_sa` 中使用对应的 M16 块偏移。核对时要按 WMMA 实际选择的 scale lane 检查，而不是把发起 A load 的 lane 直接当成 scale lane。

保存的源码审计记录包含 98,304 项 A operand dword 地址、12,288 项所选 ScaleA dword 地址、4,097 项行覆盖，以及 7,168 项 quant 行/scale 布局检查。这些是地址和覆盖的核对，完整设备数值验证另行完成。

硬件依据是 CDNA5 ISA §7.12.6.1.3 和 §7.12.6.2：FP4 的矩阵元素 lane/VGPR 排列，以及 `SCL_OPSEL[0]`、`SCL_OPSEL_HI[0]` 对 lanes 0–15/16–31 的 scale 选择。

### 6.3 输出 payload 和下一阶段 ScaleA

epilogue 使用同一 `row_rel = wmb + wm * M_ROW_STEP + lane16` 找到输出物理行。目标交错路径不再套用“一个 M-wave 对应一个连续 quant tile”的 shortcut，而是沿用原有一般表达式：

```python
scale_tile = row_i32 // QUANT_ROWS_PER_TILE
row_in_tile = row_i32 - scale_tile * QUANT_ROWS_PER_TILE
```

一般表达式在 F3 中已存在。v80 改的是交错行时的选择条件，保留原有非二次幂 quant row shape 的能力，并未把 96/192 等支持包装成新功能。

原 epilogue 已有的无效 M16 块跳过、最后一个块的 `row_valid`、OOB descriptor、scale/payload 格式和量化数学继续保留。收益主要来自 mainloop 更早跳过 padding；单纯 epilogue 跳过无效块并不是 v80 才增加的能力。

## 7. 缩短 body 后的 retirement 与 compiler boundaries

有效 M 块变少后，相邻 ScaleA/ScaleB 寄存器复用之间的 WMMA 指令距离变短，原来依靠完整 body 留出的 retirement 距离不能直接照搬。

v80 保留 v3 路线已经检查过的处理：

| 路径 | 保留的依赖处理 | 用途 |
|---|---|---|
| 六档 | scale 预取前显式 `s_wait_alu depctr_va_vdst(8)` | 保护前一次 scale 使用与当前 slot 复用 |
| 七档 | 显式 `s_wait_alu depctr_va_vdst(0)` | 覆盖更短 body 下的 retirement 依赖 |
| pin0 八档 | `load_lds_data` 前后 `rocdl.sched_barrier(0)` | 约束 LLVM 调度，保护两套输入 slot 的自动寄存器分配 |

`depctr_va_vdst(8)` 中的 8 是等待计数条件，不是“等待 8 cycles”；`depctr_va_vdst(0)` 是等待相关依赖清空，不是取消等待。`rocdl.sched_barrier(0)` 是 compiler scheduling boundary，不能与 workgroup barrier 或 TDM completion wait 混为一谈。

Shader Programming Guide §4.3.7.4.1 规定，`SCHED_MODE 2` 下软件负责处理 `VA_VDST`、`VM_VSRC` hazard。该 mode 在 F3 中已经使用；这里新增的 waits/boundaries 是对缩短 body 的适配。

实际 ELF 经原 bounded DS/WMMA 模型及 `original`、`WMMA_only` retirement stress 检查均为 0 issue，完整 MoE 数值验证也通过。该结果覆盖保存模型和用例的范围，不是对所有硬件行为的完整模拟。

## 8. v80 最关键的布局改动：六/七档邻接和一次公共 epilogue

### 8.1 为什么继续优化 v3 的布局

v3 在相同 ELF 和相同输入下出现过随实际 GPU code load base 改变的快慢档。固定 ELF、输入和原 timer 的配对正常计时，以及独立 `SQC_ICACHE_MISSES` counters，支持 instruction-cache 行为与这类地址差异有关。

这与“两个随机 seed 的有效行分布不同”是两个因素：同一个 seed/输入内的地址差异需要控制加载地址；不同 seed 的工作量和路径差异需要另行保留，不能混为一个解释。

### 8.2 v80 的 selector

v80 对原 guard 内的 pin0 交错路径使用以下等价选择逻辑；实际源码用 `llvm_dialect.intr_expect` 给第一项 predicate 一个 likely 提示：

```python
if likely(128 < r <= 224):
    if r <= 192:
        run_mainloop(m_reps=6)
    else:
        run_mainloop(m_reps=7)
else:
    if r <= 128:
        run_mainloop(m_reps=4)
    else:
        run_mainloop(m_reps=8)

emit_common_epilogue()
```

`likely` 在这里是布局提示，返回的布尔值与原 predicate 相同。它不改变四个区间的语义，四/八档仍正常计算所有需要的行。六/七档对应 192 行附近常见的执行路径，放在一起有利于编译器安排它们的相对位置。

### 8.3 为什么将 epilogue 放到 selector 后只调用一次

之前的 v79 试验保留相似 selector，但不同源码分支各调用 epilogue。实际编译结果复制了部分量化代码，合法 seed17 的耗时回退，因此没有采用。

v80 将相同的 epilogue 定义为 `emit_common_epilogue`，在 selector 后只调用一次。2026-10-29 的独立只读重建确认：

- v80 的 epilogue AST 与前身 v76 的原 epilogue AST 相同。
- 将 selector 和公共 epilogue region 恢复成 v76 后，整个源码逐字节恢复成 v76。
- 该 region 之外的所有计算源码与 v76 相同。

实际编译结果如下，采用的是保存的固定 pin0 ELF：

| 项目 | 未采用的 v79 | v80 |
|---|---:|---:|
| `.text` bytes | 115712 | **97408** |
| 静态 WMMA sites | 2000 | 2000 |
| 静态 `v_cvt_scalef32_pk8_fp4_bf16` sites | 61 | **32** |
| 静态 `tensor_store_from_lds` sites | 1 | 1 |
| VGPR / SGPR | 799 / 60 | 799 / 60 |
| LDS bytes | 327680 | 327680 |
| private / VGPR spills / SGPR spills | 0 / 0 / 0 | 0 / 0 / 0 |

这证明 v80 减少了 v79 的机器码复制，**不代表动态量化计算减少了 61→32，也不代表所有计算都同时驻留 I-cache**。

尤其要注意：v80 的 `.text=97408 B` 与原 v3 总大小相同。相对 v3，不能声称“把整个 kernel 缩到 64 KiB 以下”；改善主要来自实际路径的相对布局和公共尾部安排。相对失败的 v79，代码量确实减少。

v80 固定 GEMM1 ELF SHA256 为 `49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8`，loaded `.text` SHA256 为 `739d3cc9c731c8378cb758b6ed5af8fc0978ae5795878a6ba1152ca2a55f1f7d`。

### 8.4 硬件资料支持与实测边界

本地 `MI400_SQC_Miscellaneous_Feature_MAS#21.txt` §2/Table1 记载 MI400 I$ 为 64KB、4 banks、4 ways、32 sets，cache line 为 32 DW，并给出：

```text
Set[4:0] = address[13:9] ^ address[18:14]
```

该公式支持检查 code base 与热点 PC 的相对 set 映射。本文没有从公式唯一推导实际 bank hash、live residency 或各类 miss 的 latency；这些尚未由保存证据确定。

v80 已完成有限范围的修复验收：同一冻结 ELF、E64 目标配置、seed0/17、实际 64 KiB 对齐 load base 的 group0–3、原 pin0 timer。它覆盖这些已验证组合，不是对任意加载对齐或所有随机 routing 都没有波动的保证。

## 9. 已有性能和 counters 证据

### 9.1 v3 与 v80：同输入、不同加载地址的正常计时

下面是保存的配对 process 的组内 GEMM1 中值，单位 μs；使用旧 `--warmup 5 --iters 20`。每个 process 内两版共用同一次生成的输入和指针，覆盖全部四个地址组，保留全部 target 的正/逆序计时。

`group = (public_HSA_load_base >> 16) & 3`，本组实际 base low16 为 0。

| process / seed | group | v3 GEMM1 μs | v80 GEMM1 μs |
|---|---:|---:|---:|
| process1 / seed0 | 0 | 79.264421 | 77.679868 |
| process1 / seed0 | 1 | 83.545447 | 77.366868 |
| process1 / seed0 | 2 | 78.716211 | 77.749500 |
| process1 / seed0 | 3 | 90.462105 | 77.911763 |
| process2 / seed17 | 0 | 89.619053 | 89.580132 |
| process2 / seed17 | 1 | 92.876816 | 89.873316 |
| process2 / seed17 | 2 | 89.894342 | 89.801632 |
| process2 / seed17 | 3 | 91.922737 | 89.172316 |
| process3 / seed0 | 0 | 78.417289 | 78.482605 |
| process3 / seed0 | 1 | 82.284079 | 77.428421 |
| process3 / seed0 | 2 | 78.929684 | 77.574947 |
| process3 / seed0 | 3 | 90.536368 | 77.870132 |

seed0 中明显的地址慢档收窄；seed17 的四组中值仍约 89–90 μs，因此不能把 seed0 的 77–78 μs当成所有 random routing 的性能。process3/group0 也如实保留 v80 小幅更慢的结果，没有按读数挑选实例。

### 9.2 独立 counters：改善确实伴随 miss 放大下降

这是独立 SDK capture 中 `SQC_ICACHE_MISSES` 的事件中值，不是 μs，也不是精确 hit rate。

| seed | group | v3 miss 中值 | v80 miss 中值 | 中值降幅 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 5875 | 0 | 100.00% |
| 0 | 1 | 26122 | 474 | 98.19% |
| 0 | 2 | 5875 | 0 | 100.00% |
| 0 | 3 | 29109.5 | 3081 | **89.42%** |
| 17 | 0 | 14386.5 | 10476 | 27.18% |
| 17 | 1 | 32089.5 | 11301.5 | 64.78% |
| 17 | 2 | 14128.5 | 10452.5 | 26.02% |
| 17 | 3 | 30435 | 11567.5 | **61.99%** |

计数含义依据本地 `MI400_SH_PerfCounter_Users_Guide#71.txt` §5.1 的 non-duplicate miss 定义。正常 timer 与 instrumented counter capture 分开，counter capture 的耗时没有用于正式性能成绩。

### 9.3 与阶段 reference v60 的原入口比较

2026-10-28 的同一 boot 中，保存的前置 v60、v80、后置 v60 原入口 GEMM1 中值如下；均为旧 `--warmup 5 --iters 20`：

| 版本 | GEMM1 中值 μs | v80 相对该 reference 的耗时降幅 |
|---|---:|---:|
| 前置 v60 | 83.118789 | 1.935277% |
| v80 | **81.510211** | — |
| 后置 v60 | 83.656947 | 2.566119% |

该表说明 v80 在原入口的孤立 GEMM1 比阶段 v60 reference 更快。它不是 F3→v80 的单项消融试验，也不能把不同实验阶段的收益百分比相加。

2026-10-28 按用户更新后的“只以 GEMM1 最低耗时选版，保留完整 MoE 正确性”要求，正式源码恢复并保留 v80。此前 MoE e2e 排名曾阻止采用；最新采用依据以 GEMM1 为准。已有 e2e 结果保留在原记录中，本文不把它作为 GEMM1 排名条件。

### 9.4 此前中断的 W2/I101 记录

初稿整理前的中断批次中，v80 三条有效 reference 记录为：

| 次数 | GEMM1 μs |
|---:|---:|
| 1 | 80.52848958333333 |
| 2 | 81.22617346938779 |
| 3 | 81.25989898989893 |
| 三条中值 | **81.22617346938779** |

条件为 E64/T1536/topk8/shared0、pin0、A-preshuffle1、`AITER_LOG_MORE=0 --warmup 2 --iters 101`。随后 v101 的第五条记录未通过测后 ownership 门禁，整个比较中断，第六条未执行；v101 没有三次有效中值，没有采用。

2026-10-29 已下载该中断归档，并独立核验全部 246 个成员及五条原始记录。原 summary 和每条 valid 状态保持；整批 `performance_data=false`、`candidate_selected=false`，三条 v80 reference 观测不冒充一次完整候选采用验收或 pull 后最终成绩。

归档 SHA256 为 `8052861f5fb366dc3d83c08b26b88462899ba3413d0238ea0a2cfffd76e1ccc9a`。初稿整理时按用户“先不测性能”的要求没有补测 F3，也没有补跑 v101。用户随后授权的 baseline/v80 复测另建了第 9.5 节批次，保留本批次的原始状态和数据。

初稿时尚未完成 LOG0/W2/I101 下原始 F3 与 v80 的同期配对。原始 baseline 的 81.7383 μs和本节的 81.2262 μs分别保留其参数与记录身份；当前同期对比使用第 9.5 节的六条新样本。

### 9.5 2026-10-29 重启后、pull 后的最终同期复测

用户要求“baseline 和 v80 的性能再测一下做个对比”。本次仅复测已有两版，优化工作已经结束。测试在 a07-3 的 `dsr1_ep4` 中执行，工作区为 `/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`，分支 `hyg_dsr1_conc6144`，源码 HEAD 为 `cecc19e93bf112cb1a1670431c80a7dec5a2dc46`。

固定条件为 E64/T1536/topk8/shared0、model_dim=7168/inter_dim=2048、A4W4/SiLU、无 bias、random non-balanced routing；`pin=0`、A-preshuffle=1、`AITER_LOG_MORE=0 --warmup 2 --iters 101`、`--seed 0 --data-init uniform --scale-init auto`。两版都调用原 `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py` 的 `run_moe`，使用原 `aiter/test_common.py` kernel timer 和原 `logits_diff < 0.01` 门限。

六轮执行顺序在运行前冻结为 F3 / v80 / F3 / v80 / F3 / v80。每版三次全部保留并取中值；没有选择加载地址、替换慢样本、改变 seed 或补挑快实例。下表显示到小数点后六位，完整精度数值保存在原 summary 和本地独立 review 中。

| 版本 | 第一次 GEMM1 μs | 第二次 GEMM1 μs | 第三次 GEMM1 μs | 三次中值 μs |
|---|---:|---:|---:|---:|
| F3 baseline | 84.008918 | 83.387657 | 83.123265 | **83.387657** |
| v80 | 80.904896 | 81.120394 | 81.942660 | **81.120394** |

```text
耗时减少 = 83.38765656565648 - 81.1203939393939
         = 2.2672626262625784 μs
耗时降幅 = 耗时减少 / baseline 中值 × 100
         = 2.7189427304236764%
```

每轮性能前全机空闲检查连续通过 11 次，间隔 6 秒、跨度至少 60 秒；计时期间没有并发 GPU 状态查询、传输或编译。六个测试 child 均在约 7.98–8.34 秒内正常退出，受 `TERM 118s + KILL 2s` 的 120 秒硬上限保护。测后自有测试进程均已退出，最终 `settled.idle=true`，生产文件已逐字节恢复为正式 v80。

六轮原 kernel 检查和完整 MoE eager 精度检查均通过，`logits_diff=3.6105513278025183e-06 < 0.01`，route ids/weights fingerprint 全部一致。本次记录 GEMM1 kernel 耗时；完整 MoE 使用 `bench=False, kernel_bench=False` 验证精度，没有测量 MoE e2e 时间。证据字段 `graph_metrics` 对应这次完整 MoE eager 检查，完整 MoE CUDA graph 验证不在本次复测范围内。

两版 GEMM1 的实际 ELF SHA256 分别为：

```text
F3 : 62037bcfa621ff1c2d378e0e5face1479e6af47fbcb5beb8bacf8ee39d003897
v80: 49a8d5bfb280150bcddce335cba65e398d67d8a1f53f4d0424e9d579474fdaf8
```

两版 GEMM2 ELF 相同，为 `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be`；本节仅比较 GEMM1 耗时。

重启后的 boot_id 为 `7e035a22-f1ba-411b-be47-b404092abe68`。主机旧 `/usr/bin/rocm-smi` 在 `GetGPUMetricsFormat1` assertion，未修复；恢复 GPU 工作前，已确认容器 SDK 的 `/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin/rocm-smi --showuse --showmeminfo vram --json` 正常返回，一张 GPU、利用率 0%、VRAM used 176582656 B。两版参考 matmul 均设置 `TORCH_BLAS_PREFER_HIPBLASLT=0`，保留原参考数学和精度门限；性能测试没有设置 `AMD_SERIALIZE_KERNEL`。

批次 label 为 `f3_E64_F3_vs_v80_LOG0_W2_I101_after_pull_20261029_v4_rocblas_boot7e035a22`，以下记为 `LABEL`。前一 boot 的等待批次保留为中断记录，没有产生性能样本。

| 证据 | 路径或 SHA256 |
|---|---|
| frozen plan | `TASK/logs/LABEL.frozen_plan.json` |
| plan SHA256 | `01df1baef6f0cb79feabcdf87d0c1bcbbf4bbbcc0173341bd5546f488245fe48` |
| 原 controller 与六轮 summary | `TASK/logs/LABEL.controller.log`、`TASK/logs/LABEL.summary.json` |
| 证据归档 | `TASK/logs/LABEL_evidence.tar.gz`，266 个成员、8784857 B |
| archive SHA256 | `ffc10fc776399dd95722a77dcecf6d290ebb858d13f3c3c988e4a173452d5aec` |
| 本地独立复核 | `TASK/logs/LABEL.independent_local_review.json`，`passed=true` |
| 重启后 rocm-smi 验收 | `TASK/logs/f3_ROCM_SMI_ready_after_reboot_20261029_v3_sdk_boot7e035a22.json` |

本地 reviewer 重新验证全部归档 SHA256、六轮原 valid 方程、caller/timer 与实际 ELF/loaded text 绑定、原 precision gate、相同 route fingerprint、成功 rocm-smi receipt，并独立重算上述中值和降幅。所有检查通过。正式 kernel 保留 v80，本次没有新增 E65/topk9 测试。

## 10. 功能、精度和支持范围

完整 MoE 正确性是保留 v80 的条件。保存的 E64 seed0/17 输出 shape 均为 `[1536,7168]`，各 22,020,096 B，与已验证 reference 逐字节相同。

seed0 的原精度结果为：

```text
logits_diff = 3.6105513278025183e-06
rel_l2      = 0.0026872099842876196
原 logits_diff gate = 0.01
```

v80 保留原 precision gate，以及 activation、amax/E8M0、BF16 舍入和 FP4 转换数学。kernel 函数接口、输出 payload/scale 布局、有效 expert 路由和原 fallback 语义保持。

交错覆盖的 guard 基于已有的 A4W4/SiLU/quant/preshuffle 和 pipeline geometry：`tile_m=tile_n=tile_k=256`、`m_warp=n_warp=2`、`num_buffers=4`、`cluster_n=4`、`cluster_m=1`、non-persistent、next-stage prefetch、足够深的 K loop，以及非 row-major ScaleA、非 ScaleA prologue、`K % tile_k == 0`。条件外继续使用原路径，布局提示进一步限定到 pin0。

这说明优化没有新增 expert-count/seed specialization；**它不等于已经完成 E64/E65 共用同一个 compiled GEMM1 的验收。** 历史 E65 routed fallback 正确性仅证明当时的 fallback，不能替代共用 kernel 验收。用户已明确禁止后续所有 E65/topk9 测试，本次没有新增该规模测试。

## 11. 复现入口与证据索引

### 11.1 只读查看 baseline 与源码差异

在本地 `aiter_dsr1_e65` 根目录执行以下 Git 读取命令：

```powershell
git show 6f656cf5b02b2f1acf723949652f91b48e2933d7:aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
git diff 6f656cf5b02b2f1acf723949652f91b48e2933d7 -- aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
```

### 11.2 当前要求下的原测试入口

以下是原入口的单独 CLI 示例。第 9.5 节实际通过冻结计划的 checker wrapper 调用相同原 caller/timer，完整调用记录保存在 controller/summary 中。以后复现时，先在 a07-3 主机确认全机 GPU 空闲，再进入 `dsr1_ep4`；每次使用独立 cache 名称，并确认实际导入当前 v80 源码。

```bash
cd /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
unset COMPILE_ONLY FLYDSL_RUNTIME_RUN_ONLY AITER_FLYDSL_MMA_GROUP AITER_FLYDSL_MMA_FIRST_GROUP AITER_TDM_B_TH AITER_FLYDSL_DS_FIRST_N
export AITER_GROUPED_A4W4_A_PRESHUFFLE=1
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0
export FLYDSL_RUNTIME_CACHE_DIR="$PWD/.codex_tmp/v80_repro_YYYYmmdd_HHMMSS"
export TORCH_BLAS_PREFER_HIPBLASLT=0

PYTHONPATH=$PWD ENABLE_CK=0 AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 \
AITER_MOE_EXPERT_BALANCE=false AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_LOG_MORE=0 FLYDSL_DUMP_IR=0 AITER_FORCE_GFX1250=1 GPU_ARCHS=gfx1250 CU_NUM=256 \
timeout --signal=TERM --kill-after=2s 118s python3 -u \
op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
--scenario kernel --data-format a4w4 --act silu --no-bias --no-check-aot-cache \
--experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 \
--seed 0 --data-init uniform --scale-init auto --warmup 2 --iters 101
```

正式归档还需要原 caller 的 timed partial/cache key/IR/实际 ELF 绑定、完整 MoE 精度，以及运行前后 ownership 检查；入口命令本身不替代这些验证。原 `--scenario kernel` 会输出其他列，本任务的选版依据为 GEMM1。

### 11.3 主要证据

以下路径中的 `TASK` 为仓库根目录下 `.codex_tmp/hyg_f3_followup_20261024`。

| 内容 | 保存位置 |
|---|---|
| 初始 F3 版本和三次 baseline | [handoff](moe_gemm1_hyg_dsr1_handoff_20261024.md) |
| 各阶段源码、性能、失败与采用记录 | [followup](moe_gemm1_hyg_dsr1_f3_followup_20261024.md) |
| v3 波动诊断 | [诊断文档](moe_gemm1_v3_variability_diagnosis_20261025.md) |
| v80 来源和公共 epilogue 生成关系 | `TASK/make_f3_v80_single_epilogue_layout_local_20261027_v1.py` |
| v80 源码 plan | `TASK/logs/f3_v80_single_epilogue_layout_source_20261027_v1.plan.json` |
| 实际 ELF、native 和 layout 核对 | `TASK/logs/f3_v80_native_and_layout_local_20261027_v1.json` |
| 完整 MoE 精度独立核对 | `TASK/logs/f3_v80_precision_independent_local_review_20261027_v1.json` |
| 三次正常 process 和两次 counters 的有限地址验收 | `TASK/logs/f3_v80_complete_bounded_layout_decision_local_20261027_v1.json` |
| GEMM1 目标下的正式源码采用 receipt | `TASK/logs/f3_v80_GEMM1_only_adoption_20261028_v1.local.json` |
| 最新已有中断批次的独立 review | `TASK/logs/f3_E64_LOG0_W2_I101_v80_v101_screen_ABA_BAB_archive3_20261029_v3_boot47fbe50f_BOUNDED.CPU_window_independent_local_review.json` |
| 重启后 F3/v80 六轮同期复测 | 第 9.5 节的 `TASK/logs/LABEL.summary.json` |
| 六轮复测本地独立 review | 第 9.5 节的 `TASK/logs/LABEL.independent_local_review.json` |

硬件资料位于本地 `C:/Users/yanguahe/Documents/code/wk_sp1/mi400_hw_wiki/raw/papers/mi400_hd_txt/`：

- `MI450/amd-instinct-cdna5-instruction-set-architecture.txt`：§7.12.6.1.3、§7.12.6.2，正文 105–106 页。
- `architecture/subsystem/SH/MI400_Shader_Programming#65.txt`：§4.3.7.4.1，`SCHED_MODE 2` 的软件 hazard 责任。
- `block/sq/MI400_SQC_Miscellaneous_Feature_MAS#21.txt`：§2/Table1，I-cache 配置与 set hash。
- `architecture/subsystem/SH/MI400_SH_PerfCounter_Users_Guide#71.txt`：§5.1，ICACHE miss 事件定义。

本文件将源码中实际保留的变化、整版正常计时、counter 诊断和支持范围分别说明。各个机制没有一套新的独立同条件消融数据，因此不为每项改动虚构可节省的 μs。
