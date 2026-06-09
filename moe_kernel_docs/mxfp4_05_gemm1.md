# MXFP4 M7 — `gemm1::kernel`（GEMM1：gate/up + SiLU·mul + 中间量化）

源码：`aiter/csrc/kernels/mxfp4_moe/gemm_a4w4/gemm1_a4w4.cuh:28`（kernel）/ epilogue `common/mxfp4_epilogs.hpp:21`
host op：`aiter.mxfp4_moe_gemm1_a4w4(..., kernelName=kernelName1)`（`fused_moe.py:1186`）
trace 名：`void aiter::mxfp4_moe::gemm1::kernel<655360, 385, 7168, 1024, {16|32|128}, true|false, true|false, 0>`
（模板参数 = `<MAX_M, NUM_EXPERTS, K=7168, N_OUT=1024, BM, kUseNT, kInlineQuant, kXcdSwizzle>`）

## 1. 概述

MXFP4 路径的 stage1 GEMM。对每个「M-block × N-block × 专家」计算 `a_q × w1ᵀ`，得到
**gate（前 512 列）与 up（后 512 列）**（`N_OUT = 2*inter_dim = 1024`），随后在 epilogue 内
`y = SiLU(gate)·up` 并**就地再量化为 mxfp4**，输出给 stage2 的 `inter_sorted_quant`（fp4）+
`inter_sorted_shuffled_scale`（e8m0，已是 gemm2 的 tile 布局）。是整条流水线最耗时的 kernel（trace 中
M=4096 约 363µs，占比最大）。

## 2. 何时被选中（M 条件）与三个开关

恒被 MXFP4 路径调用；行为随 `kernelName1` 解析（`_parse_mxfp4_g1_kname`，`fused_moe.py:985`）：

| M | BM | `kInlineQuant` | `kUseNT` | 说明 |
|--:|--:|---|---|---|
| 4 / 8 / 16 / 32 / 64 / 128 | 16 | **true** | true | `BM16_INLINEQUANT`：A 不预量化，gemm1 内即时量化 hidden（融合 [mxfp4_03](mxfp4_03_quant.md)）|
| 256 | 32 | false | true | `BM32_NT`：A 读 [mxfp4_03] 的预量化结果 |
| 4096 / 8192 / 16384 | 128 | false | false | `BM128`：A 预量化，`kUseAGPR`（BM=128 用 AGPR 累加器）|

`__launch_bounds__(256, BM==128?1:(BM==16?3:2))`：BM16 占用 3 waves/EU，BM32 2，BM128 1。

## 3. Launch 参数与 shape

`<<<grid, 256>>>`，`grid = total_m_blocks * (N_OUT/256=4)`，`total_m_blocks = cumsum/BM`（`gemm1_a4w4.cuh:575`）。

| 参数 | shape | dtype | 说明 |
|---|---|---|---|
| `A_q` | `[M, 7168/2]` 或 inline 时用 `hidden_states` | fp4x2 / bf16 | 激活（sorted 经 `m_indices` gather）|
| `A_scale`(=a_scale_sorted_shuffled) | tile 布局 | e8m0 | 激活 scale（inline 时内部产出）|
| `B_ps_q`(=w1) | `[E, 1024, 7168/2]` | fp4x2 | 预 shuffle 的 gate||up 权重 |
| `B_ps_scale`(=w1_scale) | `[E*1024, 7168/32]` | e8m0 | 权重 scale |
| `sorted_expert_ids` | `[m_blocks]` | i32 | 每 M-block 的专家 |
| `cumsum_tensor` | `[1]` | i32 | 有效行数（→ grid 大小）|
| `m_indices` | `[max_sorted]` | i32 | 排序行 → token_id（A gather）|
| `A_q_out`(=inter_sorted_quant) | `[max_sorted, 512/2]` | uint8 | **输出**：fp4 中间量 |
| `A_scale_out`(=inter_sorted_shuffled_scale) | tile 布局 | uint8 | **输出**：中间量 e8m0（gemm2 tile 布局）|

## 4. Grid / Block 映射与分块

- `pid = blockIdx.x` → `(m_block_idx, n_block_idx)`，`num_n_blocks = N_OUT/256 = 4`；`e = sorted_expert_ids[m_block_idx]`。
- `BN=256, BK=256, K=7168 → K_TILES_TOTAL=28`；`kStages=2`（B 软件流水），A 的 LDS 槽位 `kAStages=3`（BM<128）/`2`（BM=128）。
- 一个 block（256 线程 = 4 wave）算 `BM×256` 的输出 tile（256 列 = gate 的 128 + up 的 128，由 `wave_n` 切分）。

## 5. 计算逻辑

### 主循环（K 方向 28 个 tile，双缓冲，`run_one` `:377`）
- **A**：`!kInlineQuant` 时 `buffer_load_lds` 把 sorted 行（经 `m_indices` gather 的 token）搬进 LDS（XOR16 swizzle，`issue_a_load_lds:121`）；`kInlineQuant` 时直接从 `hidden_states` 读 bf16，**在 kernel 内 amax→e8m0→`cvt_scalef32_pk_fp4_bf16` 量化**写进 LDS（`inline_quant_kt:209`），并把 e8m0 攒成 sorted-tile 布局。
- **B**：`issue_b_load_j`（`:312`）以 `b128` 从预 shuffle 的 w1 读入寄存器；`kUseNT` 时带 non-temporal hint。
- **MFMA**：`issue_mfma_cluster`（`:334`）调 `mfma_f4f4_*`（封装 `mfma_scale_f32_16x16x128_f8f6f4`，K=128 的缩放 fp4 MFMA），按 e8m0 在 MFMA 内做反量化；gate 与 up 各占 `J` 的偶/奇，累加进 `accm[kMChunks][4]`。
- 调度：`sched_barrier(0)` 分隔阶段，BM≠128 时 `s_setprio(1/0)` 围住 MFMA 簇做 wave-group 时分复用。

### Epilogue（`apply_cshuffle_quant_epilog`，`mxfp4_epilogs.hpp:21`）
1. 把 `accm`（gate 在 LDS 前 128 列、up 在 +128 列）经 LDS C-shuffle 重排成连续 N 向量（`:40-53`）。
2. 取 `gate_v[e]`、`up_v[e]`，`result[e] = silu_mul_fast(gate, up)`（`:88`，SiLU 用快速 exp2/rcp）。
3. 对每 32 个 result 求 amax → e8m0，`cvt_scalef32_pk_fp4_f32` 量化为 fp4（`:109-115`），写
   `inter_sorted_quant`（fp4 包）+ `inter_sorted_shuffled_scale`（e8m0，gemm2 tile 布局）。

## 6. 关键优化点

| 技术 | 目的 |
|---|---|
| `kInlineQuant`（小 M）| 把激活量化融进 gemm1，省 [mxfp4_03]+[mxfp4_04] 两个 kernel |
| 输出即 gemm2 tile 布局的 fp4+e8m0 | epilogue 直接产出 stage2 所需排布，省一次 reshuffle |
| A 双/三缓冲 LDS + B 寄存器驻留 | 隐藏 VMEM 延迟，MFMA 背靠背 |
| XOR16 LDS swizzle | 消除 ds_read/ds_write bank conflict |
| `mfma_scale_f32_16x16x128_f8f6f4` | gfx950 原生 fp4×fp4 缩放 MFMA，K=128/指令 |
| BM=128 用 AGPR 累加 + XCD remap | 大 M 提高累加器容量与跨 XCD 负载均衡 |
| `s_setprio` wave-group 时分复用（BM≠128）| 8-wave 在 4-wave 硬件上错峰 |

## 7. 与 FlyDSL gemm1（[F5](flydsl_05_moe1_gemm.md)）的差异对比

两者算的是**同一份数学**：按「M-block × N × 专家」做 fp4×fp4 缩放 MFMA → gate(128)+up(128) →
`y = SiLU(gate)·up` →（可选）按每 32 元素再量化为 fp4+e8m0，写 stage2 的 sorted-tile 布局。
但在**接口、功能、实现逻辑**三层上差异显著。下面 M7 指本 kernel（`gemm1_a4w4.cuh`），
F5 指 FlyDSL kernel（`mixed_moe_gemm_2stage.py`）。

### 7.1 接口差异

| 维度 | mxfp4 HIP gemm1（M7）| FlyDSL gemm1（F5）|
|---|---|---|
| 实现形态 | 单个 C++ `__global__` 模板 kernel（`gemm1_a4w4.cuh:28`），hipcc 预编译 | Python 运行时构建 MLIR IR，JIT 编译并缓存（`compile_mixed_moe_gemm1:92`，`~/.flydsl/cache`）|
| 配置方式 | `kernelName` 字符串解析成模板参数 `<MAX_M,NE,K,N_OUT,BM,kUseNT,kInlineQuant,kXcdSwizzle>`（`_parse_mxfp4_g1_kname`）| Python kwargs（`tile_m/n/k`、`a/b/out_dtype`、`gate_mode`、`k_batch`、`waves_per_eu`…）|
| host 调用约定 | `aiter.mxfp4_moe_gemm1_a4w4(...)`，**void 返回**，结果写进调用方预分配的 `inter_sorted_quant`/`inter_sorted_shuffled_scale`（`mxfp4_moe.py:145`）| `flydsl_moe_stage1(...)`，**有返回值**：`out`（bf16）或 `(out_packed, out_scale_sorted)`（fp4q）；`out=None` 时内部分配（`moe_kernels.py:694`）|
| A gather 入参 | `m_indices`（sorted-row→token_id）+ `cumsum_tensor`（定 grid）| `sorted_token_ids` + `num_valid_ids`（语义同，命名/打包不同）|
| 权重布局 | `shuffle_weight_a16w4(w1,16,True)`：gate‖up **在 N_OUT=1024 交织**（每 256 列 = 128 gate + 128 up），一个 WG 同 tile 同时算 gate+up | 取决于 `gate_mode`：`INTERLEAVE` 读 a16w4（同 M7）；**bench 用的 `SEPARATED`** 读 `shuffle_weight(16,16)`，gate/up 为**两条独立 B 流 + 两个累加器**（`gx=ceil(inter/tile_n/2)`）|
| 激活 scale 入参 | `a_scale_sorted_shuffled`（来自 [mxfp4_04](mxfp4_04_sort_scales.md) sort_scales，已 tile 布局）或 inline 内部产出 | `a1_scale`（sorted-tile，来自 [F2](flydsl_02_dynamic_quant.md)/[F3](flydsl_03_scale_sort.md) 或 `moe_mxfp4_sort`）|
| 输出选择 | **无选择**：恒输出 fp4+e8m0（中间量化是 gemm1 固有职责）| **可选**：`out_dtype="bf16"` 只输出 bf16（量化交给 [F2]+[F3]）；`out_dtype="fp4"`(`_fp4q`) 才融合量化 |

> 接口本质：M7 是「字符串选模板 + void 副作用写预分配 buffer」；F5 是「kwargs 选 JIT 编译 + 函数式返回张量」。
> 这也是为什么 bench 当前要给 F5 喂 `fly_w`（`SEPARATED` 布局）而非 M7 的 `mx_w`（a16w4）。

### 7.2 功能差异

| 功能 | mxfp4 HIP gemm1（M7）| FlyDSL gemm1（F5）|
|---|---|---|
| 中间量化 | **强制** fp4 requant（无 bf16 直出选项）| **可选**：M≤256 融合 fp4（`_fp4q`），M≥4096 不融合、直出 bf16 |
| inline-quant（读 bf16、即时量化 A）| `kInlineQuant`，仅 BM∈{16,32}（`gemm1_a4w4.cuh:45`）| `inline_quant=True`，配 tile_m=16 廉价 sort（`compile_mixed_moe_gemm1:119`）|
| split-K | **不支持**（K=7168 单趟 28 tile）| 支持 `k_batch`（`_kb{n}`）：K 切到 `block.z`，gate/up partials `atomic fadd` 后单独 `silu_and_mul` 归约 |
| gate 模式 | 固定 gate‖up 交织同算 | `SEPARATED`/`INTERLEAVE`/`GATE_ONLY`/`MOCK_GATE_ONLY` 四种 |
| BM=128 大 M | `kUseAGPR`（AGPR 累加）+ XCD remap（`gemm1_a4w4.cuh:49`）| 无显式 AGPR 控制（累加器交 MLIR 后端分配）；有 `xcd_swizzle` |
| bias / swiglu 限幅 | 无（纯 silu_mul）| 可选 `bias`、`swiglu_limit`（`compile_mixed_moe_gemm1:107/118`）|

### 7.3 实现逻辑差异

| 环节 | mxfp4 HIP gemm1（M7）| FlyDSL gemm1（F5）|
|---|---|---|
| A→LDS | `buffer_load_lds`（XOR16 swizzle，`issue_a_load_lds:121`）；inline 时从 `hidden_states` 读 bf16 | `_async`=`raw_ptr_buffer_load_lds` 直 DMA（`mixed:1142`）或寄存器路径 |
| K-loop 缓冲 | B 软件流水 `kStages=2` + A 三槽 `kAStages=3`（BM<128）/2（BM=128），手写 `run_one`/`issue_*` lambda | LDS ping-pong 双缓冲（`lds_x_ping/pong`），B-major 调度 |
| MFMA scale op_sel | `mfma_f4f4_*` 封装 `mfma_scale_f32_16x16x128_f8f6f4`，A-scale op_sel = `ikxdl*2+i`（固定 m-half=2）| `rocdl.mfma_scale_f32_16x16x128_f8f6f4`（cbsz=4/blgp=4），op_sel = `ikxdl*pack_M+imxdl` + `_rearrange_a_scale` |
| fp4 量化指令 | **硬件** `cvt_scalef32_pk_fp4_bf16`(inline，`gemm1_a4w4.cuh:237`)/`_f32`(epilogue，`mxfp4_epilogs.hpp`) | **软件** `_f32_to_e2m1` 位运算模拟（`mixed:2355/4553`）——硬件 per-2 `cvt_scalef32_pk_fp4` 在 rocdl 未暴露（`mixed:4697`）|
| e8m0 舍入常数 | 全路径 `+0x200000`（mxfp4 精确）| `_fp4q` cshuffle epilogue 用 `+0x400000`（`mixed:2413`），仅 inline-quant 路径用 `+0x200000`（`mixed:901`）→ 主路径与 M7 可差 1 ULP |
| 每 32 元素 amax 归约 | DPP-quad shuffle（`inline_quant_dpp_quad_amax`）| `shuffle_xor` 跨 lane（`mixed:2407/4690`）|
| 调度原语 | `sched_barrier(0)` + `s_setprio(1/0)`（BM≠128 wave-group 时分复用）| `sched_barrier(0)` + `s_setprio(1/0)`（`mixed:1758/1791`）——同思路 |
| SiLU | `silu_mul_fast`（快 exp2/rcp，`mxfp4_epilogs.hpp:88`）| `_silu_mul_vec4`/`_silu_elem`（exp2/rcp，`mixed:1942`）|

### 7.4 小结

- **相同处**：核心数学、MFMA 指令（`mfma_scale_f32_16x16x128_f8f6f4`）、LDS 双/多缓冲 + B 驻寄存器、
  cshuffle epilogue、`s_setprio` 时分复用、以及输出 sorted-tile fp4+e8m0 的目标布局，两者一致。
- **不同处**：
  1. **接口契约**：M7 = void + 预分配 buffer + 字符串模板；F5 = 返回张量 + kwargs + JIT。F5 多了
     `gate_mode`/`k_batch`/`bias`/`out_dtype` 等开关，M7 行为由模板 flag 固定。
  2. **权重布局**：M7 恒 a16w4 gate‖up 交织；F5 默认（bench）`SEPARATED` 两流，需喂 `fly_w`。
  3. **fp4 量化**：M7 走硬件 `cvt_scalef32_pk_fp4`，F5 因 rocdl 未暴露该 intrinsic 而用软件 `_f32_to_e2m1`，
     且主 `_fp4q` 路径 e8m0 舍入常数为 `+0x400000`（与 M7 的 `+0x200000` 有 1-ULP 偏差）。
  4. **split-K**：仅 F5 支持。

> 若要把 FlyDSL gemm1 做成 M7 的 **drop-in**（接口/布局/精度完全一致），需另写一份吃 a16w4 布局、void
> 副作用、`m_indices` gather、硬件/精确 e8m0 的新 kernel——即
> `aiter/aiter/ops/flydsl/kernels/mxfp4_a4w4_gemm.py::compile_mxfp4_gemm1_a4w4`（与本节描述的 F5 不是同一实现）。

### 7.5 附：为何两份文档里 w1 的 shape «看起来» 不同

本文档 §3 写 `w1=[E, 1024, 7168/2]`，而 [F5](flydsl_05_moe1_gemm.md) §3 写 `w1=[E, 2*512, 7168]`，看似形状不同。

**结论：是同一个物理张量、两种记法；真正不同的是内部字节排布（preshuffle），而非外层 shape。**

两边的 w1 都源自同一个基张量：bench `build_weights` 里 `w1=[E,2*inter,h]=[385,1024,7168]` bf16 →
`torch_quant(fp4x2)` 得 `w1_qt=[385,1024,3584]` fp4x2（fp4x2 沿 K 把每 2 个 fp4 打包进 1 字节 →
7168→3584）。两条路径只是对它做了**不同的 preshuffle**：

- mxfp4（M7）：`shuffle_weight_a16w4(w1_qt, 16, True)`
- FlyDSL（F5，bench 用的 `SEPARATED`）：`shuffle_weight(w1_qt, (16,16))`

而 `shuffle_weight`/`shuffle_weight_a16w4` 末尾都 `x_.view(*x.shape)`（`shuffle.py:32`/`:47`）——
**输出 shape == 输入 shape**，shuffle 只重排字节、不改外层形状。所以两者物理 shape **都是
`[385, 1024, 3584]` fp4x2**。文档里的差异纯粹来自**记法约定**：

| 维度 | M7 写法 | F5 写法 | 物理值 | 说明 |
|---|---|---|---|---|
| E | `E` | `E` | 385 | 同 |
| N（gate+up）| `1024` | `2*512` | 1024 | 同一个数；F5 用 `2*512` 强调 gate/up 两半 |
| K | `7168/2` | `7168` | 3584 | **M7 记打包后的 fp4x2 字节数（3584）；F5 记逻辑 fp4 元素数（7168）** |

> 即 `7168/2 = 3584`：二者是同一段数据，一个按「打包字节」记、一个按「逻辑 fp4 元素」记。
> （w1_scale 两边都写 `[E*1024, 7168/32]`，记法一致，所以没有这个分歧。）

**真正的差别在字节排布（同 shape、不同 permute）：**

| | mxfp4 a16w4（`gate_up=True`）| FlyDSL `shuffle_weight((16,16))` |
|---|---|---|
| reshape | `[E,2,N0=32,NLane=16,K0=56,KLane=4,KPack=16]` | `[E,N/16=64,16,K_pk/32=112,2,16]` |
| permute | `(0,2,1,4,5,3,6)`（`shuffle.py:28`）| `(0,1,3,4,2,5)`（`shuffle.py:45`）|
| gate/up | **交织**：N=1024 拆成 `[2(gate/up),32,16]`，gate/up 在 N0 粒度交织 → 一个 256-N tile = 128 gate + 128 up | **不交织**：N=1024 当作扁平 `[64,16]`，gate(前 512)/up(后 512) 仍是两段连续块 |
| 一个 WG 怎么算 | 同 tile 同时算 gate+up（交织取数）| `SEPARATED`：gate/up 作两条独立 B 流、两个累加器 |

所以两者**外层 shape 相同、dtype 相同，但权重字节顺序不同**，不能互换——这正是 §7.1「权重布局」一行的根因，
也是 bench 必须给 F5 喂 `fly_w`、给 M7 喂 `mx_w` 的原因。

## 8. 性能随 M 的 scaling：为何 M=4 / 8 / 16 耗时差别很大

**问题**：gemm1 用的 MFMA 指令 `v_mfma_scale_f32_16x16x128_f8f6f4` 是固定的，且 `BM=16`，按直觉
M=4 / 8 / 16（都 ≤ 16 个 token）应该「都落在一个 16 行 block 里」、耗时差不多。但实测差别很大
（`fly_vs_mx.log` 的 gemm1 段，`mx_avg` = 每次调用 device 时间）：

| M | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|
| mx_avg (µs) | 29.4 | 36.9 | 64.5 | 95.9 | 130.5 | 163.3 |

**直觉的误区**：gemm1 的耗时不取决于 token 数 M，而取决于**启动了多少个 workgroup**；WG 数随 M 增长。
根因是 MoE 的「topk fan-out + 按专家分组 + BM padding」三连，下面拆开。

### 8.1 topk=9 把 M 放大成 M×9 个「(token, 专家) 计算对」

bench 的 `build_inputs`（`bench_up_moe_v1.py:68-83`）给每个 token 选 **9 个专家**（1 个共享专家
`384` 恒在 + 8 个路由专家）。MoE 的数学是「每个 token 过它的 9 个专家再加权求和」：

```
out[t] = Σ_{i=0..8}  topk_weight[t,i] · FFN_e(x_t),   e = topk_ids[t,i]
FFN_e(x) = down_e( SiLU(gate_e·x) · (up_e·x) )         # gemm1 算 gate/up+SiLU·mul，gemm2 算 down
```

所以**逻辑上的专家-FFN 计算次数 = M × topk = M × 9**，不是 M。M=16 → 16×9 = **144** 个 (token,专家) 对。

### 8.2 按专家分组 + 每专家 padding 到 BM=16 → block 数 ≈ 活跃专家数

每个专家有独立权重 `w1[e]`，要让权重只读一次并复用，必须把指向**同一专家**的所有行排到一起做一次 GEMM。
sort 阶段（`mxfp4_moe_sort`）因此：① 把 M×9 个对按 expert id 排序；② 每个专家的行数 **padding 到 BM=16
的倍数**（gemm1 一个 WG 固定处理 16 行）；③ 产出 `cumsum`（padding 后总行数）、`sorted_expert_ids`。

padding 上界（`_mxfp4_moe_run`，`mx_sort_fly_gemm.py:186-189` 镜像）：

```python
active     = min(NE, M * topk)            # 命中的活跃专家数（上限 NE=385）
cumsum_max = M * topk + active * (BM - 1) # 真实对数 + 每个活跃专家最多 BM-1 行 padding
max_sorted = ceil(cumsum_max / BM) * BM   # → total_m_blocks = cumsum / BM
```

关键是 `active * (BM-1)`：**每命中一个新专家就至少多一个 16 行 block**（小 M 时该 block 里多半是 padding
空行）。小 M 下 `total_m_blocks ≈ 活跃专家数`。

### 8.3 每个 block 的开销固定（满 K=7168），与真实行数无关

gemm1 不因 block 里只有 1 个真实行就少算——它对这 16 行照样跑完整的 **28 个 K-tile**
（`v_mfma_scale_f32_16x16x128_f8f6f4` × 224，见 deep-intro §1.5）。grid = `total_m_blocks × (N_OUT/256 = 4)`
（`gemm1_a4w4.cuh:573-576`）。所以**总 device 时间 ≈ block 数 × 每 block 固定开销**（GPU 未饱和前近似线性）。

### 8.4 把三者合起来：耗时随「活跃专家数 / block 数」翻倍

| M | M×9 对 | 活跃专家(无碰撞上界) | 实测活跃专家 | gemm1 grid (×4) | 实测 mx_avg |
|---|---|---|---|---|---|
| 4 | 36 | 36 | 33 | ~132 | 29.4 |
| 8 | 72 | 72 | 58 | ~232 | 36.9 |
| 16 | 144 | 144 | 97 | ~388 | 64.5 |
| 32 | 288 | 288 | 147 | ~588 | 95.9 |
| 64 | 576 | min(385,576)=385 | 207 | ~828 | 130.5 |
| 128 | 1152 | min(385,1152)=385 | 254 | ~1016 | 163.3 |

- 「无碰撞上界」= `min(385, M×9)`（假设每个路由对落到不同专家）；**实测活跃专家**（§8.6）因路由碰撞
  （热门专家被多个 token 选中）而更少，但仍随 M 单调上升，且 ≈ block 数 ≈ WG/4。
- M 从 4→8→16，实测活跃专家 33→58→97 大致随 M 上升 → block/WG 数上升 → 耗时随之从 29→37→64µs。
- M≥64 路由碰撞加剧（活跃专家增速放缓 207→254 而非翻倍），每专家行数变多、padding 浪费下降，故耗时增幅
  也放缓（130→163µs）。

### 8.5 结论

`v_mfma_scale_f32_16x16x128_f8f6f4` 指令本身一样、每个 block 算的也一样多；但 **M=4 / 8 / 16 启动的
workgroup 数（≈ 活跃专家数 × 4）相差好几倍**——因为 topk=9 的 fan-out + 「每个专家 padding 到 BM=16」
放大了 block 数。**瓶颈是 block 数量（= padding 后 sorted 行 / 16 × 4），不是 token 数 M**。这也是小 M
MoE 的典型「padding 浪费」：16 行 block 里常常只有 1 行有效，但 kernel 仍按「满 block × 满 K」计时。

### 8.6 实测：固定 seed 下的专家散布（`build_inputs`, seed=1, KIMI）

`bench_up_moe_v1.py:build_inputs` 用固定 seed=1（路由由独立 `torch.Generator` 驱动，跨 M 可复现且**同一
token 的路由与 M 无关**，故 M 增大只是叠加新 token）。用脚本
`aiter/analyze_moe_expert_distribution.py` 统计 `mx_fn` 实际输入 `topk_ids[M, 9]`（1 shared + 8 routed）
散落到多少专家、每个专家处理多少 token（= 该专家在 topk_ids 中出现的次数）：

```bash
# 在 rocm 容器内、aiter 目录下运行（device=cuda 才能复现 bench 的 RNG）
python3 analyze_moe_expert_distribution.py --M-list 4,8,16,32,64,128,256,4096,8192,16384
python3 analyze_moe_expert_distribution.py --M-list 16 --print-per-expert   # 打印每个专家处理多少 token
```

实测结果（NE=385，shared 专家 id=384；全 M-list）：

| M | token-专家对 (M×9) | 散落到的专家数（active）| 单专家 token 数范围 [min, max] | 均值 tok/专家 |
|---|---|---|---|---|
| 4 | 36 | **33** | [1, 4] | 1.09 |
| 8 | 72 | **58** | [1, 8] | 1.24 |
| 16 | 144 | **97** | [1, 16] | 1.48 |
| 32 | 288 | **147** | [1, 32] | 1.96 |
| 64 | 576 | **207** | [1, 64] | 2.78 |
| 128 | 1152 | **254** | [1, 128] | 4.54 |
| 256 | 2304 | **297** | [1, 256] | 7.76 |
| 4096 | 36864 | **383** | [1, 4096] | 96.25 |
| 8192 | 73728 | **384** | [1, 8192] | 192.00 |
| 16384 | 147456 | **385** | [1, 16384] | 383.00 |

读法与要点：
- **散落到的专家数**随 M 单调上升（33→58→97→147→207→254→297→383→384→385），始终**低于无碰撞上界**
  `min(385, M×9)`（如 M=16：上界 144、实测 97）——因为路由有碰撞：`scores = randn + bias`，`bias` 让部分专家
  更「热门」、被多个 token 选中。**直到 M=16384 才把全部 385 个专家命中**（M=4096 时已用满 383/385）。这正是
  §8.4 里 block 数 / 耗时随 M 上升、但中段增速逐步放缓的根因。
- **单专家 token 数范围恒为 [1, M]**：上界 `max=M` 永远来自**共享专家 384**（在每个 token 的 topk 里，
  故处理全部 M 个 token）；下界 `min=1` 是只被一个 token 选中的冷门路由专家（各 M 下都存在）。
- **均值 tok/专家从 1.09 一路涨到 383**（M=16384 时 ≈ `M×9/385 = 147456/385 = 383`，因专家已全命中）：
  - **小 M（≤256）均值 1~8**：绝大多数活跃专家只处理 1~几个 token，但 gemm1 仍要为每个专家各开一个 16 行 block
    跑满 K=7168 → §8.5 的 **padding 浪费极重**（M=4 均值 1.09 ≈ 每个 block 只有 1 行有效），耗时由 **block 数
    （≈活跃专家×4）** 主导。
  - **大 M（≥4096）均值 96~383 ≫ BM=16**：每个专家行数远超一个 block，block 被真实行填满、padding 浪费消失，
    gemm1 转为真正 **compute-bound**（耗时由真实算力主导，不再由活跃专家数主导）。
- M=4 的逐专家分布（`--print-per-expert`）：`384:4`（共享专家 4 token）+ 其余 32 个路由专家各 `:1`，共 33 个；
  随 M 增大，热门专家（如 `26`、`1`、`339`）的计数显著拉开（M=16384 时 `26:2813`、`1:2988`），长尾仍有大量 `:1`。

### 8.7 每个专家处理的 token 数（逐专家分布）

「每个专家处理多少 token」= 该专家 id 在 `topk_ids[M, 9]` 中出现的次数。用 `--histogram` 看分桶分布
（每个 token-数区间里有多少个活跃专家），`--print-per-expert` 看逐专家明细：

```bash
python3 analyze_moe_expert_distribution.py --M-list 4,8,16,32,64,128,256,4096,8192,16384 --histogram
python3 analyze_moe_expert_distribution.py --M-list 16 --print-per-expert
```

**tokens-per-expert 直方图**（每格 = 该 token-数区间内的专家个数；首列 **`0` = 没有任何 token 的专家数**
= NE − 活跃专家 = 385 − active；每行所有列之和 = NE = 385，去掉 `0` 列之和 = 活跃专家数）：

```
     M |   0 |   1 |   2 | 3-4 | 5-8 |9-16 |17-32|33-64|65-128|129-256|257-1024|1025-4096| >4096
-------+-----+-----+-----+-----+-----+-----+-----+-----+------+-------+--------+---------+------
     4 | 352 |  32 |   0 |   1 |   0 |   0 |   0 |   0 |    0 |     0 |      0 |       0 |     0
     8 | 327 |  50 |   7 |   0 |   1 |   0 |   0 |   0 |    0 |     0 |      0 |       0 |     0
    16 | 288 |  73 |  17 |   6 |   0 |   1 |   0 |   0 |    0 |     0 |      0 |       0 |     0
    32 | 238 |  92 |  28 |  20 |   6 |   0 |   1 |   0 |    0 |     0 |      0 |       0 |     0
    64 | 178 |  98 |  49 |  27 |  25 |   6 |   1 |   1 |    0 |     0 |      0 |       0 |     0
   128 | 131 |  75 |  43 |  59 |  47 |  24 |   5 |   0 |    1 |     0 |      0 |       0 |     0
   256 |  88 |  56 |  46 |  51 |  63 |  52 |  23 |   5 |    0 |     1 |      0 |       0 |     0
  4096 |   2 |   5 |   6 |  13 |  25 |  44 |  72 |  68 |   68 |    50 |     31 |       1 |     0
  8192 |   1 |   3 |   1 |   4 |   9 |  31 |  50 |  63 |   72 |    71 |     75 |       4 |     1
 16384 |   0 |   1 |   0 |   3 |   4 |   8 |  31 |  45 |   70 |    72 |    118 |      32 |     1
```

直方图要点：
- **`0` 列（无 token 的专家）随 M 减少**：352→327→288→238→178→131→88→2→1→**0**。即 385 个专家里大量是
  「冷专家」（小 M 时大半个 MoE 的专家根本没被任何 token 选中），只有 **M=16384 才所有 385 个专家都至少接到 1
  个 token**（`0` 列 = 0）。
- **小 M（≤256）分布堆在左侧（1~8 token/专家）**：例如 M=4 有 32 个专家只处理 1 个 token；M=256 仍有
  56 个专家只处理 1 个。这些专家各占一个 16 行 block 却几乎全是 padding → §8.5/§8.6 的浪费根源。
- **随 M 增大分布整体右移**：M=4096 起绝大多数专家落在 17~256 token/专家（远超 BM=16），block 被真实行填满；
  M=16384 时已有 118 个专家落在 257~1024、32 个落在 1025~4096、1 个 >4096。
- **每行最右侧的非零格 = 共享专家 384**（处理 M 个 token）：M=128→`65-128`、M=256→`129-256`、
  M=8192→`>4096`、M=16384→`>4096`，始终是分布里 token 数最大的那个专家。

**逐专家明细（小 M，`expert_id:tokens`，按 token 数降序）**：

```
[M=4]  33 个专家：
384:4  1:1  3:1  7:1  26:1  39:1  41:1  44:1  50:1  60:1  69:1  79:1  86:1  102:1  114:1  117:1  119:1  120:1
125:1  171:1  200:1  206:1  213:1  216:1  228:1  250:1  254:1  269:1  275:1  291:1  293:1  302:1  354:1
```

```
[M=8]  58 个专家：
384:8  26:2  44:2  79:2  86:2  100:2  171:2  216:2  1:1  3:1  7:1  23:1  27:1  39:1  41:1  47:1  50:1  60:1
69:1  82:1  93:1  101:1  102:1  114:1  117:1  119:1  120:1  125:1  138:1  177:1  186:1  200:1  206:1  213:1
223:1  228:1  248:1  250:1  254:1  255:1  268:1  269:1  275:1  290:1  291:1  293:1  296:1  301:1  302:1  331:1
339:1  346:1  354:1  358:1  359:1  366:1  377:1  381:1
```

```
[M=16]  97 个专家：
384:16  44:4  86:4  171:4  1:3  39:3  318:3  7:2  26:2  27:2  60:2  79:2  100:2  101:2  120:2  138:2  200:2
216:2  269:2  272:2  291:2  293:2  339:2  377:2  3:1  11:1  23:1  28:1  41:1  47:1  50:1  69:1  70:1  82:1
93:1  102:1  113:1  114:1  117:1  119:1  125:1  132:1  133:1  141:1  148:1  155:1  157:1  158:1  176:1  177:1
186:1  187:1  190:1  191:1  206:1  213:1  214:1  215:1  220:1  223:1  228:1  233:1  242:1  248:1  250:1  254:1
255:1  259:1  262:1  263:1  268:1  273:1  275:1  278:1  284:1  290:1  296:1  300:1  301:1  302:1  311:1  317:1
327:1  331:1  338:1  345:1  346:1  354:1  357:1  358:1  359:1  365:1  366:1  369:1  374:1  380:1  381:1
```

> 大 M（32 / 64 / 128 / 256 / 4096 / 8192 / 16384）逐专家明细太长（活跃专家 147~385，单专家 token 数最高
> 16384），不在此全列；用上面的 `--print-per-expert` 命令可复现任意 M 的完整 `expert_id:tokens`。共性：
> 共享专家 384 恒为 token 数最大（=M），热门路由专家（`1`、`26`、`7`、`44`、`339`、`269`、`369`…）随 M
> 计数快速拉大，冷门专家长尾恒为 1~2。
