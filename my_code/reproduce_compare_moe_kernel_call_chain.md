# gfx1250 A4W4 MoE 调用链与 kernel 功能说明

本文档说明下面这条命令在当前分支中的调用链，并沿调用链介绍一次完整
MoE 前向从输入到最终输出期间执行的主要 GPU kernel：

```bash
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 bash my_code/reproduce_compare.sh --gemm2
```

文档重点是 kernel 的职责、输入输出和相互关系，不展开 wave、LDS、TDM
descriptor 或具体指令调度等 kernel 内部实现细节。

<a id="toc"></a>
## 目录

- [1. 文档范围与当前命令的默认配置](#scope)
- [2. 加入 GEMM1 后的完整调用链](#call-chain)
- [3. MoE 数据流总览](#data-flow)
- [4. `reproduce_compare.sh`：case 驱动与计时入口](#runner)
- [5. 测试程序：构造输入并调用公开 `fused_moe`](#test-entry)
- [6. `aiter/fused_moe.py`：选择 gfx1250 grouped FlyDSL 路径](#fused-moe-dispatch)
- [7. `grouped_moe_gfx1250.py`：编排完整 MoE pipeline](#grouped-orchestrator)
- [8. Routing 与 contiguous-M 布局](#routing)
- [9. GEMM1 的 A/ScaleA producer](#gemm1-producer)
- [10. GEMM1：up/gate projection 与 SiLU](#gemm1)
- [11. GEMM2 的 A/ScaleA producer](#gemm2-producer)
- [12. GEMM2：down projection](#gemm2)
- [13. 最终 gather-reduce 与 MoE 输出](#gather-reduce)
- [14. 当前各个 `--gemm2` case 的实际 kernel 序列](#case-sequences)
- [15. 其他输入或配置下可能出现的 kernel](#conditional-kernels)
- [16. 不属于 timed MoE 主链的准备与验证操作](#outside-timed-path)
- [17. 源文件索引](#source-index)
- [18. 总结](#summary)

---

<a id="scope"></a>
## 1. 文档范围与当前命令的默认配置

`--gemm2` 表示比较 GEMM2 case，但每个 case 仍然运行一次完整的
`fused_moe`，不是只启动 GEMM2。脚本再从 profiler 结果中分别提取 GEMM1、
GEMM2 和整个 MoE 的时间。

当前命令的默认工作负载如下：

| 参数 | 值 | 含义 |
|---|---:|---|
| `data_format` | `a4w4` | activation 和 weight 都使用 MXFP4 |
| `E` | 96 | expert 数量 |
| `T` | 16384 | token 数量 |
| `topk` | 6 | 每个 token 路由到 6 个 expert |
| `R=T*topk` | 98304 | route 总数 |
| `model_dim` | 7168 | MoE 输入和最终输出维度 |
| `inter_dim` | 3072 | expert 中间维度 |
| activation | exact `SiLU` | GEMM1 epilogue 中执行 |
| bias | 无 | GEMM1、GEMM2 都不加 bias |
| output dtype | BF16 | GEMM1 中间值和最终输出均为 BF16 |
| benchmark data | `const0` | 脚本默认模式为 `e2e-const0` |
| expert distribution | balanced | `AITER_REPRO_EXPERT_BALANCE` 默认是 `true` |

该形状命中
[`aiter/configs/tuned_grouped_fmoe.csv`](../aiter/configs/tuned_grouped_fmoe.csv)
中 `gfx1250 / T=16384 / E=96 / topk=6 / 7168x3072 / a4w4` 的配置行。
当前配置为 GEMM1 和 GEMM2 都使用 `256x256x256` tile、`2x2` wave grid、
4 个 pipeline buffer，并使用 `cluster_n=4`。

`--gemm2` 模式下，GEMM1 默认固定为：

```text
AITER_REPRO_GEMM1_CASE=sync_mg4_fc28_apre_exactopt
```

GEMM2 默认依次测试：

```text
baseline
apre
apre_wpt2
apre_wpt2_mg4_fc28
apre_wpt2_mg4_fc28_ostore2p
apre_wpt2_mg4_fc28_ostore2p_ow2
```

这里的 GEMM2 `baseline` 表示“当前 GEMM2 kernel、关闭 GEMM2 A-preshuffle”，
不是 `93665e` historical GEMM1 baseline。

其中所有 `apre*` case 默认使用：

```text
AITER_REPRO_GEMM2_APRE_PRODUCER=rowgroup
AITER_REPRO_GEMM2_APRE_RPW=2
AITER_REPRO_GEMM2_APRE_PREFETCH=2
```

由 tile-aligned expert 空间计算得到的 grouped capacity 为：

```text
contiguous_m
  = align_up(T * topk + E * tile_m - topk, tile_m)
  = align_up(98304 + 96 * 256 - 6, 256)
  = 122880 rows
```

这 `122880` 行由 96 个 expert 的有效 route 行和 expert 间的 tile padding 组成。

[返回目录](#toc)

---

<a id="call-chain"></a>
## 2. 加入 GEMM1 后的完整调用链

```text
my_code/reproduce_compare.sh
  -> op_tests/test_flydsl_grouped_gemm_gfx1250.py
  -> aiter/fused_moe.py
  -> aiter/ops/flydsl/grouped_moe_gfx1250.py
       |
       +-- Routing
       |    -> aiter/ops/flydsl/moe_kernels.py
       |    -> kernels/moe_route_maps.py
       |         -> moe_route
       |    -> kernels/moe_contiguous_psum.py
       |         -> moe_contiguous_psum_remap
       |
       +-- GEMM1 A/ScaleA producer
       |    -> aiter/ops/flydsl/moe_kernels.py
       |    -> kernels/moe_fused_route_quant_scatter.py
       |         -> moe_quant_token_fd7168_fp4_pk8
       |         -> moe_invert_route_rows_tk6
       |         -> moe_scatter_preshuffled_a_fd7168_r32_lds
       |         [无 A-preshuffle 时也可能改为一个 routeks kernel]
       |
       +-- GEMM1
       |    -> aiter/ops/flydsl/grouped_gemm_mxfp4.py
       |    -> kernels/mxfp4_preshuffle_gfx1250_tdm.py
       |         -> a8w4_tdm_fp4_..._K7168_..._act1_...
       |
       +-- GEMM2 A/ScaleA producer
       |    -> aiter/ops/flydsl/moe_kernels.py
       |    -> kernels/moe_fused_route_quant_scatter.py
       |         -> baseline: routeks fd3072 kernel
       |         -> apre 默认: moe_quant_preshuffled_a_fd3072_rpw2_pf2_direct
       |         -> three_kernel 可选: compact quant + invert + LDS scatter
       |
       +-- GEMM2
       |    -> aiter/ops/flydsl/grouped_gemm_mxfp4.py
       |    -> kernels/mxfp4_preshuffle_gfx1250_tdm.py
       |         -> a8w4_tdm_fp4_..._K3072_...
       |
       +-- Final gather/reduce
            -> kernels/moe_gather_reduce.py
                 -> moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16
            -> moe_out [16384, 7168] BF16
```

GEMM1 和 GEMM2 使用同一个通用 TDM kernel 源文件，但通过 `K`、`N`、
`stage1_act`、A layout 以及调优参数编译成两个不同的 GPU kernel。

[返回目录](#toc)

---

<a id="data-flow"></a>
## 3. MoE 数据流总览

一次完整计算的逻辑关系如下：

```text
hidden_states [T, 7168] BF16
topk_ids      [T, 6]
topk_weight   [T, 6]
        |
        | routing：为每个 (token, topk_slot) 分配 expert 内行号
        v
topids_to_rows [T, 6] ----+
m_tile_map     [E]        |
                           |
hidden_states              |
        |                  |
        | MXFP4 quant + grouped scatter + GEMM1 A-preshuffle
        v                  |
GEMM1 A/ScaleA [122880, 7168]
        |
        | 乘 W1[expert]，生成 gate/up，并执行 exact SiLU
        v
y [122880, 3072] BF16
        |
        | MXFP4 quant；baseline 保持 row-major，apre 生成专用 A layout
        v
GEMM2 A/ScaleA [122880, 3072]
        |
        | 乘 W2[expert]
        v
grouped_out [122880, 7168] BF16
        |
        | 根据 topids_to_rows gather，乘 topk_weight，并对 6 路求和
        v
moe_out [16384, 7168] BF16
```

对 token `t` 的逻辑公式可以简写为：

```text
y(t, j) = SiLU(hidden[t] * W1_gate[expert(t,j)]^T)
          * (hidden[t] * W1_up[expert(t,j)]^T)

z(t, j) = y(t, j) * W2[expert(t,j)]^T

moe_out[t] = sum(j=0..topk-1, topk_weight[t,j] * z(t,j))
```

物理实现不会为每个 token 单独保存 `z(t,j)`；它把所有 route 按 expert 放入
`grouped_out`，最后再通过 `topids_to_rows` 找回每个 token 的 6 个结果。

[返回目录](#toc)

---

<a id="runner"></a>
## 4. `reproduce_compare.sh`：case 驱动与计时入口

文件：[`my_code/reproduce_compare.sh`](reproduce_compare.sh)

脚本本身不实现 MoE GPU kernel，主要完成以下工作：

1. 设置固定测试形状和公共环境变量。
2. 在 `--gemm2` 模式下固定 GEMM1 case，再逐个设置 GEMM2 case 的环境变量。
3. 为每个 case 启动独立 Python 进程，执行 `--scenario bench`。
4. 奇数轮正序、偶数轮逆序运行 case，降低随时间漂移带来的顺序偏差。
5. 从 profiler 输出中提取 GEMM1、GEMM2 和 `fused_moe` e2e 时间。
6. 在 `RUN_VERIFY=1` 时检查数值误差和输出 hash。

默认模式是 `e2e-const0`。`ROUNDS=3` 表示每个 GEMM2 case 完整执行三轮
MoE benchmark，而不是在同一次 MoE 中连续执行三次 GEMM2。

脚本生成一个临时 runner，对测试文件做少量运行环境适配，然后执行：

```text
op_tests/test_flydsl_grouped_gemm_gfx1250.py --scenario bench ...
```

[返回目录](#toc)

---

<a id="test-entry"></a>
## 5. 测试程序：构造输入并调用公开 `fused_moe`

文件：
[`op_tests/test_flydsl_grouped_gemm_gfx1250.py`](../op_tests/test_flydsl_grouped_gemm_gfx1250.py)

测试程序在进入 timed `fused_moe` 之前完成：

- 构造 `hidden_states`、`topk_ids` 和 `topk_weight`；
- 构造 W1、W2 的 MXFP4 payload 和 E8M0 scale；
- 将 W1 转为 GUGU，即 `[g0,u0,g1,u1,...]` 的 gate/up 行交错布局；
- 对 W1/W2 及其 scale 做 GEMM 需要的 weight preshuffle；
- 构造 PyTorch reference 所需的逻辑权重。

随后 `_call()` 只通过公开 API 调用：

```python
fused_moe(
    hidden,
    w1_arg,
    w2_arg,
    topk_w,
    topk_id,
    activation=ActivationType.Silu,
    quant_type=QuantType.per_1x32,
    w1_scale=w1_scale,
    w2_scale=w2_scale,
    gate_mode=GateMode.INTERLEAVE.value,
    dtype=dtypes.bf16,
)
```

`--scenario bench` 使用 profiler 计量整个 `_call()`，因此下文 routing、两个
producer、两个 GEMM 和最终 gather-reduce 都属于 MoE e2e 时间。

测试已经直接传入 `topk_ids` 和 `topk_weight`，所以 timed `fused_moe` 内部不会
再执行 gate logits 或 `topk` 选择 kernel。

[返回目录](#toc)

---

<a id="fused-moe-dispatch"></a>
## 6. `aiter/fused_moe.py`：选择 gfx1250 grouped FlyDSL 路径

文件：[`aiter/fused_moe.py`](../aiter/fused_moe.py)

调用层级为：

```text
fused_moe()
  -> fused_moe_()
  -> _fused_moe_impl()
  -> grouped_gemm_gfx1250_a8w4()
```

这些函数主要负责参数规范化和后端选择，本身不启动本次路径中的主要计算
kernel。

当前输入满足以下条件，因此进入 gfx1250 grouped FlyDSL 路径：

- gfx1250；
- `AITER_USE_GROUPED_GEMM=1`；
- `QuantType.per_1x32`；
- W1 为 GUGU/`GateMode.INTERLEAVE`；
- activation 是 `Silu`；
- activation 和 weight 都按 MXFP4 解释；
- W1/W2 scale 均已提供。

如果 grouped helper 返回有效 tensor，`_fused_moe_impl()` 立即返回该结果，不再
进入后面的通用 CK/ASM/Triton MoE fallback。

[返回目录](#toc)

---

<a id="grouped-orchestrator"></a>
## 7. `grouped_moe_gfx1250.py`：编排完整 MoE pipeline

文件：
[`aiter/ops/flydsl/grouped_moe_gfx1250.py`](../aiter/ops/flydsl/grouped_moe_gfx1250.py)

`grouped_gemm_gfx1250_a8w4()` 先从 tuned CSV 读取 tile 配置，再进入
`_grouped_a8w4_tdm_moe()`。后者按下面的顺序组织 kernel：

1. route 计数并生成 `topids_to_rows`；
2. 将各 expert 的行压缩成 tile-aligned contiguous-M 布局；
3. 生成 GEMM1 的 MXFP4 A payload 和 ScaleA；
4. 执行 GEMM1，并在 epilogue 中完成 exact SiLU；
5. 将 GEMM1 的 BF16 输出量化为 GEMM2 的 MXFP4 A/ScaleA；
6. 执行 GEMM2；
7. 将 grouped GEMM2 输出 gather 回 token 顺序，乘 route weight 并求和。

这个 Python 函数还负责在 `kernel_bench_callable` 中注册 GEMM1/GEMM2 的独立
launch callable。`--scenario bench` 的主要结果来自完整 e2e profiler，但测试
程序也会根据 kernel symbol 将两个 GEMM 的时间分别提取出来。

[返回目录](#toc)

---

<a id="routing"></a>
## 8. Routing 与 contiguous-M 布局

### 8.1 counter 初始化 kernel

非 EP 路径首先执行：

```python
counter = torch.zeros(E, dtype=torch.int32, device=device)
```

这通常在 profiler 中显示为 PyTorch 的
`vectorized_elementwise_kernel<..., FillFunctor<int>, ...>` 或等价 fill/memset
kernel。它把每个 expert 的 route 计数器清零。

### 8.2 `moe_route`

来源：
[`kernels/moe_route_maps.py`](../aiter/ops/flydsl/kernels/moe_route_maps.py)

当前命令的非 EP 路径调用：

```text
moe_route
```

它为每个 route，即每个 `(token, topk_slot)`，完成：

1. 读取目标 expert id；
2. 对该 expert 的 counter 执行 atomic increment，得到 expert 内唯一 `slot`；
3. 写出临时 masked-layout 行号：

```text
masked_row = expert * max_m + slot
```

输出为：

- `masked_m[e]`：expert `e` 的有效 route 数；
- `topids_to_rows[t,j]`：route `(t,j)` 暂时对应的 masked row。

atomic 分配只要求同一 expert 内行号唯一，不要求 route 顺序稳定；后续所有阶段
都通过同一张 `topids_to_rows` 映射访问，因此不会影响功能正确性。

### 8.3 `moe_contiguous_psum_remap`

来源：
[`kernels/moe_contiguous_psum.py`](../aiter/ops/flydsl/kernels/moe_contiguous_psum.py)

当前路径随后调用：

```text
moe_contiguous_psum_remap
```

该 kernel 同时完成两件事：

1. 对每个 expert 的 `masked_m[e]` 按 `tile_m=256` 向上对齐并做 prefix sum，
   得到每个 expert 在 contiguous buffer 中的起点 `starts[e]`；
2. 将 `topids_to_rows` 从 `expert*max_m+slot` 原地改写为
   `starts[expert]+slot`。

它还输出：

```text
m_tile_map[e] = starts[e] + masked_m[e]
```

`m_tile_map[e]` 是 expert `e` 的有效行终点。GEMM1、GEMM2 以及 GEMM2
`rowgroup` producer 都通过 binary search 查找某个 M tile 属于哪个 expert。
该 tensor 在 `_grouped_a8w4_tdm_moe()` 的局部变量名是 `psum`，传给 GEMM
launcher 后作为 `m_tile_map` 使用。

示意图：

```text
原 masked layout：
expert 0: [valid rows........................large unused capacity]
expert 1: [valid rows........................large unused capacity]
...

contiguous-M layout：
[expert0 valid][padding to 256]
[expert1 valid][padding to 256]
[expert2 valid][padding to 256]
...
```

由于 expert 起点由实时 route count 计算，这条路径兼容 balanced 和
non-balanced token 分布。把 `AITER_REPRO_EXPERT_BALANCE=false` 只会改变
`topk_ids` 和各 expert 的实际行数，不会切换到另一套 GEMM 或删除 binary
search。

[返回目录](#toc)

---

<a id="gemm1-producer"></a>
## 9. GEMM1 的 A/ScaleA producer

入口：
[`aiter/ops/flydsl/moe_kernels.py`](../aiter/ops/flydsl/moe_kernels.py)
中的 `flydsl_moe_fused_quant_preshuffle()`。

实现：
[`kernels/moe_fused_route_quant_scatter.py`](../aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py)

### 9.1 当前默认路径：三个 kernel

默认 GEMM1 case 是 `sync_mg4_fc28_apre_exactopt`，因此
`AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1`。当前形状满足 token-once 快速路径条件，
实际依次执行三个 kernel。

#### 9.1.1 `moe_quant_token_fd7168_fp4_pk8`

功能：把每个源 token 的 BF16 activation 从 7168 维量化为 MXFP4，并生成每
32 个元素一个 E8M0 scale。

关键逻辑是“每个 token 只量化一次”。虽然每个 token 有 6 条 route，但这一步
不会重复量化 6 次。输出临时 buffer：

```text
token_payload [16384, 7168/2] uint8
token_scale   [16384, 7168/32] uint8
```

#### 9.1.2 `rows_to_tokens` 初始化 kernel

Python 通过 `torch.full((contiguous_m,), -1, int32)` 初始化逆映射 buffer。
这通常显示为第二个 `FillFunctor<int>` kernel。

#### 9.1.3 `moe_invert_route_rows_tk6`

已有 `topids_to_rows` 表示：

```text
route -> grouped_row
```

该 kernel 将其反转为：

```text
grouped_row -> token
```

因为 `source_topk=6`，源 token 可以由 `route // 6` 得到。expert padding 行继续
保持 `-1`，后续 scatter 不读取这些行。

#### 9.1.4 `moe_scatter_preshuffled_a_fd7168_r32_lds`

该 kernel 根据 `grouped_row -> token`：

1. 从 token 临时 payload/scale 读取对应 token；
2. 将数据复制到 expert-contiguous 的 grouped row；
3. 通过 32-row LDS staging 完成布局转置；
4. 直接写成 GEMM1 所需的 A-preshuffle payload 和 ScaleA layout。

三 kernel 的数据关系如下：

```text
hidden[token]
    |
    | moe_quant_token_fd7168_fp4_pk8
    v
token_payload/token_scale
    |
    | topids_to_rows --moe_invert_route_rows_tk6--> rows_to_tokens
    v
moe_scatter_preshuffled_a_fd7168_r32_lds
    |
    v
GEMM1 preshuffled A1/ScaleA1 [grouped_row, 7168]
```

### 9.2 GEMM1 不启用 A-preshuffle 时

如果通过 `AITER_REPRO_GEMM1_CASE` 选择 `sync_mg4_fc28` 等非 `apre` case，
producer 通常变为一个 route-indexed kernel：

```text
moe_fused_quant_preshuffle_routeks_fd7168_r8_fp4_pk8_srctk6_noKS
```

它沿 98304 条 route 直接读取 `hidden[route//6]`，量化并写入对应 grouped row。
因此同一个 token 可能被量化 6 次，但不需要 token 临时 buffer、逆映射和 scatter。

其中：

- `srctk6`：源行是 `route // 6`；
- `noKS`：当前 route 数足够大，每个 workgroup 自己循环 K，不额外拆 K grid；
- 没有 `_apre`：A payload 使用普通 grouped row-major layout；ScaleA 仍按 GEMM
  读取要求排列。

### 9.3 A-preshuffle 快速路径不满足条件时

如果启用了 GEMM1 A-preshuffle，但 token-once 三 kernel 路径的形状或容量条件
不满足，会回退到 route-indexed `_apre` 变体：

```text
moe_fused_quant_preshuffle_routeks_..._srctk<topk>..._apre
```

它在单个 kernel 内完成 route gather、MXFP4 quant 和 A-preshuffle，但仍会按
route 重复量化同一 token。

[返回目录](#toc)

---

<a id="gemm1"></a>
## 10. GEMM1：up/gate projection 与 SiLU

Python launcher：
[`aiter/ops/flydsl/grouped_gemm_mxfp4.py`](../aiter/ops/flydsl/grouped_gemm_mxfp4.py)

kernel 实现：
[`kernels/mxfp4_preshuffle_gfx1250_tdm.py`](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py)

GEMM1 是一个动态命名的 TDM kernel，当前 symbol 以类似下面的字段组成：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_...
```

它完成：

1. 根据当前 M tile 在 `m_tile_map` 上做 binary search，确定所属 expert；
2. 从该 expert 的 W1 读取 GUGU 排列的 gate/up MXFP4 weight 和 ScaleB；
3. 读取对应 grouped A1 和 ScaleA1；
4. 执行 MXFP4 grouped GEMM，产生 6144 个逻辑输出，即 3072 组 gate/up；
5. 在 epilogue 中对每组执行 exact：

```text
y = SiLU(gate) * up
```

6. 将结果写成：

```text
y [1, 122880, 3072] BF16
```

`sync_mg4_fc28_apre_exactopt` 改变的是同一语义 kernel 的 pipeline ownership、
WMMA operand reuse、调度和 output-store overlap；它不改变 GEMM1 接口、exact
SiLU 数学、expert 查找方式或输出 dtype。

GEMM1 对 non-balanced 分布的支持来自动态 `m_tile_map`。每个 M tile 都必须由
binary search 找到 expert，不能假设每个 expert 有相同行数。

如果显式选择 `baseline_93665e`，launcher 会改用：

```text
kernels/mxfp4_preshuffle_gfx1250_tdm_93665e.py
```

但给出的 `--gemm2` 命令默认不走这条历史 GEMM1 实现。

[返回目录](#toc)

---

<a id="gemm2-producer"></a>
## 11. GEMM2 的 A/ScaleA producer

GEMM1 输出 `y` 是 BF16，而 GEMM2 仍是 A4W4，因此必须把有效 grouped rows
再次量化为 MXFP4。这里是 `baseline` 与 `apre*` case 的第一个主要差异。

### 11.1 `baseline`：一个普通 route-indexed producer

`baseline` 设置：

```text
AITER_FLYDSL_GEMM2_A_PRESHUFFLE=0
```

当前大 route 数命中：

```text
moe_fused_quant_preshuffle_routeks_fd3072_r8_fp4_pk8_srcrow_noKS
```

功能：

1. 遍历 `topids_to_rows` 中的有效 grouped row；
2. 读取 `y[grouped_row, :]`；
3. 对 3072 维 BF16 行做 per-32 MXFP4 quant；
4. 写出普通 grouped row-major A2 payload；
5. 写出 GEMM2 所需的 ScaleA layout。

`srcrow` 表示源数据已经位于 grouped row，而不是通过 `route//topk` 回到原 token。

### 11.2 `apre*` 默认：一个 rowgroup producer

所有默认 `apre*` case 设置 `AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1`，并默认选择
`rowgroup`：

```text
moe_quant_preshuffled_a_fd3072_rpw2_pf2_direct
```

该 kernel：

1. 按相邻 grouped row 组成 row group；
2. 对 workgroup 所在的 M 区域，在 `m_tile_map` 上做 binary search；
3. 找到 expert 的有效行终点，跳过 expert padding 和 capacity tail；
4. 直接读取 GEMM1 的 BF16 grouped output；
5. 完成完全相同的 MXFP4/E8M0 quant；
6. 直接写成 GEMM2 专用 A-preshuffle payload 和 ScaleA layout。

`rpw2` 表示每个 wave 同时负责 2 行，`pf2` 表示 load prefetch depth 为 2。
它不需要 compact 中间 buffer，也不需要逆映射和第二次 scatter。

```text
y grouped BF16
    |
    | binary search m_tile_map，过滤 padding
    | quant + direct preshuffled store
    v
GEMM2 A2/ScaleA2 preshuffled layout
```

### 11.3 可选 `three_kernel` producer

设置：

```text
AITER_REPRO_GEMM2_APRE_PRODUCER=three_kernel
```

会依次执行：

```text
moe_fused_quant_preshuffle_routeks_fd3072_r8_fp4_pk8_srcrow_noKS_compact
moe_invert_route_rows
moe_scatter_preshuffled_a_fd3072_r32_lds
```

其逻辑是：

1. 只把有效 route 行量化到 compact row-major payload/scale；
2. 将 `route -> grouped_row` 反转成 `grouped_row -> route`；
3. 按 grouped row 通过 LDS scatter/transpose 写成最终 GEMM2 A-preshuffle layout。

在 `moe_invert_route_rows` 之前还会通过 `torch.full(..., -1)` 启动一个 index
buffer 初始化 kernel。

### 11.4 rowgroup 不支持当前形状时的 fallback

即使请求 `rowgroup`，如果 `feat_dim`、`expert_tile_m`、prefetch divisibility 或
buffer 地址范围不满足其约束，dispatcher 会先回退到上面的 `three_kernel`
实现。

如果连 stage2 A-preshuffle 快速分支的外层条件都不满足，则使用单 kernel 的
route-indexed `_apre` 版本：

```text
moe_fused_quant_preshuffle_routeks_..._srcrow..._apre
```

[返回目录](#toc)

---

<a id="gemm2"></a>
## 12. GEMM2：down projection

GEMM2 仍通过：

```text
grouped_gemm_mxfp4.py
  -> mxfp4_preshuffle_gfx1250_tdm.py
```

编译出的 symbol 以类似下面的字段组成：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_...
```

GEMM2 完成：

1. 根据当前 M tile 在同一个 `m_tile_map` 上做 binary search；
2. 确定当前 tile 对应的 expert；
3. 读取该 expert 的 W2 MXFP4 payload/ScaleB；
4. 按 case 读取普通 A2 layout 或 A-preshuffle A2 layout；
5. 计算：

```text
grouped_out[row, :] = A2[row, :] * W2[expert]^T
```

6. 写出：

```text
grouped_out [1, 122880, 7168] BF16
```

GEMM2 不执行 SiLU，也不在内部乘 route weight。route weight 被延迟到最后的
gather-reduce，这样 GEMM2 仍可以按 expert 处理连续 M rows。

各 `apre*` case 只改变同一个 GEMM2 语义 kernel 的数据 layout或调度方式：

| case | GEMM2 A layout | 主要额外设置 |
|---|---|---|
| `baseline` | 普通 grouped row-major | `wpt1`，无 schedule hints |
| `apre` | A-preshuffle | `wpt1` |
| `apre_wpt2` | A-preshuffle | `wpt2` |
| `apre_wpt2_mg4_fc28` | A-preshuffle | `wpt2`、schedule hints、`mg4/fc28` |
| `apre_wpt2_mg4_fc28_ostore2p` | A-preshuffle | 上述设置，并把 output store 分段穿插进尾部计算 |
| `apre_wpt2_mg4_fc28_ostore2p_ow2` | A-preshuffle | 上述设置，并让更多 output waves 分担 store |

这里的 case 不改变 GEMM2 数学结果、BF16 输出类型或 `m_tile_map` binary search。

[返回目录](#toc)

---

<a id="gather-reduce"></a>
## 13. 最终 gather-reduce 与 MoE 输出

来源：
[`kernels/moe_gather_reduce.py`](../aiter/ops/flydsl/kernels/moe_gather_reduce.py)

当前形状调用：

```text
moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16
```

名称表示：

- `bf16`：grouped input 和最终输出是 BF16；
- `d7168`：输出维度为 7168；
- `tk6`：每个 token gather 6 条 route；
- `sk1`：只有一个 GEMM2 split-K slice；
- `v4`：每个线程每次处理 4 个 dword；
- `wbf16`：route weight 为 BF16。

该 kernel 以 token 为输出单位。对每个 token `t`：

1. 读取 `topids_to_rows[t,0..5]`；
2. 从 `grouped_out` gather 6 行；
3. 每行乘对应的 `topk_weight[t,j]`；
4. 在 FP32 中累加；
5. 转换并写出 BF16 `moe_out[t,:]`。

```text
grouped_out[row0] --* weight0--+
grouped_out[row1] --* weight1--|
...                            +--> FP32 sum --> BF16 moe_out[token]
grouped_out[row5] --* weight5--+
```

这是 timed MoE 主链的最后一个自定义 kernel。函数返回：

```text
moe_out [16384, 7168] BF16
```

[返回目录](#toc)

---

<a id="case-sequences"></a>
## 14. 当前各个 `--gemm2` case 的实际 kernel 序列

### 14.1 所有默认 case 共用的前半段

因为 GEMM1 固定为 `sync_mg4_fc28_apre_exactopt`，所有 GEMM2 case 都先执行：

```text
[PyTorch int32 fill]                         # route counter = 0
moe_route
moe_contiguous_psum_remap
moe_quant_token_fd7168_fp4_pk8
[PyTorch int32 fill]                         # rows_to_tokens = -1
moe_invert_route_rows_tk6
moe_scatter_preshuffled_a_fd7168_r32_lds
a8w4_tdm_fp4_..._K7168_..._act1_...          # GEMM1 + exact SiLU
```

### 14.2 `baseline`

```text
<共用前半段>
moe_fused_quant_preshuffle_routeks_fd3072_r8_fp4_pk8_srcrow_noKS
a8w4_tdm_fp4_..._K3072_...                   # 普通 A layout GEMM2
moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16
```

### 14.3 默认 `apre*` case

在默认 `rowgroup/rpw2/pf2` 设置下：

```text
<共用前半段>
moe_quant_preshuffled_a_fd3072_rpw2_pf2_direct
a8w4_tdm_fp4_..._K3072_..._apre...           # A-preshuffle GEMM2
moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16
```

`apre`、`apre_wpt2`、`apre_wpt2_mg4_fc28` 和两个 `ostore2p` case 的
kernel 数量相同；差别主要体现在 GEMM2 的 compile-time 调度参数和动态 symbol
后缀。

### 14.4 使用 `three_kernel` 的 `apre*` case

```text
<共用前半段>
moe_fused_quant_preshuffle_routeks_fd3072_r8_fp4_pk8_srcrow_noKS_compact
[PyTorch int32 fill]                         # rows_to_routes = -1
moe_invert_route_rows
moe_scatter_preshuffled_a_fd3072_r32_lds
a8w4_tdm_fp4_..._K3072_..._apre...
moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16
```

[返回目录](#toc)

---

<a id="conditional-kernels"></a>
## 15. 其他输入或配置下可能出现的 kernel

下表只列当前 grouped FlyDSL MoE 调用链真实支持的主要条件分支。

| 条件 | 可能出现的 kernel | 功能 |
|---|---|---|
| 非 EP，当前命令 | `moe_route` | local expert route 分配 |
| EP，先构造 global-to-local LUT | `moe_g2l_lut` | 根据 `expert_mask` 生成 global expert 到 local bucket 的 LUT，并清零 route counter |
| EP 且 local bucket 数适合 LDS reduction | `moe_route_g2l_lds` | global-to-local 路由、route weight cast/mask，并减少 global atomic 次数 |
| EP 的普通 fallback | `moe_route_g2l` | 不使用 LDS 两级 reduction 的 global-to-local route |
| GEMM1 未启用 A-preshuffle | `...fd7168...srctk<topk>...` | 每条 route 直接量化源 token 并写 grouped row |
| GEMM1 A-preshuffle token-once 快速路径 | `quant_token + invert + scatter` | token 只量化一次，再复制到所有 routed rows |
| GEMM1 A-preshuffle fallback | `...fd7168...srctk<topk>..._apre` | 单 kernel route quant 并直接写 A-preshuffle layout |
| GEMM2 baseline | `...fd3072...srcrow...` | grouped BF16 行量化为普通 A2 layout |
| GEMM2 Apre rowgroup | `moe_quant_preshuffled_a_fd3072_rpw*_pf*_direct` | 直接量化 grouped rows 并写最终 Apre layout |
| GEMM2 Apre three-kernel | `compact quant + invert + scatter` | 通过 compact 中间数据和 LDS scatter 生成 Apre layout |
| route 数较少 | routeks 名称不带 `_noKS` | 将 K 维拆到更多 workgroup，提高小 grid 并行度 |
| route 数较大，如当前命令 | routeks 名称带 `_noKS` | route grid 已足够大，每个 workgroup 内循环 K |
| `AITER_REPRO_GEMM1_CASE=baseline_93665e` | historical TDM GEMM1 | 使用 in-tree 的 93665e GEMM1 实现进行对照 |

当前命令不是 EP，因此不会执行 `moe_g2l_lut`、`moe_route_g2l_lds` 或
`moe_route_g2l`。

源码中还定义了 `moe_route_psum_fused`、`moe_contiguous_psum`、
`moe_route_maps` 等 kernel，但当前 `_grouped_a8w4_tdm_moe()` 的这条大形状路径
固定使用 `moe_route + moe_contiguous_psum_remap`，所以这些 kernel 不在本命令
的实际时间线上。

[返回目录](#toc)

---

<a id="outside-timed-path"></a>
## 16. 不属于 timed MoE 主链的准备与验证操作

以下工作与测试有关，但在 `_call()` 之前或之后完成，不应当误算成 MoE e2e
中的 kernel：

### 调用前

- 构造 const/random hidden、weight、scale；
- 生成 balanced 或 random `topk_ids/topk_weight`；
- `moe_shuffle_weight()`；
- `moe_shuffle_scale()`；
- W1 的 GGUU 到 GUGU 物理布局准备。

### 调用后

- 使用 PyTorch reference 计算参考输出；
- 计算 `logits_diff`、`rel_l2` 和 hash；
- 解析 profiler 表格；
- 汇总三轮 median 和相对提升。

因此，测试日志中如果出现输入初始化、reference 或 hash 相关 kernel，它们不等同
于前文列出的 timed `fused_moe` 主链。

[返回目录](#toc)

---

<a id="source-index"></a>
## 17. 源文件索引

| 层次 | 文件 | 主要职责 |
|---|---|---|
| benchmark runner | [`my_code/reproduce_compare.sh`](reproduce_compare.sh) | case、环境变量、轮次、日志与汇总 |
| test | [`op_tests/test_flydsl_grouped_gemm_gfx1250.py`](../op_tests/test_flydsl_grouped_gemm_gfx1250.py) | 输入构造、公开 API 调用、profile 与正确性验证 |
| public API | [`aiter/fused_moe.py`](../aiter/fused_moe.py) | dtype/后端判定并选择 gfx1250 grouped 路径 |
| MoE orchestrator | [`aiter/ops/flydsl/grouped_moe_gfx1250.py`](../aiter/ops/flydsl/grouped_moe_gfx1250.py) | routing、两个 producer、两个 GEMM、gather-reduce 的整体编排 |
| GEMM launcher | [`aiter/ops/flydsl/grouped_gemm_mxfp4.py`](../aiter/ops/flydsl/grouped_gemm_mxfp4.py) | 识别 GEMM1/GEMM2 形状并传递 compile-time 参数 |
| GEMM kernel | [`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py) | GEMM1/GEMM2 的 TDM/WMMA grouped GEMM |
| historical GEMM1 | [`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_93665e.py`](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_93665e.py) | `baseline_93665e` 对照实现 |
| producer dispatcher | [`aiter/ops/flydsl/moe_kernels.py`](../aiter/ops/flydsl/moe_kernels.py) | 选择 route、quant、A-preshuffle producer 分支 |
| producer kernels | [`aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py`](../aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py) | FP4 quant、route-indexed producer、token-once、rowgroup、invert、scatter |
| route kernels | [`aiter/ops/flydsl/kernels/moe_route_maps.py`](../aiter/ops/flydsl/kernels/moe_route_maps.py) | route 分配及 EP global-to-local route |
| contiguous-M kernels | [`aiter/ops/flydsl/kernels/moe_contiguous_psum.py`](../aiter/ops/flydsl/kernels/moe_contiguous_psum.py) | tile-aligned prefix sum 和 row remap |
| EP LUT kernel | [`aiter/ops/flydsl/kernels/moe_g2l_lut.py`](../aiter/ops/flydsl/kernels/moe_g2l_lut.py) | EP global expert 到 local bucket 的 LUT |
| final output kernel | [`aiter/ops/flydsl/kernels/moe_gather_reduce.py`](../aiter/ops/flydsl/kernels/moe_gather_reduce.py) | gather grouped rows、乘 route weight、topk reduce |
| shared GEMM helpers | [`aiter/ops/flydsl/kernels/gemm_common_gfx1250.py`](../aiter/ops/flydsl/kernels/gemm_common_gfx1250.py) | barrier、pipeline fence、activation 等共享逻辑 |
| tuned config | [`aiter/configs/tuned_grouped_fmoe.csv`](../aiter/configs/tuned_grouped_fmoe.csv) | GEMM tile、wave、buffer 和 cluster 参数 |

[返回目录](#toc)

---

<a id="summary"></a>
## 18. 总结

对当前默认 `--gemm2` 测试，MoE 的核心结构不是“GEMM1 + GEMM2”两个 kernel，
而是：

```text
route
  -> contiguous-M remap
  -> GEMM1 A/ScaleA producer
  -> GEMM1 + exact SiLU
  -> GEMM2 A/ScaleA producer
  -> GEMM2
  -> weighted gather-reduce
```

默认情况下：

- GEMM1 producer 是 `quant_token + invert + LDS scatter` 三 kernel；
- GEMM1 是 `K7168 + act1` 的 TDM grouped GEMM；
- GEMM2 `baseline` 使用普通 route-indexed producer；
- GEMM2 `apre*` 默认使用一个 `rowgroup` direct producer；
- GEMM2 是 `K3072 + no activation` 的 TDM grouped GEMM；
- 最后由 `moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16` 生成 token-major
  最终输出。

Routing、两个 GEMM 和默认 GEMM2 rowgroup producer 都依赖动态 route 结果；
GEMM1/GEMM2 以及 rowgroup producer 都保留对 `m_tile_map` 的 runtime binary
search，因此同一条调用链可处理 non-balanced expert M 分布。

[返回目录](#toc)
