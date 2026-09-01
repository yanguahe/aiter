# FMHA FlyDSL 优化历史

<!-- markdown-toc-generator:start -->
## Table of Contents

- [优化 1 — Causal Triangle-Fold（q-block 镜像配对）](#优化-1-causal-triangle-foldq-block-镜像配对)
  - [1.1 思路来源](#11-思路来源)
  - [1.2 实现过程（3-way merge）](#12-实现过程3-way-merge)
  - [1.3 代码改动要点](#13-代码改动要点)
    - [kernels/flash_attn_gfx950.py](#sec-kernelsflash_attn_gfx950-py)
    - [kernels/flash_attn_generic.py](#sec-kernelsflash_attn_generic-py)
    - [单测文件](#单测文件)
  - [1.4 初版性能测试（含 spill）](#14-初版性能测试含-spill)
- [优化 2 — 消除 Causal Fold 的 VGPR Spill（30 → 0）](#优化-2-消除-causal-fold-的-vgpr-spill30-0)
  - [2.1 问题发现](#21-问题发现)
  - [2.2 根因分析](#22-根因分析)
  - [2.3 修复方法：per-pass anchor](#23-修复方法per-pass-anchor)
    - [Anchor 1：Q-load 行基底（prologue，消除 30→18 spill）](#anchor-1q-load-行基底prologue消除-3018-spill)
    - [Anchor 2：Q-load 列基底（_load_q_all 内，消除 18→2 spill）](#sec-anchor-2q-load-列基底_load_q_all-内消除-182-spill)
    - [Anchor 3：O-store 列基底（non-splitK store 循环前，消除 2→0 spill）](#anchor-3o-store-列基底non-splitk-store-循环前消除-20-spill)
  - [2.4 修复后 ISA 验证](#24-修复后-isa-验证)
  - [2.5 修复后性能测试](#25-修复后性能测试)
- [PMC 测试数据](#pmc-测试数据)
  - [3.1 spill 消除前（初版 fold，HIP_VISIBLE_DEVICES=1，iters=5）](#31-spill-消除前初版-foldhip_visible_devices1iters5)
  - [3.2 spill 消除后（修复版 fold，HIP_VISIBLE_DEVICES=3，iters=100 --compare）](#32-spill-消除后修复版-foldhip_visible_devices3iters100---compare)
  - [3.3 数据解读](#33-数据解读)
  - [3.4 B16 PMC 数据（spill 消除后）](#34-b16-pmc-数据spill-消除后)
- [分析 — ATT 负载均衡（Occupancy）](#分析-att-负载均衡occupancy)
  - [数据：B1 S8192 H64 causal bf16](#数据b1-s8192-h64-causal-bf16)
  - [数据：B16 S8192 H64 causal bf16](#数据b16-s8192-h64-causal-bf16)
  - [解读](#解读)
  - [综合结论](#综合结论)
- [性能重测：GPU 3（HIP_VISIBLE_DEVICES=3）](#性能重测gpu-3hip_visible_devices3)
- [优化 3 — 支持 Split-K 路径下的 Causal Triangle-Fold](#优化-3-支持-split-k-路径下的-causal-triangle-fold)
  - [3.1 改动](#31-改动)
  - [3.2 性能验证](#32-性能验证)
  - [3.3 持平原因分析](#33-持平原因分析)
- [任务 — 支持非对称 head_dim (qk_head_dim, v_head_dim) = (192, 128)](#sec-任务-支持非对称-head_dim-qk_head_dim-v_head_dim-192-128)
  - [需求](#需求)
  - [实现：把单一 HEAD_DIM 拆成 QK_HEAD_DIM / V_HEAD_DIM](#实现把单一-head_dim-拆成-qk_head_dim-v_head_dim)
  - [正确性验证](#正确性验证)
  - [性能现状](#性能现状)
- [任务 — (192,128) 性能对齐 hand-asm（Stage A: 4-wave layout 对齐）](#sec-任务-192128-性能对齐-hand-asmstage-a-4-wave-layout-对齐)
  - [Stage A：4-wave 单组 layout（消除 spill）](#stage-a4-wave-单组-layout消除-spill)
  - [结果（iters 100, GPU 2）](#sec-结果iters-100-gpu-2)
  - [剩余差距（待 Stage B）](#剩余差距待-stage-b)
  - [Stage B 前的 ATT cycle-budget 分析（(1,512,...,192x128)）](#sec-stage-b-前的-att-cycle-budget-分析1512---192x128)
  - [Stage B — asm LDS swizzle byte-exact 解码](#stage-b-asm-lds-swizzle-byte-exact-解码)
  - [Stage B step 1 — 隔离新内核文件（基线建立）](#stage-b-step-1-隔离新内核文件基线建立)
  - [Stage B step 2 — merged-softmax 主循环（WIP，未收敛）](#stage-b-step-2-merged-softmax-主循环wip未收敛)
  - [Stage B step 3 — merged-softmax 循环收敛（NaN 根因：MFMA 累加器未退休）](#stage-b-step-3-merged-softmax-循环收敛nan-根因mfma-累加器未退休)
  - [Stage B step 4 — 路线 B：merged-128 + 流水线（用户选定）](#stage-b-step-4-路线-bmerged-128-流水线用户选定)
  - [Stage 3 — ATT thread-trace 周期分析（按 flydsl-align-reference-kernel playbook）](#stage-3-att-thread-trace-周期分析按-flydsl-align-reference-kernel-playbook)
  - [Stage3-5 — dump asm ATT trace 对比（颠覆"追长链"假设）](#stage3-5-dump-asm-att-trace-对比颠覆追长链假设)
  - [Stage3-6 — exp2 交织进 MFMA 窗口（参考 Stage-A 手法）](#stage3-6-exp2-交织进-mfma-窗口参考-stage-a-手法)
  - [Stage3-7 — 扩 LDS + V 双缓冲领先预取（尝试，净收益为零，已回退）](#stage3-7-扩-lds-v-双缓冲领先预取尝试净收益为零已回退)
- [ISA 对齐（新目标：性能不论，让 FlyDSL ISA 与 asm ISA 对齐）](#isa-对齐新目标性能不论让-flydsl-isa-与-asm-isa-对齐)
  - [对齐-1 基线：循环体指令直方图对比](#对齐-1-基线循环体指令直方图对比)
  - [对齐-2: WAITCNT/NOP 根因 = AGPR 累加器搬运（非 ~{memory}）](#对齐-2-waitcntnop-根因-agpr-累加器搬运非-memory)

<!-- markdown-toc-generator:end -->

> 记录在 `wk_sp2`（远程：`/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_sp2/FlyDSL`）进行的内核优化，按任务顺序追加，不改历史条目。
>
> - 分支：`opus_align`
> - 硬件：AMD MI355X（gfx950 / CDNA4）
> - 容器：`hyg_fyd3`
> - 测试命令基准：`python tests/kernels/test_flash_attn_fwd.py --causal --dtype bf16 --iters 100 --compare`

---

## 优化 1 — Causal Triangle-Fold（q-block 镜像配对）

**日期**：2026-06-17  
**分支**：`opus_align`  
**改动文件**：`kernels/flash_attn_gfx950.py`、`kernels/flash_attn_generic.py`、`tests/kernels/test_flash_attn_fwd.py`、`tests/kernels/test_flash_attn_fwd_extra.py`

---

### 1.1 思路来源

手写 ASM kernel `exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s` 已实现了一种负载均衡策略——**causal 三角折叠（q-block 镜像配对）**：

在因果掩码下，KV tile 数量与 q-block 位置成正比，形成一个三角形负载：q-block 0 只需处理 1 个 KV tile，q-block N-1 需要处理全部 N 个。直接 grid 调度时，tail q-block 的 WG 做的工作大约是 head WG 的 N 倍，负载严重不均。

ASM 的解法：**把 grid_y 减半**，让每个 WG 处理一对 q-block：

- Pass 0：原始分配的 q-block `i`（轻负载，靠近三角顶部）
- Pass 1：对称 mirror 块 `(N−1)−i`（重负载，靠近三角底部）

一轻一重搭配后，每个 WG 的 KV tile 总量约等于 N+1（常数），三角不均衡在 WG 内部被平衡，不再依赖硬件轮转来平均。

---

### 1.2 实现过程（3-way merge）

由于 `causal_triangle_fold.diff` 是相对于 `wk_sp1` 的旧 base（`6f3e3d08`）生成的**合并提交格式 `diff --cc`**，而 `wk_sp2` 当前代码是 **QKV varlen** 变体（blob `593016a2`），两者有大量重叠区域，无法直接 `git apply`。

采用 **3-way `git merge-file`** 方式合并：

```
BASE = 6f3e3d08   (fold diff 的原始 base)
OURS = 593016a2   (wk_sp2 当前 varlen 版本)
FOLD = 7152f047   (fold diff 的 after 版本，从 wk_sp1 repo 提取)
```

手工解决 7 处冲突（主要集中在 q-block 索引变量重命名、per-pass token range 与 varlen 路径的合并）。关键合并决策：

- `q_gmem_elem_offset`（依赖 `q_start`，per-pass）保留在 pass 循环**内**；`kv_gmem_elem_offset`（per-batch）提到循环**外**。
- `num_kv_tiles`（loop-invariant）提到循环外；`max_num_tiles`、split-K range（依赖 `q_start`）移入循环内。
- `_n_pass` 的条件**排除 VARLEN**：fold+varlen 不兼容（`_num_qb_total` 使用 padded max seqlen，不是 per-batch 的 seqlen_q）。
- 内核和 launch 的 `grid_y` 条件必须完全一致，统一用 `_DO_FOLD` 标志。

---

### 1.3 代码改动要点

<a id="sec-kernelsflash_attn_gfx950-py"></a>
#### `kernels/flash_attn_gfx950.py`

新增参数 `dualwave_swp_causal_fold=False`，build-time flag `CAUSAL_FOLD`，以及：

```python
_DO_FOLD = bool(CAUSAL_FOLD and CAUSAL and not SPLITK and not VARLEN)
```

内核 body 外层用 `range_constexpr(_n_pass)` 包裹（fold-off 时 `_n_pass=1` → 等价原单 pass body，codegen 零变化）：

```python
_num_qb_total = (seq_len_v + BLOCK_M - 1) // BLOCK_M
_n_pass = 2 if const_expr(_DO_FOLD) else 1
for _pass in range_constexpr(_n_pass):
    if const_expr(_pass == 0):
        q_block_idx = q_block_idx_base
    else:
        q_block_idx = (_num_qb_total - fx.Index(1)) - q_block_idx_base
    q_start = q_block_idx * BLOCK_M
    q_gmem_elem_offset = (q_tok_base + q_start) * stride_q_n_v + q_head_idx * HEAD_DIM
    ...  # max_num_tiles / split-K range / pipeline body 全部在这个 pass 循环内
```

Launch 侧 `grid_y` 同步减半：

```python
if const_expr(_DO_FOLD):
    grid_y = (num_q_blocks + fx.Index(1)) // fx.Index(2)
else:
    grid_y = num_q_blocks
```

<a id="sec-kernelsflash_attn_generic-py"></a>
#### `kernels/flash_attn_generic.py`

透传环境变量，让 `build_flash_attn_dualwave_swp_module` 可接受 `dualwave_swp_causal_fold` 参数：

```python
dualwave_swp_causal_fold=os.environ.get("FLYDSL_DUALWAVE_CAUSAL_FOLD", "0") == "1",
```

#### 单测文件

在 `DEFAULT_CONFIGS` 中启用目标规模：

```python
# Causal triangle-fold perf-comparison scale.
(16, 8192, 64, 64, 128, 1),
(1, 8192, 64, 64, 128, 1),
```

两个测试文件同步修改：`test_flash_attn_fwd.py` 与 `test_flash_attn_fwd_extra.py`。

---

### 1.4 初版性能测试（含 spill）

配置：`(1, 8192, 64, 64, 128, 1)`，`--causal --dtype bf16 --iters 100 --compare`，`HIP_VISIBLE_DEVICES=1`

| 版本 | Time (µs) | TFLOPS | MaxErr |
|---|---|---|---|
| fold=0（baseline） | 961.3 | 1143.8 | 3.91e-03 |
| fold=1（初版，含 spill） | 946.9 | 1161.1 | 3.91e-03 |
| **Δ** | **−14.4 µs** | **+1.5%** | — |

`(16, 8192, 64, 64, 128, 1)` 同测（B16 grid 饱和，fold 减半 WG 反而增加每 WG 内 inter-pass bubble）：

| 版本 | Time (µs) | TFLOPS |
|---|---|---|
| fold=0 | 15341.8 | 1146.7 |
| fold=1（含 spill） | 15786.2 | 1114.4 |
| **Δ** | +444.4 µs | **≈−2.8%（GPU 1）/ +0.8%（GPU 3）** |

结论：B1 小 grid 有轻微提升，B16 大 grid 有小幅退化。

---

## 优化 2 — 消除 Causal Fold 的 VGPR Spill（30 → 0）

**日期**：2026-06-17  
**背景**：优化 1 的 fold=1 路径存在 VGPR spill，需消除。

---

### 2.1 问题发现

`vgpr_spill_count=30@21_final_isa.s` 行 7956，同时 `Scratch_Size=124`（PMC 测试中 HBM write +47% 的根源）。

对比 fold=0 与 fold=1 的 ISA 信息：

| | VGPR count | VGPR spill | scratch_store | scratch_load | SGPR count |
|---|---|---|---|---|---|
| fold=0 | **255** | 0 | 0 | 0 | 46 |
| fold=1 初版 | 256（上限） | **30** | 16 | 21 | 66 |

核心矛盾：**单 pass body 已占用 255/256 VGPR，几乎没有额外空间**。

---

### 2.2 根因分析

`range_constexpr(2)` 将两个 pass 完全**静态展开（unroll）** 成一段连续代码，LLVM 的 CSE（公共子表达式消除）把两个 pass 中**结构相同的 per-lane 元素偏移**识别为公共表达式，提升（hoist）到 kernel 入口处：

- **Q-load 偏移**：`q_row_in_block * stride_q_n_v + q_col`（每个 `ks` 对应一个 i64，共 8 个）
- **O-store 偏移**：`lane_div_32`-derived `d_col`

这些值在两个 pass 中数值上**完全相同**（`wave_q_offset`、`lane_mod_32`、`lane_div_32` 均是 warp-lane 常量），hoist 后必须在整个 pass 0 期间保持活跃以供 pass 1 复用，从而产生 30 个 VGPR spill。

尝试过的无效方案：
- 在两个 pass 之间插入 `sched_barrier(0)`：无效。这是 LLVM IR 层的 CSE/hoisting，不是 machine scheduler 问题。
- 将外层改为 runtime `scf.for`：无效。`scf.for` 的 LICM 同样会把 loop-invariant 表达式提升到 preheader。

---

### 2.3 修复方法：per-pass anchor

FlyDSL 已有 `llvm.inline_asm`（`has_side_effects=True`）opaque identity 模式，用于阻断优化器越过其边界（如 `_anchor_scalar_f32`、`_anchor_v_o` 等）。

**思路**：对生成 spilled 偏移的「per-lane 基底值」插入 anchor，使其对优化器**不透明**。由于 anchor 带 `has_side_effects=True`，优化器无法证明两个 pass 的 anchor 结果相同，从而被迫在每个 pass 内独立计算偏移，不再跨 pass 共享 → 不再 spill。

新增 `_anchor_index` helper（i32 round-trip，no-op ASM）：

```python
def _anchor_index(x):
    xi = _raw(fx.Int32(x))
    anchored = llvm.inline_asm(xi.type, [xi], "", "=v,0", has_side_effects=True)
    return fx.Index(anchored)
```

三处插入（均用 `if const_expr(_DO_FOLD)` 门控，fold=0 codegen 零变化）：

#### Anchor 1：Q-load 行基底（prologue，消除 30→18 spill）

```python
q_row_in_block = wave_q_offset + lane_mod_32
if const_expr(_DO_FOLD):
    q_row_in_block = _anchor_index(q_row_in_block)
```

<a id="sec-anchor-2q-load-列基底_load_q_all-内消除-182-spill"></a>
#### Anchor 2：Q-load 列基底（`_load_q_all` 内，消除 18→2 spill）

```python
_q_col_lane = lane_div_32 * MFMA_LANE_K
if const_expr(_DO_FOLD):
    _q_col_lane = _anchor_index(_q_col_lane)
for ks in range_constexpr(K_STEPS_QK):
    q_col = (ks * K_STEP_QK) + _q_col_lane
    ...
```

#### Anchor 3：O-store 列基底（non-splitK store 循环前，消除 2→0 spill）

```python
_o_col_lane = lane_div_32 * 8
if const_expr(_DO_FOLD):
    _o_col_lane = _anchor_index(_o_col_lane)
for dc in range_constexpr(D_CHUNKS):
    for g in range_constexpr(2):
        ...
        d_col = (dc * D_CHUNK) + (2 * g) * 8 + _o_col_lane
```

---

### 2.4 修复后 ISA 验证

| | VGPR count | VGPR spill | scratch_store | scratch_load | SGPR count |
|---|---|---|---|---|---|
| fold=0 | 255 | **0** | 0 | 0 | 46 |
| fold=1 修复后 | 255 | **0** | 0 | 0 | 66 |

- fold=0 **byte-identical**（255 VGPR、0 spill 完全不变）。
- fold=1 spill **30 → 0**，VGPR 回到 255（与 fold=0 持平）。

---

### 2.5 修复后性能测试

配置：`(1, 8192, 64, 64, 128, 1)`，`--causal --dtype bf16 --iters 100 --compare`，`HIP_VISIBLE_DEVICES=1`

| 版本 | Time (µs) | TFLOPS | MaxErr | vs fold=0 |
|---|---|---|---|---|
| fold=0（baseline） | 955.5 | 1150.7 | 3.91e-03 | — |
| fold=1（含 spill） | 946.9 | 1161.1 | 3.91e-03 | +1.5% |
| **fold=1（spill 消除后）** | **914.1** | **1202.8** | **3.91e-03** | **+4.5%** |

消除 spill 后，fold=1 比 fold=0 快 **4.5%**（原来只有 1.5%，spill 吃掉了约 3% 的收益）。

---

## PMC 测试数据

**工具**：`rocprofv3`（ROCm 7.1.0）PMC 计数器，3 组分开采样（bytes / lat / hit，避免跨 block 混合导致超时），`--kernel-iteration-range "[1]"` 捕获 warm dispatch。  
**规模**：B1 S8192 H64 MHA，causal bf16。  
**脚本**：`fmha_opt_tools/pmc_fold_compare_v2.sh`、`pmc_fold_parse_v2.py`。

### 3.1 spill 消除前（初版 fold，HIP_VISIBLE_DEVICES=1，iters=5）

| Metric | fold=0 | fold=1 | Δ |
|---|---|---|---|
| HBM read (GB) | 2.187 | 1.519 | −30.5% |
| HBM write (GB) | 0.134 | **0.197** | **+47.1%** |
| HBM rd lat (cyc/req) | 539.6 | 581.3 | +7.7% |
| HBM wr lat (cyc/req) | 270.8 | 298.4 | +10.2% |
| L1→L2 lat (cyc/req) | 362.7 | 302.9 | −16.5% |
| L2 hit% | 11.7% | 41.3% | +29.6 pp |

注意：fold=1 的 `Scratch_Size=124`（寄存器溢出），HBM write +47% 完全来自 `scratch_store` 指令（溢出写入 global scratch memory），与真实 O-store 无关。

### 3.2 spill 消除后（修复版 fold，HIP_VISIBLE_DEVICES=3，iters=100 --compare）

| Metric | fold=0 | fold=1 | Δ |
|---|---|---|---|
| HBM read (GB) | 2.189 | 1.541 | −29.6% |
| HBM write (GB) | 0.134 | **0.134** | **0.0%** |
| HBM rd lat (cyc/req) | 536.5 | 569.3 | +6.1% |
| HBM wr lat (cyc/req) | 266.4 | 306.1 | +14.9% |
| L1→L2 lat (cyc/req) | 361.7 | 302.2 | −16.4% |
| L2 hit% | 11.7% | 38.3% | +26.6 pp |

### 3.3 数据解读

**HBM read −30%（KV 数据在 WG 内 L2 复用）**

fold=1 每个 WG 处理 2 个 q-block：pass 0 先做 light block（少量 KV tiles），pass 1 再做 mirror heavy block（大量 KV tiles）。pass 0 加载的 KV tiles 是 pass 1 所需 KV tiles 的子集，且 pass 0 刚执行完、数据仍在 L2，pass 1 命中 → HBM read 减少约 30%。这是 fold 的**核心收益**，与 spill 无关，两版本均观测到。

L2 hit% 从 11.7% 升至 38%+ 印证了这一点：大量 KV 读请求现在直接从 L2 满足而非访问 HBM。

**HBM write 修复前 +47%，修复后 0%**

修复前：30 个 VGPR spill → kernel 入口大量 `scratch_store_dwordx2`，epilogue 大量 `scratch_load_dwordx2`（ISA 行 84-142 处 16 次 store，约 3700 行后 21 次 load）。Scratch memory 使用 global memory 地址空间，走 TCC_EA 通道，计入 HBM write/read 计数。

修复后：Scratch_Size=0，write 流量回到与 fold=0 完全一致（0.134 GB = 纯 O-store）。

**HBM latency 小幅升高（+6%/+15%）**

grid 减半（1024 WGs vs 2048 WGs）→ 在途请求数减少 → 延迟隐藏能力降低 → per-request 延迟升高。这是 fold 的**固有代价**，与实测时间挂钩：fold=1 grid 一半、每 WG 多一倍 KV work，占用同等 CU 资源但 WG 间切换更少，平均延迟略升。

**L1→L2 latency −16%（L2 命中率提升的副产品）**

TCP→TCC latency 同时计入 L2 hit（~10–30 cyc）和 miss（~500+ cyc）的平均。命中率从 11.7% 升至 38%，平均被快速 hit 拉低，与实际内存系统改善无关。

**净收益分析**

| 因素 | 方向 | 量级 |
|---|---|---|
| KV L2 复用 → HBM read 减少 | ✓ 正向 | −30% HBM read |
| spill 消除 → write 流量消失 | ✓ 正向（修复后） | −47% HBM write（相对初版） |
| grid 减半 → 延迟隐藏弱化 | △ 负向 | +6–15% per-req 延迟 |
| grid 减半 → 占用下降 | △ 负向 | 在 B1 规模尚可容忍 |

B1 S8192：read 节省 > 延迟代价 → 净 +4.5% 性能（spill 修复后）。  
B16 S8192（大 grid，32768 WGs，硬件高度饱和）：GPU 1 测出 −2.8%，GPU 3 重测为 +0.8%，在各卡 ±3% 噪声带内，**B16 fold=1 与 fold=0 性能接近持平**，无显著收益也无明显退化。故 fold **默认关闭**，通过 `FLYDSL_DUALWAVE_CAUSAL_FOLD=1` 为小/欠饱和 grid 按需启用。

### 3.4 B16 PMC 数据（spill 消除后）

**配置**：B16 S8192 H64 causal bf16，`HIP_VISIBLE_DEVICES=3`，`--iters 100 --compare`。  
**脚本**：`fmha_opt_tools/pmc_fold_compare_v3.sh`、`pmc_fold_parse_v3.py`。

Grid：fold=0 → 16,777,216 threads（32768 WGs），fold=1 → 8,388,608 threads（16384 WGs）。两者 `Scratch_Size=0`（无溢出）。

| Metric | fold=0 | fold=1 | Δ |
|---|---|---|---|
| HBM read (GB) | 35.7 | 30.6 | −14.4% |
| HBM write (GB) | 2.147 | 2.147 | 0.0% |
| HBM rd lat (cyc/req) | 434.8 | 454.9 | +4.6% |
| HBM wr lat (cyc/req) | 205.6 | 213.4 | +3.8% |
| L1→L2 lat (cyc/req) | 306.7 | 286.2 | −6.7% |
| L2 hit% | 9.9% | 22.0% | +12.1 pp |

**B1 vs B16 横向对比**

| | B1 HBM rd Δ | B1 L2 hit Δ | B16 HBM rd Δ | B16 L2 hit Δ |
|---|---|---|---|---|
| fold=1 vs fold=0 | −29.6% | +26.6 pp | **−14.4%** | **+12.1 pp** |

B16 的 KV L2 复用效果约为 B1 的一半。原因：B16 有 16× 更多 WG，L2 在多 WG 间竞争激烈。pass 0 缓存的 KV tiles 在 pass 1 到来前有较大概率被其他 WG 驱逐，导致 L2 hit% 提升幅度（+12 pp）远小于 B1（+27 pp），HBM read 节省也从 −30% 降至 −14%。

尽管如此，HBM write 在 B16 同样保持持平（+0.0%），确认 spill 完全消除后写流量不受影响。HBM 延迟代价更温和（+4.6% vs B1 的 +6.1%）：B16 的 16384 WGs 仍远超调度容量，延迟隐藏几乎不受 grid 减半的影响。

**结论**：B16 下 HBM read 节省（14%）大致抵消 inter-pass 开销，实测 fold=1 性能接近持平（GPU 1: −2.8%，GPU 3 重测: +0.8%，均在各卡 ±3% 噪声带内，不能认为有显著差异）。fold 默认关闭是正确决策；`FLYDSL_DUALWAVE_CAUSAL_FOLD=1` 对 B1（小 grid，L2 复用充分）有稳定 +4.5% 收益，对 B16（大 grid）性能持平。

---

## 分析 — ATT 负载均衡（Occupancy）

**日期**：2026-06-17  
**工具**：`rocprofv3` ATT（`att_target_cu=1, att_shader_engine_mask=0xf, att_buffer_size=128MB`）+ `occupancy_balance.py`（`--labels fold0 fold1`）。  
**先决条件**：`librocprof-trace-decoder.so` 0.1.6（按 `rocprofv3-trace-decoder-prereq.mdc` 安装到容器 `hyg_fyd3`）。  
**脚本**：`fmha_opt_tools/att_occupancy.sh`、`fmha_opt_tools/att_dualwave_swp.yaml`。

### 数据：B1 S8192 H64 causal bf16

| 指标 | fold=0 | fold=1 | 说明 |
|---|---|---|---|
| Busy CUs | 32 | 32 | 相同 |
| Max slots/SIMD | 2 | 2 | waves_per_eu=2 设计特性，两者相同 |
| Over-subscribed CUs | 32 | 32 | 相同，不是差异来源 |
| dur median (ts) | 1,389,508 | 1,387,844 | 相近（−0.1%） |
| dur max (ts) | 1,505,428 | 1,392,280 | **−7.5%** |
| **end-span (max−min)** | **236,264** | **11,136** | **−95.3%** |
| **imbalance (span/med)** | **0.17** | **0.01** | **17× 更均衡** |

### 数据：B16 S8192 H64 causal bf16

| 指标 | fold=0 | fold=1 | 说明 |
|---|---|---|---|
| Busy CUs | 32 | 32 | 相同 |
| Max slots/SIMD | 2 | 2 | 相同 |
| Over-subscribed CUs | 32 | 32 | 相同 |
| dur median (ts) | 10,666,310 | 10,605,866 | −0.6% |
| dur max (ts) | 10,802,148 | 10,615,000 | **−1.7%** |
| **end-span (max−min)** | **287,584** | **39,256** | **−86.4%** |
| **imbalance (span/med)** | **0.03** | **0.00** | **7× 更均衡** |

### 解读

**B1：fold 将 imbalance 0.17→0.01，是 +4.5% 性能的核心机制**

fold=0 下，B1 仅有 2048 WGs，causal 三角导致靠近序列末尾的 q-block（32 个 KV tiles）比序列开头（1 个 KV tile）重约 32×。硬件调度器无法精确平衡，straggler CU 的 dur max 比 min 长 18.4%（1,505,428 vs 1,271,228），end-span=236,264，imbalance=0.17。整个 dispatch 完成时间由最慢 CU 决定，拖慢内核。

fold=1 将轻/重 q-block 配对到**同一 WG 内**（pass 0 + pass 1），每个 WG 的总 KV 工作量约等于 N+1 tiles（常数，与位置无关）。结果：end-span 从 236,264 → 11,136（−95%），duration max/min 差异从 234,200 → 8,792，所有 CU 几乎同时结束，tail latency 消失。这直接兑现为 +4.5% 实测性能提升。

**B16：imbalance 0.03→0.00，但负载均衡已非瓶颈**

B16 有 32768 WGs，硬件轮转本身已将三角负载摊平（大量 WG 在 CU 上串行执行，轻/重 WG 自然平均），fold=0 的 imbalance 仅 0.03。fold=1 进一步改善至 ≈0，但 dur median 仅缩短 0.6%（10,666K→10,606K），说明均衡改善对 B16 性能无实质贡献。B16 fold=1 性能接近持平（GPU 1: −2.8%，GPU 3 重测: +0.8%，在各卡噪声带内），说明 B16 下 inter-pass overhead 与均衡收益大致抵消，负载均衡不是主要瓶颈。

### 综合结论

| | B1 | B16 |
|---|---|---|
| **fold=0 imbalance** | **0.17（严重）** | 0.03（可接受） |
| **fold=1 imbalance** | **0.01（极佳）** | 0.00（极佳） |
| 负载均衡是否是瓶颈 | **是**，end-span −95% | **不是** |
| fold=1 实测性能 | **+4.5% ✓** | ≈持平（GPU 1: −2.8% / GPU 3: +0.8%，噪声带内） |

Causal triangle-fold 的设计目标（平衡 causal 三角）在 B1 小 grid 上机制完整、数据吻合。B16 大 grid 由硬件轮转已足够均衡，fold 的附加代价（inter-pass overhead、per-WG KV work 翻倍后 L2 reuse 有限）反而成为主导。默认关闭 + 按需启用（`FLYDSL_DUALWAVE_CAUSAL_FOLD=1`）是正确设计。

---

## 性能重测：GPU 3（HIP_VISIBLE_DEVICES=3）

**日期**：2026-06-17  
**背景**：355_34 机器 8 张 MI355X 卡的测量结果存在 ±3% 左右的卡间差异，需在 GPU 3 重测以确认 B16 的结论。  
**脚本**：`fmha_opt_tools/fold_perf_gpu3.sh`，`HIP_VISIBLE_DEVICES=3`，`--causal --dtype bf16 --iters 100 --compare`。

| Config | fold=0 (TFLOPS / us) | fold=1 (TFLOPS / us) | Δ (fold1/fold0) | MaxErr |
|---|---|---|---|---|
| **B16 S8192 H64** | 1117.5 / 15742.4 | 1126.0 / 15623.7 | **+0.8%** | 3.91e-03 |
| **B1 S8192 H64** | 1133.6 / 969.9 | 1185.5 / 927.4 | **+4.6%** | 3.91e-03 |

**结论修订**：
- B1 fold=1 +4.6%，与 GPU 1 的 +4.5% 完全吻合，结论稳定可信。
- B16 fold=1 GPU 3 测出 +0.8%（GPU 1 曾测出 −2.8%），两者均在各卡 ±3% 噪声带内，**B16 fold=1 与 fold=0 性能接近持平**。之前文档中"B16 −2.8%"的结论需修正为持平，而非退化。
- 综合 PMC（HBM read −14%，write 持平）和 occupancy（imbalance 0.03→0.00），B16 大 grid 下 L2 复用收益与 inter-pass overhead 大致抵消，符合持平观测。

---

## 优化 3 — 支持 Split-K 路径下的 Causal Triangle-Fold

**日期**：2026-06-19  
**分支**：`opus_align`  
**改动文件**：`kernels/flash_attn_gfx950.py`

### 3.1 改动

原 `_DO_FOLD` 的条件排除了 `SPLITK`：

```python
# before
_DO_FOLD = bool(CAUSAL_FOLD and CAUSAL and not SPLITK and not VARLEN)
```

修改为允许 SPLITK：

```python
# after — SPLITK is now included
_DO_FOLD = bool(CAUSAL_FOLD and CAUSAL and not VARLEN)
```

只改这一行。`_DO_FOLD` 是 kernel pass loop（`_n_pass`）和 launch `grid_y` 的唯一判断依据，两处同步生效，无需其他修改。

**为何安全**：split-K 路径下，每个 pass 各自独立计算 `q_start` → `max_num_tiles` → split tile range，O_partial 写入用 `q_row = q_start + wave_q_offset + lane_mod_32`（per-pass），两个 pass 写到不同的行范围，无冲突。empty-split zeroing block 同样在 fold 循环内，per-pass 独立处理。

**VARLEN 仍排除**：`_num_qb_total = (seq_len_v + BLOCK_M - 1) // BLOCK_M` 使用 padded max seq_len，非 per-batch seqlen_q，fold+varlen 语义不正确。

### 3.2 性能验证

规模 `(1, 98144, 3, 3, 128, 5)`，`--causal --dtype bf16 --iters 100 --compare`，`HIP_VISIBLE_DEVICES=1`。  
脚本：`fmha_opt_tools/splitk_fold_perf.sh`。

| | fold=0 | fold=1 | Δ | MaxErr |
|---|---|---|---|---|
| Time (µs) | 5998.7 | 6006.7 | +0.1% | 3.91e-03 (PASS) |
| TFLOPS | 1233.2 | 1231.6 | −0.1% | — |
| vs aiter_ck | 139.0% | 138.0% | — | — |
| vs aiter_asm | 116.6% | 116.7% | — | — |

正确性验证通过（MaxErr 3.91e-03）。性能持平（噪声带内）。

### 3.3 持平原因分析

`(1, 98144, 3, 3, 128, 5)` 的 grid：
- fold=0: `(H=3, Q=384, Z=5)` = 5760 WGs
- fold=1: `(H=3, Q=192, Z=5)` = 2880 WGs

虽然 B1 H3 seq 很长、causal 三角不均衡，但 split-K 把每个 WG 的 KV 工作量细分为约 77 tiles/split，加上 5760 WGs 的大 grid 本身就足以被硬件轮转平均。fold 的均衡收益被 inter-pass overhead 抵消，净效果持平。

如需在 split-K + causal 场景下验证 fold 收益，应选 grid 更小（更少 heads/batch）且 split 数更少的配置，使 triangle 不均衡更显著。

---

<a id="sec-任务-支持非对称-head_dim-qk_head_dim-v_head_dim-192-128"></a>
## 任务 — 支持非对称 head_dim (qk_head_dim, v_head_dim) = (192, 128)

**日期**：2026-06-24  
**Worktree**：`FlyDSL_task`（分支 task/new_task）；容器 `hyg_fyd2`；GPU 3  
**改动文件**：`kernels/flash_attn_gfx950.py`、`kernels/flash_attn_generic.py`、`kernels/flash_attn_interface.py`、`tests/kernels/test_flash_attn_fwd_extra.py`

### 需求
kernel 原本只支持 qk==v（128/128 或 192/192）。本任务支持 **qk=192, v=128**（Q/K 宽 192，V/O 宽 128），并**删除 192/192** 支持。

### 实现：把单一 HEAD_DIM 拆成 QK_HEAD_DIM / V_HEAD_DIM
QK-path（Q×K^T 收缩，宽 192）与 V-path（P×V 输出，宽 128）分开：

- **常量**：`K_STEPS_QK = QK//16 = 12`（QK）；`D_CHUNKS = V//32 = 4`（V）；`SMEM_D_RPT_QK=3`（K tile）/ `SMEM_D_RPT_V=2`（V tile）分开；`NUM_DMA_K=SMEM_D_RPT_QK`、`NUM_DMA_V=SMEM_D_RPT_V`。
- **LDS**：K tile（8×3×520）与 V tile（8×2×544）独立 sizing；buf base、KV_PER_BUFFER、LDS_TOTAL 自动从 tile elems 派生。
- **strides**：`stride_kv_n`（运行时）=K stride（qk 宽）；V 与 O 的 stride 在 v≠qk 时用编译期常量 `DEFAULT_STRIDE_V_N = Hkv·128` / `DEFAULT_STRIDE_O_N = Hq·128`（调用方传连续张量）。
- **gmem 偏移**：`k_gmem_elem_offset`（qk 宽 + K stride）与 `v_gmem_elem_offset`（v 宽 + V stride）分开；`_kv_tile_addr(tile, is_v)` 按 K/V 选 base/stride；num_records 分 K/V/O 三套。
- **O store**：`_global_idx_q` 用 V_HEAD_DIM + O stride；split-K workspace / combine kernel 的 `HEAD_DIM//2`、O store 全部改 V_HEAD_DIM。
- **softmax scale**：`1/sqrt(d)` 用 QK（head_dim_runtime 默认 = QK）。
- **接口**：`flydsl_flash_attn_func` 从 `v.shape[-1]` 推断 v_head_dim → `_build_dense(v_head_dim=D_V)` → generic dispatch gate `(qk,v) in (128,128)/(192,128)` → `build_flash_attn_dualwave_swp_module(v_head_dim=...)`。O 分配从 `empty_like(q)` 改为 `[*q.shape[:-1], D_V]`。
- **生成回退**：generic kernel 无法加载 192 宽（192/16=12 不整除 WG）→ 由 `_generic_supports_headdim` 触发 dualwave-only 早返回（已有逻辑）。

### 正确性验证
`python tests/kernels/test_flash_attn_fwd_extra.py --causal --dtype bf16 --iters 5 --compare`（GPU 3，hyg_fyd2）：

| Config | FlyDSL MaxErr | FlyDSL TFLOPS | vs aiter_ck |
|---|---|---|---|
| D=128 B16 S8192（回归） | 3.91e-03 ✅ | 1125.8 | 128% |
| D=128 B2 S1024（回归） | 3.91e-03 ✅ | 623.2 | 120% |
| **192x128 B16 S8192** | **3.91e-03 ✅** | 304.4 | 95% |
| **192x128 B2 S1024** | **3.91e-03 ✅** | 157.5 | 65% |

D=128 无回退；(192,128) 功能正确。

### 性能现状
(192,128) ISA：`vgpr_count=256, vgpr_spill_count=188`（V/O 降为 128 宽省了累加器，但 K 仍 12 K-step 192 宽）。188 spill 限制了 (192,128) 性能（304 TFLOPS，aiter_asm 有专用 192/128 kernel 达 1148）。后续优化方向同 (192,192)：压缩 K 寄存器工作集（K 流式加载）消除 spill。

---

<a id="sec-任务-192128-性能对齐-hand-asmstage-a-4-wave-layout-对齐"></a>
## 任务 — (192,128) 性能对齐 hand-asm（Stage A: 4-wave layout 对齐）

**日期**：2026-06-24  
**目标**：(192,128) 性能对齐 `exp_isa/fmha_fwd_hd192x128_bf16_1tg_4w_128x128_350_msk1_gm0.s`（4-wave/128×128/512 VGPR/full-LDS）。GPU 2。

### Stage A：4-wave 单组 layout（消除 spill）
8-wave 布局在 qk=192 时 spill 188 VGPR（256 VGPR/wave 不够）。改为 **(192,128) 专用 4-wave/BLOCK_M=128**：每 SIMD 1 wave → 512 VGPR/wave → spill=0。

改动（`build_flash_attn_dualwave_swp_module` 内 `SINGLE_GROUP=(qk,v)==(192,128)` 条件分支）：
- `NUM_WAVES=4, BLOCK_M=128, BLOCK_SIZE=256`；`waves_per_eu` passthrough `"1,1"`。
- KV 协作加载：8 条 LDS N-line 由 4 wave 各填 2 条（`line=wave_id+line_i*NUM_WAVES`，`row=n_in_warp*SMEM_N_PER_WAVE+line`）；`NUM_DMA_D_K/V` 为 d-loop 计数，`NUM_DMA_K/V` 为 vmcnt 总数（×SMEM_N_LINES_PER_WAVE）。
- stagger off（单组无 group B）。
- softmax 本就是 wave-local（permlane32_swap 组内规约），无需改。

<a id="sec-结果iters-100-gpu-2"></a>
### 结果（iters 100, GPU 2）

| scale | 改前 | Stage A 后 | aiter_asm | vs asm |
|---|---|---|---|---|
| (1,8192,64,64,192x128) | 304 | **840.4** | 1130.3 | 74.3% |
| (2,1024,64,64,192x128) | 157 | **513.1** | 657.8 | 78.0% |

- 正确性 MaxErr 3.91e-03 ✅；D=128 无回退（1120.9 TFLOPS）。
- ISA：vgpr_count=408, **spill=0**（改前 188），LDS 84736 B。

### 剩余差距（待 Stage B）
asm 用 BLOCK_N=128（更宽 tile、每 FLOP 更少 loop/barrier）+ full-LDS 双缓冲；当前 FlyDSL 仍 BLOCK_N=64。剩 ~22-26% 差距预计来自 tile 宽度与流水线深度。

<a id="sec-stage-b-前的-att-cycle-budget-分析1512---192x128"></a>
### Stage B 前的 ATT cycle-budget 分析（(1,512,...,192x128)）
ATT trace 分类（issue latency 占比）：

| class | hit | issue_lat | stall | lat% |
|---|---|---|---|---|
| VALU | 27668 | 138140 | 24692 | 27.8% |
| WAITCNT | 400 | 111892 | 111892(全stall) | 22.6% |
| VMEM | 1120 | 78600 | 70920 | 15.8% |
| LDS | 4480 | 42356 | 23544 | 8.5% |
| MFMA | 3200 | 40112 | 27292 | 8.1% |
| TRANS | 2624 | 26752 | 5772 | 5.4% |

**结论**：MFMA 仅占 8.1%（健康内核应 >25%），内核被 **softmax VALU(27.8%) + WAITCNT stall(22.6%) + VMEM stall(15.8%)** 主导。这些 per-KV-tile 开销（每 tile 一次 row-max/sum 规约、lgkmcnt 全 drain、barrier）随 tile 数线性增长。BLOCK_N=128 把 tile 数减半 → 直接砍掉约一半的 VALU 规约 + WAITCNT + BARRIER 开销，是对的方向。

### Stage B — asm LDS swizzle byte-exact 解码
解码 asm 的 LDS 布局（s33/s34=K bufs, s35/s36=V bufs）：

| buffer | base | size | 内容 |
|---|---|---|---|
| K buf0 | 0x0 | 0x6180=24960 B | 64 rows × 192 bf16 + 384 B pad |
| K buf1 | 0x6180 | 24960 B | 同上 |
| V buf0 | 0xc300=49920 | 0x4400=17408 B | 64 rows × 128 bf16 + 1024 B pad |
| V buf1 | 0xc300+0x4400 | 17408 B | 同上 |

**总 LDS = 2×24960 + 2×17408 = 84736 B**（与当前 FlyDSL 一致，能放下）。

**关键洞察**：loop counter s61 每轮 +0x80=128 → BLOCK_N=128/iter，但每个 LDS buffer 只 64 行。asm 用 **K buf0+buf1 = 一个 128-tile 的两个 64-col 半区**（V 同理），同时这两个 buffer 又作为 prefetch double-buffer（本轮 MFMA 消费的同时 `buffer_load…lds` 重填下一个 128-tile）。即 128-wide tile 复用 64-row 双缓冲，无需额外 prefetch buffer → LDS 不超。这推翻了"BLOCK_N=128 双缓冲放不下"的顾虑。

QK：每 wave 产 32×128 的 S = 2 个 N-half（v[32:47]/v[48:63]）× 12 K-step = 24 MFMA。
PV：4 个 V-chunk（v[96:111]/[112:127]/[128:143]/[144:159]）× 8 K-step = 32 MFMA。

### Stage B step 1 — 隔离新内核文件（基线建立）
为安全迭代 BLOCK_N=128 重写，把 `flash_attn_gfx950.py` 复制为 `flash_attn_gfx950_192x128.py`，public builder 改名 `build_flash_attn_192x128_module`。generic dispatch：(192,128) 路由到新 builder，128/128 仍用原 builder（Stage A 成果原地保留为 fallback）。

隔离基线验证（GPU 2，iters 5，新模块仍内部 BLOCK_N=64）：(192,128) = 822/494 TFLOPS，MaxErr 3.91e-03 ✅；128/128 = 1111 TFLOPS（原 builder，无回退）。新文件可独立迭代，不影响 Stage A。

下一步：在新文件里把主循环改成 merged-128-softmax（2 个 64-列子块合一次 row-max/sub-row/rescale），先 --compare 验正确性再调度。

### Stage B step 2 — merged-softmax 主循环（WIP，未收敛）
在 `flash_attn_gfx950_192x128.py` 把 SWP 主循环+prologue+epilogue 替换为一个**简单顺序 online-softmax 循环**（每 iter 一个 64-tile，runtime scf.for，loop-carried = m_run/l_run/D_CHUNKS 个 v_o），作为"先正确性"基线，准备后续合并到 128-wide。

**现状：未收敛**。该简单循环产生 NaN，模式为 **q_row ≥ 96（wave 3 起）及之后所有 q-block 全 NaN**，causal 和 non-causal 都复现；wave 0-2（q_row 0-95）正确。row0/row500 的值量级合理 → 不是全错，是稀疏/分波 NaN。Stage A 用**相同**的 `_async_load_k/v`/`_mma0/1`/`_attn_*` helper 却正确，说明 bug 在我新写的循环体（barrier 序列 / loop-carried 线程 / running-max 初值 `c_neg_floor` 的交互），尚未定位。远程 JIT+probe 每轮 3-5 min，调试成本高。

**当前仓库状态（安全）**：generic dispatch 已把 (192,128) **改回 Stage A 的 4-wave/BLOCK_N=64 builder**（824/489 TFLOPS, MaxErr 3.91e-03 ✅，aiter_asm 的 ~73%/68%）。BLOCK_N=128 的 WIP 隔离在 `flash_attn_gfx950_192x128.py`，不影响主路径。恢复 BLOCK_N=128 bring-up 只需把 dispatch 的 import 切回 `build_flash_attn_192x128_module`。

**下一步定位建议**：wave-3 边界 + 简单循环 → 重点查 (a) loop-carried v_o 的 scf.for iter_arg 线程是否正确（用独立 v_o 元素已试过），(b) K/V 共用 buf0 的 barrier 是否足够（每 iter 4 个 s_barrier），(c) `_attn_exp2_slice` 两次调用在新循环里对 v_s 的就地消费。最可靠的做法是先写一个**单 tile、单 wave** 的最小复现，逐 helper 比对 Stage A 的中间值。

### Stage B step 3 — merged-softmax 循环收敛（NaN 根因：MFMA 累加器未退休）

**根因定位（cache-safe baked-DBG 逐步切片）**：
- 用 baked-in（非 env-var，避免 JIT 按源码内容缓存而忽略 env）调试探针逐项验证：
  - `l_row`（softmax 分母）逐 tile **完全正确**（causal 给 1,32,63,94,195,372 = q_row+1；non-causal ~374）→ 分数/max/exp/sum/m-l 循环线程全对。
  - V-read 各 dc 量级正常（±0.9），无 NaN。
  - **RAW_O（未归一化累加器）**：dc=0,1,2（col 0-95）干净，**dc=3（col 96-127）全是 3e38 垃圾**。
  - 跳过 `_mma1` 里 dc=3 → 垃圾**移到 dc=2**。即**永远是 `_mma1` 里最后发射的那条 MFMA** 被破坏（中间步的结果因被下一步 MFMA 依赖消费而强制正确，末条只喂给 store/scale → 读取早于退休）。
  - ISA dump 确认 `vgpr_spill_count: 0`（不是 spill），VGPR=168。

**修复**：`_mma1` 每个 PV step 后插 `_anchor_v_o(v_o)` + `rocdl.sched_barrier(0)`，强制累加器在被复用/读取前退休到 VGPR。
- 仅 per-step `_anchor_v_o`：NaN 139680 → 13056（90%↓，但残留稀疏 NaN）。
- per-step `_anchor_v_o` + `sched_barrier(0)`：**NaN = 0**，MaxErr 0.0011（non-causal 7.7e-5）✅。

**当前性能（GPU 2，iters 100 --compare，merged-128 主循环但无流水线）**：

| 规模 | FlyDSL(new) | aiter_asm | aiter_ck | Fly/asm |
|---|---|---|---|---|
| (1,8192,64,64,192x128) | 548.3 | 1128.5 | 322.5 | 48.6% |
| (2,1024,64,64,192x128) | 398.7 | 664.8 | 248.4 | 60.0% |
| (1,8192,64,64,128) | 1122.6 | 1116.8 | 875.3 | 100.5%（未回归）|
| (2,1024,64,64,128) | 624.2 | 609.6 | 481.5 | 102.4%（未回归）|

MaxErr 全部 3.91e-03（与 asm 一致）。128/128 路径走原 builder，无回归。

**注意基线口径**：304/157 是「优化前」的旧数字，真正的 **Stage-A baseline = 840/513**（4-wave/BLOCK_N=64 深度流水，见上文表格）。此处"先正确性"的顺序循环 548/399 其实**比 Stage-A 慢**（无流水线 + MFMA 串行化），是中间状态，靠后续流水线追回。

### Stage B step 4 — 路线 B：merged-128 + 流水线（用户选定）

用户选路线 B：保留 merged-128 单次 softmax（最贴近 asm），在其上手搭流水线，而非移植 Stage-A 的 BLOCK_N=64 深流水。分小步，每步 GPU 2 iters 100 --compare，全程 MaxErr 0.0011（causal）/ 7.7e-5（non-causal），nan=0。

| 步骤 | (1,8192,192x128) | (2,1024,192x128) | 说明 |
|---|---|---|---|
| Step0 per-64 顺序循环 | 548.3 | 398.7 | 每 tile 全排空，DMA 全暴露 |
| Step1 merged-128 单 softmax | 612.6 | 431.7 | step=2，两子块合并 row-max/rescale，PV 两次累加同一 O |
| Step2a V 预取 overlap | 685.7 | 455.6 | V 加载提前到 MMA0 后发起（K/V 独立 LDS 区），与 softmax 重叠 |
| Step2b +K 预取 overlap | 719.5 | 474.8 | prologue 预取首对 K；循环内读 K 到 VGPR 后立即预取下一对 K（复用 buf0/buf1），与本 iter 计算重叠；V 先发起便于 partial vmcnt |
| Step3a 精简 _mma1 串行化 | 719.9 | 475.5 | 验证真正修 NaN 的是 `sched_barrier(0)` 而非 per-step `_anchor_v_o`；去掉 8 个 inline-asm anchor，仅每步 sched_barrier + 末尾一次 anchor。性能持平、代码更净 |

**当前 = 720/476（asm 的 63.6%/72.9%）**，仍低于 Stage-A 的 840/513。

**关键发现（NaN 修复机理修正）**：之前以为修 dc=3 NaN 需要 per-step `_anchor_v_o`+`sched_barrier`，Step3a 证明**只需 per-step `sched_barrier(0)`**（调度栅栏，阻止后续读取被提前到 MFMA 退休前），inline-asm anchor 可省。

**剩余差距分析（720 vs Stage-A 840）**：merged-128 是浅流水（预取领先仅 1 对），Stage-A 是 8-cluster 深流水（领先 2-3 tile）。LDS 只有 buf0/buf1 两区被当前 iter 占满，无法存更领先的预取对 → 加深流水需扩 LDS buffer（68KB→136KB，160KB 上限内放得下）。

**下一步候选**：(a) 用 rocprofv3 profiling 确认瓶颈是访存延迟还是计算（occupancy/L2-HBM skill）；(b) 扩 LDS 到 4 区做更深 ping-pong 预取；(c) s_setprio 调 MFMA 优先级；(d) sched_barrier_pairs 式 IGroupLP 提示让 MFMA 与 exp/softmax 交织。

### Stage 3 — ATT thread-trace 周期分析（按 flydsl-align-reference-kernel playbook）

**Stage3-1 roofline regime 分类**：理论 arithmetic intensity（假设理想 L2 复用）S=8192 AI=4096、S=1024 AI=512，均 ≫ MI355X ridge ~312 FLOP/byte → **compute-bound**。按 playbook：compute-bound 下计算调度类修复（MFMA 顺序/waitcnt/setprio）才是杠杆，访存吞吐优化收益有限。

**Stage3-2 dump FlyDSL ATT trace（seq512, att_target_cu=1）**：inner-loop（hitcount=36，644 指令）per-wave issue cycle 预算：

| Cat | N | Lat/w | Stall/w | %Tot | Waste% |
|---|---|---|---|---|---|
| MFMA | 80 | 1572 | 1252 | 23.6% | 79.6% |
| VALU | 271 | 1157 | 53 | 17.4% | 4.6% |
| VMEM | 20 | 1149 | 925 | 17.3% | 80.5% |
| WAITCNT | 35 | 1064 | 1064 | 16.0% | 100% |
| LDS | 112 | 803 | 355 | 12.1% | 44.2% |
| TRANS | 65 | 520 | 0 | 7.8% | 0% |
| TOTAL | 644 | 6657 | | | |

MFMA chains（改前 dc-inner+per-step sched_barrier）：`[1,1,1,13,16, 1×11, 2,1,1,2,...]` —— QK 段有 13/16 长链，PV 段碎成 2-1-1。

**Stage3-3 关键发现 — 真瓶颈是 lgkmcnt 等待，非 MFMA 链**：
- WAITCNT 全 stall（1064），其中 inner-loop 有 ~20 对**编译器自动插入的 `s_waitcnt lgkmcnt(1)`，每个 stall ~47**（≈940 cycle，WAITCNT 的绝大部分）。来自 `ds_read_b64_tr`（K/V LDS read）与 MFMA 交织时每读一对就等一次。
- top-stall 指令：`buffer_load_dwordx4 ... lds`（VMEM，stall 100-111）+ 这些 lgkmcnt(1)。
- **MFMA stall 79.6% 主要是等操作数（前置 ds_read/lgkmcnt），不是链碎片本身**。

**Stage3-4 改 PV MFMA 为 dc-outer（对齐 asm 布局）**：asm 的 PV 是 dc-outer/k-inner（每个 O 累加器 v[96:111] 连续 12 条 MFMA 背靠背 SrcC 累加）。改 `_mma1` 为 dc-outer/step-inner（每 v_o[dc] 连续 4 step），sched_barrier 从 per-step 降到 per-dc-group。
- 纯 dc-outer **去掉 sched_barrier → 大量 NaN**：根因是末条 MFMA→读结果需 11 条独立指令间隔（Table 37），sched_barrier 阻止读取提前。
- dc-outer + per-dc-group sched_barrier：正确（nan=0, MaxErr 0.0011），性能 **721/479**（与改前 720/476 持平）。MFMA 链更优、代码更净、对齐 asm 布局，但性能没动 → 印证瓶颈在 **lgkmcnt/VMEM 访存等待**，非 MFMA 链组织。

**结论 & 下一步**：当前 721/479（asm 的 63.7%/72%）。trace 明确指向**密集 lgkmcnt(1) 等待**（ds_read↔MFMA 交织）+ buffer_load…lds 暴露。下一步应对照 asm trace 看它如何用 s_setprio 双组时分复用 + LDS/VMEM ≥32cyc 间距 + 延后 waitcnt 来隐藏这些等待。需 dump asm 端 ATT trace 做 per-phase cycle 对比（Stage3 step 1-3 的 seg_asm 对比未做）。

### Stage3-5 — dump asm ATT trace 对比（颠覆"追长链"假设）

dump aiter_asm 端 ATT trace（fwd_kernel_func, seq512, 同 att yaml），与 FlyDSL dc-outer 版对比 inner-loop cycle 预算。**注意口径**：Fly inner hitcount=48 vs asm=16（asm 循环体覆盖更多 KV，per-inner cycle 不能直接相减），但类别结构与 chains 仍有强信号：

| Cat | Fly lat/w | Fly stall | ASM lat/w | ASM stall |
|---|---|---|---|---|
| MFMA | 1573 | 1253 | 975 | 778 |
| VALU | 1138 | 35 | 1370 | 134 |
| VMEM | 1145 | 922 | 979 | 641 |
| WAITCNT | 1065 | 1065 | 1476 | 1476 |
| LDS | 800 | 352 | 1216 | 784 |
| **TRANS** | **520** | 0 | **44** | 12 |

**MFMA chains**：Fly `[...,13,16, 2,1,1,2,1,1,...(PV碎片)]` vs **asm `[1×44, 4]`（几乎全单条）**。

**颠覆性发现**：
1. **asm 不追长链**——它把 MFMA **单条散开**，靠指令级与 VALU/exp/ds_read 交织来隐藏延迟（playbook Pattern 2：用 MFMA 32-cyc 执行窗口并行发 VMEM/LDS/VALU）。我之前 dc-outer "追长链"方向错了：dc-outer 改完 trace 里 PV 仍是 2-1-1 碎片，且性能没动。
2. **TRANS 差距巨大**：Fly exp2 等超越函数 520 cyc/wave，asm 仅 44 —— **FlyDSL 的 softmax exp2 是整块串行**（两 MMA0 后、MMA1 前），没塞进 MFMA 执行窗口；asm 把 exp2 交织进 MFMA#15→#16 之间的窗口。
3. **MFMA stall 1253 vs 778**：Fly MFMA 更多在等操作数（前置 ds_read/lgkmcnt 未提前）。

**正确方向（修正）**：不是追 MFMA 长链，而是 (a) 把 softmax exp2/pack 拆开**交织进 MFMA 执行窗口**（Pattern 2）；(b) ds_read 提前发射 + 增量 lgkmcnt 让 MFMA 不等操作数（Pattern 6 waitcnt 延后）；(c) 可选 s_setprio。这些是 compute-bound 下隐藏 VALU/访存延迟的真杠杆。

### Stage3-6 — exp2 交织进 MFMA 窗口（参考 Stage-A 手法）

参考 flash_attn_gfx950.py(1446-1458) Stage-A 的交织手法：把 sub-tile b 的 2nd-half exp2 + sum + cast **延后到 sub-tile a 的 PV MFMA 之后**发射，并用 `_sched_barrier_exp_pairs(8,2,1)`（IGroupLP sched_group_barrier，MFMA 配 EXP/VALU）提示编译器把 v_exp 塞进 MFMA 执行窗口。sub-tile a 仍做完整 softmax。

**结果**：正确（nan=0, MaxErr 0.0011）；性能 **721.7/477.9**（与改前 721/479 持平，没提速）。

**trace 验证（关键）**：交织**确实生效**——TRANS 524 lat 但 **stall 从 ~0 维持、MFMA lat 1573→1379**（exp2 不再卡在关键路径）。但**总 cycle 没降**，因为瓶颈已转移/本就在访存：

| Cat | lat/w | stall | %Tot | Waste% |
|---|---|---|---|---|
| MFMA | 1379 | 1059 | 20.5% | 76.8% |
| VALU | 1272 | 61 | 18.9% | 4.8% |
| **VMEM** | **1255** | **1029** | 18.7% | **82%** |
| **WAITCNT** | **1000** | **1000** | 14.9% | **100%** |
| LDS | 794 | 346 | 11.8% | 43.6% |
| TRANS | 524 | 4 | 7.8% | 0.8% |

**精确瓶颈定位**：VMEM(`buffer_load…lds`) + WAITCNT(lgkmcnt) 合计 ~2255 cyc 全 stall（33%）= **访存延迟未隐藏**。虽然理论 AI 是 compute-bound，但实测 kernel 卡在访存等待——因为 merged-128 **流水线只领先 1 对 K/V，DMA 延迟暴露在关键路径**。MFMA stall 也主要是等操作数（ds_read 未提前）。

**确定的下一步杠杆**：**加深流水线**（领先 2+ 对 K/V，需扩 LDS 到 4 区 ping-pong，68KB→136KB，160KB 内可放），把 DMA 完全藏到计算后。这是隐藏 VMEM+WAITCNT stall 的根本手段。exp2 交织保留（已消一类 stall，且腾出 MFMA 窗口供后续利用）。

### Stage3-7 — 扩 LDS + V 双缓冲领先预取（尝试，净收益为零，已回退）

trace 显示 V 的 `buffer_load…lds` 是最大 stall（158/134）且 V 完全无领先预取。尝试：LDS 从 interleaved [K0V0K1V1] 改为分段 [K0K1][V0V1V2V3]，**K2+V4**（116.8KB，160KB 内）。V 双缓冲 ping-pong（组 {0,1}/{2,3}），prologue 预取 V pair0，循环内读组 g、预取下一对到组 1-g，runtime `v_grp=((t-t0)/2)%2` 选组。

LDS 容量实测：K4+V4=165.5KB **超**；K3+V4=141KB、**K2+V4=116.8KB** OK；4 区等宽需 padding=0（160.0KB 零裕量，bank-conflict 风险）放弃。

**结果**：正确（nan=0, MaxErr 0.0011），性能 **710/474，反而比 exp2 交织版 721/479 略降**。trace：VMEM stall 922→856（V 领先确实降了 DMA stall），但 **runtime v_grp 的 SALU 涨 144→188** 抵消，且 **WAITCNT 仍 997 全 stall 没动**。净墙钟无改善 → **回退**。

**结论**：V 双缓冲不是瓶颈所在。真顽疾是 **WAITCNT(lgkmcnt) 997 全 stall + BARRIER** —— 即 `ds_read`（K/V LDS→VGPR）与消费 MFMA 之间的 lgkmcnt 等待（~20 对 lgkmcnt(1) stall~47），以及每 iter 多个 s_barrier。VMEM(DMA) 不是关键路径（已基本被 MMA0/softmax 覆盖）。

**修正方向**：下一步应攻 **lgkmcnt/barrier**，而非访存预取。候选：(a) 重排 ds_read 发射顺序让 backend 发增量 lgkmcnt(N→0) 而非一次性等待（catalog: 第一个被消费操作数的 read 先发）；(b) 减少每 iter 的 s_barrier 数（当前读K前+读K后+读V前 共3个）；(c) ds_read 间插 SALU 凑 ≥32cyc 间距避免 LDS 队列满。这些是 FlyDSL 前端可控、且直击 trace 头号 stall 的杠杆。当前稳定版仍为 exp2 交织版 720/486（asm 的 62%/73%）。

## ISA 对齐（新目标：性能不论，让 FlyDSL ISA 与 asm ISA 对齐）

### 对齐-1 基线：循环体指令直方图对比

dump FlyDSL final ISA（21_final_isa.s, 2412行），定位循环体。**关键：FlyDSL 经 loop rotation**——稳态循环体是 `.LBB0_2`(652) 到 `.LBB0_5`(1708)，回边 1710 跳回 652；`.LBB0_3`(1340) 是循环中段（PV）。exp2 在 695-919（循环体前段，确实在体内、与 MMA0 交织）。一开始误框 1340-1708 导致以为 exp=0。

正确循环体（652-1708，1 轮=merged-128=2 子块）vs asm 单 128-tile，每 64-tile 归一化：

| 类别 | FlyDSL | asm | Δ |
|---|---|---|---|
| MFMA | 40 | 40 | **0 ✓** |
| TRANS(exp) | 32 | 32 | **0 ✓** |
| LDS(ds_read) | 56 | 56 | **0 ✓** |
| VMEM | 10 | 10 | **0 ✓** |
| BARRIER | 2 | 2 | **0 ✓** |
| VALU | 182 | 244 | -62 |
| SALU | 16 | 20 | -4 |
| **WAITCNT** | **18** | **2** | **+16** |
| **NOP** | **23** | **2** | **+21** |

**结论**：MFMA/TRANS/LDS/VMEM/BARRIER 已完全对齐（exp2 交织 + dc-outer MFMA 顺序生效）。剩两个真实差距：**WAITCNT +16、NOP +21**。根因：V 读用 inline_asm `ds_read_b64_tr_b16` 带 `~{memory}` clobber → 编译器对每个 V ds_read 保守插 waitcnt + s_nop 填 MFMA hazard，无法像 asm（手写/完全展开）那样跨 ds_read 批量等待。

**对齐方向**：收紧 V ds_read 的 inline_asm 副作用约束（`~{memory}` → 更精确的 LDS-only），或用 scope/sched_barrier 引导编译器合并 waitcnt、消除冗余 nop。VALU 少 62 是 asm 多的 mask cndmask（cross-seqlen），非核心。

### 对齐-2: WAITCNT/NOP 根因 = AGPR 累加器搬运（非 ~{memory}）

试 `~{memory}` → 精确副作用：先发现 ROCDL 有 `ds_read_tr16_b64` intrinsic（支持 alias_scopes），但 **gfx950 backend `Cannot select` 该 intrinsic**（LLVM 无 codegen pattern）；`lds_transpose_load` 是 gfx1250 专用。回退 inline_asm。再试去掉 `~{memory}` clobber（`"=v,v,~{memory}"`→`"=v,v"`）：正确性 OK 但 **WAITCNT/NOP 完全没变**（仍 18/23）→ 证明 inline_asm clobber 不是元凶，已回退。

**真根因（逐指令查循环体 1340-1521）**：密集的 `lgkmcnt(1)/lgkmcnt(0)` 交替来自 **MFMA 累加器的 AGPR↔VGPR 搬运**。FlyDSL 循环体有 **55 个 `v_accvgpr_read_b32`**（从 AGPR 读累加器到 VGPR），asm **0 个**。模式：
```
ds_read_b128 ×2
v_accvgpr_read_b32 ×4   ← 累加器从 AGPR 搬到 VGPR（asm 无此步）
s_waitcnt lgkmcnt(1)
v_mfma ..., v[64:67], v[140:143]   ← FlyDSL: VGPR 操作数
```
对比 asm：`v_mfma ..., acc[48:51], acc[0:3]` —— **直接以 AGPR 为 MFMA src/dst，无搬运**。

FlyDSL 的 `fly.mma_atom_call_ssa` 生成 VGPR 操作数的 MFMA，寄存器分配器把累加器值溢到 AGPR（gfx950 AGPR/VGPR 共享物理文件，见记忆 [[gfx942-gfx950-agpr-vgpr-shared-file]]）再 `v_accvgpr_read` 读回，每次都打断 ds_read 批量 → 碎片化 waitcnt + s_nop（MFMA hazard 填充）。这 55 个搬运 + 配套 waitcnt/nop = WAITCNT+16/NOP+21 差异的真正来源。

**修正方向**：让 MFMA 累加器全程留 VGPR（不溢 AGPR）或直接以 AGPR 为操作数（如 asm）。这是编译器 RA/lowering 层。前端可试 `_anchor_v_o`（inline-asm 把累加器 pin 在 VGPR）固定落位，或调 VGPR 预算/mma atom 配置。属深层，需验证 FlyDSL 可控性。
