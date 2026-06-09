# bench_up_moe_v1.py 两条 MoE 路径的 Kernel 调度总览

> 本文档分析 `aiter/bench_up_moe_v1.py` 第 134-135 行所计时的两个函数会调用哪些 GPU kernel，
> 以及这些 kernel（含全部 sort 相关 kernel）如何随 token 数 `M` 变化而被选择。
> 数据来源：`aiter/run.sh` 第 30-31 行的运行日志 `aiter/tt.log`（`AITER_LOG_MORE=1`，MI355X）
> 与源码 `aiter/aiter/fused_moe.py` / `aiter/csrc/kernels/mxfp4_moe/` / `aiter/csrc/kernels/quant_kernels.cu`
> / `aiter/csrc/include/moe_sorting_opus.h` / `aiter/aiter/ops/flydsl/`。

---

## 0. 测试配置（ground truth）

- **硬件**：AMD Instinct **MI355X**（gfx950，`cu_num = 256`，分 8 个 XCD）。
- **模型形状**（`bench_up_moe_v1.py:30` `KIMI`，Kimi-K2.5 TP=4）：
  - `NE = 385` 专家（含 1 个 shared expert，`shared_id = NE-1`）
  - `H = model_dim = 7168`
  - `INTER = 每分片 inter_dim = 512`（`w1` 沿 N 维堆叠 gate||up，故 `w1.shape[1] = 2*512 = 1024`）
  - `TOPK = 9`（其中 1 个固定为 shared expert，8 个 routed）
- **数据类型**：bf16 激活；权重与激活都量化为 **MXFP4**（`fp4 e2m1` 每字节 2 个 + 每 32 元素一个 `e8m0` 缩放字节）；`QuantType.per_1x32`，`ActivationType.Silu`。
- **被测 M 序列**：`4, 8, 16, 32, 64, 128, 256, 4096, 8192, 16384`（`run.sh:31`）。

性能结果（`tt.log`，越小越好，单位 µs；`fly/mx` 为 flydsl/mxfp4 比值）：

| M | mxfp4 (mx_fn) | flydsl (fly_fn) | fly/mx |
|--:|--:|--:|--:|
| 4 | 43.2 | 57.8 | 1.34x |
| 8 | 59.9 | 75.1 | 1.25x |
| 16 | 100.2 | 121.2 | 1.21x |
| 32 | 146.3 | 162.2 | 1.11x |
| 64 | 199.2 | 212.1 | 1.07x |
| 128 | 249.9 | 263.4 | 1.05x |
| 256 | 310.4 | 305.4 | 0.98x |
| 4096 | 854.3 | 871.0 | 1.02x |
| 8192 | 1141.9 | 1354.7 | 1.19x |
| 16384 | 1834.2 | 2310.2 | 1.26x |

---

## 1. 两个被测函数到底是什么

`bench_up_moe_v1.py:134-135`：

```python
_, mx_us  = run_perftest(mx_fn,  num_warmup=args.warmup, num_iters=args.iters)
_, fly_us = run_perftest(fly_fn, num_warmup=args.warmup, num_iters=args.iters)
```

`mx_fn` 与 `fly_fn` 都是 `make_fn(...)` 返回的闭包（`bench_up_moe_v1.py:84-98`），二者**只有权重不同**，都调用同一个入口 `aiter.fused_moe.fused_moe(...)`：

| 函数 | 权重来源 | `w1.shuffle_kind` | 走的实现 |
|---|---|---|---|
| **`mx_fn`** | `mx_w`（`shuffle_weight_a16w4` 布局，`bench:55-62`）| `"mxfp4_moe"` | **MXFP4 专用流水线**（PR #3470，C++/HIP 手写 kernel） |
| **`fly_fn`** | `fly_w`（`shuffle_weight(layout=(16,16))` 旧布局，`bench:44-51`）| `None` | **FlyDSL 2-stage 路径**（opus sorting + FlyDSL DSL GEMM） |

二者计算的数学是**同一个 MoE-MLP（up/gate 投影 + SiLU·mul + down 投影 + topk 加权求和）**，只是 kernel 实现不同；因此 `cos ≈ 0.98+`。

### 调度入口（`fused_moe.py`）

```
fused_moe → fused_moe_  (fused_moe.py:203 / 297)
  ├─ get_2stage_cfgs(get_padded_M(M), H, INTER, NE, TOPK, ...,
  │                   shuffle_kind=w1.shuffle_kind)        # 查 tuned CSV → kernelName1/2
  ├─ if metadata.pipeline is not None:   # mxfp4 (kernelName 形如 mxfp4_moe_g1_...)
  │      return _mxfp4_moe_run(...)      # ← mx_fn 走这里 (fused_moe.py:1043)
  └─ else:                               # flydsl (kernelName 形如 flydsl_moe1_...)
         moe_sorting(...)                # opus 排序 (fused_moe.py:147 → _moe_sorting_impl:70)
         fused_moe_2stages(...)          # ← fly_fn 走这里 (fused_moe.py:2012)
```

- `get_2stage_cfgs`（`fused_moe.py:1305`）按 `(cu_num, token_tier, H, INTER, NE, TOPK, act, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, doweight_stage1)` 查 tuned CSV，得到 `kernelName1`/`kernelName2`。
  - `token_tier = get_padded_M(M)`（`fused_moe.py:796`）：`M<32768` 时取 `nextPow2(M)`。被测 M 的 tier 就是 M 本身（4/8/.../256/4096/8192/16384）。
  - `shuffle_kind="mxfp4_moe"` 时优先查带 `_tag=mxfp4_moe` 的行（kimik2_5_mxfp4_tuned_fmoe.csv）→ kernelName 形如 `mxfp4_moe_g1_...` → `_is_mxfp4_kname` 为真 → 返回 `MOEMetadata(pipeline=_mxfp4_moe_run)`。
  - `shuffle_kind=None` 时查未打 tag 的行 → kernelName 形如 `flydsl_moe1_...` → 返回 `MOEMetadata(stage1=_flydsl_stage1_wrapper, stage2=_flydsl_stage2_wrapper, pipeline=None)`。
- **kernel 选择完全由 tuned CSV 按 M-tier 驱动**；CSV 未命中时才落到启发式（`fused_moe.py:1539+` / `:1803+`）。

---

## 2. 两条路径都遵循的 2-stage MoE 结构

```
                 ┌──────────── 排序 / 量化（prologue）────────────┐
 hidden(bf16)──▶ │ ① 路由排序 (sort)  ② 激活量化 bf16→mxfp4       │
                 └───────────────────────────────────────────────┘
                                │ (sorted, a_q+a_scale)
                                ▼
 GEMM1 (stage1):  a_q × w1 → gate/up → y = SiLU(gate)·up → 中间量(每 token×topk)
                                │ (中间量需再量化为 mxfp4 给 stage2)
                                ▼
 GEMM2 (stage2):  inter_q × w2 → 每个 (token, slot) 的 down 投影
                                │
                                ▼
 topk reduce: 把同一 token 的 topk 个 slot 结果按路由权重加权求和 → out(bf16)
```

两条路径的区别在于**每一步用谁的 kernel、哪些步骤被融合、以及 topk 归约用 atomic 还是单独 reduce**——这正是随 M 变化的关键。

---

## 3. MXFP4 路径（`mx_fn`）的 kernel 清单与随 M 选择

入口：`_mxfp4_moe_run`（`fused_moe.py:1043`）。所有 kernel 在 `aiter/csrc/kernels/mxfp4_moe/`。
该函数按 `kernelName1`/`kernelName2` 解析出三个开关：
- `inline_quant`（g1 名含 `INLINEQUANT`）：激活量化是否融进 gemm1；
- `atomic`（g2 名含 `ATOMIC`）：gemm2 是否直接 atomic-add 到输出（省掉 scatter_reduce）；
- `mxfp4out`（g2 名含 `MXFP4OUT`）：gemm2 是否把中间结果以 fp4 暂存，scatter_reduce 再解包。

### MXFP4 kernel 全集（10 个）

| # | GPU kernel（trace 名）| host op | 作用 | 文档 |
|---|---|---|---|---|
| M1 | `moe_sort_quant::sort_quant_kernel_impl` | `mxfp4_moe_sort`(prologue=0) | 小 M：单 kernel 排序 + （可选）输出清零 | [mxfp4_01](mxfp4_01_sort_quant.md) |
| M2 | `moe_3stage_sort::sort_count_kernel_impl` | `mxfp4_moe_sort`(prologue=1) | 三段排序-1：每专家计数 | [mxfp4_02](mxfp4_02_3stage_sort.md) |
| M3 | `moe_3stage_sort::sort_cumsum_kernel_impl` | 同上 | 三段排序-2：前缀和 + 写 sorted_expert_ids | [mxfp4_02](mxfp4_02_3stage_sort.md) |
| M4 | `moe_3stage_sort::sort_place_pad_kernel_impl` | 同上 | 三段排序-3：放置 token + padding | [mxfp4_02](mxfp4_02_3stage_sort.md) |
| M5 | `moe_sort_quant::quant_kernel_impl` | `mxfp4_moe_quant` | 三段路径：激活 bf16→mxfp4 量化 | [mxfp4_03](mxfp4_03_quant.md) |
| M6 | `moe_sort_scales::sort_scales_kernel_impl` | `mxfp4_moe_sort_scales` | 把 a_scale 按 sorted 顺序 gather + shuffle 成 GEMM tile 布局 | [mxfp4_04](mxfp4_04_sort_scales.md) |
| M7 | `gemm1::kernel` | `mxfp4_moe_gemm1_a4w4` | GEMM1：gate/up + SiLU·mul + 中间量再量化 | [mxfp4_05](mxfp4_05_gemm1.md) |
| M8 | `gemm2::kernel` | `mxfp4_moe_gemm2_a4w4[_mxfp4out]` | GEMM2：down 投影（atomic/nonatomic/mxfp4out） | [mxfp4_06](mxfp4_06_gemm2.md) |
| M9 | `moe_scatter_reduce::scatter_reduce_kernel_impl` | `mxfp4_moe_scatter_reduce` | topk 加权求和（bf16 输入） | [mxfp4_07](mxfp4_07_scatter_reduce.md) |
| M10 | `moe_scatter_reduce::scatter_reduce_mxfp4_kernel` | `mxfp4_moe_scatter_reduce_q` | topk 加权求和（mxfp4 暂存输入） | [mxfp4_07](mxfp4_07_scatter_reduce.md) |

### 随 M 的实际选择（来自 `tt.log` 第一轮，按执行顺序）

| M | BM | prologue | gemm2 模式 | 实际跑的 GPU kernel（按序）|
|--:|--:|---|---|---|
| 4 / 8 / 16 / 32 / 64 / 128 | 16 | inline_quant | ATOMIC | `sort_quant_kernel_impl` → `gemm1(BM16,inlineQ)` → `gemm2(BM16,atomic)` （**3 个**）|
| 256 | 32 | threestage | ATOMIC | `sort_count`+`sort_cumsum`+`sort_place_pad` → `quant_kernel` → `sort_scales` → `gemm1(BM32)` → `gemm2(BM32,atomic)` （**7 个**）|
| 4096 | 128 | threestage | NONATOMIC | 三段 sort → `quant` → `sort_scales` → `gemm1(BM128)` → `gemm2(BM128,nonatomic)` → `scatter_reduce`(bf16) （**8 个**）|
| 8192 / 16384 | 128 | threestage | NONATOMIC + MXFP4OUT | 三段 sort → `quant` → `sort_scales` → `gemm1(BM128)` → `gemm2_mxfp4out(BM128)` → `scatter_reduce_q`(mxfp4) （**8 个**）|

要点：
- **小 M（≤128）BM=16，把排序+清零融成 1 个 kernel，激活量化融进 gemm1，gemm2 用 atomic-add 省掉 reduce** → 只有 3 个 kernel，launch 开销最低。
- **M=256 起改用三段并行排序**（更高并行度），激活量化与 scale-shuffle 拆成独立 kernel。
- **大 M（≥4096）BM=128 + nonatomic**：gemm2 写出每个 (token,slot) 行，再用 scatter_reduce 做 topk 归约；M≥8192 进一步用 **MXFP4OUT**（gemm2 把中间结果压成 fp4 暂存，scatter_reduce 读取量减少约 3.8×）。

---

## 4. FlyDSL 路径（`fly_fn`）的 kernel 清单与随 M 选择

入口：`moe_sorting`（`fused_moe.py:147`）+ `fused_moe_2stages`（`fused_moe.py:2012`）。
排序走 **opus**（`csrc/include/moe_sorting_opus.h`），激活/中间量化走 `csrc/kernels/quant_kernels.cu`，
GEMM 走 **FlyDSL DSL**（`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`）。

> 注意：CSV/配置名是 `flydsl_moe1_...`/`flydsl_moe2_...`；编译出的 MLIR 符号名（即 trace 里看到的）
> 是 `mfma_moe1_silu_mul_...`/`mfma_moe2_..._cshuffle_..._vscale_fix3_...`。二者一一对应（见各 kernel 文档）。

### FlyDSL kernel 全集（7 类）

| # | GPU kernel（trace 名）| host op | 作用 | 文档 |
|---|---|---|---|---|
| F1 | `fused_mx_quant_moe_sort_kernel` | `fused_dynamic_mx_quant_moe_sort_hip` | 小 M：融合「激活 bf16→mxfp4 量化 + 把 scale 写成 sorted tile 布局」 | [flydsl_01](flydsl_01_fused_quant_sort.md) |
| F2 | `dynamic_per_group_scaled_quant_kernel` | `per_1x32_mx_quant_hip` | 大 M：纯激活量化（每 token 读一次）| [flydsl_02](flydsl_02_dynamic_quant.md) |
| F3 | `mxfp4_moe_sort_kernel` | `mxfp4_moe_sort_hip` | 大 M：把 e8m0 scale 按 sorted 顺序 shuffle 成 tile 布局 | [flydsl_03](flydsl_03_scale_sort.md) |
| F4 | `opus_moe_sorting_entry<...>` | `moe_sorting_opus_fwd` | 路由排序（single / multi-phase 多变体）| [flydsl_04](flydsl_04_opus_sorting.md) |
| F5 | `mfma_moe1_silu_mul_afp4_wfp4_*` | flydsl stage1 | GEMM1：gate/up + SiLU·mul（可融合中间 fp4 量化）| [flydsl_05](flydsl_05_moe1_gemm.md) |
| F6 | `mfma_moe2_afp4_wfp4_bf16_cshuffle_*` | flydsl stage2 | GEMM2：down 投影（atomic / reduce 模式）| [flydsl_06](flydsl_06_moe2_gemm.md) |
| F7 | `aten::sum` → `at::native::reduce_kernel` | torch | reduce 模式下的 topk 归约（PyTorch 内置）| [flydsl_07](flydsl_07_topk_reduce.md) |

### 各组件随 M 的内部派发阈值

- **激活量化（stage1 输入）`fused_dynamic_mx_quant_moe_sort`**（`quant.py:878`）：
  - `M ≤ 8*256/topk ≈ 227`（即 M=4..128）→ **F1** 融合 kernel（1 个）；
  - `M ≥ 256` → **F2 + F3** 拆分（量化每 token 只读一次，更省带宽）。
- **中间量化（stage2 输入）**：仅当 stage1 kernel 名**不含** `_fp4`（`fuse_quant==""`）时才单独跑（`fused_moe.py:2238`），否则被 gemm1 融合（`_fp4q`）。被测里 M≥8192 的 stage1 名为 `...w2_xcd4`/`...w4_bnt0_xcd4`（无 `_fp4`）→ 中间量化单独跑（再次 F2+F3）。
- **opus 排序**（`moe_sorting_opus.h`，按 `tokens` 与 LDS 容量）：
  - `M ≤ ~32`（`tokens ≤ sub_token`）→ **single** `MoeSortingKernel`（1 个）；
  - `32 < M < 2048` → **mp-small**：`P0_v2 + P23`（2 个）；
  - `M ≥ 2048`（即 4096/8192/16384）→ **mp-full**：`ClearWorkspace + P0_v1 + P1 + P23`（4 个）。
- **gemm2 归约模式**（CSV 决定 `kernelName2`）：小/中 M 用 `atomic`（gemm2 原子加到输出）；大 M 用 `reduce`（gemm2 写每 (token,slot)，再 `torch.sum` 归约 = F7）。

### 随 M 的实际选择（来自 `tt.log` 第一轮，按执行顺序）

| M | 激活量化 | opus 排序 | gemm1 | gemm2 | 中间量化 | topk归约 |
|--:|---|---|---|---|---|---|
| 4 / 8 / 16 | F1 (fused) | single | moe1 `_fp4q`（融合中间量化）| moe2 `atomic` | （融合在 gemm1）| atomic |
| 32 | F1 | single | moe1 | moe2 `atomic` | （融合）| atomic |
| 64 / 128 | F1 | `P0_v2`+`P23` | moe1 `_fp4q` | moe2 `atomic`/`persist` | （融合）| atomic |
| 256 | **F2+F3** | `P0_v2`+`P23` | moe1 `_fp4q`(t64) | moe2 `atomic_persist_sbm64` | （融合）| atomic |
| 4096 | **F2+F3** | `ClearWS`+`P0_v1`+`P1`+`P23` | moe1 `t128`（无 fp4q）| moe2 `reduce_persist_sbm128` | **F2+F3**（独立）| **F7**(torch sum) |
| 8192 | F2+F3 | mp-full(4) | moe1 `xcd4`（无 fp4q）| moe2 `reduce_persist_sbm128` | F2+F3 | F7 |
| 16384 | F2+F3 | mp-full(4) | moe1 `t64 xcd4` | moe2 `reduce_xcd4` | F2+F3 | F7 |

> 说明：autotune warmup 期间偶尔会出现额外的 `fused_mx_quant_moe_sort_kernel` 模板实例（如 M=32 的
> `<...,64,8>`+`<...,256,32>`），属于探测产物，稳态只跑上表所列 kernel。

---

## 5. 「sort 相关」kernel 横向对照（两条路径）

MoE 的核心是把「按 token 排列的 topk 路由」重排成「按专家分组、每组 padding 到 block 倍数」的布局，
让 GEMM 每个 M-block 只服务一个专家。两条路径各有一套 sort 实现：

| 功能 | MXFP4 路径 | FlyDSL 路径 |
|---|---|---|
| 计数+前缀和+放置（小 M）| `sort_quant_kernel_impl`（单 kernel，单 block 串行）| opus `MoeSortingKernel`（单 kernel，LDS mesh）|
| 计数（大 M）| `sort_count_kernel_impl`（多 CTA 分块计数）| opus `MoeSortingMultiPhaseKernel_P0_v1/v2` |
| 前缀和 / 计数归约 | `sort_cumsum_kernel_impl`（单 CTA 扫描）| opus `..._P1`（+ `ClearWorkspaceKernel` 清 workspace）|
| 放置 + padding | `sort_place_pad_kernel_impl`（多 CTA）| opus `..._P23` |
| scale 重排到 sorted tile 布局 | `sort_scales_kernel_impl` | `mxfp4_moe_sort_kernel`（大 M）/ 融合进 F1（小 M）|

两套排序输出的 `sorted_token_ids` 都用同一种 **fused id 编码**：`fused = (slot << 24) | token_id`（24 位 token + 8 位 topk slot）。
padding 槽位用 `token_id = M`（越界）让 GEMM 的 `buffer_load` 被硬件丢弃。

---

## 6. 共享底层概念

- **MXFP4 量化**：每 32 个元素一组，组内取 `absmax`，编码一个 `e8m0`（8 位 2 的幂指数）缩放字节，
  组内每个元素量化为 `fp4 e2m1`（4 位：1 符号 + 2 指数 + 1 尾数，2 个打包进 1 字节）。
  反量化 `value = fp4 * 2^(e8m0-127)`。所有 GEMM 都是 `a4w4`（激活 fp4 × 权重 fp4）。
- **scaled MFMA**：gfx950 用 `mfma_scale_f32_16x16x128_f8f6f4`（M16×N16×K128），在 MFMA 内部按 e8m0
  对 A/B 做反量化缩放（`cbsz=4`/`blgp=4` 选 fp4 子格式）。
- **fused id**、**sorted_expert_ids**、**num_valid_ids**、**reverse_sorted** 的定义见各 kernel 文档与
  `aiter/op_intro_template.txt`（模板里 §3.3/§6 给了 fused id 与排序示意）。

---

## 7. 各 kernel 详细文档索引

MXFP4 路径（`mx_fn`）：
- [mxfp4_01_sort_quant.md](mxfp4_01_sort_quant.md) — 小 M 单 kernel 排序（M1）
- [mxfp4_02_3stage_sort.md](mxfp4_02_3stage_sort.md) — 三段并行排序（M2/M3/M4）
- [mxfp4_03_quant.md](mxfp4_03_quant.md) — 激活量化（M5）
- [mxfp4_04_sort_scales.md](mxfp4_04_sort_scales.md) — scale 排序+shuffle（M6）
- [mxfp4_05_gemm1.md](mxfp4_05_gemm1.md) — GEMM1（M7）
- [mxfp4_06_gemm2.md](mxfp4_06_gemm2.md) — GEMM2（M8）
- [mxfp4_07_scatter_reduce.md](mxfp4_07_scatter_reduce.md) — topk 归约 bf16/mxfp4（M9/M10）

FlyDSL 路径（`fly_fn`）：
- [flydsl_01_fused_quant_sort.md](flydsl_01_fused_quant_sort.md) — 融合量化+scale排序（F1）
- [flydsl_02_dynamic_quant.md](flydsl_02_dynamic_quant.md) — 激活量化（F2）
- [flydsl_03_scale_sort.md](flydsl_03_scale_sort.md) — scale 排序（F3）
- [flydsl_04_opus_sorting.md](flydsl_04_opus_sorting.md) — opus 路由排序（F4）
- [flydsl_05_moe1_gemm.md](flydsl_05_moe1_gemm.md) — GEMM1（F5）
- [flydsl_06_moe2_gemm.md](flydsl_06_moe2_gemm.md) — GEMM2（F6）
- [flydsl_07_topk_reduce.md](flydsl_07_topk_reduce.md) — topk 归约（F7）
