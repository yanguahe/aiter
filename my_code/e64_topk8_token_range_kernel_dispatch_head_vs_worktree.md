# E64/topk8 不同 token 范围的 GEMM1/GEMM2 kernel dispatch

## 测试规模

本文分析下面规模仅改变 `--tokens`，且 `tokens` 范围为 `1–4098` 时，
GEMM1 和 GEMM2 实际命中的 kernel：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

比较以下两套代码状态：

1. `moe_a4w4_pr_refactor_wt` 分支的纯 `HEAD`：

   ```text
   11b389a896a93446ada07557e6af66d1ff78a851
   ```

2. 同一 `HEAD` 加上当前工作区未提交修改。

假设：

- 仅修改 `--tokens`；
- 其他参数保持不变；
- 不设置 `AITER_TDM_TILE_*` 等 tile override；
- 使用 `run_moe_prefill_switch_ab.sh`，因此
  `AITER_MOE_EXPERT_BALANCE=true`。

## token bucket 规则

CSV 查找使用：

```python
token_bucket = next_power_of_2(tokens)
```

在 `tokens=1–4098` 范围内：

| tokens | CSV token bucket | 本规模是否有精确 CSV row |
|---:|---:|:---:|
| 1–1024 | 1–1024 对应的 2 次幂 bucket | 否 |
| 1025–2048 | 2048 | 是 |
| 2049–4096 | 4096 | 否 |
| 4097–4098 | 8192 | 是 |

没有 CSV row 时，当前 grouped MoE TDM 路径使用默认参数：

```text
tile_m=64
tile_n=256
tile_k=256
m_warp=1
n_warp=4
num_buffers=3
```

## 纯 HEAD 的 dispatch

HEAD 中 `token=2048` 的 CSV row 为：

```text
GEMM1: tile_m=192, m_warp=2, n_warp=2, b4
GEMM2: tile_m=192, m_warp=2, n_warp=4, b4
```

### token 区间

| tokens | GEMM1 | GEMM2 |
|---:|---|---|
| 1–1024 | 默认 `t64/w1x4/b3` | 默认 `t64/w1x4/b3` |
| 1025–1535 | optimized `t192/w2x2/b4` | persistent `t192/w2x4/b4` |
| **1536** | **persistent `t192/w2x4/b4`** | **persistent `t192/w2x4/b4`** |
| 1537–2048 | optimized `t192/w2x2/b4` | persistent `t192/w2x4/b4` |
| 2049–4096 | 默认 `t64/w1x4/b3` | 默认 `t64/w1x4/b3` |
| 4097–4098 | optimized `t256/w2x2/b4` | optimized `t256/w2x2/b4` |

### 主要 kernel symbol

默认 GEMM1：

```text
a8w4_tdm_fp4_t64x256x256_w1x4_b3_K7168_e64_act1_q1r4...
```

默认 GEMM2：

```text
a8w4_tdm_fp4_t64x256x256_w1x4_b3_K2048_e64...
```

HEAD t192 GEMM1 optimized：

```text
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_q1r6...
```

HEAD t192 GEMM1 persistent，仅 `tokens=1536`：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist
```

HEAD t192 GEMM2 persistent：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist
```

`tokens=4097–4098` 命中的 t256 optimized kernel：

```text
GEMM1:
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cn4_prefetch_apre

GEMM2:
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre
```

因此，纯 HEAD 下要让 GEMM1 和 GEMM2 同时命中 t192 persistent，必须：

```text
tokens = 1536
```

## HEAD 加当前工作区修改后的 dispatch

当前工作区将 `token=2048` 的 CSV row 改为：

```text
GEMM1: tile_m=256, m_warp=2, n_warp=2, b4
GEMM2: tile_m=256, m_warp=2, n_warp=4, b4
```

同时增加了 balanced `_vm192` 专化。

### token 区间

| tokens | GEMM1 | GEMM2 |
|---:|---|---|
| 1–1024 | 默认 `t64/w1x4/b3` | 默认 `t64/w1x4/b3` |
| 1025–1505 | optimized `t256/w2x2/b4` | persistent `t256/w2x4/b4`，完整 t256 compute |
| 1506–1535 | optimized `t256/w2x2/b4` | persistent `t256/w2x4/b4_vm192` |
| **1536** | **persistent `t256/w2x4/b4_vm192`** | **persistent `t256/w2x4/b4_vm192`** |
| 1537 | optimized `t256/w2x2/b4` | persistent `t256/w2x4/b4_vm192` |
| 1538–2048 | optimized `t256/w2x2/b4` | persistent `t256/w2x4/b4`，完整 t256 compute |
| 2049–4096 | 默认 `t64/w1x4/b3` | 默认 `t64/w1x4/b3` |
| 4097–4098 | optimized `t256/w2x2/b4` | optimized `t256/w2x2/b4` |

### 主要 kernel symbol

t256 GEMM1 optimized：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cn4_prefetch_apre
```

t256 GEMM1 persistent `_vm192`，仅 `tokens=1536`：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4_cm1_prefetch_apre_persist_vm192
```

t256 GEMM2 persistent，完整 t256 compute：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist
```

t256 GEMM2 persistent `_vm192`：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_vm192
```

## `_vm192` 的 token 范围

balanced `_vm192` 的主要判断条件是：

```python
tile_m == 256
n_experts == 64
contiguous_m == 28672
AITER_MOE_EXPERT_BALANCE == true
```

当前参数下：

```python
contiguous_m = align_up(tokens * 8 + 64 * 256 - 8, 256)
```

令 `contiguous_m == 28672`，解得：

```text
1506 <= tokens <= 1537
```

但 GEMM1 还有额外硬条件：

```python
token_num == 1536
```

因此：

- GEMM2 会在 `tokens=1506–1537` 命中 `_vm192`；
- GEMM1 只会在 `tokens=1536` 命中 persistent `_vm192`；
- GEMM1 和 GEMM2 同时命中最终 t256 persistent `_vm192` 的唯一值是：

  ```text
  tokens = 1536
  ```

## `tokens=1537` 的边界风险

`tokens=1537` 时总 route 数为：

```text
1537 * 8 = 12296
```

balanced 分配到64个 expert 后：

- 56个 expert 为192行；
- 8个 expert 为193行。

当前代码仍会因为 `contiguous_m == 28672` 为 GEMM2 选择 `_vm192`，但
`_vm192` 只计算每个 expert 的前192行。因此 `tokens=1537` 存在超出
192行计算假设的正确性风险。

当前最终 `_vm192` 版本的安全复现值应固定为：

```text
tokens = 1536
```

## 总结

- 纯 HEAD 的 t192 persistent GEMM1/GEMM2 组合只在 `tokens=1536` 同时命中；
- 当前工作区的 t256 persistent `_vm192` GEMM1/GEMM2 组合也只在
  `tokens=1536` 同时命中；
- `tokens=1025–2048` 都会命中 CSV 的 `token=2048` row，但 GEMM1 是否进入
  w2x4 persistent 还受 `token_num == 1536` 限制；
- `tokens=2049–4096` 没有精确 CSV row，会回到默认 t64 kernel；
- `tokens=4097–4098` 会进入 `token=8192` row，使用 t256/w2x2 optimized kernel。

## 2026-10-10 全 boundary 候选 kernel 正确性与性能调优

### 测试目标与方法

本轮不再只根据 CSV bucket 和 dispatch 条件推导 kernel，而是对
`TOKEN_BOUNDARIES` 中每个值实际测试以下候选：

- `t64/w1x4/b3`；
- t192 optimized `w2x2` 与 persistent `w2x4`；
- t256 optimized `w2x2`；
- t256 persistent full `w2x4`；
- t256 persistent `w2x4_vm192`。

random 筛选除最终 `logits_diff` / `rel_l2` 外，还通过
`quant_output_capture` 检查：

- GEMM1 FP4 payload semantic mismatch；
- A-preshuffle row32 ScaleA byte mismatch；
- GEMM2 valid rows bitwise mismatch；
- GEMM2 reference/output hash。

性能测试仅对 random 正确候选执行。初筛使用2次 warmup、8次 profiler
iterations；接近的 winner 再重复3次，每次5次 warmup、20次 iterations。

隔离调优 worktree：

```text
/data/yanguahe/code/wk_sp1/aiter_e64_boundary_tune_20261010
```

### 关键正确性结论

现有 tuned A-preshuffle kernels 在 partial expert-M 边界普遍存在 ScaleA 或
GEMM2 valid-row mismatch。可靠的 tuned 点主要是 balanced 后每个 expert 行数
完全相同的4个 token 值：

| tokens | rows/expert | 可用 tuned 方案 |
|---:|---:|---|
| 1024 | 128 | t256 persistent `_vm192` |
| 1536 | 192 | t256 persistent `_vm192` |
| 2048 | 256 | t256 full persistent |
| 4096 | 512 | GEMM1 t256 optimized w2x2 + GEMM2 t256 persistent w2x4 |

其余测试边界中，唯一在整体误差和 stage-level 检查上均保持正常的通用方案是
`t64/w1x4/b3`。

`tokens=1537` 的风险已实际复现。强制 `_vm192` 会产生 ScaleA/GEMM2
valid-row mismatch，因此最终路由在1537处回到 t64，不再依赖原来的
`contiguous_m == 28672` 推断。

### 上一版 t256 `_vm192` 重复性能结果

| tokens | GEMM1 winner | GEMM1 us | GEMM2 winner | GEMM2 us |
|---:|---|---:|---|---:|
| 1024 | t256 persistent `_vm192` | ~73.7 | t256 persistent `_vm192` | ~48.9 |
| 1536 | t256 persistent `_vm192` | ~72.8 | t256 persistent `_vm192` | ~48.3 |
| 2048 | t256 full persistent | ~85.2 | t256 full persistent | ~56.5 |
| 4096 | t256 optimized w2x2 | ~156.4 | t256 persistent w2x4 | ~97.6 |

### 最终左闭右开路由

| token 区间 | GEMM1 | GEMM2 |
|---|---|---|
| `[1, 1024)` | `t64/w1x4/b3` | `t64/w1x4/b3` |
| `[1024, 1025)` | persistent `t192/w2x4/b4_pmds` | persistent `t192/w2x4/b4` |
| `[1025, 1536)` | `t64/w1x4/b3` | `t64/w1x4/b3` |
| `[1536, 1537)` | persistent `t192/w2x4/b4` | persistent `t192/w2x4/b4` |
| `[1537, 2048)` | `t64/w1x4/b3` | `t64/w1x4/b3` |
| `[2048, 2049)` | persistent full `t256/w2x4/b4` | persistent full `t256/w2x4/b4` |
| `[2049, 4096)` | `t64/w1x4/b3` | `t64/w1x4/b3` |
| `[4096, 4097)` | optimized `t256/w2x2/b4` | persistent `t256/w2x4/b4` |
| `[4097, 4099)` | `t64/w1x4/b3` | `t64/w1x4/b3` |

路由由 profile table 与左闭右开 interval table 表达，没有为每个 boundary 写独立
`if`。相邻相同区间已经合并，4个 tuned token 使用单值区间。

### 上一版 t256 `_vm192` 默认路由验证

14个 boundary 的上一版默认路由 random 全部恢复到正常 A4W4 误差范围：

```text
logits_diff ≈ 3.36e-06 ～ 3.39e-06
rel_l2 ≈ 0.00259 ～ 0.00260
pass=True
GEMM2 reference/output hash 全部一致
```

测试产物：

```text
/tmp/e64_boundary_stage_random_clean/*.jsonl
/tmp/e64_boundary_perf/*.jsonl
/tmp/e64_boundary_confirm/*.jsonl
/tmp/final_interval_random2.log
/tmp/final_interval_perf.log
```

### t192 persistent 回切、T1024 修复与 A/B 复测

按后续要求，`tokens=1024` 和 `tokens=1536` 两个单值区间均回切到 persistent
`t192/w2x4/b4`。初始 t192 在 `tokens=1024` 下出现确定性 ScaleA 错误：

```text
logits_diff = 0.0816663
rel_l2      = 0.388067
payload semantic mismatch = 0
ScaleA mismatch           = 81920 bytes
GEMM2 valid-row mismatch  = 24669510 bytes
stage_exact               = False
```

进一步统计表明：

- 8192 个 route 中，6144 行完全一致；其余2048行每行固定错40个 ScaleA byte；
- 错误只出现在每个 expert 的物理行 `96–127`，每个 expert 恰好32行；
- FP4 payload 始终一致，错误由 ScaleA 传播到 GEMM2。

t192 的 `wmma_rep=6`，每个 `wave_m` 的 ScaleA TDM tile 覆盖96行。`tokens=1536`
时每个 expert 为192行，两个96行 tile 都是完整的；`tokens=1024` 时每个 expert
只有128行，第二个 tile 只有前32行有效，其余64行没有可供原 TDM layout 使用的
row32 数据。单纯增加 `s_wait_tensorcnt 0` 没有改变 mismatch，排除了异步 drain
竞争。CDNA5 ISA §10.11.1 说明 `TENSOR_STORE_FROM_LDS` 由 `TENSORcnt` 跟踪；实验
结果表明这里的问题是确定性的 layout 缺口，而非 TDM 尚未完成。

最终修复增加 compile-time `partial_m_direct_scale` specialization：

- `tokens=1024` 的 GEMM1 使用 t192 `_pmds`，ScaleA 对有效行直接写入 canonical
  row32 global layout；
- GEMM1 FP4 payload 和 GEMM2 仍使用原 t192 persistent kernel；
- `tokens=1536` 的 specialization flag 为0，继续编译原有 LDS + TDM ScaleA 路径，
  不包含 T1024 的 runtime 分支。

两个 token 点的最终 random stage 检查均通过：

| tokens | GEMM1 kernel | payload mismatch | ScaleA mismatch | GEMM2 valid-row mismatch | logits_diff | rel_l2 | stage_exact |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 1024 | t192 persistent `_pmds` | 0 | 0 | 0 | `3.38630e-06` | `0.00260242` | True |
| 1536 | t192 persistent | 0 | 0 | 0 | `3.38491e-06` | `0.00260189` | True |

#### `tokens=1024`：t192 `_pmds` 与 t256 `_vm192`

两个版本交叉运行两批 `ROUNDS=3`，合并6个 GPU 空闲样本取中值：

| 指标 | t256 `_vm192` 6 samples (us) | t256 median (us) | t192 `_pmds` 6 samples (us) | t192 median (us) | t192 耗时变化 |
|---|---|---:|---|---:|---:|
| GEMM1 | `68.959, 71.026, 69.577, 69.634, 69.543, 69.343` | **69.560** | `69.284, 67.589, 69.934, 68.173, 69.202, 68.257` | **68.730** | **-1.19%** |
| GEMM2 | `43.570, 43.262, 43.197, 42.709, 43.140, 42.905` | **43.169** | `42.855, 43.319, 43.007, 43.732, 42.764, 43.622` | **43.163** | **-0.01%** |
| MoE e2e | `155.47, 157.63, 155.71, 155.66, 156.42, 155.27` | **155.685** | `156.61, 152.86, 155.03, 157.45, 154.37, 155.94` | **155.485** | **-0.13%** |

t192 `_pmds` 的 GEMM1 稳定快约1.19%；GEMM2 和 MoE e2e 与 t256 `_vm192`
基本持平。最终默认路由选择 t192 `_pmds`。

#### `tokens=1536`：无性能回退检查

修改前后的 t192 各取6个 GPU 空闲样本。新 specialization 在该 token 点保持 flag=0：

| 指标 | 修改前 median (us) | 修改后6 samples (us) | 修改后 median (us) | 耗时变化 |
|---|---:|---|---:|---:|
| GEMM1 | 67.954 | `66.900, 67.367, 66.853, 66.971, 67.705, 67.612` | **67.169** | **-1.15%** |
| GEMM2 | 46.699 | `46.144, 46.321, 46.218, 46.224, 46.127, 46.289` | **46.221** | **-1.02%** |
| MoE e2e | 164.205 | `161.78, 164.10, 161.92, 163.09, 163.88, 162.17` | **162.630** | **-0.96%** |

三个指标均无回退；GEMM1 symbol 仍为不带 `_pmds` 的原 t192 persistent 版本。

复现命令：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 --tokens <1024-or-1536> --topk 8 \
  --model-dim 7168 --inter-dim 2048

ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 --tokens <1024-or-1536> --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

主要性能日志：

```text
t192_pmds/T1024: 20261010T061558Z + 20261010T061836Z
t256_vm192/T1024: 20261010T061723Z + 20261010T061953Z
t192/T1536 after fix: 20261010T062113Z + 20261010T062231Z
```

stage-level random 结果：

```text
/tmp/final_t192_pmds_t1024_random.jsonl
/tmp/final_t192_t1536_random.jsonl
```
