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
- t256 persistent `w2x4`；
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
| `[1024, 1536)` | persistent `t192/w2x4/b4_pmds_pmdrain` | persistent `t192/w2x4/b4_pmdrain` |
| `[1536, 1537)` | persistent `t192/w2x4/b4_pmdrain` | persistent `t192/w2x4/b4_pmdrain` |
| `[1537, 2048)` | persistent `t256/w2x4/b4_pmds_pmdrain` | persistent `t256/w2x4/b4_pmdrain` |
| `[2048, 2049)` | persistent `t256/w2x4/b4` | persistent `t256/w2x4/b4` |
| `[2049, 3073)` | persistent `t192/w2x4/b4_pmds_pmdrain` | persistent `t192/w2x4/b4_pmdrain` |
| `[3073, 3840)` | persistent `t256/w2x4/b4_pmds_pmdrain` | persistent `t256/w2x4/b4_pmdrain` |
| `[3840, 4096)` | optimized `t256/w2x2/b4_pmds` | persistent `t256/w2x4/b4_pmdrain` |
| `[4096, 4097)` | optimized `t256/w2x2/b4` | persistent `t256/w2x4/b4` |
| `[4097, 4104)` | optimized `t256/w2x2/b4_pmds` | persistent `t256/w2x2/b4_pmdrain` |
| `[4104, 4608)` | persistent `t192/w2x4/b4_pmds_pmdrain` | persistent `t192/w2x4/b4_pmdrain` |
| `[4608, 4609)` | persistent `t192/w2x4/b4_pmdrain` | persistent `t192/w2x4/b4_pmdrain` |
| `[4609, +∞)`, non-aligned | optimized `t256/w2x2/b4_pmds` | persistent `t256/w2x2/b4_pmdrain` |
| `>=4609`, aligned override* | optimized `t256/w2x2/b4` | persistent `t256/w2x2/b4` |

路由由 profile table 与左闭右开 interval table 表达，没有为每个 boundary 写独立
`if`。partial-M 区间使用 direct ScaleA store 与 safe-restart specialization；
1536、2048、4096 三个精确 tuned token 使用单值区间。

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

### 三个精确 token 点临时回退到 t64 的性能对比

在 commit `398dbc0bc` 上，临时将以下三个单值 interval 的 GEMM1/GEMM2 profile
都改为 `t64/w1x4/b3`：

```text
[1024, 1025)
[1536, 1537)
[2048, 2049)
```

每个 case 使用 `ROUNDS=3`、const0，并在运行前确认整机 GPU 无 KFD 进程、利用率为0、
VRAM 为0。表中的百分比表示 t64 相对当前默认配置的耗时增加：

| tokens | 指标 | 当前配置 samples / median (us) | t64 samples / median (us) | t64 耗时增加 |
|---:|---|---|---|---:|
| 1024 | GEMM1 | `69.402, 69.826, 69.275` / **69.402** | `94.747, 93.984, 95.002` / **94.747** | **+36.52%** |
| 1024 | GEMM2 | `43.799, 43.464, 43.723` / **43.723** | `59.981, 59.570, 59.783` / **59.783** | **+36.73%** |
| 1024 | MoE e2e | `155.55, 157.74, 155.71` / **155.71** | `188.63, 187.36, 188.69` / **188.63** | **+21.14%** |
| 1536 | GEMM1 | `67.622, 66.928, 67.066` / **67.066** | `124.248, 125.393, 125.815` / **125.393** | **+86.97%** |
| 1536 | GEMM2 | `46.104, 46.222, 46.148` / **46.148** | `74.076, 74.156, 72.999` / **74.076** | **+60.52%** |
| 1536 | MoE e2e | `163.59, 164.58, 163.80` / **163.80** | `238.11, 240.12, 238.49` / **238.49** | **+45.60%** |
| 2048 | GEMM1 | `77.102, 76.271, 77.547` / **77.102** | `163.979, 163.905, 164.232` / **163.979** | **+112.68%** |
| 2048 | GEMM2 | `51.333, 50.974, 51.544` / **51.333** | `98.450, 98.542, 99.044` / **98.542** | **+91.97%** |
| 2048 | MoE e2e | `183.16, 179.59, 183.95` / **183.16** | `306.07, 305.31, 305.43` / **305.43** | **+66.76%** |

当前默认配置为：

| tokens | GEMM1 | GEMM2 |
|---:|---|---|
| 1024 | persistent `t192/w2x4/b4_pmds` | persistent `t192/w2x4/b4` |
| 1536 | persistent `t192/w2x4/b4` | persistent `t192/w2x4/b4` |
| 2048 | persistent `t256/w2x4/b4` | persistent `t256/w2x4/b4` |

t64 实际命中的基础 geometry 为：

```text
GEMM1: a8w4_tdm_fp4_t64x256x256_w1x4_b3_K7168_e64_act1_q1r4
GEMM2: a8w4_tdm_fp4_t64x256x256_w1x4_b3_K2048_e64
```

T1536/T2048 的 symbol 还带有现有通用调度后缀
`bth6/mg10/planarlds/interleavelds`，但 tile、warp grid 和 buffer 数均为
`t64/w1x4/b3`。三个规模的 const0 hash 均通过。

复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 --tokens <1024-or-1536-or-2048> --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

远端原始日志：

```text
/tmp/current_t1024_r3.log
/tmp/current_t1536_r3.log
/tmp/current_t2048_r3.log
/tmp/t64_t1024_r3.log
/tmp/t64_t1536_r3.log
/tmp/t64_t2048_r3.log
```

t64 在三个规模的 GEMM1、GEMM2 和 MoE e2e 上均明显回退，因此不修改默认路由。
测试结束后已恢复 `grouped_moe_gfx1250.py`，其 SHA256 与测试前备份一致，远端
tracked 工作区保持干净。

### `[1025,1536)` 路由到 persistent t192 的精度修复与性能

将 `[1025,1536)` 从 `t64/w1x4/b3` 改为 persistent `t192/w2x4/b4` 后，初始
版本存在非稳定的 GEMM1 quant payload 或 GEMM2 valid-row mismatch。同一 token
重复运行时可能通过，也可能出现 stage-level byte mismatch。失败样本集中在同一
cluster peer 的下一 N-task，说明只 drain output TDM 仍不足以保证跨 task 预取与
cluster peer 保持锁步。

最终增加独立 compile-time profile `t192_partial_interval`：

- 沿用 `partial_m_direct_scale=1`，以 canonical row32 global store 处理 partial
  expert 的 ScaleA；
- 增加 `partial_m_drain_output=1`；
- GEMM1 quant payload 和 GEMM2 output 在 partial expert tile 的下一 persistent
  N-task 启动前执行 `s_wait_tensorcnt 0`、workgroup barrier 和 cluster barrier；
- partial interval 取消跨 task 的 I0/I1 提前预取；persistent N-task loop 保留，
  但每个新 task 在上一 task 完全 drain 后重新建立4-stage input pipeline；
- T1024、T1536、T2048 三个精确边界使用 `partial_m_drain_output=0`，不编译该路径。

#### Random stage-level 正确性

`tokens=1025/1505/1506/1535` 各连续运行5轮，共20个 case，结果全部为：

```text
payload_semantic_mismatch = 0
ScaleA mismatch           = 0
GEMM2 valid-row mismatch  = 0
stage_exact               = True
```

```text
/tmp/t192_safe_restart_precision.log
```

#### t192 persistent 与 t64 的 random 精度对比

两种配置使用相同 seed。t192 每个 token 连续运行5轮；t64 也运行5轮 stage
diagnostic，并额外使用标准 `--scenario verify` 验证正式精度 gate：

| tokens | profile | logits_diff | rel_l2 | GEMM2 valid-row mismatch | standard pass | stage exact |
|---:|---|---:|---:|---:|:---:|:---:|
| 1025 | t192 persistent `_pmds_pmdrain` | `3.38636339e-06` | `0.0026024433` | 0 | True | 5/5 |
| 1025 | t64 `w1x4/b3` | `3.38636339e-06` | `0.0026024433` | 0 | True | N/A* |
| 1505 | t192 persistent `_pmds_pmdrain` | `3.38440942e-06` | `0.0026016932` | 0 | True | 5/5 |
| 1505 | t64 `w1x4/b3` | `3.38440942e-06` | `0.0026016932` | 0 | True | N/A* |
| 1506 | t192 persistent `_pmds_pmdrain` | `3.38447138e-06` | `0.0026017167` | 0 | True | 5/5 |
| 1506 | t64 `w1x4/b3` | `3.38447138e-06` | `0.0026017167` | 0 | True | N/A* |
| 1535 | t192 persistent `_pmds_pmdrain` | `3.38482976e-06` | `0.0026018545` | 0 | True | 5/5 |
| 1535 | t64 `w1x4/b3` | `3.38482976e-06` | `0.0026018545` | 0 | True | N/A* |

`*` t64 使用 generic 非 A-preshuffle quant payload layout；stage 工具的 canonical
A-preshuffle payload decoder 不适用于该中间布局，因此不报告 t64 payload
`stage_exact`。其 GEMM2 valid rows、最终 MoE 输出和标准精度 gate 均通过。

精度日志：

```text
/tmp/t192_safe_restart_precision.log
/tmp/t64_random_compare.log
/tmp/t64_standard_random_verify.log
```

#### t192 与 t64 性能对比

每个 case 使用 `ROUNDS=3`、const0，且运行前确认整机 GPU 空闲。百分比表示 t192
相对 t64 的耗时降低：

| tokens | 指标 | t192 samples / median (us) | t64 samples / median (us) | t192 耗时降低 |
|---:|---|---|---|---:|
| 1025 | GEMM1 | `78.661, 76.281, 76.624` / **76.624** | `97.788, 104.088, 104.143` / **104.088** | **26.39%** |
| 1025 | GEMM2 | `56.324, 55.162, 60.617` / **56.324** | `58.412, 62.616, 63.670` / **62.616** | **10.05%** |
| 1025 | MoE e2e | `198.09, 193.87, 197.67` / **197.67** | `193.77, 203.33, 206.24` / **203.33** | **2.78%** |
| 1505 | GEMM1 | `80.067, 76.284, 81.654` / **80.067** | `127.725, 128.832, 126.797` / **127.725** | **37.31%** |
| 1505 | GEMM2 | `58.594, 59.705, 56.463` / **58.594** | `73.925, 73.657, 75.110` / **73.925** | **20.74%** |
| 1505 | MoE e2e | `216.44, 206.73, 206.42` / **206.73** | `244.71, 250.18, 245.63` / **245.63** | **15.84%** |
| 1506 | GEMM1 | `79.528, 82.784, 137.237` / **82.784** | `127.640, 128.283, 128.677` / **128.283** | **35.47%** |
| 1506 | GEMM2 | `55.712, 57.991, 85.273` / **57.991** | `73.891, 74.429, 74.321` / **74.321** | **21.97%** |
| 1506 | MoE e2e | `203.19, 211.99, 408.78` / **211.99** | `249.61, 249.04, 244.94` / **249.04** | **14.88%** |
| 1535 | GEMM1 | `78.042, 74.603, 76.357` / **76.357** | `127.985, 127.918, 129.111` / **127.985** | **40.34%** |
| 1535 | GEMM2 | `56.108, 55.113, 54.079` / **55.113** | `74.844, 74.757, 75.594` / **74.844** | **26.36%** |
| 1535 | MoE e2e | `202.40, 199.07, 197.69` / **199.07** | `250.74, 244.89, 249.06` / **249.06** | **20.07%** |

性能原始日志：

```text
/tmp/t192safe_t1025_r3.log
/tmp/t192safe_t1505_r3.log
/tmp/t192safe_t1506_r3.log
/tmp/t192safe_t1535_r3.log
/tmp/t64new_t1025_r3.log
/tmp/t64new_t1505_r3.log
/tmp/t64new_t1506_r3.log
/tmp/t64new_t1535_r3.log
```

#### 精确边界无回退证明

对修复前 commit `398dbc0bc` 与修复版分别设置 `FLYDSL_DUMP_IR=1`，绕过 disk cache
重新编译 T1024/T1536/T2048。以下边界 kernel 的 `22_final_isa.s` SHA256 与文件
大小在修复前后逐一完全相同：

| boundary | kernel | SHA256 |
|---:|---|---|
| 1024 | GEMM1 t192 `_pmds` | `29e99e36fcdbd619a815bd442caf7228c91273e5653067a11ab6a76e9df4e40e` |
| 1024 | GEMM2 t192 persistent | `f89a411629fc70913ca018aa2fb940c2ab5b0a2da1c0667666313f8cac0b61406` |
| 1536 | GEMM1 t192 persistent | `d8337e83e38b752a43423e49db4cf7528f3ce06e193ddeeab415603e92edd6da` |
| 1536 | GEMM2 t192 persistent | `f89a411629fc70913ca018aa2fb940c2ab5b0a2da1c0667666313f8cac0b61406` |
| 2048 | GEMM1 t256 persistent | `52d9c641aabcf0a5b2d400006681a0b938232e9404bbae7fac9edc37f45859d2` |
| 2048 | GEMM2 t256 persistent | `23c916618ec58dc56e0ef040b1cea10e5548e9efd7301361be67f39f079cc102d` |

ISA dump 位于：

```text
/tmp/boundary_isa_old
/tmp/boundary_isa_safe_restart
```

因此 `[1025,1536)` 的修复不会改变三个精确边界的 kernel 二进制或其性能路径。

> 本节记录的是上一轮 safe-restart 扩展前的边界结论。后续重复 random 验证在
> T1024 复现了非确定性 payload race，因此最终路由已由文末配置表改为
> `_pmds_pmdrain`；这里的旧 ISA hash 仅作为历史对照。

### t256 partial-M 区间精度修复与 t64 对比

本轮继续修复下面三个左闭右开区间，同时保持 T2048/T4096 两个精确边界的既有
kernel 不变：

| token 区间 | GEMM1 | GEMM2 |
|---|---|---|
| `[1537, 2048)` | persistent `t256/w2x4/b4_pmds_pmdrain` | persistent `t256/w2x4/b4_pmdrain` |
| `[2049, 4096)` | persistent `t256/w2x4/b4_pmds_pmdrain` | persistent `t256/w2x4/b4_pmdrain` |
| `[4097, 4099)` | optimized `t256/w2x2/b4_pmds` | persistent `t256/w2x4/b4_pmdrain` |

#### 错误原因与修复

`t256/w2x4` partial expert tile 沿用了 t192 的 persistent N-task overlap。上一 task
的 output TDM 尚未完全 drain 时，下一 task 会提前建立 input pipeline；partial tile 下
不同 cluster peer 的有效行数边界使这种 overlap 失去原先的锁步假设。最终 profile 显式
启用以下两项：

- `partial_m_direct_scale=1`：ScaleA 按 canonical row32 地址直接写 global；
- `partial_m_drain_output=1`：下一 persistent task 启动前执行 `s_wait_tensorcnt 0`、
  workgroup barrier 和 cluster barrier，并重新建立4-stage input pipeline。

optimized `t256/w2x2` 的错误与 persistent task overlap 无关。`tokens=4097` 有8个
expert 各自包含1个有效 tail row；修复前每行只有 ScaleA 列
`0–3/16–19/32–35/48–51` 正确，其余48 byte 为0，总计：

```text
ScaleA mismatch          = 384 bytes
ScaleA bad rows          = 8
GEMM2 valid-row mismatch ≈ 100k bytes
```

原实现将 ScaleA 先放入 LDS，再按 128-row tile 使用 TDM store。full tile 中各 row32
group 都有数据，这个排列不会暴露 padding；partial tile 中未写入的 padding lane 会使
有效 tail row 的 ScaleA 无法形成 GEMM2 需要的 canonical row32 layout。修复版为
optimized kernel 增加 compile-time `partial_m_direct_scale`，只在 partial profile 中将
每个 e8m0 byte 按 `(row_tile32, scale_dword, row_in_tile32, byte_in_dword)` 直接写入
global。T4096 的 flag 为0，仍走原有 LDS + TDM ScaleA store。

#### Random stage-level 正确性

t256 配置对每个 token 连续运行5轮。20个 case 全部满足：

```text
payload_semantic_mismatch = 0
ScaleA mismatch           = 0
GEMM2 valid-row mismatch  = 0
GEMM2 reference/hash      = match
stage_exact               = True
```

对应的 overall precision 与强制 `t64/w1x4/b3` 完全相同：

| tokens | profile | logits_diff | rel_l2 | pass | stage exact |
|---:|---|---:|---:|:---:|:---:|
| 1537 | t256 partial persistent | `3.38482544e-06` | `0.0026018529` | 5/5 | 5/5 |
| 1537 | t64 `w1x4/b3` | `3.38482544e-06` | `0.0026018529` | 5/5 | N/A* |
| 2049 | t256 partial persistent | `3.38344265e-06` | `0.0026013220` | 5/5 | 5/5 |
| 2049 | t64 `w1x4/b3` | `3.38344265e-06` | `0.0026013220` | 5/5 | N/A* |
| 4095 | t256 partial persistent | `3.38294142e-06` | `0.0026011292` | 5/5 | 5/5 |
| 4095 | t64 `w1x4/b3` | `3.38294142e-06` | `0.0026011292` | 5/5 | N/A* |
| 4097 | t256 optimized/persistent | `3.38291439e-06` | `0.0026011188` | 5/5 | 5/5 |
| 4097 | t64 `w1x4/b3` | `3.38291439e-06` | `0.0026011188` | 5/5 | N/A* |

`*` t64 使用 generic 非 A-preshuffle intermediate layout，因此不使用 canonical
A-preshuffle stage decoder；其 GEMM2 reference/output hash 和最终 MoE precision gate
均为 5/5 通过。最终 production interval table 另做了一轮默认路由检查，四个 token
也全部 `stage_exact=True`。额外抽查区间内部/右端附近的 T1538、T2047、T4098，
三者同样满足 payload/ScaleA/GEMM2 mismatch 全为0且 `stage_exact=True`。

精度日志：

```text
/tmp/e64_t256_interval_random_final_20261011/
/tmp/e64_t64_route_random_final_20261011/
/tmp/e64_default_route_random_final_20261011.jsonl
/tmp/e64_default_route_interval_extra_random_20261011.jsonl
```

#### t256 与 t64 const0 性能

每个 token 使用严格相邻的 t256/t64 A/B 顺序运行3轮；每轮 `--iters 20`。运行前后
均确认整机无 KFD 进程、GPU use=0、VRAM=0。百分比按各自三轮中值计算：

| tokens | 指标 | t256 samples / median (us) | t64 samples / median (us) | t256 相对 t64 |
|---:|---|---|---|---:|
| 1537 | GEMM1 | `100.182, 101.141, 98.358` / **100.182** | `143.831, 145.088, 144.196` / **144.196** | **快30.52%** |
| 1537 | GEMM2 | `95.080, 93.599, 90.359` / **93.599** | `76.465, 76.374, 75.529` / **76.374** | **慢22.55%** |
| 1537 | MoE e2e | `263.406, 263.136, 258.171` / **263.136** | `262.085, 264.668, 261.651` / **262.085** | **慢0.40%** |
| 2049 | GEMM1 | `171.010, 170.407, 175.235` / **171.010** | `178.290, 176.916, 177.656` / **177.656** | **快3.74%** |
| 2049 | GEMM2 | `134.223, 134.611, 134.137` / **134.223** | `101.980, 102.537, 102.193` / **102.193** | **慢31.34%** |
| 2049 | MoE e2e | `383.229, 380.186, 382.841` / **382.841** | `329.688, 329.137, 328.210` / **329.137** | **慢16.32%** |
| 4095 | GEMM1 | `190.858, 195.594, 197.518` / **195.594** | `303.915, 306.328, 309.123` / **306.328** | **快36.15%** |
| 4095 | GEMM2 | `147.394, 145.067, 147.260` / **147.260** | `183.127, 182.291, 182.133` / **182.291** | **快19.22%** |
| 4095 | MoE e2e | `437.403, 440.647, 444.548` / **440.647** | `558.985, 562.523, 560.712` / **560.712** | **快21.41%** |
| 4097 | GEMM1 | `169.099, 170.389, 169.003` / **169.099** | `308.820, 309.421, 309.519` / **309.421** | **快45.35%** |
| 4097 | GEMM2 | `189.654, 189.261, 189.103` / **189.261** | `185.806, 185.835, 186.093` / **185.835** | **慢1.84%** |
| 4097 | MoE e2e | `459.151, 464.073, 459.208` / **459.208** | `568.144, 568.165, 569.369` / **568.165** | **快19.18%** |

实际 geometry：

```text
t256 GEMM1 persistent:
  a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_e64_act1_q1r8_cn4
t256 GEMM1 optimized:
  a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cn4_prefetch
t256 GEMM2 persistent:
  a8w4_tdm_fp4_t256x256x256_w2x4_b4_K2048_e64_cn4
t64 GEMM1:
  a8w4_tdm_fp4_t64x256x256_w1x4_b3_K7168_e64_act1_q1r4
t64 GEMM2:
  a8w4_tdm_fp4_t64x256x256_w1x4_b3_K2048_e64
```

性能日志：

```text
/tmp/e64_t256_vs_t64_adjacent_final_20261011/
```

#### `[2049,4096)` GEMM1 w2x2 与 w2x4 专项 A/B

该实验只通过 diagnostic profile 临时切换 GEMM1；production interval 仍保持用户指定的
persistent `t256/w2x4/b4`。GEMM2 始终使用同一个 persistent
`t256/w2x4/b4` kernel。每个 token 使用相邻 w2x4/w2x2 顺序运行3对：

| tokens | 配置 | GEMM1 samples / median (us) | GEMM2 median (us) | MoE e2e median (us) |
|---:|---|---|---:|---:|
| 2049 | persistent w2x4 | `170.671, 173.233, 171.327` / **171.327** | 135.231 | 381.572 |
| 2049 | optimized w2x2 | `141.542, 136.220, 140.013` / **140.013** | 141.953 | 358.186 |
| 4095 | persistent w2x4 | `199.106, 191.289, 191.291` / **191.291** | 144.645 | 435.613 |
| 4095 | optimized w2x2 | `166.760, 166.254, 165.554` / **166.254** | 152.583 | 418.381 |

结论：optimized w2x2 的 GEMM1 中值在 T2049/T4095 分别降低 **18.28%** 和
**13.09%**；对应 MoE e2e 中值分别降低 **6.13%** 和 **3.96%**。同一个 GEMM2
kernel 的计时会随前置 GEMM1/cache 状态变化，因此该专项实验以 GEMM1 与完整 e2e 的
相邻 A/B 为主要判断依据。

专项日志：

```text
/tmp/e64_g1_w2x2_vs_w2x4_adjacent_20261011/
```

#### T2048/T4096 精确边界无回退证明

在 detached worktree 中使用修复前 commit `398dbc0bc`，并与最终代码分别设置：

```bash
FLYDSL_DUMP_IR=1 FLYDSL_RUNTIME_ENABLE_CACHE=0
```

绕过 disk cache 后重新编译 T2048/T4096。两个边界实际执行的 GEMM1/GEMM2
`22_final_isa.s` 在修复前后 SHA256 和文件大小完全相同：

| boundary | kernel | SHA256 | bytes |
|---:|---|---|---:|
| 2048 | GEMM1 persistent t256 w2x4 | `52d9c641aabcf0a5b2d400006681a0b938232e9404bbae7fac9edc37f45859d2` | 205645 |
| 2048 | GEMM2 persistent t256 w2x4 | `23c91618ec58dc56e0ef040b1cea10e5548e9efd7301361be67f39f079cc102d` | 152872 |
| 4096 | GEMM1 optimized t256 w2x2 fused quant | `c33cda387def565e37cecad9f0a10b8f9c705ea0ada872b49753f92827ec4cca` | 303097 |
| 4096 | GEMM2 persistent t256 w2x4 | `23c91618ec58dc56e0ef040b1cea10e5548e9efd7301361be67f39f079cc102d` | 152872 |

dump 与 detached worktree：

```text
/tmp/boundary_isa_base_398dbc0bc_20261011/
/tmp/boundary_isa_current_20261011/
/data/yanguahe/code/wk_sp1/aiter_boundary_base_20261011
```

因此新逻辑只在 partial-M profile 的 compile-time flag 为1时进入，不改变 T2048/T4096
两个精确边界的 GEMM1/GEMM2 kernel 二进制及性能路径。

默认路由的复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 --tokens <1537-or-2049-or-4095-or-4097> --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

## 按 random 精度与 const0 性能重选最终路由

### 命名约定

`persistent` 表示使用 persistent N-task loop 的 launcher。此前额外添加的 `full`
修饰并不表示另一种 kernel 模式，只是为了与 `_vm192` 区分，现已删除。需要区分
specialization 时只使用真实 suffix：

- `_vm192`：t256 launcher 使用192-row compute geometry；
- `_pmds`：partial-M ScaleA 使用 canonical row32 direct store；
- `_pmdrain`：persistent task 间完整 drain，并重新建立 input pipeline；对齐 M tile
  仍可能需要该同步。

### 候选筛选与最终分界

第一轮对15个 token 采样点扫描 `t64`、safe t192、persistent t256 w2x4 和
optimized-GEMM1 t256 w2x2，四组候选的 overall random precision 全部通过。随后根据
const0 扫描与交错复测得到以下结论：

- T2049–T3072：safe t192 的 GEMM1、GEMM2 和 MoE e2e 均优于 t64/t256；
- T3073–T3839：persistent t256 w2x4 更稳定；
- T3840–T4095：optimized t256 w2x2 GEMM1 获得稳定收益，GEMM2 继续使用同一个
  persistent t256 w2x4；
- T4097/T4098：safe t192 的 GEMM2 与 e2e 明显优于刚进入第三个 M tile 的 t256；
- t256-GEMM1/t192-GEMM2 mixed tile 需要额外 expert-row padding，实际使两个 kernel
  都变慢，因此未进入 production 路由；
- T1024 原 `t192_pmds` 和 T3072 无 safe-restart 的 exact t192 均复现了非确定性
  payload race，两点都改用 `_pmds_pmdrain`；T1536 的无后缀 exact t192 在20轮中
  也有3轮失败，改用 drain-only `t192_pmdrain`。对应 safe profile 均连续5轮
  `stage_exact=True`。

w2x2/w2x4 的5轮交错结果中，T3840 与 T4095 的 GEMM1 中值分别为：

| tokens | persistent w2x4 GEMM1 | optimized w2x2 GEMM1 | winner |
|---:|---:|---:|---|
| 3840 | 173.857 us | 153.484 us | w2x2，快11.72% |
| 4095 | 172.533 us | 155.227 us | w2x2，快10.03% |

T3600/T3616 的小幅差异无法跨批次稳定复现，因此没有在该处切换；最终采用对齐且有
稳定收益的 T3840 作为 w2x2 左边界。

### 最终 random stage-level 正确性

最终 production interval table 在以下 boundary/aligned token 上全部通过：

```text
tokens = 1024, 1025, 1536, 1537, 2048, 2049, 3072,
         3073, 3584, 3600, 3840, 4096, 4097, 4098

payload_semantic_mismatch = 0
ScaleA mismatch           = 0
GEMM2 valid-row mismatch  = 0
stage_exact               = True
```

其中 T1024 和 T3072 的 safe-restart profile 额外连续运行5轮，全部得到相同结果。

### Random 输入性能与误差

每个 token 运行3轮，每轮 `--iters 20`。表中为原始 samples 与中值，时间单位为 us。

| tokens | GEMM1 samples / median | GEMM2 samples / median | MoE e2e samples / median | logits_diff | rel_l2 | pass |
|---:|---|---|---|---:|---:|:---:|
| 1024 | `91.565, 93.107, 93.484` / **93.107** | `59.724, 59.010, 61.149` / **59.724** | `211.222, 202.523, 205.304` / **205.304** | `3.38629605e-06` | `0.00260242` | True |
| 1025 | `88.937, 94.035, 88.995` / **88.995** | `53.551, 55.777, 53.920` / **53.920** | `206.131, 212.831, 205.419` / **206.131** | `3.38636339e-06` | `0.00260244` | True |
| 1536 | `95.122, 96.455, 98.180` / **96.455** | `63.587, 61.158, 67.994` / **63.587** | `217.770, 220.773, 227.743` / **220.773** | `3.38491010e-06` | `0.00260189` | True |
| 1537 | `129.471, 119.024, 108.707` / **119.024** | `108.893, 95.453, 100.276` / **100.276** | `303.697, 280.509, 276.895` / **280.509** | `3.38482544e-06` | `0.00260185` | True |
| 2048 | `115.505, 118.100, 115.526` / **115.526** | `75.177, 74.778, 72.128` / **74.778** | `254.375, 256.344, 248.754` / **254.375** | `3.38332714e-06` | `0.00260128` | True |
| 2049 | `158.767, 159.929, 158.592` / **158.767** | `93.244, 93.872, 94.534` / **93.872** | `322.215, 324.818, 323.430` / **323.430** | `3.38344265e-06` | `0.00260132` | True |
| 3072 | `167.591, 164.458, 165.790` / **165.790** | `96.590, 95.894, 96.970` / **96.590** | `341.253, 338.380, 336.745` / **338.380** | `3.38116699e-06` | `0.00260045` | True |
| 3073 | `214.373, 211.229, 208.250` / **211.229** | `144.698, 142.068, 142.758` / **142.758** | `445.935, 439.785, 436.614` / **439.785** | `3.38118279e-06` | `0.00260045` | True |
| 3584 | `201.990, 200.697, 200.615` / **200.697** | `145.465, 143.789, 144.175` / **144.175** | `433.839, 430.499, 429.798` / **430.499** | `3.38156619e-06` | `0.00260060` | True |
| 3600 | `216.947, 221.910, 215.886` / **216.947** | `158.668, 156.115, 156.041` / **156.115** | `459.788, 462.563, 455.971` / **459.788** | `3.38166411e-06` | `0.00260064` | True |
| 3840 | `222.104, 228.756, 217.919` / **222.104** | `146.049, 150.947, 146.359` / **146.359** | `455.830, 467.309, 451.340` / **455.830** | `3.38195554e-06` | `0.00260075` | True |
| 4096 | `199.970, 220.917, 194.026` / **199.970** | `112.962, 114.617, 112.519` / **112.962** | `404.722, 427.385, 397.634` / **404.722** | `3.38291909e-06` | `0.00260112` | True |
| 4097 | `227.961, 228.316, 227.921` / **227.961** | `134.828, 134.109, 135.716` / **134.828** | `464.706, 464.700, 464.454` / **464.700** | `3.38291439e-06` | `0.00260112` | True |
| 4098 | `228.063, 228.196, 228.084` / **228.084** | `135.286, 134.525, 135.431` / **135.286** | `463.971, 462.484, 463.429` / **463.429** | `3.38293414e-06` | `0.00260113` | True |

### Const0 输入性能

每个 token 运行3轮，每轮 `--iters 20`。表中时间单位为 us。

| tokens | GEMM1 samples / median | GEMM2 samples / median | MoE e2e samples / median |
|---:|---|---|---|
| 1024 | `83.347, 83.916, 79.231` / **83.347** | `53.142, 56.709, 54.607` / **54.607** | `189.807, 191.277, 185.563` / **189.807** |
| 1025 | `77.694, 82.424, 75.514` / **77.694** | `49.286, 50.131, 47.951` / **49.286** | `190.739, 195.449, 186.574` / **190.739** |
| 1536 | `78.407, 78.169, 76.982` / **78.169** | `53.388, 56.209, 58.360` / **56.209** | `193.134, 191.977, 193.773` / **193.134** |
| 1537 | `106.476, 94.030, 97.324` / **97.324** | `94.756, 91.673, 91.523` / **91.673** | `267.209, 251.126, 253.182` / **253.182** |
| 2048 | `93.455, 100.189, 88.473` / **93.455** | `69.520, 67.078, 63.207` / **67.078** | `226.650, 232.530, 214.832` / **226.650** |
| 2049 | `134.466, 135.777, 133.656` / **134.466** | `82.466, 84.528, 83.326` / **83.326** | `286.427, 290.089, 286.870` / **286.870** |
| 3072 | `138.235, 139.730, 136.515` / **138.235** | `83.845, 85.455, 84.882` / **84.882** | `296.291, 299.209, 294.484` / **296.291** |
| 3073 | `184.039, 181.821, 178.917` / **181.821** | `131.065, 127.383, 132.610` / **131.065** | `400.356, 393.876, 395.258` / **395.258** |
| 3584 | `163.397, 165.634, 169.039` / **165.634** | `129.897, 129.716, 133.888` / **129.897** | `379.117, 380.592, 386.574` / **380.592** |
| 3600 | `191.168, 190.543, 187.109` / **190.543** | `146.770, 145.250, 141.824` / **145.250** | `420.787, 418.971, 412.557` / **418.971** |
| 3840 | `198.138, 193.693, 185.797` / **193.693** | `135.790, 134.991, 134.725` / **134.991** | `422.056, 416.762, 407.954` / **416.762** |
| 4096 | `159.630, 168.002, 153.787` / **159.630** | `97.193, 97.699, 97.359` / **97.359** | `346.158, 355.008, 339.157` / **346.158** |
| 4097 | `188.637, 189.386, 188.502` / **188.637** | `118.743, 119.281, 118.382` / **118.743** | `405.810, 406.822, 405.964` / **405.964** |
| 4098 | `186.752, 187.068, 185.464` / **186.752** | `119.217, 119.171, 118.317` / **119.171** | `405.895, 404.480, 403.853` / **404.480** |

性能测试前后均确认整机无 KFD 进程、GPU use=0、VRAM=0。原始日志位于：

```text
/tmp/e64_final_route_perf_20261011/
/tmp/e64_final_route_perf_3600_3840_20261011/
/tmp/e64_final_route_stage_random_v3_20261011.jsonl
/tmp/e64_final_route_stage_3600_3840.jsonl
```

标准脚本复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 --tokens <TOKEN> --topk 8 \
  --model-dim 7168 --inter-dim 2048

ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 --tokens <TOKEN> --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

## T4097 以上 token 路由研究

### 候选与交叉点

高 token 初筛比较了 safe t192、persistent t256 w2x4、GEMM1-w2x2/GEMM2-w2x4
以及 GEMM1/GEMM2 都使用 t256 w2x2。所有候选在
T4099/T6144/T8191/T8192/T8193/T12288/T16383/T16384/T16385 上均通过 overall
random precision；最终胜出的 profile 再做 stage-level 验证。

`[4097,8192)` 不是单一 profile。三轮相邻 A/B 确认了两个由 expert M-tile 数变化
产生的交叉点：

| tokens | t192 e2e median | t256 w2x2/w2x2 e2e median | winner |
|---:|---:|---:|---|
| 4097 | 410.002 us | 375.461 us | t256 |
| 4100 | 405.998 us | 401.962 us | t256 |
| 4104 | 400.478 us | 435.141 us | t192 |
| 4608 | 405.717 us | 445.592 us | t192 |
| 4609 | 487.181 us | 449.214 us | t256 |

T4104 时64个 expert 全部从512行变为513行，t256 全部进入第三个 M tile；T4609
开始让 t192 从3个 M tile 增加到4个。因此最终分为：

```text
[4097, 4104) -> t256 w2x2/w2x2
[4104, 4608) -> partial-safe t192 w2x4/w2x4
[4608, 4609) -> aligned t192 w2x4/w2x4，保留 task drain
[4609, +∞)   -> t256 w2x2/w2x2
```

T4608 虽然每个 expert 恰好有 `576 = 3 * 192` 行，但无后缀 exact t192 的跨
persistent N-task 预取仍存在时序 race：5轮中1轮出现254个 payload mismatch 和
362130 byte GEMM2 valid-row mismatch。该错误与 ScaleA padding 无关，所以 T4608
不需要 `_pmds`；只保留 `_pmdrain` 后连续5轮 `stage_exact=True`。同批 const0 对比中，
drain-only GEMM1 中值为184.694 us，`_pmds_pmdrain` 为187.469 us。

在 `[8193,16384)` 内额外测试了 t192 的 full-tile 点 T9216/T10752/T13824/T15360；
t256 w2x2/w2x2 的 e2e 分别快3.17%、3.77%、10.17%、7.33%，未发现新的 t192
性能窗口。T32768 初筛也仍由 t256 w2x2/w2x2 获胜。

### t256 exact alignment override

当 `tokens * topk / experts` 是256的整数倍时，每个 expert 都由完整 t256 M tile
组成。T6144/T8192/T10240/T12288/T14336/T16384/T18432 各连续5轮 random 均为
`stage_exact=True`，可以安全去掉 `_pmds/_pmdrain`。三轮 const0 中值如下：

| tokens | partial-safe GEMM1 | exact GEMM1 | partial-safe e2e | exact e2e |
|---:|---:|---:|---:|---:|
| 6144 | 221.101 us | 213.382 us | 490.598 us | 478.722 us |
| 8192 | 277.939 us | 265.683 us | 618.623 us | 606.677 us |
| 10240 | 339.874 us | 329.212 us | 748.151 us | 741.020 us |
| 12288 | 387.984 us | 373.295 us | 869.308 us | 854.902 us |
| 14336 | 443.684 us | 430.732 us | 998.308 us | 986.023 us |
| 16384 | 509.446 us | 490.059 us | 1146.133 us | 1127.446 us |
| 18432 | 562.376 us | 544.699 us | 1279.071 us | 1264.359 us |

因此 dispatch 使用通用对齐判断，而不是为每个 token 写单独分支：

```text
routes_per_expert = tokens * topk / experts
aligned = routes_per_expert is integer and routes_per_expert % tile_m == 0
```

E64/topk8/tile_m256 下等价于 `tokens % 2048 == 0`。aligned 点使用 exact
t256 w2x2/w2x2；其余点使用 `_pmds/_pmdrain`。

### 最终高 token random stage-level 正确性

最终默认路由在以下 boundary/aligned token 上全部满足：

```text
tokens = 4097, 4104, 4608, 4609, 6144, 8191, 8192,
         8193, 12288, 16383, 16384, 16385, 18432

payload_semantic_mismatch = 0
ScaleA mismatch           = 0
GEMM2 valid-row mismatch  = 0
stage_exact               = True
```

### 高 token random 输入性能与误差

每个 token 运行3轮，每轮 `--iters 20`。表中时间单位为 us。

| tokens | GEMM1 samples / median | GEMM2 samples / median | MoE e2e samples / median | logits_diff | rel_l2 | pass |
|---:|---|---|---|---:|---:|:---:|
| 4097 | `197.100, 195.648, 214.371` / **197.100** | `122.034, 119.407, 124.295` / **122.034** | `419.785, 422.317, 438.877` / **422.317** | `3.38291439e-06` | `0.00260112` | True |
| 4104 | `237.188, 239.935, 234.565` / **237.188** | `136.663, 136.542, 135.976` / **136.542** | `470.891, 474.086, 467.819` / **470.891** | `3.38296953e-06` | `0.00260114` | True |
| 4608 | `235.406, 233.688, 238.461` / **235.406** | `139.848, 139.640, 139.825` / **139.825** | `481.783, 476.666, 484.296` / **481.783** | `3.38296051e-06` | `0.00260114` | True |
| 4609 | `241.814, 242.077, 248.089` / **242.077** | `155.107, 154.900, 154.903` / **154.903** | `503.960, 505.051, 510.756` / **505.051** | `3.38295717e-06` | `0.00260114` | True |
| 6144 | `263.376, 260.239, 280.139` / **263.376** | `164.536, 164.589, 165.712` / **164.589** | `545.525, 542.729, 563.240` / **545.525** | `3.38486922e-06` | `0.00260187` | True |
| 8191 | `339.969, 326.701, 345.262` / **339.969** | `225.043, 215.100, 227.473` / **225.043** | `716.140, 694.818, 727.436` / **716.140** | `3.38390860e-06` | `0.00260150` | True |
| 8192 | `335.451, 320.110, 328.688` / **328.688** | `229.724, 215.225, 222.760` / **222.760** | `710.700, 679.678, 696.936` / **696.936** | `3.38390983e-06` | `0.00260150` | True |
| 8193 | `352.284, 338.972, 348.726` / **348.726** | `232.792, 224.822, 229.963` / **229.963** | `737.426, 715.093, 732.123` / **732.123** | `3.38392919e-06` | `0.00260151` | True |
| 12288 | `498.318, 496.065, 492.807` / **496.065** | `341.417, 344.941, 341.800` / **341.800** | `1043.295, 1046.931, 1039.679` / **1043.295** | `3.38299380e-06` | `0.00260115` | True |
| 16383 | `687.129, 673.798, 678.415` / **678.415** | `459.840, 455.304, 453.263` / **455.304** | `1419.763, 1397.719, 1403.229` / **1403.229** | `3.38369592e-06` | `0.00260142` | True |
| 16384 | `653.945, 652.246, 650.944` / **652.246** | `455.064, 451.176, 451.452` / **451.452** | `1375.223, 1367.087, 1369.484` / **1369.484** | `3.38369682e-06` | `0.00260142` | True |
| 16385 | `675.029, 669.783, 673.865` / **673.865** | `455.842, 456.510, 455.947` / **455.947** | `1409.449, 1407.844, 1408.930` / **1408.930** | `3.38369733e-06` | `0.00260142` | True |
| 18432 | `709.961, 716.567, 717.058` / **716.567** | `502.303, 500.988, 504.643` / **502.303** | `1518.846, 1522.137, 1530.118` / **1522.137** | `3.38318060e-06` | `0.00260122` | True |

### 高 token const0 输入性能

| tokens | GEMM1 samples / median | GEMM2 samples / median | MoE e2e samples / median |
|---:|---|---|---|
| 4097 | `171.532, 171.427, 164.323` / **171.427** | `110.552, 109.565, 109.932` / **109.932** | `381.749, 378.779, 380.292` / **380.292** |
| 4104 | `198.564, 263.853, 198.530` / **198.564** | `120.635, 225.398, 120.377` / **120.635** | `414.759, 693.945, 414.879` / **414.879** |
| 4608 | `195.523, 188.806, 197.406` / **195.523** | `120.807, 120.852, 121.054` / **120.852** | `414.272, 412.397, 416.969` / **414.272** |
| 4609 | `201.520, 201.758, 198.047` / **201.520** | `140.605, 139.393, 139.475` / **139.475** | `450.659, 449.219, 445.536` / **449.219** |
| 6144 | `228.452, 215.811, 215.794` / **215.811** | `147.243, 144.854, 146.418` / **146.418** | `491.176, 476.112, 479.243` / **479.243** |
| 8191 | `274.355, 273.571, 272.962` / **273.571** | `198.515, 195.619, 199.249` / **198.515** | `623.800, 618.964, 623.337` / **623.337** |
| 8192 | `265.173, 264.678, 265.266` / **265.173** | `198.401, 193.245, 197.678` / **197.678** | `604.588, 597.359, 604.962` / **604.588** |
| 8193 | `283.436, 282.309, 280.782` / **282.309** | `202.064, 199.738, 203.040` / **202.064** | `636.775, 632.813, 635.853` / **635.853** |
| 12288 | `375.087, 379.692, 377.544` / **377.544** | `285.007, 293.108, 284.872` / **285.007** | `855.441, 865.978, 859.069` / **859.069** |
| 16383 | `516.261, 520.973, 524.726` / **520.973** | `386.544, 384.786, 384.621` / **384.786** | `1160.573, 1163.415, 1170.484` / **1163.415** |
| 16384 | `505.103, 508.462, 508.337` / **508.337** | `383.818, 384.522, 383.734` / **383.818** | `1140.703, 1144.985, 1145.869` / **1144.985** |
| 16385 | `516.672, 520.330, 520.239` / **520.239** | `387.139, 387.605, 388.064` / **387.605** | `1173.677, 1177.136, 1179.268` / **1177.136** |
| 18432 | `545.772, 547.406, 547.282` / **547.282** | `425.362, 425.297, 425.938` / **425.362** | `1261.882, 1264.897, 1266.702` / **1264.897** |

高 token 测试日志：

```text
/tmp/e64_high_final_stage_random_20261011.jsonl
/tmp/e64_high_final_perf_20261011/
/tmp/e64_high_final_perf_rerun_20261011/
/tmp/e64_high_boundary_confirm_20261011/
/tmp/e64_high_exact_perf_20261011/
```

### t192 对齐点 `_pmdrain` A/B

T1536 与 T4608 都是 t192 完全对齐点，但无后缀 exact t192 仍会执行跨 persistent
N-task 的 input/output overlap。stage-level 重复验证结果为：

| tokens | 无后缀 exact stage audit | drain-only stage audit |
|---:|---|---|
| 1536 | 20轮中17轮通过、3轮失败 | 5/5 `stage_exact=True` |
| 4608 | 5轮中4轮通过、1轮失败 | 5/5 `stage_exact=True` |

失败 case 的 ScaleA mismatch 均为0，错误集中在 quant payload 与 GEMM2 valid rows，
说明 `_pmdrain` 解决的是跨 N-task output/input overlap，而不是 M padding。

在 GPU 空闲状态下对无后缀 exact 与 drain-only 各做5轮相邻 A/B。random 输入结果：

| tokens | profile | GEMM1 samples / median | GEMM2 samples / median | MoE e2e samples / median | logits_diff max | rel_l2 max |
|---:|---|---|---|---|---:|---:|
| 1536 | exact，无 drain | `94.043, 90.088, 91.511, 90.786, 89.993` / **90.786** | `60.118, 60.528, 62.046, 59.579, 59.522` / **60.118** | `213.626, 208.209, 215.460, 209.067, 207.410` / **209.067** | `3.38491010e-06` | `0.00260189` |
| 1536 | drain-only | `94.661, 102.222, 94.659, 99.740, 90.480` / **94.661** | `63.397, 71.559, 68.128, 63.726, 62.822` / **63.726** | `217.769, 232.812, 222.070, 226.040, 213.782` / **222.070** | `3.38491010e-06` | `0.00260189` |
| 4608 | exact，无 drain | `228.699, 232.229, 231.748, 230.040, 228.822` / **230.040** | `132.644, 133.489, 135.516, 133.631, 132.855` / **133.489** | `459.116, 462.875, 466.273, 460.524, 460.334` / **460.524** | `3.38296051e-06` | `0.00260114` |
| 4608 | drain-only | `234.506, 234.593, 235.742, 238.316, 239.054` / **235.742** | `139.828, 141.287, 140.302, 140.519, 139.933` / **140.302** | `471.834, 474.938, 473.626, 476.508, 476.439` / **474.938** | `3.38296051e-06` | `0.00260114` |

const0 输入结果：

| tokens | profile | GEMM1 samples / median | GEMM2 samples / median | MoE e2e samples / median |
|---:|---|---|---|---|
| 1536 | exact，无 drain | `75.403, 76.842, 70.686, 74.146, 74.802` / **74.802** | `54.951, 52.045, 48.902, 54.036, 53.302` / **53.302** | `192.111, 190.914, 176.534, 184.112, 185.076` / **185.076** |
| 1536 | drain-only | `74.502, 78.839, 79.084, 75.504, 78.027` / **78.027** | `52.149, 55.676, 56.724, 56.389, 60.034` / **56.389** | `187.669, 193.145, 196.975, 190.887, 200.476` / **193.145** |
| 4608 | exact，无 drain | `183.834, 185.502, 183.025, 183.729, 185.386` / **183.834** | `117.288, 117.242, 116.679, 117.259, 117.976` / **117.259** | `395.732, 398.732, 394.516, 394.322, 396.774` / **395.732** |
| 4608 | drain-only | `186.755, 187.172, 189.059, 186.497, 190.335` / **187.172** | `121.130, 120.443, 120.263, 120.894, 119.681` / **120.443** | `404.153, 407.999, 404.018, 403.415, 404.132` / **404.132** |

以无 drain 中值为基准，drain-only 的性能成本为：

| tokens | data | GEMM1 | GEMM2 | MoE e2e |
|---:|---|---:|---:|---:|
| 1536 | random | +4.27% | +6.00% | +6.22% |
| 1536 | const0 | +4.31% | +5.79% | +4.36% |
| 4608 | random | +2.48% | +5.10% | +3.13% |
| 4608 | const0 | +1.82% | +2.72% | +2.12% |

无 drain 的5轮 timed random 恰好都通过 overall precision gate，但更长的 stage audit
已经复现错误，因此不能用该性能路径作为 production 配置。2026-10-11 的配置为
drain-only `t192_pmdrain`；2026-10-12 的修复与更新配置见下节。

原始日志：

```text
/tmp/e64_t192_pmdrain_ab_20261011/
/tmp/e64_t1536_exact_race_audit_20261011/
/tmp/e64_t4608_t192_exact_random_20261011/
```

### 2026-10-12 t192 overlap race 修复与性能验收

这是实现中的 LDS 生命周期错误。旧无 drain GEMM1 将 quant output 写入仍有其他
wave 读取的 final K tile 区域；GEMM2 的第一片 output 也可能提前覆盖 final input。
next N-task 的 input prefetch 与上一任务 output TDM 可以重叠，前提是前后任务和
各片 output 的 LDS 地址不冲突，并在复用区域前完成跨 wave release。

GEMM1 exact `t192/w2x4/b4` 将 quant payload/ScaleA 放到 stage2 前部：

- `PITCH=65536`，output 相对 LDS allocation 的范围为 `[131072,144128)`。
- penultimate carry fence 已释放 stage2 前部。final K tile 使用 stage3 前部与
  stage0 tail；next I0 使用 stage0 前部/stage1 tail，next I1 使用 stage1 前部和
  stage2 tail `[179648,192896)`，均不覆盖 output。
- 保留 next I0/I1 prefetch、下一任务 input-ready fence，以及回填 stage2/3 前的
  output-done fence；删除 exact quant output 中间冗余的 `tensor_wait(2)`。
- 保留原五 segment allocation，避免改变该 specialization 的资源占用。
- quant 的 amax 改为已取整 BF16 的 packed `V_PK_MAX_NUM_BF16` tree，随后继续使用
  原 FP32 clamp、peer reduction、E8M0 rounding 与 FP4 conversion。

GEMM2 将两个 M wave 的第一片 output 一起放进已释放的 stage2：
`PITCH=60416`，第一片占 `52224` bytes。第一片 output barrier 等待所有 wave
完成 final input DS reads，之后才写第二片进入 stage3。动态 slice LDS base 固定在
一个 VGPR 中，避免 DS immediate 的16位范围截断。保留两 tile A cache 和所有
K tile 的 ScaleA cache；保留原 input ring pitch，并删除 cached 路径的冗余 wait。

硬件事实与实现推导分开：Shader Programming Guide §4.10.1 p197 规定同一 wave
的 Tensor 按序完成，跨 wave 及相对其他 memory instructions 无序；§4.10.8 p206
的 descriptor 容量只覆盖 issue→XACK，不能证明 completion。§4.3.6 p74–75 定义
barrierComplete 的逐 wave consume；§4.7 p160–161 记载64 KiB LDS segment 与
indexed DS immediate 范围。上述具体 LDS 区域安全性由地址计算、生成 ISA 和
重复数值验证共同检查。

最终候选 `persistent_stage2_final.py` 已原样接入生产
`mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py`：

```text
before SHA256: 72db89fa28431f9076697774a7ba8ea408de8e31e4891b74c219eb46dd0b15b1
final  SHA256: 114df11c747ab9d6a33888058a2c7d19a77fb90e893e70870aff81e55848eea0
host: a07-3
repo: /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
container: hyg_fyd1
branch: moe_a4w4_pr_refactor_wt
HEAD: 398dbc0bcdee3d0681eaa1f371feeef06db72f5f
workload: E64/topk8/M7168/I2048/A4W4/SiLU/no-bias
```

性能在整机 GPU/KFD 空闲门禁后测试，每种 data/token 做24轮相邻 old/new A/B，
每轮201次。轮次交替反转先后顺序，固定所有 intermediate/output 地址。
不丢首轮、不做 IQR filtering；每个 metric、每个 variant 保留4824个 raw samples。
`MoE e2e` 是一次完整 eager MoE 的所有 GPU kernel 时间之和；CUDA event wall time
不属于本表口径。单个 GEMM 必须独立不回退。

完整 long-run report、各轮 median、raw trace 与 paired distribution 见
[stage2_final_report.json](../.codex_tmp/t192_overlap_20261012/resume_evidence/stage2_final_report.json)。
与旧无 drain 相比，12项 median 和 paired median 全部为负值：

| tokens | data | GEMM1 old/new median (us) | GEMM2 old/new median (us) | MoE e2e old/new median (us) | GEMM1 delta | GEMM2 delta | e2e delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1536 | random | 88.452 / 87.330 | 59.0675 / 58.3465 | 208.363 / 205.0765 | −1.26848% | −1.22064% | −1.57730% |
| 1536 | const0 | 71.586 / 70.845 | 48.6715 / 46.5285 | 179.3585 / 176.815 | −1.03512% | −4.40299% | −1.41811% |
| 4608 | random | 224.995 / 223.574 | 133.1785 / 127.8705 | 453.671 / 446.8985 | −0.63157% | −3.98563% | −1.49282% |
| 4608 | const0 | 180.129 / 179.207 | 117.315 / 113.789 | 395.5015 / 391.056 | −0.51186% | −3.00558% | −1.12402% |

| tokens | data | G1 paired median | G2 paired median | e2e paired median | G1/G2/e2e slower pairs (of 24) |
|---:|---|---:|---:|---:|---|
| 1536 | random | −1.31490% | −1.11820% | −1.63152% | 0 / 0 / 3 |
| 1536 | const0 | −0.92538% | −4.35469% | −1.27596% | 0 / 0 / 0 |
| 4608 | random | −0.63167% | −4.09618% | −1.43553% | 1 / 0 / 0 |
| 4608 | const0 | −0.52166% | −3.01354% | −1.15978% | 1 / 0 / 1 |

最终源码已完成 seeds0/1/7/42 × T1536/T4608 × random/const0 ×100/stage，
共3200次 stage 重放；另12个独立进程共1440次 stage 重放。所有 `stage_exact=True`，
isolated GEMM1/GEMM2 failures 均为0。T1536 random 的最大 `logits_diff` 为
`3.38491010e-06`、`rel_l2=0.00260188570`；T4608 random 为 `3.38296051e-06`、
`0.00260113669`；const0 两值均为0，与旧无 drain 一致。

packed amax helper 已逐值验证75,497,472个值，包含全部65,536 BF16 encodings、
zero peers、NaN/Inf/denormal 和 random raw encodings。FP32 scale bits、E8M0 byte、
两组 FP4 payload 均为零差异。CDNA5 ISA §15.10 p438 记载 maximumNumber 优先 numeric
而非 NaN；§7.7 p83 规定 packed BF16 保留 denorm，不受 MODE.denorm 影响。

ISA 审核记录见 [isa_stage2_review.json](../.codex_tmp/t192_overlap_20261012/resume_evidence/isa_stage2_review.json)。
GEMM1 stage2 release fence 在 `0x5174–0x5180`，first output DS store 在 `0x5dec`，
地址生成包含 `+0x20000`，生成42条 `v_pk_max_num_bf16`。GEMM2第一片 DS store
在 `0x5918`，release barrier 在 `0x5a70–0x5a78`，第二片 first DS store 在
`0x5bd0`，DS immediate 最大17504。

此前 v58 的12轮筛选虽然全过，24轮长测的 T1536 random G1/e2e 仍分别回退
`+0.74849%/+0.65174%`，各23/24 pairs变慢，因此未验收。v59/v60也存在单项回退。
本节的最终配置和数据只对应通过长测的 stage2 final，不使用这些失败候选。

原始验收证据：

```text
.codex_tmp/t192_overlap_20261012/resume_evidence/stage2_finalperf_random.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/stage2_finalperf_const0.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/stage2_final_stress_random.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/stage2_final_stress_const0.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/stage2_final_process_1..12.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/accepted_amax_boundary.jsonl
```

生产 default route 的正确性回归已完成：

| 范围 | token | seeds/data | stage replays | valid failures |
|---|---|---|---:|---:|
| E64 exact | 1536,4608 | 0/7 × random/const0 | 1600 | 0 |
| E64 boundary | 1024,1025,1535,1537,2048,3072,4096,4104,4607,4609,6144 | 0/7 × random/const0 | 2640 | 0 |
| E96/topk6/I3072 | 16384 | 0/7 × random/const0 | 240 | 0 |

exact 两点 isolated raw payload/ScaleA/GEMM2 也完全一致。E64 部分 boundary 的
whole allocation raw payload 变化：random 在1024/1025/1537/4104共240次，const0
在1024/1025/4104共127次；每次 valid FP4 semantic mismatch、ScaleA mismatch、
GEMM2 valid-byte mismatch 都为0。这些 raw 变化不能当作 bitwise-equal，也不能
当成有效输出数值失败。所有 `stage_exact=True`。回归原始文件与汇总保存为：

```text
.codex_tmp/t192_overlap_20261012/resume_evidence/production_exact_random.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/production_exact_const0.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/production_boundary_e64_random.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/production_boundary_e64_const0.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/production_boundary_e96_random.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/production_boundary_e96_const0.jsonl
.codex_tmp/t192_overlap_20261012/resume_evidence/production_summary.json
```

替换生产源码并更新两个 exact interval 后，直接使用 `--profile default_route`
再次执行24×201相邻 A/B；所有记录中的 GEMM 名称确认包含 `_persist`，不含
`_pmdrain`，12项 median 和 paired median 再次全部不回退：

| tokens | data | GEMM1 old/new median (us) | GEMM2 old/new median (us) | MoE e2e old/new median (us) | GEMM1 delta | GEMM2 delta | e2e delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1536 | random | 89.974 / 87.3295 | 59.889 / 58.5465 | 211.909 / 206.9205 | −2.93918% | −2.24165% | −2.35408% |
| 1536 | const0 | 69.503 / 68.542 | 46.148 / 45.4275 | 176.014 / 173.9715 | −1.38267% | −1.56128% | −1.16042% |
| 4608 | random | 226.398 / 225.0965 | 133.3385 / 128.8115 | 457.1545 / 451.567 | −0.57487% | −3.39512% | −1.22223% |
| 4608 | const0 | 179.508 / 177.6855 | 117.495 / 114.3705 | 393.039 / 387.910 | −1.01528% | −2.65926% | −1.30496% |

本表是单独的 production/default-route 复核，全部24轮均保留，与先前候选验收分开
归档。报告为 [default_route_report.json](../.codex_tmp/t192_overlap_20261012/resume_evidence/default_route_report.json)，
raw 为同目录 `default_routeperf_random.jsonl` 和 `default_routeperf_const0.jsonl`。
测试后再次检查整机 GPU/KFD，未发现测试进程残留。静态检查包括 `py_compile`、
Ruff 和 `git diff --check`。本地和 a07-3 仅同步任务所需文件，未 clean/reset、
未改时钟、未 commit/push。

### 最终 GEMM1/GEMM2 kernel 选择配置表

aligned override 的条件是 `tokens * topk` 能整除 `experts`，且每个 expert 的 route
行数能整除 `tile_m=256`。该行优先于 `[4609,+∞)` 的 non-aligned 默认配置。
cluster 两列按 `(cluster_m, cluster_n)` 记录实际传给 kernel launcher 的值；未指定且
无法解析时填写 `None`。t64 profile 的自动值在 launch 前解析为 `(1,1)`。

| token 区间/条件 | GEMM1 | GEMM1 cluster `(m,n)` | GEMM2 | GEMM2 cluster `(m,n)` |
|---|---|---|---|---|
| `[1, 1024)` | `t64/w1x4/b3` | `(1,1)` | `t64/w1x4/b3` | `(1,1)` |
| `[1024, 1536)` | persistent `t192/w2x4/b4_pmds_pmdrain` | `(1,4)` | persistent `t192/w2x4/b4_pmdrain` | `(1,4)` |
| `[1536, 1537)` | persistent `t192/w2x4/b4` | `(1,4)` | persistent `t192/w2x4/b4` | `(1,4)` |
| `[1537, 2048)` | persistent `t256/w2x4/b4_pmds_pmdrain` | `(1,4)` | persistent `t256/w2x4/b4_pmdrain` | `(1,4)` |
| `[2048, 2049)` | persistent `t256/w2x4/b4` | `(1,4)` | persistent `t256/w2x4/b4` | `(1,4)` |
| `[2049, 3073)` | persistent `t192/w2x4/b4_pmds_pmdrain` | `(1,4)` | persistent `t192/w2x4/b4_pmdrain` | `(1,4)` |
| `[3073, 3840)` | persistent `t256/w2x4/b4_pmds_pmdrain` | `(1,4)` | persistent `t256/w2x4/b4_pmdrain` | `(1,4)` |
| `[3840, 4096)` | optimized `t256/w2x2/b4_pmds` | `(1,4)` | persistent `t256/w2x4/b4_pmdrain` | `(1,4)` |
| `[4096, 4097)` | optimized `t256/w2x2/b4` | `(1,4)` | persistent `t256/w2x4/b4` | `(1,4)` |
| `[4097, 4104)` | optimized `t256/w2x2/b4_pmds` | `(1,4)` | persistent `t256/w2x2/b4_pmdrain` | `(1,4)` |
| `[4104, 4608)` | persistent `t192/w2x4/b4_pmds_pmdrain` | `(1,4)` | persistent `t192/w2x4/b4_pmdrain` | `(1,4)` |
| `[4608, 4609)` | persistent `t192/w2x4/b4` | `(1,4)` | persistent `t192/w2x4/b4` | `(1,4)` |
| `[4609, +∞)`, non-aligned | optimized `t256/w2x2/b4_pmds` | `(1,4)` | persistent `t256/w2x2/b4_pmdrain` | `(1,4)` |
| `>=4609`, aligned override | optimized `t256/w2x2/b4` | `(1,4)` | persistent `t256/w2x2/b4` | `(1,4)` |
