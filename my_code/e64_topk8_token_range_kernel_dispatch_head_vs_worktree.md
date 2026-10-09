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
