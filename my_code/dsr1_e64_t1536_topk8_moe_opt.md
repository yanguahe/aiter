# gfx1250 E64/T1536/topk8 MoE tile 配置对比

## 测试环境

- 机器：`a07-3`
- 容器：`hyg_fyd_e2e`
- 仓库目录：`/app/aiter`
- 数据格式：A4W4
- activation：SiLU
- bias：关闭
- shape：
  - experts：64
  - tokens：1536
  - topk：8
  - model_dim：7168
  - inter_dim：2048
- 每个测试使用 `--iters 20`
- 性能测试前后均确认 GPU/KFD 空闲

## 对比版本

### 启用前

Commit：

```text
5d9e7dd9219eb50ab269d7e3b0fcaf1e13b4e064
```

该版本没有匹配到以下 grouped MoE CSV key：

```text
token=2048
model_dim=7168
inter_dim=2048
expert=64
topk=8
```

其中实际输入的 `tokens=1536` 经 `_get_padded_m()` 后使用 `token=2048` 进行配置查找。由于没有匹配的 tuned row，代码使用默认 tile。

实际 kernel：

```text
GEMM1: a8w4_tdm_fp4_t64x256x256_w1x4_b3_K7168_e64_act1_q1r4
GEMM2: a8w4_tdm_fp4_t64x256x256_w1x4_b3_K2048_e64
```

### 启用后

Commit：

```text
9d4ccd6295811d31aa8eba2a56a8a7cc2d25b8e5
```

该版本在 `aiter/configs/tuned_grouped_fmoe.csv` 中增加了 `token=2048, expert=64, topk=8` 的精确配置，GEMM1 和 GEMM2 使用：

```text
tile_m/tile_n/tile_k       = 256/256/256
tile_m2/tile_n2/tile_k2    = 256/256/256
m_warp/n_warp              = 2/2
m_warp2/n_warp2            = 2/2
num_buffers                = 4
num_buffer_stage2          = 4
cluster_n                  = 4
waves_per_tensor_tdm       = 2
next_stage_prefetch        = 1
```

实际 kernel：

```text
GEMM1: a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p_s4
GEMM2: a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2
```

## 复现命令

### 进入测试环境

```bash
ssh -tt \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  yanguahe@heliosr-1b114-a07-3.mnb.dcgpu

docker exec -it hyg_fyd_e2e bash
cd /app/aiter
```

### 启用前版本

切换到新增 tuned row 之前的 commit：

```bash
git -c advice.detachedHead=false checkout --detach \
  5d9e7dd9219eb50ab269d7e3b0fcaf1e13b4e064
```

在同一次运行中依次测试 random 和 const0：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-both \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

启用前测试日志：

```text
/app/aiter/my_code/moe_prefill_switch_ab_runs/20260929T140651Z
```

### 启用后版本

恢复优化分支并确认 commit：

```bash
git checkout hyg/moe_a4w4_pr
git rev-parse HEAD
```

预期输出：

```text
9d4ccd6295811d31aa8eba2a56a8a7cc2d25b8e5
```

测试 random 和 const0：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-both \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

启用后也可以分别复现两种数据：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048

ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

本次启用后测试日志：

```text
const0: /app/aiter/my_code/moe_prefill_switch_ab_runs/20260929T140002Z
random: /app/aiter/my_code/moe_prefill_switch_ab_runs/20260929T140111Z
```

## 性能和正确性

### 启用前

| 数据 | GEMM1 | GEMM2 | fused MoE | logits_diff | rel_l2 | pass |
|---|---:|---:|---:|---:|---:|:---:|
| random | 135.889 us | 78.190 us | 253.09 us | 3.38491e-06 | 0.00260189 | True |
| const0 | 122.380 us | 73.498 us | 233.84 us | 0 | 0 | True |

### 启用后

| 数据 | GEMM1 | GEMM2 | fused MoE | logits_diff | rel_l2 | pass |
|---|---:|---:|---:|---:|---:|:---:|
| random | 103.385 us | 60.293 us | 221.25 us | 3.38491e-06 | 0.00260189 | True |
| const0 | 92.033 us | 55.465 us | 204.97 us | 0 | 0 | True |

## 性能提升

| 数据 | 指标 | 启用前 | 启用后 | 提升 |
|---|---|---:|---:|---:|
| const0 | GEMM1 | 122.380 us | 92.033 us | 24.80% |
| const0 | GEMM2 | 73.498 us | 55.465 us | 24.54% |
| const0 | fused MoE | 233.84 us | 204.97 us | 12.35% |
| random | GEMM1 | 135.889 us | 103.385 us | 23.92% |
| random | GEMM2 | 78.190 us | 60.293 us | 22.89% |
| random | fused MoE | 253.09 us | 221.25 us | 12.58% |

## 正确性结论

- 启用前和启用后的 random 测试均通过生产精度门限：

  ```text
  logits_diff = 3.38491e-06
  rel_l2      = 0.00260189
  ```

- 启用前和启用后的 const0 结果均完全一致：

  ```text
  logits_diff = 0
  rel_l2      = 0
  ```

- 新增 tuned row 只改变 kernel tile、cluster、A-preshuffle producer 和调度方式，没有引入可观察的精度回退。

## GEMM1 `cluster_m=1` 实验

在启用 `t256x256x256/w2x2/b4` 后，继续比较 GEMM1 的两种 cluster 几何：

```text
原版本：cluster_m=4, cluster_n=4，即 cluster=(4,4,1)
实验版：cluster_m=1, cluster_n=4，即 cluster=(4,1,1)
```

GEMM2 原本就是 `cluster=(4,1,1)`，本实验没有改变 GEMM2 的 cluster 几何。

### 代码改动

文件：

```text
aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
```

核心修改：

```diff
-    cluster_m = 4 if fp4_prefill_schedule else 1
+    cluster_m = 1
```

实验版还为 GEMM1 增加 `_cm1` symbol 标记，以避免与原 4×4 kernel 的 JIT cache 混淆，并便于 profiler 确认实际运行版本：

```diff
+    _cluster_m = "_cm1" if fp4_prefill_schedule else ""
     _kname = (
         "a8w4_tdm_fp4"
         f"_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}"
         f"_b{num_buffers}_K{K}"
-        f"{_grouped}{_act}_cn4_prefetch{_epilogue_batch}_apre_sh"
+        f"{_grouped}{_act}_cn4{_cluster_m}_prefetch{_epilogue_batch}_apre_sh"
```

实际 GEMM1 symbol：

```text
4x4:
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p_s4

4x1:
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p_s4
```

GEMM2 在两个版本中相同：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2
```

### const0 三轮性能

两个版本都在 a07-3 GPU/KFD 空闲时使用以下命令运行：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

原 4×4 cluster：

| 指标 | 三轮 samples | 中位数 |
|---|---|---:|
| GEMM1 | 90.155 us, 90.900 us, 91.214 us | 90.900 us |
| GEMM2 | 55.358 us, 55.054 us, 55.398 us | 55.358 us |
| fused MoE | 203.29 us, 203.40 us, 204.10 us | 203.40 us |

实验 4×1 cluster：

| 指标 | 三轮 samples | 中位数 |
|---|---|---:|
| GEMM1 | 84.555 us, 86.232 us, 84.808 us | 84.808 us |
| GEMM2 | 54.801 us, 55.685 us, 54.839 us | 54.839 us |
| fused MoE | 196.80 us, 198.87 us, 196.81 us | 196.81 us |

中位数对比：

| 指标 | 4×4 cluster | 4×1 cluster | 时间降低 | 性能提升 |
|---|---:|---:|---:|---:|
| GEMM1 | 90.900 us | 84.808 us | 6.091 us | 6.70% |
| GEMM2 | 55.358 us | 54.839 us | 0.518 us | 0.94% |
| fused MoE | 203.40 us | 196.81 us | 6.580 us | 3.24% |

两种 cluster 几何的 const0 结果均完全一致：

```text
logits_diff = 0
rel_l2      = 0
pass        = True
```

### 4×1 random 正确性和性能

复现命令：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

结果：

| GEMM1 | GEMM2 | fused MoE | logits_diff | rel_l2 | pass |
|---:|---:|---:|---:|---:|:---:|
| 97.237 us | 61.194 us | 216.17 us | 3.38491e-06 | 0.00260189 | True |

random 的误差与原 4×4 版本一致，没有精度回退。

### 性能变化原因

该测试启用 expert balance：

```text
1536 tokens × topk 8 / 64 experts = 192 routes/expert
```

每个 expert 的 192 行按 `tile_m=256` 对齐后只占一个 M tile。因此原 4×4 cluster 中，B/ScaleB 沿 M 方向的 expert-aware multicast 实际退化为 self-only，无法获得跨 M workgroup 的数据复用，但仍需承担 16-WG cluster、cluster barrier 和 co-residency 约束。

改成 4×1 后：

- 保留 A payload 沿 N 方向的 4-way multicast。
- 去掉 M 方向的 cluster barrier。
- 不再为本规模中没有实际收益的 B/ScaleB M 方向 multicast 组织 4 个 M peers。
- GEMM1 性能稳定提升约 6.7%，并传递为约 3.2% 的 fused MoE 提升。

### 4×1 实验复现流程

在本地修改并检查：

```bash
python -m black --check \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py

python -m ruff check --no-cache \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
```

将实验 kernel 上传到 a07-3：

```bash
scp \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py \
  yanguahe@heliosr-1b114-a07-3.mnb.dcgpu:/data/yanguahe/code/wk_sp1/mxfp4_preshuffle_gfx1250_tdm_cluster_m1.py
```

在 a07-3 创建独立 worktree：

```bash
docker exec hyg_fyd_e2e bash -lc '
  cd /app/aiter &&
  git worktree add --detach /tmp/aiter_cluster_m1_20260929 \
    9d4ccd6295811d31aa8eba2a56a8a7cc2d25b8e5
'
```

将上传文件复制进容器 worktree，并创建仅用于测试的临时 commit：

```bash
docker cp \
  /data/yanguahe/code/wk_sp1/mxfp4_preshuffle_gfx1250_tdm_cluster_m1.py \
  hyg_fyd_e2e:/tmp/aiter_cluster_m1_20260929/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py

docker exec hyg_fyd_e2e bash -lc '
  cd /tmp/aiter_cluster_m1_20260929 &&
  git add aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py &&
  git -c user.name=Codex -c user.email=codex@local commit \
    -m "Temporary GEMM1 cluster_m1 benchmark"
'
```

运行三轮 const0：

```bash
docker exec hyg_fyd_e2e bash -lc '
  cd /tmp/aiter_cluster_m1_20260929 &&
  ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048
'
```

运行 random：

```bash
docker exec hyg_fyd_e2e bash -lc '
  cd /tmp/aiter_cluster_m1_20260929 &&
  ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048
'
```

### 日志位置

```text
4×4 const0 三轮：
/app/aiter/my_code/moe_prefill_switch_ab_runs/20260929T143430Z

4×1 const0 三轮：
/app/aiter/my_code/moe_prefill_switch_ab_runs/cluster_m1_const0_rounds3_20260929T143553Z

4×1 random：
/app/aiter/my_code/moe_prefill_switch_ab_runs/cluster_m1_random_20260929T143747Z
```

4×1 实验结束后，容器正式仓库仍保持：

```text
hyg/moe_a4w4_pr
9d4ccd6295811d31aa8eba2a56a8a7cc2d25b8e5
```

## CSV 驱动的 `cluster_m=1` 与 t192 优化

本轮在前述 `t256x256x256/w2x2/b4` 配置基础上完成了两项优化：

1. 为 `aiter/configs/tuned_grouped_fmoe.csv` 新增 `cluster_m` 字段，通过 CSV 将 GEMM1 的 `cluster=(4,1,1)` 严格限制在当前调优规模。
2. 增加并修正 `t192x256x256/w2x2/b4` 调度，在通过 random MoE e2e 正确性后，将其写入当前规模的正式 tuned row。

### 最终 CSV 配置

只有以下精确 key 设置 `cluster_m=1`：

```text
token=2048
model_dim=7168
inter_dim=2048
expert=64
topk=8
```

实际输入为 `tokens=1536`，配置查询经过 padding 后使用 `token=2048` 作为 key。其他 CSV row 的 `cluster_m` 留空，GEMM1 继续使用默认 `cluster_m=4`。

目标 row 最终配置：

```text
tile_m/tile_n/tile_k       = 192/256/256
tile_m2/tile_n2/tile_k2    = 192/256/256
m_warp/n_warp              = 2/2
m_warp2/n_warp2            = 2/2
num_buffers                = 4
num_buffer_stage2          = 4
cluster_n                  = 4
cluster_m                  = 1
waves_per_tensor_tdm       = 2
next_stage_prefetch        = 1
```

因此 GEMM1 使用 `cluster=(4,1,1)`。未配置 `cluster_m` 的其他 GEMM1 规模仍默认使用 `cluster=(4,4,1)`。

### t192 实现修正

最初的 t192 版本仅放宽了 tile 检查，并直接复用了 t256 的调度参数。const0 无法暴露其中的错误，random 测试会产生错误结果。最终完成以下修正：

- t256 每个 k128 包含 32 条物理 WMMA，原调度固定使用 `FENCE_COVER_MMA=28`。
- t192 每个 k128 只有 24 条物理 WMMA。继续使用 28 会产生错误的 scheduler hint，因此改为按 tile 计算：t192 使用 `fc20`，t256 保持 `fc28`。
- t192 的 output store 从不均衡的 `4+2` 改为 `3+3`，使第一次 TDM store 后有更多独立 epilogue 工作可用于隐藏延迟。
- random 错误最终定位到 drain tail 的 `buf_ptr_opaque()` carry-target 地址。该技巧在 t192 生成的 LDS 地址形态下会读错 carry target，而全零数据会掩盖错误。t192 改用普通 foldable LDS 地址，t256 继续使用原来的 opaque 地址路径。

最终实际 kernel：

```text
GEMM1:
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_rcw_mg4_fc20_xdl0_reuse_ostore2p_s3

GEMM2:
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc20_ostore2p_s3_ow2
```

### 相邻三轮 A/B 性能

测试环境：

```text
machine:   a07-3
container: hyg_fyd_e2e
repo:      /tmp/aiter_t192_cm1_20260929
shape:     E64/T1536/topk8/M7168/I2048
data:      const0
iters:     20
```

每次正式性能测试前后均使用 `/data/yanguahe/code/gpu_users.sh` 确认没有 GPU/KFD 占用。t256 和 t192 在同一时段相邻执行。

| 配置 | GEMM1 samples | GEMM1 median | GEMM2 samples | GEMM2 median | MoE e2e samples | MoE e2e median |
|---|---|---:|---|---:|---|---:|
| t256 + `cluster_m=1` | 84.302, 83.990, 84.760 us | 84.302 us | 55.657, 56.393, 54.970 us | 55.657 us | 198.51, 200.27, 198.48 us | 198.51 us |
| t192 + `cluster_m=1` | 80.897, 80.676, 81.435 us | 80.897 us | 54.489, 54.310, 54.034 us | 54.310 us | 194.50, 194.31, 193.15 us | 194.31 us |

| 指标 | t256 median | t192 median | t192 性能提升 |
|---|---:|---:|---:|
| GEMM1 | 84.302 us | 80.897 us | 4.04% |
| GEMM2 | 55.657 us | 54.310 us | 2.42% |
| fused MoE | 198.51 us | 194.31 us | 2.12% |

最终四个文件重新同步后的单轮 const0：

```text
GEMM1      = 81.399 us
GEMM2      = 54.708 us
fused MoE  = 194.60 us
logits_diff = 0
rel_l2      = 0
```

### 正确性

random MoE e2e：

```text
logits_diff = 3.3849e-06
rel_l2      = 2.6019e-03
pass        = True
```

const0：

```text
logits_diff = 0
rel_l2      = 0
pass        = True
```

const0 下 GEMM1、GEMM2 和最终 MoE 的 reference/output hash 均完全一致。random 通过 production `logits_diff < 0.01` 门限，且误差与此前正确版本一致。

### 复现环境

从 a07-3 主机进入容器：

```bash
docker exec -it hyg_fyd_e2e bash
cd /tmp/aiter_t192_cm1_20260929
```

当前临时测试树包含未提交的测试文件，因此直接运行 `env + python` 命令，不使用会检查 clean worktree 的 `run_moe_prefill_switch_ab.sh`。

### 检查 GPU 空闲

性能测试前后在 a07-3 主机执行：

```bash
/data/yanguahe/code/gpu_users.sh
```

如果输出显示其他 GPU/KFD 进程，运行结果只能用于正确性，不能作为性能数据。

### 切换目标 CSV row 的 tile

以下命令只修改 `/tmp/aiter_t192_cm1_20260929` 临时测试树中当前规模的 row。设置 `TILE_M=256` 可复现 t256 对照，设置 `TILE_M=192` 可恢复最终 t192 配置：

```bash
cd /tmp/aiter_t192_cm1_20260929

set_target_tile() {
  TILE_M="$1" python3 - <<'PY'
import os
from pathlib import Path

path = Path("aiter/configs/tuned_grouped_fmoe.csv")
lines = path.read_text(encoding="utf-8").splitlines()
header = lines[0].split(",")
tile_m = os.environ["TILE_M"]
matches = 0

for index in range(1, len(lines)):
    fields = lines[index].split(",")
    row = dict(zip(header, fields))
    if (
        row["token"] == "2048"
        and row["model_dim"] == "7168"
        and row["inter_dim"] == "2048"
        and row["expert"] == "64"
        and row["topk"] == "8"
    ):
        fields[header.index("tile_m")] = tile_m
        fields[header.index("tile_m2")] = tile_m
        lines[index] = ",".join(fields)
        matches += 1

assert matches == 1, f"expected one target row, got {matches}"
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"target row tile_m/tile_m2 set to {tile_m}/{tile_m}")
PY
}
```

复现 t256 对照配置：

```bash
set_target_tile 256
```

恢复最终 t192 配置：

```bash
set_target_tile 192
```

### 单轮 const0 正确性和性能

```bash
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
FLYDSL_DUMP_IR=0 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20 \
  --const-init 0
```

该命令同时验证 const0 正确性，并输出 GEMM1、GEMM2 和 fused MoE 时间。性能数据仅在 GPU 空闲时有效。

### 单轮 random 正确性

```bash
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
FLYDSL_DUMP_IR=0 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20
```

random 是判断 t192 drain/carry 地址是否正确的必要测试；只运行 `--const-init 0` 会掩盖错误 LDS 地址读取。

### 三轮 const0 性能

```bash
mkdir -p /tmp/e64_t192_final_logs

for round in 1 2 3; do
  log="/tmp/e64_t192_final_logs/const0_round_${round}.log"
  ENABLE_CK=0 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_LOG_MORE=1 \
  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  FLYDSL_DUMP_IR=0 \
  python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario bench \
    --data-format a4w4 \
    --act silu \
    --no-bias \
    --no-check-aot-cache \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048 \
    --iters 20 \
    --const-init 0 \
    2>&1 | tee "$log"
done
```

提取三轮关键结果：

```bash
grep -hE \
  'eager-profile:|fused_moe end-to-end us|bench kernel timing' \
  /tmp/e64_t192_final_logs/const0_round_*.log
```

### 完整 A/B 顺序

1. 检查 GPU 空闲。
2. 使用上述 CSV 切换命令设置 `TILE_M=256`。
3. 执行三轮 const0，保存 t256 日志。
4. 再次检查 GPU 空闲。
5. 使用上述 CSV 切换命令设置 `TILE_M=192`。
6. 执行三轮 const0，保存 t192 日志。
7. 执行一轮 random，确认 `logits_diff` 和 `rel_l2`。
8. 测试结束后再次确认 GPU/KFD 没有残留进程。

本轮原始日志位于容器内：

```text
t192 三轮 const0:
/tmp/t192_final_const0_r1.log
/tmp/t192_final_const0_r2.log
/tmp/t192_final_const0_r3.log

t256 相邻三轮 const0:
/tmp/t256_cm1_adjacent_const0_r1.log
/tmp/t256_cm1_adjacent_const0_r2.log
/tmp/t256_cm1_adjacent_const0_r3.log

最终同步文件后的 random:
/tmp/t192_final_exact_random.log

最终同步文件后的 const0:
/tmp/t192_final_exact_const0.log
```

## GEMM1 B payload TDM cache hint 优化

### 优化内容

该优化只修改 GEMM1 读取权重 B payload 时使用的 TDM descriptor：

```text
tdm_b_th = 6
TH=6 = NT_HT
near cache: non-temporal
far cache: high-priority temporal
```

实现中将 `tdm_b_th` 从目标规模的 CSV row 传到 optimized GEMM launcher，并且只用于 B payload 的 `cache_modifier`。A payload、ScaleA、ScaleB、output store、tile、wave、cluster、MMA schedule 和计算逻辑均不变。kernel symbol 增加 `_bth6`，用于确认 profiler 实际运行了该版本。

目标 CSV key 和最终配置：

```text
token=2048
model_dim=7168
inter_dim=2048
expert=64
topk=8
tile_m=192
tile_m2=192
cluster_m=1
tdm_b_th=6
```

实际 GEMM1 symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_bth6_rcw_mg4_fc20_xdl0_reuse_ostore2p_s3
```

GEMM2 的两个调用点固定传入 `tdm_b_th=0`，因此正式代码只对 GEMM1 启用该优化。

### 复现命令

在 a07-3 主机上，每次性能运行前执行：

```bash
/data/yanguahe/code/gpu_users.sh
```

只有输出“当前没有进程在使用 GPU”时，才将后续结果计入性能对比。

进入容器并切换到仓库：

```bash
docker exec -it hyg_fyd_e2e bash
cd /app/aiter
```

运行正式的 GEMM1 `B_TH=6` 版本：

```bash
unset AITER_TDM_B_TH

ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
FLYDSL_DUMP_IR=0 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20 \
  --const-init 0
```

运行相同代码路径的 `B_TH=0` 对照，只需在同一命令前设置：

```bash
export AITER_TDM_B_TH=0
```

random 正确性验证使用同一命令，但删除 `--const-init 0`。

### 三轮相邻复测

测试顺序为：

```text
round 1: B_TH=0 -> B_TH=6
round 2: B_TH=6 -> B_TH=0
round 3: B_TH=0 -> B_TH=6
```

每个 case 运行前均确认整机 GPU/KFD 空闲。

| round | GEMM1 B_TH=0 | GEMM1 B_TH=6 | GEMM1 提升 | B_TH=0 MoE e2e | B_TH=6 MoE e2e |
|---:|---:|---:|---:|---:|---:|
| 1 | 85.801 us | 79.200 us | 7.69% | 216.180 us | 208.807 us |
| 2 | 95.563 us | 81.894 us | 14.30% | 224.937 us | 221.306 us |
| 3 | 95.462 us | 78.759 us | 17.50% | 225.010 us | 204.659 us |
| median | 95.462 us | 79.200 us | 17.03% | 224.937 us | 208.807 us |

由于 MI450 当前频率和性能状态会动态变化，三组绝对时间存在波动；三组相邻对比都显示 GEMM1 `B_TH=6` 更快，中位数降低 `16.262 us`。MoE e2e 中位数降低 `16.130 us`，对应提升 `7.17%`。

原始日志位于 a07-3 主机：

```text
/tmp/dsr1_bth6_final/r1_baseline.log
/tmp/dsr1_bth6_final/r1_optimized.log
/tmp/dsr1_bth6_final/r2_optimized.log
/tmp/dsr1_bth6_final/r2_baseline.log
/tmp/dsr1_bth6_final/r3_baseline.log
/tmp/dsr1_bth6_final/r3_optimized.log
```

### random 正确性

正式 GEMM1 `B_TH=6` 版本的 random MoE e2e 结果：

```text
logits_diff = 3.3849e-06
rel_l2      = 2.6019e-03
pass        = True
```

GEMM2 output hash 与 reference 完全一致；GEMM1 和最终 MoE output 因正常的 MXFP4 数值误差与 reference hash 不同，但误差与此前正确版本一致，并通过 `logits_diff < 0.01` 门限。

random 日志：

```text
/tmp/dsr1_bth6_final/random_optimized.log
```

### GEMM2 B payload `B_TH=6` 实验

为了隔离 GEMM2 的影响，实验期间 GEMM1 始终保持正式的 `B_TH=6`，只在两个 GEMM2 launch 点临时将 `tdm_b_th` 从 `0` 改为 `6`。测试结束后已恢复为 `0`，该实验代码未保留。

三轮相邻结果：

| round | GEMM2 B_TH=0 | GEMM2 B_TH=6 | B_TH=6 相对变化 |
|---:|---:|---:|---:|
| 1 | 60.378 us | 60.930 us | -0.92% |
| 2 | 59.908 us | 61.630 us | -2.88% |
| 3 | 61.822 us | 61.816 us | +0.01% |
| median | 60.378 us | 61.630 us | -2.07% |

GEMM2 `B_TH=6` 没有稳定收益，中位数反而回退 `1.253 us`。因此最终代码继续保持 GEMM2 `tdm_b_th=0`。

实验日志位于 a07-3 主机：

```text
/tmp/dsr1_bth6_final/g2_r1_b0.log
/tmp/dsr1_bth6_final/g2_r1_b6.log
/tmp/dsr1_bth6_final/g2_r2_b6.log
/tmp/dsr1_bth6_final/g2_r2_b0.log
/tmp/dsr1_bth6_final/g2_r3_b0.log
/tmp/dsr1_bth6_final/g2_r3_b6.log
```

## GEMM2 persistent task loop 优化（2026-10-01）

### 目标与基线

目标规模和复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

机器重启后的普通 GEMM2 baseline 为：

```text
59.707 us median
```

每次计入性能数据前均在 a07-3 主机执行：

```bash
/data/yanguahe/code/gpu_users.sh
```

只有整机无已有 GPU/KFD 进程时才保留该次性能结果。若运行后发现其他用户进程进入 GPU，
该次结果仅用于正确性，不计入性能比较。

### `ps7pf2`：跨 persistent task 预取 stage0/stage1

一个 4-WG cluster 固定处理同一个 M tile，并依次处理 7 个 N-cluster task。当前 task
完成 compute 后，在 output epilogue 开始前预取下一 task 的 stage0 和 stage1：

```text
I0, I1, O0, O1
```

下一 task 使用 `s_wait_tensorcnt 0x2`，确认 I0/I1 已完成，同时允许 O0/O1 继续在后台
drain。随后先计算 K tile 0，再等待旧 output 完成并补发 stages2-4。output LDS 位于：

```text
[2 * PITCH, 2 * PITCH + C_STORE_B)
```

该布局与 stage0/stage1 不重叠。目标 tile 的静态资源为：

```text
PITCH          = 60,928 B
input ring     = 243,712 B
output region  = [121,856, 226,304)
LDS limit      = 327,680 B
```

三轮相邻 const0 结果：

| round | `ps7pf2` | baseline | improvement |
|---:|---:|---:|---:|
| 1 | 56.591 us | 59.442 us | 4.80% |
| 2 | 58.147 us | 59.870 us | 2.88% |
| 3 | 55.181 us | 60.905 us | 9.40% |
| median | 56.591 us | 59.870 us | 5.48% |

balanced random 和非均衡 random 均通过；GEMM2 output hash 与对应 reference 完全一致。

### persistent metadata hoist

同一 persistent cluster 的 7 个 task 固定使用同一个 `m_tile`。因此以下值只依赖 M tile，
可以在 task loop 前计算一次：

```text
m_tile
blk_m
expert
mn_oob
tile_map pointer
`blk_m64` and other M-side address metadata
```

7 个 task 之间变化的只有 `n_unit/tile_idx`，以及由它派生的 `blk_n`、B/ScaleB 和输出
地址。kernel 不写 `arg_m_tile_map`，所以该 hoist 与原逐 task binary search 逻辑等价。

三轮相邻 const0 结果：

| round | metadata hoist | `ps7pf2` | improvement |
|---:|---:|---:|---:|
| 1 | 57.133 us | 58.419 us | 2.20% |
| 2 | 55.961 us | 58.516 us | 4.37% |
| 3 | 56.764 us | 58.348 us | 2.71% |
| median | 56.764 us | 58.419 us | 2.83% |

random 验证结果：

```text
logits_diff = 3.38491e-06
rel_l2      = 0.00260189
pass        = True
```

### metadata-hoist 版本 ATT 结果

采集 symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc20_ostore2p_s3_ow2_ps7pf2hm
```

采集目录：

```text
my_code/thread_trace_runs/e64_t1536_gemm2_ps7pf2hm_att_20261001
```

16 条代表 active wave 的统计：

| metric | value |
|---|---:|
| active-wave median | 78,939.5 cycles |
| explicit wait share | 36.04% |
| `s_barrier_wait` | 13.12% |
| `s_wait_tensorcnt` | 11.17% |
| `s_wait_dscnt` | 8.13% |
| `s_wait_kmcnt` | 3.46% |

主要重复热点：

| PC | instruction | hits/wave | wave-span share | meaning |
|---|---|---:|---:|---|
| `0x4428` | `s_barrier_wait 0xffff` | 11 | 5.00% | steady input-ring reuse barrier |
| `0x308c` | `s_barrier_wait 0xffff` | 6 | 4.05% | 后续 persistent task 的 stage0/1 ready barrier |
| `0x4b70` | `s_wait_tensorcnt 0x4` | 11 | 4.00% | steady TDM arrival |
| `0x306c` | `s_wait_tensorcnt 0x2` | 6 | 3.35% | 等待下一 task 的 stage0/1 |
| `0x27d0` | `s_wait_dscnt 0x0` | 7 | 3.11% | output LDS 第二段 drain；少数 wave 有长尾 |
| `0x6fb0` | `s_wait_tensorcnt 0x0` | 1 | 1.97% | kernel 最终 output TDM drain |

WGP completion imbalance 的 16 组 capture 平均值为 `4.85%`，中位数为 `2.48%`；主要瓶颈
仍位于单个 workgroup 内的 TDM/DS/barrier 等待，而不是全局 work 分配不均。

### zero-signal split barrier

后续 task 原逻辑在所有 descriptor setup 和 accumulator 清零之后执行完整
`tensor_wait(2) + workgroup barrier`。优化后在 descriptor setup 完成时执行：

```text
s_wait_tensorcnt 0x2
s_barrier_signal -1
```

随后用 accumulator 清零覆盖其他 wave 到达 barrier 的时间，并在读取 stage0 前执行：

```text
s_barrier_wait -1
```

两组相邻空闲复测均显示小幅稳定收益：

| round | zero-signal | metadata hoist | improvement |
|---:|---:|---:|---:|
| 1 | 54.744 us | 55.233 us | 0.89% |
| 2 | 54.251 us | 54.839 us | 1.07% |

random 结果通过，GEMM2 output hash 与 reference 完全一致。

### 已淘汰实验

- `ps7pf3`：预取 stage0/1/2，将 LDS 增至 `287,232 B`，const0 回退到约 `63.1 us`。
- 低位 buffer 立即回填：破坏 4-buffer input pipeline，random 虽正确但 GEMM2 回退到约 `107.9 us`。
- `I0,O0,I1,O1` TDM 排序：正确，但空闲 const0 为 `55.845 us`，弱于 zero-signal。
- GEMM2 persistent `B_TH=6`：正确，空闲 const0 为 `55.071 us`，未优于 `B_TH=0`。
- `t192x128/b3`：正确，但空闲 const0 为 `76.315 us`，N tile 数和固定开销翻倍抵消双驻留收益。
- `t192x256/b2`：正确，但空闲 const0 为 `68.529 us`，较浅 input pipeline 明显回退。
- Scale TDM owner 非对称重排：random 产生 NaN，已淘汰。
- 完整 A resident LDS 原型：当前实现 random 产生 NaN，未进入性能评估。

### a07-3 GPU fault boundary

`sudo dmesg -T` 显示 GPU 在 `2026-10-01 13:03:02 UTC` 首次报告：

```text
amdgpu ... [gfxhub0] no-retry page fault
Faulty UTCL2 client ID: TCP
PERMISSION_FAULTS: 0x3
RW: 0x0
```

随后在 `13:09:07-13:09:09 UTC` 报告：

```text
MES(0, 0) failed to respond to msg=REMOVE_QUEUE
MES(0, 0) failed to respond to msg=SUSPEND
failed to suspend all gangs
MES might be in unrecoverable state, issue a GPU reset
GPU recovery disabled
```

第一次 fault 与 steady split-fence 实验的运行时间紧邻，因此该实验按不安全版本淘汰。
从 `13:03:02 UTC` 起得到的所有性能数值均作废；后续 `pad8`、`B_TH=1`、单段 output
以及 A-resident-unicast 的长时间无返回不能作为这些候选自身性能或正确性的结论。恢复 GPU
测试前需要由机器管理员执行 GPU reset 或重启；优化过程不会自行执行这两项操作。

### GPU 故障前保留文件状态（已由下节更新）

GPU 故障发生前，本地工作树保留的 GEMM2 persistent 实现为
`ps7pf2 + metadata hoist + zero-signal`：

```text
aiter/ops/flydsl/grouped_gemm_mxfp4.py
aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_gemm2_persistent.py
```

本地与 a07-3 上恢复后的 SHA256 一致：

```text
107fc379a13fb677c774dbc910f684ec188a2394cc729ef5c9cee6038e9f71ed  aiter/ops/flydsl/grouped_gemm_mxfp4.py
f0b94f24a2833091c9ceb650f9d634996c9a43084e62c2a7eb1a5d14f303135f  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_gemm2_persistent.py
```

静态检查：

```bash
python -m py_compile \
  aiter/ops/flydsl/grouped_gemm_mxfp4.py \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_gemm2_persistent.py

python -m ruff check \
  aiter/ops/flydsl/grouped_gemm_mxfp4.py \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_gemm2_persistent.py
```

两项检查均通过。

## a07-3 重启后的关键节点复测（2026-10-01 14:18 UTC）

机器重启后先在主机执行：

```bash
/data/yanguahe/code/gpu_users.sh
rocm-smi --showuse --showmemuse
```

每个 case 运行前后均重新检查，确认没有既有 GPU/KFD 进程，`GPU use` 和
`GPU Memory Allocated` 均为 `0%`。四个版本按正序、逆序、正序交错测试，以降低时钟和温度
随时间变化带来的偏差。测试命令为：

```bash
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
FLYDSL_DUMP_IR=0 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20 \
  --const-init 0
```

原始日志位于：

```text
/data/yanguahe/code/wk_sp1/aiter/.codex_tmp/gemm2_reboot_rebench/runs/20261001T141807
```

三轮结果：

| round | version | GEMM1 | GEMM2 | fused MoE | pass | GEMM2 hash equals reference |
|---:|---|---:|---:|---:|:---:|:---:|
| 1 | ordinary baseline | 79.729 us | 60.038 us | 211.56 us | True | True |
| 1 | `ps7pf2` | 75.771 us | 58.391 us | 208.29 us | True | True |
| 1 | metadata hoist | 79.061 us | 54.986 us | 206.21 us | True | True |
| 1 | zero-signal | 79.660 us | 55.524 us | 208.13 us | True | True |
| 2 | zero-signal | 79.978 us | 54.565 us | 209.48 us | True | True |
| 2 | metadata hoist | 79.198 us | 55.372 us | 205.22 us | True | True |
| 2 | `ps7pf2` | 79.183 us | 58.484 us | 208.08 us | True | True |
| 2 | ordinary baseline | 79.504 us | 60.184 us | 215.52 us | True | True |
| 3 | ordinary baseline | 78.190 us | 60.753 us | 215.80 us | True | True |
| 3 | `ps7pf2` | 78.176 us | 58.427 us | 214.80 us | True | True |
| 3 | metadata hoist | 78.420 us | 53.831 us | 206.66 us | True | True |
| 3 | zero-signal | 78.365 us | 58.274 us | 220.95 us | True | True |

中位数汇总：

| version | GEMM2 median | vs ordinary baseline | fused MoE median | vs ordinary baseline |
|---|---:|---:|---:|---:|
| ordinary baseline | 60.184 us | 0.00% | 215.52 us | 0.00% |
| `ps7pf2` | 58.427 us | 2.92% | 208.29 us | 3.35% |
| metadata hoist | 54.986 us | 8.64% | 206.21 us | 4.32% |
| zero-signal | 55.524 us | 7.74% | 209.48 us | 2.80% |

本轮中 metadata hoist 的 GEMM2 中位数最好。zero-signal 的前两轮与 metadata hoist 接近，
第三轮升至 `58.274 us`，因此重启后没有复现此前约 `1%` 的稳定收益。该变化不影响正确性，
但在决定最终保留版本时应以新的相邻复测为准。

zero-signal 另做一次 random 验证：

```text
logits_diff                = 3.38491e-06
rel_l2                     = 0.00260189
pass                       = True
GEMM2 ref output hash128   = 0600dddfcca42f243e9176f595c8a2fe
GEMM2 output hash128       = 0600dddfcca42f243e9176f595c8a2fe
```

## 重启后继续优化与最终保留节点

### B-first TDM owner 分支排序

该版本只把 B/ScaleB owner 分支放到 A/ScaleA owner 分支之前，希望较大的 B payload 更早发射，
不改变地址、TDM 数量或同步协议。random 验证通过，GEMM2 output hash 与 reference 相同。

三轮相邻测试：

| round | metadata hoist | zero-signal | B-first |
|---:|---:|---:|---:|
| 1 | 53.707 us | 56.952 us | 53.342 us |
| 2 | 53.633 us | 56.138 us | 54.742 us |
| 3 | 54.449 us | 54.458 us | 53.836 us |
| median | 53.707 us | 56.138 us | 53.836 us |

B-first 相对 metadata hoist 中位数回退约 `0.24%`，没有稳定收益，未保留。

原始日志：

```text
/data/yanguahe/code/wk_sp1/aiter/.codex_tmp/gemm2_reboot_rebench/runs/20261001T142406
```

### `earlynext`：在最后一个 K tile 前启动下一 task

`ps7pf2` 原来在完整 K-loop 结束后才发射下一 persistent task 的 stage 0/1。`earlynext` 利用
最后一个 K tile 已经被前一轮 carry 到 register 的事实，将下一 task 的 `I0/I1` 提前到最后一个
K tile 的 WMMA 之前：

```text
... current stage 7 ready in rmem
I0(next), I1(next)
final K-tile WMMA
O0(current), O1(current)
```

因此仍满足 gfx1250 文档规定的同一 wave 内 TDM load/store 按发射顺序完成，同时下一 task 的
input 写入 buffers 0/1，当前 task 的 output arena 只覆盖 buffers 2/3，不存在 LDS 地址重叠。
原 epilogue 前的完整 `pipeline_fence(0)` 在 persistent 路径中可以删除；最终 task 的当前输入已经
在读取最后 K tile 前完成，kernel 末尾仍保留 `tensor_wait(0)` 等待 output 完成。

random 验证：

```text
logits_diff                = 3.38491e-06
rel_l2                     = 0.00260189
pass                       = True
GEMM2 output hash128       = 7bb3ce52d1938d8548cf80e23bab0d53
GEMM2 reference hash128    = 7bb3ce52d1938d8548cf80e23bab0d53
```

第一组三轮相邻测试：

| round | `earlynext` | metadata hoist | improvement |
|---:|---:|---:|---:|
| 1 | 53.211 us | 57.624 us | 7.66% |
| 2 | 53.291 us | 54.358 us | 1.96% |
| 3 | 55.258 us | 54.499 us | -1.39% |
| median | 53.291 us | 54.499 us | 2.22% |

另一组紧邻单轮为 `55.195 us` 对 `55.737 us`，`earlynext` 提升 `0.97%`。四组配对中三组更快，
收益幅度受机器动态状态影响，但方向可重复。当前正式工作树保留该版本。

### output split sweep

保持 metadata hoist，其余不变，只改变两个 output TDM slice 的 WMMA-row 分界。`4+2` 相比原来的
`3+3` 在三轮相邻测试中均更快：

| round | output `4+2` | output `3+3` | improvement |
|---:|---:|---:|---:|
| 1 | 55.912 us | 57.211 us | 2.27% |
| 2 | 53.863 us | 54.552 us | 1.26% |
| 3 | 54.660 us | 55.735 us | 1.93% |
| median | 54.660 us | 55.735 us | 1.93% |

`4+2` 与 `earlynext` 组合后反而回退；两者直接比较的三轮中位数为 `55.741 us` 对
`53.774 us`。因此只保留更快的 `earlynext`，不叠加 `4+2`。

原始日志：

```text
/data/yanguahe/code/wk_sp1/aiter/.codex_tmp/gemm2_reboot_rebench/runs/20261001T165148
/data/yanguahe/code/wk_sp1/aiter/.codex_tmp/gemm2_reboot_rebench/runs/20261001T170555
```

### 本轮未保留的其他候选

- A payload `TH=2` 单轮为 `55.593 us`，仅落在噪声范围；`TH=6` 为 `57.885 us`，A/ScaleA
  同时使用 `TH=6` 为 `57.809 us`。硬件资料明确指出 multicast load 会 bypass WGP$，因此这些
  hint 无法改善近端 cache 命中。
- 将 output descriptor setup 移到 LDS barrier 前通过了 random，但相邻单轮为 `53.867 us`，
  慢于 `earlynext` 的 `53.549 us`。
- 预构建 input TDM descriptor 并用 task-dependent `imm_offset` 复用，通过了 random；相邻单轮
  为 `54.491 us`，慢于 `earlynext` 的 `53.603 us`。增加的 descriptor live range 没有换来收益。
- `fence_cover_mma=8/12/16` 单轮分别为 `58.525/56.229/58.297 us`，均慢于同轮默认
  `fence_cover_mma=20` 的 `55.431 us`；`24` 会令 `mma_total == 0`，当前 scheduler helper 因此
  无法生成合法 schedule。
- `staggerednext` 更早复用 buffers 0/1，虽然最终 MoE 门限仍通过，但 GEMM2 output hash 不再等于
  reference，`rel_l2` 增至 `0.0281119`，违反精度要求，已淘汰。
- stage-0 A/ScaleA LDS resident、ScaleA-only resident 及低地址 ScaleA resident 均在 random 下产生
  NaN。它们没有进入性能比较，也不会合入正式代码。

### 当前正式候选

当前保留 kernel 为：

```text
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc20_ostore2p_s3_ow2_ps7pf2hm_earlynext
```

文件 SHA256：

```text
08692fc27794cbd7211c49de5f9f1d47702adba3582af39b638266a459120cd5  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_gemm2_persistent.py
```

本轮硬件判断依据：

- `MI400_Shader_Programming#65.txt` 2.4 节说明，同一 wave 的 TDM load/store 相互保持发射顺序；
  `earlynext` 因而维持 `I0,I1,O0,O1`，下一 task 的 `tensor_wait(2)` 可只留下旧 output。
- 同一文档 4.10.8 节说明每 wave 最多 3 个 TDM 等待 XACK、每 SIMD 最多 6 个；提前发射可能因
  XACK 限额短暂停顿，但不会改变完成顺序。
- 同一文档 LDS 章节说明 384 KiB SRAM 按 64 KiB 在 LDS/WGP$ 间分区，LDS 最大 320 KiB。
当前约 238 KiB 分配落在 256 KiB 档并保留 128 KiB WGP$；将完整 A 常驻会进入 320 KiB 档，
只剩 64 KiB WGP$，因此没有作为保留方案。

### 正式工作树验证

正式文件已切换为 `metadata hoist + earlynext`，本地与 a07-3 的 kernel SHA256 均为：

```text
08692fc27794cbd7211c49de5f9f1d47702adba3582af39b638266a459120cd5
```

使用正式入口执行 random MoE e2e：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

结果：

```text
GEMM1                    = 93.022 us
GEMM2                    = 68.335 us
fused MoE                = 235.52 us
logits_diff              = 3.38491e-06
rel_l2                   = 0.00260189
pass                     = True
GEMM2 reference hash128  = 72c5ad0345105b236d59a61386c8018d
GEMM2 output hash128     = 72c5ad0345105b236d59a61386c8018d
```

random 数据的耗时不与 const0 性能数据横向比较；这里用于确认最终正式文件仍保持数值正确。
日志位于：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/moe_prefill_switch_ab_runs/20261001T173856Z
```

静态验证：

```text
local py_compile: pass
local ruff:       pass
local diff-check: pass
remote py_compile: pass
remote ruff:       unavailable (/opt/venv/bin/python3: No module named ruff)
```
