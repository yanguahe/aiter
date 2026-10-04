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

## GEMM2 eight-wave latency hiding（2026-10-02）

### 优化动机

T512 的 `t64/w1x4/b3` ATT 显示，每个 physical SIMD 可同时驻留两个 wave，
虽然单 wave 的 exposed stall 很高，dispatch 层仍可由另一个 resident wave
隐藏 TDM 和 barrier latency。T1536 原 `t192/w2x2/b4` 每个 SIMD 只有一个
wave，因此在保持 `tile_m=192` 和现有 A-preshuffle ABI 不变的前提下，将
GEMM2 workgroup 从 4 waves 扩展到 8 waves：

```text
tile       = 192x256x256
warps      = 2x4
block size = 256 threads
buffers    = 4
```

每个 wave 的 N 范围由 128 列缩小为 64 列，单 wave accumulator 数量减半，
从而允许每个 physical SIMD 驻留两个同一 workgroup 的 wave。

### TDM 正确性修正

最初的 8-wave 版本仍让每个 wave 发射两个 next-task input TDM 和两个 output
TDM。const0 会隐藏该问题，但 random 下出现少量连续输出块错误。gfx1250
硬件资料规定每个 wave 最多 3 个、每个 SIMD 最多 6 个等待 XACK 的 TDM
operation，因此最终实现采用以下协议：

1. 保留 next task 的 stage 0/stage 1 提前预取。
2. 将两段 output store 分配给不同 wave，使每个 wave 只发一个 output TDM。
3. 同一 SIMD 的两个 resident wave 使用交错的 output phase，均匀分散 TDM
   descriptor 压力。
4. 第一段 output TDM 后执行 `s_wait_tensorcnt 0x2`，再发第二段 output；
   下一 task 开始时执行 `s_wait_tensorcnt 0x1`，确保 I0/I1 已完成，只留下
   前一 task 的 output TDM 与 tile 0 compute 重叠。
5. output row split 保持 `3+3`，即每个 wave 的 6 个 WMMA M block 均分为
   两段。

最终 kernel symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc8_ostore2p_s3_ow2_ps7pf2hm_earlynext_o1w_xor_wait2
```

### random 精确验证

使用额外的 GEMM2 grouped-output 对比，结果为：

```text
GEMM2 grouped max_abs = 0
GEMM2 grouped nonzero = 0
GEMM2 routed max_abs  = 0
GEMM2 routed rel_l2   = 0
GEMM2 reference hash128 = ad17c78eb80b4228cf0b59f190e8af9f
GEMM2 output hash128    = ad17c78eb80b4228cf0b59f190e8af9f
MoE logits_diff          = 3.38491e-06
MoE rel_l2               = 0.00260189
MoE pass                 = True
```

### const0 相邻性能

两次相邻 `w2x2` baseline 的 GEMM2 中位数分别为 `55.006 us` 和
`56.562 us`。8-wave 正确版本的三轮结果为：

```text
52.368, 52.241, 53.195 us
median = 52.368 us
```

| 对比基线 | baseline median | w2x4 median | GEMM2 提升 |
|---|---:|---:|---:|
| 较快相邻 baseline | 55.006 us | 52.368 us | 4.79% |
| 较慢相邻 baseline | 56.562 us | 52.368 us | 7.41% |

该轮 `w2x4` 的 fused MoE 三轮为：

```text
204.41, 203.75, 204.08 us
median = 204.08 us
```

对应日志：

```text
/tmp/w2x2_baseline_after_pf1
/tmp/w2x2_baseline_after_o1w
/tmp/w2x4_o1w_xor_wait2_rounds
```

### 复现命令

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048

ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

### 本轮未保留的变体

下列结果仅用于决策，未计入稳定优化：单级 next-task 预取虽正确但为
`56.490 us`；`MMA_GROUP=2` 为 `53.690 us`；`fence_cover_mma=4` 为
`53.340 us`；output `2+4` split 为 `53.844 us`；`STORE_PAD=8` 为
`53.224 us`；direct global store 单轮约 `109 us`。`w1x8`、`wpt1`、去掉
中间 tensor wait、以及把 output TDM 合并为四个大 descriptor 的版本均未通过
random exact-hash 验证。

## GEMM2 persistent A payload stage 0/1 常驻（2026-10-02）

### 优化动机

`E64/T1536/topk8/M7168/I2048` 在 expert balance 下每个 expert 恰好有
`192` 行，因此一个 persistent workgroup 会依次处理同一 expert 的 7 个 N tile。
这 7 个 task 的 A payload 完全相同，原实现仍会为每个 task 重新执行全部 8 个 K
stage 的 A TDM load。

该版本在原 4-buffer arena 之后增加两个只读 A payload cache slot：

```text
A cache stages = 2
A bytes/stage  = 24,576 B
extra LDS      = 49,152 B
total LDS      = 292,864 B (286 KiB)
```

首个 persistent task 将 K stage 0/1 的 A payload 直接加载到 cache slot；后续 6 个
task 只更新对应 stage 的 B、ScaleA 和 ScaleB，并继续从 cache slot 读取 A。ScaleA
体积较小且继续走原有 ring buffer，避免 LDS 分配跨过本轮实验中不稳定的边界。原有
4-buffer input pipeline、两阶段 output store、`tensor_wait(2)` / `tensor_wait(1)`
协议以及 output TDM wave 分配均保持不变。

kernel symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc8_ostore2p_s3_ow2_ac2payload_ps7pf2hm_earlynext_o1w_xor_wait2
```

### random 精确验证

额外的 GEMM2 stage-output 检查通过：

```text
GEMM2 grouped max_abs = 0
GEMM2 grouped nonzero = 0
GEMM2 routed max_abs  = 0
GEMM2 routed rel_l2   = 0
MoE logits_diff       = 3.38491e-06
MoE rel_l2            = 0.00260189
MoE pass              = True
```

### 相邻性能复测

第一组：

| 版本 | GEMM2 samples (us) | median | 相对原最佳 |
|---|---|---:|---:|
| 原 `w2x4 xor_wait2` | 51.468, 51.904, 51.929 | 51.904 us | baseline |
| A payload stage 0/1 cache | 52.444, 51.130, 50.228 | 51.130 us | +1.49% |

第二组：

| 版本 | GEMM2 samples (us) | median | 相对原最佳 |
|---|---|---:|---:|
| 原 `w2x4 xor_wait2` | 54.166, 54.076, 52.823 | 54.076 us | baseline |
| A payload stage 0/1 cache | 52.716, 52.454, 53.370 | 52.716 us | +2.51% |

两组相邻比较均为正收益，因此保留该实现作为新的 GEMM2 最佳节点。对应日志：

```text
my_code/moe_prefill_switch_ab_runs/20261002T040609Z
my_code/moe_prefill_switch_ab_runs/20261002T044250Z
my_code/moe_prefill_switch_ab_runs/20261002T044452Z
my_code/moe_prefill_switch_ab_runs/20261002T044715Z
```

复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048

ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
FLYDSL_DUMP_IR=0 \
python3 -u .codex_tmp/test_gemm2_diff.py \
  --scenario verify \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

未保留的相邻实验：A payload/ScaleA 同时常驻两个 stage 会使总 LDS 增至
`295,936 B` 并产生错误；常驻三个 A payload stage 虽然正确，但三轮中位数
`52.916 us`，低于两-stage 版本的收益。named barrier、单阶段 output store、
`wpt=4`、A payload `TH=2/6` 和 A/B LDS load 重排均未通过正确性或性能门槛。

## GEMM2 cached-stage TDM owner 均衡（2026-10-02）

### 优化内容

在保留 A payload stage 0/1 常驻的基础上，重新分配后续 persistent task 的
stage 0/1 TDM owner。A payload 已经不再搬运，剩余 B、ScaleA、ScaleB 若继续沿用
原 owner，会让 B 与 ScaleB 同时集中在 physical SIMD2/3，而 SIMD0/1 只承担较小的
ScaleA，造成明显的 barrier 到达偏斜。

新映射为：

```text
B payload : waves 0,1,2,3，四路均分
ScaleA    : waves 4,5
ScaleB    : waves 6,7
```

按 `wave 0/4`、`1/5`、`2/6`、`3/7` 共用 physical SIMD 的映射计算，每个 SIMD
在每个 cached stage 中承担约 `8.75–9.0 KiB`，替代原先约 `0.75 KiB` 对
`17 KiB` 的不均衡分配。同时每个 wave 每个 stage 仍只发一个 input TDM；加上唯一的
output TDM 后，仍满足每 wave 3 个、每 SIMD 6 个等待 XACK 的硬件限制。

最终 kernel symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc8_ostore2p_s3_ow2_ac2payload_balnext_ps7pf2hm_earlynext_o1w_xor_wait2
```

random 精确验证：

```text
GEMM2 grouped max_abs = 0
GEMM2 grouped nonzero = 0
GEMM2 routed max_abs  = 0
GEMM2 routed rel_l2   = 0
MoE logits_diff       = 3.38491e-06
MoE rel_l2            = 0.00260189
MoE pass              = True
```

相邻三轮结果：

| 版本 | GEMM2 samples (us) | median | GEMM2 提升 | fused MoE median |
|---|---|---:|---:|---:|
| 原 `w2x4 xor_wait2` | 54.166, 54.076, 52.823 | 54.076 us | baseline | 205.75 us |
| A-cache 两段，未均衡 owner | 52.716, 52.454, 53.370 | 52.716 us | 2.51% | 205.04 us |
| A-cache 两段 + balanced owner | 50.340, 50.127, 52.414 | 50.340 us | 6.91% | 199.18 us |

对应日志：

```text
my_code/moe_prefill_switch_ab_runs/20261002T044452Z
my_code/moe_prefill_switch_ab_runs/20261002T044715Z
my_code/moe_prefill_switch_ab_runs/20261002T045612Z
```

将 next-task 预取再提前一个 K tile 的版本虽然通过 random exact 校验，但三轮
GEMM2 中位数回退到 `54.548 us`，因此未保留。

## GEMM2 A4 persistent reuse 与 WMMA B operand reuse（2026-10-02）

### 优化内容

在 `ac2payload_balnext` 基础上增加两项互补优化：

1. task 0 将 K stage 4 的 A payload 写入 input ring 的 buffer 0。output LDS 只覆盖
   buffer 2/3，因此 buffer 0 的 A 区域可以跨后续 6 个 persistent N task 保留。后续
   task 回填 stage 4 时不再重复加载 A4。
2. 跳过 A4 后不能直接减少对应 wave 的 TDM 数量，否则会破坏后续
   `s_wait_tensorcnt(n)` 所依赖的 per-wave outstanding 距离。实现将 B4 从两路改为四路，
   让 8 个 wave 在该 stage 仍然各发一个 input TDM。
3. GEMM2 的 snake WMMA 遍历已经让相邻 M row 边界使用相同 B operand。设置
   `wmma_reuse=3`，只启用 `reuseB`，避免同时启用 `reuseA` 带来的额外约束。

最终 kernel symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc8_reuse3_ostore2p_s3_ow2_ac3payload_reuse_balnext_ps7pf2hm_earlynext_o1w_xor_wait2
```

### 正确性

random 的 GEMM2 stage-output 精确比较通过：

```text
GEMM2 grouped max_abs = 0
GEMM2 grouped nonzero = 0
GEMM2 routed max_abs  = 0
GEMM2 routed rel_l2   = 0
MoE logits_diff       = 3.38491e-06
MoE rel_l2            = 0.00260189
MoE pass              = True
```

### 空闲性能复测

复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

| 版本 | GEMM1 samples (us) | GEMM1 median | GEMM2 samples (us) | GEMM2 median | fused MoE median |
|---|---|---:|---|---:|---:|
| `ac2payload_balnext` 基线 | 78.605, 78.668, 78.519 | 78.605 us | 53.207, 53.599, 53.269 | 53.269 us | 207.26 us |
| A4 reuse + `reuseB`，第 1 组 | 77.346, 76.950, 76.625 | 76.950 us | 50.125, 49.543, 50.251 | 50.125 us | 199.38 us |
| A4 reuse + `reuseB`，第 2 组 | 77.534, 77.237, 77.325 | 77.325 us | 49.935, 50.344, 50.382 | 50.344 us | 199.19 us |

两组候选中位数平均为 `50.2345 us`，相对本轮基线 `53.269 us` 提升约 `5.70%`；
fused MoE 中位数平均为 `199.285 us`，相对 `207.26 us` 提升约 `3.85%`。

对应日志：

```text
my_code/moe_prefill_switch_ab_runs/20261002T110816Z  # baseline
my_code/moe_prefill_switch_ab_runs/20261002T113043Z  # reuseB run 1
my_code/moe_prefill_switch_ab_runs/20261002T113629Z  # reuseB run 2
```

### 本轮未保留的相邻实验

- `reuseA` 与 A4 reuse 的两组中位数为 `50.161 us`、`50.732 us`，略慢于且波动大于
  `reuseB`。
- B-major WMMA 遍历的表面中位数为 `49.983 us`，但同轮 GEMM1 降到 `75.114 us`；
  使用 GEMM1 作为频率代理归一化后约回退 `2.2%`，因此恢复 snake traversal。
- 同时复用 A4/A5 的版本虽通过 random 精确校验，但三轮中位数为 `54.479 us`，回退。
- wave-private output LDS 通过 random 精确校验，但取消两阶段 output-store overlap 后中位数
  回退到 `57.036 us`。
- ScaleA/ScaleB direct global load 原型未达到 GEMM2 stage-output 逐元素一致，已删除。


## a07-3 重启后 baseline 与当前最优 GEMM2 复测（2026-10-02）

机器重启后，测试前后均通过 `/data/yanguahe/code/gpu_users.sh` 确认全部 GPU 空闲。当前最优 GEMM2 kernel 文件 SHA256：

```text
482d83a7fd2031a6d8ee1007f3830680b8d65dc89c2d58308b17ded41dd09c32
```

实际 kernel symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc8_reuse3_ostore2p_s3_ow2_ac3payload_reuse_balnext_ps7pf2hm_earlynext_o1w_xor_wait2
```

复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

| 轮次 | GEMM1 samples (us) | GEMM1 median | GEMM2 samples (us) | GEMM2 median | fused MoE samples (us) | fused MoE median |
|---|---|---:|---|---:|---|---:|
| 重启后 baseline | 78.101, 78.025, 78.246 | 78.101 us | 51.463, 51.676, 53.445 | 51.676 us | 207.97, 201.22, 219.03 | 207.97 us |
| 当前最优独立复测 | 78.495, 78.442, 79.136 | 78.495 us | 51.670, 50.908, 51.965 | 51.670 us | 207.84, 200.90, 201.54 | 201.54 us |

两组三轮合并后，GEMM2 的六个样本中位数为 `51.673 us`。两次运行的 const0 GEMM1、GEMM2 和最终 MoE 输出 hash 均与 reference 一致。

日志：

```text
my_code/moe_prefill_switch_ab_runs/20261002T141618Z
my_code/moe_prefill_switch_ab_runs/20261002T142219Z
```


## GEMM1 优化新阶段 baseline（a07-3 重启后，2026-10-02 15:11 UTC）

测试前确认 a07-3 的全部 GPU 空闲。目标规模和复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

| 指标 | samples (us) | median |
|---|---|---:|
| GEMM1 | 77.166, 77.691, 76.337 | 77.166 us |
| GEMM2 | 52.898, 52.647, 52.856 | 52.856 us |
| fused MoE | 200.87, 202.40, 201.99 | 201.99 us |

GEMM1 减少 20% 的目标为：

```text
77.166 us * 0.80 = 61.733 us
```

const0 的 GEMM1、GEMM2 和最终 MoE 输出 hash 均与 reference 一致。日志：

```text
my_code/moe_prefill_switch_ab_runs/20261002T151110Z
```

## GEMM1 persistent x4 + eight-wave latency hiding（2026-10-02 至 2026-10-03）

### 保留的实现

在 `t192x256x256/w2x2/b4` 的基础上，将 GEMM1 改成以下组合：

```text
tile               = 192x256x256
workgroup          = w2x4（8 waves）
input buffers      = 4
persistent tasks   = 4
cluster            = 4x1
B TDM cache hint   = 6 (NT_HT)
WMMA reuse         = reuseB
next-task prefetch = stage 0 + stage 1
output TDM         = one descriptor per wave
```

`N=4096` 一共有 `16` 个 N tile。一个 4-WG cluster 覆盖四个 N tile，persistent
loop 再顺序处理四个 N task，因此每个 expert 的全部 N 方向工作由同一个 cluster
完成。task 间提前发射下一 task 的 stage 0/1，并让当前 output TDM 与下一 task 的
tile 0 compute 重叠。

`w2x4` 把每个 wave 的 accumulator 和 operand register 压低到可在每个 physical
SIMD 同时驻留两个 wave。ATT 中观察到的静态资源为：

```text
VGPR = 377 / wave
SGPR = 97 / wave
LDS  = 243,712 B / workgroup
```

MI450 每个 SIMD 有 1024 个 VGPR，因此两个 wave 合计约 754 个 VGPR，双驻留成立。
相较此前单 wave/SIMD 的版本，这一变化可在一个 wave 等待 TDM、LDS 或 barrier 时让
另一个 wave 继续发射指令。

实际 GEMM1 symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ps4pf2hm_earlynext_o1w_xor_wait1
```

### 性能结果

首次三轮结果：

```text
GEMM1 samples = 70.718, 72.824, 75.558 us
GEMM1 median  = 72.824 us
```

相对本轮重启后 baseline `77.166 us`，中位数降低约 `5.63%`。机器状态随后发生漂移，
因此又对 `reuseB`（`reuse3`）和 A+B 同时 reuse（`reuse1`）做了 ABBA 相邻复测：

| 版本 | 第 1 组三轮 median | 第 2 组三轮 median | 六样本合并 median |
|---|---:|---:|---:|
| `reuseB` (`reuse3`) | 75.796 us | 74.329 us | 75.222 us |
| A+B reuse (`reuse1`) | 76.130 us | 74.656 us | 75.740 us |

两者差异较小，但两组汇总后 `reuseB` 仍约快 `0.68%`，因此保留 `reuse3`，删除
`reuse1` 实验代码。

对应日志：

```text
my_code/moe_prefill_switch_ab_runs/20261002T171823Z
my_code/moe_prefill_switch_ab_runs/20261002T185016Z
my_code/moe_prefill_switch_ab_runs/20261002T190215Z
my_code/moe_prefill_switch_ab_runs/20261002T190357Z
my_code/moe_prefill_switch_ab_runs/20261002T190514Z
my_code/moe_prefill_switch_ab_runs/20261002T190628Z
```

### random 正确性

恢复最终 `reuseB` 文件后执行：

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
logits_diff = 3.38491e-06
rel_l2      = 0.00260189
pass        = True
```

该次 random 运行时机器上存在其他 GPU workload，因此这里只采用正确性结论，不采用其
耗时。日志：

```text
my_code/moe_prefill_switch_ab_runs/20261003T031720Z
```

### 当前 trace 结论

`w2x4` 版本的代表 full-wave span 约为 `119k–121k cycles`。不同 SIMD/slot 的主要
exposed stall 分布为：

- `s_wait_dscnt`：约 `17.8%–25.7%`；
- `s_wait_tensorcnt`：最高约 `17.0%`；
- `s_barrier_wait`：约 `11.2%–29.0%`。

因此当前瓶颈已经从固定 prologue/epilogue 转移到 K-loop 内的 LDS completion、B TDM
latency，以及两个 resident waves 到达 workgroup barrier 的偏斜。

### 本阶段已淘汰的实验

以下实验均已恢复，不保留代码：`reuseA+reuseB`、5-buffer pipeline、A stage 0/1
resident cache、`coexec`/`max-ilp`/`iterative-maxocc` scheduler、`fc12`、split/raw
barrier、B L2 software prefetch、B-first owner、balanced TDM owner、`w3x4`、`w4x4`、
`t96/b2`、`t192x128/b3`、`cluster_n=8` 和 `wpt4`。其中 `w3x4`、`w4x4`、`wpt4`
的实验版本还出现 random 精度错误，不能保留。

## 2026-10-03 重启后的 GEMM1 baseline、最优版与 invalid-block trace

a07-3 再次重启后，在 GPU 空闲状态下重新测量本阶段 baseline 和当前保留的最优版本：

| version | GEMM1 samples (us) | GEMM1 median | GEMM2 median | fused MoE median | correctness |
|---|---|---:|---:|---:|---|
| phase baseline (`t192/w2x2/b4 + B_TH=6`) | 78.128, 77.881, 79.720 | 78.128 us | 53.418 us | 208.05 us | const0 hashes match |
| retained best (`persistent x4 + t192/w2x4/b4 + B_TH=6 + reuseB`) | 75.131, 74.825, 74.104 | 74.825 us | 51.469 us | 199.53 us | const0 hashes match |

当前最优 GEMM1 相对同机 baseline 降低 `4.23%`。日志目录：

```text
baseline: my_code/moe_prefill_switch_ab_runs/20261003T054059Z
best:     my_code/moe_prefill_switch_ab_runs/20261003T054229Z
```

随后对当前最优 GEMM1 抓取单次 SIMD3 ATT。cycle 总量最大的 decoded SIMD slot
为 `SE1/SIMD3/slot1`，其中一个有效 block 消耗 `126149 cycles`，三个二分查找后
early-exit 的无效 block 合计消耗 `3267 cycles`。无效 block 占该 slot 累计 block
cycles 的比例为：

```text
3267 / 129416 = 2.524417%
```

完整方法、trace 路径和 physical-WGP 口径的交叉检查见
`my_code/e64_t1536_topk8_gemm1_invalid_block_trace_analysis_a07_3.md`。

### 当前最优 GEMM1 的 wait-cycle 瓶颈

同一份重启后 trace 的 8 条有效 compute wave 合计 `994047 cycles`。按 decoder
instruction latency 统计：

```text
s_wait_dscnt      = 222278 cycles = 22.360915%
s_wait_tensorcnt  =  15224 cycles =  1.531517%
s_barrier_wait    = 185600 cycles = 18.671149%
s_wait_kmcnt      =  24097 cycles =  2.424131%
```

其中 `s_wait_dscnt` 的 exposed stall 为 `21.682476%`，`s_wait_tensorcnt` 的
exposed stall 为 `1.437357%`。当前主要瓶颈是 K-loop 的 LDS operand completion
以及两个 resident waves 到达 workgroup barrier 的偏斜；TDM completion 已不是第一
瓶颈。完整 PC、immediate 和 per-wave 分析见
`my_code/e64_t1536_topk8_gemm1_best_thread_trace_bottleneck_a07_3.md`。

### Trace 抓取方法与统计口径

当前最优 GEMM1 的完整 symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ps4pf2hm_earlynext_o1w_xor_wait1
```

为减少 trace 次数，本次只抓取 ATT SIMD selector 3。decoder 输出了四个 shader
engine 的 occupancy 数据，并解码了每个 shader engine 上 SIMD3 的两个 slot。

```bash
cd /data/yanguahe/code/wk_sp1/aiter

REPO_ROOT=/data/yanguahe/code/wk_sp1/aiter \
TRACE_ROOT=my_code/thread_trace_runs \
bash /tmp/get_isa_runner_att.sh \
  a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ps4pf2hm_earlynext_o1w_xor_wait1 \
  e64_t1536_topk8_gemm1_best_invalid_blocks_a07_3_20261003 \
  "env AITER_MOE_EXPERT_BALANCE=true AITER_LOG_MORE=1 AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --iters 2 --const-init 0" \
  --ana-att
```

trace 目录：

```text
my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_invalid_blocks_a07_3_20261003
```

`analyze_att_capture.py` 给出的 REALTIME 加权平均 GFXCLK 为 `1893.745 MHz`，
最大 occupancy timestamp span 为 `132579 cycles`。

`trace_segment_cycles.py` 以 kernel 第一条 `global_prefetch_b8` 到 `s_endpgm`
作为完整区间，8 条有效 compute wave 的结果为：

```text
count            = 8
average          = 120636.4 cycles
p50              = 120696.0 cycles
p90              = 122250.2 cycles
minimum          = 118678 cycles
maximum          = 122512 cycles
```

wait 占比使用更完整的 `wave.begin -> wave.end` duration 作为分母。8 条有效 wave
的总 span 为 `994047 cycles`，平均每条 wave 为 `124255.875 cycles`。该口径是
active-wave cycle 的加权统计；不能把多个并发 wave 的 wait cycles 相加后除以一次
dispatch wall time。

### 无效 block 的完整统计

decoder 文件名采用 `seX_smY_slZ_wvN.json`。固定 `SE/SM/SL` 后，`wvN`
表示依次占用该 SIMD slot 的 block-wave 实例。

分类直接依据动态指令：

- 有效 block 至少执行一条 `v_wmma*`；
- 无效 early-exit block 没有执行任何 `v_wmma*`。

两类 trace 完全分离：

| 类型 | 解码 block 数 | 每个 block 的动态指令数 | duration 范围 |
|---|---:|---:|---:|
| 有效计算 block | 8 | 15,127-15,286 | 122,239-126,149 cycles |
| 无效 early-exit block | 22 | 固定为 238 | 907-1,287 cycles |

无效路径末尾为 persistent-loop 的 scalar 控制流，随后直接 drain 并退出：

```text
s_add_co_i32 ...
s_cmp_lg_u32 ...
s_cbranch_*
s_wait_tensorcnt 0x0
s_endpgm
```

该路径没有进入 GEMM hotloop，因此短 trace 确实是 expert id 无效后直接返回的
block，而不是执行异常快的有效 block。

对每个 `SE/SM/SL` 内所有 `wvN` 的 duration 求和，cycle 总量最大的是
`SE1/SIMD3/slot1`：

| block instance | cycles | 动态指令数 | `v_wmma*` 数量 | 分类 |
|---|---:|---:|---:|---|
| `wv0` | 126,149 | 15,286 | 2,688 | 有效计算 |
| `wv1` | 1,241 | 238 | 0 | 无效 early-exit |
| `wv2` | 1,011 | 238 | 0 | 无效 early-exit |
| `wv3` | 1,015 | 238 | 0 | 无效 early-exit |

```text
slot 总 cycles    = 126149 + 1241 + 1011 + 1015
                  = 129416 cycles

无效 block cycles = 1241 + 1011 + 1015
                  = 3267 cycles

无效 block 占比   = 3267 / 129416
                  = 2.524417%
```

按本次 GFXCLK，`3267 cycles` 相当于约 `1.725 us` 的累计 slot residency。
这是该 slot 上的理想局部上限；不同 WGP 的无效 block 会与其他工作并行，不能直接
推导为整个 kernel 可以降低 `1.725 us`。

全部 decoded SIMD slot：

| 排名 | decoded SIMD slot | blocks | 有效 | 无效 | 总 cycles | 无效 cycles | 无效占比 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | SE1/SIMD3/slot1 | 4 | 1 | 3 | 129,416 | 3,267 | 2.524417% |
| 2 | SE1/SIMD3/slot0 | 4 | 1 | 3 | 128,509 | 2,963 | 2.305675% |
| 3 | SE0/SIMD3/slot1 | 3 | 1 | 2 | 128,145 | 2,301 | 1.795622% |
| 4 | SE0/SIMD3/slot0 | 3 | 1 | 2 | 127,436 | 2,106 | 1.652594% |
| 5 | SE3/SIMD3/slot1 | 4 | 1 | 3 | 126,657 | 3,314 | 2.616515% |
| 6 | SE2/SIMD3/slot1 | 4 | 1 | 3 | 125,917 | 3,061 | 2.430966% |
| 7 | SE3/SIMD3/slot0 | 4 | 1 | 3 | 125,756 | 3,016 | 2.398295% |
| 8 | SE2/SIMD3/slot0 | 4 | 1 | 3 | 124,978 | 2,739 | 2.191586% |

`occupancy.json` 的 `packed_sa_wgp` 可用于更严格的
`SE/SA/WGP/SIMD/slot` 物理坐标交叉检查：

- 累计 resident cycles 最大的物理 slot 是
  `SE2/SA1/WGP2/SIMD0/slot0`，只运行一个有效 block，共 `132537 cycles`，
  无效 block 占比为 `0%`。
- 在确实被多个 block 复用的物理 slot 中，cycle 总量最大的是
  `SE1/SA1/WGP2/SIMD0/slot1`，一个有效 block 加一个无效 block 共
  `129874 cycles`；无效 block 为 `1038 cycles`，占 `0.799236%`。

按 decoder `SE/SM/SL` slot 时间线，问题所问的主要结果是 `2.524417%`。
physical-WGP 结果进一步说明，无效 block 不是当前 kernel 的主要性能瓶颈。

### 全部 `s_wait_tensorcnt` 和 `s_wait_dscnt` 占比

| wait 类型 | 动态次数 | decoder latency cycles | 占完整 wave span | exposed stall cycles | stall 占完整 wave span | 单 wave占比 min / median / max |
|---|---:|---:|---:|---:|---:|---:|
| `s_wait_tensorcnt` | 936 | 15,224 | **1.531517%** | 14,288 | **1.437357%** | 0.839339% / 1.486696% / 2.783217% |
| `s_wait_dscnt` | 6,744 | 222,278 | **22.360915%** | 215,534 | **21.682476%** | 19.583260% / 22.345841% / 24.946279% |

两类 wait 合计：

```text
decoder latency = 237502 / 994047 = 23.892432%
exposed stall   = 229822 / 994047 = 23.119832%
```

`s_wait_tensorcnt` 的 immediate 构成：

| instruction | count | latency cycles | 占完整 wave span | exposed stall |
|---|---:|---:|---:|---:|
| `s_wait_tensorcnt 0x2` | 808 | 7,024 | 0.7066% | 6,216 |
| `s_wait_tensorcnt 0x3` | 8 | 5,324 | 0.5356% | 5,316 |
| `s_wait_tensorcnt 0x0` | 64 | 2,539 | 0.2554% | 2,475 |
| `s_wait_tensorcnt 0x1` | 56 | 337 | 0.0339% | 281 |

`s_wait_dscnt` 的主要构成：

| instruction | count | latency cycles | 占完整 wave span | exposed stall |
|---|---:|---:|---:|---:|
| `s_wait_dscnt 0x0` | 1,016 | 118,246 | 11.8954% | 117,230 |
| `s_wait_dscnt 0xa` | 960 | 86,043 | 8.6558% | 85,083 |
| `s_wait_dscnt 0x8` | 952 | 4,591 | 0.4618% | 3,639 |
| `s_wait_dscnt 0x6` | 64 | 4,112 | 0.4137% | 4,048 |
| `s_wait_dscnt 0x1d` | 864 | 3,360 | 0.3380% | 2,496 |
| `s_wait_dscnt 0x5` | 32 | 2,318 | 0.2332% | 2,286 |
| 其余 immediate 合计 | 2,856 | 3,608 | 0.3630% | 1,752 |

`s_wait_dscnt 0x0` 和 `s_wait_dscnt 0xa` 合计占完整 wave span 的
`20.5512%`，占全部 DScnt wait latency 的 `91.91%`。它们分别集中在 stage
handoff/buffer 复用前的完整 LDS drain，以及 WMMA 消费下一批 operands 前的部分
LDS-read completion 等待。

### 其他主要 cycle 项

| opcode | decoder latency cycles | 占完整 wave span | exposed stall 占比 | 说明 |
|---|---:|---:|---:|---|
| `v_wmma_scale_f32_32x16x128_f4` | 330,660 | 33.2640% | 13.7945% | GEMM 有效计算主体 |
| `s_wait_dscnt` | 222,278 | 22.3609% | 21.6825% | LDS completion 主瓶颈 |
| `s_barrier_wait` | 185,600 | 18.6711% | 18.5746% | workgroup wave 到达偏斜 |
| `ds_load_b128` | 61,814 | 6.2184% | 2.6130% | A/B operand 从 LDS 进入 VGPR |
| `s_wait_kmcnt` | 24,097 | 2.4241% | 2.4169% | prologue scalar load / expert lookup |
| `s_wait_tensorcnt` | 15,224 | 1.5315% | 1.4374% | TDM load/store completion |
| `v_add_nc_u32_e32` | 12,578 | 1.2653% | 0.6086% | 地址更新 |
| `v_tanh_f32_e32` | 12,321 | 1.2395% | 0.6214% | SiLU epilogue |

所有 `s_barrier_wait` 合计占 `18.671149%`。`s_wait_dscnt + s_barrier_wait`
合计占 `41.032064%`；再加上 `s_wait_tensorcnt` 后占 `42.563581%`。

最重的静态 wait 位置：

| PC | instruction | latency/span | 作用 |
|---|---|---:|---|
| `0x36e0` | `s_barrier_wait 0xffff` | 9.9657% | steady K-loop stage handoff |
| `0x4000` | `s_wait_dscnt 0x0` | 5.4900% | 下一 stage 可读或复用前完整 LDS drain |
| `0x36d8` | `s_wait_dscnt 0x0` | 5.0805% | 下一 stage 可读或复用前完整 LDS drain |
| `0x3e28` | `s_wait_dscnt 0xa` | 4.1319% | WMMA 前等待部分 LDS operands |
| `0x4008` | `s_barrier_wait 0xffff` | 3.7900% | steady K-loop stage handoff |
| `0x3508` | `s_wait_dscnt 0xa` | 2.8735% | WMMA 前等待部分 LDS operands |

前四个主要 DScnt wait PC 合计贡献全部 DScnt wait latency 的 `78.60%`；前两个
主要 hotloop barrier PC 合计贡献全部 barrier wait latency 的 `73.67%`。

### 两个 resident slot 的不对称

| slot | `s_wait_tensorcnt` | `s_wait_dscnt` | `s_barrier_wait` |
|---|---:|---:|---:|
| slot0，四个 SE 加权 | 1.1737% | 20.9446% | 27.5693% |
| slot1，四个 SE 加权 | 1.8876% | 23.7706% | 9.8147% |

slot1 在 LDS/TDM 数据就绪上更慢，而 slot0 更早到达 barrier，并把差值暴露为较长的
`s_barrier_wait`。由于 barrier 是全 workgroup 同步，不能只根据单个 slot 断言唯一
straggler；但这一稳定不对称说明两个 resident waves 的 operand readiness 和推进速度
没有平衡。

### 当前瓶颈与优化优先级

当前最优 GEMM1 的第一瓶颈是 K-loop 的 LDS operand pipeline：

1. 每个 k128 都需要从 LDS 读取 A、B、ScaleA 和 ScaleB 到 VGPR。
2. `s_wait_dscnt 0xa` 在 WMMA 前等待所需 operands。
3. `s_wait_dscnt 0x0` 在 stage 交接或 buffer 复用前完全 drain LDS 操作。
4. wave 之间的数据就绪时间不同，最终在全 workgroup barrier 上暴露为额外等待。

`s_wait_tensorcnt` 只有 `1.53%`，说明 persistent prefetch 已经隐藏绝大部分 TDM
global-memory latency。继续单纯增加 TDM prefetch distance、增加 inflight TDM 数或
只调整 B cache hint，预计不会形成大幅收益，并且可能碰到 MI400 每 wave 3 个、每
SIMD 6 个等待 XACK 的 TDM 限制。

`s_wait_kmcnt` 仍占 `2.42%`，主要来自 expert lookup 的 scalar loads。当前 persistent
版本已经让同一 block 的 4 个 N task 共用一次 expert lookup，因此这里剩余的优化空间
明显小于 LDS/barrier 路径。

后续应优先：

1. 重排每个 k128 的 LDS reads 与 WMMA，让下一批 operands 更早进入 VGPR，并在
   `s_wait_dscnt 0xa` 之前放置更多独立 WMMA 或地址计算。
2. 根据 LDS buffer 的真实读写集合缩小 stage handoff 的 drain 范围，减少完整
   `s_wait_dscnt 0x0`，但不能直接删除正确性所需的 barrier。
3. 重新平衡两个 resident slot 的 LDS/TDM owner 工作，降低 barrier arrival skew。
4. 如果工具链提供正式入口，在 code-object descriptor 层测试
   `.amdhsa_round_robin_scheduling 1`。此前有问题的 in-kernel
   `s_setprio_inc_wg` 实验不再使用。

硬件依据：

- `MI400_Shader_Programming#65.txt` §4.3.7.2.4：LDS 操作由 DScnt 跟踪，read
  完成表示结果可以从 VGPR 使用。
- 同文档 §4.10.1：TDM completion 由 TENSORcnt 跟踪；同一 wave 的 tensor
  instructions 保持顺序。
- 同文档 §4.10.8：每 wave 最多 3 个、每 SIMD 最多 6 个等待 XACK 的 TDM。
- 同文档 §5.2.2：workgroup equal-priority scheduling 用于让协作 waves 更同步地
  到达 barrier。

复现分析：

```bash
python3 my_code/analyze_gemm_invalid_block_trace.py <ui_output_dir>
python3 my_code/analyze_gemm_invalid_blocks.py <ui_output_dir>/occupancy.json
python3 my_code/analyze_gemm_wait_cycles.py <ui_output_dir> --top 20
```

相关日志：

```text
my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_invalid_blocks_a07_3_20261003/logs/
  analyze_att_capture.log
  trace_segment_full_valid_waves.log
  invalid_block_wave_trace_analysis.log
  invalid_block_occupancy_analysis.log
  wait_cycle_analysis.log
  wait_cycle_aggregate.log
  wait_immediate_summary.log
  per_wave_wait_share.log
  top_wait_events.log
  top_wait_pc_context.log
```

## 2026-10-03：GEMM1 LDS segment-aware layout

### 硬件结论

根据本地 `MI400_Shader_Programming#65.txt` §4.7.1、§5.3.6、§5.7.6 和
`architecture.txt` §2.12：

- LDS 为 `64 banks × 4 B`，按 `64 KiB segment` 划分；单 workgroup 最多分配
  `320 KiB`。
- 两个 SIMD-pair port 同时访问同一个 segment 时会发生 secondary segment
  conflict，只允许 priority port 访问 RAM；访问不同 segment 时可以并行访问相同
  bank。
- 每个 SIMD 每周期最多发射一条 LDS/VMEM 指令，所以同一个 SIMD 上的两条 resident
  wave 并不是两个独立 LDS port。真正需要优先分离的是两个 SIMD-pair port。
- `DS_LOAD_B128` 的理想 independent repeat rate 已是 2 cycles，dependent latency
  约 58 cycles。当前 lane pattern 每个 16-lane half 已覆盖全部 64 banks，小 padding
  无法突破这个固有下限。

### 实施方案

仅对 E64/T1536/topk8/M7168/I2048 的 GEMM1 `t192x256x256/w2x4/b4`、
`cluster_m=1` 路径启用五段 LDS：

```text
segment 0: A wave_m=0 + ScaleA wave_m=0 + ScaleB port0
segment 1: A wave_m=1 + ScaleA wave_m=1 + ScaleB port1
segment 2: B port0
segment 3: B port1
segment 4: fused SiLU output
```

四个 stage 在各自 segment 内使用固定 stride：

```text
A_HALF  = 0x3000
B_HALF  = 0x4000
SA_HALF = 0x0300
SB_HALF = 0x0400
```

同时将逻辑 M wave 映射改为：

```text
physical_wave_slot = wave // 4
port               = wave_n // 2
wave_m             = physical_wave_slot ^ port
```

这样同一 resident-slot phase 下，两个 SIMD-pair port 的 A/ScaleA 访问落到不同
segment；B/ScaleB 本身按 port 分到不同 segment。外部 tensor layout、TDM 数量、
四级 ring、四个 persistent N task 和 barrier 协议保持不变。GEMM1 在修改前后均为
`A_CACHE_STAGES=0`；两级 A cache 属于 GEMM2 persistent schedule。

最终 kernel symbol：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ldsseg5_xorm_ps4pf2hm_earlynext_o1w_xor_wait1
```

### 相邻性能复测

复现命令：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048
```

GPU 测试前后均为空闲。相邻结果：

| 版本 | GEMM1 samples (us) | median (us) | 相对旧最优 |
|---|---|---:|---:|
| 旧最优 persistent linear layout | 75.835, 75.903, 74.374 | 75.835 | baseline |
| 5-segment + `wave_m` remap | 73.098, 74.648, 73.922 | 73.922 | **+2.52%** |

日志：

```text
旧最优: my_code/moe_prefill_switch_ab_runs/20261003T090157Z
新版本: my_code/moe_prefill_switch_ab_runs/20261003T092205Z
```

random 验证：

```text
logits_diff = 3.38491e-06
rel_l2      = 0.00260189
pass        = True
```

### 新旧 trace 对比

新 trace：

```text
my_code/thread_trace_runs/e64_t1536_topk8_gemm1_ldsseg5_split_xorm_a07_3_20261003
```

该 trace 来自功能等价、仅保留实验版本后缀的
`_ldsseg5_split_xorm_v8` symbol；最终清理后的 symbol 使用 `_ldsseg5_xorm`。

| 指标 | 旧最优 | 5-segment | 变化 |
|---|---:|---:|---:|
| 平均完整 wave interval | 120,636.4 | 113,773.1 cycles | -5.69% |
| 8 条有效 wave 总 span | 994,047 | 940,935 cycles | -5.34% |
| `s_wait_dscnt` latency | 222,278 (22.3609%) | 89,946 (9.5592%) | -59.53% cycles |
| `s_wait_dscnt` exposed stall | 215,534 (21.6825%) | 83,250 (8.8476%) | -61.37% cycles |
| `s_barrier_wait` latency | 185,600 (18.6711%) | 284,239 (30.2081%) | +53.15% cycles |
| `s_wait_tensorcnt` latency | 15,224 (1.5315%) | 28,310 (3.0087%) | +85.96% cycles |
| DScnt + barrier latency | 407,878 (41.0321%) | 374,185 (39.7674%) | -8.26% cycles |

`s_wait_dscnt` 的绝对 cycles 减少约 59.5%，说明 segment 重新布局确实消除了大量
跨 port 串行。但更快的 waves 随后在 workgroup barrier 等待慢 wave，导致
`s_barrier_wait` 增加约 53.1%；`s_wait_tensorcnt` 的增加来自 issue/arrival timing
变化，而不是 TDM 数量变化。因此最终 wall-time 收益只有约 2.5%，远小于 DScnt 的
局部下降幅度。

新版本两个 resident slot 的等待仍不平衡：slot0 的 barrier share 约 38.4%-41.1%，
slot1 约 19.2%-22.0%；slot1 的 DScnt share 约 10.4%-11.5%，slot0 约
7.4%-8.9%。后续优化应针对 wave arrival skew 和 barrier 前调度，继续增加 LDS padding
或再次整体搬迁数据预计收益有限。

### 未保留实验

- 实验性启用两级 A cache，并让 output 覆盖 B stage 2/3：三轮中位数 `73.321 us`，慢于
  无 cache segment 版，原因是 output TDM 与 next-task B prefetch 重新争用 segment
  2/3。
- 只保留一级 A cache，并让 ScaleB 与 output 在 segment 4 按生命周期复用：random
  出现 `logits_diff=0.16932`、`rel_l2=0.581853`，未通过正确性，已删除。
- 把 output-store 选择条件从 `store_phase ^ wave_m` 改成
  `store_phase ^ physical_wave_slot` 会漏写中间两个 48-row slice，已撤销。

完整设计说明见：

```text
my_code/e64_t1536_topk8_gemm1_lds_segment_layout_design.md
```

### `hyg/moe_a4w4_pr_refactor` v123 cyclic packing 移植结果

refactor v123 的核心布局也移植到了当前 BF16-output GEMM1：每个 ring stage 固定占用
一个 `64 KiB segment`，`wave_m=0` 的 A/ScaleA 放在当前 segment 前部，
`wave_m=1` 放在下一 segment 尾部并在 stage 3 环回 segment 0；B/ScaleB 位于当前
stage 中部。总 LDS 从 `320 KiB` 降为 `256 KiB`。

移植时发现必须同步修改 B 的 producer 和 consumer base：

```text
old: STAGE_A
new: B_OFF = A_OWNER_BYTES + SA_OWNER_BYTES
```

遗漏任一处会导致 random 输出 NaN。修正后两种版本均通过 random，误差保持：

```text
logits_diff = 3.38491e-06
rel_l2      = 0.00260189
pass        = True
```

空闲 GPU 三轮结果：

| 版本 | GEMM1 samples (us) | median (us) | 相对当前 5-segment |
|---|---|---:|---:|
| 当前 5-segment 最优 | 73.098, 74.648, 73.922 | **73.922** | baseline |
| cyclic + `32 B / 1 KiB` padding | 74.924, 74.243, 74.783 | **74.783** | -1.16% |
| cyclic、无 padding | 73.604, 75.236, 74.947 | **74.947** | -1.39% |

日志：

```text
5-segment:    my_code/moe_prefill_switch_ab_runs/20261003T092205Z
cyclic+p32:   my_code/moe_prefill_switch_ab_runs/20261003T142356Z
cyclic:       my_code/moe_prefill_switch_ab_runs/20261003T143049Z
```

结论：v123 cyclic packing 在 refactor 的 fused-quant GEMM1 上相对 v115 有 `5.19%`
收益，但在当前 BF16-output GEMM1 上没有超过现有 5-segment layout。当前 kernel 的
output staging 为 `52,224 B`，epilogue/barrier 行为也与 compact fused-quant output
不同；减少一个 LDS segment 带来的收益不足以抵消 B/ScaleB 仍在同一 stage segment
以及 cyclic 地址计算的代价。因此 cyclic 版本未保留，当前源码和 a07-3 均恢复到
5-segment `_ldsseg5_xorm` 版本。

## 当前最终最优 GEMM1 及资源使用

截至本轮测试，性能最好的 GEMM1 是保留的 5-segment LDS layout 版本：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ldsseg5_xorm_ps4pf2hm_earlynext_o1w_xor_wait1
```

空闲 GPU 三轮性能为：

```text
73.098, 74.648, 73.922 us
median = 73.922 us
```

对应日志：

```text
my_code/moe_prefill_switch_ab_runs/20261003T092205Z
```

资源数据从 a07-3 上 exact-symbol ATT capture 的实际 code object ID 11 中读取。四个
SIMD capture 使用的 code object SHA256 均为：

```text
ba5529a95889a6b34ade2e3c37d14775ce33ee7c019b665ca97985c36e0865d1
```

AMDGPU code object metadata：

```yaml
.cluster_dims: [4, 1, 1]
.group_segment_fixed_size: 327680
.private_segment_fixed_size: 0
.kernarg_segment_size: 176
.max_flat_workgroup_size: 256
.reqd_workgroup_size: [256, 1, 1]
.sgpr_count: 98
.sgpr_spill_count: 0
.vgpr_count: 370
.vgpr_spill_count: 0
.wavefront_size: 32
```

换算结果：

| 资源 | metadata 原值 | 字节换算 | 作用域 |
|---|---:|---:|---|
| LDS | 327,680 B | **320 KiB** | 每个 workgroup |
| SGPR | 98 个 32-bit SGPR | **392 B** | 每个 wave，共享于 wave 的 32 lanes |
| VGPR | 370 个 32-bit VGPR/lane | **1,480 B/lane；47,360 B，即 46.25 KiB/wave** | 每个 wave |
| scratch/private segment | 0 B | **0 B** | 每个 work-item |

当前一个 workgroup 有 `256 / 32 = 8 waves`。按未考虑硬件分配粒度取整的逻辑数量：

```text
SGPR/workgroup = 98 × 4 B × 8 = 3,136 B = 3.0625 KiB
VGPR/workgroup = 370 × 32 lanes × 4 B × 8 = 378,880 B = 370 KiB
VGPR/SIMD      = 370 × 2 resident waves = 740 VGPR entries
```

这里的字节换算用于说明寄存器状态规模；实际 register-file 分配仍由硬件按其分配粒度
取整。ATT occupancy 直接确认该 kernel 为：

```text
1 resident workgroup / WGP
8 resident waves / WGP
2 resident waves / SIMD
```

320 KiB LDS 已占满 gfx1250 对单个 workgroup 可分配的 5 个 64 KiB LDS segment，因而
只能驻留一个 workgroup。该 workgroup 的 8 个 wave 均匀分布到 4 个 SIMD，每个 SIMD
同时驻留两个 wave。`private_segment_fixed_size=0`、`sgpr_spill_count=0` 和
`vgpr_spill_count=0` 表明最终版本没有 scratch/private allocation，也没有 SGPR/VGPR
spill。

本文前面记录的 `VGPR=377`、`SGPR=97`、`LDS=243,712 B` 属于较早的 persistent
`w2x4` 实验版本；当前最终 5-segment `_ldsseg5_xorm` 版本应以上述 code object
metadata 为准。
