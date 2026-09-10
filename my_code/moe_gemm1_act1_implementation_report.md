# MoE GEMM1 ABpreShuffle 256×256 汇编接入报告

## 1. 任务结果

已完成 `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1.s` 的实现、standalone runner 支持、stage1 quant A/ScaleA preshuffle、C++ cluster launch，以及 `op_tests/test_flydsl_grouped_gemm_gfx1250.py` 的 e2e injection。

最终代码已同步到 `a07-3`：

```text
yanguahe@heliosr-1b114-a07-3.mnb.dcgpu
```

测试环境：

```text
仓库：/data/yanguahe/code/wk_sp1/aiter
容器：hyg_fyd1
分支：hyg_gfx1250_gemm_a4w4
HEAD：1ac3a46ed1359994d4bd73de29bcb566d45e07ce
GPU：1 x gfx1250
```

最终汇编 SHA256：

```text
3f40709bc179544c163b2fd80e8cffc253c261646caf659340ecaad5cfb0d681
```

本地与 `a07-3` 上的任务文件 SHA256 已逐文件核对一致。

## 2. Production ISA dump

在 `a07-3` 的 `hyg_fyd1` 容器内，从仓库根目录运行：

```bash
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=1 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --iters 20 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --const-init 0
```

Production gemm1 symbol：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_wpt1
```

该次 dump 的参考结果：

```text
gemm1 = 666.684 us
e2e   = 1588.37 us
```

本地保存的最终 ISA dump：

```text
.codex_tmp/moe_gemm1_act1/gemm1_21_final_isa.s
```

Production launch geometry：

```text
grid    = (2880, 4, 1)
cluster = (4, 4, 1)
block   = (128, 1, 1)
```

## 3. 主要代码改动

### 3.1 纯汇编 kernel

文件：

```text
aiter/my_code/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1.s
```

改动内容：

- 对齐 production 184-byte kernarg ABI。
- 使用 `grid=(2880,4,1)`、`cluster=(4,4,1)`、`block=(128,1,1)`。
- 对齐 production 的 cluster-granular DeepGEMM 16-M-tile swizzle。
- 支持 expert-local C/A/B/ScaleA/ScaleB 地址 rebasing。
- A payload 使用 16×16 `ABpreShuffle` 布局。
- ScaleA 使用 32×4 `ABpreShuffle` 布局。
- 实现 FP32 `Silu(gate) * up`。
- 将 activation 结果转换为 BF16，并写入 `[M, 3072]` 输出。
- `swiglu_limit` 固定使用 production 默认值 `7.0`。
- 清除 8 处 input TDM descriptor 的 `workgroup_mask` 和 `early_timeout`，使每个 WG 独立完成 input TDM load。
- 在第一次 `tensor_store_from_lds` 后加入：

```asm
s_wait_tensorcnt 0x0
```

该等待保证第一次 TDM store 已完成读取 LDS，之后才允许第二组 accumulator 覆盖同一块 LDS。

### 3.2 Standalone runner

文件：

```text
aiter/my_code/isa_runner/gemm_batch_isa_runner.py
```

改动内容：

- 新增 `MOE_ACT1_256_KERNEL_SYMBOL` 和 `MOE_ACT1_256_PROFILE`。
- 新增 `moe-act1-256` mode。
- 支持 184-byte production ABI packing。
- 支持 4×4 cluster launch geometry。
- 支持用户要求的 random 和 `--const-init 0` 输入。
- A payload 使用 `shuffle_weight_f4`。
- ScaleA 使用 `shuffle_scale_f4(..., 7)`。
- descriptor shape/stride 对齐 production 的 `torch.int32` view。
- 默认 `swiglu_limit=7.0`。
- 删除 direct epilogue 已不再需要的约 1.2 GB raw-GEMM scratch。
- 增加 source contract 校验。
- const 测试在汇总表中显示为 `const(0)`。

### 3.3 Stage1 quant ABpreShuffle

文件：

```text
aiter/aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py
aiter/aiter/ops/flydsl/moe_kernels.py
aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py
```

新增 `a_preshuffle: bool = False`。

A payload 目标布局：

```text
[row // 16, packed_k // 16, row % 16, packed_k % 16]
```

ScaleA 目标布局：

```text
[row // 32, k_scale // 4, row % 32, k_scale % 4]
```

当：

```text
AITER_MOE_GEMM1_LAUNCH_BACKEND=cpp
```

且 stage1 为 FP4 时，`grouped_moe_gfx1250.py` 启用 `a_preshuffle=True`。

### 3.4 C++ injection 与 cluster launch

文件：

```text
aiter/my_code/isa_runner/moe_cpp_backend.py
aiter/my_code/isa_runner/moe_gemm1_cpp_launcher.cpp
```

改动内容：

- injection target 改为新汇编 symbol。
- contract 改为 `E=96, tokens=16384, topk=6, N=6144, K=7168`。
- A Scale WMMA repeat 改为 `8`。
- launch geometry 改为 `grid=(2880,4,1)`、`cluster=(4,4,1)`。
- 使用 `hipDrvLaunchKernelEx` 和 `hipLaunchAttributeClusterDimension` 启动真正的 4×4 workgroup cluster。
- 检查 balanced `psum=[1024,2048,...,98304]`。
- 检查输出 padding 未被汇编 kernel 写入。

### 3.5 e2e 单测接入

文件：

```text
aiter/op_tests/test_flydsl_grouped_gemm_gfx1250.py
```

改动内容：

- profiler parser 能识别：

```text
moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1
```

- 能在 precision summary 中单独报告新 gemm1 的时间、TFLOP/s 和有效带宽。
- 通过以下环境变量启用 injection：

```text
AITER_MOE_GEMM1_LAUNCH_BACKEND=cpp
```

## 4. 问题定位与修复

### 4.1 Input TDM multicast 非确定性

旧 kernel 的 input descriptor 设置了 `workgroup_mask` 和 bit 21 `early_timeout`。

本地硬件文档说明：

- `D#.workgroup_mask != 0` 时，`TENSOR_LOAD_TO_LDS` 使用 `CLUSTER_LOAD_ASYNC`。
- Group 1 descriptor bit 21 为 `early_timeout`。

修改 persistent scheduler 后，旧 multicast protocol 会出现少量 tile 随机损坏。最终方案保留 4×4 launch/swizzle，但清除 8 个 input descriptor 的 multicast mask，使各 WG 独立加载。

### 4.2 Output LDS 覆盖竞态

第一次 `tensor_store_from_lds` 发出后，direct epilogue 原先立即把第二组 accumulator 写入同一块 LDS。

硬件文档说明 `tensor_store_from_lds` 由 `TensorCnt` 跟踪。原始汇编 kernel 在复用 LDS 前也执行了：

```asm
s_wait_tensorcnt 0x0
```

补回该等待后：

- standalone random 结果稳定。
- injection e2e 连续运行结果稳定。
- e2e `logits_diff` 与未注入的 FlyDSL baseline 完全一致。

## 5. Quant byte-for-byte 验证

测试脚本：

```text
.codex_tmp/moe_gemm1_act1/check_quant_apre.py
```

在容器内运行：

```bash
cd /data/yanguahe/code/wk_sp1/aiter
python /tmp/check_quant_apre.py
```

结果：

```text
route_payload: mismatches=0
route_scale:   mismatches=0
dense_payload: mismatches=0
dense_scale:   mismatches=0
quant ABpreShuffle byte-for-byte validation passed
```

远端日志：

```text
/tmp/check_quant_apre.log
```

## 6. Standalone 精度测试命令

快速 random 精度测试：

```bash
AITER_LOG_MORE=1 \
python my_code/isa_runner/gemm_batch_isa_runner.py \
  --isa ./my_code/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1.s \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --iters 1 \
  --timing-method cuda-event
```

连续精度复测使用相同命令运行 3 次。最终每次均输出：

```text
checkAllclose passed
gemm_a4w4 err = 0
```

const0 快速精度测试：

```bash
AITER_LOG_MORE=1 \
python my_code/isa_runner/gemm_batch_isa_runner.py \
  --isa ./my_code/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1.s \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --iters 1 \
  --timing-method cuda-event \
  --const-init 0
```

## 7. Standalone 正式性能测试命令

正式性能测试前后均在远端主机运行：

```bash
bash /data/yanguahe/code/gpu_users.sh
```

确认结果：

```text
当前没有进程在使用 GPU。
```

### 7.1 Random

```bash
AITER_LOG_MORE=1 \
python my_code/isa_runner/gemm_batch_isa_runner.py \
  --isa ./my_code/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1.s \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --iters 20
```

最终结果：

```text
device_time_avg = 953.4037 us
gemm_a4w4 err   = 0
TFLOP/s         = 9081.8
```

远端日志：

```text
/tmp/moe_act1_final_perf_random_i20.log
```

### 7.2 Const0

```bash
AITER_LOG_MORE=1 \
python my_code/isa_runner/gemm_batch_isa_runner.py \
  --isa ./my_code/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1.s \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --iters 20 \
  --const-init 0
```

最终结果：

```text
device_time_avg = 702.2786 us
gemm_a4w4 err   = 0
TFLOP/s         = 12329.4
```

远端日志：

```text
/tmp/moe_act1_final_perf_const0_i20.log
```

## 8. E2E injection 测试命令

### 8.1 Random verify

```bash
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_MOE_GEMM1_LAUNCH_BACKEND=cpp \
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario verify \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --iters 2 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --no-check-aot-cache
```

最终连续运行 3 次，结果均为：

```text
logits_diff = 3.3980e-06
rel_l2      = 2.6069e-03
pass        = True
```

未注入的 FlyDSL baseline 连续 3 次也是相同结果：

```text
logits_diff = 3.3980e-06
rel_l2      = 2.6069e-03
```

### 8.2 Random bench

```bash
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_MOE_GEMM1_LAUNCH_BACKEND=cpp \
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --iters 20 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --no-check-aot-cache
```

最终结果：

```text
gemm1                    = 956.745 us
gemm2                    = 535.932 us
fused MoE end-to-end     = 2064.13 us
logits_diff              = 3.39799e-06
rel_l2                   = 0.00260689
pass                     = True
```

远端日志：

```text
/tmp/moe_act1_final_e2e_random_i20_waitfix.log
```

### 8.3 Const0 bench

```bash
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_MOE_GEMM1_LAUNCH_BACKEND=cpp \
python3 -u op_tests/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --iters 20 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --const-init 0
```

最终结果：

```text
gemm1                    = 752.425 us
gemm2                    = 447.031 us
fused MoE end-to-end     = 1735.61 us
logits_diff              = 0
rel_l2                   = 0
pass                     = True
```

远端日志：

```text
/tmp/moe_act1_final_e2e_const0_i20_waitfix.log
```

## 9. 最终性能汇总

| 场景 | kernel/算子 | 时间 |
|---|---|---:|
| standalone random | 新 gemm1 | `953.404 us` |
| standalone const0 | 新 gemm1 | `702.279 us` |
| e2e random | 新 gemm1 | `956.745 us` |
| e2e random | fused MoE 总时间 | `2064.13 us` |
| e2e const0 | 新 gemm1 | `752.425 us` |
| e2e const0 | fused MoE 总时间 | `1735.61 us` |

## 10. 编译与单元测试命令

Python syntax 检查：

```bash
python -m py_compile \
  aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py \
  aiter/ops/flydsl/moe_kernels.py \
  aiter/ops/flydsl/grouped_moe_gfx1250.py \
  my_code/isa_runner/gemm_batch_isa_runner.py \
  my_code/isa_runner/moe_cpp_backend.py \
  op_tests/test_flydsl_grouped_gemm_gfx1250.py
```

Runner self-test：

```bash
python my_code/isa_runner/gemm_batch_isa_runner.py --self-test
```

C++ backend unit tests：

```bash
PYTHONPATH=. python -m unittest my_code.isa_runner.test_moe_cpp_backend
```

结果：

```text
runner SELF_TEST_OK
Ran 23 tests
OK
```

Git whitespace 检查：

```bash
git diff --check
```

## 11. 文件同步与保护检查

GPU/KFD 检查：

```bash
bash /data/yanguahe/code/gpu_users.sh
```

受保护文件检查：

```bash
git diff --exit-code -- \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
```

最终状态：

- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` 本地和远端均未修改。
- `my_code/gemm1_opt_e96_t16384_20260907/` 未触碰。
- 未在远端运行 `git clean`、`git checkout .`、`git reset` 或其他清理命令。
- 仅同步了本任务涉及的文件。
- 9 个任务文件的本地/远端 SHA256 全部一致。
- 最终测试后没有残留 GPU/KFD 进程。

## 12. 机器重连后的相对性能基线与后续优化

MI450/gfx1250 测试机的频率和系统负载会动态变化，因此后续不再跨机器或跨重启比较
绝对时间。每次 SSH 重新连通后，先在同一进程、同一输入和交错执行顺序下重测以下
三个稳定版本：

1. `baseline_act1_independent.s`：安全 baseline，SHA256
   `3f40709bc179544c163b2fd80e8cffc253c261646caf659340ecaad5cfb0d681`。
2. `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s`：第一版优化，
   SHA256 `c90bbe6221b03ae6e05d1d5570b16be25217133e4105860eac901043f316feae`。
3. `moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_double_lds.s`：
   output LDS 双缓冲候选，SHA256
   `33034bb76ada72f9fbc895c1edbf56ee339d0ed6eb28d496196b68323ceed331`。

### 12.1 一键复现脚本

脚本：

```text
my_code/moe_gemm1_act1_optimized/benchmark_history.sh
```

它会记录当前主机、Git HEAD、GFXCLK/利用率、每份 ISA 的 SHA256、正确性和逐轮性能，
并将完整输出写入 `my_code/moe_gemm1_act1_optimized/history_runs/`。

快速 random smoke test：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh quick-random
```

功能：在同一份 random 输入上依次验证安全 baseline、第一版优化和当前候选，并各测一次。

正式 standalone random 对比：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-random
```

功能：默认执行 5 次 warmup、9 轮交错采样、每个样本 5 次 launch，并报告每个版本的
median/mean/min/max。

正式 standalone const0 对比：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-const0
```

功能与 random 相同，但输入为 `--const-init 0`，用于本任务的主要性能目标。

同时执行 random 和 const0：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-both
```

完整 MoE e2e random 对比：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

功能：依次把三个历史版本注入
`op_tests/test_flydsl_grouped_gemm_gfx1250.py`，检查完整 MoE 输出并从同一 profiler
记录 ASM gemm1 和 fused MoE 时间。

完整 MoE e2e const0 对比：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

追加一个临时候选到同轮对比：

```bash
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/candidate.s \
  bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-const0
```

ATT 一键历史对比脚本：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_att_history.sh
```

功能：分别编译三个稳定版本，逐个调用 `my_code/get_isa_runner_att.sh --ana-att`，并将
trace、分析日志和 code object 放入带 UTC 时间戳的 `att_history/` 子目录。

### 12.2 2026-09-09 b8-3 重连后的 standalone 结果

本轮开始时观测到 `sclk=2351 MHz` 且 GPU utilization 为 `100%`，机器上有其他用户的
workload。按用户要求继续测试，因此只比较同轮相对值。

| 数据 | 安全 baseline | 第一版优化 | output LDS 双缓冲 | 双缓冲相对第一版 |
|---|---:|---:|---:|---:|
| random | `1017.893 us` | `886.096 us` | `876.202 us` | `-1.12%` |
| const0 | `910.636 us` | `811.222 us` | `797.913 us` | `-1.64%` |

`double_lds` 的实现方式：第一次 `tensor_store_from_lds` 发出后不立即等待；bank 2/3
改写到每个 wave 相邻的独立 8 KiB LDS 区域，最后一次等待同时覆盖两个 output TDM。
该版本连续 10 次 standalone random 均与第一版优化输出 bit-exact。

### 12.3 2026-09-09 b8-3 重连后的完整 MoE e2e 结果

| 数据 | 版本 | GEMM1 | fused MoE | 正确性 |
|---|---|---:|---:|---|
| random | 安全 baseline | `1018.402 us` | `2148.37 us` | `logits_diff=3.3980e-06`，`rel_l2=2.6069e-03`，通过 |
| random | 第一版优化 | `883.667 us` | `2023.62 us` | `logits_diff=3.3980e-06`，`rel_l2=2.6069e-03`，通过 |
| random | output LDS 双缓冲 | `870.683 us` | `2007.91 us` | `logits_diff=3.3980e-06`，`rel_l2=2.6069e-03`，通过 |
| const0 | 第一版优化 | `825.712 us` | `1923.98 us` | `logits_diff=0`，`rel_l2=0`，通过 |
| const0 | output LDS 双缓冲 | `812.408 us` | `1888.64 us` | `logits_diff=0`，`rel_l2=0`，通过 |

### 12.4 2026-09-09 b8-3 重连后的 ATT cycle

| 版本 | 最大 GFXCLK cycles | ATT wall time | 平均 GFXCLK |
|---|---:|---:|---:|
| 安全 baseline | `2,062,938` | `959.040 us` | 约 `2.151 GHz` |
| 第一版优化 | `1,830,700` | `852.500 us` | 约 `2.147 GHz` |
| output LDS 双缓冲 | `1,783,185` | `833.160 us` | 约 `2.139 GHz` |

output LDS 双缓冲相对第一版优化减少 `47,515 cycles`（`2.60%`），相对安全 baseline
减少 `279,753 cycles`（`13.56%`）。相对目标 `541,900.8 cycles` 仍为 `3.291x`。

### 12.5 本轮否决的实验

- 删除 steady cluster barrier：首次 launch deadlock。
- 每两个 ring 才执行一次 cluster barrier：首次 launch deadlock。
- 将物理 grid 从 `2880x4` 直接裁为 `2304x4`：不符合现有 ASM 的 launch/swizzle
  contract，首次 launch deadlock。
- 全局切回 normal dependency mode 并删除 `s_wait_dscnt 0x8`：random 结果不稳定且错误。
- 只在 epilogue 中依赖 normal-mode 自动 RAW tracking：random 结果不稳定且错误。
- direct `global_store_b64` 输出：结果正确，但相对同轮基线回退约 `8.6%`。
- degree-9 polynomial SiLU：standalone 检查可通过宽松逐元素 gate，但
  `rel_l2=0.0147364` 且性能回退，未通过最终 e2e 精度标准。
- `t256x128` two-buffer resident-WG 系列：结果正确，但约 `938–944 us`。
- `Ktile=512` two-buffer carry：结果正确，但约 `965.935 us`。
- 单个 128-row output TDM：random/const0 均 bit-exact，但同轮性能略慢于
  double-LDS（random `878.605 us` 对 `876.202 us`；const0 `804.616 us` 对
  `801.396 us`）。

所有本轮新增文件均位于 `my_code/moe_gemm1_act1_optimized/`。没有修改或覆盖
`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`，也没有触碰
`my_code/gemm1_opt_e96_t16384_20260907/`。

## 13. a07-3 对历史 515.603 us 版本的复测

历史 `515.603 us` 来自 b8-3，测试文件是：

```text
my_code/moe_gemm1_act1_optimized/moe_gemm1_mxfp4_ABpreShuffle_256x256_4x4_batch_ps_act1_opt.s
SHA256=c90bbe6221b03ae6e05d1d5570b16be25217133e4105860eac901043f316feae
```

在 a07-3 确认 GPU/KFD 空闲后，执行：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh perf-const0
```

脚本首先原样复现：

```bash
bash my_code/moe_gemm1_act1_optimized/test_optimized.sh perf-const0
```

结果为：

```text
device_time_avg = 522.5756 us
```

随后进行同一输入、同一进程、交错顺序的 CUDA-event 比较：

| 版本 | Median | Mean | Min | Max |
|---|---:|---:|---:|---:|
| independent-load 安全 baseline | `704.784 us` | `704.389 us` | `698.839 us` | `708.237 us` |
| 历史 515.603 us 版本 / optimized v1 | `528.216 us` | `528.033 us` | `524.802 us` | `531.268 us` |
| output LDS 双缓冲 | `514.138 us` | `513.700 us` | `507.641 us` | `519.362 us` |

因此，使用原始命令时，a07-3 的 `522.5756 us` 与此前 b8-3 的 `515.603 us`
相差约 `1.35%`。同轮交错比较中，output LDS 双缓冲相对 optimized v1 提升
`14.078 us`，即 `2.67%`。

完整日志：

```text
my_code/moe_gemm1_act1_optimized/history_runs/20260909_125250_perf-const0.log
```

## 14. 正式性能口径：MoE e2e const0 中的 GEMM1

最终性能判断统一采用下面命令输出中的 GEMM1 profiler 行：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

standalone `perf-const0` 只用于快速筛选和定位变化，不再作为最终性能结论。

a07-3 在测试前后均确认 GPU/KFD 空闲。本轮结果：

| 版本 | GEMM1 | fused MoE | 正确性 |
|---|---:|---:|---|
| independent-load 安全 baseline | `724.460 us` | `1700.75 us` | `logits_diff=0`，`rel_l2=0`，通过 |
| 历史 515.603 us 版本 / optimized v1 | `558.548 us` | `1542.56 us` | `logits_diff=0`，`rel_l2=0`，通过 |
| output LDS 双缓冲 | **`551.248 us`** | **`1530.19 us`** | `logits_diff=0`，`rel_l2=0`，通过 |

按正式口径，output LDS 双缓冲相对 optimized v1 的 GEMM1 时间降低
`7.300 us`（`1.31%`），fused MoE 总时间降低 `12.37 us`（`0.80%`）。此前
`515.603 us` 是 b8-3 上的 standalone profiler 结果，不再与 e2e GEMM1 数据直接混用。

完整日志：

```text
my_code/moe_gemm1_act1_optimized/history_runs/20260909_134433_a07_e2e-const0.log
```

## 15. d01-3 重启后的 MoE e2e const0 测试

d01-3 重启后，按用户要求只执行一次以下正式测试，没有额外执行远端状态检查、反向
顺序测试或重复测试：

```bash
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

以下数据替换上一轮 d01-3 的测试值：

| 版本 | GEMM1 | fused MoE | 正确性 |
|---|---:|---:|---|
| independent-load 安全 baseline | `765.878 us` | `2029.95 us` | `logits_diff=0`，`rel_l2=0`，通过 |
| 历史 515.603 us 版本 / optimized v1 | `580.872 us` | `1819.18 us` | `logits_diff=0`，`rel_l2=0`，通过 |
| output LDS 双缓冲 | **`554.139 us`** | **`1798.40 us`** | `logits_diff=0`，`rel_l2=0`，通过 |

output LDS 双缓冲相对 optimized v1 的 GEMM1 时间降低 `26.733 us`
（`4.602%`），fused MoE 总时间降低 `20.78 us`（`1.142%`）。相对安全
baseline，GEMM1 时间降低 `211.739 us`（`27.647%`）。脚本开始时记录的
`sclk` 为 `2358 MHz`。这些数据来自用户指定的单次重启后测试，没有进行复测取中位数。

本地保存的性能摘要：

```text
my_code/moe_gemm1_act1_optimized/history_runs/20260909_153855_d01_reboot_e2e-const0_summary.log
```

benchmark 脚本生成的远端原始日志：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/moe_gemm1_act1_optimized/history_runs/20260909_153855_e2e-const0.log
```
