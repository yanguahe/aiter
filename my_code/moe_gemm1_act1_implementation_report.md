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

