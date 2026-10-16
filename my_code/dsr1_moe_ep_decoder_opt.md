# DSR1 MoE EP decoder 性能记录

## 2026-10-13：f01-1 四卡 MegaMoE main 基准

本次在 `f01-1` 的 `dsr1_ep4` 容器中更新、安装 ROCm/aiter 的 `main`，执行 16 层、每 rank 1536 tokens 的 `bench_mega_moe.py`，对比 `base` 与 `fused` combine。运行成功，退出码为 `0`。

**16 层中值耗时：base 为 7441.424 μs，fused 为 6448.417 μs；fused 相比 base 加速 1.153992×，耗时降低 13.3443%。** 本记录为 MoE 多层 benchmark，未测量完整 DSR1 serving 的 decode 延迟或吞吐。

### 测试环境与版本

| 项目 | 实际配置 |
|---|---|
| 机器 | `f01-1` / `yanguahe@ctheliosr-1b114-f01-1.amd.com` |
| 容器 | `dsr1_ep4` |
| 仓库目录 | `/app/aiter` |
| Git remote / branch | `git@github.com:ROCm/aiter.git` / `main` |
| AITER HEAD（与更新时的 `origin/main` 一致） | `e212581c0477a58dbd2c164176e950198c651143` |
| AITER | `0.1.25.dev220+ge212581c0` |
| FlyDSL | `0.3.4.1` |
| Mori | `1.2.3.post2.dev43+g6cd0ca41b` |
| PyTorch | `2.11.0+rocm10.1.0a20260811.bkc.20260820` |
| Triton（package metadata） | `3.8.0.dev20260903+git9551a57b` |
| Python | `3.12.3` |
| GPU arch | `gfx1250` |
| ROCm SDK 路径 | `/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core` |
| 开始 / 结束时间（UTC） | `2026-10-13 08:39:41` / `2026-10-13 08:40:04` |

更新和安装前执行了 `source /data/yanguahe/code/git_env`。原分支 `alizaidy/dsv4-dsr1-20260930` 的未提交修改保存到 stash `codex-preserve-combo-before-main-20261013`，并另存补丁。更新 `main` 及 submodule 后，安装遇到 PEP 668 限制，设置 `PIP_BREAK_SYSTEM_PACKAGES=1` 后成功完成。保留了容器原有 Triton，FlyDSL 更新至 AITER main 要求的 `0.3.4.1`。benchmark 结束后的 Git 工作区干净。

安装命令：

```bash
source /data/yanguahe/code/git_env
cd /app/aiter
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_INPUT=1 PYTHONUNBUFFERED=1
export AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=gfx1250
export FLYDSL_GPU_ARCH=gfx1250 PYTORCH_ROCM_ARCH=gfx1250 MAX_JOBS=32
python3 setup.py develop
```

### 完整 benchmark 命令与配置

在 `dsr1_ep4` 容器中执行：

```bash
source /data/yanguahe/code/git_env
cd /app/aiter
export PYTHONUNBUFFERED=1
MORI_V2_KERNEL_BACKEND=hip \
MORI_EP_TOKOFF_EXT="${MORI_EP_TOKOFF_EXT:-0}" \
MEGA_DISPATCH=mori \
FLYDSL_DUMP_IR=0 \
MORI_GPU_ARCHS=gfx1250 \
FLYDSL_GPU_ARCH=gfx1250 \
AITER_FORCE_A8W4=0 \
AITER_MOE_EXPERT_BALANCE=True \
exec torchrun --standalone --nproc_per_node=4 \
"./op_tests/multigpu_tests/bench_mega_moe.py" \
-e 256 \
-k 8 \
-hd 7168 \
-id 2048 \
--layers 16 \
-tpr 1536 \
-q a4w4_mxfp4 \
--combine both \
--iters 20 \
--acc_verify 0 \
--profile_table 1 \
--dispatch_wire fp4
```

日志确认的配置：

```text
world=4 layers=16 tokens/rank=1536 hidden=7168 inter=2048
E=256 topk=8 EPR=64 quant=a4w4_mxfp4
combine=both dispatch=mori stage1_fused=False
dispatch_wire=fp4 combine_quant=none force_a8w4=0
gate=INTERLEAVE shared_E=0 expert_balance=True
data_init=norm seed=0 gfx=gfx1250
```

每个 combine mode 使用 5 次 warmup、20 次计时 graph replay；每次 replay 是完整的 16 层链。另执行默认 3 次 profiler replay，输出 kernel profile。脚本先运行 `base`，再运行 `fused`。

### MoE e2e 性能

下表来自日志的 `mega_moe summary`，保留到小数点后 3 位，时间单位为 μs。脚本分别计算每个 rank 的统计值，再对四个 rank 求平均；每层中值耗时为 `median_us / 16`。

| combine | 16 层 min | 16 层 median | 16 层 mean | 16 层 max | 每层 median | speedup_vs_base |
|---|---:|---:|---:|---:|---:|---:|
| base | 7402.824 | 7441.424 | 7714.914 | 12937.168 | 465.089 | 1.000000× |
| fused | 6421.550 | 6448.417 | 6446.979 | 6483.781 | 403.026 | 1.153992× |

以中值比较，fused 每次 16 层 replay 节省 `993.007 μs`，每层节省 `62.063 μs`。计算口径：

```text
speedup = base_median / fused_median = 1.1539923167
耗时降低 = (base_median - fused_median) / base_median = 13.3443104%
```

base 的 mean 和 max 高于 median，原始统计值如表保留；本次未追加重复运行以分析这一长尾。

### Kernel profile

记录 GEMM1、GEMM2 和通信 kernel，时间单位为 μs；完整 symbol 与各 rank 耗时如下。`avg_us` 来自独立 profiler replay，不能将其直接求和作为 e2e 中值。

**# mega_moe kernel profile [base]**

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t256x256x256_w1x4_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_mg10_planarlds_interleavelds_rolleddrain` | 115.07697 | 46.00000 | 114.67747 | 114.23525 | 115.15615 | 116.23900 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.43554 | 46.00000 | 78.14453 | 79.97691 | 78.26081 | 77.35991 |
| `a8w4_tdm_fp4_t256x256x256_w2x2_b3_K2048_e64_prefetch_mg10_planarlds_interleavelds_rolleddrain` | 69.38021 | 46.00000 | 69.39374 | 68.80289 | 69.40849 | 69.91574 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.11827 | 46.25000 | 24.47381 | 38.38918 | 28.69902 | 28.91106 |

**# mega_moe kernel profile [fused]**

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t256x256x256_w2x2_b3_K2048_e64_prefetch_epscatter_mg10_planarlds_interleavelds_rolleddrain` | 121.91003 | 42.00000 | 120.93362 | 123.86768 | 120.33284 | 122.50598 |
| `a8w4_tdm_fp4_t256x256x256_w1x4_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_mg10_planarlds_interleavelds_rolleddrain` | 113.29324 | 42.00000 | 112.40432 | 113.10257 | 113.57175 | 114.09430 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 28.58098 | 42.25000 | 30.89191 | 25.16432 | 28.39700 | 29.87067 |
| `ep_combine_fused_0` | 17.62442 | 42.00000 | 17.46162 | 17.49570 | 17.51352 | 18.02684 |
| `ep_combine_fused_sync_0` | 8.63257 | 42.00000 | 10.40753 | 7.24983 | 9.39575 | 7.47719 |

### 测试状态与验证范围

- 启动前在主机执行 GPU/KFD 检查，确认四张 GPU 均空闲：`busy=0`，每卡 `vram_used=177262592` bytes，无预先存在的 GPU/KFD 工作进程。等待期间每 6 秒检查一次，放行后停止空闲轮询，再启动 benchmark。
- 测试结束后复查：`dsr1_ep4` 无本次 benchmark 进程残留；最终四张 GPU 均为 `busy=0`，VRAM 回到启动前基线。
- 安装和运行退出码均为 `0`。本次使用 `--acc_verify 0`，未执行数值正确性验证。
- 本次对比的是同一 AITER main 提交下的 base / fused combine，不代表与历史无 drain 路径的性能对比，也不代表完整 serving 性能。

### 原始数据与复现文件

远端归档目录（主机与容器共享）：

```text
/data/yanguahe/code/wk_sp1/.codex_tmp/f01_1_dsr1_ep4_main_20261013/
```

本地副本位于工作区根目录 `.codex_tmp/f01_1_dsr1_ep4_main_20261013/`；下载后已核对 12 个文件的 SHA256。

- [完整运行日志](../../.codex_tmp/f01_1_dsr1_ep4_main_20261013/bench_mega_moe_attempt1.log)
- [原始 benchmark JSON](../../.codex_tmp/f01_1_dsr1_ep4_main_20261013/bench_mega_moe_tables.json)
- [环境版本记录](../../.codex_tmp/f01_1_dsr1_ep4_main_20261013/environment_ready.log)
- [安装日志](../../.codex_tmp/f01_1_dsr1_ep4_main_20261013/install_aiter_retry1.log)
- [运行脚本](../../.codex_tmp/f01_1_dsr1_ep4_main_20261013/run_bench_mega_moe.sh)
- [启动前 GPU 空闲检查](../../.codex_tmp/f01_1_dsr1_ep4_main_20261013/gpu_idle_pre_bench.log)
- [结束后 GPU 与进程检查](../../.codex_tmp/f01_1_dsr1_ep4_main_20261013/final_gpu_check.log)
- [SHA256 清单](../../.codex_tmp/f01_1_dsr1_ep4_main_20261013/artifact.sha256)

主要数据文件 SHA256：

```text
d0a5bd969e301729f430befd73fb25af2e1fbe0ede171b3b09b9762913fb46a9  bench_mega_moe_attempt1.log
5b46d175ec7e1109b0f46a191bc052f30ebb0d9826ac060228095c7c7ed8c86c  bench_mega_moe_tables.json
```

## 2026-10-13：refactor 分支四卡 prefill / 双 persistent


分支已安装到 `f01-1` 的 `dsr1_ep4:/app/aiter_moe_a4w4_pr_refactor_wt`。当前基于 `origin/moe_a4w4_pr_refactor_wt` 的 `e749e6cef3392295f7a1c491a1d9f51a1b75bbb2`，origin 为 `git@github.com:yanguahe/aiter.git`。所有正式性能运行保持四卡、16 层、1536 tokens/rank、E256、topk8、hidden7168、inter2048、FP4 dispatch wire、`--iters 20 --acc_verify 0 --profile_table 1 --combine both`。

### 端到端性能

单位均为 μs。`median_us` 是 16 层总耗时，`per_layer_us` 是其中值除以 16；统计为四个 rank 的均值。kernel profile 的 `avg_us` 是独立 profiler replay 中的每次 kernel 调用耗时，不可直接相加替代端到端中值。

| 运行 | combine | min_us | median_us | mean_us | max_us | per_layer_us |
|---|---|---:|---:|---:|---:|---:|
| stock | base | 7490.873 | 7519.276 | 7518.465 | 7545.674 | 469.955 |
| stock | fused | 6518.570 | 6556.998 | 6555.993 | 6585.426 | 409.812 |
| prefill3 | base | 7614.951 | 7641.054 | 7654.473 | 7934.256 | 477.566 |
| prefill3 | fused | 7384.754 | 11099.594 | 10718.792 | 19285.434 | 693.725 |
| prefill4 | base | 7646.954 | 7667.743 | 7970.211 | 13692.680 | 479.234 |
| prefill4 | fused | 6881.592 | 6900.069 | 6902.585 | 6969.062 | 431.254 |
| persistent_gemm2 | base | 7371.813 | 7394.530 | 7585.020 | 11229.109 | 462.158 |
| persistent_gemm2 | fused | 6616.075 | 6640.578 | 6682.702 | 7371.305 | 415.036 |
| persistent_both | base | 7007.769 | 7043.409 | 7040.430 | 7066.470 | 440.213 |
| persistent_both | fused | 6302.815 | 6327.777 | 6515.919 | 10094.796 | 395.486 |
| persistent_both_repeat | base | 7011.262 | 7036.037 | 7033.877 | 7047.749 | 439.752 |
| persistent_both_repeat | fused | 6292.879 | 6322.483 | 6440.899 | 8122.802 | 395.155 |
| verified_baseline | base | 6995.064 | 7029.273 | 7028.940 | 7065.687 | 439.330 |
| verified_baseline | fused | 6322.512 | 6347.078 | 6625.670 | 11887.847 | 396.692 |
| verified_baseline_pair_retry | base | 7008.426 | 7040.164 | 7039.779 | 7072.939 | 440.010 |
| verified_baseline_pair_retry | fused | 6315.405 | 6334.282 | 6333.601 | 6355.610 | 395.893 |
| verified_baseline_debug | base | 7035.885 | 7051.745 | 7396.351 | 13902.131 | 440.734 |
| verified_baseline_debug | fused | 6298.988 | 6320.273 | 6685.370 | 13588.308 | 395.017 |
| scatter_hoist | base | 6996.341 | 7018.204 | 7022.856 | 7108.066 | 438.638 |
| scatter_hoist | fused | 6267.111 | 6295.909 | 6299.460 | 6337.275 | 393.494 |
| scatter_hoist_repeat | base | 7023.674 | 7056.193 | 7693.558 | 13981.377 | 441.012 |
| scatter_hoist_repeat | fused | 6329.202 | 6345.163 | 6555.313 | 10574.240 | 396.573 |
| scatter_dgroup_cache | base | 6998.330 | 7032.208 | 7029.531 | 7063.008 | 439.513 |
| scatter_dgroup_cache | fused | 6297.304 | 6323.979 | 6556.509 | 10893.437 | 395.249 |
| scatter_dgroup_cache_debug | base | 6994.397 | 7028.533 | 7033.076 | 7084.773 | 439.283 |
| scatter_dgroup_cache_debug | fused | 6279.519 | 6312.238 | 6532.392 | 10716.589 | 394.515 |

双 persistent 复测（`persistent_both_repeat`）的 MoE e2e 性能，单位为 μs：

| combine | 16 层 min | 16 层 median | 16 层 mean | 16 层 max | 每层 median | speedup_vs_base |
|---|---:|---:|---:|---:|---:|---:|
| base | 7011.262 | 7036.037 | 7033.877 | 7047.749 | 439.752 | 1.000000× |
| fused | 6292.879 | 6322.483 | 6440.899 | 8122.802 | 395.155 | 1.112860× |

相对 prefill 基线，双 persistent 复测的 16 层中值：base 从 `7667.743 μs` 降至 `7036.037 μs`（降低 `8.2385%`），fused 从 `6900.069 μs` 降至 `6322.483 μs`（降低 `8.3707%`）。

### 已冻结的精度正确版本

`persistent_both_sources/` 保存 GEMM1、GEMM2 均使用 persistent `t192/w2x4/b4` 的精度正确基线，`persistent_both_source_manifest.json` 保存 8 个实现文件的 SHA256。2026-10-13 按“先精度正确、记录性能、再优化”的要求，已将本地和容器恢复到该快照，`verified_baseline_seed0` 精度验证和 `verified_baseline` 正式性能测试均完成，退出码为0。该版16层 base / fused 中值为 **7029.273 / 6347.078 μs**，每层为439.330 / 396.692 μs；较 `prefill4` 分别降低8.33% / 8.01%。后续候选的源码、精度结果和性能记录单独保存；基线快照保持不变。

基线的实际 kernel symbols：

```text
GEMM1 base / fused:
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist

GEMM2 base:
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist

GEMM2 fused:
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter
```

### 环境与实现

| 项目 | 实际配置 |
|---|---|
| 主机 / 容器 | `f01-1` / `dsr1_ep4` |
| 仓库 | `/app/aiter_moe_a4w4_pr_refactor_wt` |
| branch / HEAD | `moe_a4w4_pr_refactor_wt` / `e749e6cef3392295f7a1c491a1d9f51a1b75bbb2` 加本轮补丁 |
| origin | `git@github.com:yanguahe/aiter.git`，ROCm upstream 未提供该分支 |
| AITER | `0.1.1.dev3417+ge749e6cef` |
| FlyDSL | `0.3.4.1` |
| Mori | `1.2.3.post2.dev43+g6cd0ca41b` |
| PyTorch metadata | `2.11.0+rocm10.1.0a20260811.bkc.20260820` |
| Python / GPU | `3.12.3` / 四卡 `gfx1250` |

已完成 submodule 更新及 `python3 setup.py develop`。导入路径确认是上述 refactor 仓库。kernel 由 FlyDSL JIT 根据当前源码编译。

主要接入和修复：

1. **GEMM2 prefill 路径接入。** 原分支 stock 在本四卡配置中实际选中 generic `mxfp4_preshuffle_gfx1250_tdm.py`。GEMM2 的输入由本地 GEMM1 产生，与 GEMM1 dispatch wire 的布局条件分开判断，使其能够选择独立 A-preshuffle producer 和用户指定的 `mxfp4_preshuffle_gfx1250_tdm_prefill.py`。BF16 EP scatter 新增到 prefill/persistent launcher；quantized combine 使用已有支持路径。
2. **cluster 选择一致性。** producer eligibility 原来使用 stage1 的 cluster，而 stage2 launcher 对 CSV `cluster_n2=-1` 解析为另一配置。目标 E64/hidden7168/inter2048 的 EP geometry 使用一致的 stage2 cluster4，避免 producer 和 consumer 布局/调度不匹配。
3. **GEMM1 接收 FP4 dispatch。** Mori 已在 dispatch wire 上量化 FP4，GEMM1 producer 直接转换 payload 和 ScaleA。7168 维的有效 ScaleA 为 224 bytes/row，但 Mori 的实际 source pitch 为 256 bytes/row；新增真实 source stride 参数，避免把下一 token 的 scale 地址算错。
4. **无效 route 与 alignment。** inverse route map 对 `DROPPED_ROUTE_ROW=-1` 跳过写入；GEMM1、GEMM2 使用共同的 192 行 expert alignment，避免 256/192 混用导致 tile 跨 expert。
5. **GEMM1 fused ScaleA 正确性。** 原 t192 epilogue 按一组 M96 scale 行写出，而 GEMM2 读取 canonical M32 布局，造成 M/K 转置。改为每个 M wave 写三个 M32 scale groups，并使用对应的 global stride。首次双 persistent 的精度失败因此恢复通过。
6. **persistent GEMM2 的 EP scatter 与等待。** 每个 t192 output 分成两个 96 行 packed LDS slices，各用 12 个 8-row TDM stores。第二 slice 的 ownership 旋转四个 wave，使八个 wave 各发出三个 stores。下一 N-task 的 I0/I1 先于旧 output stores 发出，prologue 按每 wave 的三个 output stores 等待，再执行 WG barrier；先消费 stage0，然后在复用 stages2/3 前执行 `pipeline_fence(0)`。input/output overlap 保留，并在 LDS 复用处保证完成。
7. **JIT Tensor 参数。** 非 EP scatter 的 GEMM2 launcher 传入有效 dummy Tensor，避免默认 `None` 在相关 JIT cache-key 路径中报错。

等待逻辑的文档依据是本地 `MI400_Shader_Programming#65.txt` §4.10.1（印刷页197）：“Tensor instructions complete in-order with other Tensor instructions from the same wave ... but are unordered with instructions from other waves.” 每 wave 等待与 WG barrier 的组合是基于该顺序保证及当前实际发出的 store 数量推导；不能只用一个 wave 的等待保证整个 WG 的 input 已到达。

### 数值验证

精度使用原 benchmark 的 FP32 RefModel 和原始 `logits_diff` / `logits_tol` 计算方式。四卡、16 层、1536 tokens/rank、base/fused 均验证。no-profiler runner 只替换 profile/计时收集，保留原模型、graph、warmup 和 replay，不记录性能；其 JSON 中时间为 0。

| 实现 / 验证 | seed | expert_balance | base logits_diff | fused logits_diff | logits_tol | 结果 |
|---|---:|---|---:|---:|---:|---|
| 双 persistent / 原 benchmark，16 层 | 0 | True | 0.1842963171 | 0.1842538070 | 0.3609625668 | PASS |
| 双 persistent / no-profiler | 0 | True | 0.1842963171 | 0.1842538070 | 0.3609625668 | PASS |
| 双 persistent / no-profiler | 1 | True | 0.1838829516 | 0.1838064915 | 0.3609625668 | PASS |
| 双 persistent / no-profiler | 2 | False | 0.2309923653 | 0.2310255178 | 0.3609625668 | PASS |
| 冻结基线重新验证 / no-profiler | 0 | True | 0.1842963171 | 0.1842538070 | 0.3609625668 | PASS |
| metadata hoist / no-profiler | 0 | True | 0.1842963171 | 0.1842538070 | 0.3609625668 | PASS |

另外新增 `my_code/verify_moe_fp4_ep_apreshuffle.py`：独立 PyTorch gather + reshape/permute oracle 对比完整 payload/ScaleA 字节，并验证 padding 和 dead-tail 未写区域。以下 4 cases 每项 3 次 replay 均 PASS：

| feature_dim | source_scale_stride | experts | dropped routes / dead-tail |
|---:|---:|---:|---|
| 7168 | 256 | 64 | 有 |
| 7168 | 224 | 4 | 无 |
| 2048 | 128 | 4 | 有 |
| 2048 | 64 | 4 | 无 |

这些验证覆盖本次目标及上述输入布局/路由案例；历史单卡 T4608、其它 geometry 的性能与精度需按各自配置单独验证。

### 波动和 profiler 限制

部分补充 `acc_verify=1` 运行在 PyTorch/Kineto profiler 退出时 SIGSEGV。faulthandler 栈落在 `torch/autograd/profiler.py:418 __exit__`、`torch/profiler/profiler.py stop_trace`、`bench_mega_moe.py:829`。这项问题尚未修复；调用栈本身不足以彻底排除 kernel 原因。因此补充验证使用独立 no-profiler runner，并保留失败日志。正式 `acc_verify=0 --profile_table 1` 性能命令已有多次完成记录。

`prefill3` fused 中值为 11099.594 μs，波动明显，保留该轮所有数据但采用 `prefill4` 作为 prefill 基线。后续部分运行的 mean/max 也存在长尾，表格照录。

`scatter_hoist` 尝试将同一 WG 七个 N-task 共享的 route weights/destinations 在循环外读取一次。其首轮 fused 中值为 6295.909 μs，但独立 profile 中 GEMM2 rank1 为 224.11427 μs，其他 rank 约118–119 μs，`ep_combine_fused_sync_0` avg 为 96.46893 μs。复测中 GEMM2 avg 回到 119.23551 μs，fused e2e 中值为 6345.163 μs。两轮没有形成稳定收益，因此该候选暂未纳入精度正确基线；源码和完整 profile 保留。

`scatter_dgroup_cache` 在上述 metadata 复用基础上，将七个 N-task 共用的 gather descriptor GROUP1/2/3 构造移到循环外，只在每次 store 前更新 GROUP0 地址。seed0 的16层 base / fused 精度均 PASS，首轮 e2e 中值为7032.208 / 6323.979 μs。同窗口基线复测 `verified_baseline_pair_retry` 为7040.164 / 6334.282 μs。`verified_baseline_pair` 与 `scatter_dgroup_cache_repeat` 均在输出 base profile 后、fused 阶段报告 rank1 SIGSEGV，退出码1，只保留已收集的 base profile，不补造 fused 或 e2e 数据。

随后仅额外启用 `PYTHONFAULTHANDLER=1` 的两轮复测均完整完成，退出码0：`verified_baseline_debug` 的 base / fused 中值为7051.745 / 6320.273 μs，`scatter_dgroup_cache_debug` 为7028.533 / 6312.238 μs。候选 fused 中值比同窗口基线低约0.13%，仍没有足够证据说明形成稳定收益，因此未纳入基线。启用 faulthandler 只增加故障诊断，不能据此认定 SIGSEGV 已修复或排除 kernel 原因。当前本地及容器均保留精度正确的 `persistent_both_sources` 基线；descriptor 复用候选保存在单独源码快照中。

每轮性能前在主机检查所有四卡的 KFD/GPU 所有权、busy 和 VRAM；有其它任务时每6秒重检，空闲后才启动。计时期间停止轮询，结束后检查进程及GPU。空闲时每卡 `busy=0`、静态 `vram_used=177262592` bytes。未修改时钟、NUMA 设置或驱逐他人任务。

### 本轮复现与证据

正式 benchmark 命令：

```bash
source /data/yanguahe/code/git_env
cd /app/aiter_moe_a4w4_pr_refactor_wt
MORI_V2_KERNEL_BACKEND=hip \
MORI_EP_TOKOFF_EXT="${MORI_EP_TOKOFF_EXT:-0}" \
MEGA_DISPATCH=mori \
FLYDSL_DUMP_IR=0 \
MORI_GPU_ARCHS=gfx1250 \
FLYDSL_GPU_ARCH=gfx1250 \
AITER_FORCE_A8W4=0 \
AITER_MOE_EXPERT_BALANCE=True \
exec torchrun --standalone --nproc_per_node=4 \
./op_tests/multigpu_tests/bench_mega_moe.py \
-e 256 -k 8 -hd 7168 -id 2048 \
--layers 16 -tpr 1536 -q a4w4_mxfp4 \
--combine both --iters 20 --acc_verify 0 \
--profile_table 1 --dispatch_wire fp4
```

远端归档：`/data/yanguahe/code/wk_sp1/.codex_tmp/f01_1_dsr1_refactor_p192_20261013/`，本地副本在 workspace 根目录 `.codex_tmp/f01_1_dsr1_refactor_p192_20261013/`。

- [基线源码 SHA256](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/persistent_both_source_manifest.json)
- [冻结基线完整性能 JSON](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_verified_baseline_tables.json)
- [冻结基线精度 JSON](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/accuracy_verified_baseline_seed0_tables.json)
- [正式性能 runner](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/run_bench_mega_moe.sh)
- [独立 16 层精度 runner](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/verify_mega_moe_without_profiler.py)
- [seed0 精度 JSON](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/accuracy_persistent_both16_noprof_seed0_tables.json)
- [seed1 精度 JSON](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/accuracy_persistent_both16_noprof_seed1_tables.json)
- [不均衡路由精度 JSON](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/accuracy_persistent_both16_noprof_unbalanced_tables.json)
- [环境版本](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/final_environment.log)

下文逐轮记录 GEMM1、GEMM2 和通信 kernel 的完整 symbol、`avg_us`、`calls`、`rank0_us`～`rank3_us`。

### 每轮 GEMM 与通信 kernel profile

逐轮记录 base / fused 的 GEMM1、GEMM2、Mori dispatch/combine 和 fused combine/sync，保留完整 kernel symbol、`avg_us`、`calls`、`rank0_us`～`rank3_us`。时间为 μs，展示保留5位小数；`calls` 为跨 rank 平均值，可含小数。各 rank 数据来自独立 profiler replay。未完成的运行只展示已经收集的数据，不用于 e2e 性能对比。

各表前的 `per_layer_us` 来自同轮、同 combine 模式的 `mega_moe summary`，即 `median_us / 16`，保留3位小数。未完成且未输出 summary 的运行标为“未记录”。

#### 原分支 stock（实际 GEMM2 为 generic TDM）

原始 JSON：[bench_stock_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_stock_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**469.955 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_mg10_planarlds_interleavelds` | 117.57579 | 45.25000 | 116.05046 | 118.26709 | 115.69127 | 120.29434 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.36293 | 45.25000 | 81.14789 | 78.10278 | 78.45151 | 75.74952 |
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_prefetch_mg10_planarlds_interleavelds` | 69.88932 | 45.25000 | 69.23630 | 69.08926 | 71.19713 | 70.03457 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 39.11509 | 45.25000 | 26.63796 | 41.50191 | 43.32427 | 44.99623 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**409.812 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_prefetch_epscatter_mg10_planarlds_interleavelds` | 120.93756 | 46.25000 | 121.17084 | 120.56864 | 119.61115 | 122.39958 |
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_mg10_planarlds_interleavelds` | 115.09320 | 46.25000 | 114.87798 | 113.29102 | 115.77321 | 116.43060 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 29.00158 | 46.50000 | 31.94407 | 30.72856 | 29.72191 | 23.61179 |
| `ep_combine_fused_0` | 17.71902 | 46.25000 | 17.53449 | 17.59678 | 17.66879 | 18.07604 |
| `ep_combine_fused_sync_0` | 9.32074 | 46.25000 | 9.84576 | 11.84511 | 9.37134 | 6.22077 |



#### prefill 首次完成运行（波动较大，保留原始数据）

原始 JSON：[bench_prefill3_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_prefill3_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**477.566 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cluster1x4_prefetch_mg10_planarlds_interleavelds` | 113.84083 | 46.00000 | 112.92759 | 113.67779 | 113.69764 | 115.06031 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 77.67282 | 46.00000 | 79.39909 | 77.34070 | 77.18604 | 76.76546 |
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre` | 67.50692 | 46.00000 | 66.70983 | 67.40870 | 68.14874 | 67.76040 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 25.84010 | 46.25000 | 27.46545 | 27.04137 | 24.43472 | 24.41885 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**693.725 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cluster1x4_prefetch_mg10_planarlds_interleavelds` | 236.47894 | 47.00000 | 249.56387 | 229.25009 | 234.78210 | 232.31971 |
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_epscatter` | 192.92444 | 47.00000 | 178.77827 | 201.88907 | 196.49552 | 194.53492 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 35.58798 | 47.25000 | 36.22296 | 32.50091 | 34.52400 | 39.10404 |
| `ep_combine_fused_sync_0` | 17.98026 | 47.00000 | 6.69985 | 23.32305 | 19.83194 | 22.06621 |
| `ep_combine_fused_0` | 17.03903 | 47.00000 | 16.52548 | 17.29211 | 16.98356 | 17.35498 |



#### prefill 基线复测

原始 JSON：[bench_prefill4_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_prefill4_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**479.234 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cluster1x4_prefetch_mg10_planarlds_interleavelds` | 114.17098 | 46.00000 | 112.99568 | 113.79354 | 114.60138 | 115.29330 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 77.97956 | 46.00000 | 79.86026 | 78.92833 | 76.57525 | 76.55440 |
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre` | 68.13602 | 46.00000 | 67.38721 | 67.98402 | 68.62848 | 68.54435 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 31.61037 | 46.25000 | 32.33583 | 24.86685 | 29.93629 | 39.30251 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**431.254 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_epscatter` | 129.91908 | 45.50000 | 128.43652 | 130.46158 | 130.89819 | 129.88005 |
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cluster1x4_prefetch_mg10_planarlds_interleavelds` | 113.72869 | 45.50000 | 114.21688 | 112.85140 | 113.41877 | 114.42773 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.13248 | 45.50000 | 32.25881 | 24.46129 | 31.68383 | 32.12598 |
| `ep_combine_fused_0` | 17.34613 | 45.50000 | 17.30717 | 17.43042 | 17.27660 | 17.37032 |
| `ep_combine_fused_sync_0` | 10.89526 | 45.50000 | 5.49471 | 13.04516 | 12.21817 | 12.82302 |



#### 仅 GEMM2 persistent，中间阶段

原始 JSON：[bench_persistent_gemm2_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_persistent_gemm2_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**462.158 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cluster1x4_prefetch_mg10_planarlds_interleavelds` | 107.70458 | 44.50000 | 106.85761 | 106.82570 | 108.28702 | 108.84800 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 79.71103 | 44.50000 | 80.96513 | 82.33522 | 77.08702 | 78.45676 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 57.39548 | 44.50000 | 57.20961 | 56.42254 | 57.94224 | 58.00751 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 31.70590 | 44.75000 | 27.22754 | 25.86628 | 39.71993 | 34.00985 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**415.036 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 117.95827 | 42.75000 | 119.19633 | 119.21540 | 114.57729 | 118.84407 |
| `a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cluster1x4_prefetch_mg10_planarlds_interleavelds` | 107.23574 | 42.75000 | 106.79467 | 105.80935 | 107.86590 | 108.47302 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.63684 | 43.00000 | 36.34433 | 27.24809 | 29.70878 | 29.24618 |
| `ep_combine_fused_0` | 17.11788 | 42.75000 | 17.05523 | 17.30588 | 17.29712 | 16.81327 |
| `ep_combine_fused_sync_0` | 10.49724 | 42.75000 | 12.02535 | 13.68547 | 6.04027 | 10.23786 |



#### GEMM1 / GEMM2 双 persistent，首轮

原始 JSON：[bench_persistent_both_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_persistent_both_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**440.213 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.24191 | 44.25000 | 97.67498 | 97.27989 | 99.10471 | 98.90805 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 79.30818 | 44.25000 | 80.26114 | 81.12122 | 76.39422 | 79.45614 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 59.26363 | 44.25000 | 59.48595 | 58.13664 | 60.17796 | 59.25398 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 31.94291 | 44.50000 | 27.06914 | 35.66396 | 27.69376 | 37.34478 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**395.486 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 119.37061 | 41.75000 | 119.70500 | 120.82150 | 119.32393 | 117.63202 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 97.85926 | 41.75000 | 97.78227 | 97.34531 | 97.93217 | 98.37730 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 31.91628 | 41.75000 | 38.23332 | 25.12329 | 36.36461 | 27.94390 |
| `ep_combine_fused_0` | 17.50321 | 41.75000 | 17.39593 | 17.50995 | 17.49556 | 17.61137 |
| `ep_combine_fused_sync_0` | 7.67209 | 41.75000 | 8.20573 | 7.01907 | 7.71095 | 7.75260 |



#### GEMM1 / GEMM2 双 persistent，复测

原始 JSON：[bench_persistent_both_repeat_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_persistent_both_repeat_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**439.752 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.12277 | 45.50000 | 97.22400 | 98.22593 | 98.19104 | 98.85010 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.30552 | 45.25000 | 79.01690 | 78.35440 | 77.54637 | 78.30440 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 59.21591 | 45.50000 | 58.52533 | 59.23869 | 59.81115 | 59.28846 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 31.77928 | 45.50000 | 24.49355 | 34.72724 | 33.56762 | 34.32871 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**395.155 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 119.65297 | 45.50000 | 120.25752 | 120.57258 | 118.22234 | 119.55944 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.15036 | 45.50000 | 97.57612 | 97.96348 | 98.48018 | 98.58165 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 26.81235 | 45.50000 | 26.56629 | 28.69692 | 27.72593 | 24.26027 |
| `ep_combine_fused_0` | 17.47915 | 45.50000 | 17.01288 | 17.85317 | 17.45430 | 17.59627 |
| `ep_combine_fused_sync_0` | 7.36584 | 45.50000 | 8.00905 | 7.63279 | 7.35280 | 6.46873 |



#### 已冻结的精度正确双 persistent 基线

原始 JSON：[bench_verified_baseline_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_verified_baseline_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**439.330 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.44468 | 42.25000 | 98.23645 | 98.08136 | 98.35873 | 99.10218 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 79.05920 | 42.25000 | 80.33405 | 79.59988 | 77.99378 | 78.30910 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 58.98674 | 42.25000 | 58.54060 | 58.75255 | 59.27709 | 59.37673 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.36857 | 42.25000 | 27.86040 | 25.31107 | 29.50984 | 38.79297 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**396.692 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 119.68991 | 42.75000 | 120.23997 | 118.95344 | 121.00330 | 118.56292 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.35451 | 42.75000 | 97.91154 | 97.96423 | 98.53923 | 99.00302 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 29.54765 | 42.75000 | 31.07884 | 31.43584 | 24.62379 | 31.05215 |
| `ep_combine_fused_0` | 17.71442 | 42.75000 | 17.54300 | 17.78091 | 17.65142 | 17.88235 |
| `ep_combine_fused_sync_0` | 7.67293 | 42.75000 | 8.35316 | 8.58677 | 6.96937 | 6.78244 |



#### 基线配对测试未完成（保留已收集的 profile）

原始 JSON：[bench_verified_baseline_pair_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_verified_baseline_pair_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**未记录**（本轮中途失败，未输出 `mega_moe summary`）。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.65046 | 42.75000 | 97.35053 | 97.37983 | 100.19207 | 99.67942 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 79.33199 | 42.50000 | 80.38464 | 81.51607 | 77.34841 | 78.07885 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 59.49169 | 42.75000 | 59.30267 | 58.69060 | 60.04891 | 59.92457 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.31542 | 42.75000 | 32.03602 | 31.23617 | 24.66870 | 33.32077 |



#### 精度正确基线，同窗口配对复测

原始 JSON：[bench_verified_baseline_pair_retry_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_verified_baseline_pair_retry_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**440.010 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.54800 | 44.00000 | 97.36526 | 98.29802 | 99.55160 | 98.97712 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.77255 | 44.00000 | 80.31593 | 80.18380 | 76.60448 | 77.98600 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 58.87124 | 44.00000 | 58.16193 | 58.36884 | 59.34696 | 59.60724 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.51236 | 44.00000 | 37.41686 | 25.96762 | 26.06396 | 32.60100 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**395.893 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 119.84361 | 43.00000 | 120.73491 | 119.60700 | 120.25889 | 118.77364 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.19003 | 43.00000 | 96.84774 | 98.01241 | 98.36949 | 99.53046 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 34.41877 | 43.25000 | 25.62779 | 39.15998 | 34.04916 | 38.83815 |
| `ep_combine_fused_0` | 17.32629 | 43.00000 | 17.09168 | 17.27651 | 17.62707 | 17.30990 |
| `ep_combine_fused_sync_0` | 7.36088 | 43.00000 | 8.64081 | 7.10527 | 6.86264 | 6.83479 |



#### 精度正确基线，启用 faulthandler 复测

原始 JSON：[bench_verified_baseline_debug_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_verified_baseline_debug_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**440.734 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.26398 | 44.00000 | 97.37536 | 98.00931 | 98.24802 | 99.42323 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.46411 | 43.50000 | 79.81222 | 78.19045 | 77.88632 | 77.96743 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 59.62169 | 44.00000 | 58.97498 | 59.31624 | 59.93921 | 60.25632 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.06266 | 44.25000 | 30.11853 | 23.78478 | 27.08338 | 39.26393 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**395.017 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 118.74241 | 42.25000 | 119.23813 | 117.77419 | 119.00444 | 118.95290 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.29975 | 42.25000 | 97.90976 | 98.16074 | 98.14915 | 98.97935 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 25.60028 | 42.50000 | 27.29936 | 25.24943 | 24.40337 | 25.44898 |
| `ep_combine_fused_0` | 17.44168 | 42.25000 | 17.40631 | 17.04402 | 17.31951 | 17.99687 |
| `ep_combine_fused_sync_0` | 7.54198 | 42.25000 | 8.18551 | 7.51060 | 7.44627 | 7.02553 |



#### scatter metadata hoist 优化候选

原始 JSON：[bench_scatter_hoist_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_scatter_hoist_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**438.638 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.21197 | 44.00000 | 97.82798 | 97.59711 | 98.77642 | 98.64638 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.45061 | 44.00000 | 77.68768 | 78.76742 | 78.86196 | 78.48538 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 58.96421 | 44.00000 | 58.46002 | 59.23784 | 58.94871 | 59.21027 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 32.86093 | 44.25000 | 24.52144 | 39.90589 | 38.05950 | 28.95691 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**393.494 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 144.95165 | 44.75000 | 119.31709 | 224.11427 | 118.32788 | 118.04736 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.56994 | 44.75000 | 98.01670 | 98.64150 | 98.55642 | 99.06515 |
| `ep_combine_fused_sync_0` | 96.46893 | 44.75000 | 124.27095 | 8.62850 | 114.33700 | 138.63928 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 28.03455 | 45.25000 | 29.53462 | 26.08304 | 27.31488 | 29.20565 |
| `ep_combine_fused_0` | 17.25526 | 44.75000 | 17.32768 | 17.61708 | 16.91931 | 17.15695 |



#### scatter metadata hoist 候选复测（未纳入基线）

原始 JSON：[bench_scatter_hoist_repeat_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_scatter_hoist_repeat_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**441.012 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 97.98292 | 43.75000 | 97.16939 | 97.21684 | 98.94182 | 98.60364 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 79.01275 | 43.75000 | 80.98139 | 80.13758 | 77.55780 | 77.37422 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 59.48328 | 43.75000 | 58.79793 | 58.95482 | 59.35520 | 60.82516 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 34.49452 | 44.25000 | 24.99336 | 40.51204 | 44.90362 | 27.56907 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**396.573 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter` | 119.23551 | 41.75000 | 119.90700 | 118.84740 | 118.91560 | 119.27205 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.32131 | 41.75000 | 97.34257 | 97.89974 | 99.12228 | 98.92063 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 25.38731 | 42.00000 | 24.87910 | 26.24280 | 24.89851 | 25.52883 |
| `ep_combine_fused_0` | 17.36903 | 41.75000 | 17.44408 | 17.68870 | 17.21444 | 17.12890 |
| `ep_combine_fused_sync_0` | 6.68957 | 41.75000 | 6.68808 | 7.03837 | 7.07374 | 5.95807 |



#### scatter descriptor GROUP1/2/3 复用候选

原始 JSON：[bench_scatter_dgroup_cache_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_scatter_dgroup_cache_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**439.513 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.45597 | 43.75000 | 97.33251 | 97.89610 | 99.38182 | 99.21346 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.13522 | 43.75000 | 78.72824 | 79.20329 | 76.17884 | 78.43048 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 59.41853 | 43.75000 | 58.67005 | 59.14110 | 60.17982 | 59.68317 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 35.59091 | 43.75000 | 44.78198 | 32.18663 | 41.39431 | 24.00071 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**395.249 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter_sdgcache` | 120.80365 | 44.00000 | 120.60083 | 121.90233 | 120.57776 | 120.13369 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.48068 | 44.00000 | 97.90514 | 97.94245 | 98.96059 | 99.11454 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 29.51743 | 44.00000 | 25.37588 | 34.76002 | 26.36020 | 31.57362 |
| `ep_combine_fused_0` | 17.43945 | 44.00000 | 17.41260 | 17.41400 | 17.12811 | 17.80310 |
| `ep_combine_fused_sync_0` | 6.98755 | 44.00000 | 7.57031 | 7.07562 | 6.88500 | 6.41927 |



#### scatter descriptor 复用，同窗口配对复测

原始 JSON：[bench_scatter_dgroup_cache_repeat_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_scatter_dgroup_cache_repeat_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**未记录**（本轮中途失败，未输出 `mega_moe summary`）。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.75865 | 46.00000 | 98.34137 | 98.01936 | 98.68607 | 99.98779 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.86368 | 45.75000 | 80.77698 | 79.68704 | 78.15598 | 76.83472 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 59.45561 | 45.75000 | 58.96080 | 58.69922 | 59.84767 | 60.31477 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.37675 | 46.00000 | 32.34037 | 31.20040 | 25.38102 | 32.58521 |



#### scatter descriptor 复用，启用 faulthandler 复测

原始 JSON：[bench_scatter_dgroup_cache_debug_tables.json](../../.codex_tmp/f01_1_dsr1_refactor_p192_20261013/bench_scatter_dgroup_cache_debug_tables.json)

**# mega_moe kernel profile [base]**

`per_layer_us` (base)：**439.283 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.35896 | 44.00000 | 97.37638 | 97.74834 | 99.38622 | 98.92489 |
| `mori_ep_combine_tdm_bf16_ws4_h7168_k8_64x8` | 78.77629 | 43.75000 | 79.13482 | 80.59471 | 77.45362 | 77.92202 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist` | 59.74016 | 44.00000 | 59.35593 | 59.74198 | 59.29933 | 60.56338 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 30.98759 | 44.25000 | 24.89931 | 36.22652 | 31.91173 | 30.91280 |

**# mega_moe kernel profile [fused]**

`per_layer_us` (fused)：**394.515 μs**。

| kernel | avg_us | calls | rank0_us | rank1_us | rank2_us | rank3_us |
|---|---:|---:|---:|---:|---:|---:|
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K2048_e64_cn4_cm1_prefetch_apre_persist_epscatter_sdgcache` | 120.21847 | 42.50000 | 121.09427 | 121.11193 | 119.35902 | 119.30867 |
| `a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_q1r6_cn4_cm1_prefetch_apre_persist` | 98.36490 | 42.50000 | 97.51580 | 97.77449 | 98.70944 | 99.45986 |
| `mori_ep_dispatch_tdm_fp4x2_ws4_h3584_k8_64x8` | 28.48048 | 42.75000 | 29.65038 | 31.77832 | 25.84356 | 26.64966 |
| `ep_combine_fused_0` | 17.39025 | 42.50000 | 16.88291 | 17.68983 | 17.03124 | 17.95700 |
| `ep_combine_fused_sync_0` | 7.01340 | 42.50000 | 7.02900 | 7.53420 | 6.76232 | 6.72809 |

