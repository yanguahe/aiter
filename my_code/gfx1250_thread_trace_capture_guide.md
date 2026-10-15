# gfx1250 thread trace（ATT）抓取与分析指南

本文整理 2026-09～10 工作历史中实际使用过的 thread trace（rocprofv3 Advanced Thread Trace，简称 ATT）抓取方法、命令、工具和排错经验。共三种方法：

| 方法 | 场景 | 容器与工具链特征 | 入口 | ATT 模式 |
|---|---|---|---|---|
| **A** | 单卡 kernel，`hyg_fyd1` 类容器 | 标准开发镜像；自编译 `rocprofv3` 1.1.0（ROCm 7.12）+ decoder 0.1.5（`decoder_new`）+ `comgr_new` | [`get_isa_runner_att.sh`](../my_code1/get_isa_runner_att.sh)、[`reproduce_compare.sh att`](reproduce_compare.sh) | dispatch-based，YAML 精确匹配 kernel symbol |
| **B** | 单卡 kernel，SDK wheel 类容器（在 `hyg_fyd_e2e` 中验证过；`dsr1_ep4` 属于同类环境） | PyTorch wheel 自带 TheRock ROCm SDK（`_rocm_sdk_core` / `_rocm_sdk_devel`，`rocprofv3` 1.3.5） | [`get_isa_runner_att_safe.sh`](../../.codex_tmp/gemm1_fused_quant_att/get_isa_runner_att_safe.sh) | dispatch-based，profiler、register library、decoder 必须来自同一 SDK |
| **C** | 多卡通信 kernel（四卡 EP，例如跨卡 P2P 写回的 `_persist_epscatter` GEMM2） | 与 B 相同（`dsr1_ep4`，或 `rocm/vllm-dev` 镜像的独立容器） | [`run_capture.sh ... device graph\|gemm2`](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/run_capture.sh) | device-based，自定义 rocprofiler-sdk tool，不做 kernel-dispatch interception，只在一个 rank 加载 |

两类容器的工具链不能混用：

- 在 B 类容器使用 A 的自编译 `rocprofv3` 1.1.0，会在 `import torch` 阶段 SIGABRT（两份 SDK 同时加载）。
- 在 A 类容器使用 B 的 `_safe` 脚本：2026-10-03 在 a07-3 的 `hyg_fyd1` 中因找不到 `librocprof-trace-decoder.so.0.2` 而失败，改用方法 A 的工具链后采集成功。

文中路径的链接相对于本文件所在的 `aiter/my_code/`。远端路径以 `/data/yanguahe/code/wk_sp1/` 为共享根目录。

---

## 0. 判断容器类型

```bash
which rocprofv3 && rocprofv3 --version
python3 -c 'import torch; print(torch.__version__)'
ls -l /opt/rocm
ls /usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin/rocprofv3 2>/dev/null   # dsr1_ep4、hyg_fyd_e2e
ls /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3 2>/dev/null    # rocm/vllm-dev 镜像
ls /data/yanguahe/code/wk_sp1/rocprof-install/bin/rocprofv3                            # 自编译工具（A）
```

| 现象 | 类型 |
|---|---|
| 存在 `_rocm_sdk_core`，PyTorch 版本形如 `2.11.0+rocm10.1.0a...`，`/opt/rocm` 指向 `_rocm_sdk_core` | B（`dsr1_ep4` 是这种情况） |
| `hyg_fyd1`，FlyDSL/LLVM/ATT 栈由 `setup_work_env.flydsl.0.3.4.1.sh` 构建 | A |

`hyg_fyd_e2e` 的判断依据是 `_safe` 脚本中的注释：「hyg_fyd_e2e links module_moe_asm against the ROCm SDK 1.3.5 register library shipped with the Python environment. The older self-built rocprofv3 1.1.0 loads a second SDK and aborts during registration.」

---

## 1. 通用概念

### 1.1 dispatch-based ATT 的 YAML 字段

```yaml
jobs:
  - kernel_include_regex: '^<转义后的完整 kernel symbol>$'
    kernel_exclude_regex:
    kernel_iteration_range: "[1]"
    output_file: out
    output_directory: <case>/thread_trace/simd<N>/kernel/rpf_v3
    output_format: [csv]
    truncate_kernels: false
    sys_trace: false
    advanced_thread_trace: true
    att_target_cu: 1
    att_shader_engine_mask: "0xf"
    att_simd_select: "<N>"
    att_buffer_size: "0x10000000"
    att_library_path: ["<decoder 目录>"]
```

| 字段 | 含义与用法 |
|---|---|
| `kernel_include_regex` | 必须是完整 symbol，并用 `^...$` 锚定；`.`、`$`、`+` 需要转义。profiler 汇总表可能截断 FlyDSL 生成的长名字，应从完整 JSON、`--kernel-trace --stats` 结果、code object 或生成逻辑中取完整名字 |
| `kernel_iteration_range` | 选中该 kernel 的第几次 dispatch。按 kernel 计数、从 1 开始（旧 SDK `tool.cpp:is_targeted_kernel`）。`[1]` 是第一次，`[8]` 用于跳过 warmup。它不是 persistent kernel 内部的 N-task 编号 |
| `att_target_cu` | profiler 的硬件单元选择，历史中固定为 `1`。与 EP rank、GPU 编号无关 |
| `att_shader_engine_mask` | `"0x1"` 只采 SE0，数据少，适合先打通流程；`"0xf"` 采 SE0–3，`analyze_att_capture.py` 需要全部 SE |
| `att_simd_select` | 每次运行只采一个 SIMD。SIMD0–3 要分四次独立运行，每次都重新做 GPU 空闲检查 |
| `att_buffer_size` | 每个 SE 的 buffer，历史中用 `0x10000000`（256 MiB） |
| `att_library_path` | trace decoder 所在目录，必须支持 gfx1250，并与 profiler 版本匹配 |
| `kernel_trace`、`hip_runtime_trace` | 多卡场景中用于排查，单卡脚本没有设置 |
| `att_no_intercept` | quick-scan 模式：首次上传 code object 后扫描，不能圈选特定区间，不适合抓稳态 kernel |

硬件限制：根据 MI400 Shader Programming Guide 第 8 章（印刷页 287），每个 SE 同一时间只在一个 WGP 上采集详细指令。即使 SIMD0–3 全部采集，也不是所有 WG 的完整记录。

### 1.2 workload 和计时

- workload 尽量小，但必须保持同一个 kernel 配置。例如 MoE 单测用 `--iters 2`，四卡 benchmark 用 `--warmup 5 --iters 2`。
- ATT 期间输出的耗时是 instrumentation 产物，不能作为性能数据。性能数据来自空闲 GPU 上的正常 profiler 运行。
- 每次采集前后都做整机 GPU 空闲检查（见第 6 节）。

### 1.3 dispatch-based 的输出结构

```text
<case>/thread_trace/simd<N>/kernel/rpf_v3/
  *.att                                   # 每个 SE 一份 raw trace
  ui_output_agent_<id>/code.json          # PC → ISA 映射与逐指令统计
  ui_output_agent_<id>/se*_sm*_sl*_wv*.json   # 每个 wave 的指令时间线
  ui_output_agent_<id>/realtime.json、occupancy.json 等
```

- `code.json["code"]` 的每一行：`[ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle]`。
- wave JSON 中的指令在 `payload["wave"]["instructions"]`，每条为 `[timestamp, type, stall, latency, code_idx]`。wave JSON 可能很大，先打印 keys 和小样本，不要整份输出。

---

## 2. 方法 A：`hyg_fyd1` 类容器（自编译工具链）

### 2.1 工具链

| 组件 | 远端路径 | 说明 |
|---|---|---|
| rocprofv3 | `/data/yanguahe/code/wk_sp1/rocprof-install/bin/rocprofv3` | 1.1.0，rocm-systems `07ce631b1384f6c9aefc6244e8c481fa48ae8062`，ROCm 7.12，带 gfx1250 修复 |
| decoder | `/data/yanguahe/code/wk_sp1/decoder_new/librocprof-trace-decoder.so` | 0.1.5，已验证能解码 gfx1250；不要让脚本下载 0.1.6 |
| COMGR | `/data/yanguahe/code/wk_sp1/comgr_new/libamd_comgr.so.3 -> libamd_comgr.so.3.5.0` | 基于 FlyDSL 的 LLVM 构建，能识别 gfx1250 新指令；旧 COMGR 会把新指令解成 `.long` |
| 环境脚本 | `/data/yanguahe/code/wk_sp1/rocprof_env.sh` | 把 `/opt/rocm/lib` 加入 `LD_LIBRARY_PATH`（HSA runtime 以裸名 dlopen `libhsa-amd-aqlprofile64.so`），并前置 `comgr_new` |
| llvm-objdump | `/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/llvm-objdump` | 支持 gfx1250 的反汇编 |

这套工具由 FlyDSL v0.3.4.1 的一键脚本构建，版本组合和重建流程见 skill `C:/Users/yanguahe/.codex/skills/rebuild-flydsl-att-env/`（`SKILL.md`、`references/known-good-v0.3.4.1.md`、`references/toolchain-rebuild.md`）：

```bash
cd /data/yanguahe/code
./setup_work_env.flydsl.0.3.4.1.sh setup-new \
  --target-dir /data/yanguahe/code/wk_sp1 \
  --container-name hyg_fyd1 \
  --workdir /data/yanguahe/code/wk_sp1
```

`setup-new` 会新建或替换工作区与容器，属于破坏性操作，执行前必须列出确切目标并获得用户明确批准。只抓 trace 时先验证已安装的工具链，不要因为一次采集失败就重建整个工具链。

在容器内手动设置环境：

```bash
source /data/yanguahe/code/wk_sp1/rocprof_env.sh
export PATH=/data/yanguahe/code/wk_sp1/rocprof-install/bin:$PATH
export ROCPROF_ATT_LIBRARY_PATH=/data/yanguahe/code/wk_sp1/decoder_new
rocprofv3 --version
```

### 2.2 一键脚本 `get_isa_runner_att.sh`

本地位置：[`aiter/my_code1/get_isa_runner_att.sh`](../my_code1/get_isa_runner_att.sh)。使用前按任务需要上传到远端（2026-09-30 的采集上传到了 `/tmp/get_isa_runner_att.sh`）。

```text
用法（容器内采集）:
  get_isa_runner_att.sh <KERNEL_NAME> <output-dir-name> "<TEST_CMD>" [--all-simd] [--ana-att]
用法（主机上提交 trace 目录）:
  get_isa_runner_att.sh <output-dir-name> --git [--am]
```

脚本流程：

1. source `rocprof_env.sh`，固定自编译 `rocprofv3` 和 `decoder_new`。
2. 清空 `~/.flydsl/cache/*`，避免旧缓存让 trace 抓到之前编译的 tile 配置。
3. `rocprofv3 --kernel-trace --stats` 运行一次，作为预检。
4. 为每个 SIMD 生成一份 YAML（`kernel_iteration_range: "[1]"`、`att_target_cu: 1`、`att_shader_engine_mask: "0xf"`、buffer `0x10000000`），依次运行 `rocprofv3 -i <yaml> -- <TEST_CMD>`。不带 `--all-simd` 时只采 SIMD3。
5. 检查每个 SIMD 是否生成 `.att`，打包为 `${TRACE_ROOT}/<output-dir-name>/` 和同名 `.tar.gz`，并 `chmod -R a+rwX`。
6. 带 `--ana-att` 时运行 `python -u my_code/analyze_att_capture.py --dir <output_dir> --no-plot`。

注意：

- `TEST_CMD` 必须是单行命令；脚本会额外注入 `env -u FLYDSL_DUMP_DIR PYTORCH_ALLOC_CONF=expandable_segments:True GPU_ARCHS=gfx1250 ENABLE_CK=0 AITER_FORCE_GFX1250=1 FLYDSL_DUMP_IR=0`。
- `--git` 模式在主机执行，脚本中 push 的分支写死为 `hyg_gfx1250_gemm`，使用前先核对。

**例 1：FlyDSL MoE GEMM1（2026-09-30，a07-3 `hyg_fyd1`，E64/T1536/topk8）**

```bash
cd /data/yanguahe/code/wk_sp1/aiter
REPO_ROOT=/data/yanguahe/code/wk_sp1/aiter \
TRACE_ROOT=my_code/thread_trace_runs \
bash /tmp/get_isa_runner_att.sh \
  a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_bth6_rcw_mg4_fc20_xdl0_reuse_ostore2p_s3 \
  e64_t1536_topk8_gemm1_bth6_att_20260930 \
  "env AITER_MOE_EXPERT_BALANCE=true AITER_LOG_MORE=1 AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --iters 2 --const-init 0" \
  --all-simd \
  --ana-att
```

GEMM2 只需替换 symbol（当时为 `a8w4_tdm_fp4_t192x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc20_ostore2p_s3_ow2`）和输出目录名。这次采集的分析报告是 `hyg/moe_a4w4_pr` 分支中的 `my_code/e64_t1536_topk8_gemm1_gemm2_thread_trace_analysis_a07_3.md`（可用 `git show hyg/moe_a4w4_pr:my_code/e64_t1536_topk8_gemm1_gemm2_thread_trace_analysis_a07_3.md` 查看）。

**例 2：只重抓一个解码失败的 SIMD**

GEMM2 的 SIMD0 第一次解码报 `ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA`，`code.json`、`occupancy.json` 为空。做法是复用原 YAML，只改输出目录，单独重抓 SIMD0；失败目录保留为 `simd0_decode_error`，不重跑已经成功的 SIMD1–3：

```bash
docker exec hyg_fyd1 bash -lc '
set -euo pipefail
cd /data/yanguahe/code/wk_sp1/aiter
root=my_code/thread_trace_runs/e64_t1536_topk8_gemm2_att_20260930
retry="$root/thread_trace/simd0_retry/kernel/rpf_v3"
rm -rf "$root/thread_trace/simd0_retry"; mkdir -p "$retry"
sed "s#output_directory: .*#output_directory: $retry#" "$root/input_kernel_simd0.yaml" > /tmp/e64_gemm2_simd0_retry.yaml
source /data/yanguahe/code/wk_sp1/rocprof_env.sh
export PATH=/data/yanguahe/code/wk_sp1/rocprof-install/bin:$PATH
export ROCPROF_ATT_LIBRARY_PATH=/data/yanguahe/code/wk_sp1/decoder_new
rocprofv3 -i /tmp/e64_gemm2_simd0_retry.yaml -- \
  env -u FLYDSL_DUMP_DIR PYTORCH_ALLOC_CONF=expandable_segments:True GPU_ARCHS=gfx1250 ENABLE_CK=0 \
  AITER_FORCE_GFX1250=1 FLYDSL_DUMP_IR=0 AITER_MOE_EXPERT_BALANCE=true AITER_LOG_MORE=1 \
  AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py --scenario bench --data-format a4w4 \
  --act silu --no-bias --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 \
  --model-dim 7168 --inter-dim 2048 --iters 2 --const-init 0
' 2>&1 | tee /tmp/e64_gemm2_simd0_retry.log
```

### 2.3 `reproduce_compare.sh` 的 `att` 模式

[`reproduce_compare.sh`](reproduce_compare.sh) 把同一套 case 表同时用于 e2e benchmark 和 ATT，避免两者配置不一致。它面向 E96/T16384 workload，`EXPECTED_BRANCH` 默认为 `hyg/moe_a4w4_pr`，直接在容器内运行。

```bash
# 只验证 ATT workload 能运行，不 trace
CASE_LIST=<case> bash my_code/reproduce_compare.sh att-validate [--gemm2]

# 采集 SIMD0–3、全部 SE
AITER_ATT_SHADER_ENGINE_MASK=0xf AITER_ATT_SIMD_LIST=0,1,2,3 \
CASE_LIST=<case> bash my_code/reproduce_compare.sh att [--gemm2]

# e2e benchmark 结束后接着采 ATT
CASE_LIST=<case> ROUNDS=1 RUN_VERIFY=1 RUN_ATT=1 bash my_code/reproduce_compare.sh e2e-const0 [--gemm2]
```

| 环境变量 | 默认值 |
|---|---|
| `AITER_ATT_E2E_ITERS` | `2` |
| `AITER_ATT_KERNEL_ITERATION_RANGE` | `[8]` |
| `AITER_ATT_SIMD_LIST` | `0,1,2,3` |
| `AITER_ATT_TARGET_CU` | `1` |
| `AITER_ATT_SHADER_ENGINE_MASK` | `0x1` |
| `AITER_ATT_BUFFER_SIZE` | `0x10000000` |
| `AITER_ATT_TIMEOUT_SECONDS` | `300` |
| `AITER_ATT_LIBRARY_PATH` | `/data/yanguahe/code/wk_sp1/decoder_new` |
| `AITER_ROCPROF_ENV` | `/data/yanguahe/code/wk_sp1/rocprof_env.sh` |
| `AITER_ROCPROF_BIN_DIR` | `/data/yanguahe/code/wk_sp1/rocprof-install/bin` |

流程：先不带 profiler 跑一次预检，从日志 `gemm1: device_time_avg=... symbol=<完整名>`（或 gemm2）提取 symbol；再逐个 SIMD 写 YAML、采集、校验（至少一个非空 `.att`、正好一个 `code.json`、至少一个 wave JSON）。只有 SE mask 为 `0xf` 时才自动运行 `analyze_att_capture.py`。脚本用 `/tmp/aiter_reproduce_<target>_att_<UID>.lock` 防止并发采集，并固定 `HIP_VISIBLE_DEVICES=0`。

### 2.4 纯汇编 kernel（ISA runner）的变体

**问题**：在 `rocprofv3` 注入的环境中调用 clang 编译汇编，会触发 `clang-23 SmallPtrSet` assertion。

**做法 1：先编译，再用最小 launcher 采集。** 在普通环境中预编译 `.co`，trace 时的测试命令只加载已编译的 code object 并启动一次（历史中为 `att_launch_opt.py`，由 `run_att_const0.sh` 调用 `get_isa_runner_att.sh ... --ana-att`）。

**做法 2：clean clang wrapper**（[`my_code1/moe_gemm1_act1_optimized/benchmark_history.sh`](../my_code1/moe_gemm1_act1_optimized/benchmark_history.sh) 的 `att` 模式）：

- 生成 `clang_clean_for_att.sh`，在调用 clang 前 `unset LD_PRELOAD HSA_TOOLS_LIB ROCP_TOOL_LIBRARIES ROCPROFILER_TOOL_LIBRARIES ROCPROFILER_LIBRARY_CTOR`。
- 通过 `sitecustomize.py` 把 runner 的 `DEFAULT_CLANG` 指向这个 wrapper，并在启动前检查是否生效。
- 入口：`bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh att`，或 `att-validate`（只编译、启动，不 trace）。该目录当时位于 `my_code/`，现在本地在 `my_code1/` 下。

`--ana-att` 的输出可以得到 REALTIME 测得的 kernel wall time、occupancy 中 shader timestamp 的最大跨度、由两者推导的 GFXCLK、每个物理 slot 顺序执行的 wave 数，以及物理 WGP 之间的完成时间不均衡度。历史中的硬件 cycle 上限分析就基于这些数据，见 [`HARDWARE_CYCLE_LIMIT.md`](../my_code1/moe_gemm1_act1_optimized/HARDWARE_CYCLE_LIMIT.md) 和 [`PERSISTENT_MODE_THREAD_TRACE_ANALYSIS.md`](../my_code1/moe_gemm1_act1_optimized/PERSISTENT_MODE_THREAD_TRACE_ANALYSIS.md)。

---

## 3. 方法 B：`hyg_fyd_e2e` / `dsr1_ep4` 类容器（SDK 对齐工具链）

### 3.1 为什么不能使用自编译工具

| 现象 | 原因 |
|---|---|
| `import torch` 时 SIGABRT：`rocprofiler_at_intercept_table_registration(...) api registration failed with error code 16: Configuration request occurred outside of valid rocprofiler configuration period` | 自编译的 rocprofiler-sdk 与 PyTorch runtime 自带的 SDK 同时加载 |
| decoder 0.1.5 警告「older than 0.2 cannot emit ATT event or dispatch records」 | 新 SDK 需要 0.2 及以上的 decoder |

结论：profiler、register library 和 decoder 必须来自同一个 SDK。COMGR 仍然前置 `comgr_new`，以正确解码 gfx1250 新指令。

### 3.2 工具链路径

| 组件 | `hyg_fyd_e2e`（`_safe` 脚本默认值） | `dsr1_ep4` | `rocm/vllm-dev` 独立容器 |
|---|---|---|---|
| rocprofv3 | `/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/bin/rocprofv3`（SDK 1.3.5） | 同左；1.3.5，revision `44405e956f3708ccd53f244666f7fd768eea6bbc`，ROCm 10.1 | `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/bin/rocprofv3`，revision `d134b7823f61f46816022525b98f128f5128ded9` |
| decoder | `_rocm_sdk_core/lib/librocprof-trace-decoder.so.0.2`，通过 symlink 目录 `/tmp/codex_rocprof_trace_decoder_0_2/` 提供 | `/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel/lib/librocprof-trace-decoder.so` | `/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/librocprof-trace-decoder.so` |
| COMGR | `/data/yanguahe/code/wk_sp1/comgr_new` | 同左 | 同左 |

`dsr1_ep4` 中 `/opt/rocm` 是 `_rocm_sdk_core` 的 symlink。未经 source 的 shell 默认找到的就是 bundled `rocprofv3`。

### 3.3 环境设置要点

```bash
SDK_CORE=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core
SDK_DEVEL=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel
export PATH="$SDK_CORE/bin:$PATH"
export LD_LIBRARY_PATH="/data/yanguahe/code/wk_sp1/comgr_new:$SDK_DEVEL/lib:$SDK_CORE/lib:$SDK_CORE/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
export ROCPROF_ATT_LIBRARY_PATH="$SDK_DEVEL/lib"
```

- 必须把 `_rocm_sdk_devel/lib` 加入 `LD_LIBRARY_PATH`。core 目录只有 `libhsa-amd-aqlprofile64.so.1`，裸名 `.so` 在 devel 目录。否则 AITER 导入时启动的 `rocminfo` 子进程会继承 ATT 环境并以 `aqlprofile API table load failed` abort，AITER 随后报 `No gfx arch found in rocminfo output`。
- `_safe` 脚本采用另一种办法：在 `/tmp/codex_att_fakebin/` 放一个假的 `rocminfo`，直接输出已验证的 `Name: gfx1250`。理由是 ATT 占用设备时 `rocminfo` 无法枚举 agent。
- 每次 `rocprofv3` 调用都包上 `timeout --signal=TERM --kill-after=20s 300`，防止采集挂住。

### 3.4 `_safe` 脚本用法（`hyg_fyd_e2e`）

本地位置：[`.codex_tmp/gemm1_fused_quant_att/get_isa_runner_att_safe.sh`](../../.codex_tmp/gemm1_fused_quant_att/get_isa_runner_att_safe.sh)。命令行与方法 A 的 `get_isa_runner_att.sh` 相同：

```bash
docker exec -it hyg_fyd_e2e bash
cd /app/aiter
REPO_ROOT=/app/aiter TRACE_ROOT=my_code/thread_trace_runs \
bash <上传位置>/get_isa_runner_att_safe.sh \
  <完整 kernel symbol> <output-dir-name> "<单行测试命令>" --all-simd --ana-att
```

| 环境变量 | 作用 |
|---|---|
| `AITER_ATT_SDK_ROOT` | 覆盖 `_rocm_sdk_core` 路径 |
| `AITER_ATT_SDK_DEVEL_ROOT` | 覆盖 `_rocm_sdk_devel` 路径 |
| `AITER_ATT_TIMEOUT_SECONDS` | 每次采集的超时，默认 300 |

找不到 `${sdk_root}/lib/librocprof-trace-decoder.so.0.2` 时脚本报 `Missing SDK-aligned ATT decoder` 并退出；这说明当前不是 B 类容器，应改用方法 A。

这份脚本来自同期的 GEMM1 fused-quant 工作线（远端上传目录为 `.codex_upload_gemm1_fused_apre_quant/`）。该工作线的 all-SIMD trace 保存在 `/app/aiter/my_code/thread_trace_runs/gemm1_fused_v115_att_all_simd_a07_reboot_20261003`，记录见 [`gemm1_fused_apre_quant_optimization_log.md`](gemm1_fused_apre_quant_optimization_log.md)。`/app/aiter` 是 `hyg_fyd_e2e` 中的仓库路径。

### 3.5 在 `dsr1_ep4` 中直接使用 bundled rocprofv3

可以按方法 A 的 YAML 写法，把 `rocprofv3` 换成 bundled 版本、`att_library_path` 换成 `_rocm_sdk_devel/lib`，并按 3.3 设置环境。但历史中在 `dsr1_ep4` 运行四卡 Mori workload 时，dispatch-based 采集出现 GPU fault（见 4.1），后来改用 device-based 方法。单卡 workload 在 `dsr1_ep4` 中的 dispatch-based 采集，历史中没有单独验证。

---

## 4. 方法 C：多卡通信 kernel 的 device-based ATT

### 4.1 为什么需要单独的方法

| 问题 | 结论 |
|---|---|
| `bench_mega_moe.py --profile_table 0` 只控制表格打印，仍会执行 `torch.profiler`（Kineto） | 外层 ATT 不能再套 Kineto，需要单独的 runner |
| 把 `rocprofv3` 套在 `torchrun` 外面，无法确定到底采了哪个 rank | 用 rank wrapper，只给选定 rank 加 profiler，其他 rank 正常参与通信 |
| `--save_trace 1` 导出的是 Kineto/Chrome kernel timeline | 不是指令级 thread trace |
| dispatch-based ATT（kernel-dispatch interception）在四卡 Mori workload 中失败：`[queue-intercept] device-memory ring-buffer requested but profiling requires system-memory InterceptQueue; falling back to system-memory ring ...`，随后 `Memory access fault by GPU node-2 ... on address (nil)` 和 `Timeout while waiting for queue sync: 976 kernels still active`，rank0 SIGABRT | queue interception 是嫌疑，未证实因果；改用不配置 dispatch interception 的 device-based ATT |
| `--att-no-intercept` 只做首次上传 code object 后的 quick scan | 不能圈选稳态区间 |
| `--att-serialize-all` | 会串行化所有 kernel，多卡通信可能依赖并发，不要盲目启用 |

device-based 方法：自定义 rocprofiler-sdk tool 调用 `rocprofiler_configure_device_thread_trace_service`，同时注册 code object 加载回调，不配置任何 kernel-dispatch tracing/interception service。tool 导出 `aiter_att_start()` / `aiter_att_stop()`，由 Python runner 通过 ctypes 在目标区间前后调用。只有选定的 rank 通过 `LD_PRELOAD` 加载 tool，四个 rank 都运行原 workload。

### 4.2 组件

| 文件（本地） | 作用 |
|---|---|
| [`trace_mega_moe_rank.sh`](trace_mega_moe_rank.sh) | 通过 `torchrun --no-python` 启动的 rank wrapper。只给 `AITER_ATT_LOCAL_RANK` 指定的 rank 加 profiler 或 device tool；设置 `AITER_SMI_MONITOR=0`；`AITER_ATT_DISABLE=1` 时所有 rank 都不加 profiler |
| [`trace_mega_moe_without_profiler.py`](trace_mega_moe_without_profiler.py) | 替换 `DeviceMoEPipeline.bench`：保留原模型、routing、graph 和 kernel 选择，关闭 Kineto，报告的时间为 0；负责调用 start/stop；支持 `graph`、`gemm2` 两种圈选区间和隔离诊断副本 |
| [`trace_mega_moe_agent.cpp`](trace_mega_moe_agent.cpp) | device ATT tool（`libtrace_mega_moe_agent.so`）。按 PID 写独立目录，保存 raw `.att`、`chunks.jsonl`、所有 code object 及其索引 |
| [`decode_mega_moe_agent.cpp`](decode_mega_moe_agent.cpp) | 离线 decoder，输出每个 wave 的 timestamp、stall、duration、PC/code object，以及 occupancy、realtime、frequency、INFO、event、dispatch 记录 |
| [`analyze_mega_moe_device_att.py`](analyze_mega_moe_device_att.py) | 用 llvm-objdump 把 PC 映射到 ISA，生成 `code.json`、UI 格式的目标 wave、`analysis.json` 和代表性 trace |
| [`.codex_tmp/f01_1_dsr1_gemm2_att_20261014/`](../../.codex_tmp/f01_1_dsr1_gemm2_att_20261014/) | 主机侧 helper：`run_capture.sh`、`build_agent_tools.sh`、`probe_device_tool.sh`、`decode_capture.sh`、`analyze_capture.sh`、`process_all.sh`、`run_diagnostic_resume_20261014.sh` |
| [`.codex_tmp/f01_2_dsr1_gemm2_att_20261014/`](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/) | 备用机器独立容器的版本：`att_env.sh`、`run_capture.sh`、`prepare_diagnostic_scripts.py`、诊断副本生成器 `make_*.py` |

远端任务根目录（f01-1）：`/data/yanguahe/code/wk_sp1/.codex_tmp/f01_1_dsr1_gemm2_att_20261014`，以下记作 `$TASK_ROOT`。rank wrapper 和 runner 需要放在容器仓库的 `my_code/` 下（`/app/aiter_moe_a4w4_pr_refactor_wt/my_code/`）。

### 4.3 构建与导入探针

在 `dsr1_ep4` 容器内用 bundled SDK 的头文件和库编译（`build_agent_tools.sh`）：

```bash
SDK_CORE=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core
SDK_DEVEL=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel
cd "$TASK_ROOT/helpers"
g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ -fPIC -shared trace_mega_moe_agent.cpp \
    -I "$SDK_DEVEL/include" "$SDK_CORE/lib/librocprofiler-sdk.so.1" \
    -Wl,-rpath,"$SDK_CORE/lib" -o libtrace_mega_moe_agent.so
g++ -std=c++17 -O2 -D__HIP_PLATFORM_AMD__ decode_mega_moe_agent.cpp \
    -I "$SDK_DEVEL/include" "$SDK_CORE/lib/librocprofiler-sdk.so.1" \
    -Wl,-rpath,"$SDK_CORE/lib" -o decode_mega_moe_agent
```

导入探针（`probe_device_tool.sh`，只 `import torch`，不运行 GPU workload）：

```bash
export LD_LIBRARY_PATH="/data/yanguahe/code/wk_sp1/comgr_new:$SDK_DEVEL/lib:$SDK_CORE/lib:${LD_LIBRARY_PATH:-}"
export AITER_ATT_OUTPUT="$TASK_ROOT/device_import_probe"
export AITER_ATT_GPU_INDEX=0
export ROCPROFILER_LIBRARY_CTOR=1
export ROCP_TOOL_LIBRARIES="$TASK_ROOT/helpers/libtrace_mega_moe_agent.so"
export LD_PRELOAD="$ROCP_TOOL_LIBRARIES:$SDK_CORE/lib/librocprofiler-sdk.so.1"
python3 -c 'import torch; print("DEVICE_TOOL_IMPORT_OK", torch.__version__)'
# 期望输出：[device ATT] initialized without dispatch interception / DEVICE_TOOL_IMPORT_OK ...
```

### 4.4 采集

在 f01-1 主机的 `$TASK_ROOT` 中运行（建议放进固定名称的 tmux session）：

```bash
bash run_capture.sh <base|fused> <SIMD> <SE_MASK> <ITERATION> <LABEL> [att|none|timeline|quick|device] [graph|gemm2]
```

| 参数 | 含义 |
|---|---|
| `base` / `fused` | 采普通 `_persist` GEMM2，或带跨 rank 写回的 `_persist_epscatter` GEMM2（对应 benchmark 的 `--combine base/fused`） |
| `SIMD`、`SE_MASK` | 采集的 SIMD 和 SE mask，历史中 SE mask 用 `0xf` |
| `ITERATION` | 写入 YAML 的 `kernel_iteration_range`，只对 dispatch-based 模式有效 |
| `LABEL` | case 目录名，每次必须是新名字（脚本用 `mkdir`，已存在会失败，避免覆盖旧证据） |
| `att` / `timeline` / `quick` | dispatch-based ATT / 只做 kernel trace / `att_no_intercept` quick scan |
| `none` | 不加 profiler，只检查 workload 能否在当前环境跑通 |
| `device` | device-based ATT（四卡场景实际成功的方式） |
| `graph` / `gemm2` | 圈选区间，见下文 |

`run_capture.sh` 的步骤：

1. **空闲门禁**：循环执行 `gpu_users.sh || true`（该脚本在空闲时也返回 1）和独立的 sysfs/KFD 检查：KFD 进程为空、正好 4 张 GPU、`gpu_busy_percent == 0`、VRAM 低于 256 MiB；不满足就 `sleep 6` 后重试。
2. 记录 `started` 时间，`docker exec` 进入 `dsr1_ep4`，在 `/app/aiter_moe_a4w4_pr_refactor_wt` 生成 YAML，保存生产源码 SHA256 的 `source_manifest.json`。
3. 运行四卡 workload（下面的命令）。
4. 记录 `exit`、`finished`，在容器内 `chmod -R a+rwX` case 目录，再做一次结束后的 GPU 检查。

容器内实际执行的 workload（`device` + `graph`）：

```bash
cd /app/aiter_moe_a4w4_pr_refactor_wt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
AITER_ATT_CONFIG="$CASE_ROOT/att.yaml" \
AITER_ATT_LOCAL_RANK=0 AITER_ATT_DISABLE=0 AITER_ATT_ENGINE=device \
AITER_ATT_DEVICE_LIBRARY="$TASK_ROOT/helpers/libtrace_mega_moe_agent.so" \
AITER_ATT_OUTPUT="$CASE_ROOT/thread_trace/simd$SIMD/kernel/rpf_v3/device" \
AITER_ATT_GPU_INDEX=0 AITER_ATT_SIMD="$SIMD" AITER_ATT_SE_MASK=0xf \
AITER_ATT_REGION=graph \
MORI_V2_KERNEL_BACKEND=hip MORI_EP_TOKOFF_EXT="${MORI_EP_TOKOFF_EXT:-0}" \
MEGA_DISPATCH=mori FLYDSL_DUMP_IR=0 \
MORI_GPU_ARCHS=gfx1250 FLYDSL_GPU_ARCH=gfx1250 \
AITER_FORCE_A8W4=0 AITER_MOE_EXPERT_BALANCE=True PYTHONUNBUFFERED=1 \
torchrun --standalone --nproc_per_node=4 --no-python \
    ./my_code/trace_mega_moe_rank.sh \
    -e 256 -k 8 -hd 7168 -id 2048 --layers 16 \
    -tpr 1536 -q a4w4_mxfp4 --combine fused \
    --warmup 5 --iters 2 --prof_replays 0 \
    --acc_verify 0 --profile_table 0 --save_trace 0 --dispatch_wire fp4
```

要点：

- 不能设置 `HIP_VISIBLE_DEVICES=0`，所有 rank 都需要访问四张卡。
- 交互 shell 中 `/etc/bash.bashrc` 会把 `/app/aiter`（main）放到 `PYTHONPATH` 前面，必须先导出仓库自身路径，并用 `python3 -c 'import aiter; print(aiter.__file__)'` 确认。
- wrapper 在 `device` 模式下设置：

  ```bash
  ROCPROFILER_LIBRARY_CTOR=1
  ROCP_TOOL_LIBRARIES=$AITER_ATT_DEVICE_LIBRARY
  LD_PRELOAD=$AITER_ATT_DEVICE_LIBRARY:$SDK_CORE/lib/librocprofiler-sdk.so.1
  ```

  并前置 `comgr_new`、decoder 目录、`$SDK_CORE/lib`、`$SDK_CORE/lib/rocm_sysdeps/lib`。即使在 `device` 模式，wrapper 也会检查 `AITER_ATT_CONFIG`、profiler、环境脚本（`AITER_ATT_ENV`，默认 `/data/yanguahe/code/wk_sp1/rocprof_env.sh`）和 decoder 是否存在。

**圈选区间：**

| `AITER_ATT_REGION` | 行为 | 代表性 |
|---|---|---|
| `graph`（默认） | 先 replay 5 次 graph 并同步、`comm.barrier()`；选定 rank 调 `aiter_att_start()`，trace 一次完整的 graph replay（16 层全部 kernel），同步后 `aiter_att_stop()`；第二次 replay 不 trace | 原 graph 稳态 |
| `gemm2` | graph setup 与 warmup 照常执行；之后所有 rank 额外执行一层 eager `_layer_step(self.x0_static, 0)`；hook `grouped_gemm_mxfp4.flydsl_grouped_gemm_a8w4_masked`，只圈选第一次 `K == 2048` 的 GEMM2，前后 `torch.cuda.synchronize()` | 真实 EP 输入和 live window，但这是额外的 eager 层，与 graph 稳态不同，必须在结论中注明 |

`kernel_iteration_range` 对 device API 不起作用，不能声称采到的是「第 129 次 GEMM2」。graph replay 中包含 16 层，分析时依靠 code object 和完整 symbol 筛选目标 kernel 的 wave。

**device tool 参数（环境变量）：**

| 变量 | 默认值 | 含义 |
|---|---|---|
| `AITER_ATT_OUTPUT` | 必填 | 输出根目录，tool 写入 `<AITER_ATT_OUTPUT>/pid<PID>/` |
| `AITER_ATT_GPU_INDEX` | `0` | agent 的 `logical_node_type_id` |
| `AITER_ATT_TARGET_CU` | `1` | 目标 CU |
| `AITER_ATT_SE_MASK` | `0xf` | SE mask |
| `AITER_ATT_SIMD` | `0` | SIMD 选择 |
| `AITER_ATT_BUFFER_BYTES` | `0x10000000` | 每个 SE 的 buffer |

每个 PID 写独立目录，是为了避免继承了 `LD_PRELOAD` 的子进程（例如 `ld.lld`）混写：

```text
<AITER_ATT_OUTPUT>/pid<PID>/
  capture1_se0.att ... capture1_se3.att
  chunks.jsonl
  code_object_<id>.out
  code_objects.jsonl
  code_objects.tsv
```

**一组完整采集的示例（SIMD0–3，两个版本各一次）：**

```bash
bash run_capture.sh fused 0 0xf 129 fused_device_simd0_graph device graph
bash run_capture.sh base  0 0xf 129 base_device_simd0_graph  device graph
# SIMD1–3：把 SIMD 号和 LABEL 换成对应值，各自独立运行
bash run_capture.sh fused 0 0xf 129 fused_device_simd0_eager device gemm2
bash run_capture.sh base  0 0xf 129 base_device_simd0_eager  device gemm2
```

### 4.5 解码与分析（CPU 任务，不需要 GPU 空闲）

```bash
docker exec dsr1_ep4 bash "$TASK_ROOT/helpers/decode_capture.sh"  <LABEL> <SIMD>
docker exec dsr1_ep4 bash "$TASK_ROOT/helpers/analyze_capture.sh" <LABEL> <SIMD> <NEW_OUTPUT_NAME>
# 批量处理：bash "$TASK_ROOT/helpers/process_all.sh"
```

`docker exec` 必须使用绝对脚本路径，容器默认 cwd 与主机不同。

`decode_capture.sh` 设置 `LD_LIBRARY_PATH` 后调用：

```bash
decode_mega_moe_agent <pid_dir> <pid_dir>/decoded <SDK_DEVEL>/lib \
  0:<pid_dir>/capture1_se0.att 1:<pid_dir>/capture1_se1.att \
  2:<pid_dir>/capture1_se2.att 3:<pid_dir>/capture1_se3.att
```

- 输出目录必须是新目录。返回值：0 成功，2 解码错误，3 没有 wave。标准输出最后一行是 `DECODED_WAVES=<n> INFO_RECORDS=<n>`。
- 输出文件：`se<SE>_wave<N>.json`，以及 `occupancy.jsonl`、`realtime.jsonl`、`frequency.txt`、`info.jsonl`、`events.jsonl`、`dispatches.jsonl`。
- wave JSON：`{"se","cu","simd","slot","wg","cluster","contexts","begin","end","instructions":[[time,category,stall,duration,pc_address,code_object_id],...],"timeline":[[state,duration],...]}`。
- 时间语义（decoder header，gfx10+）：`time + stall` 是成功 issue 的时刻，`duration = stall + execution`，timestamp 单位为 shader cycles。duration 不能直接累加成串行耗时，区间耗时用 timestamp 差计算。

`analyze_capture.sh` 调用：

```bash
python3 "$TASK_ROOT/helpers/analyze_mega_moe_device_att.py" <pid_dir> \
  --kernel <完整 symbol> \
  --objdump /data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/llvm-objdump \
  --output <case>/<NEW_OUTPUT_NAME>
```

输出 `code.json`、转换成 UI 格式的目标 wave（`se*_sm*_sl*_wv*.json`，与 dispatch-based 输出的格式一致，便于复用现有分析脚本）、`target_waves.json`、`analysis.json`、`representative_trace.txt`、`code_object_<id>.disasm`。分析 persistent kernel 时，按 persistent 外层 backedge 切分 N-task，不要把内层 K-loop 当成 task（早期 v2 版本犯过这个错误）。

### 4.6 验收标准

一个 case 只有全部满足以下条件才算采集成功：

- `exit=0`，4 份 raw（SE0–3）都非空；
- `chunks.jsonl` 中 `flags=1`（END），没有 buffer-full 位（2/4）；
- decoder `INFO_RECORDS=0`；保存 events 时没有 `PACKET_LOSS` 或 `TT_STALL_BEGIN/END`；
- analyzer `warnings=[]`、`missing_pcs=[]`，完整 symbol 命中，反汇编中没有 `.long` / `v_illegal`；
- 区分有效 wave（执行了 WMMA）和 empty expert / capacity-tail wave。历史中每个 SIMD、每个版本得到 128 个有效 wave，每个 wave 7 个 N-task；目标 wave 总数为 646–818 个，统计热路径时只用有效 wave；
- 采集前后 GPU 都空闲。结束检查出现其他 KFD 进程时，这组数据不参与耗时对比，raw/ISA 证据可以保留。

采集结束时可能出现 SDK cleanup 警告 `Failed to destroy queue: 4107`，需要记录，但不能当作 kernel 故障的证据。

EP kernel 的 code object 中固化了 Mori window 地址，每次运行的 code object SHA256 都不同；比较时核对控制流、PC 和指令形状。

### 4.7 隔离诊断副本（可选）

runner 支持只在当前进程替换生产 launcher 的诊断副本，生产文件不变：

| 变量 | 作用 |
|---|---|
| `AITER_ATT_DIAGNOSTIC=<variant>` | `split_wait`、`global_window`、`global_split`、`const_geometry`、`oob_output`、`oob_sorted`、`local_output` |
| `AITER_ATT_DIAGNOSTIC_DIR` | 诊断副本所在目录（含 `persistent_<variant>.py`、`scatter_*.py`） |
| `AITER_ATT_DIAGNOSTIC_DEFER=1` | graph、warmup、eager dispatch、GEMM1 都用生产 kernel，只在选中的 eager GEMM2 前安装副本，保证输入是真实的非零 EP 数据（配合 `gemm2` 区间） |
| `AITER_ATT_DIAGNOSTIC_DUMP=<dir>` | 导出诊断副本编译出的 hsaco 和 source IR |

副本生成脚本在 `.codex_tmp/f01_1_dsr1_gemm2_att_20261014/make_diagnostics.py` 和 `.codex_tmp/f01_2_dsr1_gemm2_att_20261014/make_*.py`。备用机器上的 `run_diagnostic.sh`、`analyze_diagnostic.sh` 由 `prepare_diagnostic_scripts.py` 基于备用版 `run_capture.sh` 生成（增加 `AITER_ATT_DIAGNOSTIC` 透传，并给 kernel 名加诊断后缀）。原机器对应的固化脚本是 `run_diagnostic_resume_20261014.sh`。备用机器上的示例：

```bash
AITER_ATT_DIAGNOSTIC=split_wait bash run_diagnostic.sh fused 0 new_split_case device graph
AITER_ATT_DIAGNOSTIC=local_output AITER_ATT_DIAGNOSTIC_DEFER=1 \
  bash run_diagnostic.sh fused 0 new_local_eager_case device gemm2
AITER_ATT_DIAGNOSTIC=oob_sorted AITER_ATT_DIAGNOSTIC_DEFER=1 \
  bash run_diagnostic.sh fused 0 new_oob_sorted_case device gemm2
```

诊断副本的 kernel symbol 带 `_diag_<variant>` 后缀，分析时传入完整名字。部分副本的数值结果故意无效，只能用于因果对照，不能接入生产。

### 4.8 原机器被占用时：在备用机器上建独立容器

2026-10-13 在 f01-2 上创建过独立诊断容器（没有覆盖已有仓库和容器）：

```bash
mkdir /home/yanguahe/codex_dsr1_gemm2_att_20261014
docker run -d --name codex-dsr1-gemm2-att-20261014 --network host --ipc host \
  --device /dev/kfd --device /dev/dri \
  -v /data/yanguahe/code:/data/yanguahe/code \
  -v /home/yanguahe/codex_dsr1_gemm2_att_20261014:/work \
  -e MORI_GPU_ARCHS=gfx1250 -e FLYDSL_GPU_ARCH=gfx1250 \
  ee6d7ff29b6e sleep infinity
```

- 镜像：`rocm/vllm-dev:nightly_455_wip__torch__1010_b323`（本地 image ID `ee6d7ff29b6e`）。PyTorch `2.11.0+rocm10.1.0a20260822`，FlyDSL 0.3.4.1。
- 只复制原实现需要的文件：AITER 源码与已构建的 extension、Mori Python/CCO runtime、采集 helper；打包后核对 SHA256。
- 容器内环境 [`/work/att_env.sh`](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/att_env.sh)：

  ```bash
  export ROCM_PATH=/opt/rocm
  export HIP_PATH=/opt/rocm
  export HIP_PLATFORM=amd
  export MORI_SOURCE_ROOT=/work/mori
  export MORI_JIT_CACHE_DIR=/work/mori_jit
  export LD_LIBRARY_PATH="/work/mori/python/mori:/data/yanguahe/code/wk_sp1/comgr_new:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib:/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH="/work/aiter_moe_a4w4_pr_refactor_wt:/work/mori/python${PYTHONPATH:+:$PYTHONPATH}"
  export MAX_JOBS=16
  ```

- 备用版 [`run_capture.sh`](../../.codex_tmp/f01_2_dsr1_gemm2_att_20261014/run_capture.sh) 的参数为 `MODE SIMD LABEL [none|device|dispatch] [graph|gemm2]`，额外设置 `AITER_ATT_ENV=/work/att_env.sh`、`AITER_ATT_ROCPROF=/opt/venv/.../_rocm_sdk_core/bin/rocprofv3`、`AITER_ATT_LIBRARY_PATH=/opt/venv/.../_rocm_sdk_devel/lib`、`AITER_ATT_DEVICE_LIBRARY=/work/libtrace_mega_moe_agent.so`。解码和分析：

  ```bash
  docker exec codex-dsr1-gemm2-att-20261014 bash /work/analyze_diagnostic.sh <LABEL> <完整 symbol>
  ```

- 不同机器上的 `/data/...` 不一定是同一个文件系统；两台机器的 runtime 版本也不同。wave cycles 只在同一台机器内比较，不能跨机器相减或当作原机器的性能数据。

### 4.9 扩展到其他通信 kernel（推断，历史中未验证）

`graph` 区间 trace 的是选定 rank 上一次完整 graph replay 中的所有 kernel，tool 也保存了该进程加载的全部 code object（历史采集中同时加载了 `ep_combine_fused_sync_0`、`ep_combine_fused_0` 的 code object）。因此推断：分析 dispatch/combine 等其他 kernel 时，可以对同一份 raw 换 `--kernel <该 kernel 的完整 symbol>` 运行 analyzer；如果需要 eager 单 kernel 圈选，可参照 `install_gemm2_region()` hook 对应 kernel 的 launcher。这些用法在历史中没有经过验收，使用时要按 4.6 的标准重新确认。

---

## 5. 分析工具

### 5.1 采集校验：`validate_att_capture.py`

位于 skill `C:/Users/yanguahe/.codex/skills/rebuild-flydsl-att-env/scripts/validate_att_capture.py`，适用于方法 A、B 的 `ui_output_agent_*` 目录结构：

```bash
python validate_att_capture.py --root <case 或 thread_trace 根目录> \
  --kernel '<完整 kernel symbol>' --simds 0,1,2,3 [--json]
```

逐个 SIMD 检查：至少一个非空 `.att`、正好一个 `code.json`、至少一个 wave JSON、`code.json` 中含完整 symbol、code 列表非空、没有 `.long` / `v_illegal`。`rocprofv3` 返回 0 不代表采集成功。

### 5.2 时钟、占用与负载均衡：`analyze_att_capture.py`

[`analyze_att_capture.py`](analyze_att_capture.py)：

```bash
python3 my_code/analyze_att_capture.py --dir <case_root> --no-plot
```

需要 SE0–3 的完整采集。输出包括：每个 SE 的 GFXCLK（shader clock 与 realtime clock 之比，realtime 参考频率默认 100 MHz）、REALTIME 测得的 kernel wall time、occupancy 中 shader timestamp 的最大跨度、物理 slot 数、每个 SIMD 同时活跃的 slot 数、每个 slot 顺序执行的 wave 数，以及物理 WGP 之间的完成时间不均衡度（用于判断负载均衡）。

### 5.3 阶段与长延迟指令：`trace_segment_cycles.py`

位置：[`cursor_rules/fmha_flydsl_new_api_opt/.cursor/rules/trace_segment_cycles.py`](../../cursor_rules/fmha_flydsl_new_api_opt/.cursor/rules/trace_segment_cycles.py)（另有副本 `aiter/my_code1/moe_gemm1_act1_optimized/trace_segment_cycles.py`）。用法说明见同目录 `flydsl-align-reference-kernel.mdc` 第 104–111 行。

```bash
# compare 模式：各阶段 cycle 与指令数在版本之间的差异
python3 trace_segment_cycles.py <cfg>.json
# drill-down：代表性 wave 在指定区间内的逐指令时间线，按 long_lat_thr 过滤长延迟指令
python3 trace_segment_cycles.py <cfg>.json --specific-part-representative-trace
# 也可直接指定一个 trace 目录
python3 trace_segment_cycles.py <dir 含 code.json 与 se*_sm*_sl*_wv*.json> -k 10 \
  --start-regex '<首条指令>' --end-regex '<末条指令>'
```

配置 JSON 中定义 `interval information`（如 Prologue / Main loop / Epilogue）、各版本的 `sample points`（用指令序列定位区间边界）、`se/sm/sl/wv` 过滤和 `trace_dir`，以及 `specific_part` 的 `interval_name` 与 `long_lat_thr`。判读：cycle 差异远大于指令数差异时，差距来自 stall 而不是指令数；长延迟列表把 stall 归类为 BARRIER / VMEM / WAITCNT / MFMA / LDS / NOP。

### 5.4 指标口径

- **latency 占比** = Σ 指令 latency / Σ wave span；**exposed stall 占比** = Σ 指令 stall / Σ wave span。对 `s_wait_*` 指令通常有 latency = stall + 1 cycle（指令自身开销）。例如 `s_wait_dscnt 0x1b` 执行 864 次、latency 864 cycles、stall 0，说明每次都立即通过，不是瓶颈。判断优化空间时看 exposed stall，但 exposed stall 也不能直接等同于可获得的加速，其中一部分是必要的依赖和 LDS 复用保护。
- **阶段切分**：按 `code.json` 中真实的 PC 划分 prologue、hotloop、epilogue。
- **有效 wave**：只统计执行了 WMMA 的 wave；early-exit wave 的特征是没有 `v_wmma_*` 和 `tensor_*` 指令，执行完标量查找后以 `s_wait_tensorcnt 0x0` + `s_endpgm` 结束。
- **SIMD 覆盖**：全 WG 结论需要 SIMD0–3。只采 SIMD3 时得到的 barrier 占比是 30.29%，四个 SIMD 加权后是 24.54%。
- **代表性 wave**：选择 span 最接近有效 wave 中位数的 wave 打印逐指令时间线。
- **wave 映射**：逻辑 wave 编号、trace 文件中的 `wv`、resident slot、物理 SIMD 需要由 trace 确认。例：某版 GEMM1 的 wave 0–3 依次落在物理 SIMD0/3/2/1 的 slot0，wave 4–7 以相同顺序进入 slot1；这只对该版本成立。

### 5.5 反汇编

```bash
/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/llvm-objdump -d --mcpu=gfx1250 <code_object>
```

FlyDSL 缓存（`/root/.flydsl/cache/.../*.pkl`）中编译后的 MLIR 以 `bin = "\7FELF..."` 形式内嵌 ELF，可以抽出来反汇编，用于静态对照。

---

## 6. 采集前后的标准流程

1. **正常预检（不加 profiler）**：确认正确性，取得完整 kernel symbol，并记录正常 profiler 下的性能作为参考。
2. **GPU 空闲门禁**：检查全部 GPU 的 KFD 进程、利用率和 VRAM；忙时每 6 秒重试；采集期间不并发检查。
3. **逐个 SIMD 采集**：每次写入新目录。先用 SIMD0、SE mask `0x1` 打通流程，再补齐 SIMD0–3、`0xf`。
4. **验收**：按 5.1 或 4.6 的标准检查，不能只看 exit code。
5. **收尾**：结束后检查 GPU 是否还有本任务进程；在容器内修复输出目录权限。
6. **分析**：analyzer 和 `trace_segment_cycles.py` 输出写入新目录，不覆盖原有证据。
7. **归档**：raw、code object、decoded wave、`code.json`、命令、manifest 一起打包并生成 SHA256；下载到本地后校验。
8. **记录**：在文档中写明工具版本、完整命令、kernel symbol、每个 SIMD 的产物数量、验收结论和限制（例如 graph 与 eager 区间的差别、被排除的 case）。

---

## 7. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `import torch` 阶段 abort：`Configuration request occurred outside of valid rocprofiler configuration period` | 在 SDK wheel 容器中使用了自编译的 rocprofiler-sdk | 改用方法 B 的 bundled 工具 |
| `aqlprofile API table load failed`，随后 `No gfx arch found in rocminfo output` | `rocminfo` 子进程找不到裸名 `libhsa-amd-aqlprofile64.so` | 前置 `_rocm_sdk_devel/lib`，或使用 `_safe` 脚本的假 `rocminfo` |
| decoder 报「older than 0.2 cannot emit ATT event or dispatch records」 | decoder 与 SDK 版本不匹配 | 使用 SDK 自带的 decoder |
| `Missing SDK-aligned ATT decoder`（`librocprof-trace-decoder.so.0.2` 不存在） | 在 A 类容器运行了 `_safe` 脚本 | 改用方法 A |
| `ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA`，`code.json`/`occupancy.json` 为空 | 该次采集的数据无法解码 | 只重抓这个 SIMD，失败目录改名保留 |
| `code.json` 中出现 `.long` / `v_illegal` | COMGR 太旧，不认识 gfx1250 新指令 | 前置 `comgr_new` |
| 在 rocprof 环境中编译触发 `clang-23 SmallPtrSet` assertion | profiler 注入的环境影响了 clang | 预编译 `.co`，或使用 clean clang wrapper（2.4 节） |
| 四卡 workload 中出现 InterceptQueue 回退、GPU null-address fault、queue sync timeout | dispatch interception 与 workload 冲突（嫌疑，未证实） | 改用方法 C 的 device-based ATT |
| trace 中的 kernel 不是预期版本 | FlyDSL 缓存、导入路径错误、配置未命中 | 清空缓存，检查 `aiter.__file__`，核对完整 symbol |
| 访问容器生成的文件 Permission denied | 文件由容器内 root 创建 | 在容器内对具体路径 `chmod -R a+rwX` |
| `docker exec` 中的相对路径找不到脚本 | 容器默认 cwd 不同 | 使用绝对路径 |
| `gpu_users.sh` 空闲时也返回 1，脚本提前退出 | 脚本的返回值约定 | 加 `\|\| true`，用独立的 sysfs/KFD 检查判断是否空闲 |
| 采集结束出现 `Failed to destroy queue: 4107` | SDK cleanup 警告 | 记录在案，不作为 kernel 故障证据 |

---

## 8. 历史采集索引

| 时间 | 机器 / 容器 | 对象 | 方法 | 记录位置 |
|---|---|---|---|---|
| 2026-09-08～09-10 | b8-3、d01-3 / `hyg_fyd1` | 纯汇编 MoE GEMM1（E96/T16384） | A，ISA runner 变体 | [`HARDWARE_CYCLE_LIMIT.md`](../my_code1/moe_gemm1_act1_optimized/HARDWARE_CYCLE_LIMIT.md)、[`PERSISTENT_MODE_THREAD_TRACE_ANALYSIS.md`](../my_code1/moe_gemm1_act1_optimized/PERSISTENT_MODE_THREAD_TRACE_ANALYSIS.md) |
| 2026-09-30～10-01 | a07-3 / `hyg_fyd1` | FlyDSL GEMM1、GEMM2（E64/T1536/topk8），以及多个失败候选 | A，`get_isa_runner_att.sh --all-simd --ana-att` | `hyg/moe_a4w4_pr` 分支 `my_code/e64_t1536_topk8_gemm1_gemm2_thread_trace_analysis_a07_3.md`；远端 `my_code/thread_trace_runs/e64_t1536_topk8_*` |
| 2026-10-03 | a07-3 / `hyg_fyd1` | 当时最优 GEMM1（exact symbol，先 SIMD3 再 SIMD0–3） | A（`_safe` 脚本因缺少 0.2 decoder 失败后改用 A） | `hyg/moe_a4w4_pr` 分支 `my_code/dsr1_e64_t1536_topk8_moe_opt.md`（当前工作区的同名文件是较早版本）；远端 `my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_ldsseg5_xorm_exact_a07_3_20261003` |
| 2026-10-03 | a07-3 / `hyg_fyd_e2e`（同期另一工作线） | GEMM1 fused A-preshuffle quant v115，all-SIMD | B，`_safe` 脚本 | [`gemm1_fused_apre_quant_optimization_log.md`](gemm1_fused_apre_quant_optimization_log.md) |
| 2026-10-13～10-14 | f01-1 / `dsr1_ep4`；f01-2 / 独立容器 `codex-dsr1-gemm2-att-20261014` | 四卡 EP `_persist` 与 `_persist_epscatter` GEMM2，SIMD0–3 及隔离诊断 | C，device-based | [`dsr1_moe_ep_decoder_bug_fix.md`](dsr1_moe_ep_decoder_bug_fix.md) 第 7 节；归档 `.codex_tmp/f01_1_dsr1_gemm2_att_20261014/gemm2_att_complete.tar.gz` 等 |
