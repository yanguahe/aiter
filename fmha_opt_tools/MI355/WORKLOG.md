# FlyDSL flash-attn 工作记录 (WORKLOG)

<!-- markdown-toc-generator:start -->
## Table of Contents

- [环境与通用约定](#环境与通用约定)
- [不变量 / 约束（所有任务通用）](#不变量-约束所有任务通用)
- [任务记录](#任务记录)
  - [任务 1 — 分析手写 ASM kernel 的 q/kv seqlen 约束（2026-06-10）](#任务-1-分析手写-asm-kernel-的-qkv-seqlen-约束2026-06-10)
  - [任务 2 — 把非对齐 q/kv seqlen 处理逻辑移植到 kernels/flash_attn_gfx950.py（2026-06-10）](#sec-任务-2-把非对齐-qkv-seqlen-处理逻辑移植到-kernelsflash_attn_gfx950-py2026-06-10)
  - [任务 3 — 吸收 670.diff（split-K）并与非对齐支持共存（2026-06-11）](#sec-任务-3-吸收-670-diffsplit-k并与非对齐支持共存2026-06-11)
  - [任务 4 — 把 num_kv_splits 测试接入 tests/kernels/test_flash_attn_fwd.py（2026-06-11）](#sec-任务-4-把-num_kv_splits-测试接入-testskernelstest_flash_attn_fwd-py2026-06-11)
  - [任务 5 — compare 模式支持 --num_kv_splits（2026-06-12）](#sec-任务-5-compare-模式支持---num_kv_splits2026-06-12)
  - [任务 6 — 测试表格 + CSV 增加 kv_sp 列（2026-06-12）](#sec-任务-6-测试表格-csv-增加-kv_sp-列2026-06-12)
  - [任务 7 — DEFAULT_CONFIGS 加 num_kv_splits 列，驱动循环用 per-config 值（2026-06-12）](#sec-任务-7-default_configs-加-num_kv_splits-列驱动循环用-per-config-值2026-06-12)
  - [任务 8 — V0/V1/V2 性能基准测试 + Group A/B 对比（2026-06-13，用 test harness 重测）](#任务-8-v0v1v2-性能基准测试-group-ab-对比2026-06-13用-test-harness-重测)
    - [主配置性能表（TFLOPS，causal，higher=better）](#主配置性能表tflopscausalhigherbetter)
    - [性能变化分析](#性能变化分析)
    - [V2 Group A（splits=1）vs Group B（split-K）对比](#v2-group-asplits1vs-group-bsplit-k对比)
  - [任务 9 — flash_attn_generic 加 seq_len 保护机制（2026-06-13）](#sec-任务-9-flash_attn_generic-加-seq_len-保护机制2026-06-13)
    - [补充（2026-06-13）：causal vs non-causal 的 seqlen 非对称（guard 行为说明）](#补充2026-06-13causal-vs-non-causal-的-seqlen-非对称guard-行为说明)
  - [任务 10 — run_config 加完整的 size/dtype/arch 输入校验（2026-06-13）](#sec-任务-10-run_config-加完整的-sizedtypearch-输入校验2026-06-13)
    - [补充（2026-06-13）：校验改为直接 raise](#补充2026-06-13校验改为直接-raise)
  - [任务 11 — generic kernel 支持任意 seqlen（不回退 causal 性能）（2026-06-13）](#任务-11-generic-kernel-支持任意-seqlen不回退-causal-性能2026-06-13)
  - [任务 12 — Q-load 改用 DUALWAVE num_records 边界法（O-store 保留谓词，防回退）（2026-06-13）](#任务-12-q-load-改用-dualwave-num_records-边界法o-store-保留谓词防回退2026-06-13)
  - [任务 13 — O-store 移植 DUALWAVE 的 128-bit buffer_store_dwordx4（2026-06-13）](#任务-13-o-store-移植-dualwave-的-128-bit-buffer_store_dwordx42026-06-13)
  - [任务 14 — DUALWAVE_SWP kernel 支持 seq_len >= 1（2026-06-13）](#sec-任务-14-dualwave_swp-kernel-支持-seq_len-12026-06-13)
    - [任务 14 修复 — 极小 non-causal seqlen 的 padding-mask 漏洞（2026-06-13）](#任务-14-修复-极小-non-causal-seqlen-的-padding-mask-漏洞2026-06-13)
  - [任务 15 — 调查 FLYDSL_DUALWAVE_SWP_TRIGGER_LAZY_ELSE=1 为何更快（2026-06-13，纯调查，无代码改动）](#sec-任务-15-调查-flydsl_dualwave_swp_trigger_lazy_else1-为何更快2026-06-13纯调查无代码改动)
  - [任务 16 — 两 kernel 跨 block 负载分布分析报告（2026-06-14，纯分析，无代码改动）](#任务-16-两-kernel-跨-block-负载分布分析报告2026-06-14纯分析无代码改动)
  - [任务 17 — rocprofv3 ATT + occupancy_balance.py 实测 dualwave 负载均衡（2026-06-14，纯实测，无代码改动）](#sec-任务-17-rocprofv3-att-occupancy_balance-py-实测-dualwave-负载均衡2026-06-14纯实测无代码改动)
  - [任务 18 — 分析手写 ASM kernel fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s 的负载均衡方法（2026-06-14，纯分析，无代码改动）](#sec-任务-18-分析手写-asm-kernel-fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0-s-的负载均衡方法2026-06-14纯分析无代码改动)
  - [任务 19 — WG 内 wave→q_seq 行映射对比（dualwave kernel vs 手写 ASM kernel）（2026-06-14，纯分析，无代码改动）](#任务-19-wg-内-waveq_seq-行映射对比dualwave-kernel-vs-手写-asm-kernel2026-06-14纯分析无代码改动)
  - [任务 20 — 把 ASM 的 causal 三角负载折叠(q-block 镜像配对)移植进 flash_attn_dualwave_swp_gfx950_kernel（2026-06-14）](#sec-任务-20-把-asm-的-causal-三角负载折叠q-block-镜像配对移植进-flash_attn_dualwave_swp_gfx950_kernel2026-06-14)
  - [任务 21 — 分析手写 ASM kernel fmha_fwd_..._msk1_gm0.s 是否支持 QKV varlen（2026-06-14，纯分析，无代码改动）](#sec-任务-21-分析手写-asm-kernel-fmha_fwd_---_msk1_gm0-s-是否支持-qkv-varlen2026-06-14纯分析无代码改动)
  - [任务 22 — 给 flash_attn_dualwave_swp_gfx950_kernel 加 QKV varlen（packed cu_seqlens）（2026-06-14）](#sec-任务-22-给-flash_attn_dualwave_swp_gfx950_kernel-加-qkv-varlenpacked-cu_seqlens2026-06-14)
  - [任务 23 — /flydsl-modern-rewrite skill 重构 dualwave + combine 两 kernel（2026-06-15）](#sec-任务-23-flydsl-modern-rewrite-skill-重构-dualwave-combine-两-kernel2026-06-15)
  - [任务 24 — 整理 gfx950/CDNA4(MI350X & MI355X)硬件资料速查文档（2026-06-15）](#sec-任务-24-整理-gfx950cdna4mi350x-mi355x硬件资料速查文档2026-06-15)
  - [任务 25 — dualwave kernel 支持 seqlen_q != seqlen_kv(bottom-right causal)(2026-06-16)](#sec-任务-25-dualwave-kernel-支持-seqlen_q-seqlen_kvbottom-right-causal2026-06-16)
  - [任务 26 — seqlen_q != seqlen_kv 扩展到变长(varlen) + 修复 delta%128 mask bug(2026-06-16)](#sec-任务-26-seqlen_q-seqlen_kv-扩展到变长varlen-修复-delta128-mask-bug2026-06-16)
  - [任务 27 — flash_attn L2 wrapper + 测试函数合并重构（2026-06-18）](#任务-27-flash_attn-l2-wrapper-测试函数合并重构2026-06-18)
    - [背景与目标](#背景与目标)
    - [改动内容](#改动内容)
    - [性能数据（GPU：AMD Instinct MI355X gfx950，HIP_VISIBLE_DEVICES=3）](#sec-性能数据gpuamd-instinct-mi355x-gfx950hip_visible_devices3)
    - [正确性验证](#正确性验证)
    - [状态](#状态)
  - [技术问答 A — 循环 cluster3 为何只对 v_s_0 做 causal mask（2026-06-17）](#技术问答-a-循环-cluster3-为何只对-v_s_0-做-causal-mask2026-06-17)
  - [技术问答 B — Epilogue 阶段处理几个 kv-tile（2026-06-17）](#技术问答-b-epilogue-阶段处理几个-kv-tile2026-06-17)
  - [技术问答 C — aiter moe 封装方式 + fmha wrapper 设计建议（2026-06-17）](#技术问答-c-aiter-moe-封装方式-fmha-wrapper-设计建议2026-06-17)
  - [任务 28 -- PR #704 5-set 全量 perf sweep（2026-06-19）](#任务-28----pr-704-5-set-全量-perf-sweep2026-06-19)
    - [输入/要求](#输入要求)
    - [改动内容](#改动内容-1)
    - [性能数据汇总（MI355X gfx950，bf16+fp16 causal，HIP_VISIBLE_DEVICES=1，iters=100）](#性能数据汇总mi355x-gfx950bf16fp16-causalhip_visible_devices1iters100)
    - [踩坑](#踩坑)
    - [状态](#状态-1)
- [任务：paged-KV 解法 B —— per-page 重建 buffer 描述符（去除 4 GiB 上限）（2026-06-23）](#任务paged-kv-解法-b-per-page-重建-buffer-描述符去除-4-gib-上限2026-06-23)
    - [输入/要求](#输入要求-1)
    - [根因](#根因)
    - [改动内容](#改动内容-2)
    - [踩坑](#踩坑-1)
    - [验证结果（mi355-gpu-34 容器 hyg_fyd1，HIP_VISIBLE_DEVICES=1）](#验证结果mi355-gpu-34-容器-hyg_fyd1hip_visible_devices1)
    - [状态](#状态-2)
- [任务：split-K workspace 4 GiB 上限消除（per-split-z buffer descriptor）（2026-06-23）](#任务split-k-workspace-4-gib-上限消除per-split-z-buffer-descriptor2026-06-23)
    - [输入/要求](#输入要求-2)
    - [改动内容](#改动内容-3)
    - [踩坑](#踩坑-2)
    - [验证结果（mi355-gpu-34 容器 hyg_fyd1，HIP_VISIBLE_DEVICES=1）](#验证结果mi355-gpu-34-容器-hyg_fyd1hip_visible_devices1-1)
    - [状态](#状态-3)
- [任务：paged 路径 Q/O 自然 shape（消除 2^31 numel 上限）（2026-06-23）](#任务paged-路径-qo-自然-shape消除-231-numel-上限2026-06-23)
    - [输入/要求](#输入要求-3)
    - [根因](#根因-1)
    - [改动内容（解法 B 同款：偏移折进 48-bit base）](#改动内容解法-b-同款偏移折进-48-bit-base)
    - [效果](#效果)
    - [验证结果（hyg_fyd1，HIP_VISIBLE_DEVICES=1）](#验证结果hyg_fyd1hip_visible_devices1)
    - [状态](#状态-4)
- [任务：非 PAGED 路径 Q/K/V/O 自然 shape（dense/varlen/split-K/generic）（2026-06-23）](#任务非-paged-路径-qkvo-自然-shapedensevarlensplit-kgeneric2026-06-23)
    - [输入/要求](#输入要求-4)
    - [根因（同 PAGED 解法 B）](#根因同-paged-解法-b)
    - [改动内容](#改动内容-4)
    - [踩坑](#踩坑-3)
    - [验证结果（hyg_fyd1，HIP_VISIBLE_DEVICES=1，清 cache）](#验证结果hyg_fyd1hip_visible_devices1清-cache)
    - [状态](#状态-5)
- [任务：paged-KV 支持 split-K（修复 B=1/small-H 性能回退）（2026-06-23）](#任务paged-kv-支持-split-k修复-b1small-h-性能回退2026-06-23)
    - [输入/要求](#输入要求-5)
    - [根因](#根因-2)
    - [改动内容](#改动内容-5)
    - [验证结果（hyg_fyd1，HIP_VISIBLE_DEVICES=1，清 cache）](#验证结果hyg_fyd1hip_visible_devices1清-cache-1)
    - [状态](#状态-6)
- [任务：varlen 输入支持 paged KV（2026-06-23）](#任务varlen-输入支持-paged-kv2026-06-23)
    - [输入/要求](#输入要求-6)
    - [根因/设计](#根因设计)
    - [改动内容](#改动内容-6)
    - [验证结果（hyg_fyd1，HIP_VISIBLE_DEVICES=1，清 cache）](#验证结果hyg_fyd1hip_visible_devices1清-cache-2)
    - [状态](#状态-7)
- [任务：vectorized KV cache layout 支持（进行中，2026-06-24）](#任务vectorized-kv-cache-layout-支持进行中2026-06-24)
    - [需求](#需求)
    - [阶段1（✅ 完成，端到端 compare 通过）](#阶段1-完成端到端-compare-通过)
    - [阶段2（🔄 进行中，2026-06-24，coalesced DMA + plain read 已落地，差 10–19% 待收尾）](#阶段2-进行中2026-06-24coalesced-dma-plain-read-已落地差-1019-待收尾)
    - [状态](#状态-8)
- [当前未决事项](#当前未决事项)

<!-- markdown-toc-generator:end -->

> 本文件是该工作区所有任务的**滚动工作记录**。每完成一个任务，在「任务记录」区按时间顺序追加一节，不要重写历史条目。
>
> - 远程代码根：`/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_sp1/FlyDSL`
> - 本地镜像：`C:\Users\yanguahe\code\wk_sp1\FlyDSL`（先本地编辑，再上传到远程）

---

## 环境与通用约定

- **容器**：`hyg_trn_rocm7.1`，arch **gfx950**（MI350 级），8 卡空闲。
- **GPU 选择**：用**没有进程在跑的 GPU**（运行前先 `rocm-smi --showpids` / `rocm-smi` 查空闲卡，挑无进程的那张设 `HIP_VISIBLE_DEVICES`）。
- **FlyDSL 安装**：`./FlyDSL` 已可编辑安装（`flydsl 0.2.0.dev702` → repo 源码）。改 `kernels/*.py` 是纯 Python，**无需重编 LLVM/C++**。
- **跑测试必带**：
  ```bash
  docker exec hyg_trn_rocm7.1 bash -c "
    cd /shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_sp1/FlyDSL &&
    export HIP_VISIBLE_DEVICES=2 FLYDSL_RUNTIME_ENABLE_CACHE=0 &&
    export LD_LIBRARY_PATH=\$PWD/build-fly/python_packages/flydsl/_mlir/_mlir_libs:\${LD_LIBRARY_PATH:-} &&
    python3 <脚本/测试>
  "
  ```
- **正确性门槛**：MaxErr < 8e-3（且 cos ≈ 1.0）。
- **提交规则**：每次 `git commit` 前**必须经用户批准**；目前所有改动均**未 commit**。

## 不变量 / 约束（所有任务通用）

- 只移植/实现明确要求的逻辑；ASM 里与任务无关的部分不 copy（persistent ≤2 q-block、负载均衡 remap、LSE、双 body 奇偶 drain、`s_setprio` 等）。
- `q_seq_len == kv_seq_len == seq_len`；`seq_len >= 384`（流水线深度下限）；`head_dim == 128`。
- bottom-right 因果，对角线偏移 `kv - q`。
- 与中文交流。

---

## 任务记录

### 任务 1 — 分析手写 ASM kernel 的 q/kv seqlen 约束（2026-06-10）

- **输入**：`exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s` + `fmha_opt_tools/fmha_hand_asm_d128_deep_introduce.md`。要求以**汇编实际行为**为准。
- **结论**：
  - Q seqlen 任意 ≥1、KV seqlen 任意 ≥0，**均无对齐要求**。机制：`num_records=seq_len*stride` 行边界 + KV padding mask（尾块 `>=kv` 置 -inf）+ 奇偶 tile drain。
  - 硬限制：`head_dim==128`（全程硬编码）；bottom-right 因果（偏移 `kv-q`）；`num_records` 为 32-bit → 单 `(batch,head)` 区 <4GiB；GQA 要求 `q_head` 整除 `gqa`。
  - **doc/wrapper 不准之处**：`ptr_qseq/ptr_kseq`（`s[72:75]`）是**死加载**，并非真 varlen（同一次 launch 所有 batch 共用标量 seqlen）；kv 不需 128/64 对齐。
- **状态**：✅ 完成（纯分析，无代码改动）。

<a id="sec-任务-2-把非对齐-qkv-seqlen-处理逻辑移植到-kernelsflash_attn_gfx950-py2026-06-10"></a>
### 任务 2 — 把非对齐 q/kv seqlen 处理逻辑移植到 `kernels/flash_attn_gfx950.py`（2026-06-10）

- **要求**：只 copy ASM 的非对齐处理逻辑，其余不动。环境=容器 A（`hyg_trn_rocm7.1`）；tile 奇偶用「向上取偶 + padding mask」方案（不重构流水线）。
- **改动（4 处）**：
  1. **`num_records` 设界**：Q/K/V/O 按 `(batch+1)*seq_len*stride` 设 `num_records_bytes`（越界 load 返 0、越界 O store 被硬件丢弃）。
  2. **`max_num_tiles` 向上取偶**：`((max+1)//2)*2`，适配 prologue(1)+loop(×2)+epilogue(3) 的 drain（替代 ASM 的双 body 奇偶 drain）。对齐场景本就是 4 的倍数 → no-op。
  3. **KV padding mask**：新增 `_seq_pad_mask_inplace` / `_seq_pad_mask_if_needed`，仅在 **non-causal** 的 3 个 epilogue mask 点（C2/C6/C10）接入。**causal 因 q==kv 已被原 causal mask 覆盖，不加**。门控 `(tile+1)*BLOCK_N > seq_len`，对齐时为 false。
  4. **放宽 dispatch/docstring**：`flash_attn_generic.py` 把 `S>=384 and S%256==0` → `S>=384`。
- **踩坑**：`make_buffer_tensor(num_records_bytes=)` 不收 `fx.Index` 包装 → 用 `_raw(...)`；`max_num_tiles` 不要再套 `fx.Index(...)`（双包裹使 `.value` 不是 `ArithValue`，`scf.range` 的 stop `.ir_value()` 会失败）。
- **验证**：**18/18 PASS**，MaxErr ≤ 3.9e-3，min cos 0.99999。覆盖：对齐 512、非256对齐 384/640、非64对齐+padding 400/500、奇数tile补齐 448、GQA、B=2、causal/non-causal。对齐 perf 回归（512/4096）在 ±1~2% 噪声内，热主循环未退化。
- **状态**：✅ 完成。**未 commit**。改动文件：`kernels/flash_attn_gfx950.py`、`kernels/flash_attn_generic.py`。

<a id="sec-任务-3-吸收-670-diffsplit-k并与非对齐支持共存2026-06-11"></a>
### 任务 3 — 吸收 `670.diff`（split-K）并与非对齐支持共存（2026-06-11）

- **要求**：把 `670.diff` 吸收进 `kernels/flash_attn_gfx950.py`。
- **做法**：diff 基于**改动前原版**生成 → 用「stash 我的非对齐改动 → `git apply 670.diff` → 在重构后的文件上重新 merge 4 处非对齐改动」。
- **split-K 带来的功能（来自 diff）**：`num_kv_splits` 参数、`dualwave_splitk_workspace_elems()`、`block_idx.z=(batch,split)`、每 split 处理 `[split_t0, split_t_end)` 的 KV tile、`with _split_guard:` 包裹、空 split 写 0/l=0/m=-1e30、O_partial 16-bit 入 workspace + 每行 fp32 m/l、combine kernel（`out=Σw·O/Σw·l`）、128-bit O store（permlane32_swap 融半波）。
- **修复的交互 bug（split-K + 非对齐）**：split-K 工作区是**单一无边界 buffer**，非对齐时部分末尾 q-block 的 `q_row>=seq_len` 行会写坏相邻 head 区（S=640/500 全错，cos≈-0.25）。修法：给 workspace 写入（O_partial + m/l + 空split清零）加 `q_row < seq_len` 守卫（combine 本就只读 `s<seq_len` 行）。`lane` 与 `lane+32` 共享 `lane%32`→共享 `q_row`，故半波 `permlane32_swap` 的 partner 同时有效/无效，守卫安全（swap 在守卫外全波计算，仅 store 在守卫内）。
- **验证**：非 split-K 回归 **18/18 PASS**；split-K **14/14 PASS**（splits=2/4/8，对齐 2048/1536 + 非对齐 640/500 + GQA，causal/non-causal，MaxErr ≤ 3.9e-3；修复前 640/500 是 FAIL）。
- **状态**：✅ 完成。**未 commit**。改动文件：`kernels/flash_attn_gfx950.py`。

<a id="sec-任务-4-把-num_kv_splits-测试接入-testskernelstest_flash_attn_fwd-py2026-06-11"></a>
### 任务 4 — 把 `num_kv_splits` 测试接入 `tests/kernels/test_flash_attn_fwd.py`（2026-06-11）

- **背景**：现有 `build_flash_attn_func_module` **未透传 num_kv_splits**；split-K 只能经 `build_flash_attn_dualwave_swp_module(num_kv_splits=N)` 直驱 + 传 fp32 workspace。
- **改动（`tests/kernels/test_flash_attn_fwd.py`）**：
  1. 导入 `build_flash_attn_dualwave_swp_module`、`dualwave_splitk_workspace_elems`。
  2. 新增 `run_splitk_config(...)`：直驱 split-K 构建 + 自动分配 fp32 workspace + 对比 torch SDPA，返回与 `run_config` 兼容的结果字典，复用同一汇总表。
  3. 新增 CLI `--num_kv_splits`（默认 1）；normal 模式 `>1` 走 split-K，`=1` 走原路径（默认完全不变）。
  4. **优雅 SKIP**：对不适用形状返回 `{"skip": True}` 而非 FAIL —— `seq_len<384`、`head_dim!=128`、非 bf16/f16，以及 **workspace>4GiB**（32-bit buffer 描述符寻址，`B*splits*H*S` 过大时高位 m/l 偏移被 OOB 丢 → 错；split-K 本面向小 grid）。`_fmt_normal_row` 与统计均加 SKIP 处理。
- **跑法**：
  ```bash
  # 单配置 splits=2
  python3 tests/kernels/test_flash_attn_fwd.py --num_kv_splits 2 --batch 1 --seq_len 2048 --num_heads 64 --head_dim 128 --dtype bf16
  # splits=2，非对齐 GQA
  python3 tests/kernels/test_flash_attn_fwd.py --num_kv_splits 2 --batch 1 --seq_len 500 --num_heads 64 --num_kv_heads 8 --head_dim 128 --dtype bf16
  # 默认全扫（splits=2）：不适用自动 SKIP
  python3 tests/kernels/test_flash_attn_fwd.py --num_kv_splits 2 --dtype bf16 --causal
  ```
- **验证**：splits=2@2048 / splits=4@1536 / splits=2@500(GQA) 均 **PASS**；默认全扫支持配置全 PASS、`<384` 与 `>4GiB`(B=16/S=8192/H=64) 自动 SKIP；默认 splits=1 回归不变。
- **状态**：✅ 完成。**未 commit**。改动文件：`tests/kernels/test_flash_attn_fwd.py`。

<a id="sec-任务-5-compare-模式支持---num_kv_splits2026-06-12"></a>
### 任务 5 — compare 模式支持 `--num_kv_splits`（2026-06-12）

- **背景**：任务 4 只给 normal 模式接了 `--num_kv_splits`；`--compare` 模式的 FlyDSL 列仍固定走 `run_config`（非 split-K），无法在对比表里测 split-K。
- **改动（`tests/kernels/test_flash_attn_fwd.py`）**：
  1. compare 模式里把 FlyDSL 列改为：`args.num_kv_splits > 1` 时走 `run_splitk_config(..., num_kv_splits=args.num_kv_splits)`，否则走原 `run_config`（与 normal 模式同样的路由）。
  2. compare 模式 banner 增加一行提示：split-K 激活时打印 `num_kv_splits=N` 及 SKIP 规则。
  - 复用现有 `_fmt_result`/`_cmp_values`/`_csv_val`/`_avg_results`，它们本就处理 `skip`/`err` → 不适用配置在 FlyDSL 列显示 `--`，对比列显示 `--`/N/A，CSV 留空，均无需改动。
- **验证**（容器 hyg_trn_rocm7.1 / gfx950 / GPU2，MI355X）：
  - `--compare --num_kv_splits 2`，B1/S2048/H64/D128/bf16/causal：FlyDSL 列走 split-K，MaxErr 3.91e-3，Fly/OPUS·ck·asm 对比列正常填充 ✅
  - 回归 `--compare`（默认 splits=1）：FlyDSL 列走原 `run_config`（804.8 TFLOPS），行为不变 ✅
  - `--compare --num_kv_splits 2 --seq_len 256`（<384）：FlyDSL 列优雅 **SKIP**（显示 `--`），其它后端照常 ✅
- **状态**：✅ 完成。**未 commit**。改动文件：`tests/kernels/test_flash_attn_fwd.py`。

<a id="sec-任务-6-测试表格-csv-增加-kv_sp-列2026-06-12"></a>
### 任务 6 — 测试表格 + CSV 增加 `kv_sp` 列（2026-06-12）

- **要求**：参照 `tt.log` 的 compare 输出表，多加一列 `kv_sp`（= `num_kv_splits`），写 CSV 时也加这一列。
- **改动（`tests/kernels/test_flash_attn_fwd.py`）**：
  1. cfg 元组从 7 元扩为 8 元，末尾追加 `args.num_kv_splits`（compare 与 normal 两个构建点）。
  2. `_CFG_HDR` / `_fmt_cfg` 在 `causal` 后加 `kv_sp` 列（compare 与 normal 的打印表都自动带上，因共用 `_CFG_HDR`/`_CFG_W`）。`_tag_group` 仍用 idx 5/6，不受影响。
  3. compare CSV（`_write_cmp_csv`）：header 在 `causal` 后插 `kv_sp`；avg 行空 cfg 占位由 7 → 8。
  4. normal CSV（`_write_normal_csv`）：header 同样插 `kv_sp`；avg 行空占位 +1。
- **验证**（容器 hyg_trn_rocm7.1 / gfx950 / MI355X / GPU2）：
  - compare `--num_kv_splits 2`（B2/S1024）：表格出现 `kv_sp` 列显示 2；CSV header+数据行均含 `kv_sp=2` ✅
  - normal 默认（splits=1）：表格 `kv_sp` 列显示 1，PASS（MaxErr 3.91e-3 / cos 0.99999）✅
- **状态**：✅ 完成。**未 commit**。改动文件：`tests/kernels/test_flash_attn_fwd.py`。

<a id="sec-任务-7-default_configs-加-num_kv_splits-列驱动循环用-per-config-值2026-06-12"></a>
### 任务 7 — `DEFAULT_CONFIGS` 加 `num_kv_splits` 列，驱动循环用 per-config 值（2026-06-12）

- **要求**：`DEFAULT_CONFIGS` 追加第 6 列 `num_kv_splits`，表格与 CSV 已有的 `kv_sp` 列反映每行配置值。
- **改动（`tests/kernels/test_flash_attn_fwd.py`）**：
  1. `DEFAULT_CONFIGS` 每行追加 `, 1`（当前全部默认 split-K=1，注释说明 >1 需 gfx950/seq_len≥384/D=128）。
  2. 单配置临时 configs（CLI 指定形状时）追加 `args.num_kv_splits`。
  3. compare + normal 两处循环解包改为 6 元 `cfg_kv_splits`；优先级：`args.num_kv_splits > 1` 时 CLI 覆盖，否则用 `cfg_kv_splits`（与 `num_kv_heads` 覆盖逻辑一致）。结果值 `kv_splits` 传给 cfg 元组和 `run_splitk_config`。
- **验证**（容器 hyg_trn_rocm7.1 / MI355X / GPU2）：
  - normal，config 默认 splits=1：`kv_sp=1` PASS（MaxErr 3.91e-3）✅
  - normal，`--num_kv_splits 2` CLI 覆盖 config 的 1：`kv_sp=2` PASS（MaxErr 3.91e-3）✅
  - compare，`--num_kv_splits 2`：表格 `kv_sp=2`，CSV `kv_sp` 列值 2 ✅
- **状态**：✅ 完成。**未 commit**。改动文件：`tests/kernels/test_flash_attn_fwd.py`。

### 任务 8 — V0/V1/V2 性能基准测试 + Group A/B 对比（2026-06-13，用 test harness 重测）

> ⚠️ 本节为**修正版**。初版用自写脚本 + `time.perf_counter()`（wall-clock）测，把 Python→HIP dispatch 的 CPU 开销（~35us/次）计进了 kernel 时间，导致小 kernel（~50us）TFLOPS 严重低估、并误报 `B1 S8192 H2 splits=1` "-27% 退步"。本版改为**直接用 `tests/kernels/test_flash_attn_fwd.py` 的 perf 输出**（其计时只测 GPU 执行），数据为准。

- **测试范围**：
  - **V0**：commit `39307ea4`（modern Layout/Atom API 移植前）
  - **V1**：commit `0d8cc250`（"port to modern Layout/Atom API"，语义等价重写）
  - **V2**：HEAD `72726f4a`（含 split-K、任意 seqlen、任务 1-7 所有改动）
  - **主配置**：`test_flash_attn_fwd.py` 第 95-110 行的 12 个 shape（splits=1，`--causal --dtype bf16`），**外加** 第 100-101 行的 `(1,2048,32,32,128)` 用 **fp16** 单测一个点（共 13 个数据点）
  - **Group 对比**（V2 only）：第 107-110 行（splits=1）vs 第 112-115 行（同形状 splits=4/2）
- **环境**：容器 `hyg_trn_rocm7.1`，**GPU1**（MI355X），custom LLVM，版本切换时清 `~/.flydsl/cache`，按要求 unset 全部 debug/dump/cache-disable 环境变量
- **方法**：直接跑 harness（`DEFAULT_CONFIGS` 临时改为这 16 个目标 shape；fp16 点用 CLI `--dtype fp16` 单跑），解析其 normal-mode 表格的 Time(us)/TFLOPS。切换版本只 `git checkout <hash> -- kernels/flash_attn_gfx950.py kernels/flash_attn_generic.py`，FlyDSL 不重编（纯 Python 前端差异）

#### 主配置性能表（TFLOPS，causal，higher=better）

| Config | V0 | V1 | V2 | V1/V0 | V2/V1 | V2/V0 |
|---|---:|---:|---:|---:|---:|---:|
| B16 S8192 H64 Hkv64 MHA       | 1142.9 | 1145.7 | 1153.9 | 100.2% | 100.7% | 101.0% |
| B16 S8192 H64 Hkv8  GQA       | 1244.2 | 1245.1 | 1251.1 | 100.1% | 100.5% | 100.6% |
| B2  S1024 H64 Hkv64 MHA       |  565.2 |  575.7 |  627.5 | 101.9% | 109.0% | 111.0% |
| B1  S2048 H32 Hkv32 MHA bf16  |  662.9 |  666.1 |  680.8 | 100.5% | 102.2% | 102.7% |
| B1  S2048 H32 Hkv32 MHA fp16  |  645.0 |  647.5 |  665.3 | 100.4% | 102.7% | 103.1% |
| B4  S2048 H8  Hkv1  GQA       |  663.6 |  656.1 |  675.5 |  98.9% | 103.0% | 101.8% |
| B8  S1024 H32 Hkv32 MHA       |  619.2 |  632.8 |  669.9 | 102.2% | 105.9% | 108.2% |
| B1  S8192 H32 Hkv32 MHA       | 1066.4 | 1062.3 | 1070.9 |  99.6% | 100.8% | 100.4% |
| B1  S4096 H32 Hkv32 MHA       |  877.4 |  889.2 |  910.9 | 101.3% | 102.4% | 103.8% |
| B1  S8192 H2  Hkv2  MHA       |  250.0 |  251.5 |  247.4 | 100.6% |  98.4% |  99.0% |
| B1  S4096 H2  Hkv2  MHA       |  116.0 |  117.1 |  119.7 | 100.9% | 102.2% | 103.2% |
| B1  S2048 H4  Hkv4  MHA       |  100.6 |  102.3 |  104.9 | 101.7% | 102.5% | 104.3% |
| B1  S8192 H4  Hkv4  MHA       |  473.9 |  475.3 |  481.4 | 100.3% | 101.3% | 101.6% |

（kernel 时间 us 同步采集，例：B1 S2048 H32 bf16 = V0 51.8 / V1 51.6 / V2 50.5 us；B16 S8192 MHA ≈ 15.2-15.4 ms。）

#### 性能变化分析

**V0→V1（Layout/Atom API 重写）**：全部配置 ±2% 持平，符合"语义等价重写、算法/调度/数值不变"的预期。

**V1→V2（split-K、任意 seqlen、num_records 边界等新功能）**：**所有配置 +1~9%，无任何退步**。热主循环未受影响；小配置（B2 S1024、B8 S1024）甚至 +6~9%（可能与 cache/调度顺序的良性扰动有关）。`B1 S8192 H2` 为 247.4 vs V1 251.5（-1.6%，噪声内）——**初版报告的 "-27% 退步" 是 wall-clock 计时假象，不存在**。

#### V2 Group A（splits=1）vs Group B（split-K）对比

| Config | Group A splits=1 | Group B split-K | splits | spB/spA |
|---|---:|---:|:---:|---:|
| B1 S8192 H2 Hkv2 | 247.4 T | **618.0 T** | 4 | **+149.8%** |
| B1 S4096 H2 Hkv2 | 119.7 T | **273.4 T** | 4 | **+128.4%** |
| B1 S2048 H4 Hkv4 | 104.9 T | **180.6 T** | 4 | **+72.2%** |
| B1 S8192 H4 Hkv4 | 481.4 T | **718.7 T** | 2 | **+49.3%** |

**关键发现**：小头数（H=2/4）、大 seqlen 的低占用配置，启用 split-K 后大幅提速 **+49%~+150%**（这些场景 grid 太小、CU 利用率低，split-K 把 KV 维拆分到更多 block 填满 GPU）。这正是 split-K 的设计目标场景。

- **状态**：✅ 完成（harness 实测 + 分析）。**未 commit**（临时脚本与 DEFAULT_CONFIGS 临时改动已还原）。

<a id="sec-任务-9-flash_attn_generic-加-seq_len-保护机制2026-06-13"></a>
### 任务 9 — `flash_attn_generic` 加 seq_len 保护机制（2026-06-13）

- **要求**：研读 `kernels/flash_attn_generic.py`，搞清 seq_len 限制，在 `build_flash_attn_func_module_primary` 中加保护机制。
- **研读结论 —— seq_len 限制**：
  - **DUALWAVE_SWP 快路径**（gfx950 / D=128 / bf16,f16，且 `block_m is None`）：运行时 `seq_len >= 384` 即走，**任意对齐**（kernel 内部自己处理部分末尾 q-block + 部分/奇数 kv-tile，见任务 2-3）。
  - **generic 兜底路径**（其余情况，含 `seq_len < 384`）：**`seq_len % BLOCK_N_OUT == 0`**（N128 路径=128，否则=64）。原因：Q-load 有 `q_in_bounds` 守卫、O-store 跳过越界行，**但 KV 协作 load 是无边界的裸指针 load（`coop_load_k/v` → `load_global_f16xN`），且无 KV padding mask**；`for kv in range(0, kv_upper, BLOCK_N_OUT)` 最后一块按整块读 → 非对齐时**越界读 + 把垃圾 key 卷进 softmax**。
  - 其它（build 时已 assert）：`num_heads % num_kv_heads == 0`、`head_dim % 32 == 0`、`head_dim >= 64`、`dtype ∈ {f16,bf16}`。
- **改动（`kernels/flash_attn_generic.py`，纯前端，无需重编）**：
  1. 在函数开头按 `(path_tag, dtype, causal, head_dim)` 推导 `_FALLBACK_SEQ_ALIGN`（128/64）与 `_DUALWAVE_MIN_SEQ=384`（BLOCK_N_OUT 不依赖 block_m，故可在两个变体构建前先算出）。
  2. 新增 `_extract_seq_len`（取 launch 第 6 参 / `seq_len` kwarg，非 int 返回 None）与 `_guard_seqlen` 包装器：launch 时若 seq_len 是确定 int 且 **既不满足 dualwave（≥384）又不满足兜底（≥1 且 % align==0）** → 抛 `ValueError`（含具体约束与 shape 信息）。
  3. 把 guard 接进 `_wrap_with_dualwave_swp`，**仅在最外层（`block_m is None`）用户可见的 launcher 上施加**；内层递归构建（m128/m256，`block_m` 已设）不重复加。符号/非 int seq_len 放行（无法静态检查）。
- **验证**（容器 hyg_trn_rocm7.1 / gfx950 / GPU2，num_heads=32 D=128 bf16 causal，触发 auto 双变体+dualwave）：
  - **拒绝**：seq_len = 0 / 100 / 200 / 300 → 均抛 `ValueError`（清晰报错）✅
  - **放行**：seq_len = 256（兜底 %128）/ 384（dualwave）/ 513（dualwave，非128对齐）/ 2048（dualwave）→ 均正常运行 ✅
- **状态**：✅ 完成。**未 commit**。改动文件：`kernels/flash_attn_generic.py`。

#### 补充（2026-06-13）：causal vs non-causal 的 seqlen 非对称（guard 行为说明）

- **现象**：`seq_len=64`（H=4 D=128 bf16）下 `--no-causal` 能跑、`--causal` 报 `ValueError`（即任务 9 的 guard）。
- **根因**：路径选择不同（`flash_attn_generic.py:239`）——
  - non-causal D128 bf16 → **N32** 路径，`BLOCK_N_OUT=64` → guard 要求 `%64`，`64%64==0` ✓ 跑通且正确。
  - causal D128 bf16 → **N128** 路径，`BLOCK_N_OUT=128` → guard 要求 `%128`，`64%128!=0` ✗ 报错。
- **N128 causal 为何真需 %128**：一次读 128 行 KV block = 2×64 子块；`seq_len=64` 时第二子块 `[64,128)` 全部越界读（KV 张量只 64 行）。这些 key 因 `kv_col≥64>q_row` 被因果 mask 成 -inf → **数值结果其实对**，但 kernel 仍发出越界读（UB / 可能 fault），正是 guard 拦的点。non-causal 走 N32 只读 `[0,64)` 不越界，故安全。
- **决定**：**保持 guard 严格**（用户确认）。causal D128 需 `seq_len % 128 == 0`，或用 `seq_len >= 384` 走 DUALWAVE_SWP 快路径。不改路径选择逻辑。

<a id="sec-任务-10-run_config-加完整的-sizedtypearch-输入校验2026-06-13"></a>
### 任务 10 — `run_config` 加完整的 size/dtype/arch 输入校验（2026-06-13）

- **要求**：把所有针对规模、数据类型、GPU 架构的限制加到 `test_flash_attn_fwd.py` 的 `run_config` 开头（原 392-402 行），无法处理的规模直接报错并说明原因。
- **背景**：原校验只有 `seq_len % 128 != 0`（**已不准**：错杀 non-causal seq_len=64 这种 N32/%64 合法值，也错杀 dualwave 的非 128 对齐 `seq_len>=384`），且无 dtype / arch 检查。
- **改动（`tests/kernels/test_flash_attn_fwd.py` 的 `run_config`，与 `flash_attn_generic.py` 的 kernel guard 对齐）**：
  1. **GPU 架构**：用 `torch.cuda.get_device_properties(0).gcnArchName` 取 arch，要求 `gfx942`/`gfx950`（CDNA3/CDNA4），否则报错。
  2. **dtype**：必须 `f16`/`bf16`。
  3. **head_dim**：`%32==0 且 >=64`（DUALWAVE_SWP 快路径还需 ==128，其余 head_dim 走 generic）。
  4. **GQA**：`num_heads % num_kv_heads == 0`。
  5. **seq_len**（按将要走的路径判定）：
     - 快路径可用（gfx950 + head_dim==128 + f16/bf16）且 `seq_len>=384` → 任意对齐，放行。
     - 否则走 generic 兜底：要求 `seq_len>=1` 且 `seq_len % _kv_align == 0`，`_kv_align = 128`（causal+D128+f16/bf16，N128 路径）否则 `64`（N32）。报错信息含 head_dim/dtype/causal 和"或 seq_len>=384 走快路径"的提示。
- **验证**（容器 hyg_trn_rocm7.1 / gfx950 / GPU2，H=4 D=128 bf16）：
  - seq=64 causal → **ERROR**「must be a multiple of 128」✅
  - seq=64 non-causal → **PASS**（N32 %64）✅
  - seq=300 causal → **ERROR**（<384 且 %128≠0）✅
  - seq=513 causal → **PASS**（dualwave ≥384，非 128 对齐 OK）✅
  - seq=256 causal → **PASS**（兜底 %128）✅
- **状态**：✅ 完成。**未 commit**。改动文件：`tests/kernels/test_flash_attn_fwd.py`。

#### 补充（2026-06-13）：校验改为直接 raise

- **要求**：`run_config` 里对不支持的规模，**直接 `raise` 报错**（而非返回 `{"err":...}` dict）。
- **改动（`tests/kernels/test_flash_attn_fwd.py`）**：
  1. `run_config` 的 5 个校验（arch / dtype / head_dim / GQA / seq_len）全部改成 `raise ValueError(<原因>)`。
  2. **compare 与 normal 模式**都用 `try/except` 捕获该 raise，并**显式 `print` 出原因**（`[FlyDSL unsupported] <cfg>: <reason>`），再分别标 FlyDSL 列 `ERR` / 打印 ERROR 行 —— 既不崩溃整 sweep，又能看到原因。
     - compare：`run_config`/`run_splitk_config` 调用包 try → 打印原因 + FlyDSL 列 ERR。
     - normal：raise 走 `except`（打印原因 + ERROR 行）；`run_splitk_config` 的 err-return 走 `if "err" in r` 分支（同样打印原因 + ERROR 行）。
     - 演进：起初 compare 无 try → raise 整体崩溃；再改成只标 ERR 单元格 → "看不到报错"；最终 compare+normal 都 try + 显式打印原因（当前）。
- **验证**（gfx950 / GPU2，H4 D128 bf16，seq=64）：
  - normal causal → 打印 `[FlyDSL unsupported] ...: seq_len (64) must be a multiple of 128 ...` + ERROR 行，**不崩溃** ✅
  - compare causal → 打印同样原因 + FlyDSL 列 `ERR`，其它后端照常，**不崩溃** ✅
  - non-causal → 正常跑出数 ✅
- **状态**：✅ 完成。**未 commit**。改动文件：`tests/kernels/test_flash_attn_fwd.py`。

### 任务 11 — generic kernel 支持任意 seqlen（不回退 causal 性能）（2026-06-13）

- **要求**：改 `kernels/flash_attn_generic.py` 使其支持任意 seqlen 的 QKV，且 `--causal --dtype bf16 --compare` 这个规模性能不回退；相应放宽 `test_flash_attn_fwd.py` 的校验。
- **根因回顾**：generic 兜底 kernel 的 Q-load/O-store 已对部分末尾 q-tile 做边界守卫，但 **KV 协作 load 无边界 + 无 KV padding mask** → 非对齐 seqlen 越界读 + 垃圾 key 进 softmax（见任务 9/10）。
- **改动（`kernels/flash_attn_generic.py`，纯前端，无需重编 LLVM）**：
  1. **DMA 路径 K/V 边界**：`create_buffer_resource(K/V, max_size=False, num_records_bytes=(batch+1)*seq_len*STRIDE_TOKEN_KV*2)`。越界行硬件返回 0。对齐 seqlen 等价于 max_size，**热路径 ISA 不变**。（这是 causal D128 gfx950→N128+DMA 走的路径。）
  2. **非 DMA 路径 K/V clamp**：新增 `_kv_row_clamp`，把全局 KV 行号 clamp 到 `seq_len-1`，避免裸指针越界读（用于 N32 / gfx942；causal D128 走 DMA 不受影响）。
  3. **non-causal KV padding mask**：在 score 段 `if const_expr(CAUSAL)` 的 `else` 分支，对绝对列号 `>= seq_len` 的 key 置 -inf（与 causal mask 同样的 element→column 布局）。`const_expr` 编译期选择 → **causal 构建零影响**。causal 本身靠 `kv_col > q_row` 已盖住 padding 列。
  4. **放宽 kernel guard `_guard_seqlen`**：只剩 `seq_len >= 1`（不再要求对齐）。
- **改动（`tests/kernels/test_flash_attn_fwd.py`，`run_config` 校验第 5 项）**：seq_len 只要求 `>= 1`（arch/dtype/head_dim/GQA 校验保留）。
- **重要前提**：`flash_attn_generic.py` 里 DUALWAVE_SWP 的 dispatch 块是**注释掉的**（保持现状，勿改回），所以 `build_flash_attn_func_module` 对所有规模都跑 **generic kernel = `flash_attn_generic_kernel_0`**（dump/profile 名）。本任务所有测试都在该 generic kernel 上。
- **格式修正**：之前 Windows 往返上传把文件变成 CRLF，导致 `git diff` 整文件标红；已 `sed -i 's/\r$//'` 转回 LF，现在 `git diff kernels/` 只剩真实改动（~52 增 / 45 删）。
- **验证**（容器 hyg_trn_rocm7.1 / gfx950 / `USE_CUSTOM_LLVM=0`）：
  - **正确性**（GPU1，generic kernel）：causal & non-causal × seqlen∈{64,100,128,200,256,300,384,500,512,1000,2048} 全 **PASS**（MaxErr ≤ 3.91e-3，cos 0.99999）；H64 auto 双变体(m128/m256，含 S=9000)、GQA(H64/Hkv8)、多 batch(B2 S=700) 非对齐均 **PASS**。脚本：`correctness_test_arbitrary_seqlen.sh`。
  - **性能不回退**（GPU1，配置 = `test_flash_attn_fwd.py` 第 95-97 行，causal bf16，`--iters 100`；before=git HEAD kernel `max_size` / after=本次 `num_records` 有界；**两者都是 generic kernel**）。脚本：`perf_test_95_97.sh`。

    | 配置（行 95-97） | BEFORE TFLOPS | AFTER TFLOPS | Δ |
    |---|---:|---:|---:|
    | B16 S8192 H64 Hkv64 MHA | 710.8 | 716.0 | +0.7% |
    | B16 S8192 H64 Hkv8  GQA | 716.3 | 715.6 | −0.1% |
    | B2  S1024 H64 Hkv64 MHA | 408.8 | 404.4 | −1.1% |

    全部在 ±1.1% 噪声内 → **无回退**。（对齐 seqlen 下 `num_records` 有界与 `max_size` 访存行为完全一致，本就应 perf-neutral；这也是预期结果。早前一版报的"+50%"是 dispatch 状态混淆导致 dualwave↔generic 不同 kernel 的对比，已作废。）
  - **用户原命令** `--causal --dtype bf16 --iters 100 --compare`：generic kernel 正常出数，不再报错。
- **保留的测试脚本**（供检查，`wk_sp1/` 根目录，远程+本地）：`perf_test_95_97.sh`、`correctness_test_arbitrary_seqlen.sh`。
- **状态**：✅ 完成。**未 commit**。改动文件：`kernels/flash_attn_generic.py`、`tests/kernels/test_flash_attn_fwd.py`。

### 任务 12 — Q-load 改用 DUALWAVE num_records 边界法（O-store 保留谓词，防回退）（2026-06-13）

- **要求**：generic kernel 的 Q-load/O-store 边界处理参考 `flash_attn_gfx950.py`（num_records 有界 buffer），且配置 95-97 性能不回退。
- **做法与取舍**：
  - **Q-load**：改为 `q_rsrc = create_buffer_resource(Q, num_records_bytes=(batch+1)*seq_len*STRIDE_TOKEN_Q*2)` + `buffer_ops.buffer_load(q_rsrc, g_idx, vec_width=MFMA_LANE_K, dtype=elem)`。删除原 `q_in_bounds` select、`q_row_safe` 行 clamp、`c_zero_mfma_pack`——越界行由硬件返 0。**实测 perf-neutral**。
  - **O-store**：**保留**原 `if q_in_bounds:` 谓词 + 裸 `_store_global_half`（global_store）。试过改成 num_records 有界 `buffer_store`，但 O-store 是**单元素**写（每 lane 16×D_CHUNKS 个 bf16），`buffer_store_short` 比 `global_store_short` 慢很多 → **B2 S1024 回退 -21.6%**，故不改。原谓词对部分末尾 q-tile 已正确，且无回退。
- **隔离实验**（GPU1，configs 95-97，TFLOPS）：
  - Q-buffer **+ O-buffer**（全 DUALWAVE 法）：B16 685 / B2 **316.9** → **回退**（B2 -21.6%）。
  - Q-buffer **+ O-谓词**（最终采用）：见下表，**不回退**。
- **性能验证**（GPU1，`USE_CUSTOM_LLVM=0`，`--iters 100`，before=git HEAD / after=最终；均 generic kernel）：

  | 配置（行 95-97） | BEFORE | AFTER | Δ |
  |---|---:|---:|---:|
  | B16 S8192 H64 Hkv64 MHA | 712.0 | 713.6 | +0.2% |
  | B16 S8192 H64 Hkv8  GQA | 716.0 | 716.4 | +0.1% |
  | B2  S1024 H64 Hkv64 MHA | 410.7 | 412.8 | +0.5% |

  全部 ±0.5% 噪声内 → **无回退**。
- **正确性**：`correctness_test_arbitrary_seqlen.sh` 23/23 **PASS**（causal/non-causal × 任意 seqlen、auto 双变体、GQA、多 batch）。
- **状态**：✅ 完成。**未 commit**。改动文件：`kernels/flash_attn_generic.py`。

### 任务 13 — O-store 移植 DUALWAVE 的 128-bit buffer_store_dwordx4（2026-06-13）

- **确认**：`flash_attn_gfx950.py` 的 O-store **是** `buffer_store_dwordx4`（`_buffer_store_128` → `_store_atom_128`，写进 num_records 有界的 `o_div`）。机制：`cvt_pk_bf16_f32` 把 4 个 f32 打成 2 个 16-bit dword，再 `permlane32_swap` 把本 lane 的 4 列与半波 partner(lane^32) 的 4 列融合 → 一次 store 覆盖 8 连续列（128-bit）。
- **移植到 generic kernel**（`kernels/flash_attn_generic.py`）：
  1. O-store 改为同款 `cvt_pk_bf16_f32`(bf16)/RNE-trunc(fp16) 打包 + `permlane32_swap` 半波融合 + `buffer_ops.buffer_store(o_pack_vec4i32, o_rsrc, byte_off, offset_is_bytes=True)`。每 d_chunk 2 个 dwordx4 → 每 wave 共 **8× buffer_store_dwordx4**（替代原来 16× 单元素 store）。两 kernel 的 MFMA32 输出布局一致（D_CHUNK=32, D_CHUNKS=4, lane_div_32），故 permlane 融合直接可用。
  2. O-store 边界改用 **num_records 有界 `o_rsrc`**（= `(batch+1)*seq_len*STRIDE_TOKEN_Q*2`）→ 越界行硬件丢弃，**删掉原 `if q_in_bounds:` 谓词**。这解决了任务 12 里"单元素 buffer_store 慢"的问题（现在是向量化 dwordx4）。
- **ISA 确认**：dump `flash_attn_generic_kernel_0/21_final_isa.s` 含 **8 个 `buffer_store_dwordx4`**（原为 `global_store_short` 单元素）。
- **验证**（gfx950 / `USE_CUSTOM_LLVM=0`，均 generic kernel）：
  - **正确性**：`correctness_test_arbitrary_seqlen.sh` 23/23 **PASS**（任意 seqlen、causal/non-causal、auto 双变体、GQA、多 batch）。
  - **性能不回退**（GPU1，configs 95-97，同跑 before=HEAD generic[O 单元素谓词] / after=本次[O dwordx4]）：

    | 配置（行 95-97） | BEFORE | AFTER | Δ |
    |---|---:|---:|---:|
    | B16 S8192 H64 Hkv64 MHA | 712.5 | 714.9 | +0.3% |
    | B16 S8192 H64 Hkv8  GQA | 716.4 | 725.2 | +1.2% |
    | B2  S1024 H64 Hkv64 MHA | 408.0 | 418.6 | +2.6% |

    全部 **不回退**（实测略快，dwordx4 比单元素 store 少 50% 指令）。
- **注意（git 状态）**：用户已把前序工作 commit 进 HEAD（`05e010ef`），且 **HEAD 里 DUALWAVE dispatch 是放开的**（跑 dualwave）；工作树里该 dispatch **保持注释**（跑 generic）。`perf_test_95_97.sh` 已更新为「BEFORE=HEAD kernel 但强制注释 dispatch」，确保 before/after 都是 generic kernel 的公平对比。
- **保留脚本**：`perf_test_95_97.sh`、`correctness_test_arbitrary_seqlen.sh`。
- **状态**：✅ 完成。**未 commit**。改动文件：`kernels/flash_attn_generic.py`。

<a id="sec-任务-14-dualwave_swp-kernel-支持-seq_len-12026-06-13"></a>
### 任务 14 — DUALWAVE_SWP kernel 支持 seq_len >= 1（2026-06-13）

- **要求**：`flash_attn_dualwave_swp_gfx950_kernel` 当前只支持 seq_len>=384，扩展到 seq_len>=1；性能尽量高；configs 95-97 不回退；先跑当前性能。
- **根因**：dualwave 软件流水线 = prologue(1 tile) + 2-tile 展开 loop + 3-tile drain，需 `max_num_tiles` 为**偶数且 >=4**。seq_len 小到 tile 数 <4（约 seq_len<192）时流水线 drain 出错，故原 dispatch 限 seq_len>=384。grid 是 `(H, ceil(seq_len/256), batch)`、**无 persistent 2-q-block**，部分末尾 q-block 已由 Q/O 的 num_records 边界处理 → 所以**唯一**缺口就是 tile 数下限。
- **改动（最小、低风险，无需重构流水线）**：
  1. `kernels/flash_attn_gfx950.py`：`max_num_tiles` round-even 之后再 `max(4, …)`（`select(max_num_tiles<4, 4, max_num_tiles)`）。多出来的 tile 全部越界（key >= seq_len）→ num_records 读 0 + causal/非causal padding mask 屏蔽，贡献为 0。对 seq_len 已 >=4 tile 的规模是 **no-op**。
  2. `kernels/flash_attn_generic.py`：dispatch 阈值 `_DUALWAVE_MIN_SEQ` 384 → **1**（dualwave 现可处理任意 seq_len>=1；dispatch 块在工作树里放开/uncommented，使 gfx950+D128+bf16/f16 的所有规模都走 dualwave）。
- **当前 dualwave 基线性能**（GPU1，configs 95-97，causal bf16）：B16 S8192 MHA **1150**、GQA **1247**、B2 S1024 **627** TFLOPS（远高于 generic 的 ~715，所以让小 seqlen 也走 dualwave 本身就是性能提升）。
- **验证**（gfx950 / `USE_CUSTOM_LLVM=0`）：
  - **正确性（小 seqlen 现走 dualwave）**：seq_len ∈ {1,64,128,200,256,300,383} causal+non-causal 全 **PASS**（seq_len=1 MaxErr 0）；`correctness_test_arbitrary_seqlen.sh` 23/23 PASS（全部走 dualwave）。
  - **性能不回退（95-97，before=HEAD dualwave / after=本次）**：两次重复测，B16 MHA 1135~1151 vs 1149~1153、GQA ~1246 持平、B2 630~638 vs 625~638 —— 方向不一致、幅度 ≤2%，属 GPU1 run-to-run 噪声，`max(4,…)` 对大规模是 no-op → **无回退**。
- **保留脚本**：`perf_test_95_97_dualwave.sh`（dualwave before/after）、`correctness_test_arbitrary_seqlen.sh`。
- **未做进一步小-seqlen 专门优化**：小规模本身耗时极短（µs 级），且已从 generic 切到更快的 dualwave；按"功能完成 + 不回退"为先，未引入额外路径（避免风险）。如需对小 seqlen 再压榨，可按 playbook 走 ATT-trace 分析。

#### 任务 14 修复 — 极小 non-causal seqlen 的 padding-mask 漏洞（2026-06-13）
- **现象**（tt.log）：dualwave 在 `B1 S1 / B2 S7` 的 **non-causal** 下 MaxErr 0.99/0.65（错），causal 正常，generic 正常。
- **根因**：dualwave 的 non-causal KV padding mask 只在 **epilogue（最后 3 个 tile）** 施加；prologue（tile 0）只做 causal mask。任务 14 把 tile 数下限设为 4 后，极小 seqlen 的**唯一真实 tile 就是 tile 0（prologue）**，其 padding 列（col >= seq_len）从未被屏蔽 → softmax 把 K=0 的填充列也算进去。causal 不受影响（causal mask 每个 tile 都做）。
- **修复**：在 prologue 的 non-causal 分支也调用 `_seq_pad_mask_if_needed`（带门控 `(tile+1)*BLOCK_N > seq_len`）。门控使 `seq_len >= BLOCK_N(64)` 时 tile 0 为满 tile → no-op；且改动在 `const_expr(CAUSAL)` 的 **else** 分支，causal 构建（如 configs 95-97）**完全不编译该分支 → 零影响**。未碰主循环（不加 per-tile 开销）。
- **验证**：`B1 S1 / B2 S7` non-causal bf16/fp16 全 **PASS**（S1 MaxErr 0）；causal tiny 仍 PASS；non-causal 64/200/500/2048 仍 PASS；全量 23/23 PASS；perf 95-97（causal）AFTER vs BEFORE +0.1%/+0.0%/+0.8% → **无回退**。
- **状态**：✅ 完成。**未 commit**。改动文件：`kernels/flash_attn_gfx950.py`。

- **状态**：✅ 完成。**未 commit**。改动文件：`kernels/flash_attn_gfx950.py`、`kernels/flash_attn_generic.py`。

<a id="sec-任务-15-调查-flydsl_dualwave_swp_trigger_lazy_else1-为何更快2026-06-13纯调查无代码改动"></a>
### 任务 15 — 调查 `FLYDSL_DUALWAVE_SWP_TRIGGER_LAZY_ELSE=1` 为何更快（2026-06-13，纯调查，无代码改动）

- **问题**：B1 S8192 等规模，`TRIGGER_LAZY_ELSE=1`(1551 TFLOPS) 比 `=0`(1153 TFLOPS) 快很多，为什么？
- **结论：与 lazy 分支无关，是"输入数据 → MFMA 动态功耗 → 时钟节流"导致的。**
  1. `TRIGGER_LAZY_ELSE` 改的是**测试输入数据**，不是 kernel：`=1` 时 `Q=1, K=0(tile1=80)`、V 随机；`=0` 全随机。kernel 构建相同。
  2. **不是 lazy 分支**：`DEBUG_LAZY_COUNTS` 显示两者都 ~99% 走 skip（random 0% rescale / special 1.5%）。2×2 实验（lazy×data）：special 数据在 lazy=1 和 lazy=0 上都比 random 快 ~22-26% → 数据效应与分支无关。（lazy-skip 本身另有独立的 ~15-19% 收益。）
  3. **不是 denormal**：`daz=True` 已 flush f32 denormal，且随机 ±1 bf16 的 softmax 指数范围太小（score−max≈−4..0）产生不了 denormal。
  4. **决定性证据（rocprofv3 `GRBM_GUI_ACTIVE`，即 `_cyc3.sh` / `lazy_else_cycles_proof.sh`）**：random 与 special 的**活跃周期几乎相同（~12.6M，差<1.5%）**，但 duration random 970k–1284k ns vs special 777k–933k ns（~1.25×）。同样周期、不同时间 ⇒ 差别是**时钟频率**。且每组内后续 dispatch 越来越慢（功耗累积、时钟被压）。
     - **`_cyc3.sh` 分析原理**：用 `rocprofv3 --pmc GRBM_GUI_ACTIVE --kernel-trace --output-format csv` 跑同一个 kernel，对每个 dispatch 拿到两份数据 —— `GRBM_GUI_ACTIVE` = **GPU 活跃时钟周期数**（counter_collection.csv 的 `Counter_Value`，单位是周期、与时钟无关），以及该 dispatch 的**墙钟时长** `End_Timestamp − Start_Timestamp`（ns）。两者相除得 **有效时钟 = 周期 / 时长**（再除 agent_info.csv 的 `Num_Xcc` 得真实 per-XCC GHz；相对比较时 Num_Xcc 抵消）。
     - **判据**：同一 kernel 比较两组输入 —— **周期≈相等而时间不同 ⇒ 差异是时钟频率（节流）**；**周期不同 ⇒ 是指令工作量变了**。这把"为什么变慢"一刀切成"频率 vs 工作量"两类，避免凭感觉猜。（这正是 `measure-kernel-clock` skill 固化的方法。）
  5. **第二独立验证（ATT 逐指令 `trace_segment_cycles.py`，playbook Stage 3）**：对 B1 S8192 的 dualwave kernel 各跑一次 ATT thread-trace（else0 random / else1 special，`kernel_iteration_range "[1]"`, `att_target_cu 1`，各 256 waves），seg_asm 锚 prologue-entry → epilogue(s_endpgm) 一个 full-kernel 区间，compare 模式：
     - else0 avg **173,372** cycle/wave；else1 avg **177,556** cycle/wave → **+2.4%**（指令 +1.9%）。即 else1（更快的）反而**多做** 2.4% cycle（就是偶发 rescale 分支多出的指令）。
     - 墙钟 else1 反而快 ~25%。**多做功却更快 ⇒ 唯一解释是更高时钟**，反推 `clock_else1/clock_else0 ≈ 1.024×953/708 ≈ 1.38×`。
     - 与 `_cyc3.sh` 的 GRBM_GUI_ACTIVE 互相印证：cycle 基本相等（两法分别 <1.5% / +2.4%），time 差 25% ⇒ **耗时差异仅由时钟频率（功耗节流）导致，确认成立**。
  6. **机理**：special 数据使 MFMA 操作数大量为 0（K=0⇒分数0；tile1 大 max 后 P=exp2(−huge)=0⇒P·V 的 P=0），systolic array 跳变少 ⇒ 动态功耗低 ⇒ 这个 MFMA 密集 kernel 不撞功耗墙 ⇒ 维持更高 SCLK ⇒ 更快。random 全非零 ⇒ 功耗高 ⇒ 时钟被节流 ⇒ 慢。
- **影响/提醒**：用全零/常数输入测 MFMA 密集 kernel 会**系统性高估性能**（功耗假象）。该规模真实吞吐应以随机数据的 1153 TFLOPS 为准，1551 不可信。
- **保留脚本**：`lazy_else_study_2x2.sh`（lazy×data 2×2 perf）、`lazy_else_cycles_proof.sh`（= `_cyc3.sh`，rocprofv3 周期 vs 时间）、`_att_run.sh`（ATT thread-trace else0/else1）、`lazy_else_segcycles.json`（seg_asm 配置）、`trace_segment_cycles.py`（playbook 工具）。
- **状态**：✅ 完成（调查 + ATT 二次验证）。无代码改动。

### 任务 16 — 两 kernel 跨 block 负载分布分析报告（2026-06-14，纯分析，无代码改动）
- **目标**：分析 `flash_attn_dualwave_swp_gfx950_kernel` 与 `flash_attn_splitk_combine_kernel` 的 grid→block 工作量分布，给后续负载均衡判断打底。
- **产出**：`fmha_opt_tools/dualwave_combine_load_distribution.md`（本地+远程，LF-clean）。
- **要点**：主 kernel grid=`(NUM_HEADS_Q, ceil(S/256), batch[·NUM_KV_SPLITS])`；causal 时每 block 工作量 = `min((q_block+1)*4, ceil(S/64))` KV tile（三角形，B1 S8192 → q_block0=4 vs q_block31=128,32× 跨度）；combine kernel 均匀负载。给出"大 grid 靠硬件轮转摊平三角 / 小 grid 靠 split-K 填 CU"的经验法则。
- **状态**：✅ 完成。无代码改动。

<a id="sec-任务-17-rocprofv3-att-occupancy_balance-py-实测-dualwave-负载均衡2026-06-14纯实测无代码改动"></a>
### 任务 17 — rocprofv3 ATT + `occupancy_balance.py` 实测 dualwave 负载均衡（2026-06-14，纯实测，无代码改动）
- **配置**：`--causal --dtype bf16 --batch 1 --num_heads 64 --num_kv_heads 64 --seq_len 8192 --head_dim 128 --compare`。
- **方法**：复用任务 15 的 ATT trace（`FlyDSL/thread_trace/else0/ui_output_agent_*/occupancy.json`，random 数据），上传 `occupancy_balance.py`（来自 cursor_rules）到远程,按 `(se,cu,simd,slot)` 聚合分析。
- **结果**：采样 32 个物理 CU 全 busy；`max slots/SIMD = 2`（非 grid-limited,2 block 并存藏延迟）；`duration min/med/max = 1.27M/1.39M/1.50M cycle`，`end-span=231k`，**`imbalance = 0.17`**（最慢比中位多 17%）；32/32 CU over-subscribed。
- **结论**：**基本均衡**,17% 尾部来自 causal 三角残留（大 grid 已摊掉大部分,不需 split-K）。caveat：`att_target_cu:1` 只采样 32 CU,仅作相对判断。实测已追加进 `dualwave_combine_load_distribution.md` §5。
- **保留脚本/产出**：远程 `occupancy_balance.py`、`FlyDSL/thread_trace/else0/`。
- **状态**：✅ 完成。无代码改动。

<a id="sec-任务-18-分析手写-asm-kernel-fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0-s-的负载均衡方法2026-06-14纯分析无代码改动"></a>
### 任务 18 — 分析手写 ASM kernel `fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s` 的负载均衡方法（2026-06-14，纯分析，无代码改动）
- **触发**：`test_flash_attn_fwd.py` 的 `run_exp_isa_fmha_bench`（行 1032-1129）调用 `exp_isa.fmha_asm.forward` → 该 causal asm kernel。问其如何保持负载均衡。
- **拉取备查**（本地 `FlyDSL/exp_isa/`）：`fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s`（3210 行）、`fmha_asm.py`、`fmha_asm_ext.cc`（host launch）。
- **结论：causal 三角负载折叠 / q-block 镜像配对**（静态法,非 split-K、非 atomic work-stealing）：
  1. **Host 减半 grid_x**（`fmha_asm_ext.cc:250-254`）：`tg_div = causal?2:1; grid_x = ceil(q_blocks/2)`,causal 只发一半 WG,每个 WG 干 2 个 q-block。
  2. **Kernel 跑完一个 q-block 后镜像翻转再跑第二遍**（`.s:3023-3063`,循环尾 `label_12D8`→`label_1300`）：`s2 = (q_blocks-1) - s2`,`s36<2 → s_cbranch label_00D7` 回循环顶；`q_blocks<2`(S≤256)时 `s_cselect s36,2` 跳过第二遍。
  3. **每 q-block 的 KV trip = 三角负载**（循环顶 `label_00D7` `.s:177-189`）：`s38 = clamp((q_block+1)*256 + (kv-seq), 64, kv_seq_len)`,只遍历到对角线。
  - **均衡原理**：q-block `i`(轻,`i+1` 份) 与镜像 `N-1-i`(重,`N-i` 份) 配对,**每 WG 合计 `(i+1)+(N-i)=N+1` 份,与 `i` 无关 → 恒定**,三角不均在发射前静态消除。
  - **对比 dualwave**（任务 17）：dualwave 是 1 q-block/WG,靠大 grid 硬件轮转摊平（残留 imbalance≈0.17）；此 asm kernel 在 **WG 内部**就把不均抹平,尾部更小。
  - **次要**：`.s:55-86` 由 `msk_opt=5` 驱动的 head×q-block swizzle(`gm0` 网格映射),目的是 L2 局部性/跨 head 分布,非核心负载均衡手段。
- **状态**：✅ 完成。无代码改动。

### 任务 19 — WG 内 wave→q_seq 行映射对比（dualwave kernel vs 手写 ASM kernel）（2026-06-14，纯分析，无代码改动）
- **问题**：两 kernel 一个 WG=512 thread=8 wave，分两组(wave0-3 / wave4-7)，wave_k 与 wave_{k+4} 计算的 q_seq 位置有何关系？
- **结论：两 kernel 用同一种映射**——按 `wave_id` 把 BLOCK_M=256 行切成 8 个**连续 32 行条带**，`q_row = q_start + wave_id*32 + (lane%32)`。
  - dualwave：`wave_id=tid//64`(行264)，`wave_q_offset=wave_id*32`(行277)，`q_row`(行1081-1083)；`_stagger=wave_id//4`(行274)分两组，组B 多 1 个 s_barrier(1097-98)做流水相位错位。
  - ASM：输出地址即计算归属(`.s:2883-2902`)，`wave 行基址 = s5(wave_id)*32*r_Seqs`；`v23=v22+16行`(MFMA 32x32 拆 16+16)。注意 `.s:223-224` 的 `2*wave_id*Seqs` 是 Q DMA-to-LDS 协作搬运分布，非计算归属。
- **配对关系（两 kernel 相同）**：**wave_k 与 wave_{k+4} 的 q_seq 相差正好 128 行(= BLOCK_M/2)**，`q_row(wave_{k+4})=q_row(wave_k)+128`。即组A(wave0-3)=Q 块上半 128 行、组B(wave4-7)=下半 128 行；连续二等分、组B=组A+128，非镜像/交错/重复。
- **两组都做成软件流水两相位(ping-pong)，实现不同**（⚠️ 修正初版"ASM 无相位错位"的错误结论）：
  - dualwave：`wave_id//4` 分组(行274)，组B 多 1 个 `s_barrier`(行1097-98)错开半拍 + `SETPRIO`(行100)；两组同一份代码。
  - ASM：`.s:770` `s_cmp_lt_i32 s5,4` 把 wave0-3/wave4-7 **分流到两份独立 KV 循环**：组A `label_048C`(行796,`s_setprio 0x0000`@794)、组B `label_0A6F/0A71`(行1690/1693,`s_setprio 0x0001`@1691)；两份把 `buffer_load`/`ds_read`/`v_mfma` 摆在相对共享 `s_barrier` 不同位置 → 一组算 MFMA 另一组搬数据，优先级 0 vs 1。比 dualwave 更激进(整循环体复制两份 vs 仅多一 barrier)。
- **共同点**：q_seq 空间划分一致(组B=组A+128 行)；两组都错开成流水两相位。
- **产出**：写入 `dualwave_combine_load_distribution.md` §6。拉取备查文件在本地 `FlyDSL/exp_isa/`。
- **状态**：✅ 完成。无代码改动。

<a id="sec-任务-20-把-asm-的-causal-三角负载折叠q-block-镜像配对移植进-flash_attn_dualwave_swp_gfx950_kernel2026-06-14"></a>
### 任务 20 — 把 ASM 的 causal 三角负载折叠(q-block 镜像配对)移植进 `flash_attn_dualwave_swp_gfx950_kernel`（2026-06-14）
- **目标**：移植 ASM `..._msk1_gm0.s` 的负载均衡法(grid_y 减半，每 WG 做 q-block i + 镜像 N-1-i)；确保 `test_flash_attn_fwd.py` 行95-97 三规模(causal bf16，`--compare`)不回退。
- **实现**（`kernels/flash_attn_gfx950.py`）：
  - 把 q-block 相关计算(q_start / q_gmem / max_num_tiles / split)+ 整个 580 行执行体包进 `for _pass in range_constexpr(_n_pass):`，`_pass==0`→base、`_pass==1`→`(N-1)-base`。用脚本 `apply_qblock_fold.py` 做 +4 缩进保证可靠。
  - **关键坑**：外层 pass-loop 必须用 `range_constexpr`(不能用 `for x in <list>`)。FlyDSL `ast_rewriter.py:1224 visit_For` 只对 range/range_constexpr 迭代体递归；plain Python for 不下钻 → 内层 DSL `range(...,init=)` KV 循环不会被转成 scf.for(报 `range() takes no keyword arguments`)。
  - launch grid_y：`CAUSAL_FOLD and CAUSAL and not SPLITK` 时 `ceil(N/2)` 否则全量。
  - 新增 build 旗标 `dualwave_swp_causal_fold`（**默认 False**）；`generic` 经 env `FLYDSL_DUALWAVE_CAUSAL_FOLD=1` 透传(`kernels/flash_attn_generic.py`)。默认 False → `range_constexpr(1)` == 原单趟体(逐字节同 codegen)。
- **正确性**（`fold_correct.sh`，FOLD ON）：偶数 nqb=8/6/4、**奇数 nqb=5**(自配对中块，幂等重算)、**部分末块** S1100/S1280 全部 `MaxErr=3.91e-03`，与 FOLD OFF / 参考完全一致。
- **性能结论：该 kernel 上折叠恒为负优化,故默认关闭**：
  - always-on 折叠在三规模回退 ~1.7-3.8%(best-of-3 A/B：1146→1126 / 1260→1227 / 637→621)。
  - 运行时 scf.if 门控更糟(B2.S1024 -15%)：`range_constexpr` 必然 trace 出 2 份体,被 false 的 scf.if 包住仍占寄存器/占用率。
  - 小 grid 反而更惨(本以为受益处)：B1.S2048.H8 157 vs 205(-23%)、B1.S1100 59 vs 98(-40%)。原因：dualwave 已靠硬件轮转+8-wave ping-pong+lazy-rescale 把三角负载摊平(任务17 实测 imbalance≈0.17);静态折叠只把本可并行的 block 串行化并加 q-block 间气泡。ASM 需要折叠是因其结构不同。
- **无回退验证**（默认 OFF，精确用户命令 `--compare`，best-of-3 A/B）：AFTER≈BEFORE(1141 vs 1145 / 1260 vs 1250 / 648 vs 637,均在噪声内)。✅
- **保留脚本**：`apply_qblock_fold.py`(缩进变换)、`fold_perf.sh`(精确用户命令)、`fold_ab.sh`(baseline↔fold best-of-N A/B)、`fold_correct.sh`(正确性:奇偶/部分块)、`flash_attn_gfx950.py.bak`(基线备份)；均本地+远程。
- **状态**：✅ 完成。代码已改(未 commit)：`flash_attn_gfx950.py`(折叠,默认关)、`flash_attn_generic.py`(env 透传)。方法已移植且正确,因恒为负优化默认关闭。
- **后续**：用户要求把这两文件改动放入 `git stash`(远程 `FlyDSL` 仓库)→ 已存为 `stash@{0}`(msg: "causal triangle-fold ... + dualwave_swp_causal_fold flag (default off) + generic env passthrough"),远程工作树 `kernels/` 回到 HEAD;本地仍保留改动 + `flash_attn_gfx950.py.bak`。恢复:`git stash pop stash@{0}`。

<a id="sec-任务-21-分析手写-asm-kernel-fmha_fwd_---_msk1_gm0-s-是否支持-qkv-varlen2026-06-14纯分析无代码改动"></a>
### 任务 21 — 分析手写 ASM kernel `fmha_fwd_..._msk1_gm0.s` 是否支持 QKV varlen（2026-06-14，纯分析，无代码改动）
- **结论：不支持 varlen**，是密集等长 batch(dense/padded)布局。
- **证据**：
  1. kernarg 有 4 个变长指针槽 `qseq`(off432→s[72:73])/`kseq`(448→s[74:75])/`qseq_padding`(480→s[84:85])/`kseq_padding`(496→s[86:87]),但**load 后在 3210 行里从未作为操作数使用**(cu_seqlens 从没被读)。
  2. 每 batch 基址 = `s4(workgroup_id_z=batch) × 固定 batch stride`(行141-167:`s4*s33`q、`s4*s49`k、`s4*s78`v、`s4*s81`LSE),无"从 cu_seqlens[batch] 取可变起点"。
  3. 长度是**单标量** `seq_len`(s30)/`kv_seq_len`(s7)对所有 batch 通用(causal 上界 `.s:178-189`、列 mask `.s:842-843`),非逐 batch `cu[b+1]-cu[b]`。
  4. host/Python 也是稠密 4D:`fmha_asm_ext.cc` 把 4 个变长指针写死 `nullptr`;`fmha_asm.py forward()` 收 `[B,S,H,D]` contiguous，强制 `seq_len%256==0`、k.shape==v.shape、同一 S，无 cu_seqlens 入参。
- **这些槽位**:应为该套 FMHA kernel 共享的 kernarg ABI(变长版会用),当前 `msk1_gm0` 变体留空不实现。
- **若要支持**:① 读 `qseq/kseq` 前缀和,batch 基址改 `cu_seqlens[batch]×seq_stride`;② 每 batch 有效长度用 `cu[b+1]-cu[b]` 驱动 causal 上界/列 mask;③ host/Python 加 cu_seqlens 入参、去掉等长约束。
- **状态**：✅ 完成。无代码改动。

<a id="sec-任务-22-给-flash_attn_dualwave_swp_gfx950_kernel-加-qkv-varlenpacked-cu_seqlens2026-06-14"></a>
### 任务 22 — 给 `flash_attn_dualwave_swp_gfx950_kernel` 加 QKV varlen（packed cu_seqlens）（2026-06-14）
- **目标**：移植任务 21 总结的 varlen 法到 dualwave kernel;`build_flash_attn_func_module` 在 `num_kv_heads` 后加 `cu_seqlens_q`/`cu_seqlens_kv`;`DEFAULT_CONFIGS` 后加 4 个写死 varlen 用例;改动同步到 `test_flash_attn_fwd_ori.py`;三规模(98-100)不回退。
- **⚠️ 注意**:开工前发现**本地 test 文件已过期**(本地~1160 行 vs 远程 1799 行/`_ori.py` 1309 行)。远程是权威版(aiter 已有 cu_seqlens_q/kv,行641-642),先把远程 test 两文件 + clean(stash 后无 fold)kernel 拉回本地再动手。
- **设计**:packed 布局 Q/O=[total_q,H,D]、K/V=[total_kv,Hkv,D];`cu_seqlens` int32 [B+1] 累加;每 batch `seqlen_q==seqlen_kv`(自注意力)。编译期 `VARLEN` 旗标——关时(默认)dense 路径**逐字节同 codegen**(仅多 2 个未用 kernarg 指针,placeholder=O,实测无回退)。`varlen` 与 split-K 互斥。
- **kernel(`flash_attn_gfx950.py`)**:build 加 `varlen=False`→`VARLEN`;kernel 签名加 `CuSeqQ/CuSeqKv`;`const_expr(VARLEN)` 下读 `cu_q[z]/cu_q[z+1]`(buffer_ops.buffer_load)得 `q_tok_base/q_tok_end/seqlen_q`、kv 同;把 `batch_idx*seq_len` 全部换成 `q_tok_base/kv_tok_base`,num_records 换 `*_tok_end`,`num_kv_tiles`/列 mask 用 `seqlen_kv`;OOB q-block 跳过**复用现有 `with _split_guard`**(VARLEN→`scf.if(q_start<seqlen_q)`,免重缩进)。causal 是 batch-local(q_start/列位置)→无需改。
- **plumbing**:`launch_flash_attn_dualwave_swp` 加 `CuSeqQ/CuSeqKv`(grid_y=ceil(max_seqlen/256) 不变);`_launch/_compile` 加 `cu_seqlens_q/kv`(dense 用 O 占位);`build_flash_attn_func_module_primary` 加 `cu_seqlens_q/kv`→`varlen=(cu is not None)`,dispatch 转发捕获的 cu;非 dualwave 路径用 varlen→raise。
- **测试**:两文件 `run_config` 加 `varlen_seqlens`(构 cu+packed q/k/v,per-batch SDPA 参考,per-batch FLOPs);`VARLEN_CONFIGS` 4 例 = `[512,256,1024,128]`MHA / `[300,700,500]`非倍数 / `[1024,1024]`GQA / `[200,64,900,333]`小+非倍数;`_run_varlen_section` 自带表(compare+normal 两模式都调)。
- **结果**:
  - **正确性**:4 例 × {causal, non-causal} 全 PASS,MaxErr causal 3.91e-3 / non-causal ≤1.95e-3,与 per-batch SDPA 一致(含奇数长度、非 256/64 倍数、部分末块、GQA、seqlen<256)。`_ori.py` 直测也 PASS。
  - **无回退**(默认 dense,精确用户命令 `--compare`,best-of-3):16,8192,64,64=1150.0 / 16,8192,64,8=1253.5 / 2,1024,64,64=654.1,vs 基线 1147.4/1256.0/640.0 —— 均在噪声内(部分略好)。✅
- **踩坑**:FlyDSL `ast_rewriter` 对 `for...in range_constexpr/range` 才下钻递归(任务 20 经验复用);此处无外层 loop,无碍。
- **保留脚本**:`fold_perf.sh`/`fold_ab.sh`(复用)、`fold_varlen_nc.sh`(non-causal varlen)、`fold_perf_x2.sh`(dense 重复)、`ori_varlen_smoke.sh`(_ori varlen 直测);本地+远程。
- **状态**:✅ 完成。代码已改(未 commit):`flash_attn_gfx950.py`、`flash_attn_generic.py`、`tests/kernels/test_flash_attn_fwd.py`、`tests/kernels/test_flash_attn_fwd_ori.py`。(注:任务 20 的 fold 仍在远程 `stash@{0}`,与本 varlen 改动独立。)

<a id="sec-任务-23-flydsl-modern-rewrite-skill-重构-dualwave-combine-两-kernel2026-06-15"></a>
### 任务 23 — `/flydsl-modern-rewrite` skill 重构 dualwave + combine 两 kernel（2026-06-15）
- **目标**:按 flydsl-modern-rewrite 规则把两 kernel 不符合规则的部分改成 Layout/Atom + 现代控制流(换说法不换语义);三规模(98-100)不回退。
- **勘察**:dualwave kernel 已高度现代化;遗留只在少量点。按规则分组:
  - **dualwave §5(主路径,影响 98-100)**:`arith.index_cast/arith.constant` → `fx.Int32/fx.Index`(行281-287 wave-id、行1055 debug)。
  - **combine kernel(split-K 才启动,对 98-100 零风险)**:§1 把 m/l 的 f32 读从 raw llvm 全局指针(`create_llvm_ptr(as=0)`+`get_element_ptr`+`llvm.LoadOp`)改为复用已有 `ws_div` Layout view + `BufferCopy32b` copy atom;§4 把手搓 `scf.IfOp+ir.InsertionPoint+YieldOp+.results` 的 per-split 累加改成 `@flyc.jit` + Python if(单出口 return)。
  - **varlen cu-reads(VARLEN 关时不入 dense ISA,零风险)**:§1 把 `buffer_ops.create_buffer_resource`+`buffer_load` 改为 `make_buffer_tensor`+`logical_divide`+`BufferCopy32b` copy atom。
- **KEEP(规则的 keep-exactly,未动)**:`.ir_value()`(Vec 合法取值,非 shim);`_if_then`+`scf.IfOp` 纯副作用 guard(已是干净 context-manager 写法,非手搓 region);`_make_raw_buffer_rsrc`(喂 keep-exactly 的 `raw_buffer_atomic_fadd`,且 debug-only);LDS 的 `create_llvm_ptr(as=3)`/`get_element_ptr`(给 ds_read,无高层形式);全部 sched/sync/transcendental。
- **踩坑(2 次)**:combine §4 的 `@flyc.jit`:① 不能在每个分支各写 `return`(jit 返回 None→`cannot unpack`);② 分支内**新建**变量(`acc_out`)在 runtime-if 下不传播(`NameError`)——必须像 `_causal_mask_prologue_if_needed` 那样:条件含调用(`fx.Float32(...)>...`)以被识别为 dynamic、**重赋已存在的 acc/den 形参**、分支后**单一 return**(复用 [[flydsl-iterate-gotchas]] 的 AST if-rewrite 经验)。
- **验证(规则 Tier-1 字节级)**:dump dualwave dense `21_final_isa.s`,重构前后 `sha256` **完全一致**(`68dbb431...`)⇒ 98-100 路径**数学上不可能回退**。实测三规模 best≈基线(16,8192,64,64=1148.8 / 16,8192,64,8=1254.0 / 2,1024,64,64=640.8 vs 1147.4/1256.0/640.0;单次 1124 是时钟噪声,ISA 同字节)。combine:split-K(splits=4)`MaxErr=3.91e-3` PASS。varlen:4 例 causal 全 PASS。
- **保留脚本**:`dump_isa.sh`(Tier-1 ISA sha256)、`refactor_verify.sh`(三规模+split-K)、`splitk_dbg.sh`;本地+远程。
- **状态**:✅ 完成。仅改 `flash_attn_gfx950.py`(未 commit)。

<a id="sec-任务-24-整理-gfx950cdna4mi350x-mi355x硬件资料速查文档2026-06-15"></a>
### 任务 24 — 整理 gfx950/CDNA4(MI350X & MI355X)硬件资料速查文档（2026-06-15）
- **输入/要求**:据 `flydsl-align-reference-kernel.mdc` (125-135) 指向的硬件资料,回答 XCC↔XCD 关系、每芯 XCD/XCC/SE/CU 数、每 CU 的 SIMD/SQ/SQC 数;MI350X/MI355X 支持的数据类型与算力峰值、HBM 容量/带宽、Infinity Cache 容量/带宽、Infinity Fabric 作用、L2/L1/L0 容量带宽;重要频率参数;两者区别。结果写进 `FlyDSL/fmha_opt_tools` 一个文档。
- **资料源**:`cursor_rules/fmha_flydsl_new_api_opt/.cursor/rules/` 下 `amd-cdna-4-architecture-whitepaper.txt`、`amd-cdna-3-white-paper.txt`、`mi300-mi350-hw-docs.txt`(+index)。
- **改动**:新建 `FlyDSL/fmha_opt_tools/gfx950_MI350_MI355_hardware_notes.md`(纯文档,无代码)。每条结论给文件+行号,区分【文档】/【推算】(per-cycle 宽度×频率换算)。
- **关键结论**:XCD=物理芯粒、XCC=ACCELERATED COMPUTE CORE,本代 1:1(corpus:60624/60684);8 XCD,每 XCD 4 SE,每 SE 9 CU(32 激活),共 256 CU、16384 SP、1024 Matrix Core;每 CU 4 SIMD(corpus:521616)、1 SQ、SQC 每 2 CU 共享(WP:148);CDNA4 无独立 L0(澄清);新增 MXFP8/6/4,TF32 移出硬件;L1 32KiB/L2 4MB(per XCD)/Infinity Cache 256MB;HBM3E 288GB/8TB/s;IF 片内胶水+片间 scale-up(>1TB/s,1075.2 P2P);MI350X(风冷1000W/2.2GHz) vs MI355X(液冷1400W/2.4GHz),规模一致仅频率差。
- **踩坑**:whitepaper 未直接写 SIMD/SQ/SQC 数与 L2/L1 聚合带宽——SIMD/SQ 由 corpus SQ/SP block spec 佐证,带宽由文档 per-cycle 宽度×峰值频率推算并显式标【推算】。CDNA 无 RDNA 式 L0,已在文档专节说明。
- **状态**:✅ 完成(纯文档,无需 commit/编译验证)。
- **未决**:文档目前仅本地;如需同步远程 base 目录可后续上传。

<a id="sec-任务-25-dualwave-kernel-支持-seqlen_q-seqlen_kvbottom-right-causal2026-06-16"></a>
### 任务 25 — dualwave kernel 支持 seqlen_q != seqlen_kv(bottom-right causal)(2026-06-16)
- **输入/要求**:`flash_attn_gfx950.py` 支持 seqlen_q≠seqlen_kv,causal mask 对齐右下角(bottom-right);全 0 的行输出 0。规模 84/86/87 不回退(`--iters 100`);两测试文件加 cross-length 测试集(Sq×Skv ∈ {31,65,100,127,1024,8192}²,bf16/fp16,causal T/F,B1 H64 Hkv8 D128);`HIP_VISIBLE_DEVICES=1`。
- **关键发现**:gfx950/D128 下 `_DUALWAVE_MIN_SEQ=1`,所有规模(含 seqlen31)都走 dualwave kernel,generic 仅 fallback。故只改 dualwave + 透传。
- **改动**:
  - `kernels/flash_attn_gfx950.py`:kernel/launch 链加 `seq_len_kv`;dense 分支 kv 寻址用 seq_len_kv;`delta_i32=seqlen_kv-seqlen_q`;causal tile 上界用 delta(clamp≥0);逐元素 mask `q_row+delta`;prologue mask guard `+delta`;新增 `c_neg_floor=-3e38`,prologue row-max 取 `max(m,floor)`(仅 causal)以避免全掩码行 `-inf-(-inf)=NaN`(归一化处已有 l==0→O=0)。
  - `kernels/flash_attn_generic.py`:dispatch 透传 seq_len_kv 给 dualwave;fallback 包一层,seq_len_kv≠seq_len 时报 NotImplementedError。
  - 两测试文件:`pytorch_ref_attention_qkv_diff`(bottom-right 参考 + nan_to_num 零行)、`run_config_qkv_diff`、`_run_qkv_diff_section`(每 (dtype,causal) 建一次 module 复用),接入 main。
- **踩坑**:零行用例首轮 FAIL——`max_err=3.9e-3`(正常 bf16)但 `min_cos=0`:`cosine(0,0)=0` 误判。修正:cos 仅对非零参考行求 min,零行单独校验 `result≈0`。**kernel 本身正确**。
- **验证(HIP_VISIBLE_DEVICES=1)**:
  - 无回退:84/86/87 × bf16/fp16 × causal/nocausal,before/after TFLOPS 全在 ±1% 噪声内(如 16/8192/64/64 bf16 causal 1143.0→1149.2;2/1024/64/64 bf16 causal 630.7→646.0),MaxErr 逐位一致。
  - 正确性:cross-length 144 例全 PASS(FAIL=0),含 Sq>Skv 零行用例(8192×31 causal min_cos=1.0 MaxErr=3.91e-3)。
  - `black` + `ruff` 干净。
- **保留脚本**:`perf_noreg.sh`、`run_qkv_diff.sh`、`run_qkv_diff_driver.py`(本地+远程)。
- **状态**:✅ 完成。改动未 commit。

<a id="sec-任务-26-seqlen_q-seqlen_kv-扩展到变长varlen-修复-delta128-mask-bug2026-06-16"></a>
### 任务 26 — seqlen_q != seqlen_kv 扩展到变长(varlen) + 修复 delta%128 mask bug(2026-06-16)
- **输入/要求**:任务 25 的 seqlen_q≠seqlen_kv 仅覆盖定长;扩展到 varlen(packed cu_seqlens,per-batch q≠kv,bottom-right causal);两测试文件加 varlen 变长单测。
- **发现的真 bug(任务 25 漏网)**:cross-length causal 在 `delta % 128 != 0` 且 kv 多 tile(loop 非空)时结果错(max_err 4e-2~1.1)。根因:dualwave 流水线每轮处理 2 个 kv-tile(`v_s_1`=tile j-2、`v_s_0`=tile j-1),但**循环里只对 `v_s_0` 加 causal mask,`v_s_1` 没加**。self-attn 时对角带恒落在 v_s_0/prologue/epilogue(周期=128=2 tile),故一直没暴露;q≠kv 时对角带相位错开,落到未掩码的 v_s_1 槽 → 错。任务 25 的 dense 用例恰好都满足 delta%128==0(1024/8192 是 64 倍数)或 kv 小到 loop 为空,所以全过。
- **修复**:循环 cluster3 给 `v_s_1` 也加 `_causal_mask_prologue_if_needed(v_s_1, j_idx-2, (j_idx-1)*BLOCK_N)`(带 guard,满掩码 tile 时 no-op)。
- **性能问题与最终方案**:该 mask 直接常开会让 causal 回退 ~2-3%(每轮多一个 scf.if)。改为 **build-time `cross_seqlen` 旗标**(默认 False):自注意力/默认路径不发射该 mask → 零回退;cross-length(dense 与 varlen)build 时传 `cross_seqlen=True`。透传链:`build_flash_attn_func_module(cross_seqlen=)` → dualwave build → kernel `CROSS_SEQLEN` const → gate `v_s_1` mask。
- **改动**:
  - `flash_attn_gfx950.py`:`build_*_module(cross_seqlen=False)`、`CROSS_SEQLEN` const、`if const_expr(CAUSAL and CROSS_SEQLEN)` gate v_s_1 mask。
  - `flash_attn_generic.py`:`build_flash_attn_func_module_primary(cross_seqlen=False)` 透传给 dualwave build。
  - 两测试文件:`build_qkv_diff_module` 传 `cross_seqlen=True`;新增 `VARLEN_DIFF_CONFIGS`、`run_config_varlen_diff`、`_run_varlen_diff_section`(零行 cos 用非零行/零行≈0 判定),接入 main。
- **验证(HIP_VISIBLE_DEVICES=1)**:
  - delta 扫描(q=256/512,kv 192~1024):18/18 PASS(修复前 delta%128≠0 全 FAIL)。
  - varlen 变长 cross-length:16 例(4 config×2 dtype×2 causal)全 PASS,含 [1024,31]/[31,1024] 零行、[300,700,500]/[700,300,500] 多 tile 非 64 倍数。
  - dense cross-length 144 例:仍全 PASS。
  - 无回退:84/86/87 × bf16/fp16 × causal/nocausal,cross_seqlen 默认 off,before/after ±1% 噪声内(causal 回退已消除)。
  - `black` + `ruff` 干净。
- **保留脚本**:`varlen_diff_smoke.py`、`varlen_diag.py`、`dense_diag.py`、`delta_sweep.py`、`mask_probe.py`、`run_varlen_diff_driver.py`(+各 run_*.sh)。
- **状态**:✅ 完成。改动未 commit。

### 任务 27 — flash_attn L2 wrapper + 测试函数合并重构（2026-06-18）

#### 背景与目标

参照 aiter 中 `fmha_kernels.py` + `moe_kernels.py` 的三层封装模式（L0 kernel / L1 编译器 / L2 torch API），为 FlyDSL fmha 补充缺失的 L2 wrapper 层，并将测试文件中散乱的4个 `run_*` 函数合并为1个。

#### 改动内容

**新建 `kernels/flash_attn_interface.py`（352 行）**

`flydsl_flash_attn_func(q, k, v, *, causal, num_kv_heads, cu_seqlens_q/kv, num_kv_splits, out, waves_per_eu, daz, ..., debug_counts, stream)` 提供：
- `@functools.lru_cache(maxsize=256)` 三路 build 缓存：`_build_dense`（通过 `build_flash_attn_func_module`，gfx942 兼容）、`_build_varlen`（dualwave 直连，varlen=True）、`_build_splitk`（dualwave 直连，num_kv_splits>1）。Build key = 所有编译期标量；cu_seqlens/workspace/debug_counts 在 launch 时传。
- 自动推导 `cross_seqlen`：dense → `Sq != Skv`；varlen → `(sq_per_batch != skv_per_batch).any()`，调用方无感。
- split-K：自动分配+零初始化 fp32 workspace；4GiB descriptor guard 直接报错。
- 统一校验：is_cuda / 同 device / dtype in {bf16,f16} / head_dim≥64 且 %32 / GQA 整除 / varlen+splitK 互斥。
- `debug_counts` 张量参数：传入时 → 以 `dualwave_swp_debug_lazy_counts=True` 构建，launch 时传入。

**`tests/kernels/test_flash_attn_fwd.py` / `test_flash_attn_fwd_extra.py`**

将 `run_config` / `run_config_qkv_diff` / `run_config_varlen_diff` / `run_splitk_config` 四函数合并为单一 `run_attn_config(num_heads, head_dim, dtype, causal, warmup, iters, *, batch, seqlen_q, seqlen_kv, varlen_seqlens_q, varlen_seqlens_kv, num_kv_heads, num_kv_splits, seed, dtype_str, verbose)`：
- 通过参数组合覆盖所有模式：dense self-attn / dense cross-attn / varlen self-attn / varlen cross-attn / split-K。
- 统一参考：`pytorch_ref_attention_qkv_diff`（bottom-right，delta=0 等价 SDPA，含 GQA/零行），不再维护多份。
- 统一 metric：zero-row-safe cosine + 零行≈0 双重校验。
- 统一 FLOPs 计算：按实际注意对（bottom-right causal 三角 or 全矩阵）精确计算。
- 保留两个 env 调试路径：`FLYDSL_DUALWAVE_SWP_DEBUG_LAZY_COUNTS`（dense，debug_counts 参数传给 wrapper）、`FLYDSL_DUALWAVE_SWP_TRIGGER_LAZY_ELSE`（dense，构造 Q=1/K 特殊输入）。
- 不动：`run_aiter_bench`、`run_opus_attn_bench`、`run_exp_isa_*`；`--compare` aiter 对比路径保留；`verbose` md5 输出保留。
- 三个 section 函数（`_run_varlen_section`、`_run_qkv_diff_section`、`_run_varlen_diff_section`）及 main 两路循环（compare/normal）全部改用 `run_attn_config`，删去 `build_qkv_diff_module` 等中间辅助。
- Import 简化：仅从 `kernels.flash_attn_interface` 导入 `flydsl_flash_attn_func`；`dualwave_splitk_workspace_elems` 仅用于 split-K SKIP guard。

<a id="sec-性能数据gpuamd-instinct-mi355x-gfx950hip_visible_devices3"></a>
#### 性能数据（GPU：AMD Instinct MI355X gfx950，`HIP_VISIBLE_DEVICES=3`）

- 测试命令：`python tests/kernels/test_flash_attn_fwd.py --batch B --seq_len S --num_heads H --num_kv_heads Hkv --head_dim 128 --dtype bf16 --iters 100 --compare`（causal + nocausal 均跑）
- BEFORE = HEAD 版（重构前 `run_config` 调用链）；AFTER = 工作区版（`run_attn_config` + `flydsl_flash_attn_func` wrapper）
- 数据来源：`tt2.log`（`perf_compare_ref.sh` 输出）

| Config | causal | BEFORE FlyDSL | AFTER FlyDSL | Δ | aiter_ck | aiter_asm | Fly/ck (AFTER) | Fly/asm (AFTER) |
|---|---|---|---|---|---|---|---|---|
| B16/S8192/H64/Hkv64 | ✓ | 1125.0 T | 1122.0 T | −0.3% | 855.2 T | 1153.6 T | 131.2% | 97.3% |
| B16/S8192/H64/Hkv64 | ✗ | 1224.1 T | 1204.9 T | −1.6% | 915.2 T | 1206.3 T | 131.7% | 99.9% |
| B16/S8192/H64/Hkv8  | ✓ | 1211.9 T | 1210.0 T | −0.2% | 889.1 T | 1152.1 T | 136.1% | 105.0% |
| B16/S8192/H64/Hkv8  | ✗ | 1277.1 T | 1265.2 T | −0.9% | 872.2 T | 1217.6 T | 145.1% | 103.9% |
| B2/S1024/H64/Hkv64  | ✓ |  630.6 T |  637.2 T | +1.0% | 489.8 T |  639.1 T | 130.1% |  99.7% |
| B2/S1024/H64/Hkv64  | ✗ |  983.0 T |  972.6 T | −1.1% | 695.3 T |  840.9 T | 139.9% | 115.7% |

- **全部6条规模**：重构前后 Δ ≤ 1.6%，全在 run-to-run 噪声范围内，**无性能回退**。
- B2/S1024 数据更新说明：上一轮测量（−9.1%）系 GPU 负载异常所致；用户在 `HIP_VISIBLE_DEVICES=3` 重测后 nocausal 为 972.6 T（vs BEFORE 983.0 T，Δ −1.1%），causal 为 637.2 T（vs BEFORE 630.6 T，Δ +1.0%），均属正常噪声。

#### 正确性验证
default sweep + varlen + qkv_diff(144) + varlen_diff(16) 全部 PASS；`black` + `ruff check` 干净。

#### 状态
✅ 完成。改动未 commit（`kernels/flash_attn_interface.py` 新建；`test_flash_attn_fwd.py` / `test_flash_attn_fwd_extra.py` 修改）。

### 技术问答 A — 循环 cluster3 为何只对 v_s_0 做 causal mask（2026-06-17）

**问**：`flash_attn_gfx950.py` Cluster 6（L1311-1322）在 `CAUSAL=True` 时只对 `v_s_0` 做
`_causal_mask_prologue_if_needed`，`v_s_1`（Cluster 3）在 `CROSS_SEQLEN=False` 时完全不掩码，为何？

**答**：这是对"自注意力下对角带只落 v_s_0 槽"这一不变式的精确利用，并非遗漏。

流水线每轮步进 2 个 kv-tile，Cluster 1 把新 K 计算到 **v_s_1**（tile j−2），Cluster 5 把新 K 计算到
**v_s_0**（tile j−1）。自注意力时，causal 对角线的绝对列范围由
`max_num_tiles = ceil((q_start + BLOCK_M) / 64)` 的奇偶性与 2-tile 步进决定：

```
tile 编号: 0(prologue) 1  2  3 ... N-4  N-3  N-2  N-1
                            ↑ loop 每轮: v_s_1 v_s_0
causal 边界: 只在 tile N-1 或 N-2 附近，恒落在 epilogue / v_s_0 槽
```

对角带对应的最后有效 tile 编号在 `max_num_tiles - 1` 附近，由于步长=2，它总落在 epilogue 范围
或某次迭代的 v_s_0（奇数槽），**永远不落在 v_s_1（偶数槽）**。因此循环中 v_s_1 的所有 tile
对 guard 条件 `q_start_pos + delta < kv_end_pos` 均为 false → `_causal_mask_prologue_if_needed`
为 no-op，发不发调用 ISA 层面无差异。

当 `seqlen_q ≠ seqlen_kv`（`delta = seqlen_kv − seqlen_q ≠ 0`）时，对角线偏移打破上述对齐，
`delta % 128 ≠ 0` 时对角带相位错开，必然有 tile 落入 v_s_1 槽 → 需要 `CROSS_SEQLEN=True` 补掩。
这正是任务 26 修复的根因。

---

### 技术问答 B — Epilogue 阶段处理几个 kv-tile（2026-06-17）

**问**：Epilogue 阶段处理几个 kv-tile？

**答**：**3 个**（`max_m3 = N−3`、`max_m2 = N−2`、`max_m1 = N−1`，N = `split_t_end`）。

| Epilogue cluster 对 | 变量 | tile 编号 | causal mask 位置 |
|---|---|---|---|
| C0/C1 → v_s_1 | — | N−3（max_m3） | C2：`mask(v_s_1, max_m3, max_m2*BLOCK_N)` |
| C4/C5 → v_s_0 | — | N−2（max_m2） | C6：`mask(v_s_0, max_m2, max_m1*BLOCK_N)` |
| C8/C9 → v_s_1 | — | N−1（max_m1） | C10：`mask(v_s_1, max_m1, split_t_end*BLOCK_N)` |

3 个的来源：流水线深度决定——prologue 欠债 1 个 tile，loop 退出时 v_s_1/v_s_0 各有 1 个"在飞"，
再加退出点自身欠 1 个，共 3 个未完成的 softmax+PxV 需 epilogue 排干（= 1.5 个完整 2-tile-per-iter
循环体，C0~C11 共 12 个 cluster）。

Epilogue 对 3 个 tile 全部添加了 causal mask（带 guard），这与循环中 v_s_1 无 mask
（`CROSS_SEQLEN=False`）不矛盾：guard 对"完全保留"的 tile 为 false → no-op，
只有处于对角边界的最后 1~2 个 tile 才真正发掩码指令。

---

### 技术问答 C — aiter moe 封装方式 + fmha wrapper 设计建议（2026-06-17）

**问**：aiter 如何调用 `moe_gemm_2stage.py`，fmha kernel 是否需要再封一层？

**分析：aiter 的三层封装**

| 层 | 文件 | 职责 |
|---|---|---|
| L0 内核 | `kernels/moe_gemm_2stage.py :: @flyc.kernel moe_gemm1` | 纯 FlyDSL kernel |
| L1 编译器 | 同文件 `compile_moe_gemm1(...)` → `@flyc.jit launch_moe_gemm1` | 按编译期常量 build；`@functools.lru_cache(maxsize=1024)` 缓存；launcher 内算 grid 并 `.launch()` |
| L2 torch API | `moe_kernels.py :: flydsl_moe_stage1(a, w1, ...)` | 接收 torch tensor；推导形状/dtype；校验；构造 args；查 L1 缓存；`_run_compiled(exe, args)`；输出后处理 |

aiter 还有一个 **gfx1201 版 fmha wrapper**（`fmha_kernels.py :: flydsl_flash_attn_func`，206 行），
结构更直接与当前 fmha 对应，建议以它为模板：

```python
@lru_cache(maxsize=32)
def _get_kernel(num_heads, head_dim, causal, dtype_str, waves_per_eu, daz):
    return build_flash_attn_func_module(...)   # L1

def flydsl_flash_attn_func(q, k, v, causal=False, ...):
    # 1) 校验 is_cuda / device / arch / dtype / head_dim
    # 2) BSHD 推导 B,S,H,D；seq_len padding → BLOCK_M 倍数
    # 3) with torch.cuda.device(...); current_stream
    exe = _get_kernel(num_heads, head_dim, causal, dtype_str, waves_per_eu, daz)
    exe(q_p.reshape(-1), ..., batch, seq_len_pad, stream=s)
    # 4) 切掉 padding 后 return
```

**fmha 的结论**：L0/L1（`build_*`/`_launch`）已经有了，**建议加一层 L2 wrapper**
（如 `flash_attn_api.py`），将当前散在 test 里的 reshape/dtype/校验/cu_seqlens 构造统一收敛。

**必须配套的改动**：当前 `build_flash_attn_func_module` 在 **build 时捕获 cu_seqlens 张量**，
导致 wrapper 无法用 `lru_cache`（张量不可哈希且每次变）。建议将 cu_seqlens 从 build-time 改为
**launch-time**（kernel 签名里 `CuSeqQ/CuSeqKv` 已是运行时 tensor 参数，本身就是运行时读的），
build 只按 `varlen=True` 布尔做 key，cu_seqlens 在 `exe(...)` 时传入——这与 aiter 风格一致。

其余要点：
- `cross_seqlen` 在 wrapper 内按 `Sq != Skv` 自动推导，调用方无感。
- dense 场景不需要 seq padding（kernel 已支持任意 seqlen）。
- 多 GPU 用 `with torch.cuda.device(...)` + `current_stream`。

### 任务 28 -- PR #704 5-set 全量 perf sweep（2026-06-19）

#### 输入/要求

对 PR #704（rocm_main vs _perf_main/rocm/main@1baf0d23）做全量性能对比，
覆盖5组配置集（set1-set5），含 split-K、大 seqlen、tiny seqlen 新增集。

#### 改动内容

- `_perf_main` 测试文件 `tests/kernels/test_flash_attn_fwd.py`：
  临时 patch（不 commit），将 DEFAULT_CONFIGS 替换为 rocm_main 的5组完整版本（53 configs）。
  脚本：`$BASE/patch_default_configs.py`（远端保留）。
- 性能扫描脚本：`$BASE/run_perf_sweep.sh`、`$BASE/run_both_sweeps.sh`（远端保留）。
- 结果 CSV：`$BASE/perf4.rocm_main.csv`（9:03 UTC）、`$BASE/perf4._perf_main.csv`（9:17 UTC）。
- 分析文档：`FlyDSL/perf_analysis_rocm_main_vs_main.md`（本地 + 远端）。
- PR #704 body Test Result 章节已更新。

#### 性能数据汇总（MI355X gfx950，bf16+fp16 causal，HIP_VISIBLE_DEVICES=1，iters=100）

| Set | 描述 | configs | avg delta% |
|-----|------|---------|------------|
| 1 | MHA/GQA 标准(B=16/2,S=8192/1024,H=64) | 6 | +0.9% |
| 2 | Head+batch sweep(H=8-64,S=128-8192) | 48 | +0.9% |
| 3 | Split-K(H=2/4,kv_sp=2/4) | 8 | -14.1% (*) |
| 4 | 大 seqlen(S=12k-294k,H=3/24/32) | 24 | -0.4% |
| 5 | 小 seqlen(S=1-65,H=3-7) | 16 | -1.2% |
| 全部 | 102 configs causal | 102 | -0.9% |

(*) Set 3 回退：H=2/4 split-K 配置经 wrapper 走了不同 dispatch 路径，待查。

#### 踩坑

- `git checkout` 前须 `git checkout -- .` 重置已修改文件，否则报错中止。
- 嵌套 ssh 下 `echo $!` 为空（shell 变量不跨 ssh 层传递），但进程确实在跑。

#### 状态

✅ 完成（perf sweep + 分析文档 + PR body 更新）。
Set 3 regression 作为 follow-up item 未决。

---

## 任务：paged-KV 解法 B —— per-page 重建 buffer 描述符（去除 4 GiB 上限）（2026-06-23）

#### 输入/要求

按解法 B 改 native paged 路径：用 buffer 资源描述符的 48-bit base 字段，每个 page 重建描述符，
去掉「整个 KV cache 必须 < 4 GiB」的限制（原因：B16/S8192 padding 到 16384 → cache 恰好 4 GiB，
32-bit num_records/soffset 溢出，触发 `struct.error 'i' format`）。

#### 根因

4 GiB 上限来自 buffer descriptor 的 **32-bit num_records**（以及承载 page 偏移的 32-bit soffset），
而非 48-bit base。解法 B：把 page 偏移折进描述符的 48-bit base（base' = cache_base + page_id*page_bytes），
num_records 只界定单个 page，soffset=0，仅 page 内 voffset 变化。

#### 改动内容

- `kernels/flash_attn_gfx950.py`：
  - import `fly_rocdl.TargetAddressSpace`。
  - PAGED 分支新增 `_make_page_view(base_iter, base_iter_ty, align, page_id)`：
    `ptrtoint`→i64 base，加 `page_id*page_bytes`，`inttoptr` 回 fly.ptr(global)，再 `make_ptr`
    构造 BufferDesc 指针（num_records=单页字节，flags），`make_view` 用自带的单页 1-D layout。
  - `_kv_tile_addr` PAGED 返回 `(kv_head_elem_offset, 0)`（soffset=0）；page 偏移移入描述符 base。
  - `_async_load_k/_v` PAGED 下按 page_id 构造 per-page view 再 DMA；dense 路径不变（const_expr 门控，逐字节同构）。
- `kernels/flash_attn_interface.py`：删除 `cache_bytes > 0xFFFFFFFF` 的 ValueError 守卫；
  paged 路径 K/V 传**自然 shape**（不 reshape(-1)），避免 >4GiB cache flatten 后 numel=2^31 溢出 int32 C-ABI shape。
- `tests/kernels/test_flash_attn_fwd.py`：`PAGED_KV_MIN_CONTEXT_LENGTH` 恢复 16384（使 16/8192 case 走 >4GiB 缓存验证解法 B）。

#### 踩坑

- `get_element_ptr`（LLVM GEP）不接受 `get_iter(K)` 的 `!fly.ptr`；改用 `ptrtoint`+`inttoptr` 在 fly.ptr 域做指针算术。
- `make_layout` 操作数必须 i32/i64，不能是 index；page elems 用 `fx.Int32(...)` 转换。
- C-ABI 把每个 kernel-arg tensor 的 shape 按 int32 打包；>4GiB cache flatten 成 1-D 时 numel 达 2^31 溢出 →
  paged 路径改传自然 shape（各维 < 2^31），kernel 仅取 base 指针，stride_kv_n 本就单独传参。

#### 验证结果（mi355-gpu-34 容器 hyg_fyd1，HIP_VISIBLE_DEVICES=1）

- B=2 repro（`paged_debug2.py`）：batch0/1 全部 0 行误差。
- 完整 `--block-table --page-size 64`：3 组全 PASS，MaxErr 3.91e-3。
  - 16/8192/64/64（>4GiB cache，原先 ERR）→ **1099.9 TFLOPS**（dense 基线 1150.7）
  - 16/8192/64/8 → 1210.1（基线 1249.1）
  - 2/1024/64/64 → 601.4（基线 645.4）
  - 与 dense 基线差距约 4%~7%，无回退。
- `py_compile` 三文件通过。

#### 状态

✅ 完成。未 commit（等用户批准）。

---

## 任务：split-K workspace 4 GiB 上限消除（per-split-z buffer descriptor）（2026-06-23）

#### 输入/要求

借鉴 paged-KV 解法 B，消除 split-K workspace 的 4 GiB 上限（原来
`ws_elems * 4 >= 0xFFFFFFFF` 会直接跳过大配置）。每个 per-split-z 重建
buffer descriptor，num_records 界定单个 split 的字节数，cross-split 偏移
折进 48-bit base，不受 32-bit num_records 限制。

#### 改动内容

- `kernels/flash_attn_gfx950.py`：
  - SPLITK 分支新增 `_make_ws_rsrc(byte_offset, nrec_bytes)` via
    `buffer_ops.create_buffer_resource_from_addr(ws_base + byte_offset, num_records_bytes)` —
    直接构建 raw buffer descriptor（!llvm.ptr<8>），绕过 make_ptr/make_view 路径
    （后者因 make_view layout 语义问题产生错误结果，已验证）。
  - `_ws_store_f32(f32_val, local_elem_index, rsrc)` / `_ws_store_quad_i32(dwords, local_elem_index, rsrc)` —
    改为接受 rsrc 参数并调用 `buffer_ops.buffer_store`。
  - 主 kernel 写 O_partial/Mrow/Lrow + empty-split 清零分支：用 per-split-z rsrc。
  - Combine kernel：替换 ws_div 为 `_make_ws_rsrc_c`；Mrow/Lrow 用
    `buffer_ops.buffer_load`（vec_width=1），O_partial 用 `buffer_ops.buffer_load`（vec_width=2，i32）。
- `kernels/flash_attn_interface.py`：删除 `ws_elems * 4 >= 0xFFFFFFFF` guard。
- `tests/kernels/test_flash_attn_fwd.py`：删除同一 guard（skip → 现在真实运行）。

#### 踩坑

- `make_ptr(BufferDesc)/make_view/logical_divide` 方案在数学地址上正确，但写出来
  的值是错的（per-row 误差 ~1 ULP，s=0 行对）。root cause 未确认（可能是
  make_view 的 layout/alignment 语义）；换用 `create_buffer_resource_from_addr` +
  `buffer_store`/`buffer_load` 解决。
- `T.f32()` 在 kernel emit 时（MLIR context 内）不能 re-call（F32Type 不是 callable）；
  改用 `fx.Float32.ir_type`。
- 原始 `splitk_dbg.py` 测试 fp32 ref vs bf16 内核导致 1.56e-02 的"假 bug"；
  真正的正确性检查必须用 `test_flash_attn_fwd.py`（bf16-only ref）。

#### 验证结果（mi355-gpu-34 容器 hyg_fyd1，HIP_VISIBLE_DEVICES=1）

- set1 dense（B=16/S=8192, B=16/S=8192 Hkv=8, B=2/S=1024）：全 PASS，MaxErr 3.91e-03，
  TFLOPS 1154.3/1249.5/650.3（基线 1150.7/1249.1/645.4，无回退）。
- set3 split-K：4 configs 全 PASS，MaxErr 3.91e-03，
  TFLOPS 623.6/269.0/175.3/734.6（基线 646.8/281.1/182.6/725.6，波动 <5%）。
- `py_compile` 三文件通过。

#### 状态

✅ 完成。未 commit（等用户批准）。

---

## 任务：paged 路径 Q/O 自然 shape（消除 2^31 numel 上限）（2026-06-23）

#### 输入/要求

`flash_attn_interface.py` paged 路径里 `o_flat = out.reshape(-1)`（Q 同理）会把整张
tensor 压成 1-D，numel 可达 2^31（B·Sq·H·D），溢出 C-ABI 的 int32 shape 字段。
要求保持 O 原 shape 避免溢出。

#### 根因

- C-ABI（`jit_argument.py:171` `_LayoutPlan`）把 tensor **每个维度** 按 int32 打包
  （`"i" * len(shape)`）。1-D reshape(-1) 单维=numel，≥2^31 即溢出；自然 4-D 各维都小。
- 但仅把 O 传成自然 shape 会算错（实测 max_err 1.65）：`make_buffer_tensor(O)` 会带上
  4-D layout，而 kernel 用 **flat element index** 寻址 O。
- 第二重限制：per-lane voffset `fx.Int32(o_global)` 含 `q_tok_base = batch_idx*seq_len`，
  同样在 ~2^31 溢出。只改 shape 不改寻址 = 把干净报错变成静默错误。

#### 改动内容（解法 B 同款：偏移折进 48-bit base）

- `kernels/flash_attn_gfx950.py`（仅 PAGED 分支，dense codegen 不变）：
  - 新增 `_make_qo_view(base_iter, byte_off)`：per-batch Q/O 描述符，base = tensor_base +
    q_tok_base*stride*elem_bytes（48-bit），num_records = 单 batch 字节数，layout 为
    per-batch 1-D flat；`q_div`/`o_div` 都用它。
  - `q_gmem_elem_offset` 在 PAGED 下去掉 `q_tok_base`（0-based within batch）。
  - `_global_idx_q` 在 PAGED 下 token 0-based。
  - dense `o_div`（`make_buffer_tensor(O,...)`）用 `const_expr(not PAGED)` 守卫。
- `kernels/flash_attn_interface.py`：paged 路径 Q/K/V/O 全部传自然 shape
  （`q.contiguous()` / `out.contiguous()`，不再 reshape(-1)）。

#### 效果

Q/O 寻址上限从「整张 numel < 2^31」提升到「单 batch numel < 2^31」（B× headroom）。

#### 验证结果（hyg_fyd1，HIP_VISIBLE_DEVICES=1）

- paged B=2 自然 shape：max_err 3.91e-03。
- 完整 `--block-table --page-size 64`（7 configs，含 >4GiB 16/8192 与 split-K）：全 PASS，
  MaxErr 3.91e-03，16/8192 1098.5 TFLOPS。
- set1 dense + set3 split-K：全 PASS，1152.5/1253.0/648.2（基线 1150.7/1249.1/645.4），无回退。
- `py_compile` 通过。

#### 状态

✅ 完成。未 commit（等用户批准）。

---

## 任务：非 PAGED 路径 Q/K/V/O 自然 shape（dense/varlen/split-K/generic）（2026-06-23）

#### 输入/要求

`flash_attn_interface.py` 非 PAGED 分支 `q/k/v/out.reshape(-1)` 压成 1-D，单维=numel ≥2^31
会溢出 C-ABI 的 int32 shape 字段。要求保持自然 shape。用户确认：全部覆盖（含 generic
fallback）；varlen 保持 3-D packed，token 单维溢出不在范围。

#### 根因（同 PAGED 解法 B）

C-ABI 每维按 int32 打包；且 kernel flat voffset 含 `batch_idx*seq_len`，同样在 2^31 溢出。
只改 shape 会算错（实测 1.65/1.94）。须 kernel 端把 per-batch 偏移折进描述符 48-bit base，
index 改 batch 内 0-based，num_records 改单 batch。

#### 改动内容

- `kernels/flash_attn_gfx950.py`（dualwave，dense/varlen/split-K 共用）：
  - 新增统一 `_make_rebased_view(base_iter, byte_off, nrec_bytes, layout)`；Q/O 用
    per-batch（base=q_tok_base*stride*elem_bytes，nrec=seqlen_q*stride*elem_bytes），
    非 PAGED 的 K/V 同样 per-batch（kv_tok_base）。PAGED K/V 仍用 `_make_page_view`。
  - `q_gmem_elem_offset` 去掉 q_tok_base；`kv_gmem_elem_offset` 只剩 kv_head*HEAD_DIM；
    `_global_idx_q` 全路径 0-based。`_kv_tile_addr` 非 PAGED 不变（head_off + tile*stride 在 batch 内）。
  - combine kernel：O 改 `create_buffer_resource_from_addr`（base=b*seq*stride*2）+
    `buffer_store(..., offset_is_bytes=True, o_global*2)`，o_global 去掉 b*seq。
- `kernels/flash_attn_generic.py`（fallback）：q/k/v/o `create_buffer_resource` 加
  `base_byte_offset=batch_idx*seq_len*STRIDE_TOKEN_*\*2`，num_records 改单 batch；
  `global_idx_q/kv` 与 DMA `global_row` 去掉 `batch_idx*seq_len_v`。
- `kernels/flash_attn_interface.py`（546-553）：非 PAGED 也 `q/k/v/out.contiguous()`，不再 reshape(-1)。

#### 踩坑

- combine kernel O 用 `buffer_store` 默认 element offset 按 i32（4B）缩放，但 o_global 是
  elem_dtype（2B）单位 → 错 4 倍（MaxErr 1.94）。改 `offset_is_bytes=True` 传 `o_global*2`。

#### 验证结果（hyg_fyd1，HIP_VISIBLE_DEVICES=1，清 cache）

- set1 dense：1156.8/1253.4/642.7，MaxErr 3.91e-03，无回退。
- set3 split-K：616.8/267.6/173.7/703.3，MaxErr 3.91e-03，无回退。
- generic fallback（8/256/64、4/128/32）：MaxErr 3.91e-03。
- varlen/cross-length（--extra，~40 cases）：全 PASS，MaxErr ≤3.91e-03。
- paged（--block-table，含 >4GiB 16/8192 + split-K）：全 PASS，1093.3 TFLOPS。
- `py_compile` 三文件通过。

#### 状态

✅ 完成。未 commit（等用户批准）。

---

## 任务：paged-KV 支持 split-K（修复 B=1/small-H 性能回退）（2026-06-23）

#### 输入/要求

对比 tt2.log 同配置 dense vs vllm:linear:p64，发现 split-K 规模（kv_sp=4/2）paged 相比
dense 回退 1.6~2.8×。要求修复。

#### 根因

paged 路径不支持 split-K：`_flydsl_flash_attn_paged` 在 num_kv_splits>1 时 raise，且
test 的 paged 调用没传 num_kv_splits → 这些配置在 paged 下按单 split 跑。B=1、H=2/4 时
单 split 只有 ~64 个 workgroup，GPU 严重空载（dense 用 grid_z=B*kv_sp 填满）。

#### 改动内容

- `kernels/flash_attn_gfx950.py`：放宽互斥 guard（138-139），只禁 paged+varlen，允许
  paged+split-K。kernel 里 split-K（拆 tile 范围）与 paged（per-tile 重定位 page）正交，
  workspace/block_table 是独立参数槽，combine kernel 写 dense O 与 paged 无关 → 直接共存。
- `kernels/flash_attn_interface.py`：`_build_paged` 加 num_kv_splits 透传；
  `_flydsl_flash_attn_paged` 去掉 split-K raise，加合法性校验（D=128/bf16,f16/Sq>=384），
  splitk 时分配 fp32 workspace 并 launch 传 workspace=_ws。公共入口已传 num_kv_splits。
- `tests/kernels/test_flash_attn_fwd.py`：paged 调用（launch+benchmark）补
  num_kv_splits=int(num_kv_splits)。

#### 验证结果（hyg_fyd1，HIP_VISIBLE_DEVICES=1，清 cache）

set3 paged split-K（核心，MaxErr 全 3.91e-03）：

| 配置 | dense | paged 旧(单split) | paged 新(split-K) |
|---|---|---|---|
| 1/8192/2/2/sp4 | 629.8 | 224.5 | 602.0 |
| 1/4096/2/2/sp4 | 267.9 | 105.3 | 256.3 |
| 1/2048/4/4/sp4 | 174.9 | 93.7 | 170.7 |
| 1/8192/4/4/sp2 | 709.9 | 438.2 | 684.6 |

回退消除，paged split-K 现在在 dense 的 ~2-4% 内。回归全 PASS：set1 paged 单 split
（含 >4GiB 16/8192，1099.9）、set1 dense（1147.1/1255.3/627.6）、set3 dense split-K
（610.3/173.1）、小 shape generic（39.2），MaxErr 均 3.91e-03。`py_compile` 通过。

#### 状态

✅ 完成。未 commit（等用户批准）。

---

## 任务：varlen 输入支持 paged KV（2026-06-23）

#### 输入/要求

让 varlen 打包 Q（[total_q,H,D] + cu_seqlens）与 paged K/V cache（block_table 查页）共存
（vLLM/SGLang prefill/decode 场景）。要求 set1（run.sh:64 dense）、set3（split-K）、paged
（run.sh:65 --block-table）性能不回退。

#### 根因/设计

kernel 里 VARLEN 与 PAGED 已是两个正交的 const_expr 维度：Q/O 寻址在 VARLEN 下从 cu_seqlens
读 token-base 并折进 48-bit base；K/V paged 走 per-page view + block_table 查页（用
batch_idx=block_idx.z，varlen 下即序列号）；seqlen_kv_v 在 VARLEN 分支已是 per-batch，自动
驱动 num_kv_tiles/causal/block-table 行长。组合只需删互斥 guard + interface/test 接通。

#### 改动内容

- `kernels/flash_attn_gfx950.py`：删除 137-139 的 `PAGED and varlen` raise（保留 varlen+splitk
  禁止）。kernel 其余逻辑无需改（const_expr 组合天然成立）。
- `kernels/flash_attn_interface.py`：`_build_paged` 加 `varlen` 参数透传
  `build_flash_attn_dualwave_swp_module(paged=True, varlen=True)`；`_flydsl_flash_attn_paged`
  去掉 cu_seqlens_q 的 NotImplementedError，加 varlen 子路径（3D Q 推 B/Sq、校验
  cu_seqlens_kv/max_seqlen_q、禁 split-K、launch 传 cu_seqlens_q/kv + block_table + seq_len_kv）；
  公共入口 dispatch 给 paged 透传 cu_seqlens_kv/max_seqlen_q/cross_seqlen。
- `tests/kernels/test_flash_attn_fwd.py`：paged launch + benchmark 两处在 varlen 时传
  cu_seqlens_q/kv + max_seqlen_q + cross_seqlen。

#### 验证结果（hyg_fyd1，HIP_VISIBLE_DEVICES=1，清 cache）

- **varlen+paged 正确性**（`--compare --extra --block-table --page-size 64`）：~30 个 varlen
  cases 全 PASS，含 cross-length [1024,8192]/[8192,1024]、GQA Hkv=8、MHA Hkv=64，MaxErr
  ≤3.91e-3，与 aiter_ck 参考 1.00x。
- **set1 dense**（run.sh:64）：1148.0/1253.0/638.4，无回退（基线 1150.7/1249.1/645.4）。
- **set1 paged**（run.sh:65）：1096.4/1196.1/604.8，无回退（基线 ~1093/1199/595）。
- **set3 dense split-K**（CLI 单跑 4 config）：621.9/267.6/179.9/716.8，无回退。
- `py_compile` 三文件通过。

#### 状态

✅ 完成。未 commit（等用户批准）。

---

## 任务：vectorized KV cache layout 支持（进行中，2026-06-24）

#### 需求
paged-KV 支持 vectorized layout（aiter 5D：K [NB,Hkv,D/kVS,PageSize,kVS]，
V [NB,Hkv,PageSize/kVS,D,kVS]，kVS=8），splitk/varlen/cross 全支持；set1/set3 性能不回退，
且 vectorized 与 linear 性能一致（<2%）。分阶段：阶段1先正确，阶段2/3 优化性能。

#### 阶段1（✅ 完成，端到端 compare 通过）
- `flash_attn_interface.py`：`_flydsl_flash_attn_paged` 去掉 linear-only raise，加 vectorized
  5D shape 校验（page_size 从 shape[3]、Hkv 从 shape[1]、kVS=shape[4] 提取）；`_build_paged`
  透传 `kv_cache_layout`。
- `flash_attn_gfx950.py`：builder 加 `kv_cache_layout` 参数 + `KV_VECTORIZED` const_expr 门控。
  - K：vectorized 最内仍是 8 连续 d → 只改 `_async_load_k` 的 src_elem 公式
    `h*(D/8)*64*8 + (d//8)*64*8 + n*8`，写入 LDS 字节与 linear 一致 → 下游零改动。
  - V：vectorized 最内是 8 连续 token（转置）→ 单次 128b 读取不到 8 个 d。阶段1用 gather：
    每 lane 从 raw V page rsrc 逐个 buffer_load 8 个 d，拼成 octet 用 llvm.store 写进
    linear-V 的同一 LDS 槽位（byte-identical）→ 复用 ds_read_tr 不变。
- 验证：set1 + set3(splitk) + varlen + cross 全部 MaxErr ≤ 3.91e-3。linear/dense 无回退。
- **已知性能问题**：V gather 使 vectorized 比 linear 慢 ~2-3×（set1 16/8192：408 vs ~1100 TFLOPS）。

#### 阶段2（🔄 进行中，2026-06-24，coalesced DMA + plain read 已落地，差 10–19% 待收尾）
- **方法**：先用 dump 逆向出 linear 路径 v_v 寄存器 contract（新增 `dump_v_mapping` debug flag +
  tagged-V 两遍，工具 `wk_sp1/dump_v_map.py`）。contract:
  `n=ks*16+(e//4)*8+(e%4)+(lane//32)*4, d=dc*32+(lane%32)`（完美双射 8192 cell）。
- **关键发现**：vectorized V global 本身就是 [d][ni] 转置（ni=token inner），8 连续 bf16 = 8 个
  连续 token（同 d）→ **一次 coalesced buffer_load_lds 直送 LDS，无需 gather**。新建
  [no][d][ni] V-LDS 布局。
- **代码**：`_async_load_v` 的 vec 分支换成 coalesced `_buffer_load_lds_128`；
  `_read_v_packs_for_buf` vec 分支用 plain 2×v4f16 LoadOp（非 ds_read_tr）+ shuffle 还原 contract。
  纸上推导 + vec_probe 双确认正确。
- **正确性**：vec_probe + set1/set3 vectorized compare 全 PASS，MaxErr=3.91e-3。
- **性能**：vec vs linear（FlyDSL kernel best）：16/8192/64/64 = 975 vs 1101（−11.4%）、
  16/8192/64/8 = 1084 vs 1204（−10%）、2/1024 = 487 vs 599（−18.7%）、
  1/8192/2/2 sk4 = 510 vs 590（−13.5%）、1/8192/4/4 sk2 = 572 vs 676（−15.5%）。
  比 gather 旧版（408 vs 1100 ≈ −63%）大幅改善，但**未达 <2%**。
- **剩余 gap 主因**：plain b64 read 的 ~4-way LDS bank conflict（固定 (ks,dc) 时 lane%32 沿 d 轴
  stride=8 元素）。
- **下一步**：read 换 `ds_read_b64_tr_b16`（硬件 transpose，免 conflict，linear 正用此），在新
  [no][d][ni] 布局上重 dump 求 (urv_base+imm) offset 使产出同一 contract。详见
  `vectorized_kv_handoff.md`「下一步」节。

#### 状态
🔄 进行中（阶段1完成；阶段2 coalesced DMA+plain read 已正确落地，perf 从 −63% 收到 −10~19%，
最后一档靠 ds_read_tr 免 bank conflict）。未 commit。

---

## 当前未决事项

- 以下改动**均未 commit**（等用户批准）：`kernels/flash_attn_gfx950.py`、`kernels/flash_attn_generic.py`、`tests/kernels/test_flash_attn_fwd.py`。
- 工作树里 generic 的 DUALWAVE dispatch 处于**注释**状态（为测 generic kernel）；HEAD 里是放开的。提交前需确认要哪种状态。
- 可选后续：把 `num_kv_splits` 透传进 `build_flash_attn_func_module` / generic dispatch，让 split-K 走标准测试链路（当前需直驱 dualwave 构建）。
