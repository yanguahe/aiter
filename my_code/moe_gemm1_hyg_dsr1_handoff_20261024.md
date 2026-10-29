# gfx1250 MoE GEMM1 优化交接：从 F3 继续争取低于 70 μs

更新时间：2026-10-24。本文交接给用户新窗口中的 agent。

用户最新要求是“写个 handoff 文件，将任务交接给另一个 agent，我新开个窗口继续优化”。上一窗口已完成优化收尾、英文提交、推送及 a07-3 最终复测。本窗口只生成交接文件，没有启动新的优化实验。接手窗口应从本文记录的最终代码继续争取低于 **70 μs**，无需再次询问是否要继续优化。

后续会话已更新约束：所有新复测及优化统一使用 `AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=0`（pin=0）。
当前生产源码保留行尾v2；其79.2060 μs属于pin=1历史结果，不作为pin=0成绩。
最新进展、复测命令与决策请以 `moe_gemm1_hyg_dsr1_f3_followup_20261024.md` 为准。
下文原pin=1建议及数据仅用于理解历史，不覆盖这项最新要求。

## 1. 接手时先确认这些事实

| 项目 | 当前已确认状态 |
|---|---|
| 本地仓库 | `C:/Users/yanguahe/Documents/code/wk_sp1/aiter` |
| 当前分支 | `hyg_dsr1_conc6144` |
| 当前 HEAD | `6f656cf5b02b2f1acf723949652f91b48e2933d7` |
| ROCm 分支 | `rocm/hyg_dsr1_conc6144`，已推送到同一 HEAD |
| 最优生产源码 | F3，已经落入上述 commit |
| 最终 GEMM1 中值 | pin=1：**80.7452 μs**；pin=0：81.7383 μs |
| 最终目标 | 争取低于 **70 μs**，还差 10.7452 μs |
| 当前实验状态 | 上一轮全部结束，没有需要继续等待的优化实验 |
| 源码工作区 | 交接前 `aiter/` 下无未提交修改，暂存区为空 |
| 文档和证据 | `my_code/`、`.codex_tmp/` 为保留的未跟踪内容，未纳入上述 commit |

当前 commit 的英文说明是：

```text
Fix gfx1250 A4W4 synchronization and tune GEMM1 scheduling
```

ROCm commit：<https://github.com/ROCm/aiter/commit/6f656cf5b02b2f1acf723949652f91b48e2933d7>。

本地 remote：`origin=git@github.com:yanguahe/aiter.git`，`rocm=git@github.com:ROCm/aiter.git`。不要把个人 `origin` 当成 ROCm remote。

旧文档中的 `dev/yadai_dsr1_conc6144`、`moe_a4w4_pr_refactor_wt` 和 `339589f900b3a880ec447f605886fe803cbf9649` 属于此前分支或此前 baseline 的语境。继续本任务时，对照起点应是当前 **6f656cf5b0 / F3**，不能混用旧分支、旧 workload 或旧环境的耗时。

## 2. 目标、修改范围与用户约束

主目标 workload：E64 / T1536 / topk8 / model_dim=7168 / inter_dim=2048，A4W4、SiLU、无 bias、random non-balanced routing；当前原单测实际参数是 `--data-init uniform --scale-init auto --seed 0`。模型推理背景假设 shared expert 单独计算。

继续优化时必须保留：

- 当前 GEMM1 的功能、已有 shape 支持和结果精度；不能增加额外的 E64/E65 编译 specialization 来回避通用性要求。
- GEMM2 和其他 kernel 的既有功能、shape 支持及精度。局部 GEMM1 feature 应限制在已有目标配置 guard 内，并验证 guard 外行为。
- `AITER_GROUPED_A4W4_A_PRESHUFFLE=1`。
- 原计时与精度口径，不伪造计时、不挑最快一轮、不使用 const0 特例、不放宽精度标准。
- 尽量减少代码改动行数。此前明确暂不增加 persistent、w2x4 wave 布局或新的 OPTX pipeline；可以调优现有 DS/WMMA 调度和其他独立小 feature。
- 用户后来允许 `AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1`，当前最优 F3/pin1 已使用调整过的 VGPR pin 策略。
- 用户明确纠正过“不能减少实际计算”的限制：可以跳过数学上无效的计算，只要不损害通用性与精度；仍不能减少有效计算或硬编码 seed/route。
- 同时保持 E64、E65 的 MoE 正确性；后文单独说明 E65 当前尚未满足的历史要求，不能隐瞒或误报。

若系统优化后仍达不到 70 μs，交付最佳有效结果、可复现命令及明确的瓶颈证据，不承诺已达成目标。

## 3. 远程环境与操作规则

本任务此前明确指定的环境如下，区别于 `AGENTS.md` 中通用共享环境的默认仓库/容器/分支：

| 项目 | 值 |
|---|---|
| 机器 | `a07-3` |
| SSH | `yanguahe@heliosr-1b114-a07-3.mnb.dcgpu` |
| 主机/容器共享仓库 | `/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai` |
| 容器 | `dsr1_ep4` |
| 已复测分支/HEAD | `hyg_dsr1_conc6144` / `6f656cf5b02b2f1acf723949652f91b48e2933d7` |
| FlyDSL | `0.3.3.dev882` |
| Python | `3.12.3` |
| PyTorch | `2.11.0+rocm10.1.0a20260811-bkc.20260820` |
| HIP | `7.16.26315` |
| 最后复测 boot ID | `cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f` |

这些是上次验证的状态，不代表接手时机器仍空闲、版本仍相同。先重检 HEAD、branch、container、boot ID 和软件环境；boot/时钟或负载变化时重新测当前 F3 对照，不直接沿用旧数字。

先读本地 `C:/Users/yanguahe/Documents/code/wk_sp1/AGENTS.md`。所有用户交流用中文；代码标识符、命令、路径和 ISA 名保持原样，源码注释沿用英文，新交接/调优文档用中文、UTF-8/LF。

Git 在远程**主机**执行，构建、kernel 测试和 profiler 在 `dsr1_ep4` 中执行。不要误切到通用默认仓库 `/data/yanguahe/code/wk_sp1/aiter` 或默认容器 `hyg_fyd1` 来测本任务的成绩。

优先复用长期 SSH PTY 和 SFTP。新窗口不能假定旧窗口的工具 `session_id` 有效；可重新连接，恢复已存在的 tmux：

```bash
ssh -tt -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  yanguahe@heliosr-1b114-a07-3.mnb.dcgpu
tmux new-session -A -s codex_moe_20261024_mainrun
```

旧主机控制 tmux 名为 `codex_moe_20261024_hostctl`。主 shell 和控制 shell 上一轮均已返回 prompt。终端可能重绘旧输出；以保存的日志/JSON 为准，不把重绘当成新的一次测试。

主机 Git 认证可使用：

```bash
export GIT_SSH_COMMAND='ssh -i /data/yanguahe/code/id_rsa.hyg -o IdentitiesOnly=yes'
```

此前为拉取最终 commit，仅将三个已确认的 tracked edits 保存到远端 stash：`9493cbc4e7e72e8fc6c5599850c5e01e8627651d`，之后 `git pull --ff-only rocm hyg_dsr1_conc6144`，三个文件 SHA 与本地一致。这个 stash 是保留备份，不是待 apply 的优化；未跟踪文档和实验目录都保留了。

接手时不得为了同步执行未经授权的 `git clean -fd`、`git checkout .`、reset 或其他丢弃修改的操作。先只读检查状态，只传任务相关文件，传输前核对 SFTP 两端目录，传输后校 SHA256。不要清理 `.codex_tmp/` 或批量上传整个工作区。

### GPU 空闲与超时

- 每次性能测试前，在主机检查全机所有 GPU：优先 `/data/yanguahe/code/gpu_users.sh`，同时核对 KFD owners、busy 和 VRAM；不仅检查 GPU0。
- 任一卡占用时每 6 秒继续等待，不杀外部进程，不用占用期间的耗时作性能结论。可以做不干扰性能测试的 CPU 分析或正确性验证。
- 现成等待工具：`.codex_tmp/yadai_gemm1_param_20261021/tools/wait_normal_idle_v1.sh`，要求三次连续 idle，间隔 6 秒。
- 性能计时期间不并发运行远端编译、SFTP 或额外 GPU 检查；每轮结束检查 ownership 和 settled idle，确认无本次测试残留。
- 用户要求自己启动的测试超过两分钟必须及时结束。每个 test/compiler/profiler child 使用 `timeout --signal=TERM --kill-after=5s 120s`。多轮 host runner 总时间超过两分钟不等于单个 child 超时，不能误杀正常多轮编排。
- 容器生成的私有 cache 可能是 600 权限；在 host 读取前只修复本任务对应输出目录权限，避免重复此前的 PermissionError。保留失败记录与已完成原表，不因权限失败而重跑、替换一轮有效计时。

## 4. 标准复现命令与最终原始数据

以下在 `dsr1_ep4` 的上述仓库根目录执行。性能测试前先在主机确认全机 idle。选一个未使用的独立正常 cache 目录，避免误命中旧 ISA 注入实验。

```bash
cd /data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai
unset COMPILE_ONLY FLYDSL_RUNTIME_RUN_ONLY AITER_FLYDSL_MMA_GROUP AITER_FLYDSL_MMA_FIRST_GROUP AITER_TDM_B_TH AITER_FLYDSL_DS_FIRST_N
export AITER_GROUPED_A4W4_A_PRESHUFFLE=1
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
export FLYDSL_RUNTIME_CACHE_DIR="$PWD/.codex_tmp/yadai_gemm1_param_20261021/resume_f3_pin1_20261024_01"
PYTHONPATH=$PWD AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 AITER_LOG_MORE=1 AITER_FORCE_GFX1250=1 GPU_ARCHS=gfx1250 CU_NUM=256 \
AITER_MOE_EXPERT_BALANCE=false AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
timeout --signal=TERM --kill-after=5s 120s python3 -u \
op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 \
--experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --act silu \
--no-bias --no-check-aot-cache --warmup 5 --iters 20 --seed 0 --data-init uniform --scale-init auto
```

pin0 对照只改 pin 值和对应 cache 目录。新实验使用新 label/cache；不要覆盖已封存证据。允许在计时前通过原 caller 分阶段正常预编译，但正式耗时仍来自上述原入口。旧 `my_code/test_flydsl_grouped_gemm_gfx1250.py --scenario bench` 是另一测试口径，不拿它的数值替换本任务的 `--scenario kernel` 成绩。

已推送/拉取最终 commit 的固定三轮交错顺序为 `0/1/0/1/0/1`，六轮全有效，没有按读数重跑：

| 配置 | GEMM1 三轮 μs | GEMM1 中值 μs | GEMM2 三轮 μs | GEMM2 中值 μs |
|---|---|---:|---|---:|
| pin=0 | 82.3413 / 81.7383 / 81.4917 | 81.7383 | 63.7136 / 64.3523 / 64.6011 | 64.3523 |
| pin=1 | 81.0487 / 80.7177 / 80.7452 | **80.7452** | 61.4405 / 60.7530 / 61.7589 | 61.4405 |

六轮均 `pass=True`、`logits_diff=3.61055e-06`、`rel_l2=0.00268721`，原 logits_diff gate 保持 0.01。pin1 GEMM1 中值在此次同 commit 对比中快 1.214975%。全局 pin 开关使 GEMM2 使用各自原编译版，不能把 GEMM2 的差异算作 GEMM1 源码优化收益。

另一次此前正式 F3/pin1 独立复测中值是 81.0172 μs，属于同一代码的不同时间段，旧记录保留原样。也不能将历史候选对照中的 79.9228 μs control 组或某个单轮称为新最优正式成绩。

## 5. 生产源码、真实 kernel 名与 ELF

核心源：`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`，launcher 为 `launch_gemm_a8w4_tdm`。

F3 源 SHA256：

```text
f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec
```

pin1 实际 GEMM1 完整 kernel symbol：

```text
a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K7168_e64_act1_q1r8_cluster1x4_prefetch_bth1_wpt1_mg1x2_dsfirst16_regpart_planarlds_interleavelds_rolleddrain
```

pin1 GEMM2 symbol：

```text
a8w4_tdm_fp4_apre_t256x256x256_w2x2_b4_K2048_e64_cluster1x4_prefetch_bth1_wpt1_mg8_dsfirst16_regpart_planarlds_interleavelds_rolleddrain
```

当前是原有 **t256/w2x2/b4、non-persistent**，真实 call 中 `persistent_workers=0`、`cluster_n=4`、`cluster_m=1`、`a_preshuffled=1`、`rolled_drain=1`、`tdm_b_th=1`。不是旧 OPTX/w2x4/persistent 版本。

kernel 名中的 `_dsfirst16` 来自原 `iglp_params`，不代表目标 guard 内实际仍按 16 调度；生产代码已在对应路径使用 effective DS-first5。`_mg1x2` 才是当前 first1/group2。性能分析须看源码和编译对象，不只看 suffix。

| 对象 | 完整 ELF SHA256 |
|---|---|
| F3 pin1 GEMM1 | `c5296627d336613a8a8580ec712c3d04766081e8b9022b0012333e13fb8001d2` |
| F3 pin1 GEMM2 | `a5e9c2504863d38c85e39acdc3e28b35774a99f21957b517e338953558375311` |
| F3 pin0 GEMM1 | `62037bcfa621ff1c2d378e0e5face1479e6af47fbcb5beb8bacf8ee39d003897` |
| F3 pin0 GEMM2 | `4bd626e33a25c9297fb038cfb598d965796a306c449c11782a1da54ed6b716be` |

pin1 GEMM1 资源：VGPR1024、SGPR62、LDS327680 B（320 KiB）、private0、VGPR/SGPR spills0。静态 executable / `s_wait_alu` / `s_set_vgpr_msb` 为 5701 / 118 / 750；静态模板数不等于动态独占开销。

当前已采用的关键 feature：

- 目标 guard 内 MMA group2 / first1，首 tile 使用 inline-zero accumulator；其他路径保留原行为。
- A-only standard multicast timeout；B 和其他 descriptor 行为保留。
- 小范围调度/索引优化：目标路径 bitsearch、swizzle、closing25、future DS floor、DS_B 调度记账及 effective DS-first5。
- pin1 的 C 为 `v256:767`；A/B 两份 slot 为 `v768:895`、`v896:1023`。
- current/next LDS 地址为 `v32:35`、`v36:39`，以等价 i32 表达式和四个 scalar empty tied LLVM constraints 固定。
- 这些 feature 的具体 guard 以生产源码为准，不扩散到 GEMM2、E65 fallback 或其他几何配置。

最终 commit 共提交三个源码文件（152 行新增、47 行删除）。另外两项已完成修复/移植不是待做任务：

| 文件 | 当前内容 | SHA256 |
|---|---|---|
| `aiter/ops/flydsl/kernels/moe_apreshuffled_quant_scatter.py` | compact A-preshuffle stores/consumer 的 agent-scope release/acquire 可见性修复 | `6fcf74235c7d8cf002cc77527aafbb13e99f187bb84284ababb47c3fb01ec5d9` |
| `aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py` | 兼容 FP4 A-preshuffle receive-copy 的两条 route/wave `_rpg2` 路径，其他入口保留 | `269ff9ebb03f91784a2b1d897a44e9b6ac6cab0df2539091c1ab1229e21e2a8f` |

未修改的 grouped caller SHA 为 `005ccca824a243ead8de62128228ba6a5e6c4e70ec824512b3d501d3f9914d84`；原测试脚本 SHA 为 `7ca489c63aa1f53eda349238d40e5f5edc0ae1925111b8d1fd1281365e41c5cf`。

## 6. E65：已验证内容与尚未满足的要求

用户历史上要求 E64/T1536/topk8 和 E65/T1536/topk9 都保持 MoE e2e 精度，并曾要求共用同一个 compiled GEMM1。

本次当前分支实际状态：E65 仍走原有 **t64/w1x4/b3 fallback**，与 E64 的 t256 GEMM1 不同。已有 E65 的 random non-balanced seed0 dynamic、seed17 完整 MoE strict 验证，各 22,020,096 B 与原版零字节差异。不能把这些结果说成“E64/E65 已共用 compiled GEMM1”，也不能说成“已经验证固定 shared expert1 route”。后两项历史要求仍未完成。

此前 F3/pin1 的 E65 原入口固定三轮：GEMM1=257.825 /256.345 /254.551 μs，中值256.345；GEMM2=159.802 /160.971 /160.728 μs，中值160.728；`logits_diff=3.59489e-06`、`rel_l2=0.00268138`，原 gate 通过。最终 pin0/pin1 六轮只重测了 E64，没有重测 E65。

接手必须保持 E65 原路径精度和支持范围，并明确以上差距；不能通过强切 E65 到 t256、添加新 specialization 或修改 route 定义来默默改变当前任务范围。固定 shared expert route 的验证需要先检查测试数据是否真的构造该语义。

## 7. 已经抓好的最优版 thread trace

先读 `my_code/moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md` 中“i32地址表达式最佳候选的matched ATT”及其后采用/最终复测章节。文档前面的多个“当前/reference”指的是历史时刻，判断现状以末尾为准。

F3 matched ATT（本地与远端均已保留）：

```text
.codex_tmp/yadai_gemm1_param_20261021/att/reboot20261024_pin_addr32_i32_att_v1/
```

绑定当前 pin1 GEMM1 ELF `c5296627…f8001d2`；四个 SIMD 为四次独立 captures，64 个 active waves，各 1792 WMMA。已有证据逐 SHA 及 decoded PC/ISA 校验通过，不需要仅因开新窗口而重抓同一版本。

| phase | 累计 wave cycles 占比 | per-wave 中值 cycles |
|---|---:|---:|
| prologue | 13.07864% | 3461 |
| mainloop | 68.19389% | 17182 |
| final input drain | 0.23755% | 63 |
| activation/quant/scale/LDS+barrier | 15.31978% | 4063 |
| output descriptor | 0.18476% | 49 |
| output TDM/completion | 2.98537% | 905.5 |

这些是**累计采样 wave phase**，不是 dispatch 关键路径占比，不能直接换算成可节省的 μs。

用户要求分析 `s_wait_alu`、`s_set_vgpr_msb` 前后的空泡。当前工具按真实 fallthrough、已绑定 ELF 的区间计算：

```text
residual = next_visible_start_timestamp - previous_visible_start_timestamp
           - previous_instruction_decoded_latency
```

前条 latency 使用 SDK event[3]（duration）；保留负残差，排除越过 branch/其他隐藏 opcode 的区间，`both` 类型不重复相加。当前 F3 四处 bridge 如下：

| F3 PC 区间 | 样本数 | start gap 中值 | 前条 latency 中值 | residual 中值 |
|---|---:|---:|---:|---:|
| `0x32a4→0x32b4` | 368 | 16 | 1 | 15 |
| `0x4224→0x4234` | 368 | 16 | 1 | 15 |
| `0x51a0→0x51b0` | 368 | 16 | 1 | 15 |
| `0x6120→0x6130` | 368 | 16 | 1 | 15 |

前条是 `v_dual_add_nc_u32` 地址计算，后条为对应 `ds_load_2addr_b32 ... v34 offset0:128 offset1:160`，必要 wait0 保留。i32 版本相对前一 reference 每处下降1 cycle，主要残差仍存在。正残差不证明全部由中间 control 独占引起；MSB-only residual 中值为0、P95=4，且大量负值，不能根据静态 MSB 数量预测性能或随意删除 hazard waits。

硬件依据先读本地资料，不靠猜测：

- `C:/Users/yanguahe/Documents/code/wk_sp1/mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`，尤其 §3.3.2.3、§4.3.7.4、§5.7.1。
- 同资料树 `MI450/amd-instinct-cdna5-instruction-set-architecture.txt`。
- Shader Guide 说明 `S_SET_VGPR_MSB` 与 `S_SETREG_B32 MODE` 的 delay 行为不同；不要将控制指令区间残差当作每条 MSB 固定硬件延迟，也不要把 VGPR 编号变动直接说成解决 SRAM bank conflict。

若新候选有稳定收益且需要 matched ATT，用 `my_code/get_att_b_rocm_sdk.sh`，完整 symbol 精确匹配，`--all-simd --iteration 28 --ana-att`，每个 profiler/decoder child 120秒。profiling 期间的耗时不作为正常性能数据。

不要直接运行旧 `run_best_frame_fixed_att_v1.sh`：它硬编码历史 source SHA、pin0 和旧 symbol。先读 runner 的 gate 和实际 cache 环境，再准备与当前候选匹配的命令；source 改动后所有 PC/ELF/真实 keys 必须重新绑定。

## 8. 已放弃的实验与可选择的未执行候选

不要重新做同一批参数遍历来“追进度”。详细原表及完整 feature 名在参数调优文档，下面给出决策索引。

| 方向 | 已知结果 / 决策 |
|---|---|
| 原版 pin 编号叠加 DS5 优化 | 正反向18轮，四项相邻改善 -0.168066% /+0.229015% /+0.144225% /-0.677285%；不支持稳定收益，不采用。与 F3/pin1 调整后的策略不同。 |
| 恢复原 C/A/B 编号，保留 addr32/i32 | F3前/候选/F3后中值80.1656 /80.3678 /80.7115；不稳定，不采用。 |
| LDS 地址槽改 `v16:23` | 80.5365 /80.8589 /80.7748；两侧回退，不采用。 |
| 四级 ring constexpr 展开 | 80.5512 /140.494 /79.9228；明显回退，不采用，不追加ATT。 |
| 四级 stage LDS 地址预计算 v3 | 80.3974 /82.5563 /80.9201；两侧回退，不采用。v1/v2还分别有 runtime索引Python列表、误用`fx.select`的问题。 |
| scalar 地址约束合成 tied vector | 反向一组回退，未确认稳定收益；少10条静态MSB不是采用依据。 |
| 跳过空 M64 计算 | 实现精度通过但性能回退，trace显示交错和同步配合受损；这不否定跳过无效计算本身，只否定已测实现。 |
| t128 系列 | 多组资源/几何实验未接近保留版，不采用；不能只凭较低VGPR推断实际occupancy收益。 |
| 三处 wait `(0)→(1)` 的私有 ISA 实验 | 旧 reference 上局部 residual 16→15，反向正常改善约0.23%～0.40%；缺完整通用依赖证明，未落地生产源码。不要将旧PC直接用于F3。 |
| reuseB（最后收尾实验） | 正向F3/reuseB/F3中值81.3524 /80.6082 /80.8020；反向reuseB/F3/reuseB为80.3846 /80.4373 /80.5848。反向最后组更慢，不稳定，不采用。18轮全部有效，完整MoE与F3逐字节一致。 |

以下是**工具/CPU 准备状态**，不是已经采用的优化或已经证明的性能收益：

1. `dsf5_pin1_pair_unroll_i32_scoped_v1`：F3 上展开两次既有 iteration，源+12/-2；CPU 为1024 VGPR/58 SGPR/327680 B，static executable/wait/MSB=7435/168/1110。未做这份 F3 ELF 的设备/性能验证。此前旧 reference 上类似 `steady_pair_scoped_v1` 做过18轮、小幅改善但未采用；两者基准/ELF不同，不能混用结果。
2. `dsf5_pin1_quant_values128_i32_scoped_v1`：激活后的量化值以 tied constraints pin 到 `v128+index`；CPU 为1024/62/327680 B，static=6058/165/981。仅CPU/静态检查，未做设备/性能验证。
3. `prepare_adopted_scale_const100_isa_v1.py`、`run_adopted_scale_const100_isa_v1.py`：只在本地创建，未上传、未执行。相关 reserved field 编码观察尚未验证；先核对文档/LLVM/backend 合法性，不能称为已证明的bug或瓶颈。

可以在读完现有证据后优先选择一个未执行的独立小候选，或提出新方案；不要并行竞争 GPU，也不要把 CPU 编译成功当成精度/性能通过。ISA 注入仅用于私有 cache 的实验，最终采用仍必须修改并验证 FlyDSL 源码。

用户提供的 ISA 实验工具来源：
<https://github.com/ROCm/FlyDSL/blob/94485ffdc3e57fa79cc9e0979369a8128becbf17/tools/inject_asm/inject_asm.py>。
本地已适配 dev882 的 `tools/inject_asm_dev882_v1.py`、`v2.py` 可供阅读；不要假定上游原脚本与当前 wheel 的 cache/schema 完全兼容。

## 9. 证据、工具与阅读顺序

下列路径相对本地/远端仓库根目录，实验根统一为：

```text
.codex_tmp/yadai_gemm1_param_20261021/
```

先读：

1. 本 handoff。
2. `my_code/moe_gemm1_hyg_dsr1_parameter_tuning.md` 的末尾收尾/最终复测章节，然后按 feature 搜索历史章节，不必一次读完近万行。
3. `my_code/moe_gemm1_hyg_dsr1_best_thread_trace_analysis.md` 的 F3 matched ATT 和末尾采用/最终状态。
4. `logs/final_commit_pin_perf_20261024_v1.local_review_v1.json`、`.summary.json` 及目标 ATT 目录中的原始报告。

这些材料足以恢复当前任务，不必先加载巨大的历史 rollout JSONL。

| 已下载并校验的证据 | archive SHA256 |
|---|---|
| 最终六轮 `logs/final_commit_pin_perf_20261024_evidence_v1.pin_evidence.tgz`，118项 | `a4a3bdd37e37df69b63759f3b804fd41d9d44a10e8662d00c4f36eb4839fb41c` |
| F3 matched ATT archive，138项；见目标ATT目录/文档索引 | `2d56ed44392389d6b1f21e6ec69bff21b62312a3e034dd7546d2ffe1e2b5f9c7` |
| 三候选27轮 `logs/reboot20261024_adopted_pin_followups_perf_evidence_v1.pin_evidence.tgz`，496项 | `2715440357c331000de77f2bae64df25a52af1df38b3592498c3dc2675a3c5f3` |
| pair/quant/reuseB CPU `logs/reboot20261024_adopted_pair_quant_reuse_cpu_evidence_v1.pin_evidence.tgz`，40项 | `993c92ddcb08dc006cbc5a37da5be319a5ed010f92c90c2f18ce14cba0264d1a` |
| reuseB最后18轮 `logs/reboot20261024_b_reuse_pair_i32_adopted_final_evidence_v1.pin_evidence.tgz`，294项 | `0d9ad88c07950aa9b8567954396814b273fda5d3416b1e4be53d43a49f2fe5b1` |

全部归档不含`.out`。用户明确说`.out`不要下载；延续这个要求。已有 ISA、ELF、MLIR、decoded JSON及CSV足以做分析。

主要工具（均在实验根的 `tools/`）：

| 工具 | 用途及注意事项 |
|---|---|
| `wait_normal_idle_v1.sh` | 主机全机idle等待，每6秒轮询 |
| `prepare_normal_stage_keys_v2.py` | 在原caller记录真实source-dependent keys，分阶段正常编译；stage2准备会实际运行stage1，属于不计性能的准备 |
| `screen_adopted_pin_flydsl_v1.py` | F3门槛的既有双侧三轮编排；读取CLI后用新label，检查实际source/ELF门槛 |
| `compare_pin_moe_bytes_v1.py` | 完整MoE逐字节比较；先读参数，不能省去原precision gate |
| `audit_frame_fixed_normal_cache_v2.py` | 原source/global/argument keys、cache namespace、两阶段完整ELF绑定 |
| `analyze_control_gap_minus_latency_v1.py` | 用户指定的gap减前条latency残差分析 |
| `compile_adopted_pin_vgpr_batch_v1.py` | F3小候选CPU准备，无性能结论 |
| `final_commit_pin_perf_20261024_v1.py` | 已完成的六轮最终复测编排，固定label/cache不能原样重复运行 |
| `review_final_commit_pin_perf_local_v1.py` | 从最终原表重算中值及校验实际对象 |
| `pack_adopted_pin_evidence_v1.py` / `verify_pin_evidence_local_v1.py` | 封存/下载后逐member校SHA，不含`.out` |

旧工具中可能仍固定历史 source/ELF/boot ID；不要把“改掉assert或拼改preparation元数据”当成完成验证。任何 source/global 变化都需要从当前原caller重新生成真实 keys。不要仅靠找到某个 ELF 的缓存文件名去猜当前测试实际运行了哪个对象。

如果需要借鉴更早的独立 feature，可阅读工作区中的 `aiter_dsr1_e65/my_code/moe_gemm1_e65_t1536_nonbalanced_optimization.md` 和 `aiter-moe-pipeline-130us/my_code/moe_pipeline130_final_summary.md`；它们是不同分支的历史材料，不直接替换当前代码、计时口径或当前约束。

## 10. 接手后的执行顺序

1. 读取上述关键材料，确认当前F3源码和实际环境；保留文档、cache和失败记录。
2. GPU空闲后以同一原入口复测F3/pin1；若环境变化或结果明显漂移，再补pin0对照，建立本轮基线。
3. 选择一个具体独立小feature，冻结候选源、明确guard、用隔离正常cache，先编译和数学/依赖检查，再做原MoE精度与必要的E65/guard外验证。
4. 做固定三轮的双侧相邻对照；小幅收益补反向，不因单组或单轮更快就采用。记录全部有效/无效/失败轮次及原因。
5. 对需要解释的有效候选抓matched ATT，绑定真实ELF后分析；明显回退候选结束，不无谓追加trace。
6. 只有完整验证并稳定更好的FlyDSL源码才替换F3；维护新baseline的source/ELF/keys/证据索引，继续记录到调优文档。

本交接文件本身只记录事实与后续任务，没有修改生产源码，也没有新增提交。新的优化工作在用户新开的窗口进行。
