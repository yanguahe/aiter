# git pull 后 E64/E65 MoE 最终性能复测（2026-10-24）

用户延期的英文提交、push、远端 git pull及最终性能复测任务已完成。
在 a07-3指定目录和 `dsr1_ep4`容器中，按原完整 MoE `--scenario bench`入口测得：

| workload | 版本 | GEMM1 μs | MoE e2e μs |
|---|---|---:|---:|
| E65/topk9/shared1 | baseline | 154.8180 | 300.886 |
| E65/topk9/shared1 | 最优版 | 107.5470 | 256.846 |
| E64/topk8/shared0 | baseline | 99.3689 | 214.990 |
| E64/topk8/shared0 | 最优版 | 94.2981 | 205.589 |

以上均为固定三轮中值，取代本任务汇报中的历史参考读数；完整原始轮次保留如下。

| workload | GEMM1 耗时降幅 | MoE e2e 耗时降幅 |
|---|---:|---:|
| E65/topk9/shared1 | 30.53%（减少47.2710 μs） | 14.64%（减少44.040 μs） |
| E64/topk8/shared0 | 5.10%（减少5.0708 μs） | 4.37%（减少9.401 μs） |

计算方式为 `(baseline中值 - 最优版中值) / baseline中值 × 100%`。

## 代码、环境与测量口径

本地工作区为 `C:/Users/yanguahe/Documents/code/wk_sp1/aiter_dsr1_e65`。
远端为 `yanguahe@heliosr-1b114-a07-3.mnb.dcgpu`，仓库
`/data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007`，测试容器 `dsr1_ep4`。
Git操作均在主机，测试均在该容器。

个人 `origin`分支为 `codex/dsr1_e65_gemm1_20261017`。此前已经完成的英文 commit：

```text
Optimize gfx1250 MoE GEMM1 for non-balanced E64 and E65 routing
```

最优版 commit为 [d242f28c743701b75a01c6b7b11dca6e88582f89](https://github.com/yanguahe/aiter/commit/d242f28c743701b75a01c6b7b11dca6e88582f89)，
baseline为 `339589f900b3a880ec447f605886fe803cbf9649`。
本轮重新 fetch个人 origin，并在指定远端分支执行
`git pull --ff-only origin codex/dsr1_e65_gemm1_20261017`，结果为已是最新。
26个 pulled source的 SHA全部核验；下载的源码快照又与本地对应 commit的 Git blob逐字节比较。

两版本使用同一份 pulled测试入口、输入生成和原计时。baseline加载339589中以下三个原文件：

- `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm_prefill_persistent.py`
- `aiter/ops/flydsl/grouped_gemm_mxfp4.py`
- `aiter/ops/flydsl/grouped_moe_gfx1250.py`

最优版直接使用 d242f28c74的生产 dispatch。baseline快照、控制脚本、原表和实际完整 ELF
均包含在证据归档中；没有修改原精度 gate或计时以获取性能数字。

workload均为 T1536/model_dim7168/inter_dim2048、A4W4、SiLU、无 bias、random non-balanced
routing。完整 MoE benchmark沿用默认 seed0、warmup5、`--iters20`；GEMM1原 profiler
每轮计数为19，原表耗时与 `device_time_avg`在原日志显示精度内一致。

PyTorch为 `2.11.0+rocm10.1.0a20260811-bkc.20260820`，HIP为 `7.16.26315`，
architecture为 `gfx1250`；每个 driver均核验 `aiter.__file__`来自指定仓库。
boot ID为 `cb28eed4-51ca-4cdc-bf5b-ecbc1d54db5f`，12轮前后保持一致。

每轮性能 child之前，主机每6秒检查全机 GPU/KFD所有权、physical GPU的利用率与 VRAM，
等待三个连续 idle状态后运行；计时期间没有并发传输、编译或独立 GPU watcher。
每个 child使用 `timeout --signal=TERM --kill-after=5s 120s`，测试后检查 live ownership，
并在6秒后确认 settled idle。12轮全部合格，四组均为 attempt1，没有污染组或按读数重跑。

## 固定原始轮次

预先固定顺序为 baseline E65 → 最优版 E65 → baseline E64 → 最优版 E64，各三轮。

| workload | 版本 | GEMM1 三轮 μs | MoE e2e 三轮 μs |
|---|---|---|---|
| E65/topk9/shared1 | baseline | 154.818 / 156.589 / 152.004 | 300.886 / 301.136 / 296.903 |
| E65/topk9/shared1 | 最优版 | 107.678 / 107.547 / 107.185 | 255.092 / 256.846 / 256.882 |
| E64/topk8/shared0 | baseline | 99.7062 / 97.9857 / 99.3689 | 216.234 / 213.086 / 214.990 |
| E64/topk8/shared0 | 最优版 | 94.2981 / 93.9861 / 94.9328 | 204.446 / 205.589 / 206.391 |

## 精度及 E64/E65 共用 kernel

baseline和最优版在 seed0、17的四个已完成正确性 child，本轮按文件 SHA及完整原日志重新
验收并复用。每个 child覆盖 E65/shared1和 E64/shared0。原 MoE精度 gate通过；最优版的
payload、ScaleA、GEMM2及每份22,020,096 B完整 BF16 MoE输出与对应 baseline零字节差异。
eager、graph replay及更新同一组指针下输入/route的 dynamic replay检查也通过。
这些正确性运行的耗时没有参与性能统计。

最优版实际 GEMM1 symbol为：

```text
a8w4_tdm_fp4_t256x256x256_w2x4_b4_K7168_eruntime_act1_q1r8_cn4
```

E64、E65全部性能轮次和两个正确性 seed使用的完整 ELF SHA256均为：

```text
ad6287c8a65de03476ba4e463e3a5782db5fe7eea5238e16ec15b9f805ae90cc
```

同一正确性进程中，E65与 E64还使用同一 compiled object且 launcher cache只有一个 entry。
这是 d242f28c74的共用证明。当前 F3分支的 E65仍有原 fallback，其范围另见
[F3后续优化记录](moe_gemm1_hyg_dsr1_f3_followup_20261024.md)。

## 复现命令

先在主机进入指定目录并切到已拉取的 `codex/dsr1_e65_gemm1_20261017 / d242f28c74`，
核对工作区，再完成全机 idle检查。下列最优版命令在 `dsr1_ep4`容器内运行：

```bash
cd /data/yanguahe/code/wk_sp1/aiter_t256w2x4_e64_20261007
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=false \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
FLYDSL_DUMP_IR=0 \
timeout --signal=TERM --kill-after=5s 120s \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache \
  --experts 65 --tokens 1536 --topk 9 --shared-experts 1 \
  --model-dim 7168 --inter-dim 2048 --iters 20
```

E64改为 `--experts 64 --topk 8 --shared-experts 0`，其余相同。
baseline在同样环境下通过保存的 driver加载对应 Git快照，例如 E65：

```bash
ENABLE_CK=0 AITER_MOE_EXPERT_BALANCE=false AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 FLYDSL_DUMP_IR=0 \
timeout --signal=TERM --kill-after=5s 120s python3 -u \
  .codex_tmp/git_pull_d242f28c74_perf_20261024/final_perf_driver.py baseline perf \
  --experts 65 --seed 0 \
  --output "$PWD/.codex_tmp/git_pull_d242f28c74_perf_20261024/reproduce_baseline_e65_new_01/driver"
```

输出目录须使用未用过的值。主机编排工具为
`.codex_tmp/git_pull_d242f28c74_perf_20261024/run_deferred_perf_only_20261024_v1.py`；
它已使用的 label不能覆盖，复现时为 plan、run及输出目录使用新 label。

## 证据及环境恢复

证据根为 `.codex_tmp/git_pull_d242f28c74_perf_20261024/`。归档
`deferred_git_pull_perf_20261024_v1.evidence.tar.gz`含143个证据文件及一份 manifest，
681,927 B，SHA256为：

```text
e3b882299d541f07de7a8ef6c6a0f44ade16cec0f0fdf1d96c4918d531199d45
```

归档已下载，逐 member核对大小和 SHA，从原表重算全部中值及降幅，核对实际 ELF、源快照、
原控制脚本、三次连续 idle和 settled记录。归档不含 `.out`、原始 BF16 payload或 caches。
本地验收为 `deferred_git_pull_perf_20261024_v1.local_final_review.json`，`passed=true`。

首版本地验收曾把非原子的瞬时 KFD目录记录误判为活进程。原记录显示 fuser无 owner、
运行时 `/proc`检查的 `live_post_owners=[]`，6秒后 KFD目录为空、全机 settled idle。
已按原始采集顺序修正核验，保留失败脚本及 `local_review_attempt1_failure.json`，
没有修改采集证据或重跑性能。后续复核同时要求 live ownership和完整 settled检查。

完成后，本地及指定远端目录均为 `hyg_dsr1_conc6144 / 6f656cf5b0`，F3源 SHA恢复为
`f3c77d2aaa4698c80eb82b5d377c4f7b529247fc24c5fffce401d9a6f1fb21ec`。
临时移出的 `my_code/dsr1_moe_ep_decoder_bug_fix.md`按原 SHA
`ec102f030cf574542cc974d46d77736c27a6e5633e5495a9c77ada6222a55a20`恢复，备份保留。
远端另一 worktree已使用相同 F3分支；通过 `git switch --ignore-other-worktrees`恢复本目录，
并确认该另一 worktree的 HEAD和 tracked工作区均未变化。恢复记录为
`deferred_branch_restoration_20261024_v1.json`。

F3的原 `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel`测量结果
另见 [F3后续优化记录](moe_gemm1_hyg_dsr1_f3_followup_20261024.md)，两份成绩分别解读。
