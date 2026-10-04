# E64/T1536/topk8 GEMM1 无效 block thread trace 分析（a07-3）

## 分析范围

本文分析 grouped MoE GEMM1 中完成 expert 二分查找后，因为
`expert >= n_experts` 而直接退出的无效 block，并计算这些 block 在同一个
SIMD slot 累计 cycles 中的占比。测试规模为：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

测试环境：

```text
host:      heliosr-1b114-a07-3
container: hyg_fyd1
repo:      /data/yanguahe/code/wk_sp1/aiter
branch:    hyg/moe_a4w4_pr
HEAD:      778eef31c3f1d97d55df84e46e127d8bcd63ec29
date:      2026-10-03 UTC
```

机器刚完成重启，因此抓 trace 前重新测量了本阶段 baseline 和当前保留的最优
kernel。每次性能测试前都确认 GPU/KFD 无其他使用者，GPU utilization 和 VRAM
activity 均为空闲。

## 重启后的性能

本阶段 baseline 是 `HEAD` 中尚未接入 GEMM1 persistent `w2x4` 的实现，其
GEMM1 symbol 为：

```text
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_bth6_rcw_mg4_fc20_xdl0_reuse_ostore2p_s3
```

当前保留的最优实现为：

```text
persistent tasks = 4
tile             = 192x256x256
workgroup        = w2x4（8 waves）
buffers          = 4
cluster          = 4x1
B TDM hint       = 6（NT_HT）
WMMA reuse       = reuseB / reuse3
```

其 GEMM1 symbol 为：

```text
a8w4_tdm_fp4_t192x256x256_w2x4_b4_K7168_e64_act1_cn4_cm1_prefetch_eb4_apre_sh_bth6_rcw_mg4_fc8_xdl0_reuse3_ostore2p_s3_ps4pf2hm_earlynext_o1w_xor_wait1
```

| 版本 | GEMM1 samples (us) | GEMM1 median | GEMM2 median | fused MoE median | 正确性 |
|---|---|---:|---:|---:|---|
| 本阶段 baseline | 78.128, 77.881, 79.720 | 78.128 us | 53.418 us | 208.05 us | const0 hashes 一致 |
| 当前最优 | 75.131, 74.825, 74.104 | 74.825 us | 51.469 us | 199.53 us | const0 hashes 一致 |

当前最优 GEMM1 相对同机 baseline 降低 `4.23%`。原始日志目录：

```text
baseline: my_code/moe_prefill_switch_ab_runs/20261003T054059Z
best:     my_code/moe_prefill_switch_ab_runs/20261003T054229Z
```

此前的 `s_setprio_inc_wg` 实验已判定有问题。本次重启后没有再运行该实验，相关
kernel 修改也已经从本地工作区和 a07-3 恢复掉。

## Thread trace 抓取

为减少 trace 次数，本次只抓 ATT SIMD selector 3。decoder 仍输出了四个 shader
engine 的 occupancy 数据，并解码了各 shader engine 上 SIMD3 的两个 slot。

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

产物位置：

```text
my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_invalid_blocks_a07_3_20261003/
  logs/analyze_att_capture.log
  logs/invalid_block_wave_trace_analysis.log
  logs/invalid_block_occupancy_analysis.log
  thread_trace/kernel/rpf_v3/ui_output_agent_15658_dispatch_1561/
```

`analyze_att_capture.py` 给出的 REALTIME 加权平均 GFXCLK 为 `1893.745 MHz`，
最大 occupancy timestamp span 为 `132579 cycles`。

## 有效与无效 block 的判定

decoder 文件名采用 `seX_smY_slZ_wvN.json`。固定 `SE/SM/SL` 后，`wvN`
表示依次占用该 SIMD slot 的 block-wave 实例。

分类直接检查解码后的动态指令：

- 有效 block：至少执行一条 `v_wmma*`；
- 无效 block：没有执行任何 `v_wmma*`。

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

该路径没有进入 GEMM hotloop。由此可以确认短 trace 对应二分查找得到无效 expert 后
直接退出的 block，而不是执行异常快的有效 block。

## cycle 总量最大的 decoded SIMD slot

对每个 `SE/SM/SL` 内所有 `wvN` 的 duration 求和，cycle 总量最大的是：

```text
SE1/SIMD3/slot1
```

| block instance | cycles | 动态指令数 | `v_wmma*` 数量 | 分类 |
|---|---:|---:|---:|---|
| `wv0` | 126,149 | 15,286 | 2,688 | 有效计算 |
| `wv1` | 1,241 | 238 | 0 | 无效 early-exit |
| `wv2` | 1,011 | 238 | 0 | 无效 early-exit |
| `wv3` | 1,015 | 238 | 0 | 无效 early-exit |

因此：

```text
slot 总 cycles   = 126149 + 1241 + 1011 + 1015
                 = 129416 cycles

无效 block cycles = 1241 + 1011 + 1015
                  = 3267 cycles

无效 block 占比   = 3267 / 129416
                  = 2.524417%
```

按本次 trace 的 `1893.745 MHz` 估算，`3267 cycles` 相当于约 `1.725 us` 的
累计 slot residency。这只是该 slot 上的理想局部上限；不同 WGP 的无效 block 会与
其他工作并行，因此不能直接推导为整个 kernel 可以降低 `1.725 us`。

全部 decoded SIMD slot 如下：

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

## physical-WGP 口径交叉检查

`occupancy.json` 还提供 `packed_sa_wgp`，因此可以使用更严格的物理坐标：

```text
SE / SA / WGP / SIMD / slot
```

按该坐标统计，累计 resident cycles 最大的是
`SE2/SA1/WGP2/SIMD0/slot0`。它只运行了一个有效 block，共 `132537 cycles`，
所以严格物理 slot 口径下的无效 block 占比为 `0%`。

在确实被多个 block 复用的物理 slot 中，cycle 总量最大的是
`SE1/SA1/WGP2/SIMD0/slot1`：一个有效 block 加一个无效 block 共消耗
`129874 cycles`，其中无效 block 为 `1038 cycles`，占 `0.799236%`。

因此，按用户所指的 decoder `SE/SM/SL` slot 时间线，主要结论是
`2.524417%`。physical-WGP 交叉检查也说明，这个比例不能直接视为 kernel wall-time
收益：全局最慢的严格物理 slot 本次没有执行无效 block，无效工作主要落在 dispatch
尾部被再次复用的 slot 上。

## 复现分析

decoded wave JSON 的统计命令：

```bash
python3 my_code/analyze_gemm_invalid_block_trace.py \
  my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_invalid_blocks_a07_3_20261003/thread_trace/kernel/rpf_v3/ui_output_agent_15658_dispatch_1561
```

physical-WGP occupancy 的统计命令：

```bash
python3 my_code/analyze_gemm_invalid_blocks.py \
  my_code/thread_trace_runs/e64_t1536_topk8_gemm1_best_invalid_blocks_a07_3_20261003/thread_trace/kernel/rpf_v3/ui_output_agent_15658_dispatch_1561/occupancy.json
```
