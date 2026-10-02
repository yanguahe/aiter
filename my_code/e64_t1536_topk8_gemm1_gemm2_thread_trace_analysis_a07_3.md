# gfx1250 E64/T1536/topk8 GEMM1/GEMM2 thread trace 分析

## 分析范围

本文分析以下 MoE 规模实际使用的 GEMM1 和 GEMM2 kernel：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

采集环境：

```text
host:      a07-3
container: hyg_fyd1
repo:      /data/yanguahe/code/wk_sp1/aiter
commit:    04cc526b8f06f1e54f836718964e8f2e2c844fa8
data:      const0
ATT SEs:   SE0-SE3
ATT SIMDs: SIMD0-SIMD3，分别独立采集
target CU: 1
```

preflight 和两次正式采集前，主机上的 `/data/yanguahe/code/gpu_users.sh` 均确认没有预先存在的 GPU/KFD 进程；采集结束后也没有残留 GPU 进程。

preflight 的 const0 正确性完全通过：

```text
logits_diff = 0
rel_l2      = 0
pass        = True

GEMM1 = 79.518 us
GEMM2 = 60.270 us
MoE   = 214.329 us
```

以上普通 profiler 时间用于确认当前性能状态。启用 ATT 后 profiler 输出的亚微秒时间是 instrumentation 产物，不能作为性能数据。

## 实际抓取的 kernel

GEMM1：

```text
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_bth6_rcw_mg4_fc20_xdl0_reuse_ostore2p_s3
```

GEMM2：

```text
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc20_ostore2p_s3_ow2
```

a07-3 上的 trace 目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/thread_trace_runs/e64_t1536_topk8_gemm1_bth6_att_20260930
/data/yanguahe/code/wk_sp1/aiter/my_code/thread_trace_runs/e64_t1536_topk8_gemm2_att_20260930
```

归档文件：

```text
my_code/thread_trace_runs/e64_t1536_topk8_gemm1_bth6_att_20260930.tar.gz  104 MB
my_code/thread_trace_runs/e64_t1536_topk8_gemm2_att_20260930.tar.gz        127 MB
```

GEMM2 的第一次 SIMD0 解码出现：

```text
ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA
```

该次解码得到的 `code.json/occupancy.json` 为空，不能参与分析。随后只重新抓取 SIMD0 并成功完成解码。失败数据保存在 `thread_trace/simd0_decode_error`，有效重试结果位于 `thread_trace/simd0`，本文只使用后者。

## ATT 抓取方法

抓取过程沿用 `reproduce_compare.sh` 中的 ATT 方法：

- 先通过正常运行获取精确 kernel symbol。
- 每个 SIMD 生成一份独立的 `rocprofv3` YAML。
- 设置 `kernel_iteration_range: "[1]"`。
- 设置 `att_target_cu: 1`。
- 设置 `att_shader_engine_mask: "0xf"`，覆盖 SE0-SE3。
- 分别抓取 SIMD0、SIMD1、SIMD2、SIMD3。
- 使用 `/data/yanguahe/code/wk_sp1/decoder_new` 解码。
- 检查 `.att`、`code.json`、wave JSON、`realtime.json` 和 `occupancy.json`。

传给采集脚本的测试命令为：

```bash
env \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_LOG_MORE=1 \
  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
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
    --iters 2 \
    --const-init 0
```

GEMM1 完整采集命令：

```bash
REPO_ROOT=/data/yanguahe/code/wk_sp1/aiter \
TRACE_ROOT=my_code/thread_trace_runs \
bash /tmp/get_isa_runner_att.sh \
  a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_bth6_rcw_mg4_fc20_xdl0_reuse_ostore2p_s3 \
  e64_t1536_topk8_gemm1_bth6_att_20260930 \
  "env AITER_MOE_EXPERT_BALANCE=true AITER_LOG_MORE=1 AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --iters 2 --const-init 0" \
  --all-simd \
  --ana-att
```

GEMM2 使用相同命令，只替换 symbol 和输出目录：

```bash
REPO_ROOT=/data/yanguahe/code/wk_sp1/aiter \
TRACE_ROOT=my_code/thread_trace_runs \
bash /tmp/get_isa_runner_att.sh \
  a8w4_tdm_fp4_t192x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc20_ostore2p_s3_ow2 \
  e64_t1536_topk8_gemm2_att_20260930 \
  "env AITER_MOE_EXPERT_BALANCE=true AITER_LOG_MORE=1 AITER_USE_GROUPED_GEMM=1 AITER_GROUPED_DEBUG=0 AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --iters 2 --const-init 0" \
  --all-simd \
  --ana-att
```

采集脚本、`trace_segment_cycles.py`、代表 wave 配置和聚合脚本均保存在两个 trace 目录各自的 `analysis/` 子目录中。

## 分析方法和统计口径

按照 `flydsl-align-reference-kernel.mdc` 的说明，使用 `trace_segment_cycles.py` 的 representative trace 模式：

```bash
python3 analysis/trace_segment_cycles.py \
  analysis/full_representative.json \
  --specific-part-representative-trace
```

每个独立 SIMD capture 中，先排除没有执行任何 `v_wmma*` 的 empty/sentinel wave，然后选择 entry 到 `s_endpgm` 周期最接近该 capture active-wave 中位数的 wave。这样可以避免用早退路径分析 hot path。

ATT wave JSON 中每条指令记录的格式是：

```text
[timestamp, type, stall, latency, code_idx]
```

本文使用两种占比：

1. `latency 占比`：某条指令或某类指令的累计 decoder latency，除以所有 active wave 的累计指令 latency。
2. `stall/wave-span`：累计 exposed stall，除以所有 active wave 从 entry 到 `s_endpgm` 的 timestamp span。

由于 decoder stitching 和指令重叠，累计指令 latency 与 timestamp span 不完全相等：

```text
GEMM1 instruction-latency / wave-span = 1.0695
GEMM2 instruction-latency / wave-span = 0.9599
```

因此下面的结果用于定位瓶颈，不能把所有百分比直接加到 dispatch-level cycle 上。

本文涉及的 ISA 语义：

- `s_wait_tensorcnt N`：等待 TDM `TENSORcnt <= N`。
- `s_wait_dscnt N`：等待 LDS/DS `DSCNT <= N`。
- `s_wait_kmcnt 0`：等待 outstanding scalar-memory/message 结果返回。
- `s_barrier_wait 0xffff`：等待同一 workgroup 的所有 wave 完成 barrier signal。

对应定义见 CDNA5 ISA 的 5.6、8.2、9.3 和 15.5 节。

## Active wave 周期分布

| Kernel | Active waves | Minimum | Median | P90 | Maximum | Mean |
|---|---:|---:|---:|---:|---:|---:|
| GEMM1 | 64 | 28,173 cycles | 32,248 cycles | 35,693 cycles | 36,298 cycles | 32,354 cycles |
| GEMM2 | 112 | 10,633 cycles | 11,726 cycles | 20,568 cycles | 22,906 cycles | 13,305 cycles |

GEMM2 存在明显的首个 wave 实例效应：

| Kernel | Wave group | Count | Mean | Median | Minimum | Maximum |
|---|---|---:|---:|---:|---:|---:|
| GEMM1 | `wv0` | 16 | 34,011 | 33,990 | 31,175 | 36,298 |
| GEMM1 | 后续 wave | 48 | 31,802 | 31,636 | 28,173 | 36,265 |
| GEMM2 | `wv0` | 16 | 21,473 | 21,777 | 19,728 | 22,906 |
| GEMM2 | 后续 wave | 96 | 11,944 | 11,661 | 10,633 | 16,573 |

GEMM2 在物理 slot 上观察到的第一个 wave 实例比后续 wave 慢约 80%。额外周期主要来自冷启动阶段的 scalar-memory 等待以及较长的 output LDS drain。GEMM1 的首个 wave 实例开销约为 7%，明显较小。

## Prologue、compute 和 epilogue 占比

对每条 active wave 动态划分：

- prologue：kernel entry 到第一条 `v_wmma*` 之前；
- compute：第一条到最后一条 `v_wmma*`，包括与计算交错的 TDM/LDS wait 和 barrier；
- epilogue：最后一条 `v_wmma*` 之后到 `s_endpgm`。

| Kernel | Prologue latency | Compute latency | Epilogue latency |
|---|---:|---:|---:|
| GEMM1 | 10.78% | 81.22% | 8.00% |
| GEMM2 | 26.45% | 53.67% | 19.89% |

这是两个 kernel 最显著的结构差异。GEMM1 有 28 个 K tile，可以用较长的 compute hotloop 摊薄 setup/drain；GEMM2 只有 8 个 K tile，prologue 和 epilogue 合计占其 decoder latency 的 46.33%。

## 指令类别占比

| Category | GEMM1 latency | GEMM1 stall/span | GEMM2 latency | GEMM2 stall/span |
|---|---:|---:|---:|---:|
| `v_wmma*` | 52.79% | 19.08% | 40.60% | 12.99% |
| `s_wait_*` | 18.06% | 18.55% | 31.27% | 29.40% |
| `s_barrier_wait` | 10.35% | 10.97% | 10.20% | 9.70% |
| other VALU | 8.66% | 2.56% | 6.74% | 1.33% |
| LDS instructions | 6.19% | 0.82% | 5.56% | 0.23% |
| other SALU | 3.70% | 0.27% | 5.40% | 0.57% |
| tensor issue | 0.16% | 0% | 0.14% | 0% |

显式 wait 合计：

```text
GEMM1 s_wait_* + s_barrier_wait:
  28.40% of decoded instruction latency
  29.52% exposed stall / active-wave span

GEMM2 s_wait_* + s_barrier_wait:
  41.46% of decoded instruction latency
  39.10% exposed stall / active-wave span
```

排除冷启动 `wv0` 后，显式 wait 的 latency 占比仍为：

```text
GEMM1: 28.11%
GEMM2: 36.18%
```

因此，即使排除首个 wave 实例，GEMM2 仍明显比 GEMM1 更偏 wait-bound。

## GEMM1 长延迟指令

所有 64 条 active wave 聚合后的主要 wait：

| Phase | Instruction | Hits | Avg latency | Max | Latency share | Stall/span |
|---|---|---:|---:|---:|---:|---:|
| compute | `s_wait_tensorcnt 0x4` | 1,600 | 114.08 | 2,211 | 8.24% | 8.74% |
| compute | `s_barrier_wait 0xffff` | 1,728 | 102.83 | 2,617 | 8.02% | 8.50% |
| prologue | `s_wait_kmcnt 0x0` | 864 | 91.87 | 2,979 | 3.58% | 3.79% |
| prologue | `s_barrier_wait 0xffff` | 64 | 721.47 | 2,482 | 2.09% | 2.23% |
| prologue | `s_wait_tensorcnt 0x6` | 64 | 640.62 | 2,805 | 1.85% | 1.98% |
| compute | `s_wait_dscnt 0x0` | 1,792 | 15.52 | 1,483 | 1.26% | 1.26% |
| epilogue | `s_wait_tensorcnt 0x0` | 128 | 151.13 | 638 | 0.87% | 0.93% |
| epilogue | `s_wait_dscnt 0x0` | 128 | 72.95 | 79 | 0.42% | 0.44% |

长事件累计占 active-wave span：

```text
latency > 100 cycles:  24.18%
latency > 300 cycles:  19.83%
latency > 600 cycles:  14.61%
latency > 1000 cycles:  9.23%
```

`trace_segment_cycles.py` 选择的四条代表 wave 也表现出相同模式：

- 初始 `s_wait_tensorcnt 0x6`：941-1,534 cycles；
- steady-loop `s_wait_tensorcnt 0x4`：代表 wave 中最高 1,502 cycles；
- steady-loop `s_barrier_wait 0xffff`：代表 wave 中最高 1,515 cycles；
- 最终 `s_wait_tensorcnt 0x0`：代表 wave 中约 591-600 cycles。

所有 active wave 中最长的非 wait 事件是一条孤立的 `ds_load_b128`，延迟 1,358 cycles。但 LDS 指令整体只占 active-wave span 的 0.82% exposed stall，因此这些尖峰不是反复出现的主瓶颈。

### GEMM1 瓶颈判断

GEMM1 是 compute 和 input-pipeline synchronization 混合受限：

- 有效 `v_wmma*` 工作是最大类别，占 52.79% decoded latency。
- compute 阶段反复出现的 `s_wait_tensorcnt 0x4` 和 workgroup `s_barrier_wait` 合计占 16.27% decoded latency。
- 所有显式 wait/barrier 合计形成约 29.5% exposed stall。
- output drain wait 合计低于 1.4%，不是 GEMM1 的主要问题。

因此 GEMM1 当前最主要的非计算瓶颈是 K-loop 中的 TDM 到达延迟以及不同 wave 到达 workgroup barrier 的时间差。`B_TH=6` 已经降低 B 侧 cache 压力，但不能消除 A/Scale 到达延迟、TDM owner wave 发射偏斜或其他 wave 晚到 barrier 的问题。

GEMM1 后续应优先在 `s_wait_tensorcnt 0x4` 前增加可独立执行的 WMMA/VALU 工作，并减少进入 `s_barrier_wait` 前的 wave 到达偏斜。当前 trace 不支持优先继续优化 GEMM1 output store。

## GEMM2 长延迟指令

所有 112 条 active wave 聚合后的主要 wait：

| Phase | Instruction | Hits | Avg latency | Max | Latency share | Stall/span |
|---|---|---:|---:|---:|---:|---:|
| epilogue | `s_wait_tensorcnt 0x0` | 224 | 588.16 | 1,556 | 9.21% | 8.83% |
| epilogue | `s_wait_dscnt 0x0` | 224 | 458.58 | 5,864 | 7.18% | 6.88% |
| prologue | `s_wait_kmcnt 0x0` | 1,512 | 66.64 | 3,416 | 7.04% | 6.66% |
| prologue | `s_barrier_wait 0xffff` | 112 | 787.32 | 3,123 | 6.17% | 5.91% |
| prologue | `s_wait_tensorcnt 0x6` | 112 | 635.06 | 2,235 | 4.97% | 4.77% |
| compute | `s_barrier_wait 0xffff` | 784 | 67.19 | 1,249 | 3.68% | 3.48% |
| compute | `s_wait_dscnt 0x0` | 896 | 14.40 | 71 | 0.90% | 0.81% |
| compute | `s_wait_tensorcnt 0x4` | 560 | 9.42 | 1,796 | 0.37% | 0.32% |

长事件累计占 active-wave span：

```text
latency > 100 cycles:  34.13%
latency > 300 cycles:  30.70%
latency > 600 cycles:  28.07%
latency > 1000 cycles: 24.85%
```

最大的单次长延迟事件：

```text
s_wait_dscnt 0x0       up to 5,864 cycles, epilogue
s_wait_kmcnt 0x0       up to 3,416 cycles, prologue
s_barrier_wait 0xffff  up to 3,123 cycles, prologue
s_wait_tensorcnt 0x6   up to 2,235 cycles, prologue/input TDM
s_wait_tensorcnt 0x0   up to 1,556 cycles, epilogue/output TDM
ds_load_b128              up to 659 cycles, isolated event
```

最终 output 序列为：

```text
ds_store_b128 ...
s_wait_dscnt 0x0
s_barrier_signal -1
s_barrier_wait 0xffff
tensor_store_from_lds ...
s_wait_tensorcnt 0x0
s_endpgm
```

其中两个 output wait 单独占据：

```text
s_wait_dscnt 0x0       7.18%
s_wait_tensorcnt 0x0   9.21%
combined               16.39% of decoded instruction latency
```

即使排除冷启动 `wv0`，最终 output wait 仍然明显：

```text
s_wait_tensorcnt 0x0   9.59%
s_wait_dscnt 0x0       3.50%
```

这说明 output drain 是 GEMM2 的持续成本，并非只由冷启动样本造成。

### GEMM2 瓶颈判断

当前短 K GEMM2 主要受固定开销和同步等待限制：

- prologue 与 epilogue 合计占 46.33% decoded latency。
- 显式 wait/barrier 占 41.46% decoded latency，并形成 39.10% exposed stall。
- output LDS-to-TDM drain 是最大的稳定瓶颈。
- 第二大瓶颈是 prologue 中的 scalar-memory、初始 TDM 和 barrier 等待。
- compute 阶段的 `s_wait_tensorcnt 0x4` 只有 0.37%，steady input-TDM pipeline 不是主要限制。
- 孤立 `ds_load_b128` 会出现较长 latency，但 LDS 指令总体 exposed stall 只有 0.23%。

这也解释了 GEMM2 使用 `B_TH=6` 没有性能提升：该 hint 只影响 input B stream，但大部分可消除的 exposed latency 位于固定 prologue 和 output drain。GEMM2 只有 8 个 K tile，B cache policy 的收益无法摊薄这些固定成本。

GEMM2 最值得优先尝试的方向是覆盖或缩短 output LDS completion 和 output TDM completion，然后再考虑合并或提前执行 scalar prologue load。在解决这些等待前，继续调节 B cache hint 很难显著改变 kernel 时间。

## WGP 和 SE 负载均衡

执行命令：

```bash
python3 my_code/analyze_att_capture.py \
  --dir my_code/thread_trace_runs/e64_t1536_topk8_gemm1_bth6_att_20260930 \
  --no-plot

python3 my_code/analyze_att_capture.py \
  --dir my_code/thread_trace_runs/e64_t1536_topk8_gemm2_att_20260930 \
  --no-plot
```

每个独立 SIMD-select capture、每个 SE 都观察到 16 个 physical WGP。

| Metric | GEMM1 | GEMM2 |
|---|---:|---:|
| Physical-WGP completion imbalance mean | 0.706% | 0.699% |
| Physical-WGP completion imbalance median | 0.480% | 0.684% |
| Physical-WGP completion imbalance maximum | 2.141% | 1.249% |
| WGP envelope spread mean | 0.713% | 0.720% |
| WGP envelope spread maximum | 2.115% | 1.241% |
| 各 capture/SE 的 median idle-gap fraction 平均值 | 1.14% | 2.19% |

两个 kernel 的 physical-WGP 完成时间差都很小，没有证据表明 WGP 分配不均或最后少数 WGP 的长尾是当前主要瓶颈。

各次独立 capture 的 GFXCLK：

| Metric | GEMM1 | GEMM2 |
|---|---:|---:|
| Combined tick-weighted mean | 1,898.404 MHz | 1,961.106 MHz |
| Capture × SE mean spread | 65.380 MHz | 51.405 MHz |
| Maximum occupancy timestamp span | 146,025 cycles | 112,278 cycles |
| 对应 all-SE realtime span | 75.440 us | 57.270 us |
| 由 occupancy maximum 推导的频率 | 1,935.644 MHz | 1,960.503 MHz |

各 SIMD-select capture 是独立运行，shader cycle counter 在不同 SIMD 间也不同步。因此上表适合检查时钟状态和单次 capture 内的均衡程度，不能解释为 4 个 SIMD 同时执行时的相对时间。

## 最终判断

GEMM1 和 GEMM2 都没有明显的 physical-WGP 负载不均问题。瓶颈位于每个 active workgroup 内部：

1. GEMM1 的主体是较长的 WMMA hotloop。主要优化空间是 compute 阶段反复出现的 `s_wait_tensorcnt 0x4` 和 `s_barrier_wait`，二者合计约占 decoded latency 的 16.27%。
2. GEMM2 明显更 wait-bound。短 K-loop 无法摊薄固定 setup/drain；output `s_wait_dscnt 0x0` 和最终 `s_wait_tensorcnt 0x0` 合计占 16.39%，prologue 的 scalar/TDM/barrier wait 又占约 18.18%。
3. 原始 LDS 指令不是整体瓶颈。尽管存在少量高 latency `ds_load_b128`，其总体 exposed-stall 占比在 GEMM1 中低于 1%，在 GEMM2 中低于 0.3%。
4. GEMM2 的 B cache hint 优先级较低。trace 指向 output drain 和固定启动同步，这与 GEMM2 `B_TH=6` 实测没有收益相吻合。

## 分析产物

两个 trace 目录均包含：

```text
logs/analyze_att_capture.log
logs/trace_segment_full_representatives.log
logs/aggregate_active_wave_latency.log
logs/phase_wait_summary.json
analysis/full_representative.json
analysis/trace_segment_cycles.py
analysis/get_isa_runner_att.sh
analysis/aggregate_att_waits.py
analysis/att_phase_summary.py
```

工具 SHA256：

```text
trace_segment_cycles.py:
7f724112e19d50c74ba6b6e3498c004562411bb98eed3c165fb3efbfd31a17f5

analyze_att_capture.py:
b7269e54fb4850673f627cbb3e6e734cbd84329fdcce01bc5143c707ec6acf2b
```
