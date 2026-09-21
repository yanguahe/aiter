# GEMM2 `apre_wpt2_mg4_fc28_ostore2p_ow2` thread trace 分析

## 1. 结论摘要

本次在 a07-3 对以下 GEMM2 kernel 抓取了四份独立的 all-SE ATT thread trace：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2
```

主要结论如下：

1. 当前 kernel 不是由跨 WGP 负载不均衡主导。`analyze_att_capture.py` 观察到的 16 个
   capture×SE 组合中，physical-WGP completion imbalance 平均为 `0.359%`、中位数为
   `0.111%`，最坏值为 `1.247%`。
2. active wave 平均执行 `17,209.8 cycles`，其中：
   - prologue：`26.31%`；
   - WMMA region：`57.42%`；
   - epilogue：`16.27%`。
3. 所有显式 `s_*wait_*` 和 `s_barrier_wait` 的动态 stall 合计占 active-wave timeline 的
   `28.51%`：
   - `s_wait_tensorcnt`：`13.68%`；
   - `s_barrier_wait`：`10.06%`；
   - `s_wait_dscnt`：`2.67%`；
   - `s_wait_kmcnt`：`1.95%`；
   - `s_wait_xcnt`：`0.15%`。
4. 最大单点瓶颈是 epilogue 最后的：

   ```asm
   tensor_store_from_lds s[12:15], s[4:11]
   s_wait_tensorcnt 0x0
   s_endpgm
   ```

   其中 `PC=0x7688` 的 `s_wait_tensorcnt 0x0` 平均暴露 `1,363.5 cycles/wave`，占完整
   active-wave timeline 的 `7.92%`，且每次执行都超过 `300 cycles`。
5. hot region 仍首先受 WMMA/XDL 吞吐与 LDS operand feeding 约束。互斥 issue-gap 分解中：
   - WMMA-attributed gap：`33.38%`；
   - LDS-read-attributed gap：`10.94%`；
   - hot region 内两者合计占该阶段约 `76.08%`。
6. `ds_load_b128` 偶尔出现极长 latency，最大值为 `12,100 cycles`，但在 `519,168` 次动态
   执行中仅 `105` 次达到 `300 cycles`、仅 `31` 次达到 `1,000 cycles`；其 p99 latency 只有
   `5 cycles`。这是稀有长尾，不是普遍 LDS latency。
7. 实际 code object metadata 报告：LDS=`278,528 B`（`272 KiB`）、VGPR=`804/wave`
   （按 16 对齐实际分配 `816/wave`）、SGPR=`58/wave`，VGPR/SGPR 均无 spill，scratch 为 0。
   `272 KiB` 已占 gfx1250 每 WGP 最大 `320 KiB` LDS 的 `85%`，仅 LDS 就足以限制为一个
   resident workgroup/WGP，与 ATT 观察到的单 concurrent slot 一致。

因此当前瓶颈排序是：

```text
WMMA/XDL steady-state throughput
  > final output TDM drain
  > initial TDM readiness / workgroup rendezvous
  > LDS completion and scalar-memory waits
  > physical-WGP load imbalance
```

## 2. Capture 信息与复现方式

### 2.1 环境

```text
host:   heliosr-1b114-a07-3
GPU:    gfx1250
branch: hyg_gfx1250_gemm_a4w4
remote HEAD: 3b0ad10ad1bf98c0662efd9c05cdf006aa7791ba
capture UTC: 2026-09-21 04:21:24
```

远端 HEAD 尚未包含本地最新 commit，但实际用于运行的两个 kernel 文件与本地已提交版本内容一致：

```text
57589b068d751bca4ccc3ad18c5bfda1bab9e3bb60e2e68d52b140b89e9d9947
  aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py

d59db04b315dac5b77ced78254ee515afa89ce2ecb0c06bb81fae5a3ffdc5eab
  aiter/ops/flydsl/moe_kernels.py
```

抓取前：

```text
/data/yanguahe/code/gpu_users.sh: no GPU/KFD process
GPU use: 0%
VRAM used: 173,154,304 B
```

四个 capture 均在其他 GPU 用户出现前完成。最后一个 SIMD capture 于
`2026-09-21 04:22:06 UTC` 完成；之后约 `04:22:30 UTC` 用户 `tingchen` 才启动新的 GPU
任务，因此该任务没有污染本次 capture。

### 2.2 Capture 参数

```text
att_target_cu              = 1
att_shader_engine_mask     = 0xf
att_simd_select            = 0,1,2,3（四次独立运行）
kernel_iteration_range     = [8]
att_buffer_size            = 0x10000000
e2e iterations per capture = 2
```

原 `reproduce_compare.sh` 的 preflight 只输出 GEMM2 通用 symbol 前缀，而 ATT YAML 使用 exact
regex，第一次运行因此没有匹配到 dispatch。正式 capture 使用 `/tmp` 中的脚本副本，仅把过滤串
固定为上面的完整 symbol；仓库内的 `my_code/reproduce_compare.sh` 没有被修改。

等价的正式命令为：

```bash
AITER_ATT_EXACT_KERNEL=a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2 \
CASE_LIST=apre_wpt2_mg4_fc28_ostore2p_ow2 \
AITER_ATT_SHADER_ENGINE_MASK=0xf \
AITER_ATT_SIMD_LIST=0,1,2,3 \
AITER_ATT_TIMEOUT_SECONDS=300 \
AITER_ATT_E2E_ITERS=2 \
bash /tmp/reproduce_compare_gemm2_att_exact.sh att --gemm2
```

原始数据保存在远端：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T042124Z_gemm2_att/
    att/apre_wpt2_mg4_fc28_ostore2p_ow2/
```

该 case 的 ATT 数据约为 `611 MiB`。

### 2.3 Capture 完整性

| capture | `.att` 文件 | `code.json` | decoded wave JSON | 完整 active wave | 完整 sentinel wave |
|---|---:|---:|---:|---:|---:|
| SIMD0-select | 4 | 1 | 177 | 171 | 6 |
| SIMD1-select | 4 | 1 | 204 | 168 | 36 |
| SIMD2-select | 4 | 1 | 213 | 168 | 45 |
| SIMD3-select | 4 | 1 | 221 | 169 | 52 |
| **合计** | **16** | **4** | **815** | **676** | **139** |

四份 `code.json` 的动态 hit/latency 汇总字段不同，因此文件 SHA256 不同；去除这些动态字段后，
四份 capture 的 `(Vaddr, ISA)` 静态序列完全相同，统一 SHA256 为：

```text
020ccca26a7fc5771a97bd4e7b97a6ed908d1b814ad49db99d4aa2bd020bf624
```

每条 active wave 都动态执行了恰好：

```text
768 x v_wmma_scale_f32_32x16x128_f4
768 x ds_load_b128
 24 x tensor_load_to_lds
 64 x ds_store_b128
  2 x tensor_store_from_lds
```

因此 active-wave 样本包含完整 GEMM2 数值路径。sentinel wave 没有执行 WMMA，只执行约 207 条
指令、约 `684 cycles` 后退出；所有占比统计均将其剔除。

## 3. 分析工具与统计口径

### 3.1 `trace_segment_cycles.py`

使用 `flydsl-align-reference-kernel.mdc:104-111` 指定的 canonical 工具：

```text
my_code/moe_gemm1_act1_optimized/trace_segment_cycles.py
SHA256 = 6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a
```

四份 capture 分别以以下动态区间运行该工具：

```text
start: s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
end:   s_endpgm
```

命令形式为：

```bash
python3 trace_segment_cycles.py <ui_output_dir> \
  --interval-start 's_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2' \
  --interval-end 's_endpgm' \
  --interval-print-limit 8 \
  --hide-occurrence-details --top-events -k 40
```

canonical 输出保存在 capture 的 `analysis/canonical_full_wave_simd{0,1,2,3}.log`。

### 3.2 active-wave 过滤与 denominator

`trace_segment_cycles.py` 的完整区间输出同时包含 active 和 sentinel wave。补充聚合器复用
canonical parser，以动态执行过 `v_wmma_scale_f32_32x16x128_f4` 作为 active 判据。

本文“占整个 kernel cycle 的比例”使用以下可验证口径：

```text
denominator = 所有完整 active wave 中
              [第一条指令 issue, s_endpgm issue) span 的总和
```

即所有百分比是平均 active-wave timeline 的暴露比例，不是把并行 wave 的 cycle 简单解释成
GPU wall time。这个口径可以回答某类依赖在每条计算 wave 上暴露了多少，但不能把互相并行的
多个 wave 当作串行执行。

### 3.3 `stall`、`latency` 与 issue gap

ATT event 为：

```text
[timestamp, type, stall, latency, code_idx]
```

- 对 `s_wait_*` 和 `s_barrier_wait`，本文累加 `stall`，表示 wave 真正不能继续 issue 的暴露等待。
- 非 wait 指令的 `latency` 可能与后续独立指令重叠，不能直接相加并称为 kernel 时间。
- 为分析非 wait 指令，另用：

  ```text
  next_dynamic_instruction.timestamp - current_instruction.timestamp
  ```

  做互斥 issue-gap 归因。所有 category 恰好组成 100% timeline，但标签表示“该指令之后的
  issue gap”，不严格等同于该指令本身的执行 latency。

## 4. active-wave 总体与阶段分布

676 条完整 active wave：

| metric | cycles |
|---|---:|
| mean | `17,209.8` |
| median | `16,359` |
| p90 | `20,895` |
| p99 | `25,691` |
| min / max | `13,557 / 30,059` |
| CV | `14.92%` |

按第一条和最后一条动态 WMMA 划分：

| 阶段 | mean cycles/wave | median | p90 | 完整 wave 占比 |
|---|---:|---:|---:|---:|
| prologue | `4,528.3` | `4,239` | `5,510` | `26.31%` |
| WMMA region | `9,881.4` | `9,501` | `10,759` | `57.42%` |
| epilogue | `2,800.1` | `2,378` | `4,477` | `16.27%` |

各阶段内部显式 wait：

| 阶段 | wait cycles/wave | 占完整 wave | 占本阶段 |
|---|---:|---:|---:|
| prologue | `1,828.4` | `10.62%` | `40.38%` |
| WMMA region | `1,393.4` | `8.10%` | `14.10%` |
| epilogue | `1,684.9` | `9.79%` | `60.17%` |

这说明：

- hot region 时间最长，但其主体是 WMMA 和 LDS operand feeding，而不是显式 wait；
- epilogue 虽然只占 `16.27%`，其中约六成是显式等待，优化空间最集中；
- prologue 有约四成时间是等待，首轮 TDM ready 与 wave rendezvous 仍然明显。

## 5. `s_*wait_*` 与 barrier 分析

### 5.1 wait family 总占比

| wait family | stall cycles/wave | 完整 active-wave 占比 |
|---|---:|---:|
| `s_wait_tensorcnt` | `2,353.9` | **`13.68%`** |
| `s_barrier_wait` | `1,730.6` | **`10.06%`** |
| `s_wait_dscnt` | `460.0` | `2.67%` |
| `s_wait_kmcnt` | `336.2` | `1.95%` |
| `s_wait_xcnt` | `26.0` | `0.15%` |
| **全部显式 wait** | **`4,906.8`** | **`28.51%`** |

硬件语义来自 CDNA5 ISA：

- §5.7（page 62）说明 `S_WAIT_*CNT` 在目标 outstanding count 未降到指定值前使 wave
  处于 inactive、不能继续 issue；
- §10.11.1（pages 149–150）说明 `TENSOR_LOAD_TO_LDS` 和
  `TENSOR_STORE_FROM_LDS` 由 `TENSORcnt` 跟踪，同一 wave 的 tensor instructions 按序完成；
- §5.6（pages 54–55）说明 `S_BARRIER_WAIT` 等待同一 workgroup 的所有 waves signal；
- MI400 Shader Programming Guide §4.3.7.2 说明 `DScnt` 跟踪 LDS 指令，`KMcnt` 跟踪
  scalar-memory/message 指令。

### 5.2 主要 wait site

| PC | instruction | hits/wave | stall cycles/wave | 完整 wave 占比 | p90 latency | max latency |
|---|---|---:|---:|---:|---:|---:|
| `0x7688` | `s_wait_tensorcnt 0x0` | `1.00` | `1,363.5` | **`7.92%`** | `1,507.5` | `8,249` |
| `0x2440` | `s_barrier_wait 0xffff` | `1.00` | `826.4` | **`4.80%`** | `1,914.0` | `10,457` |
| `0x23c0` | `s_wait_tensorcnt 0x6` | `1.00` | `635.0` | **`3.69%`** | `1,620.5` | `10,202` |
| `0x32b0` | `s_barrier_wait 0xffff` | `4.02` | `474.7` | `2.76%` | `332.1` | `8,271` |
| `0x431c` | `s_wait_tensorcnt 0x4` | `3.98` | `249.7` | `1.45%` | `159.6` | `9,028` |
| `0x4330` | `s_barrier_wait 0xffff` | `3.98` | `171.2` | `0.99%` | `54.0` | `9,194` |
| `0x7670` | `s_wait_dscnt 0x0` | `1.00` | `147.8` | `0.86%` | `188.0` | `506` |
| `0x6e6c` | `s_wait_dscnt 0x0` | `1.00` | `134.6` | `0.78%` | `156.0` | `165` |
| `0x4abc` | `s_barrier_wait 0xffff` | `1.00` | `106.1` | `0.62%` | `273.5` | `1,281` |
| `0x5e14` | `s_barrier_wait 0xffff` | `1.00` | `64.4` | `0.37%` | `104.0` | `8,306` |
| `0x5458` | `s_barrier_wait 0xffff` | `1.00` | `49.7` | `0.29%` | `121.5` | `1,131` |
| `0x1d18` | `s_wait_kmcnt 0x0` | `1.00` | `45.1` | `0.26%` | `28.0` | `850` |

#### Final output drain

最大单点 `0x7688` 位于第二阶段 output TDM 之后：

```asm
0x765c  ds_store_b128 ...
0x7668  ds_store_b128 ...
0x7670  s_wait_dscnt 0x0
0x7674  s_barrier_signal -1
0x7678  s_barrier_wait 0xffff
0x767c  tensor_store_from_lds s[12:15], s[4:11]
0x7688  s_wait_tensorcnt 0x0
0x768c  s_endpgm
```

`tensor_store_from_lds` 自身平均 issue latency 只有 `1 cycle`；长时间体现在随后必须 drain
全部 outstanding output TDM 的 `s_wait_tensorcnt 0x0`。因此瓶颈不是“发 TDM 指令很慢”，而是
kernel 尾部没有剩余计算可覆盖最终 global store completion。

该单点占完整 wave 的 `7.92%`，也占 epilogue 平均 `2,800.1 cycles` 的约 `48.7%`。完全消除它
是不现实的，但它给出了通过跨 task overlap 隐藏 output drain 的一阶理论上限。

#### Initial input readiness

prologue 的主要等待是：

```asm
0x23c0  s_wait_tensorcnt 0x6
...
0x243c  s_mov_b32 s16, 4
0x2440  s_barrier_wait 0xffff
```

两者合计占完整 wave 的 `8.49%`。四个独立 SIMD-select capture 呈现明显互补：

| capture | `s_wait_tensorcnt` share | `s_barrier_wait` share |
|---|---:|---:|
| SIMD0-select | `8.76%` | `16.04%` |
| SIMD1-select | `19.08%` | `3.89%` |
| SIMD2-select | `18.40%` | `4.54%` |
| SIMD3-select | `8.58%` | `15.63%` |

这与 workgroup 内 TDM owner/consumer 的 readiness skew 一致：部分 waves 直接暴露 tensor
completion，另一些 waves 较快到达 barrier 后等待同伴。由于四个 SIMD-select 是独立运行，不能
把它们当成同一次运行中四个 wave 的同步时间线；但反复出现的两组 wait 分布仍说明首轮
load-to-LDS 与 rendezvous 尚未完全被 descriptor/setup 工作覆盖。

#### LDS 与 scalar-memory wait

两个 output `s_wait_dscnt 0x0` 共占约 `1.64%`，用于保证 LDS output stores 已完成后再让 TDM
读取 LDS。所有 `s_wait_dscnt` 合计 `2.67%`，明显低于 TENSORcnt 和 barrier。

`s_wait_kmcnt` 合计 `1.95%`，主要位于 prologue 的 scalar load、descriptor 和
`m_tile_map`/expert-boundary 准备区。因此 binary search/SMEM latency 是次要成本，而不是当前
第一瓶颈；即使假设全部消除，其理论上限也不到 `2%`。

## 6. 非 wait 长 latency 与互斥 issue-gap

### 6.1 opcode 级结果

| opcode | hits/wave | mean ATT latency | p99 latency | max latency | issue-gap cycles/wave | timeline 占比 |
|---|---:|---:|---:|---:|---:|---:|
| `v_wmma_scale_f32_32x16x128_f4` | `768` | `13.74` | `15` | `15` | `5,744.2` | **`33.38%`** |
| `s_wait_tensorcnt` | `14` | `169.14` | `1,621.4` | `10,202` | `2,368.9` | `13.76%` |
| `s_barrier_wait` | `15` | `116.38` | `1,745.5` | `10,457` | `1,775.6` | `10.32%` |
| `ds_load_b128` | `768` | `1.42` | `5` | `12,100` | `1,554.4` | `9.03%` |
| `s_mov_b32` | `103` | `1.01` | `2` | `2` | `1,163.3` | `6.76%` |
| `ds_store_b128` | `64` | `3.00` | `3` | `3` | `671.5` | `3.90%` |
| `s_wait_dscnt` | `121` | `4.80` | `136` | `1,812` | `581.5` | `3.38%` |
| `s_wait_kmcnt` | `14.5` | `24.21` | `253` | `1,357` | `350.7` | `2.04%` |
| `v_cvt_pk_bf16_f32` | `256` | `1.02` | `2` | `4` | `330.9` | `1.92%` |
| `tensor_load_to_lds` | `24` | `1.00` | `1` | `1` | `72.5` | `0.42%` |
| `tensor_store_from_lds` | `2` | `1.00` | `1` | `1` | `5.0` | `0.03%` |

`v_wmma_scale_f32_32x16x128_f4` 的 latency 稳定在最多 `15 cycles`，但每条 active wave 必须
执行 768 次。其 issue-gap 累计为 `33.38%`，说明 hotloop 的基础成本首先来自不可省略的矩阵
计算吞吐和相邻 operand/dependency 调度，而不是某几次异常长 WMMA。

`ds_load_b128` 的最大 `12,100-cycle` latency 是长尾：

```text
dynamic ds_load_b128 events = 519,168
latency >= 300 cycles       = 105  (0.0202%)
latency >= 1,000 cycles     = 31   (0.0060%)
p99 latency                 = 5 cycles
```

因此不应根据单个最大值判断 LDS 普遍需要 12k cycles。LDS read 的整体 issue-gap 占比仍有
`10.94%`，但它更接近 768 次 operand feed 的累计成本和少量 scheduler tail。

`ds_store_b128` 的 ATT latency 始终为 `3 cycles`。少数 store 后出现数千 cycle issue gap，属于
后续 dependency、wave scheduling 或同步暴露在该静态 PC 后面；不能解释为 DS store 自身耗时。
同理，`s_mov_b32` latency 约为 1，但其后 issue gap 累计 `6.76%`，不能据此把 SALU move 当成
6.76% 的执行瓶颈。

### 6.2 完整互斥 timeline 分解

下表对每对相邻动态指令之间的 timestamp gap 只归属一次，因此总和为 100%：

| category | cycles/wave | 完整 wave 占比 |
|---|---:|---:|
| WMMA-attributed gap | `5,744.2` | **`33.38%`** |
| SALU/control-attributed gap | `2,437.7` | `14.16%` |
| TENSORcnt wait | `2,368.9` | **`13.76%`** |
| LDS read-attributed gap | `1,883.1` | **`10.94%`** |
| barrier wait | `1,775.6` | **`10.32%`** |
| other VALU-attributed gap | `1,278.2` | `7.43%` |
| LDS write-attributed gap | `671.5` | `3.90%` |
| DScnt wait | `581.5` | `3.38%` |
| KMcnt wait | `350.7` | `2.04%` |
| TDM issue-attributed gap | `77.5` | `0.45%` |
| Xcnt wait | `27.0` | `0.16%` |
| scalar-memory-attributed gap | `14.0` | `0.08%` |

其中 `TDM issue-attributed gap` 很小，而 `TENSORcnt wait` 很大，再次说明问题集中在数据搬运
完成时间及其覆盖窗口，而不是 TDM opcode 的发射开销。

按阶段看：

- prologue 中 SALU/control-attributed gap 占该阶段 `43.95%`，TENSORcnt wait `14.07%`，
  barrier wait `18.34%`；
- WMMA region 中 WMMA-attributed gap 占 `58.13%`，LDS read-attributed gap 占 `17.95%`，
  barrier wait 占 `9.07%`；
- epilogue 中 TENSORcnt wait 占 `48.80%`，LDS write-attributed gap 占 `23.98%`，DScnt wait
  占 `10.16%`。

## 7. physical-WGP 负载均衡

`my_code/analyze_att_capture.py` 的 SHA256 为：

```text
b7269e54fb4850673f627cbb3e6e734cbd84329fdcce01bc5143c707ec6acf2b
```

命令：

```bash
python3 my_code/analyze_att_capture.py \
  --dir <case-root> --no-plot
```

每份独立 capture 都观察到：

```text
physical WGPs              = 64
physical-SIMD keys         = 256
max distinct slot IDs/key  = 1
max concurrent slots/key   = 1
wave lifetimes             = 13,440
```

即本 kernel 在采样中没有第二个 concurrent resident slot；等待较难通过同一 physical SIMD 上的
另一 resident wave 隐藏。

`analyze_att_capture.py` 的 completion metric 为：

```text
WGP envelope = last wave end - first wave start
final-end span = max(WGP final end) - min(WGP final end)
completion imbalance = final-end span / median(WGP envelope)
```

| capture | WGP 数 | median envelope 范围 | final-end span 范围 | 平均 imbalance | 最大 imbalance | mean GFXCLK |
|---|---:|---:|---:|---:|---:|---:|
| SIMD0-select | 64 | `736,177–752,565` | `656–5,232` | `0.396%` | `0.711%` | `2,235.0 MHz` |
| SIMD1-select | 64 | `755,385–771,686` | `598–806` | `0.086%` | `0.105%` | `2,237.5 MHz` |
| SIMD2-select | 64 | `734,119–745,672` | `552–9,298` | `0.595%` | **`1.247%`** | `2,225.1 MHz` |
| SIMD3-select | 64 | `764,470–776,121` | `842–6,461` | `0.359%` | `0.837%` | `2,241.8 MHz` |

16 个 capture×SE 行的汇总：

```text
completion imbalance mean   = 0.359%
completion imbalance median = 0.111%
completion imbalance max    = 1.247%
```

最坏情况是 SIMD2-select/SE1：16 个 WGP 的 median envelope 为 `745,671.5 cycles`，final-end
span 为 `9,298 cycles`。即使理想化地完全消除这组 completion spread，对该组的理论收益上限也
只有约 `1.25%`。

因此没有证据表明当前 kernel 被某几个 physical WGP 的严重拖尾所限制。需要注意：

- 四个 SIMD-select capture 是独立运行，跨 capture 或跨 SE 的绝对 shader timestamp 不能直接
  相减；
- occupancy 数据没有 logical WG/tile ID，不能直接给每个 WGP 统计 FLOPs 或 tile 数；
- barrier 会让结束时间趋同，因此“最终完成均衡”不意味着内部没有 barrier stall；它只排除了
  明显的 grid-level WGP straggler。

## 8. LDS、SGPR 与 VGPR 使用量

资源使用量从本次 ATT 保存的实际 ELF code object 中读取，而不是从源码 tile 大小反推。对应文件为：

```text
thread_trace/simd0/kernel/rpf_v3/out_gfx1250_code_object_id_13.out
SHA256 = 304130c9415a8d26f272b9d4ad9e4f580b6353af2b4aa179600242a63ff730c6
```

解析 ELF `.note` 中的 `NT_AMDGPU_METADATA` 得到：

```text
.name                        = a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2
.reqd_workgroup_size         = [128, 1, 1]
.wavefront_size              = 32
.cluster_dims                = [4, 1, 1]
.group_segment_fixed_size    = 278528 B
.private_segment_fixed_size  = 0 B
.sgpr_count                  = 58
.sgpr_spill_count            = 0
.vgpr_count                  = 804
.vgpr_spill_count            = 0
```

### 8.1 LDS

```text
static LDS/workgroup = 278,528 B = 272 KiB
gfx1250 LDS limit    = 320 KiB/WGP
LDS usage ratio      = 272 / 320 = 85.0%
allocation blocks    = 278,528 / 2,048 = 136 blocks
```

CDNA5 ISA §3.3.4 说明 LDS 以 `2,048 B` 为单位分配，每个 wave 或 workgroup 最多使用
`320 KiB`。本 kernel 的 `272 KiB` 已是合法 allocation-unit 的整数倍。

一个 workgroup 有 `128 / 32 = 4 waves`，且这四个 waves 共享同一份 `272 KiB` LDS。两个这样的
workgroup 至少需要：

```text
2 * 272 KiB = 544 KiB > 320 KiB
```

所以仅 LDS 一项就足以把每个 WGP 的 resident workgroup 数限制为 1。这与
`analyze_att_capture.py` 观察到的每个 physical-SIMD key 最多 1 个 concurrent slot 一致。

硬件资料还说明 WGP 的本地存储单元同时承担 LDS 和 WGP$，总容量为 `384 KiB`，其中最多
`320 KiB` 可配置为 LDS。`272 KiB` LDS 因此不仅限制 occupancy，也压缩了可用 WGP$ 容量；但
metadata 本身不包含运行时 cache partition 的直接观测值，本文不进一步假设精确 WGP$ 容量。

### 8.2 VGPR

metadata 报告：

```text
logical VGPR count = 804 VGPR/wave
VGPR spill count   = 0
```

CDNA5 ISA §3.3.2.1 规定 wave32 的 VGPR 以 16 个为一组分配，因此实际 allocation 为：

```text
ceil(804 / 16) * 16 = 816 VGPR/wave
```

一个 VGPR 对 wave32 包含 `32 lanes * 4 B = 128 B`，所以：

```text
compiler-visible VGPR data/wave = 804 * 128 B = 102,912 B = 100.5 KiB
allocated VGPR data/wave        = 816 * 128 B = 104,448 B = 102 KiB
4-wave workgroup aggregate      = 4 * 102 KiB = 408 KiB
```

最后一项分布在 WGP 的四个 SIMD vector register files 上，不应理解为一块连续共享存储。
`804` 已接近每个 wave 最多 `1,024 VGPR` 的架构上限，而且 ISA 中确实大量使用
`s_set_vgpr_msb` 访问 VGPR 256 以上的编号。虽然仅凭公开资料不能从 `804` 精确推导每个 SIMD
还能容纳多少 wave，但它表明当前 kernel 的 accumulator/live-range 压力很高；实测 occupancy
也没有第二 resident slot。

### 8.3 SGPR

metadata 报告：

```text
SGPR count       = 58 SGPR/wave
SGPR spill count = 0
```

按 32-bit SGPR 计算，compiler-visible scalar state 为：

```text
58 * 4 B = 232 B/wave
```

MI400 Shader Programming Guide §3.3.1.1 和 CDNA5 ISA §3.3.1.1 同时说明：MI400/CDNA5 每条
wave 固定分配 `106` 个 normal SGPR，另有 `VCC_LO/VCC_HI` 和 `16 TTMPs`。因此 SGPR 与旧架构
上按使用量阶梯限制 occupancy 的模式不同：`.sgpr_count=58` 是 kernel 的编译器可见使用量，
而硬件 normal-SGPR allocation 仍是固定的 106。这里不存在 SGPR spill，也没有证据表明 SGPR
是当前 occupancy=1 的直接原因。

### 8.4 资源瓶颈结论

| resource | metadata 使用量 | allocation/限制 | 判断 |
|---|---:|---:|---|
| LDS | `278,528 B` | `272 KiB / 320 KiB = 85.0%` | 单独即可限制为 1 resident WG/WGP |
| VGPR | `804/wave` | 按 16 对齐为 `816/wave` | 压力很高，无 spill |
| SGPR | `58/wave` | MI400 每 wave 固定 106 normal SGPR | 无 spill，不是首要 occupancy 限制 |
| scratch/private | `0 B` | — | 无 scratch spill |

当前最确定的 occupancy 限制来自 `272 KiB` LDS。VGPR 使用量同样很高，但在没有额外物理
VGPR-file allocation 数据的情况下，不把它单独宣称为 occupancy=1 的充分原因。

## 9. 当前性能瓶颈判断

### 9.1 第一层：WMMA/XDL 与 LDS operand feeding

每条 active wave 固定执行 768 条 scaled-FP4 WMMA。WMMA region 占完整 wave 的 `57.42%`；
在该 region 内，WMMA 与 LDS-read issue gap 合计约：

```text
58.13% + 17.95% = 76.08%
```

因此 steady-state 主体已经是矩阵计算与 operand feed。WMMA latency 本身稳定，不存在少量异常
WMMA 拉长 kernel 的证据；继续优化需要改变有效的 WMMA issue cadence、operand reuse 或 LDS
feed，而不是处理个别 latency outlier。

### 9.2 第二层：final output TDM drain

`PC=0x7688` 单点占 `7.92%`，是最明确、最集中的 exposed stall。它位于 kernel 尾部，无法再被
本 workgroup 的计算覆盖。若未来采用 persistent task loop，并能安全地让前一 task 的 output
TDM drain 与下一 task 的 expert lookup/descriptor/input prefetch 重叠，其理论上限首先由这
`7.92%` 决定。

但这需要同时证明：

- output LDS buffer 在 TDM 完成前不会被下一 task 覆盖；
- 每个 wave 的 `TENSORcnt` 顺序和 descriptor lifetime 正确；
- non-balanced `m_tile_map` binary search 仍逐 task 执行；
- occupancy=1 下新增 LDS/VGPR 不会进一步降低并行能力。

### 9.3 第三层：initial TDM readiness 与 workgroup phase skew

`0x23c0` 与 `0x2440` 合计占 `8.49%`。SIMD0/3 更偏向 barrier wait，SIMD1/2 更偏向
TENSORcnt wait，说明同一个 workgroup 内的 waves 在首轮 operand ready 时间上存在角色/阶段
偏差。可以研究更早发起首轮 TDM、减少 descriptor critical path，或重新安排 owner/consumer
之间的独立 SALU/VALU 工作；不能简单删除 barrier 或放宽 wait count。

### 9.4 不是优先方向的项目

- **physical-WGP 重平衡**：最坏 completion imbalance 仅 `1.247%`，收益上限太低。
- **追逐单个 `ds_load_b128` 最大 latency**：p99 仅 `5 cycles`，12k-cycle 事件极少。
- **只优化 TDM instruction issue**：`tensor_load_to_lds` 与 `tensor_store_from_lds` 的 issue-gap
  合计不到 `0.5%`，真正成本已经转移到 `s_wait_tensorcnt`。
- **只删除 `m_tile_map` binary search**：不仅违反 non-balanced 要求，相关 KMcnt 总占比也只有
  `1.95%`，不是主要收益来源。

## 10. 优化收益上限与建议顺序

以下是 thread trace 给出的理想化上限，不代表实际可全部实现：

| 假设完全消除的部分 | active-wave cycle 上限 |
|---|---:|
| physical-WGP completion imbalance | 平均 `0.36%`，最坏 `1.25%` |
| final `s_wait_tensorcnt 0x0` | `7.92%` |
| 全部 TENSORcnt stall | `13.68%` |
| 全部 barrier stall | `10.06%` |
| 全部显式 wait | `28.51%` |

合理的后续顺序是：

1. 优先研究跨 task 的 output TDM overlap，目标是隐藏 final `s_wait_tensorcnt 0x0` 的一部分；
2. 再研究首轮 input TDM 与 descriptor/binary-search 工作的重叠，降低 `0x23c0`/`0x2440`；
3. 若以上仍不足，再重新设计 WMMA/LDS schedule；当前 768 次 WMMA 和 768 次 `ds_load_b128`
   的 steady-state 成本不会通过小幅 wait-count 调整消失；
4. 不应优先投入 WGP load-balancing 或稀有 LDS latency outlier。

若目标是再提升约 `12%`，单纯解决 WGP balance 不可能达到；仅完全隐藏 final output wait 的理论
上限也只有 `7.92%`。必须同时减少 output drain 和 prologue/input readiness，或者对 hotloop
WMMA/LDS pipeline 做结构性改变。

## 11. 数据与限制

- ATT 会显著扰动被测程序，因此本文不使用 tracing 期间 profiler 显示的微秒数作为性能数据。
- 正常 preflight 中 GEMM2 为约 `337 us`，与该版本此前约 `334 us` 的正常 benchmark 一致，
  仅用于确认 dispatch 身份。
- `latency` 表中的最大值可能是 wave deschedule、下游 dependency 或资源竞争共同造成，不能直接
  解释为某条指令的固定硬件 latency。
- issue-gap category 是互斥 timeline 分解；wait stall 表是另一种观测口径，两张表不能再次相加。
- raw capture 保留在 a07-3 上，未把约 `611 MiB` 的运行产物加入 Git。

参考资料：

- `mi400_hw_wiki/raw/papers/mi400_hd_txt/MI450/amd-instinct-cdna5-instruction-set-architecture.txt`
  - §5.6 Barriers，pages 54–55；
  - §5.7 Data Dependency Resolution，page 62；
  - §10.11 Tensor Data Mover Instructions，pages 149–152；
  - §7.12 WMMA，pages 102–112。
- `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`
  - §4.3.7.2 Memory Dependency Counters，pages 87–89。
