# GEMM2 persistent task pipeline 优化实验记录

## 1. 目标与约束

本轮以当前保留版本：

```text
apre_wpt2_mg4_fc28_ostore2p_ow2
```

为起点，尝试把 GEMM2 改成 persistent task pipeline，目标是使 GEMM2 kernel
耗时下降 `15%`。实验始终遵守以下约束：

- 不修改 GEMM2 的外部接口、数学功能或结果精度；
- 保留每个有效 M tile 上的 runtime `m_tile_map` binary search；
- 支持 non-balanced token/expert 分布；
- 不扩大原有 A-preshuffle 路径的适用规模；
- 没有性能收益的实现不进入 `my_code/reproduce_compare.sh`，并从正式源码清理。

测试机器为 a07-3，容器为 `hyg_fyd1`。远端仓库原本存在其他工作产生的修改和未跟踪文件，
本轮没有执行 `git clean`、`git checkout .` 或任何破坏性同步；测试时只覆盖了本任务涉及的文件，
结束后又恢复为本地当前分支对应内容。

## 2. 重新建立性能起点

GPU 空闲后执行：

```bash
CASE_LIST=apre_wpt2_mg4_fc28_ostore2p_ow2 \
ROUNDS=3 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/reproduce_compare.sh --gemm2
```

得到：

| case | GEMM2 samples (us) | median (us) | MoE e2e samples (us) | median (us) | pass |
|---|---|---:|---|---:|:---:|
| `apre_wpt2_mg4_fc28_ostore2p_ow2` | `335.146, 335.037, 335.922` | `335.146` | `1243.06, 1241.37, 1239.44` | `1241.37` | True |

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T054932Z_gemm2_e2e-const0
```

后续五轮同轮对照得到更稳定的起点：

```text
GEMM2:   334.544, 335.697, 335.058, 334.683, 334.514 us
median:  334.683 us
MoE e2e: 1238.22, 1240.31, 1239.54, 1244.96, 1242.95 us
median:  1240.31 us
```

因此，`15%` 的目标相当于：

```text
334.683 us × 0.85 = 284.481 us
```

## 3. 为什么尝试 persistent task pipeline

原始 thread trace 的主要事实为：

| 指标 | 当前 winner |
|---|---:|
| active-wave mean | `17,209.8 cycles` |
| prologue | `26.31%` |
| WMMA region | `57.42%` |
| epilogue | `16.27%` |
| all explicit waits | `28.51%` |
| `s_wait_tensorcnt` | `13.68%` |
| `s_barrier_wait` | `10.06%` |
| WMMA-attributed issue gap | `33.38%` |

GEMM2 的 B/ScaleB 输入面明显大于 A/ScaleA。若同一个 resident cluster 连续执行同一
N-group 下相邻的 M tiles，则 balanced 场景中连续 4 个 M tiles 恰好对应同一 expert，
有机会让 B/ScaleB 在同一组 WGP 上保持 locality。同时，persistent loop 理论上还能为
后续跨 task 的 prologue/epilogue overlap 提供框架。

但这只是需要实验验证的假设。原始 trace 同时表明 WGP 间负载已经很均衡，因此单纯减少
launch 数或改成固定 worker 并不能自然产生收益。

## 4. 尝试一：`4×4` persistent task packet 前置实验

### 4.1 实现思路

先复用 kernel 内已有的 `cluster_m > 1` 路径，把 GEMM2 从 `1×4` cluster 改成
`4×4` cluster：

- 一个 cluster 同时覆盖 4 个连续 M tiles 和 4 个 N tiles；
- 每个 M tile 仍独立执行原 `m_tile_map` binary search；
- B/ScaleB multicast mask 仅连接属于同一个 expert 的 M peers；
- partial/sentinel cluster 继续使用原有安全 fallback。

该实验用于判断“把相邻 M tasks 放进同一协作域并共享权重搬运”是否本身足够有利。

### 4.2 结果

| case | GEMM2 (us) | 相对同轮起点 | MoE e2e (us) | const0 |
|---|---:|---:|---:|:---:|
| winner | `335.146` | `0.00%` | `1241.37` | pass |
| `ptcm4` | `356.523` | `-6.38%` | `1260.12` | pass |

结论：B/ScaleB multicast 节省没有抵消更大的 `4×4` cluster 同步和调度成本。该实现立即
判定为失败，没有保留。

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T055252Z_gemm2_e2e-const0
```

## 5. 尝试二：一维 contiguous-M persistent loop

### 5.1 实现

保持原 `1×4` cluster。每个 persistent cluster 固定一个 N-group，并连续处理一段 M
tiles：

```text
worker -> fixed n_unit
worker -> contiguous [m_begin, m_end)
for each m_tile:
    original m_tile_map binary search
    original descriptor/TDM pipeline
    original WMMA accumulation
    original BF16 output path
    tensor_wait + workgroup barrier
```

task 间 `workgroup_barrier()` 是必须的：当前 output 先写回位于 LDS 起始位置的 C staging，
而下一 task 的 input ring 也从同一 LDS arena 开始。即使每个 wave 已等待自己的 output
TDM，其他 wave 的 output TDM 仍可能读取该区域；没有 workgroup rendezvous 就可能被下一
task 的 input TDM 覆盖。

### 5.2 初版错误地按 capacity 划分 task

初版使用 `i32_m` 划分 worker 范围。当前 workload 的 capacity 包含约 480 个 M tiles，
但 `m_tile_map[n_experts-1]` 对应的真实有效范围约为 384 个 M tiles。因此末端 worker
会执行大量只做 expert lookup 后退出的 sentinel tasks。

worker-count 单轮扫描如下。它的用途是确认并行度和尾部批次效应，不作为最终性能结论：

| workers per N-group | GEMM2 (us) | 相对 `335.146 us` 起点 |
|---:|---:|---:|
| 2 | `1478.109` | `-341.03%` |
| 4 | `791.020` | `-136.02%` |
| 8 | `448.716` | `-33.88%` |
| 9 | `430.185` | `-28.35%` |
| 16 | `443.598` | `-32.36%` |
| 24 | `442.641` | `-32.08%` |
| 32 | `353.916` | `-5.60%` |
| 48 | `377.070` | `-12.51%` |
| 64 | `349.365` | `-4.24%` |
| 96 | `348.724` | `-4.05%` |
| 128 | `338.795` | `-1.09%` |
| 192 | `336.367` | `-0.36%` |
| 384 | `342.415` | `-2.17%` |

这些结果呈明显的批次/驻留阈值，而不是平滑的 cache-locality 收益。最接近 baseline 的
`m192` 也只有单轮、且仍未更快，不能作为 winner。

### 5.3 `m9` thread trace

trace 目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T060234Z_gemm2_e2e-const0/att/
  apre_wpt2_mg4_fc28_ostore2p_ow2_ptp_m9
```

使用固定 SHA256 为
`6684004f30ac4336160f41e89a3eb4313f77943b47f0eab553c6c7d69fa4420a`
的 `trace_segment_cycles.py`，并用 `my_code/analyze_att_capture.py` 分析后得到：

| 指标 | `m9` |
|---|---:|
| complete active waves | `4` |
| logical tasks / sampled wave | `53.25` |
| active cycles / wave | `935,141.8` |
| normalized cycles / task | `17,561.3` |
| explicit waits | `36.11%` |
| `s_wait_tensorcnt` | `18.01%` |
| `s_barrier_wait` | `10.22%` |
| `s_wait_loadcnt` | `5.09%` |
| physical-WGP completion imbalance mean | `85.86%` |

最慢的单点仍是末端 `s_wait_tensorcnt 0x0`，约 `1,388 cycles/task`。更关键的是，
`m9` 的 worker 范围按 capacity 切分后，一部分 worker 执行约 53 个真实 tasks，另一部分
很快落入 sentinel 区间退出，导致极严重的 completion imbalance。

code object metadata：

| 资源 | winner | persistent `m9` |
|---|---:|---:|
| LDS | `278,528 B` | `278,528 B` |
| SGPR | `58` | `107` |
| SGPR spills | `0` | `14` |
| VGPR | `804` | `876` |
| VGPR spills | `0` | `0` |

CDNA5 ISA 文档说明普通 SGPR 为 `SGPR0..SGPR105`，`SGPR106/107` 用于 VCC。该
persistent 版本的 `sgpr_count=107` 已越过普通 SGPR 范围，并实际产生 spill。

non-balanced random 验证通过：

```text
pass                     = True
logits_diff              = 3.48778e-06
rel_l2                   = 0.00264113
MoE output hash128       = 10ef188b89c427fde6c8b322b5fd4133
ref output hash128       = d043e1891d95c1af3da5a30e37b9042a
```

因此该版本的功能路径正确，并且每个 tile 的 binary search 保留；失败原因是性能，而不是
通过改变输出或 balanced shortcut 换取速度。

## 6. 尝试三：只调度真实有效 M tiles

### 6.1 修改

worker 总范围从：

```text
ceil(i32_m / tile_m)
```

改为：

```text
ceil(m_tile_map[n_experts - 1] / tile_m)
```

这个额外读取只决定 persistent loop 的上界。进入每个有效 tile 后，原 8-step
`m_tile_map` binary search 仍完整执行，因此没有引入 balanced-only expert shortcut。

选择 `96 workers/N-group` 的依据是 balanced workload 恰有 384 个有效 M tiles，因而
每个 worker 连续处理 4 个 tiles，恰好覆盖一个 expert 的 4 个 M tiles，有利于验证
B/ScaleB locality 假设。

### 6.2 五轮同轮结果

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T063903Z_gemm2_e2e-const0
```

| case | GEMM2 samples (us) | median (us) | 相对 winner | MoE e2e median (us) | pass |
|---|---|---:|---:|---:|:---:|
| winner | `334.544, 335.697, 335.058, 334.683, 334.514` | `334.683` | `0.00%` | `1240.31` | True |
| valid-M persistent `m96` | `353.519, 353.378, 354.519, 353.289, 353.545` | `353.519` | `-5.63%` | `1258.72` | True |

### 6.3 thread trace

trace 目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T064151Z_gemm2_e2e-const0/att/
  apre_wpt2_mg4_fc28_ostore2p_ow2_ptp_valid_m96
```

将 persistent wave 中的总量按每 wave 的 4 个 logical tasks 归一化：

| 指标 | winner | valid-M persistent `m96` | 变化 |
|---|---:|---:|---:|
| cycles/task | `17,209.8` | `19,149.7` | `+11.27%` |
| all explicit waits/task | 约 `4,907` | `6,803.7` | 约 `+38.7%` |
| `s_wait_tensorcnt`/task | 约 `2,354` | `2,837.4` | 约 `+20.5%` |
| `s_barrier_wait`/task | 约 `1,731` | `2,447.2` | 约 `+41.4%` |
| `s_wait_loadcnt`/task | baseline 中无显著项 | `1,003.9` | 新增主要成本 |
| WMMA-attributed gap/task | `5,744.2` | `5,808.5` | 基本不变 |
| completion imbalance mean | `0.359%` | `7.83%` | 变差 |

关键结论：persistent 没有降低 WMMA issue gap；它主要新增了 scalar spill 对应的
`s_wait_loadcnt`，同时 task 边界的 LDS 安全同步提高了 barrier 暴露时间。B/ScaleB
locality 即使存在，也不足以抵消这些成本。

metadata：

```text
LDS                     278,528 B
SGPR                    107
SGPR spills              13
VGPR                    876
VGPR spills               0
```

## 7. 尝试四：2-D grid 直接映射 task

### 7.1 动机与修改

为减少 persistent loop 的 scalar 状态，进一步改为：

- `block_idx.x` 只编码 `worker_m` 和 cluster-local N；
- `block_idx.y` 直接表示 `n_unit`；
- `run_task` 直接接收 `(m_unit, n_unit, local_n)`；
- 删除 synthetic `bid_x`、`% n_units`、`/ n_units` 和强制 `m_major_swizzle`。

### 7.2 三轮同轮结果

运行目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T064700Z_gemm2_e2e-const0
```

| case | GEMM2 samples (us) | median (us) | 相对 winner | MoE e2e median (us) | pass |
|---|---|---:|---:|---:|:---:|
| winner | `334.951, 334.999, 334.788` | `334.951` | `0.00%` | `1240.20` | True |
| 2-D direct persistent `m96` | `364.479, 365.734, 372.919` | `365.734` | `-9.19%` | `1275.92` | True |

### 7.3 thread trace 与 metadata

trace 目录：

```text
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T064841Z_gemm2_e2e-const0/att/
  apre_wpt2_mg4_fc28_ostore2p_ow2_ptp_valid_m96
```

| 指标 | 1-D valid-M | 2-D direct | 结论 |
|---|---:|---:|---|
| cycles/task | `19,149.7` | `19,104.1` | 几乎不变 |
| `s_wait_loadcnt`/task | `1,003.9` | `1,048.0` | 未改善 |
| `s_barrier_wait`/task | `2,447.2` | `1,822.1` | 有改善 |
| `s_wait_dscnt`/task | `486.9` | `838.6` | 明显变差 |
| `s_wait_kmcnt`/task | `27.8` | `63.2` | 变差 |
| `s_wait_tensorcnt`/task | `2,837.4` | `2,812.6` | 基本不变 |
| completion imbalance mean | `7.83%` | `8.67%` | 略差 |

2-D mapping 的确把 SGPR spill 从 `13` 降到 `8`，但没有降低 `sgpr_count=107` 或
`vgpr_count=876`，而且 LDS/SMEM wait 与全局调度变差，最终 wall time 进一步回退。

```text
LDS                     278,528 B
SGPR                    107
SGPR spills               8
VGPR                    876
VGPR spills               0
```

## 8. 为什么 persistent task pipeline 没有产生收益

### 8.1 当前实现不是 launch-bound

原 winner 已有大量独立 workgroups，WGP completion imbalance 只有 `0.359%`。减少
workgroup 数和 launch-side task scheduling 并不能消除主要瓶颈，反而把多个 task
串行绑定到同一个 resident workgroup。

### 8.2 dynamic loop 使寄存器跨 task 存活

原 kernel 为：

```text
58 SGPR / 804 VGPR / 0 spill
```

persistent 版本变为：

```text
107 SGPR / 876 VGPR / 8~14 SGPR spills
```

`MI400_Shader_Programming#65.txt` 第 2076～2077 行和
`amd-instinct-cdna5-instruction-set-architecture.txt` 第 1045～1046 行均说明：每个 wave
有 106 个 normal SGPR，`SGPR106/107` 保存 VCC。persistent loop 的 task state、地址、
descriptor 和 loop-carried 值把分配推过该边界，ATT 中随之出现约 `1,000 cycles/task`
的 `s_wait_loadcnt`。

### 8.3 task 边界不能免费复用 LDS

当前 kernel 的 `278,528 B` LDS arena 同时复用于 input ring 和 output staging。下一
task 覆盖 input ring 前，必须确保当前 task 的所有 wave 已完成 output TDM 对 LDS 的读取，
因此需要额外 workgroup barrier。该成本在 valid-M trace 中约为数百到上千 cycles/task。

### 8.4 B/ScaleB locality 收益被同步与 spill 抵消

`m96` 在 balanced 场景中让每个 worker 恰好处理一个 expert 的 4 个 M tiles，是验证
expert-local B/ScaleB reuse 的有利配置。即便如此，它仍稳定回退 `5.63%`。这说明当前
层级能够获得的 cache locality 不足以抵消：

- SGPR spill 和对应 `s_wait_loadcnt`；
- task-boundary barrier；
- output drain 与下一 task input 仍串行；
- persistent grid 对原 DeepGEMM swizzle/硬件调度的干扰。

### 8.5 15% 目标要求接近理论极限的真正跨 task overlap

以 `334.683 us` 为起点，目标是 `284.481 us`，需要减少约 `50.2 us`。原 trace 中
epilogue 只占 `16.27%`。因此即使把整个 epilogue 完全隐藏，理论上也只是刚刚覆盖
`15%` 目标，尚未扣除 persistent loop、同步、首尾 bubble 和资源压力。

真正的跨 task overlap 还要求 current output 与 next input 使用不重叠的 LDS 区域。
CDNA5 文档说明一个 WGP 的 LDS 上限为 `320 KiB`，并且 LDS 与 WGP$ 共享 `384 KiB`
物理空间。当前 4-stage input ring 已占 `272 KiB`；再放完整约 `136 KiB` output tile
会超过 LDS 上限。

理论上可考虑把 input ring 降为 3 stages（约 `204 KiB`），再增加约 `85 KiB` 的
half-output arena，总量约 `289 KiB`，并在当前 task epilogue 期间预取下一 task。
但这已经不是低风险 scheduling 修改，而是同时重写 ring-buffer、output layout、TDM
排序和寄存器 lifetime。既有实验又表明 `b3`、额外 prefetch 和 output/TDM 调整均容易
增加 wait 或破坏正确性；在现有证据下无法合理预期它能达到接近理论上限的 `15%`。

## 9. 最终结论与清理状态

本轮没有找到优于 `apre_wpt2_mg4_fc28_ostore2p_ow2` 的 persistent task pipeline
版本，因此：

- 没有把任何 persistent case 集成进 `my_code/reproduce_compare.sh`；
- 已从正式 kernel 和 launcher 中删除全部 persistent selector、loop 和 grid 修改；
- 已恢复 a07-3 上三个任务相关文件，使 SHA256 与本地当前分支一致；
- 保留的正式 winner 仍为 `apre_wpt2_mg4_fc28_ostore2p_ow2`。

本轮停止继续尝试 persistent task pipeline。进一步推进必须先解决“低于 106 normal
SGPR、独立 half-output LDS arena、next-input/current-output 真正重叠”三个条件；否则
继续调整 worker 数、swizzle 或普通 scheduling knob 没有数据依据，也无法接近 `15%`
目标。

## 10. SGPR overflow 专项优化（2026-09-21）

用户要求继续降低 persistent kernel 的 SGPR 使用量，目标是消除此前：

```text
107 SGPR / 876 VGPR / 8~14 SGPR spills
```

本节所有资源数据均来自 a07-3 上 `COMPILE_ONLY=1 + FLYDSL_DUMP_IR=1` 生成的
`21_final_isa.s`，性能数据仍来自 GPU 空闲时的正式 e2e profiler。

### 10.1 编译选项不能解决 overflow

先测试了不改变 kernel 结构的编译选项：

| 尝试 | next-free SGPR | VGPR | SGPR spills | 结论 |
|---|---:|---:|---:|---|
| kernarg preload count `32` | `105` | `876` | `13` | 起点 |
| kernarg preload count `16` | `105` | `876` | `13` | 无变化 |
| kernarg preload count `8` | `105` | `876` | `13` | 无变化 |
| kernarg preload count `0` | `105` | `876` | `13` | 无变化 |
| 完全关闭 kernarg preload | `105` | `876` | `13` | 无变化 |
| `lsr-drop-solution=1` | `105` | `876` | `13` | 无变化 |
| 关闭 post-misched | `105` | `876` | `13` | 无变化 |

所以 overflow 不是 kernarg preload 数量造成的，而是 persistent task body 内部的
descriptor/address live range。

### 10.2 缩短 descriptor lifetime

把 A/B/ScaleA/ScaleB job 的 global offset 和 OOB 计算从 task prologue 延迟到实际
`emit()` 分支，避免四套 descriptor state 同时存活：

```text
SGPR spills: 13 -> 12
```

只减少了一个 spill。ISA 中剩余 spill 基本都是 64-bit global pointer/address pair。

随后尝试：

- 在 output phase 重新计算 `c_outer_off/c_inner_off/c_stride`；
- 在两次 TDM job issue 之间插入 scheduling boundary；
- 用 inline assembly 在 descriptor 使用点重新物化 offset；
- 从 kernarg segment 重新加载 global pointers；
- 显式把 kernel pointer 打包到 VGPR lanes。

结果如下：

| 尝试 | SGPR spills | 结论 |
|---|---:|---|
| deferred jobs | `12` | 小幅改善 |
| output offset 重算 | `12` | 无变化 |
| serialized job issue | `6` | 有改善，但未清零 |
| localize all job offsets | `6` | 比最优 direct mapping 更差 |
| localize M index | `6` | 无进一步收益 |
| kernarg pointer 普通 reload | `30` | LLVM 延长新地址状态，明显恶化 |
| kernarg pointer inline-asm reload | `32` | 明显恶化 |
| pointer 显式 VGPR packing | `30` | 明显恶化 |

### 10.3 降低 task state 的几何实验

用 1-D/2-D direct mapping 去掉 synthetic block id、runtime `n_units` 除法和部分 loop
状态，可以把 WPT2 的 spill 降到 `4`，但 direct task argument 路径在 random 输入上产生
NaN，不能保留。

其他几何结果：

| 几何/路径 | SGPR | VGPR | SGPR spills | 正确性 |
|---|---:|---:|---:|:---:|
| `tile_m=128` runtime persistent | `105` next-free | `513` | `21` | 未进入性能验证 |
| `m_warp=4, n_warp=2` | `105` next-free | `471` | `17` | 未进入性能验证 |
| WPT1，1-D persistent | `105` next-free | `878` | `6` | random 失败 |
| WPT1，direct mapping | `105` next-free | `876` | `0` | random 失败 |
| WPT2 + direct-global scales | `92` next-free | `882` | `0` | `rel_l2≈0.529`，失败 |

WPT1 的 zero-spill 版本即使加入 cluster-wide task-boundary barrier 仍无法通过 random，
说明它不满足当前 cluster TDM 的 persistent generation 协议。direct-global scales 的
索引/载入语义也不能直接替代当前 GEMM2 ScaleA/ScaleB TDM 路径。

### 10.4 功能等价的 zero-spill 版本：`unroll2_nodefer`

最终找到的正确方案是取消 runtime task-loop backedge，让一个 workgroup 固定处理两个
相邻 M tasks，并使用 `range_constexpr(2)` 展开：

```text
worker task 0 -> original binary search + original GEMM2 body
WG barrier
worker task 1 -> original binary search + original GEMM2 body
```

launch grid 使用 runtime capacity：

```text
ceil(m_tiles / 2) * n_tiles
```

因此没有缩小支持范围；末尾只有一个有效 task 时第二个 task 由动态 valid-M 条件跳过。
两个 task 均保留原 binary search、descriptor、WMMA 和 BF16 output 路径。

去掉对展开版本无用的 deferred-job 逻辑后，资源为：

```text
next-free SGPR = 68
VGPR           = 806
SGPR spills    = 0
VGPR spills    = 0
LDS            = 278,528 B
```

相对原 runtime persistent 的：

```text
next-free SGPR = 105
VGPR           = 876
SGPR spills    = 13
```

SGPR overflow 已完全消除，VGPR 也下降 `70`。

non-balanced random 验证：

```text
pass                     = True
logits_diff              = 3.48778e-06
rel_l2                   = 0.00264113
MoE output hash128       = 10ef188b89c427fde6c8b322b5fd4133
ref output hash128       = d043e1891d95c1af3da5a30e37b9042a
```

### 10.5 性能结果

GPU 空闲时，同轮三次测试：

```text
run directory:
/data/yanguahe/code/wk_sp1/aiter/my_code/gemm1_cycle_105pct_20260909/runs/
  heliosr-1b114-a07-3_20260921T100733Z_gemm2_e2e-const0
```

| case | GEMM2 samples (us) | median (us) | 相对 winner | MoE e2e samples (us) | median (us) | 相对 winner |
|---|---|---:|---:|---|---:|---:|
| winner | `335.470, 334.251, 335.620` | `335.470` | `0.00%` | `1246.25, 1248.39, 1247.34` | `1247.34` | `0.00%` |
| `unroll2_nodefer` | `365.361, 366.664, 364.041` | `365.361` | `-8.91%` | `1275.96, 1270.52, 1275.57` | `1275.57` | `-2.26%` |

把 task-boundary cluster barrier 去掉、只保留 LDS 所需的 workgroup barrier 后，GEMM2
仍约为 `364～365 us`，说明主要回退并非 cluster barrier。

zero-spill 版本的 ISA 约 `6,843` 行，而单 task kernel 约为其一半。为消除 runtime
backedge，编译器复制了完整 task body；由此增加的 instruction footprint、重复 prologue/
epilogue 和每两 task 一次的 LDS reuse barrier，超过了消除 SGPR spill 的收益。

### 10.6 最终处理

本轮证明了 SGPR overflow 可以在功能等价条件下消除，但当前 zero-spill 实现性能回退，
所以不满足保留门槛：

- `unroll2_nodefer` 未集成到正式 `my_code/reproduce_compare.sh`；
- 所有 preload、reload、packing、direct mapping、WPT1、direct-scales、wave-grid、tile-M
  和 unroll 实验 selector 均已从正式源码删除；
- 当前保留版本仍为 `apre_wpt2_mg4_fc28_ostore2p_ow2`。

若后续继续，需要在单份 ISA hotloop 内显式控制 scalar register lifetime，例如手写 ISA
或支持 noinline device-body/call-preserved ABI 的更低层实现。当前 FlyDSL runtime loop
会产生 spill，而编译期展开虽然清零 spill，却付出了过大的 instruction footprint；继续
微调 preload 或 scheduling knob 已没有依据。
