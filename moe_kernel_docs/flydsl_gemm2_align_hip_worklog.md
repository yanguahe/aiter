# FlyDSL gemm2 (mxfp4 a4w4, BM=128/32) 对齐 HIP 工作记录

> 目标：让 FlyDSL 版 `gemm2_kernel_0` 在 gfx950 (MI35x) 上的性能对齐 HIP 参考
> kernel `aiter::mxfp4_moe::gemm2::kernel<655360, 385, 512, 7168, 9, {32,128}, ...>`。
>
> - **被优化 kernel（target）**：`aiter/ops/flydsl/kernels/mxfp4_a4w4_gemm.py` 里的
>   `compile_mxfp4_gemm2_a4w4` / `launch_gemm2`（Python 构建 MLIR → JIT 成 gfx950）。
> - **参考 kernel（reference）**：HIP `gemm_a4w4/gemm2_a4w4.cuh` 的 `run_one` /
>   `launch_nonatomic` / `launch_nonatomic_mxfp4`，epilogue `mxfp4_epilogs.hpp`。
> - **基准配置**：bench `bench_up_moe_v1.py`（Kimi-K2.5 TP=4：NE=385, H=7168,
>   K=D_INTER=512, TOPK=9），`AITER_LOG_MORE=1` + `analyze_moe_kernel_log.py` 取
>   gemm2 独立 kernel 时间。`--benchmarks mx msfg`（mx=纯 HIP，msfg=HIP sort/gemm1
>   + FlyDSL gemm2）。容器 `hyg_fyd2`，`HIP_VISIBLE_DEVICES=0`。
> - **每 M 的 gemm2 模式**（== mx_fn）：M≤128→BM16 atomic；M=256→BM32 atomic；
>   M=4096→BM128 bf16flat（nonatomic + scatter_reduce）；M≥8192→BM128 mxfp4out
>   （nonatomic + scatter_reduce_q）。**大 M（BM=128）是差距大头**。
> - **关注点**：本次聚焦 BM=128（M=4096/8192/16384，原始 1.62–1.70x 慢）与 BM=32
>   （M=256）。BM=16 路径（M≤128）此前已对齐，本次数值不变。

---

## 1. 性能进展（gemm2 独立 kernel 时间，fly/mx = FlyDSL/HIP，越低越好）

| # | 改动 | M=256(BM32) | M=4096 | M=8192 | M=16384 | cos | 证据 |
|---|---|---|---|---|---|---|---|
| 0 | 基线（serial 单 slot reg→ds_write A-load） | 1.27x | 1.62x | 1.71x | 1.70x | 0.999 | `msfg_vs_mx.v2.log` |
| 1 | **direct-LDS 2-slot A + B 全部前置**（BM32/128） | 1.13x | 1.38x | 1.44x | 1.47x | 0.999 | ATT |
| 2 | **B-load nt→cached**（BM=128 nonatomic == HIP） | 1.13x | 1.35x | 1.38x | 1.36x | 1.000 | ATT + PMC |
| 3 | **1-D grid m-major 解码**（== HIP wu 解码） | 1.22x* | 1.32x | 1.35x | 1.31x | 1.000 | PMC L2 hit 28→57% |
| 4 | **AGPR 累加器 + 预读 A**（BM=128 == HIP kUseAGPR） | 1.22x | 1.33x | 1.35x | 1.32x | 1.000 | **性能中性**;VGPR 257→148 |
| 5 | **per-K 预读 A + 2 blocks/CU**（gfx950 160KB LDS，见 §2.5） | 1.22x | **1.31x** | **0.94x** | **0.91x** | 1.000 | mxfp4out **反超 HIP**;占用率 1→2 |

> **里程碑（fix5）**：gfx950 LDS = **160KB**（非此前误判的 ~128KB 上限）。把 BM=128 的 LDS 砍到
> ≤80KB → 2 blocks/CU，藏住访存/epilogue 延迟。mxfp4out（M=8192/16384，最大差距）**反超 HIP**
> （0.91–0.94x，HIP 被 `__launch_bounds__(256,1)` 钉死在 1 block/CU）。e2e total M=8192 0.98x、
> M=16384 0.96x。

\* BM=32（M=256）run-to-run 抖动较大（atomic 累加 + 小 grid 受 epilogue 限制），三步累计
1.27x→1.22x。BM=128（大头）：**1.62/1.71/1.70x → 1.32/1.35/1.31x**。BM=16（M≤128）路径
未改、数值不变（cos 1.0000）。e2e total（M≥8192）从 1.29–1.31x 降到 1.14–1.15x。

> 因果主线（按 `flydsl-align-reference-kernel.mdc`）：性能差 ← ISA 差 ← 前端逻辑差。
> 三步都是 (a) 前端逻辑差异（A-load 结构 / cache hint / grid 解码），不是后端设置或
> lower 差异。先 profiling 定位、再前端修复、每步实测。

---

## 2. 有效改动（逐条，含根因与证据）

### 2.1 direct-LDS 2-slot A-load + B 前置（基线 → fix1）

**问题（ATT 实测，M=4096，att_target_cu=1，两版监控 tile 数相当）**：FlyDSL gemm2
BM≠16 走的是 serial 单 slot 路径：`buffer_load A → reg → ds_write → s_barrier →
load_b → mfma → s_barrier → load_b → mfma`，共 **3 个 barrier、B 在 barrier 之后才发**。
ATT cycle 预算：

| 类别 | FlyDSL lat | HIP lat |
|---|---|---|
| **WAITCNT** | **9.17M (53%)** | **0.87M (8%)** |
| MFMA | 2.17M | 1.80M |
| BARRIER | 1.10M | 2.69M |
| VMEM_LD | 1.28M | 2.43M |
| **TOTAL** | **17.2M** | **10.9M** |

FlyDSL 总 issue cycle ≈ HIP 的 1.57x，与 kernel 时间差 ~1.59x 吻合；**整段差距全在
WAITCNT stall**（`s_waitcnt vmcnt(1) lgkmcnt(7)` 单 tile 2607 cyc、`vmcnt(3) lgkmcnt(1)`
2332 cyc）。MFMA 计数两版都是 128 → 纯 stall、非指令数差。

**改动**：把已为 BM=16 实现的「direct-to-LDS（`raw_ptr_buffer_load_lds`）+ 2-slot 双缓冲 +
B/scale 全部在 drain barrier 之前发射」结构推广到 BM=32/128（== HIP `run_one` /
`issue_a_load_lds` nonatomic）：
- `_g2_a_directlds = BM in (16,32,128)`，2 slot，LDS 行宽 `BK/2=128` 字节（contiguous DMA
  写 + swizzle 进全局读）。
- `load_a_directlds(slot,kt)`：BM=16 仅 wave0,1；BM≥32 全 4 wave、`kSubBlocks=BM/32` 个 8 行
  chunk。swizzle mask = `lds_swizzle_mask<128>(actual_row) = (actual_row&14)<<3`
  （**关键**：写用 actual_row，读用 lane%16，二者 bit1-3 相同 → swizzle 抵消）。
- K-loop：`load_a_directlds(0,0); load_a_directlds(1,1); sched_barrier(0); 全部 B+scale;
  s_barrier; mfma(slot0); mfma(slot1)`。

**效果**：WAITCNT 9.17M→1.97M（53%→14%），TOTAL 17.2M→13.8M。gemm2 BM=128
1.62–1.70x → 1.38–1.47x。register 不变（A 不再过寄存器，LDS union 总量不变 = 128KB）。

### 2.2 B(w2) global-load nt → cached（fix1 → fix2，仅 BM=128）

**问题**：fix1 后头号瓶颈变成 VMEM_LD（ATT 4.52M / 43% stall）。FlyDSL B-load 是
`buffer_load_dwordx4 ... offen nt`（非临时），HIP nonatomic 是 `... offen`（cached）。
HIP `issue_b_load_j`：`kBQ_AUX = (kAtomic && kUseNT) ? 2 : 0` → nonatomic(BM128) = 0(cached)。
BM=128 大 grid 下同一 expert 的 w2 被多个 m-block 复用，nt 会把可复用权重逐出 L2。
（catalog 第 1 条：reused 数据上的 nt → 额外 HBM + 更高 per-request L2 延迟。）

**改动**：`_b_cache = 0 if (mxfp4out or bf16flat) else b_nt`，`load_b_tile` 用 `_b_cache`。
BM=16/32 atomic _NT 仍保留 nt（== HIP kUseNT）。

**效果**：gemm2 BM=128 1.38–1.47x → 1.35–1.38x。

### 2.3 1-D grid + m-major pid 解码（fix2 → fix3，L2 局部性）

**问题（PMC 实测，M=8192）**：fix2 后 FlyDSL L2 命中率仅 **28.4%**（HIT 70.9M / MISS
178.8M），HIP **52.5%**（HIT 110M / MISS 99.7M）—— FlyDSL L2 miss 多 ~80%，memory-bound
的 gemm2 更多时间等 HBM。根因：旧 2-D grid `(gx=total_m_blocks, gy=28)` 把 **m_block 放在
fast(x) 维**，硬件按 flat-id 条带分发时每个 XCD 的 block 流扫过**所有 expert** 的同一 n →
w2 在 L2 里互相打架。HIP（kXcdSwizzle=0）的 `wu` 是 **m-major**（`m=wu/28, n=wu%28`），每个
XCD 只停在少数 expert → w2 常驻 L2。

**改动**：launch 改 1-D grid `gx*num_n_blocks`，kernel 内
`m_block = pid/num_n_blocks, n_block = pid%num_n_blocks`（== HIP wu 解码）。覆盖的
(m,n) 对完全不变（correctness-neutral），只改 block 的派发序 / 落 CU。

**效果**：L2 命中率 28.4% → **56.8%**（已超 HIP 52.5%）。gemm2 BM=128 1.35–1.38x → 1.31–1.35x；
BM=32(M=256) 1.27x→1.22x。

### 2.4 AGPR 累加器 + 预读 A（fix3 → fix4，仅 BM=128）—— 性能中性，保留作 ISA 对齐 + VGPR 鲁棒性

**问题（trace_segment_cycles.py 实测，M=4096）**：MFMA-span 对比 HIP 191 指令/4240 周期（纯 AGPR
背靠背 MFMA、A 预读入寄存器、epilogue 独立）；FLY 553 指令/6326 周期 —— FLY 把 16 个 A 的
`ds_read`（带 lgkmcnt 级联）+ 128 个 epilogue cvt/store **全交织进 MFMA 区**（+362 指令）。根因：
FLY 用 VGPR 累加器（arch VGPR 257，顶到上限），HIP nonatomic 用 **AGPR 累加器**（`kUseAGPR`）。

**改动**：照搬 gemm1 BM=128 已有的 `_mfma_agpr`（`llvm.inline_asm` `"+a"` 把累加器钉进 AGPR，
A/B 用 v4i32 而非 rocdl 的 v8i32 → A/B VGPR 减半）。`mfma_ktile` BM=128 改为：先一次性预读全部 A
（`a_all[mi][k]`），再按 n-output J、k-outer/mi-inner 发 64 个 AGPR MFMA（连续 MFMA 落在不同
accm → 无 hazard s_nop）。BM=16/32 保持 VGPR 路径。

**效果（实测）**：**性能中性**（BM=128 1.31–1.35x，与 fix3 无差别）。原因（规则 Stage 3.4）：本
kernel 是 1 block/CU（128KB LDS 限）的**访存延迟瓶颈**，纯计算调度优化对它中性。但：
- arch VGPR **257→148**（accm 128 入 AGPR，A/B v4i32 减半），脱离 257 spill 边缘。
- ISA 与 HIP `kUseAGPR` 对齐（128 `v_accvgpr_read`、MFMA 背靠背），完成规则第 2 层「编译产物等价」。

故保留（中性、非回退；补强 VGPR 鲁棒性 + ISA 对齐），而非回退。

### 2.5 per-K 预读 A + 2 blocks/CU（fix4 → fix5，BM=128）—— 反超 HIP

**关键硬件信息**：gfx950 LDS = **160KB/CU**（rocminfo `Max Waves Per CU=32`、agent_info
`Lds_Size_In_Kb=160`）。此前误判 occupancy 被 ~128KB 锁死；实测两版（fix4 / HIP）都是
**1 block/CU**（`occupancy_balance.py`：max slots/SIMD=1，"no latency hiding within a SIMD"）。
HIP 被 `__launch_bounds__(256,1)` 钉死 1 block；FlyDSL 无此限，只要 LDS ≤ 80KB 即可 2 blocks/CU。

**A/B 隔离实测（M=4096，per-K 结构）**：1 block/CU = **1.36x**，2 blocks/CU = **1.30x** —— 纯
occupancy 收益 **~0.06x**（非边际）。2 blocks 让第 2 个 wave 在本 wave 等 w2 加载 / 写 epilogue
时顶上，藏住延迟（本 kernel 的瓶颈）。

**改动（三步协同）**：
1. **per-K 预读 A**（替换 fix4 的 all-K 预读）：每个 K-tile 只预读 8 个 m-chunk 的 A（~32 VGPR，
   而非 all-K 的 64），MFMA 仍背靠背 + 无 hazard（8 个连续 mi 落在不同 accm）。arch VGPR
   **148→96**、总 **276→224**（≤256，2 blocks/CU 的前提）。
2. **bf16flat LDS 128KB→32KB**：bf16flat 不用 lds_acc（直写 flat_out），allocator 只分 A 暂存
   LDS → M=4096 2 blocks/CU。
3. **mxfp4out 2-N-half cshuffle**：把 cshuffle+quant 拆成两个 N-half（cols [0,128)/[128,256)），
   过 `lds_acc[BM*128]`（**64KB**，非 full BM*BN 128KB）。每 half 由拥有该列的 2 个 wave 写、全
   256 线程 quant 其 4 个 per-32 组。accm（AGPR）跨两 half 常驻 → M=8192/16384 2 blocks/CU。

**效果（实测）**：
- M=8192 mxfp4out **1.35x → 0.94x**（反超 HIP），M=16384 **1.32x → 0.91x**（反超）。
- M=4096 bf16flat 1.33x → **1.31x**（轻 epilogue，2-block 收益小）。
- 占用率 `occupancy_balance.py` 确认：M=8192 mxfp4out = **2 blocks/CU**。
- cos 0.9998–1.0000；BM=16/32 atomic 路径未改、未回退。

> 为何 mxfp4out 反超而 bf16flat 仅小幅：mxfp4out 有重 quant epilogue（cvt+store+per-32 amax），
> 2-block 把这段延迟藏在另一 wave 后，收益巨大；bf16flat 直写 bf16（轻），可藏的少。HIP 两种模式
> 都钉死 1 block/CU，故 FlyDSL 在重 epilogue 的 mxfp4out 上反超。

---

## 3. 残余差距分析（fix5 后）

- **M=8192/16384（mxfp4out，最大差距）：已反超 HIP**（0.91–0.94x）。目标达成。
- **M=4096（bf16flat）：1.31x —— 已到本模式实际上限**（见 §3.1）。
- **小 M（BM=16/32 atomic，M≤256）：1.05–1.47x —— 受 FLY VGPR 比 HIP 略高所限**（见 §3.2）。

### 3.1 M=4096 bf16flat 已到上限（attack #1，无改动）

ATT 实测（M=4096，2 blocks/CU，load balance imbalance 0.02 良好）：cycle 预算被 **VMEM_ST
25.8%（stall 29.5%）+ VMEM_LD 24.7%（stall 30.2%）= 50%** 主导 —— 是 **memory-bound**。
VMEM_ST = 128 个 `global_store_short`（bf16 直写 flat_out，2 字节/个）。这些输出是 MFMA lane
布局散落的（行 stride=N_OUT、列 stride=16），**无法向量化**（v 跨行、j 跨 16 列均不连续）。
HIP `apply_bf16_flat_epilog_bm128` 是**同样**的 128 个散落 short store。若改成 cshuffle→行连续
再向量化，需 lds_acc（128KB）→ 退回 1 block/CU，得不偿失。故 M=4096 bf16flat 1.31x 已是该
（直写、2-block）设计的实际上限。

### 3.2 小 M（BM=16/32 atomic）受 VGPR 所限（attack #2，AGPR 尝试中性已回退）

occupancy 实测（M=64）：**FLY 4 blocks/CU，HIP 5 blocks/CU**。FLY BM=16 VGPR=100、HIP=85；
FLY 多 ~15 VGPR → 少 1 个并发 block → 慢（M=64 1.48x）。
- **尝试：给 BM=16/32 也上 AGPR 累加器**（同 BM=128）。结果：BM=16 accm 仅 16 VGPR，AGPR/v4i32
  省得极少（102→100），occupancy 仍 4 blocks/CU；full-bench 性能**中性**（M=256 1.22x 不变，
  M=16/32/64/128 在噪声内）。**已回退**（不留无收益改动）。
- 根因是 FLY 的总 VGPR（100）比 HIP 手调的 85 高 ~15，需削减 A/B/scale/地址寄存器才能上第 5 个
  block —— 收益小、风险高，未做。小 M 的 gemm2 绝对耗时小（M=64 ≈95µs，差 ~30µs），且 e2e 被
  sort/quant/gemm1 主导。

历史残余分析（fix1–4，已被 fix5 的 occupancy 推翻其"1 block/CU 固有"结论）：

**已与 HIP 对齐的全部杠杆**：load 结构（direct-LDS 2-slot）、B cache（cached）、L2/grid 局部性
（m-major 解码 57%>52%）、MFMA 调度（AGPR 背靠背）。残余仍 ~1.3x，根因是**本 kernel 形态固有的
访存延迟瓶颈，HIP 自己也撞同一堵墙**：

- **不是 HBM 带宽**：FLY ~25% / HIP ~31% 的 8TB/s 峰值，都远未打满。
- **不是 L2 traffic**：FLY L2 命中 56.8% ≥ HIP 52.5%。
- **不是 occupancy**：两版都 1 block/CU（128KB LDS union 限），4 waves。
- **不是计算调度**：AGPR 对齐后（fix4）性能中性，证实残余非 MFMA/hazard。
- **是 1 block/CU 下的访存延迟隐藏**：最大单点 stall 是 MFMA 前等 w2 的 `s_waitcnt vmcnt`
  （cold-L2 ATT ~1386–2628 cyc/tile）；K=512 仅 2 个 K-tile、计算量小，4 waves 不足以藏住 w2
  加载延迟。HIP 靠手工调度把这段藏得略好，是其剩余 ~1.3x 的来源。

**已评估否决 / 未做**：
- **persistent grid**：实测否决 —— HBM 未打满，persistent 的主要收益（打满 HBM + 摊薄 launch）
  不成立，预计中性。
- **降 LDS 提 occupancy**：bf16flat（M=4096）不需要 lds_acc，把 LDS 128KB→32KB 可上 2–3
  block/CU 改善延迟隐藏 —— 但仅惠及 M=4096 且偏离 HIP（HIP 也用 128KB union），未做。

**ATT 代表性注记**：`att_target_cu=1` 对本 memory-bound kernel 的 L2 复用不具代表性（单 CU L2
偏冷），fix2/fix3 的真实收益体现在整网 kernel 时间 + PMC，不体现在单 CU ATT 总 cycle。故以
**kernel 时间 + PMC** 为准，ATT 仅用于定位指令级 stall pattern（fix1/fix4）。

---

## 4. 复现命令

```bash
# 容器 hyg_fyd2，工作目录 .../aiter
# 1) 清 FlyDSL 缓存（改完必清）
docker exec hyg_fyd2 bash -lc 'rm -rf ~/.flydsl/cache/*'
# 2) 全量 bench + 分析（gemm2 独立时间）
docker exec hyg_fyd2 bash -lc 'cd .../aiter && AITER_LOG_MORE=1 HIP_VISIBLE_DEVICES=0 \
  python3 bench_up_moe_v1.py --M-list 4,8,16,32,64,128,256,4096,8192,16384 \
  --iters 100 --hash --benchmarks mx msfg > t.log 2>&1 && \
  python3 analyze_moe_kernel_log.py t.log | tee msfg_vs_mx.v3.log'
# 3) ATT（指令级 stall 定位，input_fly_g2.yaml / input_hip_g2.yaml；regex gemm2_kernel_0 / mxfp4_moe::gemm2::kernel）
# 4) PMC L2（pmc_fly_g2.yaml / pmc_hip_g2.yaml；TCC_HIT/TCC_MISS）
```

验收：`MaxErr` 等价 `cos ≥ 0.999`（全 M PASS：0.9994–1.0000）。
