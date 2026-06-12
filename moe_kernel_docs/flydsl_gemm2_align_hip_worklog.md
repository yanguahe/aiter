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
| 6 | **bf16flat persistent / grid-stride**（== HIP kPersistent，仅 bf16flat；见 §2.6） | 1.22x | **1.02x** | 0.94x | 0.91x | 1.000 | commit `8c1dbcd5`;standalone gemm2 432.6→357.8µs（-17%） |

> **里程碑（fix5）**：gfx950 LDS = **160KB**（非此前误判的 ~128KB 上限）。把 BM=128 的 LDS 砍到
> ≤80KB → 2 blocks/CU，藏住访存/epilogue 延迟。mxfp4out（M=8192/16384，最大差距）**反超 HIP**
> （0.91–0.94x，HIP 被 `__launch_bounds__(256,1)` 钉死在 1 block/CU）。e2e total M=8192 0.98x、
> M=16384 0.96x。

\* BM=32（M=256）run-to-run 抖动较大（atomic 累加 + 小 grid 受 epilogue 限制），三步累计
1.27x→1.22x。BM=128（大头）：**1.62/1.71/1.70x → 1.32/1.35/1.31x**。BM=16（M≤128）路径
未改、数值不变（cos 1.0000）。e2e total（M≥8192）从 1.29–1.31x 降到 1.14–1.15x。

> **里程碑（fix6，commit `8c1dbcd5`）**：把 bf16flat（M=4096）从「每 block 跑 1 tile」改成
> **persistent / grid-stride**（== HIP `kPersistent`）。固定 grid `2*NUM_CU=512` 个 block，每个
> block 在 `for wu in [block_id, total_work) step 512` 里循环处理多个 tile，把**每 tile 的
> prologue（A→LDS DMA + barrier + 地址 setup，§2.5 实测约占一个 tile 的 ~43%）摊到多个 tile** 上，
> 而非每个 1-tile block 各付一次。**M=4096 bf16flat 1.31x → 1.02x**（standalone gemm2
> 432.6→357.8µs，-17%）。**这推翻了 §3 早先「persistent 实测否决」的结论**（当时误判收益来自
> 「打满 HBM」；真实收益是 prologue 摊销，见 §2.6 / §3.1）。mxfp4out（M=8192/16384）**故意不做
> persistent**（loop-carried live range +~20 arch VGPR → occ 2→1 反而回退），保持 §2.5 的 0.91–0.94x。
>
> > 注（本 session 2026-06-11 全量复测）：清缓存 + 全量重建后 `bench_up_moe_v1.py` 重测 M=4096
> > gemm2 fly/mx ≈ **1.08x**（fly 348.2µs / mx 323.8µs）。与 commit 时的 1.02x 的差异为
> > run-to-run / 测量口径（standalone vs 全量管线里的 mx 基线）波动；当前继续攻关的目标就是这 ~8%。

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

### 2.6 bf16flat persistent / grid-stride（fix5 → fix6，仅 bf16flat）—— M=4096 1.31x→1.02x

**问题（推翻 §3 早先结论）**：fix5 后 M=4096 bf16flat 卡在 1.31x，§3.1 曾判定「直写 + 2-block 已到
上限」、§3 曾「实测否决 persistent」。复查发现早先否决的依据错了：当时以为 persistent 的收益来自
「打满 HBM + 摊薄 launch」，而本 kernel 的瓶颈不是 HBM 带宽。**persistent 的真实收益是摊销每个
tile 的 prologue**——FlyDSL 旧 dispatch 是「1 block 跑 1 tile」（`run_one(pid/nnb, pid%nnb)`），
每个 tile 都要重付一次 A→LDS direct-DMA + drain barrier + 地址 setup（§2.5 实测约占一个 tile 的
~43%），且无法被前一 tile 的计算隐藏；HIP nonatomic 是 `kPersistent`，每个 block 用 grid-stride
循环吃多个 tile，prologue 被摊薄。

**改动（== HIP `gemm2_a4w4.cuh:421-437` kPersistent，仅 bf16flat）**：
- launch：bf16flat 发**固定** grid `GRID_PERSISTENT = 2*NUM_CU = 512`（mxfp4out/atomic 仍用
  `total_m_blocks*num_n_blocks` 的 1-tile-per-block grid）。
- kernel：bf16flat 用 `scf.ForOp(block_id, total_work, GRID_PERSISTENT)` 循环，
  `total_work = (num_valid/BM)*num_n_blocks`（== HIP `total_m_blocks=cumsum/BM` × nnb，只跑有效
  tile），循环体 `run_one(wu/nnb, wu%nnb)`。
- **关键 occupancy 保护**：把 wave/lane 分解（`tx`、`wave`、`lane`、`lane_div_16/mod_16`）+ 每个
  tile 的地址 setup **全部移进 `run_one` 内**（per-tile 重算），使它们**不跨 persistent 循环
  loop-carried**——否则 loop-carried live range 会涨 ~20 arch VGPR，把 bf16flat 从 2 blocks/CU
  压回 1，抵消收益。HIP 同样把这些放在 `run_one` lambda 内。
- HIP grid 是 `min(total_work, NUM_CU=256)`（被 `__launch_bounds__(256,1)` 钉死 1 block/CU）；
  FlyDSL 无此 cap，bf16flat 仅 32KB LDS、≤256 VGPR → 发 512 个 block 跑 2 blocks/CU，把
  persistent 的 prologue 摊销叠在 occ=2 的延迟隐藏之上。

**为何 mxfp4out 不做 persistent**：mxfp4out 走 64KB lds_acc 的 occ=2 非 persistent 路径已反超
HIP（0.91–0.97x）；改 persistent 会引入 loop-carried live range（实测 arch VGPR 248→268）把 occ
2→1，反而回退。故 persistent 只用于 bf16flat（其 prologue 占比大、epilogue 轻，从摊销获益最多）。

**效果（实测）**：M=4096 bf16flat **1.31x → 1.02x**（commit 时；standalone gemm2 432.6→357.8µs，
-17%）。mxfp4out / atomic 未改、未回退。cos 1.0000。

---

## 3. 残余差距分析（fix6 后）

- **M=8192/16384（mxfp4out，最大差距）：已反超 HIP**（0.91–0.94x）。目标达成。
- **M=4096（bf16flat）：fix6 persistent 后 1.31x → 1.02x**（commit 时）；本 session 全量复测
  **1.08x**（348.2 / 323.8µs）。**§3.1 早先「已到上限」的结论已被 fix6 推翻**——见 §3.1（已标注 superseded）。
  当前正继续攻关这残余 ~8%。
- **小 M（BM=16/32 atomic，M≤256）：1.05–1.47x —— 受 FLY VGPR 比 HIP 略高所限**（见 §3.2）。

### 3.1 ~~M=4096 bf16flat 已到上限~~（SUPERSEDED by fix6 §2.6：1.31x→1.02x）

> **⚠ 本节结论已被 fix6（§2.6 persistent）推翻。** 下面是 fix5 时（非 persistent）的分析，保留作
> 历史记录。它正确指出了「epilogue 的 128 个散 short store 无法向量化」，但**错误地**把「1.31x」
> 当成了该模式的整体上限——遗漏了「每 tile prologue 无摊销」这个真正可攻的点。persistent 把
> prologue 摊到多 tile 后到了 1.02x（commit）/ 1.08x（本 session 复测）。当前残余 ~8% 的瓶颈需用
> **post-persistent 的新 ATT** 重新定位（fix5 时的下列 cycle 预算已不代表当前 dispatch 形态）。

ATT 实测（fix5，M=4096，2 blocks/CU，load balance imbalance 0.02 良好）：cycle 预算被 **VMEM_ST
25.8%（stall 29.5%）+ VMEM_LD 24.7%（stall 30.2%）= 50%** 主导 —— 是 **memory-bound**。
VMEM_ST = 128 个 `global_store_short`（bf16 直写 flat_out，2 字节/个）。这些输出是 MFMA lane
布局散落的（行 stride=N_OUT、列 stride=16），**无法向量化**（v 跨行、j 跨 16 列均不连续）。
HIP `apply_bf16_flat_epilog_bm128` 是**同样**的 128 个散落 short store。若改成 cshuffle→行连续
再向量化，需 lds_acc（128KB）→ 退回 1 block/CU，得不偿失。**（注：此「无法向量化」对 epilogue
本身仍成立；fix6 没动 epilogue，而是去摊 prologue。）**

### 3.1.1 M=4096 bf16flat post-persistent 攻关（2026-06-11，5 个尝试全部记录）

fix6（persistent）后 M=4096 bf16flat ≈ **1.06–1.08x**（fly ~344–348µs / mx ~324µs）。本次按
playbook 重新 profile（**post-persistent 新 ATT**，非 fix5 旧数据）并系统性试遍杠杆，**结论：occupancy
与跨 tile 预取都是死路,bf16flat M=4096 已到 FlyDSL 对 HIP 手工调度的实际上限。**

**post-persistent ATT（occ 1，FLY vs HIP，单 CU issue cycle）**：
| 类别 | FLY | HIP | 说明 |
|---|---|---|---|
| WAITCNT | 19.5%（vmcnt(5) 1316cyc/tile） | 8.0% | FLY 等 w2 更久 |
| VALU | 14.1%(410条) | 7.9%(240条) | 含 128 v_accvgpr_write(FLY 独有,见下) |
| BARRIER | 9.2%(2个) | 24.6%(3个) | HIP barrier 更重 |
两版「内存等待+同步」总量相当(FLY 3.26M vs HIP 3.51M),FLY 反而略少。差距是 HIP 单 block
把 w2 等待藏得更好(WAITCNT 8% vs 19.5%)。

**5 个尝试(全部 ISA+ATT 实测,均未达 ≤1.0x)**：
1. **AGPR init-zero**(== HIP `mfma_f4f4_agpr_init_zero`,C=字面量0/`=a` only)：去掉 128 个
   `v_accvgpr_write_b32`(ISA 实测 128→0)。**性能中性**(1.08→1.07,memory-bound 下 issue cycle
   本就藏在 vmcnt/barrier stall 后)。印证 Stage-3「memory-bound → 计算/调度优化中性」。**已回退**。
2. **readfirstlane B/scale 基址**(移植 gemm1 的 scalar soffset 模式)：在 gemm2 结构下 VGPR
   **264→268**(更糟),M=4096 **1.12x**(回退)。gemm1 模式不适配 gemm2(一次性发全 B,共享 voffset
   live range 反而拉长)。**已回退**。
3. **强制 `amdgpu-waves-per-eu="2,2"`**：逼到 occ 2,但把 128 AGPR 累加器挤出 AGPR(NumAgprs
   128→4)+ **spill 20**,**cos=0(算错)**,1.31x。128 AGPR 是硬底线,强制占用率行不通。**已回退**。
4. **B 逐 stage 加载 → 真实 occ 2**(减半 B live range:264→232 VGPR,spill 0)：cos 0.9999,
   **1.06x(本次最佳,但仅 1% 且脆弱)**。post-persistent occ-2 ATT 显示 **VMEM_ST 飙到 28%
   (5.99M,128 散 short store 的 write-port 争用)**——这正是 §2.6 注释说的「occ=2 epilogue
   write-port contention +149%/wave」。占用率翻倍的延迟隐藏被写争用抵消 → 净 1%。**已回退**。
5. **跨 tile w2 预取**(persistent loop 用 `scf.ForOp` iter_args 软件流水,prologue 预取首 tile,
   每轮发下个 tile 的 w2、用预取的当前 B 算、yield)：cos 0.9996,VGPR **328**(arch200+AGPR128,
   occ 1,spill 0),但 **1.16x(回退)**——携带 B(64 VGPR)+ 载下个的寄存器压力 + 循环边界值搬运
   开销 > 延迟隐藏收益,流水重叠未真正实现。**已回退**。

6. **逐指令复刻 HIP occ-1 调度 —— mfma cluster interleave(J-outer)**：把 mfma_ktile 从
   k_idx-outer 改成 HIP `issue_mfma_cluster` 的 **J-outer**(读两个 k-half 的 A 后,per-J cluster、
   sub-inner、k0-then-k1,同 SrcC 累加 0-nop),使 MFMA 消费 B 的顺序(b[J][0],b[J][1] 连续)= B
   加载顺序 → 后端能插细粒度 vmcnt。仅 bf16flat。结果:cos 1.0000,spill 0,VGPR **264→300**(A
   两 k-half 全持),**M=4096 仍 1.08x(353µs,略升,VGPR 增)**。**没用**。ISA 实测根因:第一个
   MFMA 前 `s_waitcnt vmcnt(11)` 等 w2,而 **B load 到 MFMA 之间只隔 ~16 个 ds_read(几百 cyc)
   ≪ w2 HBM 延迟 ~1300cyc** → occ-1 单 tile 内没有足够 work 藏 w2。唯一能把 w2 提前一整 tile 的
   是跨 tile 预取(尝试 5,寄存器代价失败)。per-stage barrier / 显式 waitcnt 改不了「w2 没到」
   这个事实。**已回退**。

**根因(为何到顶)**：(a) AGPR=128 固定 → occ 上限就是 2(3 blocks 需 ≤170 VGPR,光 AGPR 就 128);
(b) occ 2 被 epilogue 的 128 个不可向量化 short store 的 write-port 争用抵消;(c) 跨 tile 预取的寄存器
代价超过收益;(d) FLY 已对齐 HIP 所有结构杠杆(load/cache/L2/AGPR/persistent),残余是 HIP 手工
单 block 调度(per-stage barrier + mfma_cluster interleave 把 w2 等待藏得更好),FlyDSL 高层结构难
逐指令复刻。**当前保持 8c1dbcd5 基线(1.08x)**;上述 5 个尝试均已回退,代码干净。

### 3.2 小 M（BM=16/32 atomic）受 VGPR 所限（attack #2，AGPR 尝试中性已回退）

occupancy 实测（M=64）：**FLY 4 blocks/CU，HIP 5 blocks/CU**。FLY BM=16 VGPR=100、HIP=85；
FLY 多 ~15 VGPR → 少 1 个并发 block → 慢（M=64 1.48x）。
- **尝试：给 BM=16/32 也上 AGPR 累加器**（同 BM=128）。结果：BM=16 accm 仅 16 VGPR，AGPR/v4i32
  省得极少（102→100），occupancy 仍 4 blocks/CU；full-bench 性能**中性**（M=256 1.22x 不变，
  M=16/32/64/128 在噪声内）。**已回退**（不留无收益改动）。
- 根因是 FLY 的总 VGPR（100）比 HIP 手调的 85 高 ~15，需削减 A/B/scale/地址寄存器才能上第 5 个
  block —— 收益小、风险高，未做。小 M 的 gemm2 绝对耗时小（M=64 ≈95µs，差 ~30µs），且 e2e 被
  sort/quant/gemm1 主导。

#### 3.2.1 小 M post-revert 重新攻关（2026-06-11，2 个尝试失败）

按 playbook 重新实测当前基线(8c1dbcd5)的 BM=16/32 ISA + M=64 ATT/occupancy:
- **VGPR/occ(实测)**:FLY BM=16 = **102 VGPR / occ 4**;HIP BM=16 = 85 / occ 5。FLY BM=32 = 98 /
  occ 4;HIP BM=32 = 99 / occ 4(BM=32 占用率已相同,gap 在别处)。
- **achieved occupancy(occupancy_balance.py, M=64)**:FLY max slots/SIMD = **4**,HIP = **5**,grid
  够大(都 over-subscribed)。dur median FLY 218k vs HIP 147k(≈1.48x,与 kernel gap 吻合)。
- **cycle 预算(M=64)**:FLY TOTAL 12.3M vs HIP 9.8M;**FLY WAITCNT 44.8%(5.53M),其中 3 条
  `s_waitcnt vmcnt(0)` = 4.2M**,全在 atomic epilogue 等 `sorted_token_ids`/`sorted_weights` 的
  inline load(各 ~2000cyc HBM)。HIP epilogue **也是 inline load 同样的 token/weight**,但靠 occ 5
  跨 block 藏住。

**尝试(均失败,已回退)**:
1. **token/weight 预取到 K-loop 前**(~4 VGPR,与 GEMM 重叠):**无效**。atomic 非 persistent(1
   tile/block),BM=16 单 tile GEMM 太短(几百 cyc)≪ token/weight 的 ~2000cyc → tile 内来不及藏;
   HIP 是靠 occ 5 跨 block 藏,不是 in-tile 预取。且 VGPR 102→108(更糟),M=64 仍 1.47x。
2. **B 逐 stage(只 atomic)**:VGPR 102→**68 / occ 4→7**(spill 0),但 **全面回退**(M=64 1.49x、
   M=16/128/256 都更差)。occ 7 的跨 block 收益**补不回丢失的 B in-tile 预取**(b1 在 mfma0 后才载 →
   mfma1 等 b1)。证明小 M 需要 **B 双预取(藏 B 延迟)+ occ(藏 token/weight 延迟)二者兼得**。

**根因(为何到顶)**:需 **occ 5 且保留 B 双预取**,即 VGPR 102→≤96(砍 6)。但 B 双预取本身占
64 VGPR(编译器把 `_b0`/`_b1` 都保活;HIP 用 `b[stage][..]` 数组 + stage 循环让编译器轮转,85 VGPR
就放下了)。前端层面找不到「保住 B 预取 + 砍 6 VGPR」的干净办法(readfirstlane 已证会涨 VGPR)。这是
**寄存器分配效率差距**(HIP 85 vs FLY 102),与 M=4096 同属「FlyDSL 高层结构 vs HIP 手工 RA/调度」
的固有差。BM=32(M=256)占用率已等同 HIP(都 occ 4),其 1.20-1.26x gap 是另一类(调度,见 §3.2.2)。

#### 3.2.2 M=256(BM=32)指令 pattern 分析 + Pattern A 对齐(2026-06-11)

按 `flydsl-align-reference-kernel.mdc` Stage 3 对 M=256 做 FLY vs HIP cycle 预算。occ 两边都 4(相同),
gap 纯在指令 pattern。FLY−HIP 差值:**WAITCNT +1.42M**(主)、VALU +0.35M(`v_or` 多 20 条 vs HIP 的
融合 `v_mad_u64_u32`)、MFMA/LDS_ST/SALU 各 +0.15M;BARRIER FLY 反而少 0.5M。

- **Pattern A = atomic epilogue 的 token/weight `vmcnt(0)` 全 drain**(8 条,单 CU ATT 占 FLY-HIP 差的
  ~88%)。根因:FLY 在原子循环里 **inline 加载** sorted_token_ids/sorted_weights,vmcnt(0) 等满 ~2000cyc
  HBM。HIP 把这些 load **在 cshuffle 之前发出**(ISA 实证),与 cshuffle 重叠,原子循环 vmcnt 时已就绪。
- **修复(已落地,HIP 对齐)**:把 token/weight 的 buffer_load 移到 **cshuffle ds_write+barrier 之前**
  发出(== HIP),原子循环用预取值。**ISA 实证对齐**:FLY WAITCNT **7.86M→3.62M**(8 条 vmcnt(0) 消失),
  单 CU TOTAL 14.93M→13.47M ≈ HIP 13.27M。cos 全 1.0000,大 M 无回退。
- **但 wall-clock 只 1.22x→1.19x(小幅)**。**关键教训:`att_target_cu=1` 单 CU ATT 严重高估了 Pattern A
  的 wall-clock 权重**——三个度量(单CU-ATT 1.015x、occupancy.json dur 1.066x、真实 kernel 1.19x)互不
  一致,说明 M=256 真实 gap 主要在**跨 CU 内存吞吐 / L2**,不是单 block 的 waitcnt。需用 PMC
  (pmc_fly_g2.yaml / TCC_HIT/MISS)而非单 CU ATT 才能定位剩余 gap。Pattern A 对齐是真实小赢 + 正确对齐,
  但非 M=256 gap 的大头。

**当前状态**:Pattern A 对齐(token/weight pre-cshuffle 预取)已落地在工作树(未提交);其余小 M 尝试
(token/weight 预取-before-K-loop、B-per-stage)均已回退。bf16flat(§3.1.1)回到 8c1dbcd5 基线。

##### Pattern B 对齐(cshuffle ds-立即-offset)+ 实测中性

- **误诊澄清**:Pattern B 不是「该用 v_mad 却用了 v_or」。FLY 的输出地址**已经**用融合 `v_mad_u64_u32`
  (== HIP)。那些 `v_or` 是**位不重叠的地址拼接**(or 与 add 等价、同 1 条 VALU);真正差异是 FLY 在
  cshuffle 给每个 (i,J,v) 算**完整 LDS 地址**,而 HIP 用 `ds_read/write` 的**立即 offset 字段**。
- **对齐(已落地)**:cshuffle store + epilogue read 改成「算一次 base + 编译期常量偏移」→ LLVM 折进
  ds 的 `offset:N`(实测 32 处 ds 立即 offset)。VGPR BM=16 102→100。
- **实测中性**:M=256 仍 1.20x(与 Pattern-A-only 同)。**符合预期**:VALU 是 issue-bound(M=256 cycle
  预算 VALU stall 仅 1.8%)+ kernel memory-bound → 减 VALU 指令不减 wall-clock(同 M=4096 去 accvgpr)。

##### PMC 定位 M=256 真实 gap = L2 内存流量(2026-06-11)

`att_target_cu=1` 单 CU ATT 不代表跨 CU wall-clock。改用 **PMC(全 CU 聚合,TCC_HIT/MISS;gfx950 无
TCC_EA_RDREQ)**,M=256:

| | TCC_HIT | TCC_MISS | 总请求 | L2 命中率 |
|---|---|---|---|---|
| FLY | 740,700 | 5,679,000 | 6,419,700 | 11.5% |
| HIP | 539,900 | 5,320,500 | 5,860,400 | 9.2% |

**真实 gap = FLY L2 读请求 +9.6%、L2 miss(→HBM)+6.7%**(L2 命中率 FLY 反而略高 → 不是缓存策略,是
**请求总量**:同一问题多读 ~0.56M 次)。这才是 1.20x 的大头;Pattern A(waitcnt)/B(VALU)是单 CU
ATT 占比大但 wall-clock 权重小。**下一步**:per-load 归因找出多出的 10% 流量来自哪个 load(疑似 FLY
未 early-skip padding tile → 对 padding tile 也做了 B-load+MFMA,而 HIP `if pid>=total_work return`
早退),然后消除。

**当前工作树(未提交)改动**:atomic 路径 (1) token/weight pre-cshuffle 预取(Pattern A 对齐),
(2) cshuffle/epilogue ds-立即-offset(Pattern B 对齐)。cos 全 PASS,大 M 无回退,M=256 1.22→1.20x。

##### ★ 根因 = padding tile 冗余 GEMM(per-load 归因发现);early-skip 后全 M 追平/反超 HIP

PMC 显示 FLY 多发 ~10% L2 读请求。per-load 归因发现:**host 传给 gemm2 的 `total_m_blocks =
max_sorted/BM`(padded 最坏情况),而 atomic + mxfp4out 的 dispatch 对 grid 里每个 tile 都
`run_one(...)`,`blk_valid`(行 306)算了却没用 → FLY 对所有 padding tile 也做了完整 A+B load + MFMA**;
HIP `if (pid >= cumsum/BM * nnb) return` 只跑 valid tile(cumsum 是运行时实际数)。M=256:FLY ~445
tiles vs HIP ~385 → 多读 B → +10% L2 流量。

**修复(已落地)**:atomic + mxfp4out dispatch 用已算好的 `blk_valid`(`m_block*BM < num_valid`,
block-uniform 不发散)包住 `run_one` 的 scf.IfOp → early-skip padding tile(== HIP early-return)。
bf16flat 不受影响(persistent loop 的 bound `(num_valid/BM)*nnb` 本就是 cumsum-based)。

**效果(gemm2 fly/mx,canonical 全量 bench;cos 全 PASS)**:

| M | early-skip 前 | **after** | | M | 前 | **after** |
|---|---|---|---|---|---|---|
| 16 | 1.23x | **1.01x** | | 256 | 1.22x | **0.98x** ✅ |
| 32 | 1.25x | **1.00x** | | 4096 | 1.07x | 1.07x(bf16flat 不变) |
| 64 | **1.47x** | **0.99x** ✅ | | 8192 | 0.93x | **0.79x** ✅ |
| 128 | 1.33x | **1.00x** | | 16384 | 0.90x | **0.81x** ✅ |

e2e total(msfg/mx 完整管线):小 M 0.99–1.00x、M=8192 **0.90x**、M=16384 **0.91x**。仅剩 M=4(1.13x,
绝对 ~11µs,launch 开销主导)和 M=4096 bf16flat(1.07x,§3.1.1 的 memory-latency 墙)未达 1.0x。

**教训**:单 CU ATT 的 Pattern A/B(waitcnt/VALU)是「看着大、wall-clock 权重小」的误导项;真正的杠杆
靠 **PMC(全 CU L2 流量)+ per-load 归因**才暴露——是「跑了不该跑的 padding tile」这种**算法层冗余**,
而非指令调度。一个 early-skip 同时修好了小 M(atomic)和大 M(mxfp4out)。

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
- ~~**persistent grid**：实测否决~~ → **已采纳（fix6 §2.6）**。早先否决的依据（「HBM 未打满 →
  persistent 无收益」）是错的：persistent 的真实收益是**摊销每 tile 的 prologue**（与 HBM 是否打满
  无关），bf16flat 据此 1.31x→1.02x。教训：把 persistent 的收益只归因于「打满 HBM/摊薄 launch」会
  漏判——对 prologue 占比大的 1-tile-per-block 形态，摊销 prologue 才是主因。
- ~~**降 LDS 提 occupancy**：…未做~~ → **已采纳（fix5 §2.5）**：bf16flat LDS 128KB→32KB 已落地，
  配合 fix6 的 2 blocks/CU。
- **（仍未做）epilogue 向量化**：128 个散 short store 无法在直写布局下向量化（§3.1）；cshuffle 转
  行连续需 128KB lds_acc → 退回 1 block/CU，得不偿失。

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
