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

---

## 3. 残余差距分析（fix3 之后 BM=128 仍 ~1.31–1.35x）

- L2 命中率已 ≥ HIP（56.8% vs 52.5%），L2 miss/HBM 流量基本对齐 → **残余不再是 L2 traffic**。
- 最可能的剩余项：**persistent grid**。HIP nonatomic 用 `grid=min(total_work, NUM_CU=256)`
  的常驻网格，256 个 block 各串 ~105 个 tile，HBM 持续打满、无重启间隙；FlyDSL 是
  non-persistent（`total_m_blocks*28` ≈ 2.7 万个 1-tile block），block 频繁启停可能让 HBM
  欠饱和。这是更大改动（kernel body 包进 `scf.for` 常驻循环 + 设备端 `cumsum/BM*28` 上界），
  收益未定，**待评估/确认后再做**。
- ATT（att_target_cu=1）对本 memory-bound kernel 的 L2 复用**不具代表性**（单 CU L2 偏冷），
  fix2/fix3 的真实收益体现在整网 kernel 时间与 PMC，不体现在单 CU ATT 总 cycle —— 故本阶段
  以 **kernel 时间 + PMC** 为准，ATT 仅用于定位指令级 stall pattern（fix1）。

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
