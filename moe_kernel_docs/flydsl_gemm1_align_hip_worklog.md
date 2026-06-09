# FlyDSL gemm1 (mxfp4 a4w4, BM=16) 对齐 HIP 工作记录

> 目标：让 FlyDSL 版 `gemm1_kernel_0` 在 gfx950 (MI35x) 上的性能对齐 HIP 参考
> kernel `aiter::mxfp4_moe::gemm1::kernel<655360, 385, 7168, 1024, 16, ...>`。
>
> - **被优化 kernel（target）**：`aiter/ops/flydsl/kernels/mxfp4_a4w4_gemm.py` 里的
>   `compile_mxfp4_gemm1_a4w4` / `launch_gemm1`（Python 构建 MLIR → JIT 成 gfx950）。
> - **参考 kernel（reference）**：HIP `gemm1_a4w4.cuh` 的 `run_one`。
> - **基准配置**：bench `bench_up_moe_v1.py`，`H=7168, N_OUT=1024, BM=16, num_n_blocks=4`，
>   数据类型 a4w4（fp4×fp4 → bf16 + 中间 fp4 再量化），`waves_per_eu=3`。
> - **关注点**：小 M（M=4/8/16），此时 gemm1 占用率被 grid 限制（见 §3）。

---

## 1. 性能进展（M=8 为主，gfx950，cos 为与 HIP 的余弦相似度）

| # | 改动 | gemm1 us (Fly) | 对 HIP | full-bench msfg | ratio vs mx | cos | commit |
|---|---|---|---|---|---|---|---|
| 0 | K-loop 重写 + A-scale packed + 标量寻址 + epilogue DPP + waves_per_eu=3（2-D grid） | 50.0 | 1.38x 慢 | 73.8 us | 1.25x | 1.0000 | （工作区） |
| 1 | **launch 改 1-D grid + kernel 内 pid→(m,n) 解码** | **36.2** | **追平**（HIP 36.1） | **59.1 us** | **1.00x** | 1.0000 | `19eae7f4` |

M=4 中性（27.6 us，1.01x）；M=16 gemm1 63.3 us **反超** HIP 64.2 us（full-bench ~1.05x 是非-gemm1 的既有差距，与本改动无关）。

---

## 2. 本次改动（commit `19eae7f4`）

文件：`aiter/ops/flydsl/kernels/mxfp4_a4w4_gemm.py`

**(a) launch：2-D grid → 1-D grid**

```python
# 改前
.launch(grid=(gx, num_n_blocks, 1), ...)          # gx = total_m_blocks
# 改后
g_total = gx * arith.constant(num_n_blocks, index=True)   # = total_m_blocks * num_n
.launch(grid=(g_total, 1, 1), ...)
```

**(b) kernel 内 pid→(m,n) 解码（== HIP `m=pid/num_n, n=pid%num_n`）**

```python
_pid    = gpu.block_id("x")
m_block = _pid / arith.constant(num_n_blocks, index=True)   # pid / num_n
n_block = _pid % arith.constant(num_n_blocks, index=True)   # pid % num_n
```

`num_n_blocks = N_OUT//BN = 1024//256 = 4`（2 的幂 → 编译成 shift/and，无额外开销）。
本改动**不改变所执行的 block 集合**（valid block 完全一样，guard 不变），只改变
**block 在派发序里的 flat-index 编号**，进而改变它们落到哪个 CU。

---

## 3. 为什么 M=8 提升这么大

### 3.0 一句话

M=8 时 grid 太小，每个 CU 平均只摊到 ~1 个 block，kernel 墙钟 ≈ 最慢那个 CU 的时间。
2-D grid 的 flat-index 编号方式让**有效 block 分成 4 段、彼此相隔 `stride = total_m`**，
被硬件按 flat-index 条带分发到 CU 时这几段“模 CU 数撞车”，把 2 个有效 block 落到同一
个 CU。`waves_per_eu=3` 允许这 2 块在该 CU 上**并发同驻**（不是串行），结果该 CU 被
**过订阅**（8 wave 抢执行单元/VMEM/HBM），两块**互相拖慢**（slot0 ~73k→~99k、slot1 拖到
~117k），尾巴拖长 ~60%（straggler）；1-D grid 让有效 block 在 pid 上**连成一段**，被均匀
铺成 1 块/CU，无 CU 过订阅，straggler 消失。

### 3.1 前提：M=8 是 “grid-limited 到 ~1 block/CU”

一个 block(workgroup) = 256 线程 = 4 wave（wavefront=64）。CU 有 4 个 SIMD，4 wave
正好 **1 wave/SIMD**。`waves_per_eu=3` 表示寄存器上一个 CU 最多能驻留 3 个这样的 block
——前提是**有足够多的 block 喂进来**。

M=8 时 MoE 路由后真正有效的输出 tile 只有不多的几段，数量级和参与计算的 CU 数相当，
设备只够铺“一薄层 block”：**平均**每个 CU 约 1 个 block，远凑不满“每 CU 2~3 块”的健康占用。
后果：

- 绝大多数 CU 只有 1 个 block，**没有第 2、3 个 block 来隐藏它的访存延迟** → 纯暴露延迟
  （main loop / epilogue ~90% 卡在 WAITCNT+VMEM）。
- **kernel 不会结束，直到最后一个 CU 跑完它手上的活**，即墙钟 ≈ 最慢 CU 的收尾时刻。
- 在这个稀疏 regime 下，**block 落到哪个 CU 极其敏感**：若分配不均把 2 块塞到同一个 CU
  （见 §3.3），这 2 块只能并发同驻、过订阅互相拖慢（而非有益地隐藏延迟，因为瓶颈是访存
  带宽 + 发射口竞争），那个 CU 就成了拖尾的 straggler。

在这个 regime 下，**有效 block 怎么分到 CU 上，直接决定尾巴长度**——这就是本改动的全部杠杆。

### 3.2 关键事实：valid 在前、padding 在后（连续，非交织）

gemm1 以**固定最大 grid = `max_sorted/BM` 个 m-block**（含 padding）启动，device 端
guard 跳过 padding：

```python
# mxfp4_a4w4_gemm.py:1716-1735
_num_valid = cumsum            # 有效 sorted 行数
_blk_valid = (m_row = m_block*BM) < _num_valid      # 单一连续阈值
```

所以判定是**单一连续阈值**：`m_block < Mv`（其中 `Mv = ceil(cumsum/BM)`）为 valid，
`m_block >= Mv` 为 padding（秒退）。即每个 n 列都是

```
[V V V ... V | P P P ... P]      ← 前 Mv 个 valid，其后全是 padding（连续，不交织）
```

> 注（勘误）：早期描述写成“列内 V/P 交织”是错的。真正导致不均衡的不是列内交织，而是
> **列与列之间** valid run 的间隔（见 3.3）。小 M 时 `cumsum` 很小但 `max_sorted` 是按最大
> 容量开的，于是 `total_m = max_sorted/BM >> Mv`，padding 极多。

### 3.3 病根：2-D 让 4 段 valid 相隔 stride=total_m，条带分发撞车

2-D grid `(total_m, 4)` 按 x 最快线性化，flat index `f = m + n·total_m`。有效 block
（`m < Mv`，全部 4 个 n）落在的 flat 区间是：

```
flat:  0 .............. total_m-1 | total_m .......... 2·total_m-1 | ... (共 4 段)
n=0:   [V(0..Mv-1) | P(Mv..total_m-1)]
n=1:                                  [V(0..Mv-1) | P ...........]
n=2/3: ...
       └── 4 段 valid，每段长 Mv，彼此相隔 stride = total_m（中间隔着大量 padding）──┘
```

硬件 workgroup 分发器（SPI）按 **flat-index 跨 CU 条带分发**（round-robin / striping，
再按占用率均衡）。4 段 valid 的目标 CU 集合彼此偏移约 `(total_m mod C)`（C=参与 CU 数）。
当 `total_m` 很大（padding 多）时，这几段偏移叠加后会**落到互相重叠的 CU 集合上**：

- 少数 CU 收到来自多段 valid 的 **2 个有效 block**；
- 另一些 CU 只收到 padding（0 个有效 block）。

`waves_per_eu=3` 表示一个 CU 最多能同驻 3 个 block，所以这 2 块会在该 CU 上**并发同驻**
（occupancy.json 实测：两块的 slot0/slot1 几乎同时启动，见 §3.5），而不是串行。后果是该 CU
被**过订阅**——8 个 wave 抢同一套 MFMA/VALU/VMEM/LDS 发射口和 HBM 带宽，两块**互相拖慢**：
slot0 从单块的 ~73k 拖到 ~99k，slot1 更拖到 ~117k（约 1.35~1.6× 单块时间）。kernel 直到这
俩过订阅 CU 收尾才结束 → straggler。

### 3.4 解法：1-D 让 valid 连成一段，均匀铺成 1 块/CU

1-D grid `(total_m·4,)`，`m=pid/4, n=pid%4`，flat index `f = pid = m·4 + n`。有效 block
（`m < Mv`）落在 **pid ∈ [0, 4·Mv) 一段连续区间**，padding 全在其后连续排列：

```
flat = pid:  0 ................................. 4·Mv-1 | 4·Mv ....... end
             [ V V V V V V ...（全部 4·Mv 个有效 block）  | P P ... P ]
             └────────────── 连成一整段 ──────────────┘
```

条带分发器把一段**连续**的 flat-index 最大限度均匀摊到各 CU：每 CU 拿 `⌈4Mv/C⌉` 或
`⌊4Mv/C⌋` 个有效 block。在 M=8 的 trace 里 `4Mv ≤ 参与 CU 数`，于是每 CU 恰好 1 块，
padding 全在后面无害地秒退。**没有任何 CU 背 2 块 → straggler 消失。**

### 3.5 数据对账（occupancy.json，`ui_output_agent_*/occupancy.json`）

用临时脚本 `aiter/_occ_cmp.py` / `_occ_cmp2.py` 解析每个 CU 的 slot0/slot1 起止时刻：

**旧 2-D grid：**

| CU | slot0 `[start, end]` | slot1 `[start, end]` | 说明 |
|---|---|---|---|
| 大多数 CU（7 个） | `[~30, ~73000]` | 无 | 1 块，正常 |
| **CU5** | `[32, 98780]` | `[860, 117292]` | **slot1 在 t=860 就启动**（≪ slot0_end=98780）→ 2 块**并发同驻**、过订阅 |
| **CU6** | `[48, 100048]` | `[1644, 115816]` | 同上，slot1 t=1644 启动 |

关键：CU5/CU6 的 slot1 起始时刻（860 / 1644）远**早于**各自 slot0 的结束（98780 / 100048），
即**两块从一开始就并发重叠**（不是 slot0 跑完才跑 slot1）。一个 CU 同时扛 2 块 → 过订阅 →
两块互相拖慢：slot0 比单块的 ~73k 慢到 ~99k，slot1 拖到 ~117k。kernel 被这俩 straggler 卡到
~117k。end-span（仅 slot0）= 27724，算上 slot1 的真实尾巴 ~44k。

**新 1-D grid：**

| CU | slot0 结束 | slot1 |
|---|---|---|
| 全部 9 个 CU | **80068 ~ 84584** | 只有开头 ~3k 的短暂并发 blip（非第 2 块） |

→ end-span = **4516**（~5%），与 HIP（7500）同量级，**straggler 没了**。

### 3.6 反直觉细节：单块略慢，但净赢

旧 2-D 的单块 CU 反而比新 1-D 的单块更快（~73k vs ~82k，两份 ATT 各自时钟、仅作定性参考）：

- 旧 2-D：未过订阅的 CU 各只跑 1 块，且设备级同时在跑的有效块较少 → HBM/L2 竞争小 →
  单块快（~73k）；**但代价是 CU5/CU6 被塞了 2 块并发、过订阅拖到 ~117k**。
- 新 1-D：所有有效 block 在 t≈0 一起涌出、设备级访存竞争更大 → 单块稍慢（~82k），
  **但每个 CU 只有 1 块、无过订阅，全员 ~82k 齐刷刷收尾**。

墙钟 = 各 CU 最后一块的收尾时刻取最大值：
```
旧 2-D 墙钟 ≈ max(单块 CU ~73k, 过订阅 CU 2 块并发 ~117k) = ~117k
新 1-D 墙钟 ≈ 全员单块并发各 ~82k                        =  ~82k    →  -30%
```
与实测 gemm1 **50.0 → 36.2 us（-28%）**、full-bench **1.25x → 1.00x** 吻合。
**“消掉过订阅 CU 上的第 2 块”远比“单块快十几个百分点”值钱。**

### 3.7 为什么只有小 M 受影响

- **小 M（M=4/8）**：有效 tile 只有几十个 ≈ CU 数，1 块/CU，任何一个 CU 多背 1 块 ≈
  墙钟翻倍那一截 → 分配不均的代价被**完全暴露**。
- **大 M（如 M=8192）**：有效 tile 成千上万，每 CU 排几十块，多 1 块只是“几十分之一”
  的扰动，且块多了天然有 latency hiding → 大数定律把不均摊平，2-D/1-D 几乎无差别
  （这也是 M=16 已基本无感、M=4 中性的原因）。

HIP 本就用 1-D pid 解码，所以小 M 天生没有这个 straggler 问题。本改动本质就是
**把 FlyDSL 的 launch 拓扑对齐 HIP**，去掉“2-D grid 引入的纯结构性 block 分配不均”。

---

## 4. 验证

- 正确性：M=4/8/16 全部 `cos = 1.0000`（vs HIP）。
- 性能（gfx950，iters=100，多次复跑稳定）：M=8 gemm1 36.2 us（== HIP 36.1~36.6），
  full-bench 58.9~59.1 us（0.99~1.00x）；M=16 gemm1 63.3 us（< HIP 64.2）。
- 资源：spill 0（ISA 对齐阶段已确认）。
- 改动仅在 aiter 前端 Python（JIT 编译），不涉及 LLVM / FlyDSL C++ build；验证前已
  `rm -rf ~/.flydsl/cache/*` 强制重编。

## 5. 复现命令

```bash
# 容器 hyg_fyd2，GPU 1
docker exec hyg_fyd2 bash -c '
  cd /shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter &&
  rm -rf ~/.flydsl/cache/* &&
  export HIP_VISIBLE_DEVICES=1 AITER_LOG_MORE=1 &&
  MLIR_LIBS_DIR=/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/FlyDSL/build-fly/python_packages/flydsl/_mlir/_mlir_libs &&
  export LD_LIBRARY_PATH=$MLIR_LIBS_DIR:${LD_LIBRARY_PATH:-} &&
  python3 bench_up_moe_v1.py --M-list 4,8,16 --iters 100 --hash --benchmarks mx msfg
'

# occupancy 抓取（ATT）+ 分析（脚本：aiter/_occ_cmp.py、aiter/_occ_cmp2.py）
rocprofv3 -i ./fmha_opt_tools/tt_fly_gemm1_m8.yaml -- \
  python3 bench_up_moe_v1.py --M-list 8 --iters 10 --benchmarks mx msfg
python3 _occ_cmp.py     # 每 CU block 激活数 / 并发度 / span
python3 _occ_cmp2.py    # 每 CU slot0/slot1 起止（检测过订阅 straggler：一个 CU 并发 2 块）
```

---

# 附录 A：BM=128 NT direct-LDS A-load 变体——死 init-zero MFMA 污染 accm（cos 0.976 → 0.9998）

> 这是与正文（BM=16 launch 拓扑）**独立**的另一个 bug，发生在 BM=128 的 NT（pre-quant
> A_q）路径、A 走 HIP 式 direct VMEM→LDS DMA（`load_a_directlds`）的变体上。该变体在
> `aiter/ops/flydsl/kernels/mxfp4_a4w4_gemm.py` 里由 `MXFP4_G1_A_LOAD_VARIANT=direct_hip_2slot_kmajor`
> 选中，并被 dump 成裸 asm 在 `aiter/exp_isa/` 下用 `gemm1_asm` 直接跑（绕开 FlyDSL JIT）。

## A.1 现象

用 `aiter/_test_g1_dump_variants.py`（对一份固定输入 dump，逐变体跑 FlyDSL gemm1 并与
HIP `mxfp4_moe_gemm1_a4w4` 的输出比 cos，**只统计真实 sorted 行** `m_indices < M`）：

```text
asm_direct_hip_2slot_kmajor   q_cos 0.9763  q_exact 0.7209  q_max 241  q_mean 7.69
                              s_cos 0.8956  s_exact 0.9635  s_max 122  s_mean 3.44
```

`q_cos`（gemm1 输出 fp4）只有 0.976、`s_cos`（输出 e8m0 scale）只有 0.896，远不到目标 >0.99。

同时 ISA 指令数对不上 HIP：

| 指令 | FlyDSL（坏） | HIP | 差 |
|---|---|---|---|
| `v_mfma_scale_f32_16x16x128_f8f6f4` | **1824** | 1792 | **+32** |
| `ds_read_b128`（loop） | 450 | 448 | +2 |
| `ds_read_b128`（epilogue） | 14 | 0 | +14（cshuffle 读法不同，无害） |

K-loop 结构（prologue 无 MFMA、main 26 iter、drain 2 iter = 28 tile × 每 tile 64 MFMA）
理论上应是 `28×64 = 1792`，与 HIP 一致；但实测 1824。

## A.2 定位：多出的 32 条 MFMA 全在「第一个 tile」

按 `s_barrier`（tile 边界）切段统计 K-loop 每段的 MFMA 数：

```text
[96, 64, 64, 64, ... 64, 0, 0]
  ^ 第 1 个 tile = 96，其余全是 64
```

第一个 tile 96 = 64 + **32**。而 `32 = m_repeat(8) × num_acc_n(4) = accm 总数`。
第一个 tile 是 `off==0`，正是 `kinit=True`（init-zero）那一拍 → 嫌疑直指 init 写法。

## A.3 根因（深入到 AGPR 分配层）

`mfma_cluster_J_all_direct` 首 tile 的 k0 用了 **output-only 的 init-zero 内联汇编**
（`"=a,v,v,v,v"`，C=literal-0，与 HIP `AITER_MXFP4_MFMA_F4F4_AGPR_INIT_ZERO` 写法相同）：

```python
# init-zero 形式（kinit=True）：输出 =a，C 是字面量 0，不吃 accm
asm = "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 " + osel
llvm.inline_asm(vec4_f32, [a4, b4, sa, sb], asm, "=a,v,v,v,v", has_side_effects=True)
# 累加形式（kinit=False）：+a，C=D=accm（tie 到 operand 0）
asm = "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 " + osel
llvm.inline_asm(vec4_f32, [acc, a4, b4, sa, sb], asm, "=a,0,v,v,v,v", has_side_effects=True)
```

两种写法**单看都对**，问题在 init 输出和后续累加之间的 **AGPR 绑定**：

- **HIP（C++）**：`accm[i][J]` 是同一个变量，`INIT_ZERO(accm[i][J], ...)` 写它、
  `AGPR(accm[i][J], ...)`（`+a`）读写它 → clang 把 init 与累加 tie 到**同一个 AGPR**，
  没有死代码，首 tile 也是 64 条 MFMA。

- **FlyDSL**：`accm[idx] = init(...)` 产生一个**新 SSA 值**，再 `accm[idx] = acc(accm[idx], ...)`。
  按 `"=a,0"` 约束，累加的输出应 tie 到它的 operand 0（= init 的输出），理应同一 AGPR。
  但后端**没有**把 init 的输出 tie 到这条 loop-carried accm 链上，于是：

  1. init 是 `has_side_effects=True` 的内联汇编，后端**保留**它，但把它的输出分配到一个
     scratch 寄存器 `a[0:3]`（其结果没人用 → **死代码**）。同时另起一条「`v_accvgpr_write a0,0`
     + 一串 `v_accvgpr_mov` 清零真正的 accm AGPR + 用**累加形式重算 k0**」的路径。
     → 首 tile 每个 accm 多出 1 条 MFMA = **+32 条死 MFMA**（1824 vs 1792）。

  2. **更致命的副作用**：后端把这些死 init 的输出物理寄存器 `a[0:3]`/`a[4:7]`
     **复用成了另外 2 个真实 accm 的物理寄存器**。死 init 在 `off==0` 往这俩寄存器写入了
     「accm[0].k0 的乘积」这种**垃圾初值**，而这俩真实 accm 的累加链却从这个垃圾值起步、
     而非从 0 起步 → **2 个 accm 被污染** → `q_cos 0.976`、`s_cos 0.896`。

## A.4 ISA 证据

新 dump 的 `exp_isa/flydsl_gemm1_BM128_direct_hip_2slot_kmajor_21_final_isa.s`（坏版本）首 tile：

```asm
278: v_mfma_scale ... a[0:3],  v[66:69], v[10:13], 0,        v96, v71 op_sel:[0,0,0]  ; init-zero，输出 a[0:3] —— 死代码
280: v_accvgpr_write_b32 a0, 0                                                        ; 现场材料化 0
281-294: v_accvgpr_mov_b32 a52,a0 / a44,a0 / ...                                      ; 把 0 拷到真正的 accm AGPR
296: v_mfma_scale ... a[52:55], v[66:69], v[10:13], a[52:55], v96, v71 op_sel:[0,0,0] ; 重算 k0（累加进已清零的 a[52:55]）
313: v_mfma_scale ... a[52:55], v[118:121],v[14:17], a[52:55], v96, v71 op_sel_hi:[1,1,0] ; k1
```

`a[52:55]` 才是 accm[0]（`= 296 的 k0 + 313 的 k1`），`a[0:3]` 的 init（278）是死的。
但用 grep 统计「谁在写 `a[0:3]`」会发现它**同时是一个真实 accm**：

```text
a[0:3]  init(C=0): 1        accumulate(C=a[0:3]): 56     # 1 条死 init + 56 条真实累加 → 被污染
a[4:7]  init(C=0): 7        accumulate(C=a[4:7]): 56     # 7 条死 init + 56 条真实累加 → 被污染
```

`56 = 28 tile × 2 k`，说明 `a[0:3]`/`a[4:7]` 是货真价实的 accm 链；而那 1/7 条 `C=0` 的 init 把
它们的累加起点污染了。被污染的 2 个 accm 对应 2 组 `(mi, J)` 输出 → 一批输出行/列错，整体
`q_cos` 掉到 0.976。

> 排查中已逐一排除其它可能：DMA→read 的 vmcnt 同步（`s_waitcnt vmcnt(0)` 全 drain 仍 0.98）、
> swizzle（写读两端都用 HIP 的 `((row)&14)<<3`，自洽；强行换公式反而 0.09）、把全部 barrier
> 换成带 fence 的 `gpu.barrier`（无效）、padding 行 OOB 陈旧（预清零 LDS 无效）。
> 决定性隔离实验：**A 改回 `store_a_tile` + 其余不变 → q_cos 0.9968**，证明 barrier/调度都无关，
> 锅在 direct 路径首 tile 的 init 写法（即本节根因）。

## A.5 修复（纯前端，非 copy 其它版本）

去掉 init-zero 形式：**k0 也用累加形式，累加到「入口即已清零」的 accm**
（`accm` 进 K-loop 前就是 `acc_init = constant_vector(0)`）。这样后端只生成「清零 distinct
accm AGPR + 累加」这条本来就正确的路径，不再有死 init、也不会拿死 init 的输出去复用真实
accm 寄存器——数据流与 HIP `issue_mfma_cluster` 一致。

`aiter/ops/flydsl/kernels/mxfp4_a4w4_gemm.py::mfma_cluster_J_all_direct`：

```python
# 改前：首 tile k0 用 init-zero（=a，C=0）
accm[acc_idx] = _mfma_agpr(..., 0, inxdl, kinit=kinit)   # i0.k0
accm[acc_idx] = _mfma_agpr(..., 1, inxdl, kinit=kinit)   # i1.k0
# 改后：k0 也用累加形式（kinit=False），累加到已清零 accm
accm[acc_idx] = _mfma_agpr(..., 0, inxdl, kinit=False)   # i0.k0
accm[acc_idx] = _mfma_agpr(..., 1, inxdl, kinit=False)   # i1.k0
```

（`kinit` 形参随之失效；前提是 `accm` 在 K-loop 入口为 0，已满足。）

## A.6 验证

| 变体 | q_cos | q_exact | s_cos | mfma | 首 tile MFMA |
|---|---|---|---|---|---|
| 改前 `asm_direct_hip_2slot_kmajor` | 0.9763 | 0.7209 | 0.8956 | 1824 | 96 |
| **改后**（FlyDSL JIT `direct_hip_2slot_kmajor`） | **0.9999** | 0.9991 | **0.9994** | **1792** | 64 |
| **改后**（asm `asm_direct_hip_2slot_kmajor`，重编 .co） | **0.9998** | 0.9982 | **0.9995** | **1792** | 64 |

- `mfma` 精确对齐 HIP 的 **1792**；首 tile 由 96 回到 64；死 init 与 accm 污染消失。
- asm 变体与前端 JIT 变体 **q_hash 完全一致**（`0cff05e935b3462f`），证明 dump 出来的 asm
  与前端语义一致。
- 残留 `ds_read_b128=464`（HIP 448）：差异全在 epilogue 的 cshuffle 读法（FlyDSL 14 条
  `ds_read_b128`，HIP 0 条，用别的读指令）+ loop 2 条，**不影响正确性**（cos 已 0.9998）。

## A.7 复现命令

```bash
docker exec hyg_fyd2 bash -c '
  cd /shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter &&
  export HIP_VISIBLE_DEVICES=1 &&
  MLIR_LIBS_DIR=/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/FlyDSL/build-fly/python_packages/flydsl/_mlir/_mlir_libs &&
  export LD_LIBRARY_PATH=$MLIR_LIBS_DIR:${LD_LIBRARY_PATH:-} &&

  # 1) 改前端后，重新 dump 该变体的 21_final_isa.s
  rm -rf ~/.flydsl/cache/* &&
  FLYDSL_DUMP_IR=1 python3 _test_g1_dump_variants.py \
      --dump /tmp/mxfp4_gemm1_m4096_inputs.pt --variants direct_hip_2slot_kmajor &&
  cp /root/.flydsl/debug/gemm1_kernel_0/21_final_isa.s \
     exp_isa/flydsl_gemm1_BM128_direct_hip_2slot_kmajor_21_final_isa.s &&

  # 2) 只汇编该 .s 为 .co（== exp_isa/build.sh 第 32-34 行）
  cd exp_isa &&
  /opt/rocm/llvm/bin/clang++ -x assembler -target amdgcn-amd-amdhsa --offload-arch=gfx950 \
      flydsl_gemm1_BM128_direct_hip_2slot_kmajor_21_final_isa.s \
      -o flydsl_gemm1_BM128_direct_hip_2slot_kmajor_21_final_isa.co &&
  cd .. &&

  # 3) 跑 asm 变体核对 cos
  python3 _test_g1_dump_variants.py \
      --dump /tmp/mxfp4_gemm1_m4096_inputs.pt --variants asm_direct_hip_2slot_kmajor
'
```

---

# 附录 B：BM=128 NT direct-LDS A-load 主循环**性能**对齐 HIP

> 接附录 A（修正 direct-LDS 正确性 cos 0.976→0.9998）之后，继续 close 该路径**主循环性能**
> 与 HIP 的 gap。被优化变体仍是 `MXFP4_G1_A_LOAD_VARIANT=direct_hip_2slot_kmajor`，参考
> 是 HIP `gemm1_a4w4.cuh` BM=128 NT direct-to-LDS DMA。
>
> - **基准**：`bench_up_moe_v1.py` M=4096/8192，gfx950（MI35x），`waves_per_eu` 见 ISA。
> - **指标**：① 端到端 `msfg/mx` 时间比（**越低越接近 HIP**，1.0 = 追平）；② gemm1
>   thread-trace **稳态 per-tile latency budget**（rocprofv3 ATT，跳过头 2 尾 1 tile 取均值）。
> - **一切以 thread-trace dump 的汇编为准**（HIP `hip_gemm1_m4096`，Fly `fly_direct_gemm1_m4096`）。

## B.1 性能进展

| # | 改动 | 主循环 vs HIP（per-tile） | 主循环 s_nop | lgkmcnt 形态 | bench msfg/mx M4096 / M8192 | cos M4096/M8192 | commit |
|---|---|---|---|---|---|---|---|
| A | （附录 A）init-zero 修正，正确性达标 | +33% | 428 | `lgkmcnt(1)` 一次 drain | — | 0.9998 | （前） |
| B1 | **地址标量化**（`load_a_directlds` voff/m0row 提到循环外） | **+33%→+15.6%** | 428 | `lgkmcnt(1)` | 1.32x / 1.37x | 1.0000 / 0.9999 | `a4b9ff14` |
| B2 | **k-outer MFMA 顺序** | — | **428→182** | `lgkmcnt(1)` | 1.32x / 1.37x | 1.0000 / 0.9999 | `a4b9ff14` |
| B3 | **scale-first**（A-scale ds_read 前置） | +14.9% | 182 | **递减 `lgkmcnt(14→13→…→0)`** | 1.32x / 1.37x | 1.0000 / 1.0000 | `8bb9527e` |

> B1 地址标量化 + B2 k-outer 同 commit `a4b9ff14`；B3 scale-first 为 `8bb9527e`。
> 端到端 bench 在 B1 之后即稳定 1.32x/1.37x，B2/B3 是 **ISA 对齐**（见 B.3：访存受限，
> compute 调度的改善被 vmcnt 掩盖，bench 不动但 trace 对齐 HIP）。

## B.2 有效优化（详细）

### B.2.1 地址标量化 `load_a_directlds`（主循环 +33%→+15.6%）

**问题**：每 sub 每 tile 都用 `v_mul_lo_u32` + `m_indices` re-gather **现算** DMA 向量地址
（`voffset = (swizzle ^ col) + token*(K/2)`），与 per-tile `s_waitcnt vmcnt` **串行化** →
4 个 sub 的 A-DMA 无法 burst 发射，load 延迟落在 vmcnt stall（~2200 cyc/tile）而非与 MFMA 重叠。

**改动**：voffset（token gather + token×(K/2)，**与 kt 无关**）和 LDS 目标行基址
（`lds_row*lds_stride`，**与轮转 slot 无关**）提到 K-loop **外算一次**（== HIP `cached_actual_row`
+ hoisted base）；`load_a_directlds(slot,kt)` 每 tile 只加编译期 `slot/kt` **标量**偏移 → 4 个
sub DMA 背靠背 burst，关键路径上无地址 ALU。

**寄存器**：VGPR 308 / SGPR 54 / AGPR 128 / spill 0（多占 per-sub 几个 hoisted 寄存器）。

### B.2.2 k-outer MFMA 顺序（s_nop 428→182）

**问题**：`mfma_cluster_J_all_direct` 原用 per-sub pairwise（`i0.k0,i1.k0,i0.k1,i1.k1`），
同一 accm **每 2 条 MFMA 就复现**（如 `a[60:63].k0` 后隔 1 条就是 `a[60:63].k1`），间隔太小 →
`GCNHazardRecognizer` 每 2 条 MFMA 插 1 个 `s_nop`（trace：428 条 s_nop 在 v_mfma 之后）。

**改动**：改 **k-outer, mi-inner**（全 `m_repeat` 个 m-chunk 的 k0 背靠背，再全 k1）==
HIP thread-trace **实际 ISA 顺序**（trace 实测 16 条 MFMA 连续、0 个 s_nop）。同一 accm 的
`k0→k1` 间隔 `m_repeat=8` 条独立 MFMA → MFMA 单元背靠背发射，无 hazard nop。

**寄存器**：中性（纯重排，数值恒等——每个 accm 仍是 `0 → +k0 → +k1` 同序）。s_nop 428→182。

### B.2.3 scale-first（递减 lgkmcnt）

**问题**：主循环里 `read_a_tile_all`（16 个 A `ds_read`）**先**发、`issue_a_scale_lds_read`
（A-scale `ds_read`）**后**发；第一条 MFMA 需要 A-scale 操作数，却排在第 17/18 个 lgkm op →
`SIInsertWaitcnts` 发**单条** `lgkmcnt(1)`（drain ~全部，stall 2172）卡在 MFMA cluster 前。

**改动**：把 `issue_a_scale_lds_read` 提到 `read_a_tile_all` **之前**（主体 + drain tail 两处）。
SIInsertWaitcnts 随之发 HIP 式**递减** `lgkmcnt(14)→(13)→(12)→…→(0)`（各 stall ~560），与 MFMA
交错，每条 MFMA 只等它需要的那个 ds_read —— 与 HIP thread-trace 完全一致。

**寄存器**：中性（纯重排两组独立 ds_read）。

## B.3 关键发现：gemm1 mxfp4 **访存受限**，compute 调度已尽

稳态 per-tile latency budget（k-outer + scale-first，vs HIP `hip_gemm1_m4096`）：

| class | Fly direct | HIP | delta |
|---|---|---|---|
| WAITCNT | 348 | 87 | **+261** |
| BARRIER | 408 | 209 | **+199** |
| VMEM_LD | 1007 | 996 | +11 |
| NOP | 23 | 21 | +2 ✓ |
| MFMA | 921 | 934 | −14 ✓ |
| LDS | 209 | 280 | −71 ✓ |
| **TOTAL** | **2950** | **2567** | **+384（+14.9%）** |

- **算术强度 AI≈32 flops/byte ≪ MI300/350 roofline ridge ~260 → 访存受限**。VMEM+waitcnt
  占稳态 stall **52%**（HIP 49%）。
- compute（MFMA −14、NOP +2、LDS −71）已**对齐或优于 HIP**；B2/B3 把 ISA 对齐了 HIP，但
  端到端 bench 不动 —— LDS 侧 `lgkmcnt` 的改善被主导的 **A-DMA `vmcnt`** 掩盖。
- 剩余 **+14.9%** 来源：
  - **WAITCNT +261**：主因 barrier 后 `s_waitcnt vmcnt(10)`（等上一 tile A-DMA 排空，
    trace stall ~53000；LLVM **计数式** waitcnt 无法区分「2 tile 前已完成的 DMA」与「在飞的」，
    保守 drain）。lgkmcnt 部分已对齐 HIP。
  - **BARRIER +199**：4-wave 不均衡，访存相位 sync 代价。
- **accm-init**（`v_accvgpr_mov` 127 条清零 AGPR）：**一次性 prologue**，不在稳态，摊薄后可忽略
  （之所以不能用 HIP 的 init-zero MFMA 省掉它，见附录 A：会引入死 MFMA + accm 污染）。

## B.4 试过无效 / 未采纳

- **3-slot 更深 A 缓冲**（`_a_slots=3`，消除 `read_slot==write_slot` 的 WAR + 更深藏 DMA）：
  bench **无变化**（1.32x/1.37x）→ 证明 `vmcnt` stall **不是缓冲深度问题**，是 LLVM 计数式
  waitcnt + HBM 带宽本身。已回退到 2-slot（对齐 HIP `kAStages=2`）。
- **s_setprio 降 barrier 不均衡**：未采纳（HIP 在 BM=128 也没用；4-wave 访存瓶颈下 setprio
  难解访存相位差，投机性高）。
- **safe（reg→LDS）路径对比**：safe 1.38x/1.43x，比 direct（1.32x/1.37x）**慢 ~4-5%**，故
  direct-LDS 仍为默认（更接近 HIP）。

## B.5 验证

- 正确性：cos M=4096 **1.0000**、M=8192 **0.9999/1.0000**；M=1024/2048 **0.994**（mxfp4 量化
  噪声，与 safe 路径**同值** 0.9916/0.9939 → 非 direct 路径或本轮改动的 bug）。
- 性能（gfx950，iters=300，复跑稳定）：direct `msfg/mx` M=4096 **1.32x**、M=8192 **1.37x**
  （越低越接近 HIP）。
- 资源：VGPR 308 / SGPR 54 / AGPR 128 / **spill 0** / LDS 131072。
- s_nop 主循环 428→182；`v_mfma_scale` 1792（== HIP）。
- 纯前端 Python（JIT），不涉及 LLVM / FlyDSL C++ build；验证前 `rm -rf ~/.flydsl/cache/*` 强制重编。

## B.6 复现命令

```bash
# 容器 hyg_fyd2，GPU 1。默认变体 = direct_hip_2slot_kmajor
docker exec hyg_fyd2 bash -c '
  cd /shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter &&
  rm -rf ~/.flydsl/cache/* &&
  export HIP_VISIBLE_DEVICES=1 MXFP4_G1_A_LOAD_VARIANT=direct_hip_2slot_kmajor &&
  MLIR_LIBS_DIR=/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/FlyDSL/build-fly/python_packages/flydsl/_mlir/_mlir_libs &&
  export LD_LIBRARY_PATH=$MLIR_LIBS_DIR:${LD_LIBRARY_PATH:-} &&
  python3 bench_up_moe_v1.py -M 4096 --iters 300 --warmup 60 --hash --benchmarks mx msfg &&
  python3 bench_up_moe_v1.py -M 8192 --iters 300 --warmup 60 --hash --benchmarks mx msfg
'

# gemm1 thread-trace（ATT）：dump 后用稳态 per-tile budget 脚本对账（见 thread_trace/）
rocprofv3 -i ./fmha_opt_tools/tt_fly_direct_gemm1_m4096.yaml -- \
  python3 bench_up_moe_v1.py -M 4096 --iters 100 --benchmarks mx msfg
# 对比 HIP 基线 trace：thread_trace/hip_gemm1_m4096/stats_ui_output_agent_*.csv
```

