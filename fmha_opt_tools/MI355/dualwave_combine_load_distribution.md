# DUALWAVE_SWP + split-K combine: 负载如何划分到 block（负载均衡分析报告）

<!-- markdown-toc-generator:start -->
## Table of Contents

- [1. 主 kernel：grid → block 映射](#1-主-kernelgrid-block-映射)
- [2. 主 kernel：每个 block 的工作量（KV tile 数）](#2-主-kernel每个-block-的工作量kv-tile-数)
  - [2a. causal —— 三角形负载，block 间严重不均（主要矛盾）](#2a-causal-三角形负载block-间严重不均主要矛盾)
  - [2b. non-causal —— block 间均衡](#2b-non-causal-block-间均衡)
  - [2c. partial / padded tile（小 seqlen 或非对齐）](#2c-partial-padded-tile小-seqlen-或非对齐)
  - [2d. split-K —— 把 KV 维再切成 NUM_KV_SPLITS 份（缓解 2a + 填满小 grid）](#2d-split-k-把-kv-维再切成-num_kv_splits-份缓解-2a-填满小-grid)
- [3. combine kernel：grid → block（均衡，按输出行平铺）](#3-combine-kernelgrid-block均衡按输出行平铺)
- [4. 怎么分析负载均衡（实操指引）](#4-怎么分析负载均衡实操指引)
- [5. 实测：B1 S8192 H64 causal bf16（rocprofv3 ATT + occupancy_balance.py）](#5-实测b1-s8192-h64-causal-bf16rocprofv3-att-occupancy_balancepy)
- [6. WG 内 wave→q_seq 行映射（dualwave kernel vs 手写 ASM kernel 对比）](#6-wg-内-waveq_seq-行映射dualwave-kernel-vs-手写-asm-kernel-对比)
  - [关键代码位置速查](#关键代码位置速查)

<!-- markdown-toc-generator:end -->

分析对象：`kernels/flash_attn_gfx950.py` 的两个 kernel
- `flash_attn_dualwave_swp_gfx950_kernel`（主 FMHA kernel）
- `flash_attn_splitk_combine_kernel`（split-K 的归约 kernel，仅 `num_kv_splits>1` 时启动）

固定常量：`BLOCK_M=256`（每 block 处理 256 个 query 行）、`BLOCK_N=64`（KV tile 大小）、`BLOCK_SIZE=512`（8 waves×64）、`ROWS_PER_WAVE=32`、`HEAD_DIM=128`、`GQA_GROUP_SIZE = NUM_HEADS_Q/NUM_HEADS_KV`。

---

## 1. 主 kernel：grid → block 映射

launch（`launch_flash_attn_dualwave_swp`，行 1762-1811）：
```
num_q_blocks = ceil(seq_len / BLOCK_M)          # = ceil(seq_len/256)
grid_z       = batch                            # 非 split-K
             = batch * NUM_KV_SPLITS            # split-K
grid  = (NUM_HEADS_Q, num_q_blocks, grid_z)
block = (BLOCK_SIZE=512, 1, 1)
```

block 内解码（行 254-286）：
```
h_idx       = block_idx.x        # 0..NUM_HEADS_Q-1   —— 哪个 Q head
q_block_idx = block_idx.y        # 0..num_q_blocks-1  —— 哪个 256 行 query tile
block_idx.z:  非splitk -> batch_idx = z
              splitk   -> batch_idx = z // NUM_KV_SPLITS ; split_idx = z % NUM_KV_SPLITS
q_start = q_block_idx * 256       # 该 block 负责的 query 行起点
# GQA head 映射：
h_kv_idx  = h_idx % NUM_HEADS_KV ; group_id = h_idx // NUM_HEADS_KV
q_head_idx  = h_kv_idx * GQA_GROUP_SIZE + group_id
kv_head_idx = h_kv_idx
```

**一个 block 计算什么**：对某个 `(batch, q_head, 256 个 query 行)`（split-K 再加 `split` 维），在 KV 维上做完整 online-softmax flash-attention。512 线程 = 8 waves，每 wave 负责 32 个 query 行（`ROWS_PER_WAVE`），D=128。

> 关键：**block 的数量与 (head, q-tile, batch[, split]) 成正比，但每个 block 的"工作量"不等 —— 工作量 = 它要遍历的 KV tile 数**。负载均衡的全部问题都在"每个 block 遍历多少 KV tile"上。

---

## 2. 主 kernel：每个 block 的工作量（KV tile 数）

KV tile 数的计算（行 388-414）：
```
num_kv_tiles   = ceil(seq_len / 64)
# causal：只算到对角线
causal_num_tiles = ceil((q_block_idx+1)*256 / 64) = (q_block_idx+1) * 4
max_num_tiles  = min(causal_num_tiles, num_kv_tiles)   # causal
               = num_kv_tiles                           # non-causal
max_num_tiles  = round_up_even(max_num_tiles)           # 流水线要偶数
max_num_tiles  = max(4, max_num_tiles)                  # 流水线最少 4 tile
```

### 2a. causal —— 三角形负载，block 间严重不均（主要矛盾）
每个 block 的 KV tile 数 = `min((q_block_idx+1)*4, num_kv_tiles)`：

| q_block_idx | KV tiles | 说明 |
|---|---|---|
| 0 | 4 | 只算对角线那 256×256 |
| 1 | 8 | |
| k | (k+1)*4 | 线性增长 |
| 末块 (num_q_blocks-1) | num_kv_tiles | 算整条 KV |

**同一 (head,batch) 内,不同 q_block 的工作量从 4 → num_kv_tiles 线性铺开。** 例：B1 S8192 H64 → num_q_blocks=32、num_kv_tiles=128，则 q_block0 做 4 tile、q_block31 做 128 tile，**相差 32×**。一个 head 的 32 个 q-block 总 tile = 4·(1+2+…+32)=2112，约为 non-causal(32·128=4096) 的 52%。
→ **这是最大的负载不均来源**：哪个 CU 拿到高 q_block_idx 的 block，谁就是拖尾。

### 2b. non-causal —— block 间均衡
所有 q_block 都做 `num_kv_tiles` 个 tile，每个 block 工作量相同（仅末尾 partial tile 略有差别），block 间天然均衡。

### 2c. partial / padded tile（小 seqlen 或非对齐）
- `max(4, …)`：seq_len 小到 tile<4 时补到 4，多出的 tile 全越界（num_records 读 0 + mask），**算的是空转**，对极小 seqlen 是固定开销。
- 末尾 q-block（行 q_start..q_start+255 超过 seq_len）和末尾 kv-tile 由 Q/O 的 num_records 边界 + padding mask 处理，工作量与满 tile 基本相同（不会更省）。

### 2d. split-K —— 把 KV 维再切成 NUM_KV_SPLITS 份（缓解 2a + 填满小 grid）
`grid_z = batch*NUM_KV_SPLITS`，block 多出 `split` 维。每个 split 只做一段 KV tile（行 409-426）：
```
chunk        = round_up_even(ceil(max_num_tiles / NUM_KV_SPLITS))，且 ≥6
split_t0     = split_idx * chunk
split_t_end  = min(split_t0+chunk, max_num_tiles)；尾部不足 4 tile 折回上一个 split
split_nonempty = split_t0 + 4 <= max_num_tiles    # 否则整个 split 空转只写 0
```
- 作用：把"一个重 block(高 q_block)"拆成多个轻 block → grid 变大、单 block 变轻 → **小 batch/小 head 数时能填满 GPU**（占用率上来）。
- 代价：① 需要 combine kernel 归约；② causal 下低 q_block（tile 很少）会有**空 split**（`split_nonempty=False`，整 block 几乎只写 0），这些是浪费的 block。
- 适用：**小 grid（小 batch、小 head）**；大 grid（如 B16 S8192 H64，grid 已 2048 block）本就填满，split-K 反而增加 combine 开销，不划算。

---

## 3. combine kernel：grid → block（均衡，按输出行平铺）

仅 split-K 启动（行 1652-1759，launch 行 1812-1818）：
```
COMBINE_BLOCK          = 256 线程
COMBINE_ROWS_PER_BLOCK = 256 / (HEAD_DIM/4) = 256/32 = 8     # 每 block 处理 8 个输出行
combine_rows           = batch * NUM_HEADS_Q * seq_len        # 总输出行 = B*H*S
grid = (combine_rows / 8, 1, 1) ; block = (256,1,1)
```
block 内（行 1670-1684）：
```
row = blk*8 + tid//32        # 该线程负责的 (b,h,s) 输出行
col = (tid%32)*4             # 32 lane/行，每 lane 4 列；正好覆盖 HEAD_DIM=128
b,h,s = 从 row 解 (row // (S*H), …)
```
每行：读 `NUM_KV_SPLITS` 份 partial(O_s,m_s,l_s) → `out = Σ w_s·O_s / Σ w_s·l_s`，`w_s=exp2(m_s−m_max)`。

**负载特征：完全均衡。** 每个 block 固定处理 8 个输出行 × `NUM_KV_SPLITS` 份归约，与 b/h/s 无关、与 causal 无关。唯一的小差别：空 split（causal 尾部）走 `nonempty=False` 分支跳过 O 读取（行 1722-1741），略省。combine 是访存型小 kernel，通常不是瓶颈，但 split 数越大它越重。

---

## 4. 怎么分析负载均衡（实操指引）

**主 kernel 关注点（按重要性）：**
1. **causal 三角形不均**：block 工作量 ∈ [4, num_kv_tiles] tile，跨 q_block_idx 线性。看 grid 的 y 维跨度 `num_q_blocks=ceil(S/256)` —— S 越大跨度越大、首尾差越大（S8192 → 32×）。
2. **block→CU 分配**：总 block 数 = `NUM_HEADS_Q · ceil(S/256) · batch[· NUM_KV_SPLITS]`。和 CU 数(MI355X)比：
   - block 数 ≫ CU 数 → 硬件轮转能把轻重 block 混合摊开，三角不均被平均（大 grid，如 B16 S8192）。
   - block 数 ≈ 或 < CU 数 → 每 CU 只 1 个 block，谁拿到重 block 谁拖尾，**且 split-K 才能填满**（小 grid，如 B1/B2、小 head）。
3. **每 block 内部**：8 waves × 32 q 行，占用率受 VGPR/LDS 限制（看 `; Occupancy:` / waves_per_eu）。
4. **空转**：小 seqlen 的 `max(4,…)` padding tile、split-K 的空 split。

**量化每个 block 工作量**：`tiles(h,q,b,split)` 见 §2 公式；非 split-K 时 `tiles = min((q_block+1)*4, ceil(S/64))`(causal)。把它当作 block 的"重量"，再叠加 block→CU 映射就能估每个 CU 的总 tile 负载与拖尾。

**用工具实测**（playbook Stage 2）：跑 rocprofv3 ATT 拿 `occupancy.json`，用 `occupancy_balance.py` 按 `(se,cu,simd,slot)` 聚合，看：
- `max slots/SIMD`：=1 说明 grid-limited（1 wave/SIMD，无法靠并发藏延迟）；≥2 说明同 CU 上多 block 并存。
- busy CU 的 `end_span=max(end)−min(end)`、`imbalance=end_span/median_dur`：大 → 有 straggler CU（多半是吃到高 q_block 的重 block）。
- 对比 split-K 开/关、不同 NUM_KV_SPLITS 下的均衡变化。

**经验法则**：
- 大 grid（block 数 ≫ CU）+ causal：三角不均被轮转平均，整体接近满载；split-K 通常无益（多 combine 开销）。
- 小 grid（小 batch/head）：grid-limited、CU 跑不满 → **split-K 是主要均衡手段**（把 KV 维拆出更多 block 填 CU）。
- combine kernel 本身均衡，关注其总量随 NUM_KV_SPLITS 线性增长，别让 split 开太大。

---

## 5. 实测：B1 S8192 H64 causal bf16（rocprofv3 ATT + occupancy_balance.py）

**复现命令**
```
# 配置: python tests/kernels/test_flash_attn_fwd.py --causal --dtype bf16 \
#   --batch 1 --num_heads 64 --num_kv_heads 64 --seq_len 8192 --head_dim 128 --iters 100 --compare
# trace: FlyDSL/thread_trace/else0/ui_output_agent_*/occupancy.json （ATT att_target_cu:1, shader_engine_mask 0xf）
python3 occupancy_balance.py thread_trace/else0/ui_output_agent_*/occupancy.json
```

**结果**
```
flash_attn_dualwave_swp_gfx950_kernel_0
  采样: 4 SE, 32 个物理 CU(se,cu), 32 个都 busy, ts span=1,502,992
  Task 1a  max slots/SIMD = 2        (128 个 SIMD 全部 = 2)
  Task 1b  duration min/med/max = 1,270,232 / 1,386,412 / 1,502,800 cycles
           end-span = 231,284   imbalance = end_span/med = 0.17
           over-subscribed CU(并存≥2 block) = 32/32, serial-reuse = 0
           stragglers: SE3 CU1(1.50M) / SE0 CU6 / SE3 CU2
```

**结论：基本均衡，约 17% 尾部**
1. **占用率好、非 grid-limited**：`max slots/SIMD = 2`，每 SIMD 同时跑 2 个 block，能靠并发藏延迟。符合 §4 预期——B1 S8192 grid=2048 block ≫ CU 数，每 CU 拿多个 block 并存。
2. **负载基本均衡**：`imbalance = 0.17`（最慢 CU 比中位数多耗 17%，最慢/最快 = 1.50M/1.27M ≈ 1.18×），不是严重不均。
3. **17% 尾部 = causal 三角负载残留**：高 q_block 的重 block 未被完全摊平；但大 grid 轮转已把大部分轻重 block 平均掉，只剩中等尾部——正好印证 §4「大 grid → 三角被平均」。
4. **不需要 split-K**：grid 已大、占用已满，split-K 反增 combine 开销（与 §4 经验法则一致）。

**caveat**：`att_target_cu:1` 只采样 32 个 CU（MI355X 全部 CU 的子集），playbook 说明此采样**仅作相对判断**；单次 dispatch、random 数据(else0)。

---

## 6. WG 内 wave→q_seq 行映射（dualwave kernel vs 手写 ASM kernel 对比）

一个 WG = 512 thread = 8 wave，分两组：组A=wave0-3、组B=wave4-7。两个 kernel 用**同一种**映射：按 `wave_id` 把 BLOCK_M=256 行 Q 切成 8 个**连续 32 行条带**。

**dualwave (`flash_attn_gfx950.py`)**
```
BLOCK_M=256, NUM_WAVES=8, ROWS_PER_WAVE=32   # 行 124/130/132
wave_id = tid // 64                          # 行 264
q_row = q_start + wave_id*32 + (lane%32)      # 行 277 / 1081-1083
_stagger = wave_id // 4                       # 行 274  -> 组A=0, 组B=1，组B多1个s_barrier(1097-98)做流水相位错位
```

**ASM (`fmha_fwd_..._msk1_gm0.s`)** —— 输出地址即计算归属（`.s:2883-2902`）
```asm
s_mul_i32 s40, s5, s79     ; s5=wave_id(=tid>>6), s79=r_Seqs
s_mul_i32 s40, s40, 32     ; wave 行基址 = wave_id*32*row_stride
v_add_u32 v22, ...         ; 本 wave 前16行
v_add_u32 v23, v22+16行    ; 后16行（MFMA 32x32 把32行拆16+16）
```
（`.s:223-224` 的 `2*wave_id*Seqs` 是 Q DMA-to-LDS 的协作搬运分布，**不是**计算归属，勿据此推映射。）

**块内 wave→行 与 配对关系（两 kernel 相同）**

| wave | q 行(块内偏移) | 组 |
|---|---|---|
| 0 | 0–31 | A（上半） |
| 1 | 32–63 | A |
| 2 | 64–95 | A |
| 3 | 96–127 | A |
| 4 | 128–159 | B（下半） |
| 5 | 160–191 | B |
| 6 | 192–223 | B |
| 7 | 224–255 | B |

- **wave_k 与 wave_{k+4} 的 q_seq 位置正好相差 128 行（= BLOCK_M/2）**：`q_row(wave_{k+4}) = q_row(wave_k) + 128`。
- 即 **组A(wave0-3) = Q 块上半 128 行，组B(wave4-7) = 下半 128 行**；两组在各自半区是同样相对位置、整体平移 128 行。不是镜像/交错/重复，是"连续二等分，组B = 组A + 128"。
- **两个 kernel 都把这两组做成软件流水的两个相位（ping-pong），只是实现不同**：
  - **dualwave**：`wave_id//4` 分组（行274），组B 多执行 1 个 `s_barrier`（`_stagger_extra_barrier_if_one`，行1097-98）打开相位差；配合 `DUALWAVE_SWP_SETPRIO`（行100）。两组走同一份代码、靠"多一个 barrier"错开半拍。
  - **ASM**：更显式——`.s:770` `s_cmp_lt_i32 s5, 4` 把 wave0-3 / wave4-7 **分流到两份各自独立的 KV 循环代码**：组A 落到 `label_048C`（行796，`s_setprio 0x0000` @794），组B 跳到 `label_0A6F`/`label_0A71`（行1690/1693，`s_setprio 0x0001` @1691）。两份代码把 `buffer_load`(V DMA)/`ds_read`(K,V←LDS)/`v_mfma` 摆在相对共享 `s_barrier` 的不同位置 → 一组做 MFMA 时另一组搬数据，且两组 scheduler 优先级不同（0 vs 1）。
  - **结论修正**：ASM **并非**"8 wave 同相推进"，它对两组做了**比 dualwave 更激进的相位错位 + 优先级区分**（整份循环体复制两份 vs 仅多一个 barrier）。
- **共同点**：q_seq 空间划分规则一致（组B = 组A + 128 行）；两组都被错开成流水两相位。

---

### 关键代码位置速查
- grid 计算 / launch：`launch_flash_attn_dualwave_swp` 行 1762-1811；combine launch 行 1812-1818。
- 主 kernel block 解码：行 254-286；KV tile 数 / causal / split：行 388-426。
- combine kernel：行 1652-1759（block 解码 1670-1684，归约 1711-1746）。
