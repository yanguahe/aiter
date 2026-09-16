# exactopt 与 all-NT-RT 的四-wave TDM ownership 对比及移植方案

## 1. 分析范围与结论

本文对比以下两个 GEMM1 kernel 的 input TDM 组织：

- FlyDSL `sync_mg4_fc28_apre_exactopt`
- 纯汇编 `persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt`

目标 shape 固定为：

```text
M tile       = 256
N tile       = 256
K tile       = 256
K            = 7168
K tiles/task = 28
workgroup    = 4 wave32
cluster      = 4x4
```

核心结论如下。

1. all-NT-RT 仍是 production WPT1 ownership：

   ```text
   wave0 = 完整 A
   wave1 = 完整 B
   wave2 = 完整 ScaleA
   wave3 = 完整 ScaleB
   ```

   每个 K256 stage 的 logical destination bytes 是 `32/32/2/2 KiB`。虽然四个
   wave 都只发一条 input TDM，但 descriptor 搬运量相差最多 `16x`。

2. exactopt 是 GEMM1-only WPT2 ownership：

   ```text
   wave0 = A lower half + ScaleA lower half
   wave1 = A upper half + ScaleA upper half
   wave2 = B lower half + ScaleB lower half
   wave3 = B upper half + ScaleB upper half
   ```

   每个 wave 每个 K256 stage 都搬 `16 KiB + 1 KiB = 17 KiB`，并各发两条
   input TDM。workgroup 总搬运量仍为 `68 KiB/stage`，没有减少 HBM/GL2 数据量；
   改善来自把长 payload descriptor 拆给两个 wave，缩小 wave 到达 barrier 的偏差。

3. 两次 ATT 中观察到相同的 logical-wave 到 physical-SIMD 排列。按 gfx1250
   “每个 SIMD pair 共享一个 TDM”的结构计算，两版每个 SIMD pair 都承担
   `34 KiB/stage`，因此差异不是 pair 总字节数，而是 pair 内两个 wave 的分配：

   - all-NT-RT：`32 + 2 KiB`；
   - exactopt：`17 + 17 KiB`。

   `TENSORcnt` 是 per-wave counter，workgroup/cluster barrier 又要等所有参与者到达，
   所以 pair 总流量相等并不能消除单 wave 的完成时间偏差。

4. exactopt 的 steady cluster ring-wrap barrier 为 `825.7 cycles/wave`、占
   `2.59%`；all-NT-RT 的六次 K-ring cluster barrier 中位数为
   `2,044 cycles/task`、占 `7.31%`。这与 WPT2 更均衡的 ownership 一致。
   但两份 trace 来自不同 kernel、不同机器和不同统计口径，因此不能把全部
   `1.2~1.3k cycles/task` 差值都归因于 WPT2。

5. 移植时不应照搬 exactopt 的 stage-major LDS arena。最安全的做法是在
   all-NT-RT 现有每个 A/B/Scale ring slot 内做 half split，使每个 slot 的地址并集、
   output LDS 选择、next-task stage-0 prefetch 和跨 task output overlap 全部不变。

## 2. 核对使用的版本与证据

本地仓库 HEAD：

```text
6059ea13149d2bb853f5217582bf868d50a64831
```

分析的 ISA：

```text
my_code/gemm1_a4w4.sync_mg4_fc28_apre_exactopt.s
SHA256=a895c9b97307629ffb345c10671890dfc9207cd07a388fd959c8fd548698d8f4a

my_code/moe_gemm1_act1_optimized/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s
SHA256=8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169
```

trace 数据来自：

```text
my_code/gemm1_exactopt_a07_thread_trace_optimization_plan.md

my_code/moe_gemm1_act1_optimized/
  PERSISTENT_OVERLAP_PAD8_PREFETCH_STAGE0_ALL_NT_RT_THREAD_TRACE_OPTIMIZATION_PLAN.md
```

FlyDSL 逻辑使用 HEAD 中的以下文件进行只读核对：

```text
aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
```

没有修改该文件。

`reproduce_compare.sh` 对 exactopt 设置：

```bash
AITER_FLYDSL_GEMM1_MMA_GROUP=4
AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28
AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1
AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2
AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0
AITER_FLYDSL_GEMM1_WMMA_REUSE=1
AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE=1
```

对应 symbol 为：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p
```

## 3. 一个 K256 stage 的数据量

两版处理相同的 A/B/Scale tile。按 descriptor 的 logical tile 计算：

| tensor | descriptor shape | 元素大小 | 每 stage 字节数 |
|---|---:|---:|---:|
| A | `16 x 2048` | 1 B | `32768 B = 32 KiB` |
| B | `16 x 2048` | 1 B | `32768 B = 32 KiB` |
| ScaleA | `8 x 64` | 4 B | `2048 B = 2 KiB` |
| ScaleB | `8 x 64` | 4 B | `2048 B = 2 KiB` |
| 合计 | — | — | `69632 B = 68 KiB` |

其中：

- A/B 的一个 outer unit 对应 16 个 M/N rows，大小为 `2048 B`；
- ScaleA/ScaleB 的一个 outer unit 对应 32 个 M/N rows，大小为
  `64 dwords = 256 B`；
- `K=7168` 时，A/B 相邻 outer unit 的 global stride 是
  `Kp16=(7168/2)*16=57344 B=0xe000`；
- scale 相邻 outer unit 的 global stride 是 `K/4=1792 dwords=7168 B`。

这里统计的是每个 workgroup LDS 最终得到的 logical bytes。multicast 会减少完整
cluster 的重复 global fetch，但不会改变每个 workgroup 的 LDS tile 大小。

## 4. exactopt 的四-wave TDM 组织

### 4.1 源码 ownership

HEAD 源码先按 `num_waves_per_tensor_tdm=2` 生成：

```text
wave_groups = [(0, 1), (2, 3)]
```

随后把 A 与 ScaleA 分配给第一组，把 B 与 ScaleB 分配给第二组：

```text
data_waves[0] = waves 0/1 -> A
waves[2]      = waves 0/1 -> ScaleA

data_waves[1] = waves 2/3 -> B
waves[3]      = waves 2/3 -> ScaleB
```

`emit()` 对多 owner job 默认切 outer dimension：

```text
seg            = outer / owner_count
wave_outer_off = (wave - first_owner) * seg
```

ScaleA 的 `split_inner` 条件在本 shape 下为 false，因此四类 tensor 都沿 outer
dimension 切半。

### 4.2 每个 logical wave 的准确范围

| logical wave | payload TDM | scale TDM | payload LDS 相对 offset | scale LDS 相对 offset | 每 stage 合计 |
|---:|---|---|---:|---:|---:|
| 0 | A outer `[0,8)`，即 M `[0,128)` | ScaleA outer `[0,4)`，即 M `[0,128)` | `A+0x0000` | `SA+0x000` | `17 KiB` |
| 1 | A outer `[8,16)`，即 M `[128,256)` | ScaleA outer `[4,8)`，即 M `[128,256)` | `A+0x4000` | `SA+0x400` | `17 KiB` |
| 2 | B outer `[0,8)`，即 N `[0,128)` | ScaleB outer `[0,4)`，即 N `[0,128)` | `B+0x0000` | `SB+0x000` | `17 KiB` |
| 3 | B outer `[8,16)`，即 N `[128,256)` | ScaleB outer `[4,8)`，即 N `[128,256)` | `B+0x4000` | `SB+0x400` | `17 KiB` |

upper-half owner 的 global pointer 增量为：

```text
A/B    : 8 * 0xe000 B = 0x70000 B
SA/SB  : 4 * 0x700 dwords * 4 B = 0x7000 B
```

最终 ISA 与该推导一致：

- `s33 < 64` 选择 logical waves 0/1；`s33 > 63` 选择 waves 2/3；
- A/B half 的 LDS offset 使用 `(wave_in_group << 14)`，即 `0` 或 `0x4000`；
- SA/SB half 的 LDS offset 使用 `(wave_in_group << 10)`，即 `0` 或 `0x400`；
- 每条 owner path 按 `payload -> scale` 的顺序发出两条
  `tensor_load_to_lds`。

### 4.3 exactopt 的 LDS layout

exactopt 使用 stage-major 的 `PITCH=0x11000`：

| ring stage | A base | B base | ScaleA base | ScaleB base | stage end |
|---:|---:|---:|---:|---:|---:|
| 0 | `0x00000` | `0x08000` | `0x10000` | `0x10800` | `0x11000` |
| 1 | `0x11000` | `0x19000` | `0x21000` | `0x21800` | `0x22000` |
| 2 | `0x22000` | `0x2a000` | `0x32000` | `0x32800` | `0x33000` |
| 3 | `0x33000` | `0x3b000` | `0x43000` | `0x43800` | `0x44000` |

总 input arena 为：

```text
4 * 0x11000 = 0x44000 = 278528 B = 272 KiB
```

这与 exactopt ISA 的 `.amdhsa_group_segment_fixed_size 278528` 一致。

### 4.4 request 数和 wait 方式

每个 wave 每个 K256 stage 发两条 TDM，因此：

```text
per wave/task = 28 * 2 = 56 input TDM requests
per WG/task   = 28 * 8 = 224 input TDM requests
```

FlyDSL 中 `TDM_PER=2`。最终 ISA 可见的主要 threshold 是：

```text
initial four-stage queue : s_wait_tensorcnt 0x6
steady refill           : s_wait_tensorcnt 0x4
drain                    : s_wait_tensorcnt 0x4 / 0x2 / 0x0
```

这些数值依赖 exactopt 自己的 issue 顺序，不能原样全局替换到 persistent kernel。

### 4.5 multicast

- A/ScaleA 使用同一 cluster row 的 N-direction mask；
- B/ScaleB 使用同一 expert 内的 M-direction column mask；
- descriptor 保留 `early_timeout`；
- 每个 logical wave 的两个 job 具有相同的 multicast peer 集合。

## 5. all-NT-RT 的四-wave TDM 组织

### 5.1 当前 ownership

纯汇编通过 `s22` 选择四条独立路径：

```asm
s_cmp_eq_u32 s22, 0  ; A
s_cmp_eq_u32 s22, 1  ; B
s_cmp_eq_u32 s22, 2  ; ScaleA
s_cmp_eq_u32 s22, 3  ; ScaleB
```

owner marker 和 LDS base 分别是：

```text
A      : s_mov_b32 s95, 0
B      : s_mov_b32 s95, 0x30000
ScaleA : s_mov_b32 s95, 0x10000
ScaleB : s_mov_b32 s95, 0x22000
```

每个 K256 stage 的分配为：

| logical wave | TDM job | requests/stage | bytes/stage | bytes/task，28 stages |
|---:|---|---:|---:|---:|
| 0 | 完整 A | 1 | `32 KiB` | `896 KiB` |
| 1 | 完整 B | 1 | `32 KiB` | `896 KiB` |
| 2 | 完整 ScaleA | 1 | `2 KiB` | `56 KiB` |
| 3 | 完整 ScaleB | 1 | `2 KiB` | `56 KiB` |
| WG 合计 | A+B+SA+SB | 4 | `68 KiB` | `1904 KiB` |

四个 wave 的平均值是 `17 KiB/stage`，但标准差为 `15 KiB`，变异系数约
`88.2%`。问题是 descriptor payload 长度不均衡，不是 descriptor 条数不均衡。

all-NT-RT 的所有 input load 均保留：

```asm
tensor_load_to_lds ... th:TH_LOAD_NT_RT
```

该 cache hint 是 all-NT-RT 已验证优化的一部分，与 WPT2 ownership 是两条独立的
优化轴。

### 5.2 当前 LDS layout

all-NT-RT 没有使用 exactopt 的 stage-major `0x11000` pitch，而是保留了用于
persistent output overlap 的 tensor-major/稀疏布局：

| tensor | stage 0 | stage 1 | stage 2 | stage 3 | 每槽大小 |
|---|---:|---:|---:|---:|---:|
| A | `0x00000` | `0x08000` | `0x12000` | `0x1a000` | `0x8000` |
| ScaleA | `0x10000` | `0x10800` | `0x11000` | `0x11800` | `0x800` |
| ScaleB | `0x22000` | `0x22800` | `0x23000` | `0x23800` | `0x800` |
| B | `0x30000` | `0x38000` | `0x40000` | `0x48000` | `0x8000` |

LDS metadata 为：

```text
.amdhsa_group_segment_fixed_size 327680  # 0x50000, 320 KiB
.amdhsa_next_free_vgpr 1024
.amdhsa_next_free_sgpr 104
```

额外的地址空间和 A/B ring base 还服务于 per-wave output LDS placement、两个
64-row output half 以及跨 persistent task 的 output drain overlap。直接把该布局
替换成 exactopt 的 `4 * 0x11000` 会改变 output alias/lifetime，不能这样移植。

### 5.3 current-task 与 next-task 的顺序

all-NT-RT 在当前 task 的 K hotloop 结束后先完成 output descriptor/address setup，
然后在 SiLU 前发出下一 task 的 stage-0 input TDM；随后才发出当前 task 的两个
output TDM：

```text
next task stage-0 input
  -> current task SiLU + first output store
  -> remaining SiLU + second output store
  -> persistent task boundary
```

当前每 wave 的跨 task TDM FIFO 逻辑可抽象为：

```text
[next I0, current O0, current O1]
```

下一 task 入口的 `s_wait_tensorcnt 0x2` 只要求队首的 `next I0` 完成，允许两个
output store 继续后台执行。后续在第一次可能复用旧 output LDS 的 input issue 前，
现有代码再收紧 wait。这是 `persistent_overlap` 收益的关键，不能改回 task 边界
`s_wait_tensorcnt 0`。

## 6. physical SIMD pair 对比

MI400 Shader Programming Guide §1.4.2.1/§1.5 说明每个 SIMD pair 共享通往
LDS/WGP$ 的 bus 和一个 TDM。两次 ATT 都观察到：

| logical wave | physical SIMD selection |
|---:|---:|
| 0 | SIMD0 |
| 1 | SIMD3 |
| 2 | SIMD2 |
| 3 | SIMD1 |

这是 capture 中的实际映射，不应当表述成所有 dispatch 的架构固定映射。按该映射，
两个 physical SIMD pair 的 input 工作为：

| physical pair | all-NT-RT | exactopt | pair 总量 |
|---|---|---|---:|
| SIMD0 + SIMD1 | wave0 A `32 KiB` + wave3 ScaleB `2 KiB` | wave0 A/SA half `17 KiB` + wave3 B/SB half `17 KiB` | `34 KiB` |
| SIMD2 + SIMD3 | wave2 ScaleA `2 KiB` + wave1 B `32 KiB` | wave2 B/SB half `17 KiB` + wave1 A/SA half `17 KiB` | `34 KiB` |

因此 WPT2 没有降低一个 TDM pair 的总字节数。它改变的是：

- 长 descriptor 不再集中在单个 wave；
- 两个 wave 都有相近的 TDM completion path；
- TDM 可以在多个 descriptor/wave 间调度，而不是让一个 wave 持有完整 32 KiB job；
- per-wave `TENSORcnt` 完成点更接近，后续 WG/cluster barrier 到达时间也更接近。

硬件资料还说明：同一 wave 的 tensor instructions 按序完成，不同 wave 之间无序；
每 wave 从 issue 到 XACK 最多 3 条、每 SIMD 最多 6 条，完成计数 `TENSORcnt` 为
6 bit、最多 63。exactopt 中大于 3 的 `TENSORcnt` threshold 并不违反前一限制：
issue-to-XACK slot 通常早于完整 data movement 完成而释放。

## 7. thread trace 说明了什么

| 指标 | exactopt | all-NT-RT |
|---|---:|---:|
| 代表性周期口径 | `31,915 cycles/active wave` 平均值 | `27,951 cycles/steady task` cross-owner 中位数 |
| 全部 barrier stall | `2,540.3 cycles`, `7.96%` | `4,083.5 cycles`, `14.61%` |
| steady K-ring cluster barrier | `825.7 cycles`, `2.59%` | `2,044 cycles`, `7.31%` |
| `TENSORcnt` stall | `1,003.7 cycles`, `3.15%` | `858 cycles`, `3.07%` |
| `DScnt` stall | `580.9 cycles`, `1.82%` | `1,908 cycles`, `6.83%` |

steady K-ring barrier 的 absolute gap 为：

```text
2044.0 - 825.7 = 1218.3 cycles/task
```

若只按 all-NT-RT 的占比从 `7.31%` 降到 `2.59%` 估算，窗口约为：

```text
27951 * (7.31% - 2.59%) = 1319 cycles/task
```

所以这个方向的理想机会约为 `1.2~1.3k cycles/task`，即当前 steady task 的
`4.4%~4.7%`。这不是可直接兑现的性能承诺，原因包括：

- exactopt 是 standard-grid FlyDSL kernel，all-NT-RT 是 16x16 persistent kernel；
- 两份 trace 分别来自 a07-3 和 d01-3；
- exactopt 使用 active-wave mean，all-NT-RT 报告以 robust task median 为主；
- exactopt 同时具有不同的 code layout、VGPR/LDS 使用和 output ownership；
- WPT2 把每 stage 的 TDM instruction 数从 4 增加到 8，可能增加 SALU、issue 和
  `TENSORcnt` 压力。

因此正确结论是：WPT1 的 `32/32/2/2 KiB` per-wave 分配是 all-NT-RT barrier
到达偏差的重要结构性来源，WPT2 是有充分依据的候选；现有 trace 不能证明移植后
一定获得完整的 `4.4%~4.7%`。

all-NT-RT 自身的 owner 数据也支持这一判断：

| owner | K-ring cluster barrier | `TENSORcnt` |
|---|---:|---:|
| A | `1981 cycles/task` | `1047 cycles/task` |
| B | `2107 cycles/task` | `915 cycles/task` |
| ScaleA | `2567 cycles/task` | `450 cycles/task` |
| ScaleB | `1478 cycles/task` | `801 cycles/task` |

Scale owner 的 payload 虽小，仍可能因为 descriptor setup、issue 时点、同 pair TDM
竞争和其它路径长度成为较晚到达者。因此优化目标应是缩短最慢 wave 的 critical
path，而不是只让某个提前到达 wave 的 barrier 数字变小。

## 8. 目标 ownership：在现有 all-NT-RT LDS 内完成 WPT2 split

不迁移 exactopt 的 stage-major arena。对 all-NT-RT 的每个现有 slot 原地切分：

| wave | tensor | stage `r` 的 LDS destination | global outer offset | descriptor outer |
|---:|---|---|---:|---:|
| 0 | A lower | `A_RING[r] + 0x0000` | `0` | `8` |
| 0 | ScaleA lower | `SA_RING[r] + 0x000` | `0` | `4` |
| 1 | A upper | `A_RING[r] + 0x4000` | `0x70000 B` | `8` |
| 1 | ScaleA upper | `SA_RING[r] + 0x400` | `0x7000 B` | `4` |
| 2 | B lower | `B_RING[r] + 0x0000` | `0` | `8` |
| 2 | ScaleB lower | `SB_RING[r] + 0x000` | `0` | `4` |
| 3 | B upper | `B_RING[r] + 0x4000` | `0x70000 B` | `8` |
| 3 | ScaleB upper | `SB_RING[r] + 0x400` | `0x7000 B` | `4` |

其中：

```text
A_RING  = [0x00000, 0x08000, 0x12000, 0x1a000]
B_RING  = [0x30000, 0x38000, 0x40000, 0x48000]
SA_RING = [0x10000, 0x10800, 0x11000, 0x11800]
SB_RING = [0x22000, 0x22800, 0x23000, 0x23800]
```

每个原 slot 的区间并集完全不变：

```text
[A, A+0x4000) U [A+0x4000, A+0x8000) = [A, A+0x8000)
[S, S+0x0400) U [S+0x0400, S+0x0800) = [S, S+0x0800)
```

这意味着：

- DS consumer 地址一条也不需要改；
- A/B/Scale tile 的最终 LDS 内容不变；
- 320 KiB allocation 不变；
- 现有 output LDS base 和 `0x4800` per-wave output region 不变；
- current output 与 next-task stage-0 input 的地址并集关系不变；
- 不需要增加第五个 ring slot，也不需要移动 output buffer。

## 9. 详细实施方案

### 9.1 新建候选，不覆盖 winner

以固定 SHA 的 all-NT-RT 为唯一输入：

```text
source:
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt.s
  SHA256=8934b9767fb8295b7d1bf3c5df246fbb239a2784c76f11873f1ceb490d9f0169

candidate:
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_
  exactopt_wpt2_owner.s

builder:
  build_all_nt_rt_exactopt_wpt2_owner.py
```

仓库内已有 `build_wpt2_owner_balance_variant.py` 可复用 descriptor split 的思路，
但不能直接作为最终生成器：它固定在较早的 `...iprefetch_full.s` 上，而目标必须保留
all-NT-RT 的全部 `TH_LOAD_NT_RT` 和当前 persistent tail。

### 9.2 原子地改四条 owner path

把当前分支：

```text
wave0 -> A
wave1 -> B
wave2 -> ScaleA
wave3 -> ScaleB
```

整体改成：

```text
wave0 -> A half0 -> ScaleA half0
wave1 -> A half1 -> ScaleA half1
wave2 -> B half0 -> ScaleB half0
wave3 -> B half1 -> ScaleB half1
```

不能先部署只改 A 或只改 B 的版本。历史实验中 A-only split 虽然 random 正确但
回退，B-only cross-owner 版本发生过无法完成的 GPU launch。四个 wave、四个 stage、
steady refill、drain 和 next-task tail 必须作为一个一致的 multicast protocol 修改。

每条 path 始终按 exactopt 的顺序发：

```text
payload half first
scale half second
```

并保持 A/ScaleA 的 row mask、B/ScaleB 的 same-expert column mask，以及 bit 21
`early_timeout`。新增的每条 input TDM 都继续带 `th:TH_LOAD_NT_RT`。

### 9.3 descriptor register 规划

all-NT-RT 已使用 `104` 个 numbered SGPR，只剩 `s104:s105`，不能新增一套长期
12-SGPR descriptor。

推荐做法：

1. K hotloop 内继续用 `s32:s43` 保存 payload descriptor；
2. 复用 hotloop 期间尚未承载 output descriptor 的 `s80:s91` 保存 scale descriptor；
3. payload 每 stage 前进 `0x800 B`，scale 每 stage 前进 `0x100 B`；
4. K hotloop 结束后，按现有位置重新构造 `s80:s91` output descriptor；
5. next-task stage-0 prefetch 发生在 output descriptor 已经 live 的区域，不能覆盖
   `s80:s91`。该处用 `s32:s43` 依次构造并 issue payload half、再覆盖为 scale half
   并 issue。TDM issue 会采样 descriptor，第二次构造不需要长期保存第一套值；
6. 进入下一 task 后，在现有 descriptor-ready 路径重新恢复 payload 与 scale 的
   stage-1 state，然后进入原 hotloop。

此方案不增加 VGPR，不增加 metadata SGPR，并保留 output descriptor 生命周期。

### 9.4 issue placement

exactopt 的两个 input request 在语义上是 `payload -> scale`，但机器 ISA 没有把它们
压成完全相邻的 burst；中间穿插了独立 WMMA/SALU。all-NT-RT 也应采用相同原则：

1. 在原 payload `tensor_load_to_lds` 位置发 payload half；
2. 在后续第一个不依赖 descriptor SGPR、且不越过现有 barrier 的 `mg4` WMMA slot
   发 scale half；
3. 第一版不移动原 payload issue；
4. 第一版不提前 B-side 越过原 K-ring barrier；
5. 第一版不改变任何 DS load 或 WMMA 顺序。

这样可以避免两条 TDM 连续 setup/issue 抢占同一 scalar/VMEM scheduling window，
也不会把 ownership 优化与 hotloop 重排混在一次实验里。

实施后的 a07-3 ablation 显示这一 stagger 版本反而比相邻发射版本慢约 `1.0%`：
`517.920 us` 对 `512.701 us`。因此最终 ISA 保留 `payload -> scale` 的同-wave
顺序，但取消中间的单-WMMA stagger，以实测更快的相邻 issue 作为最终版本。

### 9.5 `TENSORcnt` 按队列语义重算

不能执行全局 `0x2 -> 0x4` 替换。每个 wait site 必须按它前面的动态 request 队列
分类。

纯 input K-ring 的预期换算为：

| 位置 | WPT1 队列 | WPT2 队列 | 预期 threshold |
|---|---:|---:|---:|
| 首 task，stage0/1/2 已 issue，准备读 stage0 | 3 | 6 | `2 -> 4` |
| steady refill，保留后续两个 stage | 3 左右 | 6 左右 | `2 -> 4` |
| drain | 按剩余 stage 递减 | 每 stage 两条 | 参考 `4 -> 2 -> 0` |

跨 persistent task 的队列不同。WPT2 tail 应保持：

```text
next I0 payload
next I0 scale
current O0
current O1
```

因此下一 task 入口：

```asm
s_wait_tensorcnt 0x2
```

应继续保留。利用同一 wave 的 TDM completion order，它保证队首两个 next-task
input request 已完成，同时允许 `O0/O1` 继续在后台 drain。随后执行现有
workgroup barrier，四个 wave 才能消费完整 stage0。

当前 prefetched task 在发 stage1 后、第一次可能覆盖旧 output LDS 的后续 input
之前使用 `s_wait_tensorcnt 0x1`。WPT2 下队列变为：

```text
[O0, O1, I1 payload, I1 scale]
```

这里应先验证实际 alias site；若目的是只退休 `O0/O1`、保留 stage1 pair，则候选
threshold 应为 `0x2`。之后用于保持两 stage lead 的普通 wait 才改为 `0x4`。

生成器必须为每个 wait site 输出一份静态 ledger：

```text
PC/label
此前可能 outstanding 的 request 顺序
本 wait 需要保证完成的最年轻 request
允许保留的 request 数
最终 immediate
```

没有 ledger 的 wait 不允许修改。

### 9.6 barrier protocol 保持不变

保留现有所有：

```text
s_barrier_signal -1 / s_barrier_wait -1
s_barrier_signal -3 / s_barrier_wait -3
```

具体约束：

- 每个 wave 先等待自己的 payload+scale request 达到消费条件；
- 随后的现有 workgroup barrier 发布本 WG 的完整 A/B/Scale LDS tile；
- 不为两个 half 增加额外 cluster barrier；
- 保留每四个 K stage 的 cluster ring-wrap generation；
- 保留 persistent task boundary 的 cluster generation；
- partial/sentinel cluster 的动态 B/ScaleB mask 和退出路径不变。

multicast 不改变这一原则：每个 WG 的请求与本地完成条件仍需匹配，cluster barrier
负责的是跨 WG ring 生命周期，不用于替代本 WG 的 input-ready rendezvous。

### 9.7 persistent overlap 必须逐项保持

以下行为不得改变：

- `grid=(16,16,1)`、`cluster=(4,4,1)`、`block=(128,1,1)`；
- DeepGEMM persistent task/swizzle 映射；
- 当前 task 末尾预取 next task stage0；
- next stage0 input 与 current SiLU/output TDM 的 overlap；
- 两个 64-row output TDM 的发射点；
- output LDS 的 per-wave base 选择和 `0x4800` region；
- task-boundary `s_wait_idle`/TENSOR wait 的现有安全语义；
- 四阶段 input ring 和六次 steady ring-wrap cluster barrier；
- 184-byte ABI、expert-local pointer、A/ScaleA preshuffle layout；
- exact SiLU、WMMA、DS load/store 顺序和 `TH_LOAD_NT_RT`。

关键安全论证是：WPT2 只改变“哪个 wave 发哪个 half”，所有 half 的 LDS union 与
当前完整 descriptor 的 destination 完全相同。因此只要 request 顺序和 wait
正确，现有 output-vs-stage0 非重叠关系仍成立。

## 10. 静态审计

生成后必须自动检查：

1. source SHA256 精确匹配，防止在变化的基线上静默 patch；
2. 四条 logical-wave path 分别只有预期的 `payload + scale` job；
3. 每个 K256 stage/WG 的 descriptor 数从 4 变成 8；
4. 每个 task 的 dynamic input request 数为 `224`；
5. A/B descriptor outer extent 从 `16` 变为 `8`；
6. SA/SB descriptor outer extent从 `8` 变为 `4`；
7. upper-half global offset 分别为 `0x70000`、`0x7000`；
8. upper-half LDS offset 分别为 `0x4000`、`0x400`；
9. 四阶段所有 half 区间不重叠，union 等于原 slot；
10. 最大 LDS 地址仍小于 `0x50000`；
11. output LDS interval 与 next-task stage0 interval 的关系与 source 完全相同；
12. 所有 input TDM 都带 `TH_LOAD_NT_RT`；
13. A/SA 与 B/SB 的 multicast mask、`early_timeout` 和 OOB bound 正确；
14. barrier instruction 的数量、类型、generation 和相对控制流不变；
15. `.amdhsa_kernarg_size 184`、LDS `327680`、VGPR `1024` 保持不变；
16. numbered SGPR 不超过 `106`，private segment/scratch 仍为 0；
17. output `tensor_store_from_lds` 数和位置不变；
18. instruction-prefetch 覆盖范围随新增代码重新核对，避免新增 hot blocks 落到原
    `iprefetch_full` 覆盖之外。

## 11. 实现与验证顺序

### 阶段 A：生成完整候选并只做静态验证

一次性覆盖以下所有动态路径：

- first-task full setup；
- prefetched-task descriptor restore；
- 四个 initial ring stages；
- steady refill；
- K drain；
- next-task stage-0 tail；
- full/partial/sentinel cluster path。

不运行只改部分 owner 的 persistent candidate，避免 multicast request 序列不匹配。

### 阶段 B：正确性

正确性测试不要求 GPU 空闲。顺序为：

1. 单次 const0，只用于发现 hang、page fault 和明显地址错误；
2. random MoE e2e，至少 seed 0/1/2；
3. 每个 seed 至少连续运行 3 次，检查 nondeterministic race；
4. 比较 `logits_diff`、`rel_l2`、MoE output hash128 和 ref output hash128；
5. 增加非均匀 expert routing，覆盖 B/ScaleB 动态 multicast mask；
6. 连续多 launch，确认没有 barrier generation 漂移或 TDM queue 泄漏。

候选可先通过 `benchmark_history.sh` 的 candidate 入口运行：

```bash
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/\
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_\
exactopt_wpt2_owner.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=1 RUN_VERIFY=1 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-random
```

### 阶段 C：同机性能

性能测试前后按 `AGENTS.md` 检查所有 GPU/KFD。把 source 与 candidate 放在同一次
交错 run 中，至少三轮，GEMM1 以 MoE e2e profiler 行为准：

```bash
AITER_HISTORY_CASE_LIST=persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt,candidate \
AITER_HISTORY_CANDIDATE=my_code/moe_gemm1_act1_optimized/\
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_\
exactopt_wpt2_owner.s \
AITER_HISTORY_CANDIDATE_GRID_X=16 \
AITER_HISTORY_CANDIDATE_GRID_Y=16 \
ROUNDS=3 RUN_VERIFY=0 RUN_ATT=0 \
bash my_code/moe_gemm1_act1_optimized/benchmark_history.sh e2e-const0
```

### 阶段 D：四-SIMD ATT

只有 random 正确且性能有稳定提升后再抓 ATT。比较：

- 六次 steady K-ring cluster barrier 的 cross-owner median；
- 四个 logical wave 的 barrier arrival spread；
- per-wave `s_wait_tensorcnt`；
- `s_wait_dscnt 0x8`；
- 新增第二条 TDM 附近的 issue gap；
- complete task cycles 和 dispatch span。

首个量化目标为：

```text
K-ring cluster barrier: 2044 -> <1600 cycles/task
```

若能接近 exactopt 的绝对值，则目标约为 `800~900 cycles/task`。只有 barrier 降低、
最慢 owner/task 同时缩短且 GEMM1 wall time 改善，才说明 ownership port 有效。

## 12. 风险与停止条件

### 12.1 主要风险

1. **multicast descriptor 次序不匹配**：可导致 cluster hang，而不仅是数值错误。
2. **secondary descriptor 覆盖 output descriptor**：`s80:s91` 只能在其 input-phase
   live hole 内使用，next-task tail 必须改回串行复用 `s32:s43`。
3. **错误的 half pointer/OOB**：const0 可能掩盖问题，必须用 random 和 expert
   boundary 检测。
4. **错误的 TENSORcnt threshold**：过小会丢失 overlap，过大可能在 LDS 尚未完成时
   消费或覆盖数据。
5. **代码尺寸和 instruction fetch 回退**：纯汇编为大规模展开代码，新增第二个
   issue 不能复制整段 descriptor setup 到每个 stage。
6. **B-side 仍是慢侧**：exactopt trace 中 B/ScaleB 的 `TENSORcnt` 仍明显高于
   A/ScaleA；WPT2 只能降低不平衡，不能消除 B multicast/地址路径本身的差异。

### 12.2 停止条件

出现以下任一情况就不晋级候选：

- random/hash 不通过或不同 launch 结果不稳定；
- page fault、GPU hang、barrier generation 不一致；
- SGPR > 106、VGPR > 1024 或产生 scratch；
- steady cluster barrier 降低但 `TENSORcnt`/SALU issue 成本等量或更大地增加；
- 三轮同机 GEMM1 median 没有可重复提升；
- 为获得收益必须删除现有 correctness-critical barrier，或改变 current-output 与
  next-input 的 LDS 生命周期。

## 13. 预期结果

该方案保持总 input bytes、WMMA、DS consumer、output epilogue 和 persistent task
loop 不变，只把 per-wave input payload 从：

```text
32 / 32 / 2 / 2 KiB
```

改为：

```text
17 / 17 / 17 / 17 KiB
```

因此它直接针对 all-NT-RT 当前最大的可优化项——六次 K-ring cluster barrier 的
arrival skew。理想机会窗口约 `4.4%~4.7%` 的 steady-task cycles；现实收益应扣除
新增 TDM instruction、secondary descriptor update、TENSORcnt backpressure 和 code
size 成本。第一版应以 `1%` 以上可重复 GEMM1 提升、random e2e 全通过、K-ring
barrier 降至 `1600 cycles/task` 以下作为继续优化的门槛。

## 14. 硬件与代码依据

- `MI400_Shader_Programming#65.txt` §1.4.2.1、§1.5：每个 SIMD pair 共享
  LDS/WGP$ bus 和一个 TDM。
- 同文档 §4.10.1：同一 wave 的 TDM completion 有序，不同 wave 之间无序；
  `TENSORcnt` 为 per-wave completion counter。
- 同文档 §4.10.3：非零 `workgroup_mask` 使 load 使用 cluster multicast。
- 同文档 §4.10.8：每 wave 最多 3 个 issue-to-XACK tensor ops、每 SIMD 最多 6 个；
  `TENSORcnt` 的 issue-to-completion 范围为 63。
- HEAD `mxfp4_preshuffle_gfx1250_tdm.py` 中的 `wave_groups`、`jobs`、`emit()` 和
  `TDM_PER` 定义。
- `gemm1_a4w4.sync_mg4_fc28_apre_exactopt.s` 中的 WPT2 owner branch、half offset、
  `s_wait_tensorcnt 0x6/0x4/0x2/0x0` 和 metadata。
- all-NT-RT ISA 中的四条 `s22` owner path、ring bases、`TH_LOAD_NT_RT`、
  next-task stage-0 prefetch 与 persistent output tail。

## 15. 实施状态（2026-09-16）

已生成：

```text
my_code/moe_gemm1_act1_optimized/build_all_nt_rt_exactopt_wpt2_owner.py

my_code/moe_gemm1_act1_optimized/
  persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_
  exactopt_wpt2_owner.s
```

候选 ISA SHA256：

```text
dee177fc1c01e2b1c3efea59b5d200fd43d0993d7543a5e8acb9af9de452a358
```

生成器执行的静态检查结果：

```text
tensor_load_to_lds                  = 120 static sites
input TDM using TH_LOAD_NT_RT       = 120
hotloop secondary-scale insertions  = 56
steady staggered scale issues       = 0
old-output wait 0x1 -> 0x2          = 4
input-ring wait 0x2 -> 0x4          = 16
tensor_store_from_lds               = 2
duplicate labels                    = 0
workgroup barrier operations        = unchanged
cluster signal/wait                 = unchanged
LDS                                 = 327680 B
VGPR                                = 1024
numbered SGPR metadata              = 104
code-object sgpr_count              = 106
kernarg                             = 184 B
```

d01-3 和 a07-3 上 clang 汇编与 code-object 链接均通过。最终相邻发射版本在
d01-3、a07-3 的目标规模 random MoE e2e 均通过：

```text
random repeat 1: logits_diff=3.3980e-06, rel_l2=2.6069e-03, pass=True
random repeat 2: logits_diff=3.3980e-06, rel_l2=2.6069e-03, pass=True
MoE output hash128 = 1556fc617347e2dabc9cff19dbfd822b
ref output hash128 = 1a5d22911ba167160b4f2c12092a5193

const0（a07-3）: logits_diff=0, rel_l2=0, pass=True
MoE output hash128 = 21291d9023c8af8a6324fe20f346a967
ref output hash128 = 21291d9023c8af8a6324fe20f346a967
```

验证命令为：

```bash
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u my_code/moe_gemm1_act1_optimized/run_e2e_candidate.py \
  --isa my_code/moe_gemm1_act1_optimized/\
persistent_overlap_pad8_prefetch_stage0_b64_clear_iprefetch_full_all_nt_rt_\
exactopt_wpt2_owner.s \
  --grid-x 16 --grid-y 16 -- \
  --scenario verify --data-format a4w4 \
  --experts 96 --tokens 16384 --topk 6 \
  --model-dim 7168 --inter-dim 3072 \
  --act silu --no-bias --no-check-aot-cache
```

d01-3 当时存在其它 workload，`gpu_busy_percent=13~100`，并曾占用约
`382 GiB/432 GiB` VRAM，因此没有采用 d01-3 的性能数据。

a07-3 在测试前后均确认无 GPU/KFD 使用者、连续三次 `gpu_busy_percent=0`、VRAM
约 `165 MiB`。三轮 MoE e2e const0 结果为：

| 版本 | GEMM1 samples | GEMM1 median | fused MoE samples | fused MoE median | 相对 source |
|---|---|---:|---|---:|---:|
| all-NT-RT source | `506.090, 506.233, 504.650 us` | `506.090 us` | `1333.76, 1332.52, 1333.30 us` | `1333.30 us` | baseline |
| final WPT2 adjacent | `514.738, 512.701, 510.155 us` | `512.701 us` | `1341.57, 1341.52, 1336.73 us` | `1341.52 us` | GEMM1 `-1.31%`，e2e `-0.62%` |

还测试了 payload 与 scale TDM 之间插入一条独立 WMMA 的版本：

```text
SHA256=917a24c91d2943acff513300dee296e081857a83042c9f052835d5785f6a1a1b
source median   = 509.263 us
stagger median  = 517.920 us
GEMM1 change    = -1.70%
```

因此最终保留相邻发射版本。实验说明 WPT2 确实均衡了每个 wave 的 logical bytes，
但在当前纯汇编 persistent pipeline 中，新增的第二条 TDM、descriptor update 和
code-size/SALU 开销超过了 barrier arrival-skew 的潜在收益。这个 ownership port
是正确实现，但不是性能 winner；all-NT-RT source 仍应作为当前正式性能版本。

还尝试过把 payload descriptor 暂存到 VGPR、跨 persistent task 恢复，以减少每个
task 的完整 descriptor rebuild。该 probe 在 random MoE e2e 中产生 NaN，未进入
性能测试，相关临时 ISA 和生成脚本已删除。
