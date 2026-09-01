# FMHA D128 OPUS C++ Kernel Deep Introduction

<!-- markdown-toc-generator:start -->
## Table of Contents

- [1. Kernel Overview and Pipeline](#1-kernel-overview-and-pipeline)
  - [1.1 Buffering: where K / V / Q / O live (GM / LDS / VGPR)](#sec-1-1-buffering-where-k-v-q-o-live-gm-lds-vgpr)
  - [1.2 Online softmax split into sub-stages](#12-online-softmax-split-into-sub-stages)
  - [1.3 Two wave-groups: group A leads by one cluster](#13-two-wave-groups-group-a-leads-by-one-cluster)
  - [1.4 Refined pseudocode](#14-refined-pseudocode)
  - [1.5 Per-operation main instructions and counts](#15-per-operation-main-instructions-and-counts)
  - [1.6 Scheduling primitives: sched_barrier_exp_pairs, sched_barrier_pairs, sched_barrier](#sec-1-6-scheduling-primitives-sched_barrier_exp_pairs-sched_barrier_pairs-sched_barrier)
    - [sched_barrier_pairs()](#sec-sched_barrier_pairs)
    - [sched_barrier_exp_pairs()](#sec-sched_barrier_exp_pairs)
    - [How C1 uses both (one group = 1 pipeline)](#sec-how-c1-uses-both-one-group-1-pipeline)
    - [sched_barrier(0) (the hard fence)](#sec-sched_barrier0-the-hard-fence)
- [2. Full GM/LDS/VGPR Layout Maps](#2-full-gmldsvgpr-layout-maps)
  - [2.1 Q: GM -> VGPR](#sec-2-1-q-gm---vgpr)
  - [2.2 K: GM -> LDS -> VGPR](#sec-2-2-k-gm---lds---vgpr)
  - [2.3 V: GM -> LDS -> VGPR](#sec-2-3-v-gm---lds---vgpr)
    - [2.3.1 MFMA A-Operand Convention and CDNA4 Transpose Load](#231-mfma-a-operand-convention-and-cdna4-transpose-load)
    - [2.3.2 ds_read_tr16_b64 / ds_read_b64_tr_b16 instruction semantics](#sec-2-3-2-ds_read_tr16_b64-ds_read_b64_tr_b16-instruction-semantics)
  - [2.4 S and P: VGPR-only Score/Probability Layout](#24-s-and-p-vgpr-only-scoreprobability-layout)
  - [2.5 O: VGPR -> GM](#sec-2-5-o-vgpr---gm)
  - [2.6 Summary: Where Each Tensor Lives](#26-summary-where-each-tensor-lives)
  - [2.7 Low-Level Call Chains and Issue Counts](#27-low-level-call-chains-and-issue-counts)

<!-- markdown-toc-generator:end -->

## Table of Contents

- [1. Kernel Overview and Pipeline](#kernel-overview-and-pipeline)
  - [1.1 Buffering: where K / V / Q / O live (GM / LDS / VGPR)](#buffering-gm-lds-vgpr)
  - [1.2 Online softmax split into sub-stages](#online-softmax-sub-stages)
  - [1.3 Two wave-groups: group A leads by one cluster](#two-wave-groups)
  - [1.4 Refined pseudocode](#refined-pseudocode)
  - [1.5 Per-operation main instructions and counts](#per-op-instruction-counts)
  - [1.6 Scheduling primitives: `sched_barrier_exp_pairs`, `sched_barrier_pairs`, `sched_barrier`](#sched-primitives)
    - [`sched_barrier_pairs<Pairs, VALU_CNT, Group>()`](#sched_barrier_pairspairs-valu_cnt-group)
    - [`sched_barrier_exp_pairs<Pairs, EXP_CNT, Group>()`](#sched_barrier_exp_pairspairs-exp_cnt-group)
    - [How C1 uses both (one `group = 1` pipeline)](#how-c1-uses-both-one-group--1-pipeline)
    - [`sched_barrier(0)` (the hard fence)](#sched_barrier0-the-hard-fence)
- [2. Full GM/LDS/VGPR Layout Maps](#full-layout-maps)
  - [2.1 Q: GM -> VGPR](#q-gm-to-vgpr)
  - [2.2 K: GM -> LDS -> VGPR](#k-gm-to-lds-to-vgpr)
  - [2.3 V: GM -> LDS -> VGPR](#v-gm-to-lds-to-vgpr)
    - [2.3.1 MFMA A-Operand Convention and CDNA4 Transpose Load](#v-mfma-a-operand-and-transpose-load)
    - [2.3.2 `ds_read_tr16_b64` / `ds_read_b64_tr_b16` instruction semantics](#ds-read-tr16-b64-semantics)
  - [2.4 S and P: VGPR-only Score/Probability Layout](#s-and-p-vgpr-only-layout)
  - [2.5 O: VGPR -> GM](#o-vgpr-to-gm)
  - [2.6 Summary: Where Each Tensor Lives](#tensor-residency-summary)
  - [2.7 Low-Level Call Chains and Issue Counts](#low-level-call-chains-and-issue-counts)

<a id="kernel-overview-and-pipeline"></a>

## 1. Kernel Overview and Pipeline

This kernel implements **Grouped-Query Flash Attention forward** with online softmax for BF16 inputs on AMD MI355X (gfx950 / CDNA4). It is the **performance-target reference** for the FlyDSL `flash_attn_opus` port and achieves ~1131 TFLOPS (causal, B=16 S=8192 H=64 D=128).

Each workgroup processes one `(batch, query-head, q-block)` of **M=256** query rows against all KV tiles in a fixed-cluster software pipeline. Every cluster ends with `s_barrier`. The body is a **prologue** (primes KV tile 0), a **main loop of 8 clusters that advances 2 KV tiles per iteration**, and a **14-cluster epilogue** that drains the trailing tiles. Clusters strictly alternate a **memory stage** (GM->LDS DMA + LDS->VGPR reads, even clusters) and a **compute stage** (MFMA + softmax, odd clusters).

**Which matrix is the MFMA left (`src0`/A) vs right (`src1`/B).** Each product is one `v_mfma_f32_32x32x16_bf16 D, A, B, C` (`D = A*B + C`; A is 32x16, B is 16x32). In the *actual assembly* the hardware **left** operand (`src0`/A) is the *mathematical right* matrix, and the hardware **right** operand (`src1`/B) is the *mathematical left* matrix -- the OPUS `mfma_adaptor_swap_ab` swaps the source-level operands before the MFMA builtin (see [2.3.1](#v-mfma-a-operand-and-transpose-load)):

| Matmul | Math form | Math left | Math right | HW `src0` (A, left) | HW `src1` (B, right) | Accumulator |
|---|---|---|---|---|---|---|
| Q*K (`mma0`) | `S = Q . K^T` | Q | K^T | **K** | **Q** | S |
| P*V (`mma1`) | `O += P . V` | P | V | **V** | **P** | O |

ISA evidence (`opus_gqa_d128_bf16_causal_gfx950.s`) and source:
- Q*K: `v_mfma_f32_32x32x16_bf16 v[32:47], v[10:13], v[144:147], 0` (line 495) -> `src0` = **K** (`v[10:13]`, from `ds_read_b128`), `src1` = **Q** (`v[144:147]`, scaled). Source: `mma0(v_q, v_k)` is passed math-order, then `mfma_adaptor_swap_ab::operator()` calls `base::operator()(v_k, v_q, c)`.
- P*V: `v_mfma_f32_32x32x16_bf16 v[80:95], v[20:23], v[112:115], v[80:95]` (line 1483) -> `src0` = **V** (`v[20:23]`, from `ds_read_b64_tr_b16`), `src1` = **P** (`v[112:115]`, bf16). Source: `mma1.step_k(..., v_p, v_v, v_o)` is swapped to `base::operator()(v_v, v_p, v_o)`.

The swap is deliberate: with `src1 = Q` the CDNA 32x32 MFMA puts the **query row on the lane axis** and (with `src0 = K`) the **key column on the accumulator registers** -- the "one query row per lane, keys across registers" layout the online softmax (max/sum over keys) consumes; likewise `src0 = V` / `src1 = P` puts the query row on lanes and the head-dim on registers for O.

<a id="buffering-gm-lds-vgpr"></a>

<a id="sec-1-1-buffering-where-k-v-q-o-live-gm-lds-vgpr"></a>
### 1.1 Buffering: where K / V / Q / O live (GM / LDS / VGPR)

| Tensor | Global memory | LDS | VGPR |
|---|---|---|---|
| Q | full Q tile | none (Q never goes through LDS) | `v_q` resident (32x128, pre-scaled by 1/sqrt(D)); reused by every `mma0` |
| K | streamed tile by tile (64 rows) | **2 buffers** `s_k[0]`, `s_k[1]` (~16.6 KB each) | `v_k` (1 live at a time) -> `mma0` |
| V | streamed tile by tile (64 rows) | **2 buffers** `s_v[0]`, `s_v[1]` (~17.4 KB each) | `v_v` (1 live at a time) -> `mma1`, read via HW-transpose `tr_load` (`ds_read_tr16_b64`) |
| S (scores `Q*K^T`) | -- | none (register-only) | **2 copies** `v_s[0]`, `v_s[1]` (C++ decl `v_s[2]`) -- 32 f32/lane each = 64 VGPR/lane; both live in C1/C5 for software pipelining. ISA: `v_s[0]` = v[2:17]+v[32:47], `v_s[1]` = v[96:111]+v[128:143] |
| P (probabilities) | -- | none (register-only) | **1 copy** `v_p` (C++ decl single `v_p`) -- 32 bf16/lane = 16 VGPR/lane (packed); cast from `v_s`, consumed by `mma1`, reused every KV tile. ISA: v[112:127] |
| O | written at the end | none | `v_o` = 4 f32 accumulator banks, resident across the whole loop |

K and V each use a **2-deep LDS double buffer** (interleaved `[K0][V0][K1][V1]`, ~68 KB total). In every memory cluster one buffer is **DMA-filled from GM** while the other is **read into VGPR** for the next MFMA. `mma0` (GEMM0, `S = Q*K^T`) reads `v_q`+`v_k`; `mma1` (GEMM2, `O += P*V`) reads `v_p`+`v_v`:

| Cluster | GM -> LDS (DMA, prefetch ahead) | LDS -> VGPR | MFMA |
|---|---|---|---|
| C0 (mem) | V tile `j-2` -> `s_v[1]` | `s_k[1]` -> `v_k` | -- |
| C1 (cmp) | -- | -- | `v_s[1] = mma0(v_q, v_k)` |
| C2 (mem) | K tile `j` -> `s_k[1]` | `s_v[0]` -> `v_v` | -- |
| C3 (cmp) | -- | -- | `v_o += mma1.step_k<0..3>(v_p, v_v)` |
| C4 (mem) | V tile `j-1` -> `s_v[0]` | `s_k[0]` -> `v_k` | -- |
| C5 (cmp) | -- | -- | `v_s[0] = mma0(v_q, v_k)` |
| C6 (mem) | K tile `j+1` -> `s_k[0]` | `s_v[1]` -> `v_v` | -- |
| C7 (cmp) | -- | -- | `v_o += mma1.step_k<0..3>(v_p, v_v)` |

The two pipeline halves (C0-C3 vs C4-C7) use opposite LDS buffers, so a tile being DMA'd into one buffer never collides with the tile being consumed from the other.

<a id="online-softmax-sub-stages"></a>

### 1.2 Online softmax split into sub-stages

To hide the transcendental `exp2` (TRANS) latency behind MFMA, a tile's softmax is split and spread over **two consecutive compute clusters** plus the P*V cluster:

1. `attn_row_max(S)` (cross-lane max via `permlane32_swap`) + **lazy-rescale** of `v_o`/`l_row` + `S -= m_row` + `exp2(S[0:16])` (**first half**).
2. `exp2(S[16:32])` (**second half**) + `l_row += attn_sum(P)` (row denominator) + `v_p = cast<bf16>(P)`.
3. `v_o += mma1.step_k(v_p, v_v)` consumes the bf16 `v_p` (P*V).

Distribution across the pipeline:
- **Prologue**: stage 1 (partial) for tile 0 -- `row_max`, `sub_row`, `exp2` first half.
- **Main loop**: tile A (scores from C1) -> stage 1 in **C3**, stage 2 in **C5**, P*V in **C7**; tile B (scores from C5) -> stage 1 in **C7**, stage 2 in **C1 of the next iter**, P*V in **C3 of the next iter**. So every compute cluster does P*V for an older tile while doing softmax stage 1 for a newer tile. Causal masking, when a tile straddles the diagonal, is applied in the **memory cluster before the row-max** (prologue / C6 / epilogue E2,E6,E10).
- **Epilogue**: same chain but rescale is **unconditional** (not lazy); E11 folds both `exp2` halves of the last tile into one cluster.

<a id="two-wave-groups"></a>

### 1.3 Two wave-groups: group A leads by one cluster

The 8 waves split into **group A (waves 0-3)** and **group B (waves 4-7)**. In the prologue group B runs **one extra `s_barrier`** (the "stagger"). Since barriers match by ordinal and each cluster has exactly one `s_barrier`, this makes **group A run one full cluster ahead of group B** for the whole main loop and epilogue:

```
group A:  [ P0+QK0 ]--b--[   C0   ]--b--[   C1   ]--b--[   C2   ]
group B:  [   P0   ]--b--[  QK0   ]--b--[   C0   ]--b--[   C1   ]
  P0  = prologue load (prefetch K[1] + read K[0] -> v_k)
  QK0 = prologue mma0 + first softmax pass (KV tile 0)
  C0..= main-loop clusters
```

Because of the one-cluster offset, when **group A is in a compute cluster (MFMA)** **group B is in the adjacent memory cluster (DMA + LDS reads + waitcnt)** -- one group computes while the other loads, swapping at every `s_barrier`, so each group's memory latency hides behind the other's MFMA. `s_setprio(1)`/`s_setprio(0)` brackets the heaviest compute clusters (C3/C7) to hand the shared MFMA issue port from the computing group to the group entering its compute phase. The epilogue's closing barrier gives **group A** the matching extra `s_barrier` to re-sync both groups before the store.

<a id="refined-pseudocode"></a>

### 1.4 Refined pseudocode

```
Prologue:
  [P1] async K[0] -> s_k[0]
       s_waitcnt lgkmcnt(0) vmcnt(0)
       s_barrier
  [P2] v_q = load Q (32x128)
       v_q.f32 *= 1/sqrt(D) * log2e (in-flight scale)
       v_q = cast<bf16>
  [P3] async K[1] -> s_k[1]
       async V[0] -> s_v[0]
       v_k = read(s_k[0])
       s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_V)
  [P4] if stagger:
         group B runs +1 s_barrier      // OPEN phase shift (A now leads by 1 cluster)
  [P5] v_s[0] = mma0(v_q, v_k)
       if CAUSAL: mask
       m_row = attn_row_max(v_s[0])
       v_s[0] -= m_row
       v_p = exp2(v_s[0][0:16])
       s_barrier
  [P6] async K[2] -> s_k[0]
       init { m_row, l_row=0, v_o=0(x4 banks), v_p }

Main loop (j = 3; j < max_tiles-1; j += 2):    // 8 clusters, +2 KV tiles per iter
  [C0 mem] async V[j-2]->s_v[1]
           v_k = read(s_k[1])
           s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K + NUM_DMA_V)
           s_barrier
  [C1 cmp] v_s[1] = mma0(v_q, v_k)
           v_p += exp2(prev[16:32])
           l_row += sum(v_p)
           v_p = cast<bf16>
           sched_barrier_exp_pairs<6,3,1>()
           sched_barrier_pairs<10,5,1>()
           s_barrier
  [C2 mem] async K[j]->s_k[1]
           v_v = tr_load(s_v[0])
           s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K + NUM_DMA_V)
           s_barrier
  [C3 cmp] s_setprio(1)
           v_o += step_k<0>(v_p, v_v);
           m_tile = attn_row_max(v_s[1])
           sched_barrier_pairs<4,5,2>()
           LAZY-RESCALE (ballot==read_exec):
             skip if all rows within 8.0,
             else
               rescale = exp2(m_row - combined)
               v_o *= rescale
               l_row *= rescale
               m_row <- combined
           v_o += step_k<1..3>(v_p, v_v)
           v_s[1] -= m_row
           v_p = exp2(v_s[1][0:16])
           sched_barrier_pairs<6,5,2>()
           sched_barrier_exp_pairs<6,3,2>()
           s_setprio(0)
           s_barrier
  [C4 mem] async V[j-1]->s_v[0]
           v_k = read(s_k[0])
           s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K + NUM_DMA_V)
           s_barrier
  [C5 cmp] v_s[0] = mma0(v_q, v_k)
           v_p += exp2(prev[16:32])
           l_row += sum
           v_p = cast<bf16>
           sched_barrier_exp_pairs<6,3,3>()
           sched_barrier_pairs<10,5,3>()
           s_barrier
  [C6 mem] async K[j+1]->s_k[0]
           v_v = tr_load(s_v[1])
           if CAUSAL: mask v_s[0]
           s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K + NUM_DMA_V)
           s_barrier
  [C7 cmp] s_setprio(1)
           v_o += step_k<0..3>(v_p, v_v)
           m_tile = attn_row_max(v_s[0])
           sched_barrier_pairs<4,5,4>()
           LAZY-RESCALE
           v_s[0] -= m_row
           v_p = exp2(v_s[0][0:16])
           sched_barrier_pairs<6,5,4>()
           sched_barrier_exp_pairs<6,3,4>()
           s_setprio(0)
           s_barrier
           yield { m_row, l_row, v_o, v_p }

Epilogue (14 clusters E0..E13; mirrors the loop but rescale is UNCONDITIONAL):
  [E0  mem] async V->s_v[1]
            v_k = read(s_k[1])
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K + NUM_DMA_V)
            s_barrier
  [E1  cmp] v_s[1] = mma0
            finish exp2[16:32]
            l_row += sum
            cast<bf16>
            sched_barrier_exp_pairs<6,3,5>()
            sched_barrier_pairs<10,5,5>()
            s_barrier
  [E2  mem] async K->s_k[1]
            v_v = tr_load(s_v[0])
            if CAUSAL: mask v_s[1]
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K + NUM_DMA_V)
            s_barrier
  [E3  cmp] v_o += mma1(v_p, v_v) (full 4 steps)
            m_row <- max
            rescale = exp2(delta)
            v_s[1] -= m_row
            v_p = exp2[0:16]
            sched_barrier_pairs<10,5,6>()
            sched_barrier_exp_pairs<6,3,6>()
            v_o *= rescale
            s_setprio(0)
            s_barrier
  [E4  mem] async V->s_v[0]
            v_k = read(s_k[0])
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K + NUM_DMA_V)
            s_barrier
  [E5  cmp] v_s[0] = mma0
            l_row *= rescale
            finish exp2[16:32]
            l_row += sum
            cast<bf16>
            sched_barrier_exp_pairs<6,3,7>()
            sched_barrier_pairs<10,5,7>()
            s_barrier
  [E6  mem] v_v = tr_load(s_v[1])
            if CAUSAL: mask v_s[0]
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_V)
            s_barrier
  [E7  cmp] v_o += mma1(v_p, v_v)
            m_row <- max
            rescale
            v_s[0] -= m_row
            v_p = exp2[0:16]
            sched_barrier_pairs<10,5,8>()
            sched_barrier_exp_pairs<6,3,8>()
            v_o *= rescale
            s_setprio(0)
            s_barrier
  [E8  mem] async V->s_v[1]
            v_k = read(s_k[1])
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_V)
            s_barrier
  [E9  cmp] v_s[1] = mma0 (last tile)
            l_row *= rescale
            finish exp2[16:32]
            l_row += sum
            cast<bf16>
            sched_barrier_exp_pairs<6,3,9>()
            sched_barrier_pairs<10,5,9>()
            s_barrier
  [E10 mem] v_v = tr_load(s_v[0])
            if CAUSAL: mask v_s[1]
            s_waitcnt lgkmcnt(0) vmcnt(0)  // all DMA drained
            s_barrier
  [E11 cmp] v_o += mma1(v_p, v_v)
            m_row <- max
            rescale
            v_s[1] -= m_row
            v_p = exp2[0:16] then exp2[16:32]
            sched_barrier_pairs<10,5,10>()
            sched_barrier_exp_pairs<6,3,10>()
            l_row *= rescale
            l_row += sum
            cast<bf16>
            v_o *= rescale
            s_barrier
  [E12 mem] v_v = tr_load(s_v[1])
            s_waitcnt lgkmcnt(0)
            s_barrier
  [E13 cmp] v_o += mma1(v_p, v_v)                 // final P*V

  Finalize:
    v_o *= 1 / l_row                              // (0 if l_row == 0)
    if stagger: group A runs +1 s_barrier         // CLOSE phase shift, re-sync A & B
    store g_o = cast<bf16/fp16>(v_o)              // 8-byte (dwordx2) packed writes
```

<a id="per-op-instruction-counts"></a>

### 1.5 Per-operation main instructions and counts

Each pseudocode operation lowers to a fixed set of ISA instructions whose count is determined by the tiling. Counts below are the **number of that instruction emitted in the ISA for one warp processing one KV tile** (each VALU / memory instruction is executed by all 64 lanes of the warp in lock-step; each MFMA is a warp-collective instruction). Verified against the C++ helpers in `gqa_d128_kernel_template.hpp` and the annotated assembly `opus_gqa_d128_bf16_causal_gfx950.s`.

Fixed sizes for `opus_gqa_traits<32, 64, 128, 8, true>` come from
`opus_attn/gqa_defs.h` plus the D=128 layout helpers in
`gqa_d128_kernel_template.hpp`:

```text
Core tile / launch shape:
  Q_TILE_SIZE       = 32
  KV_TILE_SIZE      = 64
  D_TILE_SIZE       = 128
  NUM_WARPS         = 8
  WARP_SIZE         = 64
  BLOCK_SIZE        = NUM_WARPS * WARP_SIZE = 512

Data / vector widths:
  D_ATTN            = bf16_t
  D_ACC             = float
  VEC_Q             = 8 bf16  = 16 B
  VEC_KV            = 8 bf16  = 16 B
  VEC_TR_V          = 4 bf16  = 8 B
  VEC_O             = 4 bf16  = 8 B

MFMA tile:
  ISA               = v_mfma_f32_32x32x16_bf16
  W_M               = 32
  W_N               = 32
  W_K               = 16
  SLICE_D           = D_TILE_SIZE = 128
  NUM_D_SLICES      = 1

GEMM expansion:
  GEMM0_E_M         = Q_TILE_SIZE  / W_M = 32 / 32  = 1
  GEMM0_E_N         = KV_TILE_SIZE / W_N = 64 / 32  = 2
  GEMM0_E_K         = SLICE_D      / W_K = 128 / 16 = 8
  GEMM0 MFMA count  = GEMM0_E_M * GEMM0_E_N * GEMM0_E_K
                    = 1 * 2 * 8 = 16

  GEMM1_E_M         = Q_TILE_SIZE  / W_M = 32 / 32  = 1
  GEMM1_E_N         = SLICE_D      / W_N = 128 / 32 = 4
  GEMM1_E_K         = KV_TILE_SIZE / W_K = 64 / 16  = 4
  GEMM1 full count  = GEMM1_E_M * GEMM1_E_N * GEMM1_E_K
                    = 1 * 4 * 4 = 16
  GEMM1 step_k count = GEMM1_E_M * GEMM1_E_N
                     = 1 * 4 = 4

Per-lane register tiles:
  v_q               = 64 bf16, then scaled/repacked as bf16
  v_k               = 64 bf16
  v_v               = 64 bf16, produced by 32 ds_read_b64_tr_b16 instructions
  v_s               = 32 f32 scores/probability workspace
  v_p               = 32 bf16 probability values after cast
  v_o               = 64 f32 accumulator values = 4 banks x 16

Shared-memory geometry:
  D_128B_SIZE       = 128 / sizeof(bf16) = 64 bf16
  smem_linear_wave  = WARP_SIZE * 16 / sizeof(bf16)
                    = 64 * 16 / 2 = 512 bf16
  smem_n_per_wave   = smem_linear_wave / D_128B_SIZE
                    = 512 / 64 = 8 rows/wave
  smem_n_rpt        = KV_TILE_SIZE / smem_n_per_wave
                    = 64 / 8 = 8
  smem_d_rpt        = D_TILE_SIZE / D_128B_SIZE
                    = 128 / 64 = 2

SMEM padding / buffer sizes:
  smem_padding_16B  = 16 / sizeof(bf16) = 8 bf16
  smem_padding_64B  = 64 / sizeof(bf16) = 32 bf16
  smem_k_padding    = 8 bf16   (D=128 K uses 16B padding)
  smem_v_padding    = 32 bf16  (D=128 V uses 64B padding)
  smem_k_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_k_padding)
                    = 8 * 2 * (512 + 8) = 8320 bf16
  smem_v_tile_elems = smem_n_rpt * smem_d_rpt * (smem_linear_wave + smem_v_padding)
                    = 8 * 2 * (512 + 32) = 8704 bf16
  smem_buffer_elems = smem_k_tile_elems + smem_v_tile_elems
                    = 17024 bf16
  smem_size_bytes   = 2 buffers * smem_buffer_elems * sizeof(bf16)
                    = 2 * 17024 * 2 = 68096 B

Primitive issue counts:
  k_buffer_load_insts = (KV_TILE_SIZE * D_TILE_SIZE) / (BLOCK_SIZE * VEC_KV)
                      = (64 * 128) / (512 * 8) = 2
  v_buffer_load_insts = (64 * 128) / (512 * 8) = 2
  k_ds_read_insts     = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV)
                      = (2 * 8 * 32 * 16) / (64 * 8) = 16
  v_ds_read_insts     = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_TR_V)
                      = (4 * 4 * 32 * 16) / (64 * 4) = 32
```

| # | Operation | Main instruction(s) | Count | Tiling derivation |
|---|---|---|---|---|
| 1 | `async K[0] -> s_k[0]` | `buffer_load_dwordx4` (direct-to-LDS) | **2** | `k_buffer_load_insts = (KV*D)/(BLOCK_SIZE*VEC_KV) = (64*128)/(512*8) = 2` (block-cooperative) |
| 2 | `async V[0] -> s_v[0]` | `buffer_load_dwordx4` (direct-to-LDS) | **2** | `v_buffer_load_insts = (64*128)/(512*8) = 2` |
| 3 | `v_q = load Q (32x128)` | `buffer_load_dwordx4` | **8** | per lane = (32*128)/64 = 64 bf16 = 128 B; /(VEC_Q=8 bf16=16 B) = 8 |
| 4 | `v_q.f32 *= 1/sqrt(D)*log2e` | `v_pk_mul_f32` (scale) + `v_cvt_pk_bf16_f32` (repack); `v_and`+`v_lshlrev` (unpack bf16->f32) | **32 + 32** (+32+32 unpack) | q_len = (32*128)/64 = 64 f32; packed 2/instr -> 64/2 = 32 muls and 32 repacks; unpack 32 dwords -> 32 `v_and` + 32 `v_lshlrev` |
| 5 | `v_s[0] = mma0(v_q, v_k)` | `v_mfma_f32_32x32x16_bf16` | **16** | GEMM0 = E_M*E_N*E_K = 1*2*8 = 16 |
| 6 | `if CAUSAL: mask` | `v_cmp_lt_i32_e64` + `v_cndmask_b32_e64` | **32 + 32** | s_len = 32 elems, masked in pairs by `attn_mask_vec2_imm`: 16 calls x (2 cmp + 2 cndmask) = 32 + 32 |
| 7 | `m_row = attn_row_max(v_s[0])` | `v_max3_f32` + `v_permlane32_swap_b32` + `v_max_f32` | **16 + 1 + 1** | reduce s_len=32 at 2 elems/instr -> 16 `v_max3`; then cross-32-lane (1 `permlane32_swap` + 1 `v_max`) |
| 8 | `v_s[0] -= m_row` | `v_sub_f32` | **32** | s_len = 32 elems, 1 subtract each |
| 9 | `v_p = exp2(v_s[0][0:16])` | `v_exp_f32` | **16** | first half = s_len/2 = 16 elems (second half is another 16 in the next cluster) |
| 10 | `v_k = read(s_k[1])` | `ds_read_b128` | **16** | `k_ds_read_insts = (E_N*E_K*W_N*W_K)/(WARP_SIZE*VEC_KV) = (2*8*32*16)/(64*8) = 16`; = 128 bf16/lane / 8 |
| 11 | `l_row += sum(v_p)` (attn_sum) | `v_add_f32` + `v_permlane32_swap_b32` + `v_add_f32` | **31 + 1 + 1** | serial reduction of s_len=32 (31 adds) + cross-32-lane (1 `permlane32_swap` + 1 `v_add`) |
| 12 | `v_p = cast<bf16>` | `v_cvt_pk_bf16_f32` | **16** | v_p = 32 f32/lane -> packed 2/instr = 16 |
| 13 | `v_v = tr_load(s_v[0])` | `ds_read_b64_tr_b16` (HW transpose) | **32** | `v_ds_read_insts = (E_N*E_K*W_N*W_K)/(WARP_SIZE*VEC_TR_V) = (4*4*32*16)/(64*4) = 32`; = 128 bf16/lane / 4 |
| 14 | `v_o += step_k<0>(v_p, v_v)` | `v_mfma_f32_32x32x16_bf16` | **4** | one K-step = GEMM1 E_M*E_N = 1*4 = 4 (full GEMM1 = 4 steps x 4 = 16) |
| 15 | `v_o += mma1(v_p, v_v)` (full) | `v_mfma_f32_32x32x16_bf16` | **16** | GEMM1 = E_M*E_N*E_K = 1*4*4 = 16 |
| 16 | `l_row *= rescale` | `v_mul_f32` | **1** | l_row is one f32 scalar per lane |
| 17 | `v_o *= rescale` (scale_output_tile) | `v_pk_mul_f32` | **32** | o_len = (32*128)/64 = 64 f32; packed 2/instr = 32 |
| 18 | `store g_o = cast<bf16/fp16>(v_o)` | `v_cvt_pk_bf16_f32` + `buffer_store_dwordx2` | **32 + 16** | 64 f32/lane -> 32 `v_cvt_pk` (=32 dwords=64 bf16); store dwordx2 = 2 dwords/instr -> 32/2 = 16 |

Notes:
- Counts 1-2 (`async K/V`) are issued cooperatively by the whole 512-thread block (2 instructions per lane), filling one 64x128 KV tile into LDS; counts 3-18 are per warp.
- The aggregate whole-kernel ISA counts are consistent with these per-call numbers x (call sites): `v_cmp_lt_i32` = 5 mask calls x 32 = 160; `ds_read_b64_tr_b16` = 6 tr_loads x 32 = 192; `buffer_store_dwordx2` = 16. Any higher text-search count comes from matching instruction names inside assembly comments, not from real ISA instructions.
- The full softmax of one tile costs (per warp): row_max (16 `v_max3`+`permlane`+`v_max`) + sub_row (32 `v_sub`) + exp2 (32 `v_exp`, two halves) + sum (31 `v_add`+`permlane`+`v_add`) + cast (16 `v_cvt_pk_bf16`); the rescale adds 32 `v_pk_mul` (O) + 1 `v_mul` (l_row) only when the lazy-rescale branch is taken.

<a id="sched-primitives"></a>

<a id="sec-1-6-scheduling-primitives-sched_barrier_exp_pairs-sched_barrier_pairs-sched_barrier"></a>
### 1.6 Scheduling primitives: `sched_barrier_exp_pairs`, `sched_barrier_pairs`, `sched_barrier`

The `sched_*` calls inside the clusters -- e.g. C1's `sched_barrier_exp_pairs<6,3,1>()` and `sched_barrier_pairs<10,5,1>()`, and the `sched_barrier(0)` at every cluster boundary -- are **compile-time scheduling directives only**. They lower to LLVM intrinsics that the AMDGPU IGroupLP (Instruction Group Level Parallelism) pass consumes and then removes; they emit **no real ISA instruction** and cost zero cycles. Their sole purpose is to constrain how LLVM's pre-/post-RA scheduler orders the MFMA / VALU / exp instructions inside a cluster (the hardware overlap of MFMA with VALU/exp/LDS is real; these hints just make the scheduler actually realize it).

Two different intrinsics are involved (`gqa_d128_kernel_template.hpp` lines 14-30):

```cpp
constexpr int MFMA_MASK = 0x08;   // MFMA / WMMA instruction class
constexpr int VALU_MASK = 0x02;   // VALU (v_add / v_mul / v_sub / v_cvt / ...)
constexpr int EXP_MASK  = 0x400;  // TRANS (transcendental: v_exp_f32, v_rcp, ...)
```

| Intrinsic | Kind | Effect |
|---|---|---|
| `__builtin_amdgcn_sched_group_barrier(mask, size, group)` | soft grouping hint | gathers `size` instructions of class `mask` into an ordered "group" tagged by `group`; groups sharing the same `group` are solved together as one interleaved pipeline |
| `__builtin_amdgcn_sched_barrier(mask)` | hard fence | forbids the classes selected by `mask` from being scheduled across this point (`mask = 0` => allow none, a full barrier) |

<a id="sec-sched_barrier_pairs"></a>
#### `sched_barrier_pairs<Pairs, VALU_CNT, Group>()`

A recursive helper that emits `Pairs` repetitions of "one MFMA group + one VALU group":

```cpp
template<int Pairs, int VALU_CNT, int Group>
__device__ inline void sched_barrier_pairs() {
    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);          // 1 MFMA
    __builtin_amdgcn_sched_group_barrier(VALU_MASK, VALU_CNT, Group);   // VALU_CNT VALU
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs-1, VALU_CNT, Group>();
}
```

So `sched_barrier_pairs<10,5,1>()` expands to **10 x (1 MFMA-group + 1 VALU-group of 5)** = 20 `sched_group_barrier` intrinsics, all tagged `group = 1`. It tells LLVM: "lay out this region as `[1 MFMA][up to 5 VALU]` repeated 10 times" -- i.e. spread the VALU work (here the `attn_sum` adds and the bf16 `cast`) evenly between the MFMA instructions instead of letting the scheduler bunch all MFMA or all VALU together.

<a id="sec-sched_barrier_exp_pairs"></a>
#### `sched_barrier_exp_pairs<Pairs, EXP_CNT, Group>()`

Identical shape, but the second group is the **TRANS/EXP** class instead of VALU:

```cpp
template<int Pairs, int EXP_CNT, int Group>
__device__ inline void sched_barrier_exp_pairs() {
    __builtin_amdgcn_sched_group_barrier(MFMA_MASK, 1, Group);          // 1 MFMA
    __builtin_amdgcn_sched_group_barrier(EXP_MASK, EXP_CNT, Group);     // EXP_CNT exp
    if constexpr (Pairs > 1) sched_barrier_exp_pairs<Pairs-1, EXP_CNT, Group>();
}
```

`sched_barrier_exp_pairs<6,3,1>()` expands to **6 x (1 MFMA-group + 1 EXP-group of 3)** = 12 `sched_group_barrier` intrinsics, `group = 1`. The `0x400` mask is `SchedGroupMask::TRANS`; LLVM's `SchedGroup::canAddMI()` classifies an instruction as TRANS via `TII->isTRANS(MI)`, which matches the `v_exp_f32` produced by `attn_exp2_slice`. The purpose is to keep the softmax `exp2` (a long-latency transcendental) interleaved with the MFMA chain so the exp latency hides behind MFMA, instead of the scheduler emitting all 16 `v_exp` back-to-back.

<a id="sec-how-c1-uses-both-one-group-1-pipeline"></a>
#### How C1 uses both (one `group = 1` pipeline)

C1's compute is `mma0` (16 MFMA) running concurrently with the previous tile's second-half `exp2` (16 `v_exp`), the `attn_sum` reduction (~31 `v_add`), and the bf16 `cast` (16 `v_cvt_pk`). The two hints **partition the 16 MFMA** and bind both halves into one pipeline (`group = 1`):

```text
sched_barrier_exp_pairs<6,3,1>()  ->  [1 MFMA][3 exp ] x 6    (6 MFMA paired with the exp work)
sched_barrier_pairs   <10,5,1>()  ->  [1 MFMA][5 VALU] x 10   (10 MFMA paired with sum/cast VALU)
                                       6 + 10 = 16 MFMA total  (== mma0's MFMA count)
```

The `size` operands (3, 5) are the maximum instructions of that class to attach to each MFMA slot. Because both calls share `group = 1`, IGroupLP solves them as a single interleaved schedule for the whole C1 compute body. Directionality is **bottom-up**: `IGroupLPDAGMutation` scans from each marker backward to the start of the scheduling region (`findCandidateSUnits(RIter, ..., SUnits.rend())`), so a hint describes the MFMA / exp / VALU **already emitted above it**, not future instructions in later clusters. (Other clusters reuse the same helpers with their own `group` id -- C5 uses `group = 3`, the epilogue uses 5/7/9, etc. -- so each cluster's pipeline is solved independently.)

<a id="sec-sched_barrier0-the-hard-fence"></a>
#### `sched_barrier(0)` (the hard fence)

`__builtin_amdgcn_sched_barrier(0)` is a *different* intrinsic; it lowers to the AMDGPU `SCHED_BARRIER` pseudo. With `mask = 0` ("allow none"), **every** class -- MFMA, VALU, SALU, TRANS, VMEM, DS -- is barred from being scheduled across it:

- `IGroupLPDAGMutation::addSchedBarrierEdges` inverts the mask, classifies all SUnits, and adds artificial DAG edges that preserve their original order relative to the barrier.
- `SIInstrInfo::isSchedulingBoundary()` returns true for `SCHED_BARRIER` with immediate 0, so the machine scheduler **splits the scheduling region** there and cannot reorder anything across it.

It is placed at every cluster boundary, immediately around the real `s_barrier`, so that (a) one cluster's MFMA / exp / loads cannot leak into an adjacent cluster, and (b) the hardware `s_barrier` itself stays pinned at the phase boundary (it cannot be hoisted across the fence). Like the group-barrier hints, `sched_barrier(0)` emits **no real instruction** -- the actual cross-wave synchronization is the separate `s_barrier` next to it.

<a id="full-layout-maps"></a>

## 2. Full GM/LDS/VGPR Layout Maps

This section expands the previous `make_layout_*` descriptions into concrete formulas, in the same style as an ISA address derivation. One important distinction from some persistent-attention kernels: **this OPUS D=128 kernel does not stage Q in LDS**. Q is loaded directly from GM to VGPR; only K and V are double-buffered in LDS.

Notation:

```text
lane_id = thread_id_x % 64
warp_id = thread_id_x / 64          // 0..7, readfirstlane-broadcast to SGPR

lane_m     = lane_id % 32           // Q/O row inside one warp tile
lane_group = lane_id / 32           // 0 or 1, half-wave selector

lane_n = lane_id >> 3               // 0..7, which KV row group inside a wave
lane_d = lane_id & 7                // 0..7, which 8-bf16 D vector inside a 64-bf16 half

q_block_size  = NUM_WARPS * Q_TILE_SIZE = 256
q_block_start = block_id_y * q_block_size
kv_tile(j)    = j * KV_TILE_SIZE * stride_kv_n

All element formulas below are in bf16 elements unless marked as byte addresses.
byte_addr = elem_addr * sizeof(bf16) = elem_addr * 2.
```

<a id="q-gm-to-vgpr"></a>

<a id="sec-2-1-q-gm---vgpr"></a>
### 2.1 Q: GM -> VGPR

Q has no LDS residency. `u_q` maps each lane to one Q row and two interleaved 8-bf16 D vectors per 16-wide K step.

```text
GM Q base:
  g_q = ptr_q
      + b * stride_q_b
      + q_block_start * stride_q_n
      + h * stride_q_h

Per workgroup Q rows:
  warp 0: rows q_block_start +   0..31
  warp 1: rows q_block_start +  32..63
  warp 2: rows q_block_start +  64..95
  warp 3: rows q_block_start +  96..127
  warp 4: rows q_block_start + 128..159
  warp 5: rows q_block_start + 160..191
  warp 6: rows q_block_start + 192..223
  warp 7: rows q_block_start + 224..255
```

`make_layout_q` address formula:

```text
k_iter = 0..7
vec    = 0..7

q_row = warp_id * 32 + lane_m
q_dim = k_iter * 16 + lane_group * 8 + vec

GM elem addr = q_row * stride_q_n + q_dim
```

Per-lane VGPR expansion:

| lane_id | lane_m | lane_group | `v_q` contents |
|---:|---:|---:|---|
| 0 | 0 | 0 | row 0, dim `0..7, 16..23, ..., 112..119` |
| 1 | 1 | 0 | row 1, dim `0..7, 16..23, ..., 112..119` |
| ... | ... | ... | ... |
| 31 | 31 | 0 | row 31, dim `0..7, 16..23, ..., 112..119` |
| 32 | 0 | 1 | row 0, dim `8..15, 24..31, ..., 120..127` |
| 33 | 1 | 1 | row 1, dim `8..15, 24..31, ..., 120..127` |
| ... | ... | ... | ... |
| 63 | 31 | 1 | row 31, dim `8..15, 24..31, ..., 120..127` |

VGPR shape:

```text
v_q: vector<bf16, 64> per lane
  v_q[k_iter * 8 + vec] =
      Q[q_block_start + warp_id * 32 + lane_m,
        k_iter * 16 + lane_group * 8 + vec]

Then:
  v_q_f32 *= (1 / sqrt(D)) * log2e
  v_q = cast<bf16>(v_q_f32)
```

**C++ source** (`make_layout_q`, `gqa_d128_kernel_template.hpp` lines 33-51):

```cpp
constexpr auto q_block_shape = opus::make_tuple(
    opus::number<T::GEMM0_E_M>{},          // 1
    opus::number<T::T_M>{},                // NUM_WARPS = 8
    opus::number<T::W_M>{},                // 32
    opus::number<T::GEMM0_E_K>{},          // 8
    opus::number<T::WARP_SIZE / T::W_M>{}, // 2
    opus::number<T::VEC_Q>{});             // 8
constexpr auto q_block_dim = opus::make_tuple(
    opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),   // group 0
    opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));  // group 1
return opus::make_layout(
    q_block_shape,
    opus::unfold_x_stride(q_block_dim, q_block_shape, opus::tuple{stride_q_n, 1_I}),
    opus::unfold_p_coord(q_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
```

**How the address is derived** (opus DSL, `opus.hpp` lines 2521-2604): `q_block_dim` groups the 6 shape axes into 2 stride groups; `unfold_x_stride` gives each group a base stride (here `{stride_q_n, 1}`) and, *inside* a group, axis stride = base x (product of the sizes of the axes that come after it in the same group). `unfold_p_coord` binds the three `p_dim` axes, in order, to `{warp_id, lane_id % W_M, lane_id / W_M}`; the `y_dim` axes are the loop indices (`k_iter`, `vec`).

| shape axis (size) | role | bound coord / loop var | unfolded stride | term |
|---|---|---|---|---|
| `GEMM0_E_M` (1) | y | -- (size 1) | `stride_q_n` x 8 x 32 = `stride_q_n`*256 | 0 |
| `T_M` (8) | p | `warp_id` | `stride_q_n` x 32 | `warp_id*32 * stride_q_n` |
| `W_M` (32) | p | `lane_id % W_M` (= `lane_m`) | `stride_q_n` x 1 | `(lane_id%32) * stride_q_n` |
| `GEMM0_E_K` (8) | y | `k_iter` (0..7) | 1 x 2 x 8 = 16 | `k_iter*16` |
| `WARP_SIZE/W_M` (2) | p | `lane_id / W_M` (= `lane_group`) | 1 x 8 = 8 | `(lane_id/32)*8` |
| `VEC_Q` (8) | y | `vec` (0..7) | 1 | `vec` |

Summing the terms: `q_row = warp_id*32 + lane_id%32` (group-0 base `stride_q_n`) and `q_dim = k_iter*16 + (lane_id/32)*8 + vec` (group-1 base 1), i.e. `GM elem addr = q_row * stride_q_n + q_dim` -- the formula above (`lane_m = lane_id % W_M`, `lane_group = lane_id / W_M`).

<a id="k-gm-to-lds-to-vgpr"></a>

<a id="sec-2-2-k-gm---lds---vgpr"></a>
### 2.2 K: GM -> LDS -> VGPR

K is cooperatively loaded by all 512 threads. Each lane transfers one 16-byte vector (`VEC_KV=8 bf16`) for each of the two D half-lines.

GM address formula (`u_gk`):

```text
d_half = 0..1
vec    = 0..7

k_row = lane_n * NUM_WARPS + warp_id      // interleaved rows: wave 0 gets 0,8,...; wave 1 gets 1,9,...
k_dim = d_half * 64 + lane_d * 8 + vec

GM elem addr = kv_tile(j) + k_row * stride_kv_n + k_dim
```

**C++ source** (`make_layout_gk_gv`, `gqa_d128_kernel_template.hpp` lines 76-99; V reuses the same function):

```cpp
constexpr int threads_d           = T::D_128B_SIZE / T::VEC_KV;       // 64/8 = 8  (= lane_d count)
constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_d;        // 512/8 = 64
constexpr int threads_n_per_wave  = opus::get_warp_size() / threads_d;// 64/8 = 8  (= lane_n count)
constexpr auto gk_block_shape = opus::make_tuple(
    opus::number<T::smem_d_rpt>{},                       // 2   (d_half)
    opus::number<T::KV_TILE_SIZE / threads_n_per_block>{}, // 1
    opus::number<threads_n_per_wave>{},                  // 8   (lane_n)
    opus::number<T::NUM_WARPS>{},                        // 8   (warp_id)
    opus::number<threads_d>{},                           // 8   (lane_d)
    opus::number<T::VEC_KV>{});                          // 8   (vec)
constexpr auto gk_block_dim = opus::make_tuple(
    opus::make_tuple(opus::y_dim{}),                                 // group 0
    opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),   // group 1
    opus::make_tuple(opus::p_dim{}, opus::y_dim{}));                 // group 2
return opus::make_layout(
    gk_block_shape,
    opus::unfold_x_stride(gk_block_dim, gk_block_shape,
        opus::tuple{opus::number<T::D_128B_SIZE>{}, stride_kv_n, 1_I}),
    opus::unfold_p_coord(gk_block_dim,
        opus::tuple{lane_id / threads_d, warp_id, lane_id % threads_d}));
```

**Derivation**: three stride groups with base strides `{D_128B_SIZE=64, stride_kv_n, 1}`. The three `p_dim` axes bind in order to `{lane_id/threads_d, warp_id, lane_id%threads_d}`.

| shape axis (size) | role | bound coord / loop var | unfolded stride | term |
|---|---|---|---|---|
| `smem_d_rpt` (2) | y | `d_half` (0..1) | 64 x 1 = 64 | `d_half*64` |
| `KV_TILE_SIZE/threads_n_per_block` (1) | y | -- (size 1) | `stride_kv_n` x 8 x 8 | 0 |
| `threads_n_per_wave` (8) | p | `lane_id/threads_d` (= `lane_n`) | `stride_kv_n` x 8 | `lane_n*8 * stride_kv_n` |
| `NUM_WARPS` (8) | p | `warp_id` | `stride_kv_n` x 1 | `warp_id * stride_kv_n` |
| `threads_d` (8) | p | `lane_id%threads_d` (= `lane_d`) | 1 x 8 = 8 | `lane_d*8` |
| `VEC_KV` (8) | y | `vec` (0..7) | 1 | `vec` |

So `k_row = lane_n*NUM_WARPS + warp_id` (group-1 base `stride_kv_n`, with `lane_n = lane_id/threads_d = lane_id>>3`) and `k_dim = d_half*64 + lane_d*8 + vec` (group-0 base 64 + group-2 base 1, `lane_d = lane_id%threads_d = lane_id&7`). The per-tile base `kv_tile(j) = j*KV_TILE_SIZE*stride_kv_n` is added by the caller, giving the formula above.

Per-wave GM expansion for `d_half=0`:

| wave | lane range | K row | D range |
|---:|---:|---:|---|
| 0 | `0..7` | 0 | `0..63` |
| 0 | `8..15` | 8 | `0..63` |
| 0 | `16..23` | 16 | `0..63` |
| ... | ... | ... | ... |
| 0 | `56..63` | 56 | `0..63` |
| 1 | `0..7` | 1 | `0..63` |
| 1 | `8..15` | 9 | `0..63` |
| ... | ... | ... | ... |
| 7 | `56..63` | 63 | `0..63` |

`d_half=1` repeats the same row mapping for `dim 64..127`.

K LDS buffer map:

```text
s_k[0] = smem_buf + 0
s_k[1] = smem_buf + smem_buffer_elems

K line stride = smem_linear_wave + smem_padding_16B
              = 512 + 8 = 520 bf16 = 1040 B

K tile size   = 16 lines * 520 bf16 = 8320 bf16
```

K LDS address formula (`u_sk` as the `async_load` destination):

```text
line = d_half * NUM_WARPS + warp_id       // 0..15
in_line_elem = lane_id * VEC_KV + vec     // 0..511, then 8 bf16 padding

K LDS elem addr = line * 520 + in_line_elem
K LDS byte addr = (line * 520 + lane_id * 8 + vec) * 2
```

**C++ source** (`make_layout_sk_sv<T, smem_padding>`, `gqa_d128_kernel_template.hpp` lines 102-118; called as `make_layout_sk_sv<T, T::smem_padding_16B>` for K at line 351, and `<T, T::smem_padding_64B>` for V at line 354):

```cpp
constexpr auto sk_block_shape = opus::make_tuple(
    opus::number<T::smem_d_rpt>{},               // 2   (d_half)
    opus::number<T::smem_n_rpt / T::NUM_WARPS>{}, // 8/8 = 1
    opus::number<T::NUM_WARPS>{},                // 8   (warp_id)
    opus::number<T::VEC_KV>{});                  // 8   (vec)
constexpr auto sk_block_dim = opus::make_tuple(
    opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}),   // group 0
    opus::make_tuple(opus::y_dim{}));                               // group 1
return opus::make_layout(
    sk_block_shape,
    opus::unfold_x_stride(sk_block_dim, sk_block_shape,
        opus::tuple{opus::number<T::smem_linear_wave + smem_padding>{}, 1_I}),
    opus::unfold_p_coord(sk_block_dim, opus::tuple{warp_id}));
```

with `smem_linear_wave = WARP_SIZE*16/sizeof(bf16) = 512` and `smem_padding_16B = 16/2 = 8`, so the group-0 base stride (the LDS *line* stride) is `512 + 8 = 520` bf16 for K (and `512 + 32 = 544` for V via `smem_padding_64B`).

**Derivation**: the single `p_dim` axis (`NUM_WARPS`) binds to `warp_id`; everything else is a loop/`y_dim`. Base strides `{512+pad, 1}`.

| shape axis (size) | role | bound coord / loop var | unfolded stride | term |
|---|---|---|---|---|
| `smem_d_rpt` (2) | y | `d_half` (0..1) | (512+8) x 1 x 8 = 4160 | `d_half*4160` |
| `smem_n_rpt/NUM_WARPS` (1) | y | -- (size 1) | (512+8) x 8 | 0 |
| `NUM_WARPS` (8) | p | `warp_id` | (512+8) x 1 = 520 | `warp_id*520` |
| `VEC_KV` (8) | y | `vec` (0..7) | 1 | `vec` |

This yields `d_half*4160 + warp_id*520 + vec = (d_half*NUM_WARPS + warp_id)*520 + vec = line*520 + vec`. The remaining `lane_id * VEC_KV` term in `in_line_elem` is **not** part of `u_sk`; it is supplied by the cooperative `async_load<VEC_KV>` primitive, which deposits each lane's contiguous 16-byte (`VEC_KV=8` bf16) vector at `lane_id*VEC_KV` within the line (paired with the matching `u_gk` GM read). Combining the two gives `line*520 + lane_id*8 + vec`, the formula above.

K LDS address space for one buffer:

```text
K LDS Address Space:
Address increases downward; each line is 1040B = 1024B data + 16B pad.

D half 0: dim[0..63]
                    ┌────────────────────────────┐
Wave 0 line:        │ line 0:  rows 0,8,...,56  │ ...pad...
                    └────────────────────────────┘
                    ┌────────────────────────────┐
Wave 1 line:        │ line 1:  rows 1,9,...,57  │ ...pad...
                    └────────────────────────────┘
                    ...
                    ┌────────────────────────────┐
Wave 7 line:        │ line 7:  rows 7,15,...,63 │ ...pad...
                    └────────────────────────────┘

D half 1: dim[64..127]
                    ┌────────────────────────────┐
Wave 0 line:        │ line 8:  rows 0,8,...,56  │ ...pad...
                    └────────────────────────────┘
                    ┌────────────────────────────┐
Wave 1 line:        │ line 9:  rows 1,9,...,57  │ ...pad...
                    └────────────────────────────┘
                    ...
                    ┌────────────────────────────┐
Wave 7 line:        │ line 15: rows 7,15,...,63 │ ...pad...
                    └────────────────────────────┘
```

Inside one K LDS line:

```text
Byte[0:15]       <- lane 0:  one 16B vector, row lane_n=0, dim lane_d=0
Byte[16:31]      <- lane 1:  one 16B vector, row lane_n=0, dim lane_d=1
...
Byte[112:127]    <- lane 7:  one 16B vector, row lane_n=0, dim lane_d=7
Byte[128:143]    <- lane 8:  one 16B vector, row lane_n=1, dim lane_d=0
...
Byte[1008:1023]  <- lane 63: one 16B vector, row lane_n=7, dim lane_d=7
Byte[1024:1039]  <- 8 bf16 padding
```

K VGPR read formula (`u_rk`) rearranges the padded LDS tile into GEMM0 operand order:

```text
lane_id_n = lane_id % W_N = lane_id % 32

p0 = lane_id_n % NUM_WARPS       // 0..7
p1 = lane_id_n / NUM_WARPS       // 0..3
p2 = lane_id / W_N               // 0 or 1

i_ngrp  = 0..1
i_dhalf = 0..1
i_k     = 0..3
vec     = 0..7

K read elem addr =
    p0 * 520
  + (i_ngrp * 4 + p1) * 64
  + i_dhalf * (8 * 520)
  + ((i_k * 2 + p2) * 8 + vec)
```

**C++ source** (`make_layout_rk`, `gqa_d128_kernel_template.hpp` lines 121-148):

```cpp
constexpr int n_per_wave = opus::get_warp_size() / (T::D_128B_SIZE / T::VEC_KV); // 64/8 = 8
constexpr int n_grp      = n_per_wave / (T::W_N / T::NUM_WARPS);                 // 8/(32/8) = 2
constexpr auto rk_block_shape = opus::make_tuple(
    opus::number<T::GEMM0_E_N / n_grp>{},          // 2/2 = 1
    opus::number<T::NUM_WARPS>{},                  // 8   (p0)
    opus::number<n_grp>{},                         // 2   (i_ngrp)
    opus::number<T::W_N / T::NUM_WARPS>{},         // 4   (p1)
    opus::number<T::smem_d_rpt>{},                 // 2   (i_dhalf)
    opus::number<T::GEMM0_E_K / T::smem_d_rpt>{},  // 8/2 = 4 (i_k)
    opus::number<opus::get_warp_size() / T::W_N>{},// 64/32 = 2 (p2)
    opus::number<T::VEC_KV>{});                    // 8   (vec)
constexpr auto rk_block_dim = opus::make_tuple(
    opus::make_tuple(opus::y_dim{}, opus::p_dim{}),                  // group 0
    opus::make_tuple(opus::y_dim{}, opus::p_dim{}),                  // group 1
    opus::make_tuple(opus::y_dim{}),                                 // group 2
    opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));  // group 3
auto lane_id_n = lane_id % T::W_N;
return opus::make_layout(
    rk_block_shape,
    opus::unfold_x_stride(rk_block_dim, rk_block_shape, opus::tuple{
        opus::number<T::smem_linear_wave + T::smem_padding_16B>{},                 // 520  (LDS line)
        opus::number<T::D_128B_SIZE>{},                                           // 64
        opus::number<T::smem_n_rpt * (T::smem_linear_wave + T::smem_padding_16B)>{}, // 8*520 = 4160 (d_half block)
        1_I}),
    opus::unfold_p_coord(rk_block_dim, opus::tuple{
        lane_id_n % T::NUM_WARPS, lane_id_n / T::NUM_WARPS, lane_id / T::W_N}));    // p0, p1, p2
```

**Derivation**: four stride groups, base strides `{520, 64, 4160, 1}` (note 520 = one LDS line = same line stride the `u_sk` write used; 4160 = `smem_n_rpt * 520` = one whole `d_half` block of 8 lines). The three `p_dim` axes bind in order to `{lane_id_n%NUM_WARPS, lane_id_n/NUM_WARPS, lane_id/W_N}` = `{p0, p1, p2}` (`lane_id_n = lane_id % W_N`).

| shape axis (size) | role | bound coord / loop var | unfolded stride | term |
|---|---|---|---|---|
| `GEMM0_E_N/n_grp` (1) | y | -- (size 1) | 520 x 8 | 0 |
| `NUM_WARPS` (8) | p | `lane_id_n % NUM_WARPS` (= `p0`) | 520 x 1 = 520 | `p0*520` |
| `n_grp` (2) | y | `i_ngrp` (0..1) | 64 x 4 = 256 | `i_ngrp*256` |
| `W_N/NUM_WARPS` (4) | p | `lane_id_n / NUM_WARPS` (= `p1`) | 64 x 1 = 64 | `p1*64` |
| `smem_d_rpt` (2) | y | `i_dhalf` (0..1) | 4160 x 1 = 4160 | `i_dhalf*4160` |
| `GEMM0_E_K/smem_d_rpt` (4) | y | `i_k` (0..3) | 1 x 2 x 8 = 16 | `i_k*16` |
| `WARP_SIZE/W_N` (2) | p | `lane_id / W_N` (= `p2`) | 1 x 8 = 8 | `p2*8` |
| `VEC_KV` (8) | y | `vec` (0..7) | 1 | `vec` |

Grouping the terms: `p0*520 + (i_ngrp*4 + p1)*64 + i_dhalf*(8*520) + (i_k*2 + p2)*8 + vec` -- exactly the `u_rk` formula above. The "cross-line" access (`p0` strides by a whole 520-element line) is what re-gathers the cooperatively-stored K stripe into MFMA `v_k` operand order.

Reversing this address through the LDS write layout gives the logical K coordinate:

```text
K row = (i_ngrp * 4 + p1) * NUM_WARPS + p0
K dim = i_dhalf * 64 + (i_k * 2 + p2) * 8 + vec
```

Per-lane K VGPR read expansion for `i_ngrp=0`, `i_dhalf=0`, `i_k=0`:

| lane_id | p0 | p1 | p2 | K row | D range |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 0 | `0..7` |
| 1 | 1 | 0 | 0 | 1 | `0..7` |
| 2 | 2 | 0 | 0 | 2 | `0..7` |
| ... | ... | ... | ... | ... | ... |
| 7 | 7 | 0 | 0 | 7 | `0..7` |
| 8 | 0 | 1 | 0 | 8 | `0..7` |
| 9 | 1 | 1 | 0 | 9 | `0..7` |
| ... | ... | ... | ... | ... | ... |
| 31 | 7 | 3 | 0 | 31 | `0..7` |
| 32 | 0 | 0 | 1 | 0 | `8..15` |
| 33 | 1 | 0 | 1 | 1 | `8..15` |
| ... | ... | ... | ... | ... | ... |
| 63 | 7 | 3 | 1 | 31 | `8..15` |

Thread/lane to LDS read address expansion for the same `i_ngrp=0`, `i_dhalf=0`, `i_k=0` slice:

```text
thread_id_x = warp_id * 64 + lane_id
K LDS byte range = [K read elem addr * 2, K read elem addr * 2 + 15]
```

| lane_id | thread_id_x | p0 | p1 | p2 | K LDS elem base | K LDS byte range | Logical K |
|---:|---:|---:|---:|---:|---:|---|---|
| 0 | `warp_id*64 + 0` | 0 | 0 | 0 | 0 | `0x000..0x00F` | row 0, dim `0..7` |
| 1 | `warp_id*64 + 1` | 1 | 0 | 0 | 520 | `0x410..0x41F` | row 1, dim `0..7` |
| 2 | `warp_id*64 + 2` | 2 | 0 | 0 | 1040 | `0x820..0x82F` | row 2, dim `0..7` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | `warp_id*64 + 7` | 7 | 0 | 0 | 3640 | `0x1C70..0x1C7F` | row 7, dim `0..7` |
| 8 | `warp_id*64 + 8` | 0 | 1 | 0 | 64 | `0x080..0x08F` | row 8, dim `0..7` |
| 9 | `warp_id*64 + 9` | 1 | 1 | 0 | 584 | `0x490..0x49F` | row 9, dim `0..7` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 31 | `warp_id*64 + 31` | 7 | 3 | 0 | 3832 | `0x1DF0..0x1DFF` | row 31, dim `0..7` |
| 32 | `warp_id*64 + 32` | 0 | 0 | 1 | 8 | `0x010..0x01F` | row 0, dim `8..15` |
| 33 | `warp_id*64 + 33` | 1 | 0 | 1 | 528 | `0x420..0x42F` | row 1, dim `8..15` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 63 | `warp_id*64 + 63` | 7 | 3 | 1 | 3840 | `0x1E00..0x1E0F` | row 31, dim `8..15` |

Notice that the K LDS read is intentionally not lane-contiguous. For example, lane 1 reads from byte `0x410`, i.e. K line 1, while lane 8 returns to byte `0x080` inside line 0. This cross-line access is exactly what converts the cooperative GM->LDS stripe layout into the MFMA-friendly `v_k` operand layout.

The loop axes then cover the remaining K tile:

| Axis | Values | Effect |
|---|---|---|
| `i_ngrp` | `0, 1` | selects K rows `0..31` then `32..63` |
| `i_dhalf` | `0, 1` | selects D half `0..63` then `64..127` |
| `i_k` | `0..3` | advances within a D half by `16` bf16 per step |
| `p2` | `0, 1` | lane half reads the low/high 8 bf16 inside each 16-bf16 step |

VGPR shape:

```text
v_k: vector<bf16, 64> per lane

It is not a simple row-major K vector; it is already packed as the MFMA operand
needed by:
  v_s = mma0(v_q, v_k)

GEMM0 logical tile:
  S[32 x 64] = Q[32 x 128] @ K^T[128 x 64]
```

<a id="v-gm-to-lds-to-vgpr"></a>

<a id="sec-2-3-v-gm---lds---vgpr"></a>
### 2.3 V: GM -> LDS -> VGPR

V uses the same GM coordinate layout as K and the same cooperative `async_load<VEC_KV>`, but it uses a larger LDS padding and a transpose LDS read.

<a id="v-mfma-a-operand-and-transpose-load"></a>

#### 2.3.1 MFMA A-Operand Convention and CDNA4 Transpose Load

When reading the layout maps below, distinguish the **mathematical left-hand matrix** from the **hardware MFMA A operand**. Both GEMMs in this kernel are built with `mfma_adaptor_swap_ab{}` (lines 336-346), so OPUS passes operands in mathematical order but the adaptor swaps them before calling the underlying MFMA:

```cpp
return base::operator()(b, a, c, ...);
```

Therefore the hardware `v_mfma_f32_32x32x16_bf16` source operands are:

| Source site | Mathematical GEMM | Source-level call | Mathematical left matrix | Hardware MFMA A operand (`src0`) | Hardware MFMA B operand (`src1`) |
|---|---|---|---|---|---|
| Cluster 1 | `S = Q @ K^T` | `mma0(v_q, v_k)` | `Q` / `v_q` | `K` / `v_k` | `Q` / `v_q` |
| Cluster 7 | `O += P @ V` | `mma1.step_k(..., v_p, v_v, v_o)` | `P` / `v_p` | `V` / `v_v` | `P` / `v_p` |

This is the main reason the V LDS read is special. In GEMM1/GEMM2, `v_v` is not just "the right-hand source operand" at the C++ call site; after `mfma_adaptor_swap_ab`, it becomes the **hardware MFMA A operand**. CDNA4 provides `DS_READ_B64_TR_B16` exactly for this kind of operand load.

CDNA4 ISA section **11.4 MFMA Transpose Load from LDS** defines the instruction family as LDS-to-VGPR loads that transpose matrix data while transferring 16-, 8-, 6-, or 4-bit elements. For this kernel the relevant mnemonic is:

```asm
ds_read_b64_tr_b16 v[dst:dst+1], vaddr offset:imm
```

Its per-lane interface is:

```text
Input:
  vaddr + offset  = LDS byte address
  EXEC            = all lanes active (ISA requirement)
  LDS data        = matrix fragment interpreted as 16-bit elements

Output:
  v[dst:dst+1]    = 64 bits per lane = 4 bf16 values
```

The `b64` part means one instruction returns `4 × bf16` per lane. The `tr_b16` part means the data is interpreted as 16-bit matrix elements and transposed on the LDS-to-VGPR path. A normal `ds_read_b64` would preserve each lane's contiguous 8-byte vector; `ds_read_b64_tr_b16` instead makes the lanes collectively read a matrix fragment and returns the MFMA-oriented transposed view.

ISA also states that a complete 16-bit transpose-load matrix is loaded by **two** instructions with different LDS addresses and VGPR destinations:

```text
Complete B16 transpose-load matrix:

  first  ds_read_b64_tr_b16: K slices 0..3  and 8..11
  second ds_read_b64_tr_b16: K slices 4..7  and 12..15

Here "K" is the MFMA reduction dimension, not the attention K tensor.
```

That is why the FlyDSL port's equivalent helper does:

```python
a = ds_read_tr_v4f16(lds_off_lo)
b = ds_read_tr_v4f16(lds_off_lo + OPUS_URV_I5_STRIDE)  # +64 bf16
pack = concat(a, b)                                    # 8 bf16 MFMA-A pack
```

The layout effect for one 16-lane V transpose group can be visualized as:

```text
Before ds_read_b64_tr_b16: LDS view written by DMA-friendly V layout

                 dim0..3    dim4..7    dim8..11   dim12..15
  V row 0        lane 0     lane 1     lane 2     lane 3
  V row 1        lane 4     lane 5     lane 6     lane 7
  V row 2        lane 8     lane 9     lane10     lane11
  V row 3        lane12     lane13     lane14     lane15

After hardware transpose: VGPR view consumed as MFMA A operand

  lane 0  -> [V(row0, dim0),  V(row1, dim0),  V(row2, dim0),  V(row3, dim0)]
  lane 1  -> [V(row0, dim1),  V(row1, dim1),  V(row2, dim1),  V(row3, dim1)]
  lane 2  -> [V(row0, dim2),  V(row1, dim2),  V(row2, dim2),  V(row3, dim2)]
  ...
  lane15  -> [V(row0, dim15), V(row1, dim15), V(row2, dim15), V(row3, dim15)]
```

In the dumped causal ISA, the first two V transpose-loads in the main-loop cluster appear as:

```asm
ds_read_b64_tr_b16 v[20:21], v222 offset:0
ds_read_b64_tr_b16 v[22:23], v222 offset:0x80
```

The following GEMM2 MFMA then uses the transpose-loaded V registers as `src0`, i.e. hardware A:

```asm
v_mfma_f32_32x32x16_bf16 v[80:95], v[20:23], v[112:115], v[80:95]
```

Here `v[20:23]` is V (`v_v`) and `v[112:115]` is P (`v_p`). This is why V uses the OPUS `u_rv` layout, `tr_load<VEC_TR_V>`, and the wider `smem_padding_64B`: the LDS layout is built so the hardware transpose load can turn the DMA-friendly stored V tile into an MFMA-A-friendly VGPR fragment without a software `ds_read + ds_permute/v_perm` transpose sequence.

GM address formula (`u_gv`):

```text
d_half = 0..1
vec    = 0..7

v_row = lane_n * NUM_WARPS + warp_id
v_dim = d_half * 64 + lane_d * 8 + vec

GM elem addr = kv_tile(j) + v_row * stride_kv_n + v_dim
```

**C++ source**: V reuses the *same* `make_layout_gk_gv` as K (`gqa_d128_kernel_template.hpp` line 409 calls `async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, ...)` with `u_gv = make_layout_gk_gv<T>(...)`). The GM read coordinate is therefore byte-for-byte the K derivation in 2.2 (`lane_n = lane_id/threads_d`, `lane_d = lane_id%threads_d`); only the destination padding (`u_sv`) differs.

V LDS buffer map:

```text
s_v[0] = smem_buf + smem_k_tile_elems
s_v[1] = smem_buf + smem_buffer_elems + smem_k_tile_elems

V line stride = smem_linear_wave + smem_padding_64B
              = 512 + 32 = 544 bf16 = 1088 B

V tile size   = 16 lines * 544 bf16 = 8704 bf16
```

V LDS address formula (`u_sv` as the `async_load` destination):

```text
line = d_half * NUM_WARPS + warp_id
in_line_elem = lane_id * VEC_KV + vec

V LDS elem addr = line * 544 + in_line_elem
V LDS byte addr = (line * 544 + lane_id * 8 + vec) * 2
```

**C++ source**: `u_sv = make_layout_sk_sv<T, T::smem_padding_64B>(warp_id)` (`gqa_d128_kernel_template.hpp` line 354) -- the *same* function as K's `u_sk` (see 2.2), only with `smem_padding_64B = 64/2 = 32` instead of `smem_padding_16B = 8`. So the group-0 base (line) stride is `smem_linear_wave + 32 = 544` instead of 520; every term in the derivation is identical with `520 -> 544`. The wider line keeps the 16-lane `ds_read_b64_tr_b16` transpose groups bank-conflict-free.

V LDS address space for one buffer:

```text
V LDS Address Space:
Address increases downward; each line is 1088B = 1024B data + 64B pad.

D half 0: dim[0..63]
                    ┌────────────────────────────┐
Wave 0 line:        │ line 0:  rows 0,8,...,56  │ ...pad...
                    └────────────────────────────┘
                    ...
                    ┌────────────────────────────┐
Wave 7 line:        │ line 7:  rows 7,15,...,63 │ ...pad...
                    └────────────────────────────┘

D half 1: dim[64..127]
                    ┌────────────────────────────┐
Wave 0 line:        │ line 8:  rows 0,8,...,56  │ ...pad...
                    └────────────────────────────┘
                    ...
                    ┌────────────────────────────┐
Wave 7 line:        │ line 15: rows 7,15,...,63 │ ...pad...
                    └────────────────────────────┘
```

Inside one V LDS line, the first 1024 bytes are lane-contiguous data exactly like K, followed by 64 bytes of padding:

```text
Byte[0:15]       <- lane 0
Byte[16:31]      <- lane 1
...
Byte[1008:1023]  <- lane 63
Byte[1024:1087]  <- 32 bf16 padding
```

V VGPR read formula (`u_rv`) is shaped for `ds_read_tr16_b64`.

<a id="ds-read-tr16-b64-semantics"></a>

<a id="sec-2-3-2-ds_read_tr16_b64-ds_read_b64_tr_b16-instruction-semantics"></a>
#### 2.3.2 `ds_read_tr16_b64` / `ds_read_b64_tr_b16` instruction semantics

On MI355X / gfx950, OPUS lowers `tr_load<VEC_TR_V>` to the CDNA4 LDS hardware-transpose read instruction. The CDNA4 ISA documents this family in **11.4 MFMA Transpose Load from LDS**: these instructions transpose matrix data while transferring 16-, 8-, 6-, or 4-bit elements from LDS to VGPRs. In the C++ helper this appears as the ISA mnemonic:

```cpp
ds_read_b64_tr_b16 %dst, %addr, offset:imm
```

In FlyDSL / MLIR ROCDL the same operation is exposed as:

```python
rocdl.ds_read_tr16_b64(v4f16_type, ptr)
```

The names emphasize different parts of the same operation:

| Name part | Meaning |
|---|---|
| `ds_read` | Read from LDS / shared memory. |
| `b64` | Each lane loads 64 bits = 8 bytes. For bf16 this is `4 × bf16`. |
| `tr_b16` / `tr16` | Treat the 64-bit lane payload as 16-bit elements and apply the MFMA transpose-load mapping. |

Semantically, this is not a normal `ds_read_b64`. A normal LDS read would give each lane the four bf16 values that are contiguous at that lane's address. The transpose variant makes the participating lanes collectively read a matrix fragment from LDS and returns the transposed view in VGPRs. MLIR describes this family as: each lane reads a vector from a column-major matrix in LDS, and the lane result becomes a row of the transposed matrix.

ISA constraints that matter here:

| ISA rule | Kernel implication |
|---|---|
| `EXEC` must be all 1s before executing these instructions. | The OPUS V read is issued in the uniform GEMM2 path, not under a per-lane predicate. |
| LDS address must be aligned to the element size. | bf16 `tr_b16` addresses are at least 2-byte aligned; OPUS computes them in bf16 element units and converts to bytes. |
| DS ops reading/writing 64-bit or larger data require even-aligned VGPR destinations. | The compiler allocates the 64-bit return as an aligned VGPR pair / `vector<4xbf16>`. |
| A complete 16-bit transpose-load matrix is loaded by two instructions with different LDS addresses and VGPR destinations. | This is why the FlyDSL port emits two reads, `a = ds_read_tr_v4f16(lds_off_lo)` and `b = ds_read_tr_v4f16(lds_off_lo + 64)`, then concatenates them into one `<8xbf16>` MFMA pack. |

For this kernel:

```text
one DS_READ_B64_TR_B16 / rocdl.ds_read_tr16_b64:
  per-lane payload = 64 bits = 4 bf16 = VEC_TR_V
  result per lane  = vector<4xbf16>, already transposed for MFMA operand use

two DS_READ_B64_TR_B16 instructions:
  complete the 16-bit transpose-load matrix for one OPUS V pack
  first instruction covers K slices 0..3 and 8..11
  second instruction covers K slices 4..7 and 12..15
  combined result per lane = 8 bf16, i.e. one MFMA V operand pack
```

This is exactly what GEMM2 needs. V is stored in LDS in a DMA-friendly layout where rows and D slices are contiguous enough for `buffer_load_dwordx4_lds`. GEMM2, however, consumes V as the MFMA A operand for:

```text
O[32 × 128] += P[32 × 64] @ V[64 × 128]
```

so the operand seen by MFMA is effectively `V^T`: D-chunk fragments must be presented across the MFMA M direction while the KV dimension becomes the reduction direction. `ds_read_tr16_b64` performs that lane/data transpose during the LDS read, avoiding a software sequence such as:

```text
ds_read_b64 / ds_read_b128
  + ds_permute_b32 / v_perm_b32
  + extra VGPR temporaries
```

The cost is that the LDS layout must be built for the hardware transpose. That is why V uses `smem_padding_64B` (32 bf16 padding per line) instead of K's `smem_padding_16B`: the wider line stride keeps the 16-lane transpose groups bank-conflict-free while preserving the row-major DMA store pattern.

```text
lane_per_grp = 16
lane_lo      = 4
lane_hi      = 4
num_grps     = 4
grp_n        = 2
grp_k        = 2

grp_id      = lane_id / 16        // 0..3
lane_in_grp = lane_id % 16

p0 = grp_id / grp_n               // 0..1
p1 = lane_in_grp / lane_lo        // 0..3
p2 = grp_id % grp_n               // 0..1
p3 = lane_in_grp % lane_lo        // 0..3
```

Per-lane `u_rv` transpose-group expansion:

| lane_id range | grp_id | lane_in_grp | p0 | p1 | p2 | p3 | Meaning |
|---:|---:|---:|---:|---:|---:|---:|---|
| `0..3` | 0 | `0..3` | 0 | 0 | 0 | `0..3` | group 0, first 4-lane lo subgroup |
| `4..7` | 0 | `4..7` | 0 | 1 | 0 | `0..3` | group 0, second 4-lane lo subgroup |
| `8..11` | 0 | `8..11` | 0 | 2 | 0 | `0..3` | group 0, third 4-lane lo subgroup |
| `12..15` | 0 | `12..15` | 0 | 3 | 0 | `0..3` | group 0, fourth 4-lane lo subgroup |
| `16..19` | 1 | `0..3` | 0 | 0 | 1 | `0..3` | group 1, first 4-lane lo subgroup |
| `20..23` | 1 | `4..7` | 0 | 1 | 1 | `0..3` | group 1, second 4-lane lo subgroup |
| `24..27` | 1 | `8..11` | 0 | 2 | 1 | `0..3` | group 1, third 4-lane lo subgroup |
| `28..31` | 1 | `12..15` | 0 | 3 | 1 | `0..3` | group 1, fourth 4-lane lo subgroup |
| `32..35` | 2 | `0..3` | 1 | 0 | 0 | `0..3` | group 2, first 4-lane lo subgroup |
| `36..39` | 2 | `4..7` | 1 | 1 | 0 | `0..3` | group 2, second 4-lane lo subgroup |
| `40..43` | 2 | `8..11` | 1 | 2 | 0 | `0..3` | group 2, third 4-lane lo subgroup |
| `44..47` | 2 | `12..15` | 1 | 3 | 0 | `0..3` | group 2, fourth 4-lane lo subgroup |
| `48..51` | 3 | `0..3` | 1 | 0 | 1 | `0..3` | group 3, first 4-lane lo subgroup |
| `52..55` | 3 | `4..7` | 1 | 1 | 1 | `0..3` | group 3, second 4-lane lo subgroup |
| `56..59` | 3 | `8..11` | 1 | 2 | 1 | `0..3` | group 3, third 4-lane lo subgroup |
| `60..63` | 3 | `12..15` | 1 | 3 | 1 | `0..3` | group 3, fourth 4-lane lo subgroup |

The table shows the 64-lane wave as four `ds_read_tr16_b64` transpose groups. Within each 16-lane group, `p1` selects one of the four 4-lane high subgroups, while `p3` is the lane position inside the 4-lane low subgroup. `p0` and `p2` split the four groups into the two `grp_k` and two `grp_n` directions used by the V operand layout.

`u_rv` LDS element address:

```text
i0  = 0..1
i1  = 0..1
i4a = 0..3
i4b = 0..1
vec = 0..3

V read elem addr =
    i0 * (8 * 544)
  + i1 * (grp_n * lane_lo * VEC_TR_V)          // i1 * 32
  + (p0 * lane_hi + p1) * 544
  + (i4a * 2 + i4b) * 64
  + ((p2 * lane_lo + p3) * VEC_TR_V + vec)
```

**C++ source** (`make_layout_rv`, `gqa_d128_kernel_template.hpp` lines 150-185):

```cpp
constexpr int lane_per_grp = 16;
constexpr int lane_lo = 4;
constexpr int lane_hi = lane_per_grp / lane_lo;            // 4
constexpr int num_grps = T::WARP_SIZE / lane_per_grp;      // 64/16 = 4
constexpr int grp_n = T::W_N / (lane_lo * T::VEC_TR_V);    // 32/(4*4) = 2
constexpr int grp_k = num_grps / grp_n;                    // 4/2 = 2
constexpr auto rv_block_shape = opus::make_tuple(
    opus::number<T::GEMM1_E_N / (T::D_128B_SIZE / T::W_N)>{}, // 4/(64/32) = 2 (i0)
    opus::number<T::D_128B_SIZE / T::W_N>{},                  // 2   (i1)
    opus::number<grp_k>{},                                    // 2   (p0)
    opus::number<lane_hi>{},                                  // 4   (p1)
    opus::number<T::GEMM1_E_K>{},                             // 4   (i4a)
    opus::number<T::W_K / (lane_hi * grp_k)>{},               // 16/8 = 2 (i4b)
    opus::number<grp_n>{},                                    // 2   (p2)
    opus::number<lane_lo>{},                                  // 4   (p3)
    opus::number<T::VEC_TR_V>{});                             // 4   (vec)
constexpr auto rv_block_dim = opus::make_tuple(
    opus::make_tuple(opus::y_dim{}),                                 // group 0
    opus::make_tuple(opus::y_dim{}),                                 // group 1
    opus::make_tuple(opus::p_dim{}, opus::p_dim{}),                  // group 2
    opus::make_tuple(opus::y_dim{}, opus::y_dim{}),                  // group 3
    opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));  // group 4
int grp_id = lane_id / lane_per_grp;       // 0..3
int lane_in_grp = lane_id % lane_per_grp;  // 0..15
return opus::make_layout(
    rv_block_shape,
    opus::unfold_x_stride(rv_block_dim, rv_block_shape, opus::tuple{
        opus::number<T::smem_n_rpt * (T::smem_linear_wave + T::smem_padding_64B)>{}, // 8*544 = 4352
        opus::number<grp_n * lane_lo * T::VEC_TR_V>{},                              // 2*4*4 = 32
        opus::number<T::smem_linear_wave + T::smem_padding_64B>{},                  // 544 (V LDS line)
        opus::number<T::D_128B_SIZE>{},                                            // 64
        1_I}),
    opus::unfold_p_coord(rv_block_dim, opus::tuple{
        grp_id / grp_n, lane_in_grp / lane_lo, grp_id % grp_n, lane_in_grp % lane_lo})); // p0,p1,p2,p3
```

**Derivation**: five stride groups, base strides `{4352, 32, 544, 64, 1}` (4352 = `smem_n_rpt*544` = one 8-line `d_half` block of the V tile; 544 = one V LDS line, wider than K's 520 because of `smem_padding_64B`). The four `p_dim` axes bind in order to `{grp_id/grp_n, lane_in_grp/lane_lo, grp_id%grp_n, lane_in_grp%lane_lo}` = `{p0, p1, p2, p3}` (with `grp_id = lane_id/16`, `lane_in_grp = lane_id%16` -- the 16-lane transpose group structure required by `ds_read_b64_tr_b16`).

| shape axis (size) | role | bound coord / loop var | unfolded stride | term |
|---|---|---|---|---|
| `GEMM1_E_N/(D_128B/W_N)` (2) | y | `i0` (0..1) | 4352 x 1 = 4352 | `i0*(8*544)` |
| `D_128B/W_N` (2) | y | `i1` (0..1) | 32 x 1 = 32 | `i1*32` |
| `grp_k` (2) | p | `grp_id/grp_n` (= `p0`) | 544 x 4 = 2176 | `p0*lane_hi*544` |
| `lane_hi` (4) | p | `lane_in_grp/lane_lo` (= `p1`) | 544 x 1 = 544 | `p1*544` |
| `GEMM1_E_K` (4) | y | `i4a` (0..3) | 64 x 2 = 128 | `i4a*2*64` |
| `W_K/(lane_hi*grp_k)` (2) | y | `i4b` (0..1) | 64 x 1 = 64 | `i4b*64` |
| `grp_n` (2) | p | `grp_id%grp_n` (= `p2`) | 1 x 4 x 4 = 16 | `p2*lane_lo*VEC_TR_V` |
| `lane_lo` (4) | p | `lane_in_grp%lane_lo` (= `p3`) | 1 x 4 = 4 | `p3*VEC_TR_V` |
| `VEC_TR_V` (4) | y | `vec` (0..3) | 1 | `vec` |

Collecting terms gives `i0*(8*544) + i1*32 + (p0*lane_hi + p1)*544 + (i4a*2 + i4b)*64 + ((p2*lane_lo + p3)*VEC_TR_V + vec)` -- the `u_rv` formula above. Because this layout is consumed by `tr_load` -> `ds_read_b64_tr_b16`, the `p_dim` axes are exactly the transpose-group coordinates (`grp_k`/`lane_hi` choose the row-side group, `grp_n`/`lane_lo` the 4-bf16 D chunk), so the hardware transpose delivers `v_v` already in MFMA-A order.

Reversing this address through the V LDS write layout gives the logical V coordinate:

```text
V row = (i4a * 2 + i4b) * NUM_WARPS + (p0 * lane_hi + p1)
V dim = i0 * 64 + i1 * 32 + (p2 * lane_lo + p3) * VEC_TR_V + vec
```

Per-lane V LDS read expansion for `i0=0`, `i1=0`, `i4a=0`, `i4b=0`:

| lane_id range | p0 | p1 | p2 | p3 | V LDS elem base range | V LDS byte range | Logical V |
|---:|---:|---:|---:|---:|---|---|---|
| `0..3` | 0 | 0 | 0 | `0..3` | `0, 4, 8, 12` | `0x000..0x01F` | row 0, dim `0..15` |
| `4..7` | 0 | 1 | 0 | `0..3` | `544, 548, 552, 556` | `0x440..0x45F` | row 1, dim `0..15` |
| `8..11` | 0 | 2 | 0 | `0..3` | `1088, 1092, 1096, 1100` | `0x880..0x89F` | row 2, dim `0..15` |
| `12..15` | 0 | 3 | 0 | `0..3` | `1632, 1636, 1640, 1644` | `0xCC0..0xCDF` | row 3, dim `0..15` |
| `16..19` | 0 | 0 | 1 | `0..3` | `16, 20, 24, 28` | `0x020..0x03F` | row 0, dim `16..31` |
| `20..23` | 0 | 1 | 1 | `0..3` | `560, 564, 568, 572` | `0x460..0x47F` | row 1, dim `16..31` |
| `24..27` | 0 | 2 | 1 | `0..3` | `1104, 1108, 1112, 1116` | `0x8A0..0x8BF` | row 2, dim `16..31` |
| `28..31` | 0 | 3 | 1 | `0..3` | `1648, 1652, 1656, 1660` | `0xCE0..0xCFF` | row 3, dim `16..31` |
| `32..35` | 1 | 0 | 0 | `0..3` | `2176, 2180, 2184, 2188` | `0x1100..0x111F` | row 4, dim `0..15` |
| `36..39` | 1 | 1 | 0 | `0..3` | `2720, 2724, 2728, 2732` | `0x1540..0x155F` | row 5, dim `0..15` |
| `40..43` | 1 | 2 | 0 | `0..3` | `3264, 3268, 3272, 3276` | `0x1980..0x199F` | row 6, dim `0..15` |
| `44..47` | 1 | 3 | 0 | `0..3` | `3808, 3812, 3816, 3820` | `0x1DC0..0x1DDF` | row 7, dim `0..15` |
| `48..51` | 1 | 0 | 1 | `0..3` | `2192, 2196, 2200, 2204` | `0x1120..0x113F` | row 4, dim `16..31` |
| `52..55` | 1 | 1 | 1 | `0..3` | `2736, 2740, 2744, 2748` | `0x1560..0x157F` | row 5, dim `16..31` |
| `56..59` | 1 | 2 | 1 | `0..3` | `3280, 3284, 3288, 3292` | `0x19A0..0x19BF` | row 6, dim `16..31` |
| `60..63` | 1 | 3 | 1 | `0..3` | `3824, 3828, 3832, 3836` | `0x1DE0..0x1DFF` | row 7, dim `16..31` |

The remaining axes complete the full V tile:

| Axis | Values | Effect |
|---|---|---|
| `i0` | `0, 1` | selects D half `0..63` then `64..127` |
| `i1` | `0, 1` | selects the low/high 32 bf16 within that D half |
| `i4a` | `0..3` | advances V row group by `16` rows per step |
| `i4b` | `0, 1` | advances V row group by `8` rows within each `i4a` |
| `p0/p1` | lane-derived | selects rows `0..7` inside each 8-row group |
| `p2/p3` | lane-derived | selects 4-bf16 chunks inside each 32-bf16 D segment |

VGPR shape:

```text
v_v = tr_load<VEC_TR_V>(s_v[buf], u_rv)

The compiler lowers this to ds_read_tr16_b64. Each 16-lane group reads 4 bf16
per lane and the hardware transpose presents the data in the orientation needed by:
  v_o = mma1(v_p, v_v, v_o)

GEMM1 logical tile:
  O[32 x 128] += P[32 x 64] @ V[64 x 128]
```

<a id="s-and-p-vgpr-only-layout"></a>

### 2.4 S and P: VGPR-only Score/Probability Layout

S never touches GM or LDS. It is the FP32 output of GEMM0:

```text
v_s[0], v_s[1]: vector<float, 32> per lane

Logical tile per warp:
  S[32 x 64] = Q[32 x 128] @ K^T[128 x 64]
```

S VGPR index formula:

```text
i_n  = 0..1        // 32-column N strip
rept = 0..3
j    = 0..3

idx = i_n * 16 + rept * 4 + j

q_row = warp_id * 32 + lane_m
k_col = i_n * 32 + lane_group * 4 + rept * 8 + j
```

Per-lane expansion for one warp:

| lane_id | q row inside warp | lane_group | `v_s` K columns |
|---:|---:|---:|---|
| 0 | 0 | 0 | `0..3, 8..11, 16..19, 24..27, 32..35, 40..43, 48..51, 56..59` |
| 32 | 0 | 1 | `4..7, 12..15, 20..23, 28..31, 36..39, 44..47, 52..55, 60..63` |
| 1 | 1 | 0 | same K-column pattern as lane 0, for Q row 1 |
| 33 | 1 | 1 | same K-column pattern as lane 32, for Q row 1 |

This is why `attn_row_max` and `attn_sum` use `permlane32_swap`: lanes `x` and `x+32` together hold the full 64-column score row for one Q row.

After masking, max/subtract, and `exp2`, the probability fragment uses the same index layout:

```text
v_p = cast<bf16>(v_s)
v_p: vector<bf16, 32> per lane
```

<a id="o-vgpr-to-gm"></a>

<a id="sec-2-5-o-vgpr---gm"></a>
### 2.5 O: VGPR -> GM

O is accumulated in VGPRs and only written to GM at the end. It has no LDS residency.

VGPR accumulator:

```text
v_o: vector<float, 64> per lane

Logical tile per warp:
  O[32 x 128] += P[32 x 64] @ V[64 x 128]

GEMM1_E_N = 4      // four 32-wide D strips
GEMM1_E_K = 4      // four 16-wide KV substeps
```

`v_o` index formula mirrors the S layout, replacing the S N-axis with O's D-axis:

```text
d_strip = 0..3
rept    = 0..3
j       = 0..3

idx = d_strip * 16 + rept * 4 + j

o_row = warp_id * 32 + lane_m
o_dim = d_strip * 32 + lane_group * 4 + rept * 8 + j
```

Final GM store (`u_o`):

```text
g_o = ptr_o
    + b * stride_q_b
    + q_block_start * stride_q_n
    + h * stride_q_h

Before store:
  l_inv = 1 / l_row
  v_o *= l_inv
  v_o_bf16 = cast<bf16>(v_o)

store<VEC_O=4>(g_o, v_o_bf16, u_o)
```

`make_layout_o` store address formula:

```text
d_strip = 0..3
pack    = 0..3
vec     = 0..3

o_row = warp_id * 32 + lane_m
o_dim = d_strip * 32 + pack * 8 + lane_group * 4 + vec

GM elem addr = o_row * stride_q_n + o_dim
```

**C++ source** (`make_layout_o`, `gqa_d128_kernel_template.hpp` lines 54-73):

```cpp
constexpr auto o_block_shape = opus::make_tuple(
    opus::number<T::GEMM1_E_M>{},                          // 1
    opus::number<T::T_M>{},                                // NUM_WARPS = 8
    opus::number<T::W_M>{},                                // 32
    opus::number<T::GEMM1_E_N>{},                          // 4   (d_strip)
    opus::number<T::W_M * T::W_N / T::WARP_SIZE / T::VEC_O>{}, // 32*32/64/4 = 4 (pack)
    opus::number<T::WARP_SIZE / T::W_M>{},                 // 2   (lane_group)
    opus::number<T::VEC_O>{});                             // 4   (vec)
constexpr auto o_block_dim = opus::make_tuple(
    opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),                 // group 0
    opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{})); // group 1
return opus::make_layout(
    o_block_shape,
    opus::unfold_x_stride(o_block_dim, o_block_shape, opus::tuple{stride_o_n, 1_I}),
    opus::unfold_p_coord(o_block_dim, opus::tuple{warp_id, lane_id % T::W_M, lane_id / T::W_M}));
```

**Derivation**: same two-group shape as Q's `make_layout_q`, with the GEMM0 K-axis replaced by GEMM1's D-axis. Base strides `{stride_o_n, 1}`; the three `p_dim` axes bind to `{warp_id, lane_id%W_M, lane_id/W_M}`.

| shape axis (size) | role | bound coord / loop var | unfolded stride | term |
|---|---|---|---|---|
| `GEMM1_E_M` (1) | y | -- (size 1) | `stride_o_n` x 8 x 32 | 0 |
| `T_M` (8) | p | `warp_id` | `stride_o_n` x 32 | `warp_id*32 * stride_o_n` |
| `W_M` (32) | p | `lane_id % W_M` (= `lane_m`) | `stride_o_n` x 1 | `(lane_id%32) * stride_o_n` |
| `GEMM1_E_N` (4) | y | `d_strip` (0..3) | 1 x 4 x 2 x 4 = 32 | `d_strip*32` |
| `W_M*W_N/WARP_SIZE/VEC_O` (4) | y | `pack` (0..3) | 1 x 2 x 4 = 8 | `pack*8` |
| `WARP_SIZE/W_M` (2) | p | `lane_id / W_M` (= `lane_group`) | 1 x 4 = 4 | `lane_group*4` |
| `VEC_O` (4) | y | `vec` (0..3) | 1 | `vec` |

So `o_row = warp_id*32 + lane_id%32` (group-0 base `stride_o_n`) and `o_dim = d_strip*32 + pack*8 + lane_group*4 + vec` (group-1 base 1) -- the formula above (`stride_o_n` equals `stride_q_n` here, since Q and O share the `num_heads`/`head_dim` layout).

Per-lane store expansion:

| lane_id | row inside warp | lane_group | O dims stored |
|---:|---:|---:|---|
| 0 | 0 | 0 | `0..3, 8..11, 16..19, 24..27, 32..35, ..., 120..123` |
| 32 | 0 | 1 | `4..7, 12..15, 20..23, 28..31, 36..39, ..., 124..127` |
| 1 | 1 | 0 | same D pattern as lane 0, for O row 1 |
| 33 | 1 | 1 | same D pattern as lane 32, for O row 1 |

<a id="tensor-residency-summary"></a>

### 2.6 Summary: Where Each Tensor Lives

```text
Q:
  GM [B, N, H, D]
    -> u_q load
    -> VGPR v_q[64] bf16, pre-scaled
  No LDS.

K:
  GM [B, N, H_KV, D]
    -> u_gk async_load
    -> LDS s_k[2], each tile 8320 bf16, 16B padding per line
    -> u_rk load
    -> VGPR v_k[64] bf16, GEMM0 operand

V:
  GM [B, N, H_KV, D]
    -> u_gv async_load
    -> LDS s_v[2], each tile 8704 bf16, 64B padding per line
    -> u_rv tr_load / ds_read_tr16_b64
    -> VGPR v_v, GEMM1 operand

S:
  VGPR only.
  v_s[0/1][32] fp32.
  Lanes x and x+32 together hold one full 64-column score row.

P:
  VGPR only.
  v_p[32] bf16, same logical index layout as S after softmax.

O:
  VGPR v_o[64] fp32 accumulator
    -> normalize by l_row
    -> cast bf16
    -> u_o store
    -> GM [B, N, H, D]
  No LDS.
```

<a id="low-level-call-chains-and-issue-counts"></a>

### 2.7 Low-Level Call Chains and Issue Counts

The source-level calls in the cluster code are high-level OPUS layout operations. The table below records the final backend primitive and the number of dynamic wave-instructions emitted by one source statement in the D=128 kernel.

| Source statement | High-level role | Lowest-level primitive | ISA form | Count per wave |
|---|---|---|---|---:|
| `async_load<T::VEC_KV>(g_k, s_k[1].ptr, u_gk, u_sk, kv_tile(j))` | GM K tile -> LDS K buffer | `__builtin_amdgcn_raw_ptr_buffer_load_lds(...)` | `buffer_load_dwordx4 ... lds` | 2 |
| `async_load<T::VEC_KV>(g_v, s_v[0].ptr, u_gv, u_sv, kv_tile(j - 1))` | GM V tile -> LDS V buffer | `__builtin_amdgcn_raw_ptr_buffer_load_lds(...)` | `buffer_load_dwordx4 ... lds` | 2 |
| `v_k = load<T::VEC_KV>(s_k[buf], u_rk)` | LDS K -> VGPR MFMA operand | LDS address-space vector load (`smem::_load`) | `ds_read_b128` | 16 |
| `v_v = tr_load<T::VEC_TR_V>(s_v[buf], u_rv)` | LDS V -> VGPR MFMA-A operand | inline asm `ds_read_b64_tr_b16` (`smem::_tr_load`) | `ds_read_b64_tr_b16` | 32 |
| `v_s[x] = mma0(v_q, v_k)` | GEMM0 score tile | `__builtin_amdgcn_mfma_f32_32x32x16_bf16(...)` | `v_mfma_f32_32x32x16_bf16` | 16 |
| `v_o = mma1.step_k(k, v_p, v_v, v_o)` | One GEMM1 reduction sub-step | `__builtin_amdgcn_mfma_f32_32x32x16_bf16(...)` | `v_mfma_f32_32x32x16_bf16` | 4 |

The relevant call chains are:

```text
async_load<T::VEC_KV>(g_k/g_v, ...)
  -> opus::async_load(...)
  -> gmem::async_load(layoutG, layoutS)
  -> gmem::_async_load(...)
  -> __builtin_amdgcn_raw_ptr_buffer_load_lds(...)
  -> buffer_load_dwordx4 ... lds

load<T::VEC_KV>(s_k, u_rk)
  -> opus::load(...)
  -> smem::load(layout)
  -> smem::_load(offset)
  -> LDS address-space vector pointer dereference
  -> ds_read_b128

tr_load<T::VEC_TR_V>(s_v, u_rv)
  -> opus::tr_load(...)
  -> smem::tr_load(layout)
  -> smem::_tr_load<vec, imm_offset>(base)
  -> asm volatile("ds_read_b64_tr_b16 ...")

mma0(v_q, v_k)
  -> tiled_mma_adaptor::operator()
  -> mfma_adaptor_swap_ab::operator()
  -> mfma::operator()
  -> __builtin_amdgcn_mfma_f32_32x32x16_bf16(...)

mma1.step_k(k, v_p, v_v, v_o)
  -> tiled_mma_adaptor::step_k<k>()
  -> mfma_adaptor_swap_ab::operator()
  -> mfma::operator()
  -> __builtin_amdgcn_mfma_f32_32x32x16_bf16(...)
```
