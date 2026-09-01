# FlyDSL Dual-Wave Software-Pipelined Flash-Attention (gfx950) Deep Introduction

<!-- markdown-toc-generator:start -->
## Table of Contents

- [Authoritative Sources](#authoritative-sources)
- [Benchmark / Trait Configuration](#benchmark-trait-configuration)
  - [Compiled resource usage (from 21_final_isa.s, lines 3740-3871)](#sec-compiled-resource-usage-from-21_final_isa-s-lines-3740-3871)
- [1. Kernel overview and software pipeline](#1-kernel-overview-and-software-pipeline)
  - [1.1 Buffering: where Q / K / V / S / P / O live (GM / LDS / VGPR)](#sec-1-1-buffering-where-q-k-v-s-p-o-live-gm-lds-vgpr)
    - [VGPR residency of v_s_0, v_s_1, v_p (from the ISA)](#sec-vgpr-residency-of-v_s_0-v_s_1-v_p-from-the-isa)
  - [1.2 Online softmax split into sub-stages](#12-online-softmax-split-into-sub-stages)
  - [1.3 Two wave-groups: group A leads by one cluster](#13-two-wave-groups-group-a-leads-by-one-cluster)
  - [1.4 Refined pseudocode](#14-refined-pseudocode)
  - [1.5 Per-operation main instructions and counts](#15-per-operation-main-instructions-and-counts)
    - [Whole-kernel ISA verification](#whole-kernel-isa-verification)
  - [1.6 Scheduling primitives: sched_group_barrier pairs and sched_barrier(0)](#sec-1-6-scheduling-primitives-sched_group_barrier-pairs-and-sched_barrier0)
    - [_sched_barrier_pairs(pairs, valu_cnt, group) (source 456-464)](#sec-_sched_barrier_pairspairs-valu_cnt-group-source-456-464)
    - [_sched_barrier_exp_pairs(pairs, exp_cnt, group) (source 466-474)](#sec-_sched_barrier_exp_pairspairs-exp_cnt-group-source-466-474)
    - [How one compute cluster uses both](#how-one-compute-cluster-uses-both)
    - [sched_barrier(0) (the hard fence)](#sec-sched_barrier0-the-hard-fence)
  - [1.7 KV-tile distribution and where the causal mask runs](#17-kv-tile-distribution-and-where-the-causal-mask-runs)
    - [1.7.1 How many KV tiles each stage owns](#171-how-many-kv-tiles-each-stage-owns)
    - [1.7.2 Worked tile-count cases](#172-worked-tile-count-cases)
    - [1.7.3 Which tile a wave first masks (the diagonal tile)](#sec-1-7-3-which-tile-a-wave-first-masks-the-diagonal-tile)
    - [1.7.4 Why the main loop's v_s_1 slot skips the causal mask](#sec-1-7-4-why-the-main-loops-v_s_1-slot-skips-the-causal-mask)
  - [1.8 Worked examples: block allocation and per-wave causal-mask sites](#18-worked-examples-block-allocation-and-per-wave-causal-mask-sites)
    - [1.8.1 Grid (block) allocation](#sec-1-8-1-grid-block-allocation)
    - [1.8.2 Per-q-block tile count E and mask pattern](#sec-1-8-2-per-q-block-tile-count-e-and-mask-pattern)
    - [1.8.3 Putting it together per scale](#183-putting-it-together-per-scale)
    - [1.8.4 Summary](#184-summary)
    - [1.8.5 What causal_num_tiles means, and the Pattern-C "1 real row" case](#sec-1-8-5-what-causal_num_tiles-means-and-the-pattern-c-1-real-row-case)
    - [1.8.6 Code evidence for the Pattern-C per-wave mask sites (S = 513, b = 2, E = 10)](#sec-1-8-6-code-evidence-for-the-pattern-c-per-wave-mask-sites-s-513-b-2-e-10)
- [2. Full GM/LDS/VGPR layout maps](#2-full-gmldsvgpr-layout-maps)
  - [2.1 Q: GM -> VGPR](#sec-2-1-q-gm---vgpr)
  - [2.2 K: GM -> LDS -> VGPR](#sec-2-2-k-gm---lds---vgpr)
  - [2.3 V: GM -> LDS -> VGPR (HW transpose load)](#sec-2-3-v-gm---lds---vgpr-hw-transpose-load)
    - [2.3.1 MFMA A-operand convention and CDNA4 transpose load](#231-mfma-a-operand-convention-and-cdna4-transpose-load)
    - [2.3.2 ds_read_b64_tr_b16 instruction semantics](#sec-2-3-2-ds_read_b64_tr_b16-instruction-semantics)
    - [2.3.3 u_rv address formula and lane map](#sec-2-3-3-u_rv-address-formula-and-lane-map)
  - [2.4 S and P: VGPR-only score / probability layout](#24-s-and-p-vgpr-only-score-probability-layout)
  - [2.5 O: VGPR -> GM](#sec-2-5-o-vgpr---gm)
  - [2.6 Summary: where each tensor lives](#26-summary-where-each-tensor-lives)
  - [2.7 Low-level call chains and issue counts](#27-low-level-call-chains-and-issue-counts)

<!-- markdown-toc-generator:end -->

This document deep-dives the FlyDSL kernel `flash_attn_dualwave_swp_gfx950_kernel`, the
hand-pipelined gfx950 (CDNA4 / MI350X-MI355X) fast path of FlyDSL flash attention. It is the
FlyDSL twin of the C++ OPUS `gqa_d128_kernel_template.hpp` kernel: it computes the same math and
reuses the same Q/K/V/O + LDS + MFMA addressing, but replaces the compiler-driven schedule with an
explicit 8-cluster software pipeline plus a two-wave-group time-multiplexing scheme.

Every instruction count, register-residency claim, and layout/stride formula below is traced to the
kernel source and verified against the compiled assembly. Every instruction's *semantics* are taken
from the AMD CDNA4 ISA documents, not from memory.

## Authoritative Sources

| Role | File |
|---|---|
| Target kernel source | `FlyDSL/kernels/flash_attn_gfx950.py` |
| Compiled ISA (ground truth) | `FlyDSL/flydsl_dump/flash_attn_dualwave_swp_gfx950_kernel_0/21_final_isa.s` (3878 lines) |
| LLVM IR (intermediate) | `FlyDSL/flydsl_dump/flash_attn_dualwave_swp_gfx950_kernel_0/20_llvm_ir.ll` |
| Sibling C++ implementation | OPUS `gqa_d128_kernel_template.hpp` (the structure mirrored by this port) |
| CDNA4 architecture whitepaper | `.cursor/rules/amd-cdna-4-architecture-whitepaper.txt` |
| CDNA4 ISA (instruction semantics) | `.cursor/rules/amd-instinct-cdna4-instruction-set-architecture.txt` |

## Benchmark / Trait Configuration

The ISA dump used for all counts is the **causal bf16, D=128** build.

```
Arch              : gfx950 (CDNA4, MI350X / MI355X), wavefront = 64
dtype             : bf16 (f16 supported by the same builder)
causal            : true (this dump)
Launch shape      : grid = (num_heads, num_q_blocks, batch), block = (512, 1, 1) = 8 waves
Tile shape        : BLOCK_M = 256 (8 waves x 32 rows), BLOCK_N = 64, HEAD_DIM = 128
MFMA              : v_mfma_f32_32x32x16_bf16   (W_M=32, W_N=32, W_K=16)
GQA               : num_kv_heads <= num_heads supported
```

<a id="sec-compiled-resource-usage-from-21_final_isa-s-lines-3740-3871"></a>
### Compiled resource usage (from `21_final_isa.s`, lines 3740-3871)

| Resource | Value | ISA source |
|---|---|---|
| VGPR | 252 | `.amdhsa_next_free_vgpr 252`, `.vgpr_count: 252` |
| AGPR | 0 | `.agpr_count: 0`, `.amdhsa_accum_offset 252` |
| SGPR | 50 (`numbered_sgpr` 44 + VCC etc.) | `.sgpr_count: 50` |
| VGPR spill | 0 | `.vgpr_spill_count: 0` |
| SGPR spill | 0 | `.sgpr_spill_count: 0` |
| LDS (group segment) | 68096 B | `.amdhsa_group_segment_fixed_size 68096` |
| Private / scratch | 0 | `.private_segment_fixed_size: 0` |
| waves_per_eu | 2 | `value_attrs={"rocdl.waves_per_eu": 2}` (source line 1968) |
| Max flat workgroup | 512 | `.max_flat_workgroup_size: 512` |

## Table of Contents

- [1. Kernel overview and software pipeline](#1-kernel-overview-and-software-pipeline)
  - [1.1 Buffering: where Q / K / V / S / P / O live (GM / LDS / VGPR)](#11-buffering-where-q--k--v--s--p--o-live-gm--lds--vgpr)
  - [1.2 Online softmax split into sub-stages](#12-online-softmax-split-into-sub-stages)
  - [1.3 Two wave-groups: group A leads by one cluster](#13-two-wave-groups-group-a-leads-by-one-cluster)
  - [1.4 Refined pseudocode](#14-refined-pseudocode)
  - [1.5 Per-operation main instructions and counts](#15-per-operation-main-instructions-and-counts)
  - [1.6 Scheduling primitives: `sched_group_barrier` pairs and `sched_barrier(0)`](#16-scheduling-primitives-sched_group_barrier-pairs-and-sched_barrier0)
  - [1.7 KV-tile distribution and where the causal mask runs](#17-kv-tile-distribution-and-where-the-causal-mask-runs)
  - [1.8 Worked examples: block allocation and per-wave causal-mask sites](#18-worked-examples-block-allocation-and-per-wave-causal-mask-sites)
- [2. Full GM/LDS/VGPR layout maps](#2-full-gmldsvgpr-layout-maps)
  - [2.1 Q: GM -> VGPR](#21-q-gm---vgpr)
  - [2.2 K: GM -> LDS -> VGPR](#22-k-gm---lds---vgpr)
  - [2.3 V: GM -> LDS -> VGPR (HW transpose load)](#23-v-gm---lds---vgpr-hw-transpose-load)
    - [2.3.1 MFMA A-operand convention and CDNA4 transpose load](#231-mfma-a-operand-convention-and-cdna4-transpose-load)
    - [2.3.2 `ds_read_b64_tr_b16` instruction semantics](#232-ds_read_b64_tr_b16-instruction-semantics)
    - [2.3.3 `u_rv` address formula and lane map](#233-u_rv-address-formula-and-lane-map)
  - [2.4 S and P: VGPR-only score / probability layout](#24-s-and-p-vgpr-only-score--probability-layout)
  - [2.5 O: VGPR -> GM](#25-o-vgpr---gm)
  - [2.6 Summary: where each tensor lives](#26-summary-where-each-tensor-lives)
  - [2.7 Low-level call chains and issue counts](#27-low-level-call-chains-and-issue-counts)

---

## 1. Kernel overview and software pipeline

This kernel implements **(Grouped-Query) Flash-Attention forward** with online softmax for bf16/f16
inputs on gfx950. Each workgroup (8 waves = 512 threads) owns one `(batch, query-head, q-block)` of
**BLOCK_M = 256** query rows (32 rows per wave) and streams all KV tiles of **BLOCK_N = 64** keys
each. The body is:

- a **prologue** that primes KV tile 0 (DMA tile 0, load+scale Q, first `mma0` + first softmax pass),
- a **main loop** of **8 clusters C0..C7** that advances **2 KV tiles per iteration** (`for j in
  range(3, max_num_tiles-1, 2)`, source line 1350), and
- a fully-unrolled **14-cluster epilogue (E0..E13)** that drains the trailing tiles the loop leaves
  in flight, followed by an O-normalize + store.

Clusters strictly alternate a **memory stage** (even C0/C2/C4/C6: global->LDS async DMA + LDS->VGPR
reads + causal mask) and a **compute stage** (odd C1/C3/C5/C7: MFMA + softmax). Every cluster ends
with `rocdl.s_barrier()`, and instruction ordering *inside* a cluster is pinned with
`rocdl.sched_barrier(0)` fences and `rocdl.sched_group_barrier` (IGroupLP) MFMA/VALU/EXP group hints
rather than left to the LLVM scheduler.

The two GEMMs use the CDNA **swap-A/B operand convention** (the FlyDSL port feeds the hardware MFMA
operands already swapped):

- **`mma0`** (`_mma0`, source lines 837-845): hardware `src0 = K` fragment (`k_lo`/`k_hi`),
  `src1 = Q` fragment (`q_pack`), contraction over head-dim D. This realizes the score tile
  `S = Q . K^T` (scaled), landing in a register layout where each lane owns **one query row** and the
  64 key scores are split across the lane's accumulators plus its `lane +/- 32` partner.
- **`mma1`** (`_mma1_step_k`, source lines 979-993): hardware `src0 = V` fragment (`v_pk`, transpose-
  loaded), `src1 = P` fragment (`p_pk`). This realizes `O += P . V`, with each lane owning one query
  row and the 128 output channels across 4 accumulator banks.

**Which matrix is the MFMA left (`src0`/A) vs right (`src1`/B).** Each product is one
`v_mfma_f32_32x32x16_bf16 D, A, B, C` (`D = A*B + C`; A is 32x16, B is 16x32). In the *actual
assembly* the hardware **left** operand (`src0`/A) is the *mathematical right* matrix, and the
hardware **right** operand (`src1`/B) is the *mathematical left* matrix (the CDNA swap-A/B convention):

| Matmul | Math form | Math left | Math right | HW `src0` (A, left) | HW `src1` (B, right) | Accumulator |
|---|---|---|---|---|---|---|
| Q*K (`mma0`) | `S = Q . K^T` | Q | K^T | **K** | **Q** | S |
| P*V (`mma1`) | `O += P . V` | P | V | **V** | **P** | O |

ISA evidence (`21_final_isa.s`) and source:
- Q*K: `v_mfma_f32_32x32x16_bf16 v[0:15], v[96:99], v[112:115], 0` (line 738) -> `src0` = **K**
  (`v[96:99]`), `src1` = **Q** (`v[112:115]`, pre-scaled). Source: `_mma0` calls
  `_mfma_acc(k_lo[ks], q_pack, ...)` (A = K, B = Q).
- P*V: `v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[16:19], v[32:47]` (line 996) -> `src0` =
  **V** (`v[192:195]`, transpose-loaded), `src1` = **P** (`v[16:19]`, bf16). Source: `_mma1_step_k`
  calls `_mfma_acc(v_pk[dc], p_pk, ...)` (A = V, B = P).

The swap is deliberate: with `src1 = Q` the CDNA 32x32 MFMA puts the **query row on the lane axis**
and (with `src0 = K`) the **key column on the accumulator registers** -- the "one query row per lane,
keys across registers" layout the online softmax (max/sum over keys) consumes; likewise `src0 = V` /
`src1 = P` puts the query row on lanes and the head-dim on registers for O.

<a id="sec-1-1-buffering-where-q-k-v-s-p-o-live-gm-lds-vgpr"></a>
### 1.1 Buffering: where Q / K / V / S / P / O live (GM / LDS / VGPR)

| Tensor | Global memory | LDS | VGPR (per lane) |
|---|---|---|---|
| Q | full Q tile | **none** (Q never goes through LDS) | `v_q` resident: 64 bf16, pre-scaled by `1/sqrt(D)*log2e`; reused by every `mma0` |
| K | streamed tile by tile (64 rows) | **2 buffers** `K0`,`K1` (8320 bf16 = 16640 B each), 16 B pad/line | `k_lo`/`k_hi` (16 packs of 8 bf16) -> `mma0` |
| V | streamed tile by tile (64 rows) | **2 buffers** `V0`,`V1` (8704 bf16 = 17408 B each), 64 B pad/line | `v_v` (16 packs of 8 bf16) -> `mma1`, read via HW transpose `ds_read_b64_tr_b16` |
| S (scores `Q.K^T`) | -- | none (register-only) | **2 copies**, each `(v_s_lo 16 f32, v_s_hi 16 f32)` = 32 f32. ISA: `v_s_1` (C1 tile) = `v[0:15]` + `v[96:111]`; `v_s_0` (C5 tile) = `v[0:15]` + `v[16:31]` |
| P (probabilities) | -- | none (register-only) | **2 in-flight copies** `v_p_0`,`v_p_1` (one per pipelined tile), each existing in two forms in turn: **fp32** = the 32 `exp2` outputs (1 f32/VGPR = 32 VGPR, transient, scattered in the softmax scratch pool ~`v[144:170]`) feeding `_attn_sum`; then **bf16** = the `_cast_p` result = the `mma1` `src1` operand (32 bf16 packed = 16 VGPR). The `v[16:31]`/`v[96:111]` below are the **bf16** form: `v_p_0` = `v[16:31]`, `v_p_1` = `v[96:111]` |
| O | written at the end | none | `v_o` = 4 f32 accumulator banks x 16 = 64 f32, resident across the whole loop |

K and V each use a **2-deep LDS double buffer**. The unified LDS region (`lds_kv`, 34048 bf16 =
68096 B) is laid out **interleaved `[K0][V0][K1][V1]`** (source lines 222-230):

```
bf16 element offsets within the LDS region:
  K0 : [    0 ..  8320)     V0 : [ 8320 .. 17024)
  K1 : [17024 .. 25344)     V1 : [25344 .. 34048)
  DUALWAVE_SWP_K_BUF_BASE = (0, 17024)        # source line 226
  DUALWAVE_SWP_V_BUF_BASE = (8320, 25344)     # source line 227-230
```

In every memory cluster one buffer is **DMA-filled from GM** while the other is **read into VGPR**
for the next MFMA. The cluster -> (DMA, LDS read, MFMA) mapping for one main-loop iteration:

| Cluster | GM -> LDS (async DMA, prefetch ahead) | LDS -> VGPR | MFMA |
|---|---|---|---|
| C0 (mem) | V tile `j-2` -> `V1` | `K1` -> `k_lo/k_hi` | -- |
| C1 (cmp) | -- | -- | `v_s_1 = mma0(v_q, v_k)` |
| C2 (mem) | K tile `j` -> `K1` | `V0` -> `v_v` (tr) | -- |
| C3 (cmp) | -- | -- | `v_o += step_k<0..3>(v_p_0, v_v)` |
| C4 (mem) | V tile `j-1` -> `V0` | `K0` -> `k_lo/k_hi` | -- |
| C5 (cmp) | -- | -- | `v_s_0 = mma0(v_q, v_k)` |
| C6 (mem) | K tile `j+1` -> `K0` | `V1` -> `v_v` (tr) + causal mask `v_s_0` | -- |
| C7 (cmp) | -- | -- | `v_o += step_k<0..3>(v_p_1, v_v)` |

The two pipeline halves (C0-C3 vs C4-C7) use opposite LDS buffers, so a tile being DMA'd into one
buffer never collides with the tile being consumed from the other.

<a id="sec-vgpr-residency-of-v_s_0-v_s_1-v_p-from-the-isa"></a>
#### VGPR residency of `v_s_0`, `v_s_1`, `v_p` (from the ISA)

The score and probability copies are register-only and ping-pong between two register pools across
the two halves of the loop body (SSA / register allocator reuse; `21_final_isa.s` line refs below).
The `v_s_lo`/`v_s_hi` split is the two 32-wide N-strips of GEMM0 (16 f32 each = `v[a:a+15]`).
Each tile's probability `v_p` exists in **two forms in sequence**: a transient **fp32** form (the 32
`exp2` outputs, one f32 per VGPR, scattered in the softmax scratch pool ~`v[144:170]`) that
`_attn_sum` reduces, and the **bf16** form (the `_cast_p` result, 32 bf16 packed into 16 VGPR) that
is the `mma1` `src1` operand. The table lists both forms of each copy.

| Logical buffer | Produced in | Consumed in | VGPRs | ISA evidence |
|---|---|---|---|---|
| `v_s_1` lo (`v_s_lo`, f32) | C1 `mma0` | C3 row_max/sub/exp | **`v[0:15]`** | `v_mfma v[0:15], v[96:99], v[112:115], 0` (738); `v_sub_f32 v0..v15` (1026-1043) |
| `v_s_1` hi (`v_s_hi`, f32) | C1 `mma0` | C3 row_max/sub/exp | **`v[96:111]`** | `v_mfma v[96:111], v[170:173], v[112:115], 0` (742); `v_sub_f32 v96..v111` (1044-1062) |
| `v_s_0` lo (`v_s_lo`, f32) | C5 `mma0` | C7 row_max/sub/exp | **`v[0:15]`** | `v_mfma v[0:15], ..., v[0:15]` (1183, 1215) |
| `v_s_0` hi (`v_s_hi`, f32) | C5 `mma0` | C7 row_max/sub/exp | **`v[16:31]`** | `v_mfma v[16:31], v[228:231], v[140:143], v[16:31]` (1199) |
| `v_p_0` **fp32** (32 f32 exp2 out, loop-carried) | prev-C7 `exp2[0:16]` + C1 `exp2[16:32]` | C1 `_attn_sum` then `_cast_p` | **~`v[144:170]`** + `v[20:23]`,`v31` | `_attn_sum` inputs (761-797); `_cast_p` inputs (804-852) |
| `v_p_0` **bf16** (4 packs, mma1 src1) | C1 `_cast_p` | C3 `mma1` (4 step_k) | **`v[16:31]`** | cast outputs `v16..v31` (804-852); `mma1` `src1` = `v[16:19]`,`v[20:23]`,`v[24:27]`,`v[28:31]` (996/1025/1053/1071) |
| `v_p_1` **fp32** (32 f32 exp2 out) | C3 `exp2[0:16]` + C5 `exp2[16:32]` | C5 `_attn_sum` then `_cast_p` | **~`v[144:170]`** + `v[100:103]`,`v111` | `_cast_p` inputs (1181-1223) |
| `v_p_1` **bf16** (4 packs, mma1 src1) | C5 `_cast_p` | C7 `mma1` (4 step_k) | **`v[96:111]`** | cast outputs `v98..v111` (1181-1223) |

Key consequences of the reuse:

- **P has 2 copies, and each is fp32 then bf16.** The `v[16:31]`/`v[96:111]` ranges hold the **bf16**
  P (the packed `mma1` operand, 16 VGPR each). The **fp32** P (32 f32 `exp2` output) is short-lived —
  produced by `_attn_exp2_slice`, reduced by `_attn_sum`, and packed by `_cast_p` all within one
  cluster — so the allocator parks it in the shared softmax scratch pool (~`v[144:170]`, the same
  pool that holds `v_k`/`v_v` reads) rather than a fixed contiguous range. Only the fp32 `v_p_0` (the
  first-half-exp'd carry, `_v_pair_to_vec32`) crosses the loop back-edge; no bf16 copy does.
- The `v_s_lo` pool **`v[0:15]` is shared by both score copies** — `v_s_1` is fully consumed (row-max,
  sub-row, first-half `exp2` -> `v_p_1`) in C3 *before* C5's `mma0` overwrites `v[0:15]` with `v_s_0`,
  so the two are never simultaneously live in their lo halves.
- The pools **`v[16:31]` and `v[96:111]` each alternate** between holding a score-`hi` half and a
  **bf16** probability copy: `v[16:31]` is bf16 `v_p_0` (C1-C3) then `v_s_0`'s hi (C5-C7); `v[96:111]`
  is `v_s_1`'s hi (C1-C3) then bf16 `v_p_1` (C5-C7). The earlier consumer dies before the later
  producer writes.
- Surrounding pools for context: `v_q` (8 scaled Q packs) = **`v[112:143]`** (`mma0` `src1`, lines
  738-822); `v_o` accumulators = **`v[32:47]`, `v[48:63]`, `v[64:79]`, `v[80:95]`** (`mma1` dst/`src2`,
  lines 996-1083); transpose-loaded `v_v` and LDS-read `v_k` occupy the `v[144:243]` pool.

### 1.2 Online softmax split into sub-stages

To hide the transcendental `exp2` latency behind MFMA, a tile's softmax is split and spread over
**two consecutive compute clusters** plus the P*V cluster. The FlyDSL helpers are
`_attn_row_max` (969-977), `_attn_sub_row` (995-1005), `_attn_exp2_slice` (1007-1019),
`_attn_sum` (1021-1029), `_cast_p` (1031-1041), `_lazy_rescale_o` (1122-1171):

1. `_attn_row_max(v_s)` (per-lane max over 32 f32, then cross-`lane+/-32` via
   `v_permlane32_swap_b32`) + **lazy-rescale** of `v_o`/`l_row` + `_attn_sub_row` (`S -= m_row`) +
   `_attn_exp2_slice(v_s, 0, 16)` (**first half**, elems 0..15).
2. `_attn_exp2_slice(v_p, 16, 16)` (**second half**, elems 16..31) + `_attn_sum` (row denominator into
   `l_row`) + `_cast_p` (`v_p = cast<bf16>(P)`).
3. `_mma1_step_k(...)` consumes the bf16 `v_p` (P*V).

Distribution across the pipeline:

- **Prologue**: stage-1 partial for tile 0 -- `row_max`, `sub_row`, `exp2` first half (source 1246-1248).
- **Main loop**: tile A (scores from C1) -> stage 1 in **C3**, stage 2 in **C5**, P*V in **C7**; tile B
  (scores from C5) -> stage 1 in **C7**, stage 2 in **C1 of next iter**, P*V in **C3 of next iter**.
  So every compute cluster does P*V for an older tile while doing softmax stage-1 for a newer tile.
  Causal masking, when a tile straddles the diagonal, is applied in the memory cluster before the
  row-max (prologue / C6 / E2 / E6 / E10), guarded by a runtime `if q_start < kv_end` test
  (`_causal_mask_prologue_if_needed`, source 943-967).
- **Epilogue**: same chain but rescale is **unconditional** (E3/E7/E11 use `_fmax`+`exp2`+`_scale_o`
  directly, source 1689-1700 / 1759-1770 / 1828-1846); E11 folds both `exp2` halves of the last tile.

The 1/sqrt(D) temperature is **pre-applied to Q** (`_scale_q_all`, source 523-537, folding in `log2e`
so `exp2` replaces `exp`), not folded into the per-tile exp.

### 1.3 Two wave-groups: group A leads by one cluster

The 8 waves split into **group A (waves 0-3)** and **group B (waves 4-7)** via
`_stagger = wave_id / 4` (source line 334). In the prologue group B runs **one extra `s_barrier`**
(`_stagger_extra_barrier_if_one`, source 662-677). Because `s_barrier` matches by ordinal and each
cluster has exactly one `s_barrier`, this makes **group A run one full cluster ahead of group B** for
the whole main loop and epilogue:

```
group A:  [ P0+QK0 ]--b--[   C0   ]--b--[   C1   ]--b--[   C2   ]
group B:  [   P0   ]--b--[  QK0   ]--b--[   C0   ]--b--[   C1   ]
  P0  = prologue load (prefetch K[1] + V[0] + read K[0] -> v_k)
  QK0 = prologue mma0 + first softmax pass (KV tile 0)
  C0..= main-loop clusters; 'b' = matched s_barrier
```

Because of the one-cluster offset, when **group A is in a compute cluster (MFMA)** **group B is in
the adjacent memory cluster (DMA + LDS reads + waitcnt stalls)** -- one group computes while the
other loads, swapping at every `s_barrier`, so each group's memory latency hides behind the other's
MFMA. `rocdl.s_setprio(1)`/`s_setprio(0)` brackets the heaviest compute clusters (C3/C7) to hand the
shared MFMA issue port from the computing group to the group entering its compute phase. The
epilogue's closing barrier gives **group A** the matching extra `s_barrier`
(`_stagger_extra_barrier_if_zero`, source 679-693) to re-sync both groups before the store.

In the ISA the close appears verbatim as inline asm (lines 3563-3568):

```asm
;;#ASMSTART
s_cmp_eq_u32 s25, 0      ; s25 = stagger (wave_id/4)
s_cbranch_scc0 1f        ; group B (stagger != 0): skip
s_barrier                ; group A (stagger == 0): +1 barrier to re-sync
1:
;;#ASMEND
```

### 1.4 Refined pseudocode

Variable names match the FlyDSL source (`flash_attn_gfx950.py`). `NUM_DMA_K = NUM_DMA_V = 2`.

```
Prologue (source 1180-1268):
  [P1] _async_load_k(tile 0 -> K0)
       s_waitcnt 0
       sched_barrier(0)
       s_barrier
  [P2] q_all = _load_q_all(q_row_in_block)                 # 8 buffer_load_dwordx4
       q_all_scaled = _scale_q_all(q_all)                  # *1/sqrt(D)*log2e, cast bf16
  [P3] _async_load_k(tile 1 -> K1)
       _async_load_v(tile 0 -> V0)
       v_k = _async_load_k_from_lds_to_vgpr(K0)            # 16 ds_read_b128
       sched_barrier(0)
       s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_V)
  [P4] if stagger: group B runs +1 s_barrier               # OPEN phase shift
  [P5] v_s_0 = _mma0(v_k)                                   # 16 MFMA
       if CAUSAL: v_s_0 = _causal_mask_prologue_if_needed(v_s_0)
       m_row = _attn_row_max(v_s_0)
       v_s_0 = _attn_sub_row(v_s_0, m_row)
       v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)              # first half
       sched_barrier(0)
       s_barrier
       sched_barrier(0)
  [P6] _async_load_k(tile 2 -> K0)
       init { m_row, l_row=0, v_o=0(x4 banks), v_p_0 }

Main loop  for j in range(3, max_num_tiles-1, 2):  # 8 clusters, +2 KV tiles / iter
  [C0 mem] _async_load_v(tile j-2 -> V1)
           v_k = read(K1)                                  # 16 ds_read_b128
           s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K+NUM_DMA_V)
           s_barrier
  [C1 cmp] v_s_1 = _mma0(v_k)                              # 16 MFMA
           v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)         # finish prev tile 2nd half
           l_row += _attn_sum(v_p_0)
           v_p_0 = _cast_p(v_p_0)
           _anchor_v_p(v_p_0)
           sched_barrier_exp_pairs(6,3,1)
           sched_barrier_pairs(10,5,1)
           s_barrier
  [C2 mem] _async_load_k(tile j -> K1)
           v_v = _read_v_packs_for_buf(V0)                 # 32 ds_read_b64_tr_b16
           s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K+NUM_DMA_V)
           s_barrier
  [C3 cmp] s_setprio(1)
           v_o = _mma1_step_k(0, v_p_0, v_v, v_o)          # 4 MFMA
           m_tile_max_a = _attn_row_max(v_s_1)
           sched_barrier_pairs(4,6,2)
           LAZY-RESCALE (ballot(m_diff<=8.0)==exec):
             skip if all rows within 8.0, else:
               corr = exp2(m_row - combined)
               v_o   *= corr
               v_p_0 *= corr
               l_row *= corr
               m_row <- combined
           v_o = _mma1_step_k(1..3, v_p_0, v_v, v_o)       # 12 MFMA
           v_s_1 = _attn_sub_row(v_s_1, m_row)
           v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)          # first half
           sched_barrier_pairs(6,6,2)
           sched_barrier_exp_pairs(6,3,2)
           s_setprio(0)
           s_barrier
  [C4 mem] _async_load_v(tile j-1 -> V0)
           v_k = read(K0)
           s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K+NUM_DMA_V)
           s_barrier
  [C5 cmp] v_s_0 = _mma0(v_k)                              # 16 MFMA
           v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)
           l_row += _attn_sum(v_p_1)
           v_p_1 = _cast_p(v_p_1)
           _anchor_v_p(v_p_1)
           sched_barrier_exp_pairs(6,3,3)
           sched_barrier_pairs(10,5,3)
           s_barrier
  [C6 mem] _async_load_k(tile j+1 -> K0)
           v_packs_b = _read_v_packs_for_buf(V1)           # 32 ds_read_b64_tr_b16
           if CAUSAL: v_s_0 = _causal_mask_prologue_if_needed(v_s_0, j-1, j*BLOCK_N)
           s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K+NUM_DMA_V)
           s_barrier
  [C7 cmp] s_setprio(1)
           v_o = _mma1_step_k(0, v_p_1, v_packs_b, v_o)     # 4 MFMA
           m_tile_max_b = _attn_row_max(v_s_0)
           sched_barrier_pairs(4,6,4)
           LAZY-RESCALE
           v_o = _mma1_step_k(1..3, v_p_1, v_packs_b, v_o)  # 12 MFMA
           v_s_0 = _attn_sub_row(v_s_0, m_row)
           v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
           sched_barrier_pairs(6,5,4)
           sched_barrier_exp_pairs(6,3,4)
           s_setprio(0)
           s_barrier
           yield { m_row, l_row, v_o, v_p_0 }

Epilogue (E0..E13, source 1636-1862; rescale is UNCONDITIONAL):
  [E0  mem] _async_load_v(max_m3 -> V1)
            v_k = read(K1)                                 # 16 ds_read_b128
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K+NUM_DMA_V)
            s_barrier
  [E1  cmp] v_s_1 = _mma0(v_k)                             # 16 MFMA
            v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)        # finish carried tile 2nd half
            l_row += _attn_sum(v_p_0)
            v_p_0 = _cast_p(v_p_0)
            _anchor_v_p(v_p_0)
            sched_barrier_exp_pairs(6,3,5)
            sched_barrier_pairs(10,5,5)
            s_barrier
  [E2  mem] _async_load_k(max_m1 -> K1)
            v_packs_e3 = _read_v_packs_for_buf(V0)         # 32 ds_read_b64_tr_b16
            if CAUSAL: v_s_1 = _causal_mask_prologue_if_needed(v_s_1, max_m3, max_m2*BLOCK_N)
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K+NUM_DMA_V)
            s_barrier
  [E3  cmp] s_setprio(1)
            v_o = _mma1(v_p_0, v_packs_e3, v_o)            # full 16 MFMA
            m_tile_max_e3 = _attn_row_max(v_s_1)
            row_max_e3 = max(m_row, m_tile_max_e3)
            rescale_e3 = exp2(m_row - row_max_e3)
            m_row <- row_max_e3
            v_s_1 = _attn_sub_row(v_s_1, row_max_e3)
            v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)
            sched_barrier_pairs(10,5,6)
            sched_barrier_exp_pairs(6,3,6)
            _scale_o(v_o, rescale_e3)
            _anchor_v_o(v_o)
            s_setprio(0)
            s_barrier
  [E4  mem] _async_load_v(max_m2 -> V0)
            v_k = read(K0)
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_K+NUM_DMA_V)
            s_barrier
  [E5  cmp] v_s_0 = _mma0(v_k)                             # 16 MFMA
            l_row *= rescale_e3
            v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)
            l_row += _attn_sum(v_p_1)
            v_p_1 = _cast_p(v_p_1)
            _anchor_v_p(v_p_1)
            sched_barrier_exp_pairs(6,3,7)
            sched_barrier_pairs(10,5,7)
            s_barrier
  [E6  mem] v_packs_e7 = _read_v_packs_for_buf(V1)         # 32 ds_read_b64_tr_b16
            if CAUSAL: v_s_0 = _causal_mask_prologue_if_needed(v_s_0, max_m2, max_m1*BLOCK_N)
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_V)
            s_barrier
  [E7  cmp] s_setprio(1)
            v_o = _mma1(v_p_1, v_packs_e7, v_o)            # full 16 MFMA
            m_tile_max_e7 = _attn_row_max(v_s_0)
            row_max_e7 = max(m_row, m_tile_max_e7)
            rescale_e7 = exp2(m_row - row_max_e7)
            m_row <- row_max_e7
            v_s_0 = _attn_sub_row(v_s_0, row_max_e7)
            v_p_0 = _attn_exp2_slice(v_s_0, 0, 16)
            sched_barrier_pairs(10,5,8)
            sched_barrier_exp_pairs(6,3,8)
            _scale_o(v_o, rescale_e7)
            _anchor_v_o(v_o)
            s_setprio(0)
            s_barrier
  [E8  mem] _async_load_v(max_m1 -> V1)
            v_k = read(K1)
            s_waitcnt lgkmcnt(0) vmcnt(NUM_DMA_V)
            s_barrier
  [E9  cmp] v_s_1 = _mma0(v_k)                             # 16 MFMA (last tile)
            l_row *= rescale_e7
            v_p_0 = _attn_exp2_slice(v_p_0, 16, 16)
            l_row += _attn_sum(v_p_0)
            v_p_0 = _cast_p(v_p_0)
            _anchor_v_p(v_p_0)
            sched_barrier_exp_pairs(6,3,9)
            sched_barrier_pairs(10,5,9)
            s_barrier
  [E10 mem] v_packs_e11 = _read_v_packs_for_buf(V0)        # 32 ds_read_b64_tr_b16
            if CAUSAL: v_s_1 = _causal_mask_prologue_if_needed(v_s_1, max_m1, max_num_tiles*BLOCK_N)
            s_waitcnt lgkmcnt(0) vmcnt(0)                  # all DMA drained
            s_barrier
  [E11 cmp] v_o = _mma1(v_p_0, v_packs_e11, v_o)           # full 16 MFMA
            m_tile_max_e11 = _attn_row_max(v_s_1)
            row_max_e11 = max(m_row, m_tile_max_e11)
            rescale_e11 = exp2(m_row - row_max_e11)
            m_row <- row_max_e11
            v_s_1 = _attn_sub_row(v_s_1, row_max_e11)
            v_p_1 = _attn_exp2_slice(v_s_1, 0, 16)         # first half
            sched_barrier_pairs(9,6,10)
            sched_barrier_exp_pairs(7,3,10)
            v_p_1 = _attn_exp2_slice(v_p_1, 16, 16)        # second half (folded in)
            l_row *= rescale_e11
            l_row += _attn_sum(v_p_1)
            v_p_1 = _cast_p(v_p_1)
            _anchor_v_p(v_p_1)
            _scale_o(v_o, rescale_e11)
            _anchor_v_o(v_o)
            s_barrier
  [E12 mem] v_packs_e13 = _read_v_packs_for_buf(V1)        # 32 ds_read_b64_tr_b16
            s_waitcnt lgkmcnt(0)
            s_barrier
  [E13 cmp] v_o = _mma1(v_p_1, v_packs_e13, v_o)           # final P*V (16 MFMA)

Finalize (source 1864-1923):
  inv_l = (l_row > 0) ? rcp(l_row) : 0
  _scale_o(v_o, inv_l)                                     # 32 v_pk_mul
  if stagger: group A runs +1 s_barrier                    # CLOSE phase shift
  if q_row < seq_len:
    store O = cast<bf16/f16>(v_o)                          # 16 buffer_store_dwordx2
```

### 1.5 Per-operation main instructions and counts

Counts are **per warp** unless noted (each VALU/memory instruction is executed by all 64 lanes in
lock-step; each MFMA is a warp-collective instruction). The fixed sizes come from the source
constants (lines 174-247):

```
Launch / tile:
  BLOCK_M = 256   BLOCK_N = 64   HEAD_DIM = 128   NUM_WAVES = 8   WARP_SIZE = 64   BLOCK_SIZE = 512
  ROWS_PER_WAVE = 32

MFMA tile:
  W_M = 32   W_N = 32 (K_SUB_N)   W_K = 16 (K_STEP_QK)   MFMA_LANE_K = 8

GEMM expansion:
  GEMM0 (mma0, S = Q.K^T): E_M = 32/32 = 1, E_N = BLOCK_N/W_N = 64/32 = 2, E_K = HEAD_DIM/W_K = 128/16 = 8
                           MFMA = E_M*E_N*E_K = 1*2*8 = 16
  GEMM1 (mma1, O += P.V) : E_M = 1, E_N = HEAD_DIM/W_N = 128/32 = 4 (D_CHUNKS), E_K = BLOCK_N/W_K = 64/16 = 4
                           MFMA = 1*4*4 = 16 ; step_k = 4 steps x D_CHUNKS(4) MFMA each

Data widths / LDS geometry:
  VEC_KV = 8 bf16 (16 B)   D_128B_SIZE = 64 bf16   SMEM_LINEAR_WAVE = 64*16/2 = 512 bf16
  SMEM_N_RPT = 8   SMEM_D_RPT = 2   NUM_DMA_K = NUM_DMA_V = SMEM_D_RPT = 2
  SMEM_K_LINE_STRIDE = 512+8 = 520   SMEM_V_LINE_STRIDE = 512+32 = 544
  SMEM_K_TILE_ELEMS = 8*2*520 = 8320   SMEM_V_TILE_ELEMS = 8*2*544 = 8704
```

| # | Operation (source) | Main ISA instruction(s) | Count / call | Tiling derivation |
|---|---|---|---|---|
| 1 | `_async_load_k` / `_async_load_v` | `buffer_load_dwordx4 ... lds` | **2** | `NUM_DMA = SMEM_D_RPT = 2` (block-cooperative GM->LDS DMA) |
| 2 | `_load_q_all` (504-521) | `buffer_load_dwordx4` | **8** | per lane 64 bf16 = 128 B; /(VEC=16 B) = 8 (= `K_STEPS_QK`) |
| 3 | `_scale_q_all` (523-537) | `v_and`+`v_lshlrev` (bf16->f32) + `v_pk_mul`/`v_mul` + `v_cvt_pk_bf16_f32` | unpack 32+32, scale ~32, repack 32 | 64 f32/lane packed 2/instr |
| 4 | `_mma0(v_k)` (837-845) | `v_mfma_f32_32x32x16_bf16` | **16** | GEMM0 = `E_M*E_N*E_K = 1*2*8` |
| 5 | `_causal_mask_inplace` (847-925) | `v_cmp_lt_i32_e64` + `v_cndmask_b32_e64` (inline asm) | **32 + 32** | 8 thr pairs x (lo+hi) = 16 `_attn_mask_vec2_imm` x (2 cmp + 2 cndmask) |
| 6 | `_attn_row_max` (969-977) | `v_max_f32`+`v_max3_f32` + `v_permlane32_swap_b32` + `v_max_f32` | **~16 + 1 + 1** | reduce 32 f32 (1 `v_max` + ~15 `v_max3`, verified 30 `v_max3`/2-tile loop body) then cross `lane+/-32` |
| 7 | `_attn_sub_row` (995-1005) | `v_sub_f32` | **32** | 32 score elems / lane |
| 8 | `_attn_exp2_slice` (1007-1019) | `v_exp_f32` | **16** | half = 16 elems (the other half in another cluster) |
| 9 | `_async_load_k_from_lds_to_vgpr` (785-809) | `ds_read_b128` | **16** | `K_STEPS_QK(8) * 2 (lo,hi)`; = 128 bf16/lane / 8 |
| 10 | `_attn_sum` (1021-1029) | `v_add_f32` + `v_permlane32_swap_b32` + `v_add_f32` | **~31 + 1 + 1** | serial reduce 32 f32 + cross `lane+/-32` |
| 11 | `_cast_p` (1031-1041) | `v_cvt_pk_bf16_f32` | **16** | 32 f32/lane -> packed 2/instr |
| 12 | `_read_v_packs_for_buf` (811-835) | `ds_read_b64_tr_b16` (HW transpose) | **32** | `D_CHUNKS(4) * 4 k_substep * 2 (a,b)`; = 128 bf16/lane / 4 |
| 13 | `_mma1_step_k(step)` (979-988) | `v_mfma_f32_32x32x16_bf16` | **4** | one K-step = `E_M*E_N = 1*4`; full `_mma1` = 4 steps x 4 = 16 |
| 14 | `_scale_o(v_o, s)` (1043-1046) | `v_pk_mul_f32` | **32** | 64 f32/lane packed 2/instr (cold-path rescale + final normalize) |
| 15 | `_scale_v_p` (647-660) | `v_pk_mul_f32` (+unpack/repack) | **16** | 32 f32/lane packed 2/instr (rescale cold path only) |
| 16 | `rcp(l_row)` (1868) | `v_rcp_f32` + `v_cmp_lt_f32` + `v_cndmask_b32_e32` | **1 + 1 + 1** | one f32 scalar/lane; guard `l_row > 0` |
| 17 | O store (1892-1923) | `v_cvt_pk_bf16_f32` + `buffer_store_dwordx2` | **32 + 16** | 64 f32/lane -> 32 cvt (=64 bf16); store 2 dwords/instr -> 16 |

#### Whole-kernel ISA verification

Counts grepped from `21_final_isa.s` with `;` comments stripped (the only `;` lines are
`;;#ASMSTART/ASMEND` inline-asm markers, which contain no mnemonics, so raw == real). Region split:
prologue = lines 8-712, main-loop body = 713-1582 (one rolled `scf.for` iteration = **2 KV tiles**),
epilogue+store = 1583-3878.

| Instruction | Total | Prologue | Loop body (2 tiles) | Epilogue+store | Derivation check |
|---|---:|---:|---:|---:|---|
| `v_mfma_f32_32x32x16_bf16` | **192** | 16 | 64 | 112 | 16 (pro mma0) + 2*(16+16) loop + (3 mma0 + 4 mma1)*16 epi |
| `ds_read_b64_tr_b16` | **192** | 0 | 64 | 128 | 32/call x (C2,C6,E2,E6,E10,E12) = 6 calls |
| `ds_read_b128` | **96** | 16 | 32 | 48 | 16/call x (P3,C0,C4,E0,E4,E8) = 6 calls |
| `buffer_load_dwordx4` | **32** | 16 | 8 | 8 | Q 8 + DMA 2/call x 12 calls = 24 |
| `buffer_store_dwordx2` | **16** | 0 | 0 | 16 | D_CHUNKS(4) x 4 store_groups |
| `v_cmp_lt_i32_e64` | **160** | 32 | 32 | 96 | 32/call x 5 causal-mask call sites (pro,C6,E2,E6,E10) |
| `v_cndmask_b32_e64` | **160** | 32 | 32 | 96 | same 5 mask call sites |
| `v_exp_f32` | **197** | ~17 | 64 | ~116 | 32/tile exp2 + 1/tile rescale corr |
| `v_cvt_pk_bf16_f32` | **184** | -- | 32 | -- | 16/tile cast P + 32 final store cvt |
| `v_max3_f32` | **93** | -- | 30 | -- | ~15/tile row-max reduction |
| `v_permlane32_swap_b32` | **12** | 1 | 4 | 7 | 2/tile (row_max + sum) x 6 tiles |
| `s_barrier` | **25** | 4 | 8 | 13 | 1/cluster + 2 stagger barriers |
| `s_setprio` | **8** | 0 | 4 | 4 | 2/compute-heavy cluster x (C3,C7,E3,E7) |
| `s_waitcnt` | **15** | 3 | 4 | 8 | 1/memory cluster (lgkmcnt+vmcnt merged) |

The four exactly-determined memory/MFMA classes (`v_mfma`=192, `ds_read_b64_tr_b16`=192,
`ds_read_b128`=96, `buffer_load_dwordx4`=32, `buffer_store_dwordx2`=16, causal
`v_cmp`/`v_cndmask`=160) match the tiling derivation to the instruction. The VALU classes vary by
region because rescale is conditional (lazy in the loop, unconditional in the epilogue) and the
prologue/epilogue carry partial softmax stages.

<a id="sec-1-6-scheduling-primitives-sched_group_barrier-pairs-and-sched_barrier0"></a>
### 1.6 Scheduling primitives: `sched_group_barrier` pairs and `sched_barrier(0)`

The `sched_*` calls -- `_sched_barrier_pairs` / `_sched_barrier_exp_pairs` (source 456-474) and the
`rocdl.sched_barrier(0)` at every cluster boundary -- are **compile-time scheduling directives only**.
They lower to LLVM intrinsics that the AMDGPU IGroupLP (Instruction Group Level Parallelism) pass
consumes and removes; they emit **no real ISA instruction** and cost zero cycles. Their sole purpose
is to constrain how LLVM's pre-/post-RA scheduler orders the MFMA / VALU / exp instructions inside a
cluster so the (real) hardware overlap of MFMA with VALU/exp/LDS is actually realized.

Masks (source 289-291):

```python
_MFMA_MASK = 0x008   # MFMA / WMMA instruction class
_VALU_MASK = 0x002   # VALU (v_add / v_mul / v_sub / v_cvt / ...)
_EXP_MASK  = 0x400   # TRANS (transcendental: v_exp_f32, ...)
```

| Intrinsic | Kind | Effect |
|---|---|---|
| `rocdl.sched_group_barrier(mask, size, group)` | soft grouping hint | gathers `size` instructions of class `mask` into an ordered group tagged `group`; groups sharing `group` are solved as one interleaved pipeline |
| `rocdl.sched_barrier(0)` | hard fence | with `mask = 0` ("allow none"), no class (MFMA/VALU/SALU/TRANS/VMEM/DS) may be scheduled across this point |

<a id="sec-_sched_barrier_pairspairs-valu_cnt-group-source-456-464"></a>
#### `_sched_barrier_pairs(pairs, valu_cnt, group)` (source 456-464)

```python
for _ in range_constexpr(pairs):
    rocdl.sched_group_barrier(_MFMA_MASK, 1, group)        # 1 MFMA
    rocdl.sched_group_barrier(_VALU_MASK, valu_cnt, group) # valu_cnt VALU
```

So `_sched_barrier_pairs(10, 5, 1)` emits **10 x (1 MFMA-group + 1 VALU-group of 5)**, all tagged
`group = 1`: "lay out this region as `[1 MFMA][up to 5 VALU]` repeated 10 times" -- spread the VALU
work (the `_attn_sum` adds and the bf16 `_cast_p`) evenly between the MFMA instructions instead of
bunching them.

<a id="sec-_sched_barrier_exp_pairspairs-exp_cnt-group-source-466-474"></a>
#### `_sched_barrier_exp_pairs(pairs, exp_cnt, group)` (source 466-474)

Identical shape, but the second group is the **TRANS/EXP** class (`0x400`):

```python
for _ in range_constexpr(pairs):
    rocdl.sched_group_barrier(_MFMA_MASK, 1, group)        # 1 MFMA
    rocdl.sched_group_barrier(_EXP_MASK, exp_cnt, group)   # exp_cnt exp
```

`_sched_barrier_exp_pairs(6, 3, 1)` keeps the softmax `v_exp_f32` interleaved with the MFMA chain so
the long-latency transcendental hides behind MFMA, instead of LLVM emitting all `v_exp` back-to-back.

#### How one compute cluster uses both

C1's compute is `mma0` (16 MFMA) running concurrently with the previous tile's second-half `exp2`
(16 `v_exp`), the `_attn_sum` reduction (~31 `v_add`), and `_cast_p` (16 `v_cvt_pk`). The two hints
**partition the 16 MFMA** and bind both halves into one `group = 1` pipeline:

```
_sched_barrier_exp_pairs(6,3,1)  ->  [1 MFMA][3 exp ] x 6    (6 MFMA paired with exp work)
_sched_barrier_pairs   (10,5,1)  ->  [1 MFMA][5 VALU] x 10   (10 MFMA paired with sum/cast VALU)
                                      6 + 10 = 16 MFMA total  (== mma0's MFMA count)
```

Directionality is **bottom-up**: `IGroupLPDAGMutation` scans from each marker backward to the start
of the scheduling region, so a hint describes the MFMA / exp / VALU **already emitted above it**.
Each cluster reuses the helpers with its own `group` id (C1 = 1, C3 = 2, C5 = 3, C7 = 4, epilogue =
5..10), so each cluster's pipeline is solved independently. The source carries an extended commentary
on this at lines 1422-1527.

This interleaving is directly visible in the C3 ISA (lines 996-1085): each `v_mfma` is followed by
~6 `v_max3_f32` / `v_sub_f32` / `v_exp_f32`:

```asm
v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[16:19], v[32:47]   ; mma1 step_k<0>, bank 0  (A=V, B=P)
v_max_f32_e32  v192, v0, v1                                         ; row-max reduction begins
v_max3_f32     v192, v192, v2, v3
...
v_mfma_f32_32x32x16_bf16 v[48:63], v[196:199], v[16:19], v[48:63]   ; mma1 step_k<0>, bank 1
...
```

<a id="sec-sched_barrier0-the-hard-fence"></a>
#### `sched_barrier(0)` (the hard fence)

`rocdl.sched_barrier(0)` lowers to the AMDGPU `SCHED_BARRIER` pseudo. `SIInstrInfo::isSchedulingBoundary()`
returns true for it with immediate 0, so the machine scheduler **splits the scheduling region** there
and cannot reorder anything across it. It is placed around every real `s_barrier`
(`sched_barrier(0); s_barrier(); sched_barrier(0)`), so (a) one cluster's MFMA/exp/loads cannot leak
into an adjacent cluster, and (b) the hardware `s_barrier` stays pinned at the phase boundary. Like
the group-barrier hints, it emits **no real instruction** -- the actual cross-wave synchronization is
the separate `s_barrier` next to it.

---

### 1.7 KV-tile distribution and where the causal mask runs

This section answers a recurring question: for a given wave / q-block, **how many KV tiles does each
pipeline stage process**, and **where (prologue / main loop / epilogue) is the causal mask actually
applied**. All line numbers below refer to `FlyDSL/kernels/flash_attn_gfx950.py`.

#### 1.7.1 How many KV tiles each stage owns

The tile count `max_num_tiles` that the pipeline iterates over is **always rounded up to an even
number** and floored at 4 (source 1147, 1151):

```1144:1151:FlyDSL/kernels/flash_attn_gfx950.py
            # Pipeline (prologue + 2-tile loop + 3-tile drain) needs an EVEN tile count,
            # so round ceil(seq_len/64) up to even. The extra tile is out of range ->
            # reads 0 (num_records) and is masked, contributing nothing; aligned: no-op.
            max_num_tiles = ((max_num_tiles + fx.Index(1)) // fx.Index(2)) * fx.Index(2)
            # Pipeline needs >= 4 tiles; for tiny seq_len (< ~192) floor the count at 4.
            # The extra tiles are out of range -> read 0 (num_records) and are masked,
            # contributing nothing; seq_len already yielding >= 4 tiles is unaffected.
            max_num_tiles = fx.Index(ArithValue(max_num_tiles < fx.Index(4)).select(fx.Index(4), max_num_tiles))
```

The pipeline is a fixed **prologue (1 tile) + main loop (2 tiles / iteration) + epilogue (3 tiles)**
split:

| Stage | Tiles processed | Tile index(es) | Source |
|---|---|---|---|
| Prologue | **1** | tile `split_t0` (= 0 non-split-K) | `_mma0` at 1221 |
| Main loop, each iteration | **2** | `j-2` (`v_s_1`, cluster1) and `j-1` (`v_s_0`, cluster5) | 1287, 1363 |
| Epilogue | **3** | `max_m3 / max_m2 / max_m1` = `split_t_end-3 / -2 / -1` | 1452, 1514, 1575 |

So the total tile count is `1 + 2*(#iterations) + 3`. The main loop is

```1263:1268:FlyDSL/kernels/flash_attn_gfx950.py
                for j, loop_args in range(
                    loop_lb,
                    split_t_end - fx.Index(1),
                    fx.Index(2),
                    init=init_args,
                ):
```

with `loop_lb = 3` (non-split-K, source 1261) or `split_t0 + 3` (split-K, source 1259). The loop body
runs only when `split_t_end - 1 > loop_lb`, i.e. **only when `max_num_tiles >= 6`**.

#### 1.7.2 Worked tile-count cases

- **`max_num_tiles = 4`** (e.g. a q-block diagonal of 256 keys): `range(3, 3, 2)` is empty, so the
  **main loop does not execute at all**. All 4 tiles are consumed by prologue (tile 0) + epilogue
  (tiles 1/2/3). This holds for split-K too: `chunk >= 6` but `split_t_end` is clamped back to
  `max_num_tiles = 4` (1162-1165) and `loop_lb = split_t0 + 3`, giving the same empty `range(3,3,2)`.
- **`max_num_tiles = 5`**: never used as-is -- rounded up to **6** by line 1147. The pipeline then
  runs `range(3, 5, 2) = [3]` -> **1 main-loop iteration** (tiles 1 and 2), with prologue=tile 0 and
  epilogue=tiles 3/4/5. The 6th tile (index 5) is **padding**: it is out of range, so the buffer load
  returns 0 via `num_records` and the value is masked to `-inf`, contributing nothing to softmax /
  output. Rounding to even is purely a pipeline-structure requirement, not a numerical change.

<a id="sec-1-7-3-which-tile-a-wave-first-masks-the-diagonal-tile"></a>
#### 1.7.3 Which tile a wave first masks (the diagonal tile)

A q-block's 256 rows are split across 8 waves (32 rows each, `ROWS_PER_WAVE = 32`, source 151). The
causal mask gate is per-wave (source 902): `if q_start_pos_i32 + delta_i32 < kv_end_pos`, with
`q_start_pos = q_start + wave_id * ROWS_PER_WAVE` (source 1199). For self-attention
(`seqlen_q == seqlen_kv` -> `delta = 0`), `BLOCK_M = 256`, `BLOCK_N = 64`, and q-block index `b`
(`q_start = 256*b`), each wave's diagonal tile and the number of tiles it must mask are:

| wave | diagonal tile (first masked) | tiles to mask | tile parity (slot) |
|---|---|---|---|
| 0, 1 | `4b + 0` | 4 | even (`v_s_0`) |
| 2, 3 | `4b + 1` | 3 | odd (`v_s_1`) |
| 4, 5 | `4b + 2` | 2 | even (`v_s_0`) |
| 6, 7 | `4b + 3` | 1 | odd (`v_s_1`) |

Because `max_num_tiles` is fixed by the q-block's **last row** (`q_start + BLOCK_M - 1`), the **last
tile is always a diagonal tile and always lands in the epilogue**. But the earlier waves' diagonal
tiles sit further back, so the mask does **not** only happen in the epilogue:

- **waves 2-7** (mask <= 3 tiles): the diagonal and everything after it fall within the last 3 tiles
  -> **masked entirely in the epilogue** (C2/C6/C10 at 1467, 1530, 1592).
- **waves 0, 1** (mask 4 tiles): the earliest masked tile (`4b`) is outside the last 3 tiles:
  - **q-block 0 (`b = 0`)**: diagonal = tile 0 -> masked in the **prologue** (source 1227, gate
    `0 < 64` true).
  - **q-block >= 1 (`b >= 1`)**: diagonal = tile `4b` (even -> `v_s_0` slot) -> masked in the **main
    loop**, the `v_s_0` mask at source 1380-1384 (cluster 6):

```1379:1384:FlyDSL/kernels/flash_attn_gfx950.py
                    if const_expr(CAUSAL):
                        v_s_0 = _causal_mask_prologue_if_needed(
                            v_s_0,
                            j_idx - 1,
                            j_idx * BLOCK_N,
                        )
```

So 1380-1384 is **not** dead code: it masks the diagonal tile of waves 0/1 for any non-first q-block.

<a id="sec-1-7-4-why-the-main-loops-v_s_1-slot-skips-the-causal-mask"></a>
#### 1.7.4 Why the main loop's `v_s_1` slot skips the causal mask

In the main loop the `v_s_1` (odd-tile) slot only masks when `CROSS_SEQLEN` is set (source 1314),
whereas the epilogue masks `v_s_1` unconditionally under `CAUSAL` (source 1467/1592). This asymmetry
is safe for self-attention because of the parity table above:

- every diagonal tile that can land **inside the main loop** is `4b` (even -> `v_s_0` slot, masked by
  1380-1384), and
- the **odd** diagonal tiles (`4b+1`, `4b+3`, for waves 2/3 and 6/7) always fall in the last 3 tiles
  -> handled by the epilogue's unconditional `v_s_1` mask.

The two paths jointly cover all even/odd diagonal tiles, so self-attention can drop the main-loop
`v_s_1` mask with no correctness loss (and no extra masking instructions in the hot loop).

**Summary**: the causal mask runs from each wave's own **diagonal tile** through the last tile. For
most waves that range is entirely in the epilogue, but waves 0/1 start masking earlier -- in the
prologue (first q-block) or in the main loop (later q-blocks). The pipeline always processes
`prologue 1 + loop 2N + epilogue 3` tiles, with odd tile counts padded up by one masked, zero-input
tile.

---

### 1.8 Worked examples: block allocation and per-wave causal-mask sites

This section applies 1.7 to concrete shapes. All cases are **self-attention** (`seqlen_q ==
seqlen_kv`, so `delta = 0`), causal, dense (non-split-K, non-varlen), `BLOCK_M = 256`,
`BLOCK_N = 64`, 8 waves x 32 rows. `FLYDSL_DUALWAVE_CAUSAL_FOLD=1` sets
`dualwave_swp_causal_fold=True` (`flash_attn_generic.py:166`) -> `_DO_FOLD = CAUSAL_FOLD and CAUSAL
and not VARLEN` is true (source 243). The five candidate mask sites are exactly those listed by the
question:

| Site | Source | Score slot | Tile it masks |
|---|---|---|---|
| Prologue | 1227 | `v_s_0` | `0` |
| Main loop C6 | 1380-1384 | `v_s_0` | `j_idx - 1` (even tiles `2,4,...`) per iteration |
| Epilogue C2 | 1467-1472 | `v_s_1` | `max_m3 = E-3` (odd) |
| Epilogue C6 | 1530-1534 | `v_s_0` | `max_m2 = E-2` (even) |
| Epilogue C10 | 1592-1596 | `v_s_1` | `max_m1 = E-1` (odd) |

`E = max_num_tiles`. The main loop's `v_s_1` slot (C3, source 1314) is **not** in this list because
for self-attention it does not mask (`CROSS_SEQLEN` is off). A site "executes the mask" for wave `w`
iff its runtime gate fires (source 902): `q_start + 32*w < kv_end_pos`, where `kv_end_pos = 64 *
(tile + 1)`.

<a id="sec-1-8-1-grid-block-allocation"></a>
#### 1.8.1 Grid (block) allocation

`num_q_blocks = ceil(S/256)`; `num_kv_tiles = ceil(S/64)`. The grid is `(NUM_HEADS_Q, grid_y,
batch)` (source 1951); the causal-relevant dimension is `grid_y` (source 1918-1921). Below counts
are per `(head, batch)`; multiply by `NUM_HEADS_Q * batch` for the absolute block count.

- **`FLYDSL_DUALWAVE_CAUSAL_FOLD=1`**: `grid_y = ceil(num_q_blocks / 2)` (source 1919). Each WG runs
  **2 passes** (source 1126): pass0 q-block `base`, pass1 its mirror `(_num_qb_total-1) - base`
  (source 1128-1130). When `num_q_blocks` is odd the middle WG's mirror equals `base`, so it
  recomputes the same q-block twice (idempotent).
- **`FLYDSL_DUALWAVE_CAUSAL_FOLD=0`**: `grid_y = num_q_blocks` (source 1921). Each WG runs **1 pass**
  (`_n_pass = 1`).

| S | num_q_blocks | num_kv_tiles | grid_y (FOLD=1) | WG -> q-block passes (FOLD=1) | grid_y (FOLD=0) | WG -> q-block (FOLD=0) |
|---|---|---|---|---|---|---|
| 127 | 1 | 2 | 1 | WG0: b0, b0 | 1 | WG0: b0 |
| 255 | 1 | 4 | 1 | WG0: b0, b0 | 1 | WG0: b0 |
| 511 | 2 | 8 | 1 | WG0: b0, b1 | 2 | WG0: b0; WG1: b1 |
| 513 | 3 | 9 | 2 | WG0: b0, b2; WG1: b1, b1 | 3 | WG0: b0; WG1: b1; WG2: b2 |
| 1025 | 5 | 17 | 3 | WG0: b0, b4; WG1: b1, b3; WG2: b2, b2 | 5 | WG0..WG4: b0,b1,b2,b3,b4 |

<a id="sec-1-8-2-per-q-block-tile-count-e-and-mask-pattern"></a>
#### 1.8.2 Per-q-block tile count `E` and mask pattern

`causal_num_tiles = 4*(b+1)`; `E_raw = min(causal_num_tiles, num_kv_tiles)`; `E =
max(round_up_even(E_raw), 4)` (source 1140-1151). The diagonal (first masked) tile of wave `w` is
`4b + floor(w/2)` (1.7.3), and the wave masks every tile from there to `E-1`.

| S | b | causal_tiles | E | pattern |
|---|---|---|---|---|
| 127 | 0 | 4 | 4 | A |
| 255 | 0 | 4 | 4 | A |
| 511 | 0 | 4 | 4 | A |
| 511 | 1 | 8 | 8 | B |
| 513 | 0 | 4 | 4 | A |
| 513 | 1 | 8 | 8 | B |
| 513 | 2 | 12 | 10 | C |
| 1025 | 0 | 4 | 4 | A |
| 1025 | 1 | 8 | 8 | B |
| 1025 | 2 | 12 | 12 | B |
| 1025 | 3 | 16 | 16 | B |
| 1025 | 4 | 20 | 18 | C |

The three resulting patterns (which of the 5 sites fires, per wave):

**Pattern A** — first q-block, `b=0`, `E=4` (no main-loop iteration; tiles `0|1,2,3`):

| wave | mask sites |
|---|---|
| 0, 1 | Prologue, Epi C2, Epi C6, Epi C10 |
| 2, 3 | Epi C2, Epi C6, Epi C10 |
| 4, 5 | Epi C6, Epi C10 |
| 6, 7 | Epi C10 |

**Pattern B** — interior full block, `b>=1`, `E = 4(b+1)` (diagonal band `4b..4b+3`; tile `4b` lands
in Main loop C6, `4b+1/4b+2/4b+3` in Epi C2/C6/C10):

| wave | mask sites |
|---|---|
| 0, 1 | Main loop C6, Epi C2, Epi C6, Epi C10 |
| 2, 3 | Epi C2, Epi C6, Epi C10 |
| 4, 5 | Epi C6, Epi C10 |
| 6, 7 | Epi C10 |

(Pattern B differs from A only for waves 0,1: their diagonal tile `4b` is even and now sits in the
**main loop** `v_s_0` slot, source 1380-1384, instead of the prologue.)

**Pattern C** — last partial block whose only real query row is its first row (here `S = 256k+1`, so
the tail block has 1 real row at the diagonal tile `E-2`; `E` was truncated by `num_kv_tiles`):

| wave | mask sites |
|---|---|
| 0, 1 | Epi C6, Epi C10 |
| 2, 3 | Epi C10 |
| 4, 5 | (none) |
| 6, 7 | (none) |

(The diagonal tile is `E-2` (Epi C6) and the padding tile `E-1` (Epi C10); the main loop and Epi C2
sit entirely below the diagonal, so they never fire. Waves 4-7 cover only out-of-range rows and mask
nothing.)

#### 1.8.3 Putting it together per scale

For **FOLD=0** each WG = one q-block, so its 8 waves follow that block's pattern directly. For
**FOLD=1** each WG runs two passes, so a given wave's mask sites are the **union over its two
q-blocks** (each pass independently re-derives `E` and re-runs prologue/loop/epilogue).

- **S = 127 and S = 255** (1 q-block, b0 = Pattern A):
  - FOLD=0: 1 WG, waves follow Pattern A.
  - FOLD=1: 1 WG, but it runs b0 **twice** (pass0 and pass1 both b0). Each wave therefore executes
    its Pattern-A sites in *both* passes (same sites, recomputed). Net mask sites per wave are
    identical to FOLD=0; only the work is duplicated.
  - (For S=127 only tiles 0,1 are real; tiles 2,3 are out-of-range padding, read as 0 and fully
    masked. For S=255 all 4 tiles are real. The mask *sites* are the same because the gate keys off
    q-rows, not `num_kv_tiles`.)

- **S = 511** (2 q-blocks: b0=A, b1=B):
  - FOLD=0: WG0 = b0 (Pattern A), WG1 = b1 (Pattern B).
  - FOLD=1: a single WG0 runs b0 then b1. Per-wave union:

    | wave | b0 (A) sites | b1 (B) sites |
    |---|---|---|
    | 0,1 | Prologue, Epi C2/C6/C10 | Main C6, Epi C2/C6/C10 |
    | 2,3 | Epi C2/C6/C10 | Epi C2/C6/C10 |
    | 4,5 | Epi C6/C10 | Epi C6/C10 |
    | 6,7 | Epi C10 | Epi C10 |

- **S = 513** (3 q-blocks: b0=A, b1=B, b2=C):
  - FOLD=0: WG0=b0 (A), WG1=b1 (B), WG2=b2 (C).
  - FOLD=1: WG0 runs b0(A)+b2(C); WG1 runs b1(B)+b1(B) (mirror of the middle block is itself, so
    b1 twice). E.g. WG0 wave0 fires {Prologue, Epi C2/C6/C10} (b0) and {Epi C6, Epi C10} (b2).

- **S = 1025** (5 q-blocks: b0=A, b1=B, b2=B, b3=B, b4=C):
  - FOLD=0: WG0..WG4 = b0(A), b1(B), b2(B), b3(B), b4(C).
  - FOLD=1: WG0 = b0(A)+b4(C); WG1 = b1(B)+b3(B); WG2 = b2(B)+b2(B) (middle, twice).

#### 1.8.4 Summary

- The **per-q-block** wave->site mapping is the same whether FOLD is on or off — fold only changes
  *which* q-blocks a WG owns and *how many* WGs exist (`grid_y`). With FOLD=1 a WG's waves execute
  the union of two q-blocks' sites; with FOLD=0 a WG's waves execute one q-block's sites.
- Across all these self-attention shapes the causal mask **always reaches Epilogue C10** for every
  wave that has any real row at/after its diagonal, and **Epilogue C2/C6/C10** carry the bulk of the
  masking. The **Prologue** site only fires for waves 0,1 of the **first** q-block (Pattern A), and
  **Main loop C6** only fires for waves 0,1 of an **interior** q-block (Pattern B). The Main-loop
  `v_s_1` mask is never used for self-attention.

<a id="sec-1-8-5-what-causal_num_tiles-means-and-the-pattern-c-1-real-row-case"></a>
#### 1.8.5 What `causal_num_tiles` means, and the Pattern-C "1 real row" case

**`causal_num_tiles`** is the causal upper bound on how many 64-wide KV tiles this q-block can ever
attend to, derived from the q-block's **last** query row. From source 1136-1140:

```1136:1140:FlyDSL/kernels/flash_attn_gfx950.py
                causal_end_i32 = fx.Int32(q_start + BLOCK_M) + delta_i32
                causal_end_i32 = fx.Int32(
                    ArithValue(causal_end_i32 > fx.Int32(0)).select(causal_end_i32, fx.Int32(0))
                )
                causal_num_tiles = (fx.Index(causal_end_i32) + kv_tile_size - 1) // kv_tile_size
```

The block's last row is `q_start + BLOCK_M - 1`. Under bottom-right causal (keep key col
`<= row + delta`), the largest key column any row in the block attends to is `causal_end - 1 =
q_start + BLOCK_M - 1 + delta`. So `causal_num_tiles = ceil(causal_end / BLOCK_N)` is the number of
KV tiles spanning columns `[0, causal_end)`; every tile beyond it is entirely above the diagonal for
*all* rows (its columns exceed even the last row's limit) and is skipped. For self-attention
(`delta = 0`), `causal_end = q_start + BLOCK_M = 256*(b+1)` and `causal_num_tiles = 4*(b+1)`. The
kernel then forms `E = max(round_up_even(min(causal_num_tiles, num_kv_tiles)), 4)` (source
1140-1151): `causal_num_tiles` is the *triangle* bound, `num_kv_tiles` is the *key-length* bound, and
`E` is the smaller of the two (padded to even, floored at 4).

**Pattern C and the "q[512:513]" question (S = 513, b = 2).** Here `q_start = 512`, so this q-block
owns the 256 row slots `[512, 768)`, but `seqlen_q = 513` means only **row 512** is a real query row;
rows `513..767` are out of range. Two clarifications:

- **The 4 waves are not co-computing row 512.** Each wave owns a **disjoint** 32-row slice (`q_row =
  q_start + wave_id*32 + lane_mod_32`, source 1196/1200): wave0 -> `[512,544)`, wave1 -> `[544,576)`,
  wave2 -> `[576,608)`, wave3 -> `[608,640)`, ... wave7 -> `[736,768)`. Only **wave0** contains the
  single real row (512); every row of waves 1-7 (and rows 513-543 of wave0) is out of range. So
  exactly **one** wave produces a valid output row -- the four waves that the Pattern-C table shows
  "firing" are masking their *own* (mostly OOB) slices, not collaborating on row 512.
- **Why the table lists waves 0-3.** The mask gate is per-wave-uniform and keys off the wave's
  **minimum** row `q_start_pos = q_start + wave_id*32` (source 1199), not off how many rows are real.
  For `E = 10`: Epi C6 masks tile `E-2 = 8` (`kv_end = 576`), gate `512 + 32w < 576` -> `w < 2`
  (waves 0,1); Epi C10 masks tile `E-1 = 9` (`kv_end = 640`), gate `512 + 32w < 640` -> `w < 4`
  (waves 0,1,2,3). Waves 4-7 have `q_start_pos >= 640`, whose diagonal tile `4b + floor(w/2) >= 10`
  exceeds `E-1 = 9`, so no tile needs masking. The masking done by waves 1,2,3 lands on OOB rows and
  is simply discarded.

**Is Q padded to `q[512:640]`?** No -- and not to 640 specifically. The dense (non-split-K) path runs
**all 8 waves over the full `[512, 768)` block** with no per-wave early-out (only varlen/split-K have
guards, source 1176-1182). There is no explicit Q padding; instead:

- **Q load** uses a `buffer_load` whose `num_records` bound returns **0** for any row `>= seqlen_q`,
  so OOB rows behave as if zero-filled by hardware (for the entire `513..767` range, not just to
  640).
- **O store** in the dense path is `_buffer_store_128` (source 1714), also `num_records`-bounded, so
  writes to OOB rows (`q_row >= seq_len`) are **dropped by hardware** -- no explicit guard is needed.
  (Only the split-K path adds an explicit `q_row < seq_len` guard, source 1733, because its flat
  workspace is *not* `num_records`-bounded and an OOB write would corrupt a neighbor's slot.)

So functionally it behaves "as if" Q is zero-padded out to the block boundary 768, **all 8 waves
execute the full pipeline**, but only **wave0's row 512** yields a written output; the other waves'
work (including their causal masking) is computed and then discarded. This is wasted-but-harmless
work, the price of a fixed 256-row block with no per-wave row-count specialization.

<a id="sec-1-8-6-code-evidence-for-the-pattern-c-per-wave-mask-sites-s-513-b-2-e-10"></a>
#### 1.8.6 Code evidence for the Pattern-C per-wave mask sites (S = 513, b = 2, E = 10)

The single decision point is the runtime gate inside `_causal_mask_prologue_if_needed` (a
`@flyc.jit`, so the Python `if` lowers to an `scf.if`):

```898:907:FlyDSL/kernels/flash_attn_gfx950.py
        @flyc.jit
        def _causal_mask_prologue_if_needed(v_s, tile_idx=fx.Index(0), kv_end_pos=BLOCK_N):
            """Return masked score vectors when DUALWAVE_SWP's causal guard is active."""
            s_lo, s_hi = v_s
            if q_start_pos_i32 + delta_i32 < fx.Int32(kv_end_pos):
                lo_list, hi_list = _v_s_vec_to_lists(v_s)
                _causal_mask_inplace((lo_list, hi_list), tile_idx)
                s_lo = Vec.from_elements([_raw(v) for v in lo_list], fx.Float32).ir_value()
                s_hi = Vec.from_elements([_raw(v) for v in hi_list], fx.Float32).ir_value()
            return s_lo, s_hi
```

"Executes the causal mask" == the gate `q_start_pos_i32 + delta_i32 < kv_end_pos` is **true** so the
`if` body runs `_causal_mask_inplace`; when it is false the function returns the scores untouched
(line 907). The gate is **wave-uniform**: `q_start_pos_i32` is built from `wave_id_uni`, which comes
from `readfirstlane` (a scalar, identical for all 64 lanes of a wave), so every lane of a wave takes
the same branch -- there is no intra-wave divergence and the per-wave statement below is exact.

```308:313:FlyDSL/kernels/flash_attn_gfx950.py
        _wave_id_uni_i32 = rocdl.readfirstlane(
            T.i32,
            arith.divsi(_tid_i32, _raw(fx.Int32(WARP_SIZE))),
        )
        _stagger_i32 = arith.divsi(_wave_id_uni_i32, _raw(fx.Int32(4)))
        wave_id_uni = fx.Index(_wave_id_uni_i32)
```

```1199:1199:FlyDSL/kernels/flash_attn_gfx950.py
                q_start_pos_i32 = fx.Int32(q_start + wave_id_uni * ROWS_PER_WAVE)
```

So `q_start_pos = q_start + 32*wave_id` (the wave's minimum row), and for self-attention
`delta_i32 = 0` (source 359). For S = 513, b = 2: `q_start = 512`, `E = 10`. The two epilogue sites
that fire in Pattern C pass these `kv_end_pos`:

Epilogue C6 masks `v_s_0 = tile E-2 = 8`, with `kv_end_pos = max_m1 * BLOCK_N = (E-1)*64 = 576`:

```1529:1536:FlyDSL/kernels/flash_attn_gfx950.py
                if const_expr(CAUSAL):
                    v_s_0 = _causal_mask_prologue_if_needed(
                        v_s_0,
                        max_m2,
                        max_m1 * BLOCK_N,
                    )
                else:
                    v_s_0 = _seq_pad_mask_if_needed(v_s_0, max_m2)
```

Epilogue C10 masks `v_s_1 = tile E-1 = 9`, with `kv_end_pos = split_t_end * BLOCK_N = E*64 = 640`:

```1591:1598:FlyDSL/kernels/flash_attn_gfx950.py
                if const_expr(CAUSAL):
                    v_s_1 = _causal_mask_prologue_if_needed(
                        v_s_1,
                        max_m1,
                        split_t_end * BLOCK_N,
                    )
                else:
                    v_s_1 = _seq_pad_mask_if_needed(v_s_1, max_m1)
```

**Waves 0-3 DO execute the causal mask.** Substituting `q_start_pos = 512 + 32w`, `delta = 0`:

| site (`kv_end_pos`) | gate `512 + 32w < kv_end_pos` | waves where true |
|---|---|---|
| Epi C6 (`576`) | `32w < 64` -> `w < 2` | 0, 1 |
| Epi C10 (`640`) | `32w < 128` -> `w < 4` | 0, 1, 2, 3 |

So waves 0,1 take the `if` body at **both** Epi C6 and Epi C10; waves 2,3 take it at **Epi C10
only**. (The Prologue site is tile 0, `kv_end=64`: `512+32w < 64` is false for all waves; the Main
loop C6 sites are tiles `2..E-4=6`, max `kv_end = 7*64 = 448 < 512`, also false for all waves -- which
is why neither appears for Pattern C.)

**Waves 4-7 do NOT execute the causal mask.** Their `q_start_pos = 512 + 32w` is `640, 672, 704, 736`
for `w = 4,5,6,7`, so at every site the gate is false:

- Epi C6 (`576`): `640 < 576` ... `736 < 576` -> all false.
- Epi C10 (`640`): `640 < 640` is false; `672/704/736 < 640` -> all false.

With the gate false the `scf.if` body is skipped and `_causal_mask_prologue_if_needed` returns the
scores unchanged (source 907). Equivalently, these waves' diagonal tile `4b + floor(w/2) = 8 +
floor(w/2) >= 10` exceeds the last processed tile `E-1 = 9`, so none of their tiles ever cross the
causal boundary and no masking is needed (their rows `>= 640` are all out of range anyway, so the
results are discarded at the store regardless). The mask path still *exists* in the ISA for all
waves; waves 4-7 simply never take the masked branch at runtime.

---

## 2. Full GM/LDS/VGPR layout maps

Notation (all element formulas in bf16 elements unless marked byte; `byte = elem*2 + lds_kv_offset`):

```
tid     = thread_idx.x                         # 0..511
wave_id = tid / 64    (readfirstlane uniform: wave_id_uni, source 330-335)
lane    = tid % 64
lane_mod_32 = lane % 32       lane_div_32 = lane // 32
q_start = q_block_idx * BLOCK_M                 # source 345
kv_tile_start(j) = j * BLOCK_N                  # one KV tile = 64 keys
```

A distinguishing feature vs persistent-attention kernels: **Q is not staged in LDS**. Q is loaded
GM->VGPR directly; only K and V are double-buffered in LDS.

<a id="sec-2-1-q-gm---vgpr"></a>
### 2.1 Q: GM -> VGPR

`_load_q_all` (source 504-521) loads, per lane, the full head-dim of one Q row as 8 packs of 8 bf16.

```
q_row_in_block = wave_id*32 + lane_mod_32             # source 1193
q_row          = q_start + q_row_in_block
for ks in 0..7 (K_STEPS_QK):
    q_col   = ks*16 + lane_div_32*8                   # K_STEP_QK=16, MFMA_LANE_K=8 (source 507)
    g_idx   = q_row_in_block * stride_q_n + q_col
    pack_ks = buffer_load_dwordx4(q_rsrc, g_idx*2)    # 8 bf16 (v4i32)
v_q = concat(pack_0..pack_7)                          # 64 bf16
v_q = _scale_q_all(v_q)                               # *1/sqrt(D)*log2e, cast bf16
```

The base GM offset (batch/head/q-block) is folded into the buffer resource `q_rsrc`
(`q_gmem_byte_offset`, source 352-358).

Per-lane VGPR expansion (`lane_group = lane_div_32`):

| lane | lane_mod_32 | lane_group | `v_q` contents (Q row, dims) |
|---:|---:|---:|---|
| 0 | 0 | 0 | row 0, dims `0..7, 16..23, ..., 112..119` |
| 1 | 1 | 0 | row 1, dims `0..7, 16..23, ..., 112..119` |
| ... | ... | ... | ... |
| 31 | 31 | 0 | row 31, dims `0..7, 16..23, ..., 112..119` |
| 32 | 0 | 1 | row 0, dims `8..15, 24..31, ..., 120..127` |
| 33 | 1 | 1 | row 1, dims `8..15, 24..31, ..., 120..127` |
| 63 | 31 | 1 | row 31, dims `8..15, 24..31, ..., 120..127` |

So lanes `x` and `x+32` hold the two interleaved 8-bf16 halves of each 16-wide K-step for the **same**
query row -- exactly the `mma0` `src1` (Q) operand layout. The Q-scale fuses into the prologue `mma0`
chain in the ISA (lines 250-326): `v_mul_f32_e64`/`v_pk_mul_f32` (multiply by `c_sm_scale_log2e` =
`v64`), `v_and 0xffff0000`/`v_lshlrev 16` (bf16<->f32 unpack), `v_cvt_pk_bf16_f32` (repack), all
interleaved between the 16 `v_mfma`.

<a id="sec-2-2-k-gm---lds---vgpr"></a>
### 2.2 K: GM -> LDS -> VGPR

K is cooperatively DMA'd by all 512 threads, then re-gathered into the `mma0` operand.

**GM -> LDS (`_async_load_k`, `_k_dma_inner`, source lines 1101-1109 and 1042-1063):** each lane
transfers `VEC_KV = 8` bf16 (16 B) for each of `NUM_DMA_K = 2` loads, via
`buffer_load_dwordx4 ... lds`. For paged KV with `page_size = 64`, `PAGES_PER_TILE = 1`, so linear
and vectorized layouts both use one page descriptor per 64-row KV tile. The page-id mechanism is the
same, but the `(N, D)` assignment to `wave_id` / `lane` is different.

For the following tables, `N` is the row inside the current 64-row KV page/tile and `D` is the
head-dim coordinate (`0..127`). Each table cell describes the data loaded by the lanes in that lane
range. `K` linear has a padded LDS line stride of `SMEM_K_LINE_STRIDE = 520 bf16`: `512 bf16` useful
data plus `8 bf16` padding per line.

```
n_in_warp = lane // VEC_KV(8)        d_bucket = lane % VEC_KV(8)        # source lines 660-662
for d in 0..1 (NUM_DMA_K):
    # LDS destination line base (per wave, per d):
    lds_addr  = K_buf_base*2 + wave_id*520*2 + d*(8*520*2)             # 520 = SMEM_K_LINE_STRIDE
    # GM source (per lane gather):
    n_in_tile = n_in_warp*NUM_WAVES(8) + wave_id
    global_d  = d_bucket*VEC_KV(8) + d*D_128B_SIZE(64)
    lane_byte = n_in_tile*stride_kv_n_bytes + global_d*2
    raw_ptr_buffer_load_lds(k_rsrc, lds_addr, voffset=lane_byte, soffset=tile_start*stride_kv_n_bytes)
```

Paged ps64 **linear K** uses the formula above. `lane = 8*g + r`: `g = lane//8` selects the row
group and `r = lane%8` selects the 8-bf16 D bucket. Each wave writes two LDS lines (`d=0` and `d=1`).

| wave | d | K LDS line | lane 0-7 | lane 8-15 | lane 16-23 | lane 24-31 | lane 32-39 | lane 40-47 | lane 48-55 | lane 56-63 | LDS padding |
|---:|---:|---:|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | `N=0,D=0..63` | `N=8,D=0..63` | `N=16,D=0..63` | `N=24,D=0..63` | `N=32,D=0..63` | `N=40,D=0..63` | `N=48,D=0..63` | `N=56,D=0..63` | `+8 bf16` |
| 0 | 1 | 8 | `N=0,D=64..127` | `N=8,D=64..127` | `N=16,D=64..127` | `N=24,D=64..127` | `N=32,D=64..127` | `N=40,D=64..127` | `N=48,D=64..127` | `N=56,D=64..127` | `+8 bf16` |
| 1 | 0 | 1 | `N=1,D=0..63` | `N=9,D=0..63` | `N=17,D=0..63` | `N=25,D=0..63` | `N=33,D=0..63` | `N=41,D=0..63` | `N=49,D=0..63` | `N=57,D=0..63` | `+8 bf16` |
| 1 | 1 | 9 | `N=1,D=64..127` | `N=9,D=64..127` | `N=17,D=64..127` | `N=25,D=64..127` | `N=33,D=64..127` | `N=41,D=64..127` | `N=49,D=64..127` | `N=57,D=64..127` | `+8 bf16` |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | 0 | 7 | `N=7,D=0..63` | `N=15,D=0..63` | `N=23,D=0..63` | `N=31,D=0..63` | `N=39,D=0..63` | `N=47,D=0..63` | `N=55,D=0..63` | `N=63,D=0..63` | `+8 bf16` |
| 7 | 1 | 15 | `N=7,D=64..127` | `N=15,D=64..127` | `N=23,D=64..127` | `N=31,D=64..127` | `N=39,D=64..127` | `N=47,D=64..127` | `N=55,D=64..127` | `N=63,D=64..127` | `+8 bf16` |

Within each cell, the 8 lanes do not all load the full D range individually. More precisely, lane
`8*g + r` loads one `buffer_load_dwordx4` vector: `D = 64*d + 8*r .. 64*d + 8*r + 7` at
`N = 8*g + wave`.

Paged ps64 **vectorized K** uses the `KV_VECTORIZED` branch of `_k_dma_inner`:

```
for d in 0..1:
    oct_idx = wave_id*128 + d*64 + lane
    ni      = oct_idx % 64              # lane
    dg      = oct_idx // 64             # 2*wave_id + d
    src_ni  = sigma(ni)
    D       = 8*dg .. 8*dg+7
    N       = src_ni

sigma(x) = (x & 3) | ((x & 8) >> 1) | ((x & 4) << 1) | (x & ~15)
```

Here each wave owns a 16-column D slice, and `d=0/1` selects the lower/upper 8 columns of that slice.
The native vectorized K LDS write is dense for the useful `64*128 = 8192 bf16` tile; the allocated K
buffer is `8320 bf16`, so the padding is a single `128 bf16` tail before V, not per wave line.

| wave | d | D loaded by each lane | lane 0-3 | lane 4-7 | lane 8-11 | lane 12-15 | lane 16-19 | lane 20-23 | lane 24-27 | lane 28-31 | lane 32-35 | lane 36-39 | lane 40-43 | lane 44-47 | lane 48-51 | lane 52-55 | lane 56-59 | lane 60-63 | LDS padding |
|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | `D=0..7` | `N=0..3` | `N=8..11` | `N=4..7` | `N=12..15` | `N=16..19` | `N=24..27` | `N=20..23` | `N=28..31` | `N=32..35` | `N=40..43` | `N=36..39` | `N=44..47` | `N=48..51` | `N=56..59` | `N=52..55` | `N=60..63` | `+128 bf16/tile tail` |
| 0 | 1 | `D=8..15` | same N map | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same |
| 1 | 0 | `D=16..23` | same N map | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same |
| 1 | 1 | `D=24..31` | same N map | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same |
| 2 | 0 | `D=32..39` | same N map | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same |
| 2 | 1 | `D=40..47` | same N map | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | 0 | `D=112..119` | same N map | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same |
| 7 | 1 | `D=120..127` | same N map | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same | same |

In the linear K map above, the cooperative DMA deposits each lane's 16-byte vector at `lane*VEC_KV`
within its wave's LDS line. Within one line (one wave, one `d`): lanes 0-7 = row `wave`, lanes
8-15 = row `wave+8`, ..., lanes 56-63 = row `wave+56` (8 KV rows x 64 dims).

**Linear K LDS buffer map** (one buffer; line = 520 bf16 = 1040 B = 1024 data + 16 pad):

```
K LDS line = d_half * NUM_WARPS(8) + wave_id          # 0..15
            ┌──────────────────────────────┐
 line 0:    │ d_half 0, rows 0,8,...,56     │ ...pad(16B)...
 line 1:    │ d_half 0, rows 1,9,...,57     │ ...pad...
 ...        ...
 line 7:    │ d_half 0, rows 7,15,...,63    │ ...pad...
 line 8:    │ d_half 1, rows 0,8,...,56     │ ...pad...   (dims 64..127)
 ...        ...
 line 15:   │ d_half 1, rows 7,15,...,63    │ ...pad...
```

**LDS -> VGPR (`_async_load_k_from_lds_to_vgpr` + `urk_base_per_lane`, source 419-423, 785-809):**
the following map is the linear/non-vectorized LDS re-gather used by the ISA dump; the vectorized K
path above writes native `[d//8, n, d%8]` order and uses the `KV_VECTORIZED` read branch. The linear
path re-gathers the padded stripe into `mma0` operand order via 16 `ds_read_b128` (8 K-steps x lo/hi
strip). The per-lane base and per-step offsets:

```
urk_base(lane) = (lane_mod_32 % 8) * 520               # DUALWAVE_SWP_SMEM_K_LINE_STRIDE
               + (lane_mod_32 // 8) * 64                # D_128B_SIZE
               + lane_div_32 * 8                        # VEC_KV
for ks in 0..7:
    ks_off = (ks//4)*4160 + (ks%4)*16                   # URK_KSTEP_OUTER=4160, URK_KSTEP_INNER=16
    idx_lo = K_buf_base + urk_base + ks_off             # v_s_lo strip (n_strip=0)
    idx_hi = idx_lo + 256                               # URK_N_STRIP_STRIDE (v_s_hi strip)
    k_lo[ks] = ds_read_b128(idx_lo)  ;  k_hi[ks] = ds_read_b128(idx_hi)
```

Lane -> LDS element/byte address for `ks = 0` (`byte = elem*2 + lds_kv_offset`):

| lane | lane_mod_32%8 | lane_mod_32//8 | lane_div_32 | K LDS elem base | K LDS byte (rel) | Logical K (row, dim) |
|---:|---:|---:|---:|---:|---|---|
| 0 | 0 | 0 | 0 | 0 | `0x000..0x00F` | row 0, dim `0..7` |
| 1 | 1 | 0 | 0 | 520 | `0x410..0x41F` | row 1, dim `0..7` |
| 2 | 2 | 0 | 0 | 1040 | `0x820..0x82F` | row 2, dim `0..7` |
| ... | ... | ... | ... | ... | ... | ... |
| 7 | 7 | 0 | 0 | 3640 | `0x1C70..0x1C7F` | row 7, dim `0..7` |
| 8 | 0 | 1 | 0 | 64 | `0x080..0x08F` | row 8, dim `0..7` |
| 9 | 1 | 1 | 0 | 584 | `0x490..0x49F` | row 9, dim `0..7` |
| ... | ... | ... | ... | ... | ... | ... |
| 31 | 7 | 3 | 0 | 3832 | `0x1DF0..0x1DFF` | row 31, dim `0..7` |
| 32 | 0 | 0 | 1 | 8 | `0x010..0x01F` | row 0, dim `8..15` |
| 33 | 1 | 0 | 1 | 528 | `0x420..0x42F` | row 1, dim `8..15` |
| ... | ... | ... | ... | ... | ... | ... |
| 63 | 7 | 3 | 1 | 3840 | `0x1E00..0x1E0F` | row 31, dim `8..15` |

The K read is intentionally **not lane-contiguous** (lane 1 reads byte `0x410` = line 1, while lane 8
returns to `0x080` inside line 0). This cross-line gather converts the cooperative GM->LDS stripe
into the MFMA-friendly `v_k` operand layout (`v_k: 16 packs of 8 bf16`). The loop axes cover the rest:
`ks%4` advances 16 bf16 within a D-half, `ks//4` jumps 4160 to the other D-half, `+256` selects the
second 32-col strip. ISA confirmation (one `mma0` K read, lines 1093-1099):

```asm
ds_read_b128 v[0:3],     v216           ; K pack, ks=0 lo
ds_read_b128 v[164:167], v216 offset:32 ; +16 bf16
ds_read_b128 v[168:171], v216 offset:512
...
```

<a id="sec-2-3-v-gm---lds---vgpr-hw-transpose-load"></a>
### 2.3 V: GM -> LDS -> VGPR (HW transpose load)

V uses a wider LDS padding (`SMEM_V_PAD = 32` bf16 vs K's 8) and a **hardware transpose LDS read**
instead of a software cross-line gather. For paged KV with `page_size = 64`, linear V uses the same
GM coordinate assignment as linear K, but vectorized V uses the aiter 5D V layout
`[NumBlocks, Hkv, PageSize/kVS, D, kVS]`, so one lane loads 8 consecutive `N` values for a single
`D`.

**GM -> LDS (`_async_load_v`, `_v_dma_inner`, source lines 1154-1162 and 1121-1139):** each lane
transfers `VEC_KV = 8` bf16 (16 B) for each of `NUM_DMA_V = 2` loads via
`buffer_load_dwordx4 ... lds`.

Paged ps64 **linear V** is byte-for-byte the same GM assignment as linear K (§2.2); only the
destination base and line stride change to `SMEM_V_LINE_STRIDE = 544 bf16` (`512 bf16` useful data +
`32 bf16` padding per line):

```
n_in_warp = lane // VEC_KV(8)        d_bucket = lane % VEC_KV(8)        # source lines 660-662
for d in 0..1 (NUM_DMA_V):
    # LDS destination line base (per wave, per d):
    lds_addr  = V_buf_base*2 + wave_id*544*2 + d*(8*544*2)             # 544 = SMEM_V_LINE_STRIDE
    # GM source (per-lane gather, byte-for-byte identical to K):
    n_in_tile = n_in_warp*NUM_WAVES(8) + wave_id
    global_d  = d_bucket*VEC_KV(8) + d*D_128B_SIZE(64)
    lane_byte = n_in_tile*stride_kv_n_bytes + global_d*2
    raw_ptr_buffer_load_lds(v_rsrc, lds_addr, voffset=lane_byte, soffset=tile_start*stride_kv_n_bytes)
```

| wave | d | V LDS line | lane 0-7 | lane 8-15 | lane 16-23 | lane 24-31 | lane 32-39 | lane 40-47 | lane 48-55 | lane 56-63 | LDS padding |
|---:|---:|---:|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | `N=0,D=0..63` | `N=8,D=0..63` | `N=16,D=0..63` | `N=24,D=0..63` | `N=32,D=0..63` | `N=40,D=0..63` | `N=48,D=0..63` | `N=56,D=0..63` | `+32 bf16` |
| 0 | 1 | 8 | `N=0,D=64..127` | `N=8,D=64..127` | `N=16,D=64..127` | `N=24,D=64..127` | `N=32,D=64..127` | `N=40,D=64..127` | `N=48,D=64..127` | `N=56,D=64..127` | `+32 bf16` |
| 1 | 0 | 1 | `N=1,D=0..63` | `N=9,D=0..63` | `N=17,D=0..63` | `N=25,D=0..63` | `N=33,D=0..63` | `N=41,D=0..63` | `N=49,D=0..63` | `N=57,D=0..63` | `+32 bf16` |
| 1 | 1 | 9 | `N=1,D=64..127` | `N=9,D=64..127` | `N=17,D=64..127` | `N=25,D=64..127` | `N=33,D=64..127` | `N=41,D=64..127` | `N=49,D=64..127` | `N=57,D=64..127` | `+32 bf16` |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | 0 | 7 | `N=7,D=0..63` | `N=15,D=0..63` | `N=23,D=0..63` | `N=31,D=0..63` | `N=39,D=0..63` | `N=47,D=0..63` | `N=55,D=0..63` | `N=63,D=0..63` | `+32 bf16` |
| 7 | 1 | 15 | `N=7,D=64..127` | `N=15,D=64..127` | `N=23,D=64..127` | `N=31,D=64..127` | `N=39,D=64..127` | `N=47,D=64..127` | `N=55,D=64..127` | `N=63,D=64..127` | `+32 bf16` |

Within each linear V table cell, lane `8*g + r` loads one `buffer_load_dwordx4` vector:
`D = 64*d + 8*r .. 64*d + 8*r + 7` at `N = 8*g + wave`.

Paged ps64 **vectorized V** uses the `KV_VECTORIZED` branch of `_v_dma_inner`:

```
for d in 0..1:
    row     = 2*wave_id + d                 # V LDS row and D-slice id
    g       = lane // 8                     # N group
    r       = lane % 8                      # D column inside the 8-wide slice
    N       = 8*g .. 8*g+7                  # loaded by one buffer_load_dwordx4
    D       = 8*row + r                     # single logical D column
    lds_row = row
```

Each wave writes two padded V LDS rows. Each row has `64 lanes * 8 bf16 = 512 bf16` useful data and
`32 bf16` padding, because `VEC_V_ROW_STRIDE = SMEM_V_LINE_STRIDE = 544 bf16`.

| wave | d | V LDS row | lane 0-7 | lane 8-15 | lane 16-23 | lane 24-31 | lane 32-39 | lane 40-47 | lane 48-55 | lane 56-63 | LDS padding |
|---:|---:|---:|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | `N=0..7,D=0..7` | `N=8..15,D=0..7` | `N=16..23,D=0..7` | `N=24..31,D=0..7` | `N=32..39,D=0..7` | `N=40..47,D=0..7` | `N=48..55,D=0..7` | `N=56..63,D=0..7` | `+32 bf16` |
| 0 | 1 | 1 | `N=0..7,D=8..15` | `N=8..15,D=8..15` | `N=16..23,D=8..15` | `N=24..31,D=8..15` | `N=32..39,D=8..15` | `N=40..47,D=8..15` | `N=48..55,D=8..15` | `N=56..63,D=8..15` | `+32 bf16` |
| 1 | 0 | 2 | `N=0..7,D=16..23` | `N=8..15,D=16..23` | `N=16..23,D=16..23` | `N=24..31,D=16..23` | `N=32..39,D=16..23` | `N=40..47,D=16..23` | `N=48..55,D=16..23` | `N=56..63,D=16..23` | `+32 bf16` |
| 1 | 1 | 3 | `N=0..7,D=24..31` | `N=8..15,D=24..31` | `N=16..23,D=24..31` | `N=24..31,D=24..31` | `N=32..39,D=24..31` | `N=40..47,D=24..31` | `N=48..55,D=24..31` | `N=56..63,D=24..31` | `+32 bf16` |
| 2 | 0 | 4 | `N=0..7,D=32..39` | `N=8..15,D=32..39` | `N=16..23,D=32..39` | `N=24..31,D=32..39` | `N=32..39,D=32..39` | `N=40..47,D=32..39` | `N=48..55,D=32..39` | `N=56..63,D=32..39` | `+32 bf16` |
| 2 | 1 | 5 | `N=0..7,D=40..47` | `N=8..15,D=40..47` | `N=16..23,D=40..47` | `N=24..31,D=40..47` | `N=32..39,D=40..47` | `N=40..47,D=40..47` | `N=48..55,D=40..47` | `N=56..63,D=40..47` | `+32 bf16` |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | 0 | 14 | `N=0..7,D=112..119` | `N=8..15,D=112..119` | `N=16..23,D=112..119` | `N=24..31,D=112..119` | `N=32..39,D=112..119` | `N=40..47,D=112..119` | `N=48..55,D=112..119` | `N=56..63,D=112..119` | `+32 bf16` |
| 7 | 1 | 15 | `N=0..7,D=120..127` | `N=8..15,D=120..127` | `N=16..23,D=120..127` | `N=24..31,D=120..127` | `N=32..39,D=120..127` | `N=40..47,D=120..127` | `N=48..55,D=120..127` | `N=56..63,D=120..127` | `+32 bf16` |

In the vectorized V table, a cell such as `N=0..7,D=0..7` means the lane group covers that D range as
a group. More precisely, lane `8*g + r` loads `N = 8*g .. 8*g+7` at the single column
`D = 16*wave + 8*d + r`.

**Linear V LDS buffer map** (one buffer; line = 544 bf16 = 1088 B = 1024 data + 64 pad):

```
V LDS line = d_half * NUM_WARPS(8) + wave_id          # 0..15
            ┌──────────────────────────────┐
 line 0:    │ d_half 0, rows 0,8,...,56     │ ...pad(64B)...
 line 1:    │ d_half 0, rows 1,9,...,57     │ ...pad...
 ...        ...
 line 7:    │ d_half 0, rows 7,15,...,63    │ ...pad...
 line 8:    │ d_half 1, rows 0,8,...,56     │ ...pad...   (dims 64..127)
 ...        ...
 line 15:   │ d_half 1, rows 7,15,...,63    │ ...pad...
```

For linear V, the only structural difference from the linear K LDS map (§2.2) is the wider
64 B/line padding (vs K's 16 B): the extra pad keeps the 16-lane `ds_read_b64_tr_b16` transpose
groups bank-conflict-free while preserving the row-major DMA store pattern. Vectorized V uses the
same 544-bf16 row stride, but its row id is `2*wave + d` and the row holds a D slice rather than
`N % 8 == wave`. Unlike K -- which re-gathers its operand with a cross-line software read (`u_rk`,
§2.2) -- V's MFMA-operand re-gather is performed by the hardware transpose read (§2.3.3).

#### 2.3.1 MFMA A-operand convention and CDNA4 transpose load

The CDNA `v_mfma_f32_32x32x16_bf16` (CDNA4 ISA line 13720) computes:

```
D = A (32x16) * B (16x32) + C (32x32)
A, B = bf16 ; C, D = f32 ; round-to-nearest-even ; NEG[1:0]/ABS[1:0] must be 0.
"Each operand contains a single matrix whose elements are distributed across all lanes of the wave."
```

In `_mma1_step_k` (source 979-988) the hardware operands are `src0 = v_pk` (V) and `src1 = p_pk` (P).
So after the swap-A/B convention, **V is the hardware MFMA A operand**. CDNA4 provides
`DS_READ_B64_TR_B16` exactly for loading such an A/B matrix operand directly out of LDS in transposed
orientation. ISA confirmation (one `mma1` MFMA, line 996):

```asm
v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[16:19], v[32:47]
;                        D=v_o     A=V(v_v)    B=P(v_p)  C=v_o
```

<a id="sec-2-3-2-ds_read_b64_tr_b16-instruction-semantics"></a>
#### 2.3.2 `ds_read_b64_tr_b16` instruction semantics

CDNA4 ISA section **11.4 MFMA Transpose Load from LDS** (lines 7270-7280):

> These instructions allow the user to perform matrix transpose while transferring 16/8/6/4-bit data
> from LDS to VGPRs. The operation takes **two instructions** with different LDS addresses and VGPR
> destinations. Prior to executing, **EXEC must be all 1's**. The LDS address must be aligned to the
> data size, and any DS op reading >= 64-bit data must use an **even-aligned VGPR**.
>
> `DS_READ_B64_TR_B16` -- column-major matrix A or row-major matrix B load to 2 VGPRs. Element size
> 16b. The first instruction loads K = 0..3 and 8..11; the next loads K = 4..7 and 12..15. **Each lane
> (one VGPR) holds 4 consecutive M or N values.**

In FlyDSL this is emitted as inline asm (`_ds_read_tr16_b64_imm`, source 104-115):

```python
raw = llvm.inline_asm(<2xi32>, [addr_i32], "ds_read_b64_tr_b16 $0, $1 offset:{imm}\n", "=v,v,~{memory}", ...)
return vector.BitCastOp(<4xf16>, raw)        # 64 bits/lane = 4 bf16 = VEC_TR_V
```

`_read_v_packs_for_buf` (source 811-835) issues **two** reads `+64 bf16` apart and concatenates them
into one 8-bf16 MFMA-A pack, matching the "two instructions load a complete matrix" rule:

```python
a = _ds_read_tr_v4f16_imm(lds_base, imm_lo)                              # K=0..3,8..11
b = _ds_read_tr_v4f16_imm(lds_base, imm_lo + URV_I5_STRIDE(64)*2)        # K=4..7,12..15
pack = concat(a, b)                                                      # 8 bf16
```

The transpose effect, per 16-lane group:

```
Before (LDS, DMA-friendly):           After ds_read_b64_tr_b16 (VGPR, MFMA-A order):
  V row0  lanes 0..3 hold dim0..3     lane 0 -> [V(row0,d0), V(row1,d0), V(row2,d0), V(row3,d0)]
  V row1  lanes 4..7 hold dim0..3     lane 1 -> [V(row0,d1), V(row1,d1), V(row2,d1), V(row3,d1)]
  ...                                 ...
```

This avoids a software `ds_read + v_perm/ds_permute` transpose sequence. The cost is the wider V LDS
line (`SMEM_V_LINE_STRIDE = 544` vs K's 520): the extra 64 B/line keeps the 16-lane transpose groups
bank-conflict-free.

<a id="sec-2-3-3-u_rv-address-formula-and-lane-map"></a>
#### 2.3.3 `u_rv` address formula and lane map

```
urv_base(lane) = lane_div_32        * 2176     # URV_GRPK   = 4 * 544
               + ((lane%16)//4)     * 544      # URV_LANE_HI = SMEM_V_LINE_STRIDE
               + ((lane//16)%2)     * 16       # URV_GRP_N  = lane_lo(4)*VEC_TR_V(4)
               + (lane%4)           * 4        # URV_LANE_LO = VEC_TR_V
for dc in 0..3 (D_CHUNKS):
    dc_off = (dc//2)*4352 + (dc%2)*32          # URV_DC_AXIS0=8*544, URV_DC_AXIS1=32
    for k_substep in 0..3:
        imm_lo = (k_substep*128 + dc_off) * 2  # URV_STEP_K_STRIDE = 2*64 = 128
        a = read_tr(V_buf_base + urv_base, imm_lo)
        b = read_tr(V_buf_base + urv_base, imm_lo + 64*2)
```

Per-lane V transpose-group expansion (`grp_id = lane//16`, `lane_in_grp = lane%16`), for
`dc=0, k_substep=0`:

| lane range | grp_id | p0 = lane//32 | p1 = (lane%16)//4 | p2 = (lane//16)%2 | p3 = lane%4 | V LDS elem base | Logical V (row, dim) |
|---:|---:|---:|---:|---:|---:|---|---|
| `0..3` | 0 | 0 | 0 | 0 | `0..3` | `0,4,8,12` | row 0, dim `0..15` |
| `4..7` | 0 | 0 | 1 | 0 | `0..3` | `544,548,552,556` | row 1, dim `0..15` |
| `8..11` | 0 | 0 | 2 | 0 | `0..3` | `1088,...` | row 2, dim `0..15` |
| `12..15` | 0 | 0 | 3 | 0 | `0..3` | `1632,...` | row 3, dim `0..15` |
| `16..19` | 1 | 0 | 0 | 1 | `0..3` | `16,20,24,28` | row 0, dim `16..31` |
| `20..23` | 1 | 0 | 1 | 1 | `0..3` | `560,...` | row 1, dim `16..31` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| `32..35` | 2 | 1 | 0 | 0 | `0..3` | `2176,...` | row 4, dim `0..15` |
| ... | ... | ... | ... | ... | ... | ... | ... |
| `60..63` | 3 | 1 | 3 | 1 | `0..3` | `3824,...` | row 7, dim `16..31` |

The 64-lane wave forms four 16-lane `ds_read_b64_tr_b16` transpose groups; `p1` (`lane_hi`, stride
544 = one V LDS line) selects the row inside an 8-row group, `p3` (`lane_lo`, stride 4 = `VEC_TR_V`)
the 4-bf16 D chunk. The remaining axes complete the tile: `dc//2` selects D-half (`+4352`), `dc%2`
the low/high 32-bf16 within a D-half (`+32`), `k_substep` advances the MFMA K reduction (`+128`),
`+64` (`URV_I5_STRIDE`) is the `a`/`b` split. ISA confirmation (lines 982-996):

```asm
ds_read_b64_tr_b16 v[174:175], v217 offset:9408
ds_read_b64_tr_b16 v[164:165], v217 offset:9536
...
v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[16:19], v[32:47]   ; v[192:195] = transpose-loaded V
```

### 2.4 S and P: VGPR-only score / probability layout

S never touches GM or LDS; it is the f32 output of `mma0`, stored as `(v_s_lo, v_s_hi)` = 32 f32/lane.
The score register mapping is established authoritatively by the kernel's own causal-mask derivation
(`_causal_mask_inplace`, source 847-862):

```
For output lane k (lane_group = lane//32), row index r in 0..15:
  v_s_lo[r] -> key column N = lane_group*4 + (r//4)*8 + (r%4)          # keys 0..31 region
  v_s_hi[r] -> key column N = 32 + lane_group*4 + (r//4)*8 + (r%4)     # keys 32..63 region
  query row M = q_start + wave_id*32 + lane_mod_32
Causal mask: set -inf where  M < kv_tile_start + N.
```

Per-lane interpretation:

| lane | query row M | lane_group | keys held in `v_s_lo` / `v_s_hi` |
|---:|---:|---:|---|
| 0 | q_start+0 | 0 | `{0..3,8..11,16..19,24..27}` / `{32..35,40..43,48..51,56..59}` |
| 32 | q_start+0 | 1 | `{4..7,12..15,20..23,28..31}` / `{36..39,44..47,52..55,60..63}` |
| 1 | q_start+1 | 0 | same key pattern as lane 0, query row 1 |
| 33 | q_start+1 | 1 | same key pattern as lane 32, query row 1 |

Each lane owns **one query row**; lanes `x` and `x+32` together hold all 64 key scores for that row.
This is why `_attn_row_max` / `_attn_sum` use `v_permlane32_swap_b32` (source 777-783): the per-lane
reduction over 32 elements is finished by exchanging with the `lane +/- 32` partner. After mask /
sub / exp2, `v_p = _cast_p(v_s)` keeps the same index layout in 32 bf16 (16 VGPR).

The causal mask in ISA (8 threshold pairs, source `pair_thresholds` 879-888) appears as paired
inline-asm `v_cmp_lt_i32_e64`/`v_cndmask_b32_e64` (lines 333-387):

```asm
;;#ASMSTART
v_cmp_lt_i32_e64 s[2:3],  v1, 0       ; rel < thr_x (0)
v_cmp_lt_i32_e64 s[10:11],v1, 1       ; rel < thr_y (1)
v_cndmask_b32_e64 v18, v18, v35, s[2:3]   ; v35 = 0xff800000 (-inf)
v_cndmask_b32_e64 v19, v19, v35, s[10:11]
;;#ASMEND
```

<a id="sec-2-5-o-vgpr---gm"></a>
### 2.5 O: VGPR -> GM

O is accumulated in `v_o` (4 banks x 16 f32 = 64 f32/lane), normalized by `1/l_row`, cast to
bf16/f16, and written GM-direct (no LDS). The store (source 1892-1923):

```
inv_l = (l_row > 0) ? rcp(l_row) : 0           # source 1868-1869
_scale_o(v_o, inv_l)                           # 32 v_pk_mul (source 3588-3619 in ISA)
if q_row < seq_len:
  for dc in 0..3 (D_CHUNKS):
    for store_group in 0..3:
      r_base    = store_group*4
      lo = cvt_pk_bf16_f32(v_o[dc][r_base],   v_o[dc][r_base+1])
      hi = cvt_pk_bf16_f32(v_o[dc][r_base+2], v_o[dc][r_base+3])
      d_row_rel = lane_div_32*4 + store_group*8
      d_col     = dc*32 + d_row_rel            # D_CHUNK = 32
      o_global  = (batch*seq_len + q_row)*stride_q_n + q_head*HEAD_DIM + d_col   # _global_idx_q (481-483)
      buffer_store_dwordx2([lo,hi], o_rsrc, o_global*2)
```

Per-lane store map (4 banks x 4 groups = 16 `buffer_store_dwordx2`):

| lane | query row | lane_group | O dims stored (one dwordx2 each) |
|---:|---:|---:|---|
| 0 | q_start+0 | 0 | `0..3, 8..11, 16..19, 24..27, 32..35, ..., 120..123` |
| 32 | q_start+0 | 1 | `4..7, 12..15, 20..23, 28..31, 36..39, ..., 124..127` |
| 1 | q_start+1 | 0 | same D pattern as lane 0, query row 1 |
| 33 | q_start+1 | 1 | same D pattern as lane 32, query row 1 |

ISA confirmation (lines 3622-3637): each store is preceded by two `v_cvt_pk_bf16_f32` packing 4 f32
into 2 dwords, then `buffer_store_dwordx2 v[0:1], v2, s[0:3], 0 offen offset:N`.

### 2.6 Summary: where each tensor lives

```
Q:  GM[B,N,H,D] -> _load_q_all -> VGPR v_q[64] bf16 (pre-scaled).            No LDS.
K:  GM[B,N,H_kv,D] -> _async_load_k -> LDS K0/K1 (8320 bf16, 16B pad/line)
                   -> _async_load_k_from_lds_to_vgpr (ds_read_b128 x16) -> VGPR v_k (mma0 src0).
V:  GM[B,N,H_kv,D] -> _async_load_v -> LDS V0/V1 (8704 bf16, 64B pad/line)
                   -> _read_v_packs_for_buf (ds_read_b64_tr_b16 x32) -> VGPR v_v (mma1 src0).
S:  VGPR only. v_s_0/v_s_1 = (v_s_lo 16, v_s_hi 16) f32. lane x & x+32 hold one 64-key row.
P:  VGPR only. v_p = (p_lo[2], p_hi[2]) = 32 bf16 (16 VGPR), same index layout as S.
O:  VGPR v_o[64] f32 -> *1/l_row -> cast bf16 -> buffer_store_dwordx2 -> GM[B,N,H,D].   No LDS.
```

### 2.7 Low-level call chains and issue counts

| Source statement | Role | Lowest-level primitive | ISA form | Count/call |
|---|---|---|---|---:|
| `_async_load_k/v(...)` (719-775) | GM tile -> LDS | `rocdl.raw_ptr_buffer_load_lds` | `buffer_load_dwordx4 ... lds` | 2 |
| `_load_q_all(...)` (504-521) | GM Q -> VGPR | `buffer_ops.RawPtrBufferLoadOp` | `buffer_load_dwordx4` | 8 |
| `_async_load_k_from_lds_to_vgpr` (785-809) | LDS K -> VGPR | `llvm.LoadOp` (addrspace 3, align 16) | `ds_read_b128` | 16 |
| `_read_v_packs_for_buf` (811-835) | LDS V -> VGPR (transpose) | inline asm `ds_read_b64_tr_b16` | `ds_read_b64_tr_b16` | 32 |
| `_mma0(v_k)` (837-845) | GEMM0 score tile | `rocdl.mfma_f32_32x32x16_bf16` | `v_mfma_f32_32x32x16_bf16` | 16 |
| `_mma1_step_k(...)` (979-988) | one GEMM1 K-step | `rocdl.mfma_f32_32x32x16_bf16` | `v_mfma_f32_32x32x16_bf16` | 4 |
| O store (1892-1923) | VGPR O -> GM | `rocdl.raw_buffer_store` | `buffer_store_dwordx2` | 16 |

Call chains (FlyDSL `expr` -> ROCDL/LLVM intrinsic -> ISA):

```
_async_load_k        -> rocdl.raw_ptr_buffer_load_lds(k_rsrc, lds_ptr, ...)  -> buffer_load_dwordx4 ... lds
_async_load_k_from_lds_to_vgpr -> llvm.LoadOp(<8xbf16>, lds_ptr, align=16)   -> ds_read_b128
_read_v_packs_for_buf -> _ds_read_tr16_b64_imm -> llvm.inline_asm("ds_read_b64_tr_b16 ...") -> ds_read_b64_tr_b16
_mma0 / _mma1_step_k -> _mfma_acc -> rocdl.mfma_f32_32x32x16_bf16(<16xf32>, [a,b,c]) -> v_mfma_f32_32x32x16_bf16
O store              -> rocdl.raw_buffer_store(o_pack, o_rsrc, byte_off)      -> buffer_store_dwordx2
```

Per-warp issue totals (verified against `21_final_isa.s`, section 1.5): 192 `v_mfma`, 192
`ds_read_b64_tr_b16`, 96 `ds_read_b128`, 32 `buffer_load_dwordx4`, 16 `buffer_store_dwordx2`.
