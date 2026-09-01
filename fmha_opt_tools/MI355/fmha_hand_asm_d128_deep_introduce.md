# Hand-Written Assembly Flash-Attention (gfx950, D=128) Deep Introduction

<!-- markdown-toc-generator:start -->
## Table of Contents

- [Authoritative Sources](#authoritative-sources)
- [Benchmark / Trait Configuration](#benchmark-trait-configuration)
  - [Kernel descriptor (from .rodata / .amdgpu_metadata, lines 3069-3097)](#sec-kernel-descriptor-from--rodata--amdgpu_metadata-lines-3069-3097)
  - [Kernel arguments (from .args, lines 3098-3208; pad slots elided)](#sec-kernel-arguments-from--args-lines-3098-3208-pad-slots-elided)
- [1. Kernel overview and software pipeline](#1-kernel-overview-and-software-pipeline)
  - [1.1 Buffering: where Q / K / V / S / P / O live (GM / LDS / VGPR)](#sec-1-1-buffering-where-q-k-v-s-p-o-live-gm-lds-vgpr)
  - [1.2 Online softmax (eager rescale) split into sub-steps](#sec-1-2-online-softmax-eager-rescale-split-into-sub-steps)
  - [1.3 Wave / workgroup structure and the two-body ping-pong loop](#13-wave-workgroup-structure-and-the-two-body-ping-pong-loop)
    - [1.3.1 Two wave-groups: a wave-id split + persistent asymmetric s_setprio](#sec-1-3-1-two-wave-groups-a-wave-id-split-persistent-asymmetric-s_setprio)
  - [1.4 Refined pseudocode](#14-refined-pseudocode)
  - [1.5 Per-operation main instructions and counts](#15-per-operation-main-instructions-and-counts)
    - [Whole-kernel ISA verification](#whole-kernel-isa-verification)
  - [1.6 Synchronization & scheduling primitives (s_barrier, s_waitcnt, s_setprio, m0)](#sec-1-6-synchronization)
  - [1.7 KV-tile distribution and where the causal mask runs](#17-kv-tile-distribution-and-where-the-causal-mask-runs)
    - [1.7.1 The KV-tile range per q-block](#171-the-kv-tile-range-per-q-block)
    - [1.7.2 Two-level masking: a uniform band gate + per-element thresholds](#172-two-level-masking-a-uniform-band-gate-per-element-thresholds)
    - [1.7.3 Where in the pipeline the mask code sits](#173-where-in-the-pipeline-the-mask-code-sits)
    - [1.7.4 The seq/padding mask](#174-the-seqpadding-mask)
  - [1.8 Worked examples: block allocation and causal-mask sites](#18-worked-examples-block-allocation-and-causal-mask-sites)
    - [1.8.1 Persistent two-q-block triangle fold](#181-persistent-two-q-block-triangle-fold)
    - [1.8.2 Per-q-block tile range and masked-tile band](#182-per-q-block-tile-range-and-masked-tile-band)
    - [1.8.3 Per-scale tables](#183-per-scale-tables)
    - [1.8.4 Contrast with the FlyDSL dualwave_swp sibling](#184-contrast-with-the-flydsl-dualwave_swp-sibling)
- [2. Full GM/LDS/VGPR layout maps](#2-full-gmldsvgpr-layout-maps)
  - [2.1 Q: GM -> LDS -> VGPR (load-once, resident)](#sec-2-1-q-gm---lds---vgpr-load-once-resident)
  - [2.2 K: GM -> LDS -> VGPR (ds_read_b128, double buffer)](#sec-2-2-k-gm---lds---vgpr-ds_read_b128-double-buffer)
  - [2.3 V: GM -> LDS -> VGPR (ds_read_b64_tr_b16 transpose, double buffer)](#sec-2-3-v-gm---lds---vgpr-ds_read_b64_tr_b16-transpose-double-buffer)
    - [2.3.1 ds_read_b64_tr_b16 instruction semantics](#sec-2-3-1-ds_read_b64_tr_b16-instruction-semantics)
  - [2.4 S and P: VGPR-only double-buffered score / probability](#24-s-and-p-vgpr-only-double-buffered-score-probability)
  - [2.5 O and LSE: VGPR -> GM](#sec-2-5-o-and-lse-vgpr---gm)
  - [2.6 Summary: where each tensor lives + VGPR map](#26-summary-where-each-tensor-lives-vgpr-map)
  - [2.7 LDS region map and low-level address chains](#27-lds-region-map-and-low-level-address-chains)
  - [Appendix: notable differences from the FlyDSL dualwave_swp sibling](#sec-appendix-notable-differences-from-the-flydsl-dualwave_swp-sibling)

<!-- markdown-toc-generator:end -->

This document deep-dives `fwd_kernel_func`, a **pure hand-written AMD GPU assembly** Flash-Attention
forward kernel for gfx950 (CDNA4 / MI350X-MI355X). Unlike the FlyDSL or CK kernels, there is no
higher-level source: the `.s` file **is** the source. Accordingly, every fact below (instruction
counts, register residency, address arithmetic) is traced to a line in the `.s`, and every
instruction's *semantics* are taken from the AMD CDNA4 ISA documents, not from memory.

The kernel's own header (line 5) declares its configuration:

```
//Config:BF16;FMHA;FWD;D128;1TG;8W;32mx8;64nx1;32x32x16;PROD;msk1;gm0
//Design     Author: Zhang,Niels   //KernelCode Author: Zhang,Niels; Yu,Jie
```

## Authoritative Sources

| Role | File |
|---|---|
| Target kernel (source == ISA) | `FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s` (3210 lines) |
| Sibling implementation | `FlyDSL/kernels/flash_attn_gfx950.py` (the FlyDSL "dualwave_swp" port; see `fmha_flydsl_dualwave_swp_deep_introduce.md`) |
| CDNA4 architecture whitepaper | `.cursor/rules/amd-cdna-4-architecture-whitepaper.txt` |
| CDNA4 ISA (instruction semantics) | `.cursor/rules/amd-instinct-cdna4-instruction-set-architecture.txt` |

## Benchmark / Trait Configuration

```
Arch              : gfx950 (CDNA4, MI350X / MI355X), wavefront = 64
dtype             : bf16   (Q, K, V, output R; accumulation in f32)
causal            : yes (msk1) ; also applies a seq-length / padding mask
Block / waves     : BLOCK_M = 256 (= 32 rows x 8 waves), BLOCK_N = 64, HEAD_DIM = 128
Workgroup         : 512 threads = 8 waves = "1TG"
MFMA              : v_mfma_f32_32x32x16_bf16   (W_M=32, W_N=32, W_K=16)
Features          : GQA (q_head/gqa -> kv head), variable seqlen (ptr_qseq/ptr_kseq),
                    optional LSE output, persistent (<= 2 q-blocks per workgroup)
```

<a id="sec-kernel-descriptor-from--rodata--amdgpu_metadata-lines-3069-3097"></a>
### Kernel descriptor (from `.rodata` / `.amdgpu_metadata`, lines 3069-3097)

| Resource | Value | ISA source |
|---|---|---|
| VGPR | 256 (max; waves_per_eu = 1) | `.amdhsa_next_free_vgpr 256`, `.vgpr_count: 256` |
| SGPR | 96 | `.amdhsa_next_free_sgpr 96`, `.sgpr_count: 96` |
| `accum_offset` | 160 | `.amdhsa_accum_offset 160` |
| LDS (group segment) | **163840 B = 160 KB** (full CDNA4 LDS) | `.amdhsa_group_segment_fixed_size 163840` |
| Private / scratch | 0 | `.private_segment_fixed_size: 0` |
| Workgroup | 512, fixed | `.reqd_workgroup_size : [512, 1, 1]` |
| kernarg size | 512 B | `.kernarg_segment_size: 512` |
| `ieee_mode` / `dx10_clamp` | 0 / 0 | `.amdhsa_ieee_mode 0`, `.amdhsa_dx10_clamp 0` |

The full 160 KB LDS reservation means **one workgroup occupies a whole CU's shared memory** (no
co-residency); together with 256 VGPR this is a deliberately "fat", occupancy-1 kernel that hides
latency by software pipelining rather than by wave-level parallelism.

<a id="sec-kernel-arguments-from--args-lines-3098-3208-pad-slots-elided"></a>
### Kernel arguments (from `.args`, lines 3098-3208; `pad` slots elided)

| reg | arg | meaning |
|---|---|---|
| `s[20:21]` | `R` | output O base ptr (read_write) |
| `s[8:9]` | `Q` | query base ptr |
| `s[12:13]` | `K` | key base ptr |
| `s[16:17]` | `V` | value base ptr |
| `s[24:25]` | `LSE` | log-sum-exp base ptr |
| `s28` | `scalar` | softmax scale (1/sqrt(D)) |
| `s30` / `s7` | `seq_len` / `kv_seq_len` | query / key sequence lengths |
| `s50,s31,s32,s33` | `Seqs,Ts,Hs,Bs` | Q strides (seq, tile, head, batch) |
| `s47,s48,s49` | `k_Seqs,k_Hs,k_Bs` | K strides |
| `s76,s77,s78` | `v_Seqs,v_Hs,v_Bs` | V strides |
| `s79,s80,s81` | `r_Seqs,r_Hs,r_Bs` | output strides |
| `s46` / `s90` | `gqa` / `q_head_num` | GQA group size / number of query heads |
| `s55` / `s56` | `msk_opt` / `lse` | mask mode / whether to write LSE |
| `s82` | `lse_Hs` | LSE head stride |
| `s[72:73],s[74:75]` | `ptr_qseq,ptr_kseq` | per-batch ragged sequence offsets |

## Table of Contents

- [1. Kernel overview and software pipeline](#1-kernel-overview-and-software-pipeline)
  - [1.1 Buffering: where Q / K / V / S / P / O live (GM / LDS / VGPR)](#11-buffering-where-q--k--v--s--p--o-live-gm--lds--vgpr)
  - [1.2 Online softmax (eager rescale) split into sub-steps](#12-online-softmax-eager-rescale-split-into-sub-steps)
  - [1.3 Wave / workgroup structure and the two-body ping-pong loop](#13-wave--workgroup-structure-and-the-two-body-ping-pong-loop)
    - [1.3.1 Two wave-groups: a wave-id split + persistent asymmetric `s_setprio`](#131-two-wave-groups-a-wave-id-split--persistent-asymmetric-s_setprio)
  - [1.4 Refined pseudocode](#14-refined-pseudocode)
  - [1.5 Per-operation main instructions and counts](#15-per-operation-main-instructions-and-counts)
  - [1.6 Synchronization & scheduling primitives (`s_barrier`, `s_waitcnt`, `s_setprio`, `m0`)](#sec-1-6-synchronization)
  - [1.7 KV-tile distribution and where the causal mask runs](#17-kv-tile-distribution-and-where-the-causal-mask-runs)
    - [1.7.1 The KV-tile range per q-block](#171-the-kv-tile-range-per-q-block)
    - [1.7.2 Two-level masking: a uniform band gate + per-element thresholds](#172-two-level-masking-a-uniform-band-gate--per-element-thresholds)
    - [1.7.3 Where in the pipeline the mask code sits](#173-where-in-the-pipeline-the-mask-code-sits)
    - [1.7.4 The seq/padding mask](#174-the-seqpadding-mask)
  - [1.8 Worked examples: block allocation and causal-mask sites](#18-worked-examples-block-allocation-and-causal-mask-sites)
    - [1.8.1 Persistent two-q-block triangle fold](#181-persistent-two-q-block-triangle-fold)
    - [1.8.2 Per-q-block tile range and masked-tile band](#182-per-q-block-tile-range-and-masked-tile-band)
    - [1.8.3 Per-scale tables](#183-per-scale-tables)
    - [1.8.4 Contrast with the FlyDSL dualwave_swp sibling](#184-contrast-with-the-flydsl-dualwave_swp-sibling)
- [2. Full GM/LDS/VGPR layout maps](#2-full-gmldsvgpr-layout-maps)
  - [2.1 Q: GM -> LDS -> VGPR (load-once, resident)](#21-q-gm---lds---vgpr-load-once-resident)
  - [2.2 K: GM -> LDS -> VGPR (`ds_read_b128`, double buffer)](#22-k-gm---lds---vgpr-ds_read_b128-double-buffer)
  - [2.3 V: GM -> LDS -> VGPR (`ds_read_b64_tr_b16` transpose, double buffer)](#23-v-gm---lds---vgpr-ds_read_b64_tr_b16-transpose-double-buffer)
    - [2.3.1 `ds_read_b64_tr_b16` instruction semantics](#231-ds_read_b64_tr_b16-instruction-semantics)
  - [2.4 S and P: VGPR-only double-buffered score / probability](#24-s-and-p-vgpr-only-double-buffered-score--probability)
  - [2.5 O and LSE: VGPR -> GM](#25-o-and-lse-vgpr---gm)
  - [2.6 Summary: where each tensor lives + VGPR map](#26-summary-where-each-tensor-lives--vgpr-map)
  - [2.7 LDS region map and low-level address chains](#27-lds-region-map-and-low-level-address-chains)

---

## 1. Kernel overview and software pipeline

The kernel implements **Flash-Attention forward** with online softmax for bf16 inputs. Each workgroup
(8 waves = 512 threads) owns **BLOCK_M = 256** query rows (32 per wave) and streams all KV tiles of
**BLOCK_N = 64** keys, accumulating the output O in registers. The grid is launched over
`(q-block, head, batch)`, and each workgroup is **persistent**: it processes up to **2 q-blocks**
before exiting (outer loop back-edge `s_cbranch_scc1 label_00D7` at line 3063, counter `s36 < 2`).

Both matrix products use the CDNA **swap-A/B operand convention** (operands are fed to the hardware
MFMA already swapped relative to math order):

- **GEMM0** (`v_mfma ... v[32:47], v[192:195], v[160:163]`, line 485): hardware `src0 = K`,
  `src1 = Q`, contraction over head-dim D. Realizes the score tile `S = Q . K^T` (scaled), landing
  with each lane owning **one query row** and the 64 key scores spread across the lane + its
  `lane +/- 32` partner.
- **GEMM1** (`v_mfma ... v[96:111], v[192:195], v[32:35]`, line 1084): hardware `src0 = V`,
  `src1 = P` (bf16). Realizes `O += P . V`, with each lane owning one query row and the 128 output
  channels across 4 accumulator banks.

**Which matrix is the MFMA left (`src0`/A) vs right (`src1`/B).** Each product is one
`v_mfma_f32_32x32x16_bf16 D, A, B, C` (`D = A*B + C`; A is 32x16, B is 16x32). In the *actual
assembly* the hardware **left** operand (`src0`/A) is the *mathematical right* matrix, and the
hardware **right** operand (`src1`/B) is the *mathematical left* matrix (the CDNA swap-A/B convention):

| Matmul | Math form | Math left | Math right | HW `src0` (A, left) | HW `src1` (B, right) | Accumulator |
|---|---|---|---|---|---|---|
| Q*K (scores) | `S = Q . K^T` | Q | K^T | **K** | **Q** | S |
| P*V (output) | `O += P . V` | P | V | **V** | **P** | O |

ISA evidence:
- Q*K: `v_mfma_f32_32x32x16_bf16 v[32:47], v[192:195], v[160:163], 0` (line 485) -> `src0` = **K**
  (`v[192:195]`, from `ds_read_b128`), `src1` = **Q** (`v[160:163]`, resident).
- P*V: `v_mfma_f32_32x32x16_bf16 v[96:111], v[192:195], v[32:35], v[96:111]` (line 1084) -> `src0` =
  **V** (`v[192:195]`, from `ds_read_b64_tr_b16`), `src1` = **P** (`v[32:35]`, bf16).

The swap is deliberate: with `src1 = Q` the CDNA 32x32 MFMA puts the **query row on the lane axis**
and (with `src0 = K`) the **key column on the accumulator registers** -- the "one query row per lane,
keys across registers" layout the online softmax (max/sum over keys) consumes; likewise `src0 = V` /
`src1 = P` puts the query row on lanes and the head-dim on registers for O.

<a id="sec-1-1-buffering-where-q-k-v-s-p-o-live-gm-lds-vgpr"></a>
### 1.1 Buffering: where Q / K / V / S / P / O live (GM / LDS / VGPR)

| Tensor | Global memory | LDS | VGPR (per lane) |
|---|---|---|---|
| Q | full Q tile (256x128) | **resident copy** in the high region (base `0x8200`), filled by 8 `buffer_load_dwordx4 ... lds` | `v[160:191]` = 32 VGPR (64 bf16), **read once** via `ds_read_b64` and kept for every GEMM0 |
| K | streamed tile by tile (64 rows) | **2 buffers** in the low region (base 0), `buffer_load_dwordx4 ... lds` | `v[192:255]` = 64 VGPR, read via `ds_read_b128` -> GEMM0 |
| V | streamed tile by tile (64 rows) | **2 buffers** in the high region (reuses Q's LDS) | `v[192:255]` = 64 VGPR (time-shared with K), read via HW transpose `ds_read_b64_tr_b16` -> GEMM1 |
| S (scores `Q.K^T`) | -- | none (register-only) | **2 buffers** `v[32:63]` and `v[64:95]` (32 f32 each), ping-ponged for software pipelining |
| P (probabilities) | -- | none (register-only) | bf16, overwrites the score buffer after softmax: `v[32:47]` or `v[64:79]` (16 VGPR = 32 bf16) |
| O (output) | written at the end | none | `v[96:159]` = 4 f32 banks x 16 = 64 f32, resident across the whole loop |

Two structural choices distinguish this kernel from the FlyDSL sibling:

1. **Q is staged in LDS, then made resident in VGPR.** Q is DMA'd to LDS once (lines 237-324),
   read into `v[160:191]` via `ds_read_b64` (lines 448-463), and then never touched in LDS again —
   so the high LDS region is **reused as the streaming V double-buffer**.
2. **K and V time-share the same 64-VGPR pool** `v[192:255]`: K is read there for GEMM0, then V
   overwrites it for GEMM1 of the (older) tile whose P is ready. They are never simultaneously live.

K and V each use a **2-deep LDS double buffer** (read-pointer pairs `v8`/`v9` for K, `v10`/`v11` for
V). In each tile a buffer is DMA-filled from GM while the other is read into VGPR.

<a id="sec-1-2-online-softmax-eager-rescale-split-into-sub-steps"></a>
### 1.2 Online softmax (eager rescale) split into sub-steps

The softmax is the standard online (running-max / running-sum) formulation, applied **eagerly**
every tile (there is no lazy "skip-rescale" branch like the FlyDSL kernel). One tile's softmax, as
emitted at `label_039D` (prologue) and inside each loop body (e.g. lines 1085-1204), is:

1. **row max** — `v21 = max(running_max, max_over_S)`: 16 `v_max3_f32` reduce the 32 score lanes
   pairwise, then `v_permlane32_swap_b32` + `v_max_f32` combine the `lane +/- 32` partner (lines
   654-674).
2. **correction** — `v16 = exp2(scale * (m_old - m_new))` (lines 1110-1115); `v_mul_f32` by the
   scale `s37`, then `v_exp_f32`.
3. **probabilities** — `P = exp2(scale*S - scale*m_new)`: `v_fma_f32 vX, vX, s37, -v23`
   (`v23 = scale*m_new`) followed by `v_exp_f32 vX, vX` for all 32 score lanes (lines 680-729).
4. **rescale O and l** — multiply the running output `v[96:159]` by `v16` (`v_mul_f32` /
   `v_pk_mul_f32`, lines 732-757) and `l = l*v16 + rowsum(P)`.
5. **row sum** — accumulate `rowsum(P)` into `l` (a `v_add_f32` chain + `v_permlane32_swap_b32`,
   lines 819-862).
6. **cast P -> bf16** — `v_cvt_pk_bf16_f32` packs the 32 f32 probabilities into 16 VGPR (lines
   877-894); this bf16 P is the GEMM1 `src1`.

The softmax temperature `scale = s28 * log2(e)` (line 202; `s29 = 0x3fb8aa3b = log2(e)`) is folded
into the exp argument (step 3), **not** pre-applied to Q. So `P = 2^(scale*log2e*(S - m)) =
e^(s28*(S - m))`.

### 1.3 Wave / workgroup structure and the two-body ping-pong loop

The 8 waves act as one cooperative threadgroup (`1TG`): they jointly DMA each K/V tile into LDS and
synchronize with `s_barrier`. Like the FlyDSL sibling, this kernel **also** splits the 8 waves into
**two wave-groups that overlap compute and memory** (see §1.3.1) -- but it does so with a different
mechanism: a single wave-id branch plus a *persistent asymmetric* `s_setprio`, rather than the
FlyDSL stagger barrier. There are only **2** `s_setprio` in the whole kernel (line 794 = `0`, line
1691 = `1`), one per wave-group, set once and held.

The main loop is structured as **two alternating bodies**, each **unrolled by 2 KV tiles**, that
ping-pong the double-buffered score registers (`v[32:63]` <-> `v[64:95]`) and the K/V LDS buffers.
As §1.3.1 shows, the two bodies are not just a per-wave ping-pong: they are the **two wave-groups'
separate loop bodies**, run offset in phase.

```
prologue (label_00D7)
   -> GEMM0(tile0) -> S in v[32:63] -> mask -> softmax (label_039D)
body A (label_048C / label_066F / label_095F)    # 2 tiles; K/V buffer parity 1/0
   -> GEMM0(tile i+1) -> v[64:95]
   -> GEMM1(tile i)   -> v[96:159]  ||  GEMM0(tile i+2) -> v[32:63]   ||  softmax
   -> GEMM1(tile i+1) -> v[96:159]
   -> exit to label_104A, OR fall through to body B
body B (label_0A71 / label_0C32 / label_0F1E)    # 2 tiles; mirror, K/V buffer parity 0/1
   -> same shape with the score buffers swapped, loops back to body A
drain (label_104A, label_1113) -> final GEMM1
epilogue (label_11D6) -> normalize O, write LSE, store O
outer loop: next q-block -> label_00D7 (up to 2 per workgroup)
```

Within each body the hand-scheduler **interleaves** the GEMM0 MFMA chain, the GEMM1 MFMA chain, the
softmax `v_exp`/`v_fma`/`v_add`, and the LDS reads, so the MFMA pipeline never stalls (visible in the
ISA as `v_mfma` lines spaced every ~6 VALU instructions, e.g. lines 1084-1204). The score
double-buffer lets GEMM0 of a later tile run while GEMM1 consumes an earlier tile's P; the two bodies
exist to keep the K/V LDS double-buffer parity correct across the 2-tile unroll.

<a id="sec-1-3-1-two-wave-groups-a-wave-id-split-persistent-asymmetric-s_setprio"></a>
#### 1.3.1 Two wave-groups: a wave-id split + persistent asymmetric `s_setprio`

The 8 waves split into **group A (waves 0-3)** and **group B (waves 4-7)** at exactly **one** point,
the loop back-edge right after the prologue's first tile-pair. `s5 = readfirstlane(tid >> 6)` is the
uniform wave id (source 53); the split is the second branch below (the first is the drain exit):

```768:771:FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s
  s_cmp_lt_i32  s39, s38                                // 000000001170: BF042627
  s_cbranch_scc0  label_104A                            // 000000001174: BF840BEC
  s_cmp_lt_i32  s5, 4                                   // 000000001178: BF048405
  s_cbranch_scc0  label_0A6F                            // 00000000117C: BF84060F
```

`s_cmp_lt_i32 s5, 4` is true for waves 0-3 (scc=1, fall through) and false for waves 4-7 (scc=0,
branch to `label_0A6F`). So:

- **Group A (waves 0-3)** falls through line 772: it first does its **memory** phase (V prefetch +
  16x `ds_read_b128` K, lines 772-793), then drops priority and enters body-1 compute:

```794:796:FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s
  s_setprio     0x0000                                  // 000000001228: BF8F0000
  s_barrier                                             // 00000000122C: BF8A0000
label_048C:
```

- **Group B (waves 4-7)** jumps to `label_0A6F`, raises priority, and only *after* the barrier does
  its memory phase (body-2 at `label_0A71`, V prefetch + K reads at 1694-1720):

```1690:1693:FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s
label_0A6F:
  s_setprio     0x0001                                  // 0000000029BC: BF8F0001
  s_barrier                                             // 0000000029C0: BF8A0000
label_0A71:
```

After the split each group runs its **own** body in a tight self-loop -- group A in body-1
(`label_048C`/`066F`/`095F`, back-edge `s_branch label_048C`, line 1689), group B in body-2
(`label_0A71`/`0C32`/`0F1E`, back-edge `s_branch label_0A71`, line 2576) -- and they reconverge only
at the drain (`label_104A`). Each body contains **6 `s_barrier`** per 2-tile iteration (group A:
lines 896, 1083, 1206, 1342, 1528, 1652; group B: 1722, 1819, 2012, 2163, 2260, 2453). Since
`s_barrier` matches by **ordinal position** (CDNA3/4 barrier-matching rule, `gpu-arch-mfma-valu.mdc`),
the two groups rendezvous at each of the 6 barriers while executing **different instruction schedules
in between** -- exactly the dual-group overlap.

The phase offset is set up by the asymmetric entry: group A runs its memory phase *before* the first
post-split barrier (line 795) while group B runs its memory phase *after* its barrier (line 1692). So
across the matched barriers **one group is in its MFMA compute window while the other is in its
DMA + `ds_read` memory window**, swapping at each barrier, so each group's memory latency hides behind
the other's MFMA (the MI300/MI350 MFMA-can-overlap-VMEM/LDS property, §4 of `gpu-arch-mfma-valu.mdc`):

```
group A (waves 0-3, prio 0):  [ mem: V,K reads ]--b--[ compute: GEMM0/1 + softmax ]--b--[ mem ]--b-- ...
group B (waves 4-7, prio 1):  [   compute ... ]--b--[ mem: V,K reads             ]--b--[ cmp ]--b-- ...
  b = matched s_barrier (6 per 2-tile body iteration)
```

The two `s_setprio` implement the **barrier priority-boost** pattern of `gpu-arch-mfma-valu.mdc`
(§"S_SETPRIO", `s_setprio` raised before `s_barrier` so the favored group reaches/passes the barrier
first), but applied **asymmetrically and persistently**: group B is pinned to priority **1** (high,
line 1691) and group A to priority **0** (low, line 794), each set **once** right before its barrier
and never toggled again. The high-priority group B is favored for the shared MFMA/VALU issue port, so
its loads fire promptly and it yields the pipe to whichever group is in its compute phase -- keeping
the stagger stable for the whole loop. This is simpler than the FlyDSL sibling, which instead opens
the offset with an extra prologue `s_barrier` on group B and re-brackets `s_setprio(1)`/`s_setprio(0)`
around every compute cluster.

### 1.4 Refined pseudocode

Names follow the registers in the `.s`. `scale = s28*log2e` (`s37`); `m`/`l` are the running
max/denominator; `O = v[96:159]`.

```
Prologue (label_00D7, lines 177-651):
  compute (batch, q_head, kv_head=q_head/gqa, q_block) base addresses    # lines 88-167, 178-234
  Q[256x128] -> LDS (high region, base 0x8200)                           # 8x buffer_load_dwordx4 ... lds
  zero O accumulator v[96:159]                                           # 64x v_mov_b32 = 0
  K tile0 -> LDS (low region, base 0)                                    # 2x buffer_load_dwordx4 ... lds
  if seq_len <= 0: goto epilogue (label_11D6)
  Q  -> v[160:191]  via ds_read_b64    (resident)                        # 16 reads
  K0 -> v[192:255]  via ds_read_b128                                     # 16 reads
  S0 = mma0(K0, Q) -> v[32:63]                                           # 16 MFMA
  if tile straddles diagonal: causal mask v[32:63]  (rel < thr -> -inf)  # 32 cmp + 32 cndmask
  if tile past kv_seq_len:    seq mask  v[32:63]  (thr < rel -> -inf)    # 32 cmp + 32 cndmask
  [label_039D] softmax(S0):
     m = rowmax(S0)                                                      # 16 v_max3 + permlane32_swap + v_max
     corr = exp2(scale*(m_old - m))                                      # (O is 0 here)
     P0 = exp2(scale*S0 - scale*m)                                       # 32 v_fma + 32 v_exp
     O *= corr ; l = l*corr + rowsum(P0)
     P0 -> bf16 in v[32:47]                                             # 16 v_cvt_pk_bf16_f32

Main loop (two alternating bodies A/B; each body is unrolled by 2 KV tiles):
  [body, e.g. label_048C/066F/095F]:   # the per-tile work below is emitted twice per body
     # MEMORY
     K_next -> LDS (DMA) ; advance K read pointer
     V_cur  -> LDS (DMA) ; advance V read pointer
     K_next -> v[192:255]  via ds_read_b128                              # 16 reads
     s_waitcnt vmcnt(.) ; s_barrier
     # COMPUTE (interleaved by hand)
     S_next = mma0(K_next, Q) -> other score buffer                      # 16 MFMA
     finish softmax of the current tile (exp / sum / cast P_cur -> bf16)
     V_cur  -> v[192:255]  via ds_read_b64_tr_b16  (overwrites K)        # 32 reads
     if straddle: causal mask S_next ; if past kv: seq mask S_next       # 64 cmp + 64 cndmask
     O += mma1(V_cur, P_cur) -> v[96:159]                               # 16 MFMA
     softmax(S_next): rowmax ; corr ; P_next=exp2 ; O*=corr ; l=l*corr+sum ; cast
     s_cmp tiles-remaining
     s_cbranch_scc0 label_104A          # exit to drain
     s_branch <body top>                # else next tile (swap buffers via the mirror body)

Drain (label_104A / label_1113, lines 2577-2846):
  read the last V tile via ds_read_b64_tr_b16
  finish the last softmax (exp / sum / cast)
  O += mma1(V_last, P_last) -> v[96:159]                                 # 16 MFMA

Epilogue (label_11D6, lines 2847-3020):
  inv_l = (l == 0) ? 0 : 1/l                                            # v_rcp_f32 + v_cmp + v_cndmask
  LSE   = scale*m + ln(l) = m*s28 + log2(l)*(1/log2e)                   # v_log_f32 + v_fma_f32
  O *= inv_l                                                            # 32 v_pk_mul_f32
  O -> bf16 ; reshape lanes (v_permlane32_swap_b32 + v_permlane16_swap_b32)
  store O   via 8x buffer_store_dwordx4                                  # to R
  if lse flag (s56): store LSE via buffer_store_dword

Outer loop (label_1300, line 3061):
  s36 += 1 ; recompute the second q-block's tile range (label_12FC / label_1300)
  if s36 < 2: goto label_00D7
  s_endpgm
```

### 1.5 Per-operation main instructions and counts

Counts are **per warp** (each VALU/memory op runs on all 64 lanes; each MFMA is warp-collective). The
tiling constants come from the config header and the MFMA shape:

```
BLOCK_M = 256 (= 32 rows x 8 waves)   BLOCK_N = 64   HEAD_DIM = 128
MFMA W_M = 32, W_N = 32, W_K = 16

GEMM0 (S = Q.K^T): E_M = 32/32 = 1, E_N = BLOCK_N/W_N = 64/32 = 2, E_K = HEAD_DIM/W_K = 128/16 = 8
                   MFMA = E_M*E_N*E_K = 1*2*8 = 16
GEMM1 (O += P.V) : E_M = 1, E_N = HEAD_DIM/W_N = 128/32 = 4, E_K = BLOCK_N/W_K = 64/16 = 4
                   MFMA = 1*4*4 = 16
```

| # | Operation (ISA evidence) | Main instruction(s) | Count / tile | Derivation |
|---|---|---|---|---|
| 1 | Q -> LDS (lines 237-322, **prologue only**) | `buffer_load_dwordx4 ... lds` | **8** | 256x128 bf16 / (512 lanes x 8 bf16) = 8 |
| 2 | K / V tile -> LDS (lines 441-443 / 773-775) | `buffer_load_dwordx4 ... lds` | **2 + 2** | one 64x128 KV tile, 2 D-halves each |
| 3 | Q -> VGPR (lines 448-463, **prologue only**) | `ds_read_b64` | **16** | 64 bf16/lane / 4 bf16 = 16 |
| 4 | K -> VGPR (lines 466-481 / 778-793) | `ds_read_b128` | **16** | 128 bf16/lane / 8 bf16 = 16 |
| 5 | V -> VGPR transpose (lines 908-939) | `ds_read_b64_tr_b16` | **32** | 128 bf16/lane / 4 bf16 = 32 (transpose) |
| 6 | `S = mma0(K, Q)` (line 485) | `v_mfma_f32_32x32x16_bf16` | **16** | GEMM0 = 1*2*8 |
| 7 | causal mask (lines 520-583) | `v_cmp_lt_i32` + `v_cndmask_b32` | **32 + 32** | 16 thresholds x 2 N-strips |
| 8 | seq/padding mask (lines 588-651) | `v_cmp_lt_i32` + `v_cndmask_b32` | **32 + 32** | second mask direction |
| 9 | row max (lines 654-674) | `v_max3_f32` + `v_permlane32_swap_b32` + `v_max_f32` | **16 + 1 + 1** | reduce 32 f32 + cross `lane+/-32` |
| 10 | `P = exp2(scale*(S-m))` (lines 680-729) | `v_fma_f32` + `v_exp_f32` | **32 + 32** | 32 score lanes |
| 11 | correction `exp2(scale*dm)` (1110-1115) | `v_sub_f32` + `v_mul_f32` + `v_exp_f32` | **1 + 1 + 1** | one scalar/lane |
| 12 | row sum (lines 819-862) | `v_add_f32` + `v_permlane32_swap_b32` + `v_add_f32` | **~31 + 1 + 1** | serial reduce + cross-lane |
| 13 | O rescale (lines 732-757) | `v_mul_f32` + `v_pk_mul_f32` | **2 + 31** | 64 O f32, packed 2/instr |
| 14 | cast `P -> bf16` (lines 877-894) | `v_cvt_pk_bf16_f32` | **16** | 32 f32 -> packed |
| 15 | `O += mma1(V, P)` (line 1084) | `v_mfma_f32_32x32x16_bf16` | **16** | GEMM1 = 1*4*4 |
| 16 | normalize `O *= 1/l` (lines 2904-2995) | `v_pk_mul_f32` | **32** | 64 O f32, packed |
| 17 | LSE = `scale*m + ln(l)` (2893-2900) | `v_log_f32` + `v_fma_f32` (+ `v_rcp_f32`) | **1 + 1** | one scalar/lane |
| 18 | O store reshape + write (2912-3015) | `v_cvt_pk_bf16_f32` + `v_permlane32_swap_b32` + `v_permlane16_swap_b32` + `buffer_store_dwordx4` | **32 + 16 + 16 + 8** | 64 f32 -> bf16, lane-reshape, 128-bit stores |

#### Whole-kernel ISA verification

Counts grepped from the `.s` with `//` comments stripped and the mnemonic anchored to the start of
the instruction field (the only `//` text is the disassembly address/encoding comment, never a
mnemonic):

| Instruction | Total | Notes |
|---|---:|---|
| `v_mfma_f32_32x32x16_bf16` | **176** | 11 GEMM blocks x 16 (see per-region table below) |
| `ds_read_b64_tr_b16` (V transpose) | **192** | 32/tile x 6 V-read sites |
| `ds_read_b128` (K) | **96** | 16/tile x 6 K-read sites |
| `ds_read_b64` (Q, once) | **16** | prologue only |
| `buffer_load_dwordx4 ... lds` | **34** | Q 8 + K/V DMA (2/tile) |
| `buffer_store_dwordx4` | **8** | O store (128-bit each) |
| `buffer_store_dword` | **1** | LSE |
| `v_cmp_lt_i32` / `v_cndmask_b32` | **320 / 328** | causal + seq mask, both directions |
| `v_exp_f32` | **179** | softmax P + corrections |
| `v_cvt_pk_bf16_f32` | **128** | P cast + O store cast |
| `v_pk_mul_f32` | **152** | O rescale + normalize |
| `v_fma_f32` | **161** | `scale*S - scale*m` + LSE |
| `v_add_f32` | **198** | row-sum chains |
| `v_max3_f32` / `v_max_f32` | **80 / 6** | row-max reductions |
| `v_permlane32_swap_b32` / `v_permlane16_swap_b32` | (cross-lane) | reductions + O store reshape |
| `s_barrier` | **20** | tile sync |
| `s_setprio` | **2** | minimal priority control |
| `s_waitcnt` | **23** | memory ordering |
| `v_log_f32` / `v_rcp_f32` | **1 / 3** | LSE + 1/l |

Per-region `v_mfma` distribution (consecutive labels), totalling 176:

| Region | role (accumulator targets verified in ISA) | `v_mfma` |
|---|---|---:|
| `label_00D7` | prologue GEMM0 of tile 0 -> `v[32:63]` | 16 |
| `label_039D` | softmax(tile 0) | 0 |
| `label_048C` | body A: GEMM0 of tile i+1 -> `v[64:95]` | 16 |
| `label_066F` | body A: GEMM1 of tile i -> `v[96:159]` (16) + GEMM0 of tile i+2 -> `v[32:63]` (16) | 32 |
| `label_095F` | body A: GEMM1 of tile i+1 -> `v[96:159]` | 16 |
| `label_0A71` | body B: GEMM0 -> `v[64:95]` | 16 |
| `label_0C32` | body B: GEMM1 -> `v[96:159]` + GEMM0 -> `v[32:63]` | 32 |
| `label_0F1E` | body B: GEMM1 -> `v[96:159]` | 16 |
| `label_104A` | drain: final GEMM1 -> `v[96:159]` | 16 |
| `label_1113` | drain: final GEMM1 -> `v[96:159]` | 16 |

So each loop body (A = `048C`/`066F`/`095F`, B = `0A71`/`0C32`/`0F1E`) is **unrolled by 2 KV tiles**:
2 GEMM0 + 2 GEMM1 = 64 MFMA per body. The exactly-determined memory/MFMA classes (`v_mfma`=176, `ds_read_b64_tr_b16`=192,
`ds_read_b128`=96, `ds_read_b64`=16, `buffer_store_dwordx4`=8) match the tiling derivation. The VALU
classes vary by region because rescale is applied every tile and the prologue/drain carry partial
softmax stages.

<a id="sec-1-6-synchronization"></a>
### 1.6 Synchronization & scheduling primitives (`s_barrier`, `s_waitcnt`, `s_setprio`, `m0`)

This is hand-scheduled ASM, so ordering is achieved with real synchronization instructions (there are
no compiler scheduling hints). Their semantics, from the CDNA4 ISA doc:

- **`s_barrier`** (ISA 8824): "Synchronize waves within a threadgroup." It does **not** wait for any
  memory counters — "if the barrier is being used to protect an outstanding memory operation use the
  appropriate `s_waitcnt` instruction before the barrier." So the kernel always pairs a preceding
  `s_waitcnt` with each `s_barrier` (e.g. lines 446-447, 464-465, 482-483).
- **`s_waitcnt`** (ISA 8836): waits for outstanding counts. `SIMM16[3:0]` = `vmcnt` low, `[6:4]` =
  `expcnt`, `[11:8]` = `lgkmcnt`, `[15:14]` = `vmcnt` high. The kernel uses:
  - `s_waitcnt vmcnt(N)` (e.g. `vmcnt(2)`, `vmcnt(4)`) to let the next compute proceed while N DMAs
    are still in flight — overlapping GM->LDS DMA with MFMA;
  - `s_waitcnt lgkmcnt(0)` before consuming `ds_read` results.
- **`s_setprio`** (ISA 8864): "User settable wave priority ... 0 = lowest, 3 = highest." Used twice,
  once per wave-group, to pin the two groups to **asymmetric persistent** priorities (group A =
  `s_setprio 0` at line 794, group B = `s_setprio 1` at line 1691) right before their matched
  barriers -- the dual-group overlap mechanism detailed in §1.3.1. The priorities are set once and
  held, not toggled per cluster like the FlyDSL sibling.
- **`m0` + `buffer_load_dwordx4 ... lds`**: the GM->LDS DMA destination is the `m0` register. The
  kernel sets `m0` to the LDS write base and advances it between sub-loads (e.g. `s_add_u32 m0,
  0x00002040, m0`, line 238). The `lds` modifier on the MUBUF load makes the data land directly in
  LDS (no VGPR round-trip), the CDNA direct-to-LDS path.

### 1.7 KV-tile distribution and where the causal mask runs

This section mirrors §1.7 of the FlyDSL `dualwave_swp` doc, but for the hand-ASM kernel, whose
masking model is **fundamentally different**: the causal-mask decision is **workgroup-uniform** (a
scalar "band" gate, not a per-wave gate), and the kernel is **persistent** (one workgroup processes
up to 2 q-blocks via a triangle fold). All line numbers refer to
`FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s`.

#### 1.7.1 The KV-tile range per q-block

At each q-block entry (`label_00D7`) the prologue computes the **KV column window** `[start, s38)`
this q-block streams, from the block's first and last rows (`delta = kv_seq_len - seq_len = s7 - s30`):

```177:189:FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s
label_00D7:
  s_add_u32     s38, s2, 1                              // 00000000035C: 80268102
  s_lshl_b32    s38, s38, 8                             // 000000000360: 8E268826
  s_lshl_b32    s51, s2, 8                              // 000000000364: 8E338802
  s_sub_i32     s40, s7, s30                            // 000000000368: 81A81E07
  s_add_i32     s51, s51, s40                           // 00000000036C: 81332833
  s_ashr_i32    s51, s51, 6                             // 000000000370: 90338633
  s_lshl_b32    s51, s51, 6                             // 000000000374: 8E338633
  s_add_i32     s38, s38, s40                           // 000000000378: 81262826
  s_cmp_lt_i32  s38, s7                                 // 00000000037C: BF040726
  s_cselect_b32 s38, s38, s7                            // 000000000380: 85260726
  s_cmp_lt_i32  s38, 64                                 // 000000000384: BF04C026
  s_cselect_b32 s38, 64, s38                            // 000000000388: 852626C0
```

- `s2` = q-block index, `BLOCK_M = 256`, `BLOCK_N = 64`.
- `s38 = clamp(min((s2+1)*256 + delta, kv_seq_len), >= 64)` -- the **last KV column** (loop end). This
  is the bottom-right causal end derived from the block's **last** row, capped by the real key length
  (the analogue of the FlyDSL `causal_num_tiles`/`num_kv_tiles` min, except the hand-ASM does **not**
  round to even or pad to 4 -- the two-body loop + drain absorb odd counts).
- `s51 = align_down(s2*256 + delta, 64)` -- the **mask-start column**: the diagonal column of the
  block's **first** row, rounded down to a tile boundary.

The loop runs the **running KV column** `s52` (forward `+= s53 = 64`, source 168-169, 759) while the
trip counter `s39` satisfies `s39 < s38` (`s_cmp_lt_i32 s39, s38; s_cbranch_scc0 label_104A`, source
768-769). The pipeline that consumes this window is **prologue (tile 0) + body A/B (2 KV tiles each)
+ drain (final GEMM1, no new tile)** -- see §1.3.

#### 1.7.2 Two-level masking: a uniform band gate + per-element thresholds

Unlike the FlyDSL kernel (whose `@flyc.jit` gate keys off the per-wave `q_start_pos`), the hand-ASM
guards the whole mask block with a **scalar, workgroup-uniform** compare:

```512:521:FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s
  s_cmp_lt_i32  s52, s51                                // 000000000A3C: BF043334
  s_cbranch_scc1  label_039D                            // 000000000A40: BF85010C
  s_sub_i32     s40, s51, s52                           // 000000000A44: 81A83433
  s_sub_i32     s41, s7, s30                            // 000000000A48: 81A91E07
  s_and_b32     s41, s41, 63                            // 000000000A4C: 8629BF29
  s_add_i32     s40, s40, s41                           // 000000000A50: 81282928
  v_add_i32     v12, v25, s40                           // 000000000A54: D29C000C 00005119
  s_nop         0x0000                                  // 000000000A5C: BF800000
  v_cmp_lt_i32  s[68:69], v12, 0                        // 000000000A60: D0C10044 0001010C
```

- **Level 1 (uniform band gate)**: `if s52 < s51 -> s_cbranch label_039D` skips the **entire** causal
  mask block and jumps straight to softmax. Because `s51` is derived from the block's **first** (=
  minimum) row, a tile whose start column `s52 < s51` is entirely below the diagonal for *every* row
  in the block -> no row needs masking -> the mask code is skipped for **all 8 waves together**. So
  the masked tiles form a **contiguous band** `[s51, s38)` at the top of the block's window; the
  tiles `[0, s51)` are full (no mask).
- **Level 2 (per-element thresholds)**: when the gate passes (`s52 >= s51`), the code builds
  `v12 = v25 + (s51 - s52) + (delta & 63)` where `v25 = wave*32 + (lane&31) - (lane>>5)*4` (source
  192-198), i.e. the lane's query row relative to the tile's diagonal, and then runs 16
  `v_cmp_lt_i32 v12, thr_r` + `v_cndmask_b32 ... v27` pairs (`v27 = 0xff800000 = -inf`, source 172)
  with the per-element thresholds `thr_r in {0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27}` (lines
  520-583, repeated for the second N-strip). This zeroes exactly the future keys (`col > row + delta`)
  for each row.

So the per-wave/per-row work is done by the **thresholds**, while *whether the mask block executes at
all* is the **uniform band gate**. This is the key structural contrast with the FlyDSL sibling.

#### 1.7.3 Where in the pipeline the mask code sits

The causal-mask block is emitted **once per GEMM0**, i.e. immediately after each tile's score is
produced, one tile ahead in the software pipeline. There are exactly **five** mask sites, and the
drain has **none**:

| Site | ISA (gate) | Score buffer masked | Tile it masks |
|---|---|---|---|
| Prologue | 512-513 | `v[32:63]` (S0) | tile 0 (the first window tile) |
| Body A, sub-tile 1 (`label_048C`) | 941-942 | `v[64:95]` (S1) | GEMM0 of "tile i+1" |
| Body A, sub-tile 2 (`label_066F`) | 1386-1387 | `v[32:63]` (S0) | GEMM0 of "tile i+2" |
| Body B, sub-tile 1 (`label_0A71`) | 1820-1821 | `v[64:95]` (S1) | GEMM0 of "tile i+1" |
| Body B, sub-tile 2 (`label_0C32`) | 2261-2262 | `v[32:63]` (S0) | GEMM0 of "tile i+2" |
| Drain (`label_104A`/`label_1113`) | -- | -- | none (final GEMM1 only) |

Each site is the same `s_cmp_lt_i32 s52, s51 / s_cbranch label_039x` gate followed by the 16x2
threshold compares. The drain (lines 2577-2846) only finishes the softmax + final GEMM1 of tiles
whose GEMM0 (and therefore masking) already happened inside a body, so it never re-masks. This is the
hand-ASM analogue of the FlyDSL "Prologue / Main-loop C6 / Epi C2/C6/C10" site list -- but here the
gate is uniform, so a site either masks for the whole workgroup or for none of it.

#### 1.7.4 The seq/padding mask

A second mask handles a non-64-aligned key length (`kv_seq_len % 64 != 0`). It is gated by a second
uniform compare against `s54 = align_down(kv_seq_len, 64)` (built once, source 173-174):

```584:590:FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s
  s_cmp_lt_i32  s52, s54                                // 000000000C60: BF043634
  s_cbranch_scc1  label_039D                            // 000000000C64: BF850083
  s_sub_i32     s40, s7, s52                            // 000000000C68: 81A83407
  v_sub_i32     v12, s40, v26                           // 000000000C6C: D29D000C 00023428
  v_cmp_lt_i32  s[68:69], 0, v12                        // 000000000C74: D0C10044 00021880
  v_cmp_lt_i32  s[70:71], 1, v12                        // 000000000C7C: D0C10046 00021881
  v_cndmask_b32  v32, v27, v32, s[68:69]                // 000000000C84: D1000020 0112411B
```

`if s52 < s54 -> skip`; so the seq mask runs only on the **last partial tile** (the one whose start
column reaches the final aligned 64-boundary), and it sets keys with absolute column `>= kv_seq_len`
to `-inf`. For a 64-aligned `kv_seq_len` it never fires.

### 1.8 Worked examples: block allocation and causal-mask sites

Same shapes as the FlyDSL doc -- `seqlen_q = seqlen_kv in {127, 255, 511, 513, 1025}`,
**self-attention** (`delta = 0`), causal (`msk1`), `BLOCK_M = 256`, `BLOCK_N = 64`, 8 waves.

#### 1.8.1 Persistent two-q-block triangle fold

This kernel is **persistent**: a workgroup processes up to 2 q-blocks (counter `s36 < 2`,
`s_cbranch label_00D7`, source 3062-3063). The second q-block is the **mirror** of the first
(triangle fold for load balance):

```3023:3031:FlyDSL/exp_isa/fmha_fwd_hd128_bf16_1tg_8w_256x64_350_msk1_gm0.s
  s_addk_i32    s36, 0x0001                             // 000000004B64: B7240001
  s_add_u32     s40, s30, 0x000000ff                    // 000000004B68: 8028FF1E 000000FF
  s_lshr_b32    s40, s40, 8                             // 000000004B70: 8F288828
  s_cmp_lt_u32  s40, 2                                  // 000000004B74: BF0A8228
  s_cselect_b32 s36, 2, s36                             // 000000004B78: 85242482
  s_add_u32     s41, s30, 0x000000ff                    // 000000004B7C: 8029FF1E 000000FF
  s_lshr_b32    s40, s41, 8                             // 000000004B84: 8F288829
  s_sub_u32     s40, s40, 1                             // 000000004B88: 80A88128
  s_sub_u32     s2, s40, s2                             // 000000004B8C: 80820228
```

- `num_q_blocks N = ceil(seq_len/256) = (s30 + 255) >> 8`.
- If `N < 2` (one q-block): `s36` is forced to 2 (line 3027), so the loop **does not run a second
  pass** -- a single q-block is processed exactly **once** (no idempotent double-pass, unlike the
  FlyDSL fold's middle WG).
- Else pass 2 uses `s2 = (N-1) - s2` (line 3031), the mirror block. For the heavy mirror block the
  tail (lines 3034-3055, gated by `s55 & 1`) can set a non-zero start `s52` and a **negative** stride
  `s53 = -64` (line 3054), so the mirror block streams its KV tiles **backward** from the diagonal --
  a load-balancing detail; the masking model (§1.7.2) is unchanged.

The grid `y` (q-block dimension) is set by the launcher, not this `.s`; with the fold active it is
`ceil(N/2)` workgroups, each owning a block + its mirror (for odd `N` the middle WG's mirror equals
itself, so it recomputes that block, idempotent). Below, the "WG -> q-blocks" column reflects this.

#### 1.8.2 Per-q-block tile range and masked-tile band

For self-attention (`delta = 0`): `s51 = align_down(b*256, 64) = b*256 =` column of **tile 4b**, and
`s38 = clamp(min((b+1)*256, S), >= 64)`. So:

- **masked (causal) tile band** = `[4b, ceil(s38/64))` -- tiles `< 4b` skip the mask uniformly.
- For an **interior** block (`(b+1)*256 <= S`): `s38 = 256(b+1)` -> tiles `[0, 4(b+1))`, masked band
  `{4b, 4b+1, 4b+2, 4b+3}` (4 tiles).
- For the **last** partial block: `s38 = S` -> tiles `[0, ceil(S/64))`, masked band `[4b, ceil(S/64))`.
- **seq mask**: only the last tile when `S % 64 != 0` (`s54 = align_down(S, 64)`).

(The mask is **workgroup-uniform**: on a masked tile all 8 waves run the mask block; the per-row
diagonal is then resolved by the element thresholds. There is no per-wave "this wave masks, that wave
doesn't" split as in FlyDSL.)

#### 1.8.3 Per-scale tables

`N = ceil(S/256)`; `T = ceil(S/64)` total tiles for the last block; fold pairing per §1.8.1.

| S | N | WG -> q-blocks (fold) | grid_y (fold) |
|---|---|---|---|
| 127 | 1 | WG0: b0 (single pass, no mirror) | 1 |
| 255 | 1 | WG0: b0 (single pass) | 1 |
| 511 | 2 | WG0: b0 + mirror b1 | 1 |
| 513 | 3 | WG0: b0 + b2 ; WG1: b1 + b1 | 2 |
| 1025 | 5 | WG0: b0 + b4 ; WG1: b1 + b3 ; WG2: b2 + b2 | 3 |

Per q-block: window tiles `[0, ceil(s38/64))`, causal band `[4b, ceil(s38/64))`, and the seq-mask
tile (if any):

| S | b | s38 = min(256(b+1),S) | window tiles | causal-masked tiles (uniform) | seq-mask tile |
|---|---|---|---|---|---|
| 127 | 0 | 127 | 0..1 | 0, 1 | tile 1 (`s54=64`) |
| 255 | 0 | 255 | 0..3 | 0, 1, 2, 3 | tile 3 (`s54=192`) |
| 511 | 0 | 256 | 0..3 | 0, 1, 2, 3 | none |
| 511 | 1 | 511 | 0..7 | 4, 5, 6, 7 | tile 7 (`s54=448`) |
| 513 | 0 | 256 | 0..3 | 0, 1, 2, 3 | none |
| 513 | 1 | 512 | 0..7 | 4, 5, 6, 7 | none |
| 513 | 2 | 513 | 0..8 | tile 8 | tile 8 (`s54=512`) |
| 1025 | 0 | 256 | 0..3 | 0, 1, 2, 3 | none |
| 1025 | 1 | 512 | 0..7 | 4, 5, 6, 7 | none |
| 1025 | 2 | 768 | 0..11 | 8, 9, 10, 11 | none |
| 1025 | 3 | 1024 | 0..15 | 12, 13, 14, 15 | none |
| 1025 | 4 | 1025 | 0..16 | tile 16 | tile 16 (`s54=1024`) |

Reading the table: for an interior block the causal mask runs on exactly its 4 diagonal-band tiles
(`4b..4b+3`) and is **skipped** on every lower tile; for the last partial block the band collapses to
the single diagonal tile (`b2` for S=513, `b4` for S=1025), which also takes the seq mask. Each
masked tile's mask code sits at the GEMM0 site that produced it (prologue for tile 0, otherwise one of
the body A/B sub-tile sites of §1.7.3); the drain never masks.

#### 1.8.4 Contrast with the FlyDSL dualwave_swp sibling

| Aspect | Hand-ASM `fwd_kernel_func` | FlyDSL `dualwave_swp` |
|---|---|---|
| Mask decision granularity | **workgroup-uniform** scalar band gate `s52 < s51` (lines 512-513) | **per-wave** `@flyc.jit` gate `q_start_pos + delta < kv_end_pos` |
| Below-diagonal tiles | mask code **skipped** for the whole WG (`[0, 4b)`) | each wave's tiles below its own diagonal are skipped per-wave |
| Tile-count shaping | exact window `[0, ceil(s38/64))`, **no even-pad / min-4** | rounded up to even, floored to 4, padded with masked OOB tiles |
| Multi-block | **persistent**: 2 q-blocks/WG (block + mirror), backward-stream option | one q-block/WG; optional `CAUSAL_FOLD` does the same 2-block fold |
| Single q-block | processed **once** (`s36` forced to 2) | with `FOLD=1`, processed **twice** (mirror == self) |
| Drain masking | none (drain only finishes GEMM1) | epilogue C2/C6/C10 **do** mask the last 3 tiles |
| seq/padding mask | explicit (`s52 >= s54`, lines 584-651) | causal mask alone covers the self-attn tail |

---

## 2. Full GM/LDS/VGPR layout maps

Notation:

```
tid     = v0 & 0x3ff             (packed thread id; lane = tid & 63, wave = tid >> 6)
lane    = v0 & 63                 wave_id (uniform) = s5 = readfirstlane(v0 >> 6)   (0..7)
b       = batch index (s4)        q_head = s3        kv_head = s62 = q_head / gqa
q_block = s2                      scale  = s28 (s37 = s28*log2e)
```

GM **base addresses** are strided BSHD with GQA (lines 88-167): for each tensor the per-workgroup
base is `ptr + b*Bs + head*Hs (+ q_block/seq offset)`, with the kv head folded through the GQA
divide (`s62 = q_head / gqa`, line 135). The buffer-resource descriptors are built in
`s[8:11]` (Q), `s[12:15]` (K), `s[16:19]` (V), `s[20:23]` (O) with the `0x00020000` stride word
(lines 98-102).

<a id="sec-2-1-q-gm---lds---vgpr-load-once-resident"></a>
### 2.1 Q: GM -> LDS -> VGPR (load-once, resident)

Q is the only tensor that is both staged in LDS and made resident. The flow:

```
GM Q[256x128] --(8x buffer_load_dwordx4 ... lds)--> LDS high region (base 0x8200)
            --(16x ds_read_b64)--> v[160:191]  (32 VGPR = 64 bf16, resident)
```

GM->LDS DMA address (lines 214-239): each lane reads one 16-byte (`dwordx4` = 8 bf16) vector per
sub-load; the source offset combines the lane's `(row, dim)` with the Q seq stride `s50`:

```
v13 = (lane>>3 & 1) * s50            # row group within wave
v14 = (lane>>4)     * s50 * 32       # row block
v12 = (lane & 7)    * 16             # 8-bf16 dim chunk
v4  = s59 + s5*s50*2 + v12 + v13 + v14     ; s59 = q_block * Ts (s31)
m0  = s63 = 0x8200 + 0x408*wave            # LDS write base, advanced by 0x2040/sub-load
buffer_load_dwordx4 v4, s[8:11], 0 offen lds
```

Q LDS->VGPR read (lines 448-463): 16 `ds_read_b64` from the two addresses `v2`,`v3` (the two
N-strips of the GEMM0 Q operand), at offsets `0, 8, 32, 40, 64, 72, 96, 104` within the Q LDS region
(base `0x8200`). Each `ds_read_b64` returns 4 bf16/lane; the 16 reads fill `v[160:191]` as the
GEMM0 `src1` (Q) operand. Because Q never changes across KV tiles, this read happens **once** and
`v[160:191]` is reused by every `mma0`.

The Q read address `v2` decodes (lines 327-343, byte offset relative to the Q region `0x8200`; the
Q LDS line stride is `0x408 = 1032 B = 516 bf16`):

```
v2(lane) = 0x8200 + ((lane & 31) >> 1)*0x408   # which of 16 Q-row lines
                  + (lane & 1)*0x80            # 128-B half within the line
                  + (lane >> 5)*16             # lane_group (0/1) 8-bf16 selector
                  + (wave & 3)*0x100 + (wave >> 2)*0x8100   # per-wave slab
v3 = v2 + 0x4080                               # second N-strip
```

**Per-lane Q LDS read address** (`wave`-relative; byte base for `ds_read_b64 ... offset:0`):

| lane | (lane&31)>>1 | lane&1 | lane_group | `v2` byte base | Q operand fragment |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | `0x0000` | row line 0, lo half |
| 1 | 0 | 1 | 0 | `0x0080` | row line 0, hi half |
| 2 | 1 | 0 | 0 | `0x0408` | row line 1, lo half |
| 3 | 1 | 1 | 0 | `0x0488` | row line 1, hi half |
| ... | ... | ... | ... | ... | ... |
| 31 | 15 | 1 | 0 | `0x3CF8` | row line 15, hi half |
| 32 | 0 | 0 | 1 | `0x0010` | row line 0, lo half, group 1 |
| 63 | 15 | 1 | 1 | `0x3D08` | row line 15, hi half, group 1 |

The 8 `ds_read_b64` immediate offsets (`0,8,32,40,64,72,96,104`) then walk the head-dim within each
line, so the 16 reads (2 strips x 8) gather the 64-bf16-per-lane Q `mma0` B operand into `v[160:191]`.

<a id="sec-2-2-k-gm---lds---vgpr-ds_read_b128-double-buffer"></a>
### 2.2 K: GM -> LDS -> VGPR (`ds_read_b128`, double buffer)

K is cooperatively DMA'd, double-buffered in the low LDS region, and read with regular 128-bit reads.

```
GM K[64x128] --(2x buffer_load_dwordx4 ... lds)--> LDS low region (K0 @ 0, K1 @ 0x4100)
           --(16x ds_read_b128)--> v[192:255]  (64 VGPR = 128 bf16)
```

GM->LDS DMA address (lines 392-409, 441-443): the lane->(row,dim) gather mirrors the FlyDSL kernel's
cooperative stripe — lane high bits select the KV row, low bits select the 16-byte dim chunk — with
the K seq stride `s47`. The DMA writes to `m0 = s64 = 0x410*wave` (K buffer 0) or `s65 = 0x4100 +
s64` (K buffer 1), advancing `m0` by `0x2080` per sub-load.

K LDS->VGPR read (`v9`, lines 778-793): 16 `ds_read_b128` from one of the two K read pointers
(`v8`/`v9`) at offsets that walk the D dimension and the two D-halves:

```
ds_read_b128 v[192:195], v9              # K dim 0..7   of N-group 0
ds_read_b128 v[208:211], v9 offset:512   # K dim 0..7   of N-group 1
ds_read_b128 v[196:199], v9 offset:32    # K dim 8..15
...
ds_read_b128 v[224:227], v9 offset:8320  # second D-half (dim 64..127), +0x2080
...
ds_read_b128 v[252:255], v9 offset:8928
```

The 16 reads gather the full 128-bf16-per-lane K operand into `v[192:255]` in `mma0` operand order
(A = K). The `offset:8320` jump (`= 0x2080`) crosses to the second D-half of the K tile in LDS.

The K read base `v8` decodes (lines 344-362; the K LDS line stride is `0x410 = 1040 B = 520 bf16`,
i.e. 1024 B data + 16 B pad — the same padded line as the FlyDSL sibling):

```
v8(lane) = (lane >> 5)*16            # lane_group (0/1): 8-bf16 selector inside a 16-bf16 step
         + (lane & 1)*0x80           # bit0 -> 128-B half
         + ((lane >> 3) & 3)*0x820   # which 2-line pair (cross-line gather)
         + (lane & 2)*0x208          # bit1 -> +0x410 line
         + (lane & 4)*0x40           # bit2 -> +0x100
v9 = v8 + 0x4100                     # K buffer 1
```

**K LDS address space** (one buffer; line = 1040 B; two D-halves at `+0` and `+0x2080`):

```
+---------------------------------------------+
| line 0  : KV rows (cooperative stripe)      | ...pad(16B)...   offset 0x0000  (dim 0..63)
| line 1  : ...                               | ...pad...        offset 0x0410
| ...                                         |
| line 7  : ...                               | ...pad...        offset 0x1C70
+---------------------------------------------+
| second D-half (dim 64..127): base +0x2080 (offset:8320 in the ds_read) |
+---------------------------------------------+
```

**Per-lane K LDS read address** (byte base for `ds_read_b128 ... offset:0`; non-contiguous on
purpose — this cross-line gather is what reorders the cooperative GM->LDS stripe into `mma0`
A-operand order):

| lane | bit0 | bit1 | bit2 | (lane>>3)&3 | lane_group | `v8` byte base |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | `0x0000` |
| 1 | 1 | 0 | 0 | 0 | 0 | `0x0080` |
| 2 | 0 | 1 | 0 | 0 | 0 | `0x0410` |
| 3 | 1 | 1 | 0 | 0 | 0 | `0x0490` |
| 4 | 0 | 0 | 1 | 0 | 0 | `0x0100` |
| 8 | 0 | 0 | 0 | 1 | 0 | `0x0820` |
| 16 | 0 | 0 | 0 | 2 | 0 | `0x1040` |
| 32 | 0 | 0 | 0 | 0 | 1 | `0x0010` |
| 63 | 1 | 1 | 1 | 3 | 1 | `0x1E00` |

The 16 immediate offsets (`0,32,64,96, 512,544,576,608, 8320,...,8928`) then walk the head-dim and
the second D-half (`+0x2080`), filling the 128-bf16-per-lane K operand.

<a id="sec-2-3-v-gm---lds---vgpr-ds_read_b64_tr_b16-transpose-double-buffer"></a>
### 2.3 V: GM -> LDS -> VGPR (`ds_read_b64_tr_b16` transpose, double buffer)

V uses the same cooperative GM->LDS DMA as K (lines 411-436, 773-775), double-buffered in the
**high** LDS region (the area Q was loaded into, now free): V buffer 0 at base `0x8200`, V buffer 1
at `0x8200 + 0x4400`. The DMA writes to `m0 = s66`/`s67`, advancing by `0x2200` per sub-load.

The difference from K is the **read**: V is consumed as the GEMM1 hardware `src0` (A operand), which
must be presented transposed. The kernel uses the CDNA4 hardware transpose LDS read.

V LDS->VGPR transpose read (`v10`, lines 908-939): 32 `ds_read_b64_tr_b16`, each returning 4
transposed bf16/lane, filling `v[192:255]` (overwriting the consumed K). The offsets show the
transpose-group structure (each group reads a 4-row x 4-dim fragment that the hardware presents as
MFMA-A columns):

```
ds_read_b64_tr_b16 v[192:193], v10               # K-step 0..3, low D
ds_read_b64_tr_b16 v[194:195], v10 offset:512
ds_read_b64_tr_b16 v[196:197], v10 offset:2176   # +0x880
...
ds_read_b64_tr_b16 v[224:225], v10 offset:8704   # second D-half, +0x2200
...
ds_read_b64_tr_b16 v[254:255], v10 offset:15808
```

<a id="sec-2-3-1-ds_read_b64_tr_b16-instruction-semantics"></a>
#### 2.3.1 `ds_read_b64_tr_b16` instruction semantics

CDNA4 ISA section **11.4 MFMA Transpose Load from LDS** (lines 7270-7280):

> These instructions allow the user to perform matrix transpose while transferring 16/8/6/4-bit data
> from LDS to VGPRs. The operation takes **two instructions** with different LDS addresses and VGPR
> destinations. Prior to executing, **EXEC must be all 1's**. The LDS address must be aligned to the
> data size, and any DS op reading >= 64-bit data must use an **even-aligned VGPR**.
>
> `DS_READ_B64_TR_B16` -- column-major matrix A or row-major matrix B load to 2 VGPRs. Element size
> 16b. The first instruction loads K = 0..3 and 8..11; the next loads K = 4..7 and 12..15. **Each lane
> (one VGPR) holds 4 consecutive M or N values.**

This is exactly what GEMM1 needs: V is stored in LDS in a DMA-friendly (row-major) layout, but the
MFMA A operand wants the KV (reduction) dimension along lanes. `ds_read_b64_tr_b16` performs that
lane/data transpose during the read, avoiding a software `ds_read` + `v_perm`/`ds_bpermute`
transpose sequence. The 32 reads (= 2 instructions per complete 16-bit transpose matrix x 16
fragments) deliver `v[192:255]` already in `mma1` A-operand order. The companion `mma1`
(`v_mfma ... v[96:111], v[192:195], v[32:35]`, line 1084) consumes `v[192:195]` as `src0` = V and
`v[32:35]` as `src1` = P.

The V transpose-read base `v10` decodes (lines 364-381; the V LDS line stride is `0x440 = 1088 B =
544 bf16`, i.e. 1024 B data + **64 B pad** — wider than K's 16 B to keep the 16-lane transpose
groups bank-conflict-free):

```
v10(lane) = 0x8200 + (lane & 3)*8           # 4-lane "low" subgroup -> the 4 transposed values
                   + ((lane >> 2) & 1)*0x80  # bit2
                   + ((lane >> 3) & 1)*0x440 # bit3 -> next V line
                   + ((lane >> 4) & 1)*32    # bit4
                   + (lane >> 5)*0x100       # lane_group (0/1)
v11 = v10 + 0x4400                           # V buffer 1
```

**Per-lane V transpose-read address** (byte base for `ds_read_b64_tr_b16 ... offset:0`; lanes group
into 16-lane transpose groups, each lane returning 4 transposed bf16):

| lane | lane&3 | bit2 | bit3 | bit4 | lane_group | `v10` byte base | transpose group |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 0 | 0 | `0x0000` | group 0, low-lane 0 |
| 1 | 1 | 0 | 0 | 0 | 0 | `0x0008` | group 0, low-lane 1 |
| 2 | 2 | 0 | 0 | 0 | 0 | `0x0010` | group 0, low-lane 2 |
| 3 | 3 | 0 | 0 | 0 | 0 | `0x0018` | group 0, low-lane 3 |
| 4 | 0 | 1 | 0 | 0 | 0 | `0x0080` | group 0, hi-subgroup |
| 8 | 0 | 0 | 1 | 0 | 0 | `0x0440` | group 0, next V line |
| 16 | 0 | 0 | 0 | 1 | 0 | `0x0020` | group 1 |
| 32 | 0 | 0 | 0 | 0 | 1 | `0x0100` | group 2 (lane_group 1) |
| 63 | 3 | 1 | 1 | 1 | 1 | `0x05F8` | group 3, last lane |

The 32 immediate offsets (`0, 512, 2176, 2688, ..., 8704, ..., 15808`) walk the four MFMA K-reduction
substeps and the two D-halves (`+0x2200`), so the hardware transpose delivers `v[192:255]` in
`mma1` A-operand order.

### 2.4 S and P: VGPR-only double-buffered score / probability

S never touches GM or LDS. GEMM0 produces it as 32 f32/lane in one of two ping-pong buffers:

```
S buffer 0 = v[32:63]      S buffer 1 = v[64:95]
```

The register semantics are established by the kernel's own causal-mask thresholds (lines 520-583):
for MFMA-C output lane `k` (`lane_group = lane/32`), the row index `r` (0..15 within a 16-wide bank)
maps to a key column `N`, and the query row `M = q_start + wave*32 + lane%32`. Lanes `x` and `x+32`
together hold all 64 key scores for one query row — which is why `v_permlane32_swap_b32` finishes the
row-max and row-sum reductions (lines 673, 859).

Two masks are applied (both `v_cmp_lt_i32` + `v_cndmask_b32`, with `v27 = 0xff800000 = -inf`):

```
causal  (520-583):  if (rel < thr_r)  S[r] = -inf      # future keys above the diagonal
seq/pad (588-651):  if (const_r < rel) keep else -inf  # keys past kv_seq_len
```

The exact register->element map comes from the mask index `v25 = wave*32 + (lane & 31) -
(lane>>5)*4` (lines 192-198) and the per-element thresholds `thr_r = (r//4)*8 + (r%4)` (lines
524-583). Decoding it: a lane owns **one query row** `M = q_block_start + wave*32 + (lane & 31)`; the
32 score f32 of that lane (strip 0 `v[32:47]` = keys 0..31, strip 1 `v[48:63]` = keys 32..63) plus
its `lane +/- 32` partner span the 64 keys.

**Per-lane S/P register map** (one warp; key column within the 64-key tile):

| lane | query row M (rel) | lane_group | keys held in `v[32:47]` (strip 0) | keys held in `v[48:63]` (strip 1) |
|---:|---:|---:|---|---|
| 0 | wave*32 + 0 | 0 | `{0,1,2,3, 8,9,10,11, 16,17,18,19, 24,25,26,27}` | the same set `+32` |
| 32 | wave*32 + 0 | 1 | `{4,5,6,7, 12,13,14,15, 20,21,22,23, 28,29,30,31}` | the same set `+32` |
| 1 | wave*32 + 1 | 0 | same key pattern as lane 0, for query row 1 | ... |
| 33 | wave*32 + 1 | 1 | same key pattern as lane 32, for query row 1 | ... |
| ... | ... | ... | ... | ... |
| 31 | wave*32 + 31 | 0 | same as lane 0, for query row 31 | ... |
| 63 | wave*32 + 31 | 1 | same as lane 32, for query row 31 | ... |

Within a strip, element `r` (0..15) maps to key column `lane_group*4 + (r//4)*8 + (r%4)`. Because
lanes `x` and `x+32` (same `lane%32`, hence same query row) hold complementary key halves, the
softmax row-max and row-sum reductions finish with `v_permlane32_swap_b32` (lines 673, 859).

After mask + max-subtract + exp2, `v_cvt_pk_bf16_f32` packs the 32 f32 probabilities of the **same**
score buffer in place into 16 bf16-packed VGPR (`v[32:47]` or `v[64:79]`), which becomes the GEMM1
`src1` = P.

<a id="sec-2-5-o-and-lse-vgpr---gm"></a>
### 2.5 O and LSE: VGPR -> GM

O is accumulated in `v[96:159]` (4 banks x 16 f32 = 64 f32/lane), rescaled every tile by the softmax
correction, and only written to GM in the epilogue. There is no LDS round-trip for O.

Epilogue (lines 2847-3020):

```
inv_l = (l == 0) ? 0 : 1/l                              # v_rcp_f32 (line 2897) + v_cmp_eq + v_cndmask
LSE   = m*s28 + log2(l) * (1/log2e)                     # v_log_f32 (2894) + v_fma_f32 (2900)
O    *= inv_l                                           # 32 v_pk_mul_f32 (per 16-f32 bank group)
O    -> bf16     (v_cvt_pk_bf16_f32)                    # 2912-...
reshape lanes:  v_permlane32_swap_b32 then v_permlane16_swap_b32   # 2917-2929, ...
store O via 8x buffer_store_dwordx4 to s[20:23] (R)     # 2930-3015
if s56 (lse flag): store LSE (1x buffer_store_dword) to s[24:27]   # 3018-3020
```

The `v_permlane*_swap_b32` reshape (CDNA4 ISA 11397 / 11416) rearranges the MFMA-C accumulator lane
layout into a contiguous head-dim layout so each `buffer_store_dwordx4` writes 128 contiguous output
bits:

- `v_permlane32_swap_b32`: "Rows 2 and 3 of the first operand are swapped with rows 0 and 1 of the
  second operand (one row is 16 lanes)" -- exchanges the upper/lower 32-lane halves.
- `v_permlane16_swap_b32`: "Odd rows of the first operand are swapped with even rows of the second
  operand (one row is 16 lanes)" -- interleaves 16-lane rows.

`buffer_store_dwordx4` (CDNA4 ISA 21853): "Store 128 bits of data from vector input registers into a
buffer surface." Eight of them write the 256x128 O tile (per wave: 32x128 = 8 stores of dwordx4).

The O accumulator has the same MFMA-C layout as S, but the 32-wide tiles run along the **head-dim**
(4 banks = 128/32) instead of the keys: a lane owns query row `M = wave*32 + (lane & 31)`, and within
each bank element `r` maps to head-dim `lane_group*4 + (r//4)*8 + (r%4) + bank*32`.

**Per-lane O register map** (`v[96:159]`, before the epilogue `permlane` reshape):

| lane | query row M (rel) | lane_group | bank 0 `v[96:111]` (dim) | bank 1 `v[112:127]` | bank 2 `v[128:143]` | bank 3 `v[144:159]` |
|---:|---:|---:|---|---|---|---|
| 0 | wave*32 + 0 | 0 | `{0..3,8..11,16..19,24..27}` | same `+32` | same `+64` | same `+96` |
| 32 | wave*32 + 0 | 1 | `{4..7,12..15,20..23,28..31}` | same `+32` | same `+64` | same `+96` |
| 1 | wave*32 + 1 | 0 | same dim pattern as lane 0, row 1 | ... | ... | ... |
| 63 | wave*32 + 31 | 1 | same dim pattern as lane 32, row 31 | ... | ... | ... |

The epilogue `v_permlane32_swap_b32` + `v_permlane16_swap_b32` shuffle (lines 2917-3013) converts
this strided per-lane layout into lane-contiguous 128-bit groups so each `buffer_store_dwordx4` writes
4 contiguous head-dim values to R; the store address `v22`/`v23` (lines 2874-2902) maps the lane to
its `(row, head-dim col)` in R via `r_Seqs`/`r_Hs`.

### 2.6 Summary: where each tensor lives + VGPR map

```
Q:  GM[B,S,H,D] -> 8x buffer_load_dwordx4 ... lds -> LDS@0x8200 -> 16x ds_read_b64 -> v[160:191] (resident, mma0 src1)
K:  GM[B,S,Hkv,D] -> 2x buffer_load ... lds -> LDS@0 (K0/K1) -> 16x ds_read_b128 -> v[192:255] (mma0 src0)
V:  GM[B,S,Hkv,D] -> 2x buffer_load ... lds -> LDS@0x8200 (V0/V1, reuses Q) -> 32x ds_read_b64_tr_b16 -> v[192:255] (mma1 src0)
S:  VGPR only.  v[32:63] / v[64:95] (32 f32 each, ping-pong). lane x & x+32 = one 64-key row.
P:  VGPR only.  bf16 in v[32:47] / v[64:79] (cast in place from S). mma1 src1.
O:  VGPR v[96:159] (64 f32) -> *1/l -> bf16 -> permlane reshape -> 8x buffer_store_dwordx4 -> GM[B,S,H,D]
```

VGPR pool (256 total):

| Range | Use |
|---|---|
| `v[0:31]` | thread id, addresses, scalars, temporaries |
| `v[32:63]` | score buffer 0 (S0, f32) / P0 (bf16 in v[32:47]) |
| `v[64:95]` | score buffer 1 (S1, f32) / P1 (bf16 in v[64:79]) |
| `v[96:159]` | O accumulator (4 banks x 16 f32) |
| `v[160:191]` | Q (resident, mma0 src1) |
| `v[192:255]` | K (mma0 src0) or V (mma1 src0), time-shared |

### 2.7 LDS region map and low-level address chains

The 160 KB LDS is partitioned into a low **K** region and a high **Q-then-V** region (Q and V share
because Q is consumed into VGPR before V streaming begins):

```
LDS (160 KB = 0x28000):
  [0x00000, 0x04100)  K buffer 0     (~16.6 KB; ds_read_b128 base via v8)
  [0x04100, 0x08200)  K buffer 1     (~16.6 KB; ds_read_b128 base via v9)
  [0x08200, ...     )  Q tile (256x128, ~64 KB) loaded once, read to v[160:191];
                       then REUSED as:
  [0x08200, 0x0C600)  V buffer 0     (~17.4 KB; ds_read_b64_tr_b16 base via v10)
  [0x0C600, 0x10A00)  V buffer 1     (~17.4 KB; ds_read_b64_tr_b16 base via v11)
```

DMA write bases (set into `m0`):

| Buffer | `m0` base | ISA |
|---|---|---|
| Q | `s63 = 0x8200 + 0x408*wave`, += `0x2040`/load | lines 233-238 |
| K0 / K1 | `s64 = 0x410*wave` / `s65 = 0x4100 + s64`, += `0x2080`/load | lines 411-413, 442 |
| V0 / V1 | `s66 = 0x8200 + 0x440*wave` / `s67 = 0x4400 + s66`, += `0x2200`/load | lines 437-439 |

Low-level read-address chains (built once in the prologue, lines 327-439, then reused):

```
v2, v3   = Q LDS read addresses (two N-strips)            -> ds_read_b64  -> v[160:191]
v8, v9   = K LDS read addresses (buffer 0 / 1)            -> ds_read_b128 -> v[192:255]
v10, v11 = V LDS read addresses (buffer 0 / 1)            -> ds_read_b64_tr_b16 -> v[192:255]
```

Each address is an explicit `v_lshrrev` / `v_and` / `v_mul_i32_i24` / `v_add_u32` decomposition of
`lane` into `(row, dim, N-group)` plus a per-wave and per-buffer base; e.g. the V read base `v10`
(lines 364-381) combines `lane&3` (4-bf16 chunk), `(lane>>2)&3` and `(lane>>4)` (row groups), and
`(lane>>3)` (D selector) with the `0x8200` V/Q region base. The companion store address `v22`/`v23`
(lines 2874-2902) maps each lane's MFMA-C output back to its `(row, head-dim col)` in R using the
output strides `r_Seqs`/`r_Hs` (`s79`/`s80`).

---

<a id="sec-appendix-notable-differences-from-the-flydsl-dualwave_swp-sibling"></a>
### Appendix: notable differences from the FlyDSL `dualwave_swp` sibling

Both kernels target the same shape (gfx950, D=128, 256x64, 8 waves, causal bf16). Key contrasts:

| Aspect | Hand-ASM `fwd_kernel_func` | FlyDSL `flash_attn_dualwave_swp` |
|---|---|---|
| Q residency | GM -> **LDS** -> VGPR (resident); LDS reused for V | GM -> VGPR directly (no LDS) |
| LDS | **160 KB** (full), occupancy 1 | 68 KB |
| Rescale | **eager** (every tile) | lazy (`ballot == exec` skip branch) |
| Softmax scale | folded into `exp` (`v_fma scale*S - scale*m`) | pre-applied to Q |
| Loop structure | **two alternating bodies** (ping-pong) + persistent <=2 q-blocks | explicit 8-cluster `scf.for`, 1 q-block |
| Wave scheduling | two wave-groups (0-3 / 4-7) via 1 wave-id split + **persistent asymmetric** `s_setprio` (A=0, B=1, set once) | two wave-groups + prologue stagger barrier + `s_setprio` brackets re-toggled per compute cluster |
| Scheduling control | manual instruction interleave | `sched_group_barrier` / `sched_barrier(0)` hints |
| LSE output | **yes** (`m*scale + ln l`) | no |
| O store | `permlane32/16_swap` reshape + `buffer_store_dwordx4` | `buffer_store_dwordx2` |
| V read | `ds_read_b64_tr_b16` (same) | `ds_read_b64_tr_b16` (same) |
| K read | `ds_read_b128` | `ds_read_b128` |
