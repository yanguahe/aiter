# FlyDSL Token-Scheduled WMMA Flash-Attention (gfx1250) Deep Introduction

This document deep-dives the FlyDSL kernel `fmha_fwd_kernel` for **gfx1250** (MI450, wave32,
WMMA + TDM). It is the FlyDSL flash-attention forward path built specifically for the CDNA-successor
gfx1250 ISA: it computes the same online-softmax attention math as the gfx950 dual-wave kernel, but
the hardware is fundamentally different, so almost every mechanism differs:

| Aspect | gfx950 (CDNA4, reference doc) | gfx1250 (MI450, this doc) |
|---|---|---|
| Wave size | 64 | **32** |
| Waves / workgroup | 8 (two wave-groups) | **4** |
| Matrix engine | MFMA `v_mfma_f32_32x32x16_bf16` | **WMMA `v_wmma_f32_16x16x32_bf16`** |
| GM->LDS copy | `buffer_load_dwordx4 ... lds` | **TDM `tensor_load_to_lds`** (Tensor Data Mover) |
| LDS->VGPR (V) | `ds_read_b64_tr_b16` | **`ds_load_tr16_b128`** |
| VGPR file | 256 (+AGPR namespace) | **1024, banked by `s_set_vgpr_msb`** |
| Scheduling | explicit 8-cluster SW pipeline + `sched_group_barrier` | **per-WMMA token schedule table** (160 rows) |
| Head dim | D=128 | **QK D=192, V D=128** |
| Pipeline | dual-wave time-multiplex | **2 rolled `scf.for` loops** (non-causal + causal) |

Every instruction count, register-residency claim, and layout/stride formula below is traced to the
kernel source and verified against the compiled assembly. Every instruction's *semantics* are taken
from the gfx1250 (MI400/MI450) hardware documents, not from memory.

## Authoritative Sources

| Role | File |
|---|---|
| Kernel entry / prologue / loops / epilogue | `fmha_gfx1250/ops/flydsl/kernels/fmha_gfx1250/fmha_kernel.py` (3216 lines) |
| Core loop (GEMM1/softmax/GEMM2 atoms + stage drivers) | `fmha_gfx1250/ops/flydsl/kernels/fmha_gfx1250/fmha_core_loop.py` (2545 lines) |
| Prologue helpers (Q load, TDM descriptors, LDS addr gen) | `fmha_gfx1250/ops/flydsl/kernels/fmha_gfx1250/fmha_prologue.py` (698 lines) |
| Per-WMMA token schedule tables | `fmha_gfx1250/ops/flydsl/kernels/fmha_gfx1250/fmha_schedule.py` (923 lines) |
| Compiled ISA (ground truth) | `fmha_gfx1250/mha_three_isa_dump/flydsl_192x128/fmha_fwd_kernel_0/21_final_isa.s` (9001 lines) |
| LLVM IR (intermediate) | `.../fmha_fwd_kernel_0/20_llvm_ir.ll` |
| gfx1250 ISA semantics (ground truth) | `mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt` (cited below as `SH#65:<line>`) |

The single-test that drives this build:

```
python fmha_gfx1250/test_mha_flydsl_varlen.py --causal true --return_lse true \
       -b 1 -nh 64 -sq 8192 -sk 8192 --random-value false --warmup 5 --repeat 20
```

## Benchmark / Trait Configuration

The ISA dump used for all counts is the **causal bf16, QK-D=192 / V-D=128** build (`flydsl_192x128`).

```
Arch              : gfx1250 (MI450), wavefront = 32
dtype             : bf16
causal            : true (this dump)
Launch shape      : grid = (batch, num_m_blocks, num_heads), block = (128,1,1) = 4 waves
Tile shape        : TILE_M = 128 (q rows), TILE_N = 128 (kv keys per tile)
                    QK_HDIM = 192 (Q.K^T contraction), V_HDIM = 128 (output channels)
WMMA              : v_wmma_f32_16x16x32_bf16   (W_M=16, W_N=16, W_K=32)
Softmax           : online, split PART0/PART1/PART2, token-scheduled across WMMAs
GQA               : num_kv_heads <= num_heads supported (head_index = wg / gqa)
LSE               : optional (return_lse)
```

<a id="sec-compiled-resource-usage"></a>
### Compiled resource usage (from `21_final_isa.s`, lines 8785-8994)

| Resource | Value | ISA source |
|---|---|---|
| VGPR | 796 | `.amdhsa_next_free_vgpr 796`, `.vgpr_count: 796` |
| AGPR | 0 | `.agpr_count: 0` / `num_agpr = 0` (gfx1250 has no AGPRs; the 1024-VGPR file replaces them, `SH#65:2253`) |
| SGPR | 107 (`numbered_sgpr` 105 + VCC) | `.sgpr_count: 107` |
| VGPR spill | 0 | `.vgpr_spill_count: 0` |
| SGPR spill | 43 | `.sgpr_spill_count: 43` |
| LDS (group segment) | 233472 B (228 KB) | `.amdhsa_group_segment_fixed_size 233472` |
| Private / scratch | 0 | `.private_segment_fixed_size: 0` |
| Max flat workgroup | 128 | `.max_flat_workgroup_size: 128` |
| Wavefront size | 32 | `.amdhsa_wavefront_size32 1` |
| kernarg size | 520 B | `.amdhsa_kernarg_size 520` |

The 796-VGPR footprint (near the 1024 hardware maximum) is the defining constraint of this kernel:
the whole design — the compact no-padding SP layout, the `s_set_vgpr_msb` bank pinning, and the
double-buffered partial-softmax `iter_args` — exists to fit the QK-192 / V-128 working set plus the
online-softmax scratch into one wave's register file without spilling VGPRs (spill = 0). LDS at
228 KB needs the 256 KB (4x64 KB) LDS partition of the 384 KB unified WGP cache, of which up to
320 KB may be LDS (`SH#65:1693-1694`, `SH#65:11672-11674`); it caps occupancy near 1 workgroup/WGP.

---

## 1. Kernel overview and token-scheduled pipeline

This kernel implements **(Grouped-Query) Flash-Attention forward** with online softmax for bf16 on
gfx1250. Each workgroup (**4 waves = 128 threads**) owns one `(batch, q-block, head)` of **TILE_M =
128** query rows (32 rows per wave) and streams KV tiles of **TILE_N = 128** keys each. The body is:

- a **prologue** (`fmha_kernel.py` SECTION 2, lines 1082-1308): primes KV tile 0 (TDM K tile 0,
  Q load + scale, first QK GEMM producing the score tile, causal + OOB mask, softmax PART0/PART1 and
  the first half of PART2), prefetches K tile 1, and preloads the K SU-0 fragments for loop entry;
- a **main KV loop**, split into **two rolled `scf.for` loops** (SECTION 3a non-causal tiles, 3b
  causal tiles, lines 1886-2444). Each iteration advances **one KV tile** and runs the full
  `_core_loop`: 4 GEMM1 (QK) stages + softmax + 4 GEMM2 (PV) stages;
- an **epilogue** (`_ep_finish`, SECTION 4, lines 2446-2823): drains the last tile's softmax PART2
  second half, runs the final PV GEMM (`_pv_pure_su`), normalizes O by `1/row_sum`, optionally emits
  LSE, and stores O (VGPR -> LDS -> global).

In the ISA these map to exactly **two backward-branch loops**: `.LBB0_5` (lines 2010-3596, the first
rolled loop) and `.LBB0_12` (lines 3760-5502, the second rolled loop); everything before `.LBB0_5`
is the prologue, everything after `.LBB0_12` (through `s_endpgm` at line 8782) is the epilogue and O
store.

### 1.0 The WMMA operand convention (no swap)

Unlike the CDNA MFMA swap-A/B convention used by the gfx950 kernel, gfx1250 WMMA has **no swap
modifier**: the ISA operand order is fixed `SRC0 = A`, `SRC1 = B`, `SRC2 = C`, `VDST = D`, computing
`D = A*B + C` (`SH#65:11043-11057`). Each WMMA is `v_wmma_f32_16x16x32_bf16 D, A, B, C` with A a
16x32 bf16 matrix, B a 32x16 bf16 matrix, C/D a 16x16 f32 matrix (`SH#65:10058-10061`). Each bf16
operand occupies **8 VGPRs** (a `v16bf16` in FlyDSL terms — 16 bf16/lane packed 2-per-DWORD), each
f32 accumulator **8 VGPRs** (`SH#65:10891-10936`, `SH#65:10676-10717`).

The two GEMMs (`_emit_qk_wmma` / `_emit_pv_wmma`, `fmha_core_loop.py:1470,1497`):

| Matmul | Math form | HW `src0` (A) | HW `src1` (B) | Accumulator (C/D) |
|---|---|---|---|---|
| Q*K (`_emit_qk_wmma`) | `S = Q . K^T` | `kv_tiles[k_msb][k_frag]` (**K**) | `q_tiles[q_msb][..]` (**Q**) | `sp_tiles` (S) |
| P*V (`_emit_pv_wmma`) | `O += P . V` | `v_tiles[v_msb][n]` (**V**) | `p_tiles[sp_msb]` (**P**) | `o_tiles` (O) |

ISA evidence (`21_final_isa.s`), loop body `.LBB0_12`:
- QK: `v_wmma_f32_16x16x32_bf16 v[90:97], v[66:73], v[66:73], 0` (line 3867) — first QK WMMA, `C=0`
  (fresh accumulator, `_atom_wmma_init`), src0=K frag, src1=Q frag. The K-reduction chain then feeds
  `C` back: `v_wmma_f32_16x16x32_bf16 v[90:97], v[58:65], v[50:57], v[90:97]` (line 3898).
- PV WMMAs always accumulate into the persistent O (`_atom_wmma_accum`, no init) since O lives across
  the entire loop.

The K-on-A / Q-on-B choice gives the score tile a **one-query-row-per-lane-group, keys-across-VGPRs**
layout that the online softmax (max/sum over keys) consumes; V-on-A / P-on-B gives O the
query-row-on-lanes, head-dim-on-registers layout used for the O accumulation.

<a id="sec-1-1-buffering"></a>
### 1.1 Buffering: where Q / K / V / S / P / O live (GM / LDS / VGPR)

| Tensor | Global memory | LDS | VGPR (per lane) |
|---|---|---|---|
| Q | full Q tile (128 rows x 192) | **none** (Q never goes through LDS) | `q_frags[4 bank][3 frag]` v16bf16, pre-scaled by `scalar*log2e`; resident, reused by every QK WMMA. `_phase4_q_load_flydsl` (`fmha_prologue.py:182`) |
| K | streamed tile by tile (128 rows x 192) | **2 ping-pong buffers** `K_a`,`K_b`, each `CNT_SU(4) * LDS_K_SU_P_SIZE(0x3200)` = 51200 B; row stride `K_ROW_BYTES = 400` (192 bf16 data + 8 pad) | `kv_tiles[4 msb][N_WMMA_K_TILES=3]` v16bf16 (paired from `ds_load_b128`) -> QK WMMA src0 |
| V | streamed tile by tile (128 rows x 128) | **2 ping-pong buffers** `V_a`,`V_b`, each `CNT_SU(4) * LDS_V_SU_P_SIZE(0x2400)` = 36864 B; row stride `V_ROW_BYTES = 288` (128 bf16 data + 32 pad) | `v_tiles[N_V_MSB=2][N_PV_WMMA_N=4]` v16bf16, read via HW transpose `ds_load_tr16_b128` -> PV WMMA src0 |
| S (scores `Q.K^T`) | -- | none (register-only) | per SU, `sp_tiles[4 msb][1]` v8f32 = 8 f32/lane per MSB; 4 SUs -> the compact 32-f32/lane/MSB `sp_pairs` (16 v2f32) |
| P (probabilities) | -- | none (register-only) | `p_bf16[4 msb][16]` v2bf16 -> packed to `p_tiles[CNT_SU][2]` v16bf16 = PV WMMA src1 |
| O | written at the end | **reuses V_a LDS for the D staging store** | `o_tiles[4 d_msb][N_PV_WMMA_N=4]` v8f32 = 32 f32/lane per d_msb, resident across the whole loop |

K and V each use a **2-deep LDS ping-pong** managed by four `SmemAllocator` regions
(`fmha_kernel.py:158-169`): `smem_k_a`, `smem_k_b`, `smem_v_a`, `smem_v_b`. K_a/K_b/V_a are padded to
the 64 KB `LDS_SEGMENT` boundary so a TDM tile never straddles a 64 KB segment; V_b is last and needs
no padding. The D (output) store reuses the V_a region because PV is fully done before D is staged
(`_ep_finish`, `fmha_kernel.py:2768`).

The ping-pong bases are carried as four i32 `iter_args` (`fmha_kernel.py:1945-1948`,
`k_cur/v_cur/k_next/v_next`) and swapped every iteration (`pp_swapped`, line 2142): each iteration
computes QK/PV out of the *cur* buffers while TDM fills the *next* buffers, then the yield rotates
`next -> cur`.

<a id="sec-vgpr-bank-msb"></a>
#### The `s_set_vgpr_msb` VGPR-bank mechanism (why 1886 of them)

gfx1250 waves may allocate up to 1024 VGPRs, but instructions carry only an 8-bit VGPR field. Two
"MSB" bits per operand (DST/SRC2/SRC1/SRC0) extend the address, set by `s_set_vgpr_msb`
(`SH#65:2250-2291`). The immediate is `{dst[1:0],src2[1:0],src1[1:0],src0[1:0]}`, and each 2-bit code
selects a 256-VGPR bank: `00->v0..255, 01->v256..511, 10->v512..767, 11->v768..1023`
(`SH#65:2263-2286`). So the "全-bankN" patterns `0x00/0x55/0xAA/0xFF` put **all four operands** in
bank 0/1/2/3 respectively.

This is why the kernel is organized around **4 MSB banks** (`NUM_MSB = 4`): each of the four MSB banks
holds one logical slice (one score/probability MSB, one O d_msb, etc.), and the FlyDSL
`set_vgpr_bank(value, bank)` helper (`fmha_core_loop.py:247`) tags every SSA value with a bank hint
so the register allocator places it at `bank*256 + offset`. When a computation keeps all its operands
in the same bank, no `s_set_vgpr_msb` switch is needed between instructions — this is the driving
force behind the "same-MSB grouping" seen throughout the schedule tables (§1.5). The ISA still emits
**1886** `s_set_vgpr_msb` (one before nearly every cross-bank VALU); they are the gfx1250 replacement
for MI300's ACCUM_OFFSET/ACCUM VGPRs (`SH#65:2253`) and `s_set_vgpr_msb` incurs no extra delay
(`SH#65:2258-2259`). Because the immediate encodes each operand's bank independently, most emitted
values are *not* the pure `0x00/0x55/0xAA/0xFF` patterns but mixed per-operand selectors (e.g.
`0x5142` = src0 bank2, src1 bank0, src2 bank1, dst bank1); the histogram of the 1886 immediates has a
long tail of ~300 distinct mixed values, confirming fine-grained per-operand bank selection rather
than a small fixed set.

### 1.2 Online softmax split into PART0 / PART1 / PART2

The softmax is factored into three parts so its VALU / transcendental work can be **token-interleaved
between WMMAs** (§1.5) and **pipelined across tiles** (the current tile's score reduction overlaps the
previous tile's probability finish). The parts (all in `fmha_core_loop.py`):

- **PART0** (`_build_softmax_part0_ops`, line 1157, **22 ops/MSB**): the per-tile row-max. A tree of
  `v_max3_num_f32` over the 32 valid score f32/lane in one MSB (4 groups of 8 from the 4 SUs), then a
  cross-lane exchange (`v_permlanex16_b32`, lanes 0-15 <-> 16-31, `SH#65:9534-9536`) and a final
  `max3` to get `local_max`. `v_max3_num_f32` is the NaN-suppressing IEEE maximumNumber
  (`SH#65:14981-14985`) — the softmax-safe max.
- **PART1** (`_build_softmax_part1_ops`, line 1287, **8 ops total**): the cross-MSB merge. `max3`
  merges `local_max[0..3]` pairwise across banks, then computes `delta[msb] = pre_max_log2e_scl -
  local_max*log2e_scl` (the max-shift for online rescale) via `v_fma_f32` with negated src0.
- **PART2** (`_build_softmax_part2_ops`, line 866, **setup(8) + pkfma(16) + pair_exp(8..) + cvt + sum
  tree**): the probability computation and rescale. Setup computes `exp_delta = exp2(delta)` (the O
  rescale factor) and materializes the scale into a per-bank VGPR; `v_pk_fma_f32` applies the
  max-shift to score pairs; `v_exp_f32` (base-2, on the transcendental pipe, `SH#65:5540`) produces
  the probabilities; `v_cvt_pk_bf16_f32` packs them to bf16 for the PV WMMA; and a `v_pk_add_f32` tree
  reduces the row sum. The 1/sqrt(D) temperature and `log2e` are **pre-applied to Q** (`scale =
  log2e_val * scalar_f`, `fmha_kernel.py:1068`) so `exp2` replaces `exp`.

**Cross-tile pipeline.** In `_core_loop`, PART2 runs on `sp_pairs_prev` (the *previous* tile's scores,
carried in `iter_args` as `partial_sp_pairs`) *during GEMM1 of the current tile*, while PART0/PART1
run on the *current* tile's fresh scores. `PART2_SPLIT = 32` (`fmha_core_loop.py:146`) divides PART2:
its first half (setup + pkfma + 8 pair_exp) is emitted during the previous stage/prologue and carried
as `iter_args`; its second half finishes at the start of the next `_core_loop`. So every iteration's
GEMM1 does PART2-finish for tile `i-1` while producing scores for tile `i`, and its GEMM2 does the PV
for tile `i` using the P tile that PART2 just finished.

### 1.3 Four waves, no wave-group staggering

Unlike the gfx950 kernel's two time-multiplexed wave-groups, the gfx1250 kernel runs **4 waves that
all execute the same pipeline in lockstep**, synchronized by the split workgroup barrier
`s_barrier_signal -1` / `s_barrier_wait -1` (the `-1` selects the ordinary workgroup barrier,
`SH#65:5119-5122`). The four waves cooperate on the **GM->LDS TDM loads** (each wave loads 8 of the
32 rows per SU, `fmha_kernel.py:402-404`) and each wave then owns a disjoint **32-row M-slice** of the
128-row q-block for the compute (wave `w` owns q-rows `[32w, 32w+32)`). Cross-wave synchronization is
needed only at LDS producer/consumer boundaries (after TDM fill, before LDS read) and before the
D-store LDS reuse; there is no per-wave phase offset.

Parallelism between the WMMA (XDL) pipe and the VALU/transcendental pipes is instead exploited
*within* each wave via the token schedule (§1.5) and the hardware's WMMA co-execution: 16-bit WMMA
"may co-execute (runs on XDL macc)" with other VALU but "not with transcendental ops"
(`SH#65:10434-10436`), and a wave may issue "up to 5 WMMA's before the hardware stalls"
(`SH#65:6000`). `_setreg(2074, 2)` at kernel entry sets `WAVE_SCHED_MODE = 2` (`fmha_kernel.py:743`,
ISA line 8) to control the arbiter's WMMA back-to-back behavior.

### 1.4 Refined pseudocode

Variable names match the FlyDSL source. `NUM_MSB = 4`, `CNT_SU = 4`, `TILE_N = 128`.

```
Prologue (fmha_kernel.py 743-1308):
  [P0] _setreg(WAVE_SCHED_MODE=2); s_nop
       tid/lane/wave; XCD software remap of (batch,m,head)  (1093-1145)
       load cu_seqlens -> q_start_tok/k_start_tok, actual_q_len/actual_kv_len   (THD)
       guards: _wg_valid (m_start < actual_q_len); seqlen_k==0 zero-fill+skip
  [P1] q_rsrc = create_buffer_resource(Q, num_records=q_end_tok*stride)
       q_frags = _phase4_q_load_flydsl()          # 24 buffer_load_b128 (6/bank x 4 banks)
       bridge q_frags_raw[4 bank][3] -> q_frags[q_msb in {0,2}][3]  # pre-scaled by scale*log2e
       head_index = wg / gqa ; k_offset/v_offset ; build kv_lds_addrs_a/b
  [P2] TDM K(tile 0) -> K_a   (_tdm_load_k_only: 4 tensor_load_to_lds, s_wait_tensorcnt 0, barrier)
  [P3] for su in 0..3:  kv_tiles_su = _load_k_su_from_lds(K_a)   # 24 ds_load_b128
                        sp = _qk_pure_su(q_frags, kv_tiles_su)   # 24 QK WMMA/su -> 96 total
  [P4] TDM V(tile 0) -> V_a   (_tdm_load_v_only: 4 tensor_load_to_lds)
  [P5] if CAUSAL: _apply_causal_mask(all_su_sp, n_start = -(sk-sq))
       _apply_kv_oob_mask(all_su_sp, actual_kv_len)
  [P6] sp_pairs = _sp_tiles_to_sp_pairs(all_su_sp)               # compact 16 v2f32/MSB
       _softmax_part01_only(sp_pairs)                            # PART0 + PART1
       PART2 first half ops[0..PART2_SPLIT-1]                    # setup+pkfma+8 exp
  [P7] num_tiles = CAUSAL ? min(bx+1 + sk_sq_tiles, kv_avail) : kv_avail
       TDM K(tile 1) -> K_b (prefetch)
       kv_tiles_init = _load_initial_kv_tiles(K_b, su=0)
       carry: o=0, old_max/local_max/delta/row_sums, partial_sp (lo+hi), exp_delta,
              pingpong bases (K_b,V_a,K_a,V_b)

Main loop = two rolled scf.for over tile_idx (fmha_kernel.py 1886-2444):
  3a  for tile_idx in [1, first_causal_tile):        # ISA .LBB0_5
        _core_loop(..., causal_n_start=None)
  3b  for tile_idx in [first_causal_tile, num_tiles-1):   # ISA .LBB0_12
        _core_loop(..., causal_n_start = tile_n_start - (sk-sq))

  _core_loop(tile i)   (fmha_core_loop.py _core_loop / fmha_kernel.py inner _core_loop):
     s_wait_dscnt(LDS_INST_COUNT/2)
     # ---- GEMM1 (QK): 4 stages x 24 WMMA = 96, interleaved via GEMM1_SCHEDULE ----
     build PART2(prev tile) ops ; pre-run PART2 setup 0..6
     for stage in 0..3:
        _cl_su_v3_stage(stage): 24 QK WMMA
           + token schedule GEMM1_SCHEDULE[stage] between WMMAs:
             K ds_load_b128 (or V ds_load_tr16 in stage 3), PART2(prev) exp/pkfma,
             O-rescale (o *= exp_delta), TDM K(i+1)/V(endtile) in stages 0-1
     flush remaining PART2(prev) 2nd-half ops -> p_bf16(prev) ready
     if CAUSAL: _apply_causal_mask(cur scores) ; _apply_kv_oob_mask
     sp_pairs_cur = _sp_tiles_to_sp_pairs(cur scores)
     build PART0+PART1+PART2(cur first half) ops
     p_tiles = _build_p_tiles_from_softmax(prev)          # v16bf16 PV src1
     # ---- GEMM2 (PV): 4 stages x 16 WMMA = 64, interleaved via GEMM2_SCHEDULE ----
     for stage in 0..3:
        _cl_su_v3_stage_gemm2(stage): 16 PV WMMA (o += V * P)
           + token schedule GEMM2_SCHEDULE[stage]:
             V ds_load_tr16 (or K ds_load in stage 3), PART0/PART1/PART2(cur),
             TDM V(i)->V_next in stages 0-1
     yield o, old_max, row_sums, kv_tiles(next), local_max, delta, sp(cur),
           pingpong swap, partial_sp(cur lo+hi), exp_delta(cur)

Epilogue (_ep_finish, fmha_kernel.py 2446-2823):
  s_wait_idle ; barrier
  PART2 2nd half of last tile ; p_tiles = _build_p_tiles_from_softmax(last)
  O rescale (o *= exp_delta)
  PV_pure over 4 SUs from V_cur LDS (_load_v_two_sus_from_lds + _pv_pure_su)  # 64 WMMA
  row_sum = cross-MSB + permlanex16 reduce
  if RETURN_LSE: LSE = local_max*scalar + log(row_sum)*ln2   (v_log_f32)
  O_bf16 = trunc(o * rcp(row_sum))                            # v_cvt_pk_bf16_f32
  barrier ; ds_store_b128 O -> V_a LDS ; TDM store LDS -> global (or global_store_b128)
```

### 1.5 Per-operation main instructions and counts

Counts are **per warp** (each VALU/LDS instruction is executed by all 32 lanes of the wave32 in
lock-step; each WMMA is a wave-collective instruction). The fixed sizes come from the source constants
(`fmha_core_loop.py:69-188`, `fmha_prologue.py:31-58`):

```
Launch / tile:
  WAVE_SIZE=32  NUM_WAVES=4  BLOCK_SIZE=128  NUM_MSB=4
  TILE_M=128 (q rows, 32/wave)  TILE_N=128 (kv keys)  CNT_SU=4 (32-row sub-units)
  QK_HDIM=192  V_HDIM=128

WMMA tile:
  W_M=16  W_N=16  W_K=32

GEMM expansion:
  GEMM1 (QK, S=Q.K^T): per SU stage, GEMM_INST_COUNT = 24 WMMA
                       = 2 (msb_idx) x (QK_HDIM/W_K = 192/32 = 6) x 2 (sp_msb pair) x 1n x 1m
                       x 4 SU stages = 96 WMMA / tile
  GEMM2 (PV, O+=P.V) : per SU stage, PV_GEMM_INST_COUNT = 16 WMMA = 4 (d_msb) x 4 (N_PV_WMMA_N)
                       x 4 SU stages = 64 WMMA / tile

LDS geometry:
  K_ROW_BYTES=400 (192 bf16 + 8 pad) ; LDS_K_SU_P_SIZE=0x3200 ; N_LDS_PER_MSB=6
  V_ROW_BYTES=288 (128 bf16 + 32 pad); LDS_V_SU_P_SIZE=0x2400 ; N_LDS_V_PER_MSB=4
```

| # | Operation (source) | Main ISA instruction(s) | Count / call | Tiling derivation |
|---|---|---|---|---|
| 1 | `tensor_load_to_lds` (K/V TDM) (`_tdm_load_k/v_only`, `fmha_kernel.py:343,417`) | `tensor_load_to_lds` | **4** | `CNT_SU = 4` SUs per tile (each SU = 32 rows) |
| 2 | `_phase4_q_load_flydsl` (`fmha_prologue.py:182`) | `buffer_load_b128` | **24** | 6 loads/bank (3 frags x 2) x 4 banks; QK_HDIM=192 |
| 3 | QK WMMA (`_emit_qk_wmma`, 1470) | `v_wmma_f32_16x16x32_bf16` | **24/stage** | `E = 2*6*2 = 24` (msb_idx x K/W_K x sp_msb) |
| 4 | `_load_k_su_from_lds` (`fmha_core_loop.py:2027`) | `ds_load_b128` | **24** | `NUM_MSB(4) * N_LDS_PER_MSB(6)` |
| 5 | `_load_v_*` / `_build_lds_v_schedule` (792) | `ds_load_tr16_b128` (HW transpose) | **16/su-pair** | `NUM_MSB(4) * N_LDS_V_PER_MSB(4)` |
| 6 | PV WMMA (`_emit_pv_wmma`, 1497) | `v_wmma_f32_16x16x32_bf16` | **16/stage** | `NUM_MSB(4) * N_PV_WMMA_N(4)` |
| 7 | PART0 row-max (`_build_softmax_part0_ops`, 1157) | `v_max3_num_f32` + `v_permlanex16_b32` | **~20 + 1 /MSB** | tree over 32 f32 + cross-lane 0-15<->16-31 |
| 8 | PART1 cross-MSB (`_build_softmax_part1_ops`, 1287) | `v_max3_num_f32` + `v_fma_f32` | **2 + 2 mov + 4 fma** | 4-bank merge + delta |
| 9 | PART2 pkfma (`_build_softmax_part2_ops`, 866) | `v_pk_fma_f32` | **16/MSB** | 16 v2f32 score pairs / lane |
| 10 | PART2 exp (`_atom_exp_f32`, 518) | `v_exp_f32` | **32/MSB (16 pairs x2)** | 32 score f32/lane |
| 11 | PART2 cast (`_atom_cvt_pk_bf16_f32`, 647) | `v_cvt_pk_bf16_f32` | **16/MSB** | 32 f32 -> 16 v2bf16 |
| 12 | PART2 sum tree (`_atom_pk_add_f32`, 627) | `v_pk_add_f32` | **~15/MSB** | reduce 16 v2f32 |
| 13 | causal mask (`_apply_causal_mask`, `fmha_kernel.py:834`) | `v_cmp_gt_i32_e64` + `v_cndmask_b32_e64` | **32 + 32 /call** | 4 SU x 4 MSB x 8 elems compares |
| 14 | KV OOB mask (`_apply_kv_oob_mask`, 858) | `v_cmp_lt_i32_e64` + `v_cndmask_b32_e64` | **32 + 32 /call** | same shape |
| 15 | O rescale (`fmha_kernel.py:1541`) | `v_pk_mul_f32` | **16/tile** | 4 d_msb x 4 n x (8 f32 packed /2) |
| 16 | O normalize (`_ep_finish`, 2737) | `v_cvt_pk_bf16_f32` + `rcp` | **16 + 4** | 4 d_msb x 4 n cvt; 1 rcp/MSB |
| 17 | O store (`_ep_finish`, 2777) | `ds_store_b128` then TDM / `global_store_b128` | **16 / 16** | 4 d_msb x 4 n |

#### Whole-kernel ISA verification

Counts grepped from `21_final_isa.s` with `;` comments stripped. Region split: prologue = lines
1-3759 (includes the first rolled loop `.LBB0_5` at 2010-3596), main-loop body = 3760-5502 (one rolled
iteration of `.LBB0_12`), epilogue+store = 5503-9001.

| Instruction | Total | Prologue (of which `.LBB0_5`) | Loop body (`.LBB0_12`) | Epilogue+store | Derivation check |
|---|---:|---:|---:|---:|---|
| `v_wmma_f32_16x16x32_bf16` | **704** | 256 (160) | 160 | 288 | one `_core_loop` = 96 QK + 64 PV = 160/tile. Prologue-proper = 96 (tile-0 QK only, no PV); first rolled loop `.LBB0_5` = 160; second `.LBB0_12` = 160; epilogue = 288 = both compiled branches: N>=2 endtile `_core_loop` (160) + `_ep_finish` PV_pure (64) + N==1 `_ep_finish` PV_pure (64). 96+160+160+288 = 704 |
| `ds_load_tr16_b128` | **320** | 64 (64) | 64 | 192 | V transpose reads, 64/tile (16 x 4 su-groups) |
| `ds_load_b128` | **384** | 216 (96) | 96 | 72 | K reads, 96/tile (24 x 4 su) |
| `tensor_load_to_lds` | **32** | 20 (8) | 8 | 4 | TDM K+V, 8/tile (4 K + 4 V) |
| `buffer_load_b128` | **24** | 24 | 0 | 0 | Q load, prologue only |
| `global_store_b128` | **16** | 0 | 0 | 16 | O store, D_MSB(4) x N(4) |
| `ds_store_b128` | **16** | 0 | 0 | 16 | O -> LDS staging |
| `v_exp_f32(_e32)` | **616** | 164 (130) | 130 | 322 | 32 exp/MSB x 4 = 128/tile + rescale exp |
| `v_cvt_pk_bf16_f32` | **372** | — | — | — | 16/MSB cast P + O normalize |
| `v_max3_num_f32` | **312** | — | — | — | PART0/PART1 row-max reductions |
| `v_pk_fma_f32` | **256** | — | — | — | PART2 max-shift, 16/MSB |
| `v_pk_mul_f32` | **392** | — | — | — | O rescale + normalize |
| `v_pk_add_f32` | **325** | — | — | — | PART2 sum tree |
| `v_cndmask_b32_e64` | **384** | 128 (0) | 128 | 128 | causal + KV OOB mask (32+32/call) |
| `v_permlanex16_b32` | **20** | — | — | — | cross-lane max/sum reductions |
| `s_set_vgpr_msb` | **1886** | 703 (496) | 485 | 698 | per-operand VGPR bank selection |
| `s_barrier_signal` / `s_barrier_wait` | **23** each | 17 | 2 | 4 | LDS producer/consumer + TDM sync |
| `s_wait_tensorcnt` | **10** | — | — | — | after each TDM group |

The exactly-determined WMMA/LDS/TDM classes match the tiling derivation to the instruction: 704 WMMA =
160/tile x (prologue-QK + loop bodies) + epilogue drain; `tensor_load_to_lds` = 8/tile (4 K SU + 4 V
SU); `ds_load_b128`/`ds_load_tr16_b128` follow `NUM_MSB * N_LDS_(V_)PER_MSB` per SU. The VALU classes
vary by region because the softmax is pipelined (PART2 split across tiles) and the causal mask is
present only in the causal loop `.LBB0_12` and the epilogue (prologue mask is the first-tile diagonal).

### 1.6 Scheduling: the per-WMMA token table

Where the gfx950 kernel used `sched_group_barrier` IGroupLP hints, the gfx1250 kernel uses an
**explicit per-WMMA token schedule** (`fmha_schedule.py`): two flat tables, `GEMM1_SCHEDULE` (96 rows
= 4 stages x 24 WMMA) and `GEMM2_SCHEDULE` (64 rows = 4 stages x 16 WMMA). Row `i` is the list of
non-WMMA instruction tokens emitted **between WMMA `i` and WMMA `i+1`**. The tokens
(`fmha_schedule.py:36-54`):

| Token | Meaning | Cost |
|---|---|---|
| `P0_M0..3` | PART0 op for MSB 0-3 (max3/permlane/mul/fma) | 1 cy |
| `P1` | PART1 cross-MSB merge | 1 cy |
| `P2_M0..3` | PART2 cheap op for MSB 0-3 (setup/pkfma/pkadd/cvt/sum) | 1 cy |
| `EXP_M0..3` | PART2 `pair_exp` for MSB 0-3 (transcendental) | 3 cy |
| `K_M0..3` | one `ds_load_b128` (K) for MSB 0-3 | 1 cy |
| `V_M0..3` | one `ds_load_tr16_b128` (V) for MSB 0-3 | 1 cy |
| `O_RESC0` | one O-rescale sub-op (`v_pk_mul_f32`) | 1 cy |
| `TDM` | one `tensor_load_to_lds` | 4 cy |

The tables are hand-tuned for two goals (`fmha_schedule.py:241-263`): **cycle balance** (~7 cy/WMMA so
the VALU/LDS work fully hides behind the WMMA issue) and **same-MSB grouping** (consecutive tokens
share an MSB bank to minimize `s_set_vgpr_msb` switches, §1.1). For example `GEMM1` stage-0 row 2 is
`[K_M0, K_M0, K_M0, P2_M0, P2_M0]` — three K loads then two PART2 ops, all bank 0, zero bank switches.
Row 0 of a TDM stage is `[TDM, TDM]` — the two 4-cycle TDM loads issued while the first WMMA runs.

Each `_atom_*` (`fmha_core_loop.py:387-679`) wraps its one instruction in `rocdl.sched_barrier(0)`
fences (the hard `SCHED_BARRIER` pseudo, no real instruction) so the LLVM scheduler cannot move
instructions across token boundaries — the token order in the table *is* the emitted order. The
strict softmax dependency `PART0 -> PART1 -> PART2` is enforced by placing P0 tokens in GEMM2 stages
0-1, P1 in stage 1's post-load region, and P2/EXP in stages 2-3 (`fmha_schedule.py:629-640`).

The `ALU_PER_STAGE = [40,52,56,168, 120,120,132,132]` budget (`fmha_core_loop.py:168`) sizes how many
softmax ops each of the 8 softmax-stages (4 GEMM1 + 4 GEMM2) can absorb; the schedule tables are the
concrete realization of these budgets.

---

### 1.7 KV-tile distribution and where the causal mask runs

#### 1.7.1 How many KV tiles a q-block owns

The q-block index is `bx` (grid.y after the XCD remap, `fmha_kernel.py:783`), owning q-rows
`[bx*128, bx*128+128)`. The number of KV tiles it iterates is `num_tiles` (`fmha_kernel.py:1181-1208`):

```1188:1207:fmha_gfx1250/ops/flydsl/kernels/fmha_gfx1250/fmha_kernel.py
            if const_expr(IS_CAUSAL):
                _sk_sq_diff = actual_kv_len - actual_q_len
                _sk_sq_tiles = ceil(_sk_sq_diff / TILE_N)
                _causal_tiles = (bx + 1) + _sk_sq_tiles
                num_tiles = min(_causal_tiles, _kv_tiles_avail)
            else:
                num_tiles = _kv_tiles_avail            # ceil(actual_kv_len / 128)
```

For self-attention (`actual_q_len == actual_kv_len`, so `_sk_sq_diff = 0`, `_sk_sq_tiles = 0`),
`num_tiles = min(bx + 1, ceil(S/128))` — the classic causal triangle: q-block `bx` attends to KV
tiles `0..bx`. The pipeline structure is **prologue (tile 0) + main loop (tiles 1..num_tiles-2) +
epilogue (tile num_tiles-1)**:

| Stage | Tiles processed | Source |
|---|---|---|
| Prologue | tile `0` | `_qk_pure_su` at `fmha_kernel.py:1118` |
| Main loop 3a (non-causal) | tiles `[1, first_causal_tile)` | `scf.for_` at 1886 |
| Main loop 3b (causal) | tiles `[first_causal_tile, num_tiles-1)` | `scf.for_` at 2170 |
| Epilogue | tile `num_tiles-1` | `_ep_finish` at 2572 |

So for `num_tiles = 1` (q-block 0 self-attention, `bx=0`) both loops run zero iterations: the prologue
does tile 0 and the epilogue finishes it — no main-loop body executes. This is why the ISA loops
`.LBB0_5` / `.LBB0_12` are entered conditionally (`s_cbranch` guards at ISA lines 217, 1910).

#### 1.7.2 The two-loop causal split

The main loop is split at `_first_causal_tile` (`fmha_kernel.py:1865-1881`): tiles strictly *below*
the diagonal need no causal mask, tiles *crossing* the diagonal do. `_first_causal_tile = clamp(bx +
floor(_causal_offset / TILE_N), 1, num_tiles-1)` where `_causal_offset = actual_kv_len -
actual_q_len`. Loop **3a** (ISA `.LBB0_5`) runs the below-diagonal tiles with `causal_n_start=None`
(no mask — saves the 64 `v_cmp`/`v_cndmask` per tile); loop **3b** (ISA `.LBB0_12`) runs the
diagonal-crossing tiles with `causal_n_start = tile_n_start - _causal_offset`. For self-attention
`_causal_offset = 0`, so `_first_causal_tile = bx`, meaning tile `bx-1` and below are mask-free and
only the last two tiles (the diagonal band) carry the mask. This is the gfx1250 analogue of the
gfx950 kernel's "mask only near the diagonal" optimization, realized here by loop fission rather than
per-tile runtime gating.

#### 1.7.3 The causal mask itself

`_apply_causal_mask` (`fmha_kernel.py:834`) sets score elements to `-inf` where the query row is
before the key column. The per-lane bound is
`_base = (m_start - n_start) + wave*32 + (lane_lo - lane_hi*8)` with per-(SU,MSB) column offset
`(msb//2)*16 - su*32 - (msb%2)*16` (line 842), reflecting the WMMA score register layout (§2.4). The
comparison is `v_cmp_gt_i32_e64` + `v_cndmask_b32_e64` over the 8 elements of each `v8f32` score tile
(4 SU x 4 MSB x 8 = 128 compares, but grouped as 32 `v_cmp` + 32 `v_cndmask` per the histogram since
adjacent elements share a compare). `_apply_kv_oob_mask` (line 858) does the same against
`actual_kv_len` to mask keys past the real sequence end (needed because TDM pads the tile to 128 and
OOB reads return 0, `SH#65:14241-14242`).

The bottom-right causal alignment (`n_start = -(sk - sq)` in the prologue, line 1137) anchors the
diagonal at the bottom-right corner of the QK matrix so that when `sq < sk` the last query row attends
to the last key — the standard "chunked prefill" convention.

---

## 2. Full GM/LDS/VGPR layout maps

Notation (bf16 elements unless marked byte):

```
tid      = thread_idx.x                       # 0..127
wave_id  = tid / 32   (0..3)
lane     = tid % 32   (0..31)
lane_lo  = lane & 15    lane_hi = lane >> 4
q_row    = bx*128 + wave_id*32 + (per-WMMA M index)
kv tile  = 128 keys ; SU = 32 keys (CNT_SU=4 SUs per tile)
```

Distinguishing features vs the gfx950 kernel: **Q is loaded GM->VGPR directly** (no LDS); **K and V
reach LDS via the TDM** (`tensor_load_to_lds`), not `buffer_load...lds`; and the VGPR operands are
addressed through the 4-bank `s_set_vgpr_msb` scheme (§1.1).

<a id="sec-2-1-q"></a>
### 2.1 Q: GM -> VGPR

`_phase4_q_load_flydsl` (`fmha_prologue.py:182`) loads, per lane, one Q row's full 192-dim head as 6
`buffer_load_b128` (v4i32 = 8 bf16 each) into 4 banks, then pairs them into `q_frags[4 bank][3 frag]`
v16bf16 (the WMMA B operand).

```
lane_lo = lane & 15 ; lane_hi = lane >> 4
base    = lane_lo * stride_q_seq + lane_hi * 16          # per-lane byte offset (source 197)
wave_off= (wave_id * 32) * stride_q_seq
q_byte_off = base + wave_off ; q_elem_off = q_byte_off >> 2
# bank layout: bank covers half of QK_HDIM (96 K-cols = 3 frags of 32)
_FRAGS_PER_BANK = (QK_HDIM//2)//32 = 3     _LOADS_PER_BANK = 6
bank_offsets_bytes = [0, 192, stride*16, stride*16 + 192]    # source 221-226
for bank in 0..3:
  for i in 0..5:  buffer_load_b128(q_rsrc, bank_voff + i*32)  # 6 loads
  for f in 0..2:  q_frags[bank][f] = pack(load[2f], load[2f+1])   # v16bf16
```

The base GM offset (batch/head/q-block via THD `q_start_tok`) is folded into `q_offset`
(`fmha_kernel.py:1000-1004`). Q is **pre-scaled** by `scale = scalar_f * log2e` during the bridge into
`q_frags` (the `q_frags[q_msb in {0,2}]` mapping at `fmha_kernel.py:1032-1035`), folding both the
1/sqrt(D) temperature and the `exp -> exp2` conversion into the operand.

ISA confirmation (Q load, lines 260-265):

```asm
buffer_load_b128 v[66:69], v18, s[4:7], null offen
buffer_load_b128 v[70:73], v18, s[4:7], null offen offset:32
buffer_load_b128 v[50:53], v18, s[4:7], null offen offset:64
...
```

The 6-per-bank / 4-bank layout matches the two 96-wide K-column halves of the 192-dim contraction: a
`q_msb` merges two adjacent banks (lo+hi K-col halves) so that each QK WMMA reduces over the full
K=192 in 6 W_K=32 steps (`fmha_kernel.py:1024-1035`).

**Per-lane Q GM byte offset** (`q_byte_off = lane_lo*stride_q_seq + lane_hi*16 + wave_id*32*stride`,
source 197-199). Each lane owns one query row's slice; `lane_lo` = the query row within the wave's
32-row block, `lane_hi` = which 8-bf16 D-chunk (`hi=0` -> low, `hi=1` -> high 16 elems of a K-step):

| lane | lane_lo | lane_hi | query row (in wave) | GM byte offset |
|---:|---:|---:|---:|---|
| 0 | 0 | 0 | row 0 | `0*stride + 0` |
| 1 | 1 | 0 | row 1 | `1*stride + 0` |
| 15 | 15 | 0 | row 15 | `15*stride + 0` |
| 16 | 0 | 1 | row 0 | `0*stride + 16` |
| 17 | 1 | 1 | row 1 | `1*stride + 16` |
| 31 | 15 | 1 | row 15 | `15*stride + 16` |

**Per-lane Q -> VGPR bank/fragment map.** The 4 banks split the 192-dim K axis into two 96-wide
halves (banks 0/1 = K-cols `0..95`, banks 2/3 = `96..191`) x two lane-row groups; the 3 fragments per
bank tile the 96 cols into 3 x W_K=32 (`_FRAGS_PER_BANK = (192/2)/32 = 3`, source 217):

| bank | GM base offset (bytes) | K-columns covered | frags | consumed as WMMA `src1` (Q) by |
|---:|---|---|---:|---|
| 0 | `q_base + 0` | `0..95`   (rows 0-15) | `q_frags[0][0..2]` | `q_msb 0`, k-steps 0-2 |
| 1 | `q_base + 192` | `96..191` (rows 0-15) | `q_frags[1][0..2]` | `q_msb 0`, k-steps 3-5 (bridged) |
| 2 | `q_base + stride*16` | `0..95`   (rows 16-31) | `q_frags[2][0..2]` | `q_msb 2`, k-steps 0-2 |
| 3 | `q_base + stride*16 + 192` | `96..191` (rows 16-31) | `q_frags[3][0..2]` | `q_msb 2`, k-steps 3-5 (bridged) |

(The bridge at `fmha_kernel.py:1032-1035` merges banks 0+1 into `q_msb 0` and banks 2+3 into `q_msb 2`,
each `q_msb` then supplying all 6 W_K=32 steps of one 16-row M-tile.)

<a id="sec-2-2-k"></a>
### 2.2 K: GM -> LDS -> VGPR

K is moved GM->LDS by the **TDM** and re-gathered into the WMMA A operand with plain `ds_load_b128`
(no transpose — the K layout in LDS is already A-major after the TDM tile write).

**GM -> LDS (`_tdm_load_k_only`, `fmha_kernel.py:343`):** one `tensor_load_to_lds` per SU (4 SUs =
128 rows). The TDM is programmed by a 2-group SGPR descriptor (`SH#65:14224-14240`): group-0 =
`[pred, lds_off, addr_lo, addr_hi]`, group-1 = `[config, dim0<<16, dim1_rows<<16, dim0_stride<<16,
dim1_rows, seq_stride, 0, 0]`. Here `dim0_valid = QK_HDIM = 192` (global read width), `dim0_stride =
200` elements (LDS inner stride = 400 B = `K_ROW_BYTES`), `dim1_rows = 8` (rows per wave). Each of the
4 waves loads 8 of the 32 rows of a SU (`lds_base + wave_id*8*K_ROW_BYTES`, line 402). K uses **no TDM
padding** (`_K_TDM_CONFIG = 1<<16`, data_size=bf16, pad disabled) because QK_HDIM=192 is not a
power-of-2-friendly pad interval (`fmha_prologue.py:90-93`); the 8-bf16 tail padding in
`K_ROW_BYTES=400` is the only K padding, applied via the `dim0_stride=200` vs `dim0_valid=192` gap.

OOB is handled by per-SU descriptors (`_make_kv_dg1_with_oob`, `fmha_kernel.py:311`) that set
`dim1_rows` (the tensor row count) to the clamped `actual_kv_len - su*32` so TDM reads past the
sequence end return 0 (`SH#65:14241-14242`).

ISA confirmation (prologue TDM, lines 464-471): the descriptor SGPRs are built, then
`tensor_load_to_lds s[12:15], s[4:11]` (dst LDS group + src descriptor), each followed by
`s_barrier_signal -1`. Completion is awaited with `s_wait_tensorcnt 0` (`SH#65:5618-5622`, TENSORcnt).

**LDS -> VGPR (`_build_lds_k_schedule` / `_emit_lds_load`, `fmha_core_loop.py:774,1517`):** 6
`ds_load_b128` per MSB (24 per SU), each reading a v4i32 (8 bf16) at `v_idx*32 + su_off` within its
MSB's LDS address (`kv_lds_addrs[msb]`). Two raw loads pair into one `v16bf16` K fragment
(`_pair_k_tiles_for_wmma`, line 333); `N_WMMA_K_TILES = 3` frags per MSB cover the 192-dim K in 6
W_K=32 steps. The per-lane LDS K address (`_build_kv_lds_addrs`, `fmha_kernel.py:192`):

```
lane_lo = lane & 0xF ; lane_hi = lane >> 4
k_lane_off = lane_lo * K_ROW_BYTES + lane_hi * 16
k_dh0    = k_base + k_lane_off                          # kv_lds_addrs[0], bank 0
k_dh1    = k_dh0 + K_SU_HALF_OFFSET(0x1900)             # kv_lds_addrs[1], bank 1
k_dh0_hi = k_dh0 + QK_HDIM*KV_BPP/2 (=192)             # kv_lds_addrs[2], bank 2
k_dh1_hi = k_dh1 + 192                                  # kv_lds_addrs[3], bank 3
```

Each of the 4 K addresses is a distinct SSA value tagged to its own bank (`set_vgpr_bank(k_dh0, 0)`
etc.) so a whole MSB's 6 loads use one `s_set_vgpr_msb` context. ISA confirmation (loop `ds_load_b128`
around line 3760+): loads share an address VGPR with stepped `offset:` immediates.

**Per-lane K LDS byte address, bank 0 (`k_dh0`, v_idx=0)** — `lane_lo*400 + lane_hi*16` (`lane_lo` =
key row within the 16-row transpose group, `lane_hi` selects the 8-bf16 D-chunk; each row is
`K_ROW_BYTES = 400` = 192 bf16 data + 8 bf16 pad):

| lane | lane_lo | lane_hi | K LDS byte (rel) | Logical K (key row, D chunk) |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | `0` (0x0000) | key 0, D `0..7` |
| 1 | 1 | 0 | `400` (0x0190) | key 1, D `0..7` |
| 7 | 7 | 0 | `2800` (0x0af0) | key 7, D `0..7` |
| 15 | 15 | 0 | `6000` (0x1770) | key 15, D `0..7` |
| 16 | 0 | 1 | `16` (0x0010) | key 0, D `8..15` |
| 17 | 1 | 1 | `416` (0x01a0) | key 1, D `8..15` |
| 31 | 15 | 1 | `6016` (0x1780) | key 15, D `8..15` |

The 4 bank addresses select the K-column half and SU half of the 192-dim contraction (all relative to
`k_base = kv_lds_addrs` for the current ping-pong buffer):

| bank | address | offset from `k_dh0` | K region |
|---:|---|---|---|
| 0 (`k_dh0`) | `k_base + lane_off` | `0` | D-half 0 (`0..95`), SU low |
| 1 (`k_dh1`) | `+ K_SU_HALF_OFFSET` | `0x1900` (6400 = 16*400) | D-half 0, SU high |
| 2 (`k_dh0_hi`) | `+ 192` | `192` (QK_HDIM*KV_BPP/2) | D-half 1 (`96..191`), SU low |
| 3 (`k_dh1_hi`) | `+ 0x1900 + 192` | | D-half 1, SU high |

**K LDS buffer map** (one ping-pong buffer, per SU = 32 keys x 192 D). Each of the 4 waves TDM-writes
8 contiguous key rows at `wave_id*8*400`; each row is `K_ROW_BYTES = 400` B = 384 B data (192 bf16) +
16 B pad (8 bf16):

```
K LDS (per SU, offset su*LDS_K_SU_P_SIZE=0x3200 within buffer):
 rows   0.. 7 : wave 0   (each row: 192 bf16 K-dims + 8 pad)   byte base 0
 rows   8..15 : wave 1                                          byte base 8*400
 rows  16..23 : wave 2                                          byte base 16*400
 rows  24..31 : wave 3                                          byte base 24*400
```

<a id="sec-2-3-v"></a>
### 2.3 V: GM -> LDS -> VGPR (HW transpose load)

V uses the TDM for GM->LDS (like K) but a **hardware transpose LDS read** (`ds_load_tr16_b128`) for
the operand re-gather, because the PV WMMA needs V as the A matrix in transposed orientation.

**GM -> LDS (`_tdm_load_v_only`, `fmha_kernel.py:417`):** one `tensor_load_to_lds` per SU;
`dim0_elems = 128` (V_HDIM), `dim1_rows = 8`. Unlike K, V **uses TDM padding**: `_V_TDM_CONFIG =
(1<<20)|(5<<22)|(7<<25)` sets `pad_interval` enc 5 (128 elems) and `pad_amount` enc 7 (32 B)
(`fmha_prologue.py:94-95`), giving `V_ROW_BYTES = 288` (128 bf16 + 32 B pad). The extra padding keeps
the 16-lane `ds_load_tr16_b128` transpose groups bank-conflict-free (`SH#65:14272-14277` on TDM LDS
padding).

#### 2.3.1 `ds_load_tr16_b128` semantics

From the MI400 guide (`SH#65:12077-12091`):

> "4.7.2.4. LDS to VGPR Matrix Load with Transpose — This is new for MI400. These instructions allow
> matrix data to be copied from LDS to VGPRs and transpose the data on the way. ...
> `DS_LOAD_TR16_B128` — Load A or B matrix with element-size of 16bits into VGPRs from LDS and
> transpose."

Each `ds_load_tr16_b128` moves **128 bits/lane = 8 bf16 elements into 4 consecutive VGPRs**, loading
one 16x16 16-bit tile transposed (`SH#65:13969-13972,13999`). EXEC is ignored / treated as all-ones
(`SH#65:12080-12081`) and it is wave32-only (`SH#65:12084`). A full WMMA A operand (16x32) needs
**two** such tiles, so `_build_lds_v_schedule` (`fmha_core_loop.py:792`) issues `N_LDS_V_PER_MSB = 4`
transpose reads per MSB and `_pair_v_tiles_for_wmma` (line 1998) pairs them into `v16bf16` fragments.

The MSB->V-bank mapping folds the D-column layout into the address: MSBs {0,2} -> V-bank 0 (D cols
0-63), MSBs {1,3} -> V-bank 1 (D cols 64-127), with the second column-group offset in
`kv_lds_addrs[4 + msb*2 + half]` (`fmha_kernel.py:237-249`). ISA confirmation (loop, lines 4468-4478):

```asm
ds_load_tr16_b128 v[170:173], v146
ds_load_tr16_b128 v[174:177], v146 offset:4608
ds_load_tr16_b128 v[162:165], v146 offset:32
ds_load_tr16_b128 v[166:169], v146 offset:4640
```

The paired `offset:0 / offset:4608` (and `32 / 4640`, ...) are the two 16x16 halves of each A operand
(the +4608 ≈ 16 rows x 288 B stride selects the second K-half).

#### 2.3.2 V LDS per-lane address

`_build_kv_lds_addrs` (V part, `fmha_kernel.py:216-226`):

```
lane_and_7 = lane & 7 ; lane_shr4 = lane >> 4
v_row     = lane_and_7 + (lane_shr4 << 3)              # 0..15 row within transpose group
v_sub_col = ((lane >> 3) & 1) << 4                     # 0 or 16 (D sub-column)
v_lane_off= v_row * V_ROW_BYTES(288) + v_sub_col
v_dh0     = v_base + v_lane_off                        # bank-msb
v_dh1     = v_dh0 + V_SU_HALF_OFFSET(0x1200)
```

`v_row` (stride 288 = one V LDS line) selects the transpose-group row; `v_sub_col` picks the low/high
D chunk; the per-MSB `_V_MSB_EXTRA = [0, 128, 64, 192]` offset (line 239) folds the D-half and
column-group so all 8 V addresses are distinct SSA values, each with its own bank hint — again
minimizing `s_set_vgpr_msb` switches within a V load group.

**Per-lane V LDS byte address** (`v_row*288 + v_sub_col`, base `v_dh0`). `v_row = (lane&7) +
(lane>>4)*8` gives 0..15 (the transpose-group row = V key index); `v_sub_col = ((lane>>3)&1)*16`
selects the low/high 16-elem D sub-column. Lanes 0-15 and 16-31 are the two halves of the 32-lane
wave that `ds_load_tr16_b128` transposes together:

| lane | v_row | v_sub_col | V LDS byte (rel) | Logical V (key row, D chunk) |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | `0` (0x0000) | key 0, D low |
| 1 | 1 | 0 | `288` (0x0120) | key 1, D low |
| 7 | 7 | 0 | `2016` (0x07e0) | key 7, D low |
| 8 | 0 | 16 | `16` (0x0010) | key 0, D high |
| 15 | 7 | 16 | `2032` (0x07f0) | key 7, D high |
| 16 | 8 | 0 | `2304` (0x0900) | key 8, D low |
| 23 | 15 | 0 | `4320` (0x10e0) | key 15, D low |
| 24 | 8 | 16 | `2320` (0x0910) | key 8, D high |
| 31 | 15 | 16 | `4336` (0x10f0) | key 15, D high |

**The transpose effect** (per 16-lane group, `SH#65:12077-12091`). In LDS V is stored row-major
(key-major, DMA-friendly from the TDM tile write); `ds_load_tr16_b128` delivers it column-major so
each lane's 4 VGPRs hold the same D across consecutive keys — the WMMA A-matrix order:

```
Before (LDS, TDM row-major):              After ds_load_tr16_b128 (VGPR, WMMA-A order):
  key0  D 0..7  at lanes' row 0            lane 0 -> [V(key0,d0), V(key1,d0), V(key2,d0), V(key3,d0)]
  key1  D 0..7  at lanes' row 1            lane 1 -> [V(key0,d1), V(key1,d1), V(key2,d1), V(key3,d1)]
  ...                                      ...
```

**V bank -> D-column mapping.** The 8 V addresses (`kv_lds_addrs[4 + msb*2 + half]`) fold
`_V_MSB_EXTRA = [0, 128, 64, 192]` so each MSB owns a distinct D column-group:

| MSB | V-bank (`msb%2`) | `_V_MSB_EXTRA` | D columns |
|---:|---:|---:|---|
| 0 | 0 | 0 | `0..31` |
| 1 | 1 | 128 | `64..95` |
| 2 | 0 | 64 | `32..63` |
| 3 | 1 | 192 | `96..127` |

<a id="sec-2-4-s-p"></a>
### 2.4 S and P: VGPR-only score / probability layout

S never touches GM or LDS: it is the f32 output of the QK WMMA. Per the WMMA C/D layout
(`SH#65:10676-10717`), the 16x16 f32 accumulator is "one row of the matrix striped across the lanes
within one VGPR" — for wave32, lane = the N (key) index within a 16-wide strip, and the 8 VGPRs of the
`v8f32` hold M (query rows) 0,8 / 1,9 / ... So each of the 4 MSB accumulators holds a 16x16 score
sub-tile, and the 4 SUs give the full 128-key row per query.

**Prologue accumulator -> key-column map** (`ACC_COL_BASE`, `fmha_prologue.py:70-87`; `_acc_bank =
(g_idx&1) + 2*(tile>=2)`). The 64 prologue QK WMMAs land in a dict keyed by `(g_idx, tile)`
(g_idx 0-1 = SU0, 2-3 = SU1; tile 0-3 = N-strip). Each cell's `col_base` is the first key column of
that 16x16 f32 accumulator; the 8 elements span `col_base .. col_base+7`:

| g_idx | tile 0 | tile 1 | tile 2 | tile 3 |
|---:|---|---|---|---|
| 0 | col 0 (bank 0) | col 16 (bank 0) | col 32 (bank 2) | col 48 (bank 2) |
| 1 | col 8 (bank 1) | col 24 (bank 1) | col 40 (bank 3) | col 56 (bank 3) |
| 2 | col 64 (bank 0) | col 80 (bank 0) | col 96 (bank 2) | col 112 (bank 2) |
| 3 | col 72 (bank 1) | col 88 (bank 1) | col 104 (bank 3) | col 120 (bank 3) |

So the 4 banks partition the 128-key row: bank 0 owns keys `{0..15, 64..95}`, bank 1 `{8..31, 72..103}`
(interleaved by 8), etc. — each lane owns one query row and holds a strided subset of the 128 keys,
with `v_permlanex16_b32` completing the cross-16-lane half of the row reduction.

The compact `sp_pairs` layout (`_sp_tiles_to_sp_pairs`, `fmha_core_loop.py:2140`) packs the 4 SUs'
`v8f32` per MSB into 16 `v2f32` pairs (32 f32/lane, all real, **no padding** — the design decision
that keeps the 796-VGPR budget in bounds):

| SU | sp_pairs indices (per MSB) | key columns (via `col_offset`) |
|---:|---|---|
| 0 | pairs 0-3 | `(msb//2)*16 - 0  - (msb%2)*16 + e` |
| 1 | pairs 4-7 | `(msb//2)*16 - 32 - (msb%2)*16 + e` |
| 2 | pairs 8-11 | `(msb//2)*16 - 64 - (msb%2)*16 + e` |
| 3 | pairs 12-15 | `(msb//2)*16 - 96 - (msb%2)*16 + e` |

The causal-mask column derivation (`fmha_kernel.py:842`) is the authoritative statement of which key
column each `(SU, MSB, element)` owns: `col_offset(su, msb, e) = (msb//2)*16 - su*32 - (msb%2)*16 + e`
(relative to `_base`).

After max-shift + `exp2`, `p_bf16[msb]` holds 16 `v2bf16` per MSB, packed by
`_build_p_tiles_from_softmax` (line 2345) into `p_tiles[su][m_tile]` v16bf16 — the PV WMMA B operand.
Each PV src1 concatenates two sibling MSBs (`2*m_tile`, `2*m_tile+1`) along the K_pv axis so one
v16bf16 covers K_pv=0..31 for one (M_tile, SU) chunk.

The reduction cross-lane exchange uses `v_permlanex16_b32` (lanes 0-15 <-> 16-31, `SH#65:9534-9536`),
visible in the ISA at lines 1500-1503 (prologue) and 3234/5142 (loops):

```asm
v_permlanex16_b32 v230, v230, s35, 0xfedcba98
```

<a id="sec-2-5-o"></a>
### 2.5 O: VGPR -> LDS -> GM

O is accumulated in `o_tiles[4 d_msb][N_PV_WMMA_N=4]` v8f32 (32 f32/lane per d_msb, resident across
the whole loop), rescaled every tile by `exp_delta` (the online max-shift correction,
`fmha_kernel.py:1541`), then in the epilogue (`_ep_finish`, line 2656): reduced row-sum, normalized by
`rcp(row_sum)`, cast to bf16 (`v_cvt_pk_bf16_f32`), staged to LDS (`ds_store_b128`, 16 stores), and
written to global.

```
row_sum = cross-MSB add + permlanex16 reduce         (2661-2676)
if RETURN_LSE: lse = local_max*scalar + log(row_sum)*ln2   (v_log_f32, 2677-2682)
O_bf16[d_msb][n] = trunc<bf16>(o[d_msb][n] * rcp(row_sum))   (2737-2755)
barrier ; ds_store_b128 O_bf16 -> V_a LDS (2777-2792) ; TDM/global_store_b128 -> GM
```

The O store reuses the V_a LDS region (`_lds_alloc_v_a`, line 2768) since PV is complete. The seqlen==0
fast path (`fmha_kernel.py:937-991`) zero-fills O directly with `global_store_b128` when there are no
keys. ISA confirmation (O store, lines 184-199 for the zero-fill path; 8712-8724 for `ds_store_b128`):

```asm
ds_store_b128 v64, v[0:3]
ds_store_b128 v65, v[4:7]
...
global_store_b128 v6, v[2:5], s[24:25] offset:16
```

**Per-lane O -> LDS staging byte offset** (`_loff = lane_lo*TDM_D_TILE_DIM0(256) + lane_hi*16`,
`fmha_kernel.py:2773-2776`). `lane_lo` = the query row within the wave's 16-row group, `lane_hi`
selects the 8-bf16 half of the D chunk:

| lane | lane_lo | lane_hi | O LDS `_loff` (rel to wave base) | query row (in group) |
|---:|---:|---:|---|---:|
| 0 | 0 | 0 | `0` (0x0000) | row 0 |
| 1 | 1 | 0 | `256` (0x0100) | row 1 |
| 15 | 15 | 0 | `3840` (0x0f00) | row 15 |
| 16 | 0 | 1 | `16` (0x0010) | row 0 (high half) |
| 31 | 15 | 1 | `3856` (0x0f10) | row 15 (high half) |

**Per-(d_msb, n) inner offset** `_ioff = (d_msb//2)*16*256 + (d_msb%2)*128 + n*32`
(`fmha_kernel.py:2779-2783`) — the 16 `ds_store_b128` (4 d_msb x 4 n) tile the 128 output channels:

| d_msb | n=0 | n=1 | n=2 | n=3 | D columns |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 32 | 64 | 96 | `0..31` (M-tile 0, D low) |
| 1 | 128 | 160 | 192 | 224 | `32..63` (M-tile 0, D high) |
| 2 | 4096 | 4128 | 4160 | 4192 | `0..31` (M-tile 1, D low) |
| 3 | 4224 | 4256 | 4288 | 4320 | `32..63` (M-tile 1, D high) |

(The `(d_msb//2)*4096` term steps to the second 16-query M-tile — 16 rows x 256 B; `(d_msb%2)*128`
selects the high D half; `n*32` walks the 4 WMMA N-strips of one d_msb.)

<a id="sec-2-6-summary"></a>
### 2.6 Summary: where each tensor lives

```
Q:  GM[THD,H,192] -> buffer_load_b128 x24 -> VGPR q_frags[4][3] v16bf16 (pre-scaled). No LDS.
K:  GM[THD,H_kv,192] -> tensor_load_to_lds (TDM) -> LDS K_a/K_b (K_ROW_BYTES=400)
                     -> ds_load_b128 x24/SU -> VGPR kv_tiles[4][3] (QK WMMA src0).
V:  GM[THD,H_kv,128] -> tensor_load_to_lds (TDM, +32B pad) -> LDS V_a/V_b (V_ROW_BYTES=288)
                     -> ds_load_tr16_b128 x16/SU-pair (HW transpose) -> VGPR v_tiles[2][4] (PV src0).
S:  VGPR only. sp_tiles[4 msb] v8f32 -> compact sp_pairs[4][16] v2f32 (32 f32/lane, no padding).
P:  VGPR only. p_bf16[4][16] v2bf16 -> p_tiles[4 su][2] v16bf16 (PV WMMA src1).
O:  VGPR o_tiles[4][4] v8f32 -> *exp_delta each tile -> *rcp(sum) -> bf16
                     -> ds_store_b128 -> LDS (V_a reuse) -> TDM/global_store_b128 -> GM. 
```

### 2.7 Low-level call chains and issue counts

| Source statement | Role | Lowest-level primitive | ISA form | Count/call |
|---|---|---|---|---:|
| `_tdm_load_k/v_only` (`fmha_kernel.py:343,417`) | GM tile -> LDS | `rocdl.tensor_load_to_lds` | `tensor_load_to_lds` | 4 (per K or V tile) |
| `_phase4_q_load_flydsl` (`fmha_prologue.py:182`) | GM Q -> VGPR | `rocdl.raw_ptr_buffer_load` | `buffer_load_b128` | 24 |
| `_emit_lds_load` K (`fmha_core_loop.py:1517`) | LDS K -> VGPR | `llvm.load` (addrspace 3) | `ds_load_b128` | 24/SU |
| `_emit_lds_load` V (transpose) | LDS V -> VGPR | `rocdl.ds_load_tr16_b128` | `ds_load_tr16_b128` | 16/SU-pair |
| `_emit_qk_wmma` (1470) | GEMM1 score | `rocdl.wmma_f32_16x16x32_bf16` | `v_wmma_f32_16x16x32_bf16` | 24/stage |
| `_emit_pv_wmma` (1497) | GEMM2 O | `rocdl.wmma_f32_16x16x32_bf16` | `v_wmma_f32_16x16x32_bf16` | 16/stage |
| O store (`_ep_finish`, 2777) | VGPR O -> LDS -> GM | `llvm.store` + `rocdl.tensor_store` / `global_store` | `ds_store_b128` + `global_store_b128` | 16 + 16 |

Per-warp issue totals (verified against `21_final_isa.s`, §1.5): **704** `v_wmma_f32_16x16x32_bf16`,
**384** `ds_load_b128`, **320** `ds_load_tr16_b128`, **32** `tensor_load_to_lds`, **24**
`buffer_load_b128`, **16** `global_store_b128`, **16** `ds_store_b128`.

---

## Appendix: gfx1250 vs gfx950 — why the design diverged

| Design choice | Reason (gfx1250-specific) |
|---|---|
| WMMA 16x16x32 (not MFMA 32x32x16) | gfx1250 has no MFMA; WMMA is wave32-only (`SH#65:10420`) with fixed SRC0=A/SRC1=B (no swap) |
| TDM `tensor_load_to_lds` (not `buffer_load...lds`) | The Tensor Data Mover is the gfx1250 async GM->LDS engine ("accelerate CUDA memcpy_async", `SH#65:4088`); tracked by TENSORcnt |
| 4-bank `s_set_vgpr_msb` register file | 1024 VGPRs addressed via 2-bit-per-operand MSB (`SH#65:2250-2291`); replaces MI300 AGPR/ACCUM |
| Per-WMMA token schedule (not IGroupLP hints) | Fine-grained control needed to hide VALU/exp behind WMMA at wave32 + high VGPR pressure |
| 2 rolled loops (causal fission) | Skips the causal mask's 64 `v_cmp`/`v_cndmask` on below-diagonal tiles |
| Compact no-padding SP layout | 796-VGPR budget (near the 1024 max) leaves no room for -inf-padded score tiles |
| QK D=192 / V D=128 asymmetry | The MLA-style head geometry this kernel targets (K/Q wider than V) |

Every count and semantic claim above is traceable to the cited source line or `21_final_isa.s` line,
with instruction semantics taken from `MI400_Shader_Programming#65.txt` (cited `SH#65:<line>`).
