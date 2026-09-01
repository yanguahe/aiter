	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	flash_attn_dualwave_swp_gfx950_kernel
	.p2align	8
	.type	flash_attn_dualwave_swp_gfx950_kernel,@function
flash_attn_dualwave_swp_gfx950_kernel:
; ==============================================================================
; FLYDSL DUALWAVE FOLD ISA ANNOTATION
; Source: FlyDSL/kernels/flash_attn_gfx950.py, gfx950 D=128 bf16 causal,
; B=1, S=8192, H=Hkv=64, FLYDSL_DUALWAVE_CAUSAL_FOLD=1.
; Only comments are added; executable ISA instructions are unchanged.
; This dump contains two compile-time-unrolled q-block passes:
;   Pass 0: base q-block = blockIdx.y.
;   Pass 1: mirror q-block = num_q_blocks - 1 - blockIdx.y.
; Each pass has Prologue, Main loop, Epilogue, and Finalize/store regions.
; ==============================================================================
; ==============================================================================
; PASS 0 / BASE Q-BLOCK: PROLOGUE - kernarg, indices, first LDS prime
; Loads FlyDSL kernargs, derives q/head/batch/lane coordinates, builds buffer bases,
; and starts the first K tile global->LDS copy for q_block_idx = blockIdx.y.
; ==============================================================================
	s_mov_b32 s26, s3
	s_load_dwordx4 s[8:11], s[0:1], 0x6c
	s_load_dwordx2 s[16:17], s[0:1], 0x10
	s_load_dwordx2 s[12:13], s[0:1], 0x20
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s47, s8, 31
	s_ashr_i32 s5, s9, 31
	s_ashr_i32 s6, s10, 31
	s_ashr_i32 s27, s3, 31
	s_ashr_i32 s3, s4, 31
	v_lshrrev_b32_e32 v1, 6, v0
	v_readfirstlane_b32 s7, v0
	s_lshr_b32 s30, s7, 6
	s_lshr_b32 s33, s7, 8
	s_and_b32 s7, s2, 63
	s_lshr_b32 s2, s2, 6
	s_add_i32 s2, s7, s2
	s_add_u32 s22, s4, 1
	s_addc_u32 s23, s3, 0
	s_mul_i32 s3, s9, s4
	s_mul_i32 s14, s11, s9
	s_mul_i32 s14, s14, s22
	s_mov_b32 s15, 0x27000
	s_lshl_b32 s14, s14, 1
	s_and_b32 s17, s17, 0xffff
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	v_lshlrev_b32_e32 v2, 3, v0
	s_mul_i32 s42, s30, 0x410
	v_and_or_b32 v3, v0, 56, v1
	v_and_b32_e32 v4, 56, v2
	v_add_u32_e32 v3, s3, v3
	v_mul_lo_u32 v3, v3, s11
	v_lshl_add_u32 v3, s7, 7, v3
	v_add_lshl_u32 v218, v3, v4, 1
	s_mov_b32 m0, s42
	s_nop 0
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P0: async-load first K tile into LDS
; buffer_load_dwordx4 ... lds primes K buffer 0 before the first QK MMA.
; ==============================================================================
	buffer_load_dwordx4 v218, s[16:19], 0 offen lds
	s_add_i32 s43, s42, 0x2080
	v_add_u32_e32 v219, 0x80, v218
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], 0 offen lds
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P1: Q row/global address setup
; Computes q_row_in_block, q_start_pos, q_row, and the 128-wide Q vector address.
; ==============================================================================
	s_load_dwordx2 s[20:21], s[0:1], 0x0
	s_load_dword s31, s[0:1], 0x7c
	s_mov_b32 s46, s8
	v_mov_b32_e32 v209, 0
	v_and_b32_e32 v3, 31, v0
	v_bfe_u32 v208, v0, 5, 1
	s_mov_b32 s3, 0
	s_mul_i32 s36, s8, s4
	s_mul_i32 s4, s22, s47
	s_mul_hi_u32 s7, s22, s8
	s_add_i32 s4, s7, s4
	s_mul_i32 s23, s23, s8
	s_add_i32 s4, s4, s23
	s_mul_i32 s22, s22, s8
	s_mul_i32 s6, s22, s6
	s_mul_hi_u32 s7, s22, s10
	s_add_i32 s6, s7, s6
	s_mul_i32 s4, s4, s10
	s_add_i32 s7, s6, s4
	s_mul_i32 s6, s22, s10
	s_lshl_b64 s[6:7], s[6:7], 1
	s_waitcnt lgkmcnt(0)
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s22, s6
	s_mov_b32 s23, s15
	s_and_b32 s13, s13, 0xffff
	v_and_b32_e32 v4, 7, v0
	v_mul_u32_u24_e32 v4, 0x208, v4
	v_and_b32_e32 v2, 0xc0, v2
	v_add_u32_e32 v10, v4, v2
	v_lshlrev_b32_e32 v217, 3, v208
	s_lshl_b64 s[24:25], s[26:27], 8
	s_add_u32 s4, s36, s24
	s_mul_i32 s4, s4, s10
	s_lshl_b64 s[2:3], s[2:3], 7
	s_add_u32 s51, s4, s2
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_barrier
	v_lshl_or_b32 v210, v1, 5, v3
	v_mov_b32_e32 v58, v210
	;;#ASMSTART
	;;#ASMEND
	v_mov_b32_e32 v1, v217
	;;#ASMSTART
	;;#ASMEND
	v_mad_u64_u32 v[214:215], s[28:29], v58, s10, 0
	v_add_u32_e32 v1, s51, v1
	v_add_lshl_u32 v1, v1, v214, 1
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P2: load Q tile from global memory
; Eight buffer_load_dwordx4 instructions load the full D=128 Q row fragments.
; ==============================================================================
	buffer_load_dwordx4 v[6:9], v1, s[20:23], 0 offen
	buffer_load_dwordx4 v[2:5], v1, s[20:23], 0 offen offset:32
	buffer_load_dwordx4 v[54:57], v1, s[20:23], 0 offen offset:64
	buffer_load_dwordx4 v[50:53], v1, s[20:23], 0 offen offset:96
	buffer_load_dwordx4 v[46:49], v1, s[20:23], 0 offen offset:128
	buffer_load_dwordx4 v[42:45], v1, s[20:23], 0 offen offset:160
	buffer_load_dwordx4 v[38:41], v1, s[20:23], 0 offen offset:192
	buffer_load_dwordx4 v[34:37], v1, s[20:23], 0 offen offset:224
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P3: prefetch K1 and V0 into LDS
; Starts the software pipeline: next K tile goes to K buffer 1, current V tile to V buffer 0.
; ==============================================================================
	s_add_i32 s41, s42, 0x8500
	s_lshl_b32 s48, s11, 7
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v218, s[16:19], s48 offen lds
	s_add_i32 s40, s42, 0xa580
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s48 offen lds
	s_mul_i32 s38, s30, 0x440
	s_add_i32 s39, s38, 0x4100
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds
	s_addk_i32 s38, 0x6300
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P4: read K0 from LDS into VGPRs
; ds_read_b128 sequence materializes the K fragments consumed by the first QK MMA chain.
; ==============================================================================
	v_add_lshl_u32 v215, v10, v217, 1
	ds_read_b128 v[10:13], v215
	ds_read_b128 v[128:131], v215 offset:32
	ds_read_b128 v[116:119], v215 offset:512
	ds_read_b128 v[120:123], v215 offset:544
	ds_read_b128 v[124:127], v215 offset:64
	ds_read_b128 v[102:105], v215 offset:96
	ds_read_b128 v[106:109], v215 offset:576
	ds_read_b128 v[98:101], v215 offset:608
	ds_read_b128 v[94:97], v215 offset:8320
	ds_read_b128 v[86:89], v215 offset:8352
	ds_read_b128 v[90:93], v215 offset:8832
	ds_read_b128 v[82:85], v215 offset:8864
	ds_read_b128 v[78:81], v215 offset:8384
	ds_read_b128 v[70:73], v215 offset:8416
	ds_read_b128 v[74:77], v215 offset:8896
	ds_read_b128 v[66:69], v215 offset:8928
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P5: wait/stagger barrier before first QK use
; Waits for LDS/VMEM readiness and optionally inserts the extra group-B stagger barrier.
; ==============================================================================
	s_waitcnt lgkmcnt(0)
	s_cmp_lg_u32 s33, 0
	s_cselect_b64 s[28:29], -1, 0
	s_cmp_eq_u32 s33, 0
	s_waitcnt vmcnt(2)
	s_cbranch_scc1 .LBB0_2
	s_barrier
.LBB0_2:
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P6: Q scale and first QK score tile
; Scales Q by rsqrt(D)*log2(e), packs bf16 fragments, then issues the first MMA0 chain.
; ==============================================================================
	v_cvt_f32_i32_e32 v1, s31
	v_and_b32_e32 v61, 0xffff0000, v37
	v_lshlrev_b32_e32 v60, 16, v37
	v_and_b32_e32 v63, 0xffff0000, v41
	v_rsq_f32_e32 v1, v1
	v_lshlrev_b32_e32 v62, 16, v41
	v_and_b32_e32 v65, 0xffff0000, v45
	v_lshlrev_b32_e32 v64, 16, v45
	v_mul_f32_e32 v212, 0x3fb8aa3b, v1
	v_and_b32_e32 v111, 0xffff0000, v49
	v_lshlrev_b32_e32 v110, 16, v49
	v_and_b32_e32 v133, 0xffff0000, v53
	v_lshlrev_b32_e32 v132, 16, v53
	v_and_b32_e32 v135, 0xffff0000, v57
	v_lshlrev_b32_e32 v134, 16, v57
	v_and_b32_e32 v137, 0xffff0000, v5
	v_lshlrev_b32_e32 v136, 16, v5
	v_and_b32_e32 v15, 0xffff0000, v9
	v_lshlrev_b32_e32 v14, 16, v9
	v_and_b32_e32 v9, 0xffff0000, v8
	v_lshlrev_b32_e32 v8, 16, v8
	v_and_b32_e32 v17, 0xffff0000, v7
	v_lshlrev_b32_e32 v16, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v6
	v_lshlrev_b32_e32 v6, 16, v6
	v_pk_mul_f32 v[6:7], v[212:213], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[212:213], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[212:213], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[212:213], v[14:15] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v115, v14, v15
	v_cvt_pk_bf16_f32 v114, v8, v9
	v_cvt_pk_bf16_f32 v113, v16, v17
	v_cvt_pk_bf16_f32 v112, v6, v7
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[10:13], v[112:115], 0
	v_and_b32_e32 v139, 0xffff0000, v4
	v_lshlrev_b32_e32 v138, 16, v4
	v_and_b32_e32 v141, 0xffff0000, v3
	v_lshlrev_b32_e32 v140, 16, v3
	v_and_b32_e32 v3, 0xffff0000, v2
	v_lshlrev_b32_e32 v2, 16, v2
	v_pk_mul_f32 v[142:143], v[212:213], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[116:119], v[112:115], 0
	v_mul_f32_e64 v116, v212, v140
	v_mul_f32_e64 v117, v212, v141
	v_mul_f32_e64 v138, v212, v138
	v_mul_f32_e64 v139, v212, v139
	v_mul_f32_e64 v118, v212, v136
	v_mul_f32_e64 v119, v212, v137
	v_cvt_pk_bf16_f32 v119, v118, v119
	v_cvt_pk_bf16_f32 v118, v138, v139
	v_cvt_pk_bf16_f32 v117, v116, v117
	v_cvt_pk_bf16_f32 v116, v142, v143
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[128:131], v[116:119], v[18:33]
	v_and_b32_e32 v57, 0xffff0000, v56
	v_lshlrev_b32_e32 v56, 16, v56
	v_and_b32_e32 v129, 0xffff0000, v55
	v_lshlrev_b32_e32 v128, 16, v55
	v_and_b32_e32 v55, 0xffff0000, v54
	v_lshlrev_b32_e32 v54, 16, v54
	v_pk_mul_f32 v[54:55], v[212:213], v[54:55] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[120:123], v[116:119], v[2:17]
	v_mul_f32_e64 v120, v212, v128
	v_mul_f32_e64 v121, v212, v129
	v_mul_f32_e64 v56, v212, v56
	v_mul_f32_e64 v57, v212, v57
	v_mul_f32_e64 v122, v212, v134
	v_mul_f32_e64 v123, v212, v135
	v_cvt_pk_bf16_f32 v123, v122, v123
	v_cvt_pk_bf16_f32 v122, v56, v57
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v54, v55
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[124:127], v[120:123], v[18:33]
	v_and_b32_e32 v53, 0xffff0000, v52
	v_lshlrev_b32_e32 v52, 16, v52
	v_and_b32_e32 v55, 0xffff0000, v51
	v_lshlrev_b32_e32 v54, 16, v51
	v_and_b32_e32 v51, 0xffff0000, v50
	v_lshlrev_b32_e32 v50, 16, v50
	v_pk_mul_f32 v[50:51], v[212:213], v[50:51] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[106:109], v[120:123], v[2:17]
	v_mul_f32_e64 v54, v212, v54
	v_mul_f32_e64 v55, v212, v55
	v_mul_f32_e64 v52, v212, v52
	v_mul_f32_e64 v53, v212, v53
	v_mul_f32_e64 v56, v212, v132
	v_mul_f32_e64 v57, v212, v133
	v_cvt_pk_bf16_f32 v127, v56, v57
	v_cvt_pk_bf16_f32 v126, v52, v53
	v_cvt_pk_bf16_f32 v125, v54, v55
	v_cvt_pk_bf16_f32 v124, v50, v51
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[102:105], v[124:127], v[18:33]
	v_and_b32_e32 v49, 0xffff0000, v48
	v_lshlrev_b32_e32 v48, 16, v48
	v_and_b32_e32 v51, 0xffff0000, v47
	v_lshlrev_b32_e32 v50, 16, v47
	v_and_b32_e32 v47, 0xffff0000, v46
	v_lshlrev_b32_e32 v46, 16, v46
	v_pk_mul_f32 v[46:47], v[212:213], v[46:47] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[98:101], v[124:127], v[2:17]
	v_mul_f32_e64 v50, v212, v50
	v_mul_f32_e64 v51, v212, v51
	v_mul_f32_e64 v48, v212, v48
	v_mul_f32_e64 v49, v212, v49
	v_mul_f32_e64 v52, v212, v110
	v_mul_f32_e64 v53, v212, v111
	v_cvt_pk_bf16_f32 v131, v52, v53
	v_cvt_pk_bf16_f32 v130, v48, v49
	v_cvt_pk_bf16_f32 v129, v50, v51
	v_cvt_pk_bf16_f32 v128, v46, v47
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[94:97], v[128:131], v[18:33]
	v_and_b32_e32 v45, 0xffff0000, v44
	v_lshlrev_b32_e32 v44, 16, v44
	v_and_b32_e32 v47, 0xffff0000, v43
	v_lshlrev_b32_e32 v46, 16, v43
	v_and_b32_e32 v43, 0xffff0000, v42
	v_lshlrev_b32_e32 v42, 16, v42
	v_pk_mul_f32 v[42:43], v[212:213], v[42:43] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[90:93], v[128:131], v[2:17]
	v_mul_f32_e64 v46, v212, v46
	v_mul_f32_e64 v47, v212, v47
	v_mul_f32_e64 v44, v212, v44
	v_mul_f32_e64 v45, v212, v45
	v_mul_f32_e64 v48, v212, v64
	v_mul_f32_e64 v49, v212, v65
	v_cvt_pk_bf16_f32 v135, v48, v49
	v_cvt_pk_bf16_f32 v134, v44, v45
	v_cvt_pk_bf16_f32 v133, v46, v47
	v_cvt_pk_bf16_f32 v132, v42, v43
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[86:89], v[132:135], v[18:33]
	v_and_b32_e32 v41, 0xffff0000, v40
	v_lshlrev_b32_e32 v40, 16, v40
	v_and_b32_e32 v43, 0xffff0000, v39
	v_lshlrev_b32_e32 v42, 16, v39
	v_and_b32_e32 v39, 0xffff0000, v38
	v_lshlrev_b32_e32 v38, 16, v38
	v_pk_mul_f32 v[38:39], v[212:213], v[38:39] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[82:85], v[132:135], v[2:17]
	v_mul_f32_e64 v42, v212, v42
	v_mul_f32_e64 v43, v212, v43
	v_mul_f32_e64 v40, v212, v40
	v_mul_f32_e64 v41, v212, v41
	v_mul_f32_e64 v44, v212, v62
	v_mul_f32_e64 v45, v212, v63
	v_cvt_pk_bf16_f32 v139, v44, v45
	v_cvt_pk_bf16_f32 v138, v40, v41
	v_cvt_pk_bf16_f32 v137, v42, v43
	v_cvt_pk_bf16_f32 v136, v38, v39
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[78:81], v[136:139], v[18:33]
	v_and_b32_e32 v37, 0xffff0000, v36
	v_lshlrev_b32_e32 v36, 16, v36
	v_and_b32_e32 v39, 0xffff0000, v35
	v_lshlrev_b32_e32 v38, 16, v35
	v_and_b32_e32 v35, 0xffff0000, v34
	v_lshlrev_b32_e32 v34, 16, v34
	v_pk_mul_f32 v[34:35], v[212:213], v[34:35] op_sel_hi:[0,1]
	v_mfma_f32_32x32x16_bf16 v[2:17], v[74:77], v[136:139], v[2:17]
	v_mul_f32_e64 v38, v212, v38
	v_mul_f32_e64 v39, v212, v39
	v_mul_f32_e64 v36, v212, v36
	v_mul_f32_e64 v37, v212, v37
	v_mul_f32_e64 v40, v212, v60
	v_mul_f32_e64 v41, v212, v61
	v_cvt_pk_bf16_f32 v143, v40, v41
	v_cvt_pk_bf16_f32 v142, v36, v37
	v_cvt_pk_bf16_f32 v141, v38, v39
	v_cvt_pk_bf16_f32 v140, v34, v35
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[18:33], v[70:73], v[140:143], v[18:33]
	s_sub_i32 s3, s9, s8
	s_lshl_b32 s8, s30, 5
	s_add_i32 s4, s8, s24
	v_add_u32_e32 v211, s24, v58
	v_mfma_f32_32x32x16_bf16 v[2:17], v[66:69], v[140:143], v[2:17]
	s_add_i32 s7, s3, s4
	s_cmp_gt_i32 s7, 63
	v_lshlrev_b32_e32 v222, 2, v208
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P7: causal mask for first score tile
; v_cmp/v_cndmask replace invalid causal score lanes with -inf before row max.
; ==============================================================================
	s_cbranch_scc1 .LBB0_4
	v_add_u32_e32 v1, s3, v211
	v_sub_u32_e32 v1, v1, v222
	v_subrev_u32_e32 v34, 32, v1
	v_mov_b32_e32 v35, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v1, 0
	v_cmp_lt_i32_e64 s[22:23], v1, 1
	v_cndmask_b32_e64 v18, v18, v35, s[18:19]
	v_cndmask_b32_e64 v19, v19, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v1, 2
	v_cmp_lt_i32_e64 s[22:23], v1, 3
	v_cndmask_b32_e64 v20, v20, v35, s[18:19]
	v_cndmask_b32_e64 v21, v21, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v1, 8
	v_cmp_lt_i32_e64 s[22:23], v1, 9
	v_cndmask_b32_e64 v22, v22, v35, s[18:19]
	v_cndmask_b32_e64 v23, v23, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v1, 10
	v_cmp_lt_i32_e64 s[22:23], v1, 11
	v_cndmask_b32_e64 v24, v24, v35, s[18:19]
	v_cndmask_b32_e64 v25, v25, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v1, 16
	v_cmp_lt_i32_e64 s[22:23], v1, 17
	v_cndmask_b32_e64 v26, v26, v35, s[18:19]
	v_cndmask_b32_e64 v27, v27, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v1, 18
	v_cmp_lt_i32_e64 s[22:23], v1, 19
	v_cndmask_b32_e64 v28, v28, v35, s[18:19]
	v_cndmask_b32_e64 v29, v29, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v1, 24
	v_cmp_lt_i32_e64 s[22:23], v1, 25
	v_cndmask_b32_e64 v30, v30, v35, s[18:19]
	v_cndmask_b32_e64 v31, v31, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v1, 26
	v_cmp_lt_i32_e64 s[22:23], v1, 27
	v_cndmask_b32_e64 v32, v32, v35, s[18:19]
	v_cndmask_b32_e64 v33, v33, v35, s[22:23]
	;;#ASMEND
	v_mov_b32_e32 v1, v3
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v34, 0
	v_cmp_lt_i32_e64 s[22:23], v34, 1
	v_cndmask_b32_e64 v2, v2, v35, s[18:19]
	v_cndmask_b32_e64 v1, v1, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v34, 2
	v_cmp_lt_i32_e64 s[22:23], v34, 3
	v_cndmask_b32_e64 v4, v4, v35, s[18:19]
	v_cndmask_b32_e64 v5, v5, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v34, 8
	v_cmp_lt_i32_e64 s[22:23], v34, 9
	v_cndmask_b32_e64 v6, v6, v35, s[18:19]
	v_cndmask_b32_e64 v7, v7, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v34, 10
	v_cmp_lt_i32_e64 s[22:23], v34, 11
	v_cndmask_b32_e64 v8, v8, v35, s[18:19]
	v_cndmask_b32_e64 v9, v9, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v34, 16
	v_cmp_lt_i32_e64 s[22:23], v34, 17
	v_cndmask_b32_e64 v10, v10, v35, s[18:19]
	v_cndmask_b32_e64 v11, v11, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v34, 18
	v_cmp_lt_i32_e64 s[22:23], v34, 19
	v_cndmask_b32_e64 v12, v12, v35, s[18:19]
	v_cndmask_b32_e64 v13, v13, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v34, 24
	v_cmp_lt_i32_e64 s[22:23], v34, 25
	v_cndmask_b32_e64 v14, v14, v35, s[18:19]
	v_cndmask_b32_e64 v15, v15, v35, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v34, 26
	v_cmp_lt_i32_e64 s[22:23], v34, 27
	v_cndmask_b32_e64 v16, v16, v35, s[18:19]
	v_cndmask_b32_e64 v17, v17, v35, s[22:23]
	;;#ASMEND
	v_mov_b32_e32 v3, v1
.LBB0_4:
	s_add_u32 s4, s9, 63
	s_addc_u32 s5, s5, 0
	s_lshr_b64 s[30:31], s[4:5], 6
	v_mul_u32_u24_e32 v1, 0x880, v208
	v_bfe_u32 v34, v0, 2, 2
	v_mul_u32_u24_e32 v34, 0x220, v34
	v_add_u32_e32 v1, v1, v34
	v_and_b32_e32 v34, 16, v0
	v_lshlrev_b32_e32 v0, 2, v0
	v_and_b32_e32 v0, 12, v0
	v_or3_b32 v0, v34, v1, v0
	s_mov_b32 s5, 0
	s_add_i32 s49, s3, 0x100
	s_add_i32 s4, s49, s24
	s_max_i32 s4, s4, 0
	s_add_i32 s4, s4, 63
	s_lshr_b32 s4, s4, 6
	v_mov_b64_e32 v[34:35], s[4:5]
	v_cmp_lt_u64_e32 vcc, s[30:31], v[34:35]
	s_and_b64 s[18:19], vcc, exec
	s_cselect_b32 s5, s31, 0
	s_cselect_b32 s4, s30, s4
	s_add_i32 s18, s4, 1
	s_and_b32 s18, s18, 0x7fffffe
	v_cmp_gt_u64_e64 s[4:5], s[4:5], 2
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s23, 0, 0
	s_cselect_b32 s22, s18, 4
	v_max_f32_e32 v1, v18, v19
	v_max3_f32 v1, v1, v20, v21
	v_max3_f32 v1, v1, v22, v23
	v_max3_f32 v1, v1, v24, v25
	v_max3_f32 v1, v1, v26, v27
	v_max3_f32 v1, v1, v28, v29
	v_max3_f32 v1, v1, v30, v31
	v_max3_f32 v1, v1, v32, v33
	v_max3_f32 v1, v1, v2, v3
	v_max3_f32 v1, v1, v4, v5
	v_max3_f32 v1, v1, v6, v7
	v_max3_f32 v1, v1, v8, v9
	v_max3_f32 v1, v1, v10, v11
	v_max3_f32 v1, v1, v12, v13
	v_max3_f32 v1, v1, v14, v15
	v_max3_f32 v1, v1, v16, v17
	v_mov_b32_e32 v34, v1
	s_nop 1
	v_permlane32_swap_b32_e64 v1, v34 bound_ctrl:1
	s_mov_b32 s4, 0xff61b1e6
	v_max3_f32 v216, v1, v34, s4
	v_sub_f32_e32 v1, v18, v216
	v_sub_f32_e32 v18, v19, v216
	v_sub_f32_e32 v19, v20, v216
	v_sub_f32_e32 v20, v21, v216
	v_sub_f32_e32 v21, v22, v216
	v_sub_f32_e32 v22, v23, v216
	v_sub_f32_e32 v23, v24, v216
	v_sub_f32_e32 v24, v25, v216
	v_sub_f32_e32 v25, v26, v216
	v_sub_f32_e32 v26, v27, v216
	v_sub_f32_e32 v27, v28, v216
	v_sub_f32_e32 v28, v29, v216
	v_sub_f32_e32 v34, v30, v216
	v_sub_f32_e32 v35, v31, v216
	v_sub_f32_e32 v32, v32, v216
	v_sub_f32_e32 v33, v33, v216
	v_sub_f32_e32 v31, v17, v216
	v_sub_f32_e32 v30, v16, v216
	v_sub_f32_e32 v29, v15, v216
	v_exp_f32_e32 v167, v1
	v_exp_f32_e32 v168, v18
	v_exp_f32_e32 v164, v19
	v_exp_f32_e32 v166, v20
	v_exp_f32_e32 v163, v21
	v_exp_f32_e32 v165, v22
	v_exp_f32_e32 v161, v23
	v_exp_f32_e32 v162, v24
	v_exp_f32_e32 v160, v25
	v_exp_f32_e32 v150, v26
	v_exp_f32_e32 v147, v27
	v_exp_f32_e32 v149, v28
	v_exp_f32_e32 v146, v34
	v_exp_f32_e32 v148, v35
	v_exp_f32_e32 v144, v32
	v_exp_f32_e32 v145, v33
	v_sub_f32_e32 v28, v14, v216
	v_sub_f32_e32 v27, v13, v216
	v_sub_f32_e32 v26, v12, v216
	v_sub_f32_e32 v25, v11, v216
	v_sub_f32_e32 v24, v10, v216
	v_sub_f32_e32 v23, v9, v216
	v_sub_f32_e32 v22, v8, v216
	v_sub_f32_e32 v21, v7, v216
	v_sub_f32_e32 v20, v6, v216
	v_sub_f32_e32 v19, v5, v216
	v_sub_f32_e32 v18, v4, v216
	v_sub_f32_e32 v17, v3, v216
	v_sub_f32_e32 v16, v2, v216
; ==============================================================================
; PASS 0 PROLOGUE CLUSTER P8: first softmax slice and K2 prefetch
; Computes row max/exp2 first half, synchronizes, and prefetches K tile 2.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s42
	s_lshl_b32 s50, s11, 8
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	buffer_load_dwordx4 v218, s[16:19], s50 offen lds
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s50 offen lds
; ==============================================================================
; PASS 0 MAIN LOOP PREHEADER: trip-count guard and loop-carry init
; Computes whether range(split_t0+3, split_t_end-1, step=2) executes,
; initializes m_row/l_row/O accumulators/P fragments, and dispatches to loop or epilogue.
; ==============================================================================
	s_mov_b64 s[18:19], -1
	s_add_u32 s4, s22, -1
	s_addc_u32 s5, s23, -1
	v_cmp_gt_u64_e64 s[34:35], s[22:23], 4
	s_and_b64 vcc, exec, s[34:35]
	v_lshlrev_b32_e32 v221, 1, v0
	s_cbranch_vccnz .LBB0_6
	v_add_u32_e32 v220, 0x4100, v221
	s_mov_b64 s[18:19], 0
	s_branch .LBB0_7
.LBB0_6:
.LBB0_7:
	s_add_i32 s25, s39, 0x8500
	s_lshl_b32 s44, s11, 1
	s_add_i32 s37, s39, 0xa700
	s_andn2_b64 vcc, exec, s[18:19]
	s_mul_i32 s45, s11, 0x180
	s_cbranch_vccnz .LBB0_16
	v_add_u32_e32 v220, 0x4100, v221
	v_add_u32_e32 v213, 0xc600, v221
	s_lshl_b32 s23, s44, 6
	s_lshl_b32 s52, s44, 7
	s_add_i32 s18, s24, s9
	v_add_u32_e32 v0, s18, v58
	v_subrev_u32_e32 v0, s46, v0
	v_sub_u32_e32 v0, v0, v222
	v_add_u32_e32 v224, 0xffffff60, v0
	s_lshl_b32 s53, s11, 9
	s_mov_b64 s[34:35], 3
	v_mov_b32_e32 v0, 0
	s_mov_b32 s56, 0
	s_movk_i32 s54, 0xc0
	v_mov_b32_e32 v223, 0
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	s_mov_b32 s55, 0x41000000
	v_mov_b32_e32 v225, 0xff800000
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v60, v0
	v_mov_b32_e32 v61, v0
	v_mov_b32_e32 v62, v0
	v_mov_b32_e32 v63, v0
	v_mov_b32_e32 v64, v0
	v_mov_b32_e32 v65, v0
	v_mov_b32_e32 v66, v0
	v_mov_b32_e32 v67, v0
	v_mov_b32_e32 v68, v0
	v_mov_b32_e32 v69, v0
	v_mov_b32_e32 v70, v0
	v_mov_b32_e32 v71, v0
	v_mov_b32_e32 v72, v0
	v_mov_b32_e32 v73, v0
	v_mov_b32_e32 v74, v0
	v_mov_b32_e32 v75, v0
	v_mov_b32_e32 v76, v0
	v_mov_b32_e32 v77, v0
	v_mov_b32_e32 v78, v0
	v_mov_b32_e32 v79, v0
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C0: memory stage A
; Prefetch V(j-2) to V buffer 1, read resident K buffer 1 into VGPRs, wait, then barrier.
; ==============================================================================
; ==============================================================================
; PASS 0 MAIN LOOP BODY START - steady-state software pipeline
; One iteration consumes two KV tiles. Clusters C0-C7 form the pipelined loop body.
; ==============================================================================
.LBB0_9:
	s_mov_b32 m0, s25
	s_add_i32 s57, s23, s56
	buffer_load_dwordx4 v218, s[12:15], s57 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s57 offen lds
	ds_read_b128 v[96:99], v215 offset:34048
	ds_read_b128 v[156:159], v215 offset:34080
	ds_read_b128 v[170:173], v215 offset:34560
	ds_read_b128 v[174:177], v215 offset:34592
	ds_read_b128 v[178:181], v215 offset:34112
	ds_read_b128 v[182:185], v215 offset:34144
	ds_read_b128 v[186:189], v215 offset:34624
	ds_read_b128 v[190:193], v215 offset:34656
	ds_read_b128 v[194:197], v215 offset:42368
	ds_read_b128 v[198:201], v215 offset:42400
	ds_read_b128 v[202:205], v215 offset:42880
	ds_read_b128 v[226:229], v215 offset:42912
	ds_read_b128 v[230:233], v215 offset:42432
	ds_read_b128 v[234:237], v215 offset:42464
	ds_read_b128 v[238:241], v215 offset:42944
	ds_read_b128 v[242:245], v215 offset:42976
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C1: compute stage A
; MMA0 produces v_s_1 while finishing v_p_0 exp2/sum/cast for P*V.
; ==============================================================================
	v_mfma_f32_32x32x16_bf16 v[80:95], v[96:99], v[112:115], 0
	v_exp_f32_e32 v151, v16
	v_exp_f32_e32 v152, v17
	v_exp_f32_e32 v153, v18
	v_mfma_f32_32x32x16_bf16 v[96:111], v[170:173], v[112:115], 0
	v_exp_f32_e32 v154, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[116:119], v[80:95]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v155, v24
	v_mfma_f32_32x32x16_bf16 v[96:111], v[174:177], v[116:119], v[96:111]
	v_exp_f32_e32 v156, v25
	v_exp_f32_e32 v157, v26
	v_exp_f32_e32 v158, v27
	v_mfma_f32_32x32x16_bf16 v[80:95], v[178:181], v[120:123], v[80:95]
	v_exp_f32_e32 v159, v28
	v_exp_f32_e32 v169, v29
	v_exp_f32_e32 v170, v30
	v_mfma_f32_32x32x16_bf16 v[96:111], v[186:189], v[120:123], v[96:111]
	v_exp_f32_e32 v31, v31
	v_mfma_f32_32x32x16_bf16 v[80:95], v[182:185], v[124:127], v[80:95]
	v_add_f32_e32 v16, v167, v168
	v_add_f32_e32 v16, v16, v164
	v_add_f32_e32 v16, v16, v166
	v_add_f32_e32 v16, v16, v163
	v_add_f32_e32 v16, v16, v165
	v_mfma_f32_32x32x16_bf16 v[96:111], v[190:193], v[124:127], v[96:111]
	v_add_f32_e32 v16, v16, v161
	v_add_f32_e32 v16, v16, v162
	v_add_f32_e32 v16, v16, v160
	v_add_f32_e32 v16, v16, v150
	v_add_f32_e32 v16, v16, v147
	v_mfma_f32_32x32x16_bf16 v[80:95], v[194:197], v[128:131], v[80:95]
	v_add_f32_e32 v16, v16, v149
	v_add_f32_e32 v16, v16, v146
	v_add_f32_e32 v16, v16, v148
	v_add_f32_e32 v16, v16, v144
	v_add_f32_e32 v16, v16, v145
	v_mfma_f32_32x32x16_bf16 v[96:111], v[202:205], v[128:131], v[96:111]
	v_add_f32_e32 v16, v16, v151
	v_add_f32_e32 v16, v16, v152
	v_add_f32_e32 v16, v16, v153
	v_add_f32_e32 v16, v16, v154
	v_add_f32_e32 v16, v16, v20
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[132:135], v[80:95]
	v_add_f32_e32 v16, v16, v21
	v_add_f32_e32 v16, v16, v22
	v_add_f32_e32 v16, v16, v23
	v_add_f32_e32 v16, v16, v155
	v_add_f32_e32 v16, v16, v156
	v_mfma_f32_32x32x16_bf16 v[96:111], v[226:229], v[132:135], v[96:111]
	v_add_f32_e32 v16, v16, v157
	v_add_f32_e32 v16, v16, v158
	v_add_f32_e32 v16, v16, v159
	v_add_f32_e32 v16, v16, v169
	v_add_f32_e32 v16, v16, v170
	v_mfma_f32_32x32x16_bf16 v[80:95], v[230:233], v[136:139], v[80:95]
	v_add_f32_e32 v16, v16, v31
	v_mov_b32_e32 v17, v16
	s_nop 1
	v_permlane32_swap_b32_e64 v16, v17 bound_ctrl:1
	v_add_f32_e32 v17, v223, v17
	v_add_f32_e32 v223, v17, v16
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v16, v167, v168
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[96:111], v[238:241], v[136:139], v[96:111]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v17, v164, v166
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v18, v163, v165
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v19, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v24, v151, v152
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v25, v153, v154
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[234:237], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v26, v20, v21
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v27, v22, v23
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v20, v160, v150
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v21, v147, v149
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v22, v146, v148
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[96:111], v[242:245], v[140:143], v[96:111]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v23, v144, v145
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v28, v155, v156
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v29, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v30, v159, v169
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v31, v170, v31
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C2: memory stage A
; Prefetch next K(j) to K buffer 1, read V buffer 0, apply causal mask for v_s_1.
; ==============================================================================
	s_barrier
	s_add_i32 s57, s45, s56
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v218, s[16:19], s57 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s57 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v220 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v220 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v220 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v220 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v220 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v220 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v220 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v220 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v220 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v220 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v220 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v220 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v220 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v220 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v220 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v220 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v220 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v220 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v220 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v220 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v220 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v220 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v220 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v220 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v220 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v220 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v220 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v220 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v220 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v220 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v220 offset:9536

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v220 offset:9664

	;;#ASMEND
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C3: compute stage A
; Raise priority, perform first P*V step, row-max v_s_1, lazy-rescale decision,
; finish remaining P*V steps, and start exp2 for v_s_1.
; ==============================================================================
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[0:15], v[192:195], v[16:19], v[0:15]
	v_max_f32_e32 v192, v80, v81
	v_max3_f32 v192, v192, v82, v83
	v_max3_f32 v192, v192, v84, v85
	v_max3_f32 v192, v192, v86, v87
	v_max3_f32 v192, v192, v88, v89
	v_max3_f32 v192, v192, v90, v91
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[16:19], v[32:47]
	v_max3_f32 v192, v192, v92, v93
	v_max3_f32 v192, v192, v94, v95
	v_max3_f32 v192, v192, v96, v97
	v_max3_f32 v192, v192, v98, v99
	v_max3_f32 v192, v192, v100, v101
	v_max3_f32 v192, v192, v102, v103
	v_mfma_f32_32x32x16_bf16 v[48:63], v[200:203], v[16:19], v[48:63]
	v_max3_f32 v192, v192, v104, v105
	v_max3_f32 v192, v192, v106, v107
	v_max3_f32 v192, v192, v108, v109
	v_max3_f32 v192, v192, v110, v111
	v_mov_b32_e32 v193, v192
	s_nop 1
	v_permlane32_swap_b32_e64 v192, v193 bound_ctrl:1
	v_max_f32_e32 v192, v192, v193
	v_mfma_f32_32x32x16_bf16 v[64:79], v[204:207], v[16:19], v[64:79]
	v_sub_f32_e32 v16, v192, v216
	v_cmp_ge_f32_e32 vcc, s55, v16
	s_cmp_eq_u64 vcc, exec
	s_cbranch_scc0 .LBB0_14
.LBB0_10:
	v_mfma_f32_32x32x16_bf16 v[0:15], v[188:191], v[20:23], v[0:15]
	v_sub_f32_e32 v16, v80, v216
	v_sub_f32_e32 v17, v81, v216
	v_sub_f32_e32 v18, v82, v216
	v_sub_f32_e32 v19, v83, v216
	v_sub_f32_e32 v80, v84, v216
	v_sub_f32_e32 v81, v85, v216
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[20:23], v[32:47]
	v_sub_f32_e32 v82, v86, v216
	v_sub_f32_e32 v83, v87, v216
	v_sub_f32_e32 v84, v88, v216
	v_sub_f32_e32 v85, v89, v216
	v_sub_f32_e32 v86, v90, v216
	v_sub_f32_e32 v87, v91, v216
	v_mfma_f32_32x32x16_bf16 v[48:63], v[180:183], v[20:23], v[48:63]
	v_sub_f32_e32 v88, v92, v216
	v_sub_f32_e32 v89, v93, v216
	v_sub_f32_e32 v90, v94, v216
	v_sub_f32_e32 v91, v95, v216
	v_sub_f32_e32 v96, v96, v216
	v_sub_f32_e32 v97, v97, v216
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[20:23], v[64:79]
	v_sub_f32_e32 v20, v98, v216
	v_sub_f32_e32 v98, v99, v216
	v_sub_f32_e32 v99, v100, v216
	v_sub_f32_e32 v100, v101, v216
	v_sub_f32_e32 v101, v102, v216
	v_sub_f32_e32 v102, v103, v216
	v_mfma_f32_32x32x16_bf16 v[0:15], v[152:155], v[24:27], v[0:15]
	v_sub_f32_e32 v103, v104, v216
	v_sub_f32_e32 v104, v105, v216
	v_sub_f32_e32 v105, v106, v216
	v_sub_f32_e32 v106, v107, v216
	v_sub_f32_e32 v107, v108, v216
	v_sub_f32_e32 v108, v109, v216
	v_mfma_f32_32x32x16_bf16 v[32:47], v[160:163], v[24:27], v[32:47]
	v_sub_f32_e32 v109, v110, v216
	v_sub_f32_e32 v110, v111, v216
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[24:27], v[48:63]
	v_exp_f32_e32 v111, v16
	v_exp_f32_e32 v152, v17
	v_exp_f32_e32 v153, v18
	v_mfma_f32_32x32x16_bf16 v[64:79], v[172:175], v[24:27], v[64:79]
	v_exp_f32_e32 v154, v19
	v_exp_f32_e32 v155, v80
	v_exp_f32_e32 v160, v81
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[28:31], v[0:15]
	v_exp_f32_e32 v144, v82
	v_exp_f32_e32 v145, v83
	v_exp_f32_e32 v146, v84
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[28:31], v[32:47]
	v_exp_f32_e32 v147, v85
	v_exp_f32_e32 v148, v86
	v_exp_f32_e32 v149, v87
	v_mfma_f32_32x32x16_bf16 v[48:63], v[156:159], v[28:31], v[48:63]
	v_exp_f32_e32 v150, v88
	v_exp_f32_e32 v151, v89
	v_exp_f32_e32 v156, v90
	v_mfma_f32_32x32x16_bf16 v[64:79], v[164:167], v[28:31], v[64:79]
	v_exp_f32_e32 v157, v91
	s_setprio 0
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C4: memory stage B
; Lower priority, prefetch V(j-1) to V buffer 0, read K buffer 0, wait, then barrier.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s39
	s_add_i32 s57, s52, s56
	buffer_load_dwordx4 v218, s[12:15], s57 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s57 offen lds
	ds_read_b128 v[16:19], v215
	ds_read_b128 v[164:167], v215 offset:32
	ds_read_b128 v[168:171], v215 offset:512
	ds_read_b128 v[172:175], v215 offset:544
	ds_read_b128 v[176:179], v215 offset:64
	ds_read_b128 v[180:183], v215 offset:96
	ds_read_b128 v[184:187], v215 offset:576
	ds_read_b128 v[188:191], v215 offset:608
	ds_read_b128 v[192:195], v215 offset:8320
	ds_read_b128 v[196:199], v215 offset:8352
	ds_read_b128 v[200:203], v215 offset:8832
	ds_read_b128 v[204:207], v215 offset:8864
	ds_read_b128 v[226:229], v215 offset:8384
	ds_read_b128 v[230:233], v215 offset:8416
	ds_read_b128 v[234:237], v215 offset:8896
	ds_read_b128 v[238:241], v215 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C5: compute stage B
; MMA0 produces v_s_0 while finishing v_p_1 exp2/sum/cast for P*V.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[80:95], v[16:19], v[112:115], 0
	v_exp_f32_e32 v158, v96
	v_exp_f32_e32 v159, v97
	v_exp_f32_e32 v161, v20
	v_mfma_f32_32x32x16_bf16 v[16:31], v[168:171], v[112:115], 0
	v_exp_f32_e32 v162, v98
	v_exp_f32_e32 v163, v99
	v_exp_f32_e32 v100, v100
	v_mfma_f32_32x32x16_bf16 v[80:95], v[164:167], v[116:119], v[80:95]
	v_exp_f32_e32 v101, v101
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v164, v103
	v_mfma_f32_32x32x16_bf16 v[16:31], v[172:175], v[116:119], v[16:31]
	v_exp_f32_e32 v165, v104
	v_exp_f32_e32 v166, v105
	v_exp_f32_e32 v167, v106
	v_mfma_f32_32x32x16_bf16 v[80:95], v[176:179], v[120:123], v[80:95]
	v_exp_f32_e32 v168, v107
	v_exp_f32_e32 v169, v108
	v_exp_f32_e32 v170, v109
	v_mfma_f32_32x32x16_bf16 v[16:31], v[184:187], v[120:123], v[16:31]
	v_exp_f32_e32 v171, v110
	v_mfma_f32_32x32x16_bf16 v[80:95], v[180:183], v[124:127], v[80:95]
	v_add_f32_e32 v96, v111, v152
	v_add_f32_e32 v96, v96, v153
	v_add_f32_e32 v96, v96, v154
	v_add_f32_e32 v96, v96, v155
	v_add_f32_e32 v96, v96, v160
	v_mfma_f32_32x32x16_bf16 v[16:31], v[188:191], v[124:127], v[16:31]
	v_add_f32_e32 v96, v96, v144
	v_add_f32_e32 v96, v96, v145
	v_add_f32_e32 v96, v96, v146
	v_add_f32_e32 v96, v96, v147
	v_add_f32_e32 v96, v96, v148
	v_mfma_f32_32x32x16_bf16 v[80:95], v[192:195], v[128:131], v[80:95]
	v_add_f32_e32 v96, v96, v149
	v_add_f32_e32 v96, v96, v150
	v_add_f32_e32 v96, v96, v151
	v_add_f32_e32 v96, v96, v156
	v_add_f32_e32 v96, v96, v157
	v_mfma_f32_32x32x16_bf16 v[16:31], v[200:203], v[128:131], v[16:31]
	v_add_f32_e32 v96, v96, v158
	v_add_f32_e32 v96, v96, v159
	v_add_f32_e32 v96, v96, v161
	v_add_f32_e32 v96, v96, v162
	v_add_f32_e32 v96, v96, v163
	v_mfma_f32_32x32x16_bf16 v[80:95], v[196:199], v[132:135], v[80:95]
	v_add_f32_e32 v96, v96, v100
	v_add_f32_e32 v96, v96, v101
	v_add_f32_e32 v96, v96, v102
	v_add_f32_e32 v96, v96, v164
	v_add_f32_e32 v96, v96, v165
	v_mfma_f32_32x32x16_bf16 v[16:31], v[204:207], v[132:135], v[16:31]
	v_add_f32_e32 v96, v96, v166
	v_add_f32_e32 v96, v96, v167
	v_add_f32_e32 v96, v96, v168
	v_add_f32_e32 v96, v96, v169
	v_add_f32_e32 v96, v96, v170
	v_mfma_f32_32x32x16_bf16 v[80:95], v[226:229], v[136:139], v[80:95]
	v_add_f32_e32 v226, v96, v171
	v_mov_b32_e32 v227, v226
	s_nop 1
	v_permlane32_swap_b32_e64 v226, v227 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v111, v152
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v153, v154
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v155, v160
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[16:31], v[234:237], v[136:139], v[16:31]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v144, v145
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v158, v159
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v163, v100
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v101, v102
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[230:233], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v146, v147
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v148, v149
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v150, v151
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v156, v157
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v164, v165
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[16:31], v[238:241], v[140:143], v[16:31]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v166, v167
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v168, v169
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v170, v171
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C6: memory stage B
; Prefetch K(j+1) to K buffer 0, read V buffer 1, apply causal mask for v_s_0.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s42
	s_add_i32 s56, s53, s56
	buffer_load_dwordx4 v218, s[16:19], s56 offen lds
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s56 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v213 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v213 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v213 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v213 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v213 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v213 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v213 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v213 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v213 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v213 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v213 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v213 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v213 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v213 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v213 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v213 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v213 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v213 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v213 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v213 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v213 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v213 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v213 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v213 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v213 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v213 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v213 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v213 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v213 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v213 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v213 offset:9536

	;;#ASMEND
	s_cmp_ge_i32 s7, s54
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v213 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_12
	v_add_u32_e32 v228, 32, v224
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v228, 0
	v_cmp_lt_i32_e64 s[60:61], v228, 1
	v_cndmask_b32_e64 v80, v80, v225, s[58:59]
	v_cndmask_b32_e64 v81, v81, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v228, 2
	v_cmp_lt_i32_e64 s[60:61], v228, 3
	v_cndmask_b32_e64 v82, v82, v225, s[58:59]
	v_cndmask_b32_e64 v83, v83, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v228, 8
	v_cmp_lt_i32_e64 s[60:61], v228, 9
	v_cndmask_b32_e64 v84, v84, v225, s[58:59]
	v_cndmask_b32_e64 v85, v85, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v228, 10
	v_cmp_lt_i32_e64 s[60:61], v228, 11
	v_cndmask_b32_e64 v86, v86, v225, s[58:59]
	v_cndmask_b32_e64 v87, v87, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v228, 16
	v_cmp_lt_i32_e64 s[60:61], v228, 17
	v_cndmask_b32_e64 v88, v88, v225, s[58:59]
	v_cndmask_b32_e64 v89, v89, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v228, 18
	v_cmp_lt_i32_e64 s[60:61], v228, 19
	v_cndmask_b32_e64 v90, v90, v225, s[58:59]
	v_cndmask_b32_e64 v91, v91, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v228, 24
	v_cmp_lt_i32_e64 s[60:61], v228, 25
	v_cndmask_b32_e64 v92, v92, v225, s[58:59]
	v_cndmask_b32_e64 v93, v93, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v228, 26
	v_cmp_lt_i32_e64 s[60:61], v228, 27
	v_cndmask_b32_e64 v94, v94, v225, s[58:59]
	v_cndmask_b32_e64 v95, v95, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v224, 0
	v_cmp_lt_i32_e64 s[60:61], v224, 1
	v_cndmask_b32_e64 v16, v16, v225, s[58:59]
	v_cndmask_b32_e64 v17, v17, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v224, 2
	v_cmp_lt_i32_e64 s[60:61], v224, 3
	v_cndmask_b32_e64 v18, v18, v225, s[58:59]
	v_cndmask_b32_e64 v19, v19, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v224, 8
	v_cmp_lt_i32_e64 s[60:61], v224, 9
	v_cndmask_b32_e64 v20, v20, v225, s[58:59]
	v_cndmask_b32_e64 v21, v21, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v224, 10
	v_cmp_lt_i32_e64 s[60:61], v224, 11
	v_cndmask_b32_e64 v22, v22, v225, s[58:59]
	v_cndmask_b32_e64 v23, v23, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v224, 16
	v_cmp_lt_i32_e64 s[60:61], v224, 17
	v_cndmask_b32_e64 v24, v24, v225, s[58:59]
	v_cndmask_b32_e64 v25, v25, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v224, 18
	v_cmp_lt_i32_e64 s[60:61], v224, 19
	v_cndmask_b32_e64 v26, v26, v225, s[58:59]
	v_cndmask_b32_e64 v27, v27, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v224, 24
	v_cmp_lt_i32_e64 s[60:61], v224, 25
	v_cndmask_b32_e64 v28, v28, v225, s[58:59]
	v_cndmask_b32_e64 v29, v29, v225, s[60:61]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[58:59], v224, 26
	v_cmp_lt_i32_e64 s[60:61], v224, 27
	v_cndmask_b32_e64 v30, v30, v225, s[58:59]
	v_cndmask_b32_e64 v31, v31, v225, s[60:61]
	;;#ASMEND
.LBB0_12:
	v_add_f32_e32 v223, v223, v227
	v_add_f32_e32 v223, v223, v226
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C7: compute stage B / loop yield
; Raise priority, update O/m/l with v_p_1*V, start next v_p_0, then lower priority and sync.
; The following scalar branch is the loop back-edge to C0 or exit to epilogue.
; ==============================================================================
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[0:15], v[204:207], v[96:99], v[0:15]
	v_max_f32_e32 v204, v80, v81
	v_max3_f32 v204, v204, v82, v83
	v_max3_f32 v204, v204, v84, v85
	v_max3_f32 v204, v204, v86, v87
	v_max3_f32 v204, v204, v88, v89
	v_max3_f32 v204, v204, v90, v91
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[96:99], v[32:47]
	v_max3_f32 v196, v204, v92, v93
	v_max3_f32 v196, v196, v94, v95
	v_max3_f32 v196, v196, v16, v17
	v_max3_f32 v196, v196, v18, v19
	v_max3_f32 v196, v196, v20, v21
	v_max3_f32 v196, v196, v22, v23
	v_mfma_f32_32x32x16_bf16 v[48:63], v[200:203], v[96:99], v[48:63]
	v_max3_f32 v196, v196, v24, v25
	v_max3_f32 v196, v196, v26, v27
	v_max3_f32 v196, v196, v28, v29
	v_max3_f32 v196, v196, v30, v31
	v_mov_b32_e32 v197, v196
	s_nop 1
	v_permlane32_swap_b32_e64 v196, v197 bound_ctrl:1
	v_max_f32_e32 v196, v196, v197
	v_sub_f32_e32 v197, v196, v216
	v_cmp_ge_f32_e32 vcc, s55, v197
	s_cmp_eq_u64 vcc, exec
	v_mfma_f32_32x32x16_bf16 v[64:79], v[192:195], v[96:99], v[64:79]
	s_cbranch_scc0 .LBB0_15
.LBB0_13:
	v_mfma_f32_32x32x16_bf16 v[0:15], v[188:191], v[100:103], v[0:15]
	v_sub_f32_e32 v80, v80, v216
	v_sub_f32_e32 v81, v81, v216
	v_sub_f32_e32 v82, v82, v216
	v_sub_f32_e32 v83, v83, v216
	v_sub_f32_e32 v84, v84, v216
	v_mfma_f32_32x32x16_bf16 v[32:47], v[176:179], v[100:103], v[32:47]
	v_sub_f32_e32 v85, v85, v216
	v_sub_f32_e32 v86, v86, v216
	v_sub_f32_e32 v87, v87, v216
	v_sub_f32_e32 v88, v88, v216
	v_sub_f32_e32 v89, v89, v216
	v_mfma_f32_32x32x16_bf16 v[48:63], v[180:183], v[100:103], v[48:63]
	v_sub_f32_e32 v90, v90, v216
	v_sub_f32_e32 v91, v91, v216
	v_sub_f32_e32 v92, v92, v216
	v_sub_f32_e32 v93, v93, v216
	v_sub_f32_e32 v94, v94, v216
	v_mfma_f32_32x32x16_bf16 v[64:79], v[184:187], v[100:103], v[64:79]
	v_sub_f32_e32 v95, v95, v216
	v_sub_f32_e32 v16, v16, v216
	v_sub_f32_e32 v17, v17, v216
	v_sub_f32_e32 v18, v18, v216
	v_sub_f32_e32 v19, v19, v216
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[104:107], v[0:15]
	v_sub_f32_e32 v20, v20, v216
	v_sub_f32_e32 v21, v21, v216
	v_sub_f32_e32 v22, v22, v216
	v_sub_f32_e32 v23, v23, v216
	v_sub_f32_e32 v24, v24, v216
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[104:107], v[32:47]
	v_sub_f32_e32 v25, v25, v216
	v_sub_f32_e32 v26, v26, v216
	v_sub_f32_e32 v27, v27, v216
	v_sub_f32_e32 v28, v28, v216
	v_sub_f32_e32 v29, v29, v216
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[104:107], v[48:63]
	v_exp_f32_e32 v167, v80
	v_exp_f32_e32 v168, v81
	v_exp_f32_e32 v164, v82
	v_mfma_f32_32x32x16_bf16 v[64:79], v[172:175], v[104:107], v[64:79]
	v_exp_f32_e32 v166, v83
	v_exp_f32_e32 v163, v84
	v_exp_f32_e32 v165, v85
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[108:111], v[0:15]
	v_exp_f32_e32 v161, v86
	v_exp_f32_e32 v162, v87
	v_exp_f32_e32 v160, v88
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[108:111], v[32:47]
	v_exp_f32_e32 v150, v89
	v_exp_f32_e32 v147, v90
	v_exp_f32_e32 v149, v91
	v_mfma_f32_32x32x16_bf16 v[48:63], v[152:155], v[108:111], v[48:63]
	v_exp_f32_e32 v146, v92
	v_exp_f32_e32 v148, v93
	v_exp_f32_e32 v144, v94
	v_mfma_f32_32x32x16_bf16 v[64:79], v[156:159], v[108:111], v[64:79]
	v_exp_f32_e32 v145, v95
	v_sub_f32_e32 v30, v30, v216
	v_sub_f32_e32 v31, v31, v216
	s_setprio 0
	s_barrier
; ==============================================================================
; PASS 0 MAIN LOOP BACK-EDGE
; Increments j by 2, advances rolling VMEM/LDS offsets, and branches to C0 or epilogue.
; ==============================================================================
	s_add_u32 s34, s34, 2
	s_addc_u32 s35, s35, 0
	v_add_u32_e32 v224, 0xffffff80, v224
	s_addk_i32 s54, 0x80
	v_mov_b64_e32 v[80:81], s[4:5]
	v_cmp_lt_i64_e32 vcc, s[34:35], v[80:81]
	s_mov_b32 s56, s57
	s_cbranch_vccnz .LBB0_9
	s_branch .LBB0_17
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C3 LAZY-RESCALE SLOW PATH
; Runs when m_tile_max is too far from m_row; rescales O, P, l_row, updates m_row,
; then rejoins C3 at .LBB0_10.
; ==============================================================================
.LBB0_14:
	v_sub_f32_e32 v16, v216, v192
	v_exp_f32_e32 v16, v16
	s_nop 0
	v_pk_mul_f32 v[14:15], v[16:17], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[16:17], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[16:17], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[16:17], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[16:17], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[16:17], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[16:17], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[16:17], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[16:17], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[16:17], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[16:17], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[16:17], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[16:17], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[16:17], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[16:17], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[16:17], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[16:17], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[16:17], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[16:17], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[16:17], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[16:17], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[16:17], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[16:17], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[16:17], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[16:17], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[16:17], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[16:17], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[16:17], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[16:17], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[16:17], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[16:17], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[16:17], v[64:65] op_sel_hi:[0,1]
	v_and_b32_e32 v19, 0xffff0000, v31
	v_lshlrev_b32_e32 v18, 16, v31
	v_and_b32_e32 v31, 0xffff0000, v30
	v_lshlrev_b32_e32 v30, 16, v30
	v_and_b32_e32 v195, 0xffff0000, v29
	v_lshlrev_b32_e32 v194, 16, v29
	v_and_b32_e32 v29, 0xffff0000, v28
	v_lshlrev_b32_e32 v28, 16, v28
	v_and_b32_e32 v197, 0xffff0000, v27
	v_lshlrev_b32_e32 v196, 16, v27
	v_and_b32_e32 v27, 0xffff0000, v26
	v_lshlrev_b32_e32 v26, 16, v26
	v_and_b32_e32 v199, 0xffff0000, v25
	v_lshlrev_b32_e32 v198, 16, v25
	v_and_b32_e32 v25, 0xffff0000, v24
	v_lshlrev_b32_e32 v24, 16, v24
	v_and_b32_e32 v201, 0xffff0000, v23
	v_lshlrev_b32_e32 v200, 16, v23
	v_and_b32_e32 v23, 0xffff0000, v22
	v_lshlrev_b32_e32 v22, 16, v22
	v_and_b32_e32 v203, 0xffff0000, v21
	v_lshlrev_b32_e32 v202, 16, v21
	v_and_b32_e32 v21, 0xffff0000, v20
	v_lshlrev_b32_e32 v20, 16, v20
	v_pk_mul_f32 v[204:205], v[16:17], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[16:17], v[202:203] op_sel_hi:[0,1]
	v_pk_mul_f32 v[202:203], v[16:17], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[16:17], v[200:201] op_sel_hi:[0,1]
	v_pk_mul_f32 v[200:201], v[16:17], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[16:17], v[198:199] op_sel_hi:[0,1]
	v_pk_mul_f32 v[198:199], v[16:17], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[16:17], v[196:197] op_sel_hi:[0,1]
	v_pk_mul_f32 v[196:197], v[16:17], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[16:17], v[194:195] op_sel_hi:[0,1]
	v_pk_mul_f32 v[194:195], v[16:17], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[16:17], v[18:19] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v31, v18, v19
	v_cvt_pk_bf16_f32 v30, v194, v195
	v_cvt_pk_bf16_f32 v29, v28, v29
	v_cvt_pk_bf16_f32 v28, v196, v197
	v_cvt_pk_bf16_f32 v27, v26, v27
	v_cvt_pk_bf16_f32 v26, v198, v199
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v200, v201
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v202, v203
	v_cvt_pk_bf16_f32 v21, v20, v21
	v_cvt_pk_bf16_f32 v20, v204, v205
	v_mul_f32_e32 v223, v16, v223
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v216, v192
	s_branch .LBB0_10
; ==============================================================================
; PASS 0 MAIN LOOP CLUSTER C7 LAZY-RESCALE SLOW PATH
; Second lazy-rescale slow path for the B half of the loop; rejoins C7 at .LBB0_13.
; ==============================================================================
.LBB0_15:
	v_sub_f32_e32 v96, v216, v196
	v_exp_f32_e32 v96, v96
	s_nop 0
	v_pk_mul_f32 v[14:15], v[96:97], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[96:97], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[96:97], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[96:97], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[96:97], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[96:97], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[96:97], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[96:97], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[96:97], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[96:97], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[96:97], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[96:97], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[96:97], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[96:97], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[96:97], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[96:97], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[96:97], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[96:97], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[96:97], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[96:97], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[96:97], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[96:97], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[96:97], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[96:97], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[78:79], v[96:97], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[96:97], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[74:75], v[96:97], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[72:73], v[96:97], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[70:71], v[96:97], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[68:69], v[96:97], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[66:67], v[96:97], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[64:65], v[96:97], v[64:65] op_sel_hi:[0,1]
	v_and_b32_e32 v99, 0xffff0000, v111
	v_lshlrev_b32_e32 v98, 16, v111
	v_and_b32_e32 v111, 0xffff0000, v110
	v_lshlrev_b32_e32 v110, 16, v110
	v_and_b32_e32 v193, 0xffff0000, v109
	v_lshlrev_b32_e32 v192, 16, v109
	v_and_b32_e32 v109, 0xffff0000, v108
	v_lshlrev_b32_e32 v108, 16, v108
	v_and_b32_e32 v195, 0xffff0000, v107
	v_lshlrev_b32_e32 v194, 16, v107
	v_and_b32_e32 v107, 0xffff0000, v106
	v_lshlrev_b32_e32 v106, 16, v106
	v_and_b32_e32 v199, 0xffff0000, v105
	v_lshlrev_b32_e32 v198, 16, v105
	v_and_b32_e32 v105, 0xffff0000, v104
	v_lshlrev_b32_e32 v104, 16, v104
	v_and_b32_e32 v201, 0xffff0000, v103
	v_lshlrev_b32_e32 v200, 16, v103
	v_and_b32_e32 v103, 0xffff0000, v102
	v_lshlrev_b32_e32 v102, 16, v102
	v_and_b32_e32 v203, 0xffff0000, v101
	v_lshlrev_b32_e32 v202, 16, v101
	v_and_b32_e32 v101, 0xffff0000, v100
	v_lshlrev_b32_e32 v100, 16, v100
	v_pk_mul_f32 v[204:205], v[96:97], v[100:101] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[96:97], v[202:203] op_sel_hi:[0,1]
	v_pk_mul_f32 v[202:203], v[96:97], v[102:103] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[96:97], v[200:201] op_sel_hi:[0,1]
	v_pk_mul_f32 v[200:201], v[96:97], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[96:97], v[198:199] op_sel_hi:[0,1]
	v_pk_mul_f32 v[198:199], v[96:97], v[106:107] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[96:97], v[194:195] op_sel_hi:[0,1]
	v_pk_mul_f32 v[194:195], v[96:97], v[108:109] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[96:97], v[192:193] op_sel_hi:[0,1]
	v_pk_mul_f32 v[192:193], v[96:97], v[110:111] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[96:97], v[98:99] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v111, v98, v99
	v_cvt_pk_bf16_f32 v110, v192, v193
	v_cvt_pk_bf16_f32 v109, v108, v109
	v_cvt_pk_bf16_f32 v108, v194, v195
	v_cvt_pk_bf16_f32 v107, v106, v107
	v_cvt_pk_bf16_f32 v106, v198, v199
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v200, v201
	v_cvt_pk_bf16_f32 v103, v102, v103
	v_cvt_pk_bf16_f32 v102, v202, v203
	v_cvt_pk_bf16_f32 v101, v100, v101
	v_cvt_pk_bf16_f32 v100, v204, v205
	v_mul_f32_e32 v223, v96, v223
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v216, v196
	s_branch .LBB0_13
; ==============================================================================
; PASS 0 EPILOGUE ZERO-ITER ENTRY
; If the main loop had zero iterations, materialize zero O accumulators and enter the epilogue drain.
; ==============================================================================
.LBB0_16:
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v14, v15
	v_mov_b32_e32 v13, v15
	v_mov_b32_e32 v12, v15
	v_mov_b32_e32 v11, v15
	v_mov_b32_e32 v10, v15
	v_mov_b32_e32 v9, v15
	v_mov_b32_e32 v8, v15
	v_mov_b32_e32 v7, v15
	v_mov_b32_e32 v6, v15
	v_mov_b32_e32 v5, v15
	v_mov_b32_e32 v4, v15
	v_mov_b32_e32 v3, v15
	v_mov_b32_e32 v2, v15
	v_mov_b32_e32 v1, v15
	v_mov_b32_e32 v0, v15
	v_mov_b32_e32 v47, v15
	v_mov_b32_e32 v46, v15
	v_mov_b32_e32 v45, v15
	v_mov_b32_e32 v44, v15
	v_mov_b32_e32 v43, v15
	v_mov_b32_e32 v42, v15
	v_mov_b32_e32 v41, v15
	v_mov_b32_e32 v40, v15
	v_mov_b32_e32 v39, v15
	v_mov_b32_e32 v38, v15
	v_mov_b32_e32 v37, v15
	v_mov_b32_e32 v36, v15
	v_mov_b32_e32 v35, v15
	v_mov_b32_e32 v34, v15
	v_mov_b32_e32 v33, v15
	v_mov_b32_e32 v32, v15
	v_mov_b32_e32 v63, v15
	v_mov_b32_e32 v62, v15
	v_mov_b32_e32 v61, v15
	v_mov_b32_e32 v60, v15
	v_mov_b32_e32 v59, v15
	v_mov_b32_e32 v58, v15
	v_mov_b32_e32 v57, v15
	v_mov_b32_e32 v56, v15
	v_mov_b32_e32 v55, v15
	v_mov_b32_e32 v54, v15
	v_mov_b32_e32 v53, v15
	v_mov_b32_e32 v52, v15
	v_mov_b32_e32 v51, v15
	v_mov_b32_e32 v50, v15
	v_mov_b32_e32 v49, v15
	v_mov_b32_e32 v48, v15
	v_mov_b32_e32 v79, v15
	v_mov_b32_e32 v78, v15
	v_mov_b32_e32 v77, v15
	v_mov_b32_e32 v76, v15
	v_mov_b32_e32 v75, v15
	v_mov_b32_e32 v74, v15
	v_mov_b32_e32 v73, v15
	v_mov_b32_e32 v72, v15
	v_mov_b32_e32 v71, v15
	v_mov_b32_e32 v70, v15
	v_mov_b32_e32 v69, v15
	v_mov_b32_e32 v68, v15
	v_mov_b32_e32 v67, v15
	v_mov_b32_e32 v66, v15
	v_mov_b32_e32 v65, v15
	v_mov_b32_e32 v64, v15
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E0: memory drain A
; Prefetch V(max_m3), read K buffer 1, wait, and synchronize.
; ==============================================================================
; ==============================================================================
; PASS 0 EPILOGUE START - drain final resident tiles
; Consumes the final three KV tiles left by the prologue/main-loop pipeline.
; ==============================================================================
.LBB0_17:
	v_mov_b32_e32 v213, v212
	s_mov_b32 m0, s25
	s_lshl_b32 s5, s22, 6
	s_add_i32 s18, s5, 0xffffff40
	s_mul_i32 s23, s44, s18
	buffer_load_dwordx4 v218, s[12:15], s23 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s23 offen lds
	ds_read_b128 v[96:99], v215 offset:34048
	ds_read_b128 v[156:159], v215 offset:34080
	ds_read_b128 v[170:173], v215 offset:34560
	ds_read_b128 v[174:177], v215 offset:34592
	ds_read_b128 v[178:181], v215 offset:34112
	ds_read_b128 v[182:185], v215 offset:34144
	ds_read_b128 v[186:189], v215 offset:34624
	ds_read_b128 v[190:193], v215 offset:34656
	ds_read_b128 v[194:197], v215 offset:42368
	ds_read_b128 v[198:201], v215 offset:42400
	ds_read_b128 v[202:205], v215 offset:42880
	ds_read_b128 v[224:227], v215 offset:42912
	ds_read_b128 v[228:231], v215 offset:42432
	ds_read_b128 v[232:235], v215 offset:42464
	ds_read_b128 v[236:239], v215 offset:42944
	ds_read_b128 v[240:243], v215 offset:42976
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E1: compute drain A
; MMA0 -> v_s_1; finish v_p_0 softmax second half and accumulate l_row.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[80:95], v[96:99], v[112:115], 0
	v_exp_f32_e32 v151, v16
	v_exp_f32_e32 v152, v17
	v_exp_f32_e32 v153, v18
	v_mfma_f32_32x32x16_bf16 v[96:111], v[170:173], v[112:115], 0
	v_exp_f32_e32 v154, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[116:119], v[80:95]
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v155, v24
	v_mfma_f32_32x32x16_bf16 v[96:111], v[174:177], v[116:119], v[96:111]
	v_exp_f32_e32 v156, v25
	v_exp_f32_e32 v157, v26
	v_exp_f32_e32 v158, v27
	v_mfma_f32_32x32x16_bf16 v[80:95], v[178:181], v[120:123], v[80:95]
	v_exp_f32_e32 v159, v28
	v_exp_f32_e32 v169, v29
	v_exp_f32_e32 v170, v30
	v_mfma_f32_32x32x16_bf16 v[96:111], v[186:189], v[120:123], v[96:111]
	v_exp_f32_e32 v31, v31
	v_mfma_f32_32x32x16_bf16 v[80:95], v[182:185], v[124:127], v[80:95]
	v_add_f32_e32 v16, v167, v168
	v_add_f32_e32 v16, v16, v164
	v_add_f32_e32 v16, v16, v166
	v_add_f32_e32 v16, v16, v163
	v_add_f32_e32 v16, v16, v165
	v_mfma_f32_32x32x16_bf16 v[96:111], v[190:193], v[124:127], v[96:111]
	v_add_f32_e32 v16, v16, v161
	v_add_f32_e32 v16, v16, v162
	v_add_f32_e32 v16, v16, v160
	v_add_f32_e32 v16, v16, v150
	v_add_f32_e32 v16, v16, v147
	v_mfma_f32_32x32x16_bf16 v[80:95], v[194:197], v[128:131], v[80:95]
	v_add_f32_e32 v16, v16, v149
	v_add_f32_e32 v16, v16, v146
	v_add_f32_e32 v16, v16, v148
	v_add_f32_e32 v16, v16, v144
	v_add_f32_e32 v16, v16, v145
	v_mfma_f32_32x32x16_bf16 v[96:111], v[202:205], v[128:131], v[96:111]
	v_add_f32_e32 v16, v16, v151
	v_add_f32_e32 v16, v16, v152
	v_add_f32_e32 v16, v16, v153
	v_add_f32_e32 v16, v16, v154
	v_add_f32_e32 v16, v16, v20
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[132:135], v[80:95]
	v_add_f32_e32 v16, v16, v21
	v_add_f32_e32 v16, v16, v22
	v_add_f32_e32 v16, v16, v23
	v_add_f32_e32 v16, v16, v155
	v_add_f32_e32 v16, v16, v156
	v_mfma_f32_32x32x16_bf16 v[96:111], v[224:227], v[132:135], v[96:111]
	v_add_f32_e32 v16, v16, v157
	v_add_f32_e32 v16, v16, v158
	v_add_f32_e32 v16, v16, v159
	v_add_f32_e32 v16, v16, v169
	v_add_f32_e32 v16, v16, v170
	v_mfma_f32_32x32x16_bf16 v[80:95], v[228:231], v[136:139], v[80:95]
	v_add_f32_e32 v224, v16, v31
	v_mov_b32_e32 v225, v224
	s_nop 1
	v_permlane32_swap_b32_e64 v224, v225 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v16, v167, v168
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v17, v164, v166
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v18, v163, v165
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[96:111], v[236:239], v[136:139], v[96:111]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v19, v161, v162
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v24, v151, v152
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v25, v153, v154
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v26, v20, v21
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v27, v22, v23
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[232:235], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v20, v160, v150
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v21, v147, v149
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v22, v146, v148
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v23, v144, v145
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v28, v155, v156
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[96:111], v[240:243], v[140:143], v[96:111]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v29, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v30, v159, v169
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v31, v170, v31
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E2: memory drain A
; Prefetch K(max_m1), read V buffer 0, causally mask v_s_1.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s41
	s_lshl_b32 s4, s4, 6
	s_mul_i32 s22, s44, s4
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	buffer_load_dwordx4 v218, s[16:19], s22 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s22 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v220 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v220 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v220 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v220 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v220 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v220 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v220 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v220 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v220 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v220 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v220 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v220 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v220 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v220 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v220 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v220 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v220 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v220 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v220 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v220 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v220 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v220 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v220 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v220 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v220 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v220 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v220 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v220 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v220 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v220 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v220 offset:9536

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v220 offset:9664

	;;#ASMEND
	s_add_i32 s18, s5, 0xffffff80
	s_cmp_ge_i32 s7, s18
	v_lshl_or_b32 v228, v208, 2, s5
	s_cbranch_scc1 .LBB0_19
	v_sub_u32_e32 v226, s3, v228
	v_add_u32_e32 v226, v226, v211
	v_add_u32_e32 v227, 0xc0, v226
	v_add_u32_e32 v226, 0xa0, v226
	v_mov_b32_e32 v229, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v227, 0
	v_cmp_lt_i32_e64 s[34:35], v227, 1
	v_cndmask_b32_e64 v80, v80, v229, s[18:19]
	v_cndmask_b32_e64 v81, v81, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v227, 2
	v_cmp_lt_i32_e64 s[34:35], v227, 3
	v_cndmask_b32_e64 v82, v82, v229, s[18:19]
	v_cndmask_b32_e64 v83, v83, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v227, 8
	v_cmp_lt_i32_e64 s[34:35], v227, 9
	v_cndmask_b32_e64 v84, v84, v229, s[18:19]
	v_cndmask_b32_e64 v85, v85, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v227, 10
	v_cmp_lt_i32_e64 s[34:35], v227, 11
	v_cndmask_b32_e64 v86, v86, v229, s[18:19]
	v_cndmask_b32_e64 v87, v87, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v227, 16
	v_cmp_lt_i32_e64 s[34:35], v227, 17
	v_cndmask_b32_e64 v88, v88, v229, s[18:19]
	v_cndmask_b32_e64 v89, v89, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v227, 18
	v_cmp_lt_i32_e64 s[34:35], v227, 19
	v_cndmask_b32_e64 v90, v90, v229, s[18:19]
	v_cndmask_b32_e64 v91, v91, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v227, 24
	v_cmp_lt_i32_e64 s[34:35], v227, 25
	v_cndmask_b32_e64 v92, v92, v229, s[18:19]
	v_cndmask_b32_e64 v93, v93, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v227, 26
	v_cmp_lt_i32_e64 s[34:35], v227, 27
	v_cndmask_b32_e64 v94, v94, v229, s[18:19]
	v_cndmask_b32_e64 v95, v95, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v226, 0
	v_cmp_lt_i32_e64 s[34:35], v226, 1
	v_cndmask_b32_e64 v96, v96, v229, s[18:19]
	v_cndmask_b32_e64 v97, v97, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v226, 2
	v_cmp_lt_i32_e64 s[34:35], v226, 3
	v_cndmask_b32_e64 v98, v98, v229, s[18:19]
	v_cndmask_b32_e64 v99, v99, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v226, 8
	v_cmp_lt_i32_e64 s[34:35], v226, 9
	v_cndmask_b32_e64 v100, v100, v229, s[18:19]
	v_cndmask_b32_e64 v101, v101, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v226, 10
	v_cmp_lt_i32_e64 s[34:35], v226, 11
	v_cndmask_b32_e64 v102, v102, v229, s[18:19]
	v_cndmask_b32_e64 v103, v103, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v226, 16
	v_cmp_lt_i32_e64 s[34:35], v226, 17
	v_cndmask_b32_e64 v104, v104, v229, s[18:19]
	v_cndmask_b32_e64 v105, v105, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v226, 18
	v_cmp_lt_i32_e64 s[34:35], v226, 19
	v_cndmask_b32_e64 v106, v106, v229, s[18:19]
	v_cndmask_b32_e64 v107, v107, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v226, 24
	v_cmp_lt_i32_e64 s[34:35], v226, 25
	v_cndmask_b32_e64 v108, v108, v229, s[18:19]
	v_cndmask_b32_e64 v109, v109, v229, s[34:35]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v226, 26
	v_cmp_lt_i32_e64 s[34:35], v226, 27
	v_cndmask_b32_e64 v110, v110, v229, s[18:19]
	v_cndmask_b32_e64 v111, v111, v229, s[34:35]
	;;#ASMEND
.LBB0_19:
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E3: compute drain A
; P*V for v_p_0 plus unconditional online-softmax rescale into row_max_e3/rescale_e3.
; ==============================================================================
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[0:15], v[192:195], v[16:19], v[0:15]
	v_max_f32_e32 v192, v80, v81
	v_max3_f32 v192, v192, v82, v83
	v_max3_f32 v192, v192, v84, v85
	v_max3_f32 v192, v192, v86, v87
	v_max3_f32 v192, v192, v88, v89
	v_mfma_f32_32x32x16_bf16 v[32:47], v[196:199], v[16:19], v[32:47]
	v_max3_f32 v192, v192, v90, v91
	v_max3_f32 v192, v192, v92, v93
	v_max3_f32 v192, v192, v94, v95
	v_max3_f32 v192, v192, v96, v97
	v_max3_f32 v192, v192, v98, v99
	v_mfma_f32_32x32x16_bf16 v[48:63], v[200:203], v[16:19], v[48:63]
	v_max3_f32 v192, v192, v100, v101
	v_max3_f32 v192, v192, v102, v103
	v_max3_f32 v192, v192, v104, v105
	v_max3_f32 v192, v192, v106, v107
	v_max3_f32 v192, v192, v108, v109
	v_mfma_f32_32x32x16_bf16 v[64:79], v[204:207], v[16:19], v[64:79]
	v_max3_f32 v16, v192, v110, v111
	v_mov_b32_e32 v17, v16
	s_nop 1
	v_permlane32_swap_b32_e64 v16, v17 bound_ctrl:1
	v_max3_f32 v229, v216, v16, v17
	v_sub_f32_e32 v16, v216, v229
	v_sub_f32_e32 v17, v80, v229
	v_mfma_f32_32x32x16_bf16 v[0:15], v[176:179], v[20:23], v[0:15]
	v_sub_f32_e32 v18, v81, v229
	v_sub_f32_e32 v19, v82, v229
	v_sub_f32_e32 v80, v83, v229
	v_sub_f32_e32 v81, v84, v229
	v_sub_f32_e32 v82, v85, v229
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[20:23], v[32:47]
	v_sub_f32_e32 v83, v86, v229
	v_sub_f32_e32 v84, v87, v229
	v_sub_f32_e32 v85, v88, v229
	v_sub_f32_e32 v86, v89, v229
	v_sub_f32_e32 v87, v90, v229
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[20:23], v[48:63]
	v_sub_f32_e32 v88, v91, v229
	v_sub_f32_e32 v89, v92, v229
	v_sub_f32_e32 v90, v93, v229
	v_sub_f32_e32 v91, v94, v229
	v_sub_f32_e32 v92, v95, v229
	v_mfma_f32_32x32x16_bf16 v[64:79], v[188:191], v[20:23], v[64:79]
	v_sub_f32_e32 v96, v96, v229
	v_sub_f32_e32 v97, v97, v229
	v_sub_f32_e32 v98, v98, v229
	v_sub_f32_e32 v99, v99, v229
	v_sub_f32_e32 v100, v100, v229
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[24:27], v[0:15]
	v_sub_f32_e32 v101, v101, v229
	v_sub_f32_e32 v102, v102, v229
	v_sub_f32_e32 v103, v103, v229
	v_sub_f32_e32 v104, v104, v229
	v_sub_f32_e32 v105, v105, v229
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[24:27], v[32:47]
	v_sub_f32_e32 v106, v106, v229
	v_sub_f32_e32 v107, v107, v229
	v_sub_f32_e32 v108, v108, v229
	v_sub_f32_e32 v109, v109, v229
	v_sub_f32_e32 v110, v110, v229
	v_mfma_f32_32x32x16_bf16 v[48:63], v[168:171], v[24:27], v[48:63]
	v_exp_f32_e32 v216, v16
	v_exp_f32_e32 v160, v17
	v_exp_f32_e32 v161, v18
	v_mfma_f32_32x32x16_bf16 v[64:79], v[172:175], v[24:27], v[64:79]
	v_exp_f32_e32 v162, v19
	v_exp_f32_e32 v163, v80
	v_exp_f32_e32 v164, v81
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[28:31], v[0:15]
	v_exp_f32_e32 v144, v82
	v_exp_f32_e32 v145, v83
	v_exp_f32_e32 v146, v84
	v_mfma_f32_32x32x16_bf16 v[32:47], v[148:151], v[28:31], v[32:47]
	v_exp_f32_e32 v147, v85
	v_exp_f32_e32 v148, v86
	v_exp_f32_e32 v149, v87
	v_mfma_f32_32x32x16_bf16 v[48:63], v[152:155], v[28:31], v[48:63]
	v_exp_f32_e32 v150, v88
	v_exp_f32_e32 v151, v89
	v_exp_f32_e32 v152, v90
	v_mfma_f32_32x32x16_bf16 v[64:79], v[156:159], v[28:31], v[64:79]
	v_exp_f32_e32 v153, v91
	v_exp_f32_e32 v154, v92
	v_sub_f32_e32 v111, v111, v229
	v_pk_mul_f32 v[14:15], v[216:217], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[216:217], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[216:217], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[216:217], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[216:217], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[216:217], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[216:217], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[216:217], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[216:217], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[216:217], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[216:217], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[216:217], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[216:217], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[216:217], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[216:217], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[216:217], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[216:217], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[216:217], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[216:217], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[216:217], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[216:217], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[216:217], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[216:217], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[216:217], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[216:217], v[78:79] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[216:217], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[216:217], v[74:75] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[216:217], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[216:217], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[216:217], v[68:69] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[216:217], v[66:67] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[216:217], v[64:65] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
	s_setprio 0
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E4: memory drain B
; Prefetch V(max_m2), read K buffer 0, wait, and synchronize.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s39
	s_lshl_b32 s34, s44, 6
	s_add_i32 s23, s23, s34
	buffer_load_dwordx4 v218, s[12:15], s23 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s23 offen lds
	ds_read_b128 v[64:67], v215
	ds_read_b128 v[166:169], v215 offset:32
	ds_read_b128 v[170:173], v215 offset:512
	ds_read_b128 v[174:177], v215 offset:544
	ds_read_b128 v[178:181], v215 offset:64
	ds_read_b128 v[182:185], v215 offset:96
	ds_read_b128 v[186:189], v215 offset:576
	ds_read_b128 v[190:193], v215 offset:608
	ds_read_b128 v[194:197], v215 offset:8320
	ds_read_b128 v[198:201], v215 offset:8352
	ds_read_b128 v[202:205], v215 offset:8832
	ds_read_b128 v[230:233], v215 offset:8864
	ds_read_b128 v[234:237], v215 offset:8384
	ds_read_b128 v[238:241], v215 offset:8416
	ds_read_b128 v[242:245], v215 offset:8896
	ds_read_b128 v[246:249], v215 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E5: compute drain B
; MMA0 -> v_s_0; fold rescale_e3 into l_row and finish v_p_1 softmax.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[80:95], v[64:67], v[112:115], 0
	v_exp_f32_e32 v155, v96
	v_exp_f32_e32 v156, v97
	v_exp_f32_e32 v157, v98
	v_mfma_f32_32x32x16_bf16 v[64:79], v[170:173], v[112:115], 0
	v_exp_f32_e32 v158, v99
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v101, v101
	v_mfma_f32_32x32x16_bf16 v[80:95], v[166:169], v[116:119], v[80:95]
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v159, v104
	v_mfma_f32_32x32x16_bf16 v[64:79], v[174:177], v[116:119], v[64:79]
	v_exp_f32_e32 v165, v105
	v_exp_f32_e32 v166, v106
	v_exp_f32_e32 v167, v107
	v_mfma_f32_32x32x16_bf16 v[80:95], v[178:181], v[120:123], v[80:95]
	v_exp_f32_e32 v168, v108
	v_exp_f32_e32 v169, v109
	v_exp_f32_e32 v170, v110
	v_mfma_f32_32x32x16_bf16 v[64:79], v[186:189], v[120:123], v[64:79]
	v_exp_f32_e32 v111, v111
	v_mfma_f32_32x32x16_bf16 v[80:95], v[182:185], v[124:127], v[80:95]
	v_add_f32_e32 v96, v160, v161
	v_add_f32_e32 v96, v96, v162
	v_add_f32_e32 v96, v96, v163
	v_add_f32_e32 v96, v96, v164
	v_add_f32_e32 v96, v96, v144
	v_mfma_f32_32x32x16_bf16 v[64:79], v[190:193], v[124:127], v[64:79]
	v_add_f32_e32 v96, v96, v145
	v_add_f32_e32 v96, v96, v146
	v_add_f32_e32 v96, v96, v147
	v_add_f32_e32 v96, v96, v148
	v_add_f32_e32 v96, v96, v149
	v_mfma_f32_32x32x16_bf16 v[80:95], v[194:197], v[128:131], v[80:95]
	v_add_f32_e32 v96, v96, v150
	v_add_f32_e32 v96, v96, v151
	v_add_f32_e32 v96, v96, v152
	v_add_f32_e32 v96, v96, v153
	v_add_f32_e32 v96, v96, v154
	v_mfma_f32_32x32x16_bf16 v[64:79], v[202:205], v[128:131], v[64:79]
	v_add_f32_e32 v96, v96, v155
	v_add_f32_e32 v96, v96, v156
	v_add_f32_e32 v96, v96, v157
	v_add_f32_e32 v96, v96, v158
	v_add_f32_e32 v96, v96, v100
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[132:135], v[80:95]
	v_add_f32_e32 v96, v96, v101
	v_add_f32_e32 v96, v96, v102
	v_add_f32_e32 v96, v96, v103
	v_add_f32_e32 v96, v96, v159
	v_add_f32_e32 v96, v96, v165
	v_mfma_f32_32x32x16_bf16 v[64:79], v[230:233], v[132:135], v[64:79]
	v_add_f32_e32 v96, v96, v166
	v_add_f32_e32 v96, v96, v167
	v_add_f32_e32 v96, v96, v168
	v_add_f32_e32 v96, v96, v169
	v_add_f32_e32 v96, v96, v170
	v_mfma_f32_32x32x16_bf16 v[80:95], v[234:237], v[136:139], v[80:95]
	v_add_f32_e32 v226, v96, v111
	v_mov_b32_e32 v227, v226
	s_nop 1
	v_permlane32_swap_b32_e64 v227, v226 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v160, v161
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v164, v144
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[242:245], v[136:139], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v145, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v155, v156
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v100, v101
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v102, v103
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[238:241], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v147, v148
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v149, v150
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v151, v152
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v153, v154
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v159, v165
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[246:249], v[140:143], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v166, v167
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v168, v169
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v170, v111
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E6: memory drain B
; Read V buffer 1 and causally mask v_s_0 for the next P*V.
; ==============================================================================
	s_barrier
	v_add_u32_e32 v221, 0xc600, v221
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v221 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v221 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v221 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v221 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v221 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v221 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v221 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v221 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v221 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v221 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v221 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v221 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v221 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v221 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v221 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v221 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v221 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v221 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v221 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v221 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v221 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v221 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v221 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v221 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v221 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v221 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v221 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v221 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v221 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v221 offset:9536

	;;#ASMEND
	s_cmp_ge_i32 s7, s4
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v221 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_21
	v_sub_u32_e32 v228, s3, v228
	v_add_u32_e32 v228, v228, v211
	v_add_u32_e32 v230, 0x80, v228
	v_add_u32_e32 v228, 0x60, v228
	v_mov_b32_e32 v231, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v230, 0
	v_cmp_lt_i32_e64 s[52:53], v230, 1
	v_cndmask_b32_e64 v80, v80, v231, s[18:19]
	v_cndmask_b32_e64 v81, v81, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v230, 2
	v_cmp_lt_i32_e64 s[52:53], v230, 3
	v_cndmask_b32_e64 v82, v82, v231, s[18:19]
	v_cndmask_b32_e64 v83, v83, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v230, 8
	v_cmp_lt_i32_e64 s[52:53], v230, 9
	v_cndmask_b32_e64 v84, v84, v231, s[18:19]
	v_cndmask_b32_e64 v85, v85, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v230, 10
	v_cmp_lt_i32_e64 s[52:53], v230, 11
	v_cndmask_b32_e64 v86, v86, v231, s[18:19]
	v_cndmask_b32_e64 v87, v87, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v230, 16
	v_cmp_lt_i32_e64 s[52:53], v230, 17
	v_cndmask_b32_e64 v88, v88, v231, s[18:19]
	v_cndmask_b32_e64 v89, v89, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v230, 18
	v_cmp_lt_i32_e64 s[52:53], v230, 19
	v_cndmask_b32_e64 v90, v90, v231, s[18:19]
	v_cndmask_b32_e64 v91, v91, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v230, 24
	v_cmp_lt_i32_e64 s[52:53], v230, 25
	v_cndmask_b32_e64 v92, v92, v231, s[18:19]
	v_cndmask_b32_e64 v93, v93, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v230, 26
	v_cmp_lt_i32_e64 s[52:53], v230, 27
	v_cndmask_b32_e64 v94, v94, v231, s[18:19]
	v_cndmask_b32_e64 v95, v95, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v228, 0
	v_cmp_lt_i32_e64 s[52:53], v228, 1
	v_cndmask_b32_e64 v64, v64, v231, s[18:19]
	v_cndmask_b32_e64 v65, v65, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v228, 2
	v_cmp_lt_i32_e64 s[52:53], v228, 3
	v_cndmask_b32_e64 v66, v66, v231, s[18:19]
	v_cndmask_b32_e64 v67, v67, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v228, 8
	v_cmp_lt_i32_e64 s[52:53], v228, 9
	v_cndmask_b32_e64 v68, v68, v231, s[18:19]
	v_cndmask_b32_e64 v69, v69, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v228, 10
	v_cmp_lt_i32_e64 s[52:53], v228, 11
	v_cndmask_b32_e64 v70, v70, v231, s[18:19]
	v_cndmask_b32_e64 v71, v71, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v228, 16
	v_cmp_lt_i32_e64 s[52:53], v228, 17
	v_cndmask_b32_e64 v72, v72, v231, s[18:19]
	v_cndmask_b32_e64 v73, v73, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v228, 18
	v_cmp_lt_i32_e64 s[52:53], v228, 19
	v_cndmask_b32_e64 v74, v74, v231, s[18:19]
	v_cndmask_b32_e64 v75, v75, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v228, 24
	v_cmp_lt_i32_e64 s[52:53], v228, 25
	v_cndmask_b32_e64 v76, v76, v231, s[18:19]
	v_cndmask_b32_e64 v77, v77, v231, s[52:53]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[18:19], v228, 26
	v_cmp_lt_i32_e64 s[52:53], v228, 27
	v_cndmask_b32_e64 v78, v78, v231, s[18:19]
	v_cndmask_b32_e64 v79, v79, v231, s[52:53]
	;;#ASMEND
.LBB0_21:
	s_waitcnt vmcnt(2) lgkmcnt(0)
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E7: compute drain B
; P*V for v_p_1, update m/l/O, and prepare final v_p_0 slice.
; ==============================================================================
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[0:15], v[192:195], v[96:99], v[0:15]
	s_nop 1
	v_max_f32_e32 v192, v80, v81
	v_max3_f32 v192, v192, v82, v83
	v_max3_f32 v192, v192, v84, v85
	v_max3_f32 v192, v192, v86, v87
	v_max3_f32 v192, v192, v88, v89
	v_mfma_f32_32x32x16_bf16 v[16:31], v[196:199], v[96:99], v[16:31]
	v_max3_f32 v192, v192, v90, v91
	v_max3_f32 v192, v192, v92, v93
	v_max3_f32 v192, v192, v94, v95
	v_max3_f32 v192, v192, v64, v65
	v_max3_f32 v192, v192, v66, v67
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[96:99], v[32:47]
	v_max3_f32 v192, v192, v68, v69
	v_max3_f32 v192, v192, v70, v71
	v_max3_f32 v192, v192, v72, v73
	v_max3_f32 v192, v192, v74, v75
	v_max3_f32 v192, v192, v76, v77
	v_mfma_f32_32x32x16_bf16 v[48:63], v[204:207], v[96:99], v[48:63]
	v_max3_f32 v96, v192, v78, v79
	v_mov_b32_e32 v97, v96
	s_nop 1
	v_permlane32_swap_b32_e64 v96, v97 bound_ctrl:1
	v_max3_f32 v192, v229, v96, v97
	v_sub_f32_e32 v96, v229, v192
	v_sub_f32_e32 v80, v80, v192
	v_mfma_f32_32x32x16_bf16 v[0:15], v[176:179], v[100:103], v[0:15]
	v_sub_f32_e32 v81, v81, v192
	v_sub_f32_e32 v82, v82, v192
	v_sub_f32_e32 v83, v83, v192
	v_sub_f32_e32 v84, v84, v192
	v_sub_f32_e32 v85, v85, v192
	v_mfma_f32_32x32x16_bf16 v[16:31], v[180:183], v[100:103], v[16:31]
	v_sub_f32_e32 v86, v86, v192
	v_sub_f32_e32 v87, v87, v192
	v_sub_f32_e32 v88, v88, v192
	v_sub_f32_e32 v89, v89, v192
	v_sub_f32_e32 v90, v90, v192
	v_mfma_f32_32x32x16_bf16 v[32:47], v[184:187], v[100:103], v[32:47]
	v_sub_f32_e32 v91, v91, v192
	v_sub_f32_e32 v92, v92, v192
	v_sub_f32_e32 v93, v93, v192
	v_sub_f32_e32 v94, v94, v192
	v_sub_f32_e32 v95, v95, v192
	v_mfma_f32_32x32x16_bf16 v[48:63], v[188:191], v[100:103], v[48:63]
	v_sub_f32_e32 v97, v64, v192
	v_sub_f32_e32 v98, v65, v192
	v_sub_f32_e32 v99, v66, v192
	v_sub_f32_e32 v100, v67, v192
	v_sub_f32_e32 v101, v68, v192
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[104:107], v[0:15]
	v_sub_f32_e32 v102, v69, v192
	v_sub_f32_e32 v103, v70, v192
	v_sub_f32_e32 v160, v71, v192
	v_sub_f32_e32 v161, v72, v192
	v_sub_f32_e32 v162, v73, v192
	v_mfma_f32_32x32x16_bf16 v[16:31], v[164:167], v[104:107], v[16:31]
	v_sub_f32_e32 v163, v74, v192
	v_sub_f32_e32 v164, v75, v192
	v_sub_f32_e32 v165, v76, v192
	v_sub_f32_e32 v166, v77, v192
	v_sub_f32_e32 v167, v78, v192
	v_mfma_f32_32x32x16_bf16 v[32:47], v[168:171], v[104:107], v[32:47]
	v_exp_f32_e32 v176, v96
	v_exp_f32_e32 v96, v80
	v_exp_f32_e32 v168, v81
	v_mfma_f32_32x32x16_bf16 v[48:63], v[172:175], v[104:107], v[48:63]
	v_exp_f32_e32 v104, v82
	v_exp_f32_e32 v105, v83
	v_exp_f32_e32 v106, v84
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[108:111], v[0:15]
	v_exp_f32_e32 v107, v85
	v_exp_f32_e32 v144, v86
	v_exp_f32_e32 v145, v87
	v_mfma_f32_32x32x16_bf16 v[16:31], v[148:151], v[108:111], v[16:31]
	v_exp_f32_e32 v146, v88
	v_exp_f32_e32 v147, v89
	v_exp_f32_e32 v148, v90
	v_mfma_f32_32x32x16_bf16 v[32:47], v[152:155], v[108:111], v[32:47]
	v_exp_f32_e32 v149, v91
	v_exp_f32_e32 v150, v92
	v_exp_f32_e32 v151, v93
	v_mfma_f32_32x32x16_bf16 v[48:63], v[156:159], v[108:111], v[48:63]
	v_exp_f32_e32 v108, v94
	v_exp_f32_e32 v109, v95
	v_sub_f32_e32 v110, v79, v192
	v_pk_mul_f32 v[14:15], v[176:177], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[176:177], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[176:177], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[176:177], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[176:177], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[176:177], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[176:177], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[176:177], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[176:177], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[176:177], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[176:177], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[176:177], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[176:177], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[176:177], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[176:177], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[176:177], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[176:177], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[176:177], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[176:177], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[176:177], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[176:177], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[176:177], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[176:177], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[176:177], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[176:177], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[176:177], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[176:177], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[176:177], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[176:177], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[176:177], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[176:177], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[176:177], v[48:49] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
	s_setprio 0
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E8: memory drain C
; Read K/V fragments for the last score/P*V stage.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v218, s[12:15], s22 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s22 offen lds
	ds_read_b128 v[80:83], v215 offset:34048
	ds_read_b128 v[154:157], v215 offset:34080
	ds_read_b128 v[170:173], v215 offset:34560
	ds_read_b128 v[178:181], v215 offset:34592
	ds_read_b128 v[182:185], v215 offset:34112
	ds_read_b128 v[186:189], v215 offset:34144
	ds_read_b128 v[194:197], v215 offset:34624
	ds_read_b128 v[198:201], v215 offset:34656
	ds_read_b128 v[202:205], v215 offset:42368
	ds_read_b128 v[228:231], v215 offset:42400
	ds_read_b128 v[232:235], v215 offset:42880
	ds_read_b128 v[236:239], v215 offset:42912
	ds_read_b128 v[240:243], v215 offset:42432
	ds_read_b128 v[244:247], v215 offset:42464
	ds_read_b128 v[248:251], v215 offset:42944
	ds_read_b128 v[252:255], v215 offset:42976
	s_waitcnt vmcnt(2) lgkmcnt(0)
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E9: compute drain C
; MMA0 plus softmax sum for the last tile family.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[112:115], 0
	v_exp_f32_e32 v111, v97
	v_exp_f32_e32 v152, v98
	v_exp_f32_e32 v153, v99
	v_mfma_f32_32x32x16_bf16 v[80:95], v[170:173], v[112:115], 0
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v101, v101
	v_exp_f32_e32 v102, v102
	v_mfma_f32_32x32x16_bf16 v[64:79], v[154:157], v[116:119], v[64:79]
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v112, v160
	v_exp_f32_e32 v113, v161
	v_mfma_f32_32x32x16_bf16 v[80:95], v[178:181], v[116:119], v[80:95]
	v_exp_f32_e32 v114, v162
	v_exp_f32_e32 v115, v163
	v_exp_f32_e32 v116, v164
	v_mfma_f32_32x32x16_bf16 v[64:79], v[182:185], v[120:123], v[64:79]
	v_exp_f32_e32 v117, v165
	v_exp_f32_e32 v118, v166
	v_exp_f32_e32 v119, v167
	v_mfma_f32_32x32x16_bf16 v[80:95], v[194:197], v[120:123], v[80:95]
	v_exp_f32_e32 v120, v110
	v_mfma_f32_32x32x16_bf16 v[64:79], v[186:189], v[124:127], v[64:79]
	v_add_f32_e32 v97, v96, v168
	v_add_f32_e32 v97, v97, v104
	v_add_f32_e32 v97, v97, v105
	v_add_f32_e32 v97, v97, v106
	v_add_f32_e32 v97, v97, v107
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[124:127], v[80:95]
	v_add_f32_e32 v97, v97, v144
	v_add_f32_e32 v97, v97, v145
	v_add_f32_e32 v97, v97, v146
	v_add_f32_e32 v97, v97, v147
	v_add_f32_e32 v97, v97, v148
	v_mfma_f32_32x32x16_bf16 v[64:79], v[202:205], v[128:131], v[64:79]
	v_add_f32_e32 v97, v97, v149
	v_add_f32_e32 v97, v97, v150
	v_add_f32_e32 v97, v97, v151
	v_add_f32_e32 v97, v97, v108
	v_add_f32_e32 v97, v97, v109
	v_mfma_f32_32x32x16_bf16 v[80:95], v[232:235], v[128:131], v[80:95]
	v_add_f32_e32 v97, v97, v111
	v_add_f32_e32 v97, v97, v152
	v_add_f32_e32 v97, v97, v153
	v_add_f32_e32 v97, v97, v100
	v_add_f32_e32 v97, v97, v101
	v_mfma_f32_32x32x16_bf16 v[64:79], v[228:231], v[132:135], v[64:79]
	v_add_f32_e32 v97, v97, v102
	v_add_f32_e32 v97, v97, v103
	v_add_f32_e32 v97, v97, v112
	v_add_f32_e32 v97, v97, v113
	v_add_f32_e32 v97, v97, v114
	v_mfma_f32_32x32x16_bf16 v[80:95], v[236:239], v[132:135], v[80:95]
	v_add_f32_e32 v97, v97, v115
	v_add_f32_e32 v97, v97, v116
	v_add_f32_e32 v97, v97, v117
	v_add_f32_e32 v97, v97, v118
	v_add_f32_e32 v97, v97, v119
	v_mfma_f32_32x32x16_bf16 v[64:79], v[240:243], v[136:139], v[64:79]
	v_add_f32_e32 v177, v97, v120
	v_mov_b32_e32 v178, v177
	s_nop 1
	v_permlane32_swap_b32_e64 v178, v177 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v96, v168
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v104, v105
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v106, v107
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[248:251], v[136:139], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v144, v145
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v111, v152
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v153, v100
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v101, v102
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v103, v112
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[244:247], v[140:143], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v146, v147
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v148, v149
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v150, v151
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v108, v109
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v113, v114
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[252:255], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v115, v116
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v117, v118
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v119, v120
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E10: memory drain D
; Read V transpose fragments for final GEMM1 use and apply final causal masks.
; ==============================================================================
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v220 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v220 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v220 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v220 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v220 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v220 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v220 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v220 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v220 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v220 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v220 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v220 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v220 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v220 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v220 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v220 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v220 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v220 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v220 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v220 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v220 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v220 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v220 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v220 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v220 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v220 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v220 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v220 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v220 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v220 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v220 offset:9536

	;;#ASMEND
	s_cmp_ge_i32 s7, s5
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v220 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_23
	v_lshl_or_b32 v179, v208, 2, s4
	v_sub_u32_e32 v179, s3, v179
	v_add_u32_e32 v179, v179, v211
	v_subrev_u32_e32 v180, 32, v179
	v_mov_b32_e32 v181, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v179, 0
	v_cmp_lt_i32_e64 s[18:19], v179, 1
	v_cndmask_b32_e64 v64, v64, v181, s[4:5]
	v_cndmask_b32_e64 v65, v65, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v179, 2
	v_cmp_lt_i32_e64 s[18:19], v179, 3
	v_cndmask_b32_e64 v66, v66, v181, s[4:5]
	v_cndmask_b32_e64 v67, v67, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v179, 8
	v_cmp_lt_i32_e64 s[18:19], v179, 9
	v_cndmask_b32_e64 v68, v68, v181, s[4:5]
	v_cndmask_b32_e64 v69, v69, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v179, 10
	v_cmp_lt_i32_e64 s[18:19], v179, 11
	v_cndmask_b32_e64 v70, v70, v181, s[4:5]
	v_cndmask_b32_e64 v71, v71, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v179, 16
	v_cmp_lt_i32_e64 s[18:19], v179, 17
	v_cndmask_b32_e64 v72, v72, v181, s[4:5]
	v_cndmask_b32_e64 v73, v73, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v179, 18
	v_cmp_lt_i32_e64 s[18:19], v179, 19
	v_cndmask_b32_e64 v74, v74, v181, s[4:5]
	v_cndmask_b32_e64 v75, v75, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v179, 24
	v_cmp_lt_i32_e64 s[18:19], v179, 25
	v_cndmask_b32_e64 v76, v76, v181, s[4:5]
	v_cndmask_b32_e64 v77, v77, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v179, 26
	v_cmp_lt_i32_e64 s[18:19], v179, 27
	v_cndmask_b32_e64 v78, v78, v181, s[4:5]
	v_cndmask_b32_e64 v79, v79, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v180, 0
	v_cmp_lt_i32_e64 s[18:19], v180, 1
	v_cndmask_b32_e64 v80, v80, v181, s[4:5]
	v_cndmask_b32_e64 v81, v81, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v180, 2
	v_cmp_lt_i32_e64 s[18:19], v180, 3
	v_cndmask_b32_e64 v82, v82, v181, s[4:5]
	v_cndmask_b32_e64 v83, v83, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v180, 8
	v_cmp_lt_i32_e64 s[18:19], v180, 9
	v_cndmask_b32_e64 v84, v84, v181, s[4:5]
	v_cndmask_b32_e64 v85, v85, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v180, 10
	v_cmp_lt_i32_e64 s[18:19], v180, 11
	v_cndmask_b32_e64 v86, v86, v181, s[4:5]
	v_cndmask_b32_e64 v87, v87, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v180, 16
	v_cmp_lt_i32_e64 s[18:19], v180, 17
	v_cndmask_b32_e64 v88, v88, v181, s[4:5]
	v_cndmask_b32_e64 v89, v89, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v180, 18
	v_cmp_lt_i32_e64 s[18:19], v180, 19
	v_cndmask_b32_e64 v90, v90, v181, s[4:5]
	v_cndmask_b32_e64 v91, v91, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v180, 24
	v_cmp_lt_i32_e64 s[18:19], v180, 25
	v_cndmask_b32_e64 v92, v92, v181, s[4:5]
	v_cndmask_b32_e64 v93, v93, v181, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[4:5], v180, 26
	v_cmp_lt_i32_e64 s[18:19], v180, 27
	v_cndmask_b32_e64 v94, v94, v181, s[4:5]
	v_cndmask_b32_e64 v95, v95, v181, s[18:19]
	;;#ASMEND
.LBB0_23:
	v_add_f32_e32 v179, v223, v225
	v_add_f32_e32 v179, v179, v224
	v_fmac_f32_e32 v226, v216, v179
	s_load_dwordx2 s[4:5], s[0:1], 0x30
	v_add_f32_e32 v179, v226, v227
	v_fmac_f32_e32 v177, v176, v179
	v_add_f32_e32 v176, v177, v178
	s_mov_b32 s7, 0x27000
	s_waitcnt lgkmcnt(0)
	s_and_b32 s5, s5, 0xffff
	s_waitcnt vmcnt(0) lgkmcnt(0)
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E11: compute drain D
; Final score max/exp/sum update and P packing before normalization.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[0:15], v[164:167], v[96:99], v[0:15]
	v_max_f32_e32 v164, v64, v65
	v_max3_f32 v164, v164, v66, v67
	v_max3_f32 v164, v164, v68, v69
	v_max3_f32 v164, v164, v70, v71
	v_max3_f32 v164, v164, v72, v73
	v_max3_f32 v164, v164, v74, v75
	v_mfma_f32_32x32x16_bf16 v[16:31], v[168:171], v[96:99], v[16:31]
	v_max3_f32 v164, v164, v76, v77
	v_max3_f32 v164, v164, v78, v79
	v_max3_f32 v164, v164, v80, v81
	v_max3_f32 v164, v164, v82, v83
	v_max3_f32 v164, v164, v84, v85
	v_max3_f32 v164, v164, v86, v87
	v_mfma_f32_32x32x16_bf16 v[32:47], v[172:175], v[96:99], v[32:47]
	v_max3_f32 v164, v164, v88, v89
	v_max3_f32 v164, v164, v90, v91
	v_max3_f32 v164, v164, v92, v93
	v_max3_f32 v164, v164, v94, v95
	v_mov_b32_e32 v165, v164
	s_nop 1
	v_permlane32_swap_b32_e64 v164, v165 bound_ctrl:1
	v_max3_f32 v164, v192, v164, v165
	v_mfma_f32_32x32x16_bf16 v[48:63], v[160:163], v[96:99], v[48:63]
	v_sub_f32_e32 v96, v192, v164
	v_sub_f32_e32 v64, v64, v164
	v_sub_f32_e32 v65, v65, v164
	v_sub_f32_e32 v66, v66, v164
	v_sub_f32_e32 v67, v67, v164
	v_sub_f32_e32 v68, v68, v164
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[100:103], v[0:15]
	v_sub_f32_e32 v69, v69, v164
	v_sub_f32_e32 v70, v70, v164
	v_sub_f32_e32 v71, v71, v164
	v_sub_f32_e32 v72, v72, v164
	v_sub_f32_e32 v73, v73, v164
	v_sub_f32_e32 v74, v74, v164
	v_mfma_f32_32x32x16_bf16 v[16:31], v[148:151], v[100:103], v[16:31]
	v_sub_f32_e32 v75, v75, v164
	v_sub_f32_e32 v76, v76, v164
	v_sub_f32_e32 v77, v77, v164
	v_sub_f32_e32 v78, v78, v164
	v_sub_f32_e32 v79, v79, v164
	v_sub_f32_e32 v80, v80, v164
	v_mfma_f32_32x32x16_bf16 v[32:47], v[152:155], v[100:103], v[32:47]
	v_sub_f32_e32 v81, v81, v164
	v_sub_f32_e32 v82, v82, v164
	v_sub_f32_e32 v83, v83, v164
	v_sub_f32_e32 v84, v84, v164
	v_sub_f32_e32 v85, v85, v164
	v_sub_f32_e32 v86, v86, v164
	v_mfma_f32_32x32x16_bf16 v[48:63], v[156:159], v[100:103], v[48:63]
	v_sub_f32_e32 v87, v87, v164
	v_sub_f32_e32 v88, v88, v164
	v_sub_f32_e32 v89, v89, v164
	v_sub_f32_e32 v90, v90, v164
	v_sub_f32_e32 v91, v91, v164
	v_sub_f32_e32 v92, v92, v164
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[104:107], v[0:15]
	v_sub_f32_e32 v93, v93, v164
	v_sub_f32_e32 v94, v94, v164
	v_sub_f32_e32 v95, v95, v164
	v_mfma_f32_32x32x16_bf16 v[16:31], v[132:135], v[104:107], v[16:31]
	v_exp_f32_e32 v128, v96
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_mfma_f32_32x32x16_bf16 v[32:47], v[136:139], v[104:107], v[32:47]
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_mfma_f32_32x32x16_bf16 v[48:63], v[140:143], v[104:107], v[48:63]
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mfma_f32_32x32x16_bf16 v[0:15], v[116:119], v[108:111], v[0:15]
	v_exp_f32_e32 v96, v72
	v_exp_f32_e32 v97, v73
	v_exp_f32_e32 v98, v74
	v_mfma_f32_32x32x16_bf16 v[16:31], v[120:123], v[108:111], v[16:31]
	v_exp_f32_e32 v99, v75
	v_exp_f32_e32 v76, v76
	v_exp_f32_e32 v77, v77
	v_mfma_f32_32x32x16_bf16 v[32:47], v[124:127], v[108:111], v[32:47]
	v_exp_f32_e32 v78, v78
	v_exp_f32_e32 v79, v79
	v_mfma_f32_32x32x16_bf16 v[48:63], v[112:115], v[108:111], v[48:63]
	v_exp_f32_e32 v72, v80
	v_exp_f32_e32 v73, v81
	v_exp_f32_e32 v74, v82
	v_exp_f32_e32 v75, v83
	v_exp_f32_e32 v80, v84
	v_exp_f32_e32 v81, v85
	v_exp_f32_e32 v82, v86
	v_exp_f32_e32 v83, v87
	v_exp_f32_e32 v84, v88
	v_exp_f32_e32 v85, v89
	v_exp_f32_e32 v86, v90
	v_exp_f32_e32 v87, v91
	v_exp_f32_e32 v88, v92
	v_exp_f32_e32 v89, v93
	v_exp_f32_e32 v90, v94
	v_exp_f32_e32 v91, v95
	v_add_f32_e32 v92, v64, v65
	v_add_f32_e32 v92, v92, v66
	v_add_f32_e32 v92, v92, v67
	v_add_f32_e32 v92, v92, v68
	v_add_f32_e32 v92, v92, v69
	v_add_f32_e32 v92, v92, v70
	v_add_f32_e32 v92, v92, v71
	v_add_f32_e32 v92, v92, v96
	v_add_f32_e32 v92, v92, v97
	v_add_f32_e32 v92, v92, v98
	v_add_f32_e32 v92, v92, v99
	v_add_f32_e32 v92, v92, v76
	v_add_f32_e32 v92, v92, v77
	v_add_f32_e32 v92, v92, v78
	v_add_f32_e32 v92, v92, v79
	v_add_f32_e32 v92, v92, v72
	v_add_f32_e32 v92, v92, v73
	v_add_f32_e32 v92, v92, v74
	v_add_f32_e32 v92, v92, v75
	v_add_f32_e32 v92, v92, v80
	v_add_f32_e32 v92, v92, v81
	v_add_f32_e32 v92, v92, v82
	v_add_f32_e32 v92, v92, v83
	v_add_f32_e32 v92, v92, v84
	v_add_f32_e32 v92, v92, v85
	v_add_f32_e32 v92, v92, v86
	v_add_f32_e32 v92, v92, v87
	v_add_f32_e32 v92, v92, v88
	v_add_f32_e32 v92, v92, v89
	v_add_f32_e32 v92, v92, v90
	v_add_f32_e32 v92, v92, v91
	v_mov_b32_e32 v93, v92
	s_nop 1
	v_permlane32_swap_b32_e64 v93, v92 bound_ctrl:1
	v_fmac_f32_e32 v92, v128, v176
	v_add_f32_e32 v144, v92, v93
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v64, v64, v65
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v65, v66, v67
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v66, v68, v69
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v67, v70, v71
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v72, v72, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v73, v74, v75
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v74, v80, v81
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v75, v82, v83
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v68, v96, v97
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v69, v98, v99
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v70, v76, v77
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v71, v78, v79
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v76, v84, v85
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v77, v86, v87
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v78, v88, v89
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v79, v90, v91
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	v_pk_mul_f32 v[110:111], v[128:129], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[128:129], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[128:129], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[128:129], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[128:129], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[128:129], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[128:129], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[128:129], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[94:95], v[128:129], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[128:129], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[128:129], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[128:129], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[128:129], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[128:129], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[128:129], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[128:129], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[128:129], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[128:129], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[128:129], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[128:129], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[128:129], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[128:129], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[128:129], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[128:129], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[128:129], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[128:129], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[128:129], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[128:129], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[128:129], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[128:129], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[128:129], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[128:129], v[48:49] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E12: memory drain E
; Read final V transpose fragments feeding the last GEMM1 operations.
; ==============================================================================
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v221 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v221 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v221 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v221 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v221 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v221 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v221 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v221 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v221 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v221 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v221 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v221 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v221 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v221 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v221 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v221 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v221 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v221 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v221 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v221 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v221 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v221 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v221 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v221 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v221 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v221 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v221 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v221 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v221 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v221 offset:9536

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v221 offset:9664

	;;#ASMEND
	s_waitcnt lgkmcnt(0)
; ==============================================================================
; PASS 0 EPILOGUE CLUSTER E13: final GEMM1 + normalization
; Compute last O accumulators, reciprocal l_row, scale O, convert to bf16 packs.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s42
	v_mfma_f32_32x32x16_bf16 v[96:111], v[32:35], v[64:67], v[96:111]
	v_rcp_f32_e32 v32, v144
	v_cmp_lt_f32_e32 vcc, 0, v144
	s_nop 1
	v_cndmask_b32_e32 v32, 0, v32, vcc
	v_mfma_f32_32x32x16_bf16 v[80:95], v[48:51], v[64:67], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[112:115], v[64:67], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[64:67], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[96:111], v[36:39], v[68:71], v[96:111]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[52:55], v[68:71], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[116:119], v[68:71], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[132:135], v[68:71], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[96:111], v[40:43], v[72:75], v[96:111]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[56:59], v[72:75], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[120:123], v[72:75], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[136:139], v[72:75], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[96:111], v[44:47], v[76:79], v[96:111]
	s_nop 11
	v_pk_mul_f32 v[34:35], v[110:111], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[108:109], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[106:107], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[104:105], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[102:103], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[100:101], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[98:99], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[96:97], v[32:33] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[60:63], v[76:79], v[80:95]
	s_nop 11
	v_pk_mul_f32 v[50:51], v[94:95], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[92:93], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[90:91], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[88:89], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[86:87], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[84:85], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[82:83], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[80:81], v[32:33] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[124:127], v[76:79], v[16:31]
	s_nop 11
	v_pk_mul_f32 v[30:31], v[30:31], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[32:33] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[140:143], v[76:79], v[0:15]
	s_nop 11
	v_pk_mul_f32 v[14:15], v[14:15], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[2:3], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[0:1], v[32:33] op_sel_hi:[1,0]
; ==============================================================================
; PASS 0 FINALIZE/STORE
; Permute packed bf16 lanes, guard OOB rows, store O for the base q-block,
; then set up the mirror q-block pass.
; ==============================================================================
	;;#ASMSTART
	s_cmp_eq_u32 s33, 0
	s_cbranch_scc0 1f
	s_barrier
	1:
	;;#ASMEND
	v_mov_b32_e32 v68, v217
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v48, v49
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v46, v47
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v44, v45
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v42, v43
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v42, v43
	v_mov_b32_e32 v42, v1
	v_mov_b32_e32 v44, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v42, v44
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v45, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v42, v45
	v_mov_b32_e32 v45, v3
	v_mov_b32_e32 v46, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v45, v46
	v_cmp_eq_u64_e64 s[0:1], 0, v[208:209]
	s_nop 1
	v_cndmask_b32_e64 v0, v42, v0, s[0:1]
	v_cndmask_b32_e64 v1, v45, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v43, s[0:1]
	v_cndmask_b32_e64 v3, v3, v44, s[0:1]
	v_add_u32_e32 v42, s51, v214
	v_add_lshl_u32 v42, v68, v42, 1
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v40, v41
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v38, v39
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v36, v37
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v34, v35
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v35
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v36, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v36
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v37, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v37
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v38, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v37, v38
	v_cndmask_b32_e64 v0, v34, v0, s[0:1]
	v_cndmask_b32_e64 v1, v37, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v35, s[0:1]
	v_cndmask_b32_e64 v3, v3, v36, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:32
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v64, v65
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v62, v63
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v60, v61
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v58, v59
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v35
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v36, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v36
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v37, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v37
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v38, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v37, v38
	v_cndmask_b32_e64 v0, v34, v0, s[0:1]
	v_cndmask_b32_e64 v1, v37, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v35, s[0:1]
	v_cndmask_b32_e64 v3, v3, v36, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:64
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v56, v57
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v54, v55
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v52, v53
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v50, v51
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v35
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v36, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v36
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v37, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v37
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v38, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v37, v38
	v_cndmask_b32_e64 v0, v34, v0, s[0:1]
	v_cndmask_b32_e64 v1, v37, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v35, s[0:1]
	v_cndmask_b32_e64 v3, v3, v36, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:96
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v16, v17
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v18, v19
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v20, v21
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v22, v23
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v17
	v_mov_b32_e32 v16, v1
	v_mov_b32_e32 v18, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v18
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v19, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v19
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v20, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v19, v20
	v_cndmask_b32_e64 v0, v16, v0, s[0:1]
	v_cndmask_b32_e64 v1, v19, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v17, s[0:1]
	v_cndmask_b32_e64 v3, v3, v18, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:128
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v24, v25
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v26, v27
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v28, v29
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v30, v31
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v17
	v_mov_b32_e32 v16, v1
	v_mov_b32_e32 v18, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v18
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v19, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v19
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v20, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v19, v20
	v_cndmask_b32_e64 v0, v16, v0, s[0:1]
	v_cndmask_b32_e64 v1, v19, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v17, s[0:1]
	v_cndmask_b32_e64 v3, v3, v18, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:160
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v32, v33
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v66, v67
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v4, v5
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v6, v7
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v5
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v6, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v6
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v7, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v7
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v16, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v7, v16
	v_cndmask_b32_e64 v0, v4, v0, s[0:1]
	v_cndmask_b32_e64 v1, v7, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v5, s[0:1]
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:192
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v10, v11
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v12, v13
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v14, v15
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v5
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v6, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v6
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v7, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v7
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v8, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v7, v8
	v_cndmask_b32_e64 v0, v4, v0, s[0:1]
	v_cndmask_b32_e64 v1, v7, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v5, s[0:1]
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:224
; ==============================================================================
; PASS 1 / MIRROR Q-BLOCK: PROLOGUE - mirror index and first LDS prime
; Begins the second unrolled causal-fold pass for q_block_idx = num_q_blocks - 1 - blockIdx.y.
; Reuses the same kernel body shape with mirrored q_start and fresh online-softmax state.
; ==============================================================================
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	buffer_load_dwordx4 v218, s[16:19], 0 offen lds
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], 0 offen lds
	s_not_b64 s[22:23], s[26:27]
	s_lshl_b64 s[22:23], s[22:23], 8
	s_add_u32 s22, s46, s22
	s_addc_u32 s23, s47, s23
	s_add_u32 s22, s22, 0xff
	s_addc_u32 s27, s23, 0
	s_and_b32 s26, s22, 0xffffff00
	s_add_i32 s22, s26, s36
	s_mul_i32 s22, s22, s10
	s_add_i32 s22, s22, s2
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 m0, s41
; ==============================================================================
; PASS 1 PROLOGUE CLUSTER P0: sync and Q address setup
; Synchronizes after pass-0 stores, computes mirrored q-row address, and loads Q fragments.
; ==============================================================================
	s_barrier
	;;#ASMSTART
	;;#ASMEND
	v_mov_b32_e32 v0, v217
	;;#ASMSTART
	;;#ASMEND
	v_mul_lo_u32 v1, v210, s10
	v_add_u32_e32 v0, s22, v0
	s_mov_b32 s22, s6
	s_mov_b32 s23, s15
	v_add_lshl_u32 v8, v0, v1, 1
	buffer_load_dwordx4 v[4:7], v8, s[20:23], 0 offen
	buffer_load_dwordx4 v[0:3], v8, s[20:23], 0 offen offset:32
	buffer_load_dwordx4 v[52:55], v8, s[20:23], 0 offen offset:64
	buffer_load_dwordx4 v[48:51], v8, s[20:23], 0 offen offset:96
	buffer_load_dwordx4 v[44:47], v8, s[20:23], 0 offen offset:128
	buffer_load_dwordx4 v[40:43], v8, s[20:23], 0 offen offset:160
	buffer_load_dwordx4 v[36:39], v8, s[20:23], 0 offen offset:192
	buffer_load_dwordx4 v[32:35], v8, s[20:23], 0 offen offset:224
; ==============================================================================
; PASS 1 PROLOGUE CLUSTER P1: prefetch initial K/V tiles to LDS
; Primes K/V LDS buffers for the mirror q-block pass.
; ==============================================================================
	s_nop 0
	buffer_load_dwordx4 v218, s[16:19], s48 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s48 offen lds
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dwordx4 v218, s[12:15], 0 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], 0 offen lds
; ==============================================================================
; PASS 1 PROLOGUE CLUSTER P2: read K0 from LDS into VGPRs
; Loads K fragments for the first mirror-pass QK MMA chain.
; ==============================================================================
	ds_read_b128 v[8:11], v215
	ds_read_b128 v[128:131], v215 offset:32
	ds_read_b128 v[116:119], v215 offset:512
	ds_read_b128 v[120:123], v215 offset:544
	ds_read_b128 v[124:127], v215 offset:64
	ds_read_b128 v[102:105], v215 offset:96
	ds_read_b128 v[106:109], v215 offset:576
	ds_read_b128 v[98:101], v215 offset:608
	ds_read_b128 v[94:97], v215 offset:8320
	ds_read_b128 v[86:89], v215 offset:8352
	ds_read_b128 v[90:93], v215 offset:8832
	ds_read_b128 v[82:85], v215 offset:8864
	ds_read_b128 v[78:81], v215 offset:8384
	ds_read_b128 v[70:73], v215 offset:8416
	ds_read_b128 v[74:77], v215 offset:8896
	ds_read_b128 v[66:69], v215 offset:8928
; ==============================================================================
; PASS 1 PROLOGUE CLUSTER P3: wait/stagger barrier before first QK use
; Waits for initial LDS/VMEM readiness and applies the stagger barrier gate.
; ==============================================================================
	v_mov_b32_e32 v211, 0
	s_waitcnt lgkmcnt(0)
	s_andn2_b64 vcc, exec, s[28:29]
	s_waitcnt vmcnt(2)
	s_cbranch_vccnz .LBB0_25
	s_barrier
; ==============================================================================
; PASS 1 PROLOGUE CLUSTER P4: Q scale and first QK score tile
; Scales mirrored Q, runs first MMA0 chain, masks causal lanes, and computes first softmax slice.
; ==============================================================================
.LBB0_25:
	v_and_b32_e32 v57, 0xffff0000, v33
	v_lshlrev_b32_e32 v56, 16, v33
	v_and_b32_e32 v59, 0xffff0000, v37
	v_lshlrev_b32_e32 v58, 16, v37
	v_and_b32_e32 v61, 0xffff0000, v41
	v_lshlrev_b32_e32 v60, 16, v41
	v_and_b32_e32 v65, 0xffff0000, v45
	v_lshlrev_b32_e32 v64, 16, v45
	v_and_b32_e32 v111, 0xffff0000, v49
	v_lshlrev_b32_e32 v110, 16, v49
	v_and_b32_e32 v133, 0xffff0000, v53
	v_lshlrev_b32_e32 v132, 16, v53
	v_and_b32_e32 v135, 0xffff0000, v1
	v_lshlrev_b32_e32 v134, 16, v1
	v_and_b32_e32 v13, 0xffff0000, v5
	v_lshlrev_b32_e32 v12, 16, v5
	v_and_b32_e32 v5, 0xffff0000, v4
	v_lshlrev_b32_e32 v4, 16, v4
	v_and_b32_e32 v15, 0xffff0000, v7
	v_lshlrev_b32_e32 v14, 16, v7
	v_and_b32_e32 v7, 0xffff0000, v6
	v_lshlrev_b32_e32 v6, 16, v6
	v_mov_b32_e32 v62, v212
	v_mov_b32_e32 v63, v212
	v_pk_mul_f32 v[6:7], v[62:63], v[6:7]
	v_pk_mul_f32 v[14:15], v[62:63], v[14:15]
	v_pk_mul_f32 v[4:5], v[212:213], v[4:5]
	v_pk_mul_f32 v[12:13], v[62:63], v[12:13]
	s_nop 0
	v_cvt_pk_bf16_f32 v113, v12, v13
	v_cvt_pk_bf16_f32 v112, v4, v5
	v_cvt_pk_bf16_f32 v115, v14, v15
	v_cvt_pk_bf16_f32 v114, v6, v7
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[16:31], v[8:11], v[112:115], 0
	v_and_b32_e32 v137, 0xffff0000, v0
	v_lshlrev_b32_e32 v136, 16, v0
	v_and_b32_e32 v139, 0xffff0000, v3
	v_lshlrev_b32_e32 v138, 16, v3
	v_and_b32_e32 v1, 0xffff0000, v2
	v_lshlrev_b32_e32 v0, 16, v2
	v_pk_mul_f32 v[140:141], v[62:63], v[0:1]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[116:119], v[112:115], 0
	v_mul_f32_e64 v118, v62, v138
	v_mul_f32_e64 v119, v63, v139
	v_mul_f32_e64 v136, v62, v136
	v_mul_f32_e64 v137, v63, v137
	v_mul_f32_e64 v116, v62, v134
	v_mul_f32_e64 v117, v63, v135
	v_cvt_pk_bf16_f32 v117, v116, v117
	v_cvt_pk_bf16_f32 v116, v136, v137
	v_cvt_pk_bf16_f32 v119, v118, v119
	v_cvt_pk_bf16_f32 v118, v140, v141
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[16:31], v[128:131], v[116:119], v[16:31]
	v_and_b32_e32 v53, 0xffff0000, v52
	v_lshlrev_b32_e32 v52, 16, v52
	v_and_b32_e32 v129, 0xffff0000, v55
	v_lshlrev_b32_e32 v128, 16, v55
	v_and_b32_e32 v55, 0xffff0000, v54
	v_lshlrev_b32_e32 v54, 16, v54
	v_pk_mul_f32 v[54:55], v[62:63], v[54:55]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[120:123], v[116:119], v[0:15]
	v_mul_f32_e64 v122, v62, v128
	v_mul_f32_e64 v123, v63, v129
	v_mul_f32_e64 v52, v62, v52
	v_mul_f32_e64 v53, v63, v53
	v_mul_f32_e64 v120, v62, v132
	v_mul_f32_e64 v121, v63, v133
	v_cvt_pk_bf16_f32 v121, v120, v121
	v_cvt_pk_bf16_f32 v120, v52, v53
	v_cvt_pk_bf16_f32 v123, v122, v123
	v_cvt_pk_bf16_f32 v122, v54, v55
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[16:31], v[124:127], v[120:123], v[16:31]
	v_and_b32_e32 v49, 0xffff0000, v48
	v_lshlrev_b32_e32 v48, 16, v48
	v_and_b32_e32 v53, 0xffff0000, v51
	v_lshlrev_b32_e32 v52, 16, v51
	v_and_b32_e32 v51, 0xffff0000, v50
	v_lshlrev_b32_e32 v50, 16, v50
	v_pk_mul_f32 v[50:51], v[62:63], v[50:51]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[106:109], v[120:123], v[0:15]
	v_mul_f32_e64 v52, v62, v52
	v_mul_f32_e64 v53, v63, v53
	v_mul_f32_e64 v48, v62, v48
	v_mul_f32_e64 v49, v63, v49
	v_mul_f32_e64 v54, v62, v110
	v_mul_f32_e64 v55, v63, v111
	v_cvt_pk_bf16_f32 v125, v54, v55
	v_cvt_pk_bf16_f32 v124, v48, v49
	v_cvt_pk_bf16_f32 v127, v52, v53
	v_cvt_pk_bf16_f32 v126, v50, v51
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[16:31], v[102:105], v[124:127], v[16:31]
	v_and_b32_e32 v45, 0xffff0000, v44
	v_lshlrev_b32_e32 v44, 16, v44
	v_and_b32_e32 v49, 0xffff0000, v47
	v_lshlrev_b32_e32 v48, 16, v47
	v_and_b32_e32 v47, 0xffff0000, v46
	v_lshlrev_b32_e32 v46, 16, v46
	v_pk_mul_f32 v[46:47], v[62:63], v[46:47]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[98:101], v[124:127], v[0:15]
	v_mul_f32_e64 v48, v62, v48
	v_mul_f32_e64 v49, v63, v49
	v_mul_f32_e64 v44, v212, v44
	v_mul_f32_e64 v45, v213, v45
	v_mul_f32_e64 v50, v62, v64
	v_mul_f32_e64 v51, v63, v65
	v_cvt_pk_bf16_f32 v129, v50, v51
	v_cvt_pk_bf16_f32 v128, v44, v45
	v_cvt_pk_bf16_f32 v131, v48, v49
	v_cvt_pk_bf16_f32 v130, v46, v47
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[16:31], v[94:97], v[128:131], v[16:31]
	v_and_b32_e32 v41, 0xffff0000, v40
	v_lshlrev_b32_e32 v40, 16, v40
	v_and_b32_e32 v45, 0xffff0000, v43
	v_lshlrev_b32_e32 v44, 16, v43
	v_and_b32_e32 v43, 0xffff0000, v42
	v_lshlrev_b32_e32 v42, 16, v42
	v_pk_mul_f32 v[42:43], v[62:63], v[42:43]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[90:93], v[128:131], v[0:15]
	v_mul_f32_e64 v44, v62, v44
	v_mul_f32_e64 v45, v63, v45
	v_mul_f32_e64 v40, v62, v40
	v_mul_f32_e64 v41, v63, v41
	v_mul_f32_e64 v46, v62, v60
	v_mul_f32_e64 v47, v63, v61
	v_cvt_pk_bf16_f32 v133, v46, v47
	v_cvt_pk_bf16_f32 v132, v40, v41
	v_cvt_pk_bf16_f32 v135, v44, v45
	v_cvt_pk_bf16_f32 v134, v42, v43
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[16:31], v[86:89], v[132:135], v[16:31]
	v_and_b32_e32 v37, 0xffff0000, v36
	v_lshlrev_b32_e32 v36, 16, v36
	v_and_b32_e32 v41, 0xffff0000, v39
	v_lshlrev_b32_e32 v40, 16, v39
	v_and_b32_e32 v39, 0xffff0000, v38
	v_lshlrev_b32_e32 v38, 16, v38
	v_pk_mul_f32 v[38:39], v[62:63], v[38:39]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[82:85], v[132:135], v[0:15]
	v_mul_f32_e64 v40, v62, v40
	v_mul_f32_e64 v41, v63, v41
	v_mul_f32_e64 v36, v62, v36
	v_mul_f32_e64 v37, v63, v37
	v_mul_f32_e64 v42, v62, v58
	v_mul_f32_e64 v43, v63, v59
	v_cvt_pk_bf16_f32 v137, v42, v43
	v_cvt_pk_bf16_f32 v136, v36, v37
	v_cvt_pk_bf16_f32 v139, v40, v41
	v_cvt_pk_bf16_f32 v138, v38, v39
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[16:31], v[78:81], v[136:139], v[16:31]
	v_and_b32_e32 v33, 0xffff0000, v32
	v_lshlrev_b32_e32 v32, 16, v32
	v_and_b32_e32 v37, 0xffff0000, v35
	v_lshlrev_b32_e32 v36, 16, v35
	v_and_b32_e32 v35, 0xffff0000, v34
	v_lshlrev_b32_e32 v34, 16, v34
	v_pk_mul_f32 v[34:35], v[62:63], v[34:35]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[74:77], v[136:139], v[0:15]
	v_mul_f32_e64 v36, v62, v36
	v_mul_f32_e64 v37, v63, v37
	v_mul_f32_e64 v32, v62, v32
	v_mul_f32_e64 v33, v63, v33
	v_mul_f32_e64 v38, v62, v56
	v_mul_f32_e64 v39, v63, v57
	v_cvt_pk_bf16_f32 v141, v38, v39
	v_cvt_pk_bf16_f32 v140, v32, v33
	v_cvt_pk_bf16_f32 v143, v36, v37
	v_cvt_pk_bf16_f32 v142, v34, v35
	s_nop 1
	v_mfma_f32_32x32x16_bf16 v[16:31], v[70:73], v[140:143], v[16:31]
	s_add_i32 s8, s8, s26
	v_lshl_add_u64 v[212:213], s[26:27], 0, v[210:211]
	s_mov_b32 s19, 0
	v_mfma_f32_32x32x16_bf16 v[0:15], v[66:69], v[140:143], v[0:15]
	s_add_i32 s27, s3, s8
	s_cmp_gt_i32 s27, 63
	s_cbranch_scc1 .LBB0_27
	v_sub_u32_e32 v32, s3, v222
	v_add_u32_e32 v32, v32, v212
	v_subrev_u32_e32 v33, 32, v32
	v_mov_b32_e32 v34, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v32, 0
	v_cmp_lt_i32_e64 s[22:23], v32, 1
	v_cndmask_b32_e64 v16, v16, v34, s[20:21]
	v_cndmask_b32_e64 v17, v17, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v32, 2
	v_cmp_lt_i32_e64 s[22:23], v32, 3
	v_cndmask_b32_e64 v18, v18, v34, s[20:21]
	v_cndmask_b32_e64 v19, v19, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v32, 8
	v_cmp_lt_i32_e64 s[22:23], v32, 9
	v_cndmask_b32_e64 v20, v20, v34, s[20:21]
	v_cndmask_b32_e64 v21, v21, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v32, 10
	v_cmp_lt_i32_e64 s[22:23], v32, 11
	v_cndmask_b32_e64 v22, v22, v34, s[20:21]
	v_cndmask_b32_e64 v23, v23, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v32, 16
	v_cmp_lt_i32_e64 s[22:23], v32, 17
	v_cndmask_b32_e64 v24, v24, v34, s[20:21]
	v_cndmask_b32_e64 v25, v25, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v32, 18
	v_cmp_lt_i32_e64 s[22:23], v32, 19
	v_cndmask_b32_e64 v26, v26, v34, s[20:21]
	v_cndmask_b32_e64 v27, v27, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v32, 24
	v_cmp_lt_i32_e64 s[22:23], v32, 25
	v_cndmask_b32_e64 v28, v28, v34, s[20:21]
	v_cndmask_b32_e64 v29, v29, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v32, 26
	v_cmp_lt_i32_e64 s[22:23], v32, 27
	v_cndmask_b32_e64 v30, v30, v34, s[20:21]
	v_cndmask_b32_e64 v31, v31, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v33, 0
	v_cmp_lt_i32_e64 s[22:23], v33, 1
	v_cndmask_b32_e64 v0, v0, v34, s[20:21]
	v_cndmask_b32_e64 v1, v1, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v33, 2
	v_cmp_lt_i32_e64 s[22:23], v33, 3
	v_cndmask_b32_e64 v2, v2, v34, s[20:21]
	v_cndmask_b32_e64 v3, v3, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v33, 8
	v_cmp_lt_i32_e64 s[22:23], v33, 9
	v_cndmask_b32_e64 v4, v4, v34, s[20:21]
	v_cndmask_b32_e64 v5, v5, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v33, 10
	v_cmp_lt_i32_e64 s[22:23], v33, 11
	v_cndmask_b32_e64 v6, v6, v34, s[20:21]
	v_cndmask_b32_e64 v7, v7, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v33, 16
	v_cmp_lt_i32_e64 s[22:23], v33, 17
	v_cndmask_b32_e64 v8, v8, v34, s[20:21]
	v_cndmask_b32_e64 v9, v9, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v33, 18
	v_cmp_lt_i32_e64 s[22:23], v33, 19
	v_cndmask_b32_e64 v10, v10, v34, s[20:21]
	v_cndmask_b32_e64 v11, v11, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v33, 24
	v_cmp_lt_i32_e64 s[22:23], v33, 25
	v_cndmask_b32_e64 v12, v12, v34, s[20:21]
	v_cndmask_b32_e64 v13, v13, v34, s[22:23]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[20:21], v33, 26
	v_cmp_lt_i32_e64 s[22:23], v33, 27
	v_cndmask_b32_e64 v14, v14, v34, s[20:21]
	v_cndmask_b32_e64 v15, v15, v34, s[22:23]
	;;#ASMEND
.LBB0_27:
	s_add_i32 s49, s49, s26
	s_max_i32 s8, s49, 0
	s_add_i32 s8, s8, 63
	s_lshr_b32 s18, s8, 6
	v_mov_b64_e32 v[32:33], s[18:19]
	v_cmp_lt_u64_e32 vcc, s[30:31], v[32:33]
	s_and_b64 s[20:21], vcc, exec
	s_cselect_b32 s19, s31, 0
	s_cselect_b32 s18, s30, s18
	s_add_i32 s8, s18, 1
	s_and_b32 s8, s8, 0x7fffffe
	v_cmp_gt_u64_e64 s[18:19], s[18:19], 2
	s_and_b64 s[18:19], s[18:19], exec
	s_cselect_b32 s23, 0, 0
	s_cselect_b32 s22, s8, 4
	v_max_f32_e32 v32, v16, v17
	v_max3_f32 v32, v32, v18, v19
	v_max3_f32 v32, v32, v20, v21
	v_max3_f32 v32, v32, v22, v23
	v_max3_f32 v32, v32, v24, v25
	v_max3_f32 v32, v32, v26, v27
	v_max3_f32 v32, v32, v28, v29
	v_max3_f32 v32, v32, v30, v31
	v_max3_f32 v32, v32, v0, v1
	v_max3_f32 v32, v32, v2, v3
	v_max3_f32 v32, v32, v4, v5
	v_max3_f32 v32, v32, v6, v7
	v_max3_f32 v32, v32, v8, v9
	v_max3_f32 v32, v32, v10, v11
	v_max3_f32 v32, v32, v12, v13
	v_max3_f32 v32, v32, v14, v15
	v_mov_b32_e32 v33, v32
	s_nop 1
	v_permlane32_swap_b32_e64 v32, v33 bound_ctrl:1
	s_mov_b32 s8, 0xff61b1e6
	v_max3_f32 v211, v32, v33, s8
	v_sub_f32_e32 v16, v16, v211
	v_sub_f32_e32 v17, v17, v211
	v_sub_f32_e32 v18, v18, v211
	v_sub_f32_e32 v19, v19, v211
	v_sub_f32_e32 v20, v20, v211
	v_sub_f32_e32 v21, v21, v211
	v_sub_f32_e32 v22, v22, v211
	v_sub_f32_e32 v23, v23, v211
	v_sub_f32_e32 v24, v24, v211
	v_sub_f32_e32 v25, v25, v211
	v_sub_f32_e32 v26, v26, v211
	v_sub_f32_e32 v27, v27, v211
	v_sub_f32_e32 v28, v28, v211
	v_sub_f32_e32 v29, v29, v211
	v_sub_f32_e32 v30, v30, v211
	v_sub_f32_e32 v31, v31, v211
	v_sub_f32_e32 v82, v0, v211
	v_sub_f32_e32 v81, v1, v211
	v_sub_f32_e32 v80, v2, v211
	v_sub_f32_e32 v179, v3, v211
	v_sub_f32_e32 v181, v4, v211
	v_sub_f32_e32 v180, v5, v211
	v_sub_f32_e32 v178, v6, v211
	v_sub_f32_e32 v177, v7, v211
	v_sub_f32_e32 v176, v8, v211
	v_sub_f32_e32 v164, v9, v211
	v_sub_f32_e32 v163, v10, v211
	v_sub_f32_e32 v162, v11, v211
	v_sub_f32_e32 v161, v12, v211
	v_exp_f32_e32 v96, v16
	v_exp_f32_e32 v160, v17
	v_exp_f32_e32 v97, v18
	v_exp_f32_e32 v106, v19
	v_exp_f32_e32 v98, v20
	v_exp_f32_e32 v105, v21
	v_exp_f32_e32 v99, v22
	v_exp_f32_e32 v104, v23
	v_exp_f32_e32 v100, v24
	v_exp_f32_e32 v146, v25
	v_exp_f32_e32 v101, v26
	v_exp_f32_e32 v145, v27
	v_exp_f32_e32 v102, v28
	v_exp_f32_e32 v144, v29
	v_exp_f32_e32 v103, v30
	v_exp_f32_e32 v108, v31
	v_sub_f32_e32 v165, v13, v211
	v_sub_f32_e32 v109, v14, v211
	v_sub_f32_e32 v107, v15, v211
; ==============================================================================
; PASS 1 MAIN LOOP PREHEADER: trip-count guard and loop-carry init
; Computes mirror-pass loop trip count, zero-inits O/l_row, and branches to loop or epilogue.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s42
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	buffer_load_dwordx4 v218, s[16:19], s50 offen lds
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s50 offen lds
	s_add_u32 s20, s22, -1
	s_addc_u32 s21, s23, -1
	v_cmp_lt_u64_e64 s[28:29], s[22:23], 5
	s_and_b64 vcc, exec, s[28:29]
	s_cbranch_vccnz .LBB0_36
	s_lshl_b32 s23, s44, 7
	s_not_b32 s8, s24
	s_add_i32 s8, s8, s46
	s_and_b32 s8, s8, 0xffffff00
	s_add_i32 s8, s8, s9
	v_add_u32_e32 v0, s8, v210
	v_subrev_u32_e32 v0, s46, v0
	v_sub_u32_e32 v0, v0, v222
	v_add_u32_e32 v210, 0xffffff60, v0
	s_lshl_b32 s11, s11, 9
	s_mov_b64 s[8:9], 3
	v_mov_b32_e32 v0, 0
	s_mov_b32 s28, 0
	s_movk_i32 s24, 0xc0
	v_mov_b32_e32 v209, 0
	s_mov_b32 s26, 0x41000000
	v_mov_b32_e32 v213, 0xff800000
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v60, v0
	v_mov_b32_e32 v61, v0
	v_mov_b32_e32 v62, v0
	v_mov_b32_e32 v63, v0
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C0: memory stage A
; Prefetch V(j-2) to V buffer 1, read K buffer 1, wait, then barrier.
; ==============================================================================
; ==============================================================================
; PASS 1 MAIN LOOP BODY START - steady-state software pipeline
; Mirror pass main loop. Clusters C0-C7 mirror pass 0 with different register allocation.
; ==============================================================================
.LBB0_29:
	s_mov_b32 m0, s25
	s_add_i32 s29, s34, s28
	buffer_load_dwordx4 v218, s[12:15], s29 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s29 offen lds
	ds_read_b128 v[84:87], v215 offset:34048
	ds_read_b128 v[152:155], v215 offset:34080
	ds_read_b128 v[148:151], v215 offset:34560
	ds_read_b128 v[156:159], v215 offset:34592
	ds_read_b128 v[166:169], v215 offset:34112
	ds_read_b128 v[170:173], v215 offset:34144
	ds_read_b128 v[182:185], v215 offset:34624
	ds_read_b128 v[186:189], v215 offset:34656
	ds_read_b128 v[190:193], v215 offset:42368
	ds_read_b128 v[194:197], v215 offset:42400
	ds_read_b128 v[198:201], v215 offset:42880
	ds_read_b128 v[202:205], v215 offset:42912
	ds_read_b128 v[222:225], v215 offset:42432
	ds_read_b128 v[226:229], v215 offset:42464
	ds_read_b128 v[230:233], v215 offset:42944
	ds_read_b128 v[234:237], v215 offset:42976
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C1: compute stage A
; MMA0 -> v_s_1 while finishing previous P softmax/sum/cast.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[112:115], 0
	v_exp_f32_e32 v110, v82
	v_exp_f32_e32 v111, v81
	v_exp_f32_e32 v147, v80
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[112:115], 0
	v_exp_f32_e32 v148, v179
	v_exp_f32_e32 v149, v181
	v_exp_f32_e32 v150, v180
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[116:119], v[64:79]
	v_exp_f32_e32 v151, v178
	v_exp_f32_e32 v152, v177
	v_exp_f32_e32 v153, v176
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[116:119], v[80:95]
	v_exp_f32_e32 v154, v164
	v_exp_f32_e32 v155, v163
	v_exp_f32_e32 v156, v162
	v_mfma_f32_32x32x16_bf16 v[64:79], v[166:169], v[120:123], v[64:79]
	v_exp_f32_e32 v157, v161
	v_exp_f32_e32 v158, v165
	v_exp_f32_e32 v159, v109
	v_mfma_f32_32x32x16_bf16 v[80:95], v[182:185], v[120:123], v[80:95]
	v_exp_f32_e32 v161, v107
	v_mfma_f32_32x32x16_bf16 v[64:79], v[170:173], v[124:127], v[64:79]
	v_add_f32_e32 v107, v96, v160
	v_add_f32_e32 v107, v107, v97
	v_add_f32_e32 v107, v107, v106
	v_add_f32_e32 v107, v107, v98
	v_add_f32_e32 v107, v107, v105
	v_mfma_f32_32x32x16_bf16 v[80:95], v[186:189], v[124:127], v[80:95]
	v_add_f32_e32 v107, v107, v99
	v_add_f32_e32 v107, v107, v104
	v_add_f32_e32 v107, v107, v100
	v_add_f32_e32 v107, v107, v146
	v_add_f32_e32 v107, v107, v101
	v_mfma_f32_32x32x16_bf16 v[64:79], v[190:193], v[128:131], v[64:79]
	v_add_f32_e32 v107, v107, v145
	v_add_f32_e32 v107, v107, v102
	v_add_f32_e32 v107, v107, v144
	v_add_f32_e32 v107, v107, v103
	v_add_f32_e32 v107, v107, v108
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[128:131], v[80:95]
	v_add_f32_e32 v107, v107, v110
	v_add_f32_e32 v107, v107, v111
	v_add_f32_e32 v107, v107, v147
	v_add_f32_e32 v107, v107, v148
	v_add_f32_e32 v107, v107, v149
	v_mfma_f32_32x32x16_bf16 v[64:79], v[194:197], v[132:135], v[64:79]
	v_add_f32_e32 v107, v107, v150
	v_add_f32_e32 v107, v107, v151
	v_add_f32_e32 v107, v107, v152
	v_add_f32_e32 v107, v107, v153
	v_add_f32_e32 v107, v107, v154
	v_mfma_f32_32x32x16_bf16 v[80:95], v[202:205], v[132:135], v[80:95]
	v_add_f32_e32 v107, v107, v155
	v_add_f32_e32 v107, v107, v156
	v_add_f32_e32 v107, v107, v157
	v_add_f32_e32 v107, v107, v158
	v_add_f32_e32 v107, v107, v159
	v_mfma_f32_32x32x16_bf16 v[64:79], v[222:225], v[136:139], v[64:79]
	v_add_f32_e32 v107, v107, v161
	v_mov_b32_e32 v109, v107
	s_nop 1
	v_permlane32_swap_b32_e64 v107, v109 bound_ctrl:1
	v_add_f32_e32 v109, v209, v109
	v_add_f32_e32 v209, v109, v107
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v96, v160
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[230:233], v[136:139], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v97, v106
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v98, v105
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v99, v104
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v110, v111
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v147, v148
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[226:229], v[140:143], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v149, v150
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v151, v152
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v100, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v101, v145
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v102, v144
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[234:237], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v103, v108
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v153, v154
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v155, v156
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v159, v161
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C2: memory stage A
; Prefetch next K(j), read V buffer 0, apply causal mask.
; ==============================================================================
	s_barrier
	s_add_i32 s29, s45, s28
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v218, s[16:19], s29 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s29 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v220 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v220 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v220 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v220 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v220 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v220 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v220 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v220 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v220 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v220 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v220 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v220 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v220 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v220 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v220 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v220 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v220 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v220 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v220 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v220 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v220 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v220 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v220 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v220 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v220 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v220 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v220 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v220 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v220 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v220 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v220 offset:9536

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v220 offset:9664

	;;#ASMEND
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C3: compute stage A
; Priority boost, P*V step, lazy-rescale decision, remaining P*V, and v_s_1 exp2.
; ==============================================================================
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[0:15], v[192:195], v[96:99], v[0:15]
	v_max_f32_e32 v192, v64, v65
	v_max3_f32 v192, v192, v66, v67
	v_max3_f32 v192, v192, v68, v69
	v_max3_f32 v192, v192, v70, v71
	v_max3_f32 v192, v192, v72, v73
	v_max3_f32 v192, v192, v74, v75
	v_mfma_f32_32x32x16_bf16 v[16:31], v[196:199], v[96:99], v[16:31]
	v_max3_f32 v192, v192, v76, v77
	v_max3_f32 v192, v192, v78, v79
	v_max3_f32 v192, v192, v80, v81
	v_max3_f32 v192, v192, v82, v83
	v_max3_f32 v192, v192, v84, v85
	v_max3_f32 v192, v192, v86, v87
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[96:99], v[32:47]
	v_max3_f32 v192, v192, v88, v89
	v_max3_f32 v192, v192, v90, v91
	v_max3_f32 v192, v192, v92, v93
	v_max3_f32 v192, v192, v94, v95
	v_mov_b32_e32 v193, v192
	s_nop 1
	v_permlane32_swap_b32_e64 v192, v193 bound_ctrl:1
	v_max_f32_e32 v192, v192, v193
	v_sub_f32_e32 v193, v192, v211
	v_cmp_ge_f32_e32 vcc, s26, v193
	s_cmp_eq_u64 vcc, exec
	v_mfma_f32_32x32x16_bf16 v[48:63], v[204:207], v[96:99], v[48:63]
	s_cbranch_scc0 .LBB0_34
.LBB0_30:
	v_mfma_f32_32x32x16_bf16 v[0:15], v[188:191], v[100:103], v[0:15]
	v_sub_f32_e32 v64, v64, v211
	v_sub_f32_e32 v65, v65, v211
	v_sub_f32_e32 v66, v66, v211
	v_sub_f32_e32 v67, v67, v211
	v_sub_f32_e32 v68, v68, v211
	v_sub_f32_e32 v69, v69, v211
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[100:103], v[16:31]
	v_sub_f32_e32 v70, v70, v211
	v_sub_f32_e32 v71, v71, v211
	v_sub_f32_e32 v72, v72, v211
	v_sub_f32_e32 v73, v73, v211
	v_sub_f32_e32 v74, v74, v211
	v_sub_f32_e32 v75, v75, v211
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[100:103], v[32:47]
	v_sub_f32_e32 v76, v76, v211
	v_sub_f32_e32 v77, v77, v211
	v_sub_f32_e32 v78, v78, v211
	v_sub_f32_e32 v79, v79, v211
	v_sub_f32_e32 v96, v80, v211
	v_sub_f32_e32 v97, v81, v211
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[100:103], v[48:63]
	v_sub_f32_e32 v98, v82, v211
	v_sub_f32_e32 v99, v83, v211
	v_sub_f32_e32 v100, v84, v211
	v_sub_f32_e32 v101, v85, v211
	v_sub_f32_e32 v102, v86, v211
	v_sub_f32_e32 v103, v87, v211
	v_mfma_f32_32x32x16_bf16 v[0:15], v[152:155], v[104:107], v[0:15]
	v_sub_f32_e32 v152, v88, v211
	v_sub_f32_e32 v153, v89, v211
	v_sub_f32_e32 v154, v90, v211
	v_sub_f32_e32 v155, v91, v211
	v_sub_f32_e32 v176, v92, v211
	v_sub_f32_e32 v177, v93, v211
	v_mfma_f32_32x32x16_bf16 v[16:31], v[160:163], v[104:107], v[16:31]
	v_sub_f32_e32 v160, v94, v211
	v_sub_f32_e32 v161, v95, v211
	v_mfma_f32_32x32x16_bf16 v[32:47], v[168:171], v[104:107], v[32:47]
	v_exp_f32_e32 v162, v64
	v_exp_f32_e32 v163, v65
	v_exp_f32_e32 v168, v66
	v_mfma_f32_32x32x16_bf16 v[48:63], v[172:175], v[104:107], v[48:63]
	v_exp_f32_e32 v104, v67
	v_exp_f32_e32 v105, v68
	v_exp_f32_e32 v106, v69
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[108:111], v[0:15]
	v_exp_f32_e32 v107, v70
	v_exp_f32_e32 v144, v71
	v_exp_f32_e32 v145, v72
	v_mfma_f32_32x32x16_bf16 v[16:31], v[148:151], v[108:111], v[16:31]
	v_exp_f32_e32 v146, v73
	v_exp_f32_e32 v147, v74
	v_exp_f32_e32 v148, v75
	v_mfma_f32_32x32x16_bf16 v[32:47], v[156:159], v[108:111], v[32:47]
	v_exp_f32_e32 v149, v76
	v_exp_f32_e32 v150, v77
	v_exp_f32_e32 v151, v78
	v_mfma_f32_32x32x16_bf16 v[48:63], v[164:167], v[108:111], v[48:63]
	v_exp_f32_e32 v108, v79
	s_setprio 0
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C4: memory stage B
; Lower priority, prefetch V(j-1), read K buffer 0, wait, then barrier.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s39
	s_add_i32 s29, s23, s28
	buffer_load_dwordx4 v218, s[12:15], s29 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s29 offen lds
	ds_read_b128 v[64:67], v215
	ds_read_b128 v[164:167], v215 offset:32
	ds_read_b128 v[156:159], v215 offset:512
	ds_read_b128 v[170:173], v215 offset:544
	ds_read_b128 v[178:181], v215 offset:64
	ds_read_b128 v[182:185], v215 offset:96
	ds_read_b128 v[186:189], v215 offset:576
	ds_read_b128 v[190:193], v215 offset:608
	ds_read_b128 v[194:197], v215 offset:8320
	ds_read_b128 v[198:201], v215 offset:8352
	ds_read_b128 v[202:205], v215 offset:8832
	ds_read_b128 v[222:225], v215 offset:8864
	ds_read_b128 v[226:229], v215 offset:8384
	ds_read_b128 v[230:233], v215 offset:8416
	ds_read_b128 v[234:237], v215 offset:8896
	ds_read_b128 v[238:241], v215 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C5: compute stage B
; MMA0 -> v_s_0 while finishing v_p_1 softmax/sum/cast.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[80:95], v[64:67], v[112:115], 0
	v_exp_f32_e32 v109, v96
	v_exp_f32_e32 v110, v97
	v_exp_f32_e32 v111, v98
	v_mfma_f32_32x32x16_bf16 v[64:79], v[156:159], v[112:115], 0
	v_exp_f32_e32 v156, v99
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v101, v101
	v_mfma_f32_32x32x16_bf16 v[80:95], v[164:167], v[116:119], v[80:95]
	v_exp_f32_e32 v102, v102
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v152, v152
	v_mfma_f32_32x32x16_bf16 v[64:79], v[170:173], v[116:119], v[64:79]
	v_exp_f32_e32 v153, v153
	v_exp_f32_e32 v154, v154
	v_exp_f32_e32 v155, v155
	v_mfma_f32_32x32x16_bf16 v[80:95], v[178:181], v[120:123], v[80:95]
	v_exp_f32_e32 v157, v176
	v_exp_f32_e32 v158, v177
	v_exp_f32_e32 v159, v160
	v_mfma_f32_32x32x16_bf16 v[64:79], v[186:189], v[120:123], v[64:79]
	v_exp_f32_e32 v160, v161
	v_mfma_f32_32x32x16_bf16 v[80:95], v[182:185], v[124:127], v[80:95]
	v_add_f32_e32 v96, v162, v163
	v_add_f32_e32 v96, v96, v168
	v_add_f32_e32 v96, v96, v104
	v_add_f32_e32 v96, v96, v105
	v_add_f32_e32 v96, v96, v106
	v_mfma_f32_32x32x16_bf16 v[64:79], v[190:193], v[124:127], v[64:79]
	v_add_f32_e32 v96, v96, v107
	v_add_f32_e32 v96, v96, v144
	v_add_f32_e32 v96, v96, v145
	v_add_f32_e32 v96, v96, v146
	v_add_f32_e32 v96, v96, v147
	v_mfma_f32_32x32x16_bf16 v[80:95], v[194:197], v[128:131], v[80:95]
	v_add_f32_e32 v96, v96, v148
	v_add_f32_e32 v96, v96, v149
	v_add_f32_e32 v96, v96, v150
	v_add_f32_e32 v96, v96, v151
	v_add_f32_e32 v96, v96, v108
	v_mfma_f32_32x32x16_bf16 v[64:79], v[202:205], v[128:131], v[64:79]
	v_add_f32_e32 v96, v96, v109
	v_add_f32_e32 v96, v96, v110
	v_add_f32_e32 v96, v96, v111
	v_add_f32_e32 v96, v96, v156
	v_add_f32_e32 v96, v96, v100
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[132:135], v[80:95]
	v_add_f32_e32 v96, v96, v101
	v_add_f32_e32 v96, v96, v102
	v_add_f32_e32 v96, v96, v103
	v_add_f32_e32 v96, v96, v152
	v_add_f32_e32 v96, v96, v153
	v_mfma_f32_32x32x16_bf16 v[64:79], v[222:225], v[132:135], v[64:79]
	v_add_f32_e32 v96, v96, v154
	v_add_f32_e32 v96, v96, v155
	v_add_f32_e32 v96, v96, v157
	v_add_f32_e32 v96, v96, v158
	v_add_f32_e32 v96, v96, v159
	v_mfma_f32_32x32x16_bf16 v[80:95], v[226:229], v[136:139], v[80:95]
	v_add_f32_e32 v214, v96, v160
	v_mov_b32_e32 v216, v214
	s_nop 1
	v_permlane32_swap_b32_e64 v214, v216 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v162, v163
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v168, v104
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v105, v106
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[234:237], v[136:139], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v107, v144
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v109, v110
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v111, v156
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v100, v101
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v102, v103
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[230:233], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v145, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v147, v148
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v149, v150
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v151, v108
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v152, v153
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[238:241], v[140:143], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v154, v155
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v159, v160
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C6: memory stage B
; Prefetch K(j+1), read V buffer 1, apply causal mask.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s42
	s_add_i32 s28, s11, s28
	buffer_load_dwordx4 v218, s[16:19], s28 offen lds
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s28 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v221 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v221 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v221 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v221 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v221 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v221 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v221 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v221 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v221 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v221 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v221 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v221 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v221 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v221 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v221 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v221 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v221 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v221 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v221 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v221 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v221 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v221 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v221 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v221 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v221 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v221 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v221 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v221 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v221 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v221 offset:9536

	;;#ASMEND
	s_cmp_ge_i32 s27, s24
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v221 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_32
	v_add_u32_e32 v222, 32, v210
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v222, 0
	v_cmp_lt_i32_e64 s[46:47], v222, 1
	v_cndmask_b32_e64 v80, v80, v213, s[30:31]
	v_cndmask_b32_e64 v81, v81, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v222, 2
	v_cmp_lt_i32_e64 s[46:47], v222, 3
	v_cndmask_b32_e64 v82, v82, v213, s[30:31]
	v_cndmask_b32_e64 v83, v83, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v222, 8
	v_cmp_lt_i32_e64 s[46:47], v222, 9
	v_cndmask_b32_e64 v84, v84, v213, s[30:31]
	v_cndmask_b32_e64 v85, v85, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v222, 10
	v_cmp_lt_i32_e64 s[46:47], v222, 11
	v_cndmask_b32_e64 v86, v86, v213, s[30:31]
	v_cndmask_b32_e64 v87, v87, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v222, 16
	v_cmp_lt_i32_e64 s[46:47], v222, 17
	v_cndmask_b32_e64 v88, v88, v213, s[30:31]
	v_cndmask_b32_e64 v89, v89, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v222, 18
	v_cmp_lt_i32_e64 s[46:47], v222, 19
	v_cndmask_b32_e64 v90, v90, v213, s[30:31]
	v_cndmask_b32_e64 v91, v91, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v222, 24
	v_cmp_lt_i32_e64 s[46:47], v222, 25
	v_cndmask_b32_e64 v92, v92, v213, s[30:31]
	v_cndmask_b32_e64 v93, v93, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v222, 26
	v_cmp_lt_i32_e64 s[46:47], v222, 27
	v_cndmask_b32_e64 v94, v94, v213, s[30:31]
	v_cndmask_b32_e64 v95, v95, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v210, 0
	v_cmp_lt_i32_e64 s[46:47], v210, 1
	v_cndmask_b32_e64 v64, v64, v213, s[30:31]
	v_cndmask_b32_e64 v65, v65, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v210, 2
	v_cmp_lt_i32_e64 s[46:47], v210, 3
	v_cndmask_b32_e64 v66, v66, v213, s[30:31]
	v_cndmask_b32_e64 v67, v67, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v210, 8
	v_cmp_lt_i32_e64 s[46:47], v210, 9
	v_cndmask_b32_e64 v68, v68, v213, s[30:31]
	v_cndmask_b32_e64 v69, v69, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v210, 10
	v_cmp_lt_i32_e64 s[46:47], v210, 11
	v_cndmask_b32_e64 v70, v70, v213, s[30:31]
	v_cndmask_b32_e64 v71, v71, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v210, 16
	v_cmp_lt_i32_e64 s[46:47], v210, 17
	v_cndmask_b32_e64 v72, v72, v213, s[30:31]
	v_cndmask_b32_e64 v73, v73, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v210, 18
	v_cmp_lt_i32_e64 s[46:47], v210, 19
	v_cndmask_b32_e64 v74, v74, v213, s[30:31]
	v_cndmask_b32_e64 v75, v75, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v210, 24
	v_cmp_lt_i32_e64 s[46:47], v210, 25
	v_cndmask_b32_e64 v76, v76, v213, s[30:31]
	v_cndmask_b32_e64 v77, v77, v213, s[46:47]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[30:31], v210, 26
	v_cmp_lt_i32_e64 s[46:47], v210, 27
	v_cndmask_b32_e64 v78, v78, v213, s[30:31]
	v_cndmask_b32_e64 v79, v79, v213, s[46:47]
	;;#ASMEND
.LBB0_32:
	v_add_f32_e32 v209, v209, v216
	v_add_f32_e32 v209, v209, v214
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C7: compute stage B / loop yield
; Update O/m/l with v_p_1*V, start next v_p_0, then loop back-edge or epilogue.
; ==============================================================================
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[0:15], v[204:207], v[96:99], v[0:15]
	v_max_f32_e32 v204, v80, v81
	v_max3_f32 v204, v204, v82, v83
	v_max3_f32 v204, v204, v84, v85
	v_max3_f32 v204, v204, v86, v87
	v_max3_f32 v204, v204, v88, v89
	v_max3_f32 v204, v204, v90, v91
	v_mfma_f32_32x32x16_bf16 v[16:31], v[196:199], v[96:99], v[16:31]
	v_max3_f32 v196, v204, v92, v93
	v_max3_f32 v196, v196, v94, v95
	v_max3_f32 v196, v196, v64, v65
	v_max3_f32 v196, v196, v66, v67
	v_max3_f32 v196, v196, v68, v69
	v_max3_f32 v196, v196, v70, v71
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[96:99], v[32:47]
	v_max3_f32 v196, v196, v72, v73
	v_max3_f32 v196, v196, v74, v75
	v_max3_f32 v196, v196, v76, v77
	v_max3_f32 v196, v196, v78, v79
	v_mov_b32_e32 v197, v196
	s_nop 1
	v_permlane32_swap_b32_e64 v196, v197 bound_ctrl:1
	v_max_f32_e32 v196, v196, v197
	v_sub_f32_e32 v197, v196, v211
	v_cmp_ge_f32_e32 vcc, s26, v197
	s_cmp_eq_u64 vcc, exec
	v_mfma_f32_32x32x16_bf16 v[48:63], v[192:195], v[96:99], v[48:63]
	s_cbranch_scc0 .LBB0_35
.LBB0_33:
	v_mfma_f32_32x32x16_bf16 v[0:15], v[188:191], v[100:103], v[0:15]
	v_sub_f32_e32 v96, v80, v211
	v_sub_f32_e32 v97, v81, v211
	v_sub_f32_e32 v98, v82, v211
	v_sub_f32_e32 v83, v83, v211
	v_sub_f32_e32 v84, v84, v211
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[100:103], v[16:31]
	v_sub_f32_e32 v85, v85, v211
	v_sub_f32_e32 v86, v86, v211
	v_sub_f32_e32 v87, v87, v211
	v_sub_f32_e32 v88, v88, v211
	v_sub_f32_e32 v89, v89, v211
	v_mfma_f32_32x32x16_bf16 v[32:47], v[180:183], v[100:103], v[32:47]
	v_sub_f32_e32 v90, v90, v211
	v_sub_f32_e32 v91, v91, v211
	v_sub_f32_e32 v92, v92, v211
	v_sub_f32_e32 v93, v93, v211
	v_sub_f32_e32 v94, v94, v211
	v_mfma_f32_32x32x16_bf16 v[48:63], v[184:187], v[100:103], v[48:63]
	v_sub_f32_e32 v95, v95, v211
	v_sub_f32_e32 v82, v64, v211
	v_sub_f32_e32 v81, v65, v211
	v_sub_f32_e32 v80, v66, v211
	v_sub_f32_e32 v179, v67, v211
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[104:107], v[0:15]
	v_sub_f32_e32 v181, v68, v211
	v_sub_f32_e32 v180, v69, v211
	v_sub_f32_e32 v178, v70, v211
	v_sub_f32_e32 v177, v71, v211
	v_sub_f32_e32 v176, v72, v211
	v_mfma_f32_32x32x16_bf16 v[16:31], v[164:167], v[104:107], v[16:31]
	v_sub_f32_e32 v164, v73, v211
	v_sub_f32_e32 v163, v74, v211
	v_sub_f32_e32 v162, v75, v211
	v_sub_f32_e32 v161, v76, v211
	v_sub_f32_e32 v165, v77, v211
	v_mfma_f32_32x32x16_bf16 v[32:47], v[168:171], v[104:107], v[32:47]
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v160, v97
	v_exp_f32_e32 v97, v98
	v_mfma_f32_32x32x16_bf16 v[48:63], v[172:175], v[104:107], v[48:63]
	v_exp_f32_e32 v106, v83
	v_exp_f32_e32 v98, v84
	v_exp_f32_e32 v105, v85
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[108:111], v[0:15]
	v_exp_f32_e32 v99, v86
	v_exp_f32_e32 v104, v87
	v_exp_f32_e32 v100, v88
	v_mfma_f32_32x32x16_bf16 v[16:31], v[148:151], v[108:111], v[16:31]
	v_exp_f32_e32 v146, v89
	v_exp_f32_e32 v101, v90
	v_exp_f32_e32 v145, v91
	v_mfma_f32_32x32x16_bf16 v[32:47], v[152:155], v[108:111], v[32:47]
	v_exp_f32_e32 v102, v92
	v_exp_f32_e32 v144, v93
	v_exp_f32_e32 v103, v94
	v_mfma_f32_32x32x16_bf16 v[48:63], v[156:159], v[108:111], v[48:63]
	v_exp_f32_e32 v108, v95
	v_sub_f32_e32 v109, v78, v211
	v_sub_f32_e32 v107, v79, v211
	s_setprio 0
; ==============================================================================
; PASS 1 MAIN LOOP BACK-EDGE
; Increments mirror-pass j by 2, advances offsets, and branches to C0 or epilogue.
; ==============================================================================
	s_barrier
	s_add_u32 s8, s8, 2
	s_addc_u32 s9, s9, 0
	v_add_u32_e32 v210, 0xffffff80, v210
	s_addk_i32 s24, 0x80
	v_mov_b64_e32 v[64:65], s[20:21]
	v_cmp_lt_i64_e32 vcc, s[8:9], v[64:65]
	s_mov_b32 s28, s29
	s_cbranch_vccnz .LBB0_29
	s_branch .LBB0_37
.LBB0_34:
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C3 LAZY-RESCALE SLOW PATH
; Slow lazy-rescale path for C3; rescales O/P/l and rejoins the fast path.
; ==============================================================================
	v_sub_f32_e32 v96, v211, v192
	v_exp_f32_e32 v96, v96
	s_nop 0
	v_pk_mul_f32 v[14:15], v[96:97], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[96:97], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[96:97], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[96:97], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[96:97], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[96:97], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[96:97], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[96:97], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[96:97], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[96:97], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[96:97], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[96:97], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[96:97], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[96:97], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[96:97], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[96:97], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[96:97], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[96:97], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[96:97], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[96:97], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[96:97], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[96:97], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[96:97], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[96:97], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[96:97], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[96:97], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[96:97], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[96:97], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[96:97], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[96:97], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[96:97], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[96:97], v[48:49] op_sel_hi:[0,1]
	v_and_b32_e32 v99, 0xffff0000, v111
	v_lshlrev_b32_e32 v98, 16, v111
	v_and_b32_e32 v111, 0xffff0000, v110
	v_lshlrev_b32_e32 v110, 16, v110
	v_and_b32_e32 v195, 0xffff0000, v109
	v_lshlrev_b32_e32 v194, 16, v109
	v_and_b32_e32 v109, 0xffff0000, v108
	v_lshlrev_b32_e32 v108, 16, v108
	v_and_b32_e32 v197, 0xffff0000, v107
	v_lshlrev_b32_e32 v196, 16, v107
	v_and_b32_e32 v107, 0xffff0000, v106
	v_lshlrev_b32_e32 v106, 16, v106
	v_and_b32_e32 v199, 0xffff0000, v105
	v_lshlrev_b32_e32 v198, 16, v105
	v_and_b32_e32 v105, 0xffff0000, v104
	v_lshlrev_b32_e32 v104, 16, v104
	v_and_b32_e32 v201, 0xffff0000, v103
	v_lshlrev_b32_e32 v200, 16, v103
	v_and_b32_e32 v103, 0xffff0000, v102
	v_lshlrev_b32_e32 v102, 16, v102
	v_and_b32_e32 v203, 0xffff0000, v101
	v_lshlrev_b32_e32 v202, 16, v101
	v_and_b32_e32 v101, 0xffff0000, v100
	v_lshlrev_b32_e32 v100, 16, v100
	v_pk_mul_f32 v[204:205], v[96:97], v[100:101] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[96:97], v[202:203] op_sel_hi:[0,1]
	v_pk_mul_f32 v[202:203], v[96:97], v[102:103] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[96:97], v[200:201] op_sel_hi:[0,1]
	v_pk_mul_f32 v[200:201], v[96:97], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[96:97], v[198:199] op_sel_hi:[0,1]
	v_pk_mul_f32 v[198:199], v[96:97], v[106:107] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[96:97], v[196:197] op_sel_hi:[0,1]
	v_pk_mul_f32 v[196:197], v[96:97], v[108:109] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[96:97], v[194:195] op_sel_hi:[0,1]
	v_pk_mul_f32 v[194:195], v[96:97], v[110:111] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[96:97], v[98:99] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v111, v98, v99
	v_cvt_pk_bf16_f32 v110, v194, v195
	v_cvt_pk_bf16_f32 v109, v108, v109
	v_cvt_pk_bf16_f32 v108, v196, v197
	v_cvt_pk_bf16_f32 v107, v106, v107
	v_cvt_pk_bf16_f32 v106, v198, v199
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v200, v201
	v_cvt_pk_bf16_f32 v103, v102, v103
	v_cvt_pk_bf16_f32 v102, v202, v203
	v_cvt_pk_bf16_f32 v101, v100, v101
	v_cvt_pk_bf16_f32 v100, v204, v205
	v_mul_f32_e32 v209, v96, v209
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v211, v192
	s_branch .LBB0_30
.LBB0_35:
; ==============================================================================
; PASS 1 MAIN LOOP CLUSTER C7 LAZY-RESCALE SLOW PATH
; Slow lazy-rescale path for C7; rescales O/P/l and rejoins the fast path.
; ==============================================================================
	v_sub_f32_e32 v96, v211, v196
	v_exp_f32_e32 v96, v96
	s_nop 0
	v_pk_mul_f32 v[14:15], v[96:97], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[96:97], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[96:97], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[96:97], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[96:97], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[96:97], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[96:97], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[96:97], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[96:97], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[96:97], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[96:97], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[96:97], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[96:97], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[96:97], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[96:97], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[96:97], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[96:97], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[96:97], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[96:97], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[96:97], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[96:97], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[96:97], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[96:97], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[96:97], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[96:97], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[96:97], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[96:97], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[96:97], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[96:97], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[96:97], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[96:97], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[96:97], v[48:49] op_sel_hi:[0,1]
	v_and_b32_e32 v99, 0xffff0000, v111
	v_lshlrev_b32_e32 v98, 16, v111
	v_and_b32_e32 v111, 0xffff0000, v110
	v_lshlrev_b32_e32 v110, 16, v110
	v_and_b32_e32 v193, 0xffff0000, v109
	v_lshlrev_b32_e32 v192, 16, v109
	v_and_b32_e32 v109, 0xffff0000, v108
	v_lshlrev_b32_e32 v108, 16, v108
	v_and_b32_e32 v195, 0xffff0000, v107
	v_lshlrev_b32_e32 v194, 16, v107
	v_and_b32_e32 v107, 0xffff0000, v106
	v_lshlrev_b32_e32 v106, 16, v106
	v_and_b32_e32 v199, 0xffff0000, v105
	v_lshlrev_b32_e32 v198, 16, v105
	v_and_b32_e32 v105, 0xffff0000, v104
	v_lshlrev_b32_e32 v104, 16, v104
	v_and_b32_e32 v201, 0xffff0000, v103
	v_lshlrev_b32_e32 v200, 16, v103
	v_and_b32_e32 v103, 0xffff0000, v102
	v_lshlrev_b32_e32 v102, 16, v102
	v_and_b32_e32 v203, 0xffff0000, v101
	v_lshlrev_b32_e32 v202, 16, v101
	v_and_b32_e32 v101, 0xffff0000, v100
	v_lshlrev_b32_e32 v100, 16, v100
	v_pk_mul_f32 v[204:205], v[96:97], v[100:101] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[96:97], v[202:203] op_sel_hi:[0,1]
	v_pk_mul_f32 v[202:203], v[96:97], v[102:103] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[96:97], v[200:201] op_sel_hi:[0,1]
	v_pk_mul_f32 v[200:201], v[96:97], v[104:105] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[96:97], v[198:199] op_sel_hi:[0,1]
	v_pk_mul_f32 v[198:199], v[96:97], v[106:107] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[96:97], v[194:195] op_sel_hi:[0,1]
	v_pk_mul_f32 v[194:195], v[96:97], v[108:109] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[96:97], v[192:193] op_sel_hi:[0,1]
	v_pk_mul_f32 v[192:193], v[96:97], v[110:111] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[96:97], v[98:99] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v111, v98, v99
	v_cvt_pk_bf16_f32 v110, v192, v193
	v_cvt_pk_bf16_f32 v109, v108, v109
	v_cvt_pk_bf16_f32 v108, v194, v195
	v_cvt_pk_bf16_f32 v107, v106, v107
	v_cvt_pk_bf16_f32 v106, v198, v199
	v_cvt_pk_bf16_f32 v105, v104, v105
	v_cvt_pk_bf16_f32 v104, v200, v201
	v_cvt_pk_bf16_f32 v103, v102, v103
	v_cvt_pk_bf16_f32 v102, v202, v203
	v_cvt_pk_bf16_f32 v101, v100, v101
	v_cvt_pk_bf16_f32 v100, v204, v205
	v_mul_f32_e32 v209, v96, v209
	;;#ASMSTART
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v211, v196
	s_branch .LBB0_33
; ==============================================================================
; PASS 1 EPILOGUE ZERO-ITER ENTRY
; Zero-iteration mirror-pass path: materialize zero O accumulators for epilogue drain.
; ==============================================================================
.LBB0_36:
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v14, v15
	v_mov_b32_e32 v13, v15
	v_mov_b32_e32 v12, v15
	v_mov_b32_e32 v11, v15
	v_mov_b32_e32 v10, v15
	v_mov_b32_e32 v9, v15
	v_mov_b32_e32 v8, v15
	v_mov_b32_e32 v7, v15
	v_mov_b32_e32 v6, v15
	v_mov_b32_e32 v5, v15
	v_mov_b32_e32 v4, v15
	v_mov_b32_e32 v3, v15
	v_mov_b32_e32 v2, v15
	v_mov_b32_e32 v1, v15
	v_mov_b32_e32 v0, v15
	v_mov_b32_e32 v31, v15
	v_mov_b32_e32 v30, v15
	v_mov_b32_e32 v29, v15
	v_mov_b32_e32 v28, v15
	v_mov_b32_e32 v27, v15
	v_mov_b32_e32 v26, v15
	v_mov_b32_e32 v25, v15
	v_mov_b32_e32 v24, v15
	v_mov_b32_e32 v23, v15
	v_mov_b32_e32 v22, v15
	v_mov_b32_e32 v21, v15
	v_mov_b32_e32 v20, v15
	v_mov_b32_e32 v19, v15
	v_mov_b32_e32 v18, v15
	v_mov_b32_e32 v17, v15
	v_mov_b32_e32 v16, v15
	v_mov_b32_e32 v47, v15
	v_mov_b32_e32 v46, v15
	v_mov_b32_e32 v45, v15
	v_mov_b32_e32 v44, v15
	v_mov_b32_e32 v43, v15
	v_mov_b32_e32 v42, v15
	v_mov_b32_e32 v41, v15
	v_mov_b32_e32 v40, v15
	v_mov_b32_e32 v39, v15
	v_mov_b32_e32 v38, v15
	v_mov_b32_e32 v37, v15
	v_mov_b32_e32 v36, v15
	v_mov_b32_e32 v35, v15
	v_mov_b32_e32 v34, v15
	v_mov_b32_e32 v33, v15
	v_mov_b32_e32 v32, v15
	v_mov_b32_e32 v63, v15
	v_mov_b32_e32 v62, v15
	v_mov_b32_e32 v61, v15
	v_mov_b32_e32 v60, v15
	v_mov_b32_e32 v59, v15
	v_mov_b32_e32 v58, v15
	v_mov_b32_e32 v57, v15
	v_mov_b32_e32 v56, v15
	v_mov_b32_e32 v55, v15
	v_mov_b32_e32 v54, v15
	v_mov_b32_e32 v53, v15
	v_mov_b32_e32 v52, v15
	v_mov_b32_e32 v51, v15
	v_mov_b32_e32 v50, v15
	v_mov_b32_e32 v49, v15
	v_mov_b32_e32 v48, v15
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E0: memory drain A
; Prefetch V(max_m3), read K buffer 1, wait, and synchronize.
; ==============================================================================
; ==============================================================================
; PASS 1 EPILOGUE START - drain final resident tiles
; Mirror pass epilogue drain for the final three KV tiles.
; ==============================================================================
.LBB0_37:
	s_mov_b32 m0, s25
	s_lshl_b32 s8, s22, 6
	s_add_i32 s9, s8, 0xffffff40
	s_mul_i32 s11, s44, s9
	buffer_load_dwordx4 v218, s[12:15], s11 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s11 offen lds
	ds_read_b128 v[84:87], v215 offset:34048
	ds_read_b128 v[152:155], v215 offset:34080
	ds_read_b128 v[148:151], v215 offset:34560
	ds_read_b128 v[156:159], v215 offset:34592
	ds_read_b128 v[166:169], v215 offset:34112
	ds_read_b128 v[170:173], v215 offset:34144
	ds_read_b128 v[182:185], v215 offset:34624
	ds_read_b128 v[186:189], v215 offset:34656
	ds_read_b128 v[190:193], v215 offset:42368
	ds_read_b128 v[194:197], v215 offset:42400
	ds_read_b128 v[198:201], v215 offset:42880
	ds_read_b128 v[202:205], v215 offset:42912
	ds_read_b128 v[222:225], v215 offset:42432
	ds_read_b128 v[226:229], v215 offset:42464
	ds_read_b128 v[230:233], v215 offset:42944
	ds_read_b128 v[234:237], v215 offset:42976
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E1: compute drain A
; MMA0 -> v_s_1; finish v_p_0 softmax second half and update l_row.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[64:79], v[84:87], v[112:115], 0
	v_exp_f32_e32 v110, v82
	v_exp_f32_e32 v111, v81
	v_exp_f32_e32 v147, v80
	v_mfma_f32_32x32x16_bf16 v[80:95], v[148:151], v[112:115], 0
	v_exp_f32_e32 v148, v179
	v_exp_f32_e32 v149, v181
	v_exp_f32_e32 v150, v180
	v_mfma_f32_32x32x16_bf16 v[64:79], v[152:155], v[116:119], v[64:79]
	v_exp_f32_e32 v151, v178
	v_exp_f32_e32 v152, v177
	v_exp_f32_e32 v153, v176
	v_mfma_f32_32x32x16_bf16 v[80:95], v[156:159], v[116:119], v[80:95]
	v_exp_f32_e32 v154, v164
	v_exp_f32_e32 v155, v163
	v_exp_f32_e32 v156, v162
	v_mfma_f32_32x32x16_bf16 v[64:79], v[166:169], v[120:123], v[64:79]
	v_exp_f32_e32 v157, v161
	v_exp_f32_e32 v158, v165
	v_exp_f32_e32 v159, v109
	v_mfma_f32_32x32x16_bf16 v[80:95], v[182:185], v[120:123], v[80:95]
	v_exp_f32_e32 v161, v107
	v_mfma_f32_32x32x16_bf16 v[64:79], v[170:173], v[124:127], v[64:79]
	v_add_f32_e32 v107, v96, v160
	v_add_f32_e32 v107, v107, v97
	v_add_f32_e32 v107, v107, v106
	v_add_f32_e32 v107, v107, v98
	v_add_f32_e32 v107, v107, v105
	v_mfma_f32_32x32x16_bf16 v[80:95], v[186:189], v[124:127], v[80:95]
	v_add_f32_e32 v107, v107, v99
	v_add_f32_e32 v107, v107, v104
	v_add_f32_e32 v107, v107, v100
	v_add_f32_e32 v107, v107, v146
	v_add_f32_e32 v107, v107, v101
	v_mfma_f32_32x32x16_bf16 v[64:79], v[190:193], v[128:131], v[64:79]
	v_add_f32_e32 v107, v107, v145
	v_add_f32_e32 v107, v107, v102
	v_add_f32_e32 v107, v107, v144
	v_add_f32_e32 v107, v107, v103
	v_add_f32_e32 v107, v107, v108
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[128:131], v[80:95]
	v_add_f32_e32 v107, v107, v110
	v_add_f32_e32 v107, v107, v111
	v_add_f32_e32 v107, v107, v147
	v_add_f32_e32 v107, v107, v148
	v_add_f32_e32 v107, v107, v149
	v_mfma_f32_32x32x16_bf16 v[64:79], v[194:197], v[132:135], v[64:79]
	v_add_f32_e32 v107, v107, v150
	v_add_f32_e32 v107, v107, v151
	v_add_f32_e32 v107, v107, v152
	v_add_f32_e32 v107, v107, v153
	v_add_f32_e32 v107, v107, v154
	v_mfma_f32_32x32x16_bf16 v[80:95], v[202:205], v[132:135], v[80:95]
	v_add_f32_e32 v107, v107, v155
	v_add_f32_e32 v107, v107, v156
	v_add_f32_e32 v107, v107, v157
	v_add_f32_e32 v107, v107, v158
	v_add_f32_e32 v107, v107, v159
	v_mfma_f32_32x32x16_bf16 v[64:79], v[222:225], v[136:139], v[64:79]
	v_add_f32_e32 v213, v107, v161
	v_mov_b32_e32 v214, v213
	s_nop 1
	v_permlane32_swap_b32_e64 v213, v214 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v96, v160
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v97, v106
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v98, v105
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[230:233], v[136:139], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v99, v104
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v110, v111
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v147, v148
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v149, v150
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v151, v152
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[226:229], v[140:143], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v100, v146
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v101, v145
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v102, v144
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v103, v108
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v153, v154
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[234:237], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v155, v156
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v159, v161
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E2: memory drain A
; Prefetch K(max_m1), read V buffer 0, causally mask v_s_1.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s41
	s_lshl_b32 s9, s20, 6
	s_mul_i32 s44, s44, s9
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	buffer_load_dwordx4 v218, s[16:19], s44 offen lds
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v219, s[16:19], s44 offen lds
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v220 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v220 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v220 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v220 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v220 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v220 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v220 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v220 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v220 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v220 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v220 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v220 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v220 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v220 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v220 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v220 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v220 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v220 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v220 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v220 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v220 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v220 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v220 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v220 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v220 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v220 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v220 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v220 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v220 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v220 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v220 offset:9536

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v220 offset:9664

	;;#ASMEND
	s_add_i32 s16, s8, 0xffffff80
	s_cmp_ge_i32 s27, s16
	v_lshl_or_b32 v222, v208, 2, s8
	s_cbranch_scc1 .LBB0_39
	v_sub_u32_e32 v210, s3, v222
	v_add_u32_e32 v210, v210, v212
	v_add_u32_e32 v216, 0xc0, v210
	v_add_u32_e32 v210, 0xa0, v210
	v_mov_b32_e32 v223, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v216, 0
	v_cmp_lt_i32_e64 s[18:19], v216, 1
	v_cndmask_b32_e64 v64, v64, v223, s[16:17]
	v_cndmask_b32_e64 v65, v65, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v216, 2
	v_cmp_lt_i32_e64 s[18:19], v216, 3
	v_cndmask_b32_e64 v66, v66, v223, s[16:17]
	v_cndmask_b32_e64 v67, v67, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v216, 8
	v_cmp_lt_i32_e64 s[18:19], v216, 9
	v_cndmask_b32_e64 v68, v68, v223, s[16:17]
	v_cndmask_b32_e64 v69, v69, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v216, 10
	v_cmp_lt_i32_e64 s[18:19], v216, 11
	v_cndmask_b32_e64 v70, v70, v223, s[16:17]
	v_cndmask_b32_e64 v71, v71, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v216, 16
	v_cmp_lt_i32_e64 s[18:19], v216, 17
	v_cndmask_b32_e64 v72, v72, v223, s[16:17]
	v_cndmask_b32_e64 v73, v73, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v216, 18
	v_cmp_lt_i32_e64 s[18:19], v216, 19
	v_cndmask_b32_e64 v74, v74, v223, s[16:17]
	v_cndmask_b32_e64 v75, v75, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v216, 24
	v_cmp_lt_i32_e64 s[18:19], v216, 25
	v_cndmask_b32_e64 v76, v76, v223, s[16:17]
	v_cndmask_b32_e64 v77, v77, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v216, 26
	v_cmp_lt_i32_e64 s[18:19], v216, 27
	v_cndmask_b32_e64 v78, v78, v223, s[16:17]
	v_cndmask_b32_e64 v79, v79, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v210, 0
	v_cmp_lt_i32_e64 s[18:19], v210, 1
	v_cndmask_b32_e64 v80, v80, v223, s[16:17]
	v_cndmask_b32_e64 v81, v81, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v210, 2
	v_cmp_lt_i32_e64 s[18:19], v210, 3
	v_cndmask_b32_e64 v82, v82, v223, s[16:17]
	v_cndmask_b32_e64 v83, v83, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v210, 8
	v_cmp_lt_i32_e64 s[18:19], v210, 9
	v_cndmask_b32_e64 v84, v84, v223, s[16:17]
	v_cndmask_b32_e64 v85, v85, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v210, 10
	v_cmp_lt_i32_e64 s[18:19], v210, 11
	v_cndmask_b32_e64 v86, v86, v223, s[16:17]
	v_cndmask_b32_e64 v87, v87, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v210, 16
	v_cmp_lt_i32_e64 s[18:19], v210, 17
	v_cndmask_b32_e64 v88, v88, v223, s[16:17]
	v_cndmask_b32_e64 v89, v89, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v210, 18
	v_cmp_lt_i32_e64 s[18:19], v210, 19
	v_cndmask_b32_e64 v90, v90, v223, s[16:17]
	v_cndmask_b32_e64 v91, v91, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v210, 24
	v_cmp_lt_i32_e64 s[18:19], v210, 25
	v_cndmask_b32_e64 v92, v92, v223, s[16:17]
	v_cndmask_b32_e64 v93, v93, v223, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v210, 26
	v_cmp_lt_i32_e64 s[18:19], v210, 27
	v_cndmask_b32_e64 v94, v94, v223, s[16:17]
	v_cndmask_b32_e64 v95, v95, v223, s[18:19]
	;;#ASMEND
.LBB0_39:
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E3: compute drain A
; P*V for v_p_0 plus unconditional rescale into mirror-pass row state.
; ==============================================================================
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[0:15], v[192:195], v[96:99], v[0:15]
	v_max_f32_e32 v192, v64, v65
	v_max3_f32 v192, v192, v66, v67
	v_max3_f32 v192, v192, v68, v69
	v_max3_f32 v192, v192, v70, v71
	v_max3_f32 v192, v192, v72, v73
	v_mfma_f32_32x32x16_bf16 v[16:31], v[196:199], v[96:99], v[16:31]
	v_max3_f32 v192, v192, v74, v75
	v_max3_f32 v192, v192, v76, v77
	v_max3_f32 v192, v192, v78, v79
	v_max3_f32 v192, v192, v80, v81
	v_max3_f32 v192, v192, v82, v83
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[96:99], v[32:47]
	v_max3_f32 v192, v192, v84, v85
	v_max3_f32 v192, v192, v86, v87
	v_max3_f32 v192, v192, v88, v89
	v_max3_f32 v192, v192, v90, v91
	v_max3_f32 v192, v192, v92, v93
	v_mfma_f32_32x32x16_bf16 v[48:63], v[204:207], v[96:99], v[48:63]
	v_max3_f32 v96, v192, v94, v95
	v_mov_b32_e32 v97, v96
	s_nop 1
	v_permlane32_swap_b32_e64 v96, v97 bound_ctrl:1
	v_max3_f32 v223, v211, v96, v97
	v_sub_f32_e32 v96, v211, v223
	v_sub_f32_e32 v64, v64, v223
	v_mfma_f32_32x32x16_bf16 v[0:15], v[176:179], v[100:103], v[0:15]
	v_sub_f32_e32 v65, v65, v223
	v_sub_f32_e32 v66, v66, v223
	v_sub_f32_e32 v67, v67, v223
	v_sub_f32_e32 v68, v68, v223
	v_sub_f32_e32 v69, v69, v223
	v_mfma_f32_32x32x16_bf16 v[16:31], v[180:183], v[100:103], v[16:31]
	v_sub_f32_e32 v70, v70, v223
	v_sub_f32_e32 v71, v71, v223
	v_sub_f32_e32 v72, v72, v223
	v_sub_f32_e32 v73, v73, v223
	v_sub_f32_e32 v74, v74, v223
	v_mfma_f32_32x32x16_bf16 v[32:47], v[184:187], v[100:103], v[32:47]
	v_sub_f32_e32 v75, v75, v223
	v_sub_f32_e32 v76, v76, v223
	v_sub_f32_e32 v77, v77, v223
	v_sub_f32_e32 v78, v78, v223
	v_sub_f32_e32 v79, v79, v223
	v_mfma_f32_32x32x16_bf16 v[48:63], v[188:191], v[100:103], v[48:63]
	v_sub_f32_e32 v97, v80, v223
	v_sub_f32_e32 v98, v81, v223
	v_sub_f32_e32 v99, v82, v223
	v_sub_f32_e32 v100, v83, v223
	v_sub_f32_e32 v101, v84, v223
	v_mfma_f32_32x32x16_bf16 v[0:15], v[160:163], v[104:107], v[0:15]
	v_sub_f32_e32 v102, v85, v223
	v_sub_f32_e32 v103, v86, v223
	v_sub_f32_e32 v160, v87, v223
	v_sub_f32_e32 v161, v88, v223
	v_sub_f32_e32 v162, v89, v223
	v_mfma_f32_32x32x16_bf16 v[16:31], v[164:167], v[104:107], v[16:31]
	v_sub_f32_e32 v163, v90, v223
	v_sub_f32_e32 v164, v91, v223
	v_sub_f32_e32 v165, v92, v223
	v_sub_f32_e32 v166, v93, v223
	v_sub_f32_e32 v167, v94, v223
	v_mfma_f32_32x32x16_bf16 v[32:47], v[168:171], v[104:107], v[32:47]
	v_exp_f32_e32 v210, v96
	v_exp_f32_e32 v96, v64
	v_exp_f32_e32 v168, v65
	v_mfma_f32_32x32x16_bf16 v[48:63], v[172:175], v[104:107], v[48:63]
	v_exp_f32_e32 v104, v66
	v_exp_f32_e32 v105, v67
	v_exp_f32_e32 v106, v68
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[108:111], v[0:15]
	v_exp_f32_e32 v107, v69
	v_exp_f32_e32 v144, v70
	v_exp_f32_e32 v145, v71
	v_mfma_f32_32x32x16_bf16 v[16:31], v[148:151], v[108:111], v[16:31]
	v_exp_f32_e32 v146, v72
	v_exp_f32_e32 v147, v73
	v_exp_f32_e32 v148, v74
	v_mfma_f32_32x32x16_bf16 v[32:47], v[152:155], v[108:111], v[32:47]
	v_exp_f32_e32 v149, v75
	v_exp_f32_e32 v150, v76
	v_exp_f32_e32 v151, v77
	v_mfma_f32_32x32x16_bf16 v[48:63], v[156:159], v[108:111], v[48:63]
	v_exp_f32_e32 v108, v78
	v_exp_f32_e32 v109, v79
	v_sub_f32_e32 v110, v95, v223
	v_pk_mul_f32 v[14:15], v[210:211], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[210:211], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[210:211], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[210:211], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[210:211], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[210:211], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[210:211], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[210:211], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[210:211], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[210:211], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[210:211], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[210:211], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[210:211], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[210:211], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[210:211], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[210:211], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[210:211], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[210:211], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[210:211], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[210:211], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[210:211], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[210:211], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[210:211], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[210:211], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[210:211], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[210:211], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[210:211], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[210:211], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[210:211], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[210:211], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[210:211], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[210:211], v[48:49] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
	s_setprio 0
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E4: memory drain B
; Prefetch V(max_m2), read K buffer 0, wait, and synchronize.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s39
	s_add_i32 s11, s11, s34
	buffer_load_dwordx4 v218, s[12:15], s11 offen lds
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s11 offen lds
	ds_read_b128 v[64:67], v215
	ds_read_b128 v[154:157], v215 offset:32
	ds_read_b128 v[170:173], v215 offset:512
	ds_read_b128 v[174:177], v215 offset:544
	ds_read_b128 v[178:181], v215 offset:64
	ds_read_b128 v[182:185], v215 offset:96
	ds_read_b128 v[186:189], v215 offset:576
	ds_read_b128 v[190:193], v215 offset:608
	ds_read_b128 v[194:197], v215 offset:8320
	ds_read_b128 v[198:201], v215 offset:8352
	ds_read_b128 v[202:205], v215 offset:8832
	ds_read_b128 v[224:227], v215 offset:8864
	ds_read_b128 v[228:231], v215 offset:8384
	ds_read_b128 v[232:235], v215 offset:8416
	ds_read_b128 v[236:239], v215 offset:8896
	ds_read_b128 v[240:243], v215 offset:8928
	s_waitcnt vmcnt(4) lgkmcnt(0)
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E5: compute drain B
; MMA0 -> v_s_0; fold rescale into l_row and finish v_p_1 softmax.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[80:95], v[64:67], v[112:115], 0
	v_exp_f32_e32 v111, v97
	v_exp_f32_e32 v152, v98
	v_exp_f32_e32 v153, v99
	v_mfma_f32_32x32x16_bf16 v[64:79], v[170:173], v[112:115], 0
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v101, v101
	v_exp_f32_e32 v102, v102
	v_mfma_f32_32x32x16_bf16 v[80:95], v[154:157], v[116:119], v[80:95]
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v154, v160
	v_exp_f32_e32 v155, v161
	v_mfma_f32_32x32x16_bf16 v[64:79], v[174:177], v[116:119], v[64:79]
	v_exp_f32_e32 v156, v162
	v_exp_f32_e32 v157, v163
	v_exp_f32_e32 v158, v164
	v_mfma_f32_32x32x16_bf16 v[80:95], v[178:181], v[120:123], v[80:95]
	v_exp_f32_e32 v159, v165
	v_exp_f32_e32 v160, v166
	v_exp_f32_e32 v161, v167
	v_mfma_f32_32x32x16_bf16 v[64:79], v[186:189], v[120:123], v[64:79]
	v_exp_f32_e32 v162, v110
	v_mfma_f32_32x32x16_bf16 v[80:95], v[182:185], v[124:127], v[80:95]
	v_add_f32_e32 v97, v96, v168
	v_add_f32_e32 v97, v97, v104
	v_add_f32_e32 v97, v97, v105
	v_add_f32_e32 v97, v97, v106
	v_add_f32_e32 v97, v97, v107
	v_mfma_f32_32x32x16_bf16 v[64:79], v[190:193], v[124:127], v[64:79]
	v_add_f32_e32 v97, v97, v144
	v_add_f32_e32 v97, v97, v145
	v_add_f32_e32 v97, v97, v146
	v_add_f32_e32 v97, v97, v147
	v_add_f32_e32 v97, v97, v148
	v_mfma_f32_32x32x16_bf16 v[80:95], v[194:197], v[128:131], v[80:95]
	v_add_f32_e32 v97, v97, v149
	v_add_f32_e32 v97, v97, v150
	v_add_f32_e32 v97, v97, v151
	v_add_f32_e32 v97, v97, v108
	v_add_f32_e32 v97, v97, v109
	v_mfma_f32_32x32x16_bf16 v[64:79], v[202:205], v[128:131], v[64:79]
	v_add_f32_e32 v97, v97, v111
	v_add_f32_e32 v97, v97, v152
	v_add_f32_e32 v97, v97, v153
	v_add_f32_e32 v97, v97, v100
	v_add_f32_e32 v97, v97, v101
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[132:135], v[80:95]
	v_add_f32_e32 v97, v97, v102
	v_add_f32_e32 v97, v97, v103
	v_add_f32_e32 v97, v97, v154
	v_add_f32_e32 v97, v97, v155
	v_add_f32_e32 v97, v97, v156
	v_mfma_f32_32x32x16_bf16 v[64:79], v[224:227], v[132:135], v[64:79]
	v_add_f32_e32 v97, v97, v157
	v_add_f32_e32 v97, v97, v158
	v_add_f32_e32 v97, v97, v159
	v_add_f32_e32 v97, v97, v160
	v_add_f32_e32 v97, v97, v161
	v_mfma_f32_32x32x16_bf16 v[80:95], v[228:231], v[136:139], v[80:95]
	v_add_f32_e32 v211, v97, v162
	v_mov_b32_e32 v216, v211
	s_nop 1
	v_permlane32_swap_b32_e64 v216, v211 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v96, v168
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v104, v105
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v106, v107
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[236:239], v[136:139], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v144, v145
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v111, v152
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v153, v100
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v101, v102
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v103, v154
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[232:235], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v146, v147
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v148, v149
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v150, v151
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v108, v109
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v155, v156
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[240:243], v[140:143], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v157, v158
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v159, v160
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v161, v162
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E6: memory drain B
; Read V buffer 1 and causally mask v_s_0.
; ==============================================================================
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b16 v[192:193], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[194:195], v221 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v221 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v221 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v221 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v221 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v221 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v221 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[196:197], v221 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[198:199], v221 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[176:177], v221 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[178:179], v221 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v221 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v221 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v221 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v221 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[200:201], v221 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[202:203], v221 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[184:185], v221 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[186:187], v221 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v221 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v221 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v221 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v221 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[204:205], v221 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[206:207], v221 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[188:189], v221 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[190:191], v221 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[180:181], v221 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[182:183], v221 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v221 offset:9536

	;;#ASMEND
	s_cmp_ge_i32 s27, s9
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v221 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_41
	v_sub_u32_e32 v222, s3, v222
	v_add_u32_e32 v222, v222, v212
	v_add_u32_e32 v224, 0x80, v222
	v_add_u32_e32 v222, 0x60, v222
	v_mov_b32_e32 v225, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v224, 0
	v_cmp_lt_i32_e64 s[18:19], v224, 1
	v_cndmask_b32_e64 v80, v80, v225, s[16:17]
	v_cndmask_b32_e64 v81, v81, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v224, 2
	v_cmp_lt_i32_e64 s[18:19], v224, 3
	v_cndmask_b32_e64 v82, v82, v225, s[16:17]
	v_cndmask_b32_e64 v83, v83, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v224, 8
	v_cmp_lt_i32_e64 s[18:19], v224, 9
	v_cndmask_b32_e64 v84, v84, v225, s[16:17]
	v_cndmask_b32_e64 v85, v85, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v224, 10
	v_cmp_lt_i32_e64 s[18:19], v224, 11
	v_cndmask_b32_e64 v86, v86, v225, s[16:17]
	v_cndmask_b32_e64 v87, v87, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v224, 16
	v_cmp_lt_i32_e64 s[18:19], v224, 17
	v_cndmask_b32_e64 v88, v88, v225, s[16:17]
	v_cndmask_b32_e64 v89, v89, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v224, 18
	v_cmp_lt_i32_e64 s[18:19], v224, 19
	v_cndmask_b32_e64 v90, v90, v225, s[16:17]
	v_cndmask_b32_e64 v91, v91, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v224, 24
	v_cmp_lt_i32_e64 s[18:19], v224, 25
	v_cndmask_b32_e64 v92, v92, v225, s[16:17]
	v_cndmask_b32_e64 v93, v93, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v224, 26
	v_cmp_lt_i32_e64 s[18:19], v224, 27
	v_cndmask_b32_e64 v94, v94, v225, s[16:17]
	v_cndmask_b32_e64 v95, v95, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v222, 0
	v_cmp_lt_i32_e64 s[18:19], v222, 1
	v_cndmask_b32_e64 v64, v64, v225, s[16:17]
	v_cndmask_b32_e64 v65, v65, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v222, 2
	v_cmp_lt_i32_e64 s[18:19], v222, 3
	v_cndmask_b32_e64 v66, v66, v225, s[16:17]
	v_cndmask_b32_e64 v67, v67, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v222, 8
	v_cmp_lt_i32_e64 s[18:19], v222, 9
	v_cndmask_b32_e64 v68, v68, v225, s[16:17]
	v_cndmask_b32_e64 v69, v69, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v222, 10
	v_cmp_lt_i32_e64 s[18:19], v222, 11
	v_cndmask_b32_e64 v70, v70, v225, s[16:17]
	v_cndmask_b32_e64 v71, v71, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v222, 16
	v_cmp_lt_i32_e64 s[18:19], v222, 17
	v_cndmask_b32_e64 v72, v72, v225, s[16:17]
	v_cndmask_b32_e64 v73, v73, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v222, 18
	v_cmp_lt_i32_e64 s[18:19], v222, 19
	v_cndmask_b32_e64 v74, v74, v225, s[16:17]
	v_cndmask_b32_e64 v75, v75, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v222, 24
	v_cmp_lt_i32_e64 s[18:19], v222, 25
	v_cndmask_b32_e64 v76, v76, v225, s[16:17]
	v_cndmask_b32_e64 v77, v77, v225, s[18:19]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[16:17], v222, 26
	v_cmp_lt_i32_e64 s[18:19], v222, 27
	v_cndmask_b32_e64 v78, v78, v225, s[16:17]
	v_cndmask_b32_e64 v79, v79, v225, s[18:19]
	;;#ASMEND
.LBB0_41:
	s_waitcnt vmcnt(2) lgkmcnt(0)
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E7: compute drain B
; P*V for v_p_1, update m/l/O, and prepare final v_p_0 slice.
; ==============================================================================
	s_barrier
	s_setprio 1
	v_mfma_f32_32x32x16_bf16 v[0:15], v[192:195], v[96:99], v[0:15]
	s_nop 2
	v_max_f32_e32 v192, v80, v81
	v_max3_f32 v192, v192, v82, v83
	v_max3_f32 v192, v192, v84, v85
	v_max3_f32 v192, v192, v86, v87
	v_max3_f32 v192, v192, v88, v89
	v_mfma_f32_32x32x16_bf16 v[16:31], v[196:199], v[96:99], v[16:31]
	v_max3_f32 v192, v192, v90, v91
	v_max3_f32 v192, v192, v92, v93
	v_max3_f32 v192, v192, v94, v95
	v_max3_f32 v192, v192, v64, v65
	v_max3_f32 v192, v192, v66, v67
	v_mfma_f32_32x32x16_bf16 v[32:47], v[200:203], v[96:99], v[32:47]
	v_max3_f32 v192, v192, v68, v69
	v_max3_f32 v192, v192, v70, v71
	v_max3_f32 v192, v192, v72, v73
	v_max3_f32 v192, v192, v74, v75
	v_max3_f32 v192, v192, v76, v77
	v_mfma_f32_32x32x16_bf16 v[48:63], v[204:207], v[96:99], v[48:63]
	v_max3_f32 v96, v192, v78, v79
	v_mov_b32_e32 v97, v96
	s_nop 1
	v_permlane32_swap_b32_e64 v96, v97 bound_ctrl:1
	v_max3_f32 v192, v223, v96, v97
	v_sub_f32_e32 v96, v223, v192
	v_sub_f32_e32 v80, v80, v192
	v_mfma_f32_32x32x16_bf16 v[0:15], v[168:171], v[100:103], v[0:15]
	v_sub_f32_e32 v81, v81, v192
	v_sub_f32_e32 v82, v82, v192
	v_sub_f32_e32 v83, v83, v192
	v_sub_f32_e32 v84, v84, v192
	v_sub_f32_e32 v85, v85, v192
	v_mfma_f32_32x32x16_bf16 v[16:31], v[176:179], v[100:103], v[16:31]
	v_sub_f32_e32 v86, v86, v192
	v_sub_f32_e32 v87, v87, v192
	v_sub_f32_e32 v88, v88, v192
	v_sub_f32_e32 v89, v89, v192
	v_sub_f32_e32 v90, v90, v192
	v_mfma_f32_32x32x16_bf16 v[32:47], v[184:187], v[100:103], v[32:47]
	v_sub_f32_e32 v91, v91, v192
	v_sub_f32_e32 v92, v92, v192
	v_sub_f32_e32 v93, v93, v192
	v_sub_f32_e32 v94, v94, v192
	v_sub_f32_e32 v95, v95, v192
	v_mfma_f32_32x32x16_bf16 v[48:63], v[188:191], v[100:103], v[48:63]
	v_sub_f32_e32 v97, v64, v192
	v_sub_f32_e32 v98, v65, v192
	v_sub_f32_e32 v99, v66, v192
	v_sub_f32_e32 v100, v67, v192
	v_sub_f32_e32 v101, v68, v192
	v_mfma_f32_32x32x16_bf16 v[0:15], v[152:155], v[104:107], v[0:15]
	v_sub_f32_e32 v102, v69, v192
	v_sub_f32_e32 v103, v70, v192
	v_sub_f32_e32 v152, v71, v192
	v_sub_f32_e32 v153, v72, v192
	v_sub_f32_e32 v154, v73, v192
	v_mfma_f32_32x32x16_bf16 v[16:31], v[160:163], v[104:107], v[16:31]
	v_sub_f32_e32 v155, v74, v192
	v_sub_f32_e32 v160, v75, v192
	v_sub_f32_e32 v161, v76, v192
	v_sub_f32_e32 v162, v77, v192
	v_sub_f32_e32 v163, v78, v192
	v_mfma_f32_32x32x16_bf16 v[32:47], v[172:175], v[104:107], v[32:47]
	v_exp_f32_e32 v176, v96
	v_exp_f32_e32 v96, v80
	v_exp_f32_e32 v168, v81
	v_mfma_f32_32x32x16_bf16 v[48:63], v[180:183], v[104:107], v[48:63]
	v_exp_f32_e32 v104, v82
	v_exp_f32_e32 v105, v83
	v_exp_f32_e32 v106, v84
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[108:111], v[0:15]
	v_exp_f32_e32 v107, v85
	v_exp_f32_e32 v144, v86
	v_exp_f32_e32 v145, v87
	v_mfma_f32_32x32x16_bf16 v[16:31], v[148:151], v[108:111], v[16:31]
	v_exp_f32_e32 v146, v88
	v_exp_f32_e32 v147, v89
	v_exp_f32_e32 v148, v90
	v_mfma_f32_32x32x16_bf16 v[32:47], v[156:159], v[108:111], v[32:47]
	v_exp_f32_e32 v149, v91
	v_exp_f32_e32 v150, v92
	v_exp_f32_e32 v151, v93
	v_mfma_f32_32x32x16_bf16 v[48:63], v[164:167], v[108:111], v[48:63]
	v_exp_f32_e32 v108, v94
	v_exp_f32_e32 v109, v95
	v_sub_f32_e32 v110, v79, v192
	v_pk_mul_f32 v[14:15], v[176:177], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[176:177], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[176:177], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[176:177], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[176:177], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[176:177], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[176:177], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[176:177], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[176:177], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[176:177], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[176:177], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[176:177], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[176:177], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[176:177], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[176:177], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[176:177], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[46:47], v[176:177], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[44:45], v[176:177], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[42:43], v[176:177], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[40:41], v[176:177], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[38:39], v[176:177], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[36:37], v[176:177], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[34:35], v[176:177], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[32:33], v[176:177], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[62:63], v[176:177], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[60:61], v[176:177], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[58:59], v[176:177], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[176:177], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[54:55], v[176:177], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[176:177], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[50:51], v[176:177], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[48:49], v[176:177], v[48:49] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
	s_setprio 0
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E8: memory drain C
; Read K/V fragments for the last score/P*V stage.
; ==============================================================================
	s_barrier
	s_mov_b32 m0, s25
	s_nop 0
	buffer_load_dwordx4 v218, s[12:15], s44 offen lds
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dwordx4 v219, s[12:15], s44 offen lds
	ds_read_b128 v[80:83], v215 offset:34048
	ds_read_b128 v[164:167], v215 offset:34080
	ds_read_b128 v[170:173], v215 offset:34560
	ds_read_b128 v[178:181], v215 offset:34592
	ds_read_b128 v[182:185], v215 offset:34112
	ds_read_b128 v[186:189], v215 offset:34144
	ds_read_b128 v[194:197], v215 offset:34624
	ds_read_b128 v[198:201], v215 offset:34656
	ds_read_b128 v[202:205], v215 offset:42368
	ds_read_b128 v[222:225], v215 offset:42400
	ds_read_b128 v[226:229], v215 offset:42880
	ds_read_b128 v[230:233], v215 offset:42912
	ds_read_b128 v[234:237], v215 offset:42432
	ds_read_b128 v[238:241], v215 offset:42464
	ds_read_b128 v[242:245], v215 offset:42944
	ds_read_b128 v[246:249], v215 offset:42976
	s_waitcnt vmcnt(2) lgkmcnt(0)
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E9: compute drain C
; MMA0 plus softmax sum for the last tile family.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[64:79], v[80:83], v[112:115], 0
	v_exp_f32_e32 v111, v97
	v_exp_f32_e32 v156, v98
	v_exp_f32_e32 v157, v99
	v_mfma_f32_32x32x16_bf16 v[80:95], v[170:173], v[112:115], 0
	v_exp_f32_e32 v100, v100
	v_exp_f32_e32 v101, v101
	v_exp_f32_e32 v102, v102
	v_mfma_f32_32x32x16_bf16 v[64:79], v[164:167], v[116:119], v[64:79]
	v_exp_f32_e32 v103, v103
	v_exp_f32_e32 v112, v152
	v_exp_f32_e32 v113, v153
	v_mfma_f32_32x32x16_bf16 v[80:95], v[178:181], v[116:119], v[80:95]
	v_exp_f32_e32 v114, v154
	v_exp_f32_e32 v115, v155
	v_exp_f32_e32 v116, v160
	v_mfma_f32_32x32x16_bf16 v[64:79], v[182:185], v[120:123], v[64:79]
	v_exp_f32_e32 v117, v161
	v_exp_f32_e32 v118, v162
	v_exp_f32_e32 v119, v163
	v_mfma_f32_32x32x16_bf16 v[80:95], v[194:197], v[120:123], v[80:95]
	v_exp_f32_e32 v120, v110
	v_mfma_f32_32x32x16_bf16 v[64:79], v[186:189], v[124:127], v[64:79]
	v_add_f32_e32 v97, v96, v168
	v_add_f32_e32 v97, v97, v104
	v_add_f32_e32 v97, v97, v105
	v_add_f32_e32 v97, v97, v106
	v_add_f32_e32 v97, v97, v107
	v_mfma_f32_32x32x16_bf16 v[80:95], v[198:201], v[124:127], v[80:95]
	v_add_f32_e32 v97, v97, v144
	v_add_f32_e32 v97, v97, v145
	v_add_f32_e32 v97, v97, v146
	v_add_f32_e32 v97, v97, v147
	v_add_f32_e32 v97, v97, v148
	v_mfma_f32_32x32x16_bf16 v[64:79], v[202:205], v[128:131], v[64:79]
	v_add_f32_e32 v97, v97, v149
	v_add_f32_e32 v97, v97, v150
	v_add_f32_e32 v97, v97, v151
	v_add_f32_e32 v97, v97, v108
	v_add_f32_e32 v97, v97, v109
	v_mfma_f32_32x32x16_bf16 v[80:95], v[226:229], v[128:131], v[80:95]
	v_add_f32_e32 v97, v97, v111
	v_add_f32_e32 v97, v97, v156
	v_add_f32_e32 v97, v97, v157
	v_add_f32_e32 v97, v97, v100
	v_add_f32_e32 v97, v97, v101
	v_mfma_f32_32x32x16_bf16 v[64:79], v[222:225], v[132:135], v[64:79]
	v_add_f32_e32 v97, v97, v102
	v_add_f32_e32 v97, v97, v103
	v_add_f32_e32 v97, v97, v112
	v_add_f32_e32 v97, v97, v113
	v_add_f32_e32 v97, v97, v114
	v_mfma_f32_32x32x16_bf16 v[80:95], v[230:233], v[132:135], v[80:95]
	v_add_f32_e32 v97, v97, v115
	v_add_f32_e32 v97, v97, v116
	v_add_f32_e32 v97, v97, v117
	v_add_f32_e32 v97, v97, v118
	v_add_f32_e32 v97, v97, v119
	v_mfma_f32_32x32x16_bf16 v[64:79], v[234:237], v[136:139], v[64:79]
	v_add_f32_e32 v177, v97, v120
	v_mov_b32_e32 v178, v177
	s_nop 1
	v_permlane32_swap_b32_e64 v178, v177 bound_ctrl:1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v96, v96, v168
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v97, v104, v105
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v98, v106, v107
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[242:245], v[136:139], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v99, v144, v145
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v104, v111, v156
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v105, v157, v100
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v106, v101, v102
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v107, v103, v112
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[64:79], v[238:241], v[140:143], v[64:79]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v100, v146, v147
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v101, v148, v149
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v102, v150, v151
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v103, v108, v109
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v108, v113, v114
	;;#ASMEND
	v_mfma_f32_32x32x16_bf16 v[80:95], v[246:249], v[140:143], v[80:95]
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v109, v115, v116
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v110, v117, v118
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v111, v119, v120
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E10: memory drain D
; Read V transpose fragments and apply final causal masks.
; ==============================================================================
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b16 v[172:173], v220 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[174:175], v220 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[144:145], v220 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[146:147], v220 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v220 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v220 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v220 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v220 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[168:169], v220 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[170:171], v220 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[148:149], v220 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[150:151], v220 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v220 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v220 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v220 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v220 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[164:165], v220 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[166:167], v220 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[152:153], v220 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[154:155], v220 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v220 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v220 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v220 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v220 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[160:161], v220 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[162:163], v220 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[156:157], v220 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[158:159], v220 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v220 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v220 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v220 offset:9536

	;;#ASMEND
	s_cmp_ge_i32 s27, s8
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v220 offset:9664

	;;#ASMEND
	s_cbranch_scc1 .LBB0_43
	v_lshl_or_b32 v179, v208, 2, s9
	v_sub_u32_e32 v179, s3, v179
	v_add_u32_e32 v179, v179, v212
	v_subrev_u32_e32 v180, 32, v179
	v_mov_b32_e32 v181, 0xff800000
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v179, 0
	v_cmp_lt_i32_e64 s[12:13], v179, 1
	v_cndmask_b32_e64 v64, v64, v181, s[8:9]
	v_cndmask_b32_e64 v65, v65, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v179, 2
	v_cmp_lt_i32_e64 s[12:13], v179, 3
	v_cndmask_b32_e64 v66, v66, v181, s[8:9]
	v_cndmask_b32_e64 v67, v67, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v179, 8
	v_cmp_lt_i32_e64 s[12:13], v179, 9
	v_cndmask_b32_e64 v68, v68, v181, s[8:9]
	v_cndmask_b32_e64 v69, v69, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v179, 10
	v_cmp_lt_i32_e64 s[12:13], v179, 11
	v_cndmask_b32_e64 v70, v70, v181, s[8:9]
	v_cndmask_b32_e64 v71, v71, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v179, 16
	v_cmp_lt_i32_e64 s[12:13], v179, 17
	v_cndmask_b32_e64 v72, v72, v181, s[8:9]
	v_cndmask_b32_e64 v73, v73, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v179, 18
	v_cmp_lt_i32_e64 s[12:13], v179, 19
	v_cndmask_b32_e64 v74, v74, v181, s[8:9]
	v_cndmask_b32_e64 v75, v75, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v179, 24
	v_cmp_lt_i32_e64 s[12:13], v179, 25
	v_cndmask_b32_e64 v76, v76, v181, s[8:9]
	v_cndmask_b32_e64 v77, v77, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v179, 26
	v_cmp_lt_i32_e64 s[12:13], v179, 27
	v_cndmask_b32_e64 v78, v78, v181, s[8:9]
	v_cndmask_b32_e64 v79, v79, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v180, 0
	v_cmp_lt_i32_e64 s[12:13], v180, 1
	v_cndmask_b32_e64 v80, v80, v181, s[8:9]
	v_cndmask_b32_e64 v81, v81, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v180, 2
	v_cmp_lt_i32_e64 s[12:13], v180, 3
	v_cndmask_b32_e64 v82, v82, v181, s[8:9]
	v_cndmask_b32_e64 v83, v83, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v180, 8
	v_cmp_lt_i32_e64 s[12:13], v180, 9
	v_cndmask_b32_e64 v84, v84, v181, s[8:9]
	v_cndmask_b32_e64 v85, v85, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v180, 10
	v_cmp_lt_i32_e64 s[12:13], v180, 11
	v_cndmask_b32_e64 v86, v86, v181, s[8:9]
	v_cndmask_b32_e64 v87, v87, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v180, 16
	v_cmp_lt_i32_e64 s[12:13], v180, 17
	v_cndmask_b32_e64 v88, v88, v181, s[8:9]
	v_cndmask_b32_e64 v89, v89, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v180, 18
	v_cmp_lt_i32_e64 s[12:13], v180, 19
	v_cndmask_b32_e64 v90, v90, v181, s[8:9]
	v_cndmask_b32_e64 v91, v91, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v180, 24
	v_cmp_lt_i32_e64 s[12:13], v180, 25
	v_cndmask_b32_e64 v92, v92, v181, s[8:9]
	v_cndmask_b32_e64 v93, v93, v181, s[12:13]
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_cmp_lt_i32_e64 s[8:9], v180, 26
	v_cmp_lt_i32_e64 s[12:13], v180, 27
	v_cndmask_b32_e64 v94, v94, v181, s[8:9]
	v_cndmask_b32_e64 v95, v95, v181, s[12:13]
	;;#ASMEND
.LBB0_43:
	v_add_f32_e32 v179, v209, v214
	v_add_f32_e32 v179, v179, v213
	v_fmac_f32_e32 v211, v210, v179
	v_add_f32_e32 v179, v211, v216
	v_fmac_f32_e32 v177, v176, v179
	v_add_f32_e32 v176, v177, v178
	s_waitcnt vmcnt(0) lgkmcnt(0)
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E11: compute drain D
; Final score max/exp/sum update and P packing before normalization.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[0:15], v[172:175], v[96:99], v[0:15]
	v_max_f32_e32 v172, v64, v65
	v_max3_f32 v172, v172, v66, v67
	v_max3_f32 v172, v172, v68, v69
	v_max3_f32 v172, v172, v70, v71
	v_max3_f32 v172, v172, v72, v73
	v_max3_f32 v172, v172, v74, v75
	v_mfma_f32_32x32x16_bf16 v[16:31], v[168:171], v[96:99], v[16:31]
	v_max3_f32 v168, v172, v76, v77
	v_max3_f32 v168, v168, v78, v79
	v_max3_f32 v168, v168, v80, v81
	v_max3_f32 v168, v168, v82, v83
	v_max3_f32 v168, v168, v84, v85
	v_max3_f32 v168, v168, v86, v87
	v_mfma_f32_32x32x16_bf16 v[32:47], v[164:167], v[96:99], v[32:47]
	v_max3_f32 v164, v168, v88, v89
	v_max3_f32 v164, v164, v90, v91
	v_max3_f32 v164, v164, v92, v93
	v_max3_f32 v164, v164, v94, v95
	v_mov_b32_e32 v165, v164
	s_nop 1
	v_permlane32_swap_b32_e64 v164, v165 bound_ctrl:1
	v_max3_f32 v164, v192, v164, v165
	v_mfma_f32_32x32x16_bf16 v[48:63], v[160:163], v[96:99], v[48:63]
	v_sub_f32_e32 v96, v192, v164
	v_sub_f32_e32 v64, v64, v164
	v_sub_f32_e32 v65, v65, v164
	v_sub_f32_e32 v66, v66, v164
	v_sub_f32_e32 v67, v67, v164
	v_sub_f32_e32 v68, v68, v164
	v_mfma_f32_32x32x16_bf16 v[0:15], v[144:147], v[100:103], v[0:15]
	v_sub_f32_e32 v69, v69, v164
	v_sub_f32_e32 v70, v70, v164
	v_sub_f32_e32 v71, v71, v164
	v_sub_f32_e32 v72, v72, v164
	v_sub_f32_e32 v73, v73, v164
	v_sub_f32_e32 v74, v74, v164
	v_mfma_f32_32x32x16_bf16 v[16:31], v[148:151], v[100:103], v[16:31]
	v_sub_f32_e32 v75, v75, v164
	v_sub_f32_e32 v76, v76, v164
	v_sub_f32_e32 v77, v77, v164
	v_sub_f32_e32 v78, v78, v164
	v_sub_f32_e32 v79, v79, v164
	v_sub_f32_e32 v80, v80, v164
	v_mfma_f32_32x32x16_bf16 v[32:47], v[152:155], v[100:103], v[32:47]
	v_sub_f32_e32 v81, v81, v164
	v_sub_f32_e32 v82, v82, v164
	v_sub_f32_e32 v83, v83, v164
	v_sub_f32_e32 v84, v84, v164
	v_sub_f32_e32 v85, v85, v164
	v_sub_f32_e32 v86, v86, v164
	v_mfma_f32_32x32x16_bf16 v[48:63], v[156:159], v[100:103], v[48:63]
	v_sub_f32_e32 v87, v87, v164
	v_sub_f32_e32 v88, v88, v164
	v_sub_f32_e32 v89, v89, v164
	v_sub_f32_e32 v90, v90, v164
	v_sub_f32_e32 v91, v91, v164
	v_sub_f32_e32 v92, v92, v164
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[104:107], v[0:15]
	v_sub_f32_e32 v93, v93, v164
	v_sub_f32_e32 v94, v94, v164
	v_sub_f32_e32 v95, v95, v164
	v_mfma_f32_32x32x16_bf16 v[16:31], v[132:135], v[104:107], v[16:31]
	v_exp_f32_e32 v128, v96
	v_exp_f32_e32 v64, v64
	v_exp_f32_e32 v65, v65
	v_mfma_f32_32x32x16_bf16 v[32:47], v[136:139], v[104:107], v[32:47]
	v_exp_f32_e32 v66, v66
	v_exp_f32_e32 v67, v67
	v_exp_f32_e32 v68, v68
	v_mfma_f32_32x32x16_bf16 v[48:63], v[140:143], v[104:107], v[48:63]
	v_exp_f32_e32 v69, v69
	v_exp_f32_e32 v70, v70
	v_exp_f32_e32 v71, v71
	v_mfma_f32_32x32x16_bf16 v[0:15], v[116:119], v[108:111], v[0:15]
	v_exp_f32_e32 v96, v72
	v_exp_f32_e32 v97, v73
	v_exp_f32_e32 v98, v74
	v_mfma_f32_32x32x16_bf16 v[16:31], v[120:123], v[108:111], v[16:31]
	v_exp_f32_e32 v99, v75
	v_exp_f32_e32 v76, v76
	v_exp_f32_e32 v77, v77
	v_mfma_f32_32x32x16_bf16 v[32:47], v[124:127], v[108:111], v[32:47]
	v_exp_f32_e32 v78, v78
	v_exp_f32_e32 v79, v79
	v_mfma_f32_32x32x16_bf16 v[48:63], v[112:115], v[108:111], v[48:63]
	v_exp_f32_e32 v72, v80
	v_exp_f32_e32 v73, v81
	v_exp_f32_e32 v74, v82
	v_exp_f32_e32 v75, v83
	v_exp_f32_e32 v80, v84
	v_exp_f32_e32 v81, v85
	v_exp_f32_e32 v82, v86
	v_exp_f32_e32 v83, v87
	v_exp_f32_e32 v84, v88
	v_exp_f32_e32 v85, v89
	v_exp_f32_e32 v86, v90
	v_exp_f32_e32 v87, v91
	v_exp_f32_e32 v88, v92
	v_exp_f32_e32 v89, v93
	v_exp_f32_e32 v90, v94
	v_exp_f32_e32 v91, v95
	v_add_f32_e32 v92, v64, v65
	v_add_f32_e32 v92, v92, v66
	v_add_f32_e32 v92, v92, v67
	v_add_f32_e32 v92, v92, v68
	v_add_f32_e32 v92, v92, v69
	v_add_f32_e32 v92, v92, v70
	v_add_f32_e32 v92, v92, v71
	v_add_f32_e32 v92, v92, v96
	v_add_f32_e32 v92, v92, v97
	v_add_f32_e32 v92, v92, v98
	v_add_f32_e32 v92, v92, v99
	v_add_f32_e32 v92, v92, v76
	v_add_f32_e32 v92, v92, v77
	v_add_f32_e32 v92, v92, v78
	v_add_f32_e32 v92, v92, v79
	v_add_f32_e32 v92, v92, v72
	v_add_f32_e32 v92, v92, v73
	v_add_f32_e32 v92, v92, v74
	v_add_f32_e32 v92, v92, v75
	v_add_f32_e32 v92, v92, v80
	v_add_f32_e32 v92, v92, v81
	v_add_f32_e32 v92, v92, v82
	v_add_f32_e32 v92, v92, v83
	v_add_f32_e32 v92, v92, v84
	v_add_f32_e32 v92, v92, v85
	v_add_f32_e32 v92, v92, v86
	v_add_f32_e32 v92, v92, v87
	v_add_f32_e32 v92, v92, v88
	v_add_f32_e32 v92, v92, v89
	v_add_f32_e32 v92, v92, v90
	v_add_f32_e32 v92, v92, v91
	v_mov_b32_e32 v93, v92
	s_nop 1
	v_permlane32_swap_b32_e64 v93, v92 bound_ctrl:1
	v_fmac_f32_e32 v92, v128, v176
	v_add_f32_e32 v144, v92, v93
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v64, v64, v65
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v65, v66, v67
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v66, v68, v69
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v67, v70, v71
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v72, v72, v73
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v73, v74, v75
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v74, v80, v81
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v75, v82, v83
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v68, v96, v97
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v69, v98, v99
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v70, v76, v77
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v71, v78, v79
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v76, v84, v85
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v77, v86, v87
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v78, v88, v89
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v79, v90, v91
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	;;#ASMEND
	v_pk_mul_f32 v[110:111], v[128:129], v[14:15] op_sel_hi:[0,1]
	v_pk_mul_f32 v[108:109], v[128:129], v[12:13] op_sel_hi:[0,1]
	v_pk_mul_f32 v[106:107], v[128:129], v[10:11] op_sel_hi:[0,1]
	v_pk_mul_f32 v[104:105], v[128:129], v[8:9] op_sel_hi:[0,1]
	v_pk_mul_f32 v[102:103], v[128:129], v[6:7] op_sel_hi:[0,1]
	v_pk_mul_f32 v[100:101], v[128:129], v[4:5] op_sel_hi:[0,1]
	v_pk_mul_f32 v[98:99], v[128:129], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[96:97], v[128:129], v[0:1] op_sel_hi:[0,1]
	v_pk_mul_f32 v[94:95], v[128:129], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[92:93], v[128:129], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[90:91], v[128:129], v[26:27] op_sel_hi:[0,1]
	v_pk_mul_f32 v[88:89], v[128:129], v[24:25] op_sel_hi:[0,1]
	v_pk_mul_f32 v[86:87], v[128:129], v[22:23] op_sel_hi:[0,1]
	v_pk_mul_f32 v[84:85], v[128:129], v[20:21] op_sel_hi:[0,1]
	v_pk_mul_f32 v[82:83], v[128:129], v[18:19] op_sel_hi:[0,1]
	v_pk_mul_f32 v[80:81], v[128:129], v[16:17] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[128:129], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[128:129], v[44:45] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[128:129], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[24:25], v[128:129], v[40:41] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[128:129], v[38:39] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[128:129], v[36:37] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[128:129], v[34:35] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[128:129], v[32:33] op_sel_hi:[0,1]
	v_pk_mul_f32 v[14:15], v[128:129], v[62:63] op_sel_hi:[0,1]
	v_pk_mul_f32 v[12:13], v[128:129], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[10:11], v[128:129], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[8:9], v[128:129], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[128:129], v[54:55] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[128:129], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[128:129], v[50:51] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[128:129], v[48:49] op_sel_hi:[0,1]
	;;#ASMSTART
	;;#ASMEND
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E12: memory drain E
; Read final V transpose fragments feeding last GEMM1 operations.
; ==============================================================================
	s_barrier
	;;#ASMSTART
	ds_read_b64_tr_b16 v[32:33], v221 offset:0

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[34:35], v221 offset:128

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[36:37], v221 offset:256

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[38:39], v221 offset:384

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[40:41], v221 offset:512

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[42:43], v221 offset:640

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[44:45], v221 offset:768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[46:47], v221 offset:896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[48:49], v221 offset:64

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[50:51], v221 offset:192

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[52:53], v221 offset:320

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[54:55], v221 offset:448

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[56:57], v221 offset:576

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[58:59], v221 offset:704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[60:61], v221 offset:832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[62:63], v221 offset:960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[112:113], v221 offset:8704

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[114:115], v221 offset:8832

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[116:117], v221 offset:8960

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[118:119], v221 offset:9088

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[120:121], v221 offset:9216

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[122:123], v221 offset:9344

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[124:125], v221 offset:9472

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[126:127], v221 offset:9600

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[128:129], v221 offset:8768

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[130:131], v221 offset:8896

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[132:133], v221 offset:9024

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[134:135], v221 offset:9152

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[136:137], v221 offset:9280

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[138:139], v221 offset:9408

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[140:141], v221 offset:9536

	;;#ASMEND
	;;#ASMSTART
	ds_read_b64_tr_b16 v[142:143], v221 offset:9664

	;;#ASMEND
	s_waitcnt lgkmcnt(0)
; ==============================================================================
; PASS 1 EPILOGUE CLUSTER E13: final GEMM1 + normalization
; Compute last O accumulators, reciprocal l_row, scale O, convert to bf16 packs.
; ==============================================================================
	s_barrier
	v_mfma_f32_32x32x16_bf16 v[96:111], v[32:35], v[64:67], v[96:111]
	v_rcp_f32_e32 v32, v144
	v_cmp_lt_f32_e32 vcc, 0, v144
	s_nop 1
	v_cndmask_b32_e32 v32, 0, v32, vcc
	v_mfma_f32_32x32x16_bf16 v[80:95], v[48:51], v[64:67], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[112:115], v[64:67], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[128:131], v[64:67], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[96:111], v[36:39], v[68:71], v[96:111]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[52:55], v[68:71], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[116:119], v[68:71], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[132:135], v[68:71], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[96:111], v[40:43], v[72:75], v[96:111]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[56:59], v[72:75], v[80:95]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[120:123], v[72:75], v[16:31]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[136:139], v[72:75], v[0:15]
	v_mfma_f32_32x32x16_bf16 v[96:111], v[44:47], v[76:79], v[96:111]
	s_nop 11
	v_pk_mul_f32 v[34:35], v[110:111], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[36:37], v[108:109], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[38:39], v[106:107], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[40:41], v[104:105], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[42:43], v[102:103], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[44:45], v[100:101], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[46:47], v[98:99], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[48:49], v[96:97], v[32:33] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_bf16 v[80:95], v[60:63], v[76:79], v[80:95]
	s_nop 11
	v_pk_mul_f32 v[50:51], v[94:95], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[52:53], v[92:93], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[54:55], v[90:91], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[56:57], v[88:89], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[58:59], v[86:87], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[84:85], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[62:63], v[82:83], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[64:65], v[80:81], v[32:33] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_bf16 v[16:31], v[124:127], v[76:79], v[16:31]
	s_nop 11
	v_pk_mul_f32 v[30:31], v[30:31], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[28:29], v[28:29], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[26:27], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[24:25], v[24:25], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[22:23], v[22:23], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[20:21], v[20:21], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[18:19], v[18:19], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[16:17], v[16:17], v[32:33] op_sel_hi:[1,0]
	v_mfma_f32_32x32x16_bf16 v[0:15], v[140:143], v[76:79], v[0:15]
	s_nop 11
	v_pk_mul_f32 v[14:15], v[14:15], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[12:13], v[12:13], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[10:11], v[10:11], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[8:9], v[8:9], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[6:7], v[6:7], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[4:5], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[66:67], v[2:3], v[32:33] op_sel_hi:[1,0]
	v_pk_mul_f32 v[32:33], v[0:1], v[32:33] op_sel_hi:[1,0]
	;;#ASMSTART
	s_cmp_eq_u32 s33, 0
; ==============================================================================
; PASS 1 FINALIZE/STORE
; Permute packed bf16 lanes, guard OOB rows, store O for the mirror q-block,
; then exit the kernel.
; ==============================================================================
	s_cbranch_scc0 1f
	s_barrier
	1:
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v48, v49
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v46, v47
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v44, v45
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v42, v43
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v42, v43
	v_mov_b32_e32 v42, v1
	v_mov_b32_e32 v44, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v42, v44
	v_mov_b32_e32 v42, v2
	v_mov_b32_e32 v45, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v42, v45
	v_mov_b32_e32 v45, v3
	v_mov_b32_e32 v46, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v45, v46
	v_cndmask_b32_e64 v0, v42, v0, s[0:1]
	v_cndmask_b32_e64 v1, v45, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v43, s[0:1]
	v_cndmask_b32_e64 v3, v3, v44, s[0:1]
	v_add_u32_e32 v42, s36, v212
	v_mul_lo_u32 v42, v42, s10
	v_add_u32_e32 v42, s2, v42
	v_add_lshl_u32 v42, v217, v42, 1
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v40, v41
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v38, v39
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v36, v37
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v34, v35
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v35
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v36, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v36
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v37, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v37
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v38, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v37, v38
	v_cndmask_b32_e64 v0, v34, v0, s[0:1]
	v_cndmask_b32_e64 v1, v37, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v35, s[0:1]
	v_cndmask_b32_e64 v3, v3, v36, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:32
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v64, v65
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v62, v63
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v60, v61
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v58, v59
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v35
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v36, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v36
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v37, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v37
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v38, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v37, v38
	v_cndmask_b32_e64 v0, v34, v0, s[0:1]
	v_cndmask_b32_e64 v1, v37, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v35, s[0:1]
	v_cndmask_b32_e64 v3, v3, v36, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:64
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v56, v57
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v54, v55
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v52, v53
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v50, v51
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v35
	v_mov_b32_e32 v34, v1
	v_mov_b32_e32 v36, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v36
	v_mov_b32_e32 v34, v2
	v_mov_b32_e32 v37, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v34, v37
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v38, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v37, v38
	v_cndmask_b32_e64 v0, v34, v0, s[0:1]
	v_cndmask_b32_e64 v1, v37, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v35, s[0:1]
	v_cndmask_b32_e64 v3, v3, v36, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:96
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v16, v17
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v18, v19
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v20, v21
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v22, v23
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v17
	v_mov_b32_e32 v16, v1
	v_mov_b32_e32 v18, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v18
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v19, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v19
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v20, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v19, v20
	v_cndmask_b32_e64 v0, v16, v0, s[0:1]
	v_cndmask_b32_e64 v1, v19, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v17, s[0:1]
	v_cndmask_b32_e64 v3, v3, v18, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:128
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v24, v25
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v26, v27
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v28, v29
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v30, v31
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v17
	v_mov_b32_e32 v16, v1
	v_mov_b32_e32 v18, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v18
	v_mov_b32_e32 v16, v2
	v_mov_b32_e32 v19, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v16, v19
	v_mov_b32_e32 v19, v3
	v_mov_b32_e32 v20, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v19, v20
	v_cndmask_b32_e64 v0, v16, v0, s[0:1]
	v_cndmask_b32_e64 v1, v19, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v17, s[0:1]
	v_cndmask_b32_e64 v3, v3, v18, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:160
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v32, v33
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v66, v67
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v4, v5
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v6, v7
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v5
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v6, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v6
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v7, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v7
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v16, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v7, v16
	v_cndmask_b32_e64 v0, v4, v0, s[0:1]
	v_cndmask_b32_e64 v1, v7, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v5, s[0:1]
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:192
	s_nop 1
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v0, v8, v9
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v1, v10, v11
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v2, v12, v13
	;;#ASMEND
	;;#ASMSTART
	v_cvt_pk_bf16_f32 v3, v14, v15
	;;#ASMEND
	s_nop 0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v5
	v_mov_b32_e32 v4, v1
	v_mov_b32_e32 v6, v1
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v6
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v7, v2
	s_nop 1
	v_permlane32_swap_b32_e32 v4, v7
	v_mov_b32_e32 v7, v3
	v_mov_b32_e32 v8, v3
	s_nop 1
	v_permlane32_swap_b32_e32 v7, v8
	v_cndmask_b32_e64 v0, v4, v0, s[0:1]
	v_cndmask_b32_e64 v1, v7, v1, s[0:1]
	v_cndmask_b32_e64 v2, v2, v5, s[0:1]
	v_cndmask_b32_e64 v3, v3, v6, s[0:1]
	buffer_store_dwordx4 v[0:3], v42, s[4:7], 0 offen offset:224
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel flash_attn_dualwave_swp_gfx950_kernel
		.amdhsa_group_segment_fixed_size 68096
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 128
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	flash_attn_dualwave_swp_gfx950_kernel, .Lfunc_end0-flash_attn_dualwave_swp_gfx950_kernel

	.set flash_attn_dualwave_swp_gfx950_kernel.num_vgpr, 256
	.set flash_attn_dualwave_swp_gfx950_kernel.num_agpr, 0
	.set flash_attn_dualwave_swp_gfx950_kernel.numbered_sgpr, 62
	.set flash_attn_dualwave_swp_gfx950_kernel.num_named_barrier, 0
	.set flash_attn_dualwave_swp_gfx950_kernel.private_seg_size, 0
	.set flash_attn_dualwave_swp_gfx950_kernel.uses_vcc, 1
	.set flash_attn_dualwave_swp_gfx950_kernel.uses_flat_scratch, 0
	.set flash_attn_dualwave_swp_gfx950_kernel.has_dyn_sized_stack, 0
	.set flash_attn_dualwave_swp_gfx950_kernel.has_recursion, 0
	.set flash_attn_dualwave_swp_gfx950_kernel.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .offset:         108
        .size:           4
        .value_kind:     by_value
      - .offset:         112
        .size:           4
        .value_kind:     by_value
      - .offset:         116
        .size:           4
        .value_kind:     by_value
      - .offset:         120
        .size:           4
        .value_kind:     by_value
      - .offset:         124
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 68096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 128
    .max_flat_workgroup_size: 512
    .name:           flash_attn_dualwave_swp_gfx950_kernel
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     68
    .sgpr_spill_count: 0
    .symbol:         flash_attn_dualwave_swp_gfx950_kernel.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
