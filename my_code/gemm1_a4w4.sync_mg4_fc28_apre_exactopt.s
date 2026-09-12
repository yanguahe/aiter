	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p
	.p2align	8
	.type	a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p,@function
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b96 s[24:26], s[0:1], 0xa4 nv
	s_and_b32 s6, ttmp6, 15
	v_readfirstlane_b32 s33, v0
	s_lshl2_add_u32 s27, ttmp9, s6
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s25, 0xff
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s5, s4, 31
	s_lshr_b32 s5, s5, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s5, s4, s5
	s_and_b32 s6, s5, 0xffffff00
	s_ashr_i32 s5, s5, 8
	s_cmp_lg_u32 s4, s6
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s4, 0
	s_cselect_b32 s4, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s4, s4, s6
	s_sub_co_ci_u32 s4, s5, 0
	s_add_co_i32 s5, s24, 0xff
	s_ashr_i32 s6, s5, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s6, s6, 24
	s_add_co_i32 s6, s5, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s6, 0xffffff00
	s_ashr_i32 s6, s6, 8
	s_cmp_lg_u32 s5, s7
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s5, 0
	s_cselect_b32 s5, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s5, s5, s7
	s_sub_co_ci_u32 s5, s6, 0
	s_ashr_i32 s6, s27, 31
	s_lshr_b32 s6, s6, 30
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s27, s6
	s_and_b32 s7, s6, -4
	s_ashr_i32 s8, s6, 2
	s_cmp_lg_u32 s27, s7
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s27, 0
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_and_b32 s7, s7, s6
	s_sub_co_ci_u32 s30, s8, 0
	s_ashr_i32 s6, s4, 31
	s_lshr_b32 s6, s6, 30
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s4, s6
	s_and_b32 s9, s6, -4
	s_ashr_i32 s6, s6, 2
	s_cmp_lg_u32 s4, s9
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s4, 0
	s_cselect_b32 s4, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_and_b32 s4, s4, s9
	s_sub_co_ci_u32 s4, s6, 0
	s_add_co_i32 s9, s5, 3
	s_bfe_u32 s10, ttmp6, 0x40004
	s_ashr_i32 s6, s9, 31
	s_lshr_b32 s6, s6, 30
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s11, s9, s6
	s_lshl2_add_u32 s6, ttmp7, s10
	s_and_b32 s10, s11, -4
	s_ashr_i32 s11, s11, 2
	s_cmp_lg_u32 s9, s10
	s_cselect_b32 s9, -1, 0
	s_cmp_lt_i32 s5, -3
	s_cselect_b32 s5, -1, 0
	s_lshl_b32 s10, s4, 2
	s_abs_i32 s15, s30
	s_abs_i32 s12, s10
	s_and_b32 s5, s5, s9
	s_cvt_f32_u32 s13, s12
	s_sub_co_i32 s14, 0, s12
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s13
	v_nop
	v_readfirstlane_b32 s13, v1
	s_mul_f32 s13, s13, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s13, s13
	s_mul_i32 s14, s14, s13
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s14, s13, s14
	s_add_co_i32 s13, s13, s14
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_hi_u32 s9, s15, s13
	s_xor_b32 s13, s30, s10
	s_mul_i32 s14, s9, s12
	s_ashr_i32 s13, s13, 31
	s_sub_co_i32 s14, s15, s14
	s_add_co_i32 s15, s9, 1
	s_sub_co_i32 s16, s14, s12
	s_cmp_ge_u32 s14, s12
	s_cselect_b32 s9, s15, s9
	s_cselect_b32 s14, s16, s14
	s_add_co_i32 s15, s9, 1
	s_cmp_ge_u32 s14, s12
	s_cselect_b32 s9, s15, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s9, s9, s13
	s_sub_co_i32 s12, s9, s13
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s12, s12, s10
	s_cmp_lg_u32 s30, s12
	s_cselect_b32 s12, -1, 0
	s_xor_b32 s4, s4, s30
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s4, 0
	s_cselect_b32 s4, -1, 0
	s_and_b32 s4, s4, s12
	s_sub_co_ci_u32 s4, s9, s13
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b32 s9, s4, 2
	s_mul_i32 s4, s4, s10
	s_cmp_lg_u32 s7, 0
	s_sub_co_ci_u32 s7, s8, s4
	s_cmp_lg_u32 s5, 0
	s_sub_co_ci_u32 s8, s11, s9
	s_abs_i32 s12, s7
	s_min_i32 s10, s8, 4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_abs_i32 s11, s10
	s_xor_b32 s14, s7, s10
	s_cvt_f32_u32 s4, s11
	s_sub_co_i32 s5, 0, s11
	s_ashr_i32 s14, s14, 31
	v_rcp_iflag_f32_e32 v1, s4
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_3)
	v_readfirstlane_b32 s4, v1
	s_mul_f32 s4, s4, 0x4f7ffffe
	s_cvt_u32_f32 s4, s4
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s5, s5, s4
	s_mul_hi_u32 s5, s4, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s13, s4, s5
	s_load_b64 s[4:5], s[0:1], 0x70 nv
	s_mul_hi_u32 s13, s12, s13
	s_mul_i32 s15, s13, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s12, s12, s15
	s_add_co_i32 s15, s13, 1
	s_sub_co_i32 s16, s12, s11
	s_cmp_ge_u32 s12, s11
	s_cselect_b32 s13, s15, s13
	s_cselect_b32 s12, s16, s12
	s_add_co_i32 s15, s13, 1
	s_cmp_ge_u32 s12, s11
	s_cselect_b32 s11, s15, s13
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s11, s11, s14
	s_sub_co_i32 s12, s11, s14
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s12, s10, s12
	s_cmp_lg_u32 s7, s12
	s_wait_kmcnt 0x0
	s_load_b32 s12, s[4:5], 0xc0
	s_cselect_b32 s13, -1, 0
	s_xor_b32 s8, s7, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s8, 0
	s_cselect_b32 s8, -1, 0
	s_and_b32 s8, s8, s13
	s_sub_co_ci_u32 s31, s11, s14
	s_add_co_i32 s7, s7, s9
	s_mul_i32 s8, s31, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s7, s7, s8
	s_lshl_b32 s38, s7, 2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s7, s38, s6
	s_lshl_b32 s28, s7, 8
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s12, s28
	s_cselect_b32 s7, 24, 0x48
	s_cselect_b32 s9, 0, 49
	s_load_b32 s8, s[4:5], s7 offset:0x0 scale_offset
	s_cselect_b32 s10, 48, 0x60
	s_or_b32 s11, s7, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s8, s28
	s_cselect_b32 s8, s9, s11
	s_cselect_b32 s7, s7, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s8, s7
	s_lshr_b32 s9, s9, 1
	s_load_b32 s10, s[4:5], s9 offset:0x0 scale_offset
	s_or_b32 s11, s9, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s10, s28
	s_cselect_b32 s8, s8, s11
	s_cselect_b32 s7, s9, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s8, s7
	s_lshr_b32 s12, s9, 1
	s_mov_b32 s9, 0
	s_load_b32 s10, s[4:5], s12 offset:0x0 scale_offset
	s_add_co_i32 s13, s12, 1
	s_mov_b32 s11, s9
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s10, s28
	s_cselect_b32 s10, s8, s13
	s_cselect_b32 s8, s12, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[12:13], s[10:11], s[8:9]
	s_lshr_b64 s[12:13], s[12:13], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[14:15], s[12:13], 2
	s_add_co_i32 s9, s12, 1
	s_add_nc_u64 s[14:15], s[4:5], s[14:15]
	s_load_b32 s7, s[14:15], 0x0
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s7, s28
	s_cselect_b32 s7, s10, s9
	s_cselect_b32 s8, s12, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s7, s8
	s_lshr_b32 s9, s9, 1
	s_load_b32 s10, s[4:5], s9 offset:0x0 scale_offset
	s_add_co_i32 s11, s9, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s10, s28
	s_cselect_b32 s7, s7, s11
	s_cselect_b32 s8, s9, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s7, s8
	s_lshr_b32 s9, s9, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s10, s9, 0x5f
	s_add_co_i32 s11, s9, 1
	s_load_b32 s10, s[4:5], s10 offset:0x0 scale_offset
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s10, s28
	s_cselect_b32 s7, s7, s11
	s_cselect_b32 s8, s9, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s8, s7, s8
	s_lshr_b32 s8, s8, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s9, s8, 0x5f
	s_add_co_i32 s8, s8, 1
	s_load_b32 s9, s[4:5], s9 offset:0x0 scale_offset
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s9, s28
	s_cselect_b32 s36, s7, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_u32 s36, 0x60
	s_cselect_b32 s7, -1, 0
	s_cmp_gt_u32 s36, 0x5f
	s_cbranch_scc1 .LBB0_41
	s_load_b32 s8, s[4:5], 0x17c
	s_lshr_b32 s49, s33, 5
	s_and_b32 s7, s7, exec_lo
	s_cselect_b32 s7, s36, 0x5f
	v_sub_nc_u32_e64 v1, s36, 1 clamp
	s_wait_kmcnt 0x0
	s_addk_co_i32 s8, 0xff
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s9, s8, 31
	s_lshr_b32 s9, s9, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s8, s9
	s_and_b32 s10, s9, 0xffffff00
	s_ashr_i32 s9, s9, 8
	s_cmp_lg_u32 s8, s10
	s_cselect_b32 s10, -1, 0
	s_cmp_lt_i32 s8, 0
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s8, s8, s10
	s_sub_co_ci_u32 s8, s9, 0
	s_add_co_i32 s9, s38, 4
	v_readfirstlane_b32 s10, v1
	s_cmp_le_i32 s9, s8
	s_cselect_b32 s48, -1, 0
	s_cmp_lt_u32 s36, 0x61
	s_cselect_b32 s10, s10, 0x5f
	s_clause 0x1
	s_load_b32 s34, s[4:5], s7 offset:0x0 scale_offset
	s_load_b32 s35, s[4:5], s10 offset:0x0 scale_offset
	s_cmp_gt_i32 s9, s8
	s_cbranch_scc1 .LBB0_5
	s_barrier_signal -1
	s_cmp_lg_u32 s49, 0
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_4
	s_barrier_signal -3
.LBB0_4:
	s_barrier_wait -3
.LBB0_5:
	s_wait_kmcnt 0x0
	s_sub_co_i32 s24, s34, s28
	s_lshl_b32 s4, s6, 2
	s_add_co_i32 s7, s24, 15
	s_ashr_i32 s29, s28, 31
	s_ashr_i32 s5, s7, 31
	s_lshl_b32 s51, 15, s4
	s_lshr_b32 s5, s5, 28
	s_clause 0x1
	s_load_b128 s[12:15], s[0:1], 0x28 nv
	s_load_b64 s[20:21], s[0:1], 0x38 nv
	s_add_co_i32 s6, s7, s5
	s_lshr_b64 s[4:5], s[28:29], 4
	s_and_b32 s8, s6, -16
	s_ashr_i32 s6, s6, 4
	s_cmp_lg_u32 s7, s8
	s_mul_u64 s[4:5], s[4:5], 0xe000
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s7, 0
	s_mov_b32 s39, 4
	s_cselect_b32 s7, -1, 0
	s_mov_b32 s37, 0
	s_and_b32 s7, s7, s8
	s_sub_co_ci_u32 s52, s6, 0
	s_add_co_i32 s6, s24, 31
	s_ashr_i64 s[22:23], s[28:29], 5
	s_ashr_i32 s7, s6, 31
	s_mul_u64 s[46:47], s[22:23], 0x700
	s_lshr_b32 s7, s7, 27
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[40:41], s[12:13], s[4:5]
	s_add_co_i32 s7, s6, s7
	s_mul_i32 s42, s49, 0x70000
	s_and_b32 s8, s7, 0xffffffe0
	s_ashr_i32 s7, s7, 5
	s_cmp_lg_u32 s6, s8
	s_mul_i32 s44, s49, 0x1c00
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s6, 0
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s4, s6, s8
	s_sub_co_ci_u32 s53, s7, 0
	s_cmp_lt_u32 s33, 64
	s_cselect_b32 s54, -1, 0
	s_cmp_gt_u32 s33, 63
	s_cbranch_scc1 .LBB0_7
	s_lshl_b32 s4, s49, 3
	s_mov_b32 s43, s37
	s_sub_co_i32 s4, s52, s4
	s_add_nc_u64 s[18:19], s[40:41], s[42:43]
	s_max_i32 s5, s4, 0
	s_and_b32 s12, s51, 0xffff
	s_lshl_b32 s6, s5, 16
	s_lshr_b32 s5, s5, 16
	s_lshl_b32 s17, s49, 14
	s_bitset1_b32 s19, 31
	s_mov_b32 s16, 1
	s_or_b32 s4, s12, 0x200000
	s_addk_co_i32 s6, 0x7fff
	s_or_b32 s7, s5, 0x8000000
	s_mov_b32 s9, 0xe000
	s_mov_b32 s8, 8
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s37
	s_mov_b32 s11, s37
	s_mov_b32 s45, s37
	tensor_load_to_lds s[16:19], s[4:11]
	s_lshl_b64 s[6:7], s[46:47], 2
	s_lshl_b64 s[8:9], s[44:45], 2
	s_add_nc_u64 s[6:7], s[20:21], s[6:7]
	s_lshl_b32 s4, s49, 2
	s_add_nc_u64 s[18:19], s[6:7], s[8:9]
	s_lshl_b32 s6, s49, 10
	s_sub_co_i32 s4, s53, s4
	s_add_co_i32 s17, s6, 0x10000
	s_max_i32 s6, s4, 0
	s_bitset1_b32 s19, 31
	s_lshl_b32 s7, s6, 16
	s_lshr_b32 s8, s6, 16
	s_or_b32 s4, s12, 0x220000
	s_or_b32 s6, s7, 0x7fff
	s_or_b32 s7, s8, 0x400000
	s_movk_i32 s9, 0x700
	s_mov_b32 s8, s39
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[16:19], s[4:11]
.LBB0_7:
	s_lshl_b32 s4, s30, 2
	s_ashr_i32 s7, s25, 31
	s_sub_co_i32 s13, s27, s4
	s_lshr_b32 s4, s7, 28
	s_lshl_b32 s5, s31, 10
	s_add_co_i32 s4, s25, s4
	s_lshl_b32 s6, s13, 8
	s_ashr_i32 s4, s4, 4
	s_add_co_i32 s30, s5, s6
	s_ashr_i32 s5, s4, 31
	s_mov_b32 s6, s25
	s_lshl_b64 s[8:9], s[4:5], 4
	s_ashr_i32 s31, s30, 31
	s_cmp_lg_u64 s[8:9], s[6:7]
	s_add_nc_u64 s[6:7], s[6:7], 31
	s_cselect_b32 s12, -1, 0
	s_cmp_lt_i32 s25, 0
	s_mov_b32 s9, s37
	s_cselect_b32 s27, -1, 0
	s_lshr_b32 s8, s7, 27
	s_mov_b64 s[10:11], 0xffffffffffffffe0
	s_add_nc_u64 s[8:9], s[6:7], s[8:9]
	s_and_b32 s12, s27, s12
	s_and_b64 s[10:11], s[8:9], s[10:11]
	s_ashr_i64 s[16:17], s[8:9], 5
	s_cmp_lg_u64 s[6:7], s[10:11]
	v_cndmask_b32_e64 v1, 0, 1, s12
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s25, 0xffffffe1
	s_mov_b32 s12, 1
	s_cselect_b32 s7, -1, 0
	s_add_co_i32 s8, s35, 0xff
	s_and_b32 s6, s7, s6
	s_ashr_i32 s9, s8, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s9, s9, 24
	s_add_co_i32 s9, s8, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s9, 0xffffff00
	s_ashr_i32 s9, s9, 8
	s_cmp_lg_u32 s8, s7
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s8, 0
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s8, s7
	s_sub_co_ci_u32 s7, s9, 0
	s_cmp_lg_u32 s36, 0
	s_cselect_b32 s9, s7, 0
	s_add_co_i32 s7, s34, 0xff
	s_load_b64 s[34:35], s[0:1], 0x60 nv
	s_ashr_i32 s8, s7, 31
	s_wait_xcnt 0x0
	s_mov_b32 s1, s37
	s_lshr_b32 s8, s8, 24
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s8, s7, s8
	s_and_b32 s10, s8, 0xffffff00
	s_ashr_i32 s8, s8, 8
	s_cmp_lg_u32 s7, s10
	s_cselect_b32 s10, -1, 0
	s_cmp_lt_i32 s7, 0
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s7, s10
	s_sub_co_ci_u32 s10, s8, 0
	s_cmp_ge_i32 s38, s9
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s38, s10
	s_cselect_b32 s8, -1, 0
	s_or_b32 s11, s38, 1
	s_and_b32 s18, s8, s7
	s_cmp_ge_i32 s11, s9
	v_cndmask_b32_e64 v2, 0, 1, s18
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s11, s10
	s_cselect_b32 s8, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s0, s8, s7
	s_mov_b32 s7, s37
	s_and_b32 s0, s0, exec_lo
	s_cselect_b32 s11, 16, 0
	s_or_b32 s0, s38, 2
	s_mov_b32 s8, 8
	s_cmp_ge_i32 s0, s9
	s_cselect_b32 s19, -1, 0
	s_cmp_lt_i32 s0, s10
	s_cselect_b32 s0, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s19, s0, s19
	v_readfirstlane_b32 s0, v1
	s_and_b32 s19, s19, exec_lo
	s_cselect_b32 s19, 0x100, 0
	s_or_b32 s38, s38, 3
	v_cndmask_b32_e64 v1, 0, 1, s6
	s_cmp_ge_i32 s38, s9
	s_sub_nc_u64 s[0:1], s[4:5], s[0:1]
	s_cselect_b32 s6, -1, 0
	s_cmp_lt_i32 s38, s10
	v_readfirstlane_b32 s5, v2
	s_cselect_b32 s4, -1, 0
	s_mul_u64 s[0:1], s[0:1], s[36:37]
	s_and_b32 s4, s4, s6
	v_readfirstlane_b32 s6, v1
	s_and_b32 s4, s4, exec_lo
	s_cselect_b32 s9, 0x1000, 0
	s_or_b32 s4, s11, s5
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 s10, s4, s19
	s_sub_nc_u64 s[4:5], s[16:17], s[6:7]
	s_or_b32 s10, s10, s9
	s_mul_u64 s[4:5], s[4:5], s[36:37]
	s_lshl_b32 s6, s10, s13
	s_and_b32 s7, s48, exec_lo
	s_cselect_b32 s50, s6, 0
	s_lshr_b64 s[6:7], s[30:31], 4
	s_ashr_i64 s[10:11], s[30:31], 5
	s_and_b32 s9, s33, 0xffffffc0
	s_add_nc_u64 s[0:1], s[0:1], s[6:7]
	s_add_nc_u64 s[38:39], s[4:5], s[10:11]
	s_cmp_eq_u32 s9, 64
	s_mul_u64 s[0:1], s[0:1], 0xe000
	s_mul_u64 s[16:17], s[38:39], 0x700
	s_cselect_b32 s18, -1, 0
	s_cmp_lg_u32 s9, 64
	s_add_nc_u64 s[36:37], s[14:15], s[0:1]
	s_cbranch_scc1 .LBB0_9
	s_add_co_i32 s19, s49, -2
	s_mov_b32 s11, 0
	s_mul_i32 s10, s19, 0x70000
	s_lshl_b32 s0, s19, 14
	s_add_nc_u64 s[14:15], s[36:37], s[10:11]
	s_and_b32 s43, s50, 0xffff
	s_add_co_i32 s13, s0, 0x8000
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s43, 0x200000
	s_mov_b32 s9, 0xe000
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s11
	s_lshl_b64 s[0:1], s[16:17], 2
	tensor_load_to_lds s[12:15], s[4:11]
	s_mul_i32 s10, s19, 0x1c00
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[0:1], s[34:35], s[0:1]
	s_lshl_b64 s[8:9], s[10:11], 2
	s_lshl_b32 s4, s19, 10
	s_add_nc_u64 s[14:15], s[0:1], s[8:9]
	s_add_co_i32 s13, s4, 0x10800
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s43, 0x220000
	s_movk_i32 s9, 0x700
	s_mov_b32 s8, 4
	s_mov_b32 s7, 0x407fff
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
.LBB0_9:
	v_cndmask_b32_e64 v1, 0, 1, s54
	s_and_not1_b32 vcc_lo, exec_lo, s54
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e64 s1, 1, v1
	s_cbranch_vccnz .LBB0_11
	s_lshl_b32 s0, s49, 3
	s_mov_b32 s43, 0
	s_sub_co_i32 s0, s52, s0
	s_add_nc_u64 s[4:5], s[40:41], s[42:43]
	s_max_i32 s0, s0, 0
	s_lshl_b32 s6, s49, 14
	s_add_nc_u64 s[14:15], s[4:5], 0x800
	s_and_b32 s19, s51, 0xffff
	s_lshl_b32 s5, s0, 16
	s_lshr_b32 s0, s0, 16
	s_add_co_i32 s13, s6, 0x11000
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s19, 0x200000
	s_or_b32 s6, s5, 0x7fff
	s_or_b32 s7, s0, 0x8000000
	s_mov_b32 s9, 0xe000
	s_mov_b32 s8, 8
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s43
	s_mov_b32 s11, s43
	s_mov_b32 s45, s43
	tensor_load_to_lds s[12:15], s[4:11]
	s_lshl_b64 s[6:7], s[46:47], 2
	s_lshl_b32 s0, s49, 2
	s_add_nc_u64 s[6:7], s[20:21], s[6:7]
	s_lshl_b64 s[8:9], s[44:45], 2
	s_sub_co_i32 s0, s53, s0
	s_add_nc_u64 s[6:7], s[6:7], s[8:9]
	s_max_i32 s0, s0, 0
	s_lshl_b32 s4, s49, 10
	s_add_nc_u64 s[14:15], s[6:7], 0x100
	s_lshl_b32 s6, s0, 16
	s_lshr_b32 s0, s0, 16
	s_add_co_i32 s13, s4, 0x21000
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s19, 0x220000
	s_addk_co_i32 s6, 0x7fff
	s_or_b32 s7, s0, 0x400000
	s_movk_i32 s9, 0x700
	s_mov_b32 s8, 4
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
.LBB0_11:
	v_cndmask_b32_e64 v1, 0, 1, s18
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 1, v1
	s_cbranch_vccz .LBB0_24
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccz .LBB0_25
.LBB0_13:
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB0_26
.LBB0_14:
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccz .LBB0_27
.LBB0_15:
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB0_17
.LBB0_16:
	s_add_co_i32 s18, s49, -2
	s_mov_b32 s11, 0
	s_mul_i32 s10, s18, 0x70000
	s_and_b32 s19, s50, 0xffff
	s_add_nc_u64 s[4:5], s[36:37], s[10:11]
	s_lshl_b32 s6, s18, 14
	s_add_nc_u64 s[14:15], s[4:5], 0x1800
	s_add_co_i32 s13, s6, 0x3b000
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s19, 0x200000
	s_mov_b32 s9, 0xe000
	s_mov_b32 s8, 8
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_lshl_b64 s[8:9], s[16:17], 2
	s_mul_i32 s10, s18, 0x1c00
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[8:9], s[34:35], s[8:9]
	s_lshl_b64 s[14:15], s[10:11], 2
	s_lshl_b32 s4, s18, 10
	s_add_nc_u64 s[8:9], s[8:9], s[14:15]
	s_add_co_i32 s13, s4, 0x43800
	s_add_nc_u64 s[14:15], s[8:9], 0x300
	s_or_b32 s4, s19, 0x220000
	s_bitset1_b32 s15, 31
	s_movk_i32 s9, 0x700
	s_mov_b32 s8, 4
	s_mov_b32 s7, 0x407fff
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
.LBB0_17:
	s_set_vgpr_msb 0xc0
	v_bfe_u32 v18 /*v786*/, v0, 4, 1
	s_set_vgpr_msb 0xc000
	v_and_b32_e32 v1, 31, v0
	s_set_vgpr_msb 0x80
	v_and_b32_e32 v210 /*v722*/, 15, v0
	s_cmp_eq_u32 s49, 0
	s_wait_tensorcnt 0x6
	s_set_vgpr_msb 0x800c
	s_barrier_signal -1
	v_lshlrev_b32_e32 v2, 8, v18 /*v786*/
	s_set_vgpr_msb 0xc08
	v_lshlrev_b32_e32 v3, 4, v210 /*v722*/
	s_set_vgpr_msb 0x800
	v_lshlrev_b32_e32 v1, 2, v1
	s_cselect_b32 s47, -1, 0
	s_and_b32 s46, s49, 1
	s_lshr_b32 s33, s33, 6
	s_lshl_b32 s4, s46, 14
	v_lshl_or_b32 v4, s33, 10, v1
	v_or3_b32 v32, s4, v2, v3
	v_lshl_or_b32 v1, s46, 10, v1
	s_lshl_b32 s5, s33, 14
	s_mov_b32 s8, 8
	s_set_vgpr_msb 0x80
	v_or3_b32 v213 /*v725*/, s5, v2, v3
	v_or_b32_e32 v214 /*v726*/, 0x8000, v32
	v_add_nc_u32_e32 v211 /*v723*/, 0x10000, v4
	v_or_b32_e32 v212 /*v724*/, 0x10800, v1
	s_mov_b32 s16, 4
	s_barrier_wait -1
	s_set_vgpr_msb 0x8002
	v_or_b32_e32 v2, 0x10900, v1
	v_or_b32_e32 v3, 0x10a00, v1
	v_or_b32_e32 v1, 0x10b00, v1
	s_and_b32 vcc_lo, exec_lo, s1
	s_wait_alu depctr_va_vdst(0)
	ds_load_b32 v18, v212 /*v724*/
	s_set_vgpr_msb 0x200
	ds_load_b32 v19, v2
	ds_load_b32 v22, v3
	ds_load_b32 v23, v1
	s_set_vgpr_msb 2
	ds_load_2addr_stride64_b32 v[24:25], v211 /*v723*/ offset1:1
	ds_load_2addr_stride64_b32 v[20:21], v211 /*v723*/ offset0:2 offset1:3
	s_set_vgpr_msb 0x280
	ds_load_b128 v[2:5] /*v[514:517]*/, v32 offset:32768
	ds_load_b128 v[6:9] /*v[518:521]*/, v32 offset:33280
	ds_load_b128 v[10:13] /*v[522:525]*/, v32 offset:34816
	ds_load_b128 v[14:17] /*v[526:529]*/, v32 offset:35328
	ds_load_b128 v[18:21] /*v[530:533]*/, v32 offset:36864
	ds_load_b128 v[22:25] /*v[534:537]*/, v32 offset:37376
	ds_load_b128 v[26:29] /*v[538:541]*/, v32 offset:38912
	ds_load_b128 v[30:33] /*v[542:545]*/, v32 offset:39424
	ds_load_b128 v[34:37] /*v[546:549]*/, v32 offset:40960
	ds_load_b128 v[38:41] /*v[550:553]*/, v32 offset:41472
	ds_load_b128 v[42:45] /*v[554:557]*/, v32 offset:43008
	ds_load_b128 v[46:49] /*v[558:561]*/, v32 offset:43520
	ds_load_b128 v[50:53] /*v[562:565]*/, v32 offset:45056
	ds_load_b128 v[54:57] /*v[566:569]*/, v32 offset:45568
	ds_load_b128 v[58:61] /*v[570:573]*/, v32 offset:47104
	ds_load_b128 v[62:65] /*v[574:577]*/, v32 offset:47616
	s_set_vgpr_msb 0x8082
	ds_load_b128 v[186:189] /*v[698:701]*/, v213 /*v725*/
	ds_load_b128 v[190:193] /*v[702:705]*/, v213 /*v725*/ offset:512
	ds_load_b128 v[178:181] /*v[690:693]*/, v213 /*v725*/ offset:2048
	ds_load_b128 v[182:185] /*v[694:697]*/, v213 /*v725*/ offset:2560
	ds_load_b128 v[170:173] /*v[682:685]*/, v213 /*v725*/ offset:4096
	ds_load_b128 v[174:177] /*v[686:689]*/, v213 /*v725*/ offset:4608
	ds_load_b128 v[162:165] /*v[674:677]*/, v213 /*v725*/ offset:6144
	ds_load_b128 v[166:169] /*v[678:681]*/, v213 /*v725*/ offset:6656
	ds_load_b128 v[154:157] /*v[666:669]*/, v213 /*v725*/ offset:8192
	ds_load_b128 v[158:161] /*v[670:673]*/, v213 /*v725*/ offset:8704
	ds_load_b128 v[146:149] /*v[658:661]*/, v213 /*v725*/ offset:10240
	ds_load_b128 v[150:153] /*v[662:665]*/, v213 /*v725*/ offset:10752
	ds_load_b128 v[138:141] /*v[650:653]*/, v213 /*v725*/ offset:12288
	ds_load_b128 v[142:145] /*v[654:657]*/, v213 /*v725*/ offset:12800
	ds_load_b128 v[130:133] /*v[642:645]*/, v213 /*v725*/ offset:14336
	ds_load_b128 v[134:137] /*v[646:649]*/, v213 /*v725*/ offset:14848
	s_mov_b32 s10, 0
	s_set_vgpr_msb 0x8200
	s_cbranch_vccnz .LBB0_28
	s_lshl_b32 s1, s49, 3
	s_lshl_b32 s13, s49, 2
	s_sub_co_i32 s4, s52, s1
	s_set_vgpr_msb 0x41
	v_mov_b32_e32 v242 /*v498*/, 0
	s_max_i32 s5, s4, 0
	s_mov_b32 s45, s10
	s_sub_co_i32 s13, s53, s13
	s_mul_u64 s[22:23], s[22:23], 0x1c00
	s_mov_b32 s43, s10
	s_lshl_b32 s6, s5, 16
	s_max_i32 s13, s13, 0
	s_add_nc_u64 s[20:21], s[20:21], s[22:23]
	s_lshl_b64 s[22:23], s[44:45], 2
	v_dual_mov_b32 v243 /*v499*/, v242 /*v498*/ :: v_dual_mov_b32 v244 /*v500*/, v242 /*v498*/
	s_lshr_b32 s5, s5, 16
	s_set_vgpr_msb 0x4101
	v_mov_b32_e32 v253, v242 /*v498*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v245 /*v501*/, v242 /*v498*/ :: v_dual_mov_b32 v246 /*v502*/, v242 /*v498*/
	v_dual_mov_b32 v247 /*v503*/, v242 /*v498*/ :: v_dual_mov_b32 v248 /*v504*/, v242 /*v498*/
	v_dual_mov_b32 v249 /*v505*/, v242 /*v498*/ :: v_dual_mov_b32 v250 /*v506*/, v242 /*v498*/
	v_dual_mov_b32 v251 /*v507*/, v242 /*v498*/ :: v_dual_mov_b32 v252 /*v508*/, v242 /*v498*/
	v_dual_mov_b32 v253 /*v509*/, v242 /*v498*/ :: v_dual_mov_b32 v254 /*v510*/, v242 /*v498*/
	v_mov_b32_e32 v255 /*v511*/, v242 /*v498*/
	s_set_vgpr_msb 0x4181
	v_dual_mov_b32 v0 /*v512*/, v242 /*v498*/ :: v_dual_mov_b32 v1 /*v513*/, v242 /*v498*/
	s_set_vgpr_msb 0x8141
	v_dual_mov_b32 v226 /*v482*/, v242 /*v498*/ :: v_dual_mov_b32 v227 /*v483*/, v242 /*v498*/
	v_dual_mov_b32 v228 /*v484*/, v242 /*v498*/ :: v_dual_mov_b32 v229 /*v485*/, v242 /*v498*/
	v_dual_mov_b32 v230 /*v486*/, v242 /*v498*/ :: v_dual_mov_b32 v231 /*v487*/, v242 /*v498*/
	v_dual_mov_b32 v232 /*v488*/, v242 /*v498*/ :: v_dual_mov_b32 v233 /*v489*/, v242 /*v498*/
	v_dual_mov_b32 v234 /*v490*/, v242 /*v498*/ :: v_dual_mov_b32 v235 /*v491*/, v242 /*v498*/
	v_dual_mov_b32 v236 /*v492*/, v242 /*v498*/ :: v_dual_mov_b32 v237 /*v493*/, v242 /*v498*/
	v_dual_mov_b32 v238 /*v494*/, v242 /*v498*/ :: v_dual_mov_b32 v239 /*v495*/, v242 /*v498*/
	v_dual_mov_b32 v240 /*v496*/, v242 /*v498*/ :: v_dual_mov_b32 v241 /*v497*/, v242 /*v498*/
	v_dual_mov_b32 v210 /*v466*/, v242 /*v498*/ :: v_dual_mov_b32 v211 /*v467*/, v242 /*v498*/
	v_dual_mov_b32 v212 /*v468*/, v242 /*v498*/ :: v_dual_mov_b32 v213 /*v469*/, v242 /*v498*/
	v_dual_mov_b32 v214 /*v470*/, v242 /*v498*/ :: v_dual_mov_b32 v215 /*v471*/, v242 /*v498*/
	v_dual_mov_b32 v216 /*v472*/, v242 /*v498*/ :: v_dual_mov_b32 v217 /*v473*/, v242 /*v498*/
	v_dual_mov_b32 v218 /*v474*/, v242 /*v498*/ :: v_dual_mov_b32 v219 /*v475*/, v242 /*v498*/
	v_dual_mov_b32 v220 /*v476*/, v242 /*v498*/ :: v_dual_mov_b32 v221 /*v477*/, v242 /*v498*/
	v_dual_mov_b32 v222 /*v478*/, v242 /*v498*/ :: v_dual_mov_b32 v223 /*v479*/, v242 /*v498*/
	v_dual_mov_b32 v224 /*v480*/, v242 /*v498*/ :: v_dual_mov_b32 v225 /*v481*/, v242 /*v498*/
	v_dual_mov_b32 v194 /*v450*/, v242 /*v498*/ :: v_dual_mov_b32 v195 /*v451*/, v242 /*v498*/
	v_dual_mov_b32 v196 /*v452*/, v242 /*v498*/ :: v_dual_mov_b32 v197 /*v453*/, v242 /*v498*/
	v_dual_mov_b32 v198 /*v454*/, v242 /*v498*/ :: v_dual_mov_b32 v199 /*v455*/, v242 /*v498*/
	v_dual_mov_b32 v200 /*v456*/, v242 /*v498*/ :: v_dual_mov_b32 v201 /*v457*/, v242 /*v498*/
	v_dual_mov_b32 v202 /*v458*/, v242 /*v498*/ :: v_dual_mov_b32 v203 /*v459*/, v242 /*v498*/
	v_dual_mov_b32 v204 /*v460*/, v242 /*v498*/ :: v_dual_mov_b32 v205 /*v461*/, v242 /*v498*/
	v_dual_mov_b32 v206 /*v462*/, v242 /*v498*/ :: v_dual_mov_b32 v207 /*v463*/, v242 /*v498*/
	v_dual_mov_b32 v208 /*v464*/, v242 /*v498*/ :: v_dual_mov_b32 v209 /*v465*/, v242 /*v498*/
	v_dual_mov_b32 v178 /*v434*/, v242 /*v498*/ :: v_dual_mov_b32 v179 /*v435*/, v242 /*v498*/
	v_dual_mov_b32 v180 /*v436*/, v242 /*v498*/ :: v_dual_mov_b32 v181 /*v437*/, v242 /*v498*/
	v_dual_mov_b32 v182 /*v438*/, v242 /*v498*/ :: v_dual_mov_b32 v183 /*v439*/, v242 /*v498*/
	v_dual_mov_b32 v184 /*v440*/, v242 /*v498*/ :: v_dual_mov_b32 v185 /*v441*/, v242 /*v498*/
	v_dual_mov_b32 v186 /*v442*/, v242 /*v498*/ :: v_dual_mov_b32 v187 /*v443*/, v242 /*v498*/
	v_dual_mov_b32 v188 /*v444*/, v242 /*v498*/ :: v_dual_mov_b32 v189 /*v445*/, v242 /*v498*/
	v_dual_mov_b32 v190 /*v446*/, v242 /*v498*/ :: v_dual_mov_b32 v191 /*v447*/, v242 /*v498*/
	v_dual_mov_b32 v192 /*v448*/, v242 /*v498*/ :: v_dual_mov_b32 v193 /*v449*/, v242 /*v498*/
	v_dual_mov_b32 v162 /*v418*/, v242 /*v498*/ :: v_dual_mov_b32 v163 /*v419*/, v242 /*v498*/
	v_dual_mov_b32 v164 /*v420*/, v242 /*v498*/ :: v_dual_mov_b32 v165 /*v421*/, v242 /*v498*/
	v_dual_mov_b32 v166 /*v422*/, v242 /*v498*/ :: v_dual_mov_b32 v167 /*v423*/, v242 /*v498*/
	v_dual_mov_b32 v168 /*v424*/, v242 /*v498*/ :: v_dual_mov_b32 v169 /*v425*/, v242 /*v498*/
	v_dual_mov_b32 v170 /*v426*/, v242 /*v498*/ :: v_dual_mov_b32 v171 /*v427*/, v242 /*v498*/
	v_dual_mov_b32 v172 /*v428*/, v242 /*v498*/ :: v_dual_mov_b32 v173 /*v429*/, v242 /*v498*/
	v_dual_mov_b32 v174 /*v430*/, v242 /*v498*/ :: v_dual_mov_b32 v175 /*v431*/, v242 /*v498*/
	v_dual_mov_b32 v176 /*v432*/, v242 /*v498*/ :: v_dual_mov_b32 v177 /*v433*/, v242 /*v498*/
	v_dual_mov_b32 v146 /*v402*/, v242 /*v498*/ :: v_dual_mov_b32 v147 /*v403*/, v242 /*v498*/
	v_dual_mov_b32 v148 /*v404*/, v242 /*v498*/ :: v_dual_mov_b32 v149 /*v405*/, v242 /*v498*/
	v_dual_mov_b32 v150 /*v406*/, v242 /*v498*/ :: v_dual_mov_b32 v151 /*v407*/, v242 /*v498*/
	v_dual_mov_b32 v152 /*v408*/, v242 /*v498*/ :: v_dual_mov_b32 v153 /*v409*/, v242 /*v498*/
	v_dual_mov_b32 v154 /*v410*/, v242 /*v498*/ :: v_dual_mov_b32 v155 /*v411*/, v242 /*v498*/
	v_dual_mov_b32 v156 /*v412*/, v242 /*v498*/ :: v_dual_mov_b32 v157 /*v413*/, v242 /*v498*/
	v_dual_mov_b32 v158 /*v414*/, v242 /*v498*/ :: v_dual_mov_b32 v159 /*v415*/, v242 /*v498*/
	v_dual_mov_b32 v160 /*v416*/, v242 /*v498*/ :: v_dual_mov_b32 v161 /*v417*/, v242 /*v498*/
	v_dual_mov_b32 v130 /*v386*/, v242 /*v498*/ :: v_dual_mov_b32 v131 /*v387*/, v242 /*v498*/
	v_dual_mov_b32 v132 /*v388*/, v242 /*v498*/ :: v_dual_mov_b32 v133 /*v389*/, v242 /*v498*/
	v_dual_mov_b32 v134 /*v390*/, v242 /*v498*/ :: v_dual_mov_b32 v135 /*v391*/, v242 /*v498*/
	v_dual_mov_b32 v136 /*v392*/, v242 /*v498*/ :: v_dual_mov_b32 v137 /*v393*/, v242 /*v498*/
	v_dual_mov_b32 v138 /*v394*/, v242 /*v498*/ :: v_dual_mov_b32 v139 /*v395*/, v242 /*v498*/
	v_dual_mov_b32 v140 /*v396*/, v242 /*v498*/ :: v_dual_mov_b32 v141 /*v397*/, v242 /*v498*/
	v_dual_mov_b32 v142 /*v398*/, v242 /*v498*/ :: v_dual_mov_b32 v143 /*v399*/, v242 /*v498*/
	v_dual_mov_b32 v144 /*v400*/, v242 /*v498*/ :: v_dual_mov_b32 v145 /*v401*/, v242 /*v498*/
	v_dual_mov_b32 v114 /*v370*/, v242 /*v498*/ :: v_dual_mov_b32 v115 /*v371*/, v242 /*v498*/
	v_dual_mov_b32 v116 /*v372*/, v242 /*v498*/ :: v_dual_mov_b32 v117 /*v373*/, v242 /*v498*/
	v_dual_mov_b32 v118 /*v374*/, v242 /*v498*/ :: v_dual_mov_b32 v119 /*v375*/, v242 /*v498*/
	v_dual_mov_b32 v120 /*v376*/, v242 /*v498*/ :: v_dual_mov_b32 v121 /*v377*/, v242 /*v498*/
	v_dual_mov_b32 v122 /*v378*/, v242 /*v498*/ :: v_dual_mov_b32 v123 /*v379*/, v242 /*v498*/
	v_dual_mov_b32 v124 /*v380*/, v242 /*v498*/ :: v_dual_mov_b32 v125 /*v381*/, v242 /*v498*/
	v_dual_mov_b32 v126 /*v382*/, v242 /*v498*/ :: v_dual_mov_b32 v127 /*v383*/, v242 /*v498*/
	v_dual_mov_b32 v128 /*v384*/, v242 /*v498*/ :: v_dual_mov_b32 v129 /*v385*/, v242 /*v498*/
	v_dual_mov_b32 v98 /*v354*/, v242 /*v498*/ :: v_dual_mov_b32 v99 /*v355*/, v242 /*v498*/
	v_dual_mov_b32 v100 /*v356*/, v242 /*v498*/ :: v_dual_mov_b32 v101 /*v357*/, v242 /*v498*/
	v_dual_mov_b32 v102 /*v358*/, v242 /*v498*/ :: v_dual_mov_b32 v103 /*v359*/, v242 /*v498*/
	v_dual_mov_b32 v104 /*v360*/, v242 /*v498*/ :: v_dual_mov_b32 v105 /*v361*/, v242 /*v498*/
	v_dual_mov_b32 v106 /*v362*/, v242 /*v498*/ :: v_dual_mov_b32 v107 /*v363*/, v242 /*v498*/
	v_dual_mov_b32 v108 /*v364*/, v242 /*v498*/ :: v_dual_mov_b32 v109 /*v365*/, v242 /*v498*/
	v_dual_mov_b32 v110 /*v366*/, v242 /*v498*/ :: v_dual_mov_b32 v111 /*v367*/, v242 /*v498*/
	v_dual_mov_b32 v112 /*v368*/, v242 /*v498*/ :: v_dual_mov_b32 v113 /*v369*/, v242 /*v498*/
	v_dual_mov_b32 v82 /*v338*/, v242 /*v498*/ :: v_dual_mov_b32 v83 /*v339*/, v242 /*v498*/
	v_dual_mov_b32 v84 /*v340*/, v242 /*v498*/ :: v_dual_mov_b32 v85 /*v341*/, v242 /*v498*/
	v_dual_mov_b32 v86 /*v342*/, v242 /*v498*/ :: v_dual_mov_b32 v87 /*v343*/, v242 /*v498*/
	v_dual_mov_b32 v88 /*v344*/, v242 /*v498*/ :: v_dual_mov_b32 v89 /*v345*/, v242 /*v498*/
	v_dual_mov_b32 v90 /*v346*/, v242 /*v498*/ :: v_dual_mov_b32 v91 /*v347*/, v242 /*v498*/
	v_dual_mov_b32 v92 /*v348*/, v242 /*v498*/ :: v_dual_mov_b32 v93 /*v349*/, v242 /*v498*/
	v_dual_mov_b32 v94 /*v350*/, v242 /*v498*/ :: v_dual_mov_b32 v95 /*v351*/, v242 /*v498*/
	v_dual_mov_b32 v96 /*v352*/, v242 /*v498*/ :: v_dual_mov_b32 v97 /*v353*/, v242 /*v498*/
	v_dual_mov_b32 v66 /*v322*/, v242 /*v498*/ :: v_dual_mov_b32 v67 /*v323*/, v242 /*v498*/
	v_dual_mov_b32 v68 /*v324*/, v242 /*v498*/ :: v_dual_mov_b32 v69 /*v325*/, v242 /*v498*/
	v_dual_mov_b32 v70 /*v326*/, v242 /*v498*/ :: v_dual_mov_b32 v71 /*v327*/, v242 /*v498*/
	v_dual_mov_b32 v72 /*v328*/, v242 /*v498*/ :: v_dual_mov_b32 v73 /*v329*/, v242 /*v498*/
	v_dual_mov_b32 v74 /*v330*/, v242 /*v498*/ :: v_dual_mov_b32 v75 /*v331*/, v242 /*v498*/
	v_dual_mov_b32 v76 /*v332*/, v242 /*v498*/ :: v_dual_mov_b32 v77 /*v333*/, v242 /*v498*/
	v_dual_mov_b32 v78 /*v334*/, v242 /*v498*/ :: v_dual_mov_b32 v79 /*v335*/, v242 /*v498*/
	v_dual_mov_b32 v80 /*v336*/, v242 /*v498*/ :: v_dual_mov_b32 v81 /*v337*/, v242 /*v498*/
	v_dual_mov_b32 v50 /*v306*/, v242 /*v498*/ :: v_dual_mov_b32 v51 /*v307*/, v242 /*v498*/
	v_dual_mov_b32 v52 /*v308*/, v242 /*v498*/ :: v_dual_mov_b32 v53 /*v309*/, v242 /*v498*/
	v_dual_mov_b32 v54 /*v310*/, v242 /*v498*/ :: v_dual_mov_b32 v55 /*v311*/, v242 /*v498*/
	v_dual_mov_b32 v56 /*v312*/, v242 /*v498*/ :: v_dual_mov_b32 v57 /*v313*/, v242 /*v498*/
	v_dual_mov_b32 v58 /*v314*/, v242 /*v498*/ :: v_dual_mov_b32 v59 /*v315*/, v242 /*v498*/
	v_dual_mov_b32 v60 /*v316*/, v242 /*v498*/ :: v_dual_mov_b32 v61 /*v317*/, v242 /*v498*/
	v_dual_mov_b32 v62 /*v318*/, v242 /*v498*/ :: v_dual_mov_b32 v63 /*v319*/, v242 /*v498*/
	v_dual_mov_b32 v64 /*v320*/, v242 /*v498*/ :: v_dual_mov_b32 v65 /*v321*/, v242 /*v498*/
	v_dual_mov_b32 v34 /*v290*/, v242 /*v498*/ :: v_dual_mov_b32 v35 /*v291*/, v242 /*v498*/
	v_dual_mov_b32 v36 /*v292*/, v242 /*v498*/ :: v_dual_mov_b32 v37 /*v293*/, v242 /*v498*/
	v_dual_mov_b32 v38 /*v294*/, v242 /*v498*/ :: v_dual_mov_b32 v39 /*v295*/, v242 /*v498*/
	v_dual_mov_b32 v40 /*v296*/, v242 /*v498*/ :: v_dual_mov_b32 v41 /*v297*/, v242 /*v498*/
	v_dual_mov_b32 v42 /*v298*/, v242 /*v498*/ :: v_dual_mov_b32 v43 /*v299*/, v242 /*v498*/
	v_dual_mov_b32 v44 /*v300*/, v242 /*v498*/ :: v_dual_mov_b32 v45 /*v301*/, v242 /*v498*/
	v_dual_mov_b32 v46 /*v302*/, v242 /*v498*/ :: v_dual_mov_b32 v47 /*v303*/, v242 /*v498*/
	v_dual_mov_b32 v48 /*v304*/, v242 /*v498*/ :: v_dual_mov_b32 v49 /*v305*/, v242 /*v498*/
	v_dual_mov_b32 v18 /*v274*/, v242 /*v498*/ :: v_dual_mov_b32 v19 /*v275*/, v242 /*v498*/
	v_dual_mov_b32 v20 /*v276*/, v242 /*v498*/ :: v_dual_mov_b32 v21 /*v277*/, v242 /*v498*/
	v_dual_mov_b32 v22 /*v278*/, v242 /*v498*/ :: v_dual_mov_b32 v23 /*v279*/, v242 /*v498*/
	v_dual_mov_b32 v24 /*v280*/, v242 /*v498*/ :: v_dual_mov_b32 v25 /*v281*/, v242 /*v498*/
	v_dual_mov_b32 v26 /*v282*/, v242 /*v498*/ :: v_dual_mov_b32 v27 /*v283*/, v242 /*v498*/
	v_dual_mov_b32 v28 /*v284*/, v242 /*v498*/ :: v_dual_mov_b32 v29 /*v285*/, v242 /*v498*/
	v_dual_mov_b32 v30 /*v286*/, v242 /*v498*/ :: v_dual_mov_b32 v31 /*v287*/, v242 /*v498*/
	v_dual_mov_b32 v16 /*v272*/, v242 /*v498*/ :: v_dual_mov_b32 v15 /*v271*/, v242 /*v498*/
	v_dual_mov_b32 v14 /*v270*/, v242 /*v498*/ :: v_dual_mov_b32 v13 /*v269*/, v242 /*v498*/
	v_dual_mov_b32 v12 /*v268*/, v242 /*v498*/ :: v_dual_mov_b32 v11 /*v267*/, v242 /*v498*/
	v_dual_mov_b32 v10 /*v266*/, v242 /*v498*/ :: v_dual_mov_b32 v9 /*v265*/, v242 /*v498*/
	v_dual_mov_b32 v8 /*v264*/, v242 /*v498*/ :: v_dual_mov_b32 v7 /*v263*/, v242 /*v498*/
	v_dual_mov_b32 v6 /*v262*/, v242 /*v498*/ :: v_dual_mov_b32 v5 /*v261*/, v242 /*v498*/
	v_dual_mov_b32 v4 /*v260*/, v242 /*v498*/ :: v_dual_mov_b32 v3 /*v259*/, v242 /*v498*/
	v_dual_mov_b32 v2 /*v258*/, v242 /*v498*/ :: v_dual_mov_b32 v33 /*v289*/, v242 /*v498*/
	v_dual_mov_b32 v32 /*v288*/, v242 /*v498*/ :: v_dual_mov_b32 v17 /*v273*/, v242 /*v498*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v242, v242 /*v498*/ :: v_dual_mov_b32 v243, v242 /*v498*/
	v_dual_mov_b32 v244, v242 /*v498*/ :: v_dual_mov_b32 v245, v242 /*v498*/
	v_dual_mov_b32 v246, v242 /*v498*/ :: v_dual_mov_b32 v247, v242 /*v498*/
	v_dual_mov_b32 v248, v242 /*v498*/ :: v_dual_mov_b32 v249, v242 /*v498*/
	v_dual_mov_b32 v250, v242 /*v498*/ :: v_dual_mov_b32 v251, v242 /*v498*/
	v_dual_mov_b32 v252, v242 /*v498*/ :: v_dual_mov_b32 v254, v242 /*v498*/
	v_mov_b32_e32 v255, v242 /*v498*/
	s_set_vgpr_msb 0x141
	v_dual_mov_b32 v0 /*v256*/, v242 /*v498*/ :: v_dual_mov_b32 v1 /*v257*/, v242 /*v498*/
	s_set_vgpr_msb 0x4101
	v_dual_mov_b32 v226, v242 /*v498*/ :: v_dual_mov_b32 v227, v242 /*v498*/
	v_dual_mov_b32 v228, v242 /*v498*/ :: v_dual_mov_b32 v229, v242 /*v498*/
	v_dual_mov_b32 v230, v242 /*v498*/ :: v_dual_mov_b32 v231, v242 /*v498*/
	v_dual_mov_b32 v232, v242 /*v498*/ :: v_dual_mov_b32 v233, v242 /*v498*/
	v_dual_mov_b32 v234, v242 /*v498*/ :: v_dual_mov_b32 v235, v242 /*v498*/
	v_dual_mov_b32 v236, v242 /*v498*/ :: v_dual_mov_b32 v237, v242 /*v498*/
	v_dual_mov_b32 v238, v242 /*v498*/ :: v_dual_mov_b32 v239, v242 /*v498*/
	v_dual_mov_b32 v240, v242 /*v498*/ :: v_dual_mov_b32 v241, v242 /*v498*/
	v_dual_mov_b32 v210, v242 /*v498*/ :: v_dual_mov_b32 v211, v242 /*v498*/
	v_dual_mov_b32 v212, v242 /*v498*/ :: v_dual_mov_b32 v213, v242 /*v498*/
	v_dual_mov_b32 v214, v242 /*v498*/ :: v_dual_mov_b32 v215, v242 /*v498*/
	v_dual_mov_b32 v216, v242 /*v498*/ :: v_dual_mov_b32 v217, v242 /*v498*/
	v_dual_mov_b32 v218, v242 /*v498*/ :: v_dual_mov_b32 v219, v242 /*v498*/
	v_dual_mov_b32 v220, v242 /*v498*/ :: v_dual_mov_b32 v221, v242 /*v498*/
	v_dual_mov_b32 v222, v242 /*v498*/ :: v_dual_mov_b32 v223, v242 /*v498*/
	v_dual_mov_b32 v224, v242 /*v498*/ :: v_dual_mov_b32 v225, v242 /*v498*/
	v_dual_mov_b32 v194, v242 /*v498*/ :: v_dual_mov_b32 v195, v242 /*v498*/
	v_dual_mov_b32 v196, v242 /*v498*/ :: v_dual_mov_b32 v197, v242 /*v498*/
	v_dual_mov_b32 v198, v242 /*v498*/ :: v_dual_mov_b32 v199, v242 /*v498*/
	v_dual_mov_b32 v200, v242 /*v498*/ :: v_dual_mov_b32 v201, v242 /*v498*/
	v_dual_mov_b32 v202, v242 /*v498*/ :: v_dual_mov_b32 v203, v242 /*v498*/
	v_dual_mov_b32 v204, v242 /*v498*/ :: v_dual_mov_b32 v205, v242 /*v498*/
	v_dual_mov_b32 v206, v242 /*v498*/ :: v_dual_mov_b32 v207, v242 /*v498*/
	v_dual_mov_b32 v208, v242 /*v498*/ :: v_dual_mov_b32 v209, v242 /*v498*/
	v_dual_mov_b32 v178, v242 /*v498*/ :: v_dual_mov_b32 v179, v242 /*v498*/
	v_dual_mov_b32 v180, v242 /*v498*/ :: v_dual_mov_b32 v181, v242 /*v498*/
	v_dual_mov_b32 v182, v242 /*v498*/ :: v_dual_mov_b32 v183, v242 /*v498*/
	v_dual_mov_b32 v184, v242 /*v498*/ :: v_dual_mov_b32 v185, v242 /*v498*/
	v_dual_mov_b32 v186, v242 /*v498*/ :: v_dual_mov_b32 v187, v242 /*v498*/
	v_dual_mov_b32 v188, v242 /*v498*/ :: v_dual_mov_b32 v189, v242 /*v498*/
	v_dual_mov_b32 v190, v242 /*v498*/ :: v_dual_mov_b32 v191, v242 /*v498*/
	v_dual_mov_b32 v192, v242 /*v498*/ :: v_dual_mov_b32 v193, v242 /*v498*/
	v_dual_mov_b32 v162, v242 /*v498*/ :: v_dual_mov_b32 v163, v242 /*v498*/
	v_dual_mov_b32 v164, v242 /*v498*/ :: v_dual_mov_b32 v165, v242 /*v498*/
	v_dual_mov_b32 v166, v242 /*v498*/ :: v_dual_mov_b32 v167, v242 /*v498*/
	v_dual_mov_b32 v168, v242 /*v498*/ :: v_dual_mov_b32 v169, v242 /*v498*/
	v_dual_mov_b32 v170, v242 /*v498*/ :: v_dual_mov_b32 v171, v242 /*v498*/
	v_dual_mov_b32 v172, v242 /*v498*/ :: v_dual_mov_b32 v173, v242 /*v498*/
	v_dual_mov_b32 v174, v242 /*v498*/ :: v_dual_mov_b32 v175, v242 /*v498*/
	v_dual_mov_b32 v176, v242 /*v498*/ :: v_dual_mov_b32 v177, v242 /*v498*/
	v_dual_mov_b32 v146, v242 /*v498*/ :: v_dual_mov_b32 v147, v242 /*v498*/
	v_dual_mov_b32 v148, v242 /*v498*/ :: v_dual_mov_b32 v149, v242 /*v498*/
	v_dual_mov_b32 v150, v242 /*v498*/ :: v_dual_mov_b32 v151, v242 /*v498*/
	v_dual_mov_b32 v152, v242 /*v498*/ :: v_dual_mov_b32 v153, v242 /*v498*/
	v_dual_mov_b32 v154, v242 /*v498*/ :: v_dual_mov_b32 v155, v242 /*v498*/
	v_dual_mov_b32 v156, v242 /*v498*/ :: v_dual_mov_b32 v157, v242 /*v498*/
	v_dual_mov_b32 v158, v242 /*v498*/ :: v_dual_mov_b32 v159, v242 /*v498*/
	v_dual_mov_b32 v160, v242 /*v498*/ :: v_dual_mov_b32 v161, v242 /*v498*/
	v_dual_mov_b32 v130, v242 /*v498*/ :: v_dual_mov_b32 v131, v242 /*v498*/
	v_dual_mov_b32 v132, v242 /*v498*/ :: v_dual_mov_b32 v133, v242 /*v498*/
	v_dual_mov_b32 v134, v242 /*v498*/ :: v_dual_mov_b32 v135, v242 /*v498*/
	v_dual_mov_b32 v136, v242 /*v498*/ :: v_dual_mov_b32 v137, v242 /*v498*/
	v_dual_mov_b32 v138, v242 /*v498*/ :: v_dual_mov_b32 v139, v242 /*v498*/
	v_dual_mov_b32 v140, v242 /*v498*/ :: v_dual_mov_b32 v141, v242 /*v498*/
	v_dual_mov_b32 v142, v242 /*v498*/ :: v_dual_mov_b32 v143, v242 /*v498*/
	v_dual_mov_b32 v144, v242 /*v498*/ :: v_dual_mov_b32 v145, v242 /*v498*/
	v_dual_mov_b32 v114, v242 /*v498*/ :: v_dual_mov_b32 v115, v242 /*v498*/
	v_dual_mov_b32 v116, v242 /*v498*/ :: v_dual_mov_b32 v117, v242 /*v498*/
	v_dual_mov_b32 v118, v242 /*v498*/ :: v_dual_mov_b32 v119, v242 /*v498*/
	v_dual_mov_b32 v120, v242 /*v498*/ :: v_dual_mov_b32 v121, v242 /*v498*/
	v_dual_mov_b32 v122, v242 /*v498*/ :: v_dual_mov_b32 v123, v242 /*v498*/
	v_dual_mov_b32 v124, v242 /*v498*/ :: v_dual_mov_b32 v125, v242 /*v498*/
	v_dual_mov_b32 v126, v242 /*v498*/ :: v_dual_mov_b32 v127, v242 /*v498*/
	v_dual_mov_b32 v128, v242 /*v498*/ :: v_dual_mov_b32 v129, v242 /*v498*/
	v_dual_mov_b32 v98, v242 /*v498*/ :: v_dual_mov_b32 v99, v242 /*v498*/
	v_dual_mov_b32 v100, v242 /*v498*/ :: v_dual_mov_b32 v101, v242 /*v498*/
	v_dual_mov_b32 v102, v242 /*v498*/ :: v_dual_mov_b32 v103, v242 /*v498*/
	v_dual_mov_b32 v104, v242 /*v498*/ :: v_dual_mov_b32 v105, v242 /*v498*/
	v_dual_mov_b32 v106, v242 /*v498*/ :: v_dual_mov_b32 v107, v242 /*v498*/
	v_dual_mov_b32 v108, v242 /*v498*/ :: v_dual_mov_b32 v109, v242 /*v498*/
	v_dual_mov_b32 v110, v242 /*v498*/ :: v_dual_mov_b32 v111, v242 /*v498*/
	v_dual_mov_b32 v112, v242 /*v498*/ :: v_dual_mov_b32 v113, v242 /*v498*/
	v_dual_mov_b32 v82, v242 /*v498*/ :: v_dual_mov_b32 v83, v242 /*v498*/
	v_dual_mov_b32 v84, v242 /*v498*/ :: v_dual_mov_b32 v85, v242 /*v498*/
	v_dual_mov_b32 v86, v242 /*v498*/ :: v_dual_mov_b32 v87, v242 /*v498*/
	v_dual_mov_b32 v88, v242 /*v498*/ :: v_dual_mov_b32 v89, v242 /*v498*/
	v_dual_mov_b32 v90, v242 /*v498*/ :: v_dual_mov_b32 v91, v242 /*v498*/
	v_dual_mov_b32 v92, v242 /*v498*/ :: v_dual_mov_b32 v93, v242 /*v498*/
	v_dual_mov_b32 v94, v242 /*v498*/ :: v_dual_mov_b32 v95, v242 /*v498*/
	v_dual_mov_b32 v96, v242 /*v498*/ :: v_dual_mov_b32 v97, v242 /*v498*/
	v_dual_mov_b32 v66, v242 /*v498*/ :: v_dual_mov_b32 v67, v242 /*v498*/
	v_dual_mov_b32 v68, v242 /*v498*/ :: v_dual_mov_b32 v69, v242 /*v498*/
	v_dual_mov_b32 v70, v242 /*v498*/ :: v_dual_mov_b32 v71, v242 /*v498*/
	v_dual_mov_b32 v72, v242 /*v498*/ :: v_dual_mov_b32 v73, v242 /*v498*/
	v_dual_mov_b32 v74, v242 /*v498*/ :: v_dual_mov_b32 v75, v242 /*v498*/
	v_dual_mov_b32 v76, v242 /*v498*/ :: v_dual_mov_b32 v77, v242 /*v498*/
	v_dual_mov_b32 v78, v242 /*v498*/ :: v_dual_mov_b32 v79, v242 /*v498*/
	v_dual_mov_b32 v80, v242 /*v498*/ :: v_dual_mov_b32 v81, v242 /*v498*/
	v_dual_mov_b32 v50, v242 /*v498*/ :: v_dual_mov_b32 v51, v242 /*v498*/
	v_dual_mov_b32 v52, v242 /*v498*/ :: v_dual_mov_b32 v53, v242 /*v498*/
	v_dual_mov_b32 v54, v242 /*v498*/ :: v_dual_mov_b32 v55, v242 /*v498*/
	v_dual_mov_b32 v56, v242 /*v498*/ :: v_dual_mov_b32 v57, v242 /*v498*/
	v_dual_mov_b32 v58, v242 /*v498*/ :: v_dual_mov_b32 v59, v242 /*v498*/
	v_dual_mov_b32 v60, v242 /*v498*/ :: v_dual_mov_b32 v61, v242 /*v498*/
	v_dual_mov_b32 v62, v242 /*v498*/ :: v_dual_mov_b32 v63, v242 /*v498*/
	v_dual_mov_b32 v64, v242 /*v498*/ :: v_dual_mov_b32 v65, v242 /*v498*/
	s_set_vgpr_msb 0x1c1
	v_dual_mov_b32 v20 /*v788*/, v242 /*v498*/ :: v_dual_mov_b32 v21 /*v789*/, v242 /*v498*/
	v_dual_mov_b32 v22 /*v790*/, v242 /*v498*/ :: v_dual_mov_b32 v23 /*v791*/, v242 /*v498*/
	v_dual_mov_b32 v24 /*v792*/, v242 /*v498*/ :: v_dual_mov_b32 v25 /*v793*/, v242 /*v498*/
	v_dual_mov_b32 v26 /*v794*/, v242 /*v498*/ :: v_dual_mov_b32 v27 /*v795*/, v242 /*v498*/
	v_dual_mov_b32 v28 /*v796*/, v242 /*v498*/ :: v_dual_mov_b32 v29 /*v797*/, v242 /*v498*/
	v_dual_mov_b32 v30 /*v798*/, v242 /*v498*/ :: v_dual_mov_b32 v31 /*v799*/, v242 /*v498*/
	v_dual_mov_b32 v32 /*v800*/, v242 /*v498*/ :: v_dual_mov_b32 v33 /*v801*/, v242 /*v498*/
	v_dual_mov_b32 v34 /*v802*/, v242 /*v498*/ :: v_dual_mov_b32 v35 /*v803*/, v242 /*v498*/
	s_set_vgpr_msb 0xc181
	v_dual_mov_b32 v194 /*v706*/, v242 /*v498*/ :: v_dual_mov_b32 v195 /*v707*/, v242 /*v498*/
	v_dual_mov_b32 v196 /*v708*/, v242 /*v498*/ :: v_dual_mov_b32 v197 /*v709*/, v242 /*v498*/
	v_dual_mov_b32 v198 /*v710*/, v242 /*v498*/ :: v_dual_mov_b32 v199 /*v711*/, v242 /*v498*/
	v_dual_mov_b32 v200 /*v712*/, v242 /*v498*/ :: v_dual_mov_b32 v201 /*v713*/, v242 /*v498*/
	v_dual_mov_b32 v202 /*v714*/, v242 /*v498*/ :: v_dual_mov_b32 v203 /*v715*/, v242 /*v498*/
	v_dual_mov_b32 v204 /*v716*/, v242 /*v498*/ :: v_dual_mov_b32 v205 /*v717*/, v242 /*v498*/
	v_dual_mov_b32 v206 /*v718*/, v242 /*v498*/ :: v_dual_mov_b32 v207 /*v719*/, v242 /*v498*/
	v_dual_mov_b32 v208 /*v720*/, v242 /*v498*/ :: v_dual_mov_b32 v209 /*v721*/, v242 /*v498*/
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 0x8101
	v_dual_mov_b32 v2, v242 /*v498*/ :: v_dual_mov_b32 v3, v242 /*v498*/
	v_dual_mov_b32 v4, v242 /*v498*/ :: v_dual_mov_b32 v5, v242 /*v498*/
	v_dual_mov_b32 v6, v242 /*v498*/ :: v_dual_mov_b32 v7, v242 /*v498*/
	v_dual_mov_b32 v8, v242 /*v498*/ :: v_dual_mov_b32 v9, v242 /*v498*/
	v_dual_mov_b32 v10, v242 /*v498*/ :: v_dual_mov_b32 v11, v242 /*v498*/
	v_dual_mov_b32 v12, v242 /*v498*/ :: v_dual_mov_b32 v13, v242 /*v498*/
	v_dual_mov_b32 v14, v242 /*v498*/ :: v_dual_mov_b32 v15, v242 /*v498*/
	v_dual_mov_b32 v16, v242 /*v498*/ :: v_dual_mov_b32 v17, v242 /*v498*/
	s_and_b32 s12, s51, 0xffff
	s_or_b32 s7, s5, 0x8000000
	s_mov_b32 s5, 0xffff0000
	s_lshl_b32 s51, s49, 10
	s_lshl_b32 s14, s13, 16
	s_lshr_b32 s13, s13, 16
	s_add_nc_u64 s[20:21], s[20:21], s[22:23]
	s_add_nc_u64 s[22:23], s[40:41], s[42:43]
	s_lshl_b32 s1, s49, 14
	s_or_b32 s4, s12, 0x200000
	s_addk_co_i32 s6, 0x7fff
	s_mov_b32 s9, 0xe000
	s_mov_b32 s11, s10
	s_add_co_i32 s51, s51, 0x10000
	s_or_b32 s12, s12, 0x220000
	s_addk_co_i32 s14, 0x7fff
	s_or_b32 s15, s13, 0x400000
	s_movk_i32 s17, 0x700
	s_mov_b32 s13, s5
	s_mov_b32 s18, s10
	s_mov_b32 s19, s10
	s_add_nc_u64 s[40:41], s[20:21], 0x400
	s_add_nc_u64 s[22:23], s[22:23], 0x2000
	s_mov_b32 s20, 1
	s_mov_b32 s42, s10
	s_set_vgpr_msb 0x100
	s_branch .LBB0_21
.LBB0_19:
	s_barrier_wait -3
.LBB0_20:
	s_add_nc_u64 s[40:41], s[40:41], 0x100
	s_cmp_eq_u32 s42, 24
	s_add_nc_u64 s[22:23], s[22:23], 0x800
	s_cbranch_scc1 .LBB0_29
.LBB0_21:
	s_and_b32 s44, s42, 3
	s_add_co_i32 s42, s42, 1
	s_mul_i32 s44, s44, 0x11000
	s_and_b32 s43, s42, 3
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v1, s44, v213 /*v725*/
	s_mul_i32 s21, s43, 0x11000
	v_dual_add_nc_u32 v33, s44, v214 /*v726*/ :: v_dual_add_nc_u32 v28, s44, v211 /*v723*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v29, s44, v212 /*v724*/ :: v_dual_add_nc_u32 v36, s21, v213 /*v725*/
	v_dual_add_nc_u32 v37, s21, v214 /*v726*/ :: v_dual_add_nc_u32 v38, s21, v211 /*v723*/
	v_add_nc_u32_e32 v39, s21, v212 /*v724*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v18, v24
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v19, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v22, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v23, v24 matrix_a_reuse
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[186:189] /*v[698:701]*/, v1 offset:1024
	ds_load_b128 v[190:193] /*v[702:705]*/, v1 offset:1536
	s_set_vgpr_msb 0x8000
	ds_load_2addr_b32 v[26:27], v29 offset0:32 offset1:96
	ds_load_2addr_b32 v[30:31], v29 offset0:160 offset1:224
	ds_load_2addr_b32 v[34:35], v28 offset0:32 offset1:96
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_2addr_b32 v[28:29], v28 offset0:160 offset1:224
	s_set_vgpr_msb 0x80
	ds_load_b128 v[66:69] /*v[578:581]*/, v33 offset:1024
	ds_load_b128 v[70:73] /*v[582:585]*/, v33 offset:1536
	ds_load_b128 v[74:77] /*v[586:589]*/, v33 offset:3072
	ds_load_b128 v[78:81] /*v[590:593]*/, v33 offset:3584
	ds_load_b128 v[82:85] /*v[594:597]*/, v33 offset:5120
	ds_load_b128 v[86:89] /*v[598:601]*/, v33 offset:5632
	ds_load_b128 v[90:93] /*v[602:605]*/, v33 offset:7168
	ds_load_b128 v[94:97] /*v[606:609]*/, v33 offset:7680
	ds_load_b128 v[98:101] /*v[610:613]*/, v33 offset:9216
	ds_load_b128 v[102:105] /*v[614:617]*/, v33 offset:9728
	ds_load_b128 v[106:109] /*v[618:621]*/, v33 offset:11264
	ds_load_b128 v[110:113] /*v[622:625]*/, v33 offset:11776
	ds_load_b128 v[114:117] /*v[626:629]*/, v33 offset:13312
	ds_load_b128 v[118:121] /*v[630:633]*/, v33 offset:13824
	ds_load_b128 v[122:125] /*v[634:637]*/, v33 offset:15360
	ds_load_b128 v[126:129] /*v[638:641]*/, v33 offset:15872
	ds_load_b128 v[216:219] /*v[728:731]*/, v1 offset:3072
	ds_load_b128 v[220:223] /*v[732:735]*/, v1 offset:3584
	ds_load_b128 v[224:227] /*v[736:739]*/, v1 offset:5120
	ds_load_b128 v[228:231] /*v[740:743]*/, v1 offset:5632
	ds_load_b128 v[232:235] /*v[744:747]*/, v1 offset:7168
	ds_load_b128 v[236:239] /*v[748:751]*/, v1 offset:7680
	ds_load_b128 v[240:243] /*v[752:755]*/, v1 offset:9216
	ds_load_b128 v[244:247] /*v[756:759]*/, v1 offset:9728
	ds_load_b128 v[248:251] /*v[760:763]*/, v1 offset:11264
	ds_load_b128 v[252:255] /*v[764:767]*/, v1 offset:11776
	s_set_vgpr_msb 0x80c0
	ds_load_b128 v[0:3] /*v[768:771]*/, v1 offset:13312
	ds_load_b128 v[4:7] /*v[772:775]*/, v1 offset:13824
	ds_load_b128 v[8:11] /*v[776:779]*/, v1 offset:15360
	ds_load_b128 v[12:15] /*v[780:783]*/, v1 offset:15872
	s_set_vgpr_msb 0xc05a
	s_wait_dscnt 0x30
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v23, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v22, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v19, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v18, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0x2e
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[114:129] /*v[370:385]*/, v18, v25 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[98:113] /*v[354:369]*/, v19, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[82:97] /*v[338:353]*/, v22, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[66:81] /*v[322:337]*/, v23, v25 matrix_a_reuse
	s_wait_dscnt 0x2c
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[2:17] /*v[258:273]*/, v23, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[18:33] /*v[274:289]*/, v22, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[34:49] /*v[290:305]*/, v19, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[50:65] /*v[306:321]*/, v18, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a0a
	s_wait_dscnt 0x2a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[242:257], v18, v20 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[226:241], v19, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[210:225], v22, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[194:209], v23, v20 matrix_a_reuse
	s_wait_dscnt 0x28
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[130:145], v23, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[146:161], v22, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[162:177], v19, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[178:193], v18, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0x26
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129], v18, v21 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113], v19, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97], v22, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81], v23, v21 matrix_a_reuse
	s_wait_dscnt 0x24
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[2:17], v23, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[194:209] /*v[706:721]*/, v22, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaafa
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[20:35] /*v[788:803]*/, v19, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[50:65], v18, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_add_co_i32 s21, s1, s44
	s_or_b32 s45, s23, 0x80000000
	s_mov_b64 s[54:55], s[22:23]
	s_wait_tensorcnt 0x4
	s_mov_b64 s[52:53], s[20:21]
	s_mov_b32 s55, s45
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	tensor_load_to_lds s[52:55], s[4:11]
	s_add_co_i32 s21, s51, s44
	s_or_b32 s44, s41, 0x80000000
	s_mov_b64 s[54:55], s[22:23]
	s_mov_b64 s[52:53], s[20:21]
	s_mov_b32 s54, s40
	s_mov_b32 s55, s44
	s_set_vgpr_msb 0xa5a
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v26, v34
	tensor_load_to_lds s[52:55], s[12:19]
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v27, v34 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v30, v34 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[114:129] /*v[626:641]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v31, v34 matrix_a_reuse
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_stride64_b32 v[18:19], v39 offset1:1
	ds_load_2addr_stride64_b32 v[22:23], v39 offset0:2 offset1:3
	ds_load_2addr_stride64_b32 v[24:25], v38 offset1:1
	ds_load_2addr_stride64_b32 v[20:21], v38 offset0:2 offset1:3
	s_set_vgpr_msb 0x80
	ds_load_b128 v[2:5] /*v[514:517]*/, v37
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[216:223] /*v[728:735]*/, v[130:145] /*v[386:401]*/, v31, v34 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[216:223] /*v[728:735]*/, v[146:161] /*v[402:417]*/, v30, v34 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[216:223] /*v[728:735]*/, v[162:177] /*v[418:433]*/, v27, v34 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[216:223] /*v[728:735]*/, v[178:193] /*v[434:449]*/, v26, v34 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[6:9] /*v[518:521]*/, v37 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v37 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v37 offset:2560
	ds_load_b128 v[18:21] /*v[530:533]*/, v37 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v37 offset:4608
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[224:231] /*v[736:743]*/, v[114:129] /*v[370:385]*/, v26, v35 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[224:231] /*v[736:743]*/, v[98:113] /*v[354:369]*/, v27, v35 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[224:231] /*v[736:743]*/, v[82:97] /*v[338:353]*/, v30, v35 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[224:231] /*v[736:743]*/, v[66:81] /*v[322:337]*/, v31, v35 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[26:29] /*v[538:541]*/, v37 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v37 offset:6656
	ds_load_b128 v[34:37] /*v[546:549]*/, v37 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v37 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v37 offset:10240
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[232:239] /*v[744:751]*/, v[2:17] /*v[258:273]*/, v31, v35 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[232:239] /*v[744:751]*/, v[18:33] /*v[274:289]*/, v30, v35 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[232:239] /*v[744:751]*/, v[34:49] /*v[290:305]*/, v27, v35 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[232:239] /*v[744:751]*/, v[50:65] /*v[306:321]*/, v26, v35 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[46:49] /*v[558:561]*/, v37 offset:10752
	ds_load_b128 v[50:53] /*v[562:565]*/, v37 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v37 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v37 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v37 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[240:247] /*v[752:759]*/, v[242:257], v26, v28 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[240:247] /*v[752:759]*/, v[226:241], v27, v28 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[240:247] /*v[752:759]*/, v[210:225], v30, v28 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[240:247] /*v[752:759]*/, v[194:209], v31, v28 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[186:189] /*v[698:701]*/, v36
	ds_load_b128 v[190:193] /*v[702:705]*/, v36 offset:512
	ds_load_b128 v[178:181] /*v[690:693]*/, v36 offset:2048
	ds_load_b128 v[182:185] /*v[694:697]*/, v36 offset:2560
	ds_load_b128 v[170:173] /*v[682:685]*/, v36 offset:4096
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[248:255] /*v[760:767]*/, v[130:145], v31, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[248:255] /*v[760:767]*/, v[146:161], v30, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[248:255] /*v[760:767]*/, v[162:177], v27, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[248:255] /*v[760:767]*/, v[178:193], v26, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[174:177] /*v[686:689]*/, v36 offset:4608
	ds_load_b128 v[162:165] /*v[674:677]*/, v36 offset:6144
	ds_load_b128 v[166:169] /*v[678:681]*/, v36 offset:6656
	ds_load_b128 v[154:157] /*v[666:669]*/, v36 offset:8192
	ds_load_b128 v[158:161] /*v[670:673]*/, v36 offset:8704
	s_set_vgpr_msb 0x800e
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[0:7] /*v[768:775]*/, v[114:129], v26, v29 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[0:7] /*v[768:775]*/, v[98:113], v27, v29 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[0:7] /*v[768:775]*/, v[82:97], v30, v29 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[114:129] /*v[626:641]*/, v[0:7] /*v[768:775]*/, v[66:81], v31, v29 matrix_a_reuse
	s_set_vgpr_msb 0xe80
	ds_load_b128 v[146:149] /*v[658:661]*/, v36 offset:10240
	ds_load_b128 v[150:153] /*v[662:665]*/, v36 offset:10752
	ds_load_b128 v[138:141] /*v[650:653]*/, v36 offset:12288
	ds_load_b128 v[142:145] /*v[654:657]*/, v36 offset:12800
	ds_load_b128 v[130:133] /*v[642:645]*/, v36 offset:14336
	s_set_vgpr_msb 0x800e
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[114:129] /*v[626:641]*/, v[8:15] /*v[776:783]*/, v[2:17], v31, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xeae
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[98:113] /*v[610:625]*/, v[8:15] /*v[776:783]*/, v[194:209] /*v[706:721]*/, v30, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaefe
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[82:97] /*v[594:609]*/, v[8:15] /*v[776:783]*/, v[20:35] /*v[788:803]*/, v27, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfe0e
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[66:81] /*v[578:593]*/, v[8:15] /*v[776:783]*/, v[50:65], v26, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xe80
	ds_load_b128 v[134:137] /*v[646:649]*/, v36 offset:14848
	s_cmp_eq_u32 s43, 0
	s_cselect_b32 s21, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b32 s21, s48, s21
	s_and_b32 vcc_lo, exec_lo, s21
	s_set_vgpr_msb 0x8000
	s_cbranch_vccz .LBB0_20
	s_and_not1_b32 vcc_lo, exec_lo, s47
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_19
	s_barrier_signal -3
	s_branch .LBB0_19
.LBB0_24:
	s_add_co_i32 s18, s49, -2
	s_mov_b32 s11, 0
	s_mul_i32 s10, s18, 0x70000
	s_and_b32 s19, s50, 0xffff
	s_add_nc_u64 s[4:5], s[36:37], s[10:11]
	s_lshl_b32 s6, s18, 14
	s_add_nc_u64 s[14:15], s[4:5], 0x800
	s_add_co_i32 s13, s6, 0x19000
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s19, 0x200000
	s_mov_b32 s9, 0xe000
	s_mov_b32 s8, 8
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_lshl_b64 s[8:9], s[16:17], 2
	s_mul_i32 s10, s18, 0x1c00
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[8:9], s[34:35], s[8:9]
	s_lshl_b64 s[14:15], s[10:11], 2
	s_lshl_b32 s4, s18, 10
	s_add_nc_u64 s[8:9], s[8:9], s[14:15]
	s_add_co_i32 s13, s4, 0x21800
	s_add_nc_u64 s[14:15], s[8:9], 0x100
	s_or_b32 s4, s19, 0x220000
	s_bitset1_b32 s15, 31
	s_movk_i32 s9, 0x700
	s_mov_b32 s8, 4
	s_mov_b32 s7, 0x407fff
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB0_13
.LBB0_25:
	s_mov_b32 s43, 0
	s_lshl_b32 s6, s49, 3
	s_add_nc_u64 s[4:5], s[40:41], s[42:43]
	s_sub_co_i32 s6, s52, s6
	s_add_nc_u64 s[14:15], s[4:5], 0x1000
	s_max_i32 s5, s6, 0
	s_lshl_b32 s7, s49, 14
	s_and_b32 s18, s51, 0xffff
	s_lshl_b32 s6, s5, 16
	s_lshr_b32 s5, s5, 16
	s_add_co_i32 s13, s7, 0x22000
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s18, 0x200000
	s_addk_co_i32 s6, 0x7fff
	s_or_b32 s7, s5, 0x8000000
	s_mov_b32 s9, 0xe000
	s_mov_b32 s8, 8
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s43
	s_mov_b32 s11, s43
	s_mov_b32 s45, s43
	tensor_load_to_lds s[12:15], s[4:11]
	s_lshl_b64 s[6:7], s[46:47], 2
	s_lshl_b64 s[8:9], s[44:45], 2
	s_add_nc_u64 s[6:7], s[20:21], s[6:7]
	s_lshl_b32 s4, s49, 2
	s_add_nc_u64 s[6:7], s[6:7], s[8:9]
	s_sub_co_i32 s4, s53, s4
	s_lshl_b32 s8, s49, 10
	s_add_nc_u64 s[14:15], s[6:7], 0x200
	s_max_i32 s6, s4, 0
	s_add_co_i32 s13, s8, 0x32000
	s_lshl_b32 s7, s6, 16
	s_lshr_b32 s8, s6, 16
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s18, 0x220000
	s_or_b32 s6, s7, 0x7fff
	s_or_b32 s7, s8, 0x400000
	s_movk_i32 s9, 0x700
	s_mov_b32 s8, 4
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccnz .LBB0_14
.LBB0_26:
	s_add_co_i32 s18, s49, -2
	s_mov_b32 s11, 0
	s_mul_i32 s10, s18, 0x70000
	s_and_b32 s19, s50, 0xffff
	s_add_nc_u64 s[4:5], s[36:37], s[10:11]
	s_lshl_b32 s6, s18, 14
	s_add_nc_u64 s[14:15], s[4:5], 0x1000
	s_add_co_i32 s13, s6, 0x2a000
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s19, 0x200000
	s_mov_b32 s9, 0xe000
	s_mov_b32 s8, 8
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_lshl_b64 s[8:9], s[16:17], 2
	s_mul_i32 s10, s18, 0x1c00
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[8:9], s[34:35], s[8:9]
	s_lshl_b64 s[14:15], s[10:11], 2
	s_lshl_b32 s4, s18, 10
	s_add_nc_u64 s[8:9], s[8:9], s[14:15]
	s_add_co_i32 s13, s4, 0x32800
	s_add_nc_u64 s[14:15], s[8:9], 0x200
	s_or_b32 s4, s19, 0x220000
	s_bitset1_b32 s15, 31
	s_movk_i32 s9, 0x700
	s_mov_b32 s8, 4
	s_mov_b32 s7, 0x407fff
	s_mov_b32 s10, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_and_b32 vcc_lo, exec_lo, s1
	s_cbranch_vccnz .LBB0_15
.LBB0_27:
	s_mov_b32 s43, 0
	s_lshl_b32 s6, s49, 3
	s_add_nc_u64 s[4:5], s[40:41], s[42:43]
	s_sub_co_i32 s6, s52, s6
	s_add_nc_u64 s[14:15], s[4:5], 0x1800
	s_max_i32 s5, s6, 0
	s_lshl_b32 s7, s49, 14
	s_and_b32 s18, s51, 0xffff
	s_lshl_b32 s6, s5, 16
	s_lshr_b32 s5, s5, 16
	s_add_co_i32 s13, s7, 0x33000
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s18, 0x200000
	s_addk_co_i32 s6, 0x7fff
	s_or_b32 s7, s5, 0x8000000
	s_mov_b32 s9, 0xe000
	s_mov_b32 s8, 8
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s10, s43
	s_mov_b32 s11, s43
	s_mov_b32 s45, s43
	tensor_load_to_lds s[12:15], s[4:11]
	s_lshl_b64 s[6:7], s[46:47], 2
	s_lshl_b64 s[8:9], s[44:45], 2
	s_add_nc_u64 s[6:7], s[20:21], s[6:7]
	s_lshl_b32 s4, s49, 2
	s_add_nc_u64 s[6:7], s[6:7], s[8:9]
	s_sub_co_i32 s4, s53, s4
	s_lshl_b32 s8, s49, 10
	s_add_nc_u64 s[14:15], s[6:7], 0x300
	s_max_i32 s6, s4, 0
	s_add_co_i32 s13, s8, 0x43000
	s_lshl_b32 s7, s6, 16
	s_lshr_b32 s8, s6, 16
	s_bitset1_b32 s15, 31
	s_or_b32 s4, s18, 0x220000
	s_or_b32 s6, s7, 0x7fff
	s_or_b32 s7, s8, 0x400000
	s_movk_i32 s9, 0x700
	s_mov_b32 s8, 4
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[4:11]
	s_and_b32 vcc_lo, exec_lo, s0
	s_cbranch_vccz .LBB0_16
	s_branch .LBB0_17
.LBB0_28:
	s_wait_alu depctr_vm_vsrc(6)
	v_mov_b32_e32 v2, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v15, v2 :: v_dual_mov_b32 v16, v2
	v_dual_mov_b32 v17, v2 :: v_dual_mov_b32 v3, v2
	v_dual_mov_b32 v4, v2 :: v_dual_mov_b32 v5, v2
	v_dual_mov_b32 v6, v2 :: v_dual_mov_b32 v7, v2
	v_dual_mov_b32 v8, v2 :: v_dual_mov_b32 v9, v2
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v11, v2
	v_dual_mov_b32 v12, v2 :: v_dual_mov_b32 v13, v2
	v_mov_b32_e32 v14, v2
	s_set_vgpr_msb 0x80
	v_mov_b64_e32 v[208:209] /*v[720:721]*/, v[16:17]
	s_set_vgpr_msb 0x80c0
	v_mov_b64_e32 v[34:35] /*v[802:803]*/, v[16:17]
	s_set_vgpr_msb 0xc000
	v_mov_b64_e32 v[64:65], v[16:17]
	v_mov_b64_e32 v[80:81], v[16:17]
	v_mov_b64_e32 v[96:97], v[16:17]
	v_mov_b64_e32 v[112:113], v[16:17]
	v_mov_b64_e32 v[128:129], v[16:17]
	v_mov_b64_e32 v[144:145], v[16:17]
	v_mov_b64_e32 v[160:161], v[16:17]
	v_mov_b64_e32 v[176:177], v[16:17]
	v_mov_b64_e32 v[192:193], v[16:17]
	v_mov_b64_e32 v[208:209], v[16:17]
	v_mov_b64_e32 v[224:225], v[16:17]
	v_mov_b64_e32 v[240:241], v[16:17]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[0:1] /*v[256:257]*/, v[16:17]
	v_mov_b64_e32 v[16:17] /*v[272:273]*/, v[16:17]
	v_mov_b64_e32 v[32:33] /*v[288:289]*/, v[16:17]
	v_mov_b64_e32 v[48:49] /*v[304:305]*/, v[16:17]
	v_mov_b64_e32 v[64:65] /*v[320:321]*/, v[16:17]
	v_mov_b64_e32 v[80:81] /*v[336:337]*/, v[16:17]
	v_mov_b64_e32 v[96:97] /*v[352:353]*/, v[16:17]
	v_mov_b64_e32 v[112:113] /*v[368:369]*/, v[16:17]
	v_mov_b64_e32 v[128:129] /*v[384:385]*/, v[16:17]
	v_mov_b64_e32 v[144:145] /*v[400:401]*/, v[16:17]
	v_mov_b64_e32 v[160:161] /*v[416:417]*/, v[16:17]
	v_mov_b64_e32 v[176:177] /*v[432:433]*/, v[16:17]
	v_mov_b64_e32 v[192:193] /*v[448:449]*/, v[16:17]
	v_mov_b64_e32 v[208:209] /*v[464:465]*/, v[16:17]
	v_mov_b64_e32 v[224:225] /*v[480:481]*/, v[16:17]
	v_mov_b64_e32 v[240:241] /*v[496:497]*/, v[16:17]
	s_set_vgpr_msb 0x4080
	v_mov_b64_e32 v[0:1] /*v[512:513]*/, v[16:17]
	v_mov_b64_e32 v[206:207] /*v[718:719]*/, v[14:15]
	v_mov_b64_e32 v[204:205] /*v[716:717]*/, v[12:13]
	v_mov_b64_e32 v[202:203] /*v[714:715]*/, v[10:11]
	v_mov_b64_e32 v[200:201] /*v[712:713]*/, v[8:9]
	v_mov_b64_e32 v[198:199] /*v[710:711]*/, v[6:7]
	v_mov_b64_e32 v[196:197] /*v[708:709]*/, v[4:5]
	v_mov_b64_e32 v[194:195] /*v[706:707]*/, v[2:3]
	s_set_vgpr_msb 0x80c0
	v_mov_b64_e32 v[32:33] /*v[800:801]*/, v[14:15]
	v_mov_b64_e32 v[30:31] /*v[798:799]*/, v[12:13]
	v_mov_b64_e32 v[28:29] /*v[796:797]*/, v[10:11]
	v_mov_b64_e32 v[26:27] /*v[794:795]*/, v[8:9]
	v_mov_b64_e32 v[24:25] /*v[792:793]*/, v[6:7]
	v_mov_b64_e32 v[22:23] /*v[790:791]*/, v[4:5]
	v_mov_b64_e32 v[20:21] /*v[788:789]*/, v[2:3]
	s_set_vgpr_msb 0xc000
	v_mov_b64_e32 v[62:63], v[14:15]
	v_mov_b64_e32 v[60:61], v[12:13]
	v_mov_b64_e32 v[58:59], v[10:11]
	v_mov_b64_e32 v[56:57], v[8:9]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[78:79], v[14:15]
	v_mov_b64_e32 v[76:77], v[12:13]
	v_mov_b64_e32 v[74:75], v[10:11]
	v_mov_b64_e32 v[72:73], v[8:9]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[94:95], v[14:15]
	v_mov_b64_e32 v[92:93], v[12:13]
	v_mov_b64_e32 v[90:91], v[10:11]
	v_mov_b64_e32 v[88:89], v[8:9]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[110:111], v[14:15]
	v_mov_b64_e32 v[108:109], v[12:13]
	v_mov_b64_e32 v[106:107], v[10:11]
	v_mov_b64_e32 v[104:105], v[8:9]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[126:127], v[14:15]
	v_mov_b64_e32 v[124:125], v[12:13]
	v_mov_b64_e32 v[122:123], v[10:11]
	v_mov_b64_e32 v[120:121], v[8:9]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[142:143], v[14:15]
	v_mov_b64_e32 v[140:141], v[12:13]
	v_mov_b64_e32 v[138:139], v[10:11]
	v_mov_b64_e32 v[136:137], v[8:9]
	v_mov_b64_e32 v[134:135], v[6:7]
	v_mov_b64_e32 v[132:133], v[4:5]
	v_mov_b64_e32 v[130:131], v[2:3]
	v_mov_b64_e32 v[158:159], v[14:15]
	v_mov_b64_e32 v[156:157], v[12:13]
	v_mov_b64_e32 v[154:155], v[10:11]
	v_mov_b64_e32 v[152:153], v[8:9]
	v_mov_b64_e32 v[150:151], v[6:7]
	v_mov_b64_e32 v[148:149], v[4:5]
	v_mov_b64_e32 v[146:147], v[2:3]
	v_mov_b64_e32 v[174:175], v[14:15]
	v_mov_b64_e32 v[172:173], v[12:13]
	v_mov_b64_e32 v[170:171], v[10:11]
	v_mov_b64_e32 v[168:169], v[8:9]
	v_mov_b64_e32 v[166:167], v[6:7]
	v_mov_b64_e32 v[164:165], v[4:5]
	v_mov_b64_e32 v[162:163], v[2:3]
	v_mov_b64_e32 v[190:191], v[14:15]
	v_mov_b64_e32 v[188:189], v[12:13]
	v_mov_b64_e32 v[186:187], v[10:11]
	v_mov_b64_e32 v[184:185], v[8:9]
	v_mov_b64_e32 v[182:183], v[6:7]
	v_mov_b64_e32 v[180:181], v[4:5]
	v_mov_b64_e32 v[178:179], v[2:3]
	v_mov_b64_e32 v[206:207], v[14:15]
	v_mov_b64_e32 v[204:205], v[12:13]
	v_mov_b64_e32 v[202:203], v[10:11]
	v_mov_b64_e32 v[200:201], v[8:9]
	v_mov_b64_e32 v[198:199], v[6:7]
	v_mov_b64_e32 v[196:197], v[4:5]
	v_mov_b64_e32 v[194:195], v[2:3]
	v_mov_b64_e32 v[222:223], v[14:15]
	v_mov_b64_e32 v[220:221], v[12:13]
	v_mov_b64_e32 v[218:219], v[10:11]
	v_mov_b64_e32 v[216:217], v[8:9]
	v_mov_b64_e32 v[214:215], v[6:7]
	v_mov_b64_e32 v[212:213], v[4:5]
	v_mov_b64_e32 v[210:211], v[2:3]
	v_mov_b64_e32 v[238:239], v[14:15]
	v_mov_b64_e32 v[236:237], v[12:13]
	v_mov_b64_e32 v[234:235], v[10:11]
	v_mov_b64_e32 v[232:233], v[8:9]
	v_mov_b64_e32 v[230:231], v[6:7]
	v_mov_b64_e32 v[228:229], v[4:5]
	v_mov_b64_e32 v[226:227], v[2:3]
	v_mov_b64_e32 v[254:255], v[14:15]
	v_mov_b64_e32 v[252:253], v[12:13]
	v_mov_b64_e32 v[250:251], v[10:11]
	v_mov_b64_e32 v[248:249], v[8:9]
	v_mov_b64_e32 v[246:247], v[6:7]
	v_mov_b64_e32 v[244:245], v[4:5]
	v_mov_b64_e32 v[242:243], v[2:3]
	s_set_vgpr_msb 64
	v_mov_b64_e32 v[14:15] /*v[270:271]*/, v[14:15]
	v_mov_b64_e32 v[12:13] /*v[268:269]*/, v[12:13]
	v_mov_b64_e32 v[10:11] /*v[266:267]*/, v[10:11]
	v_mov_b64_e32 v[8:9] /*v[264:265]*/, v[8:9]
	v_mov_b64_e32 v[6:7] /*v[262:263]*/, v[6:7]
	v_mov_b64_e32 v[4:5] /*v[260:261]*/, v[4:5]
	v_mov_b64_e32 v[2:3] /*v[258:259]*/, v[2:3]
	v_mov_b64_e32 v[30:31] /*v[286:287]*/, v[14:15]
	v_mov_b64_e32 v[28:29] /*v[284:285]*/, v[12:13]
	v_mov_b64_e32 v[26:27] /*v[282:283]*/, v[10:11]
	v_mov_b64_e32 v[24:25] /*v[280:281]*/, v[8:9]
	v_mov_b64_e32 v[22:23] /*v[278:279]*/, v[6:7]
	v_mov_b64_e32 v[20:21] /*v[276:277]*/, v[4:5]
	v_mov_b64_e32 v[18:19] /*v[274:275]*/, v[2:3]
	v_mov_b64_e32 v[46:47] /*v[302:303]*/, v[14:15]
	v_mov_b64_e32 v[44:45] /*v[300:301]*/, v[12:13]
	v_mov_b64_e32 v[42:43] /*v[298:299]*/, v[10:11]
	v_mov_b64_e32 v[40:41] /*v[296:297]*/, v[8:9]
	v_mov_b64_e32 v[38:39] /*v[294:295]*/, v[6:7]
	v_mov_b64_e32 v[36:37] /*v[292:293]*/, v[4:5]
	v_mov_b64_e32 v[34:35] /*v[290:291]*/, v[2:3]
	v_mov_b64_e32 v[62:63] /*v[318:319]*/, v[14:15]
	v_mov_b64_e32 v[60:61] /*v[316:317]*/, v[12:13]
	v_mov_b64_e32 v[58:59] /*v[314:315]*/, v[10:11]
	v_mov_b64_e32 v[56:57] /*v[312:313]*/, v[8:9]
	v_mov_b64_e32 v[54:55] /*v[310:311]*/, v[6:7]
	v_mov_b64_e32 v[52:53] /*v[308:309]*/, v[4:5]
	v_mov_b64_e32 v[50:51] /*v[306:307]*/, v[2:3]
	v_mov_b64_e32 v[78:79] /*v[334:335]*/, v[14:15]
	v_mov_b64_e32 v[76:77] /*v[332:333]*/, v[12:13]
	v_mov_b64_e32 v[74:75] /*v[330:331]*/, v[10:11]
	v_mov_b64_e32 v[72:73] /*v[328:329]*/, v[8:9]
	v_mov_b64_e32 v[70:71] /*v[326:327]*/, v[6:7]
	v_mov_b64_e32 v[68:69] /*v[324:325]*/, v[4:5]
	v_mov_b64_e32 v[66:67] /*v[322:323]*/, v[2:3]
	v_mov_b64_e32 v[94:95] /*v[350:351]*/, v[14:15]
	v_mov_b64_e32 v[92:93] /*v[348:349]*/, v[12:13]
	v_mov_b64_e32 v[90:91] /*v[346:347]*/, v[10:11]
	v_mov_b64_e32 v[88:89] /*v[344:345]*/, v[8:9]
	v_mov_b64_e32 v[86:87] /*v[342:343]*/, v[6:7]
	v_mov_b64_e32 v[84:85] /*v[340:341]*/, v[4:5]
	v_mov_b64_e32 v[82:83] /*v[338:339]*/, v[2:3]
	v_mov_b64_e32 v[110:111] /*v[366:367]*/, v[14:15]
	v_mov_b64_e32 v[108:109] /*v[364:365]*/, v[12:13]
	v_mov_b64_e32 v[106:107] /*v[362:363]*/, v[10:11]
	v_mov_b64_e32 v[104:105] /*v[360:361]*/, v[8:9]
	v_mov_b64_e32 v[102:103] /*v[358:359]*/, v[6:7]
	v_mov_b64_e32 v[100:101] /*v[356:357]*/, v[4:5]
	v_mov_b64_e32 v[98:99] /*v[354:355]*/, v[2:3]
	v_mov_b64_e32 v[126:127] /*v[382:383]*/, v[14:15]
	v_mov_b64_e32 v[124:125] /*v[380:381]*/, v[12:13]
	v_mov_b64_e32 v[122:123] /*v[378:379]*/, v[10:11]
	v_mov_b64_e32 v[120:121] /*v[376:377]*/, v[8:9]
	v_mov_b64_e32 v[118:119] /*v[374:375]*/, v[6:7]
	v_mov_b64_e32 v[116:117] /*v[372:373]*/, v[4:5]
	v_mov_b64_e32 v[114:115] /*v[370:371]*/, v[2:3]
	v_mov_b64_e32 v[142:143] /*v[398:399]*/, v[14:15]
	v_mov_b64_e32 v[140:141] /*v[396:397]*/, v[12:13]
	v_mov_b64_e32 v[138:139] /*v[394:395]*/, v[10:11]
	v_mov_b64_e32 v[136:137] /*v[392:393]*/, v[8:9]
	v_mov_b64_e32 v[134:135] /*v[390:391]*/, v[6:7]
	v_mov_b64_e32 v[132:133] /*v[388:389]*/, v[4:5]
	v_mov_b64_e32 v[130:131] /*v[386:387]*/, v[2:3]
	v_mov_b64_e32 v[158:159] /*v[414:415]*/, v[14:15]
	v_mov_b64_e32 v[156:157] /*v[412:413]*/, v[12:13]
	v_mov_b64_e32 v[154:155] /*v[410:411]*/, v[10:11]
	v_mov_b64_e32 v[152:153] /*v[408:409]*/, v[8:9]
	v_mov_b64_e32 v[150:151] /*v[406:407]*/, v[6:7]
	v_mov_b64_e32 v[148:149] /*v[404:405]*/, v[4:5]
	v_mov_b64_e32 v[146:147] /*v[402:403]*/, v[2:3]
	v_mov_b64_e32 v[174:175] /*v[430:431]*/, v[14:15]
	v_mov_b64_e32 v[172:173] /*v[428:429]*/, v[12:13]
	v_mov_b64_e32 v[170:171] /*v[426:427]*/, v[10:11]
	v_mov_b64_e32 v[168:169] /*v[424:425]*/, v[8:9]
	v_mov_b64_e32 v[166:167] /*v[422:423]*/, v[6:7]
	v_mov_b64_e32 v[164:165] /*v[420:421]*/, v[4:5]
	v_mov_b64_e32 v[162:163] /*v[418:419]*/, v[2:3]
	v_mov_b64_e32 v[190:191] /*v[446:447]*/, v[14:15]
	v_mov_b64_e32 v[188:189] /*v[444:445]*/, v[12:13]
	v_mov_b64_e32 v[186:187] /*v[442:443]*/, v[10:11]
	v_mov_b64_e32 v[184:185] /*v[440:441]*/, v[8:9]
	v_mov_b64_e32 v[182:183] /*v[438:439]*/, v[6:7]
	v_mov_b64_e32 v[180:181] /*v[436:437]*/, v[4:5]
	v_mov_b64_e32 v[178:179] /*v[434:435]*/, v[2:3]
	v_mov_b64_e32 v[206:207] /*v[462:463]*/, v[14:15]
	v_mov_b64_e32 v[204:205] /*v[460:461]*/, v[12:13]
	v_mov_b64_e32 v[202:203] /*v[458:459]*/, v[10:11]
	v_mov_b64_e32 v[200:201] /*v[456:457]*/, v[8:9]
	v_mov_b64_e32 v[198:199] /*v[454:455]*/, v[6:7]
	v_mov_b64_e32 v[196:197] /*v[452:453]*/, v[4:5]
	v_mov_b64_e32 v[194:195] /*v[450:451]*/, v[2:3]
	v_mov_b64_e32 v[222:223] /*v[478:479]*/, v[14:15]
	v_mov_b64_e32 v[220:221] /*v[476:477]*/, v[12:13]
	v_mov_b64_e32 v[218:219] /*v[474:475]*/, v[10:11]
	v_mov_b64_e32 v[216:217] /*v[472:473]*/, v[8:9]
	v_mov_b64_e32 v[214:215] /*v[470:471]*/, v[6:7]
	v_mov_b64_e32 v[212:213] /*v[468:469]*/, v[4:5]
	v_mov_b64_e32 v[210:211] /*v[466:467]*/, v[2:3]
	v_mov_b64_e32 v[238:239] /*v[494:495]*/, v[14:15]
	v_mov_b64_e32 v[236:237] /*v[492:493]*/, v[12:13]
	v_mov_b64_e32 v[234:235] /*v[490:491]*/, v[10:11]
	v_mov_b64_e32 v[232:233] /*v[488:489]*/, v[8:9]
	v_mov_b64_e32 v[230:231] /*v[486:487]*/, v[6:7]
	v_mov_b64_e32 v[228:229] /*v[484:485]*/, v[4:5]
	v_mov_b64_e32 v[226:227] /*v[482:483]*/, v[2:3]
	v_mov_b64_e32 v[254:255] /*v[510:511]*/, v[14:15]
	v_mov_b64_e32 v[252:253] /*v[508:509]*/, v[12:13]
	v_mov_b64_e32 v[250:251] /*v[506:507]*/, v[10:11]
	v_mov_b64_e32 v[248:249] /*v[504:505]*/, v[8:9]
	v_mov_b64_e32 v[246:247] /*v[502:503]*/, v[6:7]
	v_mov_b64_e32 v[244:245] /*v[500:501]*/, v[4:5]
	v_mov_b64_e32 v[242:243] /*v[498:499]*/, v[2:3]
	s_set_vgpr_msb 0x4000
.LBB0_29:
	s_set_vgpr_msb 0x80
	v_or_b32_e32 v216 /*v728*/, 0x8200, v32
	s_set_vgpr_msb 0x8088
	v_or_b32_e32 v215 /*v727*/, 0x200, v213 /*v725*/
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s20, 1
	s_set_vgpr_msb 0x8800
	s_cbranch_vccnz .LBB0_36
	s_add_co_i32 s5, s49, -2
	s_mov_b32 s11, 0
	s_mul_i32 s0, s5, 0x1c00
	s_mov_b32 s1, s11
	s_mul_u64 s[6:7], s[38:39], 0x1c00
	s_mul_i32 s10, s5, 0x70000
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[6:7], s[34:35], s[6:7]
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b32 s40, s5, 14
	s_lshl_b32 s5, s5, 10
	s_add_nc_u64 s[0:1], s[6:7], s[0:1]
	s_add_nc_u64 s[6:7], s[36:37], s[10:11]
	s_and_b32 s8, s50, 0xffff
	s_add_co_i32 s38, s5, 0x10800
	s_add_nc_u64 s[22:23], s[6:7], 0x2000
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_or_b32 s4, s8, 0x200000
	s_or_b32 s12, s8, 0x220000
	s_add_nc_u64 s[0:1], s[0:1], 0x400
	s_mov_b32 s9, 0xe000
	s_mov_b32 s8, 8
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s10, s11
	s_movk_i32 s17, 0x700
	s_mov_b32 s16, 4
	s_mov_b32 s15, 0x407fff
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s18, s11
	s_mov_b32 s19, s11
	s_mov_b32 s34, s11
	s_branch .LBB0_33
.LBB0_31:
	s_barrier_wait -3
.LBB0_32:
	s_add_nc_u64 s[0:1], s[0:1], 0x100
	s_cmp_eq_u32 s34, 24
	s_add_nc_u64 s[22:23], s[22:23], 0x800
	s_cbranch_scc1 .LBB0_36
.LBB0_33:
	s_and_b32 s36, s34, 3
	s_add_co_i32 s34, s34, 1
	s_mul_i32 s36, s36, 0x11000
	s_and_b32 s35, s34, 3
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 8
	v_add_nc_u32_e32 v1, s36, v213 /*v725*/
	s_mul_i32 s21, s35, 0x11000
	v_dual_add_nc_u32 v34, s36, v214 /*v726*/ :: v_dual_add_nc_u32 v28, s36, v211 /*v723*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v29, s36, v212 /*v724*/ :: v_dual_add_nc_u32 v35, s21, v213 /*v725*/
	v_dual_add_nc_u32 v36, s21, v214 /*v726*/ :: v_dual_add_nc_u32 v37, s21, v211 /*v723*/
	v_add_nc_u32_e32 v38, s21, v212 /*v724*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v18, v24
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v19, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v22, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v23, v24 matrix_a_reuse
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[186:189] /*v[698:701]*/, v1 offset:1024
	ds_load_b128 v[190:193] /*v[702:705]*/, v1 offset:1536
	s_set_vgpr_msb 0x8000
	ds_load_2addr_b32 v[26:27], v29 offset0:32 offset1:96
	ds_load_2addr_b32 v[30:31], v29 offset0:160 offset1:224
	ds_load_2addr_b32 v[32:33], v28 offset0:32 offset1:96
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_2addr_b32 v[28:29], v28 offset0:160 offset1:224
	s_set_vgpr_msb 0x80
	ds_load_b128 v[66:69] /*v[578:581]*/, v34 offset:1024
	ds_load_b128 v[70:73] /*v[582:585]*/, v34 offset:1536
	ds_load_b128 v[74:77] /*v[586:589]*/, v34 offset:3072
	ds_load_b128 v[78:81] /*v[590:593]*/, v34 offset:3584
	ds_load_b128 v[82:85] /*v[594:597]*/, v34 offset:5120
	ds_load_b128 v[86:89] /*v[598:601]*/, v34 offset:5632
	ds_load_b128 v[90:93] /*v[602:605]*/, v34 offset:7168
	ds_load_b128 v[94:97] /*v[606:609]*/, v34 offset:7680
	ds_load_b128 v[98:101] /*v[610:613]*/, v34 offset:9216
	ds_load_b128 v[102:105] /*v[614:617]*/, v34 offset:9728
	ds_load_b128 v[106:109] /*v[618:621]*/, v34 offset:11264
	ds_load_b128 v[110:113] /*v[622:625]*/, v34 offset:11776
	ds_load_b128 v[114:117] /*v[626:629]*/, v34 offset:13312
	ds_load_b128 v[118:121] /*v[630:633]*/, v34 offset:13824
	ds_load_b128 v[122:125] /*v[634:637]*/, v34 offset:15360
	ds_load_b128 v[126:129] /*v[638:641]*/, v34 offset:15872
	ds_load_b128 v[218:221] /*v[730:733]*/, v1 offset:3072
	ds_load_b128 v[222:225] /*v[734:737]*/, v1 offset:3584
	ds_load_b128 v[226:229] /*v[738:741]*/, v1 offset:5120
	ds_load_b128 v[230:233] /*v[742:745]*/, v1 offset:5632
	ds_load_b128 v[234:237] /*v[746:749]*/, v1 offset:7168
	ds_load_b128 v[238:241] /*v[750:753]*/, v1 offset:7680
	ds_load_b128 v[242:245] /*v[754:757]*/, v1 offset:9216
	ds_load_b128 v[246:249] /*v[758:761]*/, v1 offset:9728
	ds_load_b128 v[250:253] /*v[762:765]*/, v1 offset:11264
	ds_load_b128 v[254:257] /*v[766:769]*/, v1 offset:11776
	s_set_vgpr_msb 0x80c0
	ds_load_b128 v[2:5] /*v[770:773]*/, v1 offset:13312
	ds_load_b128 v[6:9] /*v[774:777]*/, v1 offset:13824
	ds_load_b128 v[10:13] /*v[778:781]*/, v1 offset:15360
	ds_load_b128 v[14:17] /*v[782:785]*/, v1 offset:15872
	s_set_vgpr_msb 0xc05a
	s_wait_dscnt 0x30
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v23, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v22, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v19, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v18, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0x2e
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[114:129] /*v[370:385]*/, v18, v25 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[98:113] /*v[354:369]*/, v19, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[82:97] /*v[338:353]*/, v22, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[66:81] /*v[322:337]*/, v23, v25 matrix_a_reuse
	s_wait_dscnt 0x2c
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[2:17] /*v[258:273]*/, v23, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[18:33] /*v[274:289]*/, v22, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[34:49] /*v[290:305]*/, v19, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[50:65] /*v[306:321]*/, v18, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a0a
	s_wait_dscnt 0x2a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[242:257], v18, v20 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[226:241], v19, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[210:225], v22, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[194:209], v23, v20 matrix_a_reuse
	s_wait_dscnt 0x28
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[130:145], v23, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[146:161], v22, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[162:177], v19, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[178:193], v18, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0x26
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129], v18, v21 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113], v19, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97], v22, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81], v23, v21 matrix_a_reuse
	s_wait_dscnt 0x24
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[2:17], v23, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[194:209] /*v[706:721]*/, v22, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaafa
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[20:35] /*v[788:803]*/, v19, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[50:65], v18, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_add_co_i32 s21, s40, s36
	s_or_b32 s37, s23, 0x80000000
	s_add_co_i32 s21, s21, 0x8000
	s_mov_b64 s[54:55], s[22:23]
	s_wait_tensorcnt 0x4
	s_mov_b64 s[52:53], s[20:21]
	s_mov_b32 s55, s37
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	tensor_load_to_lds s[52:55], s[4:11]
	s_add_co_i32 s21, s38, s36
	s_or_b32 s36, s1, 0x80000000
	s_mov_b64 s[54:55], s[22:23]
	s_mov_b64 s[52:53], s[20:21]
	s_mov_b32 s54, s0
	s_mov_b32 s55, s36
	s_set_vgpr_msb 0xa5a
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v26, v32
	tensor_load_to_lds s[52:55], s[12:19]
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v27, v32 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v30, v32 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[114:129] /*v[626:641]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v31, v32 matrix_a_reuse
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_stride64_b32 v[18:19], v38 offset1:1
	ds_load_2addr_stride64_b32 v[22:23], v38 offset0:2 offset1:3
	ds_load_2addr_stride64_b32 v[24:25], v37 offset1:1
	ds_load_2addr_stride64_b32 v[20:21], v37 offset0:2 offset1:3
	s_set_vgpr_msb 0x80
	ds_load_b128 v[2:5] /*v[514:517]*/, v36
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[218:225] /*v[730:737]*/, v[130:145] /*v[386:401]*/, v31, v32 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[218:225] /*v[730:737]*/, v[146:161] /*v[402:417]*/, v30, v32 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[218:225] /*v[730:737]*/, v[162:177] /*v[418:433]*/, v27, v32 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[218:225] /*v[730:737]*/, v[178:193] /*v[434:449]*/, v26, v32 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[6:9] /*v[518:521]*/, v36 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v36 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v36 offset:2560
	ds_load_b128 v[18:21] /*v[530:533]*/, v36 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v36 offset:4608
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[226:233] /*v[738:745]*/, v[114:129] /*v[370:385]*/, v26, v33 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[226:233] /*v[738:745]*/, v[98:113] /*v[354:369]*/, v27, v33 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[226:233] /*v[738:745]*/, v[82:97] /*v[338:353]*/, v30, v33 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[226:233] /*v[738:745]*/, v[66:81] /*v[322:337]*/, v31, v33 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[26:29] /*v[538:541]*/, v36 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v36 offset:6656
	ds_load_b128 v[34:37] /*v[546:549]*/, v36 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v36 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v36 offset:10240
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[234:241] /*v[746:753]*/, v[2:17] /*v[258:273]*/, v31, v33 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[234:241] /*v[746:753]*/, v[18:33] /*v[274:289]*/, v30, v33 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[234:241] /*v[746:753]*/, v[34:49] /*v[290:305]*/, v27, v33 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[234:241] /*v[746:753]*/, v[50:65] /*v[306:321]*/, v26, v33 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[46:49] /*v[558:561]*/, v36 offset:10752
	ds_load_b128 v[50:53] /*v[562:565]*/, v36 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v36 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v36 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v36 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[242:249] /*v[754:761]*/, v[242:257], v26, v28 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[242:249] /*v[754:761]*/, v[226:241], v27, v28 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[242:249] /*v[754:761]*/, v[210:225], v30, v28 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[242:249] /*v[754:761]*/, v[194:209], v31, v28 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[186:189] /*v[698:701]*/, v35
	ds_load_b128 v[190:193] /*v[702:705]*/, v35 offset:512
	ds_load_b128 v[178:181] /*v[690:693]*/, v35 offset:2048
	ds_load_b128 v[182:185] /*v[694:697]*/, v35 offset:2560
	ds_load_b128 v[170:173] /*v[682:685]*/, v35 offset:4096
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[250:257] /*v[762:769]*/, v[130:145], v31, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[250:257] /*v[762:769]*/, v[146:161], v30, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[250:257] /*v[762:769]*/, v[162:177], v27, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[250:257] /*v[762:769]*/, v[178:193], v26, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[174:177] /*v[686:689]*/, v35 offset:4608
	ds_load_b128 v[162:165] /*v[674:677]*/, v35 offset:6144
	ds_load_b128 v[166:169] /*v[678:681]*/, v35 offset:6656
	ds_load_b128 v[154:157] /*v[666:669]*/, v35 offset:8192
	ds_load_b128 v[158:161] /*v[670:673]*/, v35 offset:8704
	s_set_vgpr_msb 0x800e
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[2:9] /*v[770:777]*/, v[114:129], v26, v29 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[2:9] /*v[770:777]*/, v[98:113], v27, v29 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[2:9] /*v[770:777]*/, v[82:97], v30, v29 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[114:129] /*v[626:641]*/, v[2:9] /*v[770:777]*/, v[66:81], v31, v29 matrix_a_reuse
	s_set_vgpr_msb 0xe80
	ds_load_b128 v[146:149] /*v[658:661]*/, v35 offset:10240
	ds_load_b128 v[150:153] /*v[662:665]*/, v35 offset:10752
	ds_load_b128 v[138:141] /*v[650:653]*/, v35 offset:12288
	ds_load_b128 v[142:145] /*v[654:657]*/, v35 offset:12800
	ds_load_b128 v[130:133] /*v[642:645]*/, v35 offset:14336
	s_set_vgpr_msb 0x800e
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[114:129] /*v[626:641]*/, v[10:17] /*v[778:785]*/, v[2:17], v31, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xeae
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[98:113] /*v[610:625]*/, v[10:17] /*v[778:785]*/, v[194:209] /*v[706:721]*/, v30, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaefe
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[82:97] /*v[594:609]*/, v[10:17] /*v[778:785]*/, v[20:35] /*v[788:803]*/, v27, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfe0e
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[66:81] /*v[578:593]*/, v[10:17] /*v[778:785]*/, v[50:65], v26, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xe80
	ds_load_b128 v[134:137] /*v[646:649]*/, v35 offset:14848
	s_cmp_eq_u32 s35, 0
	s_cselect_b32 s21, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b32 s21, s48, s21
	s_and_b32 vcc_lo, exec_lo, s21
	s_set_vgpr_msb 0x8000
	s_cbranch_vccz .LBB0_32
	s_and_not1_b32 vcc_lo, exec_lo, s47
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_31
	s_barrier_signal -3
	s_branch .LBB0_31
.LBB0_36:
	s_mov_b32 s0, 0x11000
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_vm_vsrc(6)
	s_set_vgpr_msb 8
	v_dual_add_nc_u32 v1, s0, v213 /*v725*/ :: v_dual_add_nc_u32 v34, s0, v214 /*v726*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v35, s0, v211 /*v723*/ :: v_dual_add_nc_u32 v36, s0, v212 /*v724*/
	s_lshl_b32 s0, s33, 7
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v18, v24
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v19, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v22, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v23, v24 matrix_a_reuse
	s_set_vgpr_msb 0x5a02
	ds_load_2addr_b32 v[26:27], v212 /*v724*/ offset0:32 offset1:96
	ds_load_2addr_b32 v[30:31], v212 /*v724*/ offset0:160 offset1:224
	ds_load_2addr_b32 v[32:33], v211 /*v723*/ offset0:32 offset1:96
	ds_load_2addr_b32 v[28:29], v211 /*v723*/ offset0:160 offset1:224
	s_set_vgpr_msb 0x282
	ds_load_b128 v[66:69] /*v[578:581]*/, v214 /*v726*/ offset:1024
	ds_load_b128 v[70:73] /*v[582:585]*/, v214 /*v726*/ offset:1536
	ds_load_b128 v[74:77] /*v[586:589]*/, v214 /*v726*/ offset:3072
	ds_load_b128 v[78:81] /*v[590:593]*/, v214 /*v726*/ offset:3584
	ds_load_b128 v[82:85] /*v[594:597]*/, v214 /*v726*/ offset:5120
	ds_load_b128 v[86:89] /*v[598:601]*/, v214 /*v726*/ offset:5632
	ds_load_b128 v[90:93] /*v[602:605]*/, v214 /*v726*/ offset:7168
	ds_load_b128 v[94:97] /*v[606:609]*/, v214 /*v726*/ offset:7680
	ds_load_b128 v[98:101] /*v[610:613]*/, v214 /*v726*/ offset:9216
	ds_load_b128 v[102:105] /*v[614:617]*/, v214 /*v726*/ offset:9728
	ds_load_b128 v[106:109] /*v[618:621]*/, v214 /*v726*/ offset:11264
	ds_load_b128 v[110:113] /*v[622:625]*/, v214 /*v726*/ offset:11776
	ds_load_b128 v[114:117] /*v[626:629]*/, v214 /*v726*/ offset:13312
	ds_load_b128 v[118:121] /*v[630:633]*/, v214 /*v726*/ offset:13824
	ds_load_b128 v[122:125] /*v[634:637]*/, v214 /*v726*/ offset:15360
	ds_load_b128 v[126:129] /*v[638:641]*/, v214 /*v726*/ offset:15872
	ds_load_b128 v[186:189] /*v[698:701]*/, v213 /*v725*/ offset:1024
	ds_load_b128 v[190:193] /*v[702:705]*/, v213 /*v725*/ offset:1536
	ds_load_b128 v[218:221] /*v[730:733]*/, v213 /*v725*/ offset:3072
	ds_load_b128 v[222:225] /*v[734:737]*/, v213 /*v725*/ offset:3584
	ds_load_b128 v[226:229] /*v[738:741]*/, v213 /*v725*/ offset:5120
	ds_load_b128 v[230:233] /*v[742:745]*/, v213 /*v725*/ offset:5632
	ds_load_b128 v[234:237] /*v[746:749]*/, v213 /*v725*/ offset:7168
	ds_load_b128 v[238:241] /*v[750:753]*/, v213 /*v725*/ offset:7680
	ds_load_b128 v[242:245] /*v[754:757]*/, v213 /*v725*/ offset:9216
	ds_load_b128 v[246:249] /*v[758:761]*/, v213 /*v725*/ offset:9728
	ds_load_b128 v[250:253] /*v[762:765]*/, v213 /*v725*/ offset:11264
	ds_load_b128 v[254:257] /*v[766:769]*/, v213 /*v725*/ offset:11776
	s_set_vgpr_msb 0x82c2
	ds_load_b128 v[2:5] /*v[770:773]*/, v213 /*v725*/ offset:13312
	ds_load_b128 v[6:9] /*v[774:777]*/, v213 /*v725*/ offset:13824
	ds_load_b128 v[10:13] /*v[778:781]*/, v213 /*v725*/ offset:15360
	ds_load_b128 v[14:17] /*v[782:785]*/, v213 /*v725*/ offset:15872
	s_set_vgpr_msb 0xc20a
	s_wait_dscnt 0x2a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[242:257], v18, v20 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[226:241], v19, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[210:225], v22, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[194:209], v23, v20 matrix_a_reuse
	s_wait_dscnt 0x28
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[130:145], v23, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[146:161], v22, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[162:177], v19, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[178:193], v18, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0x26
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129], v18, v21 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113], v19, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97], v22, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81], v23, v21 matrix_a_reuse
	s_wait_dscnt 0x24
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[2:17], v23, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[194:209] /*v[706:721]*/, v22, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaafa
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[20:35] /*v[788:803]*/, v19, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[50:65], v18, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa5a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[130:145] /*v[386:401]*/, v23, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[146:161] /*v[402:417]*/, v22, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[162:177] /*v[418:433]*/, v19, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[178:193] /*v[434:449]*/, v18, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[114:129] /*v[370:385]*/, v18, v25 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[98:113] /*v[354:369]*/, v19, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[82:97] /*v[338:353]*/, v22, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[66:81] /*v[322:337]*/, v23, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[2:17] /*v[258:273]*/, v23, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[18:33] /*v[274:289]*/, v22, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[34:49] /*v[290:305]*/, v19, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[50:65] /*v[306:321]*/, v18, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[242:257] /*v[498:513]*/, v26, v32
	s_wait_tensorcnt 0x4
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[226:241] /*v[482:497]*/, v27, v32 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[210:225] /*v[466:481]*/, v30, v32 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[114:129] /*v[626:641]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[450:465]*/, v31, v32 matrix_a_reuse
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5a00
	ds_load_2addr_stride64_b32 v[18:19], v36 offset1:1
	ds_load_2addr_stride64_b32 v[22:23], v36 offset0:2 offset1:3
	ds_load_2addr_stride64_b32 v[24:25], v35 offset1:1
	ds_load_2addr_stride64_b32 v[20:21], v35 offset0:2 offset1:3
	s_set_vgpr_msb 0x80
	ds_load_b128 v[2:5] /*v[514:517]*/, v34
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[218:225] /*v[730:737]*/, v[130:145] /*v[386:401]*/, v31, v32 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[218:225] /*v[730:737]*/, v[146:161] /*v[402:417]*/, v30, v32 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[218:225] /*v[730:737]*/, v[162:177] /*v[418:433]*/, v27, v32 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[218:225] /*v[730:737]*/, v[178:193] /*v[434:449]*/, v26, v32 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[6:9] /*v[518:521]*/, v34 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v34 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v34 offset:2560
	ds_load_b128 v[18:21] /*v[530:533]*/, v34 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v34 offset:4608
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[226:233] /*v[738:745]*/, v[114:129] /*v[370:385]*/, v26, v33 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[226:233] /*v[738:745]*/, v[98:113] /*v[354:369]*/, v27, v33 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[226:233] /*v[738:745]*/, v[82:97] /*v[338:353]*/, v30, v33 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[226:233] /*v[738:745]*/, v[66:81] /*v[322:337]*/, v31, v33 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[26:29] /*v[538:541]*/, v34 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v34 offset:6656
	ds_load_b128 v[34:37] /*v[546:549]*/, v34 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v34 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v34 offset:10240
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[234:241] /*v[746:753]*/, v[2:17] /*v[258:273]*/, v31, v33 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[234:241] /*v[746:753]*/, v[18:33] /*v[274:289]*/, v30, v33 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[234:241] /*v[746:753]*/, v[34:49] /*v[290:305]*/, v27, v33 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[234:241] /*v[746:753]*/, v[50:65] /*v[306:321]*/, v26, v33 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[46:49] /*v[558:561]*/, v34 offset:10752
	ds_load_b128 v[50:53] /*v[562:565]*/, v34 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v34 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v34 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v34 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[242:249] /*v[754:761]*/, v[242:257], v26, v28 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[242:249] /*v[754:761]*/, v[226:241], v27, v28 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[242:249] /*v[754:761]*/, v[210:225], v30, v28 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[242:249] /*v[754:761]*/, v[194:209], v31, v28 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[130:133] /*v[642:645]*/, v1
	ds_load_b128 v[134:137] /*v[646:649]*/, v1 offset:512
	ds_load_b128 v[138:141] /*v[650:653]*/, v1 offset:2048
	ds_load_b128 v[142:145] /*v[654:657]*/, v1 offset:2560
	ds_load_b128 v[146:149] /*v[658:661]*/, v1 offset:4096
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[250:257] /*v[762:769]*/, v[130:145], v31, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[250:257] /*v[762:769]*/, v[146:161], v30, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[250:257] /*v[762:769]*/, v[162:177], v27, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[250:257] /*v[762:769]*/, v[178:193], v26, v28 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[150:153] /*v[662:665]*/, v1 offset:4608
	ds_load_b128 v[154:157] /*v[666:669]*/, v1 offset:6144
	ds_load_b128 v[158:161] /*v[670:673]*/, v1 offset:6656
	ds_load_b128 v[162:165] /*v[674:677]*/, v1 offset:8192
	ds_load_b128 v[166:169] /*v[678:681]*/, v1 offset:8704
	s_set_vgpr_msb 0x800e
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[2:9] /*v[770:777]*/, v[114:129], v26, v29 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[2:9] /*v[770:777]*/, v[98:113], v27, v29 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[2:9] /*v[770:777]*/, v[82:97], v30, v29 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[114:129] /*v[626:641]*/, v[2:9] /*v[770:777]*/, v[66:81], v31, v29 matrix_a_reuse
	s_set_vgpr_msb 0xe80
	ds_load_b128 v[170:173] /*v[682:685]*/, v1 offset:10240
	ds_load_b128 v[174:177] /*v[686:689]*/, v1 offset:10752
	ds_load_b128 v[178:181] /*v[690:693]*/, v1 offset:12288
	ds_load_b128 v[182:185] /*v[694:697]*/, v1 offset:12800
	ds_load_b128 v[186:189] /*v[698:701]*/, v1 offset:14336
	s_set_vgpr_msb 0x800e
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[114:129] /*v[626:641]*/, v[10:17] /*v[778:785]*/, v[2:17], v31, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xeae
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[98:113] /*v[610:625]*/, v[10:17] /*v[778:785]*/, v[194:209] /*v[706:721]*/, v30, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaefe
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[82:97] /*v[594:609]*/, v[10:17] /*v[778:785]*/, v[20:35] /*v[788:803]*/, v27, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfe0e
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[66:81] /*v[578:593]*/, v[10:17] /*v[778:785]*/, v[50:65], v26, v29 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xe80
	ds_load_b128 v[190:193] /*v[702:705]*/, v1 offset:14848
	s_mov_b32 s1, 0x22000
	;;#ASMSTART
	;;#ASMEND
	s_set_vgpr_msb 0x8008
	v_dual_add_nc_u32 v41, s1, v213 /*v725*/ :: v_dual_add_nc_u32 v42, s1, v214 /*v726*/
	v_dual_add_nc_u32 v43, s1, v211 /*v723*/ :: v_dual_add_nc_u32 v44, s1, v212 /*v724*/
	v_add_nc_u32_e32 v26, 0x11180, v212 /*v724*/
	v_add_nc_u32_e32 v27, 0x11280, v212 /*v724*/
	v_add_nc_u32_e32 v28, 0x11380, v212 /*v724*/
	v_add_nc_u32_e32 v29, 0x11080, v211 /*v723*/
	s_set_vgpr_msb 0x85a
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[242:257] /*v[498:513]*/, v18, v24
	s_set_vgpr_msb 0x5a08
	v_add_nc_u32_e32 v30, 0x11400, v216 /*v728*/
	v_add_nc_u32_e32 v45, 0x11a00, v215 /*v727*/
	v_add_nc_u32_e32 v46, 0x11c00, v215 /*v727*/
	v_add_nc_u32_e32 v47, 0x12200, v215 /*v727*/
	v_add_nc_u32_e32 v48, 0x12400, v215 /*v727*/
	v_add_nc_u32_e32 v49, 0x13a00, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v1, 0x11080, v212 /*v724*/
	s_set_vgpr_msb 0x85a
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[226:241] /*v[482:497]*/, v19, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[210:225] /*v[466:481]*/, v22, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[194:209] /*v[450:465]*/, v23, v24 matrix_a_reuse
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5a08
	ds_load_b32 v34, v26
	ds_load_b32 v35, v27
	ds_load_b32 v36, v28
	ds_load_b32 v37, v29
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v26, 0x11180, v211 /*v723*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v27, 0x11280, v211 /*v723*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v28, 0x11380, v211 /*v723*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v29, 0x11200, v216 /*v728*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[70:73] /*v[582:585]*/, v30
	s_wait_alu depctr_va_vdst(3)
	s_set_vgpr_msb 0x8000
	ds_load_b32 v38, v26
	s_wait_alu depctr_va_vdst(2)
	ds_load_b32 v39, v27
	s_wait_alu depctr_va_vdst(1)
	ds_load_b32 v40, v28
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x80
	ds_load_b128 v[66:69] /*v[578:581]*/, v29
	s_wait_alu depctr_vm_vsrc(3)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v26, 0x11a00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v27, 0x11c00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v28, 0x12200, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v29, 0x12400, v216 /*v728*/
	v_add_nc_u32_e32 v30, 0x12a00, v216 /*v728*/
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[74:77] /*v[586:589]*/, v26
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[78:81] /*v[590:593]*/, v27
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[82:85] /*v[594:597]*/, v28
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[86:89] /*v[598:601]*/, v29
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[90:93] /*v[602:605]*/, v30
	s_wait_alu depctr_vm_vsrc(4)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v26, 0x12c00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v27, 0x13200, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v28, 0x13400, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v29, 0x13a00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v30, 0x13c00, v216 /*v728*/
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[94:97] /*v[606:609]*/, v26
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[98:101] /*v[610:613]*/, v27
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[102:105] /*v[614:617]*/, v28
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[106:109] /*v[618:621]*/, v29
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[110:113] /*v[622:625]*/, v30
	s_wait_alu depctr_vm_vsrc(4)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v26, 0x14200, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v27, 0x14400, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v28, 0x14a00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v29, 0x14c00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v30, 0x11200, v215 /*v727*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[130:133] /*v[642:645]*/, v45
	ds_load_b128 v[134:137] /*v[646:649]*/, v46
	ds_load_b128 v[218:221] /*v[730:733]*/, v47
	ds_load_b128 v[222:225] /*v[734:737]*/, v48
	s_wait_alu depctr_vm_vsrc(3)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v45, 0x12a00, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v46, 0x12c00, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v47, 0x13200, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v48, 0x13400, v215 /*v727*/
	s_wait_alu depctr_va_vdst(8)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[114:117] /*v[626:629]*/, v26
	s_wait_alu depctr_va_vdst(7)
	ds_load_b128 v[118:121] /*v[630:633]*/, v27
	s_wait_alu depctr_va_vdst(6)
	ds_load_b128 v[122:125] /*v[634:637]*/, v28
	s_wait_alu depctr_va_vdst(5)
	ds_load_b128 v[126:129] /*v[638:641]*/, v29
	s_wait_alu depctr_va_vdst(4) depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8008
	ds_load_b128 v[26:29], v30
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v30, 0x11400, v215 /*v727*/
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[226:229] /*v[738:741]*/, v45
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[230:233] /*v[742:745]*/, v46
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[234:237] /*v[746:749]*/, v47
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[238:241] /*v[750:753]*/, v48
	ds_load_b128 v[242:245] /*v[754:757]*/, v49
	s_wait_alu depctr_vm_vsrc(4)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v45, 0x13c00, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v46, 0x14200, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v47, 0x14400, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v48, 0x14a00, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v49, 0x14c00, v215 /*v727*/
	ds_load_b32 v1, v1
	s_wait_alu depctr_va_vdst(5)
	ds_load_b128 v[30:33], v30
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[246:249] /*v[758:761]*/, v45
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[250:253] /*v[762:765]*/, v46
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[254:257] /*v[766:769]*/, v47
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x80c0
	ds_load_b128 v[2:5] /*v[770:773]*/, v48
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[6:9] /*v[774:777]*/, v49
	s_set_vgpr_msb 0xc00a
	s_wait_dscnt 0x2e
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[242:257], v18, v20 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[226:241], v19, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[210:225], v22, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[194:209], v23, v20 matrix_a_reuse
	s_wait_dscnt 0x2c
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[130:145], v23, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[146:161], v22, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[162:177], v19, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[178:193], v18, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0x2a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[114:129], v18, v21 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[98:113], v19, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[82:97], v22, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[66:81], v23, v21 matrix_a_reuse
	s_wait_dscnt 0x28
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[50:65] /*v[562:577]*/, v[186:193] /*v[698:705]*/, v[2:17], v23, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[34:49] /*v[546:561]*/, v[186:193] /*v[698:705]*/, v[194:209] /*v[706:721]*/, v22, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaafa
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[18:33] /*v[530:545]*/, v[186:193] /*v[698:705]*/, v[20:35] /*v[788:803]*/, v19, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[2:17] /*v[514:529]*/, v[186:193] /*v[698:705]*/, v[50:65], v18, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa5a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[130:145] /*v[386:401]*/, v23, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[146:161] /*v[402:417]*/, v22, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[162:177] /*v[418:433]*/, v19, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[178:193] /*v[434:449]*/, v18, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[114:129] /*v[370:385]*/, v18, v25 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[98:113] /*v[354:369]*/, v19, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[82:97] /*v[338:353]*/, v22, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[66:81] /*v[322:337]*/, v23, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[2:17] /*v[258:273]*/, v23, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[18:33] /*v[274:289]*/, v22, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[34:49] /*v[290:305]*/, v19, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[50:65] /*v[306:321]*/, v18, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a52
	s_wait_dscnt 0x5
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[66:81] /*v[578:593]*/, v[26:33], v[242:257] /*v[498:513]*/, v1, v37
	s_wait_tensorcnt 0x2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[82:97] /*v[594:609]*/, v[26:33], v[226:241] /*v[482:497]*/, v34, v37 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[98:113] /*v[610:625]*/, v[26:33], v[210:225] /*v[466:481]*/, v35, v37 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[114:129] /*v[626:641]*/, v[26:33], v[194:209] /*v[450:465]*/, v36, v37 matrix_a_reuse
	s_set_vgpr_msb 0x5200
	ds_load_2addr_stride64_b32 v[18:19], v44 offset1:1
	ds_load_2addr_stride64_b32 v[22:23], v44 offset0:2 offset1:3
	ds_load_2addr_stride64_b32 v[24:25], v43 offset1:1
	ds_load_2addr_stride64_b32 v[20:21], v43 offset0:2 offset1:3
	s_set_vgpr_msb 0x80
	ds_load_b128 v[2:5] /*v[514:517]*/, v42
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v36, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v35, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v34, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v1, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[6:9] /*v[518:521]*/, v42 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v42 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v42 offset:2560
	ds_load_b128 v[18:21] /*v[530:533]*/, v42 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v42 offset:4608
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[218:225] /*v[730:737]*/, v[114:129] /*v[370:385]*/, v1, v38 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[218:225] /*v[730:737]*/, v[98:113] /*v[354:369]*/, v34, v38 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[218:225] /*v[730:737]*/, v[82:97] /*v[338:353]*/, v35, v38 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[218:225] /*v[730:737]*/, v[66:81] /*v[322:337]*/, v36, v38 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[26:29] /*v[538:541]*/, v42 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v42 offset:6656
	ds_load_b128 v[34:37] /*v[546:549]*/, v42 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v42 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v42 offset:10240
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[226:233] /*v[738:745]*/, v[2:17] /*v[258:273]*/, v36, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[226:233] /*v[738:745]*/, v[18:33] /*v[274:289]*/, v35, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[226:233] /*v[738:745]*/, v[34:49] /*v[290:305]*/, v34, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[226:233] /*v[738:745]*/, v[50:65] /*v[306:321]*/, v1, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[46:49] /*v[558:561]*/, v42 offset:10752
	ds_load_b128 v[50:53] /*v[562:565]*/, v42 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v42 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v42 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v42 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[234:241] /*v[746:753]*/, v[242:257], v1, v39 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[234:241] /*v[746:753]*/, v[226:241], v34, v39 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[234:241] /*v[746:753]*/, v[210:225], v35, v39 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[234:241] /*v[746:753]*/, v[194:209], v36, v39 matrix_a_reuse
	s_set_vgpr_msb 0xa00
	ds_load_b128 v[26:29], v41
	s_wait_alu depctr_vm_vsrc(6)
	ds_load_b128 v[30:33], v41 offset:512
	s_set_vgpr_msb 0x80
	ds_load_b128 v[130:133] /*v[642:645]*/, v41 offset:2048
	ds_load_b128 v[134:137] /*v[646:649]*/, v41 offset:2560
	ds_load_b128 v[138:141] /*v[650:653]*/, v41 offset:4096
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[242:249] /*v[754:761]*/, v[130:145], v36, v39 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[242:249] /*v[754:761]*/, v[146:161], v35, v39 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[242:249] /*v[754:761]*/, v[162:177], v34, v39 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[242:249] /*v[754:761]*/, v[178:193], v1, v39 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[142:145] /*v[654:657]*/, v41 offset:4608
	ds_load_b128 v[146:149] /*v[658:661]*/, v41 offset:6144
	ds_load_b128 v[150:153] /*v[662:665]*/, v41 offset:6656
	ds_load_b128 v[154:157] /*v[666:669]*/, v41 offset:8192
	ds_load_b128 v[158:161] /*v[670:673]*/, v41 offset:8704
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[250:257] /*v[762:769]*/, v[114:129], v1, v40 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[250:257] /*v[762:769]*/, v[98:113], v34, v40 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[250:257] /*v[762:769]*/, v[82:97], v35, v40 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[114:129] /*v[626:641]*/, v[250:257] /*v[762:769]*/, v[66:81], v36, v40 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[162:165] /*v[674:677]*/, v41 offset:10240
	ds_load_b128 v[166:169] /*v[678:681]*/, v41 offset:10752
	ds_load_b128 v[170:173] /*v[682:685]*/, v41 offset:12288
	ds_load_b128 v[174:177] /*v[686:689]*/, v41 offset:12800
	ds_load_b128 v[178:181] /*v[690:693]*/, v41 offset:14336
	s_set_vgpr_msb 0x800e
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[114:129] /*v[626:641]*/, v[2:9] /*v[770:777]*/, v[2:17], v36, v40 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xeae
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[98:113] /*v[610:625]*/, v[2:9] /*v[770:777]*/, v[194:209] /*v[706:721]*/, v35, v40 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaefe
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[82:97] /*v[594:609]*/, v[2:9] /*v[770:777]*/, v[20:35] /*v[788:803]*/, v34, v40 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfe0e
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[66:81] /*v[578:593]*/, v[2:9] /*v[770:777]*/, v[50:65], v1, v40 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xe80
	ds_load_b128 v[182:185] /*v[694:697]*/, v41 offset:14848
	s_mov_b32 s1, 0x33000
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8008
	v_dual_add_nc_u32 v41, s1, v213 /*v725*/ :: v_dual_add_nc_u32 v42, s1, v214 /*v726*/
	v_dual_add_nc_u32 v43, s1, v211 /*v723*/ :: v_dual_add_nc_u32 v44, s1, v212 /*v724*/
	s_set_vgpr_msb 0x852
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[2:17] /*v[514:529]*/, v[26:33], v[242:257] /*v[498:513]*/, v18, v24
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v45, 0x22a00, v215 /*v727*/
	v_add_nc_u32_e32 v46, 0x22c00, v215 /*v727*/
	v_add_nc_u32_e32 v47, 0x23200, v215 /*v727*/
	v_add_nc_u32_e32 v48, 0x23400, v215 /*v727*/
	v_add_nc_u32_e32 v49, 0x24a00, v215 /*v727*/
	v_add_nc_u32_e32 v1, 0x22080, v212 /*v724*/
	v_add_nc_u32_e32 v34, 0x22180, v212 /*v724*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[18:33] /*v[530:545]*/, v[26:33], v[226:241] /*v[482:497]*/, v19, v24 matrix_a_reuse
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v35, 0x22280, v212 /*v724*/
	v_add_nc_u32_e32 v36, 0x22380, v212 /*v724*/
	v_add_nc_u32_e32 v37, 0x22080, v211 /*v723*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[34:49] /*v[546:561]*/, v[26:33], v[210:225] /*v[466:481]*/, v22, v24 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[50:65] /*v[562:577]*/, v[26:33], v[194:209] /*v[450:465]*/, v23, v24 matrix_a_reuse
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x5280
	ds_load_b128 v[186:189] /*v[698:701]*/, v45
	ds_load_b128 v[190:193] /*v[702:705]*/, v46
	ds_load_b128 v[218:221] /*v[730:733]*/, v47
	ds_load_b128 v[222:225] /*v[734:737]*/, v48
	s_wait_alu depctr_vm_vsrc(3)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v45, 0x23a00, v215 /*v727*/
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v26, 0x22180, v211 /*v723*/
	v_add_nc_u32_e32 v27, 0x22280, v211 /*v723*/
	v_add_nc_u32_e32 v28, 0x22380, v211 /*v723*/
	v_add_nc_u32_e32 v29, 0x22200, v216 /*v728*/
	v_add_nc_u32_e32 v30, 0x22400, v216 /*v728*/
	s_wait_alu depctr_va_vdst(4)
	ds_load_b32 v38, v26
	s_wait_alu depctr_va_vdst(3)
	ds_load_b32 v39, v27
	s_wait_alu depctr_va_vdst(2)
	ds_load_b32 v40, v28
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[66:69] /*v[578:581]*/, v29
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[70:73] /*v[582:585]*/, v30
	s_wait_alu depctr_vm_vsrc(4)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v26, 0x22a00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v27, 0x22c00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v28, 0x23200, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v29, 0x23400, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v30, 0x23a00, v216 /*v728*/
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[74:77] /*v[586:589]*/, v26
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[78:81] /*v[590:593]*/, v27
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[82:85] /*v[594:597]*/, v28
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[86:89] /*v[598:601]*/, v29
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[90:93] /*v[602:605]*/, v30
	s_wait_alu depctr_vm_vsrc(4)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v26, 0x23c00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v27, 0x24200, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v28, 0x24400, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v29, 0x24a00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v30, 0x24c00, v216 /*v728*/
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[94:97] /*v[606:609]*/, v26
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[98:101] /*v[610:613]*/, v27
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[102:105] /*v[614:617]*/, v28
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[106:109] /*v[618:621]*/, v29
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[110:113] /*v[622:625]*/, v30
	s_wait_alu depctr_vm_vsrc(4)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v26, 0x25200, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v27, 0x25400, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v28, 0x25a00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v29, 0x25c00, v216 /*v728*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v30, 0x22200, v215 /*v727*/
	v_add_nc_u32_e32 v46, 0x23c00, v215 /*v727*/
	v_add_nc_u32_e32 v47, 0x24200, v215 /*v727*/
	v_add_nc_u32_e32 v48, 0x24400, v215 /*v727*/
	s_wait_alu depctr_va_vdst(7)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[114:117] /*v[626:629]*/, v26
	s_wait_alu depctr_va_vdst(6)
	ds_load_b128 v[118:121] /*v[630:633]*/, v27
	s_wait_alu depctr_va_vdst(5)
	ds_load_b128 v[122:125] /*v[634:637]*/, v28
	s_wait_alu depctr_va_vdst(4)
	ds_load_b128 v[126:129] /*v[638:641]*/, v29
	s_wait_alu depctr_va_vdst(3) depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8008
	ds_load_b128 v[26:29], v30
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v30, 0x22400, v215 /*v727*/
	s_set_vgpr_msb 0x880
	ds_load_b128 v[226:229] /*v[738:741]*/, v45
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[230:233] /*v[742:745]*/, v46
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[234:237] /*v[746:749]*/, v47
	s_wait_alu depctr_va_vdst(1)
	ds_load_b128 v[238:241] /*v[750:753]*/, v48
	ds_load_b128 v[242:245] /*v[754:757]*/, v49
	s_wait_alu depctr_vm_vsrc(4)
	s_set_vgpr_msb 0x8008
	v_add_nc_u32_e32 v45, 0x24c00, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(3)
	v_add_nc_u32_e32 v46, 0x25200, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(2)
	v_add_nc_u32_e32 v47, 0x25400, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(1)
	v_add_nc_u32_e32 v48, 0x25a00, v215 /*v727*/
	s_wait_alu depctr_vm_vsrc(0)
	v_add_nc_u32_e32 v49, 0x25c00, v215 /*v727*/
	ds_load_b32 v1, v1
	ds_load_b32 v34, v34
	ds_load_b32 v35, v35
	ds_load_b32 v36, v36
	ds_load_b32 v37, v37
	s_wait_alu depctr_va_vdst(5)
	ds_load_b128 v[30:33], v30
	s_wait_alu depctr_va_vdst(4)
	s_set_vgpr_msb 0x880
	ds_load_b128 v[246:249] /*v[758:761]*/, v45
	s_wait_alu depctr_va_vdst(3)
	ds_load_b128 v[250:253] /*v[762:765]*/, v46
	s_wait_alu depctr_va_vdst(2)
	ds_load_b128 v[254:257] /*v[766:769]*/, v47
	s_wait_alu depctr_va_vdst(1)
	s_set_vgpr_msb 0x80c0
	ds_load_b128 v[2:5] /*v[770:773]*/, v48
	s_wait_alu depctr_va_vdst(0)
	ds_load_b128 v[6:9] /*v[774:777]*/, v49
	s_set_vgpr_msb 0xc00a
	s_wait_dscnt 0x2e
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[242:257], v18, v20 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[226:241], v19, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[210:225], v22, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[194:209], v23, v20 matrix_a_reuse
	s_wait_dscnt 0x2c
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[130:145], v23, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[146:161], v22, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[162:177], v19, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[178:193], v18, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0x2a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[114:129], v18, v21 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[98:113], v19, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[82:97], v22, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[66:81], v23, v21 matrix_a_reuse
	s_wait_dscnt 0x28
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[2:17], v23, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[194:209] /*v[706:721]*/, v22, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaafa
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[20:35] /*v[788:803]*/, v19, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[50:65], v18, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa5a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v23, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v22, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v19, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v18, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v18, v25 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v19, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v22, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v23, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[2:17] /*v[258:273]*/, v23, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[18:33] /*v[274:289]*/, v22, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[34:49] /*v[290:305]*/, v19, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[50:65] /*v[306:321]*/, v18, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a52
	s_wait_dscnt 0x5
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[66:81] /*v[578:593]*/, v[26:33], v[242:257] /*v[498:513]*/, v1, v37
	s_wait_tensorcnt 0x0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[82:97] /*v[594:609]*/, v[26:33], v[226:241] /*v[482:497]*/, v34, v37 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[98:113] /*v[610:625]*/, v[26:33], v[210:225] /*v[466:481]*/, v35, v37 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[114:129] /*v[626:641]*/, v[26:33], v[194:209] /*v[450:465]*/, v36, v37 matrix_a_reuse
	s_set_vgpr_msb 0x5200
	ds_load_2addr_stride64_b32 v[18:19], v44 offset1:1
	ds_load_2addr_stride64_b32 v[22:23], v44 offset0:2 offset1:3
	ds_load_2addr_stride64_b32 v[24:25], v43 offset1:1
	ds_load_2addr_stride64_b32 v[20:21], v43 offset0:2 offset1:3
	s_set_vgpr_msb 0x80
	ds_load_b128 v[2:5] /*v[514:517]*/, v42
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[186:193] /*v[698:705]*/, v[130:145] /*v[386:401]*/, v36, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[186:193] /*v[698:705]*/, v[146:161] /*v[402:417]*/, v35, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[186:193] /*v[698:705]*/, v[162:177] /*v[418:433]*/, v34, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[186:193] /*v[698:705]*/, v[178:193] /*v[434:449]*/, v1, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[6:9] /*v[518:521]*/, v42 offset:512
	ds_load_b128 v[10:13] /*v[522:525]*/, v42 offset:2048
	ds_load_b128 v[14:17] /*v[526:529]*/, v42 offset:2560
	ds_load_b128 v[18:21] /*v[530:533]*/, v42 offset:4096
	ds_load_b128 v[22:25] /*v[534:537]*/, v42 offset:4608
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[218:225] /*v[730:737]*/, v[114:129] /*v[370:385]*/, v1, v38 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[218:225] /*v[730:737]*/, v[98:113] /*v[354:369]*/, v34, v38 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[218:225] /*v[730:737]*/, v[82:97] /*v[338:353]*/, v35, v38 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[218:225] /*v[730:737]*/, v[66:81] /*v[322:337]*/, v36, v38 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[26:29] /*v[538:541]*/, v42 offset:6144
	ds_load_b128 v[30:33] /*v[542:545]*/, v42 offset:6656
	ds_load_b128 v[34:37] /*v[546:549]*/, v42 offset:8192
	ds_load_b128 v[38:41] /*v[550:553]*/, v42 offset:8704
	ds_load_b128 v[42:45] /*v[554:557]*/, v42 offset:10240
	s_set_vgpr_msb 0x805a
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[226:233] /*v[738:745]*/, v[2:17] /*v[258:273]*/, v36, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[226:233] /*v[738:745]*/, v[18:33] /*v[274:289]*/, v35, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[226:233] /*v[738:745]*/, v[34:49] /*v[290:305]*/, v34, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[226:233] /*v[738:745]*/, v[50:65] /*v[306:321]*/, v1, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[46:49] /*v[558:561]*/, v42 offset:10752
	ds_load_b128 v[50:53] /*v[562:565]*/, v42 offset:12288
	ds_load_b128 v[54:57] /*v[566:569]*/, v42 offset:12800
	ds_load_b128 v[58:61] /*v[570:573]*/, v42 offset:14336
	ds_load_b128 v[62:65] /*v[574:577]*/, v42 offset:14848
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[234:241] /*v[746:753]*/, v[242:257], v1, v39 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[234:241] /*v[746:753]*/, v[226:241], v34, v39 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[234:241] /*v[746:753]*/, v[210:225], v35, v39 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[234:241] /*v[746:753]*/, v[194:209], v36, v39 matrix_a_reuse
	s_set_vgpr_msb 0xa00
	ds_load_b128 v[26:29], v41
	s_wait_alu depctr_vm_vsrc(6)
	ds_load_b128 v[30:33], v41 offset:512
	s_set_vgpr_msb 0x80
	ds_load_b128 v[130:133] /*v[642:645]*/, v41 offset:2048
	ds_load_b128 v[134:137] /*v[646:649]*/, v41 offset:2560
	ds_load_b128 v[138:141] /*v[650:653]*/, v41 offset:4096
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[242:249] /*v[754:761]*/, v[130:145], v36, v39 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[242:249] /*v[754:761]*/, v[146:161], v35, v39 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[242:249] /*v[754:761]*/, v[162:177], v34, v39 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[242:249] /*v[754:761]*/, v[178:193], v1, v39 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[142:145] /*v[654:657]*/, v41 offset:4608
	ds_load_b128 v[146:149] /*v[658:661]*/, v41 offset:6144
	ds_load_b128 v[150:153] /*v[662:665]*/, v41 offset:6656
	ds_load_b128 v[154:157] /*v[666:669]*/, v41 offset:8192
	ds_load_b128 v[158:161] /*v[670:673]*/, v41 offset:8704
	s_set_vgpr_msb 0x800a
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[250:257] /*v[762:769]*/, v[114:129], v1, v40 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[250:257] /*v[762:769]*/, v[98:113], v34, v40 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[250:257] /*v[762:769]*/, v[82:97], v35, v40 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[114:129] /*v[626:641]*/, v[250:257] /*v[762:769]*/, v[66:81], v36, v40 matrix_a_reuse
	s_set_vgpr_msb 0xa80
	ds_load_b128 v[162:165] /*v[674:677]*/, v41 offset:10240
	ds_load_b128 v[166:169] /*v[678:681]*/, v41 offset:10752
	ds_load_b128 v[170:173] /*v[682:685]*/, v41 offset:12288
	ds_load_b128 v[174:177] /*v[686:689]*/, v41 offset:12800
	ds_load_b128 v[178:181] /*v[690:693]*/, v41 offset:14336
	s_set_vgpr_msb 0x800e
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[114:129] /*v[626:641]*/, v[2:9] /*v[770:777]*/, v[2:17], v36, v40 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xeae
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[98:113] /*v[610:625]*/, v[2:9] /*v[770:777]*/, v[194:209] /*v[706:721]*/, v35, v40 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaefe
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[82:97] /*v[594:609]*/, v[2:9] /*v[770:777]*/, v[20:35] /*v[788:803]*/, v34, v40 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfe0e
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[66:81] /*v[578:593]*/, v[2:9] /*v[770:777]*/, v[50:65], v1, v40 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xe80
	ds_load_b128 v[182:185] /*v[694:697]*/, v41 offset:14848
	s_set_vgpr_msb 0x8052
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[2:17] /*v[514:529]*/, v[26:33], v[242:257] /*v[498:513]*/, v18, v24
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v1, 0x33080, v212 /*v724*/
	v_add_nc_u32_e32 v34, 0x33180, v212 /*v724*/
	v_add_nc_u32_e32 v35, 0x33280, v212 /*v724*/
	v_add_nc_u32_e32 v36, 0x33380, v212 /*v724*/
	v_add_nc_u32_e32 v37, 0x33080, v211 /*v723*/
	v_add_nc_u32_e32 v38, 0x33180, v211 /*v723*/
	v_add_nc_u32_e32 v39, 0x33280, v211 /*v723*/
	v_add_nc_u32_e32 v40, 0x33380, v211 /*v723*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[18:33] /*v[530:545]*/, v[26:33], v[226:241] /*v[482:497]*/, v19, v24 matrix_a_reuse
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v41, 0x33200, v216 /*v728*/
	v_add_nc_u32_e32 v42, 0x33400, v216 /*v728*/
	v_add_nc_u32_e32 v43, 0x33a00, v216 /*v728*/
	v_add_nc_u32_e32 v44, 0x33c00, v216 /*v728*/
	v_add_nc_u32_e32 v45, 0x34200, v216 /*v728*/
	v_add_nc_u32_e32 v46, 0x34400, v216 /*v728*/
	v_add_nc_u32_e32 v47, 0x34a00, v216 /*v728*/
	s_set_vgpr_msb 0x852
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[34:49] /*v[546:561]*/, v[26:33], v[210:225] /*v[466:481]*/, v22, v24 matrix_a_reuse
	s_set_vgpr_msb 0x5208
	v_add_nc_u32_e32 v48, 0x34c00, v216 /*v728*/
	v_add_nc_u32_e32 v49, 0x35200, v216 /*v728*/
	s_set_vgpr_msb 0x888
	v_add_nc_u32_e32 v102 /*v614*/, 0x35400, v216 /*v728*/
	v_add_nc_u32_e32 v106 /*v618*/, 0x35a00, v216 /*v728*/
	v_add_nc_u32_e32 v110 /*v622*/, 0x35c00, v216 /*v728*/
	v_add_nc_u32_e32 v114 /*v626*/, 0x36200, v216 /*v728*/
	v_add_nc_u32_e32 v118 /*v630*/, 0x36400, v216 /*v728*/
	s_set_vgpr_msb 0x8852
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[50:65] /*v[562:577]*/, v[26:33], v[194:209] /*v[450:465]*/, v23, v24 matrix_a_reuse
	s_set_vgpr_msb 0x5288
	v_add_nc_u32_e32 v186 /*v698*/, 0x33a00, v215 /*v727*/
	v_add_nc_u32_e32 v187 /*v699*/, 0x33c00, v215 /*v727*/
	v_add_nc_u32_e32 v188 /*v700*/, 0x34200, v215 /*v727*/
	v_add_nc_u32_e32 v189 /*v701*/, 0x34400, v215 /*v727*/
	s_set_vgpr_msb 0x8808
	v_add_nc_u32_e32 v26, 0x36a00, v216 /*v728*/
	v_add_nc_u32_e32 v27, 0x36c00, v216 /*v728*/
	v_add_nc_u32_e32 v28, 0x33200, v215 /*v727*/
	v_add_nc_u32_e32 v29, 0x33400, v215 /*v727*/
	s_set_vgpr_msb 0x888
	v_add_nc_u32_e32 v190 /*v702*/, 0x34a00, v215 /*v727*/
	v_add_nc_u32_e32 v191 /*v703*/, 0x34c00, v215 /*v727*/
	v_add_nc_u32_e32 v192 /*v704*/, 0x35200, v215 /*v727*/
	v_add_nc_u32_e32 v193 /*v705*/, 0x35400, v215 /*v727*/
	v_add_nc_u32_e32 v211 /*v723*/, 0x35a00, v215 /*v727*/
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x8800
	ds_load_b32 v1, v1
	ds_load_b32 v32, v34
	ds_load_b32 v33, v35
	s_wait_alu depctr_vm_vsrc(1)
	ds_load_b32 v34, v36
	s_wait_alu depctr_vm_vsrc(1)
	ds_load_b32 v35, v37
	s_set_vgpr_msb 0x5a
	s_wait_dscnt 0x11
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[50:65] /*v[562:577]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v23, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0x5a88
	v_add_nc_u32_e32 v212 /*v724*/, 0x35c00, v215 /*v727*/
	v_add_nc_u32_e32 v213 /*v725*/, 0x36200, v215 /*v727*/
	s_set_vgpr_msb 0x885a
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[34:49] /*v[546:561]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v22, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[18:33] /*v[530:545]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v19, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[2:17] /*v[514:529]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v18, v24 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x5a00
	ds_load_b32 v36, v38
	s_wait_alu depctr_vm_vsrc(1)
	ds_load_b32 v37, v39
	s_wait_alu depctr_vm_vsrc(1)
	ds_load_b32 v38, v40
	s_set_vgpr_msb 0x80
	ds_load_b128 v[66:69] /*v[578:581]*/, v41
	ds_load_b128 v[70:73] /*v[582:585]*/, v42
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x14
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[2:17] /*v[514:529]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v18, v25 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[18:33] /*v[530:545]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v19, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[34:49] /*v[546:561]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v22, v25 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[50:65] /*v[562:577]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v23, v25 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[74:77] /*v[586:589]*/, v43
	ds_load_b128 v[78:81] /*v[590:593]*/, v44
	ds_load_b128 v[82:85] /*v[594:597]*/, v45
	ds_load_b128 v[86:89] /*v[598:601]*/, v46
	ds_load_b128 v[90:93] /*v[602:605]*/, v47
	s_set_vgpr_msb 0x805a
	s_wait_dscnt 0x17
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[50:65] /*v[562:577]*/, v[146:153] /*v[658:665]*/, v[2:17] /*v[258:273]*/, v23, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[34:49] /*v[546:561]*/, v[146:153] /*v[658:665]*/, v[18:33] /*v[274:289]*/, v22, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[18:33] /*v[530:545]*/, v[146:153] /*v[658:665]*/, v[34:49] /*v[290:305]*/, v19, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[2:17] /*v[514:529]*/, v[146:153] /*v[658:665]*/, v[50:65] /*v[306:321]*/, v18, v25 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a80
	ds_load_b128 v[94:97] /*v[606:609]*/, v48
	ds_load_b128 v[98:101] /*v[610:613]*/, v49
	s_set_vgpr_msb 0x8082
	ds_load_b128 v[102:105] /*v[614:617]*/, v102 /*v614*/
	ds_load_b128 v[106:109] /*v[618:621]*/, v106 /*v618*/
	ds_load_b128 v[110:113] /*v[622:625]*/, v110 /*v622*/
	s_set_vgpr_msb 0x820a
	s_wait_dscnt 0x1a
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[2:17] /*v[514:529]*/, v[154:161] /*v[666:673]*/, v[242:257], v18, v20 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[18:33] /*v[530:545]*/, v[154:161] /*v[666:673]*/, v[226:241], v19, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[34:49] /*v[546:561]*/, v[154:161] /*v[666:673]*/, v[210:225], v22, v20 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[50:65] /*v[562:577]*/, v[154:161] /*v[666:673]*/, v[194:209], v23, v20 matrix_a_reuse
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[114:117] /*v[626:629]*/, v114 /*v626*/
	ds_load_b128 v[118:121] /*v[630:633]*/, v118 /*v630*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[122:125] /*v[634:637]*/, v26
	ds_load_b128 v[126:129] /*v[638:641]*/, v27
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8000
	ds_load_b128 v[24:27], v28
	s_set_vgpr_msb 10
	s_wait_dscnt 0x1d
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[50:65] /*v[562:577]*/, v[162:169] /*v[674:681]*/, v[130:145], v23, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[34:49] /*v[546:561]*/, v[162:169] /*v[674:681]*/, v[146:161], v22, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[18:33] /*v[530:545]*/, v[162:169] /*v[674:681]*/, v[162:177], v19, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[2:17] /*v[514:529]*/, v[162:169] /*v[674:681]*/, v[178:193], v18, v20 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0xa00
	ds_load_b128 v[28:31], v29
	s_set_vgpr_msb 0x82
	ds_load_b128 v[130:133] /*v[642:645]*/, v186 /*v698*/
	ds_load_b128 v[134:137] /*v[646:649]*/, v187 /*v699*/
	ds_load_b128 v[138:141] /*v[650:653]*/, v188 /*v700*/
	ds_load_b128 v[142:145] /*v[654:657]*/, v189 /*v701*/
	s_set_vgpr_msb 0x820a
	v_add_nc_u32_e32 v20, 0x36a00, v215 /*v727*/
	s_wait_dscnt 0x20
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[2:17] /*v[514:529]*/, v[170:177] /*v[682:689]*/, v[114:129], v18, v21 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[18:33] /*v[530:545]*/, v[170:177] /*v[682:689]*/, v[98:113], v19, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[34:49] /*v[546:561]*/, v[170:177] /*v[682:689]*/, v[82:97], v22, v21 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[50:65] /*v[562:577]*/, v[170:177] /*v[682:689]*/, v[66:81], v23, v21 matrix_a_reuse
	s_set_vgpr_msb 0xaaa
	ds_load_b128 v[146:149] /*v[658:661]*/, v190 /*v702*/
	ds_load_b128 v[150:153] /*v[662:665]*/, v191 /*v703*/
	ds_load_b128 v[154:157] /*v[666:669]*/, v192 /*v704*/
	ds_load_b128 v[158:161] /*v[670:673]*/, v193 /*v705*/
	ds_load_b128 v[162:165] /*v[674:677]*/, v211 /*v723*/
	s_wait_dscnt 0x23
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[34:49] /*v[546:561]*/, v[178:185] /*v[690:697]*/, v[194:209] /*v[706:721]*/, v22, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaa08
	v_add_nc_u32_e32 v22, 0x36c00, v215 /*v727*/
	s_set_vgpr_msb 0x8fa
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[18:33] /*v[530:545]*/, v[178:185] /*v[690:697]*/, v[20:35] /*v[788:803]*/, v19, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfa0a
	v_add_nc_u32_e32 v19, 0x36400, v215 /*v727*/
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[50:65] /*v[562:577]*/, v[178:185] /*v[690:697]*/, v[2:17], v23, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[2:17] /*v[514:529]*/, v[178:185] /*v[690:697]*/, v[50:65], v18, v21 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0xa82
	ds_load_b128 v[166:169] /*v[678:681]*/, v212 /*v724*/
	ds_load_b128 v[2:5] /*v[514:517]*/, v213 /*v725*/
	s_set_vgpr_msb 0x8280
	ds_load_b128 v[6:9] /*v[518:521]*/, v19
	ds_load_b128 v[10:13] /*v[522:525]*/, v20
	ds_load_b128 v[14:17] /*v[526:529]*/, v22
	s_set_vgpr_msb 0x8052
	s_wait_dscnt 0xe
	v_wmma_scale_f32_32x16x128_f4 v[242:257] /*v[498:513]*/, v[66:81] /*v[578:593]*/, v[24:31], v[242:257] /*v[498:513]*/, v1, v35
	v_wmma_scale_f32_32x16x128_f4 v[226:241] /*v[482:497]*/, v[82:97] /*v[594:609]*/, v[24:31], v[226:241] /*v[482:497]*/, v32, v35 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225] /*v[466:481]*/, v[98:113] /*v[610:625]*/, v[24:31], v[210:225] /*v[466:481]*/, v33, v35 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[450:465]*/, v[114:129] /*v[626:641]*/, v[24:31], v[194:209] /*v[450:465]*/, v34, v35 matrix_a_reuse
	s_set_vgpr_msb 0x525a
	s_wait_dscnt 0xc
	v_wmma_scale_f32_32x16x128_f4 v[130:145] /*v[386:401]*/, v[114:129] /*v[626:641]*/, v[130:137] /*v[642:649]*/, v[130:145] /*v[386:401]*/, v34, v35 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161] /*v[402:417]*/, v[98:113] /*v[610:625]*/, v[130:137] /*v[642:649]*/, v[146:161] /*v[402:417]*/, v33, v35 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177] /*v[418:433]*/, v[82:97] /*v[594:609]*/, v[130:137] /*v[642:649]*/, v[162:177] /*v[418:433]*/, v32, v35 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193] /*v[434:449]*/, v[66:81] /*v[578:593]*/, v[130:137] /*v[642:649]*/, v[178:193] /*v[434:449]*/, v1, v35 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0xa
	v_wmma_scale_f32_32x16x128_f4 v[114:129] /*v[370:385]*/, v[66:81] /*v[578:593]*/, v[138:145] /*v[650:657]*/, v[114:129] /*v[370:385]*/, v1, v36 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113] /*v[354:369]*/, v[82:97] /*v[594:609]*/, v[138:145] /*v[650:657]*/, v[98:113] /*v[354:369]*/, v32, v36 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97] /*v[338:353]*/, v[98:113] /*v[610:625]*/, v[138:145] /*v[650:657]*/, v[82:97] /*v[338:353]*/, v33, v36 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81] /*v[322:337]*/, v[114:129] /*v[626:641]*/, v[138:145] /*v[650:657]*/, v[66:81] /*v[322:337]*/, v34, v36 matrix_a_reuse
	s_wait_dscnt 0x8
	v_wmma_scale_f32_32x16x128_f4 v[2:17] /*v[258:273]*/, v[114:129] /*v[626:641]*/, v[146:153] /*v[658:665]*/, v[2:17] /*v[258:273]*/, v34, v36 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[18:33] /*v[274:289]*/, v[98:113] /*v[610:625]*/, v[146:153] /*v[658:665]*/, v[18:33] /*v[274:289]*/, v33, v36 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[34:49] /*v[290:305]*/, v[82:97] /*v[594:609]*/, v[146:153] /*v[658:665]*/, v[34:49] /*v[290:305]*/, v32, v36 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[50:65] /*v[306:321]*/, v[66:81] /*v[578:593]*/, v[146:153] /*v[658:665]*/, v[50:65] /*v[306:321]*/, v1, v36 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0x5a0a
	s_wait_dscnt 0x6
	v_wmma_scale_f32_32x16x128_f4 v[242:257], v[66:81] /*v[578:593]*/, v[154:161] /*v[666:673]*/, v[242:257], v1, v37 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[226:241], v[82:97] /*v[594:609]*/, v[154:161] /*v[666:673]*/, v[226:241], v32, v37 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[210:225], v[98:113] /*v[610:625]*/, v[154:161] /*v[666:673]*/, v[210:225], v33, v37 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[194:209], v[114:129] /*v[626:641]*/, v[154:161] /*v[666:673]*/, v[194:209], v34, v37 matrix_a_reuse
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[130:145], v[114:129] /*v[626:641]*/, v[162:169] /*v[674:681]*/, v[130:145], v34, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[146:161], v[98:113] /*v[610:625]*/, v[162:169] /*v[674:681]*/, v[146:161], v33, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[162:177], v[82:97] /*v[594:609]*/, v[162:169] /*v[674:681]*/, v[162:177], v32, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[178:193], v[66:81] /*v[578:593]*/, v[162:169] /*v[674:681]*/, v[178:193], v1, v37 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[114:129], v[66:81] /*v[578:593]*/, v[2:9] /*v[514:521]*/, v[114:129], v1, v38 matrix_b_reuse
	v_wmma_scale_f32_32x16x128_f4 v[98:113], v[82:97] /*v[594:609]*/, v[2:9] /*v[514:521]*/, v[98:113], v32, v38 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[82:97], v[98:113] /*v[610:625]*/, v[2:9] /*v[514:521]*/, v[82:97], v33, v38 matrix_a_reuse
	v_wmma_scale_f32_32x16x128_f4 v[66:81], v[114:129] /*v[626:641]*/, v[2:9] /*v[514:521]*/, v[66:81], v34, v38 matrix_a_reuse
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[2:17], v[114:129] /*v[626:641]*/, v[10:17] /*v[522:529]*/, v[2:17], v34, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_b_reuse
	s_set_vgpr_msb 0xaaa
	v_wmma_scale_f32_32x16x128_f4 v[194:209] /*v[706:721]*/, v[98:113] /*v[610:625]*/, v[10:17] /*v[522:529]*/, v[194:209] /*v[706:721]*/, v33, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xaafa
	v_wmma_scale_f32_32x16x128_f4 v[20:35] /*v[788:803]*/, v[82:97] /*v[594:609]*/, v[10:17] /*v[522:529]*/, v[20:35] /*v[788:803]*/, v32, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xfa0a
	v_wmma_scale_f32_32x16x128_f4 v[50:65], v[66:81] /*v[578:593]*/, v[10:17] /*v[522:529]*/, v[50:65], v1, v38 matrix_b_scale:MATRIX_SCALE_ROW1 matrix_a_reuse
	s_set_vgpr_msb 0xa05
	v_max_num_f32_e32 v1, v242 /*v498*/, v242 /*v498*/
	v_max_num_f32_e64 v18, s26, s26
	s_wait_alu depctr_vm_vsrc(2)
	v_max_num_f32_e32 v19, v244 /*v500*/, v244 /*v500*/
	s_sub_f32 s16, 0, s26
	s_mov_b32 s11, 0
	s_wait_tensorcnt 0x0
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v20, v1, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v1, v246 /*v502*/, v246 /*v502*/
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v21, v19, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v19, v248 /*v504*/, v248 /*v504*/
	s_set_vgpr_msb 0x500
	s_barrier_signal -1
	v_min_num_f32_e32 v24, v1, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v1, v250 /*v506*/, v250 /*v506*/
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v25, v19, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v19, v252 /*v508*/, v252 /*v508*/
	s_set_vgpr_msb 0x588
	v_or_b32_e32 v4 /*v516*/, s0, v210 /*v722*/
	s_wait_alu depctr_vm_vsrc(0)
	s_set_vgpr_msb 0x8801
	v_med3_num_f32 v22, v243 /*v499*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v28, v1, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v1, v254 /*v510*/, v254 /*v510*/
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v29, v19, v18
	s_set_vgpr_msb 10
	v_max_num_f32_e32 v19, v0 /*v512*/, v0 /*v512*/
	s_set_vgpr_msb 0xa01
	v_med3_num_f32 v23, v245 /*v501*/, -s26, s26
	v_med3_num_f32 v26, v247 /*v503*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v32, v1, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v1, v226 /*v482*/, v226 /*v482*/
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v33, v19, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v19, v228 /*v484*/, v228 /*v484*/
	s_set_vgpr_msb 0x545
	v_max_num_f32_e32 v228 /*v484*/, v208 /*v464*/, v208 /*v464*/
	s_set_vgpr_msb 0x4501
	v_med3_num_f32 v27, v249 /*v505*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v36, v1, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v1, v230 /*v486*/, v230 /*v486*/
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v37, v19, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v19, v232 /*v488*/, v232 /*v488*/
	v_med3_num_f32 v30, v251 /*v507*/, -s26, s26
	v_med3_num_f32 v31, v253 /*v509*/, -s26, s26
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v40, v1, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v1, v234 /*v490*/, v234 /*v490*/
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v41, v19, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v19, v236 /*v492*/, v236 /*v492*/
	v_med3_num_f32 v34, v255 /*v511*/, -s26, s26
	s_set_vgpr_msb 0x502
	v_med3_num_f32 v35, v1 /*v513*/, -s26, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v44, v1, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v1, v238 /*v494*/, v238 /*v494*/
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v45, v19, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v19, v240 /*v496*/, v240 /*v496*/
	v_med3_num_f32 v38, v227 /*v483*/, -s26, s26
	v_med3_num_f32 v39, v229 /*v485*/, -s26, s26
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v48, v1, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v1, v210 /*v466*/, v210 /*v466*/
	s_set_vgpr_msb 0x500
	v_min_num_f32_e32 v49, v19, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v19, v212 /*v468*/, v212 /*v468*/
	s_set_vgpr_msb 0x541
	v_med3_num_f32 v212 /*v468*/, v211 /*v467*/, -s26, s26
	s_set_vgpr_msb 0x4101
	v_med3_num_f32 v42, v231 /*v487*/, -s26, s26
	s_set_vgpr_msb 0x140
	v_min_num_f32_e32 v210 /*v466*/, v1, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v1, v214 /*v470*/, v214 /*v470*/
	s_set_vgpr_msb 0x540
	v_min_num_f32_e32 v211 /*v467*/, v19, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v19, v216 /*v472*/, v216 /*v472*/
	s_set_vgpr_msb 0x541
	v_med3_num_f32 v216 /*v472*/, v215 /*v471*/, -s26, s26
	s_set_vgpr_msb 0x4101
	v_med3_num_f32 v43, v233 /*v489*/, -s26, s26
	s_set_vgpr_msb 0x140
	v_min_num_f32_e32 v214 /*v470*/, v1, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v1, v218 /*v474*/, v218 /*v474*/
	s_set_vgpr_msb 0x540
	v_min_num_f32_e32 v215 /*v471*/, v19, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v19, v220 /*v476*/, v220 /*v476*/
	s_set_vgpr_msb 0x541
	v_med3_num_f32 v220 /*v476*/, v219 /*v475*/, -s26, s26
	s_set_vgpr_msb 0x4101
	v_med3_num_f32 v46, v235 /*v491*/, -s26, s26
	s_set_vgpr_msb 0x140
	v_min_num_f32_e32 v218 /*v474*/, v1, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v1, v222 /*v478*/, v222 /*v478*/
	s_set_vgpr_msb 0x540
	v_min_num_f32_e32 v219 /*v475*/, v19, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v19, v224 /*v480*/, v224 /*v480*/
	s_set_vgpr_msb 0x541
	v_med3_num_f32 v224 /*v480*/, v223 /*v479*/, -s26, s26
	s_set_vgpr_msb 0x4101
	v_med3_num_f32 v47, v237 /*v493*/, -s26, s26
	s_set_vgpr_msb 0x140
	v_min_num_f32_e32 v222 /*v478*/, v1, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v1, v194 /*v450*/, v194 /*v450*/
	s_set_vgpr_msb 0x540
	v_min_num_f32_e32 v223 /*v479*/, v19, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v19, v196 /*v452*/, v196 /*v452*/
	s_set_vgpr_msb 0x541
	v_med3_num_f32 v196 /*v452*/, v195 /*v451*/, -s26, s26
	v_med3_num_f32 v226 /*v482*/, v239 /*v495*/, -s26, s26
	s_set_vgpr_msb 0x4140
	v_min_num_f32_e32 v194 /*v450*/, v1, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v1, v198 /*v454*/, v198 /*v454*/
	s_set_vgpr_msb 0x540
	v_min_num_f32_e32 v195 /*v451*/, v19, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v19, v200 /*v456*/, v200 /*v456*/
	s_set_vgpr_msb 0x541
	v_med3_num_f32 v200 /*v456*/, v199 /*v455*/, -s26, s26
	v_med3_num_f32 v227 /*v483*/, v241 /*v497*/, -s26, s26
	s_set_vgpr_msb 0x4140
	v_min_num_f32_e32 v198 /*v454*/, v1, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v1, v202 /*v458*/, v202 /*v458*/
	s_set_vgpr_msb 0x540
	v_min_num_f32_e32 v199 /*v455*/, v19, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v19, v204 /*v460*/, v204 /*v460*/
	s_set_vgpr_msb 0x541
	v_med3_num_f32 v213 /*v469*/, v213 /*v469*/, -s26, s26
	v_med3_num_f32 v217 /*v473*/, v217 /*v473*/, -s26, s26
	s_set_vgpr_msb 0x4140
	v_min_num_f32_e32 v202 /*v458*/, v1, v18
	s_set_vgpr_msb 0x4005
	v_max_num_f32_e32 v1, v206 /*v462*/, v206 /*v462*/
	s_set_vgpr_msb 0x541
	v_med3_num_f32 v221 /*v477*/, v221 /*v477*/, -s26, s26
	v_med3_num_f32 v225 /*v481*/, v225 /*v481*/, -s26, s26
	v_med3_num_f32 v197 /*v453*/, v197 /*v453*/, -s26, s26
	v_med3_num_f32 v201 /*v457*/, v201 /*v457*/, -s26, s26
	v_med3_num_f32 v204 /*v460*/, v203 /*v459*/, -s26, s26
	s_set_vgpr_msb 0x4140
	v_min_num_f32_e32 v203 /*v459*/, v19, v18
	s_set_vgpr_msb 0x4041
	v_med3_num_f32 v205 /*v461*/, v205 /*v461*/, -s26, s26
	s_set_vgpr_msb 0x4140
	v_min_num_f32_e32 v206 /*v462*/, v1, v18
	s_set_vgpr_msb 0x4041
	v_med3_num_f32 v208 /*v464*/, v207 /*v463*/, -s26, s26
	v_min_num_f32_e32 v207 /*v463*/, v228 /*v484*/, v18
	v_med3_num_f32 v209 /*v465*/, v209 /*v465*/, -s26, s26
	s_barrier_wait -1
	s_set_vgpr_msb 0x4100
	v_mul_f32_e32 v1, 0xbfb8aa3b, v20
	v_mul_f32_e32 v19, 0xbfb8aa3b, v21
	s_set_vgpr_msb 64
	v_dual_mul_f32 v228 /*v484*/, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v229 /*v485*/, 0xbfb8aa3b, v25
	v_mul_f32_e32 v230 /*v486*/, 0xbfb8aa3b, v28
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v1, v1
	v_exp_f32_e32 v19, v19
	s_set_vgpr_msb 64
	v_dual_mul_f32 v231 /*v487*/, 0xbfb8aa3b, v29 :: v_dual_mul_f32 v232 /*v488*/, 0xbfb8aa3b, v32
	v_dual_mul_f32 v233 /*v489*/, 0xbfb8aa3b, v33 :: v_dual_mul_f32 v234 /*v490*/, 0xbfb8aa3b, v36
	v_dual_mul_f32 v235 /*v491*/, 0xbfb8aa3b, v37 :: v_dual_mul_f32 v236 /*v492*/, 0xbfb8aa3b, v40
	v_dual_mul_f32 v237 /*v493*/, 0xbfb8aa3b, v41 :: v_dual_mul_f32 v238 /*v494*/, 0xbfb8aa3b, v44
	v_dual_mul_f32 v239 /*v495*/, 0xbfb8aa3b, v45 :: v_dual_mul_f32 v240 /*v496*/, 0xbfb8aa3b, v48
	v_mul_f32_e32 v241 /*v497*/, 0xbfb8aa3b, v49
	s_set_vgpr_msb 0x4044
	v_dual_mul_f32 v242 /*v498*/, 0xbfb8aa3b, v210 /*v466*/ :: v_dual_mul_f32 v243 /*v499*/, 0xbfb8aa3b, v211 /*v467*/
	v_dual_mul_f32 v244 /*v500*/, 0xbfb8aa3b, v214 /*v470*/ :: v_dual_mul_f32 v245 /*v501*/, 0xbfb8aa3b, v215 /*v471*/
	v_dual_mul_f32 v246 /*v502*/, 0xbfb8aa3b, v218 /*v474*/ :: v_dual_mul_f32 v247 /*v503*/, 0xbfb8aa3b, v219 /*v475*/
	v_dual_mul_f32 v248 /*v504*/, 0xbfb8aa3b, v222 /*v478*/ :: v_dual_mul_f32 v249 /*v505*/, 0xbfb8aa3b, v223 /*v479*/
	v_dual_mul_f32 v250 /*v506*/, 0xbfb8aa3b, v194 /*v450*/ :: v_dual_mul_f32 v251 /*v507*/, 0xbfb8aa3b, v195 /*v451*/
	v_dual_mul_f32 v252 /*v508*/, 0xbfb8aa3b, v198 /*v454*/ :: v_dual_mul_f32 v253 /*v509*/, 0xbfb8aa3b, v199 /*v455*/
	v_dual_mul_f32 v254 /*v510*/, 0xbfb8aa3b, v202 /*v458*/ :: v_dual_mul_f32 v255 /*v511*/, 0xbfb8aa3b, v203 /*v459*/
	s_set_vgpr_msb 0x4484
	v_dual_mul_f32 v0 /*v512*/, 0xbfb8aa3b, v206 /*v462*/ :: v_dual_mul_f32 v1 /*v513*/, 0xbfb8aa3b, v207 /*v463*/
	s_set_vgpr_msb 0x8441
	v_exp_f32_e32 v228 /*v484*/, v228 /*v484*/
	v_exp_f32_e32 v229 /*v485*/, v229 /*v485*/
	v_exp_f32_e32 v230 /*v486*/, v230 /*v486*/
	v_exp_f32_e32 v231 /*v487*/, v231 /*v487*/
	v_exp_f32_e32 v232 /*v488*/, v232 /*v488*/
	v_exp_f32_e32 v233 /*v489*/, v233 /*v489*/
	v_exp_f32_e32 v234 /*v490*/, v234 /*v490*/
	v_exp_f32_e32 v235 /*v491*/, v235 /*v491*/
	v_exp_f32_e32 v236 /*v492*/, v236 /*v492*/
	v_exp_f32_e32 v237 /*v493*/, v237 /*v493*/
	v_exp_f32_e32 v238 /*v494*/, v238 /*v494*/
	v_exp_f32_e32 v239 /*v495*/, v239 /*v495*/
	v_exp_f32_e32 v240 /*v496*/, v240 /*v496*/
	v_exp_f32_e32 v241 /*v497*/, v241 /*v497*/
	v_exp_f32_e32 v242 /*v498*/, v242 /*v498*/
	v_exp_f32_e32 v243 /*v499*/, v243 /*v499*/
	v_exp_f32_e32 v244 /*v500*/, v244 /*v500*/
	v_exp_f32_e32 v245 /*v501*/, v245 /*v501*/
	v_exp_f32_e32 v246 /*v502*/, v246 /*v502*/
	v_exp_f32_e32 v247 /*v503*/, v247 /*v503*/
	v_exp_f32_e32 v248 /*v504*/, v248 /*v504*/
	v_exp_f32_e32 v249 /*v505*/, v249 /*v505*/
	v_exp_f32_e32 v250 /*v506*/, v250 /*v506*/
	v_exp_f32_e32 v251 /*v507*/, v251 /*v507*/
	v_exp_f32_e32 v252 /*v508*/, v252 /*v508*/
	v_exp_f32_e32 v253 /*v509*/, v253 /*v509*/
	v_exp_f32_e32 v254 /*v510*/, v254 /*v510*/
	v_exp_f32_e32 v255 /*v511*/, v255 /*v511*/
	s_set_vgpr_msb 0x4182
	v_exp_f32_e32 v0 /*v512*/, v0 /*v512*/
	v_exp_f32_e32 v1 /*v513*/, v1 /*v513*/
	s_set_vgpr_msb 0x8200
	v_dual_add_f32 v1, 1.0, v1 :: v_dual_add_f32 v19, 1.0, v19
	s_set_vgpr_msb 0x84
	v_dual_add_f32 v2 /*v514*/, 1.0, v228 /*v484*/ :: v_dual_add_f32 v3 /*v515*/, 1.0, v229 /*v485*/
	v_dual_add_f32 v5 /*v517*/, 1.0, v230 /*v486*/ :: v_dual_add_f32 v6 /*v518*/, 1.0, v231 /*v487*/
	v_dual_add_f32 v7 /*v519*/, 1.0, v232 /*v488*/ :: v_dual_add_f32 v8 /*v520*/, 1.0, v233 /*v489*/
	v_dual_add_f32 v9 /*v521*/, 1.0, v234 /*v490*/ :: v_dual_add_f32 v10 /*v522*/, 1.0, v235 /*v491*/
	v_dual_add_f32 v11 /*v523*/, 1.0, v236 /*v492*/ :: v_dual_add_f32 v12 /*v524*/, 1.0, v237 /*v493*/
	v_dual_add_f32 v13 /*v525*/, 1.0, v238 /*v494*/ :: v_dual_add_f32 v14 /*v526*/, 1.0, v239 /*v495*/
	v_dual_add_f32 v15 /*v527*/, 1.0, v240 /*v496*/ :: v_dual_add_f32 v16 /*v528*/, 1.0, v241 /*v497*/
	v_dual_add_f32 v17 /*v529*/, 1.0, v242 /*v498*/ :: v_dual_add_f32 v18 /*v530*/, 1.0, v243 /*v499*/
	v_dual_add_f32 v19 /*v531*/, 1.0, v244 /*v500*/ :: v_dual_add_f32 v20 /*v532*/, 1.0, v245 /*v501*/
	v_dual_add_f32 v21 /*v533*/, 1.0, v246 /*v502*/ :: v_dual_add_f32 v22 /*v534*/, 1.0, v247 /*v503*/
	v_dual_add_f32 v23 /*v535*/, 1.0, v248 /*v504*/ :: v_dual_add_f32 v24 /*v536*/, 1.0, v249 /*v505*/
	v_dual_add_f32 v25 /*v537*/, 1.0, v250 /*v506*/ :: v_dual_add_f32 v26 /*v538*/, 1.0, v251 /*v507*/
	v_dual_add_f32 v27 /*v539*/, 1.0, v252 /*v508*/ :: v_dual_add_f32 v28 /*v540*/, 1.0, v253 /*v509*/
	v_dual_add_f32 v29 /*v541*/, 1.0, v254 /*v510*/ :: v_dual_add_f32 v30 /*v542*/, 1.0, v255 /*v511*/
	s_set_vgpr_msb 0x8488
	v_dual_add_f32 v31 /*v543*/, 1.0, v0 /*v512*/ :: v_dual_add_f32 v32 /*v544*/, 1.0, v1 /*v513*/
	s_set_vgpr_msb 0x8840
	v_rcp_f32_e32 v228 /*v484*/, v1
	v_rcp_f32_e32 v229 /*v485*/, v19
	s_set_vgpr_msb 0x4042
	v_rcp_f32_e32 v230 /*v486*/, v2 /*v514*/
	v_rcp_f32_e32 v231 /*v487*/, v3 /*v515*/
	v_rcp_f32_e32 v232 /*v488*/, v5 /*v517*/
	v_rcp_f32_e32 v233 /*v489*/, v6 /*v518*/
	v_rcp_f32_e32 v234 /*v490*/, v7 /*v519*/
	v_rcp_f32_e32 v235 /*v491*/, v8 /*v520*/
	v_rcp_f32_e32 v236 /*v492*/, v9 /*v521*/
	v_rcp_f32_e32 v237 /*v493*/, v10 /*v522*/
	v_rcp_f32_e32 v238 /*v494*/, v11 /*v523*/
	v_rcp_f32_e32 v239 /*v495*/, v12 /*v524*/
	v_rcp_f32_e32 v240 /*v496*/, v13 /*v525*/
	v_rcp_f32_e32 v241 /*v497*/, v14 /*v526*/
	v_rcp_f32_e32 v242 /*v498*/, v15 /*v527*/
	v_rcp_f32_e32 v243 /*v499*/, v16 /*v528*/
	v_rcp_f32_e32 v244 /*v500*/, v17 /*v529*/
	v_rcp_f32_e32 v245 /*v501*/, v18 /*v530*/
	v_rcp_f32_e32 v246 /*v502*/, v19 /*v531*/
	v_rcp_f32_e32 v247 /*v503*/, v20 /*v532*/
	v_rcp_f32_e32 v248 /*v504*/, v21 /*v533*/
	v_rcp_f32_e32 v249 /*v505*/, v22 /*v534*/
	v_rcp_f32_e32 v250 /*v506*/, v23 /*v535*/
	v_rcp_f32_e32 v251 /*v507*/, v24 /*v536*/
	v_rcp_f32_e32 v252 /*v508*/, v25 /*v537*/
	v_rcp_f32_e32 v253 /*v509*/, v26 /*v538*/
	v_rcp_f32_e32 v254 /*v510*/, v27 /*v539*/
	v_rcp_f32_e32 v255 /*v511*/, v28 /*v540*/
	s_set_vgpr_msb 0x4282
	v_rcp_f32_e32 v0 /*v512*/, v29 /*v541*/
	v_rcp_f32_e32 v1 /*v513*/, v30 /*v542*/
	v_rcp_f32_e32 v2 /*v514*/, v31 /*v543*/
	v_rcp_f32_e32 v3 /*v515*/, v32 /*v544*/
	s_set_vgpr_msb 0x8204
	v_pk_mul_f32 v[24:25], v[24:25], v[230:231] /*v[486:487]*/
	v_pk_mul_f32 v[20:21], v[20:21], v[228:229] /*v[484:485]*/
	s_set_vgpr_msb 0x40c
	v_lshlrev_b32_e32 v1, 2, v18 /*v786*/
	s_set_vgpr_msb 0xc08
	v_mul_lo_u32 v19, 0x88, v4 /*v516*/
	s_set_vgpr_msb 0x845
	v_max_num_f32_e32 v162 /*v418*/, v162 /*v418*/, v162 /*v418*/
	s_set_vgpr_msb 0x4500
	v_pk_mul_f32 v[24:25], v[26:27], v[24:25]
	v_pk_mul_f32 v[20:21], v[22:23], v[20:21]
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[22:23], v[32:33], v[234:235] /*v[490:491]*/
	v_pk_mul_f32 v[26:27], v[40:41], v[238:239] /*v[494:495]*/
	s_set_vgpr_msb 0x405
	v_pk_mul_f32 v[32:33], v[210:211] /*v[466:467]*/, v[244:245] /*v[500:501]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v20, v21
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[20:21], v[28:29], v[232:233] /*v[488:489]*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[22:23], v[34:35], v[22:23]
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[28:29], v[36:37], v[236:237] /*v[492:493]*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[26:27], v[42:43], v[26:27]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[34:35], v[222:223] /*v[478:479]*/, v[250:251] /*v[506:507]*/
	s_set_vgpr_msb 0x500
	v_pk_mul_f32 v[20:21], v[30:31], v[20:21]
	v_cvt_pk_bf16_f32 v23, v22, v23
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[30:31], v[48:49], v[242:243] /*v[498:499]*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[28:29], v[38:39], v[28:29]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[32:33], v[212:213] /*v[468:469]*/, v[32:33]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_cvt_pk_bf16_f32 v21, v26, v27
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[26:27], v[44:45], v[240:241] /*v[496:497]*/
	v_pk_mul_f32 v[30:31], v[30:31], v[226:227] /*v[482:483]*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v20, v28, v29
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[28:29], v[214:215] /*v[470:471]*/, v[246:247] /*v[502:503]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[34:35], v[224:225] /*v[480:481]*/, v[34:35]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[26:27], v[46:47], v[26:27]
	v_cvt_pk_bf16_f32 v31, v30, v31
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[36:37], v[194:195] /*v[450:451]*/, v[252:253] /*v[508:509]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[28:29], v[216:217] /*v[472:473]*/, v[28:29]
	s_set_vgpr_msb 0x109
	v_pk_mul_f32 v[38:39], v[206:207] /*v[462:463]*/, v[2:3] /*v[514:515]*/
	s_set_vgpr_msb 0x900
	v_cvt_pk_bf16_f32 v30, v26, v27
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[26:27], v[218:219] /*v[474:475]*/, v[248:249] /*v[504:505]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v35, v34, v35
	v_cvt_pk_bf16_f32 v29, v28, v29
	v_cvt_pk_bf16_f32 v28, v32, v33
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[198:199] /*v[454:455]*/, v[254:255] /*v[510:511]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[26:27], v[220:221] /*v[476:477]*/, v[26:27]
	s_set_vgpr_msb 0x109
	v_pk_mul_f32 v[40:41], v[202:203] /*v[458:459]*/, v[0:1] /*v[512:513]*/
	v_lshl_or_b32 v1, s46, 6, v1
	s_set_vgpr_msb 0x905
	v_max_num_f32_e32 v44, v186 /*v442*/, v186 /*v442*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[32:33], v[200:201] /*v[456:457]*/, v[32:33]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v34, v26, v27
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[26:27], v[196:197] /*v[452:453]*/, v[36:37]
	v_pk_mul_f32 v[36:37], v[208:209] /*v[464:465]*/, v[38:39]
	v_pk_mul_f32 v[38:39], v[204:205] /*v[460:461]*/, v[40:41]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v33, v32, v33
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v40, v178 /*v434*/, v178 /*v434*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v32, v26, v27
	v_cvt_pk_bf16_f32 v27, v36, v37
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v37, v180 /*v436*/, v180 /*v436*/
	s_set_vgpr_msb 0x545
	v_max_num_f32_e32 v180 /*v436*/, v164 /*v420*/, v164 /*v420*/
	v_med3_num_f32 v164 /*v420*/, v163 /*v419*/, -s26, s26
	s_set_vgpr_msb 0x4500
	v_add_lshl_u32 v19, v19, v1, 1
	v_cvt_pk_bf16_f32 v26, v38, v39
	v_min_num_f32_e32 v36, v40, v18
	s_set_vgpr_msb 0x41
	v_min_num_f32_e32 v163 /*v419*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v168 /*v424*/, v168 /*v424*/
	v_med3_num_f32 v168 /*v424*/, v167 /*v423*/, -s26, s26
	s_set_vgpr_msb 0x4505
	v_dual_max_num_f32 v40, v182 /*v438*/, v182 /*v438*/ :: v_dual_max_num_f32 v41, v184 /*v440*/, v184 /*v440*/
	v_med3_num_f32 v39, v181 /*v437*/, -s26, s26
	s_set_vgpr_msb 0x541
	v_min_num_f32_e32 v167 /*v423*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v172 /*v428*/, v172 /*v428*/
	v_med3_num_f32 v172 /*v428*/, v171 /*v427*/, -s26, s26
	s_set_vgpr_msb 0x4505
	v_dual_max_num_f32 v45, v188 /*v444*/, v188 /*v444*/ :: v_dual_max_num_f32 v48, v190 /*v446*/, v190 /*v446*/
	v_max_num_f32_e32 v49, v192 /*v448*/, v192 /*v448*/
	s_set_vgpr_msb 0x541
	v_min_num_f32_e32 v171 /*v427*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v176 /*v432*/, v176 /*v432*/
	v_med3_num_f32 v176 /*v432*/, v175 /*v431*/, -s26, s26
	v_max_num_f32_e32 v166 /*v422*/, v166 /*v422*/, v166 /*v422*/
	v_max_num_f32_e32 v170 /*v426*/, v170 /*v426*/, v170 /*v426*/
	v_max_num_f32_e32 v174 /*v430*/, v174 /*v430*/, v174 /*v430*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v175 /*v431*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v148 /*v404*/, v148 /*v404*/
	v_med3_num_f32 v148 /*v404*/, v147 /*v403*/, -s26, s26
	v_max_num_f32_e32 v146 /*v402*/, v146 /*v402*/, v146 /*v402*/
	v_max_num_f32_e32 v150 /*v406*/, v150 /*v406*/, v150 /*v406*/
	v_max_num_f32_e32 v154 /*v410*/, v154 /*v410*/, v154 /*v410*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v147 /*v403*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v152 /*v408*/, v152 /*v408*/
	v_med3_num_f32 v152 /*v408*/, v151 /*v407*/, -s26, s26
	v_max_num_f32_e32 v158 /*v414*/, v158 /*v414*/, v158 /*v414*/
	v_max_num_f32_e32 v130 /*v386*/, v130 /*v386*/, v130 /*v386*/
	v_max_num_f32_e32 v134 /*v390*/, v134 /*v390*/, v134 /*v390*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v151 /*v407*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v156 /*v412*/, v156 /*v412*/
	v_med3_num_f32 v156 /*v412*/, v155 /*v411*/, -s26, s26
	v_max_num_f32_e32 v138 /*v394*/, v138 /*v394*/, v138 /*v394*/
	v_dual_max_num_f32 v142 /*v398*/, v142 /*v398*/, v142 /*v398*/ :: v_dual_max_num_f32 v181 /*v437*/, v144 /*v400*/, v144 /*v400*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v155 /*v411*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v160 /*v416*/, v160 /*v416*/
	v_med3_num_f32 v160 /*v416*/, v159 /*v415*/, -s26, s26
	s_mov_b32 s12, 1
	s_set_vgpr_msb 0x4501
	v_med3_num_f32 v38, v179 /*v435*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v37, v37, v18
	s_set_vgpr_msb 0x41
	v_min_num_f32_e32 v159 /*v415*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v132 /*v388*/, v132 /*v388*/
	v_med3_num_f32 v132 /*v388*/, v131 /*v387*/, -s26, s26
	s_set_vgpr_msb 0x4500
	v_min_num_f32_e32 v40, v40, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v42, v183 /*v439*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v41, v41, v18
	s_set_vgpr_msb 0x41
	v_min_num_f32_e32 v131 /*v387*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v136 /*v392*/, v136 /*v392*/
	v_med3_num_f32 v136 /*v392*/, v135 /*v391*/, -s26, s26
	s_set_vgpr_msb 0x4501
	v_med3_num_f32 v43, v185 /*v441*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v44, v44, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v46, v187 /*v443*/, -s26, s26
	s_set_vgpr_msb 0x141
	v_min_num_f32_e32 v135 /*v391*/, v180 /*v436*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v180 /*v436*/, v140 /*v396*/, v140 /*v396*/
	s_set_vgpr_msb 0x4500
	v_min_num_f32_e32 v45, v45, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v47, v189 /*v445*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v48, v48, v18
	s_set_vgpr_msb 0x41
	v_med3_num_f32 v178 /*v434*/, v191 /*v447*/, -s26, s26
	s_set_vgpr_msb 0x4100
	v_min_num_f32_e32 v49, v49, v18
	s_set_vgpr_msb 0x41
	v_med3_num_f32 v179 /*v435*/, v193 /*v449*/, -s26, s26
	v_min_num_f32_e32 v162 /*v418*/, v162 /*v418*/, v18
	v_med3_num_f32 v165 /*v421*/, v165 /*v421*/, -s26, s26
	v_min_num_f32_e32 v166 /*v422*/, v166 /*v422*/, v18
	v_med3_num_f32 v169 /*v425*/, v169 /*v425*/, -s26, s26
	v_min_num_f32_e32 v170 /*v426*/, v170 /*v426*/, v18
	v_med3_num_f32 v173 /*v429*/, v173 /*v429*/, -s26, s26
	v_min_num_f32_e32 v174 /*v430*/, v174 /*v430*/, v18
	v_med3_num_f32 v177 /*v433*/, v177 /*v433*/, -s26, s26
	v_min_num_f32_e32 v146 /*v402*/, v146 /*v402*/, v18
	v_med3_num_f32 v149 /*v405*/, v149 /*v405*/, -s26, s26
	v_min_num_f32_e32 v150 /*v406*/, v150 /*v406*/, v18
	v_med3_num_f32 v153 /*v409*/, v153 /*v409*/, -s26, s26
	v_min_num_f32_e32 v154 /*v410*/, v154 /*v410*/, v18
	v_med3_num_f32 v157 /*v413*/, v157 /*v413*/, -s26, s26
	v_min_num_f32_e32 v158 /*v414*/, v158 /*v414*/, v18
	v_med3_num_f32 v161 /*v417*/, v161 /*v417*/, -s26, s26
	v_min_num_f32_e32 v130 /*v386*/, v130 /*v386*/, v18
	v_med3_num_f32 v133 /*v389*/, v133 /*v389*/, -s26, s26
	v_min_num_f32_e32 v134 /*v390*/, v134 /*v390*/, v18
	v_med3_num_f32 v137 /*v393*/, v137 /*v393*/, -s26, s26
	v_min_num_f32_e32 v138 /*v394*/, v138 /*v394*/, v18
	v_med3_num_f32 v140 /*v396*/, v139 /*v395*/, -s26, s26
	v_min_num_f32_e32 v139 /*v395*/, v180 /*v436*/, v18
	v_med3_num_f32 v141 /*v397*/, v141 /*v397*/, -s26, s26
	v_min_num_f32_e32 v142 /*v398*/, v142 /*v398*/, v18
	v_med3_num_f32 v144 /*v400*/, v143 /*v399*/, -s26, s26
	v_min_num_f32_e32 v143 /*v399*/, v181 /*v437*/, v18
	v_med3_num_f32 v145 /*v401*/, v145 /*v401*/, -s26, s26
	s_wait_alu depctr_va_vdst(0)
	s_set_vgpr_msb 0x4100
	ds_store_2addr_b64 v19, v[24:25], v[22:23] offset1:2
	ds_store_2addr_b64 v19, v[20:21], v[30:31] offset0:4 offset1:6
	ds_store_2addr_b64 v19, v[28:29], v[34:35] offset0:8 offset1:10
	ds_store_2addr_b64 v19, v[32:33], v[26:27] offset0:12 offset1:14
	s_wait_alu depctr_vm_vsrc(2)
	v_dual_mul_f32 v20, 0xbfb8aa3b, v36 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v37
	v_dual_mul_f32 v22, 0xbfb8aa3b, v40 :: v_dual_mul_f32 v23, 0xbfb8aa3b, v41
	v_dual_mul_f32 v24, 0xbfb8aa3b, v44 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v45
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mul_f32 v26, 0xbfb8aa3b, v48 :: v_dual_mul_f32 v27, 0xbfb8aa3b, v49
	s_set_vgpr_msb 4
	v_dual_mul_f32 v28, 0xbfb8aa3b, v162 /*v418*/ :: v_dual_mul_f32 v29, 0xbfb8aa3b, v163 /*v419*/
	v_dual_mul_f32 v30, 0xbfb8aa3b, v166 /*v422*/ :: v_dual_mul_f32 v31, 0xbfb8aa3b, v167 /*v423*/
	v_dual_mul_f32 v32, 0xbfb8aa3b, v170 /*v426*/ :: v_dual_mul_f32 v33, 0xbfb8aa3b, v171 /*v427*/
	v_dual_mul_f32 v34, 0xbfb8aa3b, v174 /*v430*/ :: v_dual_mul_f32 v35, 0xbfb8aa3b, v175 /*v431*/
	s_set_vgpr_msb 0x444
	v_dual_mul_f32 v180 /*v436*/, 0xbfb8aa3b, v146 /*v402*/ :: v_dual_mul_f32 v181 /*v437*/, 0xbfb8aa3b, v147 /*v403*/
	v_dual_mul_f32 v182 /*v438*/, 0xbfb8aa3b, v150 /*v406*/ :: v_dual_mul_f32 v183 /*v439*/, 0xbfb8aa3b, v151 /*v407*/
	v_dual_mul_f32 v184 /*v440*/, 0xbfb8aa3b, v154 /*v410*/ :: v_dual_mul_f32 v185 /*v441*/, 0xbfb8aa3b, v155 /*v411*/
	v_dual_mul_f32 v186 /*v442*/, 0xbfb8aa3b, v158 /*v414*/ :: v_dual_mul_f32 v187 /*v443*/, 0xbfb8aa3b, v159 /*v415*/
	v_dual_mul_f32 v188 /*v444*/, 0xbfb8aa3b, v130 /*v386*/ :: v_dual_mul_f32 v189 /*v445*/, 0xbfb8aa3b, v131 /*v387*/
	v_dual_mul_f32 v190 /*v446*/, 0xbfb8aa3b, v134 /*v390*/ :: v_dual_mul_f32 v191 /*v447*/, 0xbfb8aa3b, v135 /*v391*/
	v_dual_mul_f32 v192 /*v448*/, 0xbfb8aa3b, v138 /*v394*/ :: v_dual_mul_f32 v193 /*v449*/, 0xbfb8aa3b, v139 /*v395*/
	v_dual_mul_f32 v194 /*v450*/, 0xbfb8aa3b, v142 /*v398*/ :: v_dual_mul_f32 v195 /*v451*/, 0xbfb8aa3b, v143 /*v399*/
	s_set_vgpr_msb 0x4400
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v180 /*v436*/, v180 /*v436*/
	v_exp_f32_e32 v181 /*v437*/, v181 /*v437*/
	v_exp_f32_e32 v182 /*v438*/, v182 /*v438*/
	v_exp_f32_e32 v183 /*v439*/, v183 /*v439*/
	v_exp_f32_e32 v184 /*v440*/, v184 /*v440*/
	v_exp_f32_e32 v185 /*v441*/, v185 /*v441*/
	v_exp_f32_e32 v186 /*v442*/, v186 /*v442*/
	v_exp_f32_e32 v187 /*v443*/, v187 /*v443*/
	v_exp_f32_e32 v188 /*v444*/, v188 /*v444*/
	v_exp_f32_e32 v189 /*v445*/, v189 /*v445*/
	v_exp_f32_e32 v190 /*v446*/, v190 /*v446*/
	v_exp_f32_e32 v191 /*v447*/, v191 /*v447*/
	v_exp_f32_e32 v192 /*v448*/, v192 /*v448*/
	v_exp_f32_e32 v193 /*v449*/, v193 /*v449*/
	v_exp_f32_e32 v194 /*v450*/, v194 /*v450*/
	v_exp_f32_e32 v195 /*v451*/, v195 /*v451*/
	s_set_vgpr_msb 0x4100
	v_dual_add_f32 v20, 1.0, v20 :: v_dual_add_f32 v21, 1.0, v21
	v_dual_add_f32 v22, 1.0, v22 :: v_dual_add_f32 v23, 1.0, v23
	v_dual_add_f32 v24, 1.0, v24 :: v_dual_add_f32 v25, 1.0, v25
	v_dual_add_f32 v26, 1.0, v26 :: v_dual_add_f32 v27, 1.0, v27
	v_dual_add_f32 v28, 1.0, v28 :: v_dual_add_f32 v29, 1.0, v29
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v31, 1.0, v31
	v_dual_add_f32 v32, 1.0, v32 :: v_dual_add_f32 v33, 1.0, v33
	v_dual_add_f32 v34, 1.0, v34 :: v_dual_add_f32 v35, 1.0, v35
	s_set_vgpr_msb 0x44
	v_dual_add_f32 v180 /*v436*/, 1.0, v180 /*v436*/ :: v_dual_add_f32 v181 /*v437*/, 1.0, v181 /*v437*/
	v_dual_add_f32 v182 /*v438*/, 1.0, v182 /*v438*/ :: v_dual_add_f32 v183 /*v439*/, 1.0, v183 /*v439*/
	v_dual_add_f32 v184 /*v440*/, 1.0, v184 /*v440*/ :: v_dual_add_f32 v185 /*v441*/, 1.0, v185 /*v441*/
	v_dual_add_f32 v186 /*v442*/, 1.0, v186 /*v442*/ :: v_dual_add_f32 v187 /*v443*/, 1.0, v187 /*v443*/
	v_dual_add_f32 v188 /*v444*/, 1.0, v188 /*v444*/ :: v_dual_add_f32 v189 /*v445*/, 1.0, v189 /*v445*/
	v_dual_add_f32 v190 /*v446*/, 1.0, v190 /*v446*/ :: v_dual_add_f32 v191 /*v447*/, 1.0, v191 /*v447*/
	v_dual_add_f32 v192 /*v448*/, 1.0, v192 /*v448*/ :: v_dual_add_f32 v193 /*v449*/, 1.0, v193 /*v449*/
	v_dual_add_f32 v194 /*v450*/, 1.0, v194 /*v450*/ :: v_dual_add_f32 v195 /*v451*/, 1.0, v195 /*v451*/
	s_set_vgpr_msb 0x4400
	v_rcp_f32_e32 v20, v20
	v_rcp_f32_e32 v21, v21
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_rcp_f32_e32 v24, v24
	v_rcp_f32_e32 v25, v25
	v_rcp_f32_e32 v26, v26
	v_rcp_f32_e32 v27, v27
	v_rcp_f32_e32 v28, v28
	v_rcp_f32_e32 v29, v29
	v_rcp_f32_e32 v30, v30
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v32, v32
	v_rcp_f32_e32 v33, v33
	v_rcp_f32_e32 v34, v34
	v_rcp_f32_e32 v35, v35
	s_set_vgpr_msb 0x41
	v_rcp_f32_e32 v180 /*v436*/, v180 /*v436*/
	v_rcp_f32_e32 v181 /*v437*/, v181 /*v437*/
	v_rcp_f32_e32 v182 /*v438*/, v182 /*v438*/
	v_rcp_f32_e32 v183 /*v439*/, v183 /*v439*/
	v_rcp_f32_e32 v184 /*v440*/, v184 /*v440*/
	v_rcp_f32_e32 v185 /*v441*/, v185 /*v441*/
	v_rcp_f32_e32 v186 /*v442*/, v186 /*v442*/
	v_rcp_f32_e32 v187 /*v443*/, v187 /*v443*/
	v_rcp_f32_e32 v188 /*v444*/, v188 /*v444*/
	v_rcp_f32_e32 v189 /*v445*/, v189 /*v445*/
	v_rcp_f32_e32 v190 /*v446*/, v190 /*v446*/
	v_rcp_f32_e32 v191 /*v447*/, v191 /*v447*/
	v_rcp_f32_e32 v192 /*v448*/, v192 /*v448*/
	v_rcp_f32_e32 v193 /*v449*/, v193 /*v449*/
	v_rcp_f32_e32 v194 /*v450*/, v194 /*v450*/
	v_rcp_f32_e32 v195 /*v451*/, v195 /*v451*/
	s_set_vgpr_msb 0x4100
	v_pk_mul_f32 v[22:23], v[40:41], v[22:23]
	v_pk_mul_f32 v[20:21], v[36:37], v[20:21]
	v_pk_mul_f32 v[26:27], v[48:49], v[26:27]
	v_pk_mul_f32 v[24:25], v[44:45], v[24:25]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[30:31], v[166:167] /*v[422:423]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[42:43], v[22:23]
	v_pk_mul_f32 v[20:21], v[38:39], v[20:21]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[26:27], v[178:179] /*v[434:435]*/, v[26:27]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[46:47], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[36:37], v[130:131] /*v[386:387]*/, v[188:189] /*v[444:445]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[20:21], v[162:163] /*v[418:419]*/, v[28:29]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v27, v26, v27
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[28:29], v[168:169] /*v[424:425]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v26, v24, v25
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[24:25], v[174:175] /*v[430:431]*/, v[34:35]
	v_pk_mul_f32 v[20:21], v[164:165] /*v[420:421]*/, v[20:21]
	v_pk_mul_f32 v[30:31], v[170:171] /*v[426:427]*/, v[32:33]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v29, v28, v29
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[146:147] /*v[402:403]*/, v[180:181] /*v[436:437]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[24:25], v[176:177] /*v[432:433]*/, v[24:25]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v28, v20, v21
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[20:21], v[150:151] /*v[406:407]*/, v[182:183] /*v[438:439]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[30:31], v[172:173] /*v[428:429]*/, v[30:31]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[34:35], v[158:159] /*v[414:415]*/, v[186:187] /*v[442:443]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v25, v24, v25
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[32:33], v[148:149] /*v[404:405]*/, v[32:33]
	v_pk_mul_f32 v[20:21], v[152:153] /*v[408:409]*/, v[20:21]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v24, v30, v31
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[154:155] /*v[410:411]*/, v[184:185] /*v[440:441]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[34:35], v[160:161] /*v[416:417]*/, v[34:35]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[38:39], v[142:143] /*v[398:399]*/, v[194:195] /*v[450:451]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v21, v20, v21
	v_cvt_pk_bf16_f32 v20, v32, v33
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[134:135] /*v[390:391]*/, v[190:191] /*v[446:447]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[30:31], v[156:157] /*v[412:413]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v35, v34, v35
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[40:41], v[138:139] /*v[394:395]*/, v[192:193] /*v[448:449]*/
	s_set_vgpr_msb 0x540
	v_add_nc_u32_e32 v162 /*v418*/, 0x1000, v19
	s_set_vgpr_msb 0x4001
	v_pk_mul_f32 v[32:33], v[136:137] /*v[392:393]*/, v[32:33]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v34, v30, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[30:31], v[132:133] /*v[388:389]*/, v[36:37]
	v_pk_mul_f32 v[36:37], v[144:145] /*v[400:401]*/, v[38:39]
	v_pk_mul_f32 v[38:39], v[140:141] /*v[396:397]*/, v[40:41]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v33, v32, v33
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v40, v114 /*v370*/, v114 /*v370*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v36, v37
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v37, v116 /*v372*/, v116 /*v372*/
	s_set_vgpr_msb 0x545
	v_max_num_f32_e32 v116 /*v372*/, v100 /*v356*/, v100 /*v356*/
	v_med3_num_f32 v100 /*v356*/, v99 /*v355*/, -s26, s26
	s_set_vgpr_msb 0x4500
	v_cvt_pk_bf16_f32 v30, v38, v39
	v_min_num_f32_e32 v36, v40, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v40, v118 /*v374*/, v118 /*v374*/
	s_set_vgpr_msb 0x541
	v_min_num_f32_e32 v99 /*v355*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v104 /*v360*/, v104 /*v360*/
	v_med3_num_f32 v104 /*v360*/, v103 /*v359*/, -s26, s26
	s_set_vgpr_msb 0x4505
	v_max_num_f32_e32 v41, v120 /*v376*/, v120 /*v376*/
	v_med3_num_f32 v39, v117 /*v373*/, -s26, s26
	v_max_num_f32_e32 v44, v122 /*v378*/, v122 /*v378*/
	s_set_vgpr_msb 0x541
	v_min_num_f32_e32 v103 /*v359*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v108 /*v364*/, v108 /*v364*/
	v_med3_num_f32 v108 /*v364*/, v107 /*v363*/, -s26, s26
	s_set_vgpr_msb 0x4505
	v_dual_max_num_f32 v45, v124 /*v380*/, v124 /*v380*/ :: v_dual_max_num_f32 v48, v126 /*v382*/, v126 /*v382*/
	v_max_num_f32_e32 v49, v128 /*v384*/, v128 /*v384*/
	s_set_vgpr_msb 0x541
	v_min_num_f32_e32 v107 /*v363*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v112 /*v368*/, v112 /*v368*/
	v_med3_num_f32 v112 /*v368*/, v111 /*v367*/, -s26, s26
	v_max_num_f32_e32 v98 /*v354*/, v98 /*v354*/, v98 /*v354*/
	v_max_num_f32_e32 v102 /*v358*/, v102 /*v358*/, v102 /*v358*/
	v_max_num_f32_e32 v106 /*v362*/, v106 /*v362*/, v106 /*v362*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v111 /*v367*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v84 /*v340*/, v84 /*v340*/
	v_med3_num_f32 v84 /*v340*/, v83 /*v339*/, -s26, s26
	v_max_num_f32_e32 v110 /*v366*/, v110 /*v366*/, v110 /*v366*/
	v_max_num_f32_e32 v82 /*v338*/, v82 /*v338*/, v82 /*v338*/
	v_max_num_f32_e32 v86 /*v342*/, v86 /*v342*/, v86 /*v342*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v83 /*v339*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v88 /*v344*/, v88 /*v344*/
	v_med3_num_f32 v88 /*v344*/, v87 /*v343*/, -s26, s26
	v_max_num_f32_e32 v90 /*v346*/, v90 /*v346*/, v90 /*v346*/
	v_max_num_f32_e32 v94 /*v350*/, v94 /*v350*/, v94 /*v350*/
	v_max_num_f32_e32 v66 /*v322*/, v66 /*v322*/, v66 /*v322*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v87 /*v343*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v92 /*v348*/, v92 /*v348*/
	v_med3_num_f32 v92 /*v348*/, v91 /*v347*/, -s26, s26
	v_max_num_f32_e32 v70 /*v326*/, v70 /*v326*/, v70 /*v326*/
	v_max_num_f32_e32 v74 /*v330*/, v74 /*v330*/, v74 /*v330*/
	v_max_num_f32_e32 v78 /*v334*/, v78 /*v334*/, v78 /*v334*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v91 /*v347*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v96 /*v352*/, v96 /*v352*/
	v_med3_num_f32 v96 /*v352*/, v95 /*v351*/, -s26, s26
	v_max_num_f32_e32 v117 /*v373*/, v80 /*v336*/, v80 /*v336*/
	s_set_vgpr_msb 0x4501
	v_med3_num_f32 v38, v115 /*v371*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v37, v37, v18
	s_set_vgpr_msb 0x41
	v_min_num_f32_e32 v95 /*v351*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v68 /*v324*/, v68 /*v324*/
	v_med3_num_f32 v68 /*v324*/, v67 /*v323*/, -s26, s26
	s_set_vgpr_msb 0x4500
	v_min_num_f32_e32 v40, v40, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v42, v119 /*v375*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v41, v41, v18
	s_set_vgpr_msb 0x41
	v_min_num_f32_e32 v67 /*v323*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v72 /*v328*/, v72 /*v328*/
	v_med3_num_f32 v72 /*v328*/, v71 /*v327*/, -s26, s26
	s_set_vgpr_msb 0x4501
	v_med3_num_f32 v43, v121 /*v377*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v44, v44, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v46, v123 /*v379*/, -s26, s26
	s_set_vgpr_msb 0x141
	v_min_num_f32_e32 v71 /*v327*/, v116 /*v372*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v116 /*v372*/, v76 /*v332*/, v76 /*v332*/
	s_set_vgpr_msb 0x4500
	v_min_num_f32_e32 v45, v45, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v47, v125 /*v381*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v48, v48, v18
	s_set_vgpr_msb 0x41
	v_med3_num_f32 v114 /*v370*/, v127 /*v383*/, -s26, s26
	s_set_vgpr_msb 0x4100
	v_min_num_f32_e32 v49, v49, v18
	s_set_vgpr_msb 0x41
	v_med3_num_f32 v115 /*v371*/, v129 /*v385*/, -s26, s26
	v_min_num_f32_e32 v98 /*v354*/, v98 /*v354*/, v18
	v_med3_num_f32 v101 /*v357*/, v101 /*v357*/, -s26, s26
	v_min_num_f32_e32 v102 /*v358*/, v102 /*v358*/, v18
	v_med3_num_f32 v105 /*v361*/, v105 /*v361*/, -s26, s26
	v_min_num_f32_e32 v106 /*v362*/, v106 /*v362*/, v18
	v_med3_num_f32 v109 /*v365*/, v109 /*v365*/, -s26, s26
	v_min_num_f32_e32 v110 /*v366*/, v110 /*v366*/, v18
	v_med3_num_f32 v113 /*v369*/, v113 /*v369*/, -s26, s26
	v_min_num_f32_e32 v82 /*v338*/, v82 /*v338*/, v18
	v_med3_num_f32 v85 /*v341*/, v85 /*v341*/, -s26, s26
	v_min_num_f32_e32 v86 /*v342*/, v86 /*v342*/, v18
	v_med3_num_f32 v89 /*v345*/, v89 /*v345*/, -s26, s26
	v_min_num_f32_e32 v90 /*v346*/, v90 /*v346*/, v18
	v_med3_num_f32 v93 /*v349*/, v93 /*v349*/, -s26, s26
	v_min_num_f32_e32 v94 /*v350*/, v94 /*v350*/, v18
	v_med3_num_f32 v97 /*v353*/, v97 /*v353*/, -s26, s26
	v_min_num_f32_e32 v66 /*v322*/, v66 /*v322*/, v18
	v_med3_num_f32 v69 /*v325*/, v69 /*v325*/, -s26, s26
	v_min_num_f32_e32 v70 /*v326*/, v70 /*v326*/, v18
	v_med3_num_f32 v73 /*v329*/, v73 /*v329*/, -s26, s26
	v_min_num_f32_e32 v74 /*v330*/, v74 /*v330*/, v18
	v_med3_num_f32 v76 /*v332*/, v75 /*v331*/, -s26, s26
	v_min_num_f32_e32 v75 /*v331*/, v116 /*v372*/, v18
	v_med3_num_f32 v77 /*v333*/, v77 /*v333*/, -s26, s26
	v_min_num_f32_e32 v78 /*v334*/, v78 /*v334*/, v18
	v_med3_num_f32 v80 /*v336*/, v79 /*v335*/, -s26, s26
	v_min_num_f32_e32 v79 /*v335*/, v117 /*v373*/, v18
	v_med3_num_f32 v81 /*v337*/, v81 /*v337*/, -s26, s26
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v162 /*v418*/, v[22:23], v[26:27] offset0:32 offset1:34
	ds_store_2addr_b64 v162 /*v418*/, v[28:29], v[24:25] offset0:36 offset1:38
	ds_store_2addr_b64 v162 /*v418*/, v[20:21], v[34:35] offset0:40 offset1:42
	ds_store_2addr_b64 v162 /*v418*/, v[32:33], v[30:31] offset0:44 offset1:46
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4100
	v_dual_mul_f32 v20, 0xbfb8aa3b, v36 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v37
	v_dual_mul_f32 v22, 0xbfb8aa3b, v40 :: v_dual_mul_f32 v23, 0xbfb8aa3b, v41
	v_dual_mul_f32 v24, 0xbfb8aa3b, v44 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v45
	v_dual_mul_f32 v26, 0xbfb8aa3b, v48 :: v_dual_mul_f32 v27, 0xbfb8aa3b, v49
	s_set_vgpr_msb 4
	v_dual_mul_f32 v28, 0xbfb8aa3b, v98 /*v354*/ :: v_dual_mul_f32 v29, 0xbfb8aa3b, v99 /*v355*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mul_f32 v30, 0xbfb8aa3b, v102 /*v358*/ :: v_dual_mul_f32 v31, 0xbfb8aa3b, v103 /*v359*/
	v_dual_mul_f32 v32, 0xbfb8aa3b, v106 /*v362*/ :: v_dual_mul_f32 v33, 0xbfb8aa3b, v107 /*v363*/
	v_dual_mul_f32 v34, 0xbfb8aa3b, v110 /*v366*/ :: v_dual_mul_f32 v35, 0xbfb8aa3b, v111 /*v367*/
	s_set_vgpr_msb 0x444
	v_dual_mul_f32 v116 /*v372*/, 0xbfb8aa3b, v82 /*v338*/ :: v_dual_mul_f32 v117 /*v373*/, 0xbfb8aa3b, v83 /*v339*/
	v_dual_mul_f32 v118 /*v374*/, 0xbfb8aa3b, v86 /*v342*/ :: v_dual_mul_f32 v119 /*v375*/, 0xbfb8aa3b, v87 /*v343*/
	v_dual_mul_f32 v120 /*v376*/, 0xbfb8aa3b, v90 /*v346*/ :: v_dual_mul_f32 v121 /*v377*/, 0xbfb8aa3b, v91 /*v347*/
	v_dual_mul_f32 v122 /*v378*/, 0xbfb8aa3b, v94 /*v350*/ :: v_dual_mul_f32 v123 /*v379*/, 0xbfb8aa3b, v95 /*v351*/
	v_dual_mul_f32 v124 /*v380*/, 0xbfb8aa3b, v66 /*v322*/ :: v_dual_mul_f32 v125 /*v381*/, 0xbfb8aa3b, v67 /*v323*/
	v_dual_mul_f32 v126 /*v382*/, 0xbfb8aa3b, v70 /*v326*/ :: v_dual_mul_f32 v127 /*v383*/, 0xbfb8aa3b, v71 /*v327*/
	v_dual_mul_f32 v128 /*v384*/, 0xbfb8aa3b, v74 /*v330*/ :: v_dual_mul_f32 v129 /*v385*/, 0xbfb8aa3b, v75 /*v331*/
	v_dual_mul_f32 v130 /*v386*/, 0xbfb8aa3b, v78 /*v334*/ :: v_dual_mul_f32 v131 /*v387*/, 0xbfb8aa3b, v79 /*v335*/
	s_set_vgpr_msb 0x4400
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v116 /*v372*/, v116 /*v372*/
	v_exp_f32_e32 v117 /*v373*/, v117 /*v373*/
	v_exp_f32_e32 v118 /*v374*/, v118 /*v374*/
	v_exp_f32_e32 v119 /*v375*/, v119 /*v375*/
	v_exp_f32_e32 v120 /*v376*/, v120 /*v376*/
	v_exp_f32_e32 v121 /*v377*/, v121 /*v377*/
	v_exp_f32_e32 v122 /*v378*/, v122 /*v378*/
	v_exp_f32_e32 v123 /*v379*/, v123 /*v379*/
	v_exp_f32_e32 v124 /*v380*/, v124 /*v380*/
	v_exp_f32_e32 v125 /*v381*/, v125 /*v381*/
	v_exp_f32_e32 v126 /*v382*/, v126 /*v382*/
	v_exp_f32_e32 v127 /*v383*/, v127 /*v383*/
	v_exp_f32_e32 v128 /*v384*/, v128 /*v384*/
	v_exp_f32_e32 v129 /*v385*/, v129 /*v385*/
	v_exp_f32_e32 v130 /*v386*/, v130 /*v386*/
	v_exp_f32_e32 v131 /*v387*/, v131 /*v387*/
	s_set_vgpr_msb 0x4100
	v_dual_add_f32 v20, 1.0, v20 :: v_dual_add_f32 v21, 1.0, v21
	v_dual_add_f32 v22, 1.0, v22 :: v_dual_add_f32 v23, 1.0, v23
	v_dual_add_f32 v24, 1.0, v24 :: v_dual_add_f32 v25, 1.0, v25
	v_dual_add_f32 v26, 1.0, v26 :: v_dual_add_f32 v27, 1.0, v27
	v_dual_add_f32 v28, 1.0, v28 :: v_dual_add_f32 v29, 1.0, v29
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v31, 1.0, v31
	v_dual_add_f32 v32, 1.0, v32 :: v_dual_add_f32 v33, 1.0, v33
	v_dual_add_f32 v34, 1.0, v34 :: v_dual_add_f32 v35, 1.0, v35
	s_set_vgpr_msb 0x44
	v_dual_add_f32 v116 /*v372*/, 1.0, v116 /*v372*/ :: v_dual_add_f32 v117 /*v373*/, 1.0, v117 /*v373*/
	v_dual_add_f32 v118 /*v374*/, 1.0, v118 /*v374*/ :: v_dual_add_f32 v119 /*v375*/, 1.0, v119 /*v375*/
	v_dual_add_f32 v120 /*v376*/, 1.0, v120 /*v376*/ :: v_dual_add_f32 v121 /*v377*/, 1.0, v121 /*v377*/
	v_dual_add_f32 v122 /*v378*/, 1.0, v122 /*v378*/ :: v_dual_add_f32 v123 /*v379*/, 1.0, v123 /*v379*/
	v_dual_add_f32 v124 /*v380*/, 1.0, v124 /*v380*/ :: v_dual_add_f32 v125 /*v381*/, 1.0, v125 /*v381*/
	v_dual_add_f32 v126 /*v382*/, 1.0, v126 /*v382*/ :: v_dual_add_f32 v127 /*v383*/, 1.0, v127 /*v383*/
	v_dual_add_f32 v128 /*v384*/, 1.0, v128 /*v384*/ :: v_dual_add_f32 v129 /*v385*/, 1.0, v129 /*v385*/
	v_dual_add_f32 v130 /*v386*/, 1.0, v130 /*v386*/ :: v_dual_add_f32 v131 /*v387*/, 1.0, v131 /*v387*/
	s_set_vgpr_msb 0x4400
	v_rcp_f32_e32 v20, v20
	v_rcp_f32_e32 v21, v21
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_rcp_f32_e32 v24, v24
	v_rcp_f32_e32 v25, v25
	v_rcp_f32_e32 v26, v26
	v_rcp_f32_e32 v27, v27
	v_rcp_f32_e32 v28, v28
	v_rcp_f32_e32 v29, v29
	v_rcp_f32_e32 v30, v30
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v32, v32
	v_rcp_f32_e32 v33, v33
	v_rcp_f32_e32 v34, v34
	v_rcp_f32_e32 v35, v35
	s_set_vgpr_msb 0x41
	v_rcp_f32_e32 v116 /*v372*/, v116 /*v372*/
	v_rcp_f32_e32 v117 /*v373*/, v117 /*v373*/
	v_rcp_f32_e32 v118 /*v374*/, v118 /*v374*/
	v_rcp_f32_e32 v119 /*v375*/, v119 /*v375*/
	v_rcp_f32_e32 v120 /*v376*/, v120 /*v376*/
	v_rcp_f32_e32 v121 /*v377*/, v121 /*v377*/
	v_rcp_f32_e32 v122 /*v378*/, v122 /*v378*/
	v_rcp_f32_e32 v123 /*v379*/, v123 /*v379*/
	v_rcp_f32_e32 v124 /*v380*/, v124 /*v380*/
	v_rcp_f32_e32 v125 /*v381*/, v125 /*v381*/
	v_rcp_f32_e32 v126 /*v382*/, v126 /*v382*/
	v_rcp_f32_e32 v127 /*v383*/, v127 /*v383*/
	v_rcp_f32_e32 v128 /*v384*/, v128 /*v384*/
	v_rcp_f32_e32 v129 /*v385*/, v129 /*v385*/
	v_rcp_f32_e32 v130 /*v386*/, v130 /*v386*/
	v_rcp_f32_e32 v131 /*v387*/, v131 /*v387*/
	s_set_vgpr_msb 0x4100
	v_pk_mul_f32 v[22:23], v[40:41], v[22:23]
	v_pk_mul_f32 v[20:21], v[36:37], v[20:21]
	v_pk_mul_f32 v[26:27], v[48:49], v[26:27]
	v_pk_mul_f32 v[24:25], v[44:45], v[24:25]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[30:31], v[102:103] /*v[358:359]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[42:43], v[22:23]
	v_pk_mul_f32 v[20:21], v[38:39], v[20:21]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[26:27], v[114:115] /*v[370:371]*/, v[26:27]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[46:47], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[36:37], v[66:67] /*v[322:323]*/, v[124:125] /*v[380:381]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[20:21], v[98:99] /*v[354:355]*/, v[28:29]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v27, v26, v27
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[28:29], v[104:105] /*v[360:361]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v26, v24, v25
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[24:25], v[110:111] /*v[366:367]*/, v[34:35]
	v_pk_mul_f32 v[20:21], v[100:101] /*v[356:357]*/, v[20:21]
	v_pk_mul_f32 v[30:31], v[106:107] /*v[362:363]*/, v[32:33]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v29, v28, v29
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[82:83] /*v[338:339]*/, v[116:117] /*v[372:373]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[24:25], v[112:113] /*v[368:369]*/, v[24:25]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v28, v20, v21
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[20:21], v[86:87] /*v[342:343]*/, v[118:119] /*v[374:375]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[30:31], v[108:109] /*v[364:365]*/, v[30:31]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[34:35], v[94:95] /*v[350:351]*/, v[122:123] /*v[378:379]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v25, v24, v25
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[32:33], v[84:85] /*v[340:341]*/, v[32:33]
	v_pk_mul_f32 v[20:21], v[88:89] /*v[344:345]*/, v[20:21]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v24, v30, v31
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[30:31], v[90:91] /*v[346:347]*/, v[120:121] /*v[376:377]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[34:35], v[96:97] /*v[352:353]*/, v[34:35]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[38:39], v[78:79] /*v[334:335]*/, v[130:131] /*v[386:387]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v21, v20, v21
	v_cvt_pk_bf16_f32 v20, v32, v33
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[32:33], v[70:71] /*v[326:327]*/, v[126:127] /*v[382:383]*/
	s_set_vgpr_msb 0x501
	v_pk_mul_f32 v[30:31], v[92:93] /*v[348:349]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v35, v34, v35
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[40:41], v[74:75] /*v[330:331]*/, v[128:129] /*v[384:385]*/
	s_set_vgpr_msb 0x540
	v_add_nc_u32_e32 v98 /*v354*/, 0x2000, v19
	s_set_vgpr_msb 0x4001
	v_pk_mul_f32 v[32:33], v[72:73] /*v[328:329]*/, v[32:33]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v34, v30, v31
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[30:31], v[68:69] /*v[324:325]*/, v[36:37]
	v_pk_mul_f32 v[36:37], v[80:81] /*v[336:337]*/, v[38:39]
	v_pk_mul_f32 v[38:39], v[76:77] /*v[332:333]*/, v[40:41]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v33, v32, v33
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v40, v50 /*v306*/, v50 /*v306*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v36, v37
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v37, v52 /*v308*/, v52 /*v308*/
	s_set_vgpr_msb 0x545
	v_max_num_f32_e32 v52 /*v308*/, v36 /*v292*/, v36 /*v292*/
	v_med3_num_f32 v36 /*v292*/, v35 /*v291*/, -s26, s26
	s_set_vgpr_msb 0x4500
	v_cvt_pk_bf16_f32 v30, v38, v39
	v_min_num_f32_e32 v36, v40, v18
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v40, v54 /*v310*/, v54 /*v310*/
	s_set_vgpr_msb 0x541
	v_min_num_f32_e32 v35 /*v291*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v40 /*v296*/, v40 /*v296*/
	v_med3_num_f32 v40 /*v296*/, v39 /*v295*/, -s26, s26
	s_set_vgpr_msb 0x4505
	v_max_num_f32_e32 v41, v56 /*v312*/, v56 /*v312*/
	v_med3_num_f32 v39, v53 /*v309*/, -s26, s26
	v_max_num_f32_e32 v44, v58 /*v314*/, v58 /*v314*/
	s_set_vgpr_msb 0x541
	v_min_num_f32_e32 v39 /*v295*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v44 /*v300*/, v44 /*v300*/
	v_med3_num_f32 v44 /*v300*/, v43 /*v299*/, -s26, s26
	s_set_vgpr_msb 0x4505
	v_dual_max_num_f32 v45, v60 /*v316*/, v60 /*v316*/ :: v_dual_max_num_f32 v48, v62 /*v318*/, v62 /*v318*/
	v_max_num_f32_e32 v49, v64 /*v320*/, v64 /*v320*/
	s_set_vgpr_msb 0x541
	v_min_num_f32_e32 v43 /*v299*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v48 /*v304*/, v48 /*v304*/
	v_med3_num_f32 v48 /*v304*/, v47 /*v303*/, -s26, s26
	v_max_num_f32_e32 v34 /*v290*/, v34 /*v290*/, v34 /*v290*/
	v_max_num_f32_e32 v38 /*v294*/, v38 /*v294*/, v38 /*v294*/
	v_max_num_f32_e32 v42 /*v298*/, v42 /*v298*/, v42 /*v298*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v47 /*v303*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v20 /*v276*/, v20 /*v276*/
	v_med3_num_f32 v20 /*v276*/, v19 /*v275*/, -s26, s26
	v_max_num_f32_e32 v46 /*v302*/, v46 /*v302*/, v46 /*v302*/
	v_max_num_f32_e32 v18 /*v274*/, v18 /*v274*/, v18 /*v274*/
	v_max_num_f32_e32 v22 /*v278*/, v22 /*v278*/, v22 /*v278*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v19 /*v275*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v24 /*v280*/, v24 /*v280*/
	v_med3_num_f32 v24 /*v280*/, v23 /*v279*/, -s26, s26
	v_max_num_f32_e32 v26 /*v282*/, v26 /*v282*/, v26 /*v282*/
	v_max_num_f32_e32 v30 /*v286*/, v30 /*v286*/, v30 /*v286*/
	v_max_num_f32_e32 v2 /*v258*/, v2 /*v258*/, v2 /*v258*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v23 /*v279*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v28 /*v284*/, v28 /*v284*/
	v_med3_num_f32 v28 /*v284*/, v27 /*v283*/, -s26, s26
	v_max_num_f32_e32 v6 /*v262*/, v6 /*v262*/, v6 /*v262*/
	v_max_num_f32_e32 v10 /*v266*/, v10 /*v266*/, v10 /*v266*/
	v_max_num_f32_e32 v14 /*v270*/, v14 /*v270*/, v14 /*v270*/
	s_set_vgpr_msb 0x4541
	v_min_num_f32_e32 v27 /*v283*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v32 /*v288*/, v32 /*v288*/
	v_med3_num_f32 v32 /*v288*/, v31 /*v287*/, -s26, s26
	v_max_num_f32_e32 v53 /*v309*/, v16 /*v272*/, v16 /*v272*/
	s_set_vgpr_msb 0x4501
	v_med3_num_f32 v38, v51 /*v307*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v37, v37, v18
	s_set_vgpr_msb 0x41
	v_min_num_f32_e32 v31 /*v287*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v4 /*v260*/, v4 /*v260*/
	v_med3_num_f32 v4 /*v260*/, v3 /*v259*/, -s26, s26
	s_set_vgpr_msb 0x4500
	v_min_num_f32_e32 v40, v40, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v42, v55 /*v311*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v41, v41, v18
	s_set_vgpr_msb 0x41
	v_min_num_f32_e32 v3 /*v259*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v8 /*v264*/, v8 /*v264*/
	v_med3_num_f32 v8 /*v264*/, v7 /*v263*/, -s26, s26
	s_set_vgpr_msb 0x4501
	v_med3_num_f32 v43, v57 /*v313*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v44, v44, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v46, v59 /*v315*/, -s26, s26
	s_set_vgpr_msb 0x141
	v_min_num_f32_e32 v7 /*v263*/, v52 /*v308*/, v18
	s_set_vgpr_msb 0x4145
	v_max_num_f32_e32 v52 /*v308*/, v12 /*v268*/, v12 /*v268*/
	s_set_vgpr_msb 0x4500
	v_min_num_f32_e32 v45, v45, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v47, v61 /*v317*/, -s26, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v48, v48, v18
	s_set_vgpr_msb 0x41
	v_med3_num_f32 v50 /*v306*/, v63 /*v319*/, -s26, s26
	s_set_vgpr_msb 0x4100
	v_min_num_f32_e32 v49, v49, v18
	s_set_vgpr_msb 0x41
	v_med3_num_f32 v51 /*v307*/, v65 /*v321*/, -s26, s26
	v_min_num_f32_e32 v34 /*v290*/, v34 /*v290*/, v18
	v_med3_num_f32 v37 /*v293*/, v37 /*v293*/, -s26, s26
	v_min_num_f32_e32 v38 /*v294*/, v38 /*v294*/, v18
	v_med3_num_f32 v41 /*v297*/, v41 /*v297*/, -s26, s26
	v_min_num_f32_e32 v42 /*v298*/, v42 /*v298*/, v18
	v_med3_num_f32 v45 /*v301*/, v45 /*v301*/, -s26, s26
	v_min_num_f32_e32 v46 /*v302*/, v46 /*v302*/, v18
	v_med3_num_f32 v49 /*v305*/, v49 /*v305*/, -s26, s26
	v_min_num_f32_e32 v18 /*v274*/, v18 /*v274*/, v18
	v_med3_num_f32 v21 /*v277*/, v21 /*v277*/, -s26, s26
	v_min_num_f32_e32 v22 /*v278*/, v22 /*v278*/, v18
	v_med3_num_f32 v25 /*v281*/, v25 /*v281*/, -s26, s26
	v_min_num_f32_e32 v26 /*v282*/, v26 /*v282*/, v18
	v_med3_num_f32 v29 /*v285*/, v29 /*v285*/, -s26, s26
	v_min_num_f32_e32 v30 /*v286*/, v30 /*v286*/, v18
	v_med3_num_f32 v33 /*v289*/, v33 /*v289*/, -s26, s26
	v_min_num_f32_e32 v2 /*v258*/, v2 /*v258*/, v18
	v_med3_num_f32 v5 /*v261*/, v5 /*v261*/, -s26, s26
	v_min_num_f32_e32 v6 /*v262*/, v6 /*v262*/, v18
	v_med3_num_f32 v9 /*v265*/, v9 /*v265*/, -s26, s26
	v_min_num_f32_e32 v10 /*v266*/, v10 /*v266*/, v18
	v_med3_num_f32 v12 /*v268*/, v11 /*v267*/, -s26, s26
	v_min_num_f32_e32 v11 /*v267*/, v52 /*v308*/, v18
	v_med3_num_f32 v13 /*v269*/, v13 /*v269*/, -s26, s26
	v_min_num_f32_e32 v14 /*v270*/, v14 /*v270*/, v18
	v_med3_num_f32 v16 /*v272*/, v15 /*v271*/, -s26, s26
	v_min_num_f32_e32 v15 /*v271*/, v53 /*v309*/, v18
	v_med3_num_f32 v17 /*v273*/, v17 /*v273*/, -s26, s26
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v98 /*v354*/, v[22:23], v[26:27] offset0:64 offset1:66
	ds_store_2addr_b64 v98 /*v354*/, v[28:29], v[24:25] offset0:68 offset1:70
	ds_store_2addr_b64 v98 /*v354*/, v[20:21], v[34:35] offset0:72 offset1:74
	ds_store_2addr_b64 v98 /*v354*/, v[32:33], v[30:31] offset0:76 offset1:78
	s_wait_alu depctr_vm_vsrc(1)
	s_set_vgpr_msb 0x4100
	v_dual_mul_f32 v20, 0xbfb8aa3b, v36 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v37
	v_dual_mul_f32 v22, 0xbfb8aa3b, v40 :: v_dual_mul_f32 v23, 0xbfb8aa3b, v41
	v_dual_mul_f32 v24, 0xbfb8aa3b, v44 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v45
	v_dual_mul_f32 v26, 0xbfb8aa3b, v48 :: v_dual_mul_f32 v27, 0xbfb8aa3b, v49
	s_set_vgpr_msb 4
	v_dual_mul_f32 v28, 0xbfb8aa3b, v34 /*v290*/ :: v_dual_mul_f32 v29, 0xbfb8aa3b, v35 /*v291*/
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mul_f32 v30, 0xbfb8aa3b, v38 /*v294*/ :: v_dual_mul_f32 v31, 0xbfb8aa3b, v39 /*v295*/
	v_dual_mul_f32 v32, 0xbfb8aa3b, v42 /*v298*/ :: v_dual_mul_f32 v33, 0xbfb8aa3b, v43 /*v299*/
	v_dual_mul_f32 v34, 0xbfb8aa3b, v46 /*v302*/ :: v_dual_mul_f32 v35, 0xbfb8aa3b, v47 /*v303*/
	s_set_vgpr_msb 0x444
	v_dual_mul_f32 v52 /*v308*/, 0xbfb8aa3b, v18 /*v274*/ :: v_dual_mul_f32 v53 /*v309*/, 0xbfb8aa3b, v19 /*v275*/
	v_dual_mul_f32 v54 /*v310*/, 0xbfb8aa3b, v22 /*v278*/ :: v_dual_mul_f32 v55 /*v311*/, 0xbfb8aa3b, v23 /*v279*/
	v_dual_mul_f32 v56 /*v312*/, 0xbfb8aa3b, v26 /*v282*/ :: v_dual_mul_f32 v57 /*v313*/, 0xbfb8aa3b, v27 /*v283*/
	v_dual_mul_f32 v58 /*v314*/, 0xbfb8aa3b, v30 /*v286*/ :: v_dual_mul_f32 v59 /*v315*/, 0xbfb8aa3b, v31 /*v287*/
	v_dual_mul_f32 v60 /*v316*/, 0xbfb8aa3b, v2 /*v258*/ :: v_dual_mul_f32 v61 /*v317*/, 0xbfb8aa3b, v3 /*v259*/
	v_dual_mul_f32 v62 /*v318*/, 0xbfb8aa3b, v6 /*v262*/ :: v_dual_mul_f32 v63 /*v319*/, 0xbfb8aa3b, v7 /*v263*/
	v_dual_mul_f32 v64 /*v320*/, 0xbfb8aa3b, v10 /*v266*/ :: v_dual_mul_f32 v65 /*v321*/, 0xbfb8aa3b, v11 /*v267*/
	v_dual_mul_f32 v66 /*v322*/, 0xbfb8aa3b, v14 /*v270*/ :: v_dual_mul_f32 v67 /*v323*/, 0xbfb8aa3b, v15 /*v271*/
	s_set_vgpr_msb 0x4400
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v52 /*v308*/, v52 /*v308*/
	v_exp_f32_e32 v53 /*v309*/, v53 /*v309*/
	v_exp_f32_e32 v54 /*v310*/, v54 /*v310*/
	v_exp_f32_e32 v55 /*v311*/, v55 /*v311*/
	v_exp_f32_e32 v56 /*v312*/, v56 /*v312*/
	v_exp_f32_e32 v57 /*v313*/, v57 /*v313*/
	v_exp_f32_e32 v58 /*v314*/, v58 /*v314*/
	v_exp_f32_e32 v59 /*v315*/, v59 /*v315*/
	v_exp_f32_e32 v60 /*v316*/, v60 /*v316*/
	v_exp_f32_e32 v61 /*v317*/, v61 /*v317*/
	v_exp_f32_e32 v62 /*v318*/, v62 /*v318*/
	v_exp_f32_e32 v63 /*v319*/, v63 /*v319*/
	v_exp_f32_e32 v64 /*v320*/, v64 /*v320*/
	v_exp_f32_e32 v65 /*v321*/, v65 /*v321*/
	v_exp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_exp_f32_e32 v67 /*v323*/, v67 /*v323*/
	s_set_vgpr_msb 0x4100
	v_dual_add_f32 v20, 1.0, v20 :: v_dual_add_f32 v21, 1.0, v21
	v_dual_add_f32 v22, 1.0, v22 :: v_dual_add_f32 v23, 1.0, v23
	v_dual_add_f32 v24, 1.0, v24 :: v_dual_add_f32 v25, 1.0, v25
	v_dual_add_f32 v26, 1.0, v26 :: v_dual_add_f32 v27, 1.0, v27
	v_dual_add_f32 v28, 1.0, v28 :: v_dual_add_f32 v29, 1.0, v29
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v31, 1.0, v31
	v_dual_add_f32 v32, 1.0, v32 :: v_dual_add_f32 v33, 1.0, v33
	v_dual_add_f32 v34, 1.0, v34 :: v_dual_add_f32 v35, 1.0, v35
	s_set_vgpr_msb 0x44
	v_dual_add_f32 v52 /*v308*/, 1.0, v52 /*v308*/ :: v_dual_add_f32 v53 /*v309*/, 1.0, v53 /*v309*/
	v_dual_add_f32 v54 /*v310*/, 1.0, v54 /*v310*/ :: v_dual_add_f32 v55 /*v311*/, 1.0, v55 /*v311*/
	v_dual_add_f32 v56 /*v312*/, 1.0, v56 /*v312*/ :: v_dual_add_f32 v57 /*v313*/, 1.0, v57 /*v313*/
	v_dual_add_f32 v58 /*v314*/, 1.0, v58 /*v314*/ :: v_dual_add_f32 v59 /*v315*/, 1.0, v59 /*v315*/
	v_dual_add_f32 v60 /*v316*/, 1.0, v60 /*v316*/ :: v_dual_add_f32 v61 /*v317*/, 1.0, v61 /*v317*/
	v_dual_add_f32 v62 /*v318*/, 1.0, v62 /*v318*/ :: v_dual_add_f32 v63 /*v319*/, 1.0, v63 /*v319*/
	v_dual_add_f32 v64 /*v320*/, 1.0, v64 /*v320*/ :: v_dual_add_f32 v65 /*v321*/, 1.0, v65 /*v321*/
	v_dual_add_f32 v66 /*v322*/, 1.0, v66 /*v322*/ :: v_dual_add_f32 v67 /*v323*/, 1.0, v67 /*v323*/
	s_set_vgpr_msb 0x4400
	v_rcp_f32_e32 v20, v20
	v_rcp_f32_e32 v21, v21
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_rcp_f32_e32 v24, v24
	v_rcp_f32_e32 v25, v25
	v_rcp_f32_e32 v26, v26
	v_rcp_f32_e32 v27, v27
	v_rcp_f32_e32 v28, v28
	v_rcp_f32_e32 v29, v29
	v_rcp_f32_e32 v30, v30
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v32, v32
	v_rcp_f32_e32 v33, v33
	v_rcp_f32_e32 v34, v34
	v_rcp_f32_e32 v35, v35
	s_set_vgpr_msb 0x41
	v_rcp_f32_e32 v52 /*v308*/, v52 /*v308*/
	v_rcp_f32_e32 v53 /*v309*/, v53 /*v309*/
	v_rcp_f32_e32 v54 /*v310*/, v54 /*v310*/
	v_rcp_f32_e32 v55 /*v311*/, v55 /*v311*/
	v_rcp_f32_e32 v56 /*v312*/, v56 /*v312*/
	v_rcp_f32_e32 v57 /*v313*/, v57 /*v313*/
	v_rcp_f32_e32 v58 /*v314*/, v58 /*v314*/
	v_rcp_f32_e32 v59 /*v315*/, v59 /*v315*/
	v_rcp_f32_e32 v60 /*v316*/, v60 /*v316*/
	v_rcp_f32_e32 v61 /*v317*/, v61 /*v317*/
	v_rcp_f32_e32 v62 /*v318*/, v62 /*v318*/
	v_rcp_f32_e32 v63 /*v319*/, v63 /*v319*/
	v_rcp_f32_e32 v64 /*v320*/, v64 /*v320*/
	v_rcp_f32_e32 v65 /*v321*/, v65 /*v321*/
	v_rcp_f32_e32 v66 /*v322*/, v66 /*v322*/
	v_rcp_f32_e32 v67 /*v323*/, v67 /*v323*/
	s_set_vgpr_msb 0x4100
	v_pk_mul_f32 v[22:23], v[40:41], v[22:23]
	v_pk_mul_f32 v[20:21], v[36:37], v[20:21]
	v_pk_mul_f32 v[26:27], v[48:49], v[26:27]
	v_pk_mul_f32 v[24:25], v[44:45], v[24:25]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[30:31], v[38:39] /*v[294:295]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[22:23], v[42:43], v[22:23]
	v_pk_mul_f32 v[20:21], v[38:39], v[20:21]
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[26:27], v[50:51] /*v[306:307]*/, v[26:27]
	s_set_vgpr_msb 0x100
	v_pk_mul_f32 v[24:25], v[46:47], v[24:25]
	s_set_vgpr_msb 5
	v_pk_mul_f32 v[36:37], v[30:31] /*v[286:287]*/, v[58:59] /*v[314:315]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_cvt_pk_bf16_f32 v21, v26, v27
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[26:27], v[34:35] /*v[290:291]*/, v[28:29]
	v_pk_mul_f32 v[28:29], v[40:41] /*v[296:297]*/, v[30:31]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v20, v24, v25
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[24:25], v[46:47] /*v[302:303]*/, v[34:35]
	v_pk_mul_f32 v[30:31], v[42:43] /*v[298:299]*/, v[32:33]
	v_pk_mul_f32 v[26:27], v[36:37] /*v[292:293]*/, v[26:27]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[32:33], v[22:23] /*v[278:279]*/, v[54:55] /*v[310:311]*/
	v_pk_mul_f32 v[34:35], v[18:19] /*v[274:275]*/, v[52:53] /*v[308:309]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v29, v28, v29
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[24:25], v[48:49] /*v[304:305]*/, v[24:25]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v28, v26, v27
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[26:27], v[44:45] /*v[300:301]*/, v[30:31]
	v_pk_mul_f32 v[30:31], v[24:25] /*v[280:281]*/, v[32:33]
	v_pk_mul_f32 v[32:33], v[20:21] /*v[276:277]*/, v[34:35]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[34:35], v[26:27] /*v[282:283]*/, v[56:57] /*v[312:313]*/
	s_set_vgpr_msb 0x500
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v26, v27
	v_cvt_pk_bf16_f32 v27, v30, v31
	v_cvt_pk_bf16_f32 v26, v32, v33
	s_set_vgpr_msb 1
	v_pk_mul_f32 v[30:31], v[28:29] /*v[284:285]*/, v[34:35]
	v_pk_mul_f32 v[32:33], v[32:33] /*v[288:289]*/, v[36:37]
	s_set_vgpr_msb 0x105
	v_pk_mul_f32 v[34:35], v[6:7] /*v[262:263]*/, v[62:63] /*v[318:319]*/
	v_pk_mul_f32 v[36:37], v[2:3] /*v[258:259]*/, v[60:61] /*v[316:317]*/
	v_pk_mul_f32 v[38:39], v[14:15] /*v[270:271]*/, v[66:67] /*v[322:323]*/
	v_pk_mul_f32 v[40:41], v[10:11] /*v[266:267]*/, v[64:65] /*v[320:321]*/
	s_set_vgpr_msb 0x501
	v_add_nc_u32_e32 v42, 0x3000, v19
	v_pk_mul_f32 v[34:35], v[8:9] /*v[264:265]*/, v[34:35]
	v_pk_mul_f32 v[36:37], v[4:5] /*v[260:261]*/, v[36:37]
	v_pk_mul_f32 v[38:39], v[16:17] /*v[272:273]*/, v[38:39]
	v_pk_mul_f32 v[40:41], v[12:13] /*v[268:269]*/, v[40:41]
	s_set_vgpr_msb 0x100
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v34, v35
	v_cvt_pk_bf16_f32 v30, v36, v37
	v_cvt_pk_bf16_f32 v35, v38, v39
	v_cvt_pk_bf16_f32 v34, v40, v41
	s_cmp_eq_u32 s46, 0
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v42, v[22:23], v[20:21] offset0:96 offset1:98
	s_cselect_b32 s17, -1, 0
	s_cmp_lg_u32 s46, 0
	ds_store_2addr_b64 v42, v[28:29], v[24:25] offset0:100 offset1:102
	ds_store_2addr_b64 v42, v[26:27], v[32:33] offset0:104 offset1:106
	ds_store_2addr_b64 v42, v[30:31], v[34:35] offset0:108 offset1:110
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_38
	s_lshr_b32 s1, s25, 31
	s_sub_co_i32 s4, s24, s0
	s_add_co_i32 s1, s25, s1
	s_max_i32 s9, s4, 0
	s_and_b32 s4, s1, -2
	s_ashr_i32 s6, s1, 1
	s_cmp_lg_u32 s25, s4
	s_mul_i32 s13, s33, 0x8800
	s_cselect_b32 s1, -1, 0
	s_mov_b32 s8, 64
	s_and_b32 s1, s27, s1
	s_mov_b32 s5, 0x800000
	s_wait_alu depctr_vm_vsrc(3)
	v_cndmask_b32_e64 v20, 0, 1, s1
	s_ashr_i32 s1, s0, 31
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s4, v20
	s_sub_co_i32 s18, s6, s4
	s_add_nc_u64 s[6:7], s[28:29], s[0:1]
	s_ashr_i32 s19, s18, 31
	s_lshr_b32 s1, s9, 16
	s_mul_u64 s[14:15], s[6:7], s[18:19]
	s_mov_b32 s4, 0x10000
	s_lshl_b64 s[14:15], s[14:15], 1
	s_lshl_b32 s6, s9, 16
	s_add_nc_u64 s[14:15], s[2:3], s[14:15]
	s_or_b32 s7, s1, 0x880000
	s_add_nc_u64 s[14:15], s[14:15], s[30:31]
	s_and_b32 s10, s19, 0xffff
	s_bitset1_b32 s15, 31
	s_mov_b32 s9, s18
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_store_from_lds s[12:15], s[4:11]
.LBB0_38:
	v_max_num_f32_e32 v37, v228, v228
	v_max_num_f32_e32 v228, v212, v212
	v_med3_num_f32 v212, v211, s16, s26
	s_wait_alu depctr_vm_vsrc(3)
	v_dual_max_num_f32 v21, v242, v242 :: v_dual_max_num_f32 v23, v244, v244
	s_wait_alu depctr_vm_vsrc(2)
	v_max_num_f32_e32 v24, v246, v246
	v_min_num_f32_e32 v211, v228, v18
	v_max_num_f32_e32 v228, v216, v216
	v_med3_num_f32 v216, v215, s16, s26
	v_dual_max_num_f32 v25, v248, v248 :: v_dual_max_num_f32 v28, v250, v250
	v_max_num_f32_e32 v29, v252, v252
	s_delay_alu instid0(VALU_DEP_4)
	v_min_num_f32_e32 v215, v228, v18
	v_max_num_f32_e32 v228, v220, v220
	v_med3_num_f32 v220, v219, s16, s26
	s_wait_alu depctr_vm_vsrc(1)
	v_max_num_f32_e32 v32, v254, v254
	s_set_vgpr_msb 5
	v_max_num_f32_e32 v33, v0 /*v256*/, v0 /*v256*/
	s_set_vgpr_msb 0x500
	v_max_num_f32_e32 v36, v226, v226
	v_min_num_f32_e32 v219, v228, v18
	v_max_num_f32_e32 v228, v224, v224
	v_med3_num_f32 v224, v223, s16, s26
	v_dual_max_num_f32 v40, v230, v230 :: v_dual_max_num_f32 v41, v232, v232
	v_med3_num_f32 v39, v229, s16, s26
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_min_num_f32_e32 v223, v228, v18
	v_max_num_f32_e32 v228, v196, v196
	v_med3_num_f32 v196, v195, s16, s26
	v_dual_max_num_f32 v44, v234, v234 :: v_dual_max_num_f32 v45, v236, v236
	v_max_num_f32_e32 v48, v238, v238
	v_min_num_f32_e32 v195, v228, v18
	v_max_num_f32_e32 v228, v200, v200
	v_dual_max_num_f32 v49, v240, v240 :: v_dual_max_num_f32 v210, v210, v210
	v_max_num_f32_e32 v214, v214, v214
	v_max_num_f32_e32 v218, v218, v218
	v_max_num_f32_e32 v222, v222, v222
	v_max_num_f32_e32 v194, v194, v194
	v_max_num_f32_e32 v198, v198, v198
	v_max_num_f32_e32 v202, v202, v202
	v_med3_num_f32 v200, v199, s16, s26
	v_min_num_f32_e32 v199, v228, v18
	v_dual_max_num_f32 v228, v204, v204 :: v_dual_max_num_f32 v206, v206, v206
	v_max_num_f32_e32 v229, v208, v208
	v_med3_num_f32 v20, v243, s16, s26
	v_dual_min_num_f32 v22, v21, v18 :: v_dual_min_num_f32 v23, v23, v18
	v_med3_num_f32 v21, v245, s16, s26
	v_min_num_f32_e32 v24, v24, v18
	v_med3_num_f32 v26, v247, s16, s26
	v_min_num_f32_e32 v25, v25, v18
	v_med3_num_f32 v27, v249, s16, s26
	v_min_num_f32_e32 v28, v28, v18
	s_wait_alu depctr_vm_vsrc(0)
	v_med3_num_f32 v30, v251, s16, s26
	v_min_num_f32_e32 v29, v29, v18
	v_med3_num_f32 v31, v253, s16, s26
	v_min_num_f32_e32 v32, v32, v18
	v_med3_num_f32 v34, v255, s16, s26
	v_min_num_f32_e32 v33, v33, v18
	s_set_vgpr_msb 1
	v_med3_num_f32 v35, v1 /*v257*/, s16, s26
	s_set_vgpr_msb 0x100
	v_min_num_f32_e32 v36, v36, v18
	v_med3_num_f32 v38, v227, s16, s26
	v_dual_min_num_f32 v37, v37, v18 :: v_dual_min_num_f32 v40, v40, v18
	v_med3_num_f32 v42, v231, s16, s26
	v_min_num_f32_e32 v41, v41, v18
	v_med3_num_f32 v43, v233, s16, s26
	v_min_num_f32_e32 v44, v44, v18
	v_med3_num_f32 v46, v235, s16, s26
	v_min_num_f32_e32 v45, v45, v18
	v_med3_num_f32 v47, v237, s16, s26
	v_min_num_f32_e32 v48, v48, v18
	v_med3_num_f32 v226, v239, s16, s26
	v_min_num_f32_e32 v49, v49, v18
	v_med3_num_f32 v227, v241, s16, s26
	v_min_num_f32_e32 v210, v210, v18
	v_med3_num_f32 v213, v213, s16, s26
	v_min_num_f32_e32 v214, v214, v18
	v_med3_num_f32 v217, v217, s16, s26
	v_min_num_f32_e32 v218, v218, v18
	v_med3_num_f32 v221, v221, s16, s26
	v_min_num_f32_e32 v222, v222, v18
	v_med3_num_f32 v225, v225, s16, s26
	v_min_num_f32_e32 v194, v194, v18
	v_med3_num_f32 v197, v197, s16, s26
	v_min_num_f32_e32 v198, v198, v18
	v_med3_num_f32 v201, v201, s16, s26
	v_min_num_f32_e32 v202, v202, v18
	v_med3_num_f32 v204, v203, s16, s26
	v_min_num_f32_e32 v203, v228, v18
	v_med3_num_f32 v205, v205, s16, s26
	v_min_num_f32_e32 v206, v206, v18
	v_med3_num_f32 v208, v207, s16, s26
	v_min_num_f32_e32 v207, v229, v18
	v_med3_num_f32 v209, v209, s16, s26
	v_dual_mul_f32 v228, 0xbfb8aa3b, v22 :: v_dual_mul_f32 v229, 0xbfb8aa3b, v23
	v_dual_mul_f32 v230, 0xbfb8aa3b, v24 :: v_dual_mul_f32 v231, 0xbfb8aa3b, v25
	v_dual_mul_f32 v232, 0xbfb8aa3b, v28 :: v_dual_mul_f32 v233, 0xbfb8aa3b, v29
	v_dual_mul_f32 v234, 0xbfb8aa3b, v32 :: v_dual_mul_f32 v235, 0xbfb8aa3b, v33
	v_dual_mul_f32 v236, 0xbfb8aa3b, v36 :: v_dual_mul_f32 v237, 0xbfb8aa3b, v37
	v_dual_mul_f32 v238, 0xbfb8aa3b, v40 :: v_dual_mul_f32 v239, 0xbfb8aa3b, v41
	v_dual_mul_f32 v240, 0xbfb8aa3b, v44 :: v_dual_mul_f32 v241, 0xbfb8aa3b, v45
	v_dual_mul_f32 v242, 0xbfb8aa3b, v48 :: v_dual_mul_f32 v243, 0xbfb8aa3b, v49
	v_dual_mul_f32 v244, 0xbfb8aa3b, v210 :: v_dual_mul_f32 v245, 0xbfb8aa3b, v211
	v_dual_mul_f32 v246, 0xbfb8aa3b, v214 :: v_dual_mul_f32 v247, 0xbfb8aa3b, v215
	v_dual_mul_f32 v248, 0xbfb8aa3b, v218 :: v_dual_mul_f32 v249, 0xbfb8aa3b, v219
	v_dual_mul_f32 v250, 0xbfb8aa3b, v222 :: v_dual_mul_f32 v251, 0xbfb8aa3b, v223
	v_dual_mul_f32 v252, 0xbfb8aa3b, v194 :: v_dual_mul_f32 v253, 0xbfb8aa3b, v195
	v_dual_mul_f32 v254, 0xbfb8aa3b, v198 :: v_dual_mul_f32 v255, 0xbfb8aa3b, v199
	s_set_vgpr_msb 64
	v_dual_mul_f32 v0 /*v256*/, 0xbfb8aa3b, v202 :: v_dual_mul_f32 v1 /*v257*/, 0xbfb8aa3b, v203
	v_dual_mul_f32 v2 /*v258*/, 0xbfb8aa3b, v206 :: v_dual_mul_f32 v3 /*v259*/, 0xbfb8aa3b, v207
	s_set_vgpr_msb 0x4000
	v_exp_f32_e32 v228, v228
	v_exp_f32_e32 v229, v229
	v_exp_f32_e32 v230, v230
	v_exp_f32_e32 v231, v231
	v_exp_f32_e32 v232, v232
	v_exp_f32_e32 v233, v233
	v_exp_f32_e32 v234, v234
	v_exp_f32_e32 v235, v235
	v_exp_f32_e32 v236, v236
	v_exp_f32_e32 v237, v237
	v_exp_f32_e32 v238, v238
	v_exp_f32_e32 v239, v239
	v_exp_f32_e32 v240, v240
	v_exp_f32_e32 v241, v241
	v_exp_f32_e32 v242, v242
	v_exp_f32_e32 v243, v243
	v_exp_f32_e32 v244, v244
	v_exp_f32_e32 v245, v245
	v_exp_f32_e32 v246, v246
	v_exp_f32_e32 v247, v247
	v_exp_f32_e32 v248, v248
	v_exp_f32_e32 v249, v249
	v_exp_f32_e32 v250, v250
	v_exp_f32_e32 v251, v251
	v_exp_f32_e32 v252, v252
	v_exp_f32_e32 v253, v253
	v_exp_f32_e32 v254, v254
	v_exp_f32_e32 v255, v255
	s_set_vgpr_msb 0x41
	v_exp_f32_e32 v0 /*v256*/, v0 /*v256*/
	v_exp_f32_e32 v1 /*v257*/, v1 /*v257*/
	v_exp_f32_e32 v2 /*v258*/, v2 /*v258*/
	v_exp_f32_e32 v3 /*v259*/, v3 /*v259*/
	s_set_vgpr_msb 0x4100
	v_dual_add_f32 v228, 1.0, v228 :: v_dual_add_f32 v229, 1.0, v229
	v_dual_add_f32 v230, 1.0, v230 :: v_dual_add_f32 v231, 1.0, v231
	v_dual_add_f32 v232, 1.0, v232 :: v_dual_add_f32 v233, 1.0, v233
	v_dual_add_f32 v234, 1.0, v234 :: v_dual_add_f32 v235, 1.0, v235
	v_dual_add_f32 v236, 1.0, v236 :: v_dual_add_f32 v237, 1.0, v237
	v_dual_add_f32 v238, 1.0, v238 :: v_dual_add_f32 v239, 1.0, v239
	v_dual_add_f32 v240, 1.0, v240 :: v_dual_add_f32 v241, 1.0, v241
	v_dual_add_f32 v242, 1.0, v242 :: v_dual_add_f32 v243, 1.0, v243
	v_dual_add_f32 v244, 1.0, v244 :: v_dual_add_f32 v245, 1.0, v245
	v_dual_add_f32 v246, 1.0, v246 :: v_dual_add_f32 v247, 1.0, v247
	v_dual_add_f32 v248, 1.0, v248 :: v_dual_add_f32 v249, 1.0, v249
	v_dual_add_f32 v250, 1.0, v250 :: v_dual_add_f32 v251, 1.0, v251
	v_dual_add_f32 v252, 1.0, v252 :: v_dual_add_f32 v253, 1.0, v253
	v_dual_add_f32 v254, 1.0, v254 :: v_dual_add_f32 v255, 1.0, v255
	s_set_vgpr_msb 0x44
	v_dual_add_f32 v0 /*v256*/, 1.0, v0 /*v256*/ :: v_dual_add_f32 v1 /*v257*/, 1.0, v1 /*v257*/
	v_dual_add_f32 v2 /*v258*/, 1.0, v2 /*v258*/ :: v_dual_add_f32 v3 /*v259*/, 1.0, v3 /*v259*/
	s_set_vgpr_msb 0x4400
	v_rcp_f32_e32 v228, v228
	v_rcp_f32_e32 v229, v229
	v_rcp_f32_e32 v230, v230
	v_rcp_f32_e32 v231, v231
	v_rcp_f32_e32 v232, v232
	v_rcp_f32_e32 v233, v233
	v_rcp_f32_e32 v234, v234
	v_rcp_f32_e32 v235, v235
	v_rcp_f32_e32 v236, v236
	v_rcp_f32_e32 v237, v237
	v_rcp_f32_e32 v238, v238
	v_rcp_f32_e32 v239, v239
	v_rcp_f32_e32 v240, v240
	v_rcp_f32_e32 v241, v241
	v_rcp_f32_e32 v242, v242
	v_rcp_f32_e32 v243, v243
	v_rcp_f32_e32 v244, v244
	v_rcp_f32_e32 v245, v245
	v_rcp_f32_e32 v246, v246
	v_rcp_f32_e32 v247, v247
	v_rcp_f32_e32 v248, v248
	v_rcp_f32_e32 v249, v249
	v_rcp_f32_e32 v250, v250
	v_rcp_f32_e32 v251, v251
	v_rcp_f32_e32 v252, v252
	v_rcp_f32_e32 v253, v253
	v_rcp_f32_e32 v254, v254
	v_rcp_f32_e32 v255, v255
	s_set_vgpr_msb 0x41
	v_rcp_f32_e32 v0 /*v256*/, v0 /*v256*/
	v_rcp_f32_e32 v1 /*v257*/, v1 /*v257*/
	v_rcp_f32_e32 v2 /*v258*/, v2 /*v258*/
	v_rcp_f32_e32 v3 /*v259*/, v3 /*v259*/
	s_set_vgpr_msb 0x4100
	v_pk_mul_f32 v[24:25], v[24:25], v[230:231]
	v_pk_mul_f32 v[22:23], v[22:23], v[228:229]
	v_pk_mul_f32 v[32:33], v[32:33], v[234:235]
	v_pk_mul_f32 v[28:29], v[28:29], v[232:233]
	v_add_nc_u32_e32 v228, 0x4000, v19
	v_pk_mul_f32 v[24:25], v[26:27], v[24:25]
	v_pk_mul_f32 v[20:21], v[20:21], v[22:23]
	v_pk_mul_f32 v[22:23], v[40:41], v[238:239]
	v_pk_mul_f32 v[26:27], v[34:35], v[32:33]
	v_pk_mul_f32 v[28:29], v[30:31], v[28:29]
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v20, v21
	v_pk_mul_f32 v[20:21], v[36:37], v[236:237]
	v_cvt_pk_bf16_f32 v27, v26, v27
	v_pk_mul_f32 v[22:23], v[42:43], v[22:23]
	v_cvt_pk_bf16_f32 v26, v28, v29
	v_pk_mul_f32 v[28:29], v[48:49], v[242:243]
	v_pk_mul_f32 v[20:21], v[38:39], v[20:21]
	v_pk_mul_f32 v[30:31], v[44:45], v[240:241]
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_pk_mul_f32 v[32:33], v[210:211], v[244:245]
	v_pk_mul_f32 v[28:29], v[226:227], v[28:29]
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_pk_mul_f32 v[20:21], v[214:215], v[246:247]
	v_pk_mul_f32 v[30:31], v[46:47], v[30:31]
	v_pk_mul_f32 v[34:35], v[222:223], v[250:251]
	v_cvt_pk_bf16_f32 v29, v28, v29
	v_pk_mul_f32 v[32:33], v[212:213], v[32:33]
	v_pk_mul_f32 v[20:21], v[216:217], v[20:21]
	v_cvt_pk_bf16_f32 v28, v30, v31
	v_pk_mul_f32 v[30:31], v[218:219], v[248:249]
	v_pk_mul_f32 v[34:35], v[224:225], v[34:35]
	v_pk_mul_f32 v[36:37], v[194:195], v[252:253]
	v_cvt_pk_bf16_f32 v21, v20, v21
	v_cvt_pk_bf16_f32 v20, v32, v33
	v_pk_mul_f32 v[32:33], v[198:199], v[254:255]
	v_pk_mul_f32 v[30:31], v[220:221], v[30:31]
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[38:39], v[206:207], v[2:3] /*v[258:259]*/
	s_set_vgpr_msb 0x400
	v_cvt_pk_bf16_f32 v35, v34, v35
	s_set_vgpr_msb 4
	v_pk_mul_f32 v[40:41], v[202:203], v[0:1] /*v[256:257]*/
	s_set_vgpr_msb 0x400
	v_pk_mul_f32 v[32:33], v[200:201], v[32:33]
	v_cvt_pk_bf16_f32 v34, v30, v31
	v_pk_mul_f32 v[30:31], v[196:197], v[36:37]
	v_pk_mul_f32 v[36:37], v[208:209], v[38:39]
	v_pk_mul_f32 v[38:39], v[204:205], v[40:41]
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_max_num_f32_e32 v40, v178, v178
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v36, v37
	v_max_num_f32_e32 v37, v180, v180
	v_max_num_f32_e32 v180, v164, v164
	v_med3_num_f32 v164, v163, s16, s26
	v_cvt_pk_bf16_f32 v30, v38, v39
	v_min_num_f32_e32 v36, v40, v18
	v_max_num_f32_e32 v40, v182, v182
	v_min_num_f32_e32 v163, v180, v18
	v_max_num_f32_e32 v180, v168, v168
	v_med3_num_f32 v168, v167, s16, s26
	v_max_num_f32_e32 v41, v184, v184
	v_med3_num_f32 v39, v181, s16, s26
	v_max_num_f32_e32 v44, v186, v186
	v_min_num_f32_e32 v167, v180, v18
	v_max_num_f32_e32 v180, v172, v172
	v_med3_num_f32 v172, v171, s16, s26
	v_dual_max_num_f32 v45, v188, v188 :: v_dual_max_num_f32 v48, v190, v190
	v_max_num_f32_e32 v49, v192, v192
	s_delay_alu instid0(VALU_DEP_4)
	v_min_num_f32_e32 v171, v180, v18
	v_max_num_f32_e32 v180, v176, v176
	v_med3_num_f32 v176, v175, s16, s26
	v_max_num_f32_e32 v162, v162, v162
	v_max_num_f32_e32 v166, v166, v166
	v_max_num_f32_e32 v170, v170, v170
	v_min_num_f32_e32 v175, v180, v18
	v_max_num_f32_e32 v180, v148, v148
	v_med3_num_f32 v148, v147, s16, s26
	v_max_num_f32_e32 v174, v174, v174
	v_max_num_f32_e32 v146, v146, v146
	v_max_num_f32_e32 v150, v150, v150
	v_min_num_f32_e32 v147, v180, v18
	v_max_num_f32_e32 v180, v152, v152
	v_med3_num_f32 v152, v151, s16, s26
	v_max_num_f32_e32 v154, v154, v154
	v_max_num_f32_e32 v158, v158, v158
	v_max_num_f32_e32 v130, v130, v130
	v_min_num_f32_e32 v151, v180, v18
	v_max_num_f32_e32 v180, v156, v156
	v_med3_num_f32 v156, v155, s16, s26
	v_max_num_f32_e32 v134, v134, v134
	v_max_num_f32_e32 v138, v138, v138
	v_max_num_f32_e32 v142, v142, v142
	v_min_num_f32_e32 v155, v180, v18
	v_max_num_f32_e32 v180, v160, v160
	v_med3_num_f32 v160, v159, s16, s26
	v_max_num_f32_e32 v181, v144, v144
	v_med3_num_f32 v38, v179, s16, s26
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_min_num_f32 v37, v37, v18 :: v_dual_min_num_f32 v159, v180, v18
	v_max_num_f32_e32 v180, v132, v132
	v_med3_num_f32 v132, v131, s16, s26
	v_min_num_f32_e32 v40, v40, v18
	v_med3_num_f32 v42, v183, s16, s26
	v_dual_min_num_f32 v41, v41, v18 :: v_dual_min_num_f32 v131, v180, v18
	v_max_num_f32_e32 v180, v136, v136
	v_med3_num_f32 v136, v135, s16, s26
	v_med3_num_f32 v43, v185, s16, s26
	v_min_num_f32_e32 v44, v44, v18
	v_med3_num_f32 v46, v187, s16, s26
	v_min_num_f32_e32 v135, v180, v18
	v_dual_max_num_f32 v180, v140, v140 :: v_dual_min_num_f32 v45, v45, v18
	v_med3_num_f32 v47, v189, s16, s26
	v_min_num_f32_e32 v48, v48, v18
	v_med3_num_f32 v178, v191, s16, s26
	v_min_num_f32_e32 v49, v49, v18
	v_med3_num_f32 v179, v193, s16, s26
	v_min_num_f32_e32 v162, v162, v18
	v_med3_num_f32 v165, v165, s16, s26
	v_min_num_f32_e32 v166, v166, v18
	v_med3_num_f32 v169, v169, s16, s26
	v_min_num_f32_e32 v170, v170, v18
	v_med3_num_f32 v173, v173, s16, s26
	v_min_num_f32_e32 v174, v174, v18
	v_med3_num_f32 v177, v177, s16, s26
	v_min_num_f32_e32 v146, v146, v18
	v_med3_num_f32 v149, v149, s16, s26
	v_min_num_f32_e32 v150, v150, v18
	v_med3_num_f32 v153, v153, s16, s26
	v_min_num_f32_e32 v154, v154, v18
	v_med3_num_f32 v157, v157, s16, s26
	v_min_num_f32_e32 v158, v158, v18
	v_med3_num_f32 v161, v161, s16, s26
	v_min_num_f32_e32 v130, v130, v18
	v_med3_num_f32 v133, v133, s16, s26
	v_min_num_f32_e32 v134, v134, v18
	v_med3_num_f32 v137, v137, s16, s26
	v_min_num_f32_e32 v138, v138, v18
	v_med3_num_f32 v140, v139, s16, s26
	v_min_num_f32_e32 v139, v180, v18
	v_med3_num_f32 v141, v141, s16, s26
	v_min_num_f32_e32 v142, v142, v18
	v_med3_num_f32 v144, v143, s16, s26
	v_min_num_f32_e32 v143, v181, v18
	v_med3_num_f32 v145, v145, s16, s26
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v228, v[24:25], v[26:27] offset0:128 offset1:130
	ds_store_2addr_b64 v228, v[22:23], v[28:29] offset0:132 offset1:134
	ds_store_2addr_b64 v228, v[20:21], v[34:35] offset0:136 offset1:138
	ds_store_2addr_b64 v228, v[32:33], v[30:31] offset0:140 offset1:142
	s_wait_alu depctr_vm_vsrc(1)
	v_dual_mul_f32 v20, 0xbfb8aa3b, v36 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v37
	v_dual_mul_f32 v22, 0xbfb8aa3b, v40 :: v_dual_mul_f32 v23, 0xbfb8aa3b, v41
	v_dual_mul_f32 v24, 0xbfb8aa3b, v44 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v45
	v_dual_mul_f32 v26, 0xbfb8aa3b, v48 :: v_dual_mul_f32 v27, 0xbfb8aa3b, v49
	v_dual_mul_f32 v28, 0xbfb8aa3b, v162 :: v_dual_mul_f32 v29, 0xbfb8aa3b, v163
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mul_f32 v30, 0xbfb8aa3b, v166 :: v_dual_mul_f32 v31, 0xbfb8aa3b, v167
	v_dual_mul_f32 v32, 0xbfb8aa3b, v170 :: v_dual_mul_f32 v33, 0xbfb8aa3b, v171
	v_dual_mul_f32 v34, 0xbfb8aa3b, v174 :: v_dual_mul_f32 v35, 0xbfb8aa3b, v175
	v_dual_mul_f32 v180, 0xbfb8aa3b, v146 :: v_dual_mul_f32 v181, 0xbfb8aa3b, v147
	v_dual_mul_f32 v182, 0xbfb8aa3b, v150 :: v_dual_mul_f32 v183, 0xbfb8aa3b, v151
	v_dual_mul_f32 v184, 0xbfb8aa3b, v154 :: v_dual_mul_f32 v185, 0xbfb8aa3b, v155
	v_dual_mul_f32 v186, 0xbfb8aa3b, v158 :: v_dual_mul_f32 v187, 0xbfb8aa3b, v159
	v_dual_mul_f32 v188, 0xbfb8aa3b, v130 :: v_dual_mul_f32 v189, 0xbfb8aa3b, v131
	v_dual_mul_f32 v190, 0xbfb8aa3b, v134 :: v_dual_mul_f32 v191, 0xbfb8aa3b, v135
	v_dual_mul_f32 v192, 0xbfb8aa3b, v138 :: v_dual_mul_f32 v193, 0xbfb8aa3b, v139
	v_dual_mul_f32 v194, 0xbfb8aa3b, v142 :: v_dual_mul_f32 v195, 0xbfb8aa3b, v143
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_exp_f32_e32 v180, v180
	v_exp_f32_e32 v181, v181
	v_exp_f32_e32 v182, v182
	v_exp_f32_e32 v183, v183
	v_exp_f32_e32 v184, v184
	v_exp_f32_e32 v185, v185
	v_exp_f32_e32 v186, v186
	v_exp_f32_e32 v187, v187
	v_exp_f32_e32 v188, v188
	v_exp_f32_e32 v189, v189
	v_exp_f32_e32 v190, v190
	v_exp_f32_e32 v191, v191
	v_exp_f32_e32 v192, v192
	v_exp_f32_e32 v193, v193
	v_exp_f32_e32 v194, v194
	v_exp_f32_e32 v195, v195
	v_dual_add_f32 v20, 1.0, v20 :: v_dual_add_f32 v21, 1.0, v21
	v_dual_add_f32 v22, 1.0, v22 :: v_dual_add_f32 v23, 1.0, v23
	v_dual_add_f32 v24, 1.0, v24 :: v_dual_add_f32 v25, 1.0, v25
	v_dual_add_f32 v26, 1.0, v26 :: v_dual_add_f32 v27, 1.0, v27
	v_dual_add_f32 v28, 1.0, v28 :: v_dual_add_f32 v29, 1.0, v29
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v31, 1.0, v31
	v_dual_add_f32 v32, 1.0, v32 :: v_dual_add_f32 v33, 1.0, v33
	v_dual_add_f32 v34, 1.0, v34 :: v_dual_add_f32 v35, 1.0, v35
	v_dual_add_f32 v180, 1.0, v180 :: v_dual_add_f32 v181, 1.0, v181
	v_dual_add_f32 v182, 1.0, v182 :: v_dual_add_f32 v183, 1.0, v183
	v_dual_add_f32 v184, 1.0, v184 :: v_dual_add_f32 v185, 1.0, v185
	v_dual_add_f32 v186, 1.0, v186 :: v_dual_add_f32 v187, 1.0, v187
	v_dual_add_f32 v188, 1.0, v188 :: v_dual_add_f32 v189, 1.0, v189
	v_dual_add_f32 v190, 1.0, v190 :: v_dual_add_f32 v191, 1.0, v191
	v_dual_add_f32 v192, 1.0, v192 :: v_dual_add_f32 v193, 1.0, v193
	v_dual_add_f32 v194, 1.0, v194 :: v_dual_add_f32 v195, 1.0, v195
	v_rcp_f32_e32 v20, v20
	v_rcp_f32_e32 v21, v21
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_rcp_f32_e32 v24, v24
	v_rcp_f32_e32 v25, v25
	v_rcp_f32_e32 v26, v26
	v_rcp_f32_e32 v27, v27
	v_rcp_f32_e32 v28, v28
	v_rcp_f32_e32 v29, v29
	v_rcp_f32_e32 v30, v30
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v32, v32
	v_rcp_f32_e32 v33, v33
	v_rcp_f32_e32 v34, v34
	v_rcp_f32_e32 v35, v35
	v_rcp_f32_e32 v180, v180
	v_rcp_f32_e32 v181, v181
	v_rcp_f32_e32 v182, v182
	v_rcp_f32_e32 v183, v183
	v_rcp_f32_e32 v184, v184
	v_rcp_f32_e32 v185, v185
	v_rcp_f32_e32 v186, v186
	v_rcp_f32_e32 v187, v187
	v_rcp_f32_e32 v188, v188
	v_rcp_f32_e32 v189, v189
	v_rcp_f32_e32 v190, v190
	v_rcp_f32_e32 v191, v191
	v_rcp_f32_e32 v192, v192
	v_rcp_f32_e32 v193, v193
	v_rcp_f32_e32 v194, v194
	v_rcp_f32_e32 v195, v195
	v_pk_mul_f32 v[22:23], v[40:41], v[22:23]
	v_pk_mul_f32 v[20:21], v[36:37], v[20:21]
	v_pk_mul_f32 v[26:27], v[48:49], v[26:27]
	v_pk_mul_f32 v[24:25], v[44:45], v[24:25]
	v_pk_mul_f32 v[30:31], v[166:167], v[30:31]
	v_pk_mul_f32 v[22:23], v[42:43], v[22:23]
	v_pk_mul_f32 v[20:21], v[38:39], v[20:21]
	v_pk_mul_f32 v[26:27], v[178:179], v[26:27]
	v_pk_mul_f32 v[24:25], v[46:47], v[24:25]
	v_pk_mul_f32 v[36:37], v[130:131], v[188:189]
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_pk_mul_f32 v[20:21], v[162:163], v[28:29]
	v_cvt_pk_bf16_f32 v27, v26, v27
	v_pk_mul_f32 v[28:29], v[168:169], v[30:31]
	v_cvt_pk_bf16_f32 v26, v24, v25
	v_pk_mul_f32 v[24:25], v[174:175], v[34:35]
	v_pk_mul_f32 v[20:21], v[164:165], v[20:21]
	v_pk_mul_f32 v[30:31], v[170:171], v[32:33]
	v_cvt_pk_bf16_f32 v29, v28, v29
	v_pk_mul_f32 v[32:33], v[146:147], v[180:181]
	v_pk_mul_f32 v[24:25], v[176:177], v[24:25]
	v_cvt_pk_bf16_f32 v28, v20, v21
	v_pk_mul_f32 v[20:21], v[150:151], v[182:183]
	v_pk_mul_f32 v[30:31], v[172:173], v[30:31]
	v_pk_mul_f32 v[34:35], v[158:159], v[186:187]
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_pk_mul_f32 v[32:33], v[148:149], v[32:33]
	v_pk_mul_f32 v[20:21], v[152:153], v[20:21]
	v_cvt_pk_bf16_f32 v24, v30, v31
	v_pk_mul_f32 v[30:31], v[154:155], v[184:185]
	v_pk_mul_f32 v[34:35], v[160:161], v[34:35]
	v_pk_mul_f32 v[38:39], v[142:143], v[194:195]
	v_cvt_pk_bf16_f32 v21, v20, v21
	v_cvt_pk_bf16_f32 v20, v32, v33
	v_pk_mul_f32 v[32:33], v[134:135], v[190:191]
	v_pk_mul_f32 v[30:31], v[156:157], v[30:31]
	v_cvt_pk_bf16_f32 v35, v34, v35
	v_pk_mul_f32 v[40:41], v[138:139], v[192:193]
	v_add_nc_u32_e32 v162, 0x5000, v19
	v_pk_mul_f32 v[32:33], v[136:137], v[32:33]
	v_cvt_pk_bf16_f32 v34, v30, v31
	v_pk_mul_f32 v[30:31], v[132:133], v[36:37]
	v_pk_mul_f32 v[36:37], v[144:145], v[38:39]
	v_pk_mul_f32 v[38:39], v[140:141], v[40:41]
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_max_num_f32_e32 v40, v114, v114
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v31, v36, v37
	v_max_num_f32_e32 v37, v116, v116
	v_max_num_f32_e32 v116, v100, v100
	v_med3_num_f32 v100, v99, s16, s26
	v_cvt_pk_bf16_f32 v30, v38, v39
	v_min_num_f32_e32 v36, v40, v18
	v_max_num_f32_e32 v40, v118, v118
	v_min_num_f32_e32 v99, v116, v18
	v_max_num_f32_e32 v116, v104, v104
	v_med3_num_f32 v104, v103, s16, s26
	v_max_num_f32_e32 v41, v120, v120
	v_med3_num_f32 v39, v117, s16, s26
	v_max_num_f32_e32 v44, v122, v122
	v_min_num_f32_e32 v103, v116, v18
	v_max_num_f32_e32 v116, v108, v108
	v_med3_num_f32 v108, v107, s16, s26
	v_dual_max_num_f32 v45, v124, v124 :: v_dual_max_num_f32 v48, v126, v126
	v_max_num_f32_e32 v49, v128, v128
	s_delay_alu instid0(VALU_DEP_4)
	v_min_num_f32_e32 v107, v116, v18
	v_max_num_f32_e32 v116, v112, v112
	v_med3_num_f32 v112, v111, s16, s26
	v_max_num_f32_e32 v98, v98, v98
	v_max_num_f32_e32 v102, v102, v102
	v_max_num_f32_e32 v106, v106, v106
	v_min_num_f32_e32 v111, v116, v18
	v_max_num_f32_e32 v116, v84, v84
	v_med3_num_f32 v84, v83, s16, s26
	v_max_num_f32_e32 v110, v110, v110
	v_max_num_f32_e32 v82, v82, v82
	v_max_num_f32_e32 v86, v86, v86
	v_min_num_f32_e32 v83, v116, v18
	v_max_num_f32_e32 v116, v88, v88
	v_med3_num_f32 v88, v87, s16, s26
	v_max_num_f32_e32 v90, v90, v90
	v_max_num_f32_e32 v94, v94, v94
	v_max_num_f32_e32 v66, v66, v66
	v_min_num_f32_e32 v87, v116, v18
	v_max_num_f32_e32 v116, v92, v92
	v_med3_num_f32 v92, v91, s16, s26
	v_max_num_f32_e32 v70, v70, v70
	v_max_num_f32_e32 v74, v74, v74
	v_max_num_f32_e32 v78, v78, v78
	v_min_num_f32_e32 v91, v116, v18
	v_max_num_f32_e32 v116, v96, v96
	v_med3_num_f32 v96, v95, s16, s26
	v_max_num_f32_e32 v117, v80, v80
	v_med3_num_f32 v38, v115, s16, s26
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_dual_min_num_f32 v37, v37, v18 :: v_dual_min_num_f32 v95, v116, v18
	v_max_num_f32_e32 v116, v68, v68
	v_med3_num_f32 v68, v67, s16, s26
	v_min_num_f32_e32 v40, v40, v18
	v_med3_num_f32 v42, v119, s16, s26
	v_dual_min_num_f32 v41, v41, v18 :: v_dual_min_num_f32 v67, v116, v18
	v_max_num_f32_e32 v116, v72, v72
	v_med3_num_f32 v72, v71, s16, s26
	v_med3_num_f32 v43, v121, s16, s26
	v_min_num_f32_e32 v44, v44, v18
	v_med3_num_f32 v46, v123, s16, s26
	v_min_num_f32_e32 v71, v116, v18
	v_dual_max_num_f32 v116, v76, v76 :: v_dual_min_num_f32 v45, v45, v18
	v_med3_num_f32 v47, v125, s16, s26
	v_min_num_f32_e32 v48, v48, v18
	v_med3_num_f32 v114, v127, s16, s26
	v_min_num_f32_e32 v49, v49, v18
	v_med3_num_f32 v115, v129, s16, s26
	v_min_num_f32_e32 v98, v98, v18
	v_med3_num_f32 v101, v101, s16, s26
	v_min_num_f32_e32 v102, v102, v18
	v_med3_num_f32 v105, v105, s16, s26
	v_min_num_f32_e32 v106, v106, v18
	v_med3_num_f32 v109, v109, s16, s26
	v_min_num_f32_e32 v110, v110, v18
	v_med3_num_f32 v113, v113, s16, s26
	v_min_num_f32_e32 v82, v82, v18
	v_med3_num_f32 v85, v85, s16, s26
	v_min_num_f32_e32 v86, v86, v18
	v_med3_num_f32 v89, v89, s16, s26
	v_min_num_f32_e32 v90, v90, v18
	v_med3_num_f32 v93, v93, s16, s26
	v_min_num_f32_e32 v94, v94, v18
	v_med3_num_f32 v97, v97, s16, s26
	v_min_num_f32_e32 v66, v66, v18
	v_med3_num_f32 v69, v69, s16, s26
	v_min_num_f32_e32 v70, v70, v18
	v_med3_num_f32 v73, v73, s16, s26
	v_min_num_f32_e32 v74, v74, v18
	v_med3_num_f32 v76, v75, s16, s26
	v_min_num_f32_e32 v75, v116, v18
	v_med3_num_f32 v77, v77, s16, s26
	v_min_num_f32_e32 v78, v78, v18
	v_med3_num_f32 v80, v79, s16, s26
	v_min_num_f32_e32 v79, v117, v18
	v_med3_num_f32 v81, v81, s16, s26
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v162, v[22:23], v[26:27] offset0:160 offset1:162
	ds_store_2addr_b64 v162, v[28:29], v[24:25] offset0:164 offset1:166
	ds_store_2addr_b64 v162, v[20:21], v[34:35] offset0:168 offset1:170
	ds_store_2addr_b64 v162, v[32:33], v[30:31] offset0:172 offset1:174
	s_wait_alu depctr_vm_vsrc(1)
	v_dual_mul_f32 v20, 0xbfb8aa3b, v36 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v37
	v_dual_mul_f32 v22, 0xbfb8aa3b, v40 :: v_dual_mul_f32 v23, 0xbfb8aa3b, v41
	v_dual_mul_f32 v24, 0xbfb8aa3b, v44 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v45
	v_dual_mul_f32 v26, 0xbfb8aa3b, v48 :: v_dual_mul_f32 v27, 0xbfb8aa3b, v49
	v_dual_mul_f32 v28, 0xbfb8aa3b, v98 :: v_dual_mul_f32 v29, 0xbfb8aa3b, v99
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mul_f32 v30, 0xbfb8aa3b, v102 :: v_dual_mul_f32 v31, 0xbfb8aa3b, v103
	v_dual_mul_f32 v32, 0xbfb8aa3b, v106 :: v_dual_mul_f32 v33, 0xbfb8aa3b, v107
	v_dual_mul_f32 v34, 0xbfb8aa3b, v110 :: v_dual_mul_f32 v35, 0xbfb8aa3b, v111
	v_dual_mul_f32 v116, 0xbfb8aa3b, v82 :: v_dual_mul_f32 v117, 0xbfb8aa3b, v83
	v_dual_mul_f32 v118, 0xbfb8aa3b, v86 :: v_dual_mul_f32 v119, 0xbfb8aa3b, v87
	v_dual_mul_f32 v120, 0xbfb8aa3b, v90 :: v_dual_mul_f32 v121, 0xbfb8aa3b, v91
	v_dual_mul_f32 v122, 0xbfb8aa3b, v94 :: v_dual_mul_f32 v123, 0xbfb8aa3b, v95
	v_dual_mul_f32 v124, 0xbfb8aa3b, v66 :: v_dual_mul_f32 v125, 0xbfb8aa3b, v67
	v_dual_mul_f32 v126, 0xbfb8aa3b, v70 :: v_dual_mul_f32 v127, 0xbfb8aa3b, v71
	v_dual_mul_f32 v128, 0xbfb8aa3b, v74 :: v_dual_mul_f32 v129, 0xbfb8aa3b, v75
	v_dual_mul_f32 v130, 0xbfb8aa3b, v78 :: v_dual_mul_f32 v131, 0xbfb8aa3b, v79
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_exp_f32_e32 v116, v116
	v_exp_f32_e32 v117, v117
	v_exp_f32_e32 v118, v118
	v_exp_f32_e32 v119, v119
	v_exp_f32_e32 v120, v120
	v_exp_f32_e32 v121, v121
	v_exp_f32_e32 v122, v122
	v_exp_f32_e32 v123, v123
	v_exp_f32_e32 v124, v124
	v_exp_f32_e32 v125, v125
	v_exp_f32_e32 v126, v126
	v_exp_f32_e32 v127, v127
	v_exp_f32_e32 v128, v128
	v_exp_f32_e32 v129, v129
	v_exp_f32_e32 v130, v130
	v_exp_f32_e32 v131, v131
	v_dual_add_f32 v20, 1.0, v20 :: v_dual_add_f32 v21, 1.0, v21
	v_dual_add_f32 v22, 1.0, v22 :: v_dual_add_f32 v23, 1.0, v23
	v_dual_add_f32 v24, 1.0, v24 :: v_dual_add_f32 v25, 1.0, v25
	v_dual_add_f32 v26, 1.0, v26 :: v_dual_add_f32 v27, 1.0, v27
	v_dual_add_f32 v28, 1.0, v28 :: v_dual_add_f32 v29, 1.0, v29
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v31, 1.0, v31
	v_dual_add_f32 v32, 1.0, v32 :: v_dual_add_f32 v33, 1.0, v33
	v_dual_add_f32 v34, 1.0, v34 :: v_dual_add_f32 v35, 1.0, v35
	v_dual_add_f32 v116, 1.0, v116 :: v_dual_add_f32 v117, 1.0, v117
	v_dual_add_f32 v118, 1.0, v118 :: v_dual_add_f32 v119, 1.0, v119
	v_dual_add_f32 v120, 1.0, v120 :: v_dual_add_f32 v121, 1.0, v121
	v_dual_add_f32 v122, 1.0, v122 :: v_dual_add_f32 v123, 1.0, v123
	v_dual_add_f32 v124, 1.0, v124 :: v_dual_add_f32 v125, 1.0, v125
	v_dual_add_f32 v126, 1.0, v126 :: v_dual_add_f32 v127, 1.0, v127
	v_dual_add_f32 v128, 1.0, v128 :: v_dual_add_f32 v129, 1.0, v129
	v_dual_add_f32 v130, 1.0, v130 :: v_dual_add_f32 v131, 1.0, v131
	v_rcp_f32_e32 v20, v20
	v_rcp_f32_e32 v21, v21
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_rcp_f32_e32 v24, v24
	v_rcp_f32_e32 v25, v25
	v_rcp_f32_e32 v26, v26
	v_rcp_f32_e32 v27, v27
	v_rcp_f32_e32 v28, v28
	v_rcp_f32_e32 v29, v29
	v_rcp_f32_e32 v30, v30
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v32, v32
	v_rcp_f32_e32 v33, v33
	v_rcp_f32_e32 v34, v34
	v_rcp_f32_e32 v35, v35
	v_rcp_f32_e32 v116, v116
	v_rcp_f32_e32 v117, v117
	v_rcp_f32_e32 v118, v118
	v_rcp_f32_e32 v119, v119
	v_rcp_f32_e32 v120, v120
	v_rcp_f32_e32 v121, v121
	v_rcp_f32_e32 v122, v122
	v_rcp_f32_e32 v123, v123
	v_rcp_f32_e32 v124, v124
	v_rcp_f32_e32 v125, v125
	v_rcp_f32_e32 v126, v126
	v_rcp_f32_e32 v127, v127
	v_rcp_f32_e32 v128, v128
	v_rcp_f32_e32 v129, v129
	v_rcp_f32_e32 v130, v130
	v_rcp_f32_e32 v131, v131
	v_pk_mul_f32 v[22:23], v[40:41], v[22:23]
	v_pk_mul_f32 v[20:21], v[36:37], v[20:21]
	v_pk_mul_f32 v[26:27], v[48:49], v[26:27]
	v_pk_mul_f32 v[24:25], v[44:45], v[24:25]
	v_pk_mul_f32 v[30:31], v[102:103], v[30:31]
	v_pk_mul_f32 v[22:23], v[42:43], v[22:23]
	v_pk_mul_f32 v[20:21], v[38:39], v[20:21]
	v_pk_mul_f32 v[26:27], v[114:115], v[26:27]
	v_pk_mul_f32 v[24:25], v[46:47], v[24:25]
	v_pk_mul_f32 v[36:37], v[90:91], v[120:121]
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_cvt_pk_bf16_f32 v21, v26, v27
	v_pk_mul_f32 v[26:27], v[98:99], v[28:29]
	v_pk_mul_f32 v[28:29], v[104:105], v[30:31]
	v_cvt_pk_bf16_f32 v20, v24, v25
	v_pk_mul_f32 v[24:25], v[110:111], v[34:35]
	v_pk_mul_f32 v[30:31], v[106:107], v[32:33]
	v_pk_mul_f32 v[26:27], v[100:101], v[26:27]
	v_cvt_pk_bf16_f32 v29, v28, v29
	v_pk_mul_f32 v[32:33], v[86:87], v[118:119]
	v_pk_mul_f32 v[24:25], v[112:113], v[24:25]
	v_pk_mul_f32 v[30:31], v[108:109], v[30:31]
	v_cvt_pk_bf16_f32 v28, v26, v27
	v_pk_mul_f32 v[26:27], v[82:83], v[116:117]
	v_pk_mul_f32 v[32:33], v[88:89], v[32:33]
	v_pk_mul_f32 v[34:35], v[94:95], v[122:123]
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v30, v31
	v_pk_mul_f32 v[26:27], v[84:85], v[26:27]
	v_cvt_pk_bf16_f32 v31, v32, v33
	v_pk_mul_f32 v[32:33], v[96:97], v[34:35]
	v_pk_mul_f32 v[34:35], v[70:71], v[126:127]
	v_pk_mul_f32 v[38:39], v[78:79], v[130:131]
	v_cvt_pk_bf16_f32 v30, v26, v27
	v_pk_mul_f32 v[26:27], v[92:93], v[36:37]
	v_pk_mul_f32 v[36:37], v[66:67], v[124:125]
	v_pk_mul_f32 v[40:41], v[74:75], v[128:129]
	v_max_num_f32_e32 v84, v4, v4
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_pk_mul_f32 v[34:35], v[72:73], v[34:35]
	v_pk_mul_f32 v[36:37], v[68:69], v[36:37]
	v_cvt_pk_bf16_f32 v32, v26, v27
	v_pk_mul_f32 v[26:27], v[80:81], v[38:39]
	v_pk_mul_f32 v[38:39], v[76:77], v[40:41]
	v_med3_num_f32 v4, v3, s16, s26
	v_min_num_f32_e32 v3, v84, v18
	v_dual_max_num_f32 v84, v8, v8 :: v_dual_add_nc_u32 v19, 0x6000, v19
	v_cvt_pk_bf16_f32 v35, v34, v35
	v_cvt_pk_bf16_f32 v34, v36, v37
	v_max_num_f32_e32 v36, v50, v50
	v_cvt_pk_bf16_f32 v27, v26, v27
	v_cvt_pk_bf16_f32 v26, v38, v39
	v_dual_max_num_f32 v37, v52, v52 :: v_dual_max_num_f32 v40, v54, v54
	v_med3_num_f32 v38, v51, s16, s26
	v_max_num_f32_e32 v41, v56, v56
	v_med3_num_f32 v39, v53, s16, s26
	v_dual_max_num_f32 v44, v58, v58 :: v_dual_max_num_f32 v45, v60, v60
	v_med3_num_f32 v43, v57, s16, s26
	v_dual_max_num_f32 v48, v62, v62 :: v_dual_max_num_f32 v49, v64, v64
	v_med3_num_f32 v47, v61, s16, s26
	s_set_vgpr_msb 15
	v_dual_max_num_f32 v52, v20 /*v788*/, v20 /*v788*/ :: v_dual_max_num_f32 v53, v22 /*v790*/, v22 /*v790*/
	s_set_vgpr_msb 0xf00
	v_med3_num_f32 v51, v65, s16, s26
	s_set_vgpr_msb 15
	v_dual_max_num_f32 v56, v24 /*v792*/, v24 /*v792*/ :: v_dual_max_num_f32 v57, v26 /*v794*/, v26 /*v794*/
	v_dual_max_num_f32 v60, v28 /*v796*/, v28 /*v796*/ :: v_dual_max_num_f32 v61, v30 /*v798*/, v30 /*v798*/
	v_dual_max_num_f32 v64, v32 /*v800*/, v32 /*v800*/ :: v_dual_max_num_f32 v65, v34 /*v802*/, v34 /*v802*/
	s_set_vgpr_msb 0xf0a
	v_dual_max_num_f32 v68, v194 /*v706*/, v194 /*v706*/ :: v_dual_max_num_f32 v69, v196 /*v708*/, v196 /*v708*/
	v_dual_max_num_f32 v72, v198 /*v710*/, v198 /*v710*/ :: v_dual_max_num_f32 v73, v200 /*v712*/, v200 /*v712*/
	v_dual_max_num_f32 v76, v202 /*v714*/, v202 /*v714*/ :: v_dual_max_num_f32 v77, v204 /*v716*/, v204 /*v716*/
	v_dual_max_num_f32 v80, v206 /*v718*/, v206 /*v718*/ :: v_dual_max_num_f32 v81, v208 /*v720*/, v208 /*v720*/
	s_set_vgpr_msb 0xa00
	v_max_num_f32_e32 v2, v2, v2
	v_max_num_f32_e32 v6, v6, v6
	v_max_num_f32_e32 v10, v10, v10
	v_med3_num_f32 v8, v7, s16, s26
	v_min_num_f32_e32 v7, v84, v18
	v_dual_max_num_f32 v84, v12, v12 :: v_dual_max_num_f32 v14, v14, v14
	v_max_num_f32_e32 v85, v16, v16
	v_or3_b32 v0, v0, s0, 0x70
	v_dual_min_num_f32 v36, v36, v18 :: v_dual_min_num_f32 v37, v37, v18
	v_min_num_f32_e32 v40, v40, v18
	v_med3_num_f32 v42, v55, s16, s26
	v_dual_min_num_f32 v41, v41, v18 :: v_dual_min_num_f32 v44, v44, v18
	v_med3_num_f32 v46, v59, s16, s26
	v_dual_min_num_f32 v45, v45, v18 :: v_dual_min_num_f32 v48, v48, v18
	v_med3_num_f32 v50, v63, s16, s26
	v_dual_min_num_f32 v49, v49, v18 :: v_dual_min_num_f32 v52, v52, v18
	s_set_vgpr_msb 3
	v_med3_num_f32 v54, v21 /*v789*/, s16, s26
	s_set_vgpr_msb 0x300
	v_min_num_f32_e32 v53, v53, v18
	s_set_vgpr_msb 3
	v_med3_num_f32 v55, v23 /*v791*/, s16, s26
	s_set_vgpr_msb 0x300
	v_min_num_f32_e32 v56, v56, v18
	s_set_vgpr_msb 3
	v_med3_num_f32 v58, v25 /*v793*/, s16, s26
	s_set_vgpr_msb 0x300
	v_min_num_f32_e32 v57, v57, v18
	s_set_vgpr_msb 3
	v_med3_num_f32 v59, v27 /*v795*/, s16, s26
	s_set_vgpr_msb 0x300
	v_min_num_f32_e32 v60, v60, v18
	s_set_vgpr_msb 3
	v_med3_num_f32 v62, v29 /*v797*/, s16, s26
	s_set_vgpr_msb 0x300
	v_min_num_f32_e32 v61, v61, v18
	s_set_vgpr_msb 3
	v_med3_num_f32 v63, v31 /*v799*/, s16, s26
	s_set_vgpr_msb 0x300
	v_min_num_f32_e32 v64, v64, v18
	s_set_vgpr_msb 3
	v_med3_num_f32 v66, v33 /*v801*/, s16, s26
	s_set_vgpr_msb 0x300
	v_min_num_f32_e32 v65, v65, v18
	s_set_vgpr_msb 3
	v_med3_num_f32 v67, v35 /*v803*/, s16, s26
	s_set_vgpr_msb 0x300
	v_min_num_f32_e32 v68, v68, v18
	s_set_vgpr_msb 2
	v_med3_num_f32 v70, v195 /*v707*/, s16, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v69, v69, v18
	s_set_vgpr_msb 2
	v_med3_num_f32 v71, v197 /*v709*/, s16, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v72, v72, v18
	s_set_vgpr_msb 2
	v_med3_num_f32 v74, v199 /*v711*/, s16, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v73, v73, v18
	s_set_vgpr_msb 2
	v_med3_num_f32 v75, v201 /*v713*/, s16, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v76, v76, v18
	s_set_vgpr_msb 2
	v_med3_num_f32 v78, v203 /*v715*/, s16, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v77, v77, v18
	s_set_vgpr_msb 2
	v_med3_num_f32 v79, v205 /*v717*/, s16, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v80, v80, v18
	s_set_vgpr_msb 2
	v_med3_num_f32 v82, v207 /*v719*/, s16, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v81, v81, v18
	s_set_vgpr_msb 2
	v_med3_num_f32 v83, v209 /*v721*/, s16, s26
	s_set_vgpr_msb 0x200
	v_min_num_f32_e32 v2, v2, v18
	v_med3_num_f32 v5, v5, s16, s26
	v_min_num_f32_e32 v6, v6, v18
	v_med3_num_f32 v9, v9, s16, s26
	v_min_num_f32_e32 v10, v10, v18
	v_med3_num_f32 v12, v11, s16, s26
	v_min_num_f32_e32 v11, v84, v18
	v_med3_num_f32 v13, v13, s16, s26
	v_min_num_f32_e32 v14, v14, v18
	v_med3_num_f32 v16, v15, s16, s26
	v_min_num_f32_e32 v15, v85, v18
	v_med3_num_f32 v17, v17, s16, s26
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v19, v[22:23], v[20:21] offset0:192 offset1:194
	ds_store_2addr_b64 v19, v[28:29], v[24:25] offset0:196 offset1:198
	ds_store_2addr_b64 v19, v[30:31], v[32:33] offset0:200 offset1:202
	ds_store_2addr_b64 v19, v[34:35], v[26:27] offset0:204 offset1:206
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_mul_f32 v18, 0xbfb8aa3b, v36 :: v_dual_mul_f32 v19, 0xbfb8aa3b, v37
	v_dual_mul_f32 v20, 0xbfb8aa3b, v40 :: v_dual_mul_f32 v21, 0xbfb8aa3b, v41
	v_dual_mul_f32 v22, 0xbfb8aa3b, v44 :: v_dual_mul_f32 v23, 0xbfb8aa3b, v45
	v_dual_mul_f32 v24, 0xbfb8aa3b, v48 :: v_dual_mul_f32 v25, 0xbfb8aa3b, v49
	v_dual_mul_f32 v26, 0xbfb8aa3b, v52 :: v_dual_mul_f32 v27, 0xbfb8aa3b, v53
	v_dual_mul_f32 v28, 0xbfb8aa3b, v56 :: v_dual_mul_f32 v29, 0xbfb8aa3b, v57
	v_dual_mul_f32 v30, 0xbfb8aa3b, v60 :: v_dual_mul_f32 v31, 0xbfb8aa3b, v61
	v_dual_mul_f32 v32, 0xbfb8aa3b, v64 :: v_dual_mul_f32 v33, 0xbfb8aa3b, v65
	v_dual_mul_f32 v34, 0xbfb8aa3b, v68 :: v_dual_mul_f32 v35, 0xbfb8aa3b, v69
	v_dual_mul_f32 v84, 0xbfb8aa3b, v72 :: v_dual_mul_f32 v85, 0xbfb8aa3b, v73
	v_dual_mul_f32 v86, 0xbfb8aa3b, v76 :: v_dual_mul_f32 v87, 0xbfb8aa3b, v77
	v_dual_mul_f32 v88, 0xbfb8aa3b, v80 :: v_dual_mul_f32 v89, 0xbfb8aa3b, v81
	v_dual_mul_f32 v90, 0xbfb8aa3b, v2 :: v_dual_mul_f32 v91, 0xbfb8aa3b, v3
	v_dual_mul_f32 v92, 0xbfb8aa3b, v6 :: v_dual_mul_f32 v93, 0xbfb8aa3b, v7
	v_dual_mul_f32 v94, 0xbfb8aa3b, v10 :: v_dual_mul_f32 v95, 0xbfb8aa3b, v11
	v_dual_mul_f32 v96, 0xbfb8aa3b, v14 :: v_dual_mul_f32 v97, 0xbfb8aa3b, v15
	v_exp_f32_e32 v18, v18
	v_exp_f32_e32 v19, v19
	v_exp_f32_e32 v20, v20
	v_exp_f32_e32 v21, v21
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v23, v23
	v_exp_f32_e32 v24, v24
	v_exp_f32_e32 v25, v25
	v_exp_f32_e32 v26, v26
	v_exp_f32_e32 v27, v27
	v_exp_f32_e32 v28, v28
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v33, v33
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v35, v35
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v85, v85
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v87, v87
	v_exp_f32_e32 v88, v88
	v_exp_f32_e32 v89, v89
	v_exp_f32_e32 v90, v90
	v_exp_f32_e32 v91, v91
	v_exp_f32_e32 v92, v92
	v_exp_f32_e32 v93, v93
	v_exp_f32_e32 v94, v94
	v_exp_f32_e32 v95, v95
	v_exp_f32_e32 v96, v96
	v_exp_f32_e32 v97, v97
	v_dual_add_f32 v18, 1.0, v18 :: v_dual_add_f32 v19, 1.0, v19
	v_dual_add_f32 v20, 1.0, v20 :: v_dual_add_f32 v21, 1.0, v21
	v_dual_add_f32 v22, 1.0, v22 :: v_dual_add_f32 v23, 1.0, v23
	v_dual_add_f32 v24, 1.0, v24 :: v_dual_add_f32 v25, 1.0, v25
	v_dual_add_f32 v26, 1.0, v26 :: v_dual_add_f32 v27, 1.0, v27
	v_dual_add_f32 v28, 1.0, v28 :: v_dual_add_f32 v29, 1.0, v29
	v_dual_add_f32 v30, 1.0, v30 :: v_dual_add_f32 v31, 1.0, v31
	v_dual_add_f32 v32, 1.0, v32 :: v_dual_add_f32 v33, 1.0, v33
	v_dual_add_f32 v34, 1.0, v34 :: v_dual_add_f32 v35, 1.0, v35
	v_dual_add_f32 v84, 1.0, v84 :: v_dual_add_f32 v85, 1.0, v85
	v_dual_add_f32 v86, 1.0, v86 :: v_dual_add_f32 v87, 1.0, v87
	v_dual_add_f32 v88, 1.0, v88 :: v_dual_add_f32 v89, 1.0, v89
	v_dual_add_f32 v90, 1.0, v90 :: v_dual_add_f32 v91, 1.0, v91
	v_dual_add_f32 v92, 1.0, v92 :: v_dual_add_f32 v93, 1.0, v93
	v_dual_add_f32 v94, 1.0, v94 :: v_dual_add_f32 v95, 1.0, v95
	v_dual_add_f32 v96, 1.0, v96 :: v_dual_add_f32 v97, 1.0, v97
	v_rcp_f32_e32 v18, v18
	v_rcp_f32_e32 v19, v19
	v_rcp_f32_e32 v20, v20
	v_rcp_f32_e32 v21, v21
	v_rcp_f32_e32 v22, v22
	v_rcp_f32_e32 v23, v23
	v_rcp_f32_e32 v24, v24
	v_rcp_f32_e32 v25, v25
	v_rcp_f32_e32 v26, v26
	v_rcp_f32_e32 v27, v27
	v_rcp_f32_e32 v28, v28
	v_rcp_f32_e32 v29, v29
	v_rcp_f32_e32 v30, v30
	v_rcp_f32_e32 v31, v31
	v_rcp_f32_e32 v32, v32
	v_rcp_f32_e32 v33, v33
	v_rcp_f32_e32 v34, v34
	v_rcp_f32_e32 v35, v35
	v_rcp_f32_e32 v84, v84
	v_rcp_f32_e32 v85, v85
	v_rcp_f32_e32 v86, v86
	v_rcp_f32_e32 v87, v87
	v_rcp_f32_e32 v88, v88
	v_rcp_f32_e32 v89, v89
	v_rcp_f32_e32 v90, v90
	v_rcp_f32_e32 v91, v91
	v_rcp_f32_e32 v92, v92
	v_rcp_f32_e32 v93, v93
	v_rcp_f32_e32 v94, v94
	v_rcp_f32_e32 v95, v95
	v_rcp_f32_e32 v96, v96
	v_rcp_f32_e32 v97, v97
	v_pk_mul_f32 v[20:21], v[40:41], v[20:21]
	v_pk_mul_f32 v[18:19], v[36:37], v[18:19]
	v_mul_lo_u32 v0, 0x88, v0
	v_pk_mul_f32 v[24:25], v[48:49], v[24:25]
	v_pk_mul_f32 v[22:23], v[44:45], v[22:23]
	v_pk_mul_f32 v[20:21], v[42:43], v[20:21]
	v_pk_mul_f32 v[18:19], v[38:39], v[18:19]
	v_pk_mul_f32 v[28:29], v[56:57], v[28:29]
	v_pk_mul_f32 v[24:25], v[50:51], v[24:25]
	v_pk_mul_f32 v[6:7], v[6:7], v[92:93]
	v_cvt_pk_bf16_f32 v21, v20, v21
	v_cvt_pk_bf16_f32 v20, v18, v19
	v_add_lshl_u32 v36, v0, v1, 1
	v_pk_mul_f32 v[0:1], v[46:47], v[22:23]
	v_pk_mul_f32 v[18:19], v[52:53], v[26:27]
	v_pk_mul_f32 v[22:23], v[58:59], v[28:29]
	v_pk_mul_f32 v[26:27], v[64:65], v[32:33]
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v0, v1
	v_pk_mul_f32 v[0:1], v[54:55], v[18:19]
	v_cvt_pk_bf16_f32 v19, v22, v23
	v_pk_mul_f32 v[22:23], v[60:61], v[30:31]
	v_pk_mul_f32 v[26:27], v[66:67], v[26:27]
	v_pk_mul_f32 v[28:29], v[72:73], v[84:85]
	v_pk_mul_f32 v[30:31], v[68:69], v[34:35]
	v_cvt_pk_bf16_f32 v18, v0, v1
	v_pk_mul_f32 v[0:1], v[62:63], v[22:23]
	v_cvt_pk_bf16_f32 v23, v26, v27
	v_pk_mul_f32 v[26:27], v[74:75], v[28:29]
	v_pk_mul_f32 v[28:29], v[70:71], v[30:31]
	v_pk_mul_f32 v[30:31], v[76:77], v[86:87]
	v_pk_mul_f32 v[32:33], v[80:81], v[88:89]
	v_pk_mul_f32 v[2:3], v[2:3], v[90:91]
	v_pk_mul_f32 v[14:15], v[14:15], v[96:97]
	v_pk_mul_f32 v[10:11], v[10:11], v[94:95]
	v_cvt_pk_bf16_f32 v22, v0, v1
	v_cvt_pk_bf16_f32 v1, v26, v27
	v_cvt_pk_bf16_f32 v0, v28, v29
	v_pk_mul_f32 v[26:27], v[78:79], v[30:31]
	v_pk_mul_f32 v[28:29], v[82:83], v[32:33]
	v_pk_mul_f32 v[6:7], v[8:9], v[6:7]
	v_pk_mul_f32 v[2:3], v[4:5], v[2:3]
	v_pk_mul_f32 v[4:5], v[16:17], v[14:15]
	v_pk_mul_f32 v[8:9], v[12:13], v[10:11]
	v_cvt_pk_bf16_f32 v29, v28, v29
	v_cvt_pk_bf16_f32 v28, v26, v27
	v_cvt_pk_bf16_f32 v7, v6, v7
	v_cvt_pk_bf16_f32 v6, v2, v3
	v_cvt_pk_bf16_f32 v3, v4, v5
	v_cvt_pk_bf16_f32 v2, v8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s17
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v36, v[20:21], v[24:25] offset1:2
	ds_store_2addr_b64 v36, v[18:19], v[22:23] offset0:4 offset1:6
	ds_store_2addr_b64 v36, v[0:1], v[28:29] offset0:8 offset1:10
	ds_store_2addr_b64 v36, v[6:7], v[2:3] offset0:12 offset1:14
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_cbranch_vccnz .LBB0_40
	s_or_b32 s6, s0, 64
	s_lshr_b32 s0, s25, 31
	s_sub_co_i32 s1, s24, s6
	s_add_co_i32 s0, s25, s0
	s_max_i32 s9, s1, 0
	s_and_b32 s1, s0, -2
	s_ashr_i32 s10, s0, 1
	s_cmp_lg_u32 s25, s1
	s_mov_b32 s8, 64
	s_cselect_b32 s0, -1, 0
	s_ashr_i32 s7, s6, 31
	s_and_b32 s0, s27, s0
	s_add_nc_u64 s[14:15], s[28:29], s[6:7]
	s_wait_alu depctr_vm_vsrc(1)
	v_cndmask_b32_e64 v0, 0, 1, s0
	s_lshr_b32 s7, s9, 16
	s_mov_b32 s0, 1
	s_mov_b32 s5, 0x800000
	s_mov_b32 s4, 0x10000
	v_readfirstlane_b32 s1, v0
	s_or_b32 s7, s7, 0x880000
	s_sub_co_i32 s12, s10, s1
	s_mul_i32 s1, s6, 0x110
	s_ashr_i32 s13, s12, 31
	s_lshl_b32 s6, s9, 16
	s_mul_u64 s[14:15], s[14:15], s[12:13]
	s_and_b32 s10, s13, 0xffff
	s_lshl_b64 s[14:15], s[14:15], 1
	s_mov_b32 s9, s12
	s_add_nc_u64 s[2:3], s[2:3], s[14:15]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[2:3], s[2:3], s[30:31]
	s_bitset1_b32 s3, 31
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_store_from_lds s[0:3], s[4:11]
.LBB0_40:
	s_wait_tensorcnt 0x0
.LBB0_41:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p
		.amdhsa_group_segment_fixed_size 278528
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 184
		.amdhsa_user_sgpr_count 4
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 2
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 804
		.amdhsa_next_free_sgpr 56
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 255
		.amdhsa_round_robin_scheduling 0
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
	.size	a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p, .Lfunc_end0-a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p

	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.num_vgpr, 804
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.num_agpr, 0
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.numbered_sgpr, 56
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.num_named_barrier, 0
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.private_seg_size, 0
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.uses_vcc, 1
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.uses_flat_scratch, 0
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.has_dyn_sized_stack, 0
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.has_recursion, 0
	.set a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
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
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
      - .offset:         64
        .size:           28
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         112
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         120
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         128
        .size:           8
        .value_kind:     global_buffer
      - .offset:         136
        .size:           28
        .value_kind:     by_value
      - .offset:         164
        .size:           4
        .value_kind:     by_value
      - .offset:         168
        .size:           4
        .value_kind:     by_value
      - .offset:         172
        .size:           4
        .value_kind:     by_value
      - .offset:         176
        .size:           4
        .value_kind:     by_value
      - .offset:         180
        .size:           4
        .value_kind:     by_value
    .cluster_dims:
      - 4
      - 4
      - 1
    .group_segment_fixed_size: 278528
    .kernarg_segment_align: 8
    .kernarg_segment_size: 184
    .max_flat_workgroup_size: 128
    .name:           a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     804
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
