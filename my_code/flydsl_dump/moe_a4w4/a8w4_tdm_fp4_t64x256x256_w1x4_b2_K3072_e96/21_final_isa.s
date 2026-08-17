	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96
	.p2align	8
	.type	a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96,@function
a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[8:9], s[0:1], 0xa4 nv
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s9, 0xff
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
	s_add_co_i32 s5, s8, 63
	s_ashr_i32 s6, s5, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s6, s6, 26
	s_add_co_i32 s6, s5, s6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s7, s6, 0xffffffc0
	s_ashr_i32 s6, s6, 6
	s_cmp_lg_u32 s5, s7
	s_cselect_b32 s7, -1, 0
	s_cmp_lt_i32 s5, 0
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_cselect_b32 s8, -1, 0
	s_bfe_u32 s10, ttmp6, 0x4000c
	s_and_b32 s12, ttmp6, 15
	s_add_co_i32 s10, s10, 1
	s_lshl_b32 s11, s4, 4
	s_mul_i32 s10, ttmp9, s10
	s_and_b32 s7, s8, s7
	s_add_co_i32 s12, s12, s10
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s5, ttmp9, s12
	s_abs_i32 s8, s11
	s_abs_i32 s13, s5
	s_cvt_f32_u32 s10, s8
	s_sub_co_i32 s12, 0, s8
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s10
	v_nop
	v_readfirstlane_b32 s10, v1
	s_mul_f32 s10, s10, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s10, s10
	s_mul_i32 s12, s12, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s12, s10, s12
	s_add_co_i32 s10, s10, s12
	s_xor_b32 s12, s5, s11
	s_mul_hi_u32 s10, s13, s10
	s_ashr_i32 s12, s12, 31
	s_mul_i32 s14, s10, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s13, s13, s14
	s_add_co_i32 s14, s10, 1
	s_sub_co_i32 s15, s13, s8
	s_cmp_ge_u32 s13, s8
	s_cselect_b32 s10, s14, s10
	s_cselect_b32 s13, s15, s13
	s_add_co_i32 s14, s10, 1
	s_cmp_ge_u32 s13, s8
	s_cselect_b32 s8, s14, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s8, s8, s12
	s_sub_co_i32 s10, s8, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s10, s10, s11
	s_cmp_lg_u32 s5, s10
	s_cselect_b32 s10, -1, 0
	s_xor_b32 s4, s4, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s4, 0
	s_cselect_b32 s4, -1, 0
	s_and_b32 s4, s4, s10
	s_sub_co_ci_u32 s4, s8, s12
	s_load_b64 s[12:13], s[0:1], 0x70 nv
	s_mul_i32 s8, s4, s11
	s_lshl_b32 s4, s4, 4
	s_sub_co_i32 s5, s5, s8
	s_cmp_lg_u32 s7, 0
	s_sub_co_ci_u32 s6, s6, s4
	s_abs_i32 s14, s5
	s_min_i32 s7, s6, 16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_abs_i32 s8, s7
	s_cvt_f32_u32 s10, s8
	s_sub_co_i32 s11, 0, s8
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s10
	v_nop
	v_readfirstlane_b32 s10, v1
	s_mul_f32 s10, s10, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s10, s10
	s_mul_i32 s11, s11, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s11, s10, s11
	s_add_co_i32 s10, s10, s11
	s_xor_b32 s11, s5, s7
	s_mul_hi_u32 s10, s14, s10
	s_ashr_i32 s11, s11, 31
	s_mul_i32 s15, s10, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s14, s14, s15
	s_add_co_i32 s15, s10, 1
	s_sub_co_i32 s16, s14, s8
	s_cmp_ge_u32 s14, s8
	s_cselect_b32 s10, s15, s10
	s_cselect_b32 s14, s16, s14
	s_add_co_i32 s15, s10, 1
	s_cmp_ge_u32 s14, s8
	s_wait_kmcnt 0x0
	s_load_b32 s14, s[12:13], 0xc0
	s_cselect_b32 s8, s15, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s8, s8, s11
	s_sub_co_i32 s10, s8, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s10, s7, s10
	s_cmp_lg_u32 s5, s10
	s_cselect_b32 s10, -1, 0
	s_xor_b32 s6, s5, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s6, 0
	s_cselect_b32 s6, -1, 0
	s_and_b32 s6, s6, s10
	s_sub_co_ci_u32 s8, s8, s11
	s_add_co_i32 s5, s5, s4
	s_mul_i32 s4, s8, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_co_i32 s4, s5, s4
	s_lshl_b32 s34, s4, 6
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s14, s34
	s_cselect_b32 s4, 24, 0x48
	s_cselect_b32 s6, 0, 49
	s_load_b32 s5, s[12:13], s4 offset:0x0 scale_offset
	s_cselect_b32 s7, 48, 0x60
	s_or_b32 s10, s4, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s5, s34
	s_cselect_b32 s5, s6, s10
	s_cselect_b32 s4, s4, s7
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s5, s4
	s_lshr_b32 s6, s6, 1
	s_load_b32 s7, s[12:13], s6 offset:0x0 scale_offset
	s_or_b32 s10, s6, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s7, s34
	s_cselect_b32 s10, s5, s10
	s_cselect_b32 s4, s6, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s5, s10, s4
	s_lshr_b32 s11, s5, 1
	s_mov_b32 s5, 0
	s_load_b32 s6, s[12:13], s11 offset:0x0 scale_offset
	s_add_co_i32 s14, s11, 1
	s_mov_b32 s7, s5
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s6, s34
	s_cselect_b32 s6, s10, s14
	s_cselect_b32 s4, s11, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[10:11], s[6:7], s[4:5]
	s_lshr_b64 s[10:11], s[10:11], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b64 s[14:15], s[10:11], 2
	s_add_co_i32 s7, s10, 1
	s_add_nc_u64 s[14:15], s[12:13], s[14:15]
	v_readfirstlane_b32 s11, v0
	s_load_b32 s5, s[14:15], 0x0
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s5, s34
	s_cselect_b32 s5, s6, s7
	s_cselect_b32 s4, s10, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s5, s4
	s_lshr_b32 s6, s6, 1
	s_load_b32 s7, s[12:13], s6 offset:0x0 scale_offset
	s_add_co_i32 s10, s6, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s7, s34
	s_cselect_b32 s5, s5, s10
	s_cselect_b32 s4, s6, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s6, s5, s4
	s_lshr_b32 s6, s6, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s7, s6, 0x5f
	s_add_co_i32 s10, s6, 1
	s_load_b32 s7, s[12:13], s7 offset:0x0 scale_offset
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s7, s34
	s_cselect_b32 s5, s5, s10
	s_cselect_b32 s4, s6, s4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s4, s5, s4
	s_lshr_b32 s4, s4, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s6, s4, 0x5f
	s_add_co_i32 s4, s4, 1
	s_load_b32 s6, s[12:13], s6 offset:0x0 scale_offset
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s6, s34
	s_cselect_b32 s18, s5, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_u32 s18, 0x60
	s_cselect_b32 s4, -1, 0
	s_cmp_gt_u32 s18, 0x5f
	s_cbranch_scc1 .LBB0_17
	s_lshr_b32 s10, s11, 5
	s_ashr_i32 s35, s34, 31
	s_and_b32 s4, s4, exec_lo
	s_cselect_b32 s14, s18, 0x5f
	s_load_b32 s19, s[12:13], s14 offset:0x0 scale_offset
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x28 nv
	s_load_b64 s[24:25], s[0:1], 0x38 nv
	s_wait_xcnt 0x0
	s_and_b32 s12, s11, 0xffffffe0
	s_lshl_b32 s26, s10, 6
	s_ashr_i32 s13, s12, 31
	s_lshr_b64 s[14:15], s[34:35], 6
	s_add_nc_u64 s[16:17], s[34:35], s[12:13]
	s_ashr_i32 s27, s26, 31
	s_mul_u64 s[14:15], s[14:15], 0x1800
	s_mul_u64 s[16:17], s[16:17], 0x600
	s_lshl_b64 s[20:21], s[26:27], 2
	s_mov_b32 s40, s9
	s_mul_i32 s29, s10, 0x1200
	s_wait_kmcnt 0x0
	s_sub_co_i32 s33, s19, s34
	s_add_nc_u64 s[30:31], s[4:5], s[16:17]
	s_sub_co_i32 s27, s33, s12
	s_cmp_lt_u32 s11, 64
	s_add_nc_u64 s[4:5], s[24:25], s[14:15]
	s_cselect_b32 s25, -1, 0
	s_cmp_gt_u32 s11, 63
	s_add_nc_u64 s[38:39], s[4:5], s[20:21]
	s_cbranch_scc1 .LBB0_3
	s_mov_b32 s28, 1
	s_or_b32 s4, s31, 0x80000000
	s_mov_b64 s[12:13], s[28:29]
	s_mov_b64 s[14:15], s[30:31]
	s_mov_b32 s15, s4
	s_max_i32 s4, s27, 0
	s_mov_b32 s50, 0
	s_lshl_b32 s5, s4, 16
	s_lshr_b32 s4, s4, 16
	s_or_b32 s46, s5, 0x7fff
	s_or_b32 s47, s4, 0x800000
	s_movk_i32 s49, 0x600
	s_mov_b32 s48, 32
	s_mov_b32 s45, 0xffff0000
	s_mov_b32 s44, 0x7100000
	s_mov_b32 s51, s50
	s_lshl2_add_u32 s4, s26, 0xa400
	tensor_load_to_lds s[12:15], s[44:51]
	s_or_b32 s5, s39, 0x80000000
	s_mov_b64 s[12:13], s[28:29]
	s_mov_b64 s[14:15], s[30:31]
	s_mov_b32 s13, s4
	s_mov_b32 s14, s38
	s_mov_b32 s15, s5
	s_mov_b32 s47, 0x407fff
	s_mov_b32 s46, 0xffff7fff
	s_mov_b32 s44, 0x20000
	s_mov_b32 s48, s28
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[44:51]
.LBB0_3:
	s_ashr_i32 s41, s9, 31
	s_lshl_b32 s42, s8, 8
	s_lshr_b32 s4, s41, 28
	s_ashr_i32 s43, s42, 31
	s_add_co_i32 s4, s9, s4
	s_mov_b32 s19, 0
	s_ashr_i32 s12, s4, 4
	s_mov_b32 s15, s19
	s_ashr_i32 s13, s12, 31
	s_mov_b64 s[16:17], 0xffffffffffffffe0
	s_lshl_b64 s[4:5], s[12:13], 4
	s_mov_b32 s21, s19
	s_cmp_lg_u64 s[4:5], s[40:41]
	s_add_nc_u64 s[4:5], s[40:41], 31
	s_cselect_b32 s8, -1, 0
	s_cmp_lt_i32 s9, 0
	s_cselect_b32 s28, -1, 0
	s_lshr_b32 s14, s5, 27
	s_and_b32 s8, s28, s8
	s_add_nc_u64 s[14:15], s[4:5], s[14:15]
	v_cndmask_b32_e64 v1, 0, 1, s8
	s_and_b64 s[16:17], s[14:15], s[16:17]
	s_ashr_i64 s[14:15], s[14:15], 5
	s_cmp_lg_u64 s[4:5], s[16:17]
	s_load_b64 s[4:5], s[0:1], 0x60 nv
	s_cselect_b32 s16, -1, 0
	s_cmp_lt_i32 s9, 0xffffffe1
	v_readfirstlane_b32 s8, v1
	s_wait_xcnt 0x0
	s_cselect_b32 s0, -1, 0
	s_add_co_i32 s1, s10, -2
	s_and_b32 s0, s0, s16
	s_mov_b32 s9, s19
	v_cndmask_b32_e64 v1, 0, 1, s0
	s_lshl_b32 s22, s1, 3
	s_sub_nc_u64 s[12:13], s[12:13], s[8:9]
	s_ashr_i64 s[8:9], s[42:43], 4
	s_ashr_i32 s23, s22, 31
	v_readfirstlane_b32 s20, v1
	s_mul_u64 s[44:45], s[12:13], s[18:19]
	s_add_nc_u64 s[12:13], s[8:9], s[22:23]
	s_and_b32 s17, s11, 0xffffffc0
	s_add_nc_u64 s[12:13], s[12:13], s[44:45]
	s_sub_nc_u64 s[14:15], s[14:15], s[20:21]
	s_mul_u64 s[12:13], s[12:13], 0x6000
	s_lshr_b64 s[20:21], s[42:43], 5
	s_mul_u64 s[14:15], s[14:15], s[18:19]
	s_add_nc_u64 s[22:23], s[6:7], s[12:13]
	s_mul_u64 s[46:47], s[20:21], 0xc00
	s_lshl_b32 s12, s1, 2
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[20:21], s[4:5], s[46:47]
	s_mul_u64 s[48:49], s[14:15], 0xc00
	s_ashr_i32 s13, s12, 31
	s_lshl_b32 s53, s1, 14
	s_lshl_b32 s54, s1, 8
	s_add_nc_u64 s[14:15], s[20:21], s[48:49]
	s_cmp_eq_u32 s17, 64
	s_mul_u64 s[12:13], s[12:13], 0xc00
	s_mov_b32 s16, 8
	s_mov_b32 s0, 4
	s_cselect_b32 s50, -1, 0
	s_cmp_lg_u32 s17, 64
	s_add_nc_u64 s[36:37], s[14:15], s[12:13]
	s_cbranch_scc1 .LBB0_5
	s_add_co_i32 s21, s53, 0x2400
	s_mov_b32 s20, 1
	s_or_b32 s1, s23, 0x80000000
	s_mov_b64 s[58:59], s[22:23]
	s_mov_b64 s[56:57], s[20:21]
	s_mov_b32 s59, s1
	s_movk_i32 s17, 0x6000
	s_mov_b32 s15, 0x8007fff
	s_mov_b32 s14, 0xffff7fff
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s12, s19
	s_mov_b32 s18, s19
	s_lshl2_add_u32 s21, s54, 0xa600
	tensor_load_to_lds s[56:59], s[12:19]
	s_or_b32 s1, s37, 0x80000000
	s_mov_b64 s[58:59], s[22:23]
	s_mov_b64 s[56:57], s[20:21]
	s_mov_b32 s58, s36
	s_mov_b32 s59, s1
	s_movk_i32 s17, 0x300
	s_mov_b32 s15, 0x407fff
	s_mov_b32 s12, 0x20000
	s_mov_b32 s16, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[56:59], s[12:19]
.LBB0_5:
	v_cndmask_b32_e64 v1, 0, 1, s25
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_mov_b32 s12, 1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e64 s1, 1, v1
	s_cbranch_vccnz .LBB0_7
	s_max_i32 s0, s27, 0
	s_add_nc_u64 s[14:15], s[30:31], 0x80
	s_mov_b32 s62, 0
	s_lshl_b32 s16, s0, 16
	s_lshr_b32 s0, s0, 16
	s_add_co_i32 s13, s29, 0xae00
	s_bitset1_b32 s15, 31
	s_or_b32 s58, s16, 0x7fff
	s_or_b32 s59, s0, 0x800000
	s_movk_i32 s61, 0x600
	s_mov_b32 s60, 32
	s_mov_b32 s57, 0xffff0000
	s_mov_b32 s56, 0x7100000
	s_mov_b32 s63, s62
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[56:63]
	s_add_nc_u64 s[14:15], s[38:39], 0x200
	s_lshl2_add_u32 s13, s26, 0x15200
	s_bitset1_b32 s15, 31
	s_mov_b32 s59, 0x407fff
	s_mov_b32 s58, 0xffff7fff
	s_mov_b32 s56, 0x20000
	s_mov_b32 s60, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[12:15], s[56:63]
.LBB0_7:
	v_cndmask_b32_e64 v1, 0, 1, s50
	s_and_not1_b32 vcc_lo, exec_lo, s50
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 1, v1
	s_cbranch_vccnz .LBB0_9
	s_add_nc_u64 s[58:59], s[22:23], 0x800
	s_mov_b32 s12, 0
	s_add_co_i32 s57, s53, 0xd200
	s_bitset1_b32 s59, 31
	s_mov_b32 s56, 1
	s_movk_i32 s17, 0x6000
	s_mov_b32 s16, 8
	s_mov_b32 s15, 0x8007fff
	s_mov_b32 s14, 0xffff7fff
	s_mov_b32 s13, 0xffff0000
	s_mov_b32 s18, s12
	s_mov_b32 s19, s12
	s_movk_i32 s21, 0x300
	tensor_load_to_lds s[56:59], s[12:19]
	s_add_nc_u64 s[58:59], s[36:37], 0x100
	s_lshl2_add_u32 s57, s54, 0x15400
	s_bitset1_b32 s59, 31
	s_mov_b32 s20, 4
	s_mov_b32 s19, 0x407fff
	s_mov_b32 s16, 0x20000
	s_mov_b32 s17, s13
	s_mov_b32 s18, s14
	s_mov_b32 s22, s12
	s_mov_b32 s23, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[56:59], s[16:23]
.LBB0_9:
	v_bfe_u32 v193, v0, 4, 1
	v_and_b32_e32 v1, 15, v0
	s_lshr_b32 s11, s11, 7
	s_and_b32 s52, s26, 0xc0
	s_mov_b32 s18, 0
	v_dual_lshlrev_b32 v3, 8, v193 :: v_dual_bitop2_b32 v2, 16, v0 bitop3:0x40
	v_and_b32_e32 v0, 31, v0
	v_lshl_or_b32 v192, s11, 6, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_lshl_b32 s12, s52, 7
	s_lshl_b32 s11, s11, 9
	v_lshlrev_b32_e32 v0, 2, v0
	v_mad_u32 v194, 0x90, v192, v2
	s_addk_co_i32 s12, 0x2400
	s_add_co_i32 s11, s11, 0xa400
	v_or3_b32 v195, s12, v3, v1
	v_or_b32_e32 v196, s11, v0
	v_lshl_or_b32 v197, s52, 3, v0
	s_and_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s36, 1
	s_cbranch_vccnz .LBB0_12
	s_max_i32 s1, s27, 0
	s_lshr_b32 s11, s34, 6
	s_lshl_b32 s12, s1, 16
	s_lshr_b32 s1, s1, 16
	v_mov_b32_e32 v64, 0
	s_or_b32 s15, s1, 0x800000
	s_mul_i32 s1, s11, 0x1800
	s_lshl_b32 s11, s10, 8
	s_movk_i32 s17, 0x600
	s_add_co_i32 s11, s1, s11
	v_dual_mov_b32 v65, v64 :: v_dual_mov_b32 v66, v64
	v_dual_mov_b32 v67, v64 :: v_dual_mov_b32 v68, v64
	v_dual_mov_b32 v69, v64 :: v_dual_mov_b32 v70, v64
	v_dual_mov_b32 v71, v64 :: v_dual_mov_b32 v72, v64
	v_dual_mov_b32 v73, v64 :: v_dual_mov_b32 v74, v64
	v_dual_mov_b32 v75, v64 :: v_dual_mov_b32 v76, v64
	v_dual_mov_b32 v77, v64 :: v_dual_mov_b32 v78, v64
	v_dual_mov_b32 v79, v64 :: v_dual_mov_b32 v112, v64
	v_dual_mov_b32 v113, v64 :: v_dual_mov_b32 v114, v64
	v_dual_mov_b32 v115, v64 :: v_dual_mov_b32 v116, v64
	v_dual_mov_b32 v117, v64 :: v_dual_mov_b32 v118, v64
	v_dual_mov_b32 v119, v64 :: v_dual_mov_b32 v120, v64
	v_dual_mov_b32 v121, v64 :: v_dual_mov_b32 v122, v64
	v_dual_mov_b32 v123, v64 :: v_dual_mov_b32 v124, v64
	v_dual_mov_b32 v125, v64 :: v_dual_mov_b32 v126, v64
	v_dual_mov_b32 v127, v64 :: v_dual_mov_b32 v96, v64
	v_dual_mov_b32 v97, v64 :: v_dual_mov_b32 v98, v64
	v_dual_mov_b32 v99, v64 :: v_dual_mov_b32 v100, v64
	v_dual_mov_b32 v101, v64 :: v_dual_mov_b32 v102, v64
	v_dual_mov_b32 v103, v64 :: v_dual_mov_b32 v104, v64
	v_dual_mov_b32 v105, v64 :: v_dual_mov_b32 v106, v64
	v_dual_mov_b32 v107, v64 :: v_dual_mov_b32 v108, v64
	v_dual_mov_b32 v109, v64 :: v_dual_mov_b32 v110, v64
	v_dual_mov_b32 v111, v64 :: v_dual_mov_b32 v80, v64
	v_dual_mov_b32 v81, v64 :: v_dual_mov_b32 v82, v64
	v_dual_mov_b32 v83, v64 :: v_dual_mov_b32 v84, v64
	v_dual_mov_b32 v85, v64 :: v_dual_mov_b32 v86, v64
	v_dual_mov_b32 v87, v64 :: v_dual_mov_b32 v88, v64
	v_dual_mov_b32 v89, v64 :: v_dual_mov_b32 v90, v64
	v_dual_mov_b32 v91, v64 :: v_dual_mov_b32 v92, v64
	v_dual_mov_b32 v93, v64 :: v_dual_mov_b32 v94, v64
	v_dual_mov_b32 v95, v64 :: v_dual_mov_b32 v48, v64
	v_dual_mov_b32 v49, v64 :: v_dual_mov_b32 v50, v64
	v_dual_mov_b32 v51, v64 :: v_dual_mov_b32 v52, v64
	v_dual_mov_b32 v53, v64 :: v_dual_mov_b32 v54, v64
	v_dual_mov_b32 v55, v64 :: v_dual_mov_b32 v56, v64
	v_dual_mov_b32 v57, v64 :: v_dual_mov_b32 v58, v64
	v_dual_mov_b32 v59, v64 :: v_dual_mov_b32 v60, v64
	v_dual_mov_b32 v61, v64 :: v_dual_mov_b32 v62, v64
	v_dual_mov_b32 v63, v64 :: v_dual_mov_b32 v32, v64
	v_dual_mov_b32 v33, v64 :: v_dual_mov_b32 v34, v64
	v_dual_mov_b32 v35, v64 :: v_dual_mov_b32 v36, v64
	v_dual_mov_b32 v37, v64 :: v_dual_mov_b32 v38, v64
	v_dual_mov_b32 v39, v64 :: v_dual_mov_b32 v40, v64
	v_dual_mov_b32 v41, v64 :: v_dual_mov_b32 v42, v64
	v_dual_mov_b32 v43, v64 :: v_dual_mov_b32 v44, v64
	v_dual_mov_b32 v45, v64 :: v_dual_mov_b32 v46, v64
	v_dual_mov_b32 v47, v64 :: v_dual_mov_b32 v16, v64
	v_dual_mov_b32 v17, v64 :: v_dual_mov_b32 v18, v64
	v_dual_mov_b32 v19, v64 :: v_dual_mov_b32 v20, v64
	v_dual_mov_b32 v21, v64 :: v_dual_mov_b32 v22, v64
	v_dual_mov_b32 v23, v64 :: v_dual_mov_b32 v24, v64
	v_dual_mov_b32 v25, v64 :: v_dual_mov_b32 v26, v64
	v_dual_mov_b32 v27, v64 :: v_dual_mov_b32 v28, v64
	v_dual_mov_b32 v29, v64 :: v_dual_mov_b32 v30, v64
	v_dual_mov_b32 v31, v64 :: v_dual_mov_b32 v0, v64
	v_dual_mov_b32 v1, v64 :: v_dual_mov_b32 v2, v64
	v_dual_mov_b32 v3, v64 :: v_dual_mov_b32 v4, v64
	v_dual_mov_b32 v5, v64 :: v_dual_mov_b32 v6, v64
	v_dual_mov_b32 v7, v64 :: v_dual_mov_b32 v8, v64
	v_dual_mov_b32 v9, v64 :: v_dual_mov_b32 v10, v64
	v_dual_mov_b32 v11, v64 :: v_dual_mov_b32 v12, v64
	v_dual_mov_b32 v13, v64 :: v_dual_mov_b32 v14, v64
	v_mov_b32_e32 v15, v64
	s_mov_b32 s13, 0xffff0000
	s_add_co_i32 s11, s11, s24
	s_or_b32 s14, s12, 0x7fff
	s_mov_b32 s16, 32
	s_mov_b32 s12, 0x7100000
	s_mov_b32 s19, s18
	s_lshl_b32 s1, s26, 2
	s_add_nc_u64 s[50:51], s[38:39], 0x400
	s_addk_co_i32 s11, 0x400
	s_add_nc_u64 s[38:39], s[30:31], 0x100
	s_mov_b64 s[30:31], 0
	s_mov_b32 s23, 0x407fff
	s_mov_b32 s22, 0xffff7fff
	s_mov_b32 s20, 0x20000
	s_mov_b32 s21, s13
	s_mov_b32 s24, s36
	s_mov_b32 s25, s17
	s_mov_b32 s26, s18
	s_mov_b32 s27, s18
	s_mov_b32 s55, s18
.LBB0_11:
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_bitcmp1_b32 s55, 0
	s_cselect_b32 s56, 0xae00, 0
	v_nop
	v_nop
	v_dual_add_nc_u32 v128, s56, v197 :: v_dual_add_nc_u32 v168, s56, v196
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v206, s56, v195 :: v_dual_add_nc_u32 v238, s56, v194
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v144, 0xa400, v128
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[230:231], v144 offset0:128 offset1:160
	ds_load_2addr_b32 v[232:233], v168 offset1:32
	ds_load_b128 v[128:131], v206
	ds_load_b128 v[132:135], v206 offset:512
	ds_load_b128 v[136:139], v206 offset:2048
	ds_load_b128 v[140:143], v206 offset:2560
	ds_load_b128 v[160:163], v238
	ds_load_b128 v[164:167], v238 offset:32
	ds_load_2addr_b32 v[234:235], v144 offset0:192 offset1:224
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[144:147], v206 offset:4096
	ds_load_b128 v[148:151], v206 offset:4608
	ds_load_b128 v[152:155], v206 offset:6144
	ds_load_b128 v[156:159], v206 offset:6656
	ds_load_b128 v[184:187], v238 offset:4608
	ds_load_b128 v[188:191], v238 offset:4640
	ds_load_b128 v[198:201], v238 offset:6912
	ds_load_b128 v[202:205], v238 offset:6944
	s_wait_dscnt 0x9
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[128:143], v[160:167], v[64:79], v230, v232
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[144:159], v[160:167], v[112:127], v234, v232
	ds_load_b128 v[160:163], v238 offset:2304
	ds_load_b128 v[164:167], v238 offset:2336
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[128:143], v[184:191], v[48:63], v230, v233
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[144:159], v[160:167], v[80:95], v234, v232 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[128:143], v[160:167], v[96:111], v230, v232 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_2addr_b32 v[236:237], v168 offset0:64 offset1:96
	ds_load_b128 v[160:163], v206 offset:1024
	ds_load_b128 v[164:167], v206 offset:1536
	s_wait_alu depctr_vm_vsrc(2)
	ds_load_b128 v[168:171], v206 offset:3072
	ds_load_b128 v[172:175], v206 offset:3584
	ds_load_b128 v[176:179], v206 offset:5120
	ds_load_b128 v[180:183], v206 offset:5632
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[144:159], v[184:191], v[32:47], v234, v233
	ds_load_b128 v[184:187], v206 offset:7168
	ds_load_b128 v[188:191], v206 offset:7680
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[206:209], v238 offset:64
	ds_load_b128 v[210:213], v238 offset:96
	ds_load_b128 v[214:217], v238 offset:2368
	ds_load_b128 v[218:221], v238 offset:2400
	ds_load_b128 v[222:225], v238 offset:4672
	ds_load_b128 v[226:229], v238 offset:4704
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[144:159], v[198:205], v[0:15], v234, v233 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[144:147], v238 offset:6976
	ds_load_b128 v[148:151], v238 offset:7008
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[128:143], v[198:205], v[16:31], v230, v233 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_dscnt 0x6
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[160:175], v[206:213], v[64:79], v231, v236
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[176:191], v[206:213], v[112:127], v235, v236
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[176:191], v[214:221], v[80:95], v235, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[160:175], v[214:221], v[96:111], v231, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[160:175], v[222:229], v[48:63], v231, v237
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[176:191], v[222:229], v[32:47], v235, v237
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[176:191], v[144:151], v[0:15], v235, v237 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[160:175], v[144:151], v[16:31], v231, v237 matrix_b_scale:MATRIX_SCALE_ROW1
	s_barrier_signal -1
	s_add_co_i32 s37, s56, s29
	s_or_b32 s57, s39, 0x80000000
	s_mov_b64 s[62:63], s[38:39]
	s_mov_b64 s[60:61], s[36:37]
	s_mov_b32 s63, s57
	s_add_co_i32 s37, s1, s56
	s_add_nc_u64 s[56:57], s[50:51], s[30:31]
	s_add_co_i32 s37, s37, 0xa400
	s_add_co_i32 s55, s55, 1
	s_barrier_wait -1
	tensor_load_to_lds s[60:63], s[12:19]
	s_add_co_i32 s60, s11, s30
	s_or_b32 s61, s57, 0x80000000
	s_mov_b64 s[58:59], s[38:39]
	s_mov_b64 s[56:57], s[36:37]
	s_mov_b32 s58, s60
	s_mov_b32 s59, s61
	s_add_nc_u64 s[30:31], s[30:31], 0x200
	s_add_nc_u64 s[38:39], s[38:39], 0x80
	s_cmp_lg_u32 s30, 0x1400
	tensor_load_to_lds s[56:59], s[20:27]
	s_cbranch_scc1 .LBB0_11
	s_branch .LBB0_13
.LBB0_12:
	v_mov_b32_e32 v0, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v13, v0 :: v_dual_mov_b32 v14, v0
	v_dual_mov_b32 v15, v0 :: v_dual_mov_b32 v1, v0
	v_dual_mov_b32 v2, v0 :: v_dual_mov_b32 v3, v0
	v_dual_mov_b32 v4, v0 :: v_dual_mov_b32 v5, v0
	v_dual_mov_b32 v6, v0 :: v_dual_mov_b32 v7, v0
	v_dual_mov_b32 v8, v0 :: v_dual_mov_b32 v9, v0
	v_dual_mov_b32 v10, v0 :: v_dual_mov_b32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b64_e32 v[30:31], v[14:15]
	v_mov_b64_e32 v[46:47], v[14:15]
	v_mov_b64_e32 v[62:63], v[14:15]
	v_mov_b64_e32 v[94:95], v[14:15]
	v_mov_b64_e32 v[110:111], v[14:15]
	v_mov_b64_e32 v[126:127], v[14:15]
	v_mov_b64_e32 v[78:79], v[14:15]
	v_mov_b64_e32 v[28:29], v[12:13]
	v_mov_b64_e32 v[26:27], v[10:11]
	v_mov_b64_e32 v[24:25], v[8:9]
	v_mov_b64_e32 v[22:23], v[6:7]
	v_mov_b64_e32 v[20:21], v[4:5]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[44:45], v[12:13]
	v_mov_b64_e32 v[42:43], v[10:11]
	v_mov_b64_e32 v[40:41], v[8:9]
	v_mov_b64_e32 v[38:39], v[6:7]
	v_mov_b64_e32 v[36:37], v[4:5]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[60:61], v[12:13]
	v_mov_b64_e32 v[58:59], v[10:11]
	v_mov_b64_e32 v[56:57], v[8:9]
	v_mov_b64_e32 v[54:55], v[6:7]
	v_mov_b64_e32 v[52:53], v[4:5]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[92:93], v[12:13]
	v_mov_b64_e32 v[90:91], v[10:11]
	v_mov_b64_e32 v[88:89], v[8:9]
	v_mov_b64_e32 v[86:87], v[6:7]
	v_mov_b64_e32 v[84:85], v[4:5]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[108:109], v[12:13]
	v_mov_b64_e32 v[106:107], v[10:11]
	v_mov_b64_e32 v[104:105], v[8:9]
	v_mov_b64_e32 v[102:103], v[6:7]
	v_mov_b64_e32 v[100:101], v[4:5]
	v_mov_b64_e32 v[98:99], v[2:3]
	v_mov_b64_e32 v[96:97], v[0:1]
	v_mov_b64_e32 v[124:125], v[12:13]
	v_mov_b64_e32 v[122:123], v[10:11]
	v_mov_b64_e32 v[120:121], v[8:9]
	v_mov_b64_e32 v[118:119], v[6:7]
	v_mov_b64_e32 v[116:117], v[4:5]
	v_mov_b64_e32 v[114:115], v[2:3]
	v_mov_b64_e32 v[112:113], v[0:1]
	v_mov_b64_e32 v[76:77], v[12:13]
	v_mov_b64_e32 v[74:75], v[10:11]
	v_mov_b64_e32 v[72:73], v[8:9]
	v_mov_b64_e32 v[70:71], v[6:7]
	v_mov_b64_e32 v[68:69], v[4:5]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[64:65], v[0:1]
.LBB0_13:
	s_and_b32 vcc_lo, exec_lo, s0
	s_mov_b32 s20, 1
	s_cbranch_vccnz .LBB0_16
	s_mov_b32 s11, 0
	s_add_nc_u64 s[0:1], s[4:5], s[48:49]
	s_mul_u64 s[4:5], s[10:11], 0x3000
	s_add_nc_u64 s[8:9], s[8:9], s[44:45]
	s_add_nc_u64 s[0:1], s[0:1], s[4:5]
	s_mul_u64 s[4:5], s[10:11], 0x30000
	s_add_nc_u64 s[0:1], s[0:1], s[46:47]
	s_add_nc_u64 s[4:5], s[6:7], s[4:5]
	s_mul_u64 s[6:7], s[8:9], 0x6000
	s_mov_b64 s[8:9], 0xffffffffffffa200
	s_add_nc_u64 s[4:5], s[4:5], s[6:7]
	s_mov_b64 s[6:7], 0xfffffffffffa1000
	s_lshl_b32 s24, s54, 2
	s_add_nc_u64 s[22:23], s[4:5], s[6:7]
	s_mov_b32 s6, 0xffff7fff
	s_mov_b32 s5, 0xffff0000
	s_add_nc_u64 s[0:1], s[0:1], s[8:9]
	s_movk_i32 s9, 0x6000
	s_mov_b32 s8, 8
	s_mov_b32 s7, 0x8007fff
	s_mov_b32 s4, s11
	s_mov_b32 s10, s11
	s_movk_i32 s17, 0x300
	s_mov_b32 s16, 4
	s_mov_b32 s15, 0x407fff
	s_mov_b32 s12, 0x20000
	s_mov_b32 s13, s5
	s_mov_b32 s14, s6
	s_mov_b32 s18, s11
	s_mov_b32 s19, s11
	s_mov_b32 s25, s11
.LBB0_15:
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_bitcmp1_b32 s25, 0
	s_cselect_b32 s26, 0xae00, 0
	v_nop
	v_nop
	v_dual_add_nc_u32 v128, s26, v197 :: v_dual_add_nc_u32 v168, s26, v196
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v206, s26, v195 :: v_dual_add_nc_u32 v238, s26, v194
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v144, 0xa400, v128
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[230:231], v144 offset0:128 offset1:160
	ds_load_2addr_b32 v[232:233], v168 offset1:32
	ds_load_b128 v[128:131], v206
	ds_load_b128 v[132:135], v206 offset:512
	ds_load_b128 v[136:139], v206 offset:2048
	ds_load_b128 v[140:143], v206 offset:2560
	ds_load_b128 v[160:163], v238
	ds_load_b128 v[164:167], v238 offset:32
	ds_load_2addr_b32 v[234:235], v144 offset0:192 offset1:224
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[144:147], v206 offset:4096
	ds_load_b128 v[148:151], v206 offset:4608
	ds_load_b128 v[152:155], v206 offset:6144
	ds_load_b128 v[156:159], v206 offset:6656
	ds_load_b128 v[184:187], v238 offset:4608
	ds_load_b128 v[188:191], v238 offset:4640
	ds_load_b128 v[198:201], v238 offset:6912
	ds_load_b128 v[202:205], v238 offset:6944
	s_wait_dscnt 0x9
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[128:143], v[160:167], v[64:79], v230, v232
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[144:159], v[160:167], v[112:127], v234, v232
	ds_load_b128 v[160:163], v238 offset:2304
	ds_load_b128 v[164:167], v238 offset:2336
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[128:143], v[184:191], v[48:63], v230, v233
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[144:159], v[160:167], v[80:95], v234, v232 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[128:143], v[160:167], v[96:111], v230, v232 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_2addr_b32 v[236:237], v168 offset0:64 offset1:96
	ds_load_b128 v[160:163], v206 offset:1024
	ds_load_b128 v[164:167], v206 offset:1536
	s_wait_alu depctr_vm_vsrc(2)
	ds_load_b128 v[168:171], v206 offset:3072
	ds_load_b128 v[172:175], v206 offset:3584
	ds_load_b128 v[176:179], v206 offset:5120
	ds_load_b128 v[180:183], v206 offset:5632
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[144:159], v[184:191], v[32:47], v234, v233
	ds_load_b128 v[184:187], v206 offset:7168
	ds_load_b128 v[188:191], v206 offset:7680
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[206:209], v238 offset:64
	ds_load_b128 v[210:213], v238 offset:96
	ds_load_b128 v[214:217], v238 offset:2368
	ds_load_b128 v[218:221], v238 offset:2400
	ds_load_b128 v[222:225], v238 offset:4672
	ds_load_b128 v[226:229], v238 offset:4704
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[144:159], v[198:205], v[0:15], v234, v233 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[144:147], v238 offset:6976
	ds_load_b128 v[148:151], v238 offset:7008
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[128:143], v[198:205], v[16:31], v230, v233 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_dscnt 0x6
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[160:175], v[206:213], v[64:79], v231, v236
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[176:191], v[206:213], v[112:127], v235, v236
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[176:191], v[214:221], v[80:95], v235, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[160:175], v[214:221], v[96:111], v231, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[160:175], v[222:229], v[48:63], v231, v237
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[176:191], v[222:229], v[32:47], v235, v237
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[176:191], v[144:151], v[0:15], v235, v237 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[160:175], v[144:151], v[16:31], v231, v237 matrix_b_scale:MATRIX_SCALE_ROW1
	s_barrier_signal -1
	s_add_co_i32 s21, s26, s53
	s_or_b32 s27, s23, 0x80000000
	s_addk_co_i32 s21, 0x2400
	s_mov_b64 s[38:39], s[22:23]
	s_mov_b64 s[36:37], s[20:21]
	s_mov_b32 s39, s27
	s_add_co_i32 s21, s24, s26
	s_or_b32 s26, s1, 0x80000000
	s_add_co_i32 s21, s21, 0xa600
	s_add_co_i32 s25, s25, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s25, 10
	s_barrier_wait -1
	tensor_load_to_lds s[36:39], s[4:11]
	s_mov_b64 s[38:39], s[22:23]
	s_mov_b64 s[36:37], s[20:21]
	s_mov_b32 s38, s0
	s_mov_b32 s39, s26
	s_add_nc_u64 s[22:23], s[22:23], 0x800
	s_add_nc_u64 s[0:1], s[0:1], 0x100
	tensor_load_to_lds s[36:39], s[12:19]
	s_cbranch_scc1 .LBB0_15
.LBB0_16:
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	v_nop
	v_nop
	v_add_nc_u32_e32 v144, 0xa400, v197
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[230:231], v144 offset0:128 offset1:160
	ds_load_2addr_b32 v[232:233], v196 offset1:32
	ds_load_b128 v[128:131], v195
	ds_load_b128 v[132:135], v195 offset:512
	ds_load_b128 v[136:139], v195 offset:2048
	ds_load_b128 v[140:143], v195 offset:2560
	ds_load_b128 v[160:163], v194
	ds_load_b128 v[164:167], v194 offset:32
	ds_load_2addr_b32 v[234:235], v144 offset0:192 offset1:224
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[144:147], v195 offset:4096
	ds_load_b128 v[148:151], v195 offset:4608
	ds_load_b128 v[152:155], v195 offset:6144
	ds_load_b128 v[156:159], v195 offset:6656
	ds_load_b128 v[184:187], v194 offset:4608
	ds_load_b128 v[188:191], v194 offset:4640
	ds_load_b128 v[198:201], v194 offset:6912
	ds_load_b128 v[202:205], v194 offset:6944
	s_wait_dscnt 0x9
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[128:143], v[160:167], v[64:79], v230, v232
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[144:159], v[160:167], v[112:127], v234, v232
	ds_load_b128 v[160:163], v194 offset:2304
	ds_load_b128 v[164:167], v194 offset:2336
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[128:143], v[184:191], v[48:63], v230, v233
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[144:159], v[184:191], v[32:47], v234, v233
	ds_load_b128 v[184:187], v195 offset:7168
	ds_load_b128 v[188:191], v195 offset:7680
	ds_load_b128 v[206:209], v194 offset:64
	ds_load_b128 v[210:213], v194 offset:96
	ds_load_b128 v[214:217], v194 offset:2368
	ds_load_b128 v[218:221], v194 offset:2400
	ds_load_b128 v[222:225], v194 offset:4672
	ds_load_b128 v[226:229], v194 offset:4704
	s_wait_dscnt 0x8
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[144:159], v[160:167], v[80:95], v234, v232 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[128:143], v[160:167], v[96:111], v230, v232 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_2addr_b32 v[236:237], v196 offset0:64 offset1:96
	ds_load_b128 v[160:163], v195 offset:1024
	ds_load_b128 v[164:167], v195 offset:1536
	ds_load_b128 v[168:171], v195 offset:3072
	ds_load_b128 v[172:175], v195 offset:3584
	ds_load_b128 v[176:179], v195 offset:5120
	ds_load_b128 v[180:183], v195 offset:5632
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[144:159], v[198:205], v[0:15], v234, v233 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[144:147], v194 offset:6976
	ds_load_b128 v[148:151], v194 offset:7008
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v152, 0xa600, v197
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[128:143], v[198:205], v[16:31], v230, v233 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[160:175], v[206:213], v[64:79], v231, v236
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[176:191], v[206:213], v[112:127], v235, v236
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[176:191], v[214:221], v[80:95], v235, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[160:175], v[214:221], v[96:111], v231, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[160:175], v[222:229], v[48:63], v231, v237
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[176:191], v[222:229], v[32:47], v235, v237
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[176:191], v[144:151], v[0:15], v235, v237 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[160:175], v[144:151], v[16:31], v231, v237 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_nop
	v_nop
	v_nop
	v_nop
	v_add_nc_u32_e32 v144, 0xac00, v152
	v_add_nc_u32_e32 v168, 0xac00, v196
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_load_2addr_b32 v[228:229], v144 offset0:128 offset1:160
	ds_load_2addr_b32 v[230:231], v168 offset0:128 offset1:160
	ds_load_b128 v[128:131], v195 offset:44544
	ds_load_b128 v[132:135], v195 offset:45056
	ds_load_b128 v[136:139], v195 offset:46592
	ds_load_b128 v[140:143], v195 offset:47104
	ds_load_b128 v[160:163], v194 offset:44544
	ds_load_b128 v[164:167], v194 offset:44576
	ds_load_2addr_b32 v[232:233], v144 offset0:192 offset1:224
	s_wait_alu depctr_vm_vsrc(0)
	ds_load_b128 v[144:147], v195 offset:48640
	ds_load_b128 v[148:151], v195 offset:49152
	ds_load_b128 v[152:155], v195 offset:50688
	ds_load_b128 v[156:159], v195 offset:51200
	ds_load_b128 v[184:187], v194 offset:49152
	ds_load_b128 v[188:191], v194 offset:49184
	ds_load_b128 v[196:199], v194 offset:51456
	ds_load_b128 v[200:203], v194 offset:51488
	s_wait_dscnt 0x9
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[128:143], v[160:167], v[64:79], v228, v230
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[144:159], v[160:167], v[112:127], v232, v230
	ds_load_b128 v[160:163], v194 offset:46848
	ds_load_b128 v[164:167], v194 offset:46880
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[128:143], v[184:191], v[48:63], v228, v231
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[144:159], v[184:191], v[32:47], v232, v231
	ds_load_b128 v[184:187], v195 offset:51712
	ds_load_b128 v[188:191], v195 offset:52224
	ds_load_b128 v[204:207], v194 offset:44608
	ds_load_b128 v[208:211], v194 offset:44640
	ds_load_b128 v[212:215], v194 offset:46912
	ds_load_b128 v[216:219], v194 offset:46944
	ds_load_b128 v[220:223], v194 offset:49216
	ds_load_b128 v[224:227], v194 offset:49248
	s_wait_dscnt 0x8
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[144:159], v[160:167], v[80:95], v232, v230 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[128:143], v[160:167], v[96:111], v228, v230 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_2addr_b32 v[234:235], v168 offset0:192 offset1:224
	ds_load_b128 v[160:163], v195 offset:45568
	ds_load_b128 v[164:167], v195 offset:46080
	s_wait_alu depctr_vm_vsrc(2)
	ds_load_b128 v[168:171], v195 offset:47616
	ds_load_b128 v[172:175], v195 offset:48128
	ds_load_b128 v[176:179], v195 offset:49664
	ds_load_b128 v[180:183], v195 offset:50176
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[144:159], v[196:203], v[0:15], v232, v231 matrix_b_scale:MATRIX_SCALE_ROW1
	ds_load_b128 v[144:147], v194 offset:51520
	ds_load_b128 v[148:151], v194 offset:51552
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[128:143], v[196:203], v[16:31], v228, v231 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[160:175], v[204:211], v[64:79], v229, v234
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[176:191], v[204:211], v[112:127], v233, v234
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[176:191], v[212:219], v[80:95], v233, v234 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[160:175], v[212:219], v[96:111], v229, v234 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[160:175], v[220:227], v[48:63], v229, v235
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[176:191], v[220:227], v[32:47], v233, v235
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[176:191], v[144:151], v[0:15], v233, v235 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[160:175], v[144:151], v[16:31], v229, v235 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	v_mul_lo_u32 v129, 0x110, v192
	v_lshl_or_b32 v128, v193, 3, s52
	v_cvt_pk_bf16_f32 v71, v70, v71
	v_cvt_pk_bf16_f32 v70, v68, v69
	v_cvt_pk_bf16_f32 v69, v66, v67
	v_cvt_pk_bf16_f32 v68, v64, v65
	v_cvt_pk_bf16_f32 v65, v74, v75
	v_cvt_pk_bf16_f32 v74, v116, v117
	v_add_lshl_u32 v116, v129, v128, 1
	v_cvt_pk_bf16_f32 v67, v78, v79
	v_cvt_pk_bf16_f32 v66, v76, v77
	v_cvt_pk_bf16_f32 v64, v72, v73
	v_cvt_pk_bf16_f32 v75, v118, v119
	v_cvt_pk_bf16_f32 v73, v114, v115
	v_cvt_pk_bf16_f32 v72, v112, v113
	v_cvt_pk_bf16_f32 v79, v126, v127
	v_cvt_pk_bf16_f32 v78, v124, v125
	v_cvt_pk_bf16_f32 v77, v122, v123
	v_cvt_pk_bf16_f32 v76, v120, v121
	s_barrier_wait -1
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v116, v[68:71]
	ds_store_b128 v116, v[64:67] offset:32
	ds_store_b128 v116, v[72:75] offset:64
	ds_store_b128 v116, v[76:79] offset:96
	s_wait_alu depctr_vm_vsrc(2)
	v_cvt_pk_bf16_f32 v67, v102, v103
	v_cvt_pk_bf16_f32 v66, v100, v101
	v_cvt_pk_bf16_f32 v65, v98, v99
	v_cvt_pk_bf16_f32 v64, v96, v97
	v_cvt_pk_bf16_f32 v39, v38, v39
	v_cvt_pk_bf16_f32 v38, v36, v37
	v_cvt_pk_bf16_f32 v37, v34, v35
	v_cvt_pk_bf16_f32 v36, v32, v33
	v_cvt_pk_bf16_f32 v71, v110, v111
	v_cvt_pk_bf16_f32 v70, v108, v109
	v_cvt_pk_bf16_f32 v69, v106, v107
	v_cvt_pk_bf16_f32 v68, v104, v105
	v_cvt_pk_bf16_f32 v35, v46, v47
	v_cvt_pk_bf16_f32 v34, v44, v45
	v_cvt_pk_bf16_f32 v33, v42, v43
	v_cvt_pk_bf16_f32 v32, v40, v41
	s_wait_alu depctr_vm_vsrc(1)
	v_cvt_pk_bf16_f32 v75, v86, v87
	v_cvt_pk_bf16_f32 v74, v84, v85
	v_cvt_pk_bf16_f32 v73, v82, v83
	v_cvt_pk_bf16_f32 v72, v80, v81
	v_cvt_pk_bf16_f32 v23, v22, v23
	v_cvt_pk_bf16_f32 v22, v20, v21
	v_cvt_pk_bf16_f32 v21, v18, v19
	v_cvt_pk_bf16_f32 v20, v16, v17
	s_wait_alu depctr_vm_vsrc(0)
	v_cvt_pk_bf16_f32 v79, v94, v95
	v_cvt_pk_bf16_f32 v78, v92, v93
	v_cvt_pk_bf16_f32 v77, v90, v91
	v_cvt_pk_bf16_f32 v76, v88, v89
	v_cvt_pk_bf16_f32 v19, v30, v31
	v_cvt_pk_bf16_f32 v18, v28, v29
	v_cvt_pk_bf16_f32 v17, v26, v27
	v_cvt_pk_bf16_f32 v16, v24, v25
	s_mul_u64 s[0:1], s[34:35], s[40:41]
	s_bfe_u32 s4, ttmp8, 0x50019
	v_cvt_pk_bf16_f32 v55, v54, v55
	v_cvt_pk_bf16_f32 v54, v52, v53
	v_cvt_pk_bf16_f32 v53, v50, v51
	v_cvt_pk_bf16_f32 v52, v48, v49
	v_cvt_pk_bf16_f32 v7, v6, v7
	v_cvt_pk_bf16_f32 v6, v4, v5
	v_cvt_pk_bf16_f32 v5, v2, v3
	v_cvt_pk_bf16_f32 v4, v0, v1
	v_cvt_pk_bf16_f32 v51, v62, v63
	v_cvt_pk_bf16_f32 v50, v60, v61
	v_cvt_pk_bf16_f32 v49, v58, v59
	v_cvt_pk_bf16_f32 v48, v56, v57
	s_wait_alu depctr_va_vdst(14)
	ds_store_b128 v116, v[64:67] offset:8704
	ds_store_b128 v116, v[68:71] offset:8736
	ds_store_b128 v116, v[72:75] offset:8768
	ds_store_b128 v116, v[76:79] offset:8800
	s_wait_alu depctr_va_vdst(8)
	ds_store_b128 v116, v[52:55] offset:17408
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v116, v[48:51] offset:17440
	v_cvt_pk_bf16_f32 v3, v14, v15
	v_cvt_pk_bf16_f32 v2, v12, v13
	v_cvt_pk_bf16_f32 v1, v10, v11
	v_cvt_pk_bf16_f32 v0, v8, v9
	ds_store_b128 v116, v[36:39] offset:17472
	ds_store_b128 v116, v[32:35] offset:17504
	ds_store_b128 v116, v[20:23] offset:26112
	ds_store_b128 v116, v[16:19] offset:26144
	ds_store_b128 v116, v[4:7] offset:26176
	s_wait_alu depctr_va_vdst(0)
	ds_store_b128 v116, v[0:3] offset:26208
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_lshl_b64 s[0:1], s[0:1], 1
	s_and_b32 s4, s4, 3
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]
	s_lshl_b64 s[2:3], s[42:43], 1
	s_lshl_b32 s6, s4, 5
	s_mov_b32 s7, 0
	s_add_nc_u64 s[0:1], s[0:1], s[2:3]
	s_mul_u64 s[2:3], s[6:7], s[40:41]
	s_lshl_b32 s5, s4, 4
	s_add_nc_u64 s[10:11], s[2:3], s[0:1]
	s_sub_co_i32 s0, s33, s5
	s_mov_b32 s8, 1
	s_max_i32 s0, s0, 0
	s_mul_i32 s9, s4, 0x2200
	s_lshr_b32 s1, s0, 16
	s_lshl_b32 s2, s0, 16
	s_or_b32 s3, s1, 0x1100000
	s_and_b32 s0, s28, exec_lo
	s_bitset1_b32 s11, 31
	s_mov_b32 s4, 16
	s_cselect_b32 s6, 0xffff, 0
	s_mov_b32 s1, 0x1000000
	s_mov_b32 s0, 0x10000
	s_mov_b32 s5, s40
	s_barrier_wait -1
	tensor_store_from_lds s[8:11], s[0:7]
	s_wait_tensorcnt 0x0
.LBB0_17:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96
		.amdhsa_group_segment_fixed_size 89088
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
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 257
		.amdhsa_next_free_sgpr 64
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 56
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
	.size	a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96, .Lfunc_end0-a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96

	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.num_vgpr, 239
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.num_agpr, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.numbered_sgpr, 64
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.num_named_barrier, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.private_seg_size, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.uses_vcc, 1
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.uses_flat_scratch, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.has_dyn_sized_stack, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.has_recursion, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.has_indirect_call, 0
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
    .group_segment_fixed_size: 89088
    .kernarg_segment_align: 8
    .kernarg_segment_size: 184
    .max_flat_workgroup_size: 128
    .name:           a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     66
    .sgpr_spill_count: 0
    .symbol:         a8w4_tdm_fp4_t64x256x256_w1x4_b2_K3072_e96.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     239
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
