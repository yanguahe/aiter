	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4
	.p2align	8
	.type	a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4,@function
a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b96 s[48:50], s[0:1], 0xa4 nv
	v_readfirstlane_b32 s20, v0
	s_wait_kmcnt 0x0
	s_add_co_i32 s4, s49, 0xff
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
	s_add_co_i32 s5, s48, 63
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
	s_bfe_u32 s9, ttmp6, 0x4000c
	s_and_b32 s11, ttmp6, 15
	s_add_co_i32 s9, s9, 1
	s_lshl_b32 s10, s4, 4
	s_mul_i32 s9, ttmp9, s9
	s_and_b32 s7, s8, s7
	s_add_co_i32 s11, s11, s9
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s5, ttmp9, s11
	s_abs_i32 s8, s10
	s_abs_i32 s12, s5
	s_cvt_f32_u32 s9, s8
	s_sub_co_i32 s11, 0, s8
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s9
	v_nop
	v_readfirstlane_b32 s9, v1
	s_mul_f32 s9, s9, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s9, s9
	s_mul_i32 s11, s11, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s11, s9, s11
	s_add_co_i32 s9, s9, s11
	s_xor_b32 s11, s5, s10
	s_mul_hi_u32 s9, s12, s9
	s_ashr_i32 s11, s11, 31
	s_mul_i32 s13, s9, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s12, s12, s13
	s_add_co_i32 s13, s9, 1
	s_sub_co_i32 s14, s12, s8
	s_cmp_ge_u32 s12, s8
	s_cselect_b32 s9, s13, s9
	s_cselect_b32 s12, s14, s12
	s_add_co_i32 s13, s9, 1
	s_cmp_ge_u32 s12, s8
	s_cselect_b32 s8, s13, s9
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s8, s8, s11
	s_sub_co_i32 s9, s8, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s9, s9, s10
	s_cmp_lg_u32 s5, s9
	s_cselect_b32 s9, -1, 0
	s_xor_b32 s4, s4, s5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_cmp_lt_i32 s4, 0
	s_cselect_b32 s4, -1, 0
	s_and_b32 s4, s4, s9
	s_sub_co_ci_u32 s4, s8, s11
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_i32 s8, s4, s10
	s_lshl_b32 s9, s4, 4
	s_sub_co_i32 s8, s5, s8
	s_cmp_lg_u32 s7, 0
	s_sub_co_ci_u32 s6, s6, s9
	s_abs_i32 s11, s8
	s_min_i32 s7, s6, 16
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_abs_i32 s10, s7
	s_xor_b32 s13, s8, s7
	s_cvt_f32_u32 s4, s10
	s_sub_co_i32 s5, 0, s10
	s_ashr_i32 s13, s13, 31
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
	s_add_co_i32 s12, s4, s5
	s_load_b64 s[4:5], s[0:1], 0x70 nv
	s_mul_hi_u32 s12, s11, s12
	s_mul_i32 s14, s12, s10
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s11, s11, s14
	s_add_co_i32 s14, s12, 1
	s_sub_co_i32 s15, s11, s10
	s_cmp_ge_u32 s11, s10
	s_cselect_b32 s12, s14, s12
	s_cselect_b32 s11, s15, s11
	s_add_co_i32 s14, s12, 1
	s_cmp_ge_u32 s11, s10
	s_cselect_b32 s10, s14, s12
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_xor_b32 s10, s10, s13
	s_sub_co_i32 s11, s10, s13
	s_wait_kmcnt 0x0
	s_load_b32 s12, s[4:5], 0xc0
	s_mul_i32 s11, s7, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, s11
	s_cselect_b32 s11, -1, 0
	s_xor_b32 s6, s8, s6
	s_cmp_lt_i32 s6, 0
	s_cselect_b32 s6, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_and_b32 s6, s6, s11
	s_sub_co_ci_u32 s6, s10, s13
	s_add_co_i32 s8, s8, s9
	s_mul_i32 s7, s6, s7
	s_sub_co_i32 s7, s8, s7
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshl_b32 s34, s7, 6
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s12, s34
	s_cselect_b32 s7, 24, 0x48
	s_cselect_b32 s9, 0, 49
	s_load_b32 s8, s[4:5], s7 offset:0x0 scale_offset
	s_cselect_b32 s10, 48, 0x60
	s_or_b32 s11, s7, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s8, s34
	s_cselect_b32 s8, s9, s11
	s_cselect_b32 s7, s7, s10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s8, s7
	s_lshr_b32 s9, s9, 1
	s_load_b32 s10, s[4:5], s9 offset:0x0 scale_offset
	s_or_b32 s11, s9, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s10, s34
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
	s_cmp_gt_i32 s10, s34
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
	s_cmp_gt_i32 s7, s34
	s_cselect_b32 s7, s10, s9
	s_cselect_b32 s8, s12, s8
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s9, s7, s8
	s_lshr_b32 s9, s9, 1
	s_load_b32 s10, s[4:5], s9 offset:0x0 scale_offset
	s_add_co_i32 s11, s9, 1
	s_wait_kmcnt 0x0
	s_cmp_gt_i32 s10, s34
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
	s_cmp_gt_i32 s10, s34
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
	s_cmp_gt_i32 s9, s34
	s_cselect_b32 s54, s7, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_u32 s54, 0x60
	s_cselect_b32 s7, -1, 0
	s_cmp_gt_u32 s54, 0x5f
	s_cbranch_scc1 .LBB0_5
	s_lshl_b32 s52, s6, 8
	s_lshr_b32 s8, s20, 7
	s_ashr_i32 s35, s34, 31
	s_ashr_i32 s53, s52, 31
	s_cmp_lt_i32 s49, 0
	v_bfe_u32 v193, v0, 4, 1
	s_cselect_b32 s33, -1, 0
	s_and_b32 s6, s7, exec_lo
	s_cselect_b32 s6, s54, 0x5f
	v_and_b32_e32 v1, 31, v0
	s_load_b32 s4, s[4:5], s6 offset:0x0 scale_offset
	v_and_b32_e32 v2, 15, v0
	v_and_b32_e32 v3, 16, v0
	s_lshl_b32 s7, s20, 1
	v_lshlrev_b32_e32 v0, 8, v193
	v_lshlrev_b32_e32 v1, 2, v1
	v_lshl_or_b32 v192, s8, 6, v2
	s_wait_xcnt 0x0
	s_lshl_b32 s6, s8, 9
	s_and_b32 s51, s7, 0xc0
	v_lshlrev_b32_e32 v2, 4, v2
	s_add_co_i32 s6, s6, 0xa400
	s_delay_alu instid0(SALU_CYCLE_1)
	v_dual_mov_b32 v127, 0 :: v_dual_bitop2_b32 v196, s6, v1 bitop3:0x54
	s_lshl_b32 s5, s51, 7
	v_mad_u32 v194, 0x90, v192, v3
	s_addk_co_i32 s5, 0x2400
	v_lshl_or_b32 v197, s51, 3, v1
	v_or3_b32 v195, s5, v0, v2
	v_dual_mov_b32 v126, v127 :: v_dual_mov_b32 v125, v127
	v_dual_mov_b32 v124, v127 :: v_dual_mov_b32 v123, v127
	v_dual_mov_b32 v122, v127 :: v_dual_mov_b32 v121, v127
	v_dual_mov_b32 v120, v127 :: v_dual_mov_b32 v119, v127
	v_dual_mov_b32 v118, v127 :: v_dual_mov_b32 v117, v127
	v_dual_mov_b32 v116, v127 :: v_dual_mov_b32 v115, v127
	v_dual_mov_b32 v114, v127 :: v_dual_mov_b32 v113, v127
	v_dual_mov_b32 v112, v127 :: v_dual_mov_b32 v111, v127
	v_dual_mov_b32 v110, v127 :: v_dual_mov_b32 v109, v127
	v_dual_mov_b32 v108, v127 :: v_dual_mov_b32 v107, v127
	v_dual_mov_b32 v106, v127 :: v_dual_mov_b32 v105, v127
	v_dual_mov_b32 v104, v127 :: v_dual_mov_b32 v103, v127
	v_dual_mov_b32 v102, v127 :: v_dual_mov_b32 v101, v127
	v_dual_mov_b32 v100, v127 :: v_dual_mov_b32 v99, v127
	v_dual_mov_b32 v98, v127 :: v_dual_mov_b32 v97, v127
	v_dual_mov_b32 v96, v127 :: v_dual_mov_b32 v95, v127
	v_dual_mov_b32 v94, v127 :: v_dual_mov_b32 v93, v127
	v_dual_mov_b32 v92, v127 :: v_dual_mov_b32 v91, v127
	v_dual_mov_b32 v90, v127 :: v_dual_mov_b32 v89, v127
	v_dual_mov_b32 v88, v127 :: v_dual_mov_b32 v87, v127
	v_dual_mov_b32 v86, v127 :: v_dual_mov_b32 v85, v127
	v_dual_mov_b32 v84, v127 :: v_dual_mov_b32 v83, v127
	v_dual_mov_b32 v82, v127 :: v_dual_mov_b32 v81, v127
	v_dual_mov_b32 v80, v127 :: v_dual_mov_b32 v79, v127
	v_dual_mov_b32 v78, v127 :: v_dual_mov_b32 v77, v127
	v_dual_mov_b32 v76, v127 :: v_dual_mov_b32 v75, v127
	v_dual_mov_b32 v74, v127 :: v_dual_mov_b32 v73, v127
	v_dual_mov_b32 v72, v127 :: v_dual_mov_b32 v71, v127
	v_dual_mov_b32 v70, v127 :: v_dual_mov_b32 v69, v127
	v_dual_mov_b32 v68, v127 :: v_dual_mov_b32 v67, v127
	v_dual_mov_b32 v66, v127 :: v_dual_mov_b32 v65, v127
	v_dual_mov_b32 v64, v127 :: v_dual_mov_b32 v63, v127
	v_dual_mov_b32 v62, v127 :: v_dual_mov_b32 v61, v127
	v_dual_mov_b32 v60, v127 :: v_dual_mov_b32 v59, v127
	v_dual_mov_b32 v58, v127 :: v_dual_mov_b32 v57, v127
	v_dual_mov_b32 v56, v127 :: v_dual_mov_b32 v55, v127
	v_dual_mov_b32 v54, v127 :: v_dual_mov_b32 v53, v127
	v_dual_mov_b32 v52, v127 :: v_dual_mov_b32 v51, v127
	v_dual_mov_b32 v50, v127 :: v_dual_mov_b32 v49, v127
	v_dual_mov_b32 v48, v127 :: v_dual_mov_b32 v47, v127
	v_dual_mov_b32 v46, v127 :: v_dual_mov_b32 v45, v127
	v_dual_mov_b32 v44, v127 :: v_dual_mov_b32 v43, v127
	v_dual_mov_b32 v42, v127 :: v_dual_mov_b32 v41, v127
	v_dual_mov_b32 v40, v127 :: v_dual_mov_b32 v39, v127
	v_dual_mov_b32 v38, v127 :: v_dual_mov_b32 v37, v127
	v_dual_mov_b32 v36, v127 :: v_dual_mov_b32 v35, v127
	v_dual_mov_b32 v34, v127 :: v_dual_mov_b32 v33, v127
	v_dual_mov_b32 v32, v127 :: v_dual_mov_b32 v31, v127
	v_dual_mov_b32 v30, v127 :: v_dual_mov_b32 v29, v127
	v_dual_mov_b32 v28, v127 :: v_dual_mov_b32 v27, v127
	v_dual_mov_b32 v26, v127 :: v_dual_mov_b32 v25, v127
	v_dual_mov_b32 v24, v127 :: v_dual_mov_b32 v23, v127
	v_dual_mov_b32 v22, v127 :: v_dual_mov_b32 v21, v127
	v_dual_mov_b32 v20, v127 :: v_dual_mov_b32 v19, v127
	v_dual_mov_b32 v18, v127 :: v_dual_mov_b32 v17, v127
	v_dual_mov_b32 v16, v127 :: v_dual_mov_b32 v15, v127
	v_dual_mov_b32 v14, v127 :: v_dual_mov_b32 v13, v127
	v_dual_mov_b32 v12, v127 :: v_dual_mov_b32 v11, v127
	v_dual_mov_b32 v10, v127 :: v_dual_mov_b32 v9, v127
	v_dual_mov_b32 v8, v127 :: v_dual_mov_b32 v7, v127
	v_dual_mov_b32 v6, v127 :: v_dual_mov_b32 v5, v127
	v_dual_mov_b32 v4, v127 :: v_dual_mov_b32 v3, v127
	v_dual_mov_b32 v2, v127 :: v_dual_mov_b32 v1, v127
	v_mov_b32_e32 v0, v127
	s_wait_kmcnt 0x0
	s_sub_co_i32 s48, s4, s34
	s_cmp_gt_u32 s20, 0x7f
	s_cbranch_scc1 .LBB0_4
	s_ashr_i32 s5, s49, 31
	s_clause 0x1
	s_load_b128 s[44:47], s[0:1], 0x28 nv
	s_load_b64 s[56:57], s[0:1], 0x38 nv
	s_lshr_b32 s4, s5, 28
	s_lshr_b32 s60, s20, 5
	s_add_co_i32 s4, s49, s4
	s_mov_b32 s55, 0
	s_ashr_i32 s12, s4, 4
	s_mov_b32 s4, s49
	s_ashr_i32 s13, s12, 31
	s_add_nc_u64 s[8:9], s[4:5], 31
	s_lshl_b64 s[6:7], s[12:13], 4
	s_mul_i32 s68, s60, 0x900
	s_cmp_lg_u64 s[6:7], s[4:5]
	s_mov_b32 s5, s55
	s_cselect_b32 s10, -1, 0
	s_lshr_b32 s4, s9, 27
	s_mov_b64 s[6:7], 0xffffffffffffffe0
	s_add_nc_u64 s[4:5], s[8:9], s[4:5]
	s_and_b32 s14, s33, s10
	s_and_b64 s[6:7], s[4:5], s[6:7]
	s_ashr_i64 s[62:63], s[4:5], 5
	s_cmp_lg_u64 s[8:9], s[6:7]
	s_mov_b32 s5, s55
	s_cselect_b32 s40, -1, 0
	s_cmp_lt_i32 s49, 0xffffffe1
	s_mov_b32 s28, 1
	s_cselect_b32 s41, -1, 0
	s_lshl_b32 s4, s60, 4
	s_mov_b32 s29, s68
	s_add_nc_u64 s[6:7], s[34:35], s[4:5]
	s_sub_co_i32 s4, s48, s4
	s_mul_u64 s[6:7], s[6:7], 0xe00
	s_max_i32 s4, s4, 0
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[30:31], s[44:45], s[6:7]
	s_mov_b64 s[16:17], s[28:29]
	s_or_b32 s5, s31, 0x80000000
	s_mov_b64 s[18:19], s[30:31]
	s_mov_b32 s19, s5
	s_lshl_b32 s5, s4, 16
	s_lshr_b32 s4, s4, 16
	s_mov_b32 s8, 16
	s_or_b32 s6, s5, 0x7fff
	s_or_b32 s7, s4, 0x800000
	s_movk_i32 s9, 0xe00
	s_mov_b32 s5, 0xffff0000
	s_mov_b32 s4, 0x7100000
	s_mov_b32 s10, s55
	s_mov_b32 s11, s55
	v_cndmask_b32_e64 v0, 0, 1, s14
	tensor_load_to_lds s[16:19], s[4:11]
	s_mov_b32 s15, s55
	s_lshl_b32 s16, s60, 2
	s_delay_alu instid0(VALU_DEP_2)
	v_readfirstlane_b32 s14, v0
	s_mov_b32 s17, s55
	s_lshl_b32 s69, s60, 13
	s_mov_b32 s18, s55
	s_add_co_i32 s29, s69, 0x2400
	s_sub_nc_u64 s[12:13], s[12:13], s[14:15]
	s_lshr_b64 s[14:15], s[52:53], 4
	s_mul_u64 s[12:13], s[12:13], s[54:55]
	s_add_nc_u64 s[14:15], s[14:15], s[16:17]
	s_mov_b64 s[24:25], s[28:29]
	s_add_nc_u64 s[12:13], s[14:15], s[12:13]
	s_mov_b64 s[26:27], s[30:31]
	s_mul_u64 s[12:13], s[12:13], 0xe000
	s_mov_b32 s16, 4
	s_add_nc_u64 s[58:59], s[46:47], s[12:13]
	s_mov_b32 s17, 0xe000
	s_or_b32 s12, s59, 0x80000000
	s_mov_b32 s26, s58
	s_mov_b32 s27, s12
	s_mov_b32 s15, 0x8007fff
	s_mov_b32 s14, 0xffff7fff
	s_mov_b32 s12, s55
	s_mov_b32 s13, s5
	s_mov_b32 s19, s55
	s_ashr_i64 s[46:47], s[34:35], 6
	s_and_not1_b32 s20, s20, 31
	s_mul_u64 s[64:65], s[46:47], 0x3800
	s_ashr_i32 s21, s20, 31
	s_add_nc_u64 s[22:23], s[56:57], s[64:65]
	s_lshl_b64 s[66:67], s[20:21], 2
	s_lshl_b32 s70, s20, 2
	s_add_nc_u64 s[76:77], s[22:23], s[66:67]
	s_add_co_i32 s29, s70, 0xa400
	s_or_b32 s20, s77, 0x80000000
	s_mov_b64 s[38:39], s[30:31]
	s_mov_b64 s[36:37], s[28:29]
	s_mov_b32 s38, s76
	s_mov_b32 s39, s20
	s_mov_b32 s23, 0x207fff
	s_mov_b32 s20, 0x20000
	s_mov_b32 s21, s5
	s_mov_b32 s22, s14
	s_load_b64 s[78:79], s[0:1], 0x60 nv
	s_and_b32 s47, s41, s40
	s_wait_xcnt 0x0
	s_mov_b32 s1, s55
	v_cndmask_b32_e64 v0, 0, 1, s47
	s_ashr_i64 s[80:81], s[52:53], 5
	s_mov_b32 s40, 2
	s_mul_u64 s[82:83], s[80:81], 0x1c00
	s_movk_i32 s41, 0x700
	v_readfirstlane_b32 s0, v0
	s_mov_b32 s42, s55
	s_mov_b32 s43, s55
	s_mov_b32 s61, s55
	v_mov_b32_e32 v112, 0
	s_sub_nc_u64 s[0:1], s[62:63], s[0:1]
	s_lshl_b32 s63, s60, 9
	s_mul_u64 s[0:1], s[0:1], s[54:55]
	s_add_co_i32 s29, s63, 0xa600
	s_mul_u64 s[0:1], s[0:1], 0x1c00
	s_mov_b64 s[74:75], s[30:31]
	s_mov_b64 s[72:73], s[28:29]
	s_add_nc_u64 s[30:31], s[30:31], 0x80
	s_add_co_i32 s29, s68, 0xae00
	s_bitset1_b32 s31, 31
	v_dual_mov_b32 v113, v112 :: v_dual_mov_b32 v114, v112
	v_dual_mov_b32 v115, v112 :: v_dual_mov_b32 v116, v112
	v_dual_mov_b32 v117, v112 :: v_dual_mov_b32 v118, v112
	v_dual_mov_b32 v119, v112 :: v_dual_mov_b32 v120, v112
	v_dual_mov_b32 v121, v112 :: v_dual_mov_b32 v122, v112
	v_dual_mov_b32 v123, v112 :: v_dual_mov_b32 v124, v112
	v_dual_mov_b32 v125, v112 :: v_dual_mov_b32 v126, v112
	v_dual_mov_b32 v127, v112 :: v_dual_mov_b32 v96, v112
	v_dual_mov_b32 v97, v112 :: v_dual_mov_b32 v98, v112
	v_dual_mov_b32 v99, v112 :: v_dual_mov_b32 v100, v112
	v_dual_mov_b32 v101, v112 :: v_dual_mov_b32 v102, v112
	v_dual_mov_b32 v103, v112 :: v_dual_mov_b32 v104, v112
	v_dual_mov_b32 v105, v112 :: v_dual_mov_b32 v106, v112
	v_dual_mov_b32 v107, v112 :: v_dual_mov_b32 v108, v112
	v_dual_mov_b32 v109, v112 :: v_dual_mov_b32 v110, v112
	v_dual_mov_b32 v111, v112 :: v_dual_mov_b32 v80, v112
	v_dual_mov_b32 v81, v112 :: v_dual_mov_b32 v82, v112
	v_dual_mov_b32 v83, v112 :: v_dual_mov_b32 v84, v112
	v_dual_mov_b32 v85, v112 :: v_dual_mov_b32 v86, v112
	v_dual_mov_b32 v87, v112 :: v_dual_mov_b32 v88, v112
	v_dual_mov_b32 v89, v112 :: v_dual_mov_b32 v90, v112
	v_dual_mov_b32 v91, v112 :: v_dual_mov_b32 v92, v112
	v_dual_mov_b32 v93, v112 :: v_dual_mov_b32 v94, v112
	v_dual_mov_b32 v95, v112 :: v_dual_mov_b32 v64, v112
	v_dual_mov_b32 v65, v112 :: v_dual_mov_b32 v66, v112
	v_dual_mov_b32 v67, v112 :: v_dual_mov_b32 v68, v112
	v_dual_mov_b32 v69, v112 :: v_dual_mov_b32 v70, v112
	v_dual_mov_b32 v71, v112 :: v_dual_mov_b32 v72, v112
	v_dual_mov_b32 v73, v112 :: v_dual_mov_b32 v74, v112
	v_dual_mov_b32 v75, v112 :: v_dual_mov_b32 v76, v112
	v_dual_mov_b32 v77, v112 :: v_dual_mov_b32 v78, v112
	v_dual_mov_b32 v79, v112 :: v_dual_mov_b32 v48, v112
	v_dual_mov_b32 v49, v112 :: v_dual_mov_b32 v50, v112
	v_dual_mov_b32 v51, v112 :: v_dual_mov_b32 v52, v112
	v_dual_mov_b32 v53, v112 :: v_dual_mov_b32 v54, v112
	v_dual_mov_b32 v55, v112 :: v_dual_mov_b32 v56, v112
	v_dual_mov_b32 v57, v112 :: v_dual_mov_b32 v58, v112
	v_dual_mov_b32 v59, v112 :: v_dual_mov_b32 v60, v112
	v_dual_mov_b32 v61, v112 :: v_dual_mov_b32 v62, v112
	v_dual_mov_b32 v63, v112 :: v_dual_mov_b32 v32, v112
	v_dual_mov_b32 v33, v112 :: v_dual_mov_b32 v34, v112
	v_dual_mov_b32 v35, v112 :: v_dual_mov_b32 v36, v112
	v_dual_mov_b32 v37, v112 :: v_dual_mov_b32 v38, v112
	v_dual_mov_b32 v39, v112 :: v_dual_mov_b32 v40, v112
	v_dual_mov_b32 v41, v112 :: v_dual_mov_b32 v42, v112
	v_dual_mov_b32 v43, v112 :: v_dual_mov_b32 v44, v112
	v_dual_mov_b32 v45, v112 :: v_dual_mov_b32 v46, v112
	v_dual_mov_b32 v47, v112 :: v_dual_mov_b32 v16, v112
	v_dual_mov_b32 v17, v112 :: v_dual_mov_b32 v18, v112
	v_dual_mov_b32 v19, v112 :: v_dual_mov_b32 v20, v112
	v_dual_mov_b32 v21, v112 :: v_dual_mov_b32 v22, v112
	v_dual_mov_b32 v23, v112 :: v_dual_mov_b32 v24, v112
	v_dual_mov_b32 v25, v112 :: v_dual_mov_b32 v26, v112
	v_dual_mov_b32 v27, v112 :: v_dual_mov_b32 v28, v112
	v_dual_mov_b32 v29, v112 :: v_dual_mov_b32 v30, v112
	v_dual_mov_b32 v31, v112 :: v_dual_mov_b32 v0, v112
	v_dual_mov_b32 v1, v112 :: v_dual_mov_b32 v2, v112
	v_dual_mov_b32 v3, v112 :: v_dual_mov_b32 v4, v112
	v_dual_mov_b32 v5, v112 :: v_dual_mov_b32 v6, v112
	v_dual_mov_b32 v7, v112 :: v_dual_mov_b32 v8, v112
	v_dual_mov_b32 v9, v112 :: v_dual_mov_b32 v10, v112
	v_dual_mov_b32 v11, v112 :: v_dual_mov_b32 v12, v112
	v_dual_mov_b32 v13, v112 :: v_dual_mov_b32 v14, v112
	v_mov_b32_e32 v15, v112
	tensor_load_to_lds s[24:27], s[12:19]
	s_mov_b32 s24, s28
	s_mov_b32 s25, s9
	s_mov_b32 s26, s55
	s_mov_b32 s27, s55
	s_delay_alu instid0(SALU_CYCLE_1)
	tensor_load_to_lds s[36:39], s[20:27]
	s_lshl_b32 s36, s60, 1
	s_wait_kmcnt 0x0
	s_add_nc_u64 s[38:39], s[78:79], s[82:83]
	s_mov_b32 s37, s55
	s_add_nc_u64 s[38:39], s[38:39], s[0:1]
	s_mul_u64 s[84:85], s[36:37], 0x1c00
	s_mov_b32 s37, s5
	s_add_nc_u64 s[86:87], s[38:39], s[84:85]
	s_mov_b32 s39, 0x407fff
	s_or_b32 s36, s87, 0x80000000
	s_mov_b32 s74, s86
	s_mov_b32 s75, s36
	s_mov_b32 s36, s20
	s_mov_b32 s38, s14
	s_add_nc_u64 s[0:1], s[78:79], s[0:1]
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_add_nc_u64 s[0:1], s[0:1], s[84:85]
	s_add_nc_u64 s[0:1], s[0:1], s[82:83]
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[0:1], s[0:1], 0x200
	tensor_load_to_lds s[72:75], s[36:43]
	tensor_load_to_lds s[28:31], s[4:11]
	s_add_nc_u64 s[30:31], s[58:59], 0x800
	s_add_co_i32 s29, s69, 0xd200
	s_bitset1_b32 s31, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	tensor_load_to_lds s[28:31], s[12:19]
	s_add_nc_u64 s[30:31], s[76:77], 0x200
	s_add_co_i32 s29, s70, 0x15200
	s_bitset1_b32 s31, 31
	tensor_load_to_lds s[28:31], s[20:27]
	s_add_nc_u64 s[30:31], s[86:87], 0x100
	s_add_co_i32 s29, s63, 0x15400
	s_bitset1_b32 s31, 31
	s_cmp_lg_u32 s47, 0
	tensor_load_to_lds s[28:31], s[36:43]
	s_sub_co_ci_u32 s29, s62, 0
	s_mul_i32 s30, s60, 0x3800
	s_mul_i32 s29, s54, s29
	s_lshl_b32 s62, s60, 7
	s_mulk_i32 s29, 0x1c00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s29, s29, s30
	s_mul_i32 s30, s80, 0x1c00
	s_add_co_i32 s29, s29, s30
	s_add_nc_u64 s[30:31], s[64:65], s[66:67]
	s_add_co_i32 s29, s29, s78
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_co_i32 s54, s29, 0x200
	s_mul_i32 s29, s46, 0x3800
	s_add_nc_u64 s[46:47], s[30:31], 0x400
	s_mul_u64 s[30:31], s[60:61], 0xe000
	s_add_co_i32 s29, s29, s62
	s_add_nc_u64 s[30:31], s[44:45], s[30:31]
	s_mul_u64 s[44:45], s[34:35], 0xe00
	s_add_co_i32 s60, s29, 0x400
	s_add_nc_u64 s[30:31], s[30:31], s[44:45]
	s_add_nc_u64 s[44:45], s[58:59], 0x1000
	s_add_nc_u64 s[30:31], s[30:31], 0x100
	s_mov_b64 s[58:59], 0
.LBB0_3:
	s_wait_tensorcnt 0x4
	s_barrier_signal -1
	s_and_b32 s61, s55, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s61, s61, 0xae00
	v_dual_add_nc_u32 v128, s61, v197 :: v_dual_add_nc_u32 v168, s61, v196
	s_wait_alu depctr_vm_vsrc(0)
	v_dual_add_nc_u32 v206, s61, v195 :: v_dual_add_nc_u32 v238, s61, v194
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
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[128:143], v[160:167], v[112:127], v230, v232
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[144:159], v[160:167], v[96:111], v234, v232
	ds_load_b128 v[160:163], v238 offset:2304
	ds_load_b128 v[164:167], v238 offset:2336
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[128:143], v[184:191], v[48:63], v230, v233
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[144:159], v[160:167], v[64:79], v234, v232 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[128:143], v[160:167], v[80:95], v230, v232 matrix_b_scale:MATRIX_SCALE_ROW1
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
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[160:175], v[206:213], v[112:127], v231, v236
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[176:191], v[206:213], v[96:111], v235, v236
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[176:191], v[214:221], v[64:79], v235, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[160:175], v[214:221], v[80:95], v231, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[160:175], v[222:229], v[48:63], v231, v237
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[176:191], v[222:229], v[32:47], v235, v237
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[176:191], v[144:151], v[0:15], v235, v237 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[160:175], v[144:151], v[16:31], v231, v237 matrix_b_scale:MATRIX_SCALE_ROW1
	s_barrier_signal -1
	s_add_co_i32 s29, s61, s68
	s_or_b32 s62, s31, 0x80000000
	s_mov_b64 s[66:67], s[30:31]
	s_mov_b64 s[64:65], s[28:29]
	s_mov_b32 s67, s62
	s_add_co_i32 s29, s61, s69
	s_or_b32 s62, s45, 0x80000000
	s_addk_co_i32 s29, 0x2400
	s_add_co_i32 s55, s55, 1
	s_barrier_wait -1
	tensor_load_to_lds s[64:67], s[4:11]
	s_mov_b64 s[66:67], s[30:31]
	s_mov_b64 s[64:65], s[28:29]
	s_mov_b32 s66, s44
	s_mov_b32 s67, s62
	s_add_co_i32 s29, s70, s61
	s_add_co_i32 s62, s60, s56
	s_add_co_i32 s29, s29, 0xa400
	s_add_nc_u64 s[44:45], s[44:45], 0x800
	tensor_load_to_lds s[64:67], s[12:19]
	s_add_nc_u64 s[64:65], s[46:47], s[56:57]
	s_add_nc_u64 s[56:57], s[56:57], 0x200
	s_or_b32 s71, s65, 0x80000000
	s_mov_b64 s[66:67], s[30:31]
	s_mov_b64 s[64:65], s[28:29]
	s_mov_b32 s66, s62
	s_mov_b32 s67, s71
	s_add_co_i32 s29, s63, s61
	s_add_co_i32 s61, s54, s58
	s_add_co_i32 s29, s29, 0xa600
	tensor_load_to_lds s[64:67], s[20:27]
	s_add_nc_u64 s[64:65], s[0:1], s[58:59]
	s_add_nc_u64 s[58:59], s[58:59], 0x100
	s_or_b32 s62, s65, 0x80000000
	s_mov_b64 s[66:67], s[30:31]
	s_mov_b64 s[64:65], s[28:29]
	s_mov_b32 s66, s61
	s_mov_b32 s67, s62
	s_cmp_lg_u32 s58, 0x1a00
	s_add_nc_u64 s[30:31], s[30:31], 0x80
	tensor_load_to_lds s[64:67], s[36:43]
	s_cbranch_scc1 .LBB0_3
.LBB0_4:
	s_wait_tensorcnt 0x4
	s_barrier_signal -1
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
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[128:143], v[160:167], v[112:127], v230, v232
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[144:159], v[160:167], v[96:111], v234, v232
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
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[144:159], v[160:167], v[64:79], v234, v232 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[128:143], v[160:167], v[80:95], v230, v232 matrix_b_scale:MATRIX_SCALE_ROW1
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
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[160:175], v[206:213], v[112:127], v231, v236
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[176:191], v[206:213], v[96:111], v235, v236
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[176:191], v[214:221], v[64:79], v235, v236 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[160:175], v[214:221], v[80:95], v231, v236 matrix_b_scale:MATRIX_SCALE_ROW1
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
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[128:143], v[160:167], v[112:127], v228, v230
	s_wait_dscnt 0x4
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[144:159], v[160:167], v[96:111], v232, v230
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
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[144:159], v[160:167], v[64:79], v232, v230 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[128:143], v[160:167], v[80:95], v228, v230 matrix_b_scale:MATRIX_SCALE_ROW1
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
	v_wmma_scale_f32_32x16x128_f4 v[112:127], v[160:175], v[204:211], v[112:127], v229, v234
	s_wait_dscnt 0x2
	v_wmma_scale_f32_32x16x128_f4 v[96:111], v[176:191], v[204:211], v[96:111], v233, v234
	v_wmma_scale_f32_32x16x128_f4 v[64:79], v[176:191], v[212:219], v[64:79], v233, v234 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[80:95], v[160:175], v[212:219], v[80:95], v229, v234 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[48:63], v[160:175], v[220:227], v[48:63], v229, v235
	v_wmma_scale_f32_32x16x128_f4 v[32:47], v[176:191], v[220:227], v[32:47], v233, v235
	s_wait_dscnt 0x0
	v_wmma_scale_f32_32x16x128_f4 v[0:15], v[176:191], v[144:151], v[0:15], v233, v235 matrix_b_scale:MATRIX_SCALE_ROW1
	v_wmma_scale_f32_32x16x128_f4 v[16:31], v[160:175], v[144:151], v[16:31], v229, v235 matrix_b_scale:MATRIX_SCALE_ROW1
	v_max_num_f32_e64 v134, s50, s50
	v_dual_max_num_f32 v116, v116, v116 :: v_dual_max_num_f32 v118, v118, v118
	v_dual_max_num_f32 v112, v112, v112 :: v_dual_max_num_f32 v114, v114, v114
	v_max_num_f32_e32 v120, v120, v120
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_min_num_f32 v130, v116, v134 :: v_dual_min_num_f32 v131, v118, v134
	v_dual_min_num_f32 v128, v112, v134 :: v_dual_min_num_f32 v129, v114, v134
	v_max_num_f32_e32 v96, v96, v96
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v116, 0xbfb8aa3b, v130
	v_max_num_f32_e32 v104, v104, v104
	v_med3_num_f32 v103, v103, -s50, s50
	v_mul_f32_e32 v114, 0xbfb8aa3b, v129
	v_dual_max_num_f32 v80, v80, v80 :: v_dual_max_num_f32 v86, v86, v86
	v_exp_f32_e32 v116, v116
	v_max_num_f32_e32 v90, v90, v90
	v_dual_max_num_f32 v94, v94, v94 :: v_dual_max_num_f32 v64, v64, v64
	v_med3_num_f32 v71, v71, -s50, s50
	v_dual_max_num_f32 v48, v48, v48 :: v_dual_max_num_f32 v50, v50, v50
	v_lshl_or_b32 v135, v193, 3, s51
	s_delay_alu instid0(TRANS32_DEP_1)
	v_add_f32_e32 v136, 1.0, v116
	v_med3_num_f32 v116, v117, -s50, s50
	v_med3_num_f32 v117, v119, -s50, s50
	v_max_num_f32_e32 v119, v122, v122
	v_exp_f32_e32 v114, v114
	v_mul_f32_e32 v112, 0xbfb8aa3b, v128
	v_mul_f32_e32 v118, 0xbfb8aa3b, v131
	v_max_num_f32_e32 v122, v126, v126
	v_dual_min_num_f32 v119, v119, v134 :: v_dual_max_num_f32 v56, v56, v56
	v_max_num_f32_e32 v58, v58, v58
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v118, v118
	v_add_f32_e32 v133, 1.0, v114
	v_exp_f32_e32 v132, v112
	v_nop
	v_med3_num_f32 v112, v113, -s50, s50
	v_med3_num_f32 v113, v115, -s50, s50
	v_dual_max_num_f32 v60, v60, v60 :: v_dual_max_num_f32 v62, v62, v62
	v_rcp_f32_e32 v115, v133
	v_dual_add_f32 v118, 1.0, v118 :: v_dual_max_num_f32 v32, v32, v32
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_add_f32_e32 v132, 1.0, v132
	v_med3_num_f32 v39, v39, -s50, s50
	v_dual_max_num_f32 v22, v22, v22 :: v_dual_max_num_f32 v28, v28, v28
	v_rcp_f32_e32 v133, v118
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v114, v132
	v_min_num_f32_e32 v118, v120, v134
	v_rcp_f32_e32 v132, v136
	v_dual_max_num_f32 v120, v124, v124 :: v_dual_max_num_f32 v26, v26, v26
	v_max_num_f32_e32 v0, v0, v0
	v_mul_f32_e32 v124, 0xbfb8aa3b, v118
	v_dual_max_num_f32 v30, v30, v30 :: v_dual_max_num_f32 v4, v4, v4
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_pk_mul_f32 v[114:115], v[128:129], v[114:115]
	v_pk_mul_f32 v[128:129], v[130:131], v[132:133]
	v_dual_min_num_f32 v131, v122, v134 :: v_dual_min_num_f32 v130, v120, v134
	v_exp_f32_e32 v120, v124
	v_mul_f32_e32 v122, 0xbfb8aa3b, v119
	s_delay_alu instid0(VALU_DEP_3)
	v_pk_mul_f32 v[116:117], v[116:117], v[128:129]
	v_pk_mul_f32 v[112:113], v[112:113], v[114:115]
	v_lshl_or_b32 v128, v192, 8, v135
	v_max_num_f32_e32 v14, v14, v14
	v_max_num_f32_e32 v10, v10, v10
	v_cvt_pk_bf16_f32 v115, v116, v117
	v_add_f32_e32 v116, 1.0, v120
	v_mul_f32_e32 v124, 0xbfb8aa3b, v130
	v_exp_f32_e32 v117, v122
	v_med3_num_f32 v27, v27, -s50, s50
	s_wait_tensorcnt 0x0
	v_rcp_f32_e32 v116, v116
	v_exp_f32_e32 v122, v124
	v_mul_f32_e32 v114, 0xbfb8aa3b, v131
	v_max_num_f32_e32 v100, v100, v100
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v117, 1.0, v117
	s_barrier_signal -1
	v_exp_f32_e32 v120, v114
	v_nop
	v_cvt_pk_bf16_f32 v114, v112, v113
	v_med3_num_f32 v112, v121, -s50, s50
	v_add_f32_e32 v121, 1.0, v122
	v_min_num_f32_e32 v122, v96, v134
	v_max_num_f32_e32 v96, v98, v98
	v_med3_num_f32 v113, v123, -s50, s50
	v_dual_min_num_f32 v126, v100, v134 :: v_dual_add_f32 v124, 1.0, v120
	v_rcp_f32_e32 v117, v117
	s_delay_alu instid0(VALU_DEP_3)
	v_min_num_f32_e32 v123, v96, v134
	v_rcp_f32_e32 v120, v121
	v_mul_f32_e32 v98, 0xbfb8aa3b, v122
	v_rcp_f32_e32 v121, v124
	v_nop
	v_med3_num_f32 v124, v125, -s50, s50
	v_mul_f32_e32 v100, 0xbfb8aa3b, v123
	v_med3_num_f32 v125, v127, -s50, s50
	v_pk_mul_f32 v[116:117], v[118:119], v[116:117]
	v_exp_f32_e32 v98, v98
	s_lshr_b32 s0, s49, 31
	v_exp_f32_e32 v100, v100
	v_max_num_f32_e32 v96, v102, v102
	v_pk_mul_f32 v[112:113], v[112:113], v[116:117]
	v_pk_mul_f32 v[120:121], v[130:131], v[120:121]
	v_med3_num_f32 v3, v3, -s50, s50
	s_add_co_i32 s0, s49, s0
	v_add_f32_e32 v98, 1.0, v98
	s_and_b32 s1, s0, -2
	v_dual_add_f32 v100, 1.0, v100 :: v_dual_min_num_f32 v127, v96, v134
	v_mul_f32_e32 v96, 0xbfb8aa3b, v126
	v_pk_mul_f32 v[118:119], v[124:125], v[120:121]
	v_rcp_f32_e32 v98, v98
	s_ashr_i32 s0, s0, 1
	s_cmp_lg_u32 s49, s1
	v_exp_f32_e32 v116, v96
	v_nop
	v_med3_num_f32 v96, v97, -s50, s50
	v_med3_num_f32 v97, v99, -s50, s50
	v_rcp_f32_e32 v99, v100
	v_nop
	v_min_num_f32_e32 v100, v104, v134
	v_dual_max_num_f32 v104, v106, v106 :: v_dual_max_num_f32 v106, v108, v108
	v_mul_f32_e32 v102, 0xbfb8aa3b, v127
	v_cvt_pk_bf16_f32 v117, v118, v119
	v_dual_add_f32 v116, 1.0, v116 :: v_dual_max_num_f32 v108, v110, v110
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_num_f32_e32 v120, v106, v134
	v_exp_f32_e32 v102, v102
	v_pk_mul_f32 v[98:99], v[122:123], v[98:99]
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v118, v116
	v_nop
	v_cvt_pk_bf16_f32 v116, v112, v113
	v_mul_f32_e32 v106, 0xbfb8aa3b, v120
	s_cselect_b32 s1, -1, 0
	v_pk_mul_f32 v[96:97], v[96:97], v[98:99]
	s_and_b32 s1, s33, s1
	v_add_f32_e32 v119, 1.0, v102
	v_med3_num_f32 v102, v101, -s50, s50
	v_mul_f32_e32 v101, 0xbfb8aa3b, v100
	s_barrier_wait -1
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v119, v119
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v128, v[114:115], v[116:117] offset1:2
	v_exp_f32_e32 v110, v101
	v_nop
	v_min_num_f32_e32 v101, v104, v134
	s_bfe_u32 s4, ttmp8, 0x50019
	s_mov_b32 s7, 0
	s_and_b32 s4, s4, 3
	s_mov_b32 s8, 1
	v_pk_mul_f32 v[118:119], v[126:127], v[118:119]
	v_mul_f32_e32 v104, 0xbfb8aa3b, v101
	s_lshl_b32 s6, s4, 5
	s_lshl_b32 s5, s4, 4
	s_lshl_b32 s9, s4, 12
	v_pk_mul_f32 v[98:99], v[102:103], v[118:119]
	v_add_f32_e32 v102, 1.0, v110
	v_min_num_f32_e32 v121, v108, v134
	v_exp_f32_e32 v103, v104
	v_exp_f32_e32 v104, v106
	v_cvt_pk_bf16_f32 v99, v98, v99
	v_cvt_pk_bf16_f32 v98, v96, v97
	v_mul_f32_e32 v108, 0xbfb8aa3b, v121
	v_med3_num_f32 v96, v105, -s50, s50
	v_rcp_f32_e32 v102, v102
	v_med3_num_f32 v97, v107, -s50, s50
	v_add_f32_e32 v103, 1.0, v103
	v_exp_f32_e32 v106, v108
	v_add_f32_e32 v104, 1.0, v104
	v_med3_num_f32 v108, v109, -s50, s50
	v_med3_num_f32 v109, v111, -s50, s50
	v_rcp_f32_e32 v103, v103
	s_mov_b32 s4, 16
	v_max_num_f32_e32 v88, v88, v88
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_max_num_f32 v92, v92, v92 :: v_dual_add_f32 v105, 1.0, v106
	v_min_num_f32_e32 v106, v80, v134
	v_rcp_f32_e32 v104, v104
	v_dual_max_num_f32 v80, v82, v82 :: v_dual_max_num_f32 v82, v84, v84
	v_rcp_f32_e32 v105, v105
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v84, 0xbfb8aa3b, v106
	v_pk_mul_f32 v[100:101], v[100:101], v[102:103]
	v_dual_min_num_f32 v107, v80, v134 :: v_dual_min_num_f32 v111, v86, v134
	v_max_num_f32_e32 v72, v72, v72
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v80, v84
	v_min_num_f32_e32 v110, v82, v134
	v_pk_mul_f32 v[104:105], v[120:121], v[104:105]
	v_mul_f32_e32 v82, 0xbfb8aa3b, v107
	v_mul_f32_e32 v86, 0xbfb8aa3b, v111
	v_pk_mul_f32 v[96:97], v[96:97], v[100:101]
	v_max_num_f32_e32 v68, v68, v68
	v_pk_mul_f32 v[102:103], v[108:109], v[104:105]
	v_add_f32_e32 v104, 1.0, v80
	v_mul_f32_e32 v84, 0xbfb8aa3b, v110
	v_exp_f32_e32 v86, v86
	v_exp_f32_e32 v82, v82
	v_cvt_pk_bf16_f32 v101, v102, v103
	v_cvt_pk_bf16_f32 v100, v96, v97
	v_exp_f32_e32 v84, v84
	v_med3_num_f32 v80, v81, -s50, s50
	v_med3_num_f32 v81, v83, -s50, s50
	v_min_num_f32_e32 v103, v94, v134
	v_add_f32_e32 v102, 1.0, v86
	v_add_f32_e32 v96, 1.0, v82
	v_rcp_f32_e32 v82, v104
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v128, v[98:99], v[100:101] offset0:4 offset1:6
	v_add_f32_e32 v97, 1.0, v84
	v_med3_num_f32 v84, v85, -s50, s50
	v_med3_num_f32 v85, v87, -s50, s50
	v_rcp_f32_e32 v87, v102
	v_rcp_f32_e32 v83, v96
	v_rcp_f32_e32 v86, v97
	v_min_num_f32_e32 v96, v88, v134
	v_min_num_f32_e32 v102, v92, v134
	v_max_num_f32_e32 v40, v40, v40
	v_max_num_f32_e32 v20, v20, v20
	v_max_num_f32_e32 v24, v24, v24
	v_mul_f32_e32 v88, 0xbfb8aa3b, v96
	v_pk_mul_f32 v[82:83], v[106:107], v[82:83]
	v_pk_mul_f32 v[86:87], v[110:111], v[86:87]
	v_mul_f32_e32 v92, 0xbfb8aa3b, v102
	v_max_num_f32_e32 v8, v8, v8
	v_max_num_f32_e32 v12, v12, v12
	v_pk_mul_f32 v[80:81], v[80:81], v[82:83]
	v_pk_mul_f32 v[84:85], v[84:85], v[86:87]
	v_exp_f32_e32 v87, v88
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_pk_bf16_f32 v82, v80, v81
	v_cvt_pk_bf16_f32 v83, v84, v85
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v84, 1.0, v87 :: v_dual_min_num_f32 v97, v90, v134
	v_rcp_f32_e32 v80, v84
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v90, 0xbfb8aa3b, v97
	v_exp_f32_e32 v88, v90
	v_exp_f32_e32 v90, v92
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_add_f32 v85, 1.0, v88 :: v_dual_add_f32 v87, 1.0, v90
	v_min_num_f32_e32 v90, v64, v134
	v_max_num_f32_e32 v64, v66, v66
	v_med3_num_f32 v88, v93, -s50, s50
	v_rcp_f32_e32 v81, v85
	v_rcp_f32_e32 v84, v87
	v_nop
	v_med3_num_f32 v87, v91, -s50, s50
	v_min_num_f32_e32 v91, v64, v134
	v_max_num_f32_e32 v64, v70, v70
	v_mul_f32_e32 v86, 0xbfb8aa3b, v103
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[80:81], v[96:97], v[80:81]
	v_exp_f32_e32 v86, v86
	v_nop
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v86, 1.0, v86
	v_rcp_f32_e32 v85, v86
	v_nop
	v_med3_num_f32 v86, v89, -s50, s50
	v_med3_num_f32 v89, v95, -s50, s50
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_pk_mul_f32 v[80:81], v[86:87], v[80:81]
	v_pk_mul_f32 v[84:85], v[102:103], v[84:85]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_pk_mul_f32 v[84:85], v[88:89], v[84:85]
	v_min_num_f32_e32 v88, v68, v134
	v_min_num_f32_e32 v89, v64, v134
	v_cvt_pk_bf16_f32 v85, v84, v85
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v64, 0xbfb8aa3b, v88
	v_mul_f32_e32 v70, 0xbfb8aa3b, v89
	v_cvt_pk_bf16_f32 v84, v80, v81
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v80, v64
	v_exp_f32_e32 v70, v70
	v_mul_f32_e32 v68, 0xbfb8aa3b, v91
	v_med3_num_f32 v64, v65, -s50, s50
	v_med3_num_f32 v65, v67, -s50, s50
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_add_f32_e32 v80, 1.0, v80
	v_mul_f32_e32 v66, 0xbfb8aa3b, v90
	v_add_f32_e32 v81, 1.0, v70
	v_exp_f32_e32 v68, v68
	v_med3_num_f32 v70, v69, -s50, s50
	v_rcp_f32_e32 v80, v80
	v_exp_f32_e32 v66, v66
	v_rcp_f32_e32 v81, v81
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_add_f32 v68, 1.0, v68 :: v_dual_add_f32 v66, 1.0, v66
	v_rcp_f32_e32 v67, v68
	v_nop
	v_min_num_f32_e32 v68, v72, v134
	v_dual_max_num_f32 v72, v74, v74 :: v_dual_max_num_f32 v74, v76, v76
	v_rcp_f32_e32 v66, v66
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_max_num_f32 v76, v78, v78 :: v_dual_mul_f32 v69, 0xbfb8aa3b, v68
	v_min_num_f32_e32 v86, v74, v134
	v_pk_mul_f32 v[80:81], v[88:89], v[80:81]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v78, v69
	v_pk_mul_f32 v[66:67], v[90:91], v[66:67]
	v_min_num_f32_e32 v69, v72, v134
	v_mul_f32_e32 v74, 0xbfb8aa3b, v86
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_pk_mul_f32 v[64:65], v[64:65], v[66:67]
	v_pk_mul_f32 v[66:67], v[70:71], v[80:81]
	v_add_f32_e32 v70, 1.0, v78
	v_dual_min_num_f32 v87, v76, v134 :: v_dual_mul_f32 v72, 0xbfb8aa3b, v69
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v67, v66, v67
	v_cvt_pk_bf16_f32 v66, v64, v65
	v_mul_f32_e32 v76, 0xbfb8aa3b, v87
	v_med3_num_f32 v65, v75, -s50, s50
	v_min_num_f32_e32 v75, v50, v134
	v_exp_f32_e32 v71, v72
	v_exp_f32_e32 v72, v74
	v_exp_f32_e32 v74, v76
	v_add_nc_u32_e32 v92, 0x1000, v128
	v_med3_num_f32 v64, v73, -s50, s50
	v_rcp_f32_e32 v70, v70
	v_med3_num_f32 v76, v77, -s50, s50
	v_med3_num_f32 v77, v79, -s50, s50
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_dual_add_f32 v71, 1.0, v71 :: v_dual_add_f32 v72, 1.0, v72
	v_add_f32_e32 v73, 1.0, v74
	v_min_num_f32_e32 v74, v48, v134
	v_dual_max_num_f32 v48, v52, v52 :: v_dual_max_num_f32 v52, v54, v54
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_rcp_f32_e32 v72, v72
	v_rcp_f32_e32 v73, v73
	v_rcp_f32_e32 v71, v71
	s_delay_alu instid0(VALU_DEP_1)
	v_min_num_f32_e32 v78, v48, v134
	v_mul_f32_e32 v50, 0xbfb8aa3b, v74
	v_dual_mul_f32 v48, 0xbfb8aa3b, v75 :: v_dual_min_num_f32 v79, v52, v134
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v92, v[82:83], v[84:85] offset1:2
	v_mul_f32_e32 v52, 0xbfb8aa3b, v78
	v_pk_mul_f32 v[72:73], v[86:87], v[72:73]
	v_pk_mul_f32 v[68:69], v[68:69], v[70:71]
	v_exp_f32_e32 v50, v50
	v_mul_f32_e32 v54, 0xbfb8aa3b, v79
	v_exp_f32_e32 v52, v52
	v_pk_mul_f32 v[70:71], v[76:77], v[72:73]
	v_pk_mul_f32 v[64:65], v[64:65], v[68:69]
	v_min_num_f32_e32 v73, v62, v134
	v_exp_f32_e32 v54, v54
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_pk_bf16_f32 v69, v70, v71
	v_min_num_f32_e32 v71, v58, v134
	v_exp_f32_e32 v48, v48
	v_dual_add_f32 v50, 1.0, v50 :: v_dual_add_f32 v70, 1.0, v52
	v_med3_num_f32 v52, v53, -s50, s50
	v_med3_num_f32 v53, v55, -s50, s50
	v_mul_f32_e32 v58, 0xbfb8aa3b, v71
	s_delay_alu instid0(TRANS32_DEP_1)
	v_add_f32_e32 v68, 1.0, v48
	v_med3_num_f32 v48, v49, -s50, s50
	v_med3_num_f32 v49, v51, -s50, s50
	v_rcp_f32_e32 v50, v50
	v_exp_f32_e32 v58, v58
	v_rcp_f32_e32 v51, v68
	v_nop
	v_add_f32_e32 v68, 1.0, v54
	v_rcp_f32_e32 v54, v70
	v_nop
	v_min_num_f32_e32 v70, v56, v134
	v_mul_f32_e32 v62, 0xbfb8aa3b, v73
	v_rcp_f32_e32 v55, v68
	v_nop
	v_cvt_pk_bf16_f32 v68, v64, v65
	v_pk_mul_f32 v[50:51], v[74:75], v[50:51]
	v_mul_f32_e32 v56, 0xbfb8aa3b, v70
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_pk_mul_f32 v[48:49], v[48:49], v[50:51]
	v_pk_mul_f32 v[54:55], v[78:79], v[54:55]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v56, v56
	v_cvt_pk_bf16_f32 v50, v48, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_pk_mul_f32 v[52:53], v[52:53], v[54:55]
	v_cvt_pk_bf16_f32 v51, v52, v53
	v_add_f32_e32 v53, 1.0, v58
	v_min_num_f32_e32 v58, v32, v134
	v_dual_max_num_f32 v32, v34, v34 :: v_dual_max_num_f32 v34, v36, v36
	v_min_num_f32_e32 v72, v60, v134
	v_dual_max_num_f32 v36, v38, v38 :: v_dual_add_f32 v52, 1.0, v56
	v_med3_num_f32 v56, v61, -s50, s50
	v_exp_f32_e32 v55, v62
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mul_f32 v60, 0xbfb8aa3b, v72 :: v_dual_min_num_f32 v61, v36, v134
	v_rcp_f32_e32 v49, v53
	v_nop
	v_med3_num_f32 v53, v59, -s50, s50
	v_min_num_f32_e32 v59, v32, v134
	v_exp_f32_e32 v54, v60
	s_delay_alu instid0(TRANS32_DEP_3)
	v_dual_min_num_f32 v60, v34, v134 :: v_dual_add_f32 v55, 1.0, v55
	v_rcp_f32_e32 v48, v52
	v_nop
	v_med3_num_f32 v52, v57, -s50, s50
	v_med3_num_f32 v57, v63, -s50, s50
	v_mul_f32_e32 v36, 0xbfb8aa3b, v60
	v_rcp_f32_e32 v55, v55
	v_mul_f32_e32 v34, 0xbfb8aa3b, v59
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v36, v36
	v_pk_mul_f32 v[48:49], v[70:71], v[48:49]
	v_pk_mul_f32 v[48:49], v[52:53], v[48:49]
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_e32 v36, 1.0, v36
	v_mul_f32_e32 v38, 0xbfb8aa3b, v58
	v_cvt_pk_bf16_f32 v52, v48, v49
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v32, v38
	v_add_f32_e32 v54, 1.0, v54
	v_mul_f32_e32 v38, 0xbfb8aa3b, v61
	v_rcp_f32_e32 v54, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_exp_f32_e32 v38, v38
	v_pk_mul_f32 v[54:55], v[72:73], v[54:55]
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v38, 1.0, v38
	v_pk_mul_f32 v[54:55], v[56:57], v[54:55]
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v49, v38
	v_nop
	v_med3_num_f32 v38, v37, -s50, s50
	v_dual_max_num_f32 v37, v42, v42 :: v_dual_max_num_f32 v42, v44, v44
	v_cvt_pk_bf16_f32 v53, v54, v55
	v_exp_f32_e32 v54, v34
	v_nop
	v_add_f32_e32 v34, 1.0, v32
	v_med3_num_f32 v32, v33, -s50, s50
	v_med3_num_f32 v33, v35, -s50, s50
	v_min_num_f32_e32 v37, v37, v134
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(TRANS32_DEP_2)
	v_rcp_f32_e32 v34, v34
	v_add_f32_e32 v48, 1.0, v54
	v_min_num_f32_e32 v54, v42, v134
	v_max_num_f32_e32 v42, v46, v46
	v_mul_f32_e32 v44, 0xbfb8aa3b, v37
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v35, v48
	v_rcp_f32_e32 v48, v36
	v_dual_min_num_f32 v36, v40, v134 :: v_dual_min_num_f32 v55, v42, v134
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_3)
	v_exp_f32_e32 v42, v44
	v_add_nc_u32_e32 v56, 0x2000, v128
	v_pk_mul_f32 v[34:35], v[58:59], v[34:35]
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[48:49], v[60:61], v[48:49]
	v_pk_mul_f32 v[32:33], v[32:33], v[34:35]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_pk_mul_f32 v[38:39], v[38:39], v[48:49]
	v_mul_f32_e32 v34, 0xbfb8aa3b, v54
	v_dual_min_num_f32 v49, v22, v134 :: v_dual_min_num_f32 v48, v20, v134
	v_cvt_pk_bf16_f32 v35, v38, v39
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v39, v34
	v_max_num_f32_e32 v16, v16, v16
	v_cvt_pk_bf16_f32 v34, v32, v33
	v_med3_num_f32 v32, v41, -s50, s50
	v_add_f32_e32 v41, 1.0, v42
	v_med3_num_f32 v33, v43, -s50, s50
	v_mul_f32_e32 v20, 0xbfb8aa3b, v48
	v_mul_f32_e32 v22, 0xbfb8aa3b, v49
	v_dual_add_f32 v44, 1.0, v39 :: v_dual_min_num_f32 v42, v16, v134
	v_mul_f32_e32 v40, 0xbfb8aa3b, v36
	v_rcp_f32_e32 v39, v41
	v_max_num_f32_e32 v16, v18, v18
	s_delay_alu instid0(VALU_DEP_3)
	v_rcp_f32_e32 v44, v44
	v_mul_f32_e32 v41, 0xbfb8aa3b, v42
	v_exp_f32_e32 v40, v40
	v_exp_f32_e32 v20, v20
	v_min_num_f32_e32 v43, v16, v134
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v16, v41
	v_nop
	v_med3_num_f32 v41, v47, -s50, s50
	v_add_f32_e32 v38, 1.0, v40
	v_mul_f32_e32 v40, 0xbfb8aa3b, v55
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_rcp_f32_e32 v38, v38
	v_exp_f32_e32 v46, v40
	v_nop
	v_med3_num_f32 v40, v45, -s50, s50
	s_delay_alu instid0(TRANS32_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_pk_mul_f32 v[36:37], v[36:37], v[38:39]
	v_add_f32_e32 v18, 1.0, v46
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_pk_mul_f32 v[32:33], v[32:33], v[36:37]
	v_rcp_f32_e32 v45, v18
	v_nop
	v_mul_f32_e32 v18, 0xbfb8aa3b, v43
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[38:39], v[54:55], v[44:45]
	v_add_f32_e32 v44, 1.0, v16
	v_med3_num_f32 v16, v17, -s50, s50
	v_med3_num_f32 v17, v19, -s50, s50
	v_pk_mul_f32 v[36:37], v[40:41], v[38:39]
	v_min_num_f32_e32 v41, v30, v134
	v_exp_f32_e32 v18, v18
	v_dual_add_f32 v39, 1.0, v20 :: v_dual_add_f32 v40, 1.0, v22
	v_med3_num_f32 v20, v21, -s50, s50
	v_med3_num_f32 v21, v23, -s50, s50
	v_mul_f32_e32 v30, 0xbfb8aa3b, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(TRANS32_DEP_3)
	v_rcp_f32_e32 v22, v39
	v_rcp_f32_e32 v23, v40
	v_add_f32_e32 v38, 1.0, v18
	v_dual_min_num_f32 v39, v26, v134 :: v_dual_min_num_f32 v40, v28, v134
	v_rcp_f32_e32 v18, v44
	v_exp_f32_e32 v30, v30
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v19, v38
	v_nop
	v_min_num_f32_e32 v38, v24, v134
	v_mul_f32_e32 v26, 0xbfb8aa3b, v39
	v_mul_f32_e32 v28, 0xbfb8aa3b, v40
	v_pk_mul_f32 v[22:23], v[48:49], v[22:23]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v38
	v_exp_f32_e32 v26, v26
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v28, v28
	v_pk_mul_f32 v[18:19], v[42:43], v[18:19]
	v_pk_mul_f32 v[20:21], v[20:21], v[22:23]
	v_exp_f32_e32 v24, v24
	v_cvt_pk_bf16_f32 v22, v32, v33
	v_cvt_pk_bf16_f32 v23, v36, v37
	v_pk_mul_f32 v[16:17], v[16:17], v[18:19]
	v_cvt_pk_bf16_f32 v19, v20, v21
	v_add_f32_e32 v21, 1.0, v26
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dual_add_f32 v26, 1.0, v30 :: v_dual_add_f32 v18, 1.0, v24
	v_add_f32_e32 v24, 1.0, v28
	v_rcp_f32_e32 v21, v21
	s_delay_alu instid0(VALU_DEP_2)
	v_rcp_f32_e32 v33, v26
	v_nop
	v_med3_num_f32 v26, v25, -s50, s50
	v_rcp_f32_e32 v20, v18
	v_rcp_f32_e32 v32, v24
	v_nop
	v_min_num_f32_e32 v24, v0, v134
	v_max_num_f32_e32 v0, v2, v2
	v_med3_num_f32 v28, v29, -s50, s50
	v_med3_num_f32 v29, v31, -s50, s50
	v_cvt_pk_bf16_f32 v18, v16, v17
	v_mul_f32_e32 v2, 0xbfb8aa3b, v24
	v_pk_mul_f32 v[20:21], v[38:39], v[20:21]
	v_pk_mul_f32 v[30:31], v[40:41], v[32:33]
	v_min_num_f32_e32 v25, v0, v134
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_exp_f32_e32 v2, v2
	v_pk_mul_f32 v[16:17], v[26:27], v[20:21]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_pk_mul_f32 v[20:21], v[28:29], v[30:31]
	v_min_num_f32_e32 v28, v8, v134
	v_mul_f32_e32 v0, 0xbfb8aa3b, v25
	v_dual_min_num_f32 v30, v12, v134 :: v_dual_min_num_f32 v31, v14, v134
	v_cvt_pk_bf16_f32 v21, v20, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_f32_e32 v8, 0xbfb8aa3b, v28
	v_exp_f32_e32 v27, v0
	v_min_num_f32_e32 v29, v10, v134
	v_mul_f32_e32 v10, 0xbfb8aa3b, v30
	v_mul_f32_e32 v12, 0xbfb8aa3b, v31
	v_exp_f32_e32 v8, v8
	v_min_num_f32_e32 v0, v4, v134
	v_max_num_f32_e32 v4, v6, v6
	v_add_f32_e32 v6, 1.0, v2
	v_med3_num_f32 v2, v1, -s50, s50
	v_mul_f32_e32 v14, 0xbfb8aa3b, v29
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v20, 0xbfb8aa3b, v0 :: v_dual_min_num_f32 v1, v4, v134
	v_rcp_f32_e32 v26, v6
	v_exp_f32_e32 v10, v10
	v_exp_f32_e32 v12, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v6, v20
	v_nop
	v_mul_f32_e32 v20, 0xbfb8aa3b, v1
	v_exp_f32_e32 v14, v14
	v_add_f32_e32 v4, 1.0, v27
	v_exp_f32_e32 v20, v20
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_e32 v6, 1.0, v6
	v_rcp_f32_e32 v27, v4
	v_nop
	v_med3_num_f32 v4, v5, -s50, s50
	v_med3_num_f32 v5, v7, -s50, s50
	v_rcp_f32_e32 v32, v6
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_f32_e32 v20, 1.0, v20
	v_dual_add_f32 v6, 1.0, v8 :: v_dual_add_f32 v8, 1.0, v10
	v_dual_add_f32 v10, 1.0, v12 :: v_dual_add_f32 v12, 1.0, v14
	v_rcp_f32_e32 v33, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_rcp_f32_e32 v6, v6
	v_rcp_f32_e32 v36, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_rcp_f32_e32 v37, v10
	v_rcp_f32_e32 v7, v12
	v_pk_mul_f32 v[24:25], v[24:25], v[26:27]
	v_med3_num_f32 v8, v9, -s50, s50
	v_med3_num_f32 v9, v11, -s50, s50
	v_pk_mul_f32 v[0:1], v[0:1], v[32:33]
	v_med3_num_f32 v10, v13, -s50, s50
	v_med3_num_f32 v11, v15, -s50, s50
	v_pk_mul_f32 v[12:13], v[30:31], v[36:37]
	v_pk_mul_f32 v[6:7], v[28:29], v[6:7]
	v_pk_mul_f32 v[0:1], v[4:5], v[0:1]
	v_pk_mul_f32 v[2:3], v[2:3], v[24:25]
	v_cvt_pk_bf16_f32 v20, v16, v17
	v_pk_mul_f32 v[4:5], v[10:11], v[12:13]
	v_pk_mul_f32 v[6:7], v[8:9], v[6:7]
	v_add_nc_u32_e32 v8, 0x3000, v128
	v_cvt_pk_bf16_f32 v1, v0, v1
	v_cvt_pk_bf16_f32 v0, v2, v3
	v_cvt_pk_bf16_f32 v3, v4, v5
	v_cvt_pk_bf16_f32 v2, v6, v7
	s_wait_alu depctr_va_vdst(0)
	ds_store_2addr_b64 v92, v[66:67], v[68:69] offset0:4 offset1:6
	ds_store_2addr_b64 v56, v[50:51], v[52:53] offset1:2
	ds_store_2addr_b64 v56, v[34:35], v[22:23] offset0:4 offset1:6
	ds_store_2addr_b64 v8, v[18:19], v[20:21] offset1:2
	ds_store_2addr_b64 v8, v[0:1], v[2:3] offset0:4 offset1:6
	s_wait_alu depctr_vm_vsrc(0)
	v_cndmask_b32_e64 v0, 0, 1, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_readfirstlane_b32 s1, v0
	s_sub_co_i32 s12, s0, s1
	s_ashr_i32 s13, s12, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_u64 s[0:1], s[34:35], s[12:13]
	s_lshl_b64 s[0:1], s[0:1], 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[0:1], s[2:3], s[0:1]
	s_mul_u64 s[2:3], s[6:7], s[12:13]
	s_add_nc_u64 s[0:1], s[0:1], s[52:53]
	s_and_b32 s6, s13, 0xffff
	s_add_nc_u64 s[10:11], s[2:3], s[0:1]
	s_sub_co_i32 s0, s48, s5
	s_bitset1_b32 s11, 31
	s_max_i32 s0, s0, 0
	s_mov_b32 s5, s12
	s_lshl_b32 s1, s0, 16
	s_lshr_b32 s0, s0, 16
	s_or_b32 s2, s1, 0x7fff
	s_or_b32 s3, s0, 0x800000
	s_mov_b32 s1, 0xffff0000
	s_mov_b32 s0, 0x10000
	s_barrier_wait -1
	tensor_store_from_lds s[8:11], s[0:7]
	s_wait_tensorcnt 0x0
.LBB0_5:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4
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
		.amdhsa_next_free_sgpr 88
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 73
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
	.size	a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4, .Lfunc_end0-a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4

	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.num_vgpr, 239
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.num_agpr, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.numbered_sgpr, 88
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.num_named_barrier, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.private_seg_size, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.uses_vcc, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.uses_flat_scratch, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.has_dyn_sized_stack, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.has_recursion, 0
	.set a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.has_indirect_call, 0
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
    .name:           a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 1
      - 1
    .sgpr_count:     88
    .sgpr_spill_count: 0
    .symbol:         a8w4_tdm_fp4_t64x256x256_w1x4_b2_K7168_e96_act1_wpt4.kd
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
