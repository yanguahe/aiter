	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all
	.p2align	8
	.type	moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all,@function
moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s8, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	v_lshrrev_b32_e32 v1, 5, v0
	s_add_co_i32 s1, s1, s0
	s_cmp_eq_u32 s8, 0
	s_cselect_b32 s0, ttmp9, s1
	s_mov_b32 s1, 0
	v_lshl_or_b32 v1, s0, 3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s10, v1
	v_readfirstlane_b32 s0, v1
	s_and_b32 s8, vcc_lo, exec_lo
	s_cbranch_scc1 .LBB0_25
	v_and_b32_e32 v5, 3, v0
	v_bfe_u32 v1, v0, 2, 3
	s_mul_u64 s[8:9], s[0:1], 0x1800
	s_mov_b32 s14, 48
	s_add_nc_u64 s[8:9], s[8:9], s[2:3]
	v_lshlrev_b32_e32 v2, 4, v5
	s_mov_b32 s15, s1
	s_mov_b32 s12, s8
	s_mov_b32 s13, s9
	s_cvt_f32_u32 s2, s11
	v_lshl_or_b32 v3, v1, 6, v2
	s_sub_co_i32 s3, 0, s11
	s_mov_b32 s18, 12
	s_mov_b32 s19, s1
	buffer_load_b128 v[8:11], v3, s[12:15], null offen
	s_wait_xcnt 0x0
	s_mov_b64 s[14:15], 0x1ffffff
	s_wait_loadcnt 0x0
	v_and_b32_e32 v3, 0xffff0000, v8
	v_lshlrev_b32_e32 v4, 16, v8
	v_and_b32_e32 v6, 0xffff0000, v9
	v_lshlrev_b32_e32 v7, 16, v9
	v_and_b32_e32 v12, 0xffff0000, v10
	v_lshlrev_b32_e32 v13, 16, v10
	v_maximum3_f32 v3, |v4|, 0, |v3|
	v_and_b32_e32 v4, 31, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_maximum3_f32 v3, v3, |v7|, |v6|
	v_dual_lshlrev_b32 v4, 2, v4 :: v_dual_lshlrev_b32 v7, 16, v11
	v_and_b32_e32 v6, 0xffff0000, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_maximum3_f32 v12, v3, |v13|, |v12|
	v_xor_b32_e32 v3, 4, v4
	v_xor_b32_e32 v4, 8, v4
	s_delay_alu instid0(VALU_DEP_3)
	v_maximum3_f32 v6, v12, |v7|, |v6|
	v_rcp_iflag_f32_e32 v7, s2
	ds_bpermute_b32 v12, v3, v6
	v_nop
	v_readfirstlane_b32 s2, v7
	s_mul_f32 s2, s2, 0x4f7ffffe
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_cvt_u32_f32 s2, s2
	s_mul_i32 s3, s3, s2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s3, s2, s3
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	s_add_co_i32 s2, s2, s3
	s_mul_hi_u32 s2, s0, s2
	ds_bpermute_b32 v7, v4, v6
	s_mul_i32 s3, s2, s11
	s_add_co_i32 s10, s2, 1
	s_sub_co_i32 s3, s0, s3
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s12, s3, s11
	s_cmp_ge_u32 s3, s11
	s_cselect_b32 s2, s10, s2
	s_cselect_b32 s3, s12, s3
	s_add_co_i32 s10, s2, 1
	s_cmp_ge_u32 s3, s11
	s_cselect_b32 s2, s10, s2
	s_or_b64 s[12:13], s[6:7], 0xfe00000000000000
	s_mul_i32 s2, s2, s11
	s_mul_u64 s[6:7], s[0:1], 0x600
	s_sub_co_i32 s3, s0, s2
	s_mul_i32 s2, s2, 24
	s_and_b32 s11, s3, 48
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v6, v7
	v_lshrrev_b32_e32 v6, 2, v0
	s_lshr_b32 s10, s3, 6
	s_and_b32 s3, s3, 15
	s_add_co_i32 s2, s11, s2
	v_mul_f32_e32 v7, 0x3e2aaaab, v7
	v_bfe_u32 v0, v0, 2, 2
	s_mulk_i32 s10, 0x600
	s_add_co_i32 s2, s2, s3
	s_add_nc_u64 s[4:5], s[6:7], s[4:5]
	v_and_b32_e32 v12, 0x7fffff, v7
	v_bfe_u32 v7, v7, 23, 8
	s_add_co_i32 s2, s2, s10
	s_mov_b32 s16, s4
	s_mov_b32 s17, s5
	v_cmp_ne_u32_e32 vcc_lo, 0, v12
	s_mov_b64 s[6:7], 12
	s_mov_b64 s[10:11], 48
	v_add_co_ci_u32_e64 v7, null, 0, v7, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v5
	v_lshlrev_b32_e32 v5, 2, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_min_u32_e32 v7, 0xff, v7
	v_lshl_or_b32 v13, v1, 4, v5
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v12, 23, v7
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v8, v[8:11], v12
	;;#ASMEND
	buffer_store_b32 v8, v13, s[16:19], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_3
	v_lshlrev_b32_e32 v8, 4, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_and_b32_e32 v8, 64, v8
	v_add_nc_u32_e32 v8, s2, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v8, v8, 2, v0
	buffer_store_b8 v7, v8, s[12:15], null offen
.LBB0_3:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 8, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v7, v8, 6, v2
	buffer_load_b128 v[10:13], v7, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v7, 0xffff0000, v10
	v_lshlrev_b32_e32 v9, 16, v10
	v_and_b32_e32 v14, 0xffff0000, v11
	v_dual_lshlrev_b32 v15, 16, v11 :: v_dual_lshlrev_b32 v16, 16, v12
	v_lshlrev_b32_e32 v8, 4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v7, |v9|, 0, |v7|
	v_and_b32_e32 v9, 0xffff0000, v12
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_and_b32_e32 v14, 0xffff0000, v13
	v_lshlrev_b32_e32 v15, 16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v7, v7, |v16|, |v9|
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_or_b32_e32 v14, v8, v5
	ds_bpermute_b32 v9, v3, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	ds_bpermute_b32 v9, v4, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, 0x3e2aaaab, v7
	v_and_b32_e32 v9, 0x7fffff, v7
	v_bfe_u32 v7, v7, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v9
	v_add_co_ci_u32_e64 v7, null, 0, v7, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v7, 0xff, v7
	v_lshlrev_b32_e32 v9, 23, v7
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v9, v[10:13], v9
	;;#ASMEND
	buffer_store_b32 v9, v14, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_5
	v_and_b32_e32 v8, 0xc0, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v8, s2, v8
	v_lshl_or_b32 v8, v8, 2, v0
	buffer_store_b8 v7, v8, s[12:15], null offen
.LBB0_5:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v7, v8, 6, v2
	buffer_load_b128 v[10:13], v7, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v7, 0xffff0000, v10
	v_lshlrev_b32_e32 v9, 16, v10
	v_and_b32_e32 v14, 0xffff0000, v11
	v_dual_lshlrev_b32 v15, 16, v11 :: v_dual_lshlrev_b32 v16, 16, v12
	v_lshlrev_b32_e32 v8, 4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v7, |v9|, 0, |v7|
	v_and_b32_e32 v9, 0xffff0000, v12
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_and_b32_e32 v14, 0xffff0000, v13
	v_lshlrev_b32_e32 v15, 16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v7, v7, |v16|, |v9|
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_or_b32_e32 v14, v8, v5
	ds_bpermute_b32 v9, v3, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	ds_bpermute_b32 v9, v4, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, 0x3e2aaaab, v7
	v_and_b32_e32 v9, 0x7fffff, v7
	v_bfe_u32 v7, v7, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v9
	v_add_co_ci_u32_e64 v7, null, 0, v7, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v7, 0xff, v7
	v_lshlrev_b32_e32 v9, 23, v7
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v9, v[10:13], v9
	;;#ASMEND
	buffer_store_b32 v9, v14, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_7
	v_and_b32_e32 v8, 0x140, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v8, s2, v8
	v_lshl_or_b32 v8, v8, 2, v0
	buffer_store_b8 v7, v8, s[12:15], null offen
.LBB0_7:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 24, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v7, v8, 6, v2
	buffer_load_b128 v[10:13], v7, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v7, 0xffff0000, v10
	v_lshlrev_b32_e32 v9, 16, v10
	v_and_b32_e32 v14, 0xffff0000, v11
	v_dual_lshlrev_b32 v15, 16, v11 :: v_dual_lshlrev_b32 v16, 16, v12
	v_lshlrev_b32_e32 v8, 4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v7, |v9|, 0, |v7|
	v_and_b32_e32 v9, 0xffff0000, v12
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_and_b32_e32 v14, 0xffff0000, v13
	v_lshlrev_b32_e32 v15, 16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v7, v7, |v16|, |v9|
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_or_b32_e32 v14, v8, v5
	ds_bpermute_b32 v9, v3, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	ds_bpermute_b32 v9, v4, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, 0x3e2aaaab, v7
	v_and_b32_e32 v9, 0x7fffff, v7
	v_bfe_u32 v7, v7, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v9
	v_add_co_ci_u32_e64 v7, null, 0, v7, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v7, 0xff, v7
	v_lshlrev_b32_e32 v9, 23, v7
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v9, v[10:13], v9
	;;#ASMEND
	buffer_store_b32 v9, v14, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_9
	v_and_b32_e32 v8, 0x1c0, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v8, s2, v8
	v_lshl_or_b32 v8, v8, 2, v0
	buffer_store_b8 v7, v8, s[12:15], null offen
.LBB0_9:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 32, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v7, v8, 6, v2
	buffer_load_b128 v[10:13], v7, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v7, 0xffff0000, v10
	v_lshlrev_b32_e32 v9, 16, v10
	v_and_b32_e32 v14, 0xffff0000, v11
	v_dual_lshlrev_b32 v15, 16, v11 :: v_dual_lshlrev_b32 v16, 16, v12
	v_lshlrev_b32_e32 v8, 4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v7, |v9|, 0, |v7|
	v_and_b32_e32 v9, 0xffff0000, v12
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_and_b32_e32 v14, 0xffff0000, v13
	v_lshlrev_b32_e32 v15, 16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v7, v7, |v16|, |v9|
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_or_b32_e32 v14, v8, v5
	ds_bpermute_b32 v9, v3, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	ds_bpermute_b32 v9, v4, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, 0x3e2aaaab, v7
	v_and_b32_e32 v9, 0x7fffff, v7
	v_bfe_u32 v7, v7, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v9
	v_add_co_ci_u32_e64 v7, null, 0, v7, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v7, 0xff, v7
	v_lshlrev_b32_e32 v9, 23, v7
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v9, v[10:13], v9
	;;#ASMEND
	buffer_store_b32 v9, v14, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_11
	v_and_b32_e32 v8, 0x240, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v8, s2, v8
	v_lshl_or_b32 v8, v8, 2, v0
	buffer_store_b8 v7, v8, s[12:15], null offen
.LBB0_11:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 40, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v7, v8, 6, v2
	buffer_load_b128 v[10:13], v7, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v7, 0xffff0000, v10
	v_lshlrev_b32_e32 v9, 16, v10
	v_and_b32_e32 v14, 0xffff0000, v11
	v_dual_lshlrev_b32 v15, 16, v11 :: v_dual_lshlrev_b32 v16, 16, v12
	v_lshlrev_b32_e32 v8, 4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v7, |v9|, 0, |v7|
	v_and_b32_e32 v9, 0xffff0000, v12
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_and_b32_e32 v14, 0xffff0000, v13
	v_lshlrev_b32_e32 v15, 16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v7, v7, |v16|, |v9|
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_or_b32_e32 v14, v8, v5
	ds_bpermute_b32 v9, v3, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	ds_bpermute_b32 v9, v4, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, 0x3e2aaaab, v7
	v_and_b32_e32 v9, 0x7fffff, v7
	v_bfe_u32 v7, v7, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v9
	v_add_co_ci_u32_e64 v7, null, 0, v7, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v7, 0xff, v7
	v_lshlrev_b32_e32 v9, 23, v7
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v9, v[10:13], v9
	;;#ASMEND
	buffer_store_b32 v9, v14, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_13
	v_and_b32_e32 v8, 0x2c0, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v8, s2, v8
	v_lshl_or_b32 v8, v8, 2, v0
	buffer_store_b8 v7, v8, s[12:15], null offen
.LBB0_13:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v8, 48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v7, v8, 6, v2
	buffer_load_b128 v[10:13], v7, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v7, 0xffff0000, v10
	v_lshlrev_b32_e32 v9, 16, v10
	v_and_b32_e32 v14, 0xffff0000, v11
	v_dual_lshlrev_b32 v15, 16, v11 :: v_dual_lshlrev_b32 v16, 16, v12
	v_lshlrev_b32_e32 v8, 4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v7, |v9|, 0, |v7|
	v_and_b32_e32 v9, 0xffff0000, v12
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_and_b32_e32 v14, 0xffff0000, v13
	v_lshlrev_b32_e32 v15, 16, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v7, v7, |v16|, |v9|
	v_maximum3_f32 v7, v7, |v15|, |v14|
	v_or_b32_e32 v14, v8, v5
	ds_bpermute_b32 v9, v3, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	ds_bpermute_b32 v9, v4, v7
	s_wait_dscnt 0x0
	v_maximum_f32 v7, v7, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v7, 0x3e2aaaab, v7
	v_and_b32_e32 v9, 0x7fffff, v7
	v_bfe_u32 v7, v7, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v9
	v_add_co_ci_u32_e64 v7, null, 0, v7, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v7, 0xff, v7
	v_lshlrev_b32_e32 v9, 23, v7
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v9, v[10:13], v9
	;;#ASMEND
	buffer_store_b32 v9, v14, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_15
	v_and_b32_e32 v8, 0x340, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v8, s2, v8
	v_lshl_or_b32 v8, v8, 2, v0
	buffer_store_b8 v7, v8, s[12:15], null offen
.LBB0_15:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 56, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v6, v7, 6, v2
	buffer_load_b128 v[8:11], v6, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v6, 0xffff0000, v8
	v_lshlrev_b32_e32 v12, 16, v8
	v_and_b32_e32 v13, 0xffff0000, v9
	v_dual_lshlrev_b32 v14, 16, v9 :: v_dual_lshlrev_b32 v15, 16, v10
	v_lshlrev_b32_e32 v7, 4, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v6, |v12|, 0, |v6|
	v_and_b32_e32 v12, 0xffff0000, v10
	v_maximum3_f32 v6, v6, |v14|, |v13|
	v_and_b32_e32 v13, 0xffff0000, v11
	v_lshlrev_b32_e32 v14, 16, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v6, v6, |v15|, |v12|
	v_maximum3_f32 v6, v6, |v14|, |v13|
	v_or_b32_e32 v13, v7, v5
	ds_bpermute_b32 v12, v3, v6
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	ds_bpermute_b32 v12, v4, v6
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, 0x3e2aaaab, v6
	v_and_b32_e32 v12, 0x7fffff, v6
	v_bfe_u32 v6, v6, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v12
	v_add_co_ci_u32_e64 v6, null, 0, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v6, 0xff, v6
	v_lshlrev_b32_e32 v12, 23, v6
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v8, v[8:11], v12
	;;#ASMEND
	buffer_store_b32 v8, v13, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_17
	v_and_b32_e32 v7, 0x3c0, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s2, v7
	v_lshl_or_b32 v7, v7, 2, v0
	buffer_store_b8 v6, v7, s[12:15], null offen
.LBB0_17:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 64, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v6, v7, 6, v2
	buffer_load_b128 v[8:11], v6, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v6, 0xffff0000, v8
	v_lshlrev_b32_e32 v12, 16, v8
	v_and_b32_e32 v13, 0xffff0000, v9
	v_dual_lshlrev_b32 v14, 16, v9 :: v_dual_lshlrev_b32 v15, 16, v10
	v_lshlrev_b32_e32 v7, 4, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v6, |v12|, 0, |v6|
	v_and_b32_e32 v12, 0xffff0000, v10
	v_maximum3_f32 v6, v6, |v14|, |v13|
	v_and_b32_e32 v13, 0xffff0000, v11
	v_lshlrev_b32_e32 v14, 16, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v6, v6, |v15|, |v12|
	v_maximum3_f32 v6, v6, |v14|, |v13|
	v_or_b32_e32 v13, v7, v5
	ds_bpermute_b32 v12, v3, v6
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	ds_bpermute_b32 v12, v4, v6
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, 0x3e2aaaab, v6
	v_and_b32_e32 v12, 0x7fffff, v6
	v_bfe_u32 v6, v6, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v12
	v_add_co_ci_u32_e64 v6, null, 0, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v6, 0xff, v6
	v_lshlrev_b32_e32 v12, 23, v6
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v8, v[8:11], v12
	;;#ASMEND
	buffer_store_b32 v8, v13, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_19
	v_and_b32_e32 v7, 0x440, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s2, v7
	v_lshl_or_b32 v7, v7, 2, v0
	buffer_store_b8 v6, v7, s[12:15], null offen
.LBB0_19:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 0x48, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v6, v7, 6, v2
	v_lshlrev_b32_e32 v7, 4, v7
	buffer_load_b128 v[8:11], v6, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v6, 0xffff0000, v8
	v_dual_lshlrev_b32 v12, 16, v8 :: v_dual_lshlrev_b32 v14, 16, v9
	v_and_b32_e32 v13, 0xffff0000, v9
	v_lshlrev_b32_e32 v15, 16, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v6, |v12|, 0, |v6|
	v_and_b32_e32 v12, 0xffff0000, v10
	v_maximum3_f32 v6, v6, |v14|, |v13|
	v_and_b32_e32 v13, 0xffff0000, v11
	v_lshlrev_b32_e32 v14, 16, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v6, v6, |v15|, |v12|
	v_maximum3_f32 v6, v6, |v14|, |v13|
	v_or_b32_e32 v13, v7, v5
	ds_bpermute_b32 v12, v3, v6
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	ds_bpermute_b32 v12, v4, v6
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, 0x3e2aaaab, v6
	v_and_b32_e32 v12, 0x7fffff, v6
	v_bfe_u32 v6, v6, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v12
	v_add_co_ci_u32_e64 v6, null, 0, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v6, 0xff, v6
	v_lshlrev_b32_e32 v12, 23, v6
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v8, v[8:11], v12
	;;#ASMEND
	buffer_store_b32 v8, v13, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_21
	v_and_b32_e32 v7, 0x4c0, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s2, v7
	v_lshl_or_b32 v7, v7, 2, v0
	buffer_store_b8 v6, v7, s[12:15], null offen
.LBB0_21:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v7, 0x50, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v6, v7, 6, v2
	v_lshlrev_b32_e32 v7, 4, v7
	buffer_load_b128 v[8:11], v6, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v6, 0xffff0000, v8
	v_dual_lshlrev_b32 v12, 16, v8 :: v_dual_lshlrev_b32 v14, 16, v9
	v_and_b32_e32 v13, 0xffff0000, v9
	v_lshlrev_b32_e32 v15, 16, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v6, |v12|, 0, |v6|
	v_and_b32_e32 v12, 0xffff0000, v10
	v_maximum3_f32 v6, v6, |v14|, |v13|
	v_and_b32_e32 v13, 0xffff0000, v11
	v_lshlrev_b32_e32 v14, 16, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v6, v6, |v15|, |v12|
	v_maximum3_f32 v6, v6, |v14|, |v13|
	v_or_b32_e32 v13, v7, v5
	ds_bpermute_b32 v12, v3, v6
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	ds_bpermute_b32 v12, v4, v6
	s_wait_dscnt 0x0
	v_maximum_f32 v6, v6, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v6, 0x3e2aaaab, v6
	v_and_b32_e32 v12, 0x7fffff, v6
	v_bfe_u32 v6, v6, 23, 8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cmp_ne_u32_e64 s0, 0, v12
	v_add_co_ci_u32_e64 v6, null, 0, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v6, 0xff, v6
	v_lshlrev_b32_e32 v12, 23, v6
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v8, v[8:11], v12
	;;#ASMEND
	buffer_store_b32 v8, v13, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_23
	v_and_b32_e32 v7, 0x540, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v7, s2, v7
	v_lshl_or_b32 v7, v7, 2, v0
	buffer_store_b8 v6, v7, s[12:15], null offen
.LBB0_23:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s0
	v_or_b32_e32 v10, 0x58, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v1, v10, 6, v2
	buffer_load_b128 v[6:9], v1, s[8:11], null offen
	s_wait_loadcnt 0x0
	s_wait_xcnt 0x0
	v_and_b32_e32 v1, 0xffff0000, v6
	v_dual_lshlrev_b32 v2, 16, v6 :: v_dual_lshlrev_b32 v12, 16, v7
	v_and_b32_e32 v11, 0xffff0000, v7
	v_lshlrev_b32_e32 v13, 16, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v1, |v2|, 0, |v1|
	v_and_b32_e32 v2, 0xffff0000, v8
	v_maximum3_f32 v1, v1, |v12|, |v11|
	v_and_b32_e32 v11, 0xffff0000, v9
	v_lshlrev_b32_e32 v12, 16, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_maximum3_f32 v1, v1, |v13|, |v2|
	v_maximum3_f32 v1, v1, |v12|, |v11|
	ds_bpermute_b32 v2, v3, v1
	s_wait_dscnt 0x0
	v_maximum_f32 v1, v1, v2
	ds_bpermute_b32 v2, v4, v1
	s_wait_dscnt 0x0
	v_maximum_f32 v1, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, 0x3e2aaaab, v1
	v_and_b32_e32 v2, 0x7fffff, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_cmp_ne_u32_e64 s0, 0, v2
	v_lshlrev_b32_e32 v2, 4, v10
	v_bfe_u32 v1, v1, 23, 8
	v_or_b32_e32 v4, v2, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, 0, v1, s0
	v_min_u32_e32 v1, 0xff, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v3, 23, v1
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v3, v[6:9], v3
	;;#ASMEND
	buffer_store_b32 v3, v4, s[4:7], null offen
	s_wait_xcnt 0x0
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_25
	v_and_b32_e32 v2, 0x5c0, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v2, s2, v2
	v_lshl_or_b32 v0, v2, 2, v0
	buffer_store_b8 v1, v0, s[12:15], null offen
.LBB0_25:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 40
		.amdhsa_user_sgpr_count 12
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 10
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
		.amdhsa_next_free_vgpr 17
		.amdhsa_next_free_sgpr 20
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 32
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
	.size	moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all, .Lfunc_end0-moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all

	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.num_vgpr, 17
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.num_agpr, 0
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.numbered_sgpr, 20
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.num_named_barrier, 0
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.private_seg_size, 0
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.uses_vcc, 1
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.uses_flat_scratch, 0
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.has_dyn_sized_stack, 0
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.has_recursion, 0
	.set moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.has_indirect_call, 0
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 40
    .max_flat_workgroup_size: 256
    .name:           moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         moe_fused_quant_preshuffle_fd3072_r4_fp4_pk8_all.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     17
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
