	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6
	.p2align	8
	.type	moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6,@function
moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_mov_b32_e32 v1, s13
	s_cmp_eq_u64 s[14:15], 0
	s_cbranch_scc1 .LBB0_2
	s_or_b64 s[12:13], s[14:15], 0xfe00000000000000
	s_mov_b32 s15, 0
	s_mov_b32 s14, 0x1ffffff
	buffer_load_b32 v1, off, s[12:15], null
.LBB0_2:
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s10, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s1, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	v_lshrrev_b32_e32 v2, 5, v0
	s_add_co_i32 s10, s10, s0
	s_cmp_eq_u32 s1, 0
	s_cselect_b32 s0, ttmp9, s10
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_lshl_or_b32 v2, s0, 3, v2
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_ge_u32_e32 vcc_lo, v2, v1
	v_readfirstlane_b32 s0, v2
	s_and_b32 s10, vcc_lo, exec_lo
	s_cbranch_scc1 .LBB0_8
	s_bfe_u32 s10, ttmp6, 0x40010
	s_bfe_u32 s11, ttmp6, 0x40004
	s_add_co_i32 s10, s10, 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s10, ttmp7, s10
	s_add_co_i32 s11, s11, s10
	s_cmp_eq_u32 s1, 0
	s_cselect_b32 s1, ttmp7, s11
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b32 s1, s1, 3
	s_cmp_gt_u32 s1, 0xdf
	s_cbranch_scc1 .LBB0_8
	v_dual_lshrrev_b32 v1, 2, v0 :: v_dual_bitop2_b32 v14, 3, v0 bitop3:0x40
	s_lshl_b32 s10, s0, 2
	s_or_b64 s[8:9], s[8:9], 0xfe00000000000000
	v_mov_b32_e32 v2, s10
	s_delay_alu instid0(VALU_DEP_2)
	v_and_or_b32 v1, v1, 7, s1
	v_lshlrev_b32_e32 v3, 4, v14
	s_mov_b32 s11, 0
	s_mov_b32 s10, 0x1ffffff
	s_mul_hi_u32 s0, s0, 0xaaaaaaab
	buffer_load_b32 v6, v2, s[8:11], null offen
	s_wait_xcnt 0x0
	s_lshr_b32 s10, s0, 2
	v_lshl_or_b32 v2, v1, 6, v3
	s_mul_u64 s[0:1], s[10:11], 0x3800
	s_movk_i32 s10, 0x70
	s_add_nc_u64 s[8:9], s[0:1], s[2:3]
	s_mov_b32 s2, exec_lo
	buffer_load_b128 v[10:13], v2, s[8:11], null offen
	v_lshlrev_b32_e32 v9, 4, v1
	s_wait_loadcnt 0x0
	v_and_b32_e32 v2, 0xffff0000, v10
	v_dual_lshlrev_b32 v3, 16, v10 :: v_dual_lshlrev_b32 v5, 16, v11
	v_and_b32_e32 v4, 0xffff0000, v11
	v_and_b32_e32 v7, 0xffff0000, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_maximum3_f32 v2, |v3|, 0, |v2|
	v_and_b32_e32 v3, 31, v0
	v_maximum3_f32 v2, v2, |v5|, |v4|
	v_dual_lshlrev_b32 v5, 16, v13 :: v_dual_lshlrev_b32 v8, 16, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v3, 2, v3
	v_and_b32_e32 v4, 0xffff0000, v13
	v_maximum3_f32 v2, v2, |v8|, |v7|
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_xor_b32_e32 v7, 4, v3
	v_xor_b32_e32 v3, 8, v3
	v_maximum3_f32 v2, v2, |v5|, |v4|
	ds_bpermute_b32 v4, v7, v2
	v_mov_b32_e32 v7, 0
	s_wait_dscnt 0x0
	v_maximum_f32 v4, v2, v4
	ds_bpermute_b32 v5, v3, v4
	v_mul_u64_e32 v[2:3], 0xe00, v[6:7]
	s_wait_dscnt 0x0
	v_maximum_f32 v4, v4, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v4, 0x3e2aaaab, v4
	v_and_b32_e32 v5, 0x7fffff, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_cmp_ne_u32_e32 vcc_lo, 0, v5
	v_mov_b32_e32 v5, v7
	v_bfe_u32 v4, v4, 23, 8
	v_add_nc_u64_e32 v[2:3], s[4:5], v[2:3]
	v_lshl_or_b32 v7, v14, 2, v9
	v_add_co_ci_u32_e64 v4, null, 0, v4, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v14
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v8, 0xff, v4
	v_dual_mov_b32 v4, 28 :: v_dual_lshlrev_b32 v1, 23, v8
	;;#ASMSTART
	v_cvt_scalef32_pk8_fp4_bf16 v1, v[10:13], v1
	;;#ASMEND
.LBB0_5:
	v_readfirstlane_b32 s8, v2
	v_readfirstlane_b32 s9, v3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_readfirstlane_b32 s10, v4
	v_readfirstlane_b32 s11, v5
	v_cmp_eq_u64_e64 s0, s[8:9], v[2:3]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_eq_u64_e64 s1, s[10:11], v[4:5]
	s_and_b32 s0, s0, s1
	s_and_saveexec_b32 s0, s0
	buffer_store_b32 v1, v7, s[8:11], null offen
	s_wait_xcnt 0x0
	s_xor_b32 exec_lo, exec_lo, s0
	s_cbranch_execnz .LBB0_5
	s_mov_b32 exec_lo, s2
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_8
	v_and_b32_e32 v1, 0xfc0, v9
	v_lshrrev_b32_e32 v2, 6, v6
	v_bfe_u32 v0, v0, 2, 2
	s_or_b64 s[0:1], s[6:7], 0xfe00000000000000
	s_mov_b32 s3, 0
	v_and_or_b32 v1, v6, 63, v1
	s_mov_b32 s2, 0x1ffffff
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_u32 v1, 0xe00, v2, v1
	v_lshl_or_b32 v0, v1, 2, v0
	buffer_store_b8 v8, v0, s[0:3], null offen
.LBB0_8:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
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
		.amdhsa_next_free_vgpr 15
		.amdhsa_next_free_sgpr 16
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 6
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
	.size	moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6, .Lfunc_end0-moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6

	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.num_vgpr, 15
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.num_agpr, 0
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.numbered_sgpr, 16
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.num_named_barrier, 0
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.private_seg_size, 0
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.uses_vcc, 1
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.uses_flat_scratch, 0
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.has_dyn_sized_stack, 0
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.has_recursion, 0
	.set moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.has_indirect_call, 0
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
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .max_flat_workgroup_size: 256
    .name:           moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     18
    .sgpr_spill_count: 0
    .symbol:         moe_fused_quant_preshuffle_routeks_fd7168_r4_fp4_pk8_srctk6.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     15
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
