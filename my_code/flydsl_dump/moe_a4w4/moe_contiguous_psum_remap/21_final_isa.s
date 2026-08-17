	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	moe_contiguous_psum_remap
	.p2align	8
	.type	moe_contiguous_psum_remap,@function
moe_contiguous_psum_remap:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	v_cmp_ne_u32_e64 s18, 0, v0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_mov_b64 s[0:1], s[10:11]
	s_and_saveexec_b32 s10, vcc_lo
	v_mov_b32_e32 v1, 0
	ds_store_b32 v1, v1 offset:4096
	s_or_b32 exec_lo, exec_lo, s10
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_lshlrev_b32_e32 v1, 2, v0
	s_or_b64 s[24:25], s[6:7], 0xfe00000000000000
	s_mov_b64 s[26:27], 0x1ffffff
	s_cmp_lt_i32 s13, 1
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_31
	s_cvt_f32_u32 s6, s15
	v_dual_mov_b32 v3, 0 :: v_dual_lshlrev_b32 v4, 2, v0
	s_sub_co_i32 s11, 0, s15
	s_delay_alu instid0(SALU_CYCLE_1)
	v_rcp_iflag_f32_e32 v2, s6
	s_or_b64 s[28:29], s[2:3], 0xfe00000000000000
	v_cmp_lt_u32_e64 s2, 1, v0
	v_cmp_lt_u32_e64 s3, 3, v0
	v_cmp_lt_u32_e64 s6, 7, v0
	v_cmp_lt_u32_e64 s7, 15, v0
	v_cmp_lt_u32_e64 s19, 31, v0
	v_cmp_lt_u32_e64 s20, 63, v0
	v_readfirstlane_b32 s10, v2
	v_cmp_lt_u32_e64 s21, 0x7f, v0
	v_cmp_lt_u32_e64 s22, 0xff, v0
	v_dual_add_nc_u32 v5, -4, v4 :: v_dual_add_nc_u32 v6, -16, v4
	s_mul_f32 s10, s10, 0x4f7ffffe
	v_subrev_nc_u32_e32 v7, 64, v4
	v_add_nc_u32_e32 v8, 0xffffff00, v4
	v_dual_mov_b32 v10, v4 :: v_dual_add_nc_u32 v9, 0xfffffc00, v4
	s_cvt_u32_f32 s10, s10
	s_mov_b32 s35, 0
	s_add_co_i32 s23, s15, -1
	s_mov_b64 s[30:31], 0x1ffffff
	s_mul_i32 s11, s11, s10
	s_or_b32 s9, s9, 0xfe000000
	s_mul_hi_u32 s11, s10, s11
	s_mov_b32 s33, s35
	s_add_co_i32 s34, s10, s11
	s_branch .LBB0_5
.LBB0_4:
	s_or_b32 exec_lo, exec_lo, s10
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_add_nc_u32_e32 v10, 0x800, v10
	s_addk_co_i32 s33, 0x200
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lt_i32 s33, s13
	s_barrier_wait -1
	s_cbranch_scc0 .LBB0_31
.LBB0_5:
	v_dual_mov_b32 v11, 0 :: v_dual_add_nc_u32 v2, s33, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s10, s13, v2
	s_and_saveexec_b32 s11, s10
	s_cbranch_execz .LBB0_7
	buffer_load_b32 v11, v10, s[28:31], null offen
.LBB0_7:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v2, s23, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_u64_e32 v[12:13], s[34:35], v[2:3]
	v_mul_lo_u32 v12, v13, s15
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sub_nc_u32_e32 v12, v2, v12
	v_subrev_nc_u32_e32 v13, s15, v12
	v_cmp_le_u32_e64 s11, s15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v12, v12, v13, s11
	v_subrev_nc_u32_e32 v13, s15, v12
	v_cmp_le_u32_e64 s11, s15, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_cndmask_b32 v12, v12, v13, s11 :: v_dual_mov_b32 v13, 0
	v_sub_nc_u32_e32 v2, v2, v12
	ds_store_b32 v4, v2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v2, 0
	s_barrier_wait -1
	ds_load_b32 v12, v4
	s_and_saveexec_b32 s11, s18
	ds_load_b32 v13, v5
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v12, v13, v12
	ds_store_b32 v4, v12 offset:2048
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b32 v12, v4 offset:2048
	s_and_saveexec_b32 s11, s2
	ds_load_b32 v2, v4 offset:2040
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_dual_mov_b32 v13, 0 :: v_dual_add_nc_u32 v2, v2, v12
	ds_store_b32 v4, v2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v2, 0
	s_barrier_wait -1
	ds_load_b32 v12, v4
	s_and_saveexec_b32 s11, s3
	ds_load_b32 v13, v6
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v12, v13, v12
	ds_store_b32 v4, v12 offset:2048
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b32 v12, v4 offset:2048
	s_and_saveexec_b32 s11, s6
	ds_load_b32 v2, v4 offset:2016
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_dual_mov_b32 v13, 0 :: v_dual_add_nc_u32 v2, v2, v12
	ds_store_b32 v4, v2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v2, 0
	s_barrier_wait -1
	ds_load_b32 v12, v4
	s_and_saveexec_b32 s11, s7
	ds_load_b32 v13, v7
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v12, v13, v12
	ds_store_b32 v4, v12 offset:2048
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b32 v12, v4 offset:2048
	s_and_saveexec_b32 s11, s19
	ds_load_b32 v2, v4 offset:1920
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_dual_mov_b32 v13, 0 :: v_dual_add_nc_u32 v2, v2, v12
	ds_store_b32 v4, v2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v2, 0
	s_barrier_wait -1
	ds_load_b32 v12, v4
	s_and_saveexec_b32 s11, s20
	ds_load_b32 v13, v8
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v12, v13, v12
	ds_store_b32 v4, v12 offset:2048
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b32 v12, v4 offset:2048
	s_and_saveexec_b32 s11, s21
	ds_load_b32 v2, v4 offset:1536
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_dual_add_nc_u32 v2, v2, v12 :: v_dual_mov_b32 v12, 0
	ds_store_b32 v4, v2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b32 v2, v4
	s_and_saveexec_b32 s11, s22
	ds_load_b32 v12, v9
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v2, v12, v2
	ds_store_b32 v4, v2 offset:2048
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b32 v2, v3 offset:4096
	s_and_saveexec_b32 s36, s10
	s_cbranch_execz .LBB0_29
	v_mov_b32_e32 v12, 0
	s_and_saveexec_b32 s10, s18
	ds_load_b32 v12, v4 offset:2044
	s_or_b32 exec_lo, exec_lo, s10
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v12, v12, v2
	s_mov_b32 s10, s26
	s_mov_b32 s11, s27
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v11, v12, v11
	buffer_store_b32 v12, v10, s[24:27], null offen
	buffer_store_b32 v11, v10, s[8:11], null offen
.LBB0_29:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s36
	ds_load_b32 v11, v3 offset:4092
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	s_and_saveexec_b32 s10, vcc_lo
	s_cbranch_execz .LBB0_4
	v_add_nc_u32_e32 v2, v11, v2
	ds_store_b32 v3, v2 offset:4096
	s_branch .LBB0_4
.LBB0_31:
	s_and_saveexec_b32 s2, vcc_lo
	s_cbranch_execz .LBB0_33
	v_mov_b32_e32 v2, 0
	s_or_b64 s[8:9], s[0:1], 0xfe00000000000000
	s_mov_b32 s11, 0
	s_mov_b32 s10, 0x1ffffff
	ds_load_b32 v2, v2 offset:4096
	s_wait_dscnt 0x0
	v_max_i32_e32 v2, s15, v2
	buffer_store_b32 v2, off, s[8:11], null
.LBB0_33:
	s_wait_xcnt 0x0
	s_or_b32 exec_lo, exec_lo, s2
	s_wait_storecnt 0x0
	s_barrier_signal -1
	v_mov_b32_e32 v4, s12
	s_cmp_eq_u64 s[16:17], 0
	s_barrier_wait -1
	s_cbranch_scc1 .LBB0_35
	s_or_b64 s[0:1], s[16:17], 0xfe00000000000000
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0x1ffffff
	buffer_load_b32 v4, off, s[0:3], null
.LBB0_35:
	s_wait_xcnt 0x0
	s_mov_b32 s0, exec_lo
	s_wait_loadcnt 0x0
	v_cmpx_lt_i32_e64 v0, v4
	s_cbranch_execz .LBB0_38
	s_cvt_f32_u32 s0, s14
	v_mov_b32_e32 v3, 0
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v2, s0
	v_nop
	v_readfirstlane_b32 s0, v2
	s_mul_f32 s2, s0, 0x4f7ffffe
	s_or_b64 s[0:1], s[4:5], 0xfe00000000000000
	s_sub_co_i32 s5, 0, s14
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s4, s2
	s_mov_b64 s[2:3], 0x1ffffff
	s_mul_i32 s5, s5, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_hi_u32 s6, s4, s5
	s_mov_b32 s5, 0
	s_add_co_i32 s4, s4, s6
	s_mov_b32 s6, s5
.LBB0_37:
	buffer_load_b32 v2, v1, s[0:3], null offen
	v_add_nc_u32_e32 v0, 0x200, v0
	s_wait_loadcnt 0x0
	v_mul_u64_e32 v[6:7], s[4:5], v[2:3]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_lo_u32 v5, v7, s14
	v_dual_add_nc_u32 v6, 1, v7 :: v_dual_sub_nc_u32 v5, v2, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_subrev_nc_u32_e32 v8, s14, v5
	v_cmp_le_u32_e32 vcc_lo, s14, v5
	v_dual_cndmask_b32 v6, v7, v6 :: v_dual_cndmask_b32 v5, v5, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v7, 1, v6
	v_cmp_le_u32_e32 vcc_lo, s14, v5
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v5, v6, v7, vcc_lo
	v_cmp_ge_i32_e32 vcc_lo, v0, v4
	v_lshlrev_b32_e32 v6, 2, v5
	s_or_b32 s6, vcc_lo, s6
	buffer_load_b32 v6, v6, s[24:27], null offen
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v2, v6, v2
	v_mul_lo_u32 v5, s14, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_sub_nc_u32_e32 v2, v2, v5
	buffer_store_b32 v2, v1, s[0:3], null offen
	s_wait_xcnt 0x0
	v_add_nc_u32_e32 v1, 0x800, v1
	s_and_not1_b32 exec_lo, exec_lo, s6
	s_cbranch_execnz .LBB0_37
.LBB0_38:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel moe_contiguous_psum_remap
		.amdhsa_group_segment_fixed_size 4100
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 64
		.amdhsa_user_sgpr_count 18
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 16
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
		.amdhsa_next_free_vgpr 14
		.amdhsa_next_free_sgpr 37
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 13
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
	.size	moe_contiguous_psum_remap, .Lfunc_end0-moe_contiguous_psum_remap

	.set moe_contiguous_psum_remap.num_vgpr, 14
	.set moe_contiguous_psum_remap.num_agpr, 0
	.set moe_contiguous_psum_remap.numbered_sgpr, 37
	.set moe_contiguous_psum_remap.num_named_barrier, 0
	.set moe_contiguous_psum_remap.private_seg_size, 0
	.set moe_contiguous_psum_remap.uses_vcc, 1
	.set moe_contiguous_psum_remap.uses_flat_scratch, 0
	.set moe_contiguous_psum_remap.has_dyn_sized_stack, 0
	.set moe_contiguous_psum_remap.has_recursion, 0
	.set moe_contiguous_psum_remap.has_indirect_call, 0
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
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         56
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 4100
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .max_flat_workgroup_size: 512
    .name:           moe_contiguous_psum_remap
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 512
      - 1
      - 1
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         moe_contiguous_psum_remap.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     14
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
