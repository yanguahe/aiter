	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text,"axG",@progbits,_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph,comdat,unique,1
	.protected	_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph ; -- Begin function _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph
	.globl	_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph
	.p2align	8
	.type	_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph,@function
_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph: ; @_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph
; %bb.13:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.14:
.LBB0_0:
	s_mov_b64 s[24:25], s[2:3]
	s_load_dword s2, s[12:13], 0x0
	v_readfirstlane_b32 s19, v0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 28
	s_add_i32 s2, s2, s3
	s_ashr_i32 s2, s2, 4
	s_mul_i32 s2, s2, 28
	s_cmp_ge_i32 s16, s2
	s_cbranch_scc1 .LBB0_12
; %bb.1:
	s_mul_hi_i32 s2, s16, 0x92492493
	s_add_i32 s2, s2, s16
	s_lshr_b32 s3, s2, 31
	s_ashr_i32 s2, s2, 4
	s_mov_b64 s[20:21], s[6:7]
	s_add_i32 s6, s2, s3
	s_ashr_i32 s7, s6, 31
	s_lshr_b32 s17, s19, 6
	s_lshl_b64 s[2:3], s[6:7], 2
	s_add_u32 s2, s10, s2
	s_addc_u32 s3, s11, s3
	s_load_dword s7, s[2:3], 0x0
	s_lshl_b32 s18, s6, 4
	s_cmpk_gt_u32 s19, 0x7f
	s_cselect_b64 s[10:11], -1, 0
	v_and_b32_e32 v1, 63, v0
	s_and_b64 vcc, exec, s[10:11]
	s_cbranch_vccnz .LBB0_3
; %bb.2:
	v_lshrrev_b32_e32 v2, 3, v1
	v_lshl_or_b32 v2, s17, 3, v2
	v_or_b32_e32 v3, s18, v2
	s_branch .LBB0_4
.LBB0_3:
                                        ; implicit-def: $vgpr3
.LBB0_4:
	s_mul_i32 s12, s6, 28
	s_sub_i32 s16, s16, s12
	s_lshl_b32 s12, s16, 16
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s13, s7, 0x1c0000
	s_load_dwordx2 s[2:3], s[0:1], 0x48
	s_add_i32 s31, s13, s12
	s_lshl_b32 s12, s17, 14
	s_add_i32 s31, s31, s12
	s_lshl_b32 s12, s16, 12
	s_mul_i32 s33, s7, 0x1c000
	s_lshl_b32 s7, s17, 10
	s_add_i32 s12, s7, s12
	s_add_i32 s33, s33, s12
	s_or_b32 s29, s31, 0x1000
	s_or_b32 s30, s31, 0x2000
	s_or_b32 s28, s31, 0x3000
	s_or_b32 s34, s33, 0x200
	s_andn2_b64 vcc, exec, s[10:11]
	v_and_b32_e32 v4, 48, v0
	s_cbranch_vccnz .LBB0_6
; %bb.5:
	v_and_b32_e32 v2, 48, v0
	s_load_dwordx2 s[12:13], s[0:1], 0x38
	s_cbranch_execz .LBB0_7
	s_branch .LBB0_8
.LBB0_6:
                                        ; implicit-def: $vgpr2
	s_load_dwordx2 s[12:13], s[0:1], 0x38
.LBB0_7:
	s_and_b32 s25, s25, 0xffff
	s_cmp_eq_u32 s17, 1
	s_cselect_b64 vcc, -1, 0
	v_lshlrev_b32_e32 v5, 4, v0
	v_cndmask_b32_e32 v3, v3, v3, vcc
	v_and_or_b32 v2, s19, 64, v4
	v_and_b32_e32 v5, 0x70, v5
	v_lshlrev_b32_e32 v3, 8, v3
	s_mov_b32 m0, s7
	s_mov_b32 s27, 0x20000
	s_mov_b32 s26, 0xa000000
	v_bitop3_b32 v2, v3, v2, v5 bitop3:0xf6
	buffer_load_dwordx4 v2, s[24:27], 0 offen lds
	s_add_i32 m0, s7, 0x800
	s_movk_i32 s7, 0x80
	buffer_load_dwordx4 v2, s[24:27], s7 offen lds
	v_mov_b32_e32 v2, v4
.LBB0_8:
	s_load_dword s19, s[0:1], 0x40
	s_mov_b32 s23, 0x20000
	s_lshl_b32 s24, s6, 9
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s22, 0x2a1c0000
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s6, 0x1400000
	s_mov_b32 s7, s23
	s_and_b32 s9, s9, 0xffff
	s_mov_b32 s10, 0x2a1c000
	s_mov_b32 s11, s23
	; sched_barrier mask(0x00000000)
	v_and_b32_e32 v3, 15, v0
	v_lshrrev_b32_e32 v1, 4, v1
	v_lshlrev_b32_e32 v76, 2, v3
	v_lshl_or_b32 v48, v1, 6, v76
	buffer_load_dword v77, v48, s[4:7], s24 offen
	buffer_load_dword v78, v48, s[8:11], s33 offen
	v_lshlrev_b32_e32 v4, 4, v3
	v_lshl_or_b32 v64, v1, 8, v4
	buffer_load_dwordx4 v[4:7], v64, s[20:23], s31 offen nt
	buffer_load_dwordx4 v[8:11], v64, s[20:23], s29 offen nt
	buffer_load_dword v79, v48, s[8:11], s34 offen
	buffer_load_dwordx4 v[12:15], v64, s[20:23], s30 offen nt
	buffer_load_dwordx4 v[16:19], v64, s[20:23], s28 offen nt
	buffer_load_dwordx4 v[20:23], v64, s[20:23], s31 offen offset:1024 nt
	buffer_load_dwordx4 v[24:27], v64, s[20:23], s29 offen offset:1024 nt
	buffer_load_dwordx4 v[28:31], v64, s[20:23], s30 offen offset:1024 nt
	buffer_load_dwordx4 v[32:35], v64, s[20:23], s28 offen offset:1024 nt
	buffer_load_dword v80, v48, s[4:7], s24 offen offset:256
	buffer_load_dword v81, v48, s[8:11], s34 offen offset:256
	buffer_load_dword v82, v48, s[8:11], s33 offen offset:256
	buffer_load_dwordx4 v[36:39], v64, s[20:23], s31 offen offset:2048 nt
	buffer_load_dwordx4 v[40:43], v64, s[20:23], s29 offen offset:2048 nt
	buffer_load_dwordx4 v[44:47], v64, s[20:23], s30 offen offset:2048 nt
                                        ; kill: killed $sgpr8_sgpr9_sgpr10 killed $sgpr11
                                        ; kill: killed $sgpr34
                                        ; kill: killed $sgpr4_sgpr5_sgpr6 killed $sgpr7
                                        ; kill: killed $vgpr48
                                        ; kill: killed $sgpr24
                                        ; kill: killed $sgpr33
	s_nop 0
	buffer_load_dwordx4 v[48:51], v64, s[20:23], s28 offen offset:2048 nt
	buffer_load_dwordx4 v[52:55], v64, s[20:23], s31 offen offset:3072 nt
                                        ; kill: killed $sgpr31
	buffer_load_dwordx4 v[56:59], v64, s[20:23], s29 offen offset:3072 nt
	buffer_load_dwordx4 v[60:63], v64, s[20:23], s30 offen offset:3072 nt
	v_lshlrev_b32_e32 v65, 3, v0
	v_and_b32_e32 v72, 0x70, v65
	buffer_load_dwordx4 v[64:67], v64, s[20:23], s28 offen offset:3072 nt
	v_lshlrev_b32_e32 v3, 7, v3
	v_xad_u32 v83, v72, v2, v3
	;;#ASMSTART
	s_waitcnt vmcnt(23)
	;;#ASMEND
	s_barrier
	ds_read_b128 v[68:71], v83
	v_or_b32_e32 v2, 64, v2
	v_xad_u32 v84, v2, v72, v3
	ds_read_b128 v[72:75], v84
	;;#ASMSTART
	s_waitcnt vmcnt(22)
	;;#ASMEND
	s_barrier
	s_lshl_b32 s0, s16, 8
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_waitcnt lgkmcnt(0)
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_waitcnt vmcnt(19)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[68:71], v[4:7], 0, v77, v78 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(18)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[68:71], v[8:11], 0, v77, v78 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[68:71], v[12:15], 0, v77, v79 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(15)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[68:71], v[16:19], 0, v77, v79 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[72:75], v[20:23], v[2:5], v77, v78 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(13)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[72:75], v[24:27], v[6:9], v77, v78 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[18:21], v83 offset:2048
	ds_read_b128 v[22:25], v84 offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_waitcnt vmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[72:75], v[28:31], v[10:13], v77, v79 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[72:75], v[32:35], v[14:17], v77, v79 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[18:21], v[36:39], v[2:5], v80, v82 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[18:21], v[40:43], v[6:9], v80, v82 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[18:21], v[44:47], v[10:13], v80, v81 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_lshrrev_b32_e32 v6, 5, v0
	v_lshl_or_b32 v7, s17, 8, v76
	v_lshl_add_u32 v1, v1, 12, v7
	s_waitcnt vmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[18:21], v[48:51], v[14:17], v80, v81 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_e32 v20, s18, v6
	v_ashrrev_i32_e32 v21, 31, v20
	v_add_u32_e32 v7, 0xc00, v1
	s_waitcnt vmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[22:25], v[52:55], v[2:5], v80, v82 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshlrev_b32_e32 v0, 1, v0
	s_waitcnt vmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[22:25], v[56:59], v[26:29], v80, v82 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[2:3], v[20:21], 2, s[14:15]
	v_add_u32_e32 v4, 0x400, v1
	v_add_u32_e32 v5, 0x800, v1
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[22:25], v[60:63], v[8:11], v80, v81 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[22:25], v[64:67], v[12:15], v80, v81 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	ds_write2_b32 v1, v16, v26 offset1:16
	ds_write2_b32 v4, v17, v27 offset1:16
	ds_write2_b32 v5, v18, v28 offset1:16
	ds_write2_b32 v7, v19, v29 offset1:16
	s_nop 2
	ds_write2_b32 v1, v8, v12 offset0:32 offset1:48
	ds_write2_b32 v4, v9, v13 offset0:32 offset1:48
	ds_write2_b32 v5, v10, v14 offset0:32 offset1:48
	ds_write2_b32 v7, v11, v15 offset0:32 offset1:48
	s_waitcnt lgkmcnt(0)
	s_barrier
	global_load_dword v7, v[2:3], off
	v_and_b32_e32 v9, 62, v0
	v_mov_b32_e32 v1, 0
	v_lshlrev_b32_e32 v0, 1, v9
	v_lshl_add_u64 v[4:5], v[20:21], 2, s[12:13]
	v_lshl_add_u64 v[0:1], s[0:1], 0, v[0:1]
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v8, 0xffffff, v7
	v_cmp_gt_i32_e32 vcc, s19, v8
	v_lshlrev_b32_e32 v7, 2, v9
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_10
; %bb.9:
	global_load_dword v10, v[4:5], off
	v_lshl_or_b32 v11, v6, 10, v7
	ds_read_b64 v[12:13], v11
	s_movk_i32 s2, 0x3800
	v_mad_u64_u32 v[8:9], s[2:3], v8, s2, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_pk_mul_f32 v[12:13], v[10:11], v[12:13] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v12, v12, v13
	global_atomic_pk_add_bf16 v[8:9], v12, off
	ds_read_b64 v[12:13], v11 offset:256
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[12:13], v[10:11], v[12:13] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v12, v12, v13
	global_atomic_pk_add_bf16 v[8:9], v12, off offset:128
	ds_read_b64 v[12:13], v11 offset:512
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[12:13], v[10:11], v[12:13] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v12, v12, v13
	global_atomic_pk_add_bf16 v[8:9], v12, off offset:256
	ds_read_b64 v[12:13], v11 offset:768
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[10:11], v[10:11], v[12:13] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v10, v10, v11
	global_atomic_pk_add_bf16 v[8:9], v10, off offset:384
.LBB0_10:
	s_or_b64 exec, exec, s[0:1]
	global_load_dword v2, v[2:3], off offset:32
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v2, 0xffffff, v2
	v_cmp_gt_i32_e32 vcc, s19, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:
	global_load_dword v4, v[4:5], off offset:32
	v_lshl_or_b32 v5, v6, 10, v7
	ds_read_b64 v[6:7], v5 offset:8192
	s_movk_i32 s0, 0x3800
	v_mad_u64_u32 v[0:1], s[0:1], v2, s0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[4:5], v[6:7] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v2, v2, v3
	global_atomic_pk_add_bf16 v[0:1], v2, off
	ds_read_b64 v[2:3], v5 offset:8448
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[4:5], v[2:3] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v2, v2, v3
	global_atomic_pk_add_bf16 v[0:1], v2, off offset:128
	ds_read_b64 v[2:3], v5 offset:8704
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[4:5], v[2:3] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v2, v2, v3
	global_atomic_pk_add_bf16 v[0:1], v2, off offset:256
	ds_read_b64 v[2:3], v5 offset:8960
	s_waitcnt lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[4:5], v[2:3] op_sel_hi:[0,1]
	v_cvt_pk_bf16_f32 v2, v2, v3
	global_atomic_pk_add_bf16 v[0:1], v2, off offset:384
.LBB0_12:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 88
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 85
		.amdhsa_next_free_sgpr 35
		.amdhsa_accum_offset 88
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
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
	.section	.text,"axG",@progbits,_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph,comdat,unique,1
.Lfunc_end0:
	.size	_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph, .Lfunc_end0-_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph
                                        ; -- End function
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.num_vgpr, 85
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.num_agpr, 0
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.numbered_sgpr, 35
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.private_seg_size, 0
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.uses_vcc, 1
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.uses_flat_scratch, 0
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.has_dyn_sized_stack, 0
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.has_recursion, 0
	.set _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1768
; TotalNumSgprs: 41
; NumVgprs: 85
; NumAgprs: 0
; TotalNumVgprs: 85
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 10
; NumSGPRsForWavesPerEU: 41
; NumVGPRsForWavesPerEU: 85
; AccumOffset: 88
; Occupancy: 5
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 21
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_a2b4d0c3138f478c,@object ; @__hip_cuid_a2b4d0c3138f478c
	.section	.bss,"aw",@nobits,unique,2
	.globl	__hip_cuid_a2b4d0c3138f478c
__hip_cuid_a2b4d0c3138f478c:
	.byte	0                               ; 0x0
	.size	__hip_cuid_a2b4d0c3138f478c, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_a2b4d0c3138f478c
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
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
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph
    .private_segment_fixed_size: 0
    .sgpr_count:     41
    .sgpr_spill_count: 0
    .symbol:         _ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph.kd
    .uses_dynamic_stack: false
    .vgpr_count:     85
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
