	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text,"axG",@progbits,_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,comdat,unique,1
	.protected	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16 ; -- Begin function _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
	.globl	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
	.p2align	8
	.type	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,@function
_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16: ; @_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
; %bb.4:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.5:
.LBB0_0:
	s_load_dword s2, s[12:13], 0x0
	v_readfirstlane_b32 s20, v0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 28
	s_add_i32 s2, s2, s3
	s_ashr_i32 s2, s2, 4
	s_lshl_b32 s2, s2, 2
	s_cmp_ge_i32 s16, s2
	s_cbranch_scc1 .LBB0_3
; %bb.1:
	s_load_dwordx2 s[12:13], s[0:1], 0x50
	s_ashr_i32 s2, s16, 31
	s_lshr_b32 s2, s2, 30
	s_add_i32 s3, s16, s2
	s_ashr_i32 s2, s3, 2
	s_and_b32 s3, s3, -4
	s_mov_b64 s[4:5], s[6:7]
	s_sub_i32 s18, s16, s3
	s_ashr_i32 s3, s2, 31
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s9, s9, 0xffff
	s_lshr_b32 s19, s20, 6
	s_waitcnt lgkmcnt(0)
	s_and_b32 s13, s13, 0xffff
	s_lshl_b64 s[6:7], s[2:3], 2
	s_add_u32 s16, s10, s6
	s_addc_u32 s17, s11, s7
	s_lshl_b32 s3, s2, 4
	s_lshl_b32 s21, s19, 2
	v_bfe_u32 v1, v0, 4, 2
	s_add_i32 s6, s21, s3
	v_or_b32_e32 v2, s6, v1
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[14:15]
	global_load_dword v2, v[2:3], off
	s_load_dword s6, s[0:1], 0x38
	s_movk_i32 s10, 0x3800
	s_mov_b32 s7, 0x20000
	v_lshlrev_b32_e32 v3, 4, v0
	s_movk_i32 s11, 0xf0
	s_mov_b32 s15, s7
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s14, s6, 0x3800
	s_movk_i32 s6, 0x100
	s_load_dword s16, s[16:17], 0x0
	v_and_b32_e32 v110, 15, v0
	v_and_b32_e32 v6, 63, v0
	v_lshlrev_b32_e32 v8, 2, v110
	s_lshl_b32 s24, s18, 3
	s_lshl_b32 s25, s19, 1
	v_lshlrev_b32_e32 v6, 2, v6
	v_lshl_or_b32 v111, v1, 6, v8
	s_lshl_b32 s23, s18, 8
	s_add_i32 s17, s25, s24
	v_or_b32_e32 v8, s21, v1
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s21, s16, 10
	v_lshlrev_b32_e32 v7, 4, v110
	v_and_b32_e32 v9, 12, v6
	v_and_b32_e32 v6, 48, v6
	s_andn2_b32 s20, s20, 63
	s_mulk_i32 s17, 0x700
	s_mul_i32 s16, s16, 0xe000
	s_add_i32 s21, s21, s23
	s_movk_i32 s22, 0x70
	v_lshl_or_b32 v120, v1, 8, v7
	v_or_b32_e32 v7, 64, v6
	v_lshlrev_b32_e32 v10, 3, v8
	s_add_i32 s16, s16, s17
	s_add_i32 s17, s21, s20
	v_lshlrev_b32_e32 v11, 7, v8
	v_add_lshl_u32 v115, v8, v6, 2
	v_bitop3_b32 v6, v10, v6, s22 bitop3:0x6c
	v_bitop3_b32 v7, v10, v7, s22 bitop3:0x6c
	s_lshl_b32 s27, s16, 2
	s_mul_i32 s23, s17, 0xe00
	s_or_b32 s22, s17, 16
	s_or_b32 s21, s17, 32
	s_or_b32 s20, s17, 48
	v_or3_b32 v116, v6, v9, v11
	v_or3_b32 v117, v7, v9, v11
	s_mulk_i32 s22, 0xe00
	s_mulk_i32 s21, 0xe00
	s_mulk_i32 s20, 0xe00
	v_mov_b32_e32 v113, 0x200000
	v_mov_b32_e32 v82, 0
	v_mov_b32_e32 v83, 0
	v_mov_b32_e32 v84, 0
	v_mov_b32_e32 v85, 0
	s_add_i32 s28, s27, 0x1c00
	s_movk_i32 s16, 0x400
	s_movk_i32 s17, 0x500
	s_movk_i32 s33, 0xe00
	s_movk_i32 s26, 0x700
	s_movk_i32 s31, 0x1000
	s_add_i32 s25, s27, 0x1000
	s_movk_i32 s29, 0x1c00
	s_add_i32 s24, s28, 0x1000
	v_mov_b32_e32 v114, 0
	s_waitcnt vmcnt(0)
	v_mul_lo_u32 v2, v2, s10
	v_and_or_b32 v112, v3, s11, v2
	buffer_load_dwordx4 v[2:5], v112, s[12:15], 0 offen
	buffer_load_dwordx4 v[38:41], v112, s[12:15], s6 offen
	s_movk_i32 s6, 0x200
	buffer_load_dwordx4 v[42:45], v112, s[12:15], s6 offen
	s_movk_i32 s6, 0x300
	buffer_load_dwordx4 v[46:49], v112, s[12:15], s6 offen
	s_mov_b32 s6, 0x54380000
	s_mov_b32 s10, 0x5438000
	s_mov_b32 s11, s7
	buffer_load_dword v98, v111, s[8:11], s27 offen
	buffer_load_dwordx4 v[62:65], v120, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[66:69], v120, s[4:7], s23 offen offset:1024 nt
	buffer_load_dwordx4 v[22:25], v120, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[26:29], v120, s[4:7], s23 offen offset:3072 nt
	buffer_load_dwordx4 v[74:77], v120, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[78:81], v120, s[4:7], s22 offen offset:1024 nt
	buffer_load_dwordx4 v[50:53], v120, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[70:73], v120, s[4:7], s21 offen offset:1024 nt
	buffer_load_dwordx4 v[54:57], v120, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[58:61], v120, s[4:7], s20 offen offset:1024 nt
	buffer_load_dwordx4 v[14:17], v120, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v120, s[4:7], s22 offen offset:3072 nt
	buffer_load_dwordx4 v[30:33], v120, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[34:37], v120, s[4:7], s21 offen offset:3072 nt
	buffer_load_dwordx4 v[6:9], v120, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v120, s[4:7], s20 offen offset:3072 nt
	s_waitcnt vmcnt(20)
	v_and_b32_e32 v86, 0x7fff7fff, v2
	v_and_b32_e32 v87, 0x7fff7fff, v3
	v_and_b32_e32 v88, 0x7fff7fff, v4
	v_and_b32_e32 v89, 0x7fff7fff, v5
	;;#ASMSTART
	v_pk_max_u16 v86, v86, v87
	;;#ASMEND
	s_waitcnt vmcnt(19)
	v_and_b32_e32 v90, 0x7fff7fff, v38
	v_and_b32_e32 v91, 0x7fff7fff, v39
	v_and_b32_e32 v92, 0x7fff7fff, v40
	v_and_b32_e32 v93, 0x7fff7fff, v41
	;;#ASMSTART
	v_pk_max_u16 v87, v88, v89
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v88, v90, v91
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v89, v92, v93
	;;#ASMEND
	s_waitcnt vmcnt(18)
	v_and_b32_e32 v94, 0x7fff7fff, v42
	;;#ASMSTART
	v_pk_max_u16 v86, v86, v87
	;;#ASMEND
	v_and_b32_e32 v95, 0x7fff7fff, v43
	v_and_b32_e32 v96, 0x7fff7fff, v44
	v_and_b32_e32 v97, 0x7fff7fff, v45
	s_waitcnt vmcnt(17)
	v_and_b32_e32 v99, 0x7fff7fff, v46
	v_and_b32_e32 v100, 0x7fff7fff, v47
	v_and_b32_e32 v101, 0x7fff7fff, v48
	v_and_b32_e32 v102, 0x7fff7fff, v49
	;;#ASMSTART
	v_pk_max_u16 v90, v94, v95
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v91, v96, v97
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v92, v99, v100
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v93, v101, v102
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v87, v88, v89
	;;#ASMEND
	v_max_u16_sdwa v86, v86, v86 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	;;#ASMSTART
	v_pk_max_u16 v88, v90, v91
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v89, v92, v93
	;;#ASMEND
	v_max_u16_sdwa v87, v87, v87 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_max_u16_sdwa v88, v88, v88 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_max_u16_sdwa v89, v89, v89 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_max_u32_dpp v86, v86, v86 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_u32_dpp v87, v87, v87 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_u32_dpp v88, v88, v88 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_u32_dpp v89, v89, v89 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_u32_dpp v86, v86, v86 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_u32_dpp v87, v87, v87 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_u32_dpp v88, v88, v88 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_u32_dpp v89, v89, v89 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v86, v86, 16, v113
	v_lshl_add_u32 v87, v87, 16, v113
	v_lshl_add_u32 v88, v88, 16, v113
	v_lshl_add_u32 v89, v89, 16, v113
	v_bfe_u32 v86, v86, 23, 8
	v_bfe_u32 v87, v87, 23, 8
	v_bfe_u32 v88, v88, 23, 8
	v_bfe_u32 v89, v89, 23, 8
	v_max_u32_e32 v86, 2, v86
	v_max_u32_e32 v87, 2, v87
	v_max_u32_e32 v88, 2, v88
	v_max_u32_e32 v89, 2, v89
	v_add_u32_e32 v86, 0xfe, v86
	v_add_u32_e32 v87, 0xfe, v87
	v_add_u32_e32 v88, 0xfe, v88
	v_add_u32_e32 v89, 0xfe, v89
	v_and_b32_e32 v86, 0xff, v86
	v_and_b32_e32 v87, 0xff, v87
	v_and_b32_e32 v88, 0xff, v88
	v_and_b32_e32 v89, 0xff, v89
	v_lshlrev_b32_e32 v90, 23, v86
	v_lshlrev_b32_e32 v91, 23, v87
	v_lshl_or_b32 v86, v87, 16, v86
	v_lshlrev_b32_e32 v87, 23, v88
	v_lshlrev_b32_e32 v92, 23, v89
	v_cvt_scalef32_pk_fp4_bf16 v82, v2, v90
	v_cvt_scalef32_pk_fp4_bf16 v83, v38, v91
	v_cvt_scalef32_pk_fp4_bf16 v84, v42, v87
	v_cvt_scalef32_pk_fp4_bf16 v85, v46, v92
	v_cvt_scalef32_pk_fp4_bf16 v82, v3, v90 op_sel:[0,0,1,0]
	v_cvt_scalef32_pk_fp4_bf16 v83, v39, v91 op_sel:[0,0,1,0]
	v_cvt_scalef32_pk_fp4_bf16 v84, v43, v87 op_sel:[0,0,1,0]
	v_cvt_scalef32_pk_fp4_bf16 v85, v47, v92 op_sel:[0,0,1,0]
	v_cvt_scalef32_pk_fp4_bf16 v82, v4, v90 op_sel:[0,0,0,1]
	v_cvt_scalef32_pk_fp4_bf16 v83, v40, v91 op_sel:[0,0,0,1]
	v_cvt_scalef32_pk_fp4_bf16 v84, v44, v87 op_sel:[0,0,0,1]
	v_cvt_scalef32_pk_fp4_bf16 v85, v48, v92 op_sel:[0,0,0,1]
	v_cvt_scalef32_pk_fp4_bf16 v82, v5, v90 op_sel:[0,0,1,1]
	v_lshl_or_b32 v88, v89, 16, v88
	v_cvt_scalef32_pk_fp4_bf16 v83, v41, v91 op_sel:[0,0,1,1]
	v_cvt_scalef32_pk_fp4_bf16 v84, v45, v87 op_sel:[0,0,1,1]
	v_cvt_scalef32_pk_fp4_bf16 v85, v49, v92 op_sel:[0,0,1,1]
	ds_write_b32 v116, v82
	ds_write_b32 v117, v83
	ds_write_b32 v115, v86 offset:6144
	ds_write_b32 v116, v84 offset:2048
	ds_write_b32 v117, v85 offset:2048
	ds_write_b32 v115, v88 offset:6400
	buffer_load_dword v102, v111, s[8:11], s28 offen offset:256
	buffer_load_dword v99, v111, s[8:11], s28 offen
	buffer_load_dword v103, v111, s[8:11], s27 offen offset:256
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[2:5], v112, s[12:15], s16 offen
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s17 offen
	v_lshlrev_b32_e32 v39, 3, v0
	v_and_b32_e32 v38, 48, v0
	v_and_b32_e32 v39, 0x70, v39
	v_lshlrev_b32_e32 v40, 7, v110
	v_bitop3_b32 v119, v40, v39, v38 bitop3:0xf6
	v_or_b32_e32 v38, 64, v38
	v_bitop3_b32 v118, v40, v38, v39 bitop3:0xf6
	ds_read_b128 v[86:89], v118
	ds_read_b128 v[90:93], v119
	ds_read_b32 v100, v111 offset:6144
	s_load_dwordx2 s[16:17], s[0:1], 0x40
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(20) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[90:93], v[62:65], 0, v100, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(19)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[86:89], v[66:69], v[38:41], v100, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v104, 0x1000, v120
	buffer_load_dwordx4 v[62:65], v104, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[66:69], v104, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(18)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[90:93], v[74:77], 0, v100, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(17)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[86:89], v[78:81], v[38:41], v100, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 4
	buffer_load_dwordx4 v[38:41], v104, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[42:45], v104, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[90:93], v[50:53], 0, v100, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[86:89], v[70:73], v[46:49], v100, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[46:49], v104, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[50:53], v104, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[90:93], v[54:57], 0, v100, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[86:89], v[58:61], v[54:57], v100, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v86, 0x7fff7fff, v2
	v_and_b32_e32 v87, 0x7fff7fff, v3
	;;#ASMSTART
	v_pk_max_u16 v86, v86, v87
	;;#ASMEND
	v_and_b32_e32 v88, 0x7fff7fff, v4
	v_and_b32_e32 v89, 0x7fff7fff, v5
	;;#ASMSTART
	v_pk_max_u16 v87, v88, v89
	;;#ASMEND
	v_mov_b32_e32 v88, 0
	;;#ASMSTART
	v_pk_max_u16 v86, v86, v87
	;;#ASMEND
	buffer_load_dwordx4 v[54:57], v104, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[58:61], v104, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v105, v111, s[8:11], s27 offen offset:512
	buffer_load_dword v106, v111, s[8:11], s28 offen offset:512
	v_max_u16_sdwa v86, v86, v86 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s30, 0x600
	s_nop 0
	v_max_u32_dpp v86, v86, v86 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v86, v86, v86 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v86, v86, 16, v113
	v_bfe_u32 v86, v86, 23, 8
	v_max_u32_e32 v86, 2, v86
	v_add_u32_e32 v86, 0xfe, v86
	v_and_b32_e32 v86, 0xff, v86
	v_lshlrev_b32_e32 v87, 23, v86
	v_cvt_scalef32_pk_fp4_bf16 v88, v2, v87
	v_cvt_scalef32_pk_fp4_bf16 v88, v3, v87 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v2, 0x7fff7fff, v82
	v_cvt_scalef32_pk_fp4_bf16 v88, v4, v87 op_sel:[0,0,0,1]
	v_and_b32_e32 v3, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v2, v2, v3
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v88, v5, v87 op_sel:[0,0,1,1]
	v_and_b32_e32 v4, 0x7fff7fff, v84
	v_and_b32_e32 v5, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v3, v4, v5
	;;#ASMEND
	v_mov_b32_e32 v4, 0
	;;#ASMSTART
	v_pk_max_u16 v2, v2, v3
	;;#ASMEND
	ds_write_b32 v116, v88 offset:4096
	v_max_u16_sdwa v2, v2, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v2, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v2, v2, 16, v113
	v_bfe_u32 v2, v2, 23, 8
	v_max_u32_e32 v2, 2, v2
	v_add_u32_e32 v2, 0xfe, v2
	v_and_b32_e32 v2, 0xff, v2
	v_lshlrev_b32_e32 v3, 23, v2
	v_cvt_scalef32_pk_fp4_bf16 v4, v82, v3
	v_cvt_scalef32_pk_fp4_bf16 v4, v83, v3 op_sel:[0,0,1,0]
	v_lshl_or_b32 v2, v2, 16, v86
	v_cvt_scalef32_pk_fp4_bf16 v4, v84, v3 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v4, v85, v3 op_sel:[0,0,1,1]
	ds_write_b32 v117, v4 offset:4096
	ds_write_b32 v115, v2 offset:6656
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s30 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s26 offen
	ds_read_b128 v[90:93], v118 offset:2048
	ds_read_b128 v[98:101], v119 offset:2048
	ds_read_b32 v107, v111 offset:6400
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[98:101], v[22:25], v[94:97], v107, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[90:93], v[26:29], v[2:5], v107, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v104, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[22:25], v104, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[98:101], v[14:17], v[74:77], v107, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[18:21], v[14:17], v107, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[14:17], v104, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v104, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[98:101], v[30:33], v[70:73], v107, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[34:37], v[26:29], v107, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v104, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v104, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[98:101], v[6:9], v[78:81], v107, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[10:13], v[6:9], v107, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v34, 0x7fff7fff, v82
	v_and_b32_e32 v35, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v34, v34, v35
	;;#ASMEND
	v_and_b32_e32 v36, 0x7fff7fff, v84
	v_and_b32_e32 v37, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v35, v36, v37
	;;#ASMEND
	v_mov_b32_e32 v36, 0
	;;#ASMSTART
	v_pk_max_u16 v34, v34, v35
	;;#ASMEND
	buffer_load_dwordx4 v[6:9], v104, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v104, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v102, v111, s[8:11], s27 offen offset:768
	buffer_load_dword v103, v111, s[8:11], s28 offen offset:768
	v_max_u16_sdwa v34, v34, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v37, 0x7fff7fff, v88
	s_movk_i32 s26, 0x800
	v_max_u32_dpp v34, v34, v34 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s30, 0x900
	s_nop 0
	v_max_u32_dpp v34, v34, v34 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v34, v34, 16, v113
	v_bfe_u32 v34, v34, 23, 8
	v_max_u32_e32 v34, 2, v34
	v_add_u32_e32 v34, 0xfe, v34
	v_and_b32_e32 v34, 0xff, v34
	v_lshlrev_b32_e32 v35, 23, v34
	v_cvt_scalef32_pk_fp4_bf16 v36, v82, v35
	v_cvt_scalef32_pk_fp4_bf16 v36, v83, v35 op_sel:[0,0,1,0]
	v_and_b32_e32 v82, 0x7fff7fff, v89
	v_cvt_scalef32_pk_fp4_bf16 v36, v84, v35 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v36, v85, v35 op_sel:[0,0,1,1]
	v_and_b32_e32 v35, 0x7fff7fff, v86
	ds_write_b32 v116, v36
	v_and_b32_e32 v36, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v35, v35, v36
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v36, v37, v82
	;;#ASMEND
	v_mov_b32_e32 v37, 0
	;;#ASMSTART
	v_pk_max_u16 v35, v35, v36
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v35, v35, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v35, v35, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v35, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v35, v35, 16, v113
	v_bfe_u32 v35, v35, 23, 8
	v_max_u32_e32 v35, 2, v35
	v_add_u32_e32 v35, 0xfe, v35
	v_and_b32_e32 v35, 0xff, v35
	v_lshlrev_b32_e32 v36, 23, v35
	v_cvt_scalef32_pk_fp4_bf16 v37, v86, v36
	v_cvt_scalef32_pk_fp4_bf16 v37, v87, v36 op_sel:[0,0,1,0]
	v_lshl_or_b32 v34, v35, 16, v34
	v_cvt_scalef32_pk_fp4_bf16 v37, v88, v36 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v37, v89, v36 op_sel:[0,0,1,1]
	ds_write_b32 v117, v37
	ds_write_b32 v115, v34 offset:6912
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s26 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s30 offen
	ds_read_b128 v[90:93], v118 offset:4096
	ds_read_b128 v[98:101], v119 offset:4096
	ds_read_b32 v104, v111 offset:6656
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[98:101], v[62:65], v[94:97], v104, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[90:93], v[66:69], v[34:37], v104, v105 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v107, 0x2000, v120
	buffer_load_dwordx4 v[62:65], v107, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[66:69], v107, s[4:7], s23 offen offset:1024 nt
	s_movk_i32 s30, 0x2000
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[98:101], v[38:41], v[74:77], v104, v105 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[42:45], v[34:37], v104, v105 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[34:37], v107, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[38:41], v107, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[98:101], v[46:49], v[70:73], v104, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[50:53], v[42:45], v104, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[42:45], v107, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[46:49], v107, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[98:101], v[54:57], v[78:81], v104, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[90:93], v[58:61], v[50:53], v104, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v78, 0x7fff7fff, v82
	v_and_b32_e32 v79, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v78, v78, v79
	;;#ASMEND
	v_and_b32_e32 v80, 0x7fff7fff, v84
	v_and_b32_e32 v81, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v79, v80, v81
	;;#ASMEND
	v_mov_b32_e32 v80, 0
	;;#ASMSTART
	v_pk_max_u16 v78, v78, v79
	;;#ASMEND
	buffer_load_dwordx4 v[50:53], v107, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[54:57], v107, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v121, v111, s[8:11], s27 offen offset:1024
	buffer_load_dword v130, v111, s[8:11], s28 offen offset:1024
	v_max_u16_sdwa v78, v78, v78 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v81, 0x7fff7fff, v88
	s_movk_i32 s26, 0xa00
	v_max_u32_dpp v78, v78, v78 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s34, 0xb00
	s_nop 0
	v_max_u32_dpp v78, v78, v78 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v78, v78, 16, v113
	v_bfe_u32 v78, v78, 23, 8
	v_max_u32_e32 v78, 2, v78
	v_add_u32_e32 v78, 0xfe, v78
	v_and_b32_e32 v78, 0xff, v78
	v_lshlrev_b32_e32 v79, 23, v78
	v_cvt_scalef32_pk_fp4_bf16 v80, v82, v79
	v_cvt_scalef32_pk_fp4_bf16 v80, v83, v79 op_sel:[0,0,1,0]
	v_and_b32_e32 v82, 0x7fff7fff, v89
	v_cvt_scalef32_pk_fp4_bf16 v80, v84, v79 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v80, v85, v79 op_sel:[0,0,1,1]
	v_and_b32_e32 v79, 0x7fff7fff, v86
	ds_write_b32 v116, v80 offset:2048
	v_and_b32_e32 v80, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v79, v79, v80
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v80, v81, v82
	;;#ASMEND
	v_mov_b32_e32 v81, 0
	;;#ASMSTART
	v_pk_max_u16 v79, v79, v80
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v79, v79, v79 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v79, v79, v79 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v79, v79, v79 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v79, v79, 16, v113
	v_bfe_u32 v79, v79, 23, 8
	v_max_u32_e32 v79, 2, v79
	v_add_u32_e32 v79, 0xfe, v79
	v_and_b32_e32 v79, 0xff, v79
	v_lshlrev_b32_e32 v80, 23, v79
	v_cvt_scalef32_pk_fp4_bf16 v81, v86, v80
	v_cvt_scalef32_pk_fp4_bf16 v81, v87, v80 op_sel:[0,0,1,0]
	v_lshl_or_b32 v78, v79, 16, v78
	v_cvt_scalef32_pk_fp4_bf16 v81, v88, v80 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v81, v89, v80 op_sel:[0,0,1,1]
	ds_write_b32 v117, v81 offset:2048
	ds_write_b32 v115, v78 offset:7168
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[78:81], v112, s[12:15], s26 offen
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s34 offen
	ds_read_b128 v[86:89], v118
	ds_read_b128 v[90:93], v119
	ds_read_b32 v104, v111 offset:6912
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[90:93], v[2:5], v[94:97], v104, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[86:89], v[22:25], v[2:5], v104, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v107, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[22:25], v107, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[90:93], v[14:17], v[74:77], v104, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[86:89], v[18:21], v[14:17], v104, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[14:17], v107, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v107, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[90:93], v[26:29], v[70:73], v104, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[86:89], v[30:33], v[26:29], v104, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v107, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v107, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[90:93], v[6:9], v[58:61], v104, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[86:89], v[10:13], v[6:9], v104, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v70, 0x7fff7fff, v78
	v_and_b32_e32 v71, 0x7fff7fff, v79
	;;#ASMSTART
	v_pk_max_u16 v70, v70, v71
	;;#ASMEND
	v_and_b32_e32 v72, 0x7fff7fff, v80
	v_and_b32_e32 v73, 0x7fff7fff, v81
	;;#ASMSTART
	v_pk_max_u16 v71, v72, v73
	;;#ASMEND
	v_mov_b32_e32 v72, 0
	;;#ASMSTART
	v_pk_max_u16 v70, v70, v71
	;;#ASMEND
	buffer_load_dwordx4 v[6:9], v107, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v107, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v134, v111, s[8:11], s27 offen offset:1280
	buffer_load_dword v135, v111, s[8:11], s28 offen offset:1280
	v_max_u16_sdwa v70, v70, v70 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v73, 0x7fff7fff, v84
	s_movk_i32 s26, 0xc00
	v_max_u32_dpp v70, v70, v70 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s34, 0xd00
	s_nop 0
	v_max_u32_dpp v70, v70, v70 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v70, v70, 16, v113
	v_bfe_u32 v70, v70, 23, 8
	v_max_u32_e32 v70, 2, v70
	v_add_u32_e32 v70, 0xfe, v70
	v_and_b32_e32 v70, 0xff, v70
	v_lshlrev_b32_e32 v71, 23, v70
	v_cvt_scalef32_pk_fp4_bf16 v72, v78, v71
	v_cvt_scalef32_pk_fp4_bf16 v72, v79, v71 op_sel:[0,0,1,0]
	v_and_b32_e32 v78, 0x7fff7fff, v85
	v_cvt_scalef32_pk_fp4_bf16 v72, v80, v71 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v72, v81, v71 op_sel:[0,0,1,1]
	v_and_b32_e32 v71, 0x7fff7fff, v82
	ds_write_b32 v116, v72 offset:4096
	v_and_b32_e32 v72, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v71, v71, v72
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v72, v73, v78
	;;#ASMEND
	v_mov_b32_e32 v73, 0
	;;#ASMSTART
	v_pk_max_u16 v71, v71, v72
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v71, v71, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v71, v71, v71 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v71, v71, v71 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v71, v71, 16, v113
	v_bfe_u32 v71, v71, 23, 8
	v_max_u32_e32 v71, 2, v71
	v_add_u32_e32 v71, 0xfe, v71
	v_and_b32_e32 v71, 0xff, v71
	v_lshlrev_b32_e32 v72, 23, v71
	v_cvt_scalef32_pk_fp4_bf16 v73, v82, v72
	v_cvt_scalef32_pk_fp4_bf16 v73, v83, v72 op_sel:[0,0,1,0]
	v_lshl_or_b32 v70, v71, 16, v70
	v_cvt_scalef32_pk_fp4_bf16 v73, v84, v72 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v73, v85, v72 op_sel:[0,0,1,1]
	ds_write_b32 v117, v73 offset:4096
	ds_write_b32 v115, v70 offset:7424
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[102:105], v112, s[12:15], s26 offen
	buffer_load_dwordx4 v[106:109], v112, s[12:15], s34 offen
	ds_read_b128 v[90:93], v118 offset:2048
	ds_read_b128 v[122:125], v119 offset:2048
	ds_read_b32 v131, v111 offset:7168
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[122:125], v[62:65], v[94:97], v131, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], v[90:93], v[66:69], v[62:65], v131, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v136, 0x3000, v120
	buffer_load_dwordx4 v[66:69], v136, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[70:73], v136, s[4:7], s23 offen offset:1024 nt
	s_movk_i32 s26, 0x3000
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[122:125], v[34:37], v[74:77], v131, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[90:93], v[38:41], v[34:37], v131, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 0
	buffer_load_dwordx4 v[74:77], v136, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[78:81], v136, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[122:125], v[42:45], v[98:101], v131, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[90:93], v[46:49], v[34:37], v131, v130 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[82:85], v136, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[86:89], v136, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[122:125], v[50:53], v[58:61], v131, v130 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[90:93], v[54:57], v[34:37], v131, v130 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	s_nop 4
	v_and_b32_e32 v34, 0x7fff7fff, v102
	v_and_b32_e32 v35, 0x7fff7fff, v103
	;;#ASMSTART
	v_pk_max_u16 v34, v34, v35
	;;#ASMEND
	v_and_b32_e32 v36, 0x7fff7fff, v104
	v_and_b32_e32 v37, 0x7fff7fff, v105
	;;#ASMSTART
	v_pk_max_u16 v35, v36, v37
	;;#ASMEND
	v_mov_b32_e32 v36, 0
	;;#ASMSTART
	v_pk_max_u16 v34, v34, v35
	;;#ASMEND
	buffer_load_dwordx4 v[90:93], v136, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[94:97], v136, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v121, v111, s[8:11], s27 offen offset:1536
	buffer_load_dword v137, v111, s[8:11], s28 offen offset:1536
	v_max_u16_sdwa v34, v34, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v37, 0x7fff7fff, v108
	v_and_b32_e32 v38, 0x7fff7fff, v109
	v_max_u32_dpp v34, v34, v34 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s34, 0xf00
	s_nop 0
	v_max_u32_dpp v34, v34, v34 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v34, v34, 16, v113
	v_bfe_u32 v34, v34, 23, 8
	v_max_u32_e32 v34, 2, v34
	v_add_u32_e32 v34, 0xfe, v34
	v_and_b32_e32 v34, 0xff, v34
	v_lshlrev_b32_e32 v35, 23, v34
	v_cvt_scalef32_pk_fp4_bf16 v36, v102, v35
	v_cvt_scalef32_pk_fp4_bf16 v36, v103, v35 op_sel:[0,0,1,0]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v36, v104, v35 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v36, v105, v35 op_sel:[0,0,1,1]
	v_and_b32_e32 v35, 0x7fff7fff, v106
	ds_write_b32 v116, v36
	v_and_b32_e32 v36, 0x7fff7fff, v107
	;;#ASMSTART
	v_pk_max_u16 v35, v35, v36
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v36, v37, v38
	;;#ASMEND
	v_mov_b32_e32 v37, 0
	;;#ASMSTART
	v_pk_max_u16 v35, v35, v36
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v35, v35, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v35, v35, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v35, v35, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v35, v35, 16, v113
	v_bfe_u32 v35, v35, 23, 8
	v_max_u32_e32 v35, 2, v35
	v_add_u32_e32 v35, 0xfe, v35
	v_and_b32_e32 v35, 0xff, v35
	v_lshlrev_b32_e32 v36, 23, v35
	v_cvt_scalef32_pk_fp4_bf16 v37, v106, v36
	v_cvt_scalef32_pk_fp4_bf16 v37, v107, v36 op_sel:[0,0,1,0]
	v_lshl_or_b32 v34, v35, 16, v34
	v_cvt_scalef32_pk_fp4_bf16 v37, v108, v36 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v37, v109, v36 op_sel:[0,0,1,1]
	ds_write_b32 v117, v37
	ds_write_b32 v115, v34 offset:7680
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[102:105], v112, s[12:15], s33 offen
	buffer_load_dwordx4 v[106:109], v112, s[12:15], s34 offen
	ds_read_b128 v[122:125], v118 offset:4096
	ds_read_b128 v[130:133], v119 offset:4096
	ds_read_b32 v138, v111 offset:7424
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[130:133], v[2:5], v[62:65], v138, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[122:125], v[22:25], v[2:5], v138, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v136, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[38:41], v136, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[130:133], v[14:17], v[126:129], v138, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[122:125], v[18:21], v[14:17], v138, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v136, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v136, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[130:133], v[26:29], v[98:101], v138, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[122:125], v[30:33], v[18:21], v138, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v136, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v136, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[130:133], v[6:9], v[58:61], v138, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[122:125], v[10:13], v[6:9], v138, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	s_nop 4
	v_and_b32_e32 v6, 0x7fff7fff, v102
	v_and_b32_e32 v7, 0x7fff7fff, v103
	;;#ASMSTART
	v_pk_max_u16 v6, v6, v7
	;;#ASMEND
	v_and_b32_e32 v8, 0x7fff7fff, v104
	v_and_b32_e32 v9, 0x7fff7fff, v105
	;;#ASMSTART
	v_pk_max_u16 v7, v8, v9
	;;#ASMEND
	v_mov_b32_e32 v8, 0
	;;#ASMSTART
	v_pk_max_u16 v6, v6, v7
	;;#ASMEND
	buffer_load_dwordx4 v[58:61], v136, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[62:65], v136, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v111, s[8:11], s27 offen offset:1792
	buffer_load_dword v123, v111, s[8:11], s28 offen offset:1792
	v_max_u16_sdwa v6, v6, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v9, 0x7fff7fff, v108
	v_and_b32_e32 v10, 0x7fff7fff, v109
	v_max_u32_dpp v6, v6, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s33, 0x1100
	s_nop 0
	v_max_u32_dpp v6, v6, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v6, v6, 16, v113
	v_bfe_u32 v6, v6, 23, 8
	v_max_u32_e32 v6, 2, v6
	v_add_u32_e32 v6, 0xfe, v6
	v_and_b32_e32 v6, 0xff, v6
	v_lshlrev_b32_e32 v7, 23, v6
	v_cvt_scalef32_pk_fp4_bf16 v8, v102, v7
	v_cvt_scalef32_pk_fp4_bf16 v8, v103, v7 op_sel:[0,0,1,0]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v8, v104, v7 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v8, v105, v7 op_sel:[0,0,1,1]
	v_and_b32_e32 v7, 0x7fff7fff, v106
	ds_write_b32 v116, v8 offset:2048
	v_and_b32_e32 v8, 0x7fff7fff, v107
	;;#ASMSTART
	v_pk_max_u16 v7, v7, v8
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v8, v9, v10
	;;#ASMEND
	v_mov_b32_e32 v9, 0
	;;#ASMSTART
	v_pk_max_u16 v7, v7, v8
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v7, v7, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v7, v7, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v7, v7, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v7, v7, 16, v113
	v_bfe_u32 v7, v7, 23, 8
	v_max_u32_e32 v7, 2, v7
	v_add_u32_e32 v7, 0xfe, v7
	v_and_b32_e32 v7, 0xff, v7
	v_lshlrev_b32_e32 v8, 23, v7
	v_cvt_scalef32_pk_fp4_bf16 v9, v106, v8
	v_cvt_scalef32_pk_fp4_bf16 v9, v107, v8 op_sel:[0,0,1,0]
	v_lshl_or_b32 v6, v7, 16, v6
	v_cvt_scalef32_pk_fp4_bf16 v9, v108, v8 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v9, v109, v8 op_sel:[0,0,1,1]
	ds_write_b32 v117, v9 offset:2048
	ds_write_b32 v115, v6 offset:7936
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[26:29], v112, s[12:15], s31 offen
	buffer_load_dwordx4 v[30:33], v112, s[12:15], s33 offen
	ds_read_b128 v[98:101], v118
	ds_read_b128 v[102:105], v119
	ds_read_b32 v124, v111 offset:7680
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[102:105], v[66:69], v[2:5], v124, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[98:101], v[70:73], v[2:5], v124, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v125, 0x4000, v120
	buffer_load_dwordx4 v[66:69], v125, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[70:73], v125, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[102:105], v[74:77], v[14:17], v124, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[98:101], v[78:81], v[2:5], v124, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v125, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[6:9], v125, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[102:105], v[82:85], v[18:21], v124, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[98:101], v[86:89], v[10:13], v124, v137 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[10:13], v125, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[14:17], v125, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[102:105], v[90:93], v[22:25], v124, v137 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[98:101], v[94:97], v[18:21], v124, v137 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v86, 0x7fff7fff, v26
	v_and_b32_e32 v87, 0x7fff7fff, v27
	;;#ASMSTART
	v_pk_max_u16 v86, v86, v87
	;;#ASMEND
	v_and_b32_e32 v88, 0x7fff7fff, v28
	v_and_b32_e32 v89, 0x7fff7fff, v29
	;;#ASMSTART
	v_pk_max_u16 v87, v88, v89
	;;#ASMEND
	v_mov_b32_e32 v88, 0
	;;#ASMSTART
	v_pk_max_u16 v86, v86, v87
	;;#ASMEND
	buffer_load_dwordx4 v[18:21], v125, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[22:25], v125, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v121, v111, s[8:11], s27 offen offset:2048
	buffer_load_dword v124, v111, s[8:11], s28 offen offset:2048
	v_max_u16_sdwa v86, v86, v86 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s31, 0x1200
	s_movk_i32 s33, 0x1300
	v_max_u32_dpp v86, v86, v86 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v86, v86, v86 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v86, v86, 16, v113
	v_bfe_u32 v86, v86, 23, 8
	v_max_u32_e32 v86, 2, v86
	v_add_u32_e32 v86, 0xfe, v86
	v_and_b32_e32 v86, 0xff, v86
	v_lshlrev_b32_e32 v87, 23, v86
	v_cvt_scalef32_pk_fp4_bf16 v88, v26, v87
	v_cvt_scalef32_pk_fp4_bf16 v88, v27, v87 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v26, 0x7fff7fff, v30
	v_cvt_scalef32_pk_fp4_bf16 v88, v28, v87 op_sel:[0,0,0,1]
	v_and_b32_e32 v27, 0x7fff7fff, v31
	;;#ASMSTART
	v_pk_max_u16 v26, v26, v27
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v88, v29, v87 op_sel:[0,0,1,1]
	v_and_b32_e32 v28, 0x7fff7fff, v32
	v_and_b32_e32 v29, 0x7fff7fff, v33
	;;#ASMSTART
	v_pk_max_u16 v27, v28, v29
	;;#ASMEND
	v_mov_b32_e32 v28, 0
	;;#ASMSTART
	v_pk_max_u16 v26, v26, v27
	;;#ASMEND
	ds_write_b32 v116, v88 offset:4096
	v_max_u16_sdwa v26, v26, v26 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v26, v26, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v26, v26, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v26, v26, 16, v113
	v_bfe_u32 v26, v26, 23, 8
	v_max_u32_e32 v26, 2, v26
	v_add_u32_e32 v26, 0xfe, v26
	v_and_b32_e32 v26, 0xff, v26
	v_lshlrev_b32_e32 v27, 23, v26
	v_cvt_scalef32_pk_fp4_bf16 v28, v30, v27
	v_cvt_scalef32_pk_fp4_bf16 v28, v31, v27 op_sel:[0,0,1,0]
	v_lshl_or_b32 v26, v26, 16, v86
	v_cvt_scalef32_pk_fp4_bf16 v28, v32, v27 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v28, v33, v27 op_sel:[0,0,1,1]
	ds_write_b32 v117, v28 offset:4096
	ds_write_b32 v115, v26 offset:8192
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s31 offen
	buffer_load_dwordx4 v[90:93], v112, s[12:15], s33 offen
	ds_read_b128 v[94:97], v118 offset:2048
	ds_read_b128 v[98:101], v119 offset:2048
	ds_read_b32 v126, v111 offset:7936
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[98:101], v[34:37], v[106:109], v126, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[94:97], v[38:41], v[26:29], v126, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v125, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v125, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[98:101], v[42:45], v[74:77], v126, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[94:97], v[46:49], v[34:37], v126, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[34:37], v125, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[38:41], v125, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[98:101], v[50:53], v[78:81], v126, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[94:97], v[54:57], v[42:45], v126, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[42:45], v125, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v125, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[98:101], v[58:61], v[82:85], v126, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[94:97], v[62:65], v[50:53], v126, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v58, 0x7fff7fff, v86
	v_and_b32_e32 v59, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v58, v58, v59
	;;#ASMEND
	v_and_b32_e32 v60, 0x7fff7fff, v88
	v_and_b32_e32 v61, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v59, v60, v61
	;;#ASMEND
	v_mov_b32_e32 v60, 0
	;;#ASMSTART
	v_pk_max_u16 v58, v58, v59
	;;#ASMEND
	buffer_load_dwordx4 v[50:53], v125, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v125, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v106, v111, s[8:11], s27 offen offset:2304
	buffer_load_dword v107, v111, s[8:11], s28 offen offset:2304
	v_max_u16_sdwa v58, v58, v58 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v61, 0x7fff7fff, v92
	v_and_b32_e32 v62, 0x7fff7fff, v93
	v_max_u32_dpp v58, v58, v58 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s31, 0x1400
	s_movk_i32 s33, 0x1500
	v_max_u32_dpp v58, v58, v58 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v58, v58, 16, v113
	v_bfe_u32 v58, v58, 23, 8
	v_max_u32_e32 v58, 2, v58
	v_add_u32_e32 v58, 0xfe, v58
	v_and_b32_e32 v58, 0xff, v58
	v_lshlrev_b32_e32 v59, 23, v58
	v_cvt_scalef32_pk_fp4_bf16 v60, v86, v59
	v_cvt_scalef32_pk_fp4_bf16 v60, v87, v59 op_sel:[0,0,1,0]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v60, v88, v59 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v60, v89, v59 op_sel:[0,0,1,1]
	v_and_b32_e32 v59, 0x7fff7fff, v90
	ds_write_b32 v116, v60
	v_and_b32_e32 v60, 0x7fff7fff, v91
	;;#ASMSTART
	v_pk_max_u16 v59, v59, v60
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v60, v61, v62
	;;#ASMEND
	v_mov_b32_e32 v61, 0
	;;#ASMSTART
	v_pk_max_u16 v59, v59, v60
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v59, v59, v59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v59, v59, v59 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v59, v59, v59 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v59, v59, 16, v113
	v_bfe_u32 v59, v59, 23, 8
	v_max_u32_e32 v59, 2, v59
	v_add_u32_e32 v59, 0xfe, v59
	v_and_b32_e32 v59, 0xff, v59
	v_lshlrev_b32_e32 v60, 23, v59
	v_cvt_scalef32_pk_fp4_bf16 v61, v90, v60
	v_cvt_scalef32_pk_fp4_bf16 v61, v91, v60 op_sel:[0,0,1,0]
	v_lshl_or_b32 v58, v59, 16, v58
	v_cvt_scalef32_pk_fp4_bf16 v61, v92, v60 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v61, v93, v60 op_sel:[0,0,1,1]
	ds_write_b32 v117, v61
	ds_write_b32 v115, v58 offset:8448
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s31 offen
	buffer_load_dwordx4 v[90:93], v112, s[12:15], s33 offen
	ds_read_b128 v[94:97], v118 offset:4096
	ds_read_b128 v[98:101], v119 offset:4096
	ds_read_b32 v108, v111 offset:8192
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[98:101], v[66:69], v[102:105], v108, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[94:97], v[70:73], v[58:61], v108, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 0
	v_or_b32_e32 v102, 0x5000, v120
	s_nop 3
	buffer_load_dwordx4 v[58:61], v102, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[62:65], v102, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[98:101], v[2:5], v[74:77], v108, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[94:97], v[6:9], v[2:5], v108, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v102, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[6:9], v102, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[98:101], v[10:13], v[78:81], v108, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[94:97], v[14:17], v[10:13], v108, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[10:13], v102, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[14:17], v102, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[98:101], v[18:21], v[82:85], v108, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[94:97], v[22:25], v[18:21], v108, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	buffer_load_dwordx4 v[18:21], v102, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[22:25], v102, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v98, v111, s[8:11], s27 offen offset:2560
	buffer_load_dword v99, v111, s[8:11], s28 offen offset:2560
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v85, 0x7fff7fff, v92
	s_movk_i32 s31, 0x1600
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s33, 0x1700
	s_nop 0
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_and_b32_e32 v86, 0x7fff7fff, v93
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	v_and_b32_e32 v83, 0x7fff7fff, v90
	ds_write_b32 v116, v84 offset:2048
	v_and_b32_e32 v84, 0x7fff7fff, v91
	;;#ASMSTART
	v_pk_max_u16 v83, v83, v84
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v84, v85, v86
	;;#ASMEND
	v_mov_b32_e32 v85, 0
	;;#ASMSTART
	v_pk_max_u16 v83, v83, v84
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v83, v83, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v83, v83, v83 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v83, v83, v83 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v83, v83, 16, v113
	v_bfe_u32 v83, v83, 23, 8
	v_max_u32_e32 v83, 2, v83
	v_add_u32_e32 v83, 0xfe, v83
	v_and_b32_e32 v83, 0xff, v83
	v_lshlrev_b32_e32 v84, 23, v83
	v_cvt_scalef32_pk_fp4_bf16 v85, v90, v84
	v_cvt_scalef32_pk_fp4_bf16 v85, v91, v84 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v83, 16, v82
	v_cvt_scalef32_pk_fp4_bf16 v85, v92, v84 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v85, v93, v84 op_sel:[0,0,1,1]
	ds_write_b32 v117, v85 offset:2048
	ds_write_b32 v115, v82 offset:8704
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s31 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s33 offen
	ds_read_b128 v[90:93], v118
	ds_read_b128 v[94:97], v119
	ds_read_b32 v100, v111 offset:8448
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[94:97], v[26:29], v[66:69], v100, v106 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[30:33], v[26:29], v100, v106 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v102, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v102, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[34:37], v[70:73], v100, v106 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[38:41], v[34:37], v100, v106 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[34:37], v102, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[38:41], v102, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[94:97], v[42:45], v[74:77], v100, v107 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[46:49], v[42:45], v100, v107 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[42:45], v102, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v102, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[94:97], v[50:53], v[78:81], v100, v107 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[54:57], v[50:53], v100, v107 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[50:53], v102, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v102, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v100, v111, s[8:11], s27 offen offset:2816
	buffer_load_dword v101, v111, s[8:11], s28 offen offset:2816
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s31, 0x1800
	s_movk_i32 s33, 0x1900
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92 offset:4096
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84 offset:4096
	ds_write_b32 v115, v82 offset:8960
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s31 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s33 offen
	ds_read_b128 v[90:93], v118 offset:2048
	ds_read_b128 v[94:97], v119 offset:2048
	ds_read_b32 v102, v111 offset:8704
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[94:97], v[58:61], v[66:69], v102, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[62:65], v[58:61], v102, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v103, 0x6000, v120
	s_nop 4
	buffer_load_dwordx4 v[58:61], v103, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[62:65], v103, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[94:97], v[2:5], v[70:73], v102, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[6:9], v[2:5], v102, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v103, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[6:9], v103, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[94:97], v[10:13], v[74:77], v102, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[14:17], v[10:13], v102, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[10:13], v103, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[14:17], v103, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[94:97], v[18:21], v[78:81], v102, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[22:25], v[18:21], v102, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[18:21], v103, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[22:25], v103, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v98, v111, s[8:11], s27 offen offset:3072
	buffer_load_dword v99, v111, s[8:11], s28 offen offset:3072
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s31, 0x1a00
	s_movk_i32 s33, 0x1b00
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84
	ds_write_b32 v115, v82 offset:9216
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s31 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s33 offen
	ds_read_b128 v[90:93], v118 offset:4096
	ds_read_b128 v[94:97], v119 offset:4096
	ds_read_b32 v102, v111 offset:8960
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[94:97], v[26:29], v[66:69], v102, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[30:33], v[26:29], v102, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v103, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v103, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[34:37], v[70:73], v102, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[38:41], v[34:37], v102, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[34:37], v103, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[38:41], v103, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[94:97], v[42:45], v[74:77], v102, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[46:49], v[42:45], v102, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[42:45], v103, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v103, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[94:97], v[50:53], v[78:81], v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[54:57], v[50:53], v102, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[50:53], v103, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v103, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v100, v111, s[8:11], s27 offen offset:3328
	buffer_load_dword v101, v111, s[8:11], s28 offen offset:3328
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s31, 0x1d00
	s_nop 0
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92 offset:2048
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84 offset:2048
	ds_write_b32 v115, v82 offset:9472
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s29 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s31 offen
	ds_read_b128 v[90:93], v118
	ds_read_b128 v[94:97], v119
	ds_read_b32 v102, v111 offset:9216
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[94:97], v[58:61], v[66:69], v102, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[62:65], v[58:61], v102, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v103, 0x7000, v120
	s_nop 4
	buffer_load_dwordx4 v[58:61], v103, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[62:65], v103, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[94:97], v[2:5], v[70:73], v102, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[6:9], v[2:5], v102, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v103, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[6:9], v103, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[94:97], v[10:13], v[74:77], v102, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[14:17], v[10:13], v102, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[10:13], v103, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[14:17], v103, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[94:97], v[18:21], v[78:81], v102, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[22:25], v[18:21], v102, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[18:21], v103, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[22:25], v103, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v98, v111, s[8:11], s27 offen offset:3584
	buffer_load_dword v99, v111, s[8:11], s28 offen offset:3584
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s29, 0x1e00
	s_movk_i32 s31, 0x1f00
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92 offset:4096
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84 offset:4096
	ds_write_b32 v115, v82 offset:9728
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s29 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s31 offen
	ds_read_b128 v[90:93], v118 offset:2048
	ds_read_b128 v[94:97], v119 offset:2048
	ds_read_b32 v102, v111 offset:9472
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[94:97], v[26:29], v[66:69], v102, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[30:33], v[26:29], v102, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v103, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v103, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[34:37], v[70:73], v102, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[38:41], v[34:37], v102, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[34:37], v103, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[38:41], v103, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[94:97], v[42:45], v[74:77], v102, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[46:49], v[42:45], v102, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[42:45], v103, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v103, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[94:97], v[50:53], v[78:81], v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[54:57], v[50:53], v102, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[50:53], v103, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v103, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v100, v111, s[8:11], s27 offen offset:3840
	buffer_load_dword v101, v111, s[8:11], s28 offen offset:3840
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s27, 0x2100
	s_nop 0
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84
	ds_write_b32 v115, v82 offset:9984
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s30 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s27 offen
	ds_read_b128 v[90:93], v118 offset:4096
	ds_read_b128 v[94:97], v119 offset:4096
	ds_read_b32 v102, v111 offset:9728
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[94:97], v[58:61], v[66:69], v102, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[62:65], v[58:61], v102, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v103, 0x8000, v120
	s_nop 4
	buffer_load_dwordx4 v[58:61], v103, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[62:65], v103, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[94:97], v[2:5], v[70:73], v102, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[6:9], v[2:5], v102, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v103, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[6:9], v103, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[94:97], v[10:13], v[74:77], v102, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[14:17], v[10:13], v102, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[10:13], v103, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[14:17], v103, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[94:97], v[18:21], v[78:81], v102, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[22:25], v[18:21], v102, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[18:21], v103, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[22:25], v103, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v98, v111, s[8:11], s25 offen
	buffer_load_dword v99, v111, s[8:11], s24 offen
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s27, 0x2200
	s_movk_i32 s28, 0x2300
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92 offset:2048
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84 offset:2048
	ds_write_b32 v115, v82 offset:10240
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s27 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s28 offen
	ds_read_b128 v[90:93], v118
	ds_read_b128 v[94:97], v119
	ds_read_b32 v102, v111 offset:9984
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[94:97], v[26:29], v[66:69], v102, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[30:33], v[26:29], v102, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v103, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v103, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[34:37], v[70:73], v102, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[38:41], v[34:37], v102, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[34:37], v103, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[38:41], v103, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[94:97], v[42:45], v[74:77], v102, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[46:49], v[42:45], v102, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[42:45], v103, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v103, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[94:97], v[50:53], v[78:81], v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[54:57], v[50:53], v102, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[50:53], v103, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v103, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v100, v111, s[8:11], s25 offen offset:256
	buffer_load_dword v101, v111, s[8:11], s24 offen offset:256
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s27, 0x2400
	s_movk_i32 s28, 0x2500
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92 offset:4096
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84 offset:4096
	ds_write_b32 v115, v82 offset:10496
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s27 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s28 offen
	ds_read_b128 v[90:93], v118 offset:2048
	ds_read_b128 v[94:97], v119 offset:2048
	ds_read_b32 v102, v111 offset:10240
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[94:97], v[58:61], v[66:69], v102, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[62:65], v[58:61], v102, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v103, 0x9000, v120
	s_nop 4
	buffer_load_dwordx4 v[58:61], v103, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[62:65], v103, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[94:97], v[2:5], v[70:73], v102, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[6:9], v[2:5], v102, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v103, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[6:9], v103, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[94:97], v[10:13], v[74:77], v102, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[14:17], v[10:13], v102, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[10:13], v103, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[14:17], v103, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[94:97], v[18:21], v[78:81], v102, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[22:25], v[18:21], v102, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[18:21], v103, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[22:25], v103, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v98, v111, s[8:11], s25 offen offset:512
	buffer_load_dword v99, v111, s[8:11], s24 offen offset:512
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s27, 0x2600
	s_movk_i32 s28, 0x2700
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84
	ds_write_b32 v115, v82 offset:10752
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s27 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s28 offen
	ds_read_b128 v[90:93], v118 offset:4096
	ds_read_b128 v[94:97], v119 offset:4096
	ds_read_b32 v102, v111 offset:10496
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[94:97], v[26:29], v[66:69], v102, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[30:33], v[26:29], v102, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v103, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v103, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[34:37], v[70:73], v102, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[38:41], v[34:37], v102, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[34:37], v103, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[38:41], v103, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[94:97], v[42:45], v[74:77], v102, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[46:49], v[42:45], v102, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[42:45], v103, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v103, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[94:97], v[50:53], v[78:81], v102, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[54:57], v[50:53], v102, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[50:53], v103, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v103, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v100, v111, s[8:11], s25 offen offset:768
	buffer_load_dword v101, v111, s[8:11], s24 offen offset:768
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s27, 0x2800
	s_movk_i32 s28, 0x2900
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92 offset:2048
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84 offset:2048
	ds_write_b32 v115, v82 offset:11008
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s27 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s28 offen
	ds_read_b128 v[90:93], v118
	ds_read_b128 v[94:97], v119
	ds_read_b32 v102, v111 offset:10752
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[94:97], v[58:61], v[66:69], v102, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[62:65], v[58:61], v102, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v103, 0xa000, v120
	s_nop 4
	buffer_load_dwordx4 v[58:61], v103, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[62:65], v103, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[94:97], v[2:5], v[70:73], v102, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[6:9], v[2:5], v102, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v103, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[6:9], v103, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[94:97], v[10:13], v[74:77], v102, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[14:17], v[10:13], v102, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[10:13], v103, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[14:17], v103, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[94:97], v[18:21], v[78:81], v102, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[22:25], v[18:21], v102, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[18:21], v103, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[22:25], v103, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v121, v111, s[8:11], s25 offen offset:1024
	buffer_load_dword v126, v111, s[8:11], s24 offen offset:1024
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s27, 0x2a00
	s_movk_i32 s28, 0x2b00
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92 offset:4096
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84 offset:4096
	ds_write_b32 v115, v82 offset:11264
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s27 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s28 offen
	ds_read_b128 v[90:93], v118 offset:2048
	ds_read_b128 v[94:97], v119 offset:2048
	ds_read_b32 v98, v111 offset:11008
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[94:97], v[26:29], v[66:69], v98, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[90:93], v[30:33], v[26:29], v98, v100 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[26:29], v103, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[30:33], v103, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[34:37], v[70:73], v98, v100 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[90:93], v[38:41], v[34:37], v98, v100 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[34:37], v103, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[38:41], v103, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[94:97], v[42:45], v[74:77], v98, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[90:93], v[46:49], v[42:45], v98, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[42:45], v103, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v103, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[94:97], v[50:53], v[78:81], v98, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[90:93], v[54:57], v[50:53], v98, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v92, 0x7fff7fff, v84
	v_and_b32_e32 v93, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v92, v93
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dwordx4 v[50:53], v103, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v103, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v134, v111, s[8:11], s25 offen offset:1280
	buffer_load_dword v135, v111, s[8:11], s24 offen offset:1280
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_movk_i32 s27, 0x2c00
	s_movk_i32 s28, 0x2d00
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	ds_write_b32 v116, v92
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v117, v84
	ds_write_b32 v115, v82 offset:11520
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s27 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s28 offen
	ds_read_b128 v[90:93], v118 offset:4096
	ds_read_b128 v[94:97], v119 offset:4096
	ds_read_b32 v127, v111 offset:11264
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[94:97], v[58:61], v[66:69], v127, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[90:93], v[62:65], v[58:61], v127, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v136, 0xb000, v120
	buffer_load_dwordx4 v[102:105], v136, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[106:109], v136, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[94:97], v[2:5], v[70:73], v127, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[90:93], v[6:9], v[2:5], v127, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[2:5], v136, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[6:9], v136, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[94:97], v[10:13], v[74:77], v127, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[90:93], v[14:17], v[10:13], v127, v126 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v136, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[62:65], v136, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[94:97], v[18:21], v[78:81], v127, v126 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[90:93], v[22:25], v[14:17], v127, v126 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v22, 0x7fff7fff, v82
	v_and_b32_e32 v23, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v22, v22, v23
	;;#ASMEND
	v_and_b32_e32 v24, 0x7fff7fff, v84
	v_and_b32_e32 v25, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v23, v24, v25
	;;#ASMEND
	v_mov_b32_e32 v24, 0
	;;#ASMSTART
	v_pk_max_u16 v22, v22, v23
	;;#ASMEND
	buffer_load_dwordx4 v[18:21], v136, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[66:69], v136, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v121, v111, s[8:11], s25 offen offset:1536
	buffer_load_dword v137, v111, s[8:11], s24 offen offset:1536
	v_max_u16_sdwa v22, v22, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v25, 0x7fff7fff, v88
	v_and_b32_e32 v70, 0x7fff7fff, v89
	v_max_u32_dpp v22, v22, v22 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s27, 0x2e00
	s_movk_i32 s28, 0x2f00
	v_max_u32_dpp v22, v22, v22 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v22, v22, 16, v113
	v_bfe_u32 v22, v22, 23, 8
	v_max_u32_e32 v22, 2, v22
	v_add_u32_e32 v22, 0xfe, v22
	v_and_b32_e32 v22, 0xff, v22
	v_lshlrev_b32_e32 v23, 23, v22
	v_cvt_scalef32_pk_fp4_bf16 v24, v82, v23
	v_cvt_scalef32_pk_fp4_bf16 v24, v83, v23 op_sel:[0,0,1,0]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v24, v84, v23 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v24, v85, v23 op_sel:[0,0,1,1]
	v_and_b32_e32 v23, 0x7fff7fff, v86
	ds_write_b32 v116, v24 offset:2048
	v_and_b32_e32 v24, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v23, v23, v24
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v24, v25, v70
	;;#ASMEND
	v_mov_b32_e32 v25, 0
	;;#ASMSTART
	v_pk_max_u16 v23, v23, v24
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v23, v23, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v23, v23, v23 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v23, v23, v23 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v23, v23, 16, v113
	v_bfe_u32 v23, v23, 23, 8
	v_max_u32_e32 v23, 2, v23
	v_add_u32_e32 v23, 0xfe, v23
	v_and_b32_e32 v23, 0xff, v23
	v_lshlrev_b32_e32 v24, 23, v23
	v_cvt_scalef32_pk_fp4_bf16 v25, v86, v24
	v_cvt_scalef32_pk_fp4_bf16 v25, v87, v24 op_sel:[0,0,1,0]
	v_lshl_or_b32 v22, v23, 16, v22
	v_cvt_scalef32_pk_fp4_bf16 v25, v88, v24 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v25, v89, v24 op_sel:[0,0,1,1]
	ds_write_b32 v117, v25 offset:2048
	ds_write_b32 v115, v22 offset:11776
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[22:25], v112, s[12:15], s27 offen
	buffer_load_dwordx4 v[126:129], v112, s[12:15], s28 offen
	ds_read_b128 v[94:97], v118
	ds_read_b128 v[130:133], v119
	ds_read_b32 v138, v111 offset:11520
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[130:133], v[26:29], v[98:101], v138, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[94:97], v[30:33], v[26:29], v138, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v136, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v136, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[130:133], v[34:37], v[122:125], v138, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[94:97], v[38:41], v[30:33], v138, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[78:81], v136, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[82:85], v136, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[130:133], v[42:45], v[10:13], v138, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[46:49], v[10:13], v138, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[86:89], v136, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v136, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[130:133], v[50:53], v[14:17], v138, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[94:97], v[54:57], v[10:13], v138, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	s_nop 4
	v_and_b32_e32 v10, 0x7fff7fff, v22
	v_and_b32_e32 v11, 0x7fff7fff, v23
	;;#ASMSTART
	v_pk_max_u16 v10, v10, v11
	;;#ASMEND
	v_and_b32_e32 v12, 0x7fff7fff, v24
	v_and_b32_e32 v13, 0x7fff7fff, v25
	;;#ASMSTART
	v_pk_max_u16 v11, v12, v13
	;;#ASMEND
	v_mov_b32_e32 v12, 0
	;;#ASMSTART
	v_pk_max_u16 v10, v10, v11
	;;#ASMEND
	buffer_load_dwordx4 v[94:97], v136, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v136, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v134, v111, s[8:11], s25 offen offset:1792
	buffer_load_dword v135, v111, s[8:11], s24 offen offset:1792
	v_max_u16_sdwa v10, v10, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v13, 0x7fff7fff, v128
	v_and_b32_e32 v14, 0x7fff7fff, v129
	v_max_u32_dpp v10, v10, v10 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s27, 0x3100
	s_nop 0
	v_max_u32_dpp v10, v10, v10 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v10, v10, 16, v113
	v_bfe_u32 v10, v10, 23, 8
	v_max_u32_e32 v10, 2, v10
	v_add_u32_e32 v10, 0xfe, v10
	v_and_b32_e32 v10, 0xff, v10
	v_lshlrev_b32_e32 v11, 23, v10
	v_cvt_scalef32_pk_fp4_bf16 v12, v22, v11
	v_cvt_scalef32_pk_fp4_bf16 v12, v23, v11 op_sel:[0,0,1,0]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v12, v24, v11 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v12, v25, v11 op_sel:[0,0,1,1]
	v_and_b32_e32 v11, 0x7fff7fff, v126
	ds_write_b32 v116, v12 offset:4096
	v_and_b32_e32 v12, 0x7fff7fff, v127
	;;#ASMSTART
	v_pk_max_u16 v11, v11, v12
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v12, v13, v14
	;;#ASMEND
	v_mov_b32_e32 v13, 0
	;;#ASMSTART
	v_pk_max_u16 v11, v11, v12
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v11, v11, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v11, v11, v11 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v11, v11, v11 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v11, v11, 16, v113
	v_bfe_u32 v11, v11, 23, 8
	v_max_u32_e32 v11, 2, v11
	v_add_u32_e32 v11, 0xfe, v11
	v_and_b32_e32 v11, 0xff, v11
	v_lshlrev_b32_e32 v12, 23, v11
	v_cvt_scalef32_pk_fp4_bf16 v13, v126, v12
	v_cvt_scalef32_pk_fp4_bf16 v13, v127, v12 op_sel:[0,0,1,0]
	v_lshl_or_b32 v10, v11, 16, v10
	v_cvt_scalef32_pk_fp4_bf16 v13, v128, v12 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v13, v129, v12 op_sel:[0,0,1,1]
	ds_write_b32 v117, v13 offset:4096
	ds_write_b32 v115, v10 offset:12032
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[42:45], v112, s[12:15], s26 offen
	buffer_load_dwordx4 v[46:49], v112, s[12:15], s27 offen
	ds_read_b128 v[50:53], v118 offset:2048
	ds_read_b128 v[54:57], v119 offset:2048
	ds_read_b32 v126, v111 offset:11776
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[54:57], v[102:105], v[26:29], v126, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[50:53], v[106:109], v[10:13], v126, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v136, 0xc000, v120
	buffer_load_dwordx4 v[102:105], v136, s[4:7], s23 offen nt
	buffer_load_dwordx4 v[106:109], v136, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[54:57], v[2:5], v[30:33], v126, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[50:53], v[6:9], v[2:5], v126, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[10:13], v136, s[4:7], s22 offen nt
	buffer_load_dwordx4 v[14:17], v136, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[54:57], v[58:61], v[34:37], v126, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[50:53], v[62:65], v[6:9], v126, v137 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[22:25], v136, s[4:7], s21 offen nt
	buffer_load_dwordx4 v[26:29], v136, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[54:57], v[18:21], v[38:41], v126, v137 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[50:53], v[66:69], v[18:21], v126, v137 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v38, 0x7fff7fff, v42
	v_and_b32_e32 v39, 0x7fff7fff, v43
	;;#ASMSTART
	v_pk_max_u16 v38, v38, v39
	;;#ASMEND
	v_and_b32_e32 v40, 0x7fff7fff, v44
	v_and_b32_e32 v41, 0x7fff7fff, v45
	;;#ASMSTART
	v_pk_max_u16 v39, v40, v41
	;;#ASMEND
	v_mov_b32_e32 v40, 0
	;;#ASMSTART
	v_pk_max_u16 v38, v38, v39
	;;#ASMEND
	buffer_load_dwordx4 v[30:33], v136, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[34:37], v136, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v121, v111, s[8:11], s25 offen offset:2048
	buffer_load_dword v137, v111, s[8:11], s24 offen offset:2048
	v_max_u16_sdwa v38, v38, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v41, 0x7fff7fff, v48
	s_movk_i32 s26, 0x3200
	v_max_u32_dpp v38, v38, v38 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s27, 0x3300
	s_nop 0
	v_max_u32_dpp v38, v38, v38 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v38, v38, 16, v113
	v_bfe_u32 v38, v38, 23, 8
	v_max_u32_e32 v38, 2, v38
	v_add_u32_e32 v38, 0xfe, v38
	v_and_b32_e32 v38, 0xff, v38
	v_lshlrev_b32_e32 v39, 23, v38
	v_cvt_scalef32_pk_fp4_bf16 v40, v42, v39
	v_cvt_scalef32_pk_fp4_bf16 v40, v43, v39 op_sel:[0,0,1,0]
	v_and_b32_e32 v42, 0x7fff7fff, v49
	v_cvt_scalef32_pk_fp4_bf16 v40, v44, v39 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v40, v45, v39 op_sel:[0,0,1,1]
	v_and_b32_e32 v39, 0x7fff7fff, v46
	ds_write_b32 v116, v40
	v_and_b32_e32 v40, 0x7fff7fff, v47
	;;#ASMSTART
	v_pk_max_u16 v39, v39, v40
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v40, v41, v42
	;;#ASMEND
	v_mov_b32_e32 v41, 0
	;;#ASMSTART
	v_pk_max_u16 v39, v39, v40
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v39, v39, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v39, v39, v39 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v39, v39, v39 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v39, v39, 16, v113
	v_bfe_u32 v39, v39, 23, 8
	v_max_u32_e32 v39, 2, v39
	v_add_u32_e32 v39, 0xfe, v39
	v_and_b32_e32 v39, 0xff, v39
	v_lshlrev_b32_e32 v40, 23, v39
	v_cvt_scalef32_pk_fp4_bf16 v41, v46, v40
	v_cvt_scalef32_pk_fp4_bf16 v41, v47, v40 op_sel:[0,0,1,0]
	v_lshl_or_b32 v38, v39, 16, v38
	v_cvt_scalef32_pk_fp4_bf16 v41, v48, v40 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v41, v49, v40 op_sel:[0,0,1,1]
	ds_write_b32 v117, v41
	ds_write_b32 v115, v38 offset:12288
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[126:129], v112, s[12:15], s26 offen
	buffer_load_dwordx4 v[130:133], v112, s[12:15], s27 offen
	ds_read_b128 v[62:65], v118 offset:4096
	ds_read_b128 v[66:69], v119 offset:4096
	ds_read_b32 v138, v111 offset:12032
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[66:69], v[70:73], v[122:125], v138, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[62:65], v[74:77], v[38:41], v138, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 5
	buffer_load_dwordx4 v[38:41], v136, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[42:45], v136, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[66:69], v[78:81], v[2:5], v138, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[62:65], v[82:85], v[2:5], v138, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v136, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[50:53], v136, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[66:69], v[86:89], v[6:9], v138, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[62:65], v[90:93], v[2:5], v138, v135 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v136, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[58:61], v136, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[66:69], v[94:97], v[18:21], v138, v135 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], v[62:65], v[98:101], v[2:5], v138, v135 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	s_nop 4
	v_and_b32_e32 v2, 0x7fff7fff, v126
	v_and_b32_e32 v3, 0x7fff7fff, v127
	;;#ASMSTART
	v_pk_max_u16 v2, v2, v3
	;;#ASMEND
	v_and_b32_e32 v4, 0x7fff7fff, v128
	v_and_b32_e32 v5, 0x7fff7fff, v129
	;;#ASMSTART
	v_pk_max_u16 v3, v4, v5
	;;#ASMEND
	v_mov_b32_e32 v4, 0
	;;#ASMSTART
	v_pk_max_u16 v2, v2, v3
	;;#ASMEND
	buffer_load_dwordx4 v[62:65], v136, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[66:69], v136, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v111, s[8:11], s25 offen offset:2304
	buffer_load_dword v123, v111, s[8:11], s24 offen offset:2304
	v_max_u16_sdwa v2, v2, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v5, 0x7fff7fff, v132
	v_and_b32_e32 v6, 0x7fff7fff, v133
	v_max_u32_dpp v2, v2, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s26, 0x3400
	s_movk_i32 s27, 0x3500
	v_max_u32_dpp v2, v2, v2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v2, v2, 16, v113
	v_bfe_u32 v2, v2, 23, 8
	v_max_u32_e32 v2, 2, v2
	v_add_u32_e32 v2, 0xfe, v2
	v_and_b32_e32 v2, 0xff, v2
	v_lshlrev_b32_e32 v3, 23, v2
	v_cvt_scalef32_pk_fp4_bf16 v4, v126, v3
	v_cvt_scalef32_pk_fp4_bf16 v4, v127, v3 op_sel:[0,0,1,0]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v4, v128, v3 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v4, v129, v3 op_sel:[0,0,1,1]
	v_and_b32_e32 v3, 0x7fff7fff, v130
	ds_write_b32 v116, v4 offset:2048
	v_and_b32_e32 v4, 0x7fff7fff, v131
	;;#ASMSTART
	v_pk_max_u16 v3, v3, v4
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v4, v5, v6
	;;#ASMEND
	v_mov_b32_e32 v5, 0
	;;#ASMSTART
	v_pk_max_u16 v3, v3, v4
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v3, v3, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v3, v3, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v3, v3, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v3, v3, 16, v113
	v_bfe_u32 v3, v3, 23, 8
	v_max_u32_e32 v3, 2, v3
	v_add_u32_e32 v3, 0xfe, v3
	v_and_b32_e32 v3, 0xff, v3
	v_lshlrev_b32_e32 v4, 23, v3
	v_cvt_scalef32_pk_fp4_bf16 v5, v130, v4
	v_cvt_scalef32_pk_fp4_bf16 v5, v131, v4 op_sel:[0,0,1,0]
	v_lshl_or_b32 v2, v3, 16, v2
	v_cvt_scalef32_pk_fp4_bf16 v5, v132, v4 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v5, v133, v4 op_sel:[0,0,1,1]
	ds_write_b32 v117, v5 offset:2048
	ds_write_b32 v115, v2 offset:12544
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s26 offen
	buffer_load_dwordx4 v[90:93], v112, s[12:15], s27 offen
	ds_read_b128 v[94:97], v118
	ds_read_b128 v[98:101], v119
	ds_read_b32 v124, v111 offset:12288
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[98:101], v[102:105], v[70:73], v124, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], v[94:97], v[106:109], v[2:5], v124, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v102, 0xd000, v120
	buffer_load_dwordx4 v[6:9], v102, s[4:7], s23 offen nt
	s_nop 3
	buffer_load_dwordx4 v[2:5], v102, s[4:7], s23 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[98:101], v[10:13], v[74:77], v124, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[94:97], v[14:17], v[10:13], v124, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[18:21], v102, s[4:7], s22 offen nt
	s_nop 4
	buffer_load_dwordx4 v[10:13], v102, s[4:7], s22 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[98:101], v[22:25], v[78:81], v124, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], v[94:97], v[26:29], v[14:17], v124, v137 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[26:29], v102, s[4:7], s21 offen nt
	s_nop 4
	buffer_load_dwordx4 v[14:17], v102, s[4:7], s21 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[98:101], v[30:33], v[82:85], v124, v137 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[34:37], v[22:25], v124, v137 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_and_b32_e32 v83, 0x7fff7fff, v87
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_and_b32_e32 v84, 0x7fff7fff, v88
	v_and_b32_e32 v85, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v84, v85
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	buffer_load_dwordx4 v[30:33], v102, s[4:7], s20 offen nt
	buffer_load_dwordx4 v[22:25], v102, s[4:7], s20 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v98, v111, s[8:11], s25 offen offset:2560
	buffer_load_dword v99, v111, s[8:11], s24 offen offset:2560
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_waitcnt vmcnt(10)
	v_and_b32_e32 v85, 0x7fff7fff, v92
	s_movk_i32 s26, 0x3600
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_movk_i32 s27, 0x3700
	s_nop 0
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_and_b32_e32 v86, 0x7fff7fff, v93
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	v_and_b32_e32 v83, 0x7fff7fff, v90
	ds_write_b32 v116, v84 offset:4096
	v_and_b32_e32 v84, 0x7fff7fff, v91
	;;#ASMSTART
	v_pk_max_u16 v83, v83, v84
	;;#ASMEND
	;;#ASMSTART
	v_pk_max_u16 v84, v85, v86
	;;#ASMEND
	v_mov_b32_e32 v85, 0
	;;#ASMSTART
	v_pk_max_u16 v83, v83, v84
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v83, v83, v83 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v83, v83, v83 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v83, v83, v83 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v83, v83, 16, v113
	v_bfe_u32 v83, v83, 23, 8
	v_max_u32_e32 v83, 2, v83
	v_add_u32_e32 v83, 0xfe, v83
	v_and_b32_e32 v83, 0xff, v83
	v_lshlrev_b32_e32 v84, 23, v83
	v_cvt_scalef32_pk_fp4_bf16 v85, v90, v84
	v_cvt_scalef32_pk_fp4_bf16 v85, v91, v84 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v83, 16, v82
	v_cvt_scalef32_pk_fp4_bf16 v85, v92, v84 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v85, v93, v84 op_sel:[0,0,1,1]
	ds_write_b32 v117, v85 offset:4096
	ds_write_b32 v115, v82 offset:12800
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[82:85], v112, s[12:15], s26 offen
	buffer_load_dwordx4 v[86:89], v112, s[12:15], s27 offen
	ds_read_b128 v[90:93], v118 offset:2048
	ds_read_b128 v[94:97], v119 offset:2048
	ds_read_b32 v100, v111 offset:12544
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[94:97], v[38:41], v[70:73], v100, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[90:93], v[42:45], v[38:41], v100, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v102, s[4:7], s23 offen offset:2048 nt
	buffer_load_dwordx4 v[70:73], v102, s[4:7], s23 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[94:97], v[46:49], v[74:77], v100, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], v[90:93], v[50:53], v[46:49], v100, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v102, s[4:7], s22 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v102, s[4:7], s22 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[94:97], v[54:57], v[78:81], v100, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[90:93], v[58:61], v[54:57], v100, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v102, s[4:7], s21 offen offset:2048 nt
	buffer_load_dwordx4 v[78:81], v102, s[4:7], s21 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[94:97], v[62:65], v[34:37], v100, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[90:93], v[66:69], v[34:37], v100, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v102, s[4:7], s20 offen offset:2048 nt
	buffer_load_dwordx4 v[66:69], v102, s[4:7], s20 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(9)
	v_and_b32_e32 v90, 0x7fff7fff, v82
	v_and_b32_e32 v91, 0x7fff7fff, v83
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	v_and_b32_e32 v91, 0x7fff7fff, v84
	v_and_b32_e32 v92, 0x7fff7fff, v85
	;;#ASMSTART
	v_pk_max_u16 v91, v91, v92
	;;#ASMEND
	v_mov_b32_e32 v92, 0
	;;#ASMSTART
	v_pk_max_u16 v90, v90, v91
	;;#ASMEND
	buffer_load_dword v93, v111, s[8:11], s25 offen offset:2816
	v_max_u16_sdwa v90, v90, v90 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_lshl_b32 s4, s19, 5
	v_lshl_or_b32 v1, v1, 10, v110
	v_max_u32_dpp v90, v90, v90 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_lshl_u32 v1, v1, s4, 2
	s_lshl_b32 s4, s18, 6
	v_max_u32_dpp v90, v90, v90 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v90, v90, 16, v113
	v_bfe_u32 v90, v90, 23, 8
	v_max_u32_e32 v90, 2, v90
	v_add_u32_e32 v90, 0xfe, v90
	v_and_b32_e32 v90, 0xff, v90
	v_lshlrev_b32_e32 v91, 23, v90
	v_cvt_scalef32_pk_fp4_bf16 v92, v82, v91
	v_cvt_scalef32_pk_fp4_bf16 v92, v83, v91 op_sel:[0,0,1,0]
	s_waitcnt vmcnt(9)
	v_and_b32_e32 v82, 0x7fff7fff, v86
	v_cvt_scalef32_pk_fp4_bf16 v92, v84, v91 op_sel:[0,0,0,1]
	v_and_b32_e32 v83, 0x7fff7fff, v87
	v_cvt_scalef32_pk_fp4_bf16 v92, v85, v91 op_sel:[0,0,1,1]
	buffer_load_dword v91, v111, s[8:11], s24 offen offset:2816
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	v_and_b32_e32 v83, 0x7fff7fff, v88
	v_and_b32_e32 v84, 0x7fff7fff, v89
	;;#ASMSTART
	v_pk_max_u16 v83, v83, v84
	;;#ASMEND
	v_mov_b32_e32 v84, 0
	;;#ASMSTART
	v_pk_max_u16 v82, v82, v83
	;;#ASMEND
	s_nop 0
	v_max_u16_sdwa v82, v82, v82 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_nop 1
	v_max_u32_dpp v82, v82, v82 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshl_add_u32 v82, v82, 16, v113
	v_bfe_u32 v82, v82, 23, 8
	v_max_u32_e32 v82, 2, v82
	v_add_u32_e32 v82, 0xfe, v82
	v_and_b32_e32 v82, 0xff, v82
	v_lshlrev_b32_e32 v83, 23, v82
	v_cvt_scalef32_pk_fp4_bf16 v84, v86, v83
	v_cvt_scalef32_pk_fp4_bf16 v84, v87, v83 op_sel:[0,0,1,0]
	v_lshl_or_b32 v82, v82, 16, v90
	v_cvt_scalef32_pk_fp4_bf16 v84, v88, v83 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_bf16 v84, v89, v83 op_sel:[0,0,1,1]
	ds_write_b32 v116, v92
	ds_write_b32 v117, v84
	ds_write_b32 v115, v82 offset:13056
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[82:85], v119 offset:4096
	ds_read_b32 v86, v111 offset:12800
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[30:33], v[34:37], v86, v99 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	ds_read_b128 v[34:37], v118 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[82:85], v[6:9], v[38:41], v86, v98 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[18:21], v[46:49], v86, v98 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[26:29], v[54:57], v86, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[34:37], v[2:5], v[6:9], v86, v98 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[34:37], v[10:13], v[18:21], v86, v98 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[34:37], v[14:17], v[26:29], v86, v99 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[14:17], v119
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[34:37], v[22:25], v[30:33], v86, v99 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b32 v22, v111 offset:13056
	v_add_u32_e32 v23, 0x400, v1
	v_add_u32_e32 v24, 0x800, v1
	s_waitcnt vmcnt(1) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[14:17], v[42:45], v[2:5], v22, v93 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v25, 0xc00, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[14:17], v[50:53], v[6:9], v22, v93 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[14:17], v[58:61], v[10:13], v22, v91 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[14:17], v[62:65], v[18:21], v22, v91 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	ds_read_b128 v[18:21], v118
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], v[18:21], v[70:73], v[2:5], v22, v93 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[18:21], v[78:81], v[10:13], v22, v91 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[18:21], v[74:77], v[6:9], v22, v93 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[18:21], v[66:69], v[14:17], v22, v91 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 5
	ds_write2_b32 v1, v2, v10 offset1:16
	ds_write2_b32 v23, v3, v11 offset1:16
	ds_write2_b32 v24, v4, v12 offset1:16
	ds_write2_b32 v25, v5, v13 offset1:16
	ds_write2_b32 v1, v6, v14 offset0:128 offset1:144
	ds_write2_b32 v23, v7, v15 offset0:128 offset1:144
	ds_write2_b32 v24, v8, v16 offset0:128 offset1:144
	ds_write2_b32 v25, v9, v17 offset0:128 offset1:144
	v_lshrrev_b32_e32 v1, 4, v0
	v_bfe_u32 v18, v0, 2, 2
	v_and_b32_e32 v19, 3, v0
	v_lshlrev_b32_e32 v0, 10, v1
	v_lshl_or_b32 v0, v19, 5, v0
	v_lshl_or_b32 v0, v18, 7, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[2:3], v0 offset1:1
	ds_read2_b32 v[4:5], v0 offset0:2 offset1:3
	ds_read2_b32 v[6:7], v0 offset0:4 offset1:5
	ds_read2_b32 v[8:9], v0 offset0:6 offset1:7
	v_cmp_eq_u32_e32 vcc, 0, v19
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v10, 0xbfb8aa3b, v2
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v11, 0xbfb8aa3b, v3
	v_exp_f32_e32 v11, v11
	v_add_f32_e32 v10, 1.0, v10
	v_rcp_f32_e32 v20, v10
	v_add_f32_e32 v10, 1.0, v11
	v_rcp_f32_e32 v21, v10
	ds_read2_b32 v[10:11], v0 offset0:128 offset1:129
	ds_read2_b32 v[12:13], v0 offset0:130 offset1:131
	ds_read2_b32 v[14:15], v0 offset0:132 offset1:133
	ds_read2_b32 v[16:17], v0 offset0:134 offset1:135
	v_mul_f32_e32 v0, v2, v20
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v2, v10, v0
	v_mul_f32_e32 v0, v3, v21
	v_mul_f32_e32 v3, 0xbfb8aa3b, v4
	v_exp_f32_e32 v3, v3
	v_mul_f32_e32 v10, 0xbfb8aa3b, v5
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v11, v11, v0
	v_add_f32_e32 v0, 1.0, v3
	v_rcp_f32_e32 v0, v0
	v_add_f32_e32 v3, 1.0, v10
	v_mul_f32_e32 v10, 0xbfb8aa3b, v6
	v_rcp_f32_e32 v3, v3
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v0, v4, v0
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v4, v12, v0
	v_mul_f32_e32 v0, v5, v3
	v_add_f32_e32 v3, 1.0, v10
	v_rcp_f32_e32 v3, v3
	v_mul_f32_e32 v5, 0xbfb8aa3b, v7
	v_exp_f32_e32 v5, v5
	v_mul_f32_e32 v10, v13, v0
	v_mul_f32_e32 v0, v6, v3
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v3, v14, v0
	v_add_f32_e32 v0, 1.0, v5
	v_mul_f32_e32 v5, 0xbfb8aa3b, v8
	v_exp_f32_e32 v5, v5
	v_mul_f32_e32 v6, 0xbfb8aa3b, v9
	v_exp_f32_e32 v6, v6
	v_rcp_f32_e32 v0, v0
	v_add_f32_e32 v5, 1.0, v5
	v_rcp_f32_e32 v5, v5
	v_add_f32_e32 v6, 1.0, v6
	v_rcp_f32_e32 v6, v6
	v_mul_f32_e32 v0, v7, v0
	v_mul_f32_e32 v7, v15, v0
	v_mul_f32_e32 v0, v8, v5
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v5, v16, v0
	v_mul_f32_e32 v0, v9, v6
	v_mul_f32_e32 v6, v17, v0
	v_max_f32_e64 v0, |v2|, |v11|
	v_max3_f32 v0, v0, |v4|, |v10|
	v_max3_f32 v0, v0, |v3|, |v7|
	v_max3_f32 v0, v0, |v5|, |v6|
	s_nop 1
	v_mov_b32_dpp v8, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v8, v8, v8
	v_max_f32_e32 v0, v0, v8
	s_nop 1
	v_mov_b32_dpp v8, v0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v8, v8, v8
	v_max_f32_e32 v0, v0, v8
	v_add_u32_e32 v0, 0x200000, v0
	v_mul_f32_e32 v0, 0x3e800000, v0
	v_cvt_scalef32_pk_fp4_f32 v114, v2, v11, v0
	v_cvt_scalef32_pk_fp4_f32 v114, v4, v10, v0 op_sel:[0,0,1,0]
	v_or_b32_e32 v4, s3, v1
	v_cvt_scalef32_pk_fp4_f32 v114, v3, v7, v0 op_sel:[0,0,0,1]
	v_lshlrev_b32_e32 v2, 4, v18
	v_lshlrev_b32_e32 v3, 2, v19
	v_lshl_add_u32 v4, v4, 8, s4
	v_or3_b32 v4, v4, v3, v2
	v_cvt_scalef32_pk_fp4_f32 v114, v5, v6, v0 op_sel:[0,0,1,1]
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], s[16:17], 0, v[4:5]
	global_store_dword v[4:5], v114, off nt
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_3
; %bb.2:
	s_lshl_b32 s3, s18, 5
	s_load_dwordx2 s[0:1], s[0:1], 0x48
	s_lshl_b32 s2, s2, 7
	s_and_b32 s3, s3, 0x3fffffc0
	v_lshrrev_b32_e32 v0, 23, v0
	s_add_i32 s3, s3, s2
	s_lshl_b32 s2, s18, 1
	v_min_u32_e32 v3, 0xfe, v0
	v_or3_b32 v0, s3, v1, v2
	s_and_b32 s2, s2, 2
	v_lshl_or_b32 v0, v0, 2, s2
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], s[0:1], 0, v[0:1]
	global_store_byte v[0:1], v3, off
.LBB0_3:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
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
		.amdhsa_next_free_vgpr 139
		.amdhsa_next_free_sgpr 35
		.amdhsa_accum_offset 140
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
	.section	.text,"axG",@progbits,_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,comdat,unique,1
.Lfunc_end0:
	.size	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16, .Lfunc_end0-_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
                                        ; -- End function
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.num_vgpr, 139
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.num_agpr, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.numbered_sgpr, 35
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.private_seg_size, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.uses_vcc, 1
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.uses_flat_scratch, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_dyn_sized_stack, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_recursion, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 21768
; TotalNumSgprs: 41
; NumVgprs: 139
; NumAgprs: 0
; TotalNumVgprs: 139
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 17
; NumSGPRsForWavesPerEU: 41
; NumVGPRsForWavesPerEU: 139
; AccumOffset: 140
; Occupancy: 3
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 34
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_aa9ad2e6e69c225a,@object ; @__hip_cuid_aa9ad2e6e69c225a
	.section	.bss,"aw",@nobits,unique,2
	.globl	__hip_cuid_aa9ad2e6e69c225a
__hip_cuid_aa9ad2e6e69c225a:
	.byte	0                               ; 0x0
	.size	__hip_cuid_aa9ad2e6e69c225a, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_aa9ad2e6e69c225a
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
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
    .name:           _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
    .private_segment_fixed_size: 0
    .sgpr_count:     41
    .sgpr_spill_count: 0
    .symbol:         _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi16ELb1ELb1ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.kd
    .uses_dynamic_stack: false
    .vgpr_count:     139
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
