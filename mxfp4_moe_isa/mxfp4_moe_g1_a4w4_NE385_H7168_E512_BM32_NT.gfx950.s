	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text,"axG",@progbits,_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,comdat,unique,1
	.protected	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16 ; -- Begin function _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
	.globl	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
	.p2align	8
	.type	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,@function
_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16: ; @_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
; %bb.4:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
; %bb.5:
.LBB0_0:
	s_mov_b64 s[24:25], s[2:3]
	s_load_dword s2, s[12:13], 0x0
	v_readfirstlane_b32 s17, v0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 27
	s_add_i32 s2, s2, s3
	s_ashr_i32 s2, s2, 5
	s_lshl_b32 s2, s2, 2
	s_cmp_ge_i32 s16, s2
	s_cbranch_scc1 .LBB0_3
; %bb.1:
	s_ashr_i32 s2, s16, 31
	s_lshr_b32 s2, s2, 30
	s_add_i32 s3, s16, s2
	s_ashr_i32 s2, s3, 2
	s_and_b32 s3, s3, -4
	s_mov_b64 s[20:21], s[6:7]
	s_sub_i32 s12, s16, s3
	s_ashr_i32 s3, s2, 31
	s_and_b32 s21, s21, 0xffff
	s_and_b32 s9, s9, 0xffff
	s_and_b32 s5, s5, 0xffff
	s_and_b32 s25, s25, 0xffff
	s_lshr_b32 s13, s17, 6
	s_lshl_b64 s[6:7], s[2:3], 2
	s_add_u32 s18, s10, s6
	s_addc_u32 s19, s11, s7
	s_lshl_b32 s3, s2, 5
	v_bfe_u32 v4, v0, 3, 3
	s_lshl_b32 s16, s13, 3
	v_or_b32_e32 v1, s3, v4
	v_add_u32_e32 v2, s16, v1
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[14:15]
	global_load_dword v2, v[2:3], off
	s_load_dword s14, s[0:1], 0x38
	s_lshl_b32 s29, s12, 3
	s_lshl_b32 s30, s13, 1
	s_load_dword s18, s[18:19], 0x0
	s_add_i32 s19, s30, s29
	s_ashr_i32 s29, s3, 31
	v_or_b32_e32 v4, s16, v4
	s_lshr_b32 s16, s29, 27
	s_movk_i32 s15, 0xffc0
	v_mov_b32_e32 v5, s17
	s_add_i32 s16, s3, s16
	s_mov_b32 s23, 0x20000
	v_bfi_b32 v5, s15, v5, v0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s26, s14, 0xe00
	s_lshl_b32 s14, s13, 10
	s_lshr_b32 s16, s16, 5
	s_mov_b32 s6, 0x8c00000
	s_mov_b32 s7, s23
	v_lshlrev_b32_e32 v9, 4, v5
	s_lshl_b32 s33, s13, 8
	s_add_i32 m0, s14, 0x3000
	s_mulk_i32 s16, 0x1c00
	v_lshlrev_b32_e32 v5, 2, v5
	s_lshl_b32 s15, s12, 8
	s_lshl_b32 s29, s18, 10
	buffer_load_dwordx4 v9, s[4:7], s16 offen lds
	s_add_i32 s30, s16, 0x1000
	s_add_i32 m0, s33, 0x4000
	s_movk_i32 s28, 0xe00
	v_lshlrev_b32_e32 v6, 4, v0
	v_and_b32_e32 v3, 15, v0
	s_andn2_b32 s17, s17, 63
	s_mulk_i32 s19, 0x700
	v_lshlrev_b32_e32 v4, 3, v4
	s_mul_i32 s18, s18, 0xe000
	s_add_i32 s34, s16, 0x1400
	s_add_i32 s15, s29, s15
	buffer_load_dword v5, s[4:7], s30 offen lds
	s_add_i32 m0, s33, 0x4400
	s_movk_i32 s31, 0x70
	v_bfe_u32 v1, v0, 4, 2
	v_lshlrev_b32_e32 v7, 4, v3
	v_xor_b32_e32 v4, v4, v6
	s_add_i32 s35, s16, 0x1800
	s_add_i32 s18, s18, s19
	s_add_i32 s15, s15, s17
	buffer_load_dword v5, s[4:7], s34 offen lds
	s_add_i32 m0, s33, 0x4800
	s_mov_b32 s22, 0x54380000
	s_mov_b32 s27, s23
	v_lshl_or_b32 v8, v1, 8, v7
	s_lshl_b32 s29, s18, 2
	s_mul_i32 s16, s15, 0xe00
	s_or_b32 s17, s15, 16
	s_or_b32 s18, s15, 32
	buffer_load_dword v5, s[4:7], s35 offen lds
	s_mov_b32 m0, s14
	s_or_b32 s19, s15, 48
	s_mul_i32 s15, s17, 0xe00
	s_mul_i32 s7, s18, 0xe00
	buffer_load_dwordx4 v[10:13], v8, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[14:17], v8, s[20:23], s16 offen offset:1024 nt
	buffer_load_dwordx4 v[18:21], v8, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[22:25], v8, s[20:23], s15 offen offset:1024 nt
	s_movk_i32 s4, 0x80
	s_mov_b32 s10, 0x5438000
	s_mov_b32 s11, s23
	s_mul_i32 s6, s19, 0xe00
	s_add_i32 s30, s29, 0x1c00
	v_lshlrev_b32_e32 v5, 3, v0
	s_add_i32 s19, s14, 0x2000
	v_and_b32_e32 v5, 0x70, v5
	v_lshlrev_b32_e32 v78, 7, v3
	s_add_i32 s17, s29, 0x1000
	s_add_i32 s18, s30, 0x1000
	s_waitcnt vmcnt(8)
	v_mul_lo_u32 v2, v2, s28
	v_and_or_b32 v7, v4, s31, v2
	s_add_i32 s28, s14, 0x1000
	buffer_load_dwordx4 v7, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v2, 2, v3
	s_mov_b32 m0, s28
	buffer_load_dwordx4 v[26:29], v8, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[30:33], v8, s[20:23], s7 offen offset:1024 nt
	buffer_load_dwordx4 v[34:37], v8, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[38:41], v8, s[20:23], s6 offen offset:1024 nt
	v_lshl_or_b32 v4, v1, 6, v2
	buffer_load_dwordx4 v7, s[24:27], s4 offen lds
	buffer_load_dword v9, v4, s[8:11], s29 offen
	buffer_load_dwordx4 v[42:45], v8, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[46:49], v8, s[20:23], s16 offen offset:3072 nt
	buffer_load_dwordx4 v[50:53], v8, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[54:57], v8, s[20:23], s15 offen offset:3072 nt
	buffer_load_dwordx4 v[58:61], v8, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[62:65], v8, s[20:23], s7 offen offset:3072 nt
	buffer_load_dwordx4 v[66:69], v8, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[70:73], v8, s[20:23], s6 offen offset:3072 nt
	buffer_load_dword v122, v4, s[8:11], s30 offen offset:256
	buffer_load_dword v114, v4, s[8:11], s30 offen
	buffer_load_dword v123, v4, s[8:11], s29 offen offset:256
	v_and_b32_e32 v2, 48, v0
	s_movk_i32 s4, 0x100
	s_mov_b32 m0, s19
	v_bitop3_b32 v6, v78, v5, v2 bitop3:0xf6
	s_barrier
	ds_read_b32 v115, v4 offset:12288
	ds_read_b128 v[74:77], v6
	buffer_load_dwordx4 v7, s[24:27], s4 offen lds
	v_or_b32_e32 v2, 64, v2
	v_bitop3_b32 v5, v78, v2, v5 bitop3:0xf6
	ds_read_b128 v[78:81], v6 offset:2048
	ds_read_b128 v[82:85], v5
	ds_read_b128 v[86:89], v5 offset:2048
	s_load_dwordx2 s[4:5], s[0:1], 0x40
	s_movk_i32 s31, 0x700
	v_mov_b32_e32 v2, 0
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(12) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[74:77], v[10:13], 0, v115, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[78:81], v[10:13], 0, v115, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[82:85], v[14:17], v[90:93], v115, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[14:17], v[10:13], v115, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v124, 0x1000, v8
	buffer_load_dwordx4 v[14:17], v124, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v124, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[74:77], v[18:21], 0, v115, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[78:81], v[18:21], 0, v115, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[82:85], v[22:25], v[98:101], v115, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[22:25], v[18:21], v115, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[22:25], v124, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v124, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[74:77], v[26:29], 0, v115, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[78:81], v[26:29], 0, v115, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[82:85], v[30:33], v[106:109], v115, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[30:33], v[26:29], v115, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v124, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v124, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[74:77], v[34:37], 0, v115, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[78:81], v[34:37], 0, v115, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[82:85], v[38:41], v[74:77], v115, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[38:41], v[34:37], v115, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v124, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v124, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v9, v4, s[8:11], s29 offen offset:512
	buffer_load_dword v125, v4, s[8:11], s30 offen offset:512
	s_movk_i32 s33, 0x180
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v126, v4 offset:12544
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(12) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[118:121], v[42:45], v[90:93], v126, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[42:45], v[10:13], v126, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[86:89], v[46:49], v[90:93], v126, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[46:49], v[10:13], v126, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v124, s[20:23], s16 offen offset:2048 nt
	s_nop 2
	buffer_load_dwordx4 v[90:93], v124, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[118:121], v[50:53], v[98:101], v126, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[50:53], v[18:21], v126, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[86:89], v[54:57], v[98:101], v126, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[54:57], v[18:21], v126, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v124, s[20:23], s15 offen offset:2048 nt
	s_nop 2
	buffer_load_dwordx4 v[98:101], v124, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[118:121], v[58:61], v[106:109], v126, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[58:61], v[26:29], v126, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[86:89], v[62:65], v[106:109], v126, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[62:65], v[26:29], v126, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v124, s[20:23], s7 offen offset:2048 nt
	s_nop 2
	buffer_load_dwordx4 v[106:109], v124, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[118:121], v[66:69], v[74:77], v126, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[66:69], v[34:37], v126, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[86:89], v[70:73], v[74:77], v126, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[70:73], v[34:37], v126, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v124, s[20:23], s6 offen offset:2048 nt
	s_nop 2
	buffer_load_dwordx4 v[74:77], v124, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v4, s[8:11], s29 offen offset:768
	buffer_load_dword v123, v4, s[8:11], s30 offen offset:768
	s_movk_i32 s33, 0x200
	s_mov_b32 m0, s28
	s_barrier
	ds_read_b128 v[82:85], v6 offset:10240
	ds_read_b128 v[86:89], v5 offset:8192
	ds_read_b128 v[114:117], v5 offset:10240
	ds_read_b128 v[118:121], v6 offset:8192
	ds_read_b32 v124, v4 offset:12800
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], v[118:121], v[14:17], v[42:45], v124, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[14:17], v[10:13], v124, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[94:97], v[42:45], v124, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[94:97], v[10:13], v124, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0x2000, v8
	s_nop 2
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], v[118:121], v[22:25], v[50:53], v124, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[22:25], v[18:21], v124, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[102:105], v[50:53], v124, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[102:105], v[18:21], v124, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 3
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[118:121], v[30:33], v[58:61], v124, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[30:33], v[26:29], v124, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[110:113], v[58:61], v124, v125 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[110:113], v[26:29], v124, v125 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 3
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], v[118:121], v[38:41], v[66:69], v124, v125 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[38:41], v[34:37], v124, v125 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[78:81], v[66:69], v124, v125 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[78:81], v[34:37], v124, v125 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	s_nop 3
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s19
	buffer_load_dword v9, v4, s[8:11], s29 offen offset:1024
	buffer_load_dword v124, v4, s[8:11], s30 offen offset:1024
	s_movk_i32 s33, 0x280
	s_barrier
	ds_read_b32 v125, v4 offset:13056
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	ds_read_b128 v[82:85], v6
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[118:121], v5 offset:2048
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[82:85], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[114:117], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:121], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[82:85], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[114:117], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:121], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[114:117], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:121], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[82:85], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[114:117], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[118:121], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v4, s[8:11], s29 offen offset:1280
	buffer_load_dword v123, v4, s[8:11], s30 offen offset:1280
	s_movk_i32 s33, 0x300
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v125, v4 offset:13312
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0x3000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v9, v4, s[8:11], s29 offen offset:1536
	buffer_load_dword v124, v4, s[8:11], s30 offen offset:1536
	s_movk_i32 s33, 0x380
	s_mov_b32 m0, s28
	s_barrier
	ds_read_b128 v[82:85], v6 offset:10240
	ds_read_b128 v[86:89], v5 offset:8192
	ds_read_b128 v[114:117], v5 offset:10240
	ds_read_b128 v[118:121], v6 offset:8192
	ds_read_b32 v125, v4 offset:13568
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s19
	buffer_load_dword v122, v4, s[8:11], s29 offen offset:1792
	buffer_load_dword v123, v4, s[8:11], s30 offen offset:1792
	s_movk_i32 s33, 0x400
	s_barrier
	ds_read_b32 v125, v4 offset:13824
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	ds_read_b128 v[82:85], v6
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[118:121], v5 offset:2048
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[82:85], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[114:117], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:121], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0x4000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[82:85], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[114:117], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:121], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[114:117], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:121], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[82:85], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[114:117], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[118:121], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v9, v4, s[8:11], s29 offen offset:2048
	buffer_load_dword v124, v4, s[8:11], s30 offen offset:2048
	s_movk_i32 s33, 0x480
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v125, v4 offset:14080
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v4, s[8:11], s29 offen offset:2304
	buffer_load_dword v123, v4, s[8:11], s30 offen offset:2304
	s_movk_i32 s33, 0x500
	s_mov_b32 m0, s28
	s_barrier
	ds_read_b128 v[82:85], v6 offset:10240
	ds_read_b128 v[86:89], v5 offset:8192
	ds_read_b128 v[114:117], v5 offset:10240
	ds_read_b128 v[118:121], v6 offset:8192
	ds_read_b32 v125, v4 offset:14336
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0x5000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s19
	buffer_load_dword v9, v4, s[8:11], s29 offen offset:2560
	buffer_load_dword v124, v4, s[8:11], s30 offen offset:2560
	s_movk_i32 s33, 0x580
	s_barrier
	ds_read_b32 v125, v4 offset:14592
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	ds_read_b128 v[82:85], v6
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[118:121], v5 offset:2048
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[82:85], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[114:117], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:121], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[82:85], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[114:117], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:121], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[114:117], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:121], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[82:85], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[114:117], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[118:121], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v4, s[8:11], s29 offen offset:2816
	buffer_load_dword v123, v4, s[8:11], s30 offen offset:2816
	s_movk_i32 s33, 0x600
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v125, v4 offset:14848
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0x6000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v9, v4, s[8:11], s29 offen offset:3072
	buffer_load_dword v124, v4, s[8:11], s30 offen offset:3072
	s_movk_i32 s33, 0x680
	s_mov_b32 m0, s28
	s_barrier
	ds_read_b128 v[82:85], v6 offset:10240
	ds_read_b128 v[86:89], v5 offset:8192
	ds_read_b128 v[114:117], v5 offset:10240
	ds_read_b128 v[118:121], v6 offset:8192
	ds_read_b32 v125, v4 offset:15104
	buffer_load_dwordx4 v7, s[24:27], s33 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s19
	buffer_load_dword v122, v4, s[8:11], s29 offen offset:3328
	buffer_load_dword v123, v4, s[8:11], s30 offen offset:3328
	s_barrier
	ds_read_b32 v125, v4 offset:15360
	buffer_load_dwordx4 v7, s[24:27], s31 offen lds
	ds_read_b128 v[82:85], v6
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[118:121], v5 offset:2048
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[82:85], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[114:117], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:121], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0x7000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[82:85], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[114:117], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:121], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[114:117], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:121], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[82:85], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[114:117], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[118:121], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v9, v4, s[8:11], s29 offen offset:3584
	buffer_load_dword v124, v4, s[8:11], s30 offen offset:3584
	s_movk_i32 s31, 0x780
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v125, v4 offset:15616
	buffer_load_dwordx4 v7, s[24:27], s31 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v4, s[8:11], s29 offen offset:3840
	buffer_load_dword v123, v4, s[8:11], s30 offen offset:3840
	s_movk_i32 s29, 0x800
	s_mov_b32 m0, s28
	s_barrier
	ds_read_b128 v[82:85], v6 offset:10240
	ds_read_b128 v[86:89], v5 offset:8192
	ds_read_b128 v[114:117], v5 offset:10240
	ds_read_b128 v[118:121], v6 offset:8192
	ds_read_b32 v125, v4 offset:15872
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0x8000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s19
	buffer_load_dword v9, v4, s[8:11], s17 offen
	buffer_load_dword v124, v4, s[8:11], s18 offen
	s_movk_i32 s29, 0x880
	s_barrier
	ds_read_b32 v125, v4 offset:16128
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	ds_read_b128 v[82:85], v6
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[118:121], v5 offset:2048
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[82:85], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[114:117], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:121], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[82:85], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[114:117], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:121], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[114:117], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:121], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[82:85], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[114:117], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[118:121], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v4, s[8:11], s17 offen offset:256
	buffer_load_dword v123, v4, s[8:11], s18 offen offset:256
	s_movk_i32 s29, 0x900
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v125, v4 offset:16384
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0x9000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v9, v4, s[8:11], s17 offen offset:512
	buffer_load_dword v124, v4, s[8:11], s18 offen offset:512
	s_movk_i32 s29, 0x980
	s_mov_b32 m0, s28
	s_barrier
	ds_read_b128 v[82:85], v6 offset:10240
	ds_read_b128 v[86:89], v5 offset:8192
	ds_read_b128 v[114:117], v5 offset:10240
	ds_read_b128 v[118:121], v6 offset:8192
	ds_read_b32 v125, v4 offset:16640
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s19
	buffer_load_dword v122, v4, s[8:11], s17 offen offset:768
	buffer_load_dword v123, v4, s[8:11], s18 offen offset:768
	s_movk_i32 s29, 0xa00
	s_barrier
	ds_read_b32 v125, v4 offset:16896
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	ds_read_b128 v[82:85], v6
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[118:121], v5 offset:2048
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[82:85], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[114:117], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:121], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0xa000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[82:85], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[114:117], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:121], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[114:117], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:121], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[82:85], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[114:117], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[118:121], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v9, v4, s[8:11], s17 offen offset:1024
	buffer_load_dword v124, v4, s[8:11], s18 offen offset:1024
	s_movk_i32 s29, 0xa80
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v125, v4 offset:17152
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v4, s[8:11], s17 offen offset:1280
	buffer_load_dword v123, v4, s[8:11], s18 offen offset:1280
	s_movk_i32 s29, 0xb00
	s_mov_b32 m0, s28
	s_barrier
	ds_read_b128 v[82:85], v6 offset:10240
	ds_read_b128 v[86:89], v5 offset:8192
	ds_read_b128 v[114:117], v5 offset:10240
	ds_read_b128 v[118:121], v6 offset:8192
	ds_read_b32 v125, v4 offset:17408
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0xb000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s19
	buffer_load_dword v9, v4, s[8:11], s17 offen offset:1536
	buffer_load_dword v124, v4, s[8:11], s18 offen offset:1536
	s_movk_i32 s29, 0xb80
	s_barrier
	ds_read_b32 v125, v4 offset:17664
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	ds_read_b128 v[82:85], v6
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[118:121], v5 offset:2048
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[82:85], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[114:117], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:121], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[82:85], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[114:117], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:121], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[114:117], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:121], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[82:85], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[114:117], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[118:121], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v122, v4, s[8:11], s17 offen offset:1792
	buffer_load_dword v123, v4, s[8:11], s18 offen offset:1792
	s_movk_i32 s29, 0xc00
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v125, v4 offset:17920
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0xc000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v9, v4, s[8:11], s17 offen offset:2048
	buffer_load_dword v124, v4, s[8:11], s18 offen offset:2048
	s_movk_i32 s29, 0xc80
	s_mov_b32 m0, s28
	s_barrier
	ds_read_b128 v[82:85], v6 offset:10240
	ds_read_b128 v[86:89], v5 offset:8192
	ds_read_b128 v[114:117], v5 offset:10240
	ds_read_b128 v[118:121], v6 offset:8192
	ds_read_b32 v125, v4 offset:18176
	buffer_load_dwordx4 v7, s[24:27], s29 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[46:49], v[14:17], v125, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[82:85], v[46:49], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[86:89], v[90:93], v[14:17], v125, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[114:117], v[90:93], v[10:13], v125, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[54:57], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[82:85], v[54:57], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[86:89], v[98:101], v[22:25], v125, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[114:117], v[98:101], v[18:21], v125, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[62:65], v[30:33], v125, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[82:85], v[62:65], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[86:89], v[106:109], v[30:33], v125, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[114:117], v[106:109], v[26:29], v125, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[70:73], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[82:85], v[70:73], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[86:89], v[74:77], v[38:41], v125, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[114:117], v[74:77], v[34:37], v125, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s19
	buffer_load_dword v122, v4, s[8:11], s17 offen offset:2304
	buffer_load_dword v123, v4, s[8:11], s18 offen offset:2304
	s_movk_i32 s19, 0xd00
	s_barrier
	ds_read_b32 v125, v4 offset:18432
	buffer_load_dwordx4 v7, s[24:27], s19 offen lds
	ds_read_b128 v[82:85], v6
	ds_read_b128 v[86:89], v6 offset:2048
	ds_read_b128 v[114:117], v5
	ds_read_b128 v[118:121], v5 offset:2048
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[82:85], v[42:45], v[14:17], v125, v9 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[86:89], v[42:45], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[114:117], v[94:97], v[14:17], v125, v9 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[118:121], v[94:97], v[10:13], v125, v9 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v126, 0xd000, v8
	buffer_load_dwordx4 v[42:45], v126, s[20:23], s16 offen nt
	buffer_load_dwordx4 v[94:97], v126, s[20:23], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[82:85], v[50:53], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[86:89], v[50:53], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[114:117], v[102:105], v[22:25], v125, v9 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], v[118:121], v[102:105], v[18:21], v125, v9 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v126, s[20:23], s15 offen nt
	buffer_load_dwordx4 v[102:105], v126, s[20:23], s15 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[82:85], v[58:61], v[30:33], v125, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[86:89], v[58:61], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[114:117], v[110:113], v[30:33], v125, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], v[118:121], v[110:113], v[26:29], v125, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v126, s[20:23], s7 offen nt
	buffer_load_dwordx4 v[110:113], v126, s[20:23], s7 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[82:85], v[66:69], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[86:89], v[66:69], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[114:117], v[78:81], v[38:41], v125, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], v[118:121], v[78:81], v[34:37], v125, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v126, s[20:23], s6 offen nt
	buffer_load_dwordx4 v[78:81], v126, s[20:23], s6 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v124, v4, s[8:11], s17 offen offset:2560
	buffer_load_dword v125, v4, s[8:11], s18 offen offset:2560
	s_movk_i32 s19, 0xd80
	s_mov_b32 m0, s14
	s_barrier
	ds_read_b128 v[82:85], v6 offset:6144
	ds_read_b128 v[86:89], v5 offset:4096
	ds_read_b128 v[114:117], v5 offset:6144
	ds_read_b128 v[118:121], v6 offset:4096
	ds_read_b32 v127, v4 offset:18688
	buffer_load_dwordx4 v7, s[24:27], s19 offen lds
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(13) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], v[118:121], v[46:49], v[14:17], v127, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[82:85], v[46:49], v[10:13], v127, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[86:89], v[90:93], v[14:17], v127, v122 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[114:117], v[90:93], v[8:11], v127, v122 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v126, s[20:23], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[90:93], v126, s[20:23], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], v[118:121], v[54:57], v[22:25], v127, v122 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[82:85], v[54:57], v[18:21], v127, v122 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[86:89], v[98:101], v[22:25], v127, v122 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[114:117], v[98:101], v[16:19], v127, v122 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v126, s[20:23], s15 offen offset:2048 nt
	buffer_load_dwordx4 v[98:101], v126, s[20:23], s15 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	s_waitcnt vmcnt(16)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], v[118:121], v[62:65], v[30:33], v127, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[82:85], v[62:65], v[26:29], v127, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[86:89], v[106:109], v[30:33], v127, v123 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[114:117], v[106:109], v[24:27], v127, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v126, s[20:23], s7 offen offset:2048 nt
	buffer_load_dwordx4 v[106:109], v126, s[20:23], s7 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], v[118:121], v[70:73], v[38:41], v127, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[82:85], v[70:73], v[34:37], v127, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[86:89], v[74:77], v[38:41], v127, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[114:117], v[74:77], v[32:35], v127, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v126, s[20:23], s6 offen offset:2048 nt
	buffer_load_dwordx4 v[74:77], v126, s[20:23], s6 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v118, v4, s[8:11], s17 offen offset:2816
	buffer_load_dword v119, v4, s[8:11], s18 offen offset:2816
	s_barrier
	ds_read_b128 v[82:85], v6 offset:8192
	ds_read_b32 v7, v4 offset:18944
	ds_read_b128 v[86:89], v6 offset:10240
	ds_read_b128 v[114:117], v5 offset:8192
	s_waitcnt vmcnt(12) lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[82:85], v[42:45], v[12:15], v7, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshlrev_b32_e32 v1, 10, v1
	v_lshl_or_b32 v3, s13, 5, v3
	v_add_lshl_u32 v1, v3, v1, 2
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[86:89], v[42:45], v[8:11], v7, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v5 offset:10240
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[82:85], v[50:53], v[20:23], v7, v124 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v3, 0x800, v1
	s_lshl_b32 s6, s12, 6
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[86:89], v[50:53], v[16:19], v7, v124 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[50:53], v6
	s_waitcnt vmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[82:85], v[58:61], v[28:31], v7, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[86:89], v[58:61], v[24:27], v7, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[82:85], v[66:69], v[36:39], v7, v125 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[86:89], v[66:69], v[32:35], v7, v125 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v66, v4 offset:19200
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[114:117], v[94:97], v[12:15], v7, v124 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[114:117], v[102:105], v[20:23], v7, v124 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[114:117], v[110:113], v[28:31], v7, v125 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[40:43], v[94:97], v[8:11], v7, v124 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[40:43], v[102:105], v[16:19], v7, v124 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[40:43], v[110:113], v[24:27], v7, v125 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[40:43], v[78:81], v[32:35], v7, v125 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[40:43], v6 offset:2048
	ds_read_b128 v[58:61], v5
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[114:117], v[78:81], v[36:39], v7, v125 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(1) lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[50:53], v[46:49], v[12:15], v66, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[50:53], v[54:57], v[20:23], v66, v118 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[50:53], v[62:65], v[28:31], v66, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[40:43], v[46:49], v[8:11], v66, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[44:47], v5 offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[58:61], v[90:93], v[12:15], v66, v118 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[58:61], v[98:101], v[20:23], v66, v118 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 6
	ds_write_b32 v1, v13 offset:3072
	ds_write_b32 v1, v20 offset:512
	ds_write_b32 v1, v21 offset:1536
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[58:61], v[106:109], v[28:31], v66, v119 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v1, v22 offset:2560
	ds_write_b32 v1, v23 offset:3584
	s_nop 5
	ds_write2_b32 v1, v10, v28 offset1:16
	ds_write_b32 v1, v29 offset:1088
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[50:53], v[70:73], v[36:39], v66, v119 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_write2_b32 v3, v12, v30 offset1:16
	v_add_u32_e32 v3, 0x200, v1
	ds_write_b32 v1, v31 offset:3136
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[58:61], v[74:77], v[20:23], v66, v119 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[44:47], v[90:93], v[6:9], v66, v118 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 6
	ds_write2_b32 v3, v12, v11 offset0:16 offset1:128
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[40:43], v[54:57], v[16:19], v66, v118 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_write_b32 v1, v13 offset:1600
	ds_write_b32 v1, v14 offset:2624
	ds_write_b32 v1, v15 offset:3648
	ds_write_b32 v1, v5 offset:17408
	ds_write_b32 v1, v6 offset:18432
	ds_write_b32 v1, v7 offset:19456
	v_add_u32_e32 v3, 0x4000, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[44:47], v[98:101], v[8:11], v66, v118 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 7
	ds_write_b32 v1, v8 offset:16896
	ds_write_b32 v1, v9 offset:17920
	ds_write_b32 v1, v10 offset:18944
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[40:43], v[62:65], v[24:27], v66, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_write_b32 v1, v11 offset:19968
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], v[44:47], v[106:109], v[12:15], v66, v119 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], v[40:43], v[70:73], v[32:35], v66, v119 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 6
	ds_write2_b32 v3, v4, v6 offset1:16
	ds_write_b32 v1, v7 offset:17472
	ds_write_b32 v1, v8 offset:18496
	ds_write_b32 v1, v9 offset:19520
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[44:47], v[74:77], v[10:13], v66, v119 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 7
	ds_write_b32 v1, v4 offset:16960
	ds_write_b32 v1, v5 offset:17984
	ds_write_b32 v1, v6 offset:19008
	ds_write_b32 v1, v7 offset:20032
	v_and_b32_e32 v5, 3, v0
	v_lshrrev_b32_e32 v3, 4, v0
	v_bfe_u32 v4, v0, 2, 2
	v_lshlrev_b32_e32 v0, 5, v5
	v_lshl_or_b32 v0, v4, 7, v0
	v_lshl_or_b32 v29, v3, 10, v0
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[0:1], v29 offset0:4 offset1:7
	v_lshlrev_b32_e32 v9, 4, v4
	v_lshlrev_b32_e32 v11, 2, v5
	v_or_b32_e32 v12, s3, v3
	v_or3_b32 v9, v11, s6, v9
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v6, 0xbfb8aa3b, v0
	v_exp_f32_e32 v8, v6
	v_mul_f32_e32 v6, 0xbfb8aa3b, v1
	v_lshl_add_u32 v12, v12, 8, v9
	v_add_u32_e32 v9, 0x4000, v29
	v_exp_f32_e32 v10, v6
	ds_read2_b32 v[6:7], v29 offset0:132 offset1:135
	ds_read2_b32 v[16:17], v9 offset0:4 offset1:7
	ds_read2_b32 v[18:19], v9 offset0:132 offset1:135
	ds_read2_b32 v[20:21], v29 offset1:1
	ds_read2_b32 v[22:23], v29 offset0:2 offset1:3
	ds_read2_b32 v[24:25], v29 offset0:128 offset1:129
	ds_read2_b32 v[26:27], v29 offset0:5 offset1:6
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[14:15], s[4:5], 0, v[12:13]
	v_or_b32_e32 v13, 0x4000, v29
	v_or_b32_e32 v31, 0x4200, v29
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v36, 0xbfb8aa3b, v26
	v_exp_f32_e32 v42, v36
	ds_read2_b32 v[36:37], v29 offset0:130 offset1:131
	ds_read2_b32 v[38:39], v29 offset0:133 offset1:134
	ds_read2_b32 v[40:41], v13 offset1:1
	v_or_b32_e32 v33, 0x4008, v29
	v_or_b32_e32 v35, 0x4208, v29
	v_or_b32_e32 v45, 0x4014, v29
	v_or_b32_e32 v47, 0x4214, v29
	v_mul_f32_e32 v29, 0xbfb8aa3b, v27
	v_add_f32_e32 v13, 1.0, v42
	v_exp_f32_e32 v29, v29
	ds_read2_b32 v[42:43], v31 offset1:1
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v31, 0xbfb8aa3b, v40
	v_exp_f32_e32 v31, v31
	v_rcp_f32_e32 v44, v13
	v_add_f32_e32 v13, 1.0, v29
	v_rcp_f32_e32 v46, v13
	v_add_f32_e32 v13, 1.0, v31
	v_rcp_f32_e32 v29, v13
	v_mul_f32_e32 v13, 0xbfb8aa3b, v41
	v_exp_f32_e32 v13, v13
	ds_read2_b32 v[48:49], v33 offset1:1
	ds_read2_b32 v[50:51], v35 offset1:1
	ds_read2_b32 v[52:53], v45 offset1:1
	ds_read2_b32 v[54:55], v47 offset1:1
	v_mul_f32_e32 v28, 0xbfb8aa3b, v20
	v_exp_f32_e32 v28, v28
	v_add_f32_e32 v13, 1.0, v13
	v_mul_f32_e32 v30, 0xbfb8aa3b, v21
	v_mul_f32_e32 v32, 0xbfb8aa3b, v22
	v_rcp_f32_e32 v31, v13
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v13, 0xbfb8aa3b, v48
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v32, v32
	v_exp_f32_e32 v13, v13
	v_add_f32_e32 v28, 1.0, v28
	v_rcp_f32_e32 v28, v28
	v_add_f32_e32 v30, 1.0, v30
	v_add_f32_e32 v32, 1.0, v32
	v_add_f32_e32 v13, 1.0, v13
	v_rcp_f32_e32 v30, v30
	v_rcp_f32_e32 v32, v32
	v_rcp_f32_e32 v33, v13
	v_mov_b32_e32 v56, v20
	v_mov_b32_e32 v57, v40
	v_pk_mul_f32 v[28:29], v[56:57], v[28:29]
	v_mov_b32_e32 v56, v24
	v_mov_b32_e32 v57, v42
	v_mov_b32_e32 v40, v21
	v_mov_b32_e32 v42, v25
	v_mov_b32_e32 v24, v22
	v_mov_b32_e32 v25, v48
	v_pk_mul_f32 v[20:21], v[40:41], v[30:31]
	v_pk_mul_f32 v[24:25], v[24:25], v[32:33]
	v_mov_b32_e32 v30, v36
	s_waitcnt lgkmcnt(2)
	v_mov_b32_e32 v31, v50
	v_pk_mul_f32 v[24:25], v[30:31], v[24:25]
	v_mov_b32_e32 v30, v0
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v0, 0xbfb8aa3b, v52
	v_exp_f32_e32 v0, v0
	v_mul_f32_e32 v9, 0xbfb8aa3b, v16
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v34, 0xbfb8aa3b, v23
	v_mul_f32_e32 v13, 0xbfb8aa3b, v49
	v_exp_f32_e32 v34, v34
	v_exp_f32_e32 v13, v13
	v_add_f32_e32 v0, 1.0, v0
	v_mul_f32_e32 v11, 0xbfb8aa3b, v17
	v_rcp_f32_e32 v45, v0
	v_mul_f32_e32 v0, 0xbfb8aa3b, v53
	v_add_f32_e32 v8, 1.0, v8
	v_exp_f32_e32 v11, v11
	v_add_f32_e32 v9, 1.0, v9
	v_exp_f32_e32 v0, v0
	v_rcp_f32_e32 v8, v8
	v_rcp_f32_e32 v9, v9
	v_add_f32_e32 v34, 1.0, v34
	v_add_f32_e32 v13, 1.0, v13
	v_rcp_f32_e32 v34, v34
	v_rcp_f32_e32 v35, v13
	v_add_f32_e32 v10, 1.0, v10
	v_add_f32_e32 v11, 1.0, v11
	v_mov_b32_e32 v31, v16
	v_add_f32_e32 v0, 1.0, v0
	v_rcp_f32_e32 v10, v10
	v_rcp_f32_e32 v11, v11
	v_mov_b32_e32 v48, v23
	v_pk_mul_f32 v[8:9], v[30:31], v[8:9]
	v_mov_b32_e32 v30, v6
	v_mov_b32_e32 v31, v18
	v_rcp_f32_e32 v47, v0
	v_pk_mul_f32 v[28:29], v[56:57], v[28:29]
	v_pk_mul_f32 v[20:21], v[42:43], v[20:21]
	v_pk_mul_f32 v[22:23], v[48:49], v[34:35]
	v_mov_b32_e32 v50, v37
	v_pk_mul_f32 v[8:9], v[30:31], v[8:9]
	v_mov_b32_e32 v30, v26
	v_mov_b32_e32 v31, v52
	v_pk_mul_f32 v[22:23], v[50:51], v[22:23]
	v_pk_mul_f32 v[30:31], v[30:31], v[44:45]
	v_mov_b32_e32 v32, v38
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v33, v54
	v_max_f32_e64 v0, |v28|, |v20|
	v_pk_mul_f32 v[30:31], v[32:33], v[30:31]
	v_mov_b32_e32 v52, v27
	v_max_f32_e64 v6, |v29|, |v21|
	v_max3_f32 v0, v0, |v24|, |v22|
	v_mov_b32_e32 v16, v1
	v_pk_mul_f32 v[26:27], v[52:53], v[46:47]
	v_mov_b32_e32 v54, v39
	v_max3_f32 v6, v6, |v25|, |v23|
	v_max3_f32 v13, v0, |v8|, |v30|
	v_pk_mul_f32 v[0:1], v[16:17], v[10:11]
	v_mov_b32_e32 v18, v7
	v_pk_mul_f32 v[26:27], v[54:55], v[26:27]
	v_max3_f32 v32, v6, |v9|, |v31|
	v_pk_mul_f32 v[6:7], v[18:19], v[0:1]
	s_mov_b32 s6, 0x3e800000
	v_max3_f32 v1, v13, |v26|, |v6|
	v_max3_f32 v0, v32, |v27|, |v7|
	v_cmp_eq_u32_e32 vcc, 0, v5
	v_mov_b32_dpp v10, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v11, v0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v10, v10, v10
	v_max_f32_e32 v11, v11, v11
	v_max_f32_e32 v1, v1, v10
	v_max_f32_e32 v0, v0, v11
	s_nop 0
	v_mov_b32_dpp v10, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_mov_b32_dpp v11, v0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v10, v10, v10
	v_max_f32_e32 v10, v1, v10
	v_max_f32_e32 v1, v11, v11
	v_max_f32_e32 v0, v0, v1
	v_add_u32_e32 v1, 0x200000, v0
	v_add_u32_e32 v0, 0x200000, v10
	v_pk_mul_f32 v[0:1], v[0:1], s[6:7] op_sel_hi:[1,0]
	v_mov_b32_e32 v10, 0
	v_cvt_scalef32_pk_fp4_f32 v10, v28, v20, v0
	v_cvt_scalef32_pk_fp4_f32 v10, v24, v22, v0 op_sel:[0,0,1,0]
	v_cvt_scalef32_pk_fp4_f32 v2, v29, v21, v1
	v_cvt_scalef32_pk_fp4_f32 v10, v8, v30, v0 op_sel:[0,0,0,1]
	v_cvt_scalef32_pk_fp4_f32 v2, v25, v23, v1 op_sel:[0,0,1,0]
	v_cvt_scalef32_pk_fp4_f32 v10, v26, v6, v0 op_sel:[0,0,1,1]
	v_cvt_scalef32_pk_fp4_f32 v2, v9, v31, v1 op_sel:[0,0,0,1]
	v_add_u32_e32 v6, 0x1000, v12
	v_cvt_scalef32_pk_fp4_f32 v2, v27, v7, v1 op_sel:[0,0,1,1]
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	global_store_dword v[14:15], v10, off nt
	global_store_dword v[6:7], v2, off nt
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_3
; %bb.2:
	v_lshrrev_b32_e32 v1, 23, v1
	s_movk_i32 s3, 0xfe
	v_min_u32_sdwa v1, v1, s3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_lshl_b32 s3, s12, 5
	v_lshrrev_b32_e32 v0, 23, v0
	s_lshl_b32 s2, s2, 7
	s_and_b32 s3, s3, 0x3fffffc0
	s_load_dwordx2 s[0:1], s[0:1], 0x48
	v_min_u32_e32 v0, 0xfe, v0
	s_add_i32 s3, s3, s2
	v_or_b32_e32 v2, v0, v1
	v_or_b32_e32 v0, s3, v3
	s_lshl_b32 s2, s12, 1
	v_lshlrev_b32_e32 v1, 6, v4
	v_lshlrev_b32_e32 v0, 2, v0
	s_and_b32 s2, s2, 2
	v_or3_b32 v0, v0, v1, s2
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], s[0:1], 0, v[0:1]
	global_store_short v[0:1], v2, off
.LBB0_3:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
		.amdhsa_group_segment_fixed_size 32768
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
		.amdhsa_next_free_vgpr 128
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 128
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
	.section	.text,"axG",@progbits,_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,comdat,unique,1
.Lfunc_end0:
	.size	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16, .Lfunc_end0-_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
                                        ; -- End function
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.num_vgpr, 128
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.num_agpr, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.numbered_sgpr, 36
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.private_seg_size, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.uses_vcc, 1
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.uses_flat_scratch, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_dyn_sized_stack, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_recursion, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 14604
; TotalNumSgprs: 42
; NumVgprs: 128
; NumAgprs: 0
; TotalNumVgprs: 128
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 32768 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 15
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 128
; AccumOffset: 128
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 31
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_8ff49d58d471d376,@object ; @__hip_cuid_8ff49d58d471d376
	.section	.bss,"aw",@nobits,unique,2
	.globl	__hip_cuid_8ff49d58d471d376
__hip_cuid_8ff49d58d471d376:
	.byte	0                               ; 0x0
	.size	__hip_cuid_8ff49d58d471d376, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_8ff49d58d471d376
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
      - .actual_access:  read_only
        .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 32768
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.kd
    .uses_dynamic_stack: false
    .vgpr_count:     128
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
