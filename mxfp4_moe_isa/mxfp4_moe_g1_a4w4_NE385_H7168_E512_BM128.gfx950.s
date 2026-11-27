	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text,"axG",@progbits,_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,comdat,unique,1
	.protected	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16 ; -- Begin function _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
	.globl	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
	.p2align	8
	.type	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,@function
_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16: ; @_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
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
	v_readfirstlane_b32 s28, v0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s2, 31
	s_lshr_b32 s3, s3, 25
	s_add_i32 s2, s2, s3
	s_ashr_i32 s2, s2, 7
	s_lshl_b32 s2, s2, 2
	s_cmp_ge_i32 s16, s2
	s_cbranch_scc1 .LBB0_3
; %bb.1:
	s_load_dword s2, s[0:1], 0x38
	s_mov_b64 s[20:21], s[6:7]
	s_mov_b64 s[18:19], s[10:11]
	s_and_b32 s25, s25, 0xffff
	s_and_b32 s21, s21, 0xffff
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s26, s2, 0xe00
	s_ashr_i32 s2, s16, 31
	s_lshr_b32 s2, s2, 30
	s_add_i32 s3, s16, s2
	s_ashr_i32 s2, s3, 2
	s_and_b32 s3, s3, -4
	s_sub_i32 s12, s16, s3
	s_ashr_i32 s3, s2, 31
	s_and_b32 s9, s9, 0xffff
	s_and_b32 s5, s5, 0xffff
	s_lshr_b32 s30, s28, 6
	s_lshl_b64 s[16:17], s[2:3], 2
	s_add_u32 s16, s18, s16
	s_addc_u32 s17, s19, s17
	s_lshl_b32 s3, s2, 7
	v_bfe_u32 v1, v0, 3, 3
	s_lshl_b32 s13, s30, 5
	v_or_b32_e32 v1, s3, v1
	v_add_u32_e32 v2, s13, v1
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], v[2:3], 2, s[14:15]
	s_load_dword s18, s[16:17], 0x0
	global_load_dword v1, v[2:3], off
	global_load_dword v4, v[2:3], off offset:32
	global_load_dword v5, v[2:3], off offset:64
	global_load_dword v6, v[2:3], off offset:96
	s_lshl_b32 s31, s12, 3
	s_lshl_b32 s33, s30, 1
	s_add_i32 s31, s33, s31
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s14, s18, 10
	s_movk_i32 s19, 0xffc0
	s_mul_i32 s18, s18, 0xe000
	s_mulk_i32 s31, 0x700
	v_mov_b32_e32 v2, s28
	s_mov_b32 s27, 0x20000
	s_add_i32 s18, s18, s31
	v_bfi_b32 v2, s19, v2, v0
	s_lshl_b32 s19, s30, 10
	s_mov_b32 s6, 0x8c00000
	s_mov_b32 s7, s27
	s_lshl_b32 s15, s12, 8
	s_lshl_b32 s36, s18, 2
	v_lshlrev_b32_e32 v3, 4, v2
	s_mul_i32 s18, s2, 0x7000
	s_add_i32 m0, s19, 0x8000
	s_lshl_b32 s31, s30, 8
	s_add_i32 s14, s14, s15
	s_and_b32 s15, s28, 0xffffffc0
	v_lshlrev_b32_e32 v2, 2, v2
	buffer_load_dwordx4 v3, s[4:7], s18 offen lds
	s_add_i32 s28, s18, 0x1000
	s_add_i32 m0, s31, 0x9000
	s_movk_i32 s29, 0xe00
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x1400
	s_add_i32 m0, s31, 0x9400
	v_and_b32_e32 v30, 48, v0
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x1800
	s_add_i32 m0, s31, 0x9800
	s_lshl_b32 s34, s30, 12
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x1c00
	s_add_i32 m0, s19, 0x9c00
	s_or_b32 s30, s34, 0x400
	buffer_load_dwordx4 v3, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x2c00
	s_add_i32 m0, s31, 0xac00
	s_or_b32 s33, s34, 0xc00
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x3000
	s_add_i32 m0, s31, 0xb000
	v_and_b32_e32 v99, 15, v0
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x3400
	s_add_i32 m0, s31, 0xb400
	s_add_i32 s14, s14, s15
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x3800
	s_add_i32 m0, s19, 0xb800
	s_mov_b32 s22, 0x54380000
	buffer_load_dwordx4 v3, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x4800
	s_add_i32 m0, s31, 0xc800
	s_mov_b32 s23, s27
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x4c00
	s_add_i32 m0, s31, 0xcc00
	s_mul_i32 s17, s14, 0xe00
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x5000
	s_add_i32 m0, s31, 0xd000
	s_or_b32 s16, s14, 16
	buffer_load_dword v2, s[4:7], s28 offen lds
	s_add_i32 s28, s18, 0x5400
	s_add_i32 m0, s19, 0xd400
	s_add_i32 s19, s18, 0x6400
	buffer_load_dwordx4 v3, s[4:7], s28 offen lds
	s_add_i32 m0, s31, 0xe400
	s_or_b32 s15, s14, 32
	buffer_load_dword v2, s[4:7], s19 offen lds
	s_add_i32 s19, s18, 0x6800
	s_add_i32 m0, s31, 0xe800
	s_addk_i32 s18, 0x6c00
	buffer_load_dword v2, s[4:7], s19 offen lds
	s_add_i32 m0, s31, 0xec00
	s_waitcnt vmcnt(18)
	v_mul_lo_u32 v1, v1, s29
	buffer_load_dword v2, s[4:7], s18 offen lds
	v_lshlrev_b32_e32 v2, 4, v0
	s_movk_i32 s4, 0x70
	v_and_b32_e32 v3, 0x70, v2
	v_bitop3_b32 v2, v2, v30, s4 bitop3:0x6c
	v_bitop3_b32 v94, v1, v3, v30 bitop3:0xf6
	s_mov_b32 m0, s34
	s_waitcnt vmcnt(18)
	v_mul_lo_u32 v1, v4, s29
	buffer_load_dwordx4 v94, s[24:27], 0 offen lds
	v_bitop3_b32 v95, v1, v2, 64 bitop3:0xf6
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(18)
	v_mul_lo_u32 v1, v5, s29
	s_or_b32 s31, s34, 0x800
	buffer_load_dwordx4 v95, s[24:27], 0 offen lds
	v_bitop3_b32 v101, v1, v3, v30 bitop3:0xf6
	s_mov_b32 m0, s31
	s_waitcnt vmcnt(18)
	v_mul_lo_u32 v1, v6, s29
	buffer_load_dwordx4 v101, s[24:27], 0 offen lds
	v_bitop3_b32 v102, v1, v2, 64 bitop3:0xf6
	s_mov_b32 m0, s33
	v_bfe_u32 v1, v0, 4, 2
	v_lshlrev_b32_e32 v2, 4, v99
	s_add_i32 s29, s34, 0x4000
	s_or_b32 s14, s14, 48
	buffer_load_dwordx4 v102, s[24:27], 0 offen lds
	v_lshl_or_b32 v103, v1, 8, v2
	s_movk_i32 s4, 0x80
	s_mov_b32 m0, s29
	s_add_i32 s6, s34, 0x4400
	s_mulk_i32 s16, 0xe00
	s_mulk_i32 s15, 0xe00
	s_mulk_i32 s14, 0xe00
	buffer_load_dwordx4 v[10:13], v103, s[20:23], s17 offen
	buffer_load_dwordx4 v[14:17], v103, s[20:23], s17 offen offset:1024
	buffer_load_dwordx4 v[22:25], v103, s[20:23], s16 offen
	buffer_load_dwordx4 v[34:37], v103, s[20:23], s16 offen offset:1024
	buffer_load_dwordx4 v[38:41], v103, s[20:23], s15 offen
	buffer_load_dwordx4 v[42:45], v103, s[20:23], s15 offen offset:1024
	buffer_load_dwordx4 v[46:49], v103, s[20:23], s14 offen
	buffer_load_dwordx4 v[54:57], v103, s[20:23], s14 offen offset:1024
	s_add_i32 s7, s34, 0x4800
	buffer_load_dwordx4 v94, s[24:27], s4 offen lds
	s_mov_b32 m0, s6
	s_add_i32 s18, s34, 0x4c00
	buffer_load_dwordx4 v95, s[24:27], s4 offen lds
	s_mov_b32 m0, s7
	v_lshlrev_b32_e32 v2, 2, v99
	buffer_load_dwordx4 v101, s[24:27], s4 offen lds
	s_mov_b32 m0, s18
	s_mov_b32 s10, 0x5438000
	s_mov_b32 s11, s27
	v_lshl_or_b32 v96, v1, 6, v2
	buffer_load_dwordx4 v102, s[24:27], s4 offen lds
	s_add_i32 s37, s36, 0x1c00
	buffer_load_dword v148, v96, s[8:11], s36 offen
	buffer_load_dwordx4 v[62:65], v103, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[66:69], v103, s[20:23], s17 offen offset:3072
	buffer_load_dwordx4 v[58:61], v103, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[50:53], v103, s[20:23], s16 offen offset:3072
	buffer_load_dwordx4 v[26:29], v103, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[18:21], v103, s[20:23], s15 offen offset:3072
	buffer_load_dwordx4 v[6:9], v103, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[2:5], v103, s[20:23], s14 offen offset:3072
	buffer_load_dword v152, v96, s[8:11], s37 offen offset:256
	buffer_load_dword v149, v96, s[8:11], s37 offen
	buffer_load_dword v153, v96, s[8:11], s36 offen offset:256
	v_lshlrev_b32_e32 v31, 3, v0
	v_and_b32_e32 v31, 0x70, v31
	v_lshlrev_b32_e32 v32, 7, v99
	v_bitop3_b32 v97, v32, v31, v30 bitop3:0xf6
	v_or_b32_e32 v30, 64, v30
	v_bitop3_b32 v100, v32, v30, v31 bitop3:0xf6
	s_movk_i32 s4, 0x100
	s_mov_b32 m0, s34
	s_barrier
	ds_read_b128 v[70:73], v97
	ds_read_b128 v[74:77], v97 offset:2048
	ds_read_b128 v[78:81], v97 offset:4096
	ds_read_b128 v[82:85], v97 offset:6144
	ds_read_b128 v[86:89], v97 offset:8192
	ds_read_b128 v[90:93], v97 offset:10240
	ds_read_b128 v[104:107], v97 offset:12288
	ds_read_b128 v[108:111], v97 offset:14336
	ds_read_b128 v[112:115], v100
	ds_read_b128 v[116:119], v100 offset:2048
	ds_read_b128 v[120:123], v100 offset:4096
	ds_read_b128 v[124:127], v100 offset:6144
	ds_read_b128 v[128:131], v100 offset:8192
	ds_read_b128 v[132:135], v100 offset:10240
	ds_read_b128 v[136:139], v100 offset:12288
	ds_read_b128 v[140:143], v100 offset:14336
	ds_read2st64_b32 v[144:145], v96 offset0:128 offset1:156
	ds_read2st64_b32 v[146:147], v96 offset0:184 offset1:212
	buffer_load_dwordx4 v94, s[24:27], s4 offen lds
	s_mov_b32 m0, s30
	s_movk_i32 s39, 0x700
	buffer_load_dwordx4 v95, s[24:27], s4 offen lds
	s_mov_b32 m0, s31
	s_add_i32 s19, s36, 0x1000
	buffer_load_dwordx4 v101, s[24:27], s4 offen lds
	s_mov_b32 m0, s33
	s_add_i32 s28, s37, 0x1000
	buffer_load_dwordx4 v102, s[24:27], s4 offen lds
	s_load_dwordx2 s[4:5], s[0:1], 0x40
	s_waitcnt vmcnt(15) lgkmcnt(0)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[10:13], 0, v144, v148 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:77], v[10:13], 0, v144, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[78:81], v[10:13], 0, v145, v148 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[82:85], v[10:13], 0, v145, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[86:89], v[10:13], 0, v146, v148 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[90:93], v[10:13], 0, v146, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[104:107], v[10:13], 0, v147, v148 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[108:111], v[10:13], 0, v147, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[112:115], v[14:17], a[0:3], v144, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v98, 0
	s_movk_i32 s40, 0x400
	s_movk_i32 s38, 0x800
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[116:119], v[14:17], a[4:7], v144, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_movk_i32 s35, 0xc00
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[120:123], v[14:17], a[8:11], v145, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[14:17], a[12:15], v145, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[14:17], a[16:19], v146, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[14:17], a[20:23], v146, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[14:17], a[24:27], v147, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[14:17], a[28:31], v147, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v154, 0x1000, v103
	buffer_load_dwordx4 v[30:33], v154, s[20:23], s17 offen
	buffer_load_dwordx4 v[10:13], v154, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[22:25], 0, v144, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[22:25], 0, v144, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[78:81], v[22:25], 0, v145, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[82:85], v[22:25], 0, v145, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[86:89], v[22:25], 0, v146, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[22:25], 0, v146, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:107], v[22:25], 0, v147, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[22:25], 0, v147, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[112:115], v[34:37], a[32:35], v144, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[116:119], v[34:37], a[36:39], v144, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:123], v[34:37], a[40:43], v145, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[34:37], a[44:47], v145, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[34:37], a[48:51], v146, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[34:37], a[52:55], v146, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[34:37], a[56:59], v147, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[34:37], a[60:63], v147, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[22:25], v154, s[20:23], s16 offen
	buffer_load_dwordx4 v[14:17], v154, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[70:73], v[38:41], 0, v144, v149 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:77], v[38:41], 0, v144, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[38:41], 0, v145, v149 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:85], v[38:41], 0, v145, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[86:89], v[38:41], 0, v146, v149 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[90:93], v[38:41], 0, v146, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[104:107], v[38:41], 0, v147, v149 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[108:111], v[38:41], 0, v147, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:115], v[42:45], a[64:67], v144, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[42:45], a[68:71], v144, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:123], v[42:45], a[72:75], v145, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[42:45], a[76:79], v145, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[42:45], a[80:83], v146, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[42:45], a[84:87], v146, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[42:45], a[88:91], v147, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[42:45], a[92:95], v147, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v154, s[20:23], s15 offen
	buffer_load_dwordx4 v[34:37], v154, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[70:73], v[46:49], 0, v144, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:77], v[46:49], 0, v144, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[78:81], v[46:49], 0, v145, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[46:49], 0, v145, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[86:89], v[46:49], 0, v146, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[90:93], v[46:49], 0, v146, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[104:107], v[46:49], 0, v147, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[108:111], v[46:49], 0, v147, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[112:115], v[54:57], a[96:99], v144, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[116:119], v[54:57], a[100:103], v144, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:123], v[54:57], a[104:107], v145, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:127], v[54:57], a[108:111], v145, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[54:57], a[112:115], v146, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[54:57], a[116:119], v146, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[54:57], a[120:123], v147, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[54:57], a[124:127], v147, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v154, s[20:23], s14 offen
	buffer_load_dwordx4 v[42:45], v154, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s41, 0x180
	buffer_load_dword v156, v96, s[8:11], s36 offen offset:512
	buffer_load_dword v157, v96, s[8:11], s37 offen offset:512
	s_barrier
	ds_read_b128 v[70:73], v97 offset:16384
	ds_read2st64_b32 v[148:149], v96 offset0:129 offset1:157
	ds_read_b128 v[78:81], v97 offset:18432
	ds_read_b128 v[82:85], v100 offset:16384
	ds_read_b128 v[86:89], v100 offset:18432
	ds_read_b128 v[90:93], v97 offset:20480
	ds_read_b128 v[104:107], v97 offset:22528
	ds_read_b128 v[108:111], v100 offset:20480
	ds_read_b128 v[112:115], v100 offset:22528
	ds_read_b128 v[116:119], v97 offset:24576
	ds_read2st64_b32 v[150:151], v96 offset0:185 offset1:213
	ds_read_b128 v[120:123], v97 offset:26624
	ds_read_b128 v[132:135], v97 offset:28672
	ds_read_b128 v[124:127], v100 offset:24576
	ds_read_b128 v[136:139], v97 offset:30720
	ds_read_b128 v[128:131], v100 offset:26624
	ds_read_b128 v[140:143], v100 offset:28672
	ds_read_b128 v[144:147], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s41 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(15) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[62:65], a[0:3], v148, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s41 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s41 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[62:65], a[4:7], v148, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s41 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[62:65], a[8:11], v149, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[62:65], a[12:15], v149, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[62:65], a[16:19], v150, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[62:65], a[20:23], v150, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[62:65], a[24:27], v151, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[62:65], a[28:31], v151, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:85], v[66:69], a[0:3], v148, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[86:89], v[66:69], a[4:7], v148, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[66:69], a[8:11], v149, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[66:69], a[12:15], v149, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[66:69], a[16:19], v150, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[66:69], a[20:23], v150, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[66:69], a[24:27], v151, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:147], v[66:69], a[28:31], v151, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[74:77], v154, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[46:49], v154, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[58:61], a[32:35], v148, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[58:61], a[36:39], v148, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[58:61], a[40:43], v149, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[58:61], a[44:47], v149, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[58:61], a[48:51], v150, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[58:61], a[52:55], v150, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[58:61], a[56:59], v151, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[58:61], a[60:63], v151, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[50:53], a[32:35], v148, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:89], v[50:53], a[36:39], v148, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[50:53], a[40:43], v149, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[50:53], a[44:47], v149, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:127], v[50:53], a[48:51], v150, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[50:53], a[52:55], v150, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[50:53], a[56:59], v151, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[50:53], a[60:63], v151, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v154, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[50:53], v154, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[70:73], v[26:29], a[64:67], v148, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[26:29], a[68:71], v148, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[26:29], a[72:75], v149, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[26:29], a[76:79], v149, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[26:29], a[80:83], v150, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[26:29], a[84:87], v150, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[26:29], a[88:91], v151, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[26:29], a[92:95], v151, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[82:85], v[18:21], a[64:67], v148, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[86:89], v[18:21], a[68:71], v148, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[18:21], a[72:75], v149, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[112:115], v[18:21], a[76:79], v149, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[18:21], a[80:83], v150, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[128:131], v[18:21], a[84:87], v150, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[18:21], a[88:91], v151, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[18:21], a[92:95], v151, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v154, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[18:21], v154, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[70:73], v[6:9], a[96:99], v148, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[6:9], a[100:103], v148, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[6:9], a[104:107], v149, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[6:9], a[108:111], v149, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[6:9], a[112:115], v150, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[6:9], a[116:119], v150, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[6:9], a[120:123], v151, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[6:9], a[124:127], v151, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[2:5], a[96:99], v148, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[86:89], v[2:5], a[100:103], v148, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[2:5], a[104:107], v149, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[2:5], a[108:111], v149, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[2:5], a[112:115], v150, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[2:5], a[116:119], v150, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[2:5], a[120:123], v151, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:147], v[2:5], a[124:127], v151, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v154, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[2:5], v154, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	s_movk_i32 s41, 0x200
	buffer_load_dword v160, v96, s[8:11], s36 offen offset:768
	buffer_load_dword v161, v96, s[8:11], s37 offen offset:768
	s_barrier
	ds_read_b128 v[26:29], v97
	ds_read2st64_b32 v[152:153], v96 offset0:130 offset1:158
	ds_read_b128 v[82:85], v97 offset:2048
	ds_read_b128 v[86:89], v100
	ds_read_b128 v[90:93], v100 offset:2048
	ds_read_b128 v[104:107], v97 offset:4096
	ds_read_b128 v[108:111], v97 offset:6144
	ds_read_b128 v[112:115], v100 offset:4096
	ds_read_b128 v[116:119], v100 offset:6144
	ds_read_b128 v[120:123], v97 offset:8192
	ds_read2st64_b32 v[154:155], v96 offset0:186 offset1:214
	ds_read_b128 v[124:127], v97 offset:10240
	ds_read_b128 v[136:139], v97 offset:12288
	ds_read_b128 v[128:131], v100 offset:8192
	ds_read_b128 v[140:143], v97 offset:14336
	ds_read_b128 v[132:135], v100 offset:10240
	ds_read_b128 v[144:147], v100 offset:12288
	ds_read_b128 v[148:151], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s41 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[26:29], v[30:33], a[0:3], v152, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s41 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s41 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:85], v[30:33], a[4:7], v152, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s41 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[30:33], a[8:11], v153, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[30:33], a[12:15], v153, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[30:33], a[16:19], v154, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[30:33], a[20:23], v154, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[30:33], a[24:27], v155, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[30:33], a[28:31], v155, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:89], v[10:13], a[0:3], v152, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:93], v[10:13], a[4:7], v152, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[10:13], a[8:11], v153, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[10:13], a[12:15], v153, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[10:13], a[16:19], v154, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[10:13], a[20:23], v154, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:147], v[10:13], a[24:27], v155, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[10:13], a[28:31], v155, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v162, 0x2000, v103
	buffer_load_dwordx4 v[78:81], v162, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v162, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:29], v[22:25], a[32:35], v152, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[22:25], a[36:39], v152, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[22:25], a[40:43], v153, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[22:25], a[44:47], v153, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[22:25], a[48:51], v154, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[22:25], a[52:55], v154, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[22:25], a[56:59], v155, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[22:25], a[60:63], v155, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[86:89], v[14:17], a[32:35], v152, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:93], v[14:17], a[36:39], v152, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[14:17], a[40:43], v153, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[14:17], a[44:47], v153, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[14:17], a[48:51], v154, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[14:17], a[52:55], v154, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[14:17], a[56:59], v155, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[14:17], a[60:63], v155, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v162, s[20:23], s16 offen
	buffer_load_dwordx4 v[6:9], v162, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[26:29], v[38:41], a[64:67], v152, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[38:41], a[68:71], v152, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[38:41], a[72:75], v153, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[38:41], a[76:79], v153, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[38:41], a[80:83], v154, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[38:41], a[84:87], v154, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[38:41], a[88:91], v155, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[38:41], a[92:95], v155, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[86:89], v[34:37], a[64:67], v152, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[90:93], v[34:37], a[68:71], v152, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[112:115], v[34:37], a[72:75], v153, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[34:37], a[76:79], v153, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[34:37], a[80:83], v154, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[34:37], a[84:87], v154, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[34:37], a[88:91], v155, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[34:37], a[92:95], v155, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v162, s[20:23], s15 offen
	buffer_load_dwordx4 v[10:13], v162, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[26:29], v[54:57], a[96:99], v152, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[54:57], a[100:103], v152, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[54:57], a[104:107], v153, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[54:57], a[108:111], v153, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[54:57], a[112:115], v154, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[54:57], a[116:119], v154, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[54:57], a[120:123], v155, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[54:57], a[124:127], v155, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[86:89], v[42:45], a[96:99], v152, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[90:93], v[42:45], a[100:103], v152, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[42:45], a[104:107], v153, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[42:45], a[108:111], v153, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[42:45], a[112:115], v154, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[42:45], a[116:119], v154, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:147], v[42:45], a[120:123], v155, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[42:45], a[124:127], v155, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v162, s[20:23], s14 offen
	buffer_load_dwordx4 v[14:17], v162, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s41, 0x280
	buffer_load_dword v163, v96, s[8:11], s36 offen offset:1024
	buffer_load_dword v164, v96, s[8:11], s37 offen offset:1024
	s_barrier
	ds_read_b128 v[82:85], v97 offset:16384
	ds_read2st64_b32 v[156:157], v96 offset0:131 offset1:159
	ds_read_b128 v[86:89], v97 offset:18432
	ds_read_b128 v[90:93], v100 offset:16384
	ds_read_b128 v[104:107], v100 offset:18432
	ds_read_b128 v[108:111], v97 offset:20480
	ds_read_b128 v[112:115], v97 offset:22528
	ds_read_b128 v[116:119], v100 offset:20480
	ds_read_b128 v[120:123], v100 offset:22528
	ds_read_b128 v[124:127], v97 offset:24576
	ds_read2st64_b32 v[158:159], v96 offset0:187 offset1:215
	ds_read_b128 v[128:131], v97 offset:26624
	ds_read_b128 v[140:143], v97 offset:28672
	ds_read_b128 v[132:135], v100 offset:24576
	ds_read_b128 v[144:147], v97 offset:30720
	ds_read_b128 v[136:139], v100 offset:26624
	ds_read_b128 v[148:151], v100 offset:28672
	ds_read_b128 v[152:155], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s41 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:85], v[74:77], a[0:3], v156, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s41 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s41 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[86:89], v[74:77], a[4:7], v156, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s41 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[74:77], a[8:11], v157, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[74:77], a[12:15], v157, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[74:77], a[16:19], v158, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[74:77], a[20:23], v158, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[74:77], a[24:27], v159, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:147], v[74:77], a[28:31], v159, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[46:49], a[0:3], v156, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[46:49], a[4:7], v156, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[116:119], v[46:49], a[8:11], v157, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[120:123], v[46:49], a[12:15], v157, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[46:49], a[16:19], v158, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[46:49], a[20:23], v158, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[148:151], v[46:49], a[24:27], v159, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[152:155], v[46:49], a[28:31], v159, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v162, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[22:25], v162, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[58:61], a[32:35], v156, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:89], v[58:61], a[36:39], v156, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[58:61], a[40:43], v157, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[58:61], a[44:47], v157, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:127], v[58:61], a[48:51], v158, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[58:61], a[52:55], v158, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[58:61], a[56:59], v159, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[58:61], a[60:63], v159, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:93], v[50:53], a[32:35], v156, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[104:107], v[50:53], a[36:39], v156, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[116:119], v[50:53], a[40:43], v157, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[120:123], v[50:53], a[44:47], v157, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[132:135], v[50:53], a[48:51], v158, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[50:53], a[52:55], v158, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[50:53], a[56:59], v159, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[152:155], v[50:53], a[60:63], v159, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v162, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[26:29], v162, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[82:85], v[62:65], a[64:67], v156, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[86:89], v[62:65], a[68:71], v156, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[62:65], a[72:75], v157, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[112:115], v[62:65], a[76:79], v157, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[62:65], a[80:83], v158, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[128:131], v[62:65], a[84:87], v158, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[62:65], a[88:91], v159, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[62:65], a[92:95], v159, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[90:93], v[18:21], a[64:67], v156, v161 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[104:107], v[18:21], a[68:71], v156, v161 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[116:119], v[18:21], a[72:75], v157, v161 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[120:123], v[18:21], a[76:79], v157, v161 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[132:135], v[18:21], a[80:83], v158, v161 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[18:21], a[84:87], v158, v161 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[18:21], a[88:91], v159, v161 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[152:155], v[18:21], a[92:95], v159, v161 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v162, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[18:21], v162, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[66:69], a[96:99], v156, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[86:89], v[66:69], a[100:103], v156, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[66:69], a[104:107], v157, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[66:69], a[108:111], v157, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[66:69], a[112:115], v158, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[66:69], a[116:119], v158, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[66:69], a[120:123], v159, v161 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:147], v[66:69], a[124:127], v159, v161 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[90:93], v[2:5], a[96:99], v156, v161 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[104:107], v[2:5], a[100:103], v156, v161 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[116:119], v[2:5], a[104:107], v157, v161 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[120:123], v[2:5], a[108:111], v157, v161 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[2:5], a[112:115], v158, v161 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[2:5], a[116:119], v158, v161 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[148:151], v[2:5], a[120:123], v159, v161 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[152:155], v[2:5], a[124:127], v159, v161 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v162, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[2:5], v162, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	s_movk_i32 s41, 0x300
	buffer_load_dword v156, v96, s[8:11], s36 offen offset:1280
	buffer_load_dword v157, v96, s[8:11], s37 offen offset:1280
	s_barrier
	ds_read_b128 v[58:61], v97
	ds_read2st64_b32 v[152:153], v96 offset0:132 offset1:160
	ds_read_b128 v[74:77], v97 offset:2048
	ds_read_b128 v[82:85], v100
	ds_read_b128 v[90:93], v100 offset:2048
	ds_read_b128 v[104:107], v97 offset:4096
	ds_read_b128 v[108:111], v97 offset:6144
	ds_read_b128 v[112:115], v100 offset:4096
	ds_read_b128 v[116:119], v100 offset:6144
	ds_read_b128 v[120:123], v97 offset:8192
	ds_read2st64_b32 v[154:155], v96 offset0:188 offset1:216
	ds_read_b128 v[124:127], v97 offset:10240
	ds_read_b128 v[136:139], v97 offset:12288
	ds_read_b128 v[128:131], v100 offset:8192
	ds_read_b128 v[140:143], v97 offset:14336
	ds_read_b128 v[132:135], v100 offset:10240
	ds_read_b128 v[144:147], v100 offset:12288
	ds_read_b128 v[148:151], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s41 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:61], v[78:81], a[0:3], v152, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s41 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s41 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:77], v[78:81], a[4:7], v152, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s41 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[78:81], a[8:11], v153, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[78:81], a[12:15], v153, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[78:81], a[16:19], v154, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[78:81], a[20:23], v154, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[78:81], a[24:27], v155, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[78:81], a[28:31], v155, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:85], v[30:33], a[0:3], v152, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:93], v[30:33], a[4:7], v152, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[30:33], a[8:11], v153, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[30:33], a[12:15], v153, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[30:33], a[16:19], v154, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[30:33], a[20:23], v154, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:147], v[30:33], a[24:27], v155, v163 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[30:33], a[28:31], v155, v163 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v158, 0x3000, v103
	buffer_load_dwordx4 v[86:89], v158, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v158, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[58:61], v[70:73], a[32:35], v152, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[70:73], a[36:39], v152, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[70:73], a[40:43], v153, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[70:73], a[44:47], v153, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[70:73], a[48:51], v154, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[70:73], a[52:55], v154, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[70:73], a[56:59], v155, v163 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[70:73], a[60:63], v155, v163 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[6:9], a[32:35], v152, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:93], v[6:9], a[36:39], v152, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[6:9], a[40:43], v153, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[6:9], a[44:47], v153, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[6:9], a[48:51], v154, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[6:9], a[52:55], v154, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[6:9], a[56:59], v155, v163 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[6:9], a[60:63], v155, v163 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v158, s[20:23], s16 offen
	buffer_load_dwordx4 v[6:9], v158, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[58:61], v[34:37], a[64:67], v152, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:77], v[34:37], a[68:71], v152, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[34:37], a[72:75], v153, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[34:37], a[76:79], v153, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[34:37], a[80:83], v154, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[34:37], a[84:87], v154, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[34:37], a[88:91], v155, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[34:37], a[92:95], v155, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[82:85], v[10:13], a[64:67], v152, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[90:93], v[10:13], a[68:71], v152, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[112:115], v[10:13], a[72:75], v153, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[10:13], a[76:79], v153, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[10:13], a[80:83], v154, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[10:13], a[84:87], v154, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[10:13], a[88:91], v155, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[10:13], a[92:95], v155, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v158, s[20:23], s15 offen
	buffer_load_dwordx4 v[10:13], v158, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:61], v[38:41], a[96:99], v152, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:77], v[38:41], a[100:103], v152, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[38:41], a[104:107], v153, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[38:41], a[108:111], v153, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[38:41], a[112:115], v154, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[38:41], a[116:119], v154, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[38:41], a[120:123], v155, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[38:41], a[124:127], v155, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[14:17], a[96:99], v152, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[90:93], v[14:17], a[100:103], v152, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[14:17], a[104:107], v153, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[14:17], a[108:111], v153, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[14:17], a[112:115], v154, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[14:17], a[116:119], v154, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:147], v[14:17], a[120:123], v155, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[14:17], a[124:127], v155, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v158, s[20:23], s14 offen
	buffer_load_dwordx4 v[14:17], v158, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s41, 0x380
	buffer_load_dword v159, v96, s[8:11], s36 offen offset:1536
	buffer_load_dword v160, v96, s[8:11], s37 offen offset:1536
	s_barrier
	ds_read_b128 v[34:37], v97 offset:16384
	ds_read2st64_b32 v[152:153], v96 offset0:133 offset1:161
	ds_read_b128 v[38:41], v97 offset:18432
	ds_read_b128 v[58:61], v100 offset:16384
	ds_read_b128 v[82:85], v100 offset:18432
	ds_read_b128 v[104:107], v97 offset:20480
	ds_read_b128 v[108:111], v97 offset:22528
	ds_read_b128 v[112:115], v100 offset:20480
	ds_read_b128 v[116:119], v100 offset:22528
	ds_read_b128 v[120:123], v97 offset:24576
	ds_read2st64_b32 v[154:155], v96 offset0:189 offset1:217
	ds_read_b128 v[124:127], v97 offset:26624
	ds_read_b128 v[136:139], v97 offset:28672
	ds_read_b128 v[128:131], v100 offset:24576
	ds_read_b128 v[140:143], v97 offset:30720
	ds_read_b128 v[132:135], v100 offset:26624
	ds_read_b128 v[144:147], v100 offset:28672
	ds_read_b128 v[148:151], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s41 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[34:37], v[54:57], a[0:3], v152, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s41 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s41 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[38:41], v[54:57], a[4:7], v152, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s41 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[54:57], a[8:11], v153, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[54:57], a[12:15], v153, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[54:57], a[16:19], v154, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[54:57], a[20:23], v154, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[54:57], a[24:27], v155, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[54:57], a[28:31], v155, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:61], v[22:25], a[0:3], v152, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:85], v[22:25], a[4:7], v152, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[22:25], a[8:11], v153, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[22:25], a[12:15], v153, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[22:25], a[16:19], v154, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[22:25], a[20:23], v154, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:147], v[22:25], a[24:27], v155, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[22:25], a[28:31], v155, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[90:93], v158, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[22:25], v158, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[34:37], v[42:45], a[32:35], v152, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[38:41], v[42:45], a[36:39], v152, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[42:45], a[40:43], v153, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[42:45], a[44:47], v153, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[42:45], a[48:51], v154, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[42:45], a[52:55], v154, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[42:45], a[56:59], v155, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[42:45], a[60:63], v155, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[58:61], v[26:29], a[32:35], v152, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[26:29], a[36:39], v152, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[26:29], a[40:43], v153, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[26:29], a[44:47], v153, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[26:29], a[48:51], v154, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[26:29], a[52:55], v154, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[26:29], a[56:59], v155, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[26:29], a[60:63], v155, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[74:77], v158, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[26:29], v158, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[34:37], v[46:49], a[64:67], v152, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[38:41], v[46:49], a[68:71], v152, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[46:49], a[72:75], v153, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[46:49], a[76:79], v153, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[46:49], a[80:83], v154, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[46:49], a[84:87], v154, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[46:49], a[88:91], v155, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[46:49], a[92:95], v155, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[58:61], v[18:21], a[64:67], v152, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[18:21], a[68:71], v152, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[112:115], v[18:21], a[72:75], v153, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[18:21], a[76:79], v153, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[18:21], a[80:83], v154, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[18:21], a[84:87], v154, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[18:21], a[88:91], v155, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[18:21], a[92:95], v155, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[78:81], v158, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[54:57], v158, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[34:37], v[50:53], a[96:99], v152, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[38:41], v[50:53], a[100:103], v152, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[50:53], a[104:107], v153, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[50:53], a[108:111], v153, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[50:53], a[112:115], v154, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[50:53], a[116:119], v154, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[50:53], a[120:123], v155, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[50:53], a[124:127], v155, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:61], v[2:5], a[96:99], v152, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[2:5], a[100:103], v152, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[2:5], a[104:107], v153, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[2:5], a[108:111], v153, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[2:5], a[112:115], v154, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[2:5], a[116:119], v154, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:147], v[2:5], a[120:123], v155, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[2:5], a[124:127], v155, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[82:85], v158, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[58:61], v158, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	buffer_load_dword v164, v96, s[8:11], s36 offen offset:1792
	buffer_load_dword v165, v96, s[8:11], s37 offen offset:1792
	s_barrier
	ds_read_b128 v[18:21], v97
	ds_read2st64_b32 v[152:153], v96 offset0:134 offset1:162
	ds_read_b128 v[42:45], v97 offset:2048
	ds_read_b128 v[46:49], v100
	ds_read_b128 v[50:53], v100 offset:2048
	ds_read_b128 v[104:107], v97 offset:4096
	ds_read_b128 v[108:111], v97 offset:6144
	ds_read_b128 v[112:115], v100 offset:4096
	ds_read_b128 v[116:119], v100 offset:6144
	ds_read_b128 v[120:123], v97 offset:8192
	ds_read2st64_b32 v[154:155], v96 offset0:190 offset1:218
	ds_read_b128 v[124:127], v97 offset:10240
	ds_read_b128 v[136:139], v97 offset:12288
	ds_read_b128 v[128:131], v100 offset:8192
	ds_read_b128 v[140:143], v97 offset:14336
	ds_read_b128 v[132:135], v100 offset:10240
	ds_read_b128 v[144:147], v100 offset:12288
	ds_read_b128 v[148:151], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s40 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[18:21], v[86:89], a[0:3], v152, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s40 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s40 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[42:45], v[86:89], a[4:7], v152, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s40 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[86:89], a[8:11], v153, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[86:89], a[12:15], v153, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[86:89], a[16:19], v154, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[86:89], a[20:23], v154, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[86:89], a[24:27], v155, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[86:89], a[28:31], v155, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[30:33], a[0:3], v152, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[50:53], v[30:33], a[4:7], v152, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[30:33], a[8:11], v153, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[30:33], a[12:15], v153, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[30:33], a[16:19], v154, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[30:33], a[20:23], v154, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:147], v[30:33], a[24:27], v155, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[30:33], a[28:31], v155, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v166, 0x4000, v103
	buffer_load_dwordx4 v[86:89], v166, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v166, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[18:21], v[62:65], a[32:35], v152, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[42:45], v[62:65], a[36:39], v152, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[62:65], a[40:43], v153, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[62:65], a[44:47], v153, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[62:65], a[48:51], v154, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[62:65], a[52:55], v154, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[62:65], a[56:59], v155, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[62:65], a[60:63], v155, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[46:49], v[6:9], a[32:35], v152, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[50:53], v[6:9], a[36:39], v152, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[6:9], a[40:43], v153, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[6:9], a[44:47], v153, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[6:9], a[48:51], v154, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[6:9], a[52:55], v154, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[6:9], a[56:59], v155, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[6:9], a[60:63], v155, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v166, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v166, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[18:21], v[66:69], a[64:67], v152, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[42:45], v[66:69], a[68:71], v152, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[66:69], a[72:75], v153, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[66:69], a[76:79], v153, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[66:69], a[80:83], v154, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[66:69], a[84:87], v154, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[66:69], a[88:91], v155, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[66:69], a[92:95], v155, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[46:49], v[10:13], a[64:67], v152, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[50:53], v[10:13], a[68:71], v152, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[112:115], v[10:13], a[72:75], v153, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[10:13], a[76:79], v153, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[10:13], a[80:83], v154, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[10:13], a[84:87], v154, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[10:13], a[88:91], v155, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[10:13], a[92:95], v155, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v166, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v166, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[18:21], v[70:73], a[96:99], v152, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[42:45], v[70:73], a[100:103], v152, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[70:73], a[104:107], v153, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[70:73], a[108:111], v153, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[70:73], a[112:115], v154, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[70:73], a[116:119], v154, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[70:73], a[120:123], v155, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[70:73], a[124:127], v155, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[46:49], v[14:17], a[96:99], v152, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[50:53], v[14:17], a[100:103], v152, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[14:17], a[104:107], v153, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[14:17], a[108:111], v153, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[14:17], a[112:115], v154, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[14:17], a[116:119], v154, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:147], v[14:17], a[120:123], v155, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[14:17], a[124:127], v155, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v166, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v166, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s40, 0x480
	buffer_load_dword v167, v96, s[8:11], s36 offen offset:2048
	buffer_load_dword v168, v96, s[8:11], s37 offen offset:2048
	s_barrier
	ds_read_b128 v[66:69], v97 offset:16384
	ds_read2st64_b32 v[160:161], v96 offset0:135 offset1:163
	ds_read_b128 v[70:73], v97 offset:18432
	ds_read_b128 v[104:107], v100 offset:16384
	ds_read_b128 v[108:111], v100 offset:18432
	ds_read_b128 v[112:115], v97 offset:20480
	ds_read_b128 v[116:119], v97 offset:22528
	ds_read_b128 v[120:123], v100 offset:20480
	ds_read_b128 v[124:127], v100 offset:22528
	ds_read_b128 v[128:131], v97 offset:24576
	ds_read2st64_b32 v[162:163], v96 offset0:191 offset1:219
	ds_read_b128 v[132:135], v97 offset:26624
	ds_read_b128 v[144:147], v97 offset:28672
	ds_read_b128 v[136:139], v100 offset:24576
	ds_read_b128 v[148:151], v97 offset:30720
	ds_read_b128 v[140:143], v100 offset:26624
	ds_read_b128 v[152:155], v100 offset:28672
	ds_read_b128 v[156:159], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s40 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[90:93], a[0:3], v160, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s40 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s40 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[90:93], a[4:7], v160, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s40 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[90:93], a[8:11], v161, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[90:93], a[12:15], v161, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[90:93], a[16:19], v162, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[90:93], a[20:23], v162, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:147], v[90:93], a[24:27], v163, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[90:93], a[28:31], v163, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[104:107], v[22:25], a[0:3], v160, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[108:111], v[22:25], a[4:7], v160, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[120:123], v[22:25], a[8:11], v161, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[22:25], a[12:15], v161, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[22:25], a[16:19], v162, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[22:25], a[20:23], v162, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[152:155], v[22:25], a[24:27], v163, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[156:159], v[22:25], a[28:31], v163, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v166, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v166, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[74:77], a[32:35], v160, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[74:77], a[36:39], v160, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[74:77], a[40:43], v161, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[74:77], a[44:47], v161, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[74:77], a[48:51], v162, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[74:77], a[52:55], v162, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[74:77], a[56:59], v163, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[74:77], a[60:63], v163, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[104:107], v[26:29], a[32:35], v160, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:111], v[26:29], a[36:39], v160, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[120:123], v[26:29], a[40:43], v161, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[26:29], a[44:47], v161, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[26:29], a[48:51], v162, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[140:143], v[26:29], a[52:55], v162, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[152:155], v[26:29], a[56:59], v163, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[156:159], v[26:29], a[60:63], v163, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v166, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[18:21], v166, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[78:81], a[64:67], v160, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[78:81], a[68:71], v160, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[112:115], v[78:81], a[72:75], v161, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[78:81], a[76:79], v161, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[78:81], a[80:83], v162, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[78:81], a[84:87], v162, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[78:81], a[88:91], v163, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[78:81], a[92:95], v163, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[104:107], v[54:57], a[64:67], v160, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:111], v[54:57], a[68:71], v160, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[120:123], v[54:57], a[72:75], v161, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[54:57], a[76:79], v161, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[136:139], v[54:57], a[80:83], v162, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[54:57], a[84:87], v162, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[54:57], a[88:91], v163, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[156:159], v[54:57], a[92:95], v163, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v166, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[22:25], v166, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[82:85], a[96:99], v160, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[82:85], a[100:103], v160, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[82:85], a[104:107], v161, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[82:85], a[108:111], v161, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[82:85], a[112:115], v162, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[82:85], a[116:119], v162, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:147], v[82:85], a[120:123], v163, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[82:85], a[124:127], v163, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[104:107], v[58:61], a[96:99], v160, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[108:111], v[58:61], a[100:103], v160, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[120:123], v[58:61], a[104:107], v161, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:127], v[58:61], a[108:111], v161, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[58:61], a[112:115], v162, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[58:61], a[116:119], v162, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[152:155], v[58:61], a[120:123], v163, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[156:159], v[58:61], a[124:127], v163, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v166, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[26:29], v166, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	s_movk_i32 s40, 0x500
	buffer_load_dword v148, v96, s[8:11], s36 offen offset:2304
	buffer_load_dword v149, v96, s[8:11], s37 offen offset:2304
	s_barrier
	ds_read_b128 v[58:61], v97
	ds_read2st64_b32 v[144:145], v96 offset0:136 offset1:164
	ds_read_b128 v[70:73], v97 offset:2048
	ds_read_b128 v[74:77], v100
	ds_read_b128 v[78:81], v100 offset:2048
	ds_read_b128 v[82:85], v97 offset:4096
	ds_read_b128 v[90:93], v97 offset:6144
	ds_read_b128 v[104:107], v100 offset:4096
	ds_read_b128 v[108:111], v100 offset:6144
	ds_read_b128 v[112:115], v97 offset:8192
	ds_read2st64_b32 v[146:147], v96 offset0:192 offset1:220
	ds_read_b128 v[116:119], v97 offset:10240
	ds_read_b128 v[128:131], v97 offset:12288
	ds_read_b128 v[120:123], v100 offset:8192
	ds_read_b128 v[132:135], v97 offset:14336
	ds_read_b128 v[124:127], v100 offset:10240
	ds_read_b128 v[136:139], v100 offset:12288
	ds_read_b128 v[140:143], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s40 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[58:61], v[86:89], a[0:3], v144, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s40 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s40 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[86:89], a[4:7], v144, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s40 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[86:89], a[8:11], v145, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:93], v[86:89], a[12:15], v145, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[86:89], a[16:19], v146, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:119], v[86:89], a[20:23], v146, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[86:89], a[24:27], v147, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[86:89], a[28:31], v147, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[30:33], a[0:3], v144, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[30:33], a[4:7], v144, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[30:33], a[8:11], v145, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[30:33], a[12:15], v145, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[30:33], a[16:19], v146, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[30:33], a[20:23], v146, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[30:33], a[24:27], v147, v167 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[30:33], a[28:31], v147, v167 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v150, 0x5000, v103
	buffer_load_dwordx4 v[66:69], v150, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v150, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[58:61], v[34:37], a[32:35], v144, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[34:37], a[36:39], v144, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[34:37], a[40:43], v145, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:93], v[34:37], a[44:47], v145, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[34:37], a[48:51], v146, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:119], v[34:37], a[52:55], v146, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[128:131], v[34:37], a[56:59], v147, v167 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[34:37], a[60:63], v147, v167 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[2:5], a[32:35], v144, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[2:5], a[36:39], v144, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[2:5], a[40:43], v145, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[2:5], a[44:47], v145, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[2:5], a[48:51], v146, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[2:5], a[52:55], v146, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[2:5], a[56:59], v147, v167 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[2:5], a[60:63], v147, v167 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v150, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v150, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[58:61], v[38:41], a[64:67], v144, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[38:41], a[68:71], v144, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[38:41], a[72:75], v145, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[90:93], v[38:41], a[76:79], v145, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:115], v[38:41], a[80:83], v146, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[116:119], v[38:41], a[84:87], v146, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[128:131], v[38:41], a[88:91], v147, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[38:41], a[92:95], v147, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[6:9], a[64:67], v144, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[6:9], a[68:71], v144, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[6:9], a[72:75], v145, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[6:9], a[76:79], v145, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[6:9], a[80:83], v146, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[6:9], a[84:87], v146, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[6:9], a[88:91], v147, v168 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[6:9], a[92:95], v147, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v150, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v150, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[58:61], v[42:45], a[96:99], v144, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[42:45], a[100:103], v144, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[42:45], a[104:107], v145, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[90:93], v[42:45], a[108:111], v145, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[42:45], a[112:115], v146, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:119], v[42:45], a[116:119], v146, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[128:131], v[42:45], a[120:123], v147, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[42:45], a[124:127], v147, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[10:13], a[96:99], v144, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[10:13], a[100:103], v144, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[10:13], a[104:107], v145, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[10:13], a[108:111], v145, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[10:13], a[112:115], v146, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[10:13], a[116:119], v146, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[10:13], a[120:123], v147, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[10:13], a[124:127], v147, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v150, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v150, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s40, 0x580
	buffer_load_dword v151, v96, s[8:11], s36 offen offset:2560
	buffer_load_dword v152, v96, s[8:11], s37 offen offset:2560
	s_barrier
	ds_read_b128 v[70:73], v97 offset:16384
	ds_read2st64_b32 v[144:145], v96 offset0:137 offset1:165
	ds_read_b128 v[74:77], v97 offset:18432
	ds_read_b128 v[78:81], v100 offset:16384
	ds_read_b128 v[82:85], v100 offset:18432
	ds_read_b128 v[86:89], v97 offset:20480
	ds_read_b128 v[90:93], v97 offset:22528
	ds_read_b128 v[104:107], v100 offset:20480
	ds_read_b128 v[108:111], v100 offset:22528
	ds_read_b128 v[112:115], v97 offset:24576
	ds_read2st64_b32 v[146:147], v96 offset0:193 offset1:221
	ds_read_b128 v[116:119], v97 offset:26624
	ds_read_b128 v[128:131], v97 offset:28672
	ds_read_b128 v[120:123], v100 offset:24576
	ds_read_b128 v[132:135], v97 offset:30720
	ds_read_b128 v[124:127], v100 offset:26624
	ds_read_b128 v[136:139], v100 offset:28672
	ds_read_b128 v[140:143], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s40 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[62:65], a[0:3], v144, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s40 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s40 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:77], v[62:65], a[4:7], v144, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s40 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:89], v[62:65], a[8:11], v145, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:93], v[62:65], a[12:15], v145, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[62:65], a[16:19], v146, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:119], v[62:65], a[20:23], v146, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[62:65], a[24:27], v147, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[62:65], a[28:31], v147, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[14:17], a[0:3], v144, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:85], v[14:17], a[4:7], v144, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[14:17], a[8:11], v145, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[14:17], a[12:15], v145, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[14:17], a[16:19], v146, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[14:17], a[20:23], v146, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[14:17], a[24:27], v147, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[14:17], a[28:31], v147, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v150, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v150, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[46:49], a[32:35], v144, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[46:49], a[36:39], v144, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[86:89], v[46:49], a[40:43], v145, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:93], v[46:49], a[44:47], v145, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[46:49], a[48:51], v146, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:119], v[46:49], a[52:55], v146, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[128:131], v[46:49], a[56:59], v147, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[46:49], a[60:63], v147, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[18:21], a[32:35], v144, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[18:21], a[36:39], v144, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[18:21], a[40:43], v145, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[18:21], a[44:47], v145, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[18:21], a[48:51], v146, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[18:21], a[52:55], v146, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[18:21], a[56:59], v147, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[18:21], a[60:63], v147, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v150, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[18:21], v150, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[70:73], v[50:53], a[64:67], v144, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:77], v[50:53], a[68:71], v144, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[86:89], v[50:53], a[72:75], v145, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[90:93], v[50:53], a[76:79], v145, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:115], v[50:53], a[80:83], v146, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[116:119], v[50:53], a[84:87], v146, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[128:131], v[50:53], a[88:91], v147, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[50:53], a[92:95], v147, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:81], v[22:25], a[64:67], v144, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[22:25], a[68:71], v144, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[22:25], a[72:75], v145, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[22:25], a[76:79], v145, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[22:25], a[80:83], v146, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[22:25], a[84:87], v146, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[22:25], a[88:91], v147, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[22:25], a[92:95], v147, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v150, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[22:25], v150, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[70:73], v[54:57], a[96:99], v144, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:77], v[54:57], a[100:103], v144, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[86:89], v[54:57], a[104:107], v145, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[90:93], v[54:57], a[108:111], v145, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[54:57], a[112:115], v146, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:119], v[54:57], a[116:119], v146, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[128:131], v[54:57], a[120:123], v147, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[54:57], a[124:127], v147, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[26:29], a[96:99], v144, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[26:29], a[100:103], v144, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[26:29], a[104:107], v145, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[26:29], a[108:111], v145, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[26:29], a[112:115], v146, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[26:29], a[116:119], v146, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[26:29], a[120:123], v147, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[26:29], a[124:127], v147, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v150, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[26:29], v150, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	s_movk_i32 s40, 0x600
	buffer_load_dword v148, v96, s[8:11], s36 offen offset:2816
	buffer_load_dword v149, v96, s[8:11], s37 offen offset:2816
	s_barrier
	ds_read_b128 v[70:73], v97
	ds_read2st64_b32 v[144:145], v96 offset0:138 offset1:166
	ds_read_b128 v[74:77], v97 offset:2048
	ds_read_b128 v[78:81], v100
	ds_read_b128 v[82:85], v100 offset:2048
	ds_read_b128 v[86:89], v97 offset:4096
	ds_read_b128 v[90:93], v97 offset:6144
	ds_read_b128 v[104:107], v100 offset:4096
	ds_read_b128 v[108:111], v100 offset:6144
	ds_read_b128 v[112:115], v97 offset:8192
	ds_read2st64_b32 v[146:147], v96 offset0:194 offset1:222
	ds_read_b128 v[116:119], v97 offset:10240
	ds_read_b128 v[128:131], v97 offset:12288
	ds_read_b128 v[120:123], v100 offset:8192
	ds_read_b128 v[132:135], v97 offset:14336
	ds_read_b128 v[124:127], v100 offset:10240
	ds_read_b128 v[136:139], v100 offset:12288
	ds_read_b128 v[140:143], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s40 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[66:69], a[0:3], v144, v151 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s40 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s40 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:77], v[66:69], a[4:7], v144, v151 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s40 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:89], v[66:69], a[8:11], v145, v151 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:93], v[66:69], a[12:15], v145, v151 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[66:69], a[16:19], v146, v151 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:119], v[66:69], a[20:23], v146, v151 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[66:69], a[24:27], v147, v151 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[66:69], a[28:31], v147, v151 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[30:33], a[0:3], v144, v151 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:85], v[30:33], a[4:7], v144, v151 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[30:33], a[8:11], v145, v151 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[30:33], a[12:15], v145, v151 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[30:33], a[16:19], v146, v151 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[30:33], a[20:23], v146, v151 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[30:33], a[24:27], v147, v151 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[30:33], a[28:31], v147, v151 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v150, 0x6000, v103
	buffer_load_dwordx4 v[62:65], v150, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v150, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[34:37], a[32:35], v144, v151 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[34:37], a[36:39], v144, v151 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[86:89], v[34:37], a[40:43], v145, v151 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:93], v[34:37], a[44:47], v145, v151 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[34:37], a[48:51], v146, v151 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:119], v[34:37], a[52:55], v146, v151 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[128:131], v[34:37], a[56:59], v147, v151 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[34:37], a[60:63], v147, v151 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[2:5], a[32:35], v144, v151 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[2:5], a[36:39], v144, v151 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[2:5], a[40:43], v145, v151 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[2:5], a[44:47], v145, v151 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[2:5], a[48:51], v146, v151 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[2:5], a[52:55], v146, v151 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[2:5], a[56:59], v147, v151 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[2:5], a[60:63], v147, v151 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v150, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v150, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[70:73], v[38:41], a[64:67], v144, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:77], v[38:41], a[68:71], v144, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[86:89], v[38:41], a[72:75], v145, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[90:93], v[38:41], a[76:79], v145, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:115], v[38:41], a[80:83], v146, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[116:119], v[38:41], a[84:87], v146, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[128:131], v[38:41], a[88:91], v147, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[38:41], a[92:95], v147, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:81], v[6:9], a[64:67], v144, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[6:9], a[68:71], v144, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[6:9], a[72:75], v145, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[6:9], a[76:79], v145, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[6:9], a[80:83], v146, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[6:9], a[84:87], v146, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[6:9], a[88:91], v147, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[6:9], a[92:95], v147, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v150, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v150, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[70:73], v[42:45], a[96:99], v144, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:77], v[42:45], a[100:103], v144, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[86:89], v[42:45], a[104:107], v145, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[90:93], v[42:45], a[108:111], v145, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[42:45], a[112:115], v146, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:119], v[42:45], a[116:119], v146, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[128:131], v[42:45], a[120:123], v147, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[42:45], a[124:127], v147, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[10:13], a[96:99], v144, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[10:13], a[100:103], v144, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[10:13], a[104:107], v145, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[10:13], a[108:111], v145, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[10:13], a[112:115], v146, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[10:13], a[116:119], v146, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[10:13], a[120:123], v147, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[10:13], a[124:127], v147, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v150, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v150, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s40, 0x680
	buffer_load_dword v144, v96, s[8:11], s36 offen offset:3072
	buffer_load_dword v145, v96, s[8:11], s37 offen offset:3072
	s_barrier
	ds_read_b128 v[66:69], v97 offset:16384
	ds_read2st64_b32 v[140:141], v96 offset0:139 offset1:167
	ds_read_b128 v[70:73], v97 offset:18432
	ds_read_b128 v[74:77], v100 offset:16384
	ds_read_b128 v[78:81], v100 offset:18432
	ds_read_b128 v[82:85], v97 offset:20480
	ds_read_b128 v[86:89], v97 offset:22528
	ds_read_b128 v[90:93], v100 offset:20480
	ds_read_b128 v[104:107], v100 offset:22528
	ds_read_b128 v[108:111], v97 offset:24576
	ds_read2st64_b32 v[142:143], v96 offset0:195 offset1:223
	ds_read_b128 v[112:115], v97 offset:26624
	ds_read_b128 v[124:127], v97 offset:28672
	ds_read_b128 v[116:119], v100 offset:24576
	ds_read_b128 v[128:131], v97 offset:30720
	ds_read_b128 v[120:123], v100 offset:26624
	ds_read_b128 v[132:135], v100 offset:28672
	ds_read_b128 v[136:139], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s40 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[58:61], a[0:3], v140, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s40 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s40 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[58:61], a[4:7], v140, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s40 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[58:61], a[8:11], v141, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[58:61], a[12:15], v141, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[58:61], a[16:19], v142, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[58:61], a[20:23], v142, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v143, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[58:61], a[28:31], v143, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[14:17], a[0:3], v140, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[14:17], a[4:7], v140, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[14:17], a[8:11], v141, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[14:17], a[12:15], v141, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[14:17], a[16:19], v142, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[14:17], a[20:23], v142, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[14:17], a[24:27], v143, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[14:17], a[28:31], v143, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v150, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v150, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[46:49], a[32:35], v140, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[46:49], a[36:39], v140, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[46:49], a[40:43], v141, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[46:49], a[44:47], v141, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[46:49], a[48:51], v142, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[46:49], a[52:55], v142, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[46:49], a[56:59], v143, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[46:49], a[60:63], v143, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[18:21], a[32:35], v140, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[18:21], a[36:39], v140, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[18:21], a[40:43], v141, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[18:21], a[44:47], v141, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[18:21], a[48:51], v142, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[18:21], a[52:55], v142, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[18:21], a[56:59], v143, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v143, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v150, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[18:21], v150, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[50:53], a[64:67], v140, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[50:53], a[68:71], v140, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[50:53], a[72:75], v141, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[50:53], a[76:79], v141, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[50:53], a[80:83], v142, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[50:53], a[84:87], v142, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[50:53], a[88:91], v143, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[50:53], a[92:95], v143, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[22:25], a[64:67], v140, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[22:25], a[68:71], v140, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[22:25], a[72:75], v141, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[22:25], a[76:79], v141, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[22:25], a[80:83], v142, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[22:25], a[84:87], v142, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[22:25], a[88:91], v143, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v143, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v150, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[22:25], v150, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[54:57], a[96:99], v140, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[54:57], a[100:103], v140, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[54:57], a[104:107], v141, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[54:57], a[108:111], v141, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[54:57], a[112:115], v142, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[54:57], a[116:119], v142, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[54:57], a[120:123], v143, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[54:57], a[124:127], v143, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[26:29], a[96:99], v140, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[26:29], a[100:103], v140, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[26:29], a[104:107], v141, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[26:29], a[108:111], v141, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[26:29], a[112:115], v142, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[26:29], a[116:119], v142, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[26:29], a[120:123], v143, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[26:29], a[124:127], v143, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v150, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[26:29], v150, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	buffer_load_dword v146, v96, s[8:11], s36 offen offset:3328
	buffer_load_dword v147, v96, s[8:11], s37 offen offset:3328
	s_barrier
	ds_read_b128 v[66:69], v97
	ds_read2st64_b32 v[140:141], v96 offset0:140 offset1:168
	ds_read_b128 v[70:73], v97 offset:2048
	ds_read_b128 v[74:77], v100
	ds_read_b128 v[78:81], v100 offset:2048
	ds_read_b128 v[82:85], v97 offset:4096
	ds_read_b128 v[86:89], v97 offset:6144
	ds_read_b128 v[90:93], v100 offset:4096
	ds_read_b128 v[104:107], v100 offset:6144
	ds_read_b128 v[108:111], v97 offset:8192
	ds_read2st64_b32 v[142:143], v96 offset0:196 offset1:224
	ds_read_b128 v[112:115], v97 offset:10240
	ds_read_b128 v[124:127], v97 offset:12288
	ds_read_b128 v[116:119], v100 offset:8192
	ds_read_b128 v[128:131], v97 offset:14336
	ds_read_b128 v[120:123], v100 offset:10240
	ds_read_b128 v[132:135], v100 offset:12288
	ds_read_b128 v[136:139], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s39 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[62:65], a[0:3], v140, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s39 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s39 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[62:65], a[4:7], v140, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s39 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[62:65], a[8:11], v141, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[62:65], a[12:15], v141, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[62:65], a[16:19], v142, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[62:65], a[20:23], v142, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[62:65], a[24:27], v143, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[62:65], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[30:33], a[0:3], v140, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[30:33], a[4:7], v140, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[30:33], a[8:11], v141, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[30:33], a[12:15], v141, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[30:33], a[16:19], v142, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[30:33], a[20:23], v142, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[30:33], a[24:27], v143, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[30:33], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v148, 0x7000, v103
	buffer_load_dwordx4 v[62:65], v148, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v148, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[34:37], a[32:35], v140, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[34:37], a[36:39], v140, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[34:37], a[40:43], v141, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[34:37], a[44:47], v141, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[34:37], a[48:51], v142, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[34:37], a[52:55], v142, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[34:37], a[56:59], v143, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[34:37], a[60:63], v143, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[2:5], a[32:35], v140, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[2:5], a[36:39], v140, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[2:5], a[40:43], v141, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[2:5], a[44:47], v141, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[2:5], a[48:51], v142, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[2:5], a[52:55], v142, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[2:5], a[56:59], v143, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[2:5], a[60:63], v143, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v148, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v148, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[38:41], a[64:67], v140, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[38:41], a[68:71], v140, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[38:41], a[72:75], v141, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[38:41], a[76:79], v141, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[38:41], a[80:83], v142, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[38:41], a[84:87], v142, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[38:41], a[88:91], v143, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[38:41], a[92:95], v143, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[6:9], a[64:67], v140, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[6:9], a[68:71], v140, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[6:9], a[72:75], v141, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[6:9], a[76:79], v141, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[6:9], a[80:83], v142, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[6:9], a[84:87], v142, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[6:9], a[88:91], v143, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[6:9], a[92:95], v143, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v148, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v148, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[42:45], a[96:99], v140, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[42:45], a[100:103], v140, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[42:45], a[104:107], v141, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[42:45], a[108:111], v141, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[42:45], a[112:115], v142, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[42:45], a[116:119], v142, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[42:45], a[120:123], v143, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[42:45], a[124:127], v143, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[10:13], a[96:99], v140, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[10:13], a[100:103], v140, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[10:13], a[104:107], v141, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[10:13], a[108:111], v141, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[10:13], a[112:115], v142, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[10:13], a[116:119], v142, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[10:13], a[120:123], v143, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[10:13], a[124:127], v143, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v148, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v148, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s39, 0x780
	buffer_load_dword v144, v96, s[8:11], s36 offen offset:3584
	buffer_load_dword v145, v96, s[8:11], s37 offen offset:3584
	s_barrier
	ds_read_b128 v[66:69], v97 offset:16384
	ds_read2st64_b32 v[140:141], v96 offset0:141 offset1:169
	ds_read_b128 v[70:73], v97 offset:18432
	ds_read_b128 v[74:77], v100 offset:16384
	ds_read_b128 v[78:81], v100 offset:18432
	ds_read_b128 v[82:85], v97 offset:20480
	ds_read_b128 v[86:89], v97 offset:22528
	ds_read_b128 v[90:93], v100 offset:20480
	ds_read_b128 v[104:107], v100 offset:22528
	ds_read_b128 v[108:111], v97 offset:24576
	ds_read2st64_b32 v[142:143], v96 offset0:197 offset1:225
	ds_read_b128 v[112:115], v97 offset:26624
	ds_read_b128 v[124:127], v97 offset:28672
	ds_read_b128 v[116:119], v100 offset:24576
	ds_read_b128 v[128:131], v97 offset:30720
	ds_read_b128 v[120:123], v100 offset:26624
	ds_read_b128 v[132:135], v100 offset:28672
	ds_read_b128 v[136:139], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s39 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[58:61], a[0:3], v140, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s39 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s39 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[58:61], a[4:7], v140, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s39 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[58:61], a[8:11], v141, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[58:61], a[12:15], v141, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[58:61], a[16:19], v142, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[58:61], a[20:23], v142, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v143, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[58:61], a[28:31], v143, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[14:17], a[0:3], v140, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[14:17], a[4:7], v140, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[14:17], a[8:11], v141, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[14:17], a[12:15], v141, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[14:17], a[16:19], v142, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[14:17], a[20:23], v142, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[14:17], a[24:27], v143, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[14:17], a[28:31], v143, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v148, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v148, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[46:49], a[32:35], v140, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[46:49], a[36:39], v140, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[46:49], a[40:43], v141, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[46:49], a[44:47], v141, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[46:49], a[48:51], v142, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[46:49], a[52:55], v142, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[46:49], a[56:59], v143, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[46:49], a[60:63], v143, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[18:21], a[32:35], v140, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[18:21], a[36:39], v140, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[18:21], a[40:43], v141, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[18:21], a[44:47], v141, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[18:21], a[48:51], v142, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[18:21], a[52:55], v142, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[18:21], a[56:59], v143, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v143, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v148, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[18:21], v148, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[50:53], a[64:67], v140, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[50:53], a[68:71], v140, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[50:53], a[72:75], v141, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[50:53], a[76:79], v141, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[50:53], a[80:83], v142, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[50:53], a[84:87], v142, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[50:53], a[88:91], v143, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[50:53], a[92:95], v143, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[22:25], a[64:67], v140, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[22:25], a[68:71], v140, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[22:25], a[72:75], v141, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[22:25], a[76:79], v141, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[22:25], a[80:83], v142, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[22:25], a[84:87], v142, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[22:25], a[88:91], v143, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v143, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v148, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[22:25], v148, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[54:57], a[96:99], v140, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[54:57], a[100:103], v140, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[54:57], a[104:107], v141, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[54:57], a[108:111], v141, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[54:57], a[112:115], v142, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[54:57], a[116:119], v142, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[54:57], a[120:123], v143, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[54:57], a[124:127], v143, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[26:29], a[96:99], v140, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[26:29], a[100:103], v140, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[26:29], a[104:107], v141, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[26:29], a[108:111], v141, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[26:29], a[112:115], v142, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[26:29], a[116:119], v142, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[26:29], a[120:123], v143, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[26:29], a[124:127], v143, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v148, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[26:29], v148, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	buffer_load_dword v146, v96, s[8:11], s36 offen offset:3840
	buffer_load_dword v147, v96, s[8:11], s37 offen offset:3840
	s_barrier
	ds_read_b128 v[66:69], v97
	ds_read2st64_b32 v[140:141], v96 offset0:142 offset1:170
	ds_read_b128 v[70:73], v97 offset:2048
	ds_read_b128 v[74:77], v100
	ds_read_b128 v[78:81], v100 offset:2048
	ds_read_b128 v[82:85], v97 offset:4096
	ds_read_b128 v[86:89], v97 offset:6144
	ds_read_b128 v[90:93], v100 offset:4096
	ds_read_b128 v[104:107], v100 offset:6144
	ds_read_b128 v[108:111], v97 offset:8192
	ds_read2st64_b32 v[142:143], v96 offset0:198 offset1:226
	ds_read_b128 v[112:115], v97 offset:10240
	ds_read_b128 v[124:127], v97 offset:12288
	ds_read_b128 v[116:119], v100 offset:8192
	ds_read_b128 v[128:131], v97 offset:14336
	ds_read_b128 v[120:123], v100 offset:10240
	ds_read_b128 v[132:135], v100 offset:12288
	ds_read_b128 v[136:139], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s38 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[62:65], a[0:3], v140, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s38 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s38 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[62:65], a[4:7], v140, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s38 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[62:65], a[8:11], v141, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[62:65], a[12:15], v141, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[62:65], a[16:19], v142, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[62:65], a[20:23], v142, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[62:65], a[24:27], v143, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[62:65], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[30:33], a[0:3], v140, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[30:33], a[4:7], v140, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[30:33], a[8:11], v141, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[30:33], a[12:15], v141, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[30:33], a[16:19], v142, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[30:33], a[20:23], v142, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[30:33], a[24:27], v143, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[30:33], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v148, 0x8000, v103
	buffer_load_dwordx4 v[62:65], v148, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v148, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[34:37], a[32:35], v140, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[34:37], a[36:39], v140, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[34:37], a[40:43], v141, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[34:37], a[44:47], v141, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[34:37], a[48:51], v142, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[34:37], a[52:55], v142, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[34:37], a[56:59], v143, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[34:37], a[60:63], v143, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[2:5], a[32:35], v140, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[2:5], a[36:39], v140, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[2:5], a[40:43], v141, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[2:5], a[44:47], v141, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[2:5], a[48:51], v142, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[2:5], a[52:55], v142, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[2:5], a[56:59], v143, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[2:5], a[60:63], v143, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v148, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v148, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[38:41], a[64:67], v140, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[38:41], a[68:71], v140, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[38:41], a[72:75], v141, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[38:41], a[76:79], v141, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[38:41], a[80:83], v142, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[38:41], a[84:87], v142, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[38:41], a[88:91], v143, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[38:41], a[92:95], v143, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[6:9], a[64:67], v140, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[6:9], a[68:71], v140, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[6:9], a[72:75], v141, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[6:9], a[76:79], v141, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[6:9], a[80:83], v142, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[6:9], a[84:87], v142, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[6:9], a[88:91], v143, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[6:9], a[92:95], v143, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v148, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v148, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[42:45], a[96:99], v140, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[42:45], a[100:103], v140, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[42:45], a[104:107], v141, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[42:45], a[108:111], v141, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[42:45], a[112:115], v142, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[42:45], a[116:119], v142, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[42:45], a[120:123], v143, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[42:45], a[124:127], v143, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[10:13], a[96:99], v140, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[10:13], a[100:103], v140, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[10:13], a[104:107], v141, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[10:13], a[108:111], v141, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[10:13], a[112:115], v142, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[10:13], a[116:119], v142, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[10:13], a[120:123], v143, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[10:13], a[124:127], v143, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v148, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v148, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s36, 0x880
	buffer_load_dword v144, v96, s[8:11], s19 offen
	buffer_load_dword v145, v96, s[8:11], s28 offen
	s_barrier
	ds_read_b128 v[66:69], v97 offset:16384
	ds_read2st64_b32 v[140:141], v96 offset0:143 offset1:171
	ds_read_b128 v[70:73], v97 offset:18432
	ds_read_b128 v[74:77], v100 offset:16384
	ds_read_b128 v[78:81], v100 offset:18432
	ds_read_b128 v[82:85], v97 offset:20480
	ds_read_b128 v[86:89], v97 offset:22528
	ds_read_b128 v[90:93], v100 offset:20480
	ds_read_b128 v[104:107], v100 offset:22528
	ds_read_b128 v[108:111], v97 offset:24576
	ds_read2st64_b32 v[142:143], v96 offset0:199 offset1:227
	ds_read_b128 v[112:115], v97 offset:26624
	ds_read_b128 v[124:127], v97 offset:28672
	ds_read_b128 v[116:119], v100 offset:24576
	ds_read_b128 v[128:131], v97 offset:30720
	ds_read_b128 v[120:123], v100 offset:26624
	ds_read_b128 v[132:135], v100 offset:28672
	ds_read_b128 v[136:139], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s36 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[58:61], a[0:3], v140, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s36 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s36 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[58:61], a[4:7], v140, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s36 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[58:61], a[8:11], v141, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[58:61], a[12:15], v141, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[58:61], a[16:19], v142, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[58:61], a[20:23], v142, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v143, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[58:61], a[28:31], v143, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[14:17], a[0:3], v140, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[14:17], a[4:7], v140, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[14:17], a[8:11], v141, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[14:17], a[12:15], v141, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[14:17], a[16:19], v142, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[14:17], a[20:23], v142, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[14:17], a[24:27], v143, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[14:17], a[28:31], v143, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v148, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v148, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[46:49], a[32:35], v140, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[46:49], a[36:39], v140, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[46:49], a[40:43], v141, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[46:49], a[44:47], v141, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[46:49], a[48:51], v142, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[46:49], a[52:55], v142, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[46:49], a[56:59], v143, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[46:49], a[60:63], v143, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[18:21], a[32:35], v140, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[18:21], a[36:39], v140, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[18:21], a[40:43], v141, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[18:21], a[44:47], v141, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[18:21], a[48:51], v142, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[18:21], a[52:55], v142, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[18:21], a[56:59], v143, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v143, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v148, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[18:21], v148, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[50:53], a[64:67], v140, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[50:53], a[68:71], v140, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[50:53], a[72:75], v141, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[50:53], a[76:79], v141, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[50:53], a[80:83], v142, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[50:53], a[84:87], v142, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[50:53], a[88:91], v143, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[50:53], a[92:95], v143, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[22:25], a[64:67], v140, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[22:25], a[68:71], v140, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[22:25], a[72:75], v141, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[22:25], a[76:79], v141, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[22:25], a[80:83], v142, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[22:25], a[84:87], v142, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[22:25], a[88:91], v143, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v143, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v148, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[22:25], v148, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[54:57], a[96:99], v140, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[54:57], a[100:103], v140, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[54:57], a[104:107], v141, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[54:57], a[108:111], v141, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[54:57], a[112:115], v142, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[54:57], a[116:119], v142, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[54:57], a[120:123], v143, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[54:57], a[124:127], v143, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[26:29], a[96:99], v140, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[26:29], a[100:103], v140, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[26:29], a[104:107], v141, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[26:29], a[108:111], v141, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[26:29], a[112:115], v142, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[26:29], a[116:119], v142, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[26:29], a[120:123], v143, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[26:29], a[124:127], v143, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v148, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[26:29], v148, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	s_movk_i32 s36, 0x900
	buffer_load_dword v146, v96, s[8:11], s19 offen offset:256
	buffer_load_dword v147, v96, s[8:11], s28 offen offset:256
	s_barrier
	ds_read_b128 v[66:69], v97
	ds_read2st64_b32 v[140:141], v96 offset0:144 offset1:172
	ds_read_b128 v[70:73], v97 offset:2048
	ds_read_b128 v[74:77], v100
	ds_read_b128 v[78:81], v100 offset:2048
	ds_read_b128 v[82:85], v97 offset:4096
	ds_read_b128 v[86:89], v97 offset:6144
	ds_read_b128 v[90:93], v100 offset:4096
	ds_read_b128 v[104:107], v100 offset:6144
	ds_read_b128 v[108:111], v97 offset:8192
	ds_read2st64_b32 v[142:143], v96 offset0:200 offset1:228
	ds_read_b128 v[112:115], v97 offset:10240
	ds_read_b128 v[124:127], v97 offset:12288
	ds_read_b128 v[116:119], v100 offset:8192
	ds_read_b128 v[128:131], v97 offset:14336
	ds_read_b128 v[120:123], v100 offset:10240
	ds_read_b128 v[132:135], v100 offset:12288
	ds_read_b128 v[136:139], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s36 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[62:65], a[0:3], v140, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s36 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s36 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[62:65], a[4:7], v140, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s36 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[62:65], a[8:11], v141, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[62:65], a[12:15], v141, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[62:65], a[16:19], v142, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[62:65], a[20:23], v142, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[62:65], a[24:27], v143, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[62:65], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[30:33], a[0:3], v140, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[30:33], a[4:7], v140, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[30:33], a[8:11], v141, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[30:33], a[12:15], v141, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[30:33], a[16:19], v142, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[30:33], a[20:23], v142, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[30:33], a[24:27], v143, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[30:33], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v148, 0x9000, v103
	buffer_load_dwordx4 v[62:65], v148, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v148, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[34:37], a[32:35], v140, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[34:37], a[36:39], v140, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[34:37], a[40:43], v141, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[34:37], a[44:47], v141, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[34:37], a[48:51], v142, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[34:37], a[52:55], v142, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[34:37], a[56:59], v143, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[34:37], a[60:63], v143, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[2:5], a[32:35], v140, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[2:5], a[36:39], v140, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[2:5], a[40:43], v141, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[2:5], a[44:47], v141, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[2:5], a[48:51], v142, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[2:5], a[52:55], v142, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[2:5], a[56:59], v143, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[2:5], a[60:63], v143, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v148, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v148, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[38:41], a[64:67], v140, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[38:41], a[68:71], v140, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[38:41], a[72:75], v141, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[38:41], a[76:79], v141, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[38:41], a[80:83], v142, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[38:41], a[84:87], v142, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[38:41], a[88:91], v143, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[38:41], a[92:95], v143, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[6:9], a[64:67], v140, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[6:9], a[68:71], v140, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[6:9], a[72:75], v141, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[6:9], a[76:79], v141, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[6:9], a[80:83], v142, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[6:9], a[84:87], v142, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[6:9], a[88:91], v143, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[6:9], a[92:95], v143, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v148, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v148, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[42:45], a[96:99], v140, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[42:45], a[100:103], v140, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[42:45], a[104:107], v141, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[42:45], a[108:111], v141, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[42:45], a[112:115], v142, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[42:45], a[116:119], v142, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[42:45], a[120:123], v143, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[42:45], a[124:127], v143, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[10:13], a[96:99], v140, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[10:13], a[100:103], v140, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[10:13], a[104:107], v141, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[10:13], a[108:111], v141, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[10:13], a[112:115], v142, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[10:13], a[116:119], v142, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[10:13], a[120:123], v143, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[10:13], a[124:127], v143, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v148, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v148, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s36, 0x980
	buffer_load_dword v144, v96, s[8:11], s19 offen offset:512
	buffer_load_dword v145, v96, s[8:11], s28 offen offset:512
	s_barrier
	ds_read_b128 v[66:69], v97 offset:16384
	ds_read2st64_b32 v[140:141], v96 offset0:145 offset1:173
	ds_read_b128 v[70:73], v97 offset:18432
	ds_read_b128 v[74:77], v100 offset:16384
	ds_read_b128 v[78:81], v100 offset:18432
	ds_read_b128 v[82:85], v97 offset:20480
	ds_read_b128 v[86:89], v97 offset:22528
	ds_read_b128 v[90:93], v100 offset:20480
	ds_read_b128 v[104:107], v100 offset:22528
	ds_read_b128 v[108:111], v97 offset:24576
	ds_read2st64_b32 v[142:143], v96 offset0:201 offset1:229
	ds_read_b128 v[112:115], v97 offset:26624
	ds_read_b128 v[124:127], v97 offset:28672
	ds_read_b128 v[116:119], v100 offset:24576
	ds_read_b128 v[128:131], v97 offset:30720
	ds_read_b128 v[120:123], v100 offset:26624
	ds_read_b128 v[132:135], v100 offset:28672
	ds_read_b128 v[136:139], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s36 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[58:61], a[0:3], v140, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s36 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s36 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[58:61], a[4:7], v140, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s36 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[58:61], a[8:11], v141, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[58:61], a[12:15], v141, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[58:61], a[16:19], v142, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[58:61], a[20:23], v142, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v143, v146 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[58:61], a[28:31], v143, v146 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[14:17], a[0:3], v140, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[14:17], a[4:7], v140, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[14:17], a[8:11], v141, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[14:17], a[12:15], v141, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[14:17], a[16:19], v142, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[14:17], a[20:23], v142, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[14:17], a[24:27], v143, v146 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[14:17], a[28:31], v143, v146 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v148, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v148, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[46:49], a[32:35], v140, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[46:49], a[36:39], v140, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[46:49], a[40:43], v141, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[46:49], a[44:47], v141, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[46:49], a[48:51], v142, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[46:49], a[52:55], v142, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[46:49], a[56:59], v143, v146 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[46:49], a[60:63], v143, v146 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[18:21], a[32:35], v140, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[18:21], a[36:39], v140, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[18:21], a[40:43], v141, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[18:21], a[44:47], v141, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[18:21], a[48:51], v142, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[18:21], a[52:55], v142, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[18:21], a[56:59], v143, v146 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v143, v146 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v148, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[18:21], v148, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[50:53], a[64:67], v140, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[50:53], a[68:71], v140, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[50:53], a[72:75], v141, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[50:53], a[76:79], v141, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[50:53], a[80:83], v142, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[50:53], a[84:87], v142, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[50:53], a[88:91], v143, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[50:53], a[92:95], v143, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[22:25], a[64:67], v140, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[22:25], a[68:71], v140, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[22:25], a[72:75], v141, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[22:25], a[76:79], v141, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[22:25], a[80:83], v142, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[22:25], a[84:87], v142, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[22:25], a[88:91], v143, v147 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v143, v147 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v148, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[22:25], v148, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[54:57], a[96:99], v140, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[54:57], a[100:103], v140, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[54:57], a[104:107], v141, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[54:57], a[108:111], v141, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[54:57], a[112:115], v142, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[54:57], a[116:119], v142, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[54:57], a[120:123], v143, v147 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[54:57], a[124:127], v143, v147 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[26:29], a[96:99], v140, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[26:29], a[100:103], v140, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[26:29], a[104:107], v141, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[26:29], a[108:111], v141, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[26:29], a[112:115], v142, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[26:29], a[116:119], v142, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[26:29], a[120:123], v143, v147 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[26:29], a[124:127], v143, v147 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v148, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[26:29], v148, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	s_movk_i32 s36, 0xa00
	buffer_load_dword v148, v96, s[8:11], s19 offen offset:768
	buffer_load_dword v149, v96, s[8:11], s28 offen offset:768
	s_barrier
	ds_read_b128 v[66:69], v97
	ds_read2st64_b32 v[140:141], v96 offset0:146 offset1:174
	ds_read_b128 v[70:73], v97 offset:2048
	ds_read_b128 v[74:77], v100
	ds_read_b128 v[78:81], v100 offset:2048
	ds_read_b128 v[82:85], v97 offset:4096
	ds_read_b128 v[86:89], v97 offset:6144
	ds_read_b128 v[90:93], v100 offset:4096
	ds_read_b128 v[104:107], v100 offset:6144
	ds_read_b128 v[108:111], v97 offset:8192
	ds_read2st64_b32 v[142:143], v96 offset0:202 offset1:230
	ds_read_b128 v[112:115], v97 offset:10240
	ds_read_b128 v[124:127], v97 offset:12288
	ds_read_b128 v[116:119], v100 offset:8192
	ds_read_b128 v[128:131], v97 offset:14336
	ds_read_b128 v[120:123], v100 offset:10240
	ds_read_b128 v[132:135], v100 offset:12288
	ds_read_b128 v[136:139], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s36 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[62:65], a[0:3], v140, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s36 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s36 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[70:73], v[62:65], a[4:7], v140, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s36 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[62:65], a[8:11], v141, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[62:65], a[12:15], v141, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[62:65], a[16:19], v142, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[62:65], a[20:23], v142, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[62:65], a[24:27], v143, v144 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[62:65], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[30:33], a[0:3], v140, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[30:33], a[4:7], v140, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[30:33], a[8:11], v141, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[30:33], a[12:15], v141, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[30:33], a[16:19], v142, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[30:33], a[20:23], v142, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[30:33], a[24:27], v143, v144 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[30:33], a[28:31], v143, v144 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v150, 0xa000, v103
	buffer_load_dwordx4 v[62:65], v150, s[20:23], s17 offen
	buffer_load_dwordx4 v[30:33], v150, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[34:37], a[32:35], v140, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[34:37], a[36:39], v140, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[34:37], a[40:43], v141, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[34:37], a[44:47], v141, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[34:37], a[48:51], v142, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[34:37], a[52:55], v142, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[34:37], a[56:59], v143, v144 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[34:37], a[60:63], v143, v144 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[2:5], a[32:35], v140, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[2:5], a[36:39], v140, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[2:5], a[40:43], v141, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[2:5], a[44:47], v141, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[2:5], a[48:51], v142, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[2:5], a[52:55], v142, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[2:5], a[56:59], v143, v144 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[2:5], a[60:63], v143, v144 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v150, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v150, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[38:41], a[64:67], v140, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[38:41], a[68:71], v140, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[38:41], a[72:75], v141, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[38:41], a[76:79], v141, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[38:41], a[80:83], v142, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[38:41], a[84:87], v142, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[38:41], a[88:91], v143, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[38:41], a[92:95], v143, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[6:9], a[64:67], v140, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[6:9], a[68:71], v140, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[6:9], a[72:75], v141, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[6:9], a[76:79], v141, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[6:9], a[80:83], v142, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[6:9], a[84:87], v142, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[6:9], a[88:91], v143, v145 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[6:9], a[92:95], v143, v145 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v150, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v150, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[42:45], a[96:99], v140, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[42:45], a[100:103], v140, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[42:45], a[104:107], v141, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[42:45], a[108:111], v141, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[42:45], a[112:115], v142, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[42:45], a[116:119], v142, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[42:45], a[120:123], v143, v145 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[42:45], a[124:127], v143, v145 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[10:13], a[96:99], v140, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[10:13], a[100:103], v140, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[10:13], a[104:107], v141, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[10:13], a[108:111], v141, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[10:13], a[112:115], v142, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[10:13], a[116:119], v142, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[10:13], a[120:123], v143, v145 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[10:13], a[124:127], v143, v145 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v150, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v150, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s36, 0xa80
	buffer_load_dword v152, v96, s[8:11], s19 offen offset:1024
	buffer_load_dword v153, v96, s[8:11], s28 offen offset:1024
	s_barrier
	ds_read_b128 v[70:73], v97 offset:16384
	ds_read2st64_b32 v[144:145], v96 offset0:147 offset1:175
	ds_read_b128 v[74:77], v97 offset:18432
	ds_read_b128 v[78:81], v100 offset:16384
	ds_read_b128 v[82:85], v100 offset:18432
	ds_read_b128 v[86:89], v97 offset:20480
	ds_read_b128 v[90:93], v97 offset:22528
	ds_read_b128 v[104:107], v100 offset:20480
	ds_read_b128 v[108:111], v100 offset:22528
	ds_read_b128 v[112:115], v97 offset:24576
	ds_read2st64_b32 v[146:147], v96 offset0:203 offset1:231
	ds_read_b128 v[116:119], v97 offset:26624
	ds_read_b128 v[128:131], v97 offset:28672
	ds_read_b128 v[120:123], v100 offset:24576
	ds_read_b128 v[132:135], v97 offset:30720
	ds_read_b128 v[124:127], v100 offset:26624
	ds_read_b128 v[136:139], v100 offset:28672
	ds_read_b128 v[140:143], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s36 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[70:73], v[58:61], a[0:3], v144, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s36 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s36 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[74:77], v[58:61], a[4:7], v144, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s36 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:89], v[58:61], a[8:11], v145, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[90:93], v[58:61], a[12:15], v145, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[58:61], a[16:19], v146, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[116:119], v[58:61], a[20:23], v146, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[58:61], a[24:27], v147, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[132:135], v[58:61], a[28:31], v147, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[14:17], a[0:3], v144, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:85], v[14:17], a[4:7], v144, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[14:17], a[8:11], v145, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[14:17], a[12:15], v145, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[14:17], a[16:19], v146, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[14:17], a[20:23], v146, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[14:17], a[24:27], v147, v148 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[14:17], a[28:31], v147, v148 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v150, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v150, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[46:49], a[32:35], v144, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[74:77], v[46:49], a[36:39], v144, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[86:89], v[46:49], a[40:43], v145, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[90:93], v[46:49], a[44:47], v145, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[112:115], v[46:49], a[48:51], v146, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[116:119], v[46:49], a[52:55], v146, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[128:131], v[46:49], a[56:59], v147, v148 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[46:49], a[60:63], v147, v148 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[18:21], a[32:35], v144, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[18:21], a[36:39], v144, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[18:21], a[40:43], v145, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[18:21], a[44:47], v145, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[18:21], a[48:51], v146, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[18:21], a[52:55], v146, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[18:21], a[56:59], v147, v148 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[18:21], a[60:63], v147, v148 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v150, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[18:21], v150, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[70:73], v[50:53], a[64:67], v144, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[74:77], v[50:53], a[68:71], v144, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[86:89], v[50:53], a[72:75], v145, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[90:93], v[50:53], a[76:79], v145, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[112:115], v[50:53], a[80:83], v146, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[116:119], v[50:53], a[84:87], v146, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[128:131], v[50:53], a[88:91], v147, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[50:53], a[92:95], v147, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:81], v[22:25], a[64:67], v144, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[22:25], a[68:71], v144, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[22:25], a[72:75], v145, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[22:25], a[76:79], v145, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[22:25], a[80:83], v146, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[22:25], a[84:87], v146, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[22:25], a[88:91], v147, v149 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[22:25], a[92:95], v147, v149 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v150, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[22:25], v150, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[70:73], v[54:57], a[96:99], v144, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:77], v[54:57], a[100:103], v144, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[86:89], v[54:57], a[104:107], v145, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[90:93], v[54:57], a[108:111], v145, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[54:57], a[112:115], v146, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:119], v[54:57], a[116:119], v146, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[128:131], v[54:57], a[120:123], v147, v149 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[132:135], v[54:57], a[124:127], v147, v149 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[26:29], a[96:99], v144, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[26:29], a[100:103], v144, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[26:29], a[104:107], v145, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[26:29], a[108:111], v145, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[26:29], a[112:115], v146, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[26:29], a[116:119], v146, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[26:29], a[120:123], v147, v149 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[26:29], a[124:127], v147, v149 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v150, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[26:29], v150, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	s_movk_i32 s36, 0xb00
	buffer_load_dword v156, v96, s[8:11], s19 offen offset:1280
	buffer_load_dword v157, v96, s[8:11], s28 offen offset:1280
	s_barrier
	ds_read_b128 v[74:77], v97
	ds_read2st64_b32 v[148:149], v96 offset0:148 offset1:176
	ds_read_b128 v[78:81], v97 offset:2048
	ds_read_b128 v[82:85], v100
	ds_read_b128 v[86:89], v100 offset:2048
	ds_read_b128 v[90:93], v97 offset:4096
	ds_read_b128 v[104:107], v97 offset:6144
	ds_read_b128 v[108:111], v100 offset:4096
	ds_read_b128 v[112:115], v100 offset:6144
	ds_read_b128 v[116:119], v97 offset:8192
	ds_read2st64_b32 v[150:151], v96 offset0:204 offset1:232
	ds_read_b128 v[120:123], v97 offset:10240
	ds_read_b128 v[132:135], v97 offset:12288
	ds_read_b128 v[124:127], v100 offset:8192
	ds_read_b128 v[136:139], v97 offset:14336
	ds_read_b128 v[128:131], v100 offset:10240
	ds_read_b128 v[140:143], v100 offset:12288
	ds_read_b128 v[144:147], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s36 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[62:65], a[0:3], v148, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s36 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s36 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[62:65], a[4:7], v148, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s36 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[62:65], a[8:11], v149, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[62:65], a[12:15], v149, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[62:65], a[16:19], v150, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[62:65], a[20:23], v150, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[62:65], a[24:27], v151, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[62:65], a[28:31], v151, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:85], v[30:33], a[0:3], v148, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[86:89], v[30:33], a[4:7], v148, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[30:33], a[8:11], v149, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[30:33], a[12:15], v149, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[30:33], a[16:19], v150, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[30:33], a[20:23], v150, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[30:33], a[24:27], v151, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:147], v[30:33], a[28:31], v151, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v158, 0xb000, v103
	buffer_load_dwordx4 v[70:73], v158, s[20:23], s17 offen
	buffer_load_dwordx4 v[58:61], v158, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[34:37], a[32:35], v148, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[34:37], a[36:39], v148, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[34:37], a[40:43], v149, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[34:37], a[44:47], v149, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[34:37], a[48:51], v150, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[34:37], a[52:55], v150, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[34:37], a[56:59], v151, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[34:37], a[60:63], v151, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[2:5], a[32:35], v148, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:89], v[2:5], a[36:39], v148, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[2:5], a[40:43], v149, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[2:5], a[44:47], v149, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:127], v[2:5], a[48:51], v150, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[2:5], a[52:55], v150, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[2:5], a[56:59], v151, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[2:5], a[60:63], v151, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v158, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v158, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[38:41], a[64:67], v148, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[38:41], a[68:71], v148, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[38:41], a[72:75], v149, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[38:41], a[76:79], v149, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[38:41], a[80:83], v150, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[38:41], a[84:87], v150, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[38:41], a[88:91], v151, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[38:41], a[92:95], v151, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[82:85], v[6:9], a[64:67], v148, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[86:89], v[6:9], a[68:71], v148, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[6:9], a[72:75], v149, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[112:115], v[6:9], a[76:79], v149, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[6:9], a[80:83], v150, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[128:131], v[6:9], a[84:87], v150, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[6:9], a[88:91], v151, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[6:9], a[92:95], v151, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v158, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v158, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[42:45], a[96:99], v148, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[42:45], a[100:103], v148, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[42:45], a[104:107], v149, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[42:45], a[108:111], v149, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[42:45], a[112:115], v150, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[42:45], a[116:119], v150, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[42:45], a[120:123], v151, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[42:45], a[124:127], v151, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[10:13], a[96:99], v148, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[86:89], v[10:13], a[100:103], v148, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[10:13], a[104:107], v149, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[10:13], a[108:111], v149, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[10:13], a[112:115], v150, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[10:13], a[116:119], v150, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[10:13], a[120:123], v151, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:147], v[10:13], a[124:127], v151, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v158, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v158, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s36, 0xb80
	buffer_load_dword v159, v96, s[8:11], s19 offen offset:1536
	buffer_load_dword v160, v96, s[8:11], s28 offen offset:1536
	s_barrier
	ds_read_b128 v[78:81], v97 offset:16384
	ds_read2st64_b32 v[152:153], v96 offset0:149 offset1:177
	ds_read_b128 v[82:85], v97 offset:18432
	ds_read_b128 v[86:89], v100 offset:16384
	ds_read_b128 v[90:93], v100 offset:18432
	ds_read_b128 v[104:107], v97 offset:20480
	ds_read_b128 v[108:111], v97 offset:22528
	ds_read_b128 v[112:115], v100 offset:20480
	ds_read_b128 v[116:119], v100 offset:22528
	ds_read_b128 v[120:123], v97 offset:24576
	ds_read2st64_b32 v[154:155], v96 offset0:205 offset1:233
	ds_read_b128 v[124:127], v97 offset:26624
	ds_read_b128 v[136:139], v97 offset:28672
	ds_read_b128 v[128:131], v100 offset:24576
	ds_read_b128 v[140:143], v97 offset:30720
	ds_read_b128 v[132:135], v100 offset:26624
	ds_read_b128 v[144:147], v100 offset:28672
	ds_read_b128 v[148:151], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s36 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[78:81], v[66:69], a[0:3], v152, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s36 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s36 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:85], v[66:69], a[4:7], v152, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s36 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[66:69], a[8:11], v153, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[66:69], a[12:15], v153, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[120:123], v[66:69], a[16:19], v154, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[66:69], a[20:23], v154, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[66:69], a[24:27], v155, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[140:143], v[66:69], a[28:31], v155, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[86:89], v[14:17], a[0:3], v152, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[90:93], v[14:17], a[4:7], v152, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[112:115], v[14:17], a[8:11], v153, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[116:119], v[14:17], a[12:15], v153, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[128:131], v[14:17], a[16:19], v154, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[14:17], a[20:23], v154, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[144:147], v[14:17], a[24:27], v155, v156 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[148:151], v[14:17], a[28:31], v155, v156 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[74:77], v158, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v158, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[46:49], a[32:35], v152, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[82:85], v[46:49], a[36:39], v152, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[104:107], v[46:49], a[40:43], v153, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[46:49], a[44:47], v153, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[120:123], v[46:49], a[48:51], v154, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[124:127], v[46:49], a[52:55], v154, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[46:49], a[56:59], v155, v156 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[46:49], a[60:63], v155, v156 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[86:89], v[18:21], a[32:35], v152, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[90:93], v[18:21], a[36:39], v152, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[112:115], v[18:21], a[40:43], v153, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[18:21], a[44:47], v153, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[18:21], a[48:51], v154, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[18:21], a[52:55], v154, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[18:21], a[56:59], v155, v156 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[148:151], v[18:21], a[60:63], v155, v156 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v158, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[30:33], v158, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:81], v[50:53], a[64:67], v152, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[50:53], a[68:71], v152, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[104:107], v[50:53], a[72:75], v153, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[50:53], a[76:79], v153, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[120:123], v[50:53], a[80:83], v154, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[50:53], a[84:87], v154, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[50:53], a[88:91], v155, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[50:53], a[92:95], v155, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[86:89], v[22:25], a[64:67], v152, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[90:93], v[22:25], a[68:71], v152, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[112:115], v[22:25], a[72:75], v153, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[22:25], a[76:79], v153, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[128:131], v[22:25], a[80:83], v154, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[22:25], a[84:87], v154, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[22:25], a[88:91], v155, v157 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[148:151], v[22:25], a[92:95], v155, v157 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v158, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[38:41], v158, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[78:81], v[54:57], a[96:99], v152, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[54:57], a[100:103], v152, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[104:107], v[54:57], a[104:107], v153, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[108:111], v[54:57], a[108:111], v153, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[54:57], a[112:115], v154, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[54:57], a[116:119], v154, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[136:139], v[54:57], a[120:123], v155, v157 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[140:143], v[54:57], a[124:127], v155, v157 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[86:89], v[26:29], a[96:99], v152, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[90:93], v[26:29], a[100:103], v152, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[112:115], v[26:29], a[104:107], v153, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[116:119], v[26:29], a[108:111], v153, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[26:29], a[112:115], v154, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[26:29], a[116:119], v154, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[144:147], v[26:29], a[120:123], v155, v157 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[148:151], v[26:29], a[124:127], v155, v157 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v158, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[54:57], v158, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	buffer_load_dword v152, v96, s[8:11], s19 offen offset:1792
	buffer_load_dword v153, v96, s[8:11], s28 offen offset:1792
	s_barrier
	ds_read_b128 v[22:25], v97
	ds_read2st64_b32 v[148:149], v96 offset0:150 offset1:178
	ds_read_b128 v[78:81], v97 offset:2048
	ds_read_b128 v[82:85], v100
	ds_read_b128 v[86:89], v100 offset:2048
	ds_read_b128 v[90:93], v97 offset:4096
	ds_read_b128 v[104:107], v97 offset:6144
	ds_read_b128 v[108:111], v100 offset:4096
	ds_read_b128 v[112:115], v100 offset:6144
	ds_read_b128 v[116:119], v97 offset:8192
	ds_read2st64_b32 v[150:151], v96 offset0:206 offset1:234
	ds_read_b128 v[120:123], v97 offset:10240
	ds_read_b128 v[132:135], v97 offset:12288
	ds_read_b128 v[124:127], v100 offset:8192
	ds_read_b128 v[136:139], v97 offset:14336
	ds_read_b128 v[128:131], v100 offset:10240
	ds_read_b128 v[140:143], v100 offset:12288
	ds_read_b128 v[144:147], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s35 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[22:25], v[70:73], a[0:3], v148, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s35 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s35 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[70:73], a[4:7], v148, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s35 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[70:73], a[8:11], v149, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[70:73], a[12:15], v149, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[70:73], a[16:19], v150, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[70:73], a[20:23], v150, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[70:73], a[24:27], v151, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[70:73], a[28:31], v151, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:85], v[58:61], a[0:3], v148, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[86:89], v[58:61], a[4:7], v148, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[58:61], a[8:11], v149, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[58:61], a[12:15], v149, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[58:61], a[16:19], v150, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[58:61], a[20:23], v150, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[58:61], a[24:27], v151, v159 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:147], v[58:61], a[28:31], v151, v159 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v154, 0xc000, v103
	buffer_load_dwordx4 v[70:73], v154, s[20:23], s17 offen
	buffer_load_dwordx4 v[18:21], v154, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[22:25], v[34:37], a[32:35], v148, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[34:37], a[36:39], v148, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[34:37], a[40:43], v149, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[34:37], a[44:47], v149, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[34:37], a[48:51], v150, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[34:37], a[52:55], v150, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[34:37], a[56:59], v151, v159 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[34:37], a[60:63], v151, v159 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[2:5], a[32:35], v148, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:89], v[2:5], a[36:39], v148, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[2:5], a[40:43], v149, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[2:5], a[44:47], v149, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:127], v[2:5], a[48:51], v150, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[2:5], a[52:55], v150, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[2:5], a[56:59], v151, v159 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[2:5], a[60:63], v151, v159 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[26:29], v154, s[20:23], s16 offen
	buffer_load_dwordx4 v[2:5], v154, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[22:25], v[62:65], a[64:67], v148, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[62:65], a[68:71], v148, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[62:65], a[72:75], v149, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[62:65], a[76:79], v149, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[62:65], a[80:83], v150, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[62:65], a[84:87], v150, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[62:65], a[88:91], v151, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[62:65], a[92:95], v151, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[82:85], v[6:9], a[64:67], v148, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[86:89], v[6:9], a[68:71], v148, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[6:9], a[72:75], v149, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[112:115], v[6:9], a[76:79], v149, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[6:9], a[80:83], v150, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[128:131], v[6:9], a[84:87], v150, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[6:9], a[88:91], v151, v160 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[6:9], a[92:95], v151, v160 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v154, s[20:23], s15 offen
	buffer_load_dwordx4 v[6:9], v154, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[22:25], v[42:45], a[96:99], v148, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[42:45], a[100:103], v148, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[42:45], a[104:107], v149, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[42:45], a[108:111], v149, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[42:45], a[112:115], v150, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[42:45], a[116:119], v150, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[42:45], a[120:123], v151, v160 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[42:45], a[124:127], v151, v160 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[10:13], a[96:99], v148, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[86:89], v[10:13], a[100:103], v148, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[10:13], a[104:107], v149, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[10:13], a[108:111], v149, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[10:13], a[112:115], v150, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[10:13], a[116:119], v150, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[10:13], a[120:123], v151, v160 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:147], v[10:13], a[124:127], v151, v160 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v154, s[20:23], s14 offen
	buffer_load_dwordx4 v[10:13], v154, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s35, 0xc80
	buffer_load_dword v155, v96, s[8:11], s19 offen offset:2048
	buffer_load_dword v164, v96, s[8:11], s28 offen offset:2048
	s_barrier
	ds_read_b128 v[62:65], v97 offset:16384
	ds_read2st64_b32 v[148:149], v96 offset0:151 offset1:179
	ds_read_b128 v[78:81], v97 offset:18432
	ds_read_b128 v[82:85], v100 offset:16384
	ds_read_b128 v[86:89], v100 offset:18432
	ds_read_b128 v[90:93], v97 offset:20480
	ds_read_b128 v[104:107], v97 offset:22528
	ds_read_b128 v[108:111], v100 offset:20480
	ds_read_b128 v[112:115], v100 offset:22528
	ds_read_b128 v[116:119], v97 offset:24576
	ds_read2st64_b32 v[150:151], v96 offset0:207 offset1:235
	ds_read_b128 v[120:123], v97 offset:26624
	ds_read_b128 v[132:135], v97 offset:28672
	ds_read_b128 v[124:127], v100 offset:24576
	ds_read_b128 v[136:139], v97 offset:30720
	ds_read_b128 v[128:131], v100 offset:26624
	ds_read_b128 v[140:143], v100 offset:28672
	ds_read_b128 v[144:147], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s35 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[74:77], a[0:3], v148, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s35 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s35 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[74:77], a[4:7], v148, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s35 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[74:77], a[8:11], v149, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[74:77], a[12:15], v149, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[74:77], a[16:19], v150, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[74:77], a[20:23], v150, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[74:77], a[24:27], v151, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[74:77], a[28:31], v151, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[82:85], v[14:17], a[0:3], v148, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[86:89], v[14:17], a[4:7], v148, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[108:111], v[14:17], a[8:11], v149, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[112:115], v[14:17], a[12:15], v149, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[14:17], a[16:19], v150, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[14:17], a[20:23], v150, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[14:17], a[24:27], v151, v152 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[144:147], v[14:17], a[28:31], v151, v152 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v154, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v154, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:65], v[46:49], a[32:35], v148, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[46:49], a[36:39], v148, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[46:49], a[40:43], v149, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[46:49], a[44:47], v149, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[46:49], a[48:51], v150, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[46:49], a[52:55], v150, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[46:49], a[56:59], v151, v152 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[46:49], a[60:63], v151, v152 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[30:33], a[32:35], v148, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[86:89], v[30:33], a[36:39], v148, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[30:33], a[40:43], v149, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[30:33], a[44:47], v149, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:127], v[30:33], a[48:51], v150, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[30:33], a[52:55], v150, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[30:33], a[56:59], v151, v152 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[30:33], a[60:63], v151, v152 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v154, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[22:25], v154, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[50:53], a[64:67], v148, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[50:53], a[68:71], v148, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[50:53], a[72:75], v149, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[50:53], a[76:79], v149, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[50:53], a[80:83], v150, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[50:53], a[84:87], v150, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[50:53], a[88:91], v151, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[50:53], a[92:95], v151, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[82:85], v[38:41], a[64:67], v148, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[86:89], v[38:41], a[68:71], v148, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[108:111], v[38:41], a[72:75], v149, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[112:115], v[38:41], a[76:79], v149, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[38:41], a[80:83], v150, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[128:131], v[38:41], a[84:87], v150, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[38:41], a[88:91], v151, v153 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[38:41], a[92:95], v151, v153 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v154, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[30:33], v154, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[62:65], v[66:69], a[96:99], v148, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[66:69], a[100:103], v148, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[66:69], a[104:107], v149, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[66:69], a[108:111], v149, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[66:69], a[112:115], v150, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[66:69], a[116:119], v150, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[66:69], a[120:123], v151, v153 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[66:69], a[124:127], v151, v153 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[54:57], a[96:99], v148, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[86:89], v[54:57], a[100:103], v148, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[108:111], v[54:57], a[104:107], v149, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[54:57], a[108:111], v149, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[54:57], a[112:115], v150, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[54:57], a[116:119], v150, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[140:143], v[54:57], a[120:123], v151, v153 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[144:147], v[54:57], a[124:127], v151, v153 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v154, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[38:41], v154, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s34
	s_movk_i32 s34, 0xd00
	buffer_load_dword v165, v96, s[8:11], s19 offen offset:2304
	buffer_load_dword v166, v96, s[8:11], s28 offen offset:2304
	s_barrier
	ds_read_b128 v[62:65], v97
	ds_read2st64_b32 v[160:161], v96 offset0:152 offset1:180
	ds_read_b128 v[66:69], v97 offset:2048
	ds_read_b128 v[74:77], v100
	ds_read_b128 v[78:81], v100 offset:2048
	ds_read_b128 v[82:85], v97 offset:4096
	ds_read_b128 v[86:89], v97 offset:6144
	ds_read_b128 v[90:93], v100 offset:4096
	ds_read_b128 v[104:107], v100 offset:6144
	ds_read_b128 v[108:111], v97 offset:8192
	ds_read2st64_b32 v[162:163], v96 offset0:208 offset1:236
	ds_read_b128 v[112:115], v97 offset:10240
	ds_read_b128 v[124:127], v97 offset:12288
	ds_read_b128 v[116:119], v100 offset:8192
	ds_read_b128 v[128:131], v97 offset:14336
	ds_read_b128 v[120:123], v100 offset:10240
	ds_read_b128 v[132:135], v100 offset:12288
	ds_read_b128 v[136:139], v100 offset:14336
	buffer_load_dwordx4 v94, s[24:27], s34 offen lds
	s_mov_b32 m0, s30
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[70:73], a[0:3], v160, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s34 offen lds
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s34 offen lds
	s_mov_b32 m0, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[70:73], a[4:7], v160, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s34 offen lds
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[70:73], a[8:11], v161, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[70:73], a[12:15], v161, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[108:111], v[70:73], a[16:19], v162, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[70:73], a[20:23], v162, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[70:73], a[24:27], v163, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[70:73], a[28:31], v163, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[18:21], a[0:3], v160, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[18:21], a[4:7], v160, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[90:93], v[18:21], a[8:11], v161, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[18:21], a[12:15], v161, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[116:119], v[18:21], a[16:19], v162, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[18:21], a[20:23], v162, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[18:21], a[24:27], v163, v155 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[136:139], v[18:21], a[28:31], v163, v155 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v167, 0xd000, v103
	buffer_load_dwordx4 v[140:143], v167, s[20:23], s17 offen
	buffer_load_dwordx4 v[144:147], v167, s[20:23], s17 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:65], v[26:29], a[32:35], v160, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[66:69], v[26:29], a[36:39], v160, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[26:29], a[40:43], v161, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[26:29], a[44:47], v161, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[26:29], a[48:51], v162, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[112:115], v[26:29], a[52:55], v162, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[26:29], a[56:59], v163, v155 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[26:29], a[60:63], v163, v155 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[2:5], a[32:35], v160, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[78:81], v[2:5], a[36:39], v160, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[90:93], v[2:5], a[40:43], v161, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[104:107], v[2:5], a[44:47], v161, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[116:119], v[2:5], a[48:51], v162, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[120:123], v[2:5], a[52:55], v162, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[2:5], a[56:59], v163, v155 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[2:5], a[60:63], v163, v155 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[148:151], v167, s[20:23], s16 offen
	buffer_load_dwordx4 v[152:155], v167, s[20:23], s16 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[62:65], v[34:37], a[64:67], v160, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[66:69], v[34:37], a[68:71], v160, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[34:37], a[72:75], v161, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[34:37], a[76:79], v161, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[34:37], a[80:83], v162, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[34:37], a[84:87], v162, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[34:37], a[88:91], v163, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[34:37], a[92:95], v163, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[74:77], v[6:9], a[64:67], v160, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[78:81], v[6:9], a[68:71], v160, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[6:9], a[72:75], v161, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[6:9], a[76:79], v161, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[116:119], v[6:9], a[80:83], v162, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[120:123], v[6:9], a[84:87], v162, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[132:135], v[6:9], a[88:91], v163, v164 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[6:9], a[92:95], v163, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[156:159], v167, s[20:23], s15 offen
	buffer_load_dwordx4 v[2:5], v167, s[20:23], s15 offen offset:1024
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[62:65], v[42:45], a[96:99], v160, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:69], v[42:45], a[100:103], v160, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[42:45], a[104:107], v161, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[42:45], a[108:111], v161, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[42:45], a[112:115], v162, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[112:115], v[42:45], a[116:119], v162, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[42:45], a[120:123], v163, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[128:131], v[42:45], a[124:127], v163, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[74:77], v[10:13], a[96:99], v160, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[10:13], a[100:103], v160, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[90:93], v[10:13], a[104:107], v161, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[10:13], a[108:111], v161, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[10:13], a[112:115], v162, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[120:123], v[10:13], a[116:119], v162, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[132:135], v[10:13], a[120:123], v163, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[136:139], v[10:13], a[124:127], v163, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[10:13], v167, s[20:23], s14 offen
	buffer_load_dwordx4 v[6:9], v167, s[20:23], s14 offen offset:1024
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s29
	s_movk_i32 s29, 0xd80
	buffer_load_dword v136, v96, s[8:11], s19 offen offset:2560
	buffer_load_dword v103, v96, s[8:11], s28 offen offset:2560
	s_barrier
	ds_read_b128 v[42:45], v97 offset:16384
	ds_read2st64_b32 v[132:133], v96 offset0:153 offset1:181
	ds_read_b128 v[62:65], v97 offset:18432
	ds_read_b128 v[66:69], v100 offset:16384
	ds_read_b128 v[70:73], v100 offset:18432
	ds_read_b128 v[74:77], v97 offset:20480
	ds_read_b128 v[78:81], v97 offset:22528
	ds_read_b128 v[82:85], v100 offset:20480
	ds_read_b128 v[86:89], v100 offset:22528
	ds_read_b128 v[90:93], v97 offset:24576
	ds_read2st64_b32 v[134:135], v96 offset0:209 offset1:237
	ds_read_b128 v[104:107], v97 offset:26624
	ds_read_b128 v[116:119], v97 offset:28672
	ds_read_b128 v[108:111], v100 offset:24576
	ds_read_b128 v[120:123], v97 offset:30720
	ds_read_b128 v[112:115], v100 offset:26624
	ds_read_b128 v[124:127], v100 offset:28672
	ds_read_b128 v[128:131], v100 offset:30720
	buffer_load_dwordx4 v94, s[24:27], s29 offen lds
	s_mov_b32 m0, s6
	s_waitcnt vmcnt(16) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[58:61], a[0:3], v132, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v95, s[24:27], s29 offen lds
	s_mov_b32 m0, s7
	s_nop 0
	buffer_load_dwordx4 v101, s[24:27], s29 offen lds
	s_mov_b32 m0, s18
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[58:61], a[4:7], v132, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v102, s[24:27], s29 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[66:69], v[14:17], a[0:3], v132, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(13)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[70:73], v[14:17], a[4:7], v132, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[58:61], a[8:11], v133, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[58:61], a[12:15], v133, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(10)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[14:17], a[0:3], v133, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(9)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[14:17], a[4:7], v133, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[58:61], a[16:19], v134, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[58:61], a[20:23], v134, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[14:17], a[0:3], v134, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[112:115], v[14:17], a[4:7], v134, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[58:61], a[24:27], v135, v165 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[58:61], a[28:31], v135, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[14:17], a[0:3], v135, v165 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[14:17], a[4:7], v135, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[18:21], v167, s[20:23], s17 offen offset:2048
	buffer_load_dwordx4 v[14:17], v167, s[20:23], s17 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[46:49], a[32:35], v132, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[46:49], a[36:39], v132, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[66:69], v[22:25], a[0:3], v132, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[70:73], v[22:25], a[4:7], v132, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[46:49], a[40:43], v133, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[46:49], a[44:47], v133, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[22:25], a[0:3], v133, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[86:89], v[22:25], a[4:7], v133, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[46:49], a[48:51], v134, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[46:49], a[52:55], v134, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[108:111], v[22:25], a[0:3], v134, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], v[112:115], v[22:25], a[4:7], v134, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[46:49], a[56:59], v135, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[46:49], a[60:63], v135, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[22:25], a[0:3], v135, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[22:25], a[4:7], v135, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[26:29], v167, s[20:23], s16 offen offset:2048
	buffer_load_dwordx4 v[22:25], v167, s[20:23], s16 offen offset:3072
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(22)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[50:53], a[64:67], v132, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[50:53], a[68:71], v132, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[66:69], v[30:33], a[0:3], v132, v166 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[70:73], v[30:33], a[4:7], v132, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[50:53], a[72:75], v133, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[50:53], a[76:79], v133, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[30:33], a[0:3], v133, v166 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[86:89], v[30:33], a[4:7], v133, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[50:53], a[80:83], v134, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[50:53], a[84:87], v134, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[108:111], v[30:33], a[0:3], v134, v166 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[30:33], a[4:7], v134, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[50:53], a[88:91], v135, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[120:123], v[50:53], a[92:95], v135, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[30:33], a[0:3], v135, v166 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[30:33], a[4:7], v135, v166 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v167, s[20:23], s15 offen offset:2048
	buffer_load_dwordx4 v[30:33], v167, s[20:23], s15 offen offset:3072
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[42:45], v[54:57], a[96:99], v132, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[62:65], v[54:57], a[100:103], v132, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[66:69], v[38:41], a[0:3], v132, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[74:77], v[54:57], a[104:107], v133, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[38:41], a[4:7], v132, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[54:57], a[108:111], v133, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[82:85], v[38:41], a[0:3], v133, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[90:93], v[54:57], a[112:115], v134, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[86:89], v[38:41], a[4:7], v133, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[104:107], v[54:57], a[116:119], v134, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[108:111], v[38:41], a[0:3], v134, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[116:119], v[54:57], a[120:123], v135, v166 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:123], v[54:57], a[124:127], v135, v166 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[112:115], v[38:41], a[4:7], v134, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[124:127], v[38:41], a[0:3], v135, v166 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[128:131], v[38:41], a[36:39], v135, v166 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v167, s[20:23], s14 offen offset:2048
	buffer_load_dwordx4 v[38:41], v167, s[20:23], s14 offen offset:3072
	; sched_barrier mask(0x00000000)
	buffer_load_dword v101, v96, s[8:11], s19 offen offset:2816
	buffer_load_dword v102, v96, s[8:11], s28 offen offset:2816
	s_barrier
	ds_read_b128 v[62:65], v97
	ds_read_b128 v[66:69], v97 offset:2048
	ds_read2st64_b32 v[120:121], v96 offset0:154 offset1:182
	ds_read_b128 v[104:107], v97 offset:4096
	ds_read_b128 v[108:111], v97 offset:6144
	ds_read_b128 v[112:115], v100 offset:4096
	ds_read_b128 v[116:119], v100 offset:6144
	ds_read_b128 v[78:81], v100
	ds_read_b128 v[82:85], v100 offset:2048
	ds_read_b128 v[86:89], v97 offset:8192
	ds_read_b128 v[74:77], v97 offset:10240
	ds_read2st64_b32 v[94:95], v96 offset0:210 offset1:238
	ds_read_b128 v[90:93], v100 offset:8192
	ds_read_b128 v[70:73], v100 offset:10240
	ds_read_b128 v[54:57], v97 offset:12288
	ds_read_b128 v[46:49], v97 offset:14336
	s_waitcnt vmcnt(15) lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[104:107], v[140:143], a[8:11], v121, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[58:61], v100 offset:12288
	ds_read_b128 v[50:53], v100 offset:14336
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[108:111], v[140:143], a[12:15], v121, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_barrier
	v_or_b32_e32 v99, s13, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[62:65], v[140:143], a[128:131], v120, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshlrev_b32_e32 v1, 10, v1
	s_lshl_b32 s6, s12, 6
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[66:69], v[140:143], a[132:135], v120, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[112:115], v[144:147], a[8:11], v121, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[116:119], v[144:147], a[12:15], v121, v136 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[86:89], v[140:143], a[20:23], v94, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[74:77], v[140:143], a[136:139], v94, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[78:81], v[144:147], a[36:39], v120, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:85], v[144:147], a[52:55], v120, v136 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[90:93], v[144:147], a[8:11], v94, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[70:73], v[144:147], a[12:15], v94, v136 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[54:57], v[140:143], a[24:27], v95, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[46:49], v[140:143], a[28:31], v95, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[58:61], v[144:147], a[8:11], v95, v136 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[50:53], v[144:147], a[12:15], v95, v136 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[62:65], v[148:151], a[140:143], v120, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[66:69], v[148:151], a[144:147], v120, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[78:81], v[152:155], a[12:15], v120, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], v[82:85], v[152:155], a[20:23], v120, v136 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[104:107], v[148:151], a[40:43], v121, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[108:111], v[148:151], a[44:47], v121, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], v[112:115], v[152:155], a[12:15], v121, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], v[116:119], v[152:155], a[20:23], v121, v136 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[86:89], v[148:151], a[48:51], v94, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[74:77], v[148:151], a[148:151], v94, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], v[90:93], v[152:155], a[12:15], v94, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[70:73], v[152:155], a[20:23], v94, v136 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[54:57], v[148:151], a[56:59], v95, v136 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:49], v[148:151], a[60:63], v95, v136 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[66:69], v[156:159], a[68:71], v120, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[58:61], v[152:155], a[12:15], v95, v136 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[50:53], v[152:155], a[20:23], v95, v136 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v154, v99, v1, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[156:159], a[64:67], v120, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[82:85], v[2:5], a[40:43], v120, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[108:111], v[156:159], a[76:79], v121, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[78:81], v[2:5], a[20:23], v120, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[104:107], v[156:159], a[72:75], v121, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[2:5], a[40:43], v121, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[74:77], v[156:159], a[84:87], v94, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[104:107], v[10:13], a[104:107], v121, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[104:107], v97 offset:16384
	ds_read2st64_b32 v[152:153], v96 offset0:155 offset1:183
	ds_read_b128 v[124:127], v97 offset:22528
	ds_read_b128 v[128:131], v100 offset:20480
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[112:115], v[2:5], a[20:23], v121, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[112:115], v[6:9], a[84:87], v121, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[112:115], v100 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[86:89], v[156:159], a[80:83], v94, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[62:65], v[10:13], a[96:99], v120, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:73], v[2:5], a[40:43], v94, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[54:57], v[156:159], a[88:91], v95, v103 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[108:111], v[10:13], a[108:111], v121, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[108:111], v97 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[90:93], v[2:5], a[20:23], v94, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[46:49], v[156:159], a[92:95], v95, v103 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[78:81], v[6:9], a[76:79], v120, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(1) lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[104:107], v[18:21], a[112:115], v152, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[104:107], v[26:29], a[128:131], v152, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[104:107], v[34:37], a[56:59], v152, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[116:119], v[6:9], a[88:91], v121, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[116:119], v100 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[10:13], a[100:103], v120, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[14:17], a[92:95], v152, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[112:115], v[22:25], a[108:111], v152, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:115], v[30:33], a[56:59], v152, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[104:107], v[42:45], a[76:79], v152, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v104, 0x90, v99
	v_add_lshl_u32 v105, v104, v1, 2
	v_or_b32_e32 v107, 0x4100, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[82:85], v[6:9], a[80:83], v120, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 v[120:123], v97 offset:20480
	ds_read_b128 v[132:135], v100 offset:22528
	ds_read_b128 v[136:139], v97 offset:24576
	ds_read_b128 v[140:143], v97 offset:26624
	ds_read_b128 v[78:81], v97 offset:28672
	ds_read_b128 v[62:65], v97 offset:30720
	ds_read_b128 v[144:147], v100 offset:24576
	ds_read_b128 v[148:151], v100 offset:26624
	s_waitcnt lgkmcnt(9)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[108:111], v[18:21], a[116:119], v152, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[82:85], v100 offset:28672
	ds_read_b128 v[66:69], v100 offset:30720
	ds_read2st64_b32 v[96:97], v96 offset0:211 offset1:239
	v_add_u32_e32 v100, 0x80, v99
	v_add_lshl_u32 v155, v100, v1, 2
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b32 v154, a92
	ds_write_b32 v154, a93 offset:1024
	ds_write_b32 v154, a94 offset:2048
	ds_write_b32 v154, a95 offset:3072
	ds_write_b32 v155, a108
	ds_write_b32 v155, a109 offset:1024
	ds_write_b32 v155, a110 offset:2048
	ds_write_b32 v155, a111 offset:3072
	ds_write_b32 v154, a56 offset:64
	ds_write_b32 v154, a57 offset:1088
	ds_write_b32 v154, a58 offset:2112
	ds_write_b32 v154, a59 offset:3136
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[112:115], v[38:41], a[76:79], v152, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 7
	ds_write_b32 v105, a56
	ds_write_b32 v105, a57 offset:1024
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:119], v[14:17], a[96:99], v152, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v105, a58 offset:2048
	ds_write_b32 v105, a59 offset:3072
	s_nop 5
	ds_write_b32 v154, a96 offset:16384
	ds_write_b32 v154, a97 offset:17408
	ds_write_b32 v154, a98 offset:18432
	ds_write_b32 v154, a99 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[108:111], v[26:29], a[132:135], v152, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[108:111], v[34:37], a[60:63], v152, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[108:111], v[42:45], a[80:83], v152, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[120:123], v[18:21], a[120:123], v153, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[116:119], v[22:25], a[112:115], v152, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 7
	ds_write_b32 v155, a112 offset:16384
	ds_write_b32 v155, a113 offset:17408
	ds_write_b32 v155, a114 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[120:123], v[26:29], a[136:139], v153, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[116:119], v[30:33], a[56:59], v152, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v155, a115 offset:19456
	s_nop 6
	ds_write_b32 v154, a56 offset:16448
	ds_write_b32 v154, a57 offset:17472
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[116:119], v[38:41], a[60:63], v152, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v154, a58 offset:18496
	ds_write_b32 v154, a59 offset:19520
	s_nop 5
	ds_write_b32 v105, a60 offset:16384
	ds_write_b32 v105, a61 offset:17408
	ds_write_b32 v105, a62 offset:18432
	ds_write_b32 v105, a63 offset:19456
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[120:123], v[34:37], a[64:67], v153, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:131], v[14:17], a[100:103], v153, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 7
	ds_write_b32 v154, a100 offset:32768
	ds_write_b32 v154, a101 offset:33792
	ds_write_b32 v154, a102 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[22:25], a[76:79], v153, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v154, a103 offset:35840
	s_nop 6
	ds_write_b32 v155, a60 offset:32768
	ds_write_b32 v155, a61 offset:33792
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[128:131], v[30:33], a[64:67], v153, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v155, a62 offset:34816
	ds_write_b32 v155, a63 offset:35840
	s_nop 5
	ds_write_b32 v154, a64 offset:32832
	ds_write_b32 v154, a65 offset:33856
	ds_write_b32 v154, a66 offset:34880
	ds_write_b32 v154, a67 offset:35904
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[124:127], v[18:21], a[124:127], v153, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[26:29], a[140:143], v153, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[120:123], v[42:45], a[84:87], v153, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:135], v[14:17], a[104:107], v153, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[38:41], a[60:63], v153, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 7
	ds_write_b32 v105, a60 offset:32768
	ds_write_b32 v105, a61 offset:33792
	ds_write_b32 v105, a62 offset:34816
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[22:25], a[56:59], v153, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v105, a63 offset:35840
	ds_write_b32 v154, a104 offset:49152
	ds_write_b32 v154, a105 offset:50176
	ds_write_b32 v154, a106 offset:51200
	ds_write_b32 v154, a107 offset:52224
	s_nop 2
	ds_write_b32 v155, a56 offset:49152
	ds_write_b32 v155, a57 offset:50176
	ds_write_b32 v155, a58 offset:51200
	ds_write_b32 v155, a59 offset:52224
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[34:37], a[68:71], v153, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[42:45], a[88:91], v153, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[30:33], a[60:63], v153, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 7
	ds_write_b32 v154, a60 offset:49216
	ds_write_b32 v154, a61 offset:50240
	ds_write_b32 v154, a62 offset:51264
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[38:41], a[56:59], v153, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v154, a63 offset:52288
	s_nop 6
	ds_write_b32 v105, a56 offset:49152
	ds_write_b32 v105, a57 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[136:139], v[18:21], a[52:55], v96, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_write_b32 v105, a58 offset:51200
	ds_write_b32 v105, a59 offset:52224
	v_or_b32_e32 v105, 0x4000, v1
	v_add_lshl_u32 v106, v99, v105, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[86:89], v[10:13], a[32:35], v94, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_e32 v87, 0x4200, v1
	v_or_b32_e32 v89, 0x4300, v1
	v_add_lshl_u32 v86, v99, v107, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[14:17], a[52:55], v96, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v88, v99, v87, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[90:93], v[6:9], a[32:35], v94, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v90, v99, v89, 2
	v_add_lshl_u32 v91, v100, v105, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[26:29], a[144:147], v96, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	ds_write_b32 v106, a52
	ds_write_b32 v86, a53
	ds_write_b32 v88, a54
	ds_write_b32 v90, a55
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[144:147], v[22:25], a[56:59], v96, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[34:37], a[72:75], v96, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[136:139], v[42:45], a[32:35], v96, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 5
	ds_write_b32 v91, a52
	v_add_lshl_u32 v91, v100, v107, 2
	ds_write_b32 v91, a53
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[30:33], a[56:59], v96, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v91, v100, v87, 2
	ds_write_b32 v91, a54
	v_add_lshl_u32 v91, v100, v89, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[144:147], v[38:41], a[32:35], v96, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v91, a55
	s_nop 2
	ds_write_b32 v106, a56 offset:64
	ds_write_b32 v86, a57 offset:64
	v_add_lshl_u32 v86, v104, v105, 2
	ds_write_b32 v88, a58 offset:64
	ds_write_b32 v90, a59 offset:64
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[18:21], a[36:39], v96, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_write_b32 v86, a32
	v_add_lshl_u32 v86, v104, v107, 2
	ds_write_b32 v86, a33
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[10:13], a[16:19], v94, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_lshl_u32 v74, v104, v87, 2
	ds_write_b32 v74, a34
	v_add_lshl_u32 v74, v104, v89, 2
	ds_write_b32 v74, a35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[148:151], v[14:17], a[36:39], v96, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_or_b32_e32 v74, 0x5000, v1
	v_or_b32_e32 v76, 0x5100, v1
	v_add_lshl_u32 v75, v99, v74, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[70:73], v[6:9], a[16:19], v94, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_or_b32_e32 v70, 0x5200, v1
	v_or_b32_e32 v72, 0x5300, v1
	v_add_lshl_u32 v77, v99, v76, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[26:29], a[44:47], v96, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_lshl_u32 v71, v99, v70, 2
	v_add_lshl_u32 v73, v99, v72, 2
	ds_write_b32 v75, a32
	ds_write_b32 v77, a33
	ds_write_b32 v71, a34
	ds_write_b32 v73, a35
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[140:143], v[34:37], a[48:51], v96, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_lshl_u32 v86, v100, v74, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[148:151], v[22:25], a[36:39], v96, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[42:45], a[16:19], v96, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[148:151], v[30:33], a[32:35], v96, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 5
	ds_write_b32 v86, a36
	v_add_lshl_u32 v86, v100, v76, 2
	ds_write_b32 v86, a37
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[38:41], a[16:19], v96, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v86, v100, v70, 2
	ds_write_b32 v86, a38
	v_add_lshl_u32 v86, v100, v72, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[18:21], a[24:27], v97, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_write_b32 v86, a39
	ds_write_b32 v75, a32 offset:64
	ds_write_b32 v77, a33 offset:64
	ds_write_b32 v71, a34 offset:64
	v_add_lshl_u32 v71, v104, v74, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[54:57], v[10:13], a[4:7], v95, v103 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_lshl_u32 v70, v104, v70, 2
	ds_write_b32 v73, a35 offset:64
	ds_write_b32 v71, a16
	v_add_lshl_u32 v71, v104, v76, 2
	ds_write_b32 v70, a18
	v_add_lshl_u32 v70, v104, v72, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[58:61], v[2:5], a[40:43], v95, v103 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v71, a17
	ds_write_b32 v70, a19
	v_or_b32_e32 v70, 0x6000, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[82:85], v[14:17], a[24:27], v97, v101 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_or_b32_e32 v54, 0x6100, v1
	v_or_b32_e32 v56, 0x6200, v1
	v_add_lshl_u32 v71, v99, v70, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[58:61], v[6:9], a[4:7], v95, v103 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_or_b32_e32 v58, 0x6300, v1
	v_add_lshl_u32 v55, v99, v54, 2
	v_add_lshl_u32 v57, v99, v56, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[26:29], a[28:31], v97, v101 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_lshl_u32 v59, v99, v58, 2
	ds_write_b32 v71, a16
	ds_write_b32 v55, a17
	ds_write_b32 v57, a18
	ds_write_b32 v59, a19
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[82:85], v[22:25], a[24:27], v97, v101 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v60, v100, v70, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[78:81], v[34:37], a[32:35], v97, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[78:81], v[42:45], a[4:7], v97, v102 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 4
	ds_write_b32 v60, a16
	v_add_lshl_u32 v60, v100, v54, 2
	ds_write_b32 v60, a17
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:85], v[30:33], a[24:27], v97, v102 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v60, v100, v56, 2
	ds_write_b32 v60, a18
	v_add_lshl_u32 v60, v100, v58, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[82:85], v[38:41], a[4:7], v97, v102 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v60, a19
	s_nop 2
	ds_write_b32 v71, a24 offset:64
	ds_write_b32 v55, a25 offset:64
	ds_write_b32 v57, a26 offset:64
	ds_write_b32 v59, a27 offset:64
	v_add_lshl_u32 v55, v104, v70, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[18:21], a[8:11], v97, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_lshl_u32 v54, v104, v54, 2
	ds_write_b32 v55, a4
	ds_write_b32 v54, a5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[46:49], v[10:13], a[0:3], v95, v103 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_e32 v10, 0x7200, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[50:53], v[2:5], a[20:23], v95, v103 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v2, v104, v56, 2
	ds_write_b32 v2, a6
	v_add_lshl_u32 v2, v104, v58, 2
	ds_write_b32 v2, a7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[14:17], a[8:11], v97, v101 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_or_b32_e32 v2, 0x7000, v1
	v_or_b32_e32 v4, 0x7100, v1
	v_or_b32_e32 v1, 0x7300, v1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[50:53], v[6:9], a[0:3], v95, v103 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v3, v99, v2, 2
	v_add_lshl_u32 v5, v99, v4, 2
	v_add_lshl_u32 v6, v99, v10, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[26:29], a[12:15], v97, v101 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_lshl_u32 v7, v99, v1, 2
	ds_write_b32 v3, a4
	ds_write_b32 v5, a5
	ds_write_b32 v6, a6
	ds_write_b32 v7, a7
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[66:69], v[22:25], a[8:11], v97, v101 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v8, v100, v2, 2
	v_add_lshl_u32 v2, v104, v2, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[62:65], v[34:37], a[16:19], v97, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[62:65], v[42:45], a[0:3], v97, v102 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 3
	ds_write_b32 v8, a4
	v_add_lshl_u32 v8, v100, v4, 2
	ds_write_b32 v8, a5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[66:69], v[30:33], a[8:11], v97, v102 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add_lshl_u32 v8, v100, v10, 2
	ds_write_b32 v8, a6
	v_add_lshl_u32 v8, v100, v1, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[66:69], v[38:41], a[0:3], v97, v102 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_write_b32 v8, a7
	s_nop 2
	ds_write_b32 v3, a8 offset:64
	ds_write_b32 v5, a9 offset:64
	ds_write_b32 v6, a10 offset:64
	ds_write_b32 v7, a11 offset:64
	v_lshrrev_b32_e32 v3, 4, v0
	ds_write_b32 v2, a0
	v_add_lshl_u32 v2, v104, v4, 2
	ds_write_b32 v2, a1
	v_add_lshl_u32 v2, v104, v10, 2
	v_and_b32_e32 v4, 3, v0
	ds_write_b32 v2, a2
	v_bfe_u32 v2, v0, 2, 2
	v_lshlrev_b32_e32 v0, 5, v4
	v_lshl_or_b32 v0, v2, 7, v0
	v_add_lshl_u32 v1, v104, v1, 2
	v_lshl_or_b32 v6, v3, 10, v0
	ds_write_b32 v1, a3
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read2_b32 v[0:1], v6 offset1:1
	ds_read2_b32 v[8:9], v6 offset0:2 offset1:3
	ds_read2_b32 v[10:11], v6 offset0:4 offset1:5
	ds_read2_b32 v[12:13], v6 offset0:6 offset1:7
	v_lshlrev_b32_e32 v7, 4, v2
	v_lshlrev_b32_e32 v22, 2, v4
	v_or_b32_e32 v23, s3, v3
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v5, 0xbfb8aa3b, v0
	v_exp_f32_e32 v5, v5
	v_mul_f32_e32 v14, 0xbfb8aa3b, v1
	v_exp_f32_e32 v14, v14
	v_cmp_eq_u32_e32 vcc, 0, v4
	v_add_f32_e32 v5, 1.0, v5
	v_rcp_f32_e32 v5, v5
	v_add_f32_e32 v14, 1.0, v14
	v_rcp_f32_e32 v24, v14
	ds_read2_b32 v[14:15], v6 offset0:128 offset1:129
	ds_read2_b32 v[16:17], v6 offset0:130 offset1:131
	ds_read2_b32 v[18:19], v6 offset0:132 offset1:133
	ds_read2_b32 v[20:21], v6 offset0:134 offset1:135
	v_mul_f32_e32 v0, v0, v5
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v5, 0xbfb8aa3b, v8
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v0, v14, v0
	v_exp_f32_e32 v5, v5
	v_mul_f32_e32 v14, 0xbfb8aa3b, v9
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v1, v1, v24
	v_add_f32_e32 v5, 1.0, v5
	v_mul_f32_e32 v1, v15, v1
	v_rcp_f32_e32 v5, v5
	v_add_f32_e32 v14, 1.0, v14
	v_mul_f32_e32 v15, 0xbfb8aa3b, v10
	v_rcp_f32_e32 v14, v14
	v_exp_f32_e32 v15, v15
	v_mul_f32_e32 v5, v8, v5
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v8, v16, v5
	v_mul_f32_e32 v5, v9, v14
	v_add_f32_e32 v9, 1.0, v15
	v_rcp_f32_e32 v9, v9
	v_mul_f32_e32 v14, 0xbfb8aa3b, v11
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v15, v17, v5
	v_mul_f32_e32 v5, v10, v9
	v_mul_f32_e32 v10, 0xbfb8aa3b, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v9, v18, v5
	v_add_f32_e32 v5, 1.0, v14
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v14, 0xbfb8aa3b, v13
	v_rcp_f32_e32 v5, v5
	v_exp_f32_e32 v14, v14
	v_add_f32_e32 v10, 1.0, v10
	v_rcp_f32_e32 v10, v10
	v_mul_f32_e32 v5, v11, v5
	v_add_f32_e32 v11, 1.0, v14
	v_rcp_f32_e32 v11, v11
	v_mul_f32_e32 v14, v19, v5
	v_mul_f32_e32 v5, v12, v10
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v10, v20, v5
	v_mul_f32_e32 v5, v13, v11
	v_mul_f32_e32 v11, v21, v5
	v_max_f32_e64 v5, |v0|, |v1|
	v_max3_f32 v5, v5, |v8|, |v15|
	v_max3_f32 v5, v5, |v9|, |v14|
	v_max3_f32 v5, v5, |v10|, |v11|
	v_or_b32_e32 v20, 0x4018, v6
	s_nop 0
	v_mov_b32_dpp v12, v5 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v12, v12, v12
	v_max_f32_e32 v5, v5, v12
	s_nop 1
	v_mov_b32_dpp v12, v5 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v12, v12, v12
	v_max_f32_e32 v5, v5, v12
	v_add_u32_e32 v5, 0x200000, v5
	v_mul_f32_e32 v5, 0x3e800000, v5
	v_mov_b32_e32 v12, 0
	v_cvt_scalef32_pk_fp4_f32 v12, v0, v1, v5
	v_or3_b32 v0, v22, s6, v7
	v_cvt_scalef32_pk_fp4_f32 v12, v8, v15, v5 op_sel:[0,0,1,0]
	v_lshl_add_u32 v0, v23, 8, v0
	v_cvt_scalef32_pk_fp4_f32 v12, v9, v14, v5 op_sel:[0,0,0,1]
	v_ashrrev_i32_e32 v1, 31, v0
	v_cvt_scalef32_pk_fp4_f32 v12, v10, v11, v5 op_sel:[0,0,1,1]
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[0:1]
	global_store_dword v[8:9], v12, off nt
	v_or_b32_e32 v1, 0x4000, v6
	v_or_b32_e32 v12, 0x4008, v6
	v_or_b32_e32 v14, 0x4208, v6
	v_or_b32_e32 v7, 0x4200, v6
	ds_read2_b32 v[8:9], v1 offset1:1
	ds_read2_b32 v[10:11], v7 offset1:1
	ds_read2_b32 v[12:13], v12 offset1:1
	ds_read2_b32 v[14:15], v14 offset1:1
	v_or_b32_e32 v1, 0x4010, v6
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v16, 0xbfb8aa3b, v8
	v_exp_f32_e32 v16, v16
	v_mul_f32_e32 v17, 0xbfb8aa3b, v9
	v_exp_f32_e32 v17, v17
	v_or_b32_e32 v22, 0x4218, v6
	v_add_f32_e32 v16, 1.0, v16
	v_rcp_f32_e32 v24, v16
	v_add_f32_e32 v16, 1.0, v17
	v_rcp_f32_e32 v25, v16
	v_or_b32_e32 v7, 0x4210, v6
	ds_read2_b32 v[16:17], v1 offset1:1
	ds_read2_b32 v[18:19], v7 offset1:1
	ds_read2_b32 v[20:21], v20 offset1:1
	ds_read2_b32 v[22:23], v22 offset1:1
	v_mul_f32_e32 v1, v8, v24
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v8, 0xbfb8aa3b, v12
	v_mul_f32_e32 v7, v10, v1
	v_mul_f32_e32 v1, v9, v25
	v_exp_f32_e32 v8, v8
	v_mul_f32_e32 v9, 0xbfb8aa3b, v13
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v10, v11, v1
	v_add_f32_e32 v1, 1.0, v8
	v_rcp_f32_e32 v1, v1
	v_add_f32_e32 v8, 1.0, v9
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v9, 0xbfb8aa3b, v16
	v_rcp_f32_e32 v8, v8
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v1, v12, v1
	v_mul_f32_e32 v11, v14, v1
	v_mul_f32_e32 v1, v13, v8
	v_add_f32_e32 v8, 1.0, v9
	v_rcp_f32_e32 v8, v8
	v_mul_f32_e32 v9, 0xbfb8aa3b, v17
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v12, v15, v1
	v_mul_f32_e32 v1, v16, v8
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v8, v18, v1
	v_add_f32_e32 v1, 1.0, v9
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v9, 0xbfb8aa3b, v20
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v13, 0xbfb8aa3b, v21
	v_exp_f32_e32 v13, v13
	v_rcp_f32_e32 v1, v1
	v_add_f32_e32 v9, 1.0, v9
	v_rcp_f32_e32 v9, v9
	v_add_f32_e32 v13, 1.0, v13
	v_rcp_f32_e32 v13, v13
	v_mul_f32_e32 v1, v17, v1
	v_mul_f32_e32 v14, v19, v1
	v_mul_f32_e32 v1, v20, v9
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v9, v22, v1
	v_mul_f32_e32 v1, v21, v13
	v_mul_f32_e32 v13, v23, v1
	v_max_f32_e64 v1, |v7|, |v10|
	v_max3_f32 v1, v1, |v11|, |v12|
	v_max3_f32 v1, v1, |v8|, |v14|
	v_max3_f32 v1, v1, |v9|, |v13|
	v_or_b32_e32 v18, 0x8210, v6
	v_or_b32_e32 v20, 0x8018, v6
	v_mov_b32_dpp v15, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v15, v15, v15
	v_max_f32_e32 v1, v1, v15
	v_or_b32_e32 v22, 0x8218, v6
	s_nop 0
	v_mov_b32_dpp v15, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v15, v15, v15
	v_max_f32_e32 v1, v1, v15
	v_add_u32_e32 v1, 0x200000, v1
	v_mul_f32_e32 v1, 0x3e800000, v1
	v_mov_b32_e32 v15, 0
	v_cvt_scalef32_pk_fp4_f32 v15, v7, v10, v1
	v_cvt_scalef32_pk_fp4_f32 v15, v11, v12, v1 op_sel:[0,0,1,0]
	v_or_b32_e32 v7, 0x8000, v6
	v_cvt_scalef32_pk_fp4_f32 v15, v8, v14, v1 op_sel:[0,0,0,1]
	v_add_u32_e32 v8, 0x1000, v0
	v_cvt_scalef32_pk_fp4_f32 v15, v9, v13, v1 op_sel:[0,0,1,1]
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[8:9]
	global_store_dword v[8:9], v15, off nt
	v_or_b32_e32 v10, 0x8200, v6
	v_or_b32_e32 v12, 0x8008, v6
	v_or_b32_e32 v14, 0x8208, v6
	ds_read2_b32 v[8:9], v7 offset1:1
	ds_read2_b32 v[10:11], v10 offset1:1
	ds_read2_b32 v[12:13], v12 offset1:1
	ds_read2_b32 v[14:15], v14 offset1:1
	v_or_b32_e32 v7, 0x8010, v6
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v16, 0xbfb8aa3b, v8
	v_exp_f32_e32 v16, v16
	v_mul_f32_e32 v17, 0xbfb8aa3b, v9
	v_exp_f32_e32 v17, v17
	v_add_f32_e32 v16, 1.0, v16
	v_rcp_f32_e32 v24, v16
	v_add_f32_e32 v16, 1.0, v17
	v_rcp_f32_e32 v25, v16
	ds_read2_b32 v[16:17], v7 offset1:1
	ds_read2_b32 v[18:19], v18 offset1:1
	ds_read2_b32 v[20:21], v20 offset1:1
	ds_read2_b32 v[22:23], v22 offset1:1
	v_mul_f32_e32 v7, v8, v24
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v8, v10, v7
	v_mul_f32_e32 v7, v9, v25
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v9, 0xbfb8aa3b, v12
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v10, 0xbfb8aa3b, v13
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v11, v11, v7
	v_add_f32_e32 v7, 1.0, v9
	v_rcp_f32_e32 v7, v7
	v_add_f32_e32 v9, 1.0, v10
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v10, 0xbfb8aa3b, v16
	v_rcp_f32_e32 v9, v9
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v7, v12, v7
	v_mul_f32_e32 v12, v14, v7
	v_mul_f32_e32 v7, v13, v9
	v_add_f32_e32 v9, 1.0, v10
	v_rcp_f32_e32 v9, v9
	v_mul_f32_e32 v10, 0xbfb8aa3b, v17
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v13, v15, v7
	v_mul_f32_e32 v7, v16, v9
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v9, v18, v7
	v_add_f32_e32 v7, 1.0, v10
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v10, 0xbfb8aa3b, v20
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v14, 0xbfb8aa3b, v21
	v_exp_f32_e32 v14, v14
	v_rcp_f32_e32 v7, v7
	v_add_f32_e32 v10, 1.0, v10
	v_rcp_f32_e32 v10, v10
	v_add_f32_e32 v14, 1.0, v14
	v_rcp_f32_e32 v14, v14
	v_mul_f32_e32 v7, v17, v7
	v_mul_f32_e32 v15, v19, v7
	v_mul_f32_e32 v7, v20, v10
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v10, v22, v7
	v_mul_f32_e32 v7, v21, v14
	v_mul_f32_e32 v14, v23, v7
	v_max_f32_e64 v7, |v8|, |v11|
	v_max3_f32 v7, v7, |v12|, |v13|
	v_max3_f32 v7, v7, |v9|, |v15|
	v_max3_f32 v7, v7, |v10|, |v14|
	v_or_b32_e32 v18, 0xc210, v6
	v_or_b32_e32 v20, 0xc018, v6
	v_mov_b32_dpp v16, v7 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e32 v7, v7, v16
	v_or_b32_e32 v22, 0xc218, v6
	s_nop 0
	v_mov_b32_dpp v16, v7 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v16, v16, v16
	v_max_f32_e32 v7, v7, v16
	v_add_u32_e32 v7, 0x200000, v7
	v_mul_f32_e32 v7, 0x3e800000, v7
	v_mov_b32_e32 v16, 0
	v_cvt_scalef32_pk_fp4_f32 v16, v8, v11, v7
	v_cvt_scalef32_pk_fp4_f32 v16, v12, v13, v7 op_sel:[0,0,1,0]
	v_add_u32_e32 v8, 0x2000, v0
	v_cvt_scalef32_pk_fp4_f32 v16, v9, v15, v7 op_sel:[0,0,0,1]
	v_ashrrev_i32_e32 v9, 31, v8
	v_cvt_scalef32_pk_fp4_f32 v16, v10, v14, v7 op_sel:[0,0,1,1]
	v_lshl_add_u64 v[8:9], s[4:5], 0, v[8:9]
	global_store_dword v[8:9], v16, off nt
	v_or_b32_e32 v8, 0xc000, v6
	v_or_b32_e32 v10, 0xc200, v6
	v_or_b32_e32 v12, 0xc008, v6
	v_or_b32_e32 v14, 0xc208, v6
	ds_read2_b32 v[8:9], v8 offset1:1
	ds_read2_b32 v[10:11], v10 offset1:1
	ds_read2_b32 v[12:13], v12 offset1:1
	ds_read2_b32 v[14:15], v14 offset1:1
	v_or_b32_e32 v16, 0xc010, v6
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v17, 0xbfb8aa3b, v8
	v_exp_f32_e32 v17, v17
	v_mul_f32_e32 v19, 0xbfb8aa3b, v9
	v_exp_f32_e32 v19, v19
	v_add_f32_e32 v17, 1.0, v17
	v_rcp_f32_e32 v24, v17
	v_add_f32_e32 v17, 1.0, v19
	v_rcp_f32_e32 v25, v17
	ds_read2_b32 v[16:17], v16 offset1:1
	ds_read2_b32 v[18:19], v18 offset1:1
	ds_read2_b32 v[20:21], v20 offset1:1
	ds_read2_b32 v[22:23], v22 offset1:1
	v_mul_f32_e32 v8, v8, v24
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v10, v10, v8
	v_mul_f32_e32 v8, v9, v25
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v9, 0xbfb8aa3b, v12
	v_exp_f32_e32 v9, v9
	v_mul_f32_e32 v24, 0xbfb8aa3b, v13
	v_exp_f32_e32 v24, v24
	v_mul_f32_e32 v11, v11, v8
	v_add_f32_e32 v8, 1.0, v9
	v_rcp_f32_e32 v8, v8
	v_add_f32_e32 v9, 1.0, v24
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v16
	v_rcp_f32_e32 v9, v9
	v_exp_f32_e32 v24, v24
	v_mul_f32_e32 v8, v12, v8
	v_mul_f32_e32 v12, v14, v8
	v_mul_f32_e32 v8, v13, v9
	v_add_f32_e32 v9, 1.0, v24
	v_rcp_f32_e32 v9, v9
	v_mul_f32_e32 v13, 0xbfb8aa3b, v17
	v_exp_f32_e32 v13, v13
	v_mul_f32_e32 v14, v15, v8
	v_mul_f32_e32 v8, v16, v9
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v9, v18, v8
	v_add_f32_e32 v8, 1.0, v13
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v13, 0xbfb8aa3b, v20
	v_exp_f32_e32 v13, v13
	v_mul_f32_e32 v15, 0xbfb8aa3b, v21
	v_exp_f32_e32 v15, v15
	v_rcp_f32_e32 v8, v8
	v_add_f32_e32 v13, 1.0, v13
	v_rcp_f32_e32 v13, v13
	v_add_f32_e32 v15, 1.0, v15
	v_rcp_f32_e32 v15, v15
	v_mul_f32_e32 v8, v17, v8
	v_mul_f32_e32 v16, v19, v8
	v_mul_f32_e32 v8, v20, v13
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v13, v22, v8
	v_mul_f32_e32 v8, v21, v15
	v_mul_f32_e32 v15, v23, v8
	v_max_f32_e64 v8, |v10|, |v11|
	v_max3_f32 v8, v8, |v12|, |v14|
	v_max3_f32 v8, v8, |v9|, |v16|
	v_max3_f32 v8, v8, |v13|, |v15|
	v_or_b32_e32 v20, 0x10210, v6
	v_or_b32_e32 v22, 0x10018, v6
	v_mov_b32_dpp v17, v8 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v17, v17, v17
	v_max_f32_e32 v8, v8, v17
	v_or_b32_e32 v24, 0x10218, v6
	s_nop 0
	v_mov_b32_dpp v17, v8 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v17, v17, v17
	v_max_f32_e32 v8, v8, v17
	v_add_u32_e32 v8, 0x200000, v8
	v_mul_f32_e32 v8, 0x3e800000, v8
	v_mov_b32_e32 v17, 0
	v_cvt_scalef32_pk_fp4_f32 v17, v10, v11, v8
	v_cvt_scalef32_pk_fp4_f32 v17, v12, v14, v8 op_sel:[0,0,1,0]
	v_add_u32_e32 v10, 0x3000, v0
	v_cvt_scalef32_pk_fp4_f32 v17, v9, v16, v8 op_sel:[0,0,0,1]
	v_ashrrev_i32_e32 v11, 31, v10
	v_cvt_scalef32_pk_fp4_f32 v17, v13, v15, v8 op_sel:[0,0,1,1]
	v_lshl_add_u64 v[10:11], s[4:5], 0, v[10:11]
	global_store_dword v[10:11], v17, off nt
	v_or_b32_e32 v9, 0x10000, v6
	v_or_b32_e32 v12, 0x10200, v6
	v_or_b32_e32 v14, 0x10008, v6
	v_or_b32_e32 v16, 0x10208, v6
	ds_read2_b32 v[10:11], v9 offset1:1
	ds_read2_b32 v[12:13], v12 offset1:1
	ds_read2_b32 v[14:15], v14 offset1:1
	ds_read2_b32 v[16:17], v16 offset1:1
	v_or_b32_e32 v9, 0x10010, v6
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v18, 0xbfb8aa3b, v10
	v_exp_f32_e32 v18, v18
	v_mul_f32_e32 v19, 0xbfb8aa3b, v11
	v_exp_f32_e32 v19, v19
	v_add_f32_e32 v18, 1.0, v18
	v_rcp_f32_e32 v26, v18
	v_add_f32_e32 v18, 1.0, v19
	v_rcp_f32_e32 v27, v18
	ds_read2_b32 v[18:19], v9 offset1:1
	ds_read2_b32 v[20:21], v20 offset1:1
	ds_read2_b32 v[22:23], v22 offset1:1
	ds_read2_b32 v[24:25], v24 offset1:1
	v_mul_f32_e32 v9, v10, v26
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v10, v12, v9
	v_mul_f32_e32 v9, v11, v27
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v11, 0xbfb8aa3b, v14
	v_exp_f32_e32 v11, v11
	v_mul_f32_e32 v12, 0xbfb8aa3b, v15
	v_exp_f32_e32 v12, v12
	v_mul_f32_e32 v13, v13, v9
	v_add_f32_e32 v9, 1.0, v11
	v_rcp_f32_e32 v9, v9
	v_add_f32_e32 v11, 1.0, v12
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v12, 0xbfb8aa3b, v18
	v_rcp_f32_e32 v11, v11
	v_exp_f32_e32 v12, v12
	v_mul_f32_e32 v9, v14, v9
	v_mul_f32_e32 v14, v16, v9
	v_mul_f32_e32 v9, v15, v11
	v_add_f32_e32 v11, 1.0, v12
	v_rcp_f32_e32 v11, v11
	v_mul_f32_e32 v12, 0xbfb8aa3b, v19
	v_exp_f32_e32 v12, v12
	v_mul_f32_e32 v15, v17, v9
	v_mul_f32_e32 v9, v18, v11
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v11, v20, v9
	v_add_f32_e32 v9, 1.0, v12
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v12, 0xbfb8aa3b, v22
	v_exp_f32_e32 v12, v12
	v_mul_f32_e32 v16, 0xbfb8aa3b, v23
	v_exp_f32_e32 v16, v16
	v_rcp_f32_e32 v9, v9
	v_add_f32_e32 v12, 1.0, v12
	v_rcp_f32_e32 v12, v12
	v_add_f32_e32 v16, 1.0, v16
	v_rcp_f32_e32 v16, v16
	v_mul_f32_e32 v9, v19, v9
	v_mul_f32_e32 v17, v21, v9
	v_mul_f32_e32 v9, v22, v12
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v12, v24, v9
	v_mul_f32_e32 v9, v23, v16
	v_mul_f32_e32 v16, v25, v9
	v_max_f32_e64 v9, |v10|, |v13|
	v_max3_f32 v9, v9, |v14|, |v15|
	v_max3_f32 v9, v9, |v11|, |v17|
	v_max3_f32 v9, v9, |v12|, |v16|
	v_or_b32_e32 v20, 0x14210, v6
	v_or_b32_e32 v22, 0x14018, v6
	v_mov_b32_dpp v18, v9 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v18, v18, v18
	v_max_f32_e32 v9, v9, v18
	v_or_b32_e32 v24, 0x14218, v6
	s_nop 0
	v_mov_b32_dpp v18, v9 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v18, v18, v18
	v_max_f32_e32 v9, v9, v18
	v_add_u32_e32 v9, 0x200000, v9
	v_mul_f32_e32 v9, 0x3e800000, v9
	v_mov_b32_e32 v18, 0
	v_cvt_scalef32_pk_fp4_f32 v18, v10, v13, v9
	v_cvt_scalef32_pk_fp4_f32 v18, v14, v15, v9 op_sel:[0,0,1,0]
	v_add_u32_e32 v10, 0x4000, v0
	v_cvt_scalef32_pk_fp4_f32 v18, v11, v17, v9 op_sel:[0,0,0,1]
	v_ashrrev_i32_e32 v11, 31, v10
	v_cvt_scalef32_pk_fp4_f32 v18, v12, v16, v9 op_sel:[0,0,1,1]
	v_lshl_add_u64 v[10:11], s[4:5], 0, v[10:11]
	global_store_dword v[10:11], v18, off nt
	v_or_b32_e32 v10, 0x14000, v6
	v_or_b32_e32 v12, 0x14200, v6
	v_or_b32_e32 v14, 0x14008, v6
	v_or_b32_e32 v16, 0x14208, v6
	ds_read2_b32 v[10:11], v10 offset1:1
	ds_read2_b32 v[12:13], v12 offset1:1
	ds_read2_b32 v[14:15], v14 offset1:1
	ds_read2_b32 v[16:17], v16 offset1:1
	v_or_b32_e32 v18, 0x14010, v6
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v19, 0xbfb8aa3b, v10
	v_exp_f32_e32 v19, v19
	v_mul_f32_e32 v21, 0xbfb8aa3b, v11
	v_exp_f32_e32 v21, v21
	v_add_f32_e32 v19, 1.0, v19
	v_rcp_f32_e32 v26, v19
	v_add_f32_e32 v19, 1.0, v21
	v_rcp_f32_e32 v27, v19
	ds_read2_b32 v[18:19], v18 offset1:1
	ds_read2_b32 v[20:21], v20 offset1:1
	ds_read2_b32 v[22:23], v22 offset1:1
	ds_read2_b32 v[24:25], v24 offset1:1
	v_mul_f32_e32 v10, v10, v26
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v12, v12, v10
	v_mul_f32_e32 v10, v11, v27
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v11, 0xbfb8aa3b, v14
	v_exp_f32_e32 v11, v11
	v_mul_f32_e32 v26, 0xbfb8aa3b, v15
	v_exp_f32_e32 v26, v26
	v_mul_f32_e32 v13, v13, v10
	v_add_f32_e32 v10, 1.0, v11
	v_rcp_f32_e32 v10, v10
	v_add_f32_e32 v11, 1.0, v26
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v26, 0xbfb8aa3b, v18
	v_rcp_f32_e32 v11, v11
	v_exp_f32_e32 v26, v26
	v_mul_f32_e32 v10, v14, v10
	v_mul_f32_e32 v14, v16, v10
	v_mul_f32_e32 v10, v15, v11
	v_add_f32_e32 v11, 1.0, v26
	v_rcp_f32_e32 v11, v11
	v_mul_f32_e32 v15, 0xbfb8aa3b, v19
	v_exp_f32_e32 v15, v15
	v_mul_f32_e32 v16, v17, v10
	v_mul_f32_e32 v10, v18, v11
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v11, v20, v10
	v_add_f32_e32 v10, 1.0, v15
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v15, 0xbfb8aa3b, v22
	v_exp_f32_e32 v15, v15
	v_mul_f32_e32 v17, 0xbfb8aa3b, v23
	v_exp_f32_e32 v17, v17
	v_rcp_f32_e32 v10, v10
	v_add_f32_e32 v15, 1.0, v15
	v_rcp_f32_e32 v15, v15
	v_add_f32_e32 v17, 1.0, v17
	v_rcp_f32_e32 v17, v17
	v_mul_f32_e32 v10, v19, v10
	v_mul_f32_e32 v18, v21, v10
	v_mul_f32_e32 v10, v22, v15
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v15, v24, v10
	v_mul_f32_e32 v10, v23, v17
	v_mul_f32_e32 v17, v25, v10
	v_max_f32_e64 v10, |v12|, |v13|
	v_max3_f32 v10, v10, |v14|, |v16|
	v_max3_f32 v10, v10, |v11|, |v18|
	v_max3_f32 v10, v10, |v15|, |v17|
	v_or_b32_e32 v22, 0x18210, v6
	v_or_b32_e32 v24, 0x18018, v6
	v_mov_b32_dpp v19, v10 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v19, v19, v19
	v_max_f32_e32 v10, v10, v19
	v_or_b32_e32 v26, 0x18218, v6
	s_nop 0
	v_mov_b32_dpp v19, v10 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v19, v19, v19
	v_max_f32_e32 v10, v10, v19
	v_add_u32_e32 v10, 0x200000, v10
	v_mul_f32_e32 v10, 0x3e800000, v10
	v_mov_b32_e32 v19, 0
	v_cvt_scalef32_pk_fp4_f32 v19, v12, v13, v10
	v_cvt_scalef32_pk_fp4_f32 v19, v14, v16, v10 op_sel:[0,0,1,0]
	v_add_u32_e32 v12, 0x5000, v0
	v_cvt_scalef32_pk_fp4_f32 v19, v11, v18, v10 op_sel:[0,0,0,1]
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_scalef32_pk_fp4_f32 v19, v15, v17, v10 op_sel:[0,0,1,1]
	v_lshl_add_u64 v[12:13], s[4:5], 0, v[12:13]
	global_store_dword v[12:13], v19, off nt
	v_or_b32_e32 v11, 0x18000, v6
	v_or_b32_e32 v14, 0x18200, v6
	v_or_b32_e32 v16, 0x18008, v6
	v_or_b32_e32 v18, 0x18208, v6
	ds_read2_b32 v[12:13], v11 offset1:1
	ds_read2_b32 v[14:15], v14 offset1:1
	ds_read2_b32 v[16:17], v16 offset1:1
	ds_read2_b32 v[18:19], v18 offset1:1
	v_or_b32_e32 v11, 0x18010, v6
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v20, 0xbfb8aa3b, v12
	v_exp_f32_e32 v20, v20
	v_mul_f32_e32 v21, 0xbfb8aa3b, v13
	v_exp_f32_e32 v21, v21
	v_add_f32_e32 v20, 1.0, v20
	v_rcp_f32_e32 v28, v20
	v_add_f32_e32 v20, 1.0, v21
	v_rcp_f32_e32 v29, v20
	ds_read2_b32 v[20:21], v11 offset1:1
	ds_read2_b32 v[22:23], v22 offset1:1
	ds_read2_b32 v[24:25], v24 offset1:1
	ds_read2_b32 v[26:27], v26 offset1:1
	v_mul_f32_e32 v11, v12, v28
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v12, v14, v11
	v_mul_f32_e32 v11, v13, v29
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v13, 0xbfb8aa3b, v16
	v_exp_f32_e32 v13, v13
	v_mul_f32_e32 v14, 0xbfb8aa3b, v17
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v15, v15, v11
	v_add_f32_e32 v11, 1.0, v13
	v_rcp_f32_e32 v11, v11
	v_add_f32_e32 v13, 1.0, v14
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v14, 0xbfb8aa3b, v20
	v_rcp_f32_e32 v13, v13
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v11, v16, v11
	v_mul_f32_e32 v16, v18, v11
	v_mul_f32_e32 v11, v17, v13
	v_add_f32_e32 v13, 1.0, v14
	v_rcp_f32_e32 v13, v13
	v_mul_f32_e32 v14, 0xbfb8aa3b, v21
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v17, v19, v11
	v_mul_f32_e32 v11, v20, v13
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v13, v22, v11
	v_add_f32_e32 v11, 1.0, v14
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v14, 0xbfb8aa3b, v24
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v18, 0xbfb8aa3b, v25
	v_exp_f32_e32 v18, v18
	v_rcp_f32_e32 v11, v11
	v_add_f32_e32 v14, 1.0, v14
	v_rcp_f32_e32 v14, v14
	v_add_f32_e32 v18, 1.0, v18
	v_rcp_f32_e32 v18, v18
	v_mul_f32_e32 v11, v21, v11
	v_mul_f32_e32 v19, v23, v11
	v_mul_f32_e32 v11, v24, v14
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v14, v26, v11
	v_mul_f32_e32 v11, v25, v18
	v_mul_f32_e32 v18, v27, v11
	v_max_f32_e64 v11, |v12|, |v15|
	v_max3_f32 v11, v11, |v16|, |v17|
	v_max3_f32 v11, v11, |v13|, |v19|
	v_max3_f32 v11, v11, |v14|, |v18|
	v_or_b32_e32 v22, 0x1c210, v6
	v_or_b32_e32 v24, 0x1c018, v6
	v_mov_b32_dpp v20, v11 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v20, v20, v20
	v_max_f32_e32 v11, v11, v20
	s_nop 1
	v_mov_b32_dpp v20, v11 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v20, v20, v20
	v_max_f32_e32 v11, v11, v20
	v_add_u32_e32 v11, 0x200000, v11
	v_mul_f32_e32 v11, 0x3e800000, v11
	v_mov_b32_e32 v20, 0
	v_cvt_scalef32_pk_fp4_f32 v20, v12, v15, v11
	v_cvt_scalef32_pk_fp4_f32 v20, v16, v17, v11 op_sel:[0,0,1,0]
	v_add_u32_e32 v12, 0x6000, v0
	v_cvt_scalef32_pk_fp4_f32 v20, v13, v19, v11 op_sel:[0,0,0,1]
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_scalef32_pk_fp4_f32 v20, v14, v18, v11 op_sel:[0,0,1,1]
	v_lshl_add_u64 v[12:13], s[4:5], 0, v[12:13]
	global_store_dword v[12:13], v20, off nt
	v_or_b32_e32 v12, 0x1c000, v6
	v_or_b32_e32 v14, 0x1c200, v6
	v_or_b32_e32 v16, 0x1c008, v6
	v_or_b32_e32 v18, 0x1c208, v6
	ds_read2_b32 v[12:13], v12 offset1:1
	ds_read2_b32 v[14:15], v14 offset1:1
	ds_read2_b32 v[16:17], v16 offset1:1
	ds_read2_b32 v[18:19], v18 offset1:1
	v_or_b32_e32 v20, 0x1c010, v6
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v21, 0xbfb8aa3b, v12
	v_exp_f32_e32 v21, v21
	v_mul_f32_e32 v23, 0xbfb8aa3b, v13
	v_exp_f32_e32 v23, v23
	v_or_b32_e32 v6, 0x1c218, v6
	v_add_f32_e32 v21, 1.0, v21
	v_rcp_f32_e32 v28, v21
	v_add_f32_e32 v21, 1.0, v23
	v_rcp_f32_e32 v29, v21
	ds_read2_b32 v[20:21], v20 offset1:1
	ds_read2_b32 v[22:23], v22 offset1:1
	ds_read2_b32 v[24:25], v24 offset1:1
	ds_read2_b32 v[26:27], v6 offset1:1
	v_mul_f32_e32 v6, v12, v28
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v12, v14, v6
	v_mul_f32_e32 v6, v13, v29
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v13, 0xbfb8aa3b, v16
	v_exp_f32_e32 v13, v13
	v_mul_f32_e32 v14, 0xbfb8aa3b, v17
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v15, v15, v6
	v_add_f32_e32 v6, 1.0, v13
	v_rcp_f32_e32 v6, v6
	v_add_f32_e32 v13, 1.0, v14
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v14, 0xbfb8aa3b, v20
	v_rcp_f32_e32 v13, v13
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v6, v16, v6
	v_mul_f32_e32 v16, v18, v6
	v_mul_f32_e32 v6, v17, v13
	v_add_f32_e32 v13, 1.0, v14
	v_rcp_f32_e32 v13, v13
	v_mul_f32_e32 v14, 0xbfb8aa3b, v21
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v17, v19, v6
	v_mul_f32_e32 v6, v20, v13
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v13, v22, v6
	v_add_f32_e32 v6, 1.0, v14
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v14, 0xbfb8aa3b, v24
	v_exp_f32_e32 v14, v14
	v_mul_f32_e32 v18, 0xbfb8aa3b, v25
	v_exp_f32_e32 v18, v18
	v_rcp_f32_e32 v6, v6
	v_add_f32_e32 v14, 1.0, v14
	v_rcp_f32_e32 v14, v14
	v_add_f32_e32 v18, 1.0, v18
	v_rcp_f32_e32 v18, v18
	v_mul_f32_e32 v6, v21, v6
	v_mul_f32_e32 v19, v23, v6
	v_mul_f32_e32 v6, v24, v14
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v14, v26, v6
	v_mul_f32_e32 v6, v25, v18
	v_mul_f32_e32 v18, v27, v6
	v_max_f32_e64 v6, |v12|, |v15|
	v_max3_f32 v6, v6, |v16|, |v17|
	v_max3_f32 v6, v6, |v13|, |v19|
	v_max3_f32 v6, v6, |v14|, |v18|
	s_nop 1
	v_mov_b32_dpp v20, v6 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v20, v20, v20
	v_max_f32_e32 v6, v6, v20
	s_nop 1
	v_mov_b32_dpp v20, v6 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_max_f32_e32 v20, v20, v20
	v_max_f32_e32 v6, v6, v20
	v_add_u32_e32 v6, 0x200000, v6
	v_mul_f32_e32 v6, 0x3e800000, v6
	v_cvt_scalef32_pk_fp4_f32 v98, v12, v15, v6
	v_cvt_scalef32_pk_fp4_f32 v98, v16, v17, v6 op_sel:[0,0,1,0]
	v_add_u32_e32 v12, 0x7000, v0
	v_cvt_scalef32_pk_fp4_f32 v98, v13, v19, v6 op_sel:[0,0,0,1]
	v_ashrrev_i32_e32 v13, 31, v12
	v_cvt_scalef32_pk_fp4_f32 v98, v14, v18, v6 op_sel:[0,0,1,1]
	v_lshl_add_u64 v[12:13], s[4:5], 0, v[12:13]
	global_store_dword v[12:13], v98, off nt
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_3
; %bb.2:
	v_lshrrev_b32_e32 v0, 23, v6
	s_movk_i32 s3, 0xfe
	v_min_u32_sdwa v4, v0, s3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshrrev_b32_e32 v0, 23, v11
	v_min_u32_e32 v6, 0xfe, v0
	v_lshrrev_b32_e32 v0, 23, v10
	v_min_u32_sdwa v10, v0, s3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshrrev_b32_e32 v0, 23, v9
	v_min_u32_e32 v9, 0xfe, v0
	v_lshrrev_b32_e32 v0, 23, v8
	v_min_u32_sdwa v8, v0, s3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshrrev_b32_e32 v0, 23, v7
	v_min_u32_e32 v7, 0xfe, v0
	v_lshrrev_b32_e32 v0, 23, v1
	v_min_u32_sdwa v0, v0, s3 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_lshl_b32 s3, s12, 5
	s_lshl_b32 s2, s2, 9
	s_and_b32 s3, s3, 0x3fffffc0
	s_load_dwordx2 s[0:1], s[0:1], 0x48
	v_lshrrev_b32_e32 v1, 23, v5
	s_add_i32 s3, s3, s2
	v_min_u32_e32 v1, 0xfe, v1
	v_or_b32_e32 v3, s3, v3
	s_lshl_b32 s2, s12, 1
	s_and_b32 s2, s2, 2
	v_or_b32_e32 v5, v0, v1
	v_lshlrev_b32_e32 v0, 6, v2
	v_lshlrev_b32_e32 v1, 2, v3
	v_or3_b32 v0, v1, v0, s2
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[0:1]
	global_store_short v[2:3], v5, off
	v_add_u32_e32 v2, 0x200, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_or_b32_e32 v1, v8, v7
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[2:3]
	global_store_short v[2:3], v1, off
	v_add_u32_e32 v2, 0x400, v0
	v_ashrrev_i32_e32 v3, 31, v2
	v_or_b32_e32 v1, v10, v9
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[2:3]
	v_add_u32_e32 v0, 0x600, v0
	global_store_short v[2:3], v1, off
	v_ashrrev_i32_e32 v1, 31, v0
	v_or_b32_e32 v2, v4, v6
	v_lshl_add_u64 v[0:1], s[0:1], 0, v[0:1]
	global_store_short v[0:1], v2, off
.LBB0_3:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
		.amdhsa_group_segment_fixed_size 131072
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
		.amdhsa_next_free_vgpr 324
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 172
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
	.section	.text,"axG",@progbits,_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16,comdat,unique,1
.Lfunc_end0:
	.size	_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16, .Lfunc_end0-_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
                                        ; -- End function
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.num_vgpr, 169
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.num_agpr, 152
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.numbered_sgpr, 42
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.private_seg_size, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.uses_vcc, 1
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.uses_flat_scratch, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_dyn_sized_stack, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_recursion, 0
	.set _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 45148
; TotalNumSgprs: 48
; NumVgprs: 169
; NumAgprs: 152
; TotalNumVgprs: 324
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 192
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 40
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 324
; AccumOffset: 172
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 42
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_b90e201c2ca75581,@object ; @__hip_cuid_b90e201c2ca75581
	.section	.bss,"aw",@nobits,unique,2
	.globl	__hip_cuid_b90e201c2ca75581
__hip_cuid_b90e201c2ca75581:
	.byte	0                               ; 0x0
	.size	__hip_cuid_b90e201c2ca75581, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_b90e201c2ca75581
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     152
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
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         _ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi128ELb0ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16.kd
    .uses_dynamic_stack: false
    .vgpr_count:     324
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
