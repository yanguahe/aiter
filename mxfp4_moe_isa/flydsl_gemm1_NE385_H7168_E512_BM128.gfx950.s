	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	gemm1_kernel_0                  ; -- Begin function gemm1_kernel_0
	.p2align	8
	.type	gemm1_kernel_0,@function
gemm1_kernel_0:                         ; @gemm1_kernel_0
; %bb.0:
	s_load_dwordx2 s[34:35], s[0:1], 0x50
	s_load_dwordx16 s[4:19], s[0:1], 0x0
	s_mov_b32 s27, 0x27000
	s_and_b32 s33, s2, 3
	v_lshrrev_b32_e32 v88, 6, v0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s3, s35, 5
	s_and_b32 s26, s3, -4
	s_and_b32 s3, s2, -4
	s_and_b32 s25, s13, 0xffff
	s_mov_b32 s24, s12
	v_mov_b32_e32 v1, s3
	buffer_load_dword v2, v1, s[24:27], 0 offen
	s_and_b32 s25, s15, 0xffff
	s_mov_b32 s26, 4
	s_mov_b32 s24, s14
	buffer_load_dword v1, off, s[24:27], 0
	s_lshl_b32 s13, s33, 3
	s_ashr_i32 s3, s2, 31
	v_lshl_or_b32 v3, v88, 1, s13
	s_mov_b32 s12, 0xe000
	s_lshr_b64 s[28:29], s[2:3], 2
	v_mul_hi_u32_u24_e32 v5, 0x700, v3
	v_mul_u32_u24_e32 v4, 0x700, v3
	s_lshl_b32 s3, s33, 8
	s_lshl_b64 s[30:31], s[28:29], 7
	v_and_b32_e32 v6, 0xc0, v0
	s_mov_b32 s36, 0
	s_waitcnt vmcnt(1)
	v_ashrrev_i32_e32 v3, 31, v2
	v_mad_i64_i32 v[4:5], s[12:13], v2, s12, v[4:5]
	v_lshlrev_b64 v[2:3], 10, v[2:3]
	s_waitcnt vmcnt(0)
	v_cmp_ge_u32_e32 vcc, s30, v1
	v_or_b32_e32 v1, s3, v2
	v_or_b32_e32 v1, v1, v6
	s_nop 0
	v_readfirstlane_b32 s29, v1
	v_readfirstlane_b32 s3, v4
	s_cbranch_vccnz .LBB0_3
; %bb.1:
	v_bfe_u32 v2, v0, 3, 27
	v_or_b32_e32 v3, 0x60, v2
	v_or_b32_e32 v1, s30, v3
	v_or_b32_e32 v4, 64, v2
	v_or_b32_e32 v6, 32, v2
	s_lshl_b32 s14, s35, 2
	s_and_b32 s13, s17, 0xffff
	s_mov_b32 s12, s16
	s_mov_b32 s15, s27
	v_lshlrev_b32_e32 v1, 2, v1
	v_or_b32_e32 v5, s30, v4
	v_or_b32_e32 v7, s30, v6
	v_or_b32_e32 v8, s30, v2
	v_lshlrev_b32_e32 v5, 2, v5
	v_lshlrev_b32_e32 v7, 2, v7
	v_lshlrev_b32_e32 v8, 2, v8
	buffer_load_dword v9, v1, s[12:15], 0 offen
	buffer_load_dword v10, v5, s[12:15], 0 offen
	buffer_load_dword v11, v7, s[12:15], 0 offen
	buffer_load_dword v12, v8, s[12:15], 0 offen
	s_and_b32 s21, s11, 0xffff
	s_mul_i32 s11, s29, 0xe00
	v_and_b32_e32 v5, 63, v0
	s_movk_i32 s29, 0x380
	v_lshlrev_b32_e32 v90, 2, v0
	v_and_b32_e32 v8, 28, v90
	v_lshlrev_b32_e32 v91, 2, v5
	s_mul_i32 s26, s34, 0xe00
	s_and_b32 s25, s5, 0xffff
	s_mov_b32 s24, s4
	s_ashr_i32 s39, s35, 31
	s_mov_b32 s38, s35
	s_mov_b32 s12, s8
	s_lshl_b32 s8, s3, 2
	s_mov_b32 s4, s6
	s_lshr_b32 s3, s30, 5
	v_readfirstlane_b32 s6, v88
	s_lshr_b64 s[34:35], s[38:39], 5
	s_mul_i32 s35, s3, 0x1c00
	s_lshl_b32 s37, s6, 10
	s_add_u32 s3, s34, 2
	s_and_b32 s5, s7, 0xffff
	s_mov_b32 s7, s27
	v_lshlrev_b32_e32 v68, 4, v0
	s_lshl_b32 s38, s6, 8
	s_add_i32 m0, s37, 0x18000
	s_mul_i32 s6, s3, 0x1c00
	s_add_i32 s34, s35, 0x1000
	buffer_load_dwordx4 v68, s[4:7], s35 offen lds
	s_or_b32 m0, s38, 0x19000
	s_add_i32 s39, s35, 0x1400
	buffer_load_dword v90, s[4:7], s34 offen lds
	s_or_b32 m0, s38, 0x19400
	s_add_i32 s40, s35, 0x1800
	buffer_load_dword v90, s[4:7], s39 offen lds
	s_or_b32 m0, s38, 0x19800
	s_add_i32 s41, s35, 0x1c00
	buffer_load_dword v90, s[4:7], s40 offen lds
	s_add_i32 m0, s37, 0x19c00
	s_add_i32 s42, s35, 0x3000
	buffer_load_dwordx4 v68, s[4:7], s41 offen lds
	s_or_b32 m0, s38, 0x1ac00
	s_add_i32 s43, s35, 0x3400
	s_add_i32 s44, s35, 0x3800
	s_add_i32 s45, s35, 0x4800
	s_add_i32 s46, s35, 0x4c00
	s_add_i32 s47, s35, 0x5000
	s_add_i32 s48, s35, 0x5400
	v_and_b32_e32 v89, 15, v0
	v_lshlrev_b32_e32 v7, 1, v0
	s_add_i32 s49, s35, 0x6400
	v_bfe_u32 v1, v0, 4, 2
	v_and_b32_e32 v7, 0xf0, v7
	v_lshlrev_b32_e32 v2, 8, v2
	v_lshlrev_b32_e32 v69, 4, v89
	s_add_i32 s50, s35, 0x6800
	s_and_b32 s13, s9, 0xffff
	s_mov_b32 s14, 0x54380000
	s_mov_b32 s22, 0x5438000
	s_mov_b32 s20, s10
	s_mov_b32 s23, s27
	s_add_i32 s51, s35, 0x6c00
	v_lshlrev_b32_e32 v6, 8, v6
	v_lshlrev_b32_e32 v4, 8, v4
	v_lshlrev_b32_e32 v3, 8, v3
	v_lshl_or_b32 v102, v1, 8, v69
	s_add_i32 s17, s11, 0xe000
	s_add_i32 s16, s11, 0x1c000
	s_add_i32 s10, s11, 0x2a000
	s_add_i32 s9, s8, 0x1c00
	buffer_load_dword v84, v91, s[20:23], s8 offen
	buffer_load_dword v85, v91, s[20:23], s9 offen
	buffer_load_dword v66, v91, s[20:23], s9 offen offset:256
	buffer_load_dword v67, v91, s[20:23], s8 offen offset:256
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_mov_b32 a3, a0
	v_or_b32_e32 v156, 0x19c00, v91
	v_or_b32_e32 v157, 0x1b800, v91
	v_or_b32_e32 v158, 0x1d400, v91
	s_waitcnt vmcnt(12)
	v_mul_lo_u32 v5, v9, s29
	s_waitcnt vmcnt(11)
	v_mul_lo_u32 v9, v10, s29
	s_waitcnt vmcnt(10)
	v_mul_lo_u32 v10, v11, s29
	s_waitcnt vmcnt(9)
	v_mul_lo_u32 v11, v12, s29
	v_or_b32_e32 v11, v11, v8
	v_or_b32_e32 v10, v10, v8
	v_or_b32_e32 v9, v9, v8
	v_or_b32_e32 v5, v5, v8
	v_lshlrev_b32_e32 v94, 2, v11
	v_lshlrev_b32_e32 v95, 2, v10
	v_lshlrev_b32_e32 v96, 2, v9
	v_lshlrev_b32_e32 v97, 2, v5
	buffer_load_dwordx4 v[70:73], v94, s[24:27], 0 offen
	buffer_load_dwordx4 v[74:77], v95, s[24:27], 0 offen
	buffer_load_dwordx4 v[78:81], v96, s[24:27], 0 offen
	buffer_load_dwordx4 v[104:107], v97, s[24:27], 0 offen
	buffer_load_dwordx4 v[108:111], v94, s[24:27], 0 offen offset:128
	buffer_load_dwordx4 v[112:115], v95, s[24:27], 0 offen offset:128
	buffer_load_dwordx4 v[116:119], v96, s[24:27], 0 offen offset:128
	buffer_load_dwordx4 v[120:123], v97, s[24:27], 0 offen offset:128
	s_add_i32 s29, s35, 0x2c00
	buffer_load_dword v90, s[4:7], s29 offen lds
	s_or_b32 m0, s38, 0x1b000
	v_lshlrev_b32_e32 v5, 2, v8
	buffer_load_dword v90, s[4:7], s42 offen lds
	s_or_b32 m0, s38, 0x1b400
	v_bitop3_b32 v98, v2, v7, v5 bitop3:0xf6
	buffer_load_dword v90, s[4:7], s43 offen lds
	s_add_i32 m0, s37, 0x1b800
	v_bitop3_b32 v99, v6, v7, v5 bitop3:0xf6
	buffer_load_dwordx4 v68, s[4:7], s44 offen lds
	s_or_b32 m0, s38, 0x1c800
	v_bitop3_b32 v100, v4, v7, v5 bitop3:0xf6
	buffer_load_dword v90, s[4:7], s45 offen lds
	s_or_b32 m0, s38, 0x1cc00
	v_bitop3_b32 v101, v3, v7, v5 bitop3:0xf6
	buffer_load_dword v90, s[4:7], s46 offen lds
	s_or_b32 m0, s38, 0x1d000
	buffer_load_dwordx4 v[26:29], v102, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[30:33], v102, s[12:15], s11 offen offset:1024 nt
	buffer_load_dwordx4 v[38:41], v102, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[34:37], v102, s[12:15], s17 offen offset:1024 nt
	buffer_load_dwordx4 v[46:49], v102, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[42:45], v102, s[12:15], s16 offen offset:1024 nt
	buffer_load_dwordx4 v[62:65], v102, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[58:61], v102, s[12:15], s10 offen offset:1024 nt
	buffer_load_dwordx4 v[54:57], v102, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[50:53], v102, s[12:15], s11 offen offset:3072 nt
	buffer_load_dwordx4 v[22:25], v102, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v102, s[12:15], s17 offen offset:3072 nt
	buffer_load_dwordx4 v[14:17], v102, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v102, s[12:15], s16 offen offset:3072 nt
	buffer_load_dwordx4 v[6:9], v102, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v102, s[12:15], s10 offen offset:3072 nt
	v_accvgpr_mov_b32 a1, a0
	buffer_load_dword v90, s[4:7], s47 offen lds
	s_add_i32 m0, s37, 0x1d400
	v_accvgpr_mov_b32 a2, a0
	buffer_load_dwordx4 v68, s[4:7], s48 offen lds
	s_or_b32 m0, s38, 0x1e400
	v_and_b32_e32 v68, 48, v0
	buffer_load_dword v90, s[4:7], s49 offen lds
	s_or_b32 m0, s38, 0x1e800
	v_accvgpr_mov_b32 a35, a3
	buffer_load_dword v90, s[4:7], s50 offen lds
	s_or_b32 m0, s38, 0x1ec00
	v_accvgpr_mov_b32 a34, a2
	buffer_load_dword v90, s[4:7], s51 offen lds
	v_accvgpr_mov_b32 a33, a1
	v_accvgpr_mov_b32 a32, a0
	v_accvgpr_mov_b32 a31, a3
	v_add_u32_e32 v103, 0x10000, v98
	v_accvgpr_mov_b32 a30, a2
	v_accvgpr_mov_b32 a29, a1
	v_accvgpr_mov_b32 a28, a0
	v_accvgpr_mov_b32 a27, a3
	v_accvgpr_mov_b32 a26, a2
	v_accvgpr_mov_b32 a25, a1
	v_accvgpr_mov_b32 a24, a0
	v_accvgpr_mov_b32 a23, a3
	v_accvgpr_mov_b32 a22, a2
	v_accvgpr_mov_b32 a21, a1
	v_accvgpr_mov_b32 a20, a0
	v_accvgpr_mov_b32 a19, a3
	v_accvgpr_mov_b32 a18, a2
	v_accvgpr_mov_b32 a17, a1
	v_accvgpr_mov_b32 a16, a0
	v_accvgpr_mov_b32 a15, a3
	v_accvgpr_mov_b32 a14, a2
	v_accvgpr_mov_b32 a13, a1
	v_accvgpr_mov_b32 a12, a0
	v_accvgpr_mov_b32 a11, a3
	v_accvgpr_mov_b32 a10, a2
	v_accvgpr_mov_b32 a9, a1
	v_accvgpr_mov_b32 a8, a0
	v_accvgpr_mov_b32 a7, a3
	s_waitcnt vmcnt(0)
	ds_write_b128 v98, v[70:73]
	ds_write_b128 v99, v[74:77]
	ds_write_b128 v100, v[78:81]
	ds_write_b128 v101, v[104:107]
	ds_write_b128 v98, v[108:111] offset:32768
	ds_write_b128 v99, v[112:115] offset:32768
	ds_write_b128 v100, v[116:119] offset:32768
	ds_write_b128 v101, v[120:123] offset:32768
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[160:163], v94, s[24:27], 0 offen offset:256
	buffer_load_dwordx4 v[164:167], v95, s[24:27], 0 offen offset:256
	buffer_load_dwordx4 v[168:171], v96, s[24:27], 0 offen offset:256
	buffer_load_dwordx4 v[172:175], v97, s[24:27], 0 offen offset:256
	v_lshlrev_b32_e32 v70, 8, v89
	v_or_b32_e32 v71, 64, v68
	v_bitop3_b32 v93, v69, v70, v68 bitop3:0xde
	v_or_b32_e32 v107, 0x18000, v91
	v_bitop3_b32 v92, v71, v70, v69 bitop3:0xde
	ds_read_b128 v[68:71], v93
	ds_read_b128 v[72:75], v93 offset:4096
	ds_read_b128 v[76:79], v92
	ds_read_b128 v[80:83], v92 offset:4096
	ds_read_b128 v[108:111], v93 offset:8192
	ds_read_b128 v[112:115], v93 offset:12288
	ds_read_b128 v[116:119], v92 offset:8192
	ds_read_b128 v[120:123], v92 offset:12288
	ds_read_b128 v[124:127], v93 offset:16384
	ds_read_b128 v[128:131], v93 offset:20480
	ds_read_b128 v[132:135], v92 offset:16384
	ds_read_b128 v[136:139], v92 offset:20480
	ds_read_b128 v[140:143], v93 offset:24576
	ds_read_b128 v[144:147], v93 offset:28672
	ds_read_b128 v[148:151], v92 offset:24576
	ds_read_b128 v[152:155], v92 offset:28672
	ds_read_b32 v107, v107
	ds_read_b32 v156, v156
	ds_read_b32 v157, v157
	ds_read_b32 v158, v158
	v_add_u32_e32 v104, 0x10000, v99
	v_add_u32_e32 v105, 0x10000, v100
	v_add_u32_e32 v106, 0x10000, v101
	v_accvgpr_mov_b32 a6, a2
	v_accvgpr_mov_b32 a5, a1
	v_accvgpr_mov_b32 a4, a0
	v_lshrrev_b32_e32 v86, 4, v0
	v_mov_b32_e32 v87, 0
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[160:163]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[164:167]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[168:171]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[172:175]
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:71], v[26:29], 0, v107, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[26:29], a[32:35], v107, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[30:33], 0, v107, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[30:33], a[32:35], v107, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[72:75], v[26:29], 0, v107, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[26:29], a[28:31], v107, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[30:33], 0, v107, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[30:33], a[28:31], v107, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:111], v[26:29], 0, v156, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[26:29], a[24:27], v156, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[116:119], v[30:33], 0, v156, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[30:33], a[24:27], v156, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[26:29], 0, v156, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[26:29], a[20:23], v156, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:123], v[30:33], 0, v156, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[30:33], a[20:23], v156, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[124:127], v[26:29], 0, v157, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[26:29], a[16:19], v157, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:135], v[30:33], 0, v157, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[30:33], a[16:19], v157, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[128:131], v[26:29], 0, v157, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:131], v[26:29], a[12:15], v157, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[30:33], 0, v157, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[30:33], a[12:15], v157, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[26:29], 0, v158, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[26:29], a[8:11], v158, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[148:151], v[30:33], 0, v158, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[30:33], a[8:11], v158, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[26:29], 0, v158, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:147], v[26:29], a[4:7], v158, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:155], v[30:33], 0, v158, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[30:33], a[4:7], v158, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v159, 0x1000, v102
	buffer_load_dwordx4 v[30:33], v159, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[26:29], v159, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	v_accvgpr_mov_b32 a83, a3
	v_accvgpr_mov_b32 a82, a2
	v_accvgpr_mov_b32 a81, a1
	v_accvgpr_mov_b32 a80, a0
	v_accvgpr_mov_b32 a75, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[68:71], v[38:41], 0, v107, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[38:41], a[80:83], v107, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a74, a2
	v_accvgpr_mov_b32 a73, a1
	v_accvgpr_mov_b32 a72, a0
	v_accvgpr_mov_b32 a71, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[76:79], v[34:37], 0, v107, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[34:37], a[80:83], v107, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a70, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[72:75], v[38:41], 0, v107, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[38:41], a[72:75], v107, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a69, a1
	v_accvgpr_mov_b32 a68, a0
	v_accvgpr_mov_b32 a63, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[80:83], v[34:37], 0, v107, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[34:37], a[72:75], v107, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a62, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[108:111], v[38:41], 0, v156, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:111], v[38:41], a[68:71], v156, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a61, a1
	v_accvgpr_mov_b32 a60, a0
	v_accvgpr_mov_b32 a59, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[116:119], v[34:37], 0, v156, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[34:37], a[68:71], v156, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a58, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[112:115], v[38:41], 0, v156, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[112:115], v[38:41], a[60:63], v156, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a57, a1
	v_accvgpr_mov_b32 a56, a0
	v_accvgpr_mov_b32 a51, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[120:123], v[34:37], 0, v156, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[120:123], v[34:37], a[60:63], v156, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a50, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[124:127], v[38:41], 0, v157, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[38:41], a[56:59], v157, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a49, a1
	v_accvgpr_mov_b32 a48, a0
	v_accvgpr_mov_b32 a43, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[132:135], v[34:37], 0, v157, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[34:37], a[56:59], v157, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a42, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[128:131], v[38:41], 0, v157, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[38:41], a[48:51], v157, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a41, a1
	v_accvgpr_mov_b32 a40, a0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[34:37], 0, v157, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[34:37], a[48:51], v157, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[140:143], v[38:41], 0, v158, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[38:41], a[40:43], v158, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[148:151], v[34:37], 0, v158, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[34:37], a[40:43], v158, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[38:41], 0, v158, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	v_accvgpr_mov_b32 a39, a3
	v_accvgpr_mov_b32 a38, a2
	v_accvgpr_mov_b32 a37, a1
	v_accvgpr_mov_b32 a36, a0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[38:41], a[36:39], v158, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[152:155], v[34:37], 0, v158, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:155], v[34:37], a[36:39], v158, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v159, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[34:37], v159, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	v_accvgpr_mov_b32 a111, a3
	v_accvgpr_mov_b32 a110, a2
	v_accvgpr_mov_b32 a109, a1
	v_accvgpr_mov_b32 a108, a0
	v_accvgpr_mov_b32 a103, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[68:71], v[46:49], 0, v107, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[46:49], a[108:111], v107, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a102, a2
	v_accvgpr_mov_b32 a101, a1
	v_accvgpr_mov_b32 a100, a0
	v_accvgpr_mov_b32 a99, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[76:79], v[42:45], 0, v107, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[76:79], v[42:45], a[108:111], v107, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a98, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[72:75], v[46:49], 0, v107, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[46:49], a[100:103], v107, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a97, a1
	v_accvgpr_mov_b32 a96, a0
	v_accvgpr_mov_b32 a95, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[80:83], v[42:45], 0, v107, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[42:45], a[100:103], v107, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a94, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[108:111], v[46:49], 0, v156, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[108:111], v[46:49], a[96:99], v156, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a93, a1
	v_accvgpr_mov_b32 a92, a0
	v_accvgpr_mov_b32 a87, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[116:119], v[42:45], 0, v156, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:119], v[42:45], a[96:99], v156, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a86, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[112:115], v[46:49], 0, v156, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[46:49], a[92:95], v156, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a85, a1
	v_accvgpr_mov_b32 a84, a0
	v_accvgpr_mov_b32 a67, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[120:123], v[42:45], 0, v156, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[120:123], v[42:45], a[92:95], v156, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a66, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[124:127], v[46:49], 0, v157, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[46:49], a[84:87], v157, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a65, a1
	v_accvgpr_mov_b32 a64, a0
	v_accvgpr_mov_b32 a55, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[132:135], v[42:45], 0, v157, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[42:45], a[84:87], v157, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a54, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[128:131], v[46:49], 0, v157, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[128:131], v[46:49], a[64:67], v157, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a53, a1
	v_accvgpr_mov_b32 a52, a0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[136:139], v[42:45], 0, v157, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[42:45], a[64:67], v157, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[140:143], v[46:49], 0, v158, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[140:143], v[46:49], a[52:55], v158, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[148:151], v[42:45], 0, v158, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[148:151], v[42:45], a[52:55], v158, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[144:147], v[46:49], 0, v158, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	v_accvgpr_mov_b32 a47, a3
	v_accvgpr_mov_b32 a46, a2
	v_accvgpr_mov_b32 a45, a1
	v_accvgpr_mov_b32 a44, a0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[144:147], v[46:49], a[44:47], v158, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[152:155], v[42:45], 0, v158, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[152:155], v[42:45], a[44:47], v158, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v159, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[42:45], v159, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	v_accvgpr_mov_b32 a127, a3
	v_accvgpr_mov_b32 a126, a2
	v_accvgpr_mov_b32 a125, a1
	v_accvgpr_mov_b32 a124, a0
	v_accvgpr_mov_b32 a123, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[68:71], v[62:65], 0, v107, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[62:65], a[124:127], v107, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a122, a2
	v_accvgpr_mov_b32 a121, a1
	v_accvgpr_mov_b32 a120, a0
	v_accvgpr_mov_b32 a119, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[76:79], v[58:61], 0, v107, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[76:79], v[58:61], a[124:127], v107, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a118, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[72:75], v[62:65], 0, v107, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:75], v[62:65], a[120:123], v107, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a117, a1
	v_accvgpr_mov_b32 a116, a0
	v_accvgpr_mov_b32 a115, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[80:83], v[58:61], 0, v107, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[58:61], a[120:123], v107, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a114, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[108:111], v[62:65], 0, v156, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:111], v[62:65], a[116:119], v156, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a113, a1
	v_accvgpr_mov_b32 a112, a0
	v_accvgpr_mov_b32 a107, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[116:119], v[58:61], 0, v156, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:119], v[58:61], a[116:119], v156, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a106, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[112:115], v[62:65], 0, v156, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[62:65], a[112:115], v156, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a105, a1
	v_accvgpr_mov_b32 a104, a0
	v_accvgpr_mov_b32 a91, a3
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[120:123], v[58:61], 0, v156, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[58:61], a[112:115], v156, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a90, a2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[124:127], v[62:65], 0, v157, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[124:127], v[62:65], a[104:107], v157, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_accvgpr_mov_b32 a89, a1
	v_accvgpr_mov_b32 a88, a0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[58:61], 0, v157, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:135], v[58:61], a[104:107], v157, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[128:131], v[62:65], 0, v157, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[128:131], v[62:65], a[88:91], v157, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[136:139], v[58:61], 0, v157, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[58:61], a[88:91], v157, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[62:65], 0, v158, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	v_accvgpr_mov_b32 a79, a3
	v_accvgpr_mov_b32 a78, a2
	v_accvgpr_mov_b32 a77, a1
	v_accvgpr_mov_b32 a76, a0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[62:65], a[76:79], v158, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[148:151], v[58:61], 0, v158, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[58:61], a[76:79], v158, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[144:147], v[62:65], 0, v158, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:147], v[62:65], a[0:3], v158, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], v[152:155], v[58:61], 0, v158, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[58:61], a[0:3], v158, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v159, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[58:61], v159, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v176, v91, s[20:23], s8 offen offset:512
	buffer_load_dword v177, v91, s[20:23], s9 offen offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[160:163], v94, s[24:27], 0 offen offset:384
	buffer_load_dwordx4 v[164:167], v95, s[24:27], 0 offen offset:384
	buffer_load_dwordx4 v[168:171], v96, s[24:27], 0 offen offset:384
	buffer_load_dwordx4 v[172:175], v97, s[24:27], 0 offen offset:384
	v_bitop3_b32 v84, v0, 15, 48 bitop3:0xe0
	v_mov_b32_e32 v85, 0x18100
	v_or_b32_e32 v107, 0x19d00, v91
	v_or_b32_e32 v156, 0x1b900, v91
	v_lshl_or_b32 v84, v84, 2, v85
	v_or_b32_e32 v157, 0x1d500, v91
	ds_read_b32 v84, v84
	ds_read_b32 v85, v107
	ds_read_b32 v107, v156
	ds_read_b32 v156, v157
	ds_read_b128 v[68:71], v93 offset:32768
	ds_read_b128 v[72:75], v93 offset:36864
	ds_read_b128 v[76:79], v92 offset:32768
	ds_read_b128 v[80:83], v92 offset:36864
	ds_read_b128 v[108:111], v93 offset:40960
	ds_read_b128 v[112:115], v93 offset:45056
	ds_read_b128 v[116:119], v92 offset:40960
	ds_read_b128 v[120:123], v92 offset:45056
	ds_read_b128 v[124:127], v93 offset:49152
	ds_read_b128 v[128:131], v93 offset:53248
	ds_read_b128 v[132:135], v92 offset:49152
	ds_read_b128 v[136:139], v92 offset:53248
	ds_read_b128 v[140:143], v93 offset:57344
	ds_read_b128 v[144:147], v93 offset:61440
	ds_read_b128 v[148:151], v92 offset:57344
	ds_read_b128 v[152:155], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[160:163]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[164:167]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[168:171]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[172:175]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[54:57], a[32:35], v84, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[50:53], a[32:35], v84, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[54:57], a[28:31], v84, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[50:53], a[28:31], v84, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[108:111], v[54:57], a[24:27], v85, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[116:119], v[50:53], a[24:27], v85, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[112:115], v[54:57], a[20:23], v85, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[120:123], v[50:53], a[20:23], v85, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[124:127], v[54:57], a[16:19], v107, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[50:53], a[16:19], v107, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[128:131], v[54:57], a[12:15], v107, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[50:53], a[12:15], v107, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[140:143], v[54:57], a[8:11], v156, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[50:53], a[8:11], v156, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[144:147], v[54:57], a[4:7], v156, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[50:53], a[4:7], v156, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v159, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[50:53], v159, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[22:25], a[80:83], v84, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[18:21], a[80:83], v84, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[22:25], a[72:75], v84, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[18:21], a[72:75], v84, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[108:111], v[22:25], a[68:71], v85, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[116:119], v[18:21], a[68:71], v85, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[112:115], v[22:25], a[60:63], v85, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[120:123], v[18:21], a[60:63], v85, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[124:127], v[22:25], a[56:59], v107, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[18:21], a[56:59], v107, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[128:131], v[22:25], a[48:51], v107, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[18:21], a[48:51], v107, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[140:143], v[22:25], a[40:43], v156, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[18:21], a[40:43], v156, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[144:147], v[22:25], a[36:39], v156, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:155], v[18:21], a[36:39], v156, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[22:25], v159, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v159, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[14:17], a[108:111], v84, v66 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[76:79], v[10:13], a[108:111], v84, v66 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[14:17], a[100:103], v84, v66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[10:13], a[100:103], v84, v66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[108:111], v[14:17], a[96:99], v85, v66 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[116:119], v[10:13], a[96:99], v85, v66 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[112:115], v[14:17], a[92:95], v85, v66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[120:123], v[10:13], a[92:95], v85, v66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[124:127], v[14:17], a[84:87], v107, v66 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[10:13], a[84:87], v107, v66 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[128:131], v[14:17], a[64:67], v107, v66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[10:13], a[64:67], v107, v66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[140:143], v[14:17], a[52:55], v156, v66 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[148:151], v[10:13], a[52:55], v156, v66 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[144:147], v[14:17], a[44:47], v156, v66 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[152:155], v[10:13], a[44:47], v156, v66 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[14:17], v159, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v159, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[6:9], a[124:127], v84, v66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[76:79], v[2:5], a[124:127], v84, v66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:75], v[6:9], a[120:123], v84, v66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[2:5], a[120:123], v84, v66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[108:111], v[6:9], a[116:119], v85, v66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[116:119], v[2:5], a[116:119], v85, v66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[112:115], v[6:9], a[112:115], v85, v66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[120:123], v[2:5], a[112:115], v85, v66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[124:127], v[6:9], a[104:107], v107, v66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:135], v[2:5], a[104:107], v107, v66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[128:131], v[6:9], a[88:91], v107, v66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[2:5], a[88:91], v107, v66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[140:143], v[6:9], a[76:79], v156, v66 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[2:5], a[76:79], v156, v66 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[144:147], v[6:9], a[0:3], v156, v66 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[2:5], a[0:3], v156, v66 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[6:9], v159, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v159, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v91, s[20:23], s8 offen offset:768
	buffer_load_dword v180, v91, s[20:23], s9 offen offset:768
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[78:81], v94, s[24:27], 0 offen offset:512
	buffer_load_dwordx4 v[184:187], v95, s[24:27], 0 offen offset:512
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:512
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:512
	v_or_b32_e32 v107, 0x10000, v93
	v_or_b32_e32 v108, 0x10000, v92
	v_or_b32_e32 v109, 0x11000, v93
	v_or_b32_e32 v110, 0x11000, v92
	v_or_b32_e32 v111, 0x12000, v93
	v_or_b32_e32 v112, 0x12000, v92
	v_or_b32_e32 v113, 0x13000, v93
	v_or_b32_e32 v114, 0x13000, v92
	v_or_b32_e32 v115, 0x14000, v93
	v_or_b32_e32 v116, 0x14000, v92
	v_or_b32_e32 v117, 0x15000, v93
	v_or_b32_e32 v118, 0x15000, v92
	v_or_b32_e32 v119, 0x16000, v93
	v_or_b32_e32 v120, 0x16000, v92
	v_or_b32_e32 v121, 0x17000, v93
	v_or_b32_e32 v122, 0x17000, v92
	v_or_b32_e32 v66, 0x18200, v91
	v_or_b32_e32 v67, 0x19e00, v91
	v_or_b32_e32 v68, 0x1ba00, v91
	v_or_b32_e32 v69, 0x1d600, v91
	ds_read_b128 v[70:73], v107
	ds_read_b128 v[74:77], v108
	ds_read_b128 v[82:85], v109
	ds_read_b128 v[124:127], v110
	ds_read_b128 v[128:131], v111
	ds_read_b128 v[132:135], v112
	ds_read_b128 v[136:139], v113
	ds_read_b128 v[140:143], v114
	ds_read_b128 v[144:147], v115
	ds_read_b128 v[148:151], v116
	ds_read_b128 v[152:155], v117
	ds_read_b128 v[156:159], v118
	ds_read_b128 v[160:163], v119
	ds_read_b128 v[164:167], v120
	ds_read_b128 v[168:171], v121
	ds_read_b128 v[172:175], v122
	ds_read_b32 v178, v66
	ds_read_b32 v179, v67
	ds_read_b32 v181, v68
	ds_read_b32 v182, v69
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[78:81] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[184:187] offset:32768
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[188:191] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[192:195] offset:32768
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[30:33], a[32:35], v178, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[26:29], a[32:35], v178, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:85], v[30:33], a[28:31], v178, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[26:29], a[28:31], v178, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[30:33], a[24:27], v179, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[26:29], a[24:27], v179, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[30:33], a[20:23], v179, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[26:29], a[20:23], v179, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[30:33], a[16:19], v181, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[26:29], a[16:19], v181, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[30:33], a[12:15], v181, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[26:29], a[12:15], v181, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[30:33], a[8:11], v182, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[26:29], a[8:11], v182, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[30:33], a[4:7], v182, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[172:175], v[26:29], a[4:7], v182, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v183, 0x2000, v102
	buffer_load_dwordx4 v[78:81], v183, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[26:29], v183, s[12:15], s11 offen offset:1024 nt
	s_movk_i32 s4, 0x2000
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[38:41], a[80:83], v178, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[34:37], a[80:83], v178, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[38:41], a[72:75], v178, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[34:37], a[72:75], v178, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[38:41], a[68:71], v179, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[34:37], a[68:71], v179, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[38:41], a[60:63], v179, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[34:37], a[60:63], v179, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[38:41], a[56:59], v181, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[34:37], a[56:59], v181, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[38:41], a[48:51], v181, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[34:37], a[48:51], v181, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[38:41], a[40:43], v182, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[34:37], a[40:43], v182, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[38:41], a[36:39], v182, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[34:37], a[36:39], v182, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v183, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[30:33], v183, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[70:73], v[46:49], a[108:111], v178, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:77], v[42:45], a[108:111], v178, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[46:49], a[100:103], v178, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:127], v[42:45], a[100:103], v178, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[46:49], a[96:99], v179, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[42:45], a[96:99], v179, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[46:49], a[92:95], v179, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[42:45], a[92:95], v179, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[46:49], a[84:87], v181, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[42:45], a[84:87], v181, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[46:49], a[64:67], v181, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:159], v[42:45], a[64:67], v181, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[46:49], a[52:55], v182, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[42:45], a[52:55], v182, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[46:49], a[44:47], v182, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[42:45], a[44:47], v182, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v183, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[34:37], v183, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[70:73], v[62:65], a[124:127], v178, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[58:61], a[124:127], v178, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:85], v[62:65], a[120:123], v178, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[58:61], a[120:123], v178, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[62:65], a[116:119], v179, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[58:61], a[116:119], v179, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[62:65], a[112:115], v179, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[58:61], a[112:115], v179, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[62:65], a[104:107], v181, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[58:61], a[104:107], v181, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[62:65], a[88:91], v181, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[58:61], a[88:91], v181, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[62:65], a[76:79], v182, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[58:61], a[76:79], v182, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[62:65], a[0:3], v182, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[58:61], a[0:3], v182, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v183, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[38:41], v183, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v181, v91, s[20:23], s8 offen offset:1024
	buffer_load_dword v182, v91, s[20:23], s9 offen offset:1024
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[62:65], v94, s[24:27], 0 offen offset:640
	buffer_load_dwordx4 v[70:73], v95, s[24:27], 0 offen offset:640
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:640
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:640
	v_or_b32_e32 v42, 0x18300, v91
	v_or_b32_e32 v43, 0x19f00, v90
	v_or_b32_e32 v44, 0x1bb00, v91
	v_or_b32_e32 v45, 0x1d700, v91
	ds_read_b32 v184, v42
	ds_read_b32 v185, v43
	ds_read_b32 v186, v44
	ds_read_b32 v187, v45
	ds_read_b128 v[74:77], v93
	ds_read_b128 v[82:85], v93 offset:4096
	ds_read_b128 v[124:127], v92
	ds_read_b128 v[128:131], v92 offset:4096
	ds_read_b128 v[132:135], v93 offset:8192
	ds_read_b128 v[136:139], v93 offset:12288
	ds_read_b128 v[140:143], v92 offset:8192
	ds_read_b128 v[144:147], v92 offset:12288
	ds_read_b128 v[148:151], v93 offset:16384
	ds_read_b128 v[152:155], v93 offset:20480
	ds_read_b128 v[156:159], v92 offset:16384
	ds_read_b128 v[160:163], v92 offset:20480
	ds_read_b128 v[164:167], v93 offset:24576
	ds_read_b128 v[168:171], v93 offset:28672
	ds_read_b128 v[172:175], v92 offset:24576
	ds_read_b128 v[176:179], v92 offset:28672
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[62:65]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[70:73]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[188:191]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[192:195]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[54:57], a[32:35], v184, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[124:127], v[50:53], a[32:35], v184, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:85], v[54:57], a[28:31], v184, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[50:53], a[28:31], v184, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[54:57], a[24:27], v185, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[50:53], a[24:27], v185, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[54:57], a[20:23], v185, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[50:53], a[20:23], v185, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[54:57], a[16:19], v186, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[156:159], v[50:53], a[16:19], v186, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[54:57], a[12:15], v186, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[160:163], v[50:53], a[12:15], v186, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[54:57], a[8:11], v187, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[172:175], v[50:53], a[8:11], v187, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[54:57], a[4:7], v187, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[176:179], v[50:53], a[4:7], v187, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v183, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[42:45], v183, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[22:25], a[80:83], v184, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[18:21], a[80:83], v184, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[22:25], a[72:75], v184, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[128:131], v[18:21], a[72:75], v184, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[22:25], a[68:71], v185, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:143], v[18:21], a[68:71], v185, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[22:25], a[60:63], v185, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[18:21], a[60:63], v185, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[22:25], a[56:59], v186, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[18:21], a[56:59], v186, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[22:25], a[48:51], v186, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:163], v[18:21], a[48:51], v186, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[22:25], a[40:43], v187, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[18:21], a[40:43], v187, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[22:25], a[36:39], v187, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[176:179], v[18:21], a[36:39], v187, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v183, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v183, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:77], v[14:17], a[108:111], v184, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:127], v[10:13], a[108:111], v184, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[14:17], a[100:103], v184, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:131], v[10:13], a[100:103], v184, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[14:17], a[96:99], v185, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[140:143], v[10:13], a[96:99], v185, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[14:17], a[92:95], v185, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[10:13], a[92:95], v185, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[14:17], a[84:87], v186, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:159], v[10:13], a[84:87], v186, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[14:17], a[64:67], v186, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[10:13], a[64:67], v186, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[14:17], a[52:55], v187, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[172:175], v[10:13], a[52:55], v187, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[14:17], a[44:47], v187, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[176:179], v[10:13], a[44:47], v187, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v183, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[22:25], v183, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[6:9], a[124:127], v184, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[2:5], a[124:127], v184, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:85], v[6:9], a[120:123], v184, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[128:131], v[2:5], a[120:123], v184, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[6:9], a[116:119], v185, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[2:5], a[116:119], v185, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[6:9], a[112:115], v185, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[2:5], a[112:115], v185, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[6:9], a[104:107], v186, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[156:159], v[2:5], a[104:107], v186, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[6:9], a[88:91], v186, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[160:163], v[2:5], a[88:91], v186, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[6:9], a[76:79], v187, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[172:175], v[2:5], a[76:79], v187, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[6:9], a[0:3], v187, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[2:5], a[0:3], v187, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[74:77], v183, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[50:53], v183, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v91, s[20:23], s8 offen offset:1280
	buffer_load_dword v180, v91, s[20:23], s9 offen offset:1280
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[6:9], v94, s[24:27], 0 offen offset:768
	buffer_load_dwordx4 v[82:85], v95, s[24:27], 0 offen offset:768
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:768
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:768
	v_or_b32_e32 v2, 0x18400, v91
	v_or_b32_e32 v3, 0x1a000, v91
	v_or_b32_e32 v4, 0x1bc00, v91
	v_or_b32_e32 v5, 0x1d800, v91
	ds_read_b32 v183, v2
	ds_read_b32 v184, v3
	ds_read_b32 v185, v4
	ds_read_b32 v186, v5
	ds_read_b128 v[10:13], v93 offset:32768
	ds_read_b128 v[14:17], v93 offset:36864
	ds_read_b128 v[124:127], v92 offset:32768
	ds_read_b128 v[128:131], v92 offset:36864
	ds_read_b128 v[132:135], v93 offset:40960
	ds_read_b128 v[136:139], v93 offset:45056
	ds_read_b128 v[140:143], v92 offset:40960
	ds_read_b128 v[144:147], v92 offset:45056
	ds_read_b128 v[148:151], v93 offset:49152
	ds_read_b128 v[152:155], v93 offset:53248
	ds_read_b128 v[156:159], v92 offset:49152
	ds_read_b128 v[160:163], v92 offset:53248
	ds_read_b128 v[164:167], v93 offset:57344
	ds_read_b128 v[168:171], v93 offset:61440
	ds_read_b128 v[172:175], v92 offset:57344
	ds_read_b128 v[176:179], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[6:9]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[82:85]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[188:191]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[192:195]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[10:13], v[78:81], a[32:35], v183, v181 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[124:127], v[26:29], a[32:35], v183, v181 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[14:17], v[78:81], a[28:31], v183, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[26:29], a[28:31], v183, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[78:81], a[24:27], v184, v181 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[26:29], a[24:27], v184, v181 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[78:81], a[20:23], v184, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[26:29], a[20:23], v184, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[78:81], a[16:19], v185, v181 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[156:159], v[26:29], a[16:19], v185, v181 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[78:81], a[12:15], v185, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[160:163], v[26:29], a[12:15], v185, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[78:81], a[8:11], v186, v181 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[172:175], v[26:29], a[8:11], v186, v181 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[78:81], a[4:7], v186, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[176:179], v[26:29], a[4:7], v186, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v187, 0x3000, v102
	buffer_load_dwordx4 v[82:85], v187, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[78:81], v187, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[10:13], v[66:69], a[80:83], v183, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[30:33], a[80:83], v183, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[14:17], v[66:69], a[72:75], v183, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[128:131], v[30:33], a[72:75], v183, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[66:69], a[68:71], v184, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:143], v[30:33], a[68:71], v184, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[66:69], a[60:63], v184, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[30:33], a[60:63], v184, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[66:69], a[56:59], v185, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[30:33], a[56:59], v185, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[66:69], a[48:51], v185, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:163], v[30:33], a[48:51], v185, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[66:69], a[40:43], v186, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[30:33], a[40:43], v186, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[66:69], a[36:39], v186, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[176:179], v[30:33], a[36:39], v186, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v187, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[2:5], v187, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[10:13], v[46:49], a[108:111], v183, v182 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:127], v[34:37], a[108:111], v183, v182 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[14:17], v[46:49], a[100:103], v183, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:131], v[34:37], a[100:103], v183, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[46:49], a[96:99], v184, v182 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[140:143], v[34:37], a[96:99], v184, v182 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[46:49], a[92:95], v184, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[34:37], a[92:95], v184, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[46:49], a[84:87], v185, v182 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:159], v[34:37], a[84:87], v185, v182 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[46:49], a[64:67], v185, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[34:37], a[64:67], v185, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[46:49], a[52:55], v186, v182 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[172:175], v[34:37], a[52:55], v186, v182 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[46:49], a[44:47], v186, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[176:179], v[34:37], a[44:47], v186, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v187, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[6:9], v187, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[10:13], v[58:61], a[124:127], v183, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[38:41], a[124:127], v183, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[14:17], v[58:61], a[120:123], v183, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[128:131], v[38:41], a[120:123], v183, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[58:61], a[116:119], v184, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[38:41], a[116:119], v184, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[58:61], a[112:115], v184, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[38:41], a[112:115], v184, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[58:61], a[104:107], v185, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[156:159], v[38:41], a[104:107], v185, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[58:61], a[88:91], v185, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[160:163], v[38:41], a[88:91], v185, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[58:61], a[76:79], v186, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[172:175], v[38:41], a[76:79], v186, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[58:61], a[0:3], v186, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[38:41], a[0:3], v186, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v187, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[10:13], v187, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v176, v91, s[20:23], s8 offen offset:1536
	buffer_load_dword v177, v91, s[20:23], s9 offen offset:1536
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[46:49], v94, s[24:27], 0 offen offset:896
	buffer_load_dwordx4 v[188:191], v95, s[24:27], 0 offen offset:896
	buffer_load_dwordx4 v[192:195], v96, s[24:27], 0 offen offset:896
	buffer_load_dwordx4 v[196:199], v97, s[24:27], 0 offen offset:896
	v_or_b32_e32 v14, 0x18500, v91
	v_or_b32_e32 v15, 0x1a100, v91
	v_or_b32_e32 v16, 0x1bd00, v91
	v_or_b32_e32 v17, 0x1d900, v91
	ds_read_b32 v178, v14
	ds_read_b32 v179, v15
	ds_read_b32 v181, v16
	ds_read_b32 v182, v17
	ds_read_b128 v[26:29], v107
	ds_read_b128 v[58:61], v108
	ds_read_b128 v[66:69], v109
	ds_read_b128 v[124:127], v110
	ds_read_b128 v[128:131], v111
	ds_read_b128 v[132:135], v112
	ds_read_b128 v[136:139], v113
	ds_read_b128 v[140:143], v114
	ds_read_b128 v[144:147], v115
	ds_read_b128 v[148:151], v116
	ds_read_b128 v[152:155], v117
	ds_read_b128 v[156:159], v118
	ds_read_b128 v[160:163], v119
	ds_read_b128 v[164:167], v120
	ds_read_b128 v[168:171], v121
	ds_read_b128 v[172:175], v122
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[46:49] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[188:191] offset:32768
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[192:195] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[196:199] offset:32768
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:29], v[54:57], a[32:35], v178, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[58:61], v[42:45], a[32:35], v178, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[66:69], v[54:57], a[28:31], v178, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[42:45], a[28:31], v178, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[54:57], a[24:27], v179, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[42:45], a[24:27], v179, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[54:57], a[20:23], v179, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[42:45], a[20:23], v179, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[54:57], a[16:19], v181, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(10)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[42:45], a[16:19], v181, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[54:57], a[12:15], v181, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[42:45], a[12:15], v181, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[54:57], a[8:11], v182, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[42:45], a[8:11], v182, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[54:57], a[4:7], v182, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[172:175], v[42:45], a[4:7], v182, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v187, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v187, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[26:29], v[62:65], a[80:83], v178, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[58:61], v[18:21], a[80:83], v178, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[66:69], v[62:65], a[72:75], v178, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[18:21], a[72:75], v178, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[62:65], a[68:71], v179, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[18:21], a[68:71], v179, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[62:65], a[60:63], v179, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[18:21], a[60:63], v179, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[62:65], a[56:59], v181, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[18:21], a[56:59], v181, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[62:65], a[48:51], v181, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[18:21], a[48:51], v181, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[62:65], a[40:43], v182, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[18:21], a[40:43], v182, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[62:65], a[36:39], v182, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[18:21], a[36:39], v182, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v187, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v187, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[70:73], a[108:111], v178, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[58:61], v[22:25], a[108:111], v178, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[66:69], v[70:73], a[100:103], v178, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:127], v[22:25], a[100:103], v178, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[70:73], a[96:99], v179, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[22:25], a[96:99], v179, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[70:73], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[22:25], a[92:95], v179, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[70:73], a[84:87], v181, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[22:25], a[84:87], v181, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[70:73], a[64:67], v181, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:159], v[22:25], a[64:67], v181, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[70:73], a[52:55], v182, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[22:25], a[52:55], v182, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[70:73], a[44:47], v182, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[22:25], a[44:47], v182, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v187, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[22:25], v187, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[26:29], v[74:77], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[58:61], v[50:53], a[124:127], v178, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[66:69], v[74:77], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[50:53], a[120:123], v178, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[74:77], a[116:119], v179, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[50:53], a[116:119], v179, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[74:77], a[112:115], v179, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[50:53], a[112:115], v179, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[74:77], a[104:107], v181, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[50:53], a[104:107], v181, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[74:77], a[88:91], v181, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[50:53], a[88:91], v181, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[74:77], a[76:79], v182, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[50:53], a[76:79], v182, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[74:77], a[0:3], v182, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[50:53], a[0:3], v182, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v187, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[26:29], v187, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v91, s[20:23], s8 offen offset:1792
	buffer_load_dword v178, v91, s[20:23], s9 offen offset:1792
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[62:65], v94, s[24:27], 0 offen offset:1024
	buffer_load_dwordx4 v[184:187], v95, s[24:27], 0 offen offset:1024
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:1024
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:1024
	v_or_b32_e32 v58, 0x18600, v91
	v_or_b32_e32 v59, 0x1a200, v91
	v_or_b32_e32 v60, 0x1be00, v91
	v_or_b32_e32 v61, 0x1da00, v91
	ds_read_b32 v179, v58
	ds_read_b32 v180, v59
	ds_read_b32 v181, v60
	ds_read_b32 v182, v61
	ds_read_b128 v[66:69], v93
	ds_read_b128 v[70:73], v93 offset:4096
	ds_read_b128 v[74:77], v92
	ds_read_b128 v[124:127], v92 offset:4096
	ds_read_b128 v[128:131], v93 offset:8192
	ds_read_b128 v[132:135], v93 offset:12288
	ds_read_b128 v[136:139], v92 offset:8192
	ds_read_b128 v[140:143], v92 offset:12288
	ds_read_b128 v[144:147], v93 offset:16384
	ds_read_b128 v[148:151], v93 offset:20480
	ds_read_b128 v[152:155], v92 offset:16384
	ds_read_b128 v[156:159], v92 offset:20480
	ds_read_b128 v[160:163], v93 offset:24576
	ds_read_b128 v[164:167], v93 offset:28672
	ds_read_b128 v[168:171], v92 offset:24576
	ds_read_b128 v[172:175], v92 offset:28672
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[62:65]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[184:187]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[188:191]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[192:195]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[82:85], a[32:35], v179, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[78:81], a[32:35], v179, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[82:85], a[28:31], v179, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[78:81], a[28:31], v179, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[82:85], a[24:27], v180, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[78:81], a[24:27], v180, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[82:85], a[20:23], v180, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[78:81], a[20:23], v180, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[82:85], a[16:19], v181, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[78:81], a[16:19], v181, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[82:85], a[12:15], v181, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[78:81], a[12:15], v181, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[82:85], a[8:11], v182, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[168:171], v[78:81], a[8:11], v182, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[82:85], a[4:7], v182, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[172:175], v[78:81], a[4:7], v182, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v183, 0x4000, v102
	buffer_load_dwordx4 v[62:65], v183, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[58:61], v183, s[12:15], s11 offen offset:1024 nt
	s_movk_i32 s5, 0x4000
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[30:33], a[80:83], v179, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[2:5], a[80:83], v179, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[70:73], v[30:33], a[72:75], v179, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[2:5], a[72:75], v179, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[30:33], a[68:71], v180, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[2:5], a[68:71], v180, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[30:33], a[60:63], v180, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[2:5], a[60:63], v180, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[30:33], a[56:59], v181, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[152:155], v[2:5], a[56:59], v181, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[30:33], a[48:51], v181, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[2:5], a[48:51], v181, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[30:33], a[40:43], v182, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[168:171], v[2:5], a[40:43], v182, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[30:33], a[36:39], v182, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[2:5], a[36:39], v182, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v183, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[2:5], v183, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:69], v[34:37], a[108:111], v179, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:77], v[6:9], a[108:111], v179, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[34:37], a[100:103], v179, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:127], v[6:9], a[100:103], v179, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[34:37], a[96:99], v180, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[6:9], a[96:99], v180, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[34:37], a[92:95], v180, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[6:9], a[92:95], v180, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[34:37], a[84:87], v181, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[6:9], a[84:87], v181, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[34:37], a[64:67], v181, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:159], v[6:9], a[64:67], v181, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[34:37], a[52:55], v182, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:171], v[6:9], a[52:55], v182, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[34:37], a[44:47], v182, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[6:9], a[44:47], v182, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v183, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[6:9], v183, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:69], v[38:41], a[124:127], v179, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[10:13], a[124:127], v179, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:73], v[38:41], a[120:123], v179, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[10:13], a[120:123], v179, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[38:41], a[116:119], v180, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[10:13], a[116:119], v180, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[38:41], a[112:115], v180, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[10:13], a[112:115], v180, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[38:41], a[104:107], v181, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:155], v[10:13], a[104:107], v181, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[38:41], a[88:91], v181, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[10:13], a[88:91], v181, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[38:41], a[76:79], v182, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[168:171], v[10:13], a[76:79], v182, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[38:41], a[0:3], v182, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[10:13], a[0:3], v182, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v183, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[10:13], v183, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v168, v91, s[20:23], s8 offen offset:2048
	buffer_load_dword v169, v91, s[20:23], s9 offen offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[174:177], v94, s[24:27], 0 offen offset:1152
	buffer_load_dwordx4 v[184:187], v95, s[24:27], 0 offen offset:1152
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:1152
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:1152
	v_or_b32_e32 v170, 0x18700, v91
	v_or_b32_e32 v171, 0x1a300, v91
	v_or_b32_e32 v172, 0x1bf00, v90
	v_or_b32_e32 v173, 0x1db00, v91
	ds_read_b32 v170, v170
	ds_read_b32 v171, v171
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b128 v[66:69], v93 offset:32768
	ds_read_b128 v[70:73], v93 offset:36864
	ds_read_b128 v[74:77], v92 offset:32768
	ds_read_b128 v[78:81], v92 offset:36864
	ds_read_b128 v[82:85], v93 offset:40960
	ds_read_b128 v[124:127], v93 offset:45056
	ds_read_b128 v[128:131], v92 offset:40960
	ds_read_b128 v[132:135], v92 offset:45056
	ds_read_b128 v[136:139], v93 offset:49152
	ds_read_b128 v[140:143], v93 offset:53248
	ds_read_b128 v[144:147], v92 offset:49152
	ds_read_b128 v[148:151], v92 offset:53248
	ds_read_b128 v[152:155], v93 offset:57344
	ds_read_b128 v[156:159], v93 offset:61440
	ds_read_b128 v[160:163], v92 offset:57344
	ds_read_b128 v[164:167], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[174:177]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[184:187]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[188:191]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[192:195]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[54:57], a[32:35], v170, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[14:17], a[32:35], v170, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v170, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[14:17], a[28:31], v170, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:85], v[54:57], a[24:27], v171, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[14:17], a[24:27], v171, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[54:57], a[20:23], v171, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[14:17], a[20:23], v171, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[54:57], a[16:19], v172, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[14:17], a[16:19], v172, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[54:57], a[12:15], v172, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[14:17], a[12:15], v172, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[54:57], a[8:11], v173, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[14:17], a[8:11], v173, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[156:159], v[54:57], a[4:7], v173, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[14:17], a[4:7], v173, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v183, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v183, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[42:45], a[80:83], v170, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[18:21], a[80:83], v170, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[70:73], v[42:45], a[72:75], v170, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[18:21], a[72:75], v170, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[42:45], a[68:71], v171, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[18:21], a[68:71], v171, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[42:45], a[60:63], v171, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[18:21], a[60:63], v171, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[42:45], a[56:59], v172, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[18:21], a[56:59], v172, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[140:143], v[42:45], a[48:51], v172, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[18:21], a[48:51], v172, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[152:155], v[42:45], a[40:43], v173, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[18:21], a[40:43], v173, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:159], v[42:45], a[36:39], v173, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[18:21], a[36:39], v173, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v183, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v183, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:69], v[46:49], a[108:111], v170, v178 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:77], v[22:25], a[108:111], v170, v178 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[46:49], a[100:103], v170, v178 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[22:25], a[100:103], v170, v178 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[46:49], a[96:99], v171, v178 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[22:25], a[96:99], v171, v178 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[124:127], v[46:49], a[92:95], v171, v178 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[22:25], a[92:95], v171, v178 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[46:49], a[84:87], v172, v178 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[22:25], a[84:87], v172, v178 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:143], v[46:49], a[64:67], v172, v178 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[22:25], a[64:67], v172, v178 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:155], v[46:49], a[52:55], v173, v178 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[22:25], a[52:55], v173, v178 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[46:49], a[44:47], v173, v178 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[22:25], a[44:47], v173, v178 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v183, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[22:25], v183, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:69], v[50:53], a[124:127], v170, v178 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[26:29], a[124:127], v170, v178 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:73], v[50:53], a[120:123], v170, v178 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[26:29], a[120:123], v170, v178 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:85], v[50:53], a[116:119], v171, v178 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[26:29], a[116:119], v171, v178 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[50:53], a[112:115], v171, v178 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[26:29], a[112:115], v171, v178 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[50:53], a[104:107], v172, v178 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[26:29], a[104:107], v172, v178 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[50:53], a[88:91], v172, v178 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[26:29], a[88:91], v172, v178 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[152:155], v[50:53], a[76:79], v173, v178 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[26:29], a[76:79], v173, v178 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[156:159], v[50:53], a[0:3], v173, v178 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[26:29], a[0:3], v173, v178 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v183, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[26:29], v183, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v91, s[20:23], s8 offen offset:2304
	buffer_load_dword v170, v91, s[20:23], s9 offen offset:2304
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[176:179], v94, s[24:27], 0 offen offset:1280
	buffer_load_dwordx4 v[180:183], v95, s[24:27], 0 offen offset:1280
	buffer_load_dwordx4 v[184:187], v96, s[24:27], 0 offen offset:1280
	buffer_load_dwordx4 v[188:191], v97, s[24:27], 0 offen offset:1280
	v_or_b32_e32 v171, 0x18800, v91
	v_or_b32_e32 v172, 0x1a400, v91
	v_or_b32_e32 v173, 0x1c000, v91
	v_or_b32_e32 v174, 0x1dc00, v91
	ds_read_b32 v171, v171
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b128 v[66:69], v107
	ds_read_b128 v[70:73], v108
	ds_read_b128 v[74:77], v109
	ds_read_b128 v[78:81], v110
	ds_read_b128 v[82:85], v111
	ds_read_b128 v[124:127], v112
	ds_read_b128 v[128:131], v113
	ds_read_b128 v[132:135], v114
	ds_read_b128 v[136:139], v115
	ds_read_b128 v[140:143], v116
	ds_read_b128 v[144:147], v117
	ds_read_b128 v[148:151], v118
	ds_read_b128 v[152:155], v119
	ds_read_b128 v[156:159], v120
	ds_read_b128 v[160:163], v121
	ds_read_b128 v[164:167], v122
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[176:179] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[180:183] offset:32768
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[184:187] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[188:191] offset:32768
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[62:65], a[32:35], v171, v168 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[58:61], a[32:35], v171, v168 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[62:65], a[28:31], v171, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[58:61], a[28:31], v171, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:85], v[62:65], a[24:27], v172, v168 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v172, v168 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[62:65], a[20:23], v172, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[58:61], a[20:23], v172, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[62:65], a[16:19], v173, v168 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(10)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[58:61], a[16:19], v173, v168 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[62:65], a[12:15], v173, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[58:61], a[12:15], v173, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[62:65], a[8:11], v174, v168 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[58:61], a[8:11], v174, v168 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[62:65], a[4:7], v174, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[58:61], a[4:7], v174, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v175, 0x5000, v102
	buffer_load_dwordx4 v[62:65], v175, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[58:61], v175, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[30:33], a[80:83], v171, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[2:5], a[80:83], v171, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[30:33], a[72:75], v171, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[2:5], a[72:75], v171, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[30:33], a[68:71], v172, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[2:5], a[68:71], v172, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[30:33], a[60:63], v172, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[2:5], a[60:63], v172, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[30:33], a[56:59], v173, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[2:5], a[56:59], v173, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[30:33], a[48:51], v173, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[2:5], a[48:51], v173, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[152:155], v[30:33], a[40:43], v174, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[2:5], a[40:43], v174, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[30:33], a[36:39], v174, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[2:5], a[36:39], v174, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v175, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[2:5], v175, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:69], v[34:37], a[108:111], v171, v169 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[70:73], v[6:9], a[108:111], v171, v169 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:77], v[34:37], a[100:103], v171, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[6:9], a[100:103], v171, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[34:37], a[96:99], v172, v169 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[6:9], a[96:99], v172, v169 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[34:37], a[92:95], v172, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[6:9], a[92:95], v172, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[34:37], a[84:87], v173, v169 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[6:9], a[84:87], v173, v169 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[34:37], a[64:67], v173, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[6:9], a[64:67], v173, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:155], v[34:37], a[52:55], v174, v169 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[6:9], a[52:55], v174, v169 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[34:37], a[44:47], v174, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[6:9], a[44:47], v174, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v175, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[6:9], v175, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:69], v[38:41], a[124:127], v171, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[70:73], v[10:13], a[124:127], v171, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:77], v[38:41], a[120:123], v171, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[10:13], a[120:123], v171, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:85], v[38:41], a[116:119], v172, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[10:13], a[116:119], v172, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[38:41], a[112:115], v172, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[10:13], a[112:115], v172, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[38:41], a[104:107], v173, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[10:13], a[104:107], v173, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[38:41], a[88:91], v173, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[10:13], a[88:91], v173, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[152:155], v[38:41], a[76:79], v174, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[10:13], a[76:79], v174, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[38:41], a[0:3], v174, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[10:13], a[0:3], v174, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v175, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[10:13], v175, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v168, v91, s[20:23], s8 offen offset:2560
	buffer_load_dword v169, v91, s[20:23], s9 offen offset:2560
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[176:179], v94, s[24:27], 0 offen offset:1408
	buffer_load_dwordx4 v[180:183], v95, s[24:27], 0 offen offset:1408
	buffer_load_dwordx4 v[184:187], v96, s[24:27], 0 offen offset:1408
	buffer_load_dwordx4 v[188:191], v97, s[24:27], 0 offen offset:1408
	v_or_b32_e32 v171, 0x18900, v91
	v_or_b32_e32 v172, 0x1a500, v91
	v_or_b32_e32 v173, 0x1c100, v91
	v_or_b32_e32 v174, 0x1dd00, v91
	ds_read_b32 v171, v171
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b128 v[66:69], v93
	ds_read_b128 v[70:73], v93 offset:4096
	ds_read_b128 v[74:77], v92
	ds_read_b128 v[78:81], v92 offset:4096
	ds_read_b128 v[82:85], v93 offset:8192
	ds_read_b128 v[124:127], v93 offset:12288
	ds_read_b128 v[128:131], v92 offset:8192
	ds_read_b128 v[132:135], v92 offset:12288
	ds_read_b128 v[136:139], v93 offset:16384
	ds_read_b128 v[140:143], v93 offset:20480
	ds_read_b128 v[144:147], v92 offset:16384
	ds_read_b128 v[148:151], v92 offset:20480
	ds_read_b128 v[152:155], v93 offset:24576
	ds_read_b128 v[156:159], v93 offset:28672
	ds_read_b128 v[160:163], v92 offset:24576
	ds_read_b128 v[164:167], v92 offset:28672
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[176:179]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[180:183]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[184:187]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[188:191]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[54:57], a[32:35], v171, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[14:17], a[32:35], v171, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[54:57], a[28:31], v171, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[14:17], a[28:31], v171, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:85], v[54:57], a[24:27], v172, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[14:17], a[24:27], v172, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[54:57], a[20:23], v172, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[14:17], a[20:23], v172, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[54:57], a[16:19], v173, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[14:17], a[16:19], v173, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[54:57], a[12:15], v173, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[14:17], a[12:15], v173, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[54:57], a[8:11], v174, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[14:17], a[8:11], v174, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[156:159], v[54:57], a[4:7], v174, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[14:17], a[4:7], v174, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[54:57], v175, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v175, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[42:45], a[80:83], v171, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[18:21], a[80:83], v171, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[70:73], v[42:45], a[72:75], v171, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[18:21], a[72:75], v171, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[42:45], a[68:71], v172, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[18:21], a[68:71], v172, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[42:45], a[60:63], v172, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[18:21], a[60:63], v172, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[42:45], a[56:59], v173, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[18:21], a[56:59], v173, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[140:143], v[42:45], a[48:51], v173, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[18:21], a[48:51], v173, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[152:155], v[42:45], a[40:43], v174, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[18:21], a[40:43], v174, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:159], v[42:45], a[36:39], v174, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[18:21], a[36:39], v174, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v175, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v175, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:69], v[46:49], a[108:111], v171, v170 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:77], v[22:25], a[108:111], v171, v170 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[46:49], a[100:103], v171, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[22:25], a[100:103], v171, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[46:49], a[96:99], v172, v170 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[22:25], a[96:99], v172, v170 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[124:127], v[46:49], a[92:95], v172, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[22:25], a[92:95], v172, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[46:49], a[84:87], v173, v170 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[22:25], a[84:87], v173, v170 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:143], v[46:49], a[64:67], v173, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[22:25], a[64:67], v173, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:155], v[46:49], a[52:55], v174, v170 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[22:25], a[52:55], v174, v170 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[46:49], a[44:47], v174, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[22:25], a[44:47], v174, v170 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v175, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[22:25], v175, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:69], v[50:53], a[124:127], v171, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[26:29], a[124:127], v171, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:73], v[50:53], a[120:123], v171, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[26:29], a[120:123], v171, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:85], v[50:53], a[116:119], v172, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[26:29], a[116:119], v172, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[50:53], a[112:115], v172, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[26:29], a[112:115], v172, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[50:53], a[104:107], v173, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[26:29], a[104:107], v173, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[50:53], a[88:91], v173, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[26:29], a[88:91], v173, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[152:155], v[50:53], a[76:79], v174, v170 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[26:29], a[76:79], v174, v170 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[156:159], v[50:53], a[0:3], v174, v170 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[26:29], a[0:3], v174, v170 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v175, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[26:29], v175, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v91, s[20:23], s8 offen offset:2816
	buffer_load_dword v172, v91, s[20:23], s9 offen offset:2816
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[176:179], v94, s[24:27], 0 offen offset:1536
	buffer_load_dwordx4 v[180:183], v95, s[24:27], 0 offen offset:1536
	buffer_load_dwordx4 v[184:187], v96, s[24:27], 0 offen offset:1536
	buffer_load_dwordx4 v[188:191], v97, s[24:27], 0 offen offset:1536
	v_or_b32_e32 v170, 0x18a00, v91
	v_or_b32_e32 v171, 0x1a600, v91
	v_or_b32_e32 v173, 0x1c200, v91
	v_or_b32_e32 v174, 0x1de00, v91
	ds_read_b32 v170, v170
	ds_read_b32 v171, v171
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b128 v[66:69], v93 offset:32768
	ds_read_b128 v[70:73], v93 offset:36864
	ds_read_b128 v[74:77], v92 offset:32768
	ds_read_b128 v[78:81], v92 offset:36864
	ds_read_b128 v[82:85], v93 offset:40960
	ds_read_b128 v[124:127], v93 offset:45056
	ds_read_b128 v[128:131], v92 offset:40960
	ds_read_b128 v[132:135], v92 offset:45056
	ds_read_b128 v[136:139], v93 offset:49152
	ds_read_b128 v[140:143], v93 offset:53248
	ds_read_b128 v[144:147], v92 offset:49152
	ds_read_b128 v[148:151], v92 offset:53248
	ds_read_b128 v[152:155], v93 offset:57344
	ds_read_b128 v[156:159], v93 offset:61440
	ds_read_b128 v[160:163], v92 offset:57344
	ds_read_b128 v[164:167], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[176:179]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[180:183]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[184:187]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[188:191]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[62:65], a[32:35], v170, v168 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[58:61], a[32:35], v170, v168 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[70:73], v[62:65], a[28:31], v170, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[58:61], a[28:31], v170, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:85], v[62:65], a[24:27], v171, v168 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[58:61], a[24:27], v171, v168 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[124:127], v[62:65], a[20:23], v171, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[58:61], a[20:23], v171, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[136:139], v[62:65], a[16:19], v173, v168 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[58:61], a[16:19], v173, v168 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[140:143], v[62:65], a[12:15], v173, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[58:61], a[12:15], v173, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[152:155], v[62:65], a[8:11], v174, v168 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[58:61], a[8:11], v174, v168 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[156:159], v[62:65], a[4:7], v174, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[58:61], a[4:7], v174, v168 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v175, 0x6000, v102
	buffer_load_dwordx4 v[62:65], v175, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[58:61], v175, s[12:15], s11 offen offset:1024 nt
	s_movk_i32 s6, 0x6000
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[30:33], a[80:83], v170, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[2:5], a[80:83], v170, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[70:73], v[30:33], a[72:75], v170, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[2:5], a[72:75], v170, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[30:33], a[68:71], v171, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[2:5], a[68:71], v171, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[124:127], v[30:33], a[60:63], v171, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[2:5], a[60:63], v171, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[136:139], v[30:33], a[56:59], v173, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[2:5], a[56:59], v173, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[140:143], v[30:33], a[48:51], v173, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[2:5], a[48:51], v173, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[152:155], v[30:33], a[40:43], v174, v168 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[2:5], a[40:43], v174, v168 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[156:159], v[30:33], a[36:39], v174, v168 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[2:5], a[36:39], v174, v168 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v175, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[2:5], v175, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:69], v[34:37], a[108:111], v170, v169 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:77], v[6:9], a[108:111], v170, v169 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[70:73], v[34:37], a[100:103], v170, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[6:9], a[100:103], v170, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[34:37], a[96:99], v171, v169 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[6:9], a[96:99], v171, v169 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[124:127], v[34:37], a[92:95], v171, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[6:9], a[92:95], v171, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[136:139], v[34:37], a[84:87], v173, v169 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[6:9], a[84:87], v173, v169 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[140:143], v[34:37], a[64:67], v173, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[6:9], a[64:67], v173, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[152:155], v[34:37], a[52:55], v174, v169 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[6:9], a[52:55], v174, v169 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[156:159], v[34:37], a[44:47], v174, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[6:9], a[44:47], v174, v169 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v175, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[6:9], v175, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:69], v[38:41], a[124:127], v170, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[10:13], a[124:127], v170, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[70:73], v[38:41], a[120:123], v170, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[10:13], a[120:123], v170, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:85], v[38:41], a[116:119], v171, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[10:13], a[116:119], v171, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[124:127], v[38:41], a[112:115], v171, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[10:13], a[112:115], v171, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[136:139], v[38:41], a[104:107], v173, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[10:13], a[104:107], v173, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[140:143], v[38:41], a[88:91], v173, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[10:13], a[88:91], v173, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[152:155], v[38:41], a[76:79], v174, v169 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[10:13], a[76:79], v174, v169 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[156:159], v[38:41], a[0:3], v174, v169 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[10:13], a[0:3], v174, v169 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v175, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[10:13], v175, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v176, v91, s[20:23], s8 offen offset:3072
	buffer_load_dword v177, v91, s[20:23], s9 offen offset:3072
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[180:183], v94, s[24:27], 0 offen offset:1664
	buffer_load_dwordx4 v[184:187], v95, s[24:27], 0 offen offset:1664
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:1664
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:1664
	v_or_b32_e32 v70, 0x18b00, v91
	v_or_b32_e32 v71, 0x1a700, v91
	v_or_b32_e32 v72, 0x1c300, v91
	v_or_b32_e32 v73, 0x1df00, v90
	ds_read_b32 v173, v70
	ds_read_b32 v174, v71
	ds_read_b32 v178, v72
	ds_read_b32 v179, v73
	ds_read_b128 v[66:69], v107
	ds_read_b128 v[74:77], v108
	ds_read_b128 v[78:81], v109
	ds_read_b128 v[82:85], v110
	ds_read_b128 v[124:127], v111
	ds_read_b128 v[128:131], v112
	ds_read_b128 v[132:135], v113
	ds_read_b128 v[136:139], v114
	ds_read_b128 v[140:143], v115
	ds_read_b128 v[144:147], v116
	ds_read_b128 v[148:151], v117
	ds_read_b128 v[152:155], v118
	ds_read_b128 v[156:159], v119
	ds_read_b128 v[160:163], v120
	ds_read_b128 v[164:167], v121
	ds_read_b128 v[168:171], v122
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[180:183] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[184:187] offset:32768
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[188:191] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[192:195] offset:32768
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[54:57], a[32:35], v173, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[74:77], v[14:17], a[32:35], v173, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[54:57], a[28:31], v173, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:85], v[14:17], a[28:31], v173, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[54:57], a[24:27], v174, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[14:17], a[24:27], v174, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[54:57], a[20:23], v174, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[14:17], a[20:23], v174, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[54:57], a[16:19], v178, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(10)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[14:17], a[16:19], v178, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[54:57], a[12:15], v178, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[14:17], a[12:15], v178, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[54:57], a[8:11], v179, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[14:17], a[8:11], v179, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[54:57], a[4:7], v179, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[14:17], a[4:7], v179, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v175, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v175, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[42:45], a[80:83], v173, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[74:77], v[18:21], a[80:83], v173, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[42:45], a[72:75], v173, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[18:21], a[72:75], v173, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[42:45], a[68:71], v174, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[18:21], a[68:71], v174, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[42:45], a[60:63], v174, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v174, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[42:45], a[56:59], v178, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[18:21], a[56:59], v178, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[42:45], a[48:51], v178, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[18:21], a[48:51], v178, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[42:45], a[40:43], v179, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[18:21], a[40:43], v179, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[42:45], a[36:39], v179, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[18:21], a[36:39], v179, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v175, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[18:21], v175, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:69], v[46:49], a[108:111], v173, v172 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[74:77], v[22:25], a[108:111], v173, v172 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[46:49], a[100:103], v173, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[22:25], a[100:103], v173, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[46:49], a[96:99], v174, v172 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[22:25], a[96:99], v174, v172 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[46:49], a[92:95], v174, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v174, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[46:49], a[84:87], v178, v172 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[22:25], a[84:87], v178, v172 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[46:49], a[64:67], v178, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[22:25], a[64:67], v178, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[46:49], a[52:55], v179, v172 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[22:25], a[52:55], v179, v172 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[46:49], a[44:47], v179, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[22:25], a[44:47], v179, v172 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v175, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[22:25], v175, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:69], v[50:53], a[124:127], v173, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[74:77], v[26:29], a[124:127], v173, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[50:53], a[120:123], v173, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:85], v[26:29], a[120:123], v173, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[50:53], a[116:119], v174, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[26:29], a[116:119], v174, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[50:53], a[112:115], v174, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[26:29], a[112:115], v174, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[50:53], a[104:107], v178, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[26:29], a[104:107], v178, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[50:53], a[88:91], v178, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[26:29], a[88:91], v178, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[50:53], a[76:79], v179, v172 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[26:29], a[76:79], v179, v172 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[50:53], a[0:3], v179, v172 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[26:29], a[0:3], v179, v172 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v175, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[26:29], v175, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v91, s[20:23], s8 offen offset:3328
	buffer_load_dword v180, v91, s[20:23], s9 offen offset:3328
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[74:77], v94, s[24:27], 0 offen offset:1792
	buffer_load_dwordx4 v[184:187], v95, s[24:27], 0 offen offset:1792
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:1792
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:1792
	v_or_b32_e32 v54, 0x18c00, v91
	v_or_b32_e32 v55, 0x1a800, v91
	v_or_b32_e32 v56, 0x1c400, v91
	v_or_b32_e32 v57, 0x1e000, v91
	ds_read_b32 v178, v54
	ds_read_b32 v179, v55
	ds_read_b32 v181, v56
	ds_read_b32 v182, v57
	ds_read_b128 v[66:69], v93
	ds_read_b128 v[78:81], v93 offset:4096
	ds_read_b128 v[82:85], v92
	ds_read_b128 v[124:127], v92 offset:4096
	ds_read_b128 v[128:131], v93 offset:8192
	ds_read_b128 v[132:135], v93 offset:12288
	ds_read_b128 v[136:139], v92 offset:8192
	ds_read_b128 v[140:143], v92 offset:12288
	ds_read_b128 v[144:147], v93 offset:16384
	ds_read_b128 v[148:151], v93 offset:20480
	ds_read_b128 v[152:155], v92 offset:16384
	ds_read_b128 v[156:159], v92 offset:20480
	ds_read_b128 v[160:163], v93 offset:24576
	ds_read_b128 v[164:167], v93 offset:28672
	ds_read_b128 v[168:171], v92 offset:24576
	ds_read_b128 v[172:175], v92 offset:28672
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[74:77]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[184:187]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[188:191]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[192:195]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[62:65], a[32:35], v178, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[58:61], a[32:35], v178, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[62:65], a[28:31], v178, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[58:61], a[28:31], v178, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[62:65], a[24:27], v179, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[58:61], a[24:27], v179, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[62:65], a[20:23], v179, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[58:61], a[20:23], v179, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[62:65], a[16:19], v181, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[58:61], a[16:19], v181, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[62:65], a[12:15], v181, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[58:61], a[12:15], v181, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[62:65], a[8:11], v182, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[168:171], v[58:61], a[8:11], v182, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[62:65], a[4:7], v182, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[172:175], v[58:61], a[4:7], v182, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v183, 0x7000, v102
	buffer_load_dwordx4 v[74:77], v183, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[54:57], v183, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[30:33], a[80:83], v178, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[82:85], v[2:5], a[80:83], v178, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[30:33], a[72:75], v178, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[2:5], a[72:75], v178, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[30:33], a[68:71], v179, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[2:5], a[68:71], v179, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[30:33], a[60:63], v179, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[2:5], a[60:63], v179, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[30:33], a[56:59], v181, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[152:155], v[2:5], a[56:59], v181, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[30:33], a[48:51], v181, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[2:5], a[48:51], v181, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[30:33], a[40:43], v182, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[168:171], v[2:5], a[40:43], v182, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[30:33], a[36:39], v182, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[2:5], a[36:39], v182, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v183, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[30:33], v183, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:69], v[34:37], a[108:111], v178, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[6:9], a[108:111], v178, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[34:37], a[100:103], v178, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:127], v[6:9], a[100:103], v178, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[34:37], a[96:99], v179, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[6:9], a[96:99], v179, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[34:37], a[92:95], v179, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[6:9], a[92:95], v179, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[34:37], a[84:87], v181, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[6:9], a[84:87], v181, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[34:37], a[64:67], v181, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:159], v[6:9], a[64:67], v181, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[34:37], a[52:55], v182, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:171], v[6:9], a[52:55], v182, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[34:37], a[44:47], v182, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[6:9], a[44:47], v182, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[62:65], v183, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[34:37], v183, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:69], v[38:41], a[124:127], v178, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:85], v[10:13], a[124:127], v178, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[38:41], a[120:123], v178, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[10:13], a[120:123], v178, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[38:41], a[116:119], v179, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[10:13], a[116:119], v179, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[38:41], a[112:115], v179, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[10:13], a[112:115], v179, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[38:41], a[104:107], v181, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:155], v[10:13], a[104:107], v181, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[38:41], a[88:91], v181, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[10:13], a[88:91], v181, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[38:41], a[76:79], v182, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[168:171], v[10:13], a[76:79], v182, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[38:41], a[0:3], v182, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[10:13], a[0:3], v182, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[66:69], v183, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[38:41], v183, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v181, v91, s[20:23], s8 offen offset:3584
	buffer_load_dword v182, v91, s[20:23], s9 offen offset:3584
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[6:9], v94, s[24:27], 0 offen offset:1920
	buffer_load_dwordx4 v[10:13], v95, s[24:27], 0 offen offset:1920
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:1920
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:1920
	v_or_b32_e32 v2, 0x18d00, v91
	v_or_b32_e32 v3, 0x1a900, v91
	v_or_b32_e32 v4, 0x1c500, v91
	v_or_b32_e32 v5, 0x1e100, v91
	ds_read_b32 v184, v2
	ds_read_b32 v185, v3
	ds_read_b32 v186, v4
	ds_read_b32 v187, v5
	ds_read_b128 v[78:81], v93 offset:32768
	ds_read_b128 v[82:85], v93 offset:36864
	ds_read_b128 v[124:127], v92 offset:32768
	ds_read_b128 v[128:131], v92 offset:36864
	ds_read_b128 v[132:135], v93 offset:40960
	ds_read_b128 v[136:139], v93 offset:45056
	ds_read_b128 v[140:143], v92 offset:40960
	ds_read_b128 v[144:147], v92 offset:45056
	ds_read_b128 v[148:151], v93 offset:49152
	ds_read_b128 v[152:155], v93 offset:53248
	ds_read_b128 v[156:159], v92 offset:49152
	ds_read_b128 v[160:163], v92 offset:53248
	ds_read_b128 v[164:167], v93 offset:57344
	ds_read_b128 v[168:171], v93 offset:61440
	ds_read_b128 v[172:175], v92 offset:57344
	ds_read_b128 v[176:179], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[6:9]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[10:13]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[188:191]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[192:195]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[70:73], a[32:35], v184, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[124:127], v[14:17], a[32:35], v184, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:85], v[70:73], a[28:31], v184, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[128:131], v[14:17], a[28:31], v184, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[70:73], a[24:27], v185, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[140:143], v[14:17], a[24:27], v185, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[70:73], a[20:23], v185, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[144:147], v[14:17], a[20:23], v185, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[70:73], a[16:19], v186, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[156:159], v[14:17], a[16:19], v186, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[70:73], a[12:15], v186, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[160:163], v[14:17], a[12:15], v186, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[70:73], a[8:11], v187, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[172:175], v[14:17], a[8:11], v187, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[70:73], a[4:7], v187, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[176:179], v[14:17], a[4:7], v187, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[70:73], v183, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v183, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[78:81], v[42:45], a[80:83], v184, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[124:127], v[18:21], a[80:83], v184, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[42:45], a[72:75], v184, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[128:131], v[18:21], a[72:75], v184, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[42:45], a[68:71], v185, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[140:143], v[18:21], a[68:71], v185, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[42:45], a[60:63], v185, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[144:147], v[18:21], a[60:63], v185, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[42:45], a[56:59], v186, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[156:159], v[18:21], a[56:59], v186, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[42:45], a[48:51], v186, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[160:163], v[18:21], a[48:51], v186, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[42:45], a[40:43], v187, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[172:175], v[18:21], a[40:43], v187, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[42:45], a[36:39], v187, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[176:179], v[18:21], a[36:39], v187, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v183, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[6:9], v183, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[46:49], a[108:111], v184, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[124:127], v[22:25], a[108:111], v184, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[46:49], a[100:103], v184, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[128:131], v[22:25], a[100:103], v184, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[46:49], a[96:99], v185, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[140:143], v[22:25], a[96:99], v185, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[46:49], a[92:95], v185, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[144:147], v[22:25], a[92:95], v185, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[46:49], a[84:87], v186, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[156:159], v[22:25], a[84:87], v186, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[46:49], a[64:67], v186, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[160:163], v[22:25], a[64:67], v186, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[46:49], a[52:55], v187, v180 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[172:175], v[22:25], a[52:55], v187, v180 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[46:49], a[44:47], v187, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[176:179], v[22:25], a[44:47], v187, v180 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v183, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v183, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[78:81], v[50:53], a[124:127], v184, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[124:127], v[26:29], a[124:127], v184, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:85], v[50:53], a[120:123], v184, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[128:131], v[26:29], a[120:123], v184, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[50:53], a[116:119], v185, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[140:143], v[26:29], a[116:119], v185, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[50:53], a[112:115], v185, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[144:147], v[26:29], a[112:115], v185, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[50:53], a[104:107], v186, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[156:159], v[26:29], a[104:107], v186, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[50:53], a[88:91], v186, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[160:163], v[26:29], a[88:91], v186, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[50:53], a[76:79], v187, v180 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[172:175], v[26:29], a[76:79], v187, v180 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[50:53], a[0:3], v187, v180 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[176:179], v[26:29], a[0:3], v187, v180 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v183, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v183, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v91, s[20:23], s8 offen offset:3840
	buffer_load_dword v176, v91, s[20:23], s9 offen offset:3840
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[22:25], v94, s[24:27], 0 offen offset:2048
	buffer_load_dwordx4 v[184:187], v95, s[24:27], 0 offen offset:2048
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:2048
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:2048
	v_or_b32_e32 v18, 0x18e00, v91
	v_or_b32_e32 v19, 0x1aa00, v91
	v_or_b32_e32 v20, 0x1c600, v91
	v_or_b32_e32 v21, 0x1e200, v91
	ds_read_b32 v177, v18
	ds_read_b32 v178, v19
	ds_read_b32 v179, v20
	ds_read_b32 v180, v21
	ds_read_b128 v[26:29], v107
	ds_read_b128 v[78:81], v108
	ds_read_b128 v[82:85], v109
	ds_read_b128 v[124:127], v110
	ds_read_b128 v[128:131], v111
	ds_read_b128 v[132:135], v112
	ds_read_b128 v[136:139], v113
	ds_read_b128 v[140:143], v114
	ds_read_b128 v[144:147], v115
	ds_read_b128 v[148:151], v116
	ds_read_b128 v[152:155], v117
	ds_read_b128 v[156:159], v118
	ds_read_b128 v[160:163], v119
	ds_read_b128 v[164:167], v120
	ds_read_b128 v[168:171], v121
	ds_read_b128 v[172:175], v122
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[22:25] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[184:187] offset:32768
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[188:191] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[192:195] offset:32768
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:29], v[74:77], a[32:35], v177, v181 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[78:81], v[54:57], a[32:35], v177, v181 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[82:85], v[74:77], a[28:31], v177, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[54:57], a[28:31], v177, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[74:77], a[24:27], v178, v181 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[54:57], a[24:27], v178, v181 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[74:77], a[20:23], v178, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[54:57], a[20:23], v178, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[74:77], a[16:19], v179, v181 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(10)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[54:57], a[16:19], v179, v181 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[74:77], a[12:15], v179, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[54:57], a[12:15], v179, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[74:77], a[8:11], v180, v181 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[54:57], a[8:11], v180, v181 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[74:77], a[4:7], v180, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[172:175], v[54:57], a[4:7], v180, v181 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v183, 0x8000, v102
	buffer_load_dwordx4 v[74:77], v183, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[54:57], v183, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[26:29], v[58:61], a[80:83], v177, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[78:81], v[30:33], a[80:83], v177, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[82:85], v[58:61], a[72:75], v177, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[30:33], a[72:75], v177, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[58:61], a[68:71], v178, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[30:33], a[68:71], v178, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[58:61], a[60:63], v178, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[30:33], a[60:63], v178, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[58:61], a[56:59], v179, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[30:33], a[56:59], v179, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[58:61], a[48:51], v179, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[30:33], a[48:51], v179, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[58:61], a[40:43], v180, v181 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[30:33], a[40:43], v180, v181 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[58:61], a[36:39], v180, v181 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[30:33], a[36:39], v180, v181 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v183, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[18:21], v183, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[62:65], a[108:111], v177, v182 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[78:81], v[34:37], a[108:111], v177, v182 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[82:85], v[62:65], a[100:103], v177, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:127], v[34:37], a[100:103], v177, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[62:65], a[96:99], v178, v182 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[34:37], a[96:99], v178, v182 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[62:65], a[92:95], v178, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[34:37], a[92:95], v178, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[62:65], a[84:87], v179, v182 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[34:37], a[84:87], v179, v182 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[62:65], a[64:67], v179, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:159], v[34:37], a[64:67], v179, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[62:65], a[52:55], v180, v182 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[34:37], a[52:55], v180, v182 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[62:65], a[44:47], v180, v182 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[34:37], a[44:47], v180, v182 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v183, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[22:25], v183, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[26:29], v[66:69], a[124:127], v177, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[78:81], v[38:41], a[124:127], v177, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[82:85], v[66:69], a[120:123], v177, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[38:41], a[120:123], v177, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[66:69], a[116:119], v178, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[38:41], a[116:119], v178, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[66:69], a[112:115], v178, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[38:41], a[112:115], v178, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[66:69], a[104:107], v179, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[38:41], a[104:107], v179, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[66:69], a[88:91], v179, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[38:41], a[88:91], v179, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[66:69], a[76:79], v180, v182 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[38:41], a[76:79], v180, v182 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[66:69], a[0:3], v180, v182 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[38:41], a[0:3], v180, v182 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v183, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[26:29], v183, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v66, 0x1000, v91
	buffer_load_dword v67, v66, s[20:23], s8 offen
	buffer_load_dword v177, v66, s[20:23], s9 offen
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[184:187], v94, s[24:27], 0 offen offset:2176
	buffer_load_dwordx4 v[188:191], v95, s[24:27], 0 offen offset:2176
	buffer_load_dwordx4 v[192:195], v96, s[24:27], 0 offen offset:2176
	buffer_load_dwordx4 v[196:199], v97, s[24:27], 0 offen offset:2176
	v_or_b32_e32 v58, 0x18f00, v90
	v_or_b32_e32 v59, 0x1ab00, v91
	v_or_b32_e32 v60, 0x1c700, v91
	v_or_b32_e32 v61, 0x1e300, v91
	ds_read_b128 v[62:65], v93
	ds_read_b128 v[78:81], v93 offset:4096
	ds_read_b128 v[82:85], v92
	ds_read_b128 v[124:127], v92 offset:4096
	ds_read_b128 v[128:131], v93 offset:8192
	ds_read_b128 v[132:135], v93 offset:12288
	ds_read_b128 v[136:139], v92 offset:8192
	ds_read_b128 v[140:143], v92 offset:12288
	ds_read_b128 v[144:147], v93 offset:16384
	ds_read_b128 v[148:151], v93 offset:20480
	ds_read_b128 v[152:155], v92 offset:16384
	ds_read_b128 v[156:159], v92 offset:20480
	ds_read_b128 v[160:163], v93 offset:24576
	ds_read_b128 v[164:167], v93 offset:28672
	ds_read_b128 v[168:171], v92 offset:24576
	ds_read_b128 v[172:175], v92 offset:28672
	ds_read_b32 v68, v58
	ds_read_b32 v69, v59
	ds_read_b32 v178, v60
	ds_read_b32 v179, v61
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[184:187]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[188:191]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[192:195]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[196:199]
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[62:65], v[70:73], a[32:35], v68, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[2:5], a[32:35], v68, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[70:73], a[28:31], v68, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[2:5], a[28:31], v68, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[70:73], a[24:27], v69, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[2:5], a[24:27], v69, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[70:73], a[20:23], v69, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[2:5], a[20:23], v69, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[70:73], a[16:19], v178, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[2:5], a[16:19], v178, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[70:73], a[12:15], v178, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[2:5], a[12:15], v178, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[70:73], a[8:11], v179, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[168:171], v[2:5], a[8:11], v179, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[70:73], a[4:7], v179, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[172:175], v[2:5], a[4:7], v179, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v183, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v183, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[62:65], v[42:45], a[80:83], v68, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[82:85], v[6:9], a[80:83], v68, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[42:45], a[72:75], v68, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[6:9], a[72:75], v68, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[42:45], a[68:71], v69, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[6:9], a[68:71], v69, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[42:45], a[60:63], v69, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[6:9], a[60:63], v69, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[42:45], a[56:59], v178, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[152:155], v[6:9], a[56:59], v178, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[42:45], a[48:51], v178, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[6:9], a[48:51], v178, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[42:45], a[40:43], v179, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[168:171], v[6:9], a[40:43], v179, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[42:45], a[36:39], v179, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[6:9], a[36:39], v179, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v183, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[6:9], v183, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[62:65], v[46:49], a[108:111], v68, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[10:13], a[108:111], v68, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[46:49], a[100:103], v68, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:127], v[10:13], a[100:103], v68, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[46:49], a[96:99], v69, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[10:13], a[96:99], v69, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[46:49], a[92:95], v69, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[10:13], a[92:95], v69, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[46:49], a[84:87], v178, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[10:13], a[84:87], v178, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[46:49], a[64:67], v178, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:159], v[10:13], a[64:67], v178, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[46:49], a[52:55], v179, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:171], v[10:13], a[52:55], v179, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[46:49], a[44:47], v179, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[10:13], a[44:47], v179, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v183, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v183, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[62:65], v[50:53], a[124:127], v68, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:85], v[14:17], a[124:127], v68, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[50:53], a[120:123], v68, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[14:17], a[120:123], v68, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[50:53], a[116:119], v69, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[14:17], a[116:119], v69, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[50:53], a[112:115], v69, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[14:17], a[112:115], v69, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[50:53], a[104:107], v178, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:155], v[14:17], a[104:107], v178, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[50:53], a[88:91], v178, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[14:17], a[88:91], v178, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[50:53], a[76:79], v179, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[168:171], v[14:17], a[76:79], v179, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[50:53], a[0:3], v179, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[14:17], a[0:3], v179, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v183, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v183, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v66, s[20:23], s8 offen offset:256
	buffer_load_dword v176, v66, s[20:23], s9 offen offset:256
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[180:183], v94, s[24:27], 0 offen offset:2304
	buffer_load_dwordx4 v[184:187], v95, s[24:27], 0 offen offset:2304
	buffer_load_dwordx4 v[188:191], v96, s[24:27], 0 offen offset:2304
	buffer_load_dwordx4 v[192:195], v97, s[24:27], 0 offen offset:2304
	v_or_b32_e32 v62, 0x19000, v91
	v_or_b32_e32 v63, 0x1ac00, v91
	v_or_b32_e32 v64, 0x1c800, v91
	v_or_b32_e32 v65, 0x1e400, v91
	ds_read_b32 v72, v62
	ds_read_b32 v73, v63
	ds_read_b32 v178, v64
	ds_read_b32 v179, v65
	ds_read_b128 v[68:71], v93 offset:32768
	ds_read_b128 v[78:81], v93 offset:36864
	ds_read_b128 v[82:85], v92 offset:32768
	ds_read_b128 v[124:127], v92 offset:36864
	ds_read_b128 v[128:131], v93 offset:40960
	ds_read_b128 v[132:135], v93 offset:45056
	ds_read_b128 v[136:139], v92 offset:40960
	ds_read_b128 v[140:143], v92 offset:45056
	ds_read_b128 v[144:147], v93 offset:49152
	ds_read_b128 v[148:151], v93 offset:53248
	ds_read_b128 v[152:155], v92 offset:49152
	ds_read_b128 v[156:159], v92 offset:53248
	ds_read_b128 v[160:163], v93 offset:57344
	ds_read_b128 v[164:167], v93 offset:61440
	ds_read_b128 v[168:171], v92 offset:57344
	ds_read_b128 v[172:175], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[180:183]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[184:187]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[188:191]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[192:195]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[74:77], a[32:35], v72, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[82:85], v[54:57], a[32:35], v72, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[74:77], a[28:31], v72, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[124:127], v[54:57], a[28:31], v72, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[74:77], a[24:27], v73, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[136:139], v[54:57], a[24:27], v73, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[74:77], a[20:23], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[140:143], v[54:57], a[20:23], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[74:77], a[16:19], v178, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[152:155], v[54:57], a[16:19], v178, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[74:77], a[12:15], v178, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[156:159], v[54:57], a[12:15], v178, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[74:77], a[8:11], v179, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[168:171], v[54:57], a[8:11], v179, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[74:77], a[4:7], v179, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[172:175], v[54:57], a[4:7], v179, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v180, 0x9000, v102
	buffer_load_dwordx4 v[62:65], v180, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[54:57], v180, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[30:33], a[80:83], v72, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[82:85], v[18:21], a[80:83], v72, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[30:33], a[72:75], v72, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[124:127], v[18:21], a[72:75], v72, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[30:33], a[68:71], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[136:139], v[18:21], a[68:71], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[30:33], a[60:63], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[140:143], v[18:21], a[60:63], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[30:33], a[56:59], v178, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[152:155], v[18:21], a[56:59], v178, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[30:33], a[48:51], v178, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[156:159], v[18:21], a[48:51], v178, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[30:33], a[40:43], v179, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[168:171], v[18:21], a[40:43], v179, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[30:33], a[36:39], v179, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[172:175], v[18:21], a[36:39], v179, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v180, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[18:21], v180, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[34:37], a[108:111], v72, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[82:85], v[22:25], a[108:111], v72, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[34:37], a[100:103], v72, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[124:127], v[22:25], a[100:103], v72, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[34:37], a[96:99], v73, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[136:139], v[22:25], a[96:99], v73, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[34:37], a[92:95], v73, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[140:143], v[22:25], a[92:95], v73, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[34:37], a[84:87], v178, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[152:155], v[22:25], a[84:87], v178, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[34:37], a[64:67], v178, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[156:159], v[22:25], a[64:67], v178, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[34:37], a[52:55], v179, v177 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[168:171], v[22:25], a[52:55], v179, v177 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[34:37], a[44:47], v179, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[172:175], v[22:25], a[44:47], v179, v177 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v180, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[22:25], v180, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[38:41], a[124:127], v72, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[82:85], v[26:29], a[124:127], v72, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[38:41], a[120:123], v72, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[124:127], v[26:29], a[120:123], v72, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[38:41], a[116:119], v73, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[136:139], v[26:29], a[116:119], v73, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[38:41], a[112:115], v73, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[140:143], v[26:29], a[112:115], v73, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[38:41], a[104:107], v178, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[152:155], v[26:29], a[104:107], v178, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[38:41], a[88:91], v178, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[156:159], v[26:29], a[88:91], v178, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[38:41], a[76:79], v179, v177 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[168:171], v[26:29], a[76:79], v179, v177 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[38:41], a[0:3], v179, v177 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[172:175], v[26:29], a[0:3], v179, v177 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v180, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[26:29], v180, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v67, v66, s[20:23], s8 offen offset:512
	buffer_load_dword v84, v66, s[20:23], s9 offen offset:512
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[182:185], v94, s[24:27], 0 offen offset:2432
	buffer_load_dwordx4 v[186:189], v95, s[24:27], 0 offen offset:2432
	buffer_load_dwordx4 v[190:193], v96, s[24:27], 0 offen offset:2432
	buffer_load_dwordx4 v[194:197], v97, s[24:27], 0 offen offset:2432
	v_or_b32_e32 v85, 0x19100, v91
	v_or_b32_e32 v172, 0x1ad00, v91
	v_or_b32_e32 v173, 0x1c900, v91
	v_or_b32_e32 v174, 0x1e500, v91
	ds_read_b32 v85, v85
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b128 v[68:71], v107
	ds_read_b128 v[72:75], v108
	ds_read_b128 v[76:79], v109
	ds_read_b128 v[80:83], v110
	ds_read_b128 v[124:127], v111
	ds_read_b128 v[128:131], v112
	ds_read_b128 v[132:135], v113
	ds_read_b128 v[136:139], v114
	ds_read_b128 v[140:143], v115
	ds_read_b128 v[144:147], v116
	ds_read_b128 v[148:151], v117
	ds_read_b128 v[152:155], v118
	ds_read_b128 v[156:159], v119
	ds_read_b128 v[160:163], v120
	ds_read_b128 v[164:167], v121
	ds_read_b128 v[168:171], v122
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[182:185] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[186:189] offset:32768
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[190:193] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[194:197] offset:32768
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[58:61], a[32:35], v85, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:75], v[2:5], a[32:35], v85, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[58:61], a[28:31], v85, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[2:5], a[28:31], v85, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v172, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[2:5], a[24:27], v172, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[58:61], a[20:23], v172, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[2:5], a[20:23], v172, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[58:61], a[16:19], v173, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(10)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[2:5], a[16:19], v173, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[58:61], a[12:15], v173, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[2:5], a[12:15], v173, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[58:61], a[8:11], v174, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[2:5], a[8:11], v174, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[58:61], a[4:7], v174, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[2:5], a[4:7], v174, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v180, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v180, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[42:45], a[80:83], v85, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:75], v[6:9], a[80:83], v85, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:79], v[42:45], a[72:75], v85, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[6:9], a[72:75], v85, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[42:45], a[68:71], v172, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[6:9], a[68:71], v172, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[42:45], a[60:63], v172, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[6:9], a[60:63], v172, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[42:45], a[56:59], v173, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[6:9], a[56:59], v173, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[42:45], a[48:51], v173, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[6:9], a[48:51], v173, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[42:45], a[40:43], v174, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[6:9], a[40:43], v174, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[42:45], a[36:39], v174, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[6:9], a[36:39], v174, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v180, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[6:9], v180, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[46:49], a[108:111], v85, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:75], v[10:13], a[108:111], v85, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[76:79], v[46:49], a[100:103], v85, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[10:13], a[100:103], v85, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[46:49], a[96:99], v172, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[10:13], a[96:99], v172, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[46:49], a[92:95], v172, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[10:13], a[92:95], v172, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[46:49], a[84:87], v173, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[10:13], a[84:87], v173, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[46:49], a[64:67], v173, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[10:13], a[64:67], v173, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[46:49], a[52:55], v174, v176 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[10:13], a[52:55], v174, v176 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[46:49], a[44:47], v174, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[10:13], a[44:47], v174, v176 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v180, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v180, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[50:53], a[124:127], v85, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[72:75], v[14:17], a[124:127], v85, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[76:79], v[50:53], a[120:123], v85, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[14:17], a[120:123], v85, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[50:53], a[116:119], v172, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[14:17], a[116:119], v172, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[50:53], a[112:115], v172, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[14:17], a[112:115], v172, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[50:53], a[104:107], v173, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[14:17], a[104:107], v173, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[50:53], a[88:91], v173, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[14:17], a[88:91], v173, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[50:53], a[76:79], v174, v176 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[14:17], a[76:79], v174, v176 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[50:53], a[0:3], v174, v176 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[14:17], a[0:3], v174, v176 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v180, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v180, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v85, v66, s[20:23], s8 offen offset:768
	buffer_load_dword v123, v66, s[20:23], s9 offen offset:768
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[176:179], v94, s[24:27], 0 offen offset:2560
	buffer_load_dwordx4 v[180:183], v95, s[24:27], 0 offen offset:2560
	buffer_load_dwordx4 v[184:187], v96, s[24:27], 0 offen offset:2560
	buffer_load_dwordx4 v[188:191], v97, s[24:27], 0 offen offset:2560
	v_or_b32_e32 v172, 0x19200, v91
	v_or_b32_e32 v173, 0x1ae00, v91
	v_or_b32_e32 v174, 0x1ca00, v91
	v_or_b32_e32 v175, 0x1e600, v91
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b32 v175, v175
	ds_read_b128 v[68:71], v93
	ds_read_b128 v[72:75], v93 offset:4096
	ds_read_b128 v[76:79], v92
	ds_read_b128 v[80:83], v92 offset:4096
	ds_read_b128 v[124:127], v93 offset:8192
	ds_read_b128 v[128:131], v93 offset:12288
	ds_read_b128 v[132:135], v92 offset:8192
	ds_read_b128 v[136:139], v92 offset:12288
	ds_read_b128 v[140:143], v93 offset:16384
	ds_read_b128 v[144:147], v93 offset:20480
	ds_read_b128 v[148:151], v92 offset:16384
	ds_read_b128 v[152:155], v92 offset:20480
	ds_read_b128 v[156:159], v93 offset:24576
	ds_read_b128 v[160:163], v93 offset:28672
	ds_read_b128 v[164:167], v92 offset:24576
	ds_read_b128 v[168:171], v92 offset:28672
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[176:179]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[180:183]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[184:187]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[188:191]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[62:65], a[32:35], v172, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[54:57], a[32:35], v172, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[62:65], a[28:31], v172, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[54:57], a[28:31], v172, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[62:65], a[24:27], v173, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[54:57], a[24:27], v173, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[62:65], a[20:23], v173, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[54:57], a[20:23], v173, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[62:65], a[16:19], v174, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[54:57], a[16:19], v174, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[62:65], a[12:15], v174, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[54:57], a[12:15], v174, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[62:65], a[8:11], v175, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[54:57], a[8:11], v175, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[62:65], a[4:7], v175, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[54:57], a[4:7], v175, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v176, 0xa000, v102
	buffer_load_dwordx4 v[62:65], v176, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[54:57], v176, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[30:33], a[80:83], v172, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[18:21], a[80:83], v172, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[30:33], a[72:75], v172, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[18:21], a[72:75], v172, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[30:33], a[68:71], v173, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[18:21], a[68:71], v173, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[30:33], a[60:63], v173, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v173, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[30:33], a[56:59], v174, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[18:21], a[56:59], v174, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[30:33], a[48:51], v174, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[18:21], a[48:51], v174, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[30:33], a[40:43], v175, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[18:21], a[40:43], v175, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[30:33], a[36:39], v175, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[18:21], a[36:39], v175, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v176, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[18:21], v176, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[34:37], a[108:111], v172, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[76:79], v[22:25], a[108:111], v172, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[34:37], a[100:103], v172, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[22:25], a[100:103], v172, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[34:37], a[96:99], v173, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[22:25], a[96:99], v173, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[34:37], a[92:95], v173, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v173, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[34:37], a[84:87], v174, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[22:25], a[84:87], v174, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[34:37], a[64:67], v174, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[22:25], a[64:67], v174, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[34:37], a[52:55], v175, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[22:25], a[52:55], v175, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[34:37], a[44:47], v175, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[22:25], a[44:47], v175, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v176, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[22:25], v176, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[38:41], a[124:127], v172, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[76:79], v[26:29], a[124:127], v172, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:75], v[38:41], a[120:123], v172, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[26:29], a[120:123], v172, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[38:41], a[116:119], v173, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[26:29], a[116:119], v173, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[38:41], a[112:115], v173, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[26:29], a[112:115], v173, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[38:41], a[104:107], v174, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[26:29], a[104:107], v174, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[38:41], a[88:91], v174, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[26:29], a[88:91], v174, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[38:41], a[76:79], v175, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[26:29], a[76:79], v175, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[38:41], a[0:3], v175, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[26:29], a[0:3], v175, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v176, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[26:29], v176, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v67, v66, s[20:23], s8 offen offset:1024
	buffer_load_dword v84, v66, s[20:23], s9 offen offset:1024
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[178:181], v94, s[24:27], 0 offen offset:2688
	buffer_load_dwordx4 v[182:185], v95, s[24:27], 0 offen offset:2688
	buffer_load_dwordx4 v[186:189], v96, s[24:27], 0 offen offset:2688
	buffer_load_dwordx4 v[190:193], v97, s[24:27], 0 offen offset:2688
	v_or_b32_e32 v172, 0x19300, v91
	v_or_b32_e32 v173, 0x1af00, v90
	v_or_b32_e32 v174, 0x1cb00, v91
	v_or_b32_e32 v175, 0x1e700, v91
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b32 v175, v175
	ds_read_b128 v[68:71], v93 offset:32768
	ds_read_b128 v[72:75], v93 offset:36864
	ds_read_b128 v[76:79], v92 offset:32768
	ds_read_b128 v[80:83], v92 offset:36864
	ds_read_b128 v[124:127], v93 offset:40960
	ds_read_b128 v[128:131], v93 offset:45056
	ds_read_b128 v[132:135], v92 offset:40960
	ds_read_b128 v[136:139], v92 offset:45056
	ds_read_b128 v[140:143], v93 offset:49152
	ds_read_b128 v[144:147], v93 offset:53248
	ds_read_b128 v[148:151], v92 offset:49152
	ds_read_b128 v[152:155], v92 offset:53248
	ds_read_b128 v[156:159], v93 offset:57344
	ds_read_b128 v[160:163], v93 offset:61440
	ds_read_b128 v[164:167], v92 offset:57344
	ds_read_b128 v[168:171], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[178:181]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[182:185]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[186:189]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[190:193]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[58:61], a[32:35], v172, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[2:5], a[32:35], v172, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[58:61], a[28:31], v172, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[2:5], a[28:31], v172, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v173, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[2:5], a[24:27], v173, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[58:61], a[20:23], v173, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[2:5], a[20:23], v173, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[58:61], a[16:19], v174, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[2:5], a[16:19], v174, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[58:61], a[12:15], v174, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[2:5], a[12:15], v174, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[58:61], a[8:11], v175, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[2:5], a[8:11], v175, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[58:61], a[4:7], v175, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[2:5], a[4:7], v175, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v176, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v176, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[42:45], a[80:83], v172, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[6:9], a[80:83], v172, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[42:45], a[72:75], v172, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[6:9], a[72:75], v172, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[42:45], a[68:71], v173, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[6:9], a[68:71], v173, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[42:45], a[60:63], v173, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[6:9], a[60:63], v173, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[42:45], a[56:59], v174, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[6:9], a[56:59], v174, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[42:45], a[48:51], v174, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[6:9], a[48:51], v174, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[42:45], a[40:43], v175, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[6:9], a[40:43], v175, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[42:45], a[36:39], v175, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[6:9], a[36:39], v175, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v176, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[6:9], v176, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[46:49], a[108:111], v172, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[76:79], v[10:13], a[108:111], v172, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[46:49], a[100:103], v172, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[10:13], a[100:103], v172, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[46:49], a[96:99], v173, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[10:13], a[96:99], v173, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[46:49], a[92:95], v173, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[10:13], a[92:95], v173, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[46:49], a[84:87], v174, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[10:13], a[84:87], v174, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[46:49], a[64:67], v174, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[10:13], a[64:67], v174, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[46:49], a[52:55], v175, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[10:13], a[52:55], v175, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[46:49], a[44:47], v175, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[10:13], a[44:47], v175, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v176, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v176, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[50:53], a[124:127], v172, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[76:79], v[14:17], a[124:127], v172, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:75], v[50:53], a[120:123], v172, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[14:17], a[120:123], v172, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[50:53], a[116:119], v173, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[14:17], a[116:119], v173, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[50:53], a[112:115], v173, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[14:17], a[112:115], v173, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[50:53], a[104:107], v174, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[14:17], a[104:107], v174, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[50:53], a[88:91], v174, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[14:17], a[88:91], v174, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[50:53], a[76:79], v175, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[14:17], a[76:79], v175, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[50:53], a[0:3], v175, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[14:17], a[0:3], v175, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v176, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v176, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v85, v66, s[20:23], s8 offen offset:1280
	buffer_load_dword v123, v66, s[20:23], s9 offen offset:1280
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[176:179], v94, s[24:27], 0 offen offset:2816
	buffer_load_dwordx4 v[180:183], v95, s[24:27], 0 offen offset:2816
	buffer_load_dwordx4 v[184:187], v96, s[24:27], 0 offen offset:2816
	buffer_load_dwordx4 v[188:191], v97, s[24:27], 0 offen offset:2816
	v_or_b32_e32 v172, 0x19400, v91
	v_or_b32_e32 v173, 0x1b000, v91
	v_or_b32_e32 v174, 0x1cc00, v91
	v_or_b32_e32 v175, 0x1e800, v91
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b32 v175, v175
	ds_read_b128 v[68:71], v107
	ds_read_b128 v[72:75], v108
	ds_read_b128 v[76:79], v109
	ds_read_b128 v[80:83], v110
	ds_read_b128 v[124:127], v111
	ds_read_b128 v[128:131], v112
	ds_read_b128 v[132:135], v113
	ds_read_b128 v[136:139], v114
	ds_read_b128 v[140:143], v115
	ds_read_b128 v[144:147], v116
	ds_read_b128 v[148:151], v117
	ds_read_b128 v[152:155], v118
	ds_read_b128 v[156:159], v119
	ds_read_b128 v[160:163], v120
	ds_read_b128 v[164:167], v121
	ds_read_b128 v[168:171], v122
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[176:179] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[180:183] offset:32768
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[184:187] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[188:191] offset:32768
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[62:65], a[32:35], v172, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:75], v[54:57], a[32:35], v172, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[62:65], a[28:31], v172, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[54:57], a[28:31], v172, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[62:65], a[24:27], v173, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[54:57], a[24:27], v173, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[62:65], a[20:23], v173, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[54:57], a[20:23], v173, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[62:65], a[16:19], v174, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(10)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[54:57], a[16:19], v174, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[62:65], a[12:15], v174, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[54:57], a[12:15], v174, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[62:65], a[8:11], v175, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[54:57], a[8:11], v175, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[62:65], a[4:7], v175, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[54:57], a[4:7], v175, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v176, 0xb000, v102
	buffer_load_dwordx4 v[62:65], v176, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[54:57], v176, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[30:33], a[80:83], v172, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:75], v[18:21], a[80:83], v172, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:79], v[30:33], a[72:75], v172, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[18:21], a[72:75], v172, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[30:33], a[68:71], v173, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[18:21], a[68:71], v173, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[30:33], a[60:63], v173, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v173, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[30:33], a[56:59], v174, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[18:21], a[56:59], v174, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[30:33], a[48:51], v174, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[18:21], a[48:51], v174, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[30:33], a[40:43], v175, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[18:21], a[40:43], v175, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[30:33], a[36:39], v175, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[18:21], a[36:39], v175, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v176, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[18:21], v176, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[34:37], a[108:111], v172, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:75], v[22:25], a[108:111], v172, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[76:79], v[34:37], a[100:103], v172, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[22:25], a[100:103], v172, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[34:37], a[96:99], v173, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[22:25], a[96:99], v173, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[34:37], a[92:95], v173, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v173, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[34:37], a[84:87], v174, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[22:25], a[84:87], v174, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[34:37], a[64:67], v174, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[22:25], a[64:67], v174, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[34:37], a[52:55], v175, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[22:25], a[52:55], v175, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[34:37], a[44:47], v175, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[22:25], a[44:47], v175, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v176, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[22:25], v176, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[38:41], a[124:127], v172, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[72:75], v[26:29], a[124:127], v172, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[76:79], v[38:41], a[120:123], v172, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[26:29], a[120:123], v172, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[38:41], a[116:119], v173, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[26:29], a[116:119], v173, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[38:41], a[112:115], v173, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[26:29], a[112:115], v173, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[38:41], a[104:107], v174, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[26:29], a[104:107], v174, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[38:41], a[88:91], v174, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[26:29], a[88:91], v174, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[38:41], a[76:79], v175, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[26:29], a[76:79], v175, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[38:41], a[0:3], v175, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[26:29], a[0:3], v175, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v176, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[26:29], v176, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v67, v66, s[20:23], s8 offen offset:1536
	buffer_load_dword v84, v66, s[20:23], s9 offen offset:1536
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[178:181], v94, s[24:27], 0 offen offset:2944
	buffer_load_dwordx4 v[182:185], v95, s[24:27], 0 offen offset:2944
	buffer_load_dwordx4 v[186:189], v96, s[24:27], 0 offen offset:2944
	buffer_load_dwordx4 v[190:193], v97, s[24:27], 0 offen offset:2944
	v_or_b32_e32 v172, 0x19500, v91
	v_or_b32_e32 v173, 0x1b100, v91
	v_or_b32_e32 v174, 0x1cd00, v91
	v_or_b32_e32 v175, 0x1e900, v91
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b32 v175, v175
	ds_read_b128 v[68:71], v93
	ds_read_b128 v[72:75], v93 offset:4096
	ds_read_b128 v[76:79], v92
	ds_read_b128 v[80:83], v92 offset:4096
	ds_read_b128 v[124:127], v93 offset:8192
	ds_read_b128 v[128:131], v93 offset:12288
	ds_read_b128 v[132:135], v92 offset:8192
	ds_read_b128 v[136:139], v92 offset:12288
	ds_read_b128 v[140:143], v93 offset:16384
	ds_read_b128 v[144:147], v93 offset:20480
	ds_read_b128 v[148:151], v92 offset:16384
	ds_read_b128 v[152:155], v92 offset:20480
	ds_read_b128 v[156:159], v93 offset:24576
	ds_read_b128 v[160:163], v93 offset:28672
	ds_read_b128 v[164:167], v92 offset:24576
	ds_read_b128 v[168:171], v92 offset:28672
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[178:181]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[182:185]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[186:189]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[190:193]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[58:61], a[32:35], v172, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[2:5], a[32:35], v172, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[58:61], a[28:31], v172, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[2:5], a[28:31], v172, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v173, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[2:5], a[24:27], v173, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[58:61], a[20:23], v173, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[2:5], a[20:23], v173, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[58:61], a[16:19], v174, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[2:5], a[16:19], v174, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[58:61], a[12:15], v174, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[2:5], a[12:15], v174, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[58:61], a[8:11], v175, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[2:5], a[8:11], v175, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[58:61], a[4:7], v175, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[2:5], a[4:7], v175, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v176, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v176, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[42:45], a[80:83], v172, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[6:9], a[80:83], v172, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[42:45], a[72:75], v172, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[6:9], a[72:75], v172, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[42:45], a[68:71], v173, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[6:9], a[68:71], v173, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[42:45], a[60:63], v173, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[6:9], a[60:63], v173, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[42:45], a[56:59], v174, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[6:9], a[56:59], v174, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[42:45], a[48:51], v174, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[6:9], a[48:51], v174, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[42:45], a[40:43], v175, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[6:9], a[40:43], v175, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[42:45], a[36:39], v175, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[6:9], a[36:39], v175, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v176, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[6:9], v176, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[46:49], a[108:111], v172, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[76:79], v[10:13], a[108:111], v172, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[46:49], a[100:103], v172, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[10:13], a[100:103], v172, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[46:49], a[96:99], v173, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[10:13], a[96:99], v173, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[46:49], a[92:95], v173, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[10:13], a[92:95], v173, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[46:49], a[84:87], v174, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[10:13], a[84:87], v174, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[46:49], a[64:67], v174, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[10:13], a[64:67], v174, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[46:49], a[52:55], v175, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[10:13], a[52:55], v175, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[46:49], a[44:47], v175, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[10:13], a[44:47], v175, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v176, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v176, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[50:53], a[124:127], v172, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[76:79], v[14:17], a[124:127], v172, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:75], v[50:53], a[120:123], v172, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[14:17], a[120:123], v172, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[50:53], a[116:119], v173, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[14:17], a[116:119], v173, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[50:53], a[112:115], v173, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[14:17], a[112:115], v173, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[50:53], a[104:107], v174, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[14:17], a[104:107], v174, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[50:53], a[88:91], v174, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[14:17], a[88:91], v174, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[50:53], a[76:79], v175, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[14:17], a[76:79], v175, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[50:53], a[0:3], v175, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[14:17], a[0:3], v175, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v176, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v176, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v85, v66, s[20:23], s8 offen offset:1792
	buffer_load_dword v123, v66, s[20:23], s9 offen offset:1792
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[176:179], v94, s[24:27], 0 offen offset:3072
	buffer_load_dwordx4 v[180:183], v95, s[24:27], 0 offen offset:3072
	buffer_load_dwordx4 v[184:187], v96, s[24:27], 0 offen offset:3072
	buffer_load_dwordx4 v[188:191], v97, s[24:27], 0 offen offset:3072
	v_or_b32_e32 v172, 0x19600, v91
	v_or_b32_e32 v173, 0x1b200, v91
	v_or_b32_e32 v174, 0x1ce00, v91
	v_or_b32_e32 v175, 0x1ea00, v91
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b32 v175, v175
	ds_read_b128 v[68:71], v93 offset:32768
	ds_read_b128 v[72:75], v93 offset:36864
	ds_read_b128 v[76:79], v92 offset:32768
	ds_read_b128 v[80:83], v92 offset:36864
	ds_read_b128 v[124:127], v93 offset:40960
	ds_read_b128 v[128:131], v93 offset:45056
	ds_read_b128 v[132:135], v92 offset:40960
	ds_read_b128 v[136:139], v92 offset:45056
	ds_read_b128 v[140:143], v93 offset:49152
	ds_read_b128 v[144:147], v93 offset:53248
	ds_read_b128 v[148:151], v92 offset:49152
	ds_read_b128 v[152:155], v92 offset:53248
	ds_read_b128 v[156:159], v93 offset:57344
	ds_read_b128 v[160:163], v93 offset:61440
	ds_read_b128 v[164:167], v92 offset:57344
	ds_read_b128 v[168:171], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[176:179]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[180:183]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[184:187]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[188:191]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[62:65], a[32:35], v172, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[54:57], a[32:35], v172, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[62:65], a[28:31], v172, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[54:57], a[28:31], v172, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[62:65], a[24:27], v173, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[54:57], a[24:27], v173, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[62:65], a[20:23], v173, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[54:57], a[20:23], v173, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[62:65], a[16:19], v174, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[54:57], a[16:19], v174, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[62:65], a[12:15], v174, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[54:57], a[12:15], v174, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[62:65], a[8:11], v175, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[54:57], a[8:11], v175, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[62:65], a[4:7], v175, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[54:57], a[4:7], v175, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v176, 0xc000, v102
	buffer_load_dwordx4 v[62:65], v176, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[54:57], v176, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[30:33], a[80:83], v172, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[18:21], a[80:83], v172, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[30:33], a[72:75], v172, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[18:21], a[72:75], v172, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[30:33], a[68:71], v173, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[18:21], a[68:71], v173, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[30:33], a[60:63], v173, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v173, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[30:33], a[56:59], v174, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[18:21], a[56:59], v174, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[30:33], a[48:51], v174, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[18:21], a[48:51], v174, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[30:33], a[40:43], v175, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[18:21], a[40:43], v175, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[30:33], a[36:39], v175, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[18:21], a[36:39], v175, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v176, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[18:21], v176, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[34:37], a[108:111], v172, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[76:79], v[22:25], a[108:111], v172, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[34:37], a[100:103], v172, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[22:25], a[100:103], v172, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[34:37], a[96:99], v173, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[22:25], a[96:99], v173, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[34:37], a[92:95], v173, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v173, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[34:37], a[84:87], v174, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[22:25], a[84:87], v174, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[34:37], a[64:67], v174, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[22:25], a[64:67], v174, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[34:37], a[52:55], v175, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[22:25], a[52:55], v175, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[34:37], a[44:47], v175, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[22:25], a[44:47], v175, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v176, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[22:25], v176, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[38:41], a[124:127], v172, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[76:79], v[26:29], a[124:127], v172, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:75], v[38:41], a[120:123], v172, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[26:29], a[120:123], v172, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[38:41], a[116:119], v173, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[26:29], a[116:119], v173, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[38:41], a[112:115], v173, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[26:29], a[112:115], v173, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[38:41], a[104:107], v174, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[26:29], a[104:107], v174, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[38:41], a[88:91], v174, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[26:29], a[88:91], v174, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[38:41], a[76:79], v175, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[26:29], a[76:79], v175, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[38:41], a[0:3], v175, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[26:29], a[0:3], v175, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v176, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[26:29], v176, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v67, v66, s[20:23], s8 offen offset:2048
	buffer_load_dword v84, v66, s[20:23], s9 offen offset:2048
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[178:181], v94, s[24:27], 0 offen offset:3200
	buffer_load_dwordx4 v[182:185], v95, s[24:27], 0 offen offset:3200
	buffer_load_dwordx4 v[186:189], v96, s[24:27], 0 offen offset:3200
	buffer_load_dwordx4 v[190:193], v97, s[24:27], 0 offen offset:3200
	v_or_b32_e32 v172, 0x19700, v91
	v_or_b32_e32 v173, 0x1b300, v91
	v_or_b32_e32 v174, 0x1cf00, v90
	v_or_b32_e32 v175, 0x1eb00, v91
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b32 v175, v175
	ds_read_b128 v[68:71], v107
	ds_read_b128 v[72:75], v108
	ds_read_b128 v[76:79], v109
	ds_read_b128 v[80:83], v110
	ds_read_b128 v[124:127], v111
	ds_read_b128 v[128:131], v112
	ds_read_b128 v[132:135], v113
	ds_read_b128 v[136:139], v114
	ds_read_b128 v[140:143], v115
	ds_read_b128 v[144:147], v116
	ds_read_b128 v[148:151], v117
	ds_read_b128 v[152:155], v118
	ds_read_b128 v[156:159], v119
	ds_read_b128 v[160:163], v120
	ds_read_b128 v[164:167], v121
	ds_read_b128 v[168:171], v122
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[178:181] offset:32768
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[182:185] offset:32768
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[186:189] offset:32768
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[190:193] offset:32768
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[58:61], a[32:35], v172, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[72:75], v[2:5], a[32:35], v172, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[76:79], v[58:61], a[28:31], v172, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[2:5], a[28:31], v172, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[58:61], a[24:27], v173, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[128:131], v[2:5], a[24:27], v173, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[132:135], v[58:61], a[20:23], v173, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[2:5], a[20:23], v173, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[58:61], a[16:19], v174, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(10)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[144:147], v[2:5], a[16:19], v174, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[148:151], v[58:61], a[12:15], v174, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[2:5], a[12:15], v174, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[58:61], a[8:11], v175, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(6)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[160:163], v[2:5], a[8:11], v175, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[164:167], v[58:61], a[4:7], v175, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[2:5], a[4:7], v175, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v176, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v176, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[42:45], a[80:83], v172, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[72:75], v[6:9], a[80:83], v172, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[76:79], v[42:45], a[72:75], v172, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[6:9], a[72:75], v172, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[42:45], a[68:71], v173, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[128:131], v[6:9], a[68:71], v173, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[132:135], v[42:45], a[60:63], v173, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[6:9], a[60:63], v173, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[42:45], a[56:59], v174, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[144:147], v[6:9], a[56:59], v174, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[148:151], v[42:45], a[48:51], v174, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[6:9], a[48:51], v174, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[42:45], a[40:43], v175, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[160:163], v[6:9], a[40:43], v175, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[164:167], v[42:45], a[36:39], v175, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[6:9], a[36:39], v175, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v176, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[6:9], v176, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[46:49], a[108:111], v172, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[72:75], v[10:13], a[108:111], v172, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[76:79], v[46:49], a[100:103], v172, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[10:13], a[100:103], v172, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[46:49], a[96:99], v173, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[128:131], v[10:13], a[96:99], v173, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[132:135], v[46:49], a[92:95], v173, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[10:13], a[92:95], v173, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[46:49], a[84:87], v174, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[144:147], v[10:13], a[84:87], v174, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[148:151], v[46:49], a[64:67], v174, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[10:13], a[64:67], v174, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[46:49], a[52:55], v175, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[160:163], v[10:13], a[52:55], v175, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[164:167], v[46:49], a[44:47], v175, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[10:13], a[44:47], v175, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v176, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v176, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[50:53], a[124:127], v172, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[72:75], v[14:17], a[124:127], v172, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[76:79], v[50:53], a[120:123], v172, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[14:17], a[120:123], v172, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[50:53], a[116:119], v173, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[128:131], v[14:17], a[116:119], v173, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[132:135], v[50:53], a[112:115], v173, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[14:17], a[112:115], v173, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[50:53], a[104:107], v174, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[144:147], v[14:17], a[104:107], v174, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[148:151], v[50:53], a[88:91], v174, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[14:17], a[88:91], v174, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[50:53], a[76:79], v175, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[160:163], v[14:17], a[76:79], v175, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[164:167], v[50:53], a[0:3], v175, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[14:17], a[0:3], v175, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v176, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v176, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v85, v66, s[20:23], s8 offen offset:2304
	buffer_load_dword v123, v66, s[20:23], s9 offen offset:2304
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[176:179], v94, s[24:27], 0 offen offset:3328
	buffer_load_dwordx4 v[180:183], v95, s[24:27], 0 offen offset:3328
	buffer_load_dwordx4 v[184:187], v96, s[24:27], 0 offen offset:3328
	buffer_load_dwordx4 v[188:191], v97, s[24:27], 0 offen offset:3328
	v_or_b32_e32 v172, 0x19800, v91
	v_or_b32_e32 v173, 0x1b400, v91
	v_or_b32_e32 v174, 0x1d000, v91
	v_or_b32_e32 v175, 0x1ec00, v91
	ds_read_b32 v172, v172
	ds_read_b32 v173, v173
	ds_read_b32 v174, v174
	ds_read_b32 v175, v175
	ds_read_b128 v[68:71], v93
	ds_read_b128 v[72:75], v93 offset:4096
	ds_read_b128 v[76:79], v92
	ds_read_b128 v[80:83], v92 offset:4096
	ds_read_b128 v[124:127], v93 offset:8192
	ds_read_b128 v[128:131], v93 offset:12288
	ds_read_b128 v[132:135], v92 offset:8192
	ds_read_b128 v[136:139], v92 offset:12288
	ds_read_b128 v[140:143], v93 offset:16384
	ds_read_b128 v[144:147], v93 offset:20480
	ds_read_b128 v[148:151], v92 offset:16384
	ds_read_b128 v[152:155], v92 offset:20480
	ds_read_b128 v[156:159], v93 offset:24576
	ds_read_b128 v[160:163], v93 offset:28672
	ds_read_b128 v[164:167], v92 offset:24576
	ds_read_b128 v[168:171], v92 offset:28672
	s_waitcnt vmcnt(3)
	ds_write_b128 v103, v[176:179]
	s_waitcnt vmcnt(2)
	ds_write_b128 v104, v[180:183]
	s_waitcnt vmcnt(1)
	ds_write_b128 v105, v[184:187]
	s_waitcnt vmcnt(0)
	ds_write_b128 v106, v[188:191]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[62:65], a[32:35], v172, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[54:57], a[32:35], v172, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[62:65], a[28:31], v172, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[54:57], a[28:31], v172, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[62:65], a[24:27], v173, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[132:135], v[54:57], a[24:27], v173, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[62:65], a[20:23], v173, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[136:139], v[54:57], a[20:23], v173, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[62:65], a[16:19], v174, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[148:151], v[54:57], a[16:19], v174, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[62:65], a[12:15], v174, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[152:155], v[54:57], a[12:15], v174, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[62:65], a[8:11], v175, v67 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[164:167], v[54:57], a[8:11], v175, v67 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[62:65], a[4:7], v175, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[168:171], v[54:57], a[4:7], v175, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_or_b32_e32 v106, 0xd000, v102
	buffer_load_dwordx4 v[62:65], v106, s[12:15], s11 offen nt
	buffer_load_dwordx4 v[54:57], v106, s[12:15], s11 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[30:33], a[80:83], v172, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[18:21], a[80:83], v172, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[30:33], a[72:75], v172, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[18:21], a[72:75], v172, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[30:33], a[68:71], v173, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[132:135], v[18:21], a[68:71], v173, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[30:33], a[60:63], v173, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[136:139], v[18:21], a[60:63], v173, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[30:33], a[56:59], v174, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[148:151], v[18:21], a[56:59], v174, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[30:33], a[48:51], v174, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[152:155], v[18:21], a[48:51], v174, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[30:33], a[40:43], v175, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[164:167], v[18:21], a[40:43], v175, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[30:33], a[36:39], v175, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[168:171], v[18:21], a[36:39], v175, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[30:33], v106, s[12:15], s17 offen nt
	buffer_load_dwordx4 v[18:21], v106, s[12:15], s17 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[34:37], a[108:111], v172, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[76:79], v[22:25], a[108:111], v172, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[34:37], a[100:103], v172, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[22:25], a[100:103], v172, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[34:37], a[96:99], v173, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[132:135], v[22:25], a[96:99], v173, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[34:37], a[92:95], v173, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[136:139], v[22:25], a[92:95], v173, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[34:37], a[84:87], v174, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[148:151], v[22:25], a[84:87], v174, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[34:37], a[64:67], v174, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[152:155], v[22:25], a[64:67], v174, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[34:37], a[52:55], v175, v84 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[164:167], v[22:25], a[52:55], v175, v84 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[34:37], a[44:47], v175, v84 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[168:171], v[22:25], a[44:47], v175, v84 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[34:37], v106, s[12:15], s16 offen nt
	buffer_load_dwordx4 v[22:25], v106, s[12:15], s16 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[38:41], a[124:127], v172, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[76:79], v[26:29], a[124:127], v172, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:75], v[38:41], a[120:123], v172, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[26:29], a[120:123], v172, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[38:41], a[116:119], v173, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[132:135], v[26:29], a[116:119], v173, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[38:41], a[112:115], v173, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[136:139], v[26:29], a[112:115], v173, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[38:41], a[104:107], v174, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[148:151], v[26:29], a[104:107], v174, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[38:41], a[88:91], v174, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[152:155], v[26:29], a[88:91], v174, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[38:41], a[76:79], v175, v84 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[164:167], v[26:29], a[76:79], v175, v84 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[38:41], a[0:3], v175, v84 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[168:171], v[26:29], a[0:3], v175, v84 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[38:41], v106, s[12:15], s10 offen nt
	buffer_load_dwordx4 v[26:29], v106, s[12:15], s10 offen offset:1024 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v164, v66, s[20:23], s8 offen offset:2560
	buffer_load_dword v165, v66, s[20:23], s9 offen offset:2560
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v[168:171], v94, s[24:27], 0 offen offset:3456
	buffer_load_dwordx4 v[172:175], v95, s[24:27], 0 offen offset:3456
	buffer_load_dwordx4 v[176:179], v96, s[24:27], 0 offen offset:3456
	buffer_load_dwordx4 v[180:183], v97, s[24:27], 0 offen offset:3456
	v_or_b32_e32 v67, 0x19900, v91
	v_or_b32_e32 v84, 0x1b500, v91
	v_or_b32_e32 v166, 0x1d100, v91
	v_or_b32_e32 v167, 0x1ed00, v91
	ds_read_b32 v67, v67
	ds_read_b32 v84, v84
	ds_read_b32 v166, v166
	ds_read_b32 v167, v167
	ds_read_b128 v[68:71], v93 offset:32768
	ds_read_b128 v[72:75], v93 offset:36864
	ds_read_b128 v[76:79], v92 offset:32768
	ds_read_b128 v[80:83], v92 offset:36864
	ds_read_b128 v[94:97], v93 offset:40960
	ds_read_b128 v[102:105], v93 offset:45056
	ds_read_b128 v[124:127], v92 offset:40960
	ds_read_b128 v[128:131], v92 offset:45056
	ds_read_b128 v[132:135], v93 offset:49152
	ds_read_b128 v[136:139], v93 offset:53248
	ds_read_b128 v[140:143], v92 offset:49152
	ds_read_b128 v[144:147], v92 offset:53248
	ds_read_b128 v[148:151], v93 offset:57344
	ds_read_b128 v[152:155], v93 offset:61440
	ds_read_b128 v[156:159], v92 offset:57344
	ds_read_b128 v[160:163], v92 offset:61440
	s_waitcnt vmcnt(3)
	ds_write_b128 v98, v[168:171]
	s_waitcnt vmcnt(2)
	ds_write_b128 v99, v[172:175]
	s_waitcnt vmcnt(1)
	ds_write_b128 v100, v[176:179]
	s_waitcnt vmcnt(0)
	ds_write_b128 v101, v[180:183]
	s_waitcnt lgkmcnt(14)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[68:71], v[58:61], a[32:35], v67, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[76:79], v[2:5], a[32:35], v67, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[72:75], v[58:61], a[28:31], v67, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[80:83], v[2:5], a[28:31], v67, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[94:97], v[58:61], a[24:27], v84, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(13)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[124:127], v[2:5], a[24:27], v84, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:105], v[58:61], a[20:23], v84, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(12)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[128:131], v[2:5], a[20:23], v84, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(11)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[132:135], v[58:61], a[16:19], v166, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(9)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[140:143], v[2:5], a[16:19], v166, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[136:139], v[58:61], a[12:15], v166, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(8)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[144:147], v[2:5], a[12:15], v166, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(7)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[148:151], v[58:61], a[8:11], v167, v85 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(5)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[156:159], v[2:5], a[8:11], v167, v85 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[152:155], v[58:61], a[4:7], v167, v85 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[160:163], v[2:5], a[4:7], v167, v85 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[58:61], v106, s[12:15], s11 offen offset:2048 nt
	buffer_load_dwordx4 v[2:5], v106, s[12:15], s11 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[68:71], v[42:45], a[80:83], v67, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[76:79], v[6:9], a[80:83], v67, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[72:75], v[42:45], a[72:75], v67, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[80:83], v[6:9], a[72:75], v67, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[94:97], v[42:45], a[68:71], v84, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[124:127], v[6:9], a[68:71], v84, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[42:45], a[60:63], v84, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[128:131], v[6:9], a[60:63], v84, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[132:135], v[42:45], a[56:59], v166, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[140:143], v[6:9], a[56:59], v166, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[136:139], v[42:45], a[48:51], v166, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[144:147], v[6:9], a[48:51], v166, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[148:151], v[42:45], a[40:43], v167, v85 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[156:159], v[6:9], a[40:43], v167, v85 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[152:155], v[42:45], a[36:39], v167, v85 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[160:163], v[6:9], a[36:39], v167, v85 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[42:45], v106, s[12:15], s17 offen offset:2048 nt
	buffer_load_dwordx4 v[6:9], v106, s[12:15], s17 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[68:71], v[46:49], a[108:111], v67, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[76:79], v[10:13], a[108:111], v67, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[72:75], v[46:49], a[100:103], v67, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[80:83], v[10:13], a[100:103], v67, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[46:49], a[96:99], v84, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[124:127], v[10:13], a[96:99], v84, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[102:105], v[46:49], a[92:95], v84, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[128:131], v[10:13], a[92:95], v84, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[132:135], v[46:49], a[84:87], v166, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[140:143], v[10:13], a[84:87], v166, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[136:139], v[46:49], a[64:67], v166, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[144:147], v[10:13], a[64:67], v166, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[148:151], v[46:49], a[52:55], v167, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[156:159], v[10:13], a[52:55], v167, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[152:155], v[46:49], a[44:47], v167, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[160:163], v[10:13], a[44:47], v167, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[46:49], v106, s[12:15], s16 offen offset:2048 nt
	buffer_load_dwordx4 v[10:13], v106, s[12:15], s16 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[68:71], v[50:53], a[124:127], v67, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[76:79], v[14:17], a[124:127], v67, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[72:75], v[50:53], a[120:123], v67, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[80:83], v[14:17], a[120:123], v67, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[94:97], v[50:53], a[116:119], v84, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[124:127], v[14:17], a[116:119], v84, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[102:105], v[50:53], a[112:115], v84, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[128:131], v[14:17], a[112:115], v84, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[132:135], v[50:53], a[104:107], v166, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[140:143], v[14:17], a[104:107], v166, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[136:139], v[50:53], a[88:91], v166, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[144:147], v[14:17], a[88:91], v166, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[148:151], v[50:53], a[76:79], v167, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[156:159], v[14:17], a[76:79], v167, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[152:155], v[50:53], a[0:3], v167, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[160:163], v[14:17], a[0:3], v167, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	buffer_load_dwordx4 v[50:53], v106, s[12:15], s10 offen offset:2048 nt
	buffer_load_dwordx4 v[14:17], v106, s[12:15], s10 offen offset:3072 nt
	; sched_barrier mask(0x00000000)
	buffer_load_dword v123, v66, s[20:23], s8 offen offset:2816
	buffer_load_dword v140, v66, s[20:23], s9 offen offset:2816
	v_or_b32_e32 v141, 0x1b600, v91
	v_or_b32_e32 v142, 0x1d200, v91
	v_or_b32_e32 v143, 0x1ee00, v91
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[66:69], v107
	ds_read_b128 v[70:73], v108
	ds_read_b128 v[74:77], v109
	ds_read_b128 v[78:81], v110
	ds_read_b128 v[82:85], v111
	ds_read_b128 v[94:97], v112
	ds_read_b128 v[98:101], v113
	ds_read_b128 v[102:105], v114
	ds_read_b128 v[106:109], v115
	ds_read_b128 v[110:113], v116
	ds_read_b128 v[114:117], v117
	ds_read_b128 v[124:127], v118
	ds_read_b128 v[128:131], v119
	ds_read_b128 v[132:135], v120
	ds_read_b128 v[118:121], v121
	ds_read_b128 v[136:139], v122
	v_or_b32_e32 v122, 0x19a00, v91
	ds_read_b32 v141, v141
	ds_read_b32 v142, v142
	ds_read_b32 v143, v143
	ds_read_b32 v122, v122
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[66:69], v[62:65], a[32:35], v122, v164 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_or_b32_e32 v90, 0x1ef00, v90
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[70:73], v[54:57], a[32:35], v122, v164 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[74:77], v[62:65], a[28:31], v122, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[78:81], v[54:57], a[28:31], v122, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[82:85], v[62:65], a[24:27], v141, v164 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[94:97], v[54:57], a[24:27], v141, v164 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[98:101], v[62:65], a[20:23], v141, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[102:105], v[54:57], a[20:23], v141, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[106:109], v[62:65], a[16:19], v142, v164 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[110:113], v[54:57], a[16:19], v142, v164 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[114:117], v[62:65], a[12:15], v142, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[124:127], v[54:57], a[12:15], v142, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[128:131], v[62:65], a[8:11], v143, v164 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[132:135], v[54:57], a[8:11], v143, v164 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[118:121], v[62:65], a[4:7], v143, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[136:139], v[54:57], a[4:7], v143, v164 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[66:69], v[30:33], a[80:83], v122, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[70:73], v[18:21], a[80:83], v122, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[74:77], v[30:33], a[72:75], v122, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[78:81], v[18:21], a[72:75], v122, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[82:85], v[30:33], a[68:71], v141, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[94:97], v[18:21], a[68:71], v141, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[98:101], v[30:33], a[60:63], v141, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[102:105], v[18:21], a[60:63], v141, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[106:109], v[30:33], a[56:59], v142, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[110:113], v[18:21], a[56:59], v142, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[114:117], v[30:33], a[48:51], v142, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[124:127], v[18:21], a[48:51], v142, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[128:131], v[30:33], a[40:43], v143, v164 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[132:135], v[18:21], a[40:43], v143, v164 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[118:121], v[30:33], a[36:39], v143, v164 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[136:139], v[18:21], a[36:39], v143, v164 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[66:69], v[34:37], a[108:111], v122, v165 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[70:73], v[22:25], a[108:111], v122, v165 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[74:77], v[34:37], a[100:103], v122, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[78:81], v[22:25], a[100:103], v122, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[82:85], v[34:37], a[96:99], v141, v165 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[94:97], v[22:25], a[96:99], v141, v165 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[98:101], v[34:37], a[92:95], v141, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[102:105], v[22:25], a[92:95], v141, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[106:109], v[34:37], a[84:87], v142, v165 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[110:113], v[22:25], a[84:87], v142, v165 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[114:117], v[34:37], a[64:67], v142, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[124:127], v[22:25], a[64:67], v142, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[128:131], v[34:37], a[52:55], v143, v165 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[132:135], v[22:25], a[52:55], v143, v165 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[118:121], v[34:37], a[44:47], v143, v165 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[136:139], v[22:25], a[44:47], v143, v165 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[66:69], v[38:41], a[124:127], v122, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[70:73], v[26:29], a[124:127], v122, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[74:77], v[38:41], a[120:123], v122, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[78:81], v[26:29], a[120:123], v122, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[82:85], v[38:41], a[116:119], v141, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[94:97], v[26:29], a[116:119], v141, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[98:101], v[38:41], a[112:115], v141, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[102:105], v[26:29], a[112:115], v141, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[106:109], v[38:41], a[104:107], v142, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[110:113], v[26:29], a[104:107], v142, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[114:117], v[38:41], a[88:91], v142, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[124:127], v[26:29], a[88:91], v142, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[128:131], v[38:41], a[76:79], v143, v165 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[132:135], v[26:29], a[76:79], v143, v165 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[118:121], v[38:41], a[0:3], v143, v165 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[136:139], v[26:29], a[0:3], v143, v165 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_barrier
	ds_read_b128 v[18:21], v93
	ds_read_b128 v[22:25], v93 offset:4096
	ds_read_b128 v[26:29], v92
	ds_read_b128 v[30:33], v92 offset:4096
	ds_read_b128 v[34:37], v93 offset:8192
	ds_read_b128 v[38:41], v93 offset:12288
	ds_read_b128 v[54:57], v92 offset:8192
	ds_read_b128 v[62:65], v92 offset:12288
	ds_read_b128 v[66:69], v93 offset:16384
	ds_read_b128 v[70:73], v93 offset:20480
	ds_read_b128 v[74:77], v92 offset:16384
	ds_read_b128 v[78:81], v92 offset:20480
	ds_read_b128 v[82:85], v93 offset:24576
	ds_read_b128 v[94:97], v93 offset:28672
	ds_read_b128 v[98:101], v92 offset:24576
	ds_read_b128 v[102:105], v92 offset:28672
	v_or_b32_e32 v92, 0x19b00, v91
	v_or_b32_e32 v93, 0x1b700, v91
	v_or_b32_e32 v91, 0x1d300, v91
	ds_read_b32 v93, v93
	ds_read_b32 v91, v91
	ds_read_b32 v90, v90
	ds_read_b32 v92, v92
	s_waitcnt vmcnt(1) lgkmcnt(0)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[18:21], v[58:61], a[32:35], v92, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], v[26:29], v[2:5], a[32:35], v92, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[22:25], v[58:61], a[28:31], v92, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[28:31], v[30:33], v[2:5], a[28:31], v92, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[34:37], v[58:61], a[24:27], v93, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[24:27], v[54:57], v[2:5], a[24:27], v93, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[38:41], v[58:61], a[20:23], v93, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[20:23], v[62:65], v[2:5], a[20:23], v93, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[66:69], v[58:61], a[16:19], v91, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[16:19], v[74:77], v[2:5], a[16:19], v91, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[70:73], v[58:61], a[12:15], v91, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[12:15], v[78:81], v[2:5], a[12:15], v91, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[82:85], v[58:61], a[8:11], v90, v123 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[8:11], v[98:101], v[2:5], a[8:11], v90, v123 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[94:97], v[58:61], a[4:7], v90, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[4:7], v[102:105], v[2:5], a[4:7], v90, v123 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[18:21], v[42:45], a[80:83], v92, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_lshlrev_b32_e32 v2, 2, v89
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[80:83], v[26:29], v[6:9], a[80:83], v92, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[22:25], v[42:45], a[72:75], v92, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_lshl_or_b32 v2, v88, 7, v2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[72:75], v[30:33], v[6:9], a[72:75], v92, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[34:37], v[42:45], a[68:71], v93, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_lshl_or_b32 v1, v1, 12, v2
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[68:71], v[54:57], v[6:9], a[68:71], v93, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[38:41], v[42:45], a[60:63], v93, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_or_b32_e32 v2, 0x10000, v1
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[60:63], v[62:65], v[6:9], a[60:63], v93, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[66:69], v[42:45], a[56:59], v91, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[56:59], v[74:77], v[6:9], a[56:59], v91, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[70:73], v[42:45], a[48:51], v91, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[48:51], v[78:81], v[6:9], a[48:51], v91, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[82:85], v[42:45], a[40:43], v90, v123 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[40:43], v[98:101], v[6:9], a[40:43], v90, v123 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[94:97], v[42:45], a[36:39], v90, v123 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[36:39], v[102:105], v[6:9], a[36:39], v90, v123 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_waitcnt vmcnt(0)
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[18:21], v[46:49], a[108:111], v92, v140 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_lshlrev_b32_e32 v8, 8, v86
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[108:111], v[26:29], v[10:13], a[108:111], v92, v140 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[22:25], v[46:49], a[100:103], v92, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[100:103], v[30:33], v[10:13], a[100:103], v92, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[34:37], v[46:49], a[96:99], v93, v140 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[96:99], v[54:57], v[10:13], a[96:99], v93, v140 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[38:41], v[46:49], a[92:95], v93, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[92:95], v[62:65], v[10:13], a[92:95], v93, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[66:69], v[46:49], a[84:87], v91, v140 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[84:87], v[74:77], v[10:13], a[84:87], v91, v140 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[70:73], v[46:49], a[64:67], v91, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[64:67], v[78:81], v[10:13], a[64:67], v91, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[82:85], v[46:49], a[52:55], v90, v140 op_sel:[0,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[52:55], v[98:101], v[10:13], a[52:55], v90, v140 op_sel:[0,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[94:97], v[46:49], a[44:47], v90, v140 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[44:47], v[102:105], v[10:13], a[44:47], v90, v140 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[18:21], v[50:53], a[124:127], v92, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[124:127], v[26:29], v[14:17], a[124:127], v92, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[22:25], v[50:53], a[120:123], v92, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[120:123], v[30:33], v[14:17], a[120:123], v92, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[34:37], v[50:53], a[116:119], v93, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_or_b32_e32 v30, 0x1000, v8
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[116:119], v[54:57], v[14:17], a[116:119], v93, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[38:41], v[50:53], a[112:115], v93, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_or_b32_e32 v34, 0x3000, v8
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[112:115], v[62:65], v[14:17], a[112:115], v93, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[66:69], v[50:53], a[104:107], v91, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	v_mov_b32_e32 v41, 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[104:107], v[74:77], v[14:17], a[104:107], v91, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[70:73], v[50:53], a[88:91], v91, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[88:91], v[78:81], v[14:17], a[88:91], v91, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[82:85], v[50:53], a[76:79], v90, v140 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[76:79], v[98:101], v[14:17], a[76:79], v90, v140 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[94:97], v[50:53], a[0:3], v90, v140 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	;;#ASMEND
	s_nop 0
	;;#ASMSTART
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[102:105], v[14:17], a[0:3], v90, v140 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	;;#ASMEND
	s_barrier
	ds_write_b32 v1, a32
	ds_write_b32 v1, a33 offset:1024
	ds_write_b32 v1, a34 offset:2048
	ds_write_b32 v1, a35 offset:3072
	ds_write_b32 v1, a80 offset:512
	ds_write_b32 v1, a81 offset:1536
	ds_write_b32 v1, a82 offset:2560
	ds_write_b32 v1, a83 offset:3584
	ds_write_b32 v1, a108 offset:64
	ds_write_b32 v1, a109 offset:1088
	ds_write_b32 v1, a110 offset:2112
	ds_write_b32 v1, a111 offset:3136
	ds_write_b32 v1, a124 offset:576
	ds_write_b32 v1, a125 offset:1600
	ds_write_b32 v1, a126 offset:2624
	ds_write_b32 v1, a127 offset:3648
	ds_write_b32 v1, a28 offset:16384
	ds_write_b32 v1, a29 offset:17408
	ds_write_b32 v1, a30 offset:18432
	ds_write_b32 v1, a31 offset:19456
	ds_write_b32 v1, a72 offset:16896
	ds_write_b32 v1, a73 offset:17920
	ds_write_b32 v1, a74 offset:18944
	ds_write_b32 v1, a75 offset:19968
	ds_write_b32 v1, a100 offset:16448
	ds_write_b32 v1, a101 offset:17472
	ds_write_b32 v1, a102 offset:18496
	ds_write_b32 v1, a103 offset:19520
	ds_write_b32 v1, a120 offset:16960
	ds_write_b32 v1, a121 offset:17984
	ds_write_b32 v1, a122 offset:19008
	ds_write_b32 v1, a123 offset:20032
	ds_write_b32 v1, a24 offset:32768
	ds_write_b32 v1, a25 offset:33792
	ds_write_b32 v1, a26 offset:34816
	ds_write_b32 v1, a27 offset:35840
	ds_write_b32 v1, a68 offset:33280
	ds_write_b32 v1, a69 offset:34304
	ds_write_b32 v1, a70 offset:35328
	ds_write_b32 v1, a71 offset:36352
	ds_write_b32 v1, a96 offset:32832
	ds_write_b32 v1, a97 offset:33856
	ds_write_b32 v1, a98 offset:34880
	ds_write_b32 v1, a99 offset:35904
	ds_write_b32 v1, a116 offset:33344
	ds_write_b32 v1, a117 offset:34368
	ds_write_b32 v1, a118 offset:35392
	ds_write_b32 v1, a119 offset:36416
	ds_write_b32 v1, a20 offset:49152
	ds_write_b32 v1, a21 offset:50176
	ds_write_b32 v1, a22 offset:51200
	ds_write_b32 v1, a23 offset:52224
	ds_write_b32 v1, a60 offset:49664
	ds_write_b32 v1, a61 offset:50688
	ds_write_b32 v1, a62 offset:51712
	ds_write_b32 v1, a63 offset:52736
	ds_write_b32 v1, a92 offset:49216
	ds_write_b32 v1, a93 offset:50240
	ds_write_b32 v1, a94 offset:51264
	ds_write_b32 v1, a95 offset:52288
	ds_write_b32 v1, a112 offset:49728
	ds_write_b32 v1, a113 offset:50752
	ds_write_b32 v1, a114 offset:51776
	ds_write_b32 v1, a115 offset:52800
	ds_write_b32 v2, a16
	v_or_b32_e32 v2, 0x10400, v1
	ds_write_b32 v2, a17
	v_or_b32_e32 v2, 0x10800, v1
	ds_write_b32 v2, a18
	v_or_b32_e32 v2, 0x10c00, v1
	ds_write_b32 v2, a19
	v_or_b32_e32 v2, 0x10200, v1
	ds_write_b32 v2, a56
	v_or_b32_e32 v2, 0x10600, v1
	ds_write_b32 v2, a57
	v_or_b32_e32 v2, 0x10a00, v1
	ds_write_b32 v2, a58
	v_or_b32_e32 v2, 0x10e00, v1
	ds_write_b32 v2, a59
	v_or_b32_e32 v2, 0x10040, v1
	ds_write_b32 v2, a84
	v_or_b32_e32 v2, 0x10440, v1
	ds_write_b32 v2, a85
	v_or_b32_e32 v2, 0x10840, v1
	ds_write_b32 v2, a86
	v_or_b32_e32 v2, 0x10c40, v1
	ds_write_b32 v2, a87
	v_or_b32_e32 v2, 0x10240, v1
	ds_write_b32 v2, a104
	v_or_b32_e32 v2, 0x10640, v1
	ds_write_b32 v2, a105
	v_or_b32_e32 v2, 0x10a40, v1
	ds_write_b32 v2, a106
	v_or_b32_e32 v2, 0x10e40, v1
	ds_write_b32 v2, a107
	v_or_b32_e32 v2, 0x14000, v1
	ds_write_b32 v2, a12
	v_or_b32_e32 v2, 0x14400, v1
	ds_write_b32 v2, a13
	v_or_b32_e32 v2, 0x14800, v1
	ds_write_b32 v2, a14
	v_or_b32_e32 v2, 0x14c00, v1
	ds_write_b32 v2, a15
	v_or_b32_e32 v2, 0x14200, v1
	ds_write_b32 v2, a48
	v_or_b32_e32 v2, 0x14600, v1
	ds_write_b32 v2, a49
	v_or_b32_e32 v2, 0x14a00, v1
	ds_write_b32 v2, a50
	v_or_b32_e32 v2, 0x14e00, v1
	ds_write_b32 v2, a51
	v_or_b32_e32 v2, 0x14040, v1
	ds_write_b32 v2, a64
	v_or_b32_e32 v2, 0x14440, v1
	ds_write_b32 v2, a65
	v_or_b32_e32 v2, 0x14840, v1
	ds_write_b32 v2, a66
	v_or_b32_e32 v2, 0x14c40, v1
	ds_write_b32 v2, a67
	v_or_b32_e32 v2, 0x14240, v1
	ds_write_b32 v2, a88
	v_or_b32_e32 v2, 0x14640, v1
	ds_write_b32 v2, a89
	v_or_b32_e32 v2, 0x14a40, v1
	ds_write_b32 v2, a90
	v_or_b32_e32 v2, 0x14e40, v1
	ds_write_b32 v2, a91
	v_or_b32_e32 v2, 0x18000, v1
	ds_write_b32 v2, a8
	v_or_b32_e32 v2, 0x18400, v1
	ds_write_b32 v2, a9
	v_or_b32_e32 v2, 0x18800, v1
	ds_write_b32 v2, a10
	v_or_b32_e32 v2, 0x18c00, v1
	ds_write_b32 v2, a11
	v_or_b32_e32 v2, 0x18200, v1
	ds_write_b32 v2, a40
	v_or_b32_e32 v2, 0x18600, v1
	ds_write_b32 v2, a41
	v_or_b32_e32 v2, 0x18a00, v1
	ds_write_b32 v2, a42
	v_or_b32_e32 v2, 0x18e00, v1
	ds_write_b32 v2, a43
	v_or_b32_e32 v2, 0x18040, v1
	ds_write_b32 v2, a52
	v_or_b32_e32 v2, 0x18440, v1
	ds_write_b32 v2, a53
	v_or_b32_e32 v2, 0x18840, v1
	ds_write_b32 v2, a54
	v_or_b32_e32 v2, 0x18c40, v1
	ds_write_b32 v2, a55
	v_or_b32_e32 v2, 0x18240, v1
	ds_write_b32 v2, a76
	v_or_b32_e32 v2, 0x18640, v1
	ds_write_b32 v2, a77
	v_or_b32_e32 v2, 0x18a40, v1
	ds_write_b32 v2, a78
	v_or_b32_e32 v2, 0x18e40, v1
	ds_write_b32 v2, a79
	v_or_b32_e32 v2, 0x1c000, v1
	ds_write_b32 v2, a4
	v_or_b32_e32 v2, 0x1c400, v1
	ds_write_b32 v2, a5
	v_or_b32_e32 v2, 0x1c800, v1
	ds_write_b32 v2, a6
	v_or_b32_e32 v2, 0x1cc00, v1
	ds_write_b32 v2, a7
	v_or_b32_e32 v2, 0x1c200, v1
	ds_write_b32 v2, a36
	v_or_b32_e32 v2, 0x1c600, v1
	ds_write_b32 v2, a37
	v_or_b32_e32 v2, 0x1ca00, v1
	ds_write_b32 v2, a38
	v_or_b32_e32 v2, 0x1ce00, v1
	ds_write_b32 v2, a39
	v_or_b32_e32 v2, 0x1c040, v1
	ds_write_b32 v2, a44
	v_or_b32_e32 v2, 0x1c440, v1
	ds_write_b32 v2, a45
	v_or_b32_e32 v2, 0x1c840, v1
	ds_write_b32 v2, a46
	v_or_b32_e32 v2, 0x1cc40, v1
	ds_write_b32 v2, a47
	v_or_b32_e32 v2, 0x1c240, v1
	ds_write_b32 v2, a0
	v_or_b32_e32 v2, 0x1c640, v1
	ds_write_b32 v2, a1
	v_or_b32_e32 v2, 0x1ca40, v1
	ds_write_b32 v2, a2
	v_or_b32_e32 v1, 0x1ce40, v1
	v_bfe_u32 v2, v0, 2, 2
	v_and_b32_e32 v0, 3, v0
	ds_write_b32 v1, a3
	v_lshlrev_b32_e32 v1, 3, v0
	v_lshl_or_b32 v4, v2, 5, v1
	v_or_b32_e32 v1, v4, v8
	v_lshlrev_b32_e32 v3, 2, v1
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 v[14:17], v3
	ds_read_b128 v[18:21], v3 offset:16
	ds_read2st64_b32 v[24:25], v3 offset0:2 offset1:66
	v_or_b32_e32 v11, 2, v4
	v_or_b32_e32 v12, 3, v4
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v1, 0xbfb8aa3b, v14
	v_exp_f32_e32 v1, v1
	v_or_b32_e32 v13, 1, v4
	v_or_b32_e32 v7, v13, v8
	v_or_b32_e32 v31, v13, v30
	v_add_f32_e32 v1, 1.0, v1
	v_rcp_f32_e32 v5, v1
	v_mul_f32_e32 v1, 0xbfb8aa3b, v15
	v_exp_f32_e32 v6, v1
	v_lshlrev_b32_e32 v31, 2, v31
	v_mul_f32_e32 v5, v14, v5
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v5, v24, v5
	v_add_f32_e32 v6, 1.0, v6
	v_rcp_f32_e32 v6, v6
	v_lshlrev_b32_e32 v14, 2, v7
	v_lshlrev_b32_e32 v2, 4, v2
	v_or_b32_e32 v35, v13, v34
	v_mul_f32_e32 v15, v15, v6
	v_or_b32_e32 v6, v11, v8
	v_lshlrev_b32_e32 v22, 2, v6
	v_mul_f32_e32 v6, 0xbfb8aa3b, v16
	v_exp_f32_e32 v23, v6
	v_or_b32_e32 v6, v12, v8
	v_lshlrev_b32_e32 v24, 2, v6
	v_mul_f32_e32 v6, 0xbfb8aa3b, v17
	v_exp_f32_e32 v26, v6
	v_or_b32_e32 v6, 4, v4
	v_or_b32_e32 v7, v6, v8
	v_lshlrev_b32_e32 v27, 2, v7
	v_or_b32_e32 v7, 5, v4
	v_or_b32_e32 v9, v7, v8
	v_lshlrev_b32_e32 v28, 2, v9
	v_or_b32_e32 v9, 6, v4
	v_or_b32_e32 v10, v9, v8
	v_lshlrev_b32_e32 v29, 2, v10
	v_or_b32_e32 v10, 7, v4
	v_or_b32_e32 v4, v10, v8
	v_lshlrev_b32_e32 v4, 2, v4
	ds_read_b32 v14, v14 offset:512
	ds_read_b32 v22, v22 offset:512
	ds_read_b32 v24, v24 offset:512
	ds_read_b32 v27, v27 offset:512
	ds_read_b32 v28, v28 offset:512
	ds_read_b32 v29, v29 offset:512
	ds_read_b32 v4, v4 offset:512
	ds_read_b32 v31, v31 offset:512
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v15, v14, v15
	v_add_f32_e32 v14, 1.0, v23
	v_rcp_f32_e32 v14, v14
	v_add_f32_e32 v23, 1.0, v26
	v_mul_f32_e32 v26, 0xbfb8aa3b, v18
	v_rcp_f32_e32 v23, v23
	v_exp_f32_e32 v26, v26
	v_mul_f32_e32 v14, v16, v14
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v16, v22, v14
	v_mul_f32_e32 v14, v17, v23
	v_add_f32_e32 v17, 1.0, v26
	v_rcp_f32_e32 v17, v17
	v_mul_f32_e32 v22, 0xbfb8aa3b, v19
	v_exp_f32_e32 v22, v22
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v23, v24, v14
	v_mul_f32_e32 v14, v18, v17
	v_mul_f32_e32 v18, 0xbfb8aa3b, v20
	s_waitcnt lgkmcnt(4)
	v_mul_f32_e32 v17, v27, v14
	v_add_f32_e32 v14, 1.0, v22
	v_exp_f32_e32 v18, v18
	v_mul_f32_e32 v22, 0xbfb8aa3b, v21
	v_rcp_f32_e32 v14, v14
	v_exp_f32_e32 v22, v22
	v_add_f32_e32 v18, 1.0, v18
	v_rcp_f32_e32 v18, v18
	v_mul_f32_e32 v14, v19, v14
	v_add_f32_e32 v19, 1.0, v22
	v_rcp_f32_e32 v19, v19
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v22, v28, v14
	v_mul_f32_e32 v14, v20, v18
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v18, v29, v14
	v_mul_f32_e32 v14, v21, v19
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v4, v4, v14
	v_maximum3_f32 v14, |v5|, |v15|, |v15|
	v_maximum3_f32 v14, v14, |v16|, |v23|
	v_maximum3_f32 v14, v14, |v17|, |v22|
	v_maximum3_f32 v14, v14, |v18|, |v4|
	v_mov_b32_e32 v24, 0
	v_or_b32_e32 v28, v7, v30
	v_mov_b32_dpp v19, v14 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v14, v14, v19, v19
	v_or_b32_e32 v29, v9, v30
	v_lshlrev_b32_e32 v28, 2, v28
	v_mov_b32_dpp v19, v14 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v14, v14, v19, v19
	v_add_u32_e32 v14, 0x200000, v14
	v_mul_f32_e32 v14, 0x3e800000, v14
	v_cvt_scalef32_pk_fp4_f32 v24, v5, v15, v14
	v_cvt_scalef32_pk_fp4_f32 v24, v16, v23, v14 op_sel:[0,0,1,0]
	v_lshlrev_b32_e32 v29, 2, v29
	v_cvt_scalef32_pk_fp4_f32 v24, v17, v22, v14 op_sel:[0,0,0,1]
	ds_read_b128 v[20:23], v3 offset:16400
	v_cvt_scalef32_pk_fp4_f32 v24, v18, v4, v14 op_sel:[0,0,1,1]
	ds_read_b128 v[16:19], v3 offset:16384
	v_lshlrev_b32_e32 v4, 2, v0
	v_lshl_or_b32 v15, s33, 6, v4
	v_or_b32_e32 v4, s30, v86
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v27, 0xbfb8aa3b, v20
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v5, 0xbfb8aa3b, v16
	v_exp_f32_e32 v26, v5
	v_mov_b32_e32 v5, s31
	v_lshlrev_b64 v[4:5], 8, v[4:5]
	v_or3_b32 v4, v15, v2, v4
	v_add_f32_e32 v15, 1.0, v26
	v_rcp_f32_e32 v15, v15
	v_mul_f32_e32 v26, 0xbfb8aa3b, v17
	v_exp_f32_e32 v26, v26
	v_lshl_add_u64 v[4:5], v[4:5], 0, s[18:19]
	v_mul_f32_e32 v15, v16, v15
	global_store_dword v[4:5], v24, off
	v_mul_f32_e32 v24, v25, v15
	v_mul_f32_e32 v15, 0xbfb8aa3b, v18
	v_exp_f32_e32 v15, v15
	v_add_f32_e32 v16, 1.0, v26
	v_mul_f32_e32 v25, 0xbfb8aa3b, v19
	v_rcp_f32_e32 v16, v16
	v_exp_f32_e32 v25, v25
	v_add_f32_e32 v15, 1.0, v15
	v_rcp_f32_e32 v15, v15
	v_mul_f32_e32 v16, v17, v16
	v_add_f32_e32 v25, 1.0, v25
	v_mul_f32_e32 v16, v31, v16
	v_rcp_f32_e32 v25, v25
	v_exp_f32_e32 v27, v27
	v_or_b32_e32 v31, 0x2000, v8
	v_or_b32_e32 v17, v11, v30
	v_mul_f32_e32 v15, v18, v15
	v_or_b32_e32 v18, v12, v30
	v_or_b32_e32 v26, v6, v30
	v_or_b32_e32 v30, v10, v30
	v_or_b32_e32 v32, v13, v31
	v_or_b32_e32 v33, v11, v31
	v_lshlrev_b32_e32 v17, 2, v17
	v_lshlrev_b32_e32 v18, 2, v18
	v_lshlrev_b32_e32 v26, 2, v26
	v_lshlrev_b32_e32 v30, 2, v30
	v_lshlrev_b32_e32 v32, 2, v32
	v_lshlrev_b32_e32 v33, 2, v33
	ds_read_b32 v17, v17 offset:512
	ds_read_b32 v18, v18 offset:512
	ds_read_b32 v26, v26 offset:512
	ds_read_b32 v28, v28 offset:512
	ds_read_b32 v29, v29 offset:512
	ds_read_b32 v30, v30 offset:512
	ds_read_b32 v32, v32 offset:512
	ds_read_b32 v33, v33 offset:512
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v17, v17, v15
	v_mul_f32_e32 v15, v19, v25
	v_add_f32_e32 v19, 1.0, v27
	v_rcp_f32_e32 v19, v19
	v_mul_f32_e32 v25, 0xbfb8aa3b, v21
	v_exp_f32_e32 v25, v25
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v18, v18, v15
	v_mul_f32_e32 v15, v20, v19
	v_mul_f32_e32 v20, 0xbfb8aa3b, v22
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v19, v26, v15
	v_add_f32_e32 v15, 1.0, v25
	v_exp_f32_e32 v20, v20
	v_mul_f32_e32 v25, 0xbfb8aa3b, v23
	v_rcp_f32_e32 v15, v15
	v_exp_f32_e32 v25, v25
	v_add_f32_e32 v20, 1.0, v20
	v_rcp_f32_e32 v20, v20
	v_mul_f32_e32 v15, v21, v15
	v_add_f32_e32 v21, 1.0, v25
	v_rcp_f32_e32 v21, v21
	s_waitcnt lgkmcnt(4)
	v_mul_f32_e32 v25, v28, v15
	v_mul_f32_e32 v15, v22, v20
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v20, v29, v15
	v_mul_f32_e32 v15, v23, v21
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v21, v30, v15
	v_maximum3_f32 v15, |v24|, |v16|, |v16|
	v_maximum3_f32 v15, v15, |v17|, |v18|
	v_maximum3_f32 v15, v15, |v19|, |v25|
	v_maximum3_f32 v15, v15, |v20|, |v21|
	v_add_co_u32_e32 v26, vcc, s4, v4
	s_nop 0
	v_mov_b32_dpp v22, v15 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v15, v15, v22, v22
	v_addc_co_u32_e32 v27, vcc, 0, v5, vcc
	s_nop 0
	v_mov_b32_dpp v22, v15 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v15, v15, v22, v22
	v_add_u32_e32 v15, 0x200000, v15
	v_mul_f32_e32 v15, 0x3e800000, v15
	v_mov_b32_e32 v22, 0
	v_cvt_scalef32_pk_fp4_f32 v22, v24, v16, v15
	v_cvt_scalef32_pk_fp4_f32 v22, v17, v18, v15 op_sel:[0,0,1,0]
	ds_read2st64_b32 v[28:29], v3 offset0:130 offset1:194
	v_cvt_scalef32_pk_fp4_f32 v22, v19, v25, v15 op_sel:[0,0,0,1]
	ds_read_b128 v[16:19], v3 offset:32768
	v_cvt_scalef32_pk_fp4_f32 v22, v20, v21, v15 op_sel:[0,0,1,1]
	global_store_dword v[26:27], v22, off offset:-4096
	ds_read_b128 v[20:23], v3 offset:32784
	v_or_b32_e32 v36, v11, v34
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v16
	v_exp_f32_e32 v24, v24
	v_mul_f32_e32 v25, 0xbfb8aa3b, v17
	v_exp_f32_e32 v25, v25
	v_mul_f32_e32 v30, 0xbfb8aa3b, v18
	v_add_f32_e32 v24, 1.0, v24
	v_rcp_f32_e32 v24, v24
	v_add_f32_e32 v25, 1.0, v25
	v_rcp_f32_e32 v25, v25
	v_exp_f32_e32 v30, v30
	v_mul_f32_e32 v16, v16, v24
	v_mul_f32_e32 v28, v28, v16
	v_mul_f32_e32 v16, v17, v25
	v_add_f32_e32 v17, 1.0, v30
	v_mul_f32_e32 v30, v32, v16
	v_mul_f32_e32 v16, 0xbfb8aa3b, v19
	v_exp_f32_e32 v16, v16
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v24, 0xbfb8aa3b, v20
	v_exp_f32_e32 v24, v24
	v_rcp_f32_e32 v17, v17
	v_add_f32_e32 v16, 1.0, v16
	v_rcp_f32_e32 v16, v16
	v_add_f32_e32 v24, 1.0, v24
	v_mul_f32_e32 v17, v18, v17
	v_rcp_f32_e32 v24, v24
	v_mul_f32_e32 v17, v33, v17
	v_or_b32_e32 v18, v12, v31
	v_mul_f32_e32 v16, v19, v16
	v_or_b32_e32 v19, v6, v31
	v_or_b32_e32 v25, v7, v31
	v_or_b32_e32 v33, v9, v31
	v_or_b32_e32 v31, v10, v31
	v_or_b32_e32 v37, v12, v34
	v_lshlrev_b32_e32 v18, 2, v18
	v_lshlrev_b32_e32 v19, 2, v19
	v_lshlrev_b32_e32 v25, 2, v25
	v_lshlrev_b32_e32 v33, 2, v33
	v_lshlrev_b32_e32 v31, 2, v31
	v_lshlrev_b32_e32 v35, 2, v35
	v_lshlrev_b32_e32 v36, 2, v36
	v_lshlrev_b32_e32 v37, 2, v37
	v_mul_f32_e32 v32, 0xbfb8aa3b, v21
	ds_read_b32 v18, v18 offset:512
	ds_read_b32 v19, v19 offset:512
	ds_read_b32 v25, v25 offset:512
	ds_read_b32 v33, v33 offset:512
	ds_read_b32 v31, v31 offset:512
	ds_read_b32 v35, v35 offset:512
	ds_read_b32 v36, v36 offset:512
	ds_read_b32 v37, v37 offset:512
	v_exp_f32_e32 v32, v32
	s_waitcnt lgkmcnt(7)
	v_mul_f32_e32 v38, v18, v16
	v_mul_f32_e32 v16, v20, v24
	v_mul_f32_e32 v18, 0xbfb8aa3b, v22
	s_waitcnt lgkmcnt(6)
	v_mul_f32_e32 v39, v19, v16
	v_exp_f32_e32 v18, v18
	v_mul_f32_e32 v19, 0xbfb8aa3b, v23
	v_exp_f32_e32 v19, v19
	v_add_f32_e32 v16, 1.0, v32
	v_rcp_f32_e32 v16, v16
	v_add_f32_e32 v18, 1.0, v18
	v_rcp_f32_e32 v18, v18
	v_add_f32_e32 v19, 1.0, v19
	v_rcp_f32_e32 v19, v19
	v_mul_f32_e32 v16, v21, v16
	s_waitcnt lgkmcnt(5)
	v_mul_f32_e32 v32, v25, v16
	v_mul_f32_e32 v16, v22, v18
	s_waitcnt lgkmcnt(4)
	v_mul_f32_e32 v33, v33, v16
	v_mul_f32_e32 v16, v23, v19
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v31, v31, v16
	v_maximum3_f32 v16, |v28|, |v30|, |v30|
	v_maximum3_f32 v16, v16, |v17|, |v38|
	v_maximum3_f32 v16, v16, |v39|, |v32|
	v_maximum3_f32 v16, v16, |v33|, |v31|
	v_mov_b32_e32 v1, s36
	s_nop 0
	v_mov_b32_dpp v18, v16 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v16, v16, v18, v18
	ds_read_b128 v[18:21], v3 offset:49152
	s_nop 0
	v_mov_b32_dpp v22, v16 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v16, v16, v22, v22
	ds_read_b128 v[22:25], v3 offset:49168
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v40, 0xbfb8aa3b, v18
	v_exp_f32_e32 v40, v40
	v_add_u32_e32 v16, 0x200000, v16
	v_mul_f32_e32 v16, 0x3e800000, v16
	v_cvt_scalef32_pk_fp4_f32 v41, v28, v30, v16
	v_cvt_scalef32_pk_fp4_f32 v41, v17, v38, v16 op_sel:[0,0,1,0]
	v_add_f32_e32 v17, 1.0, v40
	v_rcp_f32_e32 v17, v17
	v_cvt_scalef32_pk_fp4_f32 v41, v39, v32, v16 op_sel:[0,0,0,1]
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v30, 0xbfb8aa3b, v25
	v_cvt_scalef32_pk_fp4_f32 v41, v33, v31, v16 op_sel:[0,0,1,1]
	v_mul_f32_e32 v17, v18, v17
	v_mul_f32_e32 v18, 0xbfb8aa3b, v19
	global_store_dword v[26:27], v41, off
	v_exp_f32_e32 v18, v18
	v_mul_f32_e32 v26, 0xbfb8aa3b, v20
	v_exp_f32_e32 v26, v26
	v_mul_f32_e32 v27, v29, v17
	v_add_f32_e32 v17, 1.0, v18
	v_rcp_f32_e32 v17, v17
	v_add_f32_e32 v18, 1.0, v26
	v_mul_f32_e32 v26, 0xbfb8aa3b, v21
	v_rcp_f32_e32 v18, v18
	v_exp_f32_e32 v26, v26
	v_mul_f32_e32 v17, v19, v17
	v_mul_f32_e32 v19, v35, v17
	v_mul_f32_e32 v17, v20, v18
	v_add_f32_e32 v18, 1.0, v26
	v_rcp_f32_e32 v18, v18
	v_mul_f32_e32 v20, 0xbfb8aa3b, v22
	v_exp_f32_e32 v20, v20
	v_mul_f32_e32 v26, v36, v17
	v_mul_f32_e32 v17, v21, v18
	v_mul_f32_e32 v18, v37, v17
	v_add_f32_e32 v17, 1.0, v20
	v_rcp_f32_e32 v17, v17
	v_or_b32_e32 v20, v6, v34
	v_lshlrev_b32_e32 v20, 2, v20
	ds_read_b32 v20, v20 offset:512
	v_mul_f32_e32 v17, v22, v17
	v_mul_f32_e32 v22, 0xbfb8aa3b, v23
	v_exp_f32_e32 v22, v22
	v_exp_f32_e32 v30, v30
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v20, v20, v17
	v_or_b32_e32 v21, v7, v34
	v_add_f32_e32 v17, 1.0, v22
	v_mul_f32_e32 v22, 0xbfb8aa3b, v24
	v_exp_f32_e32 v22, v22
	v_rcp_f32_e32 v17, v17
	v_or_b32_e32 v28, v9, v34
	v_or_b32_e32 v29, v10, v34
	v_add_f32_e32 v22, 1.0, v22
	v_lshlrev_b32_e32 v21, 2, v21
	v_lshlrev_b32_e32 v28, 2, v28
	v_lshlrev_b32_e32 v29, 2, v29
	v_mul_f32_e32 v17, v23, v17
	v_rcp_f32_e32 v22, v22
	v_add_f32_e32 v23, 1.0, v30
	ds_read_b32 v21, v21 offset:512
	ds_read_b32 v28, v28 offset:512
	ds_read_b32 v29, v29 offset:512
	v_rcp_f32_e32 v23, v23
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v21, v21, v17
	v_mul_f32_e32 v17, v24, v22
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v22, v28, v17
	v_mul_f32_e32 v17, v25, v23
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v23, v29, v17
	v_maximum3_f32 v17, |v27|, |v19|, |v19|
	v_maximum3_f32 v17, v17, |v26|, |v18|
	v_maximum3_f32 v17, v17, |v20|, |v21|
	v_maximum3_f32 v17, v17, |v22|, |v23|
	v_or_b32_e32 v28, 0x4000, v8
	v_or_b32_e32 v25, v11, v28
	v_mov_b32_dpp v24, v17 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v17, v17, v24, v24
	v_lshlrev_b32_e32 v25, 2, v25
	v_or_b32_e32 v31, v6, v28
	v_mov_b32_dpp v24, v17 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v17, v17, v24, v24
	v_add_u32_e32 v17, 0x200000, v17
	v_mul_f32_e32 v17, 0x3e800000, v17
	v_mov_b32_e32 v24, 0
	v_cvt_scalef32_pk_fp4_f32 v24, v27, v19, v17
	v_cvt_scalef32_pk_fp4_f32 v24, v26, v18, v17 op_sel:[0,0,1,0]
	v_or_b32_e32 v18, 0x10000, v3
	v_cvt_scalef32_pk_fp4_f32 v24, v20, v21, v17 op_sel:[0,0,0,1]
	ds_read_b128 v[18:21], v18
	v_cvt_scalef32_pk_fp4_f32 v24, v22, v23, v17 op_sel:[0,0,1,1]
	v_or_b32_e32 v23, v13, v28
	v_or_b32_e32 v22, 0x10200, v3
	v_lshlrev_b32_e32 v23, 2, v23
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v26, 0xbfb8aa3b, v18
	v_exp_f32_e32 v29, v26
	v_add_co_u32_e32 v26, vcc, s5, v4
	v_mul_f32_e32 v30, 0xbfb8aa3b, v20
	s_nop 0
	v_addc_co_u32_e32 v27, vcc, 0, v5, vcc
	global_store_dword v[26:27], v24, off offset:-4096
	v_add_f32_e32 v24, 1.0, v29
	v_mul_f32_e32 v29, 0xbfb8aa3b, v19
	v_rcp_f32_e32 v24, v24
	v_exp_f32_e32 v29, v29
	v_exp_f32_e32 v30, v30
	ds_read_b32 v22, v22
	ds_read_b32 v23, v23 offset:512
	ds_read_b32 v25, v25 offset:512
	v_mul_f32_e32 v18, v18, v24
	v_add_f32_e32 v24, 1.0, v29
	v_rcp_f32_e32 v24, v24
	v_add_f32_e32 v29, 1.0, v30
	v_rcp_f32_e32 v29, v29
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v30, v22, v18
	v_mul_f32_e32 v18, v19, v24
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v19, v23, v18
	v_mul_f32_e32 v18, v20, v29
	v_or_b32_e32 v22, 0x10010, v3
	v_mul_f32_e32 v20, 0xbfb8aa3b, v21
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v29, v25, v18
	ds_read_b128 v[22:25], v22
	v_exp_f32_e32 v20, v20
	v_or_b32_e32 v18, v12, v28
	v_or_b32_e32 v32, v7, v28
	v_lshlrev_b32_e32 v18, 2, v18
	v_add_f32_e32 v20, 1.0, v20
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v33, 0xbfb8aa3b, v22
	v_rcp_f32_e32 v20, v20
	v_exp_f32_e32 v33, v33
	v_mul_f32_e32 v34, 0xbfb8aa3b, v23
	v_lshlrev_b32_e32 v31, 2, v31
	v_lshlrev_b32_e32 v32, 2, v32
	v_exp_f32_e32 v34, v34
	ds_read_b32 v18, v18 offset:512
	ds_read_b32 v31, v31 offset:512
	ds_read_b32 v32, v32 offset:512
	v_mul_f32_e32 v20, v21, v20
	v_add_f32_e32 v21, 1.0, v33
	v_rcp_f32_e32 v21, v21
	v_add_f32_e32 v33, 1.0, v34
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v34, v18, v20
	v_mul_f32_e32 v20, 0xbfb8aa3b, v24
	v_exp_f32_e32 v20, v20
	v_rcp_f32_e32 v33, v33
	v_mul_f32_e32 v18, v22, v21
	v_mul_f32_e32 v21, 0xbfb8aa3b, v25
	v_add_f32_e32 v20, 1.0, v20
	v_exp_f32_e32 v21, v21
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v31, v31, v18
	v_mul_f32_e32 v18, v23, v33
	v_rcp_f32_e32 v20, v20
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v32, v32, v18
	v_or_b32_e32 v18, v9, v28
	v_lshlrev_b32_e32 v18, 2, v18
	ds_read_b32 v18, v18 offset:512
	v_add_f32_e32 v21, 1.0, v21
	v_mul_f32_e32 v24, v24, v20
	v_or_b32_e32 v20, v10, v28
	v_rcp_f32_e32 v28, v21
	v_lshlrev_b32_e32 v20, 2, v20
	v_or_b32_e32 v21, 0x14000, v3
	v_or_b32_e32 v33, 0x14200, v3
	ds_read_b32 v35, v20 offset:512
	ds_read_b128 v[20:23], v21
	ds_read_b32 v33, v33
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v24, v18, v24
	v_mul_f32_e32 v18, v25, v28
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v25, v35, v18
	v_maximum3_f32 v18, |v30|, |v19|, |v19|
	v_maximum3_f32 v18, v18, |v29|, |v34|
	v_maximum3_f32 v18, v18, |v31|, |v32|
	v_maximum3_f32 v18, v18, |v24|, |v25|
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v35, 0xbfb8aa3b, v20
	v_exp_f32_e32 v35, v35
	v_mov_b32_dpp v28, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v18, v18, v28, v28
	s_nop 1
	v_mov_b32_dpp v28, v18 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v18, v18, v28, v28
	v_add_u32_e32 v18, 0x200000, v18
	v_mul_f32_e32 v18, 0x3e800000, v18
	v_mov_b32_e32 v28, 0
	v_cvt_scalef32_pk_fp4_f32 v28, v30, v19, v18
	v_add_f32_e32 v19, 1.0, v35
	v_rcp_f32_e32 v19, v19
	v_cvt_scalef32_pk_fp4_f32 v28, v29, v34, v18 op_sel:[0,0,1,0]
	v_mul_f32_e32 v19, v20, v19
	v_mul_f32_e32 v20, 0xbfb8aa3b, v21
	v_exp_f32_e32 v20, v20
	v_cvt_scalef32_pk_fp4_f32 v28, v31, v32, v18 op_sel:[0,0,0,1]
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v29, v33, v19
	v_cvt_scalef32_pk_fp4_f32 v28, v24, v25, v18 op_sel:[0,0,1,1]
	v_mul_f32_e32 v24, 0xbfb8aa3b, v22
	v_add_f32_e32 v20, 1.0, v20
	v_exp_f32_e32 v24, v24
	v_rcp_f32_e32 v20, v20
	global_store_dword v[26:27], v28, off
	v_or_b32_e32 v28, 0x5000, v8
	v_or_b32_e32 v19, v13, v28
	v_add_f32_e32 v24, 1.0, v24
	v_lshlrev_b32_e32 v19, 2, v19
	v_mul_f32_e32 v20, v21, v20
	v_or_b32_e32 v21, v11, v28
	v_rcp_f32_e32 v30, v24
	v_or_b32_e32 v24, v12, v28
	v_mul_f32_e32 v25, 0xbfb8aa3b, v23
	ds_read_b32 v19, v19 offset:512
	v_lshlrev_b32_e32 v21, 2, v21
	v_lshlrev_b32_e32 v24, 2, v24
	v_exp_f32_e32 v31, v25
	v_or_b32_e32 v25, 0x14010, v3
	ds_read_b32 v21, v21 offset:512
	ds_read_b32 v32, v24 offset:512
	ds_read_b128 v[24:27], v25
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v20, v19, v20
	v_mul_f32_e32 v19, v22, v30
	v_add_f32_e32 v22, 1.0, v31
	v_rcp_f32_e32 v22, v22
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v30, 0xbfb8aa3b, v24
	v_exp_f32_e32 v30, v30
	v_mul_f32_e32 v21, v21, v19
	v_mul_f32_e32 v19, v23, v22
	v_or_b32_e32 v23, v6, v28
	v_mul_f32_e32 v22, v32, v19
	v_add_f32_e32 v19, 1.0, v30
	v_lshlrev_b32_e32 v23, 2, v23
	v_rcp_f32_e32 v19, v19
	ds_read_b32 v23, v23 offset:512
	v_mul_f32_e32 v30, 0xbfb8aa3b, v25
	v_exp_f32_e32 v30, v30
	v_mul_f32_e32 v19, v24, v19
	v_mul_f32_e32 v32, 0xbfb8aa3b, v27
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v23, v23, v19
	v_add_f32_e32 v19, 1.0, v30
	v_mul_f32_e32 v30, 0xbfb8aa3b, v26
	v_rcp_f32_e32 v19, v19
	v_exp_f32_e32 v30, v30
	v_exp_f32_e32 v32, v32
	v_or_b32_e32 v24, v7, v28
	v_or_b32_e32 v31, v9, v28
	v_or_b32_e32 v28, v10, v28
	v_mul_f32_e32 v19, v25, v19
	v_add_f32_e32 v25, 1.0, v30
	v_lshlrev_b32_e32 v24, 2, v24
	v_lshlrev_b32_e32 v31, 2, v31
	v_lshlrev_b32_e32 v28, 2, v28
	v_rcp_f32_e32 v25, v25
	v_add_f32_e32 v30, 1.0, v32
	ds_read_b32 v24, v24 offset:512
	ds_read_b32 v31, v31 offset:512
	ds_read_b32 v28, v28 offset:512
	v_rcp_f32_e32 v30, v30
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v24, v24, v19
	v_mul_f32_e32 v19, v26, v25
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v25, v31, v19
	v_mul_f32_e32 v19, v27, v30
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v26, v28, v19
	v_maximum3_f32 v19, |v29|, |v20|, |v20|
	v_maximum3_f32 v19, v19, |v21|, |v22|
	v_maximum3_f32 v19, v19, |v23|, |v24|
	v_maximum3_f32 v19, v19, |v25|, |v26|
	v_or_b32_e32 v30, 0x6000, v8
	v_or_b32_e32 v8, 0x7000, v8
	v_mov_b32_dpp v27, v19 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v19, v19, v27, v27
	v_or_b32_e32 v33, v6, v30
	v_or_b32_e32 v34, v7, v30
	v_mov_b32_dpp v27, v19 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v19, v19, v27, v27
	v_add_u32_e32 v19, 0x200000, v19
	v_mul_f32_e32 v19, 0x3e800000, v19
	v_mov_b32_e32 v27, 0
	v_cvt_scalef32_pk_fp4_f32 v27, v29, v20, v19
	v_cvt_scalef32_pk_fp4_f32 v27, v21, v22, v19 op_sel:[0,0,1,0]
	v_or_b32_e32 v20, 0x18000, v3
	v_cvt_scalef32_pk_fp4_f32 v27, v23, v24, v19 op_sel:[0,0,0,1]
	ds_read_b128 v[20:23], v20
	v_cvt_scalef32_pk_fp4_f32 v27, v25, v26, v19 op_sel:[0,0,1,1]
	v_or_b32_e32 v25, v13, v30
	v_or_b32_e32 v13, v13, v8
	v_lshlrev_b32_e32 v13, 2, v13
	ds_read_b32 v13, v13 offset:512
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v28, 0xbfb8aa3b, v20
	v_exp_f32_e32 v31, v28
	v_add_co_u32_e32 v28, vcc, s6, v4
	v_mul_f32_e32 v32, 0xbfb8aa3b, v22
	s_nop 0
	v_addc_co_u32_e32 v29, vcc, 0, v5, vcc
	global_store_dword v[28:29], v27, off offset:-4096
	v_add_f32_e32 v27, 1.0, v31
	v_mul_f32_e32 v31, 0xbfb8aa3b, v21
	v_rcp_f32_e32 v27, v27
	v_exp_f32_e32 v31, v31
	v_exp_f32_e32 v32, v32
	v_or_b32_e32 v26, v11, v30
	v_mul_f32_e32 v20, v20, v27
	v_add_f32_e32 v27, 1.0, v31
	v_or_b32_e32 v24, 0x18200, v3
	v_lshlrev_b32_e32 v25, 2, v25
	v_lshlrev_b32_e32 v26, 2, v26
	v_rcp_f32_e32 v27, v27
	v_add_f32_e32 v31, 1.0, v32
	ds_read_b32 v24, v24
	ds_read_b32 v25, v25 offset:512
	ds_read_b32 v26, v26 offset:512
	v_rcp_f32_e32 v31, v31
	v_lshlrev_b32_e32 v33, 2, v33
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v32, v24, v20
	v_mul_f32_e32 v20, v21, v27
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v21, v25, v20
	v_mul_f32_e32 v20, v22, v31
	v_or_b32_e32 v24, 0x18010, v3
	v_mul_f32_e32 v22, 0xbfb8aa3b, v23
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v31, v26, v20
	ds_read_b128 v[24:27], v24
	v_exp_f32_e32 v22, v22
	v_or_b32_e32 v20, v12, v30
	v_lshlrev_b32_e32 v20, 2, v20
	v_lshlrev_b32_e32 v34, 2, v34
	v_add_f32_e32 v22, 1.0, v22
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v35, 0xbfb8aa3b, v24
	v_rcp_f32_e32 v22, v22
	v_exp_f32_e32 v35, v35
	v_mul_f32_e32 v36, 0xbfb8aa3b, v25
	v_exp_f32_e32 v36, v36
	ds_read_b32 v20, v20 offset:512
	ds_read_b32 v33, v33 offset:512
	ds_read_b32 v34, v34 offset:512
	v_mul_f32_e32 v22, v23, v22
	v_add_f32_e32 v23, 1.0, v35
	v_rcp_f32_e32 v23, v23
	v_add_f32_e32 v35, 1.0, v36
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v36, v20, v22
	v_mul_f32_e32 v22, 0xbfb8aa3b, v26
	v_exp_f32_e32 v22, v22
	v_rcp_f32_e32 v35, v35
	v_mul_f32_e32 v20, v24, v23
	v_mul_f32_e32 v23, 0xbfb8aa3b, v27
	v_add_f32_e32 v22, 1.0, v22
	v_exp_f32_e32 v23, v23
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v33, v33, v20
	v_mul_f32_e32 v20, v25, v35
	v_rcp_f32_e32 v22, v22
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v34, v34, v20
	v_or_b32_e32 v20, v9, v30
	v_lshlrev_b32_e32 v20, 2, v20
	ds_read_b32 v20, v20 offset:512
	v_add_f32_e32 v23, 1.0, v23
	v_mul_f32_e32 v26, v26, v22
	v_or_b32_e32 v22, v10, v30
	v_rcp_f32_e32 v30, v23
	v_lshlrev_b32_e32 v22, 2, v22
	v_or_b32_e32 v23, 0x1c000, v3
	v_or_b32_e32 v35, 0x1c200, v3
	ds_read_b32 v37, v22 offset:512
	ds_read_b128 v[22:25], v23
	ds_read_b32 v35, v35
	s_waitcnt lgkmcnt(3)
	v_mul_f32_e32 v26, v20, v26
	v_mul_f32_e32 v20, v27, v30
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v27, v37, v20
	v_maximum3_f32 v20, |v32|, |v21|, |v21|
	v_maximum3_f32 v20, v20, |v31|, |v36|
	v_maximum3_f32 v20, v20, |v33|, |v34|
	v_maximum3_f32 v20, v20, |v26|, |v27|
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v37, 0xbfb8aa3b, v22
	v_exp_f32_e32 v37, v37
	v_mov_b32_dpp v30, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v20, v20, v30, v30
	v_or_b32_e32 v11, v11, v8
	v_or_b32_e32 v12, v12, v8
	v_mov_b32_dpp v30, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v20, v20, v30, v30
	v_add_u32_e32 v20, 0x200000, v20
	v_mul_f32_e32 v20, 0x3e800000, v20
	v_mov_b32_e32 v30, 0
	v_cvt_scalef32_pk_fp4_f32 v30, v32, v21, v20
	v_add_f32_e32 v21, 1.0, v37
	v_rcp_f32_e32 v21, v21
	v_cvt_scalef32_pk_fp4_f32 v30, v31, v36, v20 op_sel:[0,0,1,0]
	v_lshlrev_b32_e32 v11, 2, v11
	v_cvt_scalef32_pk_fp4_f32 v30, v33, v34, v20 op_sel:[0,0,0,1]
	v_mul_f32_e32 v21, v22, v21
	v_mul_f32_e32 v22, 0xbfb8aa3b, v23
	v_exp_f32_e32 v22, v22
	v_cvt_scalef32_pk_fp4_f32 v30, v26, v27, v20 op_sel:[0,0,1,1]
	v_mul_f32_e32 v26, 0xbfb8aa3b, v24
	v_exp_f32_e32 v26, v26
	v_add_f32_e32 v22, 1.0, v22
	v_rcp_f32_e32 v22, v22
	v_lshlrev_b32_e32 v12, 2, v12
	global_store_dword v[28:29], v30, off
	v_or_b32_e32 v3, 0x1c010, v3
	v_mul_f32_e32 v22, v23, v22
	v_add_f32_e32 v23, 1.0, v26
	v_mul_f32_e32 v26, 0xbfb8aa3b, v25
	v_exp_f32_e32 v30, v26
	ds_read_b32 v11, v11 offset:512
	ds_read_b32 v12, v12 offset:512
	ds_read_b128 v[26:29], v3
	v_rcp_f32_e32 v23, v23
	v_or_b32_e32 v6, v6, v8
	v_lshlrev_b32_e32 v6, 2, v6
	ds_read_b32 v6, v6 offset:512
	v_mul_f32_e32 v13, v13, v22
	v_add_f32_e32 v22, 1.0, v30
	v_mul_f32_e32 v3, v24, v23
	v_rcp_f32_e32 v22, v22
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v23, 0xbfb8aa3b, v26
	v_exp_f32_e32 v23, v23
	v_mul_f32_e32 v11, v11, v3
	v_mul_f32_e32 v3, v25, v22
	v_mul_f32_e32 v12, v12, v3
	v_add_f32_e32 v3, 1.0, v23
	v_rcp_f32_e32 v3, v3
	v_mul_f32_e32 v22, 0xbfb8aa3b, v27
	v_exp_f32_e32 v22, v22
	v_or_b32_e32 v7, v7, v8
	v_mul_f32_e32 v3, v26, v3
	v_or_b32_e32 v9, v9, v8
	v_or_b32_e32 v8, v10, v8
	v_mul_f32_e32 v10, 0xbfb8aa3b, v28
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v6, v6, v3
	v_add_f32_e32 v3, 1.0, v22
	v_exp_f32_e32 v10, v10
	v_mul_f32_e32 v22, 0xbfb8aa3b, v29
	v_exp_f32_e32 v22, v22
	v_rcp_f32_e32 v3, v3
	v_add_f32_e32 v10, 1.0, v10
	v_lshlrev_b32_e32 v7, 2, v7
	v_lshlrev_b32_e32 v9, 2, v9
	v_lshlrev_b32_e32 v8, 2, v8
	v_rcp_f32_e32 v10, v10
	v_add_f32_e32 v22, 1.0, v22
	ds_read_b32 v7, v7 offset:512
	ds_read_b32 v9, v9 offset:512
	ds_read_b32 v8, v8 offset:512
	v_rcp_f32_e32 v22, v22
	v_mul_f32_e32 v3, v27, v3
	s_waitcnt lgkmcnt(2)
	v_mul_f32_e32 v7, v7, v3
	v_mul_f32_e32 v3, v28, v10
	v_mul_f32_e32 v21, v35, v21
	s_waitcnt lgkmcnt(1)
	v_mul_f32_e32 v9, v9, v3
	v_mul_f32_e32 v3, v29, v22
	s_waitcnt lgkmcnt(0)
	v_mul_f32_e32 v8, v8, v3
	v_maximum3_f32 v3, |v21|, |v13|, |v13|
	v_maximum3_f32 v3, v3, |v11|, |v12|
	v_maximum3_f32 v3, v3, |v6|, |v7|
	v_maximum3_f32 v3, v3, |v9|, |v8|
	v_add_co_u32_e32 v4, vcc, 0x7000, v4
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v3, v3, v10, v10
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	s_nop 0
	v_mov_b32_dpp v10, v3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_maximum3_f32 v3, v3, v10, v10
	v_add_u32_e32 v3, 0x200000, v3
	v_mul_f32_e32 v3, 0x3e800000, v3
	v_cvt_scalef32_pk_fp4_f32 v87, v21, v13, v3
	v_cvt_scalef32_pk_fp4_f32 v87, v11, v12, v3 op_sel:[0,0,1,0]
	v_cmp_eq_u64_e32 vcc, 0, v[0:1]
	v_cvt_scalef32_pk_fp4_f32 v87, v6, v7, v3 op_sel:[0,0,0,1]
	s_nop 0
	v_cvt_scalef32_pk_fp4_f32 v87, v9, v8, v3 op_sel:[0,0,1,1]
	global_store_dword v[4:5], v87, off
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB0_3
; %bb.2:
	s_load_dwordx2 s[4:5], s[0:1], 0x40
	s_lshl_b32 s0, s2, 1
	s_lshl_b32 s1, s33, 5
	s_lshl_b32 s2, s2, 7
	s_lshl_b32 s6, s3, 9
	s_or_b32 s3, s1, s2
	s_and_b32 s1, s1, 64
	v_lshl_or_b32 v9, s28, 9, v86
	v_lshrrev_b32_e32 v7, 23, v15
	v_lshrrev_b32_e32 v8, 23, v14
	v_or_b32_e32 v9, s1, v9
	v_min_u32_e32 v7, 0xfe, v7
	v_min_u32_e32 v8, 0xfe, v8
	s_and_b32 s0, s0, 2
	v_add_u32_e32 v9, v9, v2
	s_waitcnt lgkmcnt(0)
	s_and_b32 s5, s5, 0xffff
	s_mov_b32 s7, 0x27000
	v_lshl_or_b32 v7, v7, 8, v8
	v_lshl_or_b32 v8, v9, 2, s0
	s_and_b32 s3, s3, 0x3ffffe40
	s_or_b32 s1, s1, s2
	v_lshrrev_b32_e32 v5, 23, v17
	v_lshrrev_b32_e32 v6, 23, v16
	buffer_store_short v7, v8, s[4:7], 0 offen
	v_or_b32_e32 v7, s3, v2
	v_or_b32_e32 v2, s1, v2
	v_lshrrev_b32_e32 v0, 23, v3
	v_lshrrev_b32_e32 v1, 23, v20
	v_lshrrev_b32_e32 v3, 23, v19
	v_lshrrev_b32_e32 v4, 23, v18
	v_min_u32_e32 v5, 0xfe, v5
	v_min_u32_e32 v6, 0xfe, v6
	v_or_b32_e32 v2, 0x180, v2
	v_min_u32_e32 v0, 0xfe, v0
	v_min_u32_e32 v1, 0xfe, v1
	v_min_u32_e32 v3, 0xfe, v3
	v_min_u32_e32 v4, 0xfe, v4
	v_lshl_or_b32 v5, v5, 8, v6
	v_add_u32_e32 v6, v7, v86
	v_add_u32_e32 v2, v2, v86
	v_lshl_or_b32 v6, v6, 2, s0
	v_lshl_or_b32 v3, v3, 8, v4
	v_lshl_or_b32 v0, v0, 8, v1
	v_lshl_or_b32 v1, v2, 2, s0
	buffer_store_short v5, v6, s[4:7], 0 offen offset:512
	buffer_store_short v3, v6, s[4:7], 0 offen offset:1024
	buffer_store_short v0, v1, s[4:7], 0 offen
.LBB0_3:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel gemm1_kernel_0
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 88
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 332
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 200
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
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
	.text
.Lfunc_end0:
	.size	gemm1_kernel_0, .Lfunc_end0-gemm1_kernel_0
                                        ; -- End function
	.set gemm1_kernel_0.num_vgpr, 200
	.set gemm1_kernel_0.num_agpr, 132
	.set gemm1_kernel_0.numbered_sgpr, 52
	.set gemm1_kernel_0.num_named_barrier, 0
	.set gemm1_kernel_0.private_seg_size, 0
	.set gemm1_kernel_0.uses_vcc, 1
	.set gemm1_kernel_0.uses_flat_scratch, 0
	.set gemm1_kernel_0.has_dyn_sized_stack, 0
	.set gemm1_kernel_0.has_recursion, 0
	.set gemm1_kernel_0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 51940
; TotalNumSgprs: 58
; NumVgprs: 200
; NumAgprs: 132
; TotalNumVgprs: 332
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 131072 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 41
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 332
; AccumOffset: 200
; Occupancy: 1
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 49
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
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
  - .agpr_count:     132
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
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
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
      - .address_space:  global
        .offset:         64
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 88
    .max_flat_workgroup_size: 256
    .name:           gemm1_kernel_0
    .private_segment_fixed_size: 0
    .sgpr_count:     58
    .sgpr_spill_count: 0
    .symbol:         gemm1_kernel_0.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     332
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
