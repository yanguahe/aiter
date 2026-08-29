	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	; ---------------------------------------------------------------------
	; REDUCED-LDS (128 KiB) LOAD-ONLY BANDWIDTH VARIANT of
	;   my_code/f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps/
	;   f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps_loadonly.s
	;
	; Purpose: decide whether the load-only variant's measured read bandwidth
	; is capped by workgroup occupancy rather than by HBM.  gfx1250 shares one
	; 384 KiB SRAM per WGP between LDS and the vector cache and caps the LDS
	; partition at 320 KiB, so the 206848-byte declaration leaves room for
	; exactly one resident workgroup (2 * 206848 = 413696 > 327680) and thus
	; four waves of in-flight TDM requests per WGP.  This variant declares
	; 131072 bytes, so two workgroups (eight waves) fit inside the same
	; 256 KiB LDS partition that 206848 bytes already force.
	;
	; What changed relative to the load-only variant: ONLY the LDS destination
	; immediates of the TDM descriptors (s33, and the s95..s98 slot registers)
	; plus .amdhsa_group_segment_fixed_size and its metadata copy.  The
	; four-stage software pipeline now rotates over three physical LDS slots
	; per buffer instead of four, with stage 3 reusing the stage-1 slot:
	;   A  0x00000 0x02000 0x04000 0x02000   (slot 0x2000)
	;   SA 0x06000 0x06200 0x06400 0x06200   (slot 0x200)
	;   SB 0x06800 0x07000 0x07800 0x07000   (slot 0x800)
	;   B  0x08000 0x10000 0x18000 0x10000   (slot 0x8000)
	; Footprint 0x20000 = 131072 bytes, every base aligned to its slot size.
	;
	; Nothing in this kernel ever reads LDS, so having stages 1 and 3 write the
	; same bytes is unobservable.  Every new base is a multiple of 512 bytes,
	; hence of the 128-byte LDS bank period, so each stage keeps the bank
	; pattern it had before.
	;
	; Deliberately unchanged from the load-only variant: every global-side
	; descriptor field (s[34:35] base address, s[36:43] shape and strides),
	; every tensor_load_to_lds, every s_wait_tensorcnt throttle, every
	; workgroup and cluster barrier, the persistent X/Y scheduler, the batch-Z
	; pointer prologue, the 28 K256 bodies, the tile decomposition and the
	; 120-byte kernarg ABI.  The distinct global read volume is therefore the
	; same 2269446144 bytes at M=64, N=6144, K=7168, batch=96, which is what
	; makes the two timings comparable.
	;
	; Still stores nothing, so the runner's final comparison fails by
	; construction and it exits with status 3; only the timing row is
	; meaningful, and TB/s is reported over A+B+sA+sB only.
	; ---------------------------------------------------------------------
	.protected	f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps
	.globl	f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps
	.p2align	8
	.type	f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps,@function
f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps:
	s_version UC_VERSION_GFX12|UC_VERSION_W32_BIT              ; 000000001900: B0804009
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2  ; 000000001904: B980081A 00000002
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1  ; 00000000190C: B980009A 00000001
	; In cluster mode TTMP7[31:16] is the cluster-grid Z coordinate.  Since
	; clusterDim.z=1, it is also the batch index.  Compute every byte offset
	; modulo u64 as:
	;   lo = batch * stride.lo
	;   hi = mul_hi(batch, stride.lo) + batch * stride.hi
	; and add the resulting aligned SGPR pair to each base pointer.
	s_lshr_b32 s32, ttmp7, 16
	; D batch stride: preload s[22:23].
	s_mul_hi_u32 s35, s32, s22
	s_mul_i32 s34, s32, s23
	s_add_co_u32 s35, s35, s34
	s_mul_i32 s34, s32, s22
	s_add_nc_u64 s[2:3], s[2:3], s[34:35]
	; A batch stride: preload s[24:25].
	s_mul_hi_u32 s35, s32, s24
	s_mul_i32 s34, s32, s25
	s_add_co_u32 s35, s35, s34
	s_mul_i32 s34, s32, s24
	s_add_nc_u64 s[4:5], s[4:5], s[34:35]
	; B batch stride: preload s[26:27].
	s_mul_hi_u32 s35, s32, s26
	s_mul_i32 s34, s32, s27
	s_add_co_u32 s35, s35, s34
	s_mul_i32 s34, s32, s26
	s_add_nc_u64 s[6:7], s[6:7], s[34:35]
	; ScaleA batch stride: preload s[28:29].
	s_mul_hi_u32 s35, s32, s28
	s_mul_i32 s34, s32, s29
	s_add_co_u32 s35, s35, s34
	s_mul_i32 s34, s32, s28
	s_add_nc_u64 s[8:9], s[8:9], s[34:35]
	; ScaleB batch stride: preload s[30:31].
	s_mul_hi_u32 s35, s32, s30
	s_mul_i32 s34, s32, s31
	s_add_co_u32 s35, s35, s34
	s_mul_i32 s34, s32, s30
	s_add_nc_u64 s[10:11], s[10:11], s[34:35]
	s_mov_b32 s44, s2                                          ; 000000001914: BEAC0002
	s_mov_b32 s45, s3                                          ; 000000001918: BEAD0003
	s_bfe_u32 s22, ttmp8, 0x50019                              ; 00000000191C: 9316FF74 00050019
	s_cmp_eq_u32 s22, 0                                        ; 000000001924: BF068016
	s_cbranch_scc0 .Lbranch_00000000195c                                          ; 000000001928: BFA1000C <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x5c>
	s_getreg_b32 s24, hwreg(HW_REG_WAVE_MODE)                  ; 00000000192C: B898F801
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 24, 1), 1       ; 000000001930: B9800601 00000001
	s_getreg_b32 s24, hwreg(HW_REG_WAVE_MODE)                  ; 000000001938: B898F801
.Lbranch_00000000195c:
	s_wait_kmcnt 0x0                                           ; 00000000195C: BFC70000
	v_lshrrev_b32_e32 v1, 10, v0                               ; 000000001960: 3202008A
	v_lshrrev_b32_e32 v2, 10, v1                               ; 000000001964: 3204028A
	v_and_b32_e32 v0, 0x3ff, v0                                ; 000000001968: 360000FF 000003FF
	v_and_b32_e32 v1, 0x3ff, v1                                ; 000000001970: 360202FF 000003FF
	v_and_b32_e32 v2, 0x3ff, v2                                ; 000000001978: 360404FF 000003FF
	v_lshrrev_b32_e32 v3, 5, v0                                ; 000000001980: 32060085
	v_and_b32_e32 v0, 31, v0                                   ; 000000001984: 3600009F
	v_readfirstlane_b32 s22, v3                                ; 000000001988: 7E2C0503
	s_bfe_u32 s22, ttmp8, 0x50019                              ; 00000000198C: 9316FF74 00050019
	s_bfe_u32 s52, ttmp6, 0x40010                              ; 000000001994: 9334FF72 00040010
	s_bfe_u32 s51, ttmp6, 0x4000c                              ; 00000000199C: 9333FF72 0004000C
	s_bfe_u32 s50, ttmp6, 0x40004                              ; 0000000019A4: 9332FF72 00040004
	s_bfe_u32 s49, ttmp6, 0x40000                              ; 0000000019AC: 9331FF72 00040000
	s_add_co_i32 s52, s52, 1                                   ; 0000000019B4: 81348134
	s_add_co_i32 s51, s51, 1                                   ; 0000000019B8: 81338133
	s_and_b32 s24, ttmp7, 0xffff                               ; 0000000019BC: 8B18FF73 0000FFFF
	s_lshl_b32 s24, s24, s20                                   ; 0000000019C4: 84181418
	s_add_co_u32 s28, s24, ttmp9                               ; 0000000019C8: 801C7518
	s_mov_b32 s59, s19                                         ; 0000000019CC: BEBB0013
	s_mov_b32 s70, s19                                         ; 0000000019D0: BEC60013
	s_add_co_u32 s71, s19, 0x200                               ; 0000000019D4: 8047FF13 00000200
	s_ctz_i32_b32 s25, s51                                     ; 0000000019DC: BE990833
	s_add_co_i32 s25, s25, 8                                   ; N cluster span = nwg_x * 256
	s_lshl_b32 s26, 1, s25                                     ; 0000000019E4: 841A1981
	s_sub_co_u32 s26, s26, 1                                   ; 0000000019E8: 809A811A
	s_add_co_u32 s61, s18, s26                                 ; 0000000019EC: 803D1A12
	s_lshr_b32 s61, s61, s25                                   ; 0000000019F0: 853D193D
	s_ctz_i32_b32 s25, s52                                     ; 0000000019F4: BE990834
	s_add_co_i32 s25, s25, 6                                   ; M cluster span = nwg_y * 64
	s_lshl_b32 s26, 1, s25                                     ; 0000000019FC: 841A1981
	s_sub_co_u32 s26, s26, 1                                   ; 000000001A00: 809A811A
	s_add_co_u32 s29, s17, s26                                 ; 000000001A04: 801D1A11
	s_lshr_b32 s29, s29, s25                                   ; 000000001A08: 851D191D
	s_mul_i32 s29, s29, s61                                    ; 000000001A0C: 961D3D1D
	s_mov_b32 s94, 0                                          ; take the fast restart path from the first persistent task onward
	s_cmp_lt_u32 s28, s29                                      ; 000000001BFC: BF0A1D1C
	s_cbranch_scc0 .Lbranch_00000000ba60                                       ; 000000001C00: BFA12797 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0xa160>
	s_mov_b32 s24, s61                                         ; 000000001C04: BE98003D
	s_sub_co_u32 s25, s24, 1                                   ; 000000001C08: 80998118
	s_and_b32 s26, s24, s25                                    ; 000000001C0C: 8B1A1918
	s_cmp_eq_u32 s26, 0                                        ; 000000001C10: BF06801A
	s_cbranch_scc0 .Lbranch_000000001c28                                           ; 000000001C14: BFA10004 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x328>
	s_ctz_i32_b32 s26, s24                                     ; 000000001C18: BE9A0818
	s_lshr_b32 s48, s28, s26                                   ; 000000001C1C: 85301A1C
	s_and_b32 s23, s28, s25                                    ; 000000001C20: 8B17191C
	s_branch .Lbranch_000000001ca8                                                ; 000000001C24: BFA00020 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x3a8>
.Lbranch_000000001c28:
	v_cvt_f32_u32_e32 v4, s24                                  ; 000000001C28: 7E080C18
	s_sub_co_i32 s27, 0, s24                                   ; 000000001C2C: 819B1880
	v_rcp_iflag_f32_e32 v4, v4                                 ; 000000001C30: 7E085704
	s_nop 0                                                    ; 000000001C34: BF800000
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           ; 000000001C38: 100808FF 4F7FFFFE
	v_cvt_u32_f32_e32 v4, v4                                   ; 000000001C40: 7E080F04
	v_mul_lo_u32 v5, s27, v4                                   ; 000000001C44: D72C0005 0002081B
	v_mul_hi_u32 v5, v4, v5                                    ; 000000001C4C: D72D0005 00020B04
	v_add_nc_u32_e32 v4, v4, v5                                ; 000000001C54: 4A080B04
	v_mul_hi_u32 v4, s28, v4                                   ; 000000001C58: D72D0004 0002081C
	v_mul_lo_u32 v5, v4, s24                                   ; 000000001C60: D72C0005 00003104
	v_sub_nc_u32_e32 v7, s28, v5                               ; 000000001C68: 4C0E0A1C
	v_add_nc_u32_e32 v6, 1, v4                                 ; 000000001C6C: 4A0C0881
	v_cmp_le_u32_e32 vcc_lo, s24, v7                           ; 000000001C70: 7C960E18
	v_subrev_nc_u32_e32 v5, s24, v7                            ; 000000001C74: 4E0A0E18
	s_nop 0                                                    ; 000000001C78: BF800000
	v_cndmask_b32_e32 v4, v4, v6, vcc_lo                       ; 000000001C7C: 02080D04
	v_cndmask_b32_e32 v7, v7, v5, vcc_lo                       ; 000000001C80: 020E0B07
	v_add_nc_u32_e32 v5, 1, v4                                 ; 000000001C84: 4A0A0881
	v_cmp_le_u32_e32 vcc_lo, s24, v7                           ; 000000001C88: 7C960E18
	s_nop 1                                                    ; 000000001C8C: BF800001
	v_cndmask_b32_e32 v7, v4, v5, vcc_lo                       ; 000000001C90: 020E0B04
	s_nop 3                                                    ; 000000001C94: BF800003
	v_readfirstlane_b32 s48, v7                                ; 000000001C98: 7E600507
	s_nop 3                                                    ; 000000001C9C: BF800003
	s_mul_i32 s25, s48, s24                                    ; 000000001CA0: 96191830
	s_sub_co_u32 s23, s28, s25                                 ; 000000001CA4: 8097191C
.Lbranch_000000001ca8:
	s_mul_i32 s24, s48, s52                                    ; 000000001CA8: 96183430
	s_add_co_u32 s55, s24, s50                                 ; 000000001CAC: 80373218
	s_mul_i32 s24, s23, s51                                    ; 000000001CB0: 96183317
	s_add_co_u32 s54, s24, s49                                 ; 000000001CB4: 80363118
	s_add_co_i32 s24, s20, s21                                 ; 000000001CB8: 81181514
	s_lshl_b32 s24, 1, s24                                     ; 000000001CBC: 84181881
	s_add_co_u32 s28, s28, s24                                 ; 000000001CC0: 801C181C
	s_cmp_lt_u32 s28, s29                                      ; 000000001CC4: BF0A1D1C
	s_cselect_b32 s60, 0, 1                                    ; 000000001CC8: 983C8180
	s_cbranch_scc0 .Lbranch_000000001d98                                          ; 000000001CCC: BFA10032 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x498>
	s_mov_b32 s24, s61                                         ; 000000001CD0: BE98003D
	s_sub_co_u32 s25, s24, 1                                   ; 000000001CD4: 80998118
	s_and_b32 s26, s24, s25                                    ; 000000001CD8: 8B1A1918
	s_cmp_eq_u32 s26, 0                                        ; 000000001CDC: BF06801A
	s_cbranch_scc0 .Lbranch_000000001d04                                           ; 000000001CE0: BFA10008 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x404>
	s_ctz_i32_b32 s26, s24                                     ; 000000001CE4: BE9A0818
	s_lshr_b32 s27, s28, s26                                   ; 000000001CE8: 851B1A1C
	s_and_b32 s24, s28, s25                                    ; 000000001CEC: 8B18191C
	s_mul_i32 s25, s27, s52                                    ; 000000001CF0: 9619341B
	s_add_co_u32 s69, s25, s50                                 ; 000000001CF4: 80453219
	s_mul_i32 s25, s24, s51                                    ; 000000001CF8: 96193318
	s_add_co_u32 s68, s25, s49                                 ; 000000001CFC: 80443119
	s_branch .Lbranch_000000001da0                                                ; 000000001D00: BFA00027 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4a0>
.Lbranch_000000001d04:
	v_cvt_f32_u32_e32 v4, s24                                  ; 000000001D04: 7E080C18
	s_sub_co_i32 s26, 0, s24                                   ; 000000001D08: 819A1880
	v_rcp_iflag_f32_e32 v4, v4                                 ; 000000001D0C: 7E085704
	s_nop 0                                                    ; 000000001D10: BF800000
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           ; 000000001D14: 100808FF 4F7FFFFE
	v_cvt_u32_f32_e32 v4, v4                                   ; 000000001D1C: 7E080F04
	v_mul_lo_u32 v5, s26, v4                                   ; 000000001D20: D72C0005 0002081A
	v_mul_hi_u32 v5, v4, v5                                    ; 000000001D28: D72D0005 00020B04
	v_add_nc_u32_e32 v4, v4, v5                                ; 000000001D30: 4A080B04
	v_mul_hi_u32 v4, s28, v4                                   ; 000000001D34: D72D0004 0002081C
	v_mul_lo_u32 v5, v4, s24                                   ; 000000001D3C: D72C0005 00003104
	v_sub_nc_u32_e32 v7, s28, v5                               ; 000000001D44: 4C0E0A1C
	v_add_nc_u32_e32 v6, 1, v4                                 ; 000000001D48: 4A0C0881
	v_cmp_le_u32_e32 vcc_lo, s24, v7                           ; 000000001D4C: 7C960E18
	v_subrev_nc_u32_e32 v5, s24, v7                            ; 000000001D50: 4E0A0E18
	s_nop 0                                                    ; 000000001D54: BF800000
	v_cndmask_b32_e32 v4, v4, v6, vcc_lo                       ; 000000001D58: 02080D04
	v_cndmask_b32_e32 v7, v7, v5, vcc_lo                       ; 000000001D5C: 020E0B07
	v_add_nc_u32_e32 v5, 1, v4                                 ; 000000001D60: 4A0A0881
	v_cmp_le_u32_e32 vcc_lo, s24, v7                           ; 000000001D64: 7C960E18
	s_nop 1                                                    ; 000000001D68: BF800001
	v_cndmask_b32_e32 v7, v4, v5, vcc_lo                       ; 000000001D6C: 020E0B04
	s_nop 3                                                    ; 000000001D70: BF800003
	v_readfirstlane_b32 s27, v7                                ; 000000001D74: 7E360507
	s_nop 3                                                    ; 000000001D78: BF800003
	s_mul_i32 s25, s27, s24                                    ; 000000001D7C: 9619181B
	s_sub_co_u32 s24, s28, s25                                 ; 000000001D80: 8098191C
	s_mul_i32 s25, s27, s52                                    ; 000000001D84: 9619341B
	s_add_co_u32 s69, s25, s50                                 ; 000000001D88: 80453219
	s_mul_i32 s25, s24, s51                                    ; 000000001D8C: 96193318
	s_add_co_u32 s68, s25, s49                                 ; 000000001D90: 80443119
	s_branch .Lbranch_000000001da0                                                 ; 000000001D94: BFA00002 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4a0>
.Lbranch_000000001d98:
	s_mov_b32 s68, s54                                         ; 000000001D98: BEC40036
	s_mov_b32 s69, s55                                         ; 000000001D9C: BEC50037
.Lbranch_000000001da0:
	s_mul_i32 s24, s55, 0x40
	s_mul_i32 s24, s24, s13                                    ; 000000001DA8: 96180D18
	s_add_co_u32 s72, s4, s24                                  ; 000000001DAC: 80481804
	s_add_co_ci_u32 s73, 0, s5                                 ; 000000001DB0: 82490580
	s_mul_i32 s24, s55, 0x40
	s_mul_i32 s24, s24, s15                                    ; 000000001DBC: 96180F18
	s_add_co_u32 s76, s8, s24                                  ; 000000001DC0: 804C1808
	s_add_co_ci_u32 s77, 0, s9                                 ; 000000001DC4: 824D0980
	s_mul_i32 s24, s54, 0x100
	s_mul_i32 s24, s24, s14                                    ; 000000001DD0: 96180E18
	s_add_co_u32 s74, s6, s24                                  ; 000000001DD4: 804A1806
	s_add_co_ci_u32 s75, 0, s7                                 ; 000000001DD8: 824B0780
	s_mul_i32 s24, s54, 0x100
	s_mul_i32 s24, s24, s16                                    ; 000000001DE4: 96181018
	s_add_co_u32 s78, s10, s24                                 ; 000000001DE8: 804E180A
	s_add_co_ci_u32 s79, 0, s11                                ; 000000001DEC: 824F0B80
	; D origin = ((Mtile*64)*N + Ntile*256 + wave_id*64) * sizeof(bf16).
	s_mul_i32 s24, 0x100, s54
	s_lshl_b32 s24, s24, 1                                     ; 000000001DF8: 84188118
	s_mul_i32 s25, 0x40, s55
	s_mul_i32 s25, s25, s12                                    ; 000000001E04: 96190C19
	s_add_co_u32 s26, s25, s24                                 ; 000000001E08: 801A1819
	s_mul_i32 s24, s22, 0x40
	s_lshl_b32 s24, s24, 1
	s_add_co_u32 s26, s26, s24                                 ; 000000001E30: 801A181A
	s_add_co_u32 s44, s44, s26                                 ; 000000001E34: 802C1A2C
	s_add_co_ci_u32 s45, 0, s45                                ; 000000001E38: 822D2D80
	s_mov_b32 s58, 0                                           ; 000000001E3C: BEBA0080
	s_mov_b32 s70, s19                                         ; 000000001E40: BEC60013
	s_mov_b32 s24, 0                                           ; 000000001E44: BE980080
	s_mov_b32 s25, 0                                           ; 000000001E48: BE990080
	s_mov_b32 s26, 0                                           ; 000000001E4C: BE9A0080
	s_cmp_eq_u32 s22, 0                                        ; 000000001E50: BF068016
	s_cselect_b32 s24, s72, s24                                ; 000000001E54: 98181848
	s_cselect_b32 s25, s73, s25                                ; 000000001E58: 98191949
	s_cselect_b32 s26, s13, s26                                ; 000000001E5C: 981A1A0D
	s_cmp_eq_u32 s22, 1                                        ; 000000001E60: BF068116
	s_cselect_b32 s24, s74, s24                                ; 000000001E64: 9818184A
	s_cselect_b32 s25, s75, s25                                ; 000000001E68: 9819194B
	s_cselect_b32 s26, s14, s26                                ; 000000001E6C: 981A1A0E
	s_cmp_eq_u32 s22, 2                                        ; 000000001E70: BF068216
	s_cselect_b32 s24, s76, s24                                ; 000000001E74: 9818184C
	s_cselect_b32 s25, s77, s25                                ; 000000001E78: 9819194D
	s_cselect_b32 s26, s15, s26                                ; 000000001E7C: 981A1A0F
	s_cmp_eq_u32 s22, 3                                        ; 000000001E80: BF068316
	s_cselect_b32 s24, s78, s24                                ; 000000001E84: 9818184E
	s_cselect_b32 s25, s79, s25                                ; 000000001E88: 9819194F
	s_cselect_b32 s26, s16, s26                                ; 000000001E8C: 981A1A10
	s_and_b32 s23, s22, 1
	s_cmp_eq_u32 s23, 0
	s_cselect_b32 s23, 4, 16
	s_lshl_b32 s27, s23, 4
	s_mul_i32 s27, s27, s26
	s_sub_co_i32 s27, s27, 1                                   ; 000000001E98: 819B811B
	v_mov_b32_e32 v5, 0                                        ; 000000001E9C: 7E0A0280
	v_mov_b32_e32 v7, 0                                        ; 000000001EA0: 7E0E0280
	v_and_b32_e64 v6, v0, 1                                    ; 000000001EA4: D51B0006 00010300
	v_mul_u32_u24_e64 v6, v6, 0x200000                         ; 000000001EAC: D50B0006 0001FF06 00200000
	v_and_b32_e64 v4, v0, 15                                   ; 000000001EB8: D51B0004 00011F00
	v_mul_u32_u24_e64 v4, v4, s23
	v_mul_lo_u32 v4, v4, s26                                   ; 000000001EC8: D72C0004 00003504
	s_mov_b32 exec_lo, 0xffff                                  ; 000000001ED0: BEFE00FF 0000FFFF
	global_prefetch_b8 v4, s[24:25]                            ; 000000001ED8: EE174018 00000000 00000004
	s_mov_b32 exec_lo, -1                                      ; 000000001EE4: BEFE00C1
	s_cmp_eq_u32 s22, 0                                        ; 000000001EE8: BF068016
	s_cbranch_scc1 .Lbranch_000000001f08                                           ; 000000001EEC: BFA20006 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x608>
	s_cmp_eq_u32 s22, 1                                        ; 000000001EF0: BF068116
	s_cbranch_scc1 .Lbranch_000000002b48                                         ; 000000001EF4: BFA20314 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x1248>
	s_cmp_eq_u32 s22, 2                                        ; 000000001EF8: BF068216
	s_cbranch_scc1 .Lbranch_0000000037ac                                        ; 000000001EFC: BFA2062B <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x1eac>
	s_cmp_eq_u32 s22, 3                                        ; 000000001F00: BF068316
	s_cbranch_scc1 .Lbranch_0000000043f8                                        ; 000000001F04: BFA2093C <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x2af8>
.Lbranch_000000001f08:
	s_mov_b32 s95, 0                                           ; 000000001F08: BEDF0080
	s_mov_b32 s96, 0x2000
	s_mov_b32 s97, 0x4000
	s_mov_b32 s98, 0x2000
	s_mov_b32 s32, 1                                           ; 000000001F24: BEA00081
	s_mov_b32 s33, 0                                           ; 000000001F28: BEA10080
	s_mov_b32 s34, 0                                           ; 000000001F2C: BEA20080
	s_mov_b32 s35, 0x80000000                                  ; 000000001F30: BEA300FF 80000000
	s_mov_b32 s33, 0                                           ; 000000001F38: BEA10080
	s_mov_b32 s34, s72                                         ; 000000001F3C: BEA20048
	s_and_b32 s73, s73, 0x1ffffff                              ; 000000001F40: 8B49FF49 01FFFFFF
	s_and_b32 s35, s35, 0xfe000000                             ; 000000001F48: 8B23FF23 FE000000
	s_or_b32 s35, s73, s35                                     ; 000000001F50: 8C232349
	s_mov_b32 s36, 0                                           ; 000000001F54: BEA40080
	s_mov_b32 s37, 0                                           ; 000000001F58: BEA50080
	s_mov_b32 s38, 0                                           ; 000000001F5C: BEA60080
	s_mov_b32 s39, 0                                           ; 000000001F60: BEA70080
	s_mov_b32 s40, 0                                           ; 000000001F64: BEA80080
	s_mov_b32 s41, 0                                           ; 000000001F68: BEA90080
	s_mov_b32 s42, 0                                           ; 000000001F6C: BEAA0080
	s_mov_b32 s43, 0                                           ; 000000001F70: BEAB0080
	s_lshl_b32 s26, s13, 4                                     ; 000000001F74: 841A840D
	s_and_b32 s37, s37, 0xffff                                 ; 000000001F78: 8B25FF25 0000FFFF
	s_and_b32 s38, s38, 0xffff0000                             ; 000000001F80: 8B26FF26 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000001F88: 8418901A
	s_or_b32 s37, s24, s37                                     ; 000000001F8C: 8C252518
	s_lshr_b32 s24, s26, 16                                    ; 000000001F90: 8518901A
	s_or_b32 s38, s24, s38                                     ; 000000001F94: 8C262618
	s_mul_i32 s24, s55, 0x40
	s_sub_co_u32 s26, s17, s24                                 ; 000000001FA0: 809A1811
	s_lshr_b32 s26, s26, 4                                     ; 000000001FA4: 851A841A
	s_and_b32 s38, s38, 0xffff                                 ; 000000001FA8: 8B26FF26 0000FFFF
	s_and_b32 s39, s39, 0xffff0000                             ; 000000001FB0: 8B27FF27 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000001FB8: 8418901A
	s_or_b32 s38, s24, s38                                     ; 000000001FBC: 8C262618
	s_lshr_b32 s24, s26, 16                                    ; 000000001FC0: 8518901A
	s_or_b32 s39, s24, s39                                     ; 000000001FC4: 8C272718
	s_and_b32 s39, s39, 0xffff                                 ; 000000001FC8: 8B27FF27 0000FFFF
	s_or_b32 s39, s39, 0x8000000                               ; 000000001FD0: 8C27FF27 08000000
	s_and_b32 s40, s40, 0xffff0000                             ; 000000001FD8: 8B28FF28 FFFF0000
	s_or_b32 s40, s40, 4                                      ; preshuffle blocks: 4 * (16 rows * 0x80 bytes)
	s_lshl_b32 s24, s13, 4                                     ; 000000001FE4: 8418840D
	s_mov_b32 s25, 0                                           ; 000000001FE8: BE990080
	s_mov_b32 s41, s24                                         ; 000000001FEC: BEA90018
	s_and_b32 s25, s25, 0xffff                                 ; 000000001FF0: 8B19FF19 0000FFFF
	s_and_b32 s42, s42, 0xffff0000                             ; 000000001FF8: 8B2AFF2A FFFF0000
	s_or_b32 s42, s42, s25                                     ; 000000002000: 8C2A192A
	s_bitset0_b32 s36, 20                                      ; 000000002004: BEA41094
	s_mov_b32 s53, 0xf                                         ; A multicast: all four N WGs in the M row
	s_and_b32 s36, s36, 0xffff0000                             ; 00000000202C: 8B24FF24 FFFF0000
	s_or_b32 s36, s53, s36                                     ; 00000000203C: 8C242435
	s_bitset1_b32 s36, 21                                      ; 000000002040: BEA41295
	s_mov_b32 s56, 0x800                                       ; 000000002044: BEB800FF 00000800
	s_mov_b32 s57, 0                                           ; 00000000204C: BEB90080
	s_mul_i32 s24, s69, 0x40
	s_mul_hi_u32 s63, s24, s13                                 ; 000000002058: 96BF0D18
	s_mul_i32 s24, s24, s13                                    ; 00000000205C: 96180D18
	s_add_co_u32 s62, s4, s24                                  ; 000000002060: 803E1804
	s_add_co_ci_u32 s63, s63, s5                               ; 000000002064: 823F053F
	s_and_b32 s63, s63, 0x1ffffff                              ; 000000002068: 8B3FFF3F 01FFFFFF
	s_or_b32 s63, s63, 0x80000000                              ; 000000002070: 8C3FFF3F 80000000
	s_mov_b32 s64, s36                                         ; 000000002078: BEC00024
	s_mov_b32 s65, s37                                         ; 00000000207C: BEC10025
	s_mul_i32 s27, s69, 0x40
	s_sub_co_u32 s27, s17, s27                                 ; 000000002088: 809B1B11
	s_lshr_b32 s27, s27, 4                                     ; 00000000208C: 851B841B
	s_lshl_b32 s26, s27, 16                                    ; 000000002090: 841A901B
	s_and_b32 s66, s38, 0xffff                                 ; 000000002094: 8B42FF26 0000FFFF
	s_or_b32 s66, s66, s26                                     ; 00000000209C: 8C421A42
	s_lshr_b32 s26, s27, 16                                    ; 0000000020A0: 851A901B
	s_or_b32 s67, s26, 0x8000000                               ; 0000000020A4: 8C43FF1A 08000000
	s_barrier_signal -1                                        ; 0000000020AC: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000020B0: BF94FFFF
	s_barrier_signal -3                                        ; 0000000020B4: BE804EC3
	s_barrier_wait 0xfffd                                      ; 0000000020B8: BF94FFFD
	s_mov_b32 s33, 0                                           ; 0000000020BC: BEA10080
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 0000000020C0: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x100                               ; 0000000020CC: 8018FF3A 00000100
	s_cmp_lt_u32 s24, s70                                      ; 0000000020D4: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000020D8: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000020DC: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000020E0: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000020E4: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000020E8: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000020EC: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000020F0: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000020F4: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000020F8: 98278027
	s_barrier_signal -1                                        ; 0000000020FC: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000002100: BF94FFFF
	s_mov_b32 s33, 0x2000
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 00000000210C: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x200                               ; 000000002118: 8018FF3A 00000200
	s_cmp_lt_u32 s24, s70                                      ; 000000002120: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000002124: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000002128: 98244024
	s_cselect_b32 s37, s37, s65                                ; 00000000212C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000002130: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000002134: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000002138: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 00000000213C: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000002140: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000002144: 98278027
	s_barrier_signal -1                                        ; 000000002148: BE804EC1
	s_barrier_wait 0xffff                                      ; 00000000214C: BF94FFFF
	s_mov_b32 s33, 0x4000
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000002158: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000002164: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 00000000216C: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000002170: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000002174: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000002178: 98254125
	s_cselect_b32 s38, s38, s66                                ; 00000000217C: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000002180: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000002184: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000002188: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 00000000218C: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000002190: 98278027
	s_barrier_signal -1                                        ; 000000002194: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000002198: BF94FFFF
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2                                       ; 0000000029A8: BFCB0002
	s_barrier_signal -1                                        ; 0000000029AC: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000029B0: BF94FFFF
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x2000
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000002B08: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000002B14: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000002B1C: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000002B20: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000002B24: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000002B28: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000002B2C: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000002B30: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000002B34: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000002B38: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000002B3C: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000002B40: 98278027
	s_branch .Lbranch_000000007158                                              ; 000000002B44: BFA01184 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x5858>
.Lbranch_000000002b48:
	s_mov_b32 s95, 0x8000
	s_mov_b32 s96, 0x10000
	s_mov_b32 s97, 0x18000
	s_mov_b32 s98, 0x10000
	s_mov_b32 s32, 1                                           ; 000000002B68: BEA00081
	s_mov_b32 s33, 0                                           ; 000000002B6C: BEA10080
	s_mov_b32 s34, 0                                           ; 000000002B70: BEA20080
	s_mov_b32 s35, 0x80000000                                  ; 000000002B74: BEA300FF 80000000
	s_mov_b32 s33, 0x8000
	s_mov_b32 s34, s74                                         ; 000000002B84: BEA2004A
	s_and_b32 s75, s75, 0x1ffffff                              ; 000000002B88: 8B4BFF4B 01FFFFFF
	s_and_b32 s35, s35, 0xfe000000                             ; 000000002B90: 8B23FF23 FE000000
	s_or_b32 s35, s75, s35                                     ; 000000002B98: 8C23234B
	s_mov_b32 s36, 0                                           ; 000000002B9C: BEA40080
	s_mov_b32 s37, 0                                           ; 000000002BA0: BEA50080
	s_mov_b32 s38, 0                                           ; 000000002BA4: BEA60080
	s_mov_b32 s39, 0                                           ; 000000002BA8: BEA70080
	s_mov_b32 s40, 0                                           ; 000000002BAC: BEA80080
	s_mov_b32 s41, 0                                           ; 000000002BB0: BEA90080
	s_mov_b32 s42, 0                                           ; 000000002BB4: BEAA0080
	s_mov_b32 s43, 0                                           ; 000000002BB8: BEAB0080
	s_lshl_b32 s26, s14, 4                                     ; 000000002BBC: 841A840E
	s_and_b32 s37, s37, 0xffff                                 ; 000000002BC0: 8B25FF25 0000FFFF
	s_and_b32 s38, s38, 0xffff0000                             ; 000000002BC8: 8B26FF26 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000002BD0: 8418901A
	s_or_b32 s37, s24, s37                                     ; 000000002BD4: 8C252518
	s_lshr_b32 s24, s26, 16                                    ; 000000002BD8: 8518901A
	s_or_b32 s38, s24, s38                                     ; 000000002BDC: 8C262618
	s_mul_i32 s24, s54, 0x100
	s_sub_co_u32 s26, s18, s24                                 ; 000000002BE8: 809A1812
	s_lshr_b32 s26, s26, 4                                     ; 000000002BEC: 851A841A
	s_and_b32 s38, s38, 0xffff                                 ; 000000002BF0: 8B26FF26 0000FFFF
	s_and_b32 s39, s39, 0xffff0000                             ; 000000002BF8: 8B27FF27 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000002C00: 8418901A
	s_or_b32 s38, s24, s38                                     ; 000000002C04: 8C262618
	s_lshr_b32 s24, s26, 16                                    ; 000000002C08: 8518901A
	s_or_b32 s39, s24, s39                                     ; 000000002C0C: 8C272718
	s_and_b32 s39, s39, 0xffff                                 ; 000000002C10: 8B27FF27 0000FFFF
	s_or_b32 s39, s39, 0x8000000                               ; 000000002C18: 8C27FF27 08000000
	s_and_b32 s40, s40, 0xffff0000                             ; 000000002C20: 8B28FF28 FFFF0000
	s_or_b32 s40, s40, 16                                     ; preshuffle blocks: 16 * (16 rows * 0x80 bytes)
	s_lshl_b32 s24, s14, 4                                     ; 000000002C2C: 8418840E
	s_mov_b32 s25, 0                                           ; 000000002C30: BE990080
	s_mov_b32 s41, s24                                         ; 000000002C34: BEA90018
	s_and_b32 s25, s25, 0xffff                                 ; 000000002C38: 8B19FF19 0000FFFF
	s_and_b32 s42, s42, 0xffff0000                             ; 000000002C40: 8B2AFF2A FFFF0000
	s_or_b32 s42, s42, s25                                     ; 000000002C48: 8C2A192A
	s_bitset0_b32 s36, 20                                      ; 000000002C4C: BEA41094
	s_mov_b32 s53, 1
	s_lshl_b32 s53, s53, s49                                  ; B has no M reuse: local requester 1 << wg_x
	s_and_b32 s36, s36, 0xffff0000                             ; 000000002C90: 8B24FF24 FFFF0000
	s_or_b32 s36, s53, s36                                     ; 000000002CA0: 8C242435
	s_bitset1_b32 s36, 21                                      ; 000000002CA4: BEA41295
	s_mov_b32 s56, 0x800                                       ; 000000002CA8: BEB800FF 00000800
	s_mov_b32 s57, 0                                           ; 000000002CB0: BEB90080
	s_mul_i32 s24, s68, 0x100
	s_mul_hi_u32 s63, s24, s14                                 ; 000000002CBC: 96BF0E18
	s_mul_i32 s24, s24, s14                                    ; 000000002CC0: 96180E18
	s_add_co_u32 s62, s6, s24                                  ; 000000002CC4: 803E1806
	s_add_co_ci_u32 s63, s63, s7                               ; 000000002CC8: 823F073F
	s_and_b32 s63, s63, 0x1ffffff                              ; 000000002CCC: 8B3FFF3F 01FFFFFF
	s_or_b32 s63, s63, 0x80000000                              ; 000000002CD4: 8C3FFF3F 80000000
	s_mov_b32 s64, s36                                         ; 000000002CDC: BEC00024
	s_mov_b32 s65, s37                                         ; 000000002CE0: BEC10025
	s_mul_i32 s27, s68, 0x100
	s_sub_co_u32 s27, s18, s27                                 ; 000000002CEC: 809B1B12
	s_lshr_b32 s27, s27, 4                                     ; 000000002CF0: 851B841B
	s_lshl_b32 s26, s27, 16                                    ; 000000002CF4: 841A901B
	s_and_b32 s66, s38, 0xffff                                 ; 000000002CF8: 8B42FF26 0000FFFF
	s_or_b32 s66, s66, s26                                     ; 000000002D00: 8C421A42
	s_lshr_b32 s26, s27, 16                                    ; 000000002D04: 851A901B
	s_or_b32 s67, s26, 0x8000000                               ; 000000002D08: 8C43FF1A 08000000
	s_barrier_signal -1                                        ; 000000002D10: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000002D14: BF94FFFF
	s_barrier_wait 0xfffd                                      ; 000000002D18: BF94FFFD
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x8000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000003530: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x100                               ; 00000000353C: 8018FF3A 00000100
	s_cmp_lt_u32 s24, s70                                      ; 000000003544: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000003548: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000354C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000003550: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000003554: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000003558: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 00000000355C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000003560: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000003564: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000003568: 98278027
	s_barrier_signal -1                                        ; 00000000356C: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000003570: BF94FFFF
	s_mov_b32 s33, 0x10000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 00000000357C: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x200                               ; 000000003588: 8018FF3A 00000200
	s_cmp_lt_u32 s24, s70                                      ; 000000003590: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000003594: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000003598: 98244024
	s_cselect_b32 s37, s37, s65                                ; 00000000359C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000035A0: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000035A4: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000035A8: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000035AC: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000035B0: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000035B4: 98278027
	s_barrier_signal -1                                        ; 0000000035B8: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000035BC: BF94FFFF
	s_mov_b32 s33, 0x18000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000035C8: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 0000000035D4: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 0000000035DC: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000035E0: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000035E4: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000035E8: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000035EC: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000035F0: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000035F4: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000035F8: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000035FC: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000003600: 98278027
	s_barrier_signal -1                                        ; 000000003604: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000003608: BF94FFFF
	s_wait_tensorcnt 0x2                                       ; 00000000360C: BFCB0002
	s_barrier_signal -1                                        ; 000000003610: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000003614: BF94FFFF
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x10000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 00000000376C: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000003778: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000003780: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000003784: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000003788: 98244024
	s_cselect_b32 s37, s37, s65                                ; 00000000378C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000003790: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000003794: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000003798: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 00000000379C: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000037A0: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000037A4: 98278027
	s_branch .Lbranch_000000008e6c                                              ; 0000000037A8: BFA015B0 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x756c>
.Lbranch_0000000037ac:
	s_mov_b32 s95, 0x6000
	s_mov_b32 s96, 0x6200
	s_mov_b32 s97, 0x6400
	s_mov_b32 s98, 0x6200
	s_mov_b32 s32, 1                                           ; 0000000037CC: BEA00081
	s_mov_b32 s33, 0                                           ; 0000000037D0: BEA10080
	s_mov_b32 s34, 0                                           ; 0000000037D4: BEA20080
	s_mov_b32 s35, 0x80000000                                  ; 0000000037D8: BEA300FF 80000000
	s_mov_b32 s33, 0x6000
	s_mov_b32 s34, s76                                         ; 0000000037E8: BEA2004C
	s_and_b32 s77, s77, 0x1ffffff                              ; 0000000037EC: 8B4DFF4D 01FFFFFF
	s_and_b32 s35, s35, 0xfe000000                             ; 0000000037F4: 8B23FF23 FE000000
	s_or_b32 s35, s77, s35                                     ; 0000000037FC: 8C23234D
	s_mov_b32 s36, 0                                           ; 000000003800: BEA40080
	s_mov_b32 s37, 0                                           ; 000000003804: BEA50080
	s_mov_b32 s38, 0                                           ; 000000003808: BEA60080
	s_mov_b32 s39, 0                                           ; 00000000380C: BEA70080
	s_mov_b32 s40, 0                                           ; 000000003810: BEA80080
	s_mov_b32 s41, 0                                           ; 000000003814: BEA90080
	s_mov_b32 s42, 0                                           ; 000000003818: BEAA0080
	s_mov_b32 s43, 0                                           ; 00000000381C: BEAB0080
	s_lshl_b32 s26, s15, 5                                     ; 000000003820: 841A850F
	s_and_b32 s37, s37, 0xffff                                 ; 000000003824: 8B25FF25 0000FFFF
	s_and_b32 s38, s38, 0xffff0000                             ; 00000000382C: 8B26FF26 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000003834: 8418901A
	s_or_b32 s37, s24, s37                                     ; 000000003838: 8C252518
	s_lshr_b32 s24, s26, 16                                    ; 00000000383C: 8518901A
	s_or_b32 s38, s24, s38                                     ; 000000003840: 8C262618
	s_mul_i32 s24, s55, 0x40
	s_sub_co_u32 s26, s17, s24                                 ; 00000000384C: 809A1811
	s_lshr_b32 s26, s26, 5                                     ; 000000003850: 851A851A
	s_and_b32 s38, s38, 0xffff                                 ; 000000003854: 8B26FF26 0000FFFF
	s_and_b32 s39, s39, 0xffff0000                             ; 00000000385C: 8B27FF27 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000003864: 8418901A
	s_or_b32 s38, s24, s38                                     ; 000000003868: 8C262618
	s_lshr_b32 s24, s26, 16                                    ; 00000000386C: 8518901A
	s_or_b32 s39, s24, s39                                     ; 000000003870: 8C272718
	s_and_b32 s39, s39, 0xffff                                 ; 000000003874: 8B27FF27 0000FFFF
	s_or_b32 s39, s39, 0x1000000                               ; 00000000387C: 8C27FF27 01000000
	s_and_b32 s40, s40, 0xffff0000                             ; 000000003884: 8B28FF28 FFFF0000
	s_or_b32 s40, s40, 2                                       ; preshuffle blocks: 2 * (32 rows * 0x08 bytes)
	s_lshl_b32 s24, s15, 5                                     ; 000000003890: 8418850F
	s_mov_b32 s25, 0                                           ; 000000003894: BE990080
	s_mov_b32 s41, s24                                         ; 000000003898: BEA90018
	s_and_b32 s25, s25, 0xffff                                 ; 00000000389C: 8B19FF19 0000FFFF
	s_and_b32 s42, s42, 0xffff0000                             ; 0000000038A4: 8B2AFF2A FFFF0000
	s_or_b32 s42, s42, s25                                     ; 0000000038AC: 8C2A192A
	s_bitset0_b32 s36, 20                                      ; 0000000038B0: BEA41094
	s_mov_b32 s53, 0xf                                         ; SA multicast: all four N WGs in the M row
	s_and_b32 s36, s36, 0xffff0000                             ; 0000000038D8: 8B24FF24 FFFF0000
	s_or_b32 s36, s53, s36                                     ; 0000000038E8: 8C242435
	s_bitset1_b32 s36, 21                                      ; 0000000038EC: BEA41295
	s_mov_b32 s56, 0x100                                       ; 0000000038F0: BEB800FF 00000100
	s_mov_b32 s57, 0                                           ; 0000000038F8: BEB90080
	s_mul_i32 s24, s69, 0x40
	s_mul_hi_u32 s63, s24, s15                                 ; 000000003904: 96BF0F18
	s_mul_i32 s24, s24, s15                                    ; 000000003908: 96180F18
	s_add_co_u32 s62, s8, s24                                  ; 00000000390C: 803E1808
	s_add_co_ci_u32 s63, s63, s9                               ; 000000003910: 823F093F
	s_and_b32 s63, s63, 0x1ffffff                              ; 000000003914: 8B3FFF3F 01FFFFFF
	s_or_b32 s63, s63, 0x80000000                              ; 00000000391C: 8C3FFF3F 80000000
	s_mov_b32 s64, s36                                         ; 000000003924: BEC00024
	s_mov_b32 s65, s37                                         ; 000000003928: BEC10025
	s_mul_i32 s27, s69, 0x40
	s_sub_co_u32 s27, s17, s27                                 ; 000000003934: 809B1B11
	s_lshr_b32 s27, s27, 5                                     ; 000000003938: 851B851B
	s_lshl_b32 s26, s27, 16                                    ; 00000000393C: 841A901B
	s_and_b32 s66, s38, 0xffff                                 ; 000000003940: 8B42FF26 0000FFFF
	s_or_b32 s66, s66, s26                                     ; 000000003948: 8C421A42
	s_lshr_b32 s26, s27, 16                                    ; 00000000394C: 851A901B
	s_or_b32 s67, s26, 0x1000000                               ; 000000003950: 8C43FF1A 01000000
	s_barrier_signal -1                                        ; 000000003958: BE804EC1
	s_barrier_wait 0xffff                                      ; 00000000395C: BF94FFFF
	s_barrier_wait 0xfffd                                      ; 000000003960: BF94FFFD
	s_wait_tensorcnt 0x0                                       ; 000000003964: BFCB0000
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x6000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 00000000417C: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x100                               ; 000000004188: 8018FF3A 00000100
	s_cmp_lt_u32 s24, s70                                      ; 000000004190: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000004194: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000004198: 98244024
	s_cselect_b32 s37, s37, s65                                ; 00000000419C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000041A0: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000041A4: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000041A8: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000041AC: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000041B0: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000041B4: 98278027
	s_barrier_signal -1                                        ; 0000000041B8: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000041BC: BF94FFFF
	s_mov_b32 s33, 0x6200
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000041C8: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x200                               ; 0000000041D4: 8018FF3A 00000200
	s_cmp_lt_u32 s24, s70                                      ; 0000000041DC: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000041E0: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000041E4: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000041E8: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000041EC: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000041F0: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000041F4: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000041F8: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000041FC: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000004200: 98278027
	s_barrier_signal -1                                        ; 000000004204: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000004208: BF94FFFF
	s_mov_b32 s33, 0x6400
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000004214: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000004220: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000004228: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 00000000422C: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000004230: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000004234: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000004238: 98264226
	s_cselect_b32 s70, s70, s71                                ; 00000000423C: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000004240: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000004244: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000004248: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 00000000424C: 98278027
	s_barrier_signal -1                                        ; 000000004250: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000004254: BF94FFFF
	s_wait_tensorcnt 0x2                                       ; 000000004258: BFCB0002
	s_barrier_signal -1                                        ; 00000000425C: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000004260: BF94FFFF
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x6200
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000043B8: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 0000000043C4: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 0000000043CC: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000043D0: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000043D4: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000043D8: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000043DC: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000043E0: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000043E4: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000043E8: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000043EC: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000043F0: 98278027
	s_branch .Lbranch_000000007158                                              ; 0000000043F4: BFA00B58 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x5858>
.Lbranch_0000000043f8:
	s_mov_b32 s95, 0x6800
	s_mov_b32 s96, 0x7000
	s_mov_b32 s97, 0x7800
	s_mov_b32 s98, 0x7000
	s_mov_b32 s32, 1                                           ; 000000004418: BEA00081
	s_mov_b32 s33, 0                                           ; 00000000441C: BEA10080
	s_mov_b32 s34, 0                                           ; 000000004420: BEA20080
	s_mov_b32 s35, 0x80000000                                  ; 000000004424: BEA300FF 80000000
	s_mov_b32 s33, 0x6800
	s_mov_b32 s34, s78                                         ; 000000004434: BEA2004E
	s_and_b32 s79, s79, 0x1ffffff                              ; 000000004438: 8B4FFF4F 01FFFFFF
	s_and_b32 s35, s35, 0xfe000000                             ; 000000004440: 8B23FF23 FE000000
	s_or_b32 s35, s79, s35                                     ; 000000004448: 8C23234F
	s_mov_b32 s36, 0                                           ; 00000000444C: BEA40080
	s_mov_b32 s37, 0                                           ; 000000004450: BEA50080
	s_mov_b32 s38, 0                                           ; 000000004454: BEA60080
	s_mov_b32 s39, 0                                           ; 000000004458: BEA70080
	s_mov_b32 s40, 0                                           ; 00000000445C: BEA80080
	s_mov_b32 s41, 0                                           ; 000000004460: BEA90080
	s_mov_b32 s42, 0                                           ; 000000004464: BEAA0080
	s_mov_b32 s43, 0                                           ; 000000004468: BEAB0080
	s_lshl_b32 s26, s16, 5                                     ; 00000000446C: 841A8510
	s_and_b32 s37, s37, 0xffff                                 ; 000000004470: 8B25FF25 0000FFFF
	s_and_b32 s38, s38, 0xffff0000                             ; 000000004478: 8B26FF26 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000004480: 8418901A
	s_or_b32 s37, s24, s37                                     ; 000000004484: 8C252518
	s_lshr_b32 s24, s26, 16                                    ; 000000004488: 8518901A
	s_or_b32 s38, s24, s38                                     ; 00000000448C: 8C262618
	s_mul_i32 s24, s54, 0x100
	s_sub_co_u32 s26, s18, s24                                 ; 000000004498: 809A1812
	s_lshr_b32 s26, s26, 5                                     ; 00000000449C: 851A851A
	s_and_b32 s38, s38, 0xffff                                 ; 0000000044A0: 8B26FF26 0000FFFF
	s_and_b32 s39, s39, 0xffff0000                             ; 0000000044A8: 8B27FF27 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 0000000044B0: 8418901A
	s_or_b32 s38, s24, s38                                     ; 0000000044B4: 8C262618
	s_lshr_b32 s24, s26, 16                                    ; 0000000044B8: 8518901A
	s_or_b32 s39, s24, s39                                     ; 0000000044BC: 8C272718
	s_and_b32 s39, s39, 0xffff                                 ; 0000000044C0: 8B27FF27 0000FFFF
	s_or_b32 s39, s39, 0x1000000                               ; 0000000044C8: 8C27FF27 01000000
	s_and_b32 s40, s40, 0xffff0000                             ; 0000000044D0: 8B28FF28 FFFF0000
	s_or_b32 s40, s40, 8                                       ; preshuffle blocks: 8 * (32 rows * 0x08 bytes)
	s_lshl_b32 s24, s16, 5                                     ; 0000000044DC: 84188510
	s_mov_b32 s25, 0                                           ; 0000000044E0: BE990080
	s_mov_b32 s41, s24                                         ; 0000000044E4: BEA90018
	s_and_b32 s25, s25, 0xffff                                 ; 0000000044E8: 8B19FF19 0000FFFF
	s_and_b32 s42, s42, 0xffff0000                             ; 0000000044F0: 8B2AFF2A FFFF0000
	s_or_b32 s42, s42, s25                                     ; 0000000044F8: 8C2A192A
	s_bitset0_b32 s36, 20                                      ; 0000000044FC: BEA41094
	s_mov_b32 s53, 1
	s_lshl_b32 s53, s53, s49                                  ; SB has no M reuse: local requester 1 << wg_x
	s_and_b32 s36, s36, 0xffff0000                             ; 000000004540: 8B24FF24 FFFF0000
	s_or_b32 s36, s53, s36                                     ; 000000004550: 8C242435
	s_bitset1_b32 s36, 21                                      ; 000000004554: BEA41295
	s_mov_b32 s56, 0x100                                       ; 000000004558: BEB800FF 00000100
	s_mov_b32 s57, 0                                           ; 000000004560: BEB90080
	s_mul_i32 s24, s68, 0x100
	s_mul_hi_u32 s63, s24, s16                                 ; 00000000456C: 96BF1018
	s_mul_i32 s24, s24, s16                                    ; 000000004570: 96181018
	s_add_co_u32 s62, s10, s24                                 ; 000000004574: 803E180A
	s_add_co_ci_u32 s63, s63, s11                              ; 000000004578: 823F0B3F
	s_and_b32 s63, s63, 0x1ffffff                              ; 00000000457C: 8B3FFF3F 01FFFFFF
	s_or_b32 s63, s63, 0x80000000                              ; 000000004584: 8C3FFF3F 80000000
	s_mov_b32 s64, s36                                         ; 00000000458C: BEC00024
	s_mov_b32 s65, s37                                         ; 000000004590: BEC10025
	s_mul_i32 s27, s68, 0x100
	s_sub_co_u32 s27, s18, s27                                 ; 00000000459C: 809B1B12
	s_lshr_b32 s27, s27, 5                                     ; 0000000045A0: 851B851B
	s_lshl_b32 s26, s27, 16                                    ; 0000000045A4: 841A901B
	s_and_b32 s66, s38, 0xffff                                 ; 0000000045A8: 8B42FF26 0000FFFF
	s_or_b32 s66, s66, s26                                     ; 0000000045B0: 8C421A42
	s_lshr_b32 s26, s27, 16                                    ; 0000000045B4: 851A901B
	s_or_b32 s67, s26, 0x1000000                               ; 0000000045B8: 8C43FF1A 01000000
	s_barrier_signal -1                                        ; 0000000045C0: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000045C4: BF94FFFF
	s_barrier_wait 0xfffd                                      ; 0000000045C8: BF94FFFD
	s_wait_tensorcnt 0x0                                       ; 0000000045CC: BFCB0000
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x6800
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000004DE4: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x100                               ; 000000004DF0: 8018FF3A 00000100
	s_cmp_lt_u32 s24, s70                                      ; 000000004DF8: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000004DFC: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000004E00: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000004E04: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000004E08: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000004E0C: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000004E10: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000004E14: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000004E18: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000004E1C: 98278027
	s_barrier_signal -1                                        ; 000000004E20: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000004E24: BF94FFFF
	s_mov_b32 s33, 0x7000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000004E30: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x200                               ; 000000004E3C: 8018FF3A 00000200
	s_cmp_lt_u32 s24, s70                                      ; 000000004E44: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000004E48: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000004E4C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000004E50: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000004E54: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000004E58: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000004E5C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000004E60: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000004E64: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000004E68: 98278027
	s_barrier_signal -1                                        ; 000000004E6C: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000004E70: BF94FFFF
	s_mov_b32 s33, 0x7800
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000004E7C: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000004E88: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000004E90: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000004E94: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000004E98: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000004E9C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000004EA0: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000004EA4: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000004EA8: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000004EAC: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000004EB0: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000004EB4: 98278027
	s_barrier_signal -1                                        ; 000000004EB8: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000004EBC: BF94FFFF
	s_wait_tensorcnt 0x2                                       ; 000000004EC0: BFCB0002
	s_barrier_signal -1                                        ; 000000004EC4: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000004EC8: BF94FFFF
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x7000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000005020: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 00000000502C: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000005034: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005038: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000503C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005040: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005044: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005048: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 00000000504C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005050: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005054: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005058: 98278027
	s_branch .Lbranch_000000008e6c                                              ; 00000000505C: BFA00F83 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x756c>
.Lbranch_000000005060:
	s_mul_i32 s24, s69, 0x40
	s_mul_hi_u32 s63, s24, s13                                 ; 000000005068: 96BF0D18
	s_mul_i32 s24, s24, s13                                    ; 00000000506C: 96180D18
	s_add_co_u32 s62, s4, s24                                  ; 000000005070: 803E1804
	s_add_co_ci_u32 s63, s63, s5                               ; 000000005074: 823F053F
	s_and_b32 s63, s63, 0x1ffffff                              ; 000000005078: 8B3FFF3F 01FFFFFF
	s_or_b32 s63, s63, 0x80000000                              ; 000000005080: 8C3FFF3F 80000000
	s_mov_b32 s64, s36                                         ; 000000005088: BEC00024
	s_mov_b32 s65, s37                                         ; 00000000508C: BEC10025
	s_mul_i32 s27, s69, 0x40
	s_sub_co_u32 s27, s17, s27                                 ; 000000005098: 809B1B11
	s_lshr_b32 s27, s27, 4                                     ; 00000000509C: 851B841B
	s_lshl_b32 s26, s27, 16                                    ; 0000000050A0: 841A901B
	s_and_b32 s66, s38, 0xffff                                 ; 0000000050A4: 8B42FF26 0000FFFF
	s_or_b32 s66, s66, s26                                     ; 0000000050AC: 8C421A42
	s_lshr_b32 s26, s27, 16                                    ; 0000000050B0: 851A901B
	s_or_b32 s67, s26, 0x8000000                               ; 0000000050B4: 8C43FF1A 08000000
	s_barrier_signal -3                                        ; 0000000050BC: BE804EC3
	s_barrier_wait 0xfffd                                      ; 0000000050C0: BF94FFFD
	s_mov_b32 s24, 0                                           ; 0000000050C4: BE980080
	s_mov_b32 s25, 0                                           ; 0000000050C8: BE990080
	s_mov_b32 s26, 0                                           ; 0000000050CC: BE9A0080
	s_cmp_eq_u32 s22, 0                                        ; 0000000050D0: BF068016
	s_cselect_b32 s24, s4, s24                                 ; 0000000050D4: 98181804
	s_cselect_b32 s25, s5, s25                                 ; 0000000050D8: 98191905
	s_cselect_b32 s26, s13, s26                                ; 0000000050DC: 981A1A0D
	s_cmp_eq_u32 s22, 1                                        ; 0000000050E0: BF068116
	s_cselect_b32 s24, s6, s24                                 ; 0000000050E4: 98181806
	s_cselect_b32 s25, s7, s25                                 ; 0000000050E8: 98191907
	s_cselect_b32 s26, s14, s26                                ; 0000000050EC: 981A1A0E
	s_cmp_eq_u32 s22, 2                                        ; 0000000050F0: BF068216
	s_cselect_b32 s24, s8, s24                                 ; 0000000050F4: 98181808
	s_cselect_b32 s25, s9, s25                                 ; 0000000050F8: 98191909
	s_cselect_b32 s26, s15, s26                                ; 0000000050FC: 981A1A0F
	s_cmp_eq_u32 s22, 3                                        ; 000000005100: BF068316
	s_cselect_b32 s24, s10, s24                                ; 000000005104: 9818180A
	s_cselect_b32 s25, s11, s25                                ; 000000005108: 9819190B
	s_cselect_b32 s26, s16, s26                                ; 00000000510C: 981A1A10
	s_mul_i32 s27, s69, 0x40                                  ; wave0 A tile origin
	s_mul_i32 s27, s27, s26                                    ; 000000005124: 961B1A1B
	s_add_co_u32 s24, s24, s27                                 ; 000000005128: 80181B18
	s_add_co_ci_u32 s25, 0, s25                                ; 00000000512C: 82191980
	s_mul_i32 s27, 0x40, s26
	s_sub_co_i32 s27, s27, 1                                   ; 000000005138: 819B811B
	v_mov_b32_e32 v5, 0                                        ; 00000000513C: 7E0A0280
	v_mov_b32_e32 v7, 0                                        ; 000000005140: 7E0E0280
	v_and_b32_e64 v6, v0, 1                                    ; 000000005144: D51B0006 00010300
	v_mul_u32_u24_e64 v6, v6, 0x200000                         ; 00000000514C: D50B0006 0001FF06 00200000
	v_and_b32_e64 v4, v0, 15                                   ; 000000005158: D51B0004 00011F00
	v_mul_u32_u24_e64 v4, v4, 4
	v_mul_lo_u32 v4, v4, s26                                   ; 000000005168: D72C0004 00003504
	s_mov_b32 exec_lo, 0xffff                                  ; 000000005170: BEFE00FF 0000FFFF
	global_prefetch_b8 v4, s[24:25]                            ; 000000005178: EE174018 00000000 00000004
	s_mov_b32 exec_lo, -1                                      ; 000000005184: BEFE00C1
	s_wait_tensorcnt 0x0                                       ; 00000000518C: BFCB0000
	s_cmp_eq_u32 s94, 1                                        ; 000000005190: BF06815E
	s_cbranch_scc0 .Lbranch_000000005548                                         ; 000000005194: BFA100EC <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x3c48>
	s_mov_b32 s94, 0                                           ; 000000005198: BEDE0080
	s_mov_b32 s32, 1                                           ; 00000000519C: BEA00081
	s_mov_b32 s33, 0                                           ; 0000000051A0: BEA10080
	s_mov_b32 s34, 0                                           ; 0000000051A4: BEA20080
	s_mov_b32 s35, 0x80000000                                  ; 0000000051A8: BEA300FF 80000000
	s_mov_b32 s33, 0                                           ; 0000000051B0: BEA10080
	s_mov_b32 s34, s72                                         ; 0000000051B4: BEA20048
	s_and_b32 s73, s73, 0x1ffffff                              ; 0000000051B8: 8B49FF49 01FFFFFF
	s_and_b32 s35, s35, 0xfe000000                             ; 0000000051C0: 8B23FF23 FE000000
	s_or_b32 s35, s73, s35                                     ; 0000000051C8: 8C232349
	s_mov_b32 s36, 0                                           ; 0000000051CC: BEA40080
	s_mov_b32 s37, 0                                           ; 0000000051D0: BEA50080
	s_mov_b32 s38, 0                                           ; 0000000051D4: BEA60080
	s_mov_b32 s39, 0                                           ; 0000000051D8: BEA70080
	s_mov_b32 s40, 0                                           ; 0000000051DC: BEA80080
	s_mov_b32 s41, 0                                           ; 0000000051E0: BEA90080
	s_mov_b32 s42, 0                                           ; 0000000051E4: BEAA0080
	s_mov_b32 s43, 0                                           ; 0000000051E8: BEAB0080
	s_lshl_b32 s26, s13, 4                                     ; 0000000051EC: 841A840D
	s_and_b32 s37, s37, 0xffff                                 ; 0000000051F0: 8B25FF25 0000FFFF
	s_and_b32 s38, s38, 0xffff0000                             ; 0000000051F8: 8B26FF26 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000005200: 8418901A
	s_or_b32 s37, s24, s37                                     ; 000000005204: 8C252518
	s_lshr_b32 s24, s26, 16                                    ; 000000005208: 8518901A
	s_or_b32 s38, s24, s38                                     ; 00000000520C: 8C262618
	s_mul_i32 s24, s55, 0x40
	s_sub_co_u32 s26, s17, s24                                 ; 000000005218: 809A1811
	s_lshr_b32 s26, s26, 4                                     ; 00000000521C: 851A841A
	s_and_b32 s38, s38, 0xffff                                 ; 000000005220: 8B26FF26 0000FFFF
	s_and_b32 s39, s39, 0xffff0000                             ; 000000005228: 8B27FF27 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000005230: 8418901A
	s_or_b32 s38, s24, s38                                     ; 000000005234: 8C262618
	s_lshr_b32 s24, s26, 16                                    ; 000000005238: 8518901A
	s_or_b32 s39, s24, s39                                     ; 00000000523C: 8C272718
	s_and_b32 s39, s39, 0xffff                                 ; 000000005240: 8B27FF27 0000FFFF
	s_or_b32 s39, s39, 0x8000000                               ; 000000005248: 8C27FF27 08000000
	s_and_b32 s40, s40, 0xffff0000                             ; 000000005250: 8B28FF28 FFFF0000
	s_or_b32 s40, s40, 4
	s_lshl_b32 s24, s13, 4                                     ; 00000000525C: 8418840D
	s_mov_b32 s25, 0                                           ; 000000005260: BE990080
	s_mov_b32 s41, s24                                         ; 000000005264: BEA90018
	s_and_b32 s25, s25, 0xffff                                 ; 000000005268: 8B19FF19 0000FFFF
	s_and_b32 s42, s42, 0xffff0000                             ; 000000005270: 8B2AFF2A FFFF0000
	s_or_b32 s42, s42, s25                                     ; 000000005278: 8C2A192A
	s_bitset0_b32 s36, 20                                      ; 00000000527C: BEA41094
	s_mov_b32 s53, 0xf
	s_and_b32 s36, s36, 0xffff0000                             ; 0000000052A4: 8B24FF24 FFFF0000
	s_or_b32 s36, s53, s36                                     ; 0000000052B4: 8C242435
	s_bitset1_b32 s36, 21                                      ; 0000000052B8: BEA41295
	s_mov_b32 s56, 0x800                                       ; 0000000052BC: BEB800FF 00000800
	s_mov_b32 s57, 0                                           ; 0000000052C4: BEB90080
	s_mov_b32 s33, 0                                           ; 0000000052C8: BEA10080
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 0000000052CC: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x100                               ; 0000000052D8: 8018FF3A 00000100
	s_cmp_lt_u32 s24, s70                                      ; 0000000052E0: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000052E4: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000052E8: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000052EC: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000052F0: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000052F4: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000052F8: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000052FC: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005300: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005304: 98278027
	s_barrier_signal -1                                        ; 000000005308: BE804EC1
	s_barrier_wait 0xffff                                      ; 00000000530C: BF94FFFF
	s_mov_b32 s33, 0x2000
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000005318: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x200                               ; 000000005324: 8018FF3A 00000200
	s_cmp_lt_u32 s24, s70                                      ; 00000000532C: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005330: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005334: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005338: 98254125
	s_cselect_b32 s38, s38, s66                                ; 00000000533C: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005340: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005344: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005348: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 00000000534C: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005350: 98278027
	s_barrier_signal -1                                        ; 000000005354: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000005358: BF94FFFF
	s_mov_b32 s33, 0x4000
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000005364: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000005370: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000005378: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 00000000537C: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005380: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005384: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005388: 98264226
	s_cselect_b32 s70, s70, s71                                ; 00000000538C: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005390: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005394: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005398: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 00000000539C: 98278027
	s_barrier_signal -1                                        ; 0000000053A0: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000053A4: BF94FFFF
	s_wait_tensorcnt 0x2                                       ; 0000000053A8: BFCB0002
	s_barrier_signal -1                                        ; 0000000053AC: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000053B0: BF94FFFF
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x2000
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000005508: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000005514: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 00000000551C: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005520: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005524: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005528: 98254125
	s_cselect_b32 s38, s38, s66                                ; 00000000552C: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005530: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005534: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005538: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 00000000553C: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005540: 98278027
	s_branch .Lbranch_000000007158                                              ; 000000005544: BFA00704 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x5858>
.Lbranch_000000005548:
	s_branch .Lbranch_000000006454                                               ; 000000005548: BFA003C2 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4b54>
.Lbranch_00000000554c:
	s_mul_i32 s24, s68, 0x100
	s_mul_hi_u32 s63, s24, s14                                 ; 000000005554: 96BF0E18
	s_mul_i32 s24, s24, s14                                    ; 000000005558: 96180E18
	s_add_co_u32 s62, s6, s24                                  ; 00000000555C: 803E1806
	s_add_co_ci_u32 s63, s63, s7                               ; 000000005560: 823F073F
	s_and_b32 s63, s63, 0x1ffffff                              ; 000000005564: 8B3FFF3F 01FFFFFF
	s_or_b32 s63, s63, 0x80000000                              ; 00000000556C: 8C3FFF3F 80000000
	s_mov_b32 s64, s36                                         ; 000000005574: BEC00024
	s_mov_b32 s65, s37                                         ; 000000005578: BEC10025
	s_mul_i32 s27, s68, 0x100
	s_sub_co_u32 s27, s18, s27                                 ; 000000005584: 809B1B12
	s_lshr_b32 s27, s27, 4                                     ; 000000005588: 851B841B
	s_lshl_b32 s26, s27, 16                                    ; 00000000558C: 841A901B
	s_and_b32 s66, s38, 0xffff                                 ; 000000005590: 8B42FF26 0000FFFF
	s_or_b32 s66, s66, s26                                     ; 000000005598: 8C421A42
	s_lshr_b32 s26, s27, 16                                    ; 00000000559C: 851A901B
	s_or_b32 s67, s26, 0x8000000                               ; 0000000055A0: 8C43FF1A 08000000
	s_barrier_wait 0xfffd                                      ; 0000000055A8: BF94FFFD
	s_mov_b32 s24, 0                                           ; 0000000055AC: BE980080
	s_mov_b32 s25, 0                                           ; 0000000055B0: BE990080
	s_mov_b32 s26, 0                                           ; 0000000055B4: BE9A0080
	s_cmp_eq_u32 s22, 0                                        ; 0000000055B8: BF068016
	s_cselect_b32 s24, s4, s24                                 ; 0000000055BC: 98181804
	s_cselect_b32 s25, s5, s25                                 ; 0000000055C0: 98191905
	s_cselect_b32 s26, s13, s26                                ; 0000000055C4: 981A1A0D
	s_cmp_eq_u32 s22, 1                                        ; 0000000055C8: BF068116
	s_cselect_b32 s24, s6, s24                                 ; 0000000055CC: 98181806
	s_cselect_b32 s25, s7, s25                                 ; 0000000055D0: 98191907
	s_cselect_b32 s26, s14, s26                                ; 0000000055D4: 981A1A0E
	s_cmp_eq_u32 s22, 2                                        ; 0000000055D8: BF068216
	s_cselect_b32 s24, s8, s24                                 ; 0000000055DC: 98181808
	s_cselect_b32 s25, s9, s25                                 ; 0000000055E0: 98191909
	s_cselect_b32 s26, s15, s26                                ; 0000000055E4: 981A1A0F
	s_cmp_eq_u32 s22, 3                                        ; 0000000055E8: BF068316
	s_cselect_b32 s24, s10, s24                                ; 0000000055EC: 9818180A
	s_cselect_b32 s25, s11, s25                                ; 0000000055F0: 9819190B
	s_cselect_b32 s26, s16, s26                                ; 0000000055F4: 981A1A10
	s_mul_i32 s27, s68, 0x100                                 ; wave1 B tile origin
	s_mul_i32 s27, s27, s26                                    ; 00000000560C: 961B1A1B
	s_add_co_u32 s24, s24, s27                                 ; 000000005610: 80181B18
	s_add_co_ci_u32 s25, 0, s25                                ; 000000005614: 82191980
	s_mul_i32 s27, 0x100, s26
	s_sub_co_i32 s27, s27, 1                                   ; 000000005620: 819B811B
	v_mov_b32_e32 v5, 0                                        ; 000000005624: 7E0A0280
	v_mov_b32_e32 v7, 0                                        ; 000000005628: 7E0E0280
	v_and_b32_e64 v6, v0, 1                                    ; 00000000562C: D51B0006 00010300
	v_mul_u32_u24_e64 v6, v6, 0x200000                         ; 000000005634: D50B0006 0001FF06 00200000
	v_and_b32_e64 v4, v0, 15                                   ; 000000005640: D51B0004 00011F00
	v_mul_u32_u24_e64 v4, v4, 16
	v_mul_lo_u32 v4, v4, s26                                   ; 000000005650: D72C0004 00003504
	s_mov_b32 exec_lo, 0xffff                                  ; 000000005658: BEFE00FF 0000FFFF
	global_prefetch_b8 v4, s[24:25]                            ; 000000005660: EE174018 00000000 00000004
	s_mov_b32 exec_lo, -1                                      ; 00000000566C: BEFE00C1
	s_wait_tensorcnt 0x0                                       ; 000000005674: BFCB0000
	s_cmp_eq_u32 s94, 1                                        ; 000000005678: BF06815E
	s_cbranch_scc0 .Lbranch_000000005a54                                         ; 00000000567C: BFA100F5 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4154>
	s_mov_b32 s94, 0                                           ; 000000005680: BEDE0080
	s_mov_b32 s32, 1                                           ; 000000005684: BEA00081
	s_mov_b32 s33, 0                                           ; 000000005688: BEA10080
	s_mov_b32 s34, 0                                           ; 00000000568C: BEA20080
	s_mov_b32 s35, 0x80000000                                  ; 000000005690: BEA300FF 80000000
	s_mov_b32 s33, 0x8000
	s_mov_b32 s34, s74                                         ; 0000000056A0: BEA2004A
	s_and_b32 s75, s75, 0x1ffffff                              ; 0000000056A4: 8B4BFF4B 01FFFFFF
	s_and_b32 s35, s35, 0xfe000000                             ; 0000000056AC: 8B23FF23 FE000000
	s_or_b32 s35, s75, s35                                     ; 0000000056B4: 8C23234B
	s_mov_b32 s36, 0                                           ; 0000000056B8: BEA40080
	s_mov_b32 s37, 0                                           ; 0000000056BC: BEA50080
	s_mov_b32 s38, 0                                           ; 0000000056C0: BEA60080
	s_mov_b32 s39, 0                                           ; 0000000056C4: BEA70080
	s_mov_b32 s40, 0                                           ; 0000000056C8: BEA80080
	s_mov_b32 s41, 0                                           ; 0000000056CC: BEA90080
	s_mov_b32 s42, 0                                           ; 0000000056D0: BEAA0080
	s_mov_b32 s43, 0                                           ; 0000000056D4: BEAB0080
	s_lshl_b32 s26, s14, 4                                     ; 0000000056D8: 841A840E
	s_and_b32 s37, s37, 0xffff                                 ; 0000000056DC: 8B25FF25 0000FFFF
	s_and_b32 s38, s38, 0xffff0000                             ; 0000000056E4: 8B26FF26 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 0000000056EC: 8418901A
	s_or_b32 s37, s24, s37                                     ; 0000000056F0: 8C252518
	s_lshr_b32 s24, s26, 16                                    ; 0000000056F4: 8518901A
	s_or_b32 s38, s24, s38                                     ; 0000000056F8: 8C262618
	s_mul_i32 s24, s54, 0x100
	s_sub_co_u32 s26, s18, s24                                 ; 000000005704: 809A1812
	s_lshr_b32 s26, s26, 4                                     ; 000000005708: 851A841A
	s_and_b32 s38, s38, 0xffff                                 ; 00000000570C: 8B26FF26 0000FFFF
	s_and_b32 s39, s39, 0xffff0000                             ; 000000005714: 8B27FF27 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 00000000571C: 8418901A
	s_or_b32 s38, s24, s38                                     ; 000000005720: 8C262618
	s_lshr_b32 s24, s26, 16                                    ; 000000005724: 8518901A
	s_or_b32 s39, s24, s39                                     ; 000000005728: 8C272718
	s_and_b32 s39, s39, 0xffff                                 ; 00000000572C: 8B27FF27 0000FFFF
	s_or_b32 s39, s39, 0x8000000                               ; 000000005734: 8C27FF27 08000000
	s_and_b32 s40, s40, 0xffff0000                             ; 00000000573C: 8B28FF28 FFFF0000
	s_or_b32 s40, s40, 16
	s_lshl_b32 s24, s14, 4                                     ; 000000005748: 8418840E
	s_mov_b32 s25, 0                                           ; 00000000574C: BE990080
	s_mov_b32 s41, s24                                         ; 000000005750: BEA90018
	s_and_b32 s25, s25, 0xffff                                 ; 000000005754: 8B19FF19 0000FFFF
	s_and_b32 s42, s42, 0xffff0000                             ; 00000000575C: 8B2AFF2A FFFF0000
	s_or_b32 s42, s42, s25                                     ; 000000005764: 8C2A192A
	s_bitset0_b32 s36, 20                                      ; 000000005768: BEA41094
	s_mov_b32 s53, 1
	s_lshl_b32 s53, s53, s49
	s_and_b32 s36, s36, 0xffff0000                             ; 0000000057AC: 8B24FF24 FFFF0000
	s_or_b32 s36, s53, s36                                     ; 0000000057BC: 8C242435
	s_bitset1_b32 s36, 21                                      ; 0000000057C0: BEA41295
	s_mov_b32 s56, 0x800                                       ; 0000000057C4: BEB800FF 00000800
	s_mov_b32 s57, 0                                           ; 0000000057CC: BEB90080
	s_mov_b32 s33, 0x8000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000057D8: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x100                               ; 0000000057E4: 8018FF3A 00000100
	s_cmp_lt_u32 s24, s70                                      ; 0000000057EC: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000057F0: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000057F4: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000057F8: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000057FC: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005800: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005804: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005808: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 00000000580C: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005810: 98278027
	s_barrier_signal -1                                        ; 000000005814: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000005818: BF94FFFF
	s_mov_b32 s33, 0x10000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000005824: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x200                               ; 000000005830: 8018FF3A 00000200
	s_cmp_lt_u32 s24, s70                                      ; 000000005838: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 00000000583C: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005840: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005844: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005848: 98264226
	s_cselect_b32 s70, s70, s71                                ; 00000000584C: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005850: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005854: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005858: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 00000000585C: 98278027
	s_barrier_signal -1                                        ; 000000005860: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000005864: BF94FFFF
	s_mov_b32 s33, 0x18000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000005870: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 00000000587C: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000005884: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005888: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000588C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005890: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005894: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005898: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 00000000589C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000058A0: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000058A4: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000058A8: 98278027
	s_barrier_signal -1                                        ; 0000000058AC: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000058B0: BF94FFFF
	s_wait_tensorcnt 0x2                                       ; 0000000058B4: BFCB0002
	s_barrier_signal -1                                        ; 0000000058B8: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000058BC: BF94FFFF
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x10000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000005A14: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000005A20: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000005A28: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005A2C: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005A30: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005A34: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005A38: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005A3C: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005A40: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005A44: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005A48: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005A4C: 98278027
	s_branch .Lbranch_000000008e6c                                              ; 000000005A50: BFA00D06 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x756c>
.Lbranch_000000005a54:
	s_branch .Lbranch_0000000066cc                                               ; 000000005A54: BFA0031D <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4dcc>
.Lbranch_000000005a58:
	s_mul_i32 s24, s69, 0x40
	s_mul_hi_u32 s63, s24, s15                                 ; 000000005A60: 96BF0F18
	s_mul_i32 s24, s24, s15                                    ; 000000005A64: 96180F18
	s_add_co_u32 s62, s8, s24                                  ; 000000005A68: 803E1808
	s_add_co_ci_u32 s63, s63, s9                               ; 000000005A6C: 823F093F
	s_and_b32 s63, s63, 0x1ffffff                              ; 000000005A70: 8B3FFF3F 01FFFFFF
	s_or_b32 s63, s63, 0x80000000                              ; 000000005A78: 8C3FFF3F 80000000
	s_mov_b32 s64, s36                                         ; 000000005A80: BEC00024
	s_mov_b32 s65, s37                                         ; 000000005A84: BEC10025
	s_mul_i32 s27, s69, 0x40
	s_sub_co_u32 s27, s17, s27                                 ; 000000005A90: 809B1B11
	s_lshr_b32 s27, s27, 5                                     ; 000000005A94: 851B851B
	s_lshl_b32 s26, s27, 16                                    ; 000000005A98: 841A901B
	s_and_b32 s66, s38, 0xffff                                 ; 000000005A9C: 8B42FF26 0000FFFF
	s_or_b32 s66, s66, s26                                     ; 000000005AA4: 8C421A42
	s_lshr_b32 s26, s27, 16                                    ; 000000005AA8: 851A901B
	s_or_b32 s67, s26, 0x1000000                               ; 000000005AAC: 8C43FF1A 01000000
	s_barrier_wait 0xfffd                                      ; 000000005AB4: BF94FFFD
	s_mov_b32 s24, 0                                           ; 000000005AB8: BE980080
	s_mov_b32 s25, 0                                           ; 000000005ABC: BE990080
	s_mov_b32 s26, 0                                           ; 000000005AC0: BE9A0080
	s_cmp_eq_u32 s22, 0                                        ; 000000005AC4: BF068016
	s_cselect_b32 s24, s4, s24                                 ; 000000005AC8: 98181804
	s_cselect_b32 s25, s5, s25                                 ; 000000005ACC: 98191905
	s_cselect_b32 s26, s13, s26                                ; 000000005AD0: 981A1A0D
	s_cmp_eq_u32 s22, 1                                        ; 000000005AD4: BF068116
	s_cselect_b32 s24, s6, s24                                 ; 000000005AD8: 98181806
	s_cselect_b32 s25, s7, s25                                 ; 000000005ADC: 98191907
	s_cselect_b32 s26, s14, s26                                ; 000000005AE0: 981A1A0E
	s_cmp_eq_u32 s22, 2                                        ; 000000005AE4: BF068216
	s_cselect_b32 s24, s8, s24                                 ; 000000005AE8: 98181808
	s_cselect_b32 s25, s9, s25                                 ; 000000005AEC: 98191909
	s_cselect_b32 s26, s15, s26                                ; 000000005AF0: 981A1A0F
	s_cmp_eq_u32 s22, 3                                        ; 000000005AF4: BF068316
	s_cselect_b32 s24, s10, s24                                ; 000000005AF8: 9818180A
	s_cselect_b32 s25, s11, s25                                ; 000000005AFC: 9819190B
	s_cselect_b32 s26, s16, s26                                ; 000000005B00: 981A1A10
	s_mul_i32 s27, s69, 0x40                                  ; wave2 SA tile origin
	s_mul_i32 s27, s27, s26                                    ; 000000005B18: 961B1A1B
	s_add_co_u32 s24, s24, s27                                 ; 000000005B1C: 80181B18
	s_add_co_ci_u32 s25, 0, s25                                ; 000000005B20: 82191980
	s_mul_i32 s27, 0x40, s26
	s_sub_co_i32 s27, s27, 1                                   ; 000000005B2C: 819B811B
	v_mov_b32_e32 v5, 0                                        ; 000000005B30: 7E0A0280
	v_mov_b32_e32 v7, 0                                        ; 000000005B34: 7E0E0280
	v_and_b32_e64 v6, v0, 1                                    ; 000000005B38: D51B0006 00010300
	v_mul_u32_u24_e64 v6, v6, 0x200000                         ; 000000005B40: D50B0006 0001FF06 00200000
	v_and_b32_e64 v4, v0, 15                                   ; 000000005B4C: D51B0004 00011F00
	v_mul_u32_u24_e64 v4, v4, 4
	v_mul_lo_u32 v4, v4, s26                                   ; 000000005B5C: D72C0004 00003504
	s_mov_b32 exec_lo, 0xffff                                  ; 000000005B64: BEFE00FF 0000FFFF
	global_prefetch_b8 v4, s[24:25]                            ; 000000005B6C: EE174018 00000000 00000004
	s_mov_b32 exec_lo, -1                                      ; 000000005B78: BEFE00C1
	s_wait_tensorcnt 0x0                                       ; 000000005B80: BFCB0000
	s_cmp_eq_u32 s94, 1                                        ; 000000005B84: BF06815E
	s_cbranch_scc0 .Lbranch_000000005f44                                         ; 000000005B88: BFA100EE <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4644>
	s_mov_b32 s94, 0                                           ; 000000005B8C: BEDE0080
	s_mov_b32 s32, 1                                           ; 000000005B90: BEA00081
	s_mov_b32 s33, 0                                           ; 000000005B94: BEA10080
	s_mov_b32 s34, 0                                           ; 000000005B98: BEA20080
	s_mov_b32 s35, 0x80000000                                  ; 000000005B9C: BEA300FF 80000000
	s_mov_b32 s33, 0x6000
	s_mov_b32 s34, s76                                         ; 000000005BAC: BEA2004C
	s_and_b32 s77, s77, 0x1ffffff                              ; 000000005BB0: 8B4DFF4D 01FFFFFF
	s_and_b32 s35, s35, 0xfe000000                             ; 000000005BB8: 8B23FF23 FE000000
	s_or_b32 s35, s77, s35                                     ; 000000005BC0: 8C23234D
	s_mov_b32 s36, 0                                           ; 000000005BC4: BEA40080
	s_mov_b32 s37, 0                                           ; 000000005BC8: BEA50080
	s_mov_b32 s38, 0                                           ; 000000005BCC: BEA60080
	s_mov_b32 s39, 0                                           ; 000000005BD0: BEA70080
	s_mov_b32 s40, 0                                           ; 000000005BD4: BEA80080
	s_mov_b32 s41, 0                                           ; 000000005BD8: BEA90080
	s_mov_b32 s42, 0                                           ; 000000005BDC: BEAA0080
	s_mov_b32 s43, 0                                           ; 000000005BE0: BEAB0080
	s_lshl_b32 s26, s15, 5                                     ; 000000005BE4: 841A850F
	s_and_b32 s37, s37, 0xffff                                 ; 000000005BE8: 8B25FF25 0000FFFF
	s_and_b32 s38, s38, 0xffff0000                             ; 000000005BF0: 8B26FF26 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000005BF8: 8418901A
	s_or_b32 s37, s24, s37                                     ; 000000005BFC: 8C252518
	s_lshr_b32 s24, s26, 16                                    ; 000000005C00: 8518901A
	s_or_b32 s38, s24, s38                                     ; 000000005C04: 8C262618
	s_mul_i32 s24, s55, 0x40
	s_sub_co_u32 s26, s17, s24                                 ; 000000005C10: 809A1811
	s_lshr_b32 s26, s26, 5                                     ; 000000005C14: 851A851A
	s_and_b32 s38, s38, 0xffff                                 ; 000000005C18: 8B26FF26 0000FFFF
	s_and_b32 s39, s39, 0xffff0000                             ; 000000005C20: 8B27FF27 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000005C28: 8418901A
	s_or_b32 s38, s24, s38                                     ; 000000005C2C: 8C262618
	s_lshr_b32 s24, s26, 16                                    ; 000000005C30: 8518901A
	s_or_b32 s39, s24, s39                                     ; 000000005C34: 8C272718
	s_and_b32 s39, s39, 0xffff                                 ; 000000005C38: 8B27FF27 0000FFFF
	s_or_b32 s39, s39, 0x1000000                               ; 000000005C40: 8C27FF27 01000000
	s_and_b32 s40, s40, 0xffff0000                             ; 000000005C48: 8B28FF28 FFFF0000
	s_or_b32 s40, s40, 2
	s_lshl_b32 s24, s15, 5                                     ; 000000005C54: 8418850F
	s_mov_b32 s25, 0                                           ; 000000005C58: BE990080
	s_mov_b32 s41, s24                                         ; 000000005C5C: BEA90018
	s_and_b32 s25, s25, 0xffff                                 ; 000000005C60: 8B19FF19 0000FFFF
	s_and_b32 s42, s42, 0xffff0000                             ; 000000005C68: 8B2AFF2A FFFF0000
	s_or_b32 s42, s42, s25                                     ; 000000005C70: 8C2A192A
	s_bitset0_b32 s36, 20                                      ; 000000005C74: BEA41094
	s_mov_b32 s53, 0xf
	s_and_b32 s36, s36, 0xffff0000                             ; 000000005C9C: 8B24FF24 FFFF0000
	s_or_b32 s36, s53, s36                                     ; 000000005CAC: 8C242435
	s_bitset1_b32 s36, 21                                      ; 000000005CB0: BEA41295
	s_mov_b32 s56, 0x100                                       ; 000000005CB4: BEB800FF 00000100
	s_mov_b32 s57, 0                                           ; 000000005CBC: BEB90080
	s_mov_b32 s33, 0x6000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000005CC8: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x100                               ; 000000005CD4: 8018FF3A 00000100
	s_cmp_lt_u32 s24, s70                                      ; 000000005CDC: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005CE0: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005CE4: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005CE8: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005CEC: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005CF0: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005CF4: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005CF8: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005CFC: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005D00: 98278027
	s_barrier_signal -1                                        ; 000000005D04: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000005D08: BF94FFFF
	s_mov_b32 s33, 0x6200
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000005D14: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x200                               ; 000000005D20: 8018FF3A 00000200
	s_cmp_lt_u32 s24, s70                                      ; 000000005D28: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005D2C: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005D30: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005D34: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005D38: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005D3C: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005D40: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005D44: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005D48: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005D4C: 98278027
	s_barrier_signal -1                                        ; 000000005D50: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000005D54: BF94FFFF
	s_mov_b32 s33, 0x6400
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000005D60: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000005D6C: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000005D74: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005D78: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005D7C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005D80: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005D84: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005D88: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005D8C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005D90: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005D94: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005D98: 98278027
	s_barrier_signal -1                                        ; 000000005D9C: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000005DA0: BF94FFFF
	s_wait_tensorcnt 0x2                                       ; 000000005DA4: BFCB0002
	s_barrier_signal -1                                        ; 000000005DA8: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000005DAC: BF94FFFF
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x6200
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000005F04: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000005F10: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000005F18: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000005F1C: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000005F20: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000005F24: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000005F28: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000005F2C: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000005F30: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000005F34: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000005F38: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000005F3C: 98278027
	s_branch .Lbranch_000000007158                                              ; 000000005F40: BFA00485 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x5858>
.Lbranch_000000005f44:
	s_branch .Lbranch_000000006454                                               ; 000000005F44: BFA00143 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4b54>
.Lbranch_000000005f48:
	s_mul_i32 s24, s68, 0x100
	s_mul_hi_u32 s63, s24, s16                                 ; 000000005F50: 96BF1018
	s_mul_i32 s24, s24, s16                                    ; 000000005F54: 96181018
	s_add_co_u32 s62, s10, s24                                 ; 000000005F58: 803E180A
	s_add_co_ci_u32 s63, s63, s11                              ; 000000005F5C: 823F0B3F
	s_and_b32 s63, s63, 0x1ffffff                              ; 000000005F60: 8B3FFF3F 01FFFFFF
	s_or_b32 s63, s63, 0x80000000                              ; 000000005F68: 8C3FFF3F 80000000
	s_mov_b32 s64, s36                                         ; 000000005F70: BEC00024
	s_mov_b32 s65, s37                                         ; 000000005F74: BEC10025
	s_mul_i32 s27, s68, 0x100
	s_sub_co_u32 s27, s18, s27                                 ; 000000005F80: 809B1B12
	s_lshr_b32 s27, s27, 5                                     ; 000000005F84: 851B851B
	s_lshl_b32 s26, s27, 16                                    ; 000000005F88: 841A901B
	s_and_b32 s66, s38, 0xffff                                 ; 000000005F8C: 8B42FF26 0000FFFF
	s_or_b32 s66, s66, s26                                     ; 000000005F94: 8C421A42
	s_lshr_b32 s26, s27, 16                                    ; 000000005F98: 851A901B
	s_or_b32 s67, s26, 0x1000000                               ; 000000005F9C: 8C43FF1A 01000000
	s_barrier_wait 0xfffd                                      ; 000000005FA4: BF94FFFD
	s_mov_b32 s24, 0                                           ; 000000005FA8: BE980080
	s_mov_b32 s25, 0                                           ; 000000005FAC: BE990080
	s_mov_b32 s26, 0                                           ; 000000005FB0: BE9A0080
	s_cmp_eq_u32 s22, 0                                        ; 000000005FB4: BF068016
	s_cselect_b32 s24, s4, s24                                 ; 000000005FB8: 98181804
	s_cselect_b32 s25, s5, s25                                 ; 000000005FBC: 98191905
	s_cselect_b32 s26, s13, s26                                ; 000000005FC0: 981A1A0D
	s_cmp_eq_u32 s22, 1                                        ; 000000005FC4: BF068116
	s_cselect_b32 s24, s6, s24                                 ; 000000005FC8: 98181806
	s_cselect_b32 s25, s7, s25                                 ; 000000005FCC: 98191907
	s_cselect_b32 s26, s14, s26                                ; 000000005FD0: 981A1A0E
	s_cmp_eq_u32 s22, 2                                        ; 000000005FD4: BF068216
	s_cselect_b32 s24, s8, s24                                 ; 000000005FD8: 98181808
	s_cselect_b32 s25, s9, s25                                 ; 000000005FDC: 98191909
	s_cselect_b32 s26, s15, s26                                ; 000000005FE0: 981A1A0F
	s_cmp_eq_u32 s22, 3                                        ; 000000005FE4: BF068316
	s_cselect_b32 s24, s10, s24                                ; 000000005FE8: 9818180A
	s_cselect_b32 s25, s11, s25                                ; 000000005FEC: 9819190B
	s_cselect_b32 s26, s16, s26                                ; 000000005FF0: 981A1A10
	s_mul_i32 s27, s68, 0x100                                 ; wave3 SB tile origin
	s_mul_i32 s27, s27, s26                                    ; 000000006008: 961B1A1B
	s_add_co_u32 s24, s24, s27                                 ; 00000000600C: 80181B18
	s_add_co_ci_u32 s25, 0, s25                                ; 000000006010: 82191980
	s_mul_i32 s27, 0x100, s26
	s_sub_co_i32 s27, s27, 1                                   ; 00000000601C: 819B811B
	v_mov_b32_e32 v5, 0                                        ; 000000006020: 7E0A0280
	v_mov_b32_e32 v7, 0                                        ; 000000006024: 7E0E0280
	v_and_b32_e64 v6, v0, 1                                    ; 000000006028: D51B0006 00010300
	v_mul_u32_u24_e64 v6, v6, 0x200000                         ; 000000006030: D50B0006 0001FF06 00200000
	v_and_b32_e64 v4, v0, 15                                   ; 00000000603C: D51B0004 00011F00
	v_mul_u32_u24_e64 v4, v4, 16
	v_mul_lo_u32 v4, v4, s26                                   ; 00000000604C: D72C0004 00003504
	s_mov_b32 exec_lo, 0xffff                                  ; 000000006054: BEFE00FF 0000FFFF
	global_prefetch_b8 v4, s[24:25]                            ; 00000000605C: EE174018 00000000 00000004
	s_mov_b32 exec_lo, -1                                      ; 000000006068: BEFE00C1
	s_wait_tensorcnt 0x0                                       ; 000000006070: BFCB0000
	s_cmp_eq_u32 s94, 1                                        ; 000000006074: BF06815E
	s_cbranch_scc0 .Lbranch_000000006450                                         ; 000000006078: BFA100F5 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4b50>
	s_mov_b32 s94, 0                                           ; 00000000607C: BEDE0080
	s_mov_b32 s32, 1                                           ; 000000006080: BEA00081
	s_mov_b32 s33, 0                                           ; 000000006084: BEA10080
	s_mov_b32 s34, 0                                           ; 000000006088: BEA20080
	s_mov_b32 s35, 0x80000000                                  ; 00000000608C: BEA300FF 80000000
	s_mov_b32 s33, 0x6800
	s_mov_b32 s34, s78                                         ; 00000000609C: BEA2004E
	s_and_b32 s79, s79, 0x1ffffff                              ; 0000000060A0: 8B4FFF4F 01FFFFFF
	s_and_b32 s35, s35, 0xfe000000                             ; 0000000060A8: 8B23FF23 FE000000
	s_or_b32 s35, s79, s35                                     ; 0000000060B0: 8C23234F
	s_mov_b32 s36, 0                                           ; 0000000060B4: BEA40080
	s_mov_b32 s37, 0                                           ; 0000000060B8: BEA50080
	s_mov_b32 s38, 0                                           ; 0000000060BC: BEA60080
	s_mov_b32 s39, 0                                           ; 0000000060C0: BEA70080
	s_mov_b32 s40, 0                                           ; 0000000060C4: BEA80080
	s_mov_b32 s41, 0                                           ; 0000000060C8: BEA90080
	s_mov_b32 s42, 0                                           ; 0000000060CC: BEAA0080
	s_mov_b32 s43, 0                                           ; 0000000060D0: BEAB0080
	s_lshl_b32 s26, s16, 5                                     ; 0000000060D4: 841A8510
	s_and_b32 s37, s37, 0xffff                                 ; 0000000060D8: 8B25FF25 0000FFFF
	s_and_b32 s38, s38, 0xffff0000                             ; 0000000060E0: 8B26FF26 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 0000000060E8: 8418901A
	s_or_b32 s37, s24, s37                                     ; 0000000060EC: 8C252518
	s_lshr_b32 s24, s26, 16                                    ; 0000000060F0: 8518901A
	s_or_b32 s38, s24, s38                                     ; 0000000060F4: 8C262618
	s_mul_i32 s24, s54, 0x100
	s_sub_co_u32 s26, s18, s24                                 ; 000000006100: 809A1812
	s_lshr_b32 s26, s26, 5                                     ; 000000006104: 851A851A
	s_and_b32 s38, s38, 0xffff                                 ; 000000006108: 8B26FF26 0000FFFF
	s_and_b32 s39, s39, 0xffff0000                             ; 000000006110: 8B27FF27 FFFF0000
	s_lshl_b32 s24, s26, 16                                    ; 000000006118: 8418901A
	s_or_b32 s38, s24, s38                                     ; 00000000611C: 8C262618
	s_lshr_b32 s24, s26, 16                                    ; 000000006120: 8518901A
	s_or_b32 s39, s24, s39                                     ; 000000006124: 8C272718
	s_and_b32 s39, s39, 0xffff                                 ; 000000006128: 8B27FF27 0000FFFF
	s_or_b32 s39, s39, 0x1000000                               ; 000000006130: 8C27FF27 01000000
	s_and_b32 s40, s40, 0xffff0000                             ; 000000006138: 8B28FF28 FFFF0000
	s_or_b32 s40, s40, 8
	s_lshl_b32 s24, s16, 5                                     ; 000000006144: 84188510
	s_mov_b32 s25, 0                                           ; 000000006148: BE990080
	s_mov_b32 s41, s24                                         ; 00000000614C: BEA90018
	s_and_b32 s25, s25, 0xffff                                 ; 000000006150: 8B19FF19 0000FFFF
	s_and_b32 s42, s42, 0xffff0000                             ; 000000006158: 8B2AFF2A FFFF0000
	s_or_b32 s42, s42, s25                                     ; 000000006160: 8C2A192A
	s_bitset0_b32 s36, 20                                      ; 000000006164: BEA41094
	s_mov_b32 s53, 1
	s_lshl_b32 s53, s53, s49
	s_and_b32 s36, s36, 0xffff0000                             ; 0000000061A8: 8B24FF24 FFFF0000
	s_or_b32 s36, s53, s36                                     ; 0000000061B8: 8C242435
	s_bitset1_b32 s36, 21                                      ; 0000000061BC: BEA41295
	s_mov_b32 s56, 0x100                                       ; 0000000061C0: BEB800FF 00000100
	s_mov_b32 s57, 0                                           ; 0000000061C8: BEB90080
	s_mov_b32 s33, 0x6800
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000061D4: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x100                               ; 0000000061E0: 8018FF3A 00000100
	s_cmp_lt_u32 s24, s70                                      ; 0000000061E8: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000061EC: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000061F0: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000061F4: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000061F8: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000061FC: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006200: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006204: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006208: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 00000000620C: 98278027
	s_barrier_signal -1                                        ; 000000006210: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006214: BF94FFFF
	s_mov_b32 s33, 0x7000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000006220: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x200                               ; 00000000622C: 8018FF3A 00000200
	s_cmp_lt_u32 s24, s70                                      ; 000000006234: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006238: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000623C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000006240: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006244: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006248: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 00000000624C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006250: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006254: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006258: 98278027
	s_barrier_signal -1                                        ; 00000000625C: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006260: BF94FFFF
	s_mov_b32 s33, 0x7800
	tensor_load_to_lds s[32:35], s[36:43]                      ; 00000000626C: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000006278: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000006280: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006284: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000006288: 98244024
	s_cselect_b32 s37, s37, s65                                ; 00000000628C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006290: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006294: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006298: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 00000000629C: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000062A0: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000062A4: 98278027
	s_barrier_signal -1                                        ; 0000000062A8: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000062AC: BF94FFFF
	s_wait_tensorcnt 0x2                                       ; 0000000062B0: BFCB0002
	s_barrier_signal -1                                        ; 0000000062B4: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000062B8: BF94FFFF
	s_set_vgpr_msb 0
	s_mov_b32 s33, 0x7000
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000006410: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 00000000641C: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000006424: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006428: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000642C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000006430: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006434: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006438: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 00000000643C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006440: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006444: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006448: 98278027
	s_branch .Lbranch_000000008e6c                                              ; 00000000644C: BFA00A87 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x756c>
.Lbranch_000000006450:
	s_branch .Lbranch_0000000066cc                                               ; 000000006450: BFA0009E <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4dcc>
.Lbranch_000000006454:
	s_nop 0                                                    ; 000000006454: BF800000
	s_set_vgpr_msb 0                                           ; 000000006458: BF860000
	s_cmp_eq_u32 s92, 0                                        ; 00000000645C: BF06805C
	s_cbranch_scc1 .Lbranch_00000000647c                                           ; 000000006460: BFA20006 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4b7c>
	s_cmp_eq_u32 s92, 1                                        ; 000000006464: BF06815C
	s_cbranch_scc1 .Lbranch_000000006510                                          ; 000000006468: BFA20029 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4c10>
	s_cmp_eq_u32 s92, 2                                        ; 00000000646C: BF06825C
	s_cbranch_scc1 .Lbranch_0000000065a4                                          ; 000000006470: BFA2004C <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4ca4>
	s_cmp_eq_u32 s92, 3                                        ; 000000006474: BF06835C
	s_cbranch_scc1 .Lbranch_000000006638                                         ; 000000006478: BFA2006F <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4d38>
.Lbranch_00000000647c:
	s_mov_b32 s33, s98                                         ; 00000000647C: BEA10062
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000006480: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 00000000648C: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000006494: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006498: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000649C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000064A0: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000064A4: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000064A8: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000064AC: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000064B0: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000064B4: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000064B8: 98278027
	s_barrier_signal -1                                        ; 0000000064BC: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000064C0: BF94FFFF
	s_mov_b32 s33, s95                                         ; 0000000064C4: BEA1005F
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 0000000064C8: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 0000000064D4: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 0000000064DC: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000064E0: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000064E4: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000064E8: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000064EC: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000064F0: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000064F4: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000064F8: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000064FC: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006500: 98278027
	s_barrier_signal -1                                        ; 000000006504: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006508: BF94FFFF
	s_branch .Lbranch_000000007890                                              ; 00000000650C: BFA004E0 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x5f90>
.Lbranch_000000006510:
	s_mov_b32 s33, s95                                         ; 000000006510: BEA1005F
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000006514: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000006520: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000006528: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 00000000652C: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000006530: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000006534: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006538: 98264226
	s_cselect_b32 s70, s70, s71                                ; 00000000653C: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006540: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006544: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006548: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 00000000654C: 98278027
	s_barrier_signal -1                                        ; 000000006550: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006554: BF94FFFF
	s_mov_b32 s33, s96                                         ; 000000006558: BEA10060
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 00000000655C: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000006568: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000006570: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006574: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000006578: 98244024
	s_cselect_b32 s37, s37, s65                                ; 00000000657C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006580: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006584: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006588: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 00000000658C: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006590: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006594: 98278027
	s_barrier_signal -1                                        ; 000000006598: BE804EC1
	s_barrier_wait 0xffff                                      ; 00000000659C: BF94FFFF
	s_branch .Lbranch_000000007fc8                                              ; 0000000065A0: BFA00689 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x66c8>
.Lbranch_0000000065a4:
	s_mov_b32 s33, s96                                         ; 0000000065A4: BEA10060
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 0000000065A8: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 0000000065B4: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 0000000065BC: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000065C0: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000065C4: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000065C8: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000065CC: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000065D0: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000065D4: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000065D8: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000065DC: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000065E0: 98278027
	s_barrier_signal -1                                        ; 0000000065E4: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000065E8: BF94FFFF
	s_mov_b32 s33, s97                                         ; 0000000065EC: BEA10061
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 0000000065F0: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 0000000065FC: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000006604: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006608: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000660C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000006610: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006614: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006618: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 00000000661C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006620: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006624: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006628: 98278027
	s_barrier_signal -1                                        ; 00000000662C: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006630: BF94FFFF
	s_branch .Lbranch_000000008700                                              ; 000000006634: BFA00832 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x6e00>
.Lbranch_000000006638:
	s_mov_b32 s33, s97                                         ; 000000006638: BEA10061
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 00000000663C: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000006648: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000006650: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006654: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000006658: 98244024
	s_cselect_b32 s37, s37, s65                                ; 00000000665C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006660: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006664: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006668: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 00000000666C: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006670: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006674: 98278027
	s_barrier_signal -1                                        ; 000000006678: BE804EC1
	s_barrier_wait 0xffff                                      ; 00000000667C: BF94FFFF
	s_mov_b32 s33, s98                                         ; 000000006680: BEA10062
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT        ; 000000006684: D0310000 00100000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000006690: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000006698: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 00000000669C: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000066A0: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000066A4: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000066A8: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000066AC: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000066B0: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000066B4: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000066B8: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000066BC: 98278027
	s_barrier_signal -1                                        ; 0000000066C0: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000066C4: BF94FFFF
	s_branch .Lbranch_000000007158                                               ; 0000000066C8: BFA002A3 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x5858>
.Lbranch_0000000066cc:
	s_nop 0                                                    ; 0000000066CC: BF800000
	s_set_vgpr_msb 0                                           ; 0000000066D0: BF860000
	s_cmp_eq_u32 s92, 0                                        ; 0000000066D4: BF06805C
	s_cbranch_scc1 .Lbranch_0000000066f4                                           ; 0000000066D8: BFA20006 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4df4>
	s_cmp_eq_u32 s92, 1                                        ; 0000000066DC: BF06815C
	s_cbranch_scc1 .Lbranch_000000006788                                          ; 0000000066E0: BFA20029 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4e88>
	s_cmp_eq_u32 s92, 2                                        ; 0000000066E4: BF06825C
	s_cbranch_scc1 .Lbranch_00000000681c                                          ; 0000000066E8: BFA2004C <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4f1c>
	s_cmp_eq_u32 s92, 3                                        ; 0000000066EC: BF06835C
	s_cbranch_scc1 .Lbranch_0000000068b0                                         ; 0000000066F0: BFA2006F <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4fb0>
.Lbranch_0000000066f4:
	s_mov_b32 s33, s98                                         ; 0000000066F4: BEA10062
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000066F8: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000006704: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 00000000670C: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006710: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000006714: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000006718: 98254125
	s_cselect_b32 s38, s38, s66                                ; 00000000671C: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006720: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006724: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006728: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 00000000672C: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006730: 98278027
	s_barrier_signal -1                                        ; 000000006734: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006738: BF94FFFF
	s_mov_b32 s33, s95                                         ; 00000000673C: BEA1005F
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000006740: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 00000000674C: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000006754: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006758: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000675C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000006760: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006764: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006768: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 00000000676C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006770: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006774: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006778: 98278027
	s_barrier_signal -1                                        ; 00000000677C: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006780: BF94FFFF
	s_branch .Lbranch_0000000095a4                                              ; 000000006784: BFA00B87 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x7ca4>
.Lbranch_000000006788:
	s_mov_b32 s33, s95                                         ; 000000006788: BEA1005F
	tensor_load_to_lds s[32:35], s[36:43]                      ; 00000000678C: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 000000006798: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 0000000067A0: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000067A4: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000067A8: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000067AC: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000067B0: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000067B4: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000067B8: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000067BC: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000067C0: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000067C4: 98278027
	s_barrier_signal -1                                        ; 0000000067C8: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000067CC: BF94FFFF
	s_mov_b32 s33, s96                                         ; 0000000067D0: BEA10060
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000067D4: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 0000000067E0: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 0000000067E8: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000067EC: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000067F0: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000067F4: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000067F8: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000067FC: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006800: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006804: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006808: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 00000000680C: 98278027
	s_barrier_signal -1                                        ; 000000006810: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006814: BF94FFFF
	s_branch .Lbranch_000000009cdc                                              ; 000000006818: BFA00D30 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x83dc>
.Lbranch_00000000681c:
	s_mov_b32 s33, s96                                         ; 00000000681C: BEA10060
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000006820: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 00000000682C: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 000000006834: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006838: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 00000000683C: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000006840: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006844: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006848: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 00000000684C: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006850: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006854: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006858: 98278027
	s_barrier_signal -1                                        ; 00000000685C: BE804EC1
	s_barrier_wait 0xffff                                      ; 000000006860: BF94FFFF
	s_mov_b32 s33, s97                                         ; 000000006864: BEA10061
	tensor_load_to_lds s[32:35], s[36:43]                      ; 000000006868: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000006874: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 00000000687C: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006880: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000006884: 98244024
	s_cselect_b32 s37, s37, s65                                ; 000000006888: 98254125
	s_cselect_b32 s38, s38, s66                                ; 00000000688C: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006890: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006894: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 000000006898: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 00000000689C: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000068A0: 98278027
	s_barrier_signal -1                                        ; 0000000068A4: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000068A8: BF94FFFF
	s_branch .Lbranch_00000000a414                                              ; 0000000068AC: BFA00ED9 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x8b14>
.Lbranch_0000000068b0:
	s_mov_b32 s33, s97                                         ; 0000000068B0: BEA10061
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000068B4: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x300                               ; 0000000068C0: 8018FF3A 00000300
	s_cmp_lt_u32 s24, s70                                      ; 0000000068C8: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 0000000068CC: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 0000000068D0: 98244024
	s_cselect_b32 s37, s37, s65                                ; 0000000068D4: 98254125
	s_cselect_b32 s38, s38, s66                                ; 0000000068D8: 98264226
	s_cselect_b32 s70, s70, s71                                ; 0000000068DC: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 0000000068E0: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 0000000068E4: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 0000000068E8: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 0000000068EC: 98278027
	s_barrier_signal -1                                        ; 0000000068F0: BE804EC1
	s_barrier_wait 0xffff                                      ; 0000000068F4: BF94FFFF
	s_mov_b32 s33, s98                                         ; 0000000068F8: BEA10062
	tensor_load_to_lds s[32:35], s[36:43]                      ; 0000000068FC: D0310000 00000000 7C7C2420
	s_add_co_u32 s24, s58, 0x400                               ; 000000006908: 8018FF3A 00000400
	s_cmp_lt_u32 s24, s70                                      ; 000000006910: BF0A4618
	s_cselect_b64 s[34:35], s[34:35], s[62:63]                 ; 000000006914: 98A23E22
	s_cselect_b32 s36, s36, s64                                ; 000000006918: 98244024
	s_cselect_b32 s37, s37, s65                                ; 00000000691C: 98254125
	s_cselect_b32 s38, s38, s66                                ; 000000006920: 98264226
	s_cselect_b32 s70, s70, s71                                ; 000000006924: 98464746
	s_cselect_b64 s[26:27], s[56:57], 0                        ; 000000006928: 989A8038
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]                  ; 00000000692C: A9A21A22
	s_cmp_lt_u32 s24, s71                                      ; 000000006930: BF0A4718
	s_cselect_b32 s39, s39, 0                                  ; 000000006934: 98278027
	s_barrier_signal -1                                        ; 000000006938: BE804EC1
	s_barrier_wait 0xffff                                      ; 00000000693C: BF94FFFF
	s_branch .Lbranch_000000008e6c                                              ; 000000006940: BFA0094A <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x756c>
.Lbranch_000000007158:
	s_nop 0
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_mov_b32 s33, s95
	s_barrier_wait 0xffff
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT
	s_add_co_u32 s24, s58, 0x500
	s_cmp_lt_u32 s24, s70
	s_cselect_b64 s[34:35], s[34:35], s[62:63]
	s_cselect_b32 s36, s36, s64
	s_cselect_b32 s37, s37, s65
	s_cselect_b32 s38, s38, s66
	s_cselect_b32 s70, s70, s71
	s_cselect_b64 s[26:27], s[56:57], 0
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_cmp_lt_u32 s24, s71
	s_cselect_b32 s39, s39, 0
	s_addk_co_i32 s58, 0x100
	s_cmp_lt_i32 s58, s59
	s_cbranch_scc0 .Lbranch_000000008e4c
.Lbranch_000000007890:
	s_nop 0
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_mov_b32 s33, s96
	s_barrier_wait 0xffff
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT
	s_add_co_u32 s24, s58, 0x500
	s_cmp_lt_u32 s24, s70
	s_cselect_b64 s[34:35], s[34:35], s[62:63]
	s_cselect_b32 s36, s36, s64
	s_cselect_b32 s37, s37, s65
	s_cselect_b32 s38, s38, s66
	s_cselect_b32 s70, s70, s71
	s_cselect_b64 s[26:27], s[56:57], 0
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_cmp_lt_u32 s24, s71
	s_cselect_b32 s39, s39, 0
	s_addk_co_i32 s58, 0x100
	s_cmp_lt_i32 s58, s59
	s_cbranch_scc0 .Lbranch_000000008e54
.Lbranch_000000007fc8:
	s_nop 0
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_mov_b32 s33, s97
	s_barrier_wait 0xffff
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT
	s_add_co_u32 s24, s58, 0x500
	s_cmp_lt_u32 s24, s70
	s_cselect_b64 s[34:35], s[34:35], s[62:63]
	s_cselect_b32 s36, s36, s64
	s_cselect_b32 s37, s37, s65
	s_cselect_b32 s38, s38, s66
	s_cselect_b32 s70, s70, s71
	s_cselect_b64 s[26:27], s[56:57], 0
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_cmp_lt_u32 s24, s71
	s_cselect_b32 s39, s39, 0
	s_addk_co_i32 s58, 0x100
	s_cmp_lt_i32 s58, s59
	s_cbranch_scc0 .Lbranch_000000008e5c
.Lbranch_000000008700:
	s_nop 0
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_mov_b32 s33, s98
	s_barrier_wait 0xffff
	tensor_load_to_lds s[32:35], s[36:43] th:TH_LOAD_NT
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc0 .Lbranch_000000008c54
	s_barrier_signal -3
.Lbranch_000000008c54:
	s_add_co_u32 s24, s58, 0x500
	s_cmp_lt_u32 s24, s70
	s_cselect_b64 s[34:35], s[34:35], s[62:63]
	s_cselect_b32 s36, s36, s64
	s_cselect_b32 s37, s37, s65
	s_cselect_b32 s38, s38, s66
	s_cselect_b32 s70, s70, s71
	s_cselect_b64 s[26:27], s[56:57], 0
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_cmp_lt_u32 s24, s71
	s_cselect_b32 s39, s39, 0
	s_addk_co_i32 s58, 0x100
	s_cmp_lt_i32 s58, s59
	s_cbranch_scc0 .Lbranch_000000008e64
	s_barrier_wait 0xfffd
	s_branch .Lbranch_000000007158
.Lbranch_000000008e4c:
	s_mov_b32 s92, 0                                           ; 000000008E4C: BEDC0080
	s_branch .Lbranch_00000000ab74                                              ; 000000008E50: BFA00748 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9274>
.Lbranch_000000008e54:
	s_mov_b32 s92, 1                                           ; 000000008E54: BEDC0081
	s_branch .Lbranch_00000000ab74                                              ; 000000008E58: BFA00746 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9274>
.Lbranch_000000008e5c:
	s_mov_b32 s92, 2                                           ; 000000008E5C: BEDC0082
	s_branch .Lbranch_00000000ab74                                              ; 000000008E60: BFA00744 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9274>
.Lbranch_000000008e64:
	s_mov_b32 s92, 3                                           ; 000000008E64: BEDC0083
	s_branch .Lbranch_00000000ab74                                              ; 000000008E68: BFA00742 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9274>
.Lbranch_000000008e6c:
	s_nop 0
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_mov_b32 s33, s95
	s_barrier_wait 0xffff
	tensor_load_to_lds s[32:35], s[36:43]
	s_add_co_u32 s24, s58, 0x500
	s_cmp_lt_u32 s24, s70
	s_cselect_b64 s[34:35], s[34:35], s[62:63]
	s_cselect_b32 s36, s36, s64
	s_cselect_b32 s37, s37, s65
	s_cselect_b32 s38, s38, s66
	s_cselect_b32 s70, s70, s71
	s_cselect_b64 s[26:27], s[56:57], 0
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_cmp_lt_u32 s24, s71
	s_cselect_b32 s39, s39, 0
	s_addk_co_i32 s58, 0x100
	s_cmp_lt_i32 s58, s59
	s_cbranch_scc0 .Lbranch_00000000ab54
.Lbranch_0000000095a4:
	s_nop 0
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_mov_b32 s33, s96
	s_barrier_wait 0xffff
	tensor_load_to_lds s[32:35], s[36:43]
	s_add_co_u32 s24, s58, 0x500
	s_cmp_lt_u32 s24, s70
	s_cselect_b64 s[34:35], s[34:35], s[62:63]
	s_cselect_b32 s36, s36, s64
	s_cselect_b32 s37, s37, s65
	s_cselect_b32 s38, s38, s66
	s_cselect_b32 s70, s70, s71
	s_cselect_b64 s[26:27], s[56:57], 0
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_cmp_lt_u32 s24, s71
	s_cselect_b32 s39, s39, 0
	s_addk_co_i32 s58, 0x100
	s_cmp_lt_i32 s58, s59
	s_cbranch_scc0 .Lbranch_00000000ab5c
.Lbranch_000000009cdc:
	s_nop 0
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_mov_b32 s33, s97
	s_barrier_wait 0xffff
	tensor_load_to_lds s[32:35], s[36:43]
	s_add_co_u32 s24, s58, 0x500
	s_cmp_lt_u32 s24, s70
	s_cselect_b64 s[34:35], s[34:35], s[62:63]
	s_cselect_b32 s36, s36, s64
	s_cselect_b32 s37, s37, s65
	s_cselect_b32 s38, s38, s66
	s_cselect_b32 s70, s70, s71
	s_cselect_b64 s[26:27], s[56:57], 0
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_cmp_lt_u32 s24, s71
	s_cselect_b32 s39, s39, 0
	s_addk_co_i32 s58, 0x100
	s_cmp_lt_i32 s58, s59
	s_cbranch_scc0 .Lbranch_00000000ab64
.Lbranch_00000000a414:
	s_nop 0
	s_set_vgpr_msb 0
	s_wait_tensorcnt 0x2
	s_barrier_signal -1
	s_mov_b32 s33, s98
	s_barrier_wait 0xffff
	tensor_load_to_lds s[32:35], s[36:43]
	s_add_co_u32 s24, s58, 0x500
	s_cmp_lt_u32 s24, s70
	s_cselect_b64 s[34:35], s[34:35], s[62:63]
	s_cselect_b32 s36, s36, s64
	s_cselect_b32 s37, s37, s65
	s_cselect_b32 s38, s38, s66
	s_cselect_b32 s70, s70, s71
	s_cselect_b64 s[26:27], s[56:57], 0
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_cmp_lt_u32 s24, s71
	s_cselect_b32 s39, s39, 0
	s_addk_co_i32 s58, 0x100
	s_cmp_lt_i32 s58, s59
	s_cbranch_scc0 .Lbranch_00000000ab6c
	s_barrier_wait 0xfffd
	s_branch .Lbranch_000000008e6c
.Lbranch_00000000ab54:
	s_mov_b32 s92, 0                                           ; 00000000AB54: BEDC0080
	s_branch .Lbranch_00000000ab74                                                 ; 00000000AB58: BFA00006 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9274>
.Lbranch_00000000ab5c:
	s_mov_b32 s92, 1                                           ; 00000000AB5C: BEDC0081
	s_branch .Lbranch_00000000ab74                                                 ; 00000000AB60: BFA00004 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9274>
.Lbranch_00000000ab64:
	s_mov_b32 s92, 2                                           ; 00000000AB64: BEDC0082
	s_branch .Lbranch_00000000ab74                                                 ; 00000000AB68: BFA00002 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9274>
.Lbranch_00000000ab6c:
	s_mov_b32 s92, 3                                           ; 00000000AB6C: BEDC0083
	s_branch .Lbranch_00000000ab74                                                 ; 00000000AB70: BFA00000 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9274>
.Lbranch_00000000ab74:
	s_cmp_eq_u32 s92, 3                                        ; 00000000AB74: BF06835C
	s_cbranch_scc0 .Lbranch_00000000ab80                                           ; 00000000AB78: BFA10001 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x9280>
	s_barrier_wait 0xfffd                                      ; 00000000AB7C: BF94FFFD
.Lbranch_00000000ab80:
	s_set_vgpr_msb 0
	s_wait_idle
	s_wait_tensorcnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	s_set_vgpr_msb 0
	s_barrier_signal -1
	s_barrier_wait 0xffff
.Lbranch_00000000b7f0:
	; task-end workgroup rendezvous kept from the original epilogue
	s_barrier_signal -1
	s_barrier_wait 0xffff
	; cluster convergence before the next persistent task
	s_cmp_eq_u32 s22, 0
	s_cbranch_scc0 .Lclusterwait_task
	s_barrier_signal -3
.Lclusterwait_task:
	s_barrier_wait 0xfffd
	; restore the D pointer to the tile origin for the next persistent task
	s_mul_i32 s24, 0x100, s54
	s_lshl_b32 s24, s24, 1
	s_mul_i32 s25, 0x40, s55
	s_mul_i32 s25, s25, s12
	s_add_co_u32 s26, s25, s24
	s_mul_i32 s24, s22, 0x40
	s_lshl_b32 s24, s24, 1
	s_add_co_u32 s26, s26, s24
	s_sub_co_u32 s44, s44, s26
	s_sub_co_ci_u32 s45, s45, 0
	s_cmp_eq_u32 s60, 1                                        ; 00000000B888: BF06813C
	s_cbranch_scc1 .Lbranch_00000000ba60                                         ; 00000000B88C: BFA20074 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0xa160>
	s_set_vgpr_msb 0                                           ; 00000000B890: BF860000
	s_mov_b32 s54, s68                                         ; 00000000B894: BEB60044
	s_mov_b32 s55, s69                                         ; 00000000B898: BEB70045
	s_mul_i32 s24, s55, 0x40
	s_mul_i32 s24, s24, s13                                    ; 00000000B8A4: 96180D18
	s_add_co_u32 s72, s4, s24                                  ; 00000000B8A8: 80481804
	s_add_co_ci_u32 s73, 0, s5                                 ; 00000000B8AC: 82490580
	s_mul_i32 s24, s55, 0x40
	s_mul_i32 s24, s24, s15                                    ; 00000000B8B8: 96180F18
	s_add_co_u32 s76, s8, s24                                  ; 00000000B8BC: 804C1808
	s_add_co_ci_u32 s77, 0, s9                                 ; 00000000B8C0: 824D0980
	s_mul_i32 s24, s54, 0x100
	s_mul_i32 s24, s24, s14                                    ; 00000000B8CC: 96180E18
	s_add_co_u32 s74, s6, s24                                  ; 00000000B8D0: 804A1806
	s_add_co_ci_u32 s75, 0, s7                                 ; 00000000B8D4: 824B0780
	s_mul_i32 s24, s54, 0x100
	s_mul_i32 s24, s24, s16                                    ; 00000000B8E0: 96181018
	s_add_co_u32 s78, s10, s24                                 ; 00000000B8E4: 804E180A
	s_add_co_ci_u32 s79, 0, s11                                ; 00000000B8E8: 824F0B80
	s_mul_i32 s24, 0x100, s54
	s_lshl_b32 s24, s24, 1                                     ; 00000000B8F4: 84188118
	s_mul_i32 s25, 0x40, s55
	s_mul_i32 s25, s25, s12                                    ; 00000000B900: 96190C19
	s_add_co_u32 s26, s25, s24                                 ; 00000000B904: 801A1819
	s_mul_i32 s24, s22, 0x40
	s_lshl_b32 s24, s24, 1
	s_add_co_u32 s26, s26, s24                                 ; 00000000B92C: 801A181A
	s_add_co_u32 s44, s44, s26                                 ; 00000000B930: 802C1A2C
	s_add_co_ci_u32 s45, 0, s45                                ; 00000000B934: 822D2D80
	s_mov_b32 s58, 0                                           ; 00000000B938: BEBA0080
	s_mov_b32 s70, s19                                         ; 00000000B93C: BEC60013
	s_add_co_i32 s24, s20, s21                                 ; 00000000B940: 81181514
	s_lshl_b32 s24, 1, s24                                     ; 00000000B944: 84181881
	s_add_co_u32 s28, s28, s24                                 ; 00000000B948: 801C181C
	s_cmp_lt_u32 s28, s29                                      ; 00000000B94C: BF0A1D1C
	s_cselect_b32 s60, 0, 1                                    ; 00000000B950: 983C8180
	s_cbranch_scc0 .Lbranch_00000000ba20                                          ; 00000000B954: BFA10032 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0xa120>
	s_mov_b32 s24, s61                                         ; 00000000B958: BE98003D
	s_sub_co_u32 s25, s24, 1                                   ; 00000000B95C: 80998118
	s_and_b32 s26, s24, s25                                    ; 00000000B960: 8B1A1918
	s_cmp_eq_u32 s26, 0                                        ; 00000000B964: BF06801A
	s_cbranch_scc0 .Lbranch_00000000b98c                                           ; 00000000B968: BFA10008 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0xa08c>
	s_ctz_i32_b32 s26, s24                                     ; 00000000B96C: BE9A0818
	s_lshr_b32 s27, s28, s26                                   ; 00000000B970: 851B1A1C
	s_and_b32 s24, s28, s25                                    ; 00000000B974: 8B18191C
	s_mul_i32 s25, s27, s52                                    ; 00000000B978: 9619341B
	s_add_co_u32 s69, s25, s50                                 ; 00000000B97C: 80453219
	s_mul_i32 s25, s24, s51                                    ; 00000000B980: 96193318
	s_add_co_u32 s68, s25, s49                                 ; 00000000B984: 80443119
	s_branch .Lbranch_00000000ba28                                                ; 00000000B988: BFA00027 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0xa128>
.Lbranch_00000000b98c:
	v_cvt_f32_u32_e32 v4, s24                                  ; 00000000B98C: 7E080C18
	s_sub_co_i32 s26, 0, s24                                   ; 00000000B990: 819A1880
	v_rcp_iflag_f32_e32 v4, v4                                 ; 00000000B994: 7E085704
	s_nop 0                                                    ; 00000000B998: BF800000
	v_mul_f32_e32 v4, 0x4f7ffffe, v4                           ; 00000000B99C: 100808FF 4F7FFFFE
	v_cvt_u32_f32_e32 v4, v4                                   ; 00000000B9A4: 7E080F04
	v_mul_lo_u32 v5, s26, v4                                   ; 00000000B9A8: D72C0005 0002081A
	v_mul_hi_u32 v5, v4, v5                                    ; 00000000B9B0: D72D0005 00020B04
	v_add_nc_u32_e32 v4, v4, v5                                ; 00000000B9B8: 4A080B04
	v_mul_hi_u32 v4, s28, v4                                   ; 00000000B9BC: D72D0004 0002081C
	v_mul_lo_u32 v5, v4, s24                                   ; 00000000B9C4: D72C0005 00003104
	v_sub_nc_u32_e32 v7, s28, v5                               ; 00000000B9CC: 4C0E0A1C
	v_add_nc_u32_e32 v6, 1, v4                                 ; 00000000B9D0: 4A0C0881
	v_cmp_le_u32_e32 vcc_lo, s24, v7                           ; 00000000B9D4: 7C960E18
	v_subrev_nc_u32_e32 v5, s24, v7                            ; 00000000B9D8: 4E0A0E18
	s_nop 0                                                    ; 00000000B9DC: BF800000
	v_cndmask_b32_e32 v4, v4, v6, vcc_lo                       ; 00000000B9E0: 02080D04
	v_cndmask_b32_e32 v7, v7, v5, vcc_lo                       ; 00000000B9E4: 020E0B07
	v_add_nc_u32_e32 v5, 1, v4                                 ; 00000000B9E8: 4A0A0881
	v_cmp_le_u32_e32 vcc_lo, s24, v7                           ; 00000000B9EC: 7C960E18
	s_nop 1                                                    ; 00000000B9F0: BF800001
	v_cndmask_b32_e32 v7, v4, v5, vcc_lo                       ; 00000000B9F4: 020E0B04
	s_nop 3                                                    ; 00000000B9F8: BF800003
	v_readfirstlane_b32 s27, v7                                ; 00000000B9FC: 7E360507
	s_nop 3                                                    ; 00000000BA00: BF800003
	s_mul_i32 s25, s27, s24                                    ; 00000000BA04: 9619181B
	s_sub_co_u32 s24, s28, s25                                 ; 00000000BA08: 8098191C
	s_mul_i32 s25, s27, s52                                    ; 00000000BA0C: 9619341B
	s_add_co_u32 s69, s25, s50                                 ; 00000000BA10: 80453219
	s_mul_i32 s25, s24, s51                                    ; 00000000BA14: 96193318
	s_add_co_u32 s68, s25, s49                                 ; 00000000BA18: 80443119
	s_branch .Lbranch_00000000ba28                                                 ; 00000000BA1C: BFA00002 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0xa128>
.Lbranch_00000000ba20:
	s_mov_b32 s68, s54                                         ; 00000000BA20: BEC40036
	s_mov_b32 s69, s55                                         ; 00000000BA24: BEC50037
.Lbranch_00000000ba28:
	s_add_nc_u64 s[34:35], s[34:35], s[56:57]                  ; 00000000BA28: A9A23822
	s_add_nc_u64 s[34:35], s[34:35], s[56:57]                  ; 00000000BA2C: A9A23822
	s_mov_b32 s36, s64                                         ; 00000000BA30: BEA40040
	s_mov_b32 s37, s65                                         ; 00000000BA34: BEA50041
	s_mov_b32 s38, s66                                         ; 00000000BA38: BEA60042
	s_mov_b32 s39, s67                                         ; 00000000BA3C: BEA70043
	s_cmp_eq_u32 s22, 0                                        ; 00000000BA40: BF068016
	s_cbranch_scc1 .Lbranch_000000005060                                       ; 00000000BA44: BFA2E586 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x3760>
	s_cmp_eq_u32 s22, 1                                        ; 00000000BA48: BF068116
	s_cbranch_scc1 .Lbranch_00000000554c                                       ; 00000000BA4C: BFA2E6BF <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x3c4c>
	s_cmp_eq_u32 s22, 2                                        ; 00000000BA50: BF068216
	s_cbranch_scc1 .Lbranch_000000005a58                                       ; 00000000BA54: BFA2E800 <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4158>
	s_cmp_eq_u32 s22, 3                                        ; 00000000BA58: BF068316
	s_cbranch_scc1 .Lbranch_000000005f48                                       ; 00000000BA5C: BFA2E93A <f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_ps+0x4648>
.Lbranch_00000000ba60:
	s_wait_idle                                                ; 00000000BA60: BF8A0000
	s_endpgm                                                   ; 00000000BA64: BFB00000

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps
		.amdhsa_group_segment_fixed_size 131072
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 120
		.amdhsa_user_sgpr_count 32
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 30
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 384
		.amdhsa_next_free_sgpr 104
		.amdhsa_named_barrier_count 0
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 0
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		; Hint only, deliberately sized below the reduced load-only .text so the
		; window never runs past the end of the segment.
		.amdhsa_inst_pref_size 65
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
	.size	f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps, .Lfunc_end0-f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps

	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.num_vgpr, 384
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.num_agpr, 0
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.numbered_sgpr, 104
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.num_named_barrier, 0
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.private_seg_size, 0
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.uses_vcc, 1
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.uses_flat_scratch, 0
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.has_dyn_sized_stack, 0
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.has_recursion, 0
	.set f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.has_indirect_call, 0
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
      - .actual_access:  read_write
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
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
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           8
        .value_kind:     by_value
      - .offset:         88
        .size:           8
        .value_kind:     by_value
      - .offset:         96
        .size:           8
        .value_kind:     by_value
      - .offset:         104
        .size:           8
        .value_kind:     by_value
      - .offset:         112
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 120
    .max_flat_workgroup_size: 128
    .name:           f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .symbol:         f4gemm_bf16_mxfp4_ABpreShuffle_64x256_1x4_batch_ps.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     384
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
