// Generated from mab_tdm_gemm_full_batch.s by
// mab_tdm_full_batch_loadonly_audit.py.
// Keeps the original TDM producer/control/synchronization plan and resources;
// removes LDS consumers, WMMA, C/D access, conversion, and stores.

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

.text
.globl mab_tdm_gemm_full_batch_loadonly
.protected mab_tdm_gemm_full_batch_loadonly
.type mab_tdm_gemm_full_batch_loadonly,@function
.p2align 8
mab_tdm_gemm_full_batch_loadonly:
	s_version UC_VERSION_GFX12|UC_VERSION_W32_BIT              // 000000001E00: B0804009
	s_mov_b32 s90, 0                                           // 000000001E04: BEDA0080
	s_setreg_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 3), s90      // 000000001E08: B95A101A
	v_mov_b32_e32 v224, v0                                     // 000000001E14: 7FC00300
	s_mov_b32 vcc_hi, 0                                        // 000000001E18: BEEB0080
	v_readfirstlane_b32 s90, v224                              // 000000001E1C: 7EB405E0
	s_lshr_b32 s67, s90, 5                                     // 000000001E20: 8543855A
	s_mov_b32 s90, ttmp6                                       // 000000001E24: BEDA0072
	s_mov_b32 s91, ttmp7                                       // 000000001E28: BEDB0073
	s_mov_b32 s92, ttmp9                                       // 000000001E2C: BEDC0075
	s_and_b32 s93, s90, 15                                     // 000000001E30: 8B5D8F5A
	s_bfe_u32 s94, s90, 0x40004                                // 000000001E34: 935EFF5A 00040004
	s_and_b32 s91, s91, 0xffff                                 // 000000001E3C: 8B5BFF5B 0000FFFF
	s_mul_i32 s92, s92, 1                                      // 000000001E44: 965C815C
	s_add_co_i32 s30, s92, s93                                 // 000000001E48: 811E5D5C
	s_mul_i32 s91, s91, 4                                      // 000000001E4C: 965B845B
	s_add_co_i32 s31, s91, s94                                 // 000000001E50: 811F5E5B
	s_and_b32 s32, ttmp7, 0xffff0000                           // 000000001E54: 8B20FF73 FFFF0000
	s_lshr_b32 s32, s32, 16                                    // 000000001E5C: 85209020
	s_lshl_b32 s64, 15, s93                                    // 000000001E60: 84405D8F
	s_mul_i32 s90, s94, 1                                      // 000000001E64: 965A815E
	s_lshl_b32 s65, 1, s90                                     // 000000001E68: 84415A81
	s_mov_b32 s62, 0x4ca0                                      // 000000001E6C: BEBE00FF 00004CA0
	s_mul_i32 s63, s62, 4                                      // 000000001E74: 963F843E
	s_mov_b64 s[58:59], 0                                      // 000000001E7C: BEBA0180
	s_barrier_signal -1                                        // 000000001E80: BE804EC1
	s_barrier_wait 0xffff                                      // 000000001F08: BF94FFFF
	s_cmp_eq_u32 s67, 0                                        // 000000001F0C: BF068043
	s_cbranch_scc0 .Lloadonly_0118                                           // 000000001F10: BFA10001 <MAB_TDMs+0x118>
	s_barrier_signal -3                                        // 000000001F14: BE804EC3
.Lloadonly_0118:
	s_lshl1_add_u32 s10, s58, s10                              // 000000001FB0: 870A0A3A
	s_add_co_ci_u32 s11, s11, 0                                // 000000001FB4: 820B800B
	s_lshl1_add_u32 s12, s59, s12                              // 000000001FB8: 870C0C3B
	s_add_co_ci_u32 s13, s13, 0                                // 000000001FBC: 820D800D
	s_cmp_eq_f32 s14, 0                                        // 000000001FC0: BF42800E
	s_cbranch_scc0 .Lloadonly_01cc                                           // 000000001FC4: BFA10001 <MAB_TDMs+0x1cc>
	s_mov_b32 s27, 0                                           // 000000001FC8: BE9B0080
.Lloadonly_01cc:
	s_mov_b32 s88, 0                                           // 000000002034: BED80080
	s_mul_hi_u32 s95, s30, 16                                  // 000000002040: 96DF901E
	s_mul_i32 s94, s30, 16                                     // 000000002044: 965E901E
	s_mul_hi_u32 s95, s94, s20                                 // 000000002048: 96DF145E
	s_mul_i32 s94, s94, s20                                    // 00000000204C: 965E145E
	s_mul_hi_u32 s93, s21, s32                                 // 000000002050: 96DD2015
	s_mul_i32 s92, s21, s32                                    // 000000002054: 965C2015
	s_add_co_u32 s94, s94, s92                                 // 000000002058: 805E5C5E
	s_add_co_ci_u32 s95, s95, s93                              // 00000000205C: 825F5D5F
	s_lshl_b64 s[94:95], s[94:95], 1                           // 000000002060: 84DE815E
	s_add_co_u32 s36, s10, s94                                 // 000000002064: 80245E0A
	s_add_co_ci_u32 s37, s11, s95                              // 000000002068: 82255F0B
	s_mul_hi_u32 s95, s31, 0x100                               // 00000000206C: 96DFFF1F 00000100
	s_mul_i32 s94, s31, 0x100                                  // 000000002074: 965EFF1F 00000100
	s_mul_hi_u32 s95, s94, s22                                 // 00000000207C: 96DF165E
	s_mul_i32 s94, s94, s22                                    // 000000002080: 965E165E
	s_mul_hi_u32 s93, s23, s32                                 // 000000002084: 96DD2017
	s_mul_i32 s92, s23, s32                                    // 000000002088: 965C2017
	s_add_co_u32 s94, s94, s92                                 // 00000000208C: 805E5C5E
	s_add_co_ci_u32 s95, s95, s93                              // 000000002090: 825F5D5F
	s_lshl_b64 s[94:95], s[94:95], 1                           // 000000002094: 84DE815E
	s_add_co_u32 s40, s12, s94                                 // 000000002098: 80285E0C
	s_add_co_ci_u32 s41, s13, s95                              // 00000000209C: 82295F0D
	s_mov_b32 s53, 0x100                                       // 0000000020A0: BEB500FF 00000100
	s_mov_b32 s54, 0x100                                       // 0000000020A8: BEB600FF 00000100
	s_mov_b32 s60, 0                                           // 0000000020B0: BEBC0080
	s_lshr_b32 s33, s27, 7                                     // 000000002154: 8521871B
	s_barrier_wait 0xfffd                                      // 00000000215C: BF94FFFD
	s_cmp_eq_u32 s33, 0                                        // 000000002160: BF068021
	s_cbranch_scc1 .Lloadonly_04d4                                          // 000000002164: BFA2005B <MAB_TDMs+0x4d4>
	s_mov_b32 s68, 1                                           // 000000002168: BEC40081
	s_bitcmp1_b32 s67, 1                                       // 00000000216C: BF0D8143
	s_cbranch_scc1 .Lloadonly_0404                                          // 000000002170: BFA20024 <MAB_TDMs+0x404>
	s_mov_b32 s96, 0                                           // 000000002178: BEE00080
	s_and_b32 s94, s67, 1                                      // 00000000217C: 8B5E8143
	s_lshl_b32 s95, s63, s94                                   // 000000002180: 845F5E3F
	s_cmp_eq_u32 s94, 1                                        // 000000002184: BF06815E
	s_cmov_b32 s96, s95                                        // 000000002188: BEE0025F
	s_mov_b32 s69, s96                                         // 00000000218C: BEC50060
	s_lshl_b32 s70, s88, 1                                     // 000000002190: 84468158
	s_add_co_u32 s70, s36, s70                                 // 000000002194: 80464624
	s_add_co_ci_u32 s71, s37, 0                                // 000000002198: 82478025
	s_or_b32 s71, s71, 0x80000000                              // 00000000219C: 8C47FF47 80000000
	s_or_b32 s72, s64, 0x7590000                               // 0000000021A4: 8C48FF40 07590000
	s_pack_ll_b32_b16 s73, 0, s27                              // 0000000021AC: 99491B80
	s_pack_hl_b32_b16 s74, s27, s24                            // 0000000021B0: 9ACA181B
	s_pack_hl_b32_b16 s75, s24, 0x80                           // 0000000021B4: 9ACBFF18 00000080
	s_mov_b32 s76, 8                                           // 0000000021BC: BECC0088
	s_mov_b32 s77, s20                                         // 0000000021C0: BECD0014
	s_mov_b64 s[78:79], 0                                      // 0000000021C4: BECE0180
	s_mov_b32 s80, 0                                           // 0000000021C8: BED00080
	s_lshr_b32 s94, s63, 1                                     // 0000000021CC: 855E813F
	s_mov_b32 s81, s94                                         // 0000000021D0: BED1005E
	s_lshl_b32 s82, s20, 3                                     // 0000000021D4: 84528314
	s_mov_b32 s83, 0x10000                                     // 0000000021D8: BED300FF 00010000
	s_mov_b32 s84, 0                                           // 0000000021E0: BED40080
	s_mov_b32 s85, 0                                           // 0000000021E4: BED50080
	s_mov_b32 s86, 0                                           // 0000000021E8: BED60080
	s_mov_b32 s87, 0                                           // 0000000021EC: BED70080
	tensor_load_to_lds s[68:71], s[72:79], s[80:83], s[84:87]  // 0000000021F0: D0310000 00000000 54504844
	s_mov_b32 s97, s53                                         // 0000000021FC: BEE10035
	s_branch .Lloadonly_049c                                                // 000000002200: BFA00026 <MAB_TDMs+0x49c>
.Lloadonly_0404:
	s_mov_b32 s96, 0                                           // 000000002204: BEE00080
	s_and_b32 s94, s67, 1                                      // 000000002208: 8B5E8143
	s_lshl_b32 s95, s63, s94                                   // 00000000220C: 845F5E3F
	s_cmp_eq_u32 s94, 1                                        // 000000002210: BF06815E
	s_cmov_b32 s96, s95                                        // 000000002214: BEE0025F
	s_add_co_u32 s96, s96, 0x880                               // 000000002218: 8060FF60 00000880
	s_lshl_b32 s95, s94, 7                                     // 000000002220: 845F875E
	s_mul_i32 s92, s22, s95                                    // 000000002224: 965C5F16
	s_mov_b32 s69, s96                                         // 000000002228: BEC50060
	s_lshl_b32 s70, s92, 1                                     // 00000000222C: 8446815C
	s_add_co_u32 s70, s40, s70                                 // 000000002230: 80464628
	s_add_co_ci_u32 s71, s41, 0                                // 000000002234: 82478029
	s_or_b32 s71, s71, 0x80000000                              // 000000002238: 8C47FF47 80000000
	s_or_b32 s72, s65, 0x7590000                               // 000000002240: 8C48FF41 07590000
	s_pack_ll_b32_b16 s73, 0, s27                              // 000000002248: 99491B80
	s_pack_hl_b32_b16 s74, s27, s25                            // 00000000224C: 9ACA191B
	s_pack_hl_b32_b16 s75, s25, 0x80                           // 000000002250: 9ACBFF19 00000080
	s_mov_b32 s76, 64                                          // 000000002258: BECC00C0
	s_mov_b32 s77, s22                                         // 00000000225C: BECD0016
	s_mov_b64 s[78:79], 0                                      // 000000002260: BECE0180
	s_mov_b32 s80, 0                                           // 000000002264: BED00080
	s_lshr_b32 s94, s63, 1                                     // 000000002268: 855E813F
	s_mov_b32 s81, s94                                         // 00000000226C: BED1005E
	s_lshl_b32 s82, s22, 6                                     // 000000002270: 84528616
	s_mov_b32 s83, 0x10000                                     // 000000002274: BED300FF 00010000
	s_mov_b32 s84, 0                                           // 00000000227C: BED40080
	s_mov_b32 s85, 0                                           // 000000002280: BED50080
	s_mov_b32 s86, 0                                           // 000000002284: BED60080
	s_mov_b32 s87, 0                                           // 000000002288: BED70080
	tensor_load_to_lds s[68:71], s[72:79], s[80:83], s[84:87]  // 00000000228C: D0310000 00000000 54504844
	s_mov_b32 s97, s54                                         // 000000002298: BEE10036
.Lloadonly_049c:
	s_add_co_u32 s60, s60, s62                                 // 00000000229C: 803C3E3C
	s_cmp_ge_u32 s60, s63                                      // 0000000022A0: BF093F3C
	s_cmov_b32 s60, 0                                          // 0000000022A4: BEBC0280
	s_cmp_eq_u32 s33, 1                                        // 0000000022A8: BF068121
	s_cbranch_scc1 .Lloadonly_04d4                                           // 0000000022AC: BFA20009 <MAB_TDMs+0x4d4>
	s_add_co_u32 s69, s96, s60                                 // 0000000022B0: 80453C60
	s_add_co_u32 s70, s70, s97                                 // 0000000022B4: 80466146
	s_add_co_ci_u32 s71, s71, 0                                // 0000000022B8: 82478047
	tensor_load_to_lds s[68:71], s[72:79], s[80:83], s[84:87]  // 0000000022BC: D0310000 00000000 54504844
	s_add_co_u32 s60, s60, s62                                 // 0000000022C8: 803C3E3C
	s_cmp_ge_u32 s60, s63                                      // 0000000022CC: BF093F3C
	s_cmov_b32 s60, 0                                          // 0000000022D0: BEBC0280
.Lloadonly_04d4:
	s_add_co_u32 s69, s96, s60                                 // 00000000245C: 80453C60
	s_add_co_u32 s70, s70, s97                                 // 000000002460: 80466146
	s_add_co_ci_u32 s71, s71, 0                                // 000000002464: 82478047
	tensor_load_to_lds s[68:71], s[72:79], s[80:83], s[84:87]  // 000000002468: D0310000 00000000 54504844
	s_add_co_u32 s60, s60, s62                                 // 000000002474: 803C3E3C
	s_cmp_ge_u32 s60, s63                                      // 000000002478: BF093F3C
	s_cmov_b32 s60, 0                                          // 00000000247C: BEBC0280
	s_cmp_eq_u32 s33, 0                                        // 000000002480: BF068021
	s_cbranch_scc0 .Lloadonly_068c                                           // 000000002484: BFA10001 <MAB_TDMs+0x68c>
	s_branch .Lloadonly_0f80                                               // 000000002488: BFA0023D <MAB_TDMs+0xf80>
.Lloadonly_068c:
	s_wait_tensorcnt 0x2                                       // 00000000248C: BFCB0002
	s_barrier_signal -1                                        // 000000002494: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002498: BF94FFFF
	s_cmp_eq_u32 s33, 1                                        // 000000002560: BF068121
	s_cbranch_scc1 .Lloadonly_0c4c                                         // 000000002564: BFA20139 <MAB_TDMs+0xc4c>
	s_cmp_le_u32 s33, 2                                        // 000000002568: BF0B8221
	s_cbranch_scc1 .Lloadonly_09fc                                         // 00000000256C: BFA200A3 <MAB_TDMs+0x9fc>
	s_mov_b32 s90, 2                                           // 000000002570: BEDA0082
	s_setreg_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 3), s90      // 000000002574: B95A101A
.Lloadonly_0778:
	s_add_co_u32 s69, s96, s60                                 // 000000002578: 80453C60
	s_add_co_u32 s70, s70, s97                                 // 00000000257C: 80466146
	s_add_co_ci_u32 s71, s71, 0                                // 000000002580: 82478047
	s_wait_alu depctr_sa_sdst(0)                               // 000000002584: BF88FF9E
	tensor_load_to_lds s[68:71], s[72:79], s[80:83], s[84:87]  // 000000002588: D0310000 00000000 54504844
	s_wait_tensorcnt 0x2                                       // 000000002660: BFCB0002
	s_barrier_signal -1                                        // 000000002664: BE804EC1
	s_barrier_wait 0xffff                                      // 0000000026C4: BF94FFFF
	s_cmp_eq_u32 s67, 0                                        // 0000000026C8: BF068043
	s_wait_alu depctr_sa_sdst(0)                               // 0000000026CC: BF88FF9E
	s_cbranch_scc0 .Lloadonly_08d8                                           // 0000000026D0: BFA10001 <MAB_TDMs+0x8d8>
	s_barrier_signal -3                                        // 0000000026D4: BE804EC3
.Lloadonly_08d8:
	s_add_co_u32 s60, s60, s62                                 // 0000000027D4: 803C3E3C
	s_cmp_ge_u32 s60, s63                                      // 0000000027D8: BF093F3C
	s_cmov_b32 s60, 0                                          // 0000000027DC: BEBC0280
	s_barrier_wait 0xfffd                                      // 0000000027E0: BF94FFFD
	s_sub_co_u32 s33, s33, 1                                   // 0000000027E4: 80A18121
	s_cmp_eq_i32 s33, 3                                        // 0000000027E8: BF008321
	s_cmov_b32 s68, 0                                          // 0000000027EC: BEC40280
	s_cmp_eq_i32 s33, 2                                        // 0000000027F0: BF008221
	s_wait_alu depctr_sa_sdst(0)                               // 0000000027F4: BF88FF9E
	s_cbranch_scc0 .Lloadonly_0778                                       // 0000000027F8: BFA1FF5F <MAB_TDMs+0x778>
.Lloadonly_09fc:
	s_mov_b32 s90, 0                                           // 0000000027FC: BEDA0080
	s_setreg_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 3), s90      // 000000002800: B95A101A
	s_wait_tensorcnt 0x0                                       // 0000000028D0: BFCB0000
	s_barrier_signal -1                                        // 0000000028D4: BE804EC1
	s_barrier_wait 0xffff                                      // 000000002930: BF94FFFF
	s_cmp_eq_u32 s67, 0                                        // 000000002934: BF068043
	s_wait_alu depctr_sa_sdst(0)                               // 000000002938: BF88FF9E
	s_cbranch_scc0 .Lloadonly_0b44                                           // 00000000293C: BFA10001 <MAB_TDMs+0xb44>
	s_barrier_signal -3                                        // 000000002940: BE804EC3
.Lloadonly_0b44:
	s_add_co_u32 s60, s60, s62                                 // 000000002A3C: 803C3E3C
	s_cmp_ge_u32 s60, s63                                      // 000000002A40: BF093F3C
	s_cmov_b32 s60, 0                                          // 000000002A44: BEBC0280
	s_barrier_wait 0xfffd                                      // 000000002A48: BF94FFFD
.Lloadonly_0c4c:
.Lloadonly_0f80:
	s_endpgm                                                   // 000000002D80: BFB00000
	s_code_end                                                 // 000000002D84: BF9F0000
	s_code_end                                                 // 000000002D88: BF9F0000
	s_code_end                                                 // 000000002D8C: BF9F0000
	s_code_end                                                 // 000000002D90: BF9F0000
	s_code_end                                                 // 000000002D94: BF9F0000

.Lfunc_end0:
.size mab_tdm_gemm_full_batch_loadonly, .Lfunc_end0-mab_tdm_gemm_full_batch_loadonly

// The original object has a deliberately asymmetric loader contract:
// - the 64-byte kernel descriptor encodes kernarg_size=112 and fixed LDS=0;
// - NT_AMDGPU_METADATA supplies the 256 KiB fixed group segment used by HIP.
// Its published 160-byte argument list was stale: the original launcher
// actually passes the 112-byte layout recorded below.  Keeping the descriptor
// resource words and the launch ABI byte-for-byte faithful is required for the
// preloaded SGPR mapping used by this instruction stream.
.section .rodata,"a",@progbits
.p2align 6
.amdhsa_kernel mab_tdm_gemm_full_batch_loadonly
	.amdhsa_group_segment_fixed_size 0
	.amdhsa_private_segment_fixed_size 0
	.amdhsa_kernarg_size 112
	.amdhsa_user_sgpr_count 30
	.amdhsa_user_sgpr_dispatch_ptr 0
	.amdhsa_user_sgpr_queue_ptr 0
	.amdhsa_user_sgpr_kernarg_segment_ptr 1
	.amdhsa_user_sgpr_dispatch_id 0
	.amdhsa_user_sgpr_kernarg_preload_length 28
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
	.amdhsa_next_free_vgpr 1024
	.amdhsa_next_free_sgpr 98
	.amdhsa_named_barrier_count 0
	.amdhsa_reserve_vcc 1
	.amdhsa_float_round_mode_32 0
	.amdhsa_float_round_mode_16_64 0
	.amdhsa_float_denorm_mode_32 0
	.amdhsa_float_denorm_mode_16_64 3
	.amdhsa_fp16_overflow 0
	.amdhsa_memory_ordered 1
	.amdhsa_forward_progress 1
	.amdhsa_inst_pref_size 30
	.amdhsa_round_robin_scheduling 0
	.amdhsa_exception_fp_ieee_invalid_op 0
	.amdhsa_exception_fp_denorm_src 0
	.amdhsa_exception_fp_ieee_div_zero 0
	.amdhsa_exception_fp_ieee_overflow 0
	.amdhsa_exception_fp_ieee_underflow 0
	.amdhsa_exception_fp_ieee_inexact 0
	.amdhsa_exception_int_div_zero 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           8
        .value_kind:     by_value
        .value_type:     u64
      - .offset:         8
        .size:           8
        .value_kind:     by_value
        .value_type:     u64
      - .address_space:  generic
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .address_space:  generic
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .address_space:  generic
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .address_space:  generic
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
        .value_type:     struct
      - .offset:         48
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .offset:         52
        .size:           4
        .value_kind:     by_value
        .value_type:     f32
      - .offset:         56
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         60
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         64
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         68
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         72
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         76
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         80
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         84
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         88
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         92
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         96
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         100
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         104
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
      - .offset:         108
        .size:           4
        .value_kind:     by_value
        .value_type:     u32
    .group_segment_fixed_size: 327680
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
    .max_flat_workgroup_size: 128
    .name:           mab_tdm_gemm_full_batch_loadonly
    .private_segment_fixed_size: 0
    .sgpr_count:     97
    .symbol:         mab_tdm_gemm_full_batch_loadonly.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     1024
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version:
  - 1
  - 2
...
.end_amdgpu_metadata
