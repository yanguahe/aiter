; ModuleID = '/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT.cu'
source_filename = "/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g2_a4w4_NE385_H7168_E512_TOPK9_BM16_ATOMIC_NT.cu"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%"struct.aiter::mxfp4_moe::gemm2::LDSLayout" = type { %union.anon }
%union.anon = type { [4096 x float] }
%struct.__hip_bfloat16 = type { %union.anon.17 }
%union.anon.17 = type { i16 }

$_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph = comdat any

$_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds = comdat any

@_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds = linkonce_odr hidden addrspace(3) global %"struct.aiter::mxfp4_moe::gemm2::LDSLayout" undef, comdat, align 16
@__hip_cuid_f2b2fffd0fbfad1 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_f2b2fffd0fbfad1 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_ZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16Ph(ptr addrspace(1) noalias noundef %0, ptr addrspace(1) noalias noundef %1, ptr addrspace(1) noalias noundef %2, ptr addrspace(1) noalias noundef %3, ptr addrspace(1) noalias nocapture noundef readonly %4, ptr addrspace(1) noalias nocapture noundef readonly %5, ptr addrspace(1) noalias noundef %6, ptr addrspace(1) noalias noundef %7, i32 noundef %8, ptr addrspace(1) noalias noundef %9, ptr addrspace(1) noalias nocapture noundef readnone %10) local_unnamed_addr #0 comdat {
  %12 = ptrtoint ptr addrspace(1) %0 to i64
  %13 = inttoptr i64 %12 to ptr
  %14 = ptrtoint ptr addrspace(1) %1 to i64
  %15 = inttoptr i64 %14 to ptr
  %16 = ptrtoint ptr addrspace(1) %2 to i64
  %17 = inttoptr i64 %16 to ptr
  %18 = ptrtoint ptr addrspace(1) %3 to i64
  %19 = inttoptr i64 %18 to ptr
  %20 = ptrtoint ptr addrspace(1) %6 to i64
  %21 = inttoptr i64 %20 to ptr
  %22 = ptrtoint ptr addrspace(1) %7 to i64
  %23 = inttoptr i64 %22 to ptr
  %24 = ptrtoint ptr addrspace(1) %9 to i64
  %25 = inttoptr i64 %24 to ptr
  %26 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %27 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %28 = icmp samesign ult i32 %27, 256
  tail call void @llvm.assume(i1 %28)
  %29 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %27)
  %30 = lshr i32 %29, 6
  %31 = and i32 %27, 63
  %32 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %13, i16 0, i32 range(i32 20971520, 706478081) 167772160, i32 131072)
  %33 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %17, i16 0, i32 range(i32 20971520, 706478081) 706478080, i32 131072)
  %34 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %15, i16 0, i32 range(i32 20971520, 706478081) 20971520, i32 131072)
  %35 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %19, i16 0, i32 range(i32 20971520, 706478081) 44154880, i32 131072)
  %36 = load i32, ptr addrspace(1) %5, align 4, !tbaa !7
  %37 = sdiv i32 %36, 16
  %38 = mul nsw i32 %37, 28
  %39 = icmp slt i32 %26, %38
  br i1 %39, label %40, label %316

40:                                               ; preds = %11
  %41 = sdiv i32 %26, 28
  %42 = mul i32 %41, 28
  %43 = sub i32 %26, %42
  %44 = sext i32 %41 to i64
  %45 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %44
  %46 = load i32, ptr addrspace(1) %45, align 4, !tbaa !7
  %47 = shl nsw i32 %41, 4
  %48 = icmp samesign ult i32 %46, 385
  tail call void @llvm.assume(i1 %48)
  %49 = icmp ult i32 %29, 128
  br i1 %49, label %50, label %57

50:                                               ; preds = %40
  %51 = zext nneg i32 %30 to i64
  %52 = shl nuw nsw i32 %30, 3
  %53 = lshr i32 %31, 3
  %54 = or disjoint i32 %52, %53
  %55 = or disjoint i32 %54, %47
  %56 = insertelement <2 x i32> undef, i32 %55, i64 %51
  br label %57

57:                                               ; preds = %50, %40
  %58 = phi <2 x i32> [ %56, %50 ], [ undef, %40 ]
  %59 = shl nsw i32 %43, 16
  %60 = mul nuw nsw i32 %46, 1835008
  %61 = add nsw i32 %60, %59
  %62 = shl i32 %30, 14
  %63 = add i32 %61, %62
  %64 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %63)
  %65 = or disjoint i32 %62, 4096
  %66 = add i32 %65, %61
  %67 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %66)
  %68 = or disjoint i32 %62, 8192
  %69 = add i32 %68, %61
  %70 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %69)
  %71 = or disjoint i32 %62, 12288
  %72 = add i32 %71, %61
  %73 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %72)
  %74 = shl nsw i32 %43, 12
  %75 = mul nuw nsw i32 %46, 114688
  %76 = shl i32 %30, 10
  %77 = add i32 %76, %74
  %78 = add i32 %75, %77
  %79 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %78)
  %80 = or disjoint i32 %77, 512
  %81 = add i32 %80, %75
  %82 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81)
  %83 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 range(i32 -76695844, 76695845) %41)
  %84 = shl nsw i32 %83, 9
  br i1 %49, label %87, label %85

85:                                               ; preds = %57
  %86 = and i32 %27, 48
  br label %101

87:                                               ; preds = %57
  %88 = shl nuw nsw i32 %30, 3
  %89 = and i32 %29, 64
  %90 = and i32 %27, 48
  %91 = or disjoint i32 %89, %90
  %92 = shl nuw nsw i32 %27, 4
  %93 = and i32 %92, 112
  %94 = xor i32 %91, %93
  %95 = zext nneg i32 %30 to i64
  %96 = extractelement <2 x i32> %58, i64 %95
  %97 = shl nsw i32 %96, 8
  %98 = or disjoint i32 %97, %94
  %99 = getelementptr inbounds nuw [2 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 0, i32 0, i32 %88
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %32, ptr addrspace(3) noundef %99, i32 noundef 16, i32 noundef %98, i32 noundef range(i32 0, 129) 0, i32 noundef 0, i32 noundef 0) #9
  %100 = getelementptr inbounds nuw [2 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 0, i32 1, i32 %88
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %32, ptr addrspace(3) noundef nonnull %100, i32 noundef 16, i32 noundef %98, i32 noundef range(i32 0, 129) 128, i32 noundef 0, i32 noundef 0) #9
  br label %101

101:                                              ; preds = %85, %87
  %102 = phi i32 [ %86, %85 ], [ %90, %87 ]
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %103 = lshr i32 %31, 4
  %104 = and i32 %27, 15
  %105 = shl nuw nsw i32 %103, 6
  %106 = shl nuw nsw i32 %104, 2
  %107 = or disjoint i32 %105, %106
  %108 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %34, i32 %107, i32 %84, i32 0)
  %109 = or disjoint i32 %107, 256
  %110 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %34, i32 %109, i32 %84, i32 0)
  %111 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %35, i32 %107, i32 %79, i32 0)
  %112 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %35, i32 %107, i32 %82, i32 0)
  %113 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %35, i32 %109, i32 %79, i32 0)
  %114 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %35, i32 %109, i32 %82, i32 0)
  %115 = shl nuw nsw i32 %103, 8
  %116 = shl nuw nsw i32 %104, 4
  %117 = or disjoint i32 %115, %116
  %118 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %117, i32 %64, i32 2)
  %119 = or disjoint i32 %117, 1024
  %120 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %119, i32 %64, i32 2)
  %121 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %117, i32 %67, i32 2)
  %122 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %119, i32 %67, i32 2)
  %123 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %117, i32 %70, i32 2)
  %124 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %119, i32 %70, i32 2)
  %125 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %117, i32 %73, i32 2)
  %126 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %119, i32 %73, i32 2)
  %127 = or disjoint i32 %117, 2048
  %128 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %127, i32 %64, i32 2)
  %129 = or disjoint i32 %117, 3072
  %130 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %129, i32 %64, i32 2)
  %131 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %127, i32 %67, i32 2)
  %132 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %129, i32 %67, i32 2)
  %133 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %127, i32 %70, i32 2)
  %134 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %129, i32 %70, i32 2)
  %135 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %127, i32 %73, i32 2)
  %136 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %33, i32 %129, i32 %73, i32 2)
  tail call void asm sideeffect "s_waitcnt vmcnt(23)", "~{memory}"() #9, !srcloc !11
  tail call void @llvm.amdgcn.s.barrier()
  %137 = shl nuw nsw i32 %27, 3
  %138 = and i32 %137, 112
  %139 = xor i32 %138, %102
  %140 = getelementptr inbounds nuw [2 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 0, i32 0, i32 %104, i32 %139
  %141 = load <4 x i32>, ptr addrspace(3) %140, align 16, !tbaa !12
  %142 = or disjoint i32 %102, 64
  %143 = xor i32 %142, %138
  %144 = getelementptr inbounds nuw [2 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 0, i32 0, i32 %104, i32 %143
  %145 = load <4 x i32>, ptr addrspace(3) %144, align 16, !tbaa !12
  %146 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %141, <4 x i32> noundef %118, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %108, i32 noundef 0, i32 noundef %111) #9
  %147 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %145, <4 x i32> noundef %120, <4 x float> noundef %146, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %108, i32 noundef 2, i32 noundef %111) #9
  %148 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %141, <4 x i32> noundef %121, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %108, i32 noundef 1, i32 noundef %111) #9
  %149 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %145, <4 x i32> noundef %122, <4 x float> noundef %148, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %108, i32 noundef 3, i32 noundef %111) #9
  %150 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %141, <4 x i32> noundef %123, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %108, i32 noundef 0, i32 noundef %112) #9
  %151 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %145, <4 x i32> noundef %124, <4 x float> noundef %150, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %108, i32 noundef 2, i32 noundef %112) #9
  %152 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %141, <4 x i32> noundef %125, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %108, i32 noundef 1, i32 noundef %112) #9
  %153 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %145, <4 x i32> noundef %126, <4 x float> noundef %152, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %108, i32 noundef 3, i32 noundef %112) #9
  tail call void asm sideeffect "s_waitcnt vmcnt(22)", "~{memory}"() #9, !srcloc !13
  tail call void @llvm.amdgcn.s.barrier()
  %154 = getelementptr inbounds nuw [2 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 0, i32 1, i32 %104, i32 %139
  %155 = load <4 x i32>, ptr addrspace(3) %154, align 16, !tbaa !12
  %156 = getelementptr inbounds nuw [2 x [16 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 0, i32 1, i32 %104, i32 %143
  %157 = load <4 x i32>, ptr addrspace(3) %156, align 16, !tbaa !12
  %158 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %155, <4 x i32> noundef %128, <4 x float> noundef %147, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %110, i32 noundef 0, i32 noundef %113) #9
  %159 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %157, <4 x i32> noundef %130, <4 x float> noundef %158, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %110, i32 noundef 2, i32 noundef %113) #9
  %160 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %155, <4 x i32> noundef %131, <4 x float> noundef %149, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %110, i32 noundef 1, i32 noundef %113) #9
  %161 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %157, <4 x i32> noundef %132, <4 x float> noundef %160, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %110, i32 noundef 3, i32 noundef %113) #9
  %162 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %155, <4 x i32> noundef %133, <4 x float> noundef %151, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %110, i32 noundef 0, i32 noundef %114) #9
  %163 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %157, <4 x i32> noundef %134, <4 x float> noundef %162, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %110, i32 noundef 2, i32 noundef %114) #9
  %164 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %155, <4 x i32> noundef %135, <4 x float> noundef %153, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %110, i32 noundef 1, i32 noundef %114) #9
  %165 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %157, <4 x i32> noundef %136, <4 x float> noundef %164, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %110, i32 noundef 3, i32 noundef %114) #9
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !19)
  %166 = getelementptr float, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 %104
  %167 = shl i32 %30, 8
  %168 = getelementptr i8, ptr addrspace(3) %166, i32 %167
  %169 = extractelement <4 x float> %159, i64 0
  %170 = shl nuw nsw i32 %103, 12
  %171 = getelementptr i8, ptr addrspace(3) %168, i32 %170
  store float %169, ptr addrspace(3) %171, align 4, !tbaa !21, !noalias !23
  %172 = extractelement <4 x float> %159, i64 1
  %173 = getelementptr i8, ptr addrspace(3) %168, i32 1024
  %174 = getelementptr i8, ptr addrspace(3) %173, i32 %170
  store float %172, ptr addrspace(3) %174, align 4, !tbaa !21, !noalias !23
  %175 = extractelement <4 x float> %159, i64 2
  %176 = getelementptr i8, ptr addrspace(3) %168, i32 2048
  %177 = getelementptr i8, ptr addrspace(3) %176, i32 %170
  store float %175, ptr addrspace(3) %177, align 4, !tbaa !21, !noalias !23
  %178 = extractelement <4 x float> %159, i64 3
  %179 = getelementptr i8, ptr addrspace(3) %168, i32 3072
  %180 = getelementptr i8, ptr addrspace(3) %179, i32 %170
  store float %178, ptr addrspace(3) %180, align 4, !tbaa !21, !noalias !23
  %181 = getelementptr i8, ptr addrspace(3) %166, i32 64
  %182 = getelementptr i8, ptr addrspace(3) %181, i32 %167
  %183 = extractelement <4 x float> %161, i64 0
  %184 = getelementptr i8, ptr addrspace(3) %182, i32 %170
  store float %183, ptr addrspace(3) %184, align 4, !tbaa !21, !noalias !23
  %185 = extractelement <4 x float> %161, i64 1
  %186 = getelementptr i8, ptr addrspace(3) %182, i32 1024
  %187 = getelementptr i8, ptr addrspace(3) %186, i32 %170
  store float %185, ptr addrspace(3) %187, align 4, !tbaa !21, !noalias !23
  %188 = extractelement <4 x float> %161, i64 2
  %189 = getelementptr i8, ptr addrspace(3) %182, i32 2048
  %190 = getelementptr i8, ptr addrspace(3) %189, i32 %170
  store float %188, ptr addrspace(3) %190, align 4, !tbaa !21, !noalias !23
  %191 = extractelement <4 x float> %161, i64 3
  %192 = getelementptr i8, ptr addrspace(3) %182, i32 3072
  %193 = getelementptr i8, ptr addrspace(3) %192, i32 %170
  store float %191, ptr addrspace(3) %193, align 4, !tbaa !21, !noalias !23
  %194 = getelementptr i8, ptr addrspace(3) %166, i32 128
  %195 = getelementptr i8, ptr addrspace(3) %194, i32 %167
  %196 = extractelement <4 x float> %163, i64 0
  %197 = getelementptr i8, ptr addrspace(3) %195, i32 %170
  store float %196, ptr addrspace(3) %197, align 4, !tbaa !21, !noalias !23
  %198 = extractelement <4 x float> %163, i64 1
  %199 = getelementptr i8, ptr addrspace(3) %195, i32 1024
  %200 = getelementptr i8, ptr addrspace(3) %199, i32 %170
  store float %198, ptr addrspace(3) %200, align 4, !tbaa !21, !noalias !23
  %201 = extractelement <4 x float> %163, i64 2
  %202 = getelementptr i8, ptr addrspace(3) %195, i32 2048
  %203 = getelementptr i8, ptr addrspace(3) %202, i32 %170
  store float %201, ptr addrspace(3) %203, align 4, !tbaa !21, !noalias !23
  %204 = extractelement <4 x float> %163, i64 3
  %205 = getelementptr i8, ptr addrspace(3) %195, i32 3072
  %206 = getelementptr i8, ptr addrspace(3) %205, i32 %170
  store float %204, ptr addrspace(3) %206, align 4, !tbaa !21, !noalias !23
  %207 = getelementptr i8, ptr addrspace(3) %166, i32 192
  %208 = getelementptr i8, ptr addrspace(3) %207, i32 %167
  %209 = extractelement <4 x float> %165, i64 0
  %210 = getelementptr i8, ptr addrspace(3) %208, i32 %170
  store float %209, ptr addrspace(3) %210, align 4, !tbaa !21, !noalias !23
  %211 = extractelement <4 x float> %165, i64 1
  %212 = getelementptr i8, ptr addrspace(3) %208, i32 1024
  %213 = getelementptr i8, ptr addrspace(3) %212, i32 %170
  store float %211, ptr addrspace(3) %213, align 4, !tbaa !21, !noalias !23
  %214 = extractelement <4 x float> %165, i64 2
  %215 = getelementptr i8, ptr addrspace(3) %208, i32 2048
  %216 = getelementptr i8, ptr addrspace(3) %215, i32 %170
  store float %214, ptr addrspace(3) %216, align 4, !tbaa !21, !noalias !23
  %217 = extractelement <4 x float> %165, i64 3
  %218 = getelementptr i8, ptr addrspace(3) %208, i32 3072
  %219 = getelementptr i8, ptr addrspace(3) %218, i32 %170
  store float %217, ptr addrspace(3) %219, align 4, !tbaa !21, !noalias !23
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier(), !noalias !23
  fence syncscope("workgroup") acquire
  %220 = lshr i32 %27, 5
  %221 = shl nuw nsw i32 %27, 1
  %222 = and i32 %221, 62
  %223 = or disjoint i32 %47, %220
  %224 = shl nsw i32 %43, 8
  %225 = sext i32 %224 to i64
  %226 = zext nneg i32 %222 to i64
  %227 = getelementptr %struct.__hip_bfloat16, ptr %25, i64 %225
  %228 = getelementptr %struct.__hip_bfloat16, ptr %227, i64 %226
  %229 = sext i32 %223 to i64
  %230 = getelementptr inbounds i32, ptr %21, i64 %229
  %231 = load i32, ptr %230, align 4, !tbaa !7, !alias.scope !17, !noalias !24
  %232 = and i32 %231, 16777215
  %233 = icmp slt i32 %232, %8
  br i1 %233, label %234, label %271

234:                                              ; preds = %101
  %235 = getelementptr inbounds float, ptr %23, i64 %229
  %236 = load float, ptr %235, align 4, !tbaa !21, !alias.scope !19, !noalias !25
  %237 = shl nuw nsw i32 %220, 10
  %238 = getelementptr i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 %237
  %239 = getelementptr float, ptr addrspace(3) %238, i32 %222
  %240 = getelementptr i8, ptr addrspace(3) %239, i32 256
  %241 = getelementptr i8, ptr addrspace(3) %239, i32 512
  %242 = getelementptr i8, ptr addrspace(3) %239, i32 768
  %243 = zext nneg i32 %232 to i64
  %244 = mul nuw nsw i64 %243, 14336
  %245 = getelementptr i8, ptr %228, i64 %244
  %246 = load <2 x float>, ptr addrspace(3) %239, align 8, !tbaa !21, !noalias !23
  %247 = insertelement <2 x float> poison, float %236, i64 0
  %248 = shufflevector <2 x float> %247, <2 x float> poison, <2 x i32> zeroinitializer
  %249 = fmul contract <2 x float> %248, %246
  %250 = fptrunc <2 x float> %249 to <2 x bfloat>
  %251 = addrspacecast ptr %245 to ptr addrspace(1)
  %252 = atomicrmw fadd ptr addrspace(1) %251, <2 x bfloat> %250 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %253 = load <2 x float>, ptr addrspace(3) %240, align 8, !tbaa !21, !noalias !23
  %254 = fmul contract <2 x float> %248, %253
  %255 = fptrunc <2 x float> %254 to <2 x bfloat>
  %256 = getelementptr inbounds nuw i8, ptr %245, i64 128
  %257 = addrspacecast ptr %256 to ptr addrspace(1)
  %258 = atomicrmw fadd ptr addrspace(1) %257, <2 x bfloat> %255 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %259 = load <2 x float>, ptr addrspace(3) %241, align 8, !tbaa !21, !noalias !23
  %260 = fmul contract <2 x float> %248, %259
  %261 = fptrunc <2 x float> %260 to <2 x bfloat>
  %262 = getelementptr inbounds nuw i8, ptr %245, i64 256
  %263 = addrspacecast ptr %262 to ptr addrspace(1)
  %264 = atomicrmw fadd ptr addrspace(1) %263, <2 x bfloat> %261 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %265 = load <2 x float>, ptr addrspace(3) %242, align 8, !tbaa !21, !noalias !23
  %266 = fmul contract <2 x float> %248, %265
  %267 = fptrunc <2 x float> %266 to <2 x bfloat>
  %268 = getelementptr inbounds nuw i8, ptr %245, i64 384
  %269 = addrspacecast ptr %268 to ptr addrspace(1)
  %270 = atomicrmw fadd ptr addrspace(1) %269, <2 x bfloat> %267 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  br label %271

271:                                              ; preds = %234, %101
  %272 = or disjoint i32 %223, 8
  %273 = sext i32 %272 to i64
  %274 = getelementptr inbounds i32, ptr %21, i64 %273
  %275 = load i32, ptr %274, align 4, !tbaa !7, !alias.scope !17, !noalias !24
  %276 = and i32 %275, 16777215
  %277 = icmp slt i32 %276, %8
  br i1 %277, label %278, label %316

278:                                              ; preds = %271
  %279 = getelementptr inbounds float, ptr %23, i64 %273
  %280 = load float, ptr %279, align 4, !tbaa !21, !alias.scope !19, !noalias !25
  %281 = shl nuw nsw i32 %220, 10
  %282 = getelementptr i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm26kernelILi655360ELi385ELi512ELi7168ELi9ELi16ELNS1_12EpilogPolicyE0ELb1ELi0ELb0EEEvPKhPKaS5_S7_PKiS9_S9_PKfiP14__hip_bfloat16PhE3lds, i32 %281
  %283 = getelementptr i8, ptr addrspace(3) %282, i32 8192
  %284 = getelementptr float, ptr addrspace(3) %283, i32 %222
  %285 = getelementptr i8, ptr addrspace(3) %284, i32 256
  %286 = getelementptr i8, ptr addrspace(3) %284, i32 512
  %287 = getelementptr i8, ptr addrspace(3) %284, i32 768
  %288 = zext nneg i32 %276 to i64
  %289 = mul nuw nsw i64 %288, 14336
  %290 = getelementptr i8, ptr %228, i64 %289
  %291 = load <2 x float>, ptr addrspace(3) %284, align 8, !tbaa !21, !noalias !23
  %292 = insertelement <2 x float> poison, float %280, i64 0
  %293 = shufflevector <2 x float> %292, <2 x float> poison, <2 x i32> zeroinitializer
  %294 = fmul contract <2 x float> %293, %291
  %295 = fptrunc <2 x float> %294 to <2 x bfloat>
  %296 = addrspacecast ptr %290 to ptr addrspace(1)
  %297 = atomicrmw fadd ptr addrspace(1) %296, <2 x bfloat> %295 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %298 = load <2 x float>, ptr addrspace(3) %285, align 8, !tbaa !21, !noalias !23
  %299 = fmul contract <2 x float> %293, %298
  %300 = fptrunc <2 x float> %299 to <2 x bfloat>
  %301 = getelementptr inbounds nuw i8, ptr %290, i64 128
  %302 = addrspacecast ptr %301 to ptr addrspace(1)
  %303 = atomicrmw fadd ptr addrspace(1) %302, <2 x bfloat> %300 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %304 = load <2 x float>, ptr addrspace(3) %286, align 8, !tbaa !21, !noalias !23
  %305 = fmul contract <2 x float> %293, %304
  %306 = fptrunc <2 x float> %305 to <2 x bfloat>
  %307 = getelementptr inbounds nuw i8, ptr %290, i64 256
  %308 = addrspacecast ptr %307 to ptr addrspace(1)
  %309 = atomicrmw fadd ptr addrspace(1) %308, <2 x bfloat> %306 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  %310 = load <2 x float>, ptr addrspace(3) %287, align 8, !tbaa !21, !noalias !23
  %311 = fmul contract <2 x float> %293, %310
  %312 = fptrunc <2 x float> %311 to <2 x bfloat>
  %313 = getelementptr inbounds nuw i8, ptr %290, i64 384
  %314 = addrspacecast ptr %313 to ptr addrspace(1)
  %315 = atomicrmw fadd ptr addrspace(1) %314, <2 x bfloat> %312 syncscope("agent") monotonic, align 4, !alias.scope !14, !noalias !26, !amdgpu.no.fine.grained.memory !27
  br label %316

316:                                              ; preds = %278, %271, %11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone, i16, i32, i32) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) nocapture readonly, ptr addrspace(3) nocapture writeonly, i32 immarg, i32, i32, i32 immarg, i32 immarg) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg) #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #7

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #8

attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="4" "denormal-fp-math-f32"="preserve-sign,preserve-sign" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #7 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #8 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #9 = { convergent nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!opencl.ocl.version = !{!5}
!llvm.ident = !{!6}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 600}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 1, !"Code Model", i32 4}
!5 = !{i32 2, i32 0}
!6 = !{!"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.1.0 25425 1b0eada6b0ee93e2e694c8c146d23fca90bc11c5)"}
!7 = !{!8, !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{i64 9642789}
!12 = !{!9, !9, i64 0}
!13 = !{i64 9642884}
!14 = !{!15}
!15 = distinct !{!15, !16, !"_ZN5aiter9mxfp4_moe11gemm_common24apply_atomic_bf16_epilogILi7168ELi16EEEvRAqueqT0_Li16ELi1EdvT0_Li16E_A4_KDv4_fP14__hip_bfloat16PKiPKfiiiiiiPf: argument 0"}
!16 = distinct !{!16, !"_ZN5aiter9mxfp4_moe11gemm_common24apply_atomic_bf16_epilogILi7168ELi16EEEvRAqueqT0_Li16ELi1EdvT0_Li16E_A4_KDv4_fP14__hip_bfloat16PKiPKfiiiiiiPf"}
!17 = !{!18}
!18 = distinct !{!18, !16, !"_ZN5aiter9mxfp4_moe11gemm_common24apply_atomic_bf16_epilogILi7168ELi16EEEvRAqueqT0_Li16ELi1EdvT0_Li16E_A4_KDv4_fP14__hip_bfloat16PKiPKfiiiiiiPf: argument 1"}
!19 = !{!20}
!20 = distinct !{!20, !16, !"_ZN5aiter9mxfp4_moe11gemm_common24apply_atomic_bf16_epilogILi7168ELi16EEEvRAqueqT0_Li16ELi1EdvT0_Li16E_A4_KDv4_fP14__hip_bfloat16PKiPKfiiiiiiPf: argument 2"}
!21 = !{!22, !22, i64 0}
!22 = !{!"float", !9, i64 0}
!23 = !{!15, !18, !20}
!24 = !{!15, !20}
!25 = !{!15, !18}
!26 = !{!18, !20}
!27 = !{}
