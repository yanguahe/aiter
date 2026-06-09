; ModuleID = '/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM32_NT.cu'
source_filename = "/shared/amdgpu/home/zhiming_ding_qle/yanguahe/code/wk_perf_test/aiter/aiter/jit/build/module_moe_mxfp4_gemm/blob/instances/mxfp4_moe_g1_a4w4_NE385_H7168_E512_BM32_NT.cu"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%union.LDSPool = type { [8192 x float] }

$_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16 = comdat any

$_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds = comdat any

@_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds = linkonce_odr hidden addrspace(3) global %union.LDSPool undef, comdat, align 16
@__hip_cuid_30a568c79557e862 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_30a568c79557e862 to ptr)], section "llvm.metadata"

; Function Attrs: convergent mustprogress nofree norecurse nounwind willreturn
define protected amdgpu_kernel void @_ZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16(ptr addrspace(1) noalias noundef %0, ptr addrspace(1) noalias noundef %1, ptr addrspace(1) noalias noundef %2, ptr addrspace(1) noalias noundef %3, ptr addrspace(1) noalias nocapture noundef readonly %4, ptr addrspace(1) noalias nocapture noundef readonly %5, ptr addrspace(1) noalias noundef %6, i32 noundef %7, ptr addrspace(1) noalias noundef %8, ptr addrspace(1) noalias noundef %9, ptr addrspace(1) noalias nocapture noundef readnone %10) local_unnamed_addr #0 comdat {
  %12 = ptrtoint ptr addrspace(1) %2 to i64
  %13 = inttoptr i64 %12 to ptr
  %14 = ptrtoint ptr addrspace(1) %3 to i64
  %15 = inttoptr i64 %14 to ptr
  %16 = ptrtoint ptr addrspace(1) %9 to i64
  %17 = inttoptr i64 %16 to ptr
  %18 = tail call noundef i32 @llvm.amdgcn.workgroup.id.x()
  %19 = tail call noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x()
  %20 = icmp samesign ult i32 %19, 256
  tail call void @llvm.assume(i1 %20)
  %21 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %19)
  %22 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %13, i16 0, i32 1412956160, i32 131072)
  %23 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %15, i16 0, i32 88309760, i32 131072)
  %24 = load i32, ptr addrspace(1) %5, align 4, !tbaa !7
  %25 = sdiv i32 %24, 32
  %26 = shl nsw i32 %25, 2
  %27 = icmp slt i32 %18, %26
  br i1 %27, label %28, label %1633

28:                                               ; preds = %11
  %29 = ptrtoint ptr addrspace(1) %1 to i64
  %30 = inttoptr i64 %29 to ptr
  %31 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %30, i16 0, i32 146800640, i32 131072)
  %32 = ptrtoint ptr addrspace(1) %0 to i64
  %33 = inttoptr i64 %32 to ptr
  %34 = mul nsw i32 %7, 3584
  %35 = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone %33, i16 0, i32 %34, i32 131072)
  %36 = and i32 %19, 63
  %37 = lshr i32 %21, 6
  %38 = ptrtoint ptr addrspace(1) %8 to i64
  %39 = inttoptr i64 %38 to ptr
  %40 = ptrtoint ptr addrspace(1) %6 to i64
  %41 = inttoptr i64 %40 to ptr
  %42 = sdiv i32 %18, 4
  %43 = mul i32 %42, 4
  %44 = sub i32 %18, %43
  %45 = sext i32 %42 to i64
  %46 = getelementptr inbounds i32, ptr addrspace(1) %4, i64 %45
  %47 = load i32, ptr addrspace(1) %46, align 4, !tbaa !7
  %48 = shl nsw i32 %42, 5
  %49 = icmp samesign ult i32 %47, 385
  tail call void @llvm.assume(i1 %49)
  %50 = lshr i32 %36, 3
  %51 = shl nuw nsw i32 %37, 3
  %52 = or disjoint i32 %50, %48
  %53 = add i32 %52, %51
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds i32, ptr %41, i64 %54
  %56 = load i32, ptr %55, align 4, !tbaa !7
  %57 = shl nuw nsw i32 %47, 10
  %58 = shl nsw i32 %44, 8
  %59 = add nsw i32 %57, %58
  %60 = and i32 %21, -64
  %61 = add i32 %59, %60
  %62 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %61)
  %63 = mul nsw i32 %62, 3584
  %64 = or disjoint i32 %60, 16
  %65 = add i32 %64, %59
  %66 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %65)
  %67 = mul nsw i32 %66, 3584
  %68 = or disjoint i32 %60, 32
  %69 = add i32 %68, %59
  %70 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %69)
  %71 = mul nsw i32 %70, 3584
  %72 = or disjoint i32 %60, 48
  %73 = add i32 %72, %59
  %74 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %73)
  %75 = mul nsw i32 %74, 3584
  %76 = shl nsw i32 %44, 3
  %77 = shl nuw nsw i32 %37, 1
  %78 = add nsw i32 %77, %76
  %79 = mul nuw nsw i32 %47, 57344
  %80 = mul i32 %78, 1792
  %81 = add i32 %79, %80
  %82 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81)
  %83 = shl nsw i32 %82, 2
  %84 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %83)
  %85 = add nsw i32 %84, 4096
  %86 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %81)
  %87 = shl i32 %86, 2
  %88 = add i32 %87, 7168
  %89 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %88)
  %90 = add nsw i32 %89, 4096
  %91 = or disjoint i32 %60, %36
  %92 = shl nsw i32 %91, 4
  %93 = shl nsw i32 %91, 2
  %94 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %48)
  %95 = sdiv i32 %94, 32
  %96 = mul nsw i32 %95, 7168
  %97 = shl nsw i32 %37, 10
  %98 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %97
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %31, ptr addrspace(3) noundef nonnull %98, i32 noundef 16, i32 noundef %92, i32 noundef %96, i32 noundef 0, i32 noundef 0) #12
  %99 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %96)
  %100 = add i32 %99, 4096
  %101 = shl nsw i32 %37, 8
  %102 = add nuw nsw i32 %101, 4096
  %103 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %102
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %31, ptr addrspace(3) noundef nonnull %103, i32 noundef 4, i32 noundef %93, i32 noundef %100, i32 noundef 0, i32 noundef 0) #12
  %104 = add i32 %99, 5120
  %105 = add nuw nsw i32 %101, 5120
  %106 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %105
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %31, ptr addrspace(3) noundef nonnull %106, i32 noundef 4, i32 noundef %93, i32 noundef %104, i32 noundef 0, i32 noundef 0) #12
  %107 = add i32 %99, 6144
  %108 = add nuw nsw i32 %101, 6144
  %109 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %108
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %31, ptr addrspace(3) noundef nonnull %109, i32 noundef 4, i32 noundef %93, i32 noundef %107, i32 noundef 0, i32 noundef 0) #12
  %110 = or disjoint i32 %51, %50
  %111 = shl nuw i32 %110, 3
  %112 = shl nuw nsw i32 %19, 4
  %113 = xor i32 %111, %112
  %114 = and i32 %113, 112
  %115 = mul nsw i32 %56, 3584
  %116 = or disjoint i32 %115, %114
  %117 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %51
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 0, i32 noundef 0, i32 noundef 0) #12
  %118 = lshr i32 %36, 4
  %119 = shl nuw nsw i32 %118, 8
  %120 = and i32 %19, 15
  %121 = shl nuw nsw i32 %120, 4
  %122 = or disjoint i32 %119, %121
  %123 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %122, i32 %63, i32 2)
  %124 = or disjoint i32 %122, 1024
  %125 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %124, i32 %63, i32 2)
  %126 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %122, i32 %67, i32 2)
  %127 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %124, i32 %67, i32 2)
  %128 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %122, i32 %71, i32 2)
  %129 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %124, i32 %71, i32 2)
  %130 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %122, i32 %75, i32 2)
  %131 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %124, i32 %75, i32 2)
  %132 = shl nuw nsw i32 %118, 6
  %133 = shl nuw nsw i32 %120, 2
  %134 = or disjoint i32 %132, %133
  %135 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %134, i32 %83, i32 0)
  %136 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %134, i32 %88, i32 0)
  %137 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %51
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 128, i32 noundef 0, i32 noundef 0) #12
  %138 = or disjoint i32 %122, 2048
  %139 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %138, i32 %63, i32 2)
  %140 = or disjoint i32 %122, 3072
  %141 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %140, i32 %63, i32 2)
  %142 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %138, i32 %67, i32 2)
  %143 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %140, i32 %67, i32 2)
  %144 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %138, i32 %71, i32 2)
  %145 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %140, i32 %71, i32 2)
  %146 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %138, i32 %75, i32 2)
  %147 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %140, i32 %75, i32 2)
  %148 = or disjoint i32 %134, 256
  %149 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %148, i32 %83, i32 0)
  %150 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %148, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %151 = and i32 %19, 48
  %152 = shl nuw nsw i32 %19, 3
  %153 = and i32 %152, 112
  %154 = xor i32 %153, %151
  %155 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %120, i32 %154
  %156 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %157 = or disjoint i32 %120, 16
  %158 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %157, i32 %154
  %159 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %160 = or disjoint i32 %151, 64
  %161 = xor i32 %160, %153
  %162 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %120, i32 %161
  %163 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %164 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 0, i32 %157, i32 %161
  %165 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %166 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %134
  %167 = load i32, ptr addrspace(3) %166, align 4, !tbaa !7
  %168 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %51
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 256, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %169 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %156, <4 x i32> noundef %123, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %167, i32 noundef 0, i32 noundef %135) #12
  %170 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %159, <4 x i32> noundef %123, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %167, i32 noundef 0, i32 noundef %135) #12
  %171 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %163, <4 x i32> noundef %125, <4 x float> noundef %169, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %167, i32 noundef 2, i32 noundef %135) #12
  %172 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %165, <4 x i32> noundef %125, <4 x float> noundef %170, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %167, i32 noundef 2, i32 noundef %135) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %173 = or disjoint i32 %122, 4096
  %174 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %173, i32 %63, i32 2)
  %175 = or disjoint i32 %122, 5120
  %176 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %175, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %177 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %156, <4 x i32> noundef %126, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %167, i32 noundef 1, i32 noundef %135) #12
  %178 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %159, <4 x i32> noundef %126, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %167, i32 noundef 1, i32 noundef %135) #12
  %179 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %163, <4 x i32> noundef %127, <4 x float> noundef %177, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %167, i32 noundef 3, i32 noundef %135) #12
  %180 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %165, <4 x i32> noundef %127, <4 x float> noundef %178, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %167, i32 noundef 3, i32 noundef %135) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %181 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %173, i32 %67, i32 2)
  %182 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %175, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %183 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %156, <4 x i32> noundef %128, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %167, i32 noundef 0, i32 noundef %136) #12
  %184 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %159, <4 x i32> noundef %128, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %167, i32 noundef 0, i32 noundef %136) #12
  %185 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %163, <4 x i32> noundef %129, <4 x float> noundef %183, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %167, i32 noundef 2, i32 noundef %136) #12
  %186 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %165, <4 x i32> noundef %129, <4 x float> noundef %184, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %167, i32 noundef 2, i32 noundef %136) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %187 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %173, i32 %71, i32 2)
  %188 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %175, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %189 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %156, <4 x i32> noundef %130, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %167, i32 noundef 1, i32 noundef %136) #12
  %190 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %159, <4 x i32> noundef %130, <4 x float> noundef zeroinitializer, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %167, i32 noundef 1, i32 noundef %136) #12
  %191 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %163, <4 x i32> noundef %131, <4 x float> noundef %189, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %167, i32 noundef 3, i32 noundef %136) #12
  %192 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %165, <4 x i32> noundef %131, <4 x float> noundef %190, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %167, i32 noundef 3, i32 noundef %136) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %193 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %173, i32 %75, i32 2)
  %194 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %175, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %195 = or disjoint i32 %134, 512
  %196 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %195, i32 %83, i32 0)
  %197 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %195, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %198 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %120, i32 %154
  %199 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %200 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %157, i32 %154
  %201 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %202 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %120, i32 %161
  %203 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %204 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 1, i32 %157, i32 %161
  %205 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %206 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %148
  %207 = load i32, ptr addrspace(3) %206, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 384, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %208 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %199, <4 x i32> noundef %139, <4 x float> noundef %171, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %207, i32 noundef 0, i32 noundef %149) #12
  %209 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %201, <4 x i32> noundef %139, <4 x float> noundef %172, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %207, i32 noundef 0, i32 noundef %149) #12
  %210 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %203, <4 x i32> noundef %141, <4 x float> noundef %208, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %207, i32 noundef 2, i32 noundef %149) #12
  %211 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %205, <4 x i32> noundef %141, <4 x float> noundef %209, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %207, i32 noundef 2, i32 noundef %149) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %212 = or disjoint i32 %122, 6144
  %213 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %212, i32 %63, i32 2)
  %214 = or disjoint i32 %122, 7168
  %215 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %214, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %216 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %199, <4 x i32> noundef %142, <4 x float> noundef %179, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %207, i32 noundef 1, i32 noundef %149) #12
  %217 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %201, <4 x i32> noundef %142, <4 x float> noundef %180, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %207, i32 noundef 1, i32 noundef %149) #12
  %218 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %203, <4 x i32> noundef %143, <4 x float> noundef %216, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %207, i32 noundef 3, i32 noundef %149) #12
  %219 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %205, <4 x i32> noundef %143, <4 x float> noundef %217, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %207, i32 noundef 3, i32 noundef %149) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %220 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %212, i32 %67, i32 2)
  %221 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %214, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %222 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %199, <4 x i32> noundef %144, <4 x float> noundef %185, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %207, i32 noundef 0, i32 noundef %150) #12
  %223 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %201, <4 x i32> noundef %144, <4 x float> noundef %186, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %207, i32 noundef 0, i32 noundef %150) #12
  %224 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %203, <4 x i32> noundef %145, <4 x float> noundef %222, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %207, i32 noundef 2, i32 noundef %150) #12
  %225 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %205, <4 x i32> noundef %145, <4 x float> noundef %223, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %207, i32 noundef 2, i32 noundef %150) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %226 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %212, i32 %71, i32 2)
  %227 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %214, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %228 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %199, <4 x i32> noundef %146, <4 x float> noundef %191, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %207, i32 noundef 1, i32 noundef %150) #12
  %229 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %201, <4 x i32> noundef %146, <4 x float> noundef %192, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %207, i32 noundef 1, i32 noundef %150) #12
  %230 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %203, <4 x i32> noundef %147, <4 x float> noundef %228, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %207, i32 noundef 3, i32 noundef %150) #12
  %231 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %205, <4 x i32> noundef %147, <4 x float> noundef %229, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %207, i32 noundef 3, i32 noundef %150) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %232 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %212, i32 %75, i32 2)
  %233 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %214, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %234 = or disjoint i32 %134, 768
  %235 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %234, i32 %83, i32 0)
  %236 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %234, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %237 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %120, i32 %154
  %238 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %239 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %157, i32 %154
  %240 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %241 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %120, i32 %161
  %242 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %243 = getelementptr inbounds nuw [3 x [32 x [128 x i8]]], ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 0, i32 2, i32 %157, i32 %161
  %244 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %245 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %195
  %246 = load i32, ptr addrspace(3) %245, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 512, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %247 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %238, <4 x i32> noundef %174, <4 x float> noundef %210, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %246, i32 noundef 0, i32 noundef %196) #12
  %248 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %240, <4 x i32> noundef %174, <4 x float> noundef %211, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %246, i32 noundef 0, i32 noundef %196) #12
  %249 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %242, <4 x i32> noundef %176, <4 x float> noundef %247, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %246, i32 noundef 2, i32 noundef %196) #12
  %250 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %244, <4 x i32> noundef %176, <4 x float> noundef %248, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %246, i32 noundef 2, i32 noundef %196) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %251 = or disjoint i32 %122, 8192
  %252 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %251, i32 %63, i32 2)
  %253 = or disjoint i32 %122, 9216
  %254 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %253, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %255 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %238, <4 x i32> noundef %181, <4 x float> noundef %218, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %246, i32 noundef 1, i32 noundef %196) #12
  %256 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %240, <4 x i32> noundef %181, <4 x float> noundef %219, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %246, i32 noundef 1, i32 noundef %196) #12
  %257 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %242, <4 x i32> noundef %182, <4 x float> noundef %255, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %246, i32 noundef 3, i32 noundef %196) #12
  %258 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %244, <4 x i32> noundef %182, <4 x float> noundef %256, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %246, i32 noundef 3, i32 noundef %196) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %259 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %251, i32 %67, i32 2)
  %260 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %253, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %261 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %238, <4 x i32> noundef %187, <4 x float> noundef %224, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %246, i32 noundef 0, i32 noundef %197) #12
  %262 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %240, <4 x i32> noundef %187, <4 x float> noundef %225, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %246, i32 noundef 0, i32 noundef %197) #12
  %263 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %242, <4 x i32> noundef %188, <4 x float> noundef %261, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %246, i32 noundef 2, i32 noundef %197) #12
  %264 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %244, <4 x i32> noundef %188, <4 x float> noundef %262, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %246, i32 noundef 2, i32 noundef %197) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %265 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %251, i32 %71, i32 2)
  %266 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %253, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %267 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %238, <4 x i32> noundef %193, <4 x float> noundef %230, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %246, i32 noundef 1, i32 noundef %197) #12
  %268 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %240, <4 x i32> noundef %193, <4 x float> noundef %231, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %246, i32 noundef 1, i32 noundef %197) #12
  %269 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %242, <4 x i32> noundef %194, <4 x float> noundef %267, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %246, i32 noundef 3, i32 noundef %197) #12
  %270 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %244, <4 x i32> noundef %194, <4 x float> noundef %268, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %246, i32 noundef 3, i32 noundef %197) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %271 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %251, i32 %75, i32 2)
  %272 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %253, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %273 = or disjoint i32 %134, 1024
  %274 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %273, i32 %83, i32 0)
  %275 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %273, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %276 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %277 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %278 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %279 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %280 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %234
  %281 = load i32, ptr addrspace(3) %280, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 640, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %282 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %276, <4 x i32> noundef %213, <4 x float> noundef %249, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %281, i32 noundef 0, i32 noundef %235) #12
  %283 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %277, <4 x i32> noundef %213, <4 x float> noundef %250, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %281, i32 noundef 0, i32 noundef %235) #12
  %284 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %278, <4 x i32> noundef %215, <4 x float> noundef %282, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %281, i32 noundef 2, i32 noundef %235) #12
  %285 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %279, <4 x i32> noundef %215, <4 x float> noundef %283, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %281, i32 noundef 2, i32 noundef %235) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %286 = or disjoint i32 %122, 10240
  %287 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %286, i32 %63, i32 2)
  %288 = or disjoint i32 %122, 11264
  %289 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %288, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %290 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %276, <4 x i32> noundef %220, <4 x float> noundef %257, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %281, i32 noundef 1, i32 noundef %235) #12
  %291 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %277, <4 x i32> noundef %220, <4 x float> noundef %258, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %281, i32 noundef 1, i32 noundef %235) #12
  %292 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %278, <4 x i32> noundef %221, <4 x float> noundef %290, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %281, i32 noundef 3, i32 noundef %235) #12
  %293 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %279, <4 x i32> noundef %221, <4 x float> noundef %291, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %281, i32 noundef 3, i32 noundef %235) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %294 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %286, i32 %67, i32 2)
  %295 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %288, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %296 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %276, <4 x i32> noundef %226, <4 x float> noundef %263, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %281, i32 noundef 0, i32 noundef %236) #12
  %297 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %277, <4 x i32> noundef %226, <4 x float> noundef %264, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %281, i32 noundef 0, i32 noundef %236) #12
  %298 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %278, <4 x i32> noundef %227, <4 x float> noundef %296, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %281, i32 noundef 2, i32 noundef %236) #12
  %299 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %279, <4 x i32> noundef %227, <4 x float> noundef %297, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %281, i32 noundef 2, i32 noundef %236) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %300 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %286, i32 %71, i32 2)
  %301 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %288, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %302 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %276, <4 x i32> noundef %232, <4 x float> noundef %269, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %281, i32 noundef 1, i32 noundef %236) #12
  %303 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %277, <4 x i32> noundef %232, <4 x float> noundef %270, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %281, i32 noundef 1, i32 noundef %236) #12
  %304 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %278, <4 x i32> noundef %233, <4 x float> noundef %302, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %281, i32 noundef 3, i32 noundef %236) #12
  %305 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %279, <4 x i32> noundef %233, <4 x float> noundef %303, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %281, i32 noundef 3, i32 noundef %236) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %306 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %286, i32 %75, i32 2)
  %307 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %288, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %308 = or disjoint i32 %134, 1280
  %309 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %308, i32 %83, i32 0)
  %310 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %308, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %311 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %312 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %313 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %314 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %315 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %273
  %316 = load i32, ptr addrspace(3) %315, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 768, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %317 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %311, <4 x i32> noundef %252, <4 x float> noundef %284, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %316, i32 noundef 0, i32 noundef %274) #12
  %318 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %312, <4 x i32> noundef %252, <4 x float> noundef %285, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %316, i32 noundef 0, i32 noundef %274) #12
  %319 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %313, <4 x i32> noundef %254, <4 x float> noundef %317, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %316, i32 noundef 2, i32 noundef %274) #12
  %320 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %314, <4 x i32> noundef %254, <4 x float> noundef %318, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %316, i32 noundef 2, i32 noundef %274) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %321 = or disjoint i32 %122, 12288
  %322 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %321, i32 %63, i32 2)
  %323 = or disjoint i32 %122, 13312
  %324 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %323, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %325 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %311, <4 x i32> noundef %259, <4 x float> noundef %292, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %316, i32 noundef 1, i32 noundef %274) #12
  %326 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %312, <4 x i32> noundef %259, <4 x float> noundef %293, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %316, i32 noundef 1, i32 noundef %274) #12
  %327 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %313, <4 x i32> noundef %260, <4 x float> noundef %325, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %316, i32 noundef 3, i32 noundef %274) #12
  %328 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %314, <4 x i32> noundef %260, <4 x float> noundef %326, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %316, i32 noundef 3, i32 noundef %274) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %329 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %321, i32 %67, i32 2)
  %330 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %323, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %331 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %311, <4 x i32> noundef %265, <4 x float> noundef %298, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %316, i32 noundef 0, i32 noundef %275) #12
  %332 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %312, <4 x i32> noundef %265, <4 x float> noundef %299, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %316, i32 noundef 0, i32 noundef %275) #12
  %333 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %313, <4 x i32> noundef %266, <4 x float> noundef %331, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %316, i32 noundef 2, i32 noundef %275) #12
  %334 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %314, <4 x i32> noundef %266, <4 x float> noundef %332, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %316, i32 noundef 2, i32 noundef %275) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %335 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %321, i32 %71, i32 2)
  %336 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %323, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %337 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %311, <4 x i32> noundef %271, <4 x float> noundef %304, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %316, i32 noundef 1, i32 noundef %275) #12
  %338 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %312, <4 x i32> noundef %271, <4 x float> noundef %305, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %316, i32 noundef 1, i32 noundef %275) #12
  %339 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %313, <4 x i32> noundef %272, <4 x float> noundef %337, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %316, i32 noundef 3, i32 noundef %275) #12
  %340 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %314, <4 x i32> noundef %272, <4 x float> noundef %338, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %316, i32 noundef 3, i32 noundef %275) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %341 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %321, i32 %75, i32 2)
  %342 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %323, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %343 = or disjoint i32 %134, 1536
  %344 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %343, i32 %83, i32 0)
  %345 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %343, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %346 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %347 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %348 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %349 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %350 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %308
  %351 = load i32, ptr addrspace(3) %350, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 896, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %352 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %346, <4 x i32> noundef %287, <4 x float> noundef %319, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %351, i32 noundef 0, i32 noundef %309) #12
  %353 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %347, <4 x i32> noundef %287, <4 x float> noundef %320, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %351, i32 noundef 0, i32 noundef %309) #12
  %354 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %348, <4 x i32> noundef %289, <4 x float> noundef %352, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %351, i32 noundef 2, i32 noundef %309) #12
  %355 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %349, <4 x i32> noundef %289, <4 x float> noundef %353, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %351, i32 noundef 2, i32 noundef %309) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %356 = or disjoint i32 %122, 14336
  %357 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %356, i32 %63, i32 2)
  %358 = or disjoint i32 %122, 15360
  %359 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %358, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %360 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %346, <4 x i32> noundef %294, <4 x float> noundef %327, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %351, i32 noundef 1, i32 noundef %309) #12
  %361 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %347, <4 x i32> noundef %294, <4 x float> noundef %328, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %351, i32 noundef 1, i32 noundef %309) #12
  %362 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %348, <4 x i32> noundef %295, <4 x float> noundef %360, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %351, i32 noundef 3, i32 noundef %309) #12
  %363 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %349, <4 x i32> noundef %295, <4 x float> noundef %361, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %351, i32 noundef 3, i32 noundef %309) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %364 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %356, i32 %67, i32 2)
  %365 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %358, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %366 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %346, <4 x i32> noundef %300, <4 x float> noundef %333, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %351, i32 noundef 0, i32 noundef %310) #12
  %367 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %347, <4 x i32> noundef %300, <4 x float> noundef %334, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %351, i32 noundef 0, i32 noundef %310) #12
  %368 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %348, <4 x i32> noundef %301, <4 x float> noundef %366, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %351, i32 noundef 2, i32 noundef %310) #12
  %369 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %349, <4 x i32> noundef %301, <4 x float> noundef %367, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %351, i32 noundef 2, i32 noundef %310) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %370 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %356, i32 %71, i32 2)
  %371 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %358, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %372 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %346, <4 x i32> noundef %306, <4 x float> noundef %339, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %351, i32 noundef 1, i32 noundef %310) #12
  %373 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %347, <4 x i32> noundef %306, <4 x float> noundef %340, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %351, i32 noundef 1, i32 noundef %310) #12
  %374 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %348, <4 x i32> noundef %307, <4 x float> noundef %372, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %351, i32 noundef 3, i32 noundef %310) #12
  %375 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %349, <4 x i32> noundef %307, <4 x float> noundef %373, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %351, i32 noundef 3, i32 noundef %310) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %376 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %356, i32 %75, i32 2)
  %377 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %358, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %378 = or disjoint i32 %134, 1792
  %379 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %378, i32 %83, i32 0)
  %380 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %378, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %381 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %382 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %383 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %384 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %385 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %343
  %386 = load i32, ptr addrspace(3) %385, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 1024, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %387 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %381, <4 x i32> noundef %322, <4 x float> noundef %354, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %386, i32 noundef 0, i32 noundef %344) #12
  %388 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %382, <4 x i32> noundef %322, <4 x float> noundef %355, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %386, i32 noundef 0, i32 noundef %344) #12
  %389 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %383, <4 x i32> noundef %324, <4 x float> noundef %387, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %386, i32 noundef 2, i32 noundef %344) #12
  %390 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %384, <4 x i32> noundef %324, <4 x float> noundef %388, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %386, i32 noundef 2, i32 noundef %344) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %391 = or disjoint i32 %122, 16384
  %392 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %391, i32 %63, i32 2)
  %393 = or disjoint i32 %122, 17408
  %394 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %393, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %395 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %381, <4 x i32> noundef %329, <4 x float> noundef %362, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %386, i32 noundef 1, i32 noundef %344) #12
  %396 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %382, <4 x i32> noundef %329, <4 x float> noundef %363, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %386, i32 noundef 1, i32 noundef %344) #12
  %397 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %383, <4 x i32> noundef %330, <4 x float> noundef %395, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %386, i32 noundef 3, i32 noundef %344) #12
  %398 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %384, <4 x i32> noundef %330, <4 x float> noundef %396, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %386, i32 noundef 3, i32 noundef %344) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %399 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %391, i32 %67, i32 2)
  %400 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %393, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %401 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %381, <4 x i32> noundef %335, <4 x float> noundef %368, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %386, i32 noundef 0, i32 noundef %345) #12
  %402 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %382, <4 x i32> noundef %335, <4 x float> noundef %369, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %386, i32 noundef 0, i32 noundef %345) #12
  %403 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %383, <4 x i32> noundef %336, <4 x float> noundef %401, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %386, i32 noundef 2, i32 noundef %345) #12
  %404 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %384, <4 x i32> noundef %336, <4 x float> noundef %402, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %386, i32 noundef 2, i32 noundef %345) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %405 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %391, i32 %71, i32 2)
  %406 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %393, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %407 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %381, <4 x i32> noundef %341, <4 x float> noundef %374, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %386, i32 noundef 1, i32 noundef %345) #12
  %408 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %382, <4 x i32> noundef %341, <4 x float> noundef %375, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %386, i32 noundef 1, i32 noundef %345) #12
  %409 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %383, <4 x i32> noundef %342, <4 x float> noundef %407, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %386, i32 noundef 3, i32 noundef %345) #12
  %410 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %384, <4 x i32> noundef %342, <4 x float> noundef %408, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %386, i32 noundef 3, i32 noundef %345) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %411 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %391, i32 %75, i32 2)
  %412 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %393, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %413 = or disjoint i32 %134, 2048
  %414 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %413, i32 %83, i32 0)
  %415 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %413, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %416 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %417 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %418 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %419 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %420 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %378
  %421 = load i32, ptr addrspace(3) %420, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 1152, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %422 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %416, <4 x i32> noundef %357, <4 x float> noundef %389, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %421, i32 noundef 0, i32 noundef %379) #12
  %423 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %417, <4 x i32> noundef %357, <4 x float> noundef %390, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %421, i32 noundef 0, i32 noundef %379) #12
  %424 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %418, <4 x i32> noundef %359, <4 x float> noundef %422, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %421, i32 noundef 2, i32 noundef %379) #12
  %425 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %419, <4 x i32> noundef %359, <4 x float> noundef %423, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %421, i32 noundef 2, i32 noundef %379) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %426 = or disjoint i32 %122, 18432
  %427 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %426, i32 %63, i32 2)
  %428 = or disjoint i32 %122, 19456
  %429 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %428, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %430 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %416, <4 x i32> noundef %364, <4 x float> noundef %397, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %421, i32 noundef 1, i32 noundef %379) #12
  %431 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %417, <4 x i32> noundef %364, <4 x float> noundef %398, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %421, i32 noundef 1, i32 noundef %379) #12
  %432 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %418, <4 x i32> noundef %365, <4 x float> noundef %430, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %421, i32 noundef 3, i32 noundef %379) #12
  %433 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %419, <4 x i32> noundef %365, <4 x float> noundef %431, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %421, i32 noundef 3, i32 noundef %379) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %434 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %426, i32 %67, i32 2)
  %435 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %428, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %436 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %416, <4 x i32> noundef %370, <4 x float> noundef %403, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %421, i32 noundef 0, i32 noundef %380) #12
  %437 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %417, <4 x i32> noundef %370, <4 x float> noundef %404, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %421, i32 noundef 0, i32 noundef %380) #12
  %438 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %418, <4 x i32> noundef %371, <4 x float> noundef %436, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %421, i32 noundef 2, i32 noundef %380) #12
  %439 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %419, <4 x i32> noundef %371, <4 x float> noundef %437, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %421, i32 noundef 2, i32 noundef %380) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %440 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %426, i32 %71, i32 2)
  %441 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %428, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %442 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %416, <4 x i32> noundef %376, <4 x float> noundef %409, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %421, i32 noundef 1, i32 noundef %380) #12
  %443 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %417, <4 x i32> noundef %376, <4 x float> noundef %410, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %421, i32 noundef 1, i32 noundef %380) #12
  %444 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %418, <4 x i32> noundef %377, <4 x float> noundef %442, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %421, i32 noundef 3, i32 noundef %380) #12
  %445 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %419, <4 x i32> noundef %377, <4 x float> noundef %443, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %421, i32 noundef 3, i32 noundef %380) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %446 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %426, i32 %75, i32 2)
  %447 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %428, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %448 = or disjoint i32 %134, 2304
  %449 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %448, i32 %83, i32 0)
  %450 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %448, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %451 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %452 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %453 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %454 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %455 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %413
  %456 = load i32, ptr addrspace(3) %455, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 1280, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %457 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %451, <4 x i32> noundef %392, <4 x float> noundef %424, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %456, i32 noundef 0, i32 noundef %414) #12
  %458 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %452, <4 x i32> noundef %392, <4 x float> noundef %425, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %456, i32 noundef 0, i32 noundef %414) #12
  %459 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %453, <4 x i32> noundef %394, <4 x float> noundef %457, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %456, i32 noundef 2, i32 noundef %414) #12
  %460 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %454, <4 x i32> noundef %394, <4 x float> noundef %458, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %456, i32 noundef 2, i32 noundef %414) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %461 = or disjoint i32 %122, 20480
  %462 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %461, i32 %63, i32 2)
  %463 = or disjoint i32 %122, 21504
  %464 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %463, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %465 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %451, <4 x i32> noundef %399, <4 x float> noundef %432, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %456, i32 noundef 1, i32 noundef %414) #12
  %466 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %452, <4 x i32> noundef %399, <4 x float> noundef %433, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %456, i32 noundef 1, i32 noundef %414) #12
  %467 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %453, <4 x i32> noundef %400, <4 x float> noundef %465, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %456, i32 noundef 3, i32 noundef %414) #12
  %468 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %454, <4 x i32> noundef %400, <4 x float> noundef %466, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %456, i32 noundef 3, i32 noundef %414) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %469 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %461, i32 %67, i32 2)
  %470 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %463, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %471 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %451, <4 x i32> noundef %405, <4 x float> noundef %438, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %456, i32 noundef 0, i32 noundef %415) #12
  %472 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %452, <4 x i32> noundef %405, <4 x float> noundef %439, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %456, i32 noundef 0, i32 noundef %415) #12
  %473 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %453, <4 x i32> noundef %406, <4 x float> noundef %471, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %456, i32 noundef 2, i32 noundef %415) #12
  %474 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %454, <4 x i32> noundef %406, <4 x float> noundef %472, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %456, i32 noundef 2, i32 noundef %415) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %475 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %461, i32 %71, i32 2)
  %476 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %463, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %477 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %451, <4 x i32> noundef %411, <4 x float> noundef %444, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %456, i32 noundef 1, i32 noundef %415) #12
  %478 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %452, <4 x i32> noundef %411, <4 x float> noundef %445, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %456, i32 noundef 1, i32 noundef %415) #12
  %479 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %453, <4 x i32> noundef %412, <4 x float> noundef %477, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %456, i32 noundef 3, i32 noundef %415) #12
  %480 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %454, <4 x i32> noundef %412, <4 x float> noundef %478, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %456, i32 noundef 3, i32 noundef %415) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %481 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %461, i32 %75, i32 2)
  %482 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %463, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %483 = or disjoint i32 %134, 2560
  %484 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %483, i32 %83, i32 0)
  %485 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %483, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %486 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %487 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %488 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %489 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %490 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %448
  %491 = load i32, ptr addrspace(3) %490, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 1408, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %492 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %486, <4 x i32> noundef %427, <4 x float> noundef %459, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %491, i32 noundef 0, i32 noundef %449) #12
  %493 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %487, <4 x i32> noundef %427, <4 x float> noundef %460, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %491, i32 noundef 0, i32 noundef %449) #12
  %494 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %488, <4 x i32> noundef %429, <4 x float> noundef %492, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %491, i32 noundef 2, i32 noundef %449) #12
  %495 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %489, <4 x i32> noundef %429, <4 x float> noundef %493, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %491, i32 noundef 2, i32 noundef %449) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %496 = or disjoint i32 %122, 22528
  %497 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %496, i32 %63, i32 2)
  %498 = or disjoint i32 %122, 23552
  %499 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %498, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %500 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %486, <4 x i32> noundef %434, <4 x float> noundef %467, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %491, i32 noundef 1, i32 noundef %449) #12
  %501 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %487, <4 x i32> noundef %434, <4 x float> noundef %468, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %491, i32 noundef 1, i32 noundef %449) #12
  %502 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %488, <4 x i32> noundef %435, <4 x float> noundef %500, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %491, i32 noundef 3, i32 noundef %449) #12
  %503 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %489, <4 x i32> noundef %435, <4 x float> noundef %501, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %491, i32 noundef 3, i32 noundef %449) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %504 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %496, i32 %67, i32 2)
  %505 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %498, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %506 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %486, <4 x i32> noundef %440, <4 x float> noundef %473, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %491, i32 noundef 0, i32 noundef %450) #12
  %507 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %487, <4 x i32> noundef %440, <4 x float> noundef %474, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %491, i32 noundef 0, i32 noundef %450) #12
  %508 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %488, <4 x i32> noundef %441, <4 x float> noundef %506, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %491, i32 noundef 2, i32 noundef %450) #12
  %509 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %489, <4 x i32> noundef %441, <4 x float> noundef %507, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %491, i32 noundef 2, i32 noundef %450) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %510 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %496, i32 %71, i32 2)
  %511 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %498, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %512 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %486, <4 x i32> noundef %446, <4 x float> noundef %479, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %491, i32 noundef 1, i32 noundef %450) #12
  %513 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %487, <4 x i32> noundef %446, <4 x float> noundef %480, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %491, i32 noundef 1, i32 noundef %450) #12
  %514 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %488, <4 x i32> noundef %447, <4 x float> noundef %512, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %491, i32 noundef 3, i32 noundef %450) #12
  %515 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %489, <4 x i32> noundef %447, <4 x float> noundef %513, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %491, i32 noundef 3, i32 noundef %450) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %516 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %496, i32 %75, i32 2)
  %517 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %498, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %518 = or disjoint i32 %134, 2816
  %519 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %518, i32 %83, i32 0)
  %520 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %518, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %521 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %522 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %523 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %524 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %525 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %483
  %526 = load i32, ptr addrspace(3) %525, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 1536, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %527 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %521, <4 x i32> noundef %462, <4 x float> noundef %494, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %526, i32 noundef 0, i32 noundef %484) #12
  %528 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %522, <4 x i32> noundef %462, <4 x float> noundef %495, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %526, i32 noundef 0, i32 noundef %484) #12
  %529 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %523, <4 x i32> noundef %464, <4 x float> noundef %527, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %526, i32 noundef 2, i32 noundef %484) #12
  %530 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %524, <4 x i32> noundef %464, <4 x float> noundef %528, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %526, i32 noundef 2, i32 noundef %484) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %531 = or disjoint i32 %122, 24576
  %532 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %531, i32 %63, i32 2)
  %533 = or disjoint i32 %122, 25600
  %534 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %533, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %535 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %521, <4 x i32> noundef %469, <4 x float> noundef %502, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %526, i32 noundef 1, i32 noundef %484) #12
  %536 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %522, <4 x i32> noundef %469, <4 x float> noundef %503, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %526, i32 noundef 1, i32 noundef %484) #12
  %537 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %523, <4 x i32> noundef %470, <4 x float> noundef %535, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %526, i32 noundef 3, i32 noundef %484) #12
  %538 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %524, <4 x i32> noundef %470, <4 x float> noundef %536, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %526, i32 noundef 3, i32 noundef %484) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %539 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %531, i32 %67, i32 2)
  %540 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %533, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %541 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %521, <4 x i32> noundef %475, <4 x float> noundef %508, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %526, i32 noundef 0, i32 noundef %485) #12
  %542 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %522, <4 x i32> noundef %475, <4 x float> noundef %509, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %526, i32 noundef 0, i32 noundef %485) #12
  %543 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %523, <4 x i32> noundef %476, <4 x float> noundef %541, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %526, i32 noundef 2, i32 noundef %485) #12
  %544 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %524, <4 x i32> noundef %476, <4 x float> noundef %542, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %526, i32 noundef 2, i32 noundef %485) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %545 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %531, i32 %71, i32 2)
  %546 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %533, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %547 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %521, <4 x i32> noundef %481, <4 x float> noundef %514, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %526, i32 noundef 1, i32 noundef %485) #12
  %548 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %522, <4 x i32> noundef %481, <4 x float> noundef %515, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %526, i32 noundef 1, i32 noundef %485) #12
  %549 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %523, <4 x i32> noundef %482, <4 x float> noundef %547, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %526, i32 noundef 3, i32 noundef %485) #12
  %550 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %524, <4 x i32> noundef %482, <4 x float> noundef %548, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %526, i32 noundef 3, i32 noundef %485) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %551 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %531, i32 %75, i32 2)
  %552 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %533, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %553 = or disjoint i32 %134, 3072
  %554 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %553, i32 %83, i32 0)
  %555 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %553, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %556 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %557 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %558 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %559 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %560 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %518
  %561 = load i32, ptr addrspace(3) %560, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 1664, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %562 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %556, <4 x i32> noundef %497, <4 x float> noundef %529, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %561, i32 noundef 0, i32 noundef %519) #12
  %563 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %557, <4 x i32> noundef %497, <4 x float> noundef %530, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %561, i32 noundef 0, i32 noundef %519) #12
  %564 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %558, <4 x i32> noundef %499, <4 x float> noundef %562, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %561, i32 noundef 2, i32 noundef %519) #12
  %565 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %559, <4 x i32> noundef %499, <4 x float> noundef %563, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %561, i32 noundef 2, i32 noundef %519) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %566 = or disjoint i32 %122, 26624
  %567 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %566, i32 %63, i32 2)
  %568 = or disjoint i32 %122, 27648
  %569 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %568, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %570 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %556, <4 x i32> noundef %504, <4 x float> noundef %537, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %561, i32 noundef 1, i32 noundef %519) #12
  %571 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %557, <4 x i32> noundef %504, <4 x float> noundef %538, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %561, i32 noundef 1, i32 noundef %519) #12
  %572 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %558, <4 x i32> noundef %505, <4 x float> noundef %570, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %561, i32 noundef 3, i32 noundef %519) #12
  %573 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %559, <4 x i32> noundef %505, <4 x float> noundef %571, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %561, i32 noundef 3, i32 noundef %519) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %574 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %566, i32 %67, i32 2)
  %575 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %568, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %576 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %556, <4 x i32> noundef %510, <4 x float> noundef %543, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %561, i32 noundef 0, i32 noundef %520) #12
  %577 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %557, <4 x i32> noundef %510, <4 x float> noundef %544, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %561, i32 noundef 0, i32 noundef %520) #12
  %578 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %558, <4 x i32> noundef %511, <4 x float> noundef %576, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %561, i32 noundef 2, i32 noundef %520) #12
  %579 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %559, <4 x i32> noundef %511, <4 x float> noundef %577, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %561, i32 noundef 2, i32 noundef %520) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %580 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %566, i32 %71, i32 2)
  %581 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %568, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %582 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %556, <4 x i32> noundef %516, <4 x float> noundef %549, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %561, i32 noundef 1, i32 noundef %520) #12
  %583 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %557, <4 x i32> noundef %516, <4 x float> noundef %550, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %561, i32 noundef 1, i32 noundef %520) #12
  %584 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %558, <4 x i32> noundef %517, <4 x float> noundef %582, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %561, i32 noundef 3, i32 noundef %520) #12
  %585 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %559, <4 x i32> noundef %517, <4 x float> noundef %583, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %561, i32 noundef 3, i32 noundef %520) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %586 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %566, i32 %75, i32 2)
  %587 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %568, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %588 = or disjoint i32 %134, 3328
  %589 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %588, i32 %83, i32 0)
  %590 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %588, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %591 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %592 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %593 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %594 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %595 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %553
  %596 = load i32, ptr addrspace(3) %595, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 1792, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %597 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %591, <4 x i32> noundef %532, <4 x float> noundef %564, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %596, i32 noundef 0, i32 noundef %554) #12
  %598 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %592, <4 x i32> noundef %532, <4 x float> noundef %565, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %596, i32 noundef 0, i32 noundef %554) #12
  %599 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %593, <4 x i32> noundef %534, <4 x float> noundef %597, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %596, i32 noundef 2, i32 noundef %554) #12
  %600 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %594, <4 x i32> noundef %534, <4 x float> noundef %598, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %596, i32 noundef 2, i32 noundef %554) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %601 = or disjoint i32 %122, 28672
  %602 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %601, i32 %63, i32 2)
  %603 = or disjoint i32 %122, 29696
  %604 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %603, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %605 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %591, <4 x i32> noundef %539, <4 x float> noundef %572, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %596, i32 noundef 1, i32 noundef %554) #12
  %606 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %592, <4 x i32> noundef %539, <4 x float> noundef %573, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %596, i32 noundef 1, i32 noundef %554) #12
  %607 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %593, <4 x i32> noundef %540, <4 x float> noundef %605, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %596, i32 noundef 3, i32 noundef %554) #12
  %608 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %594, <4 x i32> noundef %540, <4 x float> noundef %606, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %596, i32 noundef 3, i32 noundef %554) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %609 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %601, i32 %67, i32 2)
  %610 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %603, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %611 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %591, <4 x i32> noundef %545, <4 x float> noundef %578, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %596, i32 noundef 0, i32 noundef %555) #12
  %612 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %592, <4 x i32> noundef %545, <4 x float> noundef %579, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %596, i32 noundef 0, i32 noundef %555) #12
  %613 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %593, <4 x i32> noundef %546, <4 x float> noundef %611, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %596, i32 noundef 2, i32 noundef %555) #12
  %614 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %594, <4 x i32> noundef %546, <4 x float> noundef %612, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %596, i32 noundef 2, i32 noundef %555) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %615 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %601, i32 %71, i32 2)
  %616 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %603, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %617 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %591, <4 x i32> noundef %551, <4 x float> noundef %584, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %596, i32 noundef 1, i32 noundef %555) #12
  %618 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %592, <4 x i32> noundef %551, <4 x float> noundef %585, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %596, i32 noundef 1, i32 noundef %555) #12
  %619 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %593, <4 x i32> noundef %552, <4 x float> noundef %617, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %596, i32 noundef 3, i32 noundef %555) #12
  %620 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %594, <4 x i32> noundef %552, <4 x float> noundef %618, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %596, i32 noundef 3, i32 noundef %555) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %621 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %601, i32 %75, i32 2)
  %622 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %603, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %623 = or disjoint i32 %134, 3584
  %624 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %623, i32 %83, i32 0)
  %625 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %623, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %626 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %627 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %628 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %629 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %630 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %588
  %631 = load i32, ptr addrspace(3) %630, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 1920, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %632 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %626, <4 x i32> noundef %567, <4 x float> noundef %599, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %631, i32 noundef 0, i32 noundef %589) #12
  %633 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %627, <4 x i32> noundef %567, <4 x float> noundef %600, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %631, i32 noundef 0, i32 noundef %589) #12
  %634 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %628, <4 x i32> noundef %569, <4 x float> noundef %632, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %631, i32 noundef 2, i32 noundef %589) #12
  %635 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %629, <4 x i32> noundef %569, <4 x float> noundef %633, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %631, i32 noundef 2, i32 noundef %589) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %636 = or disjoint i32 %122, 30720
  %637 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %636, i32 %63, i32 2)
  %638 = or disjoint i32 %122, 31744
  %639 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %638, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %640 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %626, <4 x i32> noundef %574, <4 x float> noundef %607, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %631, i32 noundef 1, i32 noundef %589) #12
  %641 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %627, <4 x i32> noundef %574, <4 x float> noundef %608, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %631, i32 noundef 1, i32 noundef %589) #12
  %642 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %628, <4 x i32> noundef %575, <4 x float> noundef %640, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %631, i32 noundef 3, i32 noundef %589) #12
  %643 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %629, <4 x i32> noundef %575, <4 x float> noundef %641, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %631, i32 noundef 3, i32 noundef %589) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %644 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %636, i32 %67, i32 2)
  %645 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %638, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %646 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %626, <4 x i32> noundef %580, <4 x float> noundef %613, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %631, i32 noundef 0, i32 noundef %590) #12
  %647 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %627, <4 x i32> noundef %580, <4 x float> noundef %614, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %631, i32 noundef 0, i32 noundef %590) #12
  %648 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %628, <4 x i32> noundef %581, <4 x float> noundef %646, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %631, i32 noundef 2, i32 noundef %590) #12
  %649 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %629, <4 x i32> noundef %581, <4 x float> noundef %647, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %631, i32 noundef 2, i32 noundef %590) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %650 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %636, i32 %71, i32 2)
  %651 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %638, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %652 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %626, <4 x i32> noundef %586, <4 x float> noundef %619, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %631, i32 noundef 1, i32 noundef %590) #12
  %653 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %627, <4 x i32> noundef %586, <4 x float> noundef %620, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %631, i32 noundef 1, i32 noundef %590) #12
  %654 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %628, <4 x i32> noundef %587, <4 x float> noundef %652, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %631, i32 noundef 3, i32 noundef %590) #12
  %655 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %629, <4 x i32> noundef %587, <4 x float> noundef %653, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %631, i32 noundef 3, i32 noundef %590) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %656 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %636, i32 %75, i32 2)
  %657 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %638, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %658 = or disjoint i32 %134, 3840
  %659 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %658, i32 %83, i32 0)
  %660 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %658, i32 %88, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %661 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %662 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %663 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %664 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %665 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %623
  %666 = load i32, ptr addrspace(3) %665, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 2048, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %667 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %661, <4 x i32> noundef %602, <4 x float> noundef %634, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %666, i32 noundef 0, i32 noundef %624) #12
  %668 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %662, <4 x i32> noundef %602, <4 x float> noundef %635, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %666, i32 noundef 0, i32 noundef %624) #12
  %669 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %663, <4 x i32> noundef %604, <4 x float> noundef %667, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %666, i32 noundef 2, i32 noundef %624) #12
  %670 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %664, <4 x i32> noundef %604, <4 x float> noundef %668, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %666, i32 noundef 2, i32 noundef %624) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %671 = or disjoint i32 %122, 32768
  %672 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %671, i32 %63, i32 2)
  %673 = or disjoint i32 %122, 33792
  %674 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %673, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %675 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %661, <4 x i32> noundef %609, <4 x float> noundef %642, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %666, i32 noundef 1, i32 noundef %624) #12
  %676 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %662, <4 x i32> noundef %609, <4 x float> noundef %643, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %666, i32 noundef 1, i32 noundef %624) #12
  %677 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %663, <4 x i32> noundef %610, <4 x float> noundef %675, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %666, i32 noundef 3, i32 noundef %624) #12
  %678 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %664, <4 x i32> noundef %610, <4 x float> noundef %676, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %666, i32 noundef 3, i32 noundef %624) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %679 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %671, i32 %67, i32 2)
  %680 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %673, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %681 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %661, <4 x i32> noundef %615, <4 x float> noundef %648, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %666, i32 noundef 0, i32 noundef %625) #12
  %682 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %662, <4 x i32> noundef %615, <4 x float> noundef %649, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %666, i32 noundef 0, i32 noundef %625) #12
  %683 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %663, <4 x i32> noundef %616, <4 x float> noundef %681, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %666, i32 noundef 2, i32 noundef %625) #12
  %684 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %664, <4 x i32> noundef %616, <4 x float> noundef %682, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %666, i32 noundef 2, i32 noundef %625) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %685 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %671, i32 %71, i32 2)
  %686 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %673, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %687 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %661, <4 x i32> noundef %621, <4 x float> noundef %654, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %666, i32 noundef 1, i32 noundef %625) #12
  %688 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %662, <4 x i32> noundef %621, <4 x float> noundef %655, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %666, i32 noundef 1, i32 noundef %625) #12
  %689 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %663, <4 x i32> noundef %622, <4 x float> noundef %687, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %666, i32 noundef 3, i32 noundef %625) #12
  %690 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %664, <4 x i32> noundef %622, <4 x float> noundef %688, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %666, i32 noundef 3, i32 noundef %625) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %691 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %671, i32 %75, i32 2)
  %692 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %673, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %693 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %134, i32 %85, i32 0)
  %694 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %134, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %695 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %696 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %697 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %698 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %699 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %658
  %700 = load i32, ptr addrspace(3) %699, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 2176, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %701 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %695, <4 x i32> noundef %637, <4 x float> noundef %669, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %700, i32 noundef 0, i32 noundef %659) #12
  %702 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %696, <4 x i32> noundef %637, <4 x float> noundef %670, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %700, i32 noundef 0, i32 noundef %659) #12
  %703 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %697, <4 x i32> noundef %639, <4 x float> noundef %701, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %700, i32 noundef 2, i32 noundef %659) #12
  %704 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %698, <4 x i32> noundef %639, <4 x float> noundef %702, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %700, i32 noundef 2, i32 noundef %659) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %705 = or disjoint i32 %122, 34816
  %706 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %705, i32 %63, i32 2)
  %707 = or disjoint i32 %122, 35840
  %708 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %707, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %709 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %695, <4 x i32> noundef %644, <4 x float> noundef %677, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %700, i32 noundef 1, i32 noundef %659) #12
  %710 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %696, <4 x i32> noundef %644, <4 x float> noundef %678, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %700, i32 noundef 1, i32 noundef %659) #12
  %711 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %697, <4 x i32> noundef %645, <4 x float> noundef %709, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %700, i32 noundef 3, i32 noundef %659) #12
  %712 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %698, <4 x i32> noundef %645, <4 x float> noundef %710, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %700, i32 noundef 3, i32 noundef %659) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %713 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %705, i32 %67, i32 2)
  %714 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %707, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %715 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %695, <4 x i32> noundef %650, <4 x float> noundef %683, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %700, i32 noundef 0, i32 noundef %660) #12
  %716 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %696, <4 x i32> noundef %650, <4 x float> noundef %684, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %700, i32 noundef 0, i32 noundef %660) #12
  %717 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %697, <4 x i32> noundef %651, <4 x float> noundef %715, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %700, i32 noundef 2, i32 noundef %660) #12
  %718 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %698, <4 x i32> noundef %651, <4 x float> noundef %716, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %700, i32 noundef 2, i32 noundef %660) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %719 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %705, i32 %71, i32 2)
  %720 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %707, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %721 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %695, <4 x i32> noundef %656, <4 x float> noundef %689, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %700, i32 noundef 1, i32 noundef %660) #12
  %722 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %696, <4 x i32> noundef %656, <4 x float> noundef %690, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %700, i32 noundef 1, i32 noundef %660) #12
  %723 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %697, <4 x i32> noundef %657, <4 x float> noundef %721, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %700, i32 noundef 3, i32 noundef %660) #12
  %724 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %698, <4 x i32> noundef %657, <4 x float> noundef %722, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %700, i32 noundef 3, i32 noundef %660) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %725 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %705, i32 %75, i32 2)
  %726 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %707, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %727 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %148, i32 %85, i32 0)
  %728 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %148, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %729 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %730 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %731 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %732 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %733 = or disjoint i32 %134, 4096
  %734 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %733
  %735 = load i32, ptr addrspace(3) %734, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 2304, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %736 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %729, <4 x i32> noundef %672, <4 x float> noundef %703, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %735, i32 noundef 0, i32 noundef %693) #12
  %737 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %730, <4 x i32> noundef %672, <4 x float> noundef %704, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %735, i32 noundef 0, i32 noundef %693) #12
  %738 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %731, <4 x i32> noundef %674, <4 x float> noundef %736, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %735, i32 noundef 2, i32 noundef %693) #12
  %739 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %732, <4 x i32> noundef %674, <4 x float> noundef %737, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %735, i32 noundef 2, i32 noundef %693) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %740 = or disjoint i32 %122, 36864
  %741 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %740, i32 %63, i32 2)
  %742 = or disjoint i32 %122, 37888
  %743 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %742, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %744 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %729, <4 x i32> noundef %679, <4 x float> noundef %711, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %735, i32 noundef 1, i32 noundef %693) #12
  %745 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %730, <4 x i32> noundef %679, <4 x float> noundef %712, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %735, i32 noundef 1, i32 noundef %693) #12
  %746 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %731, <4 x i32> noundef %680, <4 x float> noundef %744, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %735, i32 noundef 3, i32 noundef %693) #12
  %747 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %732, <4 x i32> noundef %680, <4 x float> noundef %745, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %735, i32 noundef 3, i32 noundef %693) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %748 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %740, i32 %67, i32 2)
  %749 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %742, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %750 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %729, <4 x i32> noundef %685, <4 x float> noundef %717, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %735, i32 noundef 0, i32 noundef %694) #12
  %751 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %730, <4 x i32> noundef %685, <4 x float> noundef %718, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %735, i32 noundef 0, i32 noundef %694) #12
  %752 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %731, <4 x i32> noundef %686, <4 x float> noundef %750, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %735, i32 noundef 2, i32 noundef %694) #12
  %753 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %732, <4 x i32> noundef %686, <4 x float> noundef %751, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %735, i32 noundef 2, i32 noundef %694) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %754 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %740, i32 %71, i32 2)
  %755 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %742, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %756 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %729, <4 x i32> noundef %691, <4 x float> noundef %723, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %735, i32 noundef 1, i32 noundef %694) #12
  %757 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %730, <4 x i32> noundef %691, <4 x float> noundef %724, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %735, i32 noundef 1, i32 noundef %694) #12
  %758 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %731, <4 x i32> noundef %692, <4 x float> noundef %756, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %735, i32 noundef 3, i32 noundef %694) #12
  %759 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %732, <4 x i32> noundef %692, <4 x float> noundef %757, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %735, i32 noundef 3, i32 noundef %694) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %760 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %740, i32 %75, i32 2)
  %761 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %742, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %762 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %195, i32 %85, i32 0)
  %763 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %195, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %764 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %765 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %766 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %767 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %768 = or disjoint i32 %134, 4352
  %769 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %768
  %770 = load i32, ptr addrspace(3) %769, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 2432, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %771 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %764, <4 x i32> noundef %706, <4 x float> noundef %738, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %770, i32 noundef 0, i32 noundef %727) #12
  %772 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %765, <4 x i32> noundef %706, <4 x float> noundef %739, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %770, i32 noundef 0, i32 noundef %727) #12
  %773 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %766, <4 x i32> noundef %708, <4 x float> noundef %771, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %770, i32 noundef 2, i32 noundef %727) #12
  %774 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %767, <4 x i32> noundef %708, <4 x float> noundef %772, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %770, i32 noundef 2, i32 noundef %727) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %775 = or disjoint i32 %122, 38912
  %776 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %775, i32 %63, i32 2)
  %777 = or disjoint i32 %122, 39936
  %778 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %777, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %779 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %764, <4 x i32> noundef %713, <4 x float> noundef %746, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %770, i32 noundef 1, i32 noundef %727) #12
  %780 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %765, <4 x i32> noundef %713, <4 x float> noundef %747, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %770, i32 noundef 1, i32 noundef %727) #12
  %781 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %766, <4 x i32> noundef %714, <4 x float> noundef %779, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %770, i32 noundef 3, i32 noundef %727) #12
  %782 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %767, <4 x i32> noundef %714, <4 x float> noundef %780, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %770, i32 noundef 3, i32 noundef %727) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %783 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %775, i32 %67, i32 2)
  %784 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %777, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %785 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %764, <4 x i32> noundef %719, <4 x float> noundef %752, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %770, i32 noundef 0, i32 noundef %728) #12
  %786 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %765, <4 x i32> noundef %719, <4 x float> noundef %753, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %770, i32 noundef 0, i32 noundef %728) #12
  %787 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %766, <4 x i32> noundef %720, <4 x float> noundef %785, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %770, i32 noundef 2, i32 noundef %728) #12
  %788 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %767, <4 x i32> noundef %720, <4 x float> noundef %786, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %770, i32 noundef 2, i32 noundef %728) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %789 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %775, i32 %71, i32 2)
  %790 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %777, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %791 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %764, <4 x i32> noundef %725, <4 x float> noundef %758, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %770, i32 noundef 1, i32 noundef %728) #12
  %792 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %765, <4 x i32> noundef %725, <4 x float> noundef %759, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %770, i32 noundef 1, i32 noundef %728) #12
  %793 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %766, <4 x i32> noundef %726, <4 x float> noundef %791, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %770, i32 noundef 3, i32 noundef %728) #12
  %794 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %767, <4 x i32> noundef %726, <4 x float> noundef %792, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %770, i32 noundef 3, i32 noundef %728) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %795 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %775, i32 %75, i32 2)
  %796 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %777, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %797 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %234, i32 %85, i32 0)
  %798 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %234, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %799 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %800 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %801 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %802 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %803 = or disjoint i32 %134, 4608
  %804 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %803
  %805 = load i32, ptr addrspace(3) %804, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 2560, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %806 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %799, <4 x i32> noundef %741, <4 x float> noundef %773, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %805, i32 noundef 0, i32 noundef %762) #12
  %807 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %800, <4 x i32> noundef %741, <4 x float> noundef %774, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %805, i32 noundef 0, i32 noundef %762) #12
  %808 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %801, <4 x i32> noundef %743, <4 x float> noundef %806, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %805, i32 noundef 2, i32 noundef %762) #12
  %809 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %802, <4 x i32> noundef %743, <4 x float> noundef %807, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %805, i32 noundef 2, i32 noundef %762) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %810 = or disjoint i32 %122, 40960
  %811 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %810, i32 %63, i32 2)
  %812 = or disjoint i32 %122, 41984
  %813 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %812, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %814 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %799, <4 x i32> noundef %748, <4 x float> noundef %781, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %805, i32 noundef 1, i32 noundef %762) #12
  %815 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %800, <4 x i32> noundef %748, <4 x float> noundef %782, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %805, i32 noundef 1, i32 noundef %762) #12
  %816 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %801, <4 x i32> noundef %749, <4 x float> noundef %814, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %805, i32 noundef 3, i32 noundef %762) #12
  %817 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %802, <4 x i32> noundef %749, <4 x float> noundef %815, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %805, i32 noundef 3, i32 noundef %762) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %818 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %810, i32 %67, i32 2)
  %819 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %812, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %820 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %799, <4 x i32> noundef %754, <4 x float> noundef %787, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %805, i32 noundef 0, i32 noundef %763) #12
  %821 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %800, <4 x i32> noundef %754, <4 x float> noundef %788, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %805, i32 noundef 0, i32 noundef %763) #12
  %822 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %801, <4 x i32> noundef %755, <4 x float> noundef %820, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %805, i32 noundef 2, i32 noundef %763) #12
  %823 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %802, <4 x i32> noundef %755, <4 x float> noundef %821, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %805, i32 noundef 2, i32 noundef %763) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %824 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %810, i32 %71, i32 2)
  %825 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %812, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %826 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %799, <4 x i32> noundef %760, <4 x float> noundef %793, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %805, i32 noundef 1, i32 noundef %763) #12
  %827 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %800, <4 x i32> noundef %760, <4 x float> noundef %794, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %805, i32 noundef 1, i32 noundef %763) #12
  %828 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %801, <4 x i32> noundef %761, <4 x float> noundef %826, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %805, i32 noundef 3, i32 noundef %763) #12
  %829 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %802, <4 x i32> noundef %761, <4 x float> noundef %827, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %805, i32 noundef 3, i32 noundef %763) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %830 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %810, i32 %75, i32 2)
  %831 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %812, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %832 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %273, i32 %85, i32 0)
  %833 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %273, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %834 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %835 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %836 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %837 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %838 = or disjoint i32 %134, 4864
  %839 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %838
  %840 = load i32, ptr addrspace(3) %839, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 2688, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %841 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %834, <4 x i32> noundef %776, <4 x float> noundef %808, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %840, i32 noundef 0, i32 noundef %797) #12
  %842 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %835, <4 x i32> noundef %776, <4 x float> noundef %809, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %840, i32 noundef 0, i32 noundef %797) #12
  %843 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %836, <4 x i32> noundef %778, <4 x float> noundef %841, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %840, i32 noundef 2, i32 noundef %797) #12
  %844 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %837, <4 x i32> noundef %778, <4 x float> noundef %842, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %840, i32 noundef 2, i32 noundef %797) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %845 = or disjoint i32 %122, 43008
  %846 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %845, i32 %63, i32 2)
  %847 = or disjoint i32 %122, 44032
  %848 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %847, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %849 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %834, <4 x i32> noundef %783, <4 x float> noundef %816, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %840, i32 noundef 1, i32 noundef %797) #12
  %850 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %835, <4 x i32> noundef %783, <4 x float> noundef %817, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %840, i32 noundef 1, i32 noundef %797) #12
  %851 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %836, <4 x i32> noundef %784, <4 x float> noundef %849, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %840, i32 noundef 3, i32 noundef %797) #12
  %852 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %837, <4 x i32> noundef %784, <4 x float> noundef %850, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %840, i32 noundef 3, i32 noundef %797) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %853 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %845, i32 %67, i32 2)
  %854 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %847, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %855 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %834, <4 x i32> noundef %789, <4 x float> noundef %822, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %840, i32 noundef 0, i32 noundef %798) #12
  %856 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %835, <4 x i32> noundef %789, <4 x float> noundef %823, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %840, i32 noundef 0, i32 noundef %798) #12
  %857 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %836, <4 x i32> noundef %790, <4 x float> noundef %855, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %840, i32 noundef 2, i32 noundef %798) #12
  %858 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %837, <4 x i32> noundef %790, <4 x float> noundef %856, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %840, i32 noundef 2, i32 noundef %798) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %859 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %845, i32 %71, i32 2)
  %860 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %847, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %861 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %834, <4 x i32> noundef %795, <4 x float> noundef %828, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %840, i32 noundef 1, i32 noundef %798) #12
  %862 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %835, <4 x i32> noundef %795, <4 x float> noundef %829, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %840, i32 noundef 1, i32 noundef %798) #12
  %863 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %836, <4 x i32> noundef %796, <4 x float> noundef %861, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %840, i32 noundef 3, i32 noundef %798) #12
  %864 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %837, <4 x i32> noundef %796, <4 x float> noundef %862, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %840, i32 noundef 3, i32 noundef %798) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %865 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %845, i32 %75, i32 2)
  %866 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %847, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %867 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %308, i32 %85, i32 0)
  %868 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %308, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %869 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %870 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %871 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %872 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %873 = or disjoint i32 %134, 5120
  %874 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %873
  %875 = load i32, ptr addrspace(3) %874, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 2816, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %876 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %869, <4 x i32> noundef %811, <4 x float> noundef %843, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %875, i32 noundef 0, i32 noundef %832) #12
  %877 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %870, <4 x i32> noundef %811, <4 x float> noundef %844, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %875, i32 noundef 0, i32 noundef %832) #12
  %878 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %871, <4 x i32> noundef %813, <4 x float> noundef %876, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %875, i32 noundef 2, i32 noundef %832) #12
  %879 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %872, <4 x i32> noundef %813, <4 x float> noundef %877, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %875, i32 noundef 2, i32 noundef %832) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %880 = or disjoint i32 %122, 45056
  %881 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %880, i32 %63, i32 2)
  %882 = or disjoint i32 %122, 46080
  %883 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %882, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %884 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %869, <4 x i32> noundef %818, <4 x float> noundef %851, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %875, i32 noundef 1, i32 noundef %832) #12
  %885 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %870, <4 x i32> noundef %818, <4 x float> noundef %852, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %875, i32 noundef 1, i32 noundef %832) #12
  %886 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %871, <4 x i32> noundef %819, <4 x float> noundef %884, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %875, i32 noundef 3, i32 noundef %832) #12
  %887 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %872, <4 x i32> noundef %819, <4 x float> noundef %885, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %875, i32 noundef 3, i32 noundef %832) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %888 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %880, i32 %67, i32 2)
  %889 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %882, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %890 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %869, <4 x i32> noundef %824, <4 x float> noundef %857, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %875, i32 noundef 0, i32 noundef %833) #12
  %891 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %870, <4 x i32> noundef %824, <4 x float> noundef %858, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %875, i32 noundef 0, i32 noundef %833) #12
  %892 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %871, <4 x i32> noundef %825, <4 x float> noundef %890, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %875, i32 noundef 2, i32 noundef %833) #12
  %893 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %872, <4 x i32> noundef %825, <4 x float> noundef %891, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %875, i32 noundef 2, i32 noundef %833) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %894 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %880, i32 %71, i32 2)
  %895 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %882, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %896 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %869, <4 x i32> noundef %830, <4 x float> noundef %863, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %875, i32 noundef 1, i32 noundef %833) #12
  %897 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %870, <4 x i32> noundef %830, <4 x float> noundef %864, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %875, i32 noundef 1, i32 noundef %833) #12
  %898 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %871, <4 x i32> noundef %831, <4 x float> noundef %896, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %875, i32 noundef 3, i32 noundef %833) #12
  %899 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %872, <4 x i32> noundef %831, <4 x float> noundef %897, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %875, i32 noundef 3, i32 noundef %833) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %900 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %880, i32 %75, i32 2)
  %901 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %882, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %902 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %343, i32 %85, i32 0)
  %903 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %343, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %904 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %905 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %906 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %907 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %908 = or disjoint i32 %134, 5376
  %909 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %908
  %910 = load i32, ptr addrspace(3) %909, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 2944, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %911 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %904, <4 x i32> noundef %846, <4 x float> noundef %878, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %910, i32 noundef 0, i32 noundef %867) #12
  %912 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %905, <4 x i32> noundef %846, <4 x float> noundef %879, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %910, i32 noundef 0, i32 noundef %867) #12
  %913 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %906, <4 x i32> noundef %848, <4 x float> noundef %911, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %910, i32 noundef 2, i32 noundef %867) #12
  %914 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %907, <4 x i32> noundef %848, <4 x float> noundef %912, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %910, i32 noundef 2, i32 noundef %867) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %915 = or disjoint i32 %122, 47104
  %916 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %915, i32 %63, i32 2)
  %917 = or disjoint i32 %122, 48128
  %918 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %917, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %919 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %904, <4 x i32> noundef %853, <4 x float> noundef %886, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %910, i32 noundef 1, i32 noundef %867) #12
  %920 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %905, <4 x i32> noundef %853, <4 x float> noundef %887, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %910, i32 noundef 1, i32 noundef %867) #12
  %921 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %906, <4 x i32> noundef %854, <4 x float> noundef %919, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %910, i32 noundef 3, i32 noundef %867) #12
  %922 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %907, <4 x i32> noundef %854, <4 x float> noundef %920, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %910, i32 noundef 3, i32 noundef %867) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %923 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %915, i32 %67, i32 2)
  %924 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %917, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %925 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %904, <4 x i32> noundef %859, <4 x float> noundef %892, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %910, i32 noundef 0, i32 noundef %868) #12
  %926 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %905, <4 x i32> noundef %859, <4 x float> noundef %893, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %910, i32 noundef 0, i32 noundef %868) #12
  %927 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %906, <4 x i32> noundef %860, <4 x float> noundef %925, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %910, i32 noundef 2, i32 noundef %868) #12
  %928 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %907, <4 x i32> noundef %860, <4 x float> noundef %926, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %910, i32 noundef 2, i32 noundef %868) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %929 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %915, i32 %71, i32 2)
  %930 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %917, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %931 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %904, <4 x i32> noundef %865, <4 x float> noundef %898, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %910, i32 noundef 1, i32 noundef %868) #12
  %932 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %905, <4 x i32> noundef %865, <4 x float> noundef %899, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %910, i32 noundef 1, i32 noundef %868) #12
  %933 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %906, <4 x i32> noundef %866, <4 x float> noundef %931, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %910, i32 noundef 3, i32 noundef %868) #12
  %934 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %907, <4 x i32> noundef %866, <4 x float> noundef %932, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %910, i32 noundef 3, i32 noundef %868) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %935 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %915, i32 %75, i32 2)
  %936 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %917, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %937 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %378, i32 %85, i32 0)
  %938 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %378, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %939 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %940 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %941 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %942 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %943 = or disjoint i32 %134, 5632
  %944 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %943
  %945 = load i32, ptr addrspace(3) %944, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 3072, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %946 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %939, <4 x i32> noundef %881, <4 x float> noundef %913, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %945, i32 noundef 0, i32 noundef %902) #12
  %947 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %940, <4 x i32> noundef %881, <4 x float> noundef %914, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %945, i32 noundef 0, i32 noundef %902) #12
  %948 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %941, <4 x i32> noundef %883, <4 x float> noundef %946, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %945, i32 noundef 2, i32 noundef %902) #12
  %949 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %942, <4 x i32> noundef %883, <4 x float> noundef %947, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %945, i32 noundef 2, i32 noundef %902) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %950 = or disjoint i32 %122, 49152
  %951 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %950, i32 %63, i32 2)
  %952 = or disjoint i32 %122, 50176
  %953 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %952, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %954 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %939, <4 x i32> noundef %888, <4 x float> noundef %921, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %945, i32 noundef 1, i32 noundef %902) #12
  %955 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %940, <4 x i32> noundef %888, <4 x float> noundef %922, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %945, i32 noundef 1, i32 noundef %902) #12
  %956 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %941, <4 x i32> noundef %889, <4 x float> noundef %954, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %945, i32 noundef 3, i32 noundef %902) #12
  %957 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %942, <4 x i32> noundef %889, <4 x float> noundef %955, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %945, i32 noundef 3, i32 noundef %902) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %958 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %950, i32 %67, i32 2)
  %959 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %952, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %960 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %939, <4 x i32> noundef %894, <4 x float> noundef %927, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %945, i32 noundef 0, i32 noundef %903) #12
  %961 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %940, <4 x i32> noundef %894, <4 x float> noundef %928, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %945, i32 noundef 0, i32 noundef %903) #12
  %962 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %941, <4 x i32> noundef %895, <4 x float> noundef %960, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %945, i32 noundef 2, i32 noundef %903) #12
  %963 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %942, <4 x i32> noundef %895, <4 x float> noundef %961, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %945, i32 noundef 2, i32 noundef %903) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %964 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %950, i32 %71, i32 2)
  %965 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %952, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %966 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %939, <4 x i32> noundef %900, <4 x float> noundef %933, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %945, i32 noundef 1, i32 noundef %903) #12
  %967 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %940, <4 x i32> noundef %900, <4 x float> noundef %934, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %945, i32 noundef 1, i32 noundef %903) #12
  %968 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %941, <4 x i32> noundef %901, <4 x float> noundef %966, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %945, i32 noundef 3, i32 noundef %903) #12
  %969 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %942, <4 x i32> noundef %901, <4 x float> noundef %967, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %945, i32 noundef 3, i32 noundef %903) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %970 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %950, i32 %75, i32 2)
  %971 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %952, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %972 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %413, i32 %85, i32 0)
  %973 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %413, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %974 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %975 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %976 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %977 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %978 = or disjoint i32 %134, 5888
  %979 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %978
  %980 = load i32, ptr addrspace(3) %979, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %137, i32 noundef 16, i32 noundef %116, i32 noundef 3200, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %981 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %974, <4 x i32> noundef %916, <4 x float> noundef %948, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %980, i32 noundef 0, i32 noundef %937) #12
  %982 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %975, <4 x i32> noundef %916, <4 x float> noundef %949, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %980, i32 noundef 0, i32 noundef %937) #12
  %983 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %976, <4 x i32> noundef %918, <4 x float> noundef %981, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %980, i32 noundef 2, i32 noundef %937) #12
  %984 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %977, <4 x i32> noundef %918, <4 x float> noundef %982, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %980, i32 noundef 2, i32 noundef %937) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %985 = or disjoint i32 %122, 51200
  %986 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %985, i32 %63, i32 2)
  %987 = or disjoint i32 %122, 52224
  %988 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %987, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %989 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %974, <4 x i32> noundef %923, <4 x float> noundef %956, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %980, i32 noundef 1, i32 noundef %937) #12
  %990 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %975, <4 x i32> noundef %923, <4 x float> noundef %957, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %980, i32 noundef 1, i32 noundef %937) #12
  %991 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %976, <4 x i32> noundef %924, <4 x float> noundef %989, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %980, i32 noundef 3, i32 noundef %937) #12
  %992 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %977, <4 x i32> noundef %924, <4 x float> noundef %990, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %980, i32 noundef 3, i32 noundef %937) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %993 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %985, i32 %67, i32 2)
  %994 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %987, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %995 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %974, <4 x i32> noundef %929, <4 x float> noundef %962, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %980, i32 noundef 0, i32 noundef %938) #12
  %996 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %975, <4 x i32> noundef %929, <4 x float> noundef %963, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %980, i32 noundef 0, i32 noundef %938) #12
  %997 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %976, <4 x i32> noundef %930, <4 x float> noundef %995, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %980, i32 noundef 2, i32 noundef %938) #12
  %998 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %977, <4 x i32> noundef %930, <4 x float> noundef %996, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %980, i32 noundef 2, i32 noundef %938) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %999 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %985, i32 %71, i32 2)
  %1000 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %987, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1001 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %974, <4 x i32> noundef %935, <4 x float> noundef %968, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %980, i32 noundef 1, i32 noundef %938) #12
  %1002 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %975, <4 x i32> noundef %935, <4 x float> noundef %969, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %980, i32 noundef 1, i32 noundef %938) #12
  %1003 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %976, <4 x i32> noundef %936, <4 x float> noundef %1001, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %980, i32 noundef 3, i32 noundef %938) #12
  %1004 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %977, <4 x i32> noundef %936, <4 x float> noundef %1002, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %980, i32 noundef 3, i32 noundef %938) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1005 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %985, i32 %75, i32 2)
  %1006 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %987, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1007 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %448, i32 %85, i32 0)
  %1008 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %448, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1009 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %1010 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %1011 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %1012 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %1013 = or disjoint i32 %134, 6144
  %1014 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %1013
  %1015 = load i32, ptr addrspace(3) %1014, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef nonnull %168, i32 noundef 16, i32 noundef %116, i32 noundef 3328, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1016 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1009, <4 x i32> noundef %951, <4 x float> noundef %983, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1015, i32 noundef 0, i32 noundef %972) #12
  %1017 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1010, <4 x i32> noundef %951, <4 x float> noundef %984, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1015, i32 noundef 0, i32 noundef %972) #12
  %1018 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1011, <4 x i32> noundef %953, <4 x float> noundef %1016, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1015, i32 noundef 2, i32 noundef %972) #12
  %1019 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1012, <4 x i32> noundef %953, <4 x float> noundef %1017, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1015, i32 noundef 2, i32 noundef %972) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1020 = or disjoint i32 %122, 53248
  %1021 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1020, i32 %63, i32 2)
  %1022 = or disjoint i32 %122, 54272
  %1023 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1022, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1024 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1009, <4 x i32> noundef %958, <4 x float> noundef %991, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1015, i32 noundef 1, i32 noundef %972) #12
  %1025 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1010, <4 x i32> noundef %958, <4 x float> noundef %992, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1015, i32 noundef 1, i32 noundef %972) #12
  %1026 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1011, <4 x i32> noundef %959, <4 x float> noundef %1024, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1015, i32 noundef 3, i32 noundef %972) #12
  %1027 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1012, <4 x i32> noundef %959, <4 x float> noundef %1025, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1015, i32 noundef 3, i32 noundef %972) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1028 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1020, i32 %67, i32 2)
  %1029 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1022, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1030 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1009, <4 x i32> noundef %964, <4 x float> noundef %997, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1015, i32 noundef 0, i32 noundef %973) #12
  %1031 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1010, <4 x i32> noundef %964, <4 x float> noundef %998, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1015, i32 noundef 0, i32 noundef %973) #12
  %1032 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1011, <4 x i32> noundef %965, <4 x float> noundef %1030, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1015, i32 noundef 2, i32 noundef %973) #12
  %1033 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1012, <4 x i32> noundef %965, <4 x float> noundef %1031, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1015, i32 noundef 2, i32 noundef %973) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1034 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1020, i32 %71, i32 2)
  %1035 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1022, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1036 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1009, <4 x i32> noundef %970, <4 x float> noundef %1003, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1015, i32 noundef 1, i32 noundef %973) #12
  %1037 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1010, <4 x i32> noundef %970, <4 x float> noundef %1004, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1015, i32 noundef 1, i32 noundef %973) #12
  %1038 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1011, <4 x i32> noundef %971, <4 x float> noundef %1036, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1015, i32 noundef 3, i32 noundef %973) #12
  %1039 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1012, <4 x i32> noundef %971, <4 x float> noundef %1037, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1015, i32 noundef 3, i32 noundef %973) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1040 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1020, i32 %75, i32 2)
  %1041 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1022, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1042 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %483, i32 %85, i32 0)
  %1043 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %483, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1044 = load <4 x i32>, ptr addrspace(3) %198, align 16, !tbaa !11
  %1045 = load <4 x i32>, ptr addrspace(3) %200, align 16, !tbaa !11
  %1046 = load <4 x i32>, ptr addrspace(3) %202, align 16, !tbaa !11
  %1047 = load <4 x i32>, ptr addrspace(3) %204, align 16, !tbaa !11
  %1048 = or disjoint i32 %134, 6400
  %1049 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %1048
  %1050 = load i32, ptr addrspace(3) %1049, align 4, !tbaa !7
  tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) readonly %35, ptr addrspace(3) noundef %117, i32 noundef 16, i32 noundef %116, i32 noundef 3456, i32 noundef 0, i32 noundef 0) #12
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1051 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1044, <4 x i32> noundef %986, <4 x float> noundef %1018, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1050, i32 noundef 0, i32 noundef %1007) #12
  %1052 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1045, <4 x i32> noundef %986, <4 x float> noundef %1019, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1050, i32 noundef 0, i32 noundef %1007) #12
  %1053 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1046, <4 x i32> noundef %988, <4 x float> noundef %1051, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1050, i32 noundef 2, i32 noundef %1007) #12
  %1054 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1047, <4 x i32> noundef %988, <4 x float> noundef %1052, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1050, i32 noundef 2, i32 noundef %1007) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1055 = or disjoint i32 %122, 55296
  %1056 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1055, i32 %63, i32 2)
  %1057 = or disjoint i32 %122, 56320
  %1058 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1057, i32 %63, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1059 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1044, <4 x i32> noundef %993, <4 x float> noundef %1026, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1050, i32 noundef 1, i32 noundef %1007) #12
  %1060 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1045, <4 x i32> noundef %993, <4 x float> noundef %1027, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1050, i32 noundef 1, i32 noundef %1007) #12
  %1061 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1046, <4 x i32> noundef %994, <4 x float> noundef %1059, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1050, i32 noundef 3, i32 noundef %1007) #12
  %1062 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1047, <4 x i32> noundef %994, <4 x float> noundef %1060, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1050, i32 noundef 3, i32 noundef %1007) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1063 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1055, i32 %67, i32 2)
  %1064 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1057, i32 %67, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1065 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1044, <4 x i32> noundef %999, <4 x float> noundef %1032, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1050, i32 noundef 0, i32 noundef %1008) #12
  %1066 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1045, <4 x i32> noundef %999, <4 x float> noundef %1033, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1050, i32 noundef 0, i32 noundef %1008) #12
  %1067 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1046, <4 x i32> noundef %1000, <4 x float> noundef %1065, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1050, i32 noundef 2, i32 noundef %1008) #12
  %1068 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1047, <4 x i32> noundef %1000, <4 x float> noundef %1066, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1050, i32 noundef 2, i32 noundef %1008) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1069 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1055, i32 %71, i32 2)
  %1070 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1057, i32 %71, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  tail call void @llvm.amdgcn.s.setprio(i16 1)
  %1071 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1044, <4 x i32> noundef %1005, <4 x float> noundef %1038, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1050, i32 noundef 1, i32 noundef %1008) #12
  %1072 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1045, <4 x i32> noundef %1005, <4 x float> noundef %1039, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1050, i32 noundef 1, i32 noundef %1008) #12
  %1073 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1046, <4 x i32> noundef %1006, <4 x float> noundef %1071, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1050, i32 noundef 3, i32 noundef %1008) #12
  %1074 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1047, <4 x i32> noundef %1006, <4 x float> noundef %1072, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1050, i32 noundef 3, i32 noundef %1008) #12
  tail call void @llvm.amdgcn.s.setprio(i16 0)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1075 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1055, i32 %75, i32 2)
  %1076 = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly %22, i32 %1057, i32 %75, i32 2)
  tail call void @llvm.amdgcn.sched.barrier(i32 0)
  %1077 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %518, i32 %85, i32 0)
  %1078 = tail call noundef i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly %23, i32 %518, i32 %90, i32 0)
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1079 = load <4 x i32>, ptr addrspace(3) %237, align 16, !tbaa !11
  %1080 = load <4 x i32>, ptr addrspace(3) %239, align 16, !tbaa !11
  %1081 = load <4 x i32>, ptr addrspace(3) %241, align 16, !tbaa !11
  %1082 = load <4 x i32>, ptr addrspace(3) %243, align 16, !tbaa !11
  %1083 = or disjoint i32 %134, 6656
  %1084 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %1083
  %1085 = load i32, ptr addrspace(3) %1084, align 4, !tbaa !7
  %1086 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1079, <4 x i32> noundef %1021, <4 x float> noundef %1053, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1085, i32 noundef 0, i32 noundef %1042) #12
  %1087 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1080, <4 x i32> noundef %1021, <4 x float> noundef %1054, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1085, i32 noundef 0, i32 noundef %1042) #12
  %1088 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1081, <4 x i32> noundef %1023, <4 x float> noundef %1086, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1085, i32 noundef 2, i32 noundef %1042) #12
  %1089 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1082, <4 x i32> noundef %1023, <4 x float> noundef %1087, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1085, i32 noundef 2, i32 noundef %1042) #12
  %1090 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1079, <4 x i32> noundef %1028, <4 x float> noundef %1061, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1085, i32 noundef 1, i32 noundef %1042) #12
  %1091 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1080, <4 x i32> noundef %1028, <4 x float> noundef %1062, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1085, i32 noundef 1, i32 noundef %1042) #12
  %1092 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1081, <4 x i32> noundef %1029, <4 x float> noundef %1090, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1085, i32 noundef 3, i32 noundef %1042) #12
  %1093 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1082, <4 x i32> noundef %1029, <4 x float> noundef %1091, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1085, i32 noundef 3, i32 noundef %1042) #12
  %1094 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1079, <4 x i32> noundef %1034, <4 x float> noundef %1067, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1085, i32 noundef 0, i32 noundef %1043) #12
  %1095 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1080, <4 x i32> noundef %1034, <4 x float> noundef %1068, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1085, i32 noundef 0, i32 noundef %1043) #12
  %1096 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1081, <4 x i32> noundef %1035, <4 x float> noundef %1094, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1085, i32 noundef 2, i32 noundef %1043) #12
  %1097 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1082, <4 x i32> noundef %1035, <4 x float> noundef %1095, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1085, i32 noundef 2, i32 noundef %1043) #12
  %1098 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1079, <4 x i32> noundef %1040, <4 x float> noundef %1073, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1085, i32 noundef 1, i32 noundef %1043) #12
  %1099 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1080, <4 x i32> noundef %1040, <4 x float> noundef %1074, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1085, i32 noundef 1, i32 noundef %1043) #12
  %1100 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1081, <4 x i32> noundef %1041, <4 x float> noundef %1098, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1085, i32 noundef 3, i32 noundef %1043) #12
  %1101 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1082, <4 x i32> noundef %1041, <4 x float> noundef %1099, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1085, i32 noundef 3, i32 noundef %1043) #12
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %1102 = load <4 x i32>, ptr addrspace(3) %155, align 16, !tbaa !11
  %1103 = load <4 x i32>, ptr addrspace(3) %158, align 16, !tbaa !11
  %1104 = load <4 x i32>, ptr addrspace(3) %162, align 16, !tbaa !11
  %1105 = load <4 x i32>, ptr addrspace(3) %164, align 16, !tbaa !11
  %1106 = or disjoint i32 %134, 6912
  %1107 = getelementptr inbounds nuw [7168 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds, i32 12288), i32 0, i32 %1106
  %1108 = load i32, ptr addrspace(3) %1107, align 4, !tbaa !7
  %1109 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1102, <4 x i32> noundef %1056, <4 x float> noundef %1088, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1108, i32 noundef 0, i32 noundef %1077) #12
  %1110 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1103, <4 x i32> noundef %1056, <4 x float> noundef %1089, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1108, i32 noundef 0, i32 noundef %1077) #12
  %1111 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1104, <4 x i32> noundef %1058, <4 x float> noundef %1109, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1108, i32 noundef 2, i32 noundef %1077) #12
  %1112 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1105, <4 x i32> noundef %1058, <4 x float> noundef %1110, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1108, i32 noundef 2, i32 noundef %1077) #12
  %1113 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1102, <4 x i32> noundef %1063, <4 x float> noundef %1092, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1108, i32 noundef 1, i32 noundef %1077) #12
  %1114 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1103, <4 x i32> noundef %1063, <4 x float> noundef %1093, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1108, i32 noundef 1, i32 noundef %1077) #12
  %1115 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1104, <4 x i32> noundef %1064, <4 x float> noundef %1113, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1108, i32 noundef 3, i32 noundef %1077) #12
  %1116 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1105, <4 x i32> noundef %1064, <4 x float> noundef %1114, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1108, i32 noundef 3, i32 noundef %1077) #12
  %1117 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1102, <4 x i32> noundef %1069, <4 x float> noundef %1096, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1108, i32 noundef 0, i32 noundef %1078) #12
  %1118 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1103, <4 x i32> noundef %1069, <4 x float> noundef %1097, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1108, i32 noundef 0, i32 noundef %1078) #12
  %1119 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1104, <4 x i32> noundef %1070, <4 x float> noundef %1117, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1108, i32 noundef 2, i32 noundef %1078) #12
  %1120 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1105, <4 x i32> noundef %1070, <4 x float> noundef %1118, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1108, i32 noundef 2, i32 noundef %1078) #12
  %1121 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1102, <4 x i32> noundef %1075, <4 x float> noundef %1100, i32 noundef 4, i32 noundef 4, i32 noundef 0, i32 noundef %1108, i32 noundef 1, i32 noundef %1078) #12
  %1122 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1103, <4 x i32> noundef %1075, <4 x float> noundef %1101, i32 noundef 4, i32 noundef 4, i32 noundef 1, i32 noundef %1108, i32 noundef 1, i32 noundef %1078) #12
  %1123 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1104, <4 x i32> noundef %1076, <4 x float> noundef %1121, i32 noundef 4, i32 noundef 4, i32 noundef 2, i32 noundef %1108, i32 noundef 3, i32 noundef %1078) #12
  %1124 = tail call contract <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32> noundef %1105, <4 x i32> noundef %1076, <4 x float> noundef %1122, i32 noundef 4, i32 noundef 4, i32 noundef 3, i32 noundef %1108, i32 noundef 3, i32 noundef %1078) #12
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  tail call void @llvm.experimental.noalias.scope.decl(metadata !12)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  %1125 = shl nuw nsw i32 %118, 10
  %1126 = shl nuw nsw i32 %37, 5
  %1127 = or disjoint i32 %1126, %120
  %1128 = add nuw i32 %1127, 128
  %1129 = extractelement <4 x float> %1111, i64 0
  %1130 = add nuw nsw i32 %1127, %1125
  %1131 = zext nneg i32 %1130 to i64
  %1132 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1131
  %1133 = addrspacecast ptr %1132 to ptr addrspace(3)
  store float %1129, ptr addrspace(3) %1133, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1134 = extractelement <4 x float> %1111, i64 1
  %1135 = or disjoint i32 %1125, 256
  %1136 = add nuw nsw i32 %1127, %1135
  %1137 = zext nneg i32 %1136 to i64
  %1138 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1137
  %1139 = addrspacecast ptr %1138 to ptr addrspace(3)
  store float %1134, ptr addrspace(3) %1139, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1140 = extractelement <4 x float> %1111, i64 2
  %1141 = or disjoint i32 %1125, 512
  %1142 = add nuw nsw i32 %1127, %1141
  %1143 = zext nneg i32 %1142 to i64
  %1144 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1143
  %1145 = addrspacecast ptr %1144 to ptr addrspace(3)
  store float %1140, ptr addrspace(3) %1145, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1146 = extractelement <4 x float> %1111, i64 3
  %1147 = or disjoint i32 %1125, 768
  %1148 = add nuw nsw i32 %1127, %1147
  %1149 = zext nneg i32 %1148 to i64
  %1150 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1149
  %1151 = addrspacecast ptr %1150 to ptr addrspace(3)
  store float %1146, ptr addrspace(3) %1151, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1152 = extractelement <4 x float> %1115, i64 0
  %1153 = add nuw nsw i32 %1128, %1125
  %1154 = sext i32 %1153 to i64
  %1155 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1154
  %1156 = addrspacecast ptr %1155 to ptr addrspace(3)
  store float %1152, ptr addrspace(3) %1156, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1157 = extractelement <4 x float> %1115, i64 1
  %1158 = add nuw nsw i32 %1128, %1135
  %1159 = sext i32 %1158 to i64
  %1160 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1159
  %1161 = addrspacecast ptr %1160 to ptr addrspace(3)
  store float %1157, ptr addrspace(3) %1161, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1162 = extractelement <4 x float> %1115, i64 2
  %1163 = add nuw nsw i32 %1128, %1141
  %1164 = sext i32 %1163 to i64
  %1165 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1164
  %1166 = addrspacecast ptr %1165 to ptr addrspace(3)
  store float %1162, ptr addrspace(3) %1166, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1167 = extractelement <4 x float> %1115, i64 3
  %1168 = add nuw nsw i32 %1128, %1147
  %1169 = sext i32 %1168 to i64
  %1170 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1169
  %1171 = addrspacecast ptr %1170 to ptr addrspace(3)
  store float %1167, ptr addrspace(3) %1171, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1172 = or disjoint i32 %1127, 16
  %1173 = extractelement <4 x float> %1119, i64 0
  %1174 = add nuw nsw i32 %1172, %1125
  %1175 = zext nneg i32 %1174 to i64
  %1176 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1175
  %1177 = addrspacecast ptr %1176 to ptr addrspace(3)
  store float %1173, ptr addrspace(3) %1177, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1178 = extractelement <4 x float> %1119, i64 1
  %1179 = add nuw nsw i32 %1172, %1135
  %1180 = zext nneg i32 %1179 to i64
  %1181 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1180
  %1182 = addrspacecast ptr %1181 to ptr addrspace(3)
  store float %1178, ptr addrspace(3) %1182, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1183 = extractelement <4 x float> %1119, i64 2
  %1184 = add nuw nsw i32 %1172, %1141
  %1185 = zext nneg i32 %1184 to i64
  %1186 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1185
  %1187 = addrspacecast ptr %1186 to ptr addrspace(3)
  store float %1183, ptr addrspace(3) %1187, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1188 = extractelement <4 x float> %1119, i64 3
  %1189 = add nuw nsw i32 %1172, %1147
  %1190 = zext nneg i32 %1189 to i64
  %1191 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1190
  %1192 = addrspacecast ptr %1191 to ptr addrspace(3)
  store float %1188, ptr addrspace(3) %1192, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1193 = add nuw i32 %1127, 144
  %1194 = extractelement <4 x float> %1123, i64 0
  %1195 = add nuw nsw i32 %1193, %1125
  %1196 = sext i32 %1195 to i64
  %1197 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1196
  %1198 = addrspacecast ptr %1197 to ptr addrspace(3)
  store float %1194, ptr addrspace(3) %1198, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1199 = extractelement <4 x float> %1123, i64 1
  %1200 = add nuw nsw i32 %1193, %1135
  %1201 = sext i32 %1200 to i64
  %1202 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1201
  %1203 = addrspacecast ptr %1202 to ptr addrspace(3)
  store float %1199, ptr addrspace(3) %1203, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1204 = extractelement <4 x float> %1123, i64 2
  %1205 = add nuw nsw i32 %1193, %1141
  %1206 = sext i32 %1205 to i64
  %1207 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1206
  %1208 = addrspacecast ptr %1207 to ptr addrspace(3)
  store float %1204, ptr addrspace(3) %1208, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1209 = extractelement <4 x float> %1123, i64 3
  %1210 = add nuw nsw i32 %1193, %1147
  %1211 = sext i32 %1210 to i64
  %1212 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1211
  %1213 = addrspacecast ptr %1212 to ptr addrspace(3)
  store float %1209, ptr addrspace(3) %1213, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1214 = extractelement <4 x float> %1112, i64 0
  %1215 = or disjoint i32 %1125, 4096
  %1216 = add nuw nsw i32 %1127, %1215
  %1217 = zext nneg i32 %1216 to i64
  %1218 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1217
  %1219 = addrspacecast ptr %1218 to ptr addrspace(3)
  store float %1214, ptr addrspace(3) %1219, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1220 = extractelement <4 x float> %1112, i64 1
  %1221 = or disjoint i32 %1125, 4352
  %1222 = add nuw nsw i32 %1127, %1221
  %1223 = zext nneg i32 %1222 to i64
  %1224 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1223
  %1225 = addrspacecast ptr %1224 to ptr addrspace(3)
  store float %1220, ptr addrspace(3) %1225, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1226 = extractelement <4 x float> %1112, i64 2
  %1227 = or disjoint i32 %1125, 4608
  %1228 = add nuw nsw i32 %1127, %1227
  %1229 = zext nneg i32 %1228 to i64
  %1230 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1229
  %1231 = addrspacecast ptr %1230 to ptr addrspace(3)
  store float %1226, ptr addrspace(3) %1231, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1232 = extractelement <4 x float> %1112, i64 3
  %1233 = or disjoint i32 %1125, 4864
  %1234 = add nuw nsw i32 %1127, %1233
  %1235 = zext nneg i32 %1234 to i64
  %1236 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1235
  %1237 = addrspacecast ptr %1236 to ptr addrspace(3)
  store float %1232, ptr addrspace(3) %1237, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1238 = extractelement <4 x float> %1116, i64 0
  %1239 = add nuw nsw i32 %1128, %1215
  %1240 = sext i32 %1239 to i64
  %1241 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1240
  %1242 = addrspacecast ptr %1241 to ptr addrspace(3)
  store float %1238, ptr addrspace(3) %1242, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1243 = extractelement <4 x float> %1116, i64 1
  %1244 = add nuw nsw i32 %1128, %1221
  %1245 = sext i32 %1244 to i64
  %1246 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1245
  %1247 = addrspacecast ptr %1246 to ptr addrspace(3)
  store float %1243, ptr addrspace(3) %1247, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1248 = extractelement <4 x float> %1116, i64 2
  %1249 = add nuw nsw i32 %1128, %1227
  %1250 = sext i32 %1249 to i64
  %1251 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1250
  %1252 = addrspacecast ptr %1251 to ptr addrspace(3)
  store float %1248, ptr addrspace(3) %1252, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1253 = extractelement <4 x float> %1116, i64 3
  %1254 = add nuw nsw i32 %1128, %1233
  %1255 = sext i32 %1254 to i64
  %1256 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1255
  %1257 = addrspacecast ptr %1256 to ptr addrspace(3)
  store float %1253, ptr addrspace(3) %1257, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1258 = extractelement <4 x float> %1120, i64 0
  %1259 = add nuw nsw i32 %1172, %1215
  %1260 = zext nneg i32 %1259 to i64
  %1261 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1260
  %1262 = addrspacecast ptr %1261 to ptr addrspace(3)
  store float %1258, ptr addrspace(3) %1262, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1263 = extractelement <4 x float> %1120, i64 1
  %1264 = add nuw nsw i32 %1172, %1221
  %1265 = zext nneg i32 %1264 to i64
  %1266 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1265
  %1267 = addrspacecast ptr %1266 to ptr addrspace(3)
  store float %1263, ptr addrspace(3) %1267, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1268 = extractelement <4 x float> %1120, i64 2
  %1269 = add nuw nsw i32 %1172, %1227
  %1270 = zext nneg i32 %1269 to i64
  %1271 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1270
  %1272 = addrspacecast ptr %1271 to ptr addrspace(3)
  store float %1268, ptr addrspace(3) %1272, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1273 = extractelement <4 x float> %1120, i64 3
  %1274 = add nuw nsw i32 %1172, %1233
  %1275 = zext nneg i32 %1274 to i64
  %1276 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1275
  %1277 = addrspacecast ptr %1276 to ptr addrspace(3)
  store float %1273, ptr addrspace(3) %1277, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1278 = extractelement <4 x float> %1124, i64 0
  %1279 = add nuw nsw i32 %1193, %1215
  %1280 = sext i32 %1279 to i64
  %1281 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1280
  %1282 = addrspacecast ptr %1281 to ptr addrspace(3)
  store float %1278, ptr addrspace(3) %1282, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1283 = extractelement <4 x float> %1124, i64 1
  %1284 = add nuw nsw i32 %1193, %1221
  %1285 = sext i32 %1284 to i64
  %1286 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1285
  %1287 = addrspacecast ptr %1286 to ptr addrspace(3)
  store float %1283, ptr addrspace(3) %1287, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1288 = extractelement <4 x float> %1124, i64 2
  %1289 = add nuw nsw i32 %1193, %1227
  %1290 = sext i32 %1289 to i64
  %1291 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1290
  %1292 = addrspacecast ptr %1291 to ptr addrspace(3)
  store float %1288, ptr addrspace(3) %1292, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1293 = extractelement <4 x float> %1124, i64 3
  %1294 = add nuw nsw i32 %1193, %1233
  %1295 = sext i32 %1294 to i64
  %1296 = getelementptr inbounds float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1295
  %1297 = addrspacecast ptr %1296 to ptr addrspace(3)
  store float %1293, ptr addrspace(3) %1297, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  fence syncscope("workgroup") release
  tail call void @llvm.amdgcn.s.barrier(), !noalias !21
  fence syncscope("workgroup") acquire
  %1298 = lshr i32 %19, 4
  %1299 = lshr i32 %19, 2
  %1300 = and i32 %1299, 3
  %1301 = and i32 %19, 3
  %1302 = shl nuw nsw i32 %1301, 3
  %1303 = shl nuw nsw i32 %1300, 5
  %1304 = or disjoint i32 %1303, %1302
  %1305 = or disjoint i32 %1304, 128
  %1306 = shl nsw i32 %44, 6
  %1307 = shl nuw nsw i32 %1300, 4
  %1308 = shl nuw nsw i32 %1301, 2
  %1309 = or disjoint i32 %48, %1298
  %1310 = shl nuw nsw i32 %1298, 8
  %1311 = or disjoint i32 %1304, %1310
  %1312 = or disjoint i32 %1305, %1310
  %1313 = zext nneg i32 %1311 to i64
  %1314 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1313
  %1315 = zext nneg i32 %1312 to i64
  %1316 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1315
  %1317 = or disjoint i32 %1311, 2
  %1318 = zext nneg i32 %1317 to i64
  %1319 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1318
  %1320 = or disjoint i32 %1312, 2
  %1321 = zext nneg i32 %1320 to i64
  %1322 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1321
  %1323 = or disjoint i32 %1311, 4
  %1324 = zext nneg i32 %1323 to i64
  %1325 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1324
  %1326 = addrspacecast ptr %1325 to ptr addrspace(3)
  %1327 = load float, ptr addrspace(3) %1326, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1328 = or disjoint i32 %1312, 4
  %1329 = zext nneg i32 %1328 to i64
  %1330 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1329
  %1331 = addrspacecast ptr %1330 to ptr addrspace(3)
  %1332 = load float, ptr addrspace(3) %1331, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1333 = or disjoint i32 %1311, 5
  %1334 = zext nneg i32 %1333 to i64
  %1335 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1334
  %1336 = or disjoint i32 %1312, 5
  %1337 = zext nneg i32 %1336 to i64
  %1338 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1337
  %1339 = or disjoint i32 %1311, 7
  %1340 = zext nneg i32 %1339 to i64
  %1341 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1340
  %1342 = addrspacecast ptr %1341 to ptr addrspace(3)
  %1343 = load float, ptr addrspace(3) %1342, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1344 = or disjoint i32 %1312, 7
  %1345 = zext nneg i32 %1344 to i64
  %1346 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1345
  %1347 = addrspacecast ptr %1346 to ptr addrspace(3)
  %1348 = load float, ptr addrspace(3) %1347, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1349 = fmul contract float %1327, 0xBFF7154760000000
  %1350 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1349)
  %1351 = fadd contract float %1350, 1.000000e+00
  %1352 = tail call contract float @llvm.amdgcn.rcp.f32(float %1351)
  %1353 = fmul contract float %1343, 0xBFF7154760000000
  %1354 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1353)
  %1355 = fadd contract float %1354, 1.000000e+00
  %1356 = tail call contract float @llvm.amdgcn.rcp.f32(float %1355)
  %1357 = shl nsw i32 %1309, 8
  %1358 = or disjoint i32 %1308, %1306
  %1359 = or disjoint i32 %1358, %1307
  %1360 = add i32 %1359, %1357
  %1361 = sext i32 %1360 to i64
  %1362 = getelementptr inbounds i8, ptr %39, i64 %1361
  %1363 = or disjoint i32 %1310, 4096
  %1364 = or disjoint i32 %1304, %1363
  %1365 = or disjoint i32 %1305, %1363
  %1366 = zext nneg i32 %1364 to i64
  %1367 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1366
  %1368 = zext nneg i32 %1365 to i64
  %1369 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1368
  %1370 = or disjoint i32 %1364, 2
  %1371 = zext nneg i32 %1370 to i64
  %1372 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1371
  %1373 = or disjoint i32 %1365, 2
  %1374 = zext nneg i32 %1373 to i64
  %1375 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1374
  %1376 = or disjoint i32 %1364, 4
  %1377 = zext nneg i32 %1376 to i64
  %1378 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1377
  %1379 = addrspacecast ptr %1378 to ptr addrspace(3)
  %1380 = load float, ptr addrspace(3) %1379, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1381 = or disjoint i32 %1365, 4
  %1382 = zext nneg i32 %1381 to i64
  %1383 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1382
  %1384 = addrspacecast ptr %1383 to ptr addrspace(3)
  %1385 = load float, ptr addrspace(3) %1384, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1386 = or disjoint i32 %1364, 5
  %1387 = zext nneg i32 %1386 to i64
  %1388 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1387
  %1389 = or disjoint i32 %1365, 5
  %1390 = zext nneg i32 %1389 to i64
  %1391 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1390
  %1392 = or disjoint i32 %1364, 7
  %1393 = zext nneg i32 %1392 to i64
  %1394 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1393
  %1395 = addrspacecast ptr %1394 to ptr addrspace(3)
  %1396 = load float, ptr addrspace(3) %1395, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1397 = or disjoint i32 %1365, 7
  %1398 = zext nneg i32 %1397 to i64
  %1399 = getelementptr inbounds nuw float, ptr addrspacecast (ptr addrspace(3) @_ZZN5aiter9mxfp4_moe5gemm16kernelILi655360ELi385ELi7168ELi1024ELi32ELb1ELb0ELi0EEEvPKhPKaS4_S6_PKiS8_S8_iPhS9_PK14__hip_bfloat16E3lds to ptr), i64 %1398
  %1400 = addrspacecast ptr %1399 to ptr addrspace(3)
  %1401 = load float, ptr addrspace(3) %1400, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1402 = fmul contract float %1380, 0xBFF7154760000000
  %1403 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1402)
  %1404 = fadd contract float %1403, 1.000000e+00
  %1405 = tail call contract float @llvm.amdgcn.rcp.f32(float %1404)
  %1406 = fmul contract float %1396, 0xBFF7154760000000
  %1407 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1406)
  %1408 = fadd contract float %1407, 1.000000e+00
  %1409 = tail call contract float @llvm.amdgcn.rcp.f32(float %1408)
  %1410 = addrspacecast ptr %1314 to ptr addrspace(3)
  %1411 = load <2 x float>, ptr addrspace(3) %1410, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1412 = addrspacecast ptr %1316 to ptr addrspace(3)
  %1413 = load <2 x float>, ptr addrspace(3) %1412, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1414 = addrspacecast ptr %1319 to ptr addrspace(3)
  %1415 = load <2 x float>, ptr addrspace(3) %1414, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1416 = addrspacecast ptr %1322 to ptr addrspace(3)
  %1417 = load <2 x float>, ptr addrspace(3) %1416, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1418 = addrspacecast ptr %1335 to ptr addrspace(3)
  %1419 = load <2 x float>, ptr addrspace(3) %1418, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1420 = addrspacecast ptr %1338 to ptr addrspace(3)
  %1421 = load <2 x float>, ptr addrspace(3) %1420, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1422 = extractelement <2 x float> %1411, i64 0
  %1423 = fmul contract float %1422, 0xBFF7154760000000
  %1424 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1423)
  %1425 = fadd contract float %1424, 1.000000e+00
  %1426 = tail call contract float @llvm.amdgcn.rcp.f32(float %1425)
  %1427 = extractelement <2 x float> %1411, i64 1
  %1428 = fmul contract float %1427, 0xBFF7154760000000
  %1429 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1428)
  %1430 = fadd contract float %1429, 1.000000e+00
  %1431 = tail call contract float @llvm.amdgcn.rcp.f32(float %1430)
  %1432 = extractelement <2 x float> %1415, i64 0
  %1433 = fmul contract float %1432, 0xBFF7154760000000
  %1434 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1433)
  %1435 = fadd contract float %1434, 1.000000e+00
  %1436 = tail call contract float @llvm.amdgcn.rcp.f32(float %1435)
  %1437 = extractelement <2 x float> %1415, i64 1
  %1438 = fmul contract float %1437, 0xBFF7154760000000
  %1439 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1438)
  %1440 = fadd contract float %1439, 1.000000e+00
  %1441 = tail call contract float @llvm.amdgcn.rcp.f32(float %1440)
  %1442 = extractelement <2 x float> %1419, i64 0
  %1443 = fmul contract float %1442, 0xBFF7154760000000
  %1444 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1443)
  %1445 = fadd contract float %1444, 1.000000e+00
  %1446 = tail call contract float @llvm.amdgcn.rcp.f32(float %1445)
  %1447 = extractelement <2 x float> %1419, i64 1
  %1448 = fmul contract float %1447, 0xBFF7154760000000
  %1449 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1448)
  %1450 = fadd contract float %1449, 1.000000e+00
  %1451 = tail call contract float @llvm.amdgcn.rcp.f32(float %1450)
  %1452 = addrspacecast ptr %1367 to ptr addrspace(3)
  %1453 = load <2 x float>, ptr addrspace(3) %1452, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1454 = addrspacecast ptr %1369 to ptr addrspace(3)
  %1455 = load <2 x float>, ptr addrspace(3) %1454, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1456 = addrspacecast ptr %1372 to ptr addrspace(3)
  %1457 = load <2 x float>, ptr addrspace(3) %1456, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1458 = addrspacecast ptr %1375 to ptr addrspace(3)
  %1459 = load <2 x float>, ptr addrspace(3) %1458, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1460 = addrspacecast ptr %1388 to ptr addrspace(3)
  %1461 = load <2 x float>, ptr addrspace(3) %1460, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1462 = addrspacecast ptr %1391 to ptr addrspace(3)
  %1463 = load <2 x float>, ptr addrspace(3) %1462, align 4, !tbaa !19, !alias.scope !17, !noalias !21
  %1464 = extractelement <2 x float> %1453, i64 0
  %1465 = fmul contract float %1464, 0xBFF7154760000000
  %1466 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1465)
  %1467 = fadd contract float %1466, 1.000000e+00
  %1468 = tail call contract float @llvm.amdgcn.rcp.f32(float %1467)
  %1469 = shufflevector <2 x float> %1411, <2 x float> %1453, <2 x i32> <i32 0, i32 2>
  %1470 = insertelement <2 x float> poison, float %1426, i64 0
  %1471 = insertelement <2 x float> %1470, float %1468, i64 1
  %1472 = fmul contract <2 x float> %1469, %1471
  %1473 = shufflevector <2 x float> %1413, <2 x float> %1455, <2 x i32> <i32 0, i32 2>
  %1474 = fmul contract <2 x float> %1473, %1472
  %1475 = extractelement <2 x float> %1453, i64 1
  %1476 = fmul contract float %1475, 0xBFF7154760000000
  %1477 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1476)
  %1478 = fadd contract float %1477, 1.000000e+00
  %1479 = tail call contract float @llvm.amdgcn.rcp.f32(float %1478)
  %1480 = shufflevector <2 x float> %1411, <2 x float> %1453, <2 x i32> <i32 1, i32 3>
  %1481 = insertelement <2 x float> poison, float %1431, i64 0
  %1482 = insertelement <2 x float> %1481, float %1479, i64 1
  %1483 = fmul contract <2 x float> %1480, %1482
  %1484 = shufflevector <2 x float> %1413, <2 x float> %1455, <2 x i32> <i32 1, i32 3>
  %1485 = fmul contract <2 x float> %1484, %1483
  %1486 = extractelement <2 x float> %1457, i64 0
  %1487 = fmul contract float %1486, 0xBFF7154760000000
  %1488 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1487)
  %1489 = fadd contract float %1488, 1.000000e+00
  %1490 = tail call contract float @llvm.amdgcn.rcp.f32(float %1489)
  %1491 = shufflevector <2 x float> %1415, <2 x float> %1457, <2 x i32> <i32 0, i32 2>
  %1492 = insertelement <2 x float> poison, float %1436, i64 0
  %1493 = insertelement <2 x float> %1492, float %1490, i64 1
  %1494 = fmul contract <2 x float> %1491, %1493
  %1495 = shufflevector <2 x float> %1417, <2 x float> %1459, <2 x i32> <i32 0, i32 2>
  %1496 = fmul contract <2 x float> %1495, %1494
  %1497 = extractelement <2 x float> %1457, i64 1
  %1498 = fmul contract float %1497, 0xBFF7154760000000
  %1499 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1498)
  %1500 = fadd contract float %1499, 1.000000e+00
  %1501 = tail call contract float @llvm.amdgcn.rcp.f32(float %1500)
  %1502 = shufflevector <2 x float> %1415, <2 x float> %1457, <2 x i32> <i32 1, i32 3>
  %1503 = insertelement <2 x float> poison, float %1441, i64 0
  %1504 = insertelement <2 x float> %1503, float %1501, i64 1
  %1505 = fmul contract <2 x float> %1502, %1504
  %1506 = shufflevector <2 x float> %1417, <2 x float> %1459, <2 x i32> <i32 1, i32 3>
  %1507 = fmul contract <2 x float> %1506, %1505
  %1508 = insertelement <2 x float> poison, float %1327, i64 0
  %1509 = insertelement <2 x float> %1508, float %1380, i64 1
  %1510 = insertelement <2 x float> poison, float %1352, i64 0
  %1511 = insertelement <2 x float> %1510, float %1405, i64 1
  %1512 = fmul contract <2 x float> %1509, %1511
  %1513 = insertelement <2 x float> poison, float %1332, i64 0
  %1514 = insertelement <2 x float> %1513, float %1385, i64 1
  %1515 = fmul contract <2 x float> %1514, %1512
  %1516 = extractelement <2 x float> %1461, i64 0
  %1517 = fmul contract float %1516, 0xBFF7154760000000
  %1518 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1517)
  %1519 = fadd contract float %1518, 1.000000e+00
  %1520 = tail call contract float @llvm.amdgcn.rcp.f32(float %1519)
  %1521 = shufflevector <2 x float> %1419, <2 x float> %1461, <2 x i32> <i32 0, i32 2>
  %1522 = insertelement <2 x float> poison, float %1446, i64 0
  %1523 = insertelement <2 x float> %1522, float %1520, i64 1
  %1524 = fmul contract <2 x float> %1521, %1523
  %1525 = shufflevector <2 x float> %1421, <2 x float> %1463, <2 x i32> <i32 0, i32 2>
  %1526 = fmul contract <2 x float> %1525, %1524
  %1527 = extractelement <2 x float> %1461, i64 1
  %1528 = fmul contract float %1527, 0xBFF7154760000000
  %1529 = tail call contract noundef float @llvm.amdgcn.exp2.f32(float %1528)
  %1530 = fadd contract float %1529, 1.000000e+00
  %1531 = tail call contract float @llvm.amdgcn.rcp.f32(float %1530)
  %1532 = shufflevector <2 x float> %1419, <2 x float> %1461, <2 x i32> <i32 1, i32 3>
  %1533 = insertelement <2 x float> poison, float %1451, i64 0
  %1534 = insertelement <2 x float> %1533, float %1531, i64 1
  %1535 = fmul contract <2 x float> %1532, %1534
  %1536 = shufflevector <2 x float> %1421, <2 x float> %1463, <2 x i32> <i32 1, i32 3>
  %1537 = fmul contract <2 x float> %1536, %1535
  %1538 = tail call contract <2 x float> @llvm.fabs.v2f32(<2 x float> %1474)
  %1539 = tail call contract <2 x float> @llvm.fabs.v2f32(<2 x float> %1485)
  %1540 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1538, <2 x float> %1539)
  %1541 = tail call contract <2 x float> @llvm.fabs.v2f32(<2 x float> %1496)
  %1542 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1540, <2 x float> %1541)
  %1543 = tail call contract <2 x float> @llvm.fabs.v2f32(<2 x float> %1507)
  %1544 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1542, <2 x float> %1543)
  %1545 = tail call contract <2 x float> @llvm.fabs.v2f32(<2 x float> %1515)
  %1546 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1544, <2 x float> %1545)
  %1547 = tail call contract <2 x float> @llvm.fabs.v2f32(<2 x float> %1526)
  %1548 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1546, <2 x float> %1547)
  %1549 = tail call contract <2 x float> @llvm.fabs.v2f32(<2 x float> %1537)
  %1550 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1548, <2 x float> %1549)
  %1551 = insertelement <2 x float> poison, float %1343, i64 0
  %1552 = insertelement <2 x float> %1551, float %1396, i64 1
  %1553 = insertelement <2 x float> poison, float %1356, i64 0
  %1554 = insertelement <2 x float> %1553, float %1409, i64 1
  %1555 = fmul contract <2 x float> %1552, %1554
  %1556 = insertelement <2 x float> poison, float %1348, i64 0
  %1557 = insertelement <2 x float> %1556, float %1401, i64 1
  %1558 = fmul contract <2 x float> %1557, %1555
  %1559 = tail call contract <2 x float> @llvm.fabs.v2f32(<2 x float> %1558)
  %1560 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1550, <2 x float> %1559)
  %1561 = bitcast <2 x float> %1560 to <2 x i32>
  %1562 = extractelement <2 x i32> %1561, i64 0
  %1563 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1562, i32 177, i32 15, i32 15, i1 true)
  %1564 = bitcast <2 x float> %1560 to <2 x i32>
  %1565 = extractelement <2 x i32> %1564, i64 1
  %1566 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1565, i32 177, i32 15, i32 15, i1 true)
  %1567 = insertelement <2 x i32> poison, i32 %1563, i64 0
  %1568 = insertelement <2 x i32> %1567, i32 %1566, i64 1
  %1569 = bitcast <2 x i32> %1568 to <2 x float>
  %1570 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1560, <2 x float> %1569)
  %1571 = bitcast <2 x float> %1570 to <2 x i32>
  %1572 = extractelement <2 x i32> %1571, i64 0
  %1573 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1572, i32 78, i32 15, i32 15, i1 true)
  %1574 = bitcast <2 x float> %1570 to <2 x i32>
  %1575 = extractelement <2 x i32> %1574, i64 1
  %1576 = tail call i32 @llvm.amdgcn.update.dpp.i32(i32 poison, i32 %1575, i32 78, i32 15, i32 15, i1 true)
  %1577 = insertelement <2 x i32> poison, i32 %1573, i64 0
  %1578 = insertelement <2 x i32> %1577, i32 %1576, i64 1
  %1579 = bitcast <2 x i32> %1578 to <2 x float>
  %1580 = tail call contract <2 x float> @llvm.maxnum.v2f32(<2 x float> %1570, <2 x float> %1579)
  %1581 = bitcast <2 x float> %1580 to <2 x i32>
  %1582 = add <2 x i32> %1581, splat (i32 2097152)
  %1583 = bitcast <2 x i32> %1582 to <2 x float>
  %1584 = fmul contract <2 x float> %1583, splat (float 2.500000e-01)
  %1585 = extractelement <2 x float> %1584, i64 0
  %1586 = extractelement <2 x float> %1474, i64 0
  %1587 = extractelement <2 x float> %1485, i64 0
  %1588 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %1586, float %1587, float %1585, i32 0)
  %1589 = extractelement <2 x float> %1496, i64 0
  %1590 = extractelement <2 x float> %1507, i64 0
  %1591 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %1588, float %1589, float %1590, float %1585, i32 1)
  %1592 = extractelement <2 x float> %1515, i64 0
  %1593 = extractelement <2 x float> %1526, i64 0
  %1594 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %1591, float %1592, float %1593, float %1585, i32 2)
  %1595 = extractelement <2 x float> %1558, i64 0
  %1596 = extractelement <2 x float> %1537, i64 0
  %1597 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %1594, float %1596, float %1595, float %1585, i32 3)
  store i32 %1597, ptr %1362, align 4, !tbaa !7, !alias.scope !12, !noalias !22, !nontemporal !23
  %1598 = extractelement <2 x float> %1584, i64 1
  %1599 = extractelement <2 x float> %1474, i64 1
  %1600 = extractelement <2 x float> %1485, i64 1
  %1601 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 0, float %1599, float %1600, float %1598, i32 0)
  %1602 = extractelement <2 x float> %1496, i64 1
  %1603 = extractelement <2 x float> %1507, i64 1
  %1604 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %1601, float %1602, float %1603, float %1598, i32 1)
  %1605 = extractelement <2 x float> %1515, i64 1
  %1606 = extractelement <2 x float> %1526, i64 1
  %1607 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %1604, float %1605, float %1606, float %1598, i32 2)
  %1608 = extractelement <2 x float> %1558, i64 1
  %1609 = extractelement <2 x float> %1537, i64 1
  %1610 = tail call i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32 %1607, float %1609, float %1608, float %1598, i32 3)
  %1611 = add i32 %1360, 4096
  %1612 = sext i32 %1611 to i64
  %1613 = getelementptr inbounds i8, ptr %39, i64 %1612
  store i32 %1610, ptr %1613, align 4, !tbaa !7, !alias.scope !12, !noalias !22, !nontemporal !23
  %1614 = icmp eq i32 %1301, 0
  br i1 %1614, label %1615, label %1633

1615:                                             ; preds = %28
  %1616 = bitcast <2 x float> %1584 to <2 x i32>
  %1617 = lshr <2 x i32> %1616, splat (i32 23)
  %1618 = tail call <2 x i32> @llvm.umin.v2i32(<2 x i32> %1617, <2 x i32> splat (i32 254))
  %1619 = trunc nuw <2 x i32> %1618 to <2 x i8>
  %1620 = shl nsw i32 %42, 7
  %1621 = shl nsw i32 %44, 5
  %1622 = and i32 %1621, 1073741760
  %1623 = add i32 %1622, %1620
  %1624 = or disjoint i32 %1623, %1298
  %1625 = shl nuw nsw i32 %1300, 6
  %1626 = shl i32 %1624, 2
  %1627 = or disjoint i32 %1626, %1625
  %1628 = shl nsw i32 %44, 1
  %1629 = and i32 %1628, 2
  %1630 = or disjoint i32 %1627, %1629
  %1631 = sext i32 %1630 to i64
  %1632 = getelementptr inbounds i8, ptr %17, i64 %1631
  store <2 x i8> %1619, ptr %1632, align 2, !tbaa !24, !alias.scope !15, !noalias !26
  br label %1633

1633:                                             ; preds = %1615, %28, %11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #1

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p0(ptr readnone, i16, i32, i32) #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) nocapture readonly, ptr addrspace(3) nocapture writeonly, i32 immarg, i32, i32, i32 immarg, i32 immarg) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) nocapture readonly, i32, i32, i32 immarg) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.amdgcn.s.setprio(i16 immarg) #7

; Function Attrs: convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v4i32.v4i32(<4 x i32>, <4 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32) #8

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.update.dpp.i32(i32, i32, i32 immarg, i32 immarg, i32 immarg, i1 immarg) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.cvt.scalef32.pk.fp4.f32(i32, float, float, float, i32 immarg) #9

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.rcp.f32(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i32> @llvm.umin.v2i32(<2 x i32>, <2 x i32>) #11

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x float> @llvm.fabs.v2f32(<2 x float>) #11

attributes #0 = { convergent mustprogress nofree norecurse nounwind willreturn "amdgpu-flat-work-group-size"="1,256" "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="2" "denormal-fp-math-f32"="preserve-sign,preserve-sign" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx950" "target-features"="+16-bit-insts,+ashr-pk-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-buffer-pk-add-bf16-inst,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+bf8-cvt-scale-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot12-insts,+dot13-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+f16bf16-to-fp6bf6-cvt-scale-insts,+f32-to-f16bf16-cvt-sr-insts,+fp4-cvt-scale-insts,+fp6bf6-cvt-scale-insts,+fp8-conversion-insts,+fp8-cvt-scale-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+gfx950-insts,+mai-insts,+permlane16-swap,+permlane32-swap,+prng-inst,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="false" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #8 = { convergent mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #9 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #10 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #11 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #12 = { convergent nounwind }

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
!11 = !{!9, !9, i64 0}
!12 = !{!13}
!13 = distinct !{!13, !14, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi32EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 0"}
!14 = distinct !{!14, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi32EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf"}
!15 = !{!16}
!16 = distinct !{!16, !14, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi32EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 1"}
!17 = !{!18}
!18 = distinct !{!18, !14, !"_ZN5aiter9mxfp4_moe11gemm_common27apply_cshuffle_quant_epilogILi1024ELi32EEEvRAdvT0_Li16E_A4_KDv4_fPhS8_iiiiiiiPf: argument 2"}
!19 = !{!20, !20, i64 0}
!20 = !{!"float", !9, i64 0}
!21 = !{!13, !16}
!22 = !{!16, !18}
!23 = !{i32 1}
!24 = !{!25, !25, i64 0}
!25 = !{!"short", !9, i64 0}
!26 = !{!13, !18}
